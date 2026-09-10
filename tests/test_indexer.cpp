// Tests for the corpus indexer: survey, bounded-memory build, and the store
// it produces.
//
// The properties that matter here are the ones that decide whether a thousand
// files work: survey must not decode points, the build must route the bulk of
// the data through disk rather than memory, and nothing may be lost on the way.

#include "e57_fixture.h"
#include "../src/camera.h"
#include "../src/indexer.h"
#include "../src/lod.h"
#include "../src/picker.h"
#include "../src/point_store.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <set>
#include <sys/stat.h>
#include <utime.h>
#include <string>
#include <vector>

static int g_failures = 0;
static int g_checks   = 0;

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        ++g_checks;                                                             \
        if (!(cond)) {                                                          \
            ++g_failures;                                                       \
            std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, (msg));       \
        }                                                                       \
    } while (0)

#define CHECK_NEAR(a, b, tol, msg)                                              \
    do {                                                                        \
        ++g_checks;                                                             \
        double va = (a), vb = (b);                                              \
        if (!(std::fabs(va - vb) <= (tol))) {                                   \
            ++g_failures;                                                       \
            std::printf("  FAIL %s:%d  %s (%.17g vs %.17g)\n",                  \
                        __FILE__, __LINE__, (msg), va, vb);                     \
        }                                                                       \
    } while (0)

static std::string tmpDir() {
    const char* d = std::getenv("E57COV_TMPDIR");
    return d ? d : "/tmp";
}

struct Lcg {
    uint64_t s = 0xDEADBEEFCAFEF00Dull;
    double next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return double((s >> 11) & ((1ull << 53) - 1)) / double(1ull << 53);
    }
};

// Writes a corpus of one-scan files laid out across a site. Each scan holds
// points in its own local frame with the pose that registers it, which is what
// a real registered job looks like.
static std::vector<std::string> writeCorpus(const char* tag, int setups, int perSetup,
                                            double spread, double siteX, double siteY) {
    std::vector<std::string> paths;
    Lcg rng;
    for (int s = 0; s < setups; ++s) {
        const double sx = siteX + (rng.next() * 2.0 - 1.0) * spread;
        const double sy = siteY + (rng.next() * 2.0 - 1.0) * spread;
        const double sz = 1.6;
        const double yaw = rng.next() * 6.28318530718;

        fixture::Scan sc;
        sc.name = std::string("Setup ") + std::to_string(s);
        sc.hasPose = true;
        sc.q[0] = std::cos(yaw * 0.5); sc.q[3] = std::sin(yaw * 0.5);
        sc.t[0] = sx; sc.t[1] = sy; sc.t[2] = sz;
        sc.fields = {
            {"cartesianX", e57::FieldType::FloatDouble},
            {"cartesianY", e57::FieldType::FloatDouble},
            {"cartesianZ", e57::FieldType::FloatDouble},
        };
        sc.data.assign(3, {});
        // Returns in the scanner's own frame, with range a smooth function of
        // direction — one surface per ray, which is what makes a scan a range
        // image. Randomising range per point instead would stack surfaces along
        // the same direction and the structured check would reject it, quite
        // rightly.
        for (int i = 0; i < perSetup; ++i) {
            const double az = rng.next() * 6.28318530718;
            const double el = (rng.next() - 0.5) * 1.2;
            const double r  = 9.0 + 3.0 * std::sin(2.0 * az) + 2.0 * std::cos(3.0 * el);
            const double ce = std::cos(el);
            sc.data[0].push_back(r * ce * std::cos(az));
            sc.data[1].push_back(r * ce * std::sin(az));
            sc.data[2].push_back(r * std::sin(el));
        }
        char buf[256];
        std::snprintf(buf, sizeof(buf), "%s/e57cov_%s_%03d.e57", tmpDir().c_str(), tag, s);
        if (!fixture::write(buf, {sc}, 2048)) return {};
        paths.push_back(buf);
    }
    return paths;
}

// ---------------------------------------------------------------------------

static void testChunkLevel() {
    std::printf("indexer: chunk level selection\n");
    CHECK(indexer::chooseChunkLevel(1000, 20000000) == 1, "a tiny corpus needs one level");
    CHECK(indexer::chooseChunkLevel(100000000, 20000000) >= 1, "a modest corpus needs at least one");
    const uint8_t big = indexer::chooseChunkLevel(40000000000ull, 20000000);
    CHECK(big >= 3, "a 4e10-point corpus needs several levels of chunking");
    CHECK(big <= 6, "and the level stays bounded");
    CHECK(indexer::chooseChunkLevel(1, 0) >= 1, "a zero target does not divide by zero");
}

static void testCellIndex() {
    std::printf("indexer: cell addressing\n");
    lod::Aabb root;
    for (int i = 0; i < 3; ++i) { root.lo[i] = -8; root.hi[i] = 8; }

    lod::Aabb b;
    // Lowest corner is octant 0 all the way down.
    CHECK(indexer::cellIndexOf(root, 2, -7.9f, -7.9f, -7.9f, &b) == 0u, "lowest corner is cell 0");
    CHECK(b.hi[0] <= 0.0f && b.hi[1] <= 0.0f && b.hi[2] <= 0.0f, "and its bounds are the low octant");

    // Highest corner is octant 7 all the way down: 0b111111 == 63.
    CHECK(indexer::cellIndexOf(root, 2, 7.9f, 7.9f, 7.9f, &b) == 63u, "highest corner is cell 63");
    CHECK(b.lo[0] >= 0.0f && b.lo[1] >= 0.0f && b.lo[2] >= 0.0f, "and its bounds are the high octant");

    // A cell's bounds must actually contain the point that selected it.
    Lcg rng;
    bool inside = true;
    for (int i = 0; i < 500; ++i) {
        const float x = float(rng.next() * 15.8 - 7.9);
        const float y = float(rng.next() * 15.8 - 7.9);
        const float z = float(rng.next() * 15.8 - 7.9);
        indexer::cellIndexOf(root, 3, x, y, z, &b);
        if (x < b.lo[0] || x > b.hi[0] || y < b.lo[1] || y > b.hi[1] ||
            z < b.lo[2] || z > b.hi[2]) inside = false;
    }
    CHECK(inside, "every point lies inside the cell it maps to");
}

static void testSurveyIsHeaderOnly() {
    std::printf("indexer: survey reads headers only\n");
    const std::vector<std::string> paths = writeCorpus("survey", 12, 3000, 40.0, 500000.0, 6200000.0);
    CHECK(paths.size() == 12, "corpus written");

    indexer::SurveyOptions so;   // classify off: the fast path
    const indexer::Survey s = indexer::survey(paths, so, nullptr);

    CHECK(s.filesRead == 12, "every file was read");
    CHECK(s.scans.size() == 12, "one scan per file");
    CHECK(s.errors.empty(), "no errors");
    CHECK(s.usableCount() == 12, "all scans usable");
    CHECK(s.totalPoints() == 12 * 3000, "declared point count comes from headers");

    // Setup positions must be available without decoding a single point —
    // this is what lets a thousand files draw a layout in seconds.
    bool poses = true;
    for (const auto& sc : s.scans) {
        if (!sc.hasPose) poses = false;
        if (std::fabs(sc.setup[0] - 500000.0) > 100.0) poses = false;
    }
    CHECK(poses, "every setup position is known from the header");
    CHECK(s.hasBounds, "survey produced site bounds");
    CHECK(s.hi[0] - s.lo[0] > 1.0, "bounds span the site");

    // These fixtures carry no cartesianBounds, so the survey must say so
    // rather than pretending its extent is complete.
    CHECK(!s.extentComplete, "missing declared extents are reported, not assumed");

    indexer::Survey missing = indexer::survey({tmpDir() + "/e57cov_does_not_exist.e57"}, so, nullptr);
    CHECK(missing.errors.size() == 1, "an unreadable file is reported");
    CHECK(missing.scans.empty(), "and contributes no scans");
}

// The survey reads files in parallel and must answer the same either way.
//
// This is the stage the corpus size lands on — one file's check shares nothing
// with another's, so it runs across cores — and the result drives which scans get
// indexed at all. An order that depended on scheduling would mean a scan list,
// and therefore a point store, that differed between runs of the same corpus.
//
// Checked by comparing the whole survey rather than a summary of it: the scan
// order, every verdict, every extent, the site bounds and the error list. The
// bounds are the part most likely to drift, being a reduction over floats
// accumulated in whatever order the files finished.
static void testSurveyDoesNotDependOnThreadCount() {
    std::printf("indexer: the survey reads in parallel and says the same thing\n");

    // Enough files that the work actually gets spread, and deliberately uneven
    // in size so the threads finish out of order.
    std::vector<std::string> paths = writeCorpus("parsurvey", 9, 2000, 30.0, 0.0, 0.0);
    const std::vector<std::string> more =
        writeCorpus("parsurvey_big", 3, 12000, 50.0, 500000.0, 6200000.0);
    paths.insert(paths.end(), more.begin(), more.end());
    // And one that cannot be opened, so the error list is part of the comparison.
    paths.push_back(tmpDir() + "/e57cov_parsurvey_absent.e57");
    CHECK(paths.size() == 13, "corpus written");

    // Everything the survey decided, flattened, so a difference anywhere shows.
    auto flatten = [](const indexer::Survey& s) {
        std::string o;
        char buf[512];
        std::snprintf(buf, sizeof(buf), "%zu|%zu|%d|%d|%.9f,%.9f,%.9f|%.9f,%.9f,%.9f\n",
                      s.filesRead, s.scans.size(), int(s.extentComplete), int(s.hasBounds),
                      s.lo[0], s.lo[1], s.lo[2], s.hi[0], s.hi[1], s.hi[2]);
        o += buf;
        for (const auto& sc : s.scans) {
            std::snprintf(buf, sizeof(buf), "%s#%zu|%s|%d|%d|%d|%llu|%s|%.9f,%.9f,%.9f\n",
                          sc.path.c_str(), sc.scanIndex, sc.name.c_str(), int(sc.kind),
                          int(sc.usable), int(sc.hasExtent),
                          (unsigned long long)sc.recordCount, sc.status.c_str(),
                          sc.setup[0], sc.setup[1], sc.setup[2]);
            o += buf;
        }
        for (const auto& e : s.errors) { o += "E|"; o += e; o += "\n"; }
        return o;
    };

    for (int pass = 0; pass < 2; ++pass) {
        const bool classify = (pass == 1);
        std::string reference;
        for (unsigned n : {1u, 2u, 3u, 4u, 8u, 13u}) {
            indexer::SurveyOptions so;
            so.classify = classify;
            so.threads  = n;
            const std::string got = flatten(indexer::survey(paths, so, nullptr));
            if (reference.empty()) {
                reference = got;
                CHECK(!reference.empty(), "the survey produced something to compare");
            } else {
                CHECK(got == reference,
                      classify ? "same full survey at any thread count"
                               : "same header survey at any thread count");
            }
        }
        // The serial answer is the one being preserved, so it is worth saying
        // out loud that it found what it was supposed to.
        indexer::SurveyOptions so;
        so.classify = classify;
        so.threads  = 1;
        const indexer::Survey s = indexer::survey(paths, so, nullptr);
        CHECK(s.filesRead == 12, "twelve files opened");
        CHECK(s.scans.size() == 12, "one scan each");
        CHECK(s.errors.size() == 1, "and the missing one is reported once");
    }

    // A thread count above the core count is capped rather than obeyed: asking
    // for more threads than there are files cannot be allowed to spawn them.
    indexer::SurveyOptions huge;
    huge.threads = 4096;
    const indexer::Survey s = indexer::survey(paths, huge, nullptr);
    CHECK(s.scans.size() == 12, "an absurd thread count still reads the corpus");
}

// Chunks are built in parallel and the store comes out identical.
//
// Building a chunk is independent work and about 70 per cent of a build, so it runs
// across cores. Appending a finished subtree is NOT independent — the writer hands
// out node indices in append order — so the appends stay serial and in cell order.
// If that ever stopped being true the store would still open and still look
// plausible, with points hanging off the wrong nodes, so the check is content:
// every stored coordinate hashed, and the hash must not move with the thread count.
static void testChunksBuildInParallelAndTheStoreIsIdentical() {
    std::printf("indexer: parallel chunk building, identical store\n");

    // Enough points and spread to occupy several chunk cells.
    const std::vector<std::string> paths = writeCorpus("chunkpar", 8, 9000, 50.0, 0.0, 0.0);
    CHECK(paths.size() == 8, "corpus written");
    indexer::SurveyOptions so;
    so.classify = true;
    const indexer::Survey s = indexer::survey(paths, so, nullptr);
    CHECK(s.usableCount() == 8, "all usable");

    // Every stored point, in store order, hashed.
    auto digest = [](const std::string& path, uint64_t& points, uint32_t& nodes) {
        std::string err;
        store::Reader rd;
        if (!rd.open(path, err)) return uint64_t(0);
        const lod::Tree t = rd.tree();
        nodes = uint32_t(t.nodes.size());
        points = rd.header().totalPoints;
        uint64_t h = 1469598103934665603ull;
        for (size_t i = 0; i < t.nodes.size(); ++i) {
            const lod::StorePoint* pts = rd.points(i);
            if (!pts) continue;
            for (uint32_t j = 0; j < t.nodes[i].pointCount; ++j) {
                const float f[3] = {pts[j].x, pts[j].y, pts[j].z};
                const uint8_t* b = reinterpret_cast<const uint8_t*>(f);
                for (int k = 0; k < 12; ++k) { h ^= b[k]; h *= 1099511628211ull; }
                h ^= pts[j].scanId; h *= 1099511628211ull;
            }
        }
        return h;
    };

    uint64_t reference = 0, refPoints = 0;
    uint32_t refNodes = 0;
    uint32_t chunksSeen = 0;
    for (unsigned ct : {1u, 2u, 3u, 4u, 8u}) {
        const std::string sp = tmpDir() + "/e57cov_chunkpar_" + std::to_string(ct) + ".lod";
        std::remove(sp.c_str());
        indexer::BuildOptions bo;
        bo.chunkThreads = ct;
        // A small chunk target so there are several chunks to spread, and a
        // resident budget that does not then throttle the threads back to one.
        bo.targetPointsPerChunk = 8000;
        bo.maxResidentPoints    = 8000ull * 16;
        indexer::BuildStats st;
        std::string err;
        CHECK(indexer::build(s, sp, bo, st, nullptr, err),
              err.empty() ? "builds" : err.c_str());
        uint64_t points = 0;
        uint32_t nodes = 0;
        const uint64_t h = digest(sp, points, nodes);
        if (reference == 0) {
            reference = h; refPoints = points; refNodes = nodes; chunksSeen = st.chunks;
            CHECK(st.chunks > 1, "the corpus really does split into several chunks");
            CHECK(points == 8 * 9000, "and every point reached the store");
        } else {
            CHECK(h == reference, "the same store content at any chunk thread count");
            CHECK(points == refPoints, "the same point count");
            CHECK(nodes == refNodes, "and the same node count, so the same tree shape");
            CHECK(st.chunks == chunksSeen, "over the same chunks");
        }
        std::remove(sp.c_str());
    }

    // The resident budget throttles the thread count rather than being ignored:
    // one chunk's worth of budget means one chunk at a time, whatever was asked.
    {
        const std::string sp = tmpDir() + "/e57cov_chunkpar_budget.lod";
        std::remove(sp.c_str());
        indexer::BuildOptions bo;
        bo.chunkThreads = 8;
        bo.targetPointsPerChunk = 8000;
        bo.maxResidentPoints    = 8000;          // room for exactly one chunk
        indexer::BuildStats st;
        std::string err;
        CHECK(indexer::build(s, sp, bo, st, nullptr, err),
              err.empty() ? "builds under a one-chunk budget" : err.c_str());
        uint64_t points = 0;
        uint32_t nodes = 0;
        CHECK(digest(sp, points, nodes) == reference,
              "and still produces the same store");
        std::remove(sp.c_str());
    }
}

// The extent pass reads every point, so no point is silently dropped.
//
// build() has to choose an octree root before it can insert anything, and when no
// file declares cartesianBounds it reads the points to find one. That read used a
// fixed stride, which aliases: this fixture writes three concentric shells
// interleaved per direction, so a stride that is a multiple of three samples one
// shell and the root comes out sized to a third of the cloud. Points outside the
// root are absent from the store and read downstream as missing coverage — a
// silent loss pointing the wrong way. Measured before the fix: 82,869 of 120,000
// stored, 37,131 outside the root.
//
// The stride was free to remove. readPoints decodes a whole bytestream whatever
// the consumer does with it, so the stride skipped six comparisons a point and
// saved no decoding at all.
static void testTheExtentPassMissesNothing() {
    std::printf("indexer: the extent pass reads every point\n");

    const std::string path = tmpDir() + "/e57cov_aliasing_shells.e57";
    {
        fixture::Scan sc;
        sc.name = "Interleaved shells";
        sc.fields = {
            {"cartesianX", e57::FieldType::FloatDouble},
            {"cartesianY", e57::FieldType::FloatDouble},
            {"cartesianZ", e57::FieldType::FloatDouble},
        };
        sc.data.assign(3, {});
        // Deliberately the pathological order: shell 0, 1, 2 for each direction in
        // turn, so consecutive points differ in depth with period three.
        Lcg rng;
        for (int i = 0; i < 40000; ++i) {
            const double u = 2.0 * rng.next() - 1.0;
            const double th = 2.0 * 3.14159265358979 * rng.next();
            const double sr = std::sqrt(std::max(0.0, 1.0 - u * u));
            const double dx = sr * std::cos(th), dy = sr * std::sin(th), dz = u;
            for (int k = 0; k < 3; ++k) {
                const double r = 5.0 + 1.5 * dx * dy + 8.0 * double(k);
                sc.data[0].push_back(r * dx);
                sc.data[1].push_back(r * dy);
                sc.data[2].push_back(r * dz);
            }
        }
        CHECK(fixture::write(path, {sc}, 2048), "fixture written");
    }

    indexer::SurveyOptions so;
    so.classify = true;
    const indexer::Survey s = indexer::survey({path}, so, nullptr);
    CHECK(s.scans.size() == 1 && s.usableCount() == 1, "surveyed and usable");
    CHECK(!s.extentComplete, "no declared bounds, so the extent must be read");
    if (s.scans.empty()) return;

    const std::string storePath = tmpDir() + "/e57cov_aliasing_shells.lod";
    std::remove(storePath.c_str());
    indexer::BuildOptions bo;
    indexer::BuildStats st;
    std::string err;
    CHECK(indexer::build(s, storePath, bo, st, nullptr, err),
          err.empty() ? "the store builds" : err.c_str());

    CHECK(st.pointsRead == 120000, "every point was read");
    CHECK(st.outsideRoot == 0, "and none fell outside the root");
    CHECK(st.pointsStored == st.pointsRead, "so every point is in the store");

    store::Reader rd;
    CHECK(rd.open(storePath, err), err.empty() ? "store opens" : err.c_str());
    CHECK(rd.header().totalPoints == 120000, "the store holds the whole cloud");

    // The root has to contain the OUTER shell, which is what a strided sample
    // missed. Radius runs to about 21 m, so a root sized to the inner shell would
    // be a third of this.
    const lod::Tree t = rd.tree();
    CHECK(!t.nodes.empty(), "the tree has a root");
    if (!t.nodes.empty()) {
        const lod::Aabb& root = t.nodes[0].bounds;
        const float half = 0.5f * std::max(root.hi[0] - root.lo[0], root.hi[2] - root.lo[2]);
        CHECK(half > 20.0f, "the root spans the outer shell, not just the inner one");
    }

    std::remove(storePath.c_str());
    std::remove(path.c_str());

    // And the same with intensity present, which is the case build()'s extent pass
    // skips the colour work for. Getting that wrong would mean an extent measured
    // from a differently-filtered set of points than the one indexed, so the check
    // is that every point still lands inside the root.
    {
        const std::string ip = tmpDir() + "/e57cov_intensity_extent.e57";
        fixture::Scan sc;
        sc.name = "With intensity";
        sc.hasPose = true;
        sc.q[0] = 1.0;
        sc.t[0] = 11.0; sc.t[1] = -7.0; sc.t[2] = 1.6;
        sc.fields = {
            {"cartesianX", e57::FieldType::FloatDouble},
            {"cartesianY", e57::FieldType::FloatDouble},
            {"cartesianZ", e57::FieldType::FloatDouble},
            {"intensity",  e57::FieldType::FloatDouble},
        };
        sc.data.assign(4, {});
        Lcg rng;
        for (int i = 0; i < 30000; ++i) {
            const double az = rng.next() * 6.28318530718;
            const double el = (rng.next() - 0.5) * 1.2;
            const double r  = 9.0 + 3.0 * std::sin(2.0 * az) + 2.0 * std::cos(3.0 * el);
            const double ce = std::cos(el);
            sc.data[0].push_back(r * ce * std::cos(az));
            sc.data[1].push_back(r * ce * std::sin(az));
            sc.data[2].push_back(r * std::sin(el));
            // Raw counts, not 0..1 — the case the intensity range exists for.
            sc.data[3].push_back(800.0 + 4000.0 * rng.next());
        }
        // 1024 records a packet, not the 2048 used elsewhere: that is a count of
        // RECORDS, and four double fields at 2048 records overflows E57's 64 KB
        // packet limit, which shows up as a decode failure and no points at all.
        if (fixture::write(ip, {sc}, 1024)) {
            const indexer::Survey is = indexer::survey({ip}, so, nullptr);
            CHECK(is.usableCount() == 1, "the intensity scan is usable");
            const std::string isp = tmpDir() + "/e57cov_intensity_extent.lod";
            std::remove(isp.c_str());
            indexer::BuildStats ist;
            std::string ierr;
            CHECK(indexer::build(is, isp, indexer::BuildOptions{}, ist, nullptr, ierr),
                  ierr.empty() ? "it builds" : ierr.c_str());
            CHECK(ist.pointsRead == 30000, "every point read");
            CHECK(ist.outsideRoot == 0, "none outside the root");
            CHECK(ist.pointsStored == ist.pointsRead, "and all stored");
            store::Reader ird;
            if (ird.open(isp, ierr)) {
                // Colour still came from intensity on the indexing pass, so the
                // points are not all the flat fallback grey.
                const lod::Tree t = ird.tree();
                std::set<uint32_t> shades;
                for (size_t i = 0; i < t.nodes.size() && shades.size() < 5; ++i) {
                    const lod::StorePoint* pts = ird.points(i);
                    if (!pts) continue;
                    for (uint32_t j = 0; j < t.nodes[i].pointCount; ++j)
                        shades.insert((uint32_t(pts[j].r) << 16) | (uint32_t(pts[j].g) << 8) |
                                      pts[j].b);
                }
                CHECK(shades.size() > 1, "intensity still became colour on the indexing pass");
            }
            std::remove(isp.c_str());
            std::remove(ip.c_str());
        }
    }
}

// A cloud the range-spread heuristic dislikes is still indexed and still drawn.
//
// This is the regression guard for a defect that emptied the store. `usable` used
// to mean "not called merged by the heuristic", and the heuristic rises with scene
// scale — every scan of the job this was built against measured over the threshold
// — so build() refused the whole corpus with "no usable scans to index" and the
// viewer drew nothing.
//
// `usable` now asks the only question the point store needs answered: are there
// positions to draw. The verdict travels alongside as a label, in the scan's
// store flags, where the UI can show it and nothing can act on it.
static void testTheHeuristicLabelsAndNothingMore() {
    std::printf("indexer: a scan the heuristic dislikes is still indexed\n");

    // Concentric shells 3 m apart: the same shape the classifier's own merged
    // fixture uses, so the heuristic is certain to fire. No gridding metadata, so
    // nothing overrules it and the label lands.
    const std::string path = tmpDir() + "/e57cov_merged_indexed.e57";
    {
        fixture::Scan sc;
        sc.name = "Shells";
        sc.hasPose = true;
        sc.q[0] = 1.0;
        sc.t[0] = 4.0; sc.t[1] = -2.0; sc.t[2] = 1.6;
        sc.fields = {
            {"cartesianX", e57::FieldType::FloatDouble},
            {"cartesianY", e57::FieldType::FloatDouble},
            {"cartesianZ", e57::FieldType::FloatDouble},
        };
        sc.data.assign(3, {});
        // Shell by shell rather than interleaved per direction, and that matters.
        // build()'s extent pass samples at a fixed stride; with three shells
        // interleaved the stride came out a multiple of three and sampled the
        // innermost shell only, so the root was sized to a third of the cloud and
        // 31 per cent of the points landed outside it and were dropped. A real
        // scan's order is row-major, not interleaved by depth, so this ordering is
        // the representative one — but the aliasing is real and is noted in the
        // report on build().
        for (int k = 0; k < 3; ++k) {
            Lcg rng;                       // same directions in every shell
            for (int i = 0; i < 40000; ++i) {
                const double u = 2.0 * rng.next() - 1.0;
                const double th = 2.0 * 3.14159265358979 * rng.next();
                const double sr = std::sqrt(std::max(0.0, 1.0 - u * u));
                const double dx = sr * std::cos(th), dy = sr * std::sin(th), dz = u;
                const double r = 5.0 + 1.5 * dx * dy + 3.0 * double(k);
                sc.data[0].push_back(r * dx);
                sc.data[1].push_back(r * dy);
                sc.data[2].push_back(r * dz);
            }
        }
        CHECK(fixture::write(path, {sc}, 2048), "fixture written");
    }

    indexer::SurveyOptions so;
    so.classify = true;
    const indexer::Survey s = indexer::survey({path}, so, nullptr);
    CHECK(s.scans.size() == 1, "the scan was surveyed");
    if (s.scans.empty()) return;
    const indexer::ScanRef& ref = s.scans[0];

    CHECK(ref.looksMerged, "the heuristic fires on this cloud");
    CHECK(ref.kind == check::Kind::Unified, "and labels it, there being no grid declared");
    CHECK(ref.usable, "but the scan is still usable, because it has positions to draw");
    CHECK(s.usableCount() == 1, "so the corpus has a usable scan");

    // And it reaches the store, with the label on it.
    const std::string storePath = tmpDir() + "/e57cov_merged_indexed.lod";
    std::remove(storePath.c_str());
    indexer::BuildOptions bo;
    indexer::BuildStats st;
    std::string err;
    CHECK(indexer::build(s, storePath, bo, st, nullptr, err),
          err.empty() ? "the store builds" : err.c_str());
    CHECK(st.pointsRead == 120000, "every point was read");
    CHECK(st.pointsStored == st.pointsRead, "and every point stored");

    store::Reader rd;
    CHECK(rd.open(storePath, err), err.empty() ? "store opens" : err.c_str());
    CHECK(rd.scanCount() == 1, "the scan is in the table");
    if (rd.scanCount() == 1) {
        CHECK((rd.scan(0).flags & store::kScanLooksMerged) != 0,
              "carrying the heuristic's finding as a flag");
        CHECK((rd.scan(0).flags & store::kScanStructured) == 0,
              "and not claiming to be positively structured");
    }
    CHECK(rd.header().totalPoints == st.pointsRead, "with all of its points");

    // A scan with no position fields at all is the one thing that IS unusable:
    // there is nothing to put in a point store.
    {
        const std::string bare = tmpDir() + "/e57cov_no_positions.e57";
        fixture::Scan sc;
        sc.name = "Intensity only";
        sc.fields = {{"intensity", e57::FieldType::FloatDouble}};
        sc.data.assign(1, {});
        for (int i = 0; i < 1000; ++i) sc.data[0].push_back(double(i) * 0.001);
        if (fixture::write(bare, {sc}, 2048)) {
            const indexer::Survey b = indexer::survey({bare}, so, nullptr);
            if (b.scans.size() == 1)
                CHECK(!b.scans[0].usable, "a scan with no positions cannot be drawn");
        }
    }

    std::remove(storePath.c_str());
    std::remove(path.c_str());
}

// The per-file check cache: a hit must be indistinguishable from doing the work.
//
// This is the one that earns the most on a real corpus, and it is also the one
// whose failure mode is worst. The store is cached under a key over the whole
// corpus, so adding one scan to a thousand rebuilds it and re-checks all of them;
// checking depends on nothing but the file, so 999 of those are waste. The risk
// is the mirror image: a cache consulted when it should not be serves a stale
// verdict, and a verdict decides whether a scan is indexed at all.
//
// So what is checked is equality with the uncached answer, and invalidation on
// each of the things that identify a file.
static void testTheCheckCacheIsIndistinguishableFromChecking() {
    std::printf("indexer: the per-file check cache\n");

    const std::vector<std::string> paths =
        writeCorpus("cachecorpus", 6, 4000, 30.0, 0.0, 0.0);
    CHECK(paths.size() == 6, "corpus written");
    const std::string cachePath = tmpDir() + "/e57cov_checkcache.txt";
    std::remove(cachePath.c_str());

    auto flatten = [](const indexer::Survey& s) {
        std::string o;
        char buf[512];
        std::snprintf(buf, sizeof(buf), "%zu|%zu|%d|%d|%.9f,%.9f,%.9f|%.9f,%.9f,%.9f\n",
                      s.filesRead, s.scans.size(), int(s.extentComplete), int(s.hasBounds),
                      s.lo[0], s.lo[1], s.lo[2], s.hi[0], s.hi[1], s.hi[2]);
        o += buf;
        for (const auto& sc : s.scans) {
            std::snprintf(buf, sizeof(buf), "%s#%zu|%s|%s|%d|%d|%d|%llu|%s|%.9f,%.9f,%.9f|%.9f,%.9f,%.9f\n",
                          sc.path.c_str(), sc.scanIndex, sc.name.c_str(), sc.guid.c_str(),
                          int(sc.kind), int(sc.usable), int(sc.hasExtent),
                          (unsigned long long)sc.recordCount, sc.status.c_str(),
                          sc.setup[0], sc.setup[1], sc.setup[2], sc.lo[0], sc.lo[1], sc.lo[2]);
            o += buf;
        }
        return o;
    };

    // The uncached answer, which is the one being preserved.
    indexer::SurveyOptions plain;
    plain.classify = true;
    const std::string want = flatten(indexer::survey(paths, plain, nullptr));

    // Cold: nothing cached, so everything is checked and the file is written.
    indexer::SurveyOptions cached = plain;
    cached.cachePath = cachePath;
    const indexer::Survey cold = indexer::survey(paths, cached, nullptr);
    CHECK(cold.filesFromCache == 0, "a cold run reuses nothing");
    CHECK(flatten(cold) == want, "and answers exactly as the uncached survey does");

    // Warm: every file unchanged, so every file is a hit — and the answer is the
    // same one, which is the whole claim.
    const indexer::Survey warm = indexer::survey(paths, cached, nullptr);
    CHECK(warm.filesFromCache == 6, "a warm run reuses every file");
    CHECK(flatten(warm) == want, "and still answers identically");

    // The status strings are the part most likely to come back mangled: they are
    // built from measured numbers and carry punctuation and multi-byte
    // characters, so they are stored length-prefixed rather than delimited.
    bool statusesSurvived = !warm.scans.empty();
    for (size_t i = 0; i < warm.scans.size() && i < cold.scans.size(); ++i)
        if (warm.scans[i].status != cold.scans[i].status ||
            warm.scans[i].name != cold.scans[i].name) statusesSurvived = false;
    CHECK(statusesSurvived, "names and verdict strings come back intact");

    // Touched: a file whose modification time moves is re-checked, not believed.
    {
        const std::string victim = paths[2];
        struct stat st{};
        CHECK(::stat(victim.c_str(), &st) == 0, "victim stat'd");
        const struct utimbuf times{st.st_atime, st.st_mtime + 120};
        CHECK(::utime(victim.c_str(), &times) == 0, "mtime moved");
        const indexer::Survey after = indexer::survey(paths, cached, nullptr);
        CHECK(after.filesFromCache == 5, "the touched file is checked again");
        CHECK(flatten(after) == want, "and the answer does not change");
    }

    // Changed in size: same test, the other half of the stamp. Rewritten with a
    // different point count, so both the size and the verdict's inputs move —
    // which also means the expected answer moves, and every check after this
    // point compares against the rewritten corpus.
    std::string wantBigger;
    {
        const std::vector<std::string> rewritten =
            writeCorpus("cachecorpus", 6, 5000, 30.0, 0.0, 0.0);
        CHECK(rewritten.size() == 6, "corpus rewritten at a different size");
        wantBigger = flatten(indexer::survey(paths, plain, nullptr));
        CHECK(wantBigger != want, "the rewritten corpus really does survey differently");
        const indexer::Survey after = indexer::survey(paths, cached, nullptr);
        CHECK(after.filesFromCache == 0, "every rewritten file is checked again");
        CHECK(flatten(after) == wantBigger, "and matches what checking them says");
    }

    // A corpus that shrinks does not leave its old entries behind for ever.
    {
        const std::vector<std::string> two(paths.begin(), paths.begin() + 2);
        indexer::survey(two, cached, nullptr);
        indexer::SurveyCache c;
        CHECK(c.load(cachePath), "cache loaded");
        CHECK(c.entries() == 2, "only the files still in the corpus are kept");
    }

    // A truncated cache is treated as absent rather than trusted as far as it
    // goes. This is the realistic corruption — a save interrupted, a disk full —
    // and a half-read cache serving a few entries would be worse than none,
    // because the entries it did serve would look authoritative.
    {
        indexer::survey(paths, cached, nullptr);          // refill it
        std::string whole;
        {
            std::ifstream in(cachePath, std::ios::binary);
            std::ostringstream ss;
            ss << in.rdbuf();
            whole = ss.str();
        }
        CHECK(whole.size() > 200, "the cache has something in it to truncate");
        {
            std::ofstream out(cachePath, std::ios::binary | std::ios::trunc);
            out << whole.substr(0, whole.size() / 2);     // cut mid-entry
        }
        indexer::SurveyCache c;
        CHECK(!c.load(cachePath), "a truncated cache fails to load");
        CHECK(c.entries() == 0, "and yields nothing rather than a partial set");
        const indexer::Survey recovered = indexer::survey(paths, cached, nullptr);
        CHECK(recovered.filesFromCache == 0, "so the survey checks everything");
        CHECK(flatten(recovered) == wantBigger, "and is still right");
    }

    // An entry whose declared length does not match its content is refused. The
    // length prefix is what makes a status string safe to store; this is the
    // check that it is actually being honoured rather than merely written.
    {
        indexer::survey(paths, cached, nullptr);
        std::string whole;
        {
            std::ifstream in(cachePath, std::ios::binary);
            std::ostringstream ss;
            ss << in.rdbuf();
            whole = ss.str();
        }
        const size_t nl = whole.find('\n');
        const size_t colon = whole.find(':', nl);
        CHECK(colon != std::string::npos, "found a length-prefixed field");
        {
            // Claim one more byte than the field holds.
            std::string bent = whole;
            bent.insert(colon, "9");
            std::ofstream out(cachePath, std::ios::binary | std::ios::trunc);
            out << bent;
        }
        indexer::SurveyCache c;
        CHECK(!c.load(cachePath), "a field whose length does not match is refused");
        CHECK(c.entries() == 0, "and takes the whole cache with it");
    }

    // A cache of the wrong version is ignored, which is what lets the format
    // change without a stale file being read as the new one.
    {
        {
            std::ofstream wrong(cachePath, std::ios::trunc);
            wrong << "e57cov-survey-cache 99 0\n";
        }
        indexer::SurveyCache c;
        CHECK(!c.load(cachePath), "a future version is refused");
        const indexer::Survey after = indexer::survey(paths, cached, nullptr);
        CHECK(after.filesFromCache == 0, "and the work is simply done again");
        CHECK(flatten(after) == wantBigger, "correctly");
    }

    // The header-only survey must never read or write this cache: it reaches a
    // cheaper verdict from metadata alone, and storing that under the same key
    // would serve it to a run that asked for the full check.
    {
        std::remove(cachePath.c_str());
        indexer::SurveyOptions headers;
        headers.cachePath = cachePath;       // classify stays false
        const indexer::Survey h = indexer::survey(paths, headers, nullptr);
        CHECK(h.filesFromCache == 0, "the header survey reuses nothing");
        indexer::SurveyCache c;
        c.load(cachePath);
        CHECK(c.entries() == 0, "and writes nothing for a later run to find");
    }

    // And an unreadable file is not cached as an answer.
    {
        std::remove(cachePath.c_str());
        std::vector<std::string> withGhost = paths;
        withGhost.push_back(tmpDir() + "/e57cov_cache_absent.e57");
        const indexer::Survey g = indexer::survey(withGhost, cached, nullptr);
        CHECK(g.errors.size() == 1, "the missing file is reported");
        indexer::SurveyCache c;
        CHECK(c.load(cachePath), "cache loaded");
        CHECK(c.entries() == 6, "and only the six real files are remembered");
    }

    std::remove(cachePath.c_str());
}

static void testBuildRoundTrip() {
    std::printf("indexer: build a store from a corpus\n");
    const std::vector<std::string> paths = writeCorpus("build", 20, 4000, 60.0, 500000.0, 6200000.0);
    CHECK(paths.size() == 20, "corpus written");

    indexer::SurveyOptions so;
    so.classify = true;                     // the build path does the full check
    const indexer::Survey s = indexer::survey(paths, so, nullptr);
    CHECK(s.usableCount() == 20, "all setups are indexed");
    // These fixtures carry no gridding metadata, and a few thousand points
    // spread over the sphere leave too few populated direction bins to judge.
    // That is ambiguity, not a merged cloud, and it must not exclude a scan.
    size_t ambiguous = 0, merged = 0;
    for (const auto& sc : s.scans) {
        if (sc.kind == check::Kind::Ambiguous) ++ambiguous;
        if (sc.kind == check::Kind::Unified)   ++merged;
    }
    CHECK(merged == 0, "nothing is misreported as merged");
    CHECK(ambiguous == 20, "inconclusive scans are reported as ambiguous, and still indexed");

    const std::string storePath = tmpDir() + "/e57cov_corpus.lod";
    indexer::BuildOptions bo;
    bo.tree.gridResolution   = 16;
    bo.tree.maxPointsPerNode = 4000;
    bo.tree.maxLevel         = 12;
    bo.targetPointsPerChunk  = 6000;        // force real chunking on a small corpus
    bo.chunkDir              = tmpDir();

    indexer::BuildStats stats;
    std::string err;
    std::vector<std::string> stages;
    const bool ok = indexer::build(s, storePath, bo, stats,
        [&](const std::string& stage, uint64_t, uint64_t) {
            if (stages.empty() || stages.back() != stage) stages.push_back(stage);
            return true;
        }, err);
    CHECK(ok, err.empty() ? "build succeeded" : err.c_str());
    if (!ok) return;

    CHECK(stats.chunkLevel >= 1, "a chunk level was chosen");
    CHECK(stats.chunks > 1, "the build actually chunked rather than staying in memory");
    CHECK(stats.spilled > 0, "most points went through spill files");
    CHECK(stats.dropped == 0, "no points were dropped at the depth limit");
    CHECK(stats.outsideRoot == 0, "no points fell outside the root bounds");
    CHECK(stats.pointsRead == 20 * 4000, "every point was read");
    CHECK(stats.pointsStored == stats.pointsRead, "every point reached the store");
    CHECK(stages.size() >= 2, "progress reported distinct stages");

    store::Reader r;
    CHECK(r.open(storePath, err), err.empty() ? "store opens" : err.c_str());
    CHECK(r.header().totalPoints == stats.pointsRead, "store total matches what was read");
    CHECK(r.scanCount() == 20, "scan table holds every setup");
    CHECK(std::string(r.scan(3).name) == "Setup 3", "scan names carried through");
    CHECK((r.scan(3).flags & store::kScanAmbiguous) != 0,
          "ambiguity is recorded in the store rather than lost");
    CHECK(std::fabs(r.header().origin[0] - 500000.0) < 1000.0,
          "store origin sits at the site, keeping float offsets small");

    // The tree must be one connected whole: a mis-stitched chunk would strand
    // points that then silently never draw.
    const lod::Tree t = r.tree();
    std::vector<int> visits(t.nodes.size(), 0);
    std::vector<uint32_t> stack{0};
    uint64_t reachablePoints = 0;
    while (!stack.empty()) {
        const uint32_t n = stack.back(); stack.pop_back();
        if (++visits[n] > 1) continue;
        reachablePoints += t.nodes[n].pointCount;
        for (int o = 0; o < 8; ++o)
            if (t.nodes[n].child[o]) stack.push_back(t.nodes[n].child[o]);
    }
    size_t unreachable = 0, repeated = 0;
    for (int v : visits) { if (v == 0) ++unreachable; if (v > 1) ++repeated; }
    CHECK(unreachable == 0, "every node is reachable from the root");
    CHECK(repeated == 0, "and by exactly one path");
    CHECK(reachablePoints == stats.pointsRead, "every stored point is reachable");

    // Offsets must be small: the site sits at UTM magnitudes and float32 there
    // resolves to ~0.0625 m.
    float maxOffset = 0;
    for (uint64_t n = 0; n < r.nodeCount(); ++n) {
        const lod::StorePoint* p = r.points(n);
        for (uint32_t i = 0; i < r.node(n).pointCount; ++i)
            maxOffset = std::max(maxOffset, std::fabs(p[i].x));
    }
    CHECK(maxOffset < 5000.0f, "stored coordinates are local offsets, not UTM values");

    // Every setup must remain individually identifiable.
    std::set<uint16_t> ids;
    for (uint64_t n = 0; n < r.nodeCount(); ++n) {
        const lod::StorePoint* p = r.points(n);
        for (uint32_t i = 0; i < r.node(n).pointCount; ++i) ids.insert(p[i].scanId);
    }
    CHECK(ids.size() == 20, "all twenty setups survive into the store");
    CHECK(*ids.rbegin() < r.scanCount(), "no scanId points past the scan table");

    // And the store must be navigable: bounded selection over the whole site.
    viewer::OrbitCamera cam;
    cam.setViewport(1400, 900);
    cam.frameBounds({t.bounds.lo[0], t.bounds.lo[1], t.bounds.lo[2]},
                    {t.bounds.hi[0], t.bounds.hi[1], t.bounds.hi[2]});
    lod::SelectOptions sel;
    sel.pointBudget = 20000;
    const lod::Selection out =
        lod::selectNodes(t, cam.viewProjection(), cam.eye(), 900.0f / 1.0471976f, sel);
    CHECK(out.points > 0, "the store draws something");
    CHECK(out.points <= sel.pointBudget, "under a budget far below the store's size");
    CHECK(out.nodes[0] == 0, "starting from the root");

    // Picking must work straight off the mapping, restricted to what is drawn.
    lod::SelectOptions big;
    big.pointBudget = 1000000;
    const lod::Selection all =
        lod::selectNodes(t, cam.viewProjection(), cam.eye(), 900.0f / 1.0471976f, big);
    const viewer::PickResult hit =
        viewer::pickNearestInStore(r, t, all, cam, 0.0f, 0.0f, 200.0f);
    if (hit.hit) {
        const m3::Vec4 clip = cam.viewProjection() *
                              m3::Vec4{hit.world.x, hit.world.y, hit.world.z, 1.0f};
        CHECK(clip.w > 0, "picked point is in front of the camera");
        CHECK(std::fabs(clip.x / clip.w) < 0.3, "picked point is near the crosshair");
        CHECK(hit.cloudIndex < r.nodeCount(), "pick names a real node");
    } else {
        CHECK(true, "nothing under the crosshair is a valid outcome");
    }
    const viewer::PickResult none =
        viewer::pickNearestInStore(r, t, lod::Selection{}, cam, 0.0f, 0.0f, 50.0f);
    CHECK(!none.hit, "an empty selection picks nothing");

    // Spill files must not be left behind.
    std::FILE* leftover = std::fopen((tmpDir() + "/chunk_00000000.bin").c_str(), "rb");
    CHECK(leftover == nullptr, "chunk files are cleaned up");
    if (leftover) std::fclose(leftover);
}

static void testBuildRefusesImpossibleInput() {
    std::printf("indexer: build refuses what it cannot represent\n");
    indexer::Survey empty;
    indexer::BuildStats stats;
    std::string err;
    CHECK(!indexer::build(empty, tmpDir() + "/e57cov_empty.lod", indexer::BuildOptions{},
                          stats, nullptr, err),
          "an empty survey is refused");
    CHECK(err.find("no usable scans") != std::string::npos, "with a reason");

    // scanId is uint16, so the format cannot address more than 65535 setups.
    // Wrapping would silently mislabel points.
    indexer::Survey huge;
    huge.hasBounds = true;
    for (int i = 0; i < 3; ++i) { huge.lo[i] = 0; huge.hi[i] = 10; }
    huge.scans.resize(70000);
    for (auto& sc : huge.scans) { sc.usable = true; sc.recordCount = 1; }
    CHECK(!indexer::build(huge, tmpDir() + "/e57cov_huge.lod", indexer::BuildOptions{},
                          stats, nullptr, err),
          "a corpus past the scanId limit is refused");
    CHECK(err.find("65535") != std::string::npos, "and names the limit");
}

static void testBuildCancels() {
    std::printf("indexer: build is cancellable\n");
    const std::vector<std::string> paths = writeCorpus("cancel", 6, 2000, 20.0, 0.0, 0.0);
    indexer::SurveyOptions so;
    const indexer::Survey s = indexer::survey(paths, so, nullptr);

    indexer::BuildStats stats;
    std::string err;
    const bool ok = indexer::build(s, tmpDir() + "/e57cov_cancel.lod", indexer::BuildOptions{},
                                   stats, [](const std::string&, uint64_t, uint64_t) {
                                       return false;   // cancel immediately
                                   }, err);
    CHECK(!ok, "cancelling stops the build");
    CHECK(err == "cancelled", "and says so rather than reporting a corrupt store as done");
}

int main() {
    std::printf("E57 Coverage Checker — indexer tests\n\n");
    testChunkLevel();
    testCellIndex();
    testSurveyIsHeaderOnly();
    testSurveyDoesNotDependOnThreadCount();
    testTheExtentPassMissesNothing();
    testChunksBuildInParallelAndTheStoreIsIdentical();
    testTheHeuristicLabelsAndNothingMore();
    testTheCheckCacheIsIndistinguishableFromChecking();
    testBuildRoundTrip();
    testBuildRefusesImpossibleInput();
    testBuildCancels();
    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
