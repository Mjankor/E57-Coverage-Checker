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
#include <set>
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
    testBuildRoundTrip();
    testBuildRefusesImpossibleInput();
    testBuildCancels();
    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
