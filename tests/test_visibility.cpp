// Tests for the visibility job — the stage between a carve and a picture.
//
// Three things have to hold, and the last is the one that would be easy to get
// wrong and never notice:
//
//   1. The frontier reduction changes what is drawn, never what is counted. The
//      statistics from a frontier run and a solid run must be identical, and
//      the frontier voxels must be a subset of the solid ones.
//   2. The display cap samples rather than truncates: capping produces a subset
//      of the uncapped result, spread across the site rather than the first N
//      tiles' worth.
//   3. None of it depends on tile size. Same stats, and — because the sample is
//      chosen by hashing the global lattice index rather than by counting —
//      the same voxels, drawn in the same places.

#include "../src/visibility.h"
#include "e57_fixture.h"

#include <algorithm>
#include <array>
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

static std::string tmpPath(const char* name) {
    const char* d = std::getenv("E57COV_TMPDIR");
    return std::string(d ? d : "/tmp") + "/e57cov_" + name + ".e57";
}

static constexpr double kTau = 6.28318530717958648;
static constexpr int    kRows = 90, kCols = 180;

// A room 10 x 8 x 3 with an interior wall at x = 1 spanning y in [-4, 0], so the
// region behind it is shadowed from a setup at negative x and open to one at
// positive x. It also contains a sealed cupboard, 2 x 2 x 2, whose inside no ray
// can enter: that is the one thing in the scene that is genuinely a coverage
// failure, and the classification has to find it and nothing else.
static bool hitRoom(double ox, double oy, double oz,
                    double dx, double dy, double dz, double& t) {
    t = 1e300;
    auto plane = [&](double denom, double num, double lo0, double hi0,
                     double lo1, double hi1, int a0, int a1) {
        if (std::fabs(denom) < 1e-12) return;
        const double tt = num / denom;
        if (tt <= 1e-6 || tt >= t) return;
        const double p[3] = {ox + tt * dx, oy + tt * dy, oz + tt * dz};
        if (p[a0] < lo0 || p[a0] > hi0 || p[a1] < lo1 || p[a1] > hi1) return;
        t = tt;
    };
    plane(dx, -5 - ox, -4, 4, 0, 3, 1, 2);
    plane(dx,  5 - ox, -4, 4, 0, 3, 1, 2);
    plane(dy, -4 - oy, -5, 5, 0, 3, 0, 2);
    plane(dy,  4 - oy, -5, 5, 0, 3, 0, 2);
    plane(dz,  0 - oz, -5, 5, -4, 4, 0, 1);
    plane(dz,  3 - oz, -5, 5, -4, 4, 0, 1);
    plane(dx,  1 - ox, -4, 0, 0, 3, 1, 2);   // interior wall

    // A sealed cupboard in the far corner: x in [-4,-2], y in [-3,-1], and up to
    // z = 2, standing on the floor. Five faces plus the floor, so every ray stops
    // on its outside and its interior is unobservable from anywhere. Placed well
    // away from both setups — a setup standing inside it would see its interior
    // and there would be nothing to find.
    plane(dx, -4 - ox, -3, -1, 0, 2, 1, 2);
    plane(dx, -2 - ox, -3, -1, 0, 2, 1, 2);
    plane(dy, -3 - oy, -4, -2, 0, 2, 0, 2);
    plane(dy, -1 - oy, -4, -2, 0, 2, 0, 2);
    plane(dz,  2 - oz, -4, -2, -3, -1, 0, 1);
    return t < 1e299;
}

static fixture::Scan roomScan(const char* name, double sx, double sy, double sz) {
    fixture::Scan sc;
    sc.name = name;
    sc.hasPose = true;
    sc.q[0] = 1.0;
    sc.t[0] = sx; sc.t[1] = sy; sc.t[2] = sz;
    sc.hasIndexBounds = true;
    sc.rowMin = 0; sc.rowMax = kRows - 1;
    sc.colMin = 0; sc.colMax = kCols - 1;
    sc.fields = {
        {"cartesianX",  e57::FieldType::FloatDouble},
        {"cartesianY",  e57::FieldType::FloatDouble},
        {"cartesianZ",  e57::FieldType::FloatDouble},
        {"rowIndex",    e57::FieldType::Integer, 0, kRows - 1},
        {"columnIndex", e57::FieldType::Integer, 0, kCols - 1},
    };
    sc.data.assign(5, {});
    for (int r = 0; r < kRows; ++r) {
        const double el = -1.2 + double(r) * (2.4 / double(kRows));
        for (int c = 0; c < kCols; ++c) {
            const double az = double(c) * kTau / double(kCols);
            const double ce = std::cos(el);
            const double dx = ce * std::cos(az), dy = ce * std::sin(az), dz = std::sin(el);
            double t;
            if (!hitRoom(sx, sy, sz, dx, dy, dz, t)) continue;
            sc.data[0].push_back(t * dx);
            sc.data[1].push_back(t * dy);
            sc.data[2].push_back(t * dz);
            sc.data[3].push_back(double(r));
            sc.data[4].push_back(double(c));
        }
    }
    return sc;
}

// Absolute voxel positions on the global lattice, so two results can be
// compared without caring which origin each picked.
static std::set<std::array<int64_t, 3>> latticeSet(const vis::Result& r) {
    std::set<std::array<int64_t, 3>> s;
    for (const lod::StorePoint& p : r.voxels) {
        s.insert({int64_t(std::llround((double(p.x) + r.origin[0]) / r.voxelSize - 0.5)),
                  int64_t(std::llround((double(p.y) + r.origin[1]) / r.voxelSize - 0.5)),
                  int64_t(std::llround((double(p.z) + r.origin[2]) / r.voxelSize - 0.5))});
    }
    return s;
}

// `voxels` counts work done and is expected to change with the tiling; every
// other statistic describes the site and must not.
static bool statsEqual(const carve::Stats& a, const carve::Stats& b) {
    return a.reachable == b.reachable && a.visible == b.visible &&
           a.occupied == b.occupied && a.unknown == b.unknown && a.setupTests == b.setupTests;
}

// ---------------------------------------------------------------------------

// Settings that exist in two places have to agree in both. This one did not:
// rimg defaulted the drop filter off and vis defaulted it on, so the library's
// documented behaviour and every actual run disagreed, silently.
static void testDefaultsAgreeWithTheLibrary() {
    std::printf("defaults agree between the job and the library\n");

    const vis::Options v;
    const rimg::Options r;
    CHECK(v.skyRadius == r.noReturnRadius,
          "the drop filter is off in both, or on in both");
    CHECK(v.skyFraction == r.noReturnFraction, "and at the same threshold");
    CHECK(v.blindCone == r.blindCone, "the blind cone policy matches too");
    CHECK(v.maxRange == r.maxRange, "as does the rated range");
    CHECK(v.skyRadius == 0,
          "and the drop filter is off: a ray either returned or it did not");
}

static void testVoxelHash() {
    std::printf("voxel hash\n");

    CHECK(vis::voxelHash(3, -7, 11) == vis::voxelHash(3, -7, 11), "deterministic");
    CHECK(vis::voxelHash(3, -7, 11) != vis::voxelHash(3, -7, 12), "neighbours differ");
    CHECK(vis::voxelHash(0, 0, 0) != vis::voxelHash(0, 0, 1), "and at the origin too");

    // The cap works by keeping hashes below a threshold, so the hash has to be
    // spread evenly. A lattice-shaped hash would keep whole slabs and drop
    // whole slabs, which would look like missing geometry rather than a sample.
    const uint64_t half = ~0ull / 2;
    uint64_t below = 0, total = 0;
    for (int x = -10; x <= 10; ++x)
        for (int y = -10; y <= 10; ++y)
            for (int z = -10; z <= 10; ++z, ++total)
                if (vis::voxelHash(x, y, z) < half) ++below;
    const double frac = double(below) / double(total);
    CHECK(frac > 0.46 && frac < 0.54, "half the lattice falls below the half threshold");

    // Neighbours must not correlate: adjacent voxels landing on the same side
    // is exactly the lattice failure above.
    uint64_t sameAsNeighbour = 0, pairs = 0;
    for (int x = -10; x <= 10; ++x)
        for (int y = -10; y <= 10; ++y)
            for (int z = -10; z < 10; ++z, ++pairs)
                if ((vis::voxelHash(x, y, z) < half) == (vis::voxelHash(x, y, z + 1) < half))
                    ++sameAsNeighbour;
    const double agree = double(sameAsNeighbour) / double(pairs);
    CHECK(agree > 0.45 && agree < 0.55, "adjacent voxels are uncorrelated");
}

static void testTouchesVisible() {
    std::printf("frontier rule\n");

    carve::Tile t;
    t.dim = 5; t.core = 3; t.apron = 1;
    t.state.assign(125, 0);
    auto at = [&](uint32_t x, uint32_t y, uint32_t z) -> uint8_t& {
        return t.state[t.index(x, y, z)];
    };

    // A lone unknown voxel with nothing around it is not on any frontier.
    at(2, 2, 2) = carve::kReachable;
    CHECK(!vis::touchesVisible(t, 2, 2, 2), "isolated unknown touches nothing");

    // A face neighbour that is visible puts it on the frontier.
    at(3, 2, 2) = carve::kReachable | carve::kVisible;
    CHECK(vis::touchesVisible(t, 2, 2, 2), "a visible face neighbour is a frontier");

    // An occupied neighbour is not: a measured surface between them is exactly
    // what stops it being a line of sight.
    carve::Tile u = t;
    u.state.assign(125, 0);
    auto uat = [&](uint32_t x, uint32_t y, uint32_t z) -> uint8_t& {
        return u.state[u.index(x, y, z)];
    };
    uat(2, 2, 2) = carve::kReachable;
    uat(3, 2, 2) = carve::kReachable | carve::kOccupied;
    CHECK(!vis::touchesVisible(u, 2, 2, 2), "an occupied neighbour is not a frontier");

    // Diagonals do not count — a shared edge is not a path between voxels.
    carve::Tile v = u;
    v.state.assign(125, 0);
    v.state[v.index(2, 2, 2)] = carve::kReachable;
    v.state[v.index(3, 3, 2)] = carve::kReachable | carve::kVisible;
    CHECK(!vis::touchesVisible(v, 2, 2, 2), "a diagonal neighbour is not a frontier");

    // The apron is what makes an edge voxel answerable: voxel 1 is the first
    // interior one, and its neighbour at 0 lives in the apron.
    carve::Tile w = v;
    w.state.assign(125, 0);
    w.state[w.index(1, 2, 2)] = carve::kReachable;
    w.state[w.index(0, 2, 2)] = carve::kReachable | carve::kVisible;
    CHECK(vis::touchesVisible(w, 1, 2, 2), "an interior voxel can see into the apron");
}

static void testRebase() {
    std::printf("rebase\n");

    vis::Result r;
    r.origin[0] = 100.0; r.origin[1] = -50.0; r.origin[2] = 7.0;
    lod::StorePoint p{};
    p.x = 1.0f; p.y = 2.0f; p.z = 3.0f;
    r.voxels.push_back(p);

    const double newOrigin[3] = {90.0, -50.0, 10.0};
    std::vector<lod::StorePoint> out;
    vis::rebase(r, newOrigin, out);
    CHECK(out.size() == 1, "one voxel out");
    if (out.empty()) return;
    // Absolute position must not move: 100 + 1 == 90 + 11.
    CHECK(std::fabs(out[0].x - 11.0f) < 1e-4f, "x rebased");
    CHECK(std::fabs(out[0].y - 2.0f) < 1e-4f, "y unchanged where the origin did not move");
    CHECK(std::fabs(out[0].z - 0.0f) < 1e-4f, "z rebased");
}

static void testEndToEnd() {
    std::printf("end to end on a two-setup room\n");

    const std::string path = tmpPath("room");
    CHECK(fixture::write(path, {roomScan("west", -3.0, 2.0, 1.5),
                                roomScan("east",  3.0, 2.0, 1.5)}, 512),
          "fixture written");

    vis::Options opt;
    opt.voxelSize  = 0.25;
    opt.maxRange   = 8.0;
    opt.tileVoxels = 32;

    vis::Result frontier;
    std::string err;
    CHECK(vis::run({path}, opt, nullptr, frontier, err), err.empty() ? "ran" : err.c_str());
    if (frontier.stats.reachable == 0) return;

    CHECK(frontier.setupsUsed == 2, "both setups used");
    CHECK(frontier.scansSkipped == 0, "neither scan skipped");
    CHECK(frontier.stats.visible > 0, "the room was seen");
    CHECK(frontier.stats.occupied > 0, "its surfaces were measured");
    CHECK(frontier.stats.unknown > 0, "and there is unobserved space in range");

    CHECK(!frontier.classified, "the connectivity pass is off by default");
    CHECK(!frontier.voxels.empty(), "there are voxels to draw");

    // Asked for explicitly, the connectivity pass runs — and on this scene it
    // finds nothing, which is the result worth recording rather than one to
    // engineer away.
    //
    // The scene contains a sealed cupboard whose interior no ray enters. It is
    // still not "enclosed", because it stands on the floor and the floor
    // underneath it was never observed either: the interior connects downward
    // through unobserved floor into the unobserved ground and out to the world.
    // Nothing here is wrong. It is what enclosure by observation means, and it
    // is why the pass is off by default — in real data almost nothing is sealed
    // by observation on all sides, including the building interiors that are
    // usually the whole point of the survey.
    {
        vis::Options cls = opt;
        cls.classifyVoids = true;
        vis::Result cr;
        CHECK(vis::run({path}, cls, nullptr, cr, err), "ran with classification");
        CHECK(cr.classified, "the connectivity pass ran when asked");
        CHECK(statsEqual(cr.stats, frontier.stats),
              "classifying changes what is reported, never what was carved");
        CHECK(cr.exteriorVolume() > 100.0,
              "and nearly all the unobserved space reaches the outside world");
        CHECK(cr.voidReport.enclosed + cr.voidReport.exterior == cr.stats.unknown,
              "every unobserved voxel is accounted for as one or the other");
        CHECK(cr.voxels.size() < frontier.voxels.size(),
              "so classifying draws less than not classifying");
    }
    CHECK(!frontier.partial && !frontier.cancelled, "the whole domain was carved");

    // The room is about 240 m^3 and the visible set should be most of it.
    const double vox = opt.voxelSize * opt.voxelSize * opt.voxelSize;
    const double visibleVolume = double(frontier.stats.visible) * vox;
    CHECK(visibleVolume > 150.0 && visibleVolume < 260.0,
          "the visible volume is about the size of the room");

    // 1. The frontier changes what is drawn, not what is counted.
    vis::Options solidOpt = opt;
    solidOpt.solid = true;
    vis::Result solid;
    CHECK(vis::run({path}, solidOpt, nullptr, solid, err), err.empty() ? "ran solid" : err.c_str());
    CHECK(statsEqual(frontier.stats, solid.stats),
          "the frontier reduction does not change a single count");
    CHECK(solid.voxels.size() > frontier.voxels.size(),
          "and it really is a reduction");

    const auto fset = latticeSet(frontier), sset = latticeSet(solid);
    bool subset = true;
    for (const auto& v : fset) if (!sset.count(v)) subset = false;
    CHECK(subset, "every frontier voxel is an unknown voxel");

    // 2. Tile size changes neither the counts nor the picture.
    vis::Options wide = opt;
    wide.tileVoxels = 96;
    vis::Result wideRes;
    CHECK(vis::run({path}, wide, nullptr, wideRes, err), err.empty() ? "ran wide" : err.c_str());
    CHECK(statsEqual(frontier.stats, wideRes.stats), "same counts at a different tile size");
    CHECK(latticeSet(wideRes) == fset, "and exactly the same voxels drawn");

    // 2b. Nor does the thread count. This is the property that makes the
    // parallel run trustworthy: not "about the same answer, faster", but the
    // same answer.
    for (uint32_t n : {1u, 2u, 4u, 7u}) {
        vis::Options threaded = opt;
        threaded.threads = n;
        vis::Result tr;
        CHECK(vis::run({path}, threaded, nullptr, tr, err), err.empty() ? "ran threaded" : err.c_str());
        CHECK(statsEqual(frontier.stats, tr.stats), "same counts at every thread count");
        CHECK(latticeSet(tr) == fset, "and exactly the same voxels drawn");
        CHECK(tr.origin[0] == frontier.origin[0] && tr.origin[1] == frontier.origin[1] &&
              tr.origin[2] == frontier.origin[2], "and the same origin");
    }

    // The cap and the threads together: sampling must not become thread
    // dependent, which is where a per-worker share of the cap would go wrong.
    {
        vis::Options a = opt, b = opt;
        a.displayCap = frontier.voxels.size() / 3; a.threads = 1;
        b.displayCap = a.displayCap;               b.threads = 6;
        vis::Result ra, rb;
        CHECK(vis::run({path}, a, nullptr, ra, err), "ran capped, one thread");
        CHECK(vis::run({path}, b, nullptr, rb, err), "ran capped, six threads");
        CHECK(latticeSet(ra) == latticeSet(rb),
              "a capped run draws the same voxels however many threads carved it");
    }

    // 3. The cap samples rather than truncates.
    vis::Options capped = opt;
    capped.displayCap = frontier.voxels.size() / 4;
    vis::Result cappedRes;
    CHECK(vis::run({path}, capped, nullptr, cappedRes, err),
          err.empty() ? "ran capped" : err.c_str());
    CHECK(cappedRes.voxels.size() <= capped.displayCap, "the cap is respected");
    CHECK(cappedRes.voxels.size() > capped.displayCap / 3, "and not overshot into near-nothing");
    CHECK(cappedRes.qualified == frontier.qualified,
          "the same number of voxels qualified either way");
    CHECK(cappedRes.keptFraction < 1.0, "the result reports that it was sampled");

    const auto cset = latticeSet(cappedRes);
    bool cappedSubset = true;
    for (const auto& v : cset) if (!fset.count(v)) cappedSubset = false;
    CHECK(cappedSubset, "the sample is a subset of the full frontier");

    // A truncation would keep one contiguous corner of the site; a sample keeps
    // the whole extent. Compare the drawn bounding boxes.
    if (!cappedRes.voxels.empty() && !frontier.voxels.empty()) {
        const float fullDiag = frontier.bounds.diagonal();
        const float capDiag  = cappedRes.bounds.diagonal();
        CHECK(capDiag > 0.8f * fullDiag,
              "the sample still spans the site rather than one corner of it");
    }

    // 4. The domain. Narrowing to the surveyed extent must remove voxels from
    // the question without changing the verdict on any it keeps.
    {
        vis::Options spheres = opt;
        spheres.domain = vis::DomainMode::RangeSpheres;
        vis::Result sr;
        CHECK(vis::run({path}, spheres, nullptr, sr, err), "ran on the range spheres");
        CHECK(sr.domain.kind == carve::Domain::Kind::Unbounded, "which leaves it unbounded");

        vis::Options ext = opt;
        ext.domain = vis::DomainMode::MeasuredExtent;
        ext.domainMargin = 1.0;
        vis::Result er;
        CHECK(vis::run({path}, ext, nullptr, er, err), "ran on the surveyed extent");
        CHECK(er.domain.kind == carve::Domain::Kind::Box, "which is a box");

        // The room is 10 x 8 x 3 about the origin, plus a metre of margin.
        CHECK(er.domain.lo[0] < -5.5 && er.domain.hi[0] > 5.5, "the box spans the room in x");
        CHECK(er.domain.lo[2] < -0.5 && er.domain.hi[2] > 3.5, "and in z");
        CHECK(er.domain.hi[0] < 8.0 && er.domain.hi[1] < 7.0,
              "and does not reach out to the range sphere");

        CHECK(er.stats.reachable < sr.stats.reachable / 2,
              "narrowing the domain removes most of the question");
        CHECK(er.stats.setupTests < sr.stats.setupTests,
              "and most of the work with it");
        // Everything the clipped run kept, the open run agreed about. Compared
        // through the drawn frontier, which is the part that has to look right.
        const auto sset = latticeSet(sr), eset = latticeSet(er);
        uint64_t strays = 0;
        for (const auto& v : eset) if (!sset.count(v)) ++strays;
        CHECK(strays == 0, "every voxel the clipped run drew, the open run drew too");
        CHECK(!eset.empty(), "and it drew something");
    }

    // 5. The carver hook. No Metal here, so the mechanism is exercised with a
    // stand-in: one carver that does the tile by calling straight through, and
    // one that refuses everything. The plumbing — statistics accounted once,
    // refusals falling back to the CPU, verification comparing byte for byte —
    // is the part that would silently corrupt a run, and it is testable without
    // a GPU.
    {
        struct Counter { uint64_t accepted = 0, refused = 0; };
        Counter counter;

        vis::Options hooked = opt;
        hooked.carverUser = &counter;
        hooked.verifyCarver = true;
        hooked.carver = [](const carve::TileKey& k, const std::vector<carve::SetupView>& sv,
                           const carve::Params& pp, carve::Tile& t, carve::Stats& st,
                           void* user) -> bool {
            // Accept every other tile, so both paths run in one job.
            Counter* c = static_cast<Counter*>(user);
            if ((k.x + k.y + k.z) % 2 != 0) { ++c->refused; return false; }
            carve::carveTile(k, sv, pp, t, st);
            ++c->accepted;
            return true;
        };

        vis::Result hr;
        CHECK(vis::run({path}, hooked, nullptr, hr, err), err.empty() ? "ran hooked" : err.c_str());
        CHECK(hr.carverTiles > 0 && hr.carverRefused > 0, "both paths were taken");
        CHECK(hr.carverTiles == counter.accepted, "accepted tiles are counted");
        CHECK(hr.carverRefused == counter.refused, "and so are refusals");
        CHECK(hr.carverVoxelsCompared > 0, "verification actually compared something");
        CHECK(hr.carverDisagreements == 0,
              "a carver that calls carveTile agrees with carveTile");

        // Under verification the CPU's answer is the one kept. A mode whose
        // whole purpose is to find out whether the carver is lying must not
        // then show you what the carver said.
        vis::Options lying = opt;
        lying.verifyCarver = true;
        lying.carver = [](const carve::TileKey& k, const std::vector<carve::SetupView>& sv,
                          const carve::Params& pp, carve::Tile& t, carve::Stats& st,
                          void*) -> bool {
            carve::carveTile(k, sv, pp, t, st);
            // Corrupt it: claim everything in the domain was seen through.
            for (uint8_t& b : t.state) if (b & carve::kReachable) b = carve::kReachable |
                                                                     carve::kVisible;
            return true;
        };
        vis::Result lr;
        CHECK(vis::run({path}, lying, nullptr, lr, err), "ran with a lying carver");
        CHECK(lr.carverDisagreements > 0, "the lie is detected");
        CHECK(statsEqual(lr.stats, frontier.stats),
              "and the reported answer is the CPU's, not the lie");
        CHECK(latticeSet(lr) == fset, "including the voxels drawn");
        // The whole point: an accelerated run must produce the same answer, and
        // must not double-count the tiles it accelerated.
        CHECK(statsEqual(hr.stats, frontier.stats), "same counts through the carver");
        CHECK(hr.stats.voxels == frontier.stats.voxels, "and the same voxels examined");
        CHECK(latticeSet(hr) == fset, "and the same voxels drawn");

        // A carver that always refuses must be indistinguishable from none.
        vis::Options none = opt;
        none.carver = [](const carve::TileKey&, const std::vector<carve::SetupView>&,
                         const carve::Params&, carve::Tile&, carve::Stats&, void*) -> bool {
            return false;
        };
        vis::Result nr;
        CHECK(vis::run({path}, none, nullptr, nr, err), "ran with a refusing carver");
        CHECK(nr.carverTiles == 0 && nr.carverRefused == nr.tilesCarved,
              "every tile was refused");
        CHECK(statsEqual(nr.stats, frontier.stats), "and the answer is unchanged");
    }

    // Cancelling part way must report what it had rather than claiming success.
    vis::Result stopped;
    int seen = 0;
    vis::run({path}, opt,
             [&](const std::string& stage, uint64_t, uint64_t) {
                 if (stage != "carving") return true;
                 return ++seen < 2;
             }, stopped, err);
    CHECK(stopped.cancelled, "cancellation is reported");
    CHECK(stopped.partial, "and the result says it is partial");
    CHECK(stopped.tilesCarved < frontier.tilesCarved, "it really stopped early");
}

int main() {
    testDefaultsAgreeWithTheLibrary();
    testVoxelHash();
    testTouchesVisible();
    testRebase();
    testEndToEnd();

    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
