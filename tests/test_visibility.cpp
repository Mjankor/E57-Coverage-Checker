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

#include "../src/report.h"
#include "../src/visibility.h"
#include "e57_fixture.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <cstdlib>
#include <memory>
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

// `dropCloserThan` models the instrument's MINIMUM range: a return closer than
// that is not recorded, which is what a real scanner does and what leaves the cell
// looking exactly like a ray that came back from nothing.
static fixture::Scan roomScan(const char* name, double sx, double sy, double sz,
                              double dropCloserThan = 0.0) {
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
            if (t < dropCloserThan) continue;      // inside the minimum range
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

static void testTouchesObserved() {
    std::printf("frontier rule\n");

    carve::Tile t;
    t.dim = 5; t.core = 3; t.apron = 1;
    t.state.assign(125, 0);
    auto at = [&](uint32_t x, uint32_t y, uint32_t z) -> uint8_t& {
        return t.state[t.index(x, y, z)];
    };

    // A lone unknown voxel with nothing around it is not on any frontier.
    at(2, 2, 2) = carve::kReachable;
    CHECK(!vis::touchesObserved(t, 2, 2, 2), "isolated unknown touches nothing");

    // A face neighbour that is visible puts it on the frontier.
    at(3, 2, 2) = carve::kReachable | carve::kVisible;
    CHECK(vis::touchesObserved(t, 2, 2, 2), "a visible face neighbour is a frontier");

    // An OCCUPIED neighbour is a frontier too, and that is a correction. A
    // measured surface between two voxels is indeed what stops there being a line
    // of sight between them — and it is also exactly where coverage stops, which
    // is what the frontier is for. Asking only for a visible neighbour made the
    // slab above a ceiling undrawable: measured surface below it, itself
    // everywhere else, not one visible face in the whole slab.
    carve::Tile u = t;
    u.state.assign(125, 0);
    auto uat = [&](uint32_t x, uint32_t y, uint32_t z) -> uint8_t& {
        return u.state[u.index(x, y, z)];
    };
    uat(2, 2, 2) = carve::kReachable;
    uat(3, 2, 2) = carve::kReachable | carve::kOccupied;
    CHECK(vis::touchesObserved(u, 2, 2, 2), "an occupied neighbour is a frontier as well");

    // Diagonals do not count — a shared edge is not a path between voxels.
    carve::Tile v = u;
    v.state.assign(125, 0);
    v.state[v.index(2, 2, 2)] = carve::kReachable;
    v.state[v.index(3, 3, 2)] = carve::kReachable | carve::kVisible;
    CHECK(!vis::touchesObserved(v, 2, 2, 2), "a diagonal neighbour is not a frontier");

    // The apron is what makes an edge voxel answerable: voxel 1 is the first
    // interior one, and its neighbour at 0 lives in the apron.
    carve::Tile w = v;
    w.state.assign(125, 0);
    w.state[w.index(1, 2, 2)] = carve::kReachable;
    w.state[w.index(0, 2, 2)] = carve::kReachable | carve::kVisible;
    CHECK(vis::touchesObserved(w, 1, 2, 2), "an interior voxel can see into the apron");
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
        // Atomic, because a carver now runs on every carving thread rather than
        // forcing the carve down to one. A plain counter here was a race the
        // moment that restriction was lifted.
        struct Counter { std::atomic<uint64_t> accepted{0}, refused{0}; };
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
        CHECK(hr.carverTiles == counter.accepted.load(), "accepted tiles are counted");
        CHECK(hr.carverRefused == counter.refused.load(), "and so are refusals");
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

// ---------------------------------------------------------------------------
// One scene, five setups, answers known in advance.
//
// Everything above tests the machinery around the method. This tests the method:
// a world with a ground plane and a building in it, observed from five places, in
// files shaped the way a real job's are — points stored scanner-local, a pose per
// scan carrying a real translation and a real rotation, and no record at all
// where the ray came back with nothing.
//
// Three claims, and together they are the whole method:
//
//   1. Space along a ray that returned nothing is CLEARED, out to the rated range
//      and no further. That is the sky, and it is what makes the answer mean
//      anything: without it every site comes back a solid ball of "unobserved".
//   2. Space in front of a surface that DID return is cleared up to that surface,
//      and space behind it is not. That is occlusion.
//   3. The five agree in one world frame, so a hole one setup cannot see is
//      filled by another.
//
// The scene is deliberately outdoor-shaped, because that is where the sky matters:
// the volume above a site is most of the range sphere, and it clears only if a
// ray that came back empty is believed.

namespace scene {

// Ground at z = 0 and a solid box. Chosen so a probe point can be classified by
// hand: inside the box nothing can see, below the ground nothing can see, and
// above both the sky is open.
// Buildings, plural, and a ground plane that stops.
//
// One box on an unbounded plane is not a site, it is a symmetry: a flat ground
// sheet is unchanged by turning it about a vertical axis or sliding it sideways,
// which are exactly the transforms a registration check has to be able to tell
// apart. A scene made mostly of ground can be reassembled wrongly and still look
// perfectly consistent. Real sites are not like that, and neither is this one.
static constexpr double kBoxLo[3] = {10.0, -8.0, 0.0};
static constexpr double kBoxHi[3] = {25.0,  8.0, 6.0};
static constexpr double kGroundHalf = 42.0;      // beyond it, sky

struct Box { double lo[3], hi[3]; };
static const Box kBoxes[4] = {
    {{ 10.0,  -8.0, 0.0}, { 25.0,   8.0, 6.0}},
    {{-32.0,  -6.0, 0.0}, {-22.0,   6.0, 9.0}},
    {{ -8.0,  18.0, 0.0}, {  0.0,  28.0, 5.0}},
    {{  2.0, -30.0, 0.0}, { 10.0, -22.0, 7.0}},
};

// Distance along a ray to the nearest surface, or -1 for nothing within `far`.
static double castRay(const double o[3], const double d[3], double far,
                      double groundZ = 0.0) {
    double best = -1.0;
    if (d[2] < -1e-9) {                       // the ground, which stops
        const double t = (groundZ - o[2]) / d[2];
        if (t > 1e-6 && t < far &&
            std::fabs(o[0] + t * d[0]) < kGroundHalf &&
            std::fabs(o[1] + t * d[1]) < kGroundHalf) best = t;
    }
    for (const Box& box : kBoxes) {
        double t0 = 0.0, t1 = far;
        bool miss = false;
        for (int k = 0; k < 3; ++k) {
            const double lo = box.lo[k] + (k == 2 ? groundZ : 0.0);
            const double hi = box.hi[k] + (k == 2 ? groundZ : 0.0);
            if (std::fabs(d[k]) < 1e-12) {
                if (o[k] < lo || o[k] > hi) { miss = true; break; }
                continue;
            }
            double a = (lo - o[k]) / d[k], b = (hi - o[k]) / d[k];
            if (a > b) std::swap(a, b);
            t0 = std::max(t0, a);
            t1 = std::min(t1, b);
        }
        if (miss || t1 < t0) continue;
        const double t = (t0 > 1e-6) ? t0 : t1;
        if (t > 1e-6 && t < far && (best < 0 || t < best)) best = t;
    }
    return best;
}

static bool insideBox(const double p[3]) {
    for (int k = 0; k < 3; ++k)
        if (p[k] < kBoxLo[k] || p[k] > kBoxHi[k]) return false;
    return true;
}

// Modest rasters: enough that a 0.5 m voxel at 20 m is several cells across,
// which is what the method needs and all it needs.
static constexpr int kSceneRows = 150, kSceneCols = 400;
static constexpr double kPiS = 3.14159265358979323846;
static double elOfRow(int r) {
    const double lo = -50.0 * kPiS / 180.0, hi = 80.0 * kPiS / 180.0;
    return lo + (hi - lo) * double(r) / double(kSceneRows - 1);
}
static double azOfCol(int c) { return kTau * double(c) / double(kSceneCols); }

struct Setup { double x, y, z, yawDeg; };

// One scan: rays cast from the setup into the scene, hits stored in the SCANNER's
// frame with the pose carrying the setup's place in the world. A ray that reaches
// nothing stores no record, which is exactly how a real file represents a ray that
// came back empty.
// `groundZ` moves the ground plane, so a fixture can put its instruments at
// exactly z = 0 and give the datum setup a pose translation of exactly zero.
//
// `preRotated` stores the points already in the world frame while still declaring
// the pose — the non-conformant case, and on a datum setup the one the per-scan
// frame test cannot see at all.
static bool writeSetup(const std::string& path, const Setup& s, double far,
                       double groundZ = 0.0, bool preRotated = false) {
    const double yaw = s.yawDeg * kPiS / 180.0;
    const double cy = std::cos(yaw), sy = std::sin(yaw);

    fixture::Scan sc;
    sc.name = "setup";
    sc.hasPose = true;
    sc.q[0] = std::cos(0.5 * yaw); sc.q[1] = 0; sc.q[2] = 0; sc.q[3] = std::sin(0.5 * yaw);
    sc.t[0] = s.x; sc.t[1] = s.y; sc.t[2] = s.z;
    sc.hasIndexBounds = true;
    sc.rowMin = 0; sc.rowMax = kSceneRows - 1;
    sc.colMin = 0; sc.colMax = kSceneCols - 1;
    sc.fields = {
        {"cartesianX",  e57::FieldType::FloatDouble},
        {"cartesianY",  e57::FieldType::FloatDouble},
        {"cartesianZ",  e57::FieldType::FloatDouble},
        {"rowIndex",    e57::FieldType::Integer, 0, kSceneRows - 1},
        {"columnIndex", e57::FieldType::Integer, 0, kSceneCols - 1},
    };
    sc.data.assign(5, {});

    const double o[3] = {s.x, s.y, s.z};
    for (int r = 0; r < kSceneRows; ++r) {
        const double el = elOfRow(r), ce = std::cos(el), se = std::sin(el);
        for (int c = 0; c < kSceneCols; ++c) {
            const double az = azOfCol(c);
            const double lx = ce * std::cos(az), ly = ce * std::sin(az), lz = se;
            const double d[3] = {cy * lx - sy * ly, sy * lx + cy * ly, lz};
            const double t = castRay(o, d, far, groundZ);
            if (t < 0) continue;                      // nothing came back
            if (preRotated) {
                // Already turned into the world frame, but still relative to the
                // setup, which is where a producer that pre-transforms leaves them.
                sc.data[0].push_back(t * d[0]);
                sc.data[1].push_back(t * d[1]);
                sc.data[2].push_back(t * d[2]);
            } else {
                sc.data[0].push_back(t * lx);         // stored scanner-local
                sc.data[1].push_back(t * ly);
                sc.data[2].push_back(t * lz);
            }
            sc.data[3].push_back(double(r));
            sc.data[4].push_back(double(c));
        }
    }
    return fixture::write(path, {sc}, 512);
}

} // namespace scene

// The image budget, and what happens when a corpus does not fit it.
//
// Two separate things are checked because they fail separately. First the
// sizing: the default has to admit a real thousand-scan job at full resolution,
// which is pure arithmetic on a raster size and needs no corpus. Second the
// reporting: when the budget is genuinely too small the rasters get coarsened,
// and coarsening is not a blurrier answer but a different one — coarse cells
// keep the nearest return landing in them, so they clear less space and
// unobserved volume comes out overstated. A run that did that without saying so
// hands back a number nobody can interpret, which is the failure this guards.
static void testImageBudgetAndWhatCoarseningCosts() {
    std::printf("the image budget, and saying when a corpus does not fit it\n");

    // A terrestrial raster, and the corpus size this is built for.
    constexpr uint64_t kRealRasterCells = 2500ull * 5280;      // 13.2 M
    const vis::Options def;
    CHECK(vis::imageCellsPerScan(def, 1000) >= kRealRasterCells,
          "the default budget holds a thousand 2500 x 5280 rasters at full resolution");
    CHECK(vis::imageCellsPerScan(def, 1) >= kRealRasterCells,
          "and one of them, obviously");
    // Not unbounded generosity: it is a budget, and it has to bite eventually.
    CHECK(vis::imageCellsPerScan(def, 100000) < kRealRasterCells,
          "a hundred thousand scans does not fit, and is not pretended to");
    // The floor holds regardless of how the division comes out.
    vis::Options tiny;
    tiny.imageBudgetBytes = 1;
    CHECK(vis::imageCellsPerScan(tiny, 1000) == tiny.minImageCells,
          "and no scan is ever given less than the floor");

    const std::string path = tmpPath("budget");
    CHECK(fixture::write(path, {roomScan("west", -3.0, 2.0, 1.5),
                                roomScan("east",  3.0, 2.0, 1.5)}, 512),
          "fixture written");

    vis::Options opt;
    opt.voxelSize  = 0.25;
    opt.maxRange   = 8.0;
    opt.tileVoxels = 32;

    vis::Result full;
    std::string err;
    CHECK(vis::run({path}, opt, nullptr, full, err), err.empty() ? "ran" : err.c_str());
    if (full.stats.reachable == 0) return;
    CHECK(full.setupsBinned == 0, "at the default budget nothing is coarsened");
    CHECK(full.worstBinStep == 1, "so the step stays at one");
    CHECK(full.imageCellsAllowed >= uint64_t(kRows) * kCols,
          "and each raster was allowed more cells than it has");

    // Now a budget too small for these rasters. The floor has to come down with
    // it, or the floor is what decides and the budget never bites.
    vis::Options squeezed = opt;
    squeezed.imageBudgetBytes = 40000;
    squeezed.minImageCells    = 4000;
    vis::Result coarse;
    CHECK(vis::run({path}, squeezed, nullptr, coarse, err),
          err.empty() ? "ran squeezed" : err.c_str());

    CHECK(coarse.setupsBinned == 2, "both rasters were coarsened");
    CHECK(coarse.worstBinStep >= 2, "by at least two declared cells per raster cell");
    CHECK(coarse.imageCellsAllowed < uint64_t(kRows) * kCols,
          "because the budget allowed fewer cells than the raster has");
    CHECK(coarse.setupsUsed == full.setupsUsed, "the same setups still contribute");
    // The cost, stated as a number rather than as a warning about sharpness: a
    // coarse cell clears to the nearest return in it, so less space is cleared
    // and more of the site reports as never observed.
    CHECK(coarse.stats.unknown > full.stats.unknown,
          "and coarsening overstates the unobserved volume, which is why it is reported");
    CHECK(coarse.stats.visible < full.stats.visible,
          "having cleared less space than the full-resolution rasters did");
}

// Shading has to vary with the shape, or it is not telling anyone anything.
//
// The failure this guards is not a crash, it is a picture that looks shaded and
// is not: a normal computed from the wrong bits, a ramp over a zero height
// range, a light pointing down an axis so that whole faces coincide. All of
// those still produce colours, and all of them still produce a flat red blob.
// So the test asks for variety, and asks for it in the two cues separately.
static void testShadingVariesWithShapeAndHeight() {
    std::printf("shaded voxels, so the frontier reads as a shape\n");

    const std::string path = tmpPath("shade");
    CHECK(fixture::write(path, {roomScan("west", -3.0, 2.0, 1.5),
                                roomScan("east",  3.0, 2.0, 1.5)}, 512),
          "fixture written");

    vis::Options opt;
    opt.voxelSize  = 0.25;
    opt.maxRange   = 8.0;
    opt.tileVoxels = 32;

    auto run = [&](uint8_t shading, vis::Result& r) {
        vis::Options o = opt;
        o.shading = shading;
        std::string err;
        CHECK(vis::run({path}, o, nullptr, r, err), err.empty() ? "ran" : err.c_str());
    };
    // How many distinct colours came back, and how much the green channel — the
    // one both cues move — spreads.
    auto spread = [](const vis::Result& r, size_t& distinct, int& gLo, int& gHi) {
        std::map<uint32_t, int> seen;
        gLo = 255; gHi = 0;
        for (const lod::StorePoint& p : r.voxels) {
            seen[(uint32_t(p.r) << 16) | (uint32_t(p.g) << 8) | p.b] = 1;
            gLo = std::min(gLo, int(p.g));
            gHi = std::max(gHi, int(p.g));
        }
        distinct = seen.size();
    };

    vis::Result flat, lit, height, both;
    run(0, flat); run(1, lit); run(2, height); run(3, both);
    if (flat.voxels.empty()) return;

    size_t nFlat, nLit, nHeight, nBoth;
    int loF, hiF, loL, hiL, loH, hiH, loB, hiB;
    spread(flat,   nFlat,   loF, hiF);
    spread(lit,    nLit,    loL, hiL);
    spread(height, nHeight, loH, hiH);
    spread(both,   nBoth,   loB, hiB);

    CHECK(nFlat == 1, "flat really is one colour, so the comparison means something");
    CHECK(nLit > 4, "lighting gives the frontier a range of tones");
    CHECK(nHeight > 4, "the height ramp gives it another");
    CHECK(nBoth >= nLit && nBoth >= nHeight, "and together they give at least as many");
    CHECK(hiL - loL > 10, "the lit range is wide enough to see");
    CHECK(hiH - loH > 10, "and so is the ramp's");

    // Same voxels either way. Shading is a colour, and a colour that changed
    // which voxels came back would be changing the answer.
    CHECK(lit.voxels.size() == flat.voxels.size(), "shading does not change how many");
    CHECK(lit.stats.unknown == flat.stats.unknown, "nor the unobserved count");
    CHECK(both.stats.unknown == flat.stats.unknown, "under either cue");
    // As sets: tiles are handed out dynamically, so the order voxels come back
    // in is the order the workers finished, not something to assert on.
    auto places = [](const vis::Result& r) {
        std::vector<std::array<float, 3>> v;
        v.reserve(r.voxels.size());
        for (const lod::StorePoint& p : r.voxels) v.push_back({p.x, p.y, p.z});
        std::sort(v.begin(), v.end());
        return v;
    };
    CHECK(places(lit) == places(both), "and not where they are");

    // Nothing goes black: an unlit face still has to read as present.
    int darkest = 255;
    for (const lod::StorePoint& p : both.voxels) darkest = std::min(darkest, int(p.r));
    CHECK(darkest > 60, "the darkest face is still clearly there");

    // Recolouring a finished result has to land exactly where carving in that
    // mode would have. It is the same shading function over the same normals, so
    // anything less than exact means the outward normals did not survive the
    // display cap in step with the voxels they belong to — which is the one way
    // this can go wrong and the one way it would not be obvious on screen.
    CHECK(flat.voxelFaces.size() == flat.voxels.size(),
          "every drawn voxel kept its outward normal");
    // Keyed by position, not by index: tiles are handed out dynamically, so two
    // runs return the same voxels in whatever order their workers finished.
    auto byPlace = [](const vis::Result& r) {
        std::map<std::array<float, 3>, uint32_t> m;
        for (const lod::StorePoint& p : r.voxels)
            m[{p.x, p.y, p.z}] = (uint32_t(p.r) << 16) | (uint32_t(p.g) << 8) | p.b;
        return m;
    };
    auto sameColours = [&](const vis::Result& a, const vis::Result& b) {
        return byPlace(a) == byPlace(b);
    };
    for (uint8_t mode = 0; mode < 4; ++mode) {
        vis::Result direct;
        run(mode, direct);
        vis::Result switched = flat;          // carved flat, then switched
        vis::recolour(switched, mode);
        CHECK(sameColours(direct, switched),
              "switching shading afterwards matches carving with it from the start");
    }
    // And back again, so the modes are a view of the answer rather than a
    // one-way edit of it.
    vis::Result there = flat;
    vis::recolour(there, 3);
    vis::recolour(there, 0);
    CHECK(sameColours(there, flat), "and switching back returns exactly where it started");
}

static void testKnownSceneFromFiveSetups() {
    std::printf("one scene, five setups, answers known in advance\n");

    const double maxRange = 45.0;
    const scene::Setup setups[5] = {
        {  0.0,   0.0, 1.6,    0.0},
        {  0.0,  14.0, 1.6,   90.0},
        {-14.0,   0.0, 1.6,  180.0},
        {  6.0, -14.0, 1.6,  270.0},
        { 32.0,  12.0, 1.6,   45.0},
    };
    std::vector<std::string> paths;
    for (int k = 0; k < 5; ++k) {
        const std::string p = tmpPath(("scene" + std::to_string(k)).c_str());
        CHECK(scene::writeSetup(p, setups[k], 55.0), "scene fixture written");
        paths.push_back(p);
    }

    // Built the way the pipeline builds them, and combined the way the carve
    // combines them — through the same primitive, so this is the method under
    // test rather than a restatement of it.
    rimg::Options ro;
    ro.maxRange = maxRange;
    std::vector<std::unique_ptr<rimg::RangeImage>> images;
    std::vector<std::unique_ptr<e57::Reader>> readers;
    std::vector<carve::SetupView> views;
    for (size_t k = 0; k < paths.size(); ++k) {
        auto rd = std::make_unique<e57::Reader>();
        std::string err;
        CHECK(rd->open(paths[k], err), err.empty() ? "opened" : err.c_str());
        auto im = std::make_unique<rimg::RangeImage>();
        const bool built = rimg::build(*rd, 0, ro, *im, err);
        CHECK(built, err.empty() ? "range image built" : err.c_str());
        if (!built) return;
        CHECK(im->map.valid, "the measured mapping is accepted");
        CHECK(im->map.roundTripFraction > 0.99, "and the scan's own points find their cells");
        images.push_back(std::move(im));
        readers.push_back(std::move(rd));
    }
    // The blind cone, as each scan decided it while being built — which is how
    // vis::run has it too. This scene has no instrument cone: its rasters reach the
    // ground at every setup, so the only empty band is sky, and a band of sky is
    // not the size of a cone about nadir. Each scan sees that in its own raster.
    {
        std::vector<rimg::RangeImage*> raw;
        for (auto& im : images) raw.push_back(im.get());
        const rimg::ConeVerdict cv = rimg::summariseBlindCones(raw, ro);
        CHECK(cv.undecided == cv.scans,
              "no scan finds a band the size of the instrument's cone, because there is none");
        for (auto& im : images)
            CHECK(im->diag.blindConeRows == 0,
                  "and no scan marks one, so the sky still clears");
    }
    for (auto& im : images) views.push_back(carve::makeSetupView(*im));

    carve::Params p;
    p.voxelSize     = 0.5;
    p.surfaceMargin = 0.5 * p.voxelSize * 1.7320508075688772;
    p.maxRange      = maxRange;

    auto verdict = [&](double x, double y, double z) {
        uint8_t bits = 0;
        for (const carve::SetupView& v : views) bits |= carve::evidenceAt(v, p, x, y, z);
        return bits;
    };

    // --- claim 1: a ray that returned nothing clears, and only so far --------
    CHECK(verdict(0, 0, 12.0) & carve::kVisible, "12 m above a setup, open sky: cleared");
    CHECK(verdict(0, 0, 30.0) & carve::kVisible, "30 m above it: still cleared");
    CHECK(verdict(-30.0, -30.0, 20.0) & carve::kVisible, "and away over open ground");
    CHECK(verdict(0, 0, 1.6 + 46.0) == 0, "past the rated range a no-return says nothing");

    // Most of the sky, not a few lucky points. This is the number that was wrong
    // in the field: low here and the whole site comes back a solid ball.
    uint64_t skyTested = 0, skyClear = 0;
    for (int i = -30; i <= 30; i += 3)
        for (int j = -30; j <= 30; j += 3)
            for (int h = 10; h <= 25; h += 3) {
                const double d = std::sqrt(double(i * i + j * j + (h - 2) * (h - 2)));
                if (d > 38.0) continue;                    // well inside the range
                ++skyTested;
                if (verdict(i, j, h) & carve::kVisible) ++skyClear;
            }
    CHECK(skyTested > 200, "the sky sample is worth something");
    CHECK(skyClear * 100 >= skyTested * 99,
          "essentially all of the open sky within range is cleared");

    // --- claim 2: a surface stops the clearing ------------------------------
    CHECK(verdict(5.0, 0.0, 1.6) & carve::kVisible, "open air in front of the building");
    CHECK(verdict(10.0, 0.0, 3.0) & carve::kOccupied, "the building's near wall is measured");

    uint64_t insideTested = 0, insideSeen = 0;
    for (double x = 11.0; x < 24.5; x += 1.0)
        for (double y = -7.0; y < 7.5; y += 1.0)
            for (double z = 0.5; z < 5.5; z += 1.0) {
                const double q[3] = {x, y, z};
                if (!scene::insideBox(q)) continue;
                ++insideTested;
                if (verdict(x, y, z) & carve::kVisible) ++insideSeen;
            }
    CHECK(insideTested > 200, "the interior sample is worth something");
    CHECK(insideSeen == 0, "nothing inside the building is ever cleared");

    uint64_t underTested = 0, underSeen = 0;
    for (int i = -20; i <= 20; i += 2)
        for (int j = -20; j <= 20; j += 2)
            for (double z = -1.5; z > -6.0; z -= 1.0) {
                ++underTested;
                if (verdict(i, j, z) & carve::kVisible) ++underSeen;
            }
    CHECK(underTested > 200, "the underground sample is worth something");
    CHECK(underSeen == 0, "nothing below the ground is ever cleared");

    // --- claim 3: the five agree in one world frame -------------------------
    // Straight down from setup 0 is outside its own field of view; setup 1
    // measures the ground there. Only a shared world frame makes that work.
    CHECK((verdict(0, 0, 0.0) & (carve::kVisible | carve::kOccupied)) != 0,
          "ground under one setup is covered by another");

    // And the whole job end to end, so the aggregate cannot be right by accident.
    vis::Options vo;
    vo.voxelSize  = 0.5;
    vo.maxRange   = maxRange;
    vo.tileVoxels = 64;
    vo.domain     = vis::DomainMode::RangeSpheres;
    vis::Result res;
    std::string err;
    CHECK(vis::run(paths, vo, nullptr, res, err), err.empty() ? "carve ran" : err.c_str());
    CHECK(res.setupsUsed == 5, "all five setups contribute");
    CHECK(res.setupsWithoutMapping == 0, "and none had its mapping refused");
    const double seen = double(res.stats.visible + res.stats.occupied);
    const double reach = double(res.stats.reachable);
    CHECK(reach > 0 && seen / reach > 0.25,
          "a real fraction of the space in range was observed, not a sliver");
    CHECK(seen / reach < 0.95, "and the ground and the building still hide plenty");
}

// A datum setup, and the one mistake nothing inside a scan can catch.
//
// The frame test compares how far the points sit from the local origin against
// how far they sit from the pose translation. That sees the translation and
// nothing else — a rotation leaves every one of those distances exactly as it
// was. So on the first setup of a job, registered as the datum with a translation
// of millimetres and a rotation of ninety degrees, the test has no information at
// all, and whichever way it falls it turns that cloud a quarter turn.
//
// Between scans it is easy: registered scans of one site describe the same
// surfaces, so the right hypothesis is the one whose cloud lands on the others.
// This builds both cases and checks that the diagnostic says so.
static void testDatumSetupWithNoTranslation() {
    std::printf("a datum setup with no translation, framed both ways\n");

    // Ground below the instruments, so the datum's pose translation is exactly
    // zero — the case the per-scan test cannot judge.
    // The datum sits at the world origin; the others are spread around the site
    // and outside the building, so between them they describe the same surfaces.
    const scene::Setup datum{0.0, 0.0, 0.0, 90.0};
    const scene::Setup others[2] = {{-12.0, 6.0, 0.0, 0.0}, {6.0, -13.0, 0.0, 200.0}};

    auto build = [&](bool datumPreRotated, std::vector<std::string>& paths) {
        paths.clear();
        const char* tag = datumPreRotated ? "framebad" : "framegood";
        const std::string a = tmpPath((std::string(tag) + "0").c_str());
        CHECK(scene::writeSetup(a, datum, 55.0, -1.6, datumPreRotated), "datum written");
        paths.push_back(a);
        for (int k = 0; k < 2; ++k) {
            const std::string b = tmpPath((std::string(tag) + std::to_string(k + 1)).c_str());
            CHECK(scene::writeSetup(b, others[k], 55.0, -1.6, false), "setup written");
            paths.push_back(b);
        }
    };

    report::Options ro;
    ro.maxRange = 45.0;

    // Stored the way the standard says: scanner-local, pose applied. Both land.
    std::vector<std::string> good;
    build(false, good);
    std::string goodText;
    report::selfTest(good, ro, goodText);
    CHECK(goodText.find("A translation that small cannot move") != std::string::npos,
          "the per-scan test admits it has no evidence about the datum");
    CHECK(goodText.find("THE POSE IS BEING APPLIED THE WRONG WAY") == std::string::npos,
          "and with the scan stored correctly, nothing is flagged");
    CHECK(goodText.find("All scans land on each other") != std::string::npos,
          "the two agree in one world frame");

    // The same datum, its points already turned into the world frame while the
    // file still declares the pose. Applying it turns the cloud a second time.
    std::vector<std::string> bad;
    build(true, bad);
    std::string badText;
    const int failures = report::selfTest(bad, ro, badText);
    CHECK(badText.find("THE POSE IS BEING APPLIED THE WRONG WAY") != std::string::npos,
          "the corpus catches what no single scan could");
    CHECK(failures >= 0, "the self-test still completes on a mis-framed corpus");
}

// The wrap's drawable skin describes the wrap, and keeps doing so when it is
// thinned to fit the display.
//
// This is a picture, so the thing to check is not how it looks but that it is
// made of the grid rather than of something near the grid. The wrap decides what
// the whole answer covers while being invisible in that answer — a wrap built
// from less than the survey measured produces voxels that look exactly like a
// survey which missed different space — so the skin is the only way to see it,
// and a skin that drifts from the grid would be worse than none.
static void testTheWrapSkinDescribesTheWrap() {
    std::printf("the wrap's skin is the wrap's own cells, thinned or not\n");

    const std::string path = tmpPath("wrapskin");
    CHECK(fixture::write(path, {roomScan("west", -3.0, 2.0, 1.5),
                                roomScan("east",  3.0, 2.0, 1.5)}, 512),
          "fixture written");

    vis::Options opt;
    opt.voxelSize  = 0.25;
    opt.maxRange   = 8.0;
    opt.tileVoxels = 32;
    opt.domain     = vis::DomainMode::Shrinkwrap;
    opt.wrapCell   = 0.25;

    vis::Result r;
    std::string err;
    CHECK(vis::run({path}, opt, nullptr, r, err), err.empty() ? "ran" : err.c_str());
    const wrap::Grid& g = r.wrapGrid;
    CHECK(!g.empty(), "a wrap was built");
    if (g.empty()) return;

    // Counted again here, from the grid, by the rule the header states: a cell
    // holding returns, or a domain cell with a face neighbour outside the domain.
    static const int kFaces[6][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
    uint64_t occ = 0, bnd = 0;
    for (int64_t z = 0; z < int64_t(g.dim[2]); ++z)
        for (int64_t y = 0; y < int64_t(g.dim[1]); ++y)
            for (int64_t x = 0; x < int64_t(g.dim[0]); ++x) {
                if (g.cellOccupied(x, y, z)) { ++occ; continue; }
                if (!g.cellInDomain(x, y, z)) continue;
                for (int i = 0; i < 6; ++i)
                    if (!g.cellInDomain(x + kFaces[i][0], y + kFaces[i][1], z + kFaces[i][2])) {
                        ++bnd;
                        break;
                    }
            }
    CHECK(occ == g.occupiedCells, "the grid's own occupancy count is the cells that are occupied");
    CHECK(r.wrapSkinCells == occ + bnd, "the skin is exactly the occupancy and the boundary");
    CHECK(r.wrapSkin.size() == r.wrapSkinCells, "and under a generous cap, none of it was dropped");

    // Every point on a cell centre of the grid's own lattice, expressed against
    // the result's origin. Off by half a cell is the mistake this catches, and
    // half a cell of drift is invisible by eye on a surface this size.
    uint64_t offLattice = 0;
    for (const lod::StorePoint& p : r.wrapSkin) {
        const double w[3] = {double(p.x) + r.origin[0], double(p.y) + r.origin[1],
                             double(p.z) + r.origin[2]};
        for (int k = 0; k < 3; ++k) {
            const double f = w[k] / g.cell - 0.5;
            if (std::fabs(f - std::round(f)) > 1e-3) { ++offLattice; break; }
        }
    }
    CHECK(offLattice == 0, "every skin point sits on a cell centre");

    // The two kinds stay distinguishable: returns warm, boundary cool. The point
    // of drawing both is telling "the wrap is wrong" from "the returns went
    // somewhere unexpected", which needs them to be different colours.
    uint64_t warm = 0, cool = 0;
    for (const lod::StorePoint& p : r.wrapSkin) {
        if (p.r > p.b) ++warm; else if (p.b > p.r) ++cool;
    }
    CHECK(warm == occ, "the warm points are the cells holding returns");
    CHECK(cool == bnd, "and the cool ones are the boundary");

    // Thinned. A cap well under the cell count keeps the surface inside it and
    // still draws something — a stride, so what is left is a lattice and still
    // reads as a surface.
    const uint64_t cap = r.wrapSkinCells / 4;
    CHECK(cap > 0, "the fixture is big enough to thin");
    vis::Result thin = r;
    vis::buildWrapSkin(thin, cap);
    CHECK(thin.wrapSkinCells == r.wrapSkinCells, "thinning does not change what the wrap IS");
    CHECK(thin.wrapSkin.size() <= cap, "the cap is respected");
    CHECK(thin.wrapSkin.size() > cap / 2, "and it is not thinned to nothing");

    // What survived is a subset of the full skin rather than new points near it.
    std::map<std::array<int64_t, 3>, int> full;
    for (const lod::StorePoint& p : r.wrapSkin)
        full[{int64_t(std::llround(p.x * 1000)), int64_t(std::llround(p.y * 1000)),
              int64_t(std::llround(p.z * 1000))}] = 1;
    uint64_t strangers = 0;
    for (const lod::StorePoint& p : thin.wrapSkin)
        if (!full.count({int64_t(std::llround(p.x * 1000)), int64_t(std::llround(p.y * 1000)),
                         int64_t(std::llround(p.z * 1000))}))
            ++strangers;
    CHECK(strangers == 0, "the thinned skin is a subset of the full one");

    // And no wrap means no skin, rather than a skin of something else.
    vis::Result none;
    vis::buildWrapSkin(none, 1u << 20);
    CHECK(none.wrapSkin.empty() && none.wrapSkinCells == 0, "no wrap, no skin");
}

// The wrap skin is expressed against Result::origin, like everything else.
//
// This is a regression test for a shift the eye caught before any assertion did:
// the skin drew a whole site origin away from the cloud, which looked like the
// wrap being built from the wrong cells and was really a frame error.
// buildWrapSkin writes each cell centre as `centre - r.origin`, and it was called
// from vis::run one statement before r.origin was assigned — so it subtracted zero
// and produced absolute world coordinates while the voxels were relative.
//
// The existing skin test missed it because its fixture sits near the world origin
// AND that origin happened to be an exact multiple of the wrap cell, so the
// absolute and relative frames both landed on the cell lattice. This one puts the
// site at UTM magnitude with a fractional offset, where neither coincidence holds.
static void testTheWrapSkinSharesTheVoxelsFrame() {
    std::printf("the wrap skin is in the same frame as the voxels\n");

    // The same room, moved by the pose alone: point data is unchanged and local,
    // so only the coordinates the site lands on are different. The offset is
    // deliberately not a multiple of the wrap cell.
    const double offX = 500123.37, offY = 6200456.11;
    const std::string path = tmpPath("wrapskin_utm");
    {
        std::vector<fixture::Scan> scans = {roomScan("west", -3.0, 2.0, 1.5),
                                            roomScan("east",  3.0, 2.0, 1.5)};
        for (fixture::Scan& sc : scans) { sc.t[0] += offX; sc.t[1] += offY; }
        CHECK(fixture::write(path, scans, 512), "fixture written");
    }

    vis::Options opt;
    opt.voxelSize  = 0.25;
    opt.maxRange   = 8.0;
    opt.tileVoxels = 32;
    opt.domain     = vis::DomainMode::Shrinkwrap;
    opt.wrapCell   = 0.25;

    vis::Result r;
    std::string err;
    CHECK(vis::run({path}, opt, nullptr, r, err), err.empty() ? "ran" : err.c_str());
    CHECK(!r.wrapGrid.empty(), "a wrap was built");
    CHECK(!r.wrapSkin.empty(), "and a skin");
    if (r.wrapSkin.empty() || r.wrapGrid.empty()) return;

    CHECK(std::fabs(r.origin[0]) > 1000.0, "the site really is at a large origin");

    // The decisive check, and the one the picture made: skin coordinates are
    // RELATIVE. Absolute ones would be about half a million.
    float far = 0;
    for (const lod::StorePoint& p : r.wrapSkin)
        far = std::max(far, std::max(std::fabs(p.x), std::max(std::fabs(p.y), std::fabs(p.z))));
    CHECK(far < 1000.0f, "the skin is expressed against the origin, not in world coordinates");

    // And it overlaps the voxels rather than sitting beside them. The wrap
    // encloses the answer, so their boxes must intersect on every axis.
    if (!r.voxels.empty()) {
        float vlo[3] = {1e30f, 1e30f, 1e30f}, vhi[3] = {-1e30f, -1e30f, -1e30f};
        float slo[3] = {1e30f, 1e30f, 1e30f}, shi[3] = {-1e30f, -1e30f, -1e30f};
        auto span = [](const std::vector<lod::StorePoint>& v, float lo[3], float hi[3]) {
            for (const lod::StorePoint& p : v) {
                const float c[3] = {p.x, p.y, p.z};
                for (int k = 0; k < 3; ++k) {
                    lo[k] = std::min(lo[k], c[k]);
                    hi[k] = std::max(hi[k], c[k]);
                }
            }
        };
        span(r.voxels, vlo, vhi);
        span(r.wrapSkin, slo, shi);
        bool overlaps = true;
        for (int k = 0; k < 3; ++k) if (shi[k] < vlo[k] || slo[k] > vhi[k]) overlaps = false;
        CHECK(overlaps, "the skin encloses the voxels rather than sitting beside them");
    }

    // Put back through the origin, every skin point is a wrap cell centre. With a
    // fractional origin this only holds in the correct frame.
    uint64_t offLattice = 0;
    for (const lod::StorePoint& p : r.wrapSkin) {
        const double w[3] = {double(p.x) + r.origin[0], double(p.y) + r.origin[1],
                             double(p.z) + r.origin[2]};
        for (int k = 0; k < 3; ++k) {
            const double f = w[k] / r.wrapGrid.cell - 0.5;
            if (std::fabs(f - std::round(f)) > 1e-2) { ++offLattice; break; }
        }
    }
    CHECK(offLattice == 0, "and lands back on the wrap's own lattice");
}

// The unknown set intersected with the wrap VOLUME — inside the shell, not near it.
//
// The case that matters is a wall. On the scanner side the wall was seen, so there
// are no unknown voxels there and the intersection correctly keeps nothing. On the
// far side the scanner saw nothing, so unknown voxels run from the wall out to the
// range limit; the wrap reaches only a buffer past the wall, so what survives is a
// buffer's thickness of them against the back of it. That is the statement worth
// drawing — coverage stopped HERE — and the rest of that column, running to the
// range limit, is the mass that hides the site.
//
// So the filter is worth its keep on a run carved over a WIDER domain than the
// wrap. A run already carved over the wrap has its voxels inside it by
// construction, and this then changes nothing; that is checked too.
static void testKeepingOnlyTheVoxelsInsideTheWrap() {
    std::printf("unknown voxels intersected with the wrap volume\n");

    const std::string path = tmpPath("insidewrap");
    CHECK(fixture::write(path, {roomScan("west", -3.0, 2.0, 1.5),
                                roomScan("east",  3.0, 2.0, 1.5)}, 512), "fixture written");
    vis::Options opt;
    opt.voxelSize  = 0.25;
    opt.maxRange   = 8.0;
    opt.tileVoxels = 32;
    opt.wrapCell   = 0.25;

    // The wrap, from a shrinkwrap run.
    vis::Options wopt = opt;
    wopt.domain = vis::DomainMode::Shrinkwrap;
    vis::Result wrapRun;
    std::string err;
    CHECK(vis::run({path}, wopt, nullptr, wrapRun, err), err.empty() ? "wrap run" : err.c_str());
    CHECK(!wrapRun.wrapGrid.empty(), "a wrap was built");
    if (wrapRun.wrapGrid.empty()) return;

    // A run carved over the range spheres — everything within maxRange of any
    // setup, which reaches well past the wrap. This is the one carrying the
    // blanket that runs out to the range limit.
    vis::Options bopt = opt;
    bopt.domain = vis::DomainMode::RangeSpheres;
    vis::Result box;
    CHECK(vis::run({path}, bopt, nullptr, box, err), err.empty() ? "box run" : err.c_str());
    CHECK(!box.voxels.empty(), "the box run produced voxels");
    if (box.voxels.empty()) return;

    const std::vector<lod::StorePoint> before = box.voxels;
    // The grid is indexed on a global lattice in WORLD coordinates, so it
    // transplants between runs whatever origin each of them chose — the filter
    // puts each voxel back into world before asking.
    box.wrapGrid = wrapRun.wrapGrid;

    const uint64_t kept = vis::keepVoxelsInsideWrap(box);
    CHECK(kept == box.voxels.size(), "the count returned is the count kept");
    CHECK(kept > 0, "the wrap does contain some of the unknown set");
    CHECK(kept <= before.size(), "it only ever drops");
    // How much it drops depends on the scene, and on this enclosed-room fixture
    // the answer is little or nothing: the room IS the wrap, so a domain drawn
    // round the same room barely reaches past it. The reduction this exists for
    // needs a wall with open space behind it, which is an outdoor shape. What is
    // checked here is that the filter is exactly right about which side each voxel
    // falls on, in both directions.
    std::printf("      (kept %llu of %zu)\n", (unsigned long long)kept, before.size());
    if (box.voxelFaces.size() || !box.voxels.empty())
        CHECK(box.voxelFaces.empty() || box.voxelFaces.size() == box.voxels.size(),
              "shading stays attached to the voxel it was computed for");

    // Every survivor was there before, and every survivor is genuinely inside the
    // wrap; everything dropped is genuinely outside it. Both directions, because a
    // filter that kept too much would look the same at a glance.
    std::set<std::array<int64_t, 3>> was;
    for (const lod::StorePoint& p : before)
        was.insert({int64_t(std::llround(p.x * 1000)), int64_t(std::llround(p.y * 1000)),
                    int64_t(std::llround(p.z * 1000))});
    std::set<std::array<int64_t, 3>> now;
    uint64_t strangers = 0, outside = 0;
    for (const lod::StorePoint& p : box.voxels) {
        const std::array<int64_t, 3> k{int64_t(std::llround(p.x * 1000)),
                                       int64_t(std::llround(p.y * 1000)),
                                       int64_t(std::llround(p.z * 1000))};
        if (!was.count(k)) ++strangers;
        now.insert(k);
        if (!box.wrapGrid.contains(double(p.x) + box.origin[0], double(p.y) + box.origin[1],
                                   double(p.z) + box.origin[2])) ++outside;
    }
    CHECK(strangers == 0, "no voxel is invented, only dropped");
    CHECK(outside == 0, "every survivor is inside the wrap");

    uint64_t wronglyDropped = 0;
    for (const lod::StorePoint& p : before) {
        const std::array<int64_t, 3> k{int64_t(std::llround(p.x * 1000)),
                                       int64_t(std::llround(p.y * 1000)),
                                       int64_t(std::llround(p.z * 1000))};
        if (now.count(k)) continue;
        if (box.wrapGrid.contains(double(p.x) + box.origin[0], double(p.y) + box.origin[1],
                                  double(p.z) + box.origin[2])) ++wronglyDropped;
    }
    CHECK(wronglyDropped == 0, "and nothing inside the wrap was dropped");

    // A run already carved over the wrap is unchanged by this, its voxels being
    // inside the wrap by construction.
    {
        vis::Result again = wrapRun;
        const size_t n = again.voxels.size();
        CHECK(vis::keepVoxelsInsideWrap(again) == n, "a wrap run is already inside its wrap");
    }

    // And a Result with no wrap keeps everything, there being nothing to intersect.
    {
        vis::Result none;
        vis::Options n2 = bopt;
        CHECK(vis::run({path}, n2, nullptr, none, err), "ran without a wrap");
        const size_t n = none.voxels.size();
        CHECK(vis::keepVoxelsInsideWrap(none) == n, "no wrap, nothing dropped");
    }
}

// A carver runs on every carving thread, and the answer does not change.
//
// The carve used to force itself to one thread whenever a carver was set, because
// the GPU carver kept one state buffer on a singleton and two tiles could not
// share it. That left the CPU idle for the whole of each blocking dispatch and the
// device idle between them. The scratch is per-thread now and the restriction is
// gone — so the thing to prove is that the result is the same however many threads
// carve, since a carver that raced would produce a plausible-looking answer with
// tiles quietly wrong.
static void testACarverRunsOnEveryThreadAndAgrees() {
    std::printf("a threaded carver gives the same answer as a serial one\n");

    const std::string path = tmpPath("threadedcarver");
    CHECK(fixture::write(path, {roomScan("west", -3.0, 2.0, 1.5),
                                roomScan("east",  3.0, 2.0, 1.5)}, 512), "fixture written");

    vis::Options opt;
    opt.voxelSize  = 0.25;
    opt.maxRange   = 8.0;
    opt.tileVoxels = 16;          // small, so there are many tiles to spread
    // A stand-in for the GPU: carves the tile properly, and keeps per-call scratch
    // of its own so a race would show up as a wrong answer rather than a crash.
    opt.carver = [](const carve::TileKey& k, const std::vector<carve::SetupView>& sv,
                    const carve::Params& pp, carve::Tile& t, carve::Stats& st,
                    void*) -> bool {
        carve::carveTile(k, sv, pp, t, st);
        return true;
    };

    auto digest = [](const vis::Result& r) {
        std::vector<std::array<int64_t, 3>> v;
        v.reserve(r.voxels.size());
        for (const lod::StorePoint& p : r.voxels)
            v.push_back({int64_t(std::llround(p.x * 1000)), int64_t(std::llround(p.y * 1000)),
                         int64_t(std::llround(p.z * 1000))});
        std::sort(v.begin(), v.end());
        return v;
    };

    std::string err;
    vis::Options one = opt;
    one.threads = 1;
    vis::Result serial;
    CHECK(vis::run({path}, one, nullptr, serial, err), err.empty() ? "serial run" : err.c_str());
    CHECK(!serial.voxels.empty(), "the serial run produced voxels");
    const auto want = digest(serial);

    for (unsigned n : {2u, 3u, 4u, 8u}) {
        vis::Options many = opt;
        many.threads = n;
        vis::Result r;
        CHECK(vis::run({path}, many, nullptr, r, err), err.empty() ? "threaded run" : err.c_str());
        CHECK(digest(r) == want, "the same voxels at any thread count, with a carver");
        CHECK(r.stats.unknown == serial.stats.unknown, "and the same unobserved count");
        CHECK(r.carverTiles == serial.carverTiles, "with every tile still going to the carver");
    }

    // And the statistics the carver reports survive the threading: setup tests are
    // accumulated per worker and summed, so they must not depend on the split.
    {
        vis::Options many = opt;
        many.threads = 4;
        vis::Result r;
        CHECK(vis::run({path}, many, nullptr, r, err), "ran");
        CHECK(r.stats.setupTests == serial.stats.setupTests,
              "the work reported is the same work, however it was divided");
    }
}

// A negative margin pulls the question inside the walls.
//
// On an indoor job the surveyed extent hugs the building, so a positive margin
// asks about a couple of metres past every wall — space nothing was ever going to
// see, which comes back as a blanket of unobserved voxels wrapped round the
// outside. A negative margin removes it at the source instead of filtering it out
// of the answer afterwards.
//
// The two regions honour the sign differently, because the shapes differ, and both
// are checked here: a box shrinks, and a wrap takes the magnitude as its buffer
// and the sign as a request to drop the outside.
static void testANegativeMarginAsksAboutLess() {
    std::printf("a negative margin shrinks the question\n");

    const std::string path = tmpPath("negmargin");
    CHECK(fixture::write(path, {roomScan("west", -3.0, 2.0, 1.5),
                                roomScan("east",  3.0, 2.0, 1.5)}, 512), "fixture written");
    vis::Options base;
    base.voxelSize  = 0.25;
    base.maxRange   = 8.0;
    base.tileVoxels = 32;

    std::string err;

    // The box. Bigger margin, bigger question; negative, smaller.
    double volumes[3] = {0, 0, 0};
    const double margins[3] = {2.0, 0.0, -1.0};
    for (int i = 0; i < 3; ++i) {
        vis::Options o = base;
        o.domain = vis::DomainMode::MeasuredExtent;
        o.domainMargin = margins[i];
        vis::Result r;
        CHECK(vis::run({path}, o, nullptr, r, err), err.empty() ? "ran" : err.c_str());
        volumes[i] = r.domainVolume;
    }
    CHECK(volumes[0] > volumes[1], "a positive margin asks about more than none");
    CHECK(volumes[1] > volumes[2], "and a negative one asks about less");
    CHECK(volumes[2] > 0.0, "but still about something");

    // A margin more negative than the site cannot invert the box. An inverted box
    // would carve nothing and report it as complete coverage, which is the wrong
    // direction to be wrong in.
    {
        vis::Options o = base;
        o.domain = vis::DomainMode::MeasuredExtent;
        o.domainMargin = -1000.0;
        vis::Result r;
        CHECK(vis::run({path}, o, nullptr, r, err), err.empty() ? "ran" : err.c_str());
        CHECK(r.domainVolume >= 0.0, "the domain never has negative volume");
        CHECK(r.stats.unknown == 0 || r.domainVolume > 0.0,
              "nothing is reported unobserved in a domain that holds nothing");
    }

    // The wrap. A negative margin means what interiorOnly means, so the two agree.
    {
        vis::Options neg = base;
        neg.domain = vis::DomainMode::Shrinkwrap;
        neg.wrapCell = 0.25;
        neg.domainMargin = -2.0;
        vis::Result rneg;
        CHECK(vis::run({path}, neg, nullptr, rneg, err), err.empty() ? "ran" : err.c_str());

        vis::Options tick = base;
        tick.domain = vis::DomainMode::Shrinkwrap;
        tick.wrapCell = 0.25;
        tick.domainMargin = 2.0;
        tick.wrapInteriorOnly = true;
        vis::Result rtick;
        CHECK(vis::run({path}, tick, nullptr, rtick, err), err.empty() ? "ran" : err.c_str());

        CHECK(rneg.wrapGrid.interiorOnly, "a negative margin asks for interior only");
        CHECK(rneg.wrapGrid.buffer == 2.0, "with the magnitude as the buffer");
        CHECK(rneg.wrapGrid.domainCells == rtick.wrapGrid.domainCells,
              "so it is the same wrap the tick box produces");
        CHECK(rneg.stats.unknown == rtick.stats.unknown, "and the same answer");

        // And it really does drop something relative to the same buffer kept both
        // sides of every surface.
        vis::Options both = tick;
        both.wrapInteriorOnly = false;
        vis::Result rboth;
        CHECK(vis::run({path}, both, nullptr, rboth, err), "ran");
        CHECK(rneg.wrapGrid.domainCells < rboth.wrapGrid.domainCells,
              "the outside really is left out");
    }
}

// A setup parked inside its own minimum range of a wall does not carve through it.
//
// This is the end-to-end guard for what the fans were: a stairwell scan with the
// handrail at the instrument's elbow, a setup tucked against a lift shaft, a
// tripod half a metre from a wall. The surface fills a large solid angle of the
// raster and every cell of it comes back empty, because the instrument cannot
// measure anything closer than 0.45 m. Believed, each of those cells clears a
// pencil to the rated range straight through the wall — ninety times the distance
// to the thing in the way — so the space beyond it comes back OBSERVED.
//
// Run twice over the same file, with the test on and off, because the difference
// is the whole point and a single run cannot show it.
static void testASetupAgainstAWallDoesNotCarveThroughIt() {
    std::printf("a setup inside its minimum range of a wall does not carve through it\n");

    // One setup 0.3 m from the wall at x = 5, and a second across the room so the
    // domain covers the whole interior. The fixture drops every return closer than
    // 0.45 m, which is what the instrument does.
    const std::string path = tmpPath("tooclose");
    CHECK(fixture::write(path, {roomScan("wall",  4.7, 0.0, 1.5, 0.45),
                                roomScan("far",  -3.0, 2.0, 1.5, 0.45)}, 512),
          "fixture written");

    vis::Options base;
    base.voxelSize  = 0.1;
    base.maxRange   = 20.0;          // so a believed cell reaches well past the wall
    base.tileVoxels = 32;
    base.domain     = vis::DomainMode::MeasuredExtent;
    base.domainMargin = 3.0;         // ask about the space beyond the wall
    base.solid      = true;          // the volume, not its frontier

    // Voxels in the slab just beyond the wall, at the height of the setup that is
    // parked against it: the space the pencils would have cleared.
    auto beyondTheWall = [](const vis::Result& r) {
        uint64_t n = 0;
        for (const lod::StorePoint& p : r.voxels) {
            const double x = double(p.x) + r.origin[0];
            const double y = double(p.y) + r.origin[1];
            const double z = double(p.z) + r.origin[2];
            if (x < 5.3 || x > 7.5) continue;
            if (std::fabs(y) > 2.0 || z < 0.5 || z > 2.5) continue;
            ++n;
        }
        return n;
    };

    std::string err;
    vis::Options off = base;
    off.minRange = 0.0;                       // the test switched off
    vis::Result without;
    CHECK(vis::run({path}, off, nullptr, without, err), err.empty() ? "ran" : err.c_str());

    vis::Result with;
    CHECK(vis::run({path}, base, nullptr, with, err), err.empty() ? "ran" : err.c_str());

    CHECK(with.setupsTooClose == 1, "one setup is inside its minimum range of something");
    CHECK(without.setupsTooClose == 0, "and with the test off, none is reported");
    CHECK(with.tooCloseCells > 0, "cells were demoted");
    CHECK(with.tooCloseNearest > 0 && with.tooCloseNearest <= 0.60,
          "on a bordering range at or inside the bar the minimum range sets");

    // The space beyond the wall. Unobserved either way in truth — nothing ever
    // looked at it — so the run with the test off has to report LESS of it,
    // because it cleared some away through the wall.
    const uint64_t unknownWith    = beyondTheWall(with);
    const uint64_t unknownWithout = beyondTheWall(without);
    CHECK(unknownWith > 0, "the space beyond the wall is unobserved");
    CHECK(unknownWithout < unknownWith,
          "and believing the too-close cells had cleared some of it, through the wall");
    CHECK(without.stats.visible > with.stats.visible,
          "which is the same thing counted the other way round: it called more space seen");
}

// An indoor corpus does not clear a cone through its own roof and floor.
//
// This is the end-to-end guard for the defect the cross-section showed: wedges of
// space immediately above the ceiling and below the slab reported as OBSERVED,
// which would need the scanner to see through both.
//
// The chain was: a survey conducted entirely inside a building has a fixed
// unsampled band at each end of every raster — the instrument's own mount one
// way, the ceiling at a constant height the other. The corpus test asked which
// ONE end was fixed, found both, called that undecidable, and undecidable meant
// "believe both" — every scan's blind bands became rays that had seen through to
// the rated range. One cone up and one down per setup, and with enough setups the
// space above the roof and below the floor is entirely cleared.
//
// Both bands are marked unsampled now. What this checks is the consequence: the
// space beyond the ceiling stays unobserved.
// roomScan with a fixed unsampled band at EACH end of the raster, which is what a
// survey conducted entirely inside a building actually looks like: the
// instrument's own mount blocks one end and, level in a room of constant height,
// the ceiling band at the other is the same in every scan. The rows are declared
// in indexBounds and simply carry no returns.
static fixture::Scan bandedRoomScan(const char* name, double sx, double sy, double sz,
                                    int leadBand, int trailBand) {
    fixture::Scan sc = roomScan(name, sx, sy, sz);
    const long long rows = kRows + leadBand + trailBand;
    sc.rowMin = 0; sc.rowMax = rows - 1;
    for (auto& f : sc.fields)
        if (f.name == "rowIndex") { f.minimum = 0; f.maximum = rows - 1; }
    // Shift every row index up by the leading band, leaving both bands empty.
    for (double& r : sc.data[3]) r += double(leadBand);
    return sc;
}

static void testAnIndoorCorpusCannotSeeThroughItsOwnRoof() {
    std::printf("an indoor corpus does not clear through its roof or floor\n");

    // Four setups, each with the same unsampled band at both ends.
    const std::string path = tmpPath("indoorcone");
    CHECK(fixture::write(path, {bandedRoomScan("a", -3.0,  2.0, 1.5, 40, 30),
                                bandedRoomScan("b",  3.0,  2.0, 1.5, 40, 30),
                                bandedRoomScan("c", -3.0, -2.0, 1.5, 41, 31),
                                bandedRoomScan("d",  3.0, -2.0, 1.5, 40, 30)}, 512),
          "fixture written");

    vis::Options opt;
    opt.voxelSize  = 0.25;
    opt.maxRange   = 20.0;          // well past the room, so a believed cone reaches
    opt.tileVoxels = 32;
    opt.domain     = vis::DomainMode::MeasuredExtent;
    opt.domainMargin = 4.0;         // ask about space beyond the ceiling and the floor
    opt.solid      = true;          // every unobserved voxel, not just the frontier —
                                    // a region entirely unknown has no frontier inside
                                    // it, so the frontier cannot measure this
    vis::Result r;
    std::string err;
    CHECK(vis::run({path}, opt, nullptr, r, err), err.empty() ? "ran" : err.c_str());
    CHECK(r.setupsUsed == 4, "all four setups contributed");
    // Each raster has a band at both ends, both the size of the instrument's cone,
    // and nothing separates them — so neither is believed, in every scan, decided
    // by each scan on its own.
    CHECK(r.coneVerdict.bothEnds == r.coneVerdict.scans,
          "every scan marks both ends unsampled, from its own geometry");
    CHECK(r.coneVerdict.bandsBelieved == 0, "and leaves no band believed as a view");

    // The consequence, which is the whole point: space beyond the ceiling and
    // below the floor is still unobserved. A believed band would have cleared a
    // cone through both, straight up and straight down from every setup.
    uint64_t aboveCeiling = 0, belowFloor = 0;
    for (const lod::StorePoint& p : r.voxels) {
        const double x = double(p.x) + r.origin[0];
        const double y = double(p.y) + r.origin[1];
        const double z = double(p.z) + r.origin[2];
        if (std::fabs(x) > 4.0 || std::fabs(y) > 3.0) continue;   // over the setups
        if (z > 3.5) ++aboveCeiling;
        if (z < -0.5) ++belowFloor;
    }
    CHECK(aboveCeiling > 100, "space above the ceiling is still unobserved");
    CHECK(belowFloor > 100, "and so is space below the floor");

    // And it is DRAWN the way the app draws by default: the frontier, not the
    // solid volume.
    //
    // This is the second half of the same report and it was the half still
    // missing. The slab of unobserved space above a ceiling is bounded below by
    // the ceiling the beam stopped on, and above and to the sides by more of
    // itself or by the edge of the domain. Not one of its voxels has a face
    // neighbour that some setup saw THROUGH — so under a frontier rule that asks
    // only for a visible neighbour the whole slab is undrawable, and a top-down
    // view of an indoor survey shows bare roof with nothing over it. What did
    // show was the comb of thin cleared pencils where rays escaped through
    // openings, with frontier between them: radial fans outside the building.
    //
    // Coverage stops AT the ceiling, and that is the frontier. The rule has to ask
    // whether a neighbour was OBSERVED — seen through or measured — not only seen
    // through. The connectivity path has always asked it that way
    // (voids::touchesObserved); this is the plain path catching up.
    {
        vis::Options frontier = opt;
        frontier.solid = false;
        vis::Result fr;
        CHECK(vis::run({path}, frontier, nullptr, fr, err), err.empty() ? "ran" : err.c_str());
        // Above the band the ceiling itself occupies: the returns are at z = 3 and
        // a voxel within half a diagonal of them is occupied rather than unknown,
        // so the first unobserved layer sits just over 3.2. That layer is the whole
        // test — under the old rule it was empty, because the only thing beneath it
        // is the measured ceiling.
        uint64_t roof = 0;
        for (const lod::StorePoint& p : fr.voxels) {
            const double x = double(p.x) + fr.origin[0];
            const double y = double(p.y) + fr.origin[1];
            const double z = double(p.z) + fr.origin[2];
            if (std::fabs(x) > 4.0 || std::fabs(y) > 3.0) continue;
            if (z > 3.2) ++roof;
        }
        CHECK(roof > 100, "the unobserved space above the ceiling is drawn, not just counted");
        // A sheet, not the slab: the frontier is still a reduction, and one layer
        // against the back of the ceiling is what makes the roof read as covered.
        CHECK(fr.voxels.size() < r.voxels.size(),
              "and it is still a reduction — fewer voxels than the solid volume");
    }
}

int main() {
    testDefaultsAgreeWithTheLibrary();
    testVoxelHash();
    testTouchesObserved();
    testRebase();
    testEndToEnd();
    testImageBudgetAndWhatCoarseningCosts();
    testShadingVariesWithShapeAndHeight();
    testTheWrapSkinDescribesTheWrap();
    testTheWrapSkinSharesTheVoxelsFrame();
    testACarverRunsOnEveryThreadAndAgrees();
    testANegativeMarginAsksAboutLess();
    testASetupAgainstAWallDoesNotCarveThroughIt();
    testAnIndoorCorpusCannotSeeThroughItsOwnRoof();
    testKeepingOnlyTheVoxelsInsideTheWrap();
    testKnownSceneFromFiveSetups();
    testDatumSetupWithNoTranslation();

    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
