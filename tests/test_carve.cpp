// Tests for the visibility carve — the stage that turns range images into a
// statement about space.
//
// The properties that matter, in the order the pipeline depends on them:
//
//   1. One setup's verdict on one point is right: in front of a measured
//      surface is visible, on it is occupied, behind it is silence, and beyond
//      the rated range is silence too. A no-return ray clears to maxRange and
//      not a millimetre further.
//   2. Setups combine by OR. A voxel occluded from one setup but seen from
//      another is visible; one occluded from both is a candidate void.
//   3. The domain is the union of the range spheres — tiles are created for
//      exactly that and nothing beyond it.
//   4. Tiling is bookkeeping, not method. Carving a volume as one big tile and
//      as many small ones must agree voxel for voxel. This is the property the
//      Metal kernel will inherit: if the answer depended on how space was
//      partitioned, no amount of agreement with the reference would mean
//      anything.

#include "../src/carve.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
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

static constexpr double kTau = 6.28318530717958648;

// A synthetic range image: a uniform raster covering all azimuths and a band of
// elevation, every cell a hit at `range` metres. Built by hand rather than
// through the E57 reader so the carve is tested against a surface whose
// position is known exactly, not one recovered from a decoder.
static rimg::RangeImage shellImage(double range, double maxRange,
                                   uint32_t rows = 90, uint32_t cols = 180) {
    rimg::RangeImage im;
    im.rows = rows;
    im.cols = cols;
    im.cells.assign(im.cellCount(),
                    rimg::Cell{uint16_t(range * 100.0 + 0.5), uint8_t(rimg::Status::Hit)});
    im.map.az0 = 0.0;
    im.map.dAzPerCol = kTau / double(cols);
    im.map.el0 = -0.8;
    im.map.dElPerRow = 1.6 / double(rows);
    im.map.azResidualRad = 0.0;
    im.map.elResidualRad = 0.0;
    im.map.valid = true;
    im.hasPose = false;
    im.diag.usedGrid = true;
    im.diag.hits = im.cellCount();
    im.diag.furthestReturn = range;
    im.diag.nearestReturn  = range;
    (void)maxRange;
    return im;
}

// Turns a band of columns into no-returns, exactly as build() leaves them:
// status NoReturn with the range set to the clearing distance.
static void makeNoReturnColumns(rimg::RangeImage& im, uint32_t c0, uint32_t c1,
                                double maxRange) {
    for (uint32_t r = 0; r < im.rows; ++r) {
        for (uint32_t c = c0; c < c1 && c < im.cols; ++c) {
            const size_t i = size_t(r) * im.cols + c;
            im.cells[i].status  = uint8_t(rimg::Status::NoReturn);
            im.cells[i].rangeCm = uint16_t(maxRange * 100.0 + 0.5);
        }
    }
}

static void setPose(rimg::RangeImage& im, double tx, double ty, double tz,
                    double yaw = 0.0) {
    im.hasPose = true;
    im.pose.q[0] = std::cos(yaw * 0.5);
    im.pose.q[1] = 0.0;
    im.pose.q[2] = 0.0;
    im.pose.q[3] = std::sin(yaw * 0.5);
    im.pose.t[0] = tx; im.pose.t[1] = ty; im.pose.t[2] = tz;
}

// ---------------------------------------------------------------------------

static void testOneSetupAlongARay() {
    std::printf("one setup along a ray\n");

    carve::Params p;
    p.voxelSize = 0.05;
    p.maxRange  = 45.0;

    rimg::RangeImage im = shellImage(10.0, p.maxRange);
    const carve::SetupView s = carve::makeSetupView(im);

    CHECK(carve::evidenceAt(s, p, 5.0, 0, 0) == carve::kVisible,
          "in front of the surface is visible");
    CHECK(carve::evidenceAt(s, p, 9.5, 0, 0) == carve::kVisible,
          "just in front is still visible");
    CHECK(carve::evidenceAt(s, p, 10.0, 0, 0) == carve::kOccupied,
          "on the surface is occupied, not visible");
    // The margin is half a voxel diagonal, so a voxel whose centre sits just
    // short of the measurement still contains the surface. Without this the
    // carve eats the very surfaces it is measuring against.
    CHECK(p.surfaceMargin > 0.04 && p.surfaceMargin < 0.05, "margin is half a voxel diagonal");
    CHECK(carve::evidenceAt(s, p, 10.0 - 0.5 * p.surfaceMargin, 0, 0) == carve::kOccupied,
          "inside the margin on the near side is occupied, not visible");
    CHECK(carve::evidenceAt(s, p, 10.0 + 0.5 * p.surfaceMargin, 0, 0) == carve::kOccupied,
          "and on the far side too");
    CHECK(carve::evidenceAt(s, p, 10.0 - 2.0 * p.surfaceMargin, 0, 0) == carve::kVisible,
          "but clear of the margin it is line of sight again");
    CHECK(carve::evidenceAt(s, p, 20.0, 0, 0) == 0,
          "behind the surface this setup says nothing");
    CHECK(carve::evidenceAt(s, p, 44.0, 0, 0) == 0,
          "still behind the surface at the far end of the range");

    // Direction, not just the +x axis.
    CHECK(carve::evidenceAt(s, p, 0, -6.0, 0) == carve::kVisible,
          "visible along -y too");
    CHECK(carve::evidenceAt(s, p, 5.0, 0, 3.0) == carve::kVisible,
          "visible on a raised sight line, inside the elevation band");

    // Outside the raster's elevation band the scanner never looked, which is
    // silence rather than emptiness.
    CHECK(carve::evidenceAt(s, p, 0.5, 0, 5.0) == 0,
          "above the field of view clears nothing");
    CHECK(carve::evidenceAt(s, p, 0.5, 0, -5.0) == 0,
          "below the field of view clears nothing");
}

static void testNoReturnClearsToMaxRange() {
    std::printf("no-return rays\n");

    carve::Params p;
    p.maxRange = 45.0;

    rimg::RangeImage im = shellImage(10.0, p.maxRange);
    // Column 0 is azimuth 0, i.e. the +x direction.
    makeNoReturnColumns(im, 0, 1, p.maxRange);
    const carve::SetupView s = carve::makeSetupView(im);

    CHECK(carve::evidenceAt(s, p, 5.0, 0, 0) == carve::kVisible,
          "near the setup on a no-return ray is visible");
    CHECK(carve::evidenceAt(s, p, 20.0, 0, 0) == carve::kVisible,
          "past where neighbouring rays hit, but this one came back empty");
    CHECK(carve::evidenceAt(s, p, 44.9, 0, 0) == carve::kVisible,
          "just inside the rated range");
    CHECK(carve::evidenceAt(s, p, 45.5, 0, 0) == 0,
          "past the rated range a no-return establishes nothing");
    CHECK(carve::evidenceAt(s, p, 100.0, 0, 0) == 0,
          "far past the rated range likewise");

    // A tighter maxRange than the image was built with has to win: the setting
    // bounds the evidence, it does not merely annotate it.
    carve::Params tight = p;
    tight.maxRange = 20.0;
    CHECK(carve::evidenceAt(s, tight, 19.0, 0, 0) == carve::kVisible,
          "inside the tightened range");
    CHECK(carve::evidenceAt(s, tight, 21.0, 0, 0) == 0,
          "a tightened maxRange shortens the clearing");

    // And the other direction: an image built with a short clearing distance
    // does not start clearing further because the run was configured for a
    // longer one. The image records what was established, not what was wanted.
    rimg::RangeImage shortIm = shellImage(10.0, 12.0);
    makeNoReturnColumns(shortIm, 0, 1, 12.0);
    const carve::SetupView ss = carve::makeSetupView(shortIm);
    CHECK(carve::evidenceAt(ss, p, 11.0, 0, 0) == carve::kVisible,
          "inside what the image cleared");
    CHECK(carve::evidenceAt(ss, p, 30.0, 0, 0) == 0,
          "beyond what the image cleared, even though maxRange is 45");
}

static void testOcclusionCombinesAcrossSetups() {
    std::printf("two setups\n");

    carve::Params p;
    p.maxRange = 45.0;

    // A at the origin sees a shell at 10 m; B twenty metres along +x sees one
    // at 8 m. Between them the two shells leave a band neither setup reaches.
    rimg::RangeImage a = shellImage(10.0, p.maxRange);
    rimg::RangeImage b = shellImage(8.0, p.maxRange);
    setPose(b, 20.0, 0.0, 0.0);

    std::vector<carve::SetupView> setups{carve::makeSetupView(a), carve::makeSetupView(b)};
    CHECK_NEAR(setups[1].origin[0], 20.0, 1e-12, "B's origin comes from its pose");

    auto both = [&](double x, double y, double z) {
        uint8_t bits = 0;
        for (const carve::SetupView& s : setups) bits |= carve::evidenceAt(s, p, x, y, z);
        return bits;
    };

    CHECK(carve::evidenceAt(setups[0], p, 15.0, 0, 0) == 0,
          "occluded from A");
    CHECK(carve::evidenceAt(setups[1], p, 15.0, 0, 0) == carve::kVisible,
          "but in clear view of B");
    CHECK(both(15.0, 0, 0) == carve::kVisible,
          "visibility from any setup is visibility");

    // 11 m from A is behind A's shell; 9 m from B is behind B's shell.
    CHECK(both(11.0, 0, 0) == 0,
          "occluded from both setups: a candidate void");

    // A voxel one setup measures a surface in and another sees through: both
    // bits, because forcing a precedence would lose one of two real facts.
    const uint8_t onBShell = both(12.0, 0, 0);
    CHECK((onBShell & carve::kOccupied) != 0, "B measured its surface there");
    CHECK((onBShell & carve::kVisible) == 0, "and A cannot see through it");
}

static void testPoseInversion() {
    std::printf("pose inversion\n");

    carve::Params p;
    rimg::RangeImage im = shellImage(10.0, p.maxRange);
    setPose(im, 3.0, -4.0, 1.5, 0.7);
    const carve::SetupView s = carve::makeSetupView(im);

    CHECK_NEAR(s.origin[0], 3.0, 1e-12, "origin x");
    CHECK_NEAR(s.origin[1], -4.0, 1e-12, "origin y");
    CHECK_NEAR(s.origin[2], 1.5, 1e-12, "origin z");

    // The setup's own position maps to the scanner-frame origin.
    double x = 3.0, y = -4.0, z = 1.5;
    s.worldToScanner.apply(x, y, z);
    CHECK_NEAR(x, 0.0, 1e-12, "world-to-scanner puts the setup at the origin (x)");
    CHECK_NEAR(y, 0.0, 1e-12, "world-to-scanner puts the setup at the origin (y)");
    CHECK_NEAR(z, 0.0, 1e-12, "world-to-scanner puts the setup at the origin (z)");

    // A rotated setup still measures range from itself, so its shell sits at
    // 10 m in every direction regardless of yaw.
    CHECK(carve::evidenceAt(s, p, 3.0 + 5.0, -4.0, 1.5) == carve::kVisible,
          "yaw does not move the shell in +x");
    CHECK(carve::evidenceAt(s, p, 3.0, -4.0 + 5.0, 1.5) == carve::kVisible,
          "nor in +y");
    CHECK(carve::evidenceAt(s, p, 3.0 + 12.0, -4.0, 1.5) == 0,
          "and behind it is still behind it");
}

static void testDomainIsTheUnionOfSpheres() {
    std::printf("tile domain\n");

    carve::Params p;
    p.voxelSize  = 0.5;
    p.tileVoxels = 8;          // 4 m tiles
    p.maxRange   = 6.0;

    rimg::RangeImage a = shellImage(3.0, p.maxRange);
    rimg::RangeImage b = shellImage(3.0, p.maxRange);
    setPose(b, 25.0, 0.0, 0.0);   // far enough that the two spheres are disjoint

    std::vector<carve::SetupView> setups{carve::makeSetupView(a), carve::makeSetupView(b)};
    const std::vector<carve::TileKey> keys = carve::tilesForSetups(setups, p);
    CHECK(!keys.empty(), "the domain is not empty");

    bool sorted = true, unique = true, allReachable = true;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (i && !(keys[i - 1] < keys[i])) { sorted = false; unique = false; }
        if (carve::setupsForTile(keys[i], setups, p).empty()) allReachable = false;
    }
    CHECK(sorted, "tiles come back sorted, so a run is resumable by index");
    CHECK(unique, "and without duplicates where the two setups overlap");
    CHECK(allReachable, "no tile is created that no setup can reach");

    // Nothing between the two setups: the gap costs nothing, which is what
    // makes a sprawling site affordable.
    bool gapEmpty = true;
    for (const carve::TileKey& k : keys) {
        const double cx = (double(k.x) + 0.5) * p.tileMetres();
        if (cx > 8.0 && cx < 17.0) gapEmpty = false;
    }
    CHECK(gapEmpty, "no tiles in the empty stretch between setups");

    // Every point inside a range sphere lands in a tile that was created.
    std::map<std::string, bool> present;
    for (const carve::TileKey& k : keys) {
        present[std::to_string(k.x) + "," + std::to_string(k.y) + "," + std::to_string(k.z)] = true;
    }
    uint64_t missing = 0, tested = 0;
    for (int i = -12; i <= 12; ++i) {
        for (int j = -12; j <= 12; ++j) {
            for (int k = -12; k <= 12; ++k) {
                const double x = i * 0.5, y = j * 0.5, z = k * 0.5;
                if (x * x + y * y + z * z > p.maxRange * p.maxRange) continue;
                ++tested;
                const int64_t tx = int64_t(std::floor(x / p.tileMetres()));
                const int64_t ty = int64_t(std::floor(y / p.tileMetres()));
                const int64_t tz = int64_t(std::floor(z / p.tileMetres()));
                if (!present.count(std::to_string(tx) + "," + std::to_string(ty) + "," +
                                   std::to_string(tz))) ++missing;
            }
        }
    }
    CHECK(tested > 1000, "the coverage sweep actually sampled the sphere");
    CHECK(missing == 0, "every point in reach falls in a created tile");
}

// The whole carve, collected into a global voxel map so two tilings can be
// compared voxel for voxel.
struct Collected {
    std::map<std::array<int64_t, 3>, uint8_t> voxels;
    carve::Stats stats;
};

static bool collectSink(const carve::Tile& t, void* user) {
    Collected* c = static_cast<Collected*>(user);
    for (uint32_t z = 0; z < t.dim; ++z) {
        for (uint32_t y = 0; y < t.dim; ++y) {
            for (uint32_t x = 0; x < t.dim; ++x) {
                // The lattice is global and anchored at the world origin, so a
                // voxel's index does not depend on the tile that carried it.
                int64_t g[3];
                t.globalIndex(x, y, z, g);
                c->voxels[{g[0], g[1], g[2]}] = t.state[t.index(x, y, z)];
            }
        }
    }
    return true;
}

static void testTilingDoesNotChangeTheAnswer() {
    std::printf("tiling invariance\n");

    rimg::RangeImage a = shellImage(3.0, 4.0);
    rimg::RangeImage b = shellImage(2.5, 4.0);
    // Offset off the tile lattice on purpose, and yawed, so nothing lines up
    // conveniently with either tiling.
    setPose(b, 4.3, 1.7, -0.9, 0.9);
    makeNoReturnColumns(a, 40, 60, 4.0);

    std::vector<carve::SetupView> setups{carve::makeSetupView(a), carve::makeSetupView(b)};

    carve::Params coarse;
    coarse.voxelSize     = 0.25;
    coarse.surfaceMargin = 0.5 * 0.25 * 1.7320508075688772;
    coarse.maxRange      = 4.0;
    coarse.tileVoxels    = 24;      // 6 m tiles

    carve::Params fine = coarse;
    fine.tileVoxels = 8;            // 2 m tiles

    Collected cc, cf;
    cc.stats = carve::carveAll(setups, coarse, collectSink, &cc);
    cf.stats = carve::carveAll(setups, fine, collectSink, &cf);

    CHECK(cc.voxels.size() > 100000, "the coarse tiling carved a real volume");
    CHECK(cf.voxels.size() < cc.voxels.size(),
          "smaller tiles fit the domain more tightly, so they touch fewer voxels");

    // Every voxel the fine tiling carved must agree with the coarse one.
    uint64_t disagree = 0, absent = 0;
    for (const auto& kv : cf.voxels) {
        auto it = cc.voxels.find(kv.first);
        if (it == cc.voxels.end()) { ++absent; continue; }
        if (it->second != kv.second) ++disagree;
    }
    CHECK(absent == 0, "the coarse tiling covers everything the fine one does");
    CHECK(disagree == 0, "and agrees with it voxel for voxel");

    // The voxels only the coarse tiling touched are outside every range sphere,
    // so they must be silent. If any carried evidence, the domain would depend
    // on the tiling.
    uint64_t strayEvidence = 0;
    for (const auto& kv : cc.voxels) {
        if (cf.voxels.count(kv.first)) continue;
        if (kv.second != 0) ++strayEvidence;
    }
    CHECK(strayEvidence == 0, "tiles that only clip the domain carry no evidence");

    // Statistics about the site rather than about the work must match exactly.
    CHECK(cc.stats.reachable == cf.stats.reachable, "same reachable count");
    CHECK(cc.stats.visible == cf.stats.visible, "same visible count");
    CHECK(cc.stats.occupied == cf.stats.occupied, "same occupied count");
    CHECK(cc.stats.unknown == cf.stats.unknown, "same unknown count");
    CHECK(cc.stats.setupTests == cf.stats.setupTests, "same number of setup tests");
    CHECK(cc.stats.voxels > cf.stats.voxels,
          "but the coarse tiling did more work to get there");

    // And the answer is not trivially all-one-thing.
    CHECK(cc.stats.visible > 0, "something is visible");
    CHECK(cc.stats.occupied > 0, "something is occupied");
    CHECK(cc.stats.unknown > 0, "something is unknown — there is a deliverable");
    // Visible and occupied are independent predicates, so their counts can
    // overlap; the domain still brackets them.
    CHECK(cc.stats.reachable <= cc.stats.unknown + cc.stats.visible + cc.stats.occupied,
          "every reachable voxel is unknown or carries at least one bit");
    CHECK(cc.stats.reachable >= cc.stats.unknown + std::max(cc.stats.visible, cc.stats.occupied),
          "and no bit is counted outside the domain");
}

// The fast path and the oracle must agree bit for bit. This is the check every
// later optimisation re-runs: reordering, culling and early exits are all
// legal only insofar as this holds.
static void testFastPathMatchesReference() {
    std::printf("fast path against the reference\n");

    rimg::RangeImage a = shellImage(3.0, 6.0);
    rimg::RangeImage b = shellImage(2.4, 6.0);
    rimg::RangeImage c = shellImage(4.1, 6.0);
    // Offset off the brick and tile lattices, and yawed, so no boundary lines
    // up conveniently with anything.
    setPose(b, 3.3, 1.7, -0.9, 0.9);
    setPose(c, -2.1, -3.4, 1.2, -2.2);
    makeNoReturnColumns(a, 40, 60, 6.0);
    makeNoReturnColumns(c, 100, 150, 6.0);
    // With the pyramid built, carveTile takes the culling path; without it,
    // every brick falls through. Both have to match the reference, so the
    // comparison runs twice.
    rimg::buildPyramid(a);
    rimg::buildPyramid(b);

    std::vector<carve::SetupView> setups{carve::makeSetupView(a), carve::makeSetupView(b),
                                         carve::makeSetupView(c)};

    // Deliberately not a multiple of the brick size, so the partial bricks at
    // the far faces are exercised.
    for (uint32_t tileVoxels : {8u, 13u, 32u}) {
        for (uint32_t apron : {0u, 1u}) {
          for (int clipped = 0; clipped < 2; ++clipped) {
           for (carve::EarlyOut eo : {carve::EarlyOut::None, carve::EarlyOut::Saturated}) {
            carve::Params p;
            p.voxelSize     = 0.25;
            p.surfaceMargin = 0.5 * 0.25 * 1.7320508075688772;
            p.maxRange      = 6.0;
            p.tileVoxels    = tileVoxels;
            p.apron         = apron;
            p.earlyOut      = eo;
            if (clipped) {
                // A box that slices through the scene rather than containing
                // it, so tiles and bricks land inside, outside and straddling.
                p.domain.kind = carve::Domain::Kind::Box;
                p.domain.lo[0] = -3.1; p.domain.hi[0] = 4.4;
                p.domain.lo[1] = -2.6; p.domain.hi[1] = 3.9;
                p.domain.lo[2] = -2.2; p.domain.hi[2] = 2.7;
            }

            const std::vector<carve::TileKey> keys = carve::tilesForSetups(setups, p);
            CHECK(!keys.empty(), "there are tiles to compare");

            uint64_t stateDiffs = 0;
            carve::Stats fastStats, refStats;
            carve::Tile fast, ref;
            for (const carve::TileKey& k : keys) {
                carve::carveTile(k, setups, p, fast, fastStats);
                carve::carveTileReference(k, setups, p, ref, refStats);
                if (fast.state != ref.state) ++stateDiffs;
            }
            CHECK(stateDiffs == 0, "every tile's state matches the reference exactly");
            CHECK(fastStats.reachable == refStats.reachable, "same reachable count");
            CHECK(fastStats.visible == refStats.visible, "same visible count");
            CHECK(fastStats.occupied == refStats.occupied, "same occupied count");
            CHECK(fastStats.unknown == refStats.unknown, "same unknown count");
            CHECK(fastStats.voxels == refStats.voxels, "same voxels examined");
            // setupTests measures work, so it is the one statistic allowed to
            // differ — and under Saturated it had better be smaller, or the
            // early exit is not doing anything.
            if (eo == carve::EarlyOut::None)
                CHECK(fastStats.setupTests == refStats.setupTests,
                      "with no early exit, even the work matches");
            else
                CHECK(fastStats.setupTests < refStats.setupTests,
                      "the early exit really skips setups");
            // Not a trivial pass: there has to be something to disagree about.
            CHECK(fastStats.visible > 0 && fastStats.occupied > 0 && fastStats.unknown > 0,
                  "the comparison covered all three outcomes");
           }
          }
        }
    }
}

// AnyEvidence trades the visible and occupied counts for speed. What it must
// not trade is the answer: the set of voxels nobody observed.
static void testAnyEvidenceKeepsTheUnknownSet() {
    std::printf("early exit at the first evidence\n");

    rimg::RangeImage a = shellImage(3.0, 6.0);
    rimg::RangeImage b = shellImage(2.4, 6.0);
    rimg::RangeImage c = shellImage(4.1, 6.0);
    setPose(b, 3.3, 1.7, -0.9, 0.9);
    setPose(c, -2.1, -3.4, 1.2, -2.2);
    makeNoReturnColumns(a, 40, 60, 6.0);
    for (rimg::RangeImage* im : {&a, &b, &c}) rimg::buildPyramid(*im);
    std::vector<carve::SetupView> setups{carve::makeSetupView(a), carve::makeSetupView(b),
                                         carve::makeSetupView(c)};

    carve::Params exact;
    exact.voxelSize = 0.25;
    exact.surfaceMargin = 0.5 * 0.25 * 1.7320508075688772;
    exact.maxRange = 6.0;
    exact.tileVoxels = 16;
    carve::Params fast = exact;
    fast.earlyOut = carve::EarlyOut::AnyEvidence;

    carve::Stats es, fs;
    carve::Tile et, ft;
    uint64_t unknownDiffs = 0, reachDiffs = 0;
    for (const carve::TileKey& k : carve::tilesForSetups(setups, exact)) {
        carve::carveTile(k, setups, exact, et, es);
        carve::carveTile(k, setups, fast,  ft, fs);
        for (size_t i = 0; i < et.state.size(); ++i) {
            const uint8_t e = et.state[i], f = ft.state[i];
            if ((e == carve::kReachable) != (f == carve::kReachable)) ++unknownDiffs;
            if ((e & carve::kReachable) != (f & carve::kReachable)) ++reachDiffs;
        }
    }
    CHECK(unknownDiffs == 0, "exactly the same voxels come out unknown");
    CHECK(reachDiffs == 0, "and exactly the same voxels are in the domain");
    CHECK(fs.unknown == es.unknown, "so the unknown count is exact");
    CHECK(fs.reachable == es.reachable, "as is the reachable count");
    CHECK(fs.setupTests < es.setupTests, "with less work");
    // Lower bounds, not counts — which is why this is not the default.
    CHECK(fs.visible <= es.visible, "visible becomes a lower bound");
    CHECK(fs.occupied <= es.occupied, "and so does occupied");
    CHECK(fs.visible + fs.occupied < es.visible + es.occupied,
          "and they really are lower, not incidentally equal");
}

static void testDomainClipping() {
    std::printf("domain clipping\n");

    carve::Domain d;
    CHECK(d.contains(1e9, -1e9, 0), "an unbounded domain contains everything");
    const double any[3] = {5, 5, 5}, any2[3] = {6, 6, 6};
    CHECK(d.testBox(any, any2) == carve::Overlap::Full, "and every box is fully inside it");

    d.kind = carve::Domain::Kind::Box;
    d.lo[0] = 0; d.lo[1] = 0; d.lo[2] = 0;
    d.hi[0] = 10; d.hi[1] = 10; d.hi[2] = 10;
    CHECK(d.contains(5, 5, 5), "inside");
    CHECK(!d.contains(5, 5, 11), "outside");
    CHECK(d.contains(0, 0, 0) && d.contains(10, 10, 10), "the faces are inside");

    const double inLo[3] = {1, 1, 1},   inHi[3] = {2, 2, 2};
    const double outLo[3] = {11, 1, 1}, outHi[3] = {12, 2, 2};
    const double strLo[3] = {9, 1, 1},  strHi[3] = {11, 2, 2};
    CHECK(d.testBox(inLo, inHi) == carve::Overlap::Full, "a contained box is Full");
    CHECK(d.testBox(outLo, outHi) == carve::Overlap::None, "a disjoint box is None");
    CHECK(d.testBox(strLo, strHi) == carve::Overlap::Partial, "a straddling box is Partial");

    // Full must mean it: every point of such a box has to be contained, or the
    // carve would skip the per-voxel test on voxels that are actually outside.
    CHECK(d.contains(inLo[0], inLo[1], inLo[2]) && d.contains(inHi[0], inHi[1], inHi[2]),
          "Full implies the corners are contained");

    // Clipping removes voxels; it never changes the verdict on one it keeps.
    rimg::RangeImage a = shellImage(3.0, 8.0);
    rimg::buildPyramid(a);
    std::vector<carve::SetupView> setups{carve::makeSetupView(a)};

    carve::Params open_;
    open_.voxelSize = 0.25;
    open_.surfaceMargin = 0.5 * 0.25 * 1.7320508075688772;
    open_.maxRange = 8.0;
    open_.tileVoxels = 16;
    carve::Params clip = open_;
    clip.domain.kind = carve::Domain::Kind::Box;
    for (int k = 0; k < 3; ++k) { clip.domain.lo[k] = -2.0; clip.domain.hi[k] = 2.0; }

    carve::Stats os, cs;
    carve::Tile ot, ct;
    uint64_t disagree = 0, kept = 0;
    for (const carve::TileKey& k : carve::tilesForSetups(setups, clip)) {
        carve::carveTile(k, setups, open_, ot, os);
        carve::carveTile(k, setups, clip, ct, cs);
        for (uint32_t z = 0; z < ct.dim; ++z)
            for (uint32_t y = 0; y < ct.dim; ++y)
                for (uint32_t x = 0; x < ct.dim; ++x) {
                    const uint8_t c = ct.state[ct.index(x, y, z)];
                    if (!c) continue;
                    ++kept;
                    if (c != ot.state[ot.index(x, y, z)]) ++disagree;
                }
    }
    CHECK(kept > 1000, "the clipped run kept a real number of voxels");
    CHECK(disagree == 0, "and gave each of them the same verdict as the unclipped run");
    CHECK(cs.reachable < os.reachable, "while asking about fewer of them");
}

static void testEarlyStop() {
    std::printf("early stop\n");

    carve::Params p;
    p.voxelSize  = 0.5;
    p.tileVoxels = 8;
    p.maxRange   = 6.0;

    rimg::RangeImage a = shellImage(3.0, p.maxRange);
    std::vector<carve::SetupView> setups{carve::makeSetupView(a)};

    struct Counter { int seen = 0; };
    Counter c;
    const carve::Stats st = carve::carveAll(setups, p,
        [](const carve::Tile&, void* u) {
            Counter* cc = static_cast<Counter*>(u);
            return ++cc->seen < 3;
        }, &c);
    CHECK(c.seen == 3, "the sink stopped the run after three tiles");
    CHECK(st.voxels == 3ull * 8 * 8 * 8, "and only those tiles were carved");
}

int main() {
    testOneSetupAlongARay();
    testNoReturnClearsToMaxRange();
    testOcclusionCombinesAcrossSetups();
    testPoseInversion();
    testDomainIsTheUnionOfSpheres();
    testTilingDoesNotChangeTheAnswer();
    testFastPathMatchesReference();
    testDomainClipping();
    testAnyEvidenceKeepsTheUnknownSet();
    testEarlyStop();

    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
