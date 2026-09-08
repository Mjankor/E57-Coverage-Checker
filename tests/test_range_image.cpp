// Tests for the range image — the first stage of the visibility pipeline.
//
// The properties that matter: no-return rays are identified exactly from the
// declared grid rather than inferred, the measured angular mapping recovers the
// scanner's raster, and a lookup in a given direction returns what that setup
// actually saw.

#include "e57_fixture.h"
#include "../src/range_image.h"

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
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

static std::string tmpPath(const char* name) {
    const char* d = std::getenv("E57COV_TMPDIR");
    return std::string(d ? d : "/tmp") + "/e57cov_" + name + ".e57";
}

struct Lcg {
    uint64_t s = 0x5DEECE66Dull;
    double next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return double((s >> 11) & ((1ull << 53) - 1)) / double(1ull << 53);
    }
};

// The synthetic scanner: a uniform raster, azimuth across columns and
// elevation down rows, with range a smooth function of direction.
static constexpr int    kRows = 60;
static constexpr int    kCols = 120;
static constexpr double kTau  = 6.28318530717958648;
static double azOf(int col) { return double(col) * kTau / double(kCols); }
static double elOf(int row) { return -0.5 + double(row) * (1.0 / double(kRows)); }
static double rangeOf(double az, double el) {
    return 10.0 + 3.0 * std::sin(2.0 * az) + 2.0 * std::cos(3.0 * el);
}

// Writes a structured scan. Rows at the end of the grid are omitted entirely —
// an all-sky band — and a scattered fraction elsewhere, so no-returns appear
// both as a contiguous band and as speckle, as they do in a real scan.
static bool writeGridScan(const std::string& path, int skyRows, double dropFraction,
                          uint64_t& outRecords) {
    fixture::Scan sc;
    sc.name = "raster";
    sc.hasPose = true;
    sc.q[0] = 1.0;
    sc.t[0] = 3.0; sc.t[1] = -2.0; sc.t[2] = 1.5;
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

    Lcg rng;
    outRecords = 0;
    for (int r = 0; r < kRows - skyRows; ++r) {
        for (int c = 0; c < kCols; ++c) {
            if (rng.next() < dropFraction) continue;      // scattered no-returns
            const double az = azOf(c), el = elOf(r);
            const double rr = rangeOf(az, el);
            const double ce = std::cos(el);
            sc.data[0].push_back(rr * ce * std::cos(az));
            sc.data[1].push_back(rr * ce * std::sin(az));
            sc.data[2].push_back(rr * std::sin(el));
            sc.data[3].push_back(double(r));
            sc.data[4].push_back(double(c));
            ++outRecords;
        }
    }
    return fixture::write(path, {sc}, 512);
}

// ---------------------------------------------------------------------------

static void testGridPath() {
    std::printf("range image: grid path identifies no-returns exactly\n");
    const std::string path = tmpPath("raster");
    uint64_t records = 0;
    CHECK(writeGridScan(path, 8, 0.10, records), "fixture written");

    e57::Reader r;
    std::string err;
    CHECK(r.open(path, err), err.empty() ? "opened" : err.c_str());

    rimg::Options opt;
    opt.maxRange = 45.0;
    rimg::RangeImage img;
    CHECK(rimg::build(r, 0, opt, img, err), err.empty() ? "built" : err.c_str());
    if (img.cellCount() == 0) return;

    CHECK(img.diag.usedGrid, "the declared grid was used, not the angular fallback");
    CHECK(img.rows == kRows, "rows come from indexBounds");
    CHECK(img.cols == kCols, "columns come from indexBounds");
    CHECK(img.cellCount() == size_t(kRows) * kCols, "every declared cell is present");

    // This is the whole point: the misses are known exactly, not inferred.
    CHECK(img.diag.hits == records, "every stored record became a hit");
    CHECK(img.diag.hits + img.diag.noReturns + img.diag.outsideFov == img.cellCount(),
          "every cell is a hit, a believed no-return, or an unsampled direction");
    // The fixture drops a scattered 10% of returns, which is what a scanner does
    // on dark or glancing surfaces. Those cells are empty in the file and
    // indistinguishable from sky, and they must NOT be believed: each one would
    // otherwise clear a pencil of space to maxRange straight through the scene.
    CHECK(img.diag.isolatedNoReturns > 0, "scattered drops are found");
    CHECK(img.diag.outsideFov == img.diag.isolatedNoReturns,
          "and are the only cells treated as unsampled");
    CHECK_NEAR(img.diag.fillFraction, double(records) / double(kRows * kCols), 1e-9,
               "fill fraction reported correctly");

    // The all-sky band must be visible as such, since it is the one case that
    // could instead mean part of the grid was never sampled.
    CHECK(img.diag.emptyTrailingRows == 8, "the empty band at the end of the grid is counted");
    CHECK(img.diag.emptyLeadingRows == 0, "and no band is reported where returns exist");

    // A no-return must clear all the way to maxRange, or sky does no work.
    bool skyClears = true;
    for (int c = 0; c < kCols; ++c) {
        if (img.statusAt(kRows - 1, uint32_t(c)) != rimg::Status::NoReturn) skyClears = false;
        if (std::fabs(img.rangeAt(kRows - 1, uint32_t(c)) - opt.maxRange) > 0.02) skyClears = false;
    }
    CHECK(skyClears, "sky cells are no-returns clearing to maxRange");

    CHECK(img.hasPose, "the setup pose is carried through");
    CHECK_NEAR(img.pose.t[0], 3.0, 1e-9, "pose translation preserved");
    // Observations are reported separately from the maxRange setting, so a
    // scan reaching past it is visible rather than silently clamped.
    CHECK(img.diag.furthestReturn > img.diag.nearestReturn, "return extent reported");
    CHECK(img.diag.furthestReturn < opt.maxRange,
          "this fixture's returns sit inside the clearing distance");
}

static void testMappingAndLookup() {
    std::printf("range image: measured angular mapping and lookup\n");
    const std::string path = tmpPath("raster2");
    uint64_t records = 0;
    CHECK(writeGridScan(path, 0, 0.05, records), "fixture written");

    e57::Reader r;
    std::string err;
    CHECK(r.open(path, err), err.empty() ? "opened" : err.c_str());
    rimg::RangeImage img;
    rimg::Options opt;
    CHECK(rimg::build(r, 0, opt, img, err), err.empty() ? "built" : err.c_str());
    if (!img.map.valid) { CHECK(false, "mapping should fit a uniform raster"); return; }

    // The mapping is measured from the data, never declared, so it has to
    // recover the scanner's real raster.
    CHECK_NEAR(img.map.dAzPerCol, kTau / kCols, 1e-6, "azimuth step per column recovered");
    CHECK_NEAR(img.map.dElPerRow, 1.0 / kRows, 1e-6, "elevation step per row recovered");
    CHECK_NEAR(img.map.el0, elOf(0), 1e-6, "elevation origin recovered");
    CHECK(img.map.azResidualRad < 1e-4, "azimuth fits a straight line");
    CHECK(img.map.elResidualRad < 1e-4, "elevation fits a straight line");

    // Every cell must round-trip through the mapping, including across the
    // azimuth wrap at column 0.
    bool roundTrip = true;
    for (int rr = 0; rr < kRows; rr += 7)
        for (int cc = 0; cc < kCols; cc += 5) {
            uint32_t gr, gc;
            if (!img.cellOf(azOf(cc), elOf(rr), gr, gc)) { roundTrip = false; continue; }
            if (int(gr) != rr || int(gc) != cc) roundTrip = false;
        }
    CHECK(roundTrip, "every direction maps back to the cell it came from");

    uint32_t gr, gc;
    CHECK(img.cellOf(azOf(0), elOf(3), gr, gc) && gc == 0, "azimuth 0 maps to column 0");
    CHECK(img.cellOf(kTau - 1e-9, elOf(3), gr, gc), "azimuth just under 2pi still maps");

    // A direction above the grid is outside the field of view, not a no-return.
    rimg::Status st; double range;
    CHECK(!img.sample(azOf(10), 1.4, st, range), "a direction off the grid is not sampled");
    CHECK(st == rimg::Status::OutsideFov, "and is reported as outside the field of view");

    // A hit must come back with the range that surface was actually at.
    bool ranges = true;
    for (int rr = 2; rr < kRows - 2; rr += 11)
        for (int cc = 1; cc < kCols; cc += 13) {
            if (img.statusAt(uint32_t(rr), uint32_t(cc)) != rimg::Status::Hit) continue;
            if (!img.sample(azOf(cc), elOf(rr), st, range)) { ranges = false; continue; }
            if (st != rimg::Status::Hit) { ranges = false; continue; }
            if (std::fabs(range - rangeOf(azOf(cc), elOf(rr))) > 0.02) ranges = false;
        }
    CHECK(ranges, "a hit returns the range of the surface seen in that direction");
}

static void testDownsampleIsConservative() {
    std::printf("range image: binning down keeps the nearest surface\n");
    const std::string path = tmpPath("raster3");
    uint64_t records = 0;
    CHECK(writeGridScan(path, 0, 0.0, records), "fixture written");

    e57::Reader r;
    std::string err;
    CHECK(r.open(path, err), err.empty() ? "opened" : err.c_str());

    rimg::Options full;
    rimg::RangeImage fine;
    CHECK(rimg::build(r, 0, full, fine, err), "full-resolution build");

    rimg::Options capped;
    capped.maxCells = (kRows * kCols) / 8;      // forces binning
    rimg::RangeImage coarse;
    CHECK(rimg::build(r, 0, capped, coarse, err), "binned build");

    CHECK(coarse.cellCount() <= capped.maxCells, "the cell cap is respected");
    CHECK(coarse.rows < fine.rows && coarse.cols < fine.cols, "the grid really was binned down");

    // The binning factor is chosen inside build() to meet the cap, so derive
    // it here rather than assuming one — coarse.rows == ceil(fine.rows / step).
    const uint32_t step = (fine.rows + coarse.rows - 1) / coarse.rows;
    CHECK(step >= 2, "a step was actually applied");
    CHECK((fine.rows + step - 1) / step == coarse.rows, "row count matches the derived step");
    CHECK((fine.cols + step - 1) / step == coarse.cols, "column count matches the derived step");

    // Clearing must never reach further after binning than before, or a
    // downsampled image could carve through a surface the full one kept.
    bool conservative = true;
    for (uint32_t rr = 0; rr < coarse.rows; ++rr)
        for (uint32_t cc = 0; cc < coarse.cols; ++cc) {
            if (coarse.statusAt(rr, cc) != rimg::Status::Hit) continue;
            double nearest = 1e30;
            for (uint32_t sr = rr * step; sr < std::min(fine.rows, (rr + 1) * step); ++sr)
                for (uint32_t sc = cc * step; sc < std::min(fine.cols, (cc + 1) * step); ++sc)
                    if (fine.statusAt(sr, sc) == rimg::Status::Hit)
                        nearest = std::min(nearest, fine.rangeAt(sr, sc));
            if (nearest < 1e29 && coarse.rangeAt(rr, cc) > nearest + 0.02) conservative = false;
        }
    CHECK(conservative, "a binned cell never clears further than its nearest source cell");
}

static void testRefusesWhatItCannotIdentify() {
    std::printf("range image: refuses a scan whose no-returns cannot be found\n");
    // No indexBounds and no row/column index: the misses are unknowable, and
    // guessing the field of view from the returns is the trap in DESIGN.md §4.
    fixture::Scan sc;
    sc.name = "bare";
    sc.fields = {
        {"cartesianX", e57::FieldType::FloatDouble},
        {"cartesianY", e57::FieldType::FloatDouble},
        {"cartesianZ", e57::FieldType::FloatDouble},
    };
    sc.data.assign(3, {});
    Lcg rng;
    for (int i = 0; i < 4000; ++i) {
        const double az = rng.next() * kTau, el = (rng.next() - 0.5) * 1.0;
        const double rr = rangeOf(az, el), ce = std::cos(el);
        sc.data[0].push_back(rr * ce * std::cos(az));
        sc.data[1].push_back(rr * ce * std::sin(az));
        sc.data[2].push_back(rr * std::sin(el));
    }
    const std::string path = tmpPath("bare");
    CHECK(fixture::write(path, {sc}, 512), "fixture written");

    e57::Reader r;
    std::string err;
    CHECK(r.open(path, err), "opened");
    rimg::RangeImage img;
    CHECK(!rimg::build(r, 0, rimg::Options{}, img, err),
          "a scan without grid metadata is refused rather than guessed at");
    CHECK(err.find("indexBounds") != std::string::npos, "and the reason names what is missing");
}

// The pyramid is only safe to cull with if it really bounds. A single cell that
// the pyramid reports as nearer than it is would let the carve declare a brick
// in clear view of a surface that is actually in front of it.
static void testPyramid() {
    std::printf("range image: min/max pyramid bounds every cell it covers\n");

    rimg::RangeImage im;
    im.rows = 77; im.cols = 133;          // deliberately not powers of two
    im.cells.assign(im.cellCount(), rimg::Cell{});
    Lcg rng;
    for (uint32_t r = 0; r < im.rows; ++r) {
        for (uint32_t c = 0; c < im.cols; ++c) {
            rimg::Cell& cell = im.cells[size_t(r) * im.cols + c];
            cell.rangeCm = uint16_t(100 + rng.next() * 4000);
            cell.status  = uint8_t(rng.next() < 0.2 ? rimg::Status::NoReturn
                                                    : rimg::Status::Hit);
        }
    }
    im.map.valid = true;
    im.map.dElPerRow = 0.01; im.map.dAzPerCol = 0.01;
    rimg::buildPyramid(im);
    CHECK(!im.pyramid.empty(), "a pyramid was built");
    CHECK(im.pyramid.levels.front().block == rimg::kPyramidBase, "level 0 aggregates the base");
    CHECK(im.pyramid.levels.back().rows == 1 && im.pyramid.levels.back().cols == 1,
          "the top is a single node");

    // Every query must contain the true extremes of its rectangle. It may be
    // wider — the pyramid answers with a superset — but never narrower.
    uint64_t tooNarrow = 0, tooTight = 0, missedStatus = 0, checked = 0;
    for (int trial = 0; trial < 400; ++trial) {
        const int64_t r0 = int64_t(rng.next() * im.rows);
        const int64_t r1 = std::min<int64_t>(im.rows - 1, r0 + int64_t(rng.next() * 30));
        const int64_t c0 = int64_t(rng.next() * im.cols);
        const int64_t c1 = std::min<int64_t>(im.cols - 1, c0 + int64_t(rng.next() * 30));
        const rimg::RangeSpan sp = im.span(r0, r1, c0, c1);
        if (!sp.valid) continue;
        ++checked;

        double lo = 1e300, hi = -1e300;
        uint8_t st = 0;
        for (int64_t r = r0; r <= r1; ++r) {
            for (int64_t c = c0; c <= c1; ++c) {
                const rimg::Cell& cell = im.cells[size_t(r) * im.cols + size_t(c)];
                const double v = double(cell.rangeCm) * 0.01;
                lo = std::min(lo, v); hi = std::max(hi, v);
                st |= (rimg::Status(cell.status) == rimg::Status::Hit) ? rimg::kHasHit
                                                                       : rimg::kHasNoReturn;
            }
        }
        if (sp.minRange > lo + 1e-9) ++tooNarrow;    // claimed nothing nearer than it is
        if (sp.maxRange < hi - 1e-9) ++tooNarrow;    // or nothing further
        if ((st & ~sp.statuses) != 0) ++missedStatus;
        if (sp.minRange == lo && sp.maxRange == hi) ++tooTight;
    }
    CHECK(checked > 300, "the sweep actually queried");
    CHECK(tooNarrow == 0, "no query ever reports a tighter range than the truth");
    CHECK(missedStatus == 0, "and never omits a status that occurs in the rectangle");
    // A pyramid that answered exactly every time would mean it is reading cells
    // rather than nodes, which is not what it is for.
    CHECK(tooTight < checked, "the answers really are aggregated, not per-cell");
}

// The distinction the whole no-return path rests on: a large connected region of
// empty cells is sky and clears space; an empty cell surrounded by returns is a
// dropped return and clears nothing. Nothing in the file tells them apart, so
// this is the only thing that does.
static void testSkyVersusDroppedReturns() {
    std::printf("range image: sky is believed, scattered drops are not\n");

    rimg::RangeImage im;
    im.rows = 120; im.cols = 240;
    im.cells.assign(im.cellCount(), rimg::Cell{});
    auto at = [&](uint32_t r, uint32_t c) -> rimg::Cell& {
        return im.cells[size_t(r) * im.cols + c];
    };
    // Everything a hit at 10 m...
    for (uint32_t r = 0; r < im.rows; ++r)
        for (uint32_t c = 0; c < im.cols; ++c)
            at(r, c) = rimg::Cell{1000, uint8_t(rimg::Status::Hit)};
    // ...except a band of genuine sky across the top twenty rows...
    for (uint32_t r = 0; r < 20; ++r)
        for (uint32_t c = 0; c < im.cols; ++c)
            at(r, c) = rimg::Cell{4500, uint8_t(rimg::Status::NoReturn)};
    // ...a sky region straddling the azimuth seam, which wraps...
    for (uint32_t r = 40; r < 60; ++r)
        for (uint32_t c = 0; c < im.cols; ++c)
            if (c < 8 || c >= im.cols - 8)
                at(r, c) = rimg::Cell{4500, uint8_t(rimg::Status::NoReturn)};
    // ...and scattered single drops in the middle of the returns.
    Lcg rng;
    uint32_t drops = 0;
    for (uint32_t r = 80; r < 110; ++r)
        for (uint32_t c = 20; c < 200; ++c)
            if (rng.next() < 0.05) {
                at(r, c) = rimg::Cell{4500, uint8_t(rimg::Status::NoReturn)};
                ++drops;
            }
    CHECK(drops > 100, "the fixture scattered some drops");

    rimg::Options opt;
    rimg::filterIsolatedNoReturns(im, opt);

    // The sky band survives, apart from a couple of rows of erosion at its
    // silhouette edge — the price of the rule, and paid in the safe direction.
    uint32_t skyKept = 0, skyLost = 0;
    for (uint32_t r = 0; r < 18; ++r)
        for (uint32_t c = 0; c < im.cols; ++c)
            (im.statusAt(r, c) == rimg::Status::NoReturn ? skyKept : skyLost)++;
    CHECK(skyLost == 0, "the interior of a sky region is believed");
    CHECK(skyKept == 18 * im.cols, "all of it");

    // The wrapping region is one region, not two thin ones. Without wrapping
    // the azimuth seam, its two halves would each look isolated and be lost.
    uint32_t seamKept = 0;
    for (uint32_t r = 45; r < 55; ++r) {
        if (im.statusAt(r, 0) == rimg::Status::NoReturn) ++seamKept;
        if (im.statusAt(r, im.cols - 1) == rimg::Status::NoReturn) ++seamKept;
    }
    CHECK(seamKept == 20, "a sky region crossing the azimuth seam is one region");

    // Every scattered drop is demoted. This is the one that matters: each of
    // these would otherwise clear 45 m of space through solid geometry.
    uint32_t survivingDrops = 0;
    for (uint32_t r = 80; r < 110; ++r)
        for (uint32_t c = 20; c < 200; ++c)
            if (im.statusAt(r, c) == rimg::Status::NoReturn) ++survivingDrops;
    CHECK(survivingDrops == 0, "no scattered drop is believed to have seen sky");
    CHECK(im.diag.isolatedNoReturns >= drops, "and they are counted");

    // A demoted cell says nothing rather than saying something short.
    bool rangeCleared = true;
    for (uint32_t r = 80; r < 110; ++r)
        for (uint32_t c = 20; c < 200; ++c)
            if (im.statusAt(r, c) == rimg::Status::OutsideFov && im.rangeAt(r, c) != 0.0)
                rangeCleared = false;
    CHECK(rangeCleared, "a demoted cell carries no range for anyone to mistake for a measurement");

    // Switching the filter off restores the old behaviour exactly, so a corpus
    // that genuinely has no drops can be run without the erosion.
    rimg::RangeImage plain;
    plain.rows = 20; plain.cols = 20;
    plain.cells.assign(400, rimg::Cell{4500, uint8_t(rimg::Status::NoReturn)});
    rimg::Options off;
    off.noReturnRadius = 0;
    rimg::filterIsolatedNoReturns(plain, off);
    CHECK(plain.diag.isolatedNoReturns == 0, "radius 0 switches the filter off");
}

int main() {
    std::printf("E57 Coverage Checker — range image tests\n\n");
    testGridPath();
    testPyramid();
    testSkyVersusDroppedReturns();
    testMappingAndLookup();
    testDownsampleIsConservative();
    testRefusesWhatItCannotIdentify();
    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
