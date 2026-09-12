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
#include <memory>
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
static constexpr double kPi   = 3.14159265358979323846;
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
          "every cell is a hit, a no-return, or an unsampled direction");
    // A ray either returned or it did not. An empty cell is a ray that came back
    // with nothing, and it clears along its path whatever it passed through —
    // sky, a window, or a surface too dark to register. The only cells not
    // believed are the unsampled band under the tripod, and this fixture's
    // empty band is at the sky end.
    CHECK(img.diag.isolatedNoReturns == 0,
          "no cell is second-guessed: the neighbourhood filter is off by default");
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
    im.map = rimg::uniformMapping(im.rows, im.cols, -0.4, 0.01, 0.0, 0.01);
    im.map.valid = true;
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
    std::printf("range image: the optional drop filter, when asked for\n");

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

    // The filter is off by default now — a ray either returned or it did not.
    // It is switched on here because the machinery is still wanted for scans
    // that genuinely drop, and it still has to work when asked for.
    rimg::Options opt;
    opt.noReturnRadius   = 2;
    opt.noReturnFraction = 0.75;
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
    rimg::Options off;      // the default
    rimg::filterIsolatedNoReturns(plain, off);
    CHECK(off.noReturnRadius == 0, "and radius 0 is the default");
    CHECK(plain.diag.isolatedNoReturns == 0, "radius 0 switches the filter off");
}

// The failure the round-trip check exists to catch, and the one the residual
// alone cannot.
//
// Many terrestrial scanners turn the head through 180 degrees while the mirror
// sweeps a full 360. Each column is then a complete vertical circle: rows in the
// first half look forward at descending elevation, rows in the second half look
// BACKWARD — azimuth plus 180 degrees — at ascending elevation. Elevation
// against row is a triangle wave, not a line.
//
// Fitting a line to it produces a mapping that is confidently wrong. A voxel
// overhead is sent to a row holding ground and comes back unknown; a voxel
// inside a building is sent to a row holding sky and gets cleared to maxRange.
// That is sky uncarved and building interiors carved away, which is exactly what
// the field reported.
static void testDoubleCoveredMirrorIsRefused() {
    std::printf("range image: a double-covered mirror sweep is refused\n");

    const int rows = 200, cols = 100;
    const std::string path = tmpPath("doublecover");

    fixture::Scan sc;
    sc.name = "double cover";
    sc.hasPose = true;
    sc.q[0] = 1.0;
    sc.hasIndexBounds = true;
    sc.rowMin = 0; sc.rowMax = rows - 1;
    sc.colMin = 0; sc.colMax = cols - 1;
    sc.fields = {
        {"cartesianX",  e57::FieldType::FloatDouble},
        {"cartesianY",  e57::FieldType::FloatDouble},
        {"cartesianZ",  e57::FieldType::FloatDouble},
        {"rowIndex",    e57::FieldType::Integer, 0, rows - 1},
        {"columnIndex", e57::FieldType::Integer, 0, cols - 1},
    };
    sc.data.assign(5, {});
    for (int r = 0; r < rows; ++r) {
        // Mirror angle sweeps a full turn over the rows.
        const double theta = double(r) * kTau / double(rows);
        for (int c = 0; c < cols; ++c) {
            // Head sweeps only half a turn over the columns.
            const double head = double(c) * kPi / double(cols);
            double az, el;
            if (theta <= kPi * 0.5 || theta > kPi * 1.5) {
                az = head;
                el = (theta <= kPi * 0.5) ? (kPi * 0.5 - theta) : (kPi * 2.5 - theta) - kPi;
            } else {
                az = head + kPi;              // looking out the back
                el = theta - kPi * 1.5;
            }
            el = std::max(-1.4, std::min(1.4, el));
            const double rr = 10.0;
            const double ce = std::cos(el);
            sc.data[0].push_back(rr * ce * std::cos(az));
            sc.data[1].push_back(rr * ce * std::sin(az));
            sc.data[2].push_back(rr * std::sin(el));
            sc.data[3].push_back(double(r));
            sc.data[4].push_back(double(c));
        }
    }
    CHECK(fixture::write(path, {sc}, 512), "double-cover fixture written");

    e57::Reader rd;
    std::string err;
    CHECK(rd.open(path, err), err.empty() ? "opened" : err.c_str());

    rimg::Options opt;
    rimg::RangeImage img;
    const bool built = rimg::build(rd, 0, opt, img, err);
    CHECK(built, "the image still builds — the raster is there, the model is not");
    if (!built) return;

    CHECK(img.map.roundTripFraction >= 0.0,
          "the round-trip check ran, even though the residual had already condemned the fit");
    CHECK(img.map.roundTripFraction < 0.9,
          "and most of the scan's own points do not land on their own cell");
    CHECK(!img.map.valid,
          "so the mapping is refused rather than used to produce confident nonsense");
    CHECK(!img.diag.note.empty(), "and the reason is recorded");

    // Refused means every lookup says OutsideFov, which clears nothing. A carve
    // over this scan reports everything unobserved — useless, but honest, and
    // very obviously wrong rather than quietly wrong.
    rimg::Status st; double range;
    CHECK(!img.sample(0.3, 0.2, st, range), "a refused mapping answers nothing");
    CHECK(st == rimg::Status::OutsideFov, "and says so");
}

// The counterpart: an ordinary single-sweep raster must sail through, or the
// check is just a way of rejecting good data.
static void testOrdinaryRasterRoundTrips() {
    std::printf("range image: an ordinary raster round-trips\n");
    const std::string path = tmpPath("roundtrip");
    uint64_t records = 0;
    CHECK(writeGridScan(path, 4, 0.05, records), "fixture written");

    e57::Reader rd;
    std::string err;
    CHECK(rd.open(path, err), err.empty() ? "opened" : err.c_str());
    rimg::Options opt;
    rimg::RangeImage img;
    CHECK(rimg::build(rd, 0, opt, img, err), err.empty() ? "built" : err.c_str());
    CHECK(img.map.valid, "the mapping is accepted");
    CHECK(img.map.roundTripFraction > 0.99,
          "and essentially every point lands back on its own cell");
}

// The raster real instruments actually produce, and the one the linear model got
// wrong in the field: a sweep that runs past a full turn, on an elevation axis
// that is not a uniform step.
//
// Both faults are in here at once because both were in the data. The azimuth
// covers 364.5 degrees over 600 columns, so the last ~7 columns repeat the
// bearings of the first ~7 — the same 1.25% excess the five scans measured had,
// which sent every column past the modulo-2pi fold 55 to 80 columns away from
// where it belonged. The elevation carries a full-period sinusoid on top of its
// step, putting it ten cells from its own best-fit line. And the first 40 rows
// hold no returns at all, as a blind cone leaves 590 of 2500 in a real scan, so
// the gap filling has to produce elevations for rows nothing ever measured.
//
// What has to hold is not "the fit is good" — a fit is not the deliverable. It is
// that a direction resolves to the cell that looked that way, that the fractional
// coordinates the brick culling bounds with agree with the cell the lookup
// returns, and that the rows nothing measured still map somewhere.
static void testSweepPastATurnAndNonUniformRows() {
    std::printf("range image: a sweep past a full turn, on a non-uniform elevation axis\n");

    const int rows = 300, cols = 600, emptyRows = 40;
    // 364.5 degrees, decreasing with column — the sign the real scans have.
    const double azSweep = 364.5 * kPi / 180.0;
    const double dAz     = -azSweep / double(cols);
    const double azStart = -0.82;
    // Ten cells of deviation from a straight line, and still monotonic: the step
    // is 0.00917 rad and the sinusoid's steepest contribution is 0.00192.
    const double elLo = -1.4, elStep = 2.75 / double(rows), elAmp = 10.0 * elStep;
    auto azAt = [&](int c) { return azStart + dAz * double(c); };
    auto elAt = [&](int r) {
        return elLo + elStep * double(r) + elAmp * std::sin(kTau * double(r) / double(rows));
    };

    const std::string path = tmpPath("pastaturn");
    fixture::Scan sc;
    sc.name = "past a turn";
    sc.hasPose = true;
    sc.q[0] = 1.0;
    sc.hasIndexBounds = true;
    sc.rowMin = 0; sc.rowMax = rows - 1;
    sc.colMin = 0; sc.colMax = cols - 1;
    sc.fields = {
        {"cartesianX",  e57::FieldType::FloatDouble},
        {"cartesianY",  e57::FieldType::FloatDouble},
        {"cartesianZ",  e57::FieldType::FloatDouble},
        {"rowIndex",    e57::FieldType::Integer, 0, rows - 1},
        {"columnIndex", e57::FieldType::Integer, 0, cols - 1},
    };
    sc.data.assign(5, {});
    for (int r = emptyRows; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const double az = azAt(c), el = elAt(r);
            const double rr = 9.0 + 2.0 * std::sin(3.0 * az) + std::cos(2.0 * el);
            const double ce = std::cos(el);
            sc.data[0].push_back(rr * ce * std::cos(az));
            sc.data[1].push_back(rr * ce * std::sin(az));
            sc.data[2].push_back(rr * std::sin(el));
            sc.data[3].push_back(double(r));
            sc.data[4].push_back(double(c));
        }
    }
    CHECK(fixture::write(path, {sc}, 512), "fixture written");

    e57::Reader rd;
    std::string err;
    CHECK(rd.open(path, err), err.empty() ? "opened" : err.c_str());
    rimg::Options opt;
    rimg::RangeImage img;
    const bool built = rimg::build(rd, 0, opt, img, err);
    CHECK(built, err.empty() ? "built" : err.c_str());
    if (!built) return;

    // The two faults are present and reported, not smoothed over.
    CHECK(std::fabs(img.map.azSpanRad) > kTau,
          "the sweep is measured as running past a full turn");
    CHECK(img.map.elResidualRad > opt.maxMappingResidualRad,
          "and the elevation axis is measured as not uniform");
    CHECK(img.diag.note.find("not a uniform raster") != std::string::npos,
          "which is said out loud rather than left in a field nobody reads");

    // And it is accepted anyway, because the tables describe it.
    CHECK(img.map.valid, "the measured mapping is accepted");
    CHECK(img.map.monotonicEl && img.map.monotonicAz, "both axes run one way");
    CHECK(img.map.roundTripFraction > 0.99,
          "and the scan's own points land back on their own cell");

    // What the linear model would have done with the same data. Kept in the test
    // rather than described in a comment: this is the arithmetic that shipped, and
    // the number it produces is the reason the tables exist.
    int misplaced = 0;
    for (int c = 0; c < cols; ++c) {
        double az = azAt(c);
        az -= kTau * std::floor(az / kTau);                 // as toSpherical reports it
        double d = az - img.map.az0;
        while (d >  kPi) d -= kTau;
        while (d <= -kPi) d += kTau;
        long ci = std::lround(d / img.map.dAzPerCol) % cols;
        if (ci < 0) ci += cols;
        long dc = ci - c;
        if (dc >  cols / 2) dc -= cols;
        if (dc < -cols / 2) dc += cols;
        if (std::labs(dc) > 1) ++misplaced;
    }
    CHECK(misplaced > cols / 4,
          "inverting the fitted line arithmetically misplaces a large share of the raster");

    // The property that matters: a direction resolves to a column that looked that
    // way. Not necessarily the column the point was recorded in — the last few
    // columns repeat the first few, and only one of each pair is in the index — so
    // the check is on the bearing, which is the thing being asked about.
    const double azStep = std::fabs(img.map.azSpanRad) / double(cols - 1);
    int badCol = 0, badRow = 0, unresolved = 0, disagree = 0;
    for (int r = emptyRows; r < rows; r += 3) {
        for (int c = 0; c < cols; c += 7) {
            uint32_t gr, gc;
            if (!img.cellOf(azAt(c), elAt(r), gr, gc)) { ++unresolved; continue; }
            if (int(gr) != r) ++badRow;
            double da = img.map.azByCol[gc] - azAt(c);
            da -= kTau * std::round(da / kTau);
            if (std::fabs(da) > 0.51 * azStep) ++badCol;
            // The fractional coordinates `judgeBrick` bounds a brick with have to
            // agree with the cell a lookup then reads, or a brick can be culled
            // against a rectangle of the image that does not contain it.
            //
            // On the column axis "agree" means to within a whole turn, because a
            // sweep past 360 degrees resolves a bearing to either of two columns
            // that saw it. judgeBrick opens the column bound to the whole raster
            // on such a scan for exactly that reason, so the two need only name
            // the same bearing, not the same index.
            double dcc = img.colCoord(azAt(c)) - double(gc);
            const double perTurn = kTau / (std::fabs(img.map.azSpanRad) / double(cols - 1));
            dcc -= perTurn * std::round(dcc / perTurn);
            if (std::fabs(img.rowCoord(elAt(r)) - double(gr)) > 0.51 ||
                std::fabs(dcc) > 0.51) ++disagree;
        }
    }
    CHECK(unresolved == 0, "every direction the scanner looked resolves to a cell");
    CHECK(badRow == 0, "to the row it was recorded in");
    CHECK(badCol == 0, "and to a column that looked that way");
    CHECK(disagree == 0, "the fractional coordinates agree with the resolved cell");

    // What the reverse index means, checked against its definition rather than
    // against itself. Without this the index could be consistently wrong — every
    // lookup agreeing with every other lookup and all of them reading the wrong
    // part of the raster, which is exactly the failure this change is about.
    //
    // Elevations have one row each, so the rule is strict: the row named is the one
    // whose measured elevation is nearest, to within the bin a lookup quantises
    // into. Bearings do not, because the sweep covered some of them twice and only
    // one of each pair is indexed, so the rule there is the one that matters for a
    // lookup — never further than half a cell from the column it resolves to. This
    // is the assertion that caught the seam gap running to one and a half cells.
    // Half a cell, where a cell is the largest step the table actually takes. The
    // mean step is the wrong yardstick on an axis that is not uniform by
    // construction — this one's step varies by a fifth either side of its mean.
    auto widestStep = [](const std::vector<double>& t) {
        double m = 0;
        for (size_t i = 1; i < t.size(); ++i) m = std::max(m, std::fabs(t[i] - t[i - 1]));
        return m;
    };
    const double elHalfCell = 0.5 * widestStep(img.map.elByRow) + img.map.elBin;
    const double azHalfCell = 0.5 * widestStep(img.map.azByCol) + img.map.azBin;

    int notNearest = 0, tooFar = 0;
    for (int k = 0; k <= 4000; ++k) {
        const double u  = double(k) / 4000.0;
        const double el = img.map.elByRow.front() +
                          u * (img.map.elByRow.back() - img.map.elByRow.front());
        const int32_t got = img.map.rowFor(el);
        if (got < 0) { ++notNearest; continue; }
        double best = 1e300;
        for (uint32_t r = 0; r < img.rows; ++r)
            best = std::min(best, std::fabs(img.map.elByRow[r] - el));
        if (std::fabs(img.map.elByRow[got] - el) > best + img.map.elBin) ++notNearest;
        if (std::fabs(img.map.elByRow[got] - el) > elHalfCell) ++tooFar;

        const double az  = img.map.azLo + u * kTau;
        const int32_t gc = img.map.colFor(az);
        if (gc < 0) { ++tooFar; continue; }
        double da = img.map.azByCol[gc] - az;
        da -= kTau * std::round(da / kTau);
        if (std::fabs(da) > azHalfCell) ++tooFar;
    }
    CHECK(notNearest == 0, "the index names the nearest row for any elevation");
    CHECK(tooFar == 0,
          "and never sends a direction more than half a cell from where it resolves");

    // The 40 rows that hold no returns still map somewhere, extrapolated from
    // their neighbours. Without that a direction inside a blind cone would come
    // back "off the raster" rather than "a direction the instrument never got a
    // return in", and those are different findings.
    //
    // What is NOT claimed is that the extrapolated angles are the scanner's. They
    // cannot be: nothing measured them. Inside such a band every row carries the
    // same verdict anyway — unsampled if the cone was identified, an empty ray if
    // it was not — so which of them a direction resolves to changes no answer. The
    // edge of the band is the part that has to be right, and that row was measured.
    CHECK(img.diag.emptyLeadingRows == uint32_t(emptyRows),
          "the empty band is counted");
    CHECK(img.map.elByRow.size() == size_t(rows), "and the table covers it");
    CHECK(img.map.rowFor(elAt(emptyRows)) == emptyRows,
          "the first row that measured anything resolves exactly");
    bool bandOrdered = true, bandCovered = true;
    for (int r = 1; r < emptyRows; ++r) {
        if (img.map.elByRow[r] <= img.map.elByRow[r - 1]) bandOrdered = false;
        const int32_t got = img.map.rowFor(img.map.elByRow[r]);
        if (got != r) bandCovered = false;
    }
    CHECK(bandOrdered, "the extrapolated rows stay in order");
    CHECK(bandCovered, "and each is what its own elevation resolves to");

    // Off either end of the sweep is still off the raster. The tables must not
    // have turned the whole sphere into something this scan can answer for.
    uint32_t gr, gc;
    CHECK(!img.cellOf(azAt(10), img.map.elByRow.back() + 0.3, gr, gc),
          "above the top of the sweep is outside the field of view");
    CHECK(!img.cellOf(azAt(10), img.map.elByRow.front() - 0.3, gr, gc),
          "and below the bottom of it");
}

// The blind cone is the one empty region that does not mean "the ray came back
// with nothing" — no ray was fired there at all. Believed as a no-return it
// clears a cone to maxRange straight through whatever the instrument stood on.
//
// Which end of the raster it sits at is NOT assumed. A scanner mounted upside
// down has its cone pointing up, and a producer that rewrites the local frame so
// world up is +Z puts the cone at the other end of the raster from an upright
// scan. Assuming an end gets both of those wrong in the worst direction: it
// believes the cone and disbelieves the sky.
static void testBlindConeFoundGeometrically() {
    std::printf("range image: the blind cone is found from geometry, not from up\n");

    // Builds a raster with an unsampled band at one end and sky at the other.
    // `coneAtFirst` puts the instrument's mount at row 0. The rows bordering the
    // cone hold the ground close to the mount; those bordering the sky hold the
    // far side of the site.
    auto build = [](bool coneAtFirst, double groundRange, double skyBorderRange) {
        rimg::RangeImage im;
        im.rows = 120; im.cols = 200;
        im.cells.assign(im.cellCount(), rimg::Cell{});
        const uint32_t coneBand = 15, skyBand = 25;
        for (uint32_t r = 0; r < im.rows; ++r) {
            const bool inCone = coneAtFirst ? (r < coneBand) : (r >= im.rows - coneBand);
            const bool inSky  = coneAtFirst ? (r >= im.rows - skyBand) : (r < skyBand);
            for (uint32_t c = 0; c < im.cols; ++c) {
                rimg::Cell& cell = im.cells[size_t(r) * im.cols + c];
                if (inCone || inSky) {
                    cell = rimg::Cell{4500, uint8_t(rimg::Status::NoReturn)};
                    continue;
                }
                // Range ramps from the ground beside the mount to the far side
                // of the site as the beam flattens out.
                const double u = coneAtFirst
                    ? double(r - coneBand) / double(im.rows - coneBand - skyBand)
                    : double(im.rows - skyBand - 1 - r) / double(im.rows - coneBand - skyBand);
                const double rng = groundRange + u * (skyBorderRange - groundRange);
                cell = rimg::Cell{uint16_t(rng * 100.0), uint8_t(rimg::Status::Hit)};
            }
        }
        im.diag.nearestReturn = groundRange;
        im.diag.furthestReturn = skyBorderRange;
        im.diag.noReturns = uint64_t(coneBand + skyBand) * im.cols;
        // Elevation rising with row, so row 0 is the nadir end of the sweep and
        // the cone's axis can be read off the end of the measured table.
        im.map = rimg::uniformMapping(im.rows, im.cols, -1.3, 0.023, 0.0, kTau / 200.0);
        im.map.valid = true;
        return im;
    };

    rimg::Options opt;

    // An upright scanner: elevation rises with row, so the mount is at row 0 and
    // the ground beside it is 1.8 m away while the sky border is 30 m.
    rimg::RangeImage up = build(true, 1.8, 30.0);
    rimg::markBlindCone(up, opt);
    CHECK(up.diag.blindConeRows == 15, "the cone is found");
    CHECK(up.diag.blindConeAtFirstRow, "at the end the close returns border");
    CHECK(up.statusAt(0, 0) == rimg::Status::OutsideFov, "and clears nothing");
    CHECK(up.statusAt(119, 0) == rimg::Status::NoReturn, "while the sky still clears");
    CHECK(up.diag.hasConeAxis && up.diag.coneAxisWorld[2] < -0.5,
          "the cone points down, so this setup was upright");

    // The same instrument mounted upside down, its local frame rewritten so the
    // cone now sits at the OTHER end of the raster. Nothing about the elevation
    // mapping distinguishes this from the first case — only the geometry does.
    rimg::RangeImage down = build(false, 1.8, 30.0);
    rimg::markBlindCone(down, opt);
    CHECK(down.diag.blindConeRows == 15, "the cone is found when it is at the far end");
    CHECK(!down.diag.blindConeAtFirstRow, "which is where the close returns are");
    CHECK(down.statusAt(119, 0) == rimg::Status::OutsideFov, "and it clears nothing");
    CHECK(down.statusAt(0, 0) == rimg::Status::NoReturn, "while the sky at row 0 clears");
    CHECK(down.diag.hasConeAxis && down.diag.coneAxisWorld[2] > 0.5,
          "the cone points up, so this setup is reported as inverted");

    // Assuming the nadir end from the sign of the elevation mapping would have
    // put the cone at row 0 in both cases. It is at row 119 in the second.
    CHECK(up.diag.blindConeAtFirstRow != down.diag.blindConeAtFirstRow,
          "the two cases land at opposite ends despite identical mappings");

    // Ambiguity is reported, not guessed. Both bands bordered by returns at
    // similar ranges: nothing distinguishes them, so neither is believed to be
    // the cone and the note says so.
    rimg::RangeImage tie = build(true, 12.0, 14.0);
    rimg::markBlindCone(tie, opt);
    CHECK(tie.diag.blindConeRows == 0, "an unclear case is not resolved by a coin toss");
    CHECK(tie.diag.note.find("too similar") != std::string::npos, "and it is reported");
    CHECK(tie.diag.borderRangeFirst > 0 && tie.diag.borderRangeLast > 0,
          "with the evidence it was judged on");

    // Forced, for when the operator knows better than the geometry.
    rimg::Options forced = opt;
    forced.blindCone = rimg::BlindCone::LastRows;
    rimg::RangeImage f = build(true, 1.8, 30.0);
    rimg::markBlindCone(f, forced);
    CHECK(!f.diag.blindConeAtFirstRow, "an explicit setting overrides the geometry");

    // And switched off entirely, every empty cell is a no-return again.
    rimg::Options none = opt;
    none.blindCone = rimg::BlindCone::None;
    rimg::RangeImage n = build(true, 1.8, 30.0);
    rimg::markBlindCone(n, none);
    CHECK(n.diag.blindConeRows == 0, "BlindCone::None leaves every empty cell believed");
    CHECK(n.statusAt(0, 0) == rimg::Status::NoReturn, "including the cone");

    // A hole in the middle of the field of view is a measurement, not a cone.
    rimg::RangeImage holed;
    holed.rows = 100; holed.cols = 200;
    holed.cells.assign(holed.cellCount(), rimg::Cell{1000, uint8_t(rimg::Status::Hit)});
    for (uint32_t r = 40; r < 50; ++r)
        for (uint32_t c = 40; c < 50; ++c)
            holed.cells[size_t(r) * holed.cols + c] =
                rimg::Cell{4500, uint8_t(rimg::Status::NoReturn)};
    holed.map = rimg::uniformMapping(holed.rows, holed.cols, -1.3, 0.026, 0.0, kTau / 200.0);
    holed.map.valid = true;
    rimg::markBlindCone(holed, opt);
    CHECK(holed.diag.blindConeRows == 0, "a hole in the middle is not a blind cone");
    CHECK(holed.statusAt(45, 45) == rimg::Status::NoReturn, "and still clears");
}

// The corpus decides the blind cone, because no single scan reliably can.
//
// This is modelled on the job it was built against: five setups, a band of
// 590-591 rows unsampled at the start of the raster in every one of them, and a
// band at the other end running from 87 to 576 rows depending on where the
// instrument happened to be standing. One of those is fixed geometry and the other
// is scene. The bordering-range test that a single scan has to rely on got two of
// the five wrong in each direction — refusing two outright and calling two of the
// others inverted — because a beam grazing an eave overhead is indistinguishable
// from one grazing the mount.
static void testBlindConeFromTheCorpus() {
    std::printf("range image: the blind cone is decided across the corpus\n");

    // Builds one scan: a fixed cone band at the start, a scene band of the given
    // size at the end, and returns in between at the given ranges. `coneBorder`
    // and `skyBorder` are what the rows bordering each band measured, which is the
    // single-scan evidence — and here it is deliberately misleading.
    auto makeScan = [](uint32_t coneBand, uint32_t sceneBand,
                       double coneBorder, double sceneBorder) {
        rimg::RangeImage im;
        // The real raster's row count, because the bands being told apart are
        // hundreds of rows: a smaller one cannot hold them.
        im.rows = 2500; im.cols = 200;
        im.cells.assign(im.cellCount(), rimg::Cell{});
        im.map = rimg::uniformMapping(im.rows, im.cols, -1.51, 0.00123, 0.0, kTau / 200.0);
        im.map.valid = true;
        for (uint32_t r = 0; r < im.rows; ++r) {
            const bool empty = r < coneBand || r >= im.rows - sceneBand;
            for (uint32_t c = 0; c < im.cols; ++c) {
                rimg::Cell& cell = im.cells[size_t(r) * im.cols + c];
                if (empty) { cell = rimg::Cell{4500, uint8_t(rimg::Status::NoReturn)}; continue; }
                const double u = double(r - coneBand) /
                                 double(im.rows - coneBand - sceneBand - 1);
                const double rng = coneBorder + u * (sceneBorder - coneBorder);
                cell = rimg::Cell{uint16_t(rng * 100.0), uint8_t(rimg::Status::Hit)};
            }
        }
        im.diag.emptyLeadingRows  = coneBand;
        im.diag.emptyTrailingRows = sceneBand;
        im.diag.noReturns      = uint64_t(coneBand + sceneBand) * im.cols;
        im.diag.nearestReturn  = std::min(coneBorder, sceneBorder);
        im.diag.furthestReturn = std::max(coneBorder, sceneBorder);
        return im;
    };

    rimg::Options opt;

    // The five scans, with the bordering ranges the real ones had: the cone end
    // around 2.1-3.2 m and the scene end anywhere from 0.59 m to 3.3 m. Three of
    // the five have a scene border NEARER than the cone border, which is what sent
    // the single-scan test the wrong way.
    const uint32_t cone[5]  = {590, 591, 591, 590, 591};
    const uint32_t scene[5] = {125, 576, 304,  87, 116};
    const double   cb[5]    = {2.21, 2.20, 2.17, 2.08, 3.20};
    const double   sb[5]    = {3.30, 1.20, 3.14, 0.59, 1.23};

    std::vector<rimg::RangeImage> corpus;
    corpus.reserve(5);
    for (int k = 0; k < 5; ++k) corpus.push_back(makeScan(cone[k], scene[k], cb[k], sb[k]));

    // What each scan concludes on its own, which is what rimg::build leaves behind.
    int aloneRight = 0;
    for (auto& im : corpus) {
        rimg::markBlindCone(im, opt);
        if (im.diag.blindConeRows && im.diag.blindConeAtFirstRow) ++aloneRight;
    }
    CHECK(aloneRight < 5,
          "scan by scan, the bordering-range test does not get all five right");

    std::vector<rimg::RangeImage*> raw;
    for (auto& im : corpus) raw.push_back(&im);
    const rimg::ConeVerdict v = rimg::markBlindConeAcrossCorpus(raw, opt);

    CHECK(v.decided, "the corpus settles it");
    CHECK(v.atFirstRow, "on the end whose band is the same in every scan");
    CHECK(v.rowsMin == 590 && v.rowsMax == 591, "and reports the band it found");
    CHECK(v.otherMin == 87 && v.otherMax == 576, "against the other end, which is scene");
    CHECK(v.scans == 5, "over all five");
    CHECK(v.corrected == size_t(5 - aloneRight),
          "and it re-marked exactly the scans that had got it wrong alone");
    CHECK(v.why.find("fixed geometry") != std::string::npos, "with the reason recorded");

    // Every scan now has its cone at the start, clearing nothing, and its scene
    // band at the other end still believed — which is the point: that band is the
    // sky that clears the volume above the site.
    int allRight = 0, skyKept = 0;
    for (const auto& im : corpus) {
        if (im.diag.blindConeRows && im.diag.blindConeAtFirstRow) ++allRight;
        if (im.statusAt(0, 0) == rimg::Status::OutsideFov &&
            im.statusAt(im.rows - 1, 0) == rimg::Status::NoReturn &&
            std::fabs(im.rangeAt(im.rows - 1, 0) - 45.0) < 0.02) ++skyKept;
    }
    CHECK(allRight == 5, "all five end up with the cone at the start");
    CHECK(skyKept == 5, "and all five keep the band at the other end clearing");

    // Un-marking and re-marking has to be exact, or a corrected scan quietly ends
    // up with a band that clears nothing at BOTH ends.
    uint64_t worstExtra = 0;
    for (const auto& im : corpus) {
        uint64_t outside = 0;
        for (const rimg::Cell& c : im.cells)
            if (rimg::Status(c.status) == rimg::Status::OutsideFov) ++outside;
        worstExtra = std::max(worstExtra,
                              outside - uint64_t(im.diag.blindConeRows) * im.cols);
    }
    CHECK(worstExtra == 0,
          "no cell outside the cone was left unsampled by a re-marking");

    // A corpus where both ends are consistent — scans from inside one room, the
    // same ceiling band in each. It still does not pick an end, because there is
    // no reason to prefer either; but "cannot choose" is no longer read as
    // "believe both", which cleared a cone through the ceiling and the floor. A
    // band the whole corpus shares is not a view of anything, and that is as true
    // of two bands as of one, so both are marked unsampled.
    std::vector<rimg::RangeImage> room;
    for (int k = 0; k < 4; ++k) room.push_back(makeScan(600, 610, 2.0, 2.1));
    std::vector<rimg::RangeImage*> rawRoom;
    for (auto& im : room) rawRoom.push_back(&im);
    const rimg::ConeVerdict rv = rimg::markBlindConeAcrossCorpus(rawRoom, opt);
    CHECK(rv.bothEnds, "two equally consistent ends are both the instrument");
    CHECK(rv.why.find("both ends") != std::string::npos, "and the reason says which case");
    // Which is the part that matters: neither band is left believed as sky.
    for (const auto& im : room) {
        CHECK(im.diag.blindConeRows > 0 && im.diag.blindConeRowsLast > 0,
              "both bands are marked unsampled");
        uint64_t believed = 0;
        for (uint32_t r = 0; r < im.diag.blindConeRows; ++r)
            for (uint32_t c = 0; c < im.cols; ++c)
                if (rimg::Status(im.cells[size_t(r) * im.cols + c].status) ==
                    rimg::Status::NoReturn) ++believed;
        for (uint32_t i = 0; i < im.diag.blindConeRowsLast; ++i)
            for (uint32_t c = 0; c < im.cols; ++c)
                if (rimg::Status(im.cells[size_t(im.rows - 1 - i) * im.cols + c].status) ==
                    rimg::Status::NoReturn) ++believed;
        CHECK(believed == 0, "and no cell in either band clears to the rated range");
    }

    // One scan has no corpus, so it falls back to its own geometry and says so.
    rimg::RangeImage single = makeScan(590, 125, 2.21, 3.30);
    std::vector<rimg::RangeImage*> one{&single};
    const rimg::ConeVerdict sv = rimg::markBlindConeAcrossCorpus(one, opt);
    CHECK(!sv.decided && sv.scans == 1, "one scan cannot be settled by a corpus");
    CHECK(sv.why.find("one scan") != std::string::npos, "and the report says so");
}

// A raster whose measured rows do not rise strictly, which must still be used.
//
// This is the regression that took a whole job off the air. A row's elevation is
// measured as the mean of however many points landed in it, and on real data that
// mean wobbles — a few hundredths of a cell, from sparse rows and from a surface
// that is not equally far away all along the row. The tables therefore do not rise
// strictly, and a monotonicity test on them fails.
//
// A previous version of this made that test a gate, reasoning that a mirror
// sweeping past the pole would show up as a fold-back. It does — but so does
// ordinary noise, and no threshold separates them reliably. All five scans of a
// real job were refused, every lookup against them returned nothing, and the site
// came back as a solid ball of "unobserved" with the tool reporting 100% of the
// space in range as unseen.
//
// The round trip is the gate, and the only one. It asks the question that matters
// — does a direction the scanner looked in reach the cell it was recorded in — and
// a table that wobbles by a third of a cell answers it perfectly well.
static void testWobblyRowsAreStillUsable() {
    std::printf("range image: rows that do not rise strictly are still usable\n");

    const int rows = 200, cols = 300;
    const double elLo = -1.0, elStep = 2.0 / double(rows);
    const double dAz = kTau / double(cols);
    // A third of a cell either way, which is several times the quarter-cell slack
    // a monotonicity test allows, and well inside the half cell that would start
    // making neighbouring rows genuinely ambiguous.
    auto jitter = [&](int r) { return 0.33 * elStep * std::sin(11.0 * double(r)); };
    auto elAt   = [&](int r) { return elLo + elStep * double(r) + jitter(r); };
    auto azAt   = [&](int c) { return dAz * double(c); };

    const std::string path = tmpPath("wobbly");
    fixture::Scan sc;
    sc.name = "wobbly";
    sc.hasPose = true;
    sc.q[0] = 1.0;
    sc.hasIndexBounds = true;
    sc.rowMin = 0; sc.rowMax = rows - 1;
    sc.colMin = 0; sc.colMax = cols - 1;
    sc.fields = {
        {"cartesianX",  e57::FieldType::FloatDouble},
        {"cartesianY",  e57::FieldType::FloatDouble},
        {"cartesianZ",  e57::FieldType::FloatDouble},
        {"rowIndex",    e57::FieldType::Integer, 0, rows - 1},
        {"columnIndex", e57::FieldType::Integer, 0, cols - 1},
    };
    sc.data.assign(5, {});
    Lcg rng;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (rng.next() < 0.4) continue;          // sparse, as a real raster is
            const double az = azAt(c), el = elAt(r);
            const double rr = 12.0 + 3.0 * std::sin(2.0 * az);
            const double ce = std::cos(el);
            sc.data[0].push_back(rr * ce * std::cos(az));
            sc.data[1].push_back(rr * ce * std::sin(az));
            sc.data[2].push_back(rr * std::sin(el));
            sc.data[3].push_back(double(r));
            sc.data[4].push_back(double(c));
        }
    }
    CHECK(fixture::write(path, {sc}, 512), "fixture written");

    e57::Reader rd;
    std::string err;
    CHECK(rd.open(path, err), err.empty() ? "opened" : err.c_str());
    rimg::Options opt;
    rimg::RangeImage img;
    const bool built = rimg::build(rd, 0, opt, img, err);
    CHECK(built, err.empty() ? "built" : err.c_str());
    if (!built) return;

    // Whether the wobble is large enough to break strict monotonicity depends on
    // how sparse the rows are, and that is exactly why it must not be a gate: the
    // property is a fact about the sampling, not about whether lookups work.
    CHECK(img.map.valid, "the mapping is used, wobble or no wobble");
    CHECK(img.map.roundTripFraction > 0.99,
          "because the scan's own points reach their own cells");
    CHECK(img.map.rowErrorCells >= 0.0 && img.map.rowErrorCells <= 1.0,
          "with the row error reported, and small");

    // The thing a refusal costs: a scan that contributes nothing looks exactly
    // like a scan that saw nothing. Here every direction the scanner looked in
    // resolves to the cell it was recorded in, sparse rows included.
    int wrong = 0;
    for (int r = 0; r < rows; r += 3)
        for (int c = 0; c < cols; c += 7) {
            uint32_t gr, gc;
            if (!img.cellOf(azAt(c), elAt(r), gr, gc)) { ++wrong; continue; }
            if (int(gr) != r || int(gc) != c) ++wrong;
        }
    CHECK(wrong == 0, "and every sampled direction reaches its own cell");
}

// A scanner that was not quite level, which is every scanner.
//
// This is the fault that took a real job off the air, and no fixture written for
// this module could have caught it: every one of them stood the instrument
// perfectly upright, and on a perfectly upright instrument the bug does not exist.
//
// A terrestrial scanner has a dual-axis compensator and exports its points already
// levelled. So the raster's rows are lines of constant elevation about the
// INSTRUMENT'S axis while the points are stored about the vertical, and the two
// differ by however the tripod was standing. A row's elevation then runs as
// tau*cos(azimuth - phi) — one cycle per turn — and a row stops being a direction.
// Measured on five real setups of one job: 0.55, 1.94, 0.97, 2.49 and 1.02 degrees
// in five different directions, accounting for 80 to 98 per cent of how much
// elevation varied inside a row.
//
// It is a rotation, so it leaves no parallax on edges and the merged cloud is
// perfect — the manufacturer applied it correctly. It is invisible to everything
// except something that goes looking for the raster.
static void testInstrumentNotLevel() {
    std::printf("range image: an instrument that was not quite level\n");

    const int rows = 400, cols = 900;
    const double elLo = -1.2, elStep = 2.3 / double(rows);
    const double dAz  = kTau / double(cols);
    const double tiltDeg = 2.0, tiltTowardDeg = 35.0;
    const double tau = tiltDeg * kPi / 180.0, phi = tiltTowardDeg * kPi / 180.0;

    auto write = [&](const std::string& path, bool tilted) {
        fixture::Scan sc;
        sc.name = "tilted";
        sc.hasPose = true;
        sc.q[0] = 1.0;
        sc.hasIndexBounds = true;
        sc.rowMin = 0; sc.rowMax = rows - 1;
        sc.colMin = 0; sc.colMax = cols - 1;
        sc.fields = {
            {"cartesianX",  e57::FieldType::FloatDouble},
            {"cartesianY",  e57::FieldType::FloatDouble},
            {"cartesianZ",  e57::FieldType::FloatDouble},
            {"rowIndex",    e57::FieldType::Integer, 0, rows - 1},
            {"columnIndex", e57::FieldType::Integer, 0, cols - 1},
        };
        sc.data.assign(5, {});
        // The rotation the compensator applied: about the horizontal axis
        // perpendicular to the lean, which is what turns instrument elevations
        // into levelled ones.
        const double ux = -std::sin(phi), uy = std::cos(phi);
        const double c = std::cos(-tau), s = std::sin(-tau), C = 1 - c;
        const double R[9] = {
            c + ux * ux * C,  ux * uy * C,      uy * s,
            uy * ux * C,      c + uy * uy * C, -ux * s,
            -uy * s,          ux * s,           c,
        };
        for (int r = 0; r < rows; ++r) {
            const double el = elLo + elStep * double(r), ce = std::cos(el);
            for (int col = 0; col < cols; ++col) {
                const double az = dAz * double(col);
                const double rr = 6.0 + 3.0 * std::sin(2.0 * az) + std::cos(3.0 * el);
                double dx = ce * std::cos(az), dy = ce * std::sin(az), dz = std::sin(el);
                if (tilted) {
                    const double a = dx, b = dy, cc2 = dz;
                    dx = R[0] * a + R[1] * b + R[2] * cc2;
                    dy = R[3] * a + R[4] * b + R[5] * cc2;
                    dz = R[6] * a + R[7] * b + R[8] * cc2;
                }
                sc.data[0].push_back(rr * dx);
                sc.data[1].push_back(rr * dy);
                sc.data[2].push_back(rr * dz);
                sc.data[3].push_back(double(r));
                sc.data[4].push_back(double(col));
            }
        }
        return fixture::write(path, {sc}, 512);
    };

    // Level: nothing to correct, and it must not invent a correction.
    {
        const std::string p = tmpPath("level");
        CHECK(write(p, false), "level fixture written");
        e57::Reader rd;
        std::string err;
        CHECK(rd.open(p, err), err.empty() ? "opened" : err.c_str());
        rimg::Options opt;
        rimg::RangeImage img;
        CHECK(rimg::build(rd, 0, opt, img, err), err.empty() ? "built" : err.c_str());
        CHECK(img.tiltDeg < 0.01, "a level instrument is measured as level");
        CHECK(img.map.valid && img.map.roundTripFraction > 0.99,
              "and its raster round-trips");
    }

    // Two degrees of lean, which is an ordinary tripod on a driveway.
    const std::string p = tmpPath("tilted");
    CHECK(write(p, true), "tilted fixture written");
    e57::Reader rd;
    std::string err;
    CHECK(rd.open(p, err), err.empty() ? "opened" : err.c_str());
    rimg::Options opt;
    rimg::RangeImage img;
    const bool built = rimg::build(rd, 0, opt, img, err);
    CHECK(built, err.empty() ? "built" : err.c_str());
    if (!built) return;

    CHECK(std::fabs(img.tiltDeg - tiltDeg) < 0.05, "the lean is recovered");
    double dPhi = img.tiltTowardDeg - tiltTowardDeg;
    dPhi -= 360.0 * std::round(dPhi / 360.0);
    CHECK(std::fabs(dPhi) < 2.0, "and the direction it leaned in");
    CHECK(img.tiltExplained > 0.9,
          "and it accounts for essentially all of the within-row spread");

    // The point of it: without this the raster is not a raster.
    CHECK(img.map.valid, "the mapping is accepted");
    CHECK(img.map.roundTripFraction > 0.99,
          "and the scan's own points reach their own cells");

    // And a lookup in a known direction reaches the cell that measured it — with
    // the direction expressed the way the world sees it, levelled, since that is
    // what a voxel's position gives.
    const double ux = -std::sin(phi), uy = std::cos(phi);
    const double c = std::cos(-tau), s = std::sin(-tau), C = 1 - c;
    const double R[9] = {
        c + ux * ux * C,  ux * uy * C,      uy * s,
        uy * ux * C,      c + uy * uy * C, -ux * s,
        -uy * s,          ux * s,           c,
    };
    int wrong = 0;
    for (int r = 20; r < rows - 20; r += 7) {
        const double el = elLo + elStep * double(r), ce = std::cos(el);
        for (int col = 0; col < cols; col += 11) {
            const double az = dAz * double(col);
            const double a = ce * std::cos(az), b = ce * std::sin(az), cc2 = std::sin(el);
            const double lx = R[0] * a + R[1] * b + R[2] * cc2;
            const double ly = R[3] * a + R[4] * b + R[5] * cc2;
            const double lz = R[6] * a + R[7] * b + R[8] * cc2;
            double laz, lel, lr;
            rimg::toSpherical(lx, ly, lz, laz, lel, lr);
            // What evidenceAt does: into the instrument's frame, then look up.
            double ix = lx, iy = ly, iz = lz;
            img.toInstrument(ix, iy, iz);
            double iaz, iel, ir;
            rimg::toSpherical(ix, iy, iz, iaz, iel, ir);
            uint32_t gr, gc;
            if (!img.cellOf(iaz, iel, gr, gc)) { ++wrong; continue; }
            if (int(gr) != r || int(gc) != col) ++wrong;
        }
    }
    CHECK(wrong == 0, "every levelled direction reaches the cell that measured it");
}

// Range images are identified by a number that is never reused, even when the
// allocator hands back the address a freed image occupied.
//
// This is the root cause of a real defect: the GPU carver cached uploaded range
// data in a map keyed by the image's ADDRESS. Images are built fresh for each run
// of the visibility filter and freed at the end of it, so a second run allocates
// at the same addresses, the cache reports a hit, and the carve runs against the
// PREVIOUS run's range images. With changed parameters that silently drops voxels.
//
// The address reuse is the part worth demonstrating, because it is the step that
// makes the old key unsafe and it is invisible in a single-run test.
static void testRangeImagesHaveIdentityThatIsNeverReused() {
    std::printf("range image: identity survives address reuse\n");

    uint64_t firstUid = 0;
    const void* firstAddr = nullptr;
    {
        auto a = std::make_unique<rimg::RangeImage>();
        firstUid  = a->uid;
        firstAddr = a.get();
        CHECK(firstUid != 0, "an image gets a uid");
    }   // freed here

    // Allocated again, very likely at the same address.
    auto b = std::make_unique<rimg::RangeImage>();
    CHECK(b->uid != firstUid, "a later image never reuses an earlier uid");
    if (b.get() == firstAddr)
        std::printf("      (the allocator did reuse the address, as expected)\n");

    // And distinct live images differ from each other.
    rimg::RangeImage c, d;
    CHECK(c.uid != d.uid, "two live images have different uids");
    CHECK(c.uid != b->uid && d.uid != b->uid, "and differ from the heap one");

    // Monotone, so a cache can also use it to order by age.
    CHECK(d.uid > c.uid, "uids increase");

    // A copy carries the same identity, which is deliberate: it holds the same
    // content, and that is what a cache key is about.
    const rimg::RangeImage e = c;
    CHECK(e.uid == c.uid, "a copy shares the identity of what it copied");
}

// One odd scan in a corpus cannot overturn the blind-cone verdict.
//
// The cone is decided across the corpus, not per scan, because an instrument's
// blind band is the same in every scan while an empty band caused by the scene is
// not. That decision used a MIN and a MAX over every scan, and a min and a max can
// only widen as scans are added — so one scan in fifty with an unusual leading
// band, or a zero one, pushed the spread past the tolerance and undecided the
// whole corpus.
//
// Undecided is not neutral. It unmarks every scan's cone, so each setup's blind
// band becomes rays believed to have seen through to the rated range: a cone
// carved through the floor under every setup. At thirty setups the corpus decided
// and the answer was right; at fifty, one outlier cleared the building's interior
// and the unobserved space vanished. That is what this pins.
static void testOneOddScanCannotOverturnTheConeVerdict() {
    std::printf("range image: the cone verdict survives an outlier\n");

    rimg::Options opt;                       // BlindCone::Auto, spread 0.10

    // Thirty scans of one instrument: a fixed band of about 590 rows at the start,
    // and a trailing band that varies with the scene, which is the real shape.
    auto corpus = [](size_t n) {
        std::vector<std::pair<uint32_t, uint32_t>> b;
        for (size_t i = 0; i < n; ++i)
            b.push_back({590u + uint32_t(i % 3), 80u + uint32_t((i * 37) % 500)});
        return b;
    };

    {
        const rimg::ConeVerdict v = rimg::decideBlindConeEnd(corpus(30), opt);
        CHECK(v.decided, "thirty consistent scans decide");
        CHECK(v.atFirstRow, "and put the instrument's cone at the start");
    }

    // Now fifty, and one of them is odd — a scan cropped so it has no empty
    // leading rows at all. The old test disqualified the leading end outright on
    // `leadMin > 0`.
    {
        auto b = corpus(50);
        b[37].first = 0;
        const rimg::ConeVerdict v = rimg::decideBlindConeEnd(b, opt);
        CHECK(v.decided, "one scan with no leading band does not undecide the corpus");
        CHECK(v.atFirstRow, "and the verdict is the same one");
    }

    // And one whose leading band is wildly wrong, which is what moved the max.
    {
        auto b = corpus(50);
        b[11].first = 2;
        b[29].first = 4000;
        const rimg::ConeVerdict v = rimg::decideBlindConeEnd(b, opt);
        CHECK(v.decided, "two wild leading bands do not undecide it either");
        CHECK(v.atFirstRow, "same verdict");
    }

    // The judgement is still a judgement: when the leading band genuinely varies
    // across most of the corpus, it is not the instrument and the corpus says so.
    {
        std::vector<std::pair<uint32_t, uint32_t>> b;
        for (size_t i = 0; i < 50; ++i)
            b.push_back({100u + uint32_t((i * 53) % 600), 80u + uint32_t((i * 37) % 500)});
        const rimg::ConeVerdict v = rimg::decideBlindConeEnd(b, opt);
        CHECK(!v.decided, "a leading band that really does vary is not fixed geometry");
    }

    // Both ends fixed is not a tie to be broken but an answer: scans from inside
    // one room have the same ceiling band in every one, and a band the whole
    // corpus shares is not scene. Both are marked unsampled rather than believed.
    {
        std::vector<std::pair<uint32_t, uint32_t>> b;
        for (size_t i = 0; i < 50; ++i) b.push_back({590u, 120u});
        const rimg::ConeVerdict v = rimg::decideBlindConeEnd(b, opt);
        CHECK(v.bothEnds, "two equally fixed ends are both the instrument, not a tie");
        CHECK(!v.why.empty(), "and it says so");
    }

    // A minority is a minority: a fifth of the corpus disagreeing is tolerated,
    // half of it is not.
    {
        auto b = corpus(50);
        for (int i = 0; i < 9; ++i) b[size_t(i) * 5].first = 3u;      // 9 of 50
        CHECK(rimg::decideBlindConeEnd(b, opt).decided,
              "nine scans in fifty disagreeing is still a fixed band");
        auto c = corpus(50);
        for (int i = 0; i < 25; ++i) c[size_t(i)].first = 3u;         // 25 of 50
        CHECK(!rimg::decideBlindConeEnd(c, opt).decided,
              "half of them disagreeing is not");
    }
}

int main() {
    testOneOddScanCannotOverturnTheConeVerdict();
    testRangeImagesHaveIdentityThatIsNeverReused();
    std::printf("E57 Coverage Checker — range image tests\n\n");
    testInstrumentNotLevel();
    testWobblyRowsAreStillUsable();
    testBlindConeFromTheCorpus();
    testGridPath();
    testPyramid();
    testSkyVersusDroppedReturns();
    testBlindConeFoundGeometrically();
    testDoubleCoveredMirrorIsRefused();
    testOrdinaryRasterRoundTrips();
    testSweepPastATurnAndNonUniformRows();
    testMappingAndLookup();
    testDownsampleIsConservative();
    testRefusesWhatItCannotIdentify();
    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
