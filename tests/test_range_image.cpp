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
            if (std::fabs(img.rowCoord(elAt(r)) - double(gr)) > 0.51 ||
                std::fabs(img.colCoord(azAt(c)) - double(gc)) > 0.51) ++disagree;
        }
    }
    CHECK(unresolved == 0, "every direction the scanner looked resolves to a cell");
    CHECK(badRow == 0, "to the row it was recorded in");
    CHECK(badCol == 0, "and to a column that looked that way");
    CHECK(disagree == 0, "the fractional coordinates agree with the resolved cell");

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

int main() {
    std::printf("E57 Coverage Checker — range image tests\n\n");
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
