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
    // sky, a window, or a surface too dark to register. The only empty cells not
    // believed are the two the instrument itself accounts for: the unsampled cone
    // about its own nadir, and a surface inside its minimum range.
    CHECK(img.diag.blindConeCells + img.diag.tooCloseNoReturns == img.diag.outsideFov,
          "nothing else is second-guessed");
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

// A surface inside the instrument's MINIMUM range returns nothing, and that
// no-return means the opposite of every other one.
//
// This is the defect the fans came from. A setup parked half a metre from a wall —
// on a stair landing, tucked behind a column, up against a lift shaft — has a
// large solid angle of its raster inside the minimum range, and every cell of it
// comes back empty. Believed, each clears a pencil to the rated range straight
// THROUGH the wall: at 0.45 m minimum and 45 m rated, ninety times the distance to
// the thing that was in the way. On a survey conducted entirely indoors that
// leaves fans of cleared space radiating out of the setups, through the walls,
// through the roof.
//
// The evidence is the zone's border. The same surface that is too close to measure
// crosses OUT of the minimum range at the zone's edge and returns there, so the
// returns bordering the zone sit at very nearly the minimum range — where the
// returns bordering a band of sky are metres away at the very least.
static void testNoReturnsInsideTheMinimumRange() {
    std::printf("range image: no-returns inside the minimum range clear nothing\n");

    // A raster of a setup against a wall:
    //
    //   rows 0-39, all columns      the wall at 0.35 m: inside the minimum range,
    //                               so empty, and bounded at row 40 by the same
    //                               wall returning at 0.50 m where it crosses out
    //   rows 100-119, all columns   open sky: empty, bounded at row 99 by the
    //                               scene at 12 m
    //   everything else             returns at 8-12 m
    auto build = [](double wallEdgeM) {
        rimg::RangeImage im;
        im.rows = 120; im.cols = 200;
        im.cells.assign(im.cellCount(), rimg::Cell{});
        for (uint32_t r = 0; r < im.rows; ++r) {
            for (uint32_t c = 0; c < im.cols; ++c) {
                rimg::Cell& cell = im.cells[size_t(r) * im.cols + c];
                if (r < 40 || r >= 100) {
                    cell = rimg::Cell{4500, uint8_t(rimg::Status::NoReturn)};
                } else if (r == 40) {
                    cell = rimg::Cell{uint16_t(wallEdgeM * 100.0), uint8_t(rimg::Status::Hit)};
                } else {
                    cell = rimg::Cell{uint16_t(800 + (c % 5) * 100), uint8_t(rimg::Status::Hit)};
                }
            }
        }
        im.diag.noReturns = uint64_t(40 + 20) * im.cols;
        return im;
    };

    rimg::Options opt;      // minRange 0.45, so the bar is 0.60 m

    {
        rimg::RangeImage im = build(0.50);
        rimg::filterNoReturnsTooClose(im, opt);
        CHECK(im.diag.tooCloseZones == 1, "the zone against the wall is found");
        CHECK(im.diag.tooCloseNoReturns == 40 * im.cols, "all of it, to the last cell");
        CHECK(std::fabs(im.diag.tooCloseBorderRange - 0.50) < 0.005,
              "and the bordering range it was judged on is reported");

        // Every cell of it establishes nothing now, including the ones far from
        // the border. That is what the connectivity is for: a per-cell border test
        // would only have reached row 39.
        CHECK(im.statusAt(39, 100) == rimg::Status::OutsideFov, "the cell beside the border");
        CHECK(im.statusAt(0, 100) == rimg::Status::OutsideFov, "and the cell furthest from it");
        CHECK(im.rangeAt(0, 100) == 0.0,
              "with the stored range gone, so nothing reads it as a measurement");

        // The sky band is untouched. Its border is the scene at 8-12 m, which is
        // nothing like the minimum range, and it is the clearing that carves the
        // volume above the site.
        CHECK(im.statusAt(119, 100) == rimg::Status::NoReturn, "the sky band still clears");
        CHECK(im.statusAt(100, 0) == rimg::Status::NoReturn, "all of it");

        // And the accounting moved with the cells.
        CHECK(im.diag.outsideFov == 40 * im.cols, "the demoted cells are counted as unsampled");
        CHECK(im.diag.noReturns == 20 * im.cols, "and no longer as no-returns");
    }

    // The same raster with the wall's edge returning at 0.90 m — past the bar, so
    // the zone is not inside the minimum range and is a view of something. It
    // keeps clearing. The test is a test, not a licence to demote every empty
    // region that happens to touch a return.
    {
        rimg::RangeImage im = build(0.90);
        rimg::filterNoReturnsTooClose(im, opt);
        CHECK(im.diag.tooCloseZones == 0, "a zone bordered further out is left alone");
        CHECK(im.statusAt(0, 100) == rimg::Status::NoReturn, "and still clears");
    }

    // The bar tracks the instrument. Told the minimum range is 0.2 m, a border at
    // 0.50 m is no longer evidence of anything and the zone is believed; told 0.8 m,
    // the 0.90 m border is.
    {
        rimg::Options shortMin = opt;
        shortMin.minRange = 0.2;                      // bar 0.27 m
        rimg::RangeImage im = build(0.50);
        rimg::filterNoReturnsTooClose(im, shortMin);
        CHECK(im.diag.tooCloseZones == 0, "a shorter minimum range condemns less");

        rimg::Options longMin = opt;
        longMin.minRange = 0.8;                       // bar 1.07 m
        rimg::RangeImage far_ = build(0.90);
        rimg::filterNoReturnsTooClose(far_, longMin);
        CHECK(far_.diag.tooCloseZones == 1, "and a longer one condemns more");
    }

    // Off at zero, like every other filter here: what the file says, and nothing
    // inferred about the instrument.
    {
        rimg::Options off = opt;
        off.minRange = 0.0;
        rimg::RangeImage im = build(0.50);
        rimg::filterNoReturnsTooClose(im, off);
        CHECK(im.diag.tooCloseZones == 0 && im.diag.tooCloseNoReturns == 0,
              "minRange 0 switches the test off");
        CHECK(im.statusAt(0, 100) == rimg::Status::NoReturn, "and changes nothing");
    }

    // A zone crossing the azimuth seam is ONE zone. Two half-zones would each be
    // judged on half a border, and — worse — a zone whose near wall is all on one
    // side of the seam would leave the other half believed.
    {
        rimg::RangeImage im;
        im.rows = 60; im.cols = 200;
        im.cells.assign(im.cellCount(), rimg::Cell{uint16_t(1000), uint8_t(rimg::Status::Hit)});
        // Empty cells spanning the seam: columns 190-199 and 0-9, rows 20-39.
        for (uint32_t r = 20; r < 40; ++r)
            for (uint32_t k = 0; k < 20; ++k) {
                const uint32_t c = (190 + k) % im.cols;
                im.cells[size_t(r) * im.cols + c] =
                    rimg::Cell{4500, uint8_t(rimg::Status::NoReturn)};
            }
        // The wall's own edge, at 0.50 m, ringing the zone: the same surface where
        // it crosses out of the minimum range, which is what bounds a real one.
        for (uint32_t k = 0; k < 20; ++k) {
            const uint32_t c = (190 + k) % im.cols;
            im.cells[size_t(19) * im.cols + c] = rimg::Cell{50, uint8_t(rimg::Status::Hit)};
            im.cells[size_t(40) * im.cols + c] = rimg::Cell{50, uint8_t(rimg::Status::Hit)};
        }
        for (uint32_t r = 20; r < 40; ++r) {
            im.cells[size_t(r) * im.cols + 189] = rimg::Cell{50, uint8_t(rimg::Status::Hit)};
            im.cells[size_t(r) * im.cols + 10]  = rimg::Cell{50, uint8_t(rimg::Status::Hit)};
        }
        im.diag.noReturns = 20 * 20;
        rimg::filterNoReturnsTooClose(im, opt);
        CHECK(im.diag.tooCloseZones == 1, "a zone across the seam is one zone");
        CHECK(im.diag.tooCloseNoReturns == 20 * 20, "and all of it is demoted");
        CHECK(im.statusAt(30, 199) == rimg::Status::OutsideFov, "the half before the seam");
        CHECK(im.statusAt(30, 0) == rimg::Status::OutsideFov, "and the half after it");
    }

    // The case that made this fail on real data: a region with a MIXED border.
    //
    // A surface at 0.3 m blanks out most of a hemisphere, so the region it leaves
    // runs off the edges of that surface — and beyond those edges the rays reached
    // the room, three or four metres away. The border is then half at half a metre
    // and half at four, and one number over the whole of it is a number about
    // nothing. A median said "not close" and not one cell was demoted, which is why
    // 0.3, 0.45 and 0.6 all looked identical: the rule never fired at any of them.
    //
    // Deciding each cell by its NEAREST border splits the region where it physically
    // divides. The half against the close surface is that surface, too close to
    // measure; the half against the room is a view of the room.
    {
        rimg::RangeImage im;
        im.rows = 120; im.cols = 200;
        im.cells.assign(im.cellCount(), rimg::Cell{});
        for (uint32_t r = 0; r < im.rows; ++r)
            for (uint32_t c = 0; c < im.cols; ++c) {
                rimg::Cell& cell = im.cells[size_t(r) * im.cols + c];
                if (r >= 40 && r < 80)                          // the blanked region
                    cell = rimg::Cell{4500, uint8_t(rimg::Status::NoReturn)};
                else if (r < 40)                                // the surface at 0.50 m
                    cell = rimg::Cell{50, uint8_t(rimg::Status::Hit)};
                else                                            // the room at 4.00 m
                    cell = rimg::Cell{400, uint8_t(rimg::Status::Hit)};
            }
        im.diag.noReturns = 40 * im.cols;
        im.diag.nearestReturn = 0.5; im.diag.furthestReturn = 4.0;
        rimg::filterNoReturnsTooClose(im, opt);

        CHECK(im.diag.tooCloseNoReturns > 0,
              "a mixed border no longer means nothing is demoted");
        // Half the region, to the cell: rows 40-59 are nearer the close surface,
        // rows 60-79 nearer the room.
        CHECK(im.statusAt(40, 100) == rimg::Status::OutsideFov,
              "the cell against the close surface clears nothing");
        CHECK(im.statusAt(59, 100) == rimg::Status::OutsideFov,
              "and so does the middle of that half");
        CHECK(im.statusAt(60, 100) == rimg::Status::NoReturn,
              "while the half nearer the room is a view of the room, and clears");
        CHECK(im.statusAt(79, 100) == rimg::Status::NoReturn, "out to the room itself");
        CHECK(im.diag.tooCloseNoReturns == 20 * im.cols,
              "which is exactly half the region");
    }

    // One close return on the edge of a large band of sky costs that band the
    // handful of cells around it, and nothing else. Under a median the same band was
    // all or nothing.
    {
        rimg::RangeImage im;
        im.rows = 120; im.cols = 200;
        im.cells.assign(im.cellCount(), rimg::Cell{});
        for (uint32_t r = 0; r < im.rows; ++r)
            for (uint32_t c = 0; c < im.cols; ++c)
                im.cells[size_t(r) * im.cols + c] =
                    (r >= 40) ? rimg::Cell{4500, uint8_t(rimg::Status::NoReturn)}
                              : rimg::Cell{1200, uint8_t(rimg::Status::Hit)};
        // A single close return poking into the band's edge.
        im.cells[size_t(39) * im.cols + 100] = rimg::Cell{40, uint8_t(rimg::Status::Hit)};
        im.diag.noReturns = 80 * im.cols;
        im.diag.nearestReturn = 0.4; im.diag.furthestReturn = 12.0;
        rimg::filterNoReturnsTooClose(im, opt);

        CHECK(im.diag.tooCloseNoReturns == 0,
              "one close return is a speck, not a surface, and costs the band nothing");
        CHECK(im.statusAt(40, 100) == rimg::Status::NoReturn, "the cell beside it stays");
        CHECK(im.statusAt(119, 100) == rimg::Status::NoReturn, "the far side of the band stays");
        CHECK(im.statusAt(40, 20) == rimg::Status::NoReturn, "and so does the rest of its edge");

        // Two of them side by side ARE a run, and cost the band the wedge nearest
        // them. The line between a speck and a surface is drawn at the smallest
        // thing that can be a surface, because anything larger is a judgement about
        // the scene rather than about the instrument.
        rimg::RangeImage pair = im;
        for (uint32_t c = 100; c < 102; ++c) {
            pair.cells[size_t(39) * pair.cols + c] = rimg::Cell{40, uint8_t(rimg::Status::Hit)};
            pair.cells[size_t(40) * pair.cols + c] =
                rimg::Cell{4500, uint8_t(rimg::Status::NoReturn)};
        }
        rimg::filterNoReturnsTooClose(pair, opt);
        CHECK(pair.diag.tooCloseNoReturns > 0, "two together are a surface");
        CHECK(pair.statusAt(119, 0) == rimg::Status::NoReturn,
              "and the far corner of the band is still sky");
    }

    // A majority of the border, not one cell of it. A sky band with a single close
    // return on its edge — a bird, a leaf on the lens, a fence post at arm's
    // length — is still sky.
    {
        rimg::RangeImage im = build(0.90);
        for (uint32_t c = 90; c < 95; ++c)
            im.cells[size_t(99) * im.cols + c] = rimg::Cell{40, uint8_t(rimg::Status::Hit)};
        rimg::filterNoReturnsTooClose(im, opt);
        CHECK(im.statusAt(110, 100) == rimg::Status::NoReturn,
              "a handful of close returns on the edge of a sky band does not condemn it");
    }
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
                // of the site as the beam flattens out: 0 beside the cone, 1
                // beside the sky, whichever end each is at.
                //
                // Anchored on the rows that actually border the bands. The
                // earlier form divided by the wrong span in the cone-at-the-end
                // case, so the ramp ran PAST the mount and went negative, and a
                // negative range wrapped through uint16_t into 650 m sitting
                // right beside the tripod. The range test read that as the far
                // end and still landed on the right answer, for the wrong reason.
                const uint32_t rowFirst = coneAtFirst ? coneBand : skyBand;
                const uint32_t rowLast  = im.rows - 1 - (coneAtFirst ? skyBand : coneBand);
                const double u = coneAtFirst
                    ? double(r - rowFirst) / double(rowLast - rowFirst)
                    : double(rowLast - r)  / double(rowLast - rowFirst);
                const double rng = groundRange + u * (skyBorderRange - groundRange);
                cell = rimg::Cell{uint16_t(rng * 100.0), uint8_t(rimg::Status::Hit)};
            }
        }
        im.diag.nearestReturn = groundRange;
        im.diag.furthestReturn = skyBorderRange;
        im.diag.noReturns = uint64_t(coneBand + skyBand) * im.cols;
        // Elevation rising with row in BOTH cases, so the sign of the mapping
        // cannot be what decides which end is the instrument.
        //
        // The offset differs because the two cases are the same instrument: the
        // cone spans 35.3 deg from its own pole and the sky band 40.6 deg from the
        // other, and moving the cone to the far end of the raster moves where the
        // sweep has to start for that to still be true. A producer that rewrites
        // the frame to put world up along +Z, and reverses the row order with it,
        // produces exactly this — the bands swap ends and the angles travel with
        // them. Leaving the offset alone would describe an instrument whose cone
        // is 27 deg wide going one way and 48 going the other, which is no
        // instrument at all.
        const double el0 = coneAtFirst ? -1.3 : -1.437;
        im.map = rimg::uniformMapping(im.rows, im.cols, el0, 0.023, 0.0, kTau / 200.0);
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

    // Both bands bordered by returns at similar ranges — 12 m and 14 m — so the
    // range test cannot separate them. It used to refuse here. The ANGLE decides
    // it now: both bands are about the size of the instrument's cone, so the
    // scanner is assumed upright and the mount taken as the low-elevation end,
    // which in this fixture is where the cone actually is.
    rimg::RangeImage tie = build(true, 12.0, 14.0);
    rimg::markBlindCone(tie, opt);
    CHECK(tie.diag.blindConeRows == 15, "the angle resolves what the ranges could not");
    CHECK(tie.diag.blindConeAtFirstRow, "and puts the mount at the low-elevation end");
    CHECK(tie.diag.coneByElevation, "recorded as an angle decision, not a range one");
    CHECK(tie.diag.bandAngleFirstDeg > 0 && tie.diag.bandAngleLastDeg > 0,
          "with both angles measured");
    CHECK(tie.diag.borderRangeFirst > 0 && tie.diag.borderRangeLast > 0,
          "and the ranges it could not judge on");

    // When NEITHER test has anything to say, the answer is not a guess and it is
    // not a refusal either: BOTH bands are marked unsampled, so neither of them
    // clears a cone at its pole. Same scan, told to expect a cone nowhere near
    // either band's size — which is also the check that the expected angle is a
    // parameter and does something.
    //
    // Refusing used to be the answer here, and refusing left both bands believed.
    // That is not the neutral option it reads as: a believed band clears every ray
    // in it to the rated range straight at the pole, because the elevation table is
    // extrapolated across the band.
    {
        rimg::Options narrow = opt;
        narrow.blindConeFromNadirDeg = 80.0;
        narrow.blindConeAngleTolDeg  = 3.0;
        rimg::RangeImage none = build(true, 12.0, 14.0);
        rimg::markBlindCone(none, narrow);
        CHECK(none.diag.blindConeRows == 15 && none.diag.blindConeRowsLast == 25,
              "an unclear case believes neither band rather than picking one");
        CHECK(none.statusAt(0, 0) == rimg::Status::OutsideFov &&
              none.statusAt(none.rows - 1, 0) == rimg::Status::OutsideFov,
              "so no ray in either band clears to the rated range");
        CHECK(none.diag.note.find("nothing says which is the instrument") != std::string::npos,
              "and it is reported, with both kinds of evidence");
        CHECK(none.diag.note.find("80.0") != std::string::npos,
              "including the cone angle it was told to expect");
    }

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

// Every scan decides its own blind cone, and a corpus decides nothing.
//
// This is modelled on the job it was built against: five setups, a band of
// 590-591 rows unsampled at the start of the raster in every one of them, and a
// band at the other end running from 87 to 576 rows depending on where the
// instrument happened to be standing. One of those is the instrument and the other
// is scene, and the bordering ranges cannot tell them apart — on three of the five
// the scene band's border is the NEARER of the two, which is a wall a metre from
// the tripod.
//
// A corpus-wide vote used to settle it, by looking for the band that is the same
// size in every scan. It worked on this job and failed on the next one, and it
// failed in the direction that matters: where no band is common to the corpus —
// the ordinary case — it unmarked every scan's own cone, and a band left believed
// clears every ray in it to the rated range straight at the pole. One scan came
// out right and fifty came out with a cone through the roof and the floor.
//
// The angle settles it per scan, with nothing but the scan. The instrument's cone
// is a fixed half-angle about the instrument's own downward axis, so the band it
// leaves measures forty-odd degrees from its pole in every scan ever taken with
// that instrument; a band of sky runs to the pole and measures nine degrees, or
// twenty-two, depending on how high the scene reached. That is the whole
// discrimination and it needs no other scan to make it.
static void testEachScanDecidesItsOwnCone() {
    std::printf("range image: every scan decides its own blind cone\n");

    // Builds one scan: a cone band at the start, a scene band of the given size at
    // the end, and returns in between at the given ranges. `coneBorder` and
    // `sceneBorder` are what the rows bordering each band measured, and on this job
    // they are deliberately misleading.
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

    // The five scans, with the bands and the bordering ranges the real ones had:
    // the cone end around 2.1-3.2 m and the scene end anywhere from 0.59 m to
    // 3.3 m. Three of the five have a scene border NEARER than the cone border,
    // which is what sent the bordering-range test the wrong way.
    const uint32_t cone[5]  = {590, 591, 591, 590, 591};
    const uint32_t scene[5] = {125, 576, 304,  87, 116};
    const double   cb[5]    = {2.21, 2.20, 2.17, 2.08, 3.20};
    const double   sb[5]    = {3.30, 1.20, 3.14, 0.59, 1.23};

    std::vector<rimg::RangeImage> corpus;
    corpus.reserve(5);
    for (int k = 0; k < 5; ++k) corpus.push_back(makeScan(cone[k], scene[k], cb[k], sb[k]));

    // Each scan, on its own, with no other scan in the room.
    int alone = 0, byAngle = 0;
    for (auto& im : corpus) {
        rimg::markBlindCone(im, opt);
        if (im.diag.blindConeRows && im.diag.blindConeAtFirstRow) ++alone;
        if (im.diag.coneByElevation) ++byAngle;
    }
    CHECK(alone == 5, "all five find the cone at the start of the raster, scan by scan");
    CHECK(byAngle == 5, "and all five decide it by the angle the band spans about nadir");

    // Which is the point: nothing was marked at the far end of the four scans whose
    // scene band is nothing like the cone's size, so the sky there still clears —
    // and that is what carves the volume above the site.
    int skyKept = 0, secondBand = 0;
    for (const auto& im : corpus) {
        if (im.statusAt(0, 0) == rimg::Status::OutsideFov &&
            im.statusAt(im.rows - 1, 0) == rimg::Status::NoReturn &&
            std::fabs(im.rangeAt(im.rows - 1, 0) - 45.0) < 0.02) ++skyKept;
        if (im.diag.blindConeRowsLast) ++secondBand;
    }
    // The exception is the scan whose scene band is 576 rows — 41 degrees from
    // zenith, which IS the size of this instrument's cone. Its two bands cannot be
    // told apart by angle, and its bordering ranges (2.20 m and 1.20 m) are not
    // several times apart either, so NEITHER is believed and both are marked. That
    // costs a cone of unknown space at one pole, which any other setup whose sweep
    // covers that direction then clears. Believing the wrong one costs a cone of
    // space reported as seen that the instrument never looked at.
    CHECK(skyKept == 4, "four of the five keep the band at the other end clearing");
    CHECK(secondBand == 1, "and the fifth, whose scene band is also cone-sized, marks both");

    // No cell outside the marked bands was touched.
    uint64_t worstExtra = 0;
    for (const auto& im : corpus) {
        uint64_t outside = 0;
        for (const rimg::Cell& c : im.cells)
            if (rimg::Status(c.status) == rimg::Status::OutsideFov) ++outside;
        worstExtra = std::max(worstExtra,
                              outside - uint64_t(im.diag.blindConeRows +
                                                 im.diag.blindConeRowsLast) * im.cols);
    }
    CHECK(worstExtra == 0, "and nothing outside the marked bands was made unsampled");

    // The summary reports that, and reaches no verdict of its own.
    std::vector<rimg::RangeImage*> raw;
    for (auto& im : corpus) raw.push_back(&im);
    const rimg::ConeVerdict v = rimg::summariseBlindCones(raw, opt);
    CHECK(v.scans == 5, "over all five");
    CHECK(v.atFirst == 4 && v.bothEnds == 1 && v.atLast == 0 && v.undecided == 0,
          "four at the start of the raster and one at both ends");
    CHECK(v.byAngle == 5, "all five by angle");
    CHECK(v.bandsBelieved == 4, "with four bands left believed as a view of the sky");

    // Scans from inside one room, where the same band sits at each end of every
    // raster: a mount one way and, at a constant height, a ceiling the other. Both
    // bands are the size of the instrument's cone and their bordering ranges are
    // 2.0 m and 2.1 m, so nothing separates them — and then neither is believed.
    // This used to need the corpus to see it. It does not: one scan is enough,
    // because both bands are in that one scan.
    std::vector<rimg::RangeImage> room;
    for (int k = 0; k < 4; ++k) room.push_back(makeScan(600, 610, 2.0, 2.1));
    for (auto& im : room) rimg::markBlindCone(im, opt);
    for (const auto& im : room) {
        CHECK(im.diag.blindConeRows > 0 && im.diag.blindConeRowsLast > 0,
              "both bands are marked unsampled");
        CHECK(im.diag.blindConeAtFirstRow,
              "with the mount reported at the low-elevation end, upright assumed");
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
    std::vector<rimg::RangeImage*> rawRoom;
    for (auto& im : room) rawRoom.push_back(&im);
    const rimg::ConeVerdict rv = rimg::summariseBlindCones(rawRoom, opt);
    CHECK(rv.bothEnds == 4 && rv.bandsBelieved == 0,
          "and the summary says so: four scans with both ends unsampled, nothing believed");

    // One scan on its own reaches exactly the same answer as one of five, because
    // the evidence never came from the others.
    rimg::RangeImage single = makeScan(590, 125, 2.21, 3.30);
    rimg::markBlindCone(single, opt);
    CHECK(single.diag.blindConeRows == 590 && single.diag.blindConeAtFirstRow &&
          single.diag.coneByElevation,
          "one scan decides its cone the same way five do");
    CHECK(single.statusAt(single.rows - 1, 0) == rimg::Status::NoReturn,
          "and keeps its sky");
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

// The instrument's cone, identified by its ANGLE about nadir.
//
// A scanner cannot see its own mount, so a cone of a fixed half-angle about its
// own downward axis is missing from every scan it takes. Sky is not that shape: it
// runs all the way to the pole, so it measures a few degrees from zenith where a
// mount cone measures forty-odd from nadir. That is the discrimination, and it is
// available in the instrument's OWN frame — no world up, no ground plane, no pose.
//
// It LEADS, and the bordering ranges follow it, for two reasons. The angle is a
// property of the hardware where the ranges are a property of the site — a wall a
// metre from the instrument borders a band of sky more closely than the ground
// borders the mount's cone, which is three scans in five on real data. And the
// angle is measured from the band's own pole, so it reads the same for an inverted
// mounting as for an upright one and never has to be told which way up anything is.
//
// The ranges still decide where the angle cannot: where BOTH bands are the cone's
// size, and where neither is.
static void testTheConeIsIdentifiedByItsAngle() {
    std::printf("range image: the cone is found by its angle about nadir\n");

    // A raster sweeping -74.5 to +82.3 degrees, elevation rising with row. A band
    // of `lead` rows at the low end and `trail` at the high end, and the returns
    // bordering both at the same range so the range test can say nothing — which
    // is a job done entirely indoors: a ceiling and a floor both a few metres off.
    auto scan = [](uint32_t lead, uint32_t trail, double borderRange) {
        rimg::RangeImage im;
        im.rows = 120; im.cols = 200;
        im.cells.assign(im.cellCount(), rimg::Cell{});
        for (uint32_t r = 0; r < im.rows; ++r) {
            const bool empty = (r < lead) || (r >= im.rows - trail);
            for (uint32_t c = 0; c < im.cols; ++c)
                im.cells[size_t(r) * im.cols + c] = empty
                    ? rimg::Cell{4500, uint8_t(rimg::Status::NoReturn)}
                    : rimg::Cell{uint16_t(borderRange * 100.0), uint8_t(rimg::Status::Hit)};
        }
        im.diag.nearestReturn  = borderRange;
        im.diag.furthestReturn = borderRange;
        im.diag.noReturns = uint64_t(lead + trail) * im.cols;
        im.map = rimg::uniformMapping(im.rows, im.cols, -1.3, 0.023, 0.0, 6.28318530718 / 200.0);
        im.map.valid = true;
        return im;
    };
    const double kDeg = 57.29577951308232;
    auto elOfRow = [&](uint32_t r) { return (-1.3 + 0.023 * double(r)) * kDeg; };

    rimg::Options opt;      // expects 45 deg +/- 15

    // A mount cone at the low end of about 45 degrees, and a thin band of sky at
    // the high end. Only one of them is cone-shaped, so the angle decides.
    {
        // Pick `lead` so the low band spans ~45 deg from nadir: 90 + el(lead).
        uint32_t lead = 0;
        while (lead < 60 && (90.0 + elOfRow(lead)) < 45.0) ++lead;
        rimg::RangeImage im = scan(lead, 4, 3.0);
        rimg::markBlindCone(im, opt);
        CHECK(im.diag.coneByElevation, "the angle decided it");
        CHECK(im.diag.blindConeAtFirstRow, "the mount is the low-elevation band");
        CHECK(im.diag.blindConeRows == lead, "and the whole band is unsampled");
        CHECK(im.diag.note.find("identified by") != std::string::npos ||
              im.diag.note.find("by ANGLE") != std::string::npos,
              "and it says the angle is what decided");
        // The sky band is left believed, which is what clears the volume above.
        CHECK(im.statusAt(im.rows - 1, 0) == rimg::Status::NoReturn,
              "the thin band at zenith is still believed as sky");
        CHECK(im.statusAt(0, 0) == rimg::Status::OutsideFov, "and the cone clears nothing");
    }

    // Both bands cone-sized, and the returns bordering them at the same range, so
    // nothing separates them. Then NEITHER is believed: both bands are marked
    // unsampled, and the mount is merely REPORTED at the low end, the scanner
    // assumed upright.
    //
    // Marking both is the point. Picking one and believing the other is not a near
    // miss — the elevation table is extrapolated across an unsampled band, so a
    // direction near the pole maps into it and every ray in the band clears to the
    // rated range straight at the pole. That is a cone through the roof, or the
    // floor, at every setup that lands here.
    {
        uint32_t lead = 0;
        while (lead < 60 && (90.0 + elOfRow(lead)) < 40.0) ++lead;
        uint32_t trail = 0;
        while (trail < 60 && (90.0 - elOfRow(119 - trail)) < 40.0) ++trail;
        rimg::RangeImage im = scan(lead, trail, 3.0);
        rimg::markBlindCone(im, opt);
        CHECK(im.diag.coneByElevation, "still an angle decision");
        CHECK(im.diag.blindConeAtFirstRow, "upright assumed: the mount is the low end");
        CHECK(im.diag.blindConeRows == lead && im.diag.blindConeRowsLast == trail,
              "and BOTH bands are marked unsampled, not just the mount's");
        CHECK(im.statusAt(0, 0) == rimg::Status::OutsideFov &&
              im.statusAt(im.rows - 1, 0) == rimg::Status::OutsideFov,
              "so neither end clears a cone at its pole");
        CHECK(im.diag.note.find("assumed upright") != std::string::npos,
              "and the assumption is stated rather than hidden");
        CHECK(im.diag.note.find("NEITHER") != std::string::npos,
              "along with why both bands are unsampled");
    }

    // The angle leads, and it leads even where the bordering ranges say the
    // opposite. This is the case the corpus used to be needed for: a cone band of
    // about 45 degrees at the low end whose border is FAR — the beam grazes the
    // mount and reaches across the room — against a thin band of sky at the high
    // end whose border is NEAR, a wall a metre from the instrument. Three of the
    // five scans of the job this was measured against look like this, and the
    // bordering-range test points the wrong way in every one of them.
    {
        uint32_t lead = 0;
        while (lead < 60 && (90.0 + elOfRow(lead)) < 45.0) ++lead;
        rimg::RangeImage im = scan(lead, 4, 3.0);
        for (uint32_t r = lead; r < im.rows - 4; ++r) {
            const double u = double(r - lead) / double(im.rows - 4 - lead - 1);
            const double rng = 3.3 + u * (1.2 - 3.3);      // far beside the mount
            for (uint32_t c = 0; c < im.cols; ++c)
                im.cells[size_t(r) * im.cols + c] =
                    rimg::Cell{uint16_t(rng * 100.0), uint8_t(rimg::Status::Hit)};
        }
        im.diag.nearestReturn = 1.2; im.diag.furthestReturn = 3.3;
        rimg::markBlindCone(im, opt);
        CHECK(im.diag.coneByElevation, "the angle decided");
        CHECK(im.diag.blindConeAtFirstRow,
              "and it put the cone at the low end, where the band is 45 deg wide");
        CHECK(im.diag.borderRangeFirst > im.diag.borderRangeLast,
              "though the returns bordering that band are the FARTHER of the two");
        CHECK(im.statusAt(im.rows - 1, 0) == rimg::Status::NoReturn,
              "and the band at the other end, which is not the cone's size, still clears");
    }

    // The expected angle is a parameter, and it decides. Told to expect a cone
    // neither band could be — 75 degrees, where this raster's bands are about 45
    // and 13 — no band is identified as the cone, and with the bordering returns all
    // at one range nothing separates the two bands either. Both are then marked, and
    // the note says on what evidence.
    //
    // Note what a value BETWEEN them would do: set the expectation to 13 and the
    // thin band of sky at zenith becomes "the cone", because it really is that
    // size, and the band at the other end is then believed and clears. The default
    // separates them because a mount cone is forty-odd degrees and sky runs to the
    // pole, but the parameter is a statement about the instrument and a wrong one
    // is acted on. Which end it landed on is in the note for exactly that reason.
    {
        uint32_t lead = 0;
        while (lead < 60 && (90.0 + elOfRow(lead)) < 45.0) ++lead;
        rimg::Options wide = opt;
        wide.blindConeFromNadirDeg = 75.0;
        wide.blindConeAngleTolDeg  = 3.0;
        rimg::RangeImage im = scan(lead, 4, 3.0);
        rimg::markBlindCone(im, wide);
        CHECK(!im.diag.coneByElevation, "a band outside the expected angle is not the cone");
        CHECK(im.diag.blindConeRows == lead && im.diag.blindConeRowsLast == 4,
              "so neither band is believed instead of one being picked");
        CHECK(im.diag.note.find("75.0") != std::string::npos,
              "and the note carries the angle it was told to expect");
    }

    // Where NEITHER band is the size of the instrument's cone, the angle has
    // nothing to say and the bordering ranges are all there is — and they can put
    // the mount at the high end of the raster. Bands of 21 and 25 degrees against
    // an expected 45 +/- 15: too small to be this instrument's cone, so whichever
    // is the mount, the raster does not reach far enough past the last return to
    // show it.
    {
        rimg::RangeImage im = scan(4, 13, 3.0);
        CHECK(90.0 + elOfRow(4) < 30.0 && 90.0 - elOfRow(119 - 13) < 30.0,
              "neither band is within the expected cone angle");
        // Rewrite the ranges so the high-end band borders close returns and the
        // low-end band borders far ones — a mount above, sky below.
        for (uint32_t r = 4; r < im.rows - 13; ++r) {
            const double u = double(r - 4) / double(im.rows - 13 - 4 - 1);
            const double rng = 30.0 + u * (1.8 - 30.0);     // far at low, near at high
            for (uint32_t c = 0; c < im.cols; ++c)
                im.cells[size_t(r) * im.cols + c] =
                    rimg::Cell{uint16_t(rng * 100.0), uint8_t(rimg::Status::Hit)};
        }
        im.diag.nearestReturn = 1.8; im.diag.furthestReturn = 30.0;
        rimg::markBlindCone(im, opt);
        CHECK(!im.diag.coneByElevation, "the ranges decided, not the angle");
        CHECK(!im.diag.blindConeAtFirstRow,
              "and they put the mount at the high end, where the close returns are");
        CHECK(im.diag.blindConeRowsLast == 0, "one band only, so one band is marked");
    }
}

int main() {
    testTheConeIsIdentifiedByItsAngle();
    testRangeImagesHaveIdentityThatIsNeverReused();
    std::printf("E57 Coverage Checker — range image tests\n\n");
    testInstrumentNotLevel();
    testWobblyRowsAreStillUsable();
    testEachScanDecidesItsOwnCone();
    testGridPath();
    testPyramid();
    testNoReturnsInsideTheMinimumRange();
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
