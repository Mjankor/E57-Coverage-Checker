#include "range_image.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace rimg {

static constexpr double kPi  = 3.14159265358979323846;
static constexpr double kTau = 2.0 * kPi;

const char* statusName(Status s) {
    switch (s) {
    case Status::Hit:        return "hit";
    case Status::NoReturn:   return "no-return";
    case Status::OutsideFov: return "outside-fov";
    }
    return "?";
}

void toSpherical(double x, double y, double z, double& az, double& el, double& range) {
    range = std::sqrt(x * x + y * y + z * z);
    az = std::atan2(y, x);
    if (az < 0) az += kTau;
    el = (range > 1e-12) ? std::asin(std::clamp(z / range, -1.0, 1.0)) : 0.0;
}

namespace {

// Shortest signed difference between two angles, in (-pi, pi].
double angleDiff(double a, double b) {
    double d = a - b;
    while (d >  kPi) d -= kTau;
    while (d <= -kPi) d += kTau;
    return d;
}

struct Accum {
    double sumEl = 0;          // elevation: plain mean
    double sumAzX = 0, sumAzY = 0;   // azimuth: circular mean, so 0/2pi does not average to pi
    uint64_t n = 0;
};

// Fits y = a + b*x by least squares over the filled entries, returning the
// maximum residual so a non-uniform raster is visible rather than silent.
bool fitLine(const std::vector<double>& y, const std::vector<uint64_t>& n,
             double& a, double& b, double& maxResidual) {
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    uint64_t count = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        if (!n[i]) continue;
        const double x = double(i);
        sx += x; sy += y[i]; sxx += x * x; sxy += x * y[i];
        ++count;
    }
    if (count < 2) return false;
    const double denom = double(count) * sxx - sx * sx;
    if (std::fabs(denom) < 1e-12) return false;
    b = (double(count) * sxy - sx * sy) / denom;
    a = (sy - b * sx) / double(count);

    maxResidual = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        if (!n[i]) continue;
        maxResidual = std::max(maxResidual, std::fabs(y[i] - (a + b * double(i))));
    }
    return true;
}

struct Sample {
    uint32_t row, col;
    float    range;
    double   az, el;
};

// How many points to hold back for the round-trip check, and how widely to
// space them. Spread across the whole scan rather than taken from its start,
// because a mapping can be right for one band of rows and wrong for another —
// which is exactly the failure a double-covered mirror sweep produces.
constexpr uint64_t kRoundTripStride  = 37;
constexpr size_t   kRoundTripSamples = 200000;

} // namespace

bool RangeImage::cellOf(double az, double el, uint32_t& row, uint32_t& col) const {
    if (!map.valid || rows == 0 || cols == 0) return false;

    if (std::fabs(map.dElPerRow) < 1e-12) return false;
    const double rf = (el - map.el0) / map.dElPerRow;
    const long   ri = std::lround(rf);
    if (ri < 0 || ri >= long(rows)) return false;

    if (std::fabs(map.dAzPerCol) < 1e-12) return false;
    // Azimuth wraps, so work with the shortest difference from the first
    // column and let the column index wrap around the grid.
    const double d  = angleDiff(az, map.az0);
    long ci = std::lround(d / map.dAzPerCol);
    ci %= long(cols);
    if (ci < 0) ci += long(cols);

    row = uint32_t(ri);
    col = uint32_t(ci);
    return true;
}

bool RangeImage::sample(double az, double el, Status& st, double& range) const {
    uint32_t r, c;
    if (!cellOf(az, el, r, c)) { st = Status::OutsideFov; range = 0; return false; }
    st = statusAt(r, c);
    range = rangeAt(r, c);
    return true;
}

// Demotes no-returns that have too few no-return neighbours to OutsideFov.
//
// Counted with a separable sliding window, so the cost is two linear passes
// rather than one window per cell: at 13 M cells the naive form would be 330 M
// probes. Columns wrap, because azimuth does and a sky region crossing the seam
// is one region; rows clamp, because the top and bottom of the raster are real
// edges.
//
// The counts all come from the original mask. Reclassifying as it goes would
// make a cell's verdict depend on the order cells were visited, and erode a sky
// region from one side.
void filterIsolatedNoReturns(RangeImage& im, const Options& opt) {
    if (opt.noReturnRadius == 0 || im.rows == 0 || im.cols == 0) return;
    const int R    = int(opt.noReturnRadius);
    const int rows = int(im.rows), cols = int(im.cols);
    if (cols <= 2 * R || rows == 0) return;
    const size_t n = im.cells.size();

    std::vector<uint8_t> isNoReturn(n, 0);
    for (size_t i = 0; i < n; ++i)
        isNoReturn[i] = (Status(im.cells[i].status) == Status::NoReturn) ? 1u : 0u;

    // Horizontal window, wrapping in azimuth.
    std::vector<uint16_t> h(n, 0);
    const int W = 2 * R + 1;
    for (int r = 0; r < rows; ++r) {
        const uint8_t* row = &isNoReturn[size_t(r) * size_t(cols)];
        uint32_t sum = 0;
        for (int c = -R; c <= R; ++c) sum += row[((c % cols) + cols) % cols];
        uint16_t* out = &h[size_t(r) * size_t(cols)];
        for (int c = 0; c < cols; ++c) {
            out[c] = uint16_t(sum);
            sum -= row[(((c - R) % cols) + cols) % cols];
            sum += row[(((c + R + 1) % cols) + cols) % cols];
        }
    }

    // Vertical window, clamped, via a prefix sum down each column. Built in
    // row-major order so both passes stay sequential in memory.
    std::vector<uint32_t> prefix(size_t(rows + 1) * size_t(cols), 0);
    for (int r = 0; r < rows; ++r) {
        const uint16_t* src = &h[size_t(r) * size_t(cols)];
        const uint32_t* prev = &prefix[size_t(r) * size_t(cols)];
        uint32_t*       cur  = &prefix[size_t(r + 1) * size_t(cols)];
        for (int c = 0; c < cols; ++c) cur[c] = prev[c] + src[c];
    }

    uint64_t demoted = 0;
    for (int r = 0; r < rows; ++r) {
        const int lo = std::max(0, r - R);
        const int hi = std::min(rows - 1, r + R);
        const double windowCells = double(W) * double(hi - lo + 1);
        const double needed = opt.noReturnFraction * windowCells;
        const uint32_t* top = &prefix[size_t(lo) * size_t(cols)];
        const uint32_t* bot = &prefix[size_t(hi + 1) * size_t(cols)];
        for (int c = 0; c < cols; ++c) {
            const size_t i = size_t(r) * size_t(cols) + size_t(c);
            if (!isNoReturn[i]) continue;
            if (double(bot[c] - top[c]) >= needed) continue;
            // Not enough company to be sky. It says nothing rather than
            // clearing to maxRange, and its stored range goes with it so no
            // later reader mistakes it for a measurement.
            im.cells[i].status  = uint8_t(Status::OutsideFov);
            im.cells[i].rangeCm = 0;
            ++demoted;
        }
    }
    im.diag.isolatedNoReturns = demoted;
    if (demoted) {
        im.diag.noReturns  -= std::min<uint64_t>(demoted, im.diag.noReturns);
        im.diag.outsideFov += demoted;
    }
}

// Marks the unsampled band at the nadir end of the raster as a direction the
// scanner never looked.
//
// This is the one place where an empty cell does not mean "the ray came back
// with nothing". Every terrestrial scanner has a blind cone beneath it where the
// tripod is; those rows were never fired. Believed as no-returns they clear a
// cone to maxRange straight down through the ground under every setup.
//
// Only a contiguous band running off the end of the raster counts. A hole in the
// middle of the field of view is not a blind cone, it is a measurement, and this
// does not touch it.
void markNadirBand(RangeImage& im, const Options& opt) {
    im.diag.nadirBandRows  = 0;
    im.diag.nadirBandCells = 0;
    if (!opt.nadirBandUnsampled || im.rows == 0 || im.cols == 0) return;
    if (std::fabs(im.map.dElPerRow) < 1e-12) return;

    // Which end of the raster looks down. The mapping's sign says it: row 0 is
    // the nadir end when elevation increases with row.
    const bool nadirIsFirstRow = im.map.dElPerRow > 0;

    auto rowIsEmpty = [&](uint32_t r) {
        const Cell* row = &im.cells[size_t(r) * im.cols];
        for (uint32_t c = 0; c < im.cols; ++c)
            if (Status(row[c].status) == Status::Hit) return false;
        return true;
    };

    uint32_t band = 0;
    if (nadirIsFirstRow) {
        while (band < im.rows && rowIsEmpty(band)) ++band;
    } else {
        while (band < im.rows && rowIsEmpty(im.rows - 1 - band)) ++band;
    }
    // A raster with no returns at all is a different problem; do not swallow it
    // whole as a blind cone.
    if (band == 0 || band >= im.rows) return;

    for (uint32_t i = 0; i < band; ++i) {
        const uint32_t r = nadirIsFirstRow ? i : (im.rows - 1 - i);
        Cell* row = &im.cells[size_t(r) * im.cols];
        for (uint32_t c = 0; c < im.cols; ++c) {
            if (Status(row[c].status) == Status::NoReturn) {
                row[c].status  = uint8_t(Status::OutsideFov);
                row[c].rangeCm = 0;
                ++im.diag.nadirBandCells;
            }
        }
    }
    im.diag.nadirBandRows = band;
    if (im.diag.nadirBandCells) {
        im.diag.noReturns  -= std::min<uint64_t>(im.diag.nadirBandCells, im.diag.noReturns);
        im.diag.outsideFov += im.diag.nadirBandCells;
    }
}

double RangeImage::rowCoord(double el) const {
    if (std::fabs(map.dElPerRow) < 1e-12) return 0.0;
    return (el - map.el0) / map.dElPerRow;
}

double RangeImage::colCoord(double az) const {
    if (std::fabs(map.dAzPerCol) < 1e-12) return 0.0;
    return angleDiff(az, map.az0) / map.dAzPerCol;
}

void buildPyramid(RangeImage& im) {
    im.pyramid = RangePyramid{};
    if (im.rows == 0 || im.cols == 0 || im.cells.empty()) return;

    auto statusBit = [](uint8_t s) -> uint8_t {
        switch (Status(s)) {
        case Status::Hit:        return kHasHit;
        case Status::NoReturn:   return kHasNoReturn;
        case Status::OutsideFov: return kHasOutsideFov;
        }
        return kHasOutsideFov;
    };

    // Level 0: kPyramidBase x kPyramidBase cells per node.
    PyramidLevel l0;
    l0.block = kPyramidBase;
    l0.rows  = (im.rows + kPyramidBase - 1) / kPyramidBase;
    l0.cols  = (im.cols + kPyramidBase - 1) / kPyramidBase;
    l0.minCm.assign(size_t(l0.rows) * l0.cols, 0xFFFF);
    l0.maxCm.assign(size_t(l0.rows) * l0.cols, 0);
    l0.statuses.assign(size_t(l0.rows) * l0.cols, 0);
    for (uint32_t r = 0; r < im.rows; ++r) {
        const size_t nodeRow = size_t(r / kPyramidBase) * l0.cols;
        const Cell*  src     = &im.cells[size_t(r) * im.cols];
        for (uint32_t c = 0; c < im.cols; ++c) {
            const size_t n = nodeRow + c / kPyramidBase;
            const uint16_t v = src[c].rangeCm;
            if (v < l0.minCm[n]) l0.minCm[n] = v;
            if (v > l0.maxCm[n]) l0.maxCm[n] = v;
            l0.statuses[n] |= statusBit(src[c].status);
        }
    }
    im.pyramid.levels.push_back(std::move(l0));

    // Each level after halves both axes. Stop at a single node.
    while (im.pyramid.levels.back().rows > 1 || im.pyramid.levels.back().cols > 1) {
        const PyramidLevel& prev = im.pyramid.levels.back();
        PyramidLevel lv;
        lv.block = prev.block * 2;
        lv.rows  = (prev.rows + 1) / 2;
        lv.cols  = (prev.cols + 1) / 2;
        lv.minCm.assign(size_t(lv.rows) * lv.cols, 0xFFFF);
        lv.maxCm.assign(size_t(lv.rows) * lv.cols, 0);
        lv.statuses.assign(size_t(lv.rows) * lv.cols, 0);
        for (uint32_t r = 0; r < prev.rows; ++r) {
            for (uint32_t c = 0; c < prev.cols; ++c) {
                const size_t src = size_t(r) * prev.cols + c;
                const size_t dst = size_t(r / 2) * lv.cols + (c / 2);
                if (prev.minCm[src] < lv.minCm[dst]) lv.minCm[dst] = prev.minCm[src];
                if (prev.maxCm[src] > lv.maxCm[dst]) lv.maxCm[dst] = prev.maxCm[src];
                lv.statuses[dst] |= prev.statuses[src];
            }
        }
        im.pyramid.levels.push_back(std::move(lv));
    }
}

RangeSpan RangeImage::span(int64_t row0, int64_t row1, int64_t col0, int64_t col1) const {
    RangeSpan out;
    if (pyramid.empty() || rows == 0 || cols == 0) return out;
    if (row1 < row0 || col1 < col0) return out;

    // Rows outside the raster contribute nothing to any lookup, so clamping is
    // not a loss of information — a voxel projecting there gets OutsideFov,
    // which is what a caller reading `statuses` has to allow for anyway.
    row0 = std::max<int64_t>(row0, 0);
    row1 = std::min<int64_t>(row1, int64_t(rows) - 1);
    if (row1 < row0) return out;

    // Columns wrap. Rather than split the query, a range that crosses the seam
    // widens to the whole circle: it is a superset, so still conservative, and
    // it only costs precision for bricks lying along one particular bearing.
    bool allCols = false;
    if (col1 - col0 + 1 >= int64_t(cols)) allCols = true;
    else {
        int64_t lo = col0 % int64_t(cols);
        if (lo < 0) lo += int64_t(cols);
        const int64_t hi = lo + (col1 - col0);
        if (hi >= int64_t(cols)) allCols = true;
        else { col0 = lo; col1 = hi; }
    }

    // The coarsest level at which the rectangle spans few enough nodes to read
    // them all. Reading a superset of the rectangle is safe, so a level that is
    // too coarse costs sharpness, never correctness.
    size_t L = 0;
    for (; L + 1 < pyramid.levels.size(); ++L) {
        const PyramidLevel& lv = pyramid.levels[L];
        const int64_t rSpan = row1 / lv.block - row0 / lv.block + 1;
        const int64_t cSpan = allCols ? int64_t(lv.cols)
                                      : col1 / lv.block - col0 / lv.block + 1;
        if (rSpan <= 3 && cSpan <= 3) break;
    }

    const PyramidLevel& lv = pyramid.levels[L];
    const int64_t r0 = std::min<int64_t>(row0 / lv.block, int64_t(lv.rows) - 1);
    const int64_t r1 = std::min<int64_t>(row1 / lv.block, int64_t(lv.rows) - 1);
    const int64_t c0 = allCols ? 0 : std::min<int64_t>(col0 / lv.block, int64_t(lv.cols) - 1);
    const int64_t c1 = allCols ? int64_t(lv.cols) - 1
                               : std::min<int64_t>(col1 / lv.block, int64_t(lv.cols) - 1);

    uint16_t lo = 0xFFFF, hi = 0;
    uint8_t  st = 0;
    for (int64_t r = r0; r <= r1; ++r) {
        const size_t base = size_t(r) * lv.cols;
        for (int64_t c = c0; c <= c1; ++c) {
            const size_t n = base + size_t(c);
            if (lv.minCm[n] < lo) lo = lv.minCm[n];
            if (lv.maxCm[n] > hi) hi = lv.maxCm[n];
            st |= lv.statuses[n];
        }
    }
    if (lo > hi) return out;              // no populated node in range

    out.minRange = double(lo) * 0.01;
    out.maxRange = double(hi) * 0.01;
    out.statuses = st;
    out.valid = true;
    return out;
}

bool build(e57::Reader& reader, size_t scanIndex, const Options& opt,
           RangeImage& out, std::string& err) {
    if (scanIndex >= reader.scanCount()) { err = "scan index out of range"; return false; }
    const e57::Scan& s = reader.scan(scanIndex);

    out = RangeImage{};
    out.pose    = s.pose;
    out.hasPose = s.hasPose;

    const bool cartesian = s.field("cartesianX") && s.field("cartesianY") && s.field("cartesianZ");
    const bool spherical = s.field("sphericalRange") && s.field("sphericalAzimuth") &&
                           s.field("sphericalElevation");
    if (!cartesian && !spherical) { err = "scan has no position fields"; return false; }

    const bool hasRowCol = s.field("rowIndex") && s.field("columnIndex");
    const bool gridPath  = hasRowCol && s.hasIndexBounds &&
                           s.rowMax > s.rowMin && s.colMax > s.colMin;

    std::vector<std::string> want;
    if (cartesian) want = {"cartesianX", "cartesianY", "cartesianZ"};
    else           want = {"sphericalRange", "sphericalAzimuth", "sphericalElevation"};

    size_t invIdx = SIZE_MAX;
    const char* invName = nullptr;
    if      (s.field("cartesianInvalidState")) invName = "cartesianInvalidState";
    else if (s.field("sphericalInvalidState")) invName = "sphericalInvalidState";
    if (invName) { invIdx = want.size(); want.push_back(invName); }

    size_t rowIdx = SIZE_MAX, colIdx = SIZE_MAX;
    if (gridPath) {
        rowIdx = want.size(); want.push_back("rowIndex");
        colIdx = want.size(); want.push_back("columnIndex");
    }

    // Ranges are measured in the SCANNER's frame, so points must not have the
    // pose applied. Conformant files already store them that way; the
    // non-conformant "pre-transformed" case has to be undone.
    const viewer::FrameDecision fd = viewer::decideFrame(reader, scanIndex);
    const bool subtractPose = !fd.applyPose() && s.hasPose && cartesian;
    // The undo is the full inverse pose, rotation included: p_local =
    // R^T (p_world - t). Subtracting only the translation would leave the points
    // on world-aligned axes about the scanner, which is not the frame `pose`
    // describes — and the carve builds its world-to-scanner transform from
    // `pose`, so the two have to mean the same thing.
    const viewer::Rigid poseRot = viewer::rigidFromPose(s.pose);

    // Grid dimensions.
    uint32_t gridRows = 0, gridCols = 0;
    int64_t  rowMin = 0, colMin = 0;
    if (gridPath) {
        rowMin   = s.rowMin;  colMin = s.colMin;
        gridRows = uint32_t(s.rowMax - s.rowMin + 1);
        gridCols = uint32_t(s.colMax - s.colMin + 1);
    }

    // Cells are capped by binning down; minimum range per bin keeps it
    // conservative (see Options::maxCells).
    uint32_t step = 1;
    if (gridPath) {
        while ((uint64_t(gridRows + step - 1) / step) * (uint64_t(gridCols + step - 1) / step)
               > opt.maxCells) ++step;
    }

    std::vector<Sample> samples;   // for the angular fallback and the mapping fit
    std::vector<Accum>  rowAcc, colAcc;
    uint64_t decoded = 0, invalid = 0;
    double   farthest = 0, nearest = 1e300;

    if (gridPath) {
        out.rows = (gridRows + step - 1) / step;
        out.cols = (gridCols + step - 1) / step;
        out.cells.assign(out.cellCount(), Cell{});
        rowAcc.assign(out.rows, Accum{});
        colAcc.assign(out.cols, Accum{});
    }

    std::string rerr;
    const bool ok = reader.readPoints(scanIndex, want, [&](const e57::PointBlock& b) {
        for (size_t k = 0; k < b.count; ++k) {
            if (invIdx != SIZE_MAX && b.columns[invIdx][k] != 0.0) { ++invalid; continue; }

            double x, y, z;
            if (cartesian) {
                x = b.columns[0][k]; y = b.columns[1][k]; z = b.columns[2][k];
                if (subtractPose) {
                    const double a = x - s.pose.t[0], bb = y - s.pose.t[1], c = z - s.pose.t[2];
                    x = poseRot.R[0] * a + poseRot.R[3] * bb + poseRot.R[6] * c;
                    y = poseRot.R[1] * a + poseRot.R[4] * bb + poseRot.R[7] * c;
                    z = poseRot.R[2] * a + poseRot.R[5] * bb + poseRot.R[8] * c;
                }
            } else {
                const double r = b.columns[0][k], a = b.columns[1][k], e = b.columns[2][k];
                const double ce = std::cos(e);
                x = r * ce * std::cos(a); y = r * ce * std::sin(a); z = r * std::sin(e);
            }
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;

            double az, el, range;
            toSpherical(x, y, z, az, el, range);
            if (range <= 1e-6) continue;
            farthest = std::max(farthest, range);
            nearest  = std::min(nearest, range);
            if (!out.diag.hasReturnBounds) {
                out.diag.hasReturnBounds = true;
                out.diag.returnMin[0] = out.diag.returnMax[0] = x;
                out.diag.returnMin[1] = out.diag.returnMax[1] = y;
                out.diag.returnMin[2] = out.diag.returnMax[2] = z;
            } else {
                const double v[3] = {x, y, z};
                for (int a = 0; a < 3; ++a) {
                    out.diag.returnMin[a] = std::min(out.diag.returnMin[a], v[a]);
                    out.diag.returnMax[a] = std::max(out.diag.returnMax[a], v[a]);
                }
            }
            ++decoded;

            if (!gridPath) {
                samples.push_back({0, 0, float(range), az, el});
                continue;
            }

            const int64_t rr = int64_t(b.columns[rowIdx][k]) - rowMin;
            const int64_t cc = int64_t(b.columns[colIdx][k]) - colMin;
            if (rr < 0 || cc < 0 || rr >= int64_t(gridRows) || cc >= int64_t(gridCols)) continue;

            const uint32_t r = uint32_t(rr) / step;
            const uint32_t c = uint32_t(cc) / step;
            const size_t   i = size_t(r) * out.cols + c;

            // Every so often, remember where this point came from. Once the
            // mapping is fitted these get put back through it: a model of the
            // raster that cannot reproduce the raster's own points is not a
            // model of anything.
            if ((decoded % kRoundTripStride) == 0 && samples.size() < kRoundTripSamples)
                samples.push_back({r, c, float(range), az, el});

            const uint32_t cm = uint32_t(std::min(range * 100.0, 65535.0));
            // Minimum range wins: several source cells can land in one binned
            // cell, and clearing to the nearest of them never over-clears.
            if (out.cells[i].status != uint8_t(Status::Hit) || cm < out.cells[i].rangeCm) {
                out.cells[i].rangeCm = uint16_t(cm);
            }
            out.cells[i].status = uint8_t(Status::Hit);

            Accum& ra = rowAcc[r];
            ra.sumEl += el; ++ra.n;
            Accum& ca = colAcc[c];
            ca.sumAzX += std::cos(az); ca.sumAzY += std::sin(az); ++ca.n;
        }
        return true;
    }, rerr);
    if (!ok) { err = rerr; return false; }
    if (decoded == 0) { err = "no valid points decoded"; return false; }

    out.diag.usedGrid          = gridPath;
    out.diag.nearestReturn     = (nearest < 1e299) ? nearest : 0.0;
    out.diag.furthestReturn    = farthest;

    if (!gridPath) {
        // Angular fallback: bin by direction, and take the field of view from
        // the extent of the returns — which is exactly the assumption that
        // fails for an all-sky band. Recorded in the note.
        err = "scan has no indexBounds with rowIndex/columnIndex; the angular "
              "fallback is not implemented yet — see range_image.h";
        out.diag.note = "no grid metadata: no-return rays cannot be identified reliably";
        return false;
    }

    // Measure the angular mapping and fit a line through each axis.
    std::vector<double>   elByRow(out.rows, 0.0), azByCol(out.cols, 0.0);
    std::vector<uint64_t> rowN(out.rows, 0), colN(out.cols, 0);
    for (uint32_t r = 0; r < out.rows; ++r) {
        rowN[r] = rowAcc[r].n;
        if (rowAcc[r].n) elByRow[r] = rowAcc[r].sumEl / double(rowAcc[r].n);
    }
    for (uint32_t c = 0; c < out.cols; ++c) {
        colN[c] = colAcc[c].n;
        if (colAcc[c].n) azByCol[c] = std::atan2(colAcc[c].sumAzY, colAcc[c].sumAzX);
    }
    // Azimuth wraps, so unwrap the measured sequence before fitting a line
    // through it — otherwise the 2pi step reads as an enormous residual.
    double prev = 0; bool havePrev = false;
    for (uint32_t c = 0; c < out.cols; ++c) {
        if (!colN[c]) continue;
        if (!havePrev) { havePrev = true; prev = azByCol[c]; continue; }
        azByCol[c] = prev + angleDiff(azByCol[c], prev);
        prev = azByCol[c];
    }

    double a, b, res;
    if (fitLine(elByRow, rowN, a, b, res)) {
        out.map.el0 = a; out.map.dElPerRow = b; out.map.elResidualRad = res;
    }
    if (fitLine(azByCol, colN, a, b, res)) {
        out.map.az0 = a; out.map.dAzPerCol = b; out.map.azResidualRad = res;
    }
    // The check the residual is not a substitute for: put the scan's own points
    // back through the mapping and see whether they land where they came from.
    //
    // Computed before the verdict rather than after it, and with the mapping
    // applied directly rather than through cellOf, so the number is reported
    // even when the residual has already condemned the fit. It is the more
    // informative of the two — a residual says a line fits the per-row means, a
    // round trip says lookups reach the right cell — and a diagnostic that
    // vanishes exactly when something is wrong is no use to anyone.
    if (!samples.empty() && std::fabs(out.map.dElPerRow) > 1e-12 &&
        std::fabs(out.map.dAzPerCol) > 1e-12) {
        uint64_t landed = 0;
        for (const Sample& sm : samples) {
            const long ri = std::lround((sm.el - out.map.el0) / out.map.dElPerRow);
            if (ri < 0 || ri >= long(out.rows)) continue;
            long ci = std::lround(angleDiff(sm.az, out.map.az0) / out.map.dAzPerCol);
            ci %= long(out.cols);
            if (ci < 0) ci += long(out.cols);
            // One cell of slack on each axis: a direction on a cell boundary can
            // legitimately round either way, and binning down puts several
            // source cells into one.
            const long dr = ri - long(sm.row);
            long dc = ci - long(sm.col);
            if (dc >  long(out.cols) / 2) dc -= long(out.cols);
            if (dc < -long(out.cols) / 2) dc += long(out.cols);
            if (dr >= -1 && dr <= 1 && dc >= -1 && dc <= 1) ++landed;
        }
        out.map.roundTripFraction = double(landed) / double(samples.size());
    }

    out.map.valid = out.map.elResidualRad >= 0 && out.map.azResidualRad >= 0 &&
                    out.map.elResidualRad <= opt.maxMappingResidualRad &&
                    out.map.azResidualRad <= opt.maxMappingResidualRad &&
                    std::fabs(out.map.dElPerRow) > 1e-12 &&
                    std::fabs(out.map.dAzPerCol) > 1e-12 &&
                    out.map.roundTripFraction >= opt.minRoundTripFraction;

    if (!out.map.valid) {
        char buf[420];
        if (out.map.roundTripFraction >= 0.0 &&
            out.map.roundTripFraction < opt.minRoundTripFraction) {
            std::snprintf(buf, sizeof(buf),
                          "the fitted angular mapping does not describe this raster: only "
                          "%.1f%% of the scan's own points land back on their own cell when "
                          "put through it (residuals %.4f rad row, %.4f rad col). Lookups "
                          "would reach the wrong direction — sky read as ground, and "
                          "building interiors read as clear space",
                          100.0 * out.map.roundTripFraction,
                          out.map.elResidualRad, out.map.azResidualRad);
        } else {
            std::snprintf(buf, sizeof(buf),
                          "angular mapping does not fit a uniform raster "
                          "(residuals %.4f rad row, %.4f rad col, round trip %.1f%%)",
                          out.map.elResidualRad, out.map.azResidualRad,
                          100.0 * std::max(0.0, out.map.roundTripFraction));
        }
        out.diag.note = buf;
    }

    // Rows with no returns at all, at either end of the grid. Left as
    // no-returns — the declared grid is the field of view — but counted, since
    // a band that is really outside the field of view would look the same and
    // only the corpus can say which.
    uint32_t lead = 0, trail = 0;
    while (lead < out.rows && rowN[lead] == 0) ++lead;
    while (trail < out.rows - lead && rowN[out.rows - 1 - trail] == 0) ++trail;
    out.diag.emptyLeadingRows  = lead;
    out.diag.emptyTrailingRows = trail;

    for (size_t i = 0; i < out.cells.size(); ++i) {
        if (out.cells[i].status == uint8_t(Status::Hit)) { ++out.diag.hits; continue; }
        // Every cell the grid declares was sampled, so an empty one is a ray
        // that came back nothing. It clears to maxRange.
        out.cells[i].status  = uint8_t(Status::NoReturn);
        out.cells[i].rangeCm = uint16_t(std::min(opt.maxRange * 100.0, 65535.0));
        ++out.diag.noReturns;
    }
    // The blind cone under the tripod: never sampled, so it establishes nothing.
    // Everything else empty is a ray that was fired and came back with nothing,
    // and clears along its path — whatever it passed through on the way.
    markNadirBand(out, opt);
    // Off by default; see rimg::Options.
    filterIsolatedNoReturns(out, opt);

    out.diag.fillFraction = double(out.diag.hits) / double(out.cellCount());
    if (out.diag.nadirBandRows) {
        char buf[220];
        std::snprintf(buf, sizeof(buf),
                      "%u unsampled rows at the nadir end (%llu cells) treated as the blind "
                      "cone under the tripod rather than as clear space",
                      out.diag.nadirBandRows, (unsigned long long)out.diag.nadirBandCells);
        if (!out.diag.note.empty()) out.diag.note += "; ";
        out.diag.note += buf;
    }
    if (out.diag.isolatedNoReturns) {
        char buf[220];
        std::snprintf(buf, sizeof(buf),
                      "%llu of %llu empty cells had too few empty neighbours to be sky "
                      "and were treated as unsampled rather than as clear space",
                      (unsigned long long)out.diag.isolatedNoReturns,
                      (unsigned long long)(out.diag.noReturns + out.diag.isolatedNoReturns));
        if (!out.diag.note.empty()) out.diag.note += "; ";
        out.diag.note += buf;
    }
    return true;
}

} // namespace rimg
