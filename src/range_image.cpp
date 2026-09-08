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
        out.rangeCm.assign(out.cellCount(), 0);
        out.status.assign(out.cellCount(), uint8_t(Status::NoReturn));
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

            const uint32_t cm = uint32_t(std::min(range * 100.0, 65535.0));
            // Minimum range wins: several source cells can land in one binned
            // cell, and clearing to the nearest of them never over-clears.
            if (out.status[i] != uint8_t(Status::Hit) || cm < out.rangeCm[i]) {
                out.rangeCm[i] = uint16_t(cm);
            }
            out.status[i] = uint8_t(Status::Hit);

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
    out.map.valid = out.map.elResidualRad >= 0 && out.map.azResidualRad >= 0 &&
                    out.map.elResidualRad <= opt.maxMappingResidualRad &&
                    out.map.azResidualRad <= opt.maxMappingResidualRad &&
                    std::fabs(out.map.dElPerRow) > 1e-12 &&
                    std::fabs(out.map.dAzPerCol) > 1e-12;
    if (!out.map.valid) {
        char buf[192];
        std::snprintf(buf, sizeof(buf),
                      "angular mapping does not fit a uniform raster "
                      "(residuals %.4f rad row, %.4f rad col)",
                      out.map.elResidualRad, out.map.azResidualRad);
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

    for (size_t i = 0; i < out.status.size(); ++i) {
        if (out.status[i] == uint8_t(Status::Hit)) { ++out.diag.hits; continue; }
        // Every cell the grid declares was sampled, so an empty one is a ray
        // that came back nothing. It clears to maxRange.
        out.status[i]  = uint8_t(Status::NoReturn);
        out.rangeCm[i] = uint16_t(std::min(opt.maxRange * 100.0, 65535.0));
        ++out.diag.noReturns;
    }
    out.diag.fillFraction = double(out.diag.hits) / double(out.cellCount());
    return true;
}

} // namespace rimg
