#include "point_cloud.h"

#include <algorithm>
#include <cmath>

namespace viewer {

namespace {

bool has(const e57::Scan& s, const char* n) { return s.field(n) != nullptr; }

} // namespace

bool loadCloud(e57::Reader& reader, size_t scanIndex, const LoadOptions& opt,
               PointCloud& out, std::string& err) {
    if (scanIndex >= reader.scanCount()) { err = "scan index out of range"; return false; }
    const e57::Scan& s = reader.scan(scanIndex);

    out = PointCloud{};
    out.scanName         = s.name.empty() ? ("scan " + std::to_string(scanIndex)) : s.name;
    out.scanIndex        = scanIndex;
    out.sourcePointCount = s.recordCount;

    const bool cartesian = has(s, "cartesianX") && has(s, "cartesianY") && has(s, "cartesianZ");
    const bool spherical = has(s, "sphericalRange") && has(s, "sphericalAzimuth") &&
                           has(s, "sphericalElevation");
    if (!cartesian && !spherical) { err = "scan has no cartesian or spherical position fields"; return false; }

    std::vector<std::string> want;
    if (cartesian) want = {"cartesianX", "cartesianY", "cartesianZ"};
    else           want = {"sphericalRange", "sphericalAzimuth", "sphericalElevation"};

    size_t invIdx = SIZE_MAX;
    const char* invName = nullptr;
    if      (has(s, "cartesianInvalidState")) invName = "cartesianInvalidState";
    else if (has(s, "sphericalInvalidState")) invName = "sphericalInvalidState";
    if (invName) { invIdx = want.size(); want.push_back(invName); }

    size_t colIdx = SIZE_MAX, intIdx = SIZE_MAX;
    if (opt.preferColour && has(s, "colorRed") && has(s, "colorGreen") && has(s, "colorBlue")) {
        colIdx = want.size();
        want.push_back("colorRed"); want.push_back("colorGreen"); want.push_back("colorBlue");
    } else if (opt.preferIntensity && has(s, "intensity")) {
        intIdx = want.size();
        want.push_back("intensity");
    }

    // Uniform stride rather than random sampling: deterministic, and it keeps
    // the scan-line structure legible when you zoom in.
    const uint64_t stride = std::max<uint64_t>(
        1, (s.recordCount + opt.maxPoints - 1) / std::max<size_t>(1, opt.maxPoints));

    // Positions accumulate in double; the origin is fixed from the first kept
    // point so the float offsets stay small regardless of the coordinate system.
    bool     haveOrigin = false;
    double   lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};
    uint64_t seen = 0;

    out.xyz.reserve(3 * std::min<uint64_t>(opt.maxPoints, s.recordCount ? s.recordCount : 1));
    if (colIdx != SIZE_MAX || intIdx != SIZE_MAX)
        out.rgb.reserve(std::min<uint64_t>(opt.maxPoints, s.recordCount ? s.recordCount : 1));

    // Intensity is normalised against the range actually observed; E57 leaves
    // its scale to the producer, so a fixed 0..1 assumption grey-washes files
    // that store raw counts.
    double intLo = 1e300, intHi = -1e300;
    std::vector<float> intensities;

    const bool ok = reader.readPoints(scanIndex, want, [&](const e57::PointBlock& b) {
        for (size_t k = 0; k < b.count; ++k, ++seen) {
            if (seen % stride) continue;
            if (invIdx != SIZE_MAX && b.columns[invIdx][k] != 0.0) continue;

            double x, y, z;
            if (cartesian) {
                x = b.columns[0][k]; y = b.columns[1][k]; z = b.columns[2][k];
            } else {
                const double r = b.columns[0][k], az = b.columns[1][k], el = b.columns[2][k];
                // E57 spherical: elevation measured from the XY plane, not from
                // the pole.
                const double ce = std::cos(el);
                x = r * ce * std::cos(az);
                y = r * ce * std::sin(az);
                z = r * std::sin(el);
            }
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;

            if (!haveOrigin) {
                out.originX = x; out.originY = y; out.originZ = z;
                haveOrigin = true;
                lo[0] = hi[0] = 0; lo[1] = hi[1] = 0; lo[2] = hi[2] = 0;
            }
            const double dx = x - out.originX, dy = y - out.originY, dz = z - out.originZ;
            out.xyz.push_back(float(dx));
            out.xyz.push_back(float(dy));
            out.xyz.push_back(float(dz));
            lo[0] = std::min(lo[0], dx); hi[0] = std::max(hi[0], dx);
            lo[1] = std::min(lo[1], dy); hi[1] = std::max(hi[1], dy);
            lo[2] = std::min(lo[2], dz); hi[2] = std::max(hi[2], dz);

            if (colIdx != SIZE_MAX) {
                Rgb8 c;
                c.r = uint8_t(std::clamp(b.columns[colIdx + 0][k], 0.0, 255.0));
                c.g = uint8_t(std::clamp(b.columns[colIdx + 1][k], 0.0, 255.0));
                c.b = uint8_t(std::clamp(b.columns[colIdx + 2][k], 0.0, 255.0));
                out.rgb.push_back(c);
            } else if (intIdx != SIZE_MAX) {
                const double v = b.columns[intIdx][k];
                intLo = std::min(intLo, v);
                intHi = std::max(intHi, v);
                intensities.push_back(float(v));
            }
        }
        return true;
    }, err);

    if (!ok) return false;
    if (out.xyz.empty()) { err = "no valid points decoded"; return false; }

    if (intIdx != SIZE_MAX) {
        const double span = (intHi > intLo) ? (intHi - intLo) : 1.0;
        out.rgb.resize(intensities.size());
        for (size_t i = 0; i < intensities.size(); ++i) {
            const double t = std::clamp((double(intensities[i]) - intLo) / span, 0.0, 1.0);
            const uint8_t g = uint8_t(40 + 215 * t);   // keep the darkest returns visible
            out.rgb[i] = Rgb8{g, g, g, 255};
        }
    }

    for (int i = 0; i < 3; ++i) {
        out.loMin[i] = float(lo[i]);
        out.hiMax[i] = float(hi[i]);
    }

    // The setup position, expressed in the same local frame as the points.
    if (s.hasPose || cartesian) {
        out.originOffset[0] = float(s.pose.t[0] - out.originX);
        out.originOffset[1] = float(s.pose.t[1] - out.originY);
        out.originOffset[2] = float(s.pose.t[2] - out.originZ);
        out.hasSetupPosition = s.hasPose;
    }
    return true;
}

} // namespace viewer
