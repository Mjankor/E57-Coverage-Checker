#include "frame.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace viewer {

const char* conventionName(FrameConvention c) {
    switch (c) {
    case FrameConvention::IdentityPose:  return "identity pose";
    case FrameConvention::ScannerLocal:  return "scanner-local (pose applied)";
    case FrameConvention::AlreadyGlobal: return "pre-transformed (pose NOT applied)";
    case FrameConvention::Unknown:       return "undetermined";
    }
    return "?";
}

bool isIdentityPose(const e57::Pose& p) {
    const double tt = p.t[0] * p.t[0] + p.t[1] * p.t[1] + p.t[2] * p.t[2];
    // A quaternion is identity when its vector part vanishes; the sign of w is
    // irrelevant, since q and -q are the same rotation.
    const double vv = p.q[1] * p.q[1] + p.q[2] * p.q[2] + p.q[3] * p.q[3];
    return tt < 1e-18 && vv < 1e-18;
}

Rigid rigidFromPose(const e57::Pose& p) {
    Rigid r;
    double w = p.q[0], x = p.q[1], y = p.q[2], z = p.q[3];
    const double n = std::sqrt(w * w + x * x + y * y + z * z);
    if (n < 1e-12) {
        // Degenerate quaternion: fall back to no rotation rather than
        // producing NaNs that would scatter the whole scan.
        w = 1; x = y = z = 0;
    } else {
        w /= n; x /= n; y /= n; z /= n;
    }
    r.R[0] = 1 - 2 * (y * y + z * z); r.R[1] = 2 * (x * y - w * z);     r.R[2] = 2 * (x * z + w * y);
    r.R[3] = 2 * (x * y + w * z);     r.R[4] = 1 - 2 * (x * x + z * z); r.R[5] = 2 * (y * z - w * x);
    r.R[6] = 2 * (x * z - w * y);     r.R[7] = 2 * (y * z + w * x);     r.R[8] = 1 - 2 * (x * x + y * y);
    r.t[0] = p.t[0]; r.t[1] = p.t[1]; r.t[2] = p.t[2];
    return r;
}

namespace {

double median(std::vector<double>& v) {
    if (v.empty()) return -1.0;
    const size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + (ptrdiff_t)mid, v.end());
    return v[mid];
}

std::string fmt2(const char* f, double a, double b) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), f, a, b);
    return buf;
}

std::string fmt4(const char* f, double a, double b, double c, double d) {
    char buf[420];
    std::snprintf(buf, sizeof(buf), f, a, b, c, d);
    return buf;
}

} // namespace

FrameDecision decideFrame(e57::Reader& reader, size_t scanIndex, size_t sampleTarget) {
    FrameDecision d;
    if (scanIndex >= reader.scanCount()) {
        d.reason = "scan index out of range";
        return d;
    }
    const e57::Scan& s = reader.scan(scanIndex);

    if (!s.hasPose || isIdentityPose(s.pose)) {
        d.convention = FrameConvention::IdentityPose;
        d.reason = s.hasPose ? "pose is identity — points are already in the file frame"
                             : "no pose element — points are already in the file frame";
        return d;
    }

    // Spherical scans are stored about the scanner by construction; there is
    // nothing to detect and the pose always applies.
    if (!(s.field("cartesianX") && s.field("cartesianY") && s.field("cartesianZ"))) {
        d.convention = FrameConvention::ScannerLocal;
        d.reason = "spherical storage is scanner-centric by definition";
        return d;
    }

    std::vector<std::string> want = {"cartesianX", "cartesianY", "cartesianZ"};
    const char* invName = s.field("cartesianInvalidState") ? "cartesianInvalidState" : nullptr;
    if (invName) want.push_back(invName);
    const size_t invIdx = 3;

    const uint64_t stride = std::max<uint64_t>(
        1, s.recordCount / std::max<size_t>(1, sampleTarget));

    std::vector<double> dLocal, dPose;
    dLocal.reserve(sampleTarget);
    dPose.reserve(sampleTarget);
    uint64_t seen = 0;
    std::string err;

    const bool ok = reader.readPoints(scanIndex, want, [&](const e57::PointBlock& b) {
        for (size_t k = 0; k < b.count; ++k, ++seen) {
            if (seen % stride) continue;
            if (invName && b.columns[invIdx][k] != 0.0) continue;
            const double x = b.columns[0][k], y = b.columns[1][k], z = b.columns[2][k];
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;
            dLocal.push_back(std::sqrt(x * x + y * y + z * z));
            const double px = x - s.pose.t[0], py = y - s.pose.t[1], pz = z - s.pose.t[2];
            dPose.push_back(std::sqrt(px * px + py * py + pz * pz));
        }
        return true;
    }, err);

    if (!ok || dLocal.size() < 64) {
        d.convention = FrameConvention::Unknown;
        d.reason = "too few points to judge — assuming the standard (pose applied)";
        return d;
    }

    d.medianFromLocalOrigin = median(dLocal);
    d.medianFromPoseOrigin  = median(dPose);

    // How much is actually at stake, and how much evidence there could be about
    // it. The rotation is what moves the cloud; the translation is the only thing
    // these two medians can see.
    d.poseTranslationM = std::sqrt(s.pose.t[0] * s.pose.t[0] + s.pose.t[1] * s.pose.t[1] +
                                   s.pose.t[2] * s.pose.t[2]);
    {
        const double n = std::sqrt(s.pose.q[0] * s.pose.q[0] + s.pose.q[1] * s.pose.q[1] +
                                   s.pose.q[2] * s.pose.q[2] + s.pose.q[3] * s.pose.q[3]);
        const double w = (n > 1e-12) ? std::fabs(s.pose.q[0]) / n : 1.0;
        d.poseRotationDeg = 2.0 * std::acos(std::min(1.0, w)) * 57.29577951308232;
    }

    // A translation this small cannot separate the two medians, so whatever they
    // come out as is noise. Say that, rather than reporting two equal numbers as
    // though one had beaten the other. A twentieth of the points' own spread: below
    // that the difference between the two hypotheses is smaller than the sampling.
    if (d.poseTranslationM < 0.05 * d.medianFromLocalOrigin) {
        d.uninformative = true;
        d.convention = FrameConvention::ScannerLocal;      // the standard, by default
        d.reason = fmt4("the pose is a %.1f degree rotation with a %.3f m translation and the "
                        "points sit a median %.1f m out. A translation that small cannot move "
                        "either median, so NOTHING here says whether the pose belongs on these "
                        "points; the standard is assumed, and applying it turns this cloud by "
                        "%.1f degrees",
                        d.poseRotationDeg, d.poseTranslationM, d.medianFromLocalOrigin,
                        d.poseRotationDeg);
        return d;
    }

    // Whichever centre the points actually surround is where the scanner is.
    // The two differ by the magnitude of the site coordinates, so a factor of
    // two is an enormous margin, not a tuned threshold.
    if (d.medianFromPoseOrigin * 2.0 < d.medianFromLocalOrigin) {
        d.convention = FrameConvention::AlreadyGlobal;
        d.reason = fmt2("points surround the pose translation (median %.1f m) not the local "
                        "origin (median %.1f m): already transformed, pose NOT applied",
                        d.medianFromPoseOrigin, d.medianFromLocalOrigin);
    } else {
        d.convention = FrameConvention::ScannerLocal;
        d.reason = fmt2("points surround the local origin (median %.1f m) not the pose "
                        "translation (median %.1f m): scanner-local, pose applied",
                        d.medianFromLocalOrigin, d.medianFromPoseOrigin);
    }
    return d;
}

void setupPosition(const e57::Scan& s, const FrameDecision& d, double out[3]) {
    // Either way the scanner ends up at the pose translation in the file frame:
    // applying the pose maps the local origin to t, and pre-transformed data
    // already has it there. The exception is an identity pose with no
    // registration, where the scanner is at the origin.
    (void)d;
    out[0] = s.pose.t[0];
    out[1] = s.pose.t[1];
    out[2] = s.pose.t[2];
}

} // namespace viewer
