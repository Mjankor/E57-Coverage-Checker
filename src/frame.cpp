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

// Where the instrument stood, from the returns and nothing else.
//
// Two returns fired in exactly opposite directions are collinear with the
// instrument, so the line through that pair of points passes through it. Gather
// enough such lines, from enough directions, and their common point is the
// instrument's centre: the least squares minimiser of
//
//     sum_i || (I - u_i u_i^T) (O - P_i) ||^2
//
// which is a 3x3 normal system. The direction u comes from the two points
// themselves rather than from any angle, so nothing here depends on the pose, on
// any mapping, or on which frame the points are in. The only assumption is that
// the instrument swept a full turn, which is what makes a terrestrial setup's
// returns come in opposite pairs at all.
//
// That independence is why it exists. Every other statement about where a setup
// stood traces back to the file's own `pose`, so when the pose does not mean what
// it is being read to mean there is nothing to notice with. This locates a point
// rather than comparing two scales, and it is decisive where a ratio of medians
// is not — see decideFrame.
//
// Opposite pairs are found by binning directions on a spherical grid and looking
// up the antipodal bin. The grid is sized from the sample, about three returns
// per bin: a caller's sample can be twenty thousand points or two hundred
// thousand, and a grid fine enough for one finds no pairs at all in the other.
// Bins are coarse on purpose — the pairing only has to be nearly opposite, and
// the accuracy comes from the least squares over thousands of lines.
//
// CPU. A bounded sample and a 3x3 solve: a couple of milliseconds, and it does
// not grow with the scan. There is no per-point loop here to put on a GPU.
bool instrumentCentre(const std::vector<double>& xyz, double out[3], double& rms,
                      uint32_t& pairs) {
    constexpr double kPiL = 3.14159265358979323846;
    out[0] = out[1] = out[2] = 0.0;
    rms = -1.0;
    pairs = 0;
    const size_t n = xyz.size() / 3;
    if (n < 600) return false;

    // About three returns per bin, azimuth twice elevation, never finer than a
    // degree or coarser than eight.
    int nEl = int(std::sqrt(double(n) / 6.0));
    nEl = std::max(22, std::min(180, nEl));
    const int nAz = 2 * nEl;
    // How near to opposite a pair must be: two bins of slack, so the bracket
    // widens with the bins rather than silently excluding every pair.
    const double binRad = kPiL / double(nEl);
    const double opposite = -std::cos(2.0 * binRad);
    // The shortest baseline worth a line. Two returns a few centimetres apart
    // give a direction dominated by their own noise.
    constexpr double kBaseline = 1.0;   // metres

    std::vector<int32_t> bin(size_t(nAz) * nEl, -1);
    // First point in each bin wins, so the set of lines depends only on the
    // sample — never on iteration order or thread count.
    for (size_t i = 0; i < n; ++i) {
        const double x = xyz[3 * i], y = xyz[3 * i + 1], z = xyz[3 * i + 2];
        const double r = std::sqrt(x * x + y * y + z * z);
        if (!(r > 1e-6)) continue;
        const double az = std::atan2(y, x);
        const double el = std::asin(std::max(-1.0, std::min(1.0, z / r)));
        int ia = int((az + kPiL) / (2.0 * kPiL) * nAz);
        int ie = int((el + 0.5 * kPiL) / kPiL * nEl);
        ia = std::max(0, std::min(nAz - 1, ia));
        ie = std::max(0, std::min(nEl - 1, ie));
        const size_t b = size_t(ie) * nAz + ia;
        if (bin[b] < 0) bin[b] = int32_t(i);
    }

    // Normal equations. M = sum (I - u u^T), v = sum (I - u u^T) P.
    double M[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    double v[3] = {0, 0, 0};
    struct Line { double u[3], p[3]; };
    std::vector<Line> lines;
    for (int ie = 0; ie < nEl; ++ie) {
        // Opposite: azimuth half a turn round, elevation mirrored.
        const int je = nEl - 1 - ie;
        for (int ia = 0; ia < nAz; ++ia) {
            const int ja = (ia + nAz / 2) % nAz;
            // Each pair once: taken only from one half of the sphere.
            if (je < ie || (je == ie && ja <= ia)) continue;
            const int32_t i0 = bin[size_t(ie) * nAz + ia];
            const int32_t i1 = bin[size_t(je) * nAz + ja];
            if (i0 < 0 || i1 < 0) continue;
            const double a[3] = {xyz[3 * size_t(i0)], xyz[3 * size_t(i0) + 1],
                                 xyz[3 * size_t(i0) + 2]};
            const double b[3] = {xyz[3 * size_t(i1)], xyz[3 * size_t(i1) + 1],
                                 xyz[3 * size_t(i1) + 2]};
            const double ra = std::sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
            const double rb = std::sqrt(b[0]*b[0] + b[1]*b[1] + b[2]*b[2]);
            if (!(ra > 1e-6) || !(rb > 1e-6)) continue;
            double dot = 0;
            for (int k = 0; k < 3; ++k) dot += (a[k] / ra) * (b[k] / rb);
            if (dot > opposite) continue;                 // not opposite enough

            double u[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
            const double len = std::sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]);
            if (!(len > kBaseline)) continue;
            for (int k = 0; k < 3; ++k) u[k] /= len;

            Line L;
            for (int k = 0; k < 3; ++k) { L.u[k] = u[k]; L.p[k] = a[k]; }
            lines.push_back(L);
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    const double m = (r == c ? 1.0 : 0.0) - u[r] * u[c];
                    M[3 * r + c] += m;
                    v[r] += m * a[c];
                }
            }
        }
    }
    // Three lines can meet at a point; a dozen is the least worth reporting.
    if (lines.size() < 12) return false;

    // Gaussian elimination with partial pivoting on a 3x3. Singular exactly when
    // every line is parallel, which for directions over a sphere does not happen
    // — checked rather than assumed.
    double A[3][4] = {{M[0], M[1], M[2], v[0]},
                      {M[3], M[4], M[5], v[1]},
                      {M[6], M[7], M[8], v[2]}};
    const double scale = std::fabs(M[0]) + std::fabs(M[4]) + std::fabs(M[8]);
    for (int col = 0; col < 3; ++col) {
        int piv = col;
        for (int r = col + 1; r < 3; ++r)
            if (std::fabs(A[r][col]) > std::fabs(A[piv][col])) piv = r;
        if (std::fabs(A[piv][col]) < 1e-9 * std::max(1.0, scale)) return false;
        if (piv != col) for (int c = 0; c < 4; ++c) std::swap(A[col][c], A[piv][c]);
        for (int r = 0; r < 3; ++r) {
            if (r == col) continue;
            const double f = A[r][col] / A[col][col];
            for (int c = col; c < 4; ++c) A[r][c] -= f * A[col][c];
        }
    }
    double O[3];
    for (int k = 0; k < 3; ++k) O[k] = A[k][3] / A[k][k];

    // How well the lines actually met. Without this the answer has no standing:
    // a least squares fit always returns something.
    double sum = 0;
    for (const Line& L : lines) {
        double d[3] = {O[0] - L.p[0], O[1] - L.p[1], O[2] - L.p[2]};
        double along = 0;
        for (int k = 0; k < 3; ++k) along += d[k] * L.u[k];
        for (int k = 0; k < 3; ++k) d[k] -= along * L.u[k];
        sum += d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
    }
    for (int k = 0; k < 3; ++k) out[k] = O[k];
    rms = std::sqrt(sum / double(lines.size()));
    pairs = uint32_t(lines.size());
    return true;
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

    std::vector<double> dLocal, dPose, xyz;
    dLocal.reserve(sampleTarget);
    dPose.reserve(sampleTarget);
    xyz.reserve(3 * sampleTarget);
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
            xyz.push_back(x); xyz.push_back(y); xyz.push_back(z);
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

    // Where the instrument actually is, located from its own opposite rays.
    //
    // Tested once per hypothesis rather than once for both, and that is not a
    // detail. Opposite pairs are found by binning the direction each return lies
    // in *as seen from the origin of the coordinates it is handed in*, so the
    // method only finds them when the instrument is somewhere near that origin.
    // Hand it pre-transformed points and ask where the instrument is, and two
    // returns genuinely back to back are nowhere near opposite as seen from the
    // file's origin — it finds no pairs at all and says so.
    //
    // Which turns out to be the sharpest form of the test. Each hypothesis is
    // evaluated in the frame where it predicts the instrument sits at the origin;
    // the right one converges on the origin, and the wrong one does not converge.
    double cLocal[3] = {0, 0, 0}, cPose[3] = {0, 0, 0};
    double rLocal = -1, rPose = -1;
    uint32_t nLocal = 0, nPose = 0;
    const bool okLocal = instrumentCentre(xyz, cLocal, rLocal, nLocal);
    std::vector<double> shifted(xyz.size());
    for (size_t i = 0; i + 2 < xyz.size(); i += 3)
        for (int k = 0; k < 3; ++k) shifted[i + k] = xyz[i + k] - s.pose.t[k];
    const bool okPose = instrumentCentre(shifted, cPose, rPose, nPose);

    auto len = [](const double v[3]) {
        return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    };
    // Close enough to the origin to be the origin: a quarter of the distance
    // between the two candidates. They are |t| apart and the instrument is
    // located to centimetres, so this is a wide bracket around an unambiguous
    // answer, not a threshold being tuned.
    const double near = 0.25 * d.poseTranslationM;
    const bool localSays = okLocal && len(cLocal) < near;
    const bool poseSays  = okPose  && len(cPose)  < near;

    if (localSays && !poseSays) {
        d.convention = FrameConvention::ScannerLocal;
        d.haveCentre = true;
        for (int k = 0; k < 3; ++k) d.centre[k] = cLocal[k];
        d.centreRms = rLocal; d.centrePairs = nLocal;
        d.reason = fmt2("the instrument's own opposite rays cross %.2f m from the local "
                        "origin, and read about the pose translation they do not cross at "
                        "all: scanner-local, pose applied", len(cLocal), 0.0);
        return d;
    }
    if (poseSays && !localSays) {
        d.convention = FrameConvention::AlreadyGlobal;
        d.haveCentre = true;
        // Reported in the file's frame, which is the frame these points are in.
        for (int k = 0; k < 3; ++k) d.centre[k] = cPose[k] + s.pose.t[k];
        d.centreRms = rPose; d.centrePairs = nPose;
        d.reason = fmt2("the instrument's own opposite rays cross %.2f m from the pose "
                        "translation, and read about the local origin they do not cross at "
                        "all: already transformed, pose NOT applied", len(cPose), 0.0);
        return d;
    }
    // Both, or neither. Worth saying rather than quietly falling through: the
    // rays have been asked and could not tell these two apart.
    if (okLocal || okPose)
        d.reason = fmt2("the instrument's opposite rays put it %.2f m from the local origin "
                        "and %.2f m from the pose translation, which settles nothing; ",
                        okLocal ? len(cLocal) : -1.0, okPose ? len(cPose) : -1.0);
    else
        d.reason = "the returns hold no opposite pairs in either frame, so the instrument "
                   "cannot be located; ";

    // The fallback, for a scan that holds too few opposite pairs to locate the
    // instrument — one that did not sweep a full turn, or is too sparse to pair
    // up. Whichever centre the points sit nearer to is taken as the scanner's.
    //
    // Weak, and kept only because it is better than nothing: it compares two
    // radii rather than locating a point, so it is decisive at UTM magnitudes and
    // not at all on a job referenced to an origin on the site. See frame.h.
    const std::string pre = d.reason;
    if (d.medianFromPoseOrigin * 2.0 < d.medianFromLocalOrigin) {
        d.convention = FrameConvention::AlreadyGlobal;
        d.reason = pre + fmt2("points surround the pose translation (median %.1f m) not the "
                              "local origin (median %.1f m): already transformed, pose NOT "
                              "applied", d.medianFromPoseOrigin, d.medianFromLocalOrigin);
    } else {
        d.convention = FrameConvention::ScannerLocal;
        d.reason = pre + fmt2("points surround the local origin (median %.1f m) not the pose "
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
