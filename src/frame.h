// Getting each scan into the shared coordinate system.
//
// ASTM E2807 is unambiguous: the points in a scan are stored in that scan's own
// local coordinate system, and the `pose` element is the rigid transform from
// that system into the file's. A registered multi-setup job therefore stores
// each scan around its own origin and carries the registration entirely in the
// poses. Ignoring the pose does not merely offset a scan — it discards the
// registration, and every setup piles up around a common origin.
//
// So the default is simply: apply the pose. Always. A producer that writes
// already-global coordinates is required to write an identity pose, and
// applying an identity pose costs nothing.
//
// Some writers get this wrong and emit global points *with* a non-identity
// pose; applying it then displaces the scan by the setup position twice over.
// That case is detected rather than assumed, by asking where the scanner
// actually sits relative to the stored points: for scanner-local data the
// points cluster about the local origin, and for pre-transformed data they
// cluster about the pose translation. Comparing the two median distances is
// scale-free and decisive — the two cases differ by the whole magnitude of the
// site coordinates, not by a tunable margin.

#pragma once

#include "e57.h"

#include <string>

namespace viewer {

enum class FrameConvention {
    IdentityPose,   // no pose, or identity: nothing to do
    ScannerLocal,   // conformant — apply the pose
    AlreadyGlobal,  // non-conformant — points pre-transformed; do NOT apply
    Unknown,        // too few points to tell; treated as ScannerLocal
};

struct FrameDecision {
    FrameConvention convention = FrameConvention::Unknown;
    double medianFromLocalOrigin = -1.0;
    double medianFromPoseOrigin  = -1.0;
    std::string reason;

    bool applyPose() const {
        return convention == FrameConvention::ScannerLocal ||
               convention == FrameConvention::Unknown;
    }
    // True when the file appears to contradict the standard, which is worth
    // surfacing rather than silently working around.
    bool nonConformant() const { return convention == FrameConvention::AlreadyGlobal; }
};

// Rigid transform, row-major rotation.
struct Rigid {
    double R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    double t[3] = {0, 0, 0};
    void apply(double& x, double& y, double& z) const {
        const double a = x, b = y, c = z;
        x = R[0] * a + R[1] * b + R[2] * c + t[0];
        y = R[3] * a + R[4] * b + R[5] * c + t[1];
        z = R[6] * a + R[7] * b + R[8] * c + t[2];
    }
};

Rigid rigidFromPose(const e57::Pose& p);
bool  isIdentityPose(const e57::Pose& p);

// Samples the scan to decide the convention. Cheap: a wide stride, positions
// only. Returns Unknown (and applyPose() == true, the conformant default) when
// there is not enough data to judge.
FrameDecision decideFrame(e57::Reader& reader, size_t scanIndex,
                          size_t sampleTarget = 20000);

// The scanner's position in the file's coordinate system, given the decision.
void setupPosition(const e57::Scan& s, const FrameDecision& d, double out[3]);

const char* conventionName(FrameConvention c);

} // namespace viewer
