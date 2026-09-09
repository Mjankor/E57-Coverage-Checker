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
    // The test compares how far the points sit from the local origin against how
    // far they sit from the pose translation. That measures the TRANSLATION and
    // nothing else: a rotation leaves every one of those distances exactly as it
    // was, so nothing sampled here can say whether the pose's rotation belongs on
    // these points.
    //
    // Fine when the translation is large — it settles the question, and the
    // rotation comes along with the answer. Not fine when the translation is near
    // zero: the two medians are then the same number, the test has no information
    // at all, and the decision it makes still turns the cloud by the pose's whole
    // rotation angle. Real files do exactly this — a first setup registered as the
    // datum has a translation of a few millimetres and a rotation of ninety
    // degrees, and the report printed "median 2.6 m, not 2.6 m" as though that
    // were evidence.
    //
    // So the case is now detected and reported rather than answered. These say how
    // much rotation is at stake, and whether the translation could carry any
    // evidence about it at all.
    double poseRotationDeg  = 0.0;
    double poseTranslationM = 0.0;
    bool   uninformative = false;
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
