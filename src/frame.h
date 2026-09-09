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
// That case is detected rather than assumed, by asking where the instrument
// actually is: its returns come in opposite pairs, each pair collinear with it,
// so it is the point where those lines cross. Whichever of the local origin and
// the pose translation that point lands on is the frame the points are in. See
// instrumentCentre.
//
// The earlier test compared how far the points sat from each of the two candidate
// centres — a median radius, not a position — and called it on a factor of two.
// That is decisive at UTM magnitudes, where the two differ by a factor of
// thousands, and it is not decisive at all on a job referenced to a local origin
// on the site: a setup twenty metres out with a scan reaching thirty reads 28.5 m
// from one centre and 15.0 m from the other, a ratio of 1.9, and the answer comes
// out backwards. The cost of getting it backwards is the whole pose translation —
// tens of metres, sometimes straight down — on that setup's position, its marker,
// and everything the carve asks from it. The medians are kept only for the case
// the pairs cannot reach.

#pragma once

#include "e57.h"

#include <cstdint>
#include <string>
#include <vector>

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

    // The instrument, located from the returns themselves — the test that
    // actually decides, with the medians kept above only for when it cannot
    // reach. In the frame the points are stored in, so it comes out near the
    // local origin for a conformant scan and near the pose translation for a
    // pre-transformed one. `centreRms` is how well the opposite rays met, and
    // `centrePairs` how many of them there were; both zero when the scan holds
    // too few pairs to locate anything, which is when `haveCentre` is false.
    bool     haveCentre = false;
    double   centre[3]  = {0, 0, 0};
    double   centreRms  = -1.0;
    uint32_t centrePairs = 0;

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

// Where the instrument stood, from a sample of its returns and nothing else — no
// pose, no metadata, no assumption about which frame the points are in. `xyz` is
// interleaved x, y, z, in whatever frame the caller has them. See frame.cpp for
// the method.
//
// This is what decides which frame a scan's points are in: a terrestrial scan's
// returns come in opposite pairs, each pair collinear with the instrument, so the
// instrument is where those lines cross. Comparing that point against the local
// origin and the pose translation is decisive; comparing how far the points sit
// from each is only decisive when the site coordinates are huge.
//
// False when the returns hold too few opposite pairs to locate anything — a scan
// that did not sweep a full turn, or one too sparse to pair up. `rms` is how well
// the lines met, which is what says whether to believe the answer, and `pairs`
// how many there were.
bool instrumentCentre(const std::vector<double>& xyz, double out[3], double& rms,
                      uint32_t& pairs);

// Samples the scan to decide the convention. Cheap: a wide stride, positions
// only. Returns Unknown (and applyPose() == true, the conformant default) when
// there is not enough data to judge.
FrameDecision decideFrame(e57::Reader& reader, size_t scanIndex,
                          size_t sampleTarget = 20000);

// The scanner's position in the file's coordinate system, given the decision.
void setupPosition(const e57::Scan& s, const FrameDecision& d, double out[3]);

const char* conventionName(FrameConvention c);

} // namespace viewer
