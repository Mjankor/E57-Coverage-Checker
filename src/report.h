// The scan report, as text.
//
// This is what `e57cov info` prints: per scan, how the file represents its
// no-returns, which coordinate frame its points are in, the raster it declares,
// the angular mapping recovered from it and whether that mapping actually
// describes the raster, the instrument's blind cone and which way up it was
// mounted, and the checks that would expose a mis-decoded bit stream.
//
// It lives here rather than in the CLI because the CLI is not what people run.
// Xcode builds the scheme you have selected, so building the app never builds
// the command line tool, and a stale e57cov sitting on a path looks exactly like
// a current one — which cost two rounds of diagnosis on output that predated the
// fixes being discussed. The app can now produce this itself.

#pragma once

#include "range_image.h"

#include <string>
#include <vector>

namespace report {

struct Options {
    bool   verifyCrc = false;
    double maxRange  = 45.0;
    // Passed through to the range image build, so the report describes the same
    // images a carve would use rather than a differently-configured set.
    rimg::BlindCone blindCone = rimg::BlindCone::Auto;
    double   minRange         = rimg::Options{}.minRange;
};

// Appends the report for one file to `out`. Returns the number of problems
// found — a mis-decoded scan, bounds outside those declared, a refused mapping.
int scanReport(const std::string& path, const Options& opt, std::string& out);

// Every file in turn, with a heading naming the build that produced the report.
int scanReport(const std::vector<std::string>& paths, const Options& opt, std::string& out);

// Does the evidence path do what it claims, on this data?
//
// Every voxel verdict in the whole tool comes from one primitive: carve::evidenceAt,
// which takes a world point, transforms it into a setup's frame, finds the raster
// cell that direction fell in, and reads what the scanner did there. Everything
// else — the tiling, the domain, the display — is bookkeeping around that call.
//
// So this asks the primitive about each scan's own cells, where the right answer
// is not in doubt:
//
//   a cell that returned at d metres  a point at 0.5 d must be VISIBLE
//                                     a point at d must be OCCUPIED
//                                     a point at d + 1 m must say NOTHING
//   a cell that returned nothing      every metre out to the rated range must be
//                                     VISIBLE, with no gaps
//                                     past the rated range must say NOTHING
//   a cell never sampled              says NOTHING at any distance
//
// That covers the whole chain — the pose out to the world frame and the inverse
// back, the spherical conversion, the reverse index, the cell's status and range,
// the surface margin — and it reports a percentage rather than a picture.
//
// It also predicts, from each raster alone, how much space that setup ought to
// clear: every cell's pencil of solid angle out to whatever it established. That
// number comes from nothing but the file, so a carve reporting far less has a
// fault between the raster and the voxels rather than in the data.
int selfTest(const std::vector<std::string>& paths, const Options& opt, std::string& out);

} // namespace report
