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
    uint32_t noReturnRadius   = 0;
    double   noReturnFraction = 0.75;
};

// Appends the report for one file to `out`. Returns the number of problems
// found — a mis-decoded scan, bounds outside those declared, a refused mapping.
int scanReport(const std::string& path, const Options& opt, std::string& out);

// Every file in turn, with a heading naming the build that produced the report.
int scanReport(const std::vector<std::string>& paths, const Options& opt, std::string& out);

} // namespace report
