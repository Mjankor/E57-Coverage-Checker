// Deciding whether a scan is a single structured setup or a merged cloud.
//
// The whole visibility method rests on one property: every point in a scan was
// seen from ONE known origin, so the scan is a range image and each direction
// from that origin has a single first surface. A "unified" or merged cloud —
// many setups baked into one point set — violates that. Its nominal origin is
// meaningless, and treating it as a range image would silently produce
// nonsense rather than fail.
//
// Two independent lines of evidence, because neither alone is sufficient:
//
//   Metadata. Spherical coordinates, indexBounds, row/column indices, or
//   pointGroupingSchemes all imply a grid, and a grid implies one setup. Strong
//   when present — but many writers emit structured scans carrying none of it,
//   so absence proves nothing.
//
//   Geometry. Bin sampled points by direction from the claimed origin and look
//   at the spread of ranges within each bin. A true single setup sees one
//   surface per direction, so spread is near zero except at silhouette edges. A
//   merged cloud stacks the walls of other rooms behind each other along the
//   same ray, so a large fraction of bins are multi-surface. This tests the
//   property the pipeline actually depends on, rather than a proxy for it.

#pragma once

#include "e57.h"

#include <string>
#include <vector>

namespace check {

enum class Kind {
    Structured,   // safe to use
    Unified,      // merged cloud — reject
    Ambiguous,    // no metadata, geometry inconclusive — warn, allow view only
};

struct Thresholds {
    // Angular bin size for the range-spread test. About 1 degree: coarse
    // enough that bins hold several points at typical scan densities, fine
    // enough that a bin rarely straddles two genuinely different surfaces.
    double binDegrees = 1.0;
    // A bin needs this many points before its spread means anything.
    int minPointsPerBin = 4;
    // Range spread within one bin above which it is counted as multi-surface.
    // Half a metre is well beyond scanner noise and beyond the thickness of
    // any single surface, while staying under typical room-to-room spacing.
    double multiSurfaceSpreadM = 0.5;
    // Fraction of populated bins that must be multi-surface before the cloud
    // is called merged. A genuine single setup produces some multi-surface
    // bins at depth discontinuities — doorways, furniture edges — so this
    // cannot be zero.
    //
    // NOT CALIBRATED against real merged clouds. The classifier reports the
    // measured fraction alongside its verdict so the number can be judged
    // directly; tune this once real corpora have been through it.
    double multiSurfaceFraction = 0.20;
    // Points sampled for the geometric test. Enough to populate bins across
    // the sphere without reading whole scans.
    size_t sampleTarget = 200000;
};

struct Result {
    Kind        kind = Kind::Ambiguous;
    std::string summary;                  // one line, for the file list
    std::vector<std::string> evidence;    // detail, for the inspector panel

    // Measured by the geometric test; -1 when it could not run.
    double multiSurfaceFraction = -1.0;
    size_t binsTested           = 0;
    size_t pointsSampled        = 0;

    bool usable() const { return kind == Kind::Structured; }
};

// Metadata-only verdict. Cheap, no point decoding.
Result classifyMetadata(const e57::Scan& s);

// Full check: metadata plus the geometric range-image test, which decodes a
// sample of the scan's points. `reader` must be open on the file `s` came from.
Result classify(e57::Reader& reader, size_t scanIndex,
                const Thresholds& t = Thresholds{});

const char* kindName(Kind k);

} // namespace check
