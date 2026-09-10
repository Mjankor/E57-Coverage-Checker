// Deciding whether a scan is a single structured setup or a merged cloud.
//
// The whole visibility method rests on one property: every point in a scan was
// seen from ONE known origin, so the scan is a range image and each direction
// from that origin has a single first surface. A "unified" or merged cloud —
// many setups baked into one point set — violates that. Its nominal origin is
// meaningless, and treating it as a range image would silently produce
// nonsense rather than fail.
//
// Two lines of evidence, and they are NOT of equal weight. Read the second note
// before acting on anything this produces.
//
//   Metadata. Spherical coordinates, indexBounds, row/column indices, or
//   pointGroupingSchemes all imply a grid, and a grid implies one setup. Strong
//   when present — but many writers emit structured scans carrying none of it,
//   so absence proves nothing.
//
//   Geometry. Bin sampled points by direction from the claimed origin and look
//   at the spread of ranges within each bin. A true single setup sees one surface
//   per direction, so spread should be near zero except at silhouette edges.
//
// WHAT THE GEOMETRIC TEST ACTUALLY MEASURES, which is not what it was built to
// measure. A one-degree bin is `range x 0.0175` metres across, and the spread
// that counts as multi-surface is a fixed 0.5 m. So the test's sensitivity
// depends on how far away the scene is, and the same scene decided differently
// at different scales. Measured on a closed box scanned from its centre with
// every ray returning — one setup, one surface per direction, the cleanest
// possible structured scan — with only the absolute size changing:
//
//     half-extents      mean range   bin across   multi-surface
//     2 x 2 x 1.25 m       1.9 m       0.03 m        0.0 %
//     6 x 6 x 1.5 m        3.1 m       0.05 m        0.0 %
//     15 x 15 x 4 m        8.1 m       0.14 m        4.0 %
//     40 x 40 x 8 m       17.8 m       0.31 m       17.8 %
//     100 x 100 x 10 m    26.9 m       0.47 m       28.7 %
//
// The last row is over the 20 per cent threshold. Nothing about the scene changed
// except its size: once a bin is as wide as the spread allowance, the ordinary
// slope of an obliquely viewed surface fills the budget on its own. A real
// outdoor setup from the job this was built against measures 25.6 per cent, and
// the rate is uniform across every elevation band rather than concentrated on the
// ground, so it is scene scale and not one awkward surface.
//
// It fails in the other direction too. Three setups, each in its own 3 m room,
// rooms in a row, all baked into one cloud about a claimed single origin:
//
//     rooms apart    0.5 m   1.0 m   2.0 m   4.0 m   8.0 m   20.0 m
//     multi-surface  13.1%   17.4%   30.1%   30.1%    5.8%     0.7%
//
// Caught only between about 2 and 4 metres. Widely separated setups — the normal
// case in a real survey — occupy disjoint angular sectors as seen from the
// claimed origin, so no bin ever holds two surfaces and the cloud passes at 0.7
// per cent.
//
// So the measurement tracks scene scale and setup separation, not how many
// origins the cloud has. It is REPORTED AND NOT ACTED ON. Nothing excludes a scan
// because of it — see indexer::ScanRef::usable.
//
// WHAT GUARDS THE PROPERTY INSTEAD. The pipeline does depend on single-origin,
// and the check that enforces it lives where it is needed:
// rimg::Options::minRoundTripFraction. A scan's own points are put back through
// the recovered (row, column) -> (azimuth, elevation) mapping and have to land on
// the cell they were decoded from; below 90 per cent the mapping is not
// describing the raster and the range image is refused. Several setups merged
// into one cloud cannot round-trip through one mapping about one origin, so they
// fail it — and the visibility carve, which is the stage that needs the property,
// uses that and has never consulted this file. The point store does not need it
// at all: drawing points does not care where they were seen from.
//
// This test predates both the range-image builder and the carve (it arrived with
// the first viewer commit, when it was the only guard available). Keep it for the
// number, which is worth seeing next to a scan; do not let it decide anything.

#pragma once

#include "e57.h"
#include "frame.h"

#include <string>
#include <vector>

namespace check {

enum class Kind {
    Structured,   // a grid is declared, or the geometry is clean
    // The range spread suggests several origins, AND no gridding metadata said
    // otherwise. A label, not a rejection — see the header note on what the
    // measurement is worth. Never produced for a scan that declares a grid: an
    // indexBounds or a row/column index is strong evidence of one setup, and a
    // heuristic that tracks scene scale does not get to overrule it.
    Unified,
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
    // Fraction of populated bins that must be multi-surface before the cloud is
    // LABELLED merged. A genuine single setup produces some multi-surface bins at
    // depth discontinuities — doorways, furniture edges — so this cannot be zero.
    //
    // Not calibrated, and the header note explains why calibrating it would not
    // help: the quantity it thresholds rises with scene scale, so no single value
    // separates one setup from several. Left at its original figure because it now
    // only picks which label a metadata-less scan carries, and moving it would
    // change labels without changing what anything does.
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
    // The fraction came out over the threshold. Carried separately from `kind`
    // because it is true whether or not metadata overruled the label, and it is
    // what the store's kScanLooksMerged flag records.
    bool   looksMerged = false;

    // Positively identified as one structured setup, by declared metadata or by
    // clean geometry. NOT a fitness test: an Ambiguous scan is perfectly
    // indexable and drawable, and so is one labelled Unified. Nothing decides
    // whether to use a scan from this — see indexer::ScanRef::usable.
    bool positivelyStructured() const { return kind == Kind::Structured; }
};

// Metadata-only verdict. Cheap, no point decoding.
Result classifyMetadata(const e57::Scan& s);

// Full check: metadata plus the geometric range-image test, which decodes a
// sample of the scan's points. `reader` must be open on the file `s` came from.
//
// Makes its own decode pass, and is the reference. A caller that already holds a
// sample — because it is also deciding the frame from one — should use
// classifyFromSample instead and save the pass.
Result classify(e57::Reader& reader, size_t scanIndex,
                const Thresholds& t = Thresholds{});

// The same verdict from a sample already in hand: see e57::Reader::sampleXYZ,
// which is what must have produced it.
//
// `frame` is needed because the test bins directions from where the instrument
// actually is, and which of the two frames the points are in decides where that
// is. Passing it in rather than deciding it here is the point of the split — the
// frame decision wants the same sample, so whoever holds the sample decides the
// frame first and hands both on.
//
// A sample from somewhere other than sampleXYZ will still produce a verdict, and
// it will be a verdict about whatever that sample was: a prefix of a merged
// cloud reads as single-origin. The sample has to span the scan.
Result classifyFromSample(const e57::Scan& s, const std::vector<double>& xyz,
                          const viewer::FrameDecision& frame,
                          const Thresholds& t = Thresholds{});

const char* kindName(Kind k);

} // namespace check
