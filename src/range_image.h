// Turning a structured scan into the range image the visibility test needs.
//
// The whole method rests on this (DESIGN.md §2): a structured E57 *is* a range
// image, so instead of marching a ray per point through the voxel grid, voxels
// are projected into the image and compared against the surface that setup saw
// in that direction. This module produces the image.
//
// Two ways in, and which one a file affords decides how much can be trusted:
//
//   Grid path (preferred). The scan declares `indexBounds` and stores
//   `rowIndex`/`columnIndex` per point. The sampling grid is then stated
//   outright, and a cell inside it with no record is a ray that came back
//   empty. Nothing has to be inferred. Real scans do this — a 2500 x 5280 grid
//   holding 5.6 M records means 7.6 M no-returns, mostly sky, and those are
//   exactly the rays that clear the volume above a site.
//
//   Angular path (fallback). No grid metadata, so points are binned by
//   direction and the field of view is recovered from their angular extent.
//   This is where DESIGN.md §4's trap lives: a band of sky returns nothing, so
//   the extent understates the true field of view and that band gets marked
//   OUTSIDE_FOV instead of NO_RETURN — the volume above the site then never
//   clears. Used only when a file gives nothing better, and flagged.
//
// The (row, col) -> (azimuth, elevation) mapping is not declared anywhere, so
// it is measured: the mean elevation of each row and the circular mean azimuth
// of each column, then a line fitted through each. The maximum deviation of the
// measured table from that line is reported, so a scanner that is not a uniform
// raster shows up as a large residual rather than as quietly wrong lookups.

#pragma once

#include "e57.h"
#include "frame.h"

#include <cstdint>
#include <string>
#include <vector>

namespace rimg {

enum class Status : uint8_t {
    Hit        = 0,   // a point was returned in this direction
    NoReturn   = 1,   // the ray was fired and nothing came back — clears to maxRange
    OutsideFov = 2,   // the scanner never looked here — clears nothing
};

// Which end of the raster holds the instrument's blind cone, if any.
enum class BlindCone {
    Auto,        // decide from the geometry — see Options::blindCone
    None,        // there is no blind cone; every empty cell is a no-return
    FirstRows,   // force: the band running off row 0
    LastRows,    // force: the band running off the last row
};

struct Options {
    // How far a no-return ray clears. This is the scanner's rated maximum:
    // past it a return is unlikely to be meaningful, so treating the ray as
    // clearing further would assert emptiness the instrument never established.
    //
    // A setting, not a constant — a different instrument or a job that trusts
    // longer returns changes it, and `Diagnostics::furthestReturn` reports what
    // each scan actually produced so the choice can be checked against the data.
    double   maxRange = 45.0;
    // Cells above this are binned down, taking the MINIMUM range in each bin.
    // Minimum is the conservative direction: it clears less, never more, so a
    // downsampled image cannot carve through a surface it should have kept.
    uint32_t maxCells = 32u << 20;
    // A residual above this means the row/column grid is not a uniform raster
    // and the linear angular model would misplace lookups.
    double   maxMappingResidualRad = 0.02;   // ~1.15 degrees
    // The share of a scan's own points that must land back on their own cell
    // when put through the mapping. Below this the mapping is not describing the
    // raster and the image is refused rather than used to produce confident
    // nonsense. Not 1.0: a point on a cell boundary can legitimately round to
    // its neighbour, and binning down puts several source cells into one.
    double   minRoundTripFraction = 0.90;

    // OFF by default, and deliberately so.
    //
    // The rule this instrument actually follows is simple: a ray either returned
    // or it did not, and one that did not returned nothing because there was
    // nothing within range along it. That holds whether it went to the sky, out
    // of a window, or into a dark surface that reflected too little to register.
    // The carve clears along it either way. Second-guessing that from the shape
    // of the empty region — treating a scattered empty cell as a dropped return
    // rather than as a measurement — substitutes a guess about the instrument
    // for what the instrument reported.
    //
    // The machinery is kept because the guess is sometimes wanted: a scan of
    // dark or wet surfaces does drop returns, and each one then clears a pencil
    // of space to maxRange. On a fixture with a scattered 5% of returns removed,
    // believing them all took the drawn frontier from 42,192 voxels to
    // 4,035,105. Set a radius to switch it on; leave it at 0 for what the file
    // says.
    //
    // The one exception, and it is not a guess: the blind cone under the tripod,
    // handled separately below. Those directions were never sampled at all.
    uint32_t noReturnRadius   = 0;
    double   noReturnFraction = 0.75;

    // Treat the unsampled cone about the instrument's rotation axis as a
    // direction the scanner never looked, rather than as a no-return.
    //
    // Every terrestrial scanner has a blind cone at one end of its sweep, where
    // its own body and mount are. Those rows are empty in the grid for a
    // completely different reason from sky: no ray was fired, so nothing was
    // established. Believed as no-returns they clear a cone to maxRange straight
    // through the ground under every setup — DESIGN.md's original trap, and the
    // only place where an empty cell does not mean what the others mean.
    //
    // WHICH end is not assumed. A scanner mounted upside down has its cone
    // pointing up, and one whose producer rewrites the local frame to put world
    // up along +Z has it at the other end of the raster from an upright one. So
    // the end is found from the geometry: just outside the blind cone the beam
    // grazes the mount and lands on the ground a metre or two away, so the
    // returns bordering it are the closest in the scan. Just outside a sky band
    // they are distant or absent. The band bordered by the nearer returns is the
    // cone. That test does not know or care which way up anything is.
    BlindCone blindCone = BlindCone::Auto;
    // How many rows either side of a candidate band to take the median range
    // over. A handful: the ground close to the mount, before the beam flattens
    // out and starts reaching across the site.
    uint32_t blindConeProbeRows = 16;
    // The bordering returns have to be clearly nearer, or the call is refused
    // rather than guessed. Both errors are bad and they are bad in different
    // directions — believing a cone clears space through solid ground, and
    // disbelieving sky loses real coverage — so an unclear case is reported
    // rather than resolved by a coin toss.
    double   blindConeRatio = 0.6;
};

// az(col) = az0 + col * dAzPerCol, el(row) = el0 + row * dElPerRow.
struct Mapping {
    double az0 = 0, dAzPerCol = 0;
    double el0 = 0, dElPerRow = 0;
    double azResidualRad = -1;   // max deviation of the measured table from the line
    double elResidualRad = -1;
    // The fraction of the scan's own points that, put back through cellOf,
    // land on the cell they were decoded from.
    //
    // This is the check that matters, and the residual above is not a substitute
    // for it. A small residual says a line fits the per-row means; it says
    // nothing about whether a lookup in a given direction reaches the right
    // cell. The two come apart whenever the raster is not what the linear model
    // assumes — a scanner that sweeps the mirror through more than 180 degrees
    // covers each column twice, and the fitted line through half of a triangle
    // wave can look perfectly good while sending every lookup in the upper half
    // of the scan to a cell in the lower half.
    //
    // A voxel looking at the sky then samples a cell containing ground and comes
    // out unknown; a voxel inside a building samples a cell containing sky and
    // gets cleared. Both were observed in the field before this was measured.
    double roundTripFraction = -1.0;
    bool   valid = false;
};

struct Diagnostics {
    bool     usedGrid = false;          // grid path rather than angular fallback
    uint64_t hits = 0;
    uint64_t noReturns = 0;
    uint64_t outsideFov = 0;
    double   fillFraction = 0;
    // What this scan actually measured. Named apart from Options::maxRange on
    // purpose: that is a setting about how far to trust an empty ray, these are
    // observations, and conflating the two is how a setting quietly becomes a
    // conclusion.
    double   nearestReturn = 0;
    double   furthestReturn = 0;
    // Rows at the very top or bottom of the grid holding no points at all.
    // Ambiguous by nature: either an all-sky band (genuine no-returns, and the
    // grid path treats them as such) or part of the grid the scanner never
    // sampled, such as a nadir blind cone. Reported so it can be judged rather
    // than assumed away.
    uint32_t emptyLeadingRows = 0;
    uint32_t emptyTrailingRows = 0;
    // Bounding box of the returns, in the SCANNER's frame. This is the scan's
    // own statement of what it actually reached, and it is what a domain narrower
    // than the range sphere is built from — the file's declared cartesianBounds
    // would do for a conformant writer, but this is measured from the points
    // that survived decoding rather than taken on trust.
    // No-returns that failed the neighbourhood test and were demoted to
    // OutsideFov. A large number here means the scan drops returns, which is
    // worth knowing about the instrument and the surfaces, not just about this
    // run: every one of them would otherwise have cleared space to maxRange.
    // No-returns demoted to OutsideFov, and why.
    uint64_t isolatedNoReturns = 0;   // by the optional neighbourhood filter
    uint64_t blindConeCells    = 0;   // the instrument's own blind cone
    uint32_t blindConeRows     = 0;
    bool     blindConeAtFirstRow = false;
    // The evidence the decision was made on: the median range of the returns
    // bordering each candidate band, in metres. -1 where there was no band.
    double   borderRangeFirst = -1.0;
    double   borderRangeLast  = -1.0;
    // The cone's axis in the FILE's frame, once the pose is applied — which is
    // to say, which way the instrument was actually pointing. Its z component
    // says whether this setup was upright or inverted, and that is worth seeing
    // in a corpus: a scan mounted the wrong way up is a real thing that happens
    // and it is otherwise invisible.
    bool     hasConeAxis = false;
    double   coneAxisWorld[3] = {0, 0, 0};
    // Points whose declared row/column fell outside the declared grid. A few are
    // ordinary; a large share means indexBounds does not describe this scan, and
    // the cells those points should have filled stay empty — which reads as
    // no-returns and clears space to maxRange. Silently dropping them was the
    // most dangerous of the reader's quiet failures.
    uint64_t outsideGrid = 0;
    // The scanner's own position relative to the returns, in the scanner frame.
    // Returns surround the instrument, so the origin should sit inside their
    // box. When it does not, the frame decision was probably wrong and every
    // lookup is being made from the wrong place.
    bool   originInsideReturns = true;
    bool   hasReturnBounds = false;
    double returnMin[3] = {0, 0, 0};
    double returnMax[3] = {0, 0, 0};
    std::string note;
};

// A min/max pyramid over the raster, so a question about a whole region can be
// answered without reading every cell in it.
//
// This is what turns the carve from "test every voxel" into "test the ones that
// matter". A brick of voxels projects to a rectangle of cells; if every surface
// in that rectangle is nearer than the brick's closest corner, the whole brick
// is behind everything and no setup evidence can reach it. If every surface is
// further than the brick's furthest corner, the whole brick is in clear view.
// Either way 512 voxels are settled by one lookup.
//
// It must be a true min and max, not an average — the tests above are only
// conservative if the bounds really bound. (This is the trap in reaching for a
// hardware mipmap generator: those box-filter, and a box filter here would claim
// surfaces at distances nothing measured.)
//
// Level 0 aggregates kPyramidBase^2 cells and each level after halves both axes,
// so the whole pyramid costs about 5/16 of one byte per cell — a third of what a
// level-0-is-the-image pyramid would, and the fine levels it skips are finer
// than any brick ever asks for.
constexpr uint32_t kPyramidBase = 4;

// Which statuses occur somewhere in a node's region.
enum StatusBits : uint8_t {
    kHasHit        = 1u << 0,
    kHasNoReturn   = 1u << 1,
    kHasOutsideFov = 1u << 2,
};

struct PyramidLevel {
    uint32_t rows = 0, cols = 0;   // nodes, not cells
    uint32_t block = 0;            // cells per node edge
    std::vector<uint16_t> minCm, maxCm;
    std::vector<uint8_t>  statuses;
};

struct RangePyramid {
    std::vector<PyramidLevel> levels;
    bool empty() const { return levels.empty(); }
};

// The answer to a region query: the extreme surface distances anywhere in it,
// and which statuses appear. Always a conservative superset of the requested
// rectangle — a wider answer can only make the caller more cautious.
struct RangeSpan {
    double  minRange = 0;
    double  maxRange = 0;
    uint8_t statuses = 0;
    bool    valid = false;
};

// One raster cell: what the scanner did in this direction, and how far away the
// answer was. Range is in centimetres, which is well under a 5 cm voxel.
//
// Packed into one three-byte record rather than kept as parallel range and
// status arrays. Every lookup wants both fields at the same index, and two
// arrays meant two cache lines touched for one question. Three bytes rather
// than a padded four keeps the image the same size it was: a 2500 x 5280 raster
// is 38 MB either way, and at that size the extra 13 MB of padding would cost
// more than the alignment saved.
#pragma pack(push, 1)
struct Cell {
    uint16_t rangeCm = 0;
    uint8_t  status  = uint8_t(Status::NoReturn);
};
#pragma pack(pop)
static_assert(sizeof(Cell) == 3, "Cell must stay three bytes");

struct RangeImage {
    uint32_t rows = 0, cols = 0;
    std::vector<Cell> cells;
    Mapping     map;
    Diagnostics diag;

    // The setup's position and orientation in the file frame. Voxels are
    // transformed into the scanner frame with the inverse of this.
    e57::Pose pose;
    bool      hasPose = false;

    size_t cellCount() const { return size_t(rows) * size_t(cols); }
    Status statusAt(uint32_t row, uint32_t col) const {
        return Status(cells[size_t(row) * cols + col].status);
    }
    double rangeAt(uint32_t row, uint32_t col) const {
        return double(cells[size_t(row) * cols + col].rangeCm) * 0.01;
    }

    // Direction in the scanner frame to a cell. Returns false outside the grid.
    bool cellOf(double az, double el, uint32_t& row, uint32_t& col) const;

    // The visibility test's primitive: what did this setup see in this
    // direction? `range` is meaningful for Hit and NoReturn.
    bool sample(double az, double el, Status& st, double& range) const;

    // Unclamped, unwrapped raster coordinates for an angle. Used to bound a
    // region rather than to look a single direction up, so they deliberately
    // return values outside the raster instead of failing.
    double rowCoord(double el) const;
    double colCoord(double az) const;

    // Built on demand; empty until then, and everything still works without it.
    RangePyramid pyramid;

    // Extremes over an inclusive rectangle of cells. Rows are clamped to the
    // raster. Columns are a raw range that may run negative or past `cols`, and
    // wrap is handled by widening to the whole circle — bricks near the seam are
    // rare and a wider answer is still a safe one.
    RangeSpan span(int64_t row0, int64_t row1, int64_t col0, int64_t col1) const;
};

// Converts a scanner-frame position to (azimuth, elevation, range). Elevation
// is measured from the XY plane, matching E57's spherical convention.
void toSpherical(double x, double y, double z, double& az, double& el, double& range);

bool build(e57::Reader& reader, size_t scanIndex, const Options& opt,
           RangeImage& out, std::string& err);

// Builds the min/max pyramid. Not done inside build() because it is an
// accelerator, not part of the image: a caller that only samples directions has
// no use for it and should not pay the memory.
void buildPyramid(RangeImage& im);

// Exposed for testing.
void filterIsolatedNoReturns(RangeImage& im, const Options& opt);
void markBlindCone(RangeImage& im, const Options& opt);

const char* statusName(Status s);

} // namespace rimg
