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
};

// az(col) = az0 + col * dAzPerCol, el(row) = el0 + row * dElPerRow.
struct Mapping {
    double az0 = 0, dAzPerCol = 0;
    double el0 = 0, dElPerRow = 0;
    double azResidualRad = -1;   // max deviation of the measured table from the line
    double elResidualRad = -1;
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

const char* statusName(Status s);

} // namespace rimg
