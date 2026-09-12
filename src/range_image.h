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
// The (row, col) -> (azimuth, elevation) mapping is not declared anywhere, so it
// is measured: the mean elevation of each row and the circular mean azimuth of
// each column. Those tables are then used as they are, rather than having a line
// fitted to them — see Mapping, where the reason is a field report, not a
// preference.

#pragma once

#include "e57.h"
#include "frame.h"

#include <cmath>
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
    // Both bands unsampled. For a survey where the corpus finds a FIXED band at
    // each end: a band that is the same in every scan is not the scene, and that
    // is as true of two bands as of one. An instrument level in a building sees
    // its own mount one way and, at a constant height, the ceiling the other.
    BothEnds,
};

// How a corpus-wide decision came out, for reporting.
struct ConeVerdict {
    bool     decided = false;      // a corpus-wide end was identified
    bool     atFirstRow = false;
    // Both ends carry a band that is fixed across the corpus, so both are
    // unsampled and `atFirstRow` says nothing. Reported separately because the
    // alternative reading of "the corpus cannot choose an end" used to be
    // "believe both", and believing a band that every scan shares clears a cone
    // through whatever is on the other side of it.
    bool     bothEnds = false;
    uint32_t rowsMin = 0, rowsMax = 0;    // the band's size across the corpus
    uint32_t otherMin = 0, otherMax = 0;  // and the other end's, which is scene
    size_t   scans = 0;
    // Scans whose own bordering-range verdict disagreed with the corpus and were
    // re-marked. The count of scans a single-scan test got wrong, which is the
    // number worth seeing: it was two of five on the job this was built against.
    size_t   corrected = 0;
    std::string why;
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
    // A residual above this is reported as "this is not a uniform raster". It no
    // longer refuses anything: the mapping is a measured table, so a non-uniform
    // raster is handled rather than rejected. Kept because the figure is worth
    // seeing — it is how far wrong a linear model would have been.
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
    // How much the unsampled band at one end of the raster may vary across a
    // corpus and still be read as the instrument's own geometry.
    //
    // The bordering-range test above is the best a single scan can do, and on real
    // outdoor data it is not good enough: on five scans from one job it refused two
    // and called two of the others inverted, because a beam grazing an eave
    // overhead looks exactly like a beam grazing the mount. A corpus settles it
    // outright. The instrument's cone is fixed geometry, so it is the same band in
    // every scan — 590, 591, 591, 590, 591 rows on those five — while the band at
    // the other end is scene and varied 87, 116, 125, 304, 576. One of those is a
    // property of the instrument and the other is a property of where it stood,
    // and telling them apart needs no notion of up, no ground plane and no
    // threshold in metres.
    //
    // A tenth: one row in 591 is 0.2%, and the loosest thing that could still be
    // called fixed geometry is far inside 10%.
    double   coneCorpusSpread = 0.10;

    // The instrument's unsampled cone, as a half-angle measured from NADIR, in
    // degrees, and how far a measured band may sit from it and still be called
    // that cone.
    //
    // This is the strongest signal available for which end of the raster is the
    // instrument, and it is a property of the hardware rather than of the site: a
    // scanner cannot see its own mount, so a fixed cone about its own downward
    // axis is missing from every scan it ever takes. Elevation is measured in the
    // instrument's OWN frame, so this needs no notion of world up, no ground
    // plane and no pose — see markBlindCone.
    //
    // A parameter because it differs by model. Forty-five degrees is a common
    // figure and the tolerance is deliberately loose: the point is to tell a cone
    // about nadir from a band of sky at zenith, which differ by ninety degrees,
    // not to measure the cone.
    double   blindConeFromNadirDeg = 45.0;
    double   blindConeAngleTolDeg  = 15.0;
};

constexpr double kTwoPi = 6.28318530717958648;

// Bins per raster cell in the reverse index below. Eight puts the worst
// quantisation error at a sixteenth of a cell, well under the half-cell rounding
// a lookup has anyway, for two int32 tables totalling about 250 kB beside a
// 38 MB image.
constexpr uint32_t kReverseBinsPerCell = 8;

// The (row, column) -> (azimuth, elevation) mapping.
//
// Measured, and used as measured.
//
// There was a straight line here — az0 plus a slope per column, el0 plus a slope
// per row — and lookups inverted it arithmetically. On real scanner output under
// one per cent of each scan's own points could find their own cell again. The sky
// came back unobserved and building interiors came back clear. Two independent
// faults, and the fitted slopes were not either of them:
//
//   The sweep runs past a full turn. 363.8 to 365.6 degrees over 5280 columns on
//   the five scans measured, so the last ~65 columns repeat the bearings of the
//   first ~65. Inverting with a modulo-2pi fold and then a modulo-cols wrap
//   assumes the raster covers exactly one turn, and every column past the fold
//   came out shifted by precisely the excess, 5280 - 2pi/|dAzPerCol|: 65.1, 55.3,
//   67.4, 80.4 and 61.4 columns respectively. Exactly half of every raster.
//
//   Neither axis is uniform anyway. The measured elevation table deviates from
//   its own best-fit line by 3 to 33 rows depending on the scan, against the one
//   cell of slack a lookup can absorb.
//
// Forcing a line on it was never necessary. The per-row elevations and per-column
// azimuths ARE the mapping, measured from the points themselves, so they are kept
// and used directly: gaps interpolated from their neighbours, lookups through a
// reverse index built from the tables. Constant time, and right for any raster the
// instrument actually produced — a sweep past a full turn, a non-uniform step, a
// band the mirror moves through faster — rather than only for the uniform
// single-turn raster a line assumes.
//
// The line is still fitted and still reported, because its residual says how far
// from uniform the raster is. It no longer decides anything.
struct Mapping {
    // The mapping itself: one elevation per row, one azimuth per column, with no
    // gaps — a row holding no returns still has to map somewhere. The azimuths
    // are unwrapped, so they run monotonically through the sweep rather than
    // jumping at the seam, and their total span may exceed 2pi.
    std::vector<double> elByRow;
    std::vector<double> azByCol;

    // Reverse index: which row an elevation falls in, which column an azimuth.
    //
    // Elevation bins cover the table's own range plus a cell at each end, so an
    // elevation outside them is a direction off the top or bottom of the raster.
    // Azimuth bins cover the whole turn, because every bearing is somewhere on
    // it; the ones a partial sweep never reached hold -1.
    double  elLo = 0, elBin = 0;
    double  azLo = 0, azBin = 0;
    std::vector<int32_t> rowOfEl;
    std::vector<int32_t> colOfAz;

    // The best-fit line through each table. Reported, not used: a large residual
    // means the raster is not uniform, which is worth knowing and is no longer a
    // reason to refuse it.
    double az0 = 0, dAzPerCol = 0;
    double el0 = 0, dElPerRow = 0;
    double azResidualRad = -1;
    double elResidualRad = -1;
    // What the tables themselves span, end to end. The azimuth figure is how the
    // sweep-past-a-turn shows up: over 2pi means some bearings were looked at
    // twice.
    double azSpanRad = 0;
    double elSpanRad = 0;

    // Whether each table runs one way without turning back. A mirror that sweeps
    // past the pole sends the elevation up and then down again, and flips the
    // azimuth by pi while it does — a mapping no pair of separable tables can
    // express, so it is refused here rather than indexed ambiguously.
    bool monotonicEl = false;
    bool monotonicAz = false;

    // The fraction of the scan's own points that, put back through this mapping,
    // land on the cell they were decoded from. The check that matters: a residual
    // describes a fit, this describes what lookups actually do.
    double roundTripFraction = -1.0;
    // What went wrong when it is low, because a percentage on its own is not
    // something anyone can act on. The 95th percentile of how far a held-back
    // point landed from its own cell, and the share that resolved to no cell at
    // all — a few cells out is a raster that is not quite described; tens of cells
    // is the wrong model; a large lost fraction is a sweep the index does not
    // cover.
    double rowErrorCells = -1.0;
    double colErrorCells = -1.0;
    double lostFraction  = -1.0;
    bool   valid = false;

    // Which row an elevation falls in, and which column an azimuth; -1 when the
    // raster never looked there.
    //
    // Deliberately ungated by `valid`: the round-trip check is what decides
    // `valid`, and it has to be able to ask these first. Callers that care go
    // through RangeImage::cellOf, which does gate.
    int32_t rowFor(double el) const {
        if (rowOfEl.empty() || !(elBin > 0)) return -1;
        const double f = (el - elLo) / elBin;
        if (!(f >= 0.0)) return -1;                 // also rejects NaN
        const size_t b = size_t(f);
        if (b >= rowOfEl.size()) return -1;
        return rowOfEl[b];
    }
    int32_t colFor(double az) const {
        if (colOfAz.empty() || !(azBin > 0)) return -1;
        double d = az - azLo;
        if (!std::isfinite(d)) return -1;
        d -= kTwoPi * std::floor(d / kTwoPi);        // onto the turn the bins cover
        size_t b = size_t(d / azBin);
        if (b >= colOfAz.size()) b = colOfAz.size() - 1;   // only the top edge
        return colOfAz[b];
    }
};

// A uniform single-turn raster, expressed as tables: el(row) = el0 + row*dEl,
// az(col) = az0 + col*dAz. What the old linear model described, and still the
// right thing for a synthetic fixture — the tables are the general case, not a
// different case.
Mapping uniformMapping(uint32_t rows, uint32_t cols,
                       double el0, double dElPerRow, double az0, double dAzPerCol);

// Builds rowOfEl/colOfAz from elByRow/azByCol, and fills in the spans and the
// monotonicity flags. False when the tables are too small or too degenerate to
// index. Does not set `valid` — that is the round trip's call.
bool indexMapping(Mapping& m);

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
    // A SECOND band, at the other end, also marked unsampled. Non-zero only for
    // BlindCone::BothEnds — see that enumerator.
    uint32_t blindConeRowsLast = 0;
    bool     blindConeAtFirstRow = false;
    // The evidence the decision was made on: the median range of the returns
    // bordering each candidate band, in metres. -1 where there was no band.
    double   borderRangeFirst = -1.0;
    double   borderRangeLast  = -1.0;
    // Each end's unsampled band as a half-angle from the pole it runs into,
    // degrees, measured from the elevation of the last row that HAS returns —
    // the rows inside a band have no returns, so their own elevations are
    // interpolated and not evidence. -1 where there was no band.
    double   bandAngleFirstDeg = -1.0;
    double   bandAngleLastDeg  = -1.0;
    // The end was identified from those angles rather than from the range of the
    // returns bordering each band. The better signal, and the one to trust.
    bool     coneByElevation = false;
    // The pose puts this instrument's own up axis pointing downward in the file
    // frame, so the setup looks inverted. REPORTED, not acted on: the mount is
    // fixed at the instrument's own -z whatever the pose does with that frame, so
    // this cannot move the cone for a scan stored scanner-local. It is here
    // because a scan mounted the wrong way up is worth seeing.
    bool     looksInvertedByPose = false;
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
    // How many declared grid cells went into one raster cell, per edge: 1 when
    // the raster was taken at full resolution, and more when it had to be binned
    // down to fit the cell budget. See vis::Options::imageBudgetBytes for why
    // this is worth carrying — binning changes the answer, not just its
    // sharpness, so a run that did it has to be able to say so.
    uint32_t binStep = 1;

    // Cells that received more than one return, and how many of those extra
    // returns were further away than one already held. Together they say which
    // of the two causes is at work: binning several source cells into one gives
    // a mix in both directions, while a multi-return instrument firing one ray
    // at several surfaces gives mostly further-than. Either way the nearest is
    // kept — see the fill in build() — and these are how the report can say so
    // rather than leaving it to be assumed.
    uint64_t cellsWithSeveralReturns = 0;
    uint64_t returnsKeptBehindANearerOne = 0;

    bool   originInsideReturns = true;
    bool   hasReturnBounds = false;
    double returnMin[3] = {0, 0, 0};
    double returnMax[3] = {0, 0, 0};

    // Where the instrument stood, measured from its own returns and from nothing
    // else: no pose, no metadata, no assumption about which frame the points are
    // in. See measureOrigin for how.
    //
    // Expressed in the frame the cells were built about, so a file that means
    // what the standard says reads (0, 0, 0) to within the noise. Anything else
    // is how far out the setup position is — and since this is the only
    // statement about that position which does not come from the pose, it is the
    // only one that can contradict it. `originRms` says how well the lines
    // actually met, which is what decides whether to believe the number.
    bool     haveMeasuredOrigin = false;
    double   measuredOrigin[3] = {0, 0, 0};
    double   originRms   = -1.0;
    uint32_t originPairs = 0;

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

// A process-unique number for the next range image. Never reused, including
// after an image is destroyed — which is the point. See RangeImage::uid.
uint64_t nextImageUid();

struct RangeImage {
    // Identity, for anything that caches per image.
    //
    // A cache keyed on the ADDRESS of a range image is wrong in a way that only
    // shows up on the second run: images are built fresh for each run and freed at
    // the end of it, and the allocator hands the same addresses straight back. A
    // lookup then hits on a pointer that now refers to a different image and
    // returns the previous run's data. This number is never reused, so it cannot.
    //
    // Copies share it deliberately: an image and its copy hold the same content,
    // which is what a cache key is about. Images are built once and read afterwards.
    uint64_t uid = nextImageUid();

    uint32_t rows = 0, cols = 0;
    std::vector<Cell> cells;
    Mapping     map;
    Diagnostics diag;

    // The rotation from the stored frame into the instrument's own.
    //
    // A terrestrial scanner has a dual-axis compensator and exports its points
    // already levelled. So the raster's rows are lines of constant elevation about
    // the INSTRUMENT'S axis, while the points are stored about the vertical, and
    // the two differ by however the tripod happened to be standing. A row's
    // elevation then runs as tau*cos(azimuth - phi) — one cycle per turn — and a
    // row stops being a direction.
    //
    // Measured on five real setups of one job: 0.55, 1.94, 0.97, 2.49 and 1.02
    // degrees, in five different directions, accounting for 80 to 98 per cent of
    // the spread of elevation within a row. Tripod-on-a-driveway numbers, different
    // every time the instrument was moved, which is what says it is the setup and
    // not the instrument.
    //
    // It is a ROTATION, so it leaves no parallax on edges and is invisible in the
    // merged cloud: the manufacturer applied it correctly and the cloud is right.
    // It is only visible to something that goes looking for the raster. It also
    // accounts for the azimuth errors, which a tilt moves by tau*tan(elevation) —
    // nothing at the horizon, and tens of columns near the poles.
    //
    // Identity when the instrument was level, or when the fit does not improve on
    // one, so a file that needs no correction gets none.
    double    tilt[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    double    tiltDeg = 0.0;         // amplitude, for reporting
    double    tiltTowardDeg = 0.0;   // the azimuth it leans toward
    double    tiltExplained = 0.0;   // share of the within-row spread it accounts for

    // Turns a stored-frame direction into the instrument's own frame.
    void toInstrument(double& x, double& y, double& z) const {
        const double a = x, b = y, c = z;
        x = tilt[0] * a + tilt[1] * b + tilt[2] * c;
        y = tilt[3] * a + tilt[4] * b + tilt[5] * c;
        z = tilt[6] * a + tilt[7] * b + tilt[8] * c;
    }
    // And back. A rotation, so the inverse is the transpose. Needed by anything
    // that starts from a CELL — whose direction is in the instrument's frame — and
    // wants the world position that cell was looking at.
    void fromInstrument(double& x, double& y, double& z) const {
        const double a = x, b = y, c = z;
        x = tilt[0] * a + tilt[3] * b + tilt[6] * c;
        y = tilt[1] * a + tilt[4] * b + tilt[7] * c;
        z = tilt[2] * a + tilt[5] * b + tilt[8] * c;
    }

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

// Finds the instrument's blind cone across a whole corpus, then marks it in every
// image. This is the entry point a caller with more than one scan should use;
// markBlindCone below is what it falls back to for a single scan.
//
// The decision is which END of the raster the cone is at, never how many rows:
// each image's own contiguous empty band is what gets marked, so a row that holds
// returns in one scan is never marked unsampled there because it was empty in
// another.
//
// Safe to call on images build() has already marked per scan, and that is the
// intended order. build() keeps deciding for itself, so a single image is never
// left believing a cone it should not; where this disagrees it puts that band back
// as no-returns and marks the other end instead.
ConeVerdict markBlindConeAcrossCorpus(const std::vector<RangeImage*>& images,
                                      const Options& opt);

// The same decision, from nothing but the unsampled bands — {leading, trailing}
// row counts, one pair per scan. Separate from the images because that is all the
// evidence it uses, and a caller that has not got every image in memory at once
// can still reach the same verdict: the bands come from the rowIndex field alone,
// which is a few bits per point against a whole decoded raster.
//
// The band sizes need not be on the same scale as the images' — a binned-down
// raster has proportionally smaller bands — because the test is on their
// consistency across scans, which is scale-free, and what it produces is an END,
// not a row count.
ConeVerdict decideBlindConeEnd(
    const std::vector<std::pair<uint32_t, uint32_t>>& bands, const Options& opt);

// Marks a decided verdict in every image, putting back any band build() claimed at
// the other end. A verdict that decided nothing leaves every image as it is.
void applyBlindConeVerdict(const std::vector<RangeImage*>& images,
                           const ConeVerdict& v, const Options& opt);

// Exposed for testing.
void filterIsolatedNoReturns(RangeImage& im, const Options& opt);
void markBlindCone(RangeImage& im, const Options& opt);

const char* statusName(Status s);

} // namespace rimg
