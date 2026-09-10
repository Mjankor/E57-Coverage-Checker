// The shrinkwrap: which region of space the question is actually about.
//
// The domain has been a box around the measured returns, and a box is mostly
// wrong. A survey of a street reaches forty-five metres in the directions it
// could see and a few metres in the ones it could not, and the box that contains
// all of that also contains the corners no scan ever reached — every voxel of
// which comes out unobserved, correctly and uselessly. On the job this was built
// against, the box is 186,697 m^3 where the range spheres are 833,020: better,
// and still mostly air nobody asked about.
//
// What is worth asking about is the space near the surfaces that were actually
// measured. This builds that region, as a coarse binary occupancy grid over the
// site — which cells hold returns — and then a Euclidean distance transform of
// it, so "within `buffer` of something the survey saw" is a threshold on a
// distance field rather than a shape anybody has to reconstruct.
//
// WHY NOT A SURFACE. The obvious alternative is a Poisson reconstruction of the
// point cloud, and it is the wrong tool three times over. It needs oriented
// normals, which we do not have and cannot estimate reliably at grazing
// incidence or on foliage. It produces a watertight surface, which for an
// outdoor survey seals the site into a bubble and for an indoor one closes the
// doorways and windows the scanner genuinely saw through. And the innermost
// carve loop cannot ask a mesh whether a point is inside it, so the mesh would
// have to be voxelised anyway — at which point the voxels are the answer and the
// mesh was scaffolding. At the spacing a wrap wants, Poisson is not
// reconstructing a surface, it is low-pass filtering occupancy. This does that
// directly, deterministically, and about four orders of magnitude faster.
//
// TWO KINDS OF SURVEY, one switch. A survey that looks outward — a street, a
// yard, a building seen from outside — wants the buffer on both sides of every
// surface: the shadow behind a wall is part of the answer, and the buffer is
// what stops it running to the horizon. A survey conducted entirely inside a
// building wants the space outside the walls left out, not because it is far but
// because it is not the question, and because a shell of unobserved voxels
// wrapped round the outside hides everything within it.
//
// That is a topological distinction rather than a geometric one, which is what
// makes it cheap: flood the grid inward from its own boundary through everything
// the survey did not seal, and whatever the flood reaches is outside. See
// Options::interiorOnly.
//
// WHAT THIS CANNOT REPORT, and it matters when reading the number. Unobserved
// space is only asked about within `buffer` of something that was seen. A room
// nobody entered, behind a door nobody opened, is reported as a `buffer`-thick
// slab against the corridor wall and not as a room. The volume that comes out is
// therefore closer to `buffer x (area of surface not seen from both sides)` than
// to a volume of space: it says where coverage stops, which is the question, and
// doubling the buffer roughly doubles it, which is why the buffer is chosen for
// what you want to see rather than for accuracy.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace wrap {

struct Options {
    // How far past the last measured return the question still applies. The
    // dilation radius, in metres, and the only parameter with a physical meaning
    // — everything else here is derived from it.
    double buffer = 2.0;

    // Cell size, in metres. Zero derives it from the buffer.
    //
    // Derived as buffer/4, so the dilation has four cells of radius to work with
    // and comes out round rather than octagonal. Smaller would resolve detail the
    // buffer is about to swallow; larger would make the wrap's own shape visible
    // in the answer.
    double cell = 0.0;

    // Cells the grid may occupy. A site 200 m across and 20 m tall at half-metre
    // cells is 6.4 M, which is nothing; a square kilometre at the same size is
    // 400 M, which is not. Over budget the cell size is doubled until it fits,
    // and Grid::coarsened records that it happened — a coarser wrap is a blunter
    // question, not a wrong one, but nobody should have to guess that it grew.
    uint64_t maxCells = 64ull << 20;

    // How wide a hole in the building's ENVELOPE the interior-only flood is not
    // allowed through — an external door left open, a window whose glass
    // returned nothing, a stretch of wall the survey never reached. Zero derives
    // it as half the buffer, which closes a hole up to a buffer wide.
    //
    // Not about doorways inside the building: the flood starts outside, so it
    // never reaches one. And sealing removes nothing from the answer either way
    // — the seal is only the flood's barrier, where the domain is still
    // everything within the buffer of a return. See wrap.cpp.
    //
    // It closes holes up to twice this wide. Larger is safer against a leak and
    // costs nothing but a lobe of outside air near a genuine opening, which is
    // kept rather than dropped; too small lets the flood into the building,
    // where it deletes the answer. The two mistakes are not the same size, which
    // is why the default errs generous.
    double seal = 0.0;

    // Leave the space outside the surveyed shell out of the question. See the
    // header note: false for a survey that looks outward, true for one conducted
    // entirely inside a building.
    bool interiorOnly = false;
};

// The occupancy grid and the distance field over it.
//
// Cells are indexed on a global lattice of `cell` metres so the grid's own
// position is a property of the site rather than of the run: shifting the
// extent by a metre moves `lo` and changes nothing else.
struct Grid {
    double   cell = 0.0;
    int64_t  lo[3]  = {0, 0, 0};      // lattice index of the minimum corner
    uint32_t dim[3] = {0, 0, 0};
    // Whether each cell is in the domain. Built by build(); one byte a cell
    // rather than a bit, because the flood needs a third state while it runs and
    // a byte costs 6 MB on a site where the range images cost 44 GB.
    std::vector<uint8_t> inDomain;

    // What it cost and what it decided, for the report.
    uint64_t occupiedCells = 0;    // cells holding at least one return
    uint64_t domainCells   = 0;    // cells inside the wrap
    uint64_t droppedOutside = 0;   // cells the interior-only rule removed
    // The interior-only flood came in through a hole in the survey and reached a
    // cell an instrument was standing in. Nothing that happened inside a building
    // is outside it, so this says the shell did not hold — and when it is set,
    // nothing is dropped: a wrap that is merely too generous is worth having,
    // where one that has deleted the interior is not.
    bool     sealLeaked = false;
    double   seal = 0.0;           // the sealing radius actually used, in metres
    bool     coarsened = false;    // the cell size grew to fit the budget
    bool     interiorOnly = false; // which rule was applied
    double   buffer = 0.0;

    bool empty() const { return inDomain.empty(); }
    size_t index(uint32_t x, uint32_t y, uint32_t z) const {
        return (size_t(z) * dim[1] + y) * dim[0] + x;
    }
    uint64_t cellCount() const {
        return uint64_t(dim[0]) * dim[1] * dim[2];
    }
    // The volume the wrap covers, in cubic metres.
    double volume() const { return double(domainCells) * cell * cell * cell; }

    // Is this world position inside the wrap? Outside the grid is outside the
    // wrap: the grid is padded past everything the survey reached, so a position
    // beyond it is beyond the question too.
    bool contains(double wx, double wy, double wz) const;

    // How much of an axis-aligned box lies inside. 0 none, 1 some, 2 all — the
    // three answers carve::Domain::testBox needs, in the order carve::Overlap
    // declares them.
    int testBox(const double blo[3], const double bhi[3]) const;
};

// Marks every cell a scan's returns fall in.
//
// Reads the range image rather than the points: the points are decoded, posed
// and gone by the time the site's extent is known, and the image is the same
// information already in hand. Cells are sampled with a stride chosen so that
// adjacent sampled rays stay closer together than one grid cell at the furthest
// range the image reaches — so the sampling cannot open a hole in the wrap,
// whatever the raster's resolution.
//
// Safe to call concurrently on the same grid from different scans: marking is an
// OR into a byte, so the result does not depend on which scan got there first or
// on how many threads ran.
void markScan(const struct MarkSource& src, Grid& grid);

// Everything markScan needs from a scan, so wrap.h does not have to include the
// range image and the carve.
struct MarkSource {
    // Cell (row, col) -> world position, supplied by the caller because the
    // transform from a raster cell to a point belongs to range_image and the
    // pose belongs to frame.
    uint32_t rows = 0, cols = 0;
    // Returns false when the cell holds no return.
    bool (*pointAt)(const void* user, uint32_t row, uint32_t col, double out[3]) = nullptr;
    const void* user = nullptr;
    // The widest angular step in the raster, radians, and how far it reaches.
    // Together they set the stride: at `furthest` metres, one step of
    // `angularStep` moves a ray by their product, and the stride is how many
    // steps fit inside one grid cell.
    double angularStep = 0.0;
    double furthest = 0.0;
};

// Sizes the grid to a world box, in metres, padded so its boundary is clear of
// anything the survey reached — which is what lets the interior-only flood start
// from the boundary and mean "outside".
bool size(const double lo[3], const double hi[3], const Options& opt, Grid& grid,
          std::string& err);

// Turns the marked occupancy into the domain: the distance transform, the
// buffer threshold, and — under interiorOnly — the flood that removes the
// outside. Call once, after every scan has been marked.
// `setupsXYZ` is the instrument positions, interleaved, used only to check that
// the interior-only flood stayed outside — see Grid::sealLeaked. Empty skips the
// check, which is the caller saying it has no way to tell.
void build(const Options& opt, Grid& grid, const std::vector<double>& setupsXYZ = {});

} // namespace wrap
