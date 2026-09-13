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
// slab against the corridor wall and not as a room.
//
// So the VOLUME is closer to `buffer x (area of surface not seen from both
// sides)` than to a volume of space, and it scales with the buffer accordingly:
// on one real setup, 3,694 m^3 at a 1 m buffer, 8,029 at 2 m, 17,617 at 4 m —
// slightly over double each time. Read on its own it says more about the buffer
// than about the survey.
//
// The FRACTION is the number that means something, and it is worth saying why
// rather than leaving it to be discovered. Across the same buffers it moves only
// 27.9, 30.5, 34.3 per cent — it drifts up, because a wider buffer reaches
// further behind surfaces where there is nothing to see while the lit side
// saturates, but slowly. Across coverage it moves the way a coverage measure
// should: on a synthetic hall, 32.1 per cent at one setup, 14.4 at four, 8.1 at
// nine, 5.0 at sixteen. Seven times the swing from coverage against one and a
// fifth from the buffer, which is what makes it a statement about the survey.
//
// One thing it is NOT is an artefact of the voxel size: the same scan gives
// 7,635 m^3 at 0.5 m voxels, 8,029 at 0.25 and 8,217 at 0.125. It converges,
// so it is measuring a shape rather than a sampling.
//
// Nor is the volume behind the ground a special case worth clipping out. Every
// surface seen from one side only gets a buffer's depth of unobserved space
// behind it; the ground merely has the most area. Telling ground from a floor,
// a deck or a soffit in order to treat it differently would be a classifier
// standing in for the thing the buffer already says plainly.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace wrap {

struct Options {
    // How far past the last measured return the question still applies, in
    // metres, and the only parameter with a physical meaning — everything else
    // here is derived from its magnitude.
    //
    // SIGNED, and the sign chooses between two shapes rather than scaling one:
    //
    //   POSITIVE — a shell around the measured surfaces: every cell within the
    //   buffer of a return. A survey that looks outward wants this. The middle of
    //   a large room is not in it, which is right for the question: unobserved
    //   space collects against surfaces, and the volume is closer to buffer x
    //   (area seen from one side) than to a volume of space.
    //
    //   NEGATIVE — the region the survey ENCLOSES, pulled in by the magnitude, so
    //   the boundary sits that far inside the outer face of the walls. Everything
    //   outside the building, and the wall itself, is then out of the question,
    //   which is what a job conducted entirely indoors wants — said as a distance
    //   instead of as a switch, and with the wall thickness under the operator's
    //   hand. It needs the flood that finds the outside, so it implies
    //   interiorOnly and does not need it set.
    //
    //   DECIDED ONE ENCLOSED REGION AT A TIME. A site is not one building: it is a
    //   building, a boundary wall with nothing behind it, a canopy, and a shed
    //   whose door stood open while the survey ran. A region deep enough for the
    //   erosion to leave a core is pulled in; a region too thin, or one the flood
    //   got into, keeps the ordinary skin of the same width. So the answer never
    //   fails — the worst case is a region asked about too generously — and a leak
    //   in a shed can no longer take the question away from the building.
    //
    // The magnitudes are not comparable across the sign: +2 asks about a two
    // metre skin around everything measured; -2 asks about a whole interior less
    // a two metre skin. Both are useful, and they are different questions.
    double buffer = 2.0;

    // How wide an opening the shell may bridge, in METRES OF OPENING. Zero leaves
    // the measured surfaces as they are.
    //
    // The width of the hole, not a radius: a doorway is 0.9, a window 1.5, a
    // shopfront 3. That is the number an operator can measure on site, so it is
    // the number asked for; the closing radius underneath is half of it, and the
    // cell count underneath that is nobody's business.
    //
    // NOT A CLOSING, though that is the textbook answer and it was tried. Dilate
    // by r and erode by r fills a hole in a SOLID; it cannot fill one in a
    // surface, because a ball of radius r always fits through a hole of radius a
    // by sitting at sqrt(r*r - a*a) from the plane, so the erosion takes back
    // exactly what the dilation bridged. Measured on a wall with a 1.2 m window:
    // at a 1.4 m closing, 24 cells filled, and the middle of the window still open
    // at every radius tried.
    //
    // TOPOLOGY INSTEAD. A barrier half an opening wide blocks a path through it,
    // so whatever the flood cannot reach from the grid's boundary is enclosed —
    // and the cells of that enclosed region that touch the outside are the shell's
    // surface, openings included. No ball has to fit anywhere. That surface is
    // what the buffer is then measured from, so the shell crosses a window instead
    // of following the reveal inward and threading the wrap into the room behind.
    //
    // Which means this only means anything for a shell that ENCLOSES something. A
    // lone wall with a hole in it has no inside, so there is nothing for an
    // opening to be an opening into, and nothing is bridged.
    //
    // Ask for a little more than the widest hole to be closed: the barrier has to
    // exceed half the opening, so 1.2 does not quite close a 1.2 m window and 1.6
    // does. Too large starts bridging things that are genuinely apart, a lane
    // between two buildings being the obvious one.
    double spanGaps = 0.0;

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
    // The flood came in through a hole in the survey and reached a cell an
    // instrument was standing in. Nothing that happened inside a building is
    // outside it, so this says a shell somewhere did not hold.
    //
    // NOT FATAL, under a negative buffer. It used to be: one leak anywhere took
    // the pull-in away from the whole site, so a shed with its door open cost the
    // building its question. Each enclosed region is now assessed on its own, and
    // a region the flood got into simply keeps the ordinary skin while its
    // neighbours are still pulled in. Read this as "somewhere here is more
    // generous than it looks", not as a failure.
    bool     sealLeaked = false;
    double   seal = 0.0;           // the sealing radius actually used, in metres
    // The closing that bridged the openings, in metres, and the cells it added to
    // the occupancy — a window, a doorway, a stretch of wall nobody reached. See
    // Options::spanGaps.
    double   spanGaps = 0.0;
    uint64_t bridgedCells = 0;
    // The shell was pulled IN rather than grown out: the buffer was negative, so
    // the domain is what the survey encloses less that much. See Options::buffer.
    bool     pulledIn = false;
    // What the per-region assessment decided, under a negative buffer. Cells in
    // regions deep enough to pull the boundary into, and the surfaces that bound
    // no such region and kept the ordinary skin instead — a freestanding wall, a
    // canopy, a room the flood got into. Both together cover the site: a region
    // asked about too generously is the worst case, never no question at all.
    uint64_t pulledInCells = 0;
    uint64_t skinnedSurfaces = 0;
    // What the assessment had to work with, which is what says WHY a negative
    // buffer did little.
    //
    // `setupsPulledIn` against `setupsSeen` is the one to read: how many
    // instruments ended up standing in a region the pull-in kept. None of them,
    // and the outside rolled into the building — the opening the shell may bridge
    // is narrower than the way in. Measured on an open-ended 2.8 m corridor: at a
    // 1.6 m bridge nothing is pulled in and the setup is outside its own corridor;
    // at 3.0 m, where the ball no longer fits down it, 30,870 cells are kept and
    // the setup is inside. The other two are supporting detail — space the flood
    // could not reach, and how much of it was deep enough to keep.
    uint64_t enclosedCells = 0;
    uint64_t keptInCells = 0;
    uint64_t setupsSeen = 0;
    uint64_t setupsPulledIn = 0;
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

    // Cell-level readers, so a caller can draw the wrap without knowing how the
    // bits are packed. Out-of-range indices read false rather than out of the
    // array: a cell beyond the grid is beyond the question, which is the same
    // answer contains() gives for a position beyond it.
    bool cellInDomain(int64_t x, int64_t y, int64_t z) const;
    bool cellOccupied(int64_t x, int64_t y, int64_t z) const;
    void cellCentre(int64_t x, int64_t y, int64_t z, double out[3]) const {
        out[0] = (double(x + lo[0]) + 0.5) * cell;
        out[1] = (double(y + lo[1]) + 0.5) * cell;
        out[2] = (double(z + lo[2]) + 0.5) * cell;
    }
};

// Marks every cell a scan's returns fall in.
//
// Reads the range image rather than the points: the points are decoded, posed
// and gone by the time the site's extent is known, and the image is the same
// information already in hand. EVERY cell holding a return is marked — see
// markScan for why sampling a fraction of them, however carefully the fraction
// was chosen, opened holes in the wrap over exactly the surfaces that matter
// most.
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
    // Returns false when the cell holds no return. `image` is whatever the
    // caller put in `imageForStatus`, passed through untouched — it is separate
    // from `user` only so a caller can keep its precomputed tables in one and
    // the thing it is reading statuses from in the other.
    bool (*pointAt)(const void* user, const void* image, uint32_t row, uint32_t col,
                    double out[3]) = nullptr;
    const void* user = nullptr;
    const void* imageForStatus = nullptr;
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
