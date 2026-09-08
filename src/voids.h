// Telling a void from the rest of the world.
//
// The carve answers "did anything observe this voxel?", and over any region
// large enough to contain a building the honest answer is "no" for most of it.
// A box around a street scanned from five setups is 80% unobserved and always
// will be: the inside of the ground, the inside of the neighbours' houses,
// everything past the fence. All of it genuinely unseen, none of it the
// question. A sealed test room scanned from four setups inside it reports
// 928 m^3 unknown, and the two-metre margin shell outside its walls is 936 m^3 —
// the entire answer was the shell.
//
// So "unobserved" is not the deliverable. This is:
//
//   A void that matters is one you cannot reach from outside without crossing
//   space the scanners observed.
//
// That is a connectivity question, not a per-voxel one, and it is why this stage
// cannot live inside the tile loop. Flood the grid from its boundary, passing
// only through voxels nothing observed. Whatever the flood reaches is the
// outside world arriving by some route; whatever it cannot reach is enclosed by
// observed space — a shadow behind a parked car, the volume under an eave, a
// room nobody walked into. Those are the coverage failures.
//
// The rule has a property worth stating: it needs no notion of inside or
// outside, no ground plane, no building model. A void is defined entirely by the
// observation, so a site with no enclosing geometry at all correctly reports
// nothing enclosed, and a leak through a gap in the coverage correctly stops a
// void being a void — because a gap in the coverage means you genuinely do not
// know whether that space connects to the street.

#pragma once

#include "carve.h"

#include <cstdint>
#include <vector>

namespace voids {

// Set on top of carve::Bits. Kept in the same byte so the classification travels
// with the state rather than in a parallel array that could fall out of step.
enum Bits : uint8_t {
    // Reachable from outside the grid through unobserved space: the rest of the
    // world, not a void.
    kExterior = 1u << 3,
};

// A dense grid over the whole carved domain. Dense on purpose: the flood is a
// connectivity question over the entire region at once, and a tile-local
// structure cannot answer it — a void spanning a tile seam is one void.
struct Grid {
    int64_t  lo[3]  = {0, 0, 0};      // global lattice index of the minimum corner
    uint32_t dim[3] = {0, 0, 0};
    double   voxelSize = 0.05;
    std::vector<uint8_t> state;

    size_t index(uint32_t x, uint32_t y, uint32_t z) const {
        return (size_t(z) * dim[1] + y) * dim[0] + x;
    }
    uint64_t voxelCount() const {
        return uint64_t(dim[0]) * uint64_t(dim[1]) * uint64_t(dim[2]);
    }
    // World position of a voxel's centre.
    void centre(uint32_t x, uint32_t y, uint32_t z, double out[3]) const {
        out[0] = (double(lo[0] + int64_t(x)) + 0.5) * voxelSize;
        out[1] = (double(lo[1] + int64_t(y)) + 0.5) * voxelSize;
        out[2] = (double(lo[2] + int64_t(z)) + 0.5) * voxelSize;
    }
};

struct Report {
    uint64_t exterior = 0;          // unobserved and connected to the outside
    uint64_t enclosed = 0;          // unobserved and not: the deliverable
    uint64_t components = 0;        // how many separate enclosed voids
    uint64_t largestComponent = 0;  // voxels in the biggest one
    // Enclosed voxels touching an observed one: the surface of each void, which
    // is what is worth drawing.
    uint64_t enclosedSurface = 0;
};

// Marks every unobserved voxel reachable from the grid's boundary, then counts
// what is left. Modifies `g.state` in place.
Report classify(Grid& g);

// True when the voxel is unobserved and enclosed — the thing to draw and count.
inline bool isEnclosedVoid(uint8_t bits) {
    return (bits & (carve::kVisible | carve::kOccupied)) == 0 &&
           (bits & kExterior) == 0 &&
           (bits & carve::kReachable) != 0;
}

// True when the voxel touches something a scanner observed. An enclosed void's
// surface is where it meets the geometry that hides it.
bool touchesObserved(const Grid& g, uint32_t x, uint32_t y, uint32_t z);

} // namespace voids
