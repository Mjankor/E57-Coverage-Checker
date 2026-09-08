// The visibility carve: which space did the scanners actually observe?
//
// This is the CPU reference. It is written for obviousness rather than speed —
// no early exits, no cleverness, every voxel tested against every setup that
// could reach it — because its job is to be the oracle the Metal kernel is
// asserted bit-exact against. It is not the thing that will process a corpus.
//
// Structure: the outer loop is over space in tiles, the inner loop over the
// setups that reach each tile, one range image at a time.
//
//   Why not per-setup, carving a 45 m sphere around each? Visibility is an OR
//   across setups, so accumulating scan by scan is mathematically the natural
//   formulation and it is exactly correct. It just does not bound anything: a
//   voxel seen by setup 7 and again by setup 400 has to hold setup 7's verdict
//   until setup 400 runs, so peak memory is the whole grid regardless. And a
//   45 m sphere at 5 cm is 5.8 x 10^9 voxels — bigger than most whole-site
//   grids — paid once per setup, with neighbouring spheres overlapping almost
//   completely.
//
//   Tiling space instead keeps the same inner loop but retires each voxel the
//   moment its tile is done: bounded working set (one tile plus one range image
//   at a time), no merge pass, resumable a tile at a time, and parallel across
//   tiles with nothing shared.
//
// The domain is the union of the setups' range spheres. Nothing outside it is
// knowable, so tiles that intersect no sphere are never created — which is what
// stops a sprawling or linear site from costing anything across its empty
// stretches.

#pragma once

#include "frame.h"
#include "range_image.h"

#include <cstdint>
#include <string>
#include <vector>

namespace carve {

// Per-voxel evidence. Visible and Occupied are independent predicates rather
// than exclusive states: a voxel on a surface edge can genuinely be both —
// one setup saw through it, another measured a surface in it — and forcing a
// precedence there would either punch holes in the surface or erode space that
// was demonstrably seen through.
enum Bits : uint8_t {
    kVisible  = 1u << 0,   // some setup had line of sight through it
    kOccupied = 1u << 1,   // some setup measured a surface inside it
};

struct Params {
    double voxelSize = 0.05;
    // Half a voxel diagonal. Keeps the voxel holding the measured surface out
    // of the visible set: without it the carve eats the very surfaces it is
    // measuring against.
    double surfaceMargin = 0.5 * 0.05 * 1.7320508075688772;
    // How far a no-return ray clears — the scanner's rated maximum. Also
    // bounds how far any setup's evidence reaches.
    double maxRange = 45.0;
    // Voxels per tile edge. 256 at 5 cm is a 12.8 m tile: 16.7 M voxels, 16 MB
    // of state. Small enough to stream, large enough that per-tile setup
    // selection is not the dominant cost.
    uint32_t tileVoxels = 256;

    double tileMetres() const { return voxelSize * double(tileVoxels); }
};

// A range image placed in the world. The carve needs the world-to-scanner
// transform, which is the inverse of the pose the file supplies.
struct SetupView {
    const rimg::RangeImage* image = nullptr;
    viewer::Rigid worldToScanner;      // inverse pose
    double        origin[3] = {0, 0, 0};   // setup position in world
};

// Builds the view from the image's own pose.
SetupView makeSetupView(const rimg::RangeImage& image);

// What one setup can say about one world point. Returns the evidence bits it
// contributes — zero when it says nothing, which is the common case.
uint8_t evidenceAt(const SetupView& s, const Params& p,
                   double wx, double wy, double wz);

// A tile is a cube of the global voxel lattice, which is anchored at the world
// origin so a tile's contents never depend on which setups were processed.
struct TileKey {
    int64_t x = 0, y = 0, z = 0;
    bool operator==(const TileKey& o) const { return x == o.x && y == o.y && z == o.z; }
    bool operator<(const TileKey& o) const {
        if (x != o.x) return x < o.x;
        if (y != o.y) return y < o.y;
        return z < o.z;
    }
};

struct Tile {
    TileKey  key;
    uint32_t dim = 0;
    double   origin[3] = {0, 0, 0};   // world position of the minimum corner
    std::vector<uint8_t> state;

    size_t index(uint32_t x, uint32_t y, uint32_t z) const {
        return (size_t(z) * dim + y) * dim + x;
    }
    void centre(uint32_t x, uint32_t y, uint32_t z, double voxelSize, double out[3]) const {
        out[0] = origin[0] + (double(x) + 0.5) * voxelSize;
        out[1] = origin[1] + (double(y) + 0.5) * voxelSize;
        out[2] = origin[2] + (double(z) + 0.5) * voxelSize;
    }
};

struct Stats {
    // Every voxel of every carved tile. Tiles are cubes, so this counts the
    // corners of tiles that only clip the domain too — it is a measure of work
    // done, and it changes with `tileVoxels`.
    uint64_t voxels = 0;
    // Voxels within maxRange of at least one setup: the domain proper. Unlike
    // `voxels` this is a property of the site, not of the tiling.
    uint64_t reachable = 0;
    uint64_t visible = 0;
    uint64_t occupied = 0;
    // Reachable, but no setup said anything about it: seen through by none,
    // measured by none. These are the candidate voids — the deliverable.
    uint64_t unknown = 0;
    uint64_t setupTests = 0;   // (voxel, setup) pairs actually evaluated
};

// Tiles whose cube intersects at least one setup's range sphere. Sorted, so a
// run is reproducible and can be resumed by index.
std::vector<TileKey> tilesForSetups(const std::vector<SetupView>& setups, const Params& p);

// Which setups can reach a tile. Everything else is skipped for that tile.
std::vector<size_t> setupsForTile(const TileKey& key, const std::vector<SetupView>& setups,
                                  const Params& p);

// Carves one tile. Allocates and fills `out`.
void carveTile(const TileKey& key, const std::vector<SetupView>& setups,
               const Params& p, Tile& out, Stats& stats);

// Carves every tile in the domain, handing each finished tile to `sink` and
// then discarding it. Return false from `sink` to stop early.
using TileSink = bool (*)(const Tile&, void* user);
Stats carveAll(const std::vector<SetupView>& setups, const Params& p,
               TileSink sink, void* user);

} // namespace carve
