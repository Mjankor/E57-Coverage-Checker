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
    // Within maxRange of at least one setup — the domain. Set by carveTile
    // rather than by evidenceAt, because it is a property of the voxel's
    // position relative to the setups and not of anything a setup saw.
    //
    // It is what separates "nobody observed this" from "this is outside the
    // problem": a zero state means the voxel is not part of the question at
    // all, and exactly kReachable means it is, and nothing observed it.
    kReachable = 1u << 2,
};

// How much of a box lies inside the domain.
enum class Overlap : uint8_t { None, Partial, Full };

// The region of space the question is being asked about.
//
// The domain has always been the union of the setups' range spheres — the
// largest region anything could be said about. That is rarely the region anyone
// cares about. Scanning a building interior leaves those spheres bulging tens of
// metres through every wall into open air that was never the subject and could
// never have been observed: at 45 m the spheres are of order 10^6 m^3 where the
// building is 10^4, so the answer is dominated, ninety-something per cent of it,
// by outdoors. Every one of those voxels is unknown, correctly and uselessly.
//
// Narrowing the domain therefore does two things at once. It is the largest
// single speed factor available, and it is what makes the unknown count mean
// "space in the building nobody captured" rather than "mostly sky".
//
// Today the shape is a box around the measured returns. It is heading for a
// shrinkwrap of the point cloud — the surveyed envelope, which for an interior
// job is the building itself, and which would exclude the corners of a box that
// no scan ever reached. The interface is deliberately a pair of predicates
// rather than a box, so that when the wrap arrives it is a new Kind here and
// nothing above this has to change: carveTile asks whether a brick is out, in,
// or straddling, and asks about single voxel centres only where the answer was
// "straddling".
struct Domain {
    enum class Kind : uint8_t {
        Unbounded,   // whatever the range spheres reach
        Box,         // an axis-aligned envelope
    };
    Kind   kind  = Kind::Unbounded;
    double lo[3] = {0, 0, 0};
    double hi[3] = {0, 0, 0};

    Overlap testBox(const double blo[3], const double bhi[3]) const {
        if (kind == Kind::Unbounded) return Overlap::Full;
        for (int i = 0; i < 3; ++i)
            if (bhi[i] < lo[i] || blo[i] > hi[i]) return Overlap::None;
        for (int i = 0; i < 3; ++i)
            if (blo[i] < lo[i] || bhi[i] > hi[i]) return Overlap::Partial;
        return Overlap::Full;
    }
    bool contains(double x, double y, double z) const {
        if (kind == Kind::Unbounded) return true;
        return x >= lo[0] && x <= hi[0] && y >= lo[1] && y <= hi[1] &&
               z >= lo[2] && z <= hi[2];
    }
};

// When a voxel has learned enough to stop asking.
//
// Cost is domain voxels times setups in range, and at a thousand setups the
// second factor is what explodes: a voxel in a dense survey can be within range
// of eighty of them. Most are answered by the first or second.
enum class EarlyOut : uint8_t {
    // Ask every setup about every voxel. What the reference does, always.
    None,
    // Stop once a voxel has both evidence bits. Exact: both bits set is the
    // most any number of further setups could produce, so the result is
    // identical to the full OR. Only `setupTests` differs, and that counts work
    // done rather than the size of the question.
    Saturated,
    // Stop at the first bit of evidence of any kind. Faster, and still exact for
    // the reachable and unknown counts — a voxel that any setup said anything
    // about is not unknown, whichever thing was said. But `visible` and
    // `occupied` become lower bounds rather than counts, so this is a mode to
    // ask for, not a default.
    AnyEvidence,
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
    // Extra voxels carved on every face of a tile, beyond the tile proper.
    //
    // A consumer that asks about a voxel's neighbours — which is how the
    // observed frontier is found — needs the voxels just outside the tile to
    // exist. Recomputing them costs (dim+2)^3/dim^3, about 2.4% at 256, and
    // buys an answer that is still identical for any tiling: the alternative,
    // treating a tile edge as a boundary, would make a voxel's classification
    // depend on where the tile seams happened to fall.
    //
    // Apron voxels are carved but not counted: they belong to the neighbouring
    // tile, and counting them here would tally them twice.
    uint32_t apron = 0;

    // The region being asked about. Unbounded reproduces the range spheres.
    Domain domain;

    // Ignored by carveTileReference, which never stops early.
    EarlyOut earlyOut = EarlyOut::Saturated;

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
    // The carved cube, apron included: dim == core + 2 * apron. Indices run
    // over the whole of it, and the tile proper is [apron, apron + core).
    uint32_t dim = 0;
    uint32_t core = 0;
    uint32_t apron = 0;
    double   origin[3] = {0, 0, 0};   // world position of the carved minimum corner
    std::vector<uint8_t> state;

    uint32_t interiorBegin() const { return apron; }
    uint32_t interiorEnd() const { return apron + core; }
    bool isInterior(uint32_t x, uint32_t y, uint32_t z) const {
        return x >= apron && x < apron + core && y >= apron && y < apron + core &&
               z >= apron && z < apron + core;
    }

    size_t index(uint32_t x, uint32_t y, uint32_t z) const {
        return (size_t(z) * dim + y) * dim + x;
    }
    void centre(uint32_t x, uint32_t y, uint32_t z, double voxelSize, double out[3]) const {
        out[0] = origin[0] + (double(x) + 0.5) * voxelSize;
        out[1] = origin[1] + (double(y) + 0.5) * voxelSize;
        out[2] = origin[2] + (double(z) + 0.5) * voxelSize;
    }

    // The voxel's index on the global lattice, which is the same whatever
    // tiling produced it. Anything that has to be reproducible across runs with
    // different tile sizes — a decimation, an export, a comparison — keys off
    // this rather than off (tile, local index).
    void globalIndex(uint32_t x, uint32_t y, uint32_t z, int64_t out[3]) const {
        out[0] = key.x * int64_t(core) + int64_t(x) - int64_t(apron);
        out[1] = key.y * int64_t(core) + int64_t(y) - int64_t(apron);
        out[2] = key.z * int64_t(core) + int64_t(z) - int64_t(apron);
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
    // (voxel, setup) pairs actually evaluated. A measure of work done, so early
    // exits and brick culling reduce it — it is not a property of the site, and
    // it is the one statistic that legitimately differs between the fast path
    // and the reference.
    uint64_t setupTests = 0;
};

// Tiles whose cube intersects at least one setup's range sphere. Sorted, so a
// run is reproducible and can be resumed by index.
std::vector<TileKey> tilesForSetups(const std::vector<SetupView>& setups, const Params& p);

// Which setups can reach a tile. Everything else is skipped for that tile.
std::vector<size_t> setupsForTile(const TileKey& key, const std::vector<SetupView>& setups,
                                  const Params& p);

// Carves one tile. Allocates and fills `out`.
//
// Two implementations, and they must agree bit for bit — test_carve asserts it,
// and every optimisation added to the fast path has to keep asserting it.
//
//   carveTileReference is the oracle: one voxel at a time, tested against every
//   setup that can reach it, no early exits, no cleverness. It is never deleted
//   and never optimised. Its job is to be obviously correct.
//
//   carveTile is what runs. It computes the same thing in an order that suits
//   the machine: one setup at a time so a single range image is resident, and
//   in bricks so the image cells a brick projects onto stay in L1. Measured on
//   a real 2500 x 5280 raster, roughly half the cost of a voxel test was cache
//   misses on the range image, and the same lookups made coherent are 14 times
//   cheaper.
void carveTile(const TileKey& key, const std::vector<SetupView>& setups,
               const Params& p, Tile& out, Stats& stats);
void carveTileReference(const TileKey& key, const std::vector<SetupView>& setups,
                        const Params& p, Tile& out, Stats& stats);

// Voxels per brick edge, the unit of traversal in carveTile. Eight at 5 cm is a
// 40 cm cube: at 20 m it subtends about a degree, which on a 2500 x 5280 raster
// is roughly 17 x 28 cells — under 2 KB, so the whole brick reads out of L1.
// Larger bricks project onto more image than a cache line pass can hold; smaller
// ones stop amortising the per-brick range test.
constexpr uint32_t kBrickVoxels = 8;

// Carves every tile in the domain, handing each finished tile to `sink` and
// then discarding it. Return false from `sink` to stop early.
using TileSink = bool (*)(const Tile&, void* user);
Stats carveAll(const std::vector<SetupView>& setups, const Params& p,
               TileSink sink, void* user);

} // namespace carve
