// The visibility carve on the GPU.
//
// A gather kernel in the sense CLAUDE.md asks for: one thread owns one voxel,
// reads the range image, and writes its own byte once. No scatter, no atomics on
// the result — only on the work counter, and that through the simdgroup
// reduction the house pattern prescribes. That is what makes the GPU answer
// comparable to the CPU one at all: the order threads run in cannot affect it.
//
// It installs itself as a carve::TileCarver, which is a "try" rather than a
// "do". No Metal device, a buffer that will not allocate, a command buffer that
// errors, a scan whose raster is too large to upload — every one of those
// returns false and the tile is carved on the CPU instead. The CPU path is not a
// fallback that might be missing; it is the thing that always works, and this is
// an accelerator bolted alongside it.
//
// On exactness. Metal has no double, so the kernel works in float. The CPU
// reference works in double, so the two are NOT bit-identical, and claiming
// otherwise would be the exact failure CLAUDE.md warns about. What can be said
// is where they can differ: float carries about seven significant digits, so at
// 45 m a range is good to a few microns and an angle to about 1e-7 rad, which is
// 1e-4 of a raster cell. A voxel's classification therefore differs only when
// its projection falls within roughly a ten-thousandth of a cell of a cell
// boundary, or when its range sits within microns of a surface plus or minus the
// 4 cm margin. tools/validate_carve_kernel.py quantifies that against a replica
// of both, and `e57cov carve --verify-gpu` (on a Mac) counts real disagreements
// tile by tile.
//
// Everything is computed in the tile's local frame rather than in world
// coordinates, which is not an optimisation but a requirement: float32 at UTM
// magnitudes has a resolution of about 6 cm, larger than a voxel, so a kernel
// fed world positions would quantise the grid it is meant to be resolving. The
// setup position, the world-to-scanner transform and the domain box all arrive
// pre-translated into the tile's frame, where nothing exceeds a few tens of
// metres.

#pragma once

#import <Foundation/Foundation.h>

#include "../src/carve.h"

@interface CarveGpu : NSObject

// The device singleton, or nil when there is no Metal device, no command queue,
// or the shaders will not compile. A nil result is not an error to report — it
// means the CPU path runs, which it was going to do correctly anyway.
+ (instancetype)shared;

// Why it is unavailable, for the status line. Empty when it is available.
+ (NSString *)unavailableReason;

// The carve::TileCarver entry point. Pass `user` as the CarveGpu instance.
+ (carve::TileCarver)carver;

// Bytes currently held in uploaded range images, and how many tiles this
// instance has carved.
@property (nonatomic, readonly) uint64_t residentBytes;
@property (nonatomic, readonly) uint64_t tilesCarved;
@property (nonatomic, readonly) uint64_t tilesRefused;

@end
