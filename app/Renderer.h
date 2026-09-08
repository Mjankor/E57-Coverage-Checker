// Metal point-cloud renderer, drawing from an mmap'd LOD store.
//
// Nothing is uploaded per frame. Where the store fits inside the device's
// maximum buffer length — the common case, since that is tens of gigabytes on
// Apple silicon — the whole mapping is wrapped once with
// `newBufferWithBytesNoCopy` and each node is drawn straight out of it at its
// own byte offset. The GPU then reads the page cache directly, and the kernel
// pages nodes in and out as the view moves. A store larger than that falls back
// to a byte-budgeted LRU of per-node buffers.
//
// Shaders are compiled from a source string at runtime, following the house
// pattern and removing any dependency on Xcode's .metal build rule.

#pragma once

#import <MetalKit/MetalKit.h>

#include "../src/camera.h"
#include "../src/lod.h"
#include "../src/point_store.h"

#include <vector>

@interface Renderer : NSObject

+ (instancetype)rendererWithView:(MTKView *)view error:(NSString **)error;

// The store to draw. Pass nullptr to clear. The reader must outlive the call.
- (void)setStore:(const store::Reader *)reader;

// Voxels from the visibility pass, already expressed against the same origin as
// the store. They are StorePoints, so they go through the same vertex shader as
// the cloud — a voxel and a point are both a coloured position, and giving them
// separate pipelines would only mean two things to keep in step.
- (void)setVoxels:(const std::vector<lod::StorePoint> &)voxels;

// Setup positions in the store's local frame, drawn as markers. Available from
// headers alone, so these are shown while the store is still being built.
- (void)setSetupMarkers:(const std::vector<simd_float3> &)markers;

- (void)drawInView:(MTKView *)view
            camera:(const viewer::OrbitCamera &)camera
              tree:(const lod::Tree &)tree
         selection:(const lod::Selection &)selection;

@property (nonatomic) float pointSize;
// Voxels are drawn a little larger than cloud points: at the same size a 5 cm
// lattice reads as a haze rather than as a surface.
@property (nonatomic) float voxelPointScale;
@property (nonatomic) BOOL  showPoints;
@property (nonatomic) BOOL  showVoxels;
@property (nonatomic) BOOL  showSetups;
@property (nonatomic) BOOL  showCrosshair;
@property (nonatomic) BOOL  showPivot;
- (void)setPivot:(m3::Vec3)pivot;

// True when the whole store is addressed by one zero-copy buffer.
@property (nonatomic, readonly) BOOL zeroCopy;
// Bytes currently held in per-node buffers; zero on the zero-copy path.
@property (nonatomic, readonly) uint64_t cachedBytes;
@property (nonatomic, readonly) size_t   voxelCount;

@end
