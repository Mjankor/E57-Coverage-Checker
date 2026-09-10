// MTKView subclass owning the camera, the store, and all navigation input.
//
// Bindings:
//   left drag    pan
//   right drag   orbit
//   right click  set the orbit centre to the point under the crosshair
//   wheel        zoom
//
// Control-left is an alias for right, because holding a two-finger click
// through a drag on a trackpad is awkward. Pinch zooms too.
//
// The view works in two stages, matching how a large corpus opens: setup
// markers first, from headers alone, then the point store when it is ready.

#pragma once

#import <MetalKit/MetalKit.h>

#include "../src/camera.h"
#include "../src/lod.h"
#include "../src/point_store.h"
#include "../src/visibility.h"

#include <vector>

@protocol CloudViewDelegate <NSObject>
- (void)cloudViewDidChangeView:(NSString *)status;
@end

@interface CloudView : MTKView

@property (nonatomic, weak) id<CloudViewDelegate> cloudDelegate;

- (BOOL)setupRendererReturningError:(NSString **)error;

// Stage one: setup positions in the file's coordinate system, flat xyz triples.
// Drawn immediately, before any point has been decoded.
- (void)setSetups:(const std::vector<double> &)fileFrameXYZ;

// Stage two: the point store. Takes over rendering; setups are re-expressed in
// the store's origin so the two stay registered.
- (BOOL)openStore:(NSString *)path error:(NSString **)error;
- (void)closeAll;

// The visibility pass's answer. Held in the result's own frame and re-expressed
// against the store's origin whenever that changes, the same way setup markers
// are — otherwise opening a store after a carve would slide the voxels off the
// geometry they describe.
- (void)setVoxelResult:(const vis::Result &)result;

// Recolours the voxels already held — see vis::recolour. The outward normals
// came back with the result, so this is a pass over the drawn set rather than
// another carve.
- (void)setVoxelShading:(uint8_t)mode;
- (void)clearVoxels;
- (BOOL)hasVoxels;
- (size_t)voxelCount;

// Frames the voxels rather than the whole site, which is usually what you want
// immediately after a run.
- (void)frameVoxels;

- (void)frameAll;
- (BOOL)hasStore;

@property (nonatomic) float  pointSize;
@property (nonatomic) size_t pointBudget;
@property (nonatomic) BOOL   showClouds;
@property (nonatomic) BOOL   showVoxels;

@end
