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

- (void)frameAll;
- (BOOL)hasStore;

@property (nonatomic) float  pointSize;
@property (nonatomic) size_t pointBudget;

@end
