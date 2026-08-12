// MTKView subclass owning the camera, the scene, and all navigation input.
//
// Bindings, as requested:
//   left drag    pan
//   right drag   orbit
//   right click  set the orbit centre to the point under the crosshair
//   wheel        zoom
//
// Control-left is accepted as an alias for right, because on a trackpad
// "right drag" means holding a two-finger click through a drag, which is
// awkward. Pinch zooms too.

#pragma once

#import <MetalKit/MetalKit.h>

#include "../src/camera.h"
#include "../src/point_cloud.h"

#include <vector>

@protocol CloudViewDelegate <NSObject>
- (void)cloudViewDidChangeView:(NSString *)status;
@end

@interface CloudView : MTKView

@property (nonatomic, weak) id<CloudViewDelegate> cloudDelegate;

// Returns NO and fills `error` if Metal could not start.
- (BOOL)setupRendererReturningError:(NSString **)error;

// Replaces the scene and frames it.
- (void)setScene:(std::vector<viewer::PointCloud>)clouds;
- (void)frameAll;
- (BOOL)hasScene;

@property (nonatomic) float pointSize;

@end
