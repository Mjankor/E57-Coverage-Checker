// Metal point-cloud renderer.
//
// Shaders are compiled from a source string at runtime rather than built from a
// .metal file into default.metallib. That follows the house pattern from
// CartesianCapture (ICPGpu.swift, PoissonGpu.swift) and, more practically here,
// removes a dependency on Xcode's .metal build rule firing correctly in a
// hand-maintained project — something that cannot be verified without Xcode.
// The cost is two tiny shaders compiled at launch.
//
// Every failure path leaves the renderer unusable rather than half-initialised,
// and `lastError` says why, so the window can show a message instead of a blank
// view.

#pragma once

#import <MetalKit/MetalKit.h>

#include "../src/camera.h"
#include "../src/point_cloud.h"

#include <string>
#include <vector>

@interface Renderer : NSObject

// Returns nil if Metal is unavailable or the shaders fail to compile.
+ (instancetype)rendererWithView:(MTKView *)view error:(NSString **)error;

// Replaces the GPU-resident scene. Safe to call with an empty vector.
- (void)setClouds:(const std::vector<viewer::PointCloud> &)clouds;

- (void)drawInView:(MTKView *)view camera:(const viewer::OrbitCamera &)camera;

// Point size in pixels; distance attenuation is applied on top of this.
@property (nonatomic) float pointSize;
// Draws a marker at each scan's setup position.
@property (nonatomic) BOOL showSetups;
// Centre crosshair, marking where a right-click will pick the orbit centre.
@property (nonatomic) BOOL showCrosshair;
// World-space orbit centre, drawn as a marker so the pivot is visible.
@property (nonatomic) BOOL showPivot;
- (void)setPivot:(m3::Vec3)pivot;

@end
