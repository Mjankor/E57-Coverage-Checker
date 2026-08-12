#import "CloudView.h"
#import "Renderer.h"

#include "../src/picker.h"

#include <algorithm>
#include <cmath>

@implementation CloudView {
    Renderer                        *_renderer;
    viewer::OrbitCamera              _camera;
    std::vector<viewer::PointCloud>  _clouds;
    NSPoint                          _lastPoint;
    BOOL                             _dragging;
    BOOL                             _orbiting;
}

- (BOOL)setupRendererReturningError:(NSString **)error {
    _renderer = [Renderer rendererWithView:self error:error];
    if (!_renderer) return NO;
    _renderer.pointSize = 2.5f;
    self.pointSize = 2.5f;
    self.enableSetNeedsDisplay = YES;   // redraw on interaction, not at 60 Hz
    self.paused = YES;
    self.preferredFramesPerSecond = 60;
    [self updateViewport];
    return YES;
}

- (BOOL)acceptsFirstResponder { return YES; }
- (BOOL)hasScene { return !_clouds.empty(); }

- (void)setPointSize:(float)pointSize {
    _pointSize = pointSize;
    _renderer.pointSize = pointSize;
    [self setNeedsDisplay:YES];
}

- (void)updateViewport {
    const CGSize s = self.drawableSize;
    _camera.setViewport((int)std::max<CGFloat>(1, s.width), (int)std::max<CGFloat>(1, s.height));
}

- (void)setScene:(std::vector<viewer::PointCloud>)clouds {
    _clouds = std::move(clouds);
    viewer::setSceneOrigin(_clouds);
    [_renderer setClouds:_clouds];
    [self frameAll];
}

- (void)frameAll {
    [self updateViewport];
    m3::Vec3 lo, hi;
    if (viewer::sceneBounds(_clouds, lo, hi)) _camera.frameBounds(lo, hi);
    [self viewChanged];
}

- (void)viewChanged {
    [_renderer setPivot:_camera.pivot()];
    [self setNeedsDisplay:YES];
    if ([self.cloudDelegate respondsToSelector:@selector(cloudViewDidChangeView:)]) {
        const m3::Vec3 p = _camera.pivot();
        [self.cloudDelegate cloudViewDidChangeView:
            [NSString stringWithFormat:@"pivot  %.3f, %.3f, %.3f     eye distance  %.2f m",
                                       p.x, p.y, p.z, _camera.distance()]];
    }
}

// --- input ----------------------------------------------------------------

// View coordinates, origin bottom-left and +y up. NSEvent's own deltaY has a
// contested sign convention across event types, so deltas are derived from
// tracked positions instead.
- (NSPoint)viewPoint:(NSEvent *)e {
    return [self convertPoint:e.locationInWindow fromView:nil];
}

- (void)mouseDown:(NSEvent *)event {
    _lastPoint = [self viewPoint:event];
    _dragging  = YES;
    // Control-click is the trackpad's right-click: treat it as orbit.
    _orbiting  = (event.modifierFlags & NSEventModifierFlagControl) != 0;
}

- (void)mouseDragged:(NSEvent *)event {
    if (!_dragging) return;
    const NSPoint p = [self viewPoint:event];
    const float dx = (float)(p.x - _lastPoint.x) * (float)self.window.backingScaleFactor;
    const float dy = (float)(p.y - _lastPoint.y) * (float)self.window.backingScaleFactor;
    _lastPoint = p;
    if (_orbiting) _camera.orbit(dx, dy);
    else           _camera.pan(dx, dy);
    [self viewChanged];
}

- (void)mouseUp:(NSEvent *)event { (void)event; _dragging = NO; _orbiting = NO; }

- (void)rightMouseDown:(NSEvent *)event {
    _lastPoint = [self viewPoint:event];
    _dragging  = YES;
    _orbiting  = YES;
    [self pickPivotAtCentre];
}

- (void)rightMouseDragged:(NSEvent *)event { [self mouseDragged:event]; }
- (void)rightMouseUp:(NSEvent *)event { (void)event; _dragging = NO; _orbiting = NO; }

- (void)scrollWheel:(NSEvent *)event {
    // Precise deltas are pixels from a trackpad; coarse ones are wheel notches.
    const float raw = (float)event.scrollingDeltaY;
    const float ticks = event.hasPreciseScrollingDeltas ? raw * 0.1f : raw;
    if (std::fabs(ticks) < 1e-6f) return;
    _camera.zoom(ticks);
    [self viewChanged];
}

- (void)magnifyWithEvent:(NSEvent *)event {
    _camera.zoom((float)event.magnification * 10.0f);
    [self viewChanged];
}

- (void)keyDown:(NSEvent *)event {
    NSString *chars = event.charactersIgnoringModifiers;
    if (chars.length == 0) { [super keyDown:event]; return; }
    switch ([chars characterAtIndex:0]) {
    case 'f': case 'F': [self frameAll]; break;
    case '[': self.pointSize = std::max(1.0f, self.pointSize - 0.5f); break;
    case ']': self.pointSize = std::min(12.0f, self.pointSize + 0.5f); break;
    default: [super keyDown:event]; break;
    }
}

// Sets the orbit centre to the nearest point under the crosshair. The camera
// does not move — only what it turns about — so the view never jumps.
- (void)pickPivotAtCentre {
    if (_clouds.empty()) return;
    [self updateViewport];

    // A generous radius: at a distance the crosshair often falls between
    // points rather than on one, and failing to pick feels broken.
    const viewer::PickResult r = viewer::pickNearest(_clouds, _camera, 0.0f, 0.0f, 24.0f);
    if (!r.hit) {
        if ([self.cloudDelegate respondsToSelector:@selector(cloudViewDidChangeView:)])
            [self.cloudDelegate cloudViewDidChangeView:
                @"no point under the crosshair — orbit centre unchanged"];
        return;
    }
    _camera.setPivotKeepingEye(r.world);
    [self viewChanged];
}

- (void)drawRect:(NSRect)rect {
    (void)rect;
    [self updateViewport];
    [_renderer drawInView:self camera:_camera];
}

@end
