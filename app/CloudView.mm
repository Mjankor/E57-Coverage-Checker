#import "CloudView.h"
#import "Renderer.h"

#include "../src/picker.h"

#include <algorithm>
#include <cmath>

@implementation CloudView {
    Renderer                *_renderer;
    viewer::OrbitCamera      _camera;

    store::Reader            _store;
    lod::Tree                _tree;
    lod::Selection           _selection;
    BOOL                     _selectionStale;

    // Setup positions in the file frame, kept in double so they can be
    // re-expressed when the store's origin arrives.
    std::vector<double>      _setupsFileFrame;
    double                   _origin[3];
    lod::Aabb                _setupBounds;
    BOOL                     _haveSetupBounds;

    NSPoint                  _lastPoint;
    BOOL                     _dragging;
    BOOL                     _orbiting;
}

- (BOOL)setupRendererReturningError:(NSString **)error {
    _renderer = [Renderer rendererWithView:self error:error];
    if (!_renderer) return NO;
    self.pointSize   = 2.5f;
    self.pointBudget = 12000000;
    self.enableSetNeedsDisplay = YES;   // redraw on interaction, not at 60 Hz
    self.paused = YES;
    _origin[0] = _origin[1] = _origin[2] = 0.0;
    [self updateViewport];
    return YES;
}

- (BOOL)acceptsFirstResponder { return YES; }
- (BOOL)hasStore { return _store.isOpen(); }

- (void)setPointSize:(float)pointSize {
    _pointSize = pointSize;
    _renderer.pointSize = pointSize;
    [self setNeedsDisplay:YES];
}

- (void)setPointBudget:(size_t)budget {
    _pointBudget = budget;
    _selectionStale = YES;
    [self setNeedsDisplay:YES];
}

- (void)updateViewport {
    const CGSize s = self.drawableSize;
    _camera.setViewport((int)std::max<CGFloat>(1, s.width), (int)std::max<CGFloat>(1, s.height));
}

// --- content --------------------------------------------------------------

- (void)rebuildSetupMarkers {
    std::vector<simd_float3> markers;
    markers.reserve(_setupsFileFrame.size() / 3);
    _setupBounds = lod::Aabb{};
    _haveSetupBounds = NO;
    for (size_t i = 0; i + 2 < _setupsFileFrame.size(); i += 3) {
        const simd_float3 m = simd_make_float3(float(_setupsFileFrame[i + 0] - _origin[0]),
                                               float(_setupsFileFrame[i + 1] - _origin[1]),
                                               float(_setupsFileFrame[i + 2] - _origin[2]));
        markers.push_back(m);
        if (!_haveSetupBounds) {
            for (int k = 0; k < 3; ++k) { _setupBounds.lo[k] = m[k]; _setupBounds.hi[k] = m[k]; }
            _haveSetupBounds = YES;
        } else {
            _setupBounds.expand(m.x, m.y, m.z);
        }
    }
    [_renderer setSetupMarkers:markers];
}

- (void)setSetups:(const std::vector<double> &)fileFrameXYZ {
    _setupsFileFrame = fileFrameXYZ;
    if (!_store.isOpen() && !_setupsFileFrame.empty()) {
        // No store yet, so the setups define the frame. Centring on them keeps
        // float offsets small even at UTM magnitudes.
        double lo[3], hi[3];
        for (int k = 0; k < 3; ++k) { lo[k] = hi[k] = _setupsFileFrame[k]; }
        for (size_t i = 0; i + 2 < _setupsFileFrame.size(); i += 3)
            for (int k = 0; k < 3; ++k) {
                lo[k] = std::min(lo[k], _setupsFileFrame[i + k]);
                hi[k] = std::max(hi[k], _setupsFileFrame[i + k]);
            }
        for (int k = 0; k < 3; ++k) _origin[k] = 0.5 * (lo[k] + hi[k]);
    }
    [self rebuildSetupMarkers];
    [self frameAll];
}

- (BOOL)openStore:(NSString *)path error:(NSString **)error {
    std::string err;
    if (!_store.open(path.UTF8String, err)) {
        if (error) *error = [NSString stringWithUTF8String:err.c_str()];
        return NO;
    }
    _tree = _store.tree();
    for (int k = 0; k < 3; ++k) _origin[k] = _store.header().origin[k];
    // Setups were placed against a provisional origin; re-express them against
    // the store's so markers and points stay registered.
    [self rebuildSetupMarkers];
    [_renderer setStore:&_store];
    _selectionStale = YES;
    [self frameAll];
    return YES;
}

- (void)closeAll {
    [_renderer setStore:nullptr];
    _store.close();
    _tree = lod::Tree{};
    _selection = lod::Selection{};
    _setupsFileFrame.clear();
    _haveSetupBounds = NO;
    [_renderer setSetupMarkers:std::vector<simd_float3>{}];
    [self setNeedsDisplay:YES];
}

- (void)frameAll {
    [self updateViewport];
    if (!_tree.nodes.empty()) {
        const lod::Aabb &b = _tree.bounds;
        _camera.frameBounds({b.lo[0], b.lo[1], b.lo[2]}, {b.hi[0], b.hi[1], b.hi[2]});
    } else if (_haveSetupBounds) {
        // Setups alone are a plane of points; pad so the initial view is not
        // edge-on to a degenerate box.
        lod::Aabb b = _setupBounds;
        const float pad = std::max(5.0f, 0.25f * b.diagonal());
        for (int k = 0; k < 3; ++k) { b.lo[k] -= pad; b.hi[k] += pad; }
        _camera.frameBounds({b.lo[0], b.lo[1], b.lo[2]}, {b.hi[0], b.hi[1], b.hi[2]});
    }
    [self viewChanged];
}

- (void)viewChanged {
    _selectionStale = YES;
    [_renderer setPivot:_camera.pivot()];
    [self setNeedsDisplay:YES];
    [self reportStatus];
}

- (void)reportStatus {
    if (![self.cloudDelegate respondsToSelector:@selector(cloudViewDidChangeView:)]) return;
    const m3::Vec3 p = _camera.pivot();
    NSString *detail = @"";
    if (_store.isOpen()) {
        detail = [NSString stringWithFormat:@"   ·   %.1f M of %.1f M points   ·   %lu nodes%@",
                  double(_selection.points) / 1e6,
                  double(_store.header().totalPoints) / 1e6,
                  (unsigned long)_selection.nodes.size(),
                  _selection.budgetExhausted ? @"   (budget reached)" : @""];
    }
    [self.cloudDelegate cloudViewDidChangeView:
        [NSString stringWithFormat:@"pivot %.2f, %.2f, %.2f   ·   %.1f m out%@",
                                   p.x, p.y, p.z, _camera.distance(), detail]];
}

- (void)updateSelection {
    if (!_selectionStale || _tree.nodes.empty()) return;
    lod::SelectOptions opt;
    opt.pointBudget = _pointBudget;
    const float pixelsPerRadian =
        float(self.drawableSize.height) / std::max(_camera.fovY(), 1e-3f);
    _selection = lod::selectNodes(_tree, _camera.viewProjection(), _camera.eye(),
                                  pixelsPerRadian, opt);
    _selectionStale = NO;
}

// --- input ----------------------------------------------------------------

// View coordinates, origin bottom-left and +y up. NSEvent's own deltaY has a
// contested sign convention across event types, so deltas come from tracked
// positions instead.
- (NSPoint)viewPoint:(NSEvent *)e {
    return [self convertPoint:e.locationInWindow fromView:nil];
}

- (void)mouseDown:(NSEvent *)event {
    _lastPoint = [self viewPoint:event];
    _dragging  = YES;
    _orbiting  = (event.modifierFlags & NSEventModifierFlagControl) != 0;
    if (_orbiting) [self pickPivotAtCentre];
}

- (void)mouseDragged:(NSEvent *)event {
    if (!_dragging) return;
    const NSPoint p = [self viewPoint:event];
    const CGFloat scale = self.window.backingScaleFactor;
    const float dx = (float)((p.x - _lastPoint.x) * scale);
    const float dy = (float)((p.y - _lastPoint.y) * scale);
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
// does not move — only what it turns about — so the view never jumps. The
// search covers only the nodes currently drawn, so its cost is bounded by the
// point budget rather than by the size of the store.
- (void)pickPivotAtCentre {
    if (!_store.isOpen()) return;
    [self updateViewport];
    [self updateSelection];

    // A generous radius: at a distance the crosshair often falls between
    // points rather than on one, and failing to pick feels broken.
    const viewer::PickResult r =
        viewer::pickNearestInStore(_store, _tree, _selection, _camera, 0.0f, 0.0f, 24.0f);
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
    _selectionStale = YES;   // the viewport may have changed under us
    [self updateSelection];
    [_renderer drawInView:self camera:_camera tree:_tree selection:_selection];
}

@end
