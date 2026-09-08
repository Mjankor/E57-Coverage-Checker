#import "Renderer.h"

#include <algorithm>
#include <unordered_map>

// ---------------------------------------------------------------------------
// Shader source
//
// StorePoint is read straight from the mapping, so the MSL struct must match
// src/lod.h byte for byte: packed_float3 (12) + uchar4 (4) + 2 ushorts (4).

static NSString *const kShaderSource = @R"METAL(
#include <metal_stdlib>
using namespace metal;

struct StorePoint {
    packed_float3 pos;
    uchar4        rgba;
    ushort        scanId;
    ushort        pad;
};

struct Uniforms {
    float4x4 viewProj;
    float    pointSize;
    float    attenuationScale;
    float4   tint;
    uint     useVertexColour;
};

struct VOut {
    float4 position  [[position]];
    float  pointSize [[point_size]];
    half4  colour;
};

vertex VOut pointVS(uint vid [[vertex_id]],
                    const device StorePoint *pts [[buffer(0)]],
                    constant Uniforms       &u   [[buffer(1)]])
{
    StorePoint p = pts[vid];
    VOut o;
    o.position = u.viewProj * float4(p.pos, 1.0);

    // Clip w is the eye-space distance; sizing by 1/w keeps a point's on-screen
    // footprint roughly constant in world terms.
    float w = max(o.position.w, 1e-4);
    o.pointSize = clamp(u.pointSize * u.attenuationScale / w, 1.0, 24.0);

    if (u.useVertexColour != 0u) {
        o.colour = half4(half3(float3(p.rgba.r, p.rgba.g, p.rgba.b) / 255.0), 1.0h);
    } else {
        o.colour = half4(u.tint);
    }
    return o;
}

fragment half4 pointFS(VOut in [[stage_in]], float2 pc [[point_coord]])
{
    float2 d = pc - float2(0.5, 0.5);
    if (dot(d, d) > 0.25) discard_fragment();
    return in.colour;
}

// Markers and the pivot: positions supplied inline, one per draw.
struct MarkerOut {
    float4 position  [[position]];
    float  pointSize [[point_size]];
    half4  colour;
};

vertex MarkerOut markerVS(uint vid [[vertex_id]],
                          const device packed_float3 *pos [[buffer(0)]],
                          constant Uniforms          &u   [[buffer(1)]])
{
    MarkerOut o;
    o.position  = u.viewProj * float4(float3(pos[vid]), 1.0);
    o.pointSize = u.pointSize;
    o.colour    = half4(u.tint);
    return o;
}

fragment half4 markerFS(MarkerOut in [[stage_in]], float2 pc [[point_coord]])
{
    float2 d = pc - float2(0.5, 0.5);
    float r2 = dot(d, d);
    if (r2 > 0.25) discard_fragment();
    // A darker rim, so a marker reads against a bright cloud.
    half k = (r2 > 0.16h) ? 0.45h : 1.0h;
    return half4(in.colour.rgb * k, in.colour.a);
}

// Overlay: clip-space positions, no depth test.
struct OverlayOut {
    float4 position [[position]];
    half4  colour;
};

vertex OverlayOut overlayVS(uint vid [[vertex_id]],
                            const device float2 *verts [[buffer(0)]],
                            constant float4     &tint  [[buffer(1)]])
{
    OverlayOut o;
    o.position = float4(verts[vid], 0.0, 1.0);
    o.colour   = half4(tint);
    return o;
}

fragment half4 overlayFS(OverlayOut in [[stage_in]]) { return in.colour; }
)METAL";

namespace {

struct Uniforms {
    simd_float4x4 viewProj;
    float         pointSize;
    float         attenuationScale;
    simd_float4   tint;
    uint32_t      useVertexColour;
};

simd_float4x4 toSimd(const m3::Mat4 &m) {
    simd_float4x4 r;
    for (int c = 0; c < 4; ++c)
        r.columns[c] = simd_make_float4(m.at(c, 0), m.at(c, 1), m.at(c, 2), m.at(c, 3));
    return r;
}

// Bytes held in per-node buffers when the store is too large for one buffer.
// 2 GB is generous on a 64 GB machine and still far below the working-set
// limit, leaving room for the visibility grids later.
constexpr uint64_t kNodeCacheBudget = 2ull * 1024 * 1024 * 1024;

} // namespace

@implementation Renderer {
    id<MTLDevice>              _device;
    id<MTLCommandQueue>        _queue;
    id<MTLRenderPipelineState> _pointPipeline;
    id<MTLRenderPipelineState> _markerPipeline;
    id<MTLRenderPipelineState> _overlayPipeline;
    id<MTLDepthStencilState>   _depthState;

    const store::Reader       *_store;
    id<MTLBuffer>              _wholeStore;      // zero-copy path
    std::unordered_map<uint32_t, id<MTLBuffer>> _nodeCache;   // fallback path
    std::vector<uint32_t>      _cacheOrder;      // least recently used first
    uint64_t                   _cacheBytes;

    std::vector<simd_float3>   _setups;
    id<MTLBuffer>              _setupBuffer;
    id<MTLBuffer>              _voxelBuffer;
    size_t                     _voxelCount;
    m3::Vec3                   _pivot;
}

+ (instancetype)rendererWithView:(MTKView *)view error:(NSString **)error {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        if (error) *error = @"No Metal device available.";
        return nil;
    }

    Renderer *r = [[Renderer alloc] init];
    r->_device = device;
    r->_queue  = [device newCommandQueue];
    if (!r->_queue) {
        if (error) *error = @"Could not create a Metal command queue.";
        return nil;
    }

    view.device                  = device;
    view.colorPixelFormat        = MTLPixelFormatBGRA8Unorm;
    view.depthStencilPixelFormat = MTLPixelFormatDepth32Float;
    view.clearColor              = MTLClearColorMake(0.09, 0.10, 0.12, 1.0);
    view.sampleCount             = 1;

    NSError *nsErr = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:kShaderSource options:nil error:&nsErr];
    if (!lib) {
        if (error) *error = [NSString stringWithFormat:@"Shader compilation failed: %@",
                             nsErr.localizedDescription];
        return nil;
    }

    struct { NSString *vs; NSString *fs; id<MTLRenderPipelineState> __strong *slot; NSString *name; }
    pipes[] = {
        {@"pointVS",   @"pointFS",   &r->_pointPipeline,   @"point"},
        {@"markerVS",  @"markerFS",  &r->_markerPipeline,  @"marker"},
        {@"overlayVS", @"overlayFS", &r->_overlayPipeline, @"overlay"},
    };
    for (auto &p : pipes) {
        MTLRenderPipelineDescriptor *d = [[MTLRenderPipelineDescriptor alloc] init];
        d.vertexFunction                  = [lib newFunctionWithName:p.vs];
        d.fragmentFunction                = [lib newFunctionWithName:p.fs];
        d.colorAttachments[0].pixelFormat = view.colorPixelFormat;
        d.depthAttachmentPixelFormat      = view.depthStencilPixelFormat;
        *p.slot = [device newRenderPipelineStateWithDescriptor:d error:&nsErr];
        if (!*p.slot) {
            if (error) *error = [NSString stringWithFormat:@"%@ pipeline failed: %@",
                                 p.name, nsErr.localizedDescription];
            return nil;
        }
    }

    MTLDepthStencilDescriptor *dd = [[MTLDepthStencilDescriptor alloc] init];
    dd.depthCompareFunction = MTLCompareFunctionLess;
    dd.depthWriteEnabled    = YES;
    r->_depthState = [device newDepthStencilStateWithDescriptor:dd];

    r->_pointSize       = 2.5f;
    r->_voxelPointScale = 1.6f;
    r->_showPoints      = YES;
    r->_showVoxels      = YES;
    r->_showSetups    = YES;
    r->_showCrosshair = YES;
    r->_showPivot     = YES;
    return r;
}

- (void)setPivot:(m3::Vec3)pivot { _pivot = pivot; }
- (BOOL)zeroCopy { return _wholeStore != nil; }
- (uint64_t)cachedBytes { return _cacheBytes; }
- (size_t)voxelCount { return _voxelCount; }

- (void)setVoxels:(const std::vector<lod::StorePoint> &)voxels {
    _voxelBuffer = nil;
    _voxelCount  = 0;
    if (voxels.empty()) return;
    const size_t bytes = voxels.size() * sizeof(lod::StorePoint);
    _voxelBuffer = [_device newBufferWithBytes:voxels.data()
                                        length:bytes
                                       options:MTLResourceStorageModeShared];
    // A failed allocation is not fatal: the count stays zero and nothing is
    // drawn, which is better than a half-populated buffer read as geometry.
    if (_voxelBuffer) _voxelCount = voxels.size();
}

- (void)setStore:(const store::Reader *)reader {
    _store = reader;
    _wholeStore = nil;
    _nodeCache.clear();
    _cacheOrder.clear();
    _cacheBytes = 0;
    if (!reader || !reader->isOpen()) return;

    // mmap always maps whole pages, so rounding the length up stays inside the
    // mapping; newBufferWithBytesNoCopy requires a page multiple.
    const size_t pageSize = size_t(getpagesize());
    const uint64_t rounded = (reader->mappedSize() + pageSize - 1) / pageSize * pageSize;
    if (rounded <= _device.maxBufferLength) {
        _wholeStore = [_device newBufferWithBytesNoCopy:const_cast<void *>(reader->mappedBase())
                                                 length:rounded
                                                options:MTLResourceStorageModeShared
                                            deallocator:nil];
    }
    // A nil result here is not fatal: the per-node cache below handles it, just
    // with a copy per node.
}

- (void)setSetupMarkers:(const std::vector<simd_float3> &)markers {
    _setups = markers;
    _setupBuffer = nil;
    if (_setups.empty()) return;
    _setupBuffer = [_device newBufferWithBytes:_setups.data()
                                        length:_setups.size() * sizeof(simd_float3)
                                       options:MTLResourceStorageModeShared];
}

// Per-node buffer for the fallback path, with least-recently-used eviction.
- (id<MTLBuffer>)bufferForNode:(uint32_t)node {
    auto it = _nodeCache.find(node);
    if (it != _nodeCache.end()) {
        auto pos = std::find(_cacheOrder.begin(), _cacheOrder.end(), node);
        if (pos != _cacheOrder.end()) {
            _cacheOrder.erase(pos);
            _cacheOrder.push_back(node);
        }
        return it->second;
    }

    const uint64_t bytes = _store->payloadBytes(node);
    if (bytes == 0) return nil;

    while (_cacheBytes + bytes > kNodeCacheBudget && !_cacheOrder.empty()) {
        const uint32_t victim = _cacheOrder.front();
        _cacheOrder.erase(_cacheOrder.begin());
        auto v = _nodeCache.find(victim);
        if (v != _nodeCache.end()) {
            _cacheBytes -= _store->payloadBytes(victim);
            _nodeCache.erase(v);
        }
    }

    id<MTLBuffer> buf = [_device newBufferWithBytes:_store->points(node)
                                             length:bytes
                                            options:MTLResourceStorageModeShared];
    if (!buf) return nil;
    _nodeCache[node] = buf;
    _cacheOrder.push_back(node);
    _cacheBytes += bytes;
    return buf;
}

- (void)drawInView:(MTKView *)view
            camera:(const viewer::OrbitCamera &)camera
              tree:(const lod::Tree &)tree
         selection:(const lod::Selection &)selection {
    MTLRenderPassDescriptor *rp = view.currentRenderPassDescriptor;
    id<CAMetalDrawable> drawable = view.currentDrawable;
    if (!rp || !drawable) return;

    id<MTLCommandBuffer> cb = [_queue commandBuffer];
    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
    [enc setDepthStencilState:_depthState];

    const simd_float4x4 vp = toSimd(camera.viewProjection());
    // At the pivot distance a point renders at exactly `pointSize`, so the
    // control means what it says wherever the camera is.
    const float atten = std::max(camera.distance(), 1e-3f);

    if (_showPoints && _store && _store->isOpen() && !selection.nodes.empty()) {
        [enc setRenderPipelineState:_pointPipeline];
        Uniforms u{};
        u.viewProj         = vp;
        u.pointSize        = _pointSize;
        u.attenuationScale = atten;
        u.useVertexColour  = 1u;
        u.tint             = simd_make_float4(1, 1, 1, 1);
        [enc setVertexBytes:&u length:sizeof(u) atIndex:1];

        for (uint32_t node : selection.nodes) {
            if (node >= tree.nodes.size()) continue;
            const uint32_t count = tree.nodes[node].pointCount;
            if (count == 0) continue;

            if (_wholeStore) {
                [enc setVertexBuffer:_wholeStore
                              offset:NSUInteger(_store->payloadOffset(node))
                             atIndex:0];
            } else {
                id<MTLBuffer> b = [self bufferForNode:node];
                if (!b) continue;
                [enc setVertexBuffer:b offset:0 atIndex:0];
            }
            [enc drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:count];
        }
    }

    if (_showVoxels && _voxelBuffer && _voxelCount) {
        // Depth-tested along with the cloud, so a voxel behind a wall is hidden
        // by that wall. Showing the void through the geometry in front of it
        // would make every shadow look like it reached the camera.
        [enc setRenderPipelineState:_pointPipeline];
        Uniforms u{};
        u.viewProj         = vp;
        u.pointSize        = _pointSize * _voxelPointScale;
        u.attenuationScale = atten;
        u.useVertexColour  = 1u;
        u.tint             = simd_make_float4(1, 1, 1, 1);
        [enc setVertexBuffer:_voxelBuffer offset:0 atIndex:0];
        [enc setVertexBytes:&u length:sizeof(u) atIndex:1];
        [enc drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:_voxelCount];
    }

    if (_showSetups && _setupBuffer && !_setups.empty()) {
        [enc setRenderPipelineState:_markerPipeline];
        Uniforms u{};
        u.viewProj         = vp;
        u.pointSize        = 13.0f;
        u.attenuationScale = atten;
        u.tint             = simd_make_float4(1.0f, 0.35f, 0.2f, 1.0f);
        u.useVertexColour  = 0u;
        [enc setVertexBuffer:_setupBuffer offset:0 atIndex:0];
        [enc setVertexBytes:&u length:sizeof(u) atIndex:1];
        [enc drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:_setups.size()];
    }

    if (_showPivot) {
        [enc setRenderPipelineState:_markerPipeline];
        Uniforms u{};
        u.viewProj         = vp;
        u.pointSize        = 16.0f;
        u.attenuationScale = atten;
        u.tint             = simd_make_float4(1.0f, 0.95f, 0.3f, 1.0f);
        u.useVertexColour  = 0u;
        const simd_float3 p = simd_make_float3(_pivot.x, _pivot.y, _pivot.z);
        [enc setVertexBytes:&p length:sizeof(p) atIndex:0];
        [enc setVertexBytes:&u length:sizeof(u) atIndex:1];
        [enc drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:1];
    }

    if (_showCrosshair) {
        const float aspect = (float)view.drawableSize.width /
                             std::max(1.0f, (float)view.drawableSize.height);
        const float h = 0.018f, v = h / std::max(aspect, 1e-3f);
        const simd_float2 verts[8] = {
            {-v, 0}, {-v * 0.35f, 0}, {v * 0.35f, 0}, {v, 0},
            {0, -h}, {0, -h * 0.35f}, {0, h * 0.35f}, {0, h},
        };
        simd_float4 tint = simd_make_float4(1.0f, 1.0f, 1.0f, 0.85f);
        [enc setRenderPipelineState:_overlayPipeline];
        [enc setVertexBytes:verts length:sizeof(verts) atIndex:0];
        [enc setVertexBytes:&tint length:sizeof(tint) atIndex:1];
        [enc drawPrimitives:MTLPrimitiveTypeLine vertexStart:0 vertexCount:8];
    }

    [enc endEncoding];
    [cb presentDrawable:drawable];
    [cb commit];
}

@end
