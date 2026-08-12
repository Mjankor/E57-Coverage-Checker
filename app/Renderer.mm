#import "Renderer.h"

#include <algorithm>

// ---------------------------------------------------------------------------
// Shader source
//
// Points are drawn as round sprites: a square point sprite reads as a blocky
// mess at the densities a terrestrial scan produces. Size attenuates with
// distance so a wall does not turn into a solid sheet when you zoom in, but is
// clamped so distant structure stays visible rather than vanishing.

static NSString *const kShaderSource = @R"METAL(
#include <metal_stdlib>
using namespace metal;

struct Uniforms {
    float4x4 viewProj;
    float3   modelOffset;
    float    pointSize;
    float4   tint;
    float    attenuationScale;
    uint     useVertexColour;
};

struct VOut {
    float4 position [[position]];
    float  pointSize [[point_size]];
    half4  colour;
};

vertex VOut pointVS(uint vid [[vertex_id]],
                    const device packed_float3 *positions [[buffer(0)]],
                    const device uchar4        *colours   [[buffer(1)]],
                    constant Uniforms          &u         [[buffer(2)]])
{
    float3 p = float3(positions[vid]) + u.modelOffset;
    VOut o;
    o.position = u.viewProj * float4(p, 1.0);

    // Clip w is the eye-space distance; sizing by 1/w keeps a point's on-screen
    // footprint roughly constant in world terms.
    float w = max(o.position.w, 1e-4);
    o.pointSize = clamp(u.pointSize * u.attenuationScale / w, 1.0, 24.0);

    if (u.useVertexColour != 0u) {
        uchar4 c = colours[vid];
        o.colour = half4(half3(float3(c.r, c.g, c.b) / 255.0), 1.0h);
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

// Overlay: positions supplied directly in clip space, no depth test.
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
    simd_float3   modelOffset;
    float         pointSize;
    simd_float4   tint;
    float         attenuationScale;
    uint32_t      useVertexColour;
};

struct GpuCloud {
    id<MTLBuffer> positions;
    id<MTLBuffer> colours;      // nil when the scan carried none
    NSUInteger    count = 0;
    simd_float3   offset = {0, 0, 0};
    simd_float4   tint = {1, 1, 1, 1};
    simd_float3   setup = {0, 0, 0};
    bool          hasSetup = false;
};

// Distinct hues per scan so overlapping setups can be told apart when a file
// carries no colour of its own.
simd_float4 tintForIndex(size_t i) {
    static const simd_float4 palette[] = {
        {0.55f, 0.78f, 1.00f, 1.0f}, {1.00f, 0.78f, 0.45f, 1.0f},
        {0.62f, 0.90f, 0.62f, 1.0f}, {0.94f, 0.66f, 0.86f, 1.0f},
        {0.98f, 0.92f, 0.55f, 1.0f}, {0.70f, 0.70f, 0.95f, 1.0f},
        {0.55f, 0.92f, 0.90f, 1.0f}, {0.92f, 0.60f, 0.55f, 1.0f},
    };
    return palette[i % (sizeof(palette) / sizeof(palette[0]))];
}

simd_float4x4 toSimd(const m3::Mat4 &m) {
    simd_float4x4 r;
    for (int c = 0; c < 4; ++c)
        r.columns[c] = simd_make_float4(m.at(c, 0), m.at(c, 1), m.at(c, 2), m.at(c, 3));
    return r;
}

} // namespace

@implementation Renderer {
    id<MTLDevice>              _device;
    id<MTLCommandQueue>        _queue;
    id<MTLRenderPipelineState> _pointPipeline;
    id<MTLRenderPipelineState> _overlayPipeline;
    id<MTLDepthStencilState>   _depthState;
    std::vector<GpuCloud>      _clouds;
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

    view.device                    = device;
    view.colorPixelFormat          = MTLPixelFormatBGRA8Unorm;
    view.depthStencilPixelFormat   = MTLPixelFormatDepth32Float;
    view.clearColor                = MTLClearColorMake(0.09, 0.10, 0.12, 1.0);
    view.sampleCount               = 1;

    NSError *nsErr = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:kShaderSource options:nil error:&nsErr];
    if (!lib) {
        if (error) *error = [NSString stringWithFormat:@"Shader compilation failed: %@",
                             nsErr.localizedDescription];
        return nil;
    }

    MTLRenderPipelineDescriptor *pd = [[MTLRenderPipelineDescriptor alloc] init];
    pd.vertexFunction                  = [lib newFunctionWithName:@"pointVS"];
    pd.fragmentFunction                = [lib newFunctionWithName:@"pointFS"];
    pd.colorAttachments[0].pixelFormat = view.colorPixelFormat;
    pd.depthAttachmentPixelFormat      = view.depthStencilPixelFormat;
    r->_pointPipeline = [device newRenderPipelineStateWithDescriptor:pd error:&nsErr];
    if (!r->_pointPipeline) {
        if (error) *error = [NSString stringWithFormat:@"Point pipeline failed: %@",
                             nsErr.localizedDescription];
        return nil;
    }

    MTLRenderPipelineDescriptor *od = [[MTLRenderPipelineDescriptor alloc] init];
    od.vertexFunction                  = [lib newFunctionWithName:@"overlayVS"];
    od.fragmentFunction                = [lib newFunctionWithName:@"overlayFS"];
    od.colorAttachments[0].pixelFormat = view.colorPixelFormat;
    od.depthAttachmentPixelFormat      = view.depthStencilPixelFormat;
    r->_overlayPipeline = [device newRenderPipelineStateWithDescriptor:od error:&nsErr];
    if (!r->_overlayPipeline) {
        if (error) *error = [NSString stringWithFormat:@"Overlay pipeline failed: %@",
                             nsErr.localizedDescription];
        return nil;
    }

    MTLDepthStencilDescriptor *dd = [[MTLDepthStencilDescriptor alloc] init];
    dd.depthCompareFunction = MTLCompareFunctionLess;
    dd.depthWriteEnabled    = YES;
    r->_depthState = [device newDepthStencilStateWithDescriptor:dd];

    r->_pointSize     = 2.5f;
    r->_showSetups    = YES;
    r->_showCrosshair = YES;
    r->_showPivot     = YES;
    return r;
}

- (void)setPivot:(m3::Vec3)pivot { _pivot = pivot; }

- (void)setClouds:(const std::vector<viewer::PointCloud> &)clouds {
    _clouds.clear();
    _clouds.reserve(clouds.size());

    for (size_t i = 0; i < clouds.size(); ++i) {
        const viewer::PointCloud &c = clouds[i];
        if (c.pointCount() == 0) continue;

        GpuCloud g;
        g.count = c.pointCount();
        // Shared storage: on Apple silicon the GPU reads the same pages the
        // loader wrote, so there is no upload step.
        g.positions = [_device newBufferWithBytes:c.xyz.data()
                                           length:c.xyz.size() * sizeof(float)
                                          options:MTLResourceStorageModeShared];
        if (!c.rgb.empty() && c.rgb.size() == g.count) {
            g.colours = [_device newBufferWithBytes:c.rgb.data()
                                             length:c.rgb.size() * sizeof(viewer::Rgb8)
                                            options:MTLResourceStorageModeShared];
        }
        if (!g.positions) continue;   // allocation failed: skip rather than crash

        g.offset   = simd_make_float3(c.sceneOffset[0], c.sceneOffset[1], c.sceneOffset[2]);
        g.tint     = tintForIndex(i);
        g.setup    = simd_make_float3(c.originOffset[0] + c.sceneOffset[0],
                                      c.originOffset[1] + c.sceneOffset[1],
                                      c.originOffset[2] + c.sceneOffset[2]);
        g.hasSetup = c.hasSetupPosition;
        _clouds.push_back(g);
    }
}

- (void)drawInView:(MTKView *)view camera:(const viewer::OrbitCamera &)camera {
    MTLRenderPassDescriptor *rp = view.currentRenderPassDescriptor;
    id<CAMetalDrawable> drawable = view.currentDrawable;
    if (!rp || !drawable) return;

    id<MTLCommandBuffer> cb = [_queue commandBuffer];
    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
    [enc setDepthStencilState:_depthState];
    [enc setRenderPipelineState:_pointPipeline];

    const simd_float4x4 vp = toSimd(camera.viewProjection());
    // Attenuation reference: at the pivot distance a point renders at exactly
    // `pointSize`, so the control means what it says wherever you are.
    const float atten = std::max(camera.distance(), 1e-3f);

    for (const GpuCloud &g : _clouds) {
        Uniforms u{};
        u.viewProj         = vp;
        u.modelOffset      = g.offset;
        u.pointSize        = _pointSize;
        u.tint             = g.tint;
        u.attenuationScale = atten;
        u.useVertexColour  = g.colours ? 1u : 0u;

        [enc setVertexBuffer:g.positions offset:0 atIndex:0];
        if (g.colours) [enc setVertexBuffer:g.colours offset:0 atIndex:1];
        else           [enc setVertexBuffer:g.positions offset:0 atIndex:1];  // unread
        [enc setVertexBytes:&u length:sizeof(u) atIndex:2];
        [enc drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:g.count];
    }

    if (_showSetups) {
        for (const GpuCloud &g : _clouds) {
            if (!g.hasSetup) continue;
            Uniforms u{};
            u.viewProj         = vp;
            u.modelOffset      = simd_make_float3(0, 0, 0);
            u.pointSize        = 40.0f;
            u.tint             = simd_make_float4(1.0f, 0.35f, 0.2f, 1.0f);
            u.attenuationScale = atten;
            u.useVertexColour  = 0u;
            const simd_float3 p = g.setup;
            [enc setVertexBytes:&p length:sizeof(p) atIndex:0];
            [enc setVertexBytes:&p length:sizeof(p) atIndex:1];
            [enc setVertexBytes:&u length:sizeof(u) atIndex:2];
            [enc drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:1];
        }
    }

    if (_showPivot) {
        Uniforms u{};
        u.viewProj         = vp;
        u.modelOffset      = simd_make_float3(0, 0, 0);
        u.pointSize        = 26.0f;
        u.tint             = simd_make_float4(1.0f, 0.95f, 0.3f, 1.0f);
        u.attenuationScale = atten;
        u.useVertexColour  = 0u;
        const simd_float3 p = simd_make_float3(_pivot.x, _pivot.y, _pivot.z);
        [enc setVertexBytes:&p length:sizeof(p) atIndex:0];
        [enc setVertexBytes:&p length:sizeof(p) atIndex:1];
        [enc setVertexBytes:&u length:sizeof(u) atIndex:2];
        [enc drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:1];
    }

    if (_showCrosshair) {
        // Clip-space cross, sized in NDC against the drawable's aspect so it
        // stays square.
        const float aspect = (float)view.drawableSize.width /
                             std::max(1.0f, (float)view.drawableSize.height);
        const float h = 0.018f, v = h * aspect;
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
