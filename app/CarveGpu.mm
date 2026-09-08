#import "CarveGpu.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Buffer layouts
//
// Shared between this file and the shader source below, so the two definitions
// have to agree field for field. Scalars only — no vectors, no arrays of
// vectors — because MSL pads and aligns those in ways that are easy to get
// subtly wrong across the boundary, and a silently misaligned uniform would show
// up as geometry in the wrong place rather than as an error.

namespace {

struct GpuSetup {
    // World-to-scanner rotation, row-major, already folded with the tile origin:
    // q = R * localPoint + tOffset.
    float R0x, R0y, R0z;
    float R1x, R1y, R1z;
    float R2x, R2y, R2z;
    float tOffX, tOffY, tOffZ;
    // Setup position in the tile's local frame.
    float originX, originY, originZ;
    float az0, dAzPerCol, el0, dElPerRow;
    float maxRange, surfaceMargin;
    uint32_t rows, cols;
    uint32_t mapValid;
    uint32_t earlyOut;      // 0 none, 1 saturated, 2 any evidence
};

struct GpuTile {
    float    voxelSize;
    uint32_t dim;
    uint32_t domainKind;    // 0 unbounded, 1 box
    uint32_t pad;
    float    domLoX, domLoY, domLoZ;
    float    domHiX, domHiY, domHiZ;
};

// Uploaded range image: one uint per cell, range in centimetres in the low 16
// bits and status in the next 8. Four bytes rather than the CPU's three because
// a device buffer of three-byte records cannot be indexed without unaligned
// loads, and on the GPU that costs more than the extra 13 MB.
struct ImageBuffer {
    id<MTLBuffer> buffer;
    uint64_t      bytes = 0;
    uint64_t      lastUse = 0;
};

// How much of the GPU's memory to spend holding range images. 4 GB is generous
// on a 64 GB machine and leaves the point store, the octree and the tile buffers
// room; past it the least recently used image is dropped and re-uploaded when
// it is next needed.
constexpr uint64_t kImageBudget = 4ull * 1024 * 1024 * 1024;

// Above this a single tile will not fit in a sensible buffer and the tile goes
// to the CPU. 256^3 with an apron is 17 M voxels, one byte each.
constexpr uint32_t kMaxTileDim = 320;

} // namespace

// ---------------------------------------------------------------------------
// Shader source
//
// Compiled from a string at runtime, following the house pattern: no .metal
// build rule, so the Xcode project needs no special phase and the source is
// visible next to the code that dispatches it.
//
// The structure deliberately mirrors carve::evidenceAt line for line rather
// than being rewritten in a GPU idiom. It is going to be diffed against that
// function by eye for as long as both exist, and a clever reformulation would
// make every future comparison harder for a few per cent.

static NSString *const kCarveShader = @R"METAL(
#include <metal_stdlib>
using namespace metal;

constant float kPi  = 3.14159265358979323846f;
constant float kTau = 6.28318530717958648f;

constant uchar kVisible   = 1;
constant uchar kOccupied  = 2;
constant uchar kReachable = 4;

struct GpuSetup {
    float R0x, R0y, R0z;
    float R1x, R1y, R1z;
    float R2x, R2y, R2z;
    float tOffX, tOffY, tOffZ;
    float originX, originY, originZ;
    float az0, dAzPerCol, el0, dElPerRow;
    float maxRange, surfaceMargin;
    uint  rows, cols;
    uint  mapValid;
    uint  earlyOut;
};

struct GpuTile {
    float voxelSize;
    uint  dim;
    uint  domainKind;
    uint  pad;
    float domLoX, domLoY, domLoZ;
    float domHiX, domHiY, domHiZ;
};

// Whether a voxel has learned enough to stop. Mirrors carve::EarlyOut.
static bool settled(uchar bits, uint mode) {
    uchar evidence = kVisible | kOccupied;
    if (mode == 1u) return (bits & evidence) == evidence;
    if (mode == 2u) return (bits & evidence) != 0;
    return false;
}

kernel void carveVoxels(device uchar             *state   [[buffer(0)]],
                        constant GpuSetup        &s       [[buffer(1)]],
                        constant GpuTile         &t       [[buffer(2)]],
                        device const uint        *cells   [[buffer(3)]],
                        device atomic_uint       *tests   [[buffer(4)]],
                        uint3                     gid     [[thread_position_in_grid]],
                        uint                      lane    [[thread_index_in_simdgroup]])
{
    // No early return before the simdgroup reduction at the end: simd_sum has to
    // be reached by every lane of the group or the count is undefined. So the
    // whole body is written as flags rather than as returns.
    bool inGrid = gid.x < t.dim && gid.y < t.dim && gid.z < t.dim;
    uint idx = inGrid ? ((gid.z * t.dim + gid.y) * t.dim + gid.x) : 0u;
    uchar cur = inGrid ? state[idx] : uchar(0);

    bool active = inGrid && !settled(cur, s.earlyOut);

    // The voxel centre, in the tile's local frame. Everything here is tens of
    // metres at most, which is what keeps float honest.
    float3 p = (float3(float(gid.x), float(gid.y), float(gid.z)) + 0.5f) * t.voxelSize;

    if (active && t.domainKind == 1u) {
        if (p.x < t.domLoX || p.x > t.domHiX ||
            p.y < t.domLoY || p.y > t.domHiY ||
            p.z < t.domLoZ || p.z > t.domHiZ) active = false;
    }

    // Reachability, measured from the setup in the world frame — the same
    // separate computation the CPU does, not reused from the scanner-frame range.
    float3 d = p - float3(s.originX, s.originY, s.originZ);
    if (active && dot(d, d) > s.maxRange * s.maxRange) active = false;

    uchar bits = active ? kReachable : uchar(0);

    if (active) {
        // Scanner frame. The rotation is pre-folded with the tile origin.
        float3 q = float3(s.R0x * p.x + s.R0y * p.y + s.R0z * p.z + s.tOffX,
                          s.R1x * p.x + s.R1y * p.y + s.R1z * p.z + s.tOffY,
                          s.R2x * p.x + s.R2y * p.y + s.R2z * p.z + s.tOffZ);
        float r = sqrt(dot(q, q));

        if (r <= s.maxRange && r >= 1e-9f && s.mapValid != 0u &&
            abs(s.dElPerRow) > 1e-12f && abs(s.dAzPerCol) > 1e-12f) {
            float az = atan2(q.y, q.x);
            if (az < 0.0f) az += kTau;
            float el = asin(clamp(q.z / r, -1.0f, 1.0f));

            int ri = int(rint((el - s.el0) / s.dElPerRow));
            if (ri >= 0 && ri < int(s.rows)) {
                // The same bounded unwrap the CPU does. Bounded rather than a
                // while loop: az is in [0, 2pi) and az0 in (-pi, pi], so the
                // difference needs at most two steps, and a NaN must not hang a
                // GPU thread.
                float dd = az - s.az0;
                for (int i = 0; i < 4 && dd > kPi; ++i)  dd -= kTau;
                for (int i = 0; i < 4 && dd <= -kPi; ++i) dd += kTau;

                int ci = int(rint(dd / s.dAzPerCol)) % int(s.cols);
                if (ci < 0) ci += int(s.cols);

                uint  cell    = cells[uint(ri) * s.cols + uint(ci)];
                float surface = float(cell & 0xFFFFu) * 0.01f;
                uint  status  = (cell >> 16) & 0xFFu;

                if (status == 0u) {            // Hit
                    if (r < surface - s.surfaceMargin)       bits |= kVisible;
                    else if (r <= surface + s.surfaceMargin) bits |= kOccupied;
                } else if (status == 1u) {     // NoReturn
                    if (r <= min(surface, s.maxRange)) bits |= kVisible;
                }
                // OutsideFov contributes nothing, which is the point of it.
            }
        }
    }

    if (inGrid && bits != 0u) state[idx] = cur | bits;

    // Work counter. Per-simdgroup sum, then one atomic from lane zero — the
    // house pattern, and the reason there is no atomic in the result path.
    uint n = simd_sum(active ? 1u : 0u);
    if (lane == 0u && n != 0u)
        atomic_fetch_add_explicit(tests, n, memory_order_relaxed);
}
)METAL";

// ---------------------------------------------------------------------------

@interface CarveGpu ()
- (instancetype)initOrNil;
- (BOOL)carveTile:(const carve::TileKey &)key
           setups:(const std::vector<carve::SetupView> &)setups
           params:(const carve::Params &)p
              out:(carve::Tile &)out
            stats:(carve::Stats &)stats;
@end

@implementation CarveGpu {
    id<MTLDevice>              _device;
    id<MTLCommandQueue>        _queue;
    id<MTLComputePipelineState> _pipeline;

    std::unordered_map<const void *, ImageBuffer> _images;
    uint64_t                   _residentBytes;
    uint64_t                   _clock;

    id<MTLBuffer>              _stateBuffer;
    id<MTLBuffer>              _testsBuffer;
    uint64_t                   _stateCapacity;
}

static NSString *g_unavailable = @"not initialised";

+ (instancetype)shared {
    static CarveGpu *instance = nil;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        instance = [[CarveGpu alloc] initOrNil];
    });
    return instance;
}

+ (NSString *)unavailableReason {
    return [CarveGpu shared] ? @"" : g_unavailable;
}

- (instancetype)initOrNil {
    self = [super init];
    if (!self) return nil;

    _device = MTLCreateSystemDefaultDevice();
    if (!_device) { g_unavailable = @"no Metal device"; return nil; }

    _queue = [_device newCommandQueue];
    if (!_queue) { g_unavailable = @"no Metal command queue"; return nil; }

    NSError *err = nil;
    id<MTLLibrary> lib = [_device newLibraryWithSource:kCarveShader options:nil error:&err];
    if (!lib) {
        g_unavailable = [NSString stringWithFormat:@"carve shader would not compile: %@",
                         err.localizedDescription];
        return nil;
    }
    id<MTLFunction> fn = [lib newFunctionWithName:@"carveVoxels"];
    if (!fn) { g_unavailable = @"carveVoxels missing from the compiled library"; return nil; }

    _pipeline = [_device newComputePipelineStateWithFunction:fn error:&err];
    if (!_pipeline) {
        g_unavailable = [NSString stringWithFormat:@"carve pipeline failed: %@",
                         err.localizedDescription];
        return nil;
    }
    g_unavailable = @"";
    return self;
}

- (uint64_t)residentBytes { return _residentBytes; }

// --- range image upload, with a byte-budgeted LRU ---------------------------

- (id<MTLBuffer>)bufferForImage:(const rimg::RangeImage *)image {
    auto it = _images.find(image);
    if (it != _images.end()) {
        it->second.lastUse = ++_clock;
        return it->second.buffer;
    }

    const size_t count = image->cells.size();
    if (count == 0) return nil;
    const uint64_t bytes = uint64_t(count) * 4;
    if (bytes > _device.maxBufferLength) return nil;

    // Evict until it fits. Dropping an image is free — it is a copy of data the
    // CPU still holds, and re-uploading it is the only cost.
    while (_residentBytes + bytes > kImageBudget && !_images.empty()) {
        auto oldest = _images.begin();
        for (auto i = _images.begin(); i != _images.end(); ++i)
            if (i->second.lastUse < oldest->second.lastUse) oldest = i;
        _residentBytes -= oldest->second.bytes;
        _images.erase(oldest);
    }

    std::vector<uint32_t> packed(count);
    for (size_t i = 0; i < count; ++i)
        packed[i] = uint32_t(image->cells[i].rangeCm) | (uint32_t(image->cells[i].status) << 16);

    id<MTLBuffer> buf = [_device newBufferWithBytes:packed.data()
                                             length:size_t(bytes)
                                            options:MTLResourceStorageModeShared];
    if (!buf) return nil;

    ImageBuffer rec;
    rec.buffer  = buf;
    rec.bytes   = bytes;
    rec.lastUse = ++_clock;
    _images[image] = rec;
    _residentBytes += bytes;
    return buf;
}

// --- the carve --------------------------------------------------------------

- (BOOL)carveTile:(const carve::TileKey &)key
           setups:(const std::vector<carve::SetupView> &)setups
           params:(const carve::Params &)p
              out:(carve::Tile &)out
            stats:(carve::Stats &)stats {
    const uint32_t core = p.tileVoxels;
    const uint32_t dim  = core + 2 * p.apron;
    if (dim == 0 || dim > kMaxTileDim) return NO;

    // Tile geometry, in double, exactly as the CPU computes it.
    const double T = p.tileMetres();
    double lo[3] = {double(key.x) * T, double(key.y) * T, double(key.z) * T};
    const double pad = double(p.apron) * p.voxelSize;
    out.key = key; out.dim = dim; out.core = core; out.apron = p.apron;
    for (int k = 0; k < 3; ++k) out.origin[k] = lo[k] - pad;

    const size_t voxels = size_t(dim) * dim * dim;
    out.state.assign(voxels, 0);

    // Which setups reach the tile including its apron. Same test as the CPU's,
    // so the two carve identical voxel sets.
    double plo[3], phi[3];
    for (int k = 0; k < 3; ++k) {
        plo[k] = out.origin[k];
        phi[k] = out.origin[k] + double(dim) * p.voxelSize;
    }
    std::vector<size_t> reach;
    const double R2 = p.maxRange * p.maxRange;
    for (size_t i = 0; i < setups.size(); ++i) {
        if (!setups[i].image) continue;
        double d2 = 0;
        for (int k = 0; k < 3; ++k) {
            const double v = (setups[i].origin[k] < plo[k]) ? (plo[k] - setups[i].origin[k])
                           : (setups[i].origin[k] > phi[k]) ? (setups[i].origin[k] - phi[k])
                                                            : 0.0;
            d2 += v * v;
        }
        if (d2 <= R2) reach.push_back(i);
    }
    if (reach.empty()) { carve::tallyTile(out, stats); return YES; }

    // Nearest first, matching the CPU so an AnyEvidence run records the same
    // one of several true answers.
    if (p.earlyOut != carve::EarlyOut::None && reach.size() > 1) {
        double c[3];
        for (int k = 0; k < 3; ++k) c[k] = out.origin[k] + 0.5 * double(dim) * p.voxelSize;
        std::sort(reach.begin(), reach.end(), [&](size_t a, size_t b) {
            double da = 0, db = 0;
            for (int k = 0; k < 3; ++k) {
                const double u = setups[a].origin[k] - c[k];
                const double v = setups[b].origin[k] - c[k];
                da += u * u; db += v * v;
            }
            if (da != db) return da < db;
            return a < b;
        });
    }

    if (_stateCapacity < voxels || !_stateBuffer) {
        _stateBuffer = [_device newBufferWithLength:voxels options:MTLResourceStorageModeShared];
        if (!_stateBuffer) { _stateCapacity = 0; return NO; }
        _stateCapacity = voxels;
    }
    std::memset(_stateBuffer.contents, 0, voxels);

    if (!_testsBuffer) {
        _testsBuffer = [_device newBufferWithLength:sizeof(uint32_t)
                                            options:MTLResourceStorageModeShared];
        if (!_testsBuffer) return NO;
    }
    *static_cast<uint32_t *>(_testsBuffer.contents) = 0;

    GpuTile gt{};
    gt.voxelSize  = float(p.voxelSize);
    gt.dim        = dim;
    gt.domainKind = (p.domain.kind == carve::Domain::Kind::Box) ? 1u : 0u;
    if (gt.domainKind) {
        // The domain box in the tile's local frame, differenced in double before
        // the cast so the subtraction happens at full precision.
        gt.domLoX = float(p.domain.lo[0] - out.origin[0]);
        gt.domLoY = float(p.domain.lo[1] - out.origin[1]);
        gt.domLoZ = float(p.domain.lo[2] - out.origin[2]);
        gt.domHiX = float(p.domain.hi[0] - out.origin[0]);
        gt.domHiY = float(p.domain.hi[1] - out.origin[1]);
        gt.domHiZ = float(p.domain.hi[2] - out.origin[2]);
    }

    id<MTLCommandBuffer> cb = [_queue commandBuffer];
    if (!cb) return NO;

    // Uniform threadgroups over a rounded-up grid, not dispatchThreads. The
    // kernel's simd_sum has to be reached by every lane of a simdgroup, and a
    // non-uniform dispatch leaves the trailing group partly inactive — which is
    // exactly why the kernel is written with flags instead of early returns, and
    // why the out-of-grid threads have to actually run.
    const NSUInteger w = std::max<NSUInteger>(1, _pipeline.threadExecutionWidth);
    NSUInteger gy = 4;
    while (w * gy > _pipeline.maxTotalThreadsPerThreadgroup && gy > 1) gy /= 2;
    const MTLSize group = MTLSizeMake(w, gy, 1);
    const MTLSize grid  = MTLSizeMake((dim + w - 1) / w,
                                      (dim + gy - 1) / gy,
                                      dim);

    for (size_t si : reach) {
        const carve::SetupView &s = setups[si];
        id<MTLBuffer> cells = [self bufferForImage:s.image];
        if (!cells) { return NO; }         // nothing has been committed yet

        const viewer::Rigid &R = s.worldToScanner;
        GpuSetup gs{};
        gs.R0x = float(R.R[0]); gs.R0y = float(R.R[1]); gs.R0z = float(R.R[2]);
        gs.R1x = float(R.R[3]); gs.R1y = float(R.R[4]); gs.R1z = float(R.R[5]);
        gs.R2x = float(R.R[6]); gs.R2y = float(R.R[7]); gs.R2z = float(R.R[8]);
        // R * tileOrigin + t, in double, then cast: the tile origin can be at
        // UTM magnitudes and this is the one product that must not be formed in
        // float.
        for (int r = 0; r < 3; ++r) {
            const double v = R.R[3 * r + 0] * out.origin[0] +
                             R.R[3 * r + 1] * out.origin[1] +
                             R.R[3 * r + 2] * out.origin[2] + R.t[r];
            (&gs.tOffX)[r] = float(v);
        }
        gs.originX = float(s.origin[0] - out.origin[0]);
        gs.originY = float(s.origin[1] - out.origin[1]);
        gs.originZ = float(s.origin[2] - out.origin[2]);

        const rimg::RangeImage &im = *s.image;
        gs.az0           = float(im.map.az0);
        gs.dAzPerCol     = float(im.map.dAzPerCol);
        gs.el0           = float(im.map.el0);
        gs.dElPerRow     = float(im.map.dElPerRow);
        gs.maxRange      = float(p.maxRange);
        gs.surfaceMargin = float(p.surfaceMargin);
        gs.rows          = im.rows;
        gs.cols          = im.cols;
        gs.mapValid      = im.map.valid ? 1u : 0u;
        gs.earlyOut      = (p.earlyOut == carve::EarlyOut::Saturated)   ? 1u
                         : (p.earlyOut == carve::EarlyOut::AnyEvidence) ? 2u : 0u;

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!enc) return NO;
        [enc setComputePipelineState:_pipeline];
        [enc setBuffer:_stateBuffer offset:0 atIndex:0];
        [enc setBytes:&gs length:sizeof(gs) atIndex:1];
        [enc setBytes:&gt length:sizeof(gt) atIndex:2];
        [enc setBuffer:cells offset:0 atIndex:3];
        [enc setBuffer:_testsBuffer offset:0 atIndex:4];
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:group];
        [enc endEncoding];
    }

    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) return NO;

    std::memcpy(out.state.data(), _stateBuffer.contents, voxels);
    stats.setupTests += *static_cast<const uint32_t *>(_testsBuffer.contents);
    carve::tallyTile(out, stats);
    ++_tilesCarved;
    return YES;
}

+ (carve::TileCarver)carver {
    return [](const carve::TileKey &key, const std::vector<carve::SetupView> &setups,
              const carve::Params &p, carve::Tile &out, carve::Stats &stats,
              void *user) -> bool {
        CarveGpu *me = (__bridge CarveGpu *)user;
        if (!me) return false;
        @autoreleasepool {
            const BOOL ok = [me carveTile:key setups:setups params:p out:out stats:stats];
            if (!ok) ++me->_tilesRefused;
            return ok;
        }
    };
}

@end
