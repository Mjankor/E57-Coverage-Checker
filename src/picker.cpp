#include "picker.h"

#include <algorithm>
#include <cmath>

namespace viewer {

using namespace m3;

void setSceneOrigin(std::vector<PointCloud>& clouds) {
    if (clouds.empty()) return;
    const double ox = clouds[0].originX, oy = clouds[0].originY, oz = clouds[0].originZ;
    for (auto& c : clouds) {
        c.sceneOffset[0] = float(c.originX - ox);
        c.sceneOffset[1] = float(c.originY - oy);
        c.sceneOffset[2] = float(c.originZ - oz);
    }
}

bool sceneBounds(const std::vector<PointCloud>& clouds, Vec3& lo, Vec3& hi) {
    bool any = false;
    float l[3] = {0, 0, 0}, h[3] = {0, 0, 0};
    for (const auto& c : clouds) {
        if (c.pointCount() == 0) continue;
        for (int i = 0; i < 3; ++i) {
            const float cl = c.loMin[i] + c.sceneOffset[i];
            const float ch = c.hiMax[i] + c.sceneOffset[i];
            if (!any) { l[i] = cl; h[i] = ch; }
            else      { l[i] = std::min(l[i], cl); h[i] = std::max(h[i], ch); }
        }
        any = true;
    }
    if (!any) return false;
    lo = Vec3{l[0], l[1], l[2]};
    hi = Vec3{h[0], h[1], h[2]};
    return true;
}

PickResult pickNearest(const std::vector<PointCloud>& clouds,
                       const OrbitCamera& cam,
                       float ndcX, float ndcY, float radiusPx) {
    PickResult best;
    const Mat4 vp = cam.viewProjection();

    // Convert the pixel radius into an NDC radius once, rather than converting
    // every candidate back into pixels.
    const float rx = 2.0f * radiusPx / float(cam.viewportWidth());
    const float ry = 2.0f * radiusPx / float(cam.viewportHeight());

    float bestDepth = 1e30f;
    for (size_t ci = 0; ci < clouds.size(); ++ci) {
        const PointCloud& c = clouds[ci];
        const size_t n = c.pointCount();
        for (size_t i = 0; i < n; ++i) {
            const float x = c.xyz[i * 3 + 0] + c.sceneOffset[0];
            const float y = c.xyz[i * 3 + 1] + c.sceneOffset[1];
            const float z = c.xyz[i * 3 + 2] + c.sceneOffset[2];
            const Vec4 clip = vp * Vec4{x, y, z, 1.0f};
            if (clip.w <= 1e-6f) continue;             // behind the camera
            const float nx = clip.x / clip.w;
            const float ny = clip.y / clip.w;
            const float dx = (nx - ndcX) / rx;
            const float dy = (ny - ndcY) / ry;
            if (dx * dx + dy * dy > 1.0f) continue;    // outside the pick disc
            if (clip.w < bestDepth) {
                bestDepth       = clip.w;
                best.hit        = true;
                best.world      = Vec3{x, y, z};
                best.viewDepth  = clip.w;
                best.cloudIndex = ci;
                best.pointIndex = i;
            }
        }
    }
    return best;
}

} // namespace viewer
