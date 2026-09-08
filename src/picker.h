// Screen-space point picking, used to choose the orbit centre.
//
// Deliberately CPU-side rather than a depth-buffer readback. A readback needs
// an extra render target or a depth blit, both with per-GPU restrictions, and
// none of it can be tested in this environment. Projecting the decimated points
// directly costs a few tens of milliseconds on a right-click — unnoticeable for
// a one-shot action — and the result is verifiable. See tests/test_viewer.cpp.
//
// Clouds each carry their own local origin (see point_cloud.h), so a scene
// origin is chosen once and every cloud records its offset from it. Picking and
// rendering both work in that shared frame.

#pragma once

#include "camera.h"
#include "lod.h"
#include "point_cloud.h"
#include "point_store.h"

#include <vector>

namespace viewer {

// Assigns a common scene origin and fills each cloud's sceneOffset. Uses the
// first cloud's origin, so offsets stay small for a site of any coordinate
// system — the same float32 precision argument as point_cloud.h.
void setSceneOrigin(std::vector<PointCloud>& clouds);

// Combined bounds of every cloud in the scene frame.
bool sceneBounds(const std::vector<PointCloud>& clouds, m3::Vec3& lo, m3::Vec3& hi);

struct PickResult {
    bool     hit = false;
    m3::Vec3 world;              // scene frame
    float    viewDepth = 0;      // distance along the view axis
    size_t   cloudIndex = 0;
    size_t   pointIndex = 0;
};

// Nearest point to (ndcX, ndcY) within radiusPx, measured in screen pixels.
// "Nearest" means nearest to the camera, not nearest to the cursor: when the
// crosshair sits on a near wall with a far one visible past it, the near one is
// what you want to turn about.
PickResult pickNearest(const std::vector<PointCloud>& clouds,
                       const OrbitCamera& cam,
                       float ndcX, float ndcY, float radiusPx);

// The same pick against an mmap'd store, restricted to the nodes currently
// selected for drawing. Bounded by the point budget rather than by the store's
// size, so it costs the same on a 5-setup store and a 5000-setup one — and it
// picks only from what is actually on screen, which is what the crosshair
// means. `pointIndex` is the index within `cloudIndex`, which here is a node.
PickResult pickNearestInStore(const store::Reader& reader, const lod::Tree& tree,
                              const lod::Selection& selection, const OrbitCamera& cam,
                              float ndcX, float ndcY, float radiusPx);

} // namespace viewer
