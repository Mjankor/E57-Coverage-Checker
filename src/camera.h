// Orbit camera for the point cloud viewer.
//
// Z-up, because that is what terrestrial survey data is: an E57 from a
// tripod-mounted scanner has Z as the vertical axis, and a Y-up camera makes
// every building look like it fell over.
//
// Kept free of Metal and AppKit so the matrix and pivot arithmetic — the part
// that is actually easy to get subtly wrong — can be unit tested off the
// target platform. See tests/test_camera.cpp.

#pragma once

#include "math3d.h"

namespace viewer {

class OrbitCamera {
public:
    void setViewport(int widthPx, int heightPx);

    // Frames an axis-aligned box: pivot at its centre, pulled back far enough
    // to contain it.
    void frameBounds(m3::Vec3 lo, m3::Vec3 hi);

    // Drag deltas in pixels. dyUp is positive upward (AppKit's convention),
    // not downward — mixing this up inverts the vertical drag.
    void orbit(float dxPx, float dyUpPx);
    void pan  (float dxPx, float dyUpPx);
    void zoom (float wheelTicks);

    // Moves the orbit centre without moving the camera. This is what makes
    // pick-to-pivot feel right: the view must not jump when the pivot changes,
    // so distance/yaw/pitch are re-derived from the unchanged eye position.
    void setPivotKeepingEye(m3::Vec3 newPivot);

    m3::Vec3 eye() const;
    m3::Vec3 forward() const;
    m3::Vec3 right() const;
    m3::Vec3 up() const;

    m3::Mat4 view() const;
    m3::Mat4 projection() const;
    m3::Mat4 viewProjection() const { return projection() * view(); }

    // Depth is Metal clip z in [0, 1]. Returns false if the matrix is
    // singular or the sample is on the far plane (nothing was drawn there).
    bool unproject(float ndcX, float ndcY, float depth, m3::Vec3& outWorld) const;

    // Pixel coordinates, origin bottom-left, to normalised device coordinates.
    void pixelToNdc(float px, float pyUp, float& ndcX, float& ndcY) const;

    int      viewportWidth()  const { return vpW_; }
    int      viewportHeight() const { return vpH_; }

    m3::Vec3 pivot() const { return pivot_; }
    float    distance() const { return distance_; }

    // Clip planes track the current distance rather than being fixed at load.
    // A near plane pinned at some tiny fraction of the scene gives a far/near
    // ratio in the hundreds of thousands, which leaves the depth buffer with
    // almost no usable precision — and pick-to-pivot reads that depth buffer,
    // so it degrades into picking a point metres away from the one under the
    // cursor. Keeping the ratio near 1e3 is what makes the pick accurate.
    float nearPlane() const;
    float farPlane()  const;

private:
    m3::Vec3 pivot_{0, 0, 0};
    float    distance_ = 10.0f;
    float    yaw_      = 0.7853982f;   // 45 degrees, so the first view is not axis-aligned
    float    pitch_    = 0.4363323f;   // 25 degrees above horizontal
    float    fovY_     = 1.0471976f;   // 60 degrees
    float    sceneRadius_ = 10.0f;
    int      vpW_ = 1, vpH_ = 1;
    float    minDistance_ = 0.01f;
    float    maxDistance_ = 1.0e5f;
};

} // namespace viewer
