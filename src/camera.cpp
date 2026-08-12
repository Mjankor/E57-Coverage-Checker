#include "camera.h"

#include <algorithm>

namespace viewer {

using namespace m3;

// Keeping the camera off the poles avoids the degenerate case where the view
// direction is parallel to world up and the right vector collapses.
static constexpr float kPitchLimit = 1.5533431f;   // 89 degrees

void OrbitCamera::setViewport(int widthPx, int heightPx) {
    vpW_ = std::max(1, widthPx);
    vpH_ = std::max(1, heightPx);
}

void OrbitCamera::frameBounds(Vec3 lo, Vec3 hi) {
    pivot_ = (lo + hi) * 0.5f;
    const Vec3  ext    = hi - lo;
    const float radius = std::max(0.5f * length(ext), 1e-3f);
    // Pull back until the bounding sphere fits the narrower of the two FOVs.
    const float aspect = float(vpW_) / float(vpH_);
    const float fovX   = 2.0f * std::atan(std::tan(fovY_ * 0.5f) * aspect);
    const float fov    = std::min(fovY_, fovX);
    distance_ = radius / std::sin(std::max(fov * 0.5f, 1e-3f)) * 1.15f;

    sceneRadius_ = radius;
    minDistance_ = radius * 1.0e-3f;
    maxDistance_ = radius * 500.0f;
}

float OrbitCamera::nearPlane() const {
    // A fixed fraction of the viewing distance: close enough to never clip what
    // you are looking at, far enough to keep the far/near ratio ~1e3.
    return std::max(distance_ * 0.005f, 1e-4f);
}

float OrbitCamera::farPlane() const {
    // Far enough to contain the whole scene from here, whatever the zoom.
    return std::max(distance_ + sceneRadius_ * 4.0f, distance_ * 2.0f);
}

Vec3 OrbitCamera::forward() const {
    // Direction the camera looks: from eye toward pivot.
    const float cp = std::cos(pitch_), sp = std::sin(pitch_);
    return normalize(Vec3{-cp * std::cos(yaw_), -cp * std::sin(yaw_), -sp});
}

Vec3 OrbitCamera::eye() const {
    const float cp = std::cos(pitch_), sp = std::sin(pitch_);
    const Vec3  dir{cp * std::cos(yaw_), cp * std::sin(yaw_), sp};
    return pivot_ + dir * distance_;
}

Vec3 OrbitCamera::right() const {
    return normalize(cross(forward(), Vec3{0, 0, 1}));
}

Vec3 OrbitCamera::up() const {
    return cross(right(), forward());
}

Mat4 OrbitCamera::view() const {
    return lookAt(eye(), pivot_, Vec3{0, 0, 1});
}

Mat4 OrbitCamera::projection() const {
    return perspective(fovY_, float(vpW_) / float(vpH_), nearPlane(), farPlane());
}

void OrbitCamera::orbit(float dxPx, float dyUpPx) {
    // A full window width is a half turn: fast enough to spin round, slow
    // enough to place a view precisely.
    const float perPixel = 3.14159265f / float(vpW_);
    yaw_  -= dxPx * perPixel;
    pitch_ += dyUpPx * perPixel;
    pitch_ = std::clamp(pitch_, -kPitchLimit, kPitchLimit);
}

void OrbitCamera::pan(float dxPx, float dyUpPx) {
    // Scale so a drag tracks the scene at the pivot's depth: one pixel of drag
    // moves the pivot by one pixel's worth of world distance there.
    const float worldPerPixel =
        2.0f * distance_ * std::tan(fovY_ * 0.5f) / float(vpH_);
    pivot_ = pivot_ - right() * (dxPx * worldPerPixel)
                    - up()    * (dyUpPx * worldPerPixel);
}

void OrbitCamera::zoom(float wheelTicks) {
    // Multiplicative, so zooming feels the same at every scale.
    distance_ *= std::exp(-wheelTicks * 0.15f);
    distance_  = std::clamp(distance_, minDistance_, maxDistance_);
}

void OrbitCamera::setPivotKeepingEye(Vec3 newPivot) {
    const Vec3  e = eye();
    const Vec3  v = e - newPivot;
    const float d = length(v);
    if (d < 1e-6f) return;   // pivot on top of the camera: leave things alone

    pivot_    = newPivot;
    distance_ = std::clamp(d, minDistance_, maxDistance_);
    yaw_      = std::atan2(v.y, v.x);
    pitch_    = std::clamp(std::asin(std::clamp(v.z / d, -1.0f, 1.0f)),
                           -kPitchLimit, kPitchLimit);
}

void OrbitCamera::pixelToNdc(float px, float pyUp, float& ndcX, float& ndcY) const {
    ndcX = 2.0f * (px / float(vpW_)) - 1.0f;
    ndcY = 2.0f * (pyUp / float(vpH_)) - 1.0f;
}

bool OrbitCamera::unproject(float ndcX, float ndcY, float depth, Vec3& outWorld) const {
    // Depth at (or beyond) the far plane means the depth buffer was never
    // written there — no geometry under that pixel.
    if (!(depth < 0.999999f)) return false;

    Mat4 inv;
    if (!invert(viewProjection(), inv)) return false;

    const Vec4 clip{ndcX, ndcY, depth, 1.0f};
    const Vec4 w = inv * clip;
    if (std::fabs(w.w) < 1e-20f) return false;
    outWorld = Vec3{w.x / w.w, w.y / w.w, w.z / w.w};
    return true;
}

} // namespace viewer
