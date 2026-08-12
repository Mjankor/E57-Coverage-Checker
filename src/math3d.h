// Minimal 3D math. Header-only, no dependencies (see CLAUDE.md).
//
// Matrices are column-major float[16], m[col * 4 + row] — the layout Metal's
// float4x4 expects, so a Mat4 can be handed to a shader without transposing.
// The projection maps eye-space depth to clip z in [0, 1], which is Metal's
// convention (not OpenGL's [-1, 1]); getting this wrong yields a picture that
// looks plausible until you try to unproject a depth sample.

#pragma once

#include <cmath>
#include <cstring>

namespace m3 {

struct Vec3 {
    float x = 0, y = 0, z = 0;
    Vec3() = default;
    Vec3(float X, float Y, float Z) : x(X), y(Y), z(Z) {}
};

inline Vec3  operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3  operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3  operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline Vec3  operator-(Vec3 a) { return {-a.x, -a.y, -a.z}; }
inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3  cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline float length(Vec3 a) { return std::sqrt(dot(a, a)); }
inline Vec3  normalize(Vec3 a) {
    float L = length(a);
    return L > 1e-20f ? a * (1.0f / L) : Vec3{0, 0, 0};
}

struct Vec4 { float x = 0, y = 0, z = 0, w = 0; };

struct Mat4 {
    float m[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    float&       at(int col, int row)       { return m[col * 4 + row]; }
    const float& at(int col, int row) const { return m[col * 4 + row]; }
};

inline Mat4 identity() { return Mat4{}; }

inline Mat4 operator*(const Mat4& a, const Mat4& b) {
    Mat4 r;
    for (int c = 0; c < 4; ++c)
        for (int row = 0; row < 4; ++row) {
            float s = 0;
            for (int k = 0; k < 4; ++k) s += a.at(k, row) * b.at(c, k);
            r.at(c, row) = s;
        }
    return r;
}

inline Vec4 operator*(const Mat4& a, const Vec4& v) {
    Vec4 r;
    r.x = a.at(0,0)*v.x + a.at(1,0)*v.y + a.at(2,0)*v.z + a.at(3,0)*v.w;
    r.y = a.at(0,1)*v.x + a.at(1,1)*v.y + a.at(2,1)*v.z + a.at(3,1)*v.w;
    r.z = a.at(0,2)*v.x + a.at(1,2)*v.y + a.at(2,2)*v.z + a.at(3,2)*v.w;
    r.w = a.at(0,3)*v.x + a.at(1,3)*v.y + a.at(2,3)*v.z + a.at(3,3)*v.w;
    return r;
}

// Right-handed look-at: the camera looks down -Z in eye space.
inline Mat4 lookAt(Vec3 eye, Vec3 center, Vec3 up) {
    Vec3 f = normalize(center - eye);
    Vec3 s = normalize(cross(f, up));
    Vec3 u = cross(s, f);
    Mat4 r;
    r.at(0,0) = s.x; r.at(1,0) = s.y; r.at(2,0) = s.z; r.at(3,0) = -dot(s, eye);
    r.at(0,1) = u.x; r.at(1,1) = u.y; r.at(2,1) = u.z; r.at(3,1) = -dot(u, eye);
    r.at(0,2) = -f.x; r.at(1,2) = -f.y; r.at(2,2) = -f.z; r.at(3,2) = dot(f, eye);
    r.at(0,3) = 0; r.at(1,3) = 0; r.at(2,3) = 0; r.at(3,3) = 1;
    return r;
}

// Metal-convention perspective: clip z in [0, 1], near plane -> 0.
inline Mat4 perspective(float fovYRadians, float aspect, float zNear, float zFar) {
    const float ys = 1.0f / std::tan(fovYRadians * 0.5f);
    const float xs = ys / aspect;
    const float zs = zFar / (zNear - zFar);
    Mat4 r;
    std::memset(r.m, 0, sizeof(r.m));
    r.at(0,0) = xs;
    r.at(1,1) = ys;
    r.at(2,2) = zs;
    r.at(2,3) = -1.0f;
    r.at(3,2) = zNear * zs;
    return r;
}

inline bool invert(const Mat4& in, Mat4& out) {
    const float* a = in.m;
    float inv[16];

    inv[0]  =  a[5]*a[10]*a[15] - a[5]*a[11]*a[14] - a[9]*a[6]*a[15] +
               a[9]*a[7]*a[14] + a[13]*a[6]*a[11] - a[13]*a[7]*a[10];
    inv[4]  = -a[4]*a[10]*a[15] + a[4]*a[11]*a[14] + a[8]*a[6]*a[15] -
               a[8]*a[7]*a[14] - a[12]*a[6]*a[11] + a[12]*a[7]*a[10];
    inv[8]  =  a[4]*a[9]*a[15] - a[4]*a[11]*a[13] - a[8]*a[5]*a[15] +
               a[8]*a[7]*a[13] + a[12]*a[5]*a[11] - a[12]*a[7]*a[9];
    inv[12] = -a[4]*a[9]*a[14] + a[4]*a[10]*a[13] + a[8]*a[5]*a[14] -
               a[8]*a[6]*a[13] - a[12]*a[5]*a[10] + a[12]*a[6]*a[9];
    inv[1]  = -a[1]*a[10]*a[15] + a[1]*a[11]*a[14] + a[9]*a[2]*a[15] -
               a[9]*a[3]*a[14] - a[13]*a[2]*a[11] + a[13]*a[3]*a[10];
    inv[5]  =  a[0]*a[10]*a[15] - a[0]*a[11]*a[14] - a[8]*a[2]*a[15] +
               a[8]*a[3]*a[14] + a[12]*a[2]*a[11] - a[12]*a[3]*a[10];
    inv[9]  = -a[0]*a[9]*a[15] + a[0]*a[11]*a[13] + a[8]*a[1]*a[15] -
               a[8]*a[3]*a[13] - a[12]*a[1]*a[11] + a[12]*a[3]*a[9];
    inv[13] =  a[0]*a[9]*a[14] - a[0]*a[10]*a[13] - a[8]*a[1]*a[14] +
               a[8]*a[2]*a[13] + a[12]*a[1]*a[10] - a[12]*a[2]*a[9];
    inv[2]  =  a[1]*a[6]*a[15] - a[1]*a[7]*a[14] - a[5]*a[2]*a[15] +
               a[5]*a[3]*a[14] + a[13]*a[2]*a[7] - a[13]*a[3]*a[6];
    inv[6]  = -a[0]*a[6]*a[15] + a[0]*a[7]*a[14] + a[4]*a[2]*a[15] -
               a[4]*a[3]*a[14] - a[12]*a[2]*a[7] + a[12]*a[3]*a[6];
    inv[10] =  a[0]*a[5]*a[15] - a[0]*a[7]*a[13] - a[4]*a[1]*a[15] +
               a[4]*a[3]*a[13] + a[12]*a[1]*a[7] - a[12]*a[3]*a[5];
    inv[14] = -a[0]*a[5]*a[14] + a[0]*a[6]*a[13] + a[4]*a[1]*a[14] -
               a[4]*a[2]*a[13] - a[12]*a[1]*a[6] + a[12]*a[2]*a[5];
    inv[3]  = -a[1]*a[6]*a[11] + a[1]*a[7]*a[10] + a[5]*a[2]*a[11] -
               a[5]*a[3]*a[10] - a[9]*a[2]*a[7] + a[9]*a[3]*a[6];
    inv[7]  =  a[0]*a[6]*a[11] - a[0]*a[7]*a[10] - a[4]*a[2]*a[11] +
               a[4]*a[3]*a[10] + a[8]*a[2]*a[7] - a[8]*a[3]*a[6];
    inv[11] = -a[0]*a[5]*a[11] + a[0]*a[7]*a[9] + a[4]*a[1]*a[11] -
               a[4]*a[3]*a[9] - a[8]*a[1]*a[7] + a[8]*a[3]*a[5];
    inv[15] =  a[0]*a[5]*a[10] - a[0]*a[6]*a[9] - a[4]*a[1]*a[10] +
               a[4]*a[2]*a[9] + a[8]*a[1]*a[6] - a[8]*a[2]*a[5];

    float det = a[0]*inv[0] + a[1]*inv[4] + a[2]*inv[8] + a[3]*inv[12];
    if (std::fabs(det) < 1e-30f) return false;
    det = 1.0f / det;
    for (int i = 0; i < 16; ++i) out.m[i] = inv[i] * det;
    return true;
}

} // namespace m3
