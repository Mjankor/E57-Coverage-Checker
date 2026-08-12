// Decimated point cloud for display.
//
// Two things this has to get right:
//
// Precision. A georeferenced scan's coordinates are UTM/MGA eastings and
// northings around 1e5-1e6, where float32's 24-bit mantissa resolves to about
// 0.0625 m — every point snapped to a 6 cm grid. (CartesianCapture's
// E57Writer.swift documents exactly this bug, found by round-tripping a real
// export.) So positions are accumulated in double, a per-cloud origin is
// subtracted, and only the *offsets* are stored as float32. The renderer works
// in that local frame; the origin is carried alongside in double.
//
// Budget. A structured scan is 10-30 M points and a session may hold dozens.
// Everything is decimated to a point budget on load — this is a QA viewer, not
// a presentation renderer, and a uniform stride preserves the spatial
// distribution well enough to judge coverage.

#pragma once

#include "e57.h"

#include <cstdint>
#include <string>
#include <vector>

namespace viewer {

struct Rgb8 { uint8_t r = 255, g = 255, b = 255, a = 255; };

struct PointCloud {
    std::string scanName;
    size_t      scanIndex = 0;

    // World origin of the local frame. Add this to a position to get the
    // scan's own coordinates back.
    double originX = 0, originY = 0, originZ = 0;

    std::vector<float> xyz;      // 3 floats per point, relative to origin
    std::vector<Rgb8>  rgb;      // empty when the file carried no colour

    // Offset from the shared scene origin to this cloud's local origin.
    // Assigned by setSceneOrigin() when the scene is assembled; the renderer
    // applies it as a translation and the picker adds it before projecting.
    float sceneOffset[3] = {0, 0, 0};

    // Scanner setup position, in the same local frame. The visibility method
    // needs this; the viewer draws it so a bad pose is obvious on sight.
    float  originOffset[3] = {0, 0, 0};
    bool   hasSetupPosition = false;

    uint64_t sourcePointCount = 0;   // before decimation
    float    loMin[3] = {0, 0, 0};
    float    hiMax[3] = {0, 0, 0};

    size_t pointCount() const { return xyz.size() / 3; }
};

struct LoadOptions {
    // Points kept per scan. 4 M is a few hundred MB of vertex data across a
    // handful of scans and still resolves centimetre detail on a room.
    size_t maxPoints = 4'000'000;
    // Colour source preference; falls back automatically when absent.
    bool preferColour    = true;
    bool preferIntensity = true;
};

// Decodes and decimates one scan. Returns false and sets `err` on failure.
bool loadCloud(e57::Reader& reader, size_t scanIndex, const LoadOptions& opt,
               PointCloud& out, std::string& err);

} // namespace viewer
