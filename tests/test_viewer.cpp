// Tests for the parts of the viewer that are not Metal or AppKit: camera
// arithmetic, structured-vs-merged classification, and cloud decimation.
//
// The rendering layer cannot be tested in this environment. Everything that
// can be pulled out of it and checked, is.

#include "e57_fixture.h"
#include "../src/camera.h"
#include "../src/point_cloud.h"
#include "../src/picker.h"
#include "../src/scan_check.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int g_failures = 0;
static int g_checks   = 0;

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        ++g_checks;                                                             \
        if (!(cond)) {                                                          \
            ++g_failures;                                                       \
            std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, (msg));       \
        }                                                                       \
    } while (0)

#define CHECK_NEAR(a, b, tol, msg)                                              \
    do {                                                                        \
        ++g_checks;                                                             \
        double va = (a), vb = (b);                                              \
        if (!(std::fabs(va - vb) <= (tol))) {                                   \
            ++g_failures;                                                       \
            std::printf("  FAIL %s:%d  %s (%.17g vs %.17g)\n",                  \
                        __FILE__, __LINE__, (msg), va, vb);                     \
        }                                                                       \
    } while (0)

static std::string tmpPath(const char* name) {
    const char* dir = std::getenv("E57COV_TMPDIR");
    return std::string(dir ? dir : "/tmp") + "/e57cov_" + name + ".e57";
}

// Deterministic pseudo-random, so a failure is reproducible.
struct Lcg {
    uint64_t s = 0x9E3779B97F4A7C15ull;
    double next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return double((s >> 11) & ((1ull << 53) - 1)) / double(1ull << 53);
    }
};

// ---------------------------------------------------------------------------
// Camera

static void testCameraProjection() {
    std::printf("camera: projection round trip\n");
    viewer::OrbitCamera cam;
    cam.setViewport(1600, 1000);
    cam.frameBounds({-10, -10, 0}, {10, 10, 5});

    // A point inside the scene must project into the view and unproject back.
    const m3::Vec3 p{3.0f, -4.0f, 2.0f};
    const m3::Mat4 vp = cam.viewProjection();
    const m3::Vec4 clip = vp * m3::Vec4{p.x, p.y, p.z, 1.0f};
    CHECK(clip.w > 0.0f, "point is in front of the camera");
    const float ndcX = clip.x / clip.w, ndcY = clip.y / clip.w, d = clip.z / clip.w;
    CHECK(ndcX > -1.0f && ndcX < 1.0f, "projects inside the horizontal frustum");
    CHECK(ndcY > -1.0f && ndcY < 1.0f, "projects inside the vertical frustum");
    CHECK(d > 0.0f && d < 1.0f, "depth lands in Metal's [0,1] clip range");

    m3::Vec3 back;
    CHECK(cam.unproject(ndcX, ndcY, d, back), "unprojects");
    // Tolerance deliberately tight: this is the accuracy pick-to-pivot
    // inherits, and it is only achievable with a sane far/near ratio.
    CHECK_NEAR(back.x, p.x, 1e-3, "unproject x");
    CHECK_NEAR(back.y, p.y, 1e-3, "unproject y");
    CHECK_NEAR(back.z, p.z, 1e-3, "unproject z");

    m3::Vec3 nothing;
    CHECK(!cam.unproject(0, 0, 1.0f, nothing), "far-plane depth means no geometry");
}

static void testCameraNavigation() {
    std::printf("camera: orbit, pan, zoom\n");
    viewer::OrbitCamera cam;
    cam.setViewport(1200, 800);
    cam.frameBounds({-5, -5, -5}, {5, 5, 5});

    const float d0 = cam.distance();
    const m3::Vec3 pivot0 = cam.pivot();

    cam.orbit(150.0f, 60.0f);
    CHECK_NEAR(cam.distance(), d0, 1e-4, "orbit preserves distance to pivot");
    CHECK_NEAR(m3::length(cam.eye() - cam.pivot()), d0, 1e-3, "eye stays on the orbit sphere");
    CHECK_NEAR(m3::length(cam.pivot() - pivot0), 0.0, 1e-6, "orbit does not move the pivot");

    // Pitch must not flip over the pole.
    for (int i = 0; i < 200; ++i) cam.orbit(0.0f, 100.0f);
    CHECK(cam.up().z > 0.0f, "camera stays upright at the pitch limit");

    const float d1 = cam.distance();
    cam.zoom(3.0f);
    CHECK(cam.distance() < d1, "wheel forward zooms in");
    cam.zoom(-3.0f);
    CHECK_NEAR(cam.distance(), d1, 1e-3, "zoom is symmetric");

    // Panning moves the pivot in the view plane, never along the view axis.
    const m3::Vec3 before = cam.pivot();
    const m3::Vec3 fwd    = cam.forward();
    cam.pan(40.0f, -25.0f);
    const m3::Vec3 delta = cam.pivot() - before;
    CHECK(m3::length(delta) > 1e-4f, "pan moves the pivot");
    CHECK_NEAR(m3::dot(delta, fwd), 0.0, 1e-4, "pan stays perpendicular to the view direction");
    CHECK_NEAR(cam.distance(), d1, 1e-4, "pan does not change distance");
}

static void testPivotPick() {
    std::printf("camera: pick-to-pivot keeps the eye still\n");
    viewer::OrbitCamera cam;
    cam.setViewport(1000, 1000);
    cam.frameBounds({-20, -20, 0}, {20, 20, 10});

    const m3::Vec3 eyeBefore = cam.eye();
    const m3::Vec3 target{7.5f, -3.0f, 1.5f};
    cam.setPivotKeepingEye(target);

    CHECK_NEAR(m3::length(cam.eye() - eyeBefore), 0.0, 1e-3,
               "eye does not move when the pivot changes");
    CHECK_NEAR(m3::length(cam.pivot() - target), 0.0, 1e-5, "pivot moved to the target");
    CHECK_NEAR(cam.distance(), m3::length(eyeBefore - target), 1e-3,
               "distance re-derived from the new pivot");

    // And orbiting now turns about the new point.
    const m3::Vec3 p = cam.pivot();
    cam.orbit(90.0f, 10.0f);
    CHECK_NEAR(m3::length(cam.pivot() - p), 0.0, 1e-6, "orbit turns about the picked point");
}

// ---------------------------------------------------------------------------
// Classification

static void testClassifyMetadata() {
    std::printf("classify: metadata evidence\n");

    e57::Scan spherical;
    spherical.proto.push_back({"sphericalRange", e57::FieldType::ScaledInteger, 0, 1000, 0.001, 0, 10});
    CHECK(check::classifyMetadata(spherical).kind == check::Kind::Structured,
          "spherical coordinates imply a single origin");

    e57::Scan gridded;
    gridded.hasIndexBounds = true;
    gridded.rowMin = 0; gridded.rowMax = 2047;
    gridded.colMin = 0; gridded.colMax = 4095;
    CHECK(check::classifyMetadata(gridded).kind == check::Kind::Structured,
          "indexBounds declares a grid");

    e57::Scan rowcol;
    rowcol.proto.push_back({"rowIndex",    e57::FieldType::Integer, 0, 2047, 1, 0, 11});
    rowcol.proto.push_back({"columnIndex", e57::FieldType::Integer, 0, 4095, 1, 0, 12});
    CHECK(check::classifyMetadata(rowcol).kind == check::Kind::Structured,
          "varying row/column indices imply a grid");

    // A constant row index carries no information and must not count.
    e57::Scan constRow;
    constRow.proto.push_back({"rowIndex",    e57::FieldType::Integer, 7, 7, 1, 0, 0});
    constRow.proto.push_back({"columnIndex", e57::FieldType::Integer, 3, 3, 1, 0, 0});
    CHECK(check::classifyMetadata(constRow).kind != check::Kind::Structured,
          "constant row/column indices are not evidence of a grid");

    e57::Scan bare;
    bare.proto.push_back({"cartesianX", e57::FieldType::FloatDouble, 0, 0, 1, 0, 0});
    const check::Result r = check::classifyMetadata(bare);
    CHECK(r.kind == check::Kind::Ambiguous, "no metadata is ambiguous, not a rejection");
    CHECK(!r.evidence.empty(), "reports why");
}

// Builds a cartesian scan. `shells` == 1 gives one surface per direction (a
// single setup); more stacks surfaces along the same ray, which is what a
// merged cloud looks like from any one origin.
static fixture::Scan makeShellScan(const char* name, int shells, size_t perShell) {
    fixture::Scan s;
    s.name = name;
    s.fields = {
        {"cartesianX", e57::FieldType::FloatDouble},
        {"cartesianY", e57::FieldType::FloatDouble},
        {"cartesianZ", e57::FieldType::FloatDouble},
    };
    s.data.assign(3, {});
    Lcg rng;
    for (size_t i = 0; i < perShell; ++i) {
        // Uniform on the sphere.
        const double u = 2.0 * rng.next() - 1.0;
        const double th = 2.0 * 3.14159265358979 * rng.next();
        const double sr = std::sqrt(std::max(0.0, 1.0 - u * u));
        const double dx = sr * std::cos(th), dy = sr * std::sin(th), dz = u;
        for (int k = 0; k < shells; ++k) {
            // Base radius varies smoothly with direction so the "room" is not a
            // perfect sphere; shells are 3 m apart, well over the 0.5 m
            // multi-surface threshold.
            const double r = 5.0 + 1.5 * dx * dy + 3.0 * double(k);
            s.data[0].push_back(r * dx);
            s.data[1].push_back(r * dy);
            s.data[2].push_back(r * dz);
        }
    }
    return s;
}

static void testClassifyGeometry() {
    std::printf("classify: range-image geometry\n");

    {
        const std::string p = tmpPath("single_setup");
        CHECK(fixture::write(p, {makeShellScan("single", 1, 60000)}, 1024), "fixture written");
        e57::Reader r;
        std::string err;
        CHECK(r.open(p, err), err.empty() ? "opened" : err.c_str());
        const check::Result res = check::classify(r, 0);
        CHECK(res.binsTested > 200, "enough direction bins populated to decide");
        CHECK(res.multiSurfaceFraction >= 0.0 && res.multiSurfaceFraction < 0.20,
              "single setup: few directions hit multiple surfaces");
        CHECK(res.kind == check::Kind::Structured, "single setup accepted");
        CHECK(res.usable(), "single setup is usable");
    }

    {
        const std::string p = tmpPath("merged");
        CHECK(fixture::write(p, {makeShellScan("merged", 3, 60000)}, 1024), "fixture written");
        e57::Reader r;
        std::string err;
        CHECK(r.open(p, err), err.empty() ? "opened" : err.c_str());
        const check::Result res = check::classify(r, 0);
        CHECK(res.multiSurfaceFraction > 0.20,
              "merged cloud: most directions hit multiple surfaces");
        CHECK(res.kind == check::Kind::Unified, "merged cloud rejected");
        CHECK(!res.usable(), "merged cloud is not usable");
        CHECK(!res.summary.empty(), "rejection carries a reason for the UI");
    }
}

// ---------------------------------------------------------------------------
// Cloud loading

static void testDecimationAndPrecision() {
    std::printf("load: decimation and georeferenced precision\n");

    // Coordinates at UTM magnitude with millimetre structure. Held naively in
    // float32 this quantises to ~0.0625 m and the detail vanishes entirely.
    fixture::Scan s;
    s.name = "utm";
    s.hasPose = true;
    s.t[0] = 500000.0; s.t[1] = 6200000.0; s.t[2] = 45.0;
    s.fields = {
        {"cartesianX", e57::FieldType::FloatDouble},
        {"cartesianY", e57::FieldType::FloatDouble},
        {"cartesianZ", e57::FieldType::FloatDouble},
    };
    const size_t N = 20000;
    s.data.assign(3, {});
    for (size_t i = 0; i < N; ++i) {
        s.data[0].push_back(500000.0 + 0.001 * double(i));
        s.data[1].push_back(6200000.0 - 0.002 * double(i));
        s.data[2].push_back(45.0 + 0.0005 * double(i));
    }
    const std::string p = tmpPath("utm");
    CHECK(fixture::write(p, {s}, 512), "fixture written");

    e57::Reader r;
    std::string err;
    CHECK(r.open(p, err), err.empty() ? "opened" : err.c_str());

    viewer::LoadOptions opt;
    opt.maxPoints = 5000;                    // force decimation
    viewer::PointCloud pc;
    CHECK(viewer::loadCloud(r, 0, opt, pc, err), err.empty() ? "loaded" : err.c_str());

    CHECK(pc.pointCount() > 0, "produced points");
    CHECK(pc.pointCount() <= opt.maxPoints, "respects the point budget");
    CHECK(pc.sourcePointCount == N, "reports the undecimated count");

    // The origin must absorb the large magnitude.
    CHECK(std::fabs(pc.originX) > 1e5, "origin carries the UTM easting");
    float maxOff = 0;
    for (float v : pc.xyz) maxOff = std::max(maxOff, std::fabs(v));
    CHECK(maxOff < 100.0f, "stored offsets are small enough for float32");

    // Millimetre steps must survive. At 5e5 held in float32 they could not.
    const uint64_t stride = (N + opt.maxPoints - 1) / opt.maxPoints;
    bool precise = true;
    for (size_t i = 0; i < pc.pointCount() && i < 50; ++i) {
        const double wantX = 500000.0 + 0.001 * double(i * stride);
        const double gotX  = pc.originX + double(pc.xyz[i * 3 + 0]);
        if (std::fabs(gotX - wantX) > 1e-4) precise = false;
    }
    CHECK(precise, "millimetre detail survives at UTM magnitude");

    CHECK(pc.hasSetupPosition, "setup position recovered from pose");
    CHECK_NEAR(double(pc.originOffset[0]) + pc.originX, 500000.0, 1e-3, "setup x in local frame");
}

// Builds a flat grid of points in a plane at a given height, as one cloud.
static viewer::PointCloud makeGrid(double ox, double oy, double oz, float z, int n, float step) {
    viewer::PointCloud c;
    c.originX = ox; c.originY = oy; c.originZ = oz;
    for (int i = -n; i <= n; ++i)
        for (int j = -n; j <= n; ++j) {
            c.xyz.push_back(float(i) * step);
            c.xyz.push_back(float(j) * step);
            c.xyz.push_back(z);
        }
    c.loMin[0] = -n * step; c.loMin[1] = -n * step; c.loMin[2] = z;
    c.hiMax[0] =  n * step; c.hiMax[1] =  n * step; c.hiMax[2] = z;
    return c;
}

static void testPicker() {
    std::printf("picker: screen-centre pick\n");

    // Two planes: one at z=0, one at z=4. Looking down, the pick must return
    // the nearer (upper) one.
    std::vector<viewer::PointCloud> clouds;
    clouds.push_back(makeGrid(0, 0, 0, 0.0f, 40, 0.25f));
    clouds.push_back(makeGrid(0, 0, 0, 4.0f, 40, 0.25f));
    viewer::setSceneOrigin(clouds);

    m3::Vec3 lo, hi;
    CHECK(viewer::sceneBounds(clouds, lo, hi), "scene bounds computed");
    CHECK_NEAR(lo.z, 0.0, 1e-6, "bounds include the lower plane");
    CHECK_NEAR(hi.z, 4.0, 1e-6, "bounds include the upper plane");

    viewer::OrbitCamera cam;
    cam.setViewport(900, 900);
    cam.frameBounds(lo, hi);

    const viewer::PickResult r = viewer::pickNearest(clouds, cam, 0.0f, 0.0f, 12.0f);
    CHECK(r.hit, "something is under the crosshair");
    CHECK(r.cloudIndex == 1, "picks the nearer plane, not the one behind it");

    // The picked point must actually project near the crosshair.
    const m3::Vec4 clip = cam.viewProjection() *
                          m3::Vec4{r.world.x, r.world.y, r.world.z, 1.0f};
    CHECK(clip.w > 0, "picked point is in front of the camera");
    CHECK(std::fabs(clip.x / clip.w) < 0.05, "picked point is near the crosshair in x");
    CHECK(std::fabs(clip.y / clip.w) < 0.05, "picked point is near the crosshair in y");

    // Aimed off into empty space, nothing is picked.
    const viewer::PickResult miss = viewer::pickNearest(clouds, cam, 0.98f, 0.98f, 2.0f);
    CHECK(!miss.hit, "no point under an empty corner of the view");

    // Picking then pivoting is the actual gesture: eye must not move.
    const m3::Vec3 eyeBefore = cam.eye();
    cam.setPivotKeepingEye(r.world);
    CHECK_NEAR(m3::length(cam.eye() - eyeBefore), 0.0, 1e-3,
               "pick-to-pivot leaves the view unchanged");
}

// Clouds from different coordinate origins must line up in the scene frame.
static void testSceneOriginAlignment() {
    std::printf("picker: multi-cloud scene frame\n");
    std::vector<viewer::PointCloud> clouds;
    clouds.push_back(makeGrid(500000.0, 6200000.0, 40.0, 0.0f, 5, 1.0f));
    clouds.push_back(makeGrid(500010.0, 6200000.0, 40.0, 0.0f, 5, 1.0f));
    viewer::setSceneOrigin(clouds);

    CHECK_NEAR(clouds[0].sceneOffset[0], 0.0, 1e-6, "first cloud defines the origin");
    CHECK_NEAR(clouds[1].sceneOffset[0], 10.0, 1e-3, "second cloud offset by its true separation");

    m3::Vec3 lo, hi;
    CHECK(viewer::sceneBounds(clouds, lo, hi), "bounds computed");
    CHECK_NEAR(hi.x - lo.x, 20.0, 1e-3, "scene spans both clouds");
}

int main() {
    std::printf("E57 Coverage Checker — viewer tests\n\n");
    testCameraProjection();
    testCameraNavigation();
    testPivotPick();
    testClassifyMetadata();
    testClassifyGeometry();
    testDecimationAndPrecision();
    testPicker();
    testSceneOriginAlignment();
    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
