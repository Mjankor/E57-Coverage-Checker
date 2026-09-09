// Tests for the out-of-core LOD engine.
//
// This is the part that decides whether the app copes with thousands of
// setups, so the properties worth asserting are the scaling ones: the point
// budget is a hard ceiling however large the store is, detail follows the
// camera, and off-screen data costs nothing.

#include "../src/camera.h"
#include "../src/lod.h"
#include "../src/point_store.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <set>
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

struct Lcg {
    uint64_t s = 0x243F6A8885A308D3ull;
    double next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return double((s >> 11) & ((1ull << 53) - 1)) / double(1ull << 53);
    }
};

// A synthetic site: a ground plane with a few blocks on it, sampled by many
// "setups". Stands in for a real job's spatial distribution — mostly flat,
// locally dense, spread far wider than any one setup can see.
static std::vector<lod::StorePoint> makeSite(int setups, int perSetup, float extent) {
    std::vector<lod::StorePoint> pts;
    pts.reserve(size_t(setups) * size_t(perSetup));
    Lcg rng;
    for (int s = 0; s < setups; ++s) {
        const float cx = float(rng.next() * 2.0 - 1.0) * extent;
        const float cy = float(rng.next() * 2.0 - 1.0) * extent;
        for (int i = 0; i < perSetup; ++i) {
            const double a = rng.next() * 6.28318530718;
            const double r = 2.0 + rng.next() * 18.0;
            lod::StorePoint p{};
            p.x = cx + float(std::cos(a) * r);
            p.y = cy + float(std::sin(a) * r);
            p.z = (rng.next() < 0.25) ? float(rng.next() * 6.0) : 0.0f;
            p.r = 200; p.g = 200; p.b = 200; p.a = 255;
            p.scanId = uint16_t(s);
            pts.push_back(p);
        }
    }
    return pts;
}

static lod::Aabb boundsOf(const std::vector<lod::StorePoint>& pts) {
    lod::Aabb b;
    b.lo[0] = b.hi[0] = pts[0].x;
    b.lo[1] = b.hi[1] = pts[0].y;
    b.lo[2] = b.hi[2] = pts[0].z;
    for (const auto& p : pts) b.expand(p.x, p.y, p.z);
    return b;
}

// ---------------------------------------------------------------------------

static void testAabb() {
    std::printf("lod: bounds and octants\n");
    lod::Aabb b;
    b.lo[0] = -4; b.hi[0] = 6;
    b.lo[1] = -1; b.hi[1] = 1;
    b.lo[2] =  0; b.hi[2] = 2;

    const lod::Aabb cube = lod::cubeAround(b);
    CHECK_NEAR(cube.hi[0] - cube.lo[0], cube.hi[1] - cube.lo[1], 1e-4, "cube is square in x/y");
    CHECK_NEAR(cube.hi[1] - cube.lo[1], cube.hi[2] - cube.lo[2], 1e-4, "cube is square in y/z");
    CHECK(cube.lo[0] <= b.lo[0] && cube.hi[0] >= b.hi[0], "cube contains the box in x");
    CHECK(cube.lo[1] <= b.lo[1] && cube.hi[1] >= b.hi[1], "cube contains the box in y");
    CHECK(cube.lo[2] <= b.lo[2] && cube.hi[2] >= b.hi[2], "cube contains the box in z");

    // A degenerate input must still give a usable root.
    lod::Aabb pt;
    const lod::Aabb pcube = lod::cubeAround(pt);
    CHECK(pcube.diagonal() > 0.0f, "a zero-size box still yields a non-degenerate cube");

    // The eight octants must tile the parent exactly.
    float vol = 0;
    for (int o = 0; o < 8; ++o) {
        const lod::Aabb c = cube.child(o);
        vol += (c.hi[0] - c.lo[0]) * (c.hi[1] - c.lo[1]) * (c.hi[2] - c.lo[2]);
    }
    const float parentVol = (cube.hi[0] - cube.lo[0]) * (cube.hi[1] - cube.lo[1]) *
                            (cube.hi[2] - cube.lo[2]);
    CHECK_NEAR(vol, parentVol, parentVol * 1e-4, "octants tile the parent");
}

static void testBuildIsAdditiveAndSpatiallySound() {
    std::printf("lod: additive build\n");
    const std::vector<lod::StorePoint> pts = makeSite(40, 8000, 120.0f);
    const lod::Aabb root = lod::cubeAround(boundsOf(pts));

    lod::BuildOptions opt;
    opt.gridResolution = 32;
    opt.maxPointsPerNode = 40000;
    opt.maxLevel = 10;

    lod::Builder b(root, opt);
    bool allAccepted = true;
    for (const auto& p : pts) if (!b.insert(p)) allAccepted = false;
    CHECK(allAccepted, "every point inside the root is accepted");

    std::vector<std::vector<lod::StorePoint>> payload;
    const lod::Tree t = b.finish(payload);

    CHECK(!t.nodes.empty(), "tree has nodes");
    CHECK(payload.size() == t.nodes.size(), "one payload per node");

    // Additive: every point appears exactly once across the whole tree.
    size_t sum = 0;
    for (const auto& v : payload) sum += v.size();
    CHECK(sum == b.insertedCount(), "no point is duplicated or lost across the tree");
    CHECK(t.totalPoints == sum, "tree total matches the payloads");
    CHECK(b.overflowCount() == 0, "nothing overflowed at this depth");

    // Every point lies inside the node that holds it.
    bool contained = true;
    for (size_t n = 0; n < t.nodes.size() && contained; ++n) {
        const lod::Aabb& bb = t.nodes[n].bounds;
        for (const auto& p : payload[n]) {
            const float eps = 1e-3f;
            if (p.x < bb.lo[0] - eps || p.x > bb.hi[0] + eps ||
                p.y < bb.lo[1] - eps || p.y > bb.hi[1] + eps ||
                p.z < bb.lo[2] - eps || p.z > bb.hi[2] + eps) { contained = false; break; }
        }
    }
    CHECK(contained, "every point lies within its node's bounds");

    // The root must be a coarse sample of the whole site, not a dumping
    // ground: this is what makes a first frame cheap at any scale.
    CHECK(t.nodes[0].pointCount < sum / 10, "root holds a small fraction of the points");
    CHECK(t.nodes[0].pointCount > 0, "root is not empty");

    // Depth must actually develop, or there is no level of detail.
    uint8_t maxLevel = 0;
    for (const auto& n : t.nodes) maxLevel = std::max(maxLevel, n.level);
    CHECK(maxLevel >= 3, "the tree develops several levels");

    // Child links must be in range, exactly one level deeper, and each child
    // must sit inside the octant of its parent that it claims.
    bool links = true;
    for (const auto& n : t.nodes) {
        for (int o = 0; o < 8; ++o) {
            if (!n.hasChild(o)) continue;
            const uint32_t c = n.child[o];
            if (c == 0 || c >= t.nodes.size()) { links = false; break; }
            if (t.nodes[c].level != n.level + 1) { links = false; break; }
            const lod::Aabb want = n.bounds.child(o);
            for (int i = 0; i < 3; ++i)
                if (std::fabs(t.nodes[c].bounds.lo[i] - want.lo[i]) > 1e-3f) links = false;
        }
        if (!links) break;
    }
    CHECK(links, "child links are valid, one level deeper, and in the right octant");
}

static void testOverflowSpills() {
    std::printf("lod: depth limit spills rather than dropping\n");
    // Force everything into one node: depth 0 and a tiny capacity.
    std::vector<lod::StorePoint> pts = makeSite(1, 5000, 1.0f);
    const lod::Aabb root = lod::cubeAround(boundsOf(pts));

    lod::BuildOptions opt;
    opt.gridResolution = 4;
    opt.maxPointsPerNode = 50;
    opt.maxLevel = 0;

    std::vector<lod::StorePoint> spill;
    lod::Builder b(root, opt);
    b.setOverflowSink(&spill);
    for (const auto& p : pts) b.insert(p);

    std::vector<std::vector<lod::StorePoint>> payload;
    const lod::Tree t = b.finish(payload);
    CHECK(t.nodes.size() == 1, "depth limit of zero gives a single node");
    CHECK(b.insertedCount() + spill.size() == pts.size(),
          "every point is either kept or spilled, none silently dropped");
    CHECK(!spill.empty(), "the excess actually reached the sink");
}

static void testFrustumCulling() {
    std::printf("lod: frustum culling\n");
    viewer::OrbitCamera cam;
    cam.setViewport(1000, 1000);
    cam.frameBounds({-10, -10, -10}, {10, 10, 10});
    const m3::Mat4 vp = cam.viewProjection();
    const lod::Frustum f = lod::frustumFromViewProjection(vp);

    lod::Aabb centre;
    centre.lo[0] = -1; centre.hi[0] = 1;
    centre.lo[1] = -1; centre.hi[1] = 1;
    centre.lo[2] = -1; centre.hi[2] = 1;
    CHECK(f.intersects(centre), "a box at the pivot is inside the frustum");

    // Directly behind the camera, far outside the view.
    const m3::Vec3 eye = cam.eye();
    const m3::Vec3 back = eye + (eye - cam.pivot());
    lod::Aabb behind;
    for (int i = 0; i < 3; ++i) {
        const float c = (i == 0 ? back.x : i == 1 ? back.y : back.z);
        behind.lo[i] = c - 0.5f; behind.hi[i] = c + 0.5f;
    }
    CHECK(!f.intersects(behind), "a box behind the camera is culled");

    lod::Aabb faraway;
    for (int i = 0; i < 3; ++i) { faraway.lo[i] = 1e6f; faraway.hi[i] = 1e6f + 1.0f; }
    CHECK(!f.intersects(faraway), "a box far outside the view is culled");
}

static void testSelectionScales() {
    std::printf("lod: selection respects the budget and follows the camera\n");
    const std::vector<lod::StorePoint> pts = makeSite(60, 12000, 200.0f);
    const lod::Aabb root = lod::cubeAround(boundsOf(pts));

    lod::BuildOptions bopt;
    bopt.gridResolution = 32;
    bopt.maxPointsPerNode = 20000;
    bopt.maxLevel = 12;
    lod::Builder b(root, bopt);
    for (const auto& p : pts) b.insert(p);
    std::vector<std::vector<lod::StorePoint>> payload;
    const lod::Tree t = b.finish(payload);

    viewer::OrbitCamera cam;
    cam.setViewport(1600, 1000);
    cam.frameBounds({root.lo[0], root.lo[1], root.lo[2]},
                    {root.hi[0], root.hi[1], root.hi[2]});
    const float pixelsPerRadian = 1000.0f / 1.0471976f;

    // The whole site in view, with a budget far below the total.
    lod::SelectOptions sopt;
    sopt.pointBudget = 200000;
    const lod::Selection wide =
        lod::selectNodes(t, cam.viewProjection(), cam.eye(), pixelsPerRadian, sopt);

    CHECK(wide.points <= sopt.pointBudget, "the budget is a hard ceiling");
    CHECK(wide.points > 0, "something is selected");
    CHECK(wide.points < t.totalPoints, "far less than the whole store is drawn");
    CHECK(!wide.nodes.empty(), "nodes were selected");

    std::set<uint32_t> unique(wide.nodes.begin(), wide.nodes.end());
    CHECK(unique.size() == wide.nodes.size(), "no node is selected twice");

    // The root must come first: it is the largest thing on screen, so the
    // budget is spent coarse-to-fine and a partial cut still covers the site.
    CHECK(!wide.nodes.empty() && wide.nodes[0] == 0, "selection starts at the root");

    // Zooming onto real geometry must deepen the cut there. Aiming at a
    // corner of the padded root cube instead would prove nothing: that corner
    // is empty space, so there are no deep nodes to descend into.
    viewer::OrbitCamera close = cam;
    close.setPivotKeepingEye({pts[0].x, pts[0].y, pts[0].z});
    // Negative ticks zoom in: scrolling towards you pulls the model closer,
    // matching the rest of the system. See OrbitCamera::zoom.
    for (int i = 0; i < 40; ++i) close.zoom(-1.0f);

    const lod::Selection near =
        lod::selectNodes(t, close.viewProjection(), close.eye(), pixelsPerRadian, sopt);
    CHECK(near.points <= sopt.pointBudget, "the budget still holds when zoomed in");

    uint8_t wideMax = 0, nearMax = 0;
    double wideSum = 0, nearSum = 0;
    for (uint32_t n : wide.nodes) { wideMax = std::max(wideMax, t.nodes[n].level); wideSum += t.nodes[n].level; }
    for (uint32_t n : near.nodes) { nearMax = std::max(nearMax, t.nodes[n].level); nearSum += t.nodes[n].level; }
    const double wideMean = wideSum / double(wide.nodes.size());
    const double nearMean = nearSum / double(near.nodes.size());
    CHECK(nearMax > wideMax, "zooming in reaches deeper nodes than the overview");
    CHECK(nearMean > wideMean + 1.0, "and the cut is deeper on average, not just at one node");
    CHECK(near.nodesCulled > 0, "zoomed in, most of the site is culled rather than drawn");

    // A tiny budget must still produce something rather than failing.
    sopt.pointBudget = 1000;
    const lod::Selection tiny =
        lod::selectNodes(t, cam.viewProjection(), cam.eye(), pixelsPerRadian, sopt);
    CHECK(tiny.points <= 1000, "a tiny budget is respected");
    CHECK(tiny.budgetExhausted, "and is reported as exhausted");

    // A budget above the total must select everything visible without looping.
    sopt.pointBudget = t.totalPoints * 2;
    const lod::Selection all =
        lod::selectNodes(t, cam.viewProjection(), cam.eye(), pixelsPerRadian, sopt);
    CHECK(!all.budgetExhausted, "an ample budget is not reported exhausted");
    CHECK(all.points <= t.totalPoints, "never selects more than the store holds");

    // Looking away from the site should cost essentially nothing.
    viewer::OrbitCamera away = cam;
    away.setPivotKeepingEye({root.hi[0] + 1.0e5f, root.hi[1] + 1.0e5f, 0.0f});
    const lod::Selection none =
        lod::selectNodes(t, away.viewProjection(), away.eye(), pixelsPerRadian, sopt);
    CHECK(none.points < wide.points, "pointing away from the data draws less");
}

// The scaling claim itself: a store built from many setups must still be
// drawable inside one fixed budget, and selection cost must not grow with the
// number of setups.
static void testThousandsOfSetups() {
    std::printf("lod: a thousand setups stays within budget\n");
    const std::vector<lod::StorePoint> pts = makeSite(1000, 2000, 400.0f);
    CHECK(pts.size() == 2000000, "two million points across a thousand setups");

    const lod::Aabb root = lod::cubeAround(boundsOf(pts));
    lod::BuildOptions bopt;
    bopt.gridResolution = 32;
    bopt.maxPointsPerNode = 30000;
    bopt.maxLevel = 12;
    lod::Builder b(root, bopt);
    for (const auto& p : pts) b.insert(p);
    std::vector<std::vector<lod::StorePoint>> payload;
    const lod::Tree t = b.finish(payload);
    CHECK(t.totalPoints == pts.size(), "the whole corpus is in the tree");

    viewer::OrbitCamera cam;
    cam.setViewport(1600, 1000);
    cam.frameBounds({root.lo[0], root.lo[1], root.lo[2]},
                    {root.hi[0], root.hi[1], root.hi[2]});

    lod::SelectOptions sopt;
    sopt.pointBudget = 500000;
    const lod::Selection sel =
        lod::selectNodes(t, cam.viewProjection(), cam.eye(), 1000.0f / 1.0471976f, sopt);

    CHECK(sel.points <= sopt.pointBudget, "budget holds for a thousand-setup store");
    // The whole point of the exercise: work per frame is bounded by the budget
    // and the visible node count, not by how much data exists.
    CHECK(sel.nodesVisited < t.nodes.size(), "selection does not walk the whole tree");
    CHECK(sel.nodes.size() < 2000, "a bounded number of draws per frame");

    // Scan identity must survive into the payloads, since "which setups cover
    // this space" is the question the whole tool exists to answer.
    std::set<uint16_t> ids;
    for (const auto& v : payload)
        for (const auto& p : v) ids.insert(p.scanId);
    CHECK(ids.size() == 1000, "every setup is still individually identifiable");
}

// ---------------------------------------------------------------------------
// On-disk store

static std::string storePath(const char* name) {
    const char* dir = std::getenv("E57COV_TMPDIR");
    return std::string(dir ? dir : "/tmp") + "/e57cov_" + name + ".lod";
}

static void testStoreRoundTrip() {
    std::printf("store: write and mmap back\n");
    const std::vector<lod::StorePoint> pts = makeSite(24, 5000, 60.0f);
    const lod::Aabb root = lod::cubeAround(boundsOf(pts));

    lod::BuildOptions bopt;
    bopt.gridResolution = 24;
    bopt.maxPointsPerNode = 8000;
    bopt.maxLevel = 9;
    lod::Builder b(root, bopt);
    for (const auto& p : pts) b.insert(p);
    std::vector<std::vector<lod::StorePoint>> payload;
    const lod::Tree t = b.finish(payload);

    const std::string path = storePath("roundtrip");
    std::string err;
    {
        store::Writer w;
        CHECK(w.open(path, err), err.empty() ? "store created" : err.c_str());
        w.setOrigin(500000.0, 6200000.0, 42.0);
        w.setRoot(root, bopt.gridResolution, bopt.maxLevel);
        for (int i = 0; i < 24; ++i) {
            store::ScanRecord s{};
            std::snprintf(s.name, sizeof(s.name), "Setup %03d", i);
            s.poseT[0] = double(i); s.poseQ[0] = 1.0;
            s.sourcePoints = 5000;
            s.flags = store::kScanStructured | store::kScanPoseApplied;
            w.addScan(s);
        }
        uint32_t rootIndex = 0;
        CHECK(w.appendTree(t, payload, rootIndex, err), err.empty() ? "tree appended" : err.c_str());
        CHECK(rootIndex == 0, "the first appended tree roots the store");
        CHECK(w.finish(err), err.empty() ? "store finished" : err.c_str());
    }

    store::Reader r;
    CHECK(r.open(path, err), err.empty() ? "store reopened" : err.c_str());
    CHECK(r.nodeCount() == t.nodes.size(), "node count survives the round trip");
    CHECK(r.scanCount() == 24, "scan table survives");
    CHECK(r.header().totalPoints == t.totalPoints, "total point count survives");
    CHECK_NEAR(r.header().origin[0], 500000.0, 1e-9, "double-precision origin survives");
    CHECK(std::string(r.scan(7).name) == "Setup 007", "scan names survive");
    CHECK((r.scan(7).flags & store::kScanStructured) != 0, "scan flags survive");

    // Payloads must come back byte-identical, straight out of the mapping.
    bool same = true;
    size_t seen = 0;
    for (uint64_t n = 0; n < r.nodeCount() && same; ++n) {
        const lod::StorePoint* p = r.points(n);
        CHECK_NEAR(double(r.node(n).pointCount), double(payload[n].size()), 0.0,
                   "per-node count matches");
        for (uint32_t i = 0; i < r.node(n).pointCount; ++i, ++seen) {
            if (std::memcmp(&p[i], &payload[n][i], sizeof(lod::StorePoint)) != 0) same = false;
        }
    }
    CHECK(same, "every point round-trips byte for byte");
    CHECK(seen == t.totalPoints, "every point is reachable through the node table");

    // The rebuilt tree must select exactly as the in-memory one did.
    const lod::Tree rt = r.tree();
    viewer::OrbitCamera cam;
    cam.setViewport(1200, 800);
    cam.frameBounds({root.lo[0], root.lo[1], root.lo[2]}, {root.hi[0], root.hi[1], root.hi[2]});
    lod::SelectOptions so; so.pointBudget = 50000;
    const lod::Selection a = lod::selectNodes(t,  cam.viewProjection(), cam.eye(), 800.0f, so);
    const lod::Selection c = lod::selectNodes(rt, cam.viewProjection(), cam.eye(), 800.0f, so);
    CHECK(a.nodes == c.nodes, "selection on the mmap'd tree matches the in-memory tree");
    CHECK(a.points == c.points, "and selects the same number of points");
}

static void testStoreStitchesSubtrees() {
    std::printf("store: subtrees stitched into one tree\n");
    // Two independently built subtrees joined under a shared root — the shape
    // a bounded-memory build produces, where each chunk is built alone and
    // never coexists in memory with the others.
    lod::Aabb root;
    for (int i = 0; i < 3; ++i) { root.lo[i] = -100; root.hi[i] = 100; }

    lod::BuildOptions o; o.gridResolution = 8; o.maxPointsPerNode = 500; o.maxLevel = 4;

    auto buildIn = [&](const lod::Aabb& box, float cx, float cy, int n) {
        lod::Builder bb(box, o);
        Lcg rng;
        for (int i = 0; i < n; ++i) {
            lod::StorePoint p{};
            p.x = cx + float(rng.next() * 20.0 - 10.0);
            p.y = cy + float(rng.next() * 20.0 - 10.0);
            p.z = float(rng.next() * 4.0);
            p.scanId = 0;
            bb.insert(p);
        }
        return bb;
    };

    lod::Builder topB(root, o);
    Lcg rng;
    for (int i = 0; i < 400; ++i) {
        lod::StorePoint p{};
        p.x = float(rng.next() * 180.0 - 90.0);
        p.y = float(rng.next() * 180.0 - 90.0);
        p.z = float(rng.next() * 4.0);
        topB.insert(p);
    }
    std::vector<std::vector<lod::StorePoint>> topPay;
    const lod::Tree top = topB.finish(topPay);

    lod::Builder subB = buildIn(root.child(0), -50.0f, -50.0f, 3000);
    std::vector<std::vector<lod::StorePoint>> subPay;
    const lod::Tree sub = subB.finish(subPay);

    const std::string path = storePath("stitched");
    std::string err;
    uint32_t topRoot = 0, subRoot = 0;
    {
        store::Writer w;
        CHECK(w.open(path, err), "store created");
        w.setRoot(root, o.gridResolution, o.maxLevel);
        CHECK(w.appendTree(top, topPay, topRoot, err), "top appended");
        CHECK(w.appendTree(sub, subPay, subRoot, err), "subtree appended");
        CHECK(subRoot != 0, "the second subtree does not land on index 0");
        CHECK(w.linkChild(topRoot, 0, subRoot, err), "subtree linked under the top");
        CHECK(!w.linkChild(topRoot, 0, 0, err), "linking node 0 as a child is refused");
        CHECK(w.finish(err), "store finished");
    }

    store::Reader r;
    CHECK(r.open(path, err), err.empty() ? "reopened" : err.c_str());
    const lod::Tree joined = r.tree();
    CHECK(joined.nodes.size() == top.nodes.size() + sub.nodes.size(), "all nodes present");
    CHECK(joined.nodes[0].child[0] == subRoot, "the link survives the round trip");

    // Walking from the root must reach every node exactly once: a broken
    // remap would strand a subtree, and nothing on screen would show it.
    std::vector<int> visits(joined.nodes.size(), 0);
    std::vector<uint32_t> stack{0};
    while (!stack.empty()) {
        const uint32_t n = stack.back(); stack.pop_back();
        if (++visits[n] > 1) continue;
        for (int oc = 0; oc < 8; ++oc)
            if (joined.nodes[n].child[oc]) stack.push_back(joined.nodes[n].child[oc]);
    }
    size_t unreachable = 0, revisited = 0;
    for (int v : visits) { if (v == 0) ++unreachable; if (v > 1) ++revisited; }
    CHECK(unreachable == 0, "every node is reachable from the root");
    CHECK(revisited == 0, "and reachable by exactly one path");
}

static void testStoreRejectsBadFiles() {
    std::printf("store: malformed input\n");
    const std::string path = storePath("bad");
    std::FILE* f = std::fopen(path.c_str(), "wb");
    std::vector<uint8_t> junk(4096, 0x5A);
    std::fwrite(junk.data(), 1, junk.size(), f);
    std::fclose(f);

    store::Reader r;
    std::string err;
    CHECK(!r.open(path, err), "junk is rejected");
    CHECK(err.find("magic") != std::string::npos, "and the reason names the magic");

    store::Reader missing;
    CHECK(!missing.open(storePath("does_not_exist"), err), "a missing store is rejected");
}

int main() {
    std::printf("E57 Coverage Checker — LOD tests\n\n");
    testAabb();
    testBuildIsAdditiveAndSpatiallySound();
    testOverflowSpills();
    testFrustumCulling();
    testSelectionScales();
    testThousandsOfSetups();
    testStoreRoundTrip();
    testStoreStitchesSubtrees();
    testStoreRejectsBadFiles();
    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
