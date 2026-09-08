#include "lod.h"

#include <algorithm>
#include <cmath>
#include <queue>

namespace lod {

// ---------------------------------------------------------------------------
// Aabb

void Aabb::expand(float x, float y, float z) {
    lo[0] = std::min(lo[0], x); hi[0] = std::max(hi[0], x);
    lo[1] = std::min(lo[1], y); hi[1] = std::max(hi[1], y);
    lo[2] = std::min(lo[2], z); hi[2] = std::max(hi[2], z);
}

float Aabb::diagonal() const {
    const float dx = hi[0] - lo[0], dy = hi[1] - lo[1], dz = hi[2] - lo[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

void Aabb::centre(float out[3]) const {
    for (int i = 0; i < 3; ++i) out[i] = 0.5f * (lo[i] + hi[i]);
}

Aabb Aabb::child(int octant) const {
    Aabb c;
    for (int i = 0; i < 3; ++i) {
        const float mid = 0.5f * (lo[i] + hi[i]);
        if ((octant >> i) & 1) { c.lo[i] = mid;    c.hi[i] = hi[i]; }
        else                   { c.lo[i] = lo[i];  c.hi[i] = mid;   }
    }
    return c;
}

Aabb cubeAround(const Aabb& box) {
    float c[3];
    box.centre(c);
    float half = 0.0f;
    for (int i = 0; i < 3; ++i) half = std::max(half, 0.5f * (box.hi[i] - box.lo[i]));
    // A degenerate input (a single point, or a perfectly planar scan) would
    // otherwise give a zero-size root that every insert falls outside of.
    half = std::max(half, 1e-3f);
    half *= 1.0f + 1e-4f;   // guard against a point landing exactly on a face
    Aabb r;
    for (int i = 0; i < 3; ++i) { r.lo[i] = c[i] - half; r.hi[i] = c[i] + half; }
    return r;
}

// ---------------------------------------------------------------------------
// Builder

Builder::Builder(const Aabb& bounds, const BuildOptions& opt, uint8_t baseLevel)
    : opt_(opt), baseLevel_(baseLevel) {
    if (opt_.gridResolution == 0) opt_.gridResolution = 1;
    Cell root;
    root.node.level  = baseLevel;
    root.node.bounds = bounds;
    cells_.push_back(std::move(root));
}

bool Builder::claimCell(Cell& c, const StorePoint& p) {
    const uint32_t G = opt_.gridResolution;
    const Aabb& b = c.node.bounds;
    int g[3];
    const float pos[3] = {p.x, p.y, p.z};
    for (int i = 0; i < 3; ++i) {
        const float span = b.hi[i] - b.lo[i];
        const float t = span > 0 ? (pos[i] - b.lo[i]) / span : 0.0f;
        g[i] = std::clamp(int(t * float(G)), 0, int(G) - 1);
    }
    const uint64_t cell = (uint64_t(g[2]) * G + uint64_t(g[1])) * G + uint64_t(g[0]);

    // Lazily sized: a node that never fills does not pay for the whole grid.
    const size_t words = (size_t(G) * G * G + 63) / 64;
    if (c.occupancy.empty()) c.occupancy.assign(words, 0ull);
    const size_t w = size_t(cell >> 6);
    const uint64_t bit = 1ull << (cell & 63);
    if (c.occupancy[w] & bit) return false;
    c.occupancy[w] |= bit;
    return true;
}

int32_t Builder::childOf(int32_t cellIndex, int octant) {
    if (cells_[size_t(cellIndex)].child[octant] >= 0)
        return cells_[size_t(cellIndex)].child[octant];
    Cell c;
    c.node.level  = uint8_t(cells_[size_t(cellIndex)].node.level + 1);
    c.node.bounds = cells_[size_t(cellIndex)].node.bounds.child(octant);
    cells_.push_back(std::move(c));
    const int32_t idx = int32_t(cells_.size() - 1);
    // Note: taking the reference after push_back, since the vector may have
    // reallocated and invalidated any earlier one.
    cells_[size_t(cellIndex)].child[octant] = idx;
    return idx;
}

bool Builder::insert(const StorePoint& p) {
    const Aabb& rb = cells_[0].node.bounds;
    if (p.x < rb.lo[0] || p.x > rb.hi[0] ||
        p.y < rb.lo[1] || p.y > rb.hi[1] ||
        p.z < rb.lo[2] || p.z > rb.hi[2]) return false;

    int32_t cur = 0;
    for (;;) {
        Cell& c = cells_[size_t(cur)];
        const bool atLimit = c.node.level >= opt_.maxLevel;
        const bool room    = c.points.size() < opt_.maxPointsPerNode;

        // claimCell has a side effect, so it is only called when its answer is
        // going to be used.
        if (atLimit) {
            const bool keep = room && (opt_.spillAtMaxLevel ? claimCell(c, p) : true);
            if (keep) {
                c.points.push_back(p);
                ++inserted_;
                return true;
            }
            // Nothing below this level to push into: spill rather than drop.
            ++overflowed_;
            if (overflow_) overflow_(p);
            return true;
        }
        if (room && claimCell(c, p)) {
            c.points.push_back(p);
            ++inserted_;
            return true;
        }

        float mid[3];
        c.node.bounds.centre(mid);
        int octant = 0;
        if (p.x >= mid[0]) octant |= 1;
        if (p.y >= mid[1]) octant |= 2;
        if (p.z >= mid[2]) octant |= 4;
        cur = childOf(cur, octant);
    }
}

Tree Builder::finish(std::vector<std::vector<StorePoint>>& outPoints) {
    // Breadth-first relabelling so children of a node are contiguous, which
    // lets a node reference them with one index plus a mask.
    Tree t;
    t.bounds = cells_[0].node.bounds;
    outPoints.clear();
    if (cells_.empty()) return t;

    // Children of a node are appended together and in octant order, so a node
    // can reference all of them with one index plus the mask.
    std::vector<int32_t> order;          // new index -> cell index
    order.push_back(0);
    for (size_t head = 0; head < order.size(); ++head) {
        const int32_t ci = order[head];
        for (int o = 0; o < 8; ++o) {
            const int32_t child = cells_[size_t(ci)].child[o];
            if (child >= 0) order.push_back(child);
        }
    }

    std::vector<int32_t> newIndexOf(cells_.size(), -1);
    for (size_t i = 0; i < order.size(); ++i) newIndexOf[size_t(order[i])] = int32_t(i);

    t.nodes.resize(order.size());
    outPoints.resize(order.size());
    for (size_t i = 0; i < order.size(); ++i) {
        Cell& c = cells_[size_t(order[i])];
        Node n = c.node;
        n.pointCount = uint32_t(c.points.size());
        for (int o = 0; o < 8; ++o)
            n.child[o] = (c.child[o] < 0) ? 0u
                                          : uint32_t(newIndexOf[size_t(c.child[o])]);
        t.nodes[i] = n;
        outPoints[i] = std::move(c.points);
        t.totalPoints += n.pointCount;
    }
    return t;
}

// ---------------------------------------------------------------------------
// Frustum

Frustum frustumFromViewProjection(const m3::Mat4& m) {
    // Rows of the column-major matrix: row r is (at(0,r), at(1,r), at(2,r), at(3,r)).
    float row[4][4];
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) row[r][c] = m.at(c, r);

    Frustum f;
    for (int i = 0; i < 4; ++i) {
        f.p[0][i] = row[3][i] + row[0][i];   // left
        f.p[1][i] = row[3][i] - row[0][i];   // right
        f.p[2][i] = row[3][i] + row[1][i];   // bottom
        f.p[3][i] = row[3][i] - row[1][i];   // top
        // Metal clip space runs z from 0 to 1, so the near plane is row 2
        // alone — the OpenGL form (row3 + row2) would put it in the wrong place
        // and quietly cull geometry in front of the camera.
        f.p[4][i] = row[2][i];               // near
        f.p[5][i] = row[3][i] - row[2][i];   // far
    }
    for (int i = 0; i < 6; ++i) {
        const float n = std::sqrt(f.p[i][0] * f.p[i][0] + f.p[i][1] * f.p[i][1] +
                                  f.p[i][2] * f.p[i][2]);
        if (n > 1e-20f) for (int k = 0; k < 4; ++k) f.p[i][k] /= n;
    }
    return f;
}

bool Frustum::intersects(const Aabb& box) const {
    for (int i = 0; i < 6; ++i) {
        // Positive vertex: the box corner furthest along the plane normal. If
        // even that is behind the plane, the whole box is.
        const float vx = p[i][0] >= 0 ? box.hi[0] : box.lo[0];
        const float vy = p[i][1] >= 0 ? box.hi[1] : box.lo[1];
        const float vz = p[i][2] >= 0 ? box.hi[2] : box.lo[2];
        if (p[i][0] * vx + p[i][1] * vy + p[i][2] * vz + p[i][3] < 0.0f) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Selection

namespace {

struct Candidate {
    uint32_t node;
    float    pixels;
    bool operator<(const Candidate& o) const { return pixels < o.pixels; }  // max-heap
};

float projectedPixels(const Aabb& box, const m3::Vec3& eye, float pixelsPerRadian) {
    float c[3];
    box.centre(c);
    const float dx = c[0] - eye.x, dy = c[1] - eye.y, dz = c[2] - eye.z;
    const float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
    const float radius = 0.5f * box.diagonal();
    // Inside the node, treat it as filling the view rather than dividing by a
    // vanishing distance.
    if (dist <= radius) return 1e9f;
    return 2.0f * radius / dist * pixelsPerRadian;
}

} // namespace

Selection selectNodes(const Tree& tree, const m3::Mat4& viewProj,
                      const m3::Vec3& eye, float pixelsPerRadian,
                      const SelectOptions& opt) {
    Selection sel;
    if (tree.nodes.empty()) return sel;

    const Frustum frustum = frustumFromViewProjection(viewProj);
    std::priority_queue<Candidate> queue;
    queue.push({0, projectedPixels(tree.nodes[0].bounds, eye, pixelsPerRadian)});

    while (!queue.empty()) {
        const Candidate cand = queue.top();
        queue.pop();
        ++sel.nodesVisited;

        const Node& n = tree.nodes[cand.node];
        if (!frustum.intersects(n.bounds)) { ++sel.nodesCulled; continue; }

        if (sel.points + n.pointCount > opt.pointBudget) {
            // Everything still queued is smaller on screen than this, so the
            // cut stops here rather than skipping ahead to lesser detail.
            sel.budgetExhausted = true;
            break;
        }
        sel.nodes.push_back(cand.node);
        sel.points += n.pointCount;

        // Refine only while the node is big enough on screen for its children
        // to resolve into something distinguishable.
        if (cand.pixels < opt.minNodePixels) continue;

        for (int o = 0; o < 8; ++o) {
            const uint32_t child = n.child[o];
            if (child == 0) continue;
            if (child >= tree.nodes.size()) continue;   // malformed: skip rather than read past
            queue.push({child, projectedPixels(tree.nodes[child].bounds, eye, pixelsPerRadian)});
        }
    }
    return sel;
}

} // namespace lod
