// Out-of-core level-of-detail octree.
//
// The scale this exists for: a job of a few thousand setups is 10^10-10^11
// points. Nothing about that fits in memory, and no point budget applied
// per-scan makes it fit either — decimating 2000 scans to 4 M points each is
// still 8 × 10^9 points, and decimating hard enough to fit destroys exactly the
// detail the tool is for. The only way out is an octree built once over the
// whole corpus, stored on disk, and streamed by what the camera can actually
// see.
//
// The tree is *additive*, the scheme Potree uses: a node keeps a point only if
// no point already kept in that node occupies the same cell of the node's
// occupancy grid, otherwise the point is pushed to a child. So the root is a
// uniform coarse sample of the entire site, each level adds detail at roughly
// half the spacing, and rendering a set of nodes means drawing their union with
// no duplication. Selecting a cut through the tree by projected size therefore
// gives near-uniform screen-space point density regardless of how much of the
// site is in view.
//
// Kept free of Metal and AppKit: the build and the selection are the whole
// engine, and both are testable off the target platform.

#pragma once

#include "math3d.h"

#include <cstdint>
#include <string>
#include <vector>

namespace lod {

// A point as it lives in the store: coordinates are float offsets from the
// store's double-precision origin (see point_store.h), which keeps georeferenced
// sites off float32's precision cliff.
#pragma pack(push, 1)
struct StorePoint {
    float    x, y, z;
    uint8_t  r, g, b, a;
    uint16_t scanId;      // index into the store's scan table
    uint16_t pad;         // keeps the record 20 bytes and 4-byte aligned
};
#pragma pack(pop)
static_assert(sizeof(StorePoint) == 20, "StorePoint must stay 20 bytes on disk");

struct Aabb {
    float lo[3] = {0, 0, 0};
    float hi[3] = {0, 0, 0};
    void   expand(float x, float y, float z);
    bool   valid() const { return hi[0] >= lo[0] && hi[1] >= lo[1] && hi[2] >= lo[2]; }
    float  diagonal() const;
    void   centre(float out[3]) const;
    // The octant sub-box, in the standard x + 2y + 4z ordering.
    Aabb   child(int octant) const;
};

// Makes the root a cube: an octree over a non-cubic box degenerates into
// slabs, and node "size" stops meaning anything comparable across the tree.
Aabb cubeAround(const Aabb& box);

struct Node {
    uint8_t  level      = 0;
    uint32_t pointCount = 0;
    uint64_t payloadOffset = 0;   // byte offset of this node's points in the store
    Aabb     bounds;
    // Explicit child indices rather than a base-plus-mask. A bounded-memory
    // build stitches independently built subtrees into the top of the tree,
    // and any "children are contiguous" invariant has to be repaired at every
    // join. 32 bytes a node buys that away. Index 0 is the root, so it doubles
    // as the "no child" sentinel.
    uint32_t child[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    bool hasChild(int octant) const { return child[octant] != 0; }
    bool isLeaf() const {
        for (int i = 0; i < 8; ++i) if (child[i] != 0) return false;
        return true;
    }
};

struct Tree {
    std::vector<Node> nodes;      // node 0 is the root
    Aabb              bounds;     // cubic root bounds
    uint64_t          totalPoints = 0;

    bool empty() const { return nodes.empty(); }
};

// ---------------------------------------------------------------------------
// Building

struct BuildOptions {
    // Occupancy grid per node. 64 gives a node at most 64^3 cells, but real
    // fill is far lower because a node only covers the surfaces passing
    // through it. Higher means denser nodes and a shallower tree.
    uint32_t gridResolution = 64;
    // Hard cap so one pathological node cannot become the whole tree.
    uint32_t maxPointsPerNode = 120000;
    // Depth limit. 14 levels over a 200 m site puts the finest node at ~12 mm,
    // well past what a 5 cm voxel analysis or a screen pixel can resolve.
    uint8_t  maxLevel = 14;
};

// Builds a subtree in memory. Used both for the top levels during the
// streaming pass and for each on-disk chunk afterwards; see indexer.h for how
// the two are combined into a bounded-memory build over a whole corpus.
class Builder {
public:
    Builder(const Aabb& bounds, const BuildOptions& opt, uint8_t baseLevel = 0);

    // Returns false when the point falls outside the root bounds.
    bool insert(const StorePoint& p);

    // Points that reached the depth limit and were pushed no further. During
    // the streaming pass these are the ones to spill to a chunk file.
    void setOverflowSink(std::vector<StorePoint>* sink) { overflow_ = sink; }

    // Finalises and hands back the tree plus each node's points, indexed by
    // node. Payload offsets are left zero; the store writer assigns them.
    Tree finish(std::vector<std::vector<StorePoint>>& outPoints);

    size_t insertedCount() const { return inserted_; }
    size_t overflowCount() const { return overflowed_; }

private:
    struct Cell {
        Node                     node;
        std::vector<StorePoint>  points;
        std::vector<uint64_t>    occupancy;   // bitset over gridResolution^3
        int32_t                  child[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
    };

    int32_t childOf(int32_t cellIndex, int octant);
    bool    claimCell(Cell& c, const StorePoint& p);

    std::vector<Cell> cells_;
    BuildOptions      opt_;
    uint8_t           baseLevel_ = 0;
    size_t            inserted_ = 0;
    size_t            overflowed_ = 0;
    std::vector<StorePoint>* overflow_ = nullptr;
};

// ---------------------------------------------------------------------------
// Selection
//
// Chooses which nodes to draw for a given view. This is what makes thousands
// of setups navigable: the cut adapts to the camera, and the point budget is
// a hard ceiling regardless of how much data the store holds.

struct SelectOptions {
    // Points drawn per frame. 12 M is comfortable on an M4 Max at 60 Hz with
    // round point sprites; the cost is per-point, not per-scan, so this bound
    // holds whether the store came from 5 setups or 5000.
    size_t pointBudget = 12000000;
    // A node is only worth refining once it projects to at least this many
    // pixels across. Below it the node's own points already exceed screen
    // resolution and its children would add nothing visible.
    float  minNodePixels = 120.0f;
};

struct Selection {
    std::vector<uint32_t> nodes;          // indices into Tree::nodes
    size_t                points = 0;
    bool   budgetExhausted = false;       // true when detail was cut short
    size_t nodesCulled = 0;               // outside the frustum
    size_t nodesVisited = 0;
};

// Six frustum planes (left, right, bottom, top, near, far) as ax+by+cz+d.
struct Frustum {
    float p[6][4];
    bool  intersects(const Aabb& box) const;
};
Frustum frustumFromViewProjection(const m3::Mat4& viewProj);

// Front-to-back by projected size, so the budget is spent on what is largest
// on screen. `pixelsPerRadian` converts an angular size to pixels; pass
// viewportHeight / fovY.
Selection selectNodes(const Tree& tree, const m3::Mat4& viewProj,
                      const m3::Vec3& eye, float pixelsPerRadian,
                      const SelectOptions& opt = SelectOptions{});

} // namespace lod
