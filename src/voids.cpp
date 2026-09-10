#include "voids.h"

#include <vector>

namespace voids {

namespace {

// A voxel the scanners observed. These are what the flood cannot cross, and so
// what makes a void a void.
inline bool observed(uint8_t bits) {
    return (bits & (carve::kVisible | carve::kOccupied)) != 0;
}

// One run of unobserved voxels along x, waiting to have its y and z neighbours
// examined. The flood is span-based rather than voxel-based because the regions
// involved are enormous and mostly open: a per-voxel stack over the exterior of
// a street scene is hundreds of millions of entries, where the spans covering
// the same volume number in the thousands.
struct Span {
    uint32_t z, y, x0, x1;   // inclusive
};

} // namespace

uint8_t observedFaces(const Grid& g, uint32_t x, uint32_t y, uint32_t z) {
    // Same order as vis::kFaceDirs, because the shading reads these bits as
    // directions and a different order there would light the wrong faces.
    const int32_t d[6][3] = {{1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}};
    uint8_t mask = 0;
    for (int i = 0; i < 6; ++i) {
        const int64_t nx = int64_t(x) + d[i][0];
        const int64_t ny = int64_t(y) + d[i][1];
        const int64_t nz = int64_t(z) + d[i][2];
        if (nx < 0 || ny < 0 || nz < 0 ||
            nx >= int64_t(g.dim[0]) || ny >= int64_t(g.dim[1]) || nz >= int64_t(g.dim[2]))
            continue;
        if (observed(g.state[g.index(uint32_t(nx), uint32_t(ny), uint32_t(nz))]))
            mask |= uint8_t(1u << i);
    }
    return mask;
}

bool touchesObserved(const Grid& g, uint32_t x, uint32_t y, uint32_t z) {
    return observedFaces(g, x, y, z) != 0;
}

Report classify(Grid& g) {
    Report rep;
    if (g.dim[0] == 0 || g.dim[1] == 0 || g.dim[2] == 0) return rep;
    if (g.state.size() != size_t(g.voxelCount())) return rep;

    const uint32_t X = g.dim[0], Y = g.dim[1], Z = g.dim[2];

    std::vector<Span> stack;

    // Fills the maximal unobserved run containing (x, y, z), marks it exterior,
    // and returns it. Returns an empty span when the seed is not fillable.
    auto fillRun = [&](uint32_t x, uint32_t y, uint32_t z, Span& out) -> bool {
        const size_t base = (size_t(z) * Y + y) * X;
        uint8_t* row = &g.state[base];
        if (observed(row[x]) || (row[x] & kExterior)) return false;
        uint32_t x0 = x, x1 = x;
        while (x0 > 0 && !observed(row[x0 - 1]) && !(row[x0 - 1] & kExterior)) --x0;
        while (x1 + 1 < X && !observed(row[x1 + 1]) && !(row[x1 + 1] & kExterior)) ++x1;
        for (uint32_t i = x0; i <= x1; ++i) row[i] |= kExterior;
        out = Span{z, y, x0, x1};
        return true;
    };

    auto seed = [&](uint32_t x, uint32_t y, uint32_t z) {
        Span s;
        if (fillRun(x, y, z, s)) stack.push_back(s);
    };

    // The flood starts at the grid's own boundary: every face of the carved
    // region borders space this run knows nothing about, which is where the rest
    // of the world is.
    for (uint32_t z = 0; z < Z; ++z)
        for (uint32_t y = 0; y < Y; ++y) {
            const bool edgeSlab = (z == 0 || z == Z - 1 || y == 0 || y == Y - 1);
            if (edgeSlab) {
                for (uint32_t x = 0; x < X; ++x) seed(x, y, z);
            } else {
                seed(0, y, z);
                seed(X - 1, y, z);
            }
        }

    // Spread. For each finished span, walk its four neighbouring rows and seed a
    // fill at the start of every unobserved run they contain.
    while (!stack.empty()) {
        const Span s = stack.back();
        stack.pop_back();

        const int32_t steps[4][2] = {{0, -1}, {0, 1}, {-1, 0}, {1, 0}};   // dz, dy
        for (const auto& st : steps) {
            const int64_t nz = int64_t(s.z) + st[0];
            const int64_t ny = int64_t(s.y) + st[1];
            if (nz < 0 || ny < 0 || nz >= int64_t(Z) || ny >= int64_t(Y)) continue;
            const size_t base = (size_t(nz) * Y + size_t(ny)) * X;
            const uint8_t* row = &g.state[base];
            for (uint32_t x = s.x0; x <= s.x1; ++x) {
                if (observed(row[x]) || (row[x] & kExterior)) continue;
                Span filled;
                if (fillRun(x, uint32_t(ny), uint32_t(nz), filled)) {
                    // Skip past what that fill just claimed, so a long open run
                    // is seeded once rather than once per column.
                    x = filled.x1;
                    stack.push_back(filled);
                }
            }
        }
    }

    // Count, and label the enclosed voids so their number and size can be
    // reported. A run of a hundred one-voxel pockets is a different finding from
    // one room-sized hole, and the totals alone cannot tell them apart.
    std::vector<uint8_t> seen(g.state.size(), 0);
    std::vector<uint32_t> queue;
    for (uint32_t z = 0; z < Z; ++z) {
        for (uint32_t y = 0; y < Y; ++y) {
            for (uint32_t x = 0; x < X; ++x) {
                const size_t i = g.index(x, y, z);
                const uint8_t bits = g.state[i];
                if (observed(bits)) continue;
                if (bits & kExterior) {
                    if (bits & carve::kReachable) ++rep.exterior;
                    continue;
                }
                if (!(bits & carve::kReachable)) continue;   // never in the question
                ++rep.enclosed;
                if (touchesObserved(g, x, y, z)) ++rep.enclosedSurface;

                if (seen[i]) continue;
                // A new component. Breadth-first over the six neighbours, which
                // is affordable here: enclosed voids are small by definition —
                // anything large enough to worry about the queue would have
                // found a way out to the boundary.
                ++rep.components;
                uint64_t size = 0;
                queue.clear();
                queue.push_back(uint32_t(i));
                seen[i] = 1;
                while (!queue.empty()) {
                    const uint32_t cur = queue.back();
                    queue.pop_back();
                    ++size;
                    const uint32_t cx = cur % X;
                    const uint32_t cy = (cur / X) % Y;
                    const uint32_t cz = cur / (X * Y);
                    const int32_t d[6][3] = {{1,0,0}, {-1,0,0}, {0,1,0},
                                             {0,-1,0}, {0,0,1}, {0,0,-1}};
                    for (const auto& o : d) {
                        const int64_t nx = int64_t(cx) + o[0];
                        const int64_t ny = int64_t(cy) + o[1];
                        const int64_t nz = int64_t(cz) + o[2];
                        if (nx < 0 || ny < 0 || nz < 0 || nx >= int64_t(X) ||
                            ny >= int64_t(Y) || nz >= int64_t(Z)) continue;
                        const size_t n = g.index(uint32_t(nx), uint32_t(ny), uint32_t(nz));
                        if (seen[n]) continue;
                        const uint8_t nb = g.state[n];
                        if (observed(nb) || (nb & kExterior) || !(nb & carve::kReachable))
                            continue;
                        seen[n] = 1;
                        queue.push_back(uint32_t(n));
                    }
                }
                if (size > rep.largestComponent) rep.largestComponent = size;
            }
        }
    }
    return rep;
}

} // namespace voids
