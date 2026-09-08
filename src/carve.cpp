#include "carve.h"

#include <algorithm>
#include <cmath>

namespace carve {

namespace {

// Distance from a point to the closest point of an axis-aligned box, squared.
// Used for sphere/tile overlap: a tile is in a setup's reach exactly when this
// is within maxRange^2 of the setup.
double distSqPointBox(const double p[3], const double lo[3], const double hi[3]) {
    double d = 0;
    for (int i = 0; i < 3; ++i) {
        const double v = (p[i] < lo[i]) ? (lo[i] - p[i])
                       : (p[i] > hi[i]) ? (p[i] - hi[i])
                                        : 0.0;
        d += v * v;
    }
    return d;
}

void tileBounds(const TileKey& k, const Params& p, double lo[3], double hi[3]) {
    const double T = p.tileMetres();
    lo[0] = double(k.x) * T; hi[0] = lo[0] + T;
    lo[1] = double(k.y) * T; hi[1] = lo[1] + T;
    lo[2] = double(k.z) * T; hi[2] = lo[2] + T;
}

int64_t tileIndexOf(double v, double tileMetres) {
    return int64_t(std::floor(v / tileMetres));
}

} // namespace

SetupView makeSetupView(const rimg::RangeImage& image) {
    SetupView s;
    s.image = &image;

    // The file gives scanner-to-world; the carve needs the other direction.
    // For a rigid transform that is R^T with translation -R^T t, and building
    // it by transposition rather than a general inverse keeps it exact.
    viewer::Rigid fwd;
    if (image.hasPose) fwd = viewer::rigidFromPose(image.pose);

    viewer::Rigid inv;
    inv.R[0] = fwd.R[0]; inv.R[1] = fwd.R[3]; inv.R[2] = fwd.R[6];
    inv.R[3] = fwd.R[1]; inv.R[4] = fwd.R[4]; inv.R[5] = fwd.R[7];
    inv.R[6] = fwd.R[2]; inv.R[7] = fwd.R[5]; inv.R[8] = fwd.R[8];
    for (int i = 0; i < 3; ++i) {
        inv.t[i] = -(inv.R[3 * i + 0] * fwd.t[0] +
                     inv.R[3 * i + 1] * fwd.t[1] +
                     inv.R[3 * i + 2] * fwd.t[2]);
    }
    s.worldToScanner = inv;
    s.origin[0] = fwd.t[0];
    s.origin[1] = fwd.t[1];
    s.origin[2] = fwd.t[2];
    return s;
}

uint8_t evidenceAt(const SetupView& s, const Params& p,
                   double wx, double wy, double wz) {
    if (!s.image) return 0;

    double x = wx, y = wy, z = wz;
    s.worldToScanner.apply(x, y, z);

    double az, el, r;
    rimg::toSpherical(x, y, z, az, el, r);

    // Beyond the instrument's rated maximum this setup establishes nothing —
    // not emptiness, not surface. Also covers the voxel the scanner sits in,
    // where the direction is meaningless.
    if (r > p.maxRange || r < 1e-9) return 0;

    rimg::Status st;
    double surface;
    if (!s.image->sample(az, el, st, surface)) return 0;   // outside the raster

    switch (st) {
    case rimg::Status::OutsideFov:
        // The scanner never looked here. Silence, not emptiness — this is the
        // distinction DESIGN.md §4 turns on.
        return 0;

    case rimg::Status::NoReturn:
        // The ray was fired and came back empty, so everything along it out to
        // the clearing distance was seen through. That distance is whichever is
        // nearer of this run's setting and what the image was built with.
        return (r <= std::min(surface, p.maxRange)) ? kVisible : 0;

    case rimg::Status::Hit:
        // In front of the measured surface: line of sight. Straddling it: the
        // surface is inside this voxel. Behind it: occluded, and this setup
        // says nothing at all.
        if (r < surface - p.surfaceMargin) return kVisible;
        if (r <= surface + p.surfaceMargin) return kOccupied;
        return 0;
    }
    return 0;
}

std::vector<TileKey> tilesForSetups(const std::vector<SetupView>& setups, const Params& p) {
    std::vector<TileKey> keys;
    if (p.tileVoxels == 0 || p.voxelSize <= 0) return keys;

    const double T  = p.tileMetres();
    const double R  = p.maxRange;
    const double R2 = R * R;

    for (const SetupView& s : setups) {
        if (!s.image) continue;
        const int64_t x0 = tileIndexOf(s.origin[0] - R, T), x1 = tileIndexOf(s.origin[0] + R, T);
        const int64_t y0 = tileIndexOf(s.origin[1] - R, T), y1 = tileIndexOf(s.origin[1] + R, T);
        const int64_t z0 = tileIndexOf(s.origin[2] - R, T), z1 = tileIndexOf(s.origin[2] + R, T);
        for (int64_t z = z0; z <= z1; ++z) {
            for (int64_t y = y0; y <= y1; ++y) {
                for (int64_t x = x0; x <= x1; ++x) {
                    const TileKey k{x, y, z};
                    double lo[3], hi[3];
                    tileBounds(k, p, lo, hi);
                    // The bounding box of a sphere is not the sphere: the
                    // corner tiles it names can be entirely out of reach.
                    if (distSqPointBox(s.origin, lo, hi) <= R2) keys.push_back(k);
                }
            }
        }
    }

    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    return keys;
}

std::vector<size_t> setupsForTile(const TileKey& key, const std::vector<SetupView>& setups,
                                  const Params& p) {
    std::vector<size_t> out;
    double lo[3], hi[3];
    tileBounds(key, p, lo, hi);
    const double R2 = p.maxRange * p.maxRange;
    for (size_t i = 0; i < setups.size(); ++i) {
        if (!setups[i].image) continue;
        if (distSqPointBox(setups[i].origin, lo, hi) <= R2) out.push_back(i);
    }
    return out;
}

void carveTile(const TileKey& key, const std::vector<SetupView>& setups,
               const Params& p, Tile& out, Stats& stats) {
    const uint32_t dim = p.tileVoxels;
    out.key = key;
    out.dim = dim;
    double lo[3], hi[3];
    tileBounds(key, p, lo, hi);
    out.origin[0] = lo[0]; out.origin[1] = lo[1]; out.origin[2] = lo[2];
    out.state.assign(size_t(dim) * dim * dim, 0);
    if (dim == 0) return;

    const std::vector<size_t> reach = setupsForTile(key, setups, p);
    const double R2 = p.maxRange * p.maxRange;

    for (uint32_t z = 0; z < dim; ++z) {
        for (uint32_t y = 0; y < dim; ++y) {
            for (uint32_t x = 0; x < dim; ++x) {
                double c[3];
                out.centre(x, y, z, p.voxelSize, c);

                uint8_t bits = 0;
                bool inDomain = false;
                for (size_t si : reach) {
                    const SetupView& s = setups[si];
                    const double dx = c[0] - s.origin[0];
                    const double dy = c[1] - s.origin[1];
                    const double dz = c[2] - s.origin[2];
                    // A tile can reach a setup while most of its voxels do not.
                    // Skipping those is the domain definition, not an
                    // optimisation: outside the range sphere there is nothing
                    // to say.
                    if (dx * dx + dy * dy + dz * dz > R2) continue;
                    inDomain = true;
                    ++stats.setupTests;
                    // OR across setups: visibility from any one of them is
                    // visibility, and the order they are combined in cannot
                    // change the result.
                    bits |= evidenceAt(s, p, c[0], c[1], c[2]);
                }

                out.state[out.index(x, y, z)] = bits;
                ++stats.voxels;
                if (inDomain) {
                    ++stats.reachable;
                    if (!bits) ++stats.unknown;
                }
                if (bits & kVisible)  ++stats.visible;
                if (bits & kOccupied) ++stats.occupied;
            }
        }
    }
}

Stats carveAll(const std::vector<SetupView>& setups, const Params& p,
               TileSink sink, void* user) {
    Stats stats;
    const std::vector<TileKey> keys = tilesForSetups(setups, p);
    Tile tile;
    for (const TileKey& k : keys) {
        carveTile(k, setups, p, tile, stats);
        if (sink && !sink(tile, user)) break;
    }
    return stats;
}

} // namespace carve
