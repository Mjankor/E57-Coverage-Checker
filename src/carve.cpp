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
                    if (distSqPointBox(s.origin, lo, hi) > R2) continue;
                    // And a tile the domain excludes is not asked about at all.
                    if (p.domain.testBox(lo, hi) == Overlap::None) continue;
                    keys.push_back(k);
                }
            }
        }
    }

    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    return keys;
}

namespace {

std::vector<size_t> setupsForBox(const double lo[3], const double hi[3],
                                 const std::vector<SetupView>& setups, const Params& p) {
    std::vector<size_t> out;
    const double R2 = p.maxRange * p.maxRange;
    for (size_t i = 0; i < setups.size(); ++i) {
        if (!setups[i].image) continue;
        if (distSqPointBox(setups[i].origin, lo, hi) <= R2) out.push_back(i);
    }
    return out;
}

} // namespace

std::vector<size_t> setupsForTile(const TileKey& key, const std::vector<SetupView>& setups,
                                  const Params& p) {
    double lo[3], hi[3];
    tileBounds(key, p, lo, hi);
    return setupsForBox(lo, hi, setups, p);
}

namespace {

// Shared by both implementations: shape the tile, zero it, and pick the setups
// that can reach it including its apron.
std::vector<size_t> prepareTile(const TileKey& key, const std::vector<SetupView>& setups,
                                const Params& p, Tile& out) {
    const uint32_t core = p.tileVoxels;
    const uint32_t ap   = p.apron;
    const uint32_t dim  = core + 2 * ap;
    out.key   = key;
    out.dim   = dim;
    out.core  = core;
    out.apron = ap;

    double lo[3], hi[3];
    tileBounds(key, p, lo, hi);
    const double pad = double(ap) * p.voxelSize;
    for (int k = 0; k < 3; ++k) out.origin[k] = lo[k] - pad;
    out.state.assign(size_t(dim) * dim * dim, 0);
    if (dim == 0) return {};

    // The apron reaches outside the tile, so setup selection has to as well —
    // otherwise an apron voxel would be judged against the wrong setup list and
    // the neighbour answers at the seam would be wrong.
    double plo[3], phi[3];
    for (int k = 0; k < 3; ++k) { plo[k] = lo[k] - pad; phi[k] = hi[k] + pad; }
    return setupsForBox(plo, phi, setups, p);
}

} // namespace

// Split out because the fast path fills the state in a different order from the
// reference and cannot count as it goes — and because the GPU path fills it
// somewhere else entirely and still has to count it identically.
void tallyTile(const Tile& t, Stats& stats) {
    for (uint32_t z = t.interiorBegin(); z < t.interiorEnd(); ++z) {
        for (uint32_t y = t.interiorBegin(); y < t.interiorEnd(); ++y) {
            for (uint32_t x = t.interiorBegin(); x < t.interiorEnd(); ++x) {
                const uint8_t bits = t.state[t.index(x, y, z)];
                ++stats.voxels;
                if (bits & kReachable) {
                    ++stats.reachable;
                    if (bits == kReachable) ++stats.unknown;
                }
                if (bits & kVisible)  ++stats.visible;
                if (bits & kOccupied) ++stats.occupied;
            }
        }
    }
}

namespace {

// What one pyramid query says about a whole brick.
enum class BrickVerdict {
    Fallthrough,   // mixed: test each voxel
    AllVisible,    // every voxel is in clear line of sight from this setup
    NoEvidence,    // every voxel is behind every surface: reachable, nothing seen
};

// Bounds the directions a brick occupies, seen from the setup, using its
// bounding sphere. The cone of half-angle alpha around the brick's centre
// direction is contained in a latitude/longitude box, which is what the raster
// is indexed by.
//
// Deliberately loose. Every widening of this box weakens the two tests below
// and costs speed; none of them can make an answer wrong, because a wider
// region can only report a wider range of surfaces.
struct AngularBox {
    double elLo = 0, elHi = 0;
    double azLo = 0, azHi = 0;
    double rMin = 0, rMax = 0;
    bool   valid = false;
    bool   rowsInside = false;   // the whole box is within the raster's rows
};

AngularBox boundBrick(const SetupView& s, const double blo[3], const double bhi[3]) {
    AngularBox b;

    // Centre and bounding-sphere radius, in the scanner's frame.
    double cx = 0.5 * (blo[0] + bhi[0]);
    double cy = 0.5 * (blo[1] + bhi[1]);
    double cz = 0.5 * (blo[2] + bhi[2]);
    const double hx = 0.5 * (bhi[0] - blo[0]);
    const double hy = 0.5 * (bhi[1] - blo[1]);
    const double hz = 0.5 * (bhi[2] - blo[2]);
    const double rho = std::sqrt(hx * hx + hy * hy + hz * hz);

    s.worldToScanner.apply(cx, cy, cz);
    const double d = std::sqrt(cx * cx + cy * cy + cz * cz);
    // The setup inside or on the brick: every direction is possible.
    if (d <= rho + 1e-9) return b;

    b.rMin = d - rho;
    b.rMax = d + rho;

    const double alpha = std::asin(std::clamp(rho / d, -1.0, 1.0));
    const double elC   = std::asin(std::clamp(cz / d, -1.0, 1.0));
    b.elLo = elC - alpha;
    b.elHi = elC + alpha;

    // A cone reaching a pole has no bounded azimuth: every bearing is inside it.
    constexpr double kHalfPi = 1.5707963267948966;
    if (b.elLo <= -kHalfPi + 1e-6 || b.elHi >= kHalfPi - 1e-6) return b;

    // On a small circle at elevation e, an angular radius alpha spans an
    // azimuth half-width of asin(sin alpha / cos e). Taking the extreme
    // elevation of the box rather than its centre widens it, which is the safe
    // direction.
    const double elAbs = std::max(std::fabs(b.elLo), std::fabs(b.elHi));
    const double denom = std::cos(elAbs);
    if (denom <= 1e-9) return b;
    const double sinHalf = std::sin(alpha) / denom;
    if (sinHalf >= 1.0) return b;

    const double azC = std::atan2(cy, cx);
    const double half = std::asin(sinHalf);
    b.azLo = azC - half;
    b.azHi = azC + half;
    b.valid = true;
    return b;
}

BrickVerdict judgeBrick(const SetupView& s, const Params& p,
                        const double blo[3], const double bhi[3], bool allInRange) {
    const rimg::RangeImage& im = *s.image;
    if (im.pyramid.empty()) return BrickVerdict::Fallthrough;

    const AngularBox b = boundBrick(s, blo, bhi);
    if (!b.valid) return BrickVerdict::Fallthrough;

    // dElPerRow and dAzPerCol can be negative, so the raster coordinates of the
    // two ends are not ordered.
    const double ra = im.rowCoord(b.elLo), rb = im.rowCoord(b.elHi);
    const double ca = im.colCoord(b.azLo), cb = im.colCoord(b.azHi);
    // One cell of slack on every side, because cellOf rounds to the nearest
    // cell rather than truncating: a direction just outside the box can still
    // land on the neighbouring cell.
    const int64_t row0 = int64_t(std::floor(std::min(ra, rb))) - 1;
    const int64_t row1 = int64_t(std::ceil(std::max(ra, rb))) + 1;
    const int64_t col0 = int64_t(std::floor(std::min(ca, cb))) - 1;
    const int64_t col1 = int64_t(std::ceil(std::max(ca, cb))) + 1;

    const rimg::RangeSpan span = im.span(row0, row1, col0, col1);
    if (!span.valid) return BrickVerdict::Fallthrough;

    // Every voxel is further than every surface in the region, so no cell can
    // report anything: a hit is behind it, a no-return has stopped clearing, and
    // a direction outside the raster says nothing either. Voxels are still in
    // the domain — they just have no evidence — so this skips the range image,
    // not the voxels.
    if (b.rMin > span.maxRange + p.surfaceMargin) return BrickVerdict::NoEvidence;

    // Every voxel is nearer than every surface, so every one is in clear line of
    // sight. This needs the whole brick inside the raster and inside the rated
    // range: a direction the scanner never sampled proves nothing, and neither
    // does one past where it can measure.
    if (allInRange && b.rMax <= p.maxRange &&
        b.rMax < span.minRange - p.surfaceMargin &&
        !(span.statuses & rimg::kHasOutsideFov)) {
        const double rowLo = std::min(ra, rb) - 1.0, rowHi = std::max(ra, rb) + 1.0;
        if (rowLo >= 0.0 && rowHi <= double(im.rows) - 1.0) return BrickVerdict::AllVisible;
    }

    return BrickVerdict::Fallthrough;
}

} // namespace

// The fast path. Same arithmetic as the reference, reordered for the machine:
// one setup at a time so a single range image is resident, and within that in
// bricks, so the patch of image a brick projects onto stays in cache while all
// 512 of its voxels are tested against it.
void carveTile(const TileKey& key, const std::vector<SetupView>& setups,
               const Params& p, Tile& out, Stats& stats) {
    const std::vector<size_t> reach = prepareTile(key, setups, p, out);
    const uint32_t dim = out.dim;
    if (dim == 0) return;

    const double R2 = p.maxRange * p.maxRange;
    const uint32_t B  = kBrickVoxels;
    const uint32_t nb = (dim + B - 1) / B;

    // Nearest setup first. Under Saturated this only changes how quickly voxels
    // settle, never what they settle on; under AnyEvidence it decides which of
    // several true answers is recorded, so the order has to be fixed rather than
    // incidental — hence the tie-break on index.
    std::vector<size_t> order = reach;
    if (p.earlyOut != EarlyOut::None && order.size() > 1) {
        double c[3];
        for (int k = 0; k < 3; ++k)
            c[k] = out.origin[k] + 0.5 * double(dim) * p.voxelSize;
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            const double da = distSqPointBox(setups[a].origin, c, c);
            const double db = distSqPointBox(setups[b].origin, c, c);
            if (da != db) return da < db;
            return a < b;
        });
    }

    const uint8_t kEvidence = uint8_t(kVisible | kOccupied);
    const uint8_t settledMask = (p.earlyOut == EarlyOut::Saturated)  ? kEvidence
                              : (p.earlyOut == EarlyOut::AnyEvidence) ? 0u : 0xFFu;
    auto settled = [&](uint8_t bits) {
        switch (p.earlyOut) {
        case EarlyOut::None:        return false;
        case EarlyOut::Saturated:   return (bits & kEvidence) == kEvidence;
        case EarlyOut::AnyEvidence: return (bits & kEvidence) != 0;
        }
        return false;
    };
    (void)settledMask;

    for (size_t si : order) {
        const SetupView& s = setups[si];
        for (uint32_t bz = 0; bz < nb; ++bz) {
            for (uint32_t by = 0; by < nb; ++by) {
                for (uint32_t bx = 0; bx < nb; ++bx) {
                    const uint32_t x0 = bx * B, x1 = std::min(x0 + B, dim);
                    const uint32_t y0 = by * B, y1 = std::min(y0 + B, dim);
                    const uint32_t z0 = bz * B, z1 = std::min(z0 + B, dim);

                    // One range test for the whole brick. The box spans the
                    // voxels rather than their centres, so it is a superset of
                    // what the per-voxel test would accept: conservative, and
                    // therefore incapable of dropping a voxel the reference
                    // would have kept.
                    const double blo[3] = {out.origin[0] + x0 * p.voxelSize,
                                           out.origin[1] + y0 * p.voxelSize,
                                           out.origin[2] + z0 * p.voxelSize};
                    const double bhi[3] = {out.origin[0] + x1 * p.voxelSize,
                                           out.origin[1] + y1 * p.voxelSize,
                                           out.origin[2] + z1 * p.voxelSize};
                    if (distSqPointBox(s.origin, blo, bhi) > R2) continue;

                    // Out of the domain is out of the question: not unknown,
                    // not counted, not carved.
                    const Overlap ov = p.domain.testBox(blo, bhi);
                    if (ov == Overlap::None) continue;
                    const bool allInDomain = (ov == Overlap::Full);

                    // Whether every voxel in the brick is inside the rated
                    // range, which decides whether the per-voxel distance test
                    // can be skipped as well.
                    double far = 0;
                    for (int k = 0; k < 3; ++k) {
                        const double a = std::fabs(blo[k] - s.origin[k]);
                        const double b = std::fabs(bhi[k] - s.origin[k]);
                        const double m = std::max(a, b);
                        far += m * m;
                    }
                    const bool allInRange = far <= R2;

                    const BrickVerdict verdict = judgeBrick(s, p, blo, bhi, allInRange);

                    if (verdict != BrickVerdict::Fallthrough) {
                        // One pyramid lookup settled the whole brick. All that
                        // is left is to write the bits and keep the tally
                        // honest — the reference counts one test per voxel per
                        // setup in range, and so does this.
                        const uint8_t set = (verdict == BrickVerdict::AllVisible)
                                          ? uint8_t(kReachable | kVisible)
                                          : uint8_t(kReachable);
                        for (uint32_t z = z0; z < z1; ++z) {
                            for (uint32_t y = y0; y < y1; ++y) {
                                for (uint32_t x = x0; x < x1; ++x) {
                                    if (!allInRange || !allInDomain) {
                                        double c[3];
                                        out.centre(x, y, z, p.voxelSize, c);
                                        if (!allInDomain &&
                                            !p.domain.contains(c[0], c[1], c[2])) continue;
                                        const double dx = c[0] - s.origin[0];
                                        const double dy = c[1] - s.origin[1];
                                        const double dz = c[2] - s.origin[2];
                                        if (dx * dx + dy * dy + dz * dz > R2) continue;
                                    }
                                    out.state[out.index(x, y, z)] |= set;
                                    if (out.isInterior(x, y, z)) ++stats.setupTests;
                                }
                            }
                        }
                        continue;
                    }

                    for (uint32_t z = z0; z < z1; ++z) {
                        for (uint32_t y = y0; y < y1; ++y) {
                            for (uint32_t x = x0; x < x1; ++x) {
                                uint8_t& cell = out.state[out.index(x, y, z)];
                                // Nothing another setup can add. This is the
                                // whole of the win at high setup counts.
                                if (settled(cell)) continue;
                                double c[3];
                                out.centre(x, y, z, p.voxelSize, c);
                                if (!allInDomain &&
                                    !p.domain.contains(c[0], c[1], c[2])) continue;
                                const double dx = c[0] - s.origin[0];
                                const double dy = c[1] - s.origin[1];
                                const double dz = c[2] - s.origin[2];
                                if (dx * dx + dy * dy + dz * dz > R2) continue;
                                // OR across setups, so the order they are
                                // combined in cannot change the result — which
                                // is exactly what makes this reordering legal.
                                cell |= uint8_t(kReachable | evidenceAt(s, p, c[0], c[1], c[2]));
                                if (out.isInterior(x, y, z)) ++stats.setupTests;
                            }
                        }
                    }
                }
            }
        }
    }

    tallyTile(out, stats);
}

void carveTileReference(const TileKey& key, const std::vector<SetupView>& setups,
                        const Params& p, Tile& out, Stats& stats) {
    const std::vector<size_t> reach = prepareTile(key, setups, p, out);
    const uint32_t dim = out.dim;
    if (dim == 0) return;
    const double R2 = p.maxRange * p.maxRange;

    for (uint32_t z = 0; z < dim; ++z) {
        for (uint32_t y = 0; y < dim; ++y) {
            for (uint32_t x = 0; x < dim; ++x) {
                double c[3];
                out.centre(x, y, z, p.voxelSize, c);
                if (!p.domain.contains(c[0], c[1], c[2])) continue;

                uint8_t  bits = 0;
                uint64_t tests = 0;
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
                    bits |= kReachable;
                    ++tests;
                    bits |= evidenceAt(s, p, c[0], c[1], c[2]);
                }

                out.state[out.index(x, y, z)] = bits;
                // Apron voxels belong to the neighbouring tile; counting them
                // here would tally them twice.
                if (out.isInterior(x, y, z)) stats.setupTests += tests;
            }
        }
    }

    tallyTile(out, stats);
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
