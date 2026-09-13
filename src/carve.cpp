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

namespace {

// What one measured ray says about a voxel at distance `r` along it.
uint8_t verdictFor(rimg::Status st, double surface, double r, const Params& p) {
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

// The rings of a voxel's angular footprint, in units of its half-extent: the
// centre ray, then eight rays half way out, then eight grazing the voxel's
// inscribed sphere. Asked inside out, because the centre ray is the most
// representative of the voxel and the grazing ones the least.
//
// Seventeen rays, and the count is what the dead-cell share on real scans
// requires rather than a round number. A scan that reports a fraction f of its
// directions as unexplained leaves a voxel wrongly unobserved only if EVERY ray
// through it landed on one of them, which is f^17 for independent cells: six in
// a thousand million at the third of the raster APAL__0005 demotes, against
// three in ten for the single ray this replaces. Nine rays would leave two in a
// hundred thousand, which over a hundred million voxels is still thousands.
constexpr int kFootprintRings = 2;
constexpr int kRingOffsets[8][2] = {
    {-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1},
};

} // namespace

uint8_t evidenceAt(const SetupView& s, const Params& p,
                   double wx, double wy, double wz) {
    if (!s.image) return 0;

    double x = wx, y = wy, z = wz;
    s.worldToScanner.apply(x, y, z);
    // Into the instrument's own frame, which is not the stored one when the
    // tripod was not level — and it never quite is. The image's cells were built
    // about this frame, so the direction asked for and the direction stored agree.
    // See RangeImage::tilt.
    s.image->toInstrument(x, y, z);

    double az, el, r;
    rimg::toSpherical(x, y, z, az, el, r);

    // Beyond the instrument's rated maximum this setup establishes nothing —
    // not emptiness, not surface. Also covers the voxel the scanner sits in,
    // where the direction is meaningless.
    if (r > p.maxRange || r < 1e-9) return 0;

    const rimg::RangeImage& im = *s.image;
    uint32_t r0, c0;
    if (!im.cellOf(az, el, r0, c0)) return 0;              // outside the raster

    // The ray through the voxel's centre. Where it decides, it is the answer —
    // this is the whole of what the carve used to ask, and nothing below can
    // overturn a verdict it reaches.
    const uint8_t mid = verdictFor(im.statusAt(r0, c0), im.rangeAt(r0, c0), r, p);
    if (mid) return mid;

    // And where it does not decide, the rest of the rays the scanner fired
    // through this voxel.
    //
    // A voxel is a volume, not a point. At five centimetres it subtends 2.86/r
    // degrees, against raster cells of about 0.14 degrees on the instruments this
    // reads: four hundred rays pass through a voxel at two metres, a hundred at
    // four, and it takes twenty metres before one ray is the whole story. Asking
    // only about the centre threw the other four hundred away — and with a third
    // of a scan's directions unexplained, demoted to OutsideFov, which says
    // nothing at all, a third of the air in front of a wall sampled far more
    // densely than the voxel came back unobserved. Measured on APAL__0005's
    // raster geometry: 29.6% of the space between the setup and a wall at eight
    // metres, at every range from half a metre out. See testAWallSeenPastDeadCells.
    //
    // This does not widen what counts as evidence. Every ray asked about here is
    // one the instrument really fired, and each is asked only whether it passed
    // through THIS voxel, which is what the footprint bounds. The footprint comes
    // from the voxel's INSCRIBED sphere, so a direction inside it passes within
    // half a voxel of the centre and therefore through the voxel itself; the
    // circumscribed sphere would admit rays that only clip a corner.
    if (p.method != Method::VoxelFootprint) return 0;

    const double rho = 0.5 * p.voxelSize;
    if (!(rho < 0.5 * r)) return 0;     // a voxel as near as its own size: no cone

    const rimg::Mapping& m = im.map;
    if (r0 >= m.elByRow.size() || c0 >= m.azByCol.size()) return 0;
    if (m.elByRow.size() < 2 || m.azByCol.size() < 2) return 0;
    // The local step of each measured table. Local rather than averaged because
    // neither axis is uniform — see Mapping.
    const double dEl = std::fabs(r0 + 1 < m.elByRow.size()
                                     ? m.elByRow[r0 + 1] - m.elByRow[r0]
                                     : m.elByRow[r0] - m.elByRow[r0 - 1]);
    const double dAz = std::fabs(c0 + 1 < m.azByCol.size()
                                     ? m.azByCol[c0 + 1] - m.azByCol[c0]
                                     : m.azByCol[c0] - m.azByCol[c0 - 1]);
    if (!(dEl > 0.0) || !(dAz > 0.0)) return 0;

    const double alpha = std::asin(std::clamp(rho / r, -1.0, 1.0));
    // On a small circle at elevation e an angular radius alpha spans an azimuth
    // half-width of asin(sin alpha / cos e) — the same widening boundBrick does.
    // It diverges at the pole, where every bearing is inside the cone; there the
    // elevation rings alone carry the footprint.
    const double ce = std::cos(std::min(std::fabs(el) + alpha, 1.5707963267948966));
    const double sinHalf = ce > 1e-9 ? std::sin(alpha) / ce : 2.0;
    const double azHalf = sinHalf < 1.0 ? std::asin(sinHalf) : 0.0;

    const int64_t hr = int64_t(alpha / dEl);
    const int64_t hc = int64_t(azHalf / dAz);
    if (hr == 0 && hc == 0) return 0;   // the voxel really is one cell wide

    for (int ring = 1; ring <= kFootprintRings; ++ring) {
        const int64_t sr = hr * ring / kFootprintRings;
        const int64_t sc = hc * ring / kFootprintRings;
        // A narrow footprint collapses the inner ring onto the centre, which has
        // already answered. Eight reads of a cell whose verdict is known.
        if (sr == 0 && sc == 0) continue;
        for (const auto& off : kRingOffsets) {
            const int64_t rr = int64_t(r0) + off[0] * sr;
            const int64_t cc = int64_t(c0) + off[1] * sc;
            if (rr < 0 || rr >= int64_t(im.rows)) continue;
            if (cc < 0 || cc >= int64_t(im.cols)) continue;
            const uint8_t e = verdictFor(im.statusAt(uint32_t(rr), uint32_t(cc)),
                                         im.rangeAt(uint32_t(rr), uint32_t(cc)), r, p);
            if (e) return e;
        }
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
uint64_t applyDomain(Tile& t, const Params& p) {
    if (t.dim == 0 || p.domain.kind == Domain::Kind::Unbounded) return 0;
    double blo[3], bhi[3];
    for (int k = 0; k < 3; ++k) {
        blo[k] = t.origin[k];
        bhi[k] = t.origin[k] + double(t.dim) * p.voxelSize;
    }
    if (p.domain.testBox(blo, bhi) == Overlap::Full) return 0;

    uint64_t cleared = 0;
    for (uint32_t z = 0; z < t.dim; ++z)
        for (uint32_t y = 0; y < t.dim; ++y)
            for (uint32_t x = 0; x < t.dim; ++x) {
                uint8_t& cell = t.state[t.index(x, y, z)];
                if (!cell) continue;
                double c[3];
                t.centre(x, y, z, p.voxelSize, c);
                if (p.domain.contains(c[0], c[1], c[2])) continue;
                cell = 0;
                ++cleared;
            }
    return cleared;
}

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
    // A setup whose angular mapping was refused cannot answer a question about a
    // direction, so every voxel-by-voxel lookup into it returns nothing. The brick
    // tests have to reach the same conclusion: AllVisible here would mark voxels
    // visible on the strength of an image the carve will not read a single cell of.
    if (!im.map.valid) return BrickVerdict::NoEvidence;

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
    int64_t col0 = int64_t(std::floor(std::min(ca, cb))) - 1;
    int64_t col1 = int64_t(std::ceil(std::max(ca, cb))) + 1;
    // A sweep that runs past a full turn looked at some bearings twice, so a
    // direction can resolve to either of two columns a whole turn apart while
    // colCoord — a fractional position in one continuous table — can only name
    // one of them. Rather than choose, which is what the last two attempts at this
    // did and got wrong, the column bound simply opens to the whole raster on such
    // a scan. That is a superset of whatever cell the lookup reaches, so the
    // verdict stays conservative; it costs sharpness on the culling and nothing
    // else, and only on instruments that overshoot the turn.
    if (std::fabs(im.map.azSpanRad) > 6.28318530717958648) {
        col0 = 0;
        col1 = int64_t(im.cols) - 1;
    }

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

namespace {

constexpr double kTwoPi = 6.28318530717958648;

// THE CARVE, AS ORIGINALLY SPECIFIED
//
// "For each E57, identify which voxels are intersected by the scanner's rays to
// the points in the E57." A scatter: walk each measured ray from the setup to
// where it stopped and mark the voxels it passes through. That is why the grid is
// voxels and not samples — a ray is guaranteed to intersect the voxels along it,
// and nothing has to line up with anything.
//
// The alternative, asking each voxel's CENTRE which cell it falls in, is a gather,
// and it answers a different question: not "did a ray pass through this voxel" but
// "what does the one direction through this voxel's centre say". Those come apart
// wherever a voxel is larger than a raster cell, which is everywhere inside about
// sixteen metres — a 5 cm voxel subtends 2.86/r degrees against cells of about
// 0.14, so 445 rays pass through it at a metre and 111 at two. On APAL__0005,
// where the only-sky policy leaves a third of the raster saying nothing at all, a
// third of voxel centres landed on one of those cells and the voxel came back
// unobserved with hundreds of rays passing through it to a wall eight metres away.
// Measured: 29.6% of the air between the setup and that wall, at every range.
//
// The scatter has no such failure mode, and it is cheaper: it touches the space
// the rays actually swept rather than every voxel in the range sphere, most of
// which is behind a surface.

// How far along one ray each kind of evidence extends.
//
// Exactly the bands verdictFor names, written as intervals instead of as a test
// at a point, so the two cannot drift: visible over [0, visTo), occupied over
// [occLo, occHi]. Empty intervals mean the ray establishes nothing.
struct RayBands {
    double visTo = 0;
    double occLo = 0, occHi = -1;
};

RayBands bandsFor(rimg::Status st, double surface, const Params& p) {
    RayBands b;
    switch (st) {
    case rimg::Status::OutsideFov:
        // The scanner never looked here. Silence, not emptiness — this is the
        // distinction DESIGN.md §4 turns on.
        return b;
    case rimg::Status::NoReturn:
        // Fired and came back empty, so everything along it out to the clearing
        // distance was seen through, and nothing on it was ever measured.
        b.visTo = std::min(surface, p.maxRange);
        return b;
    case rimg::Status::Hit:
        b.visTo = std::min(surface - p.surfaceMargin, p.maxRange);
        b.occLo = surface - p.surfaceMargin;
        b.occHi = std::min(surface + p.surfaceMargin, p.maxRange);
        return b;
    }
    return b;
}

// Everything needed to turn a raster cell into a world-frame ray, hoisted out of
// the per-ray loop.
//
// The chain is the inverse of the one evidenceAt walks — instrument frame out of
// (az, el), off the instrument's own axis into the scanner's stored frame, then
// the pose's rotation into the world — and all of it but the first step is two
// fixed rotations, so they are composed into one matrix once per setup. The
// sines and cosines of the mapping's own tables go the same way: a raster has
// a few thousand rows and columns and millions of cells, so there are only a few
// thousand distinct angles to take a cosine of.
struct RayFrame {
    double M[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};   // instrument direction -> world
    std::vector<double> ce, se;                  // per row
    std::vector<double> ca, sa;                  // per column
};

RayFrame makeRayFrame(const SetupView& s) {
    RayFrame f;
    const double* R = s.worldToScanner.R;      // world -> scanner; its transpose returns
    const double* T = s.image->tilt;           // scanner -> instrument; likewise
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double v = 0;
            for (int k = 0; k < 3; ++k) v += R[3 * k + i] * T[3 * j + k];
            f.M[3 * i + j] = v;
        }
    const rimg::Mapping& m = s.image->map;
    f.ce.resize(m.elByRow.size());
    f.se.resize(m.elByRow.size());
    for (size_t i = 0; i < m.elByRow.size(); ++i) {
        f.ce[i] = std::cos(m.elByRow[i]);
        f.se[i] = std::sin(m.elByRow[i]);
    }
    f.ca.resize(m.azByCol.size());
    f.sa.resize(m.azByCol.size());
    for (size_t i = 0; i < m.azByCol.size(); ++i) {
        f.ca[i] = std::cos(m.azByCol[i]);
        f.sa[i] = std::sin(m.azByCol[i]);
    }
    return f;
}

// The world direction of an instrument-frame direction given by its own sines and
// cosines.
void rayDirection(const RayFrame& f, double ce, double se, double ca, double sa,
                  double d[3]) {
    const double v[3] = {ce * ca, ce * sa, se};
    for (int i = 0; i < 3; ++i)
        d[i] = f.M[3 * i + 0] * v[0] + f.M[3 * i + 1] * v[1] + f.M[3 * i + 2] * v[2];
}

// Clips the ray o + t*d to a tile's box, narrowing [ta, tb]. False when nothing
// of the interval is inside.
bool clipToTile(const double o[3], const double d[3], const Tile& t, double vs,
                double& ta, double& tb) {
    for (int k = 0; k < 3; ++k) {
        const double lo = t.origin[k], hi = t.origin[k] + double(t.dim) * vs;
        if (std::fabs(d[k]) < 1e-12) {
            if (o[k] < lo || o[k] > hi) return false;   // parallel and outside
            continue;
        }
        double t1 = (lo - o[k]) / d[k], t2 = (hi - o[k]) / d[k];
        if (t1 > t2) std::swap(t1, t2);
        ta = std::max(ta, t1);
        tb = std::min(tb, t2);
        if (tb < ta) return false;
    }
    return tb >= ta;
}

// Marks every voxel the ray passes through between ta and tb.
//
// A 3D DDA: from the voxel the ray enters, step to whichever of the three
// neighbours the ray crosses into next. Every voxel the segment touches is
// visited exactly once, which is the property the whole method rests on — no step
// size to choose, and no way for a voxel to be skipped between samples.
//
// Only voxels already marked kReachable are written. That is not an optimisation:
// reachability carries the domain and the rated range, both decided per voxel in
// the pass before this one, so gating on it is what keeps a ray from writing
// outside the question being asked.
void marchSegment(const SetupView& s, const Params& p, Tile& out,
                  const double d[3], double ta, double tb, uint8_t bits) {
    if (!(tb > ta) || bits == 0) return;
    const double vs = p.voxelSize;
    if (!clipToTile(s.origin, d, out, vs, ta, tb)) return;

    // Entry point, in the tile's own frame.
    double e[3];
    for (int k = 0; k < 3; ++k) e[k] = s.origin[k] - out.origin[k] + ta * d[k];

    int64_t ix[3];
    int64_t stp[3];
    double  tMax[3], tDel[3];
    for (int k = 0; k < 3; ++k) {
        ix[k] = int64_t(std::floor(e[k] / vs));
        // The clip put the entry point on the box, where rounding can put it a
        // hair outside. Clamping is right rather than rejecting: the ray is
        // demonstrably in the box over [ta, tb].
        ix[k] = std::clamp<int64_t>(ix[k], 0, int64_t(out.dim) - 1);
        if (std::fabs(d[k]) < 1e-12) {
            stp[k] = 0;
            tMax[k] = 1e300;
            tDel[k] = 1e300;
        } else {
            stp[k] = d[k] > 0 ? 1 : -1;
            const double bound = double(ix[k] + (d[k] > 0 ? 1 : 0)) * vs;
            tMax[k] = ta + (bound - e[k]) / d[k];
            tDel[k] = vs / std::fabs(d[k]);
        }
    }

    // The flat index is carried rather than recomputed: a step moves it by one
    // stride, which is what turns the inner loop into an add and a byte OR.
    const int64_t stride[3] = {1, int64_t(out.dim), int64_t(out.dim) * out.dim};
    int64_t idx = ix[0] * stride[0] + ix[1] * stride[1] + ix[2] * stride[2];
    uint8_t* state = out.state.data();

    // One visit per voxel along the segment, so the diagonal of the tile bounds
    // the count; the slack covers the clamp above.
    const int64_t cap = 3 * int64_t(out.dim) + 8;
    for (int64_t n = 0; n < cap; ++n) {
        if (state[idx] & kReachable) state[idx] |= bits;

        int a = 0;
        if (tMax[1] < tMax[a]) a = 1;
        if (tMax[2] < tMax[a]) a = 2;
        if (tMax[a] > tb) break;
        ix[a] += stp[a];
        if (ix[a] < 0 || ix[a] >= int64_t(out.dim)) break;
        idx += stp[a] * stride[a];
        tMax[a] += tDel[a];
    }
}

// How many sub-rays across one raster cell, so that neighbouring rays cannot
// leave a voxel between them unvisited.
//
// Adjacent rays diverge: at range t they are dEl*t and dAz*cos(el)*t apart. Inside
// about sixteen metres that is less than a voxel and one ray per cell covers the
// space between them; past it a gap opens, and the cell's own frustum has to be
// filled rather than its axis walked. So the count is the cell's width at the far
// end of the marched segment in voxels — 1 wherever the beam is finer than the
// grid, which is the whole of an indoor scan.
//
// Capped because the count is a cost: eight sub-rays an axis is a cell eight
// voxels wide, which at 5 cm and 0.14 degree cells is 165 m, past any instrument
// this reads. Past the cap the far field thins out, which is honest — the scan
// really did not sample it.
constexpr int kMaxSubRays = 8;

int subRaysFor(double cellAngle, double tFar, double voxelSize) {
    if (!(cellAngle > 0) || !(tFar > 0)) return 1;
    const int n = int(std::ceil(cellAngle * tFar / voxelSize));
    return std::clamp(n, 1, kMaxSubRays);
}

// A contiguous run of raster rows or columns.
struct Run { uint32_t lo, hi; };   // inclusive

// Which rows and columns hold rays that could cross the tile.
//
// The tile's bounding sphere, seen from the setup, is a cone; the rows and columns
// whose measured angle lies inside it are found by scanning the mapping's own
// tables. Scanning rather than inverting, because these tables are neither uniform
// nor confined to one turn (see rimg::Mapping) and a superset costs only a clipped
// ray, while a subset loses one.
//
// Columns come back as runs because a sweep past a full turn looks at some bearings
// twice, so the accepted set is up to two stretches of the table rather than one.
struct CellWindow {
    std::vector<Run> rows, cols;
    bool any = false;
};

CellWindow windowForTile(const SetupView& s, const Params& p, const Tile& t) {
    CellWindow w;
    const rimg::RangeImage& im = *s.image;
    const rimg::Mapping& m = im.map;
    const size_t nr = m.elByRow.size(), nc = m.azByCol.size();
    if (nr == 0 || nc == 0 || im.rows == 0 || im.cols == 0) return w;

    const double vs = p.voxelSize;
    // Centre and bounding-sphere radius of the tile, in the instrument's frame.
    double c[3];
    for (int k = 0; k < 3; ++k) c[k] = t.origin[k] + 0.5 * double(t.dim) * vs;
    const double rho = 0.5 * std::sqrt(3.0) * double(t.dim) * vs;
    s.worldToScanner.apply(c[0], c[1], c[2]);
    im.toInstrument(c[0], c[1], c[2]);
    const double dist = std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);

    // One cell of slack on the cone, because a sub-ray sits up to half a cell off
    // its cell's nominal direction and the tables are not uniform.
    const double elStep = nr > 1 ? std::fabs(m.elByRow[nr - 1] - m.elByRow[0]) / double(nr - 1) : 0;
    const double azStep = nc > 1 ? std::fabs(m.azByCol[nc - 1] - m.azByCol[0]) / double(nc - 1) : 0;
    const double slack  = std::max(elStep, azStep);

    w.any = true;
    // The setup inside or on the tile: every direction is possible, and the cone
    // arithmetic has nothing to say.
    if (dist <= rho + 1e-9) {
        w.rows.push_back({0, im.rows - 1});
        w.cols.push_back({0, im.cols - 1});
        return w;
    }

    const double alpha = std::asin(std::clamp(rho / dist, -1.0, 1.0)) + slack;
    const double elC   = std::asin(std::clamp(c[2] / dist, -1.0, 1.0));
    const double elLo = elC - alpha, elHi = elC + alpha;

    auto runsWhere = [](const std::vector<double>& table, uint32_t count,
                        auto&& keep, std::vector<Run>& into) {
        const uint32_t n = uint32_t(std::min<size_t>(table.size(), count));
        bool open = false;
        Run cur{0, 0};
        for (uint32_t i = 0; i < n; ++i) {
            if (keep(table[i])) {
                if (!open) { cur.lo = i; open = true; }
                cur.hi = i;
            } else if (open) {
                into.push_back(cur);
                open = false;
            }
        }
        if (open) into.push_back(cur);
    };

    runsWhere(m.elByRow, im.rows,
              [&](double v) { return v >= elLo && v <= elHi; }, w.rows);
    if (w.rows.empty()) { w.any = false; return w; }

    // On a small circle at elevation e, an angular radius alpha spans an azimuth
    // half-width of asin(sin alpha / cos e); at the pole it spans everything.
    constexpr double kHalfPi = 1.5707963267948966;
    const double elAbs = std::max(std::fabs(elLo), std::fabs(elHi));
    const double denom = std::cos(std::min(elAbs, kHalfPi));
    const double sinHalf = denom > 1e-9 ? std::sin(alpha) / denom : 2.0;
    if (!(sinHalf < 1.0) || elLo <= -kHalfPi || elHi >= kHalfPi) {
        w.cols.push_back({0, im.cols - 1});
        return w;
    }
    const double azC = std::atan2(c[1], c[0]);
    const double azHalf = std::asin(sinHalf);
    runsWhere(m.azByCol, im.cols, [&](double v) {
        double dd = v - azC;
        dd -= kTwoPi * std::floor(dd / kTwoPi + 0.5);   // to [-pi, pi)
        return std::fabs(dd) <= azHalf;
    }, w.cols);
    if (w.cols.empty()) w.any = false;
    return w;
}

// The angle a fractional row or column sits at. Fractional because a cell wider
// than a voxel at the far end of a ray is filled by sub-rays across it, and those
// sit between the table's entries.
double angleAt(const std::vector<double>& table, double idx) {
    if (table.empty()) return 0;
    const double top = double(table.size() - 1);
    const double v = std::clamp(idx, 0.0, top);
    const size_t i = size_t(v);
    if (i + 1 >= table.size()) return table.back();
    const double f = v - double(i);
    return table[i] * (1.0 - f) + table[i + 1] * f;
}

// One raster cell's whole contribution to one tile: the rays it fired, marched
// through the voxels they crossed.
void marchCell(const SetupView& s, const RayFrame& f, const Params& p, Tile& out,
               uint32_t row, uint32_t col) {
    const rimg::RangeImage& im = *s.image;
    const RayBands b = bandsFor(im.statusAt(row, col), im.rangeAt(row, col), p);
    const double far = std::max(b.visTo, b.occHi);
    if (!(far > 0)) return;                       // the cell establishes nothing

    const rimg::Mapping& m = im.map;
    const size_t nr = m.elByRow.size(), nc = m.azByCol.size();
    if (row >= nr || col >= nc) return;

    // The cell's angular size, from its own neighbours in the tables.
    const double dEl = nr > 1 ? std::fabs(row + 1 < nr ? m.elByRow[row + 1] - m.elByRow[row]
                                                       : m.elByRow[row] - m.elByRow[row - 1]) : 0;
    const double dAz = nc > 1 ? std::fabs(col + 1 < nc ? m.azByCol[col + 1] - m.azByCol[col]
                                                       : m.azByCol[col] - m.azByCol[col - 1]) : 0;
    const int nEl = subRaysFor(dEl, far, p.voxelSize);
    const int nAz = subRaysFor(dAz * std::max(f.ce[row], 1e-6), far, p.voxelSize);

    // The common case by a wide margin: the cell is finer than the grid at this
    // range, so its axis is the only ray it has, and its trigonometry is already
    // on the table.
    if (nEl == 1 && nAz == 1) {
        double d[3];
        rayDirection(f, f.ce[row], f.se[row], f.ca[col], f.sa[col], d);
        marchSegment(s, p, out, d, 0.0, b.visTo, kVisible);
        marchSegment(s, p, out, d, b.occLo, b.occHi, kOccupied);
        return;
    }

    for (int i = 0; i < nEl; ++i) {
        const double fr = double(row) + (double(i) + 0.5) / double(nEl) - 0.5;
        const double el = angleAt(m.elByRow, fr);
        for (int j = 0; j < nAz; ++j) {
            const double fc = double(col) + (double(j) + 0.5) / double(nAz) - 0.5;
            const double az = angleAt(m.azByCol, fc);
            double d[3];
            rayDirection(f, std::cos(el), std::sin(el), std::cos(az), std::sin(az), d);
            marchSegment(s, p, out, d, 0.0, b.visTo, kVisible);
            marchSegment(s, p, out, d, b.occLo, b.occHi, kOccupied);
        }
    }
}

// Reachability, and with it the domain: which voxels of this tile are part of the
// question at all.
//
// Separate from the march because it is a property of where a voxel is, not of
// what any ray did — and because the march needs it to already be decided: it
// writes only to reachable voxels, so the domain and the rated range are enforced
// once, here, rather than at every step of every ray.
void markReachable(const std::vector<size_t>& reach, const std::vector<SetupView>& setups,
                   const Params& p, Tile& out, Stats& stats) {
    const double R2 = p.maxRange * p.maxRange;
    const uint32_t dim = out.dim;
    const uint32_t B = kBrickVoxels;
    const uint32_t nb = (dim + B - 1) / B;

    for (uint32_t bz = 0; bz < nb; ++bz) {
        for (uint32_t by = 0; by < nb; ++by) {
            for (uint32_t bx = 0; bx < nb; ++bx) {
                const uint32_t x0 = bx * B, x1 = std::min(x0 + B, dim);
                const uint32_t y0 = by * B, y1 = std::min(y0 + B, dim);
                const uint32_t z0 = bz * B, z1 = std::min(z0 + B, dim);
                const double blo[3] = {out.origin[0] + x0 * p.voxelSize,
                                       out.origin[1] + y0 * p.voxelSize,
                                       out.origin[2] + z0 * p.voxelSize};
                const double bhi[3] = {out.origin[0] + x1 * p.voxelSize,
                                       out.origin[1] + y1 * p.voxelSize,
                                       out.origin[2] + z1 * p.voxelSize};
                // Out of the domain is out of the question: not unknown, not
                // counted, not carved.
                const Overlap ov = p.domain.testBox(blo, bhi);
                if (ov == Overlap::None) continue;
                const bool allInDomain = (ov == Overlap::Full);

                for (size_t si : reach) {
                    const SetupView& s = setups[si];
                    if (distSqPointBox(s.origin, blo, bhi) > R2) continue;
                    double far = 0;
                    for (int k = 0; k < 3; ++k) {
                        const double a = std::fabs(blo[k] - s.origin[k]);
                        const double bb = std::fabs(bhi[k] - s.origin[k]);
                        const double mx = std::max(a, bb);
                        far += mx * mx;
                    }
                    const bool allInRange = far <= R2;
                    for (uint32_t z = z0; z < z1; ++z) {
                        for (uint32_t y = y0; y < y1; ++y) {
                            for (uint32_t x = x0; x < x1; ++x) {
                                if (!allInDomain || !allInRange) {
                                    double c[3];
                                    out.centre(x, y, z, p.voxelSize, c);
                                    if (!allInDomain &&
                                        !p.domain.contains(c[0], c[1], c[2])) continue;
                                    const double dx = c[0] - s.origin[0];
                                    const double dy = c[1] - s.origin[1];
                                    const double dz = c[2] - s.origin[2];
                                    if (dx * dx + dy * dy + dz * dz > R2) continue;
                                }
                                out.state[out.index(x, y, z)] |= uint8_t(kReachable);
                                if (out.isInterior(x, y, z)) ++stats.setupTests;
                            }
                        }
                    }
                }
            }
        }
    }
}

// The scatter, over one tile.
//
// Reachability first, so the domain and the rated range are settled per voxel;
// then, per setup, the rays whose directions could cross this tile, each marched
// through the voxels it intersects. `whole` asks every cell of the raster instead
// of the windowed subset — the window is a superset of the cells that can reach
// the tile, so the two give the same answer, and the expensive one is the oracle
// that says so.
void marchTile(const std::vector<size_t>& reach, const std::vector<SetupView>& setups,
               const Params& p, Tile& out, Stats& stats, bool whole) {
    markReachable(reach, setups, p, out, stats);

    for (size_t si : reach) {
        const SetupView& s = setups[si];
        if (!s.image || !s.image->map.valid) continue;
        const rimg::RangeImage& im = *s.image;
        if (im.map.elByRow.size() < im.rows || im.map.azByCol.size() < im.cols) continue;
        const RayFrame f = makeRayFrame(s);

        if (whole) {
            for (uint32_t r = 0; r < im.rows; ++r)
                for (uint32_t c = 0; c < im.cols; ++c)
                    marchCell(s, f, p, out, r, c);
            continue;
        }

        const CellWindow w = windowForTile(s, p, out);
        if (!w.any) continue;
        for (const Run& rr : w.rows)
            for (uint32_t r = rr.lo; r <= rr.hi; ++r)
                for (const Run& cr : w.cols)
                    for (uint32_t c = cr.lo; c <= cr.hi; ++c)
                        marchCell(s, f, p, out, r, c);
    }
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

    if (p.method == Method::RayMarch) {
        marchTile(reach, setups, p, out, stats, /*whole=*/false);
        tallyTile(out, stats);
        return;
    }

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

    if (p.method == Method::RayMarch) {
        // Every cell of every reaching setup, with no window and no culling.
        marchTile(reach, setups, p, out, stats, /*whole=*/true);
        tallyTile(out, stats);
        return;
    }

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
