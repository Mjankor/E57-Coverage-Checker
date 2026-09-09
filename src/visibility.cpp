#include "visibility.h"

#include "e57.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <memory>
#include <mutex>
#include <thread>

namespace vis {

namespace {

// The colour unknown voxels are drawn in. One colour for now because there is
// only one category: when the classification pass lands, shadow / unvisited /
// review get their own and this becomes a lookup.
constexpr uint8_t kUnknownR = 255, kUnknownG = 64, kUnknownB = 96;

// The face directions, in the order visibleFaces reports them.
const int32_t kFaceDirs[6][3] = {{1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}};

// Shading, so a wall of identical dots reads as a shape.
//
// A frontier voxel is drawn as one flat-coloured sprite, and a hundred thousand
// of them in one colour is a silhouette with no interior: you can see where the
// unobserved volume is, and nothing at all about its form. Every part of the fix
// below is free at collection time, which is why it is done here rather than in
// the renderer — no depth pass, no normals buffer, no second geometry pass.
//
// Two cues, because they answer different questions:
//
//   Form. visibleFaces already looked at the six face neighbours to decide this
//   voxel is on the frontier; the visible ones summed give the outward normal,
//   and a fixed world-space light on that normal turns the blob into a surface
//   with lit and shaded sides. World-space rather than a headlight on purpose:
//   the shading then stays put as the model turns, which is what lets you read
//   which way a face points instead of everything moving together.
//
//   Height. A ramp on z within the domain, so a slab seen edge-on still has a
//   top and a bottom. On its own it is what was asked for and it is weak; under
//   the lighting it stops distant unrelated surfaces reading as one mass.
//
// The base colour stays the unknown red in all cases — this modulates it, it
// does not replace it, because the colour is what says these voxels are the
// answer rather than the scene.
enum class Shade : uint8_t { Flat, Lit, Height, LitAndHeight };

// The light. Above, and off to one side, so no principal face of a voxel is left
// exactly unlit and none is at full brightness — a light down any axis makes two
// of the six faces identical and flattens the very thing this is for.
constexpr double kLight[3] = {-0.35, -0.50, 0.79};
// How dark a fully turned-away face goes. Not zero: an unlit face still has to
// read as present, and these are the deliverable, not scenery.
constexpr double kAmbient = 0.42;

// The lit colour for one frontier voxel.
void shadeFrontier(uint8_t faces, double heightT, Shade mode,
                   uint8_t& r, uint8_t& g, uint8_t& b) {
    double lit = 1.0;
    if (mode == Shade::Lit || mode == Shade::LitAndHeight) {
        double n[3] = {0, 0, 0};
        for (int i = 0; i < 6; ++i) {
            if (!(faces & (1u << i))) continue;
            for (int k = 0; k < 3; ++k) n[k] += kFaceDirs[i][k];
        }
        const double len = std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
        // A voxel with opposite faces both visible — a one-voxel film between two
        // observed regions — sums to nothing and has no normal to speak of. It
        // gets flat ambient rather than an arbitrary direction.
        if (len > 1e-9) {
            double d = 0;
            for (int k = 0; k < 3; ++k) d += (n[k] / len) * kLight[k];
            lit = kAmbient + (1.0 - kAmbient) * std::max(0.0, d);
        } else {
            lit = kAmbient + 0.5 * (1.0 - kAmbient);
        }
    }
    double rr = kUnknownR * lit, gg = kUnknownG * lit, bb = kUnknownB * lit;
    if (mode == Shade::Height || mode == Shade::LitAndHeight) {
        // Low is the base red, high runs toward yellow: one hue sweep, so it
        // stays legible to the colour-blind and stays obviously the unknown set
        // rather than turning into a rainbow that competes with the point cloud.
        const double t = std::min(1.0, std::max(0.0, heightT));
        gg += (215.0 - kUnknownG) * t * lit;
        bb += (40.0  - kUnknownB) * t * lit;
    }
    r = uint8_t(std::min(255.0, std::max(0.0, rr)));
    g = uint8_t(std::min(255.0, std::max(0.0, gg)));
    b = uint8_t(std::min(255.0, std::max(0.0, bb)));
}

std::string fmt(const char* f, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof(buf), f, ap);
    va_end(ap);
    return buf;
}

} // namespace

// Which of a voxel's six face neighbours are visible, as a bit per face.
//
// The frontier test only needs to know whether any of them is, but the same six
// lookups also say which way the frontier faces — and that is a surface normal,
// for nothing. Summing the directions of the visible faces points from the
// unobserved voxel out into the space that was seen, which is the outward normal
// of the shadow's mouth. It is what makes the drawn result readable as a shape
// rather than as a fog of identical dots. See shadeFrontier.
//
// Faces only, not the 26-neighbourhood: a diagonal touch is a shared edge or
// corner, which is not a line of sight passing between the two.
uint8_t visibleFaces(const carve::Tile& t, uint32_t x, uint32_t y, uint32_t z) {
    uint8_t mask = 0;
    for (int i = 0; i < 6; ++i) {
        const int64_t nx = int64_t(x) + kFaceDirs[i][0];
        const int64_t ny = int64_t(y) + kFaceDirs[i][1];
        const int64_t nz = int64_t(z) + kFaceDirs[i][2];
        // The apron guarantees these are in range for every interior voxel, so
        // a tile seam cannot change the answer. Without it this bounds check
        // would silently make the frontier depend on the tiling.
        if (nx < 0 || ny < 0 || nz < 0 ||
            nx >= int64_t(t.dim) || ny >= int64_t(t.dim) || nz >= int64_t(t.dim)) continue;
        if (t.state[t.index(uint32_t(nx), uint32_t(ny), uint32_t(nz))] & carve::kVisible)
            mask |= uint8_t(1u << i);
    }
    return mask;
}

bool touchesVisible(const carve::Tile& t, uint32_t x, uint32_t y, uint32_t z) {
    return visibleFaces(t, x, y, z) != 0;
}

// A position hash, used to pick which voxels survive the display cap. It has to
// depend only on the global lattice index — never on tile size or carve order —
// or the same site would draw differently from run to run.
uint64_t voxelHash(int64_t x, int64_t y, int64_t z) {
    uint64_t h = 1469598103934665603ull;                    // FNV-1a over the triple
    const int64_t v[3] = {x, y, z};
    for (int i = 0; i < 3; ++i) {
        uint64_t u = uint64_t(v[i]);
        for (int b = 0; b < 8; ++b) { h ^= (u >> (8 * b)) & 0xFF; h *= 1099511628211ull; }
    }
    // FNV alone leaves the low bits of nearby keys correlated, which would make
    // the kept set a lattice rather than a sample. One avalanche round fixes it.
    h ^= h >> 33; h *= 0xff51afd7ed558ccdull;
    h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ull;
    h ^= h >> 33;
    return h;
}

namespace {

// Collects the voxels to draw, sampling down whenever the cap is reached.
//
// The origin is fixed before collection starts rather than taken from the first
// voxel to arrive. With one thread those are the same thing; with several,
// "first to arrive" is a race, and an origin that varied run to run would make
// the float offsets — and so the drawn positions — vary with it.
struct Collector {
    std::vector<lod::StorePoint> out;
    uint64_t cap = 0;
    uint64_t threshold = ~0ull;   // keep when voxelHash < threshold
    uint64_t qualified = 0;
    double   origin[3] = {0, 0, 0};

    // The shading needs the whole domain's height range before it can place a
    // voxel within it, so these are set once, before collection starts, for the
    // same reason `origin` is.
    Shade  shade = Shade::LitAndHeight;
    double zLo = 0, zSpan = 0;

    void add(const double centre[3], const int64_t gi[3], uint8_t faces) {
        ++qualified;
        const uint64_t h = voxelHash(gi[0], gi[1], gi[2]);
        if (h >= threshold) return;

        lod::StorePoint p{};
        p.x = float(centre[0] - origin[0]);
        p.y = float(centre[1] - origin[1]);
        p.z = float(centre[2] - origin[2]);
        if (shade == Shade::Flat) {
            p.r = kUnknownR; p.g = kUnknownG; p.b = kUnknownB;
        } else {
            shadeFrontier(faces, zSpan > 0 ? (centre[2] - zLo) / zSpan : 0.5,
                          shade, p.r, p.g, p.b);
        }
        p.a = 255;
        p.scanId = 0;
        out.push_back(p);
        keys.push_back(h);

        if (cap && out.size() > cap) halve();
    }

    // Takes another collector's voxels. Both were filtered at their own
    // threshold; re-filtering the union at the lower of the two restores the
    // exact invariant, and the caller then halves down to the cap as a single
    // thread would have.
    void absorb(Collector& other) {
        qualified += other.qualified;
        threshold = std::min(threshold, other.threshold);
        out.insert(out.end(), other.out.begin(), other.out.end());
        keys.insert(keys.end(), other.keys.begin(), other.keys.end());
        other.out.clear();
        other.keys.clear();
    }

    void filterToThreshold() {
        size_t w = 0;
        for (size_t i = 0; i < out.size(); ++i)
            if (keys[i] < threshold) { out[w] = out[i]; keys[w] = keys[i]; ++w; }
        out.resize(w);
        keys.resize(w);
    }

    // Halving the threshold drops about half the kept set, and re-testing what
    // is already held keeps the invariant exact: whatever order voxels arrived
    // in, and however many times the threshold has moved, the result is always
    // "every voxel whose hash is below the current threshold". That is what
    // makes the same site draw the same way at any tile size.
    void halve() {
        threshold /= 2;
        filterToThreshold();
    }

    std::vector<uint64_t> keys;
};

} // namespace

uint64_t imageCellsPerScan(const Options& opt, uint64_t scanCount) {
    if (scanCount == 0) return opt.minImageCells;
    // Bytes into cells. A cell is three bytes of raster, and the min/max pyramid
    // over it adds one node per 4x4 block at every level — five bytes a node,
    // which summed over the levels is a third of a byte per cell. Kept as thirds
    // of a byte in integers so there is no rounding to argue about.
    constexpr uint64_t kThirdBytesPerCell = 10;          // 3 + 1/3, times three
    uint64_t cells = (opt.imageBudgetBytes / kThirdBytesPerCell * 3) / scanCount;
    cells = std::max<uint64_t>(cells, opt.minImageCells);
    // rimg::Options::maxCells is 32 bits, and no raster approaches it.
    return std::min<uint64_t>(cells, 0xFFFFFFFFull);
}

void rebase(const Result& r, const double origin[3], std::vector<lod::StorePoint>& out) {
    out = r.voxels;
    const float dx = float(r.origin[0] - origin[0]);
    const float dy = float(r.origin[1] - origin[1]);
    const float dz = float(r.origin[2] - origin[2]);
    for (lod::StorePoint& p : out) { p.x += dx; p.y += dy; p.z += dz; }
}

bool run(const std::vector<std::string>& paths, const Options& opt,
         const Progress& progress, Result& out, std::string& err) {
    out = Result{};
    out.voxelSize = opt.voxelSize;

    auto tick = [&progress](const char* stage, uint64_t done, uint64_t total) {
        return progress ? progress(stage, done, total) : true;
    };

    // --- how many scans are there, so the per-image budget can be set ------
    // Opening every file twice is cheap next to decoding: the first pass reads
    // headers only.
    std::vector<std::unique_ptr<e57::Reader>> readers;
    uint64_t scanCount = 0;
    for (size_t i = 0; i < paths.size(); ++i) {
        auto r = std::make_unique<e57::Reader>();
        std::string e;
        if (!r->open(paths[i], e)) { err = paths[i] + ": " + e; return false; }
        scanCount += r->scanCount();
        readers.push_back(std::move(r));
        if (!tick("opening files", i + 1, paths.size())) {
            out.cancelled = true;
            err = "cancelled";
            return false;
        }
    }
    if (scanCount == 0) { err = "no scans in the selected files"; return false; }

    const uint64_t perImage = imageCellsPerScan(opt, scanCount);
    out.imageCellsAllowed = perImage;

    rimg::Options ro;
    ro.maxRange = opt.maxRange;
    ro.maxCells = uint32_t(perImage);
    ro.noReturnRadius   = opt.skyRadius;
    ro.noReturnFraction = opt.skyFraction;
    ro.blindCone        = opt.blindCone;

    // --- range images -----------------------------------------------------
    std::vector<std::unique_ptr<rimg::RangeImage>> images;
    std::vector<carve::SetupView> setups;
    images.reserve(size_t(scanCount));
    setups.reserve(size_t(scanCount));

    uint64_t isolated = 0, believedSky = 0;

    // Built in parallel across scans. Range images are independent — one scan's
    // raster, tilt, mapping and round trip involve no other scan — so this is
    // the one stage of the pipeline that parallelises with nothing shared, and
    // at a second a scan it is the difference between a quarter of an hour and
    // two minutes on a thousand-scan job.
    //
    // Each worker opens its own reader rather than sharing one. The readers
    // above are kept for their headers; decoding through the same object from
    // several threads would mean sharing its field decoders, which carry the
    // bit cursor that makes a bytestream continuous across packets.
    //
    // Results are written into a slot indexed by job, never appended, so the
    // order of `images` is the order of the corpus whatever order the threads
    // finish in. Every count derived from them is taken afterwards, in that same
    // order, for the same reason.
    struct Job { size_t path; size_t scan; };
    std::vector<Job> jobs;
    jobs.reserve(size_t(scanCount));
    for (size_t f = 0; f < readers.size(); ++f)
        for (size_t i = 0; i < readers[f]->scanCount(); ++i) jobs.push_back({f, i});

    std::vector<std::unique_ptr<rimg::RangeImage>> built(jobs.size());
    {
        unsigned nb = opt.threads ? opt.threads : std::thread::hardware_concurrency();
        if (nb == 0) nb = 1;
        // Each concurrent build holds that scan's decoded returns — about 20
        // bytes a point, so a hundred megabytes for a large scan. That is the
        // cost of this, and it is why the thread count is not simply unbounded.
        nb = unsigned(std::min<uint64_t>(nb, std::max<uint64_t>(1, jobs.size())));

        std::atomic<uint64_t> next{0};
        std::atomic<uint64_t> done{0};
        std::atomic<bool>     stop{false};
        std::mutex            tickLock;

        auto worker = [&]() {
            e57::Reader own;
            size_t openPath = SIZE_MAX;
            for (;;) {
                if (stop.load(std::memory_order_relaxed)) break;
                const uint64_t j = next.fetch_add(1, std::memory_order_relaxed);
                if (j >= jobs.size()) break;
                const Job& job = jobs[size_t(j)];
                // Reopened only when the file changes, so a multi-scan file is
                // opened once per worker rather than once per scan.
                std::string e;
                if (openPath != job.path) {
                    own = e57::Reader{};
                    if (!own.open(paths[job.path], e)) { openPath = SIZE_MAX; }
                    else openPath = job.path;
                }
                if (openPath == job.path) {
                    auto img = std::make_unique<rimg::RangeImage>();
                    std::string rerr;
                    if (rimg::build(own, job.scan, ro, *img, rerr))
                        built[size_t(j)] = std::move(img);
                }
                const uint64_t d = done.fetch_add(1, std::memory_order_relaxed) + 1;
                std::lock_guard<std::mutex> lk(tickLock);
                if (!tick("building range images", d, scanCount))
                    stop.store(true, std::memory_order_relaxed);
            }
        };

        if (nb == 1) {
            worker();
        } else {
            std::vector<std::thread> pool;
            pool.reserve(nb);
            for (unsigned i = 0; i < nb; ++i) pool.emplace_back(worker);
            for (std::thread& t : pool) t.join();
        }
        if (stop.load()) { out.cancelled = true; err = "cancelled"; return false; }
    }

    for (auto& img : built) {
        if (!img) {
            // A scan with no usable raster cannot contribute evidence, and
            // inventing one would invent visibility. Skipped and counted.
            ++out.scansSkipped;
            continue;
        }
        if (!img->map.valid) {
            ++out.setupsWithoutMapping;
            if (out.mappingRefusedWhy.empty()) out.mappingRefusedWhy = img->diag.note;
        }
        if (img->diag.binStep > 1) {
            ++out.setupsBinned;
            out.worstBinStep = std::max(out.worstBinStep, img->diag.binStep);
        }
        images.push_back(std::move(img));
    }

    // The blind cone, decided once across every scan rather than scan by scan.
    // It has to happen here, after all the images exist and before anything reads
    // a cell, because the evidence is the corpus: a band unsampled at the same
    // size in every scan is the instrument, and one that varies from 87 rows to
    // 576 is the scene it was standing in. No single scan can tell those apart —
    // on the job this was built against the per-scan test refused two and called
    // two more inverted, and every one of those mistakes either clears a cone to
    // the rated range straight through the ground or throws away the sky that
    // clears the volume above the site.
    {
        std::vector<rimg::RangeImage*> raw;
        raw.reserve(images.size());
        for (auto& im : images) raw.push_back(im.get());
        out.coneVerdict = rimg::markBlindConeAcrossCorpus(raw, ro);
    }

    for (auto& img : images) {
        isolated    += img->diag.isolatedNoReturns;
        believedSky += img->diag.noReturns;
        if (img->diag.blindConeRows) {
            ++out.setupsWithBlindCone;
            if (img->diag.hasConeAxis && img->diag.coneAxisWorld[2] > 0.5)
                ++out.setupsInverted;
        }
        // The accelerator the carve culls with. About 5/16 of a byte per cell, and
        // it settles most bricks with one lookup instead of 512 voxel tests. Built
        // after the cone is marked: it summarises cell statuses, so a pyramid
        // built before the marking would answer for an image that no longer
        // exists.
        rimg::buildPyramid(*img);
        setups.push_back(carve::makeSetupView(*img));
    }

    if (setups.empty()) {
        err = "no scan produced a usable range image — none of them declare a "
              "sampling grid (indexBounds with rowIndex/columnIndex)";
        return false;
    }
    out.setupsUsed = setups.size();

    // --- carve ------------------------------------------------------------
    carve::Params p;
    p.voxelSize = opt.voxelSize;
    // Half a voxel diagonal, so it tracks the voxel size rather than being a
    // constant that silently stops matching it.
    p.surfaceMargin = 0.5 * opt.voxelSize * 1.7320508075688772;
    p.maxRange   = opt.maxRange;
    p.tileVoxels = opt.tileVoxels;
    // The frontier rule asks about face neighbours; the apron is what lets it
    // do that without the answer depending on where tile seams fall.
    p.apron      = opt.solid ? 0u : 1u;
    p.earlyOut   = opt.earlyOut;

    // --- the domain -------------------------------------------------------
    // What the range spheres alone would cover, kept for comparison so the
    // report can say how much narrowing the question actually saved.
    {
        double slo[3] = {0, 0, 0}, shi[3] = {0, 0, 0};
        bool first = true;
        for (const carve::SetupView& s : setups) {
            for (int k = 0; k < 3; ++k) {
                const double a = s.origin[k] - opt.maxRange, b = s.origin[k] + opt.maxRange;
                if (first) { slo[k] = a; shi[k] = b; }
                else { slo[k] = std::min(slo[k], a); shi[k] = std::max(shi[k], b); }
            }
            first = false;
        }
        out.sphereVolume = (shi[0] - slo[0]) * (shi[1] - slo[1]) * (shi[2] - slo[2]);
    }

    if (opt.domain == DomainMode::MeasuredExtent) {
        // The union of what every scan actually returned. Bounds are measured in
        // each scanner's own frame, so the eight corners of each box go through
        // that setup's pose — transforming a box by a rotation and re-bounding
        // it grows it, which is the safe direction for a region that decides
        // what gets asked about.
        double lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};
        bool have = false;
        auto include = [&](double x, double y, double z) {
            const double v[3] = {x, y, z};
            for (int k = 0; k < 3; ++k) {
                if (!have) { lo[k] = hi[k] = v[k]; }
                else { lo[k] = std::min(lo[k], v[k]); hi[k] = std::max(hi[k], v[k]); }
            }
            have = true;
        };
        for (const carve::SetupView& s : setups) {
            // The setup itself is always part of the surveyed region, even for a
            // scan that returned nothing.
            include(s.origin[0], s.origin[1], s.origin[2]);
            const rimg::Diagnostics& d = s.image->diag;
            if (!d.hasReturnBounds) continue;
            const viewer::Rigid fwd = s.image->hasPose
                                    ? viewer::rigidFromPose(s.image->pose)
                                    : viewer::Rigid{};
            for (int corner = 0; corner < 8; ++corner) {
                double c[3];
                for (int k = 0; k < 3; ++k)
                    c[k] = (corner & (1 << k)) ? d.returnMax[k] : d.returnMin[k];
                fwd.apply(c[0], c[1], c[2]);
                include(c[0], c[1], c[2]);
            }
        }
        if (have) {
            p.domain.kind = carve::Domain::Kind::Box;
            for (int k = 0; k < 3; ++k) {
                p.domain.lo[k] = lo[k] - opt.domainMargin;
                p.domain.hi[k] = hi[k] + opt.domainMargin;
            }
            out.domainVolume = (p.domain.hi[0] - p.domain.lo[0]) *
                               (p.domain.hi[1] - p.domain.lo[1]) *
                               (p.domain.hi[2] - p.domain.lo[2]);
        }
    }
    out.domain = p.domain;

    const std::vector<carve::TileKey> keys = carve::tilesForSetups(setups, p);
    out.tilesTotal = keys.size();
    const uint64_t plannedTiles = opt.maxTiles
                                ? std::min<uint64_t>(opt.maxTiles, keys.size())
                                : keys.size();

    // The origin is the centre of the domain, computed from the tile list before
    // anything is carved. Deterministic, independent of thread count, and it
    // keeps the float offsets small even at UTM magnitudes.
    double origin[3] = {0, 0, 0};
    if (!keys.empty()) {
        int64_t lo[3] = {keys[0].x, keys[0].y, keys[0].z};
        int64_t hi[3] = {keys[0].x, keys[0].y, keys[0].z};
        for (const carve::TileKey& k : keys) {
            const int64_t v[3] = {k.x, k.y, k.z};
            for (int i = 0; i < 3; ++i) {
                lo[i] = std::min(lo[i], v[i]);
                hi[i] = std::max(hi[i], v[i]);
            }
        }
        for (int i = 0; i < 3; ++i)
            origin[i] = 0.5 * (double(lo[i]) + double(hi[i]) + 1.0) * p.tileMetres();
    }

    // The classification grid. One byte per voxel of the whole domain at once,
    // because connectivity cannot be answered tile by tile: a void that spans a
    // seam is one void, and a void with a way out through a neighbouring tile is
    // not a void at all.
    voids::Grid grid;
    bool useGrid = false;
    if (opt.classifyVoids && !keys.empty()) {
        int64_t klo[3] = {keys[0].x, keys[0].y, keys[0].z};
        int64_t khi[3] = {keys[0].x, keys[0].y, keys[0].z};
        for (const carve::TileKey& k : keys) {
            const int64_t v[3] = {k.x, k.y, k.z};
            for (int i = 0; i < 3; ++i) {
                klo[i] = std::min(klo[i], v[i]);
                khi[i] = std::max(khi[i], v[i]);
            }
        }
        uint64_t cells = 1;
        bool overflow = false;
        for (int i = 0; i < 3; ++i) {
            grid.lo[i]  = klo[i] * int64_t(p.tileVoxels);
            const int64_t span = (khi[i] - klo[i] + 1) * int64_t(p.tileVoxels);
            if (span <= 0 || span > int64_t(UINT32_MAX)) { overflow = true; break; }
            grid.dim[i] = uint32_t(span);
            cells *= uint64_t(span);
            if (cells > (1ull << 62)) { overflow = true; break; }
        }
        grid.voxelSize = p.voxelSize;
        if (overflow || cells > opt.classifyBudgetBytes) {
            out.classifySkipped = fmt("the domain is %.1f G voxels, over the %.1f G byte "
                                      "budget for the connectivity pass",
                                      double(cells) / 1e9,
                                      double(opt.classifyBudgetBytes) / 1e9);
        } else {
            grid.state.assign(size_t(cells), 0);
            useGrid = true;
        }
    }

    unsigned nthreads = opt.threads ? opt.threads : std::thread::hardware_concurrency();
    if (nthreads == 0) nthreads = 1;
    // A carver is the parallelism; a pool of threads feeding it would only
    // contend for one device queue.
    if (opt.carver) nthreads = 1;
    nthreads = unsigned(std::min<uint64_t>(nthreads, std::max<uint64_t>(1, plannedTiles)));

    // The height range the ramp spans. The domain when it is bounded — that is
    // the region being asked about, so the ramp uses all of its contrast on the
    // part of the site anyone is looking at — and the tiles' own extent when it
    // is not. Fixed before collection for the same reason `origin` is: with
    // several workers, anything derived as voxels arrive is a race.
    double shadeZLo = 0, shadeZSpan = 0;
    if (p.domain.kind == carve::Domain::Kind::Box) {
        shadeZLo = p.domain.lo[2];
        shadeZSpan = p.domain.hi[2] - p.domain.lo[2];
    } else if (!keys.empty()) {
        int64_t zlo = keys[0].z, zhi = keys[0].z;
        for (const carve::TileKey& k : keys) {
            zlo = std::min(zlo, k.z);
            zhi = std::max(zhi, k.z);
        }
        shadeZLo   = double(zlo) * p.tileMetres();
        shadeZSpan = double(zhi - zlo + 1) * p.tileMetres();
    }

    // Per worker: its own statistics, its own collector, its own tile scratch.
    // Nothing is shared but the tile cursor and the progress lock.
    struct Worker {
        carve::Stats stats;
        Collector    col;
        carve::Tile  tile;
        carve::Tile  check;          // verification only
        uint64_t     carverTiles = 0;
        uint64_t     carverRefused = 0;
        uint64_t     disagreements = 0;
        uint64_t     compared = 0;
    };
    std::vector<Worker> workers(nthreads);
    for (Worker& w : workers) {
        // Each worker gets the whole cap rather than a share of it. A worker's
        // voxels are a subset of the run's, so a subset can never need a lower
        // threshold than the whole would — which is what makes the merge below
        // land on exactly the threshold one thread would have reached. Tiles are
        // handed out dynamically, so in practice a worker holds about its share.
        w.col.cap = opt.displayCap;
        for (int k = 0; k < 3; ++k) w.col.origin[k] = origin[k];
        w.col.shade = Shade(opt.shading);
        w.col.zLo = shadeZLo; w.col.zSpan = shadeZSpan;
    }

    std::atomic<uint64_t> cursor{0};
    std::atomic<uint64_t> finished{0};
    std::atomic<bool>     stop{false};
    std::mutex            progressLock;

    auto body = [&](unsigned id) {
        Worker& w = workers[id];
        for (;;) {
            if (stop.load(std::memory_order_relaxed)) break;
            const uint64_t t = cursor.fetch_add(1, std::memory_order_relaxed);
            if (t >= plannedTiles) break;

            bool carved = false;
            if (opt.carver) {
                // Under verification the carver's output goes to a scratch tile
                // and the CPU's is the one used. The CPU is the oracle; a mode
                // whose purpose is to find out whether the carver is lying must
                // not then hand you the carver's answer.
                carve::Tile&  target = opt.verifyCarver ? w.check : w.tile;
                // A scratch tally too: a carver that declines part way may
                // already have counted some of the tile, and the CPU pass that
                // follows would count it again.
                carve::Stats attempt;
                carved = opt.carver(keys[size_t(t)], setups, p, target, attempt,
                                    opt.carverUser);
                if (carved) {
                    ++w.carverTiles;
                    if (!opt.verifyCarver) {
                        w.stats.voxels     += attempt.voxels;
                        w.stats.reachable  += attempt.reachable;
                        w.stats.visible    += attempt.visible;
                        w.stats.occupied   += attempt.occupied;
                        w.stats.unknown    += attempt.unknown;
                        w.stats.setupTests += attempt.setupTests;
                    }
                } else {
                    ++w.carverRefused;
                }
            }
            if (!carved || opt.verifyCarver) {
                carve::carveTile(keys[size_t(t)], setups, p, w.tile, w.stats);
            }
            if (carved && opt.verifyCarver) {
                w.compared += w.tile.state.size();
                for (size_t i = 0; i < w.check.state.size() && i < w.tile.state.size(); ++i)
                    if (w.check.state[i] != w.tile.state[i]) ++w.disagreements;
            }

            const carve::Tile& tile = w.tile;
            if (useGrid) {
                // Tiles are disjoint, so workers write to disjoint regions of
                // the grid and need no lock between them.
                for (uint32_t z = tile.interiorBegin(); z < tile.interiorEnd(); ++z) {
                    for (uint32_t y = tile.interiorBegin(); y < tile.interiorEnd(); ++y) {
                        for (uint32_t x = tile.interiorBegin(); x < tile.interiorEnd(); ++x) {
                            int64_t gi[3];
                            tile.globalIndex(x, y, z, gi);
                            const int64_t gx = gi[0] - grid.lo[0];
                            const int64_t gy = gi[1] - grid.lo[1];
                            const int64_t gz = gi[2] - grid.lo[2];
                            if (gx < 0 || gy < 0 || gz < 0 ||
                                gx >= int64_t(grid.dim[0]) || gy >= int64_t(grid.dim[1]) ||
                                gz >= int64_t(grid.dim[2])) continue;
                            grid.state[grid.index(uint32_t(gx), uint32_t(gy), uint32_t(gz))] =
                                tile.state[tile.index(x, y, z)];
                        }
                    }
                }
            } else {
                for (uint32_t z = tile.interiorBegin(); z < tile.interiorEnd(); ++z) {
                    for (uint32_t y = tile.interiorBegin(); y < tile.interiorEnd(); ++y) {
                        for (uint32_t x = tile.interiorBegin(); x < tile.interiorEnd(); ++x) {
                            // Exactly kReachable: in the domain, and nothing
                            // observed it. Anything else is either outside the
                            // question or was seen by something.
                            if (tile.state[tile.index(x, y, z)] != carve::kReachable) continue;
                            // The same six lookups decide the frontier and give
                            // the outward normal — see visibleFaces.
                            const uint8_t faces = visibleFaces(tile, x, y, z);
                            if (!opt.solid && !faces) continue;
                            double c[3];
                            tile.centre(x, y, z, p.voxelSize, c);
                            int64_t gi[3];
                            tile.globalIndex(x, y, z, gi);
                            w.col.add(c, gi, faces);
                        }
                    }
                }
            }

            const uint64_t d = finished.fetch_add(1, std::memory_order_relaxed) + 1;
            // Serialised because the callback ends up on one UI thread. It is
            // taken once per tile, which is milliseconds of work apart.
            std::lock_guard<std::mutex> lk(progressLock);
            if (!tick("carving", d, plannedTiles)) stop.store(true, std::memory_order_relaxed);
        }
    };

    if (nthreads == 1) {
        body(0);
    } else {
        std::vector<std::thread> pool;
        pool.reserve(nthreads);
        for (unsigned i = 0; i < nthreads; ++i) pool.emplace_back(body, i);
        for (std::thread& th : pool) th.join();
    }

    out.tilesCarved = finished.load();
    if (stop.load()) { out.cancelled = true; out.partial = true; }
    else if (opt.maxTiles && opt.maxTiles < keys.size()) out.partial = true;

    // Merge. Integer sums are order-independent, so the statistics are exact
    // whatever order the tiles finished in.
    Collector col;
    col.cap = opt.displayCap;
    col.shade = Shade(opt.shading);
    col.zLo = shadeZLo; col.zSpan = shadeZSpan;
    for (int k = 0; k < 3; ++k) col.origin[k] = origin[k];
    for (Worker& w : workers) {
        out.stats.voxels     += w.stats.voxels;
        out.stats.reachable  += w.stats.reachable;
        out.stats.visible    += w.stats.visible;
        out.stats.occupied   += w.stats.occupied;
        out.stats.unknown    += w.stats.unknown;
        out.stats.setupTests += w.stats.setupTests;
        out.carverTiles          += w.carverTiles;
        out.carverRefused        += w.carverRefused;
        out.carverDisagreements  += w.disagreements;
        out.carverVoxelsCompared += w.compared;
        col.absorb(w.col);
    }
    col.filterToThreshold();

    if (useGrid) {
        if (!tick("classifying voids", 0, 0)) { out.cancelled = true; out.partial = true; }
        out.voidReport = voids::classify(grid);
        out.classified = true;

        // Only enclosed voids are drawn now. Everything else unobserved is the
        // rest of the world arriving by some route, and drawing it buries the
        // finding under the whole planet.
        for (uint32_t z = 0; z < grid.dim[2]; ++z) {
            for (uint32_t y = 0; y < grid.dim[1]; ++y) {
                for (uint32_t x = 0; x < grid.dim[0]; ++x) {
                    const uint8_t bits = grid.state[grid.index(x, y, z)];
                    if (!voids::isEnclosedVoid(bits)) continue;
                    if (!opt.solid && !voids::touchesObserved(grid, x, y, z)) continue;
                    double c[3];
                    grid.centre(x, y, z, c);
                    const int64_t gi[3] = {grid.lo[0] + x, grid.lo[1] + y, grid.lo[2] + z};
                    // The connectivity grid answers "observed", not "visible", so
                    // its faces are found here rather than by visibleFaces.
                    col.add(c, gi, voids::observedFaces(grid, x, y, z));
                }
            }
        }
    }

    while (col.cap && col.out.size() > col.cap) col.halve();

    out.voxels = std::move(col.out);
    for (int k = 0; k < 3; ++k) out.origin[k] = origin[k];
    out.qualified = col.qualified;
    out.keptFraction = col.qualified ? double(out.voxels.size()) / double(col.qualified) : 1.0;

    bool first = true;
    for (const lod::StorePoint& v : out.voxels) {
        if (first) {
            out.bounds.lo[0] = out.bounds.hi[0] = v.x;
            out.bounds.lo[1] = out.bounds.hi[1] = v.y;
            out.bounds.lo[2] = out.bounds.hi[2] = v.z;
            first = false;
        } else {
            out.bounds.expand(v.x, v.y, v.z);
        }
    }

    std::string note;
    // Whether rasters were actually coarsened, not whether the budget looked
    // tight. This used to fire whenever the per-image allowance fell below 32 M
    // cells, which is a guess about what probably happened rather than a report of
    // what did: at a thousand scans the allowance is 15 M cells and a 13.2 M cell
    // raster fits inside it untouched, so the guess cries coarsening over a run
    // that coarsened nothing — and the real thing, which overstates the answer,
    // would read the same as the false alarm.
    if (out.setupsBinned)
        note += fmt("%llu of %llu raster(s) COARSENED, up to %u declared cells into one per "
                    "edge, to fit %llu cells each — coarse cells clear less space, so the "
                    "unobserved volume below is OVERSTATED; raise the image budget; ",
                    (unsigned long long)out.setupsBinned,
                    (unsigned long long)out.setupsUsed, out.worstBinStep,
                    (unsigned long long)perImage);
    if (out.scansSkipped)
        note += fmt("%llu scan(s) skipped for want of a sampling grid; ",
                    (unsigned long long)out.scansSkipped);
    if (out.keptFraction < 1.0)
        note += fmt("showing %.1f%% of %llu frontier voxels (display cap); ",
                    100.0 * out.keptFraction, (unsigned long long)out.qualified);
    if (out.setupsWithoutMapping) {
        note += fmt("%llu of %llu setup(s) had their angular mapping refused and contribute "
                    "NOTHING — every lookup against them falls outside the raster; ",
                    (unsigned long long)out.setupsWithoutMapping,
                    (unsigned long long)out.setupsUsed);
    }
    if (out.setupsInverted) {
        note += fmt("%llu setup(s) were mounted inverted (blind cone pointing up); ",
                    (unsigned long long)out.setupsInverted);
    }
    if (isolated) {
        note += fmt("%llu empty cells looked like dropped returns rather than sky and "
                    "cleared nothing (%llu were believed); ",
                    (unsigned long long)isolated, (unsigned long long)believedSky);
    }
    if (out.classified) {
        note += fmt("%llu enclosed void(s), %.1f m^3, largest %.1f m^3; the other %.0f m^3 "
                    "of unobserved space reaches the outside world; ",
                    (unsigned long long)out.voidReport.components,
                    out.enclosedVolume(),
                    double(out.voidReport.largestComponent) * out.voxelVolume(),
                    out.exteriorVolume());
    } else if (!out.classifySkipped.empty()) {
        note += "voids not classified: " + out.classifySkipped + "; ";
    }
    if (out.domain.kind == carve::Domain::Kind::Box && out.sphereVolume > 0)
        note += fmt("domain narrowed to the surveyed extent, %.0f m^3 instead of %.0f; ",
                    out.domainVolume, out.sphereVolume);
    else if (out.domain.kind == carve::Domain::Kind::Unbounded)
        note += "domain is the full range spheres, so most of it is open air; ";
    if (out.carverTiles && opt.verifyCarver)
        note += fmt("%llu of %llu voxels differ between the carver and the CPU; ",
                    (unsigned long long)out.carverDisagreements,
                    (unsigned long long)out.carverVoxelsCompared);
    if (out.carverRefused)
        note += fmt("%llu tile(s) declined by the carver and done on the CPU; ",
                    (unsigned long long)out.carverRefused);
    if (opt.earlyOut == carve::EarlyOut::AnyEvidence)
        note += "stopped at the first evidence, so visible and occupied are lower bounds; ";
    if (out.partial)
        note += out.cancelled ? "cancelled part way; " : "stopped at the tile limit; ";
    out.note = note;
    return true;
}

} // namespace vis
