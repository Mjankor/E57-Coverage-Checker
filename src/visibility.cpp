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

std::string fmt(const char* f, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof(buf), f, ap);
    va_end(ap);
    return buf;
}

} // namespace

// A voxel is on the observed frontier when a face neighbour is visible. Faces
// only, not the 26-neighbourhood: a diagonal touch is a shared edge or corner,
// which is not a line of sight passing between the two.
bool touchesVisible(const carve::Tile& t, uint32_t x, uint32_t y, uint32_t z) {
    const int32_t d[6][3] = {{1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}};
    for (const auto& o : d) {
        const int64_t nx = int64_t(x) + o[0], ny = int64_t(y) + o[1], nz = int64_t(z) + o[2];
        // The apron guarantees these are in range for every interior voxel, so
        // a tile seam cannot change the answer. Without it this bounds check
        // would silently make the frontier depend on the tiling.
        if (nx < 0 || ny < 0 || nz < 0 ||
            nx >= int64_t(t.dim) || ny >= int64_t(t.dim) || nz >= int64_t(t.dim)) continue;
        if (t.state[t.index(uint32_t(nx), uint32_t(ny), uint32_t(nz))] & carve::kVisible)
            return true;
    }
    return false;
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

    void add(const double centre[3], const int64_t gi[3]) {
        ++qualified;
        const uint64_t h = voxelHash(gi[0], gi[1], gi[2]);
        if (h >= threshold) return;

        lod::StorePoint p{};
        p.x = float(centre[0] - origin[0]);
        p.y = float(centre[1] - origin[1]);
        p.z = float(centre[2] - origin[2]);
        p.r = kUnknownR; p.g = kUnknownG; p.b = kUnknownB; p.a = 255;
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

    uint64_t perImage = opt.totalImageCells / scanCount;
    perImage = std::max<uint64_t>(perImage, opt.minImageCells);
    perImage = std::min<uint64_t>(perImage, 0xFFFFFFFFull);

    rimg::Options ro;
    ro.maxRange = opt.maxRange;
    ro.maxCells = uint32_t(perImage);

    // --- range images -----------------------------------------------------
    std::vector<std::unique_ptr<rimg::RangeImage>> images;
    std::vector<carve::SetupView> setups;
    images.reserve(size_t(scanCount));
    setups.reserve(size_t(scanCount));

    uint64_t seen = 0;
    for (const auto& r : readers) {
        for (size_t i = 0; i < r->scanCount(); ++i) {
            auto img = std::make_unique<rimg::RangeImage>();
            std::string rerr;
            if (rimg::build(*r, i, ro, *img, rerr)) {
                // The accelerator the carve culls with. About 5/16 of a byte
                // per cell, and it settles most bricks with one lookup instead
                // of 512 voxel tests.
                rimg::buildPyramid(*img);
                images.push_back(std::move(img));
                setups.push_back(carve::makeSetupView(*images.back()));
            } else {
                // A scan with no usable raster cannot contribute evidence, and
                // inventing one would invent visibility. Skipped and counted.
                ++out.scansSkipped;
            }
            if (!tick("building range images", ++seen, scanCount)) {
                out.cancelled = true;
                err = "cancelled";
                return false;
            }
        }
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

    unsigned nthreads = opt.threads ? opt.threads : std::thread::hardware_concurrency();
    if (nthreads == 0) nthreads = 1;
    // A carver is the parallelism; a pool of threads feeding it would only
    // contend for one device queue.
    if (opt.carver) nthreads = 1;
    nthreads = unsigned(std::min<uint64_t>(nthreads, std::max<uint64_t>(1, plannedTiles)));

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
                // Statistics go to a scratch tally: if the carver declines part
                // way it may already have counted some of the tile, and the CPU
                // pass that follows would count it again.
                carve::Stats attempt;
                carved = opt.carver(keys[size_t(t)], setups, p, w.tile, attempt,
                                    opt.carverUser);
                if (carved) {
                    ++w.carverTiles;
                    w.stats.voxels     += attempt.voxels;
                    w.stats.reachable  += attempt.reachable;
                    w.stats.visible    += attempt.visible;
                    w.stats.occupied   += attempt.occupied;
                    w.stats.unknown    += attempt.unknown;
                    w.stats.setupTests += attempt.setupTests;
                } else {
                    ++w.carverRefused;
                }
            }
            if (!carved) {
                carve::carveTile(keys[size_t(t)], setups, p, w.tile, w.stats);
            } else if (opt.verifyCarver) {
                carve::Stats ignored;
                carve::carveTile(keys[size_t(t)], setups, p, w.check, ignored);
                w.compared += w.check.state.size();
                for (size_t i = 0; i < w.check.state.size() && i < w.tile.state.size(); ++i)
                    if (w.check.state[i] != w.tile.state[i]) ++w.disagreements;
            }

            const carve::Tile& tile = w.tile;
            for (uint32_t z = tile.interiorBegin(); z < tile.interiorEnd(); ++z) {
                for (uint32_t y = tile.interiorBegin(); y < tile.interiorEnd(); ++y) {
                    for (uint32_t x = tile.interiorBegin(); x < tile.interiorEnd(); ++x) {
                        // Exactly kReachable: in the domain, and nothing
                        // observed it. Anything else is either outside the
                        // question or was seen by something.
                        if (tile.state[tile.index(x, y, z)] != carve::kReachable) continue;
                        if (!opt.solid && !touchesVisible(tile, x, y, z)) continue;
                        double c[3];
                        tile.centre(x, y, z, p.voxelSize, c);
                        int64_t gi[3];
                        tile.globalIndex(x, y, z, gi);
                        w.col.add(c, gi);
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
    if (perImage < (32ull << 20))
        note += fmt("range images binned to %llu cells each; ",
                    (unsigned long long)perImage);
    if (out.scansSkipped)
        note += fmt("%llu scan(s) skipped for want of a sampling grid; ",
                    (unsigned long long)out.scansSkipped);
    if (out.keptFraction < 1.0)
        note += fmt("showing %.1f%% of %llu frontier voxels (display cap); ",
                    100.0 * out.keptFraction, (unsigned long long)out.qualified);
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
