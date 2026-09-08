#include "visibility.h"

#include "e57.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <memory>

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
struct Collector {
    std::vector<lod::StorePoint> out;
    uint64_t cap = 0;
    uint64_t threshold = ~0ull;   // keep when voxelHash < threshold
    uint64_t qualified = 0;
    double   origin[3] = {0, 0, 0};
    bool     haveOrigin = false;

    void add(const double centre[3], const int64_t gi[3]) {
        ++qualified;
        const uint64_t h = voxelHash(gi[0], gi[1], gi[2]);
        if (h >= threshold) return;

        if (!haveOrigin) {
            for (int k = 0; k < 3; ++k) origin[k] = centre[k];
            haveOrigin = true;
        }
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

    // Halving the threshold drops about half the kept set, and re-testing what
    // is already held keeps the invariant exact: whatever order voxels arrived
    // in, and however many times the threshold has moved, the result is always
    // "every voxel whose hash is below the current threshold". That is what
    // makes the same site draw the same way at any tile size.
    void halve() {
        threshold /= 2;
        size_t w = 0;
        for (size_t i = 0; i < out.size(); ++i) {
            if (keys[i] < threshold) { out[w] = out[i]; keys[w] = keys[i]; ++w; }
        }
        out.resize(w);
        keys.resize(w);
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

    const std::vector<carve::TileKey> keys = carve::tilesForSetups(setups, p);
    out.tilesTotal = keys.size();

    Collector col;
    col.cap = opt.displayCap;

    carve::Tile tile;
    for (size_t t = 0; t < keys.size(); ++t) {
        if (opt.maxTiles && out.tilesCarved >= opt.maxTiles) { out.partial = true; break; }
        carve::carveTile(keys[t], setups, p, tile, out.stats);
        ++out.tilesCarved;

        for (uint32_t z = tile.interiorBegin(); z < tile.interiorEnd(); ++z) {
            for (uint32_t y = tile.interiorBegin(); y < tile.interiorEnd(); ++y) {
                for (uint32_t x = tile.interiorBegin(); x < tile.interiorEnd(); ++x) {
                    // Exactly kReachable: in the domain, and nothing observed
                    // it. Anything else is either outside the question or was
                    // seen by something.
                    if (tile.state[tile.index(x, y, z)] != carve::kReachable) continue;
                    if (!opt.solid && !touchesVisible(tile, x, y, z)) continue;
                    double c[3];
                    tile.centre(x, y, z, p.voxelSize, c);
                    int64_t gi[3];
                    tile.globalIndex(x, y, z, gi);
                    col.add(c, gi);
                }
            }
        }

        if (!tick("carving", out.tilesCarved, keys.size())) {
            out.cancelled = true;
            out.partial = true;
            break;
        }
    }

    out.voxels = std::move(col.out);
    for (int k = 0; k < 3; ++k) out.origin[k] = col.origin[k];
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
    if (out.partial)
        note += out.cancelled ? "cancelled part way; " : "stopped at the tile limit; ";
    out.note = note;
    return true;
}

} // namespace vis
