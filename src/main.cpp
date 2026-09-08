// e57cov — E57 Coverage Checker.
//
// Only the `info` subcommand exists so far. It is deliberately the first thing
// built, because the visibility pipeline cannot be written correctly until two
// questions are answered about the actual corpus, and both are properties of
// the files rather than of the algorithm:
//
//   1. How does each file represent a ray that returned nothing? Explicit
//      invalid points, or simply absent data? DESIGN.md §4 — the whole
//      NO_RETURN / OUTSIDE_FOV distinction, and therefore whether the tool
//      carves a cone through the floor under every tripod, hangs on this.
//   2. Are the points in scanner-local coordinates with a meaningful pose, or
//      already transformed into a global frame?
//
// `info` also runs the two checks that would catch a mis-decoded bit stream:
// decoded record count against the file's declared recordCount, and decoded
// bounds against the file's own cartesianBounds. See README.md — the reader
// has not yet been validated against real scanner output, and these are the
// checks that would reveal it.

#include "carve.h"
#include "e57.h"
#include "frame.h"
#include "range_image.h"

#include <chrono>
#include <memory>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

const char* typeName(e57::FieldType t) {
    switch (t) {
    case e57::FieldType::Integer:       return "Integer";
    case e57::FieldType::ScaledInteger: return "ScaledInteger";
    case e57::FieldType::FloatSingle:   return "Float(single)";
    case e57::FieldType::FloatDouble:   return "Float(double)";
    case e57::FieldType::String:        return "String";
    }
    return "?";
}

struct Stats {
    uint64_t decoded   = 0;
    uint64_t invalid   = 0;   // records flagged as no-return
    double   minX =  1e300, minY =  1e300, minZ =  1e300;
    double   maxX = -1e300, maxY = -1e300, maxZ = -1e300;
    double   sumX = 0, sumY = 0, sumZ = 0;
    bool     any = false;
};

void accumulate(Stats& st, double x, double y, double z) {
    st.minX = std::min(st.minX, x); st.maxX = std::max(st.maxX, x);
    st.minY = std::min(st.minY, y); st.maxY = std::max(st.maxY, y);
    st.minZ = std::min(st.minZ, z); st.maxZ = std::max(st.maxZ, z);
    st.sumX += x; st.sumY += y; st.sumZ += z;
    st.any = true;
}

int info(const std::string& path, bool verifyCrc, double maxRange) {
    e57::Reader r;
    std::string err;
    if (!r.open(path, err)) {
        std::printf("%s: ERROR %s\n", path.c_str(), err.c_str());
        return 1;
    }

    std::printf("%s\n", path.c_str());
    if (verifyCrc) {
        std::printf("  CRC: %s\n", r.verifyCrc(err) ? "all pages OK" : err.c_str());
    }
    std::printf("  scans: %zu\n", r.scanCount());

    int failures = 0;
    for (size_t i = 0; i < r.scanCount(); ++i) {
        const e57::Scan& s = r.scan(i);
        std::printf("\n  [%zu] %s\n", i, s.name.empty() ? "(unnamed)" : s.name.c_str());
        std::printf("      records   : %llu\n", (unsigned long long)s.recordCount);
        if (s.hasPose)
            std::printf("      pose      : t=(%.4f, %.4f, %.4f)  q=(%.6f, %.6f, %.6f, %.6f)\n",
                        s.pose.t[0], s.pose.t[1], s.pose.t[2],
                        s.pose.q[0], s.pose.q[1], s.pose.q[2], s.pose.q[3]);
        else
            std::printf("      pose      : absent (identity)\n");

        if (s.hasIndexBounds)
            std::printf("      structure : rows %lld..%lld, cols %lld..%lld\n",
                        (long long)s.rowMin, (long long)s.rowMax,
                        (long long)s.colMin, (long long)s.colMax);
        else
            std::printf("      structure : no indexBounds — grid must be recovered from data\n");

        std::printf("      prototype :");
        for (const auto& f : s.proto) {
            std::printf(" %s:%s", f.name.c_str(), typeName(f.type));
            if (f.isPacked()) std::printf("(%db)", f.bits);
        }
        std::printf("\n");

        // Pick the position fields this file actually carries.
        std::vector<std::string> want;
        bool cartesian = s.field("cartesianX") && s.field("cartesianY") && s.field("cartesianZ");
        bool spherical = s.field("sphericalRange");
        if (cartesian)      want = {"cartesianX", "cartesianY", "cartesianZ"};
        else if (spherical) want = {"sphericalRange"};
        else {
            std::printf("      DECODE    : skipped — no cartesian or spherical position fields\n");
            continue;
        }

        // The invalid-state field is the direct answer to question 1.
        const char* invName = nullptr;
        if      (s.field("cartesianInvalidState")) invName = "cartesianInvalidState";
        else if (s.field("sphericalInvalidState")) invName = "sphericalInvalidState";
        const size_t invIdx = want.size();
        if (invName) want.push_back(invName);

        Stats st;
        bool ok = r.readPoints(i, want, [&](const e57::PointBlock& b) {
            for (size_t k = 0; k < b.count; ++k) {
                const bool bad = invName && b.columns[invIdx][k] != 0.0;
                if (bad) { ++st.invalid; }
                else if (cartesian) {
                    accumulate(st, b.columns[0][k], b.columns[1][k], b.columns[2][k]);
                }
            }
            st.decoded += b.count;
            return true;
        }, err);

        if (!ok) {
            std::printf("      DECODE    : FAILED — %s\n", err.c_str());
            ++failures;
            continue;
        }

        // Check 1: a drifting bit cursor almost always truncates or overruns.
        const bool countOk = (st.decoded == s.recordCount);
        std::printf("      decoded   : %llu / %llu  %s\n",
                    (unsigned long long)st.decoded, (unsigned long long)s.recordCount,
                    countOk ? "OK" : "*** MISMATCH ***");
        if (!countOk) ++failures;

        // Question 1: how are no-returns represented? A declared sampling grid
        // answers it outright, and supersedes anything the invalid-state field
        // does or does not say.
        const bool hasGrid = s.hasIndexBounds && s.field("rowIndex") && s.field("columnIndex");
        if (invName) {
            const double pct = s.recordCount ? 100.0 * double(st.invalid) / double(s.recordCount) : 0.0;
            std::printf("      no-return : %s present — %llu of %llu records (%.2f%%)\n",
                        invName, (unsigned long long)st.invalid,
                        (unsigned long long)s.recordCount, pct);
            if (st.invalid == 0)
                std::printf("                  field present but never set: no misses are stored as records\n");
        } else {
            std::printf("      no-return : no invalid-state field — no misses are stored as records\n");
        }
        if (hasGrid) {
            std::printf("                  the declared grid identifies them exactly: every cell\n"
                        "                  with no record is a ray that came back empty\n");
        } else {
            std::printf("                  and without a declared grid they cannot be identified\n"
                        "                  exactly — see DESIGN.md §4\n");
        }

        if (cartesian && st.any) {
            std::printf("      bounds    : x[%.3f, %.3f] y[%.3f, %.3f] z[%.3f, %.3f]\n",
                        st.minX, st.maxX, st.minY, st.maxY, st.minZ, st.maxZ);
            // Check 2: an independent statement of the same extent.
            if (s.hasCartesianBounds) {
                const double tol = 1e-3;
                const bool within = st.minX >= s.xMin - tol && st.maxX <= s.xMax + tol &&
                                    st.minY >= s.yMin - tol && st.maxY <= s.yMax + tol &&
                                    st.minZ >= s.zMin - tol && st.maxZ <= s.zMax + tol;
                std::printf("      declared  : x[%.3f, %.3f] y[%.3f, %.3f] z[%.3f, %.3f]  %s\n",
                            s.xMin, s.xMax, s.yMin, s.yMax, s.zMin, s.zMax,
                            within ? "OK (decoded within declared)" : "*** OUTSIDE DECLARED BOUNDS ***");
                if (!within) ++failures;
            } else {
                std::printf("      declared  : no cartesianBounds — cross-check unavailable\n");
            }
            // The range image is what the visibility pass consumes, so report
            // what this scan would actually yield: how many rays came back
            // empty, and whether the raster is regular enough to look up.
            if (hasGrid) {
                rimg::RangeImage img;
                rimg::Options ro;
                ro.maxRange = maxRange;
                std::string rerr;
                if (rimg::build(r, i, ro, img, rerr)) {
                    std::printf("      grid      : %u x %u = %.2f M cells, %.1f%% filled\n",
                                img.rows, img.cols,
                                double(img.cellCount()) / 1e6, 100.0 * img.diag.fillFraction);
                    std::printf("      rays      : %llu returns, %llu no-returns "
                                "(these are what clear space)\n",
                                (unsigned long long)img.diag.hits,
                                (unsigned long long)img.diag.noReturns);
                    std::printf("      range     : returns from %.2f m to %.2f m; "
                                "no-returns clear to %.0f m\n",
                                img.diag.nearestReturn, img.diag.furthestReturn, ro.maxRange);
                    if (img.diag.furthestReturn > ro.maxRange * 1.05) {
                        std::printf("                  note: returns reach past --max-range, so some "
                                    "measured\n                  surfaces sit beyond where "
                                    "no-return rays stop clearing\n");
                    }
                    std::printf("      raster    : %s (residuals %.5f rad row, %.5f rad col)\n",
                                img.map.valid ? "uniform, lookups exact"
                                              : "*** NOT UNIFORM — lookups unreliable ***",
                                img.map.elResidualRad, img.map.azResidualRad);
                    if (!img.map.valid) ++failures;
                    if (img.diag.emptyLeadingRows || img.diag.emptyTrailingRows) {
                        // Either an all-sky band or a part of the grid the
                        // scanner never sampled. Treated as no-returns; only
                        // the corpus can say whether that is right.
                        std::printf("      note      : %u leading and %u trailing grid rows hold no\n"
                                    "                  returns. Treated as no-returns (all-sky). If\n"
                                    "                  either band is really outside the field of\n"
                                    "                  view — a nadir blind cone, say — it would\n"
                                    "                  wrongly clear space. Check against the scan.\n",
                                    img.diag.emptyLeadingRows, img.diag.emptyTrailingRows);
                    }
                } else {
                    std::printf("      grid      : range image failed — %s\n", rerr.c_str());
                    ++failures;
                }
            } else {
                std::printf("      grid      : no indexBounds + row/column index — no-return rays\n"
                            "                  cannot be identified exactly (DESIGN.md §4)\n");
            }

            // Question 2: which coordinate frame, and is the pose applied?
            const viewer::FrameDecision fd = viewer::decideFrame(r, i);
            std::printf("      frame     : %s\n", viewer::conventionName(fd.convention));
            std::printf("                  %s\n", fd.reason.c_str());
            if (fd.nonConformant()) {
                std::printf("      *** this file stores pre-transformed points WITH a non-identity\n"
                            "          pose, which contradicts ASTM E2807. The pose is being\n"
                            "          ignored for this scan; verify against a known-good viewer.\n");
                ++failures;
            }
        }
    }

    std::printf("\n");
    return failures == 0 ? 0 : 1;
}

// ---------------------------------------------------------------------------
// carve — the visibility pass, on the CPU reference.
//
// This runs the oracle, not the production path: every voxel is tested against
// every setup that can reach it, single-threaded. It is here so the reference
// can be pointed at real files rather than only at synthetic fixtures, and so
// the Metal kernel has something to be compared against on real data. On a full
// corpus at 5 cm it will be slow — the estimate is printed before it starts,
// and --max-tiles stops it after a sample.
//
// It also holds every range image in memory at once, which is what the
// production path will not do: it will stream setups per tile. For a handful of
// scans that is the difference between 200 MB and an architecture.

struct CarveOptions {
    double   voxelSize = 0.05;
    double   maxRange  = 45.0;
    uint32_t tileVoxels = 256;
    uint64_t maxTiles  = 0;      // 0 = the whole domain
    uint32_t maxCells  = 32u << 20;
};

struct CarveProgress {
    uint64_t tiles = 0;
    uint64_t limit = 0;
    uint64_t total = 0;
    std::chrono::steady_clock::time_point start;
};

bool carveProgressSink(const carve::Tile&, void* user) {
    CarveProgress* pr = static_cast<CarveProgress*>(user);
    ++pr->tiles;
    if (pr->tiles % 16 == 0 || pr->tiles == pr->total) {
        const double secs = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - pr->start).count();
        std::fprintf(stderr, "\r  tiles %llu / %llu  (%.1f s)",
                     (unsigned long long)pr->tiles, (unsigned long long)pr->total, secs);
        std::fflush(stderr);
    }
    return !(pr->limit && pr->tiles >= pr->limit);
}

int carveCorpus(const std::vector<std::string>& paths, const CarveOptions& co) {
    // The readers have to outlive the range images, and the range images have to
    // outlive the setup views, which hold pointers into them.
    std::vector<std::unique_ptr<e57::Reader>>    readers;
    std::vector<std::unique_ptr<rimg::RangeImage>> images;
    std::vector<carve::SetupView>                setups;

    rimg::Options ro;
    ro.maxRange = co.maxRange;
    ro.maxCells = co.maxCells;

    uint64_t skipped = 0;
    for (const std::string& path : paths) {
        auto r = std::make_unique<e57::Reader>();
        std::string err;
        if (!r->open(path, err)) {
            std::printf("%s: ERROR %s\n", path.c_str(), err.c_str());
            return 1;
        }
        for (size_t i = 0; i < r->scanCount(); ++i) {
            auto img = std::make_unique<rimg::RangeImage>();
            std::string rerr;
            if (!rimg::build(*r, i, ro, *img, rerr)) {
                // A scan with no usable raster cannot contribute evidence, and
                // guessing one would invent visibility. Skipped and counted.
                std::printf("  skipped %s [%zu]: %s\n", path.c_str(), i, rerr.c_str());
                ++skipped;
                continue;
            }
            images.push_back(std::move(img));
            setups.push_back(carve::makeSetupView(*images.back()));
        }
        readers.push_back(std::move(r));
    }

    if (setups.empty()) {
        std::printf("no usable setups — nothing to carve\n");
        return 1;
    }

    carve::Params p;
    p.voxelSize  = co.voxelSize;
    // Half a voxel diagonal, so it tracks the voxel size rather than being a
    // constant that silently stops matching it.
    p.surfaceMargin = 0.5 * co.voxelSize * 1.7320508075688772;
    p.maxRange   = co.maxRange;
    p.tileVoxels = co.tileVoxels;

    std::printf("\nsetups    : %zu (%llu scan(s) skipped)\n",
                setups.size(), (unsigned long long)skipped);
    for (size_t i = 0; i < setups.size() && i < 8; ++i) {
        std::printf("  [%zu] origin (%.3f, %.3f, %.3f)  %llu returns, %llu no-returns\n",
                    i, setups[i].origin[0], setups[i].origin[1], setups[i].origin[2],
                    (unsigned long long)setups[i].image->diag.hits,
                    (unsigned long long)setups[i].image->diag.noReturns);
    }
    if (setups.size() > 8) std::printf("  ... and %zu more\n", setups.size() - 8);

    const std::vector<carve::TileKey> keys = carve::tilesForSetups(setups, p);
    const uint64_t perTile = uint64_t(p.tileVoxels) * p.tileVoxels * p.tileVoxels;
    std::printf("\nvoxel     : %.3f m, surface margin %.4f m\n", p.voxelSize, p.surfaceMargin);
    std::printf("tile      : %u^3 voxels = %.2f m cube, %.1f MB of state\n",
                p.tileVoxels, p.tileMetres(), double(perTile) / 1048576.0);
    std::printf("domain    : %zu tiles, %.3f G voxels\n",
                keys.size(), double(keys.size()) * double(perTile) / 1e9);
    if (co.maxTiles)
        std::printf("            stopping after %llu tiles (--max-tiles)\n",
                    (unsigned long long)co.maxTiles);

    CarveProgress pr;
    pr.limit = co.maxTiles;
    pr.total = co.maxTiles ? std::min<uint64_t>(co.maxTiles, keys.size()) : keys.size();
    pr.start = std::chrono::steady_clock::now();

    const carve::Stats st = carve::carveAll(setups, p, carveProgressSink, &pr);
    const double secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - pr.start).count();
    std::fprintf(stderr, "\n");

    const double voxelVolume = p.voxelSize * p.voxelSize * p.voxelSize;
    const double pct = st.reachable ? 100.0 / double(st.reachable) : 0.0;
    std::printf("\nresult    : %.1f s, %.2f M voxel-setup tests\n",
                secs, double(st.setupTests) / 1e6);
    std::printf("  examined  %llu voxels in %llu tiles\n",
                (unsigned long long)st.voxels, (unsigned long long)pr.tiles);
    std::printf("  reachable %llu  (%.1f m^3) — within %.0f m of some setup\n",
                (unsigned long long)st.reachable,
                double(st.reachable) * voxelVolume, p.maxRange);
    std::printf("  visible   %llu  (%.1f%%) — some setup had line of sight\n",
                (unsigned long long)st.visible, double(st.visible) * pct);
    std::printf("  occupied  %llu  (%.1f%%) — some setup measured a surface\n",
                (unsigned long long)st.occupied, double(st.occupied) * pct);
    std::printf("  unknown   %llu  (%.1f%%, %.1f m^3) — neither: the candidate voids\n",
                (unsigned long long)st.unknown, double(st.unknown) * pct,
                double(st.unknown) * voxelVolume);
    std::printf("\nThe unknown set still mixes three things: shadows inside the site,\n"
                "material behind measured surfaces, and space outside the building\n"
                "that no setup could ever see. Separating them is the next stage\n"
                "(DESIGN.md) — this number is not yet the answer.\n");
    return 0;
}

void usage() {
    std::printf(
        "e57cov — E57 Coverage Checker\n"
        "\n"
        "usage: e57cov info  [--crc] [--max-range <m>] <file.e57> [more.e57 ...]\n"
        "       e57cov carve [options] <file.e57> [more.e57 ...]\n"
        "\n"
        "  info    Inspect scans and audit format conventions. Reports how each\n"
        "          file represents no-return rays and which coordinate frame its\n"
        "          points are in, and cross-checks the decode against the file's\n"
        "          own recordCount and cartesianBounds.\n"
        "\n"
        "  carve   Run the visibility pass and report how much space the setups\n"
        "          actually observed. This is the CPU reference: correct, single\n"
        "          threaded, and slow — it exists to be the oracle the GPU path\n"
        "          is checked against. Use --max-tiles to sample a large site.\n"
        "\n"
        "  --crc   (info) Also verify every page checksum (costs a full pass).\n"
        "  --max-range <m>\n"
        "          How far a no-return ray clears, and how far any setup's\n"
        "          evidence reaches. Default 45 m, the scanner's rated maximum.\n"
        "  --voxel <m>\n"
        "          (carve) Voxel edge. Default 0.05 m. The surface margin is\n"
        "          derived from it as half a voxel diagonal.\n"
        "  --tile <voxels>\n"
        "          (carve) Voxels per tile edge. Default 256. Affects working\n"
        "          set and nothing else — the answer is identical either way.\n"
        "  --max-tiles <n>\n"
        "          (carve) Stop after n tiles. Default 0, the whole domain.\n");
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 2; }

    const std::string cmd = argv[1];
    if (cmd == "-h" || cmd == "--help" || cmd == "help") { usage(); return 0; }
    if (cmd != "info" && cmd != "carve") {
        std::printf("unknown command '%s'\n\n", cmd.c_str());
        usage();
        return 2;
    }

    bool                     crc = false;
    CarveOptions             co;
    std::vector<std::string> paths;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--crc") == 0) { crc = true; continue; }
        if (std::strcmp(argv[i], "--max-range") == 0 && i + 1 < argc) {
            co.maxRange = std::strtod(argv[++i], nullptr);
            if (!(co.maxRange > 0.0)) { std::printf("--max-range must be positive\n"); return 2; }
            continue;
        }
        if (std::strcmp(argv[i], "--voxel") == 0 && i + 1 < argc) {
            co.voxelSize = std::strtod(argv[++i], nullptr);
            if (!(co.voxelSize > 0.0)) { std::printf("--voxel must be positive\n"); return 2; }
            continue;
        }
        if (std::strcmp(argv[i], "--tile") == 0 && i + 1 < argc) {
            const long v = std::strtol(argv[++i], nullptr, 10);
            if (v <= 0 || v > 4096) { std::printf("--tile must be in 1..4096\n"); return 2; }
            co.tileVoxels = uint32_t(v);
            continue;
        }
        if (std::strcmp(argv[i], "--max-tiles") == 0 && i + 1 < argc) {
            co.maxTiles = std::strtoull(argv[++i], nullptr, 10);
            continue;
        }
        paths.push_back(argv[i]);
    }
    if (paths.empty()) { usage(); return 2; }

    if (cmd == "carve") return carveCorpus(paths, co);

    int failures = 0;
    for (const auto& p : paths) failures += info(p, crc, co.maxRange);
    if (failures)
        std::printf("%d file(s) reported problems.\n", failures);
    return failures == 0 ? 0 : 1;
}
