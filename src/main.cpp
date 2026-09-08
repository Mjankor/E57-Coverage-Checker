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
#include "visibility.h"

#include <chrono>
#include <memory>
#include <thread>

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
// carve — the visibility pass.
//
// The pipeline itself lives in src/visibility.{h,cpp} so the CLI and the app
// run exactly the same code. This is the terminal's view of it: parameters in,
// numbers out, and a progress line that can be watched on a long run.
//
// It runs the CPU reference — every voxel tested against every setup that can
// reach it, single-threaded — because that is the oracle the Metal kernel will
// be asserted bit-exact against. On a full corpus at 5 cm it will be slow; the
// domain size is printed before it starts and --max-tiles stops it after a
// sample.

int carveCorpus(const std::vector<std::string>& paths, const vis::Options& opt) {
    const auto start = std::chrono::steady_clock::now();
    uint64_t lastTiles = 0;

    vis::Progress progress = [&](const std::string& stage, uint64_t done, uint64_t total) {
        // Every tile on a big run would be thousands of lines of scrollback.
        if (stage == "carving" && done != total && done - lastTiles < 16) return true;
        lastTiles = done;
        const double secs = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        std::fprintf(stderr, "\r  %-22s %llu / %llu  (%.1f s)          ",
                     stage.c_str(), (unsigned long long)done,
                     (unsigned long long)total, secs);
        std::fflush(stderr);
        return true;
    };

    vis::Result res;
    std::string err;
    if (!vis::run(paths, opt, progress, res, err)) {
        std::fprintf(stderr, "\n");
        std::printf("carve failed: %s\n", err.c_str());
        return 1;
    }
    std::fprintf(stderr, "\n");

    const double secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    const carve::Stats& st = res.stats;
    const double voxelVolume = opt.voxelSize * opt.voxelSize * opt.voxelSize;
    const double pct = st.reachable ? 100.0 / double(st.reachable) : 0.0;

    std::printf("\nsetups    : %llu used, %llu skipped\n",
                (unsigned long long)res.setupsUsed, (unsigned long long)res.scansSkipped);
    std::printf("voxel     : %.3f m   ·   tile %u^3   ·   max range %.0f m   ·   %u thread(s)\n",
                opt.voxelSize, opt.tileVoxels, opt.maxRange,
                opt.threads ? opt.threads : std::thread::hardware_concurrency());
    if (res.domain.kind == carve::Domain::Kind::Box) {
        std::printf("region    : surveyed extent  x[%.1f, %.1f] y[%.1f, %.1f] z[%.1f, %.1f]\n"
                    "            %.0f m^3, against %.0f m^3 of range spheres (%.0fx smaller)\n",
                    res.domain.lo[0], res.domain.hi[0], res.domain.lo[1], res.domain.hi[1],
                    res.domain.lo[2], res.domain.hi[2], res.domainVolume, res.sphereVolume,
                    res.domainVolume > 0 ? res.sphereVolume / res.domainVolume : 0.0);
    } else {
        std::printf("region    : full range spheres, %.0f m^3 — mostly open air\n",
                    res.sphereVolume);
    }
    std::printf("domain    : %llu tiles, %llu carved%s\n",
                (unsigned long long)res.tilesTotal, (unsigned long long)res.tilesCarved,
                res.partial ? "  (stopped early — the numbers below are a sample)" : "");

    std::printf("\nresult    : %.1f s, %.2f M voxel-setup tests\n",
                secs, double(st.setupTests) / 1e6);
    std::printf("  examined  %llu voxels\n", (unsigned long long)st.voxels);
    std::printf("  reachable %llu  (%.1f m^3) — within %.0f m of some setup\n",
                (unsigned long long)st.reachable,
                double(st.reachable) * voxelVolume, opt.maxRange);
    std::printf("  visible   %llu  (%.1f%%) — some setup had line of sight\n",
                (unsigned long long)st.visible, double(st.visible) * pct);
    std::printf("  occupied  %llu  (%.1f%%) — some setup measured a surface\n",
                (unsigned long long)st.occupied, double(st.occupied) * pct);
    std::printf("  unknown   %llu  (%.1f%%, %.1f m^3) — neither: the candidate voids\n",
                (unsigned long long)st.unknown, double(st.unknown) * pct,
                double(st.unknown) * voxelVolume);
    std::printf("\ndrawable  : %zu voxels%s%s\n", res.voxels.size(),
                opt.solid ? " (solid)" : " on the observed frontier",
                res.keptFraction < 1.0 ? ", sampled to fit the display cap" : "");
    if (!res.note.empty()) std::printf("note      : %s\n", res.note.c_str());

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
        "          (carve) Voxels per tile edge. Default 128. Affects working\n"
        "          set and nothing else — the answer is identical either way.\n"
        "  --max-tiles <n>\n"
        "          (carve) Stop after n tiles. Default 0, the whole domain.\n"
        "  --domain extent|spheres\n"
        "          (carve) Which region to ask about. 'extent' (default) is a box\n"
        "          around what the scans actually returned, grown by\n"
        "          --domain-margin; 'spheres' is everything within --max-range of\n"
        "          any setup. A building interior scanned from inside fills only a\n"
        "          small part of its range spheres, so 'spheres' spends most of the\n"
        "          run, and most of the answer, on open air.\n"
        "  --domain-margin <m>\n"
        "          (carve) How far past the last return the question still applies.\n"
        "          Default 2 m: wall thickness, eaves and registration slop.\n"
        "  --threads <n>\n"
        "          (carve) Worker threads over the tile list. Default 0, the\n"
        "          machine's count. The answer is identical at any count.\n"
        "  --solid (carve) Keep every unknown voxel rather than only those on the\n"
        "          frontier with observed space. Far more voxels, same answer:\n"
        "          an opaque volume hides its own interior anyway.\n");
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
    vis::Options             co;
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
        if (std::strcmp(argv[i], "--solid") == 0) { co.solid = true; continue; }
        if (std::strcmp(argv[i], "--domain") == 0 && i + 1 < argc) {
            const std::string v = argv[++i];
            if      (v == "extent")  co.domain = vis::DomainMode::MeasuredExtent;
            else if (v == "spheres") co.domain = vis::DomainMode::RangeSpheres;
            else { std::printf("--domain must be 'extent' or 'spheres'\n"); return 2; }
            continue;
        }
        if (std::strcmp(argv[i], "--domain-margin") == 0 && i + 1 < argc) {
            co.domainMargin = std::strtod(argv[++i], nullptr);
            if (co.domainMargin < 0.0) { std::printf("--domain-margin must not be negative\n"); return 2; }
            continue;
        }
        if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            const long v = std::strtol(argv[++i], nullptr, 10);
            if (v < 0 || v > 1024) { std::printf("--threads must be in 0..1024\n"); return 2; }
            co.threads = uint32_t(v);
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
