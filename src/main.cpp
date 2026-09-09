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
#include "version.h"
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
                    if (img.diag.isolatedNoReturns) {
                        const double pct = 100.0 * double(img.diag.isolatedNoReturns) /
                                           double(img.diag.isolatedNoReturns + img.diag.noReturns);
                        std::printf("      drops     : %llu empty cells (%.1f%% of them) had too "
                                    "few empty\n                  neighbours to be sky and clear "
                                    "nothing. Believed, each\n                  would have cleared "
                                    "a line to %.0f m through solid geometry.\n",
                                    (unsigned long long)img.diag.isolatedNoReturns, pct,
                                    ro.maxRange);
                    }
                    std::printf("      range     : returns from %.2f m to %.2f m; "
                                "no-returns clear to %.0f m\n",
                                img.diag.nearestReturn, img.diag.furthestReturn, ro.maxRange);
                    if (img.diag.furthestReturn > ro.maxRange * 1.05) {
                        std::printf("                  note: returns reach past --max-range, so some "
                                    "measured\n                  surfaces sit beyond where "
                                    "no-return rays stop clearing\n");
                    }
                    std::printf("      raster    : %s\n"
                                "                  residuals %.5f rad row, %.5f rad col\n",
                                img.map.valid ? "uniform"
                                              : "*** REFUSED — lookups would reach the wrong "
                                                "direction ***",
                                img.map.elResidualRad, img.map.azResidualRad);
                    // The mapping itself, and the field of view it implies. A
                    // vertical span near 360 degrees means the mirror covers
                    // each column twice and a single line cannot describe it.
                    std::printf("      mapping   : el = %+.6f %+.8f * row   "
                                "(%.1f deg over %u rows)\n"
                                "                  az = %+.6f %+.8f * col   "
                                "(%.1f deg over %u cols)\n",
                                img.map.el0, img.map.dElPerRow,
                                std::fabs(img.map.dElPerRow) * img.rows * 57.29577951308232,
                                img.rows,
                                img.map.az0, img.map.dAzPerCol,
                                std::fabs(img.map.dAzPerCol) * img.cols * 57.29577951308232,
                                img.cols);
                    // The check that matters: the scan's own points put back
                    // through the mapping. A low figure here with low residuals
                    // means lookups land in the wrong place while the fit looks
                    // healthy, which is how sky comes back unobserved and
                    // building interiors come back clear.
                    if (img.map.roundTripFraction >= 0.0) {
                        const double rt = 100.0 * img.map.roundTripFraction;
                        std::printf("      round trip: %.2f%% of this scan's own points land back "
                                    "on their own cell  %s\n", rt,
                                    img.map.roundTripFraction >= 0.90 ? "OK"
                                                                      : "*** BROKEN ***");
                    } else {
                        std::printf("      round trip: not measured\n");
                    }
                    if (img.diag.blindConeRows) {
                        const double here = img.diag.blindConeAtFirstRow
                                          ? img.diag.borderRangeFirst : img.diag.borderRangeLast;
                        const double there = img.diag.blindConeAtFirstRow
                                          ? img.diag.borderRangeLast : img.diag.borderRangeFirst;
                        std::printf("      blind cone: %u unsampled rows at the %s of the raster "
                                    "(%llu cells)\n"
                                    "                  bordering returns %.2f m here; ",
                                    img.diag.blindConeRows,
                                    img.diag.blindConeAtFirstRow ? "start" : "end",
                                    (unsigned long long)img.diag.blindConeCells, here);
                        if (there >= 0) std::printf("%.2f m at the other end\n", there);
                        else            std::printf("no unsampled band at the other end\n");
                        if (img.diag.hasConeAxis) {
                            std::printf("      mounting  : cone axis (%.3f, %.3f, %.3f) — %s\n",
                                        img.diag.coneAxisWorld[0], img.diag.coneAxisWorld[1],
                                        img.diag.coneAxisWorld[2],
                                        img.diag.coneAxisWorld[2] > 0.5 ? "*** INVERTED ***"
                                      : img.diag.coneAxisWorld[2] < -0.5 ? "upright"
                                                                         : "on its side");
                        }
                    } else {
                        std::printf("      blind cone: none identified — every empty cell is "
                                    "treated as a no-return\n");
                    }
                    if (img.diag.outsideGrid)
                        std::printf("      off-grid  : %llu points fell outside the declared "
                                    "indexBounds\n",
                                    (unsigned long long)img.diag.outsideGrid);
                    if (!img.diag.originInsideReturns)
                        std::printf("      *** the scanner sits outside the box of its own "
                                    "returns — check the frame\n");
                    if (!img.diag.note.empty())
                        std::printf("      note      : %s\n", img.diag.note.c_str());
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
    if (res.setupsWithoutMapping)
        std::printf("            *** %llu of them contribute NOTHING: their angular mapping\n"
                    "                was refused, so every lookup falls outside the raster\n",
                    (unsigned long long)res.setupsWithoutMapping);
    if (res.setupsWithBlindCone)
        std::printf("            %llu with an identified blind cone, %llu of those inverted\n",
                    (unsigned long long)res.setupsWithBlindCone,
                    (unsigned long long)res.setupsInverted);
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
    const bool bounds = opt.earlyOut == carve::EarlyOut::AnyEvidence;
    std::printf("  visible   %llu  (%.1f%%) — some setup had line of sight%s\n",
                (unsigned long long)st.visible, double(st.visible) * pct,
                bounds ? "   [lower bound]" : "");
    std::printf("  occupied  %llu  (%.1f%%) — some setup measured a surface%s\n",
                (unsigned long long)st.occupied, double(st.occupied) * pct,
                bounds ? "   [lower bound]" : "");
    std::printf("  unknown   %llu  (%.1f%%, %.1f m^3) — neither: nobody observed it\n",
                (unsigned long long)st.unknown, double(st.unknown) * pct,
                double(st.unknown) * voxelVolume);

    if (res.classified) {
        const voids::Report& v = res.voidReport;
        std::printf("\nvoids     : %llu enclosed void(s), %.1f m^3 in total\n"
                    "            largest %.1f m^3   ·   %.1f m^3 of the unobserved space\n"
                    "            reaches the outside world and is not a finding\n",
                    (unsigned long long)v.components, res.enclosedVolume(),
                    double(v.largestComponent) * voxelVolume, res.exteriorVolume());
    } else if (!res.classifySkipped.empty()) {
        std::printf("\nvoids     : not classified — %s\n", res.classifySkipped.c_str());
    }
    std::printf("\ndrawable  : %zu voxels%s%s\n", res.voxels.size(),
                opt.solid ? " (solid)" : " on the observed frontier",
                res.keptFraction < 1.0 ? ", sampled to fit the display cap" : "");
    if (!res.note.empty()) std::printf("note      : %s\n", res.note.c_str());

    if (!res.classified) {
        std::printf("\nWithout the connectivity pass the unknown figure mixes three things:\n"
                    "shadows inside the site, material behind measured surfaces, and space\n"
                    "outside the building that no setup could ever see. Over any region\n"
                    "containing a building the third dominates, so read `unknown` as\n"
                    "\"everything nobody looked at\" rather than as a finding.\n");
    }
    return 0;
}

// ---------------------------------------------------------------------------
// probe — why does one point come out the way it does?
//
// Aggregate figures cannot answer "the house interior is being cleared, why?".
// This can: it names the setup that cleared it, the direction, the raster cell
// that direction lands on, and what that cell holds. Every wrong answer this
// tool has produced was diagnosable from those five things, and none of them was
// visible from the statistics.

int probePoint(const std::vector<std::string>& paths, const vis::Options& opt,
               const double world[3]) {
    std::vector<std::unique_ptr<e57::Reader>> readers;
    std::vector<std::unique_ptr<rimg::RangeImage>> images;
    std::vector<carve::SetupView> setups;

    rimg::Options ro;
    ro.maxRange         = opt.maxRange;
    ro.noReturnRadius   = opt.skyRadius;
    ro.noReturnFraction = opt.skyFraction;

    for (const std::string& path : paths) {
        auto r = std::make_unique<e57::Reader>();
        std::string err;
        if (!r->open(path, err)) { std::printf("%s: %s\n", path.c_str(), err.c_str()); return 1; }
        for (size_t i = 0; i < r->scanCount(); ++i) {
            auto img = std::make_unique<rimg::RangeImage>();
            std::string rerr;
            if (!rimg::build(*r, i, ro, *img, rerr)) continue;
            images.push_back(std::move(img));
            setups.push_back(carve::makeSetupView(*images.back()));
        }
        readers.push_back(std::move(r));
    }
    if (setups.empty()) { std::printf("no usable scans\n"); return 1; }

    carve::Params p;
    p.voxelSize     = opt.voxelSize;
    p.surfaceMargin = 0.5 * opt.voxelSize * 1.7320508075688772;
    p.maxRange      = opt.maxRange;

    std::printf("probe (%.3f, %.3f, %.3f)   voxel %.3f m   max range %.1f m\n\n",
                world[0], world[1], world[2], p.voxelSize, p.maxRange);

    uint8_t total = 0;
    for (size_t i = 0; i < setups.size(); ++i) {
        const carve::SetupView& s = setups[i];
        const rimg::RangeImage& im = *s.image;

        const double dx = world[0] - s.origin[0];
        const double dy = world[1] - s.origin[1];
        const double dz = world[2] - s.origin[2];
        const double dist = std::sqrt(dx * dx + dy * dy + dz * dz);

        std::printf("  setup %zu at (%.3f, %.3f, %.3f)   %.2f m away\n",
                    i, s.origin[0], s.origin[1], s.origin[2], dist);
        if (dist > p.maxRange) {
            std::printf("      out of range — contributes nothing\n\n");
            continue;
        }

        double x = world[0], y = world[1], z = world[2];
        s.worldToScanner.apply(x, y, z);
        double az, el, r;
        rimg::toSpherical(x, y, z, az, el, r);
        std::printf("      direction   az %7.3f rad (%7.2f deg)   el %7.3f rad (%7.2f deg)\n",
                    az, az * 57.29577951308232, el, el * 57.29577951308232);
        std::printf("      raster      row %.2f of %u   col %.2f of %u\n",
                    im.rowCoord(el), im.rows, im.colCoord(az), im.cols);

        uint32_t row, col;
        if (!im.cellOf(az, el, row, col)) {
            std::printf("      cell        NONE — this direction is off the raster%s\n",
                        im.map.valid ? "" : " (the mapping was refused)");
            std::printf("      verdict     says nothing\n\n");
            continue;
        }
        const rimg::Status st = im.statusAt(row, col);
        const double surface  = im.rangeAt(row, col);
        std::printf("      cell        [%u, %u]  %s", row, col, rimg::statusName(st));
        if (st == rimg::Status::Hit)      std::printf("  surface at %.3f m", surface);
        if (st == rimg::Status::NoReturn) std::printf("  clears to %.3f m", surface);
        std::printf("\n");

        const uint8_t bits = carve::evidenceAt(s, p, world[0], world[1], world[2]);
        total |= bits;
        std::printf("      verdict     %s\n\n",
                    (bits & carve::kVisible)  ? "VISIBLE — this setup saw through it"
                  : (bits & carve::kOccupied) ? "OCCUPIED — a surface is measured here"
                                              : "says nothing");
    }

    std::printf("combined    : %s\n",
                (total & carve::kVisible) ? "VISIBLE"
              : (total & carve::kOccupied) ? "OCCUPIED"
                                           : "UNOBSERVED — no setup said anything about it");
    return 0;
}

void usage() {
    std::printf(
        "e57cov — E57 Coverage Checker\n"
        "\n"
        "usage: e57cov info  [--crc] [--max-range <m>] <file.e57> [more.e57 ...]\n"
        "       e57cov carve [options] <file.e57> [more.e57 ...]\n"
        "       e57cov probe <x> <y> <z> <file.e57> [more.e57 ...]\n"
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
        "  probe   Explain one point: for every setup, the direction to it, the\n"
        "          raster cell that direction lands on, what that cell holds, and\n"
        "          the verdict. This is the tool for \"why is the inside of the\n"
        "          house being cleared?\" — a question no aggregate can answer.\n"
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
        "  --early-out none|saturated|any\n"
        "          (carve) When a voxel stops asking further setups. 'saturated'\n"
        "          (default) stops once it has both evidence bits, which is exact.\n"
        "          'any' stops at the first bit: still exact for the unknown set,\n"
        "          but visible and occupied become lower bounds. 'none' asks every\n"
        "          setup, matching the reference's work exactly.\n"
        "  --sky-radius <cells>   --sky-fraction <0..1>\n"
        "          (carve) How much company an empty cell needs before it is\n"
        "          believed to have seen sky rather than dropped a return.\n"
        "          Default radius 2 (a 5x5 window) and 0.75 of it. Nothing in an\n"
        "          E57 distinguishes the two, and believing a dropped return\n"
        "          clears a pencil of space to --max-range through solid\n"
        "          geometry. Raise the fraction if a scan drops heavily; set the\n"
        "          radius to 0 to believe every empty cell, as before.\n"
        "  --classify\n"
        "          (carve) Keep only unobserved space you cannot reach from\n"
        "          outside without crossing observed space. Off by default: it\n"
        "          also excludes a building interior whose walls were only ever\n"
        "          seen from one side, which is usually the space you wanted.\n"
        "  --blind-cone auto|none|first|last\n"
        "          (carve) Which end of each raster holds the instrument's own\n"
        "          blind cone, where no ray was fired. Default auto, which finds\n"
        "          it from the geometry: the returns bordering the cone are the\n"
        "          ground beside the mount, metres away, where those bordering\n"
        "          sky are distant. That works whichever way up the scanner was.\n"
        "  --threads <n>\n"
        "          (carve) Worker threads over the tile list. Default 0, the\n"
        "          machine's count. The answer is identical at any count.\n"
        "  --solid (carve) Keep every unknown voxel rather than only those on the\n"
        "          frontier with observed space. Far more voxels, same answer:\n"
        "          an opaque volume hides its own interior anyway.\n");
}

} // namespace

int main(int argc, char** argv) {
    // First line of every run. A report that cannot say which binary made it is
    // a report you cannot act on.
    std::printf("e57cov %s\n\n", ver::describe());
    if (argc < 2) { usage(); return 2; }

    const std::string cmd = argv[1];
    if (cmd == "-h" || cmd == "--help" || cmd == "help") { usage(); return 0; }
    if (cmd != "info" && cmd != "carve" && cmd != "probe") {
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
        if (std::strcmp(argv[i], "--classify") == 0) { co.classifyVoids = true; continue; }
        if (std::strcmp(argv[i], "--blind-cone") == 0 && i + 1 < argc) {
            const std::string v = argv[++i];
            if      (v == "auto")  co.blindCone = rimg::BlindCone::Auto;
            else if (v == "none")  co.blindCone = rimg::BlindCone::None;
            else if (v == "first") co.blindCone = rimg::BlindCone::FirstRows;
            else if (v == "last")  co.blindCone = rimg::BlindCone::LastRows;
            else { std::printf("--blind-cone must be auto, none, first or last\n"); return 2; }
            continue;
        }
        if (std::strcmp(argv[i], "--sky-radius") == 0 && i + 1 < argc) {
            co.skyRadius = uint32_t(std::strtoul(argv[++i], nullptr, 10));
            continue;
        }
        if (std::strcmp(argv[i], "--sky-fraction") == 0 && i + 1 < argc) {
            co.skyFraction = std::strtod(argv[++i], nullptr);
            if (co.skyFraction < 0.0 || co.skyFraction > 1.0) {
                std::printf("--sky-fraction must be in 0..1\n"); return 2;
            }
            continue;
        }
        if (std::strcmp(argv[i], "--early-out") == 0 && i + 1 < argc) {
            const std::string v = argv[++i];
            if      (v == "none")      co.earlyOut = carve::EarlyOut::None;
            else if (v == "saturated") co.earlyOut = carve::EarlyOut::Saturated;
            else if (v == "any")       co.earlyOut = carve::EarlyOut::AnyEvidence;
            else { std::printf("--early-out must be none, saturated or any\n"); return 2; }
            continue;
        }
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

    if (cmd == "probe") {
        if (paths.size() < 4) {
            std::printf("usage: e57cov probe <x> <y> <z> <file.e57> [more.e57 ...]\n");
            return 2;
        }
        const double world[3] = {std::strtod(paths[0].c_str(), nullptr),
                                 std::strtod(paths[1].c_str(), nullptr),
                                 std::strtod(paths[2].c_str(), nullptr)};
        const std::vector<std::string> files(paths.begin() + 3, paths.end());
        return probePoint(files, co, world);
    }
    if (cmd == "carve") return carveCorpus(paths, co);

    int failures = 0;
    for (const auto& p : paths) failures += info(p, crc, co.maxRange);
    if (failures)
        std::printf("%d file(s) reported problems.\n", failures);
    return failures == 0 ? 0 : 1;
}
