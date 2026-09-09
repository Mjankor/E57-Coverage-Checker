#include "report.h"

#include "e57.h"
#include "frame.h"
#include "version.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

namespace report {

namespace {

// printf into a growing string. The report is assembled rather than printed so
// the same code serves the terminal, a window, and a file on disk.
struct Out {
    std::string& s;
    void add(const char* fmt, ...) {
        va_list ap;
        va_start(ap, fmt);
        va_list copy;
        va_copy(copy, ap);
        const int n = std::vsnprintf(nullptr, 0, fmt, copy);
        va_end(copy);
        if (n > 0) {
            const size_t at = s.size();
            s.resize(at + size_t(n) + 1);
            std::vsnprintf(&s[at], size_t(n) + 1, fmt, ap);
            s.resize(at + size_t(n));
        }
        va_end(ap);
    }
};

// The rows at each end of a scan's declared grid that hold no returns at all.
//
// Decoded from rowIndex alone — a packed field of a dozen bits — so this costs a
// fraction of building the raster, and it is the whole of the evidence the
// corpus-wide blind cone decision uses. Cheap enough to run over every scan of
// every file before the report proper starts, which is what lets the report
// describe the same images a carve would use rather than a differently-decided set.
bool gridRowBands(e57::Reader& r, size_t scanIndex, uint32_t& lead, uint32_t& trail) {
    lead = trail = 0;
    if (scanIndex >= r.scanCount()) return false;
    const e57::Scan& s = r.scan(scanIndex);
    if (!s.hasIndexBounds || !s.field("rowIndex") || s.rowMax <= s.rowMin) return false;
    const int64_t span = s.rowMax - s.rowMin + 1;
    if (span <= 0 || span > (1 << 20)) return false;

    std::vector<std::string> want{"rowIndex"};
    size_t invIdx = SIZE_MAX;
    if      (s.field("cartesianInvalidState")) { invIdx = 1; want.push_back("cartesianInvalidState"); }
    else if (s.field("sphericalInvalidState")) { invIdx = 1; want.push_back("sphericalInvalidState"); }

    std::vector<uint8_t> has(size_t(span), 0);
    std::string err;
    const bool ok = r.readPoints(scanIndex, want, [&](const e57::PointBlock& b) {
        for (size_t k = 0; k < b.count; ++k) {
            if (invIdx != SIZE_MAX && b.columns[invIdx][k] != 0.0) continue;
            const int64_t rr = int64_t(b.columns[0][k]) - s.rowMin;
            if (rr >= 0 && rr < span) has[size_t(rr)] = 1;
        }
        return true;
    }, err);
    if (!ok) return false;

    uint32_t lo = 0;
    while (lo < uint32_t(span) && !has[lo]) ++lo;
    if (lo == uint32_t(span)) return false;         // no returns at all
    uint32_t hi = 0;
    while (hi + lo < uint32_t(span) && !has[size_t(span) - 1 - hi]) ++hi;
    lead = lo; trail = hi;
    return true;
}

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
    uint64_t invalid   = 0;
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

} // namespace

int scanReport(const std::string& path, const Options& opt, std::string& out) {
    Out o{out};
    e57::Reader r;
    std::string err;
    if (!r.open(path, err)) {
        o.add("%s: ERROR %s\n", path.c_str(), err.c_str());
        return 1;
    }

    o.add("%s\n", path.c_str());
    if (opt.verifyCrc) {
        o.add("  CRC: %s\n", r.verifyCrc(err) ? "all pages OK" : err.c_str());
    }
    o.add("  scans: %zu\n", r.scanCount());

    int failures = 0;
    for (size_t i = 0; i < r.scanCount(); ++i) {
        const e57::Scan& s = r.scan(i);
        o.add("\n  [%zu] %s\n", i, s.name.empty() ? "(unnamed)" : s.name.c_str());
        o.add("      records   : %llu\n", (unsigned long long)s.recordCount);
        if (s.hasPose)
            o.add("      pose      : t=(%.4f, %.4f, %.4f)  q=(%.6f, %.6f, %.6f, %.6f)\n",
                        s.pose.t[0], s.pose.t[1], s.pose.t[2],
                        s.pose.q[0], s.pose.q[1], s.pose.q[2], s.pose.q[3]);
        else
            o.add("      pose      : absent (identity)\n");

        if (s.hasIndexBounds)
            o.add("      structure : rows %lld..%lld, cols %lld..%lld\n",
                        (long long)s.rowMin, (long long)s.rowMax,
                        (long long)s.colMin, (long long)s.colMax);
        else
            o.add("      structure : no indexBounds — grid must be recovered from data\n");

        o.add("      prototype :");
        for (const auto& f : s.proto) {
            o.add(" %s:%s", f.name.c_str(), typeName(f.type));
            if (f.isPacked()) o.add("(%db)", f.bits);
        }
        o.add("\n");

        // Pick the position fields this file actually carries.
        std::vector<std::string> want;
        bool cartesian = s.field("cartesianX") && s.field("cartesianY") && s.field("cartesianZ");
        bool spherical = s.field("sphericalRange");
        if (cartesian)      want = {"cartesianX", "cartesianY", "cartesianZ"};
        else if (spherical) want = {"sphericalRange"};
        else {
            o.add("      DECODE    : skipped — no cartesian or spherical position fields\n");
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
            o.add("      DECODE    : FAILED — %s\n", err.c_str());
            ++failures;
            continue;
        }

        // Check 1: a drifting bit cursor almost always truncates or overruns.
        const bool countOk = (st.decoded == s.recordCount);
        o.add("      decoded   : %llu / %llu  %s\n",
                    (unsigned long long)st.decoded, (unsigned long long)s.recordCount,
                    countOk ? "OK" : "*** MISMATCH ***");
        if (!countOk) ++failures;

        // Question 1: how are no-returns represented? A declared sampling grid
        // answers it outright, and supersedes anything the invalid-state field
        // does or does not say.
        const bool hasGrid = s.hasIndexBounds && s.field("rowIndex") && s.field("columnIndex");
        if (invName) {
            const double pct = s.recordCount ? 100.0 * double(st.invalid) / double(s.recordCount) : 0.0;
            o.add("      no-return : %s present — %llu of %llu records (%.2f%%)\n",
                        invName, (unsigned long long)st.invalid,
                        (unsigned long long)s.recordCount, pct);
            if (st.invalid == 0)
                o.add("                  field present but never set: no misses are stored as records\n");
        } else {
            o.add("      no-return : no invalid-state field — no misses are stored as records\n");
        }
        if (hasGrid) {
            o.add("                  the declared grid identifies them exactly: every cell\n"
                        "                  with no record is a ray that came back empty\n");
        } else {
            o.add("                  and without a declared grid they cannot be identified\n"
                        "                  exactly — see DESIGN.md §4\n");
        }

        if (cartesian && st.any) {
            o.add("      bounds    : x[%.3f, %.3f] y[%.3f, %.3f] z[%.3f, %.3f]\n",
                        st.minX, st.maxX, st.minY, st.maxY, st.minZ, st.maxZ);
            // Check 2: an independent statement of the same extent.
            if (s.hasCartesianBounds) {
                const double tol = 1e-3;
                const bool within = st.minX >= s.xMin - tol && st.maxX <= s.xMax + tol &&
                                    st.minY >= s.yMin - tol && st.maxY <= s.yMax + tol &&
                                    st.minZ >= s.zMin - tol && st.maxZ <= s.zMax + tol;
                o.add("      declared  : x[%.3f, %.3f] y[%.3f, %.3f] z[%.3f, %.3f]  %s\n",
                            s.xMin, s.xMax, s.yMin, s.yMax, s.zMin, s.zMax,
                            within ? "OK (decoded within declared)" : "*** OUTSIDE DECLARED BOUNDS ***");
                if (!within) ++failures;
            } else {
                o.add("      declared  : no cartesianBounds — cross-check unavailable\n");
            }
            // The range image is what the visibility pass consumes, so report
            // what this scan would actually yield: how many rays came back
            // empty, and whether the raster is regular enough to look up.
            if (hasGrid) {
                rimg::RangeImage img;
                rimg::Options ro;
                ro.maxRange         = opt.maxRange;
                ro.blindCone        = opt.blindCone;
                ro.noReturnRadius   = opt.noReturnRadius;
                ro.noReturnFraction = opt.noReturnFraction;
                std::string rerr;
                if (rimg::build(r, i, ro, img, rerr)) {
                    o.add("      grid      : %u x %u = %.2f M cells, %.1f%% filled\n",
                                img.rows, img.cols,
                                double(img.cellCount()) / 1e6, 100.0 * img.diag.fillFraction);
                    o.add("      rays      : %llu returns, %llu no-returns "
                                "(these are what clear space)\n",
                                (unsigned long long)img.diag.hits,
                                (unsigned long long)img.diag.noReturns);
                    if (img.diag.isolatedNoReturns) {
                        const double pct = 100.0 * double(img.diag.isolatedNoReturns) /
                                           double(img.diag.isolatedNoReturns + img.diag.noReturns);
                        o.add("      drops     : %llu empty cells (%.1f%% of them) had too "
                                    "few empty\n                  neighbours to be sky and clear "
                                    "nothing. Believed, each\n                  would have cleared "
                                    "a line to %.0f m through solid geometry.\n",
                                    (unsigned long long)img.diag.isolatedNoReturns, pct,
                                    ro.maxRange);
                    }
                    o.add("      range     : returns from %.2f m to %.2f m; "
                                "no-returns clear to %.0f m\n",
                                img.diag.nearestReturn, img.diag.furthestReturn, ro.maxRange);
                    if (img.diag.furthestReturn > ro.maxRange * 1.05) {
                        o.add("                  note: returns reach past --max-range, so some "
                                    "measured\n                  surfaces sit beyond where "
                                    "no-return rays stop clearing\n");
                    }
                    // The measured mapping: what the tables actually span, and how
                    // far from a straight line they run. The residual is in cells
                    // rather than radians because that is the unit a lookup cares
                    // about — a line is wrong by so many rows and columns, and one
                    // cell is all the slack a lookup has.
                    const double kDeg = 57.29577951308232;
                    const double rowErr = img.map.elResidualRad /
                                          std::max(1e-12, std::fabs(img.map.dElPerRow));
                    const double colErr = img.map.azResidualRad /
                                          std::max(1e-12, std::fabs(img.map.dAzPerCol));
                    o.add("      raster    : %s\n"
                                "                  sweep %.2f deg over %u rows, %.2f deg over "
                                "%u cols\n",
                                img.map.valid
                                    ? (rowErr > 1.0 || colErr > 1.0 ? "measured, not uniform"
                                                                    : "measured, uniform")
                                    : "*** REFUSED — lookups would reach the wrong "
                                      "direction ***",
                                img.map.elSpanRad * kDeg, img.rows,
                                img.map.azSpanRad * kDeg, img.cols);
                    // A sweep past 360 degrees is the fault that broke every
                    // lookup past the seam, so it is stated rather than left to be
                    // noticed in the figure above.
                    if (std::fabs(img.map.azSpanRad) > rimg::kTwoPi) {
                        const double extra = (std::fabs(img.map.azSpanRad) - rimg::kTwoPi) /
                                             std::max(1e-12, std::fabs(img.map.dAzPerCol));
                        o.add("                  the sweep runs %.0f columns past a full turn, "
                                    "so those bearings\n                  were looked at twice; "
                                    "one of each pair answers a lookup\n", extra);
                    }
                    // How far a straight line would have been from the measured
                    // tables. Reported, and no longer used: the tables are.
                    o.add("      mapping   : measured tables, used directly\n"
                                "                  a straight line through them would sit "
                                "%.1f rows / %.1f cols out\n"
                                "                  (el = %+.6f %+.8f * row, "
                                "az = %+.6f %+.8f * col)\n",
                                rowErr, colErr,
                                img.map.el0, img.map.dElPerRow,
                                img.map.az0, img.map.dAzPerCol);
                    if (!img.map.monotonicEl || !img.map.monotonicAz)
                        o.add("                  *** %s turns back on itself — a mirror past "
                                    "the pole ***\n",
                                    !img.map.monotonicEl ? "elevation" : "azimuth");
                    // A sample of the tables themselves, so the raster's actual
                    // shape is visible rather than only its summary. Five points
                    // across each axis, with the step between them: a uniform
                    // raster shows the same step five times, and anything else
                    // shows where it varies.
                    if (img.map.elByRow.size() == img.rows && img.rows >= 5 &&
                        img.map.azByCol.size() == img.cols && img.cols >= 5) {
                        o.add("      el by row :");
                        for (int k = 0; k < 5; ++k) {
                            const uint32_t r = uint32_t(uint64_t(k) * (img.rows - 1) / 4);
                            o.add("  %u:%+.2f", r, img.map.elByRow[r] * kDeg);
                        }
                        o.add("  deg\n      az by col :");
                        for (int k = 0; k < 5; ++k) {
                            const uint32_t c = uint32_t(uint64_t(k) * (img.cols - 1) / 4);
                            o.add("  %u:%+.1f", c, img.map.azByCol[c] * kDeg);
                        }
                        o.add("  deg\n");
                    }
                    // The check that matters: the scan's own points put back
                    // through the mapping. A low figure here with low residuals
                    // means lookups land in the wrong place while the fit looks
                    // healthy, which is how sky comes back unobserved and
                    // building interiors come back clear.
                    if (img.map.roundTripFraction >= 0.0) {
                        const double rt = 100.0 * img.map.roundTripFraction;
                        o.add("      round trip: %.2f%% of this scan's own points land back "
                                    "on their own cell  %s\n", rt,
                                    img.map.roundTripFraction >= 0.90 ? "OK"
                                                                      : "*** BROKEN ***");
                    } else {
                        o.add("      round trip: not measured\n");
                    }
                    if (img.diag.blindConeRows) {
                        const double here = img.diag.blindConeAtFirstRow
                                          ? img.diag.borderRangeFirst : img.diag.borderRangeLast;
                        const double there = img.diag.blindConeAtFirstRow
                                          ? img.diag.borderRangeLast : img.diag.borderRangeFirst;
                        o.add("      blind cone: %u unsampled rows at the %s of the raster "
                                    "(%llu cells)\n"
                                    "                  bordering returns %.2f m here; ",
                                    img.diag.blindConeRows,
                                    img.diag.blindConeAtFirstRow ? "start" : "end",
                                    (unsigned long long)img.diag.blindConeCells, here);
                        if (there >= 0) o.add("%.2f m at the other end\n", there);
                        else            o.add("no unsampled band at the other end\n");
                        if (img.diag.hasConeAxis) {
                            o.add("      mounting  : cone axis (%.3f, %.3f, %.3f) — %s\n",
                                        img.diag.coneAxisWorld[0], img.diag.coneAxisWorld[1],
                                        img.diag.coneAxisWorld[2],
                                        img.diag.coneAxisWorld[2] > 0.5 ? "*** INVERTED ***"
                                      : img.diag.coneAxisWorld[2] < -0.5 ? "upright"
                                                                         : "on its side");
                        }
                    } else {
                        o.add("      blind cone: none identified — every empty cell is "
                                    "treated as a no-return\n");
                    }
                    if (img.diag.outsideGrid)
                        o.add("      off-grid  : %llu points fell outside the declared "
                                    "indexBounds\n",
                                    (unsigned long long)img.diag.outsideGrid);
                    if (!img.diag.originInsideReturns)
                        o.add("      *** the scanner sits outside the box of its own "
                                    "returns — check the frame\n");
                    if (!img.diag.note.empty())
                        o.add("      note      : %s\n", img.diag.note.c_str());
                    if (!img.map.valid) ++failures;
                    if (img.diag.emptyLeadingRows || img.diag.emptyTrailingRows) {
                        // The bands at each end, and which of them this run is
                        // believing. One of them establishes nothing and the other
                        // clears space to the rated range; the corpus decides
                        // which, and this line is what shows the decision landing
                        // on this scan.
                        const uint32_t lead = img.diag.emptyLeadingRows;
                        const uint32_t trail = img.diag.emptyTrailingRows;
                        const char* believed =
                            img.diag.blindConeRows == 0 ? "both are believed as no-returns"
                          : img.diag.blindConeAtFirstRow
                                ? "the leading band is unsampled; the trailing band clears"
                                : "the trailing band is unsampled; the leading band clears";
                        o.add("      bands     : %u leading and %u trailing grid rows hold no "
                                    "returns\n                  %s\n", lead, trail, believed);
                    }
                } else {
                    o.add("      grid      : range image failed — %s\n", rerr.c_str());
                    ++failures;
                }
            } else {
                o.add("      grid      : no indexBounds + row/column index — no-return rays\n"
                            "                  cannot be identified exactly (DESIGN.md §4)\n");
            }

            // Question 2: which coordinate frame, and is the pose applied?
            const viewer::FrameDecision fd = viewer::decideFrame(r, i);
            o.add("      frame     : %s\n", viewer::conventionName(fd.convention));
            o.add("                  %s\n", fd.reason.c_str());
            if (fd.nonConformant()) {
                o.add("      *** this file stores pre-transformed points WITH a non-identity\n"
                            "          pose, which contradicts ASTM E2807. The pose is being\n"
                            "          ignored for this scan; verify against a known-good viewer.\n");
                ++failures;
            }
        }
    }

    o.add("\n");
    return failures == 0 ? 0 : 1;
}

int scanReport(const std::vector<std::string>& paths, const Options& opt, std::string& out) {
    Out o{out};
    o.add("e57cov scan report — build %s\n", ver::describe());
    o.add("%zu file(s), max range %.1f m\n", paths.size(), opt.maxRange);

    // Which end of the raster holds the instrument's blind cone, decided once over
    // every scan of every file. It has to come first: a scan's empty cells either
    // clear space to the rated range or establish nothing at all, and that is the
    // difference between a coverage report and a picture of a cone carved through
    // the ground under each setup. No single scan can tell which — see
    // rimg::decideBlindConeEnd — so the report would otherwise be describing a
    // decision the carve does not make.
    std::vector<std::pair<uint32_t, uint32_t>> bands;
    for (const std::string& p : paths) {
        e57::Reader r;
        std::string e;
        if (!r.open(p, e)) continue;
        for (size_t i = 0; i < r.scanCount(); ++i) {
            uint32_t lead = 0, trail = 0;
            if (gridRowBands(r, i, lead, trail)) bands.push_back({lead, trail});
        }
    }
    rimg::Options ro;
    ro.blindCone        = opt.blindCone;
    ro.coneCorpusSpread = rimg::Options{}.coneCorpusSpread;
    const rimg::ConeVerdict cone = rimg::decideBlindConeEnd(bands, ro);
    o.add("blind cone: %s\n            %s\n",
          cone.decided ? (cone.atFirstRow ? "the START of each raster"
                                          : "the END of each raster")
                       : "*** not identified — see below ***",
          cone.why.c_str());
    o.add("\n");

    // Every file then reports against that decision rather than its own.
    Options scoped = opt;
    if (cone.decided)
        scoped.blindCone = cone.atFirstRow ? rimg::BlindCone::FirstRows
                                           : rimg::BlindCone::LastRows;

    int failures = 0;
    for (const std::string& p : paths) {
        failures += scanReport(p, scoped, out);
        Out{out}.add("\n");
    }
    if (failures) Out{out}.add("%d file(s) reported problems.\n", failures);
    else          Out{out}.add("No problems reported.\n");
    return failures;
}

} // namespace report
