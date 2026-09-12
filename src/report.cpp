#include "report.h"

#include "carve.h"
#include "e57.h"
#include "frame.h"
#include "version.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
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

        // How many returns one ray produced, and which field the producer says
        // indexes a scan line. Both are parts of the standard that were read past
        // until now; neither changes what the carve does, and both change what can
        // be said about whether it is doing the right thing.
        if (s.hasReturnIndexBounds) {
            if (s.multiReturn())
                o.add("      returns   : returnIndex %lld..%lld — MULTI-RETURN: one ray can "
                      "report several\n                  surfaces, and a cell keeps the "
                      "nearest, since sight stops there\n",
                      (long long)s.returnIndexMin, (long long)s.returnIndexMax);
            else
                o.add("      returns   : returnIndex %lld..%lld — one return per ray\n",
                      (long long)s.returnIndexMin, (long long)s.returnIndexMax);
        }
        if (!s.groupingIdElement.empty()) {
            o.add("      lines     : the file groups points by %s", s.groupingIdElement.c_str());
            if (s.groupCount) o.add(", %llu lines", (unsigned long long)s.groupCount);
            // Which axis carries elevation is measured from the points, never
            // taken from here, so this is a cross-check. Only a declaration naming
            // neither index is worth remarking on.
            if (s.groupingIdElement != "rowIndex" && s.groupingIdElement != "columnIndex")
                o.add("  (neither rowIndex nor columnIndex — unexpected)");
            o.add("\n");
        } else if (s.hasPointGrouping) {
            o.add("      lines     : pointGroupingSchemes present but no "
                  "groupingByLine/idElementName\n");
        }

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
                    // Coarsening is not a loss of sharpness, it is a change of
                    // answer: a coarse cell keeps the nearest return landing in
                    // it, so it clears less space than the cells it replaced.
                    if (img.diag.binStep > 1)
                        o.add("                  *** COARSENED %ux: %u x %u declared cells "
                              "binned into one, to fit the\n                  cell budget. "
                              "Coarse cells clear less space, so unobserved volume\n"
                              "                  comes out overstated ***\n",
                              img.diag.binStep, img.diag.binStep, img.diag.binStep);
                    o.add("      rays      : %llu returns, %llu no-returns "
                                "(these are what clear space)\n",
                                (unsigned long long)img.diag.hits,
                                (unsigned long long)img.diag.noReturns);
                    // Cells that took more than one return, measured rather than
                    // inferred from the declaration above. The nearest is kept;
                    // how many of the rest sat behind it says which cause it is.
                    if (img.diag.cellsWithSeveralReturns) {
                        const uint64_t n = img.diag.cellsWithSeveralReturns;
                        const uint64_t behind = img.diag.returnsKeptBehindANearerOne;
                        o.add("      several   : %llu cells took more than one return, %llu of "
                              "those behind one already\n                  held — the nearest "
                              "is kept, since line of sight stops there%s\n",
                              (unsigned long long)n, (unsigned long long)behind,
                              s.multiReturn()
                                  ? " (multi-return)"
                                  : (s.hasIndexBounds &&
                                     uint64_t(s.rowMax - s.rowMin + 1) > img.rows
                                         ? " (the raster was binned down)" : ""));
                    }
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
                    // How the tripod was standing. Not a curiosity: until this is
                    // taken out, a row is not a line of constant elevation and no
                    // cell-to-direction mapping exists to be measured at all.
                    if (img.tiltDeg > 0.001) {
                        o.add("      levelling : the instrument leaned %.3f deg toward azimuth "
                              "%.1f deg\n"
                              "                  (%.1f%% of how much elevation varied inside a "
                              "row; taken out before\n                  the raster was built, "
                              "since a row is only a direction about the\n"
                              "                  instrument's own axis)\n",
                              img.tiltDeg, img.tiltTowardDeg, 100.0 * img.tiltExplained);
                    } else {
                        o.add("      levelling : the instrument was level; no correction "
                              "needed\n");
                    }
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
                    // Where the returns say the instrument stood, against where the
                    // pose says. The only check on the setup position that does not
                    // come from the pose — see originAudit, and the red markers in
                    // the viewer, which are drawn from the pose and nothing else.
                    {
                        const rimg::Diagnostics& d = img.diag;
                        if (!d.haveMeasuredOrigin) {
                            o.add("      setup pos : the returns hold no opposite pairs, so the "
                                  "frame cannot be checked against them\n");
                        } else {
                            const double off =
                                std::sqrt(d.measuredOrigin[0] * d.measuredOrigin[0] +
                                          d.measuredOrigin[1] * d.measuredOrigin[1] +
                                          d.measuredOrigin[2] * d.measuredOrigin[2]);
                            const double bar = std::max(0.25, 3.0 * d.originRms);
                            if (off <= bar) {
                                o.add("      setup pos : the returns put the instrument %.3f m "
                                      "from the origin of the frame\n"
                                      "                  they are read in, which is where it "
                                      "belongs (%u ray pairs, rms %.3f m)\n",
                                      off, d.originPairs, d.originRms);
                            } else {
                                o.add("      setup pos : *** the returns put the instrument "
                                      "%.2f m from the origin of the\n"
                                      "                  frame they are read in, at (%.3f, %.3f, "
                                      "%.3f) ***\n"
                                      "                  %u ray pairs meeting to %.3f m, so this "
                                      "is not scatter. These points\n"
                                      "                  are not in the frame they are being "
                                      "read as, which displaces this\n"
                                      "                  setup — and its red marker — by that "
                                      "much\n",
                                      off, d.measuredOrigin[0], d.measuredOrigin[1],
                                      d.measuredOrigin[2], d.originPairs, d.originRms);
                                ++failures;
                            }
                        }
                    }
                    if (!img.diag.note.empty())
                        o.add("      note      : %s\n", img.diag.note.c_str());
                    if (!img.map.valid) ++failures;
                    if (img.diag.emptyLeadingRows || img.diag.emptyTrailingRows) {
                        // The bands at each end, and which of them this run is
                        // believing. A believed band clears space to the rated
                        // range and an unsampled one establishes nothing, so this
                        // line is what shows this scan's own decision landing.
                        const uint32_t lead = img.diag.emptyLeadingRows;
                        const uint32_t trail = img.diag.emptyTrailingRows;
                        const char* believed =
                            img.diag.blindConeRows == 0 ? "both are believed as no-returns"
                          : img.diag.blindConeRowsLast != 0
                                ? "NEITHER is believed: both bands are unsampled"
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

namespace {

// One scan's points, in the world frame, plus a coarse occupancy set.
struct Cloud {
    std::string name;
    double setup[3] = {0, 0, 0};          // where the file says the scanner was
    double lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};
    std::vector<float> xyz;               // under the chosen hypothesis
    std::vector<float> alt;               // under the other one
    std::unordered_set<int64_t> cells;    // of `xyz`
    bool   applied = false;               // the chosen hypothesis: pose applied?
    bool   uninformative = false;
    double rotationDeg = 0, translationM = 0;
    std::string reason;
};

// Coarse enough that registration error and scan-to-scan sampling do not matter,
// fine enough that a cloud turned by a few degrees stops matching.
constexpr double kAgreeCell = 0.5;

// How far in front of another setup's measured surface a point has to sit before
// it counts as contradicted. Well clear of the surface margin, so a point on a
// shared surface — which is agreement, not conflict — is never counted.
constexpr double kContradictSlack = 0.75;

int64_t cellKey(double x, double y, double z) {
    const int64_t i = int64_t(std::floor(x / kAgreeCell));
    const int64_t j = int64_t(std::floor(y / kAgreeCell));
    const int64_t k = int64_t(std::floor(z / kAgreeCell));
    return ((i & 0x1FFFFF) << 42) | ((j & 0x1FFFFF) << 21) | (k & 0x1FFFFF);
}

// The share of `pts` that another setup saw straight through.
//
// This is the measurement that settles it, and simple overlap is not: a ground
// plane is unchanged by turning it about a vertical axis, so most of a mis-yawed
// cloud still lands on ground and "agreement" stays high whichever way round it
// is. Contradiction cannot be faked that way. A point is a surface something
// measured, and no setup can have had a clear line of sight through a surface —
// so a point sitting well in front of another setup's own measured surface, in
// space that setup cleared, is a physical impossibility. Under the right
// transform these are rare; under a wrong one the building lands in the middle of
// space everyone else saw through, and the rate goes through the roof.
struct Conflict {
    double rate = -1.0;      // of the points anyone could speak to
    double judged = 0.0;     // what share of the scan that was
};

Conflict contradicted(const std::vector<float>& pts,
                      const std::vector<carve::SetupView>& views, size_t skip,
                      const carve::Params& p) {
    Conflict c;
    if (pts.empty()) return c;
    carve::Params loose = p;
    loose.surfaceMargin = kContradictSlack;
    uint64_t total = 0, n = 0, bad = 0;
    for (size_t i = 0; i + 2 < pts.size(); i += 3) {
        ++total;
        bool judged = false, through = false;
        for (size_t v = 0; v < views.size(); ++v) {
            if (v == skip) continue;
            const uint8_t e = carve::evidenceAt(views[v], loose, pts[i], pts[i + 1], pts[i + 2]);
            if (e & (carve::kVisible | carve::kOccupied)) judged = true;
            if (e & carve::kVisible) through = true;
        }
        if (!judged) continue;            // nobody could speak to this point
        ++n;
        if (through) ++bad;
    }
    // Coverage matters as much as the rate. A hypothesis that flings a cloud
    // somewhere nobody is looking scores a beautiful conflict rate over a handful
    // of points, so a rate is only worth comparing against one measured over a
    // comparable share of the scan.
    c.judged = total ? double(n) / double(total) : 0.0;
    c.rate   = n ? double(bad) / double(n) : -1.0;
    return c;
}

// The share of `pts` that land on or beside an occupied cell of `set`.
double agreement(const std::vector<float>& pts, const std::unordered_set<int64_t>& set) {
    if (pts.empty() || set.empty()) return -1.0;
    uint64_t hit = 0, n = 0;
    for (size_t i = 0; i + 2 < pts.size(); i += 3) {
        ++n;
        bool found = false;
        for (int dx = -1; dx <= 1 && !found; ++dx)
            for (int dy = -1; dy <= 1 && !found; ++dy)
                for (int dz = -1; dz <= 1 && !found; ++dz)
                    if (set.count(cellKey(pts[i] + dx * kAgreeCell,
                                          pts[i + 1] + dy * kAgreeCell,
                                          pts[i + 2] + dz * kAgreeCell))) found = true;
        if (found) ++hit;
    }
    return n ? double(hit) / double(n) : -1.0;
}

constexpr uint64_t kCloudSamples = 120000;

} // namespace

// Where does a point's own direction resolve to, against the cell it was stored in?
//
// This is the round-trip check with its workings shown. The gate reports one
// percentage, which says a mapping is wrong without saying how, and three quite
// different faults produce the same percentage:
//
//   an error that shrinks with range      the ray does not start where the points
//                                         are measured from
//   an error flat in range                the mapping from cell to direction is
//                                         the wrong shape
//   an error only in one band of the      the raster is not one raster — a sweep
//   raster, or only past a bearing        that overlaps itself, or turns over
//
// So the error is broken down by range and by where in the raster it happened, in
// cells, which tells those three apart in one run. Read straight from the file
// rather than from anything build() kept, so a fault inside build() cannot hide.
// Is the instrument where the frame being used puts it?
//
// rimg::measureOrigin finds the instrument from the returns alone: opposite rays
// are collinear with it, so it is where those lines cross. The answer is in the
// frame the cells were built about, where it should be the origin.
//
// What that can and cannot settle is worth being exact about, because the
// obvious reading of it is wrong. For a conformant scan the points are stored
// about the instrument, so this measures zero and says nothing whatever about
// where the setup stood in the file's coordinates — that is carried entirely by
// `pose`, and one scan's own points cannot check it. Only the registration
// between scans can.
//
// What it does catch is the frame being wrong: a pre-transformed file whose pose
// was applied anyway, a pose subtracted that should not have been, points not
// stored about the instrument at all. Those displace the setup by the whole
// magnitude of the site coordinates — tens of metres, sometimes straight down —
// and they are the reason a setup marker lands nowhere near its scan. That is
// the class of fault this puts a number on. diag.originInsideReturns already
// asked the same question as a yes or no; this answers it in metres, which is
// the difference between knowing something is wrong and knowing what.
//
// Reported, never used to overrule the file. Returns true when the frame puts
// the instrument further from the returns' own answer than the fit's scatter
// explains.
static bool originAudit(const rimg::RangeImage& im, const viewer::Rigid& fwd, Out& o) {
    const rimg::Diagnostics& d = im.diag;
    (void)fwd;
    if (!d.haveMeasuredOrigin) {
        o.add("      setup position: the returns hold no opposite pairs, so the frame "
              "cannot be checked against them\n");
        return false;
    }
    const double off = std::sqrt(d.measuredOrigin[0] * d.measuredOrigin[0] +
                                 d.measuredOrigin[1] * d.measuredOrigin[1] +
                                 d.measuredOrigin[2] * d.measuredOrigin[2]);
    // The bar is the scatter of the fit itself, floored so a very clean scan does
    // not start reporting centimetres as a disagreement.
    const double bar = std::max(0.25, 3.0 * d.originRms);
    if (off <= bar) {
        o.add("      setup position: the returns put the instrument %.3f m from the origin "
              "of the frame they\n        are read in, which is where it belongs "
              "(%u ray pairs, meeting to %.3f m)\n",
              off, d.originPairs, d.originRms);
        return false;
    }
    o.add("      setup position: *** the returns put the instrument at (%.3f, %.3f, %.3f) "
          "in the frame\n        they are read in, %.2f m from the origin where it belongs — "
          "%u ray pairs\n        meeting to %.3f m, so this is not scatter. The points are not "
          "in the frame\n        they are being read as, which displaces this setup by that "
          "much ***\n",
          d.measuredOrigin[0], d.measuredOrigin[1], d.measuredOrigin[2],
          off, d.originPairs, d.originRms);
    return true;
}

static void rasterAudit(e57::Reader& r, size_t scanIndex, const rimg::RangeImage& im,
                        Out& o) {
    const e57::Scan& s = r.scan(scanIndex);
    if (!s.hasIndexBounds || !s.field("rowIndex") || !s.field("columnIndex")) return;
    if (im.rows == 0 || im.cols == 0 || im.map.rowOfEl.empty()) return;

    std::vector<std::string> want{"cartesianX", "cartesianY", "cartesianZ",
                                  "rowIndex", "columnIndex"};
    size_t invIdx = SIZE_MAX;
    if (s.field("cartesianInvalidState")) { invIdx = want.size(); want.push_back("cartesianInvalidState"); }

    // Exactly what build() does about the frame, so this describes the image that
    // was built and not a differently-decided one.
    const viewer::FrameDecision fd = viewer::decideFrame(r, scanIndex);
    const bool subtractPose = !fd.applyPose() && s.hasPose;
    const viewer::Rigid poseRot = viewer::rigidFromPose(s.pose);

    const int64_t spanR = s.rowMax - s.rowMin + 1, spanC = s.colMax - s.colMin + 1;
    const uint32_t stepR = spanR > 0 ? uint32_t((spanR + im.rows - 1) / im.rows) : 1;
    const uint32_t stepC = spanC > 0 ? uint32_t((spanC + im.cols - 1) / im.cols) : 1;

    // Range bands, and one bucket of row/column errors in each.
    const double edges[7] = {0.0, 2.0, 5.0, 10.0, 20.0, 45.0, 1e9};
    std::vector<double> dRow[6], dCol[6];
    uint64_t seen = 0, used = 0, unresolved = 0;
    const double azStep = (im.cols > 1) ? std::fabs(im.map.azSpanRad) / double(im.cols - 1) : 0.0;
    const double perTurn = (azStep > 1e-12) ? 6.28318530717958648 / azStep : double(im.cols);
    const uint64_t stride = std::max<uint64_t>(1, s.recordCount / 60000);

    std::string err;
    r.readPoints(scanIndex, want, [&](const e57::PointBlock& b) {
        for (size_t k = 0; k < b.count; ++k, ++seen) {
            if (seen % stride) continue;
            if (invIdx != SIZE_MAX && b.columns[invIdx][k] != 0.0) continue;
            double x = b.columns[0][k], y = b.columns[1][k], z = b.columns[2][k];
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;
            if (subtractPose) {
                const double a = x - s.pose.t[0], bb = y - s.pose.t[1], c = z - s.pose.t[2];
                x = poseRot.R[0] * a + poseRot.R[3] * bb + poseRot.R[6] * c;
                y = poseRot.R[1] * a + poseRot.R[4] * bb + poseRot.R[7] * c;
                z = poseRot.R[2] * a + poseRot.R[5] * bb + poseRot.R[8] * c;
            }
            // Into the instrument's frame, because that is the frame the mapping
            // was built in — a row is only a line of constant elevation about the
            // axis the instrument actually had.
            im.toInstrument(x, y, z);
            double az, el, range;
            rimg::toSpherical(x, y, z, az, el, range);
            if (range <= 1e-6) continue;

            const int64_t rr = int64_t(b.columns[3][k]) - s.rowMin;
            const int64_t cc = int64_t(b.columns[4][k]) - s.colMin;
            if (rr < 0 || cc < 0 || rr >= spanR || cc >= spanC) continue;
            const int32_t row = int32_t(uint32_t(rr) / stepR);
            const int32_t col = int32_t(uint32_t(cc) / stepC);

            const int32_t gr = im.map.rowFor(el);
            const int32_t gc = im.map.colFor(az);
            ++used;
            if (gr < 0 || gc < 0) { ++unresolved; continue; }

            int band = 0;
            while (band < 5 && range >= edges[band + 1]) ++band;
            double dc = double(gc) - double(col);
            dc -= perTurn * std::round(dc / perTurn);
            dRow[band].push_back(std::fabs(double(gr) - double(row)));
            dCol[band].push_back(std::fabs(dc));
        }
        return true;
    }, err);

    if (!used) return;
    auto med = [](std::vector<double>& v) {
        if (v.empty()) return -1.0;
        std::nth_element(v.begin(), v.begin() + ptrdiff_t(v.size() / 2), v.end());
        return v[v.size() / 2];
    };
    auto within1 = [](const std::vector<double>& v) {
        if (v.empty()) return -1.0;
        size_t n = 0;
        for (double d : v) if (d <= 1.0) ++n;
        return 100.0 * double(n) / double(v.size());
    };

    // Returns only — every record in the file is one. A return came from a
    // direction the scanner looked in, so this figure is near zero by
    // construction and says nothing whatever about the empty cells that are the
    // sky. Reported only so that a non-zero value, which would mean the index does
    // not even cover the directions the scan measured in, is visible.
    o.add("      raster audit (returns only; empty cells are not sampled here)\n"
          "        %llu points, %.1f%% resolved to no cell\n",
          (unsigned long long)used, 100.0 * double(unresolved) / double(used));
    o.add("        %-14s %8s %10s %10s %10s %10s\n", "range", "points",
          "row err", "row ok", "col err", "col ok");
    const char* names[6] = {"under 2 m", "2 - 5 m", "5 - 10 m",
                            "10 - 20 m", "20 - 45 m", "over 45 m"};
    for (int i = 0; i < 6; ++i) {
        if (dRow[i].empty()) continue;
        const double n = double(dRow[i].size());
        const double rw = within1(dRow[i]), cw = within1(dCol[i]);
        o.add("        %-14s %8.0f %10.1f %9.1f%% %10.1f %9.1f%%\n",
              names[i], n, med(dRow[i]), rw, med(dCol[i]), cw);
    }
}

// Does a row actually have one elevation, and a column one azimuth?
//
// Everything this module does rests on that and nothing checks it. A raster cell
// is only a direction if every return in a row left at the same elevation and
// every return in a column at the same azimuth. If that is not true of the data,
// no mapping from (row, column) to a direction can be built, however it is fitted
// or measured — and the round trip will fail without saying why.
//
// So this measures the spread itself, with no model on top and nothing taken from
// what build() constructed. Each row's elevation is taken from its FAR returns
// only, which are the ones any ray-origin offset barely moves; then every return
// is compared against its own row's figure, and the deviation reported in cells,
// bucketed by range.
//
// The shape of the answer names the cause:
//
//   deviation grows as range falls   the rays do not start where the points are
//                                    measured from, by about (deviation x range)
//   deviation flat in range          rows and columns are not lines of constant
//                                    angle at all, and the raster is not one
//   deviation near zero everywhere   the raster is fine and the fault is in the
//                                    index built from it
static void spreadAudit(e57::Reader& r, size_t scanIndex, const rimg::RangeImage& im,
                        Out& o) {
    const e57::Scan& s = r.scan(scanIndex);
    if (!s.hasIndexBounds || !s.field("rowIndex") || !s.field("columnIndex")) return;
    if (im.rows < 2 || im.cols < 2) return;

    std::vector<std::string> want{"cartesianX", "cartesianY", "cartesianZ",
                                  "rowIndex", "columnIndex"};
    size_t invIdx = SIZE_MAX;
    if (s.field("cartesianInvalidState")) { invIdx = want.size(); want.push_back("cartesianInvalidState"); }

    const viewer::FrameDecision fd = viewer::decideFrame(r, scanIndex);
    const bool subtractPose = !fd.applyPose() && s.hasPose;
    const viewer::Rigid poseRot = viewer::rigidFromPose(s.pose);

    const int64_t spanR = s.rowMax - s.rowMin + 1, spanC = s.colMax - s.colMin + 1;
    if (spanR < 2 || spanC < 2) return;

    struct Obs { float el, az, range; int32_t row, col; };
    std::vector<Obs> obs;
    const uint64_t stride = std::max<uint64_t>(1, s.recordCount / 400000);
    obs.reserve(s.recordCount / stride + 16);

    uint64_t seen = 0;
    std::string err;
    r.readPoints(scanIndex, want, [&](const e57::PointBlock& b) {
        for (size_t k = 0; k < b.count; ++k, ++seen) {
            if (seen % stride) continue;
            if (invIdx != SIZE_MAX && b.columns[invIdx][k] != 0.0) continue;
            double x = b.columns[0][k], y = b.columns[1][k], z = b.columns[2][k];
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;
            if (subtractPose) {
                const double a = x - s.pose.t[0], bb = y - s.pose.t[1], c = z - s.pose.t[2];
                x = poseRot.R[0] * a + poseRot.R[3] * bb + poseRot.R[6] * c;
                y = poseRot.R[1] * a + poseRot.R[4] * bb + poseRot.R[7] * c;
                z = poseRot.R[2] * a + poseRot.R[5] * bb + poseRot.R[8] * c;
            }
            // Same frame the raster was built in; without this the audit measures
            // the tripod's lean and calls it a fault in the mapping.
            im.toInstrument(x, y, z);
            double az, el, range;
            rimg::toSpherical(x, y, z, az, el, range);
            if (range <= 1e-6) continue;
            const int64_t rr = int64_t(b.columns[3][k]) - s.rowMin;
            const int64_t cc = int64_t(b.columns[4][k]) - s.colMin;
            if (rr < 0 || cc < 0 || rr >= spanR || cc >= spanC) continue;
            obs.push_back({float(el), float(az), float(range), int32_t(rr), int32_t(cc)});
        }
        return true;
    }, err);
    if (obs.size() < 1000) return;

    // Each row's and column's angle, from far returns only. Ten metres is far
    // enough that a few centimetres of anything is a fraction of a cell, and near
    // enough that most scans have plenty.
    constexpr float kFar = 10.0f;
    std::vector<double> rowEl(size_t(spanR), 0.0), colX(size_t(spanC), 0.0),
                        colY(size_t(spanC), 0.0);
    std::vector<uint32_t> rowN(size_t(spanR), 0), colN(size_t(spanC), 0);
    for (const Obs& p : obs) {
        if (p.range < kFar) continue;
        rowEl[size_t(p.row)] += p.el; ++rowN[size_t(p.row)];
        colX[size_t(p.col)] += std::cos(double(p.az));
        colY[size_t(p.col)] += std::sin(double(p.az));
        ++colN[size_t(p.col)];
    }
    uint32_t rowsWithFar = 0, colsWithFar = 0;
    for (uint32_t v : rowN) if (v >= 4) ++rowsWithFar;
    for (uint32_t v : colN) if (v >= 4) ++colsWithFar;

    // A cell, in radians, from the raster's own extent.
    double elLo = 1e9, elHi = -1e9;
    for (size_t i = 0; i < rowN.size(); ++i)
        if (rowN[i] >= 4) { const double e = rowEl[i] / rowN[i];
                            elLo = std::min(elLo, e); elHi = std::max(elHi, e); }
    const double elCell = (elHi > elLo) ? (elHi - elLo) / double(spanR - 1) : 0.0;
    const double azCell = 6.28318530717958648 / double(spanC);
    if (!(elCell > 0)) return;

    const double edges[7] = {0.0, 2.0, 5.0, 10.0, 20.0, 45.0, 1e9};
    std::vector<double> dEl[6], dAz[6];
    for (const Obs& p : obs) {
        if (rowN[size_t(p.row)] < 4 || colN[size_t(p.col)] < 4) continue;
        int band = 0;
        while (band < 5 && p.range >= edges[band + 1]) ++band;
        dEl[band].push_back(std::fabs(double(p.el) - rowEl[size_t(p.row)] / rowN[size_t(p.row)])
                            / elCell);
        const double ca = std::atan2(colY[size_t(p.col)], colX[size_t(p.col)]);
        double da = double(p.az) - ca;
        da -= 6.28318530717958648 * std::round(da / 6.28318530717958648);
        dAz[band].push_back(std::fabs(da) / azCell);
    }

    auto med = [](std::vector<double>& v) {
        if (v.empty()) return -1.0;
        std::nth_element(v.begin(), v.begin() + ptrdiff_t(v.size() / 2), v.end());
        return v[v.size() / 2];
    };
    auto p95 = [](std::vector<double>& v) {
        if (v.empty()) return -1.0;
        const size_t k = std::min(v.size() - 1, (v.size() * 95) / 100);
        std::nth_element(v.begin(), v.begin() + ptrdiff_t(k), v.end());
        return v[k];
    };

    o.add("      spread audit: is a row one elevation and a column one azimuth?\n"
          "        %llu returns sampled; %u of %lld rows and %u of %lld columns have "
          "returns past %.0f m\n",
          (unsigned long long)obs.size(), rowsWithFar, (long long)spanR,
          colsWithFar, (long long)spanC, double(kFar));
    o.add("        %-14s %8s %9s %9s %9s %9s\n", "range", "points",
          "el med", "el p95", "az med", "az p95");
    const char* names[6] = {"under 2 m", "2 - 5 m", "5 - 10 m",
                            "10 - 20 m", "20 - 45 m", "over 45 m"};
    for (int i = 0; i < 6; ++i) {
        if (dEl[i].empty()) continue;
        o.add("        %-14s %8zu %9.2f %9.2f %9.2f %9.2f\n", names[i], dEl[i].size(),
              med(dEl[i]), p95(dEl[i]), med(dAz[i]), p95(dAz[i]));
    }
    o.add("        (deviation from the row's or column's own far-return angle, in cells)\n");

    // Is the elevation deviation a single cycle around the azimuth?
    //
    // If it is, the instrument was not level. Every terrestrial scanner has a
    // compensator and exports its points already levelled, so the raster's rows are
    // lines of constant elevation ABOUT THE INSTRUMENT'S OWN AXIS while the stored
    // points are about the vertical. Tilt the one against the other by tau and a
    // row's elevation runs as tau*cos(azimuth - phi): one cycle per turn, amplitude
    // tau, and almost nothing in azimuth near the horizon.
    //
    // That is a rotation, not an offset, so it leaves no parallax on edges and is
    // invisible in the merged cloud — the manufacturer applied it correctly. It is
    // only visible if you go looking for the raster, which is what this module does
    // and nothing else in the pipeline does.
    //
    // Fitted globally by least squares, and reported with the share of the spread
    // it accounts for. A high share is the whole answer; a low one rules it out.
    {
        std::vector<double> rowMean(size_t(spanR), 0.0);
        std::vector<uint32_t> rowAll(size_t(spanR), 0);
        for (const Obs& p : obs) { rowMean[size_t(p.row)] += p.el; ++rowAll[size_t(p.row)]; }
        for (size_t i = 0; i < rowMean.size(); ++i)
            if (rowAll[i]) rowMean[i] /= double(rowAll[i]);

        double scc = 0, sss = 0, scs = 0, sdc = 0, sds = 0, sdd = 0;
        uint64_t n = 0;
        for (const Obs& p : obs) {
            if (rowAll[size_t(p.row)] < 8) continue;
            const double d = double(p.el) - rowMean[size_t(p.row)];
            const double c = std::cos(double(p.az)), si = std::sin(double(p.az));
            scc += c * c; sss += si * si; scs += c * si;
            sdc += d * c; sds += d * si; sdd += d * d;
            ++n;
        }
        if (n > 1000) {
            const double det = scc * sss - scs * scs;
            if (std::fabs(det) > 1e-12) {
                const double A = ( sss * sdc - scs * sds) / det;
                const double B = (-scs * sdc + scc * sds) / det;
                const double explained = A * sdc + B * sds;
                const double frac = (sdd > 0) ? explained / sdd : 0.0;
                const double amp = std::sqrt(A * A + B * B);
                const double resid = std::sqrt(std::max(0.0, (sdd - explained) / double(n)));
                o.add("        one cycle per turn: amplitude %.3f deg toward azimuth %.1f deg,\n"
                      "        accounting for %.1f%% of the elevation spread; %.2f cells left "
                      "after it\n",
                      amp * 57.29577951308232,
                      std::atan2(B, A) * 57.29577951308232,
                      100.0 * frac, resid / elCell);
            }
        }
    }
}

// Are the setups where the files say they are?
//
// Every voxel this tool produces is placed by one transform per scan: the pose,
// applied or not according to viewer::decideFrame. That decision is made from a
// single scan's points, and it can only see the pose's TRANSLATION — a rotation
// leaves every distance-to-a-centre exactly as it was. So on a file whose first
// setup is the registration datum, with a translation of millimetres and a
// rotation of ninety degrees, the test has no information and the answer is a
// coin toss that turns the whole cloud a quarter turn.
//
// Nothing inside one scan can settle that. Between scans it is easy: registered
// scans of one site describe the same surfaces, so the right hypothesis is the one
// whose cloud lands on the others. This tries both for every scan and reports what
// each is worth, which turns "the setups look wrong" into a number.
static void registrationCheck(const std::vector<std::string>& paths, const Options& opt,
                              const std::vector<carve::SetupView>& views,
                              const carve::Params& params, Out& o) {
    std::vector<Cloud> clouds;
    for (const std::string& path : paths) {
        e57::Reader r;
        std::string e;
        if (!r.open(path, e)) continue;
        for (size_t i = 0; i < r.scanCount(); ++i) {
            const e57::Scan& s = r.scan(i);
            if (!(s.field("cartesianX") && s.field("cartesianY") && s.field("cartesianZ")))
                continue;
            const viewer::FrameDecision fd = viewer::decideFrame(r, i);
            const viewer::Rigid R = viewer::rigidFromPose(s.pose);

            Cloud c;
            c.name = s.name.empty() ? path : s.name;
            c.applied       = fd.applyPose();
            c.uninformative = fd.uninformative;
            c.rotationDeg   = fd.poseRotationDeg;
            c.translationM  = fd.poseTranslationM;
            c.reason        = fd.reason;
            for (int k = 0; k < 3; ++k) c.setup[k] = s.pose.t[k];

            std::vector<std::string> want{"cartesianX", "cartesianY", "cartesianZ"};
            size_t invIdx = SIZE_MAX;
            if (s.field("cartesianInvalidState")) { invIdx = 3; want.push_back("cartesianInvalidState"); }
            const uint64_t stride =
                std::max<uint64_t>(1, s.recordCount / std::max<uint64_t>(1, kCloudSamples));
            uint64_t seen = 0;
            bool first = true;
            std::string err;
            r.readPoints(i, want, [&](const e57::PointBlock& b) {
                for (size_t k = 0; k < b.count; ++k, ++seen) {
                    if (seen % stride) continue;
                    if (invIdx != SIZE_MAX && b.columns[invIdx][k] != 0.0) continue;
                    const double x = b.columns[0][k], y = b.columns[1][k], z = b.columns[2][k];
                    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;
                    const double raw[3] = {x, y, z};
                    double w[3] = {x, y, z};
                    R.apply(w[0], w[1], w[2]);
                    // Both hypotheses are kept: the chosen one, and the one the
                    // decision rejected. Judging them against the same corpus is
                    // the whole point.
                    const double* use   = c.applied ? w : raw;
                    const double* other = c.applied ? raw : w;
                    for (int a = 0; a < 3; ++a) {
                        c.xyz.push_back(float(use[a]));
                        c.alt.push_back(float(other[a]));
                    }
                    if (first) {
                        first = false;
                        for (int a = 0; a < 3; ++a) { c.lo[a] = c.hi[a] = use[a]; }
                    } else {
                        for (int a = 0; a < 3; ++a) {
                            c.lo[a] = std::min(c.lo[a], use[a]);
                            c.hi[a] = std::max(c.hi[a], use[a]);
                        }
                    }
                }
                return true;
            }, err);
            for (size_t k = 0; k + 2 < c.xyz.size(); k += 3)
                c.cells.insert(cellKey(c.xyz[k], c.xyz[k + 1], c.xyz[k + 2]));
            if (!c.xyz.empty()) clouds.push_back(std::move(c));
        }
    }
    if (clouds.size() < 2) {
        o.add("registration: needs two or more scans to check\n\n");
        return;
    }

    o.add("registration\n");
    for (size_t i = 0; i < clouds.size(); ++i) {
        const Cloud& c = clouds[i];
        o.add("  [%zu] %s\n", i, c.name.c_str());
        o.add("      setup at (%.3f, %.3f, %.3f)   pose %.1f deg / %.3f m   pose is %s\n",
              c.setup[0], c.setup[1], c.setup[2], c.rotationDeg, c.translationM,
              c.applied ? "APPLIED" : "not applied");
        o.add("      its returns span x[%.1f, %.1f] y[%.1f, %.1f] z[%.1f, %.1f]\n",
              c.lo[0], c.hi[0], c.lo[1], c.hi[1], c.lo[2], c.hi[2]);
        if (c.uninformative)
            o.add("      *** %s\n", c.reason.c_str());
    }

    // Each scan judged against the rest of the corpus, under both hypotheses.
    //
    // Two numbers, and only the second decides. Overlap says how much of this
    // scan the others also saw, which is useful context but cannot settle
    // anything — a ground plane is unchanged by turning it about a vertical axis.
    // Contradiction says how much of this scan sits in space the others saw
    // straight through, which is impossible for a real surface and which no
    // symmetry of the scene can hide.
    o.add("\n  Each scan against the rest of the corpus, under both hypotheses:\n");
    o.add("      %-22s %9s %9s   %9s %9s   %s\n", "scan",
          "overlap", "(other)", "conflict", "(other)", "verdict");
    size_t suspect = 0;
    const bool haveViews = views.size() == clouds.size();
    for (size_t i = 0; i < clouds.size(); ++i) {
        std::unordered_set<int64_t> others;
        for (size_t j = 0; j < clouds.size(); ++j) {
            if (j == i) continue;
            others.insert(clouds[j].cells.begin(), clouds[j].cells.end());
        }
        const double oa = agreement(clouds[i].xyz, others);
        const double ob = agreement(clouds[i].alt, others);
        const Conflict ca = haveViews ? contradicted(clouds[i].xyz, views, i, params) : Conflict{};
        const Conflict cb = haveViews ? contradicted(clouds[i].alt, views, i, params) : Conflict{};

        const char* verdict = "ok";
        // Both measurements have to prefer the alternative, and neither is
        // trusted alone.
        //
        // Neither has a meaningful absolute scale. Overlap is high for anything
        // that lands on ground, because a flat plane is unchanged by turning it.
        // Conflict carries a floor of twenty to forty per cent from the raster
        // itself: ground at grazing incidence puts a cell's near edge metres in
        // front of its far edge, so a point on the far edge reads as sitting in
        // cleared space through no fault of the registration. Picking a threshold
        // on either one alone means picking a number to fit the last dataset,
        // which is how the previous two versions of this went wrong.
        //
        // What is trustworthy is the direction both move in together. A wrongly
        // placed scan overlaps the others worse AND contradicts them more; a
        // correctly placed one does neither. On the fixtures the two agree in
        // every case and the margins are fifteen points, not five.
        const double kMargin = 0.05;
        if (ca.rate >= 0 && cb.rate >= 0 && cb.judged > 0.5 * ca.judged &&
            ob > oa + kMargin && cb.rate < ca.rate - kMargin) {
            verdict = "*** THE POSE IS BEING APPLIED THE WRONG WAY ***";
            ++suspect;
        }
        o.add("      %-22s %8.1f%% %8.1f%%   %8.1f%% %8.1f%%   %s\n",
              clouds[i].name.c_str(), 100.0 * oa, 100.0 * ob,
              100.0 * ca.rate, 100.0 * cb.rate, verdict);
    }
    if (suspect)
        o.add("\n  %zu scan(s) do not agree with the rest. Every voxel is placed by these\n"
              "  transforms, so a scan in the wrong place carves its evidence into the\n"
              "  wrong part of the site and leaves the right part unobserved.\n", suspect);
    else
        o.add("\n  All scans land on each other under the transform being used.\n");
    o.add("\n");
    (void)opt;
}

int selfTest(const std::vector<std::string>& paths, const Options& opt, std::string& out) {
    Out o{out};
    o.add("e57cov evidence self-test — build %s\n", ver::describe());
    o.add("%zu file(s), max range %.1f m\n\n", paths.size(), opt.maxRange);

    rimg::Options ro;
    ro.maxRange         = opt.maxRange;
    ro.blindCone        = opt.blindCone;
    ro.noReturnRadius   = opt.noReturnRadius;
    ro.noReturnFraction = opt.noReturnFraction;

    std::vector<std::unique_ptr<e57::Reader>> readers;
    std::vector<std::unique_ptr<rimg::RangeImage>> images;
    std::vector<std::string> names;
    // Which reader and which scan each image came from, so the raster audit can
    // go back to the file rather than trusting anything build() kept.
    std::vector<size_t> readerOf, scanOf;
    for (const std::string& path : paths) {
        auto r = std::make_unique<e57::Reader>();
        std::string e;
        if (!r->open(path, e)) { o.add("%s: %s\n", path.c_str(), e.c_str()); continue; }
        for (size_t i = 0; i < r->scanCount(); ++i) {
            auto img = std::make_unique<rimg::RangeImage>();
            std::string rerr;
            if (!rimg::build(*r, i, ro, *img, rerr)) {
                o.add("%s [%zu]: range image failed — %s\n", path.c_str(), i, rerr.c_str());
                continue;
            }
            images.push_back(std::move(img));
            names.push_back(r->scan(i).name.empty() ? path : r->scan(i).name);
            readerOf.push_back(readers.size());
            scanOf.push_back(i);
        }
        readers.push_back(std::move(r));
    }
    if (images.empty()) { o.add("no usable scans\n"); return 1; }

    // What each scan decided about its own blind cone. build() has already marked
    // them, so this reads the same images the carve will and reports them rather
    // than reaching a second opinion of its own.
    {
        std::vector<rimg::RangeImage*> raw;
        for (auto& im : images) raw.push_back(im.get());
        const rimg::ConeVerdict v = rimg::summariseBlindCones(raw, ro);
        o.add("blind cone: %s\n            %s\n\n", v.headline.c_str(), v.why.c_str());
    }

    carve::Params p;
    p.voxelSize     = 0.05;
    p.surfaceMargin = 0.5 * p.voxelSize * 1.7320508075688772;
    p.maxRange      = opt.maxRange;

    // The registration check needs the setups the carve would use, because the
    // question it asks — did another setup see through this point? — is answered
    // by the same primitive everything else is.
    std::vector<carve::SetupView> views;
    views.reserve(images.size());
    for (auto& im : images) views.push_back(carve::makeSetupView(*im));
    registrationCheck(paths, opt, views, p, o);

    // Roughly this many cells per scan, spread evenly over the whole raster
    // rather than taken from one corner: a mapping can be right for one band and
    // wrong for another, and that is the failure worth catching.
    constexpr uint64_t kSamplesWanted = 40000;

    int failures = 0;
    double totalPredicted = 0;
    for (size_t k = 0; k < images.size(); ++k) {
        const rimg::RangeImage& im = *images[k];
        const carve::SetupView& s = views[k];
        const viewer::Rigid fwd = im.hasPose ? viewer::rigidFromPose(im.pose) : viewer::Rigid{};

        o.add("  [%zu] %s\n", k, names[k].c_str());
        if (im.rows == 0 || im.cols == 0 || im.map.elByRow.size() != im.rows ||
            im.map.azByCol.size() != im.cols) {
            o.add("      no usable raster\n\n");
            ++failures;
            continue;
        }
        o.add("      setup at (%.3f, %.3f, %.3f)   raster %u x %u   mapping %s\n",
              fwd.t[0], fwd.t[1], fwd.t[2], im.rows, im.cols,
              im.map.valid ? "accepted" : "*** REFUSED — every lookup returns nothing ***");
        if (im.tiltDeg > 0.001)
            o.add("      levelling: leaned %.3f deg toward %.1f deg, %.1f%% of the within-row "
                  "elevation spread\n", im.tiltDeg, im.tiltTowardDeg, 100.0 * im.tiltExplained);
        if (!im.map.valid) ++failures;
        // The reason, which this left out and should never have: "refused" on its
        // own sends you off to run something else to find out why.
        if (!im.diag.note.empty()) o.add("      why: %s\n", im.diag.note.c_str());
        if (originAudit(im, fwd, o)) ++failures;
        rasterAudit(*readers[readerOf[k]], scanOf[k], im, o);
        spreadAudit(*readers[readerOf[k]], scanOf[k], im, o);

        // A world point in the direction cell (r, c) looked, at range rho.
        auto pointAt = [&](uint32_t r, uint32_t c, double rho, double w[3]) {
            const double el = im.map.elByRow[r], az = im.map.azByCol[c];
            const double ce = std::cos(el);
            // The tables are angles in the INSTRUMENT'S frame, so the direction has
            // to come back out of that frame before the pose puts it in the world.
            // Skipping this misplaces every probe by the tripod's lean, which is
            // nothing next to "is it in front of the surface" and everything next
            // to "is it within half a voxel of it".
            double q[3] = {rho * ce * std::cos(az), rho * ce * std::sin(az),
                           rho * std::sin(el)};
            im.fromInstrument(q[0], q[1], q[2]);
            for (int i = 0; i < 3; ++i)
                w[i] = fwd.R[3 * i + 0] * q[0] + fwd.R[3 * i + 1] * q[1] +
                       fwd.R[3 * i + 2] * q[2] + fwd.t[i];
        };
        auto ask = [&](uint32_t r, uint32_t c, double rho) {
            double w[3];
            pointAt(r, c, rho, w);
            return carve::evidenceAt(s, p, w[0], w[1], w[2]);
        };

        // Which cell a probe actually landed in. A probe is built from cell
        // (r, c)'s own table entry, so it should come back to (r, c) — but the
        // reverse index is binned, and on a raster with uneven rows a direction
        // near a bin edge can resolve to the neighbour.
        //
        // That distinction is the whole reason this test used to cry wolf. A
        // probe that resolved to a neighbouring cell was never a test of the
        // carve at all: it asked about a different cell, got that cell's answer,
        // and was scored as a failure of this one. On the five real setups the
        // mapping round trip runs 91.75 to 99.43 per cent, so between half a per
        // cent and eight per cent of probes were mis-scored — and against a flat
        // 99 per cent bar that printed SELF-TEST FAILED on data that was fine.
        //
        // So they are separated. Probes that reached their own cell are the test,
        // and they have to be perfect: the arithmetic is exact and there is
        // nothing left to be approximately right about. Probes that did not are
        // reported as what they are — the mapping's resolution, already measured
        // and already gated by the round trip in range_image — and never counted
        // as a fault here.
        auto landedHere = [&](uint32_t r, uint32_t c, double rho) {
            double w[3];
            pointAt(r, c, rho, w);
            s.worldToScanner.apply(w[0], w[1], w[2]);
            im.toInstrument(w[0], w[1], w[2]);
            double az, el, rr;
            rimg::toSpherical(w[0], w[1], w[2], az, el, rr);
            uint32_t gr = 0, gc = 0;
            return im.cellOf(az, el, gr, gc) && gr == r && gc == c;
        };

        const uint64_t cells = uint64_t(im.rows) * im.cols;
        const uint32_t stride = uint32_t(std::max<uint64_t>(1, cells / kSamplesWanted));

        // Per line: probes that reached their own cell and were right, probes
        // that reached it at all, and probes the mapping sent to a neighbour.
        struct Tally { uint64_t ok = 0, n = 0, lost = 0; };
        Tally near_, on, beyond, half, clean_, past, unsampled;

        // One probe. Fired only if it reaches the cell it was built from, and
        // then required to be exactly right.
        auto probe = [&](uint32_t r, uint32_t c, double rho, uint8_t want, Tally& t) {
            if (!landedHere(r, c, rho)) { ++t.lost; return; }
            ++t.n;
            if (ask(r, c, rho) == want) ++t.ok;
        };

        uint64_t nHit = 0, nHitTested = 0, nEmpty = 0, nUnsampled = 0, skipped = 0;

        uint64_t i = 0;
        for (uint32_t r = 0; r < im.rows; ++r) {
            for (uint32_t c = 0; c < im.cols; ++c, ++i) {
                if (i % stride) continue;
                const rimg::Status st = im.statusAt(r, c);
                const double d = im.rangeAt(r, c);

                if (st == rimg::Status::OutsideFov) {
                    ++nUnsampled;
                    // Two ranges, both of which have to say nothing, so this one
                    // is scored as a pair rather than through `probe`.
                    const bool a = landedHere(r, c, 5.0);
                    const bool b = landedHere(r, c, 0.4 * opt.maxRange);
                    if (!a || !b) { ++unsampled.lost; continue; }
                    ++unsampled.n;
                    if (ask(r, c, 5.0) == 0 && ask(r, c, 0.4 * opt.maxRange) == 0)
                        ++unsampled.ok;
                    continue;
                }
                if (st == rimg::Status::NoReturn) {
                    ++nEmpty;
                    const double clear = std::min(d, opt.maxRange);
                    if (clear < 2.0) { ++skipped; continue; }
                    probe(r, c, 0.5 * clear, carve::kVisible, half);
                    probe(r, c, clear + 2.0, 0, past);
                    // Cleanly, not just at one point: every metre of the ray. A
                    // ray is only scored if every metre of it reached this cell.
                    bool reached = true, ok = true;
                    for (double rho = 1.0; rho <= clear - 0.5; rho += 1.0) {
                        if (!landedHere(r, c, rho)) { reached = false; break; }
                        if (ask(r, c, rho) != carve::kVisible) { ok = false; break; }
                    }
                    if (!reached) ++clean_.lost;
                    else { ++clean_.n; if (ok) ++clean_.ok; }
                    continue;
                }
                // A return. Too near and the three probes are not separable; past
                // the rated range and the setting, not the surface, decides.
                ++nHit;
                if (d < 2.0 || d > opt.maxRange - 2.0) { ++skipped; continue; }
                ++nHitTested;
                probe(r, c, 0.5 * d, carve::kVisible,  near_);
                probe(r, c, d,       carve::kOccupied, on);
                probe(r, c, d + 1.0, 0,                beyond);
            }
        }

        auto pct = [](uint64_t a, uint64_t b) { return b ? 100.0 * double(a) / double(b) : -1.0; };
        // A line fails when a probe that reached its own cell got the wrong
        // answer. Nothing else: the arithmetic on that path is exact, so one
        // wrong answer is a fault and a thousand near misses are not.
        auto line = [&](const char* what, const Tally& t, const char* should) {
            if (!t.n && !t.lost) { o.add("      %-22s none in this scan\n", what); return; }
            if (!t.n) {
                o.add("      %-22s *** NOTHING TESTED — all %llu probes were sent to a "
                      "neighbouring cell ***\n", what, (unsigned long long)t.lost);
                ++failures;
                return;
            }
            const double f = pct(t.ok, t.n);
            o.add("      %-22s %7.2f%% of %llu  %s%s", what, f,
                  (unsigned long long)t.n, should, t.ok == t.n ? "" : "   *** BROKEN ***");
            if (t.lost)
                o.add("   (+%llu, %.1f%%, landed in a neighbouring cell — the mapping's "
                      "resolution, not this)", (unsigned long long)t.lost,
                      100.0 * double(t.lost) / double(t.n + t.lost));
            o.add("\n");
            if (t.ok != t.n) ++failures;
        };
        o.add("      cells: %llu returns, %llu empty, %llu unsampled  (%llu sampled, "
              "%llu skipped as too near or too far)\n",
              (unsigned long long)nHit, (unsigned long long)nEmpty,
              (unsigned long long)nUnsampled,
              (unsigned long long)(nHit + nEmpty + nUnsampled),
              (unsigned long long)skipped);
        line("in front of a return", near_,     "read VISIBLE");
        line("on a return",          on,        "read OCCUPIED");
        line("behind a return",      beyond,    "read nothing");
        line("halfway along empty",  half,      "read VISIBLE");
        line("all along empty",      clean_,    "clear with no gaps");
        line("past an empty ray",    past,      "read nothing");
        line("along unsampled",      unsampled, "read nothing");

        // What this raster says the setup ought to clear, from the file alone.
        // Each cell is a pencil of solid angle dAz*dEl*cos(el) reaching however
        // far that cell established; the volume of a pencil of solid angle W out
        // to rho is W*rho^3/3. Setups overlap, so this over-counts the union — but
        // a carve reporting a small fraction of it is not overlapping, it is
        // failing.
        double predicted = 0;
        const double dAz = std::fabs(im.map.azSpanRad) / double(std::max(1u, im.cols - 1));
        for (uint32_t r = 0; r < im.rows; ++r) {
            const double elLo = (r == 0) ? im.map.elByRow[0]
                                         : 0.5 * (im.map.elByRow[r - 1] + im.map.elByRow[r]);
            const double elHi = (r + 1 == im.rows)
                                    ? im.map.elByRow[r]
                                    : 0.5 * (im.map.elByRow[r] + im.map.elByRow[r + 1]);
            const double dEl = std::fabs(elHi - elLo);
            const double w = dAz * dEl * std::cos(im.map.elByRow[r]);
            for (uint32_t c = 0; c < im.cols; ++c) {
                const rimg::Status st = im.statusAt(r, c);
                if (st == rimg::Status::OutsideFov) continue;
                double rho = im.rangeAt(r, c);
                if (st == rimg::Status::Hit) rho = std::max(0.0, rho - p.surfaceMargin);
                rho = std::min(rho, opt.maxRange);
                predicted += w * rho * rho * rho / 3.0;
            }
        }
        totalPredicted += predicted;
        o.add("      this raster alone should clear about %.0f m^3\n\n", predicted);
    }

    const double sphere = 4.0 / 3.0 * 3.14159265358979323846 *
                          opt.maxRange * opt.maxRange * opt.maxRange;
    o.add("Together the rasters say these setups clear about %.0f m^3, before any\n"
          "overlap between them is taken off. One setup's whole range sphere is\n"
          "%.0f m^3, so a carve that reports far less visible than this has a fault\n"
          "between the raster and the voxels, not in the data.\n\n",
          totalPredicted, sphere);
    o.add(failures ? "*** SELF-TEST FAILED ***\n" : "Self-test passed.\n");
    return failures;
}

int scanReport(const std::vector<std::string>& paths, const Options& opt, std::string& out) {
    Out o{out};
    o.add("e57cov scan report — build %s\n", ver::describe());
    o.add("%zu file(s), max range %.1f m\n", paths.size(), opt.maxRange);

    // Each file reports its own scans' own blind-cone decisions. There used to be a
    // pass here that read the row bands out of every file first, took a
    // corpus-wide vote on which end of the raster the cone is at, and then forced
    // that answer on every file so the report could not disagree with the carve.
    //
    // The carve no longer has a corpus-wide answer to agree with: what makes a band
    // the instrument's cone is a property of the instrument, each scan was taken
    // with that instrument, and so each scan decides — see rimg::markBlindCone. The
    // vote is gone from both, which is what keeps them in step.
    int failures = 0;
    for (const std::string& p : paths) {
        failures += scanReport(p, opt, out);
        Out{out}.add("\n");
    }
    if (failures) Out{out}.add("%d file(s) reported problems.\n", failures);
    else          Out{out}.add("No problems reported.\n");
    return failures;
}

} // namespace report
