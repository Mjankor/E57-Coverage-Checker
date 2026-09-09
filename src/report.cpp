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
        // Only the comparison decides, and only when the alternative was judged
        // over a comparable share of the scan. Both rates carry a floor from the
        // raster itself: ground at grazing incidence puts a cell's near edge
        // metres in front of its far edge, and a point on the far edge reads as
        // being in cleared space through no fault of the registration.
        if (ca.rate >= 0 && cb.rate >= 0 && cb.judged > 0.5 * ca.judged &&
            ca.rate > 0.20 && cb.rate < ca.rate * 0.6) {
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
        }
        readers.push_back(std::move(r));
    }
    if (images.empty()) { o.add("no usable scans\n"); return 1; }

    // The same corpus-wide cone decision the carve makes, so this describes the
    // run rather than a differently-decided one.
    {
        std::vector<rimg::RangeImage*> raw;
        for (auto& im : images) raw.push_back(im.get());
        const rimg::ConeVerdict v = rimg::markBlindConeAcrossCorpus(raw, ro);
        o.add("blind cone: %s\n            %s\n\n",
              v.decided ? (v.atFirstRow ? "the START of each raster" : "the END of each raster")
                        : "not identified",
              v.why.c_str());
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
        if (!im.map.valid) ++failures;

        // A world point in the direction cell (r, c) looked, at range rho.
        auto pointAt = [&](uint32_t r, uint32_t c, double rho, double w[3]) {
            const double el = im.map.elByRow[r], az = im.map.azByCol[c];
            const double ce = std::cos(el);
            const double q[3] = {rho * ce * std::cos(az), rho * ce * std::sin(az),
                                 rho * std::sin(el)};
            for (int i = 0; i < 3; ++i)
                w[i] = fwd.R[3 * i + 0] * q[0] + fwd.R[3 * i + 1] * q[1] +
                       fwd.R[3 * i + 2] * q[2] + fwd.t[i];
        };
        auto ask = [&](uint32_t r, uint32_t c, double rho) {
            double w[3];
            pointAt(r, c, rho, w);
            return carve::evidenceAt(s, p, w[0], w[1], w[2]);
        };

        const uint64_t cells = uint64_t(im.rows) * im.cols;
        const uint32_t stride = uint32_t(std::max<uint64_t>(1, cells / kSamplesWanted));

        uint64_t nHit = 0, nHitTested = 0, hitNearOk = 0, hitOnOk = 0, hitBeyondOk = 0;
        uint64_t nEmpty = 0, emptyHalfOk = 0, emptyPastOk = 0, emptyClean = 0;
        uint64_t nUnsampled = 0, unsampledOk = 0;
        uint64_t skipped = 0;

        uint64_t i = 0;
        for (uint32_t r = 0; r < im.rows; ++r) {
            for (uint32_t c = 0; c < im.cols; ++c, ++i) {
                if (i % stride) continue;
                const rimg::Status st = im.statusAt(r, c);
                const double d = im.rangeAt(r, c);

                if (st == rimg::Status::OutsideFov) {
                    ++nUnsampled;
                    if (ask(r, c, 5.0) == 0 && ask(r, c, 0.4 * opt.maxRange) == 0) ++unsampledOk;
                    continue;
                }
                if (st == rimg::Status::NoReturn) {
                    ++nEmpty;
                    const double clear = std::min(d, opt.maxRange);
                    if (clear < 2.0) { ++skipped; continue; }
                    if (ask(r, c, 0.5 * clear) == carve::kVisible) ++emptyHalfOk;
                    if (ask(r, c, clear + 2.0) == 0) ++emptyPastOk;
                    // Cleanly, not just at one point: every metre of the ray.
                    bool clean = true;
                    for (double rho = 1.0; rho <= clear - 0.5; rho += 1.0)
                        if (ask(r, c, rho) != carve::kVisible) { clean = false; break; }
                    if (clean) ++emptyClean;
                    continue;
                }
                // A return. Too near and the three probes are not separable; past
                // the rated range and the setting, not the surface, decides.
                ++nHit;
                if (d < 2.0 || d > opt.maxRange - 2.0) { ++skipped; continue; }
                ++nHitTested;
                if (ask(r, c, 0.5 * d) == carve::kVisible)  ++hitNearOk;
                if (ask(r, c, d)       == carve::kOccupied) ++hitOnOk;
                if (ask(r, c, d + 1.0) == 0)                ++hitBeyondOk;
            }
        }

        auto pct = [](uint64_t a, uint64_t b) { return b ? 100.0 * double(a) / double(b) : -1.0; };
        auto line = [&](const char* what, uint64_t ok, uint64_t n, const char* should) {
            if (!n) { o.add("      %-22s none in this scan\n", what); return; }
            const double f = pct(ok, n);
            o.add("      %-22s %7.2f%% of %llu  %s%s\n", what, f,
                  (unsigned long long)n, should, f >= 99.0 ? "" : "   *** BROKEN ***");
            if (f < 99.0) ++failures;
        };
        o.add("      cells: %llu returns, %llu empty, %llu unsampled  (%llu sampled, "
              "%llu skipped as too near or too far)\n",
              (unsigned long long)nHit, (unsigned long long)nEmpty,
              (unsigned long long)nUnsampled,
              (unsigned long long)(nHit + nEmpty + nUnsampled),
              (unsigned long long)skipped);
        line("in front of a return", hitNearOk,   nHitTested, "read VISIBLE");
        line("on a return",          hitOnOk,     nHitTested, "read OCCUPIED");
        line("behind a return",      hitBeyondOk, nHitTested, "read nothing");
        line("halfway along empty",  emptyHalfOk, nEmpty, "read VISIBLE");
        line("all along empty",      emptyClean,  nEmpty, "clear with no gaps");
        line("past an empty ray",    emptyPastOk, nEmpty, "read nothing");
        line("along unsampled",      unsampledOk, nUnsampled, "read nothing");

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
