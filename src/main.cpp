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

#include "e57.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
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

// Which of the two coordinate conventions this scan uses. Both appear in the
// wild, sometimes within one corpus assembled from several jobs, so it is
// detected rather than assumed.
const char* frameGuess(const e57::Scan& s, const Stats& st) {
    if (!st.any) return "unknown (no points decoded)";
    const double n = double(st.decoded);
    const double cx = st.sumX / n, cy = st.sumY / n, cz = st.sumZ / n;
    const double centroid = std::sqrt(cx * cx + cy * cy + cz * cz);
    const double poseT    = std::sqrt(s.pose.t[0] * s.pose.t[0] +
                                      s.pose.t[1] * s.pose.t[1] +
                                      s.pose.t[2] * s.pose.t[2]);
    // A scanner-local cloud is centred near its own origin. If the centroid
    // instead sits near where the pose says the scanner is, the producer has
    // already applied the pose.
    if (centroid < 5.0)                       return "scanner-local (apply pose)";
    if (poseT > 1.0 && std::fabs(centroid - poseT) < 0.25 * poseT)
                                              return "global (pose already applied)";
    if (poseT < 1e-9 && centroid > 50.0)      return "global (identity pose, far from origin)";
    return "ambiguous — inspect manually";
}

int info(const std::string& path, bool verifyCrc) {
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

        // Question 1: how are no-returns represented?
        if (invName) {
            const double pct = s.recordCount ? 100.0 * double(st.invalid) / double(s.recordCount) : 0.0;
            std::printf("      no-return : %s present — %llu of %llu records (%.2f%%)\n",
                        invName, (unsigned long long)st.invalid,
                        (unsigned long long)s.recordCount, pct);
            if (st.invalid == 0)
                std::printf("                  field present but never set: treat as NOT storing no-returns\n");
        } else {
            std::printf("      no-return : NO invalid-state field — no-returns are not stored.\n"
                        "                  Field of view must be recovered from angular extent\n"
                        "                  (DESIGN.md §4, resolution path 2).\n");
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
            // Question 2: which coordinate frame?
            std::printf("      frame     : %s\n", frameGuess(s, st));
        }
    }

    std::printf("\n");
    return failures == 0 ? 0 : 1;
}

void usage() {
    std::printf(
        "e57cov — E57 Coverage Checker\n"
        "\n"
        "usage: e57cov info [--crc] <file.e57> [more.e57 ...]\n"
        "\n"
        "  info    Inspect scans and audit format conventions. Reports how each\n"
        "          file represents no-return rays and which coordinate frame its\n"
        "          points are in, and cross-checks the decode against the file's\n"
        "          own recordCount and cartesianBounds.\n"
        "\n"
        "  --crc   Also verify every page checksum (costs a full pass).\n"
        "\n"
        "The visibility pipeline (index / carve) is not built yet; see DESIGN.md.\n");
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 2; }

    const std::string cmd = argv[1];
    if (cmd == "-h" || cmd == "--help" || cmd == "help") { usage(); return 0; }
    if (cmd != "info") {
        std::printf("unknown command '%s'\n\n", cmd.c_str());
        usage();
        return 2;
    }

    bool                     crc = false;
    std::vector<std::string> paths;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--crc") == 0) crc = true;
        else                                    paths.push_back(argv[i]);
    }
    if (paths.empty()) { usage(); return 2; }

    int failures = 0;
    for (const auto& p : paths) failures += info(p, crc);
    if (failures)
        std::printf("%d file(s) reported problems.\n", failures);
    return failures == 0 ? 0 : 1;
}
