#include "scan_check.h"

#include "frame.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <unordered_map>

namespace check {

const char* kindName(Kind k) {
    switch (k) {
    case Kind::Structured: return "structured";
    case Kind::Unified:    return "unified/merged";
    case Kind::Ambiguous:  return "ambiguous";
    }
    return "?";
}

namespace {

bool hasField(const e57::Scan& s, const char* n) { return s.field(n) != nullptr; }

// A row/column index that never varies carries no grid information — some
// writers emit the field and leave it constant.
bool hasVaryingField(const e57::Scan& s, const char* n) {
    const e57::ProtoField* f = s.field(n);
    if (!f) return false;
    if (f->isPacked() && f->bits == 0) return false;
    return true;
}

std::string fmt(const char* f, double a, double b = 0, double c = 0) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), f, a, b, c);
    return buf;
}

} // namespace

Result classifyMetadata(const e57::Scan& s) {
    Result r;
    bool structured = false;

    if (hasField(s, "sphericalRange")) {
        r.evidence.push_back("spherical coordinates in prototype — inherently single-origin");
        structured = true;
    }
    if (s.hasIndexBounds && s.rowMax > s.rowMin && s.colMax > s.colMin) {
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "indexBounds declares a %lld x %lld grid",
                      (long long)(s.rowMax - s.rowMin + 1),
                      (long long)(s.colMax - s.colMin + 1));
        r.evidence.push_back(buf);
        structured = true;
    }
    if (hasVaryingField(s, "rowIndex") && hasVaryingField(s, "columnIndex")) {
        r.evidence.push_back("rowIndex and columnIndex present and varying");
        structured = true;
    }
    if (s.hasPointGrouping) {
        r.evidence.push_back("pointGroupingSchemes present");
        structured = true;
    }

    if (structured) {
        r.kind    = Kind::Structured;
        r.summary = "structured scan";
        return r;
    }

    r.kind = Kind::Ambiguous;
    r.summary = "no gridding metadata";
    r.evidence.push_back("no spherical coordinates, no indexBounds, no row/column "
                         "index, no pointGroupingSchemes");
    if (!s.hasPose)
        r.evidence.push_back("no pose — a merged cloud typically has none");
    return r;
}

Result classify(e57::Reader& reader, size_t scanIndex, const Thresholds& t) {
    const e57::Scan& s = reader.scan(scanIndex);

    // The geometric test needs cartesian points. Spherical scans are already
    // proven single-origin by their own storage format, so there is nothing
    // left to test.
    const bool cartesian = hasField(s, "cartesianX") && hasField(s, "cartesianY") &&
                           hasField(s, "cartesianZ");
    if (!cartesian || s.recordCount == 0) return classifyMetadata(s);

    // One pass, serving both the frame decision and the test below it. This used
    // to be three: decideFrame made one, and the binning made another, and the
    // caller in indexer::survey had already made a third.
    std::vector<double> xyz;
    std::string err;
    if (!reader.sampleXYZ(scanIndex, t.sampleTarget, xyz, err)) {
        Result r = classifyMetadata(s);
        r.evidence.push_back("geometric test skipped: " + err);
        return r;
    }
    return classifyFromSample(s, xyz, viewer::decideFrameFromSample(s, xyz), t);
}

Result classifyFromSample(const e57::Scan& s, const std::vector<double>& xyz,
                          const viewer::FrameDecision& frame, const Thresholds& t) {
    Result r = classifyMetadata(s);

    const bool cartesian = hasField(s, "cartesianX") && hasField(s, "cartesianY") &&
                           hasField(s, "cartesianZ");
    if (!cartesian || s.recordCount == 0) return r;

    // Bin about wherever the scanner actually is. This used to be guessed from
    // the sample centroid, which fails exactly when setups are close together:
    // an outdoor scan whose centroid sits a few metres off its own origin was
    // tested about the pose translation instead, scattering the direction bins
    // and reporting a perfectly good single setup as merged. The frame
    // decision answers the same question properly.
    double ox = 0, oy = 0, oz = 0;
    if (!frame.applyPose() && s.hasPose) {
        // Points are already in the file frame, so the scanner is at the pose
        // translation.
        ox = s.pose.t[0]; oy = s.pose.t[1]; oz = s.pose.t[2];
    }
    r.evidence.push_back(frame.reason);

    // Bin by direction, track min and max range per bin.
    struct Bin { float lo = 1e30f, hi = -1e30f; int n = 0; };
    std::unordered_map<uint32_t, Bin> bins;
    bins.reserve(t.sampleTarget / 8);

    const double binRad  = t.binDegrees * 3.14159265358979 / 180.0;
    const int    nAz     = std::max(1, int(std::round(2.0 * 3.14159265358979 / binRad)));
    const int    nEl     = std::max(1, int(std::round(3.14159265358979 / binRad)));
    uint64_t used = 0;

    const size_t n = xyz.size() / 3;
    for (size_t i = 0; i < n; ++i) {
        const double dx = xyz[3 * i] - ox, dy = xyz[3 * i + 1] - oy, dz = xyz[3 * i + 2] - oz;
        const double rr = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (rr < 1e-6) continue;
        const double az = std::atan2(dy, dx) + 3.14159265358979;     // 0..2pi
        const double el = std::acos(std::clamp(dz / rr, -1.0, 1.0)); // 0..pi
        int ia = std::min(nAz - 1, int(az / binRad));
        int ie = std::min(nEl - 1, int(el / binRad));
        if (ia < 0) ia = 0;
        if (ie < 0) ie = 0;
        Bin& bin = bins[uint32_t(ie) * uint32_t(nAz) + uint32_t(ia)];
        bin.lo = std::min(bin.lo, float(rr));
        bin.hi = std::max(bin.hi, float(rr));
        ++bin.n;
        ++used;
    }

    size_t tested = 0, multi = 0;
    for (const auto& kv : bins) {
        if (kv.second.n < t.minPointsPerBin) continue;
        ++tested;
        if (double(kv.second.hi - kv.second.lo) > t.multiSurfaceSpreadM) ++multi;
    }

    r.pointsSampled = size_t(used);
    r.binsTested    = tested;
    if (tested < 200) {
        r.evidence.push_back("geometric test inconclusive: too few populated bins");
        return r;
    }

    r.multiSurfaceFraction = double(multi) / double(tested);
    r.looksMerged = r.multiSurfaceFraction > t.multiSurfaceFraction;
    r.evidence.push_back(fmt("%.1f%% of %.0f populated direction bins hold multiple surfaces",
                             100.0 * r.multiSurfaceFraction, double(tested)));

    // Declared metadata WINS. A scan that states an indexBounds grid, or carries
    // a varying row and column index, has said it is one setup in the file
    // format's own terms; this measurement rises with scene scale — 0 per cent in
    // a small room and 29 in a yard, for the same geometry — so it does not get to
    // call that file a merged cloud. The number is still reported, because it is
    // worth seeing, and the header note says what it is worth.
    if (r.kind == Kind::Structured) {
        if (r.looksMerged)
            r.evidence.push_back(fmt("that is over the %.0f%% the range-spread test calls "
                                     "merged, but this scan declares a sampling grid, which "
                                     "is the stronger evidence — see scan_check.h",
                                     100.0 * t.multiSurfaceFraction));
        return r;
    }

    if (r.looksMerged) {
        // No grid declared, so this is the only evidence there is. A label, and
        // the scan is still indexed and drawn: see indexer::ScanRef::usable.
        r.kind = Kind::Unified;
        r.summary = fmt("no grid declared, and %.0f%% of directions hit multiple surfaces",
                        100.0 * r.multiSurfaceFraction);
        return r;
    }

    r.kind = Kind::Structured;
    r.summary = "single origin by geometry (no gridding metadata)";
    return r;
}

} // namespace check
