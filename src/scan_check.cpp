#include "scan_check.h"

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
    Result r = classifyMetadata(s);

    // The geometric test needs cartesian points. Spherical scans are already
    // proven single-origin by their own storage format, so there is nothing
    // left to test.
    const bool cartesian = hasField(s, "cartesianX") && hasField(s, "cartesianY") &&
                           hasField(s, "cartesianZ");
    if (!cartesian || s.recordCount == 0) return r;

    // Points are relative to the scanner origin. If the producer already
    // applied the pose, the origin is the pose translation; otherwise it is
    // zero. Deciding wrongly would scatter directions meaninglessly, so use
    // the centroid as a tiebreak the same way `e57cov info` does.
    std::vector<std::string> want = {"cartesianX", "cartesianY", "cartesianZ"};
    const char* invName = hasField(s, "cartesianInvalidState") ? "cartesianInvalidState"
                                                              : nullptr;
    if (invName) want.push_back(invName);
    const size_t invIdx = 3;

    const uint64_t stride = std::max<uint64_t>(1, s.recordCount / std::max<size_t>(1, t.sampleTarget));

    // Pass 1: centroid of the sample, to choose the origin.
    double cx = 0, cy = 0, cz = 0;
    uint64_t n = 0, seen = 0;
    std::string err;
    auto sample = [&](const e57::PointBlock& b, auto&& fn) {
        for (size_t k = 0; k < b.count; ++k, ++seen) {
            if (seen % stride) continue;
            if (invName && b.columns[invIdx][k] != 0.0) continue;
            fn(b.columns[0][k], b.columns[1][k], b.columns[2][k]);
        }
        return true;
    };

    if (!reader.readPoints(scanIndex, want, [&](const e57::PointBlock& b) {
            return sample(b, [&](double x, double y, double z) {
                cx += x; cy += y; cz += z; ++n;
            });
        }, err) || n < 100) {
        r.evidence.push_back("geometric test skipped: too few decodable points");
        return r;
    }
    cx /= double(n); cy /= double(n); cz /= double(n);

    double ox = 0, oy = 0, oz = 0;
    const double poseLen = std::sqrt(s.pose.t[0] * s.pose.t[0] +
                                     s.pose.t[1] * s.pose.t[1] +
                                     s.pose.t[2] * s.pose.t[2]);
    const double centLen = std::sqrt(cx * cx + cy * cy + cz * cz);
    if (poseLen > 1.0 && centLen > 0.5 * poseLen) {
        ox = s.pose.t[0]; oy = s.pose.t[1]; oz = s.pose.t[2];
        r.evidence.push_back("testing about the pose translation (points appear pre-transformed)");
    } else {
        r.evidence.push_back("testing about the local origin");
    }

    // Pass 2: bin by direction, track min and max range per bin.
    struct Bin { float lo = 1e30f, hi = -1e30f; int n = 0; };
    std::unordered_map<uint32_t, Bin> bins;
    bins.reserve(t.sampleTarget / 8);

    const double binRad  = t.binDegrees * 3.14159265358979 / 180.0;
    const int    nAz     = std::max(1, int(std::round(2.0 * 3.14159265358979 / binRad)));
    const int    nEl     = std::max(1, int(std::round(3.14159265358979 / binRad)));
    seen = 0;
    uint64_t used = 0;

    if (!reader.readPoints(scanIndex, want, [&](const e57::PointBlock& b) {
            return sample(b, [&](double x, double y, double z) {
                const double dx = x - ox, dy = y - oy, dz = z - oz;
                const double rr = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (rr < 1e-6) return;
                const double az = std::atan2(dy, dx) + 3.14159265358979;   // 0..2pi
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
            });
        }, err)) {
        r.evidence.push_back("geometric test skipped: decode failed on second pass");
        return r;
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
    r.evidence.push_back(fmt("%.1f%% of %.0f populated direction bins hold multiple surfaces",
                             100.0 * r.multiSurfaceFraction, double(tested)));

    if (r.multiSurfaceFraction > t.multiSurfaceFraction) {
        // Overrides metadata: a file can declare a grid and still contain a
        // merged cloud, and the geometry is what the pipeline actually relies on.
        r.kind = Kind::Unified;
        r.summary = fmt("looks merged — %.0f%% of directions hit multiple surfaces",
                        100.0 * r.multiSurfaceFraction);
        return r;
    }

    if (r.kind == Kind::Ambiguous) {
        r.kind = Kind::Structured;
        r.summary = "single origin by geometry (no gridding metadata)";
    }
    return r;
}

} // namespace check
