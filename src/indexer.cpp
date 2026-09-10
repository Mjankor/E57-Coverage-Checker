#include "indexer.h"

#include <fstream>
#include <sstream>
#include <limits>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <atomic>
#include <mutex>
#include <thread>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <unordered_map>

namespace indexer {

size_t Survey::usableCount() const {
    size_t n = 0;
    for (const auto& s : scans) if (s.usable) ++n;
    return n;
}

uint64_t Survey::totalPoints() const {
    uint64_t n = 0;
    for (const auto& s : scans) if (s.usable) n += s.recordCount;
    return n;
}

uint8_t chooseChunkLevel(uint64_t totalPoints, uint64_t targetPointsPerChunk) {
    if (targetPointsPerChunk == 0) targetPointsPerChunk = 1;
    // Chunks needed, then the level whose 8^level cell count covers them. Real
    // sites occupy far fewer than 8^level cells — points lie on surfaces, not
    // through the volume — so this errs toward larger chunks, which is the
    // safer direction: too many chunks means too many open spill buffers.
    const double chunks = double(totalPoints) / double(targetPointsPerChunk);
    uint8_t level = 1;
    while (level < 6 && std::pow(8.0, double(level)) < chunks) ++level;
    return level;
}

uint32_t cellIndexOf(const lod::Aabb& root, uint8_t level,
                     float x, float y, float z, lod::Aabb* outBounds) {
    lod::Aabb b = root;
    uint32_t path = 0;
    for (uint8_t l = 0; l < level; ++l) {
        float mid[3];
        b.centre(mid);
        int octant = 0;
        if (x >= mid[0]) octant |= 1;
        if (y >= mid[1]) octant |= 2;
        if (z >= mid[2]) octant |= 4;
        path = (path << 3) | uint32_t(octant);
        b = b.child(octant);
    }
    if (outBounds) *outBounds = b;
    return path;
}

namespace {

// --- survey helpers --------------------------------------------------------

void expandBounds(Survey& s, const double p[3]) {
    if (!s.hasBounds) {
        for (int i = 0; i < 3; ++i) { s.lo[i] = s.hi[i] = p[i]; }
        s.hasBounds = true;
        return;
    }
    for (int i = 0; i < 3; ++i) {
        s.lo[i] = std::min(s.lo[i], p[i]);
        s.hi[i] = std::max(s.hi[i], p[i]);
    }
}

// cartesianBounds is stated in the scan's own frame, so its corners have to go
// through the same transform the points will.
bool declaredExtentInFileFrame(const e57::Scan& sc, const viewer::FrameDecision& fd,
                               double lo[3], double hi[3]) {
    if (!sc.hasCartesianBounds) return false;
    const viewer::Rigid R = viewer::rigidFromPose(sc.pose);
    const bool applyPose = fd.applyPose() && sc.hasPose;

    bool first = true;
    for (int c = 0; c < 8; ++c) {
        double p[3] = {(c & 1) ? sc.xMax : sc.xMin,
                       (c & 2) ? sc.yMax : sc.yMin,
                       (c & 4) ? sc.zMax : sc.zMin};
        if (applyPose) R.apply(p[0], p[1], p[2]);
        for (int i = 0; i < 3; ++i) {
            if (first) { lo[i] = hi[i] = p[i]; }
            else { lo[i] = std::min(lo[i], p[i]); hi[i] = std::max(hi[i], p[i]); }
        }
        first = false;
    }
    return true;
}

// --- streaming point decode ------------------------------------------------

struct ColourPlan {
    bool     rgb = false;
    bool     intensity = false;
    double   intLo = 0, intHi = 1;
    uint8_t  fallback[3] = {200, 200, 200};
};

// Distinct hues per scan when a file carries no colour, so overlapping setups
// stay distinguishable.
void paletteFor(size_t i, uint8_t out[3]) {
    static const uint8_t pal[8][3] = {
        {140, 199, 255}, {255, 199, 115}, {158, 230, 158}, {240, 168, 219},
        {250, 235, 140}, {179, 179, 242}, {140, 235, 230}, {235, 153, 140},
    };
    for (int k = 0; k < 3; ++k) out[k] = pal[i % 8][k];
}

// Samples a scan's intensity so it can be mapped to grey. E57 leaves the scale
// to the producer, so assuming 0..1 grey-washes files that store raw counts.
void sampleIntensityRange(e57::Reader& r, size_t idx, ColourPlan& plan) {
    std::vector<std::string> want = {"intensity"};
    double lo = 1e300, hi = -1e300;
    uint64_t seen = 0;
    const uint64_t stride = std::max<uint64_t>(1, r.scan(idx).recordCount / 20000);
    std::string err;
    r.readPoints(idx, want, [&](const e57::PointBlock& b) {
        for (size_t k = 0; k < b.count; ++k, ++seen) {
            if (seen % stride) continue;
            lo = std::min(lo, b.columns[0][k]);
            hi = std::max(hi, b.columns[0][k]);
        }
        return true;
    }, err);
    if (hi > lo) { plan.intLo = lo; plan.intHi = hi; }
    else         { plan.intensity = false; }
}

using PointFn = std::function<void(const lod::StorePoint&)>;

// Decodes one scan straight to StorePoints in the file frame, without ever
// holding the scan in memory.
//
// `frameApplyPose` is the survey's frame decision, passed in rather than worked
// out here. It used to call viewer::decideFrame itself, which samples the points
// and therefore costs a whole decode pass — and build() calls this twice per scan,
// so that was two passes spent re-learning what the survey already knew. See
// ScanRef::frameApplyPose.
//
// `positionsOnly` skips the colour plan. A caller that just wants coordinates —
// build()'s extent pass — has no use for colour, and working it out is not free:
// an intensity-bearing scan costs another decode pass to find the intensity range,
// because E57 leaves that scale to the producer.
bool streamScan(e57::Reader& r, size_t idx, uint16_t scanId,
                const double origin[3], uint64_t maxPoints,
                size_t paletteIndex, bool frameApplyPose, bool positionsOnly,
                const PointFn& sink, uint64_t& readCount, std::string& err) {
    const e57::Scan& s = r.scan(idx);
    const bool cartesian = s.field("cartesianX") && s.field("cartesianY") && s.field("cartesianZ");
    const bool spherical = s.field("sphericalRange") && s.field("sphericalAzimuth") &&
                           s.field("sphericalElevation");
    if (!cartesian && !spherical) { err = "no position fields"; return false; }

    std::vector<std::string> want;
    if (cartesian) want = {"cartesianX", "cartesianY", "cartesianZ"};
    else           want = {"sphericalRange", "sphericalAzimuth", "sphericalElevation"};

    size_t invIdx = SIZE_MAX;
    const char* invName = nullptr;
    if      (s.field("cartesianInvalidState")) invName = "cartesianInvalidState";
    else if (s.field("sphericalInvalidState")) invName = "sphericalInvalidState";
    if (invName) { invIdx = want.size(); want.push_back(invName); }

    ColourPlan plan;
    size_t colIdx = SIZE_MAX, intIdx = SIZE_MAX;
    if (positionsOnly) {
        // Nothing will read the colour, so do not pay for it — least of all the
        // intensity-range pass, which decodes the scan again.
        paletteFor(paletteIndex, plan.fallback);
    } else if (s.field("colorRed") && s.field("colorGreen") && s.field("colorBlue")) {
        plan.rgb = true;
        colIdx = want.size();
        want.push_back("colorRed"); want.push_back("colorGreen"); want.push_back("colorBlue");
    } else if (s.field("intensity")) {
        plan.intensity = true;
        sampleIntensityRange(r, idx, plan);
        if (plan.intensity) { intIdx = want.size(); want.push_back("intensity"); }
    }
    if (!plan.rgb && !plan.intensity) paletteFor(paletteIndex, plan.fallback);

    const viewer::Rigid rigid = viewer::rigidFromPose(s.pose);
    // Spherical storage is scanner-centric by definition, so the pose always
    // applies and there was never a decision to make. For cartesian storage the
    // survey made it.
    const bool applyPose = spherical ? (s.hasPose && !viewer::isIdentityPose(s.pose))
                                     : (frameApplyPose && s.hasPose);

    const uint64_t stride = (maxPoints && s.recordCount > maxPoints)
                          ? (s.recordCount + maxPoints - 1) / maxPoints : 1;
    uint64_t seen = 0;
    readCount = 0;

    return r.readPoints(idx, want, [&](const e57::PointBlock& b) {
        for (size_t k = 0; k < b.count; ++k, ++seen) {
            if (stride > 1 && (seen % stride)) continue;
            if (invIdx != SIZE_MAX && b.columns[invIdx][k] != 0.0) continue;

            double x, y, z;
            if (cartesian) {
                x = b.columns[0][k]; y = b.columns[1][k]; z = b.columns[2][k];
            } else {
                const double rr = b.columns[0][k], az = b.columns[1][k], el = b.columns[2][k];
                const double ce = std::cos(el);
                x = rr * ce * std::cos(az);
                y = rr * ce * std::sin(az);
                z = rr * std::sin(el);
            }
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;
            if (applyPose) rigid.apply(x, y, z);

            lod::StorePoint p{};
            // Offsets from the store origin, so float32 never sees a UTM
            // magnitude (see point_cloud.h for what that costs).
            p.x = float(x - origin[0]);
            p.y = float(y - origin[1]);
            p.z = float(z - origin[2]);
            if (colIdx != SIZE_MAX) {
                p.r = uint8_t(std::clamp(b.columns[colIdx + 0][k], 0.0, 255.0));
                p.g = uint8_t(std::clamp(b.columns[colIdx + 1][k], 0.0, 255.0));
                p.b = uint8_t(std::clamp(b.columns[colIdx + 2][k], 0.0, 255.0));
            } else if (intIdx != SIZE_MAX) {
                const double t = std::clamp(
                    (b.columns[intIdx][k] - plan.intLo) / (plan.intHi - plan.intLo), 0.0, 1.0);
                const uint8_t g = uint8_t(40 + 215 * t);
                p.r = p.g = p.b = g;
            } else {
                p.r = plan.fallback[0]; p.g = plan.fallback[1]; p.b = plan.fallback[2];
            }
            p.a = 255;
            p.scanId = scanId;
            sink(p);
            ++readCount;
        }
        return true;
    }, err);
}

// --- spill files -----------------------------------------------------------

// One append-only file per occupied cell, fed through a small write buffer.
// Only occupied cells cost anything, and a site's points lie on surfaces, so
// far fewer than 8^level cells are ever touched.
class Spiller {
public:
    Spiller(std::string dir, size_t bufferPoints) : dir_(std::move(dir)), cap_(bufferPoints) {}
    ~Spiller() { flushAll(); }

    void add(uint32_t cell, const lod::StorePoint& p) {
        Bucket& b = buckets_[cell];
        b.buf.push_back(p);
        if (b.buf.size() >= cap_) flush(cell, b);
    }

    bool flushAll() {
        bool ok = true;
        for (auto& kv : buckets_) if (!flush(kv.first, kv.second)) ok = false;
        return ok;
    }

    std::vector<uint32_t> cells() const {
        std::vector<uint32_t> v;
        v.reserve(buckets_.size());
        for (const auto& kv : buckets_) if (kv.second.written) v.push_back(kv.first);
        return v;
    }

    std::string path(uint32_t cell) const {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "/chunk_%08X.bin", cell);
        return dir_ + buf;
    }

    void removeAll() {
        for (const auto& kv : buckets_) std::remove(path(kv.first).c_str());
    }

    size_t bucketCount() const { return buckets_.size(); }

private:
    struct Bucket {
        std::vector<lod::StorePoint> buf;
        uint64_t written = 0;
    };

    bool flush(uint32_t cell, Bucket& b) {
        if (b.buf.empty()) return true;
        std::FILE* f = std::fopen(path(cell).c_str(), b.written ? "ab" : "wb");
        if (!f) return false;
        const size_t bytes = b.buf.size() * sizeof(lod::StorePoint);
        const bool ok = std::fwrite(b.buf.data(), 1, bytes, f) == bytes;
        std::fclose(f);
        b.written += b.buf.size();
        b.buf.clear();
        return ok;
    }

    std::string dir_;
    size_t      cap_;
    std::map<uint32_t, Bucket> buckets_;
};

std::string directoryOf(const std::string& path) {
    const size_t slash = path.find_last_of('/');
    return slash == std::string::npos ? std::string(".") : path.substr(0, slash);
}

// ---------------------------------------------------------------------------
// the per-file check cache

// One line a field, which is what makes a status string safe to store: statuses
// are produced from measured numbers and read back into the UI, and they contain
// commas, percent signs and em dashes. A length-prefixed line carries any of
// those without quoting or escaping, and a line whose length does not match is a
// corrupt entry rather than a mis-parse that goes unnoticed.
//
// Text rather than a binary blob because this is a cache of verdicts about files
// on disk: when one looks wrong the first thing anybody wants is to read it.
constexpr const char* kCacheMagic = "e57cov-survey-cache";
// 2: ScanRef gained looksMerged.
// 3: and frameApplyPose.
// Bumped rather than tolerated each time, because an older entry read as a newer
// one comes back with the new field silently false — and a cache that quietly
// differs from a fresh check is the one thing it must not be. frameApplyPose
// especially: false where it should be true means a scan's points are not posed,
// which puts a whole setup in the wrong place.
constexpr int         kCacheVersion = 3;

void writeStr(std::string& o, const std::string& s) {
    o += std::to_string(s.size());
    o += ':';
    o += s;
    o += '\n';
}

bool readStr(std::istream& in, std::string& s) {
    std::string head;
    if (!std::getline(in, head)) return false;
    const size_t colon = head.find(':');
    if (colon == std::string::npos) return false;
    size_t n = 0;
    try { n = size_t(std::stoull(head.substr(0, colon))); }
    catch (...) { return false; }
    s = head.substr(colon + 1);
    // The value itself may have held newlines; keep reading until it is the
    // declared length. A declared length longer than what is there fails rather
    // than silently truncating.
    while (s.size() < n) {
        std::string more;
        if (!std::getline(in, more)) return false;
        s += '\n';
        s += more;
    }
    return s.size() == n;
}

} // namespace

bool SurveyCache::load(const std::string& path) {
    byPath.clear();
    std::ifstream in(path);
    if (!in) return true;   // no cache yet is not a failure

    std::string magic;
    int version = 0;
    if (!(in >> magic >> version)) return false;
    if (magic != kCacheMagic || version != kCacheVersion) return false;
    size_t files = 0;
    if (!(in >> files)) return false;
    in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    for (size_t f = 0; f < files; ++f) {
        std::string key;
        if (!readStr(in, key)) { byPath.clear(); return false; }
        Entry e;
        size_t nscans = 0;
        int ec = 1;
        if (!(in >> e.size >> e.mtime >> ec >> nscans)) { byPath.clear(); return false; }
        in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        e.extentComplete = (ec != 0);
        for (size_t s = 0; s < nscans; ++s) {
            ScanRef r;
            if (!readStr(in, r.path) || !readStr(in, r.name) || !readStr(in, r.guid) ||
                !readStr(in, r.status)) { byPath.clear(); return false; }
            int kind = 0, usable = 0, hasPose = 0, hasExtent = 0, looksMerged = 0,
                frameApplyPose = 0;
            if (!(in >> r.scanIndex >> r.recordCount >> kind >> usable >> hasPose >>
                  hasExtent >> looksMerged >> frameApplyPose)) { byPath.clear(); return false; }
            for (int k = 0; k < 4; ++k) if (!(in >> r.pose.q[k])) { byPath.clear(); return false; }
            for (int k = 0; k < 3; ++k) if (!(in >> r.pose.t[k])) { byPath.clear(); return false; }
            for (int k = 0; k < 3; ++k) if (!(in >> r.setup[k])) { byPath.clear(); return false; }
            for (int k = 0; k < 3; ++k) if (!(in >> r.lo[k])) { byPath.clear(); return false; }
            for (int k = 0; k < 3; ++k) if (!(in >> r.hi[k])) { byPath.clear(); return false; }
            in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            if (kind < 0 || kind > 2) { byPath.clear(); return false; }
            r.kind        = check::Kind(kind);
            r.usable      = (usable != 0);
            r.hasPose     = (hasPose != 0);
            r.hasExtent   = (hasExtent != 0);
            r.looksMerged = (looksMerged != 0);
            r.frameApplyPose = (frameApplyPose != 0);
            e.scans.push_back(std::move(r));
        }
        byPath[key] = std::move(e);
    }
    return true;
}

bool SurveyCache::save(const std::string& path) const {
    std::string o;
    o += kCacheMagic;
    o += ' ';
    o += std::to_string(kCacheVersion);
    o += ' ';
    o += std::to_string(byPath.size());
    o += '\n';
    for (const auto& kv : byPath) {
        writeStr(o, kv.first);
        const Entry& e = kv.second;
        o += std::to_string(e.size); o += ' ';
        o += std::to_string(e.mtime); o += ' ';
        o += (e.extentComplete ? "1 " : "0 ");
        o += std::to_string(e.scans.size());
        o += '\n';
        for (const ScanRef& r : e.scans) {
            writeStr(o, r.path);
            writeStr(o, r.name);
            writeStr(o, r.guid);
            writeStr(o, r.status);
            char buf[512];
            std::snprintf(buf, sizeof(buf),
                          "%zu %llu %d %d %d %d %d %d %.17g %.17g %.17g %.17g %.17g %.17g "
                          "%.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
                          r.scanIndex, (unsigned long long)r.recordCount, int(r.kind),
                          int(r.usable), int(r.hasPose), int(r.hasExtent), int(r.looksMerged),
                          int(r.frameApplyPose),
                          r.pose.q[0], r.pose.q[1], r.pose.q[2], r.pose.q[3],
                          r.pose.t[0], r.pose.t[1], r.pose.t[2],
                          r.setup[0], r.setup[1], r.setup[2],
                          r.lo[0], r.lo[1], r.lo[2], r.hi[0], r.hi[1], r.hi[2]);
            o += buf;
        }
    }
    // Written through a temporary and renamed, so an interrupted save leaves the
    // previous cache rather than a half-written one that would be read back as
    // corrupt — and a corrupt cache discards every entry, not just the last.
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) return false;
        out.write(o.data(), std::streamsize(o.size()));
        if (!out) return false;
    }
    return std::rename(tmp.c_str(), path.c_str()) == 0;
}

size_t SurveyCache::entries() const { return byPath.size(); }

namespace {

// How a file is identified: its path, size and modification time. The same
// ingredients the store's corpus key uses, per file rather than over all of
// them. Zeroed when it cannot be stat'd, which never matches a stored entry.
void fileStamp(const std::string& path, uint64_t& size, int64_t& mtime) {
    size = 0;
    mtime = 0;
    struct stat st{};
    if (::stat(path.c_str(), &st) == 0) {
        size  = uint64_t(st.st_size);
        mtime = int64_t(st.st_mtime);
    }
}

// ---------------------------------------------------------------------------
// survey

// What checking ONE file produces. Kept per file so the work can run on several
// at once and still be merged in path order: the scan list, the bounds and the
// error list then read the same whatever the threads did, which matters because
// this list drives which scans get indexed.
struct FileResult {
    bool                 opened = false;
    std::string          error;           // when it could not be opened
    std::vector<ScanRef> scans;
    bool                 hasBounds = false;
    double               lo[3] = {0, 0, 0};
    double               hi[3] = {0, 0, 0};
    bool                 extentComplete = true;
};

// The same expansion as expandBounds, against a file's own accumulator.
void expandFileBounds(FileResult& f, const double p[3]) {
    if (!f.hasBounds) {
        for (int i = 0; i < 3; ++i) { f.lo[i] = f.hi[i] = p[i]; }
        f.hasBounds = true;
        return;
    }
    for (int i = 0; i < 3; ++i) {
        f.lo[i] = std::min(f.lo[i], p[i]);
        f.hi[i] = std::max(f.hi[i], p[i]);
    }
}

void checkOneFile(const std::string& path, const SurveyOptions& opt, FileResult& out) {
    e57::Reader r;
    std::string err;
    if (!r.open(path, err)) {
        out.error = path + ": " + err;
        return;
    }
    out.opened = true;

    for (size_t i = 0; i < r.scanCount(); ++i) {
        const e57::Scan& sc = r.scan(i);
        ScanRef ref;
        ref.path        = path;
        ref.scanIndex   = i;
        ref.name        = sc.name.empty() ? ("scan " + std::to_string(i)) : sc.name;
        ref.guid        = sc.guid;
        ref.pose        = sc.pose;
        ref.hasPose     = sc.hasPose;
        ref.recordCount = sc.recordCount;

        // ONE decode pass for both decisions that need points.
        //
        // The frame decision and the merged-cloud test ask different questions
        // of the same sample, and they used to take three passes between them —
        // one here, one inside check::classify, and one more because classify
        // called decideFrame itself. Measured on a 5.65 M point scan, that was
        // 0.205 s a scan against 0.09 s for one pass, which over a thousand
        // scans is three minutes against one.
        //
        // The sample is the finer of the two that were being taken
        // (check::Thresholds::sampleTarget, 200k, against decideFrame's 20k).
        // More points cannot hurt the frame decision: it is a median and a
        // least-squares fit, so a larger sample of the same distribution gives
        // the same answer more precisely.
        check::Result res;
        viewer::FrameDecision fd;
        if (opt.classify) {
            std::vector<double> xyz;
            std::string serr;
            const check::Thresholds th;
            if (r.sampleXYZ(i, th.sampleTarget, xyz, serr)) {
                fd  = viewer::decideFrameFromSample(sc, xyz);
                res = check::classifyFromSample(sc, xyz, fd, th);
            } else {
                // No cartesian points to sample — spherical storage, or a
                // decode that failed. Both are answered from metadata, and
                // spherical storage is scanner-centric by definition.
                res = check::classifyMetadata(sc);
                fd  = viewer::decideFrameFromSample(sc, {});
            }
        } else {
            // A header-only survey must not decode points, so the frame
            // decision here is the cheap one: trust the standard.
            res = check::classifyMetadata(sc);
            fd.convention = (sc.hasPose && !viewer::isIdentityPose(sc.pose))
                          ? viewer::FrameConvention::ScannerLocal
                          : viewer::FrameConvention::IdentityPose;
        }
        ref.kind        = res.kind;
        ref.status      = res.summary;
        ref.looksMerged = res.looksMerged;
        // Kept so build() does not decode the scan again to re-learn it.
        ref.frameApplyPose = fd.applyPose();
        // Can it be drawn? That is the whole question — see ScanRef::usable. A
        // scan with no position fields has nothing to put in the store; every
        // other scan goes in, whatever the merged-cloud heuristic thinks of it.
        ref.usable = (sc.field("cartesianX") && sc.field("cartesianY") &&
                      sc.field("cartesianZ")) ||
                     (sc.field("sphericalRange") && sc.field("sphericalAzimuth") &&
                      sc.field("sphericalElevation"));

        for (int k = 0; k < 3; ++k) ref.setup[k] = sc.pose.t[k];
        expandFileBounds(out, ref.setup);

        if (declaredExtentInFileFrame(sc, fd, ref.lo, ref.hi)) {
            ref.hasExtent = true;
            expandFileBounds(out, ref.lo);
            expandFileBounds(out, ref.hi);
        } else if (ref.usable) {
            out.extentComplete = false;
        }
        out.scans.push_back(std::move(ref));
    }
}

} // namespace

Survey survey(const std::vector<std::string>& paths, const SurveyOptions& opt,
              const ProgressFn& progress) {
    Survey out;
    if (paths.empty()) {
        if (progress) progress("reading headers", 0, 0);
        return out;
    }

    // Files at once. Checking one shares nothing with checking another, so this
    // is the stage that scales with the corpus: on eight copies of a 154 MB scan
    // it went 1.49 s to 0.43 s across four cores. Capped at the core count
    // because eight threads on four cores measured slower than four — the work
    // is decode-bound, and oversubscribing it only adds contention.
    unsigned nt = opt.threads ? opt.threads : std::thread::hardware_concurrency();
    if (nt == 0) nt = 1;
    const unsigned cores = std::max(1u, std::thread::hardware_concurrency());
    nt = std::min({nt, cores, unsigned(paths.size())});

    // The cache of per-file verdicts, when one was asked for. Only for the
    // classifying survey: the header-only path costs nothing and reaches a
    // cheaper verdict that must not be mistaken for this one.
    SurveyCache cache;
    const bool useCache = opt.classify && !opt.cachePath.empty();
    if (useCache) cache.load(opt.cachePath);   // unreadable is empty, not fatal

    // What each file's stamp is now, read before any work so a file that changes
    // mid-survey is stored under what was actually checked.
    std::vector<uint64_t> sizes(paths.size(), 0);
    std::vector<int64_t>  mtimes(paths.size(), 0);
    if (useCache)
        for (size_t fi = 0; fi < paths.size(); ++fi) fileStamp(paths[fi], sizes[fi], mtimes[fi]);

    // A slot per file, so nothing is shared but the cursor and the progress
    // report. Merged in path order below, which is what keeps the result
    // independent of how the threads happened to interleave.
    std::vector<FileResult> results(paths.size());
    std::vector<char>       fromCache(paths.size(), 0);
    std::atomic<size_t> next{0};
    std::atomic<size_t> done{0};
    std::atomic<bool>   stop{false};
    std::mutex          reportLock;

    auto worker = [&]() {
        for (;;) {
            if (stop.load(std::memory_order_relaxed)) break;
            const size_t fi = next.fetch_add(1, std::memory_order_relaxed);
            if (fi >= paths.size()) break;

            // A hit means this exact file — same path, same size, same
            // modification time — was checked before, and checking depends on
            // nothing else. Reading the cache needs no lock because nothing
            // writes to it while the workers run; the entries they produce go to
            // their own slots and are folded in afterwards.
            bool hit = false;
            if (useCache && sizes[fi] != 0) {
                const auto it = cache.byPath.find(paths[fi]);
                if (it != cache.byPath.end() && it->second.size == sizes[fi] &&
                    it->second.mtime == mtimes[fi]) {
                    FileResult& out = results[fi];
                    out.opened = true;
                    out.scans = it->second.scans;
                    out.extentComplete = it->second.extentComplete;
                    // The bounds are re-derived rather than stored: they are a
                    // reduction over the very scans just restored, so storing
                    // them would be a second copy of the same fact that could
                    // disagree with the first.
                    for (const ScanRef& r : out.scans) {
                        expandFileBounds(out, r.setup);
                        if (r.hasExtent) {
                            expandFileBounds(out, r.lo);
                            expandFileBounds(out, r.hi);
                        }
                    }
                    fromCache[fi] = 1;
                    hit = true;
                }
            }
            if (!hit) checkOneFile(paths[fi], opt, results[fi]);

            const size_t d = done.fetch_add(1, std::memory_order_relaxed) + 1;
            if (progress) {
                std::lock_guard<std::mutex> lk(reportLock);
                if (!progress("reading headers", d, paths.size()))
                    stop.store(true, std::memory_order_relaxed);
            }
        }
    };

    if (nt == 1) {
        worker();
    } else {
        std::vector<std::thread> pool;
        pool.reserve(nt);
        for (unsigned k = 0; k < nt; ++k) pool.emplace_back(worker);
        for (std::thread& t : pool) t.join();
    }

    // Saved BEFORE the merge, because the merge moves each ScanRef's strings out
    // into the survey and would leave the entries here holding empty names and
    // statuses. Copying them first instead would work and costs a copy of every
    // scan on every run; doing it in this order costs nothing.
    //
    // Only files in this corpus are kept, so the cache does not grow without
    // bound as directories come and go. A cancelled run saves what it reached —
    // entries are per file and each is complete or absent.
    if (useCache) {
        SurveyCache fresh;
        for (size_t fi = 0; fi < paths.size(); ++fi) {
            if (sizes[fi] == 0) continue;            // unstattable: nothing to key on
            if (fromCache[fi]) {
                const auto it = cache.byPath.find(paths[fi]);
                if (it != cache.byPath.end()) fresh.byPath[paths[fi]] = it->second;
                continue;
            }
            const FileResult& f = results[fi];
            if (!f.opened) continue;                 // unreadable or never reached
            SurveyCache::Entry e;
            e.size  = sizes[fi];
            e.mtime = mtimes[fi];
            e.scans = f.scans;
            e.extentComplete = f.extentComplete;
            fresh.byPath[paths[fi]] = std::move(e);
        }
        fresh.save(opt.cachePath);                   // a failed save costs a re-check
    }

    // Merged in path order. A cancelled run leaves the files it never reached
    // unopened, and an unopened file with no error contributes nothing — the
    // same as the serial version breaking out of its loop.
    for (size_t fi = 0; fi < paths.size(); ++fi) {
        FileResult& f = results[fi];
        if (!f.error.empty()) { out.errors.push_back(f.error); continue; }
        if (!f.opened) continue;
        ++out.filesRead;
        if (fromCache[fi]) ++out.filesFromCache;
        if (f.hasBounds) {
            expandBounds(out, f.lo);
            expandBounds(out, f.hi);
        }
        if (!f.extentComplete) out.extentComplete = false;
        for (ScanRef& ref : f.scans) out.scans.push_back(std::move(ref));
    }

    if (progress) progress("reading headers", paths.size(), paths.size());
    return out;
}

// ---------------------------------------------------------------------------
// build

bool build(const Survey& s, const std::string& storePath, const BuildOptions& opt,
           BuildStats& stats, const ProgressFn& progress, std::string& err) {
    stats = BuildStats{};

    std::vector<const ScanRef*> usable;
    for (const auto& sc : s.scans) if (sc.usable) usable.push_back(&sc);
    if (usable.empty()) { err = "no usable scans to index"; return false; }
    if (usable.size() > 65535) {
        // scanId is uint16. Wrapping would silently mislabel points, and "which
        // setups cover this space" is the question the tool answers.
        err = "corpus has " + std::to_string(usable.size()) +
              " usable scans; the store format supports 65535";
        return false;
    }
    if (!s.hasBounds) { err = "survey produced no bounds"; return false; }

    // Store origin at the centre of the site, so float offsets stay small.
    double origin[3];
    for (int i = 0; i < 3; ++i) origin[i] = 0.5 * (s.lo[i] + s.hi[i]);

    // Declared extents are often absent — the survey says so rather than
    // guessing — and setup positions alone bound nothing: every point of every
    // scan lies outside them. So read the actual points before choosing a root.
    //
    // EVERY point, no stride. That is a correction, and the stride it replaces
    // was free to remove because it never saved any work: readPoints decodes a
    // whole bytestream regardless, so a stride only skipped the min/max
    // comparisons, not the decode. It bought six floating-point comparisons a
    // point and cost correctness.
    //
    // What it cost: a stride aliases against any periodic structure in point
    // order. On a fixture of three concentric shells written interleaved per
    // direction, the stride came out a multiple of three and sampled the
    // INNERMOST SHELL ONLY — the root was sized to a third of the cloud and 31
    // per cent of the points fell outside it. A point outside the root is simply
    // absent from the store, which then reads as missing coverage: the failure is
    // silent and points the wrong way. Real scans are stored row-major, so a
    // stride sharing a factor with the column count samples a few columns of a
    // raster and misses whatever is only visible elsewhere in it.
    //
    // An exact extent also means the root is the right size, so nothing is
    // dropped and the generous padding below is belt and braces rather than the
    // thing standing between the corpus and a quiet 31 per cent loss.
    double lo[3] = {s.lo[0], s.lo[1], s.lo[2]};
    double hi[3] = {s.hi[0], s.hi[1], s.hi[2]};
    if (!s.extentComplete) {
        for (size_t i = 0; i < usable.size(); ++i) {
            if (progress && !progress("measuring extent", i, usable.size())) {
                err = "cancelled"; return false;
            }
            e57::Reader r;
            std::string ferr;
            if (!r.open(usable[i]->path, ferr)) continue;
            const double zero[3] = {0, 0, 0};
            uint64_t got = 0;
            streamScan(r, usable[i]->scanIndex, 0, zero, 0, i,
                       usable[i]->frameApplyPose, /*positionsOnly=*/true,
                       [&](const lod::StorePoint& p) {
                           lo[0] = std::min(lo[0], double(p.x)); hi[0] = std::max(hi[0], double(p.x));
                           lo[1] = std::min(lo[1], double(p.y)); hi[1] = std::max(hi[1], double(p.y));
                           lo[2] = std::min(lo[2], double(p.z)); hi[2] = std::max(hi[2], double(p.z));
                       }, got, ferr);
        }
        for (int i = 0; i < 3; ++i) origin[i] = 0.5 * (lo[i] + hi[i]);
    }

    lod::Aabb rough;
    for (int i = 0; i < 3; ++i) {
        rough.lo[i] = float(lo[i] - origin[i]);
        rough.hi[i] = float(hi[i] - origin[i]);
    }
    // Declared bounds are not always present, and a scan can exceed them.
    // Padding is far cheaper than a point falling outside the root, which the
    // builder would reject outright.
    // The extent came from a sample, and a scan can exceed its declared
    // bounds, so pad generously: a point outside the root is simply absent
    // from the store, which reads as missing coverage.
    const float pad = std::max(2.0f, 0.15f * rough.diagonal());
    for (int i = 0; i < 3; ++i) { rough.lo[i] -= pad; rough.hi[i] += pad; }
    const lod::Aabb root = lod::cubeAround(rough);

    const uint64_t totalPoints = s.totalPoints();
    stats.chunkLevel = opt.chunkLevel ? opt.chunkLevel
                                      : chooseChunkLevel(totalPoints, opt.targetPointsPerChunk);
    if (stats.chunkLevel < 1) stats.chunkLevel = 1;

    const std::string chunkDir = opt.chunkDir.empty() ? directoryOf(storePath) : opt.chunkDir;

    // What this is about to need, checked before an hour of work discovers it.
    //
    // A store is one StorePoint per point, 20 bytes, and every point is spilled to
    // a chunk file on the way in. Chunk files are deleted as they are consumed, so
    // the two do not both peak in full — but they overlap, and the honest estimate
    // is the store plus the largest slice of spill still unconsumed. Assume half.
    // A thousand scans of five million points is 100 GB of store, which is not a
    // number anyone guesses.
    {
        const uint64_t storeBytes = s.totalPoints() * sizeof(lod::StorePoint);
        const uint64_t needBytes  = storeBytes + storeBytes / 2;
        struct statvfs vfs {};
        if (::statvfs(chunkDir.c_str(), &vfs) == 0) {
            const uint64_t freeBytes = uint64_t(vfs.f_bavail) * uint64_t(vfs.f_frsize);
            if (freeBytes < needBytes) {
                char buf[320];
                std::snprintf(buf, sizeof(buf),
                    "not enough disk for the index: about %.1f GB needed in %s, %.1f GB free. "
                    "The store is %zu bytes per point and every point is also spilled to a "
                    "chunk file on the way in.",
                    double(needBytes) / 1073741824.0, chunkDir.c_str(),
                    double(freeBytes) / 1073741824.0, sizeof(lod::StorePoint));
                err = buf;
                return false;
            }
        }
    }

    Spiller spiller(chunkDir, 32768);

    store::Writer writer;
    if (!writer.open(storePath, err)) return false;
    writer.setOrigin(origin[0], origin[1], origin[2]);
    writer.setRoot(root, opt.tree.gridResolution, opt.tree.maxLevel);

    for (size_t i = 0; i < usable.size(); ++i) {
        store::ScanRecord rec{};
        std::snprintf(rec.name, sizeof(rec.name), "%s", usable[i]->name.c_str());
        std::snprintf(rec.guid, sizeof(rec.guid), "%s", usable[i]->guid.c_str());
        for (int k = 0; k < 3; ++k) rec.poseT[k] = usable[i]->pose.t[k];
        for (int k = 0; k < 4; ++k) rec.poseQ[k] = usable[i]->pose.q[k];
        rec.sourcePoints = usable[i]->recordCount;
        rec.flags = (usable[i]->kind == check::Kind::Structured) ? store::kScanStructured
                                                                  : store::kScanAmbiguous;
        if (usable[i]->looksMerged) rec.flags |= store::kScanLooksMerged;
        writer.addScan(rec);
    }

    // --- pass 1: top tree in memory, everything else spilled ---------------
    lod::BuildOptions topOpt = opt.tree;
    topOpt.maxLevel        = uint8_t(stats.chunkLevel - 1);
    topOpt.spillAtMaxLevel = true;

    lod::Builder top(root, topOpt);
    uint64_t spilled = 0;
    top.setOverflowSink([&](const lod::StorePoint& p) {
        spiller.add(cellIndexOf(root, stats.chunkLevel, p.x, p.y, p.z), p);
        ++spilled;
    });

    uint64_t outsideRoot = 0;
    for (size_t i = 0; i < usable.size(); ++i) {
        if (progress && !progress("indexing points", i, usable.size())) {
            err = "cancelled"; return false;
        }
        e57::Reader r;
        std::string ferr;
        if (!r.open(usable[i]->path, ferr)) continue;

        uint64_t got = 0;
        streamScan(r, usable[i]->scanIndex, uint16_t(i), origin, opt.maxPointsPerScan, i,
                   usable[i]->frameApplyPose, /*positionsOnly=*/false,
                   [&](const lod::StorePoint& p) {
                       if (!top.insert(p)) ++outsideRoot;
                   }, got, ferr);
        stats.pointsRead += got;
    }
    if (!spiller.flushAll()) { err = "failed writing chunk files to " + chunkDir; return false; }
    stats.spilled = spilled;

    std::vector<std::vector<lod::StorePoint>> topPayload;
    const lod::Tree topTree = top.finish(topPayload);
    uint32_t topRootIndex = 0;
    if (!writer.appendTree(topTree, topPayload, topRootIndex, err)) return false;
    if (topRootIndex != 0) { err = "top tree did not land at store index 0"; return false; }

    // --- pass 2: one chunk at a time --------------------------------------
    const std::vector<uint32_t> cells = spiller.cells();
    stats.chunks = cells.size();

    // Built in parallel, appended serially.
    //
    // Reading a spill file and building its subtree shares nothing with any other
    // chunk. Appending the result does: node indices come out of the writer in
    // append order, so the appends run in cell order on this thread and the store
    // is identical to what the serial build produced.
    //
    // In BATCHES of `ct` rather than all at once, because a finished subtree holds
    // its whole payload in memory and the point of this build is that memory does
    // not grow with the corpus. Peak is a batch, not a build. See
    // BuildOptions::maxResidentPoints.
    unsigned ct = opt.chunkThreads ? opt.chunkThreads : std::thread::hardware_concurrency();
    if (ct == 0) ct = 1;
    {
        const uint64_t perChunk = std::max<uint64_t>(1, opt.targetPointsPerChunk);
        const uint64_t byMemory = std::max<uint64_t>(1, opt.maxResidentPoints / perChunk);
        const unsigned cores = std::max(1u, std::thread::hardware_concurrency());
        ct = unsigned(std::min<uint64_t>({ct, byMemory, cores, std::max<size_t>(1, cells.size())}));
    }

    // One chunk's finished work, waiting to be appended.
    struct Built {
        lod::Tree                                  sub;
        std::vector<std::vector<lod::StorePoint>>  payload;
        uint32_t                                   parent = 0;
        int                                        lastOctant = 0;
        uint64_t                                   dropped = 0;
        bool                                       ok = false;
        std::string                                err;
    };

    // The cell path gives the parent node in the top tree and the octant the chunk
    // hangs from. Read-only against topTree, so it is safe to do on any thread.
    auto locate = [&](uint32_t cell, lod::Aabb& cellBounds, uint32_t& parent,
                      int& lastOctant) {
        cellBounds = root;
        parent = 0;
        lastOctant = 0;
        for (uint8_t l = 0; l < stats.chunkLevel; ++l) {
            const int octant = int((cell >> (3 * (stats.chunkLevel - 1 - l))) & 7);
            cellBounds = cellBounds.child(octant);
            if (l + 1 == stats.chunkLevel) { lastOctant = octant; return true; }
            if (!topTree.nodes[parent].hasChild(octant)) return false;
            parent = topTree.nodes[parent].child[octant];
        }
        return true;
    };

    auto buildOne = [&](size_t ci, Built& out) {
        const uint32_t cell = cells[ci];
        lod::Aabb cellBounds;
        if (!locate(cell, cellBounds, out.parent, out.lastOctant)) {
            out.err = "chunk cell has no parent in the top tree — internal inconsistency";
            return;
        }
        std::FILE* f = std::fopen(spiller.path(cell).c_str(), "rb");
        if (!f) { out.err = "cannot reopen chunk " + spiller.path(cell); return; }

        lod::BuildOptions chunkOpt = opt.tree;
        lod::Builder cb(cellBounds, chunkOpt, stats.chunkLevel);
        cb.setOverflowSink([&out](const lod::StorePoint&) { ++out.dropped; });

        std::vector<lod::StorePoint> buf(65536);
        for (;;) {
            const size_t n = std::fread(buf.data(), sizeof(lod::StorePoint), buf.size(), f);
            if (n == 0) break;
            for (size_t k = 0; k < n; ++k) cb.insert(buf[k]);
        }
        std::fclose(f);
        out.sub = cb.finish(out.payload);
        out.ok = true;
    };

    std::vector<Built> batch(ct);
    for (size_t base = 0; base < cells.size(); base += ct) {
        const size_t n = std::min<size_t>(ct, cells.size() - base);
        for (size_t k = 0; k < n; ++k) batch[k] = Built{};

        if (n == 1 || ct == 1) {
            buildOne(base, batch[0]);
        } else {
            std::vector<std::thread> pool;
            pool.reserve(n);
            for (size_t k = 0; k < n; ++k)
                pool.emplace_back([&, k]() { buildOne(base + k, batch[k]); });
            for (std::thread& t : pool) t.join();
        }

        for (size_t k = 0; k < n; ++k) {
            Built& b = batch[k];
            if (!b.ok) { err = b.err.empty() ? "chunk build failed" : b.err; return false; }
            uint32_t subRoot = 0;
            if (!writer.appendTree(b.sub, b.payload, subRoot, err)) return false;
            if (!writer.linkChild(b.parent, b.lastOctant, subRoot, err)) return false;
            stats.dropped += b.dropped;
            // Freed as it is consumed, so the batch does not stay resident while
            // the rest of it is appended.
            b.payload.clear();
            b.payload.shrink_to_fit();
            // And its spill file goes now rather than at the end of the build.
            //
            // The spill files hold every point once and the store holds them
            // again, so keeping all of both alive means peak disk is TWICE the
            // corpus — on a thousand scans that is a couple of hundred gigabytes
            // of scratch, and running out of it surfaces as a short write halfway
            // through writing the store. A consumed chunk is dead weight: its
            // points are in the store and nothing reads it again.
            std::remove(spiller.path(cells[base + k]).c_str());
            if (progress && !progress("building chunks", base + k + 1, cells.size())) {
                err = "cancelled"; return false;
            }
        }
    }

    if (!writer.finish(err)) return false;
    spiller.removeAll();

    stats.nodes        = writer.nodeCount();
    stats.pointsStored = writer.pointCount();
    stats.outsideRoot  = outsideRoot;
    return true;
}

} // namespace indexer
