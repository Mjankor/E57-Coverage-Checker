#include "point_store.h"

#include <cerrno>
#include <cstring>

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <cstring>

namespace store {

// ---------------------------------------------------------------------------
// Writer

Writer::~Writer() {
    if (fp_) std::fclose(fp_);
}

bool Writer::open(const std::string& path, std::string& err) {
    fp_ = std::fopen(path.c_str(), "wb");
    if (!fp_) { err = "cannot create " + path; return false; }

    std::memset(&header_, 0, sizeof(header_));
    std::memcpy(header_.magic, "E57COVLD", 8);
    header_.version         = kVersion;
    header_.pointRecordSize = uint32_t(sizeof(lod::StorePoint));

    // Reserve the header; it is patched in finish() once the tables are placed.
    if (std::fwrite(&header_, 1, sizeof(header_), fp_) != sizeof(header_)) {
        err = "short write on header"; return false;
    }
    cursor_ = sizeof(Header);
    return true;
}

void Writer::setOrigin(double x, double y, double z) {
    header_.origin[0] = x; header_.origin[1] = y; header_.origin[2] = z;
}

void Writer::setRoot(const lod::Aabb& r, uint32_t gridResolution, uint32_t maxLevel) {
    for (int i = 0; i < 3; ++i) { header_.rootLo[i] = r.lo[i]; header_.rootHi[i] = r.hi[i]; }
    header_.gridResolution = gridResolution;
    header_.maxLevel       = maxLevel;
}

uint32_t Writer::addScan(const ScanRecord& s) {
    scans_.push_back(s);
    return uint32_t(scans_.size() - 1);
}

bool Writer::appendTree(const lod::Tree& tree,
                        const std::vector<std::vector<lod::StorePoint>>& payloads,
                        uint32_t& outRootIndex, std::string& err) {
    if (!fp_) { err = "store is not open"; return false; }
    if (tree.nodes.size() != payloads.size()) {
        err = "node count does not match payload count"; return false;
    }
    if (tree.nodes.empty()) { err = "cannot append an empty tree"; return false; }

    const uint32_t base = uint32_t(nodes_.size());
    // Index 0 is the root of the whole store and doubles as the "no child"
    // sentinel, so nothing else may land there.
    if (base == 0 && tree.nodes.empty()) { err = "first appended tree must be non-empty"; return false; }

    for (size_t i = 0; i < tree.nodes.size(); ++i) {
        const lod::Node& n = tree.nodes[i];
        NodeRecord rec{};
        rec.payloadOffset = cursor_;
        rec.pointCount    = uint32_t(payloads[i].size());
        rec.level         = n.level;
        for (int k = 0; k < 3; ++k) { rec.lo[k] = n.bounds.lo[k]; rec.hi[k] = n.bounds.hi[k]; }
        for (int o = 0; o < 8; ++o)
            rec.child[o] = n.child[o] ? (base + n.child[o]) : 0u;

        if (!payloads[i].empty()) {
            const size_t bytes = payloads[i].size() * sizeof(lod::StorePoint);
            if (std::fwrite(payloads[i].data(), 1, bytes, fp_) != bytes) {
                // Say WHY. A short write here is almost always the disk filling
                // up, and "short write on node payload" sends the reader looking
                // for a bug in the writer. A store is one StorePoint per point at
                // 20 bytes, so a large corpus is tens of gigabytes and the number
                // written so far is the useful part of the message.
                const int e = errno;
                err = "short write on node payload after " +
                      std::to_string(cursor_ / (1024 * 1024)) + " MB: " +
                      (e ? std::strerror(e) : "no space, or the file is on a full or "
                                              "read-only volume");
                return false;
            }
            cursor_ += bytes;
        }
        totalPoints_ += rec.pointCount;
        nodes_.push_back(rec);
    }
    outRootIndex = base;
    return true;
}

bool Writer::linkChild(uint32_t parentNode, int octant, uint32_t childNode, std::string& err) {
    if (parentNode >= nodes_.size() || childNode >= nodes_.size()) {
        err = "linkChild: node index out of range"; return false;
    }
    if (octant < 0 || octant > 7) { err = "linkChild: octant out of range"; return false; }
    if (childNode == 0) { err = "linkChild: node 0 is the root and cannot be a child"; return false; }
    nodes_[parentNode].child[octant] = childNode;
    return true;
}

bool Writer::finish(std::string& err) {
    if (!fp_) { err = "store is not open"; return false; }
    if (finished_) return true;

    header_.scanTableOffset = cursor_;
    if (!scans_.empty()) {
        const size_t bytes = scans_.size() * sizeof(ScanRecord);
        if (std::fwrite(scans_.data(), 1, bytes, fp_) != bytes) {
            err = "short write on scan table"; return false;
        }
        cursor_ += bytes;
    }

    header_.nodeTableOffset = cursor_;
    if (!nodes_.empty()) {
        const size_t bytes = nodes_.size() * sizeof(NodeRecord);
        if (std::fwrite(nodes_.data(), 1, bytes, fp_) != bytes) {
            err = "short write on node table"; return false;
        }
        cursor_ += bytes;
    }

    header_.nodeCount   = nodes_.size();
    header_.scanCount   = scans_.size();
    header_.totalPoints = totalPoints_;

    if (std::fseek(fp_, 0, SEEK_SET) != 0) { err = "seek to header failed"; return false; }
    if (std::fwrite(&header_, 1, sizeof(header_), fp_) != sizeof(header_)) {
        err = "short write patching the header"; return false;
    }
    if (std::fclose(fp_) != 0) { err = "close failed"; fp_ = nullptr; return false; }
    fp_ = nullptr;
    finished_ = true;
    return true;
}

// ---------------------------------------------------------------------------
// Reader

Reader::~Reader() { close(); }

void Reader::close() {
    if (base_) { ::munmap(const_cast<uint8_t*>(base_), size_); base_ = nullptr; }
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
    size_ = 0; scans_ = nullptr; nodes_ = nullptr;
}

bool Reader::open(const std::string& path, std::string& err) {
    close();
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) { err = "cannot open " + path; return false; }

    struct stat st{};
    if (::fstat(fd_, &st) != 0) { err = "fstat failed on " + path; close(); return false; }
    size_ = uint64_t(st.st_size);
    if (size_ < sizeof(Header)) { err = "file is too small to be a point store"; close(); return false; }

    void* m = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (m == MAP_FAILED) { err = "mmap failed on " + path; close(); return false; }
    base_ = static_cast<const uint8_t*>(m);

    const Header& h = header();
    if (std::memcmp(h.magic, "E57COVLD", 8) != 0) {
        err = "not a point store (bad magic)"; close(); return false;
    }
    if (h.version != kVersion) {
        err = "point store version " + std::to_string(h.version) +
              " but this build expects " + std::to_string(kVersion);
        close(); return false;
    }
    if (h.pointRecordSize != sizeof(lod::StorePoint)) {
        // Silently reading a store written with a different record layout would
        // produce plausible-looking garbage rather than an error.
        err = "point record size mismatch: store has " + std::to_string(h.pointRecordSize) +
              ", this build has " + std::to_string(sizeof(lod::StorePoint));
        close(); return false;
    }

    // Every table must lie inside the mapping; a truncated store must fail
    // here rather than when something reads past the end.
    const uint64_t scanBytes = h.scanCount * sizeof(ScanRecord);
    const uint64_t nodeBytes = h.nodeCount * sizeof(NodeRecord);
    if (h.scanTableOffset + scanBytes > size_ || h.nodeTableOffset + nodeBytes > size_) {
        err = "point store is truncated"; close(); return false;
    }
    scans_ = reinterpret_cast<const ScanRecord*>(base_ + h.scanTableOffset);
    nodes_ = reinterpret_cast<const NodeRecord*>(base_ + h.nodeTableOffset);

    for (uint64_t i = 0; i < h.nodeCount; ++i) {
        const NodeRecord& n = nodes_[i];
        if (n.payloadOffset + uint64_t(n.pointCount) * sizeof(lod::StorePoint) > size_) {
            err = "node " + std::to_string(i) + " payload runs past the end of the store";
            close(); return false;
        }
    }
    return true;
}

const lod::StorePoint* Reader::points(uint64_t nodeIndex) const {
    if (!base_ || nodeIndex >= header().nodeCount) return nullptr;
    return reinterpret_cast<const lod::StorePoint*>(base_ + nodes_[nodeIndex].payloadOffset);
}

lod::Tree Reader::tree() const {
    lod::Tree t;
    if (!base_) return t;
    const Header& h = header();
    for (int i = 0; i < 3; ++i) { t.bounds.lo[i] = h.rootLo[i]; t.bounds.hi[i] = h.rootHi[i]; }
    t.nodes.resize(size_t(h.nodeCount));
    for (uint64_t i = 0; i < h.nodeCount; ++i) {
        const NodeRecord& r = nodes_[i];
        lod::Node n;
        n.level         = r.level;
        n.pointCount    = r.pointCount;
        n.payloadOffset = r.payloadOffset;
        for (int k = 0; k < 3; ++k) { n.bounds.lo[k] = r.lo[k]; n.bounds.hi[k] = r.hi[k]; }
        for (int o = 0; o < 8; ++o) n.child[o] = r.child[o];
        t.nodes[size_t(i)] = n;
    }
    t.totalPoints = h.totalPoints;
    return t;
}

} // namespace store
