// On-disk LOD point store.
//
// The store is written once by the indexer and then only read. Reads are
// mmap'd, so a node's payload is a pointer into the page cache: on Apple
// silicon that same range can be handed to `newBufferWithBytesNoCopy` and read
// by the GPU without ever being copied, and the kernel evicts cold nodes on its
// own. That is what lets a store far larger than RAM be navigated — at a few
// thousand setups the store is hundreds of gigabytes and nothing else would.
//
// Layout, all little-endian:
//
//   [ Header      128 bytes                       ]
//   [ payloads    node points, in write order     ]
//   [ scan table  ScanRecord * scanCount          ]
//   [ node table  NodeRecord * nodeCount          ]
//
// Payloads come first because they are written streaming, as each subtree is
// built, while the tables are only complete at the end.

#pragma once

#include "lod.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace store {

constexpr uint32_t kVersion = 1;

#pragma pack(push, 1)

struct Header {
    char     magic[8];              // "E57COVLD"
    uint32_t version;
    uint32_t pointRecordSize;       // sizeof(lod::StorePoint); a mismatch is fatal
    uint64_t nodeCount;
    uint64_t scanCount;
    uint64_t totalPoints;
    double   origin[3];             // add to a point to get file-frame coordinates
    float    rootLo[3], rootHi[3];  // cubic root bounds, store-local
    uint64_t nodeTableOffset;
    uint64_t scanTableOffset;
    uint32_t gridResolution;
    uint32_t maxLevel;
    uint8_t  reserved[16];
};
static_assert(sizeof(Header) == 128, "Header must stay 128 bytes");

struct ScanRecord {
    char     name[64];
    char     guid[40];
    double   poseT[3];
    double   poseQ[4];              // w, x, y, z
    uint64_t sourcePoints;          // before any decimation
    uint32_t flags;                 // see ScanFlags
    uint32_t reserved;
};
static_assert(sizeof(ScanRecord) == 176, "ScanRecord must stay 176 bytes");

struct NodeRecord {
    uint64_t payloadOffset;
    uint32_t pointCount;
    uint32_t child[8];              // 0 = none, since node 0 is the root
    float    lo[3], hi[3];
    uint8_t  level;
    uint8_t  pad[3];
};
static_assert(sizeof(NodeRecord) == 72, "NodeRecord must stay 72 bytes");

#pragma pack(pop)

enum ScanFlags : uint32_t {
    kScanStructured  = 1u << 0,   // passed the structured check
    kScanPoseApplied = 1u << 1,   // pose was applied to its points
    kScanFrameWarned = 1u << 2,   // non-conformant frame handling
};

// ---------------------------------------------------------------------------

class Writer {
public:
    ~Writer();
    bool open(const std::string& path, std::string& err);

    void setOrigin(double x, double y, double z);
    void setRoot(const lod::Aabb& cubicRoot, uint32_t gridResolution, uint32_t maxLevel);

    // Returns the scan index to put in StorePoint::scanId.
    uint32_t addScan(const ScanRecord& s);

    // Appends a subtree, writing its payloads immediately and remapping its
    // child indices. Returns the index the subtree's root landed at, so the
    // caller can link it into the tree already written.
    bool appendTree(const lod::Tree& tree,
                    const std::vector<std::vector<lod::StorePoint>>& payloads,
                    uint32_t& outRootIndex, std::string& err);

    // Attaches an already-appended subtree under a node written earlier. This
    // is how a bounded-memory build joins per-chunk subtrees to the top of the
    // tree without holding either in memory at once.
    bool linkChild(uint32_t parentNode, int octant, uint32_t childNode, std::string& err);

    bool finish(std::string& err);

    uint64_t nodeCount() const { return nodes_.size(); }
    uint64_t pointCount() const { return totalPoints_; }

private:
    std::FILE*              fp_ = nullptr;
    Header                  header_{};
    std::vector<NodeRecord> nodes_;
    std::vector<ScanRecord> scans_;
    uint64_t                cursor_ = 0;       // next payload byte offset
    uint64_t                totalPoints_ = 0;
    bool                    finished_ = false;
};

// ---------------------------------------------------------------------------

class Reader {
public:
    ~Reader();
    bool open(const std::string& path, std::string& err);
    void close();
    bool isOpen() const { return base_ != nullptr; }

    const Header& header() const { return *reinterpret_cast<const Header*>(base_); }

    uint64_t scanCount() const { return header().scanCount; }
    uint64_t nodeCount() const { return header().nodeCount; }

    const ScanRecord& scan(uint64_t i) const { return scans_[i]; }
    const NodeRecord& node(uint64_t i) const { return nodes_[i]; }

    // Pointer straight into the mapping — no copy, no allocation.
    const lod::StorePoint* points(uint64_t nodeIndex) const;
    uint64_t payloadOffset(uint64_t nodeIndex) const { return nodes_[nodeIndex].payloadOffset; }
    uint64_t payloadBytes(uint64_t nodeIndex) const {
        return uint64_t(nodes_[nodeIndex].pointCount) * sizeof(lod::StorePoint);
    }

    // The whole mapping, for wrapping in a zero-copy GPU buffer.
    const void* mappedBase() const { return base_; }
    uint64_t    mappedSize() const { return size_; }

    // Rebuilds the in-memory tree used by selectNodes. Node records only —
    // payloads stay on disk.
    lod::Tree tree() const;

private:
    int               fd_ = -1;
    const uint8_t*    base_ = nullptr;
    uint64_t          size_ = 0;
    const ScanRecord* scans_ = nullptr;
    const NodeRecord* nodes_ = nullptr;
};

} // namespace store
