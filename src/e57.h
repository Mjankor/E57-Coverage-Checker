// ASTM E2807 (E57) reader — dependency-free.
//
// Written directly against the standard, inverting format knowledge already
// verified in a sibling project: the E57 writer in Mjankor/CartesianCapture
// (Export/E57Writer.swift), whose comments record which details of pagination
// and CRC-32c, the 48-byte file header, the 32-byte
// CompressedVectorSectionHeader and the DataPacket bytestream layout were
// confirmed against real readers — and which had been reconstructed wrongly
// from memory. Avoiding libE57Format drops its Xerces-C
// dependency and — more importantly for this tool — lets the decoder skip
// unwanted fields without unpacking them, which is roughly a 2x saving on a
// corpus where only range and invalid-state are needed out of a prototype that
// also carries intensity and colour.
//
// Scope note: this reads what terrestrial scanners actually emit — bit-packed
// Integer and ScaledInteger, single/double Float, structured scans with
// row/column indices. It is not a general-purpose E57 library.

#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <map>
#include <functional>

namespace e57 {

// ---------------------------------------------------------------------------
// Paged file access
//
// An E57 file is a sequence of 1024-byte physical pages, each holding 1020
// bytes of payload followed by a 4-byte CRC-32c of that payload. Every offset
// stored in the header and in the XML is *physical*; every structure the format
// describes lives in the *logical* (CRC-stripped) address space. Mixing the two
// up is the single easiest way to produce a reader that works on small files
// and fails past the first page boundary.
// ---------------------------------------------------------------------------

class PagedFile {
public:
    static constexpr uint64_t kPageBytes     = 1024;
    static constexpr uint64_t kPageDataBytes = 1020;

    ~PagedFile();
    bool open(const std::string& path, std::string& err);
    void close();

    // Copies `len` bytes starting at logical offset `off` into `dst`.
    bool readLogical(uint64_t off, void* dst, size_t len) const;

    uint64_t physicalSize() const { return size_; }
    uint64_t logicalSize()  const { return toLogical(size_); }

    // Verifies every page checksum. Costs a full pass over the file, so it is
    // opt-in rather than automatic.
    bool verifyCrc(std::string& err) const;

    static uint64_t toLogical (uint64_t phys) {
        return (phys / kPageBytes) * kPageDataBytes + (phys % kPageBytes);
    }
    static uint64_t toPhysical(uint64_t log) {
        return (log / kPageDataBytes) * kPageBytes + (log % kPageDataBytes);
    }

private:
    int            fd_   = -1;
    const uint8_t* base_ = nullptr;
    uint64_t       size_ = 0;
};

uint32_t crc32c(const uint8_t* data, size_t len);

// ---------------------------------------------------------------------------
// Minimal XML
//
// The E57 XML section is a plain element tree: no DTD, no entity declarations,
// no namespace prefixes in practice (a single default xmlns on the root). A
// ~200-line parser covers it and keeps the tool dependency-free.
// ---------------------------------------------------------------------------

struct XmlNode {
    std::string                        name;
    std::map<std::string, std::string> attrs;
    std::string                        text;      // CDATA and character data
    std::vector<XmlNode>               children;

    const XmlNode* child(const std::string& n) const;
    std::string    attr (const std::string& k, const std::string& dflt = "") const;
    double         asDouble(double dflt = 0.0) const;
    int64_t        asInt   (int64_t dflt = 0)  const;
};

bool parseXml(const char* data, size_t len, XmlNode& root, std::string& err);

// ---------------------------------------------------------------------------
// Prototype fields
// ---------------------------------------------------------------------------

enum class FieldType { Integer, ScaledInteger, FloatSingle, FloatDouble, String };

struct ProtoField {
    std::string name;
    FieldType   type    = FieldType::Integer;
    int64_t     minimum = 0;
    int64_t     maximum = 0;
    double      scale   = 1.0;
    double      offset  = 0.0;
    // Bits per packed value for Integer/ScaledInteger. Zero when minimum ==
    // maximum: the value is constant and the encoder emits no data at all for
    // it, which a decoder that assumes >= 1 bit will stall on forever.
    int         bits    = 0;

    bool isPacked() const {
        return type == FieldType::Integer || type == FieldType::ScaledInteger;
    }
};

int bitsNeeded(int64_t minimum, int64_t maximum);

// ---------------------------------------------------------------------------
// Field decoder
//
// One per prototype field. E57's bitpack codec runs a *continuous* bit stream
// per field across packet boundaries: a packet's chunk for a field can end
// mid-value, with the remaining bits supplied by the next packet's chunk. So
// each decoder owns persistent bit state and is fed chunk by chunk, mirroring
// libE57Format's own architecture. Treating each packet's chunk as
// independently byte-aligned decodes the first packet correctly and then
// silently drifts, which is the failure mode this design exists to avoid.
// ---------------------------------------------------------------------------

class FieldDecoder {
public:
    void   init(const ProtoField& f);
    void   feed(const uint8_t* p, size_t n);
    // Records currently decodable. SIZE_MAX for a zero-bit (constant) field,
    // which is never the limiting stream.
    size_t available() const;
    // Decodes `count` records as doubles. Integer and ScaledInteger both widen
    // to double: at 0.05 m voxels the tool never needs more than float64's 53
    // bits of mantissa, and it keeps one code path through the range-image
    // builder.
    void   decode(size_t count, double* out);
    void   reset();

private:
    uint64_t   readBits(int bits);
    void       compact();

    ProtoField           f_;
    std::vector<uint8_t> buf_;
    size_t               bitCursor_ = 0;
};

// ---------------------------------------------------------------------------
// Scans
// ---------------------------------------------------------------------------

struct Pose {
    // Rotation as a quaternion (w, x, y, z); identity when the file omits pose.
    double q[4] = {1.0, 0.0, 0.0, 0.0};
    double t[3] = {0.0, 0.0, 0.0};
};

struct Scan {
    std::string             guid;
    std::string             name;
    Pose                    pose;
    bool                    hasPose     = false;
    uint64_t                recordCount = 0;
    std::vector<ProtoField> proto;

    // Structured-scan extent, when the file declares it. Absent extent is the
    // common case and is recovered from the data instead (see DESIGN.md §4).
    bool     hasIndexBounds = false;
    int64_t  rowMin = 0, rowMax = 0, colMin = 0, colMax = 0;

    // The file's own declared extent. Worth carrying because it is an
    // independent statement of what the data should contain: decoding the
    // points and comparing against it is the cheapest available check that the
    // bit-stream decode did not drift.
    bool     hasCartesianBounds = false;
    double   xMin = 0, xMax = 0, yMin = 0, yMax = 0, zMin = 0, zMax = 0;

    uint64_t cvFileOffset = 0;   // physical, from the XML attribute

    const ProtoField* field(const std::string& n) const;
};

// A decoded block of points, column-major: one contiguous array per requested
// field, which is exactly how the bytestreams are laid out and what the
// range-image builder consumes.
struct PointBlock {
    std::vector<const double*> columns;
    size_t                     count = 0;
};

class Reader {
public:
    bool open(const std::string& path, std::string& err);
    bool verifyCrc(std::string& err) { return file_.verifyCrc(err); }

    size_t      scanCount() const { return scans_.size(); }
    const Scan& scan(size_t i) const { return scans_[i]; }
    const XmlNode& root() const { return root_; }

    // Decodes only `fields`; every other prototype field is advanced past
    // without being unpacked. Returns false and sets `err` if any name is
    // missing from the prototype.
    bool readPoints(size_t scanIndex,
                    const std::vector<std::string>& fields,
                    const std::function<bool(const PointBlock&)>& sink,
                    std::string& err);

private:
    PagedFile         file_;
    XmlNode           root_;
    std::vector<Scan> scans_;
};

} // namespace e57
