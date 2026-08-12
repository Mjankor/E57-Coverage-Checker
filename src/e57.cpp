#include "e57.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <cstring>
#include <cstdlib>
#include <cmath>
#include <limits>

namespace e57 {

// ===========================================================================
// CRC-32c (Castagnoli, reflected polynomial 0x82F63B78)
//
// Same construction as the writer in Mjankor/CartesianCapture. Note the spec
// stores the page checksum most-significant-byte-first while every other
// integer in the format is little-endian — an easy detail to miss in both
// directions.
// ===========================================================================

static const uint32_t* crcTable() {
    static uint32_t t[256];
    static bool built = false;
    if (!built) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k)
                c = (c & 1) ? ((c >> 1) ^ 0x82F63B78u) : (c >> 1);
            t[i] = c;
        }
        built = true;
    }
    return t;
}

uint32_t crc32c(const uint8_t* data, size_t len) {
    const uint32_t* t = crcTable();
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i)
        crc = (crc >> 8) ^ t[(crc ^ data[i]) & 0xFF];
    return crc ^ 0xFFFFFFFFu;
}

// ===========================================================================
// PagedFile
// ===========================================================================

PagedFile::~PagedFile() { close(); }

void PagedFile::close() {
    if (base_) { ::munmap(const_cast<uint8_t*>(base_), size_); base_ = nullptr; }
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
    size_ = 0;
}

bool PagedFile::open(const std::string& path, std::string& err) {
    close();
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) { err = "cannot open " + path; return false; }

    struct stat st{};
    if (::fstat(fd_, &st) != 0) { err = "fstat failed on " + path; close(); return false; }
    size_ = static_cast<uint64_t>(st.st_size);
    if (size_ < kPageBytes) { err = "file shorter than one page: " + path; close(); return false; }

    void* m = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (m == MAP_FAILED) { err = "mmap failed on " + path; close(); return false; }
    base_ = static_cast<const uint8_t*>(m);
    return true;
}

bool PagedFile::readLogical(uint64_t off, void* dst, size_t len) const {
    uint8_t* out = static_cast<uint8_t*>(dst);
    while (len > 0) {
        uint64_t page      = off / kPageDataBytes;
        uint64_t inPage    = off % kPageDataBytes;
        uint64_t phys      = page * kPageBytes + inPage;
        size_t   available = static_cast<size_t>(kPageDataBytes - inPage);
        size_t   n         = len < available ? len : available;
        if (phys + n > size_) return false;
        std::memcpy(out, base_ + phys, n);
        out += n; off += n; len -= n;
    }
    return true;
}

bool PagedFile::verifyCrc(std::string& err) const {
    if (size_ % kPageBytes != 0) {
        err = "file size is not a whole number of 1024-byte pages";
        return false;
    }
    for (uint64_t p = 0; p < size_ / kPageBytes; ++p) {
        const uint8_t* page = base_ + p * kPageBytes;
        uint32_t want = (uint32_t(page[kPageDataBytes + 0]) << 24) |
                        (uint32_t(page[kPageDataBytes + 1]) << 16) |
                        (uint32_t(page[kPageDataBytes + 2]) <<  8) |
                        (uint32_t(page[kPageDataBytes + 3]));
        if (crc32c(page, kPageDataBytes) != want) {
            err = "CRC mismatch on page " + std::to_string(p);
            return false;
        }
    }
    return true;
}

// ===========================================================================
// XML
// ===========================================================================

const XmlNode* XmlNode::child(const std::string& n) const {
    for (const auto& c : children) if (c.name == n) return &c;
    return nullptr;
}

std::string XmlNode::attr(const std::string& k, const std::string& dflt) const {
    auto it = attrs.find(k);
    return it == attrs.end() ? dflt : it->second;
}

double  XmlNode::asDouble(double dflt) const {
    if (text.empty()) return dflt;
    return std::strtod(text.c_str(), nullptr);
}

int64_t XmlNode::asInt(int64_t dflt) const {
    if (text.empty()) return dflt;
    return static_cast<int64_t>(std::strtoll(text.c_str(), nullptr, 10));
}

namespace {

struct XmlParser {
    const char* d;
    size_t      n, i = 0;
    std::string err;

    XmlParser(const char* data, size_t len) : d(data), n(len) {}

    bool eof() const { return i >= n; }
    void skipSpace() { while (i < n && (d[i]==' '||d[i]=='\t'||d[i]=='\r'||d[i]=='\n')) ++i; }

    bool match(const char* s) {
        size_t len = std::strlen(s);
        if (i + len > n || std::memcmp(d + i, s, len) != 0) return false;
        i += len; return true;
    }

    // Consumes prologs, comments and doctype declarations between elements.
    void skipMisc() {
        for (;;) {
            skipSpace();
            if (match("<?")) { while (i < n && !match("?>")) ++i; continue; }
            if (match("<!--")) { while (i < n && !match("-->")) ++i; continue; }
            if (i + 2 < n && d[i]=='<' && d[i+1]=='!' && d[i+2]!='[') {
                while (i < n && d[i] != '>') ++i;
                if (i < n) ++i;
                continue;
            }
            return;
        }
    }

    static void appendEntity(const std::string& e, std::string& out) {
        if      (e == "lt")   out += '<';
        else if (e == "gt")   out += '>';
        else if (e == "amp")  out += '&';
        else if (e == "quot") out += '"';
        else if (e == "apos") out += '\'';
        else if (!e.empty() && e[0] == '#') {
            long cp = (e.size() > 1 && (e[1]=='x'||e[1]=='X'))
                    ? std::strtol(e.c_str() + 2, nullptr, 16)
                    : std::strtol(e.c_str() + 1, nullptr, 10);
            // E57 payloads are ASCII in practice; anything wider is passed
            // through as UTF-8 rather than dropped.
            if (cp < 0x80) out += static_cast<char>(cp);
            else if (cp < 0x800) {
                out += static_cast<char>(0xC0 | (cp >> 6));
                out += static_cast<char>(0x80 | (cp & 0x3F));
            } else {
                out += static_cast<char>(0xE0 | (cp >> 12));
                out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                out += static_cast<char>(0x80 | (cp & 0x3F));
            }
        }
    }

    std::string parseName() {
        size_t s = i;
        while (i < n && (isalnum((unsigned char)d[i]) || d[i]=='_' || d[i]=='-' ||
                         d[i]=='.' || d[i]==':')) ++i;
        std::string nm(d + s, i - s);
        // Strip any namespace prefix; E57 uses a single default xmlns, but a
        // producer emitting prefixed elements should still parse.
        size_t colon = nm.rfind(':');
        return colon == std::string::npos ? nm : nm.substr(colon + 1);
    }

    bool parseElement(XmlNode& out) {
        if (!match("<")) { err = "expected '<'"; return false; }
        out.name = parseName();
        if (out.name.empty()) { err = "empty element name"; return false; }

        for (;;) {
            skipSpace();
            if (i < n && d[i] == '/') { ++i; if (!match(">")) { err = "bad self-close"; return false; } return true; }
            if (match(">")) break;
            std::string key = parseName();
            if (key.empty()) { err = "bad attribute name in <" + out.name + ">"; return false; }
            skipSpace();
            if (!match("=")) { err = "expected '=' after " + key; return false; }
            skipSpace();
            char q = (i < n) ? d[i] : 0;
            if (q != '"' && q != '\'') { err = "unquoted attribute " + key; return false; }
            ++i;
            std::string val;
            while (i < n && d[i] != q) {
                if (d[i] == '&') {
                    size_t s = ++i;
                    while (i < n && d[i] != ';') ++i;
                    appendEntity(std::string(d + s, i - s), val);
                    if (i < n) ++i;
                } else val += d[i++];
            }
            if (i >= n) { err = "unterminated attribute " + key; return false; }
            ++i;
            out.attrs[key] = val;
        }

        // Content: character data, CDATA, child elements, until the end tag.
        std::string text;
        for (;;) {
            if (i >= n) { err = "unterminated <" + out.name + ">"; return false; }
            if (match("<![CDATA[")) {
                size_t s = i;
                while (i < n && !(d[i]==']' && i+2 < n && d[i+1]==']' && d[i+2]=='>')) ++i;
                text.append(d + s, i - s);
                if (i < n) i += 3;
                continue;
            }
            if (match("<!--")) { while (i < n && !match("-->")) ++i; continue; }
            if (i + 1 < n && d[i]=='<' && d[i+1]=='/') {
                i += 2;
                std::string close = parseName();
                skipSpace();
                if (!match(">")) { err = "bad end tag for " + out.name; return false; }
                if (close != out.name) { err = "mismatched </" + close + "> in <" + out.name + ">"; return false; }
                break;
            }
            if (d[i] == '<') {
                XmlNode c;
                if (!parseElement(c)) return false;
                out.children.push_back(std::move(c));
                continue;
            }
            if (d[i] == '&') {
                size_t s = ++i;
                while (i < n && d[i] != ';') ++i;
                appendEntity(std::string(d + s, i - s), text);
                if (i < n) ++i;
                continue;
            }
            text += d[i++];
        }

        // Trim: E57 scalars are written as indented element text.
        size_t b = text.find_first_not_of(" \t\r\n");
        size_t e = text.find_last_not_of(" \t\r\n");
        out.text = (b == std::string::npos) ? std::string() : text.substr(b, e - b + 1);
        return true;
    }
};

} // namespace

bool parseXml(const char* data, size_t len, XmlNode& root, std::string& err) {
    XmlParser p(data, len);
    p.skipMisc();
    if (p.eof()) { err = "empty XML section"; return false; }
    if (!p.parseElement(root)) { err = p.err; return false; }
    return true;
}

// ===========================================================================
// Prototype fields and decoding
// ===========================================================================

int bitsNeeded(int64_t minimum, int64_t maximum) {
    // Unsigned subtraction so the full int64 span (the spec's default Integer
    // bounds) wraps to 0xFFFF...FF and yields 64 rather than overflowing.
    uint64_t range = static_cast<uint64_t>(maximum) - static_cast<uint64_t>(minimum);
    int bits = 0;
    while (range) { ++bits; range >>= 1; }
    return bits;
}

void FieldDecoder::init(const ProtoField& f) {
    f_ = f;
    reset();
}

void FieldDecoder::reset() {
    buf_.clear();
    bitCursor_ = 0;
}

void FieldDecoder::feed(const uint8_t* p, size_t n) {
    buf_.insert(buf_.end(), p, p + n);
}

size_t FieldDecoder::available() const {
    if (f_.isPacked()) {
        // A constant field (minimum == maximum) carries no data and must never
        // gate how many records the other streams can produce.
        if (f_.bits == 0) return std::numeric_limits<size_t>::max();
        size_t bits = buf_.size() * 8 - bitCursor_;
        return bits / static_cast<size_t>(f_.bits);
    }
    size_t width = (f_.type == FieldType::FloatSingle) ? 4 : 8;
    size_t bytes = buf_.size() - (bitCursor_ >> 3);
    return bytes / width;
}

uint64_t FieldDecoder::readBits(int bits) {
    if (bits <= 0) return 0;
    uint64_t v = 0;
    int got = 0;
    while (got < bits) {
        size_t byte = (bitCursor_ + got) >> 3;
        int    off  = static_cast<int>((bitCursor_ + got) & 7);
        int    take = 8 - off;
        if (take > bits - got) take = bits - got;
        uint64_t chunk = (static_cast<uint64_t>(buf_[byte]) >> off) &
                         ((1ull << take) - 1ull);
        v |= chunk << got;
        got += take;
    }
    bitCursor_ += static_cast<size_t>(bits);
    return v;
}

void FieldDecoder::decode(size_t count, double* out) {
    switch (f_.type) {
    case FieldType::Integer:
        for (size_t k = 0; k < count; ++k) {
            uint64_t raw = readBits(f_.bits);
            out[k] = static_cast<double>(f_.minimum + static_cast<int64_t>(raw));
        }
        break;
    case FieldType::ScaledInteger:
        for (size_t k = 0; k < count; ++k) {
            uint64_t raw = readBits(f_.bits);
            int64_t  iv  = f_.minimum + static_cast<int64_t>(raw);
            out[k] = f_.offset + static_cast<double>(iv) * f_.scale;
        }
        break;
    case FieldType::FloatSingle:
        for (size_t k = 0; k < count; ++k) {
            float v; std::memcpy(&v, buf_.data() + (bitCursor_ >> 3), 4);
            bitCursor_ += 32;
            out[k] = static_cast<double>(v);
        }
        break;
    case FieldType::FloatDouble:
        for (size_t k = 0; k < count; ++k) {
            double v; std::memcpy(&v, buf_.data() + (bitCursor_ >> 3), 8);
            bitCursor_ += 64;
            out[k] = v;
        }
        break;
    case FieldType::String:
        for (size_t k = 0; k < count; ++k) out[k] = 0.0;
        break;
    }
    compact();
}

void FieldDecoder::compact() {
    size_t whole = bitCursor_ >> 3;
    if (whole == 0) return;
    buf_.erase(buf_.begin(), buf_.begin() + static_cast<ptrdiff_t>(whole));
    bitCursor_ &= 7;
}

// ===========================================================================
// Reader
// ===========================================================================

const ProtoField* Scan::field(const std::string& n) const {
    for (const auto& f : proto) if (f.name == n) return &f;
    return nullptr;
}

namespace {

bool parseProtoField(const XmlNode& x, ProtoField& f, std::string& err) {
    f.name = x.name;
    const std::string t = x.attr("type");

    auto num = [&](const char* k, double dflt) {
        std::string v = x.attr(k);
        return v.empty() ? dflt : std::strtod(v.c_str(), nullptr);
    };
    auto inum = [&](const char* k, int64_t dflt) {
        std::string v = x.attr(k);
        return v.empty() ? dflt : static_cast<int64_t>(std::strtoll(v.c_str(), nullptr, 10));
    };

    if (t == "Integer") {
        f.type    = FieldType::Integer;
        // Spec default is the full int64 range, which encodes as 64 raw bits.
        f.minimum = inum("minimum", std::numeric_limits<int64_t>::min());
        f.maximum = inum("maximum", std::numeric_limits<int64_t>::max());
        f.bits    = bitsNeeded(f.minimum, f.maximum);
    } else if (t == "ScaledInteger") {
        f.type    = FieldType::ScaledInteger;
        f.minimum = inum("minimum", std::numeric_limits<int64_t>::min());
        f.maximum = inum("maximum", std::numeric_limits<int64_t>::max());
        f.scale   = num("scale", 1.0);
        f.offset  = num("offset", 0.0);
        f.bits    = bitsNeeded(f.minimum, f.maximum);
    } else if (t == "Float") {
        f.type = (x.attr("precision", "double") == "single")
               ? FieldType::FloatSingle : FieldType::FloatDouble;
    } else if (t == "String") {
        f.type = FieldType::String;
    } else {
        err = "unsupported prototype field type '" + t + "' on " + x.name;
        return false;
    }
    return true;
}

void parsePose(const XmlNode& x, Scan& s) {
    const XmlNode* pose = x.child("pose");
    if (!pose) return;
    s.hasPose = true;
    if (const XmlNode* r = pose->child("rotation")) {
        const char* k[4] = {"w", "x", "y", "z"};
        for (int j = 0; j < 4; ++j)
            if (const XmlNode* c = r->child(k[j])) s.pose.q[j] = c->asDouble(s.pose.q[j]);
    }
    if (const XmlNode* t = pose->child("translation")) {
        const char* k[3] = {"x", "y", "z"};
        for (int j = 0; j < 3; ++j)
            if (const XmlNode* c = t->child(k[j])) s.pose.t[j] = c->asDouble(0.0);
    }
}

void parseIndexBounds(const XmlNode& x, Scan& s) {
    const XmlNode* ib = x.child("indexBounds");
    if (!ib) return;
    const XmlNode* r0 = ib->child("rowMinimum");
    const XmlNode* r1 = ib->child("rowMaximum");
    const XmlNode* c0 = ib->child("columnMinimum");
    const XmlNode* c1 = ib->child("columnMaximum");
    if (!r0 || !r1 || !c0 || !c1) return;
    s.hasIndexBounds = true;
    s.rowMin = r0->asInt(); s.rowMax = r1->asInt();
    s.colMin = c0->asInt(); s.colMax = c1->asInt();
}

} // namespace

bool Reader::open(const std::string& path, std::string& err) {
    if (!file_.open(path, err)) return false;

    // 48-byte file header, at logical 0.
    uint8_t hdr[48];
    if (!file_.readLogical(0, hdr, sizeof(hdr))) { err = "short read on file header"; return false; }
    if (std::memcmp(hdr, "ASTM-E57", 8) != 0) { err = "not an E57 file (bad signature)"; return false; }

    uint64_t xmlPhysicalOffset, xmlLogicalLength;
    std::memcpy(&xmlPhysicalOffset, hdr + 24, 8);
    std::memcpy(&xmlLogicalLength,  hdr + 32, 8);

    std::vector<char> xml(static_cast<size_t>(xmlLogicalLength));
    if (xml.empty()) { err = "empty XML section"; return false; }
    if (!file_.readLogical(PagedFile::toLogical(xmlPhysicalOffset), xml.data(), xml.size())) {
        err = "short read on XML section"; return false;
    }
    if (!parseXml(xml.data(), xml.size(), root_, err)) return false;

    const XmlNode* data3D = root_.child("data3D");
    if (!data3D) { err = "no data3D vector in file"; return false; }

    for (const auto& vc : data3D->children) {
        if (vc.name != "vectorChild") continue;
        Scan s;
        if (const XmlNode* g = vc.child("guid")) s.guid = g->text;
        if (const XmlNode* n = vc.child("name")) s.name = n->text;
        parsePose(vc, s);
        parseIndexBounds(vc, s);

        const XmlNode* pts = vc.child("points");
        if (!pts) { err = "scan '" + s.name + "' has no points node"; return false; }
        s.recordCount  = static_cast<uint64_t>(std::strtoull(pts->attr("recordCount", "0").c_str(), nullptr, 10));
        s.cvFileOffset = static_cast<uint64_t>(std::strtoull(pts->attr("fileOffset",  "0").c_str(), nullptr, 10));

        const XmlNode* proto = pts->child("prototype");
        if (!proto) { err = "scan '" + s.name + "' has no prototype"; return false; }
        for (const auto& f : proto->children) {
            ProtoField pf;
            if (!parseProtoField(f, pf, err)) return false;
            s.proto.push_back(pf);
        }
        scans_.push_back(std::move(s));
    }
    if (scans_.empty()) { err = "file contains no scans"; return false; }
    return true;
}

bool Reader::readPoints(size_t scanIndex,
                        const std::vector<std::string>& fields,
                        const std::function<bool(const PointBlock&)>& sink,
                        std::string& err) {
    if (scanIndex >= scans_.size()) { err = "scan index out of range"; return false; }
    const Scan& s = scans_[scanIndex];

    // Map requested names onto prototype positions. Bytestreams are positional
    // — there is no field identifier in the binary — so the index into the
    // prototype is the only link between a stream and its meaning.
    std::vector<size_t> want;
    want.reserve(fields.size());
    for (const auto& name : fields) {
        size_t idx = SIZE_MAX;
        for (size_t k = 0; k < s.proto.size(); ++k)
            if (s.proto[k].name == name) { idx = k; break; }
        if (idx == SIZE_MAX) { err = "field '" + name + "' not in prototype of scan " + s.name; return false; }
        want.push_back(idx);
    }

    std::vector<FieldDecoder> dec(want.size());
    for (size_t k = 0; k < want.size(); ++k) dec[k].init(s.proto[want[k]]);

    // 32-byte CompressedVectorSectionHeader at the XML's fileOffset.
    const uint64_t secLogical = PagedFile::toLogical(s.cvFileOffset);
    uint8_t sec[32];
    if (!file_.readLogical(secLogical, sec, sizeof(sec))) { err = "short read on CV section header"; return false; }
    if (sec[0] != 1) {
        err = "expected sectionId=1 at CV offset, got " + std::to_string(int(sec[0]));
        return false;
    }
    uint64_t sectionLogicalLength, dataPhysicalOffset;
    std::memcpy(&sectionLogicalLength, sec + 8,  8);
    std::memcpy(&dataPhysicalOffset,   sec + 16, 8);

    const uint64_t endLogical = secLogical + sectionLogicalLength;
    uint64_t       at         = PagedFile::toLogical(dataPhysicalOffset);

    std::vector<std::vector<double>> outBuf(want.size());
    std::vector<const double*>       cols(want.size());
    std::vector<uint8_t>             packet;
    uint64_t                         decoded = 0;

    while (decoded < s.recordCount && at < endLogical) {
        uint8_t ph[6];
        if (!file_.readLogical(at, ph, 4)) { err = "short read on packet header"; return false; }
        const uint8_t  type   = ph[0];
        const uint8_t  flags  = ph[1];
        uint16_t       lenM1;  std::memcpy(&lenM1, ph + 2, 2);
        const size_t   pktLen = static_cast<size_t>(lenM1) + 1;

        if (type != 1) {   // index (0) and ignore/empty (2) packets carry no records
            at += pktLen;
            continue;
        }

        packet.resize(pktLen);
        if (!file_.readLogical(at, packet.data(), pktLen)) { err = "short read on data packet"; return false; }
        at += pktLen;

        // COMPRESSOR_RESTART: streams begin afresh, so any carried bit state is
        // stale and must be dropped.
        if (flags & 0x01) for (auto& d : dec) d.reset();

        uint16_t bsCount; std::memcpy(&bsCount, packet.data() + 4, 2);
        if (static_cast<size_t>(bsCount) != s.proto.size()) {
            err = "packet declares " + std::to_string(bsCount) +
                  " bytestreams but prototype has " + std::to_string(s.proto.size());
            return false;
        }

        const size_t tableOff = 6;
        const size_t dataOff  = tableOff + 2u * bsCount;
        if (dataOff > pktLen) { err = "packet too short for its bytestream table"; return false; }

        // Only requested fields are fed. Every other stream is stepped over
        // without being unpacked at all — each field's bit stream is
        // independent, so skipping one costs nothing downstream.
        size_t cursor = dataOff;
        for (uint16_t b = 0; b < bsCount; ++b) {
            uint16_t blen; std::memcpy(&blen, packet.data() + tableOff + 2u * b, 2);
            if (cursor + blen > pktLen) { err = "bytestream overruns packet"; return false; }
            for (size_t k = 0; k < want.size(); ++k)
                if (want[k] == b) dec[k].feed(packet.data() + cursor, blen);
            cursor += blen;
        }

        // Records decodable now: limited by the shortest requested stream, and
        // by the declared record count.
        size_t n = SIZE_MAX;
        for (auto& d : dec) { size_t a = d.available(); if (a < n) n = a; }
        const uint64_t remaining = s.recordCount - decoded;
        // Every requested field constant (or none requested): no stream paces
        // the decode, so the declared record count is the only bound. Without
        // this the loop would emit nothing and then report a truncated stream.
        if (n == SIZE_MAX || n > remaining) n = static_cast<size_t>(remaining);
        if (n == 0) continue;

        for (size_t k = 0; k < want.size(); ++k) {
            outBuf[k].resize(n);
            dec[k].decode(n, outBuf[k].data());
            cols[k] = outBuf[k].data();
        }
        PointBlock blk;
        blk.columns = cols;
        blk.count   = n;
        if (!sink(blk)) return true;   // consumer asked to stop
        decoded += n;
    }

    if (decoded < s.recordCount) {
        err = "stream ended after " + std::to_string(decoded) + " of " +
              std::to_string(s.recordCount) + " records";
        return false;
    }
    return true;
}

} // namespace e57
