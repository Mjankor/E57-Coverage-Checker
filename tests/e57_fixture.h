// Generates E57 files for round-trip testing the reader.
//
// This exists because the reference writer this reader was derived from (in
// Mjankor/CartesianCapture) only ever emits the byte-aligned subset — raw
// Integer 0-255 and raw Float — while terrestrial scanners overwhelmingly
// write bit-packed ScaledInteger. The reader has to handle the packed case, so
// the fixtures have to produce it — including values that straddle packet
// boundaries, which is where a decoder that re-aligns per packet starts to
// drift instead of failing outright.
//
// Header-only: test support, not shipped code.

#pragma once

#include "../src/e57.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace fixture {

// Accumulates a continuous bit stream, handing out whole bytes on demand and
// retaining the trailing partial byte — the same discipline the E57 encoder
// uses, and the reason a field's stream stays continuous across packets.
class BitBuf {
public:
    void appendBits(uint64_t v, int nbits) {
        for (int i = 0; i < nbits; ++i) {
            if ((bits_ & 7) == 0) bytes_.push_back(0);
            if ((v >> i) & 1ull) bytes_.back() |= static_cast<uint8_t>(1u << (bits_ & 7));
            ++bits_;
        }
    }
    void appendRaw(const void* p, size_t n) {
        const uint8_t* q = static_cast<const uint8_t*>(p);
        for (size_t i = 0; i < n; ++i) appendBits(q[i], 8);
    }
    std::vector<uint8_t> takeWholeBytes() {
        size_t whole = bits_ / 8;
        std::vector<uint8_t> out(bytes_.begin(), bytes_.begin() + static_cast<ptrdiff_t>(whole));
        std::vector<uint8_t> rest(bytes_.begin() + static_cast<ptrdiff_t>(whole), bytes_.end());
        bytes_.swap(rest);
        bits_ -= whole * 8;
        return out;
    }
    std::vector<uint8_t> flush() {
        while (bits_ & 7) appendBits(0, 1);
        return takeWholeBytes();
    }
    size_t pendingBits() const { return bits_; }

private:
    std::vector<uint8_t> bytes_;
    size_t               bits_ = 0;
};

struct Field {
    std::string    name;
    e57::FieldType type    = e57::FieldType::ScaledInteger;
    int64_t        minimum = 0;
    int64_t        maximum = 0;
    double         scale   = 1.0;
    double         offset  = 0.0;
};

struct Scan {
    std::string                      name = "Scan";
    bool                             hasPose = false;
    double                           q[4] = {1, 0, 0, 0};
    double                           t[3] = {0, 0, 0};
    std::vector<Field>               fields;
    // Per field, one value per record. For packed fields these are the *raw*
    // integer values (before scale/offset); for Float fields, the values.
    std::vector<std::vector<double>> data;

    size_t recordCount() const { return data.empty() ? 0 : data[0].size(); }
};

namespace detail {

inline void put16(std::vector<uint8_t>& v, uint16_t x) {
    v.push_back(uint8_t(x & 0xFF)); v.push_back(uint8_t(x >> 8));
}
inline void put32(std::vector<uint8_t>& v, uint32_t x) {
    for (int i = 0; i < 4; ++i) v.push_back(uint8_t((x >> (8 * i)) & 0xFF));
}
inline void put64(std::vector<uint8_t>& v, uint64_t x) {
    for (int i = 0; i < 8; ++i) v.push_back(uint8_t((x >> (8 * i)) & 0xFF));
}

// Encodes one scan's data packets. Bytestreams are positional and appear in
// prototype order; the packet header is 6 bytes, then one uint16 length per
// field, then the streams back to back, then zero padding to 4 bytes.
inline std::vector<uint8_t> encodePackets(const Scan& s, size_t recordsPerPacket) {
    const size_t nf = s.fields.size();
    const size_t nr = s.recordCount();

    std::vector<BitBuf> bufs(nf);
    std::vector<int>    bits(nf, 0);
    for (size_t f = 0; f < nf; ++f)
        if (s.fields[f].type == e57::FieldType::Integer ||
            s.fields[f].type == e57::FieldType::ScaledInteger)
            bits[f] = e57::bitsNeeded(s.fields[f].minimum, s.fields[f].maximum);

    std::vector<uint8_t> out;
    for (size_t start = 0; start < nr; start += recordsPerPacket) {
        const size_t n    = std::min(recordsPerPacket, nr - start);
        const bool   last = (start + n >= nr);

        std::vector<std::vector<uint8_t>> streams(nf);
        for (size_t f = 0; f < nf; ++f) {
            const Field& fd = s.fields[f];
            for (size_t k = 0; k < n; ++k) {
                double raw = s.data[f][start + k];
                switch (fd.type) {
                case e57::FieldType::Integer:
                case e57::FieldType::ScaledInteger: {
                    int64_t  iv  = static_cast<int64_t>(raw);
                    uint64_t enc = static_cast<uint64_t>(iv - fd.minimum);
                    bufs[f].appendBits(enc, bits[f]);
                    break;
                }
                case e57::FieldType::FloatSingle: {
                    float v = static_cast<float>(raw);
                    bufs[f].appendRaw(&v, 4);
                    break;
                }
                case e57::FieldType::FloatDouble: {
                    double v = raw;
                    bufs[f].appendRaw(&v, 8);
                    break;
                }
                case e57::FieldType::String:
                    break;
                }
            }
            // Whole bytes go out now; a trailing partial byte stays buffered
            // and leads the next packet's chunk. On the final packet it is
            // padded out, since there is no next packet to carry it.
            streams[f] = last ? bufs[f].flush() : bufs[f].takeWholeBytes();
        }

        size_t payload = 0;
        for (const auto& st : streams) payload += st.size();
        const size_t unpadded = 6 + 2 * nf + payload;
        const size_t pad      = (4 - unpadded % 4) % 4;
        const size_t pktLen   = unpadded + pad;

        out.push_back(1);                                   // packetType = DATA
        out.push_back(0);                                   // packetFlags
        put16(out, static_cast<uint16_t>(pktLen - 1));      // covers the padded packet
        put16(out, static_cast<uint16_t>(nf));              // bytestreamCount
        for (const auto& st : streams) put16(out, static_cast<uint16_t>(st.size()));
        for (const auto& st : streams) out.insert(out.end(), st.begin(), st.end());
        for (size_t i = 0; i < pad; ++i) out.push_back(0);
    }
    return out;
}

inline std::string protoXml(const Field& f) {
    char buf[512];
    switch (f.type) {
    case e57::FieldType::Integer:
        std::snprintf(buf, sizeof(buf),
            "     <%s type=\"Integer\" minimum=\"%lld\" maximum=\"%lld\"/>\n",
            f.name.c_str(), (long long)f.minimum, (long long)f.maximum);
        break;
    case e57::FieldType::ScaledInteger:
        std::snprintf(buf, sizeof(buf),
            "     <%s type=\"ScaledInteger\" minimum=\"%lld\" maximum=\"%lld\" scale=\"%.17g\" offset=\"%.17g\"/>\n",
            f.name.c_str(), (long long)f.minimum, (long long)f.maximum, f.scale, f.offset);
        break;
    case e57::FieldType::FloatSingle:
        std::snprintf(buf, sizeof(buf), "     <%s type=\"Float\" precision=\"single\"/>\n", f.name.c_str());
        break;
    default:
        std::snprintf(buf, sizeof(buf), "     <%s type=\"Float\" precision=\"double\"/>\n", f.name.c_str());
        break;
    }
    return buf;
}

inline std::vector<uint8_t> paginate(const std::vector<uint8_t>& logical) {
    const size_t D = e57::PagedFile::kPageDataBytes;
    std::vector<uint8_t> out;
    for (size_t i = 0; i < logical.size(); i += D) {
        std::vector<uint8_t> page(logical.begin() + static_cast<ptrdiff_t>(i),
                                  logical.begin() + static_cast<ptrdiff_t>(std::min(i + D, logical.size())));
        page.resize(D, 0);
        uint32_t crc = e57::crc32c(page.data(), D);
        out.insert(out.end(), page.begin(), page.end());
        // Spec 9.2: the page CRC is stored most-significant byte first, unlike
        // every other integer in the format.
        out.push_back(uint8_t(crc >> 24)); out.push_back(uint8_t(crc >> 16));
        out.push_back(uint8_t(crc >>  8)); out.push_back(uint8_t(crc));
    }
    return out;
}

} // namespace detail

// Writes `scans` to `path`. `recordsPerPacket` deliberately small in tests so
// packed values straddle packet boundaries.
inline bool write(const std::string& path,
                  const std::vector<Scan>& scans,
                  size_t recordsPerPacket = 64) {
    using namespace detail;

    std::vector<std::vector<uint8_t>> packets;
    packets.reserve(scans.size());
    for (const auto& s : scans) packets.push_back(encodePackets(s, recordsPerPacket));

    // Pass 1: XML with fixed-width 10-char offset placeholders, so patching
    // them is length-preserving and the measured length stays valid.
    std::string xml =
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<e57Root type=\"Structure\" xmlns=\"http://www.astm.org/COMMIT/E57/2010-e57-v1.0\">\n"
        " <formatName type=\"String\"><![CDATA[ASTM E57 3D Imaging Data File]]></formatName>\n"
        " <guid type=\"String\"><![CDATA[{00000000-0000-0000-0000-000000000000}]]></guid>\n"
        " <versionMajor type=\"Integer\">1</versionMajor>\n"
        " <versionMinor type=\"Integer\">0</versionMinor>\n"
        " <e57LibraryVersion type=\"String\"><![CDATA[e57cov test fixture]]></e57LibraryVersion>\n"
        " <data3D type=\"Vector\" allowHeterogeneousChildren=\"0\">\n";

    for (size_t i = 0; i < scans.size(); ++i) {
        const Scan& s = scans[i];
        char tok[32];
        std::snprintf(tok, sizeof(tok), "OFF%07zu", i);   // exactly 10 chars
        char buf[1024];

        xml += "  <vectorChild type=\"Structure\">\n";
        std::snprintf(buf, sizeof(buf),
            "   <guid type=\"String\"><![CDATA[{scan-%zu}]]></guid>\n"
            "   <name type=\"String\"><![CDATA[%s]]></name>\n", i, s.name.c_str());
        xml += buf;

        if (s.hasPose) {
            std::snprintf(buf, sizeof(buf),
                "   <pose type=\"Structure\">\n"
                "    <rotation type=\"Structure\">\n"
                "     <w type=\"Float\">%.17g</w><x type=\"Float\">%.17g</x>"
                "<y type=\"Float\">%.17g</y><z type=\"Float\">%.17g</z>\n"
                "    </rotation>\n"
                "    <translation type=\"Structure\">\n"
                "     <x type=\"Float\">%.17g</x><y type=\"Float\">%.17g</y>"
                "<z type=\"Float\">%.17g</z>\n"
                "    </translation>\n"
                "   </pose>\n",
                s.q[0], s.q[1], s.q[2], s.q[3], s.t[0], s.t[1], s.t[2]);
            xml += buf;
        }

        std::snprintf(buf, sizeof(buf),
            "   <points type=\"CompressedVector\" fileOffset=\"%s\" recordCount=\"%zu\">\n"
            "    <prototype type=\"Structure\">\n", tok, s.recordCount());
        xml += buf;
        for (const auto& f : s.fields) xml += protoXml(f);
        xml += "    </prototype>\n"
               "    <codecs type=\"Vector\" allowHeterogeneousChildren=\"1\"/>\n"
               "   </points>\n"
               "  </vectorChild>\n";
    }
    xml += " </data3D>\n"
           " <images2D type=\"Vector\" allowHeterogeneousChildren=\"0\"/>\n"
           "</e57Root>";

    const size_t headerLogical = 48;
    const size_t xmlLen        = xml.size();

    // Pass 2: lay the sections out and patch in their physical offsets.
    uint64_t cur = headerLogical + xmlLen;
    std::vector<uint8_t> bin;
    for (size_t i = 0; i < scans.size(); ++i) {
        const uint64_t sectionLogicalLength = 32 + packets[i].size();
        const uint64_t fileOffsetPhys       = e57::PagedFile::toPhysical(cur);
        const uint64_t dataPhysicalOffset   = e57::PagedFile::toPhysical(cur + 32);

        char tok[32], rep[32];
        std::snprintf(tok, sizeof(tok), "OFF%07zu", i);
        std::snprintf(rep, sizeof(rep), "%010llu", (unsigned long long)fileOffsetPhys);
        size_t at = xml.find(tok);
        if (at == std::string::npos || std::strlen(rep) != 10) return false;
        xml.replace(at, 10, rep);

        std::vector<uint8_t> sec;
        sec.push_back(1);                              // sectionId
        for (int k = 0; k < 7; ++k) sec.push_back(0);  // reserved
        put64(sec, sectionLogicalLength);
        put64(sec, dataPhysicalOffset);
        put64(sec, 0);                                 // indexPhysicalOffset: no index
        bin.insert(bin.end(), sec.begin(), sec.end());
        bin.insert(bin.end(), packets[i].begin(), packets[i].end());
        cur += sectionLogicalLength;
    }
    if (xml.size() != xmlLen) return false;

    const size_t totalLogical = headerLogical + xmlLen + bin.size();
    const size_t nPages       = (totalLogical + e57::PagedFile::kPageDataBytes - 1)
                              / e57::PagedFile::kPageDataBytes;

    std::vector<uint8_t> logical;
    logical.reserve(totalLogical);
    const char sig[8] = {'A','S','T','M','-','E','5','7'};
    logical.insert(logical.end(), sig, sig + 8);
    put32(logical, 1);                                                    // versionMajor
    put32(logical, 0);                                                    // versionMinor
    put64(logical, nPages * e57::PagedFile::kPageBytes);                  // filePhysicalLength
    put64(logical, e57::PagedFile::toPhysical(headerLogical));            // xmlPhysicalOffset
    put64(logical, xmlLen);                                               // xmlLogicalLength
    put64(logical, e57::PagedFile::kPageBytes);                           // pageSize
    logical.insert(logical.end(), xml.begin(), xml.end());
    logical.insert(logical.end(), bin.begin(), bin.end());

    std::vector<uint8_t> file = paginate(logical);
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return false;
    bool ok = std::fwrite(file.data(), 1, file.size(), fp) == file.size();
    std::fclose(fp);
    return ok;
}

} // namespace fixture
