#include "ply.h"

#include <cstdio>
#include <cstring>

namespace ply {

namespace {

// Whether this machine stores integers little end first, decided at run time
// rather than assumed.
//
// The header this writes SAYS binary_little_endian, so on a big-endian machine
// every value would have to be byte-swapped on the way out. No such machine is a
// target — x86 and every Apple and Windows ARM part are little-endian — but a
// format declaration that is silently false is the kind of thing that is found
// years later by someone whose file will not open, so it is checked and refused
// rather than assumed.
bool littleEndian() {
    const uint16_t one = 1;
    uint8_t b[2];
    std::memcpy(b, &one, 2);
    return b[0] == 1;
}

// One record: three doubles and three bytes, packed exactly as the header
// declares them. Built in a buffer rather than written field by field because a
// hundred million three-byte writes through stdio is minutes of system calls.
constexpr size_t kRecordBytes = 3 * sizeof(double) + 3;

} // namespace

bool writePoints(const std::string& path,
                 const std::vector<lod::StorePoint>& pts,
                 const double origin[3],
                 const Note& note,
                 std::string& err) {
    if (!littleEndian()) {
        err = "this build is big-endian and the PLY writer declares little-endian records";
        return false;
    }

    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { err = "cannot create " + path; return false; }

    auto fail = [&](const std::string& why) {
        std::fclose(f);
        std::remove(path.c_str());          // no half-written file left openable
        err = why;
        return false;
    };

    std::string head = "ply\nformat binary_little_endian 1.0\n";
    for (const std::string& c : note.comments) {
        // A newline inside a comment would end the record and corrupt the header,
        // so it is flattened rather than trusted.
        std::string line = c;
        for (char& ch : line) if (ch == '\n' || ch == '\r') ch = ' ';
        head += "comment " + line + "\n";
    }
    head += "element vertex " + std::to_string(pts.size()) + "\n";
    head += "property double x\nproperty double y\nproperty double z\n";
    head += "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    head += "end_header\n";
    if (std::fwrite(head.data(), 1, head.size(), f) != head.size())
        return fail("short write on the header of " + path);

    // In blocks, so the cost is a few thousand writes rather than one per point.
    constexpr size_t kBlockPoints = 1 << 14;
    std::vector<uint8_t> buf(kBlockPoints * kRecordBytes);
    size_t used = 0;
    for (const lod::StorePoint& p : pts) {
        const double xyz[3] = {origin[0] + double(p.x),
                               origin[1] + double(p.y),
                               origin[2] + double(p.z)};
        uint8_t* rec = buf.data() + used;
        std::memcpy(rec, xyz, sizeof(xyz));
        rec[24] = p.r; rec[25] = p.g; rec[26] = p.b;
        used += kRecordBytes;
        if (used == buf.size()) {
            if (std::fwrite(buf.data(), 1, used, f) != used)
                return fail("short write on " + path);
            used = 0;
        }
    }
    if (used && std::fwrite(buf.data(), 1, used, f) != used)
        return fail("short write on " + path);

    if (std::fclose(f) != 0) {
        std::remove(path.c_str());
        err = "could not close " + path + " — the file may be incomplete";
        return false;
    }
    return true;
}

} // namespace ply
