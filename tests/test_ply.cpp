// Tests for the point writer — the stage that gets the answer back out.
//
// What matters about a saved file is that something else can open it, so these
// read the bytes back the way another reader would: parse the header as text,
// then decode the records from their declared types and offsets. Nothing here
// calls the writer's own code to check the writer.

#include "../src/ply.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_failures = 0;
static int g_checks   = 0;

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        ++g_checks;                                                             \
        if (!(cond)) {                                                          \
            ++g_failures;                                                       \
            std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, (msg));       \
        }                                                                       \
    } while (0)

// The same convention the other suites use, and no POSIX: the core is meant to
// build anywhere and a test that needs unistd.h would be the first thing to say
// otherwise.
static std::string tmpPath(const char* stem) {
    const char* d = std::getenv("E57COV_TMPDIR");
    return std::string(d ? d : "/tmp") + "/e57cov_" + stem + ".ply";
}

static std::vector<uint8_t> readAll(const std::string& path) {
    std::vector<uint8_t> out;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return out;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (n > 0) {
        out.resize(size_t(n));
        if (std::fread(out.data(), 1, out.size(), f) != out.size()) out.clear();
    }
    std::fclose(f);
    return out;
}

// A saved cloud, read back the way any other tool would read it.
static void testAFileAnotherReaderCanOpen() {
    std::printf("a file another reader can open\n");

    // A site at UTM magnitudes, which is the case the double coordinates exist
    // for: these eastings need sixteen digits of mantissa to hold a millimetre,
    // and float32 has seven.
    const double origin[3] = {327451.125, 6251887.5, 42.25};
    std::vector<lod::StorePoint> pts(3);
    pts[0] = {0.0f,   0.0f,  0.0f, 255,  64,  96, 255, 0, 0};
    pts[1] = {1.5f,  -2.25f, 0.75f, 10, 200,  30, 255, 1, 0};
    pts[2] = {-0.125f, 8.0f, -3.5f,  1,   2,   3, 255, 2, 0};

    ply::Note note;
    note.comments.push_back("e57cov test");
    note.comments.push_back("a comment with\na newline in it");

    const std::string path = tmpPath("save");
    std::string err;
    CHECK(ply::writePoints(path, pts, origin, note, err), err.empty() ? "written" : err.c_str());

    const std::vector<uint8_t> bytes = readAll(path);
    CHECK(!bytes.empty(), "the file has contents");
    if (bytes.empty()) return;

    // The header, as text up to and including end_header.
    const std::string all(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    const size_t endPos = all.find("end_header\n");
    CHECK(endPos != std::string::npos, "the header is terminated");
    if (endPos == std::string::npos) return;
    const std::string head = all.substr(0, endPos + 11);

    CHECK(head.rfind("ply\n", 0) == 0, "it starts with the magic");
    CHECK(head.find("format binary_little_endian 1.0\n") != std::string::npos,
          "and declares the format it writes");
    CHECK(head.find("element vertex 3\n") != std::string::npos, "with the right count");
    CHECK(head.find("property double x\n") != std::string::npos,
          "and double coordinates, which is what a georeferenced site needs");
    CHECK(head.find("property uchar red\n") != std::string::npos, "and colour");
    CHECK(head.find("comment e57cov test\n") != std::string::npos, "the notes survive");
    // A newline inside a comment would end the record and leave a header line that
    // is not a PLY keyword, which readers reject outright.
    CHECK(head.find("comment a comment with a newline in it\n") != std::string::npos,
          "and a newline inside one is flattened rather than breaking the header");

    // The records, decoded from the declared layout.
    const size_t rec = 3 * sizeof(double) + 3;
    CHECK(bytes.size() == head.size() + 3 * rec, "the body is exactly three records");
    if (bytes.size() != head.size() + 3 * rec) return;

    for (int i = 0; i < 3; ++i) {
        const uint8_t* p = bytes.data() + head.size() + size_t(i) * rec;
        double xyz[3];
        std::memcpy(xyz, p, sizeof(xyz));
        // Absolute world position, to the millimetre, at UTM magnitudes — which is
        // the whole point of writing doubles.
        CHECK(std::fabs(xyz[0] - (origin[0] + double(pts[size_t(i)].x))) < 1e-6,
              "x is the origin plus the offset, absolutely");
        CHECK(std::fabs(xyz[1] - (origin[1] + double(pts[size_t(i)].y))) < 1e-6, "y likewise");
        CHECK(std::fabs(xyz[2] - (origin[2] + double(pts[size_t(i)].z))) < 1e-6, "z likewise");
        CHECK(p[24] == pts[size_t(i)].r && p[25] == pts[size_t(i)].g &&
              p[26] == pts[size_t(i)].b, "and the colour came through");
    }
    std::remove(path.c_str());
}

// A file that cannot be written must not be left half-written: a truncated PLY
// opens, shows a fraction of the points, and says nothing about being partial.
static void testAFailedSaveLeavesNoFile() {
    std::printf("a failed save leaves no file\n");

    std::vector<lod::StorePoint> pts(1);
    const double origin[3] = {0, 0, 0};
    ply::Note note;
    std::string err;
    const std::string bad = "/nonexistent-directory-for-e57cov/out.ply";
    CHECK(!ply::writePoints(bad, pts, origin, note, err), "a bad path fails");
    CHECK(!err.empty(), "with a reason");
    CHECK(readAll(bad).empty(), "and nothing is left behind");
}

// Zero points is a valid file, and saying so is better than refusing: a run that
// found nothing unobserved is a result worth keeping.
static void testAnEmptyCloudIsStillAFile() {
    std::printf("an empty cloud is still a file\n");

    const std::string path = tmpPath("empty");
    const double origin[3] = {1, 2, 3};
    ply::Note note;
    std::string err;
    CHECK(ply::writePoints(path, {}, origin, note, err), "written");
    const std::vector<uint8_t> bytes = readAll(path);
    const std::string all(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    CHECK(all.find("element vertex 0\n") != std::string::npos, "declaring no vertices");
    CHECK(all.find("end_header\n") != std::string::npos, "and a complete header");
    CHECK(bytes.size() == all.find("end_header\n") + 11, "with no body after it");
    std::remove(path.c_str());
}

int main() {
    std::printf("E57 Coverage Checker — point writer tests\n\n");
    testAFileAnotherReaderCanOpen();
    testAFailedSaveLeavesNoFile();
    testAnEmptyCloudIsStillAFile();
    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
