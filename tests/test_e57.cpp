// Round-trip tests for the E57 reader.
//
// The fixtures deliberately use awkward bit widths and small packets so that
// packed values straddle packet boundaries — the case that separates a correct
// decoder from one that re-aligns per packet and drifts silently.

#include "e57_fixture.h"
#include "../src/e57.h"

#include <cmath>
#include <cstdio>
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

#define CHECK_NEAR(a, b, tol, msg)                                              \
    do {                                                                        \
        ++g_checks;                                                             \
        double va = (a), vb = (b);                                              \
        if (!(std::fabs(va - vb) <= (tol))) {                                   \
            ++g_failures;                                                       \
            std::printf("  FAIL %s:%d  %s (%.17g vs %.17g)\n",                  \
                        __FILE__, __LINE__, (msg), va, vb);                     \
        }                                                                       \
    } while (0)

static std::string tmpPath(const char* name) {
    const char* dir = std::getenv("E57COV_TMPDIR");
    return std::string(dir ? dir : "/tmp") + "/e57cov_" + name + ".e57";
}

// Collects every decoded column into flat vectors.
static bool readAll(e57::Reader& r, size_t scan,
                    const std::vector<std::string>& fields,
                    std::vector<std::vector<double>>& out,
                    std::string& err) {
    out.assign(fields.size(), {});
    return r.readPoints(scan, fields, [&](const e57::PointBlock& b) {
        for (size_t k = 0; k < fields.size(); ++k)
            out[k].insert(out[k].end(), b.columns[k], b.columns[k] + b.count);
        return true;
    }, err);
}

// ---------------------------------------------------------------------------

static void testCrcAndBits() {
    std::printf("crc32c + bitsNeeded\n");
    const char* v = "123456789";
    CHECK(e57::crc32c(reinterpret_cast<const uint8_t*>(v), 9) == 0xE3069283u,
          "CRC-32c check vector");

    CHECK(e57::bitsNeeded(0, 0)   == 0,  "constant field needs 0 bits");
    CHECK(e57::bitsNeeded(0, 1)   == 1,  "0..1 needs 1 bit");
    CHECK(e57::bitsNeeded(0, 255) == 8,  "0..255 needs 8 bits");
    CHECK(e57::bitsNeeded(0, 256) == 9,  "0..256 needs 9 bits");
    CHECK(e57::bitsNeeded(-100, 100) == 8, "-100..100 needs 8 bits");
    CHECK(e57::bitsNeeded(std::numeric_limits<int64_t>::min(),
                          std::numeric_limits<int64_t>::max()) == 64,
          "full int64 span needs 64 bits");
}

static void testXml() {
    std::printf("xml parser\n");
    const char* doc =
        "<?xml version=\"1.0\"?>\n"
        "<!-- a comment -->\n"
        "<root a=\"1\" b='two &amp; more'>\n"
        "  <child type=\"Float\">3.5</child>\n"
        "  <name><![CDATA[has <angle> brackets]]></name>\n"
        "  <empty/>\n"
        "</root>";
    e57::XmlNode root;
    std::string err;
    CHECK(e57::parseXml(doc, std::strlen(doc), root, err), "parses");
    CHECK(root.name == "root", "root name");
    CHECK(root.attr("a") == "1", "numeric attribute");
    CHECK(root.attr("b") == "two & more", "entity in attribute");
    CHECK(root.child("child") != nullptr, "child present");
    CHECK_NEAR(root.child("child")->asDouble(), 3.5, 1e-12, "child text as double");
    CHECK(root.child("name")->text == "has <angle> brackets", "CDATA passthrough");
    CHECK(root.child("empty") != nullptr, "self-closing element");
    CHECK(root.child("missing") == nullptr, "absent child is null");

    e57::XmlNode bad;
    CHECK(!e57::parseXml("<a></b>", 7, bad, err), "mismatched end tag rejected");
}

// The core case: 19-bit packed values, 7 records per packet. 19 and 7 are
// mutually awkward, so most packets end mid-value.
static void testPackedStraddlingPackets() {
    std::printf("packed ScaledInteger straddling packet boundaries\n");
    fixture::Scan s;
    s.name = "packed";
    s.fields = {
        {"cartesianX", e57::FieldType::ScaledInteger, -262144, 262143, 0.0001, 0.0},
        {"cartesianY", e57::FieldType::ScaledInteger, -262144, 262143, 0.0001, 0.0},
        {"intensity",  e57::FieldType::Integer,        0,      2047,   1.0,    0.0},
    };
    // Guard the premise of this test: if the bit width and packet size ever
    // become mutually aligned, every packet would end on a byte boundary and
    // the test would keep passing while no longer exercising the case it
    // exists for.
    const int  bits            = e57::bitsNeeded(-262144, 262143);
    const size_t recsPerPacket = 7;
    CHECK(bits == 19, "19-bit packed field");
    CHECK((bits * recsPerPacket) % 8 != 0, "packets genuinely end mid-value");

    const size_t N = 977;   // coprime with the packet size
    s.data.assign(3, {});
    for (size_t i = 0; i < N; ++i) {
        s.data[0].push_back(static_cast<double>(int64_t(i) * 37 - 100000));
        s.data[1].push_back(static_cast<double>(-int64_t(i) * 11 + 5000));
        s.data[2].push_back(static_cast<double>(i % 2048));
    }
    const std::string p = tmpPath("packed");
    CHECK(fixture::write(p, {s}, recsPerPacket), "fixture written");

    e57::Reader r;
    std::string err;
    CHECK(r.open(p, err), err.empty() ? "opened" : err.c_str());
    CHECK(r.verifyCrc(err), err.empty() ? "CRC verified" : err.c_str());
    CHECK(r.scanCount() == 1, "one scan");
    CHECK(r.scan(0).recordCount == N, "record count");

    std::vector<std::vector<double>> got;
    CHECK(readAll(r, 0, {"cartesianX", "cartesianY", "intensity"}, got, err),
          err.empty() ? "read all fields" : err.c_str());
    CHECK(got[0].size() == N, "decoded every record");

    bool allOk = true;
    for (size_t i = 0; i < N && allOk; ++i) {
        if (std::fabs(got[0][i] - s.data[0][i] * 0.0001) > 1e-9) allOk = false;
        if (std::fabs(got[1][i] - s.data[1][i] * 0.0001) > 1e-9) allOk = false;
        if (std::fabs(got[2][i] - s.data[2][i])          > 1e-9) allOk = false;
    }
    CHECK(allOk, "every packed value round-trips exactly");
}

static void testMixedTypesAndFieldSkipping() {
    std::printf("mixed field types + selective decode\n");
    fixture::Scan s;
    s.name = "mixed";
    s.fields = {
        {"cartesianX",          e57::FieldType::FloatDouble},
        {"cartesianY",          e57::FieldType::FloatDouble},
        {"cartesianZ",          e57::FieldType::FloatDouble},
        {"intensity",           e57::FieldType::FloatSingle},
        {"cartesianInvalidState", e57::FieldType::Integer, 0, 2},
        // Constant field: minimum == maximum, so the encoder emits no bits at
        // all for it and the decoder must not wait on it.
        {"rowIndex",            e57::FieldType::Integer, 7, 7},
    };
    const size_t N = 1500;
    s.data.assign(6, {});
    for (size_t i = 0; i < N; ++i) {
        s.data[0].push_back(0.001 * double(i));
        s.data[1].push_back(-2.5   * double(i));
        s.data[2].push_back(1e5 + double(i));      // large magnitude: needs float64
        s.data[3].push_back(float(i % 100) * 0.5f);
        s.data[4].push_back(double(i % 3));
        s.data[5].push_back(7.0);
    }
    const std::string p = tmpPath("mixed");
    CHECK(fixture::write(p, {s}, 33), "fixture written");

    e57::Reader r;
    std::string err;
    CHECK(r.open(p, err), err.empty() ? "opened" : err.c_str());

    // Decode a subset only; the skipped streams must be stepped over cleanly.
    std::vector<std::vector<double>> got;
    CHECK(readAll(r, 0, {"cartesianZ", "cartesianInvalidState"}, got, err),
          err.empty() ? "read subset" : err.c_str());
    CHECK(got[0].size() == N, "subset decoded every record");

    bool zOk = true, invOk = true;
    for (size_t i = 0; i < N; ++i) {
        if (std::fabs(got[0][i] - (1e5 + double(i))) > 1e-9) zOk = false;
        if (std::fabs(got[1][i] - double(i % 3))     > 1e-9) invOk = false;
    }
    CHECK(zOk,   "float64 survives large magnitudes");
    CHECK(invOk, "invalid-state field decodes past the skipped streams");

    // A field that is constant across the whole scan.
    std::vector<std::vector<double>> constOnly;
    CHECK(readAll(r, 0, {"rowIndex"}, constOnly, err),
          err.empty() ? "read constant-only field" : err.c_str());
    CHECK(constOnly[0].size() == N, "constant field yields every record");
    bool cOk = true;
    for (double v : constOnly[0]) if (std::fabs(v - 7.0) > 1e-12) cOk = false;
    CHECK(cOk, "constant field decodes to its minimum");

    std::vector<std::vector<double>> none;
    CHECK(!readAll(r, 0, {"noSuchField"}, none, err), "missing field is an error");
}

// A scan large enough to span many 1020-byte pages, exercising the logical /
// physical mapping past the first page boundary — where a reader that conflates
// the two address spaces first goes wrong.
static void testMultiPageAndPose() {
    std::printf("multi-page addressing + pose\n");
    fixture::Scan s;
    s.name    = "big";
    s.hasPose = true;
    s.q[0] = 0.7071067811865476; s.q[1] = 0.0; s.q[2] = 0.0; s.q[3] = 0.7071067811865476;
    s.t[0] = 123.456; s.t[1] = -78.9; s.t[2] = 2.5;
    s.fields = {
        {"sphericalRange",     e57::FieldType::ScaledInteger, 0, 6553500, 0.0001, 0.0},
        {"sphericalAzimuth",   e57::FieldType::ScaledInteger, -3141593, 3141593, 1e-6, 0.0},
        {"sphericalElevation", e57::FieldType::ScaledInteger, -1570796, 1570796, 1e-6, 0.0},
    };
    const size_t N = 50000;   // ~ hundreds of pages
    s.data.assign(3, {});
    for (size_t i = 0; i < N; ++i) {
        s.data[0].push_back(double((i * 6151) % 6553500));
        s.data[1].push_back(double(int64_t((i * 977) % 6283186) - 3141593));
        s.data[2].push_back(double(int64_t((i * 613) % 3141592) - 1570796));
    }
    const std::string p = tmpPath("big");
    CHECK(fixture::write(p, {s}, 512), "fixture written");

    e57::Reader r;
    std::string err;
    CHECK(r.open(p, err), err.empty() ? "opened" : err.c_str());
    CHECK(r.verifyCrc(err), err.empty() ? "CRC verified across all pages" : err.c_str());

    const e57::Scan& sc = r.scan(0);
    CHECK(sc.hasPose, "pose present");
    CHECK_NEAR(sc.pose.q[0], 0.7071067811865476, 1e-15, "quaternion w");
    CHECK_NEAR(sc.pose.q[3], 0.7071067811865476, 1e-15, "quaternion z");
    CHECK_NEAR(sc.pose.t[0], 123.456, 1e-12, "translation x");
    CHECK_NEAR(sc.pose.t[1], -78.9,   1e-12, "translation y");

    std::vector<std::vector<double>> got;
    CHECK(readAll(r, 0, {"sphericalRange", "sphericalAzimuth", "sphericalElevation"}, got, err),
          err.empty() ? "read across page boundaries" : err.c_str());
    CHECK(got[0].size() == N, "decoded every record");

    bool ok = true;
    for (size_t i = 0; i < N && ok; ++i) {
        if (std::fabs(got[0][i] - s.data[0][i] * 0.0001) > 1e-9) ok = false;
        if (std::fabs(got[1][i] - s.data[1][i] * 1e-6)   > 1e-11) ok = false;
        if (std::fabs(got[2][i] - s.data[2][i] * 1e-6)   > 1e-11) ok = false;
    }
    CHECK(ok, "all values correct across hundreds of pages");
}

static void testMultipleScans() {
    std::printf("multiple scans per file\n");
    std::vector<fixture::Scan> scans;
    for (int k = 0; k < 4; ++k) {
        fixture::Scan s;
        s.name = "setup" + std::to_string(k);
        s.hasPose = true;
        s.t[0] = double(k) * 10.0;
        s.fields = {{"cartesianX", e57::FieldType::ScaledInteger, -100000, 100000, 0.001, 0.0}};
        s.data.assign(1, {});
        for (int i = 0; i < 300 + k * 51; ++i) s.data[0].push_back(double(i * (k + 1)));
        scans.push_back(s);
    }
    const std::string p = tmpPath("multi");
    CHECK(fixture::write(p, scans, 17), "fixture written");

    e57::Reader r;
    std::string err;
    CHECK(r.open(p, err), err.empty() ? "opened" : err.c_str());
    CHECK(r.scanCount() == 4, "four scans");

    bool ok = true;
    for (size_t k = 0; k < 4; ++k) {
        CHECK(r.scan(k).name == "setup" + std::to_string(k), "scan name");
        CHECK_NEAR(r.scan(k).pose.t[0], double(k) * 10.0, 1e-12, "per-scan pose");
        std::vector<std::vector<double>> got;
        if (!readAll(r, k, {"cartesianX"}, got, err)) { ok = false; break; }
        if (got[0].size() != scans[k].data[0].size()) { ok = false; break; }
        for (size_t i = 0; i < got[0].size(); ++i)
            if (std::fabs(got[0][i] - scans[k].data[0][i] * 0.001) > 1e-9) { ok = false; break; }
    }
    CHECK(ok, "every scan's section decodes independently");
}

static void testRejectsBadFiles() {
    std::printf("malformed input\n");
    const std::string p = tmpPath("bad");
    FILE* fp = std::fopen(p.c_str(), "wb");
    std::vector<uint8_t> junk(4096, 0xAB);
    std::fwrite(junk.data(), 1, junk.size(), fp);
    std::fclose(fp);

    e57::Reader r;
    std::string err;
    CHECK(!r.open(p, err), "junk file rejected");
    CHECK(err.find("signature") != std::string::npos, "reports a bad signature");

    e57::Reader missing;
    CHECK(!missing.open(tmpPath("does_not_exist"), err), "missing file rejected");
}

int main() {
    std::printf("E57 Coverage Checker — reader tests\n\n");
    testCrcAndBits();
    testXml();
    testPackedStraddlingPackets();
    testMixedTypesAndFieldSkipping();
    testMultiPageAndPose();
    testMultipleScans();
    testRejectsBadFiles();

    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
