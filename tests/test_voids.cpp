// Tests for the void classification — the stage that separates a coverage
// failure from the rest of the world.
//
// This is the stage the tool was missing, and its absence is what made every
// earlier result useless rather than wrong. The carve says "nothing observed
// this", and over any region containing a building that is true of most of it:
// inside the ground, inside the neighbours' houses, past the fence. The
// deliverable is the subset you cannot reach from outside without crossing
// space the scanners saw.
//
// What has to hold:
//
//   1. A sealed pocket surrounded by observed space is enclosed.
//   2. The same pocket with a single voxel of leak is not — because a leak means
//      you genuinely do not know whether that space connects to the street.
//   3. Everything outside the observed shell is exterior, however much of it
//      there is.
//   4. Voxels that were never in the domain are neither: they are not the
//      question, and they must not be counted as a finding.
//   5. Components are counted separately, so a hundred one-voxel pockets read
//      differently from one room-sized hole.

#include "../src/voids.h"

#include <cstdio>
#include <cstdlib>
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

// A grid where everything is in the domain and nothing has been observed yet.
static voids::Grid makeGrid(uint32_t n) {
    voids::Grid g;
    g.dim[0] = g.dim[1] = g.dim[2] = n;
    g.voxelSize = 0.1;
    g.state.assign(size_t(n) * n * n, carve::kReachable);
    return g;
}

// A hollow observed shell: the surface of a cube, marked as measured.
static void shell(voids::Grid& g, uint32_t lo, uint32_t hi, uint8_t bits) {
    for (uint32_t z = lo; z <= hi; ++z)
        for (uint32_t y = lo; y <= hi; ++y)
            for (uint32_t x = lo; x <= hi; ++x) {
                const bool onFace = (x == lo || x == hi || y == lo || y == hi ||
                                     z == lo || z == hi);
                if (onFace) g.state[g.index(x, y, z)] |= bits;
            }
}

// ---------------------------------------------------------------------------

static void testSealedPocketIsEnclosed() {
    std::printf("a sealed pocket is a void\n");

    voids::Grid g = makeGrid(20);
    shell(g, 5, 12, carve::kOccupied);          // a measured box, hollow inside

    const voids::Report rep = voids::classify(g);

    // Inside the shell is 6^3 = 216 voxels, none of them reachable from outside
    // without crossing the measured surface.
    CHECK(rep.enclosed == 216, "the interior of the shell is enclosed");
    CHECK(rep.components == 1, "and it is one void, not many");
    CHECK(rep.largestComponent == 216, "of the whole size");
    CHECK(rep.exterior > 0, "everything outside the shell reaches the boundary");
    CHECK(rep.exterior + rep.enclosed + 8u * 8u * 8u - 6u * 6u * 6u == 20u * 20u * 20u,
          "every voxel is exterior, enclosed, or part of the observed shell");

    // The classification has to be readable per voxel, not only in aggregate.
    CHECK(voids::isEnclosedVoid(g.state[g.index(8, 8, 8)]), "a voxel inside is a void");
    CHECK(!voids::isEnclosedVoid(g.state[g.index(1, 1, 1)]), "a voxel outside is not");
    CHECK(!voids::isEnclosedVoid(g.state[g.index(5, 8, 8)]), "nor is the shell itself");

    // The surface of the void is what gets drawn: the 6^3 interior has a 6^3
    // minus 4^3 shell touching the observed box.
    CHECK(rep.enclosedSurface == 216 - 64, "the void's own surface is counted");
}

static void testOneLeakIsEnough() {
    std::printf("a pocket with a leak is not a void\n");

    voids::Grid g = makeGrid(20);
    shell(g, 5, 12, carve::kOccupied);
    // Punch a single voxel out of one face. That is a hole in the coverage, and
    // through it the interior connects to the street — which is exactly the
    // situation where you cannot claim to have found a void.
    g.state[g.index(5, 8, 8)] = carve::kReachable;

    const voids::Report rep = voids::classify(g);
    CHECK(rep.enclosed == 0, "one voxel of leak un-encloses the whole pocket");
    CHECK(rep.components == 0, "and there is no void to report");
    CHECK(!voids::isEnclosedVoid(g.state[g.index(8, 8, 8)]),
          "the interior is now part of the outside world");
}

static void testVisibleSpaceSealsAsWellAsSurface() {
    std::printf("space seen through seals as well as surface measured\n");

    // A shell of visible voxels rather than occupied ones. Both are observation;
    // neither can be passed through by something arriving from outside without
    // the scanners having seen it happen.
    voids::Grid g = makeGrid(20);
    shell(g, 5, 12, carve::kVisible);

    const voids::Report rep = voids::classify(g);
    CHECK(rep.enclosed == 216, "a shell of line-of-sight encloses just as well");
    CHECK(rep.components == 1, "as one void");
}

static void testUnreachableVoxelsAreNotAFinding() {
    std::printf("space outside the domain is not a finding\n");

    voids::Grid g = makeGrid(20);
    shell(g, 5, 12, carve::kOccupied);
    // Half the interior was never in the domain at all — outside the surveyed
    // extent, or beyond every setup's range. It is unobserved, but it was never
    // the question, and counting it would inflate the answer with space nobody
    // asked about.
    for (uint32_t z = 6; z <= 11; ++z)
        for (uint32_t y = 6; y <= 11; ++y)
            for (uint32_t x = 6; x <= 8; ++x)
                g.state[g.index(x, y, z)] = 0;

    const voids::Report rep = voids::classify(g);
    CHECK(rep.enclosed == 216 / 2, "only the half that was in the domain counts");
    CHECK(!voids::isEnclosedVoid(g.state[g.index(7, 8, 8)]),
          "a voxel outside the domain is not a void");
    CHECK(voids::isEnclosedVoid(g.state[g.index(10, 8, 8)]),
          "one inside it still is");
}

static void testComponentsAreCountedSeparately() {
    std::printf("separate pockets are separate findings\n");

    voids::Grid g = makeGrid(30);
    shell(g, 2, 8, carve::kOccupied);
    shell(g, 12, 22, carve::kOccupied);
    shell(g, 25, 28, carve::kOccupied);

    const voids::Report rep = voids::classify(g);
    const uint64_t a = 5 * 5 * 5, b = 9 * 9 * 9, c = 2 * 2 * 2;
    CHECK(rep.components == 3, "three pockets, three findings");
    CHECK(rep.enclosed == a + b + c, "and their volumes add up");
    CHECK(rep.largestComponent == b, "the largest is reported");
}

static void testEmptyAndDegenerate() {
    std::printf("degenerate grids\n");

    voids::Grid empty;
    const voids::Report r0 = voids::classify(empty);
    CHECK(r0.enclosed == 0 && r0.components == 0, "an empty grid reports nothing");

    // Nothing observed anywhere: everything is the outside world, and there is
    // no such thing as a void. A site with no enclosing geometry must report no
    // findings rather than reporting itself.
    voids::Grid open_ = makeGrid(8);
    const voids::Report r1 = voids::classify(open_);
    CHECK(r1.enclosed == 0, "with nothing observed, nothing is enclosed");
    CHECK(r1.exterior == 8u * 8u * 8u, "it is all the outside world");

    // Everything observed: nothing unobserved to classify either way.
    voids::Grid full = makeGrid(8);
    for (uint8_t& b : full.state) b |= carve::kVisible;
    const voids::Report r2 = voids::classify(full);
    CHECK(r2.enclosed == 0 && r2.exterior == 0, "a fully observed grid has neither");
}

static void testFloodCrossesTileSeams() {
    std::printf("connectivity ignores where tiles were\n");

    // A corridor of unobserved space running the length of the grid, walled off
    // except at one end. It is not a void — but only a flood that crosses the
    // whole grid can tell, and a tile-local pass never could. This is why the
    // stage needs the domain in one piece.
    voids::Grid g = makeGrid(40);
    for (uint8_t& b : g.state) b |= carve::kOccupied;
    for (uint32_t x = 0; x < 39; ++x)                    // open at x = 0 only
        g.state[g.index(x, 20, 20)] = carve::kReachable;

    const voids::Report open_ = voids::classify(g);
    CHECK(open_.enclosed == 0, "a corridor open at one end is not a void");

    // Seal that one end and the whole corridor becomes one.
    voids::Grid h = makeGrid(40);
    for (uint8_t& b : h.state) b |= carve::kOccupied;
    for (uint32_t x = 1; x < 39; ++x)
        h.state[h.index(x, 20, 20)] = carve::kReachable;

    const voids::Report sealed = voids::classify(h);
    CHECK(sealed.enclosed == 38, "sealed, all 38 voxels of it are one void");
    CHECK(sealed.components == 1, "and one component, however long it is");
}

int main() {
    std::printf("E57 Coverage Checker — void classification tests\n\n");
    testSealedPocketIsEnclosed();
    testOneLeakIsEnough();
    testVisibleSpaceSealsAsWellAsSurface();
    testUnreachableVoxelsAreNotAFinding();
    testComponentsAreCountedSeparately();
    testEmptyAndDegenerate();
    testFloodCrossesTileSeams();
    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
