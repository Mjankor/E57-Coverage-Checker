// Tests for the shrinkwrap — the region the question is actually asked about.
//
// The properties that matter, in the order the answer depends on them:
//
//   1. The dilation is Euclidean and exact. This is the one piece of real
//      arithmetic in the file and the one that already failed: a distance
//      transform seeded with infinities produces infinity minus infinity, and a
//      NaN loses the comparison that ends the scan. Checked against brute force
//      rather than against itself, because a transform that is consistently
//      wrong is exactly what brute force catches.
//   2. testBox agrees with contains. The carve settles whole bricks on testBox
//      and only asks about single voxels where the answer was Partial, so a
//      testBox that says Full over a cell that contains() would refuse is a
//      silent wrong answer over five hundred voxels at a time.
//   3. Interior-only drops the outside and keeps what is enclosed. That is the
//      whole of the difference between the two kinds of survey.
//   4. Narrowing never changes an answer, only removes questions.

#include "../src/wrap.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
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

// A source that hands back a fixed list of world points, so a test can put
// occupancy exactly where it wants it and still go through the real marking
// path. One "row" per point, one column.
struct PointList {
    std::vector<double> xyz;
    static bool at(const void* user, uint32_t row, uint32_t col, double out[3]) {
        (void)col;
        const PointList& p = *static_cast<const PointList*>(user);
        if (size_t(row) * 3 + 2 >= p.xyz.size()) return false;
        out[0] = p.xyz[size_t(row) * 3 + 0];
        out[1] = p.xyz[size_t(row) * 3 + 1];
        out[2] = p.xyz[size_t(row) * 3 + 2];
        return true;
    }
};

static wrap::MarkSource sourceFor(const PointList& pl) {
    wrap::MarkSource src;
    src.rows = uint32_t(pl.xyz.size() / 3);
    src.cols = 1;
    src.user = &pl;
    src.pointAt = &PointList::at;
    // No angular spread: stride one, every point marked.
    src.angularStep = 0.0;
    src.furthest = 0.0;
    return src;
}

// Which cells hold a point, worked out independently of the grid's own
// arithmetic, so the brute force below is not checking the code against itself.
static std::vector<std::array<int64_t, 3>> occupiedCells(const PointList& pl,
                                                         const wrap::Grid& g) {
    std::vector<std::array<int64_t, 3>> out;
    for (size_t i = 0; i + 2 < pl.xyz.size(); i += 3) {
        std::array<int64_t, 3> c{};
        for (int k = 0; k < 3; ++k)
            c[k] = int64_t(std::floor(pl.xyz[i + size_t(k)] / g.cell));
        bool seen = false;
        for (const auto& e : out) if (e == c) { seen = true; break; }
        if (!seen) out.push_back(c);
    }
    return out;
}

// ---------------------------------------------------------------------------

static void testDilationIsExactlyEuclidean() {
    std::printf("the dilation is Euclidean, and exact\n");

    // Three separated blobs, so the lower envelope has several parabolas to
    // choose between rather than trivially one.
    PointList pl;
    const double pts[3][3] = {{0.2, 0.1, 0.3}, {4.6, 2.2, 0.4}, {-3.1, 3.9, 1.1}};
    for (auto& p : pts) for (int k = 0; k < 3; ++k) pl.xyz.push_back(p[k]);

    const double lo[3] = {-4.0, -1.0, -1.0}, hi[3] = {6.0, 5.0, 2.0};
    for (double buffer : {0.75, 1.5, 3.0}) {
        wrap::Options opt;
        opt.buffer = buffer;
        opt.cell   = 0.25;                 // fixed, so the brute force is comparable
        wrap::Grid g;
        std::string err;
        CHECK(wrap::size(lo, hi, opt, g, err), err.empty() ? "sized" : err.c_str());
        wrap::markScan(sourceFor(pl), g);
        wrap::build(opt, g);

        const auto occ = occupiedCells(pl, g);
        CHECK(g.occupiedCells == occ.size(), "the marked cells are the cells with points");

        // Every cell, against the distance to the nearest occupied cell centre.
        uint64_t wrong = 0, inside = 0;
        for (uint32_t z = 0; z < g.dim[2]; ++z)
            for (uint32_t y = 0; y < g.dim[1]; ++y)
                for (uint32_t x = 0; x < g.dim[0]; ++x) {
                    const double c[3] = {
                        (double(int64_t(x) + g.lo[0]) + 0.5) * g.cell,
                        (double(int64_t(y) + g.lo[1]) + 0.5) * g.cell,
                        (double(int64_t(z) + g.lo[2]) + 0.5) * g.cell};
                    double best = 1e30;
                    for (const auto& o : occ) {
                        double d2 = 0;
                        for (int k = 0; k < 3; ++k) {
                            const double oc = (double(o[size_t(k)]) + 0.5) * g.cell;
                            d2 += (c[k] - oc) * (c[k] - oc);
                        }
                        best = std::min(best, d2);
                    }
                    // The transform works in whole cells, so the comparison is
                    // too: a radius of `buffer` is `buffer / cell` cells, and the
                    // brute force has to be asked the same question.
                    const double cells = std::sqrt(best) / g.cell;
                    const bool want = cells <= buffer / g.cell + 1e-9;
                    const bool got  = g.contains(c[0], c[1], c[2]);
                    if (want != got) ++wrong;
                    if (want) ++inside;
                }
        CHECK(wrong == 0, "every cell agrees with the brute-force distance");
        CHECK(inside > 8, "and the buffer actually reached something");
        CHECK(g.domainCells == inside, "the domain count matches what was counted");
    }
}

static void testOutsideTheGridIsOutsideTheQuestion() {
    std::printf("beyond the grid is beyond the question\n");

    PointList pl{{0, 0, 0}};
    const double lo[3] = {-1, -1, -1}, hi[3] = {1, 1, 1};
    wrap::Options opt;
    opt.buffer = 1.0;
    opt.cell   = 0.25;
    wrap::Grid g;
    std::string err;
    CHECK(wrap::size(lo, hi, opt, g, err), "sized");
    wrap::markScan(sourceFor(pl), g);
    wrap::build(opt, g);

    CHECK(g.contains(0, 0, 0), "the marked cell is in");
    CHECK(!g.contains(500, 0, 0), "a position far outside the grid is out");
    CHECK(!g.contains(0, -500, 0), "in every direction");
    // The grid is padded past the buffer, so its own boundary is genuinely
    // outside — which is what lets the interior flood start there.
    const double edge = (double(g.lo[0]) + 0.5) * g.cell;
    CHECK(!g.contains(edge, 0, 0), "and the grid's own edge cell is not in the wrap");
}

static void testBoxAgreesWithContains() {
    std::printf("testBox agrees with contains\n");

    PointList pl;
    for (int i = 0; i < 12; ++i) {
        const double t = 0.5 * i;
        pl.xyz.push_back(std::cos(t) * 2.0);
        pl.xyz.push_back(std::sin(t) * 2.0);
        pl.xyz.push_back(0.3 * i - 1.5);
    }
    const double lo[3] = {-4, -4, -3}, hi[3] = {4, 4, 3};
    wrap::Options opt;
    opt.buffer = 1.0;
    opt.cell   = 0.25;
    wrap::Grid g;
    std::string err;
    CHECK(wrap::size(lo, hi, opt, g, err), "sized");
    wrap::markScan(sourceFor(pl), g);
    wrap::build(opt, g);

    uint64_t full = 0, none = 0, partial = 0, wrong = 0;
    // Boxes of a few sizes at a few places, including ones that straddle the
    // grid's own edge — where Full has to be refused however the cells came out,
    // because outside the grid is outside the wrap.
    for (double s : {0.2, 0.6, 1.7}) {
        for (double x = -5.0; x < 5.0; x += 0.37) {
            for (double y = -5.0; y < 5.0; y += 0.53) {
                const double blo[3] = {x, y, -0.4}, bhi[3] = {x + s, y + s, 0.4};
                const int verdict = g.testBox(blo, bhi);
                // Sample the box densely and see what contains() says.
                bool anyIn = false, allIn = true;
                for (double sx = blo[0]; sx <= bhi[0] + 1e-9; sx += 0.1)
                    for (double sy = blo[1]; sy <= bhi[1] + 1e-9; sy += 0.1)
                        for (double sz = blo[2]; sz <= bhi[2] + 1e-9; sz += 0.1) {
                            if (g.contains(sx, sy, sz)) anyIn = true;
                            else                        allIn = false;
                        }
                if (verdict == 0) { ++none; if (anyIn) ++wrong; }
                else if (verdict == 2) { ++full; if (!allIn) ++wrong; }
                else { ++partial; }
            }
        }
    }
    CHECK(wrong == 0, "None never hides an included point, and Full never hides an excluded one");
    CHECK(none > 0 && full > 0 && partial > 0,
          "and all three verdicts occurred, so this is not a vacuous pass");
}

static void testInteriorOnlyDropsTheOutside() {
    std::printf("interior only: the outside goes, what is enclosed stays\n");

    // A hollow box of surface, as an interior survey leaves behind, with a small
    // sealed pocket inside it — a cupboard nobody opened.
    PointList pl;
    // Sampled finer than a cell, which is what makes the marked set
    // 26-connected and so watertight against a 6-connected flood. markScan's
    // stride guarantees the same thing on a real raster.
    const double half = 6.0, step = 0.2;
    for (double a = -half; a <= half + 1e-9; a += step)
        for (double b = -half; b <= half + 1e-9; b += step) {
            const double faces[6][3] = {{-half, a, b}, {half, a, b},
                                        {a, -half, b}, {a, half, b},
                                        {a, b, -half}, {a, b, half}};
            for (auto& f : faces) for (int k = 0; k < 3; ++k) pl.xyz.push_back(f[k]);
        }
    // The pocket: a smaller closed shell, well clear of the outer one.
    const double ph = 1.6;
    for (double a = -ph; a <= ph + 1e-9; a += step)
        for (double b = -ph; b <= ph + 1e-9; b += step) {
            const double faces[6][3] = {{-ph, a, b}, {ph, a, b},
                                        {a, -ph, b}, {a, ph, b},
                                        {a, b, -ph}, {a, b, ph}};
            for (auto& f : faces) for (int k = 0; k < 3; ++k) pl.xyz.push_back(f[k]);
        }

    const double lo[3] = {-half, -half, -half}, hi[3] = {half, half, half};
    wrap::Options both;
    both.buffer = 1.0;
    both.cell   = 0.25;
    wrap::Options inside = both;
    inside.interiorOnly = true;

    wrap::Grid gBoth, gIn;
    std::string err;
    CHECK(wrap::size(lo, hi, both, gBoth, err), "sized");
    CHECK(wrap::size(lo, hi, inside, gIn, err), "sized again");
    wrap::markScan(sourceFor(pl), gBoth);
    wrap::markScan(sourceFor(pl), gIn);
    wrap::build(both, gBoth);
    wrap::build(inside, gIn);

    CHECK(gBoth.occupiedCells == gIn.occupiedCells, "both wrapped the same returns");
    CHECK(gIn.domainCells < gBoth.domainCells, "the interior rule removed something");
    CHECK(gIn.droppedOutside > 0, "and says how much");
    CHECK(!gIn.sealLeaked, "and the shell held");

    // Just outside the wall: in the question for an outward survey, out of it for
    // an interior one. That is the whole of the switch.
    const double justOut = half + 0.6;
    CHECK(gBoth.contains(0, 0, justOut), "outward: the space past the wall is asked about");
    CHECK(!gIn.contains(0, 0, justOut), "interior: it is not");
    // Just inside: in the question either way.
    const double justIn = half - 0.6;
    CHECK(gBoth.contains(0, 0, justIn), "outward: the space inside the wall is asked about");
    CHECK(gIn.contains(0, 0, justIn), "interior: so is it");
    // The sealed pocket, which is what separates this from simply clipping to a
    // box: it is enclosed by observed surface, so it survives the flood even
    // though the flood surrounds it on every side.
    CHECK(gIn.contains(0, 0, ph - 0.6), "the sealed pocket's own inside is kept");
    CHECK(gBoth.contains(0, 0, ph - 0.6), "as it is for an outward survey");
    // Its middle is not, and for a reason that has nothing to do with the flood:
    // it is further from any measured surface than the buffer reaches. That is
    // the wrap's real limit, and it is the same limit in both modes — unobserved
    // space is only asked about near something that was seen.
    CHECK(!gBoth.contains(0, 0, 0), "the pocket's middle is beyond the buffer");
    CHECK(!gIn.contains(0, 0, 0), "in either mode");

    // Nothing the interior rule keeps was outside the outward wrap: narrowing
    // removes questions, it does not invent them.
    uint64_t invented = 0;
    for (uint32_t z = 0; z < gIn.dim[2]; ++z)
        for (uint32_t y = 0; y < gIn.dim[1]; ++y)
            for (uint32_t x = 0; x < gIn.dim[0]; ++x) {
                const double c[3] = {(double(int64_t(x) + gIn.lo[0]) + 0.5) * gIn.cell,
                                     (double(int64_t(y) + gIn.lo[1]) + 0.5) * gIn.cell,
                                     (double(int64_t(z) + gIn.lo[2]) + 0.5) * gIn.cell};
                if (gIn.contains(c[0], c[1], c[2]) && !gBoth.contains(c[0], c[1], c[2]))
                    ++invented;
            }
    CHECK(invented == 0, "the interior wrap is a subset of the outward one");
}

// A hole in the building's ENVELOPE — an external door left open, a window that
// returned nothing, a stretch of wall the survey never reached. Without a seal
// the flood walks through it and the interior rule deletes the building; with
// one it closes, and the outside label is then grown back to the wall so no skin
// of unobserved cells survives outside it.
//
// Not a doorway between two scanned rooms: the flood starts outside, so it never
// reaches one of those, and nothing is sealed on their account.
//
// Both halves are checked, because each fails in a way the other hides: a seal
// that does not close leaks, and a seal that closes but is not grown back leaves
// exactly the shell round the building that this mode exists to remove.
static void testASealedDoorwayAndNoSkinOutside() {
    std::printf("interior only: a doorway is closed, and no skin is left outside\n");

    // A closed box with a gap in one face — a doorway a metre wide.
    PointList pl;
    const double half = 6.0, step = 0.2;
    auto inDoor = [&](double a, double b) {
        return std::fabs(a) < 0.5 && b > -half && b < -half + 2.0;
    };
    for (double a = -half; a <= half + 1e-9; a += step)
        for (double b = -half; b <= half + 1e-9; b += step) {
            const double faces[6][3] = {{-half, a, b}, {half, a, b},
                                        {a, -half, b}, {a, half, b},
                                        {a, b, -half}, {a, b, half}};
            for (int f = 0; f < 6; ++f) {
                if (f == 1 && inDoor(a, b)) continue;   // the doorway, in +x
                for (int k = 0; k < 3; ++k) pl.xyz.push_back(faces[f][k]);
            }
        }

    const double lo[3] = {-half, -half, -half}, hi[3] = {half, half, half};
    wrap::Options opt;
    opt.buffer = 1.0;
    opt.cell   = 0.25;
    opt.interiorOnly = true;
    // The instrument stood inside, which is what makes the leak test meaningful.
    const std::vector<double> setups{0.0, 0.0, -4.0};

    wrap::Grid sealed;
    std::string err;
    CHECK(wrap::size(lo, hi, opt, sealed, err), "sized");
    wrap::markScan(sourceFor(pl), sealed);
    wrap::build(opt, sealed, setups);
    CHECK(!sealed.sealLeaked, "the seal closed the doorway");
    CHECK(sealed.droppedOutside > 0, "and the outside was dropped");
    // Just inside the far wall — in the question, and it is the flood reaching
    // here that the seal prevents.
    CHECK(sealed.contains(0, 0, -half + 0.6), "the space inside the wall is in the question");

    // No skin outside. Every cell beyond the wall, right up against it, has to be
    // out — that band is the whole failure mode, and it is only a cell or two
    // thick, so a test that samples a metre away would miss it entirely.
    uint64_t skin = 0;
    for (double d = 0.05; d < 0.95; d += 0.05)
        for (double a = -3.0; a <= 3.0; a += 0.5)
            if (sealed.contains(-half - d, a, 0.0)) ++skin;   // the face without a doorway
    CHECK(skin == 0, "and not one cell of skin survives outside the wall");

    // Turn the seal off and the same survey leaks, which is what the seal is for.
    wrap::Options unsealed = opt;
    unsealed.seal = 1e-6;             // smaller than a cell: closes nothing
    wrap::Grid leaky;
    CHECK(wrap::size(lo, hi, unsealed, leaky, err), "sized again");
    wrap::markScan(sourceFor(pl), leaky);
    wrap::build(unsealed, leaky, setups);
    CHECK(leaky.sealLeaked, "without a seal the doorway lets the outside in");
    CHECK(leaky.droppedOutside == 0, "and nothing is dropped when it does");
    CHECK(leaky.domainCells > sealed.domainCells,
          "so the answer is the generous one, not the deleted one");
}

// What the seal does as it widens, which is not what it first looked like.
//
// The guess was that a wider seal keeps more, so erring generous is free. The
// measurement says otherwise and the reason matters: a seal too narrow to close
// the opening leaks, the leak is caught, and NOTHING is dropped — so a narrow
// seal gives the largest domain of all, by failing. Above the width that closes
// the opening the answer stops moving entirely, because the seal is a threshold
// on "is this a portal or a missing wall" rather than a dial on how much to
// remove.
//
// So the property to hold on to is not monotonicity, it is that the answer is
// stable once the seal is wide enough, and safe when it is not.
static void testTheSealIsAThresholdNotADial() {
    std::printf("the seal is a threshold, and a narrow one fails safe\n");

    // A box with a 1.6 m hole in the envelope, and an instrument inside it.
    PointList pl;
    const double half = 5.0, step = 0.2;
    for (double a = -half; a <= half + 1e-9; a += step)
        for (double b = -half; b <= half + 1e-9; b += step) {
            const double faces[6][3] = {{-half, a, b}, {half, a, b},
                                        {a, -half, b}, {a, half, b},
                                        {a, b, -half}, {a, b, half}};
            for (int f = 0; f < 6; ++f) {
                if (f == 1 && std::fabs(a) < 0.8 && std::fabs(b) < 0.8) continue;
                for (int k = 0; k < 3; ++k) pl.xyz.push_back(faces[f][k]);
            }
        }
    const double lo[3] = {-half, -half, -half}, hi[3] = {half, half, half};
    const std::vector<double> setups{0.0, 0.0, 0.0};

    wrap::Options base;
    base.buffer = 1.0;
    base.cell   = 0.25;
    base.interiorOnly = true;

    auto wrapAt = [&](double seal, bool interior) {
        wrap::Options o = base;
        o.seal = seal;
        o.interiorOnly = interior;
        wrap::Grid g;
        std::string err;
        CHECK(wrap::size(lo, hi, o, g, err), "sized");
        wrap::markScan(sourceFor(pl), g);
        wrap::build(o, g, setups);
        return g;
    };

    const wrap::Grid outward = wrapAt(1.0, false);
    // Too narrow to close a 1.6 m hole: it leaks, and failing safe means the
    // answer is the generous one rather than a deleted building.
    for (double seal : {0.25, 0.5}) {
        const wrap::Grid g = wrapAt(seal, true);
        CHECK(g.sealLeaked, "a seal narrower than the hole leaks");
        CHECK(g.droppedOutside == 0, "and drops nothing when it does");
        CHECK(g.domainCells == outward.domainCells,
              "leaving exactly the outward answer, not a smaller one");
    }
    // Wide enough, and every wider seal agrees with it to the cell. The seal is
    // deciding what counts as a portal, not how much to remove.
    const wrap::Grid first = wrapAt(1.0, true);
    CHECK(!first.sealLeaked, "a seal as wide as the hole closes it");
    CHECK(first.droppedOutside > 0, "and the outside is dropped");
    CHECK(first.domainCells < outward.domainCells, "so the domain is smaller than the outward one");
    for (double seal : {1.5, 2.0, 3.0}) {
        const wrap::Grid g = wrapAt(seal, true);
        CHECK(!g.sealLeaked, "a wider seal still closes it");
        CHECK(g.domainCells == first.domainCells,
              "and gives the same answer to the cell — the seal is a threshold");
        CHECK(g.droppedOutside == first.droppedOutside, "dropping exactly as much");
    }

    // The grid has to be padded past the seal as well as the buffer, or the
    // barrier reaches the boundary and the flood has nowhere to start — which
    // drops nothing at all, silently, having decided the whole site is inside.
    // A 3 m seal on a grid padded for a 1 m buffer did exactly that.
    const wrap::Grid wide = wrapAt(3.0, true);
    CHECK(wide.droppedOutside > 0, "a seal wider than the buffer still leaves the flood a seed");
}

static void testBudgetCoarsensAndThenRefuses() {
    std::printf("the cell budget coarsens, and says when it cannot\n");

    PointList pl{{0, 0, 0}};
    const double lo[3] = {0, 0, 0}, hi[3] = {200, 200, 20};

    wrap::Options opt;
    opt.buffer   = 2.0;
    opt.cell     = 0.0;              // derived: 0.5 m
    opt.maxCells = 1ull << 30;       // generous
    wrap::Grid g;
    std::string err;
    CHECK(wrap::size(lo, hi, opt, g, err), "a roomy budget fits at the derived size");
    CHECK(!g.coarsened, "and does not coarsen");
    CHECK(std::fabs(g.cell - 0.5) < 1e-9, "which is a quarter of the buffer");

    opt.maxCells = 200000;           // too small for half-metre cells
    wrap::Grid g2;
    CHECK(wrap::size(lo, hi, opt, g2, err), "a tight budget still fits, coarser");
    CHECK(g2.coarsened, "and says it grew");
    CHECK(g2.cell > g.cell, "having actually grown");
    CHECK(g2.cellCount() <= opt.maxCells, "to inside the budget");

    // A cell coarser than the buffer is not a wrap, it is the grid's own shape.
    opt.maxCells = 64;
    wrap::Grid g3;
    CHECK(!wrap::size(lo, hi, opt, g3, err), "an impossible budget is refused");
    CHECK(!err.empty(), "with a reason");
}

static void testNoReturnsMeansNoWrap() {
    std::printf("a wrap with nothing in it stays empty\n");

    const double lo[3] = {-1, -1, -1}, hi[3] = {1, 1, 1};
    wrap::Options opt;
    opt.buffer = 1.0;
    opt.cell   = 0.25;
    wrap::Grid g;
    std::string err;
    CHECK(wrap::size(lo, hi, opt, g, err), "sized");
    wrap::build(opt, g);              // nothing marked
    CHECK(g.occupiedCells == 0, "nothing was marked");
    CHECK(g.domainCells == 0, "so nothing is in the question");
    // The caller falls back to the box; a wrap that quietly answered "everything"
    // would be a domain that had stopped narrowing without saying so.
    CHECK(!g.contains(0, 0, 0), "and the wrap does not claim the site");
}

int main() {
    std::printf("E57 Coverage Checker — shrinkwrap tests\n\n");
    testDilationIsExactlyEuclidean();
    testOutsideTheGridIsOutsideTheQuestion();
    testBoxAgreesWithContains();
    testInteriorOnlyDropsTheOutside();
    testASealedDoorwayAndNoSkinOutside();
    testTheSealIsAThresholdNotADial();
    testBudgetCoarsensAndThenRefuses();
    testNoReturnsMeansNoWrap();
    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
