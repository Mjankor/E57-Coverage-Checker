#include "wrap.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace wrap {

namespace {

// The state a cell is in while the domain is being built. Packed into the same
// byte the finished answer lives in, so there is one array rather than three.
constexpr uint8_t kOccupied = 1u << 0;   // holds at least one return
constexpr uint8_t kInDomain = 1u << 1;   // within the buffer of an occupied cell
constexpr uint8_t kSeed     = 1u << 2;   // scratch: what the skin is measured from
constexpr uint8_t kOutside  = 1u << 3;   // the flood reached it
constexpr uint8_t kBarrier  = 1u << 4;   // sealed: what the flood cannot cross
// The watertight surface of the surveyed shell: the measured returns, plus the
// cells that close the openings between them. See the envelope note in build().
constexpr uint8_t kEnvelope = 1u << 5;
// A region the survey encloses that is deep enough to pull the boundary into, and
// the whole of the enclosed region it belongs to. See the per-region note in
// build(): the negative buffer is decided one enclosed region at a time.
constexpr uint8_t kCore     = 1u << 6;
constexpr uint8_t kKeptIn   = 1u << 7;

// WHAT STOPS THE INTERIOR-ONLY FLOOD, and how the outside is then peeled back.
//
// The barrier is the occupancy dilated by a small seal, and the seal is about
// the building's ENVELOPE, not about doorways inside it.
//
// A doorway between two scanned rooms is not a problem and is not sealed for its
// own sake: the flood starts outside the building, so to reach an interior
// doorway it would have to be inside already. Interior openings connect surveyed
// air to surveyed air, and the flood never gets to either side of them.
//
// What does let the flood in is a hole in the envelope: an external door left
// open, a window whose glass returned nothing, a stretch of wall the survey did
// not reach. Those connect the outside to the inside, and topology alone cannot
// then tell one from the other — which is what the seal is for. It is the
// statement that an opening narrower than a given width is a portal rather than
// the absence of a wall.
//
// The seal narrows nothing by itself. It is the flood's barrier and nothing
// else: the domain is still everything within the buffer of a return, so the air
// in a doorway — sealed or not — stays in the question because it is next to the
// frame. Sealing an interior doorway splits two rooms' air into separate
// components and keeps both, neither having been reached from outside.
//
// What it does decide is whether the interior rule runs at all, and that is not
// a dial. Below the width that closes the envelope's openings the flood gets in,
// the leak is caught, and nothing is dropped; above it the answer stops moving —
// measured, identical to the cell at every seal from one to three times the
// width that first closed the hole. So the seal is a threshold on what counts as
// a portal rather than a knob on how much to remove, and erring wide costs a
// lobe of outside air near a genuine opening and nothing else.
//
// What makes the raw occupancy watertight in the first place is digital topology
// rather than morphology: a 6-connected flood is blocked by a 26-connected
// barrier, and a surface marked at every cell it passes through gives one. That
// is why markScan samples the raster exactly rather than at a stride — the seal
// is for holes in the SURVEY, and it should not be quietly covering for holes in
// the sampling as well.
//
// A dilated barrier on its own then has the failure the whole switch exists to
// avoid: the flood stops a seal's width short of the wall, so a skin of
// unobserved cells survives on the OUTSIDE of the building — exactly the closed
// shell round it that interior-only mode is there to remove. Eroding the barrier
// back does not rescue it, because the outermost layer of a dilated surface is a
// union of spheres rather than a solid slab: eroding by what was dilated eats the
// very layer that did the bridging, and the flood goes through the holes it was
// supposed to have filled. That was tried, and it collapsed the domain to the
// occupancy itself.
//
// So the outside label is grown back afterwards instead, GEODESICALLY — one cell
// at a time, through cells that hold no return. That reaches the wall's outer
// face and stops there, because the occupancy blocks it, so the skin goes and
// the inside is untouched however thin the wall is. A plain dilation of the
// label would cross a wall thinner than twice the seal and delete the inside
// with it.
//
// What none of this can guarantee is that the SURVEY has no hole wider than the
// seal. That failure is silent and total — the flood fills the building and the
// interior rule deletes the answer — so it is caught rather than assumed: see
// the setup test in build(), which asks whether the flood reached a cell an
// instrument was standing in. Nothing that happened inside a building is outside
// it.

// Spreads `mark` through everything that is not `blocked`, from wherever `mark`
// already is. The same span walk floodOutside uses, for the same reason: a region
// can be tens of millions of cells and a stack of single cells over it is not
// affordable where the spans covering it number in the thousands.
void spreadThrough(Grid& g, uint8_t mark, uint8_t blocked) {
    const uint32_t X = g.dim[0], Y = g.dim[1], Z = g.dim[2];
    struct Span { uint32_t z, y, x0, x1; };
    std::vector<Span> stack;

    auto open = [&](uint32_t x, uint32_t y, uint32_t z) {
        const uint8_t b = g.inDomain[g.index(x, y, z)];
        return !(b & blocked) && !(b & mark);
    };
    auto fill = [&](uint32_t x, uint32_t y, uint32_t z) {
        if (!open(x, y, z)) return;
        uint32_t x0 = x, x1 = x;
        while (x0 > 0 && open(x0 - 1, y, z)) --x0;
        while (x1 + 1 < X && open(x1 + 1, y, z)) ++x1;
        for (uint32_t i = x0; i <= x1; ++i) g.inDomain[g.index(i, y, z)] |= mark;
        stack.push_back({z, y, x0, x1});
    };

    // Seeded from every run already marked, so the caller marks the seeds and this
    // carries them as far as they reach.
    //
    // The seed runs are EXTENDED along x as they are found, not merely pushed. A
    // run of already-marked cells cannot be grown by `fill`, which refuses a cell
    // that carries the mark — so pushing it as it stands leaves the only way out
    // of it through a neighbouring line in y or z. A region reachable from its
    // seed only along x is then never marked at all: a corridor a cell wide in the
    // other two axes, and, less obviously, any region whose seed happens to sit in
    // a line that its walls close off.
    for (uint32_t z = 0; z < Z; ++z)
        for (uint32_t y = 0; y < Y; ++y)
            for (uint32_t x = 0; x < X; ++x)
                if (g.inDomain[g.index(x, y, z)] & mark) {
                    uint32_t x0 = x, x1 = x;
                    while (x1 + 1 < X &&
                           ((g.inDomain[g.index(x1 + 1, y, z)] & mark) ||
                            open(x1 + 1, y, z))) ++x1;
                    while (x0 > 0 && open(x0 - 1, y, z)) --x0;
                    for (uint32_t i = x0; i <= x1; ++i) g.inDomain[g.index(i, y, z)] |= mark;
                    stack.push_back({z, y, x0, x1});
                    x = x1;
                }

    while (!stack.empty()) {
        const Span s = stack.back();
        stack.pop_back();
        const int32_t dy[4] = {-1, 1, 0, 0};
        const int32_t dz[4] = {0, 0, -1, 1};
        for (int k = 0; k < 4; ++k) {
            const int64_t ny = int64_t(s.y) + dy[k], nz = int64_t(s.z) + dz[k];
            if (ny < 0 || nz < 0 || ny >= int64_t(Y) || nz >= int64_t(Z)) continue;
            for (uint32_t x = s.x0; x <= s.x1; ++x)
                fill(x, uint32_t(ny), uint32_t(nz));
        }
    }
}

// Exact squared Euclidean distance transform, one axis at a time.
//
// Felzenszwalb and Huttenlocher's lower envelope of parabolas: for each line,
// the transform is the lower envelope of one parabola per sample, found in a
// single forward scan because the parabolas are added in order of vertex. Linear
// in the number of cells per axis, exact rather than the chamfer approximations
// that make a dilation come out octagonal, and separable — which is what lets a
// three-dimensional field be three passes of a one-dimensional routine.
//
// Distances are in cells, squared, so the whole thing stays in integers until
// the caller compares against a radius.
//
// One detail that is not in the paper and bites immediately: the samples that
// are not seeds have to carry a LARGE FINITE value, not an infinity. The scan
// computes where one parabola overtakes another as a difference of their heights
// over a difference of their positions, and two infinite heights give
// infinity minus infinity, which is a NaN — and a NaN fails the comparison that
// ends the loop, so it walks the stack index off the bottom of the array. An
// empty first line is enough to do it, which is to say almost every grid.
constexpr double kFar = 1e18;

void edt1d(const std::vector<double>& f, std::vector<double>& d, size_t n,
           std::vector<int>& v, std::vector<double>& z) {
    const double kInf = std::numeric_limits<double>::infinity();
    int k = 0;
    v[0] = 0;
    z[0] = -kInf;
    z[1] = kInf;
    for (size_t q = 1; q < n; ++q) {
        // Where this parabola overtakes the one currently on top. Walking k back
        // discards the parabolas it has hidden.
        double s;
        for (;;) {
            const double vq = double(v[k]);
            s = ((f[q] + double(q) * double(q)) - (f[size_t(v[k])] + vq * vq)) /
                (2.0 * double(q) - 2.0 * vq);
            if (s > z[size_t(k)]) break;
            --k;
        }
        ++k;
        v[size_t(k)] = int(q);
        z[size_t(k)] = s;
        z[size_t(k) + 1] = kInf;
    }
    k = 0;
    for (size_t q = 0; q < n; ++q) {
        while (z[size_t(k) + 1] < double(q)) ++k;
        const double dx = double(q) - double(v[size_t(k)]);
        d[q] = dx * dx + f[size_t(v[size_t(k)])];
    }
}

// Squared distance in cells from every cell to the nearest seed. With `invert`
// the seeds are the cells WITHOUT the bit, which is what an erosion needs:
// eroding a set by r is keeping the cells further than r from its complement.
std::vector<double> distanceTo(const Grid& g, uint8_t seed, bool invert = false) {
    const uint32_t X = g.dim[0], Y = g.dim[1], Z = g.dim[2];
    std::vector<double> d(size_t(X) * Y * Z);
    for (size_t i = 0; i < d.size(); ++i)
        d[i] = (((g.inDomain[i] & seed) != 0) != invert) ? 0.0 : kFar;

    const size_t longest = std::max(std::max<size_t>(X, Y), size_t(Z));
    std::vector<double> f(longest), out(longest), z(longest + 1);
    std::vector<int>    v(longest);

    for (uint32_t z0 = 0; z0 < Z; ++z0)
        for (uint32_t y = 0; y < Y; ++y) {
            for (uint32_t x = 0; x < X; ++x) f[x] = d[g.index(x, y, z0)];
            edt1d(f, out, X, v, z);
            for (uint32_t x = 0; x < X; ++x) d[g.index(x, y, z0)] = out[x];
        }
    for (uint32_t z0 = 0; z0 < Z; ++z0)
        for (uint32_t x = 0; x < X; ++x) {
            for (uint32_t y = 0; y < Y; ++y) f[y] = d[g.index(x, y, z0)];
            edt1d(f, out, Y, v, z);
            for (uint32_t y = 0; y < Y; ++y) d[g.index(x, y, z0)] = out[y];
        }
    for (uint32_t y = 0; y < Y; ++y)
        for (uint32_t x = 0; x < X; ++x) {
            for (uint32_t z0 = 0; z0 < Z; ++z0) f[z0] = d[g.index(x, y, z0)];
            edt1d(f, out, Z, v, z);
            for (uint32_t z0 = 0; z0 < Z; ++z0) d[g.index(x, y, z0)] = out[z0];
        }
    return d;
}

// Marks everything the outside can reach without crossing a sealing cell.
//
// Span-based rather than per-cell, for the same reason voids::classify is: the
// exterior of a site is enormous and mostly open, and a stack of single cells
// over it runs to hundreds of millions of entries where the spans covering the
// same volume number in the thousands.
void floodOutside(Grid& g) {
    const uint32_t X = g.dim[0], Y = g.dim[1], Z = g.dim[2];
    struct Span { uint32_t z, y, x0, x1; };
    std::vector<Span> stack;

    auto open = [&](uint32_t x, uint32_t y, uint32_t z) {
        const uint8_t b = g.inDomain[g.index(x, y, z)];
        return !(b & kBarrier) && !(b & kOutside);
    };
    // Fills the run containing (x, y, z) and pushes it.
    auto fill = [&](uint32_t x, uint32_t y, uint32_t z) {
        if (!open(x, y, z)) return;
        uint32_t x0 = x, x1 = x;
        while (x0 > 0 && open(x0 - 1, y, z)) --x0;
        while (x1 + 1 < X && open(x1 + 1, y, z)) ++x1;
        for (uint32_t i = x0; i <= x1; ++i) g.inDomain[g.index(i, y, z)] |= kOutside;
        stack.push_back({z, y, x0, x1});
    };

    // Seeded from the whole boundary. The grid is padded past everything the
    // survey reached, so every face of it is genuinely outside.
    for (uint32_t z = 0; z < Z; ++z)
        for (uint32_t y = 0; y < Y; ++y) {
            fill(0, y, z);
            if (X > 1) fill(X - 1, y, z);
        }
    for (uint32_t z = 0; z < Z; ++z)
        for (uint32_t x = 0; x < X; ++x) {
            fill(x, 0, z);
            if (Y > 1) fill(x, Y - 1, z);
        }
    for (uint32_t y = 0; y < Y; ++y)
        for (uint32_t x = 0; x < X; ++x) {
            fill(x, y, 0);
            if (Z > 1) fill(x, y, Z - 1);
        }

    while (!stack.empty()) {
        const Span s = stack.back();
        stack.pop_back();
        const int32_t dy[4] = {-1, 1, 0, 0};
        const int32_t dz[4] = {0, 0, -1, 1};
        for (int k = 0; k < 4; ++k) {
            const int64_t ny = int64_t(s.y) + dy[k], nz = int64_t(s.z) + dz[k];
            if (ny < 0 || nz < 0 || ny >= int64_t(Y) || nz >= int64_t(Z)) continue;
            for (uint32_t x = s.x0; x <= s.x1; ++x)
                fill(x, uint32_t(ny), uint32_t(nz));
        }
    }
}

// Walks the outside label back to the surfaces the seal held it away from.
//
// DOWNHILL on the distance to the nearest return, and that is the whole of what
// makes it safe. The flood stops a seal's width short of every surface, so the
// label has to be walked in to the wall or a skin of unobserved cells survives
// outside the building — which is the closed shell interior-only mode exists to
// remove. But the seal also spans the openings in the envelope, and a walk that
// simply took `seal` steps would travel straight down one of those into the
// building and delete a seal's width of the answer behind it. That was the first
// attempt, and it made the result non-monotone in the seal: widening the seal
// dropped cells a narrower one had kept.
//
// Requiring each step to land strictly nearer a return fixes it by geometry
// rather than by a step count. Approaching a wall, the distance falls to zero,
// so the walk arrives and stops. Approaching an opening, the distance rises
// towards the middle of the gap before it falls again, and the walk cannot climb
// that ridge — so it gets no further than the mouth. It also needs no bound: the
// distance strictly decreases, so it terminates on its own.
void growOutside(Grid& g, const std::vector<double>& d2) {
    const uint32_t X = g.dim[0], Y = g.dim[1], Z = g.dim[2];
    std::vector<uint8_t> next;
    for (;;) {
        next.assign(g.inDomain.size(), 0);
        bool grew = false;
        for (uint32_t z = 0; z < Z; ++z)
            for (uint32_t y = 0; y < Y; ++y)
                for (uint32_t x = 0; x < X; ++x) {
                    const size_t i = g.index(x, y, z);
                    if (!(g.inDomain[i] & kOutside)) continue;
                    const int32_t d[6][3] = {{1,0,0}, {-1,0,0}, {0,1,0},
                                             {0,-1,0}, {0,0,1}, {0,0,-1}};
                    for (const auto& o : d) {
                        const int64_t nx = int64_t(x) + o[0];
                        const int64_t ny = int64_t(y) + o[1];
                        const int64_t nz = int64_t(z) + o[2];
                        if (nx < 0 || ny < 0 || nz < 0 || nx >= int64_t(X) ||
                            ny >= int64_t(Y) || nz >= int64_t(Z)) continue;
                        const size_t j = g.index(uint32_t(nx), uint32_t(ny), uint32_t(nz));
                        if (g.inDomain[j] & (kOutside | kOccupied)) continue;
                        if (!(d2[j] < d2[i])) continue;          // downhill only
                        next[j] = 1;
                        grew = true;
                    }
                }
        if (!grew) return;
        for (size_t i = 0; i < g.inDomain.size(); ++i)
            if (next[i]) g.inDomain[i] |= kOutside;
    }
}

} // namespace

bool Grid::cellInDomain(int64_t x, int64_t y, int64_t z) const {
    if (x < 0 || y < 0 || z < 0 || x >= int64_t(dim[0]) || y >= int64_t(dim[1]) ||
        z >= int64_t(dim[2])) return false;
    return (inDomain[index(uint32_t(x), uint32_t(y), uint32_t(z))] & kInDomain) != 0;
}

bool Grid::cellOccupied(int64_t x, int64_t y, int64_t z) const {
    if (x < 0 || y < 0 || z < 0 || x >= int64_t(dim[0]) || y >= int64_t(dim[1]) ||
        z >= int64_t(dim[2])) return false;
    return (inDomain[index(uint32_t(x), uint32_t(y), uint32_t(z))] & kOccupied) != 0;
}

bool Grid::contains(double wx, double wy, double wz) const {
    if (inDomain.empty() || !(cell > 0)) return true;
    const double w[3] = {wx, wy, wz};
    uint32_t c[3];
    for (int k = 0; k < 3; ++k) {
        const int64_t i = int64_t(std::floor(w[k] / cell)) - lo[k];
        if (i < 0 || i >= int64_t(dim[k])) return false;
        c[k] = uint32_t(i);
    }
    return (inDomain[index(c[0], c[1], c[2])] & kInDomain) != 0;
}

int Grid::testBox(const double blo[3], const double bhi[3]) const {
    if (inDomain.empty() || !(cell > 0)) return 2;
    // The cells the box touches, clamped. A box reaching outside the grid is
    // partly outside the wrap, which is why the clamp is recorded rather than
    // just applied.
    int64_t c0[3], c1[3];
    bool clipped = false;
    for (int k = 0; k < 3; ++k) {
        c0[k] = int64_t(std::floor(blo[k] / cell)) - lo[k];
        c1[k] = int64_t(std::floor(bhi[k] / cell)) - lo[k];
        if (c1[k] < 0 || c0[k] >= int64_t(dim[k])) return 0;      // wholly outside
        if (c0[k] < 0) { c0[k] = 0; clipped = true; }
        if (c1[k] >= int64_t(dim[k])) { c1[k] = int64_t(dim[k]) - 1; clipped = true; }
    }
    bool any = false, all = true;
    for (int64_t z = c0[2]; z <= c1[2] && (!any || all); ++z)
        for (int64_t y = c0[1]; y <= c1[1] && (!any || all); ++y)
            for (int64_t x = c0[0]; x <= c1[0] && (!any || all); ++x) {
                if (inDomain[index(uint32_t(x), uint32_t(y), uint32_t(z))] & kInDomain)
                    any = true;
                else
                    all = false;
            }
    if (!any) return 0;
    // Clipped means part of the box was outside the grid, and outside the grid is
    // outside the wrap — so it cannot be Full however the cells came out.
    return (all && !clipped) ? 2 : 1;
}

bool size(const double lo[3], const double hi[3], const Options& opt, Grid& grid,
          std::string& err) {
    grid = Grid{};
    // The MAGNITUDE sizes the grid. A negative buffer is a shell pulled in rather
    // than grown out — see Options::buffer — and it wants the same resolution and
    // the same clearance at the boundary as the positive one.
    const double reach = std::fabs(opt.buffer);
    double cell = (opt.cell > 0) ? opt.cell : reach / 4.0;
    if (!(cell > 0)) { err = "wrap: the buffer must not be zero"; return false; }
    if (!(reach > 0)) { err = "wrap: the buffer must not be zero"; return false; }
    for (int k = 0; k < 3; ++k)
        if (!(hi[k] >= lo[k])) { err = "wrap: the site's extent is empty"; return false; }

    // Padded past the buffer AND the seal, so the grid's boundary is clear of
    // both and the interior-only flood can start there and mean it.
    //
    // The seal has to be in here. It dilates the barrier the flood cannot cross,
    // so a seal that reaches the boundary leaves the flood with nowhere to start
    // — and the interior rule then drops nothing at all, silently, having decided
    // that the whole site is inside. Measured before this was fixed: a 3 m seal
    // on a grid padded 1.5 m dropped zero cells where a 1 m seal dropped 44,412.
    // The closing is in here too: it dilates before it erodes, and a dilation that
    // reached the boundary would touch the grid's own edge and be eroded back
    // against it rather than against open air.
    const double seal = (opt.seal > 0.0) ? opt.seal
                                        : std::max(0.5 * reach, 0.5 * opt.spanGaps);
    const double pad = reach + seal + 0.5 * std::max(0.0, opt.spanGaps) + 2.0 * cell;
    for (;;) {
        uint64_t cells = 1;
        bool fits = true;
        int64_t glo[3];
        uint32_t gdim[3];
        for (int k = 0; k < 3; ++k) {
            glo[k] = int64_t(std::floor((lo[k] - pad) / cell));
            const int64_t ghi = int64_t(std::floor((hi[k] + pad) / cell));
            const int64_t span = ghi - glo[k] + 1;
            if (span <= 0 || span > int64_t(1) << 31) { fits = false; break; }
            gdim[k] = uint32_t(span);
            cells *= uint64_t(span);
            if (cells > opt.maxCells) { fits = false; break; }
        }
        if (fits) {
            grid.cell = cell;
            for (int k = 0; k < 3; ++k) { grid.lo[k] = glo[k]; grid.dim[k] = gdim[k]; }
            grid.inDomain.assign(size_t(cells), 0);
            grid.buffer = opt.buffer;
            grid.interiorOnly = opt.interiorOnly;
            return true;
        }
        cell *= 2.0;
        grid.coarsened = true;
        // A cell coarser than the buffer stops being a wrap at all: the dilation
        // would have less than one cell of radius and the shape would be the
        // grid's rather than the site's. Better to say so than to hand back a
        // domain shaped like its own bookkeeping.
        if (cell > opt.buffer) {
            char buf[220];
            std::snprintf(buf, sizeof(buf),
                          "wrap: the site needs cells coarser than the %.2f m buffer to fit "
                          "%llu cells — raise the budget or the buffer",
                          opt.buffer, (unsigned long long)opt.maxCells);
            err = buf;
            return false;
        }
    }
}

void markScan(const MarkSource& src, Grid& grid) {
    if (grid.empty() || src.rows == 0 || src.cols == 0 || !src.pointAt) return;

    // EVERY cell, no stride. That is a correction, and the reason is worth
    // keeping because the mistake is an easy one to make twice.
    //
    // This used to sample one raster cell in `n`, with n chosen so that adjacent
    // sampled rays stay closer together than a grid cell at the furthest range
    // the image reaches. That reasoning is only true for a surface square to the
    // ray. At grazing incidence — which is most of the ground, and every wall
    // seen nearly edge on — consecutive rays land further apart by a factor of
    // one over the sine of the incidence angle, without bound. On a real scan a
    // stride of six missed FIFTY-SIX PER CENT of the occupied cells: 16,127
    // marked where marking every cell finds 37,012.
    //
    // Under-marking shrinks the wrap, and a smaller domain removes questions
    // rather than answering them differently — the dangerous direction. So the
    // sampling is exact, and the cost was moved instead: see the direction
    // tables the caller builds, which take the trigonometry out of the inner
    // loop and leave about twenty flops a cell.
    for (uint32_t r = 0; r < src.rows; ++r) {
        for (uint32_t c = 0; c < src.cols; ++c) {
            double w[3];
            if (!src.pointAt(src.user, src.imageForStatus, r, c, w)) continue;
            int64_t i[3];
            bool in = true;
            for (int k = 0; k < 3; ++k) {
                i[k] = int64_t(std::floor(w[k] / grid.cell)) - grid.lo[k];
                if (i[k] < 0 || i[k] >= int64_t(grid.dim[k])) { in = false; break; }
            }
            if (!in) continue;
            // An ATOMIC or into the byte, so several scans can mark at once.
            //
            // A plain |= would not do, however harmless it looks. Two threads
            // reading the same byte, setting their bit and writing it back lose
            // one of the two, and what is lost is an occupied cell — which
            // shrinks the wrap, and a smaller domain removes questions rather
            // than answering them differently. Relaxed ordering is enough: the
            // bits are independent and nothing is published through them, so
            // only the read-modify-write needs to be indivisible.
            uint8_t* cellByte =
                &grid.inDomain[grid.index(uint32_t(i[0]), uint32_t(i[1]), uint32_t(i[2]))];
            __atomic_or_fetch(cellByte, kOccupied, __ATOMIC_RELAXED);
        }
    }
}

void build(const Options& opt, Grid& grid, const std::vector<double>& setupsXYZ) {
    if (grid.empty()) return;
    grid.buffer = opt.buffer;
    grid.interiorOnly = opt.interiorOnly || opt.buffer < 0.0;
    grid.pulledIn = opt.buffer < 0.0;
    grid.occupiedCells = 0;
    for (uint8_t b : grid.inDomain) if (b & kOccupied) ++grid.occupiedCells;
    if (grid.occupiedCells == 0) return;   // nothing marked: leave the domain empty

    grid.spanGaps = 0.0;
    grid.bridgedCells = 0;
    // The domain: everything within the buffer of a measured return — or, where
    // the buffer is negative, the region the survey encloses pulled in by that
    // much. See Options::buffer. The magnitude sizes everything either way.
    const bool   pullIn   = opt.buffer < 0.0;
    const double bufCells = std::fabs(opt.buffer) / grid.cell;
    const double bufSq    = bufCells * bufCells;
    // The seal: how wide a hole in the survey the flood is not allowed through.
    // Half the buffer by default, so it closes holes up to a buffer wide — a
    // doorway is nine hundred millimetres and a window less, and both are holes
    // in a survey of an interior however carefully it was done. A physical length
    // rather than a cell count, because a doorway is a doorway whatever
    // resolution the grid happens to be at. The same distance field answers this
    // and the buffer, so the seal costs nothing beyond the comparison.
    // Derived from the OPENING the shell was told to bridge where there is one,
    // and from the buffer otherwise. The seal is a statement about how wide a hole
    // in the envelope counts as a portal, which is the same statement spanGaps
    // makes — and tying it to the buffer alone breaks as soon as the buffer is
    // small: half a metre of buffer gives a quarter metre of seal, and the flood
    // walks in through the first raster-sized gap in a wall.
    const double sealCells =
        ((opt.seal > 0.0) ? opt.seal
                          : std::max(0.5 * std::fabs(opt.buffer), 0.5 * opt.spanGaps)) /
        grid.cell;
    const double sealSq    = sealCells * sealCells;
    grid.seal = sealCells * grid.cell;
    const std::vector<double> d2 = distanceTo(grid, kOccupied);
    for (size_t i = 0; i < grid.inDomain.size(); ++i) {
        // Pulling in, the domain is not a skin around the surfaces at all, and it
        // cannot be decided here: it is what the flood below does not reach, less
        // the magnitude. Only the barrier is set now.
        if (!pullIn && d2[i] <= bufSq) grid.inDomain[i] |= kInDomain;
        if (d2[i] <= sealSq) grid.inDomain[i] |= kBarrier;
    }

    // Pulling in IS the interior question, and asks for the same flood.
    // The flood runs whenever the answer depends on which side of the shell a cell
    // is on: to pull the boundary in, to drop the outside, or to span an opening.
    if (opt.interiorOnly || pullIn || opt.spanGaps > 0.0) {
        floodOutside(grid);
        // Back to the surfaces the seal held it away from — see growOutside.
        growOutside(grid, d2);

        // Did the shell hold? An instrument stood inside the building, so if the
        // flood reached the cell it was standing in, it came in through a hole
        // and everything it then reached is wrongly called outside. That is a
        // total failure and a silent one, so it is checked before it is acted on
        // rather than after somebody notices the answer is empty.
        for (size_t i = 0; i + 2 < setupsXYZ.size(); i += 3) {
            int64_t c[3];
            bool inGrid = true;
            for (int k = 0; k < 3; ++k) {
                c[k] = int64_t(std::floor(setupsXYZ[i + size_t(k)] / grid.cell)) - grid.lo[k];
                if (c[k] < 0 || c[k] >= int64_t(grid.dim[k])) { inGrid = false; break; }
            }
            if (!inGrid) continue;
            if (grid.inDomain[grid.index(uint32_t(c[0]), uint32_t(c[1]), uint32_t(c[2]))]
                & kOutside) {
                grid.sealLeaked = true;
                break;
            }
        }

        // THE ENVELOPE: the watertight surface of the surveyed shell.
        //
        // This is what spans an opening, and it is a topological answer rather
        // than a morphological one because morphology cannot give it. A closing —
        // dilate by r, erode by r — is the textbook way to fill a hole, and it
        // does not work on a SURFACE: a ball of radius r always fits through a
        // hole of radius a by sitting at sqrt(r*r - a*a) from the plane, so the
        // erosion takes back exactly what the dilation bridged. Measured on a wall
        // with a 1.2 m window: at a 1.4 m closing, 24 cells filled, and the middle
        // of the window open at every radius tried.
        //
        // The flood does work, because it asks a different question. A barrier
        // half an opening wide blocks a 6-connected path through it, so whatever
        // the flood cannot reach is enclosed — and the cells of that region that
        // touch the outside ARE the shell's surface, openings included. No ball
        // has to fit anywhere.
        //
        // Marked here, between the flood and the domain, and only where an opening
        // was asked to be spanned: without spanGaps the envelope is the occupancy
        // and this is the shape it has always been.
        if (opt.spanGaps > 0.0 && !grid.sealLeaked) {
            const int64_t X = grid.dim[0], Y = grid.dim[1], Z = grid.dim[2];
            for (int64_t z = 0; z < Z; ++z)
                for (int64_t y = 0; y < Y; ++y)
                    for (int64_t x = 0; x < X; ++x) {
                        const size_t i = grid.index(uint32_t(x), uint32_t(y), uint32_t(z));
                        const uint8_t b = grid.inDomain[i];
                        if (b & (kOutside | kOccupied)) continue;
                        static const int64_t d[6][3] = {{1,0,0},{-1,0,0},{0,1,0},
                                                        {0,-1,0},{0,0,1},{0,0,-1}};
                        for (const auto& n : d) {
                            const int64_t nx = x + n[0], ny = y + n[1], nz = z + n[2];
                            if (nx < 0 || ny < 0 || nz < 0 || nx >= X || ny >= Y || nz >= Z)
                                continue;
                            if (!(grid.inDomain[grid.index(uint32_t(nx), uint32_t(ny),
                                                           uint32_t(nz))] & kOutside))
                                continue;
                            grid.inDomain[i] = uint8_t(b | kEnvelope);
                            ++grid.bridgedCells;
                            break;
                        }
                    }
            grid.spanGaps = 2.0 * sealCells * grid.cell;
        }

        if (pullIn) {
            // THE SHELL, PULLED IN — ONE ENCLOSED REGION AT A TIME.
            //
            // Pulling the boundary inside the walls only means anything where
            // there are walls with an inside. A real site is not one building: it
            // is a building, and a boundary wall with nothing behind it, and a
            // canopy, and a stretch of fence, and a lean-to whose door stood open
            // while the survey ran. Deciding the whole site on one flood made all
            // of those share a verdict — and the verdict failed on the hardest
            // one, so a leak in a shed took the question away from the building.
            //
            // So each enclosed region is assessed on its own merits, which is the
            // same rule the scans themselves get:
            //
            //   Deep enough to pull in — the erosion leaves a core — and that
            //   region is the question, with the walls around it and everything
            //   beyond them left out. This is the building.
            //
            //   Too thin, or not enclosed at all — a freestanding wall, a canopy,
            //   a room the flood got into — and that surface keeps the ordinary
            //   skin of the buffer's width. Not as tight as an interior, and the
            //   right answer where there is no interior to be had.
            //
            // FAILING IS NOT AN OUTCOME. Every surface is covered by one rule or
            // the other, so the worst case is a region asked about too generously
            // rather than a site with no question at all.
            // WHICH REGIONS ARE ENCLOSED, and which of them are deep enough.
            //
            // Note what this CANNOT see, because it decides whether the negative
            // buffer does anything at all. An indoor survey stops where the job
            // stops: mid corridor, at the edge of the area of interest, with no
            // wall across the end because there is no wall there. The flood from
            // the grid's boundary pours in through every one of those open ends,
            // and a site with no closed end anywhere has no enclosed region, no
            // core, and every surface falling back to the skin — which is the
            // negative buffer quietly doing nothing. `enclosedRegions` and
            // `coredRegions` are reported so that case is visible as a number
            // rather than as a picture that looks like the positive answer.
            const std::vector<double> dOut = distanceTo(grid, kOutside);
            for (size_t i = 0; i < grid.inDomain.size(); ++i)
                if (!(grid.inDomain[i] & kOutside) && dOut[i] > bufSq)
                    grid.inDomain[i] |= uint8_t(kCore | kKeptIn);
            // Which enclosed regions those cores belong to: everything reachable
            // from a core without crossing the outside.
            spreadThrough(grid, kKeptIn, kOutside);

            // How much of the site was enclosed at all, against how much of that
            // was deep enough to pull into. Two numbers rather than one because
            // they fail differently: no enclosed space at all is an open-ended
            // survey, and enclosed space with no core is a site whose rooms are
            // thinner than the buffer asked for.
            for (size_t i = 0; i < grid.inDomain.size(); ++i) {
                const uint8_t b = grid.inDomain[i];
                if (!(b & (kOutside | kOccupied))) ++grid.enclosedCells;
                if (b & kKeptIn) ++grid.keptInCells;
            }
            for (size_t i = 0; i + 2 < setupsXYZ.size(); i += 3) {
                int64_t c[3];
                bool inGrid = true;
                for (int k = 0; k < 3; ++k) {
                    c[k] = int64_t(std::floor(setupsXYZ[i + size_t(k)] / grid.cell)) - grid.lo[k];
                    if (c[k] < 0 || c[k] >= int64_t(grid.dim[k])) { inGrid = false; break; }
                }
                if (!inGrid) continue;
                ++grid.setupsSeen;
                if (grid.inDomain[grid.index(uint32_t(c[0]), uint32_t(c[1]),
                                             uint32_t(c[2]))] & kKeptIn) ++grid.setupsPulledIn;
            }

            for (size_t i = 0; i < grid.inDomain.size(); ++i) {
                const uint8_t b = grid.inDomain[i];
                if (b & kCore) { grid.inDomain[i] = uint8_t(b | kInDomain); ++grid.pulledInCells; }
            }

            // The skin, for everything the pull-in did not speak for. Seeded from
            // the surfaces that do NOT bound a region deep enough to pull into: a
            // wall of the building has the building on one side and is left to the
            // erosion, while a freestanding wall bounds nothing and keeps its skin.
            uint64_t seeds = 0;
            const int64_t X = grid.dim[0], Y = grid.dim[1], Z = grid.dim[2];
            for (int64_t z = 0; z < Z; ++z)
                for (int64_t y = 0; y < Y; ++y)
                    for (int64_t x = 0; x < X; ++x) {
                        const size_t i = grid.index(uint32_t(x), uint32_t(y), uint32_t(z));
                        const uint8_t b = grid.inDomain[i];
                        if (!(b & (kOccupied | kEnvelope))) continue;
                        bool bounds = false;
                        static const int64_t d[6][3] = {{1,0,0},{-1,0,0},{0,1,0},
                                                        {0,-1,0},{0,0,1},{0,0,-1}};
                        for (const auto& n : d) {
                            const int64_t nx = x + n[0], ny = y + n[1], nz = z + n[2];
                            if (nx < 0 || ny < 0 || nz < 0 || nx >= X || ny >= Y || nz >= Z)
                                continue;
                            if (grid.inDomain[grid.index(uint32_t(nx), uint32_t(ny),
                                                         uint32_t(nz))] & kKeptIn) {
                                bounds = true;
                                break;
                            }
                        }
                        if (!bounds) { grid.inDomain[i] |= kSeed; ++seeds; }
                    }
            grid.skinnedSurfaces = seeds;
            if (seeds) {
                const std::vector<double> dSkin = distanceTo(grid, kSeed);
                for (size_t i = 0; i < grid.inDomain.size(); ++i)
                    if (dSkin[i] <= bufSq) grid.inDomain[i] |= kInDomain;
            }
            for (uint8_t& b : grid.inDomain) b = uint8_t(b & ~kSeed);
        } else {
            // The skin, now measured from the envelope rather than from the bare
            // returns, so it crosses an opening instead of following it inward.
            if (opt.spanGaps > 0.0 && !grid.sealLeaked) {
                const std::vector<double> dEnv =
                    distanceTo(grid, uint8_t(kOccupied | kEnvelope));
                for (size_t i = 0; i < grid.inDomain.size(); ++i)
                    if (dEnv[i] <= bufSq) grid.inDomain[i] |= kInDomain;
            }
            if (!grid.sealLeaked && opt.interiorOnly) {
                // A domain cell the flood reached is on the outside of the
                // surveyed shell. Occupied cells are never dropped: they hold
                // measured surface, and a surface is not unobserved space
                // whichever side it was seen from.
                for (size_t i = 0; i < grid.inDomain.size(); ++i) {
                    const uint8_t b = grid.inDomain[i];
                    if ((b & kInDomain) && (b & kOutside) && !(b & kOccupied)) {
                        grid.inDomain[i] = uint8_t(b & ~kInDomain);
                        ++grid.droppedOutside;
                    }
                }
            }
        }
    }

    grid.domainCells = 0;
    for (uint8_t b : grid.inDomain) if (b & kInDomain) ++grid.domainCells;
}

} // namespace wrap
