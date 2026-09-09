#include "range_image.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace rimg {

static constexpr double kPi  = 3.14159265358979323846;
static constexpr double kTau = 2.0 * kPi;

const char* statusName(Status s) {
    switch (s) {
    case Status::Hit:        return "hit";
    case Status::NoReturn:   return "no-return";
    case Status::OutsideFov: return "outside-fov";
    }
    return "?";
}

void toSpherical(double x, double y, double z, double& az, double& el, double& range) {
    range = std::sqrt(x * x + y * y + z * z);
    az = std::atan2(y, x);
    if (az < 0) az += kTau;
    el = (range > 1e-12) ? std::asin(std::clamp(z / range, -1.0, 1.0)) : 0.0;
}

namespace {

// Shortest signed difference between two angles, in (-pi, pi].
double angleDiff(double a, double b) {
    double d = a - b;
    while (d >  kPi) d -= kTau;
    while (d <= -kPi) d += kTau;
    return d;
}

struct Accum {
    double sumEl = 0;          // elevation: plain mean
    double sumAzX = 0, sumAzY = 0;   // azimuth: circular mean, so 0/2pi does not average to pi
    uint64_t n = 0;
};

// Fits y = a + b*x by least squares over the filled entries, returning the
// maximum residual so a non-uniform raster is visible rather than silent.
bool fitLine(const std::vector<double>& y, const std::vector<uint64_t>& n,
             double& a, double& b, double& maxResidual) {
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    uint64_t count = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        if (!n[i]) continue;
        const double x = double(i);
        sx += x; sy += y[i]; sxx += x * x; sxy += x * y[i];
        ++count;
    }
    if (count < 2) return false;
    const double denom = double(count) * sxx - sx * sx;
    if (std::fabs(denom) < 1e-12) return false;
    b = (double(count) * sxy - sx * sy) / denom;
    a = (sy - b * sx) / double(count);

    maxResidual = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        if (!n[i]) continue;
        maxResidual = std::max(maxResidual, std::fabs(y[i] - (a + b * double(i))));
    }
    return true;
}

// Fills the gaps in a measured table, so every row and column maps somewhere.
//
// Real scans need this. The blind cone leaves 590 of 2500 rows with no returns at
// all, and a raster with 43% of its cells filled has columns that happen to be
// empty too. Those entries still have to hold an angle: an elevation inside the
// cone that fell outside the table would be rejected as off the raster, and the
// difference between "off the raster" and "a row the instrument never got a return
// in" is the difference between two diagnostics.
//
// Interior gaps interpolate between the entries either side. The ends extrapolate
// from a line through the nearest few populated entries rather than from the two
// nearest, because a single row's mean wobbles and 590 rows of extrapolation would
// ride on that wobble.
constexpr size_t kExtrapFitSpan = 64;

bool fillTable(std::vector<double>& t, const std::vector<uint64_t>& n) {
    const size_t N = t.size();
    if (N < 2 || n.size() != N) return false;

    std::vector<size_t> have;
    have.reserve(N);
    for (size_t i = 0; i < N; ++i) if (n[i]) have.push_back(i);
    if (have.size() < 2) return false;

    for (size_t k = 0; k + 1 < have.size(); ++k) {
        const size_t i0 = have[k], i1 = have[k + 1];
        if (i1 == i0 + 1) continue;
        const double slope = (t[i1] - t[i0]) / double(i1 - i0);
        for (size_t i = i0 + 1; i < i1; ++i) t[i] = t[i0] + slope * double(i - i0);
    }

    auto endSlope = [&](bool low) -> double {
        const size_t m = std::min(kExtrapFitSpan, have.size());
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (size_t k = 0; k < m; ++k) {
            const size_t i = low ? have[k] : have[have.size() - 1 - k];
            const double x = double(i), y = t[i];
            sx += x; sy += y; sxx += x * x; sxy += x * y;
        }
        const double den = double(m) * sxx - sx * sx;
        if (std::fabs(den) < 1e-12) return 0.0;
        return (double(m) * sxy - sx * sy) / den;
    };

    if (have.front() > 0) {
        const double s = endSlope(true);
        const size_t i0 = have.front();
        for (size_t i = 0; i < i0; ++i) t[i] = t[i0] - s * double(i0 - i);
    }
    if (have.back() + 1 < N) {
        const double s = endSlope(false);
        const size_t i1 = have.back();
        for (size_t i = i1 + 1; i < N; ++i) t[i] = t[i1] + s * double(i - i1);
    }
    // Deliberately not clamped to +-pi/2. An extrapolated elevation past the pole
    // is never queried — no direction has one — and clamping would flatten the
    // end of the table into a run of equal values, which is a fold-back as far as
    // the monotonicity check is concerned.
    return true;
}

// Does the table run one way, to within a fraction of a cell?
//
// Not exactly: a row's mean elevation comes from however many points landed in
// that row, and a few of them wobble it. A wobble of a quarter of a cell is not a
// mirror turning back on itself — a real fold-back reverses over hundreds of rows
// and hundreds of cells.
constexpr double kMonotonicSlackCells = 0.25;

bool monotonicWithin(const std::vector<double>& t, double slack) {
    bool up = true, down = true;
    for (size_t i = 1; i < t.size(); ++i) {
        const double d = t[i] - t[i - 1];
        if (d < -slack) up = false;
        if (d >  slack) down = false;
    }
    return up || down;
}

// Stamps each bin of a reverse index with the nearest entry of a measured table.
//
// `sorted` is (angle, index) ascending. One merge pass over the bins, which makes
// no assumption that the table is monotonic: a table that turns back on itself
// produces an index that is merely ambiguous rather than one that is wrong, and
// the round-trip check is what refuses it. A bin further than `tolerance` from
// every entry is a direction the sweep never covered and gets -1.
void stampIndex(const std::vector<std::pair<double, int32_t>>& sorted,
                double lo, double bin, double tolerance,
                std::vector<int32_t>& out) {
    if (sorted.empty()) { std::fill(out.begin(), out.end(), -1); return; }
    size_t j = 0;
    for (size_t b = 0; b < out.size(); ++b) {
        const double v = lo + (double(b) + 0.5) * bin;
        while (j + 1 < sorted.size() && sorted[j + 1].first <= v) ++j;
        size_t best = j;
        if (j + 1 < sorted.size() &&
            std::fabs(sorted[j + 1].first - v) < std::fabs(sorted[j].first - v))
            best = j + 1;
        out[b] = (std::fabs(sorted[best].first - v) <= tolerance) ? sorted[best].second : -1;
    }
}

// The largest step between adjacent entries. A bin inside the table can be half
// of one of these from the nearest entry, so it sets the tolerance that keeps
// `stampIndex` from punching -1 into the middle of a raster whose step varies.
double maxAdjacentStep(const std::vector<double>& t) {
    double m = 0;
    for (size_t i = 1; i < t.size(); ++i) m = std::max(m, std::fabs(t[i] - t[i - 1]));
    return m;
}

// How many bins to spend. Capped so a pathological raster cannot ask for a
// gigabyte of index; at the cap the quantisation is still finer than a cell on any
// raster this tool will see.
constexpr size_t kMaxIndexBins = 1u << 22;

// Fractional position of a value in a monotonic table, extrapolating past both
// ends rather than clamping.
//
// Extrapolating is the point. The caller is bounding a brick, and a brick off the
// top of the raster has to produce a coordinate off the top of the raster — a
// clamped one would claim the brick projects onto the edge row, and `judgeBrick`
// would then read a cell that says nothing about it.
//
// `hint` comes from the reverse index, so inside the table the walk is a step or
// two. Outside it the answer is closed-form and the walk never runs.
double tableCoord(const std::vector<double>& t, int32_t hint, double v) {
    const size_t n = t.size();
    if (n < 2) return 0.0;
    const bool up = t[n - 1] >= t[0];

    // Past an end: extend the end's own step.
    const double dLo = t[1] - t[0];
    if ((up && v <= t[0]) || (!up && v >= t[0]))
        return (std::fabs(dLo) < 1e-15) ? 0.0 : (v - t[0]) / dLo;
    const double dHi = t[n - 1] - t[n - 2];
    if ((up && v >= t[n - 1]) || (!up && v <= t[n - 1]))
        return (std::fabs(dHi) < 1e-15) ? double(n - 1)
                                        : double(n - 1) + (v - t[n - 1]) / dHi;

    int64_t i = (hint >= 0 && size_t(hint) < n) ? int64_t(hint) : 0;
    i = std::clamp<int64_t>(i, 0, int64_t(n) - 2);
    auto reached = [&](int64_t k) { return up ? (v >= t[k]) : (v <= t[k]); };
    // Bounded: an unusable hint costs a bounded walk rather than a scan of the
    // whole table, and a non-monotonic table cannot spin here.
    constexpr int kWalk = 64;
    for (int s = 0; s < kWalk && i > 0 && !reached(i); ++s) --i;
    for (int s = 0; s < kWalk && i + 2 < int64_t(n) && reached(i + 1); ++s) ++i;
    const double d = t[i + 1] - t[i];
    if (std::fabs(d) < 1e-15) return double(i);
    return double(i) + (v - t[i]) / d;
}

struct Sample {
    uint32_t row, col;
    float    range;
    double   az, el;
};

// How many points to hold back for the round-trip check, and how widely to
// space them. Spread across the whole scan rather than taken from its start,
// because a mapping can be right for one band of rows and wrong for another —
// which is exactly the failure a double-covered mirror sweep produces.
constexpr uint64_t kRoundTripStride  = 37;
constexpr size_t   kRoundTripSamples = 200000;

} // namespace

Mapping uniformMapping(uint32_t rows, uint32_t cols,
                       double el0, double dElPerRow, double az0, double dAzPerCol) {
    Mapping m;
    if (rows == 0 || cols == 0) return m;
    m.elByRow.resize(rows);
    m.azByCol.resize(cols);
    for (uint32_t r = 0; r < rows; ++r) m.elByRow[r] = el0 + dElPerRow * double(r);
    for (uint32_t c = 0; c < cols; ++c) m.azByCol[c] = az0 + dAzPerCol * double(c);
    m.el0 = el0; m.dElPerRow = dElPerRow;
    m.az0 = az0; m.dAzPerCol = dAzPerCol;
    m.elResidualRad = 0.0;
    m.azResidualRad = 0.0;
    indexMapping(m);
    return m;
}

bool indexMapping(Mapping& m) {
    m.rowOfEl.clear();
    m.colOfAz.clear();
    m.monotonicEl = m.monotonicAz = false;
    m.elSpanRad = m.azSpanRad = 0;
    m.elLo = m.elBin = m.azLo = m.azBin = 0;

    const size_t rows = m.elByRow.size(), cols = m.azByCol.size();
    if (rows < 2 || cols < 2) return false;
    for (double v : m.elByRow) if (!std::isfinite(v)) return false;
    for (double v : m.azByCol) if (!std::isfinite(v)) return false;

    m.elSpanRad = m.elByRow.back() - m.elByRow.front();
    m.azSpanRad = m.azByCol.back() - m.azByCol.front();
    const double elStep = std::fabs(m.elSpanRad) / double(rows - 1);
    const double azStep = std::fabs(m.azSpanRad) / double(cols - 1);
    if (!(elStep > 0) || !(azStep > 0)) return false;

    m.monotonicEl = monotonicWithin(m.elByRow, kMonotonicSlackCells * elStep);
    m.monotonicAz = monotonicWithin(m.azByCol, kMonotonicSlackCells * azStep);

    // --- elevation ---------------------------------------------------------
    // Bins over the table's own range plus one cell at each end, so a direction
    // within rounding distance of the first or last row still finds it and one
    // genuinely off the raster falls outside the bins and is rejected.
    {
        double lo = m.elByRow[0], hi = m.elByRow[0];
        for (double v : m.elByRow) { lo = std::min(lo, v); hi = std::max(hi, v); }
        const size_t nbins = std::min<size_t>(kMaxIndexBins,
                                              (rows + 2) * kReverseBinsPerCell);
        m.elLo = lo - elStep;
        m.elBin = ((hi + elStep) - m.elLo) / double(nbins);
        if (!(m.elBin > 0)) return false;

        std::vector<std::pair<double, int32_t>> srt(rows);
        for (size_t r = 0; r < rows; ++r) srt[r] = {m.elByRow[r], int32_t(r)};
        std::sort(srt.begin(), srt.end());

        m.rowOfEl.assign(nbins, -1);
        stampIndex(srt, m.elLo, m.elBin,
                   std::max(elStep, 0.5 * maxAdjacentStep(m.elByRow)) + m.elBin,
                   m.rowOfEl);
    }

    // --- azimuth -----------------------------------------------------------
    // Bins over the whole turn, because every bearing lies on it. Which columns
    // compete for them needs care on these instruments: a sweep of 364.5 degrees
    // means the first and last ~65 columns looked at the same bearings, and if
    // both were in the index a lookup could land on either while `colCoord` — the
    // fractional version `judgeBrick` bounds a brick with — could only ever
    // report one. The two have to agree or a brick can be culled against a
    // rectangle that does not contain the cell the lookup then reads.
    //
    // So the index keeps one turn's worth, centred on the middle of the sweep, and
    // the duplicates at the ends are dropped. Both saw the same direction; which
    // of the two answers a lookup is arbitrary either way, and this way it is
    // arbitrary consistently. Their cells are still in the image and still counted
    // as rays — they are simply never the cell a direction resolves to.
    //
    // How many to keep is chosen by index rather than by clipping to a window in
    // angle, and the remainder is split between the two ends. Clipping left a seam
    // gap of one and a half steps, wide enough that a bearing in it resolved to a
    // column more than half a cell away while a dropped column sat right on it.
    // Keeping floor(2pi/step)+1 columns puts the gap under one step, and centring
    // it leaves every bearing within half a cell of the column it resolves to.
    {
        const size_t turn = size_t(std::floor(kTwoPi / azStep)) + 1;
        const size_t keep = std::max<size_t>(2, std::min(turn, cols));
        const size_t c0 = (cols - keep) / 2, c1 = c0 + keep - 1;
        const double a0 = std::min(m.azByCol[c0], m.azByCol[c1]);
        const double a1 = std::max(m.azByCol[c0], m.azByCol[c1]);
        m.azLo = a0 - 0.5 * std::max(0.0, kTwoPi - (a1 - a0));

        const size_t nbins = std::min<size_t>(kMaxIndexBins, cols * kReverseBinsPerCell);
        m.azBin = kTwoPi / double(nbins);

        // Absolute azimuths, on the same scale the bins are laid out on — the bin
        // a lookup lands in is found from an offset, but the angle it is compared
        // against is not.
        std::vector<std::pair<double, int32_t>> srt;
        srt.reserve(keep);
        for (size_t c = c0; c <= c1; ++c) srt.push_back({m.azByCol[c], int32_t(c)});
        if (srt.size() < 2) return false;
        std::sort(srt.begin(), srt.end());
        // The turn closes, so the first and last kept columns are neighbours
        // across the seam. Without these the two bins at the seam would read as a
        // gap in the sweep on a raster that has none.
        const std::pair<double, int32_t> wrapLo{srt.back().first - kTwoPi, srt.back().second};
        const std::pair<double, int32_t> wrapHi{srt.front().first + kTwoPi, srt.front().second};
        srt.insert(srt.begin(), wrapLo);
        srt.push_back(wrapHi);

        m.colOfAz.assign(nbins, -1);
        stampIndex(srt, m.azLo, m.azBin,
                   std::max(azStep, 0.5 * maxAdjacentStep(m.azByCol)) + m.azBin,
                   m.colOfAz);
    }
    return true;
}

bool RangeImage::cellOf(double az, double el, uint32_t& row, uint32_t& col) const {
    if (!map.valid || rows == 0 || cols == 0) return false;
    const int32_t r = map.rowFor(el);
    if (r < 0 || r >= int32_t(rows)) return false;
    const int32_t c = map.colFor(az);
    if (c < 0 || c >= int32_t(cols)) return false;
    row = uint32_t(r);
    col = uint32_t(c);
    return true;
}

bool RangeImage::sample(double az, double el, Status& st, double& range) const {
    uint32_t r, c;
    if (!cellOf(az, el, r, c)) { st = Status::OutsideFov; range = 0; return false; }
    st = statusAt(r, c);
    range = rangeAt(r, c);
    return true;
}

// Demotes no-returns that have too few no-return neighbours to OutsideFov.
//
// Counted with a separable sliding window, so the cost is two linear passes
// rather than one window per cell: at 13 M cells the naive form would be 330 M
// probes. Columns wrap, because azimuth does and a sky region crossing the seam
// is one region; rows clamp, because the top and bottom of the raster are real
// edges.
//
// The counts all come from the original mask. Reclassifying as it goes would
// make a cell's verdict depend on the order cells were visited, and erode a sky
// region from one side.
void filterIsolatedNoReturns(RangeImage& im, const Options& opt) {
    if (opt.noReturnRadius == 0 || im.rows == 0 || im.cols == 0) return;
    const int R    = int(opt.noReturnRadius);
    const int rows = int(im.rows), cols = int(im.cols);
    if (cols <= 2 * R || rows == 0) return;
    const size_t n = im.cells.size();

    std::vector<uint8_t> isNoReturn(n, 0);
    for (size_t i = 0; i < n; ++i)
        isNoReturn[i] = (Status(im.cells[i].status) == Status::NoReturn) ? 1u : 0u;

    // Horizontal window, wrapping in azimuth.
    std::vector<uint16_t> h(n, 0);
    const int W = 2 * R + 1;
    for (int r = 0; r < rows; ++r) {
        const uint8_t* row = &isNoReturn[size_t(r) * size_t(cols)];
        uint32_t sum = 0;
        for (int c = -R; c <= R; ++c) sum += row[((c % cols) + cols) % cols];
        uint16_t* out = &h[size_t(r) * size_t(cols)];
        for (int c = 0; c < cols; ++c) {
            out[c] = uint16_t(sum);
            sum -= row[(((c - R) % cols) + cols) % cols];
            sum += row[(((c + R + 1) % cols) + cols) % cols];
        }
    }

    // Vertical window, clamped, via a prefix sum down each column. Built in
    // row-major order so both passes stay sequential in memory.
    std::vector<uint32_t> prefix(size_t(rows + 1) * size_t(cols), 0);
    for (int r = 0; r < rows; ++r) {
        const uint16_t* src = &h[size_t(r) * size_t(cols)];
        const uint32_t* prev = &prefix[size_t(r) * size_t(cols)];
        uint32_t*       cur  = &prefix[size_t(r + 1) * size_t(cols)];
        for (int c = 0; c < cols; ++c) cur[c] = prev[c] + src[c];
    }

    uint64_t demoted = 0;
    for (int r = 0; r < rows; ++r) {
        const int lo = std::max(0, r - R);
        const int hi = std::min(rows - 1, r + R);
        const double windowCells = double(W) * double(hi - lo + 1);
        const double needed = opt.noReturnFraction * windowCells;
        const uint32_t* top = &prefix[size_t(lo) * size_t(cols)];
        const uint32_t* bot = &prefix[size_t(hi + 1) * size_t(cols)];
        for (int c = 0; c < cols; ++c) {
            const size_t i = size_t(r) * size_t(cols) + size_t(c);
            if (!isNoReturn[i]) continue;
            if (double(bot[c] - top[c]) >= needed) continue;
            // Not enough company to be sky. It says nothing rather than
            // clearing to maxRange, and its stored range goes with it so no
            // later reader mistakes it for a measurement.
            im.cells[i].status  = uint8_t(Status::OutsideFov);
            im.cells[i].rangeCm = 0;
            ++demoted;
        }
    }
    im.diag.isolatedNoReturns = demoted;
    if (demoted) {
        im.diag.noReturns  -= std::min<uint64_t>(demoted, im.diag.noReturns);
        im.diag.outsideFov += demoted;
    }
}

// Finds and marks the instrument's blind cone.
//
// The cone is an unsampled band running off one end of the raster, and which end
// is not something to assume. A scanner mounted upside down has its cone
// pointing up; a producer that rewrites the local frame so world up is +Z puts
// the cone at the opposite end of the raster from an upright scan. Assuming
// "nadir" from the sign of the elevation mapping gets both of those wrong, and
// gets them wrong in the worst direction: it believes the cone, clearing space
// to maxRange through whatever the instrument was standing on, and disbelieves
// the sky, losing the clearing that carves the volume above a site.
//
// So the end is found from the geometry. Just outside the blind cone the beam
// grazes the instrument's own mount and lands on the ground a metre or two away,
// so the returns bordering the cone are among the closest in the scan. Just
// outside a sky band they are distant, or absent. The band bordered by the
// nearer returns is the cone. Nothing in that test refers to up.
void markBlindCone(RangeImage& im, const Options& opt) {
    im.diag.blindConeRows  = 0;
    im.diag.blindConeCells = 0;
    im.diag.borderRangeFirst = -1.0;
    im.diag.borderRangeLast  = -1.0;
    im.diag.hasConeAxis = false;
    if (opt.blindCone == BlindCone::None) return;
    if (im.rows == 0 || im.cols == 0) return;

    auto rowHasReturn = [&](uint32_t r) {
        const Cell* row = &im.cells[size_t(r) * im.cols];
        for (uint32_t c = 0; c < im.cols; ++c)
            if (Status(row[c].status) == Status::Hit) return true;
        return false;
    };

    // The contiguous empty bands running off each end.
    uint32_t bandFirst = 0;
    while (bandFirst < im.rows && !rowHasReturn(bandFirst)) ++bandFirst;
    uint32_t bandLast = 0;
    while (bandLast + bandFirst < im.rows && !rowHasReturn(im.rows - 1 - bandLast)) ++bandLast;
    // A raster with no returns at all is a different problem, and swallowing it
    // whole as a blind cone would hide it.
    if (bandFirst + bandLast >= im.rows) return;

    // The median range of the returns bordering a band. Median rather than mean
    // because a single stray long return past the mount would drag a mean.
    auto borderMedian = [&](uint32_t band, bool fromFirst) -> double {
        if (band == 0) return -1.0;
        std::vector<uint16_t> cm;
        uint32_t taken = 0;
        for (uint32_t i = 0; i < opt.blindConeProbeRows && taken < opt.blindConeProbeRows; ++i) {
            const int64_t r = fromFirst ? int64_t(band) + i
                                        : int64_t(im.rows) - 1 - int64_t(band) - int64_t(i);
            if (r < 0 || r >= int64_t(im.rows)) break;
            const Cell* row = &im.cells[size_t(r) * im.cols];
            bool any = false;
            for (uint32_t c = 0; c < im.cols; ++c)
                if (Status(row[c].status) == Status::Hit) { cm.push_back(row[c].rangeCm); any = true; }
            if (any) ++taken;
        }
        if (cm.empty()) return -1.0;
        std::nth_element(cm.begin(), cm.begin() + ptrdiff_t(cm.size() / 2), cm.end());
        return double(cm[cm.size() / 2]) * 0.01;
    };

    const double mFirst = borderMedian(bandFirst, true);
    const double mLast  = borderMedian(bandLast, false);
    im.diag.borderRangeFirst = mFirst;
    im.diag.borderRangeLast  = mLast;

    bool atFirst = false;
    uint32_t band = 0;
    std::string why;

    if (opt.blindCone == BlindCone::FirstRows) {
        atFirst = true;  band = bandFirst;  why = "forced to the first rows";
    } else if (opt.blindCone == BlindCone::LastRows) {
        atFirst = false; band = bandLast;   why = "forced to the last rows";
    } else if (bandFirst && bandLast) {
        // Both ends are empty, which is the ordinary outdoor case: sky at one
        // end, the mount at the other. Compare like with like within the one
        // scan — no absolute threshold needed, and no notion of up.
        if (mFirst < 0 || mLast < 0) return;
        const double near_ = std::min(mFirst, mLast), far_ = std::max(mFirst, mLast);
        if (near_ > opt.blindConeRatio * far_) {
            // Too close to call. Refusing is not neutral — it leaves the cone
            // clearing through the ground — but guessing risks the same error
            // silently, and this way it is reported.
            char buf[420];
            std::snprintf(buf, sizeof(buf),
                          "both ends of the raster are unsampled (%u and %u rows) and the "
                          "returns bordering them are at %.2f m and %.2f m — too similar to "
                          "tell the instrument's blind cone from sky, so neither is treated "
                          "as unsampled. Set the blind cone explicitly if you know it",
                          bandFirst, bandLast, mFirst, mLast);
            if (!im.diag.note.empty()) im.diag.note += "; ";
            im.diag.note += buf;
            return;
        }
        atFirst = (mFirst < mLast);
        band    = atFirst ? bandFirst : bandLast;
    } else if (bandFirst || bandLast) {
        // Only one end is empty. It is the cone only if the returns bordering it
        // are much closer than the scan's returns generally; otherwise it is a
        // field of view that simply stops, or sky.
        atFirst = (bandFirst != 0);
        band    = atFirst ? bandFirst : bandLast;
        const double m = atFirst ? mFirst : mLast;
        if (m < 0) return;
        double overall = im.diag.nearestReturn > 0 ? im.diag.furthestReturn : 0.0;
        if (overall <= 0) return;
        // Half the scan's furthest return is a generous bar: a mount's ground
        // ring is metres where a site is tens of metres.
        if (m > opt.blindConeRatio * 0.5 * overall) return;
    } else {
        return;      // no unsampled band at either end
    }
    if (band == 0 || band >= im.rows) return;

    for (uint32_t i = 0; i < band; ++i) {
        const uint32_t r = atFirst ? i : (im.rows - 1 - i);
        Cell* row = &im.cells[size_t(r) * im.cols];
        for (uint32_t c = 0; c < im.cols; ++c) {
            if (Status(row[c].status) == Status::NoReturn) {
                row[c].status  = uint8_t(Status::OutsideFov);
                row[c].rangeCm = 0;
                ++im.diag.blindConeCells;
            }
        }
    }
    im.diag.blindConeRows       = band;
    im.diag.blindConeAtFirstRow = atFirst;
    if (im.diag.blindConeCells) {
        im.diag.noReturns  -= std::min<uint64_t>(im.diag.blindConeCells, im.diag.noReturns);
        im.diag.outsideFov += im.diag.blindConeCells;
    }

    // Which way the instrument was actually pointing. The cone sits at the end
    // of the elevation sweep that this band occupies, so its axis is the local
    // pole on that side; the pose then says where that points in the file's
    // frame. A negative z is an upright scanner and a positive one is inverted,
    // and that is worth reporting rather than merely handling.
    if (im.map.elByRow.size() == im.rows && im.rows > 0) {
        const double elAtBand = atFirst ? im.map.elByRow.front() : im.map.elByRow.back();
        double axis[3] = {0, 0, elAtBand < 0 ? -1.0 : 1.0};
        if (im.hasPose) {
            const viewer::Rigid R = viewer::rigidFromPose(im.pose);
            const double a = axis[0], b = axis[1], c = axis[2];
            axis[0] = R.R[0] * a + R.R[1] * b + R.R[2] * c;
            axis[1] = R.R[3] * a + R.R[4] * b + R.R[5] * c;
            axis[2] = R.R[6] * a + R.R[7] * b + R.R[8] * c;
        }
        for (int k = 0; k < 3; ++k) im.diag.coneAxisWorld[k] = axis[k];
        im.diag.hasConeAxis = true;
    }
}

double RangeImage::rowCoord(double el) const {
    return tableCoord(map.elByRow, map.rowFor(el), el);
}

double RangeImage::colCoord(double az) const {
    if (map.azByCol.size() < 2) return 0.0;
    // Onto the same turn the reverse index covers, so that this and `colFor`
    // cannot disagree about which of two columns looking the same way is meant.
    double v = az - map.azLo;
    if (!std::isfinite(v)) return 0.0;
    v -= kTwoPi * std::floor(v / kTwoPi);
    v += map.azLo;
    return tableCoord(map.azByCol, map.colFor(az), v);
}

void buildPyramid(RangeImage& im) {
    im.pyramid = RangePyramid{};
    if (im.rows == 0 || im.cols == 0 || im.cells.empty()) return;

    auto statusBit = [](uint8_t s) -> uint8_t {
        switch (Status(s)) {
        case Status::Hit:        return kHasHit;
        case Status::NoReturn:   return kHasNoReturn;
        case Status::OutsideFov: return kHasOutsideFov;
        }
        return kHasOutsideFov;
    };

    // Level 0: kPyramidBase x kPyramidBase cells per node.
    PyramidLevel l0;
    l0.block = kPyramidBase;
    l0.rows  = (im.rows + kPyramidBase - 1) / kPyramidBase;
    l0.cols  = (im.cols + kPyramidBase - 1) / kPyramidBase;
    l0.minCm.assign(size_t(l0.rows) * l0.cols, 0xFFFF);
    l0.maxCm.assign(size_t(l0.rows) * l0.cols, 0);
    l0.statuses.assign(size_t(l0.rows) * l0.cols, 0);
    for (uint32_t r = 0; r < im.rows; ++r) {
        const size_t nodeRow = size_t(r / kPyramidBase) * l0.cols;
        const Cell*  src     = &im.cells[size_t(r) * im.cols];
        for (uint32_t c = 0; c < im.cols; ++c) {
            const size_t n = nodeRow + c / kPyramidBase;
            const uint16_t v = src[c].rangeCm;
            if (v < l0.minCm[n]) l0.minCm[n] = v;
            if (v > l0.maxCm[n]) l0.maxCm[n] = v;
            l0.statuses[n] |= statusBit(src[c].status);
        }
    }
    im.pyramid.levels.push_back(std::move(l0));

    // Each level after halves both axes. Stop at a single node.
    while (im.pyramid.levels.back().rows > 1 || im.pyramid.levels.back().cols > 1) {
        const PyramidLevel& prev = im.pyramid.levels.back();
        PyramidLevel lv;
        lv.block = prev.block * 2;
        lv.rows  = (prev.rows + 1) / 2;
        lv.cols  = (prev.cols + 1) / 2;
        lv.minCm.assign(size_t(lv.rows) * lv.cols, 0xFFFF);
        lv.maxCm.assign(size_t(lv.rows) * lv.cols, 0);
        lv.statuses.assign(size_t(lv.rows) * lv.cols, 0);
        for (uint32_t r = 0; r < prev.rows; ++r) {
            for (uint32_t c = 0; c < prev.cols; ++c) {
                const size_t src = size_t(r) * prev.cols + c;
                const size_t dst = size_t(r / 2) * lv.cols + (c / 2);
                if (prev.minCm[src] < lv.minCm[dst]) lv.minCm[dst] = prev.minCm[src];
                if (prev.maxCm[src] > lv.maxCm[dst]) lv.maxCm[dst] = prev.maxCm[src];
                lv.statuses[dst] |= prev.statuses[src];
            }
        }
        im.pyramid.levels.push_back(std::move(lv));
    }
}

RangeSpan RangeImage::span(int64_t row0, int64_t row1, int64_t col0, int64_t col1) const {
    RangeSpan out;
    if (pyramid.empty() || rows == 0 || cols == 0) return out;
    if (row1 < row0 || col1 < col0) return out;

    // Rows outside the raster contribute nothing to any lookup, so clamping is
    // not a loss of information — a voxel projecting there gets OutsideFov,
    // which is what a caller reading `statuses` has to allow for anyway.
    row0 = std::max<int64_t>(row0, 0);
    row1 = std::min<int64_t>(row1, int64_t(rows) - 1);
    if (row1 < row0) return out;

    // Columns wrap. Rather than split the query, a range that crosses the seam
    // widens to the whole circle: it is a superset, so still conservative, and
    // it only costs precision for bricks lying along one particular bearing.
    bool allCols = false;
    if (col1 - col0 + 1 >= int64_t(cols)) allCols = true;
    else {
        int64_t lo = col0 % int64_t(cols);
        if (lo < 0) lo += int64_t(cols);
        const int64_t hi = lo + (col1 - col0);
        if (hi >= int64_t(cols)) allCols = true;
        else { col0 = lo; col1 = hi; }
    }

    // The coarsest level at which the rectangle spans few enough nodes to read
    // them all. Reading a superset of the rectangle is safe, so a level that is
    // too coarse costs sharpness, never correctness.
    size_t L = 0;
    for (; L + 1 < pyramid.levels.size(); ++L) {
        const PyramidLevel& lv = pyramid.levels[L];
        const int64_t rSpan = row1 / lv.block - row0 / lv.block + 1;
        const int64_t cSpan = allCols ? int64_t(lv.cols)
                                      : col1 / lv.block - col0 / lv.block + 1;
        if (rSpan <= 3 && cSpan <= 3) break;
    }

    const PyramidLevel& lv = pyramid.levels[L];
    const int64_t r0 = std::min<int64_t>(row0 / lv.block, int64_t(lv.rows) - 1);
    const int64_t r1 = std::min<int64_t>(row1 / lv.block, int64_t(lv.rows) - 1);
    const int64_t c0 = allCols ? 0 : std::min<int64_t>(col0 / lv.block, int64_t(lv.cols) - 1);
    const int64_t c1 = allCols ? int64_t(lv.cols) - 1
                               : std::min<int64_t>(col1 / lv.block, int64_t(lv.cols) - 1);

    uint16_t lo = 0xFFFF, hi = 0;
    uint8_t  st = 0;
    for (int64_t r = r0; r <= r1; ++r) {
        const size_t base = size_t(r) * lv.cols;
        for (int64_t c = c0; c <= c1; ++c) {
            const size_t n = base + size_t(c);
            if (lv.minCm[n] < lo) lo = lv.minCm[n];
            if (lv.maxCm[n] > hi) hi = lv.maxCm[n];
            st |= lv.statuses[n];
        }
    }
    if (lo > hi) return out;              // no populated node in range

    out.minRange = double(lo) * 0.01;
    out.maxRange = double(hi) * 0.01;
    out.statuses = st;
    out.valid = true;
    return out;
}

bool build(e57::Reader& reader, size_t scanIndex, const Options& opt,
           RangeImage& out, std::string& err) {
    if (scanIndex >= reader.scanCount()) { err = "scan index out of range"; return false; }
    const e57::Scan& s = reader.scan(scanIndex);

    out = RangeImage{};
    out.pose    = s.pose;
    out.hasPose = s.hasPose;

    const bool cartesian = s.field("cartesianX") && s.field("cartesianY") && s.field("cartesianZ");
    const bool spherical = s.field("sphericalRange") && s.field("sphericalAzimuth") &&
                           s.field("sphericalElevation");
    if (!cartesian && !spherical) { err = "scan has no position fields"; return false; }

    const bool hasRowCol = s.field("rowIndex") && s.field("columnIndex");
    const bool gridPath  = hasRowCol && s.hasIndexBounds &&
                           s.rowMax > s.rowMin && s.colMax > s.colMin;

    std::vector<std::string> want;
    if (cartesian) want = {"cartesianX", "cartesianY", "cartesianZ"};
    else           want = {"sphericalRange", "sphericalAzimuth", "sphericalElevation"};

    size_t invIdx = SIZE_MAX;
    const char* invName = nullptr;
    if      (s.field("cartesianInvalidState")) invName = "cartesianInvalidState";
    else if (s.field("sphericalInvalidState")) invName = "sphericalInvalidState";
    if (invName) { invIdx = want.size(); want.push_back(invName); }

    size_t rowIdx = SIZE_MAX, colIdx = SIZE_MAX;
    if (gridPath) {
        rowIdx = want.size(); want.push_back("rowIndex");
        colIdx = want.size(); want.push_back("columnIndex");
    }

    // Ranges are measured in the SCANNER's frame, so points must not have the
    // pose applied. Conformant files already store them that way; the
    // non-conformant "pre-transformed" case has to be undone.
    const viewer::FrameDecision fd = viewer::decideFrame(reader, scanIndex);
    const bool subtractPose = !fd.applyPose() && s.hasPose && cartesian;
    // The undo is the full inverse pose, rotation included: p_local =
    // R^T (p_world - t). Subtracting only the translation would leave the points
    // on world-aligned axes about the scanner, which is not the frame `pose`
    // describes — and the carve builds its world-to-scanner transform from
    // `pose`, so the two have to mean the same thing.
    const viewer::Rigid poseRot = viewer::rigidFromPose(s.pose);

    // Grid dimensions.
    uint32_t gridRows = 0, gridCols = 0;
    int64_t  rowMin = 0, colMin = 0;
    if (gridPath) {
        // indexBounds is a declaration, not a fact. A negative, absurd or
        // overflowing span would produce a garbage raster that then reads as
        // mostly empty — and mostly empty reads as no-returns, which clears
        // space to maxRange. Checked before it is trusted.
        const int64_t spanR = s.rowMax - s.rowMin + 1;
        const int64_t spanC = s.colMax - s.colMin + 1;
        constexpr int64_t kMaxSpan = 1 << 20;          // a million rows or columns
        constexpr int64_t kMaxCells = int64_t(1) << 32;
        if (spanR <= 0 || spanC <= 0 || spanR > kMaxSpan || spanC > kMaxSpan ||
            spanR * spanC > kMaxCells) {
            char buf[220];
            std::snprintf(buf, sizeof(buf),
                          "indexBounds declares a %lld x %lld grid, which is not a raster this "
                          "scan could have produced",
                          (long long)spanR, (long long)spanC);
            err = buf;
            return false;
        }
        rowMin   = s.rowMin;  colMin = s.colMin;
        gridRows = uint32_t(spanR);
        gridCols = uint32_t(spanC);
    }

    // Cells are capped by binning down; minimum range per bin keeps it
    // conservative (see Options::maxCells).
    uint32_t step = 1;
    if (gridPath) {
        while ((uint64_t(gridRows + step - 1) / step) * (uint64_t(gridCols + step - 1) / step)
               > opt.maxCells) ++step;
    }

    std::vector<Sample> samples;   // for the angular fallback and the mapping fit
    std::vector<Accum>  rowAcc, colAcc;
    uint64_t decoded = 0, invalid = 0;
    double   farthest = 0, nearest = 1e300;

    if (gridPath) {
        out.rows = (gridRows + step - 1) / step;
        out.cols = (gridCols + step - 1) / step;
        out.cells.assign(out.cellCount(), Cell{});
        rowAcc.assign(out.rows, Accum{});
        colAcc.assign(out.cols, Accum{});
    }

    std::string rerr;
    const bool ok = reader.readPoints(scanIndex, want, [&](const e57::PointBlock& b) {
        for (size_t k = 0; k < b.count; ++k) {
            if (invIdx != SIZE_MAX && b.columns[invIdx][k] != 0.0) { ++invalid; continue; }

            double x, y, z;
            if (cartesian) {
                x = b.columns[0][k]; y = b.columns[1][k]; z = b.columns[2][k];
                if (subtractPose) {
                    const double a = x - s.pose.t[0], bb = y - s.pose.t[1], c = z - s.pose.t[2];
                    x = poseRot.R[0] * a + poseRot.R[3] * bb + poseRot.R[6] * c;
                    y = poseRot.R[1] * a + poseRot.R[4] * bb + poseRot.R[7] * c;
                    z = poseRot.R[2] * a + poseRot.R[5] * bb + poseRot.R[8] * c;
                }
            } else {
                const double r = b.columns[0][k], a = b.columns[1][k], e = b.columns[2][k];
                const double ce = std::cos(e);
                x = r * ce * std::cos(a); y = r * ce * std::sin(a); z = r * std::sin(e);
            }
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;

            double az, el, range;
            toSpherical(x, y, z, az, el, range);
            if (range <= 1e-6) continue;
            farthest = std::max(farthest, range);
            nearest  = std::min(nearest, range);
            if (!out.diag.hasReturnBounds) {
                out.diag.hasReturnBounds = true;
                out.diag.returnMin[0] = out.diag.returnMax[0] = x;
                out.diag.returnMin[1] = out.diag.returnMax[1] = y;
                out.diag.returnMin[2] = out.diag.returnMax[2] = z;
            } else {
                const double v[3] = {x, y, z};
                for (int a = 0; a < 3; ++a) {
                    out.diag.returnMin[a] = std::min(out.diag.returnMin[a], v[a]);
                    out.diag.returnMax[a] = std::max(out.diag.returnMax[a], v[a]);
                }
            }
            ++decoded;

            if (!gridPath) {
                samples.push_back({0, 0, float(range), az, el});
                continue;
            }

            const int64_t rr = int64_t(b.columns[rowIdx][k]) - rowMin;
            const int64_t cc = int64_t(b.columns[colIdx][k]) - colMin;
            if (rr < 0 || cc < 0 || rr >= int64_t(gridRows) || cc >= int64_t(gridCols)) {
                ++out.diag.outsideGrid;
                continue;
            }

            const uint32_t r = uint32_t(rr) / step;
            const uint32_t c = uint32_t(cc) / step;
            const size_t   i = size_t(r) * out.cols + c;

            // Every so often, remember where this point came from. Once the
            // mapping is fitted these get put back through it: a model of the
            // raster that cannot reproduce the raster's own points is not a
            // model of anything.
            if ((decoded % kRoundTripStride) == 0 && samples.size() < kRoundTripSamples)
                samples.push_back({r, c, float(range), az, el});

            const uint32_t cm = uint32_t(std::min(range * 100.0, 65535.0));
            // Minimum range wins: several source cells can land in one binned
            // cell, and clearing to the nearest of them never over-clears.
            if (out.cells[i].status != uint8_t(Status::Hit) || cm < out.cells[i].rangeCm) {
                out.cells[i].rangeCm = uint16_t(cm);
            }
            out.cells[i].status = uint8_t(Status::Hit);

            Accum& ra = rowAcc[r];
            ra.sumEl += el; ++ra.n;
            Accum& ca = colAcc[c];
            ca.sumAzX += std::cos(az); ca.sumAzY += std::sin(az); ++ca.n;
        }
        return true;
    }, rerr);
    if (!ok) { err = rerr; return false; }
    if (decoded == 0) { err = "no valid points decoded"; return false; }

    // A scan whose points do not fit the grid it declares is not a scan this
    // can read. The cells they should have filled stay empty, and empty reads as
    // a ray that came back with nothing — so a stale or wrong indexBounds does
    // not produce a sparse image, it produces one that clears the whole site.
    if (decoded > 0 && out.diag.outsideGrid * 100 > decoded) {
        char buf[260];
        std::snprintf(buf, sizeof(buf),
                      "%llu of %llu points fall outside the grid indexBounds declares "
                      "(%.1f%%); their cells would stay empty and read as clear space",
                      (unsigned long long)out.diag.outsideGrid,
                      (unsigned long long)(decoded + out.diag.outsideGrid),
                      100.0 * double(out.diag.outsideGrid) /
                          double(decoded + out.diag.outsideGrid));
        err = buf;
        return false;
    }

    // The instrument stands among its own returns: in the scanner's frame they
    // surround the origin. When the origin sits outside their box the points are
    // not in the frame they are being read as, which puts every subsequent
    // lookup in the wrong place. Reported rather than refused, because a scan
    // from inside a corner can legitimately be lopsided.
    if (out.diag.hasReturnBounds) {
        bool inside = true;
        for (int k = 0; k < 3; ++k)
            if (out.diag.returnMin[k] > 0.0 || out.diag.returnMax[k] < 0.0) inside = false;
        out.diag.originInsideReturns = inside;
        if (!inside) {
            char buf[300];
            std::snprintf(buf, sizeof(buf),
                          "the scanner's own position is outside the box of its returns "
                          "(x[%.1f,%.1f] y[%.1f,%.1f] z[%.1f,%.1f]) — these points may not be "
                          "in the frame they are being read as",
                          out.diag.returnMin[0], out.diag.returnMax[0],
                          out.diag.returnMin[1], out.diag.returnMax[1],
                          out.diag.returnMin[2], out.diag.returnMax[2]);
            if (!out.diag.note.empty()) out.diag.note += "; ";
            out.diag.note += buf;
        }
    }

    out.diag.usedGrid          = gridPath;
    out.diag.nearestReturn     = (nearest < 1e299) ? nearest : 0.0;
    out.diag.furthestReturn    = farthest;

    if (!gridPath) {
        // Angular fallback: bin by direction, and take the field of view from
        // the extent of the returns — which is exactly the assumption that
        // fails for an all-sky band. Recorded in the note.
        err = "scan has no indexBounds with rowIndex/columnIndex; the angular "
              "fallback is not implemented yet — see range_image.h";
        out.diag.note = "no grid metadata: no-return rays cannot be identified reliably";
        return false;
    }

    // Measure the angular mapping: one elevation per row, one azimuth per column.
    std::vector<uint64_t> rowN(out.rows, 0), colN(out.cols, 0);
    out.map.elByRow.assign(out.rows, 0.0);
    out.map.azByCol.assign(out.cols, 0.0);
    for (uint32_t r = 0; r < out.rows; ++r) {
        rowN[r] = rowAcc[r].n;
        if (rowAcc[r].n) out.map.elByRow[r] = rowAcc[r].sumEl / double(rowAcc[r].n);
    }
    for (uint32_t c = 0; c < out.cols; ++c) {
        colN[c] = colAcc[c].n;
        if (colAcc[c].n) out.map.azByCol[c] = std::atan2(colAcc[c].sumAzY, colAcc[c].sumAzX);
    }
    // Azimuth comes back from atan2 on one turn, so unwrap the measured sequence
    // into a continuous sweep. This is where a sweep past 2pi becomes visible
    // rather than folding back on itself: the table simply keeps climbing.
    double prev = 0; bool havePrev = false;
    for (uint32_t c = 0; c < out.cols; ++c) {
        if (!colN[c]) continue;
        if (!havePrev) { havePrev = true; prev = out.map.azByCol[c]; continue; }
        out.map.azByCol[c] = prev + angleDiff(out.map.azByCol[c], prev);
        prev = out.map.azByCol[c];
    }

    // The line through each table. Reported only — see Mapping — but fitted
    // before the gaps are filled, so the residual describes what was measured
    // rather than what was interpolated.
    double a, b, res;
    if (fitLine(out.map.elByRow, rowN, a, b, res)) {
        out.map.el0 = a; out.map.dElPerRow = b; out.map.elResidualRad = res;
    }
    if (fitLine(out.map.azByCol, colN, a, b, res)) {
        out.map.az0 = a; out.map.dAzPerCol = b; out.map.azResidualRad = res;
    }

    const bool filled = fillTable(out.map.elByRow, rowN) &&
                        fillTable(out.map.azByCol, colN) &&
                        indexMapping(out.map);

    // The check that matters: put the scan's own points back through the mapping
    // and see whether they land where they came from.
    //
    // Computed before the verdict rather than after it, and through the mapping's
    // own lookups rather than a copy of them, so the number reported is the number
    // the carve will get. A diagnostic that vanishes exactly when something is
    // wrong is no use to anyone.
    if (filled && !samples.empty()) {
        // A sweep past a full turn looks at some bearings twice, and only one of
        // the two columns is in the index. A point recorded in the other one is
        // not misplaced — its bearing resolved to a column that looked the same
        // way — so the column difference is reduced modulo a whole turn before it
        // is judged.
        const double azStep = std::fabs(out.map.azSpanRad) / double(out.cols - 1);
        const double colsPerTurn = (azStep > 1e-12) ? kTwoPi / azStep : double(out.cols);
        uint64_t landed = 0;
        for (const Sample& sm : samples) {
            const int32_t ri = out.map.rowFor(sm.el);
            const int32_t ci = out.map.colFor(sm.az);
            if (ri < 0 || ci < 0) continue;
            // One cell of slack on each axis: a direction on a cell boundary can
            // legitimately round either way, and binning down puts several source
            // cells into one.
            const long dr = long(ri) - long(sm.row);
            double dc = double(ci) - double(sm.col);
            dc -= colsPerTurn * std::round(dc / colsPerTurn);
            if (dr >= -1 && dr <= 1 && std::fabs(dc) <= 1.0) ++landed;
        }
        out.map.roundTripFraction = double(landed) / double(samples.size());
    }

    out.map.valid = filled && out.map.monotonicEl && out.map.monotonicAz &&
                    out.map.roundTripFraction >= opt.minRoundTripFraction;

    if (!out.map.valid) {
        char buf[460];
        if (!filled) {
            std::snprintf(buf, sizeof(buf),
                          "the angular mapping could not be measured: too few rows or "
                          "columns hold returns to say where the raster points");
        } else if (!out.map.monotonicEl || !out.map.monotonicAz) {
            // A mirror that sweeps past the pole sends elevation back down and
            // flips azimuth by pi at the same time. No pair of separable tables
            // can express that, and indexing it anyway would send half the scan's
            // lookups to the other half of the raster.
            std::snprintf(buf, sizeof(buf),
                          "the raster turns back on itself (%s is not monotonic over "
                          "%u rows / %u cols, spans %.1f deg row and %.1f deg col) — a "
                          "mirror sweeping past the pole, which a row/column mapping "
                          "cannot describe. This scan contributes no evidence",
                          !out.map.monotonicEl ? "elevation" : "azimuth",
                          out.rows, out.cols,
                          out.map.elSpanRad * 57.29577951308232,
                          out.map.azSpanRad * 57.29577951308232);
        } else {
            std::snprintf(buf, sizeof(buf),
                          "the measured angular mapping does not describe this raster: only "
                          "%.1f%% of the scan's own points land back on their own cell when "
                          "put through it. Lookups would reach the wrong direction — sky "
                          "read as ground, and building interiors read as clear space",
                          100.0 * std::max(0.0, out.map.roundTripFraction));
        }
        out.diag.note = buf;
    } else if (out.map.elResidualRad > opt.maxMappingResidualRad ||
               out.map.azResidualRad > opt.maxMappingResidualRad) {
        // Accepted, and worth saying why it would not have been before: the
        // tables describe it, a line did not.
        char buf[300];
        std::snprintf(buf, sizeof(buf),
                      "this is not a uniform raster — the measured mapping deviates from a "
                      "straight line by %.0f rows and %.0f cols — so the measured tables "
                      "are used directly rather than a fitted line",
                      out.map.elResidualRad / std::max(1e-12, std::fabs(out.map.dElPerRow)),
                      out.map.azResidualRad / std::max(1e-12, std::fabs(out.map.dAzPerCol)));
        out.diag.note = buf;
    }

    // Rows with no returns at all, at either end of the grid. Left as
    // no-returns — the declared grid is the field of view — but counted, since
    // a band that is really outside the field of view would look the same and
    // only the corpus can say which.
    uint32_t lead = 0, trail = 0;
    while (lead < out.rows && rowN[lead] == 0) ++lead;
    while (trail < out.rows - lead && rowN[out.rows - 1 - trail] == 0) ++trail;
    out.diag.emptyLeadingRows  = lead;
    out.diag.emptyTrailingRows = trail;

    for (size_t i = 0; i < out.cells.size(); ++i) {
        if (out.cells[i].status == uint8_t(Status::Hit)) { ++out.diag.hits; continue; }
        // Every cell the grid declares was sampled, so an empty one is a ray
        // that came back nothing. It clears to maxRange.
        out.cells[i].status  = uint8_t(Status::NoReturn);
        out.cells[i].rangeCm = uint16_t(std::min(opt.maxRange * 100.0, 65535.0));
        ++out.diag.noReturns;
    }
    // The blind cone under the tripod: never sampled, so it establishes nothing.
    // Everything else empty is a ray that was fired and came back with nothing,
    // and clears along its path — whatever it passed through on the way.
    markBlindCone(out, opt);
    // Off by default; see rimg::Options.
    filterIsolatedNoReturns(out, opt);

    out.diag.fillFraction = double(out.diag.hits) / double(out.cellCount());
    if (out.diag.blindConeRows) {
        const double here  = out.diag.blindConeAtFirstRow ? out.diag.borderRangeFirst
                                                          : out.diag.borderRangeLast;
        const double there = out.diag.blindConeAtFirstRow ? out.diag.borderRangeLast
                                                          : out.diag.borderRangeFirst;
        char other[64];
        if (there >= 0) std::snprintf(other, sizeof(other), "%.2f m at the other end", there);
        else            std::snprintf(other, sizeof(other), "no unsampled band at the other end");
        char buf[340];
        std::snprintf(buf, sizeof(buf),
                      "%u unsampled rows at the %s of the raster (%llu cells) are the "
                      "instrument's blind cone and clear nothing; the returns bordering it "
                      "are at %.2f m against %s%s",
                      out.diag.blindConeRows,
                      out.diag.blindConeAtFirstRow ? "start" : "end",
                      (unsigned long long)out.diag.blindConeCells, here, other,
                      (out.diag.hasConeAxis && out.diag.coneAxisWorld[2] > 0.5)
                          ? " — it points UP, so this setup was inverted" : "");
        if (!out.diag.note.empty()) out.diag.note += "; ";
        out.diag.note += buf;
    }
    if (out.diag.isolatedNoReturns) {
        char buf[220];
        std::snprintf(buf, sizeof(buf),
                      "%llu of %llu empty cells had too few empty neighbours to be sky "
                      "and were treated as unsampled rather than as clear space",
                      (unsigned long long)out.diag.isolatedNoReturns,
                      (unsigned long long)(out.diag.noReturns + out.diag.isolatedNoReturns));
        if (!out.diag.note.empty()) out.diag.note += "; ";
        out.diag.note += buf;
    }
    return true;
}

} // namespace rimg
