#!/usr/bin/env python3
"""Replica validation of the Metal carve kernel against the CPU reference.

There is no Metal toolchain in the development environment, so CLAUDE.md's rule
applies: a new kernel is validated by a Python replica against the CPU reference
before it ships, and the first real Metal run happens on the target Mac.

This replicates both sides of app/CarveGpu.mm's `carveVoxels` and
src/carve.cpp's `evidenceAt`, line for line, in float32 and float64
respectively, and reports where they disagree. It cannot prove the MSL compiles
or that the buffer layouts line up — only that the arithmetic the kernel
performs reaches the same conclusion as the arithmetic the reference performs.

What it is looking for is not "zero disagreements". Metal has no double, so
exact agreement is impossible and claiming it would be the failure mode this
whole exercise exists to avoid. What has to hold is that every disagreement sits
against a decision boundary — a voxel whose projection falls on a cell edge, or
whose range sits on the surface margin — and that the count of such voxels is a
vanishing fraction. A disagreement anywhere else means the replica of the kernel
has a real bug in it.

The raster here is the one real instruments produce, not a tidy one:

  Azimuth sweeps 364.5 degrees over the columns, so the last few repeat the
  bearings of the first few. This is what broke the old arithmetic inversion —
  folding modulo 2pi and then wrapping modulo cols shifted every column past the
  seam by the excess — and it is the case a replica of a table lookup has to be
  exercised on, because it is the case the table exists for.

  Elevation carries a full-period sinusoid on its step, ten cells from a
  straight line, so neither axis is invertible by arithmetic at all.

Usage:  python3 tools/validate_carve_kernel.py [trials]
"""

import bisect
import math
import random
import sys

import struct


def f32(x):
    """Round a Python float to float32, the way the GPU stores it."""
    return struct.unpack("f", struct.pack("f", float(x)))[0]


HIT, NO_RETURN, OUTSIDE_FOV = 0, 1, 2
VISIBLE, OCCUPIED, REACHABLE = 1, 2, 4
TAU = 6.28318530717958648
PI = 3.14159265358979323846

# src/range_image.h: kReverseBinsPerCell.
BINS_PER_CELL = 8


class Raster:
    """A range image with a measured (row, col) -> (el, az) mapping.

    Mirrors rimg::indexMapping: the tables are the mapping, and the reverse index
    stamps each bin with the nearest entry. Both sides of the comparison read the
    same index — the question is whether float32 lands in the same bin as float64,
    not whether the index is right, which is tests/test_range_image.cpp's job.
    """

    def __init__(self, rows, cols, rng):
        self.rows, self.cols = rows, cols

        # Elevation: a step with a full-period sinusoid on it, monotonic.
        el_lo, el_step = -0.8, 1.6 / rows
        el_amp = 10.0 * el_step
        self.el_by_row = [el_lo + el_step * r + el_amp * math.sin(TAU * r / rows)
                          for r in range(rows)]
        # Azimuth: 364.5 degrees, decreasing with column, as the real scans are.
        az_sweep = 364.5 * PI / 180.0
        self.d_az = -az_sweep / cols
        az0 = rng.uniform(-PI, PI)          # drawn once: this is one sweep, not cols of them
        self.az_by_col = [az0 + self.d_az * c for c in range(cols)]

        self._index()

        # A surface that varies smoothly, plus a scattering of no-returns, so both
        # branches of evidenceAt are exercised across a brick.
        self.range_cm = []
        self.status = []
        for r in range(rows):
            for c in range(cols):
                v = 8.0 + 4.0 * math.sin(3.0 * c / cols) + 2.0 * math.cos(5.0 * r / rows)
                self.range_cm.append(int(v * 100.0 + 0.5))
                self.status.append(NO_RETURN if rng.random() < 0.15 else HIT)

    def _stamp(self, sorted_pairs, lo, bin_w, tol, nbins):
        vals = [v for v, _ in sorted_pairs]
        out = []
        for b in range(nbins):
            v = lo + (b + 0.5) * bin_w
            j = bisect.bisect_right(vals, v) - 1
            j = max(0, min(j, len(vals) - 1))
            best = j
            if j + 1 < len(vals) and abs(vals[j + 1] - v) < abs(vals[j] - v):
                best = j + 1
            out.append(sorted_pairs[best][1] if abs(vals[best] - v) <= tol else -1)
        return out

    def _index(self):
        def max_step(t):
            return max(abs(t[i] - t[i - 1]) for i in range(1, len(t)))

        el_step = abs(self.el_by_row[-1] - self.el_by_row[0]) / (self.rows - 1)
        lo, hi = min(self.el_by_row), max(self.el_by_row)
        n = (self.rows + 2) * BINS_PER_CELL
        self.el_lo = lo - el_step
        self.el_bin = ((hi + el_step) - self.el_lo) / n
        srt = sorted((self.el_by_row[r], r) for r in range(self.rows))
        self.row_of_el = self._stamp(
            srt, self.el_lo, self.el_bin,
            max(el_step, 0.5 * max_step(self.el_by_row)) + self.el_bin, n)
        self.n_el_bins = n

        az_step = abs(self.az_by_col[-1] - self.az_by_col[0]) / (self.cols - 1)
        # One turn's worth of consecutive columns, chosen by index and centred, so
        # the seam gap stays under one step — see rimg::indexMapping.
        keep = max(2, min(int(math.floor(TAU / az_step)) + 1, self.cols))
        c0 = (self.cols - keep) // 2
        c1 = c0 + keep - 1
        a0 = min(self.az_by_col[c0], self.az_by_col[c1])
        a1 = max(self.az_by_col[c0], self.az_by_col[c1])
        self.az_lo = a0 - 0.5 * max(0.0, TAU - (a1 - a0))
        n = self.cols * BINS_PER_CELL
        self.az_bin = TAU / n
        kept = sorted((self.az_by_col[c], c) for c in range(c0, c1 + 1))
        kept = ([(kept[-1][0] - TAU, kept[-1][1])] + kept
                + [(kept[0][0] + TAU, kept[0][1])])
        self.az_kept = keep
        self.col_of_az = self._stamp(
            kept, self.az_lo, self.az_bin,
            max(az_step, 0.5 * max_step(self.az_by_col)) + self.az_bin, n)
        self.n_az_bins = n

    def cell(self, r, c):
        i = r * self.cols + c
        return self.range_cm[i], self.status[i]


def evidence_cpu(raster, world_to_scanner, origin, p, max_range, margin):
    """src/carve.cpp evidenceAt, in double."""
    R, t = world_to_scanner
    x = R[0] * p[0] + R[1] * p[1] + R[2] * p[2] + t[0]
    y = R[3] * p[0] + R[4] * p[1] + R[5] * p[2] + t[1]
    z = R[6] * p[0] + R[7] * p[1] + R[8] * p[2] + t[2]

    r = math.sqrt(x * x + y * y + z * z)
    az = math.atan2(y, x)
    if az < 0:
        az += TAU
    el = math.asin(max(-1.0, min(1.0, z / r))) if r > 1e-12 else 0.0

    if r > max_range or r < 1e-9:
        return 0

    # rimg::Mapping::rowFor
    fb = (el - raster.el_lo) / raster.el_bin
    if not fb >= 0.0:
        return 0
    b = int(fb)
    if b >= raster.n_el_bins:
        return 0
    ri = raster.row_of_el[b]

    # rimg::Mapping::colFor
    d = az - raster.az_lo
    d -= TAU * math.floor(d / TAU)
    cb = int(d / raster.az_bin)
    if cb >= raster.n_az_bins:
        cb = raster.n_az_bins - 1
    ci = raster.col_of_az[cb]

    if ri < 0 or ri >= raster.rows or ci < 0 or ci >= raster.cols:
        return 0

    cm, st = raster.cell(ri, ci)
    surface = cm * 0.01
    if st == OUTSIDE_FOV:
        return 0
    if st == NO_RETURN:
        return VISIBLE if r <= min(surface, max_range) else 0
    if r < surface - margin:
        return VISIBLE
    if r <= surface + margin:
        return OCCUPIED
    return 0


def evidence_gpu(raster, world_to_scanner, origin, p, max_range, margin):
    """app/CarveGpu.mm carveVoxels, in float32."""
    R, t = world_to_scanner
    Rf = [f32(v) for v in R]
    tf = [f32(v) for v in t]
    pf = [f32(v) for v in p]

    x = f32(f32(f32(Rf[0] * pf[0]) + f32(Rf[1] * pf[1])) + f32(f32(Rf[2] * pf[2]) + tf[0]))
    y = f32(f32(f32(Rf[3] * pf[0]) + f32(Rf[4] * pf[1])) + f32(f32(Rf[5] * pf[2]) + tf[1]))
    z = f32(f32(f32(Rf[6] * pf[0]) + f32(Rf[7] * pf[1])) + f32(f32(Rf[8] * pf[2]) + tf[2]))

    r = f32(math.sqrt(f32(f32(x * x) + f32(f32(y * y) + f32(z * z)))))
    mr = f32(max_range)
    mg = f32(margin)
    if r > mr or r < 1e-9:
        return 0

    az = f32(math.atan2(y, x))
    if az < 0:
        az = f32(az + f32(TAU))
    el = f32(math.asin(max(-1.0, min(1.0, f32(z / r)))))

    el_lo, el_bin = f32(raster.el_lo), f32(raster.el_bin)
    az_lo, az_bin = f32(raster.az_lo), f32(raster.az_bin)

    ri = -1
    fb = f32(f32(el - el_lo) / el_bin)
    if fb >= 0.0 and fb < f32(raster.n_el_bins):
        ri = raster.row_of_el[int(fb)]

    dd = f32(az - az_lo)
    dd = f32(dd - f32(f32(TAU) * math.floor(f32(dd / f32(TAU)))))
    cb = f32(dd / az_bin)
    cbi = int(cb) if (cb >= 0.0 and cb < f32(raster.n_az_bins)) else raster.n_az_bins - 1
    ci = raster.col_of_az[cbi]

    if ri < 0 or ri >= raster.rows or ci < 0 or ci >= raster.cols:
        return 0

    cm, st = raster.cell(ri, ci)
    surface = f32(cm * f32(0.01))
    if st == OUTSIDE_FOV:
        return 0
    if st == NO_RETURN:
        return VISIBLE if r <= min(surface, mr) else 0
    if r < f32(surface - mg):
        return VISIBLE
    if r <= f32(surface + mg):
        return OCCUPIED
    return 0


def near_a_boundary(raster, world_to_scanner, p, max_range, margin):
    """Is this voxel sitting on a decision edge, where a difference is expected?

    Three kinds of edge: the range sitting within a hair of the surface plus or
    minus the margin or of the rated range, the projection landing within a
    whisker of a reverse-index bin boundary, and the projection landing near the
    top or bottom of the raster where a row either resolves or does not. A
    disagreement anywhere else is a bug.
    """
    R, t = world_to_scanner
    x = R[0] * p[0] + R[1] * p[1] + R[2] * p[2] + t[0]
    y = R[3] * p[0] + R[4] * p[1] + R[5] * p[2] + t[1]
    z = R[6] * p[0] + R[7] * p[1] + R[8] * p[2] + t[2]
    r = math.sqrt(x * x + y * y + z * z)
    if r < 1e-9:
        return True
    if abs(r - max_range) < 1e-3:
        return True

    az = math.atan2(y, x)
    if az < 0:
        az += TAU
    el = math.asin(max(-1.0, min(1.0, z / r)))

    # float32 gives ~1e-7 relative precision. A bin is el_bin / az_bin wide, so
    # allow a thousand times the expected error as "on the edge".
    fb = (el - raster.el_lo) / raster.el_bin
    d = az - raster.az_lo
    d -= TAU * math.floor(d / TAU)
    cb = d / raster.az_bin
    for v in (fb, cb):
        if abs(v - math.floor(v)) < 1e-3 or abs(v - math.floor(v) - 1.0) < 1e-3:
            return True
    if fb < 1.0 or fb > raster.n_el_bins - 1.0:
        return True

    b = max(0, min(int(fb), raster.n_el_bins - 1))
    ri = raster.row_of_el[b]
    ci = raster.col_of_az[max(0, min(int(cb), raster.n_az_bins - 1))]
    if ri < 0 or ci < 0:
        return True
    cm, _ = raster.cell(ri, ci)
    surface = cm * 0.01
    for edge in (surface - margin, surface + margin, surface):
        if abs(r - edge) < 1e-3:
            return True
    return False


def main():
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
    rng = random.Random(20260908)
    raster = Raster(240, 480, rng)

    # A pose with a real rotation, so the transform is not a special case.
    yaw, pitch = 0.7, -0.3
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    # World-to-scanner: the transpose of a yaw-then-pitch rotation, with a
    # translation putting the setup somewhere off the tile origin.
    R = [cy * cp, sy * cp, -sp,
         -sy, cy, 0.0,
         cy * sp, sy * sp, cp]
    origin = [3.7, -2.1, 1.4]
    t = [-(R[0] * origin[0] + R[1] * origin[1] + R[2] * origin[2]),
         -(R[3] * origin[0] + R[4] * origin[1] + R[5] * origin[2]),
         -(R[6] * origin[0] + R[7] * origin[1] + R[8] * origin[2])]

    max_range, margin = 20.0, 0.5 * 0.05 * math.sqrt(3.0)

    # The raster the replica is exercised on is the awkward one, so say so: a
    # report that did not mention a 364.5 degree sweep would let the next reader
    # assume it had been checked on a tidy one.
    print(f"raster                 : {raster.rows} x {raster.cols}, "
          f"azimuth sweep {abs(raster.d_az) * raster.cols * 180.0 / PI:.1f} deg "
          f"({raster.n_el_bins} el bins, {raster.n_az_bins} az bins)")
    indexed = len({v for v in raster.col_of_az if v >= 0})
    print(f"  columns indexed      : {indexed} of {raster.cols}, one turn's worth "
          f"being {raster.az_kept} — the rest repeat bearings already covered")
    # Every column in the kept window has to own some bins, or a cell of the raster
    # is unreachable and the replica is not exercising what it claims to.
    if indexed != raster.az_kept:
        print("  *** columns in the kept window are missing from the index")
        return 1

    disagree = 0
    disagree_off_boundary = []
    counts = {0: 0, VISIBLE: 0, OCCUPIED: 0}
    for _ in range(trials):
        p = [rng.uniform(-18, 18), rng.uniform(-18, 18), rng.uniform(-8, 8)]
        a = evidence_cpu(raster, (R, t), origin, p, max_range, margin)
        b = evidence_gpu(raster, (R, t), origin, p, max_range, margin)
        counts[a] = counts.get(a, 0) + 1
        if a != b:
            disagree += 1
            if not near_a_boundary(raster, (R, t), p, max_range, margin):
                disagree_off_boundary.append((p, a, b))

    print(f"trials                 : {trials}")
    print(f"  none / visible / occupied : {counts.get(0,0)} / "
          f"{counts.get(VISIBLE,0)} / {counts.get(OCCUPIED,0)}")
    print(f"disagreements          : {disagree}  ({100.0*disagree/trials:.4f}%)")
    print(f"  away from a boundary : {len(disagree_off_boundary)}")
    for p, a, b in disagree_off_boundary[:10]:
        print(f"    at ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}): cpu={a} gpu={b}")

    ok = len(disagree_off_boundary) == 0 and disagree < trials * 0.01
    print("\n" + ("PASS — every disagreement is a boundary case"
                  if ok else "FAIL — the kernel replica differs where it should not"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
