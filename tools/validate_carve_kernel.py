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

Usage:  python3 tools/validate_carve_kernel.py [trials]
"""

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


class Raster:
    """A range image: a uniform raster with a surface distance per cell."""

    def __init__(self, rows, cols, rng):
        self.rows, self.cols = rows, cols
        self.az0 = rng.uniform(-PI, PI)
        self.d_az = TAU / cols
        self.el0 = -0.8
        self.d_el = 1.6 / rows
        # A surface that varies smoothly, plus a scattering of no-returns, so
        # both branches of evidenceAt are exercised across a brick.
        self.range_cm = []
        self.status = []
        for r in range(rows):
            for c in range(cols):
                v = 8.0 + 4.0 * math.sin(3.0 * c / cols) + 2.0 * math.cos(5.0 * r / rows)
                self.range_cm.append(int(v * 100.0 + 0.5))
                self.status.append(NO_RETURN if rng.random() < 0.15 else HIT)

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

    ri = round((el - raster.el0) / raster.d_el)
    if ri < 0 or ri >= raster.rows:
        return 0
    dd = az - raster.az0
    while dd > PI:
        dd -= TAU
    while dd <= -PI:
        dd += TAU
    ci = round(dd / raster.d_az) % raster.cols

    cm, st = raster.cell(int(ri), int(ci))
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

    ri = int(round(f32(f32(el - f32(raster.el0)) / f32(raster.d_el))))
    if ri < 0 or ri >= raster.rows:
        return 0
    dd = f32(az - f32(raster.az0))
    for _ in range(4):
        if dd <= f32(PI):
            break
        dd = f32(dd - f32(TAU))
    for _ in range(4):
        if dd > -f32(PI):
            break
        dd = f32(dd + f32(TAU))
    ci = int(round(f32(dd / f32(raster.d_az)))) % raster.cols
    if ci < 0:
        ci += raster.cols

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

    Two kinds of edge: the projection landing within a small fraction of a cell
    of a cell boundary, and the range sitting within a hair of the surface plus
    or minus the margin. A disagreement anywhere else is a bug.
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

    # float32 gives ~1e-7 relative precision; a cell is d_el / d_az wide. Allow
    # a thousand times the expected error as "on the edge".
    rf = (el - raster.el0) / raster.d_el
    cf = ((az - raster.az0 + PI) % TAU - PI) / raster.d_az
    for v in (rf, cf):
        if abs(v - math.floor(v) - 0.5) < 1e-3:
            return True
    if rf < 0.5 or rf > raster.rows - 0.5:
        return True

    ri = int(round(rf)) % raster.rows
    ci = int(round(cf)) % raster.cols
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
