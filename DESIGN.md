# E57 Coverage Checker — design

`e57cov` is a standalone macOS command-line tool. It reads a directory of
structured E57 scans (~1000 files), builds a voxel representation of the space
**observed** by the scanners, and reports the space that was **not** observed.

Target hardware: M4 Max, 64 GB unified memory.

### Relationship to Mjankor/CartesianCapture

None at build time — no shared code, no dependency, no linkage. That project is
a Swift/Metal iOS capture app; this is a desktop C++ batch tool consuming
third-party terrestrial scanner output.

What it does provide is prior art, referenced by name where this design leans
on it: its E57 *writer* records which details of the binary format were
confirmed against real readers and which had been reconstructed wrongly from
memory, and several of its GPU stages (Poisson reconstruction, marching cubes)
are the same operations needed here. Those are reference implementations to
**port**, in a different language, not libraries to reuse.

---

## 1. What the tool computes

Given `N` scan setups, each with a pose and a structured (gridded) point cloud:

- **VISIBLE** — space a scanner had line of sight through. A voxel is visible
  if, from at least one setup, it lies strictly nearer than the surface that
  setup saw in that direction, or nearer than `maxRange` in a direction where
  that setup got no return at all.
- **OCCUPIED** — a voxel containing at least one returned point. A scanned
  surface.
- **UNKNOWN** — everything else.

The deliverable is the subset of UNKNOWN that represents genuine coverage
failure, separated from wall material and exterior space (§6).

### Parameters

Every tunable is derived from `voxelSize` or another physical quantity, and is
documented where it is declared. Defaults:

| parameter | default | meaning |
|---|---|---|
| `voxelSize` | 0.05 m | grid spacing |
| `maxRange` | 45 m | how far a no-return ray clears |
| `surfaceMargin` | `0.5 · voxelSize · √3` | half a voxel diagonal; keeps the surface voxel out of VISIBLE |
| `maxVoidDepth` | 2.0 m (`40 · voxelSize`) | geodesic reach of the void dilation (§6) |
| `minVoidVolume` | 0.125 m³ (1000 voxels) | components smaller than this are dropped as noise |
| `brickDim` | 8 | voxels per brick edge |

---

## 2. Why the loop is inverted

The literal formulation — DDA a ray per point — is not viable and not correct:

- 1000 scans × 10–30 M points ≈ **1–3 × 10¹⁰ rays**, ~300 voxel steps each at
  0.05 m, so **10¹²–10¹³ voxel updates**.
- Worse, it aliases. At 45 m, adjacent rays are 16–90 mm apart depending on the
  scanner's angular step — wider than a voxel. Ray marching leaves unpainted
  speckle *inside* observed space, which in the positive formulation surfaces as
  **false voids**: noise in the actual deliverable.

So: a structured E57 *is* a range image. Keep it as one, and test voxels
against it rather than tracing rays through the grid.

```
for each voxel v:
    for each setup s within maxRange of v:      # from a spatial index
        (az, el, r) = to_spherical(pose_s⁻¹ · center(v))
        if status_s(az, el) == OUTSIDE_FOV: continue
        if r < D_s(az, el) - surfaceMargin:  mark VISIBLE(v); break
```

O(1) per (voxel, setup) with no traversal, and it paints the **swept solid
angle** rather than a set of infinitely thin lines, so the aliasing problem
does not arise. Point data never reaches the GPU — only range images do.

This is a *gather*: one thread owns one voxel and writes once. No atomics, no
inter-thread contention, deterministic and bit-reproducible run to run — which
matters because it makes validation against the CPU reference exact rather than
statistical.

---

## 3. Two phases

64 GB of unified memory is enough to hold every scan's range image at once
(4096 × 2048 bins, `uint16` centimetre range + 2-bit status ≈ 17 MB/scan,
× 1000 ≈ **17 GB**). That allows a clean split:

### Phase 1 — `e57cov index`: E57 → range-image cache

Decode all scans once, in parallel across the performance cores, and write a
compact cache. Purely I/O and CPU bound; this is where essentially all the wall
clock goes.

Only the fields actually needed are decoded — `cartesianX/Y/Z` (or
`sphericalRange/Azimuth/Elevation`) plus the invalid-state field. Skipping
intensity and colour is roughly a 2× saving on decode, and E57's
`CompressedVector` layout is one bytestream per field per packet, so unwanted
fields are skipped without being unpacked.

### Phase 2 — `e57cov carve`: cache → visibility grid

`mmap` the cache, wrap it with `newBufferWithBytesNoCopy` (page-aligned, zero
copy — the unified-memory payoff), run the GPU passes, write results.

Phase 2 re-runs in seconds. Sweeping `voxelSize`, `maxRange`, `maxVoidDepth` or
`minVoidVolume` no longer costs a re-read of ~150 GB of E57. For a tool whose
output is a judgement call about coverage, that iteration loop is worth more
than the raw runtime.

### Expected wall clock (M4 Max, data on internal SSD)

| stage | time |
|---|---|
| E57 read (~150 GB at ~6 GB/s) | ~25 s |
| decode (~2 × 10¹⁰ points, 12 P-cores) | ~1.5–2 min |
| carve + post-process | seconds |
| **phase 1 + 2, cold** | **~3–5 min** |
| **phase 2 re-run** | **seconds** |

If the scans live on an external enclosure or a NAS, that link becomes the
bottleneck and these numbers do not hold.

---

## 4. Range image construction

Per scan, a spherical image `D(az, el)` in the **scanner-local** frame — the
only frame in which it is a clean image. Voxel centres are transformed into
that frame; the image is never resampled.

Each bin holds a `uint16` range in centimetres (0–655 m, 1 cm quantisation, well
under a 5 cm voxel) and a 2-bit status:

| status | meaning | effect on the visibility test |
|---|---|---|
| `HIT` | a point was returned | clears to `D − surfaceMargin`; the endpoint voxel becomes OCCUPIED |
| `NO_RETURN` | ray fired, nothing came back | clears to `maxRange` |
| `OUTSIDE_FOV` | the scanner never looked here | clears nothing |

### The `NO_RETURN` / `OUTSIDE_FOV` distinction is the correctness crux

Both appear in the file as absent data, and they mean opposite things. Every
terrestrial scanner has a blind cone under the tripod; treating those bins as
`NO_RETURN` carves a cone through the floor beneath all 1000 setups.

Resolution order:
1. If the scan stores explicit invalid points (`cartesianInvalidState` /
   `sphericalInvalidState` ≠ 0), use them — those bins are `NO_RETURN`.
2. Otherwise derive the field of view from the observed angular extent of the
   returned points (and `indexBounds` where present), and mark empty bins
   `NO_RETURN` inside it, `OUTSIDE_FOV` outside it.

Many writers drop invalid points entirely rather than storing them, so path 2
is the common case, not the fallback. `e57cov index --report` prints which
convention each file uses; **audit a sample of the real corpus before trusting
the defaults.**

### Other construction details

- Bin resolution is chosen at slightly finer than the scan's native angular
  step, from `rowIndex`/`columnIndex` where the file provides them, otherwise
  from the point count and angular extent.
- Where several points fall in one bin, keep the **minimum** range.
- Azimuth wraps; elevation clamps. Bins near the poles are tiny in solid angle
  and are not special-cased beyond that.
- A point beyond `maxRange` clears to `maxRange` and does **not** mark an
  OCCUPIED voxel.
- The E57 `pose` (quaternion + translation) may place points in scanner-local
  or already-global coordinates depending on the producer. Detect per file
  rather than assuming; mixed conventions across a 1000-file corpus assembled
  from several jobs is a realistic risk.

### Hierarchical acceleration (deferred)

A min/max mip pyramid over the range image supports brick-level accept/reject
(the classic two-sided HZB test): a brick whose angular footprint lies entirely
nearer than the footprint `min` is wholly visible; entirely beyond the footprint
`max` is wholly unresolvable by that scan. Use `min` over the footprint for the
accept test so silhouette edges are not eaten.

**Not in the first implementation.** The flat gather is expected to be fast
enough; this is an optimisation to add if profiling demands it. Note that
Metal's `generateMipmaps` is a box-filter *average* and is silently wrong for a
min/max pyramid — the reduction must be hand-rolled.

---

## 5. Grid representation

VISIBLE is stored **positively and sparsely**: 8³ bricks (512 bits = 64 bytes =
one cache line) behind a two-level page table, allocated on first touch.

Memory scales with observed space rather than with the bounding box, so voxels
outside the building are never allocated — the "how do we exclude space past
the walls" question is answered at the storage layer before any filtering runs.
For a tight bounding box around a single building this is a modest win over a
dense grid; for an L-shaped site, a campus, or anything where the box is mostly
not-building, it is decisive.

Dense equivalents, for scale: 600 M voxels is 75 MB as a bitset. VISIBLE and
OCCUPIED together are 150 MB. Even a `uint8` per-voxel observation count is
600 MB. Nothing here is close to the ~48 GB working set limit
(`recommendedMaxWorkingSetSize`, raisable via the `iogpu.wired_limit_mb`
sysctl), so the sparse layout is chosen for scaling headroom, not necessity.

Check `maxBufferLength` and shard before assuming one allocation covers the
grid — at 0.02 m spacing (9.4 × 10⁹ voxels) the bitset alone is 1.2 GB.

### Binary fast path vs. quality accumulation

Two modes, because they have very different cost:

- **Binary** (default): early-out on the first setup that sees a voxel. Most
  voxels terminate after 1–2 lookups.
- **Quality** (`--counts`): no early-out; visits every (voxel, setup) pair to
  accumulate observation count, best incidence angle, nearest range. ~30
  candidate setups × 600 M voxels ≈ 1.8 × 10¹⁰ lookups.

The quality pass answers "which space is seen by only one setup" and "which is
seen only at grazing incidence beyond 30 m", which for a coverage audit is
usually more useful than the binary mask. It is a separate opt-in pass so the
default stays fast.

---

## 6. From UNKNOWN to reportable voids

UNKNOWN is three different things — occlusion shadow (wanted), wall material
(not wanted), exterior space (not wanted) — and visibility alone cannot
separate them. The separation uses OCCUPIED as a barrier:

```
void = geodesicDilate(VISIBLE, maxVoidDepth, blockedBy: OCCUPIED) − VISIBLE
```

A wavefront BFS through UNKNOWN starting at the VISIBLE frontier, where
OCCUPIED voxels are impassable. One GPU pass, one physically meaningful
parameter: *unobserved space within `maxVoidDepth` of somewhere we did observe,
not reached through a scanned surface.*

Why this and not the obvious topological test: flood-filling UNKNOWN from the
grid boundary and discarding whatever it reaches is clean in principle, but the
OCCUPIED membrane has holes — the patch of wall behind a desk was never
scanned, so the flood leaks from the exterior through the wall into the very
pocket the tool exists to find. Since occlusion shadows are nearly always
adjacent to surfaces, that failure hits the common case. Bounding the dilation
contains the leak instead of relying on the membrane being watertight: a leak
through an unscanned wall patch yields a ~`maxVoidDepth` shell of exterior
rather than the entire outdoors, and component filtering removes it.

Optional cleanup filters, **not primary mechanisms** — the dilation subsumes
what they were doing, and they exist for corpora where it underperforms:

- storey slab clip (Z histogram → floor/ceiling planes)
- plan-footprint clip (2D closing of projected VISIBLE, hole fill, dilate by
  wall thickness)

For non-boxy geometry — curved façades, atria, complex massing — the principled
alternative is a watertight envelope from a screened-Poisson solve at coarse
resolution (0.2–0.5 m, 10–60 M cells) with normals oriented toward the
observing setup. Normal orientation is usually the fragile part of Poisson;
here every point knows which scanner saw it, so it is free and unambiguous.
CartesianCapture's `Processing/PoissonReconstructor.swift` is a working
implementation of the same solve, in Swift/Metal — a reference to port rather
than a library to call. Deferred until the dilation is shown to be
insufficient.

### The irreducible ambiguity

**An entire room never entered is geometrically indistinguishable from the
exterior.** Both are large UNKNOWN volumes behind unscanned surfaces. The
dilation reaches `maxVoidDepth` into it and no further, so a large unvisited
room is reported only partially.

No algorithm resolves this from geometry alone. In order of preference:

1. **Supply a floor plan, footprint survey, or IFC/BIM model** as the envelope
   and intersect against its interior volume. Every remaining void is then
   real. Worth more than every filter above combined.
2. **Aperture analysis** — an unvisited room connects to observed space through
   a doorway-sized opening (compact, roughly planar, ~0.8–2.1 m); the exterior
   connects through an unscanned wall patch of arbitrary shape. Promote the
   former rather than truncating it.
3. **Do not auto-delete.** Emit borderline components for human adjudication.

### Output classes

Components are labelled rather than merged into one undifferentiated mask:

| class | shape | action |
|---|---|---|
| `SHADOW` | small, shallow, adjacent to VISIBLE | add a setup here |
| `UNVISITED` | large, aperture-connected | this space was never scanned |
| `REVIEW` | failed classification | human adjudication |

Exports: OpenVDB grid, a surface-nets/marching-cubes mesh of the void boundary
(CartesianCapture's `Processing/MarchingCubesGpu.swift` is the same operation,
again as a port target), and a CSV of components with centroid, volume, bbox,
storey and class.

### Scan topology matters

If a meaningful fraction of the setups are **outside** the building, the
exterior is observed rather than unknown, walls become bounded by VISIBLE on
both sides, and §6 gets substantially more reliable. Check the setup
distribution before tuning: the right configuration for an interior-only job
differs from one with a full exterior circuit.

---

## 7. Implementation

C++20, CMake, `metal-cpp` for the GPU layer. C++ rather than Swift because
`libE57Format`-equivalent decoding is C++-shaped and because the reader can be
built and tested in the remote dev environment, where no Swift or Metal
toolchain exists.

**No third-party dependencies.** The E57 reader is written directly against
ASTM E2807, inverting format knowledge already verified in CartesianCapture's
`Export/E57Writer.swift` (pagination and CRC-32c, the 48-byte header, the
32-byte `CompressedVectorSectionHeader`, DataPacket bytestream layout). This
avoids `libE57Format`'s Xerces-C dependency and lets the decoder skip unwanted
fields natively.

GPU code follows the pattern established in CartesianCapture's `ICPGpu.swift`
and `PoissonGpu.swift`: singleton device wrapper, kernels compiled from a
source string at runtime, and **every failure path returns null so the caller
falls through to the CPU reference implementation** — which is retained
permanently as the correctness oracle, never deleted.

### Validation

No Metal toolchain exists in the remote dev environment, so kernels are
validated by a Python replica against the CPU reference before shipping — a
convention carried over from CartesianCapture, which develops under the same
constraint. The reader is validated by round-tripping generated fixtures that
cover the encodings real scanners emit — in particular bit-packed
`ScaledInteger`, which the reference writer never produces.

**A round trip against our own encoder proves self-consistency, not
conformance.** The fixture writer and the reader share one reading of the
standard, so any misreading common to both passes every test. The load-bearing
assumption is that a field's bit stream is *continuous across packet
boundaries* — a packet's chunk may end mid-value, with the remaining bits
supplied by the next packet — rather than each packet's chunk being
independently byte-aligned. If that is wrong, the reader decodes the first
packet of every scan correctly and then drifts, which looks like plausible but
subtly wrong geometry rather than an outright failure.

That assumption is only settled by decoding real scanner output. Until then,
treat the reader as unverified against the corpus. First checks on real files:
decoded point count matches `recordCount` exactly (a drifting bit cursor
usually truncates or overruns), and the Cartesian bounds match the file's own
`cartesianBounds` element.

### Build order

1. E57 reader → `(pose, range image, status mask)` for one scan.
2. Single-threaded CPU reference carve on a small grid. **Keep permanently.**
3. GPU gather kernel, flat, no hierarchy. Assert bit-exact against (2).
4. Sparse bricks and, if profiling demands, HZB culling. Re-assert bit-exact —
   these are pure optimisations and must not change a single bit.
5. Geodesic dilation, components, classification, export.

Steps 3–4 are where this class of tool goes wrong silently. The bit-exactness
gate matters more than the profiling.
