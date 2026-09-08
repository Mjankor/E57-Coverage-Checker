# E57 Coverage Checker

`e57cov` reads a corpus of structured E57 scans (~1000 files from terrestrial
laser scanners), works out which space the scanners actually had line of sight
through, and reports the space they did not — occlusion shadows behind
furniture, under stairs, behind pipes, and rooms that were never entered.

Built for macOS on Apple silicon (developed against an M4 Max, 64 GB unified
memory). C++20 + `metal-cpp`, no third-party dependencies.

See [DESIGN.md](DESIGN.md) for the full design and the reasoning behind it.

## Status

**In progress.** The reader, the viewer and the corpus indexer are done and
validated against real scanner files. The visibility analysis itself — the
reason the tool exists — is one stage in.

| stage | state |
|---|---|
| E57 reader | done, 63 round-trip checks passing |
| `e57cov info` — format audit CLI | done |
| structured-vs-merged check | done |
| viewer app (open, inspect, navigate) | done — **rendering layer unrun**, see below |
| LOD octree + selection (scale to 1000s of setups) | done and tested |
| on-disk point store (mmap, zero-copy) | done and tested |
| indexer: E57 corpus → store | done and tested |
| viewer on the store, two-phase open | done — **rendering layer unrun** |
| **range-image builder** | done and tested |
| CPU reference visibility pass | not started |
| Metal gather kernel | not started |
| void extraction, classification, export | not started |

### Reader validation

The reader **has now been run against real scanner output** and passes both
checks that would expose a mis-decoded bit stream: on a 5,646,018-point scan
with 32-bit packed `ScaledInteger` coordinates it decoded exactly
5,646,018 records and its decoded bounds matched the file's own
`cartesianBounds`. A drifting bit cursor cannot produce either result, so the
continuous-bit-stream reading of the standard is confirmed in practice.

`e57cov info` runs both checks and exits nonzero on failure, so a whole
directory can be swept as a smoke test before indexing it.

## What the reader handles

Written directly against ASTM E2807, targeting what terrestrial scanners
actually emit rather than the whole standard:

- paged file structure with CRC-32c verification, and the logical/physical
  address mapping that goes with it
- the XML section (own parser — no Xerces-C)
- `CompressedVector` decode: bit-packed `Integer` and `ScaledInteger`,
  single/double `Float`, zero-bit constant fields, index and ignore packets,
  compressor restart
- multiple scans per file, pose (quaternion + translation), index bounds
- **selective field decode** — name the fields you want and the rest are
  stepped over without being unpacked, roughly halving decode cost on a
  prototype that also carries intensity and colour

## The viewer

`E57CoverageChecker.app` opens an E57 corpus, checks each scan, and draws it.
**File ▸ Open Folder** takes a whole directory, which is the shape a
thousand-setup job actually arrives in.

### Opening is two-stage

A thousand files cannot be decoded before showing anything, and decoding them
in the foreground would beachball for minutes. So:

1. **Survey — headers only.** Pose, prototype, declared extent, metadata
   classification. No point decoding, so a thousand files take seconds. The
   setup layout is drawn immediately and the list fills in — enough to see the
   shape of the job and spot a stray setup.
2. **Index — the octree build**, in the background with progress and a
   cancel. The result is cached, keyed by the file list plus each file's size
   and modification time, so reopening the same corpus goes straight to the
   store. Edit or replace a scan and the key changes, so a stale store is
   never shown.

Once the store is open the viewer never holds the corpus in memory. It draws a
cut through the octree chosen by projected size under a hard point budget, so
**per-frame cost is set by the budget and the visible node count, not by how
much data exists** — a 5-setup store and a 5000-setup store cost the same to
draw. Where the store fits inside the device's maximum buffer length, the whole
mapping is wrapped once with `newBufferWithBytesNoCopy` and nodes are drawn at
their byte offsets, so the GPU reads the page cache directly and the kernel
pages nodes in and out as the view moves. Larger stores fall back to a
byte-budgeted LRU of per-node buffers.

**Navigation**

| input | action |
|---|---|
| left drag | pan |
| right drag | orbit |
| **right click** | set the orbit centre to the point under the crosshair |
| wheel / pinch | zoom |
| `F` | frame all |
| `[` `]` | smaller / larger points |

Control-left is an alias for right throughout, because holding a two-finger
click through a drag on a trackpad is awkward.

The orbit centre is picked from the nearest point under the centre crosshair,
and the camera does not move when it changes — only what it turns about — so
the view never jumps. If the crosshair is over empty space the centre is left
alone and the status bar says so, rather than flinging the view somewhere
arbitrary.

**Scans are placed by their `pose`.** E57 stores each scan in its own local
coordinate system and carries the registration in the `pose` element, so a
registered multi-setup job only lines up once every pose is applied. The list's
Status tooltip reports how each scan was placed. Files that store
already-transformed points *with* a non-identity pose contradict the standard;
those are detected rather than assumed, flagged `⚠︎ frame`, and left
untransformed instead of being displaced twice.

**Merged clouds are excluded.** Every scan is classified and the verdict shown
in the list: green structured, amber ambiguous, red excluded, with the reasoning
on hover. `src/scan_check.h` uses metadata *and* a geometric test of whether the
scan actually behaves like a range image, because metadata alone is not decisive
in either direction.

Only a positive merged-cloud finding excludes a scan. **Ambiguous scans are
indexed and drawn**: "no gridding metadata, and too few populated direction bins
to judge" is the ordinary verdict for a perfectly good scan from a terse writer,
and treating it as disqualifying would discard most of a corpus. The store
records the ambiguity so it stays visible rather than being quietly forgotten.

The status line reports how many points are drawn out of how many the store
holds, and says when the budget cut the detail short.

## Auditing a corpus

`e57cov info` answers the two questions the visibility pipeline needs settled
before it can be written correctly — and both are properties of your files, not
of the algorithm:

```sh
e57cov info --crc /path/to/scan.e57
```

For each scan it reports the pose, the prototype with per-field bit widths, and:

- **how no-return rays are represented** — an explicit `cartesianInvalidState` /
  `sphericalInvalidState` field, or nothing at all. This decides whether the
  field of view has to be recovered from angular extent, and getting it wrong
  carves a cone straight through the floor beneath every tripod (DESIGN.md §4).
- **which coordinate frame the points are in** — scanner-local with a
  meaningful pose, or already transformed to global. Both conventions appear in
  the wild, sometimes within one corpus.

It also runs the two checks that would expose a mis-decoded bit stream: decoded
record count against the declared `recordCount`, and decoded bounds against the
file's own `cartesianBounds`. Exit status is nonzero if either fails, so it can
be run across a whole directory as a smoke test.

## Build and test

Two build systems, both first-class. Xcode is the one to use on the Mac — the
forthcoming Metal work depends on its GPU capture and shader debugger. CMake
keeps the reader buildable and testable off the target platform.

**Xcode**

```sh
open E57CoverageChecker.xcodeproj
```

Seven targets, all C++20 with shared schemes:

| target | kind | what it is |
|---|---|---|
| `E57CoverageChecker` | app | the viewer |
| `e57cov` | tool | the format-audit CLI |
| `test_e57` | tool | reader tests |
| `test_viewer` | tool | camera / classifier / picker tests |
| `test_lod` | tool | LOD octree, selection and point store tests |
| `test_indexer` | tool | survey, bounded-memory build, store round trip |
| `test_range_image` | tool | grid path, angular mapping, conservative binning |

⌘R on a test scheme runs that suite in the console.

The project uses explicit file lists, so a new source file needs four pbxproj
entries: `PBXBuildFile`, `PBXFileReference`, a group child, and a `Sources`
build phase entry. `tools/validate_xcodeproj.py` checks all of that:

```sh
python3 tools/validate_xcodeproj.py E57CoverageChecker.xcodeproj
```

It verifies the object graph resolves, that referenced files exist on disk,
that no source is silently missing from the build, and that schemes point at
real targets. Run it after editing the project.

**CMake**

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/test_e57
```

Tests write fixtures to `/tmp`; set `E57COV_TMPDIR` to redirect them.

Both build systems list sources explicitly, so **adding a file means editing
both.** The validator catches an omission on the Xcode side; nothing catches it
on the CMake side but a failed build.

## Layout

```
src/e57.{h,cpp}             ASTM E2807 reader
src/scan_check.{h,cpp}      structured-vs-merged classification
src/point_cloud.{h,cpp}     decode + decimate for display
src/lod.{h,cpp}             LOD octree: additive build and view selection
src/point_store.{h,cpp}     on-disk store, mmap'd and zero-copy
src/indexer.{h,cpp}         corpus survey and bounded-memory build
src/range_image.{h,cpp}     structured scan -> range image (visibility stage 1)
src/camera.{h,cpp}          orbit camera
src/picker.{h,cpp}          screen-space point picking (orbit centre)
src/math3d.h                vectors and matrices
src/main.cpp                e57cov CLI
app/                        macOS app: AppKit window, Metal renderer
tests/e57_fixture.h         E57 writer used to generate test files
tests/test_e57.cpp          reader round-trip tests
tests/test_viewer.cpp       camera, classifier, picker, decimation tests
tests/test_lod.cpp          octree, selection, store tests
tests/test_indexer.cpp      survey and build tests
tests/test_range_image.cpp  range image tests
tools/validate_xcodeproj.py pbxproj structural validator
E57CoverageChecker.xcodeproj
DESIGN.md                   design and rationale
```

Everything that can be tested off a Mac lives in `src/` and is covered by
`test_viewer`. `app/` holds only AppKit and Metal glue — the split is
deliberate, because `app/` cannot be compiled or run in the development
environment at all.

## Related

[Mjankor/CartesianCapture](https://github.com/Mjankor/CartesianCapture) is a
separate Swift/Metal iOS capture app. There is no code or build dependency
between the two. Its E57 *writer* was the reference for this reader's
understanding of the binary format, and several of its GPU stages (Poisson
reconstruction, marching cubes) are the same operations this tool will need —
as implementations to port, not to link against.
