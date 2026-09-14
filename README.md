# E57 Coverage Checker

**E57 Coverage Checker** reads a corpus of structured E57 scans (~1000 files from
terrestrial laser scanners), works out which space the scanners actually had line
of sight through, and reports the space they did not — occlusion shadows behind
furniture, under stairs, behind pipes, and rooms that were never entered.

It is a **macOS app**. Everything it does is a menu item: open the scans, run the
visibility filter, look at the result over the point cloud, ask why one point came
out the way it did, and save the answer. There is no command-line tool — there was,
and keeping two front ends in step meant keeping two sets of defaults in step,
which drifted.

Built for macOS on Apple silicon (developed against an M4 Max, 64 GB unified
memory). C++20 + `metal-cpp`, no third-party dependencies.

See [DESIGN.md](DESIGN.md) for the full design and the reasoning behind it.

## Status

**In progress.** The reader, the viewer and the corpus indexer are done and
validated against real scanner files. The visibility analysis runs end to end,
in the app and on the command line, and its answer can be written out. The
remaining work is classification — separating occlusion shadows inside the site
from material behind walls and from open air outside it — and putting the
ray-march carve on the GPU.

| stage | state |
|---|---|
| E57 reader | done, 63 round-trip checks passing |
| format audit (Processing ▸ Scan Report) | done |
| structured-vs-merged check | done |
| viewer app (open, inspect, navigate) | done |
| LOD octree + selection (scale to 1000s of setups) | done and tested |
| on-disk point store (mmap, zero-copy) | done and tested |
| indexer: E57 corpus → store | done and tested |
| viewer on the store, two-phase open | done |
| **range-image builder** | done and tested |
| no-return classification (cone, min range, sky) | done and tested |
| per-setup indoor/outdoor, with a manual override | done and tested |
| **CPU reference visibility pass** | done and tested |
| parallel tiles, brick culling, domain clipping | done and tested |
| **shrinkwrap domain, signed buffer, gap bridging** | done and tested |
| **ray-march carve** (the specified method, default) | done and tested — **CPU only** |
| **Metal carve kernel** | done, for the two gather methods; declines march tiles |
| **visibility filter in the app** | done — Processing ▸ Run Visibility Filter |
| **voxel display + layer toggles** | done |
| **export** — voxels and shell as PLY | done — File ▸ Save |
| void extraction and classification | partial — a tick box on the run sheet, off by default |

### Reader validation

The reader **has now been run against real scanner output** and passes both
checks that would expose a mis-decoded bit stream: on a 5,646,018-point scan
with 32-bit packed `ScaledInteger` coordinates it decoded exactly
5,646,018 records and its decoded bounds matched the file's own
`cartesianBounds`. A drifting bit cursor cannot produce either result, so the
continuous-bit-stream reading of the standard is confirmed in practice.

The scan report runs both checks over a whole directory and counts the failures,
so a corpus can be swept as a smoke test before carving it.

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
| right drag | orbit — dragging up tips the camera down, turntable style |
| **right click** | set the orbit centre to the point under the crosshair |
| wheel | zoom — scrolling towards you pulls the model closer |
| pinch | zoom — pinching apart zooms in |
| `F` | frame all |
| `⇧F` | frame the voxels |
| `⌘1` `⌘2` | show / hide the clouds, show / hide the voxels |
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
holds, and says when the budget cut the detail short — and, after a visibility
run, how much unobserved space was found.

## Auditing a corpus

**Processing ▸ Scan Report…** (⌘I) answers the two questions the visibility
pipeline needs settled before it can be written correctly — and both are properties
of your files, not of the algorithm. It shows the report in a window, copies it to
the clipboard, and writes it to `~/Desktop/e57cov-scan-report.txt`.

For each scan it reports the pose, the prototype with per-field bit widths, and:

- **how no-return rays are represented** — an explicit `cartesianInvalidState` /
  `sphericalInvalidState` field, or nothing at all. This decides whether the
  field of view has to be recovered from angular extent, and getting it wrong
  carves a cone straight through the floor beneath every tripod (DESIGN.md §4).
- **whether the fitted angular mapping actually describes the raster** — the
  scan's own points are put back through it, and the percentage that land on the
  cell they came from is reported. A low figure with healthy residuals means
  lookups reach the wrong direction, which shows up as sky read as ground and
  building interiors read as clear space.
- **the instrument's blind cone**, and which way up it was mounted. This is the
  one empty region that does not mean "the ray came back with nothing" — no ray
  was fired at all — and believing it clears a cone to the maximum range straight
  through whatever the scanner stood on. Which end of the raster it occupies is
  found from the geometry, never assumed: the returns bordering the cone are the
  ground beside the mount, metres away, where those bordering sky are distant.
  That holds for a scanner mounted upside down, and the reported cone axis says
  when one was.
- **whether the scan fits the grid it declares** — points falling outside
  `indexBounds` are counted, and a scan where too many do is refused rather than
  read as a mostly-empty raster, because empty reads as clear space.
- **whether the scanner sits inside its own returns** — it should, in its own
  frame. When it does not, the points are probably not in the frame they are
  being read as.
- **which coordinate frame the points are in** — scanner-local with a
  meaningful pose, or already transformed to global. Both conventions appear in
  the wild, sometimes within one corpus.

It also runs the two checks that would expose a mis-decoded bit stream: decoded
record count against the declared `recordCount`, and decoded bounds against the
file's own `cartesianBounds`. Exit status is nonzero if either fails, so it can
be run across a whole directory as a smoke test.

## The visibility pass

**Processing ▸ Run Visibility Filter…** (⌘R). The sheet asks for the voxel size,
the range, the region and the settings that decide what an empty cell means; hover
any of them for what it does and why. It runs on a background queue with a progress
line and File ▸ Cancel, and draws the result over the point cloud.

Builds a range image per scan, then walks space in tiles and works out, for each
voxel, what the setups that reach it saw there. A voxel comes out carrying two
independent bits:

- **visible** — some setup had line of sight through it.
- **occupied** — some setup measured a surface inside it.

A voxel within range of a setup and carrying neither is one nobody observed:
a candidate void. That is the number the tool exists to produce, though it is
not yet the answer — at this stage it still mixes occlusion shadows inside the
site with material behind walls and with the open air outside the building.
Separating those is the next stage.

The two buttons at the top right of the view turn the original clouds and the
voxels on and off independently (⌘1 and ⌘2 do the same), because reading the result
means flicking between them.

### Three methods, and they nest

**Method** on the run sheet chooses how a setup's evidence reaches a voxel. They differ only in how
many of the rays that crossed it they consult, so their answers nest: every voxel
one ray finds, seventeen find, and every voxel seventeen find, the ray that
actually passed through it finds.

| | rays per voxel | unobserved | ms |
|---|---|---|---|
| one ray through the voxel centre | 1 | 37.31% | 482 |
| the voxel's own footprint | ≤17 | 0.51% | 605 |
| march each ray (default) | every ray that crossed it | 0.45% | 4581 |

Measured over an 8 m room at the raster geometry of a real station, a third of its
directions unexplained, 4.1 M voxels at 5 cm, one thread. The first asks the one
direction through the voxel's centre — but a voxel is bigger than a raster cell
everywhere inside about 16 m, so where a scan cannot explain a third of its
directions, a third of near voxels land on a cell that says nothing. Marching is the
method the tool was specified around: walk each measured ray and mark the voxels it
crosses, so nothing has to line up with anything. See DESIGN.md §4.

The march runs on the CPU only for now; the Metal carver is a gather and declines
those tiles rather than answering a different question.

Two implementations of each, and they must agree. `carveTileReference` is the
oracle — no windowing, no culling, never optimised and never deleted. `carveTile`
is what runs: parallel across tiles, culling bricks against a min/max pyramid for
the gathers, and windowing the raster down to the cells whose rays can reach a tile
for the march. `test_carve` asserts the two produce byte-identical tiles for every
method, at several tile sizes, with and without an apron, with and without a domain
box slicing through the scene, and with and without the pyramid built. Every
optimisation re-runs that check; it is the only reason any of them can be trusted.

The run holds every range image in memory at once, which the production path will
not do — the region, and a coarser voxel, are the levers when a corpus is too big
for that.

**Region** is the setting that matters most. By default the question is a
**shrinkwrap** of the returns — the space within the buffer of something a scanner
actually measured — which drops the corners of a box that no scan reached. The
alternatives are that box, and everything within range of any setup: honest, but
for a building scanned from inside mostly sky, since the range spheres reach tens
of metres out through every wall.

The **buffer** is *signed*, and defaults to −0.2 m. Negative pulls the shell inside
the walls, which is how the space outside a building leaves the question rather
than being filtered out of the answer afterwards; positive grows it outward for
wall thickness and eaves.

**Tile size** changes the working set and nothing else. Carving a volume as one
large tile and as many small ones gives identical results, voxel for voxel —
the voxel lattice is global and anchored at the world origin, so a voxel's
verdict never depends on which tile carried it. The test suite asserts this
directly, because a carve whose answer depended on how space was partitioned
could not be validated against anything.

The same report is available in the app as **Processing ▸ Scan Report…** (⌘I),
which shows it in a window, copies it to the clipboard, and writes it to
`~/Desktop/e57cov-scan-report.txt`. That is there because Xcode builds the scheme
you have selected, so building the app never builds `e57cov`, and a stale
command line tool on a path is indistinguishable from a current one.

Every report names the build that produced it. If the revision at the top is not
the one you just built, nothing below it is worth reading.

## Indoor or outdoor, per setup

The setups list has a **Sky** column. The carve fills it in from each scan's own
sky test — an opening at the zenith wide enough, far enough around, and not
bordered inside the instrument's minimum range (DESIGN.md §4) — and hovering a cell
gives the three numbers it turned on, so "outdoor" is something you can check
rather than something you are told.

It is editable. Auto leaves the test to decide; Indoor and Outdoor say you know
better, which happens: a station in a glazed atrium reads as indoors, one in a
doorway can read as out. Marks are per scan, survive re-runs, and only take effect
when **Use the indoor/outdoor column instead of the sky test** is ticked on the run
sheet — a switch rather than an implicit "marks win", so the two can be compared on
the same corpus without clearing and retyping them. Setups left on Auto still fall
to the test, so marking two stations out of nine hundred leaves the rest measured.

One thing a mark cannot do is conjure sky where the zenith holds returns. There is
nothing there to believe, and clearing a cone up through a roof on the strength of a
tick box is the exact failure this tool exists to catch — so Outdoor on such a scan
leaves it indoors, and the column says so.

## Getting the answer out

A carve that takes minutes and exists only as pixels is not a deliverable, so both
of its point sets can be written to a file something else opens: **File ▸ Save
Unobserved Voxels…** (⌘S) and **File ▸ Save Shrinkwrap Shell…** (⇧⌘S).

The format is binary PLY, which CloudCompare, Recap, Cyclone, MeshLab and Blender
all read without a plugin. **Coordinates are doubles**, which matters: a
georeferenced site sits at UTM magnitudes where float32 resolves about a metre, so
the file carries absolute position at full precision and needs no shift agreed out
of band. One point per voxel centre, coloured as drawn.

The header records the build and every setting that decides what counts as
observed — the sky angle and arc, the minimum range, the carve method, the region
and its buffer — because these runs are parameter sweeps by nature and two clouds
saved from different settings are otherwise indistinguishable.

It also says, in as many words, when the set is **sampled**. The drawn-voxel cap
thins what is displayed and the saved set is the displayed set, so a capped run
would otherwise hand over a thinned cloud that looks exactly like a complete one.
Raise the cap and run again to save the lot.

Saving the shell is worth doing beside the voxels: it decides what the whole
answer covers while being invisible in that answer, so a shell that went wrong
looks, in the voxels alone, exactly like a survey that missed different space.

## Explaining one point

**Processing ▸ Explain a Point…** (⌘P), prefilled with the point the view is
orbiting — click in the view to move it, since "why is this bit wrong?" is asked by
pointing at it.

For every setup: the distance and direction to the point, the raster cell that
direction lands on, what that cell holds, and the verdict. Aggregate figures
cannot answer "the inside of the house is being cleared, why?"; this can, and
every wrong answer this tool has produced was diagnosable from those five things
and invisible in the statistics.

## Build and test

Two build systems. Xcode builds the app and the tests, and is the one to use on
the Mac — the Metal work depends on its GPU capture and shader debugger. CMake
builds the library and the tests only, which is what keeps the core buildable and
testable off the target platform; there is no app and no tool target there.

**Xcode**

```sh
open E57CoverageChecker.xcodeproj
```

Eleven targets, all C++20 with shared schemes:

| target | kind | what it is |
|---|---|---|
| `E57CoverageChecker` | app | the whole tool |
| `test_e57` | tool | reader tests |
| `test_viewer` | tool | camera / classifier / picker tests |
| `test_lod` | tool | LOD octree, selection and point store tests |
| `test_indexer` | tool | survey, bounded-memory build, store round trip |
| `test_range_image` | tool | grid path, angular mapping, conservative binning |
| `test_carve` | tool | per-setup evidence, OR across setups, tiling invariance |
| `test_visibility` | tool | frontier reduction, display sampling, end-to-end run |
| `test_wrap` | tool | shrinkwrap: distance transform, gaps, signed buffer |
| `test_voids` | tool | connectivity and classification |
| `test_ply` | tool | the point writer, read back as another reader would |

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
src/carve.{h,cpp}           tiled visibility carve, CPU reference (stage 2)
src/visibility.{h,cpp}      the carve as a job: files in, drawable voxels out
src/ply.{h,cpp}             writing the answer back out as a point cloud
app/CarveGpu.{h,mm}         the carve as a Metal gather kernel
src/camera.{h,cpp}          orbit camera
src/picker.{h,cpp}          screen-space point picking (orbit centre)
src/math3d.h                vectors and matrices
app/                        macOS app: AppKit window, Metal renderer
tests/e57_fixture.h         E57 writer used to generate test files
tests/test_e57.cpp          reader round-trip tests
tests/test_viewer.cpp       camera, classifier, picker, decimation tests
tests/test_lod.cpp          octree, selection, store tests
tests/test_indexer.cpp      survey and build tests
tests/test_ply.cpp          point writer tests
tests/test_range_image.cpp  range image tests
tests/test_carve.cpp        visibility carve tests
tests/test_visibility.cpp   frontier reduction, display sampling, tiling invariance
tools/genproj.py            regenerates the Xcode project from a file list
tools/validate_carve_kernel.py  replica check of the Metal kernel vs the reference
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
