# E57 Coverage Checker

`e57cov` reads a corpus of structured E57 scans (~1000 files from terrestrial
laser scanners), works out which space the scanners actually had line of sight
through, and reports the space they did not — occlusion shadows behind
furniture, under stairs, behind pipes, and rooms that were never entered.

Built for macOS on Apple silicon (developed against an M4 Max, 64 GB unified
memory). C++20 + `metal-cpp`, no third-party dependencies.

See [DESIGN.md](DESIGN.md) for the full design and the reasoning behind it.

## Status

**Early. The E57 reader is complete and tested; nothing downstream of it is
built yet.**

| stage | state |
|---|---|
| E57 reader | done, 63 round-trip checks passing |
| `e57cov info` — format audit CLI | done |
| range-image builder | not started |
| CPU reference visibility pass | not started |
| Metal gather kernel | not started |
| void extraction, classification, export | not started |

### Known caveat

The reader has **not been validated against real scanner files.** Its tests
round-trip fixtures produced by an encoder in this same repo, which proves
self-consistency but not conformance: the fixture writer and the reader share
one reading of the standard, so any misreading common to both passes every
test.

The load-bearing assumption is that a field's bit stream is *continuous across
packet boundaries* rather than each packet's chunk being independently
byte-aligned. If that is wrong, the reader decodes the first packet of every
scan correctly and then drifts — plausible-looking but wrong geometry, not a
crash.

First checks to run against real files:

- decoded point count matches the file's declared `recordCount` exactly (a
  drifting bit cursor usually truncates or overruns)
- decoded bounds match the file's own `cartesianBounds` element

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

Targets `e57cov` and `test_e57`, both macOS command-line tools, C++20, with
shared schemes. ⌘R on the `test_e57` scheme runs the test suite in the console.

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
src/main.cpp                e57cov CLI
tests/e57_fixture.h         E57 writer used to generate test files
tests/test_e57.cpp          round-trip tests
tools/validate_xcodeproj.py pbxproj structural validator
E57CoverageChecker.xcodeproj
DESIGN.md                   design and rationale
```

## Related

[Mjankor/CartesianCapture](https://github.com/Mjankor/CartesianCapture) is a
separate Swift/Metal iOS capture app. There is no code or build dependency
between the two. Its E57 *writer* was the reference for this reader's
understanding of the binary format, and several of its GPU stages (Poisson
reconstruction, marching cubes) are the same operations this tool will need —
as implementations to port, not to link against.
