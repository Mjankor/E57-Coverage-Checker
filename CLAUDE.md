# E57 Coverage Checker — project rules for Claude

## GPU-first (standing rule)

**Always code for GPU as a priority. CPU only if necessary/appropriate.**

Any new compute-heavy code (per-voxel, per-point, per-texel, per-brick loops)
is written as a Metal kernel first. A CPU implementation is acceptable only
when one of these genuinely holds — and the justification goes in a doc
comment at the dispatch site:

- The work is inherently serial or too small to fill a GPU.
- It's irregular/recursive with dynamic memory AND its measured wall-clock
  time is already small.
- It's I/O-bound (E57 decode, file writers).

Every stage has to run fast in absolute terms. Do not justify leaving
something on CPU because it is small *relative to* another stage.

House pattern for GPU code:

- Singleton device wrapper; kernels compiled from a source string at runtime.
- Every failure path (no Metal, buffer alloc, command-buffer error) returns
  null and the caller falls through to a CPU reference implementation — the
  CPU path is kept as the correctness reference, **never deleted**.
- Keep data GPU-resident across passes; batch dispatches per command buffer;
  read back scalars as rarely as possible.
- Prefer gather formulations (one thread owns one output, writes once) over
  scatter with atomics — they are deterministic and make bit-exact validation
  against the CPU reference possible.

## Validation

There is no Metal toolchain in the remote dev environment, so kernels are
validated by a Python replica against the CPU reference before shipping, and
the first real Metal run happens on the target Mac.

Optimisations must not change results. When adding a hierarchy, a culling
pass, or a sparse layout, re-assert **bit-exact** equality with the
pre-optimisation output. This class of tool goes wrong silently; the
bit-exactness gate matters more than the profiling.

Round-tripping the E57 reader against this repo's own fixture writer proves
self-consistency, not conformance — both share one reading of the standard.
Claims about format correctness need real scanner files, not more fixtures.

## Conventions

- Branching: one feature branch per change, cut from `main`, merged back to
  `main` via PR. Do not push to `main` without asking.
- All tuning parameters must be derived from `voxelSize` or another physical
  quantity, and documented at their declaration. See DESIGN.md §1.
- No third-party dependencies in the reader. It is written directly against
  ASTM E2807 and must stay buildable on plain C++20 with no external libraries,
  so it can be tested off the target platform.
- Keep the reader platform-neutral; confine macOS/Metal code to the GPU layer.
- DESIGN.md is the spec. When an implementation decision contradicts it,
  update the document in the same change rather than letting them diverge.
