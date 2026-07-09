# 07.09 — Jump-flooding Voronoi/distance transforms (easy, visual, useful)

**Difficulty:** ★ beginner · **Domain:** 7. Collision Detection & Geometry

> Catalog bullet (source of truth, verbatim): `★ Jump-flooding Voronoi/distance transforms (easy, visual, useful)`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

Scatter a handful of "seed" cells across a grid and answer, for **every** cell: *which seed is
nearest, and how far is it?* The labels form a **Voronoi diagram**; the distances form a
**distance transform**. In robotics clothing: seeds are obstacle cells, and the distance transform
is the **clearance field** that local planners, costmap inflation, and safety monitors consume.
This project computes both with the **jump-flooding algorithm (JFA)** — a classic GPU technique
that replaces the exact O(W·H·N) scan with O(log max(W,H)) gather passes over the grid — verifies
it every run against an *exact* brute-force CPU oracle under documented approximation bounds, and
writes two viewable images (regions + clearance field). It is the repository's first
**grid/stencil-pattern** project (after 33.01/09.01's thread-per-problem pattern) and its first
whose oracle checks an *algorithm's promise* rather than a ported computation.

## What this computes & why the GPU helps

The exact answer costs `W·H·N` distance evaluations (every cell × every seed) — 134 million for
the demo's 1024²×128 batch. JFA gets within a whisker of it in `log₂(1024) + 1` passes of
`W·H·9` cheap gathers:

- **Pattern:** grid/stencil map with ping-pong double buffering — one thread per **cell**, 2-D
  16×16 blocks, each pass gathering from 8 neighbors at offset ±step (step halves every pass), the
  read and write buffers swapped between passes.
- **Why it fits the GPU:** every cell's update is independent within a pass; a million cells means
  a million threads, and the "information hops exponentially far" schedule does in ~11 passes what
  brute force does in 128 scans.
- **The honest twist:** JFA is *approximate* (rare boundary cells settle on the second-best seed).
  Quantifying that against the exact oracle — and fixing it with the **1+JFA** variant — is half
  the teaching value ([THEORY.md](THEORY.md)).

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** the geometry substrate of the **planning layer** (domain 07 serves 06/23),
  with a safety-monitor branch (31/21): distance-to-obstacle fields are what those layers query.
- **Upstream inputs:** an occupancy/costmap layer (message shape: `nav_msgs/OccupancyGrid` — a
  W×H byte grid + resolution in m/cell) or any obstacle-cell list; here, a seed list
  (`S,id,x,y` rows, layout in [`src/kernels.cuh`](src/kernels.cuh)).
- **Downstream consumers:** costmap inflation and DWA scoring (23.01), trajectory-clearance costs
  in local planners (06.x), speed-and-separation monitoring (21.04), and — with labels kept —
  generalized-Voronoi skeleton extraction for topological maps (05.16) and coverage partitions
  (22.04).
- **Rate / latency budget:** costmap updates run at 5–20 Hz on real stacks (SYSTEM_DESIGN item 1);
  the measured ~4–6 ms end-to-end JFA on a 1024² grid fits inside even the fast end with room for
  the rest of the costmap pipeline.
- **Reference robot(s):** the **warehouse AMR** (clearance for DWA/costmaps) and the
  **autonomous-vehicle stack** (drivable-space distance fields); the manipulator cell uses the 3-D
  cousin (see "In production").
- **In production:** costmap_2d's inflation layer (CPU, incremental), OpenCV `distanceTransform`
  (CPU, exact, image-sized), and for 3-D the ESDF builders in nvblox/Voxblox — JFA's 3-D extension
  is one of the standard GPU routes there.
- **Owning team:** navigation/planning within an autonomy group; the safety-monitor consumers live
  with the functional-safety team (SYSTEM_DESIGN item 5).

## The algorithm in brief

- **JFA proper** — initialize seeds; passes at step = P/2, P/4, …, 1 (P = grid size rounded up to a
  power of two): each cell adopts the closest seed any of its 9 step-offset samples knows. →
  [THEORY.md](THEORY.md) §The algorithm.
- **1+JFA** — one extra step-1 pass *before* the schedule; suppresses JFA's rare long-range misses
  (adopted here after the plain variant exceeded the error bound — the story is in
  [THEORY.md](THEORY.md) §How we verify correctness).
- **Exact oracle** — brute-force nearest-seed scan on the CPU, integer arithmetic throughout, so
  verification is exact counting, not float tolerance.

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/jump-flooding-voronoi-distance-transforms.sln`](build/jump-flooding-voronoi-distance-transforms.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/jump-flooding-voronoi-distance-transforms.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none** — CUDA runtime + C++17 standard library only.

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at — **including the two images the demo writes to `demo/out/`**.

## Data

Fully **synthetic** (labeled so everywhere): `data/sample/jfa_seeds.csv` (~1.2 KiB, committed) — 64
distinct seed cells on the 512×512 sample grid, generated by
[`scripts/make_synthetic.py`](scripts/make_synthetic.py) with seed 42, byte-identical on
regeneration. No public dataset applies (seed cells are fully synthesizable; the oracle supplies
ground truth at run time), so `scripts/download_data.ps1` is an honest no-op. Format and checksum:
[`data/README.md`](data/README.md).

## Expected output

Eight stable lines — banner, `PROBLEM:`, `SAMPLE:`, `SAMPLE RESULT: PASS`, `ARTIFACT:`, `BATCH:`,
`BATCH RESULT: PASS`, `RESULT: PASS` — checked as a subset diff by
[`demo/expected_output.txt`](demo/expected_output.txt). Verification is a **bounds check against
exactness**: label mismatches ≤ 0.5% of cells (exact-distance ties count as agreement — a tie cell
has two true answers) and max distance error ≤ 2 cells. The entire comparison is integer
arithmetic, so the `[info]` mismatch counts are deterministic: on this machine the 512² sample
matches the exact field **perfectly** (0 mismatches), and the 1024²/128-seed batch mismatches 4
cells in a million with max error 0.88 cells. The demo also writes `demo/out/voronoi.pgm` and
`demo/out/distance.pgm` (git-ignored, regenerated every run).

## Code tour

A guided reading order through `src/`:

1. [`src/main.cu`](src/main.cu) — the two stages, the bounds-check comparator (read its
   tie-handling), the PGM artifact writer.
2. [`src/kernels.cuh`](src/kernels.cuh) — the cell-state layout (why `int4`), the seed-list layout,
   and the approximation contract.
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the *exact* oracle, and the header comment on
   why this oracle is a different algorithm, not a serial twin.
4. [`src/kernels.cu`](src/kernels.cu) — the heart: the JFA pass kernel and the ping-pong launcher.
   The single most interesting thing: the **1+JFA priming pass** and the comment explaining the
   bug-hunt that put it there.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers (copied, not shared — §4 rule).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **Rong & Tan (2006), "Jump Flooding in GPU with Applications to Voronoi Diagram and Distance
  Transform"** — the original paper; the 1+JFA and JFA² variants live there too.
- **OpenCV `distanceTransform`** — the exact CPU answer (Felzenszwalb/Borgefors families); know
  when image-sized exact beats grid-sized approximate.
- **Felzenszwalb & Huttenlocher (2012)** — the exact O(W·H) separable distance transform; the
  strongest CPU competitor and Exercise 5's subject.
- **nvblox / Voxblox ESDF builders** — production 3-D distance fields for planning; JFA's 3-D
  extension (26 neighbors) is one standard GPU route.
- **costmap_2d (Nav2)** — how classic stacks *incrementally* inflate costs instead of recomputing
  fields; the engineering trade against a 4 ms full recompute.
- **`thrust`/CUB** — not used here on purpose; everything is a hand-rolled kernel so the pass
  structure stays visible.

## Exercises

1. **Draw more seeds:** bump the sample to 512 seeds and look at `voronoi.pgm` — then explain why
   JFA cost is *independent* of seed count while the CPU oracle scales linearly in it.
2. **Wavefront contrast:** implement the naive brute-force GPU kernel (one thread per cell, loop
   all seeds) and time it against JFA at N = 16, 128, 1024 — find the crossover and explain it.
3. **3-D JFA:** extend cells to `int4`(x,y,z,id) and the gather to 26 neighbors — you have built
   the core of an ESDF pipeline (the nvblox connection).
4. **Shared-memory tiling for the small-step passes:** cache a (16+2·step)² tile when step < 8 and
   measure whether it ever wins (spoiler worth verifying: the early long-range passes dominate).
5. **Exact separable transform:** implement Felzenszwalb–Huttenlocher on the CPU (or GPU per-row/
   per-column) and compare exactness *and* speed against JFA — then argue which your robot needs.

## Limitations & honesty

- **JFA is approximate** — documented, measured (4/1,048,576 cells, ≤0.88-cell error on the batch),
  and bounded by the verification contract; if your application needs *exact* fields (safety
  certification arguments do), use an exact algorithm (Exercise 5) and pay its structure.
- **The plain-JFA failure was real:** the first implementation exceeded the 2-cell bound
  (3.9-cell max error) and was upgraded to 1+JFA — kept in the record (push-note 2026-07-08-03)
  because "the oracle caught it" is the whole point of oracles.
- **2-D only, unweighted, point seeds** — no obstacle footprints/polygons (rasterize first),
  no anisotropic metrics; 3-D is Exercise 3.
- **Euclidean metric on cell centers** — real costmaps often want obstacle-boundary distance;
  consumers handle the half-cell subtleties.
- **Timings are teaching artifacts** — end-to-end JFA including its internal allocations
  (deliberately honest accounting), single-shot, one machine.
- **Nothing here commands hardware**; the sim-only caveat applies to consumers (planners, monitors)
  that act on these fields.
