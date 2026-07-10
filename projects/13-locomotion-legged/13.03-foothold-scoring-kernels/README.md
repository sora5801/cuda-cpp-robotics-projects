# 13.03 — Foothold scoring kernels: slope, roughness, edge distance from elevation maps

**Difficulty:** intermediate · **Domain:** 13. Locomotion — Legged

> Catalog bullet (source of truth, verbatim): `Foothold scoring kernels: slope, roughness, edge distance from elevation maps`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

**A quadruped's foot planner deciding, cell by cell, "can I stand here?"** This project takes a
256x256 (5.12x5.12 m, 0.02 m/cell) elevation map — the kind an onboard depth camera + elevation
mapper hands a legged robot ten to thirty times a second — and turns it into a per-cell foothold
score in `[0,1]` through four GPU kernels: a least-squares **slope** and **roughness** fit over each
cell's local neighborhood, a bounded **edge-distance** gather that measures how far each safe cell
sits from the nearest hazard, and a **fusion** step that blends the three into one score with two
hard vetoes (unknown ground, and slope past a friction-derived limit). A fifth, consumer-facing
kernel then answers the question a real gait planner actually asks: given a **nominal** landing point
for the next step, which nearby cell is the best place to actually put the foot? The demo builds one
synthetic map with a documented ramp, step, rock field, and sensor dropout (a "hole"), scores 1000
such queries along a path that crosses every feature, and checks the result against the terrain's own
known ground truth — the ramp's constructed angle, the step's location, the flat region's near-zero
slope — not just against a CPU port of the same code. Every component named in the catalog bullet
(slope, roughness, edge distance) is fully implemented; the foothold-selection consumer step is this
project's addition, included because a *scoring* kernel with nothing that ever reads the scores is
only half the pipeline (README §Limitations elaborates).

## What this computes & why the GPU helps

Per elevation map: 65536 cells, each independently fitting a 5x5-neighborhood least-squares plane,
searching a 21x21 window for the nearest hazard, and blending four numbers into a score — then 1000
independent queries, each searching a small disc of that score grid for its best foothold.

- **Pattern:** three back-to-back per-cell **stencils/maps** (slope+roughness is a windowed *stencil*;
  edge-distance is a bounded *gather*; fusion is a pure *map*) followed by a per-query **batched
  search** (the foothold-selection kernel) — four different-shaped GPU problems chained into one
  pipeline, each still the simplest correct mapping for its shape (one thread per cell, then one
  thread per query).
- **Measured reality** (RTX 2080 SUPER, this demo's 256x256 map): the slope/roughness fit — the
  cheapest-per-cell kernel but touching all 65536 cells — runs in 0.44-0.51 ms on the GPU against
  7.6-7.7 ms on one CPU core (**~15-17x**); the edge-distance search — the same cell count but a much
  wider 21x21 window per thread — runs in ~0.32 ms against ~40 ms CPU (**~126-127x**, the widest
  margin in the pipeline, exactly because its *per-thread* work is largest while its *per-thread*
  independence is unchanged). Fusion and selection are both sub-0.1 ms GPU-side; a CPU-timed
  comparison for them would be dominated by measurement noise at this problem size (see
  [`demo/README.md`](demo/README.md)'s honest single-shot-timing caveat). Numbers are printed by every
  demo run on `[time]` lines — a **teaching artifact, never a benchmark claim** (CLAUDE.md §12).
- **Why this scales to a real robot:** a production stack re-scores the map every time it moves
  (10-30 Hz elevation updates, README §System context) — at CPU speed, 40+50 ms just for two of the
  four kernels already blows a 10 Hz budget; on the GPU the whole pipeline fits inside 1 ms with
  headroom for everything else on the compute budget.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** the boundary between **state estimation/world model** (the elevation map itself)
  and **planning** — specifically the locomotion-planning layer named in SYSTEM_DESIGN §2.3's
  quadruped diagram and spelled out exactly in §4.3's Chain C: `[05.05 Elevation mapping] -> [13.03
  Foothold scoring kernels] -> [13.02 Centroidal MPC] -> whole-body control`.
- **Upstream inputs:** a 2.5-D elevation map (message shape: `GridMap`-like — SYSTEM_DESIGN §3.6's
  `OccupancyGrid` sketch, generalized from occupancy bytes to float heights; here the flattened,
  row-major `height_m[W*H]` from [`src/kernels.cuh`](src/kernels.cuh)) from an elevation mapper such
  as project 05.05, itself fed by a depth camera or LiDAR simulator (11.02) and the robot's own base
  pose (04.08's invariant EKF).
- **Downstream consumers:** the nominal-foothold generator inside a gait/footstep planner (13.02
  Centroidal MPC, or a simpler heuristic footstep planner like 13.08) — it proposes *where* a foot
  would nominally land for the current gait phase; this project answers *whether that is safe and
  where exactly, precisely, to put it* (message shape: a scored `PointStamped`-like reply, this
  project's `FootholdResult`).
- **Rate / latency budget:** a trotting gait's swing phase lasts roughly 200-300 ms — the window in
  which a foothold must be chosen, committed to, and tracked before the foot touches down — so this
  pipeline sits inside a 10-50 Hz locomotion-planning budget (SYSTEM_DESIGN §1.1's local-planner band,
  restated for legged locomotion in §2.3/§4.3). The measured GPU pipeline (~1 ms total across all four
  map kernels at this map's resolution) leaves the overwhelming majority of even the *tightest* swing
  phase for the gait/footstep search built on top of it.
- **Reference robot(s):** the **quadruped** (SYSTEM_DESIGN §2.3) — the only reference robot whose
  block diagram this project's exact name appears in (§4.3 Chain C). A biped or hexapod locomotion
  stack would consume the identical scoring pipeline; only the gait layer above it changes per
  morphology.
- **In production:** ETH Zurich's `elevation_mapping` / `grid_map` ecosystem (and its commercial
  descendants at companies like ANYbotics and Boston Dynamics) computes exactly this class of
  traversability layer on real robots; MIT Cheetah-family controllers use simpler heuristic foothold
  scoring at higher rates with less per-cell modeling. Both trade the plane-fit's fidelity against the
  compute/latency budget differently than this teaching version does (THEORY.md §real-world).
- **Owning team:** locomotion/controls, a specialization inside controls & autonomy (SYSTEM_DESIGN
  item 5) — the team that also owns state estimation (04), elevation mapping (05), and the gait
  planner (13.02/13.08) this project's output feeds.

## The algorithm in brief

- **Slope + roughness** — fit `z = a*x + b*y + c` by least squares over each cell's `(2*kFitRadius+1)^2`
  window (5x5 = 0.1x0.1 m), solved in closed form via Cramer's rule; `slope = atan(sqrt(a^2+b^2))`,
  `roughness = std-dev of the fit's residuals`. → [THEORY.md](THEORY.md) §The math, §The algorithm.
- **Edge distance** — mark cells hazardous (unknown height, degenerate fit, too steep, or too rough)
  and, for every other cell, gather the distance to the nearest hazard within a bounded 21x21-cell
  window (a hand-rolled, capped distance transform — conceptually 07.09's problem at a much smaller,
  bounded scale). → THEORY §The algorithm, §The GPU mapping.
- **Fusion** — blend `slope_score`, `rough_score`, `edge_score` (each `clamp(1 - x/limit, 0, 1)`-shaped)
  with documented weights (0.4/0.3/0.3), forced to exactly 0 by two hard vetoes: unknown ground, or
  slope past `atan(mu)` — the friction-cone limit derived in THEORY §The problem.
- **Foothold selection** — for each of 1000 nominal landing points, argmax the fused score over a
  0.10 m search disc, with a deterministic raster-order, strict-`>` tie-break. → THEORY §The GPU
  mapping (the batched-search pattern), §How we verify correctness (why this stage alone gets an
  *exact* GPU/CPU match).

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/foothold-scoring-kernels.sln`](build/foothold-scoring-kernels.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/foothold-scoring-kernels.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none** — CUDA runtime + C++17 standard library only. No
cuBLAS/cuFFT/Thrust: every kernel here is a hand-rolled per-cell/per-query map or windowed gather, the
size this repo's "no black boxes" rule (CLAUDE.md §1) asks you to be able to write without a library.

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) — including
**the two artifacts to view**.

## Data

The committed sample is a **terrain RECIPE**, not a recorded map or a committed grid:
`data/sample/terrain_scenario.csv` (~1.9 KiB, synthetic, rock placement seeded 42) — a background
ripple, a 15-degree ramp, a 0.12 m step, 16 scattered rock domes, one rectangular NaN "hole", and a
1000-point, 5-segment foothold-query path, all in five non-overlapping map regions with documented
coordinates. `src/main.cu`'s `build_terrain()`/`build_queries()` turn the recipe into the actual
65536-cell grid and query list at start-up — deterministically, with zero run-time randomness (the
recipe already carries the one random choice, where the rocks sit, as literal numbers).
`scripts/download_data.ps1`/`.sh` are honest no-ops: no public dataset ships exact per-cell
slope/roughness/edge-distance ground truth the way a hand-designed synthetic terrain can. Full
field-by-field format and the checksum: [`data/README.md`](data/README.md).

## Expected output

Six stable lines — banner, `PROBLEM:`, `SCENARIO:`, `VERIFY: PASS`, `ARTIFACT:`, `RESULT: PASS` —
checked as a subset diff by [`demo/expected_output.txt`](demo/expected_output.txt). Two layers of
verification, described fully in [THEORY.md](THEORY.md) §How we verify correctness:

1. **VERIFY (four stage-isolated GPU-vs-CPU gates):** each of the four kernels is checked against its
   CPU twin fed *identical* upstream arrays, isolating each gate to exactly one kernel (measured worst
   case: slope 1.4e-6 rad, roughness 1.5e-8 m, edge distance and fusion agree to within float rounding
   of a pinned input — 0 and 1.2e-7 respectively — and foothold selection matches its CPU oracle's cell
   index **exactly**, for all 1000 queries).
2. **Four analytic gates against the terrain's own known ground truth**, run on the real, all-GPU
   pipeline: the flat control region measures max slope 1.30 deg (bound 3.4) and mean score 0.987
   (bound >0.95); the ramp measures mean slope 15.01 deg against a constructed 15.00 deg (tolerance
   ±1.5 deg); the step's edge cell is hard-vetoed to score exactly 0 while a cell just beyond the
   search reach saturates its edge-distance at the 0.20 m cap; and all 1000 foothold selections are
   both valid and within their 0.10 m search radius.

## Code tour

A guided reading order through `src/`:

1. [`src/main.cu`](src/main.cu) — the whole pipeline in plain sight: terrain synthesis -> the
   four-gate, stage-isolated VERIFY stage -> the real all-GPU PIPELINE stage -> four analytic gates ->
   artifacts. Its file header explains *why* verification is stage-isolated here (a real design
   decision worth reading even if you skip the rest of the file first pass).
2. [`src/kernels.cuh`](src/kernels.cuh) — the elevation-map layout, the tuned algorithm constants
   (friction, thresholds, weights), and the `FootholdQuery`/`FootholdResult` records — the project's
   one-place contracts.
3. [`src/kernels.cu`](src/kernels.cu) — the heart: `solve_plane_3x3`'s Cramer's-rule derivation, then
   the four kernels in pipeline order. The single most interesting thing to read: the `solve_plane_3x3`
   comment block and the slope/roughness kernel's naming note — a real variable-shadowing bug this
   project shipped and caught (THEORY.md §How we verify correctness tells the full story).
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the line-by-line CPU twin of every kernel.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers (copied, not shared — §4 rule).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **ETH Zurich's `elevation_mapping` / `grid_map`** — the open-source ecosystem this project's whole
  shape (a 2.5-D grid with per-cell traversability layers) is modeled on; study its variance-aware
  fusion and its GPU-accelerated successor work for what a fielded version adds over this teaching core.
- **MIT Cheetah-family controllers** — simpler, faster heuristic foothold scoring tuned for very high
  gait rates; compare their speed/fidelity trade against this project's full per-cell plane fit.
- **ANYbotics ANYmal / Boston Dynamics Spot's public materials** — commercial quadrupeds shipping this
  exact capability class; read their published papers for how variance/uncertainty (not modeled here)
  changes the fusion story.
- **07.09 (this repo)** — the *exact*, unbounded jump-flooding distance transform; this project's
  edge-distance kernel is the bounded, brute-force sibling of that algorithm, cross-referenced in its
  own header comment.
- **09.03 GPU Featherstone ABA/RNEA, 13.02 Centroidal MPC** (this repo) — the dynamics and gait-timing
  layers this project's output feeds, per SYSTEM_DESIGN §4.3 Chain C.
- **cuRobo, nvblox** (NVIDIA) — production GPU robotics libraries in the same spirit (hand-rollable
  primitives replaced by optimized, batched GPU kernels) applied to manipulation and mapping instead.

## Exercises

1. **Plot the artifacts:** open `demo/out/foothold_score.pgm` in any image viewer and overlay
   `demo/out/selected_footholds.csv`'s nominal vs. selected points (plot both `(x_nom_m,y_nom_m)` and
   `(x_sel_m,y_sel_m)`) — watch the selector pull queries away from the step edge and the rocks.
2. **Break the friction assumption:** raise `kFrictionMu` in `kernels.cuh` toward 1.0 and rebuild —
   watch gate C's step-edge veto weaken as the slope limit climbs past the step's local slope, and
   explain from the friction-cone derivation (THEORY §The problem) why that is physically correct.
3. **Add variance-awareness:** real elevation mappers (05.05) carry a per-cell height *uncertainty*,
   not just a point estimate. Extend `FootholdQuery`/the fusion kernel to discount cells with high
   uncertainty even when their point-estimate slope looks fine — the single biggest gap between this
   teaching core and ETH's `elevation_mapping` (README §Prior art).
4. **Shared-memory tiling:** the slope/roughness and edge-distance kernels re-read heavily overlapping
   windows across neighboring threads (up to 25x and 441x redundant global-memory reads respectively).
   Tile the input into shared memory per thread block and measure the speed-up — THEORY.md §The GPU
   mapping sets up exactly what to build.
5. **Warp-per-query selection:** the foothold-selection kernel is one thread per query, scanning up to
   ~80 cells sequentially. Rewrite it as one WARP per query (each lane handles a slice of the disc,
   then a warp-shuffle reduction picks the max) and measure whether it helps at this query count — and
   explain, from occupancy numbers, why it likely will not until the query count is much larger.

## Limitations & honesty

- **The foothold-selection kernel is this project's own addition**, not named in the catalog bullet
  ("slope, roughness, edge distance from elevation maps"). It is included because a scoring pipeline
  with nothing consuming the scores teaches only half the story — SYSTEM_DESIGN §4.3 Chain C names
  exactly this consumer step (13.02) as the next link, and CLAUDE.md §4.1's own README template asks
  every project to name its downstream consumer; implementing a minimal, honest version of that
  consumer here makes the "downstream consumer" claim demonstrable instead of asserted.
- **No uncertainty/variance modeling.** Real elevation maps (05.05) carry per-cell height confidence;
  this project treats every known height as exact. Production stacks (README §Prior art) discount
  low-confidence cells directly — Exercise 3 sketches the extension.
- **Synthetic terrain, by design (CLAUDE.md §8).** The ramp/step/rocks/hole are hand-composed with
  documented, known ground truth specifically so this project's own analytic gates can check the
  kernels against the truth, not just against each other — a real elevation map has no such oracle.
- **The plane-fit window is small and fixed (5x5 cells, 0.1x0.1 m)**, sized to a mid-size quadruped's
  foot contact patch; a different robot's foot size would need a different `kFitRadius` — not
  autotuned here.
- **The edge-distance search is bounded and brute-force**, not the exact, unbounded jump-flooding
  transform of 07.09 — a deliberate scope choice (README §algorithm) appropriate because footholds are
  only ever compared to *nearby* hazards, never distant ones.
- **Timings are teaching artifacts** — single-shot, one machine (RTX 2080 SUPER), kernel-only where
  labeled (CLAUDE.md §12).
- **Sim-validated only (CLAUDE.md §1):** this project's output — a foothold's map-frame coordinates —
  is exactly the kind of number a real gait planner would turn into leg motion. Everything here ran
  only against synthetic terrain; nothing is safety-certified, and any real-hardware use would demand
  the full testing ladder in [`PRACTICE.md`](PRACTICE.md) §3 plus an independent safety envelope.
