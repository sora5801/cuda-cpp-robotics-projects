# 14.02 — Traversability costmaps fusing semantics + geometry

**Difficulty:** intermediate · **Domain:** 14. Locomotion — Wheeled & Tracked

> Catalog bullet (source of truth, verbatim): `Traversability costmaps fusing semantics + geometry`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

**An off-road wheeled robot's world model asking two independent witnesses the same question:
"can I drive here?"** This project fuses TWO evidence channels into one traversability costmap for a
25.6x25.6 m off-road patch (256x256 cells @ 0.10 m/cell): a **geometric** channel (slope, step height,
and roughness, fit from a synthetic elevation map — the wheeled-vehicle cousin of
[13.03](../../13-locomotion-legged/13.03-foothold-scoring-kernels/)'s foothold scorer) and a
**semantic** channel (a six-class per-cell label — dirt/gravel/grass/vegetation/water/unknown — with a
simulated segmentation-net confidence, the kind of signal [12.x](../../12-ml-ai/)/[30.x](../../30-field-robotics/)
projects would actually produce). Four GPU kernels turn those two channels into one deliverable: a
fused cost in `[0,1]` with two independent hard vetoes (impassable geometry; standing water,
regardless of confidence) and a weighted blend everywhere else, followed by a curvature-free
speed-limit layer that makes the costmap directly actionable for a downstream controller. The demo
builds one synthetic scenario — rolling terrain, an 18-degree berm, a 0.5 m V-shaped ditch, a rock
patch, a water pool sitting inside geometrically flat ground, and a vegetation patch that is
geometrically noisy but semantically benign — and checks the fused result against the scenario's own
designed ground truth: the two cases where geometry and semantics **disagree** are the heart of the
teaching (README "Expected output" and THEORY.md both center on them). Every kernel named in the
catalog bullet — geometry, semantics, fusion — is implemented in full; the speed-limit layer is this
project's own consumer-facing addition (README §Limitations explains why, mirroring 13.03's precedent
for adding a minimal, honest downstream-consumer step).

## What this computes & why the GPU helps

Per costmap: 65,536 cells, each independently fitting two windowed elevation gathers (a 7x7
least-squares plane for slope/roughness, a 5x5 max-min gather for step height), one confidence-weighted
semantic lookup, one two-hard-veto fusion, and one closed-form speed bound.

- **Pattern:** two per-cell **stencils** (the geometric layer's wide plane-fit window and its own
  tighter step-height window — different-shaped hazard signals from the same input, THEORY.md §The
  algorithm), then three per-cell pure **maps** (semantic cost, fusion, speed limit) — a deliberately
  MORE UNIFORM pipeline shape than 13.03's mixed per-cell/per-query design, because a costmap has no
  natural "query" stage: every cell needs an answer, not just a handful of nominated points.
- **Measured reality** (RTX 2080 SUPER, this demo's 256x256 map): the geometric layer — the widest
  windows in the pipeline, touching all 65,536 cells twice each (slope/roughness pass, then the
  step-height pass) — runs in 0.92 ms on the GPU against 12.12 ms on one CPU core (**~13x**); fusion
  and the speed-limit layer are both sub-0.3 ms GPU-side, with GPU-vs-CPU comparisons at this problem
  size dominated by measurement noise (an honest single-shot-timing caveat, [`demo/README.md`](demo/README.md)).
  Numbers are printed on every demo run's `[time]` lines — a **teaching artifact, never a benchmark
  claim** (CLAUDE.md §12).
- **Why this scales to a real robot:** a production stack rebuilds this costmap every time the
  elevation map or the semantic labels update (5-20 Hz, README §System context) — at CPU speed, the
  geometric layer alone already eats a meaningful slice of even the loosest of those budgets; on the
  GPU the whole four-kernel pipeline fits comfortably inside 1.3 ms with headroom for the local planner
  built on top of it.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** the boundary between **state estimation/world model** (an elevation map plus a
  semantic segmentation map) and **planning** — the same costmap layer SYSTEM_DESIGN §1's stack diagram
  marks with "(+23 costmaps)" under STATE ESTIMATION / WORLD MODEL, specialized here to OFF-ROAD
  wheeled locomotion (domain 14) rather than 23.01's structured-environment DWA costmap.
- **Upstream inputs:** a 2.5-D elevation map (message shape: `GridMap`-like, SYSTEM_DESIGN §3.6's
  `OccupancyGrid` sketch generalized from occupancy bytes to floats — this project's flattened,
  row-major `elevation_m[W*H]`), from an elevation mapper such as project
  [05.05](../../05-slam-mapping-localization/05.05-elevation-mapping-with-uncertainty-for-legged/); and
  a per-cell semantic label + confidence map (message shape: a label `Image`-like grid, SYSTEM_DESIGN
  §3.6's `Image` sketch generalized from pixel channels to a class id, this project's
  `semantic_class[W*H]` + `confidence[W*H]`), from a segmentation network such as a project in domain
  [12](../../12-ml-ai/) (e.g. `12.01` TensorRT deployment) run on imagery from a field-robotics
  perception stack ([30.x](../../30-field-robotics/)).
- **Downstream consumers:** an off-road local planner/controller. This project names its consumer
  explicitly: **[`14.01` MPPI off-road racing with learned GPU dynamics](../14.01-mppi-off-road-racing-with-learned-gpu-dynamics/)**
  — the fused `fused_cost[W*H]` layer is exactly the per-cell running-cost term a sampling controller's
  rollout scorer would add to every candidate trajectory's cost, and `speed_limit_mps[W*H]` is a
  per-cell constraint/penalty term on the commanded `Twist.linear.x` at whatever cell a rollout's state
  currently occupies (SYSTEM_DESIGN §3.6's `Twist` message shape).
- **Rate / latency budget:** costmap layers sit in SYSTEM_DESIGN §1.1's **5-20 Hz** local-costmap row
  (the same row 23.01's README quotes) — this project's off-road scale (25.6x25.6 m vs. 23.01's smaller
  indoor patch) trades a coarser cell for a wider sensing horizon at a comparable rate. The measured GPU
  pipeline (~1.3 ms total across all four kernels) leaves the overwhelming majority of even a 20 Hz
  (50 ms) tick for elevation mapping, semantic segmentation, and the local planner built on top.
- **Reference robot(s):** the **warehouse AMR** (SYSTEM_DESIGN §2.1, generalized outdoors — the same
  "2D/3D LiDAR + depth cameras -> costmap -> local planner" block, off-road) and the
  **autonomous-vehicle stack** (§2.5's off-road/rural driving case, where domain 14's wheeled dynamics
  block sits directly beside the costmap this project builds). Domain 14's own catalog spans exactly
  this off-road wheeled-vehicle territory.
- **In production:** ETH Zurich/ANYbotics's `elevation_mapping`/`grid_map` ecosystem (13.03's closest
  production analogue) increasingly adds a learned semantic layer on top of its geometric one — exactly
  this project's fusion story; NASA/JPL's Mars-rover traversability analysis (slope/roughness/step
  limits derived from wheel geometry, evaluated before every drive command) is the closest real-world
  precedent for this project's wheeled-vehicle-derived hard vetoes; DARPA RACER-era off-road autonomy
  work fuses exactly these two channels (geometric LiDAR-derived costmaps + learned semantic
  traversability) for high-speed unstructured terrain. THEORY.md §Where this sits in the real world
  expands each comparison.
- **Owning team:** controls & autonomy, specifically the navigation/costmap specialization
  (SYSTEM_DESIGN item 5) — the team that also owns elevation mapping (05), semantic segmentation
  consumption (12), and the local planner/controller (14.01/23) this project's output feeds.

## The algorithm in brief

- **Geometric layer** — a least-squares plane fit (`z = a*x + b*y + c`, Cramer's-rule solve, 13.03's
  technique) over a 7x7 window gives slope and roughness; a SEPARATE, tighter 5x5 max-min gather gives
  step height — two independently-sized windows because a smooth plane fit alone blurs away a sharp
  discrete edge (THEORY.md §The algorithm walks through why numerically). → [THEORY.md](THEORY.md)
  §The math, §The algorithm.
- **Semantic layer** — a per-cell class prior cost, pulled toward a documented pessimistic fallback as
  confidence drops: `semantic_cost = confidence*prior[class] + (1-confidence)*pessimistic_prior`. →
  THEORY §The math (the confidence-weighting derivation).
- **Fusion** — two INDEPENDENT hard vetoes (impassable slope/step, derived from vehicle rollover
  geometry and wheel-radius friction-cone climbing; water, regardless of confidence) forcing cost to
  exactly 1.0, and a weighted blend of the geometric and semantic costs everywhere else — letting a
  confident, cheap semantic reading "rescue" a geometrically noisy cell. → THEORY §The two-channel
  fusion problem (the conservative-max-vs-optimistic-blend design choice and its failure modes).
- **Speed limit** — a curvature-free stopping-distance bound, `v = min(v_max, sqrt(2*a_avail(cost)*d_stop))`,
  turns the fused cost into an actionable m/s ceiling — the layer [14.01](../14.01-mppi-off-road-racing-with-learned-gpu-dynamics/)'s
  MPPI rollout scorer would consume directly. → THEORY §The math (the friction-circle derivation).

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/traversability-costmaps-fusing-semantics.sln`](build/traversability-costmaps-fusing-semantics.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/traversability-costmaps-fusing-semantics.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none** — CUDA runtime + C++17 standard library only. No
cuBLAS/cuFFT/Thrust: every kernel here is a hand-rolled per-cell map or windowed gather, exactly the
size CLAUDE.md §1's "no black boxes" rule asks a robotics engineer to be able to write by hand.

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) — including
**the three artifacts to view**.

## Data

The committed sample is a **scenario RECIPE**, not a recorded map or a committed grid:
`data/sample/traversability_scenario.csv` (2546 bytes, synthetic, rock placement seeded 42) — a
background rolling-terrain ripple, an 18-degree berm, a 0.5 m V-shaped ditch, 10 scattered rock domes,
a high-frequency vegetation-canopy bump, seven semantic regions covering the six-class palette, and a
6-waypoint teaching-transect path, all in six non-overlapping map bands with documented coordinates.
`src/main.cu`'s `build_elevation()`/`build_semantics()`/`build_transect()` turn the recipe into the
actual 65,536-cell grids and transect at start-up — deterministically, with zero run-time randomness
(the recipe already carries the one random choice, where the rocks sit, as literal numbers).
`scripts/download_data.ps1`/`.sh` are honest no-ops: no public dataset ships exact per-cell geometric
AND semantic ground truth the way a hand-designed synthetic scenario can. Full field-by-field format
and the checksum: [`data/README.md`](data/README.md).

## Expected output

Six stable lines — banner, `PROBLEM:`, `SCENARIO:`, `VERIFY: PASS`, `ARTIFACT:`, `RESULT: PASS` —
checked as a subset diff by [`demo/expected_output.txt`](demo/expected_output.txt). Two layers of
verification, described fully in [THEORY.md](THEORY.md) §How we verify correctness:

1. **VERIFY (four stage-isolated GPU-vs-CPU gates):** each of the four kernels is checked against its
   CPU twin fed *identical* upstream arrays, isolating each gate to exactly one kernel (measured worst
   case: slope 1.05e-6 rad, step height exactly 0 (a pure min/max gather — no accumulation, no rounding
   gap possible), roughness 2.24e-8 m, semantic cost 6.0e-8, fused cost 6.0e-8, veto-reason mismatches
   0/65536, speed limit exactly 0).
2. **Two DESIGNED-DISAGREEMENT analytic gates, run on the real, all-GPU pipeline** — this project's
   central teaching point:
   - **Gate A (flat-but-water):** the water pool sits inside geometrically near-flat ground (measured
     mean geo_cost 0.1043 — "geometry alone looks safe") yet **every** sampled cell is vetoed (224/224,
     all by the semantic-water veto, mean fused_cost exactly 1.0000) — semantics wins on its own terms.
   - **Gate B (rough-but-vegetation):** the vegetation-canopy patch shows meaningfully elevated geometry
     (measured mean geo_cost 0.4442 — "geometry alone looks bad") yet **no** cell is hard-vetoed
     (0/1870), the fused cost stays valid (0.4591, at or under the 0.60 validity bound), and the speed
     limit is measurably reduced but nonzero (2.224 m/s against a 2.50 m/s cruise cap) — semantics
     rescues the cell, at reduced speed.
3. **Two pure-geometry analytic gates:** the ditch's wall is hard-vetoed (fused_cost 1.0000) even though
   it is labeled cheap GRAVEL — geometry vetoes on its own terms, regardless of semantics; the berm's
   measured mean slope (17.861 deg) tracks its constructed 18.00 deg within a ±1.5 deg tolerance.

## Code tour

A guided reading order through `src/`:

1. [`src/main.cu`](src/main.cu) — the whole pipeline in plain sight: scenario synthesis -> the
   four-gate, stage-isolated VERIFY stage -> the real all-GPU PIPELINE stage -> the two
   designed-disagreement gates + two pure-geometry gates -> artifacts. Its file header explains *why*
   verification is stage-isolated (13.03's technique, reused for a four-kernel pipeline).
2. [`src/kernels.cuh`](src/kernels.cuh) — the elevation-map and semantic-map layouts, the six-class
   palette and its prior costs, the wheeled-vehicle constants the two hard-veto limits are derived
   from, and the cost-vs-13.03's-score naming distinction — the project's one-place contracts.
3. [`src/kernels.cu`](src/kernels.cu) — the heart: `geometric_layer_kernel`'s two independent windows
   (read the comment on why step height needs its OWN, tighter gather), and `fusion_kernel`'s two
   independent hard vetoes — the single most interesting thing to read in this project.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the CPU twin (deliberately double-precision in
   the plane fit, a documented departure from 13.03's float-both-sides choice — the file header
   explains why).
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers (copied, not shared — §4 rule).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **ETH Zurich's `elevation_mapping`/`grid_map` ecosystem (and ANYbotics's commercial descendants)** —
  13.03's closest production analogue, increasingly extended with a learned semantic traversability
  layer fused on top of the geometric one, exactly this project's two-channel story.
- **NASA/JPL Mars-rover mobility & traversability analysis** — the closest real-world precedent for
  deriving hard slope/step limits from wheel geometry and vehicle rollover/tip-over margins before ever
  issuing a drive command; read their published mobility papers for how much more detailed a real
  wheel-terramechanics model gets than this project's friction-cone/rollover-geometry teaching version.
- **DARPA RACER-era off-road autonomy research** — high-speed unstructured-terrain autonomy that fuses
  geometric LiDAR costmaps with learned semantic traversability classifiers, the closest modern
  military/research analogue to this project's exact fusion shape.
- **23.01 (this repo)** — the structured-environment (Nav2-style) sibling costmap: inflation, raytrace
  clearing, multi-layer fusion, and a DWA local planner closing the loop. Compare its byte-cost `[0,254]`
  Nav2 convention against this project's float `[0,1]` cost — both are legitimate, documented choices;
  a real integration would quantize one into the other at a system boundary.
- **13.03 (this repo)** — the legged-robot sibling: the same "geometry -> per-cell traversability"
  idea, scored for discrete foothold placement instead of continuous wheeled drivability. Read its
  THEORY.md for the friction-cone derivation this project's own slope-traction limit reuses.
- **cuRobo, nvblox (NVIDIA)** — production GPU robotics libraries in the same spirit (hand-rollable
  primitives replaced by optimized, batched GPU kernels) applied to manipulation and mapping instead.

## Exercises

1. **Plot the artifacts.** Open `demo/out/traversability.pgm` and `demo/out/speed_limit.pgm` in any
   image viewer, and plot `demo/out/layers.csv`'s columns against `sample_index` (or `x_m`/`y_m`) —
   watch `geo_cost`, `semantic_cost`, and `fused_cost` diverge exactly at the water pool and the
   vegetation patch, and watch `speed_limit_mps` fall to zero at every hard veto.
2. **Push the vegetation patch past its rescue.** Raise `VEGBUMP`'s amplitude in
   `data/sample/traversability_scenario.csv` (regenerate with `scripts/make_synthetic.py` after editing
   its default, or edit the committed file directly for a quick experiment) until Gate B's own
   `vetoed=0/1870` requirement starts failing — find, empirically, the exact terrain roughness at which
   this project's own wheeled-vehicle step-height limit stops being "rescuable" by any semantic reading,
   and explain why a HARD veto (not just a weighted penalty) is the physically correct behavior there.
3. **Implement MAX-fusion as a real kernel.** `main.cu`'s "teaching comparison" computes the
   conservative MAX-fusion alternative post-hoc on the host from already-downloaded layers; promote it
   to a real fifth kernel (or a compile-time `#define` branch inside `fusion_kernel`) and re-run Gate B
   — then retune `VEGBUMP`'s amplitude until MAX-fusion's worst-cell cost actually crosses
   `kMaxValidCost` while the weighted blend does not, reproducing THEORY.md's documented failure mode
   with your own measured numbers.
4. **Add channel-reliability weighting.** `kWeightGeo`/`kWeightSem` are a flat 0.5/0.5 split
   (`kernels.cuh`). Real systems weight per-channel by measured reliability (e.g., a segmentation net's
   validation-set class-confusion rate, or a LiDAR's known range-noise floor) — extend `fusion_kernel`
   to accept per-cell or per-class weights instead of one global pair, and explain what new failure mode
   this introduces (a badly-calibrated weight silently masking one channel's genuine warning).
5. **Shared-memory tiling.** The geometric layer's two windows re-read heavily overlapping cells across
   neighboring threads (up to 49x and 25x redundant global-memory reads respectively — 13.03's exact
   optimization opportunity, reused here). Tile the input into shared memory per thread block and
   measure the speed-up — THEORY.md §The GPU mapping sets up exactly what to build.

## Limitations & honesty

- **The speed-limit kernel is this project's own addition**, not literally named in the catalog bullet
  ("Traversability costmaps fusing semantics + geometry"). It is included because a costmap with
  nothing consuming it teaches only half the story — README §System context names 14.01's MPPI as
  exactly the consumer this layer serves, and CLAUDE.md §4.1's own README template asks every project to
  name its downstream consumer; implementing a minimal, honest version of the interface that consumer
  would use makes the "downstream consumer" claim demonstrable instead of asserted (13.03's own
  foothold-selection kernel sets this precedent).
- **No NaN/sensor-dropout modeling**, unlike 13.03's hole-riddled elevation map — a deliberate scope
  choice so the two-channel FUSION story, not sensor dropout, stays the center of the teaching. A real
  elevation mapper (05.05) absolutely produces holes; extending this project's geometric layer to
  propagate NaN through fusion's hard veto (13.03's exact discipline) is a natural, straightforward
  follow-on.
- **The MAX-fusion alternative is demonstrated, not shipped**, and — honestly — in this project's own
  demo scenario its WORST measured cell (0.5374) stays just under the validity bound (0.60), so it does
  not visibly misclassify a cell as invalid in this specific committed scenario, only get measurably
  closer to doing so (Exercise 2/3 invite pushing the terrain until it does). THEORY.md discusses the
  general argument in full; the printed `[info]` line reports the real, measured numbers either way —
  never a fabricated "it fails here" claim the committed scenario does not actually produce.
- **Synthetic scenario, by design (CLAUDE.md §8).** The berm/ditch/rocks/water/vegetation are
  hand-composed with documented, known ground truth specifically so this project's own analytic gates
  can check the kernels against the truth, not just against each other — a real fused elevation+semantic
  map has no such oracle.
- **The wheeled-vehicle constants (wheel radius, friction, track width, CoG height) are illustrative**,
  not measured from a real vehicle — PRACTICE.md §2 dates and caveats every one.
- **Timings are teaching artifacts** — single-shot, one machine (RTX 2080 SUPER), kernel-only where
  labeled (CLAUDE.md §12).
- **Sim-validated only (CLAUDE.md §1):** this project's output — a per-cell cost and speed limit — is
  exactly the kind of number a real off-road planner would turn into throttle and steering commands.
  Everything here ran only against a synthetic scenario; nothing is safety-certified, and any
  real-hardware use would demand the full testing ladder in [`PRACTICE.md`](PRACTICE.md) §3 plus an
  independent safety envelope.
