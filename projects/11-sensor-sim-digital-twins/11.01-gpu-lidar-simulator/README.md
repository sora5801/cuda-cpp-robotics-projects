# 11.01 — GPU LiDAR simulator: BVH raycasting + beam divergence, intensity, dropout noise

**Difficulty:** ★ beginner · **Domain:** 11. Sensor Simulation & Digital Twins

> Catalog bullet (source of truth, verbatim): `★ GPU LiDAR simulator: BVH raycasting + beam divergence, intensity, dropout noise`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

**Every perception, mapping, and planning project downstream of a real LiDAR needs test data before
any hardware exists — this project generates it.** It spins a simulated 32-channel LiDAR through a
full 360° sweep over a synthetic 2,264-triangle warehouse (floor, perimeter walls, shelving racks,
crates), casting every one of 32,768 beams through a **hand-built bounding-volume hierarchy** (build
your own before touching OptiX, CLAUDE.md §5) via **Möller–Trumbore** ray/triangle intersection, then
layers on the catalog bullet's three effect models: **beam divergence** (a small cone of jittered
subrays approximating a real beam's finite footprint), **intensity** (Lambertian radiometry from
per-material albedo, incidence angle, and range), and **dropout** (a range- and incidence-dependent
probability of losing a return, plus additive range noise). The demo builds the BVH once on the host,
raycasts the full frame on the GPU, and writes a `PointCloud`-shaped point cloud (`demo/out/cloud.csv`)
plus a native LiDAR range image (`demo/out/range_image.pgm`) — the picture this project exists to
produce. All three effect models are fully implemented (not a reduced-scope bundle); a hand-built
LBVH-on-GPU alternative to this project's host build is project 07.03's dedicated subject, named
honestly throughout as a scoping choice, not a gap.

## What this computes & why the GPU helps

Per frame: 32,768 beams x (1 central ray + 4 divergence subrays) x a BVH traversal (~10 node visits to
depth 10) x a handful of Möller–Trumbore tests per leaf — independent, embarrassingly parallel raycasts
that only interact through the SHARED, READ-ONLY mesh/BVH they all traverse.

- **Pattern:** batched sampling / map — one thread per BEAM (not per triangle, not per pixel): the
  repo's usual thread-per-problem shape (33.01/09.01/08.01/02.06), here applied to a TREE-STRUCTURED
  memory access instead of a flat array. That is the project's one genuinely new GPU idea: **divergent
  traversal** — neighboring threads in a warp can walk completely different root-to-leaf paths, because
  neighboring beams point in different directions (THEORY.md "The GPU mapping" measures the cost).
- **Measured reality (RTX 2080 SUPER):** the full 32,768-beam frame costs **~1.0 ms of GPU kernel
  time** versus **~230–290 ms on one CPU core** running the byte-for-byte identical algorithm
  sequentially — roughly a **200–280x** single-shot speed-up (teaching artifact, not a benchmark;
  [`demo/expected_output.txt`](demo/expected_output.txt) doesn't check the number, `[time]` lines do).
- **The BVH is what makes this tractable at all:** a brute-force scan of all 2,264 triangles per ray
  (5 rays/beam x 32,768 beams = ~163,840 rays) would be ~370M ray/triangle tests per frame; the BVH
  cuts each ray's real triangle-test count to single digits (THEORY.md "The algorithm" derives the
  depth bound that makes this guarantee, not a hope).

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** the **SIMULATION & DIGITAL TWIN** cross-cutting layer, feeding the head of the
  **SENSORS -> PERCEPTION** boundary — this project stands in for a physical LiDAR so every
  perception/mapping/planning project downstream can be built and tested before real hardware exists.
- **Upstream inputs:** none from the autonomy stack itself — a scene mesh (`data/sample/`, standing in
  for a CAD model, a prior map, or a simulator's world geometry) and a sensor configuration/pose are
  this project's own "sensors" (SYSTEM_DESIGN §3.6's message-shaped-struct convention still applies to
  the *output*, just not an *input* here — there is nothing upstream of a sensor simulator inside the
  autonomy stack itself).
- **Downstream consumers:** a `PointCloud` (SYSTEM_DESIGN §3.6: xyz + intensity, extended here with a
  `ring`/channel field the way real spinning-LiDAR drivers do) at the LiDAR-perception rate. SYSTEM_DESIGN
  §4.1 **Chain A** names this project explicitly, first in the chain:
  `[11.01 GPU LiDAR simulator] -> [02.06 ICP registration] -> [05.01 TSDF fusion] -> [07.09 distance
  field] -> [06.05 STOMP] -> [08.01 MPPI]` — this project's output is the very first arrow.
- **Rate / latency budget:** spinning LiDARs scan at **10–20 Hz** (SYSTEM_DESIGN item 1's rate table),
  budget **< 100 ms** per scan for the perception/mapping stage that consumes it; this project's
  measured ~1 ms GPU frame cost leaves that budget almost entirely to the DOWNSTREAM consumer, exactly
  as a real sensor (whose "compute cost" is fixed in silicon) would.
- **Reference robot(s):** the **warehouse AMR** (SYSTEM_DESIGN §2.1 — this project's own scene is
  literally a warehouse) and the **autonomous-vehicle stack** (§2.5, which names 1–5 real LiDARs);
  quadruped and quadrotor terrain/obstacle sensing (§2.3/§2.4) use the same simulator pattern with a
  different scan geometry (README §Exercises).
- **In production:** Isaac Sim's RTX LiDAR (OptiX-accelerated, physically based), CARLA's `sensor.lidar.ray_cast`,
  and Gazebo's `gpu_ray`/`gpu_lidar` plugin all occupy this exact role — GPU raycasting against a scene
  BVH, with progressively richer physical models (CARLA/Isaac add atmospheric attenuation, multi-bounce,
  and sensor-specific intrinsics this project's teaching core omits, README §Limitations).
- **Owning team:** simulation & tools (SYSTEM_DESIGN item 5's org map) — the team that builds the
  digital twins every other team (perception, controls/autonomy, QA) tests against before touching
  real hardware.

## The algorithm in brief

- **Median-split BVH construction** (host, once, at load) — recursively halve each node's triangle set
  by centroid position along its largest-extent axis, using `std::nth_element` for an O(N) partition
  per level; the count-based split GUARANTEES tree depth `<= ceil(log2(N/leaf_size))` regardless of
  scene geometry. → [THEORY.md](THEORY.md) §The algorithm.
- **Small-stack BVH traversal** (GPU + CPU oracle) — a fixed 64-entry per-thread stack, provably
  sufficient because of the depth guarantee above (not a "probably enough" guess). →
  THEORY §The GPU mapping.
- **Möller–Trumbore ray/triangle intersection** — solve for the hit point in the triangle's own
  barycentric basis via Cramer's rule, restructured so every rejection test (u, v, u+v, t range) can
  short-circuit before the next determinant is computed. → THEORY §The math.
- **Beam divergence** — the central ray plus `subray_count` rays evenly jittered around a small cone,
  keeping the nearest hit among all of them. → THEORY §The problem.
- **Intensity radiometry** — Lambertian: `intensity = gain * albedo * |cos(incidence)| / range^2`,
  clamped to `[0,1]` (sensor saturation), derived from solid-angle radiometry. → THEORY §The problem.
- **Dropout + range noise** — a per-beam deterministic xorshift32 + Box–Muller stream (08.01's exact
  generator, reseeded per beam) decides a range/incidence-dependent dropout probability and an additive
  Gaussian range perturbation. → THEORY §Numerical considerations.

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/gpu-lidar-simulator.sln`](build/gpu-lidar-simulator.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/gpu-lidar-simulator.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none** — CUDA runtime + C++17 standard library only. The
OBJ/CSV loaders, the BVH, and the ray/triangle intersection are all hand-rolled (no OptiX, no mesh
library — CLAUDE.md §5's "build your own BVH before touching OptiX" stance, applied literally).

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) — including the
**two artifacts to view**: `cloud.csv` (plot as a 3-D scatter) and `range_image.pgm` (open as an image).

## Data

The committed sample is **synthetic and RNG-free**: a scene mesh and a sensor spec are geometry and
constants, not recordings — `data/sample/warehouse_scene.obj` (2,264 triangles, ~68 KiB),
`materials.csv` (per-triangle-range albedo table), `sensor_config.csv` (scan pattern + noise-model
constants, seed 42), and `sensor_poses.csv` (`T_world_sensor`) — ~70 KiB total. All beam-level
randomness (dropout, range noise) is generated *inside the demo* from the committed seed; no public
LiDAR dataset ships the exact analytic ground truth (a flat floor, known albedos) this project's
verification gates need, so `scripts/download_data.ps1`/`.sh` are honest no-ops. Regenerate
byte-identically with `python scripts/make_synthetic.py`. Details, checksums, and every field's
meaning: [`data/README.md`](data/README.md).

## Expected output

Twelve stable lines — banner, `SCENE:`, `BVH:`, `PROBLEM:`, `VERIFY:`, five `CHECK:` lines, two
`ARTIFACT:` lines, `RESULT:` — checked as a subset diff by
[`demo/expected_output.txt`](demo/expected_output.txt); every `[info]`/`[time]` line above them carries
the actual measured numbers (not diffed — see `src/main.cu`'s "output contract"). Measured on the RTX
2080 SUPER reference machine:

1. **The §5 GPU-vs-CPU gate (VERIFY):** every one of 32,768 beams' hit/dropped decision matches the CPU
   oracle EXACTLY (0 mismatches); intensity agrees within rel tol 1e-3 (measured worst: 1.95e-4). Range
   uses a WIDER, separately-justified tolerance (rel 2e-2): 5 of 23,340 hit beams (0.02%) exceed rel
   1e-3, because the 5-ray divergence bundle's nearest-hit selection is a genuinely DISCONTINUOUS
   function near geometric silhouette edges — an ulp-level GPU/CPU rounding difference can occasionally
   flip which of two near-tied candidate surfaces "wins", moving the reported range by the geometric
   gap between them (centimeters), not by an ulp. Measured worst case: 1.166e-2, comfortably under the
   2e-2 gate. THEORY.md "Numerical considerations" walks the full argument.
2. **Ground-plane range gate:** a beam aimed at the open floor returns range 7.550235 m against the
   closed-form `h/sin(|elevation|) = 7.550234` m — relative error 7.7e-8.
3. **Inverse-square intensity gate:** normal-incidence intensity at R=1.5 m vs R=3.0 m ratios
   4.000000 (exact, to the printed precision).
4. **Dropout statistics gate:** 20,000 i.i.d. beams at a known (range, incidence) measure an empirical
   dropout rate of 0.02405 against a theoretical 0.02375 — inside a documented 5-sigma binomial bound
   of ±0.00538.
5. **Frame-level sanity gates:** the full demo frame's hit fraction (0.7123) and mean returned range
   (11.127 m) fall inside documented, measured-with-margin bounds.

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — the project's single source of truth: the `Triangle`/
   `Material`/`BvhNode` layouts (with the BVH's "children allocated in pairs" flattening scheme
   diagrammed), the `SensorConfig`/`SensorPose` structs, the channel-major beam-indexing contract, and
   the per-beam output layout. Read this FIRST.
2. [`src/main.cu`](src/main.cu) — load the mesh + materials + sensor config/pose (hand-rolled OBJ/CSV
   loaders), BUILD the BVH on the host (`BvhBuilder`, median-split by triangle count), run the §5
   VERIFY stage, run the three analytic gates and the two frame-level gates, pack the point cloud, and
   write both artifacts. The single most interesting thing to look at: `BvhBuilder::build()` — how
   little code a correct, depth-bounded BVH construction actually needs.
3. [`src/kernels.cu`](src/kernels.cu) — the heart: `intersect_bvh()` (the small-stack traversal),
   `moller_trumbore()` (the ray/triangle intersection), and `simulate_beam()` (the whole per-beam
   pipeline: direction -> raycast -> divergence -> radiometry -> dropout/noise) — the entire "GPU LiDAR"
   fits in about 150 lines once the pattern is right.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the line-by-line CPU oracle twin of every function
   in `kernels.cu`; diff the two side by side to see exactly what parallelization changed (spoiler:
   almost nothing but the qualifiers).
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers (copied, not shared — §4 rule).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **Möller & Trumbore (1997), "Fast, Minimum Storage Ray-Triangle Intersection"** — the exact
  intersection algorithm this project implements; THEORY.md's derivation follows their paper's
  structure closely and credits it by name at every formula it borrows.
- **Jacco Bikker, "How to Build a BVH" (blog series)** — the "children allocated in pairs" flattened
  node layout this project's `BvhNode` uses; a widely cited, approachable derivation of exactly the
  scheme implemented here, one step past what this project's teaching core covers (surface-area
  heuristic splitting, ordered traversal, packet tracing).
- **NVIDIA OptiX** — the production hardware-accelerated raytracing API (RT cores do the BVH traversal
  this project's kernel does in software); study its API shape once this project's traversal is
  familiar — CLAUDE.md §5's "build your own before touching OptiX" stance, completed.
- **Isaac Sim RTX LiDAR** — NVIDIA's OptiX-accelerated, physically based LiDAR sensor model inside
  Isaac Sim; the production descendant of exactly this project's pipeline, with atmospheric
  attenuation, multi-bounce, and per-product intrinsics this teaching core omits.
- **CARLA (`sensor.lidar.ray_cast`)** and **Gazebo (`gpu_ray`/`gpu_lidar` plugin)** — the two most
  widely used open-source GPU LiDAR simulators in robotics/AV research; both raycast a scene BVH from a
  spinning-scanner model, exactly this project's shape, at production scope.
- **Project 07.03 (Linear BVH build + stackless traversal)** — this repo's dedicated GPU-side BVH
  CONSTRUCTION project (LBVH via Morton codes); this project's host-built median-split tree is the
  simpler, correct-first alternative named honestly as a scoping choice (README §Limitations).

## Exercises

1. **View the artifacts:** plot `demo/out/cloud.csv` as a 3-D scatter (color by `intensity` or `ring`)
   and open `demo/out/range_image.pgm` as an image — find the walls, the shelving racks, and the
   crates' shadows in both.
2. **Break the divergence bundle:** set `SUBRAY_COUNT,0` in `sensor_config.csv` (divergence off) and
   rerun — the VERIFY gate's range tolerance should tighten dramatically (measure the new worst-case
   deviation); explain why from THEORY.md's near-tie argument.
3. **Multi-return:** the catalog's dropout model reports at most ONE range per beam. Extend
   `intersect_bvh()` to record the first TWO hits along the central ray (a real multi-return LiDAR
   reports both a near partial-occluder and a far surface) and add a second output channel.
4. **GPU BVH construction:** replace `build_bvh_median_split()`'s host build with project 07.03's
   LBVH/Morton-code GPU construction, and measure the frame-to-frame cost of rebuilding the tree every
   frame (useful for a scene with moving objects, which this project's static warehouse does not need).
5. **A second reference robot's scan geometry:** swap `sensor_config.csv`'s elevation range/channel
   count for a quadrotor's forward-facing solid-state pattern (README §System context) or an
   automotive 128-channel long-range spec, and see how hit fraction and mean range change on the same
   scene.

## Limitations & honesty

- **The BVH is built on the HOST, once, at load** — GPU BVH construction (Morton-code LBVH) is project
  07.03's dedicated subject; this project spends its teaching budget on the RAYCAST that walks the
  tree, and names the alternative honestly rather than silently omitting it.
- **Multi-return is not modeled** — a real LiDAR can report several returns per beam (e.g., a near
  fence and a far wall behind it); this project reports the single nearest surface found by the
  divergence bundle. README Exercise 3 sketches the extension.
- **Beam divergence is a nearest-hit approximation, not a footprint integral** — real divergence
  produces smeared/mixed-pixel returns at edges (partial illumination of two surfaces at once); this
  project's 5-ray bundle approximates "the closest surface in the footprint dominates" and does NOT
  reproduce edge blur (THEORY.md "The problem" states this explicitly).
- **No atmospheric or multi-bounce effects** — rain/fog attenuation, retroreflector saturation, and
  secondary bounces (a beam grazing one surface into another) are real LiDAR phenomena this teaching
  core omits; production simulators (Isaac Sim RTX LiDAR, CARLA) model several of them.
- **A single sensor origin per beam** — real spinning LiDARs have each laser physically offset by
  millimeters from the rotation axis; this project uses one shared origin for all channels (an error on
  the order of the offset, negligible at this scene's scale but real).
- **The VERIFY range tolerance is wider than the repo's usual 1e-3** (rel 2e-2, with the measured
  justification in README §Expected output and THEORY.md §Numerical considerations) — a genuine,
  documented consequence of argmin-over-independent-rays being discontinuous, not a loosened bug bar.
- **Timings are teaching artifacts** — single-shot, one machine (RTX 2080 SUPER), kernel-only where
  labeled; never a benchmark claim (CLAUDE.md §12).
- **Sim-validated only (CLAUDE.md §1):** this project's output is synthetic sensor data that downstream
  perception/mapping/planning projects would train and test against. Nothing here is a certified sensor
  model; a real deployment decision about what a physical LiDAR will actually see needs real sensor
  data and the validation ladder in [`PRACTICE.md`](PRACTICE.md) §3, not this simulator alone.
