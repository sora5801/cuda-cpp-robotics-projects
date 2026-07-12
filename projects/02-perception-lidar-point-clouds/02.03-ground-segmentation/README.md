# 02.03 — Ground segmentation: RANSAC plane fit; Patchwork++-style GPU port

**Difficulty:** ★ beginner · **Domain:** 2. Perception — LiDAR & Point Clouds

> Catalog bullet (source of truth, verbatim): `★ Ground segmentation: RANSAC plane fit; Patchwork++-style GPU port`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

Ground segmentation answers one question for every LiDAR point in a scan: *is this the surface a robot
drives on, or is it something else?* It is usually the **first semantic decision** any LiDAR pipeline
makes — before clustering, before mapping, before planning — because almost everything downstream
either wants the ground removed (obstacle detection, clustering) or wants the ground *specifically*
(traversability, costmaps).

This is a **bundled catalog project** (CLAUDE.md §2): the bullet names two methods, both implemented in
full as this project's two milestones, sharing one designed scene and one eigensolver:

- **Milestone 1 — GPU RANSAC plane fit.** Samples random 3-point plane hypotheses, counts inliers for
  all of them in parallel, refines the winner by least squares. Fast, simple, and correct exactly where
  its assumption holds: the ground is **one infinite plane**.
- **Milestone 2 — Patchwork++-style GPU concentric-zone model (CZM).** A reduced-scope, honestly-scoped
  teaching port of Patchwork++ (Lee et al., IEEE RA-L 2022): partitions the scan into ~160 small local
  patches (a polar grid centered on the sensor) and fits an independent plane per patch, with a simple
  region-growing rule that lets the recovered ground *bend* — handling slopes and multiple levels that
  defeat a single global plane. See [THEORY.md](THEORY.md) "Where this sits in the real world" for
  exactly what the full research system adds beyond this teaching version.

Both milestones are implemented **fully**, not stubbed — this is not a case of one being the "real"
method and the other "documented-only" (CLAUDE.md §2's bundling rule). To make their behavior *provably
different*, the demo runs on a **designed scene** (`scripts/make_synthetic.py`): a flat segment (RANSAC's
home turf), an 8° ramp, and a raised plateau — a single plane can fit only one of the three, and the
demo's `single_plane_failure` / `czm_recovery` gates measure that designed contrast directly. The scene
also carries six standing obstacles and a floating canopy overhang so both methods are graded on
*rejecting* non-ground, not just recognizing ground.

## What this computes & why the GPU helps

**Milestone 1** computes a **batched-solve + sampling** problem: `K = 1,024` random-triplet plane
hypotheses, each scored against **all** `N ≈ 160,000` points — a `K × N ≈ 165,000,000`-evaluation
workload that is embarrassingly parallel (every (hypothesis, point) pair is an independent distance
check) and is mapped **one GPU block per hypothesis**, each block's threads reducing their block's
inlier count via shared memory. The refinement step (least-squares plane fit over the winning
hypothesis's inliers) is a **reduce** (accumulate a 3×3 covariance matrix) into a tiny **batched-solve**
(one 3×3 eigendecomposition).

**Milestone 2** computes a **patch-parallel map + many small batched-solves**: point → polar patch id is
a **map** (one thread per point); the patch fit is **160 independent 3×3 eigendecompositions running in
parallel**, one per patch (one GPU block per patch-*column*, processing that column's two radial rings in
sequence) — genuinely useful parallelism at a much *smaller* scale than Milestone 1's, a deliberate,
named contrast (see THEORY.md "The GPU mapping").

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** Perception, immediately after point-cloud preprocessing — the first *semantic*
  split of a scan (SYSTEM_DESIGN.md §1's autonomy-stack layer diagram: `[Sensors] → [Perception] →
  [State estimation / World model] → ...`; ground segmentation sits at the perception→world-model
  boundary).
- **Upstream inputs:** a `PointCloud` (SYSTEM_DESIGN.md §3.6 message shape) — typically **already
  downsampled**, by name: [`02.01`](../02.01-voxel-grid-downsampling-with-gpu-spatial-hashing/README.md)'s
  voxel-grid output slots directly into this project's input layout (same interleaved-xyz, sensor-frame
  convention).
- **Downstream consumers:** [`02.04`](../02.04-euclidean-clustering-via-gpu-union-find/README.md)
  Euclidean clustering — obstacles only exist as *clusters* once the ground is removed from the cloud, so
  02.04's input is this project's `not-ground` mask applied to the scan; and traversability/costmap
  consumers by name — [`14.02`](../../14-locomotion-wheeled/14.02-traversability-costmaps-fusing-semantics/README.md)
  traversability costmaps and [`23.01`](../../23-navigation-stack/23.01-gpu-costmaps/README.md) GPU
  costmaps both want the `ground` mask *specifically* (drivable-surface evidence), the mirror image of
  what 02.04 wants.
- **Rate / latency budget:** a spinning LiDAR scan arrives at 10–20 Hz (SYSTEM_DESIGN.md §1.1); ground
  segmentation must complete well inside that period to leave headroom for clustering/costmapping in the
  same cycle. Measured on this project's ~160k-point scene (RTX 2080 SUPER): **RANSAC full-scene
  ≈ 5.9 ms** (generate + evaluate + accumulate + refine, unwarmed single-shot) and **CZM ≈ 0.8 ms**
  (patch ids + sort + fit/classify) — both comfortably inside a 50–100 ms budget with an order of
  magnitude to spare for everything else in the cycle.
- **Reference robot(s):** the **warehouse AMR** (SYSTEM_DESIGN.md §2.1 — ground segmentation feeds its
  costmap directly) and the **autonomous-vehicle stack** (§2.5 — road-surface segmentation is the same
  problem at highway scale, with the same slope/curb edge cases this project's ramp/plateau scene
  exercises).
- **In production:** a tuned, temporally-filtered version of exactly these algorithms (or a learned
  segmentation network trained to imitate them) — see [`README.md` "Prior art"](#prior-art--further-reading).
- **Owning team:** Perception (SYSTEM_DESIGN.md §5.1) — usually the same team that owns point-cloud
  preprocessing and hands its output to Planning/Controls' costmap consumers.

## The algorithm in brief

- **RANSAC hypothesis generation** — counter-based per-hypothesis xorshift32 RNG streams, random
  3-point triplets, degenerate-triplet rejection (near-collinear/coincident points). See
  [THEORY.md "The algorithm"](THEORY.md#the-algorithm).
- **RANSAC batched evaluation** — the K×N inlier-counting kernel, one block per hypothesis.
- **RANSAC least-squares refinement** — 3×3 covariance + eigendecomposition over the winning
  hypothesis's inliers (smallest-eigenvalue eigenvector = plane normal; cites
  [33.01](../../33-foundational-libraries/33.01-batched-small-matrix-linalg/README.md)).
- **RANSAC iteration-count formula** — the classical `k = log(1-p) / log(1-w^3)` bound, derived and
  *checked* against this scene's measured inlier ratio (the `ransac_formula` gate).
- **Patchwork++-style concentric-zone model (CZM)** — polar patch partition (zones by range, rings by
  sub-range, sectors by azimuth, density-adaptive sector counts); per-patch PCA plane fit; uprightness +
  flatness tests; a height-carry region-growing rule between a column's two radial rings.

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/ground-segmentation.sln`](build/ground-segmentation.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/ground-segmentation.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **Thrust** (`thrust::stable_sort_by_key`, `thrust::sequence`,
`thrust::lower_bound`) for the CZM's patch sort + vectorized boundary search — header-only, part of the
CUDA Toolkit, no extra install; see `build/ground-segmentation.vcxproj`'s `CudaCompile` comments for the
two MSVC/CCCL compiler flags this requires (ratified from project 02.01's identical fix). No other
optional dependency; default policy applies (CUDA toolkit + C++17 standard library only).

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at.

## Data

Synthetic by default (CLAUDE.md §8): a **designed** 3-level-ground scene (flat + 8° ramp + raised
plateau), six standing obstacles, and a floating canopy overhang — 161,836 points, exact per-point
ground truth, generated by `scripts/make_synthetic.py` (seed 42, xorshift32, stdlib only). No public
dataset applies (no dataset ships pre-labeled with a *designed* single-plane-defeating geometry — the
whole point of this scene is that it is engineered, not recorded). Full field documentation, byte
layout, and checksum in [`data/README.md`](data/README.md).

## Expected output

The demo runs **both milestones**, verifies every GPU stage against an independent CPU twin (8
`VERIFY(...)` lines — see [`THEORY.md` "How we verify correctness"](THEORY.md#how-we-verify-correctness)
for which are near-bit-exact-with-tolerance vs. independently-derived), and scores both against the
scene's ground truth with **6 gates**, all measured on the committed sample (RTX 2080 SUPER, Release|x64)
and recorded verbatim in [`demo/expected_output.txt`](demo/expected_output.txt):

| Gate | Measured result | What it proves |
|------|------------------|-----------------|
| `ransac_flat` | angle 0.084° from vertical, offset error 5.2 mm, precision 0.981, recall 1.000 | RANSAC is accurate and reliable on its home turf (a near-field flat-only crop). |
| `ransac_formula` | measured `w = 0.570` → `k_needed = 33.8` ≤ `K = 1024` | the classical iteration-count bound holds — `K=1024` is comfortably (30×) more than needed. |
| `single_plane_failure` | **93.25%** of true ramp+plateau ground misclassified as *not ground* | the designed failure: one global plane cannot also fit a sloped/raised level. |
| `czm_recovery` | overall precision 0.984, recall 0.969, IoU 0.954; ramp recall 0.971; plateau recall 0.983 | the CZM recovers almost all the ground RANSAC misses — the reason Milestone 2 exists. |
| `overhang` | CZM canopy false-positive rate **0.00%** (ceiling 2%) | the safety-relevant gate: overhead canopy is never called "ground" (see PRACTICE.md — calling it ground would let a planner treat the space beneath as driveable). |
| `obstacle_rejection` | CZM obstacle false-positive rate 8.96% (ceiling 13%; RANSAC's rate on the same points: 3.97%, `[info]`) | standing obstacles are mostly rejected; the residual rate is an honest, named limitation (obstacle *bases* sit at ground height by construction — see PRACTICE.md "the curb problem"), not a bug. |

Every floor above is **measured then margined** (CLAUDE.md §12) from an actual run, not guessed — see
`src/main.cu`'s gate-constant declarations for the exact tolerance reasoning.

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — the shared contract: point-cloud layout, `PlaneModel`, the
   RANSAC hypothesis-RNG formulas, the CZM patch-id polar formula, and the shared symmetric-3×3
   eigensolver (`jacobi_eigen_3x3`) both milestones' plane fits use.
2. [`src/kernels.cu`](src/kernels.cu) — the GPU kernels: RANSAC's `generate → evaluate → accumulate →
   refine` chain, then the CZM's `compute_patch_ids → sort_and_index (Thrust) → fit_and_classify`
   chain — read `czm_fit_and_classify_kernel`'s header comment for the region-growing walkthrough.
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the CPU correctness oracles; its file header
   explains which twins are shared-formula (drift detectors) vs. genuinely independent re-implementations.
4. [`src/main.cu`](src/main.cu) — orchestration: loads the scene, runs both RANSAC instances and the
   CZM, verifies, scores the 6 gates against ground truth, writes the artifacts.
5. [`scripts/make_synthetic.py`](scripts/make_synthetic.py) — the designed scene's full geometry and
   the reasoning behind every design choice (module docstring).
6. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, and `paths.h`'s data/artifact resolution.

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **Patchwork++** (Lee, Jung, Yoon, Kim — IEEE RA-L 2022) — the real system this project's Milestone 2
  is a reduced-scope port of. Adds adaptive ground-likelihood estimation, a Gaussian-model seed
  selection, temporal reversion, and RNR (reflected noise removal) this teaching version omits — see
  THEORY.md "Where this sits in the real world" for the full comparison.
  [github.com/url-kaist/patchwork-plusplus](https://github.com/url-kaist/patchwork-plusplus)
- **Patchwork** (Lim et al., IEEE RA-L 2021) — the original concentric-zone model this project's CZM
  cites for the zone/ring/sector partition idea specifically.
- **PCL's `SACSegmentation`** — the textbook RANSAC-plane implementation most robotics engineers meet
  first; study its API for how RANSAC composes with other PCL filters in a real pipeline.
- **Autoware's `ground_segmentation` package** (`ray_ground_filter`, `scan_ground_filter`) — a
  production autonomous-driving stack's actual ground filters; compare its ring-based scan-line approach
  to this project's polar-patch approach.
- **PCL's `RandomSampleConsensus` framework** — the general RANSAC template PCL's `SACSegmentation`
  specializes; worth reading for how a production codebase generalizes RANSAC beyond just planes.
- **[33.01](../../33-foundational-libraries/33.01-batched-small-matrix-linalg/README.md)** — this
  repository's foundations flagship for batched small-matrix linear algebra; the eigensolver this
  project hand-rolls (Jacobi rotation) is exactly the kind of operation 33.01 teaches how to batch at
  scale.

## Exercises

3–5 "try this next" extensions for the learner, ordered easiest first.

1. **Tune the thresholds.** Change `kRansacInlierThresholdM` or `kCzmClassifyDistM` in `kernels.cuh` and
   rerun. Watch `obstacle_rejection`'s false-positive rate move — this is the exact parameter real
   systems tune to trade obstacle-base rejection against ground recall on rough terrain.
2. **Add a fourth ground level.** Extend `scripts/make_synthetic.py` with a second ramp/plateau pair and
   `kernels.cuh`'s `kCzmRingsPerZone` from 2 to 3; observe how region growing chains across more rings.
3. **Implement the real seed-selection rule.** This project uses a height-*margin* seed rule (points
   within a fixed band of the patch minimum); Patchwork++ uses the lowest N% by height. Implement that
   version in `czm_fit_and_classify_kernel` and `czm_fit_and_classify_cpu`, and measure whether it
   changes `czm_recovery`'s numbers.
4. **Batch the RANSAC refinement across many candidate planes** (not just the single best) using
   33.01's batched-solve techniques, and compare the top-5 hypotheses' refined planes.
5. **Add temporal filtering.** Real systems (and Patchwork++ itself) revert a patch's "ground" verdict if
   it disagrees too much with the previous frame's fit. Simulate two consecutive frames from a slightly
   moved sensor pose and add this reversion rule.

## Limitations & honesty

- **Milestone 2 is an honest, reduced-scope teaching port of Patchwork++**, not a reproduction. It omits:
  adaptive ground-likelihood estimation (a Gaussian model per patch, not a fixed-margin height rule),
  reflected-noise removal (RNR), temporal reversion across frames, and the paper's specific seed-point
  selection (lowest N% by height vs. this project's height-margin rule) — see THEORY.md for the full
  list and why each was scoped out.
- **The obstacle-base ambiguity is real, not swept under the rug.** Both milestones misclassify a
  measurable fraction of standing obstacles' *lowest* points as ground (CZM: 8.96%, RANSAC: 3.97%,
  measured on this scene) — because a box or pole's base sits *at* ground height by construction. This
  is a genuine, named limitation of height-threshold ground segmentation (PRACTICE.md "the curb
  problem"), not a bug this project hides behind a loose gate.
- **The scene is synthetic and designed, not recorded.** It deliberately has no room-bounding walls (an
  explicit scope decision documented in `scripts/make_synthetic.py`'s module docstring, made after
  measuring that walls dominated the point count and could out-vote true ground in RANSAC's own
  hypothesis pool on an early version of this scene) — real scans have walls, ceilings, and far more
  scene complexity than this project's controlled setting.
- **Sim-validated only.** Nothing here has been run on a physical robot or a physical LiDAR, and nothing
  in this repository is safety-certified (no ISO 13482, no UL 4600). If this project's ground/not-ground
  output were ever wired into a real robot's motion, it is the owner's decision and responsibility, and
  it would need the validation ladder PRACTICE.md §3 describes before any free-running use.
- **No temporal or multi-sensor fusion.** Real ground segmentation systems cross-check with camera
  semantics, wheel odometry, or IMU-derived ground-plane priors; this project reasons from one static
  LiDAR scan alone.
