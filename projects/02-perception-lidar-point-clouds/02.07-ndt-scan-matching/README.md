# 02.07 — NDT scan matching (Autoware-style map localizer)

**Difficulty:** intermediate · **Domain:** 2. Perception — LiDAR & Point Clouds

> Catalog bullet (source of truth, verbatim): `NDT scan matching (Autoware-style map localizer)`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

The Normal Distributions Transform (NDT) is the registration algorithm behind Autoware's
`ndt_scan_matcher` — the map-localizer real autonomous-vehicle stacks run at 10 Hz to answer "where
am I on this pre-built map?" from a live LiDAR scan. Instead of matching a scan point to its
nearest MAP POINT (02.06's ICP), NDT pre-compiles the map into a grid of local Gaussians — one
mean + covariance per voxel — and scores a transformed scan point against the smooth, closed-form
density of whichever voxel it lands in. No nearest-neighbor search, ever: voxel lookup is O(1)
direct indexing. This project builds that whole pipeline on the GPU (voxel-grid construction,
score/gradient/Hessian assembly, a multi-resolution coarse→fine Newton optimizer) and then runs the
SAME registration problem through a compact point-to-point ICP contrast from the SAME 240 perturbed
initial poses on the SAME synthetic corridor-into-a-room map, so a learner can measure — not just
be told — whether NDT's smoother objective really does converge from a wider range of starting
guesses, AND how it stacks up against a simpler baseline. Measured, honestly: the multi-resolution
schedule dramatically beats spending the identical iteration budget entirely at fine resolution
(65% vs. 7.5% at the smallest perturbation bin, `demo/out/basin_curve.csv`) — multi-resolution is
unambiguously the right design *within NDT*. Against the ICP contrast, this project's own compact
scene tells a more nuanced story than "NDT wins": ICP's accurate nearest-neighbor correspondence
search is *more* sample-efficient than NDT's voxel discretization on a map this small and this
simple (ICP converges 24.2% of the whole cohort and is more accurate besides, vs. NDT's 13.3%) — a
real, measured lesson about when reaching for NDT actually pays off (larger, noisier, more
ambiguous scenes where correspondence search itself becomes the bottleneck — THEORY.md "Where this
sits in the real world" says more). The smallest-perturbation bin's own 65% splits cleanly along
one axis this project measures directly: 0% for initial guesses offset along the corridor's
degenerate sliding direction, 84% for every other direction (`bin0_corridor_axis_split` in the demo
output) — production localizers resolve that specific direction with wheel/IMU odometry, not the
LiDAR match alone, exactly the scoping this project's cohort generator already applies to Z.

## What this computes & why the GPU helps

Per registration trial: build a ~600-voxel Gaussian map (once, reused across every trial) and then
run up to 27 Newton iterations, each iteration scoring ~1,300 scan points against their voxels and
reducing 28 scalars (a 6×6 Hessian's upper triangle + 6-entry gradient + score) into one update.

- **Pattern:** the voxel build is a **map + scatter-reduce** (each point atomically folds into its
  voxel's running statistics — 02.01's spatial-hash idiom, applied to a dense grid instead of a
  hash table); the score/gradient/Hessian assembly is a **map + tree-reduction** (01.17's exact
  28-scalar block-reduction pattern, applied to a Mahalanobis-distance objective instead of
  reprojection error).
- **Measured reality:** building both resolution levels of the voxel grid from 40,000 map points
  takes under half a millisecond of GPU kernel time; one full 240-trial × 2-algorithm cohort sweep
  (NDT-multires + NDT-fine-only, both GPU-orchestrated) plus the CPU-only ICP contrast finishes in
  under ten seconds end to end (the single-threaded ICP correspondence search, not the GPU NDT
  path, is the slow part — see "Limitations & honesty").
- **The GPU concept this project adds beyond 02.06/01.17:** a *dense, direct-indexed* voxel grid
  (not hashed — see "The algorithm in brief" and THEORY.md "The GPU mapping" for why a bounded,
  known map favors direct indexing over 02.01's hash table) built via a **two-pass** point-parallel
  accumulation (mean, then centered covariance) sandwiched around **voxel-parallel** finalize
  passes that regularize and invert each 3×3 covariance in place.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** **state estimation / localization** — NDT scan matching is THE production
  map-localizer lineage this repo studies (Autoware's `ndt_scan_matcher` node is named, not
  invented, by the catalog bullet); it sits between perception and the state estimator, producing a
  pose measurement, not a control command.
- **Upstream inputs:** a downsampled LiDAR scan (02.01's voxel-grid downsampling is the standard
  pre-filter a real pipeline would run before NDT — this project's synthetic scan stands in for
  that output directly, same `PointCloud` message shape) and a PRE-BUILT MAP (the output of a SLAM
  mapping pipeline, 05.xx's domain — this project treats the map as already-built survey data, the
  same assumption a deployed localizer makes).
- **Downstream consumers:** a pose-fusion EKF/UKF (04.xx's domain — a real stack never trusts NDT's
  raw pose alone; it fuses it with wheel odometry and IMU) and, through that filter, every planning
  and control node that needs "where am I" (06.xx motion planning, 08.xx control, 23.xx navigation).
- **Rate / latency budget:** production NDT localizers run at 10 Hz (SYSTEM_DESIGN item 1's
  localization budget) — this demo's single multi-resolution registration (27 Newton iterations
  over ~1,300–5,100 points) measures in the low single-digit milliseconds of GPU kernel time,
  comfortable headroom inside a 100 ms budget.
- **Reference robot(s):** the **autonomous-vehicle stack** (NDT's original production home) and the
  **warehouse AMR** (indoor map-based localization, the same algorithm at smaller scale) — both
  named explicitly in SYSTEM_DESIGN's reference-robot table.
- **In production:** Autoware's `ndt_scan_matcher`, PCL's `NormalDistributionsTransform`, and
  (increasingly) learned place-recognition + pose-refinement pipelines sit here; see "Prior art"
  below and THEORY.md "Where this sits in the real world" for how each differs from this project's
  teaching core.
- **Owning team:** localization/state-estimation (a sub-team of controls/autonomy, SYSTEM_DESIGN
  item 5) — adjacent to mapping/SLAM (who build and maintain the map this project consumes,
  PRACTICE.md §1) and perception (who hand it the filtered scan).

## The algorithm in brief

- **Two-pass voxel build** — mean, then centered covariance, per voxel, GPU-parallel with
  `atomicAdd(double*)` accumulators (chosen over Welford or the naive one-pass raw-second-moment
  trick — THEORY.md "Numerical considerations" explains why). → [THEORY.md](THEORY.md) §The algorithm.
- **Eigenvalue-floored covariance regularization** — a 3×3 Jacobi eigensolve (02.06's PCA-normal
  solver, cited) that floors near-zero eigenvalues (thin/flat voxels — walls!) so the inverse
  covariance stays well-conditioned. → THEORY.md §The problem — physics & engineering first.
- **The NDT mixture-model score** — `score = -d1*exp(-d2/2 * Mahalanobis²)`, with `d1,d2` derived
  from a Gaussian-plus-uniform-outlier mixture (the Biber & Straßer / Magnusson derivation, the
  exact Autoware/PCL parameterization). → THEORY.md §The math.
- **Analytic gradient + Gauss-Newton Hessian** via the chain rule through `R·x+t` into the
  Mahalanobis form (01.17's rotation-Jacobian formula, reused) — including the extra curvature term
  that can make NDT's Hessian INDEFINITE, unlike ICP's `JᵀJ`. → THEORY.md §The math.
- **Multi-resolution Newton** — coarse (2.0 m) then fine (1.0 m) voxels, sign-safe scaled
  Levenberg damping, accept/reject step control. → THEORY.md §The algorithm.
- **The ICP contrast** — a compact point-to-point ICP (02.06's full GPU treatment, cited; CPU-only
  here by documented scope) run from the identical initial poses for an apples-to-apples basin
  comparison. → THEORY.md §Where this sits in the real world.

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/ndt-scan-matching.sln`](build/ndt-scan-matching.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/ndt-scan-matching.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none** — CUDA runtime + C++17 standard library only.
No cuBLAS/cuFFT/Thrust — every kernel (voxel build, assembly) is hand-rolled per CLAUDE.md §5's
"no black boxes" rule.

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at.

## Data

Fully synthetic, by necessity, not just by repo default (CLAUDE.md §8): NDT needs a map, a scan,
and an EXACT ground-truth pose to be verifiable at all, and no real LiDAR recording ships an answer
key. `scripts/make_synthetic.py` builds an L-shaped corridor-into-a-room scene, simulates a 16-beam
(VLP-16-like) scan with range noise and a documented outlier fraction, and writes a 240-trial
(40 trials × 6 magnitude bins) perturbed-initial-pose cohort — all from a fixed seed (42),
reproducible bit-for-bit. No public dataset applies; `scripts/download_data.ps1`/`.sh` are honest,
permanent no-ops. Full provenance, checksums, and field documentation: [`data/README.md`](data/README.md).

## Expected output

Nine gated verification stages (`VOXEL_STATS_TWIN`, `JACOBIAN_CHECK`, `ASSEMBLY_TWIN`,
`TRAJECTORY_TWIN`, `SCORE_SANITY`, `CONVERGENCE`, `ACCURACY`, `BASIN_CONTRAST`,
`OUTLIER_ROBUSTNESS`), each printing `PASS`/`FAIL`, plus two honesty-only `[info]` reports
(`failure_diagnosis`/`bin0_corridor_axis_split` and `degenerate_axis`, never gated). The GPU-vs-CPU
checks (§9's Definition-of-Done gate) are `VOXEL_STATS_TWIN` (voxel mean/covariance, rel tol
1e-3/1e-2), `ASSEMBLY_TWIN` (score/gradient/Hessian, rel tol 5e-3), and `TRAJECTORY_TWIN` (one full
multi-resolution optimization, measured-then-margined). Measured on the reference machine (RTX 2080
SUPER, Release|x64): NDT multi-resolution converges 32/240 = 13.3% of the perturbation cohort
(floor gate ≥10%), converged poses average 51.3 mm / 1.02° from ground truth (worst 77.1 mm /
2.67°, gate <100 mm/<4°), and the multi-resolution schedule's basin (13.3%) dramatically beats the
same iteration budget spent at fine resolution alone (1.2%). The compact ICP contrast converges
24.2% of the same cohort (more, and more accurately, than NDT here — see "Overview" and
"Limitations & honesty" for the honest reading of that number) and the smallest-perturbation bin's
own 65% splits into 0% along the corridor's degenerate axis vs. 84% off it — see
[`demo/expected_output.txt`](demo/expected_output.txt)'s header comment and
`demo/out/gates_metrics.csv` for every gate's exact number.

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — the contract: point-cloud/SE(3)/voxel-grid layouts, the
   shared SE(3) retraction (01.17's, cited), the mixture-model `d1`/`d2` derivation, the 28-scalar
   reduction packing, and the sign-safe scaled-damping Cholesky solve — read this FIRST, it is the
   single most information-dense file in the project.
2. [`src/kernels.cu`](src/kernels.cu) — the four voxel-build kernels, then `ndt_assemble_kernel`
   (the project's central new GPU concept — the per-point chain-rule math, live).
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the independent CPU twins, PLUS the compact
   ICP contrast (`icp_point_to_point_cpu`) — read its file header for exactly what is/isn't shared
   with the GPU path and why.
4. [`src/main.cu`](src/main.cu) — the nine-stage orchestration: data loading, grid building,
   every gate, the multi-resolution Newton driver (`run_ndt_stage_gpu`/`run_ndt_multires_gpu`), and
   the four `demo/out/` artifact writers.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, `paths.h` (copied, not shared — §4 rule).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **Biber & Straßer (2003), "The Normal Distributions Transform: A New Approach to Laser Scan
  Matching"** — the original 2-D NDT paper; the mixture-model score this project implements.
- **Magnusson (2009 thesis), "The Three-Dimensional Normal-Distributions Transform"** — the 3-D
  extension and the `d1`/`d2` derivation this project's `ndt_compute_d1_d2` follows.
- **Autoware `ndt_scan_matcher`** — the production ROS 2 node this project's system context names
  directly; compare its multi-threaded voxel structure and initial-pose (NDT-omp/GNSS) handling
  against this project's teaching-scale dense grid.
- **PCL `NormalDistributionsTransform`** — the reference open-source CPU implementation; its
  `outlier_ratio_`/`resolution_` parameterization is the one `kAssumedOutlierRatio`/`kLeafFine` mirror.
- **02.06 (this repo) — ICP: point-to-point → point-to-plane → GICP** — the direct contrast this
  project measures against; read it for the full GPU brute-force correspondence-search treatment
  this project's compact CPU-only ICP deliberately does not re-teach.
- **01.17 (this repo) — camera-LiDAR/camera-camera extrinsic calibration** — the source of this
  project's SE(3) retraction, 28-scalar reduction, and jacobian-check discipline, all cited by name
  throughout `src/`.

## Exercises

1. **Plot the artifacts:** `demo/out/registration_topview.csv` (before/after scatter) and
   `demo/out/basin_curve.csv` (converged-% per magnitude bin, three methods) — reproduce this
   README's basin-width claim visually.
2. **Widen the map:** increase `kMapSizeX`/the corridor length in `scripts/make_synthetic.py` and
   re-measure the near-field-only condition ratio in the `degenerate_axis` `[info]` lines — does a
   longer corridor make the degeneracy worse, as THEORY.md's physics argument predicts?
3. **Try Welford's algorithm:** replace the two-pass voxel covariance build with a single-pass
   Welford accumulator (per-voxel, still atomically parallel — you will need a different atomic
   update than a plain sum). Measure the numerical difference against the two-pass CPU oracle.
4. **GPU-accelerate the ICP contrast:** port `icp_point_to_point_cpu`'s brute-force correspondence
   search to a GPU kernel (02.06 is the full worked example) and re-measure `demo/out/basin_curve.csv`'s
   ICP wall-clock time.
5. **Add a third resolution:** insert a very-coarse (4.0 m) stage before the current coarse/fine
   schedule and measure whether the basin widens further, or whether returns diminish.

## Limitations & honesty

- **CPU-only ICP contrast.** This project's GPU-teaching payload is the NDT voxel-build and
  assembly kernels; a second GPU brute-force correspondence-search kernel would duplicate 02.06's
  own didactic content rather than teach a new GPU-mapping lesson. `icp_point_to_point_cpu` is
  therefore plain, single-threaded C++ — correct, but not fast (the full 240-trial ICP sweep is
  measurably the slowest single step in the demo).
- **Roll/pitch/z are not perturbed** in the cohort (`scripts/make_synthetic.py`'s
  `build_cohort()`) — a deliberate scope reduction standing in for a real system's wheel/IMU-based
  level-and-height prior; only translation (x,y) and yaw are randomized.
- **Genuine local minima, not just an iteration budget.** During development this project's own
  convergence gate caught two real bugs (a damping-scale bug and a damping-symmetry bug, both
  documented in `kernels.cuh`'s `kLambdaInit`/`cholesky6_solve_flat` comments and THEORY.md
  "Numerical considerations") — but even after both fixes, most of the cohort's larger-magnitude
  bins (≥1.2 m / 20°) converge 0% of the time for BOTH NDT and ICP: a real, measured basin
  boundary, not a hidden defect. A finisher pass re-confirmed this from scratch with a MUCH larger
  budget (60 coarse + 80 fine = 140 Newton steps, up from 27): the specific trials that plateau do
  so at an UNCHANGED score to two decimal places, not a slowly-improving one — a genuine stationary
  point of the objective, not an iteration-starved optimizer (THEORY.md "Numerical considerations"
  has the full trace).
- **ICP wins this particular scene, and that is taught honestly, not hidden.** This project's
  compact, small, single-room-scale corridor map is exactly the regime where an accurate
  nearest-neighbor correspondence search (ICP) is *more* sample-efficient than NDT's voxel
  discretization: ICP converges more of the cohort (24.2% vs. NDT's 13.3%) and is more accurate
  among its converged trials (36.6 mm / 0.73° vs. NDT's 51.3 mm / 1.02°). A finisher pass tried
  every honest lever available to change this (the assumed outlier ratio, cohort/scan density,
  fine voxel size, coarse voxel size, the eigenvalue-floor ratio, the accept/reject damping
  schedule, a backtracking line search, and the iteration budget) and measured either no
  improvement or active regressions from every one of them — reported honestly in THEORY.md
  "Numerical considerations" as a genuine negative-result lesson, not smoothed over. NDT's real
  advantage — no correspondence SEARCH at all, `O(1)` per point vs. ICP's `O(n·m)` brute force —
  only pays off once the target cloud is large enough that `m` (this project's 724-point ICP
  target) dominates; a city-block map would flip this comparison, and Exercise 4 asks you to build
  the GPU ICP kernel needed to measure that crossover directly.
- **A real RNG bug this project's own outlier-robustness gate exposed and a finisher pass fixed.**
  `scripts/make_synthetic.py`'s outlier "wrong-depth" draw used to come from the SAME RNG stream as
  every inlier's range noise — consuming an extra draw only on the outlier branch silently
  desynchronized the "clean" and "with-outliers" paired scans' noise from the first outlier onward,
  even though the docstring claimed beam-for-beam alignment. The result: an early version of this
  project measured the WITH-outliers cohort converging MORE often than the outlier-free one (16.7%
  vs. 11.1%) — backwards, and a symptom of the confound, not of outliers actually helping. Fixed by
  giving the outlier depth draw its own independent stream; `generate_scan()`'s docstring in
  `scripts/make_synthetic.py` tells the full story.
- **A flat-floor, no-ceiling scene.** The synthetic building has no ceiling (an open-top
  simplification, `scripts/make_synthetic.py`'s scene comment) and every surface is a flat
  axis-aligned rectangle — real indoor scenes have curved/cluttered geometry that would change the
  voxel covariance statistics this project's regularization is tuned against.
- **Assumed vs. true outlier ratio.** `kAssumedOutlierRatio` (0.40, the `d1`/`d2` robustness knob)
  is deliberately NOT equal to the data's true injected outlier fraction (5%) — this mirrors how a
  real system is tuned (you do not know the true rate in advance) but means the mixture model is
  not "solving the exact generative problem," by design.
- **Sim-validated only (CLAUDE.md §1):** this project's output is a POSE, not a control command, so
  the direct hardware-motion risk is lower than a controller project — but any real localization
  stack built from this teaching core would need full sensor calibration, a real pre-built map, and
  integration testing before it could safely inform navigation; see PRACTICE.md §3 for the testing
  ladder this teaching core does not itself climb.
