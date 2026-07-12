# 02.20 — LiDAR intensity calibration across channels

**Difficulty:** intermediate · **Domain:** 2. Perception — LiDAR & Point Clouds

> Catalog bullet (source of truth, verbatim): `LiDAR intensity calibration across channels`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

A spinning LiDAR's 16 beams are not 16 identical sensors — each one has its own laser diode and its
own avalanche photodiode, and those parts age, drift, and get assembled with slightly different
optical alignment. The result: the SAME material, at the SAME range, read by two different beams,
reports two different intensities. Nothing in the point cloud says so — every beam's return looks
equally trustworthy — so anything downstream that trusts intensity (a low-intensity weather filter,
an intensity-thresholded lane marking, a place-recognition descriptor) silently inherits a per-channel
bias. This project recovers all 16 beams' relative gains from **one ordinary scan, with no
reflectance targets**: it finds small patches of world space that multiple channels happened to
observe, cancels out the one thing every channel already agrees on (how range and incidence angle
affect return strength), and solves a small least-squares system over what is left — a channel-vs-
channel disagreement that can only be gain. The whole pipeline — self-calibration, an explicit
observability check for a channel nothing else ever saw, and a bonus non-parametric recovery of the
range-falloff curve itself — is implemented, GPU-accelerated, and measured against a synthetic scene
built for exactly this purpose. All components named in the catalog bullet are implemented; the
"recover the range curve too" extension is a documented bonus milestone, reported but not gated (see
"Expected output").

## What this computes & why the GPU helps

The computation has three GPU stages, each a different classic pattern: (1) a per-point **MAP** —
invert the forward model (divide out range falloff and incidence-angle cosine) and compute a voxel
key, independently for every point; (2) a **SCATTER-REDUCE** — every point atomically adds its
range/incidence-corrected log-intensity into a small per-(voxel, channel) accumulator, the same
histogram-style pattern project 01.09 uses for its radial bins, here keyed two-dimensionally; (3) a
second SCATTER-REDUCE, one thread per voxel this time, that turns each qualifying voxel's per-channel
statistics into its contribution to a 16×16 least-squares system. All three stages parallelize
perfectly at this project's scale (roughly one thousand points, a few hundred candidate voxels) and
finish in well under a millisecond of GPU time (measured below); the final 16×16 solve is deliberately
NOT parallelized — a problem that size has no meaningful GPU mapping (see THEORY.md "The GPU mapping"
and project 33.01, cited throughout, for where GPU-side batched small solves actually pay off).

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial
whole (see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** perception (`SYSTEM_DESIGN.md` §1) — a **calibration** step that runs on the raw
  point cloud alongside (not instead of) other point-cloud pre-processing (02.01's voxel downsampling,
  02.13's dynamic-point removal, 02.18's weather filtering): it does not change *where* a point is, only
  what its intensity means.
- **Upstream inputs:** a raw `PointCloud` (`SYSTEM_DESIGN.md` §3.6: `xyz` + `intensity`, per-point
  channel id from the driver) straight off the LiDAR, ideally accumulated over one or more full
  sweeps so enough channel-overlap voxels exist to solve.
- **Downstream consumers:** everything that reads intensity and assumes it means the same thing on
  every beam. Named explicitly, because this project's whole reason to exist is closing a dependency
  another project in this repo already measured: **project 02.18 (weather filtering)** cites this
  project by name as its forward-looking sibling, and its `intensity_dependence` gate quantifies the
  cost of *not* having it — LIOR's (Low-Intensity Outlier Removal) recall on the snow scan drops from
  91.1% (clean, as-calibrated intensity) to 85.2% under a documented per-channel miscalibration
  perturbation, a **-5.9 percentage-point** hit from exactly the failure mode this project fixes. This
  project's own `consistency_improvement` gate closes that loop with a compact LIOR-style
  decision-flip demo on this project's own data (see "Expected output"). Beyond 02.18: intensity-based
  lane/marking detection (a road-marking's intensity contrast must not depend on which beam saw it),
  place-recognition descriptors that bin intensity (a per-channel bias corrupts the bins), and
  **project 02.16 (multi-LiDAR merging)** by name — when two LiDARs' point clouds are fused, their
  intensities must ALSO agree with each other, which starts with each sensor's own channels agreeing
  with themselves first.
- **Rate/latency budget:** unlike most perception projects in this repo, this is not a per-scan,
  per-frame computation. A real deployment runs it at **factory calibration time** and then
  **periodically** (weeks to months) to track detector aging drift (`SYSTEM_DESIGN.md` §1.1's
  perception-boundary rates do not apply here — see PRACTICE.md §3 for the actual cadence). The output
  (16 gain scalars) is small enough to apply per-point at full sensor rate with negligible cost once
  computed.
- **Reference robot(s):** the autonomous-vehicle stack (`SYSTEM_DESIGN.md` §2.5 — intensity-dependent
  perception, e.g. lane markings and retroreflective signage, is safety-relevant) and the warehouse/
  outdoor AMR (`SYSTEM_DESIGN.md` §2.1 — any fleet running weather filtering or place recognition off
  raw intensity).
- **In production:** LiDAR vendors (Velodyne/Ouster/Hesai and others) ship factory-measured per-channel
  intensity calibration tables in firmware; this project's self-calibration is the *field* technique a
  fleet operator or integrator uses to verify or refresh those tables without shipping the unit back
  to a lab (PRACTICE.md §3).
- **Owning team:** **perception/calibration** (`SYSTEM_DESIGN.md` §5.1) — often a small team adjacent
  to the main perception group, responsible for every sensor's calibration lifecycle (intensity gain
  here; extrinsics in project 02.16; camera photometric calibration in project 01.09, this project's
  camera-side twin).
- **Domain closer:** this is the **last project in Domain 2 (Perception — LiDAR & Point Clouds)**.
  Read end to end, the domain's 20 projects trace one arc: raw points arrive (02.01 downsamples them,
  02.02 crops them, 02.03 finds the ground); they get filtered clean (02.13 removes dynamic points,
  02.14 segments moving objects, 02.18 removes weather); they get matched, registered, and mapped
  (02.05–02.07, 02.09–02.11); they get compressed for the fleet uplink (02.15), fused across sensors
  and with cameras (02.16, 02.17), voxelized for a neural network (02.19) — and now, last, the raw
  *signal* those points carry (not just their positions) gets trusted, because every beam that produced
  them finally agrees with itself. A raw scan becomes a calibrated, ML-ready, all-weather perception
  input.

## The algorithm in brief

- **Forward-model inversion** — dividing a raw intensity by the known range-falloff and Lambertian
  incidence-angle terms to isolate `gain × reflectivity` (THEORY.md "The math").
- **Voxel binning as observability currency** — grouping points into small, (usually) single-material
  patches of world space, keeping only the ones multiple channels touched (THEORY.md "The algorithm").
- **Per-channel log-gain least squares over shared voxels** — a centering-projector normal-equations
  assembly that eliminates the unknown per-voxel reflectivity analytically, reducing to a pure
  16-channel problem (THEORY.md "The math").
- **Gauge fixing and observability as graph connectivity** — only *relative* gains are recoverable; a
  channel disconnected from the dominant observation graph is flagged unobservable, never guessed
  (THEORY.md "The math", "Numerical considerations").
- **Bonus: nonparametric range-falloff recovery** — the recovered gains let the range-falloff curve
  itself be estimated from data and compared to the generator's true curve (THEORY.md "Where this sits
  in the real world").

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/lidar-intensity-calibration-across-channels.sln`](build/lidar-intensity-calibration-across-channels.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/lidar-intensity-calibration-across-channels.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: none — only the CUDA runtime + C++17 standard library
(CLAUDE.md §5's default budget). The 16×16 gain solve is a plain host-only Gaussian elimination
(kernels.cu SECTION 8); no cuBLAS/cuSOLVER/Thrust is linked.

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at.

## Data

Fully synthetic (CLAUDE.md §8 default): a 16-beam LiDAR ray-casts a structured scene (ground plane,
near wall, a small brighter test panel beside it, and a far wall — four different reflectivities) with
16 hidden, documented per-channel TRUE gains (spanning 0.60–1.40) baked into a physically-derived
forward model. A second scan retargets one channel to an isolated target no other channel ever sees,
for the observability gate. Generated by [`scripts/make_synthetic.py`](scripts/make_synthetic.py)
(fixed seed 42, xorshift32 — no public dataset used; no LiDAR vendor publishes the true per-channel
gain it used to render a scan, which is exactly the ground truth this project's gates need — see
`scripts/download_data.ps1`/`.sh` for the full reasoning). Full field documentation, checksums, and
provenance: [`data/README.md`](data/README.md).

## Expected output

**From an actual run on an RTX 2080 SUPER, CUDA 13.3, Release|x64 — every number below is measured,
never invented:**

- **VERIFY** (GPU vs CPU, primary scan, n=1,032 points, four pipeline stages): per-point corrected
  log-intensity max `|gpu-cpu|` = `0.000000` (tol 1e-4); voxel index mismatches = `0/1032` (exact);
  per-(voxel,channel) accumulation max `|gpu-cpu|` = `0.000001` (tol 3e-3, atomic-order tolerance);
  16×16 least-squares assembly max `|gpu-cpu|` = `0.000003` (tol 3e-3); gain-correction max
  `|gpu-cpu|` = `0.000000` (tol 1e-4). All PASS.
- **Channel graph:** all 16/16 channels observable on the primary scan, via 147 shared (≥2-channel)
  voxels out of 39,576 candidate voxels in the scan's bounding box.
- **`gain_recovery`** (the headline): worst per-channel relative error, gauge-aligned against ground
  truth, over all 16 observable channels = **5.8%** (floor: ≤ 9%).

  | ch | recovered | true | ch | recovered | true |
  |----|-----------|------|----|-----------|------|
  | 00 | 0.627 | 0.620 | 08 | 1.494 | 1.400 |
  | 01 | 1.173 | 1.180 | 09 | 0.693 | 0.650 |
  | 02 | 0.845 | 0.850 | 10 | 1.208 | 1.120 |
  | 03 | 1.380 | 1.350 | 11 | 0.845 | 0.780 |
  | 04 | 0.703 | 0.700 | 12 | 1.012 | 0.950 |
  | 05 | 1.114 | 1.050 | 13 | 1.339 | 1.220 |
  | 06 | 1.370 | 1.280 | 14 | 0.654 | 0.600 |
  | 07 | 0.970 | 0.900 | 15 | 1.199 | 1.080 |

- **`consistency_improvement`** (the reason-to-exist gate): the average per-voxel cross-channel
  coefficient of variation over the 147 shared voxels collapses from **0.238** (before calibration) to
  **0.044** (after) — a **5.5x** reduction (floors: after ≤ 0.12, collapse ≥ 2.5x). The paired
  LIOR-style decision-flip demo (wall_far cohort, n=308 points, threshold = the after-calibration
  median): **130/308** points flip keep/reject classification once the true per-channel gain is
  divided back out — the demand signal 02.18 measured as a -5.9pp recall drop is closed by this
  project's own recovered gains.
- **`multi_material_robustness`** (the currency-works gate): comparing the all-materials solve
  (ground+wall_near+panel+wall_far, 147 voxels) against a solve restricted to a single, pure material
  (wall_near only, re-binned at a coarser leaf so it alone reconnects all 16 channels — see THEORY.md
  "Numerical considerations" for why), on the 16 channels both solves flag observable: worst relative
  error 5.8% (all-materials) vs. 1.4% (single-material) — delta **4.4 percentage points** (floor: ≤ 8
  points). Mixing four different reflectivities into the shared-voxel pool does not meaningfully
  degrade recovery.
- **`unobservable_channel`** (the observability-honesty gate): on the degenerate scan, channel 15 is
  retargeted to a target no other channel reaches; the solver flags it **unobservable** (asserted, not
  guessed) while correctly recovering the other **15/15** channels via their own 128 shared voxels.
- **`[info]` range_profile** (bonus milestone, not gated): nonparametric recovery of the range-falloff
  curve's shape, normalized to the nearest-to-plateau populated bin — max deviation from the true
  curve **0.20** over 6/12 populated range bins spanning 7–37 m.
- **`[info]` noise_floor** (bootstrap precision, not gated): mean per-channel recovered-gain standard
  deviation across 8 voxel-resampled re-solves = **0.019** — honesty about how precise this one scan's
  noise realization lets the estimate be.
- **Timing** (teaching artifact, not a benchmark): the 3-kernel GPU pipeline on the primary scan —
  **~0.4 ms** total GPU kernel time, single-shot, one machine.

Tolerances: every VERIFY comparison uses a tolerance justified in `src/main.cu`'s constant comments
(measured-then-margined, never set AT the measured value); THEORY.md "How we verify correctness"
explains each in full.

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — read this FIRST: the forward model, the scene geometry, the
   voxel grid, and the least-squares model are all documented once, here.
2. [`src/kernels.cu`](src/kernels.cu) — the three GPU kernels (`point_features_kernel`,
   `bin_accumulate_kernel`, `assemble_ls_kernel`, `apply_gain_kernel`) and SECTION 8's shared 16×16
   solve — the heart of the project.
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the independently-typed CPU twins; read side by
   side with `kernels.cu` to see exactly what stayed shared (formulas) and what was retyped
   (accumulation order and precision).
4. [`src/main.cu`](src/main.cu) — orchestration: load both scans, the VERIFY stage, the shared solve,
   4 gates, 6 artifacts.
5. [`scripts/make_synthetic.py`](scripts/make_synthetic.py) — the forward model that generates the
   committed sample; its module docstring is the didactic derivation of the whole scene.
6. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, `find_data_file`/`resolve_out_dir`.

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1):

- **Velodyne / Ouster / Hesai factory intensity-calibration documentation** — real LiDAR vendors ship
  measured per-channel intensity correction tables in firmware; this project's self-calibration is the
  *field-verification* technique an integrator uses without a lab rig (PRACTICE.md §1).
- **Project 01.09 (Photometric/vignetting calibration kernels)** — this project's camera-side twin:
  the SAME "decompose a measurement into a per-pixel/per-channel gain times an unknown scene quantity,
  solve via shared/overlapping observations, gauge-fix the unobservable global scale" pattern, applied
  to a camera's vignette + per-pixel gain instead of a LiDAR's per-channel gain. Its SECTION-5 "shared,
  host-only, no GPU mapping for a tiny solve" precedent and its twin-independence ruling are followed
  here directly.
- **Project 02.18 (Weather filtering)** — this project's forward-looking sibling from the other
  direction: 02.18's LIOR filter assumes calibrated intensity and measures the cost of not having it
  (see "System context" above); this project is what closes that dependency.
- **Project 33.01 (Batched small-matrix linear algebra)** — where GPU-side batched small solves earn
  their keep, at a scale (many independent small systems) this project's single 16×16 solve never
  reaches.
- **Heckman & Krotkov, "Practical Methods for Geometric and Photometric Correction" and general
  radiometric self-calibration literature** — the "recover a per-sensor-element gain from overlapping
  observations without a calibration target" idea this project's algorithm instantiates for LiDAR.
- **SLAM/place-recognition literature using LiDAR intensity** (e.g. intensity-augmented scan context,
  project 02.11's sibling) — a direct downstream consumer of calibrated intensity.

## Exercises

1. Lower `kVoxelLeafM` (`kernels.cuh` SECTION 4) toward 0.3 m and re-run — watch the shared-voxel
   count and the channel graph's connectivity both drop (THEORY.md "Numerical considerations" explains
   why a fixed-origin grid is more fragile than it looks); find the smallest leaf that still connects
   all 16 channels on the primary scan.
2. Implement the profile-elimination least squares as a literal joint solve over (16 log-gains + one
   log-reflectivity per shared voxel) instead of the analytic voxel-mean elimination this project
   uses, and confirm the two give identical answers (THEORY.md "The math" derives why they must).
3. Add a THIRD, EVEN MORE isolated degenerate scenario — two channels that only ever see each other,
   never any of the other 14 — and confirm the solver reports them as their OWN small connected
   component, correctly excluded from the dominant one (kernels.cu SECTION 8).
4. Plot `demo/out/range_profile.csv` — where does the nonparametric recovery deviate most from the
   true curve, and why (hint: which range bins have the fewest, dimmest points)?
5. Extend the forward model with a per-channel BLACK-LEVEL offset in addition to gain (`I = g*R*f(r)*
   cos(theta) + o[ch] + noise`) and extend the least-squares solve to recover both — you will need a
   second gauge-fixing convention for the offset (THEORY.md "The math" hints at why).

## Limitations & honesty

- **Analytic scene geometry, not a general plane fit.** The calibration algorithm's incidence-angle
  normal comes from `kernels.cuh`'s known, axis-aligned scene constants, not a per-voxel plane fit
  (project 02.03's ground segmentation / 02.09's normal estimation are the general-purpose techniques
  a deployment against an ARBITRARY scene would need). This is a documented, deliberate teaching
  simplification — legitimate for a system calibrating against a known target environment, not a
  drop-in replacement for scene-agnostic self-calibration.
- **Dense grid, not a spatial hash.** This project's voxel counts (tens of thousands of candidate
  cells, at most a few hundred occupied) stay small enough for a plain dense array; project 02.01's
  spatial hash table is the right tool at real LiDAR-scan point counts (10⁵–10⁶ points/scan), which
  this project's synthetic scans (~1,000 points) are far below.
- **Single static scan, sensor at the origin.** Like project 02.18's generator, this project compares
  one static capture, not a continuously moving platform accumulating overlap over time (the more
  realistic real-world case, where MOTION — not scene layout — is what creates most channel overlap).
- **A fixed-origin voxel grid is more fragile than it looks.** This project's build measured a real
  failure mode directly (see THEORY.md "Numerical considerations"): a beam fan symmetric about
  elevation 0 puts a permanent graph cut at a plain-floor grid boundary, healed here by a half-leaf
  grid offset — a lesson, not a universal fix.
- **Illustrative gain magnitudes, dated 2026-07-12.** `TRUE_GAINS` in `scripts/make_synthetic.py`
  spans 0.60–1.40 as an illustrative "detector aging/alignment variance" range, not a measured
  hardware specification.
- **Not safety-certified.** Nothing here commands real hardware. If a production version of this
  calibration ever fed a costmap or a safety-relevant intensity-based detector on a real robot, the
  usual sim-validated-only caveat applies in full (CLAUDE.md §1).
