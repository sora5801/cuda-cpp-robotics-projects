# 02.10 — FPFH descriptors + RANSAC global registration

**Difficulty:** intermediate · **Domain:** 2. Perception — LiDAR & Point Clouds

> Catalog bullet (source of truth, verbatim): `FPFH descriptors + RANSAC global registration`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

This project aligns two LiDAR scans of the same scene **with no initial guess** — the "kidnapped
robot" or "lost robot" problem for point clouds. Unlike [ICP](../02.06-icp-point-to-point-point-to-plane-gicp/README.md),
which needs to already be *close* to the right answer, this pipeline finds the transform from
scratch: it computes a **pose-invariant local-geometry descriptor** (FPFH — Fast Point Feature
Histograms) at every point, matches descriptors between the two scans, runs **RANSAC over
correspondence triplets** to find a rigid transform that a large fraction of matches agree with, and
hands the result to a few point-to-plane ICP iterations for a final polish. The committed demo scans
are related by a deliberately extreme **140-degree yaw + 8-meter translation** — far outside what any
local method could recover — so the demo can *measure*, not merely assert, that global registration
succeeds where ICP alone fails.

Every stage in the catalog bullet is implemented as real, GPU-accelerated code: FPFH descriptor
computation (normals → SPFH → FPFH), brute-force descriptor matching with a ratio test, a GPU RANSAC
hypothesis farm over correspondence triplets with an edge-length prescreen, a closed-form (Horn 1987)
rigid-transform solver, and a compact point-to-plane ICP handoff. Nothing here is documented-only.

## What this computes & why the GPU helps

The pipeline is dominated by three **embarrassingly parallel, independent-per-point (or
independent-per-hypothesis)** computations, each mapped to one GPU thread per problem instance —
the *map* pattern throughout, with one *batched-solve* flavor in RANSAC:

- **Descriptors (map):** every point's FPFH computation depends only on its own `k`-nearest
  neighbors — one thread per point, no cross-thread synchronization, for KNN, normals, SPFH, and FPFH
  alike (four kernels, each embarrassingly parallel).
- **Matching (map):** each source point's nearest-target-descriptor search is independent of every
  other source point — one thread per source point, scanning all targets.
- **RANSAC (batched-solve / sampling):** each of 8,192 hypotheses samples its own 3 correspondences,
  fits its own rigid transform (a tiny closed-form eigenproblem), and scores itself against the whole
  correspondence set — independently of every other hypothesis. One thread per hypothesis turns a
  sequential "try K random samples" loop into a single parallel launch.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** Perception / SLAM & mapping — specifically the **relocalization and map-merging**
  sub-problem: recovering a sensor pose in a known map (or aligning two independently-built maps) with
  **no prior pose estimate**, as opposed to frame-to-frame tracking (which local ICP/NDT handle once a
  good prior exists).
- **Upstream inputs:** two `PointCloud` messages (this repo's flattened `sensor_msgs/PointCloud2`
  shape, `docs/SYSTEM_DESIGN.md` §3) — typically a live scan and either a stored map submap or another
  robot's/session's submap. Point normals are computed internally (project [02.09](../02.09-normal-curvature-estimation-at-millions/README.md)'s
  lineage, reimplemented compactly here — see "The algorithm in brief").
- **Downstream consumers:** the recovered `Twist`/pose feeds (a) a **loop-closure geometric
  verification** stage (project [02.11](../02.11-scan-context-ring-descriptor-loop-closure-search/README.md),
  which first proposes *candidate* loop-closure pairs by place descriptor, this project's natural
  partner: 02.11 answers "which pairs might match", 02.10 answers "what transform, if any, actually
  aligns them"); (b) a **pose-graph SLAM** back-end (multi-session mapping, `docs/SYSTEM_DESIGN.md`
  domain 05) as a new edge constraint; (c) local ICP/NDT (project [02.06](../02.06-icp-point-to-point-point-to-plane-gicp/README.md))
  as its required initial guess, closing the global-then-local loop this project's own pipeline already
  demonstrates internally.
- **Rate / latency budget:** this is an **event-driven**, not a per-frame, computation — it runs on
  *relocalization events* (startup, kidnapped-robot recovery, loop-closure candidates, multi-robot
  rendezvous), typically at most a few Hz and often far less (once per candidate event, sub-second to a
  few seconds acceptable — `docs/SYSTEM_DESIGN.md` §1's per-boundary latency table has no entry for
  this because it is not part of the steady-state 10-400 Hz perception/estimation loop; PRACTICE.md §3
  states this honestly). Running it every frame like ICP would be both wasteful and the wrong tool —
  see [Limitations & honesty](#limitations--honesty).
- **Reference robot(s):** the warehouse AMR (relocalization after a wheel-slip/e-stop event, or
  merging two mapping sessions) and the autonomous-vehicle stack (initial localization against an
  HD-map submap without GPS, or after a GPS dropout) — both cited in `docs/SYSTEM_DESIGN.md` §2.
- **In production:** would sit beside/inside the SLAM back-end as its "global localization" or
  "kidnapped robot recovery" service, typically triggered by a watchdog (pose-covariance blow-up,
  tracking loss) rather than run continuously.
- **Owning team:** localization/mapping (part of the broader perception/autonomy org,
  `docs/SYSTEM_DESIGN.md` §5), closely adjacent to the SLAM/pose-graph team that consumes its output as
  loop-closure/relocalization edges.

## The algorithm in brief

- **Normals** — brute-force KNN + mean-shifted covariance + Jacobi eigensolve, a compact
  reimplementation of [02.09](../02.09-normal-curvature-estimation-at-millions/README.md)'s lineage at
  this project's smaller point-count scale ([`THEORY.md` § The algorithm](THEORY.md#the-algorithm)).
- **SPFH (Simplified Point Feature Histogram)** — a per-point Darboux-frame angle triplet
  (alpha, phi, theta) against every k-nearest neighbor, histogrammed into 3×11 bins — the pose-invariant
  core of the whole method ([`THEORY.md` § The problem](THEORY.md#the-problem--physics--engineering-first)
  derives *why* it is pose-invariant).
- **FPFH ("Fast" PFH)** — a weighted re-accumulation of each point's own SPFH with its neighbors'
  already-computed SPFH values, achieving PFH's descriptive power in O(nk) instead of O(nk²)
  ([`THEORY.md` § The algorithm](THEORY.md#the-algorithm)).
- **Descriptor matching + ratio test** — nearest/second-nearest FPFH distance, 01.04's ambiguity-check
  lesson arriving in 3-D ([`THEORY.md` § The problem](THEORY.md#the-problem--physics--engineering-first)).
- **RANSAC over correspondence triplets**, with an **edge-length prescreen** before every fit — 02.03's
  RANSAC discipline (iteration-count formula included), generalized from 3 points to 3 correspondence
  *pairs* ([`THEORY.md` § The math](THEORY.md#the-math)).
- **Horn's closed-form rigid fit** (1987) — a 4×4 symmetric eigenproblem whose largest eigenvalue's
  eigenvector is the optimal rotation quaternion ([`THEORY.md` § The math](THEORY.md#the-math)).
- **Point-to-plane ICP handoff** — a compact reimplementation of
  [02.06](../02.06-icp-point-to-point-point-to-plane-gicp/README.md)'s point-to-plane linearization,
  polishing RANSAC's coarse result ([`THEORY.md` § The algorithm](THEORY.md#the-algorithm)).

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/fpfh-descriptors-ransac-global-registration.sln`](build/fpfh-descriptors-ransac-global-registration.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/fpfh-descriptors-ransac-global-registration.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: none — CUDA toolkit runtime + C++17 standard library only.
(Earlier drafts of this project considered a voxel-hash neighbor index (02.01/02.09 lineage) or Thrust;
at this project's point counts, brute-force GPU KNN is faster to build, simpler to verify, and just as
fast in wall-clock terms — see `THEORY.md` "Where this sits in the real world" for the crossover point.)

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at.

## Data

Three synthetic `(source, target)` scan pairs of one room (floor + 4 walls + a box crate + a
cylindrical pillar), scanned from two sensor poses related by a **known** 140-degree yaw + 8-meter
translation. `pair0` is clean/62%-overlap, `pair1` is noisy (1 cm sigma)/62%-overlap (the demo's
headline pair), `pair2` is noisy/33.5%-overlap (a deliberately harder, honestly-reported stress
cohort). Every source/target `.bin` also carries a `world_idx` per point — the **ground-truth**
correspondence key used by the `descriptor_invariance` gate, never by the registration algorithm
itself. Fully documented (byte layout, checksums, regeneration command) in
[`data/README.md`](data/README.md); generated by [`scripts/make_synthetic.py`](scripts/make_synthetic.py)
(seed 42, xorshift32, Python stdlib only — CLAUDE.md §8/§12).

## Expected output

The demo prints a `VERIFY(...)` line (PASS/FAIL) for every GPU-vs-CPU cross-check (KNN, normals, SPFH,
FPFH, descriptor matching, the RANSAC hypothesis farm, the RANSAC refit, and the ICP point-to-plane
system — tolerances documented beside each check in [`src/main.cu`](src/main.cu) and derived in
[`THEORY.md`](THEORY.md#how-we-verify-correctness)), then a `GATE ...` line (PASS/FAIL) for every
**independent** ground-truth/invariant check: `descriptor_invariance` (the pose-invariance property
itself, measured), `registration_recovery` (recovered vs. true pose, the headline result),
`icp_negative_control` (proving local ICP alone fails at this relative pose), and `ransac_formula`
(02.03's analytic iteration-count check, re-derived for the 3-correspondence case). `[info]` lines
report every measured number (percentages, angular/translation errors, iteration counts) — deliberately
*not* diffed, so the checked contract survives running on a different GPU architecture whose
atomic-reduction order can shift a float in its last few bits (see `src/main.cu`'s "Output contract").
The canonical lines live in [`demo/expected_output.txt`](demo/expected_output.txt), captured from an
actual run on an RTX 2080 SUPER.

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — **read this first.** The full pipeline description (STAGES
   1–6), every data-layout contract, every shared formula (RNG, edge-length prescreen, Horn's rigid
   fit, the Darboux-triplet/binning formula), and the "Twin-vs-shared ruling" that explains exactly
   what is shared vs. independently reimplemented at each stage.
2. [`src/kernels.cu`](src/kernels.cu) — the GPU kernels themselves, one stage at a time: brute-force
   KNN, normals, SPFH (`compute_spfh_kernel` — the heart of the project, read its header for the
   Darboux-frame derivation pointer and the "no atomics" GPU-mapping argument), FPFH
   (`compute_fpfh_kernel` — the "two-ring" complexity story), descriptor matching, the RANSAC
   hypothesis farm (`ransac_hypotheses_kernel`), and the ICP handoff kernels (including
   `icp_accumulate_kernel`'s atomics — the direct GPU-mapping *contrast* with SPFH/FPFH's atomic-free
   design).
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the independent CPU oracle twins, and the
   file-header table of what is shared vs. independently retyped per stage.
4. [`src/main.cu`](src/main.cu) — orchestration: data loading, the descriptor/match/RANSAC/ICP pipeline
   run per pair, every `VERIFY`/`GATE` check, and the artifact writers.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, and `paths.h`'s multi-candidate data/output
   resolution.

## Prior art & further reading

- **PCL** (`pcl::FPFHEstimation`, `pcl::SampleConsensusPrerejective`) — the reference C++
  implementation this project teaches toward; study its polygonal-prerejection correspondence rejector
  (the "standard" version of this project's edge-length prescreen) and its OpenMP-parallel FPFH.
- **Open3D** (`open3d.pipelines.registration.registration_ransac_based_on_feature_matching`) — a
  modern, actively-maintained global-registration pipeline with the identical FPFH-match-then-RANSAC
  shape; its `CorrespondenceCheckerBasedOnEdgeLength` is this project's prescreen, named.
- **Rusu, Blodow & Beetz, "Fast Point Feature Histograms (FPFH) for 3D Registration", ICRA 2009** — the
  original paper; read it for the exact SPFH/FPFH formulas this project implements.
- **Horn, "Closed-form solution of absolute orientation using unit quaternions", JOSA A 4(4), 1987** —
  the rigid-fit-from-correspondences solver used for both the minimal 3-point hypothesis and the final
  inlier refit.
- **TEASER++** (Yang, Shi & Carlone, 2020) and learned descriptors (e.g. FCGF, D3Feat) — the modern
  research frontier: certifiably-robust registration under extreme outlier rates, and
  learning-based alternatives to hand-crafted FPFH — named honestly in [`THEORY.md`](THEORY.md#where-this-sits-in-the-real-world)
  as what this teaching version does not attempt.
- [02.03 ground-segmentation](../02.03-ground-segmentation/README.md) — the RANSAC discipline (the
  iteration-count formula, hypothesis-farm GPU pattern) this project generalizes from planes to
  correspondence triplets.
- [02.06 ICP](../02.06-icp-point-to-point-point-to-plane-gicp/README.md) — the point-to-plane
  linearization and `Rigid3`/`T_target_source` pose convention this project's STAGE 6 handoff reuses.

## Exercises

1. **Tighten the ratio test.** `kMatchRatioMax` is 0.95 (looser than SIFT's classic 0.7-0.8) — measure
   what happens to the correspondence-set size `nc` and the measured inlier ratio `w` (the
   `ransac_formula` gate's `[info]` line) as you tighten it toward 0.8. Does registration still succeed
   with fewer, cleaner correspondences and a smaller `kRansacK`?
2. **Add a mutual-nearest-neighbor filter** alongside the ratio test (STAGE 4's documented alternative,
   `kernels.cuh`'s file header) — does it change which correspondences survive on this scene's
   self-similar wall patches?
3. **Reduce `kRansacK`** until `GATE ransac_formula` fails, then explain from the printed `w` why —
   connect the measured inlier ratio to the classical iteration-count formula by hand.
4. **Build the block-shared-memory reduction** for `icp_accumulate_kernel` (currently atomics — a
   documented simplification, `kernels.cu`'s file header) mirroring 02.06's block-tree pattern, and
   measure the speed-up at this project's point counts.
5. **[R&D-adjacent]** Swap FPFH for a learned descriptor stub (even a random-but-consistent per-point
   embedding) and see how the RANSAC/prescreen/refit machinery downstream is completely unaffected —
   the exercise TEASER++'s and FCGF's papers make explicit: descriptor and registration are separable
   design choices.

## Limitations & honesty

- **Brute-force KNN and matching**, not a spatial index: the right choice at this project's ~1,500-3,200
  points/scan (see `THEORY.md` "Where this sits in the real world" for the crossover to a voxel hash or
  a KD-tree), explicitly not the millions-of-points throughput [02.09](../02.09-normal-curvature-estimation-at-millions/README.md)
  targets.
- **Visibility model is range + back-face, not ray-casting**: `scripts/make_synthetic.py`'s scene
  generator does not model one object occluding another (e.g. the crate shadowing a patch of wall
  behind it) — an honest, stated simplification (see `data/README.md`), not a hidden one. A true
  ray-caster is project [11.01](../../11-sensor-sim-digital-twins/11.01-gpu-lidar-simulator/README.md)'s job.
- **`icp_accumulate_kernel` uses atomics**, not 02.06's shared-memory block-tree reduction — a
  documented, justified simplification for this project's smaller point counts (`kernels.cuh`/`kernels.cu`
  file headers), not a silent shortcut.
- **`refinement_payoff` is reported, not gated** (the ratified scope explicitly allows either): on this
  scene, RANSAC's many-point inlier refit is already extremely accurate (correspondences are
  near-exact, same physical points with only i.i.d. sensor noise), so ICP's own marginal contribution
  is small and occasionally slightly negative (a nearest-point correspondence search landing on a
  different-but-close target point than the refit's exact correspondence) — an honest finding, not a
  bug; see `THEORY.md` for when real-world correspondence sets make ICP's payoff larger.
- **Not safety-certified.** Nothing here is validated for real robot motion; all data is synthetic and
  labeled as such (CLAUDE.md §1/§8). If this pipeline's recovered pose ever fed a real robot's motion
  planner, it would need extensive additional validation (RANSAC can, rarely, converge on a
  plausible-looking WRONG transform when the correspondence set is adversarial or degenerate — the
  `low_overlap` cohort's honest failure mode is a small-scale preview of this risk).
