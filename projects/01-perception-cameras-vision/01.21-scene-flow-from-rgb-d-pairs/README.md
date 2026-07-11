# 01.21 — Scene flow from RGB-D pairs

**Difficulty:** intermediate · **Domain:** 1. Perception — Cameras & Vision

> Catalog bullet (source of truth, verbatim): `Scene flow from RGB-D pairs`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

Two RGB-D frames come in — a color image plus a depth map, captured a fraction of a second apart
from a robot's own moving camera, with one object in the scene also moving on its own. This project
recovers, per pixel, **where every point of the scene went** in 3-D (scene flow), separates the
part of that motion caused by the *camera's own movement* (ego-motion) from the part caused by
*genuinely independent motion*, and uses the leftover ("residual") motion to segment the one moving
object — without ever being told which pixels belong to it. The pipeline is entirely classical
(no learned components): dense pyramidal Lucas-Kanade for 2-D flow, pinhole back-projection for the
3-D lift, a robust (IRLS) closed-form rigid-motion fit (Horn's quaternion method) for ego-motion,
and a physically-derived residual threshold plus morphological cleanup for segmentation. The demo
runs it on a synthetic RGB-D pair with *exactly known* camera motion and object motion, so every
stage's output can be graded against ground truth, not just eyeballed.

## What this computes & why the GPU helps

Every stage below is dominated by a **per-pixel MAP or STENCIL** (compute one output value from a
small, independent neighborhood of inputs) except one: fitting the dominant rigid motion is a
weighted-least-squares problem, which becomes a **REDUCTION** (collapse thousands of independent
per-point contributions into one small 4×4/16-scalar system via a block-level parallel tree, the
same "GPU partial-reduce, host finishes it" shape projects 02.06 and 01.17 use for their own normal
equations).

- Optical flow (Lucas-Kanade): per-pixel *map*, each pixel solves its own local 2×2 normal
  equations from a *stencil* window — no cross-pixel dependency within a pyramid level.
- 3-D lifting (back-projection): pure per-pixel *map* (depth → 3-D point, with a small stencil for
  the bilinear depth-consistency check).
- Ego-motion fit: per-point *map* (residual, weighted contribution) feeding a *block-level tree
  reduction* — the one place this project moves beyond thread-per-pixel.
- Residual segmentation + morphological cleanup: per-pixel *map*/*stencil*.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial
whole (see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** Perception — specifically the boundary between raw geometric perception and
  the world model / dynamics understanding a planner needs. Scene flow is where "what does the
  scene look like" (01.01 rectified RGB, 01.18/01.20 depth) becomes "what is the scene *doing*" —
  the input every downstream motion-aware component needs.
- **Upstream inputs:** two time-adjacent `Image`+depth pairs from an RGB-D sensor (named by source
  in READMEs 01.18 "Depth completion" and 01.20 for structured-light/ToF-style dense depth; a
  stereo depth source like 01.02 "Stereo SGM" would feed the same interface).
- **Downstream consumers:** dynamic-object handling in SLAM (05.xx mapping/localization pipelines —
  a moving object treated as a static landmark corrupts the map and the pose estimate; scene flow's
  moving-object mask is exactly the "reject these correspondences" signal those systems need), the
  autonomy stack's **Prediction** layer (SYSTEM_DESIGN.md §1 — a segmented mover's velocity is a
  prediction seed), and navigation-around-movers logic (23.xx costmap/planning stacks, which need to
  know "this obstacle is not just an obstacle, it is moving, in this direction").
- **Rate / latency budget:** a real RGB-D sensor streams at 15–30 Hz; scene flow between adjacent
  frames should run comfortably inside that budget (SYSTEM_DESIGN.md §1's perception-layer figures).
  Measured on this project's 128×96 demo (RTX 2080 SUPER): 2-D flow ≈ 1 ms GPU; the full pipeline
  (flow + lift + 8-round robust fit + segmentation, twice — once for the demo, once for the
  negative control) completes in well under the 33–66 ms a 15–30 Hz budget allows, even before any
  production-grade optimization (THEORY.md "The GPU mapping" names the un-implemented shared-memory
  tiling that would matter at full sensor resolution).
- **Reference robot(s):** an AMR sharing space with people (the moving-object mask is precisely
  the "is that a person/cart and where is it going" signal a warehouse robot needs) and a
  manipulator work cell with moving parts nearby (a second arm, a conveyor).
- **In production:** a shipping stack would likely replace the classical LK/Horn combination with a
  learned scene-flow network (RAFT-3D-class, THEORY.md "Where this sits in the real world") for
  robustness to textureless/reflective surfaces, but the DECOMPOSITION this project teaches — ego-
  motion vs. independent motion, robust rejection of the minority mover — is architecturally
  identical in every dynamic-SLAM system, learned or classical.
- **Owning team:** perception (with close coupling to SLAM/state-estimation, since ego-motion
  recovery here is literally visual odometry's own problem wearing different clothes).

## The algorithm in brief

- **2-D optical flow** — 2-level pyramidal Lucas-Kanade (Scharr gradients, 5×5 structure tensor,
  forward-additive iterative solve, coarse-to-fine propagation) — see [`THEORY.md`](THEORY.md#the-algorithm).
- **3-D lifting** — pinhole back-projection of both frames' depth, with bilinear sub-pixel depth
  sampling and a depth-consistency guard against fabricating correspondences across real depth
  edges — see [`THEORY.md`](THEORY.md#the-math).
- **Robust ego-motion fit** — iteratively reweighted least squares (Tukey biweight, MAD-based
  robust scale) wrapping Horn's (1987) closed-form quaternion solution via a shifted power
  iteration (SVD-free) — see [`THEORY.md`](THEORY.md#the-algorithm).
- **Residual segmentation** — threshold the post-ego-motion residual against a noise-derived
  bound, then a 3×3 morphological opening, then a connected-component size filter (iterative
  label propagation) that removes components no larger than the opening operator's own noise
  floor — see [`THEORY.md`](THEORY.md#numerical-considerations).
- **Object motion** — a robust (IRLS + Tukey biweight), fixed-rotation offset fit restricted to
  the segmented mask: the rotation is held at the already-accurate recovered ego-motion rotation
  (the object translates only) and only the translation offset is robustly estimated — see
  [`THEORY.md`](THEORY.md#numerical-considerations) for why the original free 6-DOF fit failed.

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/scene-flow-from-rgb-d-pairs.sln`](build/scene-flow-from-rgb-d-pairs.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/scene-flow-from-rgb-d-pairs.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: none — CUDA toolkit runtime + C++17 standard library only.
The rigid-motion solve (Horn's quaternion method) is hand-rolled (no cuBLAS/cuSOLVER), consistent
with this project's "no black boxes" brief for a small, textbook 4×4 eigenproblem.

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at.

## Data

100% synthetic (CLAUDE.md §8 default): a ray-cast RGB-D pair with *exactly known* camera ego-motion
and one independently-moving textured box, generated by `scripts/make_synthetic.py` (fixed seed 42,
xorshift32-only, byte-identical on any machine). Full provenance, field-by-field format, and SHA-256
checksums in [`data/README.md`](data/README.md).

## Expected output

The demo prints a `VERIFY:` line per pipeline stage (GPU kernel vs. independent CPU twin, on the
real loaded data — not synthetic toy inputs) and a `GATE:` line per evaluation criterion (measured
result vs. the scene's known ground truth). `RESULT: PASS` requires every VERIFY and every GATE to
pass. Canonical lines are in [`demo/expected_output.txt`](demo/expected_output.txt); numeric detail
(the actual measured EPE, IoU, rotation/translation error) prints on `[info]` lines, which vary
slightly run-to-run in the sense that they report exact measured values but are **deterministic**
(no RNG at runtime — a fixed seed drives only the offline data generator) and were captured from an
actual run on the reference machine (RTX 2080 SUPER).

**Honest headline numbers** (measured, see THEORY.md "How we verify correctness" for the full
table): 2-D flow median endpoint error ≈0.25 px (mean ≈2.4 px — see Limitations below for why);
recovered ego-motion rotation error ≈0.017°, translation error ≈1 mm (vs. a naive unweighted fit's
≈0.28°/36 mm — the robustness gate's whole point); moving-object segmentation IoU ≈0.20, precision
≈0.31, recall ≈0.38 (precision up from ≈0.29 after the connected-component size filter, IoU roughly
flat — see Limitations below for the honest reason); object-motion offset direction cos(angle)
≈0.95 (well-aligned) but magnitude only ≈0.4-0.5× the true offset (still not gated — Limitations).

## Code tour

A guided reading order through `src/`:

1. [`src/main.cu`](src/main.cu) — entry point: loads the RGB-D pair, runs `run_pipeline()` (the
   shared driver both the dynamic and negative-control runs use), then every VERIFY/GATE/ARTIFACT.
2. [`src/kernels.cuh`](src/kernels.cuh) — the single-sourced contract: image/camera constants, the
   depth-noise model, the 16-scalar covariance-reduction layout, and `build_rigid_from_covariance16`
   (Horn's solve) — read its file header in full before anything else; it explains the ground-truth
   frame convention that this project's own build got wrong once (a genuinely instructive bug).
3. [`src/kernels.cu`](src/kernels.cu) — the GPU kernels themselves, grouped by milestone (2-D flow →
   3-D lift → residual/reduction → segmentation → connected-component size filter).
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the independent CPU twins.
5. [`scripts/make_synthetic.py`](scripts/make_synthetic.py) — the ray-cast scene generator; its
   module docstring derives the ground-truth transform this whole project is graded against.
6. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, `paths.h`.

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **Horn, "Closed-form solution of absolute orientation using unit quaternions"** (1987) — the
  exact rigid-motion solve this project implements; read it for the derivation this project's
  `THEORY.md` summarizes.
- **RAFT-3D** (Teed & Deng, 2021) and its successors — the learned, dense scene-flow state of the
  art; study how it replaces this project's hand-derived stages with learned features and iterative
  refinement while keeping the same rigid-motion-field idea at its core.
- **KITTI Scene Flow benchmark** — the standard real-world evaluation for this exact task; its EPE/
  outlier-rate metrics are the same family this project's `flow_2d`/`scene_flow_3d` gates use.
- **PCL (Point Cloud Library)** — production implementations of ICP-family rigid registration this
  project's Horn fit is a close cousin of (see also 02.06 in this repo).
- **OpenCV's `calcOpticalFlowPyrLK`** — the production pyramidal LK implementation this project's
  Milestone 1 studies toward (01.03 in this repo goes deeper on LK specifically).
- **nvblox / dynamic-SLAM systems** (e.g., DynaSLAM, DS-SLAM) — how a real system uses a moving-
  object mask like this one's to keep dynamic objects out of the map.

## Exercises

1. Print the per-IRLS-round robust scale and weight histogram (main.cu's loop already computes
   both) — watch how quickly the moving object's weights collapse toward zero.
2. Replace the scalar-magnitude Tukey biweight (kernels.cuh's `tukey_biweight`) with the textbook
   per-axis formulation and compare the recovered ego-motion accuracy.
3. Implement a RANSAC-lite alternative to IRLS (sample a minimal point set, score inliers, repeat)
   and compare robustness/runtime against the shipped IRLS loop on the same data.
4. The object-motion fit (`main.cu` Milestone 5) now robustly (IRLS+Tukey) estimates only a
   TRANSLATION offset with the rotation fixed at `T_robust`'s — and its recovered DIRECTION is
   good (cos(angle) ≈0.95) while its MAGNITUDE still under-shoots truth by roughly half. Instrument
   `main.cu`'s Milestone-5 loop to print the per-point deviation histogram on the final IRLS round
   and investigate: is the shrinkage coming from a specific subset of masked pixels (e.g. ones near
   the object's own silhouette), and does excluding them by a *tighter* Tukey cutoff (`kTukeyC`)
   recover more of the true magnitude — or does it just throw away too many points?
5. Extend the depth-consistency guard (kernels.cuh's `kDepthEdgeGuardM`) to also reject pixels whose
   *forward-backward* flow is inconsistent (01.03's census `census_consistency_kernel` pattern) and
   measure whether it shrinks the disocclusion-boundary blob that survives `kMinComponentSizePx`
   (`moving_mask_postmorph.pgm` vs. `moving_mask.pgm` in `demo/out/` shows exactly where it sits) —
   this is this project's own honestly-documented next step for `object_segmentation`'s IoU.

## Limitations & honesty

- **No black-box learned components** — every stage is classical, by design (the catalog bullet and
  this repo's teaching brief). A real dynamic-SLAM stack increasingly uses learned scene flow
  (RAFT-3D-class) for robustness a hand-derived LK pyramid cannot match.
- **2-level pyramid** (not 01.03's 3) — sufficient at this project's demo scale (the largest
  frame-to-frame flow is ≈10.4 px, well inside a 2-level pyramid's capture range); a sensor with
  larger inter-frame motion (lower frame rate, faster robot) would need more levels.
- **Mean-EPE outliers are real and measured, not hidden.** A genuine minority of pixels (occlusion/
  disocclusion boundaries at real depth edges, where Lucas-Kanade's brightness-constancy assumption
  fundamentally breaks — the scene point visible in frame0 may simply not correspond to anything
  sensible in frame1) produce large flow errors even though the median pixel is tracked to a
  fraction of a pixel. This inflates the `flow_2d`/`scene_flow_3d` gates' MEAN-based bounds well
  above the median — THEORY.md "Numerical considerations" reports the measured histogram honestly
  and this README's gate bounds are set from that measurement, not an aspirational target. A
  forward-backward consistency check (Exercise 5) is the natural next step to catch these
  automatically rather than accept them.
- **The residual-segmentation threshold is calibrated to THIS run's measured spread**, not purely
  the theoretical depth-noise model (kernels.cuh's `kSegThresholdKSigma` comment and THEORY.md
  explain why the pure depth-noise prediction under-counts the real spread: 2-D flow position
  uncertainty contributes too, and the depth-only model does not include it) — an honest, documented
  simplification, not silently absorbed into an unexplained constant.
- **The connected-component size filter (`kMinComponentSizePx`) helps precision but not IoU, and
  that gap is itself the honest finding.** This project's build originally assumed the mask's false
  positives were mostly scattered single-pixel speckle a size floor would cleanly remove. Measuring
  it pixel-by-pixel instead found the dominant false-positive source is a spatially COHERENT
  disocclusion-boundary blob roughly the same size as the object's own largest surviving fragment,
  immediately adjacent to it (visible by comparing `moving_mask_postmorph.pgm` to `moving_mask.pgm`
  in `demo/out/`). A pixel-count size floor cannot discriminate a coherent wrong-shaped blob from a
  coherent right-shaped one — precision measurably improved (≈0.29→≈0.31) and recall dropped in
  proportion (≈0.41→≈0.38), leaving IoU roughly flat (≈0.20). THEORY.md "Numerical considerations"
  reports the full before/after table and the size-floor sweep that shows this is not a tuning
  miss — no single floor does much better on this scene. Exercise 5 is the natural next step.
- **`object_motion` is reported `[info]`-only, not yet gated — but for a DIFFERENT, now root-caused
  reason than before.** The ORIGINAL implementation fit a free 6-DOF Horn transform on the segmented
  mask and recovered an offset nearly OPPOSITE the true direction (cos(angle) ≈ -0.91). Diagnosing
  this (feeding the fit EXACT ground-truth flow/depth at truth-mask pixels reproduces the known
  offset bit-exactly, proving no frame/axis bug; feeding the pipeline's own estimated points through
  the SAME free fit — even restricted to the exact TRUTH mask — still recovered a badly wrong
  answer) showed the real cause was CONDITIONING, not a convention error: a full 6-DOF fit on a
  small (~300-point), spatially narrow (one box face at ~7-8 m range) point set is ill-conditioned,
  and a small rotation error there, multiplied by the ~7-8 m lever arm from the fit's centroid to
  the camera origin, produces a large translation error. The fix (now shipped) holds the rotation
  fixed at the already-accurate `T_robust` (physically justified — the object translates only) and
  robustly (IRLS+Tukey) estimates just the translation offset. Measured result: DIRECTION recovers
  well (cos(angle) ≈0.95, clears a 0.9 bar) but MAGNITUDE still under-shoots truth by roughly half
  (≈0.4-0.5×, short of the 0.75-1.25× a gate would require) — consistent with a mixed-pixel/partial-
  volume bias at the object's boundary-dense footprint (bilinear depth sampling blending object and
  background depth near the silhouette systematically shrinks the recovered motion). Still `[info]`,
  honestly, with the real quality (not a blanket "solve ok") printed every run.
- **Rotation-free object motion.** The moving object translates only (no rotation) — a deliberate
  simplification (see `kernels.cuh`'s ground-truth derivation) that keeps the "recovered vs. truth"
  comparison a single unambiguous 3-vector rather than a second rotation-estimation sub-problem.
- **Sim-validated only.** Every number in this project comes from a synthetic scene with known
  ground truth; nothing here has been run on real sensor data or a physical robot, and none of it
  is safety-certified. If any output of a project like this ever fed a real robot's motion planning
  (which it plausibly would in production), it would need extensive real-world validation first.
