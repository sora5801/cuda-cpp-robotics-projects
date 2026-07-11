# Push note — 2026-07-11-02: batch 2d — the industrial-vision quartet

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Batch 2d (**51/505**; domain 01 at 15/24) is the factory-floor quartet: Canny+Hough alignment
(01.13), NCC pick verification (01.14), workspace background subtraction (01.15), and
ChArUco calibration-rig detection (01.16). Two themes define the batch. First, **integer
determinism as a design choice**: 01.13 routes Hough vote addressing through a Q16 fixed-point
trig table so its 144,180-cell accumulator is *bit-exact* GPU-vs-CPU, and 01.14's int64 sum
tables make all three of its NCC variants agree at exactly 0.0 while sidestepping the classic
variance-identity cancellation (whose uint32 overflow analysis turned out to be real on its own
tray — 4.63e9). Second, **process worked as designed**: 01.16 is the first Phase-2 project the
lead held at the gate — the original build's checkerboard-first ordering left the project's
defining lessons underpowered (3/8 views), so a focused finisher inverted it to production-style
**marker-first** anchoring, after which every previously-failing category orders exactly (large
tilt, 180° rotation, occlusion 29/29) and the ambiguity lesson demonstrates for real. Also in
the batch: an analytic EMA absorption-time gate landing within one frame of its derivation
(01.15), and a disclosed-and-remediated stray-file incident (01.15's path-truncation bug —
caught by the builder, re-verified clean by the lead).

## What changed

- **[projects/01-perception-cameras-vision/01.13-canny-hough-line-circle-detection-for-industrial/](../projects/01-perception-cameras-vision/01.13-canny-hough-line-circle-detection-for-industrial/)** —
  Canny (with CCL-style hysteresis reaching the same fixed point as an independent CPU flood
  fill) → bit-exact Hough lines + known-radius circles → rigid part alignment; 6 gates.
- **[projects/01-perception-cameras-vision/01.14-template-matching-at-scale-for-pick-verification/](../projects/01-perception-cameras-vision/01.14-template-matching-at-scale-for-pick-verification/)** —
  NCC three ways (naive / integral-table / shared-memory), 24-slot tray with six designed
  cohorts, SSD-vs-NCC shadow lesson, rotation-brittleness lesson; 5 gates.
- **[projects/01-perception-cameras-vision/01.15-background-subtraction-for-fixed-workspace-cells/](../projects/01-perception-cameras-vision/01.15-background-subtraction-for-fixed-workspace-cells/)** —
  frame-diff vs single-Gaussian vs MOG-lite on a 160-frame designed sequence (committed in
  full at 128×96 — size math documented); 5 gates; NOT a safety device (§8, 21.04 cited).
- **[projects/01-perception-cameras-vision/01.16-checkerboard-charuco-detection-acceleration/](../projects/01-perception-cameras-vision/01.16-checkerboard-charuco-detection-acceleration/)** —
  saddle-point corners + gradient-orthogonality sub-pixel refinement + marker-first grid
  anchoring + Zhang-lite mini-calibration; 6 gates; built by worker + finisher (see below).
- **[docs/STATUS.md](../docs/STATUS.md)** — 01.13–01.16 → `done` (**51/505**).

## New projects (didactic blurbs)

**01.13 — Canny + Hough.** Point-line duality derived, then made *deterministic*: integer
atomics + fixed-point trig = a bit-exact accumulator (why float atomics can't promise that).
The factory payoff: part alignment recovered to (7.99, −4.90, 6.72°) vs applied (8, −5, 7°),
and the hysteresis lesson quantified — the engineered weak scratch survives double-thresholding
(100%) and dies under single (20%).

**01.14 — NCC pick verification.** Why correlation beats SSD under real lighting (the shadowed
slot: NCC 0.984 vs SSD wrong by 62×), integral-image algebra with an overflow analysis that
bites for real, and rotation brittleness measured into a straddle (0.612 < threshold < 0.691).
24/24 slots classified across six designed cohorts.

**01.15 — Background subtraction.** A fixed camera makes "background" a statistical object:
EMA time constants derived (absorption predicted 19 frames, measured 18), frame differencing
killed by an illumination ramp exactly as theory says (FP 0.315 vs ≤0.017 adaptive), and the
blinking-lamp region splitting MOG from single-Gaussian at 0.0000 vs 1.0000.

**01.16 — ChArUco detection.** Why calibration targets carry markers: a plain checkerboard's
180° ambiguity is *shown* (the same corner labels (6,3) plain vs (0,0) marker-anchored), and
occlusion tolerance comes by construction once identity is absolute. Six marker-decode
subtleties (exact dictionary collisions, handedness traps, mirror-symmetric codes) are kept as
case studies, plus an honest calibration finding: adding weakly-informative views made
unweighted Zhang *worse* (15.5% vs 9.9% focal) — data quality beats data quantity.

## How to build & run

```powershell
projects\01-perception-cameras-vision\01.13-canny-hough-line-circle-detection-for-industrial\demo\run_demo.ps1
projects\01-perception-cameras-vision\01.14-template-matching-at-scale-for-pick-verification\demo\run_demo.ps1
projects\01-perception-cameras-vision\01.15-background-subtraction-for-fixed-workspace-cells\demo\run_demo.ps1
projects\01-perception-cameras-vision\01.16-checkerboard-charuco-detection-acceleration\demo\run_demo.ps1
```

## What to study here

The quartet is one working cell: 01.16 calibrates the camera, 01.15 watches the workspace,
01.13 measures the fixture, 01.14 verifies the pick. Study the two determinism designs (01.13's
fixed-point voting, 01.14's integer tables) side by side — they are the same idea applied to
scatter and to gather. Then read 01.16's THEORY for the held-and-reworked story: what
checkerboard-first ordering missed and why marker-first is robust *by construction*. Exercise:
run 01.14's NCC on 01.15's illumination-drift frames and predict which gates survive.

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-11), each project re-verified independently by the lead after the builder's
self-gate — all four: `Release|x64` **and** `Debug|x64` clean rebuilds, zero errors zero
warnings; demos exit 0 with all stable lines matched (15/15, 13/13, 18/18, 12/12);
`tools/verify_project.py` all structural gates PASS.

- **01.13:** hysteresis fixed point exact vs an independent flood fill; line accumulator
  BIT-EXACT (144,180 cells); alignment 7.99/−4.90/6.72° vs 8/−5/7°; edge P/R 0.949/0.991;
  negative control zero detections.
- **01.14:** integral images + window stats bit-exact; three variants pairwise 0.0; 24/24
  classification; localization 0.00 px; naive→shared 1.8×.
- **01.15:** 0/1,966,080 mask mismatches ×3 models; absorption 18 vs predicted 19; drift FP
  0.3151 (frame-diff, designed fail) vs ≤0.0172; bimodal 0.0000 vs 1.0000. Builder's
  stray-file incident disclosed, remediated, re-verified clean by the lead.
- **01.16:** worker+finisher; lead held the first delivery (core lessons underpowered),
  finisher's marker-first rework re-verified: 6/8 views exact including every previously-
  failing category; twins tight (refine 3.8e-5 px, decode 0/192); mean corner 0.71 px after
  2.47× refinement; Zhang honestly analyzed (see blurb); negative control zero corners.

## Known limitations / TODOs

- 01.13: 4-sector (not interpolated) NMS, known-radius circles (both stated). 01.14: NCC only
  (geometric matching named as production for rotation/scale). 01.15: didactic monitoring, not
  a certified safety function (§8). 01.16: two views remain honestly unordered (upstream
  detector sparsity; noise-marginal quad decode) and Zhang is the unweighted linear core —
  weighted/nonlinear refinement documented as the extension.

## Next push preview

Batch 2e: 01.17 camera-LiDAR extrinsic calibration, 01.18 depth completion, 01.19
structured-light decoding, 01.20 time-of-flight processing — the 3-D sensing quartet.
