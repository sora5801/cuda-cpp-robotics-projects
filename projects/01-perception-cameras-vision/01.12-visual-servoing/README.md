# 01.12 — Visual servoing: image-Jacobian control loop entirely on GPU

**Difficulty:** intermediate · **Domain:** 1. Perception — Cameras & Vision

> Catalog bullet (source of truth, verbatim): `Visual servoing: image-Jacobian control loop entirely on GPU`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> **This project's output is a camera-twist command that would command motion of a real robot.**
> It is **sim-validated only** — see [Limitations & honesty](#limitations--honesty).

## Overview

This project implements classic **Image-Based Visual Servoing (IBVS)**: an eye-in-hand camera watches
a 4-point fiducial-like target, and a closed feedback loop drives a camera-twist command that steers
the observed image features toward their goal positions — no explicit 3-D pose estimation, ever. That
is the whole idea of IBVS: control directly in image space.

The GPU angle is not the controller itself (IBVS is a small, cheap computation) — it is a **batched
convergence-basin study**: 4096 independent closed IBVS loops, each starting from a different camera
pose, run *fully in parallel* on the GPU (one CUDA thread per loop, each simulating its own closed
feedback loop for up to 400 control steps). The same batch is repeated for **three controller
variants** that differ only in what depth estimate feeds the image Jacobian — turning "does the
classic fixed-depth approximation still work?" from a claim into a measured percentage. A designed
cohort of near-180°-rotation starting poses reproduces, on purpose, the best-known IBVS failure mode:
the camera physically **retreating** instead of rotating.

Everything here is implemented and runs in the committed demo: the controller (3 variants), the
4096-loop batch across three designed cohorts, all three verification twins, all three gates, and all
five artifacts. Nothing is documented-only.

## What this computes & why the GPU helps

**The computation:** for each of K=4096 independent starting camera poses, simulate a closed IBVS
control loop — project 4 known 3-D points into the image, form the damped Gauss-Newton normal
equations of the image Jacobian, solve a 6×6 system for a camera twist, integrate the pose, repeat
until the feature error converges or the step budget (400) is spent.

**The pattern: a *rollout farm of closed loops*.** Project 08.01 (MPPI) teaches the canonical version
of this idiom for *open-loop* candidate futures scored by an external cost; here, the "rollout" **is**
a whole closed feedback loop — there is no scoring or blending step afterward, each thread's local
decisions are the final answer. The loops never interact (no shared memory, no atomics beyond the
tiny broadcast reads of the target geometry), so one GPU thread per loop is the natural mapping —
exactly the same reasoning 08.01 and 10.03 use for their thousands of independent simulations. See
[`THEORY.md`](THEORY.md) "The GPU mapping" for the register-pressure and warp-divergence story that
is genuinely new here (a closed loop's threads finish at *different* step counts, unlike an
open-loop rollout's fixed horizon).

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial
whole (see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** IBVS closes the loop across the **PERCEPTION → CONTROL** boundary directly — it
  sits at the very *end* of a manipulation perception pipeline and feeds straight into the arm's
  **CONTROL** layer, bypassing the usual STATE ESTIMATION / PLANNING boxes entirely (SYSTEM_DESIGN.md
  §1: "control directly in image space" is IBVS's whole point — there is no explicit 3-D pose estimate
  in the loop at all, unlike Position-Based Visual Servoing, named honestly in
  [Prior art](#prior-art--further-reading)).
- **Upstream inputs:** four 2-D point observations per frame — in production, the output of a
  feature/fiducial detector, named by ID in the catalog: **01.04** (feature pipeline: FAST/Harris +
  descriptors, for natural features) or, far more commonly for this exact use case, **01.06**
  (AprilTag/ArUco GPU detector-decoder — "fiducial-based servoing is the industrial norm" precisely
  because a printed marker gives 4 known, unambiguous, sub-pixel corners for free, sidestepping
  correspondence and 3-D reconstruction entirely). This project simulates those 4 points directly
  (§Data) so it can focus on the control law.
- **Downstream consumers:** a camera-twist command (the message-shaped `Twist` struct,
  SYSTEM_DESIGN.md §3.6: `linear[3]` m/s, `angular[3]` rad/s, camera/end-effector frame) that a real
  robot would hand to **joint-velocity control** — the repo's domain **08** control projects (e.g.
  08.01 MPPI, 08.06 LQR gain scheduling) tracking it as a Cartesian velocity target — via the **robot
  Jacobian** (project **09.01**'s batched forward kinematics is the foundation; the differential
  counterpart, project 09.02's batched analytic Jacobians, is what actually maps this twist to joint
  velocities: q̇ = J⁺ v_c). On a real arm the vendor's joint-space `JointState`-shaped controller
  closes that inner loop.
- **Rate/latency budget — stated with multirate honesty:** in production, IBVS runs at **camera
  rate, 30–60 Hz** (SYSTEM_DESIGN.md §1.1 "Camera → perception") — a new twist command every frame —
  while the arm's own trajectory-tracking control loop runs at **0.5–1 kHz** and interpolates between
  twist updates. This demo **collapses that multirate reality into one 100 Hz simulated loop**
  (`kDt` = 0.01 s in `kernels.cuh`) for teaching simplicity — stated honestly in
  [Limitations](#limitations--honesty), not hidden.
- **Reference robot(s):** the **6-DoF manipulator work cell** (SYSTEM_DESIGN.md §2.2: "VISION …
  object pose [01→] … JOINT CONTROL [08→]" — IBVS is one concrete way to close that arrow directly,
  used for high-precision terminal moves like peg-in-hole or connector insertion where a
  camera-relative correction beats an open-loop planned approach) and, as a less common but real
  extension, a **quadrotor** performing vision-based landing-pad alignment (SYSTEM_DESIGN.md §2.4's
  FLIGHT CONTROL block — not diagrammed there explicitly, but the same image-Jacobian idea applied to
  a camera looking down at a fiducial landing pad instead of forward at a workpiece).
- **In production:** a fiducial-based IBVS loop like this one is genuinely shipped for
  high-precision terminal alignment (peg-in-hole, connector/PCB insertion, drone landing); most
  general pick-and-place uses **Position-Based Visual Servoing (PBVS)** or a **hybrid 2.5-D** scheme
  instead, and increasingly a **learned** visual servo policy — all three named honestly in
  [Prior art](#prior-art--further-reading).
- **Owning team:** this work lives with **controls & manipulation** (SYSTEM_DESIGN.md §5.1), adjacent
  to the perception team that would own the upstream fiducial detector and the embedded/firmware team
  that owns the arm's inner joint-velocity loop.

## The algorithm in brief

- **Perspective projection & the image Jacobian (interaction matrix)** — the classical per-point
  2×6 block relating a camera twist to feature velocity; see [`THEORY.md`](THEORY.md#the-math).
- **Damped Gauss-Newton / Levenberg-Marquardt control law** — `v_c = -λ · L̂⁺ · e`, solved via the
  6×6 normal equations `(L̂ᵀL̂ + μI) x = L̂ᵀe` with a hand-rolled Cholesky solve (33.01's small-SPD-solve
  idiom); see [`THEORY.md`](THEORY.md#the-math).
- **SE(3) pose integration** — exact quaternion exponential for rotation, first-order Euler for
  translation; see [`THEORY.md`](THEORY.md#numerical-considerations).
- **Three controller variants over one batch** — true-depth, fixed-depth, and the desired-Jacobian
  (`L(s*,Z*)`) scheme; see [`THEORY.md`](THEORY.md#the-algorithm).
- **The retreat pathology** — why near-180°-rotation initial errors make pinv(L) drive the camera
  physically backward; derived geometrically in [`THEORY.md`](THEORY.md#the-problem--physics--engineering-first).

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/visual-servoing.sln`](build/visual-servoing.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/visual-servoing.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none** — the 6×6 damped solve is hand-rolled Cholesky
(the same teaching idiom as project 33.01), no cuBLAS/cuSOLVER dependency. Default CUDA toolkit
libraries + C++17 standard library only.

## Run the demo

One command, from this folder (builds first if needed, runs the batched study, checks GPU vs CPU,
writes artifacts):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at (five artifacts: two PPM images, three CSVs).

## Data

This project's "data" is a **batch size** (how many closed IBVS loops to simulate), not recordings —
the target geometry, goal pose, every initial camera pose, and the controller constants are all
generated **inside the demo executable** from documented compile-time constants and a fixed
xorshift32 seed (base seed 42), the same pattern project 08.01 uses for its exploration noise.
Details and provenance in [`data/README.md`](data/README.md).

## Expected output

**Verification (3 GPU-vs-CPU twins):** a single loop's full 400-step trajectory (early-step tolerance
2e-4, late-step tolerance 5e-3 — honest accumulation drift over 400 steps), the Jacobian/pseudoinverse
linear algebra at 16 sampled poses across all 3 variants (tolerances 1e-3 / 2e-2 / 2e-3 for
v/A/b — measured worst deviations 3.9e-5 / 6.8e-3 / 3.4e-4 on this machine; a Debug (unoptimized)
build shows **exact** 0.0 agreement, confirming the small Release-build gap is FMA-fusion rounding,
not a bug — see `THEORY.md` §numerics), and a 128-loop batch-statistics subset (100% converged-flag
agreement measured).

**Three independent gates** (control-theory/literature predictions, not cross-checked against either
twin — see `reference_cpu.cpp`'s twin-vs-shared ruling):

| Gate | What it checks | Measured on this machine |
|------|-----------------|---------------------------|
| `exponential_decay` | Small pure-translation errors under true-depth IBVS decay at rate ≈ λ | fitted 2.008 /s vs λ=2.000 /s (0.4% deviation) |
| `convergence_basin` | ≥90% true-depth / ≥85% fixed-depth convergence in the nominal region | 97.9% / 96.5% (desired-Jacobian 96.3%, [info]) |
| `retreat_pathology` | ≥80% of the near-180°-rotation cohort shows camera retreat | 100% detected |

The canonical lines live in [`demo/expected_output.txt`](demo/expected_output.txt); every measured
number above appears on an `[info]` line in the real run (unchecked — see the determinism note atop
`src/main.cu`), while the `GATE`/`VERIFY`/`RESULT` lines themselves carry no numbers and are
byte-stable across platforms.

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — the full contract: frame conventions, target/goal geometry,
   controller constants, the three cohorts, the three variants, every buffer layout. Read this first.
2. [`src/kernels.cu`](src/kernels.cu) — the GPU implementation: quaternion math, the smart
   "accumulate the normal equations, never materialize the dense interaction matrix" per-point loop
   (`ibvs_compute_step`), the hand-rolled 6×6 Cholesky solve, SE(3) integration, and the batch/
   single-step kernels. The heart of the project.
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the independent CPU oracle (see its header for
   the twin-vs-shared ruling this project follows), plus the shared setup: target/goal geometry and
   the initial-pose cohort generator.
4. [`src/main.cu`](src/main.cu) — orchestration: load the scenario, run the batch for 3 variants +
   the basin grid, run the 3 verification twins, compute the 3 gates + 2 info comparisons, write the
   5 artifacts.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, and `paths.h` (data/artifact resolution).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **ViSP** (Visual Servoing Platform, Inria) — the reference open-source IBVS/PBVS library; its
  `vpFeaturePoint`/`vpServo` classes are the production version of exactly what `kernels.cu` hand-rolls.
- **Chaumette & Hutchinson, "Visual Servo Control" Parts I & II (IEEE RAM, 2006/2007)** — the
  canonical tutorial this project's `THEORY.md` derives from; Part I §IV discusses the retreat
  pathology this project reproduces.
- **PBVS (Position-Based Visual Servoing)** — the alternative scheme that estimates full 3-D pose
  first, then controls in Cartesian space; immune to the image-space retreat pathology, but needs an
  accurate pose estimate and a calibrated model — the classic trade this project's gates make concrete.
- **Hybrid / 2.5-D visual servoing (Malis, Chaumette, Boudet 1999)** — decouples translation (image-
  space) from rotation (partial-pose), avoiding both IBVS's retreat pathology and PBVS's model
  dependency; the modern practical default.
- **Learned visual servoing** (e.g. end-to-end pose-to-twist policies, CNN-based feature Jacobian
  estimators) — an active research area replacing the hand-derived interaction matrix with a learned
  one; named honestly as a real alternative, not implemented here.
- **AprilTag / ArUco** (see project 01.06) — the fiducial systems that make the 4-point correspondence
  this project assumes trivial and unambiguous on a real robot.

## Exercises

1. **Memoize the desired-Jacobian variant.** `kVariantDesiredJacobian`'s interaction matrix `L(s*,Z*)`
   never changes during a loop, but `ibvs_compute_step` recomputes it every step anyway (see the
   comment in `kernels.cu`). Factor it — and its Cholesky factorization — once before the time loop
   and measure the speed-up.
2. **Measure the early-exit divergence cost.** `THEORY.md` §GPU-mapping predicts that a warp's wall
   time is bounded by its slowest thread. Add a per-warp step-count histogram (via `cudaEvent`s around
   sub-ranges of the grid, or Nsight Compute's warp-execution-efficiency metric) and check the
   prediction against a warp drawn entirely from the nominal cohort vs. one straddling a cohort
   boundary.
3. **On-device cuRAND.** The initial poses are generated on the host (like 08.01's noise). Move
   `generate_batch_init_poses_cpu`'s xorshift32 stream onto the device with cuRAND and compare
   wall-clock time for K=100,000+ loops.
4. **Add a fourth variant: mean-depth.** Implement `Z = (Z_min+Z_max)/2` over the 4 points each step
   (a middle ground between true-depth and fixed-depth) and add it to `batch_stats.csv`.
5. **Widen the nominal cohort and watch the basin gate degrade.** Raise `kNominalAngleMaxDeg` back
   toward 35° (its first, too-optimistic value during this project's own tuning — see the git history
   or just try it) and observe true-depth convergence fall from ~98% into the 50-60% range — a direct,
   measured illustration of a convergence *basin* actually having an edge.

## Limitations & honesty

- **Multirate collapsed to one loop.** A real IBVS system runs feature extraction at camera rate
  (30–60 Hz) and often a faster inner control loop; this demo simulates both at one 100 Hz rate
  (`kDt`=0.01 s) for teaching simplicity (§System context states this explicitly).
- **Ideal, noiseless feature observations.** The 4 target points are read back exactly (their true
  camera-frame projection, in FP32) — no detector noise, no false correspondence, no partial
  occlusion. A real fiducial detector (01.06) has sub-pixel but nonzero noise; THEORY.md's numerics
  section discusses how this project's damping term is the honest analog of the robustness margin a
  real system needs against that noise.
- **The world/goal frame convention is a simplification.** The world frame is defined to share the
  camera's optical-frame axis convention at the goal pose (kernels.cuh's frame-convention note) — a
  real system has a genuine, separately-calibrated extrinsic offset between "world" and "camera" that
  this demo sidesteps to keep the linear algebra the focus.
- **The nominal cohort's ranges were TUNED, not assumed.** The convergence-basin gate's floors reflect
  what this specific controller (λ=2.0/s, μ=0.05 damping, 400-step/4 s budget) actually achieves at
  the documented ranges — not a universal claim about IBVS. Exercise 5 shows the basin has a real edge
  just outside the documented region.
- **`[R&D]` scoping:** not applicable — this catalog bullet is `intermediate`, not `[R&D]`.
- **Safety caveat (CLAUDE.md §1, §8):** this project's output is literally a camera-twist command
  that would drive a real robot's motion. Everything here is **simulation only** — no camera, no arm,
  no hardware-in-the-loop. Running an IBVS loop derived from this code on physical hardware is the
  owner's decision and responsibility; see `PRACTICE.md` §3 for the safe hardware-testing ladder
  (simulation → HIL → bench jig / tethered / current-limited → free running) any real deployment
  must climb, with E-stop and workspace limits at every rung.
