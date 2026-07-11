# 01.07 — Fisheye/omnidirectional unwarping and multi-camera surround-view stitching

**Difficulty:** intermediate · **Domain:** 1. Perception — Cameras & Vision

> Catalog bullet (source of truth, verbatim): `Fisheye/omnidirectional unwarping and multi-camera surround-view stitching`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

This project has two connected halves, both built around ONE physical fisheye lens model
(equidistant, `r = f*theta`, `../src/kernels.cuh` PART 1). **Half 1** takes a single 185-degree-class
fisheye camera image and unwarps it two ways: onto a narrow rectilinear (pinhole) sub-FOV and onto a
wide cylindrical panorama — the classic "dewarp" problem every fisheye security/dashcam/mirror-
replacement product solves. **Half 2** mounts FOUR such fisheye cameras on a vehicle (front, left,
right, rear, each tilted down toward the ground) and stitches them into a single top-down
bird's-eye-view (BEV) image — the standard automotive/AMR "surround view" used for parking, docking,
and close-quarters maneuvering. The two halves share data: Half 1's single camera IS the rig's FRONT
camera, so the same rendered fisheye image demonstrates both a per-camera unwarp and a piece of the
multi-camera stitch. Both are implemented in full (not reduced-scope); the demo runs GPU kernels for
every stage, verifies each against an independent CPU oracle, and checks 7 physical correctness gates
— most notably a deliberate, honestly-measured failure of the BEV's flat-ground assumption near tall
objects (the reason production BEV systems exist as a research topic, not a solved problem).

## What this computes & why the GPU helps

Both halves are **inverse-mapping gather** operations — for every OUTPUT pixel, compute which INPUT
pixel(s) it should sample, then bilinear-sample and (for the BEV) blend. This is the same *map*
pattern 01.01 uses for its undistort/rectify stage, generalized in two ways this project introduces:
Half 1 swaps 01.01's Brown-Conrady polynomial pinhole-undistort model for a fisheye equidistant model
(closed-form both directions — no iteration); Half 2 turns the map into a genuinely 3-D one — every
BEV pixel's source is found by projecting a GROUND-PLANE POINT through 4 different camera geometries,
not by looking up one fixed 2-D table. Every output pixel is independent of every other, so one GPU
thread per output pixel (a 2-D grid matching the image shape) is the natural mapping; the BEV kernel
additionally runs a short, fixed-trip-count (4-camera) loop inside each thread rather than splitting
that loop across threads — `../src/kernels.cu`'s `bev_compose_kernel` header argues why.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** Perception (camera pre-processing / early perception), immediately downstream of
  raw sensor capture and upstream of both human-facing display and machine perception. Half 2's BEV
  output specifically sits at the boundary between perception and the driver-display / local-planning
  layers.
- **Upstream inputs:** 4 raw fisheye camera frames (message-shaped as `Image` — see SYSTEM_DESIGN.md
  item 3 — one per rig position, each carrying the camera's known, calibrated intrinsics + extrinsics
  as metadata).
- **Downstream consumers:** the human driver/operator display (BEV is a direct visualization product);
  a 14.02-style traversability/costmap layer (BEV pixels reclassified as drivable/obstacle); a 23.xx
  navigation-stack docking/parking controller (BEV or the rectilinear unwarp as a visual servo input);
  a person/obstacle detector run on the rectilinear or cylindrical unwarp (undoing fisheye distortion
  before a rectilinear-image-trained detector is a common production step).
- **Rate / latency budget:** 30 Hz camera capture is standard (SYSTEM_DESIGN.md item 1's camera rate);
  a production surround-view ECU budgets single-digit milliseconds per frame for unwarp + stitch so the
  whole pipeline (capture -> ISP -> unwarp/stitch -> display) stays under one frame period. This
  project's GPU kernels measure well under a millisecond each on the committed sample (see `demo/`
  timings) — real headroom is consumed by higher resolution, not this algorithm.
- **Reference robot(s):** the autonomous-vehicle stack (01/02/03/04/05/06/14/31/32) for the on-road
  parking/low-speed use case, and the warehouse AMR (02/04/05/23/06/08/25/31/32) for docking and
  close-quarters obstacle awareness — SYSTEM_DESIGN.md item 2.
- **In production:** a dedicated surround-view ECU or an SoC's ISP/vision accelerator running vendor
  firmware (Bosch, Continental, Valeo, or a Tier-1's in-house stack — PRACTICE.md §4), fed by
  GMSL2/FPD-Link-serdes-connected fisheye modules, with per-camera photometric calibration this
  project does not model (PRACTICE.md §3).
- **Owning team:** perception / camera systems (with close collaboration from the ADAS/autonomy
  integration team that consumes the BEV output) — SYSTEM_DESIGN.md item 5.

## The algorithm in brief

- **Equidistant fisheye projection/unprojection** (`r = f*theta`) — closed-form both directions; see
  [`THEORY.md`](THEORY.md) "The math" for the derivation and its contrast with the pinhole
  `r = f*tan(theta)` family.
- **Inverse-mapping LUT + bilinear gather** for both unwarp surfaces (rectilinear, cylindrical) — see
  [`THEORY.md`](THEORY.md) "The algorithm" (same pattern as 01.01, different camera model).
- **Rigid-body rig extrinsics** (`T_vehicle_cam` per camera, a nominal-basis + 45-degree-tilt
  construction) mapping a ground-plane point into each camera's frame — see
  [`THEORY.md`](THEORY.md) "The math".
- **The flat-ground assumption** — the geometric trick that turns a 3-D BEV reconstruction problem into
  a per-pixel closed-form computation, and its deliberate, measured failure near tall objects — see
  [`THEORY.md`](THEORY.md) "The problem".
- **Distance-weighted (linear feather) multi-camera blending** in overlap regions — see
  [`THEORY.md`](THEORY.md) "The algorithm".

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/fisheye-omnidirectional-unwarping-and-multi.sln`](build/fisheye-omnidirectional-unwarping-and-multi.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/fisheye-omnidirectional-unwarping-and-multi.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: none — this project links only the CUDA runtime + C++17
standard library (CLAUDE.md §5's default dependency budget).

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU and
7 physical gates):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at.

## Data

4 synthetic fisheye camera renders (front/left/right/rear, analytic ray-cast, 2x2 supersampled) of a
parking-lot-style ground plane + 3 tall objects, plus an exact orthographic ground-truth BEV crop —
generated by `scripts/make_synthetic.py` (fixed seed 42, ~13 s to regenerate). Full provenance,
checksums, and field documentation in [`data/README.md`](data/README.md).

## Expected output

The demo prints `PASS` for `VERIFY` (every GPU kernel matches `src/reference_cpu.cpp`'s independent CPU
twin within a documented tolerance) and all 7 physical gates (`model_roundtrip`,
`straightness_rectilinear`, `distortion_negative_control`, `bev_ground_truth`,
`flat_ground_assumption`, `seam_consistency`, `coverage`). The canonical lines live in
[`demo/expected_output.txt`](demo/expected_output.txt); measured numbers behind each verdict (never
diffed, since they can drift slightly across GPU architectures) are documented, with the actual values
measured on the reference machine, in `src/main.cu`'s tolerance block and in
[`THEORY.md`](THEORY.md) "How we verify correctness". The demo writes 7 artifacts to `demo/out/` —
see [`demo/README.md`](demo/README.md) for what each one shows.

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — **start here.** The single source of truth: the fisheye
   camera model (PART 1), the two unwarp output surfaces (PART 2), and the 4-camera rig + BEV
   compositor geometry (PART 3). Every camera-model and rig formula lives here, once.
2. [`src/main.cu`](src/main.cu) — entry point: loads data, runs both halves, verifies GPU vs CPU, runs
   the 7 gates, writes artifacts. Read its output-contract and tolerance-block comments before the code.
3. [`src/kernels.cu`](src/kernels.cu) — the GPU kernels: 2 LUT-builders, 1 generic bilinear-gather
   remap kernel (reused for both unwarp surfaces), and `bev_compose_kernel` (the 4-camera-loop-in-
   thread compositor — the project's centerpiece).
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the plain-C++ correctness oracle; read it beside
   `kernels.cu` to see exactly what parallelization changed.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, `paths.h`, and why they are copied, not shared.
6. [`scripts/make_synthetic.py`](scripts/make_synthetic.py) — the ray-cast renderer that authors the
   whole synthetic world (ground texture, objects, and the 4 fisheye views) from first principles.

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **OpenCV `fisheye` module** — the widely-used equidistant/Kannala-Brandt fisheye calibration and
  undistort API; study its `initUndistortRectifyMap` for the production version of this project's LUT-build step.
- **Kannala & Brandt, "A Generic Camera Model and Calibration Method for Conventional, Wide-Angle, and
  Fish-Eye Lenses" (2006)** — the polynomial generalization `r = f*(theta + k1*theta^3 + ...)` this
  project's equidistant model implicitly sets all `k_i = 0` for; the production fisheye calibration
  target virtually every real system fits against.
- **NVIDIA DriveWorks / VPI surround-view modules** — the closest production analogue to this project's
  Half 2: GPU-accelerated multi-camera BEV stitching with photometric blending, built for exactly the
  automotive use case this project's rig models.
- **Kalibr** (ETH Zurich) — the open-source multi-camera calibration toolbox that would, on a real rig,
  produce the intrinsics/extrinsics this project hardcodes as known.
- **OpenCV `stitching` module / Hugin / PTGui** — general panorama stitchers; study their cylindrical
  and spherical projection surfaces (the same idea as this project's Half 1b, generalized to many
  overlapping narrow-FOV photos instead of one fisheye source).

## Exercises

1. Change `kFishFx` in `kernels.cuh` (try 60.0f or 90.0f) and re-run — how does `bev_ground_truth`'s
   measured error change, and why (THEORY.md "Numerical considerations" gives the angular-resolution
   argument)?
2. Implement the Kannala-Brandt polynomial (`r = f*(theta + k1*theta^3)`) in `fisheye_project`/
   `fisheye_unproject`, non-zero `k1`, and update the model-roundtrip gate's hand-retyped formula to
   match — observe that the round-trip is no longer bit-exact (no closed-form inverse) and needs
   fixed-point iteration, exactly like 01.01's Brown-Conrady model.
3. Add a 5th "top-down" virtual camera to the rig (a bonus not modeled here) and extend the BEV
   compositor's camera loop and coverage bitmask to 5 bits.
4. Replace the linear feather weight (`kFeatherBandRad`) with a smoothstep or cosine ramp and measure
   whether `seam_consistency`'s measured error changes.
5. Move an object closer to a camera mount and re-run — watch the `flat_ground_assumption` gate's
   measured object-region error grow, and inspect `demo/out/error_heatmap.pgm` to see the ghost smear
   lengthen.

## Limitations & honesty

- The fisheye model is pure equidistant (`k_i = 0` in the Kannala-Brandt sense) — a real lens's
  calibrated distortion polynomial is never exactly zero-order; THEORY.md "Where this sits in the real
  world" names the production fitting procedure this project's model stands in for.
- The rig geometry (mount positions, 45-degree tilt) is illustrative, not measured from a real vehicle
  or calibration target — PRACTICE.md §3 describes the real calibration procedure this project skips.
- `bev_ground_truth` and `seam_consistency` measure real, non-trivial reconstruction error (roughly
  4-7% of the 0-255 scale) dominated by this project's deliberately low fisheye resolution (74
  px/radian, chosen for a small, fast-to-build committed sample) interacting with thin high-contrast
  features (lane stripes, object silhouettes) — `src/main.cu`'s tolerance-block comment explains the
  measured numbers in full; this is a real, honestly-reported limitation of the committed scene, not a
  hidden bug.
- No photometric (auto-exposure/white-balance) mismatch between cameras is modeled — every camera
  renders the identical synthetic scene under identical (implicit) lighting, so `seam_consistency`'s
  measured disagreement is purely geometric/sampling, never photometric; a real rig's biggest seam
  artifact is usually photometric (PRACTICE.md §1 discusses this honestly).
- Objects are static and the scene has no motion; a real BEV system also fuses multiple frames over
  time, which this project's single-frame reconstruction does not attempt.
- Everything here is educational and **sim-validated only** — this project's output is a visualization/
  perception artifact, not a control signal, but if any derivative work feeds it into a planner or
  controller that could command real hardware motion, that work is the owner's decision and
  responsibility, and must be validated far beyond what a synthetic scene like this one can prove.
