# 20.01 — GelSight/DIGIT processing: contact patch, shear field via optical flow, slip detection in real time

**Difficulty:** ★ beginner · **Domain:** 20. Tactile & Force Sensing

> Catalog bullet (source of truth, verbatim): `★ GelSight/DIGIT processing: contact patch, shear field via optical flow, slip detection in real time`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

GelSight-style tactile sensors read touch as **images**: a soft, marker-printed gel membrane presses
against an object, an internal camera watches the membrane deform, and a GPU pipeline turns that video
into physical numbers a grasp controller can act on. This project builds that pipeline end to end —
**contact patch** (where and how much is the gel touching?), **shear field** (how is the touched patch
sliding, via sparse marker tracking rather than dense optical flow?), and **slip detection** (has the
grip started to fail?) — and runs it on a fully synthetic, physics-grounded gel-sensor sequence: a
sphere presses in, shears sideways, then partially slips outward-in exactly as Cattaneo-Mindlin contact
theory predicts. All three components named in the catalog bullet are implemented and gated against the
physics that generated the scene, not against themselves. The demo produces a contact-mask image, a
shear-field vector table, and a slip-score timeline — the three artifacts a learner should study first.

> **Template placeholder notice.** As scaffolded, `src/` contained a tiny fully-working SAXPY
> placeholder to validate the toolchain. It has been fully replaced by this project's real pipeline —
> every kernel below is the tactile-processing implementation, not the placeholder.

## What this computes & why the GPU helps

Per frame: a **map** over 76,800 pixels (contact-mask threshold, twice more for the 3x3 morphological
open) plus a **small-N search** over 221 markers (local-minimum detection in a bounded window, then a
trivial per-marker displacement/validity computation) — five kernels, each mapped to whichever GPU
pattern actually fits its data size, not a one-size-fits-all kernel:

- **Contact mask + morphological open** — pure per-pixel *map* (threshold), then two 3x3 *stencil*
  passes (erode, dilate) that kill single-pixel speckle without hand-tuning the threshold to be
  conservative. → [THEORY.md](THEORY.md) §The GPU mapping.
- **Patch stats** — *map + atomic reduction*: every lit mask pixel adds itself to a running area/
  centroid sum. At 320x240 this is sub-microsecond work; the file is explicit about the shared-memory
  block-reduction optimization it deliberately skips, and why.
- **Marker detect + track** — *one thread per marker* (221 threads, not 76,800): each thread searches
  a small window near its marker's known rest position for the darkest pixel, then computes
  displacement, validity, and contact-patch membership. This is the project's clearest lesson in
  matching GPU granularity to the actual size of the problem — see "The algorithm in brief" below.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** the **sensors/perception boundary** — SYSTEM_DESIGN §1's stack lists domain 20
  (tactile) under the SENSORS box, but this project's *code* is exactly the PERCEPTION-layer processing
  a raw tactile-camera stream needs before anything downstream can use it — the tactile analogue of
  [`01.02`](../../01-perception-cameras-vision/01.02-stereo-depth/README.md)'s stereo-depth pipeline: a
  camera-shaped sensor in, a physically meaningful measurement out.
- **Upstream inputs:** raw gel-camera frames — an `Image` (SYSTEM_DESIGN §3.6: `header`, `width=320`,
  `height=240`, `channels=1`) at the sensor's native camera rate. In this project the frames are
  synthetic (rendered from a physics scenario, not captured — see "Data" below); on real hardware they
  come from the sensor's internal USB/MIPI camera (PRACTICE §2–§3).
- **Downstream consumers:** a grasp-force control loop. Concretely, this pipeline's `slip_score`/
  `slip_declared` output is exactly the signal
  [`20.04` (Learned slip prediction fused into the grasp control loop)](../20.04-learned-slip-prediction-fused-into-the-grasp/README.md)
  builds on (this project computes slip *analytically*, from geometry; 20.04's bullet is the learned
  counterpart), and the contact-patch/shear-field output is the kind of feedback a grasp executor like
  [`19.01` (Parallel grasp-candidate scoring)](../../19-manipulation-grasping/19.01-parallel-grasp-candidate-scoring/README.md)
  would close its loop around once a grasp is executing, not just planned.
- **Rate / latency budget:** tactile cameras run at ordinary camera rates, **30-60 Hz**
  (SYSTEM_DESIGN §1.1) — this project's own measured GPU pipeline is ~0.2 ms/frame (Release,
  RTX 2080 SUPER), leaving comfortable headroom at 30-60 Hz. The honest pressure is downstream: a
  **slip reaction budget is much tighter than the sensor rate** — grasp-force literature and
  commercial tactile-sensing stacks target closing a slip-detected-to-grip-adjusted loop in roughly
  **~10 ms**, i.e. within 1-2 camera frames, because a slipping object accelerates under gravity the
  instant grip force drops below what friction needs (a few tens of milliseconds of inaction can mean a
  dropped part). At 30-60 Hz camera rate, that budget is a **single-digit number of frames**, not
  dozens — this project's pipeline easily fits the *compute* budget, but a real system's total latency
  (sensor read + USB transfer + this pipeline + the force controller's own tick) must be budgeted end
  to end, and this project states honestly that it does not model sensor/bus latency at all (see
  Limitations).
- **Reference robot(s):** the **6-DoF manipulator work cell** (SYSTEM_DESIGN §2.2). Note honestly:
  SYSTEM_DESIGN's own work-cell block diagram (§2.2, domains 01/19/09/06/07/08/21/24) does not yet draw
  an explicit tactile-feedback path — this project's natural position is a fast local loop alongside
  **JOINT CONTROL [08 →]**, feeding a slip signal into the gripper's force command, the same slot a
  wrist F/T sensor ([`20.05`](../20.05-f-t-processing/)) or an e-skin array
  ([`20.02`](../20.02-e-skin-taxel-arrays/)) would occupy.
- **In production:** GelSight Inc.'s and Meta's DIGIT sensor stacks, and academic marker-tracking
  pipelines built on OpenCV's optical-flow/blob-detection primitives — see "Prior art" below.
- **Owning team:** manipulation / tactile sensing, inside controls & autonomy (SYSTEM_DESIGN §5.1) —
  adjacent to perception (owns the camera/ISP side of the sensor) and to the ML/data team once slip
  detection becomes learned (20.04's territory).

## The algorithm in brief

- **Contact patch** — background subtraction against a calibration (no-contact) frame, thresholded,
  then a binary **morphological open** (erosion then dilation over a 3x3 neighborhood) to remove
  speckle without shrinking the real patch; area and centroid via a reduction. →
  [THEORY.md](THEORY.md) §The algorithm.
- **Shear field via marker tracking** — GelSight-style sensors print a regular dot grid on the gel
  specifically so the sensor doesn't need dense optical flow: each of 221 markers is found by
  searching a small window around its known rest position for the darkest pixel (a scoped
  local-minimum search), and displacement is that position minus rest — sparse, cheap, and immune to
  the low-texture gel background that would defeat a Lucas-Kanade-style dense tracker. →
  THEORY.md §The algorithm, §The GPU mapping.
- **Slip detection** — a closed-form 2-D rigid (rotation + translation) least-squares fit over every
  in-contact marker's displacement (the complex-number Procrustes trick — no SVD needed in 2-D, kept on
  the host as O(markers) trivial arithmetic, the same call [`08.01`](../../08-control-systems/08.01-mppi-controller-the-canonical-gpu-controller/README.md)
  makes for its softmin blend); the fraction of markers whose fit RESIDUAL exceeds a threshold is the
  slip score, and slip is declared when that score crosses a documented bound. →
  THEORY.md §The math, §How we verify correctness.
- **The synthetic sensor model itself** — a Hertzian sphere-contact footprint, a paraboloid
  "intensity-proxy" shading model, and a Cattaneo-Mindlin partial-slip annulus are what generate the
  scene AND the ground truth the demo grades against — see THEORY.md §The problem for the physics.

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/gelsight-digit-processing.sln`](build/gelsight-digit-processing.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/gelsight-digit-processing.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none** — CUDA runtime + C++17 standard library only. The
synthetic sensor renderer runs entirely on the host in plain C++ (no image-loading library needed —
there are no image files to load; see "Data" below).

## Run the demo

One command, from this folder (builds first if needed, runs the full 100-frame sequence, checks GPU vs
CPU on every frame, then checks the result against the scene's own physics):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at, and for the three artifacts it writes.

## Data

The committed sample is a **scenario**, not recordings: `data/sample/tactile_scenario.csv` (581 bytes,
synthetic, no RNG) — which indenter shape presses in (`sphere`, the calibrated default) and the gel's
fixed micro-texture seed. Every frame of the 100-frame sequence is *rendered in code* from that
scenario plus the fixed physical model in `src/kernels.cuh` (Hertzian contact radius, the shading
model, the Cattaneo-Mindlin stick/slip law) — the same "commit the task, render in-demo" pattern
[`08.01`](../../08-control-systems/08.01-mppi-controller-the-canonical-gpu-controller/README.md) uses
for its cart-pole scenario. No public dataset applies (this project needs known, dense, analytic ground
truth to grade against — a real captured tactile log has no such thing); `scripts/download_data.ps1` is
an honest, permanent no-op. Details, the field-by-field format, and the "why a scenario, not images"
reasoning: [`data/README.md`](data/README.md).

## Expected output

Nine stable lines — banner, `PROBLEM:`, `SCENARIO:`, `VERIFY:`, `CONTACT:`, `SHEAR:`, `SLIP:`,
`ARTIFACT:`, `RESULT:` — checked as a subset diff by
[`demo/expected_output.txt`](demo/expected_output.txt). Two independent kinds of verification:

1. **The §5 GPU-vs-CPU gate (VERIFY), EXACT, every frame:** all five kernels' outputs (mask,
   morphology, patch stats, marker detect, marker track) are compared against
   [`src/reference_cpu.cpp`](src/reference_cpu.cpp) on all 100 frames — every operation in this
   pipeline is integer/threshold arithmetic on a shared uint8 input, so this is bit-for-bit equality,
   not a tolerance (measured: **0 mismatches**).
2. **Three ground-truth gates**, the algorithm's measurement vs. the physics that generated the scene
   (never against itself), all measured on the reference machine (RTX 2080 SUPER, sm_75):
   - **CONTACT** — patch area within **1.3%** (mean and max) of the Hertzian footprint, centroid within
     **0.13 px**, over the 8 press-hold frames. Gate: 5% area, 1.0 px centroid.
   - **SHEAR** — mean tracked displacement across in-contact markers within **0.00 px** (exact) of the
     commanded translation, over the 12 shear-hold frames. Gate: 0.5 px.
   - **SLIP** — onset detected at frame **85**, modeled onset (from the same Cattaneo-Mindlin formula
     the scenario is built from) at frame **86**, |error| = **1 frame**; zero false slip declarations
     across the 60 stick-phase frames. Gate: |error| <= 2 frames, per the catalog's own tolerance.

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — read this FIRST, not second: it is both the kernel contract
   AND the full sensor-model specification (image/marker-grid layout, the Hertzian contact formulas,
   the Cattaneo-Mindlin stick/slip law, the phase timeline) — every number the renderer and every gate
   use traces back to one place here.
2. [`src/main.cu`](src/main.cu) — the synthetic-sensor renderer (Section A: `compute_frame_state`,
   `render_frame`), the host-side rigid-fit slip scorer (Section B), and the orchestration loop
   (Section C/`main`): render → GPU pipeline → CPU oracle → per-frame exact verify → ground-truth
   gates → artifacts.
3. [`src/kernels.cu`](src/kernels.cu) — the five GPU kernels themselves, each commented with its
   thread-to-data mapping and why that mapping fits the kernel's data size. Start with
   `detect_markers_kernel` — the single most interesting kernel here (one thread per marker, not per
   pixel, and the file header argues why that is the *correct* choice, not a shortcut).
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the plain-C++ correctness oracle; every function
   is a line-by-line sequential twin of its `kernels.cu` counterpart.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, and why they are copied, not shared.

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **GelSight Inc. / Yuan, Dong & Adelson (2017), "GelSight: High-Resolution Robot Tactile Sensors for
  Estimating Geometry and Force"** — the sensor family this project is named after; production GelSight
  reconstructs full surface depth from *three-color photometric stereo*, not the single-scalar
  intensity proxy this project uses (see THEORY.md's honest scoping note).
- **Lambeta et al. (2020), "DIGIT: A Novel Design for a Low-Cost Compact Robotic Tactile Sensor"**
  (Meta/Facebook AI Research + GelSight Inc. + Wonik Robotics) — the compact, open-hardware descendant
  that made vision-based tactile sensing affordable enough for research fleets; its marker-tracking
  firmware/software is the direct ancestor of this project's shear-field kernel.
- **TacTip** (Bristol Robotics Laboratory, Ward-Cherrier et al.) — a biomimetic alternative: pins on a
  soft dome, camera watches pin-TIP deflection rather than a flat gel's surface shading. Compare its
  marker-tracking pipeline (very close to this project's) against its completely different mechanical
  transduction.
- **uSkin** (XELA Robotics) — a magnetic (not vision-based) tactile skin: 3-axis MEMS Hall-effect
  taxels under a soft cover. Worth comparing against GelSight's approach for the classic
  resolution-vs-coverage-vs-cost tradeoff in tactile sensor families (PRACTICE §2).
- **Johnson, K. L., *Contact Mechanics* (1985)** — the textbook source for both physics formulas this
  project implements directly: the Hertzian contact-radius law and the Cattaneo-Mindlin partial-slip
  annulus (ch. 3 and ch. 7).
- **OpenCV's `goodFeaturesToTrack` / Lucas-Kanade optical flow** — the general-purpose dense/sparse
  tracking machinery this project's marker search deliberately avoids; THEORY.md's "GPU mapping"
  section makes the comparison explicit.

## Exercises

1. **Plot the artifacts:** `demo/out/slip_timeline.csv` → slip_score vs. frame (the clearest picture of
   the whole run); `demo/out/shear_field.csv` → a quiver plot of marker displacement at the shear-hold
   frame; open `demo/out/contact_mask.pgm` directly in an image viewer.
2. **Break the residual threshold:** halve and double `kResidualSlipThresholdPx` in `kernels.cuh`,
   rebuild, and watch the measured slip-onset frame move — then explain from the Cattaneo-Mindlin
   annulus model why the relationship is not linear (THEORY.md §Numerical considerations has the
   ingredients).
3. **Regenerate with a different seed:** `python scripts/make_synthetic.py --seed 7`, rebuild, rerun —
   confirm the ground-truth gates still pass (they should; the gel's fixed micro-texture noise never
   drives the physics, only the exact pixel bytes).
4. **Try the edge indenter:** `python scripts/make_synthetic.py --indenter edge`, rebuild, rerun. The
   contact patch becomes an elongated strip instead of a disk — watch `VERIFY:` still pass (the
   pipeline doesn't know or care about indenter shape) while the `CONTACT:`/`SHEAR:`/`SLIP:` gates,
   calibrated for the sphere scenario, are not expected to hold unmodified (Limitations below).
5. **Sub-pixel marker localization:** `detect_markers_kernel` returns the integer-pixel argmin; add an
   intensity-weighted centroid refinement over the 3x3 neighborhood around it and measure how much the
   SHEAR gate's already-tiny error changes.

## Limitations & honesty

- **Intensity-proxy indentation, not photometric-stereo depth.** Real GelSight/DIGIT sensors
  reconstruct a full depth map from three-color directional lighting and gradient integration; this
  project darkens pixels by a single scalar proportional to local indentation depth — a deliberately
  simplified v1, stated in THEORY.md's "physics-first" section, not hidden.
- **A purely GEOMETRIC contact/friction model, not a solved elastic boundary-value problem.** The
  paraboloid shading profile and the Cattaneo-Mindlin stick/slip law are the standard SMALL-DEFLECTION
  approximations from Johnson's *Contact Mechanics* — real gel deformation, especially near large
  strains or with the gel's own viscoelastic relaxation, would differ from this teaching model.
- **The rigid fit is ordinary least squares, not robust.** THEORY.md's "Numerical considerations"
  section names the specific failure mode this causes late in the slip phase (once slipping markers
  outnumber stuck ones, the fit itself starts drifting toward the majority) — a RANSAC/IRLS fit is the
  production fix (Exercise territory).
- **No sensor/bus latency is modeled.** The System-context section above states the real ~10 ms
  slip-reaction budget honestly; this project only measures GPU *compute* time, not the camera-read,
  USB-transfer, or force-controller-tick latency a real system would have to budget end to end.
- **Edge-indenter mode is supported but not gated** (Exercise 4) — the renderer and pipeline work for
  it, but this project's ground-truth thresholds are calibrated for the sphere scenario only.
- **Sim-validated only, and this one matters here (CLAUDE.md §1):** this project's output (a slip flag)
  is exactly the kind of signal that would feed a grasp-force controller commanding real motor current.
  Everything here ran only against a synthetic, physics-modeled scene; nothing is safety-certified, and
  any real-hardware use would demand the full testing ladder (PRACTICE §3) plus an independent safety
  envelope around the grasp controller it feeds.
