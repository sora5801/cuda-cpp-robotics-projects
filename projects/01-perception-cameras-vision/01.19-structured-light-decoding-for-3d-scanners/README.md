# 01.19 — Structured-light decoding (Gray code, phase shift) for 3D scanners

**Difficulty:** intermediate · **Domain:** 1. Perception — Cameras & Vision

> Catalog bullet (source of truth, verbatim): `Structured-light decoding (Gray code, phase shift) for 3D scanners`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

A structured-light 3-D scanner replaces one camera of a stereo pair with a **projector**: instead of
matching texture between two cameras, it projects known patterns and reads back where each pattern
landed. This project implements the two classic **temporal coding** schemes — binary **Gray code**
(coarse, absolute, integer-column) and sinusoidal **phase shift** (fine, sub-pixel, but wrapped) —
teaches them separately, then combines them the way real scanners do: Gray code resolves *which*
period the phase is in, phase resolves *where within it*, sub-pixel. The result triangulates into a
metric 3-D point cloud. A synthetic scene (a tilted plane, a sphere, and a box's raised top face) is
rendered under all 20 patterns with realistic ambient light, sensor noise, mild blur, and a
deliberately low-albedo "dark stripe" region, so the demo can measure — not just claim — Gray code's
famous single-bit-boundary robustness over plain binary coding, the sub-pixel payoff of adding phase
shift, and whether the confidence mask genuinely rejects (rather than hallucinates through) a
low-signal region. Every stage runs on the GPU and is checked against an independent CPU oracle;
eight further gates check the decoded/reconstructed results against the synthetic ground truth.

Everything named in the catalog bullet is implemented: Gray code, phase shift, and the hybrid
combination. Nothing here is a documented-only stub.

## What this computes & why the GPU helps

Five stages, **every one a pure per-pixel (or per-sample) MAP** — the output of pixel `i` depends
only on the inputs at pixel `i`, never on any other pixel's inputs or outputs:

- **Gray decode** — per camera pixel, threshold 7 (pattern, inverse) frame pairs and un-reflect the
  resulting Gray codeword into an absolute integer projector column.
- **Phase decode** — per camera pixel, `atan2` four phase-shifted captures into a wrapped sub-pixel
  phase, and derive a modulation-amplitude **confidence** signal for free from the same arithmetic.
- **Hybrid combine** — per camera pixel, snap the phase's period estimate to the nearest integer
  consistent with the Gray-decoded column, producing one absolute, sub-pixel projector column.
- **Triangulate** — per camera pixel, intersect the camera ray with the projector-COLUMN's plane in
  3-D (closed form, no iteration) to get a metric `(x,y,z)` point.
- **Boundary stress test** — per synthetic 1-D probe, the same bit-decision physics as Gray decode,
  applied to a designed noise experiment instead of the real scene (the Gray-vs-binary lesson).

Because every stage is a MAP with **zero cross-pixel/cross-sample dependencies**, the natural (and, on
a GPU, close to optimal) mapping is one thread per pixel/sample — the entire pipeline needs no shared
memory and, notably, **no atomics anywhere** (contrast with 01.18's depth-completion scatter/gather,
which genuinely needs them because multiple sources can write one destination). The pattern-index loop
inside each kernel is deliberately the INNER loop over a **pattern-major** memory layout
(`pattern[p][pixel]`), so every iteration is one coalesced 128-byte warp transaction — the same
coalescing lesson 08.01's transposed noise array teaches, applied here to images. See
[`THEORY.md`](THEORY.md) "The GPU mapping" for the full argument, including why this project's
kernels are bandwidth-bound, not compute-bound, and honestly reports a modest (~3x) measured speed-up
at this problem's small size (30,000 pixels — a toy compared to a real 1–20 MP scanner, where the
same mapping would show a dramatically larger win).

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial
whole (see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** `Sensors → Perception`, specifically the 3-D-from-2-D reconstruction step that
  feeds everything downstream of it (SYSTEM_DESIGN.md §1).
- **Upstream inputs:** the raw camera frame stack from a synchronized camera + projector rig (20
  `Image` messages, one per pattern, `1` channel, 8-bit — SYSTEM_DESIGN.md §3.6), plus a one-time
  **calibration** (camera intrinsics, projector-as-inverse-camera intrinsics, and the camera-
  projector extrinsic baseline) produced by a project like 01.16 (checkerboard/ChArUco intrinsics)
  and 01.17 (camera-camera extrinsic calibration — the SAME reprojection-error math this project's
  triangulation assumes was already solved, by name).
- **Downstream consumers:** a `PointCloud` (SYSTEM_DESIGN.md §3.6) in the camera frame — consumed by
  grasp planning (19.01 antipodal grasp scoring), pick-verification / metrology inspection (01.13/
  01.14), or SLAM/mapping fusion (05.x) depending on the cell.
- **Rate / latency budget:** temporal coding costs **N frames per scan** (20 here) — this is a
  fundamentally single-shot-per-N-frames, STATIC-SCENE technique, not a streaming 30-60 Hz sensor
  (SYSTEM_DESIGN.md §1.1's camera row assumes ONE frame per estimate; structured light spends its
  frame budget acquiring the code instead). A real bin-picking cell budgets ~200 ms-2 s per full
  scan-and-decode cycle (pattern projection is usually the bottleneck, not this project's decode
  math, which THEORY.md's GPU-mapping section shows comfortably fits in a millisecond at 1 MP). Motion
  during the scan corrupts every temporally-coded pixel it touches — the honest cost this project's
  README "Limitations" and PRACTICE.md both flag.
- **Reference robot(s):** the **6-DoF manipulator work cell** (SYSTEM_DESIGN.md §2.2), which
  explicitly lists "wrist- or frame-mounted stereo/**structured-light** 3D camera" in its sensor
  suite, and composition Chain B (§4.2, `01.02 → 19.01 → 09.05 → 06.07 → 08.03`) — this project is a
  drop-in alternative FRONT END to that chain's first block, `01.02` (passive stereo), for exactly
  the scenes where passive stereo struggles (see "Prior art" below).
- **In production:** industrial structured-light scanners (Zivid, Photoneo, Ensenso, LMI Gocator) —
  see README "Prior art" for what to study and PRACTICE.md for the hardware/BOM story.
- **Owning team:** perception / 3-D sensing, inside the Perception org (SYSTEM_DESIGN.md §5.1: "01,
  02, 03, 20"), adjacent to the calibration team (01.16/01.17) and the manipulation/grasp-planning
  team (19.x) that consumes this project's point cloud.

## The algorithm in brief

- **Gray-code binary temporal coding** — `N=7` bit-plane patterns, each captured direct + inverse,
  thresholded pixel-by-pixel and un-reflected into an absolute integer column ([`THEORY.md`](THEORY.md#the-math)).
- **Single-bit-adjacency proof (why Gray, not plain binary)** — consecutive Gray codewords differ in
  exactly one bit; consecutive plain-binary codewords can differ in up to `N` bits at once (the MSB
  boundary is the worst case) — measured on this project's own boundary stress test
  ([`THEORY.md`](THEORY.md#the-algorithm)).
- **4-step phase-shift profilometry** — `atan2`-based phase retrieval with the ambient/albedo
  cancellation derived from first principles, and the modulation amplitude read off the SAME
  arithmetic as a confidence signal ([`THEORY.md`](THEORY.md#the-math)).
- **Phase-guided period snapping (the hybrid)** — the production pattern: Gray code's coarse,
  absolute period estimate is corrected to be CONSISTENT with phase's precise fractional position,
  rather than trusted verbatim ([`THEORY.md`](THEORY.md#the-algorithm)).
- **Ray / projector-plane triangulation** — a fixed projector column is a PLANE in 3-D (not a ray);
  closed-form ray-plane intersection recovers the 3-D point, no iterative solve needed
  ([`THEORY.md`](THEORY.md#the-math)).

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/structured-light-decoding-for-3d-scanners.sln`](build/structured-light-decoding-for-3d-scanners.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/structured-light-decoding-for-3d-scanners.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: none — CUDA toolkit runtime + C++17 standard library only
(no cuBLAS/cuFFT/Thrust; the small plane/sphere normal-equations solves in the reconstruction gates
are hand-rolled Gaussian elimination on the host, see `src/main.cu`).

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at.

## Data

100% synthetic — a ray-cast scene (tilted plane + sphere + box) rendered under all 20 patterns with
realistic ambient light, sensor noise, and mild blur, plus exact ground truth (continuous projector
column, depth, surface identity) that no photograph could provide. Regenerate with
`python scripts/make_synthetic.py --seed 42`. Full field documentation, size accounting, checksums,
and the "how the sample was tuned" measurement log: [`data/README.md`](data/README.md).

## Expected output

The demo runs all **5 pipeline stages** on the GPU and an independent CPU oracle and requires exact
or near-exact agreement (`VERIFY:` lines — integer stages exact, floating-point stages within a
documented tolerance, e.g. `phase_decode` within `1e-3` rad for `atan2f`/`sqrtf` host/device ULP
drift). It then runs **8 independent gates** against the synthetic ground truth (`GATE:` lines) —
measured on the committed sample: Gray decode recovers the exact integer column on **97.8%** of
confident pixels; the phase-refined hybrid answer's mean error (**0.071** columns) is **3.8x** tighter
than Gray alone (**0.267** columns); plain binary coding's boundary error rate (**7.9%**) is **~31x**
worse than Gray's (**0.26%**) under identical noise; adding 40 counts of ambient light to every phase
frame changes the decoded phase by **0** (the ambient-cancellation algebra, exactly); the reconstructed
background plane fits its truth within **3.8 mm** RMS, the sphere's radius within **1.4%**, and the
box step's height within **0.5%**; and the confidence mask rejects **99.1%** of the deliberately
low-albedo "dark stripe" cohort with **zero** surviving pixels hallucinating a depth more than 60 mm
wrong. The canonical stable lines live in [`demo/expected_output.txt`](demo/expected_output.txt);
every measured number above is reproduced live as an `[info]` line (not diffed — see `demo/README.md`).

Six labeled artifacts land in `demo/out/` every run: two sample patterns, the decoded column map, the
confidence map, the point cloud (CSV), an orthographic profile render (PPM — the sphere and box
literally look like bumps in the rendered profile), and a full-precision gate-metrics CSV.

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — the scanner model, code parameters, and the whole
   pattern-stack/state contract every other file shares. Read this FIRST.
2. [`src/main.cu`](src/main.cu) — orchestration: load the 20-pattern stack + ground truth, run all
   five stages GPU + CPU, the eight gates, the artifacts.
3. [`src/kernels.cu`](src/kernels.cu) — the five GPU kernels (the heart of the project).
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the independently-written CPU twin of every
   stage, plus the full derivations in comments.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, and `paths.h`'s data/artifact resolution.

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **OpenCV `structured_light` module** (`GrayCodePattern`, `SinusoidalPattern`) — the reference
  open-source implementation of exactly the two codes this project teaches; compare its decode API
  to this project's kernels.
- **Zhang & Huang, "Novel method for structured light system calibration"** (2006) and the broader
  phase-shifting-profilometry literature — the standard references for the `atan2` phase-retrieval
  derivation this project's THEORY.md walks through.
- **Sansoni, Carocci & Rodella, "Three-dimensional vision based on a combination of gray-code and
  phase-shift light projection"** (Applied Optics, 1999) — the original "Gray resolves the period,
  phase refines it" hybrid this project implements.
- **Salvi, Fernandez, Pribanic & Llado, "A state of the art in structured light patterns for
  surface profilometry"** (Pattern Recognition, 2010) — a broad survey of the coding-scheme design
  space (temporal vs. spatial, binary vs. N-ary, single-shot vs. multi-shot) this project's
  README "Limitations" situates itself within.
- **PCL (Point Cloud Library)** — what a real pipeline would do with this project's output point
  cloud next (filtering, registration, meshing).
- **01.02 (this repo, stereo SGM)** and **01.20 (this repo, time-of-flight)** — the two competing
  active/passive 3-D sensing modalities named by contrast throughout this project's docs.

## Exercises

3–5 "try this next" extensions for the learner, ordered easiest first.

1. **Verge the rig.** Change `kRcp`/`kTcp` in `kernels.cuh` from the parallel-axis identity rotation
   to a small verged rotation (a few degrees toed-in) and confirm `triangulate_kernel`'s general
   `n_cam = R_cp * n_p` formula still triangulates correctly — no code changes needed, only data.
2. **Tighten or loosen the confidence floor.** Sweep `kDefaultConfidenceFloor` and watch the
   `dark_stripe_honesty` and `gray_decode` gates trade off (see `data/README.md` "How the sample was
   tuned" for the measurements that picked 25).
3. **Add a third phase-shift step count.** Implement a 3-step phase decode (the theoretical minimum —
   THEORY.md derives why 4 was chosen here) and compare its confidence/noise sensitivity to the 4-step
   version on the same rendered scene.
4. **On-device noise generation.** The boundary stress test currently pre-draws its noise on the
   host (like 08.01's MPPI eps array) so GPU/CPU comparison stays exact — port it to cuRAND and
   measure what changes about the verification story (08.01 Exercise 4 is the same lesson).
5. **A second dark region with a DIFFERENT failure mode.** Add a highly OBLIQUE (not just dark)
   patch to `make_synthetic.py` and see whether the modulation-amplitude confidence signal catches
   foreshortening-driven fringe-contrast loss the same way it catches low albedo (THEORY.md "The
   problem" previews why it should, physically).

## Limitations & honesty

- **Single-shot cost.** Every scan needs all 20 patterns projected and captured — a moving scene (or
  camera) corrupts the temporal code. Real single-shot alternatives exist (spatial/frequency-domain
  coding, e.g. Kinect-v1-era pseudorandom dot patterns) and are named honestly in THEORY.md "Where
  this sits in the real world" as a scoping decision, not implemented here — the catalog bullet asks
  specifically for the temporal (Gray code, phase shift) family.
- **The box is a simplification.** `make_synthetic.py` models the "box" as a single fronto-parallel
  raised plane bounded by a world-space rectangle (a clean depth STEP), not a full 3-D box with side
  walls and self-occlusion — chosen because it teaches the reconstruction-accuracy gate just as well
  with far less rendering complexity (documented in the generator's own comments).
- **A narrow, honestly-measured blind spot in the confidence mask.** Confidence here is modulation
  amplitude from the PHASE decode; Gray decode has its own, separate noise sensitivity (per-bit
  threshold margin) that the same signal does not perfectly guard against at marginal SNR. The
  committed sample's confidence floor was tuned (data/README.md) to drive this residual risk to zero
  on THIS scene — a different scene or noise level could reopen it. Production scanners layer
  additional consistency checks (e.g. multi-frequency phase unwrapping, temporal averaging) precisely
  because single-signal confidence is not airtight; see THEORY.md "Numerical considerations".
- **Camera-projector calibration is assumed, not performed.** This project consumes fixed, known
  intrinsics/extrinsics (`kernels.cuh`) exactly as 01.16/01.17 would produce them — no calibration
  routine is implemented here (that is those projects' job).
- **Not safety-certified; sim-validated only.** This project computes a point cloud from synthetic
  data. If used to steer a real robot's motion (e.g. bin-picking grasp planning), that downstream
  motion is the owner's decision and responsibility — see PRACTICE.md §3 for the hardware-testing
  ladder every real deployment should climb.
