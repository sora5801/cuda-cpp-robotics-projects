# 01.06 — AprilTag / ArUco GPU detector-decoder for high-rate fiducial localization

**Difficulty:** ★ beginner · **Domain:** 1. Perception — Cameras & Vision

> Catalog bullet (source of truth, verbatim): `★ AprilTag / ArUco GPU detector-decoder for high-rate fiducial localization`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

A square fiducial marker — a black-bordered checkerboard-like tag with a unique bit pattern inside —
is one of the cheapest, most reliable ways a robot can find EXACTLY where a known object or
workspace point is, in 3-D, from a single camera frame. This project builds a complete
**detector-decoder for a home-grown 32-code fiducial family** (6x6 grid: a solid black border ring
around a 4x4 = 16-bit payload — the same geometry as AprilTag 16h5 and ArUco's 4x4 dictionaries, but
with independently generated codes, never their published bit tables) entirely on the GPU: adaptive
threshold, connected-component labeling, quad extraction, a per-candidate 4-point homography solve,
perspective grid sampling + dictionary decoding, and pose recovery from the homography — six stages,
each taught as its own small kernel.

A learner who studies this project comes away understanding: why a fiducial dictionary needs a
minimum Hamming distance (and how to search for one); how a homography turns 4 point
correspondences into a full 2-D-to-2-D projective map via a tiny linear solve; how to warp-sample an
image through that map to read a digital payload back out of a photograph; and how to recover an
approximate 6-DoF pose from that same homography — plus, honestly, where each of these teaching-scope
methods is weaker than the production algorithms named throughout (AprilTag's line-fitting corner
refinement, IPPE pose estimation). The demo runs the whole pipeline on the GPU **and** an independent
CPU oracle on three synthetic scenes, cross-checks them, and grades the result against five
independent gates — including a **built-in negative control**: tags with payload bits corrupted
exactly at, and one bit beyond, the dictionary's own error-correction capacity.

## What this computes & why the GPU helps

The pipeline has two very different halves, and naming the contrast IS the lesson (THEORY.md "The GPU
mapping" has the measured numbers):

- **Stages 1-2 (adaptive threshold, connected-component labeling) are PIXEL-parallel.** Every one of
  the scene's 172,800 pixels does independent (stage 1: MAP/STENCIL) or neighbor-coupled (stage 2:
  iterative STENCIL+ATOMIC relaxation) work — one thread per pixel is the natural mapping, identical in
  spirit to 30.01's connected-component labeler.
- **Stages 3-6 (quad extraction, homography solve, grid decode, pose) are CANDIDATE-parallel.** After
  the pixel-parallel stages and a small host-side compaction scan, a scene has single digits to a few
  dozen SURVIVING candidate components — one thread per candidate does a short, sequential job (a
  36-point sample loop, an 8x8 linear solve) that would not itself parallelize further, because the
  work item so small it doesn't need to.
- **Measured (RTX 2080 SUPER, Release, this project's committed scenes):** the pixel-parallel stages
  take ~7-13 ms per 480x360 scene (dominated by the CCL convergence loop's ~150-190 sweeps, each a
  host round-trip — see THEORY.md "Numerical considerations" for the production fix); the
  candidate-parallel stages, over 2-6 candidates, take ~0.16-0.25 ms — three orders of magnitude
  less work for two orders of magnitude fewer threads, exactly the contrast the two launch geometries
  predict.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** the **perception layer**, a specialized sibling of 01.01's general image
  pipeline — it consumes a RECTIFIED image (SYSTEM_DESIGN section 1: `SENSORS -> PERCEPTION -> STATE
  ESTIMATION`) and outputs a small list of precisely-localized 6-DoF poses, one of the few perception
  outputs precise enough to feed directly into CONTROL without an intervening estimator.
- **Upstream inputs:** an undistorted, rectified grayscale `Image` (SYSTEM_DESIGN section 3.3 message
  shape) — named concretely, this project's `scene_*.pgm` is exactly the kind of frame **01.01 (full
  GPU image pipeline)**'s rectified-stage output would hand off; a real system would insert this
  project's pipeline immediately after 01.01's `remap`/`resize` stages.
- **Downstream consumers:** a list of `Detection` records (tag ID, image corners, `(R, t)` pose in the
  camera frame — `src/kernels.cuh`) feeds, named concretely: **project 01.17 (extrinsic
  calibration)** — a fiducial at a known world position is the classical way to solve for a camera's
  pose relative to a robot or workspace; **domain 23 docking behaviors** (an AMR aligning to a charging
  dock or a pallet stand marked with a tag); and **domain 21 AR/teleoperation overlays** (rendering a
  virtual object locked to a physical tag's pose for an operator).
- **Rate / latency budget:** SYSTEM_DESIGN section 1.1's `camera -> perception` row: 30-60 Hz. Fiducial
  localization specifically wants the HIGH end of that range and beyond (embedded/Jetson-class GPUs
  commonly run AprilTag detection at 60-120+ Hz) because the two things it typically feeds — a
  docking control loop and an AR overlay — are both closed-loop and latency-sensitive: a stale tag pose
  shows up immediately as overlay jitter or docking overshoot, unlike a slower-changing map or
  classification result elsewhere in the stack.
- **Reference robot(s):** an **AMR performing precision docking** (SYSTEM_DESIGN section 2.1 — a tag
  at the charging dock or a pallet-pickup point gives centimeter-level terminal-approach localization
  no SLAM map alone reliably provides) and a **manipulator work cell** (SYSTEM_DESIGN section 2.2 — a
  tag on a fixture or part gives the arm a ground-truth pose for calibration verification and
  low-cost part localization when a full 6-DoF pose estimator is overkill).
- **In production:** a dedicated, heavily-optimized detector (AprilTag 3's C library, OpenCV's ArUco
  module, or NVIDIA Isaac ROS's hardware-accelerated AprilTag node — all named in "Prior art" below)
  running continuously on an embedded GPU/VPU, feeding a `geometry_msgs/PoseStamped` per detected tag
  into TF.
- **Owning team:** perception/calibration (SYSTEM_DESIGN section 5.1) — fiducial tooling is
  characteristically owned by whichever team is closest to bring-up and integration (it is as much a
  "get the robot working" tool as a shipped autonomy feature), often the same team that owns camera
  calibration and extrinsic bring-up procedures (PRACTICE.md section 4 expands).

## The algorithm in brief

- **Adaptive threshold** — a separable box filter computes each pixel's local mean brightness; a pixel
  is foreground if it reads meaningfully darker than its own local mean, tolerating the scene's
  illumination gradient where a single global threshold would fail. -> [THEORY.md](THEORY.md) "The
  problem", "The GPU mapping".
- **Connected-component labeling by iterative label propagation** — the same convergence-provable GPU
  CCL algorithm taught in 30.01 (cited and re-derived independently here), isolating each tag's dark
  border-plus-payload blob. -> THEORY.md "The algorithm".
- **Quad extraction** — a packed-64-bit-atomic "argmax" trick finds each component's 4 extreme
  corners in one pixel-parallel pass, then a candidate-parallel radial sub-pixel search refines each
  corner against the real image — an honestly weaker teaching stand-in for AprilTag's gradient-line-
  fitting corner refinement, and the README/THEORY sections say exactly why. -> THEORY.md "The
  algorithm", "Numerical considerations".
- **DLT homography** — 4 point correspondences give exactly 8 linear equations in 8 unknowns; one
  thread per candidate solves them by Gaussian elimination with partial pivoting, in double precision
  (the same small-dense-linear-solve-per-thread pattern 33.01 teaches for robot-arm Jacobians, applied
  here to a 2-D projective map). -> THEORY.md "The math", "The GPU mapping".
- **Perspective grid sampling + dictionary decode** — warp-sample the tag's 6x6 cell centers through
  the fitted homography, threshold each against the SAME local-mean field detection used, require the
  border ring to read (mostly) black, then try the sampled 4x4 payload against the dictionary at all 4
  in-plane rotations and accept the closest entry within the dictionary's measured correction capacity.
  -> THEORY.md "The math" (coding theory), "How we verify correctness".
- **Pose from homography** — classical `K^-1 * H` column-normalization decomposition; production
  systems refine this with IPPE (Infinitesimal Plane-based Pose Estimation), named not implemented.
  -> THEORY.md "The math", "Where this sits in the real world".

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/apriltag-aruco-gpu-detector-decoder-for-high.sln`](build/apriltag-aruco-gpu-detector-decoder-for-high.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/apriltag-aruco-gpu-detector-decoder-for-high.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none** — CUDA runtime + C++17 standard library only.
Data generation (`scripts/make_synthetic.py`) uses only the Python standard library (`math`, `os`,
`struct`, `hashlib`) — no NumPy, no image libraries, no `random` module (a hand-rolled xorshift32 PRNG
per the project brief).

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU and
against ground truth):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what the two
image artifacts show and how to read every line of output.

## Data

Three committed synthetic PGM scenes plus a home-grown 32-code dictionary, all reproducible bit-for-
bit from `python scripts/make_synthetic.py` (seed 42, xorshift32): `scene_main.pgm` (6 tags under
full perspective, with exact ground-truth corners and camera poses), `scene_distractor.pgm` (no tags
at all — a checkerboard and filled disks, for the false-positive gate), and `scene_robustness.pgm`
(4 tags with payload bits deliberately corrupted at and beyond the dictionary's correction capacity).
No public dataset applies — see [`data/README.md`](data/README.md) for the field-by-field
documentation, checksums, and why synthetic-with-exact-ground-truth beats a photographed dataset for
this project's verification needs (`scripts/download_data.ps1`/`.sh` are honest no-ops that say so).

## Expected output

Eleven stable lines — banner, `PROBLEM:`, `DATA:`, `VERIFY: PASS`, five `GATE <name>: PASS` lines,
`ARTIFACT:`, `RESULT: PASS` — checked as a subset diff by
[`demo/expected_output.txt`](demo/expected_output.txt). The measured numbers behind every verdict
print on unchecked `[info]`/`[time]` lines (they can shift a little across GPU architectures — see
THEORY.md "Numerical considerations"). On the reference machine (RTX 2080 SUPER, Release|x64):

- **VERIFY:** GPU matches CPU EXACTLY on the local mean (0.0 max diff — the box-sum is exact float32
  integer arithmetic), the foreground mask, and the CCL labels (0/172,800 mismatches, all 3 scenes);
  candidate statistics match exactly; refined corners agree within 0.15 px, homography entries within
  1.8 (mixed-scale units), pose rotation-matrix entries within 0.0074 — all comfortably inside their
  documented, measured-and-margined tolerances (THEORY.md "How we verify correctness" derives every
  number).
- **GATE detection:** 6/6 `scene_main` tags found, correct dictionary ID, zero extras.
- **GATE corner_accuracy:** max corner error over the 6 matched tags = 2.63 px (gate <= 3.5 px),
  measured against the renderer's own analytic corner positions.
- **GATE pose:** max rotation error 9.56 deg, max translation error 12.5% of tag size, over the 6
  matched tags, against the renderer's analytic camera pose — the honest cost of a homography-only
  pose estimate on a moderately-tilted 56-93 px tag without IPPE refinement (Limitations & honesty
  below; THEORY.md "Numerical considerations" derives the amplification chain).
- **GATE decode_robustness:** 4/4 `scene_robustness` tags behaved as designed — both 2-bit-flip
  (at capacity) tags decoded to their true ID, both 3-bit-flip (beyond capacity) tags were rejected.
- **GATE false_positive:** 0 accepted detections on the tag-free `scene_distractor` (2 of its
  components reach quad extraction — the two large filled disks — and are rejected there by the
  degenerate-all-black-payload safeguard).

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — start here: image/dictionary/candidate layouts, camera and
   dictionary-geometry constants, the packed-corner-key trick, and every struct the GPU path, the CPU
   oracle, and `main.cu` all share — one place, extensively commented.
2. [`src/main.cu`](src/main.cu) — the whole pipeline in plain sight: load 3 scenes + dictionary -> run
   GPU path -> run CPU path -> VERIFY -> five ground-truth GATES -> write artifacts. The single most
   interesting thing to look at: `best_shift_corner_error()`/`best_rotation_error_deg()`, which resolve
   the quad's unknown 90-degree corner-labeling ambiguity by trying all 4 alignments rather than
   hand-deriving a sign convention.
3. [`src/kernels.cu`](src/kernels.cu) — the ten GPU kernels; start with `box_sum_h_kernel`/
   `box_sum_v_kernel` (the separable-filter lesson), then `component_stats_accumulate_kernel` (the
   packed-atomic argmax trick), then `homography_solve_kernel` (the per-thread linear solve).
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the CPU oracle; note where it shares pure
   data-layout code with `kernels.cuh` (bit-rotation, corner-key packing) and where every algorithm is
   typed fresh (box filter, CCL — a genuinely different algorithm, union-find — DLT solve, decode, pose).
5. [`scripts/make_synthetic.py`](scripts/make_synthetic.py) — the dictionary generator (read
   `generate_dictionary()`'s docstring for the coding-theory argument) and the three scene renderers.
6. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, `find_data_file`/`resolve_out_dir`
   (copied, not shared — CLAUDE.md §4).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **AprilTag 3** (Wang & Olson, and the `apriltag` C library) — the reference implementation this
  project studies toward most directly: gradient-based edge detection, line fitting per tag edge, and
  a family of published dictionaries (16h5, 36h11, ...) built by the same minimum-Hamming-distance
  search this project's `generate_dictionary()` performs, over a much larger search with additional
  structure (BCH-code-derived families) this project's greedy random search does not attempt.
- **OpenCV ArUco module** (`cv2.aruco`) — the most widely deployed fiducial library; its 4x4/5x5/6x6/7x7
  dictionaries share this project's grid-with-border geometry; study its `detectMarkers` contour-based
  quad extraction as an alternative to this project's connected-component approach.
- **NVIDIA Isaac ROS AprilTag** — a hardware-accelerated (GPU/VPU) production AprilTag detector for
  robotics, the direct industrial descendant of the pixel-parallel-then-candidate-parallel architecture
  this project teaches, running at the 60+ Hz this README's "System context" names as the target rate.
- **IPPE (Collins & Bartoli, 2014), "Infinitesimal Plane-based Pose Estimation"** — the production fix
  for this project's pose-gate limitation: resolves the planar-homography pose ambiguity this project's
  simple column-normalization decomposition cannot, named throughout THEORY.md.
- **33.01 (batched small-matrix linalg)** — the general per-thread small-linear-solve pattern this
  project applies to homographies; read it for the GPU-mapping argument in more depth.
- **30.01 (agriculture, Milestone 1)** — this project's connected-component-labeling stage is the same
  algorithm, re-derived independently; read its `kernels.cu` for the full convergence proof.

## Exercises

1. **Plot the artifacts:** open `demo/out/detections_overlay.ppm` and `demo/out/decoded_grid_debug.ppm`
   next to `data/sample/scene_main.pgm` — find a tag whose corner or ID you can verify by eye against
   `data/sample/scene_main_ground_truth.csv`.
2. **Break the border tolerance:** tighten `kMaxBorderErrors` in `src/kernels.cuh` toward 0 (rebuild)
   and watch the `detection` gate start failing — this reproduces the exact failure mode this project's
   own build process hit and fixed (see `kMaxBorderErrors`'s doc comment) and shows why a strict
   all-cells-must-be-black rule is too fragile for this project's honestly-imperfect quad extraction.
3. **Push past +/-45 degrees:** edit `scripts/make_synthetic.py`'s `roll_deg` parameter for
   `make_main_scene` past 45 and regenerate the data — watch the `corner_accuracy` and/or `detection`
   gates degrade or fail, and connect what you see back to the extreme-corner-picking weakness named in
   `kernels.cuh`'s file header.
4. **Widen the correction capacity:** lower `try_from_distance` in `generate_dictionary()` to force a
   smaller minimum Hamming distance (hence a smaller `correction_capacity`), regenerate, and re-run
   `scene_robustness` — measure how much less bit-flip corruption the dictionary can now tolerate.
5. **Implement IPPE** (or just the orthogonal-Procrustes / SVD-based rotation refinement) in place of
   this project's Gram-Schmidt pose decomposition, and measure how much the pose gate's rotation error
   improves — this is the single most direct "close the gap to production" exercise this project offers.

## Limitations & honesty

- **Extreme-corner quad extraction is honestly weaker than production.** AprilTag/ArUco cluster
  gradient orientations and fit a LINE to each of the 4 tag edges independently, then intersect
  adjacent lines for each corner — robust to any single noisy pixel and to any in-plane rotation. This
  project's extreme-corner-then-radial-search method is a smaller, tractable teaching version: it
  degrades specifically near +/-45-degree in-plane rotations (where a flat EDGE, not a vertex, becomes
  the extremum — see `kernels.cuh`'s file header for the geometric argument), which is why the
  synthetic scenes deliberately keep roll away from that band. A production-grade corner refiner is a
  natural, scoped-out extension (Exercise 5's sibling).
- **The border-ring decode check is TOLERANT, not strict**, after this project's own build process
  measured that a strict all-20-cells-must-be-black rule rejected legitimately-detected tags whose
  homography fit was a few pixels imprecise at the corners (`kMaxBorderErrors`'s doc comment in
  `kernels.cuh` tells the full story, including the actual bug — a mis-sized corner-refinement search
  radius — this uncovered and fixed).
- **Pose-from-homography has real angular sensitivity to corner noise**, especially for a moderately
  tilted, moderately small (56-93 px) tag, without IPPE refinement — the pose gate's tolerance (13
  degrees rotation, 16% translation) reflects this project's ACTUAL measured accuracy with margin, not
  an aspirational target. THEORY.md "Numerical considerations" derives the amplification chain from a
  sub-pixel corner difference through the DLT solve to the pose decomposition.
- **The dictionary is independently generated, never a published bit table.** This project's 32 codes
  are NOT AprilTag 16h5 or ArUco's dictionaries; the geometry (6x6 grid, 16-bit payload) and the
  minimum-distance design PRINCIPLE match real families, and the achieved minimum distance (5) happens
  to match AprilTag 16h5's, but the actual codes are this project's own seeded search output.
- **Synthetic scenes only, deliberately.** No public fiducial dataset gives the EXACT analytic
  corner/pose ground truth this project's gates check against (`data/README.md` explains this choice).
  A learner should expect real camera images (motion blur, JPEG artifacts, non-planar tag mounting,
  specular highlights on laminated tags) to be harder than this project's synthetic noise+blur model.
- **Sim-validated only (CLAUDE.md §1):** this project computes poses from images; it commands no real
  hardware. If a docking or teleoperation loop were built on top of this output, the full real-hardware
  safety caveat (sim-validated only, staged testing ladder, independent E-stop) would apply at full
  strength — see [`PRACTICE.md`](PRACTICE.md) section 3.
