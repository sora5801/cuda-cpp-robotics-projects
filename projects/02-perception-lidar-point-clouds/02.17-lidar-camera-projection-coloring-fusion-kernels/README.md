# 02.17 — LiDAR-camera projection/coloring fusion kernels

**Difficulty:** intermediate · **Domain:** 2. Perception — LiDAR & Point Clouds

> Catalog bullet (source of truth, verbatim): `LiDAR-camera projection/coloring fusion kernels`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

A LiDAR knows exactly *where* things are and nothing about what they look like; a camera knows exactly
what things look like and nothing about how far away they are. Fusing the two — painting camera color
onto LiDAR geometry, and painting LiDAR depth onto the camera's image plane — sounds like a lookup: for
each LiDAR point, project it into the image and read a pixel. It is not that simple, and this project is
built to show exactly why. Two directions, taught together on one synthetic scene: **Direction A (point
coloring)** projects every LiDAR return into the camera and bilinear-samples a color, and confronts the
central failure head-on — a point on a far surface can project into the *same pixel* a near surface
occupies, so naive coloring paints hidden points with the wrong (near surface's) color. This project
measures that failure directly (89.1% of a designed occlusion cohort colored wrong, naively) and fixes
it with an honest z-buffer occlusion check (down to 0.7% wrong, same cohort — both numbers measured on
the reference GPU, README "Expected output"). **Direction B (depth painting)** is the same z-buffer
projection pass read a different way: a sparse, RGBD-like depth image, no completion (01.18 is this
project's completion sibling — cited, not reimplemented). On top of both directions, a
**calibration-error sensitivity sweep** perturbs the camera-LiDAR extrinsic by documented rotation/
translation errors and turns "a wrong calibration paints the wrong color" into a measured curve, cross-
checked against 01.17's analytic pixel-displacement formula. All four kernels are checked against
independent CPU twins; every accuracy claim is graded against ground truth a ray-cast synthetic scene
computes independently of the pipeline itself (never seen by either code path).

## What this computes & why the GPU helps

Four small, independent-per-point computations, one shared geometric core:

- **Pattern:** overwhelmingly *map* — one thread per LiDAR point, computing its own projection, its own
  color sample, its own occlusion verdict, with zero interaction between points. The one exception is
  the z-buffer's projection pass, a **scatter**: many points can target the same output pixel, resolved
  by an `atomicMin`-on-encoded-depth race (01.18's trick, cited).
- **Why the GPU helps:** every stage is embarrassingly parallel across a few thousand LiDAR points —
  exactly the shape that wastes a CPU's time in loop overhead and a GPU's time in nothing. The four
  kernels measure well under a millisecond each on the reference GPU (`[time]` lines in a real run); the
  interesting cost in this project is not raw kernel time but how the SAME four kernels are *reused* —
  the calibration sensitivity sweep re-runs kernels 2+3 at six perturbed extrinsics with zero new code.
- **The two-pass shape, named:** scatter (z-buffer) then gather (color sampling) is a standard fusion
  taxonomy pattern — 01.18's kernels.cu names the same two verbs for its own projection stage; THEORY.md
  "The GPU mapping" discusses why this project splits them into four kernels instead of 01.18's two.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** Perception — a fusion/enrichment layer that sits between raw sensor ingestion and
  everything downstream that wants either colored geometry or dense-ish depth. It senses nothing new; it
  makes two already-sensed signals mutually legible.
- **Upstream inputs:** the LiDAR driver's raw scan (`PointCloud`-shaped, meters, LiDAR frame — this
  project's `LidarPointF`, identical shape to [`01.18`](../../01-perception-cameras-vision/01.18-depth-completion)/[`02.02`](../02.02-roi-crop-passthrough-organizedunorganized)); the
  camera's `Image` stream; and — the calibration-quality dependency this project makes *visible* — the
  extrinsic `T_camera_lidar` [`01.17`](../../01-perception-cameras-vision/01.17-camera-lidar-camera-camera-extrinsic-calibration)
  solves for. This project is where 01.17's recovery error and 01.18/02.02's fixed extrinsic stop being
  an abstract millimeter/degree number and become a visibly wrong-colored point cloud — the sensitivity
  sweep quantifies exactly that link. A moving platform additionally needs per-point motion deskew
  ([`02.08`](../02.08-per-point-motion-deskew-with-pose-interpolation)) before fusion — this project
  assumes a single, already-deskewed frame and states that assumption plainly (README "Limitations").
- **Downstream consumers:** anything that wants colored geometry or a fused depth product — semantic
  point painting (e.g. [`02.19`](../02.19-pointpillars-centerpoint-voxelization-scatter)'s PointPainting-
  style pipelines, named by lineage in THEORY.md), colorized mapping/visualization products, and dense
  completion (`01.18`, which shares this project's projection math and z-buffer trick and consumes the
  same kind of sparse depth this project's Direction B produces).
- **Rate / latency budget:** a per-frame fusion pass runs at camera rate (SYSTEM_DESIGN.md item 1's
  10–30 Hz perception band); this project's whole four-kernel pipeline measures well under a millisecond
  of GPU kernel time on ~3,400 points (`[time]` lines in a real run), comfortably inside that budget with
  headroom for the rest of the perception stack sharing the same tick.
- **Reference robot(s):** the **AV stack** (camera+LiDAR fusion feeding planning, SYSTEM_DESIGN.md item 2)
  and the **warehouse AMR** (colored/depth-fused obstacle perception) both use this component.
- **In production:** this teaching pipeline is the classical predecessor to learned point-painting
  networks (README "Prior art"); the ROS 2 ecosystem's `image_geometry`/`depth_image_proc` packages
  implement the same projection geometry as production-grade library calls.
- **Owning team:** perception / sensor-fusion (SYSTEM_DESIGN.md item 5) — the seam between the LiDAR/
  camera driver teams and the mapping/planning teams that consume colored or depth-fused output.

## The algorithm in brief

- **Rigid transform + pinhole projection** — `P_cam = R·P_lidar + t` then the pinhole formula, the SAME
  convention 01.17/01.18/02.02 use (cited, not re-derived). → [THEORY.md](THEORY.md) §The math.
- **Z-buffer visibility pass** — scatter every point's encoded depth into its pixel via `atomicMin`
  (01.18's trick, cited); nearest wins. This pass alone IS Direction B's product. → THEORY §The GPU
  mapping.
- **Bilinear color sampling** — four-tap gather at the point's continuous (sub-pixel) projected
  coordinate (01.01 lineage). → THEORY §Numerical considerations.
- **Occlusion depth-consistency check** — a point is accepted as "the pixel's own visible surface" only
  if its own depth is close to the NEAREST z-buffer evidence in a small pixel window around it (not just
  the exact pixel — a sparse LiDAR scan's own angular gaps make an exact-pixel-only check miss most real
  occlusions, measured in THEORY.md). → THEORY §The math, §How we verify correctness.
- **Calibration-error sensitivity sweep** — perturb `T_camera_lidar` by 0.2/0.5/1.0° (rotation) and
  1/2/5 cm (translation), re-run the shared projection+sampling kernels, and measure how many sampled
  colors cross a designed color boundary — cross-checked against 01.17's `fx·Δθ` (rotation) and
  `fx·Δt/R` (translation) pixel-displacement formulas. → THEORY §The math.

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/lidar-camera-projection-coloring-fusion-kernels.sln`](build/lidar-camera-projection-coloring-fusion-kernels.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/lidar-camera-projection-coloring-fusion-kernels.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none** — CUDA toolkit runtime + C++17 standard library only.
`src/` hand-rolls its own PPM/CSV I/O rather than pulling in an image library (CLAUDE.md §5's "no black
boxes" spirit).

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at, including the artifact table (the colored-cloud "money shot" and the occlusion cohort
before/after the fix).

## Data

The committed sample (`data/sample/rgb.ppm`, `lidar_points.csv`) is **synthetic** (CLAUDE.md §8 default):
a ray-cast scene rendered from two physically separate sensor origins (a pinhole camera and a 2-D-
scanning LiDAR), including a foreground occluder deliberately positioned so the LiDAR — mounted higher
than the camera — sees background points the camera cannot (the occlusion cohort). Every LiDAR return
carries EVALUATION-ONLY ground truth (its true surface color and a camera-visibility flag, computed by
an independent second ray cast) that the pipeline itself never reads. Regenerate with
`python scripts/make_synthetic.py --seed 42`. Full field documentation, the scene's occlusion geometry
worked out in numbers, and SHA-256 checksums are in [`data/README.md`](data/README.md).

## Expected output

The demo prints four `VERIFY:` lines (GPU-vs-CPU agreement per kernel — projection+z-buffer, project-
points, bilinear-sample, occlusion-check — each within a documented tolerance in `src/main.cu`) and five
`GATE:` lines, every threshold **measured, then margined** (run once on the reference GPU, an RTX 2080
SUPER, the actual number recorded, the threshold set below/above it with stated headroom — 08.01's
technique):

| Gate | What it checks | Measured | Threshold |
|---|---|---|---|
| `frustum_accounting` | in-frustum + out-of-frustum + colored + filtered counts are exactly consistent | exact | bookkeeping, exact |
| `coloring_accuracy` | ground-truth-visible points colored within tol 0.12 (normalized) of their true color | 76.9% (2483/3230) | ≥ 70% |
| `occlusion_correctness` | the occluded cohort's wrong-color rate, WITHOUT vs. WITH the occlusion check | WITHOUT 89.1%, WITH 0.7% (n=138) | WITHOUT ≥ 80%, WITH ≤ 5% |
| `depth_image_fidelity` | painted depth matches an independently re-derived per-pixel minimum | exact (0.0 m) | tol 1e-4 m |
| `sensitivity_curve` | flip-fraction rises with \|perturbation\|; smallest level's pixel displacement matches 01.17's analytic formula | rotation 1.09×, translation 1.26× of predicted | within 4× |

`RESULT: PASS` requires every `VERIFY:` and every `GATE:` to pass. The canonical stable lines live in
[`demo/expected_output.txt`](demo/expected_output.txt); measured numbers (exact counts, ratios, per-level
sensitivity rows) print on unchecked `[info]`/`[time]` lines because they can vary by GPU architecture
even when every verdict does not.

Artifacts written to `demo/out/`: `cloud_topview.ppm`, `cloud_sideview.ppm` (the colored-cloud money
shot), `occlusion_cohort_naive.ppm`/`occlusion_cohort_checked.ppm` (the designed failure, before/after
the fix), `painted_depth.pgm`, `sensitivity_curve.csv`, `gates_metrics.csv` — see
[`demo/README.md`](demo/README.md) for what each one shows.

## Code tour

A guided reading order through `src/`:

1. [`src/main.cu`](src/main.cu) — entry point: loads data, runs the VERIFY stage (GPU vs CPU per
   kernel), computes and gates every evaluation metric against ground truth, runs the calibration
   sensitivity sweep, writes artifacts.
2. [`src/kernels.cuh`](src/kernels.cuh) — the single-sourced contract: camera/extrinsic constants
   (identical to 01.17/01.18/02.02), the `Rigid3`/`LidarPointF` layouts, the occlusion band/window
   constants, and every kernel/launcher/CPU-twin declaration. Read this before `kernels.cu`.
3. [`src/kernels.cu`](src/kernels.cu) — the four GPU kernels (the heart of the project): start at
   `project_zbuffer_kernel` (the scatter + atomic trick, Direction B's product), then
   `project_points_kernel` (the shared geometric core), then `sample_bilinear_kernel` (Direction A,
   naive), then `check_occlusion_kernel` (Direction A, the fix — the most interesting kernel here: read
   its header comment on why a single exact pixel is not enough).
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the independent CPU oracle for all four kernels;
   its file header states this project's twin-independence ruling.
5. [`scripts/make_synthetic.py`](scripts/make_synthetic.py) — the ray-cast scene generator, including the
   occlusion geometry derivation and the independent per-point visibility ground truth.
6. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, and `paths.h`'s data/artifact resolution.

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **PointPainting** (Vora, Lang, Helou & Beijbom, 2020) — the production lineage this project's
  Direction A points toward: paint LiDAR points with per-pixel *semantic* labels from a camera network
  (not raw RGB) before feeding a 3-D detector; this project teaches the geometric plumbing (projection,
  occlusion) that any such painting scheme depends on getting right first.
- **ROS 2's `image_geometry` / `depth_image_proc`** — the production library calls for exactly this
  project's pinhole projection and depth-image conventions; study their `PinholeCameraModel` for the
  hardened, distortion-aware version of this project's ideal-pinhole formula.
- **nvblox** (NVIDIA) — a production GPU dense-mapping stack; its colored-TSDF path consumes
  point-coloring output shaped like this project's.
- **KITTI's calibration/raw data format** — the real-world convention for shipping a fixed
  `T_camera_lidar` alongside camera and LiDAR streams, the same shape this project's `kTCameraLidar`
  takes (cited by 01.18, reused here).
- **[`01.18`](../../01-perception-cameras-vision/01.18-depth-completion) — Depth completion** — this
  project's completion sibling: takes Direction B's sparse depth image and densifies it; this project
  deliberately stops at the sparse product and does not reimplement completion.
- **[`01.17`](../../01-perception-cameras-vision/01.17-camera-lidar-camera-camera-extrinsic-calibration) —
  extrinsic calibration** — solves for the `T_camera_lidar` this project consumes as a fixed constant,
  and derives the rotation/translation pixel-error formulas the sensitivity sweep's analytic consistency
  gate checks against.

## Exercises

1. **Widen the sensitivity sweep.** Add a fourth, larger perturbation level to both the rotation and
   translation sweeps (e.g. 2° and 10 cm) and extend `sensitivity_curve.csv`'s monotonicity check —
   at what magnitude does the flip fraction start to saturate, and why (hint: most of the scene's points
   are far from any true color boundary at all, README "Data").
2. **Tune the occlusion window.** `kernels.cuh`'s `kOcclusionWindowRadiusPx` is a measured-then-chosen
   constant (currently 2, a 5×5 window). Halve it and double it, rerun, and record how
   `occlusion_correctness`'s WITH-check wrong-color rate trades off — this is the central tension every
   sparse z-buffer occlusion check faces (too narrow misses evidence, too wide over-filters near real
   depth edges).
3. **Plot the sensitivity curve.** `demo/out/sensitivity_curve.csv` → flip_fraction vs. level, for both
   sweeps. Compare the shapes: why does the translation sweep's flip fraction stay near zero until the
   5 cm level while rotation rises steadily from the smallest level (THEORY.md "The math" derives the
   range-dependence difference)?
4. **True k-nearest-neighbor occlusion check.** Replace the fixed-radius window search in
   `check_occlusion_kernel` with a real spatial structure (following 02.05's precedent) and measure the
   speed/accuracy trade-off against the current windowed approximation at larger point counts.
5. **Add per-point motion deskew.** This project assumes one static frame (README "Limitations"). Chain
   02.08's deskew ahead of this project's projection stage and re-measure `coloring_accuracy` on a scene
   with platform motion — the sensor-timing-honesty story 02.08's own README names by name.

## Limitations & honesty

- **Flat-shaded synthetic surfaces.** Every scene object is unshaded (no Lambertian gradient) so a
  per-point "true color" is an exact, unambiguous scalar rather than a continuously-varying value — a
  deliberate simplification that keeps the coloring-accuracy gate meaningful without needing a much
  looser tolerance. Real surfaces are shaded; THEORY.md names the tolerance consequence.
- **The occlusion check is a window search, not a scene-aware oracle.** It can only find evidence a
  LiDAR point actually left nearby; a sparse enough scan can still fool it in either direction (miss a
  real occlusion, or over-filter a genuinely visible point near a real depth edge) — Exercise 2 explores
  the trade-off directly, and `occlusion_correctness`'s measured 0.7% WITH-check wrong-color rate (not
  0%) is left visible on purpose rather than tuned away.
- **No per-point motion deskew.** The pipeline assumes one static frame; a moving platform needs 02.08's
  deskew ahead of this project's projection stage — named, not implemented (Exercise 5).
- **Ideal pinhole, no lens distortion** — same simplification 01.16/01.17/01.18 make; a real camera's
  radial/tangential distortion would need correcting before this project's projection formula applies.
- **The extrinsic is a fixed, hand-derived constant, not solved by this project** — see
  [`01.17`](../../01-perception-cameras-vision/01.17-camera-lidar-camera-camera-extrinsic-calibration)
  for the calibration this project assumes has already happened (and whose ERRORS this project's
  sensitivity sweep is built to characterize).
- **Small, synthetic scene.** 160×120 resolution, ~3,400 LiDAR points, four flat-colored objects — chosen
  so the CPU twin and every gate run in milliseconds and the committed sample stays kilobytes. Real
  automotive LiDAR/camera pairs run at 1000×+ resolution with tens of thousands of points per frame.
- **The evaluation gates' numeric thresholds are set from measured runs on one GPU** (an RTX 2080 SUPER)
  with documented margins (README "Expected output", THEORY.md "How we verify correctness") — not
  universal claims about this method's accuracy on arbitrary scenes.
- **Sim-validated only, not safety-certified.** This project computes colored point clouds and depth
  images for study purposes; it makes no claim of production-grade accuracy and must never be treated as
  a certified perception component for a real robot (CLAUDE.md §1, §8). If output from a pipeline like
  this ever feeds a planner, controller, or any system commanding real hardware, that integration is the
  owner's decision and responsibility.
