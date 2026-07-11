# 01.18 — Depth completion: sparse LiDAR + RGB → dense depth

**Difficulty:** intermediate · **Domain:** 1. Perception — Cameras & Vision

> Catalog bullet (source of truth, verbatim): `Depth completion: sparse LiDAR + RGB → dense depth`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

A spinning LiDAR sees the world through a handful of beams — this project's committed sample covers
only **~4.4%** of the camera's pixels. Depth completion turns that sparse, exact point cloud plus a
dense RGB image into a dense, approximate depth map: every pixel gets a depth estimate, guided by the
prior that RGB edges usually coincide with depth edges. This project implements the full classical
(non-learned) pipeline — LiDAR-to-camera projection with an honest occlusion z-buffer, an
inverse-distance-weighted (IDW) baseline, and an edge-aware anisotropic-diffusion densifier — on a
synthetic scene **purpose-built with two adversarial test surfaces**: a flat, high-contrast
checkerboard (where the "RGB edge ⇒ depth edge" prior is a *false alarm*) and a pair of identically
colored surfaces at different depths (where the prior *fails to fire*). The demo doesn't just report
accuracy; it measures, gates, and reports both failure modes explicitly, alongside the accuracy win the
method is built to deliver. All four pipeline stages run on the GPU and are checked against independent
CPU references; all evaluation is against the synthetic scene's *exact* ray-cast dense truth.

## What this computes & why the GPU helps

Four computations, four GPU patterns:

- **Projection + z-buffer** — a **scatter**: each of ~1000 LiDAR points computes its own camera pixel
  and races (via `atomicMin`) with every other point that lands on the same pixel to keep the nearest.
- **IDW baseline** — a bounded **stencil/search**: every pixel independently scans a 33×33 window for
  valid samples — embarrassingly parallel, and exactly the kind of `O(W·H·R²)` cost a CPU single core
  pays for in milliseconds per call and a GPU absorbs in microseconds.
- **Conductance** — a **map**: one thread per pixel reads two forward neighbors.
- **Anisotropic diffusion** — a **stencil iterated 1400 times** (ping-pong buffered): every pixel reads
  its 4 neighbors' *previous* value every iteration — a Jacobi update, the natural GPU mapping for a
  PDE solve, and the same "many independent small updates, many times" shape 07.09's jump-flooding and
  31.01's Hamilton–Jacobi level-set marching use (cited in `THEORY.md`).

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** Perception — a fusion/enrichment step that sits between raw sensor ingestion and
  everything downstream that wants dense geometry. It does not sense anything new; it makes an existing
  sparse sensor's output usable where a dense signal is assumed.
- **Upstream inputs:** the LiDAR driver's raw scan (a `PointCloud`-shaped message, meters, LiDAR frame —
  domain **02, perception-lidar-point-clouds**, e.g. [`02.17`](../../02-perception-lidar-point-clouds/02.17-lidar-camera-projection-coloring-fusion-kernels) which shares this project's
  projection math) and the camera's `Image` stream; the extrinsic `T_camera_lidar` this project
  *consumes* as a fixed constant is exactly what
  [`01.17`](../01.17-camera-lidar-camera-camera-extrinsic-calibration) *solves for* — cite it by name,
  this project is its direct downstream consumer.
- **Downstream consumers:** anything that wants a dense `Image`-shaped depth map instead of a sparse
  `PointCloud` — obstacle/traversability costmaps (e.g.
  [`14.02`](../../14-locomotion-wheeled/14.02-traversability-costmaps-fusing-semantics)), grasp
  candidate scoring from RGB-D (e.g.
  [`19.01`](../../19-manipulation-grasping/19.01-parallel-grasp-candidate-scoring)), and dense mapping
  front-ends such as TSDF fusion (e.g.
  [`05.01`](../../05-slam-mapping-localization/05.01-tsdf-fusion-marching-cubes-mesh-extraction)).
- **Rate / latency budget:** a spinning automotive/AMR LiDAR turns over at **10–20 Hz**
  (SYSTEM_DESIGN.md item 1); this project's whole GPU pipeline — projection, conductance, 1400 diffusion
  iterations, and IDW — measures **single-digit-to-low-teens milliseconds** on an RTX 2080 SUPER (see
  `[time]` lines in a real run), comfortably inside that budget with headroom for the rest of the
  perception stack sharing the same tick.
- **Reference robot(s):** the **AV stack** (camera+LiDAR fusion feeding planning) and the **warehouse
  AMR** (dense depth feeding local costmaps) both use this component (SYSTEM_DESIGN.md item 2).
- **In production:** this teaching pipeline is the classical predecessor to the learned sparse-to-dense
  networks that dominate production stacks today (README "Prior art" below); many shipping systems still
  use a guided/joint-bilateral or diffusion-style refinement stage even downstream of a learned network.
- **Owning team:** perception / sensor-fusion (SYSTEM_DESIGN.md item 5) — sits at the seam between the
  LiDAR/camera driver teams and the mapping/planning teams that consume dense geometry.

## The algorithm in brief

- **LiDAR-to-camera projection with z-buffer occlusion handling** — rigid transform + pinhole
  projection + nearest-wins occlusion resolution via an `atomicMin`-on-encoded-float trick
  (`THEORY.md` "The GPU mapping").
- **Inverse-distance-weighted (IDW) interpolation** — the RGB-blind densification baseline
  (`THEORY.md` "The algorithm").
- **Edge-aware anisotropic diffusion** (Perona–Malik conductance, forward-Euler PDE solve, sparse
  samples as Dirichlet boundary conditions) — the project's main method (`THEORY.md` "The math",
  "The algorithm").
- **Region-conditioned evaluation gates** — overall accuracy, edge quality, the texture-trap and
  camo-edge honesty checks, and input fidelity, all measured against exact synthetic ground truth
  (`THEORY.md` "How we verify correctness").

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/depth-completion.sln`](build/depth-completion.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/depth-completion.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: none — CUDA toolkit libraries (runtime only) + C++17
standard library. `src/` hand-rolls its own PPM/PGM/CSV I/O rather than pulling in an image library
(CLAUDE.md §5's "no black boxes" spirit: every byte this project touches should be readable by a
learner).

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at.

## Data

The committed sample (`data/sample/rgb.ppm`, `truth_depth.bin`, `lidar_points.csv`) is **synthetic**
(CLAUDE.md §8 default): a small ray-cast 3-D scene rendered from two different sensor origins (a
pinhole camera and a 16-beam spinning LiDAR), including two purpose-built adversarial test surfaces
(a texture-trap checkerboard, a low-contrast real depth edge). Regenerate with
`python scripts/make_synthetic.py --seed 42 --az-count 280`. Full field documentation, the exact
measured LiDAR density at three subsampling levels, and SHA-256 checksums are in
[`data/README.md`](data/README.md).

## Expected output

The demo prints five `VERIFY:` lines (GPU-vs-CPU agreement per pipeline stage — projection+z-buffer,
conductance, diffusion, IDW — each within a documented tolerance in `src/main.cu`), a `STABILITY:` line
(the diffusion PDE's CFL-style stability bound, also enforced at compile time by a `static_assert` in
`src/kernels.cuh`), and five `GATE:` lines (overall accuracy, edge quality, texture-trap honesty,
camo-edge honesty, input fidelity — `THEORY.md` "How we verify correctness" documents every threshold
and the measured number it was set from). `RESULT: PASS` requires every VERIFY and every GATE to pass.
The canonical stable lines live in [`demo/expected_output.txt`](demo/expected_output.txt); measured
numbers (MAE/RMSE, timings, region pixel counts) print on unchecked `[info]`/`[time]` lines because
they can vary by GPU architecture even when every verdict does not.

Artifacts written to `demo/out/`: `rgb.ppm`, `sparse_depth_vis.pgm`, `completed_guided.pgm`,
`completed_idw.pgm`, `truth_depth.pgm`, `error_guided.pgm`, `error_idw.pgm`, `gates_metrics.csv` — see
[`demo/README.md`](demo/README.md) for the visualization convention.

## Code tour

A guided reading order through `src/`:

1. [`src/main.cu`](src/main.cu) — entry point: loads data, runs the VERIFY stage (GPU vs CPU per
   stage), computes and gates the evaluation metrics against ground truth, writes artifacts.
2. [`src/kernels.cuh`](src/kernels.cuh) — the single-sourced contract: image/camera constants, the
   `Rigid3`/`LidarPointF` data layouts, every kernel/launcher/CPU-twin declaration, and the diffusion
   stability `static_assert`. Read this before `kernels.cu`.
3. [`src/kernels.cu`](src/kernels.cu) — the four GPU kernels (the heart of the project): start at
   `project_zbuffer_kernel` (the scatter + atomic trick), then `compute_conductance_kernel`, then
   `diffusion_step_kernel` + `launch_diffusion` (the ping-pong iteration loop), then `idw_kernel`.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the independent CPU oracle for all four stages;
   its file header states this project's twin-independence ruling.
5. [`scripts/make_synthetic.py`](scripts/make_synthetic.py) — the ray-cast scene generator, including
   the texture-trap and camo-edge test surfaces.
6. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, and `paths.h`'s data/artifact resolution.

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **KITTI Depth Completion benchmark** (Uhrig et al., 2017) — the dataset and evaluation protocol
  (MAE/RMSE on withheld LiDAR points) that made depth completion a standard perception task; this
  project's `overall_accuracy` gate mirrors its metric choice at a teaching scale.
- **Sparsity Invariant CNNs** (Uhrig et al., 2017) and the **sparse-to-dense** line of work (Ma & Karaman
  2018; Cheng et al.'s *Convolutional Spatial Propagation Network*, 2018–2020) — the learned-method era
  this project's classical pipeline is the predecessor of; CSPN in particular learns an affinity (like
  this project's hand-set conductance) end to end.
- **Perona & Malik, "Scale-space and edge detection using anisotropic diffusion"** (1990) — the PDE this
  project's main method implements; the original edge-stopping-function paper.
- **Guided Image Filtering** (He, Sun, Tang, 2010/2013) — the production-grade alternative to
  hand-tuned anisotropic diffusion for edge-aware upsampling; a linear-time, closed-form filter used
  throughout real depth-completion and matting pipelines (named as the documented alternative in
  `THEORY.md` "The algorithm").
- **PCL / OpenCV** — production point-cloud projection and interpolation primitives; study their
  z-buffering and inpainting utilities for the production-hardened version of this project's
  `project_zbuffer_kernel`/`idw_kernel`.
- **nvblox** (NVIDIA) — a production GPU dense-mapping stack that consumes depth exactly like this
  project produces it, feeding TSDF fusion (cited above as a downstream consumer).

## Exercises

1. **Mean-fill vs. IDW-seed.** The diffusion PDE currently seeds unknown pixels with the mean of the
   valid sparse samples (`THEORY.md` "Numerical considerations" explains why). Re-seed with the IDW
   result instead and re-measure the `texture_trap` and `overall_accuracy` gates — does bootstrapping
   from a spatially-aware prior change the story?
2. **Tune `kConductanceK`.** `src/kernels.cuh` documents the current value; halve it and double it,
   rerun, and record how the `edge_quality` vs. `texture_trap` gates trade off against each other —
   this is the central hyperparameter tension anisotropic diffusion always faces.
3. **Guided image filtering.** Implement He et al.'s guided filter as a fifth method and add it to the
   evaluation gates — a linear-time alternative to the iterative PDE that many production stacks prefer.
4. **True k-nearest-neighbor IDW.** Replace the fixed-radius window search with a real k-NN search
   (e.g. a spatial hash or small k-d tree, following 02.05's precedent) and measure the speed/accuracy
   trade-off against the current windowed approximation.
5. **Vary the LiDAR beam count.** `scripts/make_synthetic.py`'s `N_BEAMS` mirrors a 16-beam automotive
   LiDAR; regenerate at 32 or 64 beams (denser elevation) and extend the density sweep — how much of
   the horizon-region error (visible in `error_guided.pgm`'s bottom band) does denser elevation fix
   versus denser azimuth?

## Limitations & honesty

- **Classical, not learned.** This is the pre-deep-learning depth completion pipeline (README "Prior
  art"). Modern production systems (and the KITTI leaderboard) are dominated by learned sparse-to-dense
  networks that this project's THEORY.md names but does not implement.
- **Small, synthetic scene.** 160×120 resolution and a handful of primitive objects, chosen so the CPU
  twin and every gate run in milliseconds and the committed sample stays kilobytes. Real automotive
  LiDAR/camera pairs run at 1000×+ resolution with 30–128+ beams.
- **IDW is a fixed-radius window search, not true k-NN** (`THEORY.md` "The algorithm" names this
  simplification explicitly) — a teaching simplification, not a claim about production IDW.
- **The extrinsic is a fixed, hand-derived constant**, not solved by this project — see
  [`01.17`](../01.17-camera-lidar-camera-camera-extrinsic-calibration) for the calibration this project
  assumes has already happened.
- **The evaluation gates' numeric thresholds are set from measured runs on one GPU** (an RTX 2080
  SUPER) with a documented margin (THEORY.md "How we verify correctness") — they are not universal
  claims about anisotropic diffusion's accuracy on arbitrary scenes.
- **Sim-validated only, not safety-certified.** This project computes depth maps for study purposes; it
  makes no claim of production-grade accuracy and must never be treated as a certified perception
  component for a real robot (CLAUDE.md §1, §8). If depth from a pipeline like this ever feeds a
  planner or controller on real hardware, that integration is the owner's decision and responsibility.
