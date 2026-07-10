# 30.01 — Agriculture: fruit detection + 3D localization + ripeness; weed-vs-crop segmentation at frame rate; per-plant spray targeting; crop-row following; canopy volume from LiDAR; under-canopy navigation; yield mapping

**Difficulty:** ★ beginner · **Domain:** 30. Field & Industry-Specific Robotics

> Catalog bullet (source of truth, verbatim): `★ Agriculture: fruit detection + 3D localization + ripeness; weed-vs-crop segmentation at frame rate; per-plant spray targeting; crop-row following; canopy volume from LiDAR; under-canopy navigation; yield mapping`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

**This is one project bundling seven agriculture components (CLAUDE.md section 2).** The catalog
bullet names a whole orchard-robot perception/action stack in one line; per the repository contract
that stays ONE project, and the seven named components become milestones inside it:

| # | Milestone (as named in the catalog bullet) | Status |
|---|---|---|
| 1 | Fruit detection + 3-D localization + ripeness | **IMPLEMENTED** — this project, in full, below |
| 2 | Weed-vs-crop segmentation at frame rate | Documented only — THEORY.md "Where this sits in the real world" |
| 3 | Per-plant spray targeting | Documented only — consumes Milestone 1's 3-D fruit/plant positions |
| 4 | Crop-row following | Documented only — a navigation-stack (domain 23) sibling problem |
| 5 | Canopy volume from LiDAR | Documented only — a 3-D reconstruction (domains 02/05) sibling problem |
| 6 | Under-canopy navigation | Documented only — GNSS-denied localization under foliage |
| 7 | Yield mapping | Documented only, but **seeded** — `demo/out/fruit_map.csv` (per-fruit 3-D position + ripeness) is exactly the per-frame record a yield map aggregates across a robot's whole traverse |

**Milestone 1, implemented here:** detect fruit in a single synthetic RGB-D orchard frame, localize
each one in 3-D (camera frame), and estimate its ripeness — entirely with **classical GPU image
processing** (color-space classification, morphology, connected-component labeling, robust depth
estimation), no neural network. A learner who studies this project comes away understanding: why HSV
separates a fruit's color from its lighting; how to implement a real, convergence-provable GPU
connected-component labeler from scratch (not just call one); how a depth camera's pixel-space output
becomes a metric 3-D object list; and — just as important — *where a classical pipeline like this one
breaks* (touching same-colored objects at different depths, green-on-green ripeness, and the honest
gap between "hue" and "ripeness"). The demo runs the whole thing on the GPU **and** a plain-C++ CPU
oracle, checks them against each other, and checks BOTH against the synthetic scene's exact 3-D ground
truth — with the resulting numbers printed plainly, including where the pipeline's classical design
genuinely fails (two fruit pairs deliberately placed to overlap in the image get merged into one blob
each; see [Limitations & honesty](#limitations--honesty)).

## What this computes & why the GPU helps

Per frame (640x480 = 307,200 pixels): a chain of per-pixel MAPs (RGB->HSV, hue/sat/val gate), two
STENCILs (3x3 morphological erode/dilate), an iterative STENCIL+ATOMIC relaxation (connected-component
label propagation, ~60 sweeps to convergence), and a handful of ATOMIC-SCATTER passes that turn labeled
pixels into per-fruit statistics — roughly 2-3 million pixel-visits total, every one of them
independent of every other pixel *within its stage*.

- **Pattern:** the perception-pipeline pattern this repo's domain 01 flagship (01.02 stereo depth)
  also teaches — a MAP/STENCIL front end (one thread per pixel) feeding a small ATOMIC-SCATTER
  reduction (one thread per pixel, many-to-few writes) — here extended with a genuinely PARALLEL
  connected-component labeler (02.04 is this repo's LiDAR-point-cloud sibling of that exact problem).
- **Measured reality (RTX 2080 SUPER, Release, this project's committed scene):** the full seven-stage
  GPU pipeline runs in **~3-4 ms** total (front-end ~0.4-1.1 ms, CCL convergence ~2.5-3.3 ms over
  ~56-64 sweeps, component statistics ~0.15-0.19 ms) versus **~10-12 ms** for the single-core CPU
  oracle doing the identical work — a modest ~3x, and the demo says so honestly: at this frame size the
  CCL convergence loop's per-sweep host round-trip (checking "did anything change?") dominates the GPU
  time, not raw pixel throughput; THEORY.md "Numerical considerations" explains the trade and names the
  production fix.
- **Why this is the right teaching pattern despite the modest speed-up:** the *shape* — per-pixel GPU
  front end, atomics-based reduction to a handful of objects, host does the tiny final bookkeeping — is
  the exact shape every real-time perception front end in this repo's domains 01-03 uses; at production
  frame rates (1280x720+, multiple cameras, 30-60 Hz) the same pattern's absolute numbers matter far
  more, and the architecture does not change.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** the **perception layer's front end** — the first box after the sensor
  (SYSTEM_DESIGN section 1: `SENSORS -> PERCEPTION -> STATE ESTIMATION`). It turns a raw RGB-D frame
  into a small, structured object list; nothing here estimates the ROBOT's own state.
- **Upstream inputs:** an RGB-D camera's `Image` (SYSTEM_DESIGN section 3.6 message shape: `width`,
  `height`, `channels`, row-major `data`) — here two images, `rgb` (channels=3) and `depth` (channels=1,
  meters), sharing one pinhole intrinsic model (`fx,fy,cx,cy` in `src/kernels.cuh`).
- **Downstream consumers:** a per-frame list of `FruitDetection` records (3-D center, radius, ripeness
  — `src/kernels.cuh`), which is exactly the input shape a **harvesting manipulator's grasp planner**
  needs. Named concretely: **project 19.01 (parallel grasp candidate scoring)** would consume this
  project's 3-D fruit centers directly as its candidate-object list, the same way project README's
  "Chain B" (SYSTEM_DESIGN section 4.2) has `01.02 stereo depth -> 19.01 grasp scoring` — this project
  is a domain-30 sibling entry point into that identical chain shape. The documented Milestone 3
  (per-plant spray targeting) and Milestone 7 (yield mapping, seeded by `demo/out/fruit_map.csv`) are
  two more downstream consumers of the same record.
- **Rate / latency budget:** SYSTEM_DESIGN section 1.1's `camera -> perception` row: 30-60 Hz, <1 frame
  (16-33 ms) end-to-end. A harvesting robot's cycle-time argument (mirroring SYSTEM_DESIGN section 2.2's
  manipulator work cell): if a robotic arm needs ~1-3 s to reach, grasp, and retract per fruit, the
  PERCEPTION budget can be generously amortized across that cycle (a fresh detection pass once per
  arm cycle, not necessarily every camera frame) — this project's measured ~3-4 ms GPU pipeline has two
  full orders of magnitude of headroom against either budget, camera-rate or harvest-cycle-rate.
- **Reference robot(s):** SYSTEM_DESIGN's five reference robots do not name an agricultural machine
  explicitly, but the two nearest archetypes compose directly: a **harvesting platform** (the
  manipulator-work-cell archetype, SYSTEM_DESIGN section 2.2, wheeled/tracked instead of fixed-base,
  arm+gripper over a moving canopy) consuming this project's output as its grasp planner's input; and
  a **field AMR / spray robot** (the warehouse-AMR archetype, SYSTEM_DESIGN section 2.1, GNSS+row
  navigation instead of warehouse localization) consuming the documented Milestone 3/4 outputs.
- **In production:** deep-learning object detectors (YOLO-family, Faster-R-CNN, and orchard-specific
  fine-tunes) are the actual production reality for fruit detection today — trained on exactly the kind
  of labeled 2-D imagery this project's data section explains why it cannot commit. **This project's
  classical pipeline is the didactic baseline that teaches the underlying GEOMETRY** (camera model,
  connected components, robust depth estimation) a learned detector's output still has to pass through
  to become a 3-D grasp target — said plainly, not hidden (README "Prior art" expands this).
- **Owning team:** perception (SYSTEM_DESIGN section 5.1) in an agtech robotics company — this exact
  capability (color/vision-based fruit or weed detection) is commonly one of the FIRST hires an agtech
  startup makes, often before a controls/autonomy team exists, because it is the capability the
  business case depends on proving first (PRACTICE.md section 4 expands).

## The algorithm in brief

- **RGB -> HSV** — the standard max/min/chroma conversion (no trig, so GPU and CPU agree to sub-ULP
  precision); separates a fruit's ripening COLOR from its Lambertian SHADING. -> [THEORY.md](THEORY.md)
  "The math".
- **Fruit-likelihood mask** — a three-gate AND (hue, saturation, value), each threshold derived from
  the measured color separation in the committed scene, not guessed. -> THEORY.md "The algorithm".
- **Morphological opening** (erode then dilate, 3x3 8-connected) — removes small false-positive
  speckle without eating real fruit blobs; cross-references 20.01's GelSight contact-mask cleanup.
  -> THEORY.md "The algorithm".
- **Connected-component labeling by ITERATIVE LABEL PROPAGATION** — the ratified GPU teaching CCL
  algorithm: every foreground pixel relaxes its label toward the minimum linear index reachable via
  4-connected foreground neighbors, converging (proof sketch in `src/kernels.cuh`, full proof in
  THEORY.md) to a UNIQUE fixed point regardless of thread schedule — cross-references 02.04's
  union-find GPU clustering for the point-cloud sibling of this problem and its asymptotically better
  alternative. -> THEORY.md "The algorithm".
- **Per-component statistics via atomics** — pixel count, bounding box, centroid, mean hue, and a
  TWO-PASS ROBUST depth estimate (mean, then an inlier band derived from the depth sensor's own
  documented noise model) — all keyed by each pixel's canonical component label. -> THEORY.md "The GPU
  mapping".
- **3-D localization** — pinhole back-projection of the pixel centroid, PLUS a derived correction for
  the fact that a camera sees a sphere's near SURFACE, not its center (THEORY.md "The math" derives the
  exact `(2/3)*radius` offset and shows the measured error this correction removes).
- **Ripeness** — each component's mean hue mapped to a ripeness scalar, the exact inverse of the
  synthetic scene's forward color model — with an explicit, documented statement of what this metric
  does NOT capture about real fruit ripeness (firmness, sugar content, spectral signatures beyond
  visible hue). -> THEORY.md "Where this sits in the real world".

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/agriculture.sln`](build/agriculture.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/agriculture.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md section 5).

Optional dependencies and their fallbacks: **none** — CUDA runtime + C++17 standard library only. Data
generation (`scripts/make_synthetic.py`) uses only the Python standard library (`math`, `random`,
`hashlib`, `csv`, `argparse`) — no NumPy, no image libraries.

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU and
against ground truth):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) — including the
**two artifacts the demo writes** (a detection-ring overlay image and a "yield map" CSV).

## Data

The committed sample is a **rendered synthetic RGB-D orchard frame** with exact 3-D ground truth:
`data/sample/rgb.ppm` (640x480 color), `data/sample/depth.pgm` (640x480, 16-bit millimeter depth, with
realistic sensor noise baked in), and `data/sample/ground_truth.csv` (25 fruit, exact 3-D
center/radius/ripeness plus a measured occlusion statistic per fruit). Fully reproducible from
`python scripts/make_synthetic.py` (fixed seed 42). No public dataset applies here — see
[`data/README.md`](data/README.md) "Why synthetic, not a public fruit-detection dataset" for the
specific, load-bearing reason (public sets have 2-D labels, not the 3-D ground truth this project's
verification needs); `scripts/download_data.ps1`/`.sh` are honest no-ops that say so.

## Expected output

Ten stable lines — banner, `PROBLEM:`, `DATA:`, `VERIFY: PASS`, `DETECT: ... -> PASS`,
`LOCALIZE: ... -> PASS`, `RIPENESS: ... -> PASS`, two `ARTIFACT:` lines, `RESULT: PASS` — checked as a
subset diff by [`demo/expected_output.txt`](demo/expected_output.txt). The measured numbers behind each
gate print on the unchecked `[info]` line immediately above it (`src/main.cu`'s "NOTE on determinism"
comment explains why: two of the per-component statistics are built from GPU floating-point atomics,
which are not bit-reproducible across arbitrary hardware in principle, even though they have been
bit-stable across every run on the reference GPU — the repo's usual floating-point honesty, 08.01's
same convention). On the reference machine (RTX 2080 SUPER):

- **VERIFY:** GPU matches CPU EXACTLY on the mask (0/307,200 mismatches) and on connected-component
  labels (0/7,059 foreground-pixel mismatches, after both sides canonicalize to "label = minimum linear
  pixel index in the component" — THEORY.md proves why exact equality, not a tolerance, is the correct
  bar here); per-fruit statistics agree within relative 4-7e-7 (far inside the documented 1e-2 gate).
- **DETECT:** 20/24 detectable fruit found (rate 0.83, gate >=0.80), 2 unmatched detections (gate
  <=2) — both are the scene's two DESIGNED cross-depth merge cases, identified and explained by fruit
  ID in [Limitations & honesty](#limitations--honesty), not sensor noise or a pipeline bug.
- **LOCALIZE:** 3-D center error mean 1.8 mm, max 6.9 mm (gate <=15 mm); radius error mean 0.9 mm, max
  2.6 mm (gate <=6 mm) — after the pipeline's derived surface-to-center depth correction (THEORY.md
  "The math"), which measurably reduced this error from a mean of ~2.7 cm to ~1.8 mm.
- **RIPENESS:** Spearman rank correlation rho=0.998 over the 20 matched fruit (gate >=0.70).

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — start here: every layout, camera constant, threshold, and
   sentinel this project shares between the GPU, the CPU oracle, and `main.cu` — one place, extensively
   commented, including the full convergence argument for the CCL algorithm.
2. [`src/main.cu`](src/main.cu) — the whole pipeline in plain sight: load data -> run GPU path -> run
   CPU path -> VERIFY -> ground-truth GATES -> write artifacts. The single most interesting thing to
   look at: the `build_detections()` function's surface-to-center depth correction and its derivation
   comment.
3. [`src/kernels.cu`](src/kernels.cu) — the ten GPU kernels, each one small and single-concept; read the
   `ccl_propagate_sweep_kernel` comment for the convergence proof made concrete in code.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the CPU oracle; note where it is a LINE-BY-LINE
   twin of `kernels.cu` (HSV, mask, morphology) and where it is DELIBERATELY a different algorithm
   (union-find CCL) — the file header explains why that is the more, not less, rigorous choice here.
5. [`scripts/make_synthetic.py`](scripts/make_synthetic.py) — the scene generator; read this to
   understand exactly what "ground truth" means for a 3-D-rendered scene and how the two designed
   merge cases and the one fully-occluded fruit came to exist.
6. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers (copied, not shared — CLAUDE.md section 4).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md section
4.1).

- **YOLO-family detectors fine-tuned for orchard fruit** (and academic work like MinneApple, DeepFruits,
  Fuji-SfM) — the actual PRODUCTION approach to fruit detection today: a learned 2-D detector, often
  fused with stereo/depth for 3-D localization exactly as this project's back-projection stage does.
  This project's classical HSV+CCL pipeline is the didactic baseline that makes the GEOMETRY (camera
  model, connected components, robust depth) explicit before a black-box detector's output has to pass
  through the identical math.
- **OpenCV** (`cv2.connectedComponents`, `cv2.morphologyEx`, `cv2.cvtColor`) — the production library
  implementation of every classical stage here; study its (CPU) union-find CCL and compare with this
  project's GPU label-propagation choice.
- **nvblox / PCL** — where a detected object's 3-D position would typically be fused into a persistent
  map across many frames (this project is single-frame; Milestone 7 "yield mapping" is exactly that
  fusion problem, documented not implemented).
- **02.04 (Euclidean clustering via GPU union-find)** — the point-cloud sibling of this project's CCL
  stage; read it alongside `kernels.cu`'s label-propagation kernel to compare the two GPU CCL families
  directly.
- **19.01 (parallel grasp candidate scoring)** — the concrete downstream consumer named in "System
  context" above; its input shape is exactly this project's `FruitDetection` list.
- **Khoshelham & Elberink (2012), "Accuracy and Resolution of Kinect Depth Data"** — the real-world
  source for this project's quadratic-in-range depth noise model (`kDepthNoiseK` in `kernels.cuh`).

## Exercises

1. **Plot the artifact:** open `demo/out/detections.pgm` next to `data/sample/rgb.ppm` and find the two
   designed merge cases by eye (hint: look for a ring straddling two visually distinct fruit). Then open
   `demo/out/fruit_map.csv` and compute, by hand, which entry each merged ring corresponds to.
2. **Break a threshold:** loosen `kHueMaxDeg` in `src/kernels.cuh` toward 120 (rebuild) and watch the
   mask start admitting foliage pixels — count how many spurious components appear and why the
   `kMinComponentPixels` filter alone cannot always save you.
3. **Split the merges:** add a DEPTH-DISCONTINUITY check inside the connected-component stage (e.g.,
   only propagate a label to a neighbor if `|depth[p]-depth[q]|` is below a threshold) and re-run —
   verify the two designed merge cases become four separate, correctly-localized detections, and measure
   what (if anything) it costs elsewhere in the scene.
4. **8-connected CCL:** change the neighbor set in `ccl_propagate_sweep_kernel` (and its CPU union-find
   twin) from 4- to 8-connected and measure how sweep count and detection count change.
5. **On-device convergence check:** replace the per-sweep host round-trip (`cudaMemcpy` of the `changed`
   flag) with a fixed sweep count or a device-side early-exit trick (e.g., a persistent kernel with a
   grid-wide barrier), and measure the CCL stage's time before/after — THEORY.md "Numerical
   considerations" names this as the production fix for exactly the overhead this exercise removes.

## Limitations & honesty

- **Classical, not learned.** Production fruit detection is dominated by learned detectors (see "Prior
  art"); this project teaches the GEOMETRY underneath, not a competitive detection accuracy claim.
- **Ripeness is hue, and only hue.** Real fruit ripeness depends on firmness, sugar content, and
  spectral signatures a visible-light camera's hue channel cannot fully capture — THEORY.md
  "ripeness-vs-color honesty" states exactly what this metric can and cannot promise, which is why the
  demo's ripeness gate checks RANK correlation, not absolute agreement.
- **Green-on-green is out of scope.** A fully unripe fruit (hue ~120 degrees) is, by this pipeline's
  hue-only logic, indistinguishable from foliage (also hue ~100-140 degrees) — the committed scene is
  deliberately scoped to ripeness >= 0.35 to keep the benchmark honest about what a classical
  color-segmentation pipeline can and cannot do (see `scripts/make_synthetic.py`'s header).
- **Touching objects at different depths merge.** This project's CCL groups pixels by COLOR and 2-D
  CONNECTIVITY only, never by depth discontinuity. The committed scene contains two fruit pairs placed
  at noticeably different depths (about 1.2 m and 0.5 m apart) whose 2-D silhouettes nonetheless touch —
  both get reported as one blob each, honestly counted as 2 "false positives" and 4 missed individual
  fruit in the DETECT gate rather than hidden or excluded. Exercise 3 above sketches the fix (a
  depth-discontinuity split), left undone here to keep Milestone 1's scope to what the catalog names.
- **One-frame, camera-frame only.** No temporal fusion across frames, no transform to a world/map frame,
  no robot pose — Milestone 7 (yield mapping) and PRACTICE.md section 3 discuss what a real system adds.
- **Six other milestones documented, not implemented** — see the "Overview" table above; each is a
  natural, separately-scoped extension of this same code (THEORY.md "Where this sits in the real
  world" sketches all six).
- **Timings are teaching artifacts** — single-shot, one machine (RTX 2080 SUPER), never a benchmark
  claim (CLAUDE.md section 12).
- **Sim-validated only (CLAUDE.md section 1):** nothing in this project commands real hardware — it is
  a pure perception computation on a synthetic frame. If Milestone 3 (per-plant spray targeting) or any
  future actuation milestone were built on top of this output, the full real-hardware safety caveat
  (sim-validated only, staged testing ladder, independent E-stop) would apply at full strength — see
  [`PRACTICE.md`](PRACTICE.md) section 3.
