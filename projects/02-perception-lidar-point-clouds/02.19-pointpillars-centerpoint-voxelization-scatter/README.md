# 02.19 — PointPillars/CenterPoint voxelization + scatter kernels feeding TensorRT

**Difficulty:** intermediate · **Domain:** 2. Perception — LiDAR & Point Clouds

> Catalog bullet (source of truth, verbatim): `PointPillars/CenterPoint voxelization + scatter kernels feeding TensorRT`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

This project builds the **preprocessing bridge** between raw LiDAR points and a learned 3-D detection
network: it bins points into bird's-eye-view "pillars" (PointPillars) or coarse 3-D "voxels"
(CenterPoint-style), caps and augments each pillar's points into the classic 9-D feature vector, runs a
tiny **fixed-weight** ("PFN-lite") feature network, **scatters** the result into a dense tensor a
convolutional head can read, and closes the loop with a small **hand-designed** (never trained) 3×3-conv
detection head so the whole pipeline runs end to end **without needing TensorRT or any learned weights
installed**. TensorRT itself is *not* built here — this project produces exactly the tensors a real
TensorRT engine (project [`12.01`](../../12-ml-ai/12.01-tensorrt-deployment-with-custom-cuda-pre-post/README.md))
would consume, and documents that hand-off precisely (`[info] trt_handoff`), following 12.01's own
"optional-SDK, plain-CUDA-fallback-is-the-real-demo" precedent (README §Build below).

**What is implemented vs. documented-only** (CLAUDE.md §2 — this catalog bullet bundles several ideas):
implemented — pillar/voxel binning (two methods: atomic and sorted), the 9-D feature augmentation, the
fixed PFN-lite, the scatter kernel, a hand-designed 2-layer conv head, peak extraction + NMS, and a
pillar-vs-voxel memory/time comparison. Documented-only — the TensorRT engine build itself (12.01 is
the sibling project that does that), a *learned* PFN/head (THEORY.md explains what training would add),
and box-orientation regression (this project's synthetic cars are axis-aligned; see Limitations).

A learner who studies this project will understand: why learned 3-D detectors need a bounded, dense,
regular input tensor when LiDAR itself is sparse and unbounded; the exact 9-D PointPillars feature
vector and why each term exists; why max-pooling over a pillar's points makes point order irrelevant —
*except* when a point cap truncates the set, which is where GPU atomic nondeterminism becomes an ML
reproducibility bug hiding inside "just a preprocessing kernel"; and the sparse-to-dense "scatter" trade
that motivates real sparse convolution libraries.

> **Template placeholder notice — resolved.** The scaffold's SAXPY placeholder has been fully replaced;
> every file below is this project's real implementation.

## What this computes & why the GPU helps

The computation is a chain of GPU-shaped patterns, not one kernel:

- **Binning** (`compute_pillar_keys_kernel` / `compute_voxel_keys_kernel`) — a pure **map**: one thread
  computes one point's destination cell, independent of every other point.
- **Sort + segment** (`launch_sort_and_compact`, Thrust `stable_sort_by_key` + a boundary-mark **map** +
  `reduce`/`copy_if` **compaction**) — turns "which points share a cell" into "a contiguous run in a
  sorted array," the same 02.01-lineage machinery this project reuses almost verbatim.
- **Atomic scatter-claim** (`atomic_bin_kernel`) — a **scatter with atomics**: many threads race to claim
  slots in the same destination via `atomicAdd`, the canonical GPU counting-allocator pattern.
- **Per-pillar reduction** (`pfn_stats_kernel`, `pfn_lite_kernel`) — a small, bounded **reduction**
  (mean, max-pool) per output element, one thread per pillar (the reduction is over ≤32 points, too
  small to parallelize further profitably — see THEORY.md).
- **Scatter** (`scatter_kernel`) — the catalog's second named kernel: a sparse list **scattered** into a
  dense tensor, one thread per occupied pillar.
- **Stencil convolution** (`conv3x3_kernel`) — the classic **stencil** pattern: each output pixel reads a
  small, fixed neighborhood of inputs.

The bottleneck this project teaches is not raw FLOPs (the arithmetic here is tiny) — it is **irregular
memory access and unpredictable output cardinality**: N unordered points must become a fixed-shape
tensor, and *how many* pillars end up occupied is not known until runtime. Every kernel above is the
GPU-idiomatic answer to one piece of that irregularity.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** Perception — specifically the **preprocessing bridge** between classical point-cloud
  perception (SYSTEM_DESIGN.md §1's "POINT-CLOUD PERCEPTION [02 →]" box) and a **learned** 3-D object
  detector's forward pass. It sits *after* this domain's cleaning stages and *before* the ML/inference
  layer.
- **Upstream inputs:** a cleaned `PointCloud` (x,y,z,intensity) — exactly the output shape of this
  domain's earlier projects. Named upstream siblings: [`02.01`](../02.01-voxel-grid-downsampling-with-gpu-spatial-hashing/README.md)
  voxel-grid downsampling (this project reuses its sort/hash binning lineage directly),
  [`02.03`](../02.03-ground-segmentation/README.md) ground segmentation,
  [`02.13`](../02.13-dynamic-point-removal/README.md)/[`02.14`](../02.14-moving-object-segmentation-from-sequential-scans/README.md)
  dynamic-object filtering, and [`02.18`](../02.18-weather-filtering/README.md) weather filtering — a
  fully cleaned scan is exactly what this project's `points.bin` layout represents.
- **Downstream consumers:** the scattered `[C,H,W]` canvas and the `[P_occ,32,9]` pillar-feature tensor
  are what a real detector's backbone/head consumes; in production that head is a **trained** CNN
  deployed via TensorRT — [`12.01`](../../12-ml-ai/12.01-tensorrt-deployment-with-custom-cuda-pre-post/README.md)
  is that exact deployment sibling: this project produces precisely the tensors 12.01's custom
  pre/post-processing kernels would hand to an inference engine (`[info] trt_handoff` prints the exact
  shapes). The detector's output (3-D boxes) would then feed multi-object tracking and prediction
  (domain 04) — outside this project's scope.
- **Rate / latency budget:** LiDAR arrives at **10–20 Hz** (SYSTEM_DESIGN.md §1.1); a real pipeline's
  entire preprocessing stage (binning + features + PFN + scatter) must be a SMALL FRACTION of that
  50–100 ms scan period, leaving the rest for the network forward pass and downstream tracking. This
  run's measured `sort_and_compact` kernel time is well under 1 ms on 7,530 points (`[time]` line) —
  comfortably inside budget at this scale; THEORY.md discusses how the real ~100k-point scale changes
  the arithmetic.
- **Reference robot(s):** the **autonomous-vehicle stack** (SYSTEM_DESIGN.md §2.5 — LiDAR-based 3-D
  detection is core to AV perception) and the **warehouse AMR** (§2.1 — a 3-D LiDAR detector is a common
  upgrade path for AMRs sharing space with forklifts and pallets, beyond the simpler 2D-LiDAR obstacle
  detection the base AMR stack uses).
- **In production:** the fixed PFN-lite and hand-designed head would be replaced by a **trained**
  PointPillars/CenterPoint network (learned weights, real convolutions, non-maximum suppression tuned on
  a labeled dataset), deployed via TensorRT with FP16/INT8 calibration (12.01, 12.03). The voxelization
  and scatter *kernels themselves* — this project's actual subject — carry over largely unchanged into
  that production stack; NVIDIA's own CUDA-PointPillars reference implementation is architected the
  same way (THEORY.md "Where this sits in the real world").
- **Owning team:** this straddles **Perception** (owns the point-cloud input contract, SYSTEM_DESIGN.md
  §5.1) and **ML/data** (owns the learned network and its TensorRT deployment) — in practice a
  "perception ML infrastructure" role that owns exactly this preprocessing boundary, so the tensor
  contract between the two teams (PRACTICE.md §3) is a real, versioned interface.

## The algorithm in brief

- **Pillar/voxel key computation** — `pillar_key_of`/`voxel_key_of` (THEORY.md "The math").
- **Two binning methods** — atomic per-pillar slot claim (Method A) vs. sort + fixed-order truncation
  (Method B), reusing [`02.01`](../02.01-voxel-grid-downsampling-with-gpu-spatial-hashing/README.md)'s
  Method A/B lineage (THEORY.md "The algorithm").
- **9-D per-point feature augmentation** — raw (x,y,z,intensity) + offset-from-pillar-mean +
  offset-from-pillar-center (THEORY.md "The math").
- **PFN-lite** — a fixed linear layer + ReLU + max-pool per pillar, the permutation-invariance argument
  (THEORY.md "The math" — the Deep Sets connection).
- **Scatter** — sparse pillar list → dense `[C,H,W]` canvas (THEORY.md "The GPU mapping").
- **Hand-designed 2-layer conv head** — a spatial-coherence smoothing pass, an occupancy gate, a
  sharpening pass, then peak extraction + NMS (THEORY.md "The algorithm").
- **CenterPoint-style voxel comparison** — the identical sort/compact machinery, a 3-D key instead of a
  2-D one (THEORY.md "The problem" — the z-collapse trade).

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md). **TensorRT is NOT required** for
this project — the catalog's "feeding TensorRT" premise is realized by producing the *tensors* TensorRT
would ingest (`[info] trt_handoff`), following project
[`12.01`](../../12-ml-ai/12.01-tensorrt-deployment-with-custom-cuda-pre-post/README.md)'s
optional-SDK/plain-CUDA-fallback precedent — there is no optional TensorRT code path here at all,
because unlike 12.01 this project's *subject* is the pre/post kernels themselves, not the engine.

1. Open [`build/pointpillars-centerpoint-voxelization-scatter.sln`](build/pointpillars-centerpoint-voxelization-scatter.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/pointpillars-centerpoint-voxelization-scatter.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

**Dependencies:** the CUDA Toolkit + C++17 standard library only (CLAUDE.md §5 default). `kernels.cu`
uses Thrust (`stable_sort_by_key`/`reduce`/`copy_if`) — header-only, part of the CUDA Toolkit, so no
extra library or install step; the `.vcxproj` carries the `/Zc:preprocessor /Zc:__cplusplus` flags CCCL
needs under MSVC (see that file's comments, and project
[`02.01`](../02.01-voxel-grid-downsampling-with-gpu-spatial-hashing/README.md), the first project in this
repo to hit and document this requirement).

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at.

## Data

A synthetic 80×80 m bird's-eye-view scene: 6 hollow-box "cars," a sparse ground plane, isolated clutter,
and one deliberately pillar-cap-overflowing cluster (60 points into a 32-point cap). Fully synthetic by
design — real datasets cannot supply this project's two load-bearing requirements: exact ground-truth
object centers, and a precisely-placed cap-overflow fixture. Generated by `scripts/make_synthetic.py`
(seed 42, xorshift32, stdlib only). Full provenance, checksums, and field documentation in
[`data/README.md`](data/README.md).

## Expected output

The demo runs the full pipeline once (the deterministic, sorted binning path), verifies every stage
against an independent CPU reference (`VERIFY(keys/binning/pfn/scatter/head/peaks)`), and runs four
additional independent gates (`GATE cap_truncation/layout_roundtrip/feature_semantics/
detection_closure`). All are expected to `PASS` — see [`demo/expected_output.txt`](demo/expected_output.txt)
for the exact stable lines and [`demo/README.md`](demo/README.md) for what each one means. Tolerances:
bit-exact where the math guarantees it (keys, the sorted binning's kept-point sets and feature means —
both paths sum identical points in identical order, so IEEE-754 float addition rounds identically;
scatter, a pure copy), and 1e-4 where compiler FMA-fusion differences between nvcc and cl.exe can shift
the last bit or two (the PFN's linear layer, the conv head) — THEORY.md "Numerical considerations"
explains the exact boundary. Artifacts: `demo/out/occupancy.pgm`, `demo/out/heatmap.pgm` (peaks marked),
`demo/out/feature_stats.csv`, `demo/out/gates_metrics.csv`.

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — **read this first.** The contract: every constant, every
   layout decision, the two-binning-methods determinism story, and the "same machinery, different key
   function" pillar-vs-voxel connection.
2. [`src/kernels.cu`](src/kernels.cu) — the GPU kernels, in pipeline order: keys → Method A/B binning →
   features → PFN-lite → scatter/gather → conv/gate → peak extraction.
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the independent CPU oracle, one function per
   pipeline stage (the same granularity as the GPU path, so `main.cu` can `VERIFY` stage by stage).
4. [`src/main.cu`](src/main.cu) — orchestration: data loading, the cap_truncation determinism
   experiment, the production pipeline run, every VERIFY/GATE, and the artifacts.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, and `paths.h`'s data/artifact resolution.

## Prior art & further reading

- **Lang et al., "PointPillars: Fast Encoders for Object Detection from Point Clouds" (CVPR 2019)** —
  the paper this project's pillarization + 9-D feature vector + PFN come directly from. Study the real
  paper for the *learned* PFN architecture (a full PointNet-style MLP, not this project's fixed linear
  layer) and the real detection head (SSD-style anchors, not this project's hand-designed conv).
- **Yin, Zhou, Krähenbühl, "Center-based 3D Object Detection and Tracking" (CVPR 2021, CenterPoint)** —
  the sparse 3-D voxel backbone and center-heatmap detection head this project's `pillar_vs_voxel`
  comparison and toy peak-extraction head are simplified stand-ins for.
- **NVIDIA CUDA-PointPillars** (open-source reference implementation) — the real production kernel
  suite for exactly this pipeline's voxelization/scatter stages, TensorRT-integrated; study its actual
  memory layouts and kernel fusion choices once this project's simplified version makes sense.
- **spconv / MinkowskiEngine** — the real sparse-convolution libraries the `sparsity_economics` `[info]`
  line motivates; study how they avoid ever materializing the dense canvas this project scatters into.
- **Project [`12.01`](../../12-ml-ai/12.01-tensorrt-deployment-with-custom-cuda-pre-post/README.md)** —
  this project's deployment sibling: how the tensors produced here would actually reach a TensorRT
  engine, with custom CUDA pre/post-processing.
- **Project [`02.01`](../02.01-voxel-grid-downsampling-with-gpu-spatial-hashing/README.md)** — the
  sort/hash binning machinery this project's Method A/B lineage is built on; read it first if the
  determinism story here is unfamiliar.

## Exercises

1. Change `kMaxPointsPerPillar` (32 → 16) and re-run — watch `sparsity_economics` and the cap_truncation
   gate's numbers change; explain why the cap-stress pillar's story gets *more* dramatic at a smaller cap.
2. Replace the fixed PFN-lite weights with an *actually trained* tiny linear classifier (hand-derive
   weights that separate car-pillars from ground-pillars using `feature_stats.csv`'s ranges) and see how
   much smaller the hand-designed head's job becomes.
3. Add a third, coarser CenterPoint z-band split (`kNumZBins = 4`) and extend the `pillar_vs_voxel`
   comparison — does the memory cost scale exactly linearly with `kNumZBins`? Measure it.
4. Implement on-device cuRAND-based input-order shuffling (instead of this project's host-side
   Fisher-Yates) for the cap_truncation gate's shuffled-order runs, and compare the measured variance.
5. Extend `augment_features_kernel`/`_cpu` with a 10th feature (range `sqrt(x²+y²)`, motivated by
   LiDAR's 1/r² density falloff — see [`02.01`](../02.01-voxel-grid-downsampling-with-gpu-spatial-hashing/THEORY.md))
   and re-run the whole pipeline; does `feature_semantics` still need updating, and why/why not?

## Limitations & honesty

- **No learned weights anywhere.** The PFN-lite's linear layer and the detection head's conv weights are
  fixed/hand-designed, never trained (kernels.cuh/THEORY.md say so at every relevant site). This is a
  **layout and kernel-correctness project**, not a detection-accuracy project — the detection_closure
  gate proves the *pipeline* is right, not that the *model* is good.
- **No box orientation.** Every synthetic car is axis-aligned; this project does not implement or teach
  yaw/orientation regression, which real PointPillars/CenterPoint heads do.
- **No fixed `P_max` padding.** Production PointPillars pads/truncates the pillar-feature tensor to one
  fixed `P_max` (e.g. 12,000) per frame so a TensorRT engine's input shape never changes; this demo uses
  the scene's actual occupied-pillar count directly (documented in `[info] trt_handoff`) — a legitimate
  simplification at this project's tiny synthetic scale, but not what a real deployed engine does.
- **The atomic-binning nondeterminism demonstration uses input-order shuffling**, not raw same-order
  GPU-scheduler nondeterminism (which this project also measures and reports, honestly, as sometimes
  zero on a quiet GPU) — see `kernels.cuh`'s file header for why shuffling is the reliably-reproducible,
  and arguably more realistic, way to demonstrate the same bug class.
- **Synthetic data only** — every point in `data/sample/` is analytically generated (CLAUDE.md §8),
  labeled synthetic everywhere it appears; no recording of any real sensor or real space.
- **Sim-validated only.** This project's output never commands real hardware and carries no motion-safety
  claim of any kind; nothing here is a certified implementation of any detection standard.
