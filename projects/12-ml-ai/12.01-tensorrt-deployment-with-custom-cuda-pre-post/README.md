# 12.01 — TensorRT deployment with custom CUDA pre/post kernels: NMS, argmax decode, keypoint extraction

**Difficulty:** ★ beginner · **Domain:** 12. Machine Learning & AI for Robots

> Catalog bullet (source of truth, verbatim): `★ TensorRT deployment with custom CUDA pre/post kernels: NMS, argmax decode, keypoint extraction`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

This is the repository's first **heavy-SDK** project — the first one allowed to touch TensorRT
(CLAUDE.md §5) — and it takes that permission literally: TensorRT appears **only** behind an
opt-in compile flag that is **off by default**. What you get out of the box, with nothing but the
CUDA toolkit installed, is the project's real didactic heart: a complete, hand-rolled GPU inference
**deployment** pipeline around a tiny fixed "detector" — preprocessing (resize/normalize/transpose),
two conv+ReLU layers and a detection head, and the catalog bullet's three named post-processing
kernels, **argmax class decode**, **score-threshold + anchor-arithmetic box decode with NMS**, and
**keypoint extraction**, all in plain CUDA, all verified stage-by-stage against a CPU oracle, all
graded against a scene with three known objects. The network's weights are **synthetic and fixed**
(hand-designed, not trained — generated once, deterministically, by `scripts/make_synthetic.py`):
this project teaches how a model gets **deployed**, not how one gets **trained** (that is project
12.06's job). A second, fully documented, compile-ready-but-not-built TensorRT path shows exactly
how the same custom kernels bracket a real `nvinfer1::IExecutionContext::enqueueV3()` call once you
have the SDK — the point being that those custom kernels **do not change at all** when you swap the
inference core, because both cores speak the same device-buffer contract (`kernels.cuh`).

## What this computes & why the GPU helps

Per image: a ~12,300-element preprocess (resize+normalize+transpose), ~2,600 conv MACs across three
tiny layers, a 256-cell argmax/threshold/box-decode pass, an all-pairs IoU matrix over a few dozen
candidates, and a handful of keypoint window searches — six kernels, each a **map** over a small,
independent index space (one thread per output pixel / grid cell / candidate pair / detection).

- **Pattern:** batched map, six times over — the same "one thread per output element" idiom this
  whole repository teaches (33.01), applied to the specific shapes a deployed detector needs.
- **Where the real parallelism win lives at production scale:** NOT in this teaching image's tiny
  256-cell grid (which is intentionally small and fast so the demo stays inspectable — see
  "Limitations" below) but in the **all-pairs IoU matrix** (kernel 5) and the **stream-compaction**
  threshold/decode step (kernel 4, via `atomicAdd`) — both patterns that scale directly to a real
  detector's thousands of anchors across a multi-scale feature pyramid.
- **The one non-parallel step, on purpose:** greedy NMS suppression is *sequentially* dependent
  (survivor `i` decides survivor `i+1`'s fate) and is kept on the host, exactly like 08.01 keeps its
  softmin blend on the host — see THEORY.md "The GPU mapping" for the parallelism-tension discussion
  this project was explicitly asked to document.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** the **perception** layer, specifically the **inference/deployment boundary**
  inside it — the box between "a camera frame exists in device memory" and "a list of detections
  exists for the tracker/planner" (SYSTEM_DESIGN §1's `SENSORS -> PERCEPTION` arrow, domains 01/12).
- **Upstream inputs:** a camera frame (message shape: `sensor_msgs/Image`-like, HWC uint8 — the
  `Image` struct in SYSTEM_DESIGN §3.6) from a camera driver/ISP, already demosaiced/undistorted.
- **Downstream consumers:** a tracker or the planning stack's obstacle layer, consuming a detection
  list (class, box, keypoint per object — this project's `Detection` struct, kernels.cuh) — the
  natural next stop is multi-object tracking (04.x fusion/tracking) or a costmap update (23.x).
- **Rate / latency budget:** camera → perception runs at **30–60 Hz with <1 frame (16–33 ms)
  end-to-end latency** (SYSTEM_DESIGN §1.1's exact row); this demo's whole conv-chain-plus-decode
  pipeline measures well under a millisecond of GPU kernel time at this teaching image's tiny size
  (see the demo's `[time]` line) — real detector backbones at 640×640+ with a ResNet/YOLO-class
  network consume most of that 16–33 ms budget in the CONV LAYERS TensorRT replaces, which is
  exactly why this project separates "the inference core" (swappable) from "the pre/post kernels"
  (the fixed, portable part a deployment engineer actually writes by hand).
- **Reference robot(s):** the **warehouse AMR** (§2.1 — obstacle/tote detection feeding navigation)
  and the **autonomous-vehicle stack** (§2.5 — the VISION box feeding fusion/tracking); both name
  domain 01/12 detection pipelines explicitly in their block diagrams.
- **In production:** this exact shape — custom CUDA pre/post kernels wrapping a TensorRT engine —
  is how NVIDIA's own DeepStream pipelines are built, and is a common bespoke pattern anywhere a
  team needs NMS/keypoint logic TensorRT's built-in layers do not cover; Triton Inference Server and
  torch-tensorrt/ONNX Runtime-TensorRT-EP are the higher-level serving/export layers built on the
  same engine concept (README "Prior art").
- **Owning team:** **ML/perception deployment** — the team that takes a trained model (from an ML/
  data team, SYSTEM_DESIGN §5.1) and makes it run in the robot's real-time budget; adjacent to the
  perception team that trains/owns the model and the embedded team that owns the target compute.

## The algorithm in brief

- **Preprocessing** — bilinear resize (half-pixel-center convention) + per-channel normalize +
  HWC→CHW transpose, the standard first kernel of every deployed vision model. →
  [THEORY.md](THEORY.md) §The math.
- **The fixed "network"** — 2 conv+ReLU layers + a 1×1-conv detection head, hand-designed so the
  whole forward pass is checkable by hand; weights are synthetic and fixed, not trained. →
  THEORY §The algorithm.
- **Argmax class decode** — one thread per grid cell, argmax over the class-score channels. →
  THEORY §The GPU mapping.
- **Score threshold + box decode (anchor arithmetic)** — the standard `sigmoid`/`exp` anchor decode,
  with GPU stream compaction (`atomicAdd`) turning a sparse mask into a dense candidate array. →
  THEORY §The algorithm.
- **NMS** — a real GPU all-pairs IoU-matrix kernel, plus a documented, deliberately sequential
  greedy-suppression host scan (the parallelism-tension discussion). → THEORY §The GPU mapping.
- **Keypoint extraction** — a local-window argmax over the winning class's own score heatmap
  (CenterNet-style reuse), per surviving detection. → THEORY §The algorithm.
- **The optional TensorRT path** — an equivalent engine built from the same weights, wired to the
  SAME unmodified post-processing kernels via a shared device-buffer contract. → THEORY §Where this
  sits in the real world.

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md). **TensorRT is NOT required**
for the default build described here.

1. Open [`build/tensorrt-deployment-with-custom-cuda-pre-post.sln`](build/tensorrt-deployment-with-custom-cuda-pre-post.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/tensorrt-deployment-with-custom-cuda-pre-post.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

**Optional dependencies and their fallback (read this — it is this project's ratified design rule):**
`src/tensorrt_path.cpp` is compiled by BOTH configurations above, but its entire body sits behind
`#ifdef USE_TENSORRT`; with that macro undefined (the default), the file reduces to one stub
function and needs nothing beyond the C++ standard library. `demo/expected_output.txt` — the
checked, stable output contract — is generated by, and only ever exercises, this default (fallback)
path, on every build configuration.

### Enabling the optional TensorRT path (not exercised on the reference machine — no TensorRT SDK installed here; verify against your own install)

1. Install TensorRT (matching your CUDA 13.x toolkit; see NVIDIA's TensorRT download page — a
   version-gated SDK, not `pip install`-able the way plain CUDA is).
2. In Visual Studio, duplicate the `Release|x64` configuration into a new one (e.g.
   `Release-TensorRT|x64`, Configuration Manager → New...), then edit ONLY that new configuration:
   - `C/C++ → Preprocessor → Preprocessor Definitions`: add `USE_TENSORRT`.
   - `C/C++ → General → Additional Include Directories`: add your TensorRT `include/` folder.
   - `Linker → General → Additional Library Directories`: add your TensorRT `lib/` folder.
   - `Linker → Input → Additional Dependencies`: add `nvinfer.lib`.
3. Build the new configuration. `src/tensorrt_path.cpp`'s real body (engine build + inference,
   `#ifdef USE_TENSORRT`) compiles in; `main.cu` will additionally attempt the optional TensorRT
   demonstration (extra `[info] [trt]` lines — never part of the checked stable contract).
4. CMake users: see the commented `option(USE_TENSORRT ...)` block at the bottom of
   [`CMakeLists.txt`](CMakeLists.txt).

The exact same steps, spelled out with the reasoning behind each one, live in
[`THEORY.md`](THEORY.md) "Where this sits in the real world" and in
[`src/tensorrt_path.cpp`](src/tensorrt_path.cpp)'s file header.

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) — including
the **detection artifacts to view**.

## Data

Three tiny, synthetic, paired files under `data/sample/` (≈19.7 KiB total): `weights.bin` (the fixed
network's parameters, 460 bytes), `test_scene.ppm` (an 80×80 RGB test image, binary PPM), and
`ground_truth.csv` (the three placed objects' true class/position/size). All three are generated
byte-identically by `python scripts/make_synthetic.py` (fixed seed 42, no external entropy) — no
public dataset applies (a hand-designed, non-trained network has no use for one); `scripts/
download_data.ps1` is an honest no-op. Full format, units, and SHA-256 checksums: [`data/README.md`](data/README.md).

## Expected output

Eight stable lines — `[demo]`, `PROBLEM:`, `WEIGHTS:`, `SCENE:`, `VERIFY:`, `GROUNDTRUTH:`,
`ARTIFACT:`, `RESULT:` — checked as a subset diff by
[`demo/expected_output.txt`](demo/expected_output.txt). Three layers of verification, all measured
on the reference machine (RTX 2080 SUPER, sm_75) and all passing with wide margins:

1. **Stage-wise GPU-vs-CPU tensor agreement** (the §5 gate, applied at every stage, not just the
   end): preprocessed tensor, conv1, conv2, and head output each compared element-wise; measured
   worst relative deviation **2.25e-7** against a documented tolerance of **1e-3** (~4,400× headroom).
2. **Pipeline-stage COUNT agreement** (exact, not tolerance-based): GPU and CPU paths must produce
   the identical pre-NMS candidate count (measured: **21 = 21**) and post-NMS detection count
   (measured: **3 = 3**) — any mismatch is a real indexing/threshold bug, never FP32 rounding.
3. **The ground-truth gate**: all 3 known objects detected (measured: **3/3**), worst center error
   **2.40 px** against a 6.0 px bound, worst IoU **0.601** against a 0.30 minimum, **0** false
   positives against a bound of 1, and NMS reducing 21 pre-NMS candidates to 3 post-NMS detections —
   a **7.0×** reduction against a documented minimum of 3×.

None of these measured numbers appear in `expected_output.txt` itself (repo convention, CLAUDE.md
§12): only the PASS/FAIL verdicts are checked, because — while this project has NO run-time
randomness and the numbers above are expected to reproduce on any machine — the convention keeps the
contract robust to the one thing that legitimately can shift them: FP32 rounding differences across
GPU architectures (`sm_75` vs `sm_86` vs `sm_89`), which the wide margins above absorb harmlessly.

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — start here. The whole project's contract in one file: the
   fixed "network" architecture, the tensor layout convention, the `Detection` record, and the
   byte-exact weight-file format.
2. [`src/main.cu`](src/main.cu) — the orchestration: load data → run the GPU pipeline → run the CPU
   pipeline → stage-wise VERIFY → ground-truth gate → write artifacts. Read the file header first.
3. [`src/kernels.cu`](src/kernels.cu) — the six GPU kernels, in pipeline order; each kernel's header
   comment explains its thread mapping and — for kernels 4 and 5 — the atomics/parallelism-tension
   reasoning this project was specifically asked to teach.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the independent CPU twin of the entire
   pipeline (the correctness oracle `main.cu`'s VERIFY stage checks against).
5. [`src/tensorrt_path.cpp`](src/tensorrt_path.cpp) — the optional, documented, not-run-here
   TensorRT path; read its file header even if you never build it — the "what changes, what doesn't"
   framing is the project's punchline.
6. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers (copied, not shared — §4 rule).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **NVIDIA TensorRT** — the engine/builder/tactic-selection/precision-calibration system this
  project's optional path targets directly; the [official docs](https://docs.nvidia.com/deeplearning/tensorrt/)
  own the authoritative API reference this project's `tensorrt_path.cpp` approximates from memory.
- **NVIDIA Triton Inference Server** — the production serving layer built ON TOP of engines like the
  one this project builds: model versioning, batching, multi-framework backends, metrics — the
  "how do 50 robots share one inference server" problem, one level above this project's scope.
- **torch-tensorrt / ONNX Runtime (TensorRT execution provider)** — the higher-level export paths
  most teams actually use instead of hand-writing `INetworkDefinition` layers as `tensorrt_path.cpp`
  does: trace a PyTorch/ONNX model, let the tool build the engine, and attach custom pre/post exactly
  the way this project does — the "you rarely write addConvolutionNd by hand" honest caveat.
- **NVIDIA DeepStream** — the closest production analogue to this project's SHAPE (custom CUDA
  pre/post kernels, including NMS, bracketing a TensorRT engine) at video-analytics scale; its
  open-source plugin library ships a real `EfficientNMS` TensorRT plugin — compare against this
  project's "keep NMS as separate kernels, not a plugin" choice (THEORY.md explains the trade).
- **YOLO family (Redmon et al. and successors)** — the anchor-relative box-decode arithmetic
  (`sigmoid`/`exp` on `tx,ty,tw,th`) this project's threshold/box-decode kernel implements verbatim.
- **Zhou, Wang & Krähenbühl (2019), "Objects as Points" (CenterNet)** — the keypoint-heatmap
  design this project's keypoint-extraction kernel borrows (reusing a class-confidence map as a
  center-point heatmap, refined by a local-window peak search).

## Exercises

1. **Plot the artifact:** open `demo/out/detections.pgm` (any PGM-capable viewer, or `magick
   detections.pgm out.png`) and inspect `demo/out/detections.csv` side by side with
   `data/sample/ground_truth.csv` — confirm by eye that the boxes land where the rectangles are.
2. **Break a threshold:** lower `kScoreThreshold` in `kernels.cuh` toward 0.0, rebuild, and watch
   `[info] verify: candidate counts` explode — explain from the measured score grid (THEORY.md
   works it out) why the background never quite reaches 0 score and where the false positives
   start appearing.
3. **Loosen NMS:** raise `kNmsIouThreshold` toward 0.5 and observe the post-NMS detection count grow
   (duplicate boxes around the same object survive) — connect this to the IoU-vs-cell-spacing
   arithmetic THEORY.md derives.
4. **Add a third class:** extend `kNumClasses` to 3, add a "green" object type to
   `scripts/make_synthetic.py` and a third hand-designed weight row, and confirm the argmax-decode
   kernel needs zero code changes (it is already generic over `kNumClasses`).
5. **Flip on the TensorRT path:** if you have TensorRT installed, follow README "Build"'s opt-in
   steps and read what `[info] [trt]` lines the optional demonstration prints — compare the engine's
   reported serialized size against this project's own weight file (460 bytes) and explain the gap
   (THEORY.md "What TensorRT actually does").

## Limitations & honesty

- **The network is NOT trained — this project teaches DEPLOYMENT, not LEARNING.** Every weight in
  `data/sample/weights.bin` is hand-chosen so the pipeline's behavior is checkable by hand
  (THEORY.md works the arithmetic); nothing here was fit to data by gradient descent. Project 12.06
  (and the wider domain 12) is where this repository teaches training. Calling this a "detector" is
  accurate in the narrow, literal sense (it detects red/blue rectangles) and should not be read as a
  claim about real-world detection accuracy.
- **The scene is tiny and clean on purpose.** 80×80 pixels, a 16×16 grid, 2 classes, 3 objects — big
  enough to exercise every named kernel meaningfully, small enough that a learner can read every
  number in the `[info]` lines and a stranger could hand-verify the arithmetic (THEORY.md does,
  twice). Real detector backbones run at 640×640+ with multi-scale feature pyramids and tens of
  thousands of anchors; the KERNEL PATTERNS here (stream compaction, all-pairs IoU, local-window
  argmax) are exactly what production kernels use, just at production scale instead of teaching
  scale (THEORY.md "GPU mapping" makes this explicit at every kernel).
- **Box regression is present in the arithmetic but zero-weighted in the synthetic net.** The
  anchor-decode formula (`kernels.cu` KERNEL 4) is the real, general YOLO-style decode; this
  project's hand-designed weights happen to leave `tx,ty,tw,th` at zero, so every box resolves to
  the bare anchor. A trained head would learn a genuine per-cell refinement on top of the same
  formula — the arithmetic taught here is unaffected either way.
- **The keypoint heatmap is reused from the class-score map, not independently learned** (a
  CenterNet-style choice, stated honestly rather than dressed up — see README "Prior art"). The
  KERNEL (local-window argmax over a heatmap) is what this project teaches; a production network
  would typically learn a separate keypoint branch feeding the same kernel.
- **The TensorRT path is unverified on this machine.** No TensorRT SDK is installed on the reference
  machine (RTX 2080 SUPER, sm_75, CUDA 13.3); `src/tensorrt_path.cpp` is written to the best of
  documented knowledge of the TensorRT C++ API and is clearly labeled not-compiled, not-run here.
  Verify it against your installed TensorRT version before relying on it (README "Build").
- **Timings are teaching artifacts** — single-shot, one machine, and at this teaching image's tiny
  size the conv-chain kernels measure well under a millisecond; that is expected and not a
  benchmark claim (CLAUDE.md §12).
- **Not safety-relevant as shipped** — this project's output is a detection list, not an actuation
  command; nonetheless, everything in this repository is sim-validated/synthetic-only and not
  safety-certified (CLAUDE.md §1). Any real perception pipeline feeding a robot's planner/control
  stack inherits the full testing-ladder discipline in `PRACTICE.md` §3.
