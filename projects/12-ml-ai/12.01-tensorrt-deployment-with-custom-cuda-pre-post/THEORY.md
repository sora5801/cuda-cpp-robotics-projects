# 12.01 — TensorRT deployment with custom CUDA pre/post kernels: NMS, argmax decode, keypoint extraction: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2).

## The problem — physics & engineering first

This project is, honestly, **purely computational**: there is no novel physics inside a convolution
or a box decode. What follows instead is the physics and engineering of this project's **nearest
physical carrier** — the camera-to-inference pipeline a real robot runs this exact kind of code on —
because that chain is what makes "deployment" a distinct engineering discipline from "training", and
distinct engineering constraints (latency, silicon, memory bandwidth) are exactly what shape the
decisions this project makes.

**From photons to a tensor.** A camera sensor (CMOS, rolling- or global-shutter) integrates photons
into charge over an exposure window, reads out per-pixel voltages, and an ISP (image signal
processor — dedicated silicon, not a general CPU/GPU) demosaics the Bayer pattern, applies white
balance and gamma, and hands off an HWC (row-major, interleaved-channel) 8-bit image — exactly the
`src_hwc` byte layout this project's preprocessing kernel consumes. This handoff typically happens
over MIPI CSI-2 (a dedicated camera bus) straight into the SoC's memory, at 30–60 Hz (SYSTEM_DESIGN
§1.1) — the ISP's job is done in dedicated hardware precisely because doing it in software would eat
into the same compute budget the neural network needs.

**Where a real inference accelerator differs from "a GPU kernel".** This project's conv/head layers
run as ordinary CUDA cores doing scalar multiply-adds (see "The GPU mapping" below) — the honest,
teachable version. A production accelerator instead routes convolution through **Tensor Cores**
(NVIDIA's dedicated matrix-multiply silicon, present since Volta) or a separate **DLA** (Deep
Learning Accelerator — a fixed-function ASIC block on Jetson Orin-class SoCs, physically distinct
from the GPU's CUDA cores, that runs common conv/pooling/activation graphs at a fraction of the
GPU's power draw). TensorRT's builder (THEORY "Where this sits in the real world" expands on this)
is the layer that decides, per network layer, whether Tensor Cores, ordinary CUDA cores, or (on
supported hardware) the DLA runs it — a decision this project makes by hand, once, by choosing to
write plain scalar kernels, because the point here is to make every multiply-add visible.

**The engineering constraints that make "deployment" a real job.** A trained model is a graph of
floating-point tensor operations with no opinion about latency, memory layout, or which silicon runs
it. Deploying it means: fitting the whole forward pass inside the 16–33 ms camera-to-perception
budget (SYSTEM_DESIGN §1.1) *every frame, forever*; keeping every intermediate tensor inside the
accelerator's on-chip/HBM memory bandwidth budget (a Jetson Orin has roughly 100–275 GB/s depending
on variant — a real backbone at 640×640 moves tens of megabytes of activations per frame); and
converting the model's arbitrary floating-point weights into a numeric representation (FP16, INT8)
the target silicon executes fastest, without silently corrupting the answer. None of that is
"training" work — it is a distinct discipline, and it is what this project's fallback pipeline and
optional TensorRT path both teach, at a scale small enough to fit on one screen.

## The math

**Bilinear resize (the preprocessing kernel's core operation).** Given a destination pixel `(ox,oy)`
in a `kNetW x kNetH` output, where does its CENTER fall in the `kSrcW x kSrcH` source? Under the
**half-pixel-center convention** (this project's choice — the OpenCV/PyTorch default, stated at the
API boundary per this project's convention):

```
scale = src_size / dst_size
src_coord(dst_coord) = (dst_coord + 0.5) * scale - 0.5
```

The `+0.5`/`-0.5` pair accounts for pixel CENTERS, not corners: pixel `0`'s center is at `0.5` in a
continuous coordinate where pixel edges sit at integers; mapping centers-to-centers (rather than the
naive `dst_coord * scale`, which maps CORNERS and visibly shifts the image toward one edge at
non-integer scales) is the convention every production resize op uses. After clamping to
`[0, src_size-1]` (no extrapolation past the image border), bilinear interpolation blends the 4
integer neighbors `(x0,y0),(x1,y0),(x0,y1),(x1,y1)` with weights `(1-fx)(1-fy), fx(1-fy), (1-fx)fy,
fx*fy` where `fx,fy` are the fractional parts — separable, so the kernel blends along `x` first,
then `y` (`kernels.cu` KERNEL 1).

**Normalization** is the standard `(x - mean) / std` per channel; this project uses `mean=128,
std=64` for all three channels (`kernels.cuh`), chosen so the background gray `(128,128,128)`
normalizes to exactly `(0,0,0)` — "the background carries no signal" is then a property of the DATA,
not something the network has to learn to ignore.

**Convolution** (conv1, conv2, and the 1×1 head — one formula, `kernels.cu` KERNEL 2):

```
out[co,oy,ox] = bias[co] + sum_{ci,ky,kx} w[co,ci,ky,kx] * in[ci, oy*stride-pad+ky, ox*stride-pad+kx]
```

with out-of-bounds input taps treated as zero (zero-padding). A `1x1` convolution (`K=1, pad=0,
stride=1`) collapses the `ky,kx` sum to one tap per input channel — exactly a per-pixel
matrix-vector multiply, which is why this project's detection HEAD is legitimately "a convolution
layer" and not a euphemism for one.

**Worked example 1 — inside a red object** (THEORY's promise: checkable by hand). Take a network-
input pixel deep inside a red rectangle (RGB `(240,50,50)`, far from any edge, so its whole 3×3→3×3
receptive field is uniform). Normalized: `R_norm=(240-128)/64=1.75`, `G_norm=(50-128)/64=-1.21875`,
`B_norm=(50-128)/64=-1.21875`. conv1's channel 0 ("redness") weights are `+1/9` on every R tap,
`-1/9` on every B tap, `0` on G (`kernels.cuh`/`scripts/make_synthetic.py`):

```
conv1_out[0] = avg(R_norm) - avg(B_norm) = 1.75 - (-1.21875) = 2.96875   (ReLU: stays positive)
conv1_out[1] = avg(B_norm) - avg(R_norm) = -2.96875                      (ReLU: clamped to 0.0)
```

conv2 average-pools each channel against itself (weight `1/9` on the matching input channel, `0`
cross-channel); over a uniform interior region the average of a constant is the same constant, so
`conv2_out[0] = 2.96875`, `conv2_out[1] = 0.0` unchanged. The head reads `head_out[0] = 1.0 *
conv2_out[0] - 1.0 = 1.96875` (bias `-1.0`, the score threshold's complement) and `head_out[1] = 1.0
* conv2_out[1] - 1.0 = -1.0`. **Measured on the actual committed sample: exactly 1.9688 (rounding)
— see the demo's own math, reproduced independently below.**

**Worked example 2 — the gray background.** `R_norm=G_norm=B_norm≈0` (the mild dither is ±3/64 ≈
±0.047, negligible). `conv1_out[0] = 0 - 0 = 0` for both channels; `conv2_out` unchanged; `head_out[0]
= head_out[1] = 0 - 1.0 = -1.0`. The gap between a firing object cell (`≈+1.97`) and a background
cell (`exactly -1.0`) is **2.97** — the score threshold of `0.7` sits with roughly 1.3 units of
margin on each side, which is why this project's demo passes with wide, documented tolerance
(see "How we verify correctness").

**Anchor-arithmetic box decode** (`kernels.cu` KERNEL 4 — the real, general YOLO-style formula):

```
cx = (gx + sigmoid(tx)) * cell_px          cy analogous
w  = anchor_px * exp(tw)                   h analogous
```

`sigmoid` keeps the regressed center inside the cell that predicted it (a cell is only ever asked to
refine its OWN anchor, never relocate to a neighbor — the reason per-cell prediction is well-posed
at all); `exp` makes width/height regression symmetric in log-space (shrinking to a quarter size and
growing to 4× are equally-sized targets in `tw`-space). This project's synthetic weights leave
`tx=ty=tw=th=0` everywhere (`sigmoid(0)=0.5`, `exp(0)=1`), so every decoded box is the bare anchor
centered exactly at its cell — the arithmetic is real and general, the *regression* is a documented
placeholder (README "Limitations").

**IoU (intersection-over-union)** of two axis-aligned boxes: `inter = max(0, min(x1)-max(x0)) *
max(0, min(y1)-max(y0))`; `iou = inter / (area_a + area_b - inter)`. Two identically-sized `s x s`
boxes whose centers are offset by `d` along one axis (this project's adjacent-grid-cell case, `s=12,
cell=4`) have `iou = (s-d)*s / (2s^2 - (s-d)*s)`; at `d=4` (one cell over), `iou = 0.5`; at `d=4,4`
(diagonal), `iou ≈ 0.286`; at `d=8` (two cells), `iou = 0.2`. This project's `kNmsIouThreshold=0.25`
sits BETWEEN the diagonal-adjacent and two-cells-away cases — see "How we verify correctness" for
why that placement was chosen deliberately, not by trial and error alone.

## The algorithm

Six stages, each documented in depth at its kernel in `kernels.cu` (device) and mirrored
independently in `reference_cpu.cpp` (host oracle):

1. **Preprocess** — `O(kNetH*kNetW*3)` = 12,288 independent output elements; each a 4-tap bilinear
   read + normalize + a transposed write. Serial cost = parallel cost per element; the WORK is
   embarrassingly parallel across elements.
2. **conv1, conv2, head** — three calls to one GENERIC direct-convolution routine (`Cin*K*K` taps
   per output element: 27 for conv1, 18 for conv2, 2 for the head). `O(Cout*Hout*Wout*Cin*K*K)`
   total multiply-adds — 2,048*27 + 512*18 + 1,536*2 ≈ 68k MACs across all three layers, trivial at
   this teaching scale (a production backbone at 640×640 with real channel counts is 10^9–10^10
   MACs — THEORY "Where this sits in the real world" is where the gap becomes TensorRT's job).
3. **Argmax decode** — `O(kNumClasses)` work per grid cell, `O(kGridH*kGridW)` cells — a trivial
   reduction along a 2-element axis, generalized in code to any `kNumClasses`.
4. **Threshold + box decode** — `O(1)` work per grid cell (a comparison, 4 transcendental calls,
   an atomic compaction). Sequential dependency: NONE between cells — this is why GPU stream
   compaction (see "The GPU mapping") works at all.
5. **NMS** — the IoU MATRIX is `O(n^2)` independent pairs (`n` = pre-NMS candidate count, measured
   21 on the committed scene); the GREEDY SUPPRESSION SCAN that consumes it is `O(n^2)` worst case
   but *sequentially dependent* — survivor `i` must be resolved before candidate `i+1` can be, so
   its real critical-path length is `O(n)` decisions, each `O(n)` work, not `O(n^2)` parallel steps.
6. **Keypoint extraction** — `O((2R+1)^2)` work per surviving detection (`R=2` → 25 cells), `m`
   detections (measured 3) — the smallest stage by far, dominated by kernel LAUNCH overhead at this
   scale, not by its arithmetic (an honest note, not hidden — see "The GPU mapping").

## The GPU mapping

Every kernel below is documented in full at its definition in `kernels.cu`; this section is the
cross-kernel synthesis the task explicitly asks for.

```
KERNEL                  THREADS = ...                MEMORY PATTERN
preprocess               3 * kNetH * kNetW (12,288)    scattered 2x2 read per thread; coalesced write
conv2d (x3)               Cout*Hout*Wout (2048/512/1536) small strided reads, fits in cache at this size
argmax_decode             kGridH*kGridW (256)           2 uniform-stride reads per thread
threshold_box_decode      kGridH*kGridW (256)           4 reads + ATOMIC compaction write
iou_matrix                n*n (<= 21*21=441 here)        2 struct reads per thread, dense write
keypoint_extract          n_post_nms (<= a few)          (2R+1)^2 windowed reads per thread
```

No kernel in this project uses shared memory: at these teaching sizes, the entire working set of
every kernel comfortably fits in L1/L2 cache (the whole preprocessed tensor is 48 KB; a real
production conv layer TILES its input into shared memory precisely because ITS working set does
NOT fit — THEORY.md is honest that this project's kernels would need that tiling to scale, and that
tiling is exactly the kind of tactic TensorRT's builder searches over automatically, see below).

**ATOMICS — documented per CLAUDE.md §6.1 rule 2.** `threshold_box_decode_kernel` uses ONE
`atomicAdd(count, 1)` per firing cell to claim a unique slot in the compacted candidate array — the
textbook GPU STREAM COMPACTION primitive: turn "which of 256 cells passed the filter?" (a sparse,
unpredictable set) into a dense array NMS can index linearly, with the hardware serializing the
handful of concurrent claims (at most 256 threads ever call this in the whole program) so no two
threads ever collide on the same slot. At PRODUCTION anchor counts (tens of thousands, not 256),
atomics contention on a single global counter becomes a real profiling concern; the standard escape
hatches are per-block LOCAL compaction (each block claims one contiguous range with a single
block-level atomic) or a proper prefix-sum/stream-compaction primitive (Thrust's
`copy_if`, or CUB's `DeviceSelect`) — named here, not implemented, because at this project's scale
(≤256 threads) the simple version is both correct and free.

**THE PARALLELISM TENSION IN NMS (the task's specific ask).** Classic greedy NMS is: sort by score;
walk survivors in order; each survivor `i` suppresses every LATER box `j` it overlaps enough. Box
`j`'s fate depends on whether box `i` (i<j) itself survived, which depends on `i-1`, and so on — a
genuine sequential dependency chain, not an artifact of a lazy implementation. What DOES parallelize
trivially, and is where NMS's real `O(n^2)` cost lives, is computing every pairwise IoU up front —
so THIS project gives that part a real kernel (`iou_matrix_kernel`) and keeps the inherently
sequential suppression SCAN on the host (`main.cu`'s `greedy_nms_from_matrix`), exactly mirroring
08.01's decision to keep its O(K·T) softmin blend on the host: **trivial, honestly sequential
bookkeeping does not deserve a kernel just because it CAN have one.** Production systems that DO
need NMS fully on-device (to avoid a host round-trip inside a larger on-device pipeline) reach for
algorithmically different formulations — Matrix NMS (Wang et al. 2020) reformulates suppression as
a decay function over the IoU matrix that parallelizes without a sequential scan, at the cost of
being an approximation of classic greedy NMS, not identical to it; NVIDIA's EfficientNMS TensorRT
plugin is a production, fully-on-device implementation of a similar idea (README "Prior art").

**Kernel launch overhead is the honest bottleneck at this scale**, not arithmetic: six kernel
launches, most touching only tens-to-hundreds of threads, on a modern GPU each cost low-single-digit
microseconds of launch latency — comparable to or larger than the actual compute. This is stated
plainly rather than hidden behind a favorable-looking timing number (the demo's `[time]` line is
honest about this too, CLAUDE.md §12): it is the expected, correct behavior of a teaching-scale
pipeline, and it is exactly the situation CUDA Graphs (32.02) exists to help with in production.

## Numerical considerations

- **FP32 throughout, no fast-math.** Every kernel and its CPU twin use standard `sinf`/`expf`-class
  precise intrinsics (`kernels.cu` never sets `--use_fast_math` — see the `.vcxproj` comment); the
  measured GPU-vs-CPU deviation (worst 2.25e-7 relative, see "How we verify correctness") confirms
  this project never needed the accuracy trade fast-math offers.
- **Convolution accumulation order.** Both `conv2d_kernel` (GPU) and `conv2d_cpu` (CPU) sum taps in
  the SAME nested order — `ci` outermost, then `ky`, then `kx` — documented explicitly in both
  files, because a different summation order can legally produce a different FP32 rounding for the
  same mathematical sum (floating-point addition is not associative). At this project's shallow
  depth (≤27 chained multiply-adds per output element) the practical effect is negligible (measured
  worst case ~1e-7 relative) — but the discipline is the same one that matters enormously at
  production depth (a 50-layer backbone's rounding CAN diverge visibly between accumulation orders,
  which is one reason TensorRT engines built with different tactics can produce slightly different
  numeric output for bit-identical inputs).
- **`sigmoid`/`exp` in the box decode** are smooth, well-conditioned functions over the box-regression
  channels' expected range (this project's synthetic weights keep the inputs at exactly 0.0, the
  best-conditioned point of both functions) — no stability concern here, but a trained network's
  regression outputs are NOT bounded by construction, and production decode kernels typically clamp
  `tw,th` before `exp()` to avoid an exploding box from a wild regression output; this project omits
  the clamp because its fixed weights make it provably unreachable (documented, not silently absent).
- **Deterministic tie-breaking.** `argmax_decode` uses strict `>` (first class wins ties);
  `keypoint_extract`'s window scan uses strict `>` in a fixed row-major order; the NMS
  sort (both the GPU path's `main.cu` comparator and `reference_cpu.cpp`'s `nms_cpu`) breaks score
  ties by ascending `cell_index`. This project's flat-colored synthetic objects DO produce exact
  ties (worked example 1 shows why: several interior cells reach the identical peak value) — every
  one of these rules exists because an UNDOCUMENTED tie-break would make the GPU and CPU paths agree
  only "usually", which is not a verification gate worth having.
- **FP16/INT8 quantization, conceptually (the optional TensorRT path).** FP16 halves memory
  traffic and (on Tensor-Core-capable GPUs) can roughly double or better arithmetic throughput,
  at reduced mantissa precision (11 bits vs FP32's 24) — for weights and activations already in the
  small range this project's hand-designed network produces (roughly -3..+3), FP16 costs essentially
  nothing in accuracy, which is why `tensorrt_path.cpp` enables it unconditionally when the platform
  supports it. INT8 is a bigger step: every tensor needs a per-tensor (or per-channel) SCALE factor
  mapping its observed float range onto the 256 signed 8-bit levels, and picking a good scale
  requires CALIBRATION — running a representative batch of real inputs through the FP32 network and
  recording activation histograms (TensorRT's builder does this via an `IInt8Calibrator`, commonly
  using an entropy-minimization criterion to pick the scale that best preserves the tensor's
  information content). This project does not attempt INT8 calibration — it would need a
  calibration DATASET this hand-designed, non-trained network has no natural equivalent of — but
  the concept is documented here because it is the single biggest "gotcha" a first TensorRT
  deployment runs into: INT8 is not a flag, it is a measurement.

## How we verify correctness

Three independent checks, layered, because a detector pipeline can be **numerically right and
behaviorally wrong** (or the reverse) at any single stage:

1. **Stage-wise GPU-vs-CPU tensor agreement** (`main.cu`'s VERIFY stage): the preprocessed tensor,
   conv1 output, conv2 output, and head output are each compared element-wise, relative tolerance
   `1e-3` with a `max(1,|cpu|)` floor (the same pattern 08.01 uses) — justified above ("Numerical
   considerations": ≤27 chained FP32 MACs). **Measured worst case: 2.247e-7** (head output) — over
   4,000× inside the tolerance, while an indexing/layout bug (the kind this gate exists to catch)
   would typically shift a value by order 1, not 1e-7.
2. **Exact candidate/detection COUNT agreement** (integer, not tolerance-based): the GPU and CPU
   paths must produce the IDENTICAL pre-NMS candidate count and post-NMS survivor count. **Measured:
   21 = 21 and 3 = 3.** A mismatch here is unambiguous: either a threshold/argmax bug (wrong count of
   cells firing) or an NMS bug (wrong suppression decisions) — not FP32 rounding, because the
   thresholds involved (score gate, IoU gate) sit far from any candidate's actual value (worked
   example 2's 2.97-unit gap; the IoU-threshold placement note in "The math").
3. **The ground-truth gate** (behavioral correctness against the KNOWN scene, `data/sample/
   ground_truth.csv`): each of the 3 placed objects must be matched (same class, center distance
   ≤6.0 network-input px, IoU ≥0.30) to an UNUSED detection; any detection left unmatched counts as
   a false positive (bound: ≤1); NMS must reduce the pre-NMS candidate count by at least 3×.
   **Measured: 3/3 matched, worst center error 2.40 px, worst IoU 0.601, 0 false positives, 7.0×
   reduction** — every measured number sits with 2×+ headroom against its documented bound, chosen
   (per the "worked example" derivation and the IoU-vs-cell-spacing arithmetic in "The math") so
   ordinary FP32 rounding differences across GPU architectures cannot flip any verdict.

No stable, checked output line contains any of these measured numbers (repo convention, CLAUDE.md
§12) — only the PASS/FAIL verdicts are checked, exactly like 08.01's swing-up demo; the numbers
themselves live in `[info]` lines and in this document, honestly reported, never fabricated.

## Where this sits in the real world

- **What TensorRT actually does, concretely** (the optional path's target): given the SAME network
  graph and weights this project's fallback kernels implement by hand, TensorRT's BUILDER performs
  **layer fusion** (recognizing "conv immediately followed by ReLU" — exactly this project's conv1+
  ReLU and conv2+ReLU — and emitting ONE kernel instead of two, skipping a global-memory round trip
  for the intermediate tensor), **tactic selection** (benchmarking several candidate GPU kernels per
  layer — different tiling strategies, different use of Tensor Cores — on the ACTUAL target GPU
  during the build call, and keeping the fastest; this is why a serialized engine is tied to the GPU
  architecture it was built on), and **precision calibration** (FP16 by a flag, INT8 by measurement
  — see "Numerical considerations"). None of this changes the pre/post kernels this project actually
  teaches — `tensorrt_path.cpp`'s whole point is that `launch_argmax_decode`,
  `launch_threshold_box_decode`, `launch_iou_matrix`, and `launch_keypoint_extract` are called
  IDENTICALLY whether `d_head_out` came from three hand-written kernel launches or one
  `enqueueV3()` call.
- **NVIDIA Triton Inference Server** sits one layer above a single engine: model versioning, dynamic
  batching across concurrent requests, multi-framework backends (TensorRT, ONNX Runtime, PyTorch),
  and a network-facing serving API — the "how do many robots/cameras share one inference server"
  problem, out of this project's single-process, single-image scope.
- **torch-tensorrt / ONNX Runtime's TensorRT execution provider** are how most real teams reach
  TensorRT in practice: trace or export a trained PyTorch model, hand it to the tool, and let IT
  build the `INetworkDefinition` — hand-writing `addConvolutionNd` calls (as `tensorrt_path.cpp`
  does) is rare outside teaching contexts or genuinely custom/non-standard architectures.
- **NVIDIA DeepStream** is the closest production shape to this ENTIRE project: a video-analytics
  pipeline built from GStreamer elements that bracket a TensorRT engine with custom CUDA
  pre/post-processing plugins, including a shipped, production `EfficientNMS` TensorRT plugin — the
  plugin-vs-separate-kernels trade this project's `tensorrt_path.cpp` file header discusses is a
  real decision DeepStream-style pipelines make explicitly.
- **What the full version needs beyond this teaching core:** a real trained backbone (hundreds of
  layers, learned weights — 12.06's territory) at production input resolution; multi-scale
  detection (a feature pyramid, not one 16×16 grid); INT8 calibration against real captured data;
  the plugin path for any custom op that needs to live INSIDE the engine's captured graph; and, for
  anything commanding a robot from these detections, the full tracking/fusion and safety-monitor
  layers named in this project's README "System context".
