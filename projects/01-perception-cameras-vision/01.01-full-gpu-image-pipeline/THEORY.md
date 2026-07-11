# 01.01 — Full GPU image pipeline: debayer → undistort → rectify → resize → normalize, zero CPU copies: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

### Why a sensor sees one color per pixel: the Bayer filter array

A digital image sensor's photosites (the physical light-collecting wells on the silicon) cannot tell
red light from green from blue — a photon liberates the same electron regardless of its wavelength.
To capture color, almost every consumer and machine-vision camera glues a **color filter array**
(CFA) directly onto the sensor: a mosaic of tiny red, green, and blue dye filters, one per photosite,
so each site only ever sees light of roughly one color. The **Bayer pattern** (Bryce Bayer, Kodak,
1976) is the near-universal choice:

```
R G R G R G
G B G B G B
R G R G R G
G B G B G B
```

Green gets **twice** the photosites of red or blue — a deliberate choice matched to human (and,
usefully, most machine-vision) luminance sensitivity peaking in the green band, so green carries most
of the spatial detail. The RGGB variant used here names the 2x2 repeating tile starting at the
top-left corner: Red, Green, Green, Blue. **Debayer** (also called demosaicing) is the algorithm that
reconstructs a full 3-channel image from this 1-channel-per-pixel mosaic — every pixel's other two
colors are *estimates*, interpolated from nearby same-color photosites, never measured directly. This
is a real, permanent loss of information relative to a 3-sensor (prism/beamsplitter) camera; it is
also why real ISPs spend so much engineering effort on demosaic quality.

### Why lenses distort: the physics of Brown-Conrady

A pinhole camera (an idealized aperture with zero size) projects a 3-D point onto the image plane by
simple, distortion-free central projection: `x_pixel = fx * X/Z + cx`. Real lenses are not pinholes —
they are stacks of curved glass elements that must simultaneously focus light, fit a form factor, and
stay affordable to grind and mold. Two physical effects bend the pinhole projection:

- **Radial distortion** (`k1, k2` terms): a spherical (or close-to-spherical) lens element bends
  light rays passing through its edge MORE than rays passing through its center — a fundamental
  consequence of Snell's law applied to a curved surface where the local angle of incidence grows
  with distance from the optical axis. The result is **barrel distortion** (image features appear to
  bulge outward, common in wide-angle lenses, `k1 < 0`) or **pincushion distortion** (features pull
  inward toward the corners, common in telephoto zooms, `k1 > 0`). This project uses `k1 = -0.22`
  (mild-to-moderate barrel, typical of a wide-ish machine-vision lens) plus a small `k2` correction
  term for the next order of the same effect.
- **Tangential distortion** (`p1, p2` terms): the lens elements are never PERFECTLY parallel to the
  sensor plane — a real assembly line has finite tolerance in how the lens barrel screws onto (or is
  glued to) the sensor housing. A lens tilted by a fraction of a degree relative to the sensor
  introduces an asymmetric, direction-dependent warp that a purely radial model cannot capture. This
  project's `p1 = 0.0010, p2 = -0.0008` are small (a well-assembled machine-vision lens), consistent
  with tangential distortion being the SMALLER of the two effects in most real cameras.

The **Brown-Conrady model** (Brown, 1966, generalizing Conrady, 1919) is the standard closed-form
polynomial approximation engineers use for both effects together — not a first-principles optics
derivation (that would require ray-tracing the actual lens prescription, element by element), but an
empirical fit accurate enough that every camera calibration toolkit (OpenCV, Kalibr, MATLAB) uses it.
"The problem" this project solves is inverting that fit well enough to recover the pinhole image the
same scene would have produced through an ideal, distortion-free lens.

### Why "rectify" is a separate step from "undistort"

A camera is not glued perfectly straight onto its mount. Manufacturing tolerance on the sensor
PCB-to-lens-barrel-to-housing chain routinely leaves a fraction of a degree to a few degrees of
unintended tilt — small enough that the image still looks "basically right" to a human, but large
enough to break any downstream algorithm that assumes pixel rows are exactly horizontal in the
robot's reference frame (most importantly: STEREO rectification, where even 0.1 degrees of relative
tilt between two cameras measurably corrupts triangulated depth — see sibling project 01.02). This
project's **rectifying rotation** (2 degrees about the camera's Y axis) models exactly that kind of
correctable mounting error — deliberately made large enough to be a genuine, visible correction (not
a token 0.01-degree no-op), separate from the lens's own (per-element) distortion.

### Why area-average resizing, and why normalize at all

**Resize**: shrinking an image by dropping samples (nearest-neighbor) or naively re-sampling
(bilinear at the wrong scale) causes **aliasing** — high spatial frequencies in the source that
exceed the new sampling rate's Nyquist limit fold back into false low-frequency patterns (moiré).
The correct anti-aliasing filter for an INTEGER decimation factor N is a box average over exactly an
NxN window — every source pixel contributes to exactly one destination pixel with equal weight,
which is both the correct low-pass filter for this specific case and, not coincidentally, the
cheapest possible one to compute.

**Normalize**: a neural network trained with inputs centered near zero and scaled near unit variance
converges faster and is less sensitive to weight initialization than one fed raw `[0,255]` pixel
values (a well-established practical finding, not a law of physics) — this is why virtually every
camera-fed perception model (domain 12 of this repository) expects a zero-mean/unit-std tensor at its
input, and why a real camera ISP pipeline's LAST stage before handing off to a neural net is this
exact affine transform.

### Engineering constraints a real robot imposes

- **Rolling vs. global shutter**: a CMOS rolling-shutter sensor exposes rows sequentially (top to
  bottom, microseconds apart), so a fast-moving robot or a fast-moving object in the scene produces a
  skewed image — straight vertical edges become slanted. Global-shutter sensors (all pixels exposed
  simultaneously) avoid this at a cost (usually more silicon, sometimes lower fill-factor / worse
  low-light performance) and are strongly preferred for anything moving quickly (drones, fast
  manipulators). This project assumes a global shutter — see "Where this sits in the real world"
  for what a rolling-shutter correction stage would add.
- **Bandwidth/latency**: at 30-60 Hz camera->perception (SYSTEM_DESIGN.md §1.1), this whole pipeline
  has a 16-33 ms budget SHARED with everything downstream of it — this project's staged pipeline
  measuring well under 1 ms at 384x288 leaves enormous headroom (real sensors run at higher
  resolutions, where the same kernels scale roughly linearly with pixel count).
- **Fixed-point vs. float in real ISPs**: dedicated ISP silicon (as opposed to a general-purpose GPU)
  usually runs debayer/undistort/resize in FIXED-POINT arithmetic for power efficiency and
  determinism — this project's float32 GPU implementation is the teaching-friendly, more general
  choice; "Where this sits in the real world" names the fixed-point alternative.

## The math

**Notation**: pixel coordinates `(x, y)`, `x` rightward, `y` downward (image convention). Camera
frame is z-forward/x-right/y-down (the stated exception to the repo's default body-frame convention —
see `kernels.cuh`'s file header). `K = (fx, fy, cx, cy)` is the pinhole intrinsic matrix in pixel
units; `(fx, fy)` are the two focal lengths in pixels (equal here — square pixels, no aspect
correction needed), `(cx, cy)` is the principal point in pixels.

**Pixel <-> normalized coordinates**: `x_n = (x - cx)/fx`, `y_n = (y - cy)/fy`, with implicit `z_n =
1` (this is central projection: a normalized coordinate is where a ray through pixel `(x,y)` and the
camera's optical center crosses the `z=1` plane).

**Brown-Conrady forward distortion** — ideal (undistorted) normalized `(xu, yu)` -> distorted
normalized `(xd, yd)`:

```
r^2 = xu^2 + yu^2
xd = xu * (1 + k1*r^2 + k2*r^4) + 2*p1*xu*yu + p2*(r^2 + 2*xu^2)
yd = yu * (1 + k1*r^2 + k2*r^4) + p1*(r^2 + 2*yu^2) + 2*p2*xu*yu
```

This is a **closed-form forward map**: given any ideal point, distortion is one polynomial
evaluation. There is **no closed-form inverse** — `(xd, yd) -> (xu, yu)` requires solving a
degree-5 polynomial system with no general algebraic solution, so every real system (this one
included) solves it numerically (see "Numerical considerations").

**Rectifying rotation** `R_rect_raw` (T_parent_child convention, parent = rectified frame, child =
raw frame; `t = kRectifyAngleDeg = 2 deg` about the camera's Y axis):

```
R_rect_raw = [ cos(t)   0   sin(t) ]        v_rect = R_rect_raw * v_raw
             [   0      1     0    ]
             [-sin(t)   0   cos(t) ]
```

**The full forward camera model** (ideal, rectified pixel `(xo, yo)` -> raw pixel `(u, v)` to sample
— exactly `compute_source_pixel()` in `kernels.cuh`):

```
1. (xr, yr) = ((xo-cx)/fx, (yo-cy)/fy)                       ray in the rectified frame
2. (rx, ry, rz) = R_rect_raw^T . (xr, yr, 1)                 rotate into the raw frame
3. (xn, yn) = (rx/rz, ry/rz)                                 perspective-divide
4. (xd, yd) = distort_forward(xn, yn)                        the physical lens's effect
5. (u, v) = (fx*xd + cx, fy*yd + cy)                          back to raw pixel coordinates
```

**Area-average resize** (`kResizeFactor = N = 2`): output pixel `(xo, yo)` averages the `N x N` block
of input pixels starting at `(N*xo, N*yo)`:

```
out(xo, yo) = (1/N^2) * sum_{dy=0}^{N-1} sum_{dx=0}^{N-1} in(N*xo+dx, N*yo+dy)
```

**Normalize** (per channel `c`, over all `n` pixels of the resized image):

```
mean_c = (1/n) * sum_i pixel_i[c]
var_c  = (1/n) * sum_i pixel_i[c]^2  -  mean_c^2          (population variance, E[x^2]-E[x]^2)
std_c  = sqrt(max(var_c, eps))
out_i[c] = (pixel_i[c] - mean_c) / std_c
```

## The algorithm

**Stage 1 — DEBAYER (bilinear demosaic).** For output pixel `(x,y)`, `bayer_channel_at(x,y)`
(`kernels.cuh`) says which of {R,G,B} was measured NATIVELY there. The other two channels are
estimated from same-color neighbors:

```
if native == R:  G = avg(N,S,E,W);           B = avg(NE,NW,SE,SW)
if native == B:  G = avg(N,S,E,W);           R = avg(NE,NW,SE,SW)
if native == G, row is an R-row:  R = avg(E,W);  B = avg(N,S)
if native == G, row is a  B-row:  B = avg(E,W);  R = avg(N,S)
```

Complexity: `O(W*H)` work, `O(1)` per pixel (up to 8 neighbor reads) — serial cost `O(W*H)`,
perfectly parallel cost `O(1)` given `W*H` processors (the standard "embarrassingly parallel" story
for every stage in this project).

**Stage 2+3 — UNDISTORT+RECTIFY as ONE inverse-mapped remap.** Two ways to warp an image between
two pixel grids exist:

- **Forward mapping**: for each INPUT pixel, compute where it lands in the output, and write it
  there. Problem: the mapping is generally NOT onto — some output pixels receive zero writes (holes)
  and some receive multiple (overlaps needing blending), because the forward map's Jacobian is not
  everywhere exactly `N x N` pixels-to-pixels.
- **Inverse mapping** (this project): for each OUTPUT pixel, compute where it CAME FROM in the input,
  and bilinear-sample there. Every output pixel gets exactly one well-defined answer — no holes are
  even possible, because the loop is structured over the output grid, not the input grid. This is why
  virtually every production remap (OpenCV's `remap`/`initUndistortRectifyMap`, every camera ISP)
  uses inverse mapping, and why `compute_source_pixel()` (the math above) is defined output-to-input,
  not input-to-output.

The **LUT** (lookup table) precomputes `compute_source_pixel()` for every output pixel ONCE — it
depends only on the (fixed) camera model, never on image content, so a real ISP builds it once at
calibration time and reuses it every single frame thereafter (this project's `build_remap_lut_kernel`
mirrors that). **Bilinear sampling** at the LUT's fractional `(u,v)` then interpolates the four
nearest debayered pixels — necessary because `compute_source_pixel()` almost never lands on an exact
integer coordinate.

**Stage 4 — RESIZE.** The area-average formula above, `O(W*H)` work total (`O(N^2)` per output
pixel, `O(1)` given one thread per output pixel).

**Stage 5 — NORMALIZE.** A **two-pass** algorithm: pass 1 computes `mean_c`/`std_c` (a REDUCTION —
`O(n)` serial, `O(log n)` parallel depth with `n` processors); pass 2 applies the affine map (a MAP —
`O(n)` serial, `O(1)` parallel depth). Two passes are required because the affine map needs the
WHOLE image's statistics before it can transform even the first pixel — unlike every other stage
here, pixel `i`'s output genuinely depends on every other pixel, not just its neighbors.

## The GPU mapping

**Thread-to-data mapping** (stages 1-4): one thread per output pixel, a 16x16 2-D block (a
warp-multiple; at 384x288 this is 24x18=432 blocks, far more than an RTX 2080 SUPER's 48 SMs need to
stay fed) — the same launch-geometry reasoning as sibling flagship 01.02.

**Memory hierarchy**: every kernel here reads/writes GLOBAL memory only — no shared memory tiling.
This is a deliberate teaching simplification, named honestly: debayer's 3x3 stencil and the
resize/fused kernels' 2x2/4x-bilinear windows both re-read overlapping neighbor data across adjacent
threads (a classic shared-memory-tiling opportunity — cache the block's footprint once, in
on-chip shared memory, instead of every thread independently re-fetching from DRAM). This repo's
default here favors READABILITY (see CLAUDE.md §1); README Exercise 3 and this project's honest
memory-traffic accounting below both point at where tiling would help most.

**The kernel-fusion argument — memory traffic, derived by hand.** Let `W, H` be the full resolution,
`N = kResizeFactor`. Assume an IDEALIZED, no-cache-reuse cost model (every read counted, even ones an
L2/texture cache would likely serve for free in practice — an honest worst-case bound, not a
profiler measurement):

*STAGED* (`remap_bilinear_kernel` then `resize_area2x_kernel`):
```
remap:  reads 1 bilinear sample (4 texels x 3 bytes = 12 bytes) per pixel, writes 3 bytes/pixel
          -> over W*H pixels: 12*W*H read, 3*W*H write
resize: reads N^2 texels (N^2*3 bytes) per OUTPUT pixel, writes 3 bytes/pixel
          -> over (W*H)/N^2 output pixels: N^2*3*(W*H/N^2) = 3*W*H read, (3*W*H)/N^2 write
STAGED total = 12WH + 3WH + 3WH + 0.75WH = 18.75 * W*H bytes  (N=2)
```

*FUSED* (one kernel, per resized output pixel, `N^2` bilinear samples):
```
fused: reads N^2 bilinear samples (N^2 * 12 bytes) per OUTPUT pixel, writes 3 bytes/pixel
          -> over (W*H)/N^2 output pixels: N^2*12*(W*H/N^2) = 12*W*H read, 0.75*W*H write
FUSED total = 12WH + 0.75WH = 12.75 * W*H bytes  (N=2)
```

The gap, `18.75WH - 12.75WH = 6*W*H` bytes, is EXACTLY the staged path's intermediate full-resolution
image: written once (`3*W*H` bytes, by `remap_bilinear_kernel`) and read once (`3*W*H` bytes, by
`resize_area2x_kernel`) — a round trip through global memory carrying information that is used
EXACTLY once on the far end. Fusing the two kernels deletes that round trip and nothing else — the
"genuine" sampling work (walking the debayered image via bilinear interpolation) is IDENTICAL in
both versions (`12*W*H` bytes either way), which is the honest way to see that fusion's saving here
is a **memory-traffic** optimization, not a **compute** optimization. At this project's committed
384x288 scene: STAGED = 2,073,600 bytes, FUSED = 1,410,048 bytes, a measured-and-derived **32.0%**
reduction (`src/main.cu` prints both numbers every run).

**Honesty about the idealized model**: real GPUs cache aggressively, and adjacent output pixels'
bilinear footprints overlap heavily (a texture/L2 cache typically serves most of those "duplicate"
12-byte reads for free) — so the MEASURED kernel-time gap between staged and fused is usually smaller
than the derived byte-count gap predicts (see README "Expected output" for the actual measured
times). Both numbers are printed side by side precisely so a learner can see that "bytes requested"
and "wall-clock time" are related but different quantities — profiling with Nsight Compute (README
Exercise, `docs/BUILD_GUIDE.md`) would show the ACTUAL DRAM traffic and confirm the cache's rescue.

**cuBLAS/cuFFT/Thrust**: none used. Every stage here is small enough, and specific enough to image
geometry, that a hand-written kernel teaches more than a library call would (CLAUDE.md §5's "prefer
hand-rolled" default) — see README "Prior art" for where a production system WOULD reach for a
library (OpenCV `cv::cuda`, NVIDIA VPI).

## Numerical considerations

**Precision**: every stage computes in FP32 except the normalize reduction's accumulators (FP64 —
see below); the final tensor is FP32. FP64 is never needed for per-pixel image math at 8-bit input
precision (FP32 carries ~7 decimal digits, vastly more than the ~2.4 decimal digits an 8-bit pixel
value needs).

**Why the normalize reduction accumulates in FP64.** Summing `n ~= 27,648` terms (this project's
resized image, `192x144`) in FP32 risks real precision loss: FP32 has ~7 significant decimal digits,
and a running sum-of-squares over 8-bit-squared values (up to `255^2 = 65,025` per term) can reach
into the millions — leaving only 2-3 digits of headroom for each NEW term being added, a textbook
case of "catastrophic cancellation by accumulation" that grows worse as image size grows. FP64
carries ~15-16 significant digits, comfortably enough for sums this size; this project's kernels
accumulate in `double` throughout the reduction and only cast down to `float` at the very last step
(`launch_normalize_finalize`).

**The atomics-vs-tree-reduction determinism choice (CLAUDE.md §12).** The fastest way to sum millions
of GPU-computed values into one number is usually `atomicAdd` from every thread straight into a
single global accumulator — but floating-point addition is **not associative**
(`(a+b)+c != a+(b+c)` in general, because each addition independently rounds), and `atomicAdd`'s
arrival order is whatever the hardware scheduler happens to produce that run — meaning the exact bit
pattern of an atomic-accumulated sum can differ between two runs of the IDENTICAL kernel on the
IDENTICAL input. This project instead uses a FIXED two-level tree: (1) each block reduces its own
slice via an in-block shared-memory binary tree (`normalize_block_stats_kernel`) — a fixed reduction
order, identical every run, because the STRUCTURE of a within-block tree reduction never depends on
scheduling, only its INPUT does; (2) a single `<<<1,1>>>` thread (`normalize_finalize_kernel`) sums
the resulting (169 at this project's resolution) block partials in a fixed sequential order. **No
atomic instruction appears anywhere in this pipeline.** The cost is one tiny extra kernel launch; the
payoff is bit-reproducible results run after run on the same GPU — which is what lets the `normalize`
gate assert a tight numeric tolerance instead of a statistical one.

**FMA contraction: the actual source of this project's GPU-vs-CPU drift.** `nvcc` contracts
`a*b+c`-shaped expressions into a single fused-multiply-add (FMA) instruction on the device BY
DEFAULT (one rounding step); `cl.exe` does NOT contract multiply-add by default on the host (two
rounding steps) unless `/fp:fast` is set (this project does not set it — CLAUDE.md §5 wants
reproducible floats by default). `compute_source_pixel()`'s distortion polynomial and
`bilinear_sample_rgb`'s `v00 + (v10-v00)*fx` interpolation are both exactly this shape. The
DIFFERENCE this causes is tiny in absolute terms — the remap LUT differs by at most `5.2e-5` px
between GPU and CPU (measured) — but because pixel values are then ROUNDED to the nearest uint8, a
sample that lands extremely close to an exact `X.5` boundary can round to a DIFFERENT integer on the
two platforms: this project measures exactly a `+-1` (out of 255) maximum difference on the
remap/resize/fused stages, on a small handful of pixels. This single-unit rounding difference then
propagates through the normalize stage's division by `std` (empirically ~65 on this project's scene)
into a roughly `1/65 ~= 0.015`-normalized-unit difference — which is why `kTolNormApply` (in
`src/main.cu`) is calibrated to `0.05`, not the naive `0.005` a first guess might pick (see that
file's comment block for the exact derivation and the measured numbers it is based on). None of this
is a bug: it is the honest, expected signature of two independently-rounding IEEE-754 pipelines
computing the same well-conditioned formula.

**Angle wrapping / quaternion drift**: not applicable — this project's only rotation is a single
FIXED 2-degree constant baked in at compile time (`kRectCos`/`kRectSin`), never integrated or
composed at runtime, so none of the usual robotics rotation hazards (drift, wrap-around) arise here.

**uint8 quantization**: every stage before normalize rounds to the nearest 8-bit integer
(`+0.5f` then truncate) — a real, bounded, and unavoidable loss (up to 0.5 units per rounding event,
compounding slightly across debayer -> remap -> resize) that is exactly why a real ISP keeps images
in a WIDER fixed-point or float format internally when precision matters, and only quantizes to 8-bit
at the very end (or, as here, converts to float precisely to STOP quantizing further after
normalize).

## How we verify correctness

**Two independent tiers**, per the repo's twin-independence ruling (`reference_cpu.cpp`'s file
header):

1. **GPU-vs-CPU twin comparison** (`VERIFY:` in the demo). Every kernel's output is compared,
   element-wise, against `reference_cpu.cpp`'s independently-typed CPU implementation (bilinear
   sampling, debayer neighbor logic, and the fused-kernel re-derivation are ALL retyped from scratch
   — NOT calling the same device functions). The one deliberate exception is the camera model itself
   (`compute_source_pixel`/`distort_forward`/`bayer_channel_at`, shared via `kernels.cuh`'s `HD`
   macro) — sharing THAT formula is a documented, permitted exception (it is the hardware fact being
   modeled, not the algorithm under test), which is exactly why tier 2 exists.
2. **Six physical gates**, each independent of the shared camera-model code:
   - **`roundtrip`**: `src/main.cu` hand-retypes BOTH the forward distortion formula AND an
     independent fixed-point undistort iteration (never calling `kernels.cuh`), forward-maps a grid
     of points, inverts them, and checks the round trip returns to the start (max 0.05 px tolerance —
     measured 0.00000 px). This is the gate that would catch a bug INSIDE the shared
     `compute_source_pixel()` itself, which tier 1 structurally cannot.
   - **`straightness_rectified` + `distortion_negative_control`**: a from-scratch, host-side
     sub-pixel edge detector (linear interpolation across the 50%-gray threshold) locates a KNOWN
     straight checkerboard boundary in both the rectified output (should measure straight) and the
     RAW debayered image (should measure curved) — a genuine, physically-grounded pass/fail pair,
     with the raw-image check acting as a NEGATIVE CONTROL: if distortion correction were a silent
     no-op, this gate would not distinguish the two images.
   - **`color_fidelity`**: compares the rectified stage's actual pixel values against
     `true_rgb.ppm`'s independently-authored ground truth (built by an entirely separate Python
     script's inverse camera model — three independent implementations of the SAME formula, in three
     languages/files, now must agree), scored only in `smooth_mask`-valid regions to avoid conflating
     unavoidable edge-interpolation blur with a real correctness bug.
   - **`resize_conservation`**: an analytic invariant (area averaging preserves the image's mean) —
     no camera model involved at all.
   - **`normalize`**: recomputes the final tensor's OWN mean/std from scratch (never reusing the
     GPU's own `d_mean3`/`d_std3` — that would be circular) and checks it against the promised
     zero-mean/unit-std property.
   - **`fused_vs_staged`**: the two independently-launched kernel chains must agree within a
     tolerance derived from the documented single-vs-double-rounding difference between them.

**Tolerances** are calibrated from ACTUAL measured runs on the reference machine (RTX 2080 SUPER,
sm_75) with margin, never asserted from theory alone or set at the exact measured value (CLAUDE.md
§8's "never fabricate" — the exact numbers and their derivation live in `src/main.cu`'s top-of-file
comment block and README "Expected output").

## Where this sits in the real world

- **Debayer**: production ISPs (Jetson's ISP hardware, most SoC camera pipelines) run AHWB-optimized
  demosaic variants — Malvar-He-Cutler (a gradient-corrected bilinear extension, the natural next
  step from this project's baseline — README Exercise 2), adaptive/directional interpolation, or
  learned (neural) demosaicing in high-end phone ISPs. This project's bilinear baseline is the
  correct FIRST algorithm to learn, not the one shipped in a flagship product.
- **Undistort+rectify**: OpenCV's `cv::initUndistortRectifyMap` + `cv::remap` (or their CUDA
  equivalents) implement exactly this project's LUT-then-gather structure at production quality
  (sub-pixel-accurate iterative undistort where THIS project's forward-only LUT needs none, because
  it is built output-to-input); NVIDIA VPI's `Remap` algorithm adds hardware-backend acceleration
  (PVA, VIC) beyond the GPU. Calibrating the REAL `k1,k2,p1,p2,fx,fy,cx,cy` for a physical lens (this
  project hardcodes them) is its own well-studied problem — Zhang's method (a 2000 paper, still the
  standard), implemented in OpenCV's `calibrateCamera`, Kalibr, or ROS's `camera_calibration` package.
- **Resize**: `cv::cuda::resize` and NPP's resize primitives support the same area-average mode
  (`INTER_AREA`) for downscaling, plus higher-order filters (Lanczos) for upscaling, which this
  project does not need (it only ever downscales by an exact integer factor).
- **Normalize**: identical in spirit to `torchvision.transforms.Normalize` or TensorRT's input
  pre-processing layer — the exact affine transform every camera-fed neural network expects.
- **Rolling shutter**: a real correction stage (not modeled here) needs per-ROW timestamps and either
  the camera's own motion (IMU-fused) or the scene's optical flow to un-skew each row independently —
  meaningfully harder than this project's per-frame, single-timestamp camera model, and an active
  research area for high-speed robotics (drones especially).
- **Kernel fusion at scale**: this project's hand-fused kernel is the same idea production ISP
  compilers (and general-purpose ones like XLA/TVM for neural-net graphs) apply automatically —
  "operator fusion" is a standard compiler pass precisely because the memory-traffic argument derived
  above generalizes far beyond image pipelines.
