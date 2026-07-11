# 01.04 — Feature pipeline: FAST/Harris detection, ORB descriptors, brute-force Hamming matcher: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

### What makes a point in an image "trackable" at all?

A camera measures irradiance — photons per unit time per unit area — integrated over each pixel's
sensor well and quantized to an intensity value `I(x, y)` in `[0, 255]` (8-bit, this project's
convention; SYSTEM_DESIGN.md's interface conventions apply generally, but pixel intensity has no SI
unit here — it is a relative, sensor-referred quantity). If a robot's camera moves and takes a second
picture, most pixels tell you almost nothing about that motion: a smooth wall, a flat floor, the sky —
regions where intensity is locally uniform look IDENTICAL in every direction you could have moved.
Only regions with LOCAL STRUCTURE — where intensity actually changes as you move a small window around
— carry information about how the camera moved.

### The aperture problem, from first principles

Consider a small window sliding over the image, and ask: "by how much does the window's CONTENT change
if I shift the window by `(u, v)`?" Formally, the sum of squared differences (SSD) is

```
E(u, v) = sum_{(dx,dy) in window} [ I(x+dx+u, y+dy+v) - I(x+dx, y+dy) ]^2
```

For a small shift, a first-order Taylor expansion of `I` gives `E(u,v) ≈ [u v] M [u v]^T` where `M` is
the **structure tensor** (derived formally in "The math" below). `M`'s EIGENVALUES tell you everything:

- **Both eigenvalues small** — the window is flat (a wall, the sky). `E` barely changes for ANY shift:
  you cannot tell where you moved. This is the trivial, uninteresting case.
- **One eigenvalue large, one small** — the window straddles a straight EDGE. Shifting ALONG the edge
  barely changes `E` (the edge looks the same slid along itself); shifting ACROSS it changes `E` a lot.
  This is the **aperture problem**: looking through a small window (an "aperture") at a straight edge,
  you can measure motion perpendicular to the edge but NOT along it — the tangential component is
  invisible locally, no matter how sharp the edge is. (Every optical-flow algorithm in this repo, e.g.
  project 01.03, fights this same problem.)
- **Both eigenvalues large** — the window contains a CORNER (or any two-dimensional texture): shifting
  in ANY direction changes `E` substantially. This is exactly the property that makes a point
  trackable, and it is what both FAST and Harris are, in their very different ways, testing for.

This is why "feature pipeline" means CORNER detection specifically, not edge detection: a corner is the
minimal local structure that pins down motion in BOTH image dimensions.

### Brightness constancy — and its violations

Every step above assumes **brightness constancy**: the same physical point produces the same
intensity in both images. Real cameras violate this constantly:

- **Illumination changes** — auto-exposure, a cloud passing, indoor lighting flicker. This project's
  `scene_b.pgm` includes a deliberate `+18` intensity offset specifically to exercise (and test) claims
  of brightness-offset robustness (see "How we verify correctness").
- **Specular reflection** — a shiny surface's brightness depends on VIEWING ANGLE, not just the
  surface itself, so it can look completely different from two viewpoints even though nothing moved.
- **Non-Lambertian shading** — most real materials are not perfectly diffuse; apparent brightness
  shifts with the light-camera-surface geometry as the camera moves.

FAST's THRESHOLDED comparisons (`brighter than I(p)+t` / `darker than I(p)-t`, not an exact-equality
test) and ORB's BINARY intensity-ORDER comparisons (`I(p1) < I(p2)`, not their difference) are both
deliberately designed to survive a UNIFORM brightness or contrast shift: adding a constant to every
pixel in a window changes none of these order relationships. Neither is robust to the NON-uniform
violations above (specularity, strong local shading change) — a real, honest limitation shared by every
classical feature pipeline (see "Where this sits in the real world").

### Rolling-shutter honesty

Most CMOS cameras (rolling shutter, as opposed to global shutter) do not capture a frame all at once —
each row is exposed at a slightly later time than the row above it, typically tens of microseconds
apart. A fast-moving robot (a spinning quadrotor, a car on a bump) can see visible SKEW between the top
and bottom of one "frame": the corners this project detects are not actually all measured at the same
instant. This project's synthetic scenes are rendered as ideal, instantaneous (global-shutter) captures
— a real, named simplification. Project 01.10 (rolling-shutter correction using IMU rates) is the
project in this repository that addresses this specific effect; a production VO/SLAM front end
downstream of a rolling-shutter camera needs that correction (or an explicit rolling-shutter-aware
camera model) upstream of a feature pipeline like this one.

### Engineering constraints a real robot imposes

- **Bandwidth/latency**: a feature front end competes for the same 16-33 ms frame budget as everything
  else in perception (SYSTEM_DESIGN.md §1.1) — it must finish in a small fraction of that, leaving time
  for the state estimator and planner downstream.
- **Determinism**: a SLAM back end (bundle adjustment, pose-graph optimization) is sensitive to
  correspondence OUTLIERS; a feature front end's false-match rate directly costs the back end robustness
  budget — this is why this project's matcher chains THREE independent filters (ratio test, absolute
  distance cap, mutual cross-check) rather than one.
- **Texture-poverty**: real environments (blank walls, uniform floors, fog, low light) can starve a
  feature pipeline of corners entirely — a production stack needs a fallback (direct/photometric
  methods, IMU-only dead reckoning) for exactly this failure mode; this project's synthetic scene is
  deliberately texture-RICH and never exercises that failure case.

## The math

### Notation

- `I : Z^2 -> [0, 255]`, the grayscale image, row-major, `I(x, y)` = intensity at pixel column `x`, row
  `y` (integer pixel coordinates; `(0,0)` is the top-left pixel's center — no `-0.5` calibration offset
  is used in this project, unlike the pinhole-camera convention in project 01.01, since this project
  never needs to relate pixels to a metric focal length).
- `kW = kH = 256` px, this project's fixed image size (`kernels.cuh`).
- Angles are radians internally (`theta`, `atan2f`'s native `(-pi, pi]` range), reported in degrees at
  the human-facing `main.cu` print statements only.

### The structure tensor, derived

Starting from the SSD windowed-shift cost `E(u,v)` in "The problem" above, a first-order Taylor
expansion of `I(x+dx+u, y+dy+v)` around `(x+dx, y+dy)` gives `I(x+dx+u,y+dy+v) ≈ I(x+dx,y+dy) + u*Ix +
v*Iy`, where `Ix = dI/dx`, `Iy = dI/dy` are the LOCAL image gradients at `(x+dx, y+dy)`. Substituting:

```
E(u,v) ≈ sum_{window} (u*Ix + v*Iy)^2 = [u v] * M * [u v]^T,   M = sum_{window} [ Ix^2   Ix*Iy ]
                                                                                 [ Ix*Iy  Iy^2  ]
```

`M` (the **structure tensor**, a 2x2 symmetric matrix, one per pixel) is exactly what
`harris_response_kernel` computes (`kernels.cu`), with the window sum implemented as a
`(2*kHarrisWinRadius+1)^2 = 5x5` BOX sum (uniform weight; the original Harris paper and OpenCV's
`cornerHarris` use a smoother Gaussian window — see "Where this sits in the real world"). `M`'s
eigenvalues `lambda1 >= lambda2 >= 0` characterize the local structure exactly as "The problem"
describes qualitatively: `lambda1 ≈ lambda2 ≈ 0` -> flat; `lambda1 >> lambda2 ≈ 0` -> edge; `lambda1 ≈
lambda2 >> 0` -> corner.

### The Harris-Stephens response (why not eigenvalues directly)

Computing eigenvalues of a 2x2 matrix per pixel needs a square root (`lambda = (trace ± sqrt(trace^2 -
4*det))/2`) — cheap, but Harris & Stephens (1988) found an equivalent-in-spirit, SQRT-FREE score:

```
R = det(M) - k * trace(M)^2 = lambda1*lambda2 - k*(lambda1+lambda2)^2,   k in [0.04, 0.06] empirically
```

`R` is large and positive only when BOTH eigenvalues are large and comparable (a corner); `R` is
negative when one eigenvalue dominates (an edge); `R` is small (near 0) when both are small (flat).
This project uses `kHarrisK = 0.04` (`kernels.cuh`), the traditional value.

### The FAST-9 contiguous-arc test

FAST asks a purely COMBINATORIAL question, no derivatives, no matrix, no eigenvalues at all: sample the
Bresenham circle of 16 pixels at radius 3 around candidate corner `p` (`kFastCircleX/Y` in
`kernels.cuh`); `p` is a corner if there exists a set of `kFastArcLen = 9` CONSECUTIVE (circularly)
points that are ALL brighter than `I(p) + t` or ALL darker than `I(p) - t`, for threshold `t =
kFastThreshold = 20`. Formally: `p` is a corner iff

```
exists s in {0..15}: for all j in {0..8}, I(ring[(s+j) mod 16]) > I(p) + t     (the "bright" case)
   OR
exists s in {0..15}: for all j in {0..8}, I(ring[(s+j) mod 16]) < I(p) - t     (the "dark" case)
```

This is exactly a statement that a large fraction of a small disk's boundary is uniformly brighter or
darker than the center — a cheap, integer-only PROXY for "both structure-tensor eigenvalues are large"
that needs no gradients at all (`kernels.cu`'s `fast_score_kernel` computes it directly from raw
intensities).

### Intensity centroid orientation

For a keypoint at `(cx, cy)`, sum intensity-weighted offsets over a disk of radius `R =
kOrientPatchRadius = 15`:

```
m10 = sum_{(dx,dy): dx^2+dy^2<=R^2} dx * I(cx+dx, cy+dy)          m01 = sum_{...} dy * I(cx+dx, cy+dy)
theta = atan2(m01, m10)
```

This is the vector from the keypoint to the DISK'S intensity-weighted centroid (Rosin 1999) — a smooth,
low-noise proxy for "which way does this patch's brightness lean", used by ORB as the patch's canonical
orientation so the descriptor below can be made rotation-invariant.

### ORB descriptor bit test

Given `theta` (quantized to a bin, see "The GPU mapping"), and 256 precomputed sample-pair offsets
`(dx1,dy1,dx2,dy2)_k` ROTATED by the bin's angle and rounded to the nearest integer pixel:

```
bit_k = [ I(cx+dx1_k, cy+dy1_k) < I(cx+dx2_k, cy+dy2_k) ],   k = 0..255
```

packed 32 bits per `uint32_t` word, LSB-first (`OrbDescriptor::w[8]`, `kernels.cuh`).

### Hamming distance

For two 256-bit descriptors `a`, `b`: `d(a,b) = popcount(a XOR b)` — the number of differing bits, an
integer in `[0, 256]`. This is the discrete, binary analogue of Euclidean distance for float
descriptors (SIFT/SURF); "The algorithm" below works the complexity comparison in full.

## The algorithm

### FAST-9, step by step

1. For each pixel `p` with a full margin (`kDetectBorder = 3`, enough for the radius-3 ring): read the
   center intensity `I(p)` and the 16 ring samples.
2. **High-speed quick test**: check 4 evenly-spaced ring points (`kFastQuadIdx = {0,4,8,12}`); if fewer
   than 3 are consistently brighter (or darker) than `I(p) ± t`, no 9-length contiguous arc can exist
   (a pigeonhole argument: a run of 9 out of 16 must intersect at least 3 of any 4 points spaced 4
   apart), so `p` is rejected without the full test.
3. **Full arc test**: for each polarity that survived step 2, scan all 16 possible arc start positions;
   for each, check whether the 9 consecutive points all qualify, and if so record the WEAKEST point's
   margin (`|I(ring) - I(p)| - t`, always `> 0` for a qualifying arc). The corner's SCORE is the MAXIMUM
   such margin over all qualifying arcs and both polarities (0 if none qualify).
4. **Non-max suppression**: keep `p` only if its score is a STRICT local maximum among its 8 immediate
   neighbors' scores (`nms_select_fast_kernel`).
5. **Top-N selection**: sort surviving candidates by `(score desc, y asc, x asc)`, keep the top
   `kTopNFast = 300`.

Serial cost: step 1-3 is `O(1)` work per pixel (16 ring samples, up to `16*9*2 = 288` comparisons),
`O(W*H)` total. Step 4 is `O(1)` per pixel. Step 5 is `O(C log C)` for `C` candidates (typically a few
hundred, negligible next to `W*H = 65536`).

### Harris, step by step

1. **Sobel gradients** (`sobel_gradient_kernel`): a 3x3 stencil, `O(1)` per pixel.
2. **Structure tensor + response** (`harris_response_kernel`): a `5x5 = 25`-tap box sum of `(Gx^2,
   Gy^2, Gx*Gy)`, then `R = det - k*trace^2`. `O(25)` per pixel.
3. **Adaptive threshold**: `main.cu` computes the frame's peak response and floors candidates at `1% of
   peak` (`kHarrisRelThreshold`) — see "The GPU mapping" for why this is adaptive, not a fixed magic
   number.
4. **NMS + top-N**: identical shape to FAST's steps 4-5, over the float response map.

### ORB describe, step by step

1. **Orientation** (`orientation_kernel`): the intensity-centroid sum over a radius-15 disk (~709
   samples inside the disk out of a 31x31=961 bounding square), then one `atan2f`.
2. **Bin quantization** (`orient_to_bin`, host, `main.cu`): `theta` is folded into `[0, 2*pi)` and
   rounded to the nearest of `kOrientBins = 30` bins (12 degrees each) — matching OpenCV ORB's actual
   implementation choice (see "Where this sits in the real world").
3. **Descriptor** (`describe_kernel`): 256 lookups into the precomputed `RotatedOffset` table
   (`build_rotated_pattern_table`, `kernels.cuh`) for this keypoint's bin, two pixel reads and one
   comparison each, packed into 8 `uint32_t` words.

Serial cost: orientation is `O(R^2) ≈ O(709)` per keypoint; description is `O(256)` per keypoint (2
reads + 1 compare each); both `O(1)` in the number of keypoints (a few hundred), utterly dominated by
the detection stage's `O(W*H)` pixel sweep.

### Brute-force Hamming matching, step by step

For `nQuery` query descriptors and `nTrain` train descriptors: for each query, scan every train
descriptor, compute `popcount(q XOR t)` (`8` word-XORs + `8` popcounts = `O(1)` per comparison, since
descriptor width is fixed at 256 bits), track the running best and second-best. `O(nQuery * nTrain)`
total — genuinely the "naive" all-pairs algorithm, as opposed to an approximate structure (k-d tree,
LSH). **Why brute force is still the right choice here**: each comparison is `~16` cheap integer
instructions (vs. SIFT's 128-float L2 distance, which needs 128 multiply-adds PLUS a `sqrt`) — at
`nQuery, nTrain` in the hundreds (this project) to low thousands (a real VO frame), the whole all-pairs
sweep is microseconds on a GPU and often still competitive on a CPU; a k-d tree's `O(log n)` per query
only starts winning at query volumes this project's scale never reaches. The ratio test then costs one
divide per query; the mutual cross-check costs one array lookup.

## The GPU mapping

### Detection: per-pixel maps and stencils

`fast_score_kernel`, `sobel_gradient_kernel`, and `harris_response_kernel` are all launched with the
SAME 2-D geometry: a `16x16` thread block (`kBlock2D`, matching 01.02's precedent), grid sized to cover
`kW x kH` with a ragged-tail guard. Each thread owns exactly one output pixel and touches only GLOBAL
memory — no shared memory, no cross-thread communication, the simplest possible mapping for
embarrassingly-parallel per-pixel work. This is a deliberate teaching simplification, named honestly:
`harris_response_kernel`'s 25-tap box window is read INDEPENDENTLY by every thread even though adjacent
threads' windows overlap heavily (a `5x5` window shifted by one pixel shares 20 of 25 taps with its
neighbor) — the natural next optimization is a SHARED-MEMORY TILE: each block cooperatively loads a
`(16+4)x(16+4)` tile of `gx`/`gy` into shared memory once, then every thread's 25-tap sum reads from
that fast on-chip memory instead of re-fetching from global memory 25 times. At this project's `256x256`
scale the naive version already runs in microseconds, so the tiled version is left as README Exercise
material rather than implemented — but the REUSE argument is exactly why 01.02's 7x7 census stencil
(and most convolution-shaped kernels in this repository) eventually earn a tiled version in practice.

### NMS + compaction: stencil + atomics

`nms_select_fast_kernel`/`nms_select_harris_kernel` read a 3x3 neighborhood of the ALREADY-COMPUTED
score/response array (cheap — 8 extra reads, no recomputation) and, for surviving local maxima, append
to a shared output array via `atomicAdd` on a single counter. This is the standard GPU STREAM
COMPACTION pattern: many threads produce a variable, data-dependent number of outputs, and atomics hand
out unique output SLOTS without any thread needing to know in advance how many other threads will also
succeed. The trade-off: output ORDER is whatever order threads happen to finish in (not reproducible
run to run in general, though CUDA's typically-consistent scheduling makes it often-but-not-guaranteed
stable) — `main.cu` restores determinism with an explicit host-side sort immediately after downloading
the candidate list, which is also the exact convention `reference_cpu.cpp`'s CPU-side NMS applies
internally, making the two paths' FINAL keypoint lists comparable bit-for-bit (see "How we verify
correctness").

### Description: one thread per keypoint (a granularity SWITCH)

Detection processes `kW*kH = 65,536` independent pixels; description processes at most `kTopNFast =
300` keypoints. Continuing to launch pixel-grained kernels post-detection would waste 200x the
necessary parallelism launching threads for pixels that were already rejected. Switching to ONE THREAD
PER KEYPOINT (`orientation_kernel`, `describe_kernel`, 1-D grid, `kBlock1D = 128`) matches the actual
unit of remaining, independent work — this granularity shift (map-over-pixels -> map-over-a-much-
smaller-object-list) recurs throughout the repository any time a detection/segmentation stage feeds a
per-object stage (e.g., project 06.05's per-trajectory-sample kernels after a similar cardinality drop).
Each thread now does MORE serial work per thread (a ~709-sample disk sum, 256 comparisons) — the right
trade when there are only a few hundred independent items and thousands of threads available: full GPU
occupancy is no longer the limiting concern (a few hundred threads cannot fill a modern GPU's tens of
thousands of concurrent-thread capacity on their own), so the kernel is latency-bound, not throughput-
bound, and the goal shifts to "finish this modest, independent workload fast" rather than "maximize
warps in flight."

### Matching: a map with a serial inner loop, plus the popcount instruction

`hamming_match_kernel` is one thread per QUERY, each running a serial `O(nTrain)` loop internally — the
natural mapping when the OUTER dimension (queries) is embarrassingly parallel but the INNER dimension
(comparing against every train descriptor) has no further parallelism worth extracting at this scale
(nTrain in the hundreds; a parallel reduction across threads would add synchronization overhead that
dwarfs the tiny per-comparison cost it would save). `__popc(x)` is a SINGLE SASS instruction (`POPC`) on
every CUDA-capable GPU — genuinely a hardware population-count circuit, not a software loop the compiler
happens to recognize. `reference_cpu.cpp`'s `popcount32_portable()` (used via `kernels.cuh`, shared
since it is DATA-independent bit arithmetic, not the algorithmic core under test) is the "what would
this cost to hand-roll" answer CLAUDE.md's no-black-boxes rule asks for: the classic SWAR (SIMD-within-
a-register) bit-trick — pairwise-sum bit pairs, then nibbles, then bytes, then one multiply-and-shift to
horizontally sum the four byte-lane counts. Same answer, ~12 ALU instructions instead of 1.

### Why the Harris threshold is adaptive, not a fixed constant

Harris responses on this project's committed scene span roughly `[0, 2.5e13]` (see "Numerical
considerations") — an absolute pre-NMS floor tuned for THIS scene would be meaningless on a differently-
lit or differently-textured one. `main.cu` instead computes each frame's OWN peak response and floors
candidates at `kHarrisRelThreshold = 1%` of that peak — precisely the strategy OpenCV's
`goodFeaturesToTrack` uses via its `qualityLevel` parameter (a fraction of the frame's own best corner
score, not an absolute number), so this project's threshold choice is not an ad hoc simplification but
a reproduction of standard practice.

## Numerical considerations

- **FAST is exact-integer end to end.** Intensities are `uint8_t`, the threshold `t` is an `int`, every
  comparison and margin computation is integer arithmetic — there is categorically no rounding anywhere
  in `fast_score_kernel`/`fast_score_cpu`, which is exactly why this project can (and does) demand
  BIT-EXACT GPU-vs-CPU agreement for FAST, not just "close enough".
- **Harris responses have an enormous dynamic range.** `Gx, Gy` reach `~1020` (Sobel weights up to 4,
  intensity up to 255); the 25-tap box sums `Sxx, Syy` reach `~2.6e7`; `det(M) = Sxx*Syy - Sxy^2` — a
  PRODUCT of two already-large numbers — reaches `~1e13` at the sharpest corners on this project's
  committed scene (measured peak: `1.09e13`). float32 carries only ~7 significant decimal digits, so
  the ABSOLUTE precision floor at `1e13` is itself around `1e6` — this project's `VERIFY(harris)` gate
  therefore compares GPU vs. CPU with a RELATIVE tolerance (`kTolHarrisResponse = 2e-3`, measured
  `5.2e-4`), not an absolute one; `main.cu`'s `max_relative_diff_float()` comment derives this in full.
  `reference_cpu.cpp`'s Harris twin ALSO deliberately accumulates in `double` (an independent numerical
  PATH from the GPU's `float` accumulation, not merely an independent implementation of the same path)
  — the two are expected, correctly, to diverge by a few parts in `10^4`, and the gate says so honestly
  rather than papering over it with an oversized absolute number.
- **Orientation isolates its one source of divergence.** The intensity-centroid sums `m10, m01` are
  accumulated in exact INTEGER arithmetic (small `dx, dy, I` values; `~700` terms, well within `int32`
  range) — the SOLE floating-point operation in the entire orientation stage is the final `atan2f()`
  call, which is why any GPU/CPU disagreement here can ONLY come from the two platforms' distinct
  `atan2` implementations (measured: exactly `0` disagreement, to float32 printable precision, on the
  committed sample — the two happen to agree perfectly here, though the tolerance exists precisely
  because a different GPU architecture's `atan2f` need not).
- **Descriptors are bit-exact by DESIGN, not by luck.** `kernels.cuh`'s header works through why: the
  256-bin-quantized orientation used for descriptor rotation is SHARED data (computed once, fed to both
  the GPU kernel and the CPU twin), so the only remaining computation is an integer pixel-intensity
  comparison — no float anywhere in the hot loop. The 12-degree BIN WIDTH is ~20x this project's
  orientation tolerance (`0.01 rad ≈ 0.57 deg` vs. `12 deg / 2 = 6 deg` half-bin-width), so a bin
  disagreement between GPU and CPU angles is not just unlikely but actively CHECKED for and asserted
  never to happen (`main.cu`'s "verify(orientation bin agreement)" line) — descriptor bit-exactness is
  therefore guaranteed by construction on this project's data, not merely observed.
- **Hamming distances are pure integers.** `popcount` and the best/second-best running reduction never
  touch a float; GPU and CPU are required (and, measured, do) agree exactly.
- **No angle-wrapping subtlety beyond the standard one.** `atan2f`'s native range is already `(-pi,
  pi]`; the one place this project must wrap explicitly is comparing TWO angles (`main.cu`'s
  `wrap_angle_rad()`, used for both the orientation-tolerance check and the rotation-recovery gate's
  matched-pair delta), since a naive subtraction of two angles near `+pi`/`-pi` can report a spuriously
  large difference for two angles that are actually close together going "the other way around the
  circle".

## How we verify correctness

Two tiers, per CLAUDE.md's ruling (see `reference_cpu.cpp`'s header for the full statement): GPU-vs-CPU
TWINS catch parallelization bugs (wrong indexing, races, stale memory); INDEPENDENT GATES catch bugs a
twin comparison structurally cannot, because both "twins" could share the same conceptual mistake.

**Twins** (measured on the committed sample, RTX 2080 SUPER, sm_75, Release):

| Twin | Type | Tolerance | Measured |
|------|------|-----------|----------|
| FAST score map (A, B) | bit-exact | 0 | 0 |
| FAST final keypoint list (A, B) | bit-exact (positions + score) | 0 | identical |
| Sobel gradients (A) | bit-exact (exact-integer floats) | 0 | 0 |
| Harris response map (A) | relative tolerance | 2e-3 | 5.2e-4 |
| Orientation angle (A, B) | tolerance, radians | 0.01 | 0.000000 |
| Orientation bin agreement (A, B) | bit-exact (0 mismatches required) | 0 | 0 |
| ORB descriptors (A, B) | bit-exact | 0 bits | 0 / 34560, 0 / 44288 |
| Hamming distances (both directions) | bit-exact | 0 | 0 |

**Gates** (independent of any twin — see each one's rationale in README "Expected output" and
`main.cu`'s per-gate comments):

| Gate | What it catches that a twin cannot | Floor/Ceiling | Measured |
|------|-------------------------------------|----------------|----------|
| `ground_truth_transform` | A shared bug in BOTH kernels.cu and reference_cpu.cpp (e.g. a wrong sign in the descriptor rotation) would still pass every twin above; this gate is checked against the KNOWN transform, retyped a THIRD time, independently, in `main.cu`'s `forward_transform()` | >= 90% inliers within 5.0 px | 92.3% (60/65) |
| `rotation_recovery` | Same class of shared-bug blind spot, applied to orientation specifically | error <= 1.0 deg | 0.49 deg |
| `repeatability` | Bypasses descriptors and matching ENTIRELY — a purely geometric check of detection alone | >= 50% | 63.7% (86/135) |
| `negative_control` | Proves the OTHER THREE gates are not vacuously true (a matcher that accepted everything, or a gate that always reports "close enough", would also pass a real pair) | <= 10% inliers | 0.0% (0/17) |

**Why the negative control matters most.** A pipeline with a subtle bug that makes EVERY match "pass"
the ground-truth-transform gate regardless of correctness (e.g., a gate that was accidentally checking
`err <= 1e18` instead of `err <= kGtPixelTol`) would sail through `ground_truth_transform` and
`rotation_recovery` — the negative control is what would catch it, because matching `scene_a.pgm`
against the entirely unrelated `neg_scene_c.pgm` and STILL reporting a high "inlier" fraction would
immediately reveal the gate itself is not discriminating true from false correspondences. Measuring
`0/17 = 0.0%` here is the strongest single piece of evidence in this project that Gate 1's `92.3%` is
real geometric recovery, not an artifact of a permissive check.

**A concrete example of what went into calibrating these numbers honestly** (CLAUDE.md §8: never
fabricate): an EARLIER, non-anti-aliased version of `scripts/make_synthetic.py`'s scene renderer (hard-
edged shapes, no supersampling) measured `repeatability = 4/46 ≈ 8.7%` and `ground_truth_transform =
3/5 = 60%` — both far below any reasonable floor. Root cause: FAST/Harris corner LOCATIONS near a
single-pixel-sharp step edge are hypersensitive to exactly which side of the boundary each integer pixel
falls on, and a 12-degree rotation (`scene_b.pgm`'s ground truth) shifts that boundary's exact pixel
alignment enough to move or destroy many corners entirely. Adding 4x4 supersampled anti-aliasing (the
committed generator's actual behavior — see `data/README.md`) fixed this: repeatability jumped to
`63.7%` and the ground-truth inlier fraction to `92.3%` on the SAME scene layout, same detector
parameters, same everything except the anti-aliasing. This is recorded here, with both numbers, rather
than silently tuning the thresholds to fit — the repository's honesty rule applies to the DEVELOPMENT
process, not just the final constants.

## Where this sits in the real world

- **FAST**: production systems (ORB-SLAM3, OpenCV's `FastFeatureDetector`, PX4's optical-flow modules)
  replace this project's hand-written "high-speed test" pre-filter with a TRAINED DECISION TREE (ID3,
  learned offline over a large corpus of the 16 binary ring comparisons) that reorders and prunes
  comparisons far more aggressively — often deciding corner/non-corner in 2-3 comparisons on average
  instead of this project's textbook 4-then-16. The corner SET found is the same (or very close);
  only the SPEED of rejecting non-corners differs.
- **Harris**: OpenCV's `cornerHarris` uses a Gaussian-weighted window (via `cv::cornerEigenValsAndVecs`
  or a separable Gaussian blur pass) instead of this project's box window, for a directionally-smoother,
  less axis-biased response; `goodFeaturesToTrack` layers Harris (or the related Shi-Tomasi
  min-eigenvalue criterion) with a MINIMUM-DISTANCE spatial suppression this project's plain 3x3 NMS
  does not attempt.
- **ORB descriptors**: OpenCV's real `cv::ORB` uses a LEARNED 256-pair sampling pattern (`bit_pattern_
  31_` in `orb.cpp`), chosen offline to maximize pairwise decorrelation and per-test variance across a
  large training corpus — this project's isotropic-random pattern (kernels.cuh's `build_orb_base_
  pattern`) is an honest, fully-functional substitute that reproduces every MECHANICAL step (rotate,
  round, sample, compare, pack) faithfully, but would measurably under-perform the trained pattern's
  matching precision on real, non-synthetic imagery.
- **Matching**: production visual-SLAM systems (ORB-SLAM3) do not brute-force match every frame against
  every other frame; they use a BAG-OF-WORDS vocabulary (DBoW2/DBoW3) built on the SAME ORB descriptors
  to retrieve a small candidate set for loop closure, and use brute-force/grid-restricted matching only
  for consecutive-frame tracking, where `nTrain` is small enough (as in this project) for brute force to
  already be the fastest correct answer.
- **The field's current frontier**: learned, end-to-end alternatives (SuperPoint for joint detection +
  description, SuperGlue/LightGlue for learned matching with global context) now outperform hand-crafted
  FAST/ORB/brute-force pipelines on wide-baseline and low-texture scenes specifically BECAUSE they are
  trained end-to-end on real data rather than hand-designed — but they cost a neural-network forward
  pass per frame (real GPU/NPU budget) where this entire project's three stages together cost
  microseconds; classical pipelines like the one built here remain the default choice for compute-
  constrained platforms (a small quadrotor's companion computer, an embedded AMR) where every millijoule
  and millisecond is budgeted, and remain an essential fallback path even in systems that also carry a
  learned front end.
- **`[R&D]` scoping note**: this catalog bullet carries no `[R&D]` tag (it is `★` beginner-difficulty),
  so no reduced-scope teaching-version disclaimer applies here — every named component (FAST, Harris,
  ORB, brute-force Hamming matching) is implemented in full, not a stripped-down stand-in.
