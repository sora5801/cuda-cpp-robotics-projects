# 01.08 — HDR exposure fusion + tone mapping for outdoor robots: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

### The photometry of an outdoor scene

Illuminance (how much light falls on a surface) is measured in **lux** (lm/m²). A handful of real-world
reference points, useful because they set the scale this project's synthetic scene is *proportional* to
(see the honesty note at the end of this section):

| Condition | Illuminance |
|---|---|
| Direct sunlight | ~30,000-130,000 lx |
| Overcast daylight / open shade | ~1,000-20,000 lx |
| A shaded loading dock, under a building overhang | ~500-2,000 lx |
| Under a parked vehicle, direct light fully blocked | ~5-50 lx |
| Deep twilight | ~1-10 lx |

Radiance reflected off a diffuse surface is proportional to illuminance times the surface's albedo
(reflectance) — so a single outdoor scene, in one frame, can span sunlit concrete (~40% albedo, full sun)
against the shadow under a vehicle (indirect skylight only, low illuminance) by **4-5 orders of
magnitude**, exactly the ratio this project's scene targets (see `scripts/make_synthetic.py`'s
`R_SUN`/`R_SHADOW` constants, whose ratio is `1e5`).

### Why the sensor cannot just "see" all of it

A camera sensor converts photons into charge, then charge into a digital code value, with two hard
physical floors that bound its usable range in a *single* exposure:

- **Full well capacity** — a pixel's charge well saturates at some maximum electron count. Above the
  corresponding irradiance x exposure-time product, more light produces no more signal: the pixel is
  **clipped white**.
- **Read noise floor** — below some electron count, the signal is smaller than the noise the sensor's
  readout electronics themselves inject. Signal below that floor is indistinguishable from noise: the
  pixel is effectively **clipped black**, even though its analog value is technically nonzero.

The ratio between these two floors is the sensor's **dynamic range**, typically quoted in dB
(20*log10(ratio)) or stops (log2(ratio)). A good machine-vision CMOS sensor manages roughly **60-70 dB**
(~10-12 stops, a ratio of ~1,000-4,000x) in a *single* exposure — nowhere near the outdoor scene's
100,000x (100 dB) span computed above. This ~30-40 dB gap between what one exposure can capture and what
an outdoor scene actually contains is the entire reason HDR techniques exist, and it is a genuine,
physical gap — not a processing artifact to be "fixed" by better software alone.

### What one exposure controls, and what it does not

For a fixed scene, the only knob a single exposure has is **exposure time** `t` (shutter speed) and
**gain** (ISO) — this project varies only `t`, holding gain fixed, the simpler and more common bracketing
convention. Exposure (the physical quantity a sensor pixel actually integrates) is

```
X = R * t
```

where `R` is the scene radiance reaching that pixel (relative units here — see the honesty note below)
and `t` is the exposure time in seconds. Doubling `t` doubles `X` for every pixel uniformly — it *shifts*
which 10-12 stops of the scene fall inside the sensor's usable window, but never *widens* that window.
Four different `t` values, as this project uses, therefore sample four different (overlapping) 10-12-stop
windows of the same underlying scene — the entire mechanism bracketing exploits.

### Honesty about the synthetic scene's units

`scripts/make_synthetic.py`'s radiance tiers (`R_SHADOW=2` ... `R_SUN=200000`, relative units) are chosen
to reproduce the **order-of-magnitude ratios** discussed above between shadow/shade/concrete/sky/sun —
not a calibrated photometric simulation with real lux values. This project is about the *algorithms*
that recover and compress dynamic range, not about radiometric calibration; where a real lux number would
require sensor-specific calibration data this project does not have, the honest choice is a clearly-
labeled relative scale, not a fabricated absolute one.

### Engineering constraints a real camera imposes

- **Bracket capture time.** Four sequential exposures (this project: up to 1/8 s for the longest) take
  on the order of ~150 ms total — three orders of magnitude slower than a single frame at 30-60 Hz. A
  moving scene or a moving robot introduces real inter-frame motion during that window (see
  [Numerical considerations](#numerical-considerations) and README "Limitations & honesty").
- **Vibration and rolling shutter.** A vehicle-mounted camera vibrates; if the sensor uses a rolling
  (not global) shutter, each row of each bracketed frame is captured at a slightly different instant,
  compounding the motion problem bracketing already has (see `01.10-rolling-shutter-correction`, a
  sibling project in this repository).
- **Thermal drift.** Sensor read noise and dark current both increase with temperature; an outdoor camera
  in direct sun can drift several degrees over a capture sequence, subtly changing the noise floor
  between the first and last bracketed exposure — a second-order effect this project's fixed noise model
  does not simulate, but real HDR pipelines must tolerate.

## The math

### Notation

- `(x, y)` — pixel coordinates, `x` in `[0, W)`, `y` in `[0, H)`, `W=160, H=120` (`kW, kH` in
  `kernels.cuh`). Row-major raster convention, **not** a robot body frame — see the frame-convention note
  at the end of this section.
- `R(x, y)` — true scene radiance at pixel `(x, y)`, relative units, `R > 0`.
- `t_j` — the exposure time of bracketed frame `j`, `j = 0..3` (`kExposureTimes[j]`), seconds.
- `X = R * t_j` — the exposure a pixel integrates in frame `j`.
- `Z_j(x, y)` — the 8-bit code value (`0..255`) frame `j` records at `(x, y)`.
- `f: X -> Z` — the camera response function (CRF), the (unknown, to be recovered) map from exposure to
  code value. `g = ln(f^{-1})`, i.e. `g(Z) = ln(X)` — Debevec & Malik's own convention, adopted here.

### The camera response function

This project's **synthetic, known** CRF (`scripts/make_synthetic.py`) is a Naka-Rushton / Michaelis-
Menten saturating curve — the same functional form used to model **photoreceptor response** in vision
science (a real, physically-motivated choice, not an arbitrary polynomial):

```
z_frac(X) = X^gamma / (X^gamma + S^gamma)          in [0, 1),  Z = round(255 * z_frac)
```

with `gamma = 0.85` and half-saturation exposure `S = 3.0` (`CRF_GAMMA`, `CRF_S_HALF`). For `X << S` this
is approximately a power law `z_frac ~ (X/S)^gamma` (the classic photographic "gamma" region); for
`X >= S` it saturates smoothly toward 1 (the "shoulder") — no hard knee anywhere, so every code value is
reachable, and the curve is analytically invertible:

```
X = S * (z_frac / (1 - z_frac))^(1/gamma),     g_true(z) = ln(X)
```

Note the singularities at `z_frac -> 0` (`X -> 0`, `g -> -infinity`) and `z_frac -> 1` (`X -> infinity`,
`g -> +infinity`) — a **real** property of an ideal noiseless response, not a bug: no finite exposure
ever produces an *exact* code value of 0 or 255 in the continuous model, only 8-bit rounding does. This
is exactly why `main.cu`'s `crf_recovery` gate excludes `z=0` and `z=255` (and, empirically, a margin
around them — see [How we verify correctness](#how-we-verify-correctness)) from its comparison.

### Debevec-Malik CRF recovery (`gsolve`)

Given `P` sample pixels observed across all `N=4` exposures, Debevec & Malik (1997) recover `g` and every
sample's log-irradiance `ln(E_i)` by minimizing the weighted sum of squared residuals

```
sum_{i,j} w(Z_ij) * [ g(Z_ij) - ln(E_i) - ln(t_j) ]^2
  + lambda * sum_{z=1}^{254} w(z) * [ g(z-1) - 2*g(z) + g(z+1) ]^2
```

The first term says: "for every sample `i` and exposure `j`, the response's log-exposure estimate should
equal the sample's own log-irradiance plus that exposure's known log-time" — the *data* term, directly
from `X = R*t` and `g(Z) = ln(X)`. The second term is a **smoothness prior** on the *second derivative* of
`g` (a discrete curvature penalty, weighted `lambda`): without it, `z` values touched by only one or two
samples are almost entirely unconstrained and the solve can produce wild, non-physical wiggles; the prior
says "prefer the smoothest curve consistent with the data." `w(z) = min(z, 255-z)` is Debevec & Malik's
**hat weight**: zero at `z=0` and `z=255`, peaking at `z=127/128` — the data-term downweights
near-clipped samples (least reliable), and the smoothness term downweights the prior itself in the same
regions (where data, if any, should be trusted over the prior).

**The scale ambiguity.** Replacing `g(z) -> g(z) + c` and `ln(E_i) -> ln(E_i) + c` for every `z` and `i`
leaves the data-term residual `g(Z) - ln(E_i) - ln(t_j)` **exactly unchanged** — the system is rank-
deficient by exactly one dimension. Debevec & Malik (and this project) fix it with one extra equation,
**pinning** `g(128) = 0` (`crf_solve_debevec`'s "pin" row). This removes the singularity and makes the
solve well-posed, but the resulting `g` is only guaranteed to agree with any *other* absolute log-
exposure scale (such as `scripts/make_synthetic.py`'s own units) up to that same additive constant `c` —
see [Numerical considerations](#numerical-considerations) for how this project measures and corrects for
it when grading against ground truth.

`crf_solve_debevec` builds this as **weighted normal equations** (`A^T A x = A^T b`) rather than the
paper's own SVD-based least-squares solve: since every equation touches at most 3 unknowns (a data term
touches `g[z]` and `lnE[p]`; a smoothness term touches three neighboring `g[]` entries), `A^T A` and
`A^T b` are accumulated directly, in `O(1)` work per equation, without ever materializing the sparse
design matrix `A` — a standard technique, and exactly the "small dense system, Gaussian elimination"
approach 33.01-batched-small-matrix-linalg teaches for problems this size (see
[The GPU mapping](#the-gpu-mapping) for why this is a *host*, not *device*, computation).

### Radiance merge

Given the recovered `g`, every pixel's radiance is estimated by combining all `N` exposures' independent
log-radiance estimates, again hat-weighted:

```
ln(E) = [ sum_j w(Z_j) * (g(Z_j) - ln(t_j)) ] / [ sum_j w(Z_j) ]
```

### Reinhard global tone mapping

Photographic Tone Reproduction (Reinhard et al. 2002) first estimates the scene's overall brightness as
the **log-average luminance**

```
L_avg = exp( (1/n) * sum_i ln(delta + E_i) )
```

(`delta` a small constant avoiding `ln(0)`) — the **geometric**, not arithmetic, mean: radiance is
naturally log-distributed (this scene spans *decades*, and a handful of very bright pixels would swamp
an arithmetic mean, while the geometric mean weights every stop of brightness equally, matching how human
brightness perception itself is roughly logarithmic). The scene is then rescaled to a target "middle
gray" **key** (photographic convention `key = 0.18`, this project's `kReinhardKey`) and squashed into
`[0, 1)`:

```
L_scaled = (key / L_avg) * E,        L_d = L_scaled / (1 + L_scaled)
```

`L_d` is **strictly increasing** in `E` for `E >= 0` (its derivative `1/(1+L_scaled)^2 * key/L_avg` is
always positive) and strictly bounded in `[0, 1)` — by construction, not by clamping, which is exactly
what `main.cu`'s `tone_map_range` gate checks. (Reinhard's paper also derives an *extended* form with an
explicit white-point burnout parameter, `L_d = L_scaled*(1+L_scaled/Lwhite^2)/(1+L_scaled)`; this project
deliberately uses the simple form specifically so the `[0,1)` guarantee holds without a final clamp — see
[Numerical considerations](#numerical-considerations).)

### Local tone mapping (bilateral-grid-lite base/detail split)

Global tone mapping applies ONE curve to every pixel — it cannot simultaneously make a dark region bright
enough to see AND keep a bright region from washing out, if both need *different* amounts of compression.
Local tone mapping addresses this by splitting the image into a **low-frequency base** (compressed
aggressively) and a **high-frequency detail** (left mostly alone), then recombining:

```
logL = ln(E + eps)
base = EXPAND(EXPAND(REDUCE(REDUCE(logL))))      (a 2-level Gaussian low-pass, re-expanded to full res)
detail = logL - base
base_compressed = compression_factor * base + (1 - compression_factor) * mean(coarsest level)
composite = base_compressed + detail_boost * detail
output = min_max_normalize(composite)             (exact, guarantees [0, 1])
```

`compression_factor < 1` (this project: `0.35`) shrinks the base layer's *range* around the scene's own
mean log-brightness — the base carries the ~11-12 log-unit spread of the whole scene, and after
compression carries roughly `compression_factor` times that spread, tone-mappable in one pass. The detail
layer, which carries local contrast/texture rather than absolute brightness level, is left at (or boosted
slightly above) its original magnitude, so texture that would be crushed by the base's own compression is
restored. This is the project's honestly-labeled simplification of a *true* bilateral filter's edge-aware
base/detail split — see [Numerical considerations](#numerical-considerations) for the difference.

### Mertens exposure fusion

Mertens, Kautz & Van Reeth (2007) skip radiance and the CRF entirely. For each exposure `j`, a per-pixel
**quality** weight combines (this project keeps 2 of the published 3 terms — see the note below):

```
contrast_j(x,y)       = | laplacian3x3( img_j )(x,y) |
wellexposed_j(x,y)    = exp( -(img_j(x,y) - 0.5)^2 / (2*sigma^2) )
raw_weight_j           = contrast_j ^ wc  *  wellexposed_j ^ we
```

(`img_j` is `Z_j / 255`, the normalized *display* value — Mertens never converts to radiance at all,
which is precisely why it needs no CRF). Weights are normalized to sum to 1 at every pixel, then blended
**per pyramid level**, using a **Laplacian pyramid** for the images (each level a band-pass "detail"
signal) and a **Gaussian pyramid** for the weights (each level a low-pass, smoothly-varying blend
coefficient):

```
fused_level_l = sum_j  Gaussian(W_j)[l]  *  Laplacian(img_j)[l]
output = collapse(fused_levels)     (reconstruct top-down: coarsest first, EXPAND + add each finer band)
```

Blending band-pass image content weighted by low-pass (smoothly varying) coefficients, rather than
blending the raw images weighted by the (sharp, discontinuous) full-resolution weight map directly, is
the entire mechanism by which multiscale blending avoids the seams/halos a naive single-scale weighted
average produces — see [How we verify correctness](#how-we-verify-correctness) for how `halo_check`
makes this quantitative.

**On the dropped saturation term:** Mertens' third weight component is `std(R, G, B)` at each pixel —
undefined for a single-channel image. This project computes only `contrast x well-exposedness`
(`wc=1.0`, `we=2.0` — sharper than Mertens' own defaults, see `main.cu`'s comment on why this project's
very dim shadow region needed a more decisive weight) rather than inventing a fake substitute.

### Frame/units convention note

Unlike the rest of this repository's robot-body-frame conventions (`docs/SYSTEM_DESIGN.md`: right-handed,
`T_parent_child`, SI units), this project's `(x, y)` are plain 2D pixel-raster coordinates, row-major,
`y` increasing downward — the standard image convention, and a genuinely different thing from a physical
frame. Radiance/exposure values are relative synthetic units (see "Honesty about the synthetic scene's
units" above), not SI-calibrated photometric quantities.

## The algorithm

### Step-by-step, both paths

```
                         ┌─────────────────────────────┐
  4 LDR exposures  ─────▶│ crf_solve_debevec (host,     │
  (Z_0..Z_3, t_0..t_3)   │ ONE-TIME, shared GPU+CPU)    │
                         └──────────────┬──────────────┘
                                        │ g[256]
              ┌─────────────────────────┼─────────────────────────┐
              │ PATH A                  │                 PATH B  │
              ▼                         │                         ▼
   radiance_merge_kernel                │           u8_to_unit + mertens_raw_weight
   (per-pixel map)                      │           (per exposure) + normalize_weights4
              │                         │                         │
   ┌──────────┴──────────┐              │              ┌──────────┴──────────┐
   ▼                     ▼              │              ▼                     ▼
 run_reinhard_global   run_local_tonemap│         weighted_sum4          multiscale blend
 (reduction + map)     (pyramid split)  │         (naive, full-res)      (Laplacian pyramid,
                                        │                                 Gaussian-pyramid weights)
```

### Complexity

Every per-pixel map and stencil kernel is `O(W*H)` per call (constant work per pixel); a full Gaussian
pyramid is `O(W*H)` total across all levels (a geometric series: `WH + WH/4 + WH/16 + ... < (4/3)*WH`).
`crf_solve_debevec`'s normal-equations solve is `O((256+P)^3)` for the Gaussian elimination
(`P = 64` sample pixels here, so `n=320`, ~3.3e7 double-precision operations) plus `O(P*N)` to
*accumulate* those equations — negligible next to the per-pixel work at any realistic image size, and run
exactly **once** per capture (not once per pixel, not once per frame of a video stream — see
[The GPU mapping](#the-gpu-mapping)).

### Key data structures

- The exposure stack: `N=4` flat `uint8_t[W*H]` arrays (device and host).
- The CRF table: `float[256]`, `g[z] = ln(exposure)`, in GPU `__constant__` memory and a plain host array.
- Pyramid levels: `kNumLevels=3` flat `float[]` arrays per "pyramid" (sized `level_w(l)*level_h(l)`),
  indexed by exposure `j` and level `l` where PATH B needs a full `[4][3]` grid of them (images, weights,
  and derived Laplacian bands) — see `kernels.cu`'s `run_mertens_gpu` for the full bookkeeping.

## The GPU mapping

### Thread-to-data mapping, per kernel family

- **Per-pixel maps** (`radiance_merge_kernel`, `reinhard_map_kernel`, `affine_kernel`, `log_kernel`,
  `u8_to_unit_kernel`) — thread `i = blockIdx.x*blockDim.x + threadIdx.x` owns output element `i`. No
  shared memory: no data is reused between threads. `kBlock1D = 256` threads/block, `grid1d(n) =
  ceil(n/256)` blocks — this project's largest buffer is `kN = 19,200` elements (`75` blocks), far under
  any grid-size limit, so unlike `docs/PROJECT_TEMPLATE`'s SAXPY placeholder, no grid-stride loop is
  needed (every kernel here covers its whole buffer in one launch).
- **2D stencils** (`gaussian_reduce_kernel`, `bilinear_expand_kernel`, `mertens_raw_weight_kernel`) —
  thread `(ox, oy)` owns one OUTPUT pixel and reads a small (5x5 or 3x3) neighborhood of the input,
  clamped to the image border. `kBlock2D = 16` (256 threads/block, same occupancy target as `kBlock1D`).
  No shared-memory tiling: at 160x120 (and smaller pyramid levels), the working set already fits
  comfortably in L1/L2 cache, so the classic "load a halo'd tile into shared memory once, reuse it across
  the block's threads" optimization (the textbook next step for a stencil this shape) would save little
  at this image size — a documented, deliberate simplification (see README "Exercises" for the separable-
  convolution follow-up, a related but distinct optimization).
- **Reduction** (`luminance_log_sum_kernel`) — the classic two-phase pattern: each thread loads and
  transforms (`ln`) its element into **shared memory** (fast, on-chip, and the ONLY memory space every
  thread in a block can both read and write to reach *other* threads' partial results); a binary-tree
  reduction halves the live thread count each step until `partial[0]` holds the block's total; then ONE
  `atomicAdd` per block (not per thread) into a single global `double` accumulator. Contention is across
  `~75` blocks, not `19,200` threads — negligible.
- **Batched-solve** (`crf_solve_debevec`) — **host-only**, deliberately. A `~320x320` dense solve run
  ONCE per capture has nowhere near enough work to amortize a kernel launch (microseconds of launch
  overhead against a solve that itself takes ~1.3 ms on a single CPU core — see
  `demo/expected_output.txt`'s `[time]` line), and the *iteration count* here is small (320 columns of
  elimination), not the *massively parallel, repeated-many-times* shape that makes a computation worth
  moving to the GPU. Contrast this with 33.01-batched-small-matrix-linalg, whose entire point is
  *many* small matrices solved *simultaneously* — that shape genuinely parallelizes; a single one-time
  calibration solve does not.

### Memory hierarchy choices

- **`__constant__` memory** for the 256-entry CRF table (`g_crf_table`, `kernels.cu`) — read by every
  thread in `radiance_merge_kernel`, small enough (1 KiB) to sit entirely in the constant cache. Honestly
  assessed: constant memory is *fastest* under a true broadcast (every thread in a warp reading the
  *same* address), which this is not quite (neighboring pixels usually have similar but not identical
  brightness) — but the constant cache still comfortably outperforms an equivalent global-memory lookup
  once warmed, and is the idiomatic home for a small per-pixel LUT in CUDA image pipelines.
- **Shared memory** for the reduction's per-block partial sums (see above) — the textbook use case:
  fast, on-chip, block-scoped communication.
- **Global memory** for every image/pyramid-level buffer — no reuse pattern here justifies anything more
  specialized (texture memory's hardware bilinear interpolation would be a natural fit for
  `bilinear_expand_kernel`'s access pattern — a documented exercise, not implemented, to keep this
  project's memory-space surface small and legible).

## Numerical considerations

- **Precision.** Every kernel computes in FP32; `crf_solve_debevec`'s Gaussian elimination and
  `luminance_log_sum_kernel`'s reduction accumulator use `double` specifically because they *sum many
  small quantities* (a classic float-accumulation-error scenario — see CLAUDE.md §12's "atomics reorder
  float sums" note) — a single FP32 reduction over `19,200` log-luminance terms, or a `320`-step Gaussian
  elimination, both risk visible rounding drift in FP32 that `double` comfortably avoids at negligible
  cost.
- **The scale ambiguity, quantified.** As derived in [The math](#the-math), Debevec-Malik recovers `g`
  up to an additive constant. On this project's reference run, that constant (`crf_offset`, `main.cu`)
  measures **~1.09 ln-exposure units** (`radiance_scale = exp(1.09) ~= 2.97x`) — i.e. the *raw* recovered
  radiance is consistently about 1/3 of the ground truth's absolute scale, uniformly across every pixel.
  This is NOT an error to chase down: it is exactly the rank-deficiency the pin `g(128)=0` was always
  going to leave, since that pin fixes the recovered curve's OWN zero point, not scripts/make_synthetic.
  py's absolute radiance units. `main.cu`'s `crf_recovery` and `radiance_reconstruction` gates measure
  and correct for it explicitly (computing the offset as the mean gap between recovered and true `g` over
  a well-supported `z` range, then applying `exp(offset)` before comparing radiance) — the *residual*
  error after that correction (measured: max `~0.06` ln-units in `z in [10,245]`, mean relative radiance
  error `~4%`) is what actually reflects the algorithm's fidelity.
- **Why the tail `z` values are excluded from the CRF gate.** `g_true` is analytically singular at
  `z=0/255` (see [The math](#the-math)), and even inside `[0,5)`/`(250,255]` the *recovered* curve is
  reconstructed almost entirely by the smoothness PRIOR rather than data (few samples land there with
  meaningful hat weight) — comparing there tests the prior's shape assumption, not the recovery
  algorithm. Measured residual at `z=5` alone is `~0.18` (3x the `[10,245]` range's worst residual of
  `~0.06`) purely from this effect.
- **Angle wrapping, quaternion drift, stiff ODEs:** not applicable — this project has no orientation
  state or continuous-time integration.
- **Ill-conditioned Jacobians:** not applicable in the robotics-kinematics sense, but the CRF solve's
  normal-equations matrix `A^T A` IS a place conditioning matters: the smoothness prior's weight
  `lambda=20` exists specifically to keep `A^T A` well-conditioned in `z` regions with few or zero data
  samples (pure data-only rows there would leave the corresponding `g[z]` entirely unconstrained, a
  singular sub-block); Debevec & Malik's own paper reports `lambda` in the same range.
- **Monotonicity noise.** The tone-mapped output along this project's noise-free calibration strip is
  *expected* to be monotonically non-decreasing (radiance IS monotonic there by construction — see
  `scripts/make_synthetic.py`), but the RECOVERED CRF, fit from noisy pixel data, is not perfectly smooth
  at every `z` — small (measured: up to `~0.019` in `[0,1)` output units, roughly 5 out of 255 8-bit
  levels) backward wiggles are visible and expected; `main.cu`'s `tone_map_range` gate allows a small,
  measured-then-margined epsilon rather than demanding bit-perfect monotonicity, which would be dishonest
  given real sensor noise is part of this project's own synthetic model.
- **Simplified pyramid EXPAND.** `bilinear_expand_kernel` uses bilinear interpolation rather than Burt &
  Adelson's textbook zero-insertion + 4x-kernel EXPAND. For a SINGLE image's own Laplacian pyramid,
  reconstruction with a bilinear EXPAND is still lossless in principle (REDUCE/EXPAND applied consistently
  cancel by construction — see the derivation implicit in `run_mertens_gpu`'s Laplacian-band construction:
  `L[l] = G[l] - EXPAND(G[l+1])`, then reconstruction `R[l] = FL[l] + EXPAND(R[l+1])` is exact for a
  single un-fused image). The visible ringing this project's `halo_check` gate is designed around comes
  specifically from FUSING different exposures' Laplacian bands with different per-exposure weight
  mixtures at each scale — a genuine, well-documented property of multiscale blending at very extreme
  (near-100x+) raw brightness discontinuities, which is *why* `halo_check` deliberately scans a gentler
  (3x contrast) boundary rather than the scene's most extreme edge (see README "Limitations & honesty"
  and `main.cu`'s `kHaloScanY` comment for the full reasoning and the specific numbers that motivated it).
- **Determinism.** Every RNG in this project (`scripts/make_synthetic.py`'s texture/noise) is a seeded
  `xorshift32` stream (seed 42), never `std::uniform_real_distribution` (repo convention) — the demo's
  GPU-vs-CPU comparison and every gate is therefore exactly reproducible run to run.

## How we verify correctness

**Two independent tiers**, per this repository's twin-independence ruling (see `kernels.cuh` SECTION 5
and `reference_cpu.cpp`'s file header for the full statement):

1. **GPU-vs-CPU twin comparison** (`VERIFY:` in `main.cu`) — every kernel in `kernels.cu` has an
   independently-*typed* (not copy-pasted) CPU counterpart in `reference_cpu.cpp`; `main.cu` runs both
   full pipelines and compares five major-stage outputs (radiance, Reinhard, local tone map, naive blend,
   Mertens fusion) within a documented, MEASURED-then-margined tolerance per stage:

   | Stage | Tolerance | Why |
   |---|---|---|
   | radiance_merge | 10% **relative** | radiance spans ~5 decades; an absolute bound is meaningless at either end (measured drift: ~2e-6 relative) |
   | reinhard_global | 1e-4 absolute | a pure per-pixel map after radiance is already agreed — near-bit-exact (measured: 0) |
   | local_tonemap | 5e-3 absolute | two float/double host round-trips (mean, min/max) accumulate more drift |
   | naive_blend | 1e-2 absolute | `we=2, sigma=0.12` make the well-exposedness weight numerically SENSITIVE (a small exponent denominator amplifies float32-vs-double differences before `exp()`); naive_blend uses the RAW unblurred weight, so it shows this at full strength (measured: ~3.3e-3) |
   | mertens_fusion | 2e-3 absolute | the SAME weight sensitivity, but Gaussian-pyramid blurring smooths most of it back out (measured: ~2.7e-4, an order of magnitude less than naive) |

   The **exception**: `crf_solve_debevec` is host-only, SHARED code (see [The GPU mapping](#the-gpu-mapping)
   for why no GPU parallelization is worthwhile at this problem size) — the twin comparison above is
   therefore blind to bugs inside it, which is exactly why an independent tier exists.

2. **Six independent gates against ground truth** (`GATE ...:` in `main.cu`), none of which route through
   `crf_solve_debevec`'s own output as their reference — each compares against something computed a
   DIFFERENT way:

   - **`crf_recovery`** — the recovered `g` (after the measured offset correction) vs. `crf_true_g`, an
     INDEPENDENT re-derivation of `scripts/make_synthetic.py`'s closed-form CRF, re-typed in C++
     (never shared code, never a library call). Measured: max deviation `~0.058` ln-units over
     `z in [10,245]`, tolerance `0.15`.
   - **`radiance_reconstruction`** — recovered radiance (offset-corrected) vs. the EXACT ground-truth
     radiance dump (`data/sample/ground_truth_radiance.bin`), restricted to pixels unclipped in `>=2`
     exposures (a fair test — a pixel clipped in 3+ exposures has too little information for ANY
     algorithm). Measured: mean relative error `~4.3%` over `19174/19200` qualifying pixels; tolerance
     `15%`. Pixels clipped in EVERY exposure (measured: 0 in this scene) are reported, not gated.
   - **`tone_map_range`** — Reinhard output strictly in `[0,1)` (measured: `[0.00076, 0.9944)`) AND
     non-decreasing along the noise-free calibration strip within a small, measured epsilon (see
     [Numerical considerations](#numerical-considerations)).
   - **`dynamic_range_coverage`** — the reason this project exists, made quantitative: the fraction of
     pixels "well-exposed" (normalized value in `[0.05, 0.95]`) for local_tonemap and mertens_fusion MUST
     exceed the BEST of the four raw exposures (the negative control). Measured: best single exposure
     `86.4%`, local_tonemap `99.97%`, mertens_fusion `91.4%`.
   - **`detail_preservation`** — local RMS contrast (population std) in a deep-shadow AND a highlight ROI;
     BOTH local_tonemap and mertens_fusion must exceed the best single exposure in BOTH ROIs
     simultaneously (no single exposure can, by construction — see README). The shadow ROI is sized to
     cover most of the actual shadow rectangle, not a small sub-patch — see
     [Numerical considerations](#numerical-considerations) for why a too-small ROI made mertens_fusion's
     measured contrast fall BELOW the single-exposure baseline (a genuine property of weight-pyramid
     blurring diluting a small region's signal, not a bug).
   - **`halo_check`** — an overshoot metric (deviation beyond the flanking plateau values) along a
     scanline crossing a real image edge; the naive single-scale blend must show measurably (>=1.15x)
     MORE overshoot than the real multiscale Mertens fusion. Measured: naive `0.348`, fused `0.229`
     (ratio `~1.52x`).

Every tolerance and gate bound above followed this repository's calibration discipline: run the real
pipeline, read the actual number, THEN set a bound with margin above/below it — never a bound chosen
before the number was known, and never a bound placed exactly at the measured value.

## Where this sits in the real world

- **OpenCV** ships production implementations of every algorithm this project teaches:
  `cv::createCalibrateDebevec()` (CRF recovery), `cv::createMergeDebevec()` (radiance merge),
  `cv::createMergeMertens()` (exposure fusion), `cv::createTonemapReinhard()` /
  `cv::createTonemapDrago()` / `cv::createTonemapMantiuk()` (a family of tone operators beyond this
  project's two). Compare their exposed parameters against this project's `main.cu` constants.
- **Production ISPs** (image signal processors) implement HDR tone mapping as a fixed-function or
  programmable hardware block, running at full video rate on every frame — a very different engineering
  problem from this project's offline, four-exposure batch process, but the same underlying math
  (usually a local, tile-based tone curve, closer in spirit to this project's local tone mapping than to
  a naive global curve).
- **Sensor-level HDR silicon** — the real answer for anything moving at video rate. Two dominant
  approaches: (1) **dual conversion gain (DCG)** sensors read each pixel through both a high- and a
  low-gain path in the SAME exposure, combined on-chip into an extended-range single frame; (2)
  **split-pixel / DOL-HDR** sensors (Sony's "Digital Overlap" HDR is the best-known example) stagger
  multiple short/long exposures within a single frame readout, achieving bracketing's dynamic-range
  benefit without bracketing's multi-frame capture-time cost or motion sensitivity. Both eliminate the
  inter-frame-motion problem this project's software bracketing cannot avoid (see README "Limitations &
  honesty"), at the cost of sensor silicon complexity and price — the real engineering trade-off a
  camera-systems team makes (see `PRACTICE.md`).
- **Ghost/motion-artifact rejection** — production HDR merge pipelines (Debevec-Malik-based or Mertens-
  based) commonly add explicit motion detection between bracketed frames (e.g., comparing local
  gradients or using optical flow, cf. 01.03-optical-flow in this repository) to exclude or downweight
  regions that moved between exposures — entirely absent from this project's teaching-scope
  implementation (see README "Limitations & honesty").
- **This project's reduced scope vs. production**, summarized: single-channel (no color/saturation term,
  no white balance); no motion/ghost handling; a simplified (bilinear) pyramid EXPAND rather than Burt &
  Adelson's polyphase form; a hand-rolled Gaussian-elimination CRF solve rather than SVD or a GPU-
  accelerated batched solver. Every one of these is a documented, deliberate simplification in service of
  a learner building and verifying the core algorithms from first principles — see README "Limitations &
  honesty" and this file's notes throughout for exactly where and why.
