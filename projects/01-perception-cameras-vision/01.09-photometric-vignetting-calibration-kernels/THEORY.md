# 01.09 — Photometric/vignetting calibration kernels: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

### Where vignetting comes from: the cos^4 law, derived

Picture a thin lens of focal length `f`, aperture diameter `D`, imaging a scene onto a flat sensor. A
point on the sensor at the OPTICAL AXIS (dead center) collects light straight through the lens. A point at
radial distance `r` from the axis collects light through the SAME aperture, but seen at an angle
`theta = atan(r / f)` off-axis. Four independent geometric effects each shrink the light reaching that
off-axis point by a factor of `cos(theta)`, and they MULTIPLY:

1. **Inverse-square falloff with an angled path.** The off-axis point is farther from the lens (by a
   factor `1/cos(theta)` along the ray), so the light reaching it, ALREADY accounting for solid angle,
   falls as `cos^2(theta)` relative to the on-axis point (one power from the path-length increase, one
   more from projecting that onto the sensor plane).
2. **Foreshortened aperture.** Seen from the off-axis point, the (circular, on-axis) aperture appears as
   an ELLIPSE — its projected area shrinks by `cos(theta)`. Less aperture area visible means less light
   collected, one more factor of `cos(theta)`.
3. **Foreshortened pixel.** The sensor pixel itself, receiving light at an angle rather than head-on,
   presents a smaller effective collecting area to the incoming ray bundle — the same foreshortening
   argument applied at the OTHER end of the optical path, one final factor of `cos(theta)`.

Multiply all four: `V(theta) = cos^4(theta)`. This is **natural** (optical) vignetting — a property of
basic geometric optics, present even in a theoretically perfect, aberration-free lens. With
`theta = atan(r / f_eff)` (`f_eff` an "effective focal ratio" in PIXEL units, since this project never
calibrates a physical pixel pitch — see [Limitations & honesty](README.md#limitations--honesty)):

```
V(r) = cos^4(atan(r / f_eff))
```

This is exactly `vignette_v()` in `scripts/make_synthetic.py` and `v_true_of_r()` in `src/main.cu` — see
[The math](#the-math) for the code-value consequence.

### Mechanical (pupil) vignetting — a second, different mechanism

A real lens is not a single thin element — it is a STACK of elements inside a barrel, each with a finite
aperture, plus an adjustable iris (aperture stop). At WIDE apertures (small f-numbers), off-axis rays can
be physically CLIPPED by the barrel or by a lens element's rim before they even reach the aperture stop —
an entirely different mechanism from the smooth cos^4 falloff above: a hard geometric occlusion that
typically produces a faster-than-cos^4, sometimes asymmetric, falloff that improves (partially) as the
lens is stopped down. Real vignetting is the PRODUCT of natural (optical) and mechanical (pupil)
vignetting; this project models only the natural term, stated as a deliberate simplification — see
[Limitations & honesty](README.md#limitations--honesty).

### Microlens / chief-ray-angle (CRA) mismatch — a sensor-side contributor

Modern backside-illuminated (BSI) CMOS sensors place a microlens over every pixel to focus incoming light
onto a photodiode that occupies only part of the pixel's physical area. Each microlens is manufactured
with a slight LATERAL OFFSET from its photodiode, increasing with distance from the sensor center, so that
its optical axis points roughly toward the lens's EXIT PUPIL rather than straight up — this is the
sensor's own "chief ray angle" (CRA) profile, and it is DESIGNED for one specific lens's CRA curve. Pair
that sensor with a DIFFERENT lens whose actual CRA-vs-radius curve does not match the microlens array's
assumption, and the mismatch produces additional falloff (and sometimes color shading, on a Bayer sensor)
on top of the pure optical vignette — a real, sensor-specific effect this project's single scalar `V(r)`
folds into the same multiplicative field rather than modeling separately (see
[Limitations & honesty](README.md#limitations--honesty)).

### PRNU and DSNU — where the per-pixel device physics comes from

Even with vignetting perfectly corrected, no two pixels on a real sensor respond identically:

- **PRNU (photo-response non-uniformity)** — pixel-to-pixel variation in QUANTUM EFFICIENCY: the
  fraction of incident photons that actually generate a collected electron. Its physical sources include
  microscopic variation in photodiode area across the wafer (lithography tolerances), doping-concentration
  variation, anti-reflective-coating thickness variation, and (per the previous section) microlens
  alignment error. PRNU is a FIXED, illumination-proportional pattern — it multiplies the signal, and a
  well-made sensor keeps it to roughly +-1-3% RMS (this project's synthetic PRNU, +-2%, sits at the
  friendly end of that real range).
- **DSNU (dark-signal non-uniformity)** — pixel-to-pixel variation in DARK CURRENT: thermally-generated
  electrons that accumulate in a pixel's charge well even with ZERO incident light, caused by crystal
  lattice defects and trap states that vary across the wafer (a "hot pixel" is simply the extreme tail of
  this same distribution). Dark current is ADDITIVE (independent of illumination) and, critically,
  TEMPERATURE-DEPENDENT: it roughly DOUBLES every 6-8 degC — the reason DSNU calibration has a much
  shorter validity window than PRNU/vignette calibration in a real deployed system (see
  [Where this sits in the real world](#where-this-sits-in-the-real-world) and `PRACTICE.md` §1).
  Real sensors also add a deliberate BLACK-LEVEL PEDESTAL — a small constant offset baked into the ADC —
  specifically so that read noise's negative excursions never clip against an unsigned ADC's floor of
  zero; this project folds that pedestal into the same additive field `o(x,y)` the calibration recovers
  (see `scripts/make_synthetic.py`'s `BLACK_LEVEL` constant and [Numerical considerations](#numerical-considerations)).

### Engineering constraints a real camera imposes

- **Storage.** A full-resolution gain map PLUS a full-resolution offset map, in float32, cost 8 bytes per
  pixel — 66 MB for a 4K sensor. A production system either stores a compact PARAMETRIC representation
  (this project's `a2,a4,a6`, 12 bytes total for the vignette) or a coarsely SUBSAMPLED nonparametric map
  interpolated at runtime — the exact parametric-vs-nonparametric trade this project's `radial_fit` gate
  makes quantitative (see [The algorithm](#the-algorithm)).
- **Temperature.** PRNU and the optical vignette are essentially temperature-INDEPENDENT (they are
  geometric/quantum-efficiency effects); DSNU is strongly temperature-dependent (see above) — a real
  system either recalibrates dark current periodically (a fast, cheap dark-frame-only re-capture) or
  models its temperature dependence explicitly and corrects for the sensor's live temperature reading.
- **Division noise amplification.** Correcting `(I - o) / g` divides by a number close to but not exactly
  1 — at the DARKEST corners (`g` as low as ~0.6 in this project's synthetic rig, and far lower for a
  dead/underperforming pixel), any residual noise in the numerator is AMPLIFIED by `1/g`. A real
  correction pipeline must guard against `g` values near zero (`kGainFloor` in `kernels.cuh`) — see
  [Numerical considerations](#numerical-considerations).
- **Recalibration triggers.** Any change to the optical path — a lens swap, a bump that shifts the lens
  barrel, a significant temperature excursion — invalidates the vignette/PRNU calibration and demands a
  recapture; see `PRACTICE.md` §3 for the concrete bring-up and recalibration procedure.

## The math

### Notation

- `(x, y)` — pixel coordinates, `x` in `[0, W)`, `y` in `[0, H)`, `W=160, H=120` (`kW, kH` in
  `kernels.cuh`). Row-major raster convention, **not** a robot body frame.
- `I(x,y)` — the observed (raw) code value at pixel `(x,y)`, one frame, code-value units (this project's
  whole model lives in this domain, not normalized `[0,1]` — a difference from 01.08's tone-mapped
  outputs, stated explicitly since it changes every artifact's clamp/stretch convention, see `main.cu`).
- `L(x,y)` — the true scene radiance the sensor would see with a perfect (uniform-gain, zero-offset)
  sensor, same units as `I`.
- `g(x,y) = V(x,y) * PRNU(x,y)` — the multiplicative field, dimensionless, `g` near 1.
- `o(x,y) = BLACK_LEVEL + DSNU(x,y)` — the additive field, code-value units.
- `r(x,y) = sqrt((x+0.5-cx)^2 + (y+0.5-cy)^2)` — radial distance from a center `(cx,cy)`, PIXEL-CENTER
  sampling (the `+0.5`: a pixel's nominal position is its cell's center, not its integer corner —
  consistent across `scripts/make_synthetic.py` and every C++ reader).

### The sensor model

```
I(x,y) = g(x,y) * L(x,y) + o(x,y) + noise(x,y)
```

a first-order LINEAR model: the observed value is the true radiance, scaled by a per-pixel gain, plus a
per-pixel offset, plus noise. This is the textbook flat-field model used throughout astronomical and
machine-vision imaging (see [Where this sits in the real world](#where-this-sits-in-the-real-world)); the
composition note in `README.md`'s Overview explains why this project stops BEFORE the nonlinear camera
response function 01.08 teaches.

### The vignette, in code-value terms

`V(x,y) = cos^4(atan2(r(x,y), f_eff))` (derived in [The problem](#the-problem--physics--engineering-first)),
with `f_eff = FOCAL_EFF_PX` and `(cx,cy)` the TRUE (decentered) optical center — the exact formula
`scripts/make_synthetic.py`'s `vignette_v()` and `src/main.cu`'s `v_true_of_r()` both implement
(independently — see [How we verify correctness](#how-we-verify-correctness)).

### Calibration: recovering `o` and `g` from stacks

**Dark-stack mean.** With the aperture closed, `L=0` everywhere, so `I_f(x,y) = o(x,y) + noise_f(x,y)`
for each dark frame `f`. Averaging `N_dark=16` such frames:

```
o_hat(x,y) = (1/N) * sum_f I_f(x,y)  ->  E[o_hat] = o(x,y),  Var[o_hat] = Var[noise] / N
```

an unbiased estimator whose STANDARD DEVIATION shrinks as `1/sqrt(N)` — derived in full in the noise
section below, and exactly what the `noise_averaging` gate measures against `N=1,4,16`.

**Flat-stack mean, dark-subtract, center-normalize.** With UNIFORM illumination `L = L_FLAT`, the
flat-stack mean estimates `g(x,y)*L_FLAT + o(x,y)`. Subtracting the recovered `o_hat` leaves
`g(x,y)*L_FLAT` (up to averaged-down noise). Dividing by the MEAN of this quantity over a small region
near the image center (`kCenterRoi*`) — call that mean `c` — gives the **nonparametric gain map**:

```
g_hat(x,y) = (flat_avg(x,y) - o_hat(x,y)) / c,     c = mean over ROI of (flat_avg - o_hat)
```

Since `c ~= L_FLAT * mean_ROI(g)` and `mean_ROI(g) ~= 1` (the center ROI sits where `V ~= 1` and PRNU
averages toward its own mean of ~1 over 64 pixels), `g_hat(x,y) ~= g(x,y)` — the map divides out `L_FLAT`
entirely, needing no absolute radiometric calibration to do so. This is the **industrial flat-field
standard**: `corrected = (raw - dark) / (flat - dark) * mean(flat - dark)`, applied here in normalized
(mean-1) form.

### The parametric radial fit — least squares, derived

The nonparametric map `g_hat` captures PRNU's per-pixel detail but is noisy and memory-heavy (see
[The problem](#the-problem--physics--engineering-first)). This project ALSO fits a compact 3-parameter
radial model:

```
V_fit(r) = 1 + a2*r_n^2 + a4*r_n^4 + a6*r_n^6,     r_n = r / rNorm
```

(only even powers — a physically motivated choice: `cos^4(atan(r/f))`, Taylor-expanded, is an even
function of `r`; odd terms would fit noise, not signal). The intercept is FIXED at 1 (`V(0)=1` by
construction of the physical model), so fitting means choosing `(a2,a4,a6)` to minimize the sum of squared
residuals against the BINNED nonparametric map (binning first — see [The GPU mapping](#the-gpu-mapping) —
averages down PRNU's per-pixel noise within each bin, so the fit sees a cleaner V(r) signal than a
per-pixel fit would):

```
minimize  sum_i ( basis_i . [a2,a4,a6] - target_i )^2,     basis_i = [r_n_i^2, r_n_i^4, r_n_i^6], target_i = mean_gain_bin_i - 1
```

Standard calculus (set the gradient to zero) gives the NORMAL EQUATIONS `A^T A x = A^T b`, where `A`'s
rows are the `basis_i` and `b`'s entries are the `target_i` — a dense but tiny (3x3) linear system,
solved by Gaussian elimination with partial pivoting in `kernels.cu`'s `solve3x3` (THEORY.md
[Numerical considerations](#numerical-considerations) explains why `r_n` must be NORMALIZED, not raw
pixel radius, for this solve to be well-conditioned).

### Noise: read + shot, and the `1/sqrt(N)` averaging law

Real sensor noise has (at least) two independent components:

- **Read noise** — signal-INDEPENDENT Gaussian noise from the readout electronics (amplifiers, ADC),
  fixed standard deviation `sigma_read`.
- **Shot noise** — the fundamental statistical fluctuation of counting DISCRETE photons/electrons: a
  Poisson process, whose variance EQUALS its mean, `Var = signal`, so `sigma_shot = sqrt(signal)` in
  electron-equivalent units. More light means more absolute noise (though a SMALLER noise fraction —
  `sigma/signal = 1/sqrt(signal)` falls as the signal grows, the reason bright-scene SNR is always better
  than dark-scene SNR at the same relative gain).

Combined (independent noise sources add in QUADRATURE): `sigma_total = sqrt(sigma_read^2 + sigma_shot^2)`
— exactly `scripts/make_synthetic.py`'s noise model, with `sigma_shot = SHOT_NOISE_K * sqrt(light_signal)`
a documented linear-in-electron-units simplification (see
[Numerical considerations](#numerical-considerations) for the honest accounting).

**The `1/sqrt(N)` law, derived.** For `N` INDEPENDENT draws of a random variable with variance `sigma^2`,
the variance of their MEAN is `sigma^2 / N` (a direct consequence of `Var[sum X_i] = sum Var[X_i]` for
independent `X_i`, and `Var[c*X] = c^2*Var[X]` for the `1/N` scaling). Taking square roots:

```
std(mean of N draws) = sigma / sqrt(N)
```

So averaging `N=4` frames should cut the residual noise standard deviation to `1/sqrt(4) = 1/2` of its
`N=1` value, and `N=16` frames to `1/sqrt(16) = 1/4` — exactly the ratios the `noise_averaging` gate
measures (`std_1/std_4 ~= 2.0`, `std_1/std_16 ~= 4.0`) against the KNOWN expected value
`gain_true*L_FLAT + dsnu_true`, independent of anything the calibration pipeline itself computed.

## The algorithm

Step by step (complexity is per the full `W*H = kN = 19,200`-pixel image unless noted; `N_stack=16`):

1. **Dark-stack mean** — `stack_mean` over `N_stack` dark frames. Serial cost: `O(N_stack * kN)`.
   Parallel cost: `O(N_stack)` (one thread per pixel, `kN`-way parallel, each doing a length-`N_stack`
   serial reduction) — see [The GPU mapping](#the-gpu-mapping).
2. **Flat-stack mean** — same primitive, on the flat stack.
3. **Dark-subtract** — `elementwise_sub`, `O(kN)` serial / `O(1)` parallel depth.
4. **Center-normalize** — `roi_mean_reduce` (a `O(kN)` serial / `O(log(block size))` parallel-depth
   shared-memory reduction, MASKED to the 8x8 center ROI) produces one scalar `c`; `affine` then rescales
   the whole map by `1/c`. This is the **nonparametric gain map** — the industrial standard, capturing
   PRNU's per-pixel detail, at the cost of being noisy (averaged down only by `N_stack`) and full-resolution
   (memory-heavy — see [The problem](#the-problem--physics--engineering-first)).
5. **Radial binning** — `radial_bin`: `O(kN)` serial / `O(1)` parallel depth (a scatter-reduce; see
   [The GPU mapping](#the-gpu-mapping)), producing `kNumRadialBins=44` `(sum, count)` pairs.
6. **Parametric fit** — the shared, HOST-ONLY `fit_vignette_radial_ls`: `O(numBins)` to build the 3x3
   normal equations, `O(1)` (a fixed 3x3 solve) to solve them. This is the **parametric gain model** — a
   compact, SMOOTH, extrapolatable curve (defined for ANY radius, even beyond the calibrated field of
   view) that, BY CONSTRUCTION, cannot capture PRNU's per-pixel ripple (a smooth function of `r` alone has
   no freedom to vary independently at two pixels sharing the same radius). This is the exact
   parametric-vs-nonparametric trade [The problem](#the-problem--physics--engineering-first) names, made
   quantitative by the `radial_fit` gate's residual-decomposition check.

   **Scoping decision, stated honestly:** the fit assumes the vignette is centered at the GEOMETRIC image
   center `(W/2, H/2)`, not the TRUE (decentered) optical center the synthetic ground truth actually uses.
   A production pipeline that wants to recover `(cx,cy)` too needs a NONLINEAR least-squares refinement
   (the model is linear in `a2,a4,a6` but nonlinear in `cx,cy` — Gauss-Newton or Levenberg-Marquardt, an
   iterative extension out of this project's "simplest correct teaching version" scope, CLAUDE.md §13).
   Because the true decentering here is small (3 px in x, -2 px in y, against a ~100 px radius domain),
   this scoping choice contributes only a small, honestly-measured bias — see
   [Numerical considerations](#numerical-considerations).
7. **Correction** — `correction`: `(I - o_hat) / max(g_hat, floor)`, `O(kN)` serial / `O(1)` parallel
   depth, applied per-frame in production (see README "System context" for the two DIFFERENT rate budgets
   — calibration offline, correction per-frame).

## The GPU mapping

- **`stack_mean_kernel`** — one thread per PIXEL, looping serially over `N_stack=16` FRAMES. The
  FRAME-MAJOR stack layout (`kernels.cuh`) makes every iteration's access coalesced: at loop step `f`,
  adjacent threads (adjacent pixels) read adjacent addresses `stack[f*n + p]`, `stack[f*n + p+1]`, ...,
  one 128-byte-aligned transaction per warp per step. A per-pixel SHARED-MEMORY tree reduction would be
  the "textbook" answer for reducing many elements, but `N_stack=16` is too few to be worth it — the
  crossover where shared-memory staging pays for itself is closer to hundreds of elements per reduction,
  not 16; the memory-bound serial loop already saturates bandwidth at this problem size.
- **`elementwise_sub_kernel` / `affine_kernel` / `correction_kernel`** — pure MAPs, one thread per pixel,
  fully coalesced, no shared memory (no data reused between threads — shared memory only pays when threads
  SHARE or REVISIT data, none of which applies to an elementwise map).
- **`roi_mean_reduce_kernel`** — the CLASSIC two-phase reduction: each thread converts its pixel to
  "value if inside the ROI, else 0.0" (the additive-identity MASKING trick — a rectangular ROI sum
  implemented with a flat 1-D kernel and no compaction pass), a binary-tree reduction in SHARED memory
  collapses each block to one partial sum, then ONE atomicAdd per block (not per thread) merges the ~75
  block partials into a single global `double` accumulator. Shared memory here is fast, on-chip storage
  every thread in the block needs to read OTHER threads' partial sums from — global memory could do the
  same reduction but at 10-30x the latency per access.
- **`radial_bin_kernel`** — a GENUINELY DIFFERENT reduction pattern: a SCATTER-REDUCE (histogram). Every
  one of `kN=19,200` threads computes its OWN pixel's bin index and atomicAdd's DIRECTLY into one of only
  `kNumRadialBins=44` GLOBAL bins — no shared-memory staging, because the output has 44 DIFFERENT
  destinations, not one. Average contention per bin: `kN / 44 ~= 436` threads, spread across many warps
  and blocks; a per-block PARTIAL histogram (accumulate locally in shared memory, merge once per block)
  would reduce that contention further and is a real optimization (README Exercise 4) — not implemented
  here because this is a ONE-TIME calibration step (not a per-frame hot path): atomics contention here
  costs microseconds, a rounding error against the capture time the calibration half of this pipeline
  already spends (see README "System context").
- **The parametric fit (`fit_vignette_radial_ls`) — NO GPU mapping, by design.** A 3x3 dense linear
  solve has no meaningful parallelism to extract: the entire problem fits in a handful of registers, and
  the overhead of a kernel launch (module load, grid setup, at minimum a few microseconds) would dwarf
  the O(1) work being done. This is the SAME judgment call 01.08's `crf_solve_debevec` makes for its own
  one-time calibration solve — 33.01-batched-small-matrix-linalg is where this repo teaches the GPU-BATCHED
  version of small solves, at a problem SIZE (many independent small systems at once) where batching
  genuinely amortizes launch overhead across real parallelism.

## Numerical considerations

- **Double accumulators for every reduction.** `stack_mean_kernel`'s per-pixel sum over 16 frames, and
  `roi_mean_reduce_kernel`'s atomicAdd target, both accumulate in `double` even though the INPUTS are
  `float32` — cheap here (16 adds; 64 ROI pixels), and it keeps this project's numerics unquestionably not
  the source of any GPU-vs-CPU disagreement, so a real mismatch is never confused with accumulation drift.
- **Division by a small gain.** `correction_kernel` computes `(I - o) / max(g, kGainFloor)`. A pixel with
  a near-zero TRUE gain (a dead or badly underperforming pixel — this project's synthetic rig never
  produces one; measured minimum recovered gain is ~0.627, see `demo/expected_output.txt`) would otherwise
  amplify any residual numerator noise by `1/g` — a `+-1` code-value residual becomes a `+-100` correction
  error at `g=0.01`. `kGainFloor=0.05` is the documented, always-applied guard.
- **Float32 atomicAdd, ordering, and why `radial_bin` needs a nonzero VERIFY tolerance.** GPU atomics from
  different threads/blocks arrive in an UNDEFINED order; float32 addition is NOT associative
  (`(a+b)+c != a+(b+c)` in general, due to rounding at each step), so the GPU's bin sums can differ in
  their last few bits from the CPU reference's fixed left-to-right sequential sum — measured on the
  reference machine at ~1.0e-3 absolute, well inside the documented `kTolRadialBinSum` tolerance. This is
  the textbook "same math, different order, different rounding" story every GPU-vs-CPU reduction in this
  repo eventually tells (see 01.08's `luminance_log_sum_kernel` for the identical lesson).
- **Basis normalization for the least-squares fit.** Raw pixel radii up to `r~100` raised to the 6th power
  are `~1e12` — catastrophic for a 3x3 normal-equations solve in either FP32 or FP64 (the condition number
  of `A^T A` scales roughly as the SQUARE of the basis's dynamic range). Normalizing `r_n = r / rNorm`
  (`rNorm=100`, roughly the image's own radius scale) keeps every basis column in `[0,1]`-ish range and
  the solve well-conditioned — a general lesson (always normalize your design matrix's columns to similar
  scale before a dense solve), not specific to vignetting.
- **The decentering the fit ignores.** Because the fit assumes the GEOMETRIC center while the true optical
  center is offset by `(3, -2)` px, `V_fit(r)` is a slightly-biased estimate of the true `V` at any given
  PIXEL (though an accurate estimate of the true `V` as a function of RADIUS alone, averaged
  azimuthally) — this shows up as a small, honestly bounded residual contribution on top of PRNU in the
  `radial_fit` gate's residual-ratio check (measured ratio ~1.01, comfortably inside the documented
  `[0.5, 2.0]` band — the decentering's contribution here is small relative to PRNU's own ~2% amplitude,
  by the deliberate choice of a modest synthetic decentering; see [The algorithm](#the-algorithm)).
- **Uint8 quantization vs. the +-2 LSB DSNU signal.** DSNU's true excursion (+-2 code-value units) is only
  a few times the sensor's own quantization step (1 LSB) — a SINGLE dark frame's DSNU is barely
  distinguishable from noise+quantization by eye; only the `1/sqrt(16)` averaging of the dark stack pulls
  it cleanly above the noise floor, which is exactly why `dsnu_recovered.pgm` is written with an ADAPTIVE
  contrast stretch (see `main.cu`'s `stretch_to_pgm`) rather than a fixed one — the raw signal would be
  invisible otherwise.
- **The black-level pedestal.** Without `BLACK_LEVEL=8.0` added to `o(x,y)`, a dark frame's negative noise
  excursions would clip against an unsigned 8-bit floor of zero, biasing the dark-stack mean upward in a
  way that does NOT average out with more frames (a systematic, not a random, error) — see
  [The problem](#the-problem--physics--engineering-first) for the physical justification real sensors share.

## How we verify correctness

**GPU-vs-CPU twins (VERIFY).** Every kernel (`stack_mean`, `elementwise_sub`, `affine`, `roi_mean_reduce`,
`radial_bin`, `correction`) is implemented TWICE — once as a CUDA kernel in `kernels.cu`, once as a plain
sequential loop in `reference_cpu.cpp` — and compared element-wise (or bin-wise) in `main.cu`, each with
its own documented tolerance (`src/main.cu`'s `kTol*` constants, each set by MEASURING the actual
reference-machine value and margining, never set at the measured value). The ONE piece of "the algorithm"
that is deliberately SHARED, not duplicated, is the parametric least-squares fit
(`fit_vignette_radial_ls`) — a single 3x3 host solve with no meaningful GPU/CPU split to compare (see
[The GPU mapping](#the-gpu-mapping)), mirroring 01.08's `crf_solve_debevec` precedent exactly.

**Independent gates (never routed through the pipeline being graded).** Because the LS fit is shared, the
twin comparison above is BLIND to bugs inside it — the twin-independence ruling (`reference_cpu.cpp`'s
file header) therefore REQUIRES at least one gate that does not route through it. This project carries
SIX independent gates, each grading a different claim against ground truth that was NEVER an input to the
calibration pipeline:

1. **`dsnu_recovery`** — mean absolute error AND Pearson correlation between the recovered `o_hat` and
   the committed `dsnu_true.bin` (generated by `scripts/make_synthetic.py`, never touched by the pipeline).
2. **`gain_recovery`** — mean relative error between the nonparametric `g_hat` and `gain_true.bin`.
3. **`radial_fit`** — (a) the fitted `V_fit(r)` vs. the KNOWN analytic `cos^4` curve, INDEPENDENTLY
   re-derived in `main.cu` from `FOCAL_EFF_PX` (never shared code with the fit itself or with the Python
   generator — the 01.08 `crf_true_g` precedent); (b) a residual-decomposition consistency check: the
   fit's own residual (`g_hat - V_fit(r)`) must have the SAME statistical scale as the TRUE PRNU-induced
   ripple (`gain_true - V_fit(r)`) — proving the decomposition's SEMANTICS, not just its magnitude.
4. **`noise_averaging`** — the measured `std(N=1)/std(N=4)` and `std(N=1)/std(N=16)` ratios against the
   `1/sqrt(N)` law's ideal `2.0`/`4.0`, using the KNOWN expected flat-field value (`gain_true*L_FLAT +
   dsnu_true`) as the residual's reference point — never the pipeline's own recovered fields.
5. **`correction_efficacy`** — THE reason-to-exist gate: five identical-radiance swatches (baked into
   `scene.pgm`, never seen during calibration) must read as EQUAL after correction, with the UNCORRECTED
   disparity reported as the negative-control baseline (measured ~26% on the reference machine — real
   vignette-induced bias, not a bug).
6. **`flatness`** — a corrected single flat frame must read spatially uniform; the UNCORRECTED
   (dark-subtracted-only) version is reported as the negative-control baseline. Honestly weaker evidence
   than gate 5 (it is NOT drawn from held-out data — see README "Limitations & honesty"), included anyway
   because it directly exercises the same `correction_kernel` the live per-frame pipeline runs.

## Where this sits in the real world

- **Factory NUC tables.** Every commercial camera module ships with a factory-measured non-uniformity
  correction (NUC) table — this project's `gain_recovered`/`dsnu_recovered` maps, at production scale —
  baked into the sensor's own ISP or firmware and applied in HARDWARE before a single byte reaches the
  host driver. The calibration RIG (integrating sphere or diffuser plate, dark-frame capture procedure) is
  described concretely in `PRACTICE.md` §1-2.
- **DSO's photometric calibration.** Engel, Koltun & Cremers' *Direct Sparse Odometry* (PAMI 2018) is the
  named downstream consumer in README "System context" — its direct (photometric, not feature-based)
  residuals require EXACTLY this project's output shape (a vignette map + a response function) before
  intensity differences between frames mean anything geometrically. DSO ships its own calibration tool
  that recovers a similar model from a captured video sequence rather than a dedicated dark/flat rig — see
  Goldman (PAMI 2010) for the general "recover vignette from a natural image sequence, no rig needed"
  family of techniques, a genuinely different (and harder, more ill-posed) problem than this project's
  dedicated-rig approach.
- **EMVA 1288.** The machine-vision industry's formal standard for sensor characterization defines
  precise measurement procedures for PRNU, DSNU, dark current, conversion gain, and noise — this project's
  dark/flat-stack pipeline is a teaching-scale version of exactly those procedures (full standard named as
  orientation, not reproduced, in `PRACTICE.md` §4).
- **Flat-field correction in production machine vision.** Industrial inspection cameras (bin-picking
  vision, PCB inspection, produce grading) apply flat-field correction as one of the FIRST stages in their
  image pipeline, precisely because absolute or relative intensity measurements (not just edges/shapes)
  are often the actual measurand — an uncorrected vignette reads as a spatially-varying MEASUREMENT BIAS,
  not merely a cosmetic defect.
- **Decentering, done right.** Production camera-calibration toolchains (e.g., the vignette models bundled
  with photogrammetry and SfM pipelines, or dedicated research code following Goldman 2010) fit the
  optical center as a free parameter via NONLINEAR least squares (Gauss-Newton/Levenberg-Marquardt) —
  this project's Exercise 2 names the extension explicitly.
