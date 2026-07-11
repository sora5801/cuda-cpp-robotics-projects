# 01.22 — Motion deblurring and super-resolution for inspection zoom: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

### Motion blur is an exposure INTEGRAL

A camera does not sample a scene instantaneously — every pixel's photosite integrates incoming
light over its entire exposure window `[t0, t0+T)` (`T` = exposure time, seconds). If the camera
(or the scene) moves during that window, each photosite integrates light from a MOVING patch of the
scene, not a fixed one. Project **01.10** derives this exact idea for a ROLLING shutter, where each
image row `v` has its OWN exposure window `[t0 + v*t_line, t0 + v*t_line + T)` — cited by name here
because this project specializes that same physics to the simpler **GLOBAL shutter** case: every
pixel shares the SAME exposure window, so the "motion during exposure" story collapses from a
per-row integral to a single, image-wide one.

Formally, let `s(x, y)` be the (hypothetical, instantaneous) sharp scene irradiance and let the
camera translate at constant velocity `v = (vx, vy)` (px/s) during the exposure. The observed pixel
at `(x, y)` integrates the scene along the camera's motion PATH:

```
   blurred(x, y) = (1/T) * integral_{t=0}^{T} s(x - vx*t, y - vy*t) dt
```

Substituting `u = vx*t` (so `du = vx dt`, and similarly for `y`) turns this time-integral into a
SPATIAL integral along a line segment of length `L = |v| * T` pixels, oriented along the motion
direction `theta = atan2(vy, vx)`:

```
   blurred(x, y) = integral over the line segment of s(x - u, y - u') , normalized to unit area
```

This is EXACTLY a convolution of `s` with a **line point-spread function (PSF)**: a 1-D segment of
length `L` at angle `theta`, uniform intensity along its length (constant velocity => the camera
dwells equally on every point along the path), zero elsewhere, normalized to sum to 1 (energy is
redistributed, never created or destroyed). `kernels.cuh` fixes `L = kBlurLengthPx = 9.0` px and
`theta = kBlurAngleDeg = 20.0` degrees for this project's demo; `scripts/make_synthetic.py`
rasterizes that continuous line into a discrete 15x15 kernel (its header derives the rasterization).

**Engineering reality this simplifies away:** a real inspection robot's motion during an exposure is
rarely perfectly linear and constant-velocity (vibration, acceleration, a rotating joint sweeping the
camera) — a curved or accelerating PSF ("comet trail") needs a more general rasterizer than the
straight-line one here (README "Exercises" suggests trying one). Exposure time `T` itself is set by
the imaging system's shutter/strobe electronics (PRACTICE.md §2) and trades directly against motion
blur length: half the exposure time halves `L` at the same velocity — the classic freeze-motion vs.
signal-to-noise trade every camera engineer makes (PRACTICE.md §1 discusses strobe-freezing as the
alternative to deblurring entirely).

### Where super-resolution's information comes from: aliasing

A camera's spatial sampling rate is fixed by its pixel pitch (the physical spacing between
photosites — PRACTICE.md §2). By the Nyquist-Shannon sampling theorem, a sensor sampling at pitch
`p` (px spacing) can only FAITHFULLY represent scene detail with a period of `>= 2*p` (i.e. spatial
frequency `<= 1/(2p)`, the sensor's **Nyquist frequency**). Any true scene detail with a SHORTER
period than `2*p` does not simply vanish — it **ALIASES**: it gets folded onto a spurious LOWER
apparent frequency, indistinguishable (from a single frame) from genuine low-frequency content. This
project's `sr_resolution` gate demonstrates this directly: the "fine" bar-chart group (period 3
truth-px = 1.5 LR-px/cycle, below the low-res grid's 2-LR-px/cycle Nyquist limit) aliases in EVERY
single low-res frame into a wrong-period moire pattern — measured: a single frame's bicubic upscale
matches the TRUE pattern with correlation only ~0.22 (near-uncorrelated), despite having plausible-
looking local contrast (README "Expected output"; `src/main.cu`'s `bar_pattern_correlation()`
comment tells the full measurement story).

Multi-frame super-resolution's information does NOT come from "inventing" detail — it comes from
**diversity of sampling phase**. Each of this project's 8 low-res frames samples the SAME
band-limited-by-optics scene at a DIFFERENT sub-pixel shift (a quarter-LR-pixel lattice,
`data/sample/shifts_truth.csv`). Combined, N frames at N distinct sub-pixel phases approximate
sampling the scene at an `N`x finer effective grid — exactly like combining several
below-Nyquist-rate ADC samples, each phase-shifted, to reconstruct a signal one single ADC could
not. This project's 2x SR grid needs only 4 well-chosen phases (a half-LR-pixel lattice) to reach
its Nyquist rate exactly; the quarter-pixel lattice used here is a comfortable oversampling of that
requirement (`kernels.cuh` Section 3).

**The hard physical floor multi-frame SR CANNOT cross:** no amount of frame-combining recovers
detail LOST TO OPTICS before the sensor ever sampled it. A lens has its own diffraction limit
(Airy-disk blur set by aperture and wavelength) and the sensor's own pixel-pitch acts as a physical
low-pass (box) filter BEFORE sampling (`make_synthetic.py`'s box-downsample step models exactly this
pre-sampling low-pass). Multi-frame SR recovers information LOST TO UNDERSAMPLING (aliasing); it is
powerless against information the OPTICS never delivered to the sensor plane at all. Zoom optics
(a physically longer focal length) attacks the OPTICAL limit directly; multi-frame SR attacks the
SAMPLING limit — PRACTICE.md §1/§2 discusses this trade-off in cost and mechanical terms.

## The math

### Wiener deconvolution — MMSE derivation

Model the blurred, noisy observation `y = h * x + n` (`*` = 2-D convolution, `x` = true sharp scene,
`h` = the known PSF, `n` = additive noise, independent of `x`). In the FREQUENCY domain (capital
letters = 2-D DFT, `f` = a 2-D frequency index), convolution becomes pointwise multiplication:
`Y(f) = H(f) X(f) + N(f)`. We want a LINEAR estimator `Xhat(f) = G(f) Y(f)` minimizing the
mean-squared error `E[|X(f) - Xhat(f)|^2]`. Standard Wiener-filter calculus (setting the derivative
of the MSE with respect to `G(f)` to zero, assuming `X` and `N` are uncorrelated) gives:

```
   G(f) = conj(H(f)) / ( |H(f)|^2 + Sn(f)/Sx(f) )
```

where `Sn(f)`/`Sx(f)` are the noise/signal POWER SPECTRA. This project uses the standard teaching
simplification of a single CONSTANT `K` in place of the frequency-dependent ratio `Sn(f)/Sx(f)`
(`kernels.cuh`'s `kWienerK`, measured/tuned to 0.006 on the reference scene) — the **parametric
Wiener filter**, "Numerical considerations" below discusses the consequence.

### Why the naive inverse filter explodes

Setting `K = 0` in the formula above gives the **naive inverse filter**, `G(f) = 1/H(f)`. Where
`H(f)` is exactly (or near) zero — and a line PSF's spectrum IS sinc-like along its motion axis,
with genuine zero crossings — dividing by `H(f)` divides by (near) zero. Crucially, `Y(f)` at that
same bin is `H(f)X(f) + N(f) ~= N(f)` (the signal term also vanishes with `H(f)`, but the NOISE term
does not — noise is independent of the blur). So `Xhat(f) = Y(f)/H(f) ~= N(f)/H(f)` — a finite noise
sample divided by a near-zero number, AMPLIFIED without bound. `K > 0` in the Wiener formula is
exactly what prevents this: it puts a floor under the denominator that dominates wherever `|H(f)|^2`
is small, deliberately trading a LITTLE detail loss (where `H(f)` is large, `K` is negligible and the
two filters agree) for NOT amplifying noise into garbage (where `H(f)` is small).

### Richardson-Lucy — from Poisson maximum likelihood

Model each observed pixel `y_i` as a Poisson-distributed count with mean `(h * x)_i` (a physically
apt model for photon-counting imaging — the same Poisson story project 01.11 uses for shot noise,
cited there in its kernels.cuh). The log-likelihood of observing `y` given estimate `x` is
`sum_i [ y_i log((h*x)_i) - (h*x)_i ]` (dropping the `x`-independent `log(y_i!)` term).
Maximizing this via Expectation-Maximization (the derivation is the classical Richardson (1972) /
Lucy (1974) result) yields the multiplicative update

```
   x_{k+1} = x_k .* ( h_flip * (y ./ (h * x_k)) )
```

where `.*`/`./` are elementwise multiply/divide, `*` is (circular) convolution, and `h_flip` is `h`
rotated 180 degrees (the ADJOINT of the convolution-by-`h` operator — correlation with `h` is
convolution with `h_flip`; see "Numerical considerations" for why, for THIS project's specific PSF,
`h_flip` happens to equal `h`). Each iteration is PROVABLY non-decreasing in the Poisson
log-likelihood (a property of the EM algorithm), which is why RL needs no step-size / learning-rate
parameter and never diverges the way a naive gradient step could — only the ITERATION COUNT is a
free choice (`kernels.cuh`'s `kRlIterations = 30`, chosen for visible convergence within the demo's
runtime budget; more iterations trade sharper recovery against amplifying more noise, the classic
RL "semi-convergence" trade-off named in "Where this sits in the real world" below).

### Shift-and-add and iterative back-projection

**Shift-and-add.** Given `N` low-res frames `y_1..y_N`, each a KNOWN-shift, aliased sampling of the
same HR scene `x` (an `S`x finer grid, `S = kLrScale = 2`), the simplest possible SR estimate splats
every LR sample onto its known HR location and averages overlapping contributions:

```
   xhat(p) = sum_n sum_{q in footprint(n,p)} w(n,q,p) y_n(q)  /  sum_n sum_{q in footprint(n,p)} w(n,q,p)
```

where `footprint`/`w` are the bilinear splat weights `kernels.cu`'s `shift_and_add_kernel` computes.
This is a (weighted) DENSITY ESTIMATE, not a physically-modeled inverse — it ignores the sensor's
own point-spread function entirely, which is why it is only the STARTING point for refinement.

**Iterative back-projection (Irani & Peleg, 1991).** Treat the current HR estimate `x_k` as a
hypothesis; SIMULATE what each LR sensor would have measured from it (`forward_simulate_kernel`'s
bilinear gather, this project's simplified stand-in for the sensor's true box-integration PSF — see
"Numerical considerations"); compare against the ACTUAL measurement; and push the DISAGREEMENT back
onto the HR estimate through the adjoint of the same forward model:

```
   x_{k+1} = x_k + lambda * BackProject( y_n - Forward_n(x_k) )   for all n, averaged
```

`lambda = kIbpStep = 0.6` is a relaxation (step-size) factor — see "Numerical considerations" for
why `lambda < 1`. Unlike Richardson-Lucy, IBP has no likelihood-maximization guarantee baked in; its
convergence is monitored EMPIRICALLY via the reprojection RMS `||y_n - Forward_n(x_k)||` this
project's `sr_consistency` gate checks falls monotonically.

## The algorithm

**Naive inverse / Wiener (per call):** forward FFT of the blurred image (`O(WH log(WH))`), forward
FFT of the (once-only, cached) zero-padded PSF, one complex pointwise divide (`O(WH)`), inverse FFT
(`O(WH log(WH))`). Total: `O(WH log(WH))`, dominated by the FFTs — the entire reason to use an FFT
instead of a direct `O(W^2 H^2)` spatial-domain deconvolution.

**Richardson-Lucy:** `kRlIterations` (30) iterations, each two `O(WH * P^2)` circular convolutions
(`P = kPsfSize = 15`) plus two `O(WH)` elementwise maps. Total `O(iterations * WH * P^2)` — for this
project's sizes (`WH=16,384`, `P^2=225`), ~110M multiply-adds, dwarfed by the FFT-domain methods'
asymptotic advantage at large image sizes but perfectly fine at this project's scale (measured GPU
time: a few milliseconds, `demo/expected_output.txt`'s companion `[time]` line).

**Shift-and-add:** `O(N * kLrN)` (`N=8` frames), each contributing to up to 4 HR cells — `O(N *
kLrN)` total scatter work, `O(kN)` finalize work.

**IBP:** `kIbpIterations` (12) iterations, each `O(N * kLrN)` forward-simulate + `O(kN * N)`
back-project (every HR pixel gathers from all `N` frames) — `O(iterations * N * kN)` total, still
linear in image size and frame count.

## The GPU mapping

**cuFFT (naive inverse / Wiener):** this project calls `cufftPlan2d` + `cufftExecR2C`/`cufftExecC2R`
— a CUDA-toolkit LIBRARY, used here (as in project 03.01, cited by name) because a GOOD parallel FFT
(bit-reversal permutation, per-pass twiddle-factor generation, shared-memory-staged butterfly
scheduling across multiple kernel launches) is itself a substantial teaching project on its own; this
project's SUBJECT is deconvolution, not FFT internals. `reference_cpu.cpp` hand-rolls an INDEPENDENT
radix-2 Cooley-Tukey FFT (bit-reversal + `log2(N)` butterfly passes, textbook form) specifically so
this project does not merely "trust cuFFT" — the VERIFY gate proves the two agree.

**Pointwise spectral ops (`naive_inverse_kernel`/`wiener_kernel`/`scale_real_kernel`):** the simplest
possible GPU mapping — one thread per complex frequency bin (or per real pixel for the scale
kernel), a pure MAP with no shared memory and no cross-thread communication, the same shape as the
SAXPY placeholder every project in this repo starts from.

**`convolve_circular_kernel` (Richardson-Lucy):** a dense STENCIL, one thread per output pixel,
reading a 15x15 neighborhood from GLOBAL memory with wraparound indexing (matching the FFT-domain
methods' implicit circular-convolution semantics — `main.cu`'s `build_padded_psf()` derives the
exact wraparound placement this requires). No shared-memory tiling is used (unlike 01.11's bilateral
filter, cited by name) — README "Exercises" names tiling as the natural follow-up optimization.

**`shift_and_add_kernel` — the project's one SCATTER kernel.** Each thread (one per (frame,
LR-pixel) pair) computes ONE physical measurement's continuous HR-grid location and bilinearly
SPLATS it into up to 4 output cells. Because DIFFERENT threads (different frames, or neighboring
pixels of the same frame) can target the SAME output cell in the same launch, every write MUST be an
`atomicAdd` — the identical scatter/atomics story as 01.11's BM3D-lite group kernel (cited by name).

**`forward_simulate_kernel` / `backproject_kernel` — the GATHER counterparts.** These invert the
EXACT SAME geometric relationship shift-and-add uses, algebraically solved for the OTHER direction:
"given an HR pixel, which LR sample(s) correspond to it" instead of "given an LR sample, which HR
cells does it touch". Because each output element's dependency set is bounded and known in advance,
NO atomics are needed — every thread owns exactly one output element. This scatter-vs-gather
CONTRAST (same physics, opposite GPU mapping, opposite synchronization need) is this project's most
important GPU-mapping lesson, and is why milestone 2 implements BOTH shift-and-add (scatter) and IBP
(gather) rather than just one.

**`bicubic_upscale_kernel`:** a data-dependent-weight STENCIL — one thread per HR pixel, gathering a
4x4 LR neighborhood with Keys' (1981) cubic-convolution weights computed from the pixel's fractional
sub-pixel position.

## Numerical considerations

**FFT precision.** cuFFT operates in FP32; `reference_cpu.cpp`'s hand-rolled FFT operates in FP64
(`Complex64`) specifically so the CPU twin is the MORE trustworthy of the two, not a second copy of
the GPU's own rounding. Measured GPU-vs-CPU disagreement for the well-conditioned Wiener filter is
tiny (~0.0002 DN, `demo/expected_output.txt`'s companion `[info]` line); for the naive inverse
filter, the SAME small FFT-implementation rounding difference gets AMPLIFIED by the designed
instability at PSF spectral near-zeros — `src/main.cu`'s `kTolNaiveInverse` is set noticeably looser
than the other tolerances for exactly this reason, documented at its definition.

**Division near PSF zeros — the epsilon story.** `kNaiveInverseEpsilon` (`kernels.cuh`) is
DELIBERATELY tiny (1e-4) and explicitly NOT a regularizer: its only job is to keep `1/(|H(f)|^2 +
epsilon)` finite (prevent literal IEEE division-by-zero producing `inf`/`NaN`, which would corrupt
the ENTIRE inverse FFT via that single bin, not just fail gracefully at it) while still letting the
explosion the naive_inverse_failure gate demonstrates actually happen. Contrast this with `kWienerK`
(0.006), which is TWO ORDERS OF MAGNITUDE larger and genuinely changes the filter's behavior — the
difference between "a numerically-necessary floor" and "an actual regularizer" is exactly the gap
between these two constants.

**Richardson-Lucy positivity and the flip-is-a-no-op observation.** RL's multiplicative update
preserves positivity automatically (a ratio of non-negative quantities times a non-negative estimate
stays non-negative) as long as the initial estimate and PSF are non-negative — no explicit clamping
is needed, unlike an additive gradient-descent update. `reference_cpu.cpp`'s header notes that THIS
project's specific PSF (a line segment sampled symmetrically about its own center) is
POINT-SYMMETRIC (`psf(-δ) == psf(δ)`), which makes the "flip 180 degrees" step in RL's adjoint
convolution a no-op FOR THIS PSF SPECIFICALLY — the code still builds and uses a general flipped
buffer (never special-cased on symmetry) so it reads correctly for the general algorithm, which DOES
need a genuine flip for an asymmetric PSF (an accelerating-camera "comet trail", README "Exercises").

**IBP step size.** `kIbpStep = 0.6` (< 1) is a stability/convergence trade: a step of 1.0 would apply
the FULL measured correction every iteration, which can overshoot and oscillate when multiple
frames' corrections partially disagree (registration noise, interpolation error); a smaller step
converges more slowly but more reliably. This project's `sr_consistency` gate empirically verifies
the chosen step converges monotonically over `kIbpIterations` — an empirical check, not a proof, in
contrast to Richardson-Lucy's built-in likelihood-monotonicity guarantee above.

**Angle and quantization.** PSF angles are stored/compared in degrees (human-readable, matching
`kernels.cuh`'s documented constants); no angle wrapping logic is needed anywhere in this project
(`kMismatchAngleDeg = kBlurAngleDeg + 25.0` never approaches the ±180-degree wrap boundary for the
demo's chosen values). Every restored image is quantized to 8-bit DN only at the FINAL artifact-write
step (`dn_to_pgm()` in `main.cu`); all intermediate computation stays FP32 (GPU) / FP32-or-FP64 (CPU
twins, per method).

## How we verify correctness

Two independent tiers, per the repo's standard discipline (`reference_cpu.cpp`'s header states the
full ruling this project follows):

1. **VERIFY (GPU vs. an independent CPU twin), per method.** `naive_inverse_cpu`/`wiener_cpu` use a
   from-scratch radix-2 CPU FFT (never calling cuFFT) — the "CPU FFT twin" option this project's
   task brief names explicitly, chosen over a spatial-domain direct-convolution twin because it also
   exercises cuFFT's own correctness, not just the pointwise math around it.
   `richardson_lucy_cpu`/`bicubic_upscale_cpu`/`shift_and_add_cpu`/`ibp_refine_cpu` are independent
   nested-loop implementations of the same well-known formulas. Tolerances range from ~0.05 DN
   (deterministic, well-conditioned methods) to a looser bound for `shift_and_add` (atomic-order
   nondeterminism, the 01.11 BM3D-lite precedent) — every tolerance is measured on the reference
   machine, then margined, and documented at its definition in `src/main.cu`.
2. **Independent, ground-truth-based GATES**, never routed through the shared FFT/bilinear
   machinery: `wiener_recovery`/`rl_recovery` (PSNR + edge-gradient improvement over the blurred
   baseline, against `truth.pgm`), `naive_inverse_failure` (PSNR must be WORSE than doing nothing —
   a designed negative result, not a positive one), `psf_mismatch` (measured PSNR degradation under
   a wrong PSF), `sr_resolution` (pattern CORRELATION against `truth.pgm`, chosen over raw contrast
   after contrast proved misleading — see "The problem" above and `src/main.cu`'s
   `bar_pattern_correlation()` comment for the measurement that surfaced this), and
   `sr_consistency` (IBP's reprojection RMS falls monotonically — a physical invariant of a
   converging iterative solver, not a twin-agreement check at all).

## Where this sits in the real world

**Deconvolution in production:** OpenCV and scikit-image ship well-tested Wiener/Richardson-Lucy
implementations (README "Prior art"); real ISPs typically use a frequency-DEPENDENT Wiener
regularizer (an estimated noise power spectrum, not this project's single constant `K`) and often
blend classical deconvolution with a learned prior. **Blind deconvolution** (PSF unknown, estimated
jointly with the sharp image — e.g. via a Richardson-Lucy variant that also updates the PSF
estimate) is the harder, more general version of milestone 1, worth an entire project of its own;
this project's `psf_mismatch` gate demonstrates exactly WHY blind deconvolution matters — an
inaccurate non-blind PSF assumption costs real reconstruction quality.

**Multi-frame super-resolution in production:** every modern smartphone camera runs a production
version of this project's milestone 2 on nearly every photo (Google's "Super Res Zoom", Apple's deep
fusion/multi-frame pipelines) — using the camera's natural hand-tremor (or a deliberately actuated
OIS module) to capture several sub-pixel-shifted frames per shot, then a learned (not hand-derived)
fusion network combines them. Industrial inspection systems use the SAME idea more deliberately: a
part on an indexed conveyor, or a robot arm executing a small deliberate scan pattern, captures known
sub-pixel-shifted frames on purpose (PRACTICE.md §3). **Learned restoration** (DnCNN-era CNNs through
current diffusion-based and transformer restoration models, README "Prior art") increasingly
outperforms both classical deconvolution and classical multi-frame SR on real photographs — at the
cost of needing training data, losing closed-form guarantees (Richardson-Lucy's likelihood-
monotonicity has no learned-network analogue), and, for inspection/metrology use, raising the
"can I trust this pixel" evidentiary question PRACTICE.md §4 discusses honestly.
