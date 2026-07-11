# 01.11 — Low-light denoising (bilateral, non-local means, fast BM3D variant): Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2).

## The problem — physics & engineering first

**Photon arrival is a counting process.** A camera pixel does not measure light continuously — it
counts individual photons converted to photoelectrons over an exposure. Photon arrivals in a fixed
time window follow a **Poisson process**: if the *expected* number of photoelectrons collected is
`lambda`, the *actual* count `k` on any given exposure is Poisson-distributed,

```
P(k arrivals) = lambda^k * exp(-lambda) / k!,      E[k] = lambda,      Var[k] = lambda
```

The defining, physically-forced fact is `Var[k] = lambda` — **the variance of shot noise equals the
mean signal itself**, not a free parameter you can tune away. This is not an engineering choice; it
falls straight out of counting independent, memoryless events (a Poisson process's variance-equals-
mean property is a textbook result of its generating function, `E[e^{tX}] = exp(lambda(e^t-1))`,
whose first two derivatives at `t=0` both come out to `lambda`).

**Why this makes low light qualitatively different.** Signal-to-noise ratio from shot noise alone is
`SNR = lambda / sqrt(lambda) = sqrt(lambda)`. At `lambda = 10,000` electrons (a well-lit daytime
pixel), `SNR ~ 100` — noise is a rounding error. At `lambda = 40` electrons (this project's peak
signal, code value 255), `SNR ~ 6.3` — noise is loud enough to see with the naked eye, and at
`lambda = 4.4` electrons (the darkest committed flat patch, code value 28), `SNR ~ 2.1` — the noise
is comparable in magnitude to the signal itself. **A denoising filter tuned on daylight images
(SNR ~ 100) is tuned for a noise regime that simply does not resemble this one**, which is the
robotics-relevant lesson: whatever noise-floor assumption a perception pipeline makes in bright light
needs re-examining before it is trusted at night.

**Read noise and the sensor's electronic floor.** On top of shot noise, every readout circuit adds its
own noise — thermal (Johnson-Nyquist) and reset noise in the pixel's amplifier chain, roughly Gaussian
and, crucially, **signal-independent**: it is there even with the shutter closed (a "dark frame", the
subject of sibling project 01.09's dark-stack calibration). This project models it as
`N(0, kReadNoiseE^2)` electrons rms, `kReadNoiseE = 2.0` — the middle of a real back-illuminated (BSI)
CMOS sensor's typical 1-3 e- rms range at low analog gain (PRACTICE.md §2 dates and sources this).

**Quantization.** The analog-to-digital converter (ADC) rounds the amplified electron count to an
integer code value — a bounded rounding error, `~ +-0.5` DN, negligible next to the ~18-40 DN shot+read
noise this project's operating point produces (`kernels.cuh`'s `predicted_noise_std_dn()` comment
quantifies the omission as harmless).

**The engineering frame.** A real low-light camera cannot simply "turn up the exposure" (motion blur,
frame-rate budget) or "turn up the gain" (amplifies read noise along with signal, and does nothing for
shot noise's fundamental `sqrt(lambda)` floor) — physics sets a hard limit on what any denoising
ALGORITHM can recover once the photons that would carry the missing information were simply never
collected. Denoising is therefore always a *statistical* recovery (using redundancy — spatial
smoothness, self-similarity — to estimate the underlying signal), never a *physical* one; every method
in this project is a different way of finding and exploiting that redundancy.

## The math

**The sensor model** (single-sourced in `kernels.cuh` Section 2, mirrored independently in
`scripts/make_synthetic.py` — see that file's module docstring for why the Poisson draw is EXACT, not
Gaussian-approximated):

```
signal_e(clean_dn)   = clean_dn / kDnPerElectron                         (electrons, from the noise-free ground truth)
shot_e                ~ Poisson(signal_e)                                (EXACT sampling, Knuth's inversion algorithm)
noisy_e                = shot_e + N(0, kReadNoiseE^2)                    (read noise, electrons)
noisy_dn                = clamp_round(noisy_e * kDnPerElectron)          (back to DN, 8-bit quantized)
```

and the analytic prediction every gate in `main.cu` compares against (`predicted_noise_std_dn()`,
`kernels.cuh`):

```
Var[noisy_dn] = (signal_e + kReadNoiseE^2) * kDnPerElectron^2
std[noisy_dn] = sqrt(signal_e + kReadNoiseE^2) * kDnPerElectron
```

(Poisson variance in electrons, plus independent read-noise variance, scaled to DN by the *squared*
gain — `Var[aX] = a^2 Var[X]` for a linear rescale.) Measured on the reference machine: the three flat
patches (code values 28/128/175, i.e. ~4.4/20.1/27.5 expected electrons) give measured/predicted std
ratios of 1.025 / 0.922 / 0.886 — a few percent off, well inside this project's honest sanity band
(main.cu's `noise_model_sanity` gate).

### Bilateral filtering — joint-domain filtering

A plain Gaussian blur weights neighbor `q` of pixel `p` by distance alone:
`w(p,q) = exp(-|p-q|^2 / 2*sigma_s^2)`. This assumes *nearby pixels are similar* — true in flat
regions, false across an edge (a neighbor two pixels away, on the wrong side of a step, is not
"probably the same surface" just because it is close). The bilateral filter (Tomasi & Manduchi, 1998)
adds a second weight in the **range** (intensity) domain:

```
w(p,q) = exp(-|p-q|^2 / 2*sigma_s^2) * exp(-(I(p)-I(q))^2 / 2*sigma_r^2)
out(p) = sum_q w(p,q)*I(q) / sum_q w(p,q)
```

Now a neighbor must be BOTH spatially close AND photometrically similar to contribute much weight —
"joint spatial x range" filtering. Across a true edge, `I(p)-I(q)` is large, the range term collapses
toward zero, and the filter naturally stops averaging across it — no edge detector, no special case,
just the product of two Gaussians. `kernels.cuh`'s `kBilateralSigmaSpatial=2.5` px /
`kBilateralSigmaRange=40` DN were tuned against this project's own measured noise scale (~18-40 DN).

### Non-local means — the self-similarity prior

Bilateral's assumption ("nearby is similar") is local. NLM (Buades, Coll & Morel, 2005) makes a
different, non-local assumption: **natural images are full of repeated local structure** — a patch of
grass, a stretch of brick, a stretch of hashed texture, recurs many times across an image, not just
next door. Instead of comparing single pixel intensities, NLM compares whole PATCHES:

```
d(p,q)   = mean_{delta in patch} ( I(p+delta) - I(q+delta) )^2      (mean squared patch difference)
w(p,q)   = exp( -d(p,q) / h^2 )
out(p)   = sum_{q in search window} w(p,q)*I(q) / sum_q w(p,q)
```

`h` plays sigma_r's role (a similarity tolerance), but on PATCH distance rather than single-pixel
difference — averaging 5x5=25 independent noise samples per comparison makes `d(p,q)` a far more
reliable "are these really the same underlying signal" test than a 1-pixel comparison could ever be,
which is exactly why NLM tolerates heavier noise than bilateral before it starts blurring structure
away (measured: NLM retains 89% of the clean edge gradient at this noise level vs bilateral's 98% —
bilateral is actually MORE conservative here because its 9x9 window is smaller than NLM's 13x13 search
window; the point stands at matched window sizes, see Exercise 2). The search is restricted to a
finite window (13x13 here) purely for cost — in principle NLM searches the whole image.

### BM3D — collaborative filtering (full pipeline; this project builds stage 1)

BM3D (Dabov, Foi, Katkovnik & Egiazarian, 2007) pushes the self-similarity idea one step further:
instead of averaging matched patches PIXEL BY PIXEL (NLM), it stacks them into a 3-D array and
denoises the WHOLE STACK JOINTLY in a transform domain, exploiting correlation along the "which patch"
axis that pixel-wise averaging throws away.

**Stage 1 (hard-thresholding — what this project implements):**
1. **Group:** for a reference 8x8 patch, find its `kBm3dStackSize=16` most similar 8x8 patches (by
   SSD) within a search window — a HARD top-K selection, unlike NLM's soft weighting of everything.
2. **Transform:** apply an orthonormal **2-D DCT-II** to each of the 16 patches (a rotation of the
   64-pixel patch into a frequency-like basis where a smooth patch's ENERGY concentrates into a few
   low-frequency coefficients), then an orthonormal **1-D Haar transform ACROSS the stack** (the
   "3-D" in BM3D: correlated content across similar patches concentrates the SAME way along this third
   axis too — a patch that repeats 16 times has a huge "stack-DC" coefficient and near-zero
   "stack-detail" coefficients).
3. **Hard-threshold:** zero every coefficient with `|coefficient| < kBm3dThreshLambda * sigma`. This
   is where the magic happens: BECAUSE both transforms are orthonormal, i.i.d. noise of variance
   `sigma^2` in pixel space becomes i.i.d. noise of the SAME variance `sigma^2` in EVERY transform
   coefficient (proved below, "Numerical considerations") — so a threshold at `lambda*sigma` is a
   PRINCIPLED noise-vs-signal cutoff, not a magic number, and it removes noise from EVERY coefficient
   at once rather than pixel by pixel.
4. **Invert** both transforms, recovering 16 denoised patches.
5. **Aggregate:** each denoised patch is written back to its OWN matched location (not just the
   reference location), weighted by `1/(1+N_nonzero)` — a group whose coefficients mostly vanished is
   judged confidently denoised (little genuine detail survived thresholding) and trusted more; since
   patches from DIFFERENT reference groups overlap, every pixel ends up as a weighted blend of several
   independent denoising estimates — the "collaborative" in collaborative filtering.

**Stage 2 (collaborative Wiener filtering — documented here, NOT implemented; "the -lite"):** real
BM3D uses stage 1's output as an ORACLE spectrum estimate and re-groups/re-transforms the ORIGINAL
noisy patches, applying a Wiener filter (`W = |oracle|^2 / (|oracle|^2 + sigma^2)` per coefficient)
instead of a hard threshold — a soft, MMSE-optimal shrinkage that typically improves on stage 1 by
several dB. This project's "-lite" scope cut is exactly this omission, stated up front in README and
here, not hidden.

## The algorithm

Per method, the teaching-scale complexity (200x150 = 30,000 pixels, this project's frame):

| Method | Per-pixel/group cost | Total (this frame) | Parallel unit |
|---|---|---|---|
| Bilateral | O(9x9) = 81 weighted reads | ~2.4M ops | 1 thread / pixel |
| Gaussian baseline | O(9x9) = 81 weighted reads | ~2.4M ops | 1 thread / pixel |
| NLM | O(13x13 x 5x5) = 4,225 squared diffs | ~127M ops | 1 thread / pixel |
| BM3D-lite | O(13x13 x 8x8) match + O(16 x DCT8 + Haar16) transform, per GROUP | ~1,813 groups x ~15K ops | 1 thread / **group** |

Bilateral and the Gaussian baseline are `O(N * window)` — linear in pixel count. NLM is
`O(N * search_area * patch_area)` — the steepest of the four, and the measured CPU cost shows it
plainly (202 ms on one core, vs bilateral's 14 ms and BM3D-lite's 46 ms, for the SAME 30,000 pixels).
BM3D-lite's group count (1,813, roughly `N / stride^2` since patches overlap by `1 - stride/patch`)
is far smaller than `N`, but each group does much more work (a full block search plus two chained
transforms over 16 patches) — the two costs land in the same ballpark for this frame size, but scale
differently as the image grows (NLM's `search_area x patch_area` factor is fixed per pixel; BM3D-
lite's group count grows with `N` but the per-group cost does not).

## The GPU mapping

```
BILATERAL / GAUSSIAN (stencil):
  one thread = one output pixel, 2-D grid (16x16 blocks)
  naive:  9x9 window read from GLOBAL memory every time (redundant: each
          interior pixel is read by up to 81 different threads)
  tiled:  block cooperatively loads a (16+2*4)x(16+2*4)=24x24 SHARED tile
          ONCE (halo included), every thread's 9x9 window then reads
          shared memory — same loop, same order, same arithmetic as naive
          -> BIT-IDENTICAL output (kernels.cu's header proves this), lower
          global traffic -> measured ~9-20x kernel speedup on this frame.

NLM (search):
  one thread = one output pixel, 2-D grid
  every candidate's patch SSD re-reads 25 pixels from global memory, no
  reuse across candidates OR threads -- the naive-est possible mapping.
  Production fix (documented, not built): an INTEGRAL IMAGE of per-offset
  squared differences turns each patch SSD into 4 array reads instead of
  25 -- the classic O(1)-per-candidate trick (see README Exercise 4).

BM3D-LITE (batched group processing):
  one thread = one REFERENCE GROUP, 1-D grid (block=64; a smaller block
  than the repo's usual 256 -- see below)
  the "one thread does a LOT of self-contained work" pattern -- the same
  SHAPE as 08.01 MPPI's "one thread simulates one whole rollout", here
  applied to "one thread block-matches, transforms, thresholds, and
  aggregates one whole group". No shared memory used (each thread's group
  is independent data, unlike bilateral's overlapping neighborhoods).
  REGISTER/LOCAL-MEMORY HONESTY: the local stack[16][8][8] alone is 1,024
  floats (4 KiB) per thread -- far beyond a healthy register budget (a
  modern SM has ~64K 32-bit registers shared across ALL resident threads;
  4 KiB/thread would blow that budget at any real occupancy), so nvcc
  SPILLS it to per-thread LOCAL memory (backed by global memory, L1/L2
  cached). For 1,813 threads total on a tiny teaching frame this is still
  fast in absolute terms (measured ~8.2 ms). It is NOT how a production
  BM3D-GPU kernel would be shaped: one BLOCK per group, with the group's
  16x8x8 stack living in SHARED memory and threads cooperating on the
  DCT/Haar/threshold/inverse-transform steps, is the standard fix -- named
  here, not built (an honest "-lite" scope cut, not an oversight).
  ATOMICS: many groups' matched patches overlap in pixel space (that
  overlap IS the aggregation -- BM3D's whole point), and groups run on
  different threads with no ordering guarantee, so scatter-accumulation
  into out_sum/out_weight MUST be atomicAdd -- unavoidable here, and the
  reason this method's GPU-vs-CPU VERIFY tolerance is the loosest of the
  four (see "Numerical considerations").

Why no cuFFT for the 8x8 DCT: an 8x8 DCT is 64 output values from 64 input
values -- a fixed-size 8x8 matrix multiply (two of them, row-pass then
column-pass, since the transform is separable) costs ~1,024 multiply-adds,
computed directly in registers. cuFFT's setup/plan overhead and general
N-any-size machinery would be pure overhead for a transform this
structurally tiny and this fixed in size -- exactly the "no black boxes,
and sometimes hand-rolling teaches more than the library call" call
CLAUDE.md makes, and the SAME call 07.09 makes for its own small dense
solves (kernels.cuh Section 5's docstring draws the parallel).
```

## Numerical considerations

- **Orthonormal transforms preserve noise variance (the fact BM3D-lite's threshold leans on).** If `C`
  is an orthonormal matrix (`C^T C = I`) and `x` is i.i.d. Gaussian noise with variance `sigma^2` per
  entry, then `y = Cx` is ALSO i.i.d. Gaussian with variance `sigma^2` per entry — a rotation of an
  isotropic Gaussian is still isotropic with the same total variance, and orthonormality means the
  transform is exactly a rotation (no scaling). Both the 8x8 DCT-II basis here (`alpha(0)=sqrt(1/8)`,
  `alpha(k>0)=sqrt(2/8)` — chosen precisely to make the basis rows unit-norm) and the Haar transform
  (`1/sqrt(2)` scale at every butterfly step) are orthonormal by construction — `bm3d_group_kernel`'s
  header cites this proof, and it is why ONE threshold value (`kBm3dThreshLambda * kBm3dAssumedSigmaDn`)
  applies uniformly to all 1,024 coefficients in a group without per-position rescaling.
- **The single-sigma approximation this project makes, honestly.** BM3D's threshold assumes ONE
  known, uniform noise sigma (classic AWGN denoising). This project's noise is signal-DEPENDENT
  (heteroskedastic — the whole point of "The problem" above), so `kBm3dAssumedSigmaDn` (calibrated at
  mid-gray, DN=128, by hand: `sqrt(20.078 + 2.0^2) * 6.375 = 31.283`) is necessarily wrong everywhere
  else in the image — too aggressive in bright regions (higher true sigma, under-thresholds noise) and
  too timid in dark regions (lower true sigma, over-thresholds real signal). Real low-light BM3D
  pipelines fix this with a **variance-stabilizing transform** (the Anscombe transform,
  `f(x) = 2*sqrt(x + 3/8)`, which makes Poisson-ish noise APPROXIMATELY constant-variance Gaussian)
  applied before BM3D and inverted after — named here, not built.
- **DCT orthonormality as a numerical self-check.** `sum_n basis[k][n]*basis[j][n] == (k==j ? 1 : 0)`
  for every row pair — a property a learner can verify directly from `dct8_basis()`'s output and a
  useful sanity check when adapting this code (a broken basis is a silent, hard-to-spot correctness bug
  that would NOT necessarily crash, just degrade denoising quality).
- **Weight underflow (bilateral/NLM).** `expf(-large_value)` underflows toward 0.0f gracefully in
  IEEE-754 (denormals, then true zero) — never NaN or a crash — so a badly-mismatched neighbor simply
  contributes (numerically) zero weight rather than corrupting the sum; `wsum` is guaranteed nonzero
  because the `dx=dy=0` / `sx=sy=0` term always contributes weight exactly 1.
- **Float accumulation order — bilateral/gaussian/NLM are DETERMINISTIC** (fixed loop order, no
  atomics, no cross-thread interaction), so GPU-vs-CPU disagreement is PURELY `expf` (device) vs
  `std::exp` (host) implementation ULPs — measured max 0.0001 DN, comfortably inside generous
  tolerances (0.05 / 0.02 / 0.15 DN respectively).
- **BM3D-lite's aggregation is NOT deterministic across GPU runs or against the CPU** — `atomicAdd`
  serializes concurrent writes to the same address in SOME order, but that order is a hardware
  scheduling detail, not a program invariant; float addition is not associative, so different orders
  can produce different (tiny) results. `reference_cpu.cpp`'s oracle uses a FIXED raster-order
  DOUBLE accumulator specifically so the CPU side has a stable, high-precision reference; the resulting
  GPU-vs-CPU tolerance (3.0 DN) is the loosest of the four methods and is measured, not guessed
  (observed: 0.0002 DN on the reference machine — GPUs with more SMs or different scheduling could
  plausibly show more accumulation-order drift; the tolerance carries real headroom for that).
- **The 8-bit quantization floor.** Every artifact this project writes is 8-bit PGM; DN values
  computed in float (which can slightly overshoot `[0,255]` near a hard edge from filter ringing) are
  clamped before quantization (`dn_to_pgm()`), never silently wrapped or truncated.

## How we verify correctness

Two independent tiers, because a denoiser can be *numerically faithful to its own algorithm and still
denoise badly* (or vice versa) — the same two-tier argument 08.01's THEORY.md makes for its controller:

1. **VERIFY — GPU vs CPU, per method (the twin-agreement tier).** Every GPU kernel's output is
   compared, element-wise, against an INDEPENDENTLY re-typed CPU implementation (never sharing
   algorithmic code — `reference_cpu.cpp`'s header states the ruling). This catches indexing bugs,
   missed tail elements, race conditions, and clamp/border mistakes — the class of bug that produces a
   plausible-LOOKING but subtly wrong image. Per-method tolerances are measured then margined (main.cu
   comments quote the measured numbers); the bilateral naive-vs-tiled check is additionally required
   to be EXACTLY bit-identical (0.0 tolerance), because both kernels perform the IDENTICAL sequence of
   floating-point operations by construction — any nonzero difference there would mean the tiling
   refactor accidentally changed the arithmetic, not just its memory source.
2. **Five INDEPENDENT gates against ground truth (the "is this actually good denoising" tier) —
   NEVER routed through the pipeline being graded.** `psnr_improvement` catches a denoiser that VERIFIES
   correctly against its own (possibly badly-tuned) parameters but does not actually help. 
   `edge_preservation` catches the specific failure mode PSNR alone cannot see — the Gaussian baseline
   is a DESIGNED NEGATIVE CONTROL proving this gate has discriminating power (it passes
   `psnr_improvement`, at +2.31 dB, and still fails `edge_preservation` outright, at 16% vs a 55%
   floor — exactly the "numerically fine, behaviorally wrong" failure mode the two-tier argument
   predicts). `flat_noise_floor` is the complementary direct check (residual std in a KNOWN-constant
   region, not proxied through PSNR at all). `method_ordering` is reported, not forced — see the
   surprise finding below. `noise_model_sanity` checks the GENERATOR, not a denoiser — proving the
   ground truth this whole project grades against is itself trustworthy.

**The method_ordering finding, investigated honestly (not smoothed over).** Measured texture-ROI PSNR:
NLM 37.41 dB, BM3D-lite 36.30 dB, bilateral 26.86 dB — NLM slightly AHEAD of BM3D-lite, differing from
the textbook BM3D-lite >= NLM >= bilateral expectation. The likely explanation: this project's
synthetic background is a SMOOTH, continuously-correlated multi-octave hashed texture (bilinear-
interpolated value noise, not sharp real-world texture with hard boundaries) — a regime where nearly
every patch has SOME resemblance to nearby patches, favoring NLM's soft, continuous similarity
weighting over BM3D-lite's HARD top-16 selection and single global hard threshold (which can zero out
genuine low-contrast texture detail sitting just below `kBm3dThreshold`, a cost NLM's softer weighting
does not pay). This is exactly the kind of scene-dependent outcome the literature would predict IS
possible for a hard-threshold-only, single-sigma BM3D variant against a well-tuned NLM — the "lite"
scope cuts (no Wiener refinement, one global sigma) are the most likely levers that would close the
gap, and README Exercise 3 asks the learner to test that hypothesis directly rather than take this
paragraph's word for it.

## Where this sits in the real world

- **OpenCV** ships production-grade `cv::bilateralFilter` (separable/box-filter approximations for
  speed), `cv::fastNlMeansDenoising` (integral-image accelerated, exactly the trick this project
  documents but does not build), and `cv::xphoto::bm3dDenoising` (the full two-stage pipeline). Compare
  their runtime against this project's teaching kernels on the same image size — the gap IS the value
  of the optimizations this project deliberately skips for clarity.
- **Fixed-function ISP hardware NR blocks** (README "System context") implement a cheaper,
  pipelined equivalent of bilateral/NLM-family filtering directly in silicon, running at full camera
  frame rate with a power budget classical software cannot touch.
- **Learned, self-supervised denoisers** (Noise2Noise and its descendants — Noise2Void,
  Neighbor2Neighbor; burst-denoising kernel-prediction networks) increasingly replace BM3D-class
  methods in production real-time pipelines: a single forward pass through a small CNN, trained without
  ever needing clean ground truth (exactly the kind of ground truth this project's synthetic generator
  had the luxury of fabricating, and a real sensor never has). The classical methods this project
  builds remain the field's reference baseline and the conceptual ancestors (self-similarity, joint
  spatial-range weighting) that inform learned architectures' design.
- **What full BM3D adds beyond this "-lite" version:** the collaborative Wiener second stage (a
  several-dB improvement in the literature), Kaiser-window aggregation (smoother patch blending than
  this project's uniform weighting), and typically a variance-stabilizing transform for real
  (non-Gaussian, signal-dependent) sensor noise — all three are named above at their exact point of
  omission, not bundled into one vague "future work" note.
