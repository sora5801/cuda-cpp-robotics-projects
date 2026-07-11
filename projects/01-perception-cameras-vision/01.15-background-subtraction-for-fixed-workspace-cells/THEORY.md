# 01.15 — Background subtraction for fixed-workspace cells: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

### What changes a pixel's value when nothing in the scene moved?

A fixed camera watching a static work cell should, in an ideal world, report the exact same
intensity at pixel `(x, y)` forever. Real cameras never do, for reasons that are worth naming
individually because each one attacks a background model differently:

- **Shot noise.** Light arrives at a sensor as discrete photons; the number counted in a fixed
  exposure window is a Poisson process. Poisson variance equals its mean — brighter pixels are
  noisier in *absolute* terms (though a *smaller fraction* of their signal). This is the
  fundamental noise floor of any photon-counting sensor, and it is signal-DEPENDENT.
- **Read noise.** The sensor's analog front end (amplifier, ADC) adds noise that does not depend
  on how much light arrived — a fixed-variance Gaussian-ish contribution present even in complete
  darkness. Project **01.11** (low-light denoising) is this repository's deep treatment of
  separating and removing exactly these two noise sources; this project deliberately does NOT
  denoise anything, because a background model's whole job is to tolerate noise, not remove it.
- **Illumination drift.** Real work-cell lighting is rarely perfectly stable: mains-frequency
  flicker (50/60 Hz, usually invisible at typical frame rates but capable of aliasing), HVAC-driven
  fixture dimming, sunlight leaking through a window or a dock door over the course of a shift, and
  LED fixture thermal drift all shift the SCENE's true brightness slowly, over seconds to hours —
  categorically different from frame-to-frame noise because it is spatially UNIFORM and temporally
  SLOW.
- **Auto-exposure / auto-gain.** Most real cameras actively fight brightness changes by adjusting
  exposure time or sensor gain — which means a naive background model built on RAW intensities can
  see brightness jumps that have NOTHING to do with the scene, purely from the camera's own control
  loop reacting to something elsewhere in frame. This project's honest position: it assumes a
  **locked-exposure** camera (manual exposure/gain, no auto-anything) — see `PRACTICE.md` §1 for why
  that is a real installation requirement, not just a modeling convenience, and why "auto-anything
  is the enemy" for this class of algorithm.
- **Quantization.** An 8-bit sensor rounds continuous light intensity to 256 levels — a small,
  bounded, deterministic error this project folds into its synthetic noise model rather than
  treating separately.

This project's synthetic sequence (`scripts/make_synthetic.py`) models **read noise only** — a
constant-sigma Gaussian-ish contribution (`NOISE_SIGMA = 3.0` intensity units, drawn via Box-Muller
from a seeded xorshift32 stream) added uniformly regardless of scene brightness. This is an honest,
named simplification: a true shot-noise model would need per-pixel variance proportional to
intensity, which would change the classification math (the `k*sigma` test would need brightness-
dependent `sigma` even for a completely static pixel) without teaching a different LESSON about
*background modeling* — the three algorithms below react identically to "there is some noise floor,
estimate it" whether that floor is constant or brightness-dependent. A constant-sigma regime is the
honest description of a read-noise-DOMINATED camera (common in machine-vision cameras running at
high gain, low light, or short exposure — exactly a work-cell's likely operating point).

### Why a fixed camera turns "background" into a statistical object

A moving camera has no stable notion of "what pixel (x, y) shows" — the physical point it looks at
changes every frame. A **fixed** camera is different: pixel `(x, y)` maps to the same physical
direction, frame after frame, forever (up to the small vibration `PRACTICE.md` §1 discusses).
That turns "the background" from a single reference image into a genuine **statistical estimation
problem**: each pixel has its own probability distribution over intensities, built from everything
that has ever happened at that physical location, and "foreground" is formally "this frame's sample
is unlikely under that pixel's own learned distribution." This is the conceptual leap this project
teaches across its three models: frame differencing pretends the distribution is a single fixed
value (the reference frame); the running single Gaussian estimates a UNIMODAL distribution that
tracks slow drift; MOG-lite estimates a genuinely MULTI-MODAL distribution — because some pixels,
like this project's blinking status lamp, legitimately have more than one "normal" appearance.

## The math

### Notation

- `I(x, y, t)` — the observed intensity at pixel `(x, y)`, frame `t`, `t = 0 .. T-1`, unitless
  synthetic camera counts in `[0, 255]` (no radiometric calibration modeled).
- `mu(x, y, t)`, `var(x, y, t)` — a background model's running estimate of that pixel's mean and
  variance, evolving over `t`.
- `alpha in (0, 1)` — an exponential-moving-average (EMA) learning rate.
- `k` — the number of standard deviations a sample must exceed its model to be classified
  foreground.

### The EMA estimator and its time constant

Every adaptive model in this project uses the same one-line update:

```
mu_{t+1} = mu_t + alpha * (I_t - mu_t)   =   (1 - alpha) * mu_t + alpha * I_t
```

This is a first-order IIR (infinite impulse response) low-pass filter: it exponentially discounts
old samples, weighting sample `I_{t-n}` by `alpha * (1 - alpha)^n`. Its "memory" is characterized by
the **time constant** `tau = 1 / alpha` frames — roughly how many frames of history dominate the
current estimate. This project's `SG_ALPHA = 0.08` gives `tau ~= 12.5` frames (~0.4 s at the
project's assumed 30 Hz) — fast enough to track the +15% illumination ramp (which spreads its
change over all 160 frames, far slower than 12.5 frames) but slow enough not to instantly "absorb"
a genuinely new object as background.

The matching variance estimator:

```
var_{t+1} = (1 - alpha) * var_t + alpha * (I_t - mu_t)^2
```

— the same EMA machinery applied to the squared deviation from the (pre-update) mean, i.e. an
online estimate of `E[(I - mu)^2]`.

### Deriving the absorption-time closed form

This is the calculation the `absorption` gate checks the running single-Gaussian model against.
Suppose a pixel's true background is a stable value, the model has converged (`mu ~= mu_pre`), and
at frame `t0` a NEW object appears and stays — the pixel's observed value jumps to (and stays at,
approximately) some new constant `I_new`. Define the "distance" `d_t = mu_t - I_new` for `t >= t0`.
Substituting the EMA update:

```
d_{t+1} = mu_{t+1} - I_new = mu_t + alpha*(I_new - mu_t) - I_new = (1 - alpha) * (mu_t - I_new) = (1 - alpha) * d_t
```

So `d_t` decays GEOMETRICALLY: `d_t = d_0 * (1 - alpha)^(t - t0)`, where `d_0 = mu_{t0} - I_new`
is the jump size at the moment of the event (this project measures it as `mu_pre - I_new`, both
taken from the model's OWN state, never the scene's hidden ground truth — a real background
subtractor never gets to see that). The pixel stops being classified foreground once
`|d_t| <= k * sigma` (the classification threshold, `THEORY.md`'s "how we verify" derives which
`sigma` to use below). Solving for the smallest `t - t0` where this holds:

```
(1 - alpha)^(t-t0) <= k*sigma / |d_0|
(t - t0) * ln(1 - alpha) <= ln(k*sigma / |d_0|)          [both sides: apply ln, an increasing function]
(t - t0) >= ln(k*sigma / |d_0|) / ln(1 - alpha)           [DIVIDING BY ln(1-alpha), which is NEGATIVE, flips the inequality]
```

so:

```
t_abs = ceil( ln(k*sigma / |d_0|) / ln(1 - alpha) )     =    ceil( ln(|d_0| / (k*sigma)) / (-ln(1 - alpha)) )
```

Both forms are algebraically identical; this project's code uses the SECOND (dividing by
`-ln(1-alpha)`, which is positive) specifically because it is easy to get the sign wrong by
transcription — see "Numerical considerations" below for why this is treated as a first-class
numerics lesson, not a footnote.

**Which `sigma`?** The obvious answer — the pixel's PRE-event variance — turns out to be wrong for
this project's actual model, because of a variance-ceiling interaction discovered while building it
(see "Numerical considerations"). The honest closed form uses `sigma = sqrt(SG_VAR_CEIL)`, the
CEILING value, because the large jump at `t0` saturates the ceiling on the very first post-event
update and keeps it saturated for the whole foreground phase.

### The MOG-lite update rules

For each pixel, `K = 3` modes, each `(w_k, mu_k, var_k)`. Every frame:

1. **Match test.** For each mode `k`: `sigma_k = sqrt(max(var_k, floor))`;
   `matched_k = |I - mu_k| <= match_k_sigma * sigma_k`. Among matched modes, keep the CLOSEST
   (smallest `|I - mu_k|`).
2. **If a mode `m` matched:**
   ```
   w_k <- w_k + lr_w * (M_k - w_k)     for every k, where M_k = 1 if k==m else 0
   mu_m  <- mu_m + lr_p * (I - mu_m)
   var_m <- (1 - lr_p) * var_m + lr_p * (I - mu_m_old)^2
   ```
   Only the matched mode's mean/variance update — the others are untouched (they did not explain
   this sample). **Claim: this weight update conserves `sum_k w_k = 1` exactly**, whenever it held
   before the update and exactly one mode matches (`sum_k M_k = 1`):
   ```
   sum_k w_k_new = sum_k [w_k + lr_w*(M_k - w_k)] = sum_k w_k + lr_w*(sum_k M_k - sum_k w_k) = S + lr_w*(1 - S)
   ```
   which equals `1` whenever `S = 1`. This is why the matched branch alone would never need
   renormalization — the no-match branch below is what breaks it.
3. **If no mode matched:** the WEAKEST mode (lowest weight; ties keep the lowest index) is replaced
   outright: `w_weakest <- w_init_new`, `mu_weakest <- I`, `var_weakest <- var_init`; every OTHER
   mode decays as "unmatched": `w_k <- (1 - lr_w) * w_k`. This does NOT provably conserve
   `sum_k w_k = 1` (the hard-set `w_init_new` is not derived from the conservation identity above),
   which is why step 4 renormalizes explicitly.
4. **Renormalize:** `w_k <- w_k / sum_k w_k` (numerical hygiene, guarded against a near-zero sum).
5. **Rank by confidence** `c_k = w_k / sigma_k` descending — Stauffer-Grimson's own ranking metric,
   favoring modes that are both FREQUENT (high weight) and TIGHT (low variance): a mode that is
   rarely seen or wildly variable is a worse "background" candidate than one that is common and
   consistent, even at equal weight.
6. **Background set:** accumulate weights in ranked order until the running sum reaches
   `bg_fraction = 0.8`; every mode visited up to and including the one that crosses the threshold is
   "background." (This is Stauffer-Grimson's own criterion, unchanged from the original paper.)
7. **Classify:** foreground if no mode matched at all, OR the matched mode is outside the
   background set.

## The algorithm

### Frame differencing

```
for every (t, pixel) pair, independently:
    mask[t][pixel] = |I(t, pixel) - reference[pixel]| > FRAME_DIFF_THRESHOLD
```

`reference` is frame 0, captured once. Serial cost: `O(T*N)` comparisons, `N = 128*96 = 12,288`
pixels. Parallel cost: `O(1)` — every output is independent of every other, so a single kernel
launch with `T*N` threads (grid-strided) computes the whole sequence at once; there is no
frame-to-frame dependency to serialize on. This is the **fastest possible** model and, as the
`illumination_drift` gate demonstrates, the **least correct** one under slow scene change.

### Running single Gaussian

```
mu[pixel], var[pixel] <- init from frame 0
for t = 1 .. T-1:
    for every pixel, independently:
        classify against (mu[pixel], var[pixel])          # see "The math"
        update (mu[pixel], var[pixel]) by EMA
```

Serial cost: `O(T*N)`, same total work as frame differencing. Parallel cost: `O(T)` — the OUTER
loop over frames cannot be parallelized (frame `t`'s state is a function of frame `t-1`'s), but the
INNER loop over pixels is `N`-way parallel with zero cross-pixel interaction, so each of the `T-1`
sequential steps is one GPU kernel launch covering all `N` pixels simultaneously. This is the
project's clearest illustration of "some serial structure survives no matter how parallel the
per-step work is" — see "The GPU mapping" for the concrete launch-count consequence.

### MOG-lite (K=3)

Same `O(T)`-sequential / `O(N)`-parallel shape as the single Gaussian, with `O(K)` extra work
per pixel per frame for the match test, the K-way weight update, and the K-element sort (steps 1-7
above) — for this project's fixed `K=3`, that is a constant factor, not an asymptotic change:
still `O(T*N)` serial work, `O(T)` parallel depth.

### 3x3 morphological open (erode, then dilate)

```
erode:  out[pixel] = AND over the pixel's 3x3, 8-connected, zero-padded neighborhood of mask_in
dilate: out[pixel] = OR  over the pixel's 3x3, 8-connected, zero-padded neighborhood of mask_in
```

A **stencil**, not a map: each output reads 9 inputs. Erosion deletes anything not at least
1-pixel-thick everywhere (a lone speck, having fewer than 9 set neighbors anywhere, vanishes
entirely); dilation regrows what erosion left standing by the same margin. OPENING (erode then
dilate, not the reverse "closing") is the right choice here because this project's failure mode is
salt-and-pepper FALSE POSITIVES (isolated misclassified pixels) that opening is specifically built
to delete, not small gaps INSIDE a true-positive blob (which closing would fix instead — not this
project's problem). No cross-frame dependency exists (`t`'s opening only reads `t`'s raw mask), so —
like frame differencing — this runs as ONE kernel launch covering the whole `T*N` array, per stage.

## The GPU mapping

### Per-pixel independence, and what it does and does not buy you

Every classification decision in this project depends only on ITS OWN pixel's history — there is no
spatial coupling in any of the three models (morphology, the post-processing step, is the only
kernel that reads neighbors). This is the textbook MAP regime: one GPU thread per pixel, zero
inter-thread communication, zero atomics, zero races — the simplest and most GPU-friendly shape a
per-pixel algorithm can have. What it does NOT buy you is freedom from the TEMPORAL dependency: a
map over pixels within one frame is embarrassingly parallel, but frame `t+1`'s map cannot start
until frame `t`'s state update has fully landed in global memory — hence one kernel LAUNCH per
frame for the two adaptive models (`sg_step_kernel`, `mog_step_kernel`), 159 launches each, all
enqueued into CUDA's default stream, which serializes them automatically (no explicit
`cudaDeviceSynchronize` needed between iterations — see `main.cu`'s GPU pipeline comment). An
advanced alternative worth knowing about: a single PERSISTENT kernel using CUDA's cooperative-groups
grid synchronization (`cudaLaunchCooperativeKernel` + `grid.sync()`) could fold all 159 steps into
one launch, trading 159 small launch overheads for one large kernel with an internal barrier — not
implemented here (it adds a real portability/occupancy constraint — cooperative launches must fit
the WHOLE grid on the device simultaneously — for a teaching project whose per-launch overhead is
already a rounding error against the 33 ms/frame budget), but worth knowing the option exists.

### The MOG state layout: mode-major, and why it coalesces

`kernels.cuh` SECTION 4 lays out the MOG state as THREE flat arrays (`weight`, `mean`, `var`), each
`MOG_K * IMG_N` floats, indexed `k * IMG_N + pixel` — MODE-MAJOR, not pixel-major. One GPU thread
owns one PIXEL (not one mode), so at the instant every thread in a warp executes "read mode 0's
weight," adjacent threads (adjacent pixel indices) read ADJACENT addresses — the classic
128-byte-warp coalescing pattern, repeated three times (once per mode). A pixel-major layout
(`pixel * MOG_K + k`, three consecutive floats per pixel) would instead scatter one warp's "read
mode 0" accesses 3 floats apart per thread: still technically one coalesced 384-byte transaction per
warp for that one field, but it forces the SAME total bytes to move through a stride pattern that
also drags in mode 1 and mode 2's data on every access (since they share a cache line), wasting
bandwidth when the kernel wants only one mode's field at a time (as it does for the confidence
computation in step 5). Mode-major keeps each field access a clean, minimal-footprint coalesced
read.

### Occupancy: does K=3 fit in registers?

`mog_step_kernel` keeps EIGHT `float[3]` local arrays (`w`, `m`, `v`, `d`, `conf`, plus the boolean
`matched[3]`/`is_background[3]` and `int idxs[3]`) — at `K=3` this is roughly 24-30 live 32-bit
values per thread in the worst case, comfortably inside a modern GPU's per-thread register budget
(sm_75 offers up to 255 registers/thread before spilling, and the compiler is free to reuse
registers across the kernel's sequential phases since not everything is live simultaneously). This
is a claim worth VERIFYING, not just asserting: `nvcc --ptxas-options=-v` (or Nsight Compute's
occupancy view) reports the actual register count per thread for `mog_step_kernel`; a learner
extending `MOG_K` upward (README Exercise 2) should re-check this, because a general-`K` version
that no longer fits fixed-size register arrays would need shared or global memory scratch instead —
a real GPU-mapping trade-off, not just a code-size one.

### Warp divergence: the natural home for this lesson

`mog_step_kernel`'s match-and-update logic is **data-dependent per pixel**: within one 32-thread
warp, processing 32 adjacent pixels of the SAME frame, some lanes may match mode 0, others mode 1,
others mode 2, others none at all (triggering the replace-weakest branch). CUDA's SIMT execution
model handles this by running EVERY distinct code path the warp's lanes take, SERIALLY, masking off
the lanes that do not apply to each pass, then reconverging — a warp with `d` distinct active
branches pays roughly `d` times the instruction issue cost of a warp where every lane takes the same
path. For this kernel, `d` is bounded and small: at most 4 distinct paths per warp (matched mode 0,
matched mode 1, matched mode 2, or no-match/replace), so the worst-case overhead here is a **~4x**
slowdown on the branch-dependent portion of the kernel relative to a hypothetical branch-free
version — small in absolute terms (this project's whole 325-launch pipeline still finishes in single
digit milliseconds), but the mechanism generalizes badly: a hypothetical `K=100` full Gaussian
mixture would not just do more per-pixel arithmetic, it would multiply the NUMBER of distinct
per-warp code paths a large-`K` match/replace decision can produce, which is a much worse cost curve
than the linear-in-K arithmetic cost alone suggests. "Small, fixed K, fully unrolled" (this
project's `#pragma unroll` loops over `K=3`) is therefore not just a code-clarity choice — it is
what keeps this kernel's divergence cost bounded and cheap.

## Numerical considerations

### Precision, and why FP32 is enough here

All model state (`mu`, `var`, `weight`, `mean`) is `float` (FP32). Intensities are `[0, 255]`
integers promoted to float; sums, products, and EMA updates at this magnitude carry no meaningful
FP32 rounding risk (24-bit mantissa gives ~7 decimal digits of precision against values that never
exceed a few hundred) — this project has no accumulation-over-millions-of-terms pattern (unlike,
say, a large reduction) that would make FP64 worth its cost.

### The variance floor (by design) and the variance CEILING (discovered, not designed)

`SG_VAR_FLOOR` (and `MOG_VAR_FLOOR`) exists by design: without a floor, a perfectly static run of
identical samples drives the EMA variance toward 0, and the very next 1-intensity-unit noise sample
would then appear infinitely many "sigmas" away, false-firing foreground forever after. Flooring
`sigma` at the sequence's own designed noise level (matching `NOISE_SIGMA = 3.0`) is the standard
fix, and it was ALWAYS part of this project's plan.

`SG_VAR_CEIL` was not planned — it was found empirically while first building this project's
absorption gate, and it is worth walking through as a genuine debugging story, because it is the
kind of bug that only shows up once you build the closed-form check that catches it (exactly the
lesson `reference_cpu.cpp`'s twin-independence ruling argues for: a twin comparison alone would
NEVER have caught this, because the bug was in the ALGORITHM itself, identically present in both
the CPU and GPU implementations). The single-Gaussian model's BLIND update feeds
`var <- (1-alpha)*var + alpha*diff*diff` into the variance EMA every frame, including the very
frame a brand-new object appears. The box event's jump (`diff ~= 69` intensity units) alone
contributes `alpha * diff^2 ~= 0.08 * 4761 ~= 381` to the very first post-event variance update —
DOZENS of times the pre-event steady-state variance (~9-16). That inflated variance INFLATES
`sigma`, which RAISES the `k*sigma` detection threshold, which — perversely — makes the model
LESS sensitive to the very anomaly that just caused the inflation. The measured result, before the
fix: the box was "absorbed" (majority-background) in **2 frames**, not the ~20-26 the mean-EMA-only
math predicted — not because the model's MEAN had genuinely converged, but because its VARIANCE had
inflated enough to make almost anything look plausible. Capping the STORED variance at
`SG_VAR_CEIL = 36.0` (`sigma <= 6.0`, twice the noise floor) fixes this: the ceiling saturates
immediately on a large jump (verified: `alpha * diff0^2 >> SG_VAR_CEIL` for every designed event in
this sequence) and HOLDS there for the whole foreground phase, so the detection threshold stays a
small, stable multiple of the true noise floor instead of ballooning with the very event it should
be reacting to. After the fix, measured absorption is **18 frames** against a **19-frame** closed-form
prediction — the story `THEORY.md`'s "the math" section derives in full.

**The general lesson:** an EMA update that feeds its OWN dispersion estimate from the SAME signal
that triggers detection has a built-in desensitization failure mode whenever a genuine event is
large enough to dominate that estimate in one step. Any "adaptive threshold from adaptive variance"
design should ask this question explicitly, not just add a floor and assume the job is done.

### The sign lesson in the absorption-time formula

`ln(1 - alpha)` is NEGATIVE for any `0 < alpha < 1` (since `1 - alpha < 1`). A careless
transcription of "solve `d_t <= threshold` for `t`" that forgets to track the inequality FLIP when
dividing by a negative number yields a formula that predicts a NEGATIVE number of frames — an
immediate, cheap sanity check ("is my answer even positive?") that catches the mistake before it
ever reaches code. `main.cu`'s `predicted_absorption_frames()` divides by `-ln(1 - alpha)`
(positive) rather than `ln(1 - alpha)` (negative) for exactly this reason, with the derivation
spelled out in "The math" above.

### Determinism and tie-breaking

Every classification and update rule in this project is a deterministic function of the (fixed,
committed) input sequence — there is no run-time randomness anywhere in `src/`. The two places a
genuine TIE could occur: (1) MOG's match test, when two modes are EQUIDISTANT from the sample
(`kernels.cu`'s `mog_step_kernel` breaks this by keeping the FIRST strict improvement, i.e. lowest
index wins); (2) MOG's replace-weakest selection, when two modes have EQUAL weight (same rule,
lowest index wins). Both rules are documented at their point of use rather than left implicit, and
both are essentially unreachable in practice with this project's real-valued, noisy input (exact
float equality between two independently-evolved state values is astronomically unlikely) — but the
rule is still stated, because "what happens on a tie" should never be an accident of code order a
reader has to reverse-engineer.

### Robotics-specific numerical hazards: not applicable here, honestly

This project estimates scalar pixel intensities, not poses or orientations — angle wrapping,
quaternion normalization drift, stiff ODE integration, and ill-conditioned Jacobians (CLAUDE.md
§4.2's standard robotics numerics checklist) are **N/A** here: there are no angles, no rotations, no
dynamics, and no Jacobians anywhere in this project's math. The variance-floor/ceiling story above
is this project's actual numerically-hazardous territory, and it gets the full treatment instead.

## How we verify correctness

Two independent tiers, per `reference_cpu.cpp`'s ruling (see that file's header for the full
argument and the project 13.03 story that motivated it):

**Tier 1 — GPU vs. CPU twin comparison.** `reference_cpu.cpp` implements all three models and the
morphological open a SECOND time, independently: the same formulas (kernels.cuh's single-sourced
data-layout contract is shared, per the ruling's first bullet), but genuinely different code shape
where the algorithm allows it — `mog_step_cpu` sorts with `std::stable_sort` and a `Mode` struct
where `mog_step_kernel` hand-unrolls a compare-swap network over flat `float[3]` registers;
`morph_open_cpu` fuses erode+dilate into one function with an internal scratch buffer where the GPU
path needs two explicit kernel launches (a kernel-launch barrier is the ONLY way to guarantee
erosion has fully finished everywhere before dilation reads a neighbor a different thread computed).
`main.cu` runs both paths on the full 160-frame sequence and requires: (a) raw-mask element-wise
agreement within `kTwinMaskMismatchFrac = 0.0005` (measured: **0.0** — bit-exact — on the reference
GPU; the tolerance stays nonzero because 159 sequential EMA steps give float-order divergence
(different, architecture-dependent FMA fusion between `cl.exe` and `nvcc`) room to compound and, in
principle, flip a classification landing within ~1 ULP of a threshold on a DIFFERENT compute
capability than the one this was measured on); (b) final model state agreement within
`kTwinStateAbsTol = 0.01` (measured: **0.000008** max, a ~1250x margin).

**Tier 2 — five independent gates, none routing through the twin comparison.** Twin agreement
proves the GPU is a faithful PARALLELIZATION of the CPU reference; it says nothing about whether
that shared algorithm is measuring the right thing, or whether this project's OWN closed-form
predictions hold. Each gate below computes its ground truth directly from `kernels.cuh` SECTION 2's
designed-event schedule (never from the models' own output) and is documented at its exact
tolerance, with the measurement it was derived from, as a comment above its `constexpr` in
`main.cu`:

- **`intrusion_detection`** — mean IoU of each model's opened mask against the arm's EXACT
  rectangle-union ground truth, over every E1+E5 frame (lamp pixels excluded — see the gate's own
  comment for why a legitimately-blinking pixel elsewhere in frame should never penalize an
  intrusion-localization score).
- **`illumination_drift`** — false-positive rate in a late, event-free, heavily-drifted window;
  asserts BOTH that the adaptive models stay low AND that frame differencing EXCEEDS a floor — the
  designed failure, checked as a requirement, not just observed as a curiosity.
- **`absorption`** — the analytic closed-form check from "The math" above, compared against a
  measured "frames until the box region's foreground fraction stays majority-background for 5
  consecutive frames."
- **`bimodal_lesson`** — false-positive rate restricted to the blinking lamp's own pixels; asserts
  MOG stays low AND single-Gaussian EXCEEDS a floor — again, the model comparison this project
  exists to teach, checked as a requirement.
- **`noise_floor`** — false-positive rate in an early, event-free, drift-negligible window; the
  simplest possible check that neither adaptive model cries wolf on ordinary sensor noise.

## Where this sits in the real world

Production video-analytics and machine-vision stacks do not hand-roll MOG updates in raw CUDA the
way this project does (deliberately, for teaching) — they reach for:

- **OpenCV `cv::BackgroundSubtractorMOG2`** (Zivkovic, 2004/2006) — this project's MOG-lite is a
  direct, simplified ancestor: MOG2 adds an ADAPTIVE per-pixel component count (not a fixed K),
  automatic shadow detection (a classified "shadow" label distinct from foreground/background,
  useful because a moving shadow is a common false positive this project's synthetic scene never
  needed to model), and a more principled, density-proportional learning rate where this project
  uses a single fixed `MOG_LR_PARAM`.
- **OpenCV `cv::BackgroundSubtractorKNN`** — a non-parametric alternative: instead of fitting
  Gaussians, it keeps a small sample history per pixel and classifies by nearest-neighbor distance
  in that sample set, sidestepping the Gaussian-shape assumption entirely (useful for backgrounds
  whose true distribution is neither unimodal nor cleanly multi-Gaussian).
- **ViBe (Barnich & Van Droogenbroeck, 2011)** — a different sample-consensus model with a
  distinctive neighbor-diffusion update (a matched pixel occasionally propagates its sample into a
  RANDOM neighbor's history too), giving it fast spatial adaptation this project's purely-per-pixel
  models do not have.
- **Learned (CNN-based) change detection** (e.g. FgSegNet, BSUV-Net) — outperforms every classical
  model above on hard public benchmarks (dynamic backgrounds: waving trees, rippling water;
  camouflage: foreground color close to background) at the cost of needing labeled training data —
  precisely the kind of ground truth this project's fully-synthetic, closed-form approach exists to
  avoid needing.
- **GPU-accelerated classical models** already exist too — `cv::cuda::BackgroundSubtractorMOG2` runs
  essentially this project's algorithm (the full, non-"lite" version) on GPU in production OpenCV
  builds, which is worth knowing before assuming this project's hand-rolled kernels are the only way
  to get MOG on a GPU.

**The industrial safety reality.** This project's intrusion signal (README "System context")
could plausibly gate real cell behavior, and it is worth being unambiguous about why production
systems never let a background subtractor do that job. Project **21.04** (speed-and-separation
monitoring) is this repository's dedicated treatment, and it states the caveat this project inherits
verbatim: **"DIDACTIC IMPLEMENTATION — NOT A CERTIFIED SAFETY FUNCTION."** Real collaborative-safety
systems watching for human intrusion use CERTIFIED hardware — safety-rated laser scanners, light
curtains, pressure-sensitive floor mats — engineered and certified to defined Performance Levels
under ISO 13849 and evaluated against ISO/TS 15066 for collaborative applications, with independent,
redundant sensing paths a single RGB camera and a software heuristic cannot provide. See
`PRACTICE.md` §4 and `docs/SYSTEM_DESIGN.md` item 6's regulatory map for the fuller orientation —
never guidance, never a substitute for an actual functional-safety engineer's sign-off.
