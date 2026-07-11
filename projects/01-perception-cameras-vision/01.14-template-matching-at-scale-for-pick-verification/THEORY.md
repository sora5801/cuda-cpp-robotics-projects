# 01.14 — Template matching (NCC) at scale for pick verification: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

**Why does the SAME part look different in two photographs?** Pick verification's whole difficulty is
packed into that question. Three physical effects separate "the template" (a clean reference image,
captured once at teach-in) from "the window" (a live camera frame of a tray slot that may or may not
hold that exact part):

1. **Illumination.** Real light sources are never perfectly uniform: a ring light falls off toward the
   edges of its field, a single directional light casts shadows, and a nearby fixture arm or robot body
   can partially occlude the light path to one tray position — exactly this project's `shadow` cohort.
   The physics is straightforward radiometry: the light reaching a surface point depends on the source's
   irradiance at that point, the surface's reflectance, and the angle between the surface normal and the
   light direction (Lambert's cosine law for a diffuse surface) — none of which the camera can separate
   from "how bright is this part, really" without a calibrated setup. Project **01.09**'s photometric
   vignetting calibration and **01.11**'s sensor simulation both model pieces of this chain; this project
   treats "the pixel values I get depend on lighting I do not fully control" as a given and asks what
   similarity measure survives it.
2. **Pose.** A robot's placement has real, bounded tolerance: the gripper's own repeatability, small
   slip during release, and rotational play in how a part settles into a tray pocket all show up as a
   small translation and rotation between "where the part nominally is" and "where it actually is" — the
   physical origin of this project's `+-8` px search window and 5-angle rotation set.
3. **Sensor noise.** Every camera pixel's reported value is the true scene radiance PLUS photon shot
   noise (proportional to the square root of the signal — a fundamental quantum-optics limit, not an
   engineering defect) plus the sensor's own read/dark-current noise. `scripts/make_synthetic.py`'s
   per-pixel noise term is a simplified additive stand-in for this (see "Limitations & honesty").

Given all three, the engineering question is: **what similarity measure between a template and a live
window is invariant to (1) — illumination — while still being sensitive to (2) — is this actually the
right part in the right place?** The answer this project implements is **normalized cross-correlation**.

**Why correlation, not raw difference, survives illumination change.** A naive "how different are these
two images" measure — sum of squared differences (SSD), `sum((w_i - t_i)^2)` — treats every gray-level
unit the same regardless of WHY it differs: a genuinely wrong part and a correctly-lit-but-dimmer correct
part both produce large per-pixel differences, and SSD cannot tell them apart. The physical insight NCC
exploits is that a LIGHTING change (within the linear, unsaturated regime of the sensor) acts on pixel
values approximately as an AFFINE transform — `w_i -> a*w_i + b` for some brightness gain `a > 0` and
offset `b` — the SAME `a, b` for every pixel in a locally-uniform lighting patch. NCC is constructed to
be exactly invariant to that transform (derived below); SSD is not invariant to it at all. This
project's `illumination_robustness` gate MEASURES that gap directly, not just asserts it.

## The math

**Setup.** A template `t` and a same-sized window `w` are both `N = TEMPLATE_SIZE^2` (here `24^2=576`)
pixel patches, `w_i, t_i` for `i in [0, N)`, values in `[0, 255]`.

**Zero-normalized cross-correlation (ZNCC / "NCC" throughout this project):**

```
mean_w = (1/N) * sum(w_i)          mean_t = (1/N) * sum(t_i)

NCC(w, t) = sum( (w_i - mean_w) * (t_i - mean_t) )
            -----------------------------------------------------------
            sqrt( sum((w_i - mean_w)^2) * sum((t_i - mean_t)^2) )
```

This is exactly the **Pearson correlation coefficient** between the two patches, treated as `N`-vectors.
By the Cauchy-Schwarz inequality applied to the mean-subtracted vectors, `NCC in [-1, +1]`: `+1` is a
perfect match (up to a positive affine transform), `-1` is a perfect photographic-negative match, `0` is
uncorrelated.

**Invariance proof (why this is the RIGHT similarity measure here).** Let `w_i' = a*w_i + b`, `a > 0`.
Then `mean_w' = a*mean_w + b`, so `w_i' - mean_w' = a*(w_i - mean_w)` — the mean-subtracted values scale
by exactly `a` and are unaffected by `b`. Substituting into the formula: the numerator picks up a factor
of `a` (linear in `w`), and the denominator's `w`-half picks up a factor of `|a| = a` (a square root of
`a^2`) — they cancel exactly. `NCC(w', t) = NCC(w, t)`, for ANY `a > 0, b`. This is the formal statement
of "NCC survives a global brightness/contrast change" — the property `THEORY.md`'s "The problem" section
motivated physically and the `illumination_robustness` gate checks empirically.

**Rewriting in raw sums (the form every kernel actually computes).** Expanding the mean-subtracted sums
in terms of raw sums `S_w = sum(w_i), S_t = sum(t_i), S_ww = sum(w_i^2), S_tt = sum(t_i^2),
S_wt = sum(w_i * t_i)`:

```
numerator_unnorm = N*S_wt - S_w*S_t
var_w_unnorm      = N*S_ww - S_w*S_w        (this is N^2 * Var(w); always >= 0, Cauchy-Schwarz again)
var_t_unnorm      = N*S_tt - S_t*S_t

NCC = numerator_unnorm / sqrt(var_w_unnorm * var_t_unnorm)
```

This is the form `kernels.cuh` SECTION 6 documents and every kernel implements — and it is the classic
**sum-of-squares variance identity**, `Var(x) = E[x^2] - E[x]^2`, restated without division so it can be
computed in EXACT INTEGER arithmetic (the "Numerical considerations" section below is built entirely
around that one design choice).

**The integral-image algebra.** An INTEGRAL IMAGE (summed-area table) `II(x,y) = sum` of every pixel
with `x' <= x, y' <= y`, padded so `II(0,*) = II(*,0) = 0`. The sum over any axis-aligned box
`[x0,x1) x [y0,y1)` is then, by inclusion-exclusion:

```
box_sum = II(x1,y1) - II(x0,y1) - II(x1,y0) + II(x0,y0)
```

Proof sketch: `II(x1,y1)` counts everything up to `(x1,y1)`; subtracting `II(x0,y1)` removes everything
with `x' <= x0` (over-subtracting the `x'<=x0, y'<=y0` corner once); subtracting `II(x1,y0)` removes
everything with `y' <= y0` (over-subtracting the SAME corner a second time); adding `II(x0,y0)` back
restores it — the standard 2-D inclusion-exclusion argument (the same 4-term pattern this project's
sibling **01.13** uses for `WINDOW_STATS`-style box lookups). Building `II` for BOTH the running sum and
the running sum-of-squares turns `S_w` and `S_ww` for ANY box into 4 array reads + 3 adds — O(1),
independent of the box size `N` — which is the entire reason the sum-table kernel beats the naive one
(see "The GPU mapping").

## The algorithm

**Problem size.** `K = NUM_SLOTS = 24` slots, `M = NUM_TEMPLATES = 15` (3 types x 5 rotations),
`P = NUM_OFFSETS = 17*17 = 289` candidate offsets, `T = TEMPLATE_SIZE = 24` (`N = T^2 = 576`). The full
score volume is `K*M*P = 104,040` NCC evaluations.

**Serial cost.** A textbook direct implementation recomputes, for every evaluation, `S_w` and `S_ww`
(an `O(N)` re-scan of the window) AND `S_wt` (an unavoidable `O(N)` correlation sum) — `O(N)` work per
evaluation for the naive path's window-stats half, PLUS the numerator's `O(N)` every path pays:
`O(K*M*P*N)` total, roughly `104,040 * 576 ~= 60M` "pixel visits" just for the redundant half.

**The redundancy the sum-table removes.** `S_w, S_ww` depend ONLY on `(slot, offset)`, not on which of
the 15 templates is being scored — the naive kernel recomputes them **15x more often than necessary**
for a fixed `(slot, offset)` (once per template, when 15 templates share the same answer). Building the
integral image once (`O(W*H)`, `71,280` pixels for this project's tray — negligible next to `60M`) and
looking `S_w, S_ww` up in `O(1)` removes that entire redundant half, leaving only the unavoidable
`O(N)` correlation sum per evaluation: `O(K*M*P*N + W*H)`. MEASURED speed-up (Release, RTX 2080 SUPER,
this project's committed scene — see `demo/expected_output.txt`'s companion `[time]` line): **naive
0.33 ms -> sum-table 0.22 ms (~1.5x)** — smaller than the ~2x the "removed half the work" argument
predicts, because the correlation-sum half (unchanged between the two kernels) still dominates total
time at this problem size; the redundant-work argument is about ASYMPTOTIC scaling with `M` (more
templates -> naive gets relatively worse, sum-table does not), not a fixed constant.

**The shared-memory kernel's further win.** Even with `O(1)` window statistics, the numerator's `O(N)`
correlation loop still re-reads GLOBAL memory: 289 threads in one `(slot, template)` block read
overlapping `T x T` patches of the SAME `WINDOW x WINDOW` region and the exact SAME template — up to
`289x` redundant global-memory traffic for bytes that fit entirely in `2,176` bytes of shared memory.
Caching both once per block and reading them from shared memory for the correlation loop MEASURED a
further **0.22 ms -> 0.17 ms (~1.3x over sum-table, ~2.0x over naive)** — a real, if smaller-than-naive
speed-up (the correlation loop was always memory-latency-bound from L2/global; shared memory shortens
that latency, but 24x24 patches are small enough that the win is measured in tenths of a millisecond at
this problem size, not the order-of-magnitude wins shared memory gives on larger stencils).

## The GPU mapping

**Building the integral image — a 2-pass separable prefix SCAN.** A textbook parallel scan (Hillis-
Steele or Blelloch) achieves `O(log n)` DEPTH for an `n`-element prefix sum by recursively combining
partial sums — the right tool when a single line is itself huge. This project's lines are short
(`IMG_W=324`, `IMG_H=220`), so a SIMPLER two-level parallelism suffices and is more honest to read:
Pass 1 launches ONE THREAD PER ROW, each running a plain SEQUENTIAL `O(W)` accumulation across its own
row (`H=220` independent rows in parallel — the same "parallel ACROSS independent lines, sequential
WITHIN each line" idea project 01.13's separable Gaussian blur uses, here applied to a running sum
instead of a weighted stencil). Pass 2 launches one thread per COLUMN of pass 1's result, each running a
sequential `O(H)` accumulation down its column. Composing the two (proved in `kernels.cu`'s file header)
gives the true 2-D integral image. This is a valid, simpler-than-general-purpose-scan algorithm exactly
because the per-line lengths here are small enough that `O(W)`/`O(H)` sequential work per thread is
cheap — a genuinely different trade than a GPU histogram-equalization CDF over millions of bins would
need to make.

**The three-axis parallel mapping.** Every NCC kernel launches `grid = (NUM_SLOTS, NUM_TEMPLATES)`,
`block = (NUM_OFFSETS_1D, NUM_OFFSETS_1D) = (17,17) = 289` threads — three independent, embarrassingly
parallel axes (slot, template, offset) mapped directly onto `blockIdx.x`, `blockIdx.y`,
`(threadIdx.x, threadIdx.y)`. No evaluation depends on any other, so this is a pure 3-D "one thread per
independent problem instance" map — the same idiom as flagship 08.01's one-thread-per-MPPI-rollout,
extended from one parallel axis to three.

**Memory hierarchy, per kernel:**

| Kernel | Window stats (`S_w`, `S_ww`) | Correlation sum (`S_wt`) | Template stats (`S_t`, `S_tt`) |
|--------|-------------------------------|----------------------------|-----------------------------------|
| naive | direct `O(T^2)` re-scan, GLOBAL memory | direct `O(T^2)` loop, GLOBAL memory | `__constant__` memory (broadcast) |
| sum-table | `O(1)` box query, GLOBAL integral image | direct `O(T^2)` loop, GLOBAL memory | `__constant__` memory |
| shared | `O(1)` box query, GLOBAL integral image | direct `O(T^2)` loop, SHARED memory (cached once per block) | `__constant__` memory |

`__constant__` memory for the 15 templates' `(S_t, S_tt)` pairs is the right home for the same reason
project 01.13 puts its fixed-point Hough theta table there: EVERY thread scoring template `t` reads the
identical two numbers — a broadcast pattern the constant cache is built for, at a trivial `240` bytes.

**Shared-memory reuse economics.** A block's 289 offset-threads each read a `24x24` sub-patch of the
slot's `40x40` search window; by construction (`WINDOW = TEMPLATE_SIZE + 2*SEARCH_RADIUS`, exactly sized
so every candidate offset's patch stays inside it), the UNION of every patch any thread in the block
touches is precisely the whole `40x40` window. Caching that union (`1,600` bytes) plus the template
(`576` bytes) — `2,176` bytes total, a small fraction of a Turing/Ampere/Ada SM's 48+ KiB shared-memory
budget per block — turns up to `289x` redundant global reads of the same bytes into exactly one read
each, cooperatively loaded by all 289 threads in a grid-stride-style loop before the correlation loop
runs (`kernels.cu`'s `ncc_shared_kernel`).

## Numerical considerations

**The classic catastrophic-cancellation trap — and how integer arithmetic sidesteps it.** The
"sum-of-squares minus square-of-sum" variance identity, `Var(x) = E[x^2] - E[x]^2`, is a textbook
numerical-analysis WARNING when computed in FLOATING POINT: for a low-variance sample with a large mean
(e.g. a near-uniform 24x24 patch, `mean ~ 190`), `E[x^2]` and `E[x]^2` are both large, close numbers
(`~36,100`), and their FLOATING-POINT difference — the true (small) variance — can lose most of its
significant digits to rounding, because EACH operand already carries its own accumulated rounding error
from the summation, and cancellation AMPLIFIES that pre-existing error relative to the small true
result. This project sidesteps the trap entirely, not by avoiding the formula, but by keeping every sum
(`S_w, S_ww, S_wt, S_t, S_tt`) as an EXACT 64-bit INTEGER, computed with zero rounding at every step
(integer addition and multiplication are exact until they overflow the type). The subtraction
`N*S_ww - S_w*S_w` is then an EXACT integer subtraction of two EXACT integers — there is no pre-existing
rounding error for the subtraction to amplify, so the only possible failure mode left is INTEGER
OVERFLOW, a well-understood, boundable hazard (worked below), not silent precision loss.

**The overflow analysis (worked, not asserted).** `uint8` pixel values are `[0, 255]`.

*Integral-image tables (cover the WHOLE tray, the largest accumulation domain in this project):*
For a `320x240` image (the canonical size project 01.13 also uses — a useful generic reference point):
`76,800` pixels, worst case ALL at `255`. Running SUM's corner entry: `76,800 * 255 = 19,584,000` —
fits `uint32` with `~7` bits to spare (needs `25` bits). Running SUM-OF-SQUARES' corner entry:
`76,800 * 255^2 = 76,800 * 65,025 = 4,993,920,000 ~= 4.99e9`, which EXCEEDS `UINT32_MAX = 4,294,967,295
~= 4.29e9` — a `uint32` sum-of-squares table for a `320x240` image OVERFLOWS. This project's OWN tray
(`324x220 = 71,280` pixels) makes the SAME point concretely, not hypothetically: worst-case sum-of-squares
`71,280 * 65,025 = 4,634,982,000 ~= 4.63e9`, still over `UINT32_MAX` by more than `300` million. This is
exactly why `kernels.cuh` SECTION 4 types `II_SUM` as `uint32_t` but `II_SUMSQ` as `uint64_t` — the sum
table genuinely does not need the wider type (a real, useful contrast: overflow risk is a property of
the ACCUMULATION DOMAIN size and the operation, `sum` vs `sum-of-squares`, not a blanket "always use the
widest type" rule).

*Per-template / per-box statistics (cover only `N=576` pixels — much smaller, but still worth checking):*
worst case `S_t, S_w <= 576*255 = 146,880` and `S_tt, S_ww <= 576*65,025 = 37,454,400` — BOTH fit `uint32`
comfortably (`< 2^26`). The box-QUERY RESULT is always small even though the TABLE it was computed from
can be large — the table's storage type must accommodate its own worst-case CORNER entry (the whole-image
sum), not the much smaller worst-case of any individual box subtracted from it.

*The combining algebra's PRODUCTS (a second, easy-to-miss overflow site).* `numerator_unnorm = N*S_wt -
S_w*S_t`: `N=576`, `S_wt <= 37,454,400`, so `N*S_wt` alone reaches `~2.16e10` — nearly `5x` past
`UINT32_MAX`, needing `int64_t` even though `S_wt` itself (`~26` bits) fits comfortably in `uint32`. The
general lesson: the PRODUCT of two `~26`-bit quantities can need up to `~52` bits, far past a 32-bit
type's range, regardless of how safe each FACTOR looked in isolation — every such product in this
project's code is computed in `int64_t` from the start for exactly this reason.

*The final pre-sqrt product — the one genuine (if exotic) residual risk.* `var_w_unnorm` and
`var_t_unnorm` are each bounded, by Cauchy-Schwarz, by roughly `N^2 * (255/2)^2 ~= 576^2 * 16,256
~= 5.39e9` in the theoretical worst case (a perfectly bimodal `0`/`255` checkerboard template AND
window — an adversarial case no real photograph produces, but worth bounding honestly). Their PRODUCT
could then reach `~2.9e19`, which EXCEEDS even `uint64_t`'s range (`~1.8e19`). `kernels.cu`'s
`ncc_from_sums` therefore promotes BOTH variance terms to `double` BEFORE multiplying them (not after),
sidestepping the overflow entirely — a deliberate, documented choice, not an oversight (and it also
happens to be where the unavoidable floating-point `sqrt` has to enter regardless).

**Float rounding — the ONE place this project is not bit-exact.** Everything through
`numerator_unnorm`, `var_w_unnorm`, `var_t_unnorm` is exact 64-bit integer arithmetic, verified
BIT-EXACT between GPU and CPU (`main.cu`'s `VERIFY:` lines for the integral images and window
statistics). The final `sqrt` + divide is IEEE-754 double-precision floating point on both paths — the
one step where device (`nvcc`) and host (MSVC) rounding COULD, in principle, differ by a few ULP.
MEASURED on this project's committed scene (RTX 2080 SUPER, CUDA 13.3, MSVC 14.51): the worst observed
`|GPU - CPU|` NCC score, over all `104,040` evaluations, was **exactly `0.0`** — both toolchains'
`sqrt`/divide happened to round identically here. `main.cu`'s tolerance (`5e-4`) is real headroom for
platforms where that equality is not guaranteed, not a number chosen to paper over an observed gap.

**Rotation is NOT handled by this numerics story at all.** NCC's invariance proof above covers brightness
AFFINE changes only; a ROTATED part is a genuinely different pixel pattern, not an affine transform of
the original, so NCC offers no rotation invariance whatsoever — the entire reason this project evaluates
a discrete rotation SET rather than relying on the 0-degree template alone (see "How we verify
correctness" below for the measured falloff, and README "Limitations & honesty" for the scope this
implies).

## How we verify correctness

**What is and is not twinned** (the full independence ruling lives in `src/reference_cpu.cpp`'s header):

| Stage | GPU | CPU oracle | Comparison |
|-------|-----|------------|------------|
| Integral images | 2-pass separable scan | single-pass 2-D recurrence (a DIFFERENT algorithm) | **bit-exact** (integer) |
| Window statistics | `O(1)` box query | independently-typed box query | **bit-exact** (integer) |
| NCC score volume (all 3 kernels) | naive / sum-table / shared | independent box-query + correlation oracle | float tolerance (measured `0.0` worst case — see "Numerical considerations") |
| Cross-GPU-variant agreement | — | — | pairwise, measured `0.0` worst case, tolerance `1e-6` |
| Classification / localization / rotation / illumination | host-only downstream analysis (not twinned) | — | 5 independent GATES against known synthetic truth |

**The measured per-cohort score table** (this project's committed scene, `T_OK = 0.65`):

| Cohort | Slot | Expected-template score | Notes |
|--------|------|--------------------------|-------|
| `plain` (19 slots) | various | `0.996 - 0.997` | well-placed correct parts; texture+noise keep this under `1.0` |
| `offset` | 3 | `0.996` | `(+5,-6)` px placement error, fully recovered by the `+-8` px search |
| `wrong_part` | 5 | `0.449` (expected type) / `0.996` (actual type) | a bracket scores near-perfectly against ITS OWN template, and far below `T_OK` against the connector block that was expected |
| `empty` | 4 | `0.053 - 0.065` | pure background; near-zero correlation with any template |
| `rotated` | 6 | `0.612` (single 0-degree) / `0.691` (best of 5-angle set) | see "The algorithm"/`rotation_lesson` below |
| `shadow` | 7 | `0.984` | illumination-gradient-degraded, still confidently matched |

**`rotation_lesson`'s falloff curve** (measured, `demo/out/score_vs_angle.csv`, true rotation `24`
degrees): `-6->0.538, -3->0.561, 0->0.612, +3->0.636, +6->0.691` — monotonically increasing as the
rotation-set template's angular distance from the true `24` degrees shrinks (`+6` is `18` degrees away,
the closest available; `-6` is `30` degrees away, the farthest), exactly the trend NCC's total lack of
rotation invariance predicts. The gate's two asserted points (`single < 0.65 <= best-of-set`) are the
two ends of this same measured curve, not independent numbers.

**`illumination_robustness`'s designed comparison** (measured): the shadowed slot's best NCC score
(`0.984`) clears `T_OK` with a wide margin; the SAME slot's best SSD score (`ssd_best`, minimized over
the search window against the SAME 0-degree template) is `1,177,358` — **123x** the SSD a same-type,
non-shadowed baseline slot achieves (`9,548`), and **62x** past a reject threshold set at `2x` that
baseline. A plain, non-normalized SSD matcher would emphatically REJECT this genuinely-correct match;
NCC does not. This is the concrete, measured version of "The problem" section's invariance argument, not
a restatement of it.

## Where this sits in the real world

**OpenCV's `cv::matchTemplate(..., TM_CCOEFF_NORMED)`** computes exactly the ZNCC formula this project
derives, and its CPU implementation uses the same sum-table acceleration (`cv::integral` internally) —
this project's three kernels are a from-scratch, commented walk through what that one library call does
and why it is fast. OpenCV's CUDA module additionally offers `cv::cuda::TemplateMatching`, the
production GPU version.

**FFT-domain correlation** (documented here, not implemented — the catalog bullet's "documented-only"
scope): the `O(N)`-per-evaluation direct correlation sum this project cannot avoid becomes, for a FULL
scene-wide search, an `O(W*H*log(W*H))` FFT-based cross-correlation instead of `O(W*H*N)` direct sliding —
a real win once the search area is large enough that `log(W*H)` beats `N`, the SAME crossover argument
project **03.01**'s FMCW radar cube uses cuFFT for. This project's search area per slot (`17x17=289`
candidate positions) is far too small for that crossover to ever favor an FFT — the direct sum is both
simpler AND faster here — but a learner extending Exercise 2 (full-image search) would cross that
threshold quickly.

**Commercial geometric / edge-based pattern matching** (Cognex PatMax, MVTec Halcon's shape-based
matching) is what a real production line uses instead of pixel-correlation NCC specifically because
these libraries are robust to CONTINUOUS rotation AND scale (not a discrete angle set), partial
occlusion, and clutter — they match GEOMETRIC FEATURES (edges, contours) extracted at multiple scales, a
fundamentally different (and much more expensive to implement well) representation than this project's
raw-pixel correlation. Project **01.13**'s Canny+Hough pipeline is this repository's own step in that
direction — a real edge/geometric detector, though not (yet) fused with a matcher — worth reading
alongside this project for the contrast.
