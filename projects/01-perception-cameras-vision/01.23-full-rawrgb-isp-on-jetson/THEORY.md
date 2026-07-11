# 01.23 — Full RAW→RGB ISP on Jetson (Argus + custom CUDA stages): Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md section 4.2).

## The problem — physics & engineering first

**From photons to electrons to DN.** A camera sensor is a grid of photosites; each accumulates
photoelectrons proportional to the light striking it during the exposure, then an ADC converts that
charge to a digital number (DN). Four physical facts every ISP stage below exists to undo:

1. **Every photosite is covered by exactly one color filter** (red, green, or blue — the Bayer
   Color Filter Array, or CFA), arranged RGGB in this project (and most consumer sensors): so a raw
   frame has **one** color sample per pixel, not three. Reconstructing the missing two is *demosaic*
   — see "The algorithm" below. This project's sibling **01.09** and **01.11** both build on the same
   raw-sensor physics (respectively: the vignetting this project also corrects, and the shot/read
   noise this project's synthetic model also injects but does not denoise).
2. **No color filter is perfectly narrowband.** Physically, a dye or interference filter still
   transmits some light outside its target band — a "red" filter still lets a little green and blue
   through. This *spectral crosstalk* means each raw sample is a **mixture** of R, G, and B light, not
   a pure sample of one — undone by the color-correction matrix (CCM), derived below.
3. **A lens does not deliver uniform illuminance to the sensor.** The cosine-fourth law of
   illumination falloff (irradiance at the sensor drops roughly with the fourth power of the cosine
   of the off-axis angle for a simple lens, with real lenses somewhat better or worse depending on
   design) means corner pixels receive systematically less light than the center — *vignetting* or
   *lens shading* — corrected by dividing out a measured or modeled radial gain map (sibling **01.09**
   is this repo's dedicated calibration-kernels project for exactly this effect; this project reuses
   its functional form).
4. **Manufacturing is imperfect.** A sensor with millions of photosites always ships with a handful
   of dead, stuck, or noisy ones — a *defect map*, measured at the factory and corrected at runtime.

**Engineering constraints a real robot imposes.** Read noise (the sensor's electronic noise floor,
independent of light level) and shot noise (the fundamental statistical noise of counting discrete
photons, growing with the square root of signal — this project models both, see "The math") set a
noise floor no ISP stage can remove without a dedicated denoiser (out of scope here, see sibling
**01.11**). A real ISP also has a hard **latency budget**: it must keep up with the sensor's frame
rate (30–60 Hz for a typical robot camera) with enough margin left for everything downstream — see
README "System context" for the numbers. **Bandwidth**: at 60 Hz and even a modest 2 MP RAW12 sensor,
the raw stream alone is ~1.4 Gbit/s before any processing — real ISPs are engineered around memory
bandwidth first, arithmetic throughput a distant second (exactly why every kernel in this project is
described below as *memory-bound*, not *compute-bound*).

## The math

**Notation.** Scalars in `[0,1]` unless noted; RGB vectors written `(r,g,b)`; matrices row-major,
applied as `y = M x`. Pixel coordinates `(x,y)`, `x` right, `y` down (image convention, same as every
other project in this repo's camera-vision domain). All quantities in this project are
dimensionless (normalized sensor code or reflectance), not SI physical units — a deliberate
simplification named honestly in "Numerical considerations."

**Bayer phase.** `bayer_phase_at(x,y)` returns one of four cases (RGGB, period 2 in each axis):
`R` at even row/even col, `Gr` (green, red row) at even row/odd col, `Gb` (green, blue row) at odd
row/even col, `B` at odd row/odd col. The distinction between `Gr` and `Gb` does not matter for
white balance (both are the same green filter) but **does** matter for demosaic (see below): a `Gr`
pixel's horizontal neighbors are red, a `Gb` pixel's horizontal neighbors are blue.

**Black level & saturation (stage 1).** `bl(x,y) = clamp((raw(x,y) - black) / (white - black), 0, 1)`.
This project uses `black=64`, `white=1023` (10-bit codes, RAW10-in-uint16 container — see
`kernels.cuh` section 0).

**Lens shading (stage 2).** A 2-term radial polynomial (a truncation of sibling 01.09's 3-term
model): `V(r) = 1 + a2·r² + a4·r⁴`, `r = |p - c| / r_norm` (normalized so `r=1` at the image's
farthest corner). Correction divides by `V(r)`, floored at `kShadeGainFloor` (a division guard,
inactive at this project's chosen coefficients — `V(1) = 0.75` — but present because a correction
stage that can silently divide by a near-zero gain is a real, reported ISP failure mode).

**Illuminant physics & chromatic adaptation (stage 4, one paragraph).** Different light sources emit
different relative amounts of red/green/blue energy — a tungsten bulb (~2856K) is red-heavy and
blue-poor compared to daylight (D65, ~6500K). A camera pointed at a *truly neutral* gray surface
therefore records **different raw RGB ratios** depending on what light is illuminating it — the
"white balance problem." The human visual system does something similar and compensates for it
(chromatic adaptation — the reason a white sheet of paper looks white to you under both a candle and
the midday sun even though the *light itself* is very different colors); the classical computational
model of this adaptation is the **von Kries hypothesis**: each cone/color channel's response is
independently rescaled by a gain that depends only on the illuminant, not on the scene content. This
project's white-balance stage **is** a von Kries-style correction: `wb(x,y) = mosaic(x,y) · gain[phase]`,
a single per-channel scalar gain, chosen so a neutral surface reads equally in all three channels
after correction. This project's synthetic model represents the illuminant's effect exactly this way:
`kIlluminantD65Gain = (1,1,1)` (the neutral reference), `kIlluminantTungstenGain = (1.42, 1.00, 0.53)`
(illustrative, not measured — `data/README.md` "Provenance & honesty notes").

Two estimators of `gain[phase]` from scene content alone (no metadata, exactly the real problem a
camera solves every frame):
- **Gray-world**: assume the AVERAGE of the whole scene is neutral gray. `gain[c] = mean(G) / mean(c)`.
  Fails predictably when the scene's average color is genuinely NOT gray — e.g. a photo dominated by
  a red wall. This project's `awb_red_crop_failure` gate demonstrates exactly this failure on
  purpose, restricting gray-world to a crop of only warm chart patches.
- **White-patch / max-RGB**: assume the BRIGHTEST pixel is a white or specular highlight, and that a
  true white reflects all three channels equally. `gain[c] = max(G) / max(c)`. Fails when no genuine
  white/specular region exists in frame, or succeeds/fails depending on noise at the single brightest
  sample (a real, named limitation — production white-patch estimators use a percentile or clipped
  average, not a literal maximum; `kernels.cu`'s AWB reduction kernel documents this simplification).

**Spectral crosstalk and the CCM — derived by hand.** The synthetic sensor's crosstalk matrix (each
row = one color filter's relative response to R/G/B light, rows summing to 1 so a spectrally flat
input reproduces itself — `kernels.cuh` section 2):

```
        [ 0.72  0.22  0.06 ]
    M = [ 0.10  0.78  0.12 ]
        [ 0.06  0.20  0.74 ]
```

The CCM undoes this mixing: `CCM = M⁻¹`. Full 3×3 inversion (cofactor expansion), worked by hand:

```
det(M) = 0.72·(0.78·0.74 − 0.12·0.20) − 0.22·(0.10·0.74 − 0.12·0.06) + 0.06·(0.10·0.20 − 0.78·0.06)
       = 0.72·0.5532 − 0.22·0.0668 + 0.06·(−0.0268)
       = 0.398304 − 0.014696 − 0.001608 = 0.382000

Cofactors:
  C11=+0.5532  C12=−0.0668  C13=−0.0268
  C21=−0.1508  C22=+0.5292  C23=−0.1308
  C31=−0.0204  C32=−0.0804  C33=+0.5396

CCM = adj(M)/det(M) = transpose(cofactor matrix)/0.382 =

        [  1.44817  −0.39476  −0.05340 ]
  CCM = [ −0.17487   1.38534  −0.21047 ]
        [ −0.07016  −0.34241   1.41257 ]
```

**A free correctness check, worth stating explicitly:** because every row of `M` sums to 1 (a
spectrally flat input reproduces itself — `M · (1,1,1)ᵀ = (1,1,1)ᵀ`, i.e. `(1,1,1)ᵀ` is a right
eigenvector of `M` with eigenvalue 1), `M⁻¹` shares this eigenvector automatically:
`(1,1,1)ᵀ = M⁻¹ · M · (1,1,1)ᵀ = M⁻¹ · (1,1,1)ᵀ`. So **every row of the CCM above also sums to 1** —
verify it yourself (`1.44817 − 0.39476 − 0.05340 = 1.00001`, rounding). This means the CCM introduces
**no white-point shift**: unlike the general CCM-derivation case (where a separate white-point
renormalization step is required), this project's CCM needs none — a direct consequence of choosing
a row-stochastic crosstalk matrix. Spot-check: `M · CCM ≈ I` (verified numerically to 5 decimal
places when this project's constants were derived; a residual exercise for the reader: verify
`(row 1 of M) · (col 2 of CCM) ≈ 0` by hand).

**MHC demosaic — the gradient-correction argument.** A naive (bilinear) demosaic estimates a missing
channel as the plain average of same-color neighbors — e.g. `G` at an `R` site = average of the four
orthogonal `G` neighbors. This ignores information the OTHER channels carry: natural images have
strongly **correlated** luma (brightness) edges across R, G, and B — where the scene has an edge, all
three channels usually have one too, at the same place. Malvar-He-Cutler (2004) exploits this: it
still averages same-color neighbors, but **adds a Laplacian-style correction term** built from the
*center pixel's own native channel*, which carries the local high-frequency (edge) content the
same-color average alone would blur. Concretely, all four MHC kernels below are 5×5, applied to the
RAW MOSAIC directly (not a per-channel-separated image — Bayer's own 2-pixel periodicity puts the
right same/cross-phase samples at the right taps automatically), coefficients in **eighths**:

```
G at R or B:              R at Gr / B at Gb           R at Gb / B at Gr          R at B / B at R
                           (horizontal emphasis)       (vertical emphasis,        (diagonal)
                                                        = transpose of left)
 0  0 -1  0  0              0  0  .5  0  0              0  0 -1  0  0            0    0  -1.5 0    0
 0  0  2  0  0              0 -1  0  -1  0              0 -1  4 -1  0            0    2   0   2    0
-1  2  4  2 -1             -1  4  5   4 -1              .5 0  5  0 .5           -1.5  0   6   0  -1.5
 0  0  2  0  0              0 -1  0  -1  0              0 -1  4 -1  0            0    2   0   2    0
 0  0 -1  0  0              0  0  .5  0  0              0  0 -1  0  0            0    0  -1.5 0    0
```

(all four sum to 8, i.e. normalize to exactly 1 — verified by hand for each table when this project
was built: a flat/constant scene demosaics back to itself exactly, the correctness sanity check every
demosaic kernel should pass trivially). Kernel selection: at an `R` site, `G` uses the "G at R/B"
table and `B` uses the diagonal table (and symmetrically at a `B` site); at a `Gr` site, `R` (its
horizontal same-row neighbor color) uses the horizontal-emphasis table and `B` (vertical) uses the
vertical-emphasis table (and symmetrically at `Gb`). Coefficients as commonly reproduced in the
demosaicing literature — verify against the original 2004 paper before use outside this teaching
context (CLAUDE.md section 8 honesty about sourcing).

**sRGB transfer function (stage 7 — perceptual coding, one paragraph).** Human brightness perception
is roughly logarithmic, not linear — we are far more sensitive to differences among dark tones than
among bright ones. Storing linear light in only 8 bits per channel would waste most of those bits on
distinctions the eye cannot see in bright regions while badly under-resolving shadows (visible
banding). The sRGB transfer function — a near-power-law with exponent ~1/2.4 (a "gamma" curve,
though the true standard is the exact piecewise function below, not a pure power law) — allocates
more of the 8-bit code space to darker tones, matching perceptual sensitivity:

```
encode(L) = 12.92·L                          if L ≤ 0.0031308
          = 1.055·L^(1/2.4) − 0.055          otherwise
```

Terminology note (an honest, common mix-up): the function above, applied linear→encoded, is
technically the sRGB **OETF** (opto-electronic transfer function); "EOTF" formally names the
**opposite** direction (a *display's* encoded→linear response). Casual usage calls both "gamma."

## The algorithm

Eight sequential stages, each documented with its complexity (`n = W·H` raw pixels):

1. **Black level + saturation** — `O(n)`, one flop-light map, no neighbors.
2. **Lens shading** — `O(n)`, one map, `shading_gain_at(x,y)` is `O(1)` per pixel (a few flops).
3. **Defect correction** — `O(n·d)` where `d` = defect-list length (this project: 16, effectively a
   small constant) for the membership scan, `O(1)` extra (4 neighbor reads + a 5-compare sort) only
   for the (rare) defective pixels; effectively `O(n)`.
4. **White balance** — a 2-phase pipeline: `O(n)` reduction (block-tree sum+max, `O(log(block size))`
   depth) to estimate gains, then `O(n)` map to apply them.
5. **Demosaic** — `O(n)` map; MHC's constant factor is larger (25-tap stencil, several skipped-zero
   taps) than bilinear's (9-tap), but both are `O(n)` overall — see "The GPU mapping" for the actual
   tap count that matters.
6. **CCM** — `O(n)`, 9 multiplies + 6 adds per pixel, no neighbors.
7. **Gamma** — `O(n)`, one `pow()` call (or the linear branch) per channel per pixel.

Serial cost on a CPU: `O(n)` with a modest constant (dominated by the 5×5 demosaic stencil); this
project's CPU oracle (all stages, D65, `n=19,200`) measures **≈3 ms** on the reference machine. There
is no algorithmic step in this pipeline with worse-than-linear complexity in `n` — the entire teaching
point of an ISP being "the textbook first GPU program" is that **every stage is embarrassingly
parallel across pixels**, so the GPU mapping (next section) is close to the simplest possible one at
every stage except the AWB reduction.

## The GPU mapping

**Thread-to-data mapping** is `i = blockIdx.x*blockDim.x + threadIdx.x`, `(x,y) = (i%W, i/W)` for
every map/stencil kernel (a 1-D grid over the mosaic — simpler to read than a 2-D block at this
project's small resolution, and every kernel needs `(x,y)` anyway for `bayer_phase_at`/
`shading_gain_at`). Block size 256 (a warp multiple) throughout.

**Memory hierarchy, stage by stage:**
- **Global memory** dominates every stage — this is a bandwidth-bound pipeline, not a compute-bound
  one (see "The problem"'s bandwidth arithmetic). No stage reuses enough data between threads to
  justify shared-memory tiling at this project's teaching scale (contrast with a convolution over a
  LARGE image, where shared-memory tiling amortizes the 5×5 stencil's redundant reads across a whole
  thread block — an explicit exercise-territory optimization this project's THEORY names but does not
  implement, matching sibling 01.01's census-kernel precedent of the same honest simplification).
- **Constant memory** — the ONE deliberate exception: the defect list (`g_defect_x`/`g_defect_y`,
  `kernels.cu`) and the four MHC coefficient tables (`d_kMhcG` etc.) are declared `__constant__`.
  Every thread in the grid reads the SAME small, read-only array — exactly the broadcast-cache use
  case constant memory exists for. A `global`-memory version would work identically but waste L1/L2
  bandwidth on redundant reads of the same handful of values by every one of thousands of threads.
- **Registers** — the demosaic stencil's five accumulator variables (`R, G, B, native, acc`) and the
  AWB reduction's six per-thread partials (three sums, three maxes) live entirely in registers; no
  spilling at this project's modest per-thread state.

**The 5×5 stencil regime (demosaic).** `mhc_eval()`'s loop is `#pragma unroll`ed into 25
straight-line taps; roughly half of each table's 25 entries are exactly zero (see the tables above)
and are skipped at the SOURCE level (`if (w == 0.0f) continue;`), so the compiler need not even
schedule a load for them — the REAL per-pixel tap count is 9–13 depending on which of the four tables
is active, not 25. Border pixels clamp neighbor coordinates to the image edge (the same "repeated
edge pixel" bias every stencil kernel in this repo accepts and names, e.g. 01.01's debayer).

**Fusion economics — stages 1–4, extending 01.01's staged-vs-fused lesson.** 01.01 fused kernels
whose dependency was on *materialized DATA* (four bilinear samples of a full remapped image); this
project's stage-3-on-stage-1/2 dependency is on a *materialized FORMULA* instead — black level +
shading is four FLOPs, a pure function of `raw[]` and `(x,y)`, nothing another kernel's OUTPUT. That
makes the fusion below unusually cheap:

```
FUSED kernel, one thread per raw pixel:
    bl_sh = bl_shading_at(raw, x, y)                    // always: 1 recompute (own pixel)
    if defective:
        bl_sh = median4( bl_shading_at(raw, x, y-2),    // recompute (up to) 4 EXTRA times
                          bl_shading_at(raw, x, y+2),    // -- but ONLY for ~16 of 19,200 pixels
                          bl_shading_at(raw, x+2, y),
                          bl_shading_at(raw, x-2, y) )
    out = bl_sh * wb_gain[phase]
```

For all but the ~0.08% defective pixels, the fused kernel does **exactly** the same FLOP count as the
staged path's black-level kernel alone — the shading and white-balance steps are folded in for free —
while STILL eliminating three full-resolution intermediate buffers' write-then-read round trip
through global memory. Measured (idealized, no-cache-reuse byte model, D65, `n=19,200`): staged
345,600 bytes vs fused 115,200 bytes — a **66.7%** reduction (vs. 01.01's fused case, which saves a
smaller fraction because its fusion recomputes real *extra* work — four bilinear samples per output
pixel — where this project's fusion recomputes real work for almost nobody). Measured kernel time:
staged ≈0.16–0.19 ms, fused ≈0.05 ms on the reference machine (RTX 2080 SUPER). `main.cu`'s
`fused_vs_staged` gate confirms the two paths agree to `1e-4` — see "Numerical considerations" for why
they are not required to be bit-identical (they are, in practice, at this project's precision; the
formal argument for why they *must* agree follows next).

**Why median-then-scale equals scale-then-median.** The fused kernel applies white balance AFTER the
median (for a defective pixel); the staged path applies it in a wholly separate final pass. These
give IDENTICAL results because all same-phase neighbors of one defective pixel share exactly ONE
gain value, and multiplying every element of a set by the same positive scalar before taking an
order statistic (median) commutes with taking the median first and then scaling:
`median(g·a, g·b, g·c, g·d) = g · median(a,b,c,d)` for `g > 0`. This is not a numerical coincidence
this project's tolerance happens to absorb — it is an algebraic identity, which is exactly why the
measured `fused_vs_staged` gap (0.000000 on the reference machine) is essentially machine-epsilon,
not "small but real."

## Numerical considerations

**uint16 headroom through the chain — the arithmetic, done by hand.** The raw sample is a 10-bit code
(0–1023) stored in a 16-bit container — 6 bits of unused headroom at the INPUT. After black-level
subtraction and normalization, the pipeline works entirely in **float32**, so integer headroom stops
being the relevant question after stage 1 — the float pipeline's actual risk is not overflow but
**clipping loss of information**: white balance can push a channel's normalized value above 1.0 (a
gain of 1.89 — this project's measured tungsten blue-channel gray-world gain — applied to an
already-bright blue pixel can exceed 1.0 easily), and the CCM's negative off-diagonal terms
(`−0.39476`, `−0.34241`, ...) can push a channel below 0.0 at a strong color edge (this project's
"Exercise 1" ringing artifact is exactly this: MHC's own negative-lobe undershoot, amplified further
by the CCM's negative coefficients, occasionally clipped hard at both ends). This project clamps
**exactly once**, at the very last step (`srgb_encode`'s input clamp to `[0,1]`) — every earlier stage
that could produce an out-of-range value (WB gain, CCM, MHC's negative lobes) is left alone
deliberately, so headroom is preserved as long as possible and only the FINAL display-encoding step
decides what "out of range" means. A real 12-bit-ADC sensor's extra headroom above its nominal 10-bit
signal range exists for exactly this reason — to survive a WB gain applied in the RAW domain (this
project's WB stage, stage 4) without clipping before the CCM has a chance to correct the color.

**Float vs. fixed-point.** This project is float32 throughout (registers, not fixed-point/integer
arithmetic) — the didactic default for this repo (CLAUDE.md section 12) and the realistic choice for
a desktop/Jetson-class GPU ISP (fixed-function *hardware* ISPs, by contrast, often use fixed-point
internally for silicon-area and power reasons — PRACTICE.md section 2 names this trade-off for real
chips).

**Gamma at the toe.** The sRGB encode function's LINEAR segment near black (`L ≤ 0.0031308`, slope
`12.92`) exists specifically to avoid the infinite slope a pure power law (`L^(1/2.4)`) would have at
`L=0` — a numerically important detail: without the linear toe, small floating-point noise near black
would map to wildly different 8-bit codes. This is also why this project's flat, dark background
region (reflectance ~0.03 linear) is the most numerically SENSITIVE region in the whole synthetic
scene for the end-to-end PSNR gate — small linear-domain errors there are proportionally amplified
more than the same absolute error in a bright region, because the gamma curve's slope is steepest
near black.

**Determinism.** The AWB reduction (`kernels.cu`) uses the SAME deterministic two-level block-tree
pattern as sibling 01.01's normalize stage — no atomics anywhere, a fixed summation order, bit-
reproducible run to run on one GPU. Both the sum (gray-world) and max (white-patch) trees are
combined in ONE kernel to avoid a second full pass over the mosaic.

**Where noise matters.** This project's synthetic sensor injects shot+read noise
(`σ² = σ_read² + k_shot·signal_DN`, `σ_read=2.0 DN`, `k_shot=0.02`) — small (≈0.3–0.5% of full range)
by design, so it is a real but minor contributor to every stage-truth gate's residual, never the
dominant error source (that honor goes to demosaic error at hard edges — see "How we verify
correctness" below). No stage in this pipeline denoises; the noise a real ISP's stages 1–3 do NOT
remove is exactly what a dedicated denoiser (sibling 01.11) exists for.

## How we verify correctness

**The CPU reference** (`reference_cpu.cpp`) independently retypes every stage's application logic
(the neighbor clamping, the median network, the tap-gathering, the reduction arithmetic) while
sharing exactly the DOCUMENTED "hardware fact" formulas with the GPU kernels (`bayer_phase_at`,
`shading_gain_at`, `srgb_encode`/`decode`, `ccm_apply_at`, and the MHC coefficient TABLES — never the
stencil-application code around them) — the same twin-independence ruling sibling 01.01 established,
restated in full in `reference_cpu.cpp`'s file header. **Measured on the reference machine:** every
float stage agrees GPU-vs-CPU to `max|gpu-cpu| = 0.000000` (bit-exact on this architecture — a
stronger result than the `≤5e-4` tolerance demands, itself set to survive legitimate FMA-contraction
differences across `sm_75`/`sm_86`/`sm_89`, never observed as needed here but kept honest for
portability).

**Edge cases exercised:** the defect list deliberately sits only in flat background (so its
"recovery" target is a locally uniform truth — `scripts/make_synthetic.py`'s placement logic);
`awb_red_crop_failure` deliberately exercises a KNOWN failure mode (a "must fail" gate — the reader
should not read a `PASS` there as "AWB always works," but as "gray-world's known failure mode is
real, measured, and bounded"); `tungsten_wrong_awb_negative_control` deliberately mismatches AWB
gains to the wrong illuminant, proving the measured color cast is attributable to AWB, not to some
other stage.

**Where a stage-truth gate's tolerance comes from.** `black_level_residual` and `shading_flatness`
compare against `true_sensor_rgb_d65.bin` (the generator's noiseless, pre-shading ground truth) — the
residual these gates measure is therefore almost entirely the injected sensor noise (σ ≈0.3–0.5% of
range), and the tolerance (0.030, measured 0.002–0.003) carries roughly 10× margin. `demosaic_psnr`
and `end_to_end_psnr` are calibrated differently: PSNR against a scene with genuinely hard content
(a 24-patch chart with real edges, a deliberately adversarial hashed texture) is fundamentally lower
than PSNR on a smooth photograph, so these floors were set from an ACTUAL measured run (33.31/31.79
dB demosaic; 26.36/23.47 dB end-to-end) with margin below the measurement, not from an assumed
"good" number — CLAUDE.md section 8's "never fabricate" rule applied to gate design itself, not just
to reported results. A mild [1,2,1]/4 pre-mosaic blur (`scripts/make_synthetic.py`'s `blur_scene`) —
modeling every real lens's point-spread function and every real sensor's optical low-pass filter —
was added specifically because the FIRST version of this project's scene (unblurred, literal
step-function edges) made even MHC ring badly enough to swamp every downstream gate with a
demosaic-only artifact unrelated to the ISP stages actually under test; see "Where this sits in the
real world" for why this project needed to add, by hand, an effect real hardware provides for free.

## Where this sits in the real world

**Jetson's hardware ISP, libargus, and V4L2** — the concrete production stack this project's stages
1–7 stand in for; README "The Jetson story" and `PRACTICE.md` section 3 give the full architectural
and bring-up picture. In one sentence: on Jetson, most of this project IS already built, in
fixed-function silicon, running at full frame rate for a fraction of the GPU's power draw — custom
CUDA stages earn their keep only where algorithm control, non-standard sensors, or (as here)
transparency/teaching matter more than that efficiency.

**libcamera** (the open-source Linux camera stack, used by Raspberry Pi OS and others) implements a
full software AWB/AE/demosaic pipeline in the open — a genuinely excellent next read once this
project's stages feel familiar, since it shows the SAME algorithms (gray-world-family AWB, a modern
demosaic) integrated into a real, shipping, multi-sensor stack with a tuning-file system.

**Phone ISPs** (Apple, Google, Samsung's camera pipelines) do everything this project does, plus:
multi-frame HDR fusion (sibling 01.08's territory), learned demosaic/denoise (replacing MHC-family
hand-designed kernels with a trained network in the flagship tier), and per-scene AI-driven AWB/AE
(scene classification feeding the gray-world/white-patch-style estimators this project implements
with a learned prior).

**The tuning-guide profession, named honestly.** Every real ISP — hardware or software — ships with
dozens to hundreds of tunable parameters (this project's `kShadeA2`/`kShadeA4`, the CCM, the AWB
gray-world/white-patch blend weight, the gamma curve shape) that a **camera tuning engineer** (a real,
specialized role — see `PRACTICE.md` section 4) adjusts per sensor/lens/module combination against a
lab full of calibration charts and captured scenes, often for weeks per camera module. This project's
gates are a miniature, automated version of exactly that validation process — a real tuning team's
"is this parameter set acceptable" workflow, at a scale a single learner can run in milliseconds.
