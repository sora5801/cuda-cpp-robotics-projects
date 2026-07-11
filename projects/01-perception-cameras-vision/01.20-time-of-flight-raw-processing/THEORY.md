# 01.20 — Time-of-flight raw processing: phase unwrapping, flying-pixel removal: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

### Direct vs. indirect time-of-flight

The most literal way to measure distance with light is **direct ToF (dToF)**: fire a very short pulse,
start a stopwatch, stop it the instant a reflection returns, and compute `distance = c * t_flight / 2`
(the round trip covers the distance twice). At `c = 3x10^8` m/s, one meter of round trip takes about
6.7 nanoseconds — measuring that directly needs picosecond-class timing electronics (a SPAD — single-
photon avalanche diode — plus a time-to-digital converter). This project is about the OTHER family:
**continuous-wave indirect ToF (iToF)**, which never times an individual pulse at all. Instead, the
illuminator's brightness is modulated continuously (typically a sine or square wave) at a fixed
**modulation frequency `f`** (tens of MHz — an IR LED or VCSEL array driven electronically, not
mechanically), and every pixel **cross-correlates** the returning light against a reference copy of
that same modulation waveform. The round-trip delay shows up not as a stopwatch reading but as a
**phase shift** between the emitted and received waveforms — the correlation, not a clock, does the
timing. This trades dToF's raw temporal resolution for a MUCH simpler, cheaper, more compact sensor:
iToF is why phone-class and consumer depth cameras (Kinect v2, Azure Kinect, PMD/Melexis automotive
modules) exist at all; dToF (LiDAR-class SPAD arrays) is reserved for where its extra range and
outdoor-sunlight robustness justify the added electronics — see "Where this sits in the real world".

### The correlation/mixer integral (deriving the 4-tap formula from first principles)

Let the illuminator's optical power be `P(t) = P0 * (1 + cos(2*pi*f*t))` (a sinusoidally modulated
source riding on a DC bias so power never goes negative — a real physical constraint LEDs/VCSELs
share with any light source). A surface at round-trip delay `tau = 2*Z/c` (`Z` = depth, meters)
reflects an attenuated, delayed copy back to the pixel: `P_rx(t) = R * P(t - tau)`, where `R` folds in
surface albedo, illumination falloff, and lens transmission. Each pixel's in-silicon **demodulator**
does not record `P_rx(t)` directly (that would need a picosecond-fast photodetector output digitized
at gigahertz rates — dToF's approach); instead it **multiplies** `P_rx(t)` by an internally generated
reference square/sine wave `ref_k(t) = cos(2*pi*f*t - k*pi/2)` (a copy of the modulation, phase-shifted
by `k*pi/2` for tap `k in {0,1,2,3}`) and **integrates** the product over many modulation cycles — the
photodiode's charge storage wells physically ARE that integrator. This multiply-then-integrate step is
exactly a **lock-in amplifier** (or, in signal-processing language, a correlator/mixer): of the product
`cos(2*pi*f*t - phi) * cos(2*pi*f*t - k*pi/2)` (writing `phi = 2*pi*f*tau` for the round-trip phase),
trigonometric product-to-sum turns this into a DC term `cos(phi - k*pi/2)/2` plus a `2f`-frequency term
that averages to (almost) zero over the integration window. What survives integration is therefore

```
C_k  ~  A + B * cos(phi - k*pi/2)          (the sign convention this project uses: see "The math")
```

— a DC offset `A` (ambient light + the illuminator's own bias, both of which correlate to a nonzero
constant against ANY reference phase) plus a term proportional to `cos` of the ROUND-TRIP PHASE minus
the reference phase. This is the whole trick: a purely electrical multiply-and-average recovers a
PHASE without ever timing anything directly. Four taps (`k=0,1,2,3`, spaced 90 degrees apart) give four
linear equations in the three unknowns `A`, `B`, `phi` — solvable in closed form (below) with one
equation to spare, which this project uses to make the ambient term cancel exactly rather than merely
being solved for (the "offset invariance" property `offset_invariance` measures).

### Ambiguity: why phase alone is not enough

`phi = 2*pi*f*tau = 2*pi*f*(2Z/c) = 4*pi*f*Z/c` grows WITHOUT bound as `Z` grows, but the sensor can
only ever measure `phi` modulo `2*pi` (a phase is an angle; `cos`/`sin` cannot distinguish `phi` from
`phi + 2*pi`). Solving for the largest unambiguous depth (`phi = 2*pi` exactly):

```
Z_ambiguous = c / (2*f)              ("the ambiguity range", this project's kAmbig1M / kAmbig2M)
```

At `f = 60 MHz` (this project's fine channel), `Z_ambiguous ~= 2.50 m`: any surface at 3.5 m reports
back the SAME raw phase as a surface at `3.5 - 2.50 = 1.0 m`. This project's committed scene puts its
background wall at ~5.0-5.6 m specifically so this genuinely happens (the `aliasing_demo` gate proves
it) — a single CW frequency simply cannot tell "1 m" from "3.5 m" from "6 m" apart; it only ever
reports where you are WITHIN one 2.50 m cycle. "The math" and "The algorithm" below derive this
project's fix: a SECOND, lower frequency whose OWN ambiguity range comfortably covers the whole scene,
used to resolve which cycle the fine frequency is in.

### The physical mechanism behind flying pixels

A camera pixel is not an infinitesimal ray — it integrates light over a small solid angle subtended by
its physical photosensitive area. At an ordinary (non-edge) surface point this is a non-issue: every
photon arriving within that solid angle comes from (very nearly) the SAME depth and the SAME material,
so the pixel's four taps integrate one coherent signal. At a **silhouette edge** — where a foreground
object's boundary crosses a pixel's footprint — the SAME pixel simultaneously receives light reflected
from the FOREGROUND surface (at depth `Z_fg`) and the BACKGROUND surface behind it (at depth `Z_bg`,
often meters farther away, hence a very different phase). Because the correlator's integral is LINEAR
in incident optical power (this is not a numerical approximation — it is what an integral of a sum
literally is), the pixel's measured tap value is the AREA-WEIGHTED SUM of what each surface would have
produced alone:

```
C_k(mixed)  =  w * C_k(phi_fg, B_fg)  +  (1-w) * C_k(phi_bg, B_bg)         w in (0,1): the foreground's area fraction
```

Substituting the tap formula, this is a sum of two `A + B*cos(phi - k*pi/2)` terms — algebraically, the
DECODED phase of the SUM is emphatically **not** a weighted average of `phi_fg` and `phi_bg`, and the
decoded DEPTH is not a weighted average of `Z_fg` and `Z_bg` either. "The math" below makes this
precise via the phasor (complex-number) representation: adding two sinusoids of different phase is
adding two 2-D vectors, and the angle of a vector sum is a genuinely nonlinear function of the two
input angles and magnitudes. The result is a pixel that reports a depth belonging to NEITHER real
surface — it appears to "fly" in space, often hovering between the foreground and background, which is
exactly the artifact `flying_pixel_detect_kernel` is built to catch. This is a real, physical
phenomenon on every iToF camera ever built (Kinect v2's noisy silhouette fringes are exactly this);
`scripts/make_synthetic.py`'s supersampled forward model reproduces the mechanism directly rather than
faking its symptom.

### Engineering constraints a real iToF camera imposes

- **Modulation contrast and eye safety.** The illuminator is a real IR emitter (850 or 940 nm is
  common — outside the visible band, near a solar-irradiance dip at 940 nm that helps outdoor
  performance) whose peak/average optical power is constrained by eye-safety limits (IEC 62471 /
  60825 — PRACTICE.md §4), which in turn caps the achievable modulation amplitude `B` and hence the
  achievable SNR/depth precision at any given range — "The math" derives the direct link between `B`
  and depth noise.
- **Multipath and inter-reflections.** iToF measures whatever phase the SUM of all light paths reaching
  a pixel produces; a concave corner, a shiny floor, or another nearby iToF camera's OWN modulated
  light can add a second, spurious path — the deep, NOT-fixed-by-this-project failure mode named
  honestly in "Where this sits in the real world" below (this project implements the two-surface,
  SPATIAL-mixing flying-pixel case; general multipath from MULTIPLE reflection bounces off a single
  ray path is a documented-only extension).
- **Motion.** Because a depth estimate needs 4 (or, with unwrapping, 8) sequentially captured taps,
  a moving scene or camera smears the correlation exactly like a long camera exposure smears a photo —
  fast motion corrupts the phase estimate in a way this project's static-scene demo does not exercise
  (README "Limitations").
- **Thermal drift.** The reference waveform's phase (generated on-chip) and the illuminator's own drive
  electronics both drift with temperature, shifting the effective ambiguity-range origin — real
  cameras factory- or field-calibrate this ("wiggling error" calibration, PRACTICE.md §1) exactly the
  way this project's fixed, assumed-calibrated constants in `kernels.cuh` sidestep.

## The math

**Notation.** `Z` depth (m, along the camera's optical axis, `dz=1` pixel-center ray convention shared
with 01.17/01.19). `f` modulation frequency (Hz). `c` speed of light (`kSpeedOfLightMps`, exact SI
value). `phi` round-trip phase (radians, `[0, 2*pi)` by this project's convention — the SAME
"CLAUDE.md defaults to `(-pi,pi]`, this project deviates to `[0,2*pi)` because it must map
monotonically onto a physical delay" reasoning 01.19 states for its own phase convention). `C_k`
(`k=0..3`) the four raw correlation taps (intensity counts, `[0,255]` on this project's 8-bit sensor
model). `B >= 0` the modulation amplitude (counts) — this project's confidence signal.

### Tap forward model and its inversion

This project's tap convention (fixed once in `kernels.cuh`, shared bit-for-bit by
`kernels.cu`/`reference_cpu.cpp`/`scripts/make_synthetic.py`):

```
C_k(phi) = A + B * cos(phi + k*pi/2)          k = 0, 1, 2, 3
```

(a `+k*pi/2` sign, chosen — see kernels.cuh — specifically so the decode formula below comes out with
NO extra minus signs; a real sensor's actual sign convention is a hardware/firmware detail this
project is free to fix arbitrarily, as long as the forward model and the decoder agree, exactly the
freedom 01.19 notes for its own phase-wrap convention). Expanding all four taps:

```
C_0 = A + B*cos(phi)          C_1 = A - B*sin(phi)
C_2 = A - B*cos(phi)          C_3 = A + B*sin(phi)
```

so `C_3 - C_1 = 2*B*sin(phi)` and `C_0 - C_2 = 2*B*cos(phi)` — **the ambient/DC term `A` cancels
EXACTLY** in both differences, regardless of what `A` actually is (adding any constant to all four taps
leaves both differences, hence `phi` and `B`, bit-for-bit unchanged up to float rounding — the exact
algebra the `offset_invariance` gate measures at `<=1e-4` rad). Recovering phase and amplitude:

```
phi = atan2(C_3 - C_1, C_0 - C_2)                 wrapped to [0, 2*pi) by this project's convention
B   = 0.5 * sqrt((C_3-C_1)^2 + (C_0-C_2)^2)        the modulation amplitude / confidence signal
```

exactly 01.19's `phase_decode_kernel` pattern (`atan2` + vector length from a pair of differenced
samples), applied to a temporal correlator instead of a spatial fringe pattern — see README "Prior
art" for the explicit kinship.

### Depth from phase, and the ambiguity range

The round-trip phase accumulated over distance `2*Z` at angular rate `2*pi*f` is `phi = 2*pi*f*(2Z/c) =
4*pi*f*Z/c`. Inverting for depth and for the phase-wrap period in depth units:

```
Z(phi) = (c * phi) / (4*pi*f)              D = c / (2*f)     ("ambiguity range": Z(2*pi) = D)
```

`single_freq_depth_kernel` computes exactly `Z(phi) = (phi/2*pi) * D` — correct ONLY for `Z < D`; for
any `Z >= D` this returns `Z mod D`, not `Z` — the aliasing the `aliasing_demo` gate proves happens on
this project's far wall (`Z ~ 5-5.6 m >> kAmbig1M ~= 2.50 m`).

### Dual-frequency unwrapping as an integer consistency search

With two frequencies `f1` (fine, `kAmbig1M ~= 2.50 m`) and `f2` (coarse, `kAmbig2M ~= 7.49 m`,
`f1/f2 == 3` exactly — a deliberate scene-design choice, kernels.cuh "Scene depth budget"), each
frequency's raw phase gives a FAMILY of candidate depths, one per integer "wrap count" it could be
hiding:

```
z1(n1) = (phi1/2*pi + n1) * kAmbig1M       n1 in {0, 1, ..., kMaxWraps1-1}
z2(n2) = (phi2/2*pi + n2) * kAmbig2M       n2 in {0, 1, ..., kMaxWraps2-1}
```

Because `kAmbig2M > kMaxSceneDepthM` by construction, `n2` is ALWAYS 0 on this scene (`kMaxWraps2 =
1`) — the coarse channel alone already reports an unambiguous, if noisy, depth over the whole scene.
The unwrap search (`dual_freq_unwrap_kernel`/`_cpu`) tries every `(n1, n2)` pair and keeps the one
whose two candidate depths AGREE most closely:

```
(n1*, n2*) = argmin_{n1,n2} | z1(n1) - z2(n2) |             final depth = z1(n1*)
```

This is a **CRT-style** (Chinese-Remainder-Theorem-flavoured) consistency search: rather than solving a
modular congruence symbolically, it exploits the SMALL, exactly-known number of candidates
(`kMaxWraps1 * kMaxWraps2 <= 3` here) to brute-force the one hypothesis under which both frequencies
describe the SAME physical point. The winning `n1*` is reported as the wrap count; the fine channel's
depth AT that wrap (`z1(n1*)`), not an average of `z1` and `z2`, is reported as the answer, because
"Numerical considerations" below shows the fine channel is intrinsically more precise — the coarse
channel's only job is choosing which cycle, exactly 01.19's "Gray code resolves the period, phase
refines the position" pattern with (low-frequency CW, high-frequency CW) standing in for (Gray code,
phase-shift).

**Failure probability of the wrap decision (derived).** The search picks the WRONG `n1` exactly when
noise pushes the true-`n1` candidate's `|z1-z2|` above a WRONG candidate's — approximately, whenever
the combined depth-noise pushes `z1` and `z2` more than `kAmbig1M/2` apart (half the fine ambiguity
range: past that point, the search's tie-break favors the neighboring wrap). Modeling the combined
noise `z1 - z2` as approximately Gaussian with standard deviation `sigma_combined = sqrt(sigma_z1^2 +
sigma_z2^2)` (independent-noise approximation), the failure probability is

```
P(wrong wrap)  ~  2 * Phi(-kAmbig1M / (2*sigma_combined))      Phi = standard normal CDF
```

— i.e. it falls off VERY fast (the argument of `Phi` scales with `1/sigma`, and `Phi` itself decays
faster than exponentially) as SNR improves, but is never exactly zero: a long enough tail of unlucky
noise draws always exists. This is exactly why this project's committed sample shows a SMALL but
nonzero wrap-failure rate (`unwrap_recovery`'s measured ~2% wrap-count error) rather than either 0% or
a large fraction — the noise level was chosen (`data/README.md`) to make this failure mode measurable
and honest, not to hide it.

### Flying pixels as phasor addition

Represent tap `k`'s "AC part" as a 2-D vector (a **phasor**) `V = B * (cos(phi), sin(phi))` — exactly
the `(C_0-C_2, C_3-C_1)/2` pair the decode formula already computes. Because the tap-mixing forward
model (kernels.cuh file header) sums CONTRIBUTIONS linearly, a mixed pixel's phasor is the ordinary
vector sum of the two surfaces' phasors, weighted by area:

```
V_mixed = w * V_fg + (1-w) * V_bg                     V_fg = B_fg*(cos phi_fg, sin phi_fg), etc.
```

**Worked numeric example** (this project's box vs. wall, roughly): `phi_fg ~= 3.77` rad (box at 1.5 m,
freq1), `phi_bg ~= 5.03` rad (wall's wrapped phase at this silhouette), `B_fg ~= 45`, `B_bg ~= 58`,
`w = 0.5` (a pixel straddling the edge evenly). `V_fg ~= 45*(cos 3.77, sin 3.77) ~= (-35.5, -27.7)`;
`V_bg ~= 58*(cos 5.03, sin 5.03) ~= (23.4, -52.5)`. The mixed phasor `V_mixed = 0.5*V_fg + 0.5*V_bg ~=
(-6.1, -40.1)`, whose angle `atan2(-40.1, -6.1) ~= -1.72` rad (`~= 4.56` rad after wrapping to
`[0,2pi)`) is neither `phi_fg` (`3.77`) nor `phi_bg` (`5.03`) nor their average (`4.40`) — a THIRD,
distinct phase, decoding to a THIRD depth that belongs to neither surface: the flying pixel. Its
magnitude `|V_mixed| ~= 40.6` is also noticeably BELOW the simple average of `|V_fg|` and `|V_bg|`
(`51.5`) — the **destructive interference** signature `flying_pixel_detect_kernel`'s amplitude-ratio
test exploits: by the triangle inequality `|V_mixed| <= w|V_fg| + (1-w)|V_bg|`, with equality only when
`phi_fg == phi_bg`; whenever the two phases genuinely differ (as engineered by this project's scene
depth budget), the mixed amplitude is strictly, often substantially, weaker.

## The algorithm

Per-pixel pipeline, `n = kNPix = 19,200` camera pixels:

| Stage | Serial cost (CPU, per pixel) | Parallel cost (GPU) |
|-------|-------------------------------|----------------------|
| 1. Extract phase/amplitude | `O(1)` (4 reads, `atan2`, `sqrt`); called twice (per frequency) | `O(1)` |
| 2. Single-freq depth | `O(1)` (one multiply) | `O(1)` |
| 3. Dual-freq unwrap | `O(kMaxWraps1 * kMaxWraps2)` <= 3 candidate evaluations | `O(1)` (small constant) |
| 4. Confidence mask | `O(1)` (one compare) | `O(1)` |
| 5. Flying-pixel detect | `O(1)` (a FIXED 8-neighbor gather) | `O(1)` (but reads 8x the data of a pure map) |
| 6. Back-projection | `O(1)` (two multiplies) | `O(1)` |

Every stage is `O(n)` SERIAL total work, `O(1)` PARALLEL depth — the entire pipeline is, asymptotically,
the same "one thread per pixel, trivial per-thread work" story 01.19 tells for its five stages; the
interesting content here is entirely in "The GPU mapping" below (constants, memory patterns, and one
genuinely new access pattern), not in asymptotics.

The reconstruction GATES (plane fit, sphere fit, step height — `main.cu`) are, like 01.19's identical
trio, a small (3x3 or 4x4) one-off least-squares solve over a few thousand ALREADY-BACK-PROJECTED
points, done ONCE on the host — "The GPU mapping" explains why this deliberately stays off the GPU.

## The GPU mapping

**Thread-to-data mapping (Stages 1-4, 6):** thread `pix = blockIdx.x*blockDim.x + threadIdx.x` owns
camera pixel `pix` (row-major, `pix = row*kCamW + col`); grid `= ceil(n/256)`, the repo-default block
size (a warp multiple with good occupancy on `sm_75..sm_89` — the same geometry 08.01/33.01/01.19 use).
These five kernels are pure MAPS: `output[pix]` depends only on `input[*, pix]` — no thread reads or
writes another thread's pixel, so (as in 01.19) there is no shared memory and no atomics anywhere in
this project either.

**Stage 5 is this project's ONE genuine STENCIL — the new idea beyond 01.19.** `flying_pixel_detect_
kernel` maps identically (one thread per pixel), but its OUTPUT for pixel `pix` depends on up to 8
NEIGHBORING pixels' `depth`/`amplitude`/`confidence_valid` values (kernels.cu's kernel header spells
out the 3x3 gather). This is still race-free — every thread only ever WRITES its own output element —
so no atomics are needed even here (contrast with 01.18's depth completion, which genuinely needs
atomics because MULTIPLE threads can write the SAME destination pixel; this project's stencil, like
every kernel here, never has two threads target one output). What makes it a "stencil" rather than a
"map" is purely the READ side: each thread reads up to 9 pixels' worth of upstream data instead of 1,
and — unlike Stages 1-4/6 — a border pixel's neighbor set is genuinely SMALLER (the `r<0 || r>=h ||
c<0 || c>=w` guard drops out-of-bounds neighbors rather than wrapping or padding), a boundary-handling
decision every real stencil kernel must make explicitly.

**Why no shared-memory tiling for the stencil.** At `n = 19,200` pixels (a `160x120` image), each
interior pixel's value is read by at most 8 different neighboring threads — a modest redundancy factor
compared to, say, a large-radius convolution. A classic optimization here would TILE the image into
blocks, cooperatively load each tile PLUS a 1-pixel halo into `__shared__` memory once, and have every
thread in the block read its neighbors from shared memory instead of global memory — cutting redundant
global traffic roughly in half for this radius. This project deliberately does NOT do that: the
kernel's total memory traffic (about `9 * 3` floats/bytes read per pixel, `19,200` pixels) is well
under a megabyte, comfortably inside L2 cache residency on any `sm_75+` GPU, and the added code
complexity (halo boundary handling, shared-memory declaration and synchronization) would buy a real but
small win at this problem size — the SAME "recognize when a computation has no exploitable
optimization headroom, don't reflexively add complexity" judgment 01.19 applies to keeping its
reconstruction-gate solves off the GPU entirely (README Exercise: implement the shared-memory tiled
version and MEASURE whether it actually wins at this size, then again at 10x the resolution).

**Why the reconstruction gates stay on the host.** Identical reasoning to 01.19: fitting a plane or
sphere to a few thousand already-computed points is a single small linear solve done ONCE — there is no
"thousands of independent small problems" here to parallelize, and launching a kernel to perform one
Gaussian-elimination step would pay real launch overhead for microseconds of actual work.

**Occupancy and bandwidth.** At `n = 19,200` pixels this pipeline moves well under a megabyte per
stage — far below saturating any `sm_75+` memory bus, and (as in 01.19) kernel LAUNCH overhead across
seven-plus launches per run is a comparable-magnitude cost to the actual memory traffic. The demo's own
measured GPU-vs-CPU speed-up is therefore modest by design (`[time]` lines) — an honest, expected
result at this toy problem size (a real 1-2 MP iToF sensor frame is 50-100x more pixels), not a red
flag.

## Numerical considerations

- **Precision.** FP32 throughout the GPU/CPU pipeline (repo default); the reconstruction gates' small
  linear solves use FP64 on the host (01.19's identical choice: cheap insurance at this tiny problem
  size).
- **`atan2` conditioning at low modulation.** Identical argument to 01.19's phase-decode: `phi =
  atan2(y,x)` with `r = sqrt(x^2+y^2) = 2B`; for a fixed noise magnitude `sigma_n` perturbing `(x,y)`,
  the induced angular error scales like `sigma_n / (2B)` — UNBOUNDED as `B -> 0`. `confidence_mask_
  kernel`/`_cpu` mask pixels below `kDefaultAmplitudeFloor` for exactly this reason.
- **Depth-noise scaling (derived — the `noise_scaling` `[info]` diagnostic).** Combining the `atan2`
  conditioning above with `Z(phi) = (c*phi)/(4*pi*f)`: `sigma_Z = |dZ/dphi| * sigma_phi = (c/(4*pi*f))
  * sigma_phi`, and `sigma_phi ~ sigma_n/(2B)` (`sigma_n` a per-tap noise scale), so

  ```
  sigma_Z  ~  c/(4*pi*f) * sigma_n / (2*B)         (SHRINKS as amplitude B grows; GROWS at lower f)
  ```

  — the reason a real iToF camera's depth noise visibly increases on dim/distant/low-albedo surfaces
  (lower `B`) and why the FINE (higher-`f`) channel gives proportionally tighter raw precision than the
  coarse channel per unit of phase noise, even though it wraps sooner. The `noise_scaling` diagnostic
  buckets pixels by measured amplitude and reports the empirical depth-error standard deviation per
  bucket — a live check that this derived, inverse relationship actually holds on the rendered sample.
- **Wrap-decision brittleness near the boundary.** "The math" derives the wrap-failure probability;
  the practical corollary is that pixels whose TRUE depth sits very near a `kAmbig1M` boundary (where
  `z1(n1)` and `z1(n1+1)` are both plausible) are the most fragile — a small noise draw flips the
  decision. This is a genuine, physical brittleness of dual-frequency unwrapping, not a bug: production
  systems mitigate it with MORE than two frequencies, or wider frequency SEPARATION, trading acquisition
  time/complexity for robustness margin (README "Prior art").
- **Float precision in phasor sums.** `scripts/make_synthetic.py`'s forward model and `kernels.cu`'s
  Stage-1 decode both work with `cos`/`sin` values of order 1 and amplitudes of order 10-100 counts —
  comfortably within FP32's ~7-decimal-digit precision with no cancellation catastrophe; the phasor
  ADDITION itself (Stage 5's physical mechanism) is exact arithmetic on already-rendered noisy tap
  bytes, not a further source of numerical error.
- **Determinism.** No `curand`, no platform RNG anywhere in the C++ pipeline — every tap value the
  demo reads is a byte already baked into the committed PGMs by `make_synthetic.py`'s xorshift32
  generator (CLAUDE.md §12); the only runtime randomness-adjacent operation (the `offset_invariance`
  perturbation test) adds a FIXED constant, not a random draw, so every run of the demo is bit-for-bit
  reproducible on the same GPU.

## How we verify correctness

Two independent tiers, per the repo's twin-verification ruling (`reference_cpu.cpp`'s file header,
restated from 01.19's precedent):

1. **GPU-vs-CPU agreement** (the `VERIFY:` lines) — every stage's independently-written CPU function
   is run on the SAME inputs as the GPU kernel, as part of a FULLY SEPARATE end-to-end cascade
   (main.cu never feeds a CPU stage's output into a GPU stage or vice versa), and compared: tight
   floating tolerances for the `atan2f`/`sqrtf`-based Stage 1 (`1e-3` rad / `5e-2` counts, ULP drift);
   near-exact integer/flag agreement (a small, documented mismatch allowance) for the wrap-count,
   confidence, and flying-pixel decisions, which can occasionally tie-break differently after
   accumulating upstream ULP drift; tight tolerances for the purely-arithmetic depth and `xyz` stages.
2. **Ground-truth gates** (the `GATE:` lines) — comparisons against `scripts/make_synthetic.py`'s EXACT
   synthetic ground truth (true depth, surface identity, and — uniquely to this project — an
   INDEPENDENT flying-pixel label derived from sub-ray supersampling the C++ pipeline never sees), a
   THIRD, independent codebase (Python). Twin agreement alone cannot catch a bug shared by both C++
   implementations (a swapped tap index, a sign error in the phasor math); only a comparison against
   truth computed a different way can. `data/README.md` documents every gate's measured value and the
   margin behind its floor/bound.

**Edge cases specifically exercised:** the designed single-frequency aliasing on the far wall
(`aliasing_demo`); genuine dual-frequency wrap-decision FAILURES at the noise level chosen
(`unwrap_recovery`'s <100% wrap-correctness, honestly measured, not hidden); a deliberately low-
reflectivity cohort whose amplitude sits below the confidence floor (`dark_cohort`); real, physically-
rendered flying pixels at every silhouette edge in the scene, scored against ground truth the detector
never sees (`flying_pixel`).

## Where this sits in the real world

- **Consumer/prosumer iToF: Microsoft Kinect v2, Azure Kinect (iToF mode), PMD Technologies /
  Infineon REAL3, Melexis MLX75027-class automotive/industrial modules.** All use the SAME 4-tap (or
  higher-tap-count) CW correlation principle this project implements, typically with MULTIPLE
  frequencies (often 3+, not 2) for more robust unwrapping, on-chip fixed-pattern-noise (FPN) and
  "wiggling error" calibration (PRACTICE.md §1), and dedicated flying-pixel / edge-confidence filters
  in their vendor SDKs — the exact computation this project's Stage 5 teaches, at production
  robustness.
- **Direct ToF (dToF) SPAD arrays** (automotive/robotics LiDAR: Velodyne/Ouster/Hesai-class mechanical
  or solid-state units; phone-class dToF modules) measure round-trip TIME directly via
  picosecond-class timing electronics rather than a correlated phase — no ambiguity-range problem at
  all (a pulse's time-of-flight is unambiguous up to the pulse repetition interval, typically hundreds
  of meters), at the cost of far more complex, power-hungry timing silicon. Named here as the
  competing modality this project's catalog sibling 03.01 (FMCW radar) and the LiDAR-focused domain 02
  cover; not implemented in this project (scoped to CW iToF per the catalog bullet).
- **FMCW (frequency-modulated continuous-wave) LiDAR and radar** are a THIRD family: instead of a
  fixed modulation frequency, the illumination frequency is SWEPT continuously (a "chirp"), and the
  beat frequency between the transmitted chirp and its delayed echo encodes range directly, with
  velocity available for free via Doppler shift — structurally different math from this project's
  fixed-frequency correlation, named for contrast in README "Prior art" (project 03.01 in this repo).
- **Multipath / global illumination correction** is an active, NOT-solved-here research area: this
  project's flying-pixel detector catches the SPATIAL two-surface-per-pixel mixing case; a SINGLE ray
  path receiving light via MULTIPLE bounces (e.g. a concave corner, or a shiny floor bouncing the
  illuminator's own light back at a steep angle) corrupts the phase of even a geometrically "clean"
  pixel, and correcting it in general requires either multiple additional frequencies/phase
  measurements or full inverse-rendering-style global light-transport estimation — named honestly as
  a real, deeper failure mode this project does not attempt to fix (documented-only, per the repo's
  `[R&D]` scoping convention even though this catalog bullet itself is not tagged `[R&D]`).
