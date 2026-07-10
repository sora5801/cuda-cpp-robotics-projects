# 03.01 — FMCW radar cube processing: range-Doppler-angle FFTs + CA/OS-CFAR detection: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

### What a radar measures, and why it is hard

A radar sends out radio energy and listens for the echo. Everything downstream of that sentence is
about extracting THREE numbers from the echo — how far away the reflector is, how fast it is moving,
and in what direction — from a signal that, on the wire, is just a sequence of complex voltage
samples. The whole trick of FMCW ("frequency-modulated continuous wave") radar is choosing a
transmit waveform whose echo turns each of those three physical quantities into a different,
separable KIND of phase drift, so that three ordinary Fourier transforms can pull all three back out.

### The chirp, and the "beat frequency" idea

Instead of a single-frequency tone, an FMCW radar transmits a **chirp**: a signal whose instantaneous
frequency sweeps linearly from `fc` to `fc + B` over a duration `Tc` (`fc` = carrier frequency, `B` =
sweep bandwidth). At the receiver, the returning echo — delayed by the round-trip travel time
`tau = 2R/c` (`R` = range, `c` = speed of light) — is **mixed** (multiplied) with a copy of the
currently-transmitting chirp. Multiplying two chirps that are offset in time by `tau` produces a
signal at the DIFFERENCE of their instantaneous frequencies, which — because the chirp's frequency
increases at a constant rate `S = B/Tc` (the **slope**, Hz/s) — is a CONSTANT frequency:

```
f_beat = S * tau = S * (2R/c) = 2*R*S/c
```

This is the "beat frequency" and it is the whole reason FMCW radar is tractable: a hard problem (find
the delay of a received copy of a known waveform) has been turned into an easy one (find the frequency
of a single tone). Sample that beat signal `Ns` times over one chirp (rate `fs = Ns/Tc`) and its
frequency shows up directly as a peak in an `Ns`-point FFT — **range from a single tone's frequency.**

### Doppler: the SAME idea, one chirp-period later

A stationary target's beat frequency is identical from one chirp to the next. A MOVING target's is
not, quite — but far more usefully, its range changes by a tiny, nearly-invisible amount from chirp to
chirp (`v * Tc`, a fraction of a millimeter at automotive closing speeds and Tc ~ tens of
microseconds), which is not enough to shift which RANGE BIN it lands in, but IS enough to shift the
PHASE of that bin's complex value by a measurable amount — because phase is far more sensitive to
small distance changes than a beat-frequency measurement is (phase accumulates at the carrier
frequency `fc`, not the beat frequency). Over one chirp period `Tc`, a target moving at radial velocity
`v` changes range by `v*Tc`, which changes the round-trip phase by

```
delta_phi = 2*pi * (2*v*Tc) / lambda = 2*pi * f_d * Tc,      f_d = 2*v/lambda  (the Doppler frequency)
```

(`lambda = c/fc`, the carrier wavelength). Stack up `Nc` chirps, and the SAME range bin's complex
value rotates in phase by `delta_phi` every chirp — exactly a tone of frequency `f_d`, sampled once per
chirp. FFT across chirps and that tone's frequency reveals velocity — **the identical "read off a
tone's frequency with an FFT" trick as range, just applied to a signal one level of abstraction up**
(phase-across-samples instead of frequency-across-time).

### Angle: the SAME idea again, in space instead of time

Now put more than one receive antenna on the radar, spaced a fixed distance `d` apart in a line (a
**ULA**, uniform linear array). A target at azimuth angle `theta` (0 = broadside, straight ahead) is
an infinitesimal amount FARTHER from one antenna than its neighbor — extra path length
`d*sin(theta)` — which is again invisible as a range or Doppler shift but shows up as a constant PHASE
STEP from antenna to antenna:

```
delta_phi_antenna = 2*pi * d*sin(theta) / lambda
```

Sample `Na` antennas simultaneously (same chirp, same range bin) and that phase step is — once more —
exactly a tone, this time sampled once per ANTENNA instead of once per chirp or once per fast-time
sample. FFT across antennas and its frequency reveals angle.

**The unification this project exists to teach:** range, velocity, and angle are the SAME
phenomenon — a linear phase ramp turned into a spectral peak by an FFT — applied along three
different axes of the SAME data cube, at three wildly different physical time/space scales
(microseconds within a chirp, milliseconds across chirps, millimeters across antennas). Once you see
that unification, "radar signal processing" stops being three unrelated tricks and becomes one trick,
applied three times. `src/kernels.cu`'s `synthesize_cube_kernel` writes exactly this — one phasor
formula, three phase terms, added because phasors compose by addition of their phases (multiplication
of the complex exponentials).

### The engineering frame

A real radar's ADC delivers `Ns` complex samples every `Tc` (tens of microseconds), `Nc` chirps every
frame (milliseconds), at a 10-20 Hz frame rate — this project's "System context" quotes the exact
budget. The front end that produces this cube is analog RF engineering with its own hard constraints
(phase noise, IQ imbalance, ADC dynamic range, thermal noise floor) that `PRACTICE.md` §1-2 describes;
this project starts one layer up, from an idealized-but-honestly-labeled synthetic cube, and teaches
everything from "cube in hand" onward.

## The math

### Signal model (the exact formula this project synthesizes)

For one target at range `R`, radial velocity `v` (positive = **approaching**, i.e. range decreasing —
this project's fixed sign convention, applied consistently in `src/kernels.cuh`), azimuth `theta`, and
reflection amplitude `A`, the complex baseband sample at fast-time index `n` (0..Ns-1), chirp index `c`
(0..Nc-1), antenna index `a` (0..Na-1) is:

```
x[n,c,a] = A * exp( j * 2*pi * (f_beat*n/fs + f_d*c*Tc) ) * exp( j*pi*sin(theta)*a )   + noise[n,c,a]

  f_beat = 2*R*S/c          S = B/Tc         (range -> fast-time phase rate)
  f_d    = 2*v/lambda        lambda = c/fc    (velocity -> slow-time phase rate)
```

The antenna term uses `d = lambda/2` (half-wavelength spacing — see "Resolution & ambiguity" for why),
which collapses `2*pi*d*sin(theta)/lambda` to exactly `pi*sin(theta)`. This is the **decoupled**
teaching model: range and Doppler are treated as two INDEPENDENT phase accumulators, ignoring the
small "range-Doppler coupling" a moving target's beat frequency actually carries within a single
chirp (see "Numerical considerations"). The full cube is the sum of this expression over every
target, plus independent complex Gaussian noise per sample (thermal noise floor, `sigma` per I/Q
component).

### Resolution and ambiguity — the formulas this project's constants come from

An `N`-sample transform can only distinguish frequencies `1/(N*dt)` apart (`dt` = the sample spacing)
and cannot distinguish a frequency `f` from `f + 1/dt` (aliasing). Applying that once per axis:

| Axis | Sample spacing | Resolution | Unambiguous span |
|------|-----------------|------------|-------------------|
| Range (fast-time) | `1/fs` (ADC sample) | `dR = c/(2B)` | `R_max = Ns*dR` (this project's complex-baseband cube; see the numerical note below) |
| Velocity (slow-time) | `Tc` (chirp period) | `dv = lambda/(2*Nc*Tc)` | `v_max = +/- lambda/(4*Tc)` |
| Angle (antenna) | `d` (element spacing) | `~lambda/(Na*d)` rad (broadside) | `+/-90 deg` (unaliased) when `d = lambda/2` |

At this project's parameters (`fc=77 GHz, B=300 MHz, Tc=50 us, Ns=256, Nc=128, Na=8`):

```
lambda    = c/fc                = 3.893 mm
S         = B/Tc                = 6e12 Hz/s
fs        = Ns/Tc                = 5.12 MHz
dR        = c/(2B)               = 0.500 m         R_max = Ns*dR       = 127.9 m
dv        = lambda/(2*Nc*Tc)     = 0.304 m/s        v_max = lambda/(4*Tc) = 19.47 m/s (~70 km/h)
```

These are exactly the numbers `src/kernels.cuh` computes as `constexpr` and `main.cu` prints on the
`RESOLUTION:` line — one source of truth, no magic numbers duplicated anywhere else.

**Why `d = lambda/2`?** The antenna array is a SPATIAL sampling process exactly like the ADC is a
TEMPORAL one, and it obeys the same Nyquist logic: sampling a spatial phase ramp with spacing `d`
aliases (creates duplicate "grating lobe" directions) unless `d <= lambda/2`. Choosing exactly
`lambda/2` is the tightest spacing that avoids aliasing while using the fewest antennas — the
standard automotive/robotics ULA choice.

**Why the zero-padded angle FFT (`kNaFft = 64` from `Na = 8` real antennas) does not improve
resolution.** Zero-padding a DFT input increases how finely the OUTPUT is SAMPLED (more bins across
the same span), which sharpens a single peak's readout — it does not narrow the peak itself or add
any new information the original 8 antennas did not already carry (the true angular resolution, ~2/Na
radians at broadside, is set by the ARRAY'S PHYSICAL APERTURE, not by how many FFT bins you choose to
compute). This project's `find_angle_peaks_kernel` therefore reads a sharper, more accurate peak
LOCATION from a zero-padded FFT, but two targets closer together than the true 8-element beamwidth
would still not be resolved into two separate peaks.

## The algorithm

Nine stages (numbered to match `src/kernels.cu`'s file header and `src/main.cu`'s call order):

1. **Synthesize the cube** — sum every target's phasor (the formula above) plus per-sample noise.
   `O(Ns*Nc*Na*Ntargets)` work, embarrassingly parallel across samples.
2. **Hann-window the range axis**, **3. Range FFT** — `O(Ns log Ns)` per (chirp, antenna) pair,
   `Nc*Na` batches.
3. **Hann-window the Doppler axis**, **4. Doppler FFT** — `O(Nc log Nc)` per (range, antenna) pair,
   `Ns*Na` batches.
4. **fftshift the Doppler axis** — a pure re-index (see "Numerical considerations" for why this is a
   separate, explicit step rather than folded into the FFT).
5. **Noncoherent antenna integration** — `rd_power[n,c] = mean_a |cube[n,c,a]|^2`, `O(Ns*Nc*Na)`.
6. **CA-CFAR and OS-CFAR** — `O(Ns*Nc*Ntrain)` each, `Ntrain ~ 200` per cell (below).
7. **Cluster detections** (local-maximum suppression) — `O(#flagged cells)`, tiny and host-side.
8. **Gather + zero-pad + FFT the angle snapshot** per detection — `O(#detections * NaFft log NaFft)`.
9. **Find the angle peak** per detection — `O(#detections * NaFft)`.

Stages 1-6 and 8 dominate the FLOP count and are exactly the "batched, independent transforms" pattern
README quotes; stage 6 (CFAR) is the one STENCIL in the pipeline — every cell reads a wide, fixed
neighborhood rather than being independent of its neighbors.

### CFAR: why a fixed threshold cannot work, and what CA-CFAR does about it

A detector needs a threshold: "flag this cell if its power exceeds X." A FIXED X fails for two
reasons that CFAR (constant-**false-alarm-rate**) is built to solve: (a) the noise/clutter floor is
not perfectly flat — it varies with range (near/far clutter), with scene content, even with
temperature — so a threshold tuned for one region either misses weak targets in a quiet region or
floods a noisy region with false alarms; (b) you generally do not know the noise power in advance.
**CA-CFAR's answer:** estimate the LOCAL noise floor, cell by cell, from a ring of neighboring
"training" cells assumed target-free, and set the threshold PROPORTIONAL to that local estimate:

```
threshold(i,j) = alpha * mean( training cells around (i,j) )
```

`alpha` is chosen (theory: from the assumed noise statistics; here: **calibrated empirically**, see
"How we verify correctness") so that pure noise crosses the threshold with the desired small
probability `P_fa`, REGARDLESS of the local noise power — the "constant false-alarm rate" property.
A **guard band** of cells immediately around the cell under test (CUT) is EXCLUDED from the training
average, because the CUT's own energy leaks into its immediate neighbors (finite mainlobe width, even
windowed) — including guard cells in the average would make a detector partially measure its own
target as "background," raising its own threshold and desensitizing itself.

### Why CA-CFAR fails on closely-spaced targets — and what OS-CFAR does differently

CA-CFAR's ENTIRE robustness rests on the assumption that every training cell is genuinely
target-free clutter/noise. The moment a SECOND target's response falls inside the training window
(a common situation: real scenes have multiple targets, and "closely spaced" is common at automotive
ranges), one or more training cells carry real target energy — often orders of magnitude above the
noise floor. The ARITHMETIC MEAN is not robust to outliers: even ONE contaminated cell among 200 can
drag the mean up enough that a weaker, genuinely-present target elsewhere in the SAME window sits
below the inflated threshold and is **masked** — never flagged at all. This project's committed scene
manufactures exactly this: target 5 (60.0 m, +6.0 m/s, amplitude 1.0) and target 6 (61.5 m, +6.9 m/s,
amplitude 0.15) are 3 range bins and 3 Doppler bins apart — close enough that target 6's CFAR window
contains target 5's response, and target 5 is 6.7x stronger in amplitude (~45x in power).

**OS-CFAR's fix:** SORT the training cells and read a FIXED RANK (this project: the 75th percentile,
`kOsRankFrac = 0.75`, sometimes called the "3rd quartile" order statistic) instead of the mean:

```
threshold(i,j) = alpha_OS * sorted(training cells)[ round(0.75 * (Ntrain-1)) ]
```

A single strong interferer contributes AT MOST a handful of large values, which land at the TOP of
the sorted list — far above a rank chosen at the 75th percentile of ~200 cells — and simply do not
move that rank's value much. The statistic is robust to a MINORITY of contaminated cells by
construction. This project's own measured demo output shows the predicted result exactly: **CA-CFAR
detects 5 of 6 targets (misses target 6, 0 false alarms); OS-CFAR detects all 6 (0 false alarms)** —
see `demo/out/detections.csv` and the `CFAR_COMPARE:` line for the live numbers.

**The price OS-CFAR pays** (named honestly, not hidden): sorting `Ntrain` values per cell is more
expensive than a running mean (this project's kernel uses a simple `O(Ntrain^2)` insertion sort — see
"The GPU mapping" for why), and under PURELY homogeneous noise, a percentile-based statistic has
slightly higher variance than the sample mean, which can translate to a marginally higher realized
false-alarm rate for the same nominal `P_fa` — a real, textbook trade-off (Rohling 1983; README "Prior
art"), not specific to this implementation.

### CA/OS-CFAR geometry, concretely

```
                     kCfarHalf = kCfarGuard + kCfarTrain = 7
        <----------------------- 15 -------------------------->
        T T T T T T T G G G G G T T T T T T T      <- one row of the window
        T T T T T T T G G G G G T T T T T T T         G = guard (5x5 block, excluded)
        T T T T T T T G G G G G T T T T T T T         T = training (used for mean/rank)
        T T T T T T T G [X] G G T T T T T T T         X = cell under test (CUT)
        T T T T T T T G G G G G T T T T T T T
        ... (15 rows total) ...
```

`Ntrain = 15*15 - 5*5 = 225 - 25 = 200` — the training-cell count both CFAR kernels and the CPU oracle
share (`kCfarNTrain` in `src/kernels.cuh`).

## The GPU mapping

### The three cuFFT calls, and the one that needs a host-side loop

`src/kernels.cuh`'s cube layout is `[Ns][Nc][Na]` row-major (range slowest, antenna fastest). This
matters because `cufftPlanMany`'s "batch" parameter is a SINGLE 1-D loop (`offset(b) = b * idist`,
one multiply, one add) — it cannot express a batch that must enumerate through TWO axes with
different strides.

- **Range FFT** transforms `n` (the SLOWEST/outermost axis). Fixing a `(chirp, antenna)` pair and
  varying `n` walks stride `Nc*Na`; crucially, the `(chirp, antenna)` PAIR ITSELF enumerates offsets
  `0, 1, 2, ..., Nc*Na-1` (that literally IS the `n=0` slice, contiguous by construction). One
  `cufftPlanMany` call, `batch = Nc*Na`, `istride = Nc*Na`, `idist = 1` — the "easy" case, because the
  transformed axis is OUTERMOST.
- **Doppler FFT** transforms `c` (the MIDDLE axis). Fixing a range bin, the `Na` antennas at each
  chirp ARE contiguous and batchable in one call (`batch = Na`, `istride = Na`, `idist = 1`) — but
  extending that across all `Ns` range bins in a SINGLE call would need
  `offset(n, a) = n*(Nc*Na) + a` to be affine in one combined index, and it is not (as shown by
  direct substitution in `src/kernels.cu`'s `launch_doppler_fft` comment). This is a genuine,
  well-known limit of `cufftPlanMany`'s batching model, not a corner cut for convenience. The fix:
  build ONE plan and **execute it `Ns` times**, offsetting the pointer by one range bin's worth of
  data each time — the standard "create once, exec many" cuFFT idiom.
- **Angle FFT** transforms a zero-padded per-detection snapshot — plain CONTIGUOUS batching
  (`istride=1, idist=kNaFft`), the simplest of the three, placed last in the file so a reader sees the
  full spectrum: contiguous, strided-outer, strided-middle-looped.

**The alternative not taken:** transpose the cube once (e.g., into `[Nc][Ns][Na]`) so Doppler becomes
the outer axis too, trading one full-cube copy for a single clean Doppler FFT call. Real pipelines do
this when Doppler-FFT throughput matters more than the extra memory traffic; this project keeps the
ONE layout throughout deliberately, because the "loop a plan over the outer axis" idiom is itself
worth learning, and because the actual runtime cost here (256 small batched-FFT launches) is
negligible next to CFAR's cost (below).

### CFAR: a stencil with expensive per-thread local storage

One thread per range-Doppler cell (2-D grid, `16x16` blocks) gathers `Ntrain = 200` neighboring
values into a per-thread LOCAL array (`float cells[200]`, 800 bytes). This is large enough that it
does **not** fit in registers and spills to local memory (off-chip, cached, addressed per-thread) —
an honest, deliberate cost of a fixed-size, teaching-simple implementation. CA-CFAR then sums the 200
values (`O(N)`); OS-CFAR SORTS them with a plain insertion sort (`O(N^2)`, ~40,000 comparisons per
thread, ~1.3 billion across all 32,768 cells) — the simplest correct algorithm, not the fastest (see
"Where this sits in the real world" for the production alternative). No shared memory is used: unlike
a stencil where NEIGHBORING threads' windows overlap heavily and a tiled shared-memory cache would pay
off (07.x-style stencils), a 200-cell training window relative to a 32,768-cell map means most of the
neighborhood-caching benefit is already captured by the L2 cache across the whole run — a genuine
`float cells[200]` per-thread copy (rather than a shared-memory tile) keeps the kernel simple without
leaving much performance on the table for a one-shot demo.

### Noncoherent integration: why not a tree reduction

`Na = 8` is small enough that a straight per-thread serial loop (`#pragma unroll`) beats any
shared-memory or warp-shuffle reduction machinery — the setup cost of a tree reduction exceeds 8
sequential adds. The general lesson: reduction PATTERNS (tree, warp-shuffle, `cub::BlockReduce`) earn
their complexity only once the reduced dimension is large enough that serial work dominates; small,
fixed-size reductions (this one; MPPI's `kNX=4` state vector in 08.01) are better served by a plain
loop.

## Numerical considerations

- **FP32 throughout**, matching the repo standard: real ADCs are 12-16 bit integers, and FP32's ~24-bit
  mantissa is comfortably beyond that precision floor for every quantity here.
- **Precise `sinf`/`cosf`, never the fast intrinsics.** `synthesize_cube_kernel`'s phase argument spans
  many multiples of `2*pi` for realistic `n`/`c` values; the fast intrinsics' relative error grows with
  argument magnitude — the same reasoning 08.01/09.01 give for large or wrapped angles.
- **Window choice: Hann, and why it is NOT optional.** A finite DFT is mathematically a RECTANGULAR
  window's spectrum (a sinc), whose first sidelobe is only ~-13 dB down and decays slowly. This
  project's OWN first working version omitted the Doppler-axis window and immediately hit the
  consequence: a strong target's rectangular-window sidelobes, spread across nearly every Doppler bin
  at its OWN range row, blew past both CFAR detectors' thresholds by the hundreds (dozens of spurious
  "detections" per strong target instead of one). Adding a Hann taper on BOTH the range and Doppler
  axes (`w[i] = 0.5*(1-cos(2*pi*i/(N-1)))`) pushed sidelobes below -31 dB and fixed it — the honest
  price is a WIDER mainlobe (the `dR`/`dv` formulas above are a rectangular-window best case; a
  Hann-windowed peak is measurably broader, which is why this project's ground-truth tolerances are
  set to a FULL resolution cell, not a fraction of one).
- **A genuine noise-generator bug, found and fixed during this project's own development.** The
  per-sample noise generator seeds an `xorshift32` stream directly from
  `base_seed + K*sample_index` (a value that differs from its neighbor by a constant). Used with only
  1-2 mixing steps before the first draw, this produced measurably NON-WHITE noise: its FFT showed
  elevated, structured power concentrated at specific range bins, enough to trigger over 400 false
  alarms on a scene where the (corrected) pipeline produces zero. The fix — `hash32_mix` in
  `src/kernels.cu`, a well-characterized 32-bit avalanche hash applied to the seed BEFORE running
  `xorshift32` — breaks the linear relationship between neighboring samples' seeds and restores a flat
  noise floor. This is kept in the code and documented here deliberately: it is a real lesson about
  per-thread RNG seeding on a GPU (every thread's stream must be independently WELL-MIXED, not merely
  differently-seeded), not smoothed over as if the first attempt had been correct.
- **Range-Doppler coupling, ignored by design.** A target's TRUE beat frequency has a small
  velocity-dependent term this project's decoupled model omits (`f_IF = 2*R*S/c + 2*v*fc/c`,
  approximately, up to sign/chirp-direction convention); at this project's parameters, the omitted
  term can bias a fast target's apparent range by a fraction of a range bin (order 0.1-0.5 bin at the
  velocities in the committed scene) — small enough that the ground-truth tolerance (1 full bin)
  absorbs it, but real enough that production systems correct for it (see "Where this sits in the
  real world").
- **The fftshift is an explicit, separate GPU pass** (`launch_fftshift_doppler`), not folded into the
  Doppler FFT or left implicit: cuFFT's natural bin order (DC first, negative frequencies wrapped to
  the back half) would make the Doppler axis WRAP AROUND at the CFAR window's edges (a training window
  centered near the "seam" would silently average unrelated velocities together) — shifting first,
  once, keeps every downstream stage's "index kNc/2 = zero velocity" contract simple and correct.
- **Power values are relative, uncalibrated units** (see README "Limitations"), so `detections.csv`'s
  dB column (`10*log10(power)`) is a RELATIVE dB scale useful for comparing cells within one run, not
  a calibrated dBm/dBsm measurement — stated explicitly so a reader never mistakes a teaching artifact
  for a calibrated radar-equation output.

## How we verify correctness

Two independent checks (mirroring 08.01's "numerically right vs. behaviorally right" split):

1. **The §5 GPU-vs-CPU VERIFY gate.** `reference_cpu.cpp` computes the ENTIRE pipeline — synthesis,
   both windowed DFTs, integration, both CFAR detectors, angle estimation — with a hand-rolled
   `O(N^2)` DFT (precomputed twiddle tables, no recursive Cooley-Tukey) instead of cuFFT, and with the
   IDENTICAL per-sample noise formula (so the CPU and GPU cubes agree to near-ULP precision with
   **zero data transfer** between the two paths — the same noise-parity strategy 08.01 uses for its
   host-generated exploration noise, here re-derived per-sample). The two range-Doppler power maps are
   compared cell-by-cell (all 32,768 cells); measured worst relative deviation is 0.55%, well inside
   the 5% tolerance `main.cu` gates on — a bug in indexing, layout, windowing, or the FFT call would
   shift values by orders of magnitude, not fractions of a percent, so this gate has enormous margin
   against real bugs while still being honest about FP32/algorithm-order differences.
2. **Ground-truth gates.** Every injected target's estimated range/velocity/azimuth is compared
   against its TRUE value, with tolerances derived directly from the resolution formulas above (one
   full range bin `dR`, one full Doppler bin `dv`, and 3 degrees of azimuth — roughly 3x the
   zero-padded-FFT quantization step at the committed scene's largest angles, giving headroom for
   noise-driven jitter without weakening the check). A false-alarm-count bound (8, against a measured
   0 for both detectors) and the CA-vs-OS masking comparison (CA must miss the close pair's weak
   target; OS must not) round out the gate. `main.cu`'s "Output contract" keeps EXACT counts off the
   diffed stable lines deliberately — they are printed (on `[info]` lines) but not asserted
   byte-for-byte, because cuFFT's internal algorithm (hence FP32 rounding) differs across GPU
   architectures (sm_75 vs sm_86 vs sm_89), and a borderline cell could in principle flip by one
   architecture to the next even though the pipeline is fully deterministic on any ONE machine.

## Where this sits in the real world

- **Production automotive/robotics radar SoCs** (TI AWR/IWR-class, NXP S32R, Infineon) run essentially
  this exact range-Doppler-angle-CFAR chain in on-chip DSP or hardware accelerators, at 10-20 Hz,
  correcting for range-Doppler coupling (often via alternating up/down chirp slopes, resolving the
  ambiguity algebraically from the two slopes' beat-frequency difference) and running CFAR variants
  tuned per product (CA, OS, GO/SO "greatest/smallest-of" hybrids, and adaptive multi-pass schemes) —
  README "Exercises" invites implementing GO/SO as a direct extension of this project's CFAR kernels.
- **CFAR theory beyond CA/OS:** GO-CFAR (max of leading/lagging training-half means) reduces false
  alarms at clutter EDGES; SO-CFAR (min of the two halves) improves detection of closely-spaced targets
  at the cost of a higher false-alarm rate in uniform clutter — a genuinely different trade-off region
  than either CA or OS. Adaptive schemes (e.g. censoring detected outliers before averaging) chase even
  better robustness at higher computational cost.
- **The OS-CFAR sort, done properly:** production implementations needing only ONE order statistic use
  `nth_element`/quickselect (`O(N)` average) rather than a full sort (`O(N log N)` or, as here, an
  `O(N^2)` insertion sort) — this project's simple sort is a deliberate teaching choice (CLAUDE.md §1),
  not a claim that it is how a real system would do it.
- **MIMO virtual arrays — the 8-antenna array's generalization.** Modern "imaging radar" uses time- or
  frequency-division MIMO (multiple transmit antennas, each distinguishable at the receiver) to
  synthesize a much LARGER virtual array than the physical receive-antenna count — hundreds of virtual
  channels instead of this project's 8 — dramatically sharpening true angular resolution (not just
  peak-localization) and adding ELEVATION as a fourth axis ("4D radar": range, velocity, azimuth,
  elevation). The angle-FFT stage this project builds is the exact building block that generalizes;
  only the array geometry and channel count grow.
- **Beyond FFT-based angle estimation:** subspace methods (MUSIC, ESPRIT) can resolve targets closer
  together than the array's Rayleigh (FFT) resolution limit, at higher computational cost and with
  more restrictive assumptions (known number of sources, sufficient snapshots) — a natural `[R&D]`-
  flavored extension of this project's angle-estimation stage.
