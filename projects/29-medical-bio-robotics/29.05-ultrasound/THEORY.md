# 29.05 — Ultrasound: GPU beamforming, elastography, image-based servoing: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2).
>
> **Educational/synthetic only** — see README's title block. This file teaches acoustics and
> signal processing; it makes no medical claim of any kind.

## The problem — physics & engineering first

**Sound, not light.** An ultrasound probe is a phased array of piezoelectric elements: apply a
voltage pulse and an element mechanically rings, launching a pressure wave into tissue; a returning
pressure wave rings the element mechanically, which the same piezo effect converts back into a
voltage. Everything downstream — beamforming, imaging, elastography — is signal processing on
those voltages. The wave itself obeys the acoustic wave equation in a lossy, inhomogeneous medium;
this project uses the field's universal simplifying convention, **c = 1540 m/s**, the accepted
"speed of sound in soft tissue" average (real tissue ranges roughly 1450–1600 m/s depending on fat
vs. muscle vs. fibrous content — a real scanner's biggest single geometric error source is this
single assumed constant being wrong for the tissue actually in the beam path).

**Pulse-echo, the whole imaging principle in one idea.** A short pulse is transmitted; wherever the
tissue's *acoustic impedance* (density × sound speed) changes — a vessel wall, a fat/muscle
boundary, a manufactured wire target — a fraction of the pulse's energy reflects back. The time
between transmission and the echo's arrival, multiplied by c and divided by two (there and back),
is the reflector's range. This project's phantom scatterers (`data/sample/phantom.csv`) are exactly
that: idealized point-like impedance changes, each with its own `amp_rel` "how much energy bounces
back" coefficient.

**Why speckle is interference, not noise.** This is the single most important, most
counter-intuitive physical idea this project teaches. Real tissue is riddled with structure far
finer than the wavelength (λ ≈ 0.31 mm at 5 MHz) — collagen fibers, cell clusters — acting as many
sub-resolution point scatterers within every resolution cell. The received signal at any instant
is the **coherent sum** of all their individual echoes, each with its own phase (set by its exact,
sub-wavelength position). When many random-phase contributions of comparable amplitude add, the
result is NOT their average — it is a random-walk sum whose magnitude follows a **Rayleigh
distribution**, with bright spots where contributions happen to add constructively and dark nulls
where they cancel. That texture is speckle. It is deterministic given the exact scatterer
positions (this project's phantom fixes them and `channel_data` reproduces the same speckle every
run) and it is *not* removable by "more signal" the way thermal noise is — averaging independent
looks (different angles, different frequencies) is the only real remedy, and production scanners
spend real engineering effort on exactly that (README "Limitations"). `src/main.cu`'s
`simulate_channel_data()` builds this project's speckle from first principles: 18,000
identical-amplitude, randomly-positioned background scatterers, summed coherently — the textbook
"many random phasors" setup, reproduced literally rather than approximated with an added noise
term.

**The engineering frame.** A transducer element is a resonant mechanical/electrical system with a
center frequency (here fc = 5 MHz) and a finite bandwidth that sets how short a pulse it can ring
out cleanly (this project models that as a Gaussian-windowed carrier — "The math" below). Sampling
the returning voltage fast enough to resolve that pulse (fs = 40 MHz, 8× fc) is a real front-end
engineering constraint (analog front-end bandwidth, ADC clock, per-channel data rate × 64
channels). And — the reason this project exists in a *robotics* repository — every one of these
constraints eventually meets a **frame-rate budget**: an image-guided robot needs frames fast
enough to close a control loop on (System context in README.md; "Where this sits in the real
world" below).

## The math

**Array geometry.** `kNumElements` = 64 elements on a pitch `kElementPitchM` = 0.3 mm, centered:
element *e* sits at lateral position

```
x_e = (e - (N-1)/2) · pitch
```

the one-line formula restated in `kernels.cu`, `reference_cpu.cpp`, and `main.cu`'s channel
synthesizer (kernels.cuh's file header names this the project's single most-duplicated fact).

**Plane-wave transmit.** A single, unsteered wavefront leaves the array parallel to it and
descends at constant speed c. At depth z, that wavefront arrives at time

```
t_tx(z) = z / c
```

— independent of lateral position x, which is exactly what "unsteered" and "plane" mean. Compare
a **focused transmit** (README/THEORY "Prior art"; README Exercise 3), where the array applies
per-element transmit delays so every element's wavefront arrives at one focal point
*simultaneously*, concentrating transmit energy there at the cost of only imaging that one depth
well per transmit. Plane-wave gives up transmit focusing gain everywhere in exchange for imaging
the **whole field of view from a single transmit** — the trade this project's THEORY leans on
throughout, and the trade that makes ultrafast (kHz) frame rates possible (README "System
context").

**Receive delay — the pulse-echo distance formula.** A point scatterer at (x, z) reflects the
plane wave; the echo travels back to element e (at x_e) along a straight line of length

```
r_rx(x, z, e) = sqrt( (x - x_e)^2 + z^2 )
```

so the **total two-way delay** — transmit descent plus scattered return — is

```
t(x, z, e) = t_tx(z) + r_rx(x, z, e) / c  =  z/c + sqrt((x-x_e)^2 + z^2) / c
```

This ONE formula is: (a) what `das_kernel`/`das_cpu` evaluate for every (pixel, element) pair to
know which channel sample to read; (b) what `simulate_channel_data()` evaluates for every
(scatterer, element) pair to know where to place that scatterer's pulse in the channel trace
(the phantom is placed in the exact frame the beamformer assumes); and (c) what `main.cu`'s
DELAY_CHECK gate re-derives independently in double precision as a sanity check. Three consumers,
one formula — CLAUDE.md §12's single-source-of-truth discipline made concrete.

**The pulse.** A short burst at the carrier frequency, shaped by a Gaussian envelope (this
project's documented choice — real transducers ring out something close to this, set by their
resonance bandwidth):

```
p(dt) = exp( -dt^2 / (2 sigma^2) ) · cos(2*pi*fc*dt)
```

with `dt` the time offset from the exact arrival and `sigma = kPulseSigmaS` chosen so the pulse
spans `kPulseCycles` = 2.5 carrier cycles (kernels.cuh derives the exact relation).

**DAS — the delay-and-sum image formation equation.** For every pixel (x, z), sum every element's
interpolated channel sample at that pixel's own delay, weighted by an apodization function w:

```
rf(x, z) = [ sum_e  w(x, z, e) * s_e( t(x, z, e) ) ]  /  [ sum_e w(x, z, e) ]
```

where `s_e(·)` is element e's continuous-time channel trace (interpolated between recorded
samples — "Numerical considerations" below) and the division by Σw is this project's documented
choice of *averaging* rather than summing (README "Limitations"; `kernels.cu`'s `das_kernel`
comment).

**Apodization — f-number-limited Hann, one formula, two jobs.** Define the active receive
half-aperture at depth z as

```
a(z) = z / (2 * F#)
```

(F# = `kFNumber` = 1.5, a fixed knob) and the element's normalized lateral offset from the pixel

```
u(x, z, e) = (x_e - x) / a(z)
```

then

```
w(u) = 0.5 + 0.5*cos(pi*u)   for |u| <= 1,   0 otherwise
```

is a **Hann window shaped continuously by distance**: 1 at the pixel's own lateral position,
tapering smoothly to exactly 0 at the f-number-limited aperture edge. Two classic beamforming
ideas fall out of one formula: (1) **aperture growth with depth** — a(z) grows linearly with z, so
near-field pixels use a narrow slice of the array (avoiding severe off-axis phase error and
grating lobes at wide angles) while far-field pixels use progressively more of the array, up to
the physical limit (`kApertureM` ≈ 18.9 mm — beyond which `a(z)` simply saturates against the
finite element loop); (2) **sidelobe control** — the smooth Hann taper (vs. a hard
rectangular window that would use every element in `a(z)` at equal weight) trades a wider
mainlobe for much lower sidelobes, the textbook windowing trade from digital filter design applied
spatially.

**Resolution — derive it, then measure it (the project's spine).**

*Axial* (along the beam, set by pulse length): a pulse of `kPulseCycles` cycles at wavelength
λ = c/fc has a "spatial pulse length" of `kPulseCycles · λ`; because the received echo is the
pulse convolved with itself on the way out and back, the classic first-order estimate for the
**resolvable axial separation** is half that:

```
axial_res ≈ kPulseCycles * lambda / 2
```

With `kPulseCycles` = 2.5 and λ ≈ 0.308 mm: **axial_res ≈ 0.385 mm**.

*Lateral* (across the beam, set by the receive aperture): a receive aperture of angular width
subtending the f-number-limited half-angle gives the classic diffraction-limited beamwidth
`lateral_res ≈ lambda * z / D_active(z)`. Substituting the f-number relation `D_active(z) = z / F#`
makes the depth cancel — the whole point of f-number-*constant* apodization:

```
lateral_res ≈ lambda * F#
```

With F# = 1.5 and λ ≈ 0.308 mm: **lateral_res ≈ 0.462 mm**. `kernels.cuh` names both formulas
`kAxialResM`/`kLateralResM`; `main.cu` prints them on an `[info]` line every run and then
**measures** the real thing from the simulated point-spread function (below) rather than trusting
the formula blindly.

**Quadrature demodulation — recovering the envelope.** The beamformed RF image still oscillates
at the carrier; `dB`-worthy "brightness" needs the envelope (the pulse amplitude, not its
instantaneous phase). Mixing a real narrowband signal `rf(t) = a(t)*cos(2*pi*fc*t + phi(t))`
(amplitude `a(t)` and phase `phi(t)` both slowly varying relative to fc — the **narrowband
assumption**) with `cos`/`-sin` at fc and low-pass filtering produces

```
I(t) = LPF[ rf(t)*cos(2*pi*fc*t) ]  ≈  (a(t)/2)*cos(phi(t))
Q(t) = LPF[ -rf(t)*sin(2*pi*fc*t) ] ≈  (a(t)/2)*sin(phi(t))
envelope = sqrt(I^2+Q^2) ≈ a(t)/2
```

This is the same operation an analytic-signal/Hilbert-transform envelope detector computes, valid
exactly when the narrowband assumption holds (mixing shifts the spectrum down by fc; the low-pass
keeps the near-DC term and rejects the image near 2·fc — a textbook single-sideband/downconversion
argument, not specific to ultrasound). This project references the phase `2*pi*fc*t` to each
pixel's own **on-axis** two-way arrival time `t(z) = 2z/c` (the DAS delay formula's on-axis
approximation), so the mixing reference varies smoothly with depth across the image — see
`quadrature_demod_kernel`'s comment for why that specific choice is correct for a *pixel-based*
(not scanline-based) beamformer.

## The algorithm

Per demo run (the numbered stages are labeled in `main.cu`):

1. **Load** the committed phantom (`data/sample/phantom.csv`, cross-checked against
   `kernels.cuh`'s constants via `data/sample/array_params.csv` — the 03.01 pattern).
2. **Synthesize channel data** (host): for every scatterer × every element, compute the two-way
   delay above, place a windowed pulse replica in that element's trace (only within
   `kChannelWindowSigmas` = 5σ of the exact arrival — an O(1)-per-scatterer windowed accumulate,
   not an O(samples) sweep), scaled by 1/r receive-spreading loss; add independent per-sample
   thermal noise. Complexity: O(scatterers × elements × window) ≈ 19,409 × 64 × 51 ≈ 63M — cheap,
   and NOT the taught algorithm (README "What this computes").
3. **GPU pipeline** (the taught algorithm, `kernels.cu`):
   a. `das_kernel` — for every pixel, sum every element's apodized, interpolated channel sample at
      the pixel's own two-way delay. Complexity: O(pixels × elements) ≈ 206k × 64 ≈ 13.2M.
   b. `quadrature_demod_kernel` — mix with cos/sin at fc. O(pixels).
   c. `envelope_lowpass_kernel` — 17-tap FIR along depth, then magnitude. O(pixels × 17).
   d. `log_compress_kernel` — 20·log10(env/max), clamped to [−50, 0] dB. O(pixels).
4. **CPU reference** (`reference_cpu.cpp`) — the same four stages, sequential, `std::` math.
5. **VERIFY** — three independent GPU-vs-CPU stage comparisons ("How we verify correctness").
6. **Four ground-truth gates** — localization, resolution, contrast, delay sanity (below).
7. **Artifacts** — `demo/out/bmode.pgm`, `demo/out/psf_profile.csv`.

**Serial vs. parallel cost.** Stage 3a (DAS) dominates: 13.2M element evaluations, each a handful
of FLOPs plus one `sqrtf` and one linear interpolation — GPU: ~1 ms; single CPU core: tens of
milliseconds (README "What this computes" quotes the measured pipeline totals). The pattern is
identical to every "map" kernel in this repository (33.01, 03.01): trivially, embarrassingly
parallel across the output index, serial-legible on the CPU, and that gap *is* the didactic point.

## The GPU mapping

```
das_kernel:            one thread PER PIXEL (idx = iz*Nx + ix), inner loop over 64 elements
quadrature_demod:       one thread PER PIXEL, pointwise (no neighbors)
envelope_lowpass:       one thread PER PIXEL, 17-tap 1-D STENCIL along the iz (depth) axis
log_compress:           one thread PER PIXEL, pointwise
```

**Why pixel-parallel, not element-parallel (the classic GPU-beamforming argument).** A naive
"parallelize the small axis" instinct would put one thread per ELEMENT (only 64 — a single warp's
worth, wildly under-occupying a GPU with dozens of SMs). Beamforming's actual parallelism lives on
the OUTPUT axis: ~206,000 pixels, each an independent small problem. This is the same lesson
33.01/09.01 teach with "thread per problem, not thread per operation," here at the scale (hundreds
of thousands of independent reconstructions) that makes software beamforming a genuinely
GPU-native workload — the reason every serious research ultrasound platform (Verasonics, README
"Prior art") is built around a GPU.

**Memory hierarchy, kernel by kernel:**

- `das_kernel`: `d_channel` reads are **data-dependent per thread** (each pixel's delay differs
  continuously per element) — not literally coalesced, but neighboring pixels (adjacent `ix` at
  fixed `iz`, one pixel pitch = 0.075 mm apart) compute *nearly identical* delays, so a warp's 32
  threads touch nearby channel addresses even without an affine access pattern; `__restrict__`
  hints the read-only/L2 cache path that actually absorbs this locality. No shared memory: nothing
  is reused *within* one pixel's own computation (each element is visited once).
- `envelope_lowpass_kernel`: for a FIXED tap offset, consecutive threads (consecutive `ix`) read
  consecutive addresses — genuinely coalesced despite being a "stencil down the image" (kernels.cu's
  comment works this out explicitly); `__constant__` memory holds the 17 FIR taps every thread
  reads identically every call — the textbook broadcast-cache use case, contrasted with 09.01's
  `__constant__` kinematic tree (bigger, read differently) and 08.01's uniform `u_nom` reads
  (same idea, global memory instead).
- No atomics anywhere in the beamforming pipeline — every pixel writes exactly one output, once.

**Occupancy.** All four kernels launch ~206,000/256 ≈ 805 blocks of 256 threads — far more blocks
than the reference RTX 2080 SUPER's 46 SMs can run concurrently, so the GPU stays saturated
end-to-end; `das_kernel`'s ~15–20 live registers (delay/weight/interp scratch) leave occupancy
register-bound-free, and the kernel is compute/latency-bound on `sqrtf`/`cosf`, not
bandwidth-bound — the healthy regime for this pattern (contrast with SAXPY-class kernels, which
are bandwidth-bound by design).

## Numerical considerations

- **Why linear interpolation, not nearest-neighbor, for the delay.** At fs = 40 MHz (25 ns/sample)
  and fc = 5 MHz (200 ns/cycle), a half-sample delay error is ~6% of one carrier period — enough
  to visibly distort the reconstructed pulse's phase pixel-to-pixel and show up as axial ringing
  in the image (README Exercise 2 asks you to break this on purpose and observe it). Linear
  interpolation is the cheap, adequate fix; a sinc/Lanczos interpolator would do better still at
  more cost per sample.
- **Delay precision in FP32.** Two-way delays here are O(1e-5) s; FP32's ~7 decimal digits of
  precision give sub-picosecond representable steps at that magnitude — utterly negligible next to
  the 25 ns sample period the interpolation error is measured against. `main.cu`'s DELAY_CHECK
  gate confirms this directly: the kernel-identical delay formula (float) and an independently
  re-derived double-precision closed form agree to ~7e-5 ns (measured), against a 5 ns tolerance —
  ~70,000x headroom.
- **Absolute, not relative, GPU-vs-CPU tolerances — a real bug this project's development caught.**
  An early version of the VERIFY gate compared GPU and CPU outputs with a *relative* tolerance
  guarded by a small floor (the common pattern elsewhere in this repo, e.g. 08.01/03.01). It
  reported a spurious worst-case relative deviation of **5%** — alarming, until the offending pixel
  was inspected: `cpu_rf` there was ≈0.003 (a near-null point, exactly where destructive
  interference among nearby speckle scatterers drives the coherent sum toward zero — "The
  problem"'s speckle-is-interference argument, showing up as a verification gotcha) and the
  absolute GPU-vs-CPU difference was ≈0.00065 — tiny, and entirely explained by `cosf`/`sqrtf`
  (GPU) vs. `std::cos`/`std::sqrt` (CPU) differing by roughly a ULP. Dividing that genuinely small
  absolute difference by the tiny floor manufactured a large, meaningless ratio. The RF image
  oscillates through zero by construction (it is a carrier-frequency signal) and the envelope is a
  Rayleigh-distributed magnitude that visits near-zero values at genuine destructive-interference
  nulls — relative comparison is the wrong tool at exactly the points where nulls occur. The fix:
  **absolute** tolerances at all three VERIFY stages (`kTolAbsRf` = 0.01, `kTolAbsEnv` = 0.005,
  `kTolAbsDb` = 0.05 dB), sized with generous headroom (~7–35x) over the measured worst-case
  absolute deviations. This is the same underlying lesson 08.01's "wide margin so ULP differences
  cannot flip the verdict" philosophy teaches, applied to a signal that genuinely crosses zero
  rather than one (like a rollout cost) that never does.
- **The resolution-formula ratio is not 1.0, and that is expected.** Measured lateral/axial
  −6 dB widths land at roughly 2.3–2.5x and 0.7–0.8x the first-order formulas above. Two real
  effects explain both directions: (1) **Hann apodization measurably widens the mainlobe** versus
  the uniform-illumination assumption the `lambda*F#` formula implicitly makes — a well-known
  windowing trade (narrower sidelobes cost mainlobe width; a rectangular window would measure
  closer to 1.0x but ring badly, defeating the point of apodizing at all); (2) the axial formula's
  "cycles·λ/2" rule of thumb ignores the envelope-detection FIR's own frequency response, which
  can sharpen the measured −6 dB crossing relative to the raw pulse autocorrelation. `main.cu`'s
  RESOLUTION gate therefore checks a **documented factor range (0.3x–3.0x)**, not equality — the
  formula teaches the right dependence (on λ, F#, and pulse length) and the code measures the real
  number, exactly the repo's "derive, then measure, then compare honestly" pattern (CLAUDE.md §9).
- **Determinism.** Channel-data synthesis and both beamforming pipelines are fully deterministic
  given the fixed phantom seed (42) — no run-to-run randomness anywhere in the taught algorithm.
  Every stable output line is a textual verdict or a compile-time-fixed parameter, never a measured
  number (README "Expected output"), so the same "wide margin, no volatile numbers on stable
  lines" discipline 08.01/03.01 use applies here too.

## How we verify correctness

Five independent checks, because a beamformer can be *numerically identical to its own CPU twin
and still image the wrong thing* (a systematic delay-formula bug would pass a self-consistent
GPU-vs-CPU check while still mis-locating every target):

1. **Three-stage GPU-vs-CPU gate (`VERIFY:`)** — DAS output, envelope, and log-compressed dB image
   each compared pixel-by-pixel against `reference_cpu.cpp`'s line-by-line twins, absolute
   tolerance (see "Numerical considerations" for why absolute). Catches indexing, layout, formula,
   and interpolation bugs — anything that would make the GPU and CPU compute *different* things,
   independent of whether either is *physically correct*.
2. **`LOCALIZATION:`** — every wire target's image peak lands within one resolution cell (0.462
   mm) of its true, phantom-file position. Catches sign errors in the delay/apodization formulas,
   wrong array geometry, or a wrong image-grid mapping — bugs invisible to check 1 if they are
   present identically on both the GPU and CPU paths (both wrong the same way still passes a
   GPU-vs-CPU-only check; comparing against known ground truth is the only way to catch that
   class of bug).
3. **`RESOLUTION:`** — the isolated center wire's measured point-spread function matches the
   derived formulas within a documented factor (above). Turns "the image looks reasonably sharp"
   into a number, and ties that number back to the physics (λ, F#, pulse length) that predicts it.
4. **`CONTRAST:`** — the high-scattering inclusion's mean dB exceeds background speckle's mean dB
   by a documented margin (measured ~7.6–8.7 dB, gated at ≥2 dB). Confirms the log-compression and
   normalization pipeline preserves a real reflectivity difference rather than washing it out.
5. **`DELAY_CHECK:`** — the delay FORMULA itself, isolated from the rest of the pipeline, evaluated
   two independent ways (the project's exact float formula vs. a from-scratch double-precision
   closed form). A pure unit test of the physics, not the execution.

The phantom is committed (`data/sample/`) so every check runs offline, deterministically; the two
artifacts (`demo/out/bmode.pgm`, `demo/out/psf_profile.csv`) make the result *inspectable*, not
just pass/fail.

## Where this sits in the real world

- **Commercial and research beamformers** (Verasonics Vantage, and every clinical scanner's
  internal pipeline) implement the same three conceptual stages at far greater sophistication:
  multiple transmit angles compounded per frame (coherent plane-wave compounding — trades some of
  the frame-rate headroom this project preserves for image quality), adaptive apodization
  (minimum-variance / eigenspace beamforming instead of a fixed Hann window), harmonic imaging
  (transmitting at fc, receiving at 2fc to reject near-field clutter), and extensive
  speckle-reduction post-processing. This project's fixed-apodization, single-plane-wave DAS is
  the honest, teachable core those systems are built on top of.
- **k-Wave and Field II** simulate the acoustic wave equation (or a rigorous linear-systems
  approximation of it) with frequency-dependent attenuation, nonlinear propagation, and realistic
  tissue models — a far more faithful channel-data synthesizer than this project's Gaussian-pulse,
  single-reflection-coefficient model. Studying either is the natural next step for anyone who
  wants the acoustics this project's "The problem" section only sketches.
- **PICMUS** is the public benchmark this project's plane-wave/DAS/apodization choices are modeled
  on; its real (not synthetic) RF datasets and published resolution/contrast metrics are the
  natural target for re-running this project's exact verification gates against real data.
- **Milestone 2 — elastography, what it would add.** Strain/shear-wave elastography estimates
  tissue *stiffness* — a mechanical, not acoustic, property — by tracking sub-resolution tissue
  displacement between successive B-mode frames (this project's `bmode.pgm`, repeated) under a
  small applied stress (manual palpation, an acoustic radiation-force "push" pulse, or natural
  physiological motion), typically via normalized cross-correlation or optical-flow-style matching
  between corresponding image patches (README Exercise 5 is the first step). The displacement
  field, differentiated spatially, gives strain; combined with a shear-wave speed measurement
  (from a tracked, laterally-propagating displacement pulse) it gives a real stiffness estimate
  (shear modulus). This project's B-mode pipeline is exactly the frame source that method needs.
- **Milestone 3 — image-based servoing, what it would add.** A control loop that extracts a
  feature (an anatomical landmark's position, a target's centroid) from each new B-mode frame and
  commands a probe-holding robot arm to keep that feature centered, or to sweep a path — the same
  `image → feature error → joint velocity` shape as this repo's camera-based visual servoing
  projects (domain 21), with a B-mode frame instead of a camera frame as the sensor. This project's
  measured ~1–2 ms GPU reconstruction time is not the bottleneck for that loop (README "System
  context" derives why plane-wave frame rate is); closing it would additionally need a feature
  tracker, a hand-eye calibration between the probe and the arm's kinematic chain (09.x), and — the
  moment it touches a real patient or a real arm — the full safety envelope PRACTICE.md §3
  describes and this repo's sim-validated-only rule (CLAUDE.md §1) applies to at full strength.
