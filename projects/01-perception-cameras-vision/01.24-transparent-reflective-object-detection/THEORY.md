# 01.24 — Transparent/reflective object detection via polarization imaging: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

### Light as a transverse wave, and what "polarized" means

Light is an oscillating electromagnetic field. The part that matters here is the **electric field
vector** `E`, which oscillates perpendicular to the direction the light travels (a *transverse* wave —
unlike sound, which oscillates *along* its direction of travel). At any instant, `E` points somewhere in
the plane perpendicular to travel; as the wave propagates, that direction can:

- stay **fixed** (oscillating back and forth along one line) — **linearly polarized** light, the only
  kind this project studies;
- **rotate** at the wave's frequency (circular/elliptical polarization) — not modeled here, and not
  produced by the physical processes (reflection off dielectrics and metals) this project simulates;
- **change randomly**, faster than any sensor can track — **unpolarized** light. A single atom emits a
  short polarized wave-train, but ordinary light sources (the sun, an LED, a light bulb) are enormous
  ensembles of atoms emitting independently, so the OBSERVED field's direction jumps around
  nanosecond-to-nanosecond. A camera integrating for microseconds-to-milliseconds sees the *time
  average* of that ensemble — which, for truly unpolarized light, contains equal energy along every
  direction in the transverse plane. This ensemble-average framing is exactly why the Stokes
  formalism (below) describes light with a handful of *intensity* numbers, never a single instantaneous
  field vector: a sensor cannot measure the instantaneous vector, only statistics of it.

### Where partial polarization comes from: specular reflection

Sunlight and most artificial light sources start unpolarized. It becomes **partially** polarized when it
reflects off a smooth (specular) dielectric or metal surface — glass windows, wet pavement, car
windshields, still water, and (this project's third object) brushed or polished metal. A rough, matte
surface (paper, cloth, most powder-coated robot housings) scatters light via countless microscopic
facets pointing every which way, and the polarization each micro-facet imparts averages back out across
the pixel's footprint — which is why this project's matte background carries only a small **residual**
DoLP (a few percent, `kBgDolp` in `kernels.cuh`), not exactly zero: no real matte surface is a perfect
depolarizer, and even a smooth-looking surface has SOME specular micro-facet component.

### The engineering constraint that motivates this project

A robot's other depth-sensing modalities are built on the *opposite* assumption: stereo and structured
light assume a diffuse, opaque, single-bounce surface so triangulation gives a meaningful depth; ToF
assumes the same for its round-trip-time measurement. Glass violates "opaque" (light mostly transmits,
so triangulation either fails outright or triangulates on whatever is *behind* the glass); a mirror-like
metal surface violates "diffuse" (a stereo pair sees a moving specular highlight, not the object's
actual geometry, and a structured-light pattern bounces away at the mirror angle instead of scattering
back toward the sensor). Both failure modes are catastrophic, not just noisy — a depth sensor reports
a *confidently wrong* number, which is worse for a planner than reporting nothing at all (README
"System context" names this explicitly for 01.02/01.19/01.20). Polarization imaging is the modality
that is not fooled by transmission or mirror reflection — it directly measures the light's
*polarization state*, which specular reflection off ANY smooth surface (dielectric or metal) reliably
alters, transmission mostly does not, and diffuse scattering mostly does not either. That asymmetry —
"specular reflection polarizes, (almost) nothing else does" — is the whole reason this sensing modality
exists on a robot.

## The math

### Notation

- `theta` — a linear polarizer's transmission-axis angle, degrees, measured the same way this project
  measures every in-image angle: counter-clockwise from image `+x` (CLAUDE.md §12's convention).
- `I(theta)` — the intensity (DN, this project's units) a linear polarizer at angle `theta` transmits
  from the incident light.
- `(S0, S1, S2)` — the (linear-polarization-only) Stokes parameters of the incident light, all in DN:
  `S0` is total intensity, `S1` is the excess of horizontal (`0 deg`) over vertical (`90 deg`)
  intensity, `S2` is the excess of `45 deg` over `135 deg`. (The full Stokes vector has a fourth
  parameter, `S3`, describing *circular* polarization — irrelevant here since specular reflection off a
  dielectric or metal under unpolarized illumination produces no circular component; this project omits
  it, as almost every DoFP polarization-imaging paper does.)
- `n` — refractive index of a dielectric (glass = 1.5 in this project), unitless.
- `theta_i`, `theta_t` — angle of incidence and angle of transmission/refraction, radians, measured from
  the local surface normal (the standard optics convention, not this project's image-angle convention).

### Malus's law and the DoFP forward model

A linear polarizer transmits the fraction `cos^2(alpha)` of an incident LINEARLY polarized wave's
intensity, where `alpha` is the angle between the wave's polarization axis and the polarizer's
transmission axis — that is Malus's law for a single polarized wave. Partially polarized light (this
project's actual subject) is the ensemble-average superposition of a polarized component (magnitude
`sqrt(S1^2+S2^2)`, axis `AoLP`, defined below) and an unpolarized component (magnitude
`S0-sqrt(S1^2+S2^2)`, contributing equally to every polarizer angle). Summing a polarizer's response to
both components (a short derivation using the double-angle identity `cos^2(x)=(1+cos(2x))/2`) gives
this project's central equation, quoted in `kernels.cuh`'s Section 2 as `(*)`:

```
I(theta) = S0/2 + (S1/2)*cos(2*theta) + (S2/2)*sin(2*theta)                       (*)
```

A DoFP sensor measures `(*)` at exactly four angles per scene point — `theta = 0, 45, 90, 135` degrees —
because its four polarizer orientations are etched directly over four adjacent photosites (Section
"The GPU mapping" below). This is a **sampled sinusoid**: `(*)` as a function of `theta` is a sinusoid
of period 180 degrees (note the `2*theta`, not `theta` — Malus's law is `pi`-periodic, half the period a
naive reading of "angle" would suggest, which is the ROOT CAUSE of the half-angle wrap this project
teaches below), and four samples at 45-degree spacing are enough to recover its three coefficients
`(S0,S1,S2)` exactly, in the absence of noise.

### Recovering (S0, S1, S2): why summing then averaging is least-squares, not a shortcut

Evaluate `(*)` at the four sampled angles:

```
I0   = S0/2 + S1/2          I45  = S0/2 + S2/2
I90  = S0/2 - S1/2          I135 = S0/2 - S2/2
```

Two *independent* combinations both recover `S0`: `I0+I90 = S0` and, separately, `I45+I135 = S0`. In the
presence of independent, equal-variance measurement noise on each of the four channels, the
**minimum-variance unbiased estimator** of `S0` is the average of every independent estimate available —
here, the average of the two: `S0_hat = ((I0+I90) + (I45+I135)) / 2 = (I0+I45+I90+I135)/2`, exactly the
formula `kernels.cu`'s `stokes_kernel` computes. `S1` and `S2` each have only ONE combination that
isolates them (`I0-I90` and `I45-I135` respectively) — no redundancy, so no averaging is possible or
needed for them. This is the precise sense in which Stage 2's Stokes estimation is a (trivial,
closed-form) **least-squares fit** of a 3-parameter model to 4 noisy measurements — not a shortcut, the
provably optimal combination under the stated noise model.

### The Malus self-consistency residual — the free 1-DOF invariant

Four measurements, three fitted parameters, leaves exactly **one degree of freedom unused by the fit** —
and that leftover DOF is directly observable with no ground truth at all: the model predicts
`I0+I90 = S0 = I45+I135` EXACTLY, so

```
residual = (I0+I90) - (I45+I135)                                                  (Stage 4)
```

is zero in noise-free physics and measures ONLY how inconsistent the four raw channel measurements are
with each other — sensor noise, a demosaic bug reading the wrong neighbor, a registration error between
channels, or Malus's law itself breaking down (e.g. from sensor nonlinearity) all show up here, with NO
dependence on the true scene. `main.cu`'s `GATE malus_consistency` reads this residual over the WHOLE
image (measured mean|residual| ~3.8 DN on the committed sample, well within the noise floor `(*)`
predicts — see "Numerical considerations" below) as a pure self-consistency check, independent of every
other gate.

### DoLP and AoLP

```
DoLP = sqrt(S1^2 + S2^2) / S0                              (unitless, in [0,1])
AoLP = 0.5 * atan2(S2, S1)                                  (radians)
```

`DoLP` (Degree of Linear Polarization) is the fraction of the light's total intensity that is
polarized: `0` for fully unpolarized light, `1` for fully polarized. `AoLP` (Angle of Linear
Polarization) is the polarization axis itself — the direction a linear polarizer would need to be
rotated to for MAXIMUM transmission.

**The half-angle wrap.** `atan2` naturally returns a value in `(-pi, pi]`; dividing by 2 (because
`(*)`'s argument is `2*theta`, not `theta`) compresses that into `(-pi/2, pi/2]` — but a polarization
AXIS at `theta` and at `theta+pi` are the SAME physical axis (a line has no arrowhead; rotating a
polarizer's transmission axis by 180 degrees does not change what it transmits). The convention this
project adopts (`kernels.cu`'s `dolp_aolp_kernel`): if the raw `0.5*atan2(...)` result is negative, add
`pi`, landing in `[0, pi)` — every physically distinct axis has exactly one representative in that
range. This means comparing two AoLP values for "how different are they" is **not** ordinary subtraction
(179 degrees and 1 degree are only 2 degrees apart on the actual axis, not 178) — `main.cu`'s
`circular_diff_rad_to_deg()` computes `min(|a-b|, pi-|a-b|)` for exactly this reason, and every AoLP
accuracy gate in this project uses it.

### The Fresnel equations, and why specular reflection polarizes

At a smooth interface between two media of refractive index `n1` (this project: air, `n1=1`) and `n2`
(glass, `n2=1.5`), Maxwell's equations plus the electromagnetic boundary conditions (continuity of the
tangential `E` and `H` fields) give DIFFERENT reflectances for the two independent linear polarization
components of the incident light, relative to the **plane of incidence** (the plane containing the
incoming ray and the surface normal):

```
r_s = (n1*cos(theta_i) - n2*cos(theta_t)) / (n1*cos(theta_i) + n2*cos(theta_t))      Rs = r_s^2
r_p = (n2*cos(theta_i) - n1*cos(theta_t)) / (n2*cos(theta_i) + n1*cos(theta_t))      Rp = r_p^2
```

with `theta_t` from Snell's law, `n1*sin(theta_i) = n2*sin(theta_t)`. `s`-polarized light (`E`
perpendicular to the plane of incidence) and `p`-polarized light (`E` within that plane) reflect with
DIFFERENT power fractions `Rs != Rp` at every incidence angle except `0` (straight-on) — where
`Rs = Rp` by symmetry, both equal to the normal-incidence reflectance `((n2-n1)/(n2+n1))^2`. Illuminate
the surface with UNPOLARIZED light (equal `s` and `p` power) and look at the REFLECTED beam: it now
carries `Rs` times as much `s`-power as incident but only `Rp` times as much `p`-power, so it is no
longer balanced — it has become **partially linearly polarized**, with

```
DoLP_reflected = (Rs - Rp) / (Rs + Rp)                                            (kernels.cuh's fresnel_dolp)
```

`Rs >= Rp >= 0` for every `theta_i` in `[0,90)` degrees for external reflection (`n2>n1`, going from a
less to a more optically dense medium — proved by inspecting the two ratios' signs), so `DoLP_reflected`
is always non-negative and in `[0,1]`, and — crucially — it is NEVER zero except at exactly two special
angles: `theta_i=0` (`Rs=Rp` by symmetry) and `theta_i=90` (`Rs,Rp -> 1` at grazing incidence, both
reflectances saturate to total reflection). Between those, `DoLP_reflected` RISES from 0.

### Brewster's angle: where `DoLP_reflected` peaks at exactly 1.0

Set `Rp = 0` and solve: this happens exactly when `theta_i + theta_t = 90` degrees, which (substituting
into Snell's law) gives the closed form

```
tan(theta_Brewster) = n2 / n1        (this project, n1=1: theta_Brewster = atan(n) = atan(1.5) = 56.31 deg)
```

At exactly this angle, the REFLECTED beam contains ZERO `p`-polarized light — it is perfectly, 100%
`s`-polarized (`DoLP_reflected = 1`, the physical reason polarized sunglasses cut glare from wet roads
and water: they block exactly the `s`-polarized reflection glare, which peaks near Brewster's angle for
those surfaces). This project's `GATE brewster_sweep` sweeps `theta_i` from 5 to 85 degrees through the
closed form above (`brewster_curve.csv`) and checks the peak lands within 2 degrees of `atan(1.5) =
56.31`.

### Why the dome shows a ring, and the pane and metal bar do not

The curved glass dome's LOCAL incidence angle is not constant — under an orthographic (far-camera)
approximation, a point on a hemisphere at 2-D image-plane radius `r` from the dome's center has a
surface normal tilted `theta_i(r) = asin(r/R)` from the view axis (`R` = dome radius). Substituting into
the Fresnel formula above, `DoLP(r)` RISES from `0` at the center (`r=0`, normal incidence) to `1.0` at
the radius where `theta_i(r) = 56.31 deg` (`r/R = sin(56.31 deg) = 0.832`) and FALLS back to `0` at the
silhouette (`r=R`, grazing incidence) — the real "polarization donut"/Brewster-ring pattern
photographed on specular spheres. The flat pane has ONE fixed `theta_i` everywhere, so it shows ONE
DoLP value, not a ring. The metal bar's DoLP curve (below) never reaches a Brewster zero at all, so it
shows a smooth gradient, not a ring either.

### AoLP as a shape cue — why the dome's AoLP is radial and the bar's is constant

The reflected `s`-polarized component (the dominant one, since `Rs>=Rp` always here) has its `E` field
PERPENDICULAR to the plane of incidence — so `AoLP`, projected into the image, points perpendicular to
that plane's trace in the image. For the dome, the plane of incidence at image point `(x,y)` contains
the view axis and the RADIAL direction from the dome's center (the local surface normal always tilts
radially outward on a sphere), so `AoLP(x,y) = radial_angle(x,y) + 90 deg` — a genuinely 2-D, spatially
varying pattern (`kernels.cuh`'s dome model). For the metal bar (a cylinder with its axis HORIZONTAL,
curvature only in `y`), the local surface normal at every point along the curve tilts ONLY within the
same fixed vertical plane (the `y`-`z` plane, independent of `x`) — so the plane of incidence has the
SAME orientation everywhere on the bar, and `AoLP` is CONSTANT across it, even though `DoLP` varies with
`y`. This constant-vs-radial AoLP contrast is itself a real, used diagnostic in shape-from-polarization
research (README "Prior art"): it distinguishes cylindrical from spherical curvature from the
polarization image alone, with no depth sensor at all.

## The algorithm

1. **Demosaic** (`O(n)`, `n`=pixel count): for each pixel, its OWN channel is a direct copy; the other 3
   are recovered by BILINEAR interpolation across that channel's own spacing-2 sub-lattice (the 4
   nearest same-phase samples, `kernels.cuh`'s `PhaseSample`). Serial and parallel cost are both `O(n)` —
   this stage is memory-bound, not compute-bound (Section "The GPU mapping").
2. **Stokes** (`O(n)`): three closed-form combinations of the 4 channels per pixel.
3. **DoLP/AoLP** (`O(n)`): one `sqrt`, one division, one `atan2`, one conditional add per pixel.
4. **Malus residual** (`O(n)`): one subtraction per pixel.
5. **Detection** (`O(n)` per stage, `O(n)` total, run TWICE — once per signal):
   a. **Threshold** (`O(n)`): `signal >= thresh -> {0,1}`.
   b. **Morphological open** (`O(n)`, fixed 3x3 stencil): erode (AND over the 3x3 neighborhood) then
      dilate (OR over the 3x3 neighborhood) — removes isolated few-pixel false positives without
      shrinking a real blob's silhouette (the standard opening operator; 01.21/30.01's cited pattern).
   c. **Connected-component labeling** (`O(n * k)`, `k` = number of sweeps to convergence, `k` bounded
      by the largest component's diameter — see "The GPU mapping"): label-propagation to a fixed point.
   d. **Size filter** (`O(n)`, an atomic scatter counting each component's size, then a map dropping
      components below `kMinComponentSizePx`) — the same "count-then-filter" pattern 01.06/01.21/30.01
      use, cited by name.

## The GPU mapping

**Stages 1-4 (demosaic through the Malus residual) are pure MAPs**, the simplest GPU pattern that
exists: thread `i` (`= blockIdx.x*blockDim.x + threadIdx.x`) owns output pixel `i`, reads a small fixed
set of GLOBAL-memory inputs, writes its own output, and touches no other thread's data. No shared
memory is used because no data is reused ACROSS threads (only within one thread's own 4-corner or
4-channel gather) — shared memory only pays when threads in the SAME block revisit the same bytes,
which does not happen here (contrast this project's kernels with, say, a convolution kernel with
overlapping stencils across neighboring threads, or 08.01 MPPI's rollout kernel, which is the repo's
other flagship "no shared memory, pure independent map" example, for a different reason: no data reuse
at all rather than no CROSS-thread reuse). Demosaic's 4-corner gather reads up to 16 global loads per
pixel (4 channels x 4 corners), which is more traffic than a 1-corner map but still fully coalesced
within a warp (adjacent threads read adjacent mosaic addresses) and still embarrassingly parallel.

**Morphological open (Stage 5b) is a fixed-radius STENCIL**: each output pixel reads its own 3x3
neighborhood. `kernels.cu`'s `erode3x3_kernel`/`dilate3x3_kernel` use a 2-D thread mapping
(`(blockIdx.x*blockDim.x+threadIdx.x, blockIdx.y*blockDim.y+threadIdx.y)`, block `16x16=256` threads)
because the access pattern is inherently 2-D — a 1-D flat index would still work correctness-wise but
loses the natural 2-D coalescing a `16x16` tile gives (adjacent threads in `x` read adjacent memory,
adjacent threads in `y` read one row apart, both cache-friendly). No shared-memory tiling is used
despite the 3x3 reuse (a genuine simplification, named honestly): each pixel's 9-neighbor read overlaps
its 8 neighbors' reads, so a shared-memory HALO tile would cut global traffic roughly 9x for a
production kernel — this project keeps the direct-global-read version because the mask is 1 byte/pixel
(the whole 128x128 mask is 16 KB, comfortably L2-cache-resident on any GPU this repo targets) and
because the READABILITY win of "every load is where you'd expect it" matters more here than the modest
bandwidth saving would (the tiling exercise is a natural follow-on for a learner comfortable with
kernels.cu already).

**Connected-component labeling (Stage 5c) is the project's only ITERATIVE kernel and its only use of
ATOMICS.** `ccl_init_kernel` seeds every foreground pixel with its own linear index (a map). Each
`ccl_propagate_sweep_kernel` SWEEP is itself a map (thread `i` reads its 4-connected neighbors' CURRENT
labels and keeps the minimum via `atomicMin`), but the ALGORITHM needs repeated sweeps because
information about a component's true minimum label must physically propagate, one 4-connected hop per
sweep, across the whole component — a component of diameter `d` pixels needs up to `d` sweeps to
converge (kernels.cuh's `kMaxCclSweeps=256` safety cap, `max(kW,kH)=128` worst case, matching 01.21's
identical reasoning on the identical canvas size). `atomicMin`/`atomicOr` (the convergence flag) are
needed because MULTIPLE threads can discover a smaller label for the SAME pixel in the SAME sweep from
different neighbors, and the update must be race-free; the algorithm's correctness argument (every
label only ever DECREASES, bounded below by 0) is exactly what makes convergence to a UNIQUE fixed
point — independent of scheduling order — provable, which is in turn exactly why this GPU algorithm and
`reference_cpu.cpp`'s completely different sequential union-find algorithm can be held to a BIT-EXACT
tolerance (both converge to "min linear index in the component" — see "How we verify correctness").

**The size filter (Stage 5d) is an ATOMIC SCATTER followed by a map**: `component_size_count_kernel` has
every foreground pixel `atomicAdd` 1 into a per-LABEL bucket (many pixels write the SAME bucket — a true
scatter, unlike every earlier stage in this project) — this is the one place in the pipeline where
threads genuinely CONTEND for the same memory location, and atomics are the correct, simplest fix (the
buckets are small, `kN` `int`s, easily resident; contention is bounded by the largest component's
pixel count, at most a few thousand atomics into one address — negligible next to a GPU's throughput).
`component_filter_kernel` then re-reads each pixel's own bucket, a pure map.

**No cuBLAS/cuFFT/Thrust/CUB anywhere** — every operation here (elementwise map, fixed-radius stencil,
label propagation, atomic scatter-count) is simple enough, and different enough in kind from stage to
stage, that hand-rolling teaches more than a library call would; there is no black box to explain away
(CLAUDE.md §1).

## Numerical considerations

- **Precision.** Every continuous stage is FP32 throughout (matching the repo's default and the DoFP
  sensor's realistic 8-12 bit ADC output). `reference_cpu.cpp` deliberately uses the SAME `float`
  precision as the GPU (unlike, e.g., 01.22's FFT twin, which uses `double` for its stronger oracle) —
  because every formula here is a few arithmetic ops with no accumulated rounding to worry about, a
  plain `float` CPU re-implementation is already a trustworthy oracle, and keeping both sides FP32 lets
  `VERIFY` tolerances be TIGHT (a few `1e-3`, not the loose multi-DN tolerances an FFT-based project
  needs) — see "How we verify correctness".
- **DoLP's low-`S0` bias.** `DoLP = sqrt(S1^2+S2^2)/S0` divides by `S0`; as `S0 -> 0` this blows up (or,
  with `kernels.cu`'s epsilon floor `max(S0, 1e-3)`, saturates at a large-but-finite value) even for
  small, PURELY-noise `S1,S2` — a classic "ratio of two noisy quantities" bias. This project's scene
  never puts real `S0` anywhere near that floor (`S0` ranges ~118-195 DN throughout, "Data" in
  `data/README.md`), so the floor is a numerical-safety guard, not a bias source that affects any gate
  here — but a learner adapting this code to a scene with near-black regions (a shadowed glass edge,
  say) should expect DoLP there to be UNRELIABLE, not merely noisy, and should gate on a minimum `S0`
  before trusting a DoLP measurement (a real system's "confidence mask", named in PRACTICE.md §3).
- **AoLP wrap statistics — the circular mean, taught here explicitly.** Because AoLP lives on a
  half-circle (`[0,pi)`, see "The math"), naively AVERAGING two AoLP values near the wrap boundary (say
  `1 deg` and `179 deg`) gives `90 deg` — physically nonsensical (their true "average axis" is close to
  `0/180 deg`, not `90 deg`). Every AoLP statistic in this project (`main.cu`'s `stokes_accuracy` gate)
  is therefore a mean of PER-PIXEL CIRCULAR DISTANCES (`circular_diff_rad_to_deg`), never a mean of raw
  angles — the correct way to summarize directional data, the same lesson standard circular-statistics
  texts teach for compass headings or clock times.
- **Float noise in `S1`/`S2` differences.** `S1=I0-I90` and `S2=I45-I135` are DIFFERENCES of two
  similar-magnitude noisy quantities — the classic catastrophic-cancellation setup, though at this
  project's DN scale (values in the tens-to-hundreds) and FP32 precision (~7 decimal digits), the
  cancellation loses negligible SIGNAL precision; what it DOES do is directly propagate roughly
  `sqrt(2)` times the per-channel sensor noise standard deviation into `S1`/`S2` (two independent noisy
  channels, subtracted) — this is real, physical noise amplification (not a numerical artifact), and is
  exactly why the background's measured DoLP (mean ~0.03-0.04 on the committed sample, "Numerical
  considerations" is where a learner should look when `kDolpThreshold` needs re-tuning) is noticeably
  higher than the TRUE background DoLP (`kBgDolp=0.018`) baked into the generator — the gap is this
  noise amplification, not a bug.
- **Determinism.** `scripts/make_synthetic.py` fixes its RNG seed (42, xorshift32); every stage in this
  project's C++/CUDA pipeline is a deterministic map or a race-free (atomic-guarded) reduction/scatter —
  re-running the demo on the SAME GPU reproduces the SAME `[info]` numbers exactly (the `atomicMin`
  label-propagation sweep count can, in principle, vary with scheduling on some GPUs since different
  interleavings can reach the fixed point in a different number of ROUNDS even though the FINAL labels
  are provably identical — the committed `expected_output.txt` never checks the sweep count itself, only
  the `PASS`/`FAIL` verdicts, for exactly this reason).

## How we verify correctness

Two INDEPENDENT verification legs, deliberately never routed through each other (kernels.cuh's
twin-independence ruling, restated at the top of `reference_cpu.cpp`):

1. **VERIFY: GPU vs. an independently-coded CPU twin**, per stage. The continuous stages
   (demosaic/Stokes/DoLP/AoLP/Malus residual) use TIGHT float tolerances (`1e-2` DN / `1e-3` unitless —
   see "Numerical considerations" for why FP32-vs-FP32 with no accumulation allows this). The detection
   stages (threshold/morph/CCL/filter) are checked BIT-EXACT — but only after feeding BOTH sides the
   SAME already-verified-close signal array (the `main.cu` "VERIFY isolation" pattern, borrowed from
   01.22's IBP twin): this isolates the test to "does the discrete algorithm (deterministic integer/
   boolean ops on IDENTICAL floats) agree", not "do two independently-rounded float pipelines happen to
   threshold the same way at a boundary pixel" — the latter would occasionally and legitimately
   disagree by a pixel or two for reasons that have nothing to do with a bug. Connected-component
   labeling gets the REPO'S STRONGEST independence form: the GPU runs iterative label-propagation
   (Section "The GPU mapping"), the CPU runs classical two-pass Rosenfeld union-find — a genuinely
   different ALGORITHM, not a retyped twin, so a bug shared by "the same algorithm written twice" cannot
   hide behind two structurally unrelated ones that still must agree (01.21's precedent, cited by name).
2. **GATE: pipeline output vs. physics/ground truth**, six independent checks that never call either
   implementation above:
   - `stokes_accuracy` — DoLP/AoLP MAE against `truth_maps.csv`'s per-pixel noise-free ground truth
     (the SYNTHETIC generator's own physics, computed independently in Python — see `data/README.md`),
     restricted to each object's INTERIOR (eroded by `kInteriorMarginPx`) to exclude the demosaic
     edge-blur ring, an honestly-documented artifact rather than something papered over by a looser
     tolerance.
   - `malus_consistency` — the free 1-DOF invariant, needs no ground truth at all ("The math").
   - `fresnel_anchor` — **the physics gate**: `main.cu`'s OWN C++ evaluation of the closed-form Fresnel
     equations (`kernels.cuh`'s `fresnel_dolp()`) against the MEASURED DoLP on the rendered pane. This
     closes a genuine loop: `scripts/make_synthetic.py` independently re-derives the SAME equations in
     Python to RENDER the pane; `main.cu` independently re-evaluates them in C++ to CHECK it. Two
     different languages computing the same physics and finding pixel-level agreement (measured
     `|diff|=0.00034` on the committed sample) is a substantially stronger claim than "the code agrees
     with itself".
   - `detection` — recall in BOTH directions against `truth_maps.csv`'s object labels: DoLP-based
     detection must FIND the objects (floors on glass/metal recall); intensity-based detection, run
     through the IDENTICAL detection code with a different input signal, must MISS the glass (a ceiling
     on intensity-based glass recall) — the project's central, falsifiable claim, stated as a measurable
     number rather than an assertion.
   - `brewster_sweep` — a pure closed-form check with NO rendering involved at all: does the analytic
     Fresnel curve's peak land near `atan(1.5)`?
   - `negative_control` — the WHOLE pipeline, re-run on a matte-only scene with an independent noise
     draw, must detect exactly zero objects — the standard false-positive-rate sanity check every
     detector in this repository runs (01.01's distortion negative control, 01.11's noise floor, and
     others, cited by convention).

## Where this sits in the real world

Production polarization-imaging systems (README "Prior art") differ from this teaching pipeline in
several concrete ways:

- **Sensors**: real systems use commercial DoFP sensors (Sony's IMX250MZR/IMX253MZR family and similar
  parts from LUCID Vision Labs, FLIR, and others — PRACTICE.md §2 dates and sources this) whose
  per-pixel polarizer EXTINCTION RATIO (how well each polarizer actually blocks the "wrong" angle) is
  imperfect and must be CALIBRATED per pixel — this project assumes ideal polarizers (extinction ratio
  infinite), a stated simplification (PRACTICE.md §2's calibration discussion is where the real
  procedure lives).
- **Demosaic**: production ISPs use edge-aware or intensity-guided demosaic (the polarization analogue
  of 01.23's Malvar-He-Cutler upgrade over plain bilinear), reducing the instantaneous-FOV blur this
  project documents rather than works around.
- **Metal physics**: a rigorous treatment of metallic (conductor) reflection needs the COMPLEX
  refractive index `n_complex = n - i*k` (`k` the extinction coefficient) in the SAME Fresnel-equation
  structure, producing reflectances and a DoLP curve that, unlike a dielectric, generally NEVER reaches
  zero (no real Brewster null for most metals) — this project's saturating phenomenological curve
  (`kernels.cuh`'s `kMetalDolpMax`/`kMetalSat`) is a hand-tuned STAND-IN for that physics, chosen to be
  qualitatively right (monotonic rise, no Brewster zero, lower ceiling than glass's peak) without
  requiring complex-number Fresnel algebra (Exercise 4 points at the real formula).
- **Detection**: modern systems (README "Prior art"'s ClearGrasp-era citation) use LEARNED segmentation
  networks fed DoLP/AoLP/intensity as extra input channels, far more robust to lighting, clutter, and
  partial occlusion than this project's fixed-threshold detector — but they are trained on exactly the
  physics this project makes explicit, and a from-scratch DoLP-threshold baseline like this one remains
  the right FIRST thing to try, and the right sanity check for a learned model's behavior at the
  physics-obvious cases.
- **Shape-from-polarization**: beyond detection, AoLP's surface-normal-azimuth information (this
  project's dome vs. metal-bar contrast) is the basis of an entire research area recovering fine surface
  shape from polarization cues alone, or fused with a coarse depth sensor — named in README "Prior art"
  and worth a literature dive for a learner who found the dome's "polarization donut" interesting.
