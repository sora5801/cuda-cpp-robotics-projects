# 01.19 — Structured-light decoding (Gray code, phase shift) for 3D scanners: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

### Why active triangulation at all

A single camera measures a 2-D projection of the world; recovering the missing third dimension needs
a SECOND, independent measurement of the same point. **Passive stereo** (project 01.02) gets that
second measurement from a second camera and finds correspondences by matching TEXTURE — pixels that
look alike. That fails exactly where texture is absent: a matte white wall, a plain machined metal
part, a person's skin. **Structured light** solves the correspondence problem a completely different
way: replace the second camera with a **projector**, and instead of hoping the scene carries enough
texture to match, PAINT the texture onto it yourself. The projector becomes an "inverse camera" —
instead of measuring which pixel a ray of light arrives at, it EMITS a ray of light from a known
pixel. If a camera pixel can be told *which projector pixel's light it is seeing*, that is exactly the
same correspondence a stereo pair needs — recovered by construction, not by search, and it works
identically well on a textureless surface, because now the LIGHT carries the texture.

### The projector as an inverse pinhole camera

A real projector (DLP: a digital micromirror device: an array of ~10^6 individually-tiltable mirrors
switching a light source's illumination on/off per pixel at kHz rates; or 3LCD/LCoS: transmissive or
reflective liquid-crystal panels modulating light intensity per pixel) obeys the SAME pinhole
projection equations as a camera, run backwards: a 3-D point maps to a projector pixel via
`(fx_p, fy_p, cx_p, cy_p)` exactly like a camera's `(fx, fy, cx, cy)` (project 01.17's camera model,
cited here, is the identical math). This project encodes only the projector's COLUMN axis (vertical
stripe patterns) — a real scanner CAN encode rows too (denser correspondence, more patterns), but a
single axis is already enough to determine 3-D position once paired with a camera, and it halves the
frame count for the same code depth. The consequence, geometrically: a fixed projector COLUMN does
not pick out one ray — every ROW at that column shares it — so a decoded column corresponds to a
**plane** of light in 3-D, not a ray. "The math" below makes this precise.

### The baseline/depth-precision relation (derived)

Treat the camera-projector pair as a stereo rig with baseline `b` (meters) and the projector's
column-axis focal length `f_p` (pixels). For this project's parallel-axis rig (`kernels.cuh`:
identity rotation, translation `(b,0,0)`), a 3-D point at depth `Z` and camera-frame `X` projects to
camera column `u` and projector column `u_p` related by a disparity-like quantity. Define the
**generalized disparity** `D = dx - m`, where `dx = (u-cx)/fx` (camera ray slope) and
`m = (u_p - cx_p)/f_{xp}` (projector-plane slope) — exactly the quantities `triangulate_kernel`
computes. The closed-form triangulation (derived in full below) gives

```
Z = b / D          =>          D = b / Z
```

Differentiating `D = b/Z` with respect to `Z`:

```
dD/dZ = -b/Z^2      =>      dZ = -(Z^2/b) dD
```

A column-decode error `d(col)` (in projector columns) propagates to `dm = d(col)/f_{xp}` and hence
`dD = -dm` (holding the camera side fixed), giving the familiar stereo depth-precision law:

```
      Z^2
dZ ~ ----- * d(col)          (the catalog's "dz ~ z^2/(f*b) * d_disparity", f = f_xp here)
     f_xp * b
```

**Worked check against this project's own measurement:** at the scene's typical depth `Z ~ 0.75 m`,
baseline `b = 0.12 m`, `f_xp = 180 px`, and the HYBRID stage's measured mean column error
`d(col) ~ 0.071` columns (README "Expected output"): `dZ ~ 0.75^2 / (180*0.12) * 0.071 ~ 1.85 mm` — the
right order of magnitude for, and consistent with, the demo's own measured plane-fit RMS residual
(~3.8 mm; the RMS also includes sensor-noise-driven per-pixel scatter the mean-error figure above does
not capture). This is precision improving with the SQUARE of getting closer and LINEARLY with a wider
baseline or finer sub-pixel decoding — the three knobs every real scanner design trades against each
other (a wider baseline improves precision but shrinks the overlapping field of view both devices can
see; THEORY.md "Where this sits in the real world" and `PRACTICE.md` §1 return to this trade).

### Radiometry: what the camera actually receives

A camera pixel viewing a surface point with diffuse albedo `a in [0,1]` under projector illumination
pattern intensity `I_pat in [0,1]` and ambient light `A_amb` receives (this project's forward model,
`scripts/make_synthetic.py`):

```
captured = A_amb + a * I_pat * G + noise         (G: a fixed radiometric gain, counts)
```

Every decode step below either CANCELS the `A_amb + a*G` common factor (Gray code's direct/inverse
comparison) or CANCELS `A_amb` exactly while leaving a `a`-scaled signal (phase shift's differencing)
— "The math" derives both. The one thing NO amount of clever decoding can recover is signal that was
never there: a point outside the projector's illuminated column range, or one whose albedo is so low
the illuminated/ambient signal difference sits inside the sensor's noise floor, carries no usable
correspondence. This project's confidence gate exists to DETECT that condition, not to fix it — see
"Numerical considerations" and the `dark_stripe_honesty` gate.

### Engineering constraints a real scanner imposes

- **Optical defocus/blur.** Neither the projector lens nor the camera lens is a perfect point-spread
  function; both blur fine stripe/fringe detail, worse at close range or wide aperture. This project's
  synthetic renderer applies a mild 3x3 blur to every captured frame for exactly this reason — and the
  `gray_vs_binary` boundary stress test's OWN blur model (kernels.cu) captures the mechanism by which
  blur turns a code BOUNDARY into a genuine decode risk.
- **Motion.** Every temporally-coded pixel needs `N` frames captured at the SAME surface position; any
  relative motion during that window corrupts the code (README "Limitations" is explicit about this
  cost, and names single-shot alternatives).
- **Working-distance and field-of-view overlap.** The camera and projector must both see the same
  patch of scene — this project's own synthetic scene has real pixels OUTSIDE the projector's
  illuminated range (documented, honest `kInvalidColumnF` sentinel, not hidden).
- **Ambient light and specular surfaces.** Bright ambient light or shiny/specular surfaces (raw metal,
  glass) reduce fringe/stripe CONTRAST — exactly the modulation-amplitude confidence signal this
  project computes and thresholds on.

## The math

**Notation.** Camera frame = world frame (`T_world_camera = I`). `(u,v)` camera pixel (columns, rows,
pixel-CENTER convention: `dx=(u+0.5-cx)/fx`, `dy=(v+0.5-cy)/fy`, shared bit-for-bit between
`kernels.cuh`'s C++ and `make_synthetic.py`'s Python ray casting). `u_p` projector column (pixels,
continuous when "true", integer when Gray-decoded). `R_cp` (3x3), `t_cp` (3x1): the projector's pose
IN the camera frame (`T_camera_projector`).

### Gray-code decode

For bit plane `i in [0,N)` (`N = kGrayBits = 7`), the projector shows the direct pattern (illuminate
column `c` iff bit `i` of `Gray(c)` is 1) and its photometric inverse. A camera pixel decides bit `i`
by comparing the two captures:

```
b_i = 1  if  captured_direct_i > captured_inverse_i  else 0
```

Substituting the radiometric model: `captured_direct_i = A_amb + a*b_i*G + n_1`,
`captured_inverse_i = A_amb + a*(1-b_i)*G + n_2`. The DIFFERENCE `captured_direct_i - captured_inverse_i
= a*G*(2*b_i - 1) + (n_1-n_2)` — independent of `A_amb` and scaled (not shifted) by albedo `a`. A
pixel that is uniformly bright OR uniformly dark needs no separate per-pixel calibration: the SIGN of
the difference is the bit, regardless of scale — the whole reason "captured + inverse" beats a single
fixed brightness threshold (which would need to know `A_amb` and `a` in advance).

**Gray -> binary.** `Gray(v) = v XOR (v >> 1)`. To invert: with MSB-first bit indexing
`g_0..g_{N-1}` and `v_0..v_{N-1}`, the defining relation is `g_0 = v_0`, `g_k = v_k XOR v_{k-1}` for
`k >= 1`. Solving forward: `v_0 = g_0`, `v_k = g_k XOR v_{k-1}` — a running XOR, exactly the
`decode_gray_to_binary` recurrence in `kernels.cu`/`reference_cpu.cpp`.

**Single-bit adjacency (the property the catalog bullet asks to prove).** Reflected Gray code is
CONSTRUCTED recursively to have this property (not merely observed to): the 1-bit code is `(0,1)`.
The `(k+1)`-bit code is the `k`-bit sequence prefixed with `0`, followed by the SAME sequence
REVERSED and prefixed with `1`. Every step within either half changes only the low `k` bits by
induction (the `k`-bit code's own adjacency property); the single step ACROSS the middle (from the
end of the first half to the start of the reversed second half, i.e. from the largest `k`-bit code to
itself again but prefixed `1`) changes ONLY the new top bit, because the two halves meet at the
IDENTICAL `k`-bit suffix (one written forward, one at the start of its own reversal). By induction,
every one of the `2^{k+1}-1` consecutive-pair transitions changes exactly one bit. Plain binary has no
such structure: incrementing `0111111` (63) to `1000000` (64) flips all 7 bits at once — the MSB
boundary, the worst case this project's `gray_vs_binary` gate specifically targets.

### Phase-shift decode

The projector shows `K = 4` (`kPhaseSteps`) sinusoidal patterns with period `P = 8` columns
(`kPhasePeriodCols`), step `k` phase-offset by `k*pi/2`:

```
I_k(u_p) = 0.5 + 0.5*cos(2*pi*(u_p mod P)/P - k*pi/2)      k = 0..3
```

A camera pixel receives `captured_k = A_amb + a*I_k*G + noise_k = A_k + B_k*cos(phi - k*pi/2)`, where
`phi = 2*pi*(u_p mod P)/P` is the phase we want, `A_k = A_amb + a*G/2` (constant across `k` — no
`k`-dependence), and `B_k = a*G/2` (also constant across `k`). Expanding the four samples:

```
I_0 = A + B cos(phi)          I_1 = A + B sin(phi)
I_2 = A - B cos(phi)          I_3 = A - B sin(phi)
```

so `I_1 - I_3 = 2B sin(phi)` and `I_0 - I_2 = 2B cos(phi)` — **the ambient term `A` cancels EXACTLY**
in both differences (subtracting two captures that share the SAME additive ambient contribution), and
critically it cancels regardless of WHAT `A` is — add any constant to all four `I_k` and the two
differences, hence `phi` and `B`, are UNCHANGED. This is the exact arithmetic the
`phase_ambient_invariance` gate exploits (and measures at `0.0` rad change, to float precision, on the
committed sample). Recovering the phase and the modulation amplitude:

```
phi = atan2(I_1-I_3, I_0-I_2)                    (wrapped; this project uses [0, 2*pi), see below)
B   = 0.5 * sqrt((I_1-I_3)^2 + (I_0-I_2)^2)       (the CONFIDENCE signal — falls out for free)
```

**Why 4 steps, and why not 3.** The theoretical minimum is `K=3` (3 unknowns — `A`, `B`, `phi` — need
3 equations); this project uses `K=4` for the SAME reason most real scanners do: the extra sample lets
the ambient/albedo cancellation above happen through two clean SUBTRACTIONS instead of a more
noise-sensitive combination of three asymmetric samples, and it symmetrizes the noise across all four
quadrature terms (each of `I_0..I_3` contributes to exactly one of the two differences, not a mix).

**Angle-wrapping convention (a documented deviation).** CLAUDE.md's repo-wide default wraps angles to
`(-pi, pi]`. This project instead wraps `phi` to `[0, 2*pi)` (`if (phi<0) phi += 2*pi` after
`atan2`) because it must map MONOTONICALLY onto a projector column offset in `[0, kPhasePeriodCols)`
for the hybrid stage below — a negative offset has no physical meaning here. This is the ONE defined
wrap point for this project's phase values (CLAUDE.md §12: "wrapped... at defined points only").

### Hybrid combine — phase-guided period snapping

Gray code gives an absolute but INTEGER column `g in [0, 128)`; phase gives a precise but WRAPPED
fractional position within one period, `frac = (phi/2pi)*P in [0,P)`. The naive combination
`period = floor(g/P); hybrid = period*P + frac` fails exactly at period boundaries: Gray-decoded
`g=8` (quantization put it just past a boundary) with a true fractional position `frac ~= 7.98`
(correctly measuring "almost at the top of period 0") gives `floor(8/8)=1`, so `hybrid = 8 + 7.98 =
15.98` — nearly a full period of error from a Gray answer that was off by ONE integer.

**This project's rule:** recompute the period as the one CONSISTENT with the precise fraction,

```
period = round((g - frac) / P)
hybrid = period * P + frac
```

Re-solving the example: `round((8 - 7.98)/8) = round(0.0025) = 0`, giving `hybrid = 7.98` — correct.
Gray code only has to get the period right to the NEAREST period (an error tolerance of `+-P/2`
columns, `+-4` here), not exactly — the coarse, absolute code buys robustness for the fine, wrapped
one. This is the production pattern named by the catalog bullet and implemented identically (twice,
independently) in `kernels.cu`'s `hybrid_combine_kernel` and `reference_cpu.cpp`'s
`hybrid_combine_cpu`.

### Ray / projector-plane triangulation

A projector COLUMN `u_p` does not pick out a ray — every row at that column shares it, so it picks
out a PLANE. In the projector's own frame, the pinhole equation `u_p = f_{xp}*X_p/Z_p + c_{xp}` at
FIXED `u_p` is `X_p - m*Z_p = 0` (`m = (u_p-c_{xp})/f_{xp}`) for ANY `Y_p, Z_p` — a plane through the
projector's optical center with normal `n_p = (1, 0, -m)`. Transforming into the camera frame via
`P_proj = R_cp^T (P_cam - t_cp)`, the same plane is

```
n_cam . (P_cam - t_cp) = 0,      n_cam = R_cp * n_p
```

Intersecting with the camera ray `P_cam = t*d` (`d = (dx,dy,1)`, pixel-center convention) and solving
for the ray parameter `t` (which equals depth `Z`, since `d`'s third component is exactly 1):

```
t = (n_cam . t_cp) / (n_cam . d)
```

closed form, no iteration, no matrix inversion — the "thousands of independent tiny calculations, one
per thread" pattern 33.01 teaches for small LINEAR SOLVES, specialized here to a problem simple enough
that even the solve itself collapses to one scalar division. **Numerics note:** this formula is
invariant to scaling `n_cam` by any nonzero constant (numerator and denominator scale together and
cancel) — so `n_cam` is deliberately NOT normalized in `kernels.cu`/`reference_cpu.cpp`: one fewer
`sqrt` per pixel, free, because the formula never needed it.

## The algorithm

Per-pixel pipeline, `n = kNPix = 30,000` camera pixels; per-sample for the boundary stress test,
`M = kBoundarySamples = 20,000`:

| Stage | Serial cost (CPU, per pixel/sample) | Parallel cost (GPU) |
|-------|--------------------------------------|----------------------|
| Gray decode | `O(N)` comparisons + `O(N)` XORs, `N=7` | `O(1)` (all `n` pixels in parallel; `O(N)` per thread) |
| Phase decode | `O(1)` (4 reads, `atan2`, `sqrt`) | `O(1)` |
| Hybrid combine | `O(1)` (one `round`, one compare) | `O(1)` |
| Triangulate | `O(1)` (a `3x3` matvec + one division) | `O(1)` |
| Boundary stress | `O(N)` per code (Gray AND binary), `N=7` | `O(1)` (all `M` samples in parallel) |

Every stage is `O(n)` (or `O(M)`) SERIAL total work, `O(1)` (or `O(max(N))`, a small constant)
PARALLEL depth — a textbook embarrassingly-parallel map, with the entire asymptotic story already told
by "one thread per pixel" (THEORY.md "The GPU mapping" below is about CONSTANTS — memory layout,
coalescing, bandwidth — not asymptotics, because there are none left to argue about).

The reconstruction GATES (plane fit, sphere fit, step height — `main.cu`) are the one genuinely
sequential-feeling piece: each is a small (`3x3` or `4x4`) normal-equations LEAST-SQUARES solve over
a few thousand ALREADY-TRIANGULATED points, done ONCE, on the host, via hand-rolled Gaussian
elimination with partial pivoting (double precision). "The GPU mapping" explains why this deliberately
stays off the GPU.

## The GPU mapping

**Thread-to-data mapping (every kernel but the last):** thread
`pix = blockIdx.x*blockDim.x + threadIdx.x` owns camera pixel `pix` (row-major,
`pix = row*kCamW + col`); grid `= ceil(n/256)`, the repo-default block size (a warp multiple with
good occupancy on `sm_75..sm_89` without starving the register file — the same geometry 08.01 and
33.01 use). `boundary_stress_kernel` maps identically over `M` synthetic samples instead of real
pixels.

**Memory hierarchy: global memory only, no shared memory, NO atomics anywhere.** Every kernel here is
a pure MAP: `output[pix]` is a function of `input[*, pix]` for THAT SAME `pix` and nothing else — no
thread ever reads or writes another thread's pixel. Shared memory earns its cost only when threads
REUSE or EXCHANGE data (neighboring-pixel stencils, block-wide reductions); nothing here does. Atomics
exist to arbitrate CONCURRENT writes to the same location; nothing here ever has two threads target
the same output element. Contrast with **01.18 (depth completion)**, whose scatter/gather fill-in
genuinely needs atomics because multiple SOURCE pixels can contribute to one DESTINATION pixel — this
project's clean per-pixel independence, top to bottom, is precisely why it teaches the "map" pattern
at its purest.

**The pattern dimension as the kernel's INNER loop (the layout argument).** Each pixel-owning thread
must read several PATTERN frames (7 Gray bit-pairs, or 4 phase steps) for its own pixel. The pattern
stack is stored FLAT and PATTERN-MAJOR: `pattern_stack[p*n + pix]` — every pattern is one contiguous
`[H x W]` image, patterns laid one after another (`kernels.cuh` "Pattern-stack memory layout"). At
iteration `p` of the inner pattern loop, EVERY thread in the warp is at the SAME `p` (the loop is
uniform, no divergence) and reads `pattern_stack[p*n + pix_of_that_thread]` — since a warp's 32
threads own 32 CONSECUTIVE pixels, this is one coalesced 128-byte transaction, repeated once per
pattern. The alternative layout (pixel-major: all of one pixel's patterns adjacent) would make that
SAME loop iteration scatter each thread's read `n` floats apart from its neighbor's — the exact
coalescing lesson 08.01's transposed MPPI noise array teaches (there: rollouts vs. horizon steps;
here: pixels vs. patterns — the same underlying principle, applied to images).

**Uniform vs. per-thread reads.** `kRcp`/`kTcp`/all scanner constants are compile-time `constexpr`
literals baked directly into the instruction stream by nvcc — not a runtime `__constant__` array — so
every thread's read of them costs nothing extra (no memory transaction at all; the values are
immediate operands). This sits at one end of the "same-address-read" spectrum the repo's other
flagships walk: 09.01's runtime `__constant__` broadcast, through 08.01's L2-cached uniform global
read, to 07.09's fully divergent per-thread global reads — this project's compile-time-constant case
is the cheapest point on that spectrum, because there is no runtime value to fetch at all.

**Why the reconstruction gates stay on the host.** Fitting a plane or sphere to a few thousand
ALREADY-TRIANGULATED points is a single `3x3`/`4x4` linear solve, done ONCE per gate. There is no
"thousands of independent small problems" here to hand to a thread grid — forcing this onto the GPU
would launch a kernel to do one iteration of Gaussian elimination and then synchronize, paying launch
overhead for work a single CPU core finishes in microseconds. Recognizing when a computation has NO
exploitable parallelism (as opposed to reflexively parallelizing everything) is as much a GPU-mapping
skill as recognizing when it does.

**Occupancy and bandwidth.** At `n=30,000` pixels, this pipeline moves at most a few hundred KB per
stage — far below what saturates an `sm_75`-class GPU's memory bus, and small enough that KERNEL
LAUNCH overhead (a few microseconds per launch, five-plus launches per run) is a comparable-magnitude
cost to the actual memory traffic. This is WHY the demo's own measured GPU-vs-CPU speed-up is modest
(a few times, not orders of magnitude, `[time]` lines) — an honest, expected result at this toy
problem size, not a red flag. A real 1-20 MP scanner frame (30-1300x more pixels) would move the same
kernels solidly into the bandwidth-bound regime where the mapping's real payoff shows up.

## Numerical considerations

- **Precision.** FP32 throughout the GPU/CPU pipeline (repo default); the reconstruction gates' small
  linear solves use FP64 on the host (Gaussian elimination accumulates round-off across a `3x3`/`4x4`
  system — cheap insurance at this tiny problem size, see `main.cu`'s `solve_dense`).
- **`atan2` conditioning at low modulation (the confidence floor's justification).** `phi =
  atan2(y,x)` with `y = I_1-I_3`, `x = I_0-I_2`, `r = sqrt(x^2+y^2) = 2B`. The angle of a
  near-origin vector is dominated by whatever noise perturbs `(x,y)`: for a fixed noise magnitude
  `sigma_n`, the induced angular error scales like `sigma_n / r = sigma_n / (2B)` — UNBOUNDED as
  `B -> 0`. This is precisely why `hybrid_combine_kernel`/`_cpu` mask pixels below
  `kDefaultConfidenceFloor` rather than trust an increasingly meaningless angle; `data/README.md` "How
  the sample was tuned" records the actual measured modulation-vs-albedo numbers that set the floor.
- **Bit-threshold margins (Gray decode).** `direct - inverse` is this project's per-bit "margin"; a
  small margin means the two captures nearly matched (an ambiguous bit). Because Gray code guarantees
  at most ONE bit is genuinely ambiguous at any true boundary (proved above), a small-margin bit is
  rare and isolated; plain binary coding can present MULTIPLE simultaneously-small margins at once
  (the MSB-boundary worst case) — the exact mechanism `boundary_stress_kernel`'s controlled blur+noise
  experiment measures (kernels.cu's kernel header derives the model in full).
- **A residual blind spot, honestly measured.** The phase-modulation confidence signal does NOT
  directly observe Gray-decode bit-margin risk — a pixel can have adequate phase modulation yet still
  carry a wrong Gray-decoded PERIOD if its bit thresholds were individually marginal (both signals
  correlate with albedo, but imperfectly). `data/README.md` documents the sweep that tuned
  `kDefaultConfidenceFloor` to drive this risk to zero on the committed sample, and README
  "Limitations" states plainly that this is a narrow, real gap, not a general guarantee.
- **Triangulation conditioning at small baseline-angle.** The denominator `n_cam . d = dx - m` (this
  project's parallel-axis rig) approaches zero when the camera ray's slope `dx` nearly EQUALS the
  projector plane's slope `m` — physically, when the camera and projector "agree" on a direction,
  which happens as the effective baseline angle subtended at the surface point shrinks (very far
  scenes, or points seen near-parallel to the baseline). `triangulate_kernel`/`_cpu` guard this with
  an explicit `|denom| < 1e-6` check (a genuine degenerate case, not just defensive programming) —
  the SAME ill-conditioning ordinary stereo triangulation exhibits at small parallax.
- **Determinism.** No `curand`, no platform RNG: `main.cu`'s boundary-stress noise is xorshift32
  (host, `08.01`'s generator), pre-drawn and uploaded as data so the GPU kernel and its CPU twin never
  need transcendental-function agreement to match bit-for-bit — a deliberate design choice (see
  `kernels.cuh`'s `boundary_stress_kernel` declaration comment) specifically to avoid the ULP drift
  that DOES appear (and is tolerated, with a documented tolerance) in the `atan2f`/`sqrtf`-based
  `phase_decode` stage.

## How we verify correctness

Two independent tiers, per the repo's twin-verification ruling (`reference_cpu.cpp`'s file header):

1. **GPU-vs-CPU agreement** (the `VERIFY:` lines) — every stage's independently-written CPU function
   (`reference_cpu.cpp`) is run on the SAME inputs as the GPU kernel and compared: exact integer
   match for Gray decode, hybrid valid-flags, and the boundary stress test (pure float add/compare,
   no transcendentals — genuinely bit-reproducible across host/device); tight-but-honest floating
   tolerances for phase decode (`1e-3` rad, `atan2f`/`sqrtf` ULP drift), hybrid columns (`2e-3`
   columns), and triangulated `xyz` (`2e-3` m).
2. **Ground-truth gates** (the `GATE:` lines) — comparisons against `scripts/make_synthetic.py`'s
   EXACT synthetic ground truth (continuous projector column, depth, surface identity), a THIRD,
   independent codebase (Python). Twin agreement alone cannot catch a bug shared by both C++
   implementations (a wrong sign, a swapped axis) — only a comparison against ground truth computed a
   different way can. `data/README.md` documents every gate's measured value and the margin behind
   its floor/bound.

**Edge cases specifically exercised:** pixels outside the projector's illuminated column range
(`kInvalidColumnF` sentinel, real on this scene's periphery — ~29% of pixels); the deliberately
low-albedo dark stripe (confidence correctly rejects ~99% of it); a genuine ray/plane near-degeneracy
guard in triangulation (exercised defensively, though absent on this particular scene's geometry); and
the boundary stress test's uniformly-sampled sweep over `[0, kProjCols)`, which by construction
includes both safe mid-cell probes and risky near-boundary ones in realistic proportion (not a
cherry-picked worst case).

## Where this sits in the real world

- **Industrial structured-light scanners** (Zivid, Photoneo MotionCam-3D, IDS Ensenso, LMI Gocator,
  Keyence) use the SAME family of ideas at far higher resolution and sophistication: multi-FREQUENCY
  phase shifting (several periods, not one, for unwrapping robustness beyond a single Gray-code
  assist), calibrated per-pixel lookup tables instead of a closed-form pinhole model, and dedicated
  FPGA/ASIC pattern generation and capture pipelines running at kHz pattern rates. `PRACTICE.md`
  covers the hardware/BOM story.
- **Kinect-v1-era spatial coding** (PrimeSense "Light Coding") trades this project's TEMPORAL
  redundancy (many frames, high accuracy, motion-sensitive) for SPATIAL redundancy: a single
  pseudorandom dot pattern, decoded by matching local NEIGHBORHOODS of dots (like tiny stereo
  patches) rather than by reading a per-pixel code across frames — single-shot, motion-tolerant, but
  lower accuracy and denser-scene-dependent (a locally repeated dot neighborhood is genuinely
  ambiguous). Named honestly here as the "why not single-shot" answer this project's own
  README "Limitations" raises, not implemented.
- **Event-camera structured light** is an active [R&D] research direction: replacing the frame-based
  camera with an event sensor (per-pixel asynchronous brightness-CHANGE reporting, microsecond
  latency) to decode a continuously-swept single-stripe pattern far faster than any frame-rate-limited
  temporal code could — genuinely promising for high-speed scanning, but immature compared to the
  frame-based methods this project teaches; named honestly as a frontier, not a scoped-down feature.
