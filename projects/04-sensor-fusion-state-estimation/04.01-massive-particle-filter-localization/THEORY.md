# 04.01 — Massive particle filter localization (10⁵–10⁶ particles, GPU likelihoods + resampling): Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

**Why a robot cannot just trust its wheels.** Every mobile robot has odometry: integrate wheel
rotation (encoder ticks × wheel radius) and, optionally, gyro rate, into a running estimate of
pose. Odometry is *dead reckoning* — no external reference — and every dead-reckoning system
accumulates unbounded error. The physical culprits are mundane and universal: wheel slip on smooth
or debris-covered floors (the wheel rotates without the robot translating the matching distance),
encoder quantization (a finite tick count per revolution rounds every tiny motion), tire/wheel
diameter drift with load and wear, and uneven floor contact. None of these are bugs — they are
what happens when you infer distance traveled from a proxy (shaft rotation) instead of measuring
position directly. Over a 12 s, 120-step drive like this project's sample, a few percent of
per-step velocity noise (this project injects σ=0.05 m/s and 0.05 rad/s at the *generator*, and the
filter assumes slightly more, σ=0.10/0.12, to be conservative) compounds into pose error that would
eventually exceed a corridor's width if nothing corrected it.

**Why a range sensor can.** A range sensor (LiDAR, sonar, or — as modeled here — an abstract
16-beam range fan that could represent a downsampled planar LiDAR or a ring of sonar/ToF units)
measures distance to nearby structure directly, typically by timing a pulse's round trip
(time-of-flight: `range = c·t/2`, `c ≈ 3×10⁸ m/s` for light, `≈ 343 m/s` for sound — the ranging
*physics* differs by sensor family, but the *geometry* consumed downstream is the same: a distance
along a known bearing). Crucially, a range measurement's error does **not** grow with elapsed
time — it depends only on the sensor's own noise floor (millimeters to centimeters for a good
LiDAR) and how well the surrounding structure matches the map. That is the whole reason this
project exists: **fuse a drifting-but-smooth signal (odometry) with a noisy-but-bounded one (range
scans) and get an estimate that is both smooth *and* bounded.** This project's beams are a
simplification — a real planar LiDAR sweeps continuously and returns hundreds to over a thousand
beams per revolution at 10–20 Hz (SYSTEM_DESIGN §1.1); 16 fixed bearings keep the ray-march and the
math on this page tractable while teaching the identical algorithm.

**The engineering frame.** A localization filter is a real-time system with a hard(ish) budget: it
must produce a fresh pose estimate every scan period (here 100 ms, 10 Hz) or the robot is
navigating on stale information. It must also be honest about *how sure* it is — a single point
estimate without uncertainty is unusable for safe planning, which is exactly what the particle
cloud's spread (and its effective-sample-size diagnostic, below) provides for free. Real sensors
add engineering headaches this teaching core sidesteps entirely: multipath reflections off glass
and mirrors, specular dropout on shiny floors, dynamic obstacles (people, forklifts) that are not
in the static map, and sensor mounting tolerances that must be calibrated before any of the math
below is trustworthy (PRACTICE.md §3).

## The math

**Notation (SI, right-handed, CLAUDE.md §12; matches `src/kernels.cuh` exactly).** World frame:
origin at the map's lower-left corner, x right, y up; pose `x_t = (p_x, p_y, θ)` — position in
meters, heading θ in radians CCW from +x, kept *unwrapped* in particle state. Control input (the
odometry twist) `u_t = (v, ω)` — linear velocity (m/s) and angular velocity (rad/s). Time step
`dt = 0.1 s`. Map `m`: an occupancy grid, cell `(ix, iy)` free (0) or occupied (1). Measurement
`z_t = (z_1, …, z_16)`: 16 ranges (m), beam `b` at body-relative bearing `−π + b·2π/16`.

**The recursive Bayes filter — the formal target.** Localization asks for the *belief*
`bel(x_t) = p(x_t | z_{1:t}, u_{1:t}, m)` — the full probability distribution over poses given
everything observed so far. It satisfies the two-step recursion:

```
 predict:  bel-bar(x_t) = ∫ p(x_t | x_{t-1}, u_t) · bel(x_{t-1}) dx_{t-1}
 update:   bel(x_t)      = η · p(z_t | x_t, m) · bel-bar(x_t)             (η normalizes to integrate to 1)
```

`p(x_t | x_{t-1}, u_t)` is the **motion model** (how a twist moves a pose, with uncertainty);
`p(z_t | x_t, m)` is the **measurement model** (how likely a scan is, given a candidate pose and
the map). Computing this integral in closed form is intractable for a nonlinear, non-Gaussian
belief over a 2-D-plus-heading manifold — which is exactly the gap a particle filter fills.

**The particle approximation.** Represent `bel(x_t)` by K weighted samples
`{(x_t^[k], w_t^[k])}, k=1..K`, drawn so that dense clusters of particles mark high-probability
poses. The predict/update recursion above becomes, per particle:

```
 predict (motion model, unicycle, Euler-integrated — the EXACT order src/kernels.cu implements):
   v_k = v + N(0, σ_v²)          w_k = ω + N(0, σ_ω²)      ← this particle's noisy belief of the twist
   p_x^[k] += v_k·cos(θ^[k])·dt + N(0, σ_xy²)               ← position advances with the OLD heading
   p_y^[k] += v_k·sin(θ^[k])·dt + N(0, σ_xy²)
   θ^[k]   += w_k·dt                                        ← heading turns AFTER the position step

 update (measurement model, independent-beam Gaussian range likelihood):
   p(z_t | x_t^[k], m) = Π_{b=1}^{16} N( z_b ; ẑ_b(x_t^[k], m), σ_z² )
   log w_t^[k] = log w_{t-1}^[k] − (1/2σ_z²) · Σ_b ( z_b − ẑ_b(x_t^[k], m) )²    (up to an additive
                                                                                   constant, dropped —
                                                                                   see "why log space" below)
```

`ẑ_b(x, m)` — the **expected range** — is not a formula but an *algorithm*: fixed-step ray-marching
from pose `x` along the beam's world bearing until the first occupied cell (or `kRMax = 8 m`; see
§The algorithm). This is what makes the measurement model a *simulator* rather than a closed-form
density, and why it runs on the GPU.

**Why log space.** Sixteen independent Gaussian factors multiply sixteen numbers that can each be
tiny (a poorly-matched beam contributes `exp(−(Δz)²/2σ_z²)`, which underflows float for `Δz` of a
few sigma). Summing the sixteen *log*-likelihoods instead is numerically safe over a much wider
range; the host exponentiates once, after subtracting the maximum log-weight across all K particles
(so the best particle maps to `exp(0)=1` and nothing overflows) — the same softmax hygiene project
08.01 uses for its softmin.

**Systematic resampling.** Draw one `u_0 ~ Uniform(0, 1/K]`; for `j = 0..K-1`, probe the weight CDF
at `u_0 + j/K` and clone the particle whose cumulative weight first exceeds the probe. This is a
*low-variance* resampler: every particle with weight above `1/K` is guaranteed at least one copy
(plain multinomial resampling cannot promise that), and it costs one random draw instead of K.

**Circular mean.** The pose estimate is the weighted mean `(Σ w_k p_x^[k], Σ w_k p_y^[k]) / Σw_k`
for position — but heading cannot be averaged the same way: θ and θ+2π are the same angle, so a
naive mean near the ±π seam is wrong. The fix is to average the unit vectors
`(Σ w_k sinθ^[k], Σ w_k cosθ^[k])` and take `atan2` of the result — the standard circular mean.

## The algorithm

The **bootstrap particle filter** (Sequential Importance Resampling, SIR), step by step — numbered
exactly as `main.cu`'s closed loop implements them:

1. **Initialize** (host): draw K particles from a Gaussian around the known start pose (pose
   *tracking* — README Exercise 4 replaces this with a uniform spread for global localization).
2. **Predict** (GPU, one kernel launch): every particle through the motion model above — O(1) work
   per particle, fully independent across k.
3. **Weight** (GPU, one kernel launch): every particle scored against the scan — O(16×≤64) ray-march
   steps per particle, fully independent across k.
4. **Normalize** (host): subtract the max log-weight, exponentiate, sum — O(K).
5. **Estimate** (host): weighted mean / circular mean — O(K).
6. **Resample** (host): systematic resampling — O(K).
7. Repeat from step 2 for the next scan.

**Complexity.** Serial cost per step: `O(K·(1 + B·S))` where `B=16` beams and `S≤64` ray-march
steps — dominated entirely by step 3 (measured: a single CPU core needs ~350 ms for K=100,000,
~3.7 s for K=1,000,000 — see below). Steps 2–3 are **embarrassingly parallel across k** (no
particle reads or writes another's state), so the natural GPU mapping is one thread per particle,
turning the `O(K·B·S)` serial loop into `O(B·S)` **wall-clock** work per particle, all K of them
paying it simultaneously (up to the GPU's own thread-level concurrency limits). Steps 4–6 stay
`O(K)` serial scalar work on the host — small enough (~40 lines, microseconds to a few ms) that
keeping them in plain C++ costs little and keeps the whole algorithm on one screen (README
Exercise 5 explores moving them to the GPU too).

## The GPU mapping

```
predict kernel                              weight kernel
one thread = one particle k                 one thread = one particle k
grid: ceil(K/256) blocks x 256 threads      grid: ceil(K/256) blocks x 256 threads

registers only: pose, twist noise (2         registers: pose (read once), direction/marching state
  Box-Muller pairs), no shared memory,          per beam (~30 registers/thread)
  no atomics — particles never interact       global reads:
global r/w: px[k], py[k], pth[k]               px/py/pth[k]   coalesced SoA read (one float each)
  (SoA layout -> each warp's access is           scan[b]        UNIFORM read, same address for every
  32 CONSECUTIVE floats, fully coalesced)                        thread -> broadcasts from L2/read-
                                                                   only cache (like 08.01's u_nom[t])
                                                map[iy*w+ix]   DIVERGENT reads (each particle's ray
                                                                   hits a wall after a different step
                                                                   count) but the WHOLE sample map is
                                                                   4 KiB -> after first touch, every
                                                                   lookup is an L1/L2 cache hit
                                              global write: logw[k]  one coalesced write at the end
```

**Divergence, honestly.** Lanes in a warp hold *different* particle poses, so their 16 rays hit
walls (or run out of steps) at different iteration counts; SIMT execution makes the whole warp pay
for its slowest lane, every beam. Near-converged clouds (most of this demo's 120 steps, once the
filter has locked on) have similar poses and similar ray lengths, so the tax is small; a freshly
scattered cloud (step 0, or Exercise 4's global-localization variant) pays more. Production
remedies — sorting particles by pose before the weight kernel (so nearby threads see similar ray
lengths) or replacing the ray-march with an `O(1)` likelihood-field lookup (below) — trade extra
host-side work or extra memory for less divergence.

**Cache behavior is the honest surprise here.** The catalog frames this project as
"memory-latency-soaked" work the GPU hides effortlessly, and at production scale (a building-sized
map at centimeter resolution, megabytes to gigabytes) that framing is exactly right — the map
cannot fit any on-chip cache and every ray-march step is a genuine DRAM round trip, hidden by
running thousands of other warps while any one warp waits. This project's *sample* map, however, is
only 64×64 cells = **4 KiB** — smaller than one SM's L1 cache. After the first few particles touch
it, the entire map lives in L1/L2 for the rest of the kernel, and the measured cost is closer to
compute/divergence-bound (branchy ray-marching) than DRAM-bandwidth-bound. That contrast is worth
sitting with: the *algorithm* does not change with map size, but which resource it is bound by
does — the opposite regime from project 07.09's large-grid distance-field kernels, worth comparing
directly.

**No CUDA libraries used.** Both kernels are hand-rolled — no cuBLAS/cuFFT/Thrust/CUB — because the
per-particle work (a handful of trig calls and a ray-march loop) has no natural batched-library
shape; the "library" this project *would* reach for in production is a precomputed likelihood-field
lookup table (built once with a distance transform — project 07.09's jump-flooding SDF is exactly
that data structure), turning the O(64)-step ray-march into an O(1) table read. THEORY.md's closing
section names this explicitly.

## Numerical considerations

- **FP32 with a deliberate FP64 island.** Particle state, noise, and the ray-march all run in
  float32 (matching CLAUDE.md §12's default and keeping the GPU's native throughput). Beam
  *bearings* — `θ + (−π + b·2π/16)` — are computed in **double**, then cast to float once
  (`kernels.cu`'s `raycast_range_dev`): `|cos_double − cos_msvc_float|` is on the order of double
  ulps (~1e-16), far below the float cast's own rounding step (~6e-8), so the direction vector is
  identical on CPU and GPU essentially always — cheap insurance against a subtle host/device
  mismatch in a place where mismatch would be expensive to debug.
- **Contraction safety in a discontinuous function.** The ray-march advances the marching position
  by *running additions of pre-scaled steps* (`sx += step_x`, never the fused `x0 + i·step_x`), and
  converts world position to a cell index with a *lone* multiply-then-floor. Neither expression has
  the `a*b+c` shape nvcc's FMA-contraction pass looks for, so the GPU and CPU visit **bit-identical**
  map cells given bit-identical inputs. This matters *specifically* here because
  `raycast_range_dev`/`_host` is discontinuous in its inputs — one flipped cell index near a
  corner can change the returned range by meters, not ulps — unlike a smooth accumulation (e.g. the
  predict kernel's pose update, or the weight kernel's final `sq_sum` accumulation), where nvcc's
  FMA reordering only shifts the last bit or two and a *tolerance* comparison is the right tool.
  This project therefore uses two different verification strategies for two different kinds of
  arithmetic — see "How we verify correctness" below.
- **Determinism via a counter-based RNG, not cuRAND.** Every particle's noise draw is a pure
  function of `(kBaseSeed, particle id k, step t)`: an in-kernel xorshift32 (Marsaglia's 3-shift
  generator — full 2³²−1 period, three integer ops, bit-identical across every compiler and device)
  seeded by mixing k and t with two large odd multipliers, then two warm-up rounds to decorrelate
  similar seeds. No cuRAND state array, no per-thread persistent state — call it, get the same
  answer, forever, on CPU or GPU (README Exercise 3 swaps in cuRAND Philox to compare quality and
  cost).
- **Weight degeneracy and effective sample size (the classic particle-filter failure mode).** After
  repeated resampling, weight mass concentrates onto fewer and fewer *distinct* ancestors — even
  though the array still holds K particles, only a handful may trace back to genuinely different
  trajectories. The standard diagnostic is **effective sample size**,
  `ESS = (Σw_k)² / Σw_k²` (ranges from 1, total collapse, to K, perfectly uniform weights).
  `main.cu` computes it every step; on the reference run at K=100,000 it ranged from a minimum of
  1,376 to a mean of 38,016 — i.e., even the *worst* step still carried over a thousand effectively
  independent particles, and resampling every step (rather than only when ESS drops, the common
  "adaptive resampling" refinement) keeps that number from decaying further. Two defenses are baked
  into the model constants: **σ_z inflation** (the filter assumes `σ_z = 0.15 m`, deliberately
  larger than the generator's actual `σ_z = 0.10 m` noise — the slack absorbs ray-march
  quantization and keeps weights from collapsing to numerical zero) and **roughening**
  (`σ_xy = 0.01 m` additive positional noise every predict step, so resampled clones do not stay
  exact copies — the classic defense against *sample impoverishment*, where the cloud loses
  diversity after repeated cloning).
- **Angle wrapping — one defined point.** Particle headings stay **unwrapped** for the whole filter
  run (only `sin`/`cos` of θ are ever consumed, and those are periodic — wrapping would only risk
  a discontinuity for no benefit). The single wrap point in the entire project is `main.cu`'s
  `wrap_angle` helper, applied only when turning an estimate-minus-ground-truth heading *difference*
  into an error number for the RMSE report (CLAUDE.md §12's angle-wrapping discipline, applied
  literally).

## How we verify correctness

Two tolerances for two different kinds of arithmetic, applied at two different stages (`main.cu`'s
VERIFY block):

1. **Predict — absolute tolerance 1e-4 (m or rad).** The pose update is a *smooth* function of its
   inputs (trig, multiply, add) — GPU and CPU results differ only by libm/FMA-contraction ulps,
   which is on the order of 1e-6 at these magnitudes. Measured worst-case deviation over 100,000
   particles: **~4.8e-7** — roughly 200× inside the 1e-4 gate, while any real seeding/model/layout
   bug would move a pose by order 0.1 (a whole noise sigma), immediately visible.
2. **Weight — relative tolerance 1e-3 (floor 1.0), fed IDENTICAL poses.** The weight kernel and its
   CPU twin are run on the *same*, GPU-predicted poses (not each side's own predict output) so the
   comparison isolates the weight kernel alone — because `raycast_range_*` is discontinuous, feeding
   the twins two *merely close* poses (one from GPU predict, one from CPU predict) could make them
   visit different cells and diverge at order 1 for reasons that have nothing to do with a weight-kernel
   bug. On identical poses, the contraction-safe ray-march visits bit-identical cells, so the only
   remaining difference is the final sum-of-squares accumulation's rounding (~1e-6 relative).
   Measured worst-case: **~2.4e-7** at K=100,000 (and ~2.5e-7 at K=1,000,000) — again far inside the
   1e-3 gate, while an indexing, layout, or sensor-model bug moves log-likelihoods at order 1.

Both checks together catch what a single end-to-end diff would miss: chaining GPU-predict →
CPU-weight (or vice versa) through a *discontinuous* ray-march would smear a false failure from
ordinary ulp noise into an apparent order-1 mismatch. Testing kernel-by-kernel, on shared inputs,
is what makes a tight tolerance possible at all.

**The closed-loop success check is a different kind of test.** VERIFY proves the kernels are
numerically right; it says nothing about whether the *filter* actually tracks the robot (a
correctly-computed weight kernel cannot save a filter with a broken resampler, a badly-tuned
`σ_z`, or a motion model that disagrees with reality). `main.cu` therefore also runs the full
120-step closed loop and requires position RMSE against the (synthetic, exactly-known) ground truth
to beat `kRmseGateM = 0.15 m` — measured **~0.019 m** at K=100,000 and **~0.019 m** at K=1,000,000
(heading RMSE ~0.004 rad both). The gate carries roughly 8× margin over the measured value
*deliberately*: this is a stochastic algorithm, and while every random draw is seeded and
reproducible *on one machine*, host `std::log`/`std::cos` and device `log`/`cos` can differ in a
low bit across platforms — and because resampling is a discontinuous selection (a particle either
survives a CDF probe or does not), even a single flipped low bit can chaotically alter *which*
individual particles survive a given step. The RMSE is an aggregate over 100,000 particles × 120
steps, which is statistically robust to that chaos even though any single particle's trajectory is
not — which is exactly why the stable output line is a thresholded RMSE with wide margin, and no
raw trajectory numbers appear in the checked output contract.

## Where this sits in the real world

- **`nav2_amcl`** (ROS 2 Nav2's Adaptive Monte Carlo Localization) is the direct production
  descendant most warehouse AMRs run today — same bootstrap-filter shape, but on CPU, with
  **KLD-sampling** (adapting particle count per step based on how spread the cloud is — typically
  hundreds to a few thousand particles, not this project's 10⁵–10⁶) and a **likelihood-field**
  sensor model instead of raw ray-marching: precompute, once per map, the distance from every cell
  to the nearest occupied cell (a distance transform — project 07.09's jump-flooding SDF builds
  exactly this structure), then score a beam's endpoint by looking up that precomputed distance —
  `O(1)` per beam instead of this project's `O(≤64)` ray-march steps, and no per-thread ray-march
  divergence at all. That single change is the most impactful thing separating this teaching core
  from a production sensor model.
- **Scan-matching (ICP/NDT, project 02.06)** increasingly does the heavy lifting with modern
  high-resolution LiDAR: it converges from a good prior in far fewer "samples"/iterations than a
  particle filter needs, and handles the higher-dimensional 3-D pose autonomous vehicles need.
  Production stacks often run both — scan-matching for continuous tracking, a particle filter (or
  its descendants) kept in reserve for *global* relocalization (the kidnapped-robot problem,
  README Exercise 4) where scan-matching's local optimization can get stuck.
- **GPU-accelerated MCL research** exists precisely because this project's core observation —
  the weight kernel is embarrassingly parallel and a GPU affords 10–1000× more particles than a CPU
  core for the same wall-clock budget — is real and useful: bigger clouds recover faster from bad
  initializations, degenerate corridors (many poses look alike), and multi-modal ambiguity that a
  KLD-sampled CPU filter with a few hundred particles cannot represent.
- **What the full research version needs beyond this teaching core:** a likelihood-field sensor
  model (above), adaptive particle count (KLD-sampling), GPU-resident resampling (README
  Exercise 5 — eliminating the host round-trip this version accepts for readability), multi-sensor
  fusion (wheel odometry + IMU + range, not odometry + range alone), and — for the full "SLAM"
  generalization where the map is *not* known in advance — a factor-graph or FastSLAM-style
  architecture (GTSAM; domain 05) that estimates the map alongside the pose.
