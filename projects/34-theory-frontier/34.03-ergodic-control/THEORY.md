# 34.03 — Ergodic control: spectral multiscale coverage (FFT-based — very GPU-friendly): Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

**Why does uniform coverage fail?** Picture a drone surveying a field for crop stress. A raster
(lawnmower) scan visits every square meter for an equal amount of time — simple, predictable,
easy to verify. But if 80% of the field's *useful information* lives in two stressed patches
covering 15% of the area, the raster scan spends 85% of its flight time on the boring 85% of the
ground. This is not a bug in the raster scan — it is *by construction*: a raster scan has no idea
where the information is, because it never looks at a target density at all. The fix sounds
obvious — "spend more time where it matters" — but making that rigorous requires an actual
mathematical object for "where it matters" (a **target density** `phi(x)`) and a controllable
notion of "spend time in proportion to" (this is the theoretical content of the rest of this
document).

**A time-allocation argument.** Suppose a robot has a fixed mission time `T` to explore a region.
Its trajectory `x(t)` for `t in [0,T]` induces an **empirical time-averaged density**:

```
C(x) = (1/T) * integral_0^T delta(x - x(t)) dt
```

(a Dirac-comb over the visited points, averaged over time — informally, "the fraction of `T` spent
near each point `x`"). A raster scan's `C` is close to *uniform* over the swept region, regardless
of `phi`. **Ergodic** trajectories, by contrast, are ones whose `C` converges to (matches the
statistics of) a target `phi` as `T -> infinity`. The word borrows its meaning from dynamical
systems theory: an ergodic dynamical system's time averages equal its space (ensemble) averages.
Here we *design* the dynamics (the control law) so this holds, on purpose, for a chosen `phi` —
the opposite of a system that merely happens to be ergodic.

**The engineering frame.** A coverage controller is a **guidance-layer** component (SYSTEM_DESIGN
item 1): it does not touch actuators directly, and it does not need millisecond reaction time — the
statistic it tracks (a spatial occupancy average) only changes meaningfully over seconds to
minutes. What it *does* need: (a) a target density `phi(x)` from somewhere (an information-gain
front end — README names 05.15/23.09), (b) a state estimate, and (c) enough compute per control
tick to update its internal bookkeeping (here: 1,024 spectral coefficients) fast enough to stay
inside the 10–100 Hz guidance band. That compute budget — small, parallel, and repeated every tick
— is exactly what this project's GPU kernel is shaped around (see §The GPU mapping).

## The math

**Setup.** Domain `Omega = [0,1]^2` (a normalized, unitless workspace — PRACTICE.md §2 discusses
scaling to real meters). Agent state `x = (x1,x2) in Omega`, first-order dynamics `xdot = u`,
`||u|| <= v_max` (a pure **speed budget** — the controller's only freedom is *direction*).

**The basis.** We compare distributions by their Fourier coefficients with respect to an
**orthonormal cosine basis** on `Omega`:

```
f_k(x) = (1/h_k) * cos(k1*pi*x1) * cos(k2*pi*x2),   k = (k1,k2), k1,k2 = 0..K-1
h_k     = sqrt( integral_Omega f_k_raw(x)^2 dx )    (the L2 normalizer)
```

**Why cosines, not sines or a complex Fourier basis?** `d/dx cos(k*pi*x) = -k*pi*sin(k*pi*x)`,
which is **zero at x=0 and x=1** for every integer `k`. This is the **Neumann ("no-flux")
boundary condition** — physically, "nothing crosses the wall." A workspace an agent cannot leave
is exactly a no-flux domain, so the cosine basis is the boundary-condition-correct choice, the
same way a vibrating string with free (not clamped) ends is naturally expanded in cosines, not
sines. It is also why this project's agent **reflects** off the domain walls
(`integrate_agent_cpu` in `reference_cpu.cpp`) rather than clamping or wrapping: reflection is the
trajectory behavior consistent with a no-flux boundary; clamping (sticking) or wrapping
(teleporting) would silently contradict the basis the whole rest of the algorithm assumes.

`h_k` works out to `1` (both `k_i=0`), `1/sqrt(2)` (exactly one `k_i=0`), or `1/2` (neither zero) —
derived from `integral_0^1 cos(k*pi*t)^2 dt` being `1` at `k=0` and `1/2` at `k>0` (kernels.cu
derives this in the code comments beside `basis_norm_h_dev`).

**Fourier coefficients.** The target's:

```
phi_k = integral_Omega phi(x) * f_k(x) dx
```

The trajectory's **running time-average**, sampled at the controller's own tick rate (dt=0.01 s,
so "time average" and "sample average" coincide for a uniformly-sampled trajectory):

```
c_k(t_n) = (1/n) * sum_{i=1}^{n} f_k(x(t_i))
```

**The ergodic metric** — the single scalar this whole project drives toward zero:

```
epsilon(t) = sum_k Lambda_k * (c_k(t) - phi_k)^2,     Lambda_k = (1 + ||k||^2)^(-s),  s = (d+1)/2
```

with `d=2` spatial dimensions, so `s=1.5`. `||k||^2 = k1^2+k2^2` (this project uses the raw
integer mode-index norm — a documented scaling convention; some references instead use the
angular wavenumber `(pi*k1)^2+(pi*k2)^2`, which only rescales `Lambda_k` by a constant per mode and
does not change which trajectory minimizes `epsilon`).

**Why the `s=(d+1)/2` exponent — smoothness duality.** `Lambda_k` decaying with `||k||` means
**high-frequency (fine-detail) mismatches count less than low-frequency (coarse-shape) ones**. The
reason this specific exponent is the "correct" one (not just a reasonable heuristic) comes from
Sobolev-space duality: if `phi` lives in the Sobolev space `H^s(Omega)` (roughly, "has `s`
bounded derivatives on average"), its Fourier coefficients decay at least as fast as
`||k||^(-s)`; comparing `c_k` and `phi_k` in the *dual* space `H^{-s}` (i.e., weighting by
`Lambda_k = (1+||k||^2)^{-s}`) makes `epsilon` a well-defined, finite metric even though `C(x)`
itself (a sum of point masses along a trajectory) is not a function that lives in any `L2`-type
space — it is a measure, and only a *sufficiently negative*-index Sobolev norm can see it. This is
exactly the technical device Mathew & Mezic use: pick `s` just large enough (`(d+1)/2`) that a
one-dimensional-curve-mass-distribution (which "smears" `d`-1 dimensions thin) is still comparable
to a full-dimensional density in that weighted sense.

**The SMC control law.** Differentiating `epsilon` along a candidate direction `u` and taking the
steepest-descent direction (a calculus-of-variations argument over the trajectory, not repeated
here — Mathew & Mezic's paper owns the full derivation) gives:

```
B(x,t) = sum_k Lambda_k * (c_k(t) - phi_k) * grad_x f_k(x)
u(t)   = -v_max * B / ||B||
```

`grad_x f_k` is closed-form (no numerical differentiation needed):

```
d f_k/dx1 = -k1*pi*sin(k1*pi*x1)*cos(k2*pi*x2) / h_k
d f_k/dx2 = -cos(k1*pi*x1)*k2*pi*sin(k2*pi*x2) / h_k
```

This is a **bang-bang** law: the agent always moves at the full speed budget `v_max`, and the
*controller's entire freedom* is the heading `B/||B||`. There is no path planning, no lookahead, no
optimization horizon — every control tick is a single gradient evaluation. This is what makes SMC
cheap enough to run in real time even in a purely-sequential (non-GPU) implementation at small `K`
— and what makes it embarrassingly parallel across modes, which is this project's GPU story.

## The algorithm

Two phases, matching `kernels.cuh`'s two GPU launchers:

**Phase 1 — build `phi_k` (once).** Evaluate `phi` on an `(N x N)` trapezoidal grid
(`N=kPhiGridN=129`), then compute all `K*K=1,024` coefficients via the **DCT-I-via-FFT identity**
(the full derivation is in `kernels.cu`'s file header; summary below). Cost: **O(M log M)** for an
`M x M = 256 x 256` FFT (M = 2*(N-1)), versus **O(N^2*K)** for the direct trapezoidal sum this
project's CPU oracle uses as an independent check (a further, deliberate simplification versus the
naive O(N^2*K^2), achieved by precomputing cosine tables once — see `reference_cpu.cpp`).

**Phase 2 — the closed loop (6,000 steps).** Each step, for **every mode independently**:
update the running sum `S_k += f_k(x)`, form `c_k = S_k/n`, and compute this mode's contribution
`B_k` to the descent direction. Then, **on the host**: sum `B = sum_k B_k` (O(K) = O(1,024)
scalar adds), normalize to get `u`, and Euler-integrate the agent one `dt=0.01 s` step (with wall
reflection). Complexity per step: **O(K)** GPU work (parallel across `K`) + **O(K)** host reduction
— genuinely cheap, the reason SMC belongs to the guidance-layer 10–100 Hz band rather than needing
a dedicated compute budget the way MPPI's thousands-of-rollouts-per-tick controller does (08.01).

**The DCT-I-via-FFT identity, in brief** (see `kernels.cu` for the fully worked derivation with
every algebraic step): the 1-D trapezoidal integral of `x(t)*cos(k*pi*t)` over a grid with `N`
points (endpoints included, spacing `h=1/(N-1)`) equals `(h/2)` times the **DCT-I** coefficient of
`x`, and the DCT-I of a length-`N` sequence is *exactly* the FFT of its length-`M=2(N-1)`
even-symmetric ("mirrored through both endpoints") periodic extension — a classical, textbook fact
(the same family of identities Numerical Recipes and Makhoul (1980) catalog for fast cosine
transforms). Mirroring through the endpoints (rather than through cell centers, the DCT-II
convention) means the resulting FFT output is **already real** — no phase-correction multiply is
needed, the simplest member of the DCT-via-FFT family and the reason this project uses it. The 2-D
case is the tensor product of two 1-D cases, computed with **one** 2-D FFT of the doubly-mirrored
array.

## The GPU mapping

```
PHASE 1 — build_phi_k (ONCE):
  build_even_extension_kernel: 65,536 threads (256x256), 1 thread/output cell, pure index remap
  cufftPlan2d + cufftExecZ2Z:  ONE 2-D, 256x256, double-complex, in-place forward transform
  extract_phi_k_kernel:        1,024 threads, 1 thread/mode, rescale + orthonormalize

PHASE 2 — smc_step (x6,000 steps):
  smc_step_kernel: 1,024 threads, 1 thread/mode k=(k1,k2)
    reads:  x1,x2 (KERNEL PARAMETERS — broadcast via constant/parameter memory,
                   no device buffer/upload needed, unlike 08.01's 4-float state)
            phi_k[idx], S[idx]           (coalesced: idx is the natural offset)
    writes: S[idx] (persists), c[idx], Bx[idx], By[idx]   (coalesced)
    no shared memory, no atomics — every mode is fully independent
  HOST: Bx = sum(Bx[]), By = sum(By[])   O(K) scalar reduction, plain C++
```

**Memory hierarchy.** Every array in this project (`phi_k`, `S`, `c`, `Bx`, `By`) is small enough
(`8 KB` at `K=1,024` doubles) to live entirely in L2 cache across a kernel launch — there is no
tiling, no shared-memory staging, because there is nothing to *reuse*: each thread touches exactly
its own mode's data once. This is the same "nothing to share -> no shared memory" reasoning as
08.01's MPPI rollout kernel, one level further: MPPI's rollouts are big (a whole simulated second
each); this project's "rollouts" (mode updates) are a handful of trig calls each.

**Why pass `x1,x2` by value instead of a device pointer (contrast with 08.01).** MPPI's 4-float
cart-pole state needed a device buffer because CUDA's C-linkage kernel-launch syntax does not make
passing a runtime-sized array by value idiomatic; two independent `double` scalars have no such
restriction — they ride in the kernel's parameter/constant memory, broadcast to every thread at
essentially zero cost, and the whole per-tick `cudaMalloc`/`cudaMemcpy`/`cudaFree` MPPI's launcher
pays for its state upload simply does not exist here. A smaller state, a simpler mapping.

**Why cuFFT and not a hand-rolled DCT (CLAUDE.md §6.1 rule 6).** A correct, numerically robust FFT
— even a small, fixed-size 256-point 2-D one — is a multi-week project in its own right (see this
repo's 33.x-style foundational-library projects for what that actually entails). This project's
subject is ergodic *control*, not FFT internals, so `cufftPlan2d`/`cufftExecZ2Z` (both explained in
full in `kernels.cu`'s comments — what each computes, what shape of data it expects, why Z2Z
double-precision) does the work, and `reference_cpu.cpp`'s `phi_k_direct_cpu` proves the result
independently, with **zero FFT code anywhere** — the transform is never a black box even though it
is not reimplemented.

**Honesty about where the GPU pays off at THIS scale.** At `K=1,024` and a single agent, none of
this project's kernels come close to saturating an RTX 2080 SUPER's 46 SMs — the closed loop's
measured per-step kernel time (~0.04–0.1 ms, see README §Expected output) is dominated by launch
overhead and the H2D/D2H round trips of a handful of kilobytes, not compute. The pattern is the
point: **N independent agents × K^2 modes** is an `N*K^2`-thread launch with the *identical* kernel
body (each thread still owns exactly one `(agent, mode)` pair, still touches only its own `S`/`c`
entry) — at `N=100` agents this is already a 100K-thread launch, comfortably GPU-shaped. This
project's single-agent instance is the smallest case of a pattern that scales exactly the way the
full research version (THEORY.md §Where this sits in the real world) needs it to.

## Numerical considerations

- **Double precision throughout, and why it costs nothing here.** `phi_k`, `S`, `c`, `Bx`, `By`,
  and the cuFFT plan itself (`CUFFT_Z2Z`) are all `double`. At `K=1,024` threads and a one-time
  `256x256` transform, the throughput difference between FP32 and FP64 on any sm_75+ GPU (which
  supports native double math, just at lower peak throughput than float) is unmeasurable next to
  the kernel-launch and PCIe-transfer overhead that actually dominates this project's per-step
  cost (see §GPU mapping's honesty note). Choosing double removes an entire class of "is this
  disagreement a real bug or just float32 running-sum drift" argument from the TRANSFORM and
  VERIFY gates — the measured GPU-vs-CPU deviations (1.4e-11 and 8.3e-12, README §Expected output)
  are then attributable almost entirely to **algorithmic** differences (summation order, cuFFT's
  internal Cooley-Tukey factorization vs. a direct sum) rather than precision, which is exactly
  what those gates are meant to isolate.
- **Determinism.** Nothing in this project is randomized — no RNG, anywhere, on either the GPU or
  CPU path. A run on a fixed machine reproduces bit-for-bit. Across *different* GPU architectures,
  cuFFT's internal factorization and the device math library's transcendental-function rounding can
  differ in the last few bits (a different, smaller source of the same "float ops are not
  associative across implementations" fact 08.01/03.01 document for their own noise-driven demos)
  — which is why this project's stable output lines are PASS/FAIL verdicts, never the raw measured
  epsilon/coverage numbers (README §Expected output states the measured numbers separately, as
  documentation, not as a diffed contract).
- **Bang-bang chattering near ergodicity.** As `epsilon -> 0`, `B -> 0`, and the *direction*
  `B/||B||` becomes numerically ill-conditioned (dividing a near-zero vector by its own near-zero
  norm). `kBEps=1e-9` in the denominator prevents a literal `0/0`, but does not remove the
  underlying phenomenon: near-ergodic states can produce rapidly alternating control directions
  (measured in this run: the ERGODICITY gate's one windowed uptick, +16.9%, is this effect visible
  in the metric curve — a real, expected artifact of the *law*, not a bug in this
  implementation). Miller & Murphey's trajectory-optimization reformulation (README §Prior art)
  smooths this by optimizing over a horizon instead of reacting greedily every tick.
- **Wall reflection, single overshoot.** `kVmax*kDt = 0.4*0.01 = 0.004`, two orders of magnitude
  below the unit domain — a position can overshoot a wall by at most that much per step, so
  `integrate_agent_cpu`'s single reflect-per-axis-per-step is exact at this step size (the trailing
  clamp is defensive, not load-bearing, and is never exercised at these parameters).
- **Grid resolution vs. quadrature error.** `kPhiGridN=129` (128 cells) resolves the narrowest
  Gaussian hotspot (`sigma=0.07`, ~9 grid cells across its `1-sigma` width) comfortably — the
  trapezoidal quadrature error this introduces into `phi_k` is far below the `1e-6` TRANSFORM
  tolerance (confirmed by the gate's own measured 1.4e-11 deviation, which reflects FFT-vs-direct-
  sum algorithmic agreement, not quadrature truncation — both paths use the identical grid).

## How we verify correctness

Four independent checks, because a spectral controller can be *numerically right and behaviorally
wrong* (or vice versa) — mirroring 08.01's two-tier philosophy, extended to cover the extra
FFT-specific correctness question this project introduces:

1. **TRANSFORM (transform correctness).** `phi_k` computed via the GPU DCT-via-cuFFT pipeline vs.
   `phi_k_direct_cpu` — a *completely independent* code path (no FFT, precomputed-cosine-table
   direct trapezoidal sum) computing the *same* mathematical quantity. Agreement to `1e-6` relative
   tolerance (measured: `1.4e-11`) is strong evidence the mirror/FFT/extract pipeline in
   `kernels.cu` implements the DCT-I identity correctly — an indexing or scale-factor bug would
   show up as an O(1) disagreement, not a rounding-level one.
2. **VERIFY (the §5 GPU-vs-CPU gate).** The per-mode SMC update (`smc_step_kernel` vs.
   `smc_step_cpu`) driven through an identical 50-step position sequence, comparing `c_k`/`Bx`/`By`
   for all 1,024 modes at every step (rel tol `1e-6`, measured `8.3e-12`). Catches indexing, launch
   configuration, or running-sum-state bugs instantly (any such bug shifts a mode's value at O(1),
   not O(1e-10)).
3. **ERGODICITY.** The behavioral claim the whole algorithm exists to deliver: does the metric
   actually go down? Checked via window-averaged decrease (≥5x, measured 116.5x) plus a windowed-
   monotonicity bound that tolerates the bang-bang law's known transient upticks (§numerics) without
   accepting an unbounded or sustained climb.
4. **COVERAGE + NEGATIVE-CONTROL.** The definition of ergodic coverage made checkable: does the
   *fraction of time* spent near each hotspot approach that hotspot's *actual probability mass*
   (numerically integrated from the same grid `phi_k` was built from — never hand-typed)? And does a
   deliberately naive baseline (lawnmower, ignoring `phi`) do measurably worse? Together these prove
   the controller is not merely "dense" (visiting everywhere a lot) but *ergodic in particular* — the
   negative control is what rules out "any sufficiently wiggly path would pass."

The scenario (start position, step count) is committed under `data/sample/` so every check runs
offline, deterministically, with no downloads (CLAUDE.md §8).

## Where this sits in the real world

- **Mathew & Mezic (2011)** established SMC as presented here — the ergodic metric, the Sobolev
  weighting, and the steepest-descent bang-bang law. This project implements their algorithm at
  reduced scale (single agent, fixed K, first-order dynamics) as a teaching core.
- **Miller & Murphey and successors** reformulate ergodic coverage as a **receding-horizon
  trajectory-optimization problem** (closer in spirit to 08.01's MPPI than to this project's greedy
  law) — trading SMC's per-tick simplicity for smoother, less chattery trajectories and the ability
  to incorporate constraints (obstacles, dynamics limits) directly into the optimization.
- **What the full research version needs beyond this teaching core:**
  - **Multi-agent ergodic coverage** — `N` agents sharing one target `phi_k` but with `N` independent
    running sums `S_k^{(i)}`, and a *joint* empirical density `C = (1/N) sum_i C^{(i)}`; an open
    design question is how to divide exploration credit so agents do not redundantly chase the same
    hotspot (this repo's 22.x swarm domain is the natural neighbor).
  - **Second-order (double-integrator) dynamics** — real vehicles have momentum; the bang-bang
    control law derived here for `xdot=u` does not directly generalize to `xddot=u`, and the smoother
    trajectory-optimization formulations above are the usual route taken instead.
  - **Obstacles and dynamic environments** — this project's domain is empty and static; real
    coverage missions need the ergodic metric traded off against a collision-avoidance cost (a
    natural fit for 07.x's distance fields feeding an additional cost term).
  - **Adaptive / streaming mode truncation** — this project fixes K=32×32 for the whole run; a
    genuinely adaptive controller would refine K (add modes) as the mission progresses and the
    target `phi` is refined by an online information-gain estimator (05.15/23.09), an active
    research question in how to do this without discarding the running `c_k` state already
    accumulated at the old K.
  - **Learned or time-varying information maps** — `phi` here is fixed for the whole run; a live
    exploration mission would update `phi` as the robot learns more, requiring `phi_k` (and the
    metric's target) to be recomputed online — a natural fit for this project's already-fast
    (single-cuFFT-call) `launch_build_phi_k`, but the *control-theoretic* question of how a moving
    target changes the SMC guarantees is, itself, still an open research question.
