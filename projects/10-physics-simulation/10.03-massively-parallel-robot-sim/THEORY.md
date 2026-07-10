# 10.03 — Massively parallel robot sim (Isaac-Gym-style: one robot, 10,000 environments): Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2).

## The problem — physics & engineering first

**Why parallel simulation exists at all.** Three real needs in robotics share one answer:

1. **Reinforcement learning is sample-hungry.** A modern locomotion policy (the kind that lets a
   quadruped walk on rubble) trains on the order of **billions** of environment-steps before it is any
   good — PPO-style algorithms need thousands of noisy trials to estimate a useful gradient. One
   physical robot generates experience at wall-clock speed: 1 second of robot life costs 1 second, and
   you have exactly one robot. **Isaac Gym's actual contribution** (Makoviychuk et al., 2021) was
   showing that a single GPU can step **thousands of physics instances in parallel**, entirely
   resident in GPU memory, at throughputs that make months of simulated experience possible in hours.
2. **Controller tuning needs many trials, not one.** Even a "classical" gain-scheduled controller
   (this project's fixed pole-placement gains, §The math below) should be validated against
   uncertainty, not just the nominal plant — you want to know if it works for the *whole family* of
   robots your factory will ship, not the one CAD model.
3. **Monte Carlo validation needs a distribution of outcomes.** "Does this policy fail more than 1% of
   the time under realistic manufacturing tolerance?" is a question about a DISTRIBUTION, and you
   cannot estimate a distribution's tail from one sample.

All three needs reduce to the same computational shape: **run the same dynamics model many times,
each with a slightly different randomized instance, and look at the aggregate/worst-case result.**
That shape — independent, embarrassingly parallel, identical-code-different-data — is exactly what a
GPU is built for (the 33.01/09.01 lesson, reused at simulation scale instead of kernel scale).

**The plant, again.** This project reuses [08.01](../../08-control-systems/08.01-mppi-controller-the-canonical-gpu-controller/THEORY.md)'s
force-limited cart-pole verbatim — a cart of mass `m_c` on a frictionless track, a uniform pole of
mass `m_p` and half-length `l` pivoting freely on it, state `x = [p, ṗ, θ, θ̇]` (θ = 0 upright). 08.01
derives the equations of motion from the Lagrangian; this project does not re-derive them (see
[`src/kernels.cuh`](src/kernels.cuh)'s `cartpole_deriv`, a byte-for-byte match of 08.01's formulas,
generalized so `m_c`, `m_p`, `l` are **per-environment runtime values** instead of shared compile-time
constants — the one change that makes domain randomization possible).

**The engineering frame — what changes when the plant is a FARM, not a controller.** 08.01's MPPI
controller drives ONE plant and must fit inside a 20 ms wall-clock tick, forever — its engineering
constraint is *latency*. This project's engineering constraint is *throughput*: nobody is waiting on
any single environment's next tick in real time; what matters is how many environment-ticks the whole
farm completes per wall-clock second, because that number sets how many days of simulated robot
experience you can generate before lunch. This is also why the farm's controller can be almost
laughably simple (§The algorithm) — the interesting engineering problem here is not "control this one
robot well", it is "run ten thousand of them without the memory layout or the reset bookkeeping
becoming the bottleneck instead of the physics."

## The math

### The controller — fixed linear state feedback, pole-placed on the nominal plant

Rather than re-deriving swing-up (08.01's territory), this project needs a controller that reliably
*holds* the cart-pole near upright so that domain randomization has something interesting to stress —
a fixed **linear state-feedback law**:

```
u = Kx*x + Kxd*ẋ + Kth*θ + Kthd*θ̇                (θ, θ̇ measured from upright)
u = clamp(u, -kUmax, +kUmax)
```

with the SAME gains `(Kx, Kxd, Kth, Kthd) = (12.0, 14.0, 73.0, 19.0)` applied in **every** environment
— one fixed policy, stress-tested across N randomized copies of the plant. That is deliberately the
Isaac-Gym farm's actual job description: 12.06 (RL training kernels, cited in SYSTEM_DESIGN.md Chain
C) replaces this hand-derived gain vector with a learned one; the farm around it — parallel envs,
domain randomization, episode reset — is unchanged.

**Where the gains come from.** Linearizing `cartpole_deriv` about `θ=0, θ̇=0` (nominal `m_c=1, m_p=0.1,
l=0.5`) gives `ẋ = Az + Bu` with `z = [p, ṗ, θ, θ̇]`:

```
A = [[0, 1,       0, 0],       B = [0,
     [0, 0, -0.7178, 0],            0.9756,
     [0, 0,       0, 1],            0,
     [0, 0, 15.7917, 0]]           -1.4634]
```

(the `15.79` and `-0.7178` entries come straight out of `cartpole_deriv`'s `g·sinθ` and cross-coupling
terms, linearized). Placing the closed-loop poles of `A - B·K` at `{-3, -3.5, -4±j}` via Ackermann's
formula gives `K ≈ [-12.43, -13.55, -73.41, -18.94]` (note the sign: `u = -K·z`, so the **positive**
gains above are `-K`) — rounded to the clean constants `kernels.cuh` ships. Those poles are a
deliberate, moderate choice: fast enough to recover from the ±8.6° initial-angle draw well inside the
±10 N actuator limit, slow enough not to demand implausibly large forces.

**Measured robustness margin.** Sweeping 5,000 `(m_c, m_p, l)` draws across the committed
domain-randomization envelope (±20% / ±30% / ±15%) and recomputing the closed-loop eigenvalues of the
*linearized* system at each draw, the worst-case (least-stable) eigenvalue's real part measured
**−1.72** — safely negative, i.e. every draw in the envelope remains linearly stable with real margin.
This is the theoretical explanation for what the farm run measures empirically (§How we verify
correctness): the SAME fixed gains balance every one of 10,000 randomized environments.

### Energy — the integrator's honest meter

For the **undriven** (u=0), frictionless cart-pole, total mechanical energy is an exact invariant of
the true (continuous-time) ODE — nothing removes energy, nothing adds it. Deriving it from the same
Lagrangian 08.01 cites: with the pole's center of mass at world position `(p + l·sinθ, l·cosθ)`,

```
E = KE_cart + KE_pole,trans + KE_pole,rot + PE_pole

KE_cart      = 1/2 · m_c · ṗ²
KE_pole,trans = 1/2 · m_p · [(ṗ + l·cosθ·θ̇)² + (−l·sinθ·θ̇)²]
KE_pole,rot   = 1/2 · I_cm · θ̇²,             I_cm = (1/3)·m_p·l²   (uniform rod, length 2l, about its OWN center)
PE_pole       = m_p · g · l·cosθ              (cart track is the y=0 reference)
```

`I_cm = (1/3)·m_p·l²` is the same "4/3" family as `cartpole_deriv`'s denominator
`l·(4/3 − m_p·cos²θ/(m_c+m_p))` — that 4/3 comes from `I_pivot/(m_p·l²) = I_cm/(m_p·l²) + 1 =
1/3 + 1 = 4/3` (parallel-axis theorem, pivot vs. center of mass) — the two formulas are not
coincidentally related, they come from the same rigid-body model (`cartpole_energy` in
`kernels.cuh`).

**Expected drift scaling.** RK4 has LOCAL truncation error `O(dt⁵)` per step; over `N_steps` steps the
GLOBAL error accumulates to roughly `O(N_steps · dt⁵)` for a well-behaved (non-chaotic, bounded)
trajectory. At `dt = 0.02 s` and `N_steps = 1000`: `dt⁵ ≈ 3.2×10⁻⁹`, so a naive estimate is
`~1000 × 3.2×10⁻⁹ ≈ 3×10⁻⁶` in absolute integration error terms — and because energy is a SMOOTH
function of state near this trajectory's operating point, the fractional energy drift should be the
same order of magnitude. §How we verify correctness reports the MEASURED number this predicts.

### Domain randomization — what it is and is not

`mass_cart[i]`, `mass_pole[i]`, `pole_half_len[i]` are drawn **once**, at farm init, from
`Uniform(nominal·(1−dr), nominal·(1+dr))` and held fixed for the environment's entire lifetime —
including across every episode reset. This models **per-unit uncertainty**: manufacturing tolerance,
payload variation, wear — the kind of thing that is fixed for a given physical robot but unknown to
the engineer at design time. It is deliberately **different** from `theta0`, which is redrawn at
**every** reset (init AND every mid-run reset): that models "where does THIS episode happen to start",
a property of the *episode*, not the *robot*. Conflating the two would be a real modeling bug: if
`mass_pole` were redrawn every reset, the farm would be testing "does this controller work for an
AVERAGE robot" rather than the more useful "does this controller work for EVERY robot in this
population" — domain randomization's actual research purpose (Tobin et al. 2017; the sim-to-real
transfer literature broadly) is the latter.

## The algorithm

Two kernel calls per farm "run", each covering the WHOLE population in one launch:

1. **`init_farm_kernel`** — one thread per environment: seed the env's RNG stream (`env_seed`), draw
   `mass_cart/mass_pole/pole_half_len` from the domain-randomization envelope, zero the metrics, take
   the first episode reset (draws `theta0`). `O(N)` work, `O(1)` per thread.
2. **`step_farm_kernel`** — one thread per environment, `T` ticks run INTERNALLY in a register-resident
   loop: evaluate the controller, clamp, RK4-integrate one step, classify (failed? balanced-quality?),
   advance the episode clock, reset in place on FAIL or CAP. `O(N·T)` total work, perfectly parallel
   across `N` (§The GPU mapping quantifies the "one launch, not `T` launches" design).

**Complexity.** Serial (one CPU core, one environment at a time): `O(N·T)` RK4 evaluations, each
`O(1)` (4 fixed-size derivative evaluations, 4 states) — for `N=10,000, T=1,000` that is 10 million
per-tick evaluations, ~40 million derivative calls total. Parallel (GPU, `N` threads): the SAME total
work, but spread across thousands of SMs' worth of concurrent threads, so WALL-CLOCK time scales with
`T` (the serial dependency WITHIN one environment — tick `t+1` needs tick `t`'s result), not with `N`
(environments never depend on each other). This is the textbook "embarrassingly parallel across the
batch dimension, sequential within the trajectory" shape shared with 08.01's rollouts and every
Monte-Carlo method in the repo.

## The GPU mapping

### SoA vs. AoS — the project's central new lesson

08.01 and 09.01 both use the pattern "one thread owns one unit of work, entirely in registers" — but
both units of work were STATELESS between kernel calls (a fresh rollout every MPPI tick; a fresh pose
every FK call). This project's units of work — environments — must PERSIST in global memory across
calls (and, on a real training farm, across the RL algorithm's read/update/step cycle). Once state
must live in global memory, the LAYOUT of that memory is the whole performance story:

```
SoA (this project):                 AoS (the alternative NOT used here):
  x[0] x[1] x[2] ... x[N-1]           env[0] = {x,ẋ,θ,θ̇}
  ẋ[0] ẋ[1] ẋ[2] ... ẋ[N-1]           env[1] = {x,ẋ,θ,θ̇}
  θ[0] θ[1] θ[2] ... θ[N-1]           env[2] = {x,ẋ,θ,θ̇}
  θ̇[0] θ̇[1] θ̇[2] ... θ̇[N-1]          ...

warp reading x[i..i+31]:            warp reading env[i..i+31].x:
  32 CONSECUTIVE floats               32 floats, each 16 BYTES APART
  -> ONE 128-byte transaction         -> up to 32 separate transactions,
     covers the whole warp               3/4 of every fetched cache line
                                         wasted on fields this load ignores
```

Because every thread in the farm is at the SAME tick `t` at the SAME time (lockstep — the defining
trait of this pattern, §The algorithm), a warp's 32 threads always want the SAME FIELD at the SAME
instant: 32 `x`-reads, then (inside the shared per-tick function) 32 `ẋ`-reads, and so on. SoA makes
every one of those a single coalesced transaction; AoS would scatter each one across 32 cache lines.
This is the 33.01 coalescing lesson, but the REASON it applies here — persistent, lockstep-accessed
state — is new relative to every prior GPU-mapping lesson in the repo, which is why kernels.cuh's
layout comment calls it out explicitly.

### Fusing the whole run into one kernel launch

08.01's rollout kernel is called ONCE PER 20 ms CONTROL TICK because the next tick's plan depends on a
HOST-side decision (the softmin blend) that must happen between GPU calls. Nothing in this project's
farm has that property: the controller, the fail/cap check, and the reset are all fully determined by
the PREVIOUS tick's on-device state, with no host decision in the loop. So `step_farm_kernel` runs the
ENTIRE `T`-tick loop internally, per thread, entirely in registers between one entry read and one exit
write of global memory (kernels.cu's header walks the memory traffic in detail). This is the concrete,
measurable difference between "a control loop that happens to use a GPU" and "a resident GPU
simulation": the aggregate throughput this project reports (§How we verify correctness) is only
possible because the host is not in the loop at all.

### Lockstep and divergence at resets

All `N` threads execute the SAME kernel code, but once environments start failing/capping at different
times, the `if (failed || capped) { ...reset... }` branch is taken by SOME threads and not others in
the same warp — a genuine, if small, amount of warp divergence (threads in a divergent branch execute
serially with the others masked off, then reconverge). Two things keep this cheap: (1) the reset body
is `O(1)` — a few multiply-adds and one RNG draw, not a loop, so the DURATION of divergence is bounded
and tiny relative to the `O(1)`-per-tick RK4 integration every thread does regardless; (2) with the
measured farm behavior (every environment resets at t≈200, 400, 600, 800 — the CAP boundary, an
INTEGER condition identical for every environment that has not already failed), most divergence in
this farm is actually **synchronized**: entire warps tend to hit the cap together, which is the
friendliest possible divergence pattern (contrast a hypothetical farm where failures were common and
uncorrelated across threads in a warp — THEORY's honest caveat, not this project's measured reality).

### Per-environment RNG streams

Each environment owns a persistent `rng_state[i]` (xorshift32), seeded once via `env_seed(base_seed,
i)` and advanced by every domain-randomization draw and every reset's `theta0` draw — entirely on the
device, no host round-trip (contrast 08.01, which generated noise on the HOST specifically to keep
runs bit-reproducible across platforms; this project cannot afford that round-trip at farm scale, so
it accepts the platform-portability trade the other way, §Numerical considerations).

## Numerical considerations

- **FP32 throughout**, matching repo convention; the farm's angles stay small by construction
  (episodes end at `|θ| > 12°`), so this project never meets the large-angle regime where 08.01/09.01's
  "precise trig, not intrinsics" choice matters as much — it is kept anyway (`sinf`/`cosf`, not
  `__sinf`/`__cosf`) so this file's numerics story stays a strict SUBSET of 08.01's, not a divergent one.
- **fmaf() discipline (new relative to 08.01).** Every RK4 stage blend uses `fmaf()` — a single,
  IEEE-754 correctly-rounded fused multiply-add — instead of a bare `a*b+c`, which nvcc's device
  compiler contracts into an FMA by default while cl.exe's host compiler does NOT (unless `/fp:fast`).
  Because "correctly rounded" has one unique bit-pattern answer, host `fmaf()`/`std::fma` and device
  `fmaf()` return the SAME result for the same inputs — this removes the entire FMA-contraction
  divergence class that the SAXPY scaffold placeholder's own comments called out as its ~1-ULP source
  of GPU/CPU disagreement.
- **Sharing dynamics across host and device (kernels.cuh's `HD` macro).** `cartpole_deriv`, `rk4_step`,
  `reset_episode`, and the RNG are written ONCE and compiled for both host and device — removing
  hand-copy bugs as a possible explanation for any measured GPU-vs-CPU divergence, but NOT making the
  two paths bit-identical by itself: nvcc's device `sinf`/`cosf` and the host CRT's `sinf`/`cosf`
  remain two independently-implemented, independently-rounded approximations (both accurate to a
  couple of ULP, neither obligated to agree with the other).
- **What IS bit-identical: the RNG.** `xorshift32`/`uniform01`/`uniform_range` are pure integer and
  fixed-point-style bit operations with no transcendental call — host and device produce IDENTICAL
  streams for the same seed. Consequently every domain-randomization draw, every initial angle, and —
  because episode-cap resets are triggered by an INTEGER step counter, not a float comparison — every
  CAP-triggered reset's TIMING is bit-for-bit identical between the GPU kernel and the CPU oracle. Only
  the continuous STATE (and any FAIL-triggered reset that depends on a float comparison crossing a
  threshold at a borderline instant) can differ, and only by the residual sinf/cosf disagreement.
- **Measured residual divergence (the §5 gate, "try for exact" honestly reported).** On the reference
  machine, the worst absolute state deviation between the GPU kernel and the CPU oracle over 256
  environments × 220 ticks measured **4.77×10⁻⁷** (Release) / **1.49×10⁻⁸** (Debug) — NOT exactly
  zero, so this project does not claim bit-exact GPU/CPU agreement, but small enough that the ONLY
  plausible explanation left, after the fmaf() discipline and the shared-source design removed every
  other candidate, is the sinf/cosf implementation gap. `reset_count` matched **exactly** (256/256) in
  every observed run, confirming the integer-triggered-reset argument above empirically, not just
  theoretically.
- **Determinism across platforms.** A GIVEN machine (fixed GPU, fixed driver) reproduces this run
  bit-for-bit run to run. A DIFFERENT GPU/driver may see its device `sinf`/`cosf` disagree with THIS
  machine's by a further ULP or two — the same honest caveat 08.01 documents. Stable output lines
  therefore carry no trajectory numbers, only PASS/FAIL against thresholds with measured, documented
  margin (`../src/main.cu`'s header comment restates this).

## How we verify correctness

Three INDEPENDENT checks, because a resident GPU farm can be numerically right and behaviorally wrong
in several unrelated ways at once:

1. **VERIFY (the §5 GPU-vs-CPU gate).** 256 environments, 220 ticks (> the 200-tick episode cap, so
   every environment passes through exactly one CAP-triggered reset — this window exercises
   randomization, integration, AND the reset path, not just free-running dynamics), on the GPU kernels
   and the CPU oracle from identical seeds. Three sub-checks: (a) worst absolute state deviation ≤
   1×10⁻³ (measured 4.77×10⁻⁷, ~2000× headroom); (b) `reset_count` exact match (measured: 256/256,
   every run observed); (c) `steps_balanced` within an integer slack of 3 (measured worst diff: 0).
2. **FARM (the farm-level physics/stability gate).** The full 10,000-environment run: (a) every state
   component finite (measured: 10,000/10,000); (b) every `reset_count` in `[5, 12]` — 5 is PROVABLE
   from `episode_cap=200, T=1000` (§The algorithm), 12 is a >2× margin over the MEASURED value (every
   single environment reset exactly 5 times — the pole-placement controller generalizes across the
   entire domain-randomization envelope with zero observed failures, matching the −1.72 linear-margin
   prediction in §The math).
3. **ENERGY (the integrator-honesty gate).** One undriven, unbounded trajectory (θ₀=0.5 rad, 1000
   steps): measured maximum relative energy drift **1.045×10⁻⁵** against a documented bound of 1×10⁻³
   (~100× headroom) — consistent with the `O(N_steps·dt⁵) ≈ 3×10⁻⁶` order-of-magnitude prediction in
   §The math (FP32 rounding adds noise on top of the true RK4 truncation signal, but does not change
   the order of magnitude here).

Every number above is the ACTUAL measured value on the reference RTX 2080 SUPER (CUDA 13.3, VS 2026),
not an estimate — see `demo/out/env_metrics.csv` and `demo/out/energy_drift.csv` for the full traces
behind the summary statistics.

## Where this sits in the real world

- **Isaac Gym / Isaac Lab** (NVIDIA) is this pattern's namesake and, at production scale, its
  destination: PhysX running THOUSANDS of environments' full **articulated rigid-body dynamics**
  (multi-link robots, not a single hinged pole), **contacts and friction** (this project has none — a
  cart-pole never touches anything but its own track), a **tensor API** exposing every environment's
  state as one big GPU tensor (this project's SoA arrays ARE that idea, at teaching scale), and deep
  integration with RL libraries (rl_games, RSL-RL) that consume exactly the `steps_balanced`-shaped
  reward signal this project's metrics stand in for.
- **MuJoCo MJX** and **Brax** take a different implementation strategy — the ENTIRE physics step is
  compiled via JAX/XLA (not hand-written CUDA kernels), trading some of this project's "no black
  boxes, hand-rolled kernel" transparency for auto-differentiability (gradients through the whole
  simulation, which this project's fixed-gain, non-learned controller has no use for, but a
  policy-gradient training loop very much does) and cross-hardware portability (TPU included).
- **What full engines add that this teaching core omits, honestly:** contact/collision resolution
  (LCP or impulse-based solvers — an entire discipline, 07.x's territory in this repo), multi-body
  articulation (Featherstone ABA/RNEA — 09.03's territory, cited from SYSTEM_DESIGN.md's Chain C as
  what a REAL quadruped farm's dynamics kernel would use instead of this project's closed-form 4-state
  cart-pole), actuator/motor dynamics beyond an ideal force clamp, and — the single biggest practical
  gap — SIM-TO-REAL validation that domain randomization alone does not guarantee (PRACTICE.md §3
  is explicit about this).
- **The RL connection, concretely.** 12.06 (PPO/GAE training kernels) is the intended NEXT project in
  this pattern's lineage: it would replace this project's fixed `(Kx,Kxd,Kth,Kthd)` gains with a small
  neural network, replace `steps_balanced` with a proper discounted-return computation, and add a
  GPU-side policy-gradient update — but keep the SoA state layout, the domain-randomization design, and
  the fused-kernel-per-rollout structure this project teaches completely unchanged. That invariance is
  the whole point of learning the PATTERN here before the learning algorithm on top of it.
