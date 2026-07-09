# 08.01 — MPPI controller — the canonical GPU controller: cart-pole → quadrotor → AGV → off-road racer: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2).

## The problem — physics & engineering first

**The plant.** A cart of mass `m_c = 1 kg` slides on a frictionless track, driven by a horizontal
force `u` limited to ±10 N. A uniform pole (mass `m_p = 0.1 kg`, half-length `l = 0.5 m`) pivots
freely on the cart. State `x = [p, ṗ, θ, θ̇]` (SI; θ = 0 upright — layout contract in
[`src/kernels.cuh`](src/kernels.cuh)). From the Lagrangian (kinetic energy of cart + pole minus
pole's gravitational potential), the equations of motion reduce to the classic form implemented in
`cartpole_deriv`:

```
tmp = (u + m_p·l·θ̇²·sinθ) / (m_c + m_p)                      cart accel if the pole just rode along
θ̈   = (g·sinθ − cosθ·tmp) / ( l·(4/3 − m_p·cos²θ/(m_c+m_p)) )  gravity torque vs cart reaction
p̈   = tmp − m_p·l·θ̈·cosθ / (m_c + m_p)                        pole's back-reaction on the cart
```

**Why swing-up is the honest benchmark.** Hanging (θ = π) and upright (θ = 0) differ by
`ΔE = 2·m_p·g·l ≈ 0.98 J` of potential energy. The force limit means the cart cannot inject that
energy in one stroke — check: even the full 10 N acting over the whole track budget cannot
statically lift the pole; the controller must **pump** — rock the cart in resonance with the
pole's swing, adding energy each half-period, then *catch* the pole as it arrives at the top and
switch from pumping to stabilizing. That strategy is (a) genuinely nonlinear (a linearized model
around upright knows nothing about swinging), (b) non-obvious (it emerges from optimization, not
from a hand-written rule), and (c) exactly what makes this tiny plant the standard test that
separates real nonlinear controllers from linear ones.

**The engineering frame.** A controller is the last software before current flows in motors:
it runs on a fixed tick (here 50 Hz — SYSTEM_DESIGN item 1's local-planner/controller band), must
respect actuator saturation (the clamp in the rollout is not a detail; optimizing with forces the
motor cannot deliver is a small sim-to-real gap you can create in one line), and its compute
budget is hard: whatever MPPI does must fit inside 20 ms, every tick, forever. The GPU is what
makes "simulate 4096 futures per tick" fit (~0.3 ms measured); the physics is what makes those
futures worth simulating.

## The math

**Problem statement.** Discrete-time dynamics `x_{t+1} = f(x_t, u_t)` (our RK4 step), horizon T,
cost `S(u_{0:T-1}) = Σ_t q(x_t, u_t)` with stage cost `q` (the kW* weights in kernels.cuh).
Find the control sequence minimizing expected cost.

**The MPPI update.** Perturb the nominal plan with Gaussian exploration noise
`ε_k ~ N(0, σ²I)`, roll out `u_k = clamp(u_nom + ε_k)`, score `S_k`, then:

```
w_k    = exp( −(S_k − S_min)/λ )               (softmin weights; S_min subtracted for
                                                numerical safety — exp(0) instead of exp(−10⁴))
u_nom ← u_nom + Σ_k w_k·ε_k / Σ_k w_k          (noise blended by trajectory quality)
```

Apply `u_nom[0]`, shift the plan left, append 0, repeat next tick (receding horizon).

**Where the exponential comes from (the honest sketch).** Path-integral / information-theoretic
control (Williams et al. 2017) shows that the *optimal* control distribution reweights sampled
trajectories by `exp(−S/λ)` — the free-energy/KL-divergence machinery makes "weight futures by
exponentiated negative cost" not a heuristic but the closed-form importance-sampling answer, with
λ the temperature that trades exploitation (λ→0: argmin wins outright) against averaging (λ→∞:
all samples equal). The full derivation belongs to the paper; what this project teaches is the
*shape*: sample → exponential weights → blend, and why each knob exists.

**The knobs, concretely** (values in kernels.cuh, tuned for this plant):
σ = 2.5 N (exploration width: too small → can't discover pumping; too big → clamp saturates
everything), λ = 0.5 (temperature), K = 4096 (samples: more = smoother updates, linear GPU cost),
T = 50 at dt = 0.02 s (a 1 s lookahead — long enough to contain a whole pump-and-catch arc, which
is *why* the optimizer can discover it).

## The algorithm

Per control tick (the numbered steps are labeled in `main.cu`):

1. Draw fresh `ε` (host xorshift32 + Box–Muller; new stream every tick — reusing noise freezes
   exploration, a classic MPPI bug).
2. **GPU:** K rollouts — each thread: clamp, RK4-step, accumulate cost, T times. (>99% of the
   arithmetic lives here.)
3. Softmin weights on the host (double accumulators; S_min subtraction).
4. Blend: `u_nom[t] += Σ_k w_k ε_k[t]/Σw` — O(K·T) trivial host arithmetic, kept on the host so
   the whole algorithm reads in one file (fusing it on-GPU is README Exercise 3).
5. Act: apply `u_nom[0]` to the plant; log.
6. Shift the horizon.

Complexity per tick: O(K·T) dynamics evaluations (parallel across K), O(K·T) blend (host).
The **cost function** deserves its own note: the angle term is `kWAngle·(1−cosθ)` — smooth,
minimal at upright, and **wrap-free** (rollouts integrate θ unwrapped; cos doesn't care). Using
θ² instead would punish poles that swing "the long way" up — an accidental prior that visibly
degrades swing-up. Velocity and position quadratics damp the catch and recenter the cart; the
effort term is small (the force limit already constrains energy).

## The GPU mapping

```
one thread = one rollout k:   x[4] + RK4 scratch in REGISTERS, T-step loop
grid = ceil(K/256) × 256      (repo default; ragged tail guarded)

per step t, per thread k:
  u_nom[t]      UNIFORM read   (same address, all threads → L2/read-only cache,
                                broadcast-like; the middle of the spectrum
                                between 09.01's __constant__ and 07.09's
                                divergent reads)
  eps[t*K + k]  COALESCED read (TRANSPOSED layout: warp reads 32 consecutive
                                floats — the 33.01 lesson, applied by design;
                                the naive eps[k*T+t] would stride T floats)
  cost[k]       one coalesced write at the end
```

No shared memory (rollouts share nothing), no atomics, no divergence beyond the tail guard and
the clamp's `fminf/fmaxf` (branchless). Occupancy: ~30 registers/thread — light; the kernel is
compute-bound on the RK4 arithmetic, which is the healthy regime for this pattern. The kernel's
structure *is* the project: everything else (noise, weights, plant) exists to feed it.

**Why the blend stays on the host** (a deliberate anti-optimization): it is O(K·T) scalar work
(~200k multiply-adds — microseconds), and keeping it in plain C++ puts the entire MPPI algorithm
on one screen next to the kernel call. The eps download it requires (~800 KB/tick) is the
measured, documented price; Exercise 3 removes it and teaches a weighted reduction in the process.

## Numerical considerations

- **Integrator honesty.** RK4 vs Euler at dt = 0.02 s: Euler's O(dt²) per-step error visibly
  distorts the pendulum's energy budget over a 50-step horizon — and an energy-pumping controller
  optimizing a wrong energy model pumps wrongly. RK4's O(dt⁵) local error is orders below the
  process noise MPPI injects anyway. Four derivative evaluations per step is the standard price.
- **Angle wrapping discipline (CLAUDE.md §12):** rollouts integrate θ **unwrapped** (the cost uses
  cosθ, which is periodic); only the PLANT step wraps to (−π, π] — the project's single defined
  wrap point, in `cartpole_step_cpu`. Wrapping inside rollouts would corrupt θ̇ continuity at ±π
  mid-horizon.
- **Softmin hygiene:** subtract S_min before exp (overflow guard), accumulate weights in double
  (K=4096 tiny weights can vanish in float), divide once.
- **Trig:** precise `sinf/cosf` on both paths, not fast intrinsics — swing-up passes |θ| = π
  constantly, exactly where intrinsic error grows (same decision and reasoning as 09.01).
- **Determinism:** host-generated noise (xorshift32 + Box–Muller) makes runs bit-reproducible on
  a given machine. Across platforms, host `std::log/std::cos` may differ in ulps → noise low bits
  → (chaotically) trajectories. The demo's verdict is engineered to be robust to that: success
  thresholds carry wide margins, and no stable output line contains trajectory numbers. On-device
  cuRAND (Exercise 4) is what production uses — trading this reproducibility for per-tick upload
  savings.
- **FP32 chain depth:** VERIFY measures the GPU-vs-CPU rollout-cost divergence at 1.8e-07 relative
  over 50 chained RK4 steps — rounding plus trig-implementation differences, the by-now-familiar
  reason the gate is a tolerance (1e-3, ~100× headroom) and not equality.

## How we verify correctness

Two independent checks, because a controller can be *numerically right and behaviorally wrong*
(or vice versa):

1. **The §5 GPU-vs-CPU gate (VERIFY stage):** iteration 0's exact inputs — same x0, same u_nom,
   same 4096×50 noise array — through the kernel and through
   [`src/reference_cpu.cpp`](src/reference_cpu.cpp)'s line-by-line twin; per-rollout costs must
   agree within rel 1e-3 (floor max(1,|S|)). Catches indexing/layout/clamp/integrator divergence
   instantly (any such bug shifts costs at order 1, not 1e-7).
2. **The closed-loop success check (RESULT):** |θ| < 0.2 rad for every one of the final 100 steps
   (2 s) of the 400-step run. Catches everything the pointwise check cannot: broken weights, a
   mis-tuned temperature, noise reuse, horizon-shift bugs — failures that leave every individual
   rollout "correct" while the *controller* never swings up. Measured margin: balanced for the
   final 287 steps, final |θ| = 0.006 rad — the thresholds sit far from the achieved behavior on
   purpose (see the determinism note above).

The scenario is committed (data/sample) so the whole check runs offline; the trajectory artifact
(`demo/out/trajectory.csv`) makes the behavior *inspectable*, not just pass/fail.

## Where this sits in the real world

- **Williams et al.'s MPPI on AutoRally** drove a real off-road car at racing speeds — the
  bullet's last rung. Differences from this demo: learned/identified dynamics (not
  plant-as-model), cuRAND on-device noise, GPU-side weight reduction, thousands of rollouts of a
  much richer model, and a safety envelope around the whole thing.
- **MuJoCo-MPC** ships sampling MPC (MPPI among them) as an interactive research tool; NVIDIA's
  stack pushes rollout-based planning (and 10.03's massively-parallel simulation is this pattern
  scaled to full physics engines).
- **The gradient-MPC family** (iLQR/DDP — 08.03; QP/NMPC via acados/OSQP — 08.04) is MPPI's
  complement: faster convergence near a good solution, needs differentiable dynamics and behaves
  badly with contacts/discontinuities — exactly where sampling shines. Production stacks
  increasingly run both (sampling to explore, gradients to polish — 08.05+ territory).
- **Certification reality:** stochastic controllers are hard to certify; where a safety case is
  required, the actuation seat is typically held by something deterministic and simple, with
  sampling MPC planning above it and a monitored envelope between (31.x; PRACTICE §4).
- **What the full version adds** beyond this teaching core: covariance-adapted sampling, colored/
  smoothed noise, tube/robust variants (08.05), learned dynamics residuals (12.x), and the
  contact-implicit frontier ([R&D] 08.07-08).
