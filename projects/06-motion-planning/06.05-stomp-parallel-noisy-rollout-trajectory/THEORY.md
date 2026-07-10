# 06.05 — STOMP: parallel noisy-rollout trajectory optimization (born for GPU): Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2).

## The problem — physics & engineering first

**What a trajectory is, physically.** A robot moves its body (or its end-effector, or its joints)
along a path through space over time. That path is not free: every point on it is executed by real
actuators with hard limits. Three physical facts turn "draw a line from A to B" into an optimization
problem:

- **Obstacles are hard constraints.** Driving a robot through a shelf, a wall, or a person is a
  collision — the single most expensive failure mode a robot has. The path must keep clear of every
  obstacle, ideally with *margin* (sensors are noisy, control tracking is imperfect, the world moves).
- **Actuators have limits.** A path with a sharp corner demands, at that corner, infinite acceleration
  — and therefore infinite force/torque. Real motors saturate; commanding a kink either clips (the
  robot cuts the corner, possibly into an obstacle) or slams (jerk spikes that wear gears, trip current
  limits, and shake the payload). So a *good* trajectory is **smooth**: bounded curvature, bounded
  acceleration, low jerk. Smoothness is not aesthetics — it is what keeps torque and jerk inside the
  envelope the hardware can deliver.
- **Shorter is usually better, but not at any cost.** A path that takes a huge detour to stay clear
  wastes time and energy; one that hugs an obstacle to stay short invites collision. The planner trades
  these off.

**Our teaching instance.** We strip this to its essence: a **point robot in a 2-D plane** must get from
a start to a goal without hitting circular obstacles, along a smooth path. The point-robot / 2-D
simplification removes the robot's geometry and kinematics so the *optimization* stands alone — but the
math is exactly what a real planner runs in higher dimensions (a 6-DoF arm plans in its 6-D joint
space; the "obstacles" become configuration-space collisions). The obstacles are inflated into a smooth
**cost field**: high inside, a smooth halo out to a radius `kInfl`, zero beyond (the same idea as a ROS
`costmap_2d` inflation layer or CHOMP's obstacle cost). This is the engineering trick that makes
gradients-free optimization work — a *smooth* cost has a slope everywhere that pushes the path away
from obstacles even before it touches.

**Where the physics is, honestly.** STOMP itself is a computational method — it does not model motor
dynamics or thermals. Its physical *carrier* is the actuation chain that will execute the trajectory:
the smoothness term exists precisely so the emitted path respects that chain's acceleration/jerk
limits, and the collision margin exists because the sensors and controllers between the plan and the
motors are imperfect. PRACTICE.md §1–2 grounds this in the real drive chain.

## The math

**Trajectory representation.** We optimize `N = 64` interior waypoints per spatial dimension. Write the
free parameters for one dimension as a vector `θ ∈ ℝ^N`. The full path for cost evaluation prepends the
fixed start and appends the fixed goal:

```
P = [ start, θ_0, θ_1, …, θ_{N-1}, goal ]         (N+2 = 66 points, N+1 = 65 segments)
```

Start and goal are **never** perturbed or updated — the endpoints are boundary conditions.

**The smoothness cost and the matrix A.** We measure smoothness by summed squared **acceleration**,
approximated by the second finite difference. For interior waypoint `i`, the discrete acceleration is
`a_i = θ_{i-1} − 2θ_i + θ_{i+1}`. Collect these into a linear operator `A` (N×N), the discrete
Laplacian with a zero/Dirichlet boundary:

```
        ⎡ -2   1   0   0  …  0 ⎤
        ⎢  1  -2   1   0  …  0 ⎥
   A =  ⎢  0   1  -2   1  …  0 ⎥          (A θ) gives the vector of accelerations
        ⎢  ⋮           ⋱      ⎥
        ⎣  0   …       1  -2  ⎦
```

The smoothness cost is then `½ ‖A θ‖² = ½ θᵀ (AᵀA) θ = ½ θᵀ R θ`, with

```
   R = AᵀA          (N×N, symmetric positive definite, PENTADIAGONAL — bandwidth 2)
```

`R` is the discrete **biharmonic** operator; `θᵀRθ` is (twice) the total squared acceleration. Because
`A` uses a zero boundary, `R` "pins" the ends — waypoints near the boundary are penalized for moving,
which is exactly the endpoint-preserving behavior we want.

**The smoothing matrix M — the heart of STOMP.** Here is the single idea that most distinguishes STOMP
from a naive optimizer. Consider its inverse:

```
   M = R⁻¹ , with each COLUMN scaled so its largest entry equals 1/N   (the STOMP scaling)
```

`R⁻¹` is **dense and smooth**. Its `j`-th column is the response of the biharmonic operator to a unit
poke at waypoint `j` — the discrete **Green's function** of `AᵀA` — which is a smooth bump peaking at
`j` and decaying to ~0 at both ends. Two consequences flow from this, and they are the whole method:

1. **Smooth noise.** To sample a noisy trajectory we draw per-waypoint **white** noise `z ∼ 𝒩(0, σ²I)`
   and mix it: `ε = M z`. Since each column of `M` is a smooth basis function, `ε` is a smooth,
   spatially-correlated perturbation — a gentle bend of the whole path, not a per-waypoint jitter.
   *Independent* per-waypoint noise (`ε = z`) would give a jagged, high-frequency perturbed
   trajectory that violates the very smoothness we are trying to preserve; multiplying by `M` projects
   the noise onto the low-curvature subspace. (This is exactly `ε ∼ 𝒩(0, MMᵀ)` — correlated Gaussian
   noise whose correlation structure is the inverse biharmonic. README Exercise 3 lets you see the
   jagged version.)
2. **Endpoints stay put.** Because `M`'s columns decay to ~0 at the boundary, both the noise **and** the
   update barely move the waypoints next to the fixed start/goal — the boundary conditions are
   respected structurally, for free.

**The cost function.** For a candidate path we score:

```
   Q(P) = Σ_{segments}  ∫ c_obs(p) ds     +     w_smooth · Σ_i ‖a_i‖²
          └─ obstacle line-integral ─┘           └─ smoothness (θᵀRθ) ─┘
```

The obstacle term is a **line integral** of the cost field `c_obs` along the path (midpoint rule,
`kSegSamples` samples per segment × the arc-length step). Sampling *between* waypoints — not just at
them — is what lets a thin obstacle sitting mid-segment still register: a path could otherwise "hop
over" a wall between two waypoints unpunished.

**The per-waypoint softmin update — STOMP's signature, contrasted with MPPI.** Sample `K` noisy
trajectories `θ~_k = θ + ε_k`. For each **waypoint** `j`, gather the K local obstacle costs
`S_k(j)` (the cost of the two segments incident to waypoint `j` in rollout `k`), normalize them to
[0,1] across the K rollouts, and exponentiate with sensitivity `h`:

```
   for each waypoint j:
       w_k(j) = exp( −h · ( S_k(j) − min_k S(j) ) / ( max_k S(j) − min_k S(j) ) )
       P_k(j) = w_k(j) / Σ_k w_k(j)                        (per-waypoint softmin weights)
       δθ~(j) = Σ_k P_k(j) · ε_k(j)                        (blend the perturbations AT j)
   δθ = M · δθ~                                            (smooth the whole update)
   θ  ← θ + δθ
```

The contrast with **MPPI** (project 08.01) is the teaching point, and it is fundamental:

| | MPPI (08.01) | STOMP (this project) |
|---|---|---|
| Weight granularity | **one** weight per whole rollout: `w_k = exp(−(S_k−S_min)/λ)` | **N** weights, one per waypoint: `w_k(j)` |
| A rollout that is "good here, bad there" | counts as one number — its good part is diluted by its bad part | contributes **only at the waypoints where it helps** |
| Applied to | the whole control sequence uniformly | each waypoint independently, then `M`-smoothed |

STOMP's per-waypoint weighting extracts more signal from each batch of rollouts: a perturbation that
neatly clears the left obstacle but wanders near the right one still gets full credit at the left
waypoints. That per-timestep normalization `(S − min)/(max − min)` also keeps the softmin well-scaled
regardless of the absolute cost magnitude — a small robustness win MPPI gets from its `S_min`
subtraction instead.

**The knobs, concretely** (values in `kernels.cuh`): `N = 64` waypoints, `K = 1024` rollouts,
`σ = 4.0` white-noise std (fed into `ε = M z`), `h = 10` softmin sensitivity (the STOMP paper's value),
`kInfl = 0.6 m` inflation radius, `kSegSamples = 8` scoring samples/segment.

## The algorithm

Per iteration (the numbered steps are labeled in `main.cu`):

1. Draw white noise `z` (host xorshift32 + Box–Muller; fresh stream every iteration) and mix to smooth
   noise `ε = M z` for all K rollouts and both dimensions; upload the transposed `eps[j*K+k]` arrays.
2. **GPU:** K rollouts — each thread scores one noisy path: integrate obstacle cost along its N+1
   segments and accumulate the per-waypoint local costs `Sloc[j][k]`. (>99% of the arithmetic lives here.)
3. Per-waypoint softmin on the host (per waypoint: normalize, exponentiate, blend).
4. Smooth the raw update through `M` and apply it.
5. Evaluate the nominal trajectory's cost (host) for convergence; stop when the relative improvement
   stays below `1e-3` for 5 consecutive iterations.

**Complexity.** Per iteration: scoring is `O(K · N · kSegSamples)` field lookups — **parallel across
K** on the GPU, serial on the CPU oracle. The host update is `O(K · N)` for the softmin blend plus
`O(N²)` per dimension for the two `M`-matvecs (N=64 → ~4k mults, negligible). The one-time setup
inverts `R` in `O(N³)` (microseconds at N=64). The serial-vs-parallel gap is entirely in step 2, which
is exactly why STOMP is "born for GPU": the expensive part is `K` independent rollouts.

**Why the update stays on the host** (a deliberate anti-optimization, matching 08.01): the softmin
blend is `O(K·N)` trivial arithmetic (~65k multiply-adds — microseconds), and keeping it in plain C++
puts the entire STOMP update on one screen next to the kernel call. The per-iteration `eps` upload it
requires (~512 KB) is the measured, documented price; Exercise 4 removes it.

## The GPU mapping

```
one thread = one noisy rollout k:   scalar accumulators in REGISTERS (no per-waypoint arrays cached)
grid = ceil(K/256) × 256            (repo default; ragged tail guarded)

per waypoint j, per thread k:
  theta_x/theta_y[j]  UNIFORM read   (same address, all threads → L2/read-only cache, broadcast-like)
  epsx/epsy[j*K + k]  COALESCED read (TRANSPOSED layout: a warp's 32 reads are consecutive floats —
                                      the 08.01/33.01 lesson, applied by design; the naive eps[k*N+j]
                                      would stride N floats and waste ~90% of each transaction)
  field[...]          bilinear GLOBAL reads (data-dependent addresses as the path wanders; a real
                                      planner puts the field in a TEXTURE whose HW filter does the
                                      bilinear blend for free — we do it by hand so nothing is a black box)
  Sloc[j*K + k]       one coalesced write per waypoint; cost[k] one coalesced write at the end
```

No shared memory (rollouts share nothing), no atomics, no divergence beyond the tail guard. Register
pressure is deliberately low: the kernel never caches the 64 (x,y) waypoints (that would need ~128
registers and spill) — it recomputes each path point on demand from the uniform `theta` + coalesced
`eps` reads. The kernel is memory-latency-bound on the field lookups, which the SM hides by keeping
many rollouts in flight (high occupancy from the light register footprint) — the healthy regime for
this pattern.

**Why output a per-waypoint array at all?** MPPI's kernel returns one scalar per rollout; STOMP's must
return `Sloc[j][k]` (N values per rollout) because the host update reweights each waypoint separately.
That extra output is the concrete GPU-side consequence of the per-waypoint-vs-per-trajectory
distinction above.

## Numerical considerations

- **Conditioning of R.** `R = AᵀA` is the discrete biharmonic; its condition number scales like `N⁴`
  (roughly `1.7×10⁷` at N=64). We therefore build and invert `R` in **double precision** (Gauss–Jordan
  with partial pivoting): double carries ~15–16 significant digits, so we lose ~7 to conditioning and
  keep ~8 — plenty for a smoothing matrix. `M` is then stored as FP32 for the runtime blends, where the
  small residual error is irrelevant (it only shapes exploration noise). Doing the inversion in FP32
  would be reckless at this condition number.
- **FP32 in the scoring path.** The obstacle line-integral sums ~520 FP32 bilinear samples per rollout
  plus N smoothness terms. Kernel and CPU oracle do the same operations in the same per-rollout order,
  so they differ only by **FMA contraction** — nvcc fuses `a·b+c` into one rounding; MSVC may round
  twice. That is ~1 ulp per op, accumulating to ~1e-6 relative. (Measured: **2.2e-07** worst in
  Release; exactly **0** in Debug, where `-G` disables device FMA fusion so the two paths become
  bit-identical — a nice confirmation that the *only* difference is contraction.)
- **Softmin hygiene.** The per-waypoint normalization `(S − min)/(max − min)` bounds the exponent's
  argument to [0, h] regardless of cost scale; we guard the degenerate `max == min` case (all rollouts
  equal at a waypoint — e.g., all in free space) by falling back to uniform weights, which produce a
  ~zero update there. Weights are summed in **double**.
- **No angle wrapping here.** The state is a 2-D position, not an orientation, so the quaternion/angle-
  wrap hazards CLAUDE.md §12 flags do not arise. (In a joint-space extension they would, at the joint
  limits — noted for the honest record.)
- **Determinism.** Host-generated noise (xorshift32 + Box–Muller, base seed 42, fresh stream per
  iteration) makes runs bit-reproducible on a given machine. Across platforms, host `std::log/std::cos`
  ulp differences and GPU low-bit scoring differences can perturb the trajectory; the demo's verdict is
  engineered to be robust to that — the collision threshold (25, i.e. ≥0.30 m clearance vs. an achieved
  0.000 field value = ≥0.6 m clearance) and the cost-reduction bar (<5% of initial vs. an achieved
  0.0002%) both carry wide margins, and no stable output line contains a trajectory number.

## How we verify correctness

Two independent checks, because a planner can be *numerically right and behaviorally wrong* (or vice
versa) — the same philosophy as 08.01:

1. **The §5 GPU-vs-CPU gate (VERIFY):** iteration 0's exact inputs — same field, same `θ`, same
   1024×64 noise arrays — through the scoring kernel and through
   [`src/reference_cpu.cpp`](src/reference_cpu.cpp)'s line-by-line twin; per-rollout **total** costs
   must agree within rel 1e-3 (floor `max(1,|S|)`). Catches indexing/layout/sampling/clamp divergence
   instantly — any such bug shifts costs at order 1, not 1e-7. Measured worst deviation: **2.2e-07**.
2. **The end-to-end verdict (RESULT):** after the full optimization loop, the final trajectory is
   scored by the dense single-path evaluator (`kCheckSamples = 32` samples/segment, stricter than the
   scoring path). It must be **collision-free with margin** — the max field value anywhere along it
   below 25, which corresponds to staying ≥0.30 m clear of every obstacle (solve
   `100·((0.6−d)/0.6)² = 25 ⟹ d = 0.30 m`) — **and** the total cost must fall below 5% of the
   straight-line initialization. This catches everything the pointwise check cannot: a broken update, a
   mis-tuned sensitivity, jagged noise, a horizon bug — failures that leave every individual rollout
   "correctly scored" while the *planner* never routes around anything. Measured margin: final max
   field **0.000** (fully clear — the path stays out of every halo, ≥0.6 m from all obstacles), and the
   total cost falls **591.2 → 0.0013** (the collision cost is eliminated entirely; the residual is pure
   smoothness). Thresholds sit far from the achieved behavior on purpose.

The scenario is committed (`data/sample`) so the whole check runs offline; the artifacts
(`demo/out/trajectory.csv` and `demo/out/costfield.pgm`) make the behavior *inspectable*, not just
pass/fail.

## Where this sits in the real world

- **MoveIt's STOMP plugin** runs this exact algorithm for real robot arms: it plans in the arm's
  configuration space against a full collision world (meshes, octomaps), not a 2-D circle field, and
  feeds a real controller. The math — smooth noise from `R⁻¹`, per-waypoint update, `M`-smoothing — is
  unchanged; the cost function (a real collision check) and the dimension (7-DoF, not 2) are what grow.
- **CHOMP, the gradient-based cousin.** CHOMP optimizes the *same* obstacle+smoothness objective but by
  **gradient descent** on a differentiable cost (it needs a signed-distance field and its gradient). The
  honest trade: when the cost is smooth and differentiable, CHOMP converges faster and needs no
  sampling; but gradient descent gets **stuck in local minima** — near a thin obstacle the gradient can
  point *along* the wall instead of around it, and CHOMP stalls. STOMP is **derivative-free**: it needs
  no gradient (so it tolerates a noisy, non-differentiable, even discontinuous cost) and its stochastic
  exploration escapes shallow minima — at the price of scoring `K` samples per iteration. That price is
  exactly what the GPU pays cheaply, which is why *GPU* + *sampling* is a natural pair. In practice:
  sampling (STOMP) wins around thin/complex obstacles and non-differentiable costs; gradients (CHOMP)
  win when you already have a smooth SDF and want speed; modern stacks sometimes run both.
- **OMPL** is the *planning* alternative: RRT*/PRM find *a* feasible path from scratch by sampling the
  configuration space. STOMP instead *improves* an existing (possibly colliding) trajectory. Real
  pipelines chain them — a sampling planner for a coarse feasible path, then STOMP/CHOMP to smooth and
  shorten it.
- **cuRobo (NVIDIA)** is the modern industrial answer: a GPU trajectory optimizer for manipulators that
  batches thousands of rollouts/seeds over full robot geometry and collision spheres, with warm-starting
  and constraint handling. This project is the one-screen teaching core of what cuRobo productionizes.
- **What the full version adds** beyond this teaching core: configuration-space collision (not a 2-D
  field), joint/velocity/acceleration limit constraints, dynamic obstacles and re-planning, warm-starting
  from the previous plan, and adaptive noise covariance — and, on the GPU, on-device noise generation and
  a fused reduction to eliminate the host round-trip.
