# 22.01 — 100k-agent swarm simulator: flocking, pheromone grids, stigmergy: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

**The biological carrier.** Flocks of starlings, schools of fish, and colonies of ants all produce
coordinated, adaptive group behavior with **no leader and no global communication channel**. Every
individual reacts only to what it can locally perceive — nearby flockmates, or a chemical trace in the
air — and the flock's shape, the school's evasive maneuver, the colony's foraging trail all *emerge*
from millions of purely local decisions. Craig Reynolds' 1987 "boids" model showed that three simple
local rules — separation, alignment, cohesion — are *sufficient* to reproduce convincing flocking in
simulation, and it became the founding result of a research area (multi-agent systems, swarm
intelligence) that robotics inherited directly.

**The robotics carrier.** A swarm of small robots — quadrotors doing search-and-rescue in a collapsed
building, ground robots inspecting a field, satellites in a constellation — faces the *same*
constraints biology solved: **communication bandwidth and range are limited** (a robot can talk
reliably to a handful of nearby teammates, not to all N of them at once — this is not a simplification
of this project's local interaction radius, it is the actual physical reason the radius exists),
**onboard sensing is local and noisy** (a camera or UWB radio tells you about nearby robots, not global
positions), **individual robots fail** (a swarm that depends on any one member, or on a central
planner talking to all N, is not robust — decentralization is a *reliability* requirement, not just an
algorithmic curiosity), and **compute and power are scarce per unit** (a 30-gram quadrotor cannot run
a global optimizer; it can run a few dozen floating-point operations per neighbor, every tick). Boids-
style local rules are attractive for robot swarms precisely because they satisfy all four constraints
by construction: a rule that only reads nearby neighbors needs only local sensing, tolerates
individual failures (the rest of the flock does not depend on any one agent), and costs O(neighbors),
not O(N), per robot per tick.

**Stigmergy — coordination through the environment.** Ants do not broadcast messages to other ants;
they deposit pheromone, and other ants read the pheromone where they happen to walk. This is
**stigmergy**: indirect coordination through modifications to a shared environment, discovered in
termite-mound biology (Grassé, 1959) and now a standard swarm-robotics coordination pattern. Its
appeal for robots mirrors boids': no addressed messages, no routing, no need to know who else exists —
an agent just reads and writes a shared field. This project's pheromone grid is a direct, if
simplified, computational model of exactly that field. In real deployed swarm robotics, since actual
volatile chemicals are hard for a robot to secrete and sense cheaply, stigmergy is more often emulated
with a shared occupancy/traversal map (each robot marks cells it has visited), radio beacons whose
signal strength decays with time (a physical analog of decay), or — in indoor swarm-robotics research —
literal light projected onto the floor that a downward camera reads as a "pheromone."

**The engineering frame.** This is a *coordination-layer* problem, not a low-level control problem:
the rules run at a modest rate (this project uses 20 Hz, `kDt = 0.05 s`, matching a realistic
local-planner/controller tick — SYSTEM_DESIGN §1.1) that sits comfortably above each individual
robot's own faster attitude/velocity control loop (0.5–1 kHz, outside this project's scope). The hard
engineering constraint that actually drives every design decision in this project is **count**: N is
not 10 or 100, it is 100,000, and the central lesson is that the *algorithm class* has to change (from
pairwise comparison to spatial binning) before any of the interesting swarm behavior becomes
computable at all.

## The math

**State.** Agent `i`'s state is position `(px_i, py_i)` and velocity `(vx_i, vy_i)`, both in meters
and meters/second, in a fixed, right-handed, gravity-irrelevant 2-D arena frame `[0, kArena]²` with
`kArena = 256 m` (SI throughout, per CLAUDE.md §12; `src/kernels.cuh` is the single layout authority).
The arena is **walled, not toroidal**: a soft linear force ramps up inside `kWallMargin = 8 m` of each
edge, and the integrator additionally hard-clamps positions into `[0, kArena]` as a belt-and-suspenders
bound — so no agent can ever leave the simulated world, the property the demo's "bounded" success
check rests on.

**The three boids rules**, computed over the set of neighbors `j` within radius `r = kRNeighbor = 1 m`
of agent `i` (Reynolds 1987, with a smoothing detail explained under Numerical considerations below):
for each neighbor, a **hat weight** `w = 1 − d/r` (1 at contact, 0 at the radius, linear between) is
computed from distance `d = |p_j − p_i|`, and three running sums are accumulated:

```
separation:  sum over neighbors closer than r_sep = 0.5 m of  −(1 − d/r_sep) · (p_j − p_i)/d
alignment:   weighted mean neighbor velocity   v̄ = (Σ w·v_j) / (Σ w)
cohesion:    weighted mean neighbor offset     c̄ = (Σ w·(p_j − p_i)) / (Σ w)
```

Combined into an acceleration `a = k_sep·separation + k_ali·(v̄ − v_i) + k_coh·c̄`: separation pushes
directly away from crowding neighbors, alignment is a first-order relaxation of `v_i` toward the local
mean heading, and cohesion is a linear spring toward the local centroid (the weighted mean *offset*,
not the weighted mean *position* — accumulating offsets instead of positions avoids subtracting two
large, nearly equal numbers, a free numerics improvement documented at `NeighborAccum` in
`kernels.cu`). A **wall force** (same linear-ramp shape) and a **weak pheromone-gradient pull**
(`k_pher · ∇φ`, deliberately small relative to the flocking gains so stigmergy *biases* the flock
rather than overpowering it) are added, then the total acceleration is clamped to `kAMax = 8 m/s²` by
**scaling the vector** (preserves direction; never per-axis clipping, which would distort it) before a
semi-implicit Euler step integrates velocity then position, and the resulting speed is clamped into
`[kVMin, kVMax] = [0.3, 2.0] m/s`.

**The pheromone field**, `φ(x,y,t)`, unitless "concentration" on the same 256×256 grid used for
neighbor binning (`kCellSize = 1 m`, chosen equal to `kRNeighbor` — see below), evolves by a
**diffusion–decay–deposit reaction equation**:

```
∂φ/∂t = D∇²φ − λφ + s(x,y,t)         D = diffusion coefficient, λ = decay rate,
                                      s = deposit rate (agents/cell this step × kDeposit)
```

discretized with the standard explicit 5-point Laplacian stencil (07.09's stencil pattern) and one
Euler step per simulation tick:

```
φ_c^{n+1} = (1 − λ)·(φ_c^n + κ·(φ_N + φ_S + φ_E + φ_W − 4φ_c)) + kDeposit · counts[c]
```

with `κ = kDiffuse = 0.15` and `λ = kDecay = 0.02`. **Stability**: the explicit 2-D diffusion stencil
is stable for `κ ≤ 0.25` (the von Neumann/CFL-style bound for this scheme); `κ = 0.15` sits inside that
bound with margin. **Equilibrium**: at steady state, uniform deposition roughly balances decay —
mean deposit per cell per step is `kDeposit · N / kNumCells = 0.05 · 100000 / 65536 ≈ 0.0763`, and at
equilibrium the decay term must remove that much per step, `λ·φ̄ ≈ 0.0763`, giving
`φ̄ ≈ 0.0763 / 0.02 ≈ 3.8` — matching the equilibrium estimate documented in `kernels.cuh` (diffusion
redistributes but does not change the *total*, so it drops out of this mean-field estimate). Boundary
condition: **zero-flux** — a missing neighbor across the arena edge is replaced by the center value,
which makes that term's contribution to the Laplacian exactly zero, so pheromone can neither enter nor
leave through the walls (matching the physical walls the agents themselves feel).

## The algorithm

**Why brute force is impossible, precisely.** Testing every pair of N agents against the interaction
radius costs `N·(N−1)/2` distance computations. At N = 100,000 that is **~5 × 10⁹ pairs per step**
(counting each pair once; ~10¹⁰ if counted both directions, as the natural "for every i, test every
j" loop does) — at 300 steps, ~3 × 10¹² total distance tests for one headline run. No general-purpose
processor evaluates trillions of floating-point comparisons at 20 Hz. The whole project exists to
replace this with something sub-quadratic.

**The uniform-grid counting sort**, rebuilt fresh every step (agents move, so yesterday's bins are
stale):

1. **Count** — one kernel, one thread per agent: compute the agent's cell index
   `cell(cx,cy) = cy·kGridDim + cx` from `cx = floor(px/kCellSize)`, `cy = floor(py/kCellSize)`
   (clamped at the edges), and `atomicAdd` the per-cell counter. O(N).
2. **Scan** — an **exclusive prefix sum** over the 65,536-cell histogram, computed on the *host*
   (`starts[c] = Σ_{c'<c} counts[c']`, `starts[kNumCells] = N`): so cell `c`'s agents will occupy
   contiguous slots `[starts[c], starts[c+1])` in the sorted-by-cell array. 65,536 additions is
   trivial serial work — a deliberate teaching choice to keep the round trip honest and visible rather
   than hide it behind a library call (README Exercise 3 replaces it with a GPU scan). O(kNumCells).
3. **Scatter** — one kernel, one thread per agent: each agent atomically claims the next free slot in
   its cell's range (`atomicAdd` on a *working copy* of `starts`, called `cursor`) and writes its own
   index there. O(N).
4. **Gather** — inside the flock kernel, each agent visits its **3×3 block of cells** (9 cells) and
   walks each cell's `[starts[c], starts[c+1])` range. Because `kCellSize == kRNeighbor` **exactly**,
   this 3×3 block is *provably* the minimal set of cells that can contain every point within radius
   `r` of any point in the center cell (a point in the center cell is within `r` of a point in a
   diagonal-neighbor cell only if that point is within the shared corner region, which the 3×3 block
   always covers) — so the grid gather finds the *identical* neighbor set brute force would, not an
   approximation. This is the load-bearing invariant that makes the GPU-vs-CPU lockstep comparison a
   real correctness proof rather than a "close enough" check.

**Complexity.** Average density is `N / kNumCells = 100000/65536 ≈ 1.53` agents/cell; a 3×3 block
therefore examines roughly `9 × 1.53 ≈ 13.7` candidates *on average* for a uniformly-scattered swarm.
Once agents actually flock, local density inside a flock core is far higher than the arena average
(the whole point of cohesion), so the *realized* per-agent candidate count runs higher — the ~30
figure documented in `kernels.cu`'s launch-configuration comment is a conservative, measured-in-practice
bound for a flocked swarm, not the uniform-density estimate. Either way this is **O(N × constant)**,
not O(N²): total work is bin (O(N)) + scan (O(kNumCells), independent of N) + scatter (O(N)) + gather
(O(N × ~30)) ≈ **O(N)** for fixed grid resolution — the qualitative change that makes 100,000 agents at
20 Hz possible on a single GPU. The same three-kernel counting-sort structure (histogram → scan →
scatter) is the standard building block behind GPU radix sort, SPH/fluid neighbor search, and BVH
construction — recognizing it here is meant to transfer directly to those contexts.

## The GPU mapping

Four kernels, three launch geometries (`kernels.cu` "Launch geometry" comment):

```
bin_count_kernel / bin_scatter_kernel / flock_step_kernel:
    1-D, one thread per AGENT, 256-thread blocks (repo default), ceil(N/256) blocks,
    ragged-tail guard (if (i >= n) return;)

pheromone_step_kernel:
    2-D, one thread per CELL, 16x16 tiles (07.09's grid/stencil geometry — the data is
    2-D and neighbor offsets are 2-D, so 2-D indices read like grid coordinates and keep
    x as the fast (coalesced) axis)
```

**Memory hierarchy, per kernel:**

- `bin_count_kernel` / `bin_scatter_kernel` — coalesced `px`/`py` reads (SoA pays off: consecutive
  threads read consecutive floats); `counts[]`/`cursor[]` take **atomic** read-modify-writes into a
  65,536-entry global array — contention is mild (~1.5 agents/cell average), so most atomics resolve
  uncontended in L2 on sm_75+ rather than serializing the whole kernel.
- `flock_step_kernel` — the agent's own state lives in **registers** for the whole step (read once,
  reused through the neighbor loop and `finish_agent`; the `NeighborAccum` running sums are also
  register-resident, ~40 registers/thread estimated in the kernel's memory-spaces comment — light
  enough that occupancy is not the bottleneck). `starts[]` reads are effectively random access into a
  65k-entry table (a warp's 32 agents are scattered across the arena) and lean on L2. `bin_agents[s]`
  then `px/py/vx/vy[j]` is the kernel's one genuinely **uncoalesced GATHER** — the `j` indices a warp's
  32 lanes fetch are scattered across the whole agent array, unlike every other array access in this
  project, which are all coalesced by construction (SoA + consecutive-`i` thread mapping). This is
  intrinsic to any neighbor search, not a bug: production simulators address it by **reordering agent
  state into bin order** after every scatter, so a warp's gathers become contiguous (README Exercise 5
  quantifies the win this demo leaves on the table for teaching clarity).
- `pheromone_step_kernel` — the classic 5-point stencil access pattern: center + one coalesced write;
  N/S neighbors are one full grid row away (still coalesced, different cache line); E/W neighbors
  overlap the warp's own loads and are served from L1/L2. No shared memory in either kernel: bin sizes
  are data-dependent and variable, and a stencil block small enough to tile (16×16) does not amortize a
  shared-memory halo load enough to be worth the complexity at this problem size — a fair trade-off to
  question in a larger stencil (README Exercise 5's second half explores the analogous tiling question
  for the flock kernel's gather).

**Divergence, honestly.** Warp lanes in `flock_step_kernel` loop over different numbers of neighbor
candidates (bin sizes vary with local density), so a warp runs as long as its *busiest* lane while idle
lanes wait — a genuinely **data-dependent** divergence pattern, the first one in this repo's
foundational-through-flagship sequence whose trip count depends on the input rather than only on
`blockIdx`/`threadIdx`. At the ~1.5 agents/cell average density here this costs tens of percent of
throughput, not an integer factor — visible in the measured 0.45 ms/step flock-kernel time relative to
a naive occupancy estimate, not a correctness problem.

**No CUDA libraries are linked.** The exclusive scan (a natural fit for Thrust's `exclusive_scan` or
CUB's `DeviceScan::ExclusiveSum`) runs on the host instead, by design (README Exercise 3 names the
swap); noise/spawn use the repo's portable xorshift32 rather than cuRAND, for bit-reproducible spawns
across machines without a cuRAND state-management dependency.

## Numerical considerations

**This project's central numerics lesson: atomics and float reordering.** Two different atomic uses
appear, with two different determinism outcomes, and telling them apart is the point:

- **`bin_count_kernel`'s histogram** uses `atomicAdd` on **integers**. Integer addition is
  **associative** — the finished histogram is bit-exact *no matter what order* the atomics land in.
  Because every agent deposits the identical `kDeposit` amount, the pheromone deposit map is therefore
  `kDeposit * counts[c]`, computed exactly from that same bit-exact histogram — a deliberate design
  choice (uniform per-agent deposits) that keeps a second major subsystem deterministic for free.
  README Exercise 4 breaks this on purpose (variable per-agent deposits require **float** atomics,
  which do *not* associate) so the contrast is felt, not just read.
- **`bin_scatter_kernel`'s cursor** also uses integer `atomicAdd` (so the *slot assignment* is race-free
  and valid), but **which agent lands in which slot** depends on which thread's atomic executes first —
  a property of the GPU's scheduler, not of agent data, and therefore **not reproducible run to run**.
  This reorders the sequence in which `flock_step_kernel`'s neighbor loop folds contributions into each
  agent's sums, and **floating-point addition is not associative** — `(a+b)+c` and `a+(b+c)` can differ
  in the last bit. The neighbor *set* is identical every run (the 3×3 grid gather is exact, see §The
  algorithm); only the *summation order* within that set varies.

**Why this is survivable rather than a bug: the hat weight.** Every rule contribution is weighted by
`w = 1 − d/r`, which goes **smoothly to zero at the interaction radius** rather than cutting off with a
hard `if (d < r)` step. If an ulp of rounding difference flips whether a borderline pair (`d` within a
few ulps of `r`) counts as "inside" the radius, its contribution changes by an amount proportional to
that same ulp — not by an O(1) force jump. Without this smoothing, a hard cutoff plus reordering could
occasionally flip a force by a large, discrete amount right at the boundary, and the GPU-vs-CPU gate
would flake unpredictably on exactly the runs where a pair happened to sit near `r`. The unweighted
(textbook Reynolds) formulation is README Exercise 2 — building it is the fastest way to feel why this
matters.

**Measured consequence.** Lockstep verification (N = 4,096, 100 steps, both paths re-anchored to the
same shared state every step — see below) measured a **worst per-step deviation of ~1.5 × 10⁻⁵ m**
position, **~1.2 × 10⁻⁷ m/s** velocity, and **~1.2 × 10⁻⁷** pheromone concentration — consistent with
accumulated float rounding across a handful of reordered terms, not a bug (a bug in the binning or
gather would show up as an O(1) disagreement on step 0, because the neighbor *sets* would differ, not
just the summation order within them). The chosen tolerances (`kTolPos = kTolVel = kTolPher = 1e-3`)
carry roughly **65×–8,000× headroom** over these measured deviations.

**Chaos and why the gate runs in LOCKSTEP.** Flocking is a chaotic dynamical system: two trajectories
that start an ulp apart diverge exponentially, not linearly, over time. A **free-running** comparison
(run the GPU for 100 steps, run the CPU for 100 steps independently, then compare final positions)
would amplify the benign ulp-scale reordering above into meters of disagreement well before step 100 —
correctly reporting that the two paths *are* chaotically different, while incorrectly implying a bug.
The fix (`verify_lockstep` in `main.cu`) **re-anchors every step**: both paths compute one step from
the *same* shared state, are compared, and then the GPU's output (not the CPU's) becomes the next
shared input for both. This measures "how much does one step of floating-point reordering cost," which
is small and bounded, instead of "how much does 100 steps of chaos amplify a small difference," which
is unbounded and uninteresting for a correctness check.

**Other numerics choices, briefly:**

- **Clamp by scaling, never by clipping** — both the acceleration clamp and the speed clamp scale the
  whole vector down when it exceeds a limit, preserving direction. Per-axis clipping would silently
  turn a diagonal force into a different direction, corrupting the physics for the sake of a cheaper
  branch — never worth it here.
- **Semi-implicit Euler** — velocity update first, then position with the *new* velocity — the standard
  stable choice for a velocity-clamped particle system at this `dt`; nothing in this model is stiff
  enough to need RK4 (contrast 08.01's cart-pole, whose energy-sensitive swing-up *does* need it).
- **Continuous wall force, not reflection** — a hard velocity-sign-flip at the wall would introduce
  exactly the kind of discontinuity the hat-weight section above is designed to avoid; the linear ramp
  keeps the whole force field continuous everywhere.
- **`sqrtf`/`floorf` on device, `std::sqrt`/`std::floor` on host** — both IEEE-754 correctly rounded,
  so identical input bits produce identical output bits on both paths; this is *why* `cell_coord()` is
  bit-identical between `kernels.cu` and `reference_cpu.cpp`, which is in turn why the neighbor sets
  match exactly rather than approximately.
- **No angle wrapping anywhere in this project** — agents are described by Cartesian velocity, not
  heading angle, so the robotics-standard angle-wrap hazard (CLAUDE.md §12) simply does not arise here;
  worth naming explicitly since most of this repo's control/planning projects do wrestle with it.

## How we verify correctness

Two independent checks, because a swarm simulator can be *numerically right and behaviorally boring*
(or the reverse — see 08.01's controller for the same two-check philosophy):

1. **The §5 GPU-vs-CPU lockstep gate (VERIFY stage).** A small deterministic swarm, N = 4,096 (not the
   headline 100,000 — see below for why), spawn seed 42, run 100 steps in lockstep against
   `reference_cpu.cpp`'s brute-force O(N²) oracle. Position, velocity, and pheromone must agree within
   absolute tolerance `1e-3` every step. This is the check that catches **structural bugs**: a wrong
   cell index, a missed bin, an off-by-one in `starts`, a forgotten ping-pong swap — any of these would
   change the *neighbor set* an agent sees, producing an O(1) disagreement on step 0, not an ulp-scale
   one. Measured worst deviations (see Numerical considerations) sit far inside tolerance, as expected
   for a working implementation.
2. **The headline behavioral check (RESULT).** All 100,000 agents finite and inside `[0, kArena]`
   ("bounded" — catches NaN/divergence the lockstep gate's small N might not exercise), and the mean
   local velocity alignment (cosine similarity between each agent's velocity and its local neighborhood
   mean, averaged over agents that had neighbors) reaches **≥ 0.5**. Measured: **~0.97**, against a
   maximally-disordered random-heading spawn that scores ~0 — a wide margin chosen deliberately so
   run-to-run atomic-ordering ulps and (on a different machine) libm ulps in the trig used for spawn
   headings can never flip the verdict. This check catches what the pointwise gate cannot: a flock
   step that is locally "correct" at N=4,096 but whose emergent behavior fails to flock at N=100,000
   (e.g., a mistuned gain, or a rule that only works at low density).

**Why N = 4,096 for the CPU oracle, never N = 100,000.** The oracle is deliberately O(N²) brute force
(§The algorithm's whole point — an *independent* algorithm from the grid, so grid bugs cannot hide).
At N = 4,096, 100 lockstep steps measured **~1.5–1.6 seconds of total CPU time** on the reference
machine — acceptable for a demo gate. Extrapolating the same O(N²) cost to N = 100,000 (a
`(100000/4096)² ≈ 596×` slowdown per step from the squared term alone, times 3× more steps for a full
300-step comparison) gives an estimated **~46 minutes** just for the reference computation — not
measured, deliberately not run, and named here as an extrapolation so the "why O(N²) is impossible"
claim in the README is backed by an honest number rather than only the abstract 10¹⁰-pairs argument.

**Edge cases exercised:** wall-adjacent agents (the spawn margin plus wall force are tested every run,
since agents spawn near but not at the walls and the wall force must be smooth there); the
zero-neighbor case (`kNoNeighborScore` sentinel, filtered out of the alignment metric rather than
contributing a spurious 0); the near-zero-speed case in the speed clamp (a deterministic kick to
`kVMin` rather than a `0/0` division).

## Where this sits in the real world

- **Reynolds' boids** remain foundational — used directly in computer graphics (crowd/flock animation)
  and as the conceptual starting point for swarm robotics research, but production **multi-agent
  collision avoidance** more often reaches for **ORCA/RVO2** (van den Berg et al.), which guarantees
  collision-free trajectories by solving a local velocity-obstacle optimization each tick — a stronger
  guarantee than boids' potential-field separation rule, which can still produce close calls under
  adversarial configurations. A production swarm stack commonly layers both: boids-style rules (or a
  learned policy) for the coarse coordinated *behavior*, ORCA (or a similar certified-safe layer)
  underneath as a hard collision-avoidance guarantee.
- **NVIDIA Warp / Isaac Sim** implement this same class of large-N agent simulation — differentiable,
  GPU-resident, and used for training and stress-testing autonomy policies against simulated crowds and
  fleets at scales that make this project's 100,000 agents look modest; the counting-sort spatial grid
  here is architecturally the same primitive their neighbor queries use.
- **Real physical swarms** — Bitcraze's Crazyflie quadrotors with the Crazyswarm/Crazyswarm2 ROS 2
  stack, and Rubenstein, Ahler & Nagpal's **Kilobot** swarm (1,024 real, cheap robots, *Science* 2014,
  shape formation from local rules) — are the hardware end of this project's spirit: real decentralized
  local-rule coordination, at a physical scale three to four orders of magnitude below this project's
  simulated 100,000, which is itself the honest measure of how much harder physical deployment is than
  simulation.
- **Stigmergic robotics research** (Dorigo's IRIDIA group and successors) has built real robot swarms
  that coordinate via **virtual pheromones** — light projected on the floor, or radio broadcasts that
  decay with time — standing in for the chemical trails ants use, because real robots cannot easily
  secrete and sense actual chemicals cheaply. This project's pheromone grid is a direct computational
  model of exactly that mechanism.
- **What the full version would add** beyond this teaching core (README's bundled-bullet note and
  Exercises 4–5 name the concrete next steps): per-agent local/noisy sensing instead of exact global
  neighbor queries (the single biggest simulation-to-reality gap here — PRACTICE §1–§3); multiple
  pheromone channels with different semantics (attractive "food" trails vs. repulsive "danger" zones,
  the ant-colony-optimization pattern); asynchronous, physically-distributed computation instead of one
  synchronous GPU kernel; heterogeneous agent roles; and an obstacle/no-fly-zone field (07.09's distance
  transform is the natural building block) that a deployed swarm would need and this teaching version
  does not implement.
