# 23.01 — GPU costmaps: inflation, raytrace clearing, multi-layer fusion: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2).

## The problem — physics & engineering first

**The sensor, briefly (see 04.01 for the fuller treatment).** A 2-D LiDAR spins a laser and measures
range by time-of-flight (or, in cheaper units, phase shift): a pulse leaves the sensor, reflects off
the first surface it meets, and the elapsed time `t` gives range `r = c*t/2`. Real units report one
range per angular step (`sensor_msgs/LaserScan`'s `angle_min`, `angle_increment`, `ranges[]`), with
finite angular resolution (a beam has some divergence, not a mathematical ray), a maximum range set by
laser power and receiver sensitivity, and range noise that grows with distance and shrinks with
surface reflectivity. This project simplifies all of that to the essential shape a costmap consumer
actually needs: `kNumBeams = 360` idealized rays, evenly spaced over the full circle, each returning
either a hit range or "nothing within `kMaxRangeM`." 04.01 (massive particle-filter localization) is
where this repository develops the fuller beam-noise sensor model properly; this project deliberately
does not re-derive it — a costmap's job starts the instant a scan (real or simulated) exists, and this
project's honest scope is everything from that point onward.

**Why costmaps exist — burying uncertainty in a margin, not a probability.** A planner cannot treat
"the wall is exactly at cell (140, 82)" as ground truth: sensor noise, localization drift, and the
robot's own finite size all mean the *true* danger zone around a detected obstacle is a swollen region,
not a single cell. A costmap is the standard robotics answer: instead of carrying that uncertainty
through the planner as a probability distribution (expensive, and most local planners cannot consume
one anyway), **bury it into geometry** — inflate every obstacle outward by a margin large enough that
"the cost is below the danger threshold" is a genuinely safe place for the robot's *center point* to
be, even though the underlying detection was only ever a point.

**How big must that margin be?** Two numbers set the floor, and this project's constants
(`src/kernels.cuh`) are chosen so the arithmetic actually clears them, not just gestures at them:

1. **The robot's own radius** (`kInscribedRadiusCells = 4` cells = 0.20 m): any cell within this
   distance of a lethal cell is a cell the robot's *body* would already be touching the obstacle from,
   independent of any dynamics — pure geometry. This project's `kCostInscribed` plateau exists for
   exactly this radius.
2. **Stopping distance at the control rate**: even a perfectly-planned robot needs room to react and
   brake. With this project's constants — `kVMax = 0.6` m/s, deceleration bound `kAccelV = 0.8` m/s²,
   and a 0.1 s control period (`kDtControl`, i.e. one tick of "the robot hasn't replanned yet") — the
   worst-case stopping distance is
   ```
   d_stop = v_max * dt_control + v_max^2 / (2 * a_decel)
          = 0.6*0.1 + 0.6^2/(2*0.8)
          = 0.06 + 0.225
          = 0.285 m  (5.7 cells)
   ```
   `kInflationRadiusCells = 10` cells (0.50 m) clears this with **~1.75x margin** over the raw physics
   — deliberately generous, because the inflation radius must also cover the robot's own body radius
   from point 1 (they are not independent: a cell is "safe to plan into" only once BOTH the body-size
   margin and the stopping-distance margin are satisfied, and this project folds both into one radius
   for simplicity, stated honestly as a simplification real stacks sometimes split apart, e.g. Nav2's
   separate `inflation_radius` and footprint-based collision checking).

**The engineering frame.** Costmap update and local planning share the SYSTEM_DESIGN.md §1.1 rate
band (costmap 5-20 Hz, local planner 10-50 Hz) — this project runs both at 10 Hz. Every tick's GPU
work must fit comfortably inside that 100 ms budget, every cell's cost must reflect this tick's actual
sensor data (a stale costmap is a costmap lying about where it is safe), and the whole pipeline must
degrade safely (never silently "trust" an obstacle that vanished from a race — the raytrace kernel's
mark-wins-over-clear discipline, below, is exactly this engineering requirement expressed as an atomic
operation).

## The math

**Grid and cost.** An occupancy grid is a `W x H` array of cells, resolution `kResolutionM` meters
per cell, following the `nav_msgs/OccupancyGrid` convention (SYSTEM_DESIGN.md §3.6). This project's
cost byte lives in `[0, 254]`: `0` = free, `1..252` = an inflation gradient, `253` = inscribed
(certain collision by body size alone), `254` = lethal (an actual obstacle) — Nav2's exact convention,
by design (README §11).

**The three-layer fusion.** Given a static prior map `S` (known a priori) and a per-tick sensed layer
`O` (this tick's scan, discretized), the inflation layer `I` is computed from the union of their
lethal cells, and the master costmap is the pointwise maximum:
```
lethal(x,y)  =  [S(x,y) = 254]  OR  [O(x,y) = 254]
I(x,y)       =  decay( min over lethal (x',y') of dist2((x,y),(x',y')) )
M(x,y)       =  max( S(x,y), O(x,y), I(x,y) )
```
`max`, not a weighted blend: a cell is exactly as dangerous as its MOST dangerous layer says it is —
one confident lethal detection must never be diluted by two layers that both happen to say "free."

**The dynamic window (the "Dynamic" in Dynamic Window Approach, Fox/Burgard/Thrun 1997).** A robot
cannot instantaneously change velocity. Given the current commanded velocity `(v_prev, w_prev)` and
acceleration limits `(a_v, a_w)`, the set of velocities reachable within one control period
`dt_control` is a small window centered on the current velocity:
```
V_d = [ v_prev - a_v*dt_c ,  v_prev + a_v*dt_c ]  x  [ w_prev - a_w*dt_c ,  w_prev + a_w*dt_c ]
```
intersected with the robot's absolute limits `[0, v_max] x [-w_max, w_max]` (`dynamic_window()` in
`src/main.cu`). This project samples that window on a fixed `kVSamples x kWSamples = 64x64` grid —
4096 candidates, one GPU thread each.

**Admissibility.** The ORIGINAL DWA paper defines admissible velocities via a closed-form
stopping-distance criterion: `(v,w)` is admissible only if the robot could stop before reaching the
nearest obstacle on its arc, given its deceleration limit. This project uses the common, simpler
PRACTICAL substitute many real implementations use instead: forward-simulate the FULL 2-second
candidate arc and reject it outright if any sampled point along it touches a lethal cell or leaves the
mapped area (`dwa_score_kernel` / `dwa_scores_cpu`). The two are closely related (both ultimately ask
"can I survive committing to this velocity"), but the practical version is honest about being an
approximation of the textbook one — stated so a reader who goes looking for the closed-form stopping
criterion in this code knows not to expect it.

**The score.** Among admissible candidates, this project minimizes a weighted sum (the "document the
tuning, don't hide it" convention 08.01 established for its own `kW*` constants):
```
score(v,w) = kWObstacle * mean(sampled costmap cost along the arc) / kCostLethal      (avoid danger)
           + kWGoalDist  * (distance from the arc's END to the goal) / mission_dist   (make progress)
           + kWHeading   * (1 - cos(bearing_to_goal - heading at arc's end))          (face the goal)
           - kWSpeed     * (v / kVMax)                                                (don't just stop)
```
The heading term reuses 08.01's exact trick: `1 - cos(angle error)` is smooth, minimal when aligned,
and needs no `wrap_to_pi` — raw, unwrapped angle subtraction is safe because `cos` is periodic.

**DWA's known failure mode — local minima (an honest citation, not swept under the rug).** Because
DWA only ever looks `kHorizonS = 2` seconds ahead and only ever compares velocities reachable THIS
tick, it has no notion of "I need to move away from the goal briefly to find a way around this."
A textbook trap: a long wall between the robot and the goal, with the only opening far to one side.
Every admissible candidate that moves toward the opening scores WORSE on the heading/goal terms than
one that just pushes straight at the wall (which then becomes inadmissible at the wall itself) — the
robot can get pinned, oscillating or stalled, arbitrarily far from the goal. This is not a bug in this
project's implementation; it is DWA's textbook behavior, and it is exactly why production stacks (Nav2
included) always run DWA/DWB as a LOCAL planner underneath a GLOBAL route planner (A*/Dijkstra over
the costmap) that already knows which side the opening is on — DWA then only ever has to track a route
that is known feasible, never discover one. **This project has no global planner** (out of the
catalog bullet's scope), so its committed scenario (`scripts/make_synthetic.py`) is deliberately built
from small, isolated pillar obstacles that a purely reactive planner CAN solve — README Exercise 3
describes swapping back to an earlier full-width "slalom" layout that reliably reproduces the trap
described above, worth doing once to feel the failure mode firsthand rather than only read about it.

## The algorithm

Per control tick (`kDtControl = 0.1` s, the numbered steps are labeled in `src/main.cu`):

1. **Sense** — simulate the LiDAR scan from the robot's current pose against the TRUE world map
   (host, deterministic DDA stepping — `simulate_lidar_scan`), producing a pre-discretized integer
   endpoint cell per beam (why pre-discretized: see §Numerical considerations below).
2. **GPU costmap pipeline**, three kernels in stream order (`launch_costmap_update`):
   a. **Raytrace** — one thread per beam, Bresenham-walk the grid from the robot's cell to the
      beam's endpoint, marking/clearing via `atomicMax` (§The GPU mapping).
   b. **Inflate** — one thread per cell, bounded `(2R+1)^2` gather over `static ∪ obstacle`.
   c. **Fuse** — one thread per cell, `master = max(static, obstacle, inflation)`.
3. **GPU DWA scoring** — one thread per `(v,w)` sample, forward-simulate 2 s (20 RK4 substeps of
   0.1 s each) against THIS tick's master costmap, accumulate the weighted score above.
4. **Argmin** (host, plain C++, the same "keep the reduction visible" choice 08.01 makes for its
   softmin blend) — pick the lowest-scoring ADMISSIBLE sample; if none exists, apply the standard DWA
   safety fallback (brake at `a_v` toward `v=0`, `w=0` — measured 0 times over the committed run).
5. **Act** — apply the chosen `(v, w)` to the diff-drive plant for one `dt_control` (RK4, wraps theta
   — the project's single defined wrap point, `diffdrive_step_cpu`).
6. **Repeat**, using the newly-driven pose as next tick's sensing origin (a genuinely CLOSED loop:
   each tick's costmap depends on where the previous tick's chosen action actually put the robot).

Complexity per tick: raytrace `O(beams * beam_length)` ≈ `O(360*120)`; inflation
`O(cells * R^2)` ≈ `O(65536*441)`; DWA `O(samples * substeps)` = `O(4096*20)`. All three are
embarrassingly parallel across their respective "one thread per X" — the whole reason this pipeline
belongs on a GPU rather than a single CPU core (measured CPU-vs-GPU timings: README §"What this
computes").

## The GPU mapping

Three distinct access patterns, one project — each cross-references the sibling flagship that
established it:

```
RAYTRACE  (kernels.cu: raytrace_kernel)          — thread-per-BEAM, 360 threads, tiny launch
  pattern: map/scatter WITH a genuine race (07.09's ping-pong AVOIDS races by construction;
           this kernel instead RESOLVES one with atomicMax — a third point on that spectrum)
  memory : Bresenham walk touches a handful of cells per thread, data-dependent addresses;
           obstacle_layer is `int` (not byte) because CUDA has no native 1-byte atomicMax

INFLATION (kernels.cu: inflation_kernel)         — thread-per-CELL, 2-D 16x16 tiling (07.09's tile)
  pattern: bounded stencil/gather — same FAMILY as 07.09's jump-flooding pass, but brute-force
           over a small fixed radius instead of propagated over O(log) passes (self-contained
           by design — no JFA dependency, per this project's scope)
  memory : each thread reads up to (2R+1)^2 = 441 neighbor cells from TWO input layers; x is the
           fast axis (07.09's coalescing lesson, restated: swapping x/y here is the classic mistake)

FUSION    (kernels.cu: fusion_kernel)            — thread-per-CELL, flat 1-D, the scaffold's SAXPY
  pattern: pure map — three coalesced reads, one coalesced write, no interaction between threads

DWA SCORE (kernels.cu: dwa_score_kernel)         — thread-per-SAMPLE, 4096 threads (08.01's MPPI shape)
  pattern: sampling rollout — reused for SCORING/argmin instead of a softmin control blend
  memory : pose/goal/window bounds are UNIFORM (kernel arguments, broadcast-cheap, like 08.01's
           u_nom); master[] reads are DATA-DEPENDENT per thread — the fourth point on the
           repo's memory-access spectrum: 09.01 __constant__ broadcast -> 08.01 mixed uniform+
           coalesced -> 07.09 divergent FIXED-offset reads -> HERE, divergent DATA-DEPENDENT
           reads (every thread's arc goes somewhere different; no coalescing trick exists because
           the addresses are decided by physics, not thread index — an honest limit, not a bug)
```

No kernel in this project uses shared memory: the raytrace walk touches too few cells per thread to
amortize a load, the inflation gather's window is read-only and reused *across* threads only through
the L2 cache (a tile could cache the shared window — Exercise territory, not implemented, matching
07.09's own "measured honesty over speculative tiling" choice), and the DWA rollout's memory pattern
is inherently thread-private.

## Numerical considerations

- **Why the raytrace/inflation/fusion layers are byte-exact, not tolerance-compared.** Every
  operation in those three kernels is integer arithmetic: Bresenham stepping (no trig inside the
  walk — the one angle-to-cell conversion happens ONCE, on the host, before either the GPU or CPU
  path ever runs, producing the SAME integer endpoint both consume — see `simulate_lidar_scan`'s
  comment in `src/main.cu`), `atomicMax`/`std::max` combine (an exact integer reduction, not a
  floating accumulation), and inflation's squared-distance decay (`kernels.cu`'s file header derives
  why linear-in-`dist^2` was chosen specifically to avoid `sqrtf`/`expf`, whose GPU and CPU
  implementations are accurate but not bit-identical). The payoff, measured: **0/65,536 cells differ**
  between the GPU master costmap and the CPU oracle on the committed scenario's first tick.
- **Why DWA scoring is NOT byte-exact.** `dwa_score_kernel`/`dwa_scores_cpu` both call `cosf/sinf/
  atan2f` (device) vs `std::cos/sin/atan2` (host) — different, individually-accurate implementations
  that can differ in the last few ULPs, compounded over 20 chained RK4 steps per candidate. Measured
  worst relative deviation over 4096 samples: **2.188e-07**, checked against a documented tolerance of
  **1e-3** — ~4500x headroom, the same "the gate has enormous margin against real bugs while floating
  noise never gets close" story 08.01's MPPI verification tells.
- **Angle wrapping discipline (CLAUDE.md §12, SYSTEM_DESIGN.md §3.7).** DWA rollouts (both GPU and
  CPU) integrate `theta` UNWRAPPED — the heading term's `cos(bearing - theta)` is periodic and does
  not care. Only the PLANT step (`diffdrive_step_cpu`) wraps `theta` to `(-pi, pi]`, the project's
  single defined wrap point, mirroring 08.01's `cartpole_step_cpu` exactly.
- **The `int` obstacle layer.** CUDA has no native `atomicMax` for 1-byte types; this project spends
  4 bytes/cell (256 KiB at 256x256 — trivial) on the ONE layer that needs atomics rather than bury the
  mark/clear lesson under bit-packing (README Exercise 4 is the packed version).
- **`fmaf`/RK4 in the unicycle rollout.** Same integrator shape as 08.01's cart-pole (`k1..k4`, the
  standard blend); because `thetadot = w` is constant, RK4 integrates `theta` EXACTLY (a linear ODE
  has zero truncation error at any order) — only `x,y` accumulate the usual, small `O(dt^5)`-local
  RK4 error as `theta` curves the path.
- **Determinism.** No RNG anywhere in this project: the world, the scan, the costmap, and the plant
  are all deterministic functions of the committed seed-42 map and scenario. Release and Debug builds
  produced byte-identical stable output lines when measured (see `demo/expected_output.txt`'s header).

## How we verify correctness

Two independent GPU-vs-CPU checks (the §5 gate) PLUS a closed-loop behavioral check, because a
navigation stack can be numerically correct at every individual layer and still behaviorally wrong
(or vice versa) — the same reasoning 08.01 gives for running two distinct checks:

1. **Costmap gate (VERIFY COSTMAP) — byte-exact equality.** Tick 0's exact scan through both the GPU
   pipeline and `costmap_update_cpu`; every one of 65,536 master-costmap cells must match exactly.
   Chosen deliberately as equality, not tolerance, because the whole pipeline is integer arithmetic
   (§Numerical considerations) — a single mismatched cell would mean a real indexing, race, or
   decay-formula bug, not floating-point noise. Measured: 0 mismatches.
2. **DWA gate (VERIFY DWA) — relative tolerance.** Tick 0's dynamic window scored by both
   `dwa_score_kernel` and `dwa_scores_cpu`; worst relative deviation compared against 1e-3 (floor
   `max(1, |score|)`, the same scale-invariant comparison 08.01 uses for its rollout costs). Measured
   worst: 2.188e-07.
3. **Closed-loop success (RESULT).** The two gates above prove the MATH is right on one snapshot; they
   cannot prove the whole DECISION LOOP behaves — a mis-tuned weight, a sign error in the dynamic
   window, or a stale buffer could leave every individual score "correct" while the robot never
   reaches the goal, or worse, reaches it having driven through an obstacle. The closed loop is
   therefore ALSO checked end-to-end: goal reached within the step cap, **and** an independent
   re-check that every driven cell along the ACTUAL path stayed below the lethal threshold (`src/
   main.cu` step 7 — deliberately re-derived from the driven pose, not merely trusting the planner's
   own admissibility flag). Measured: goal reached in 288/500 steps, 0 lethal-cell entries, 0
   emergency brakes.

The scenario is fully committed (`data/sample/`) so the whole three-part check runs offline, and the
`demo/out/costmap.pgm` + `demo/out/path.csv` artifacts make tick-level and path-level behavior
inspectable, not just pass/fail.

## Where this sits in the real world

- **Nav2's `costmap_2d`** implements the same three-layer idea (`StaticLayer`, `ObstacleLayer`,
  `InflationLayer`) this project teaches, with one major architectural difference worth stating
  honestly: **this project fully recomputes the entire grid every tick** (all three kernels touch
  every relevant cell each time), while Nav2 maintains the costmap INCREMENTALLY — each sensor update
  only touches a bounded "rolling window" of cells near the robot and near what actually changed, and
  the inflation layer only re-propagates from cells whose lethal status actually flipped. The
  full-recompute approach this project takes is simpler to teach and to verify byte-exact (every cell
  has one well-defined value derived from this tick's complete inputs, with no "what was left over
  from three ticks ago" state to reason about) — and, at this project's scale (65,536 cells, a
  desktop GPU), it is fast enough that the incremental optimization is not YET necessary (the whole
  pipeline measures under a millisecond). Nav2 needs the incremental approach because real costmaps
  run on embedded compute at larger scales (rolling windows over a building-sized map) where a full
  recompute genuinely would blow the latency budget. This is the honest trade every full-vs-incremental
  systems decision makes: simplicity and provable correctness vs. asymptotic efficiency at scale —
  and it is a fair one to make explicitly, in either direction, once you understand both sides.
- **Nav2's `DWB` local planner** generalizes this project's fixed weighted-sum score into a pluggable
  stack of "critic" plugins (obstacle, goal-distance, path-alignment, oscillation, twirling...),
  each independently tunable and swappable — the production answer to "the tuning story is told, not
  hidden" this project's fixed four-term score only gestures at.
- **The global-planner gap** (§The math's local-minima discussion): production stacks pair DWA/DWB
  with a global route planner (Nav2's default is a lattice/NavFn A* variant) precisely so the local
  planner never has to discover topology, only track a known-feasible route reactively. 06.x
  (motion planning) is where this repository's global planners live; a fuller AMR chain would slot
  one upstream of this project (SYSTEM_DESIGN.md §4.1's Chain A composition map).
- **What the full version adds** beyond this teaching core: incremental costmap updates (above),
  a real robot footprint (not just an inflation radius), 3-D voxel layers for non-2-D obstacles
  (overhangs, shelving), multiple simultaneous sensor sources fused into one obstacle layer, and a
  global planner feeding DWA/DWB a route instead of a bare goal point.
