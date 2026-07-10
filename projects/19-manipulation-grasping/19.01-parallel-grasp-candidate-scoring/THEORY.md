# 19.01 — Parallel grasp-candidate scoring: antipodal sampling over point clouds: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

**A grasp is a contact-mechanics problem before it is anything else.** A two-finger parallel-jaw
gripper closes until each finger touches the object at one point (idealized — real fingertips have a
small contact patch, PRACTICE.md §1). At that contact, the finger can push the object (a NORMAL
force, along the surface's inward normal) and, thanks to friction, can also resist a small amount of
SLIDING (a TANGENTIAL force, in the surface's local plane). What the gripper *cannot* do is pull —
fingers push, they do not glue — so the normal force is constrained to be non-negative.

**Coulomb friction, precisely.** For a rigid point contact with friction coefficient `mu`, Coulomb's
law says the tangential force magnitude cannot exceed `mu` times the normal force:

```
|f_t| <= mu * f_n ,      f_n >= 0
```

Geometrically, this means the TOTAL contact force `f = f_n * n_hat + f_t` (n_hat = inward unit
normal) must lie inside a cone centered on `n_hat`, with half-angle `alpha = atan(mu)` — the
**friction cone**. A contact cannot apply ANY force outside that cone without the finger slipping.
This is the single physical fact this project's scoring kernel exists to test, twice per candidate
(once per contact).

**Why two contacts is the interesting case.** A single point contact can resist forces only within
its own cone — trivially insufficient to hold anything against gravity from an arbitrary direction. A
parallel-jaw gripper gives you exactly TWO point contacts (this project's `p1`, `p2`), and the
classical result this project implements (derived below) says: two frictional point contacts can
resist an arbitrary NET FORCE (not yet any torque — see "Numerical considerations" and README
"Limitations") if and only if the LINE connecting them lies inside BOTH friction cones. That single
geometric condition — "does the line p1-p2 point into both cones" — is `THEORY.md`'s heart and
`src/kernels.cu`'s `score_candidates_kernel`.

**The engineering frame.** In a real bin-picking cell (SYSTEM_DESIGN.md §2.2), grasp scoring runs
once per view, budgeted at well under 100 ms inside a few-hundred-millisecond pick cycle — fast
enough that it is not the pacing item (README "System context" measures this project's GPU cost at
1–2 ms). The friction coefficient itself is not a fixed physical constant a robot can look up: it
depends on the ACTUAL fingertip material (rubber, silicone, bare aluminum), the object's surface
finish, humidity, and dust — real cells characterize it empirically and keep a conservative margin
(PRACTICE.md §2). Range-sensor noise (this project's synthetic 0.3 mm axial sigma, `data/README.md`)
directly corrupts the estimated surface normal, which directly corrupts the friction-cone test — bad
normals are not a cosmetic problem, they can make an infeasible grasp look feasible.

## The math

**Notation.** Points and vectors are in meters, SI, object-local frame (README/SYSTEM_DESIGN.md §3.1
conventions). `p1, p2` — the two contact points. `n1, n2` — OUTWARD unit surface normals at `p1, p2`
(this project's normal-estimation orientation policy — see "Numerical considerations"). `mu` —
Coulomb friction coefficient (unitless). `alpha = atan(mu)` — the friction cone's half-angle (rad).

**Friction cone at a contact** (derived above): the set of forces a contact `i` can apply is

```
FC_i = { f in R^3 : angle(f, -n_i) <= alpha }
```

(`-n_i`, the INWARD normal, because the finger pushes INTO the object).

**The antipodal / force-closure condition for two contacts.** Let `u = (p2 - p1) / |p2 - p1|` be the
unit vector from contact 1 to contact 2 (this project's `axis`). Consider the pure "squeeze" force
pair: contact 1 pushes along `+u` (toward contact 2), contact 2 pushes along `-u` (toward contact 1)
— equal and opposite, so their NET force and NET torque about the line itself are automatically zero
regardless of magnitude, and scaling both by the SAME factor lets this pair resist gravity in the
`-u`/`+u` direction with arbitrarily large magnitude PROVIDED each direction lies inside its own
cone:

```
force closure (2-contact, squeeze) holds  <=>  angle(u, -n1) <= alpha  AND  angle(-u, -n2) <= alpha
```

This is exactly `theta1 <= alpha` and `theta2 <= alpha` in `GraspScore` (`src/kernels.cuh`), where

```
theta1 = angle(u,  -n1)      (does contact 1's push-direction lie in ITS cone?)
theta2 = angle(-u, -n2)      (does contact 2's push-direction lie in ITS cone?)
```

Nguyen (1988) proves the converse too — for a SPHERICAL (frictional point) two-contact grasp, this
squeeze pair is not just *a* way to resist force, it is essentially necessary: if the line `p1-p2`
falls OUTSIDE either cone, NO combination of forces at those two contacts (each individually
friction-cone-limited) can produce an equilibrium against an arbitrary applied force. That is why
"the line connecting the two contacts lies inside both friction cones" is stated as an iff, not just
a sufficient condition — the theorem `README`'s "Prior art" cites by name.

**Antipodal quality (the ranking score).** Among grasps that already pass the binary friction-cone
test, this project ranks by how well-OPPOSED the two surface normals are:

```
antipodal_cos = dot(n1, -n2)        (score field, GraspScore)
```

`antipodal_cos = 1` means `n1` and `n2` point EXACTLY opposite (the textbook antipodal pair on a
locally flat/round surface); it decreases as the surfaces "converge" at an angle. This is a fast,
purely geometric proxy for grasp quality — the full quantitative measure, the Ferrari-Canny
epsilon-quality (the radius of the largest ball centered at the origin that fits inside the grasp's
WRENCH space — force AND torque, over BOTH contacts' friction cones, not just this project's binary
test) is named honestly in README "Limitations" as the version this project does not implement (19.02
does).

## The algorithm

Per object (`src/main.cu` orchestrates; each stage's launcher is documented in `src/kernels.cuh`):

1. **Normals** — O(n) work per point (n = kPcaK-nearest-neighbor search, itself O(n) per point, so
   O(n²) total per object) — project 02.06's exact PCA + Jacobi pattern (§The GPU mapping expands the
   one policy change: outward orientation).
2. **Candidate generation** — for each of K=4096 candidates: hash-pick a contact `p1` (O(1)), then
   scan the WHOLE cloud once (O(n)) for the best antipodal partner along `p1`'s inward-normal ray —
   total O(K·n) per object.
   - The ray-proximity test, precisely (also documented at `generate_candidates_kernel`): a candidate
     partner `q` qualifies if its projection `t = dot(q - p1, -n1)` onto the ray lies in
     `[kSearchTMinM, kSearchTMaxM]` (5 mm – 130 mm — `kernels.cuh` derives both bounds from the
     objects' point spacing and dimensions), its perpendicular distance to the ray is at most
     `kSearchPerpTolM` (6 mm), AND its normal `n_j` satisfies `dot(n1, n_j) <= kGenConeCosThreshold`
     (`cos(140°)` — a coarse "roughly opposing" prefilter, looser than the real friction-cone test
     stage 3 applies). Among all qualifying points, the one with the SMALLEST perpendicular distance
     wins (ties broken by lowest index — deterministic, matches the CPU oracle exactly).
3. **Scoring** — for each candidate with a partner: O(1) friction-cone/width math, plus a SECOND
   O(n) clearance scan — total O(K·n) again. Three gates (`friction_ok`, `width_ok`, `clearance_ok`)
   AND together into `feasible`; `score = feasible ? antipodal_cos : kRejectedScore` (a sentinel
   strictly below every valid `antipodal_cos in [-1,1]`, so a descending sort always ranks every
   feasible candidate ahead of every infeasible one).
4. **Ranking** — `std::sort` by `score`, descending, on the host; keep the top `kTopM = 10`.

**Serial cost** for one object: O(n²) [normals] + O(K·n) [generation] + O(K·n) [scoring] ≈ for the
cylinder (n=9000, K=4096): 81M + 37M + 37M ≈ 155M elementary point/candidate operations.

**Parallel cost**: normals is n independent problems, generation and scoring are each K independent
problems — on a GPU with enough resident threads to cover the larger of `n` and `K`, wall-clock time
is dominated by ONE object's slowest single-thread inner loop (still O(n) per thread), not by the
serial total — exactly the transformation every "thread = independent problem" project in this repo
makes (08.01's rollouts, 02.06's correspondence search, this project's three kernels).

## The GPU mapping

```
Kernel 1 (normals):     one thread = one cloud point j.       grid = ceil(n/256), block = 256
Kernel 2 (candidates):  one thread = one candidate k.          grid = ceil(K/256), block = 256
Kernel 3 (scoring):     one thread = one candidate c.          grid = ceil(K'/256), block = 256
                                                                (K' = K, or K + 12 adversarial for the box)
```

**Memory:** no shared memory anywhere in this project (contrast 02.06's `build_normal_system_kernel`,
which genuinely needs a block-level reduction because many SOURCE points sum into ONE shared 6×6
system). Every kernel here computes a fully independent per-thread answer that is simply WRITTEN to
global memory — the natural stopping point once a problem does not need cross-thread communication.
Registers hold each thread's whole per-point/per-candidate working set (normals: ~64 registers for
the k=16 neighbor lists, the same honest occupancy trade 02.06 documents; candidates/scoring: a
handful of floats for `p1,n1,p2,n2,axis`).

**Coalescing and the "divergent ray search" question.** Every thread in `generate_candidates_kernel`
scans the SAME loop bound `n` — no early exit — so at loop step `m`, every lane in a warp reads
`xyz[m*3..]`, the SAME address: a broadcast, not a gather, exactly like 02.06's correspondence
search. There is therefore NO warp-level memory divergence, even though every thread's RAY points in
a completely different direction (different `p1`, different `n1` per thread) — the loop bound is
uniform, only the per-iteration ARITHMETIC RESULT differs, and that is a straight-line predicated
branch (`if (qualifies) update best`), not a control-flow divergence.

**The honest load-imbalance story.** Uniform loop bound is not the same as uniform USEFUL work.
Measured on this demo's box object: 4020 of 4096 candidates (98.1%) found SOME partner, but only
2766 (67.5%) were fully feasible — every thread pays the SAME O(n) cost whether it lands a great
grasp in the first few loop iterations or finds nothing usable at all. An EARLY-EXIT version (stop
scanning once a "good enough" partner is found) would cut the AVERAGE work per thread, but at the
cost of reintroducing genuine per-thread divergence (different threads finish at different loop
counts, so a warp's slowest lane sets its pace anyway) — the same trade-off 02.06 documents for its
brute-force correspondence search. README "Exercises" names the accelerated (spatial-grid) version
as the real fix for BOTH the average-cost and the worst-case-cost problems at once.

**Why no reduction kernel.** Unlike 02.06 (many points summing into one shared least-squares system)
or 08.01 (many rollouts blended by a softmin weight), this project's candidates are independent
FINAL ANSWERS — nothing downstream needs to combine two candidates' scores into a third number. Top-M
selection is therefore a plain sort, not a reduction; `kernels.cuh`'s launcher comments and README
"Prior art" (the 12.01 NMS parallel) explain why that stays on the host at K~4096.

## Numerical considerations

- **Normal orientation — the inward/outward disambiguation problem.** PCA determines a normal's AXIS
  (the eigenvector), never its SIGN — `n` and `-n` are equally valid principal directions. Some
  external reference is always needed to pick one. Project 02.06 (scanning the INSIDE of a room)
  orients every normal TOWARD its reference point (the room's interior). This project scans the
  OUTSIDE of solid, convex objects, and every grasp formula here is written in terms of "the inward
  normal is where a finger pushes" — the natural convention is therefore OUTWARD (away from the
  object's centroid, which is guaranteed interior for every convex shape this project samples). One
  inequality flip (`estimate_normals_kernel`'s header comment shows exactly where) separates the two
  policies; get it backwards and every `theta1`/`theta2` in this project is off by `180° - theta`,
  silently inverting every friction-cone verdict. This is the general lesson: normal orientation is
  never "solved" by PCA alone — every consumer of a PCA normal must state its own sign policy.
- **Angle clamping.** `dot(unit, unit)` is mathematically in `[-1,1]`, but float32 rounding can push
  it a few ULPs past either bound; `acosf`/`std::acos` of an out-of-range argument returns NaN, not a
  clamped angle. Every dot product feeding an `acos` in this project (`cos_t1`, `cos_t2`) is clamped
  to `[-1,1]` first — the same defensive habit 02.06's `rotation_angle_deg` uses for its trace-based
  formula.
- **Tie-breaking determinism.** The candidate search keeps the STRICTLY smallest perpendicular
  distance seen so far (`<`, not `<=`) — on an exact tie, the first (lowest-index) candidate wins,
  deterministically, on both GPU and CPU. This matters because candidate generation is the one stage
  required to match the CPU oracle EXACTLY (see "How we verify correctness"), and a `<=` policy would
  make the winner depend on iteration order, which the GPU's arbitrary thread scheduling does not
  guarantee to match the CPU's sequential loop.
- **FP32 chain depth and cross-architecture stability.** Every dot product/cross product in this
  project is a handful of FMAs deep — nowhere near 08.01's 50-step RK4 chains — so GPU-vs-CPU
  agreement is tight (measured worst relative deviation: 5.96e-08, §"How we verify correctness"
  below). The one place non-associativity CAN matter is exactly at the search's tie-breaking
  boundary: two different GPU architectures (sm_75 vs. sm_86) computing the SAME dot product via a
  different FMA schedule could, in principle, pick a different winner on an almost-exact tie. This
  project's real synthetic data has no such near-ties in practice (measured: 0/4096 GPU-vs-CPU
  candidate mismatches — see below), but it is exactly why every stable CHECK line in the demo's
  output is textual PASS/FAIL, never a specific measured number (README "Expected output";
  `src/main.cu`'s "Output contract" header comment).
- **Determinism of candidate seeding.** `grasp_hash_u32` (`src/kernels.cuh`) is a pure function of
  `(seed, k)` — no shared state, no ordering dependency, so every candidate's `p1` selection is
  bit-identical regardless of which thread/warp/GPU computes it, or in what order. This is the
  project's alternative to 08.01's host-generated-and-uploaded noise stream: a counter-based hash
  needs no upload and no host-side generation step at all (README "Prior art" draws the cuRAND
  Philox parallel).

## How we verify correctness

Three independent checks (`src/main.cu`'s VERIFY stage), all on the box object — chosen because its
flat faces give PCA the cleanest possible normals to recover, making it the best isolator of a real
bug from expected floating-point noise:

1. **Normals: GPU vs. an INDEPENDENT CPU computation**, both starting from the same raw `xyz` (not
   fed the same intermediate — a genuinely independent double-computation). Compared by the ANGLE
   between the two unit vectors (not a componentwise tolerance, which would be meaningless near a
   sign flip): measured worst deviation **0.034°**, against a documented tolerance of 0.5° — headroom
   the residual `rsqrtf`-vs-`std::sqrt` and Jacobi-rotation-ordering differences (both machines run
   the SAME algorithm, SAME sweep count, but not bit-identical reciprocal-sqrt implementations) never
   come close to using up.
2. **Candidate generation: GPU vs. CPU, EXACT match required.** The CPU twin is fed the GPU's OWN
   (already-verified-close) normals — isolating this stage's check from any residual normals
   deviation, the same "feed the same intermediate to both paths" discipline 02.06 uses for its
   normal-system stage. Measured: **0 of 4096 index mismatches.** This is the one stage held to
   EXACT agreement rather than a tolerance, because `idx1`/`idx2` are integers — "close" is not a
   meaningful notion for an index, and the whole rest of the pipeline (scoring, ranking, the
   analytic gates) is only as trustworthy as the assumption that the GPU's chosen candidate pair is
   the SAME pair the CPU oracle would have chosen.
3. **Scoring: GPU vs. CPU, relative tolerance.** Fed the GPU's own candidates and normals (again
   isolating this stage). Compared on `width_m`, `antipodal_cos`, and `score` with a floor-1.0
   relative tolerance (the same shape 02.06/08.01/33.01 use): measured worst relative deviation
   **5.96e-08** against a documented tolerance of 1e-3 — five orders of magnitude of headroom, and
   **0 of 4096** boolean feasibility-flag disagreements.

**The analytic gates are the project's SECOND, independent layer of verification** — not a
replacement for GPU-vs-CPU agreement, but a check on a DIFFERENT axis: even a GPU/CPU-agreeing
pipeline could still implement the WRONG algorithm. Because this project's three objects have
closed-form correct answers, "is the top-10 list actually right" is directly checkable: box grasp
widths must fall within `kGateWidthTolM` (6 mm — sized from the search's own 6 mm perpendicular
tolerance plus the 0.3 mm sensor noise, `src/main.cu`'s constant comment) of 40 mm or 60 mm, with the
grasp axis at least 98% aligned to a coordinate axis; cylinder widths near 50 mm with the axis
roughly PERPENDICULAR to the cylinder's own axis; sphere widths near 60 mm. The 12 hand-picked
adjacent-face box candidates (built from REAL committed cloud points, not invented coordinates — see
`src/main.cu`'s `nearest_point_index`) are a NEGATIVE control: every one is confirmed rejected by the
friction-cone gate specifically (not merely absent from the top-10, which a bug could achieve by
accident) — measured: **12 of 12** rejected. Separately, the box's 100 mm axis (geometrically
antipodal, but wider than the modeled gripper's 90 mm stroke) is confirmed rejected by the WIDTH gate
specifically — measured: 614 of 4096 random candidates found that pairing and were correctly flagged
`width_ok = false`. Every one of these is a genuinely different failure mode a broken implementation
could exhibit, and each has its own dedicated check.

## Where this sits in the real world

- **GPD and Dex-Net's GQ-CNN** replace this project's hand-built antipodal search + friction-cone
  score with a LEARNED grasp-quality network — but both were trained against (Dex-Net) or validated
  with (GPD) analytic force-closure/antipodal labels not far from what this project computes exactly.
  A learned scorer earns its keep on messy, partially-observed real point clouds (occlusion, sensor
  noise, unknown object geometry) where a closed-form antipodal search has no ground truth to check
  against — this project's synthetic, fully-known objects are precisely the case where the classical
  approach is both correct AND checkable.
- **GraspIt!** implements the fuller wrench-space / force-closure machinery (multi-finger, arbitrary
  contact counts, full 6-DOF closure) this project deliberately narrows to the two-contact
  parallel-jaw case; its contact model and quality metrics are the direct ancestor of 19.02's project.
- **Ferrari-Canny epsilon-quality** (Ferrari & Canny 1992) is the standard QUANTITATIVE grasp-quality
  metric production planners rank by — the radius of the largest ball, centered at the wrench-space
  origin, that fits inside the convex hull of every contact's discretized friction cone mapped into
  wrench space. This project's `antipodal_cos` is a much cheaper, purely geometric stand-in that
  captures the SAME intuition (better-opposed normals resist more directions) without the wrench-space
  machinery; 19.02 implements the real metric.
- **Commercial bin-picking stacks** (e.g. industrial vision+grasp systems from major automation
  vendors) typically fuse THIS project's antipodal geometry with a learned or CAD-matched pose
  estimate, a reachability filter (19.08/09.05), and a collision-aware motion planner (06.07) into
  one cycle-time-budgeted pipeline (SYSTEM_DESIGN.md §4.2) — production robustness comes less from
  any single stage being smarter than this project's, and more from the WHOLE chain handling failure
  gracefully (a grasp that looks good geometrically but fails reachability just gets skipped in favor
  of the next-ranked candidate — this project's ranked top-M output is exactly what makes that
  fallback possible).
- **What the full version adds** beyond this teaching core: multi-finger and suction contact models
  (19.06), learned quality scoring (GPD/Dex-Net-style), full wrench-space epsilon-quality (19.02),
  spatial-acceleration structures for the search (k-d-trees, not brute force), and closed-loop
  re-grasping when the first attempt fails.
