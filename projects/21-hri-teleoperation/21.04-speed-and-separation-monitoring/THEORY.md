# 21.04 — Speed-and-separation monitoring: depth streams → minimum-distance fields at frame rate (ISO/TS 15066 helper): Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2).
>
> **Didactic implementation — NOT a certified safety function.** Everything below explains what this
> project computes and why; none of it is an argument that the result is safe to trust with a real
> robot. See [`kernels.cuh`](src/kernels.cuh)'s header comment and `PRACTICE.md` §4 for the caveat at
> full strength.

## The problem — physics & engineering first

**Why speed-and-separation monitoring exists at all.** A person and a robot sharing a workspace can
collide. The two facts that make "keep them apart by a fixed fence" unnecessary — and "let them work
right next to each other" survivable — are (1) a human's reaction and evasion speed is bounded, and
(2) a robot's stopping distance, once it starts braking, is bounded and (usually) knowable. Speed-and-
separation monitoring turns those two bounds into a *live* distance threshold: keep the *measured*
clearance between person and robot always larger than the distance either of them could close before
the robot can safely stop. Shrink the clearance below that line, and the robot must slow down (buying
back reaction time) or stop outright (removing the risk entirely). This is the physical idea beneath
ISO/TS 15066's speed-and-separation-monitoring (SSM) function, which this project's `compute_Sp()`
computes a structurally-inspired, illustrative version of (kernels.cuh SECTION 6 — see the standard
disclaimer there and in `PRACTICE.md` §4 before reading any further number as authoritative).

**The physics of "how far can a person move."** A standing adult's peak voluntary walking/lunging
speed is on the order of 1.5–2 m/s; secondary literature summarizing ISO/TS 15066 commonly cites
1.6 m/s as *the standard's* default assumed human approach speed (kVHumanMax here) — a conservative
number meant to upper-bound ordinary reaching and stepping, not sprinting. This project's `T_r`
(reaction time) is the same physical idea from the *machine's* side: the total delay from "the
person's true position changed" to "the robot's control loop has issued a stop command" — camera
frame period, processing, and communication latency, chained (kernels.cuh's `kTReaction`).

**The physics of "how far can the robot travel before it actually stops."** A robot moving at
Cartesian speed `v_r` under a protective stop does not stop instantly: torque limits, motor/gearbox
dynamics, and the controller's own deceleration profile bound how fast it can shed kinetic energy.
This project models that with a single illustrative constant deceleration `a_stop` (`kAStop`), giving
stopping time `T_s = v_r / a_stop` — the simplest physically sound model (constant deceleration is
exactly what you'd derive from a constant maximum braking torque and a lumped effective inertia); a
real drive's deceleration profile is rarely perfectly constant (torque saturates, then the controller
ramps down — `PRACTICE.md` §1 discusses the real mechanism), but the *shape* of the physics — bigger
`v_r` costs more stopping distance, closable only by slowing down — is exactly what this constant-`a`
model teaches.

**The engineering frame.** This is a *sensing* problem wearing a control-theory formula: the entire
S_p calculation is worthless if the system cannot *measure* the clearance at the rate and accuracy the
formula assumes. That is why this project spends most of its code not on the formula (kernels.cuh
SECTION 6 is twenty lines) but on the depth-camera pipeline that measures d_min: rendering,
classification, robot self-filtering, and the min-distance reduction, all of which must run inside the
camera's own frame period (30 Hz here, `docs/SYSTEM_DESIGN.md` §1.1's camera band) for the S_p
formula's `T_r` term to mean what it claims to mean. A monitor that measures clearance correctly but
too slowly is not actually monitoring at the rate its own formula assumes — this project's README
"System context" makes that connection explicit.

## The math

**Depth-camera geometry (orthographic, top-down — the scenario's stated simplification).** A real
depth camera is a **perspective** projection: rays diverge from a single optical center, so a
farther object subtends a smaller image angle, and the camera sees around and behind near objects
differently depending on viewing angle. This project uses an **orthographic** top-down model instead:
every camera ray is a vertical line `x = const, y = const`, and the recorded depth at pixel `(px,py)`
is `H_cam - z_top(x,y)`, where `z_top(x,y)` is the *height of the highest surface* directly below that
ray and `H_cam` is the camera's mount height (`kCamHeight`). This is honest and standard for a
ceiling-mounted overhead monitoring camera at a modest cell size (SYSTEM_DESIGN.md's manipulator
work-cell reference), where the perspective distortion across a 4×4 m floor from 3 m up is modest —
but it is a real simplification (Exercise 5 names the perspective generalization), and it is the root
cause of this project's most interesting numerical finding (see "Numerical considerations" below):
an orthographic top-down camera can see a body's *top* surface and nothing below it, ever, regardless
of resolution.

**The capsule.** Every solid in this scene — every robot link, the human's torso, the human's arm —
is a **capsule**: the set of points within `radius` of a 3-D segment `A→B` (a *swept sphere*). Formally,
`Capsule(A,B,r) = { p ∈ ℝ³ : dist(p, segment(A,B)) ≤ r }`. Capsules are the standard rounded-link
collision primitive in robotics (Drake, MoveIt/FCL, cuRobo all use them) because they are cheap (a
handful of FLOPs to test) and because sweeping a sphere along a segment approximates a real link's
rounded silhouette far better than a box or a single sphere, at almost no extra cost.

**Point-to-segment distance (derived).** For point `P` and segment `A→B`, parametrize the segment as
`A + t·(B−A)`, `t ∈ [0,1]`, and minimize `f(t) = |P − (A + t·AB)|²` where `AB = B−A`. Expanding,
`f(t) = |AB|²t² − 2·(P−A)·AB·t + |P−A|²` — a quadratic in `t` with leading coefficient `|AB|² ≥ 0`
(convex, or degenerate flat if `A=B`). Setting `f'(t) = 0` gives the unconstrained minimizer
`t* = (P−A)·AB / |AB|²`. Because `f` is convex on all of ℝ, the constrained minimum over `[0,1]` is
obtained by **clamping** `t*` into that range: if the unconstrained minimizer already lies in `[0,1]`,
it *is* the constrained minimum (a convex function's constrained minimum over an interval containing
its unconstrained minimum is that same point); if it falls outside, convexity guarantees the
constrained minimum is at the nearer endpoint, which is exactly what `clamp(t*, 0, 1)` produces. The
nearest segment point is `C = A + t·AB`; the point-to-segment distance is `|P − C|`, and the
**point-to-capsule distance** is `max(|P − C| − r, 0)` (the capsule's surface sits `r` outside the
segment, so subtract the radius and floor at zero once `P` is inside the capsule). `kernels.cu`'s
`point_capsule_distance` and `reference_cpu.cpp`'s `point_capsule_distance_cpu` are line-by-line twins
of exactly this.

**Top-down capsule rendering (derived, exact for this project's capsules).** A capsule's surface at
horizontal query point `(x,y)` reaches its highest visible `z` where a sphere of radius `r`, centered
on the *nearest axis point in the relevant sense*, is tangent to the vertical ray through `(x,y)`.
kernels.cuh SECTION 1 constrains every capsule in this project's scenes to be exactly **horizontal**
(`A.z = B.z`) or exactly **vertical** (`A.x=B.x, A.y=B.y`), which makes "the relevant sense" unambiguous
and the formula exact, not approximate:
- **Horizontal** (`z` constant `= z0` along the axis): the nearest axis point to `(x,y)` in the
  ordinary 2-D sense (clamped point-to-segment projection, the same math as above with the `z`
  coordinate dropped) is well defined regardless of `z`, because `z` never varies along the axis. At
  horizontal offset `d` from that nearest axis point, the top of the swept sphere sits at
  `z0 + √(r² − d²)` (Pythagoras: a chord of the circular cross-section of radius `r`).
- **Vertical** (`x,y` constant along the axis): every axis point projects to the *same* `(x,y)`, so
  the footprint is a plain disc of radius `r`, and by the same Pythagorean argument the visible top
  sits at `B.z + √(r² − d²)` above the *higher* endpoint `B` (kernels.cuh's stated authoring
  convention), `d` = horizontal distance from the axis.

A **tilted** capsule's true top height is a genuinely harder problem: the axis point nearest `(x,y)`
in 2-D and the axis point whose swept sphere reaches highest at `(x,y)` are no longer the same point
in general, and finding the true maximum requires optimizing jointly over the segment parameter *and*
the vertical rise — this project's scene generator never produces a tilted capsule, precisely to keep
`capsule_top_at()` exact instead of iterative (README Exercise 5 sketches the general case).

**Segment-segment distance (the analytic ground truth; prior art, not original — see README "Prior
art").** For two segments `P1(s) = p1 + s·d1`, `s∈[0,1]` and `P2(t) = p2 + t·d2`, `t∈[0,1]`, minimize
`g(s,t) = |P1(s) − P2(t)|²` — a quadratic form in two variables. Setting `∂g/∂s = 0` and `∂g/∂t = 0`
gives the 2×2 linear system
```
[ a  −b ] [s]   [−c]        a = d1·d1,  b = d1·d2,  c = d1·r
[ b  −e ] [t] = [−f]        e = d2·d2,  f = d2·r,   r = p1 − p2
```
with closed-form solution `s = (b·f − c·e)/(a·e − b²)`, `t = (b·s + f)/e` when the denominator
`a·e − b²` (the segments' non-parallelism) is nonzero, followed by clamping `s,t` into `[0,1]` and
re-solving the reduced 1-D problem along whichever edge the clamp lands on (exactly Ericson's
`ClosestPtSegmentSegment`, reimplemented in `main.cu`'s `analytic::closest_seg_seg_distance`). The
**capsule-capsule distance** is then `max(segment_distance − r1 − r2, 0)`. This is used *only* by the
analytic ground truth (never by a kernel) — it needs no GPU acceleration (16 capsule pairs, a handful
of FLOPs each) and is deliberately run in **double** precision, more precise than the FP32 pipeline it
verifies.

**The protective separation distance, term by term** (kernels.cuh SECTION 6; the standard disclaimer
there and in `PRACTICE.md` §4 applies to every number below):
```
S_p(v_r) = v_h·(T_r + T_s(v_r))   +   v_r·T_r   +   C_reach   +   Z_detection   +   Z_robot
           \_____________________/     \______/     \_______/     \___________     \______/
             human can close this      robot closes    fixed        sensor            robot
             much ground during        this much        reach/     position           position
             the robot's reaction      ground during    intrusion  uncertainty        uncertainty
             + stopping time           its OWN           allowance
                                       reaction time
                                       (before it even
                                       starts stopping)

T_s(v_r) = v_r / a_stop            -- stopping time from a constant illustrative deceleration
```
Evaluated at `v_r = kVRobotFull` (0.8 m/s) gives `S_p_full` — the boundary below which the robot must
be at or below its **reduced** speed. Evaluated at `v_r = kVRobotReduced` (0.2 m/s) gives
`S_p_reduced` — the boundary below which the robot must be **stopped**. Computing `S_p` at two speeds
to get two boundaries (three zones: NORMAL / REDUCED / PROTECTIVE_STOP) mirrors how real multi-zone
SSM systems are structured — the physical logic is the same formula, applied once per operating
speed the robot is allowed to be in.

## The algorithm

Per frame (the numbered steps match `main.cu`'s SEQUENCE loop and `kernels.cu`'s three kernels):

1. **Scene generation** (`build_scene`, host, shared by every path): a single raised-cosine profile
   `reach_fraction(t) = 0.5·(1 − cos(2π t / T_total))` (0 at both ends, 1 at the midpoint, smooth
   everywhere) drives *both* the SCARA arm's three joint angles and the human's walk position — a
   deliberate scenario choice (not a claim that robots and humans move in synchrony) that gives this
   teaching demo one unambiguous closest-approach event instead of an arbitrary phase relationship.
2. **Render + classify** (`render_classify_kernel`, **map**): for every pixel, the tallest surface
   among the 8 robot capsules and the tallest among the 2 human capsules (via `capsule_top_at`, above)
   determine what the camera "sees"; the taller of the two — or the floor — is the recorded depth.
   Classification is one comparison: does the sensed height match the robot's *own known pose*,
   within a tolerance band (`kSelfFilterEps`)? If yes, ROBOT (the self-filter real systems need
   because a robot's commanded and true pose never match exactly — `PRACTICE.md` §1). If it's above
   the floor and not explained by the robot, HUMAN. This is exactly the two-part extraction the
   catalog bullet implies: background subtraction against the empty-cell baseline (`surface_z >
   kFloorEps`), and the robot's silhouette masked out via its known FK pose.
3. **Minimum-distance field** (`human_min_distance_kernel`, **map + reduce**): every HUMAN pixel is
   reconstructed to a 3-D point in-register (fused with the distance computation — no materialized
   point-cloud buffer, a deliberate trade discussed in "The GPU mapping" below) and scored against the
   8 robot capsules; a canonical shared-memory tree reduction collapses the frame to one `(d_min,
   closest_capsule)` pair.
4. **SSM decision** (`classify_raw` + `HysteresisFsm::step`, host): compare `d_min` to `S_p_full` and
   `S_p_reduced`. **Escalation** (toward a more restrictive state) applies *immediately* — zero-frame
   delay. **De-escalation** (toward a less restrictive state) requires the less-restrictive condition
   to hold for `kHysteresisHoldFrames` (5) *consecutive* frames before the state actually relaxes.
5. **Dense field** (`dense_distance_field_kernel`, **map**, once per demo): the same distance function
   applied to every pixel (not just HUMAN ones) for the visual clearance-field artifact.

**The false-stop / missed-stop asymmetry, argued.** Why should escalation and de-escalation behave
differently? Because the two kinds of error this system can make are not equally bad. A **missed
stop** — the robot keeps moving when it should have stopped — risks a collision: potentially
irreversible harm. A **false stop** — the robot stops or slows when it did not strictly need to —
costs *availability*: a wasted cycle, a production-line hiccup, money. Those are not symmetric costs,
so the *system's* asymmetry should not be symmetric either: **never delay the safe direction, only
ever delay the available direction.** Escalating immediately (0-frame delay) means a single bad frame
of measurement can never be the reason a stop was late; requiring `kHysteresisHoldFrames` consecutive
frames of a *genuinely* safe reading before relaxing means a single noisy frame near the boundary
cannot cause the robot to resume prematurely, either — the hysteresis exists entirely on the
*recovery* side, where a few extra frames of caution cost availability, not safety. This is the same
logic real SSM/PFL (power-and-force-limiting) systems and most industrial safety functions use
(debounce on recovery, never on trip) — this project's `kNoFalseStopMargin`/`kNoMissedStopMargin`
verification gates (kernels.cuh SECTION 7) exist specifically to check the asymmetry holds in
practice, not just in the code's intent.

## The GPU mapping

```
render_classify_kernel:       one thread per PIXEL, grid-stride, pure MAP.
                               reads: d_robot_capsules[8], d_human_capsules[2] (__constant__)
                               writes: depth[i], label[i] (coalesced — consecutive threads,
                                       consecutive linear pixel indices)

human_min_distance_kernel:    one thread strides several PIXELS (grid-stride), keeping its own
                               running (dist, capsule_id) minimum in REGISTERS, then a
                               shared-memory TREE REDUCTION (blockDim.x = 256, power of two)
                               collapses the block to one pair; kReduceBlocks (<=256) partial
                               results finish with a trivial HOST scan (mirrors 08.01's
                               "host finishes a small reduction" choice).

dense_distance_field_kernel:  one thread per PIXEL, grid-stride, pure MAP — same device
                               distance function as above, no reduction (every pixel's answer
                               is independent).
```

**Why `__constant__` memory for the capsules.** At most 10 capsules exist at once (≤ 640 bytes), read
identically by *every* thread in *every* launch this frame, and never written during a kernel's
execution. That is precisely `__constant__` memory's design point: a small, cached, broadcast-friendly
read-only region (64 KB budget on any CUDA GPU). This sits at the "tiny data, same address for every
thread, unchanged for the whole kernel" end of the memory spectrum project 08.01's THEORY.md names —
the same category as 09.01's per-launch model constants, one step more read-heavy than 08.01's uniform
global reads of `u_nom[t]`, and a world away from 07.09's necessarily-divergent Voronoi seed reads.

**Why the minimum-distance reduction is a genuinely good teaching example of a reduction.** MIN is
**exactly** commutative and associative in IEEE-754 floating point (unlike SUM, whose result depends
on evaluation order because floating-point addition is not associative). That means this kernel's
tree reduction, the host's final linear scan over block partials, and the CPU oracle's straight
sequential scan are all computing the *same* mathematical minimum, not merely *statistically similar*
ones — which is exactly why the VERIFY stage's d_min tolerance (1e-4 m) can be so much tighter than a
summed reduction (like a softmin weight sum) would ever allow, and why the measured worst-case
GPU-vs-CPU divergence (1.2e-7 m) is essentially pure single-ULP FP32 rounding, not a reduction-order
artifact.

**Why reconstruction is fused, not materialized.** The catalog bullet's "3-D point reconstruction"
step exists in this project as *inline arithmetic inside* `human_min_distance_kernel` and
`dense_distance_field_kernel` (`pixel_to_world` + `z = kCamHeight - depth[i]`), not as a separate
kernel writing an `(x,y,z)` point-cloud buffer to global memory and reading it back. This avoids one
full global-memory round trip per frame (40,000 points × 12 bytes = 480 KB, twice, at 30 Hz) at zero
loss of information — every thread that needs a 3-D point has everything it needs (its own pixel
index and the depth image) to compute one in registers. A design that *does* need a materialized point
cloud (visualization, ICP registration onto the cloud, feeding a downstream perception stage) would
reintroduce that buffer deliberately; README Exercise territory names the alternative (stream
compaction via atomics or Thrust's `copy_if`) for exactly that case, and discusses why the
non-deterministic point *order* an atomic-counter compaction produces is harmless here (the downstream
op is a MIN — order-invariant) but would matter for an order-sensitive consumer.

## Numerical considerations

**Precision.** The rendering and distance kernels run entirely in FP32 (`sqrtf`, `fminf`/`fmaxf`) —
ordinary CUDA numerics, no atomics, no race conditions (every thread's output pixel/reduction slot is
written by exactly one thread). The **analytic ground truth** (`main.cu`'s `analytic::` namespace)
runs in **double** precision throughout, a deliberate asymmetry: it exists to *check* the FP32
pipeline, so its own rounding error must be negligible next to what it is checking, not merely
"pretty good."

**Angle wrapping / quaternions:** not applicable here — the SCARA joint angles are pure inputs to
`sinf`/`cosf` inside `build_scene()`, never accumulated, differenced, or compared across a wrap
boundary, so there is no wrap-point discipline to get right (contrast 08.01's cart-pole, which has
exactly this issue and documents its single wrap point).

**Pixel quantization (derived).** The nearest pixel center to any given `(x,y)` is within
`dx·√2⁄2 ≈ 0.0141 m` (half the pixel diagonal; `dx = 0.02 m` here). Point-to-convex-set distance is
1-Lipschitz in the query point (`|dist(p,C) − dist(q,C)| ≤ |p−q|` for any fixed convex set `C`, a
standard property — moving the query point by `ε` can change its distance to `C` by at most `ε`).
Combined: if the pixel pipeline's reconstructed point sat exactly *on* the human capsule's true
surface at a location within half a pixel diagonal of the true globally-closest surface point, the
pipeline's reported distance could exceed the true minimum by at most `dx·√2⁄2`. That "if" is the
catch — see the next paragraph.

**The silhouette-visibility bound — a real finding, not a footnote.** The pixel-quantization argument
above assumes the pipeline can sample a point *close to* the true closest point on the human's
surface. An orthographic top-down camera cannot: it only ever reports the **highest** `z` at each
`(x,y)`, which for a capsule means only the **upper half** of its local circular cross-section is ever
visible, at any resolution. Early development of this project used a larger human torso radius under
which the *analytically* true closest pair was occasionally the torso's cylindrical **side** — a point
sitting well below the torso's visible top hemisphere, at the height of whatever robot part was
nearby (well below the torso's own head-height top) — a point the top-down renderer can *never*
produce, at any pixel resolution, because the entire straight section of a vertical capsule between
its two hemispherical caps is invisible from directly above. Measured consequence: `GATE D_MIN BOUND`
failed by roughly **5 cm**, an order of magnitude larger than the ~1.4 cm quantization term, and
*resolution-independent* — shrinking the pixel size would not have fixed it. This project's response
was to make the design honest in the right direction: `kHumanTorsoRadius` (kernels.cuh SECTION 4) is
chosen small enough that the **arm** capsule (horizontal, at a height close to the robot's own
operating heights, so its top-down-visible "equator" is close to the true closest point in practice)
is analytically the closer human part on every frame of the committed scenario — verified by the demo
itself, every run, not merely asserted once. The residual bound after that fix,
`kSilhouetteSagBound = 0.05 m` (kernels.cuh SECTION 7), is calibrated to *this* scenario's own known
height gaps (the arm sits within 0.05 m of every robot capsule it actually approaches) — not a
universal bound for an arbitrary height mismatch (README Exercise 4 asks for the general derivation).
Measured worst-case total overestimate with the fix in place: **0.029 m**, comfortably inside the
combined derived bound of **0.070 m** (`kPixelQuantBound + kSilhouetteSagBound + kDminBoundSlack`).
The broader lesson: a single overhead depth camera has a *real*, physically-grounded blind spot, and
production SSM systems address it with side-mounted or multiple fused viewpoints (`PRACTICE.md` §2) —
this project turned that limitation into a measured, gated, honestly-bounded number instead of
quietly designing around it.

**Reduction determinism (see "The GPU mapping"):** because MIN is exactly associative/commutative in
IEEE-754, the GPU tree reduction and the CPU's sequential scan compute the identical real number up to
per-operation FP32 rounding — the tightest GPU-vs-CPU agreement any reduction in this repository
achieves, and worth contrasting explicitly with a summed reduction (08.01's softmin weights, verified
with a much looser tolerance for exactly this reason).

## How we verify correctness

Three independent code paths, on purpose (`main.cu`'s file header explains why three, not the usual
two):

1. **GPU pipeline vs. CPU oracle** (the ordinary §5 gate): `kernels.cu`'s three kernels against
   `reference_cpu.cpp`'s line-by-line twins, on identical capsule geometry, at two frames (the far
   start and the scenario's designed midpoint). Catches indexing, threading, and formula bugs in the
   pixel pipeline itself. Tolerance 1e-4 m for depth/d_min/dense-field (FP32 rounding-order headroom);
   **exact** equality required for pixel labels (an integer classification has no rounding to excuse).
   Measured: 0 label mismatches, worst value divergence 1.55e-6 m.
2. **Pipeline vs. analytic ground truth** (four gates, all comparing the *pixel* pipeline's output
   against `analytic::scene_min_distance` — geometry evaluated directly from the scenario's closed
   form, touching no pixel, no depth image, no label, ever): this catches a different class of bug —
   "the pipeline agrees with itself but is measuring the wrong thing" (a systematic rendering bias, an
   occlusion the scenario should not have, a threshold compared against the wrong quantity). The four
   gates (`GATE NO-FALSE-STOP`, `GATE NO-MISSED-STOP`, `GATE TRANSITIONS`, `GATE D_MIN BOUND`) are
   documented individually in kernels.cuh SECTION 7 and README "Expected output"; all four are checked
   on every one of the 240 frames (not spot-checked), and the demo prints the actual measured
   margins/offsets every run rather than asserting silently.
3. **The scenario's own self-consistency** (informal, but real): `scripts/make_synthetic.py`'s
   docstring documents that the committed default parameters were chosen so the scenario produces a
   clean single approach-retreat cycle — and then the program *itself* re-derives and checks that
   claim analytically on every run (the transition-frame gate), so a future edit to the scenario that
   breaks the clean cycle fails loudly instead of silently producing a misleading demo.

## Where this sits in the real world

- **Certified area scanners and safety-rated SSM systems** (SICK, Pilz, and similar functional-safety
  vendors) occupy this project's slot for real: certified sensors (often 2D safety laser scanners or
  purpose-built 3D safety systems, not a general-purpose depth camera), redundant sensing channels,
  and outputs wired into a certified safety controller with a hardwired stop path — never a single
  GPU program's decision alone. `PRACTICE.md` §2–§4 grounds this comparison in hardware and regulation.
- **Research SSM/PFL systems** in the HRI literature explore richer sensing (multi-camera fusion,
  learned human pose/velocity estimation, dynamic reach-envelope prediction) than this project's
  static per-frame minimum distance — the natural next step beyond this teaching core, and exactly
  where a "measured human velocity" term (rather than this project's conservative constant
  `kVHumanMax`) would come from.
- **What full certification actually requires beyond a correct distance calculation**: certified
  sensors with documented failure modes and diagnostic coverage, redundancy (so a single sensor
  failure cannot silently disable monitoring), a validated stopping-time model measured on the *real*
  robot (not an illustrative constant deceleration), and a safety case built and audited against
  ISO/TS 15066 / ISO 10218 by people whose job that is — SYSTEM_DESIGN.md §6's regulatory-landscape
  map and hardware-architecture diagram are the orientation references `PRACTICE.md` §4 cites for all
  of this. This project computes one number in that much larger system, correctly and verifiably, and
  is honest that "correctly" and "certified" are not the same word.
