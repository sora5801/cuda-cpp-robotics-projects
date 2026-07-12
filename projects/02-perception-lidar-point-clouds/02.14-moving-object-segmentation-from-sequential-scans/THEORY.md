# 02.14 — Moving-object segmentation from sequential scans: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

**What a spinning LiDAR actually measures.** A 16-beam spinning LiDAR fires beams at fixed elevation
angles (`kBeamElevMinDeg + ring * kBeamElevStepDeg`, ring 0..15, -15..+15 degrees in this project) as
it rotates, sampling azimuth at `kAzimuthStepDeg` (1 degree) intervals. Each beam either returns a
range (time-of-flight to the nearest reflecting surface) or nothing (no surface within `kMaxRangeM`).
This is a physical measurement of the sensor's **instantaneous 3-D visibility boundary** — the set of
nearest surfaces in every sampled direction, "frozen" at capture time. Comparing two such boundaries,
captured moments apart, is the only way a LiDAR can ever "see" that something moved: it has no
brightness channel, no persistent feature appearance, just range. **Motion, in a range image, IS a
change in the range value seen along a fixed direction.**

**Why this must run ONLINE, from a SHORT window — the dual of 02.13.** This repository's sibling
project **02.13** ("dynamic point removal") solves an adjacent but genuinely different problem: given
a LONG history of scans (K=10 in that project, accumulated over a full mapping session), which map
VOXELS were only ever transiently occupied? It answers OFFLINE, after the fact, with the full history
available, and its unit of evidence is a per-VOXEL ledger (hit-count vs. free-space-pass-count)
accumulated over the whole run. This project answers a harder-in-latency question: given ONLY the
CURRENT scan and a SHORT window of `kMaxWindowM=4` immediately preceding scans, which points in the
CURRENT scan are moving RIGHT NOW? There is no long history to lean on — a robot's planner needs the
answer before the NEXT scan arrives (README "System context": 10-20 Hz, < 1 scan period). The
representation is also different: 02.13 works in 3-D VOXEL space (a dense grid the size of the whole
mapped area); this project works in 2-D RANGE-IMAGE space (a compact 16x360 grid the size of one
sensor sweep), because 02.12's thesis — cells that are neighbors in the image are neighbors on the
sensor's sphere of view — is exactly the structure this project needs to compare TWO scans cell-by-
cell with no search.

**Why comparison needs REPROJECTION, not a raw pixel diff.** The sensor itself usually moves between
scans (a robot driving, a car turning). A raw cell-by-cell diff of two range images captured from
DIFFERENT sensor positions would report "motion" at every static surface too, simply because the
sensor's own displacement changes every range value — exactly the same lesson **02.08**'s motion-
deskew project teaches for a single sweep, one level up: there, un-deskewed points blur because the
sensor moves WITHIN one sweep; here, un-reprojected COMPARISONS lie because the sensor moves BETWEEN
sweeps. The fix in both cases is the same rigid-transform algebra — 02.08 calls it "project point i's
own instant into a reference instant"; this project applies the identical formula with "instant" swapped
for "scan": reproject each previous scan's points into the CURRENT sensor's frame before comparing.
After reprojection, a residual at a cell tells you ONLY about a change in the WORLD, with the sensor's
own motion algebraically cancelled out.

**The disocclusion problem — physics of an occluding edge.** A rigid, opaque object blocks every
LiDAR beam that would otherwise pass through it, exactly the way it blocks light. When that object
moves, some beams that WERE blocked become unblocked (a "disocclusion": previously hidden background
is revealed) and some that WERE clear become blocked (a fresh occlusion). Both events look, to a
residual computed purely from range, EXACTLY like "this cell moved" — even though the revealed/newly-
hidden cell itself, physically, never moved at all; it is the OCCLUDER that moved. This is the SAME
structural ambiguity **01.21**'s scene-flow project documents on the camera side (its "Limitations"
section: occlusion/disocclusion boundaries are where brightness-constancy optical flow systematically
fails, and raising a confidence threshold does not fix it, because confidence measures local texture,
not correspondence validity). In LiDAR form, the ambiguity is: a residual at a WALL cell caused by a
car's shadow sweeping across it is geometrically indistinguishable, from a single comparison, from the
SAME wall cell genuinely being a mover. The standard mitigation — and this project's central
engineering lesson — is **residual consistency across multiple previous scans**: a one-off occlusion
event (present in only some of the M comparisons) is a weaker, less consistent signal than a genuine,
sustained mover (present in essentially all of them). "The math" and "The algorithm" below derive
exactly how much this buys you, and "Numerical considerations" reports the measured, sometimes
surprising, limits of that mitigation.

**Reprojection quantization — an engineering constraint, not an approximation choice.** A real
sensor's 16 beams exist at 16 FIXED elevation angles. When a previous scan's point is reprojected into
the current sensor's frame, its recovered elevation/azimuth is, in general, a continuous value that
does NOT land exactly on one of those 16 fixed beam directions (it was measured by a DIFFERENT beam,
at a DIFFERENT instant, possibly from a slightly different sensor position). This project's
`nearest_ring_for_elev_deg`/`az_bin_for_az_deg` snap it to the nearest cell — a real, physical
discretization step (not a numerical convenience), and the direct cause of why a thin object (this
project's POLE cohort, radius 5 cm) can vanish between adjacent azimuth samples, and why a slanted
static surface can show a small, nonzero residual purely from this snap, never from any real motion.

## The math

**Frames and notation.** SI units throughout; right-handed, x-forward/y-left/z-up body frames
(CLAUDE.md §12, SYSTEM_DESIGN.md §3.2). `T_world_sensor` = (position `p` in meters, unit quaternion
`q`, repo order `(w,x,y,z)`) — SYSTEM_DESIGN.md §3.3/3.4's convention, identical to 02.08's.

**Range-image geometry.** A point in the sensor's own local frame, `P_local = (x,y,z)`, has range
`r = |P_local|`, elevation `elev = asin(z/r)` (degrees), azimuth `az = atan2(y,x)` (degrees, CCW from
local +x). The organized cell is `(ring, az_bin)` with `ring = round((elev - elev_min) / elev_step)`
clamped to `[0,15]` and `az_bin = round(az / az_step) mod 360` — `kernels.cuh`'s
`cell_for_local_point()`, the single formula every stage in this project shares.

**Reprojection.** Let scan `j`'s pose be `(p_j, q_j)` and the current scan's pose be `(p_cur, q_cur)`.
A point measured in scan `j`'s own local frame, `P_local_j`, is first the same physical point in the
WORLD:

```
P_world = p_j + R(q_j) * P_local_j
```

then re-expressed in the CURRENT sensor's local frame by inverting the current pose's rigid transform:

```
P_in_current = R(q_cur)^-1 * (P_world - p_cur) = R(conj(q_cur)) * (P_world - p_cur)
```

(using `R(q)^-1 = R(conj(q))` for a unit quaternion). Substituting and expanding — **this is
token-for-token 02.08's `deskew_one_point` derivation**, with scan `j`'s pose playing the role of "the
point's own instant" and the current scan's pose playing the role of "the reference instant":

```
P_in_current = R(conj(q_cur)) * (p_j - p_cur)  +  R(conj(q_cur)) * R(q_j) * P_local_j
             = t_rel + R(q_rel) * P_local_j
     where  q_rel = conj(q_cur) (x) q_j   (relative rotation, current <- j)
            t_rel = R(conj(q_cur)) * (p_j - p_cur)   (relative translation, expressed in current's frame)
```

`kernels.cuh`'s `reproject_point_to_current()` implements exactly this two-line result. The re-
projected point's range and `(ring, az_bin)` cell are then recovered by the SAME range-image geometry
formula above (`cell_for_local_point`) — this is the "nearest-elevation snap" the problem section
names, applied to the ALREADY-reprojected point.

**The residual and its two-sided sign — derived, not asserted.** Fix a current-scan cell with range
`r_cur` (the CURRENT scan genuinely has a return there). Let `r_prev` be scan `j`'s REPROJECTED range
at that same cell (may not exist — see "insufficient evidence" below). Define the signed residual

```
residual_j = r_cur - r_prev
```

Two exhaustive cases, both physically grounded:

1. **`residual_j < 0` (current CLOSER than scan j's reprojected surface): ARRIVAL.** Something now
   occupies this line of sight CLOSER than whatever scan `j` (after accounting for the sensor's own
   motion) observed there. Either (a) scan `j` saw open background farther away — a mover has SWEPT
   INTO this direction (this project's `crossing_car` cohort, whose azimuth changes scan to scan), or
   (b) scan `j` saw the SAME mover, but farther away — the mover has moved CLOSER along this exact
   line of sight (the `oncoming_car` cohort, constructed to hold azimuth/elevation EXACTLY fixed
   relative to the sensor — see "How we verify correctness"). Both physical stories share one sign.

2. **`residual_j > 0` (current FARTHER than scan j's reprojected surface): DEPARTURE / REVEALED
   BACKGROUND.** Either (a) something that used to occupy this direction has fully LEFT, revealing
   permanent background behind it (a disocclusion event — the "problem" section's central hazard), or
   (b) the SAME mover that scan `j` saw here is now farther away along this exact line of sight (the
   `receding_car` cohort). Both share the opposite sign from case 1.

This is the "two-sided logic" — and it is exactly why a SIGN-ONLY test cannot, on its own,
distinguish "genuine departure of a mover" from "a static point revealed by something ELSE'S
departure" (case 2a vs 2b): both are geometrically the SAME event from one comparison's point of view.
`main.cu`'s `sign_semantics` gate proves the DERIVATION above is actually implemented (not
accidentally satisfied) by checking that the `oncoming_car` cohort — case 1(b) by construction —
predominantly shows negative residuals, and `receding_car` — case 2(b) by construction — predominantly
shows positive ones.

**Multi-scan evidence fusion.** Given the M included previous scans (nearest lag first —
`kernels.cuh`'s `kPrevScanIdx`), define the fused evidence at a cell as

```
fused = MIN_{j in window} |residual_j|      (over j with a VALID r_prev; -1 if none valid)
candidate_moving = (fused >= 0) AND (fused >= kDynamicThresholdM)
```

"The algorithm" below derives why MIN, specifically, is the disocclusion-resistant choice, and what it
costs.

**Deriving the threshold from range noise.** Each range measurement in this project's synthetic scene
carries independent Gaussian noise, `sigma_r = kRangeNoiseSigmaM = 0.02` m (README/data:
`RANGE_NOISE_SIGMA_M`). A residual is the DIFFERENCE of two independent noisy ranges (current and one
reprojected previous), so its noise-only standard deviation is

```
sigma_residual = sqrt(sigma_r^2 + sigma_r^2) = sigma_r * sqrt(2) ~= 0.0283 m
```

A pure noise-floor threshold at, say, 6 sigma would sit near 0.17 m. This project's operating
threshold, `kDynamicThresholdM = 0.20 m`, sits just above that noise-only bound — deliberately: as
"Numerical considerations" reports, the MEASURED residual spread on static structure in this scene is
dominated by REPROJECTION QUANTIZATION (the "problem" section's snap-to-nearest-beam effect), not by
range noise alone, so the operating value is chosen from the actual measured static-vs-mover
separation on this scene (the same "theoretical bound printed alongside the measured operating value"
honesty 01.21 practices for its own segmentation threshold), not from the noise formula in isolation.

## The algorithm

1. **Organize the current scan** (`O(n_cur)`, a scatter): one thread per current-scan point, encode
   `(range, index)`, `atomicMin` into its NATIVE `(ring, az_bin)` cell (02.02/02.12 lineage). Then one
   thread per cell decodes the winner into a plain range image plus ground-truth payload.
2. **Reproject each of the M=4 previous scans** (`O(sum n_prev_j)`, four independent scatters): one
   thread per point in scan `j`, apply `reproject_point_to_current`, recompute its cell via
   `cell_for_local_point` (NOT its native cell — "The math"), `atomicMin`-race into scan `j`'s OWN
   range image. Fully independent across the 4 scans — no data dependency between them.
3. **Residual + MIN-fusion** (`O(kNumCells)`, a map): one thread per CURRENT cell, loop over the
   (at most 4) included previous range images, accumulate the signed residual, its sign, and the
   running minimum absolute value. `O(1)` work per previous scan per cell — the loop bound is the tiny,
   fixed `window_m <= 4`, not a data-dependent count.
4. **Range-image CCL cleanup** (`O(kNumCells)` edges + `O(sweeps * num_edges)` union-find, both tiny):
   build up to 2 forward-neighbor edges per candidate-moving cell (02.12's beta-criterion edge pattern,
   reused with a new predicate), then run the GENERIC lock-free GPU union-find (02.04/02.12) to
   convergence, then filter components smaller than `kMinMovingClusterSize` on the host (02.12's
   identical "small bookkeeping stays on the host" scoping).

**Complexity, serial vs. parallel.** Every stage above is `O(input size)` serially and `O(1)`
per-thread in parallel (steps 1-3), or `O(sweeps)` parallel rounds for step 4's union-find (sweeps
bounded by the largest connected component's diameter in the image, `kMaxUfSweeps=64` as a safety cap
— measured convergence on this project's demo: 2 sweeps). At this project's scale (`kNumCells=5,760`,
under 2,000 points per scan) every stage completes in low single-digit milliseconds on the GPU
(measured — README "Expected output"); the SAME kernels, unchanged, would scale to a real sensor's
100,000+ points/scan with only a change in launch grid size.

## The GPU mapping

Every kernel in this project follows the repo's `map`/`scatter` taxonomy directly (CLAUDE.md's
"map / reduce / stencil / scan / batched-solve / sampling" vocabulary): stages 1-2 are **SCATTERS**
(many threads write to data-dependent, possibly colliding destinations — resolved by the encoded
`atomicMin` "nearest wins" race 02.02 introduces and this project's whole domain reuses); stage 3 is a
pure **MAP** (each thread reads a small, FIXED number of neighbors — at most 4 previous images — and
writes one output, no collisions, no atomics); stage 4's edge-build is a **STENCIL** (each thread reads
its two forward image neighbors, exactly 02.12's depth-image edge kernel) followed by the union-find,
whose GPU mapping is REDUCTION-like but genuinely its own idiom: repeated **SCATTER-with-atomicCAS**
sweeps that monotonically shrink every element's parent pointer (02.04/02.12's proof, cited in
`kernels.cu`, not re-derived here) until a host-observed fixed point.

**Memory hierarchy.** Every array here is GLOBAL memory — no shared memory, no textures, no constant
memory. This is a deliberate, DOCUMENTED simplification distinct from several siblings (02.08 uses
`__constant__` for its tiny trajectory table; 02.13 uses it for its per-scan origin table): this
project's per-cell working set (a handful of `float`/`int` reads per cell) is small enough, and the
access pattern coalesced enough (adjacent threads read adjacent cells), that global-memory bandwidth
is not the bottleneck at this project's scale — README Exercise 2 (extend the window) is exactly the
experiment that would start to make shared-memory tiling of the residual-fusion stage worth profiling.

**Occupancy and bandwidth.** At `kNumCells=5,760` and under 2,000 points/scan, EVERY kernel here
launches far fewer threads than a single RTX 2080 SUPER's 46 SMs can keep resident — this project's
measured GPU-vs-CPU timings ([time] lines) are dominated by LAUNCH OVERHEAD, not compute or bandwidth,
exactly the same honest caveat 01.21's THEORY.md states for its own small demo scale. At a real
sensor's full point count (a Velodyne-class 32/64/128-beam unit, 30k-500k points/scan), the SAME five
kernels would become genuinely occupancy- and bandwidth-bound, and the launch-count itself (5 kernel
launches per scan, independent of point count) stays cheap relative to the per-point work — this is
precisely why the range-image representation scales the way 02.02/02.12 first taught.

**No CUDA library calls.** Every kernel here (including the union-find sweep) is hand-rolled — no
Thrust, no CUB, no cuBLAS (CLAUDE.md §1: no black boxes; README "Build" states the empty dependency
list explicitly).

## Numerical considerations

**Precision.** FP32 throughout — ranges, residuals, and the fused evidence are all `float`. No
accumulation chains long enough to need FP64 (contrast 02.06's ICP normal equations or 01.21's
covariance reduction, which DO accumulate in double): every value here is either a single subtraction
or a MIN over at most 4 terms.

**Inverse-trig domain guard.** `cell_for_local_point`'s elevation recovery, `asinf(z/range)`, is
mathematically defined only for arguments in `[-1,1]`; `z/range` is exactly in that range BY
CONSTRUCTION (`z` is one component of a vector whose norm is `range`), but float division can push the
ratio a few ULPs past `+-1.0` at near-vertical elevations. `kernels.cuh` clamps defensively before the
call (the same guard 08.01's rotation-angle-from-trace computation and 02.12's grazing-incidence
handling both apply before their own inverse-trig calls) — undefined behavior avoided at negligible
cost.

**Reprojection: GPU-vs-CPU agreement uses a TOLERANCE, not bit-exactness — and why.** Unlike stages 1
and 3 (pure data movement / arithmetic, verified bit-exact), stage 2's reprojection calls `sinf`,
`cosf`, `asinf`, `atan2f` — GPU device intrinsics and host `libm` implementations of these
transcendental functions are NOT guaranteed to agree to the last bit (a universal fact about
floating-point transcendentals across compilers/platforms, the same reasoning 08.01 states for its
own `sinf`/`cosf`-heavy rollout kernel). A ULP-level disagreement can, in principle, flip which of two
nearly-tied `(ring, az_bin)` cells a reprojected point snaps into. `main.cu`'s VERIFY stage (b)
therefore compares with a 5 mm tolerance and allows up to 2% of populated cells to differ — MEASURED
on this project's committed scene at **0.00%** (0 of 6,972 populated cells), with the single worst
observed deviation at `3.8e-6 m` — the tolerance carries roughly 1,300x headroom over what is actually
observed, wide enough that no ordinary platform/architecture difference should ever trip it, tight
enough that an indexing or sign-convention bug would still be caught immediately.

**Pose-error sensitivity — the honest system dependency.** MOS quality is BOUNDED BY localization
quality, and the coupling is direct and derivable from "The math" above: if scan `j`'s pose carries an
error `delta_p` (meters, world frame), the reprojected point `P_in_current` shifts by approximately
`R(conj(q_cur)) * delta_p` (to first order, treating the error as small relative to scene scale) —
i.e., EVERY point reprojected from scan `j` picks up a roughly CONSTANT (per-scan) offset. For a
surface nearly perpendicular to the sensor's line of sight, this offset projects almost entirely into
a spurious RANGE residual — precisely the mechanism that would degrade `static_precision` in the real
world: a localization drift of just a few centimeters between scans is comparable to this project's own
noise-derived threshold (`kDynamicThresholdM=0.20 m`), and a drift approaching that threshold would
start flagging genuinely static, well-observed structure as moving, exactly the false-positive failure
mode `static_precision` measures. This project's demo does not inject synthetic pose error (README
Exercise 4 names the natural follow-up experiment); the derivation here states the mechanism precisely
so the coupling is understood, not asserted — PRACTICE.md §3 discusses the operational consequence
(this module's accuracy is only as good as the localization feeding it).

**Determinism.** No RNG at runtime — the only randomness anywhere in this project is the OFFLINE,
fixed-seed (42) synthetic data generator. Every GPU kernel here uses either a pure map (no ordering
dependence at all) or the encoded-atomicMin/union-find idioms, both of which are PROVABLY
order-independent in their final result (02.02's proof for the former, 02.04/02.12's for the latter,
both cited) — a rerun on the same GPU reproduces bit-identical results; a different GPU architecture
may differ only in the reprojection stage's transcendental-function ULPs, comfortably inside the
tolerance above.

## How we verify correctness

Two tiers, per CLAUDE.md's twin-independence ruling (`reference_cpu.cpp`'s file header):

**Tier 1 — per-stage GPU-vs-CPU twins**, each independently implemented (`kernels.cu` vs.
`reference_cpu.cpp`), on the REAL loaded data:

| Stage | Tolerance | Measured (this run) |
|-------|-----------|----------------------|
| Current-scan organize | bit-exact | 0/5,760 mismatches |
| Reprojection (per previous scan) | 5 mm, <=2% of populated cells outside | 0/6,972 outside (worst 3.8e-6 m) |
| Residual fusion (M=4, fed identical verified inputs) | bit-exact | 0/5,760 mismatches |
| CCL edge set | exact set equality | identical (248 edges) |
| CCL union-find partition | bit-exact | 0/5,760 root mismatches |

**Tier 2 — independent, ground-truth gates**, exercising the FINAL labels against truth the algorithm
never reads (`main.cu`'s five `GATE:` lines — README "Expected output" states every measured number).
The `sign_semantics` gate in particular is the closed-form/known-answer check this project's shared
pose algebra (permitted to be shared per the independence ruling — see `kernels.cuh`'s file header)
NEEDS: the `oncoming_car`/`receding_car` cohorts are constructed with an EXACT, analytically known
residual sign (radial motion relative to the sensor, by construction — `make_synthetic.py`'s
`_radial_car_box`), so a bug INSIDE the shared reprojection formula that happened to still pass the
GPU-vs-CPU twin comparison (because both paths share the SAME buggy formula) would still be caught
here, because ground truth was never computed FROM that formula.

## Where this sits in the real world

**LiDAR-MOS** (Chen, Mersch, Behley, Stachniss, IROS 2021 / RA-L 2021) is the production-lineage
ancestor of this project: it computes the SAME range-image residual representation this project
computes by hand, then feeds it to a small CNN (rather than this project's fixed threshold) to
classify each cell — the learned version absorbs exactly the reprojection-quantization and
disocclusion noise this project's THEORY derives analytically, instead of requiring a hand-tuned
threshold and a hand-chosen fusion rule.

**4DMOS** (Mersch, Guadagnino, Chen, Behley, Stachniss, 2022) extends this with a genuine spatio-
temporal (sparse 4-D convolution) architecture that remembers motion state across a LONGER window than
this project's fixed M=4 — the direct, production answer to this project's measured
`temporal_boundary` limitation (a mover that just stopped is invisible to MIN-fusion at any M).

**Removert** (Kim & Kim, 2020) is this project's own sibling **02.13**'s closest real-world analogue —
offline, multi-resolution range-image comparison for map cleaning, the OFFLINE dual this README and
"The problem" section name explicitly throughout.

**Scene flow** (see this repo's **01.21**) is the camera-side sibling of this project's central
disocclusion lesson: both domains hit the SAME wall — a residual/flow computed from a single pair of
observations cannot, by itself, distinguish "this point moved" from "this point's correspondence
became invalid because something else moved" — and both domains' production answers are the same
shape (learned features robust to the ambiguity, or additional temporal/semantic context) rather than
a purely geometric fix.

**Dynamic SLAM systems** (DynaSLAM, DS-SLAM and descendants) consume a mover/static mask structurally
identical to this project's output to REJECT dynamic-object correspondences before they corrupt a map
or pose estimate — README "System context"'s downstream "statics -> mapping" lane, and the reason
02.13 exists as this project's offline complement: a real stack often runs BOTH — an online filter like
this one to keep movers out of the map as it is built, and an offline cleanup like 02.13 to catch
whatever slips through.
