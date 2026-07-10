# 11.01 — GPU LiDAR simulator: BVH raycasting + beam divergence, intensity, dropout noise: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2).

## The problem — physics & engineering first

**How a real spinning LiDAR works.** A mechanical spinning LiDAR (Velodyne/Ouster/Hesai-class — the
family this project models) mounts `CHANNELS` laser/detector pairs at fixed elevation angles on a head
that rotates about a vertical axis. Each laser fires a short (nanosecond-scale) infrared pulse; a
photodetector (APD or SPAD) waits for the reflected pulse and times its round trip. One full mechanical
rotation is one FRAME; the rotation is divided into `AZIMUTH_STEPS` firing slots, so a real sensor emits
`channels x azimuth_steps` pulses per rotation — for a Velodyne HDL-32E-class 32-channel unit spinning
at 10 Hz, that is on the order of 700,000 points/second, matching this project's own 32 x 1,024 =
32,768 beams/frame default almost exactly (the catalog bullet's own worked example).

**Time-of-flight ranging.** `range = c * delta_t / 2`, where `c ~ 3x10^8 m/s` is the speed of light and
`delta_t` is the measured round-trip time. Precision is set by how sharply the receiver can time the
pulse's leading edge against detector noise and pulse-shape jitter — physically, this is where this
project's `RANGE_NOISE_BASE_M`/`RANGE_NOISE_PER_M` (centimeter-scale, growing slightly with range as the
returned pulse weakens — see the radiometry derivation below) come from: not a made-up number, but the
right ORDER OF MAGNITUDE for a real time-of-flight sensor's ranging noise floor.

**Beam divergence.** A real laser beam is not an infinitely thin ray: diffraction and the emitter's
optics give it a finite angular spread, typically 1–4 milliradians (mrad) full or half-angle depending
on the product (this project's `DIVERGENCE_HALF_ANGLE_MRAD,1.5` sits in that real range — an
illustrative, order-of-magnitude value, not a specific datasheet number). At range `R`, a half-angle
`theta` beam illuminates a footprint of radius `~ R * tan(theta) ~ R * theta` (small-angle) — a
few centimeters at 20 m for 1.5 mrad. Where that footprint straddles a depth discontinuity (an object's
silhouette edge against the background), a REAL sensor returns a MIXED or SMEARED value — part near
surface, part far surface, sometimes two separate returns. This project's honest scope: it approximates
"the closest surface in the footprint usually dominates the return" (a small bundle of jittered rays,
nearest-hit wins — see "The algorithm" below) and does **not** reproduce edge smearing or multi-return
splitting; README §Limitations says so plainly, and README Exercise 3 sketches the extension.

**Radiometry — why intensity goes as `cos(incidence) / range^2`, not `range^4`.** This is worth
deriving properly, because "LiDAR is inverse-square, radar is inverse-fourth-power" is a genuinely
common point of confusion. Let the emitter send total pulse power `P_t` into a beam of half-angle
`theta`. At range `R` the beam covers a footprint area:

```
A_footprint(R) ~ pi * (R * theta)^2        (small-angle cone cross-section)
```

If that footprint lands entirely on a surface larger than itself (true almost everywhere in this
project's scene, and true of the analytic ground-plane gate by construction), the WHOLE beam power
`P_t` is intercepted by the target, independent of `R` — the beam's *outgoing* leg does not lose power
to distance once it is fully captured by a footprint-sized patch of a continuous surface (contrast this
with a target *smaller* than the footprint, where captured power falls as `1/R^2` on the way out too).

The surface re-radiates that power diffusely. A Lambertian surface (this project's whole reflectance
model — one scalar, `albedo`, per material) has reflected radiance independent of viewing angle, so its
radiant intensity (power per steradian) in the direction back toward the sensor is:

```
I(theta_i) = (albedo * P_t * cos(theta_i)) / pi      (Lambert's cosine law; theta_i = incidence angle)
```

The receiver, an aperture of area `A_rx` at range `R`, subtends solid angle `Omega_rx = A_rx / R^2`
from the target. Received power is intensity times subtended solid angle:

```
P_r = I(theta_i) * Omega_rx = (albedo * P_t * A_rx * cos(theta_i)) / (pi * R^2)
```

**One factor of `1/R^2` total** — from the *return* leg only, because the outgoing leg's power capture
was range-independent once the footprint criterion holds. Lumping every constant (`P_t`, `A_rx`, `pi`,
detector gain, calibration) into a single `intensity_gain`, this is exactly this project's radiometric
model: `intensity = intensity_gain * albedo * |cos(incidence)| / range^2`, clamped to `[0,1]` for sensor
saturation. (Contrast with monostatic RADAR against a point-like scatterer: there the outgoing beam
does **not** get fully captured by a small target, so BOTH legs lose power as `1/R^2`, compounding to
the classic `1/R^4` radar range equation — a different physical regime the same "distance costs
signal" intuition gets conflated with.)

**Dropout — the physics behind losing a return.** A return is lost when the received power (the `P_r`
above) falls below the detector's noise floor, or when the surface is specular/retroreflective enough
to send the return somewhere else entirely (or, rarely, saturate the detector). `P_r` falls with range
(the `1/R^2` term above) and with grazing incidence (the `cos(theta_i)` term — a beam striking near
90 deg from the surface normal spreads its already-small footprint over an even larger illuminated
ellipse, and the diffuse reflection more of that light away from the receiver too). This project's
dropout model, `p = clamp(dropout_base + range_coeff*(R/range_max) + incidence_coeff*(1-|cos_theta|), 0,
1)`, is an honest EMPIRICAL fit to that qualitative SNR story — a monotonically increasing function of
range and of grazing incidence — not a derivation from a detector noise-equivalent-power spec sheet
(which would need real photodetector characterization data this project does not have and does not
claim to have).

**Multi-return — what this project omits, honestly.** A real pulsed LiDAR can digitize and report
SEVERAL returns from a single pulse (through foliage: leaves then ground; through a chain-link fence:
fence then wall behind it; in rain: raindrops then the real target). This project reports the single
NEAREST return per beam. README Exercise 3 sketches the extension (record the first two BVH hits along
the central ray instead of one); the full multi-return story — pulse-shape deconvolution, echo
discrimination thresholds — belongs to "Where this sits in the real world" below.

## The math

**Formal problem statement.** Given a triangle mesh `{T_0, ..., T_{n-1}}` in world coordinates, a
sensor pose `T_world_sensor = (R, t)`, and a scan pattern (`channels`, `azimuth_steps`, elevation
range, azimuth start), compute for every beam `(channel, azimuth_idx)` the range, intensity, hit, and
dropped flags defined by the pipeline below.

**Beam direction.** Channel `c` (0-indexed) has elevation linearly interpolated across
`[elevation_min, elevation_max]`:

```
elevation(c) = elevation_min + (c / (channels-1)) * (elevation_max - elevation_min)
azimuth(a)   = azimuth_start + a * (2*pi / azimuth_steps)

dir_sensor = ( cos(elevation)*cos(azimuth),
               cos(elevation)*sin(azimuth),
               sin(elevation) )                 (unit vector, sensor frame: x-fwd, y-left, z-up)

dir_world  = R * dir_sensor                     (R orthonormal, dir_sensor unit -> dir_world unit)
```

**Möller–Trumbore ray/triangle intersection (Möller & Trumbore, 1997).** A point on the ray is
`P(t) = O + t*D`; a point in the triangle is `v0 + u*e1 + v*e2` (`e1 = v1-v0`, `e2 = v2-v0`, barycentric
`u, v >= 0`, `u+v <= 1`). Setting them equal:

```
O + t*D = v0 + u*e1 + v*e2
=>  -t*D + u*e1 + v*e2 = O - v0 = T
```

Three scalar equations, three unknowns `(t, u, v)`. Writing this as a 3x3 linear system
`[-D, e1, e2] * [t, u, v]^T = T` and solving by Cramer's rule (each unknown is a ratio of 3x3
determinants), the textbook simplification uses the SCALAR TRIPLE PRODUCT identity
`det[a,b,c] = a . (b x c) = -a . (c x b)` to rewrite every determinant as one cross product and one dot
product, computed ONCE and reused:

```
pvec = D x e2          det   = e1 . pvec           (shared by every determinant below)
qvec = T x e1
u = (T . pvec) / det        v = (D . qvec) / det        t = (e2 . qvec) / det
```

Reject early: `|det| ~ 0` (ray parallel to the triangle's plane, or a degenerate triangle), `u` outside
`[0,1]`, `v` outside `[0, 1-u]`, or `t` outside `[tmin, tmax]` — each test can short-circuit BEFORE the
next determinant is even computed, which is why `kernels.cu`'s `moller_trumbore()` tests `u` right after
computing `pvec`/`det`, before ever touching `qvec`.

**AABB slab test.** A ray enters an axis-aligned box exactly when, for EVERY axis independently, its
parametric entry time is before its exit time on that axis, AND the three per-axis intervals overlap.
For axis `i`: `t1 = (box_min[i] - O[i]) / D[i]`, `t2 = (box_max[i] - O[i]) / D[i]`,
`lo_i = min(t1,t2)`, `hi_i = max(t1,t2)`; the ray hits the box iff `max_i(lo_i) <= min_i(hi_i)`
(intersected with the ray's own `[tmin, tmax]`). `kernels.cu`'s `aabb_hit()` implements exactly this.

**The dropout and radiometry formulas** are derived in full in "The problem" above; restated compactly:

```
intensity = clamp( intensity_gain * albedo * |cos(theta_i)| / range^2,  0, 1 )
p_drop    = clamp( dropout_base + dropout_range_coeff*(range/range_max)
                                 + dropout_incidence_coeff*(1 - |cos(theta_i)|),  0, 1 )
```

## The algorithm

**BVH construction — median-split by triangle COUNT.** `main.cu`'s `BvhBuilder::build()` recursively
subdivides a set of triangles (represented as a permutation array `tri_indices`, never the triangles
themselves) into a binary tree:

1. Compute the node's AABB (bounding every triangle in its subtree) and, separately, the AABB of just
   the triangle CENTROIDS.
2. If the triangle count is `<= kBvhLeafSize` (4), stop: this is a leaf.
3. Otherwise, pick the axis where the centroid AABB is widest, and partition `tri_indices[first,
   first+count)` around its EXACT MIDPOINT `mid = first + count/2` using `std::nth_element` — an
   average-case `O(count)` selection algorithm (not a full `O(count log count)` sort) that guarantees
   every triangle left of `mid` has a centroid-`axis` coordinate `<=` every triangle at or right of
   `mid`, without fully ordering either side.
4. Recurse on `[first, mid)` and `[mid, first+count)`.

**Why COUNT-based splitting (not spatial-midpoint splitting) guarantees a depth bound.** This is the
project's one genuinely load-bearing piece of math, so it deserves a real proof, not just an assertion.
Claim: after `d` levels of this recursion, every node holds at most `ceil(N / 2^d)` triangles, where `N`
is the total triangle count.

*Proof by induction.* Base case `d=0`: the root holds all `N` triangles, and `ceil(N/2^0) = N`. ✓
Inductive step: assume a node at depth `d` holds `count <= ceil(N/2^d)` triangles. Step 3 above splits
it into two children of `floor(count/2)` and `ceil(count/2)` triangles — by construction, `mid =
first + count/2` (integer division), so NEITHER child can exceed `ceil(count/2) <= ceil(ceil(N/2^d)/2)
<= ceil(N/2^{d+1})`. ✓ This holds regardless of WHERE the triangles sit in space — the split index is
chosen by triangle COUNT, not by a spatial threshold, so no arrangement of geometry can produce an
unbalanced split (contrast with a spatial-midpoint split, e.g. "everything left of the box's center
x-coordinate", which degenerates to `O(N)` depth on a scene where every triangle clusters on one side —
exactly the kind of pathological case a naive BVH tutorial warns about).

A node becomes a leaf once its count is `<= kBvhLeafSize`. Combining with the bound above: the maximum
depth `D` satisfies `ceil(N / 2^D) <= kBvhLeafSize`, i.e. `D <= ceil(log2(N / kBvhLeafSize))`. For this
project's committed scene (`N = 2264`, `kBvhLeafSize = 4`): `D <= ceil(log2(566)) = 10` — and the demo's
own `BVH:` line prints BOTH the guaranteed bound and the measured depth, and they are equal (10 = 10) on
the committed scene, i.e. the tree is exactly as balanced as the count-based scheme promises.

**Complexity.** Build: `O(N log N)` (an `O(count)`-average `nth_element` call at each of `O(log(N/leaf))`
levels, summing to `O(N)` work per level times `O(log N)` levels — the same shape as building a
balanced binary search tree). Traversal of one ray: in the well-balanced case, `O(depth + leaf_size)` —
walk down `~D` interior nodes (each an O(1) AABB test) to a leaf, test `<= kBvhLeafSize` triangles —
versus `O(N)` for a brute-force scan of every triangle. For this project's scene that is roughly
`10 + 4 = 14` real tests per ray against `2264` for brute force — the entire reason a BVH exists.

**Beam divergence.** For each beam, cast the central ray plus `subray_count` extra rays evenly spaced
in azimuth around a cone of half-angle `divergence_half_angle_rad` centered on the central direction:
build an orthonormal basis `(u, v)` perpendicular to the central direction (`make_basis()`: cross the
central direction with whichever world axis is LEAST aligned with it, normalize, cross again), then for
`k = 0 .. subray_count-1`:

```
phi_k     = k * (2*pi / subray_count)
subray_k  = normalize( central*cos(half_angle) + (u*cos(phi_k) + v*sin(phi_k))*sin(half_angle) )
```

— the standard cone-sampling parametrization (evenly spaced points on a small circle around the central
axis, tilted out to the cone's half-angle). Keep the NEAREST hit among the central ray and all subrays
(README/kernels.cuh's documented divergence approximation, physically motivated in "The problem" above).

## The GPU mapping

```
one thread = one BEAM (not one triangle, not one pixel)
grid = ceil(num_beams/256) x 256    (repo default; ragged tail guarded)

per thread:
  registers      : the traversal stack (int[64]), the running best_t/best_tri,
                    the F3 direction/origin math — everything private per ray
  global (read)  : Triangle[], Material[], BvhNode[], tri_indices[] — the
                    ENTIRE mesh + tree, read-only for the whole kernel
  global (write) : range[beam], intensity[beam], hit[beam], dropped[beam]
                    — ONE coalesced write per array per thread, at the very end
```

**Divergent traversal — the project's one genuinely new GPU idea.** Every earlier project in this
repository's control/estimation lineage (33.01, 09.01, 08.01, 02.06) maps one thread to one INDEPENDENT
FLAT computation: same number of RK4 steps, same number of correspondence checks, same instruction
sequence for every thread, just different data. A BVH traversal is different: two neighboring threads
(beam `k` and beam `k+1`, adjacent in a warp) point in DIFFERENT directions, so they walk DIFFERENT
root-to-leaf paths through the SAME tree — one might reach a leaf in 6 hops, its neighbor in 14, with a
completely different sequence of "was this node's box hit" decisions along the way. CUDA's SIMT
execution model runs all 32 threads of a warp in lockstep on the SAME instruction; when threads take
different branches (a hit here, a miss there; a leaf here, an interior node there), the hardware
executes the UNION of both paths, masking off threads that are not on the currently-executing branch —
this is WARP DIVERGENCE, and a tree traversal is close to the textbook case that produces it, because
the very SHAPE of the computation (how many nodes visited, in what order) is data-dependent per thread.

This is honestly costly relative to a flat map kernel — but it is not catastrophic here, for two
reasons this project's own numbers support. First, the depth guarantee above bounds how BADLY paths can
diverge: every ray visits at most `~10` interior decisions and `<=4` leaf triangle tests, so even in the
worst case no thread does more than a small, bounded amount of extra work. Second, the measured
GPU-vs-CPU speed-up (README §What this computes: ~200–280x on the committed scene) is strong indirect
evidence that divergence cost, while real, is far from erasing the benefit of parallelism at this
scene's scale — 32,768 independent (if unevenly-shaped) traversals still vastly outperform one CPU core
doing the identical work sequentially. (This project's shipped demo does not include an Nsight Compute
profiling run — see `../../../docs/BUILD_GUIDE.md` for how to profile any project's kernel yourself;
README Exercise 2 gives a concrete, measurable follow-up: disable divergence subrays and see the VERIFY
tolerance — and, if you profile it, the warp efficiency — both tighten.)

**Coalesced output, divergent middle.** Despite the divergent TRAVERSAL, the kernel's final writes to
`range[]`/`intensity[]`/`hit[]`/`dropped[]` ARE fully coalesced: thread `beam` writes exactly to index
`beam`, so a warp's 32 threads write 32 consecutive addresses in one transaction, regardless of how
differently they got there. This is worth internalizing as a general lesson: coalescing is about the
FINAL memory access pattern, not about whether the WORK that produced the value was uniform.

**No shared memory, no constant memory, no atomics.** Shared memory earns its cost only when threads in
a block REUSE or SHARE data — here, no two threads in a block need each other's node/triangle reads (a
thread's traversal path is its own). Constant memory's advantage is BROADCAST: many threads reading the
SAME address on the SAME cycle (08.01's `u_nom[t]` is the textbook example, and the middle point on this
repo's "same-address-read spectrum": 09.01's `__constant__` symbol -> 08.01's uniform global read ->
02.06's per-launch kernel parameter). Divergent traversal breaks that assumption at its root: different
threads are reading DIFFERENT nodes at any given moment, so there is no broadcast to exploit even before
asking whether the ~130 KB working set (triangles + nodes + index permutation) fits `__constant__`'s
64 KB budget at all (it does not). No atomics: unlike 05.01's marching-cubes triangle-append kernel,
every thread here owns its own fixed output slot — nothing is contended.

## Numerical considerations

- **Precision: FP32 throughout, with DOUBLE used deliberately at a few angle-sensitive spots.** Every
  beam's elevation/azimuth/cone trig (`sin`/`cos` of the scan-pattern angles) is computed in `double`
  and cast to `float` only at the end — not because FP32 `sinf`/`cosf` are imprecise in general, but
  because this project computes each beam's direction exactly ONCE (never chained over many steps, the
  way 08.01's RK4 integrates 50 times), so the cost of `double` transcendentals is negligible, and using
  it maximizes agreement between the GPU's device `sin`/`cos` and the CPU oracle's `std::sin`/`std::cos`
  — directly serving the §5 verification gate (09.01/08.01 make the same "double where it is cheap and
  it matters for cross-path agreement" trade at their own angle-sensitive spots).
- **IEEE-754 divide-by-zero is SAFE, not undefined behavior.** The AABB slab test divides by each ray
  direction component (`inv_dir = 1/dir`); a ray exactly parallel to an axis makes that component's
  `inv_dir` `+-infinity`. Standard C++ (and CUDA, without `--use_fast_math`, which this project never
  sets — CLAUDE.md §5) follows IEEE 754 float division semantics: `1.0f/0.0f` is well-defined
  `+infinity`, not a crash or a garbage value, and the slab test's `min`/`max` arithmetic handles that
  infinity correctly for free — the classic branch-free AABB test relies on exactly this property.
- **The VERIFY range tolerance is wider than this repo's usual 1e-3, for a specific, measured reason.**
  Intensity's worst measured GPU/CPU deviation is 1.95e-4 — ordinary FP32 chained-arithmetic drift, the
  same character as 08.01/02.06's own ~1e-6..1e-3 stories (rounding-order differences in a deterministic
  computation). Range is qualitatively DIFFERENT: this kernel's divergence bundle picks the ARGMIN range
  across up to 5 independent rays. Near a geometric silhouette edge, two of those rays can legitimately
  hit DIFFERENT surfaces at nearly-tied distances (say, a crate's near corner and the floor just past
  it) — an ulp-level rounding difference between the GPU and CPU paths (in the BVH traversal's AABB
  tests, or in Möller–Trumbore's determinant, or in the cone-sampling trig) can flip WHICH of the two
  near-tied candidates is reported as "nearest". When that happens, the reported range moves by the
  GEOMETRIC gap between the two surfaces — centimeters — not by a rounding ulp. This is a genuine
  DISCONTINUITY in the argmin-over-independent-samples function, not a numerical error budget in the
  traditional sense: measured on the committed scene, 5 of 23,340 hit beams (0.02%) exceed the repo's
  usual rel-1e-3 tolerance, with a worst case of 1.166e-2; `kVerifyRangeTol = 2e-2` in `main.cu` carries
  ~1.7x headroom over that measured worst case. A real indexing/logic bug would push error to `O(1)`
  across a large fraction of the frame (a systematically wrong triangle, not five isolated edge beams),
  which is why this remains a meaningful gate and not a rubber stamp — README Exercise 2 gives a direct
  way to observe the effect shrink to nothing when divergence is disabled.
- **Determinism.** Every beam's dropout/noise draws come from a PER-BEAM xorshift32 stream (08.01's
  exact generator, reseeded `base_seed + 1000003*(beam_idx+1)` instead of per control tick), so a given
  seed reproduces bit-identical hit/dropped decisions on a given machine — confirmed by this project's
  own VERIFY stage, which measures ZERO hit/dropped mismatches between the GPU kernel and the CPU oracle
  across all 32,768 beams. Box–Muller's `sqrt`/`log`/`cos` run in `double` on both paths (08.01's exact
  reasoning: the cheap way to keep FP32 tails well-behaved and to keep the GPU/CPU streams close).
- **Winding-order independence.** Every incidence-angle computation uses `|cos(incidence)|` (absolute
  value), not the signed cosine — a deliberate simplification (kernels.cuh, kernels.cu) that makes the
  radiometry and dropout models indifferent to triangle winding order, which in turn makes the synthetic
  scene generator simpler to get right (no back-face-culling bugs possible, because there is no
  back-face culling).

## How we verify correctness

Three independent kinds of check, because a raycaster can be numerically self-consistent (GPU matches
CPU) while still simulating the WRONG PHYSICS, and vice versa:

1. **The §5 GPU-vs-CPU gate (VERIFY stage):** the full 32,768-beam demo frame through both the GPU
   kernel and the CPU oracle (`lidar_raycast_cpu`, a deliberate line-by-line twin of every device
   function in `kernels.cu`). Hit/dropped must match EXACTLY (measured: 0/32,768 mismatches — strong
   evidence the two implementations agree on every DISCRETE decision, not just approximately); intensity
   within rel 1e-3 (measured worst 1.95e-4); range within rel 2e-2, justified above (measured worst
   1.166e-2, affecting 5/23,340 hit beams).
2. **Analytic gates (run through the CPU oracle alone — see `kernels.cuh`'s long comment on
   `lidar_raycast_cpu` for why that is sufficient once (1) has already proven GPU-CPU agreement):**
   dedicated, hand-built `SensorConfig`/`SensorPose` probes with EFFECTS DISABLED (`subray_count=0`,
   dropout/noise coefficients zeroed) isolate the pure geometry and radiometry against closed-form
   physics:
   - **Ground-plane range:** a single beam aimed at the open floor (elevation -0.20 rad, azimuth 0,
     sensor at height 1.5 m) must return `h / sin(|elevation|)`. Measured: 7.550235 m against a
     closed-form 7.550234 m — relative error 7.7e-8, essentially the float32 noise floor for this
     computation (see "The math" for the elevation-based derivation: `range = height / sin(|elevation|)`
     follows directly from the ray hitting `z=0` starting at `z=height` with vertical direction
     component `sin(elevation)`).
   - **Inverse-square intensity:** two straight-down (normal-incidence, `cos_theta = 1` exactly) beams
     at heights 1.5 m and 3.0 m must ratio `(3.0/1.5)^2 = 4` exactly (the radiometry derivation in "The
     problem" predicts this precisely, since `cos_theta` and `albedo` cancel in the ratio, leaving pure
     `1/R^2`). Measured: 4.000000.
   - **Dropout statistics:** 20,000 beams at an IDENTICAL, precisely known `(range, incidence)` (all
     pointed straight down at the same floor point, so every beam's THEORETICAL dropout probability `p`
     is identical, but each draws an INDEPENDENT RNG stream) give an empirical dropout rate that must
     fall within a `5*sqrt(p*(1-p)/N)` binomial standard-error bound of the theoretical `p` — the
     correct statistical test for "does an i.i.d. Bernoulli sampler match its configured probability",
     not an arbitrary tolerance. Measured: empirical 0.02405 vs theoretical 0.02375, bound +-0.00538
     (well inside).
3. **Frame-level sanity (from the GPU's own full-frame output):** hit fraction (measured 0.7123) and
   mean returned range (measured 11.127 m) of the REAL demo frame — the unroofed warehouse scene, real
   sensor pose, all effects on — must fall inside documented, MEASURED-with-margin bounds `(0.40, 0.95)`
   and `(1.0, 20.0)` m respectively. These are not closed-form predictions (unlike gates 2 above); they
   exist to catch a gross regression (e.g., a sign error that makes every beam miss, or a scene-loading
   bug that halves the triangle count) that the tighter analytic gates, by design, do not exercise.

## Where this sits in the real world

- **Isaac Sim's RTX LiDAR** (NVIDIA) is this project's most direct production descendant: OptiX
  hardware-accelerated BVH traversal (this project's software traversal, done in silicon), a
  physically-based sensor model with atmospheric attenuation and per-product intrinsics, and multiple
  return modes — everything this teaching core names as future work in one polished, GPU-native tool.
- **CARLA's `sensor.lidar.ray_cast`** and **Gazebo's `gpu_ray`/`gpu_lidar` plugin** occupy the exact same
  architectural slot (GPU raycast against a scene BVH from a spinning-scanner model) at production scope
  for autonomous-vehicle and general robotics simulation respectively — both are excellent next reading
  once this project's pipeline feels familiar, and both make different simplifying choices worth
  comparing against this project's own (e.g., how each handles beam divergence and multi-return).
- **NVIDIA OptiX** is the hardware/API layer underneath Isaac Sim's LiDAR: dedicated RT cores execute
  BVH traversal and ray/triangle intersection in fixed-function silicon, at a scale (millions of rays
  per frame, real-time) this project's software kernel does not attempt to match — but the ALGORITHM is
  the same one this project hand-builds (CLAUDE.md §5's "build your own BVH before touching OptiX"
  stance, completed by this project).
- **Project 07.03 (Linear BVH build + stackless traversal)** is this repository's dedicated GPU-side BVH
  CONSTRUCTION project (Morton-code LBVH, built and refit entirely on the device — essential for a scene
  with moving geometry, rebuilt every frame). This project's host-built median-split tree is deliberately
  simpler and is built ONCE (this project's scene is static); chaining this project's raycast kernel
  onto a 07.03-built tree is a natural, named extension (README Exercise 4).
- **Multi-return and waveform digitization** (recording the FULL reflected pulse shape, not just a
  single leading-edge time, and discriminating several echoes from it) is where real high-end LiDAR
  research and product differentiation increasingly lives — full-waveform LiDAR processing is its own
  substantial signal-processing subfield this project's single-nearest-return model does not enter.
- **FMCW (frequency-modulated continuous-wave) LiDAR** is a different physical principle entirely
  (measuring a beat frequency between a frequency-swept transmitted beam and its Doppler-shifted return,
  giving simultaneous range AND velocity per point, immune to interference from other LiDARs) —
  genuinely different hardware and math from the pulsed time-of-flight model this project simulates;
  `PRACTICE.md` §2 names it as a real, increasingly common alternative.
