# 05.01 — TSDF fusion (KinectFusion clone) + marching-cubes mesh extraction: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

**What a depth camera actually measures.** A depth sensor does not measure "distance to the nearest
object" the way a laser rangefinder measures one number — it measures a whole *image* of distances,
one per pixel, using one of two physical principles:

- **Time-of-flight (ToF, e.g. early Kinect, many industrial sensors):** an IR emitter fires modulated
  light; each pixel measures the phase shift of the return, which encodes round-trip time and hence
  range. Noise grows with distance² (signal falls off with the inverse-square law both ways) and with
  surface albedo/angle; multi-path reflections near concave corners produce systematic (not random)
  errors.
- **Structured light / active stereo (e.g. Kinect v1, RealSense D400 series):** a projector casts a
  known IR pattern; a second camera observes the pattern's disparity, and depth follows from stereo
  triangulation, `depth = f·baseline/disparity`. Precision degrades with `1/depth²` (a fixed pixel's
  disparity error maps to a range error that grows quadratically with range) and fails entirely on
  low-texture, highly specular, or IR-absorbing surfaces (black rubber, water, glass).

**What every real depth sensor shares:** per-pixel noise that grows with range, systematic bias near
depth discontinuities ("flying pixels" — a pixel straddling foreground/background reports a blended,
physically meaningless depth), and a valid range window (typically ~0.3–5 m for consumer RGB-D). None
of these are visible in this project's *rendered* depth (closed-form ray casting is noise-free by
construction — see "How we verify correctness" below), but every one of them is exactly why TSDF
*fusion* — averaging many noisy single-frame estimates into one confident field — is the right answer
in the real system this project teaches toward, and every one of them is a documented simplification
here (README §Limitations).

**The robotics task.** A mobile or manipulator robot needs a *dense* model of the space around it — not
just "is this point occupied" but a smooth, continuous surface it can mesh, plan collision-free motion
against, or hand to a human as a reconstructed scan. TSDF fusion is the standard way to build that
model *incrementally*, online, from a stream of noisy, partial depth views taken from different poses
as the robot (or a hand-held sensor) moves — which is exactly why it earns a permanent seat in the
autonomy stack's state-estimation/world-model layer (SYSTEM_DESIGN §1).

**Engineering constraints a real system imposes.** Memory is the first-order constraint: a dense voxel
grid is `O(N³)` in the number of voxels per axis — this project's 128³ grid at one `float` TSDF + one
`float` weight per voxel is `2 × 128³ × 4 bytes ≈ 16.8 MiB`, comfortable, but a room-scale reconstruction
at the same 2 cm resolution over a 10 m cube would need `500³ × 8 bytes ≈ 1 GiB` — the reason production
systems hash sparse voxel blocks instead of allocating the whole bounding box (README §Limitations, and
"Where this sits in the real world" below). Compute is the second: fusing a new frame must finish well
inside the sensor's frame period (30–60 Hz cameras, SYSTEM_DESIGN §1.1) or the map falls behind the
robot's motion — precisely the bottleneck the GPU removes (this project measures ~0.1–0.3 ms per frame
at 128³ voxels, three orders of magnitude inside a 30 Hz, ~33 ms budget).

## The math

**Signed distance function (SDF).** For a solid region `Ω ⊂ ℝ³` with boundary (surface) `∂Ω`, the
signed distance function is

```
sdf(p) = ± min_{q ∈ ∂Ω} ‖p − q‖        (+ outside Ω, − inside Ω, 0 exactly on ∂Ω)
```

`sdf` is a *scalar field* over all of space whose zero level set `{p : sdf(p) = 0}` **is** the surface.
This project's scene is chosen because its SDF has a closed form — a sphere of radius `r` centered at
`c`, unioned with a half-space (the ground plane, solid for `z ≤ z_plane`):

```
sdf(p) = min( ‖p − c‖ − r,   p.z − z_plane )
```

`min()` of two SDFs is the *exact* SDF of their union **only** where the two bodies' nearby regions do
not interact — precisely true here because the sphere floats `c.z − r − z_plane = 0.25 m` above the
plane, more than the `2·μ = 0.24 m` a truncated comparison would ever need (kernels.cuh documents the
exact numbers; this is why `scene_sdf()` in `main.cu` can be trusted as *real ground truth*, not an
approximation, everywhere this project checks it).

**Truncated SDF (TSDF).** Storing the *exact* distance to the nearest surface at every voxel, from
*every* frame's single noisy observation, would be both expensive (global nearest-surface queries) and
wrong far from the surface (a single depth pixel says nothing reliable about a voxel 2 meters behind
it — occlusion, not distance, dominates there). KinectFusion's fix is to **only trust distance near the
surface**: clamp every measurement to `[−μ, +μ]` for a small truncation distance `μ` (this project:
`μ = 0.12 m`, 6 voxels), and store the *normalized* value `F ∈ [−1, +1] = sdf/μ`. Outside the trusted
band, `F` simply saturates at `±1` — "confidently free" or "confidently unknown/behind," which is all a
single frame can honestly claim there anyway.

**The fusion rule.** For voxel `v` observed with instantaneous truncated estimate `f` and observation
weight `w_obs` (this project: constant `1`), the running weighted average update is

```
F_new(v) = ( F_old(v)·W_old(v) + f·w_obs ) / ( W_old(v) + w_obs )
W_new(v) = min( W_old(v) + w_obs,  W_max )
```

This is the same running-mean identity you would derive for combining two independent Gaussian
estimates weighted by inverse variance, specialized to constant per-observation confidence; the weight
cap `W_max` (this project: 64) keeps the average *adaptive* — a very old voxel's estimate can still be
revised by new evidence, at a floor of `1/(W_max+1)` influence per new frame, rather than being frozen
in forever (relevant for a moving scene; not exercised by this static one).

**Projective vs. true SDF (the approximation KinectFusion makes, and this project inherits).** The
*correct* per-voxel measurement would be the true Euclidean distance from the voxel to the nearest point
on the observed surface. KinectFusion instead uses the *projective* (a.k.a. "PSDF") approximation:
project the voxel into the depth image at pixel `(u,v)`, read `depth(u,v)`, and take

```
f_projective = depth(u,v) − z_cam(voxel)
```

i.e. the difference measured **along the camera's optical axis**, not along the line to the nearest
surface point. This is cheap (one projection, one texture read, one subtraction — no search), and it is
a *good* approximation exactly where the viewing ray is close to the local surface normal (near-normal
incidence). It is a *bad* approximation at grazing incidence: writing `θ` for the angle between the
viewing ray and the surface normal, the projective error grows like `1/cos θ` and diverges as
`θ → 90°`. THEORY's "Numerical considerations" section below shows this is not a footnote here — it is
the single largest source of measured error in this project's ground-truth check.

## The algorithm

**Per-frame integration** (one call per depth frame, any order — `launch_tsdf_integrate`):

```
for every voxel v (parallel):
    p_world  = voxel center of v                         # kVolOrigin + (i+0.5)*voxel_size
    p_cam    = T_cam_world * p_world                      # world -> camera
    if p_cam.z <= 0: skip                                 # behind the camera
    (u, v_px) = project(p_cam, intrinsics); round to nearest pixel
    if outside image bounds: skip
    d = depth(u, v_px)
    if d <= 0: skip                                       # sensor reported "no return"
    f = d - p_cam.z                                       # projective SDF (m)
    if f < -mu: skip                                      # confidently occluded: says nothing usable
    f = min(f/mu, 1.0)                                    # truncate + normalize to [-1, 1]
    (F[v], W[v]) = fuse(F[v], W[v], f, w_obs=1)            # the running weighted average above
```

Serial cost per frame: `O(N³)` voxel visits (`N = 128` here, `~2.1 M`); every visit is independent, so
the GPU parallel cost is `O(N³/P)` for `P` GPU threads in flight — the textbook **map** pattern.

**Mesh extraction — marching cubes** (`launch_marching_cubes`, one call after all frames are fused):

```
for every cell c (a cell = 8 neighboring voxel corners; parallel):
    if any corner unobserved (weight == 0): skip           # no surface claim outside sensed space
    cubeindex = bitmask of which corners have F < 0        # "inside" the surface
    if cubeindex == 0 or 255: skip                          # fully outside / fully inside: no crossing
    for each triangle (0-5) listed for this cubeindex in the case table:
        for each of its 3 edges:
            interpolate the zero-crossing point linearly along that edge
        atomically reserve 3 output vertex slots and write the triangle
```

Serial cost: `O((N-1)³)` cell visits (`127³ ≈ 2.05 M` here), each doing `O(1)` table-driven work — again
independent across cells, again a natural one-thread-per-cell GPU mapping, but this time the *output* is
variable-length per thread (0 to 5 triangles), which is the new pattern this project adds (see "The GPU
mapping"). The 256-case table itself is a **precomputed answer**, derived once (by Lorensen & Cline in
1987) to the combinatorial question "given these 8 corner in/out signs, which triangles approximate the
surface passing through this cell?" — reproducing that derivation by hand is a well-known but lengthy
exercise; using the published table (`src/mc_tables.h`, with provenance) is standard practice, exactly
like using a BLAS routine instead of hand-deriving GEMM (CLAUDE.md §1's "no black boxes" is satisfied by
*documenting* the table's origin and *verifying* its output, not by re-deriving it).

## The GPU mapping

**Kernel 1 — TSDF integration, voxel-parallel:**

```
thread v = blockIdx.x*blockDim.x + threadIdx.x   owns voxel v
grid = ceil(N^3 / 256) blocks of 256 threads       (N=128 -> 8192 threads, 32 blocks)

memory per thread:
  registers : voxel world/camera coordinates, the update arithmetic (~20 regs)
  global    : tsdf[v], weight[v]   — READ-MODIFY-WRITE, but only by thread v: no atomics,
                                      no races (every voxel has exactly one owning thread)
              depth[pixel]         — a GATHER whose locality mirrors scene geometry:
                                      neighboring voxels project to neighboring pixels, so a
                                      warp's 32 lanes touch a compact image region and the L2
                                      cache serves most of it — not perfectly coalesced (that
                                      would need the THREADS, not the voxels, ordered by
                                      projected pixel), but far from random
  by value  : intrinsics K, pose T — every thread reads the SAME struct: a broadcast, the
                                      same access-pattern lesson 09.01 teaches with
                                      __constant__ memory, here achieved for free because
                                      kernel arguments live in fast, uniformly-read parameter
                                      space
```

Divergence is spatially coherent and cheap: whole warps of voxels behind the camera, outside the image,
or beyond the truncation band exit together (nearby voxels tend to share visibility), so the early-exit
guards cost little in practice.

**Kernel 2 — marching cubes, cell-parallel with atomic append:**

```
thread c = blockIdx.x*blockDim.x + threadIdx.x   owns cell c
grid = ceil((N-1)^3 / 256) blocks

memory per thread:
  global    : 8 corner tsdf + 8 corner weight loads (x-neighbors coalesced; y/z neighbors
                                                       land kVolN and kVolN^2 floats apart —
                                                       the same honest stencil cost 07.09 pays)
              triangle writes at an ATOMIC-RESERVED offset (see below)
  constant  : c_tri_table[256][16], c_edge_corner_a/b[12]  — 256 possible corner patterns,
                                                              looked up by cubeindex. Warps
                                                              whose 32 threads share a
                                                              cubeindex get a broadcast from
                                                              the constant cache; warps that
                                                              disagree serialize across the
                                                              distinct rows touched — the same
                                                              tool 09.01 uses for pure-uniform
                                                              reads, here handling narrow,
                                                              cell-dependent divergence instead
  registers : the 8 corner values, the cubeindex, interpolation math
```

**The atomic-append output pattern (new in this project).** Each surface cell must emit a *variable*
number of triangles (0 to 5), and thousands of cells run concurrently — there is no way to know in
advance which output slot a given cell's triangles should land in. The one-pass fix:
`slot = atomicAdd(tri_count, 1)` reserves a globally unique output index for each triangle the instant
before it is written; every triangle from every cell gets its own slot with zero collisions, and the
final value of `tri_count` is the exact total. The price is **nondeterministic order**: which cell's
atomic wins the race to slot `k` varies run to run (thread scheduling is not guaranteed), so the
triangle *buffer contents in position order* are not reproducible — only the triangle **set** (as a
collection) and **count** are. This project verifies exactly that invariant (see "How we verify
correctness"), not byte-identical output.

**The honest, better-for-production alternative — two-pass count-then-scan:** pass 1 classifies every
cell and writes its triangle count (0–5) to an array; an exclusive prefix sum (scan) turns those counts
into stable, deterministic output offsets; pass 2 re-classifies (or reuses cached results) and writes
each triangle to its now-known offset. This costs classifying every cell twice (or caching the case
index) plus a scan, in exchange for deterministic output order and zero atomic contention — the
approach production libraries (VTK-m, NVIDIA's own marching-cubes sample) take. This project teaches the
one-pass atomic version because it is the *minimal correct pattern*, and because the nondeterminism it
introduces is narrowly scoped (order only) and directly demonstrable by the verification strategy below
— README Exercise 4 is exactly this upgrade.

## Numerical considerations

- **The truncation band `μ` is a real design knob, not a magic number.** Too small and a single noisy
  observation can push a voxel across zero when it should not (surface holes, "swiss cheese" meshes);
  too large and unrelated nearby surfaces (this project's sphere and plane; in general, thin structures)
  start contaminating each other's estimates because their trusted bands overlap. `μ = 0.12 m = 6
  voxels` here follows the "a handful of voxels" KinectFusion convention; `kernels.cuh` computes and
  comments the exact clearance margin this scene provides against that failure mode.
- **The weight cap `W_max` bounds "staleness."** Without a cap, a voxel observed 1000 times would need a
  1001st observation of nearly the same magnitude to move its average at all — appropriate for a static
  scan, wrong for anything that changes. `W_max = 64` here keeps every new frame worth at least `1/65`
  of the running estimate.
- **Determinism by construction — the fmaf contract.** Every multiply-add in both the integration kernel
  (`kernels.cu`) and its CPU twin (`reference_cpu.cpp`) is spelled as an *explicit* `fmaf`/`std::fmaf`
  call rather than left to the compiler's discretion. `nvcc` would likely contract `a*b+c` into an FMA
  automatically; MSVC's host compiler, by default, would **not** — leaving the two paths one rounding
  step apart at every accumulation, compounding over the projection's several chained multiply-adds.
  Spelling both explicitly makes the GPU and CPU execute the *same* IEEE-754 operations in the *same*
  order, and this project's measured VERIFY deviation is **exactly `0.0`** (bit-identical) — not "small,"
  identical — which is the strongest possible confirmation that no indexing, layout, or projection
  divergence exists between the two paths.
- **The projective-SDF bias is real and measured, not hand-waved.** As derived above, the projective
  approximation's error grows like `1/cos θ` with incidence angle `θ`. This project's camera path is a
  circle at **constant height and radius**, so every one of the 24 views observes the sphere's lower
  "belly" (nearest the plane, where the surface curves away steeply from any downward-looking camera) at
  nearly the *same* shallow, near-grazing angle — there is no second, steeper vantage point for the
  running average to correct against. The measured consequence: of the ~24,000 voxels in the true
  surface shell, ~83% show sub-2-cm error (one voxel or better), but a ~0.65% tail reaches error near a
  full truncation band (~0.114 m) — concentrated, by direct inspection, exactly at that low-latitude
  belt. This is the textbook signature of the projective bias diverging at grazing incidence, *not* an
  indexing or projection bug (verified by locating the offending voxels and confirming their geometry);
  README §Limitations and §Exercises point at the fix (incidence-aware weighting, or a camera path with
  varied elevation) without applying it, so the effect stays visible and teachable.
- **FP32 throughout**, matching the repo default and real depth-sensor precision; `μ` and voxel size are
  both far above `float`'s relative precision at the scene's ~2 m working distance, so FP32 rounding is
  not a contributor to the errors discussed above (the projective-incidence bias dominates by roughly
  four orders of magnitude).
- **No angle wrapping, no quaternion drift risk in the hot path.** Poses are converted from quaternion
  to a 3×3 rotation matrix **once per frame on the host** (`quat_to_rot`, with defensive renormalization
  against the CSV's finite-precision digits) specifically so the kernel's inner loop never touches
  trigonometry or normalization — 2 million voxels reuse the same 9 numbers.

## How we verify correctness

Three independent checks, because a fusion+meshing pipeline can be *numerically identical to its own
CPU twin and still geometrically wrong*, or *geometrically plausible and still numerically diverged*:

1. **The §5 GPU-vs-CPU gate (`VERIFY`).** Fuse the first 4 frames through the kernel and through
   `reference_cpu.cpp`'s line-by-line twin, into two separate volumes from the same defined initial
   state; every voxel's TSDF must agree within abs tol `1e-5` and every weight must match exactly.
   Because both paths are engineered to execute identical IEEE-754 operations (the `fmaf` contract
   above), the *expected* deviation is exactly zero — and that is exactly what this project measures.
   This check catches indexing, projection, or layout bugs instantly (any such bug shifts a voxel's
   value by order 0.1–2.0, six to seven orders of magnitude above the `1e-5` floor).
2. **Ground truth against the analytic scene (`GROUND TRUTH`).** This is the check that makes an
   *invented* scene worth more than a stored fixture: because `scene_sdf()` is the scene's real,
   closed-form signed distance function (exact within the truncation band — see "The math"), the fully
   fused TSDF can be compared directly against ground truth, not just against another implementation of
   the same (possibly wrong) algorithm. Checked in two shells — a tight "surface" shell
   (`|sdf_gt| ≤ voxel/2`, where the mesh will actually land) and a wider "band" shell
   (`|sdf_gt| ≤ μ/2`) — with bounds set from measured values plus documented headroom (`main.cu`'s
   `GROUND-TRUTH CHECK` comment walks the exact numbers and the physical reason for the tail).
3. **Mesh checks (`MESH`), three of them.** *(a)* The GPU triangle **count** must equal an
   independently-computed CPU **recount** *exactly* — not within tolerance. Because both classify the
   same downloaded float values with the same comparisons and the same table, their totals are an
   *order-independent invariant* of the atomic-append pattern: it does not matter which thread's atomic
   fired first, only that every surface cell was classified once and correctly. *(b)* The count sits in
   a wide, sanity-check range (`[40000, 100000]`; measured `54822`) — loose enough that legitimate GPU
   generation-to-generation drift (a handful of borderline cells resolving differently) never trips it,
   tight enough that "zero triangles" or "ten million triangles" (a broken pass) fails instantly.
   *(c)* Every emitted vertex's position is checked against `scene_sdf()` directly — the single
   strongest geometric check available, because a wrong table row, wrong edge-to-corner mapping, or
   broken interpolation would throw vertices centimeters off the true surface immediately, not subtly.

All three checks run on **every demo execution**, offline, because the scene is synthetic and
deterministic end to end (no RNG anywhere in this project — the poses are closed-form trigonometry, the
depth is exact ray casting, the fusion arithmetic is the `fmaf` contract above). The mesh artifact
(`demo/out/mesh.obj`) and the volume-slice artifact (`demo/out/tsdf_slice.pgm`) make the result
*inspectable*, not just pass/fail — open the slice and you can literally see the sphere, the plane, and
the shadow of never-observed space between them.

## Where this sits in the real world

- **KinectFusion (Newcombe et al., 2011)** is the paper this project's fusion half reimplements
  didactically. The largest omission relative to the original is pose **tracking**: real KinectFusion
  estimates each incoming frame's pose by ICP against the model being built so far — a tight loop this
  repo splits out as its own project (`02.06`) so each piece can be verified independently.
- **nvblox and Voxblox** are this project's production-scale descendants: both replace the fixed dense
  grid with **voxel hashing** (sparse blocks allocated only where the sensor has actually looked), which
  removes the `O(N³)` memory ceiling and lets the map grow to room- or building-scale without a
  pre-committed bounding box; both also generate an **ESDF** (Euclidean signed distance field, not just
  truncated-near-surface) because planners need distance-to-obstacle *everywhere*, not just near
  surfaces — `07.09`'s jump-flooding distance transform is this repo's teaching version of that next
  step, and SYSTEM_DESIGN's Chain A literally chains the two (`05.01 → 07.09`).
- **Confidence/incidence-weighted fusion.** Production systems commonly replace this project's constant
  per-observation weight with something proportional to viewing-angle confidence (e.g. `cos θ` or
  measured sensor noise models), directly damping the grazing-incidence bias this project's ground-truth
  check measures and explains rather than hides (README Exercise 3 is exactly this change).
- **Bilinear/discontinuity-aware depth sampling.** This project's nearest-pixel lookup is the classic
  KinectFusion choice (bilinear would blur across real depth discontinuities and invent surface where an
  object ends); production stacks add discontinuity detection specifically to suppress "flying pixel"
  contamination near silhouette edges — the physical mechanism behind the very error tail this project's
  ground-truth check surfaces.
- **Marching cubes at scale.** VTK-m and NVIDIA's own CUDA marching-cubes sample use the two-pass
  count-then-scan emission this project's THEORY section above describes as the honest alternative;
  Open3D and PCL ship complete, tested integration + extraction pipelines worth reading once this
  project's mental model of *why* each piece exists is solid.
