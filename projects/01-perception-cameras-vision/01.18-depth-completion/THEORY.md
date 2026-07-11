# 01.18 — Depth completion: sparse LiDAR + RGB → dense depth: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

### Why LiDAR is sparse

A spinning LiDAR does not "see" a scene the way a camera does. A camera has millions of photosites
sampling a dense 2-D grid simultaneously. A mechanical spinning LiDAR has a small, FIXED number of laser
emitter/receiver pairs — **beams** — arranged at fixed elevation angles on a rotating head. This
project's synthetic sensor models a 16-beam unit (Velodyne VLP-16-class), with beams spanning
elevation angles `θ_el ∈ [-15°, +15°]` in 2° steps. As the head spins about the vertical (z) axis, each
beam sweeps out one **scan line** — a fixed-elevation cone, not a raster. At azimuth angle `θ_az`
(measured counter-clockwise from +x/forward, right-hand rule about +z/up), a beam's unit direction in
the LiDAR's own frame is

```
d = ( cos(θ_el) cos(θ_az),  cos(θ_el) sin(θ_az),  sin(θ_el) )
```

so **one revolution produces exactly 16 rings of returns**, not a dense 2-D image. Projected into a
camera's image plane (the geometry `src/kernels.cu`'s `project_zbuffer_kernel` computes), those 16
rings become 16 roughly-horizontal curves of points — everything BETWEEN two adjacent rings, in image
space, has zero direct LiDAR evidence. This project's committed sample measures this directly: 16
beams × ~280 azimuth samples inside the camera's field of view project to only **~5.99%** of the
160×120 image's pixels (`data/README.md` documents the exact count) — a real, physically-derived
sparsity, not an artificially thinned-out dense scan.

**The elevation/azimuth pattern on the image plane, derived.** For a beam looking at elevation `θ_el`
below horizontal, hitting the flat ground at height `h` below the sensor, the ground range is
`r = h / tan(|θ_el|)` — this is why a coarse elevation fan produces WILDLY UNEVEN ground coverage:
`dr/dθ_el = -h / sin²(θ_el)` blows up as `θ_el → 0` (the beam becomes horizontal). Near the horizon, a
tiny elevation change corresponds to an enormous range change — physically, this is why sparse-beam
LiDAR ground coverage is dense close to the vehicle and vanishingly sparse near the horizon (this
project's `error_guided.pgm` artifact shows exactly this: a bright, high-error band near the image's
horizon row, where the LiDAR simply has no return between the last in-range beam and the horizon).

### Beam divergence and the eye-safety power ceiling

A laser beam is not an infinitely thin ray: diffraction gives it an angular spread (divergence,
typically ~2–3 mrad for a collimated automotive LiDAR), so the illuminated spot grows with range
(a 3 mrad beam is a ~3 cm spot at 10 m, ~15 cm at 50 m) — one physical reason beam COUNT cannot simply
be increased for free: more/narrower beams need more optical elements and more precise alignment.

The other, often-overlooked reason 16-, 32-, and 64-beam units are common but 1000-beam units are not,
is **eye safety**. Automotive/robotics LiDAR is regulated (orientation only — see
[`PRACTICE.md`](PRACTICE.md) §4 and SYSTEM_DESIGN.md item 6) under **IEC 60825-1**, typically to Class 1
("safe under all conditions of normal use, including staring into the beam with an optical instrument").
Class 1 sets a Maximum Permissible Exposure (MPE) that caps the *average* optical power a beam can
deliver into an eye's pupil at any wavelength/pulse-duration combination. For a pulsed time-of-flight
LiDAR, the point rate is (very roughly)

```
points/sec  ≈  (available average optical power) / (energy per pulse required for the target SNR at max range)
```

Both factors are capped: total average power by the eye-safety limit, energy-per-pulse by the
receiver's noise floor at the required range. This is a genuine physical ceiling on point rate, not an
engineering laziness — it is *why* real automotive-grade LiDAR (2020s) tops out in the low millions of
points/second even as compute keeps getting cheaper, and *why* sparsity is fundamental to the sensor,
not a cost-cutting artifact this project should apologize for.

### Camera–LiDAR resolution mismatch, in numbers

This project's camera is 160×120 = 19,200 pixels. Its LiDAR, after projecting into that image, covers
~1,150 pixels (5.99%). A real automotive stack is worse in relative terms, not better: a 1920×1080
(2.07M pixel) camera paired with a 64-beam LiDAR spinning at 10 Hz and ~300,000 points/sec produces
~30,000 points per revolution, of which only a fraction land inside any one camera's FOV — well under
2% coverage is typical. **Depth completion exists because this mismatch is structural**, not a bug to
be fixed by a faster LiDAR.

### Engineering constraints that shape the algorithm

- **Range noise.** Real time-of-flight LiDAR has a range-measurement noise floor (this project models
  it as σ = 2 cm, illustrative of a MEMS/ToF-class unit — `data/README.md`); an algorithm that trusts
  sparse samples EXACTLY (zero smoothing) would also trust their noise exactly.
- **Latency budget.** A 10–20 Hz spinning LiDAR (SYSTEM_DESIGN.md item 1) sets the pipeline's real-time
  ceiling — this project's whole GPU pipeline measures single-digit-to-low-teens milliseconds
  (README "System context"), comfortably inside it.
- **The "edges coincide" prior is PHYSICS, not a heuristic of convenience** — depth discontinuities and
  albedo/color discontinuities coincide at the physical boundary of most opaque, individually-painted
  objects, because that boundary is where one solid surface ends and either another surface or empty
  space begins. But it is a prior, not a law: two physically distinct, adjacent surfaces painted the
  same color break it (this project's **camo edge**), and a flat surface with printed/painted texture
  breaks it the other way (this project's **texture trap**). Both are studied explicitly below.

## The math

**Notation.** All lengths in meters. `P = (x, y, z)` denotes a 3-D point; a subscript names its frame
(`P_lidar`, `P_cam`). `T_camera_lidar = (R, t)` is a rigid transform, `R` a 3×3 rotation (row-major,
`R[r·3+c]`), such that `P_camera = R·P_lidar + t` — the SAME `Rigid3` shape and convention
[`01.17`](../01.17-camera-lidar-camera-camera-extrinsic-calibration) solves for by nonlinear
least-squares; this project's `kTCameraLidar` (`src/kernels.cuh`) is one fixed, already-solved instance.
The camera's OPTICAL frame is z-forward, x-right, y-down (the documented exception to the body frame's
x-forward/y-left/z-up, SYSTEM_DESIGN.md §3.2). **Depth** always means `Pcam.z` (the perpendicular
distance to the image plane — the pinhole/z-buffer convention), never Euclidean range from the camera
center; the distinction matters because a point directly off-axis has range `> Pcam.z`, and mixing the
two conventions silently corrupts a depth map's geometry.

### Extrinsic used by this project (derived)

`kTCameraLidar`'s rotation is a pure axis PERMUTATION, not a general Rodrigues rotation — chosen
deliberately as the simplest non-trivial case (README "Limitations" contrasts this with 01.17's general
solve). With the LiDAR mounted parallel to the vehicle body (x-forward, y-left, z-up) and the camera's
optical axis aligned with body-forward (no relative tilt):

```
x_cam = -y_lidar        (camera "right" is body "-left")
y_cam = -z_lidar        (camera "down" is body "-up")
z_cam =  x_lidar         (camera "forward" is body "forward")
```

i.e. `R = [[0,-1,0],[0,0,-1],[1,0,0]]` (row-major). With the LiDAR mounted 0.30 m above and 0.05 m
behind the camera (body-frame offset `(-0.05, 0, 0.30)`), the translation in the CAMERA frame is
`t = R·(L_body − C_body) = R·(-0.05, 0, 0.30) = (0, -0.30, -0.05)` — exactly `kTCameraLidar.t`.

### Pinhole projection

For a point already in the camera's optical frame with `Pcam.z > 0`,

```
u = fx·Pcam.x / Pcam.z + cx        v = fy·Pcam.y / Pcam.z + cy
```

(`fx, fy, cx, cy` — `src/kernels.cuh`'s `kFx/kFy/kCx/kCy`, the SAME naming
[`01.16`](../01.16-checkerboard-charuco-detection-acceleration)/01.17 use). Points with `Pcam.z ≤ 0`
are behind the camera and dropped; the z-buffer (below) resolves the remaining many-to-one collisions.

### Inverse-distance weighting (the baseline)

For an unknown pixel `p`, gather every valid sparse sample `q` within a fixed pixel-space radius `R`
(`kIdwRadiusPx`), and estimate

```
D̂(p) = Σ_q  w_q · D(q) / Σ_q w_q,        w_q = 1 / ‖p − q‖₂^n        (n = kIdwPower = 2)
```

Pure spatial interpolation — the image never enters the formula. This is deliberately the "no prior"
baseline every gate compares the guided method against.

### Anisotropic diffusion (the main method) — the PDE

Treat the unknown depth field `D(x, y, t)` as the state of a 2-D heat-diffusion process evolving in a
FICTITIOUS time `t` (iteration index, not wall-clock time), with spatially-varying conductivity
`c(x, y)` gated by the guidance image `I`:

```
∂D/∂t = ∇ · ( c(x,y) ∇D )
```

the anisotropic-diffusion equation Perona & Malik introduced for edge-preserving smoothing (1990;
README "Prior art"). Where `c → 0` (a strong image edge), the PDE stops moving heat (depth) across that
boundary — exactly the "edges coincide" prior, expressed as a differential equation. The conductance
this project uses (Perona–Malik's own exponential form, `src/kernels.cuh`'s `kConductanceK`):

```
c(∇I) = exp( -(∇I / K)² )
```

`∇I` here is the MAX ABSOLUTE per-channel RGB difference between neighboring pixels (not grayscale
luminance — "Numerical considerations" explains why), and `K` is the gradient scale at which
conductance falls to `1/e`.

### Discretization and the stability bound (derived)

On the pixel grid (unit spacing), the discrete Laplacian with a spatially-varying, per-EDGE
conductance (one value per axis-aligned edge, computed once by `compute_conductance_kernel` and
reused for both endpoints — `THEORY.md` "The GPU mapping" explains why this guarantees symmetric flow)
becomes, for pixel `(x,y)` with neighbor conductances `g_L, g_R, g_U, g_D ∈ (0, 1]`:

```
D^{n+1}(x,y) = D^n(x,y) + dt · [ g_L(D^n_L − D^n) + g_R(D^n_R − D^n) + g_U(D^n_U − D^n) + g_D(D^n_D − D^n) ]
             = (1 − dt·Σg) · D^n(x,y)  +  dt·(g_L D^n_L + g_R D^n_R + g_U D^n_U + g_D D^n_D)
```

The second line is a weighted average of `D^n(x,y)` and its 4 neighbors' PREVIOUS values, with weights
`(1 − dt·Σg)` and `dt·g_i`. This is a **convex combination — and therefore satisfies a discrete maximum
principle (no new extrema, no oscillation, no blow-up)** — exactly when every weight is non-negative,
i.e. `1 − dt·Σg ≥ 0`, i.e.

```
dt ≤ 1 / Σg  =  1 / (g_L + g_R + g_U + g_D)
```

Since Perona–Malik conductance is bounded in `(0, 1]`, the WORST case is `Σg = 4` (every neighbor edge
fully open), giving the universal bound **`dt ≤ 1/4 = 0.25`** independent of the image. This project
runs `kDiffusionDt = 0.20` — inside the bound with margin — and `kernels.cuh` `static_assert`s the bound
at COMPILE TIME (README "Expected output"; a real numerics gate, not a comment asserting one exists).

## The algorithm

**Stage 1 — projection + z-buffer.** For each of `N` LiDAR points (a few hundred to ~2000 in this
project): transform to camera frame, project to `(u,v)`, round to a pixel, and race an `atomicMin`
(GPU) / sequential compare (CPU) against every other point landing on the same pixel, keeping the
smallest depth (nearest wins — the physically correct occlusion resolution: a near surface visible to
the LiDAR from its own perspective legitimately hides a farther one at the SAME projected pixel).
Serial cost `O(N)`; parallel cost `O(N / P)` plus the (rare, at this density) atomic contention.

**Stage 2 — conductance.** One pass over all `W·H` pixels, `O(1)` work each: two neighbor reads, two
`exp()` evaluations. `O(W·H)` serial and parallel (embarrassingly parallel, no dependencies).

**Stage 3 — anisotropic diffusion.** `kDiffusionIters = 1400` full sweeps over the `W·H` grid, each
`O(1)` work per pixel (4 neighbor reads, one FMA-heavy update, one Dirichlet-anchor branch). Serial cost
`O(iters · W · H)` ≈ 1400 × 19,200 ≈ 26.9M scalar updates (milliseconds on one CPU core — see
`[time]` lines in a real run); parallel cost `O(iters · W · H / P)` — the ITERATION COUNT is not
parallelizable (each sweep depends on the previous one), but each sweep's `W·H` pixels are fully
independent, which is exactly what a GPU exploits.

**Stage 4 — IDW.** One pass over all `W·H` pixels, each scanning a `(2R+1)²` window (`R = 16` →
33×33 = 1089 candidate cells). Serial cost `O(W·H·R²)`, parallel cost `O(W·H·R² / P)` — the single
most compute-heavy per-pixel kernel in this project, and the clearest "why the GPU helps" story
(a plain nested loop per pixel, no cleverness, that a CPU pays for directly).

**Evaluation (not a kernel).** Region masks are built once from ground truth depth and RGB gradients
(`O(W·H)`); every gate is a masked mean/RMS over the two densified fields — all `O(W·H)`, trivial next
to the pipeline itself.

## The GPU mapping

**Projection — scatter with atomic z-buffering.** One thread per LiDAR POINT (not per pixel): the
opposite of every other kernel in this project. Threads write to data-dependent, potentially colliding
output addresses, which is exactly the situation atomics exist for. CUDA has no native `atomicMin` for
`float`; the fix reinterprets each depth's IEEE-754 bit pattern as `uint32_t` via `__float_as_uint` and
runs the INTEGER `atomicMin` on that. For any two POSITIVE, finite floats, the raw bit pattern preserves
numeric ordering (the exponent occupies the high bits and dominates the comparison) — no transformation
needed. (The fully general encoding, for possibly-negative floats, flips every bit if the sign bit is
set, else flips only the sign bit — documented in `kernels.cu`'s `encode_depth_for_zbuffer` for
completeness; this project's depths are always positive in front of the camera, so the simple branch
is what actually runs.) The output buffer is pre-filled with `UINT32_MAX` (`cudaMemset(..., 0xFF, ...)`)
as the "empty pixel" sentinel — safely larger than any realistic depth's encoded bit pattern.

**Conductance — a 2-D map/stencil hybrid.** One thread per pixel (16×16 blocks, the standard 2-D
stencil default this repo uses throughout — 01.11's bilateral kernels are the direct precedent), each
reading 2 forward neighbors (right, down) and writing 2 outputs. No shared memory: each pixel's
neighbors are read exactly once by this kernel (unlike a wide stencil such as bilateral filtering, there
is no redundant re-reading to amortize with a tile).

**Diffusion — ping-pong stencil iteration.** The precedent this project follows explicitly is
[`07.09`](../../07-collision-geometry/07.09-jump-flooding-voronoi-distance-transforms)'s jump-flooding
kernel and [`31.01`](../../31-safety-verification/31.01-hamilton-jacobi-reachability)'s
Hamilton–Jacobi level-set marching: BOTH are "iterate a stencil update over a full grid many times,
each iteration a fresh kernel launch reading buffer A and writing buffer B, then swap" — the discipline
that makes every iteration a clean, order-independent Jacobi update (see "Numerical considerations"
below for why writing in place would be a race). `launch_diffusion` owns the WHOLE unit of work —
allocating the ping-pong pair and the conductance buffers, seeding the initial condition, running all
`kDiffusionIters` launches, and freeing everything — mirroring `launch_jfa`'s "launcher owns the
schedule" contract (kernels.cu file header).

**IDW — bounded search, one thread per output pixel.** Each thread performs its OWN independent
`(2R+1)²`-window scan; no shared memory (unlike bilateral filtering's overlapping-tile shared-memory
optimization, IDW's window only reads a SPARSE subset of cells — most are `kInvalidDepth` — so a
shared-memory tile would mostly cache misses; README Exercise 4 assigns the k-NN alternative that
would justify a spatial acceleration structure instead).

**Memory hierarchy.** Every kernel in this project uses GLOBAL memory only, with `__restrict__` on every
read-only pointer parameter (unlocking the compiler's non-aliasing optimizations) — at 160×120
resolution, the whole problem's working set (a handful of `19,200`-float arrays) is a few hundred
kilobytes, well under any GPU's L2 cache; occupancy and bandwidth are not the bottleneck at this scale
(THEORY.md is honest that a production-resolution version WOULD need the shared-memory tiling 01.11
demonstrates for its stencil).

## Numerical considerations

**Float depth encoding for atomicMin.** Covered above (The GPU mapping); the risk case (a NaN or
non-finite depth reaching the encoder) cannot occur here — `zc ≤ 0` is filtered before the encode call,
and the ray-cast synthetic data never produces NaN/Inf depths by construction.

**Diffusion stability.** Derived above (The math); enforced by a compile-time `static_assert`.

**Division by tiny weights.** IDW's weight `1/dist^n` is well-defined for every `dist > 0` reached by
the window search (the `dist == 0` / "this pixel IS a sample" case is handled by an EARLY RETURN before
any division, both GPU and CPU — never a `1/0`). The only remaining hazard is an EMPTY window (no valid
sample anywhere within radius `R`) — `wsum == 0` — handled as an explicit, documented fallback to `0.0`
rather than an unguarded `NaN`-producing division (`kernels.cu`/`reference_cpu.cpp`); this project's
committed sample density makes the fallback rare but not impossible near the image border, and the
evaluation gates measure real output, so a fallback firing would show up as elevated error rather than
being silently hidden.

**Full-color, not grayscale, conductance.** An earlier version of this project computed conductance
from grayscale luminance (`0.299R + 0.587G + 0.114B`) alone — a plausible-looking simplification that
turned out to be a measured bug, not a harmless shortcut: this scene's `near_box` (RGB `(185,60,50)`,
a clearly red object to a human or to a color-aware algorithm) has luminance `≈0.377` (normalized),
while the ground plane's gray asphalt has luminance `≈0.36` — a difference of `0.016`, an order of
magnitude BELOW `kConductanceK = 0.12`. Grayscale-only conductance therefore failed to gate the
box/ground boundary AT ALL, and every measured gate (`edge_quality` especially) was worse before
switching to a max-absolute-per-channel-difference conductance (see `kernels.cu`'s
`max_channel_diff`). The general lesson: two surfaces can differ strongly in HUE while landing at
nearly the same LUMINANCE — collapsing to grayscale before computing an edge-aware guidance signal
throws away exactly the information a robotics image-guided algorithm often needs, and can silently
recreate an "accidental camo edge" everywhere a scene happens to have this property (not just at this
project's deliberately-designed camo pair).

**Diffusion seed value.** Unknown (non-anchor) pixels must start the PDE somewhere. This project seeds
them at the MEAN of the valid sparse samples (computed once, on the host, from the array both the GPU
and CPU paths already hold) rather than at `kInvalidDepth`'s sentinel value — an earlier version that
seeded at the sentinel measured guided MAE nearly DOUBLE the mean-seeded version, because any region
whose conductance gate is closed on every side (the checkerboard texture trap is the designed example)
can only reach a Dirichlet anchor if one falls inside it; a pixel that never reaches one keeps its seed
value for the ENTIRE run. A sentinel there is an obviously-wrong depth; the sparse mean is an honest
"we have no local evidence, guess the scene average" fallback. README Exercise 1 asks the learner to try
seeding from the IDW result instead — a more spatially-aware prior still.

**Determinism.** Both the CPU and GPU diffusion paths are Jacobi updates (ping-pong buffered, never
in place), so pixel evaluation ORDER never matters within one iteration — the two independent
implementations are expected to (and measured to) agree far tighter than the loose 5 cm/1400-iteration
VERIFY tolerance allows; the loose bound exists because 1400 chained FP operations through two
INDEPENDENT code paths (different compilers' instruction selection, FMA contraction) can accumulate
rounding drift, and the VERIFY stage's job is to catch a BUG, not to demand bit-identical output from
two honestly-different implementations.

## How we verify correctness

**Tier 1 — GPU-vs-CPU twins**, per the repo's independence ruling (`reference_cpu.cpp`'s file header):
projection+z-buffer (tol `1e-4 m`, effectively exact — both paths do the same finite compare, no
accumulated iteration), conductance (tol `1e-5`, one `exp()` evaluation), diffusion (tol `5e-2 m` after
the full 1400 iterations — the loosest, for the reason given above), IDW (tol `1e-3 m`). All four ran
PASS on the committed sample at every measured revision of this project.

**Tier 2 — independent gates against exact synthetic ground truth** (the tier the twin comparison is
BLIND to — see the sharing ruling): `overall_accuracy` (guided MAE bounded, `< 1.3 m`, set from a
measured `0.92 m` with headroom), `edge_quality` (guided beats IDW by `≥ 1.15×` at CLEAN depth
boundaries — real depth edges the RGB image also shows — measured `1.27×`), `texture_trap` (guided
RMSE stays within `1.8×` of the RGB-blind IDW baseline on the checkerboard patch — measured `1.35×`;
IDW cannot be "fooled" by texture by construction, so this bounds how much conductance gating is
allowed to hurt where its prior does not apply), `camo_edge_honesty` (error at the deliberately
low-contrast REAL depth edge must EXCEED the ordinary flat-region error by `≥ 2.0×` — measured
`3.63×` — this gate does not reward accuracy, it rewards the demo actually SHOWING the failure mode
rather than getting lucky and hiding it), and `input_fidelity` (every Dirichlet-anchored pixel
reproduces its sparse input exactly, both methods, tol `1e-3 m` — measured exact `0.0`). Every threshold
is set BELOW (or above, for a ceiling) a measured run with a stated margin, never chosen blind
(CLAUDE.md's "quote only measured numbers" rule) — the exact measured figures live in README and in
`demo/out/gates_metrics.csv` after any run.

**Edge cases exercised:** pixels with no truth (sky, or beyond `kMaxDepthM`) are excluded from every
gate; the image border (where a stencil neighbor would fall outside `[0,W)×[0,H)`) is handled as a
zero-flux/natural boundary for diffusion and a clipped window for IDW; the `dist==0` IDW case and the
"no sample in the window" case are both explicit, tested branches, not accidents of arithmetic.

## Where this sits in the real world

Classical depth completion (this project's whole pipeline) was the state of the art through roughly
2016; the KITTI Depth Completion benchmark's 2017 launch coincided with, and helped drive, the shift to
LEARNED methods. Sparsity Invariant CNNs (Uhrig et al. 2017) showed that a plain CNN, told explicitly
which pixels are "real" via a validity mask, beats hand-tuned interpolation; the sparse-to-dense line
(Ma & Karaman 2018) added RGB-guided encoder-decoder architectures; Convolutional Spatial Propagation
Networks (Cheng et al., 2018 → CSPN++, 2020) LEARN an affinity matrix that plays the same role this
project's hand-set Perona–Malik conductance does, but tuned end-to-end on millions of labeled frames
instead of one scalar `K`. Current production perception stacks (automotive and warehouse-AMR alike)
overwhelmingly use learned depth completion or skip explicit densification entirely, fusing sparse
LiDAR and dense RGB directly inside a downstream learned perception network (e.g. a BEV encoder).

The classical toolkit taught here has not disappeared, though: **guided/joint-bilateral filtering**
(the alternative to iterative diffusion this project names — README "Prior art") remains a common,
cheap, closed-form post-processing or upsampling step even downstream of a learned depth network, and
anisotropic-diffusion-family PDEs are still standard in medical and scientific imaging for exactly the
edge-preserving-smoothing reason this project studies. Understanding the classical pipeline — and,
crucially, its FAILURE TAXONOMY (texture-on-flat surfaces, low-contrast real edges, and — not modeled
in this project's opaque-object scene — transparent/specular surfaces, which break the "any single
LiDAR return corresponds to one real surface" assumption entirely) — is what makes a learned method's
behavior legible: a CNN trained on real data learns to approximate the SAME prior this project encodes
by hand, and inherits a version of the SAME failure modes wherever its training data under-represents
them.
