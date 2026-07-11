# 01.10 — Rolling-shutter correction using IMU rates: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

### Why does a camera sensor have a "readout time" at all?

A digital image sensor is a 2-D grid of photosites, each accumulating charge proportional to incident
light during its exposure. To turn that charge into a digital number, every photosite's charge must pass
through an analog-to-digital converter (ADC) — but a sensor with millions of pixels cannot afford one ADC
per pixel (cost, power, die area), so it shares a small number of ADCs (often one per column, or one per
some number of columns) across many photosites. The sensor reads the array out a slice at a time — for a
CMOS **rolling shutter** sensor, one ROW at a time — because a row's photosites can all be connected to
their column ADCs simultaneously, while different rows must take turns.

This means: row 0 is exposed, then read out, while row 1 is *still exposing* — by the time row 1 is read,
row 2 is exposing — and so on. Row `v`'s exposure window starts `t_line` seconds later than row `v-1`'s,
where `t_line` is the sensor's fixed per-row readout time. For a whole frame of `H` rows, the *last* row
starts exposing `(H-1) * t_line` seconds after the *first* — this project's synthetic sensor has
`H = 288` rows and a 25 ms full-frame readout, giving `t_line ≈ 86.8 microseconds/row` (`kLineTimeS` in
`kernels.cuh`). If the camera is perfectly still during those 25 ms, this is invisible: every row shows
the same static scene regardless of when it was exposed. If the camera **rotates** during readout — a
drone's motor vibration, a handheld device's tremor — each row captures the world through a measurably
different orientation, and the frame comes out visibly warped.

### The taxonomy of rolling-shutter artifacts, derived from the row-homography model (§"The math" below)

- **Skew** — a *constant* angular velocity (e.g. a steady yaw) shifts every row's effective orientation
  by an amount that grows linearly with row index → straight vertical lines become straight, but
  *slanted*, lines.
- **Wobble** — an *oscillating* angular velocity (vibration, e.g. this project's sinusoidal jitter
  profile) produces a row-dependent shift that oscillates with row index → straight lines become
  S-curved. This is what `../scripts/make_synthetic.py`'s rendered `rs_input.pgm` shows.
- **Jello** — the colloquial name for wobble severe enough to make rigid objects appear to flex like
  gelatin, a well-known artifact on drone footage and older phone cameras during vibration.

### Global-shutter and stacked-CMOS alternatives — the engineering trade-off this project's correction avoids paying

A sensor can avoid rolling-shutter artifacts entirely by reading every row at once:

- **CCD global shutter**: every photosite transfers its charge to a shielded storage element
  *simultaneously*, then all storage elements are read out (slowly, row by row, but AFTER exposure ends
  — so every pixel's *exposure window* is identical even though the *readout* is still sequential). CCDs
  cost more to fabricate, draw more power, and have historically lagged CMOS in resolution/frame-rate for
  the same cost.
- **Stacked-CMOS global shutter**: modern sensors add a second silicon layer bonded under the photosite
  layer, giving each pixel its own small in-pixel storage node (a "memory" pixel) so charge can be
  latched simultaneously and read out afterward at leisure — the CMOS-cost-structure answer to a CCD's
  global shutter, at the price of extra process complexity, larger pixel pitch (lower fill factor: the
  fraction of the pixel occupied by the actual light-sensing area, since some area is now storage
  circuitry), and thus somewhat worse low-light sensitivity per unit die area.
- **Why rolling shutter still ships everywhere**: it is cheaper, simpler, and — for a STILL or slowly-
  moving camera — invisible. Most phone, webcam, and many robotics cameras are rolling-shutter; global-
  shutter sensors are reserved for applications (machine vision, some drones, some AR headsets) where the
  cost premium is justified by motion severity.

### The engineering constraints a real system imposes

- **Noise floor**: the gyro signal this project corrects with is never noise-free; every real MEMS gyro
  has bias (a slowly-drifting DC offset) and white-noise-like jitter on top of the true signal (this
  project's `gyro_degraded.csv` models both, `README.md`'s Exercises invite tuning them).
- **Bandwidth/latency**: correction must run within the camera's frame period (30-60 Hz elsewhere in this
  repo's reference robots) or it becomes the bottleneck — see README "System context".
- **Timing accuracy**: `t_line` and the camera-IMU time offset must be known to a small fraction of
  `t_line` for the correction to be worth applying at all; `PRACTICE.md` §3 covers how a real rig
  measures both.

## The math

### Notation

- Frames: this project uses the **camera-optical** frame at every camera pose — z-forward (down the
  optical axis), x-right, y-down (row 0 = image top) — the same documented exception 01.01's
  `kernels.cuh` states at this API boundary (CLAUDE.md §3.2 permits camera optics its own convention).
- `K = (fx, fy, cx, cy)` — the shared pinhole intrinsic matrix (pixels); no lens distortion is modeled
  (out of scope for this project — see 01.01 for that lesson).
- `q_world_cam(t)` — a UNIT quaternion, REPO ORDER `(w, x, y, z)` (docs/SYSTEM_DESIGN.md §3.4, same order
  09.01's forward-kinematics model uses), the camera's orientation at time `t`: for a vector `v_cam`
  expressed in the camera's own axes, `v_world = R(q_world_cam(t)) * v_cam` (the `T_parent_child`
  convention, §3.3, "child (camera) expressed in parent (world)").
- `omega(t) = (wx, wy, wz)`, rad/s — the gyro's measurement: BODY-frame (camera-axes) angular velocity.
- `t(v) = t0 + v * t_line` — the exposure start time of output row `v`, `v = 0..H-1`.
- `t_ref = t0 + 0.5*(H-1)*t_line` — this project's chosen reference instant: the exposure time of the
  MIDDLE row (`(H-1)/2`), which by construction equals `cy = (H-1)/2` — a deliberate alignment so the
  reference view's "center row" and "reference row" are the same row.

### Body-rate quaternion kinematics

A gyro measures angular velocity in its own (body/camera) axes. The standard kinematic equation relating
orientation rate to body-frame angular velocity is

```
dq_world_cam/dt = 0.5 * q_world_cam ⊗ [0, omega]        ⊗ = quaternion (Hamilton) product
```

This project integrates it via the **exponential map**: over a short interval `dt` during which `omega`
is (approximately) constant, the EXACT solution is a rotation by angle `|omega|*dt` about axis
`omega/|omega|`, expressed as the quaternion

```
dq = [ cos(|omega|*dt/2),  (omega/|omega|) * sin(|omega|*dt/2) ]
q_world_cam(t+dt) = q_world_cam(t) ⊗ dq             (RIGHT-multiply: dq is a BODY-frame delta rotation)
```

**Small-angle vs. exact**: the popular alternative is first-order Euler,
`q_next = normalize(q + 0.5*q⊗[0,omega]*dt)`, which is only a *linearization* of the same closed-form
exponential-map solution — its per-step error is `O(dt^2)` versus the exponential map's `O(dt^3)` (the
exponential map is exact for truly constant `omega`; its only error source is `omega`'s deviation from
constant WITHIN the step). `kernels.cuh`'s `quat_integrate_step()` uses the exact exponential-map form
throughout, at the cost of one extra `sin`/`cos` pair per step — a cost that is irrelevant here since the
whole integration is a few hundred sequential steps total (see "The algorithm" below).

### The pure-rotation row homography

Fix the reference orientation `q_ref = q_world_cam(t_ref)`. An OUTPUT (reference-view) pixel `(xo, yo)`
corresponds to the world ray

```
ray_world = R(q_ref) * K^-1 [xo, yo, 1]^T
```

The SAME world ray, if it were instead observed by the camera at time `t(v)` (row `v`'s exposure
instant), would land at RAW rolling-shutter pixel `(xs, ys)`:

```
ray_row = R(q_world_cam(t(v)))^T * ray_world
        = R(q_world_cam(t(v)))^T * R(q_ref) * K^-1 [xo, yo, 1]^T
[xs, ys, w]^T = K * ray_row  ;  (xs, ys) = (xs/w, ys/w)
```

Collapsing the middle rotation composition into one quaternion (using `R(q1)^T = R(conj(q1))` for unit
quaternions, and `R(q1)*R(q2) = R(q1⊗q2)`):

```
q_rel(v) = conj(q_world_cam(t(v))) ⊗ q_ref
H(v)     = K * R(q_rel(v)) * K^-1                    THE ROW HOMOGRAPHY
```

This is EXACTLY 01.01's `K * R_rect_raw^T * K^-1` construction (that project's `compute_source_pixel()`),
generalized from ONE fixed rectifying rotation to a relative rotation THAT DEPENDS ON THE ROW. `kernels.cuh`
implements `H(v)`'s action in three decomposed steps (`apply_row_rotation()`) rather than pre-multiplying
a literal 3x3 pixel-space matrix, exactly mirroring `compute_source_pixel()`'s structure.

### Why pure rotation, and when the assumption breaks

`H(v)` above has NO translation term because the derivation assumed the camera's optical center never
moves — every camera pose differs from every other only by a rotation. This is EXACT when the scene is
infinitely distant (or, equivalently, when the camera genuinely only rotates and never translates: with
zero translation, a given camera RAY always samples the same world point regardless of scene depth,
because there is no parallax). It becomes an APPROXIMATION the moment the camera also translates during
readout: a translating camera's ray to a NEARBY point sweeps across the scene differently than its ray to
a FAR point (near-field parallax) — a homography alone cannot capture that row-dependent, depth-dependent
shift. This project's synthetic scene is authored as a reference-plane texture (README "What this
computes", `make_synthetic.py`'s file header) specifically so the pure-rotation model is exact by
construction; README "Exercises" invites breaking this assumption on purpose to see the restoration gate
degrade.

## The algorithm

### Step-by-step

1. **Gyro integration** (host, `main.cu::integrate_gyro_to_fine_trajectory`): read the sparse (~10-sample,
   200 Hz) gyro CSV; between every consecutive pair of samples, LINEARLY interpolate `omega` and take
   `kIntegrationSubsteps = 32` exponential-map sub-steps (the "The math" section's exact per-step
   update), producing a dense trajectory of `(t, q)` pairs roughly every 156 microseconds.
   **Complexity**: `O(n_gyro_samples * kIntegrationSubsteps)` — a few hundred sequential quaternion
   multiplies. Inherently sequential (each step's output feeds the next) — no useful GPU parallelism
   exists here (see "The GPU mapping").
2. **Row-LUT build** (host, `main.cu::build_row_lut`): for every output row `v = 0..H-1`, interpolate the
   fine trajectory at `t(v)`, and compute `q_rel(v) = conj(q_row) ⊗ q_ref`. **Complexity**: `O(H)` = 288
   interpolations, each `O(log n_fine)` via binary search.
3. **Upload** the 288-quaternion row LUT to GPU `__constant__` memory (`set_row_lut`).
4. **Per-pixel fixed-point row-time search + bilinear sample** (GPU, `rs_correct_kernel` — the ONE kernel
   in this project): for each output pixel `(xo, yo)`,
   ```
   v_guess = yo
   repeat kFixedPointIters = 3 times:
       q_rel = interpolate row LUT at v_guess          (lerp_row_quat)
       R     = matrix(q_rel)                            (quat_to_mat3)
       (xs, ys) = apply_row_rotation(R, xo, yo)
       v_guess = ys
   sample = bilinear_sample_gray(rs_frame, xs, ys)
   ```
   **Complexity**: `O(W*H*kFixedPointIters)` = 110,592 * 3 ≈ 332k row-homography evaluations, each `O(1)`.

### The fixed-point argument (why 3 iterations is enough, and why it converges at all)

The circularity: which row's rotation to use depends on WHICH ROW the output pixel's ray lands in after
rotation — but that row is exactly what we are trying to compute. This is a classic **fixed-point
problem**: define `f(v) = ys(H(v) applied to (xo, yo))`, the row that guessing "source row = v" produces.
We want `v* = f(v*)`.

**Why iterating `v_{k+1} = f(v_k)` converges**: `f` is a composition of (a) a smooth interpolation between
adjacent rows' quaternions (`lerp_row_quat`, Lipschitz-continuous with a SMALL Lipschitz constant — see
"Numerical considerations" below for the bound) and (b) the row-homography's action on the FIXED point
`(xo, yo)`, whose sensitivity to `v` is governed entirely by how fast the camera's orientation changes per
row — for this project's rotation rates (peak ~1.4 rad/s) and `t_line ≈ 87 microseconds`, the orientation
changes by at most ~`1.4 * 87e-6 ≈ 1.2e-4` radians between adjacent rows, an utterly tiny perturbation. `f`
is therefore a CONTRACTION near the true fixed point (its derivative `df/dv` has magnitude far below 1),
and contraction-mapping iteration converges geometrically: each iteration reduces the remaining error by
roughly the same small factor. `main.cu`'s `row_time_convergence` gate MEASURES this directly (the
iteration-2-vs-3 delta), rather than trusting the contraction argument blindly — and measures
`~0.002 px` after 3 iterations on the committed sample (README "Expected output"), many orders of
magnitude below one pixel.

## The GPU mapping

- **Thread-to-data mapping**: one thread per OUTPUT pixel, `(xo, yo) = (blockIdx.x*blockDim.x+threadIdx.x,
  blockIdx.y*blockDim.y+threadIdx.y)` — a 2-D grid, block `(32, 8)` (`kernels.cu`'s `launch_rs_correct`).
  32 along `x` is not arbitrary: it makes `blockDim.x` exactly one warp, so every thread in a warp shares
  the same `yo` and therefore issues the SAME first LUT read (`v_guess = yo` at iteration 0) — a genuine
  constant-memory broadcast, not a decorative one.
- **Memory hierarchy**:
  - **`__constant__` memory** (`c_row_lut[kImgH]`, 288 quaternions = 4,608 bytes) — read by every thread,
    often at the SAME address within a warp (see above); constant memory caches one fetch and broadcasts
    it to the whole warp, the textbook use case (same pattern as 09.01's robot model). It is written ONCE
    per gyro variant, from the HOST, via `cudaMemcpyToSymbol` — kernels cannot write `__constant__`
    memory at all, which is exactly why the row-LUT build (Step 2 above) stays on the host: there is no
    legal way for a kernel to populate it directly.
  - **Global memory** (`rs_frame`, `corrected`, `valid_mask`, `iter_delta`) — read/written once per pixel,
    fully coalesced (adjacent threads along `x` touch adjacent addresses).
  - **Registers only** for the per-pixel search state (`v_guess`, `xs`, `ys`, the 3x3 matrix `R`) — no
    shared memory, because nothing is reused BETWEEN threads (each thread's search path is its own).
- **LUT-vs-recompute — the central lesson**: re-deriving the gyro integration PER PIXEL would be both
  wrong (it is a sequential recurrence, not independently re-derivable per pixel) and enormously wasteful
  (110,592x redundant identical work). Precomputing it ONCE into a small LUT, then letting every pixel do
  O(1) lookups, is the same "LUT vs. recompute" trade-off 01.01's remap LUT teaches for lens distortion —
  except here the LUT is per-ROW (288 entries) rather than per-PIXEL (110,592 entries in 01.01), because
  the row homography does not depend on the output COLUMN at all.
- **Occupancy/bandwidth**: this kernel is tiny by modern GPU standards (110,592 threads, each doing a
  few dozen FLOPs and one 4-byte read + a few 1-byte reads/writes) — memory-bound but far under any real
  bandwidth ceiling; the `[time]` lines in the demo output report the measured kernel time, honestly
  labeled a teaching artifact rather than a benchmark (CLAUDE.md §12).

## Numerical considerations

- **Precision**: FP32 throughout (matching this repo's default); no FP64 is needed because every
  quantity (pixel coordinates, small angles, unit quaternion components) stays well inside FP32's dynamic
  range and precision budget at this project's scale.
- **Quaternion normalization drift**: every quaternion operation in `kernels.cuh`
  (`quat_mul`/`quat_integrate_step`/`lerp_row_quat`/the fine-trajectory interpolation) renormalizes its
  result immediately — chained multiplications and linear (non-unit-preserving) interpolation both drift
  off the unit sphere if left unchecked (CLAUDE.md §12 names this hazard explicitly).
- **LERP vs. SLERP for `lerp_row_quat`**: linearly interpolating quaternion COMPONENTS (rather than
  spherically interpolating, SLERP) is only accurate for small angular gaps between the two quaternions
  being blended. Row-to-row, the gap is the ~1.2e-4 radian figure computed in "The algorithm" above — many
  orders of magnitude below where LERP and SLERP diverge measurably (that divergence grows with the
  angle, and only becomes visible above tens of degrees) — so LERP is not a shortcut so much as a genuinely
  equivalent, cheaper computation at this scale. README "Exercises" invites verifying this by
  implementing SLERP and confirming the gates' numbers do not move.
- **The fixed-point search's own numerics**: the LUT lookup CLAMPS `v_guess` into `[0, H-1]`
  (`lerp_row_quat`) so a pathological guess can never index out of bounds; `main.cu`'s
  `row_time_convergence` gate reports how far the clamped search actually still moves between the last
  two iterations, so a search that WOULD have wanted to walk out of bounds is visible in that number, not
  silently hidden.
- **`sz` near zero**: `apply_row_rotation`'s perspective divide (`1/sz`) could in principle become
  ill-conditioned if the accumulated rotation approached 90 degrees (a ray rotated to lie in the image
  plane). This project's rotation amplitudes (a few degrees peak) keep `sz` within a fraction of a percent
  of 1 at all times — nowhere near that singularity; `kernels.cuh`'s comment on `apply_row_rotation`
  states this bound explicitly rather than leaving it implicit.
- **Invalid pixels, not silent clamping**: `bilinear_sample_gray` reports a pixel as INVALID rather than
  clamping to the nearest edge pixel when the resolved source coordinate falls outside the RS frame — a
  deliberate choice (kernels.cuh's file header) so a wrong-looking pixel is never manufactured to LOOK
  plausible; `valid_coverage`'s gate and the restoration gate's masking both account for this honestly.
- **Determinism**: every random draw in this project (`make_synthetic.py`'s texture hashing and gyro
  noise) uses a hand-rolled `xorshift32` PRNG, seeded 42, never `std::uniform_real_distribution` or
  numpy — reproducible byte-for-byte across machines and runs (CLAUDE.md §12).

## How we verify correctness

Two independent tiers (per the twin-independence ruling stated in full in `reference_cpu.cpp`'s file
header):

1. **GPU-vs-CPU twin comparison (`VERIFY`)**: `reference_cpu.cpp`'s `rs_correct_cpu` independently types
   the SAME per-pixel fixed-point search loop `rs_correct_kernel` runs on the GPU (sharing only the small,
   `HD`-marked camera-model primitives — quaternion algebra, `apply_row_rotation`,
   `bilinear_sample_gray` — the documented "pure token-for-token transcription" exception). `main.cu`
   compares the corrected image (tolerance 2.0, 0-255 scale — covers legitimate FMA-contraction
   differences between `nvcc` and `cl.exe`), the convergence-delta buffer (tolerance 0.01 px), and the
   valid-mask buffer (tolerance: at most 8 disagreeing pixels, covering boundary ties where a coordinate
   lands within float rounding of the frame edge). **Measured**: max|gpu-cpu| = 1.0000 (image), 0.00006
   px (convergence delta), 0 mismatched pixels (valid mask) — comfortably inside every tolerance.
2. **Independent physical gates, bypassing the shared primitives**: `quat_integration_analytic` (integrates
   a KNOWN constant angular velocity and compares to the closed-form analytic angle — touches NONE of
   `apply_row_rotation`/`lerp_row_quat`/`bilinear_sample_gray`), `restoration`/`restoration_negative_control`
   (compares actual PIXEL CONTENT against an independently-rendered ground-truth image, never re-deriving
   the homography), and `straightness_corrected`/`straightness_negative_control` (a from-scratch, host-side
   threshold-crossing line detector that reads pixel bytes directly, exactly 01.01's straightness-gate
   pattern). A bug shared between the GPU kernel and its CPU twin's camera-model primitives would pass
   VERIFY but fail these — precisely why the repo requires both tiers.

Every tolerance in `main.cu` is a floor/ceiling with margin OVER an actually-measured run on the
committed sample (never AT the measured value) — the exact numbers are recorded in `main.cu`'s
tolerance-block comment and in README "Expected output".

## Where this sits in the real world

- **This project's approach — correct the IMAGE**: resample the captured frame into a single consistent
  pose before handing it downstream. Used in broadcast/consumer video stabilization and some AR
  pipelines, where downstream consumers expect an ordinary, geometrically-simple image.
- **The more common production approach — model the SENSOR inside the ESTIMATOR**: modern VIO/SLAM
  systems (OKVIS's and VINS-Fusion's rolling-shutter-aware variants, among others) do NOT pre-correct the
  image at all. Instead, they fold the PER-ROW exposure timestamp directly into the bundle-adjustment/
  filter optimization, treating each observed feature's row as implying its own small pose offset from
  the frame's nominal pose. This avoids ever resampling/blurring the image (feature detectors run on the
  raw pixels) and lets the optimizer account for RS geometry with the SAME uncertainty-aware machinery it
  already uses for everything else — at the cost of a more complex estimator formulation than this
  project's standalone image-correction step.
- **Sensor-level global reset**: as covered in "The problem", buying a global-shutter (or stacked-CMOS
  global-shutter) sensor sidesteps the problem in hardware — the production answer whenever the cost/
  fill-factor/low-light trade-off is acceptable for the platform.
- **Gimbal stabilization**: many drones physically counter-rotate the camera (a 2-3 axis mechanical
  gimbal) to keep it nearly stationary during capture — a MECHANICAL answer that reduces the rotation
  rate RS correction has to handle, but rarely eliminates it entirely (gimbals have finite bandwidth and
  cannot cancel high-frequency vibration perfectly), so software RS correction and gimbal stabilization
  are complementary in practice, not substitutes.
- **Why this project's degraded-gyro study matters**: a raw, uncalibrated gyro's bias and noise
  (`gyro_degradation` gate) directly limit how well ANY of the above approaches can correct for rolling
  shutter — this is exactly why production VIO systems estimate gyro bias ONLINE (as part of the same
  filter/optimization that estimates the robot's pose) rather than trusting the raw sensor: the bias
  estimate improves as the system runs, which a one-shot image-correction step like this project's cannot
  do on its own.
