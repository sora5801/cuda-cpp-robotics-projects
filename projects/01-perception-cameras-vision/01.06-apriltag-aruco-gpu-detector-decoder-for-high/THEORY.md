# 01.06 — AprilTag / ArUco GPU detector-decoder for high-rate fiducial localization: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2).

## The problem — physics & engineering first

**Why a printed black square works so well as an optical target.** A fiducial marker is, physically,
just a piece of paper (or laminate) with matte black ink on a matte white/light substrate. Three
physical facts make that humble object an excellent machine-vision target:

1. **Albedo contrast, not color.** Matte black ink reflects roughly 3-6% of incident light (albedo
   ~0.03-0.06); a light substrate reflects 70-90% (albedo ~0.7-0.9). This is more than an order of
   magnitude of *reflectance* contrast — far larger than typical scene texture contrast — and,
   critically, it survives almost any COLOR cast in the illuminant (a black ink stays dark under
   incandescent, fluorescent, or LED light; a colored object's apparent hue does not). That is why
   this project's whole pipeline works in GRAYSCALE, never touching hue: the signal that matters is
   reflectance, not spectral color.
2. **Lambertian (matte) surfaces are robust to viewing angle.** A Lambertian reflector's apparent
   brightness is *independent of viewing direction* (only the incident-light geometry matters, via
   Lambert's cosine law: radiance proportional to `cos(theta_incidence)`). A glossy/specular tag would
   show a bright highlight that MOVES with viewpoint and can wash out the pattern entirely from some
   angles — a real, practical reason production tags are printed on matte media and laminated with
   matte, not gloss, film (PRACTICE.md §1 returns to this).
3. **The solid border is a self-locating edge.** A single, unbroken black frame around the payload
   gives the detector one continuous high-contrast boundary to find — a purely geometric feature (an
   edge, then a corner) that does not depend on correctly reading a single bit first. This project's
   pipeline finds tags via their BORDER (connected-component labeling on "dark" pixels) entirely before
   attempting to read the payload — the border does the finding, the payload does the identifying.

**Failure physics, named honestly.** Three real-world effects break this simple picture, and this
project's synthetic scenes model the first two: (a) **non-uniform illumination** — a light source
brighter on one side of the frame than the other (this project's `apply_illumination_gradient`)
defeats any single GLOBAL brightness threshold, which is exactly why stage 1 below is *adaptive*, not
global; (b) **optical blur and sensor noise** — defocus, motion blur, and photon/read noise all soften
the sharp black/white edges a printed tag has on paper into gradual ramps a few pixels wide (this
project's 5-tap Gaussian blur models the former, additive Gaussian noise the latter); (c) **specular
glare and print defects** (lamination glare, ink cracking, dirt) are NOT modeled here — a scoped
limitation named in README "Limitations & honesty".

## The math

**Notation.** Pixel coordinates `(x, y)`, `x` rightward, `y` downward, row-major linear index
`i = y*W + x` (matches every PGM in this repo). Camera uses the OPTICAL convention (x-right, y-down,
z-forward — SYSTEM_DESIGN.md §3.2's stated exception). Pinhole intrinsics `K = [[fx,0,cx],[0,fy,cy],
[0,0,1]]`, `fx=fy=350 px`, `cx=239.5, cy=179.5 px` (exact image center). Tag frame: right-handed,
X-right, Y-down (matching the image when viewed face-on), Z pointing away from the printed face
(toward the camera when the tag faces it); a point on the tag's own plane is `(X, Y, 0)` meters, with
`X, Y` in `[-halfSize, +halfSize]`, `halfSize = 0.08 m`.

**The 6x6 grid.** The tag is divided into a `kGridN x kGridN = 6x6` array of equal cells, cell size
`cellSize = tagSize / 6`. Cell `(r, c)`, `r, c` in `[0, 6)`, is a BORDER cell iff it touches the outer
ring (`r in {0,5}` or `c in {0,5}`, `kBorderCells = 20` such cells) and is a PAYLOAD cell otherwise
(the interior `4x4 = 16` cells, `payload_bit_index(r,c) = (r-1)*4 + (c-1)` for `r,c` in `[1,4]`, a
single source-of-truth formula in `src/kernels.cuh`). Cell `(r,c)`'s CENTER in tag-frame meters is
`X = -halfSize + (c+0.5)*cellSize`, `Y = -halfSize + (r+0.5)*cellSize`.

**Homography.** A planar point `(X, Y, 0)` in the tag's own frame is related to the camera frame by a
rigid transform `p_cam = R*(X,Y,0) + t = X*r1 + Y*r2 + t`, where `r1, r2` are the FIRST TWO COLUMNS of
the 3x3 rotation `R` (the third column, `r3`, never appears — a planar point has no `Z` component to
multiply it by). Projecting through the pinhole model, `pixel_homog = K*p_cam = X*(K r1) + Y*(K r2) +
K t = H*(X,Y,1)^T`, where
```
H = K [ r1  r2  t ]     (a 3x3 matrix, columns K*r1, K*r2, K*t)
```
is the tag's HOMOGRAPHY: the single 3x3 matrix (up to overall scale) that maps every point on the
tag's own plane directly to its pixel location, with NO knowledge of `R`/`t`/`K` individually needed
once `H` is known. This is the central mathematical fact the whole pipeline exploits: a PLANAR target
under perspective projection is fully described by 8 numbers (`H` has 9 entries but is only defined up
to scale), and those 8 numbers are recoverable from just 4 point correspondences.

**DLT (Direct Linear Transform).** Given 4 correspondences `(X_k, Y_k) <-> (x_k, y_k)`, `k=0..3` (tag
model corners <-> detected image corners), fixing the free scale by setting `h33 = 1` turns
`x*(h31 X + h32 Y + h33) = h11 X + h12 Y + h13` and the analogous `y` equation into TWO LINEAR
equations per correspondence in the 8 unknowns `[h11,h12,h13,h21,h22,h23,h31,h32]`:
```
h11*X + h12*Y + h13            - h31*(x*X) - h32*(x*Y) = x
                     h21*X + h22*Y + h23 - h31*(y*X) - h32*(y*Y) = y
```
4 correspondences give exactly 8 equations for 8 unknowns — a square, generically well-posed linear
system, solved here by Gaussian elimination with partial pivoting (the same small-dense-linear-solve-
per-thread pattern 33.01 teaches for robot kinematics Jacobians, applied here to 2-D projective
geometry instead).

**Pose from homography.** Given `H` and the KNOWN intrinsics `K`, `M = K^-1 H = [r1 r2 t] * s` for
some unknown scalar `s` (the scale DLT's `h33=1` normalization introduced is *a priori* unrelated to
the physical scale that makes `r1,r2` unit vectors). Since `|r1| = |r2| = 1` for a true rotation,
`s = 1 / |m1| ~= 1 / |m2|` (the two estimates agree exactly only in the absence of numerical/
homography noise — this project averages them: `s = 2/(|m1|+|m2|)`). Then `r1 = s*m1`, `r2 = s*m2`,
`r3 = r1 x r2` (completing a right-handed orthonormal frame via the cross product), `t = s*m3`. A
sign check (`t_z > 0`, tag in front of the camera) resolves `H`'s inherent `+-1` scale ambiguity.

**Coding theory: why a fiducial dictionary needs a minimum Hamming distance.** A 16-bit payload can
represent `2^16 = 65536` distinct codes; a dictionary picks a small SUBSET (here 32) such that no
sensor-noise-plausible corruption of one code can be confused with another. Formalized: the HAMMING
DISTANCE between two `n`-bit codes is the number of bit positions where they differ; a code set with
MINIMUM pairwise distance `d` (over ALL rotations of ALL codes, including a code against its own other
rotations — a tag can be mounted at any of 4 quarter-turns) can CORRECT up to
`t = floor((d-1)/2)` bit errors: given a received pattern within Hamming distance `t` of exactly one
dictionary code, that code is the UNIQUE nearest one (any other code is, by the triangle inequality,
at distance `>= d - t > t` away). This project's `scripts/make_synthetic.py` GENERATES such a set by a
seeded greedy search (measured achieved distance: **5**, giving correction capacity **2**) — the exact
design principle real families (AprilTag 16h5, also distance 5; ArUco's 4x4 dictionaries) use,
independently applied here (never their published bit tables).

## The algorithm

1. **Adaptive threshold**, `O(W*H)`: separable box filter (local mean, window `25x25` px) then
   compare `gray < local_mean - bias`. Complexity: `O(W*H*r)` naive per axis with radius `r=12`, i.e.
   `O(W*H)` total for a SEPARABLE filter (vs. `O(W*H*r^2)` for a naive 2-D box) — the classic
   separable-filter saving, `2*(2r+1)` taps per pixel instead of `(2r+1)^2`.
2. **Connected-component labeling**, iterative label propagation: every foreground pixel starts
   labeled with its own linear index and relaxes `label[p] <- min(label[p], min over 4-connected
   foreground neighbors' label)` each sweep until no label changes. CONVERGENCE (full proof in
   `src/kernels.cu`'s file header, identical to 30.01's): every label only ever DECREASES and is
   bounded below by 0, so the process converges in finitely many sweeps to the UNIQUE fixed point
   `label[p] = min{q : q 4-connected to p} q` — independent of thread scheduling. Cost: as many sweeps
   as the component's 4-connected GRAPH DIAMETER (measured on this project's committed scenes:
   148-192 sweeps for `480x360` scenes with tags ~56-125 px across) — asymptotically worse than a
   CPU union-find's near-`O(1)`-amortized convergence (why the CPU oracle uses union-find, not this
   algorithm, as its independent twin — see `reference_cpu.cpp`'s header).
3. **Quad extraction**, `O(#components)` then `O(#candidates)`: (a) a pixel-parallel pass finds, per
   component, 4 "extreme corner" pixels via `argmin/argmax` of `x+y` and `x-y` (packed into 64-bit
   atomics — see "The GPU mapping" below); (b) a candidate-parallel pass refines each corner with a
   short 1-D radial sub-pixel search (bilinear-sampled dark/light crossing) along the ray from the
   component's centroid through its raw extreme pixel.
4. **DLT homography**, `O(1)` per candidate: build the 8x9 augmented system, Gaussian-eliminate with
   partial pivoting (`O(8^3)` = constant work), back-substitute.
5. **Grid decode**, `O(36)` per candidate: sample all 36 cell centers through `H`, threshold each,
   check the border-ring tolerance, try the 4x4 payload against all 32 dictionary codes at 4 rotations
   (`O(32*4)` Hamming-distance comparisons — 128 short integer ops, negligible).
6. **Pose**, `O(1)` per candidate: the closed-form `K^-1 H` decomposition above.

## The GPU mapping

**Two launch geometries, deliberately contrasted** (`src/kernels.cu`'s file header names this as the
project's central GPU-mapping lesson):

- **PIXEL-parallel** (stages 1-2, the stats scatter): `grid = ceil(W*H/256)`, `block = 256` — one
  thread per pixel, the SAME idiom every flagship in this repo uses for dense image work. Memory
  hierarchy: global memory only (no shared-memory tiling — Exercise territory: the box filter's
  25-tap window per axis would benefit from a shared-memory halo tile, named but not implemented, to
  keep this project's kernel count and complexity in the didactic sweet spot).
- **CANDIDATE-parallel** (stages 3-6): `grid = ceil(n/32)`, `block = 32` — `n` is typically single
  digits to a few dozen (measured: 2-6 on this project's committed scenes), so a single warp usually
  covers the WHOLE launch. Occupancy is essentially irrelevant at this scale (a handful of active
  warps on a device with dozens of SMs); what matters is that each thread does a nontrivial SEQUENTIAL
  job (a 36-sample loop, an 8x8 elimination) that itself has no useful internal parallelism to expose
  at this problem size — the lesson is "not every GPU stage needs thousands of threads to be the right
  GPU stage": doing this work on the HOST would mean a device-to-host round trip and serial CPU
  execution; doing it here keeps all data device-resident and lets the (already-parallel) pixel stages
  and this stage pipeline without a host bounce in between.
- **The packed-atomic argmax trick** (`component_stats_accumulate_kernel`): finding the pixel that
  ACHIEVES a component's extreme `x+y` (etc.) in one pixel-parallel pass, without a second reduction
  pass, by packing `(score, pixel_index)` into one 64-bit integer with the score in the HIGH bits —
  ordinary integer `atomicMin`/`atomicMax` on the packed value is then simultaneously an arg-extremum,
  because comparing packed keys compares scores first (ties break on the low-order index bits). Built
  from `atomicCAS` (a manual retry loop, `src/kernels.cu`'s `atomicMin64`/`atomicMax64`) rather than a
  possibly-available native 64-bit min/max, on purpose (CLAUDE.md §1, "no black boxes" — the CAS-loop
  idiom underlies every lock-free atomic operation and is worth seeing built from scratch once).
- **Measured occupancy contrast** (RTX 2080 SUPER, this project's committed `scene_main`): the
  pixel-parallel stages launch `ceil(172800/256) = 675` blocks (fully saturating the GPU's 48 SMs many
  times over); the candidate-parallel stages launch `ceil(6/32) = 1` block (a single warp, 6 of 32
  lanes active) — occupancy near-zero by the usual metric, and correct anyway, because the WORK, not
  the thread count, is what this stage needed to move off the host.

## Numerical considerations

- **Precision:** the box filter accumulates in `float32`; every addend is a `uint8` in `[0,255]` and a
  running sum of up to `2*kBoxRadius+1 = 25` such terms never exceeds `65025`, far below `float32`'s
  `2^24` exact-integer ceiling — the box SUM is bit-EXACT regardless of summation order, on any
  IEEE-754-compliant adder, device or host (measured: `0.0` GPU-vs-CPU difference on every committed
  scene). The homography solve and pose decomposition use `double` throughout (cheap per-thread at
  this problem size, and removes conditioning as a variable — see below).
- **A real bug this project's own build process found and fixed, worth reading as a case study**
  (`src/kernels.cu`'s `refine_one_corner` doc comment tells the full story): an early version of the
  corner-refinement radial search used a MULTIPLICATIVE search margin (`1.6x` the centroid-to-corner
  distance). For a tag whose raw extreme corner sat 58 px from its centroid, that is a 93 px search
  radius — 35 px of which is "extra" beyond the true corner, easily far enough to wander into an
  unrelated dark region and lock onto a spurious, FAR-AWAY "last crossing". Measured effect: one tag's
  refined corner landed 31 px from its true position (on a 56-93 px tag — more than a third of the
  tag's own size). The fix is a small, FIXED additive margin (a few pixels — the scale of the
  anti-aliasing blur, not the tag's own size); after the fix, the same tag's corner error dropped to
  under 3 px. The general lesson: a search-range margin should scale with the UNCERTAINTY you are
  compensating for, never with the magnitude of the quantity you are searching around.
- **Amplification chain: sub-pixel corner noise -> homography -> pose.** DLT fits `H` EXACTLY to its 4
  input corners (no smoothing, no least-squares slack — 4 correspondences for 8 unknowns is exactly
  determined). A sub-pixel difference in one input corner (e.g. the ~0.15 px GPU-vs-CPU difference this
  project measures, from the corner search occasionally landing on a different "last crossing" step
  when a ray grazes a threshold boundary within FMA-contraction-level floating-point noise) therefore
  propagates, UNDAMPED, into `H`'s entries (measured: up to ~1.8 absolute difference in `H`'s
  pixel-scale entries), and further into the pose decomposition's rotation matrix (measured: up to
  ~0.0074 per-entry difference) — each step in this chain is individually well-conditioned for THIS
  project's tag sizes and tilts, but the chain has no smoothing anywhere, so small input noise is
  never damped, only carried forward. This is why `main.cu`'s VERIFY tolerances GROW through the
  pipeline (tight for the pixel stages, looser for corners, looser still for homography and pose) —
  not sloppiness, but an honest reflection of where damping does and does not exist in this pipeline
  (contrast with 01.01's fused-vs-staged tolerance, which is loose for the analogous reason: rounding
  order, not amplification).
- **Homography conditioning:** this project does NOT apply Hartley normalization (rescaling point
  coordinates to unit-ish magnitude before solving, then undoing the scaling after) — the production
  DLT best practice for numerical conditioning. It relies instead on (a) double precision throughout
  the solve and (b) the committed scenes' bounded geometry (tag half-size `0.08 m`, comfortable
  `[-0.08,0.08]` model coordinates; image coordinates in the low hundreds of pixels) keeping the linear
  system's condition number modest without normalization. A tag imaged extremely small (a few pixels)
  or extremely tilted (near grazing incidence) would need normalization; named here as a real
  production-vs-teaching tradeoff, not silently assumed away.
- **Angle/rotation ambiguity, not angle WRAPPING:** this project's corner order (from extreme-pixel
  picking) is arbitrary-but-consistent, not aligned with the tag's TRUE physical orientation — an
  unknown 0/90/180/270-degree offset. Rather than hand-derive the sign/direction convention linking
  "which cyclic shift of ground-truth corners matches the detection" to "which quarter-turn rotation
  the grid decoder matched against the dictionary" (a real risk of a silent sign-convention bug),
  `main.cu`'s `best_shift_corner_error()`/`best_rotation_error_deg()` each try all 4 candidate
  alignments and report the best — an honest, robust choice over a possibly-wrong hand derivation.
- **Determinism:** every RNG in this project (the synthetic-scene generator) is a seeded xorshift32,
  never a library RNG — bit-identical output on every machine, every run (CLAUDE.md §12).

## How we verify correctness

Two tiers, per this repo's twin-independence ruling (`reference_cpu.cpp`'s file header):

- **GPU-vs-CPU VERIFY** (`main.cu`, all 3 scenes): local mean (float, tol `0.02`, expected `~0`
  exactly per the exact-arithmetic argument above), mask and CCL labels (EXACT integer/boolean
  equality — mask because the threshold comparison is a single correctly-rounded op with identical
  operands on both platforms; labels because label propagation and the CPU's union-find both converge
  to the SAME unique fixed point, a genuinely different-algorithm cross-check), candidate statistics
  (exact — integer sums, deterministic packed-key argmax), refined corners (tol `0.30` px), homography
  entries (tol `5.0`, mixed pixel/dimensionless/inverse-pixel scales in one struct), decoded tag ID /
  rotation / accept flag (EXACT) with Hamming distance allowed to drift by at most 1 bit, and pose `R`/
  `t` (tol `0.02` / `0.005 m`) — every tolerance carries real measured-and-margined headroom (README
  "Expected output" states the actual numbers this project's committed scenes produce).
- **Five gates that do NOT route through the GPU-vs-CPU comparison at all** (the twin-independence
  ruling's second tier — a shared bug would pass VERIFY but fail here): `detection` and
  `corner_accuracy` compare against `scripts/make_synthetic.py`'s own rendering math (an independent,
  third implementation, in Python, of the projection/homography math); `pose` compares against the
  SAME script's `(R, t)` used to render the scene — analytic ground truth this pipeline's own code
  never touches; `decode_robustness` reasons about known, hand-computed bit-flip counts and the
  dictionary's own measured minimum distance (a closed-form coding-theory guarantee, not a
  pipeline-internal check); `false_positive` checks a scene this pipeline's own detection code has
  never seen "correct" output for (there is none to compare against — only "did it stay silent").

## Where this sits in the real world

Every production fiducial system (AprilTag 3, OpenCV ArUco, NVIDIA Isaac ROS AprilTag — README "Prior
art" has one line each) differs from this project in the same three places this file has flagged
honestly: (1) CORNER REFINEMENT — gradient-clustered line fitting per edge, not single-ray radial
search, immune to the +/-45-degree weakness named above; (2) POSE — IPPE or a full nonlinear
reprojection-error refinement, not a closed-form column-normalization decomposition, resolving the
planar-homography pose ambiguity this project's pose gate's wide tolerance reflects; (3) DICTIONARIES
— published, standardized code families built with additional algebraic structure (BCH-code-derived
constructions for AprilTag) rather than this project's greedy random search, giving denser code sets
at a given minimum distance. All three are genuine, scoped-out extensions (README "Exercises" 3 and 5
sketch two of them); the ARCHITECTURE this project teaches — pixel-parallel detection feeding
candidate-parallel per-tag refinement — is the one that scales to production frame rates, and is
exactly what Isaac ROS AprilTag's GPU acceleration exploits at industrial scale.
