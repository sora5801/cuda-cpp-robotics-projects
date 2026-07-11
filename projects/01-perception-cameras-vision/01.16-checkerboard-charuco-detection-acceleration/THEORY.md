# 01.16 — Checkerboard/ChArUco detection acceleration for auto-calibration rigs: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2).

## The problem — physics & engineering first

### Why a printed planar target at all

A camera is a mapping from 3-D rays to 2-D pixels. To recover that mapping (the pinhole intrinsics
`fx, fy, cx, cy`, and in a full treatment, lens distortion), you need a target whose 3-D geometry
you already know *exactly*, photographed from several *different* orientations. A flat, rigid,
printed checkerboard is the simplest such target: its geometry is a trivial, exactly-known regular
grid (every inner corner sits at `(i*s, j*s, 0)` in the board's own frame, `s` the square size), it
is cheap to manufacture precisely, and — the key physical fact this project's whole detection
pipeline exploits — a checkerboard's inner corners are **saddle points** of image intensity, a
locally very distinctive, sub-pixel-localizable feature (derived below).

Two physical/manufacturing facts matter for why real calibration targets are built the way they
are (grounded further in `PRACTICE.md` section 1):

- **Flatness tolerance.** Zhang's method (and this project's) assumes the target is *exactly*
  planar. A target printed on ordinary paper and taped to a board can bow by fractions of a
  millimeter — at typical calibration distances (0.3–1 m) that bow projects to a fraction of a
  pixel of systematic error, small but nonzero. Precision rigs use rigid substrates (aluminum
  composite, glass) specifically to hold flatness to well under this budget.
- **Print accuracy.** The corner positions this project's DLT/Zhang math treats as ground truth are
  only as good as the printer's own geometric accuracy (inkjet/laser printers have real, measurable
  distortion — a few hundred microns over an A4 sheet is typical for consumer printers).
  Photolithographic or precision-engraved boards (glass, chrome-on-glass) exist specifically to
  push this error source below everything else in the budget.

### What limits corner accuracy physically

Three effects, all present in this project's own synthetic rendering (`scripts/make_synthetic.py`)
and named honestly rather than idealized away:

- **Optical blur.** No real lens is a perfect pinhole; even a well-focused system smears a sharp
  black/white edge over a few pixels (a point-spread function on the order of 1 px at typical
  apertures — see 01.11's sensor-simulation project for the fuller optical treatment). This
  project's renderer applies a 5-tap Gaussian blur for exactly this reason.
- **Sensor noise.** Photon shot noise and read noise add a per-pixel random perturbation (this
  project uses additive Gaussian noise, sigma ≈ 2.5 intensity levels — small relative to the
  ≈190-level black/white contrast, but not zero).
- **Perspective shear.** Under a tilted view, a locally-square neighborhood around a true corner
  becomes a sheared quadrilateral; the saddle-point symmetry this project's detector assumes (see
  "The math" below) is only *approximately* true near the corner and degrades with tilt — one of
  the two root causes of this project's own measured grid-ordering limitation (see "Numerical
  considerations").

This project is purely computational from here on (no further physical carrier), so the remainder
of this document is math, algorithm, and GPU mapping — but every numerical choice below is
justified against these three physical error sources, not picked arbitrarily.

## The math

### Notation

- Board-plane coordinates `(X, Y)`, meters, in the board's own frame, `Z=0` always (planarity).
- Pixel coordinates `(u, v)`, `u` rightward, `v` downward (the repo's optical convention,
  `docs/SYSTEM_DESIGN.md` §3.2).
- `K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`, the pinhole intrinsic matrix (this project assumes
  zero skew, matching the physical camera used to render `data/sample/`).
- A homography `H` (3x3, up to scale) maps board-plane homogeneous points to pixel homogeneous
  points: `[u, v, 1]^T ~ H [X, Y, 1]^T`.

### The X-corner as a saddle point

Near a true checkerboard corner, intensity locally looks like a 4-quadrant "pinwheel": two
diagonally-opposite quadrants are bright, the other two dark. Model this locally as
`I(x, y) ≈ I0 + a*x*y` (a pure bilinear saddle, `x, y` local coordinates centered at the corner).
The Hessian of this model is

```
H = [ Ixx  Ixy ]  = [ 0   a  ]
    [ Ixy  Iyy ]    [ a   0  ]
```

with `det(H) = -a^2 < 0` for any `a != 0` — a saddle, by definition (one positive and one negative
eigenvalue/principal curvature). **Contrast with 01.04's Harris detector:** Harris builds a
*structure tensor* from FIRST derivatives, `M = sum (grad I)(grad I)^T` over a window, and flags a
corner where BOTH eigenvalues of `M` are large — geometrically, "gradient energy in every
direction", which fires on an "L" where two straight edges meet at any angle. The Hessian-saddle
test here is a fundamentally different object: it looks at SECOND derivatives and specifically
wants one positive, one negative curvature — the signature of the 4-quadrant pinwheel, not an
L-corner (whose Hessian, away from the vertex itself, is near zero — the surface is locally
flat/planar in each region).

This project's saddle response, computed by finite differences at step `s` (`kSaddleStep`):

```
Ixx = I(x+s,y) - 2*I(x,y) + I(x-s,y)
Iyy = I(x,y+s) - 2*I(x,y) + I(x,y-s)
Ixy = [I(x+s,y+s) - I(x+s,y-s) - I(x-s,y+s) + I(x-s,y-s)] / 4
response = max(0, -(Ixx*Iyy - Ixy^2))
```

the standard central-difference second-derivative and mixed-partial estimators, direct discrete
analogues of the continuous Hessian above.

### The gradient-orthogonality sub-pixel refinement (`cornerSubPix`)

At a TRUE corner `c`, the image gradient `g(q)` at any nearby point `q` is (in the noise-free
limit) orthogonal to the vector from `q` to `c` — intuitively, gradients point ALONG the edges that
meet at the corner, never tangentially past it. This gives, for every sample `q_k` with gradient
`g_k`:

```
g_k . (q_k - c) = 0   =>   g_k . q_k = g_k . c
```

Stack this over a window of `M` samples and solve the least-squares system for `c`:

```
[ sum g_k g_k^T ] c = sum g_k g_k^T q_k
        G                    b
```

a 2x2 linear system, solved here by Cramer's rule (cite 33.01's batched-small-solve pattern — this
is the same idea at the smallest possible scale, a 2x2 instead of an NxN Jacobian). Because `c`
appears inside `q_k` implicitly (the window is centered at the CURRENT estimate), this is a
fixed-point iteration: solve for a new `c`, re-center the window there, repeat (`kRefineIters = 5`
times — THEORY "Numerical considerations" discusses why a FIXED count, not a convergence
threshold).

### DLT homography (Hartley-normalized)

Given `n >= 4` correspondences `(X_k, Y_k) <-> (u_k, v_k)`, each gives two linear equations in the
9 unknowns of `H` (row-major `h0..h8`):

```
h0 X + h1 Y + h2  - h6 X u - h7 Y u - h8 u = 0
h3 X + h4 Y + h5  - h6 X v - h7 Y v - h8 v = 0
```

Fixing `h8 = 1` (valid as long as `H`'s true `(3,3)` entry is not near zero, which Hartley
normalization below guarantees) turns this into an 8-unknown least-squares system, solved by
Gaussian elimination with partial pivoting over the `8x8` normal matrix accumulated from all `n`
correspondences (this project's `solve_gauss_partial_pivot`, the same algorithm 01.06's DLT solve
uses, generalized to a runtime `N`).

**Why normalize first (Hartley 1997):** the raw design matrix mixes wildly different scales — board
coordinates in meters (`~0.01–0.24`) against pixel coordinates (`~0–320`) — which makes the
normal-equation matrix needlessly ill-conditioned (entries spanning many orders of magnitude in
their squares and products). Translating each point set to its own centroid and rescaling so the
average distance from that centroid is `sqrt(2)` (this project's `bs`/`ps` scale factors) puts every
entry of the design matrix in a comparable numeric range, for free — a pure change of coordinates,
undone exactly by the final `H = T2^-1 H_norm T1` denormalization.

### Zhang's absolute-conic linear method

Let `omega = K^-T K^-1` (the "image of the absolute conic", a symmetric 3x3 matrix,
`b = [B11, B12, B22, B13, B23, B33]` its six independent upper-triangular entries). Because a
homography's first two columns `h1, h2` are `K` times the first two (orthonormal) columns of a
rotation matrix, scaled by the same factor `lambda`:

```
h1 = lambda K r1,   h2 = lambda K r2,   r1 . r2 = 0,   |r1| = |r2|
```

two facts about `omega` follow directly:

```
h1^T omega h2 = lambda^2 r1^T K^-T K^-1 K r2 = lambda^2 r1^T r2 = 0
h1^T omega h1 = lambda^2 r1^T r1 = lambda^2 = r2^T r2 = h2^T omega h2
```

Both are LINEAR in `b` (any bilinear form `p^T omega q` expands to `v_pq . b` for a fixed vector
`v_pq` built from `p, q`'s entries — this project's `v_pq()` helper). Each homography therefore
contributes 2 rows to a homogeneous linear system `A b = 0`; stacking `>= 3` homographies
(2 rows each) gives an overdetermined system whose solution (up to scale) is the eigenvector of
`A^T A`'s SMALLEST eigenvalue — found here by a cyclic Jacobi eigenvalue sweep (`Golub & Van Loan`;
`jacobi_eigen_symmetric6`), a classical, numerically stable method for small symmetric matrices:
repeatedly pick the largest off-diagonal entry, apply a rotation that zeroes it (a similarity
transform, so eigenvalues are preserved), and accumulate the rotations into an eigenvector matrix.
After convergence the diagonal holds the eigenvalues and the corresponding columns of the
accumulated rotation product are the eigenvectors.

Zhang's closed-form intrinsics extraction (his paper, Appendix B) then recovers, in order:

```
cy    = (B12 B13 - B11 B23) / (B11 B22 - B12^2)
lambda = B33 - [B13^2 + cy(B12 B13 - B11 B23)] / B11
fx    = sqrt(lambda / B11)
fy    = sqrt(lambda B11 / (B11 B22 - B12^2))
skew  = -B12 fx^2 fy / lambda
cx    = skew * cy / fy - B13 fx^2 / lambda
```

each a direct algebraic rearrangement of the definitions above once `b`'s sign is fixed so
`B11 > 0` (an unavoidable `+-` ambiguity of any eigenvector).

## The algorithm

Per view (`kernels.cuh`'s file header gives the full pipeline overview):

1. **Saddle response** — `O(W*H)` per view, `O(B*W*H)` total, embarrassingly parallel.
2. **NMS + compaction** — `O(W*H)` per view, each thread scans a fixed `(2r+1)^2` window.
3. **Sub-pixel refinement** — `O(candidates)`, each thread does `kRefineIters` fixed-window passes.
4. **Plain grid ordering** (host, serial, `reference_cpu.cpp`'s `order_grid_for_view`) — RETIRED as
   the pipeline's output of record, kept only as the ambiguity-lesson comparison baseline: for each
   of up to 12 candidate seed corners (ranked "most top-left-ish" first), find two roughly-
   orthogonal nearest-neighbor directions, walk BOTH ways along each (bidirectional), disambiguate
   the 7-corner axis from the 5-corner axis by LENGTH, fix handedness so the two axes never combine
   into a physically-impossible reflection, DLT-fit a homography from the two chains, and predict +
   snap every remaining grid slot. `O(seeds * offsets * corners)` per view — negligible next to
   stage 1, but fragile under combined tilt/rotation/occlusion (see "Numerical considerations").
5. **Marker-first grid ordering** (host, serial, `reference_cpu.cpp`'s
   `order_grid_marker_first_for_view`) — THE pipeline's output of record:
   1. **Fix axis identity once per view** (`estimate_view_axes`) — a walk-free, order-independent
      statistic over EVERY detected corner's own nearest-neighbor direction (doubled-angle
      averaging to avoid a real cancellation bug this project's own build hit — see "Numerical
      considerations") plus each axis's own pixel extent, exploiting the board's non-square aspect
      ratio (7 vs. 5) the same way the retired algorithm's chain-length disambiguation does, but
      over the WHOLE point cloud instead of one fragile walked chain.
   2. **Per candidate corner as a quad seed** — find its nearest neighbor and the most-orthogonal
      neighbor within a magnitude band (reused from the retired algorithm's own local search, with
      NO fallback to an unrestricted search — a real bug found and fixed, "Numerical
      considerations"), predict the diagonal 4th corner and snap it within a tight tolerance,
      classify which neighbor plays the board's i-role from step 1's axes (refusing any quad whose
      pick is not clearly won), and check HANDEDNESS explicitly (another real bug: axis identity
      alone does not guarantee a proper, non-reflected local frame).
   3. **Decode** — fit the quad's own tiny one-square DLT homography and test every dictionary code
      under the identity/180-degree-mirror hypothesis (the ONE ambiguity this dictionary's own
      min-distance design protects against — not a brute-forced axis-transpose hypothesis, a real,
      measured false-accept risk this project's own build found and reports honestly). Requires an
      EXACT (Hamming-0) match, and a UNIQUE winner (a second pass catches genuinely
      mirror-self-symmetric dictionary codes, resolved from the view's other, unambiguous quads'
      consensus instead of guessed).
   4. **Vote + extend** (host) — a corner reached by more than one quad needs either unanimous
      agreement or a strict majority; a SECOND, different conflict (two DIFFERENT corners both
      independently winning a label for the SAME `(i,j)` slot) is resolved by keeping the
      stronger-supported one. Every marker-anchored correspondence then fits ONE global homography,
      predicting + snapping every remaining corner (same tight-tolerance discipline as the retired
      algorithm), refit once more from the final correspondence set.
   `O(quads * codes)` per view (`<= 48` decode attempts per quad-seed) — still negligible next to
   stage 1 (README "Expected output" reports the measured total).
6. **Marker decode** (GPU, twinned) — `O(views * 24)`, each thread samples 2 x 25 points through the
   PLAIN algorithm's homography and compares two hypotheses (identity, 180-degree mirror) against
   the dictionary — proving the decode PRIMITIVE correct, independent of which ordering strategy the
   pipeline uses (`order_grid_marker_first_for_view` calls the same, already-proven CPU half of this
   twin directly, at new arguments — kernels.cuh's independence note explains why that is safe).
7. **Zhang** (host) — one 6x6 eigensolve, from the marker-first-exact views' final homographies.

## The GPU mapping

Stages 1–3 and 6 are GPU kernels; stages 4-5 (both grid-ordering strategies) and 7 (Zhang) are host
code — but the Amdahl argument for keeping them there is WEAKER than it used to be, and this
project measures that honestly rather than repeating the old claim unchanged (CLAUDE.md §8 "never
fabricate"). On the reference machine, stages 1–3 (the pixel-parallel work,
`O(8*240*320) ≈ 614,000` pixels) take a combined **~0.7-0.9 ms** on the GPU; the host-side grid-
ordering pass for all 8 views — running BOTH the retired plain walk and marker-first ordering's own
brute-force-over-24-codes decode search per local quad — measures **roughly 6-7 ms** (several runs
on the reference machine), i.e. it is now
the SINGLE LARGEST piece of wall-clock time in the whole pipeline, several times the combined GPU
stages, not "negligible" the way the retired algorithm's grid-ordering-alone pass was (tens of
microseconds). This project still keeps it host-side, but for a DIFFERENT, honestly-stated reason
than pure Amdahl dominance: at this problem size (8 views, ~30 candidates each, 24 codes) 6 ms is
still fast enough for the didactic offline-calibration use case this project targets (README
"System context": calibration runs at manufacturing cadence, not sensor rate — even 100 ms would be
fine), and the serial, branch-heavy, small-`n` nature of the per-quad search (variable-length
neighbor searches, early-exit tie checks) does not map cleanly onto a GPU kernel without real
redesign. README "Exercises" names the natural next step: since each quad-seed's own decode search
is independent of every other seed's, a GPU kernel with one thread per `(view, candidate-seed)` pair
computing the whole `<=48`-attempt decode is a legitimate, measurable follow-up port — unlike the
RETIRED algorithm's inherently sequential seed-and-walk search, which resists this kind of
parallelization.

**Thread-to-data mapping, stage by stage:**

- **Saddle response / NMS:** a single flat grid-stride loop over `idx in [0, num_views*kViewPixels)`.
  `batch_pixel_index()` is the ONE formula every kernel uses to decompose `idx` into
  `(view, x, y)` — deliberately not a 3-D `<<<grid, block>>>` launch, because adjacent `idx` values
  are adjacent pixels within the SAME view (coalesced global memory access), and the flat form lets
  the SAME kernel process the 8-view rig batch or the negative control's batch-of-1 with no special
  casing (`main.cu` calls it both ways). Global memory only — no shared memory: each thread's stencil
  is entirely private, and adjacent threads' stencils overlap by only a few pixels, not enough reuse
  to justify shared-memory tiling at this problem size (README "Exercises" 5 asks the learner to
  measure this trade-off for the LARGER sub-pixel-refinement window instead, where the answer is
  less clear-cut).
- **Sub-pixel refinement:** one thread per candidate, candidates flattened across all 8 views
  (each candidate carries its own `view` field so the thread knows which image slice to read).
  Registers only: each thread's `11x11`-ish window sum (`G00, G01, G11, bx, by`) is a handful of
  doubles, entirely private, `kRefineIters` times.
- **Marker decode:** one thread per `(view, marker_id)` pair — only 192 threads total, launched as
  a single block-grid pair (`block=64, grid=3`) since the work is trivially small; each thread's
  `2 x 25` bilinear samples are private, no shared memory.

**Occupancy:** stages 1–2 process `~614,000` pixels with `block=256` — thousands of blocks, easily
saturating an RTX 2080 SUPER's 46 SMs many times over. Stage 3 (a few hundred candidates) and stage
5 (192 threads) are far too small to saturate the GPU alone — they are cheap enough in absolute
terms (sub-millisecond, measured) that under-occupancy costs nothing observable here, a fact worth
naming rather than hiding (a "real" high-throughput calibration rig batching MANY boards at once
would restore full occupancy at these later stages too).

## Numerical considerations

### Precision & determinism

Saddle response and NMS operate on exact small integers (pixel intensities `0..255`, sums/products
well under float32's `2^24` exact-integer range) — GPU and CPU compute BIT-IDENTICAL results
regardless of FMA contraction (measured: `max|gpu-cpu| = 0.000000` every run). Sub-pixel refinement
and marker decode involve bilinear interpolation (genuine floating-point division/multiplication),
so a tight-but-nonzero tolerance is used (`0.05 px`, `1` Hamming-distance bit) — both comfortably
clear their measured maxima (`0.00004 px`, `0` bits) with real margin.

### Two false-positive confounds this project's own build found and fixed

**T-junction three-region confound.** The board's own outer silhouette, where the top row of
alternating black/white squares meets the (uniform) background, forms a three-region junction
(background above; two DIFFERENT-colored squares below). A naive `det(Hessian) < 0` test — even
combined with a per-axis curvature floor — can still fire there: the asymmetric intensity pattern
can produce a negative determinant with real curvature on both axes, without being a genuine
two-color DIAGONAL saddle. The fix (kernels.cuh's `kMaxDiagonalAsymmetry`): require the two
DIAGONALLY-OPPOSITE sample pairs to each be near-equal in intensity (`|NW-SE| <= threshold`,
`|NE-SW| <= threshold`) — the actual geometric definition of a checkerboard saddle's two-color
symmetry, which a T-junction structurally cannot satisfy on both pairs at once. This measurably
collapsed spurious candidate counts (an early build saw 200+ candidates per view against ~35 real
corners; the fix brought typical counts to 15–40).

**The axis-identity, handedness, and anchor-offset bugs (RETIRED plain-checkerboard grid
ordering).** Three real, distinct bugs surfaced while validating this project's own committed
views against `order_grid_for_view` (kept only as the ambiguity-lesson comparison baseline — see
below), each narrated in `reference_cpu.cpp`'s comments at its fix site (CLAUDE.md §6 "narrate the
thought process, including the one that failed"):

1. *Axis-identity swap* — nothing about "the nearest neighbor" (direction 1) knows whether it
   found the board's 7-corner (X) axis or its 5-corner (Y) axis; labeling whichever direction was
   found first as "i" regardless produced a silent transpose relative to ground truth. Fixed by
   using chain LENGTH to disambiguate (the board is deliberately non-square).
2. *Handedness* — even with axis identity fixed, each axis's own SIGN (which end is "small index")
   is independently unconstrained; flipping exactly one axis's sign is a REFLECTION, which never
   happens to a rigid board viewed from the front (only a 180-degree ROTATION — flipping BOTH
   signs together — is physically possible, and is the intended ambiguity lesson). Fixed by pinning
   the 2-D cross product of the two axis directions to a consistent sign.
3. *Anchor-offset cascade* — a chain shorter than its axis's known full length (a genuinely
   undetected corner at the true boundary) leaves the walk's own "where is index 0" anchor
   ambiguous; a too-generous match tolerance in the predict-remaining phase let a neighboring
   REAL corner get mislabeled to fill the gap, cascading an off-by-one down the rest of that line.
   Fixed by (a) tightening the match tolerance well below one grid step, and (b) searching over
   every plausible anchor shift when a chain is short, keeping whichever placed the most corners.

Even with all three fixes, this RETIRED algorithm achieves EXACT ordering on only 3 of the
project's 8 committed views: sparse candidate sets (fewer real corners survive detection under
combined tilt/rotation/occlusion) leave the seed-and-direction search with less redundancy, and —
the deepest limitation — "most corners placed" is not a perfect proxy for "most CORRECTLY placed":
a wrong-but-internally-self-consistent labeling can occasionally place as many or more points than
the correct one, because DLT does not know the CANONICAL scale/index a chain "should" represent,
only that SOME labeling fits the points handed to it. This gap — not a mystery, a genuine limit of
checkerboard-only geometry — is exactly why marker-first ordering (below) replaced it as the
pipeline's output of record, the production ChArUco strategy (README "Prior art").

### Real bugs marker-first ordering's own build found and fixed

`order_grid_marker_first_for_view` (`reference_cpu.cpp`) replaced the plain walk above with markers
decoded independent of any global corner walk (kernels.cuh's file header and "The algorithm" walk
every step). Getting it RIGHT took several real, measured bugs — narrated here in full per CLAUDE.md
§6, because each one teaches something the "obvious" first design misses:

1. **A malformed "diagonal" quad from an unrestricted fallback search.** The first version copied
   `try_order_from_seed`'s own "if the magnitude-banded orthogonal-neighbor search finds nothing,
   retry unrestricted" fallback verbatim. For a WALKED CHAIN this is a minor accuracy hit; for a
   single 2x2 LOCAL quad it is much worse: when the true axis-2 neighbor is missing (a corner the
   saddle detector missed), the unrestricted fallback regularly locked onto a DIAGONAL point
   instead of a true axis neighbor — silently building a "quad" that was not a real one-square unit
   cell at all. The resulting tiny DLT homography read every marker cell a fraction of a cell off,
   producing the measured symptom: every attempted quad on one committed view (view04) landing 1-2
   payload bits short of ANY dictionary code, never zero, on every seed. Fixed by REFUSING the
   fallback outright — a local quad this function cannot confirm as genuine is skipped, not guessed.
2. **Cancellation in the view-wide axis estimate.** `estimate_view_axes` averages every corner's own
   nearest-neighbor direction to find the view's two lattice axes (a robust, walk-free replacement
   for the retired algorithm's chain-length cue). A first version averaged raw `(cos,sin)` unit
   vectors directly — but two corners can legitimately find their nearest neighbor along the SAME
   physical line in OPPOSITE senses (one seed's neighbor to its "east", another's to its "west"),
   and naively averaging those CANCELS instead of reinforcing. The fix is the standard circular-
   statistics trick for AXIAL (undirected-line) data: average in DOUBLED-angle space
   (`cos(2*theta), sin(2*theta)`), where a direction and its 180-degree opposite map to the
   IDENTICAL point, then halve the result back down.
3. **Brute-forcing axis identity per quad is unsafe against THIS dictionary.** An early version,
   lacking a view-wide axis estimate, tried BOTH possible axis assignments per local quad and let
   the marker decode itself pick the winner — symmetric with how the 180-degree hypothesis is
   already resolved. MEASURED, this failed badly: `scripts/make_synthetic.py`'s own
   `generate_marker_dictionary()` docstring says the dictionary was only ever built to separate its
   24 codes from each other under the identity/180-degree reading, never against a TRANSPOSE — and
   a one-off audit of the committed dictionary (`data/sample/marker_dictionary.csv`) found EXACT
   (Hamming-0) cross-code collisions under transpose (e.g. marker 5's transpose+mirror reading is
   bit-identical to marker 23's own code). The fix: resolve axis identity GEOMETRICALLY (bug 2's
   `estimate_view_axes`, computed once per view, never per quad) instead of by brute force, so only
   the ONE ambiguity the dictionary's own min-Hamming-distance design DOES protect (180-degree) is
   ever decode-tested.
4. **The dictionary's own permissive `correction_capacity` is a liability when searched, not just
   used.** Even with axis identity fixed geometrically, accepting any of the 24 codes within the
   dictionary's own 1-bit correction capacity (rather than requiring an exact match) was tried and
   MEASURED to make several already-correct views WORSE, not better — sensor-noise tolerance meant
   for a single, already-known-correct hypothesis becomes a false-accept liability when searched
   across 24 candidates. Fixed by requiring an EXACT (Hamming-0) match for this local search
   specifically, ignoring the dictionary's own capacity value entirely.
5. **A REFLECTED (not just mis-assigned) local frame.** Classifying which quad neighbor plays the
   i-role from the view-wide axis estimate (bug 2/3) does not, by itself, guarantee the resulting
   `(seed, i-neighbor, j-neighbor)` triple is PROPERLY handed. A real camera image of a flat board
   is never reflected (only a 0- or 180-degree in-plane ROTATION is physically possible), so
   `cross(i-direction, j-direction)` has one fixed sign, measured directly from a frontal view — but
   nothing forced the axis pick to respect it. A wrongly-handed (reflected) quad can still decode
   "successfully" if the specific marker it lands on happens to be symmetric under that specific
   single-axis flip, producing a confident but WRONG label. Fixed by computing the cross product
   explicitly and refusing any quad with the wrong sign.
6. **A marker code that is genuinely orientation-symmetric.** Two of this dictionary's 24 codes
   read bit-for-bit IDENTICAL whether sampled identity or 180-degree-mirrored (a pure, precomputable
   fact about the 9-bit payload, independent of any image). For such a marker, no amount of image
   evidence can ever resolve ITS OWN orientation from itself alone — an early version silently
   defaulted to "identity" (whichever hypothesis its decode loop happened to check first), which is
   wrong whenever the view is actually rotated, and was traced directly to a clean, symmetric
   NEIGHBOR-SWAP pair (truth corners `(0,1)` and `(1,1)` trading labels on view06). Fixed with a
   two-pass design: quads anchored by NON-symmetric markers vote on the view's overall orientation
   first; symmetric-marker quads (which confidently identify their own SQUARE but not their own
   ORIENTATION) borrow that vote afterward, rather than guessing alone.
7. **Two DIFFERENT corners can each independently win a label for the SAME `(i,j)` slot.** The
   per-corner majority vote (every proposal FOR ONE corner index must agree) cannot see this: it
   only checks consistency PER CORNER, never across corners. A genuinely inconsistent quad — its
   diagonal corner accidentally coinciding with a neighboring quad's own seed — can win its OWN
   local majority while claiming a board position another corner ALSO won. Fixed by a second pass:
   group every anchored corner by its label, and where more than one corner claims the same slot,
   keep only the one with more supporting votes, un-anchoring the rest (recoverable, honestly, by
   the predict-and-snap extension phase instead of guessed).

**What remains, honestly measured (not a mystery, two DIFFERENT, named limits):** with all seven
fixes, marker-first ordering achieves EXACT ordering on **6 of the project's 8 committed views** —
every category the retired algorithm's own limitation named (large tilt, the 180-degree rotation,
occlusion) now resolves correctly. The 2 that remain are NOT grid-ordering fragility carried over
from the retired algorithm:

- **view01** has only 5 raw candidate corners survive the UNCHANGED, independently-verified
  saddle/NMS stage (`corner_accuracy` gate 1 is unaffected and still passes) — a stage 1-2
  characteristic entirely out of this rewrite's scope. No local-quad algorithm can form even one
  2x2 cell from 5 scattered points.
- **view04**'s local quads are all geometrically sound (properly axis-aligned and handed — bugs 1
  and 5 above are both cleanly fixed here), but every one lands 1-2 payload bits short of an exact
  match: this project's own measured ~0.7px mean corner-refinement noise (`corner_accuracy` gate),
  averaged over only 4 correspondences instead of a whole board's worth, is enough for this view's
  specific geometry to tip a marker cell across the black/white threshold. Requiring an EXACT match
  (bug 4) is the right call in aggregate (it is what makes the OTHER 6 views reliable), but it costs
  this one view's yield — a genuine, named trade-off, not hidden.

Production systems close the view04-style gap by scoring candidate readings against an INDEPENDENT
signal (reprojection RMS, or averaging several overlapping quads' own homographies before deciding)
rather than trusting any single 4-point fit alone — named here as the honest next step (README
"Exercises"), not hidden.

### Corner-scale vs. marker-scale interference

An earlier design used high-contrast (black=30, white=220) markers occupying most of each white
square; this let the marker's OWN internal bit-boundary corners produce saddle responses as strong
as real board corners (same ink contrast, similar local geometry at a smaller scale), badly
polluting the candidate set. The fix — LOWER-CONTRAST marker ink (black=95, white=155, same
midpoint threshold 125 so decoding is unaffected, since a 60-level gap against sigma=2.5 sensor
noise remains enormously significant) — cuts the marker-internal second-derivative response by
roughly `(60/190)^2 ≈ 10%` of the board's own corner response, while leaving decoding untouched
(marker decode is homography-GUIDED sampling at precisely known locations, not blind detection, so
it never needed high contrast for LOCALIZATION, only for classification).

## How we verify correctness

Two tiers, per the repo's twin-independence ruling (`reference_cpu.cpp`'s file header):

- **Twinned (GPU vs. CPU, independently coded):** saddle response, NMS, sub-pixel refinement,
  marker decode. Compared element-wise / set-wise within the tolerances above.
- **Shared, not twinned, but independently GATED:** grid ordering, DLT, Zhang, and the Jacobi
  eigensolver are single-sourced host functions (duplicating a Gaussian-elimination solve or an
  eigenvalue sweep a second time, byte-for-byte the same algorithm, would be pure transcription,
  not independent verification). Instead, `main.cu` gates their OUTPUT against a THIRD, independent
  source of truth: `scripts/make_synthetic.py`'s own recorded corner positions, poses, and camera
  intrinsics (a different language, a different implementation, computed before the C++ pipeline
  ever runs) — exactly the "independent gate" tier the ruling requires whenever code is shared
  rather than duplicated.

Every gate ceiling in `main.cu` is MEASURED from an actual run on the reference machine and
margined, never assumed (CLAUDE.md §8): `kCornerAccMeanGateTolPx`, `kZhangFxFyPctTol`,
`kMinExactOrderedViews`, and friends all carry a comment naming the measured value they were set
against.

## Where this sits in the real world

OpenCV's `findChessboardCorners` (checkerboard) and `aruco::CharucoDetector` (the full ChArUco
pipeline) are the production versions of this project's stages 1–7. `aruco::CharucoDetector`
specifically validates the SAME strategic choice this rewrite converged on independently: it runs
ArUco marker detection FIRST (contour-based quad candidates, not saddle-point corners), decodes
each marker's identity from its own 4 corners, and only THEN infers the checkerboard's inner-corner
grid from the union of detected markers — exactly "markers anchor identity, corners fill in the
rest" this project's `order_grid_marker_first_for_view` reimplements at a smaller scope. Where
production still goes further: a much more elaborate corner detector (adaptive multi-scale
thresholding, a validated quadrangle-growing algorithm, not a single fixed-threshold saddle test),
RANSAC-robust marker detection (many candidate quads per frame, scored and pruned, rather than this
project's tight local-quad search from already-detected saddle points), and — critically for
production use — `calibrateCamera` fits the FULL Zhang model including radial (`k1, k2, k3`) and
tangential (`p1, p2`) lens distortion via nonlinear (Levenberg-Marquardt) refinement AFTER this
project's linear stage, typically achieving sub-0.1-pixel mean reprojection error on a well-behaved
rig — and, relevant to this project's own measured `fx`/`fy` gap (README "Expected output"), a real
bundle adjustment WEIGHTS each view's contribution by its own residual/covariance rather than
stacking every homography's constraint rows unweighted the way this project's linear-only
`solve_zhang_calibration` does; that weighting is exactly what would stop an occluded or
extreme-pose view from pulling the joint fit off, the gap this project measures honestly rather than
hides (README "Limitations & honesty", "Exercises"). Kalibr (ETH Zurich) extends this further to
multi-camera and camera-IMU rigs, jointly optimizing extrinsics and time offsets. A real factory EOL
calibration station additionally controls for things this project's synthetic renderer sidesteps
entirely: target illumination uniformity, vibration during capture, and — the whole reason
`PRACTICE.md` exists — target manufacturing tolerance and mounting repeatability.
