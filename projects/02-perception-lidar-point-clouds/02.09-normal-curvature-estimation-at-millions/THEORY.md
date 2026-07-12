# 02.09 — Normal + curvature estimation at millions of points/sec: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

**What a surface normal physically means.** At a point on a physical surface, the normal is the unit
vector perpendicular to the surface's local tangent plane, pointing away from the material. It is not a
computational abstraction — it is the single most useful piece of local geometry a robot's contact or
optical model needs:

- **Contact & grasping** (`19.01`): the force a gripper's finger can apply through friction, without
  slipping, is bounded by the **friction cone** around the contact normal (Coulomb friction: the
  tangential force component must satisfy |F_tangential| <= mu * F_normal, where mu is the friction
  coefficient — a real material property, typically 0.3-0.8 for rubber-on-hard-surfaces). Two contact
  normals nearly antiparallel (pointing at each other) let a two-finger grasp resist gravity with force
  closure; two contacts whose normals are nearly parallel cannot. Antipodal grasp scoring IS normal
  geometry.
- **Traversability** (mobile robots, off-road/legged): the angle between a candidate foothold/wheel-
  contact normal and the gravity vector is exactly the local SLOPE — a robot's maximum climbable grade is
  a normal-angle threshold, not an abstract "cost".
- **Optical reflectance**: a LiDAR return's intensity and a camera pixel's shading both depend on the
  angle between the surface normal and the incident ray (Lambertian reflectance ~ cos(angle) — the same
  cosine this project's orientation-disambiguation heuristic uses, for a different purpose).

**Why estimate it from a point cloud instead of measuring it directly.** No sensor measures the normal
directly (a LiDAR measures RANGE along one ray; a stereo camera measures DISPARITY). The normal must be
INFERRED from the local arrangement of nearby measured points — this is fundamentally a **local surface
fitting** problem: given a noisy, finite sample of points believed to lie near a smooth surface, find the
tangent plane that best explains them.

**Engineering constraints a real robot imposes.** A spinning mechanical LiDAR returns points in
**anisotropic rings**, not a uniform 2-D grid: each of N_beams laser rings sweeps a full 360 deg azimuth
per revolution, so along-ring point spacing (azimuth direction) is much finer than cross-ring spacing
(elevation direction) at most ranges — projects [`02.01`](../02.01-voxel-grid-downsampling-with-gpu-spatial-hashing/THEORY.md)
and [`02.05`](../02.05-kd-tree-or-lbvh-construction-knn-radius-search/THEORY.md) derive this ring
structure and its 1/r² density falloff in full. The consequence for THIS project: a K-nearest-neighbor
patch on a real spinning-LiDAR scan is systematically ELONGATED along the ring direction and compressed
across it — the neighborhood is anisotropic even on a perfectly flat surface. A naive covariance fit over
such a patch is biased toward the elongation axis (the fit "sees" more variance along-ring than across-
ring purely from sampling density, not from surface shape), which can bias the estimated normal AWAY
from the true one, especially at long range where the elongation is most severe. This project's own
synthetic data (grid+jitter, near-isotropic per cohort — `scripts/make_synthetic.py`) deliberately
SIDESTEPS this specific bias to isolate and measure the OTHER sources of error (curvature-fit bias,
sensor noise, grazing incidence) cleanly — see "Where this sits in the real world" for what a real
ring-structured scan would add on top.

## The math

**Notation.** Points p_i in R^3, meters, sensor frame (CLAUDE.md §12: origin at the sensor, +x forward,
+z up). For query point p with K nearest neighbors {q_1..q_K} (K = `kernels.cuh`'s `kK` = 16, INCLUDING
p itself):

```
mean    = (1/K) * sum_i q_i                                    (centroid, R^3)
Cov     = (1/K) * sum_i (q_i - mean)(q_i - mean)^T              (covariance, symmetric 3x3, meters^2)
Cov     = V * Lambda * V^T                                      (eigendecomposition)
Lambda  = diag(lambda_0, lambda_1, lambda_2),  lambda_0 <= lambda_1 <= lambda_2 >= 0
```

**Why the smallest eigenvector is the plane normal (total-least-squares).** The covariance matrix's
quadratic form v^T Cov v = (1/K) * sum_i [(q_i - mean) . v]^2 measures the sum of squared projections of
the (centered) neighborhood onto direction v — i.e., the VARIANCE of the neighborhood along v. Fitting a
plane through `mean` with normal v so as to MINIMIZE the sum of squared PERPENDICULAR distances from
every q_i to that plane is exactly minimizing v^T Cov v subject to |v|=1 (perpendicular distance from
q_i to the plane through mean with normal v IS (q_i-mean).v). By the Rayleigh-quotient/spectral theorem,
this quadratic form is minimized by the eigenvector of the SMALLEST eigenvalue — hence `eigenvectors[0]`
(kernels.cuh's ascending-order convention) is the least-squares TOTAL-LEAST-SQUARES plane normal (total,
not ordinary, least squares: it minimizes PERPENDICULAR distance, unlike a height-field regression z =
f(x,y) which minimizes VERTICAL distance and is biased on any tilted patch — [`02.03`](../02.03-ground-segmentation/THEORY.md)
makes exactly this same point for its own plane fit).

**Sign ambiguity.** Cov*v = lambda*v holds identically for v and -v — an eigenvector is a LINE, not a
direction. `kernels.cuh` STEP 5 breaks the tie by flipping toward the sensor: if dot(v, sensor - p) < 0,
negate v. This is PCL's `flipNormalTowardsViewpoint` convention. **Why it degrades at grazing incidence**
(surfaces seen edge-on): if the TRUE normal is nearly PERPENDICULAR to the sensor viewing direction (a
surface seen almost edge-on — dot(true_normal, view_dir) approx 0), then the raw eigenvector's small
numerical perturbation (from finite K, noise, or just which of two near-tied eigenvalue directions the
solver happens to converge along in a near-degenerate case) can push the SIGN CHOICE either way with
almost equal likelihood — the heuristic is choosing between two options separated by a vanishingly small
margin. Measured honestly, `GATE orientation`/`[info] orientation_grazing` in `main.cu` split points by
`|cos(true_normal, direction-to-sensor)|` (the "grazing cosine", precomputed in the committed sample) and
report the two cohorts' success rates SEPARATELY for exactly this reason.

**Curvature: surface variation, not differential-geometry curvature.** This project's `curvature` output
is

```
c = lambda_0 / (lambda_0 + lambda_1 + lambda_2),   c in [0, 1/3]
```

(Pauly, Gross & Kobbelt 2002). This is **NOT** the mean curvature H = (kappa_1+kappa_2)/2 nor the
Gaussian curvature K_g = kappa_1*kappa_2 of differential geometry (kappa_1, kappa_2 = the two principal
curvatures — the reciprocals of the two extreme osculating-circle radii at a point). Surface variation c
is a FLATNESS PROXY: it is exactly 0 for any perfectly flat neighborhood (lambda_0=0), grows with how
much the neighborhood deviates from flat, and — this is the honest distinction — grows for TWO PHYSICALLY
DIFFERENT reasons that this single number cannot disentangle: (a) genuine smooth SURFACE BENDING (a
sphere/cylinder, where every neighbor is a little off the local tangent plane because the surface
curves), and (b) a DISCONTINUITY the neighborhood straddles (an edge/corner, where the "bending" is
infinite in a differential sense — two locally-flat faces meeting at an angle). `GATE curvature_ordering`
measures the CORRELATION with true curvature honestly on the smooth surfaces (plane=0 < cylinder=1/r <
sphere: at matched radius, a sphere's surface variation is LARGER than a cylinder's because the sphere
bends in TWO independent directions instead of one — a purely geometric fact, not a noise or estimation
artifact) while the edge cohort demonstrates reason (b) is a SEPARATE, larger effect (`GATE
degeneracy_flags`).

## The algorithm

**Step-by-step**, per point (see `kernels.cuh`'s file header for the same list with full GPU-mapping
commentary):

1. **Voxel-hash index build** (once, shared by every point): pack each point's cell coordinate
   `floor(p/cell)` into a 64-bit key (`pack_voxel_key`, `02.01`/`02.05`'s biased 21-bit-per-axis scheme,
   reused); `thrust::stable_sort_by_key` the points by key; mark cell boundaries; compact into
   `(unique_key[], seg_start[])` — a sorted array + binary search index, O(N log N) build, O(1) expected
   per-cell lookup thereafter.
2. **K-nearest-neighbor search**, per point: scan the 3x3x3 cell stencil (ring 1) around the point's own
   cell; maintain a bounded max-heap of the best K candidates seen. **The correctness-critical subtlety**
   (found and fixed during this project's own development via `GATE brute_force_anchor`, which initially
   FAILED with a naive "stop once K candidates are found" rule): having found K candidates within a
   scanned region does NOT prove they are the true K NEAREST, because the query point can sit anywhere
   inside its own cell — a point in an unscanned cell just beyond the current ring can still be closer
   than the worst kept candidate. The provable stopping rule: after scanning every cell within Chebyshev
   distance `ring` of the query's cell, EVERY unscanned cell's nearest possible point is at Euclidean
   distance >= `ring * cell_size` from the query (proof: an unscanned cell has offset >= ring+1 along at
   least one axis; the near face of that cell along that axis is `ring * cell_size` away from the FAR
   edge of the query's own cell in the worst case, i.e. query sitting at the near edge of its cell). The
   search is safe to stop only once the heap is full AND its worst (largest) kept distance is already
   <= `ring * cell_size`; otherwise it widens to the next ring (a SHELL, not a re-scan — `kMaxRing=4`,
   a measured-then-margined cap). This is a small, self-contained, provable piece of computational
   geometry, and getting it wrong is a genuinely SILENT bug: an approximate-but-plausible-looking
   neighbor set still produces a plausible-looking (but subtly wrong) normal — exactly why the
   independent, hash-free `GATE brute_force_anchor` exists (see "How we verify correctness").
3. **Mean-shifted covariance** (two passes over the <=K cached neighbor positions): centroid, then
   covariance around it — see "Numerical considerations" for why not the naive one-pass formula.
4. **Eigendecomposition** via cyclic Jacobi (below).
5. **Normal, curvature, degeneracy** — direct formulas, "The math" above.

**Complexity.** Serial (one CPU core, brute-force neighbor search): O(N^2) — every point scans every
other point. Serial with a spatial index: O(N log N) build + O(N * k_avg) query, k_avg = average
candidates scanned per point (a small constant for near-uniform density, this project's
`estimate_normals_cpu` and the GPU's hash-based search both achieve this). Parallel (GPU, N independent
threads after the shared index exists): O(log N) for the index build's sort (across N/P threads with P
processors), O(1) amortized wall-clock per point for the fused pipeline (embarrassingly parallel, no
inter-thread dependency) — the whole reason a million-plus points/sec is achievable at all.

## The GPU mapping

**Thread-to-data mapping.** One thread per QUERY point, `q = blockIdx.x*blockDim.x + threadIdx.x`
(`estimate_normals_kernel`, `kernels.cu`). Every stage — neighbor search, covariance, eigensolve, normal,
curvature, degeneracy — happens inside that ONE thread, for that ONE point, start to finish. Grid-stride
is not needed at this project's scale (N fits in one grid launch comfortably even at 1M+ points: `ceil(N
/ 256)` blocks).

**Why ONE FUSED kernel instead of five separate ones.** Splitting the pipeline into
`knn_kernel -> covariance_kernel -> eigen_kernel -> normal_kernel` would require writing each stage's
output to GLOBAL MEMORY and reading it back in the next kernel — at N=1,050,000 points and K=16
neighbors, just the neighbor-id array alone would cost 1,050,000*16*4 bytes = 67.2 MiB written AND read
back, for information nothing outside this one thread ever needs again. Fusing into one kernel keeps
every intermediate value (the K-heap, the cached neighbor xyz, the covariance accumulator, the Jacobi
working matrices) in REGISTERS/LOCAL memory, touched only by the owning thread, and the only GLOBAL
memory traffic is: read this point's own xyz + its K neighbors' xyz (unavoidable), write 3+3+1+1+1 =
9 floats/ints of final output per point. This is the single biggest throughput lever in this project.

**Memory hierarchy.**
- **Global memory**: the point cloud `xyz[n*3]` (read, scattered access pattern through neighbor
  lookups — not perfectly coalesced, since neighbor indices are not sequential; the voxel-hash sort DOES
  improve locality somewhat, since spatially-close points end up index-close after the sort, but this
  project does not go further and re-permute the point array itself into sorted order, unlike some
  production implementations — a documented simplification, see "Where this sits in the real world");
  the voxel-hash index (`unique_key[]`, `seg_start[]`, `idx_sorted[]`, read-only, reused across all N
  threads — every thread performs its own independent binary searches into the SAME small `unique_key[]`
  array, which fits comfortably in L2 cache for this project's scale (num_voxels is O(N/points_per_cell),
  a small fraction of N) and is a natural broadcast-read pattern); the output arrays (write-once per
  point).
- **Registers / local memory (the honest occupancy story)**: this kernel's LIVE state at its peak
  includes a size-16 max-heap (16 floats + 16 int32 = 128 bytes), the CACHED neighbor xyz used for the
  two-pass covariance (16*3 floats = 192 bytes), the Jacobi working matrices A and V (9+9 floats = 72
  bytes), plus loop/index scratch — well over 100 live 32-bit values per thread. `nvcc
  --ptxas-options=-v` on this project's Release build reports this kernel's register usage and any local-
  memory spill directly; a kernel this register-heavy trades OCCUPANCY (fewer resident warps per SM,
  since each thread claims more of the fixed per-SM register file) for AVOIDING repeated global-memory
  round trips per stage — the right trade at this project's memory-access pattern (scattered neighbor
  reads dominate; more resident warps would not hide THAT latency as effectively as it hides a purely
  compute-bound kernel's latency, because the scattered reads themselves are the bottleneck, not compute
  throughput). Measure, don't assume: re-run `nvcc --ptxas-options=-v` after any change to `kK` and
  compare against this project's own measured baseline (Exercise 4).
- **Shared memory**: NOT used. Nothing in this kernel is shared BETWEEN threads (each thread's K-heap,
  covariance, and eigensolve are entirely private) — shared memory only pays when threads within a block
  cooperate on the SAME data, which does not happen here (unlike, say, a tiled matrix multiply). This is
  a deliberate, not accidental, absence — worth noticing precisely because most of this repo's other
  kernels DO use shared memory, and recognizing when NOT to is as much the lesson as when to.

**What Thrust computes here (no black boxes).** `thrust::stable_sort_by_key` performs a GPU **radix
sort**: repeated STABLE partitioning of the 64-bit voxel keys by a few bits at a time, least-significant-
bits first, carrying the paired index array along — O(N) passes over O(N) elements per pass for a
fixed key width, i.e. O(N) total for a bounded key width (64 bits here), not O(N log N) like a
comparison sort. STABLE matters because many points legitimately share one voxel key; a stable sort keeps
their relative order deterministic run to run. `thrust::reduce`/`thrust::copy_if` implement a standard
parallel reduction and stream-compaction (both O(log N) depth, O(N) work) — hand-rolling either would
teach the same lesson [`02.01`](../02.01-voxel-grid-downsampling-with-gpu-spatial-hashing/kernels.cu)
already does for this exact voxel-hash pipeline; reusing Thrust here keeps this project's own code
focused on the genuinely NEW material (the fused pipeline kernel).

## Numerical considerations

**The mean-shift trick (why NOT the one-pass covariance formula).** The textbook "one-pass" covariance
formula, Cov = E[pp^T] - mean*mean^T, computes each term as a running SUM OF SQUARES of the RAW
coordinates, then subtracts. At real LiDAR ranges (tens of meters from the sensor), a point's raw
coordinate magnitude is O(10) meters while its LOCAL neighborhood variance is O(0.01) meters^2 or
smaller — E[pp^T] and mean*mean^T are then two numbers of magnitude ~100 whose DIFFERENCE is ~0.01: a
relative cancellation of four orders of magnitude, right at the edge of float32's ~7 decimal digits of
precision, and in bad cases past it (catastrophic cancellation: the true small difference is dominated by
rounding error in the two large terms being subtracted). The two-pass formula this project uses —
compute the centroid FIRST, then accumulate Cov = (1/K) sum (p-mean)(p-mean)^T around it — never forms
those large intermediate magnitudes at all: every term being summed is already O(local variance), so the
sum's precision is limited only by the actual spread of the data, not by the distance from the origin.
This project's own committed sample places every cohort 14-24 meters from the sensor specifically so this
effect is real and measurable, not a strawman (Exercise: disable the mean-shift and re-run `VERIFY(eigen)`
— the tolerance would need to loosen measurably, especially for the noise=none plane cohort where the
TRUE answer is exactly determinable).

**Eigensolver conditioning.** Cyclic Jacobi (kJacobiSweeps=8 fixed sweeps, `02.03`'s measured-sufficient
count for float32 3x3) converges rapidly for well-separated eigenvalues; near a DEGENERATE covariance
(two eigenvalues nearly equal — e.g., an isotropic/sphere-like neighborhood, or a corner where the
neighborhood is genuinely 3-D-spread with no dominant flat direction), the EIGENVALUES remain numerically
well-determined (Jacobi's convergence guarantee does not depend on eigenvalue separation) but the
individual EIGENVECTORS become ill-conditioned — a tiny perturbation of the input can rotate the
eigenvector pair spanning the near-degenerate subspace almost arbitrarily within that subspace. This
project's curvature computation (needs only eigenVALUES) is therefore more robust near degeneracy than
the normal (needs the specific eigenVECTOR direction) — precisely the geometric situation `kDegenClean`/
`kDegenEdgeCorner`/`kDegenIsolated` are meant to FLAG rather than silently trust.

**Determinism.** Every stage here is per-thread-independent with NO atomics and NO cross-thread
reduction — the entire pipeline is exactly reproducible bit-for-bit given the same input and the same
compiled kernel (unlike, say, a reduction using floating-point atomics, whose summation ORDER — and
therefore its exact rounding — depends on which thread happens to arrive first). Two consecutive runs of
this project's demo on the same GPU produce byte-identical `VERIFY`/`GATE` PASS lines and byte-identical
measured `[info]` numbers (confirmed during development); only wall-clock `[time]` lines and
architecture-sensitive ULP-scale diffs (`[info] eigen_diff` etc., deliberately kept OFF the diffed stable-
line set — see `demo/README.md`) vary run to run or machine to machine.

**Angle wrapping / quaternion drift**: not applicable — this project has no rotation state carried across
iterations and produces plain unit-vector normals, not quaternions or Euler angles.

## How we verify correctness

**Three independently-typed neighbor-search implementations**, the project's central verification
strategy (`reference_cpu.cpp`'s file header states the full ruling):

1. **GPU**: `estimate_normals_kernel` — sorted array + binary search + a streaming bounded max-heap.
2. **CPU twin** (`estimate_normals_cpu`): an `std::unordered_map` voxel index (a DIFFERENT data
   structure) + batch-collect-then-`std::partial_sort` (a DIFFERENT algorithm) — `VERIFY(knn)` compares
   the two exactly, `VERIFY(eigen)`/`VERIFY(normals)`/`VERIFY(curvature)`/`VERIFY(degeneracy)` compare
   the full downstream pipeline, all against a documented tolerance (ULP-scale float diffs deliberately
   reported on `[info]` lines rather than embedded in the diffed verdict — see `demo/README.md`).
3. **Brute-force anchor** (`estimate_normal_brute_force`): NO spatial index at all — an O(n) linear scan
   over every point, for a documented stride subset (`kAnchorStride=20`, 420 of 8,400 points). This tier
   exists specifically to catch bugs common to BOTH the GPU kernel and its CPU twin, e.g. a shared
   misunderstanding of the correct stopping rule — which is EXACTLY what caught this project's own real
   bug during development: an early version's ring search stopped as soon as it found K candidates,
   without the safe-radius check ("The algorithm" above); `VERIFY(knn)` still passed (GPU and CPU agreed
   with EACH OTHER, since both shared the same incomplete stopping rule), but `GATE brute_force_anchor`
   FAILED (20 of 420 anchor points had a demonstrably wrong neighbor set) — the anchor tier is what
   distinguishes "GPU and CPU agree" from "GPU and CPU are both right".

**The eigensolver**, independently typed twice (kernels.cu's `d_jacobi_eigen_3x3` uses the numerically
preferred stable-tan-half-angle rotation formula; `reference_cpu.cpp`'s `jacobi_eigen_3x3_cpu` uses the
textbook direct `theta = 0.5*atan2(2*apq, aqq-app)` formula) — two different, both textbook-correct
routes to the same rotation, so a bug unique to either formula shows up as a genuine `VERIFY(eigen)`
disagreement instead of hiding behind one shared function.

**Independent gates against CLOSED-FORM analytic truth** (the tier neither twin nor anchor above can
provide, since all three ultimately implement the SAME algorithm — total-least-squares PCA fitting —
and would agree even if that algorithm itself had a conceptual flaw): `GATE plane_normals` (curvature is
EXACTLY 0 on the plane, so noise=none is a genuine near-exact anchor: measured mean 0.0024 deg, well
under the tol<=0.05 deg gate); `GATE sphere_normals` (a CURVED surface has real curvature-fit bias even
at noise=none — measured mean 1.44 deg — so this gate uses ONE measured-then-margined bound across all
three noise cohorts, not a near-exact anchor, and documents WHY in its own printed text); `GATE
cylinder_axis` (a FREE aggregate check: fit the cylinder's axis from the ESTIMATED normals' scatter-
matrix smallest eigenvector alone, compare against the true axis stored in the committed sample —
measured 0.07 deg apart, tol 2.0 deg); `GATE curvature_ordering` (plane < cylinder < sphere < edge
medians, the geometric ordering "The math" derives); `GATE degeneracy_flags` (edge cohort flagged >=40%,
measured 46.9%; plane-interior flagged <=5%, measured 0%).

**Tolerances, stated and why**: `kEigenTol=5e-4 m^2`, `kCurvatureTol=5e-4` — both loose enough to absorb
independently-typed float32 rounding differences (measured max diffs ~1e-7 to 1e-8, two to three orders
of margin under tolerance) but tight enough to catch a real algorithmic disagreement.
`kNormalAngleTolDeg=0.5` for the full-pipeline twin comparison (measured max 0.046 deg). The ANALYTIC
gates use two distinct tolerance philosophies deliberately: `kExactAnchorMeanTolDeg=0.05`/
`kExactAnchorMaxTolDeg=0.5` for the plane (curvature exactly 0, so any measurable error is purely
numerical, not geometric bias); `kCurvedMeanTolDeg=4.0`/`kCurvedMaxTolDeg=15.0` for the sphere/cylinder
(curvature-fit bias is a REAL, expected, non-zero effect at finite K — the gate's job is to catch a
REGRESSION, not to demand zero bias that should not exist).

## Where this sits in the real world

**PCL's `NormalEstimation`** implements the identical algorithm (covariance PCA + viewpoint
orientation) as this project's core, over a `pcl::search::KdTree` or `OrganizedNeighbor` (an
integral-image-based O(1) neighbor lookup specific to STRUCTURED, depth-image-shaped point clouds — a
fundamentally different, faster index than this project's voxel hash, available only when the cloud
retains its sensor's row/column structure, e.g. straight off an RGB-D camera or a rasterized LiDAR
sweep). **Open3D's `estimate_normals`** additionally implements normal-CONSISTENCY propagation (a graph
traversal that flips normals to agree with their spatial neighbors, not just a fixed viewpoint) — this
project's Exercise #2 names the classic algorithm (Hoppe et al. 1992) behind it. **nvblox** and
production LiDAR perception stacks run a materially similar fused-kernel normal estimator as one stage of
a much longer perception pipeline (often computing normals directly from a RANGE IMAGE via a stencil, when
the data is still organized — dramatically cheaper than a general KNN search, since "neighbor" is just
"adjacent pixel"; `02.12`, range-image conversion, is this repo's project for that representation).
**Learned normal estimation** (PCPNet, DeepFit — README "Prior art") replaces the fixed-K PCA fit with a
neural network trained to be robust to noise and to sharp features that a fixed-K linear fit cannot
distinguish from noise — the open research question this project's own curvature-vs-degeneracy ambiguity
("The math") motivates directly: a learned model can, in principle, learn to tell "smoothly curved" apart
from "sharp discontinuity" from the RAW neighborhood shape, where a single scalar (surface variation)
cannot.

**What a full production pipeline adds that this project does not**: normal consistency propagation
(above); an ORGANIZED/range-image fast path when the data supports it (`02.12`); density-adaptive K or
radius, to handle the real 1/r^2 point-density falloff `02.05`'s THEORY.md derives (this project's
committed sample is deliberately near-uniform per cohort to isolate other error sources cleanly); and,
for the anisotropic-ring bias this project's own "The problem" section names but does not correct for, a
ring-aware neighbor weighting or an explicit deskew pass (`02.08`) before normal estimation.
