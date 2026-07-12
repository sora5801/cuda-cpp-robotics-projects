# 02.12 — Range-image conversion + depth-clustering segmentation: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2).

## The problem — physics & engineering first

### The sensor sphere: why a range image is a LiDAR's *native* geometry

A spinning mechanical LiDAR (the kind this project models, and the kind still common on ground robots
and many AVs) has a fixed number of laser/detector pairs — **beams** — mounted at fixed **elevation**
angles on a rotating head. As the head spins through **azimuth**, each beam fires at a regular angular
step and measures the round-trip time of flight to whatever surface it hits, converting that into a
**range** (a radial distance). The sensor therefore does not "see" Cartesian space directly — it
samples a **sphere of directions** centered on itself, at a fixed angular grid: 16 elevation rows (this
project's beam table, -15° to +15° in 2° steps, identical to 02.01/02.02's) by however many azimuth
columns one revolution divides into (1024 here, 0.3516° per column). Every sample is a **spherical
projection**: a direction $(\theta, \phi)$ (elevation, azimuth) plus a range $r$. This is EXACTLY the
geometry a camera's rolling shutter produces for a 2-D image sensor (row, column, intensity) — the
range image is the LiDAR's direct analogue, not a derived representation. Reconstructing Cartesian
$(x,y,z)$ from $(\theta,\phi,r)$ is a cheap, exact, closed-form conversion:

$$x = r\cos\theta\cos\phi, \quad y = r\cos\theta\sin\phi, \quad z = r\sin\theta$$

(the same formula 01.18 and 02.01 derive and cite; this project reuses it verbatim — see
`scripts/make_synthetic.py`).

### The engineering reality: why real drivers hand you exactly this

A real LiDAR driver (Velodyne/Ouster/Hesai-style) transmits UDP packets containing, per beam-firing:
the beam's known ring index, a rotation-angle timestamp (from which azimuth is derived), and one or more
range/intensity returns. **The driver already knows the ring and can compute the azimuth bin** — it is
handing you range-image-shaped data whether you organize it or not (`PRACTICE.md` section 1 has the
real packet-format story). Treating the stream as an "unorganized point cloud" (as most point-cloud
libraries default to) THROWS AWAY the adjacency structure the sensor gave you for free, and every
consumer that needs neighbors (clustering, ground removal, normal estimation) then has to rebuild it
with a spatial hash or a tree (02.01, 02.04, 02.05, 02.09 — the domain's other neighbor-search
projects). This project's thesis: don't throw it away.

### Engineering constraints this project's scene design must respect

- **Angular resolution is coarse.** At 1024 azimuth bins and 16 beams, a real 5 m object subtends only
  a modest number of image cells even at moderate range — thin/small/far objects (this project's
  `thin_pole`, `far_pole`) are genuinely marginal, not an artifact of a lazy scene.
- **Range noise is real and range-axis-dominated** for time-of-flight sensors (as opposed to angular
  noise, which is comparatively tiny, set by encoder precision). This project's synthetic noise
  (`RANGE_NOISE_SIGMA_M = 3` mm) is applied purely along the ray direction, matching this physical
  reality.
- **Occlusion is unforgiving and directional.** A point behind an object, along the SAME ray, is simply
  never measured — it is not "hidden," it does not exist in the data at all. This project's own scene
  design had to be corrected once this was measured directly (see "Numerical considerations" below):
  a naively-placed object behind another does not produce the naive face-to-face gap you would compute
  from their bounding boxes, because the occlusion shadow widens with range.

## The math

### Notation

- $r$ — range, meters, always $\ge 0$; $r=0$ is this project's sentinel for "no return."
- Ring index $i \in \{0,\dots,15\}$ (elevation $\theta_i$, from the fixed table); azimuth-bin index
  $j \in \{0,\dots,1023\}$ (azimuth $\phi_j = j \cdot \Delta\phi$, $\Delta\phi = 2\pi/1024$).
- Cell index (02.02's convention, cited): $c(i,j) = i \cdot 1024 + j$ — ring-major flat index.
- $\alpha$ — the **angular step** between two adjacent beams being compared: either
  $\Delta\phi = 2\pi/1024$ (an azimuth-adjacent pair, same ring) or $\theta_{i+1}-\theta_i$ (a
  ring-adjacent pair, same column; a constant $2°$ on this project's uniform beam table, but computed
  from the table rather than hardcoded, so the derivation generalizes to a non-uniform table).

### Range-image conversion (the data-layout contract)

Unorganized → organized is a **partial function** $f: \{1,\dots,N\} \to \{0,\dots,16383\}$ mapping
point index to cell index via $f(k) = c(\mathrm{ring}_k, \mathrm{az\_bin}_k)$; where multiple points
share a cell (never happens on a clean single-revolution scan, but must be handled — see "How we verify
correctness"), the WINNER is $\arg\min_k r_k$ over that cell's contenders — the physically correct rule
(nearest surface occludes farther ones). Organized → unorganized is simply this map's **image**: the
set of cells with a winner, read back out as a flat list.

### Ground removal: the column-wise vertical-angle criterion

For a column $j$, walk rings $i=0,1,\dots,15$ in order (bottom, i.e. most negative elevation, to top).
Maintain a "current reference point" $(\rho_{\text{prev}}, z_{\text{prev}})$ in the column's own
$(\rho, z)$ half-plane, $\rho = \sqrt{x^2+y^2}$ the horizontal radius. Initialize it to the **virtual
sensor-mount point** $(\rho_{\text{prev}}, z_{\text{prev}}) = (0, -h)$, $h$ = the known sensor mounting
height above the ground — a standard assumption in practical ground-removal systems (Zermas et al. 2017;
LeGO-LOAM's ground-plane removal uses the identical prior). For each valid return at $(\rho_i, z_i)$:

$$
\beta_{\text{ground}} = \operatorname{atan2}(z_i - z_{\text{prev}},\ \rho_i - \rho_{\text{prev}})
$$

Label the return **ground** iff $|\beta_{\text{ground}}| \le \theta_g$ (`kGroundAngleThresholdDeg`,
10°), then set $(\rho_{\text{prev}}, z_{\text{prev}}) \leftarrow (\rho_i, z_i)$ and continue — **every**
step, ground or not, updates the reference to the last VALID return (a local slope test, not "compare
only to the last ground point"). A flat ground plane produces $\beta_{\text{ground}} \approx 0$ at every
step (small only from range noise); a vertical obstacle face produces $\beta_{\text{ground}}$ near
$\pm 90°$. `atan2` handles $\rho_i - \rho_{\text{prev}} \le 0$ correctly (a near-radial jump, exactly
what a depth discontinuity produces at the base of an object) — the angle saturates toward $\pm90°$,
correctly reading "not ground" rather than dividing by zero or wrapping incorrectly.

### The beta criterion — full derivation

This is Bogoslavskyi & Stachniss's central contribution (IROS 2016), derived here from the
line-of-sight triangle:

```
                    B (farther return, range r1)
                   /|
                  / |
                 /  |
                /   | <- the segment AB: is it a continuous
               /    |    surface, or a depth "step"?
              /     |
             A------+
            /  \  (a right-angle construction line, NOT part
           /    \   of the actual geometry -- see below)
          /      \
         O--------- (sensor origin)
     r2 = |OA|, r1 = |OB|, angle AOB = alpha (the beams' angular step)
```

Let $O$ be the sensor origin, $A$ the NEARER of two adjacent returns (range $r_2$), $B$ the FARTHER
(range $r_1 \ge r_2$), and $\alpha$ the angular step between the two beams that produced $A$ and $B$.
We want $\beta$, the angle at vertex $A$ between the segment $AB$ and the ray $AO$ extended backward
(equivalently: the angle the local surface tangent at $A$, along $AB$, makes with $A$'s own line of
sight) — this angle characterizes whether $A$ and $B$ lie on one continuous surface (as seen by the
sensor) or across a depth step.

Drop a perpendicular from $B$ onto the line $OA$ extended, meeting it at point $C$. In the right
triangle $OCB$: $\angle BOC = \alpha$, so

$$
|OC| = r_1 \cos\alpha, \qquad |CB| = r_1 \sin\alpha
$$

Point $A$ lies on segment $OC$ at distance $r_2$ from $O$, so

$$
|AC| = |OC| - |OA| = r_1\cos\alpha - r_2
$$

Now triangle $ACB$ is right-angled at $C$, with legs $|AC| = r_1\cos\alpha - r_2$ (along the line
through $O$ and $A$) and $|CB| = r_1\sin\alpha$ (perpendicular to it). The angle at $A$ between $AB$
and the direction FROM $A$ BACK TOWARD $O$ (i.e. the $-|AC|$ direction) is

$$
\beta = \operatorname{atan2}\bigl(r_1\sin\alpha,\ r_2 - r_1\cos\alpha\bigr)
$$

This project's code (and Bogoslavskyi & Stachniss's own convention) instead writes it with $r_1$
(farther) and $r_2$ (nearer) swapped in the roles above, giving the algebraically equivalent

$$
\boxed{\beta = \operatorname{atan2}\bigl(r_2\sin\alpha,\ r_1 - r_2\cos\alpha\bigr)}
$$

(`beta_criterion_rad` in `kernels.cuh` — the two forms differ only in which point's local frame $\beta$
is measured in; both are used in the literature, and both give the same qualitative reading, since the
triangle is symmetric in the relevant sense for small $\alpha$). Reading the formula:

- **$r_1 \approx r_2$ (a continuous, roughly sensor-facing surface):** the denominator
  $r_1 - r_2\cos\alpha \approx r(1-\cos\alpha)$ is small and positive, the numerator
  $r_2\sin\alpha \approx r\sin\alpha$ is also small — but their RATIO is
  $\sin\alpha/(1-\cos\alpha) = \cot(\alpha/2)$ (a half-angle identity), which **diverges as
  $\alpha\to0$** — i.e. $\beta \to 90°$ regardless of range, for any continuous surface, as the angular
  step shrinks. This project's own `beta_angle_map.csv` artifact and `depth_edges_cpu`/kernel both
  confirm this empirically on flat surfaces (measured beta values near 90° across face-on panels — see
  `demo/out/gates_metrics.csv` after a run for the person's OWN front-face beta values, all comfortably
  above threshold).
- **$r_1 \gg r_2$ or $r_1 \ll r_2$ (a depth step):** the segment $AB$ points nearly ALONG the line of
  sight; $\beta \to 0$. The formula is **scale-invariant**: multiplying both $r_1,r_2$ by the same
  constant $k$ leaves $\beta$ unchanged (both numerator and denominator scale by $k$), which is exactly
  why this criterion works at ANY range — a depth-gap pair at 2 m and the "same shape" pair scaled to
  20 m produce the identical $\beta$. A FIXED metric-distance threshold (what Euclidean clustering uses)
  has no such invariance: the same physical gap reads differently at different ranges purely because
  adjacent-beam arc spacing scales with range (`README.md`'s depth-gap showcase measures exactly this
  contrast).

Grazing/shallow-incidence behavior (the known weakness, this project's `grazing_wall` cohort): for a
flat surface running roughly ALONG the sensor's line of sight rather than facing it, adjacent-column
range values change rapidly even though the surface is perfectly continuous — differentiating the
range-vs-azimuth relation $r(\phi) = y/\sin\phi$ for a wall at fixed lateral offset $y$:

$$
\frac{dr}{d\phi} = -\frac{y\cos\phi}{\sin^2\phi}
$$

which **diverges as $\phi \to 0$** (viewed edge-on). For a small but nonzero angular step $\alpha$, the
range jump between adjacent columns is approximately $|dr/d\phi| \cdot \alpha$; once this exceeds what
$\cot(\alpha/2)$'s "continuous surface" reading can absorb, $\beta$ drops below threshold and the
criterion (correctly, given only two neighbor samples' information) reads it as a depth step — even
though geometrically it is one continuous surface, just steeply foreshortened. This is not a bug in the
formula; it is what a LOCAL, two-sample test can and cannot see, and it is exactly what this project's
`GATE grazing_fragmentation` measures happening (13 fragments on the committed scene).

## The algorithm

**Stage 1a — unorganized → organized** (per point, GPU-parallel): compute cell index, encode
`(range, point_index)`, `atomicMin` race into the cell array; O(N) work, O(1) per point.
**Stage 1b — organized → unorganized**: per cell, if marked obstacle, atomic-append to a flat list;
O(16384) work (this project's fixed grid size).
**Stage 2 — ground removal**: per column (1024 of them, GPU-parallel), a serial 16-step walk; O(16384)
total work, but only $O(16)$ SERIAL depth per thread — the whole stage completes in the time one column
takes, not sixteen.
**Stage 3 — depth-clustering edges**: per cell, test 2 fixed neighbors; O(2·16384) = O(32768) work,
zero search.
**Stage 4 — union-find**: per edge per sweep, path-halving find + union-by-min; O(E) work per sweep,
converging in $O(\log D)$ sweeps ($D$ = graph diameter) by the same argument 02.04's kernels.cuh derives
in full (cited).
**Stage 5 — Euclidean comparison**: voxel-key, sort ($O(M\log M)$, $M$ = obstacle point count),
27-cell-stencil neighbor search per point ($O(M \cdot \bar{k})$, $\bar{k}$ = average local density), then
the SAME union-find over the resulting edges.

**The complexity contrast that is this project's whole point:** stage 3's neighbor-finding is
$O(1)$ per cell (two fixed lookups) versus stage 5's $O(\log M + \bar{k})$ per point (a sort plus a
stencil search). On this project's committed scene ($M=3{,}697$ points), the measured Euclidean pipeline
takes several times longer than the depth-image pipeline end to end (`GATE timing_payoff`'s `[time]`
lines) — and the GAP GROWS with point count, since stage 3's cost is fixed by grid SHAPE while stage 5's
grows with point DENSITY.

## The GPU mapping

- **Range-image conversion**: a classic scatter/gather pair. `scatter_encode_kernel` is a pure MAP over
  points (one thread per point, no shared memory — no data reuse between threads); `atomicMin` on a
  64-bit key is the one synchronization primitive needed, and it requires compute capability >= 5.0
  (Maxwell) for 64-bit atomics — this repo's sm_75 floor clears it with room to spare (02.02's note,
  cited). `finalize_organized_kernel` is a pure MAP over cells (one thread per cell).
- **Ground removal**: one thread per COLUMN, each running a short (16-iteration) SEQUENTIAL loop. This
  is deliberately NOT decomposed further with a parallel scan (contrast 02.02's Blelloch-scan chapter,
  which pays off at array lengths in the thousands-to-millions): at 16 elements, a scan's setup
  overhead (multiple kernel launches, shared-memory staging) would dwarf the ~16 sequential adds/atan2
  calls it replaces. The crossover point between "just loop" and "decompose with a parallel primitive"
  is itself a lesson — see 02.02's own scan chapter for where that crossover actually pays off.
- **Depth-clustering edges**: a fixed 2-neighbor STENCIL, one thread per cell — global-memory reads
  only (`range_img`, `obstacle_mask`), no shared memory (each cell's 2 neighbor reads are NOT reused by
  any other thread, so there is nothing to cache locally; contrast a convolution stencil with overlapping
  neighborhoods, where shared-memory tiling pays off). The wrap-around (`col+1` mod `kAzimuthBins`) is a
  single conditional — cheap, and essential for correctness at the seam (this project's own scene
  straddles it, by construction, so the wrap is genuinely exercised, not merely present).
- **Union-find**: the SAME generic kernel set clusters both graphs; occupancy is edge-count-bound
  (one thread per edge per sweep), and `d_uf_find_halve`'s single non-atomic store per hop is safe under
  concurrent access by the "monotone-parent" argument 02.04's kernels.cuh proves in full (cited
  verbatim — union-by-min never decreases a node's eventual root value, so a racing halve can only ever
  redirect a pointer to a value that is still a valid ancestor).
- **Euclidean comparison**: `thrust::stable_sort_by_key` (radix sort on 64-bit voxel keys — what it
  computes: an ascending permutation of the point array by voxel; why Thrust: a hand-rolled radix sort is
  a full project of its own, 02.01/33.01's territory, cited rather than re-derived here) +
  `thrust::copy_if`/`reduce` for boundary compaction (02.01 Method B's idiom, cited) + a per-point
  27-cell stencil with a device binary search (`d_lower_bound`) over the sorted, deduplicated voxel-key
  array — the SAME technique 02.04's `build_edges_kernel` uses, cited and reused near-verbatim.

## Numerical considerations

- **Precision**: FP32 throughout (range, angle, all geometry) — matches the sensor's own native
  precision (real LiDAR ranges are FP32-class, ~mm-level, well within FP32's ~7 decimal digits at this
  project's scene scale of a few to ~15 meters).
- **`atan2` conditioning**: every angle test in this project (`ground_step_angle_deg`,
  `beta_criterion_rad`) uses `atan2`, never a bare `atan` of a ratio — `atan2` is well-conditioned at
  BOTH arguments near zero (returns a well-defined angle instead of a `0/0` NaN) and correctly handles
  the sign of both arguments (needed for the ground test's $\rho_i-\rho_{\text{prev}}\le0$ case and for
  a depth-step pair where the near/far assignment could in principle be either endpoint before the
  $\max/\min$ swap is applied).
- **Range-noise sensitivity of the ground test**: the ground test's angular noise contribution is
  approximately $\operatorname{atan}(\sigma_r / \Delta\rho)$, where $\Delta\rho$ is the horizontal
  spacing between consecutive valid returns in a column. On this project's scene, ground returns occur
  at ranges of several meters where ring spacing $\Delta\rho \gtrsim 0.3$ m, giving noise contribution
  $\operatorname{atan}(0.003/0.3) \approx 0.57°$ — small relative to the 10° threshold, chosen
  deliberately in `scripts/make_synthetic.py`'s `RANGE_NOISE_SIGMA_M = 3` mm (documented arithmetic in
  that script's module docstring) to keep `GATE ground_removal`'s precision/recall high without being an
  unrealistically noiseless sensor.
- **The occlusion-shadow correction (a real numerical/geometric lesson this project's OWN development
  surfaced)**: an early version of the scene placed the wall 0.30 m behind the person's face, expecting
  a 0.30 m visible gap — a brute-force nearest-neighbor search over the actually-generated points
  measured **0.50 m** instead, because the nearest theoretical wall point (directly behind the person)
  is OCCLUDED; the nearest VISIBLE wall point sits outside the person's angular shadow, which widens
  with range by a factor of (wall range / person range). The scene was re-derived (narrower person,
  smaller face gap) to make the MEASURED visible gap (0.19 m, verified) actually clear the design intent
  — see `scripts/make_synthetic.py`'s module docstring for the full arithmetic. The lesson: in any
  occlusion-aware system, "distance between two objects' bounding boxes" is not the same quantity as
  "distance between two objects' VISIBLE surfaces," and conflating them is an easy, silent bug.
- **A second, vertical version of the same lesson**: if `wall_behind` were taller than `person`, some
  beam rings would clear the person's top and hit the wall directly above it in the SAME azimuth
  column — a RING-adjacent pair, whose angular step (2°) is ~6x the azimuth step (0.35°). For the SAME
  physical range gap, a 6x larger $\alpha$ produces a MUCH larger $\beta$ (the $\cot(\alpha/2)$
  divergence above is steeper for larger $\alpha$ at fixed range ratio), which can push a genuine depth
  step ABOVE threshold in the ring direction even while the azimuth-direction boundary correctly cuts.
  Measured on an earlier version of this scene: ring-adjacent $\beta \approx 18°$ (connects, wrongly)
  vs. azimuth-adjacent $\beta \approx 3.4°$ (correctly cuts) for the SAME physical gap. Fixed by keeping
  the wall's top at or below the person's top (`scripts/make_synthetic.py`'s `wall_behind` comment
  derives why this is sufficient: a farther object subtends a smaller angle for the same physical
  height, so equal top heights already guarantee the wall's angular top sits below the person's).
- **Determinism**: every comparison in this project (`atomicMin` scatter, union-find's union-by-min) is
  ORDER-INDEPENDENT by construction (min and union-by-min are commutative/associative operations), so
  GPU results are bit-exact and reproducible across runs and across thread-scheduling orders — verified
  empirically (see "How we verify correctness").

## How we verify correctness

Every GPU stage has an independently-written CPU twin in `reference_cpu.cpp` (the independence ruling —
shared DATA-LAYOUT FORMULAS vs. genuinely reimplemented ALGORITHMIC CORES — is stated in full in that
file's header, mirroring 02.02's and 02.04's identical rulings):

- **`VERIFY(range_image)`**: exact equality expected and measured (0 mismatches) — the scatter is a pure
  selection among a fixed set of already-computed values via a commutative minimum, so no rounding or
  ordering ambiguity is possible.
- **`VERIFY(ground_removal)` / `VERIFY(depth_edges)` / `VERIFY(euclidean_edges)`**: exact equality
  expected (0 mismatches, measured); a small documented allowance (8 mismatches) exists ONLY to absorb
  the theoretical possibility of an FMA-rounding difference landing a measurement EXACTLY on a threshold
  boundary (the same honest allowance 02.03's RANSAC inlier-count comparison carries, cited) — this
  project's scene was deliberately designed with comfortable margins around every threshold (measured
  beta values: ~3-4° for genuine depth gaps vs. the 10° threshold; ~90° for continuous surfaces), so the
  allowance is never actually needed in practice.
- **`VERIFY(union_find_depth)` / `VERIFY(union_find_euclid)`**: BIT-EXACT equality required (0
  mismatches, no allowance) — union-by-min's final partition is mathematically order-independent (any
  correct sequence of unions over the same edge set converges to the same canonical roots), so a
  massively-parallel GPU sweep and a single-threaded sequential CPU union-find are expected to agree
  EXACTLY despite completely different execution orders. This is the strongest correctness statement
  this project's verification makes, and it is the same argument 02.04's kernels.cuh states in full for
  its own identical union-find (cited).
- **The four teaching gates** (`partition_vs_truth`, `depth_gap_showcase`, `grazing_fragmentation`,
  `ground_removal`) compare the (already-verified) GPU pipeline's output against the generator's
  ground-truth labels — a DIFFERENT kind of check from the VERIFY lines above (behavioral correctness
  against a designed scenario, not GPU-vs-CPU numerical agreement).

## Where this sits in the real world

**Bogoslavskyi & Stachniss (IROS 2016)** and its reference implementation, `depth_clustering`
(github.com/PRBonn/depth_clustering), are the direct lineage this project reimplements didactically —
study the real codebase for its handling of multi-echo returns, its smoothing pre-pass (a Gaussian blur
over the range image before computing beta, to reduce sensor-noise-induced fragmentation — a mitigation
this project's THEORY names but does not implement, scoped out for teaching-clarity), and its reported
runtimes on real KITTI-scale scans (tens of thousands of points, sub-10ms segmentation — the same
order-of-magnitude speed argument this project's own timing gate demonstrates at a smaller scale).

**Production AV/robotics LiDAR perception** (Autoware's `euclidean_cluster`/`scan_ground_filter` nodes,
Apollo's LiDAR perception stack) increasingly favor range-image-native processing for exactly the
latency reason this project's System-context section names — Autoware's ground filter, in particular, is
a column-wise scan-line method closely related to this project's ground-removal stage (though with
additional smoothing and a locally-adaptive threshold this project's flat-ground scene does not need).

**Learned segmentation over the SAME representation**: RangeNet++ (Milioto et al., IROS 2019) and its
successors replace the hand-crafted beta criterion with a CNN trained directly on the range image (a
2-D convolution over exactly the grid this project builds) — the range-image representation itself
outlives the hand-crafted algorithm; this project teaches the REPRESENTATION as much as the specific
segmenter built on top of it.

**Where this project's ground-removal simplification would need to grow**: 02.03's full RANSAC/CZM
treatment (cited throughout this project, honestly, at every point of contact) is the "what a real
system does differently" answer for ground removal specifically — multi-plane fitting, region growing
across terrain breaks, adaptive per-patch height margins. This project's compact column-walk is
appropriate for the flat-ground scene it needs to solve and is explicitly scoped as the SIMPLER,
range-image-native alternative, not a replacement for 02.03's fuller treatment.
