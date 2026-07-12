// ===========================================================================
// kernels.cuh — interface for project 02.09
//               Normal + curvature estimation at millions of points/sec
//               (per-point surface normals and a curvature proxy, fused into
//                ONE kernel per point: voxel-hash KNN -> mean-shifted
//                covariance -> Jacobi eigensolve -> sensor-oriented normal
//                -> surface-variation curvature -> a degeneracy flag)
//
// Role in the project
// -------------------
// The CONTRACT shared by main.cu (load data, drive the pipeline, run every
// VERIFY/GATE, write artifacts), kernels.cu (the GPU kernels), and
// reference_cpu.cpp (the independent CPU oracle twins). Every data-layout
// decision all three must agree on — the point-cloud layout, the voxel-hash
// key packing, the neighbor heap ordering, the covariance/eigen packing, the
// output arrays — is defined HERE, once (CLAUDE.md §12).
//
// POINT CLOUD LAYOUT — float* xyz, interleaved, meters, LiDAR "sensor" frame
// (origin at the sensor, +x forward, +y left, +z up), the SAME convention
// 02.01/02.05 use: xyz[i*3+0..2] = x,y,z.
//
// THE NEIGHBOR-ENGINE CHOICE (this project's first design decision, and the
// one the worker brief asked to be justified explicitly): project 02.05
// builds and contrasts TWO engines — a from-scratch LBVH (Karras radix
// tree) and a fixed-radius voxel hash — and shows the LBVH wins for
// adaptive-radius KNN queries over an IRREGULAR point cloud (its dense
// cluster / sparse region contrast). THIS project's synthetic surfaces are
// deliberately generated at a near-UNIFORM local point density per cohort
// (see scripts/make_synthetic.py) — exactly the regime where 02.05's own
// header names the voxel hash as "fast and correct": a FIXED cell size
// tuned to the data's own spacing reliably returns >= K candidates from a
// small stencil around every interior point, at a MUCH lower one-time build
// cost (one sort + one boundary scan, no tree topology, no per-query stack)
// than an LBVH. Because this project's whole teaching point is THROUGHPUT
// (points/sec, catalog: "at millions of points/sec"), not adaptive-radius
// robustness, the voxel hash is the right tool — reimplemented compactly
// below, citing 02.01 (the hash-key packing) and 02.05 (the sort+compact
// build and the bounded-heap KNN idea) rather than importing either
// (CLAUDE.md §4 self-containment rule). The honest cost, paid explicitly:
// a hard-coded cell size means a badly mismatched local density (a point
// cloud with wildly varying spacing) would need re-tuning or the expanding-
// ring fallback below to actually engage — see kMaxRing and the ISOLATED
// degeneracy flag, which detect (never silently hide) that failure mode.
//
// THE PIPELINE (the whole point of this project; every stage below is one
// conceptual step, fused into a SINGLE kernel per point — see kernels.cu's
// estimate_normals_kernel header for why fusing beats writing intermediate
// buffers at 1M+ point scale):
//
//   STEP 1 — VOXEL-HASH INDEX (built once, before the per-point kernel):
//   hash every point into a cell of side kCellSizeM (02.01's Method-B key
//   packing, reused compactly), sort by key (thrust::radix sort), compact
//   into (unique_key[], seg_start[]) — the same "sorted array + binary
//   search" index 02.01/02.04/02.05 all build, cited and reimplemented.
//
//   STEP 2 — K-NEAREST-NEIGHBOR SEARCH (per point, K = kK = 16, INCLUDING
//   the query point itself — the standard convention for local surface
//   fitting, e.g. PCL's default): scan the 3x3x3 cell stencil around the
//   point's own cell (ring 1); if fewer than K points were found, widen to
//   the ring-2 SHELL (5x5x5 minus the already-scanned 3x3x3 — a shell, not
//   a re-scan, so no point is ever counted twice); a bounded max-heap
//   (ordered by (dist2, index), 02.05's knn_less tie-break, reimplemented)
//   keeps the K best candidates seen so far. If ring kMaxRing still leaves
//   the heap short of K, the point is flagged ISOLATED (kDegenIsolated) and
//   proceeds with whatever it found — never silently padded with garbage.
//
//   STEP 3 — MEAN-SHIFTED COVARIANCE (the "mean-shift trick", THEORY.md
//   "Numerical considerations" derives it in full): compute the neighbor
//   centroid FIRST (one pass over the <=K cached neighbor positions, held
//   in per-thread local/register memory from step 2), THEN accumulate
//   Cov = (1/m) * sum (p_j - mean)(p_j - mean)^T in a SECOND pass. This
//   avoids the catastrophic cancellation of the textbook one-pass formula
//   Cov = E[pp^T] - mean*mean^T, which loses precision badly when points
//   sit far from the coordinate origin (as real LiDAR returns do, at tens
//   of meters range) but are tightly clustered locally (the whole point of
//   "local" covariance) — see 02.03's fit_plane_from_cov_accum for the
//   one-pass version this project deliberately does NOT copy, and THEORY.md
//   for a worked numeric example of the cancellation.
//
//   STEP 4 — EIGENDECOMPOSITION — a hand-rolled cyclic JACOBI solve on the
//   symmetric 3x3 covariance (02.03's jacobi_eigen_3x3 precedent, cited but
//   INDEPENDENTLY re-implemented per project below — see kernels.cu). Why
//   Jacobi and not 02.08's closed-form Smith (1961) trigonometric solve
//   (also cited): Smith's method is a fast path to the SMALLEST eigenvalue
//   alone (exactly what 02.08's wall-plane fit needs); THIS project needs
//   ALL THREE eigenvalues (curvature needs the full spectrum, degeneracy
//   needs the eigenvalue RATIOS) and the smallest eigenvalue's eigenVECTOR
//   (the normal) — Jacobi produces the whole (eigenvalues, eigenvectors)
//   pair in one pass, which is the right tool once more than the smallest
//   root is needed. Ascending order out: eigenvalues[0] <= [1] <= [2].
//
//   STEP 5 — NORMAL = eigenvectors[0] (the smallest-eigenvalue eigenvector
//   — the least-squares total-least-squares plane normal; THEORY.md "The
//   math" derives why). SIGN-DISAMBIGUATED toward the sensor: Jacobi (like
//   any eigensolver) returns a normal defined only up to sign (+-n both
//   satisfy Cov*n = lambda0*n) — flipped so dot(n, sensor - p) > 0, the
//   standard viewpoint-orientation convention (PCL's
//   `flipNormalTowardsViewpoint`, cited in README). THEORY.md derives WHY
//   this heuristic degrades at GRAZING incidence (surfaces seen edge-on,
//   where the dot product is near zero and a small estimation error can
//   flip its sign) — main.cu's GATE orientation measures this directly via
//   the committed sample's precomputed grazing-angle cosine.
//
//   STEP 6 — CURVATURE (surface variation) c = lambda0 / (lambda0+lambda1+
//   lambda2) — Pauly, Gross & Kobbelt 2002's "surface variation" measure,
//   cited in README. THEORY.md is explicit that this is NOT mean curvature
//   H nor Gaussian curvature K from differential geometry — it is a
//   flatness proxy in [0, 1/3] that happens to CORRELATE with true
//   curvature on smooth surfaces (main.cu's curvature_ordering gate
//   measures the correlation honestly, cohort by cohort) but spikes at
//   sharp edges/corners for a different reason (the neighborhood straddles
//   two surfaces, not because either surface bends fast there).
//
//   STEP 7 — DEGENERACY FLAG: kDegenClean (0) by default; kDegenEdgeCorner
//   (1) if c exceeds kCurvatureDegenThreshold (a neighborhood not well
//   described by one plane — could be a ridge, a corner, or noise blowing
//   up the flat direction); kDegenIsolated (2) if step 2 could not find K
//   neighbors even after the ring-2 shell (a genuinely sparse/boundary
//   point, where any covariance-based estimate is unreliable regardless of
//   its numeric value).
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cstdint>       // int32_t, uint32_t, uint64_t — exact-width integers everywhere below
#include <cmath>         // std::floor, std::sqrt, std::fabs — identical overloads to cl.exe and nvcc's host pass
#include <vector>        // reference_cpu.cpp's independent oracle outputs
#include <unordered_map> // reference_cpu.cpp's independent voxel-hash oracle (02.04/02.05 lineage)
#include <algorithm>     // std::sort/std::max/std::min (reference_cpu.cpp's independent twins)

// ===========================================================================
// Problem-scale constants — the numbers every stage and every CPU twin below
// must agree on bit-for-bit.
// ===========================================================================

// Repo-default block size (warp multiple; good occupancy on sm_75..sm_89 —
// see kernels.cu's per-kernel launch-configuration comments).
constexpr int kThreadsPerBlock = 256;

// kK — the neighborhood size for the local surface fit, INCLUDING the query
// point itself. 16 sits in PCL/Open3D's typical 10-30 range for normal
// estimation and is this project's worker-briefed value; THEORY.md's
// "Numerical considerations" derives the K-vs-noise tradeoff (larger K
// averages down noise but blurs sharp features and grows the register
// footprint of every per-thread heap/covariance accumulator below).
constexpr int kK = 16;

// kMaxRing — how far the neighbor search widens (in CELLS, Chebyshev
// distance) before giving up and flagging kDegenIsolated. Ring 1 = the
// 3x3x3 stencil (27 cells) around the query's own cell; ring r>1 adds the
// (2r+1)^3-minus-(2r-1)^3 SHELL (cells with max(|dx|,|dy|,|dz|)==r,
// deliberately excluding already-scanned inner cells so no point is ever
// counted twice — see kernels.cu's estimate_normals_kernel). The search
// does NOT stop merely because it has found kK candidates: the query can
// sit anywhere inside its own cell, so a point in an UNSCANNED cell just
// beyond the current ring can still be closer than the worst candidate
// found so far. It stops only once the SAFE-RADIUS rule holds -- heap full
// AND the worst kept distance <= ring*kCellSizeM, the provable distance to
// the nearest point of any unscanned cell (THEORY.md "The algorithm"
// derives this). kMaxRing=4 is a measured-then-margined cap: this
// project's own density (scripts/make_synthetic.py) satisfies the safe-
// radius rule within ring 1-2 for essentially every point; ring 3-4 exist
// as a documented safety margin for patch-boundary/corner points whose
// local neighborhood is sparser on one side (kDegenIsolated flags the rare
// point that still falls short even at ring kMaxRing).
constexpr int kMaxRing = 4;

// kCellSizeM — the voxel-hash cell size, meters. Sized against THIS
// project's own synthetic point spacing (~0.09 m over each analytic
// surface patch, scripts/make_synthetic.py's grid+jitter sampler): a
// 3-cell-wide stencil (kCellSizeM*3 per axis) needs to comfortably clear
// kK=16 points on an interior patch point. 0.30 m gives a 0.9 m stencil
// width against ~0.09 m spacing — roughly a (0.9/0.09)^2 ~= 100-point
// interior yield for a locally-flat surface (a 2-D manifold occupies a
// near-zero-thickness slab of the 3-D stencil cube), comfortably above 16
// with real margin for jitter and patch-boundary thinning. Retuning this
// constant for a different point density is the single knob to turn.
constexpr float kCellSizeM = 0.30f;

// kCurvatureDegenThreshold — the surface-variation threshold above which a
// point is flagged kDegenEdgeCorner. MEASURED against this project's own
// data (see THEORY.md "How we verify correctness" for the actual noise-0
// plane-interior vs. edge-cohort medians this value was chosen between) —
// not a universal constant, a documented, data-informed choice.
constexpr float kCurvatureDegenThreshold = 0.05f;

// kOrientationGrazingCos — the |cos(angle between true normal and the
// sensor viewing direction)| threshold below which a point counts as
// "grazing" (surface seen near edge-on) for GATE orientation's split
// between the exact-gated "confidently viewable" cohort and the [info]
// grazing cohort (THEORY.md derives why grazing incidence is the honest
// failure mode of the viewpoint sign-disambiguation heuristic).
constexpr float kOrientationGrazingCos = 0.20f;

// Degeneracy flag values (kernels.cuh-wide vocabulary; see STEP 7 above).
constexpr int32_t kDegenClean      = 0;  // neighborhood well-described by one plane
constexpr int32_t kDegenEdgeCorner = 1;  // high surface variation: ridge/corner/noise-blown-flat-direction
constexpr int32_t kDegenIsolated   = 2;  // fewer than kK neighbors found even after ring kMaxRing

// Surface-id vocabulary for the synthetic cohorts (scripts/make_synthetic.py
// writes these; main.cu groups gates by this field — see data/README.md).
constexpr int32_t kSurfacePlane    = 0;
constexpr int32_t kSurfaceSphere   = 1;
constexpr int32_t kSurfaceCylinder = 2;
constexpr int32_t kSurfaceEdge     = 3;   // the degeneracy cohort: no single true curvature (see data/README.md)

// blocks_for — integer ceiling division: how many `threads`-wide blocks
// cover `count` independent problems (the 02.01/02.04/02.05 idiom).
inline int blocks_for(int count, int threads)
{
    return (count + threads - 1) / threads;
}

// ===========================================================================
// Voxel-hash key packing — shared, plain-inline data-layout arithmetic
// (host+device compilable, per 02.01/02.05's precedent: unqualified so
// cl.exe sees it too; nvcc's __global__ kernels carry their own literal
// __device__ transcription, see kernels.cu's file header for why).
// ===========================================================================

// voxel_coord — which integer cell (along one axis) a world coordinate
// falls into at cell size `cell`. floor(), not truncation, so negative
// coordinates (routine: the sensor sits at the scene's origin, so roughly
// half of every real LiDAR scan has negative x/y) bucket correctly — floor
// rounds -0.1 into cell -1, not cell 0 (02.01's identical d_voxel_coord).
inline int32_t voxel_coord(float p, float cell)
{
    return static_cast<int32_t>(std::floor(p / cell));
}

// kHashCoordBias / kHashCoordMask21 — 21-bit-per-axis biased packing (the
// SAME scheme 02.01/02.04/02.05 use, re-derived here per the self-
// containment rule rather than shared across project folders): bias every
// signed cell coordinate by 2^20 so it becomes a non-negative 21-bit value
// (range +-2^20 cells = +-2^20 * 0.30 m ~= +-315 km at this project's cell
// size — no real scene here comes remotely close), then pack three 21-bit
// fields into one 64-bit key (63 bits used, top bit always 0).
constexpr int32_t  kHashCoordBias   = 1 << 20;
constexpr uint64_t kHashCoordMask21 = (1ull << 21) - 1ull;

inline unsigned long long pack_voxel_key(int32_t vx, int32_t vy, int32_t vz)
{
    const uint64_t ux = static_cast<uint64_t>(vx + kHashCoordBias) & kHashCoordMask21;
    const uint64_t uy = static_cast<uint64_t>(vy + kHashCoordBias) & kHashCoordMask21;
    const uint64_t uz = static_cast<uint64_t>(vz + kHashCoordBias) & kHashCoordMask21;
    return ux | (uy << 21) | (uz << 42);
}

// squared_distance3 — |p-q|^2. Squared (never sqrt) everywhere only the
// ORDERING or a threshold test matters (02.04/02.05's identical rationale):
// monotonic for x>=0, so every heap comparison below is exact without the
// cost of a square root per candidate.
inline float squared_distance3(const float p[3], const float q[3])
{
    const float dx = p[0] - q[0], dy = p[1] - q[1], dz = p[2] - q[2];
    return dx * dx + dy * dy + dz * dz;
}

// ---------------------------------------------------------------------------
// The shared total order every KNN heap (GPU, CPU twin, brute-force oracle)
// compares by, and its documented TIE-BREAK: smaller dist2 wins; on an
// EXACT dist2 tie, the SMALLER original point index wins (02.05's knn_less,
// cited, reimplemented). Because the K-heap's final contents are the K
// smallest elements under a FIXED total order regardless of the ORDER
// candidates are discovered in, every implementation below must produce the
// IDENTICAL final K-set and sorted order, not just an equivalent one —
// the precondition for VERIFY(knn)'s exact-equality gate.
// ---------------------------------------------------------------------------
inline bool knn_less(float da, int32_t ia, float db, int32_t ib)
{
    if (da != db) return da < db;
    return ia < ib;
}

// ===========================================================================
// The shared symmetric-3x3 eigensolver interface. See kernels.cu for the
// GPU __device__ transcription and reference_cpu.cpp for the independent
// CPU re-implementation (STEP 4's independence ruling: SAME algorithm
// family (cyclic Jacobi, 02.03 precedent), INDEPENDENTLY typed per the
// reference_cpu.cpp file header — not a shared function, so VERIFY(eigen)
// is a real cross-check, not a tautology).
//
// Input:  cov[6] — the upper triangle of the symmetric 3x3 covariance,
//         packed (c00,c01,c02,c11,c12,c22) — meters^2.
// Output: eigenvalues[3] ascending (lambda0 <= lambda1 <= lambda2);
//         eigenvectors[3][3], eigenvectors[i] the UNIT eigenvector for
//         eigenvalues[i] — eigenvectors[0] is this point's (unoriented)
//         surface normal candidate.
// ===========================================================================
constexpr int kJacobiSweeps = 8;  // 02.03's measured-sufficient sweep count for float32 3x3 Jacobi

// ===========================================================================
// Per-cohort synthetic-data metadata (scripts/make_synthetic.py writes an
// array of these; main.cu groups every gate by them — see data/README.md
// for the exact binary layout this struct mirrors field-for-field).
// ===========================================================================
struct Cohort {
    int32_t surface_id;   // kSurfacePlane/.../kSurfaceEdge
    int32_t noise_level;  // 0 = none, 1 = low (3 mm), 2 = high (15 mm) — along-normal sigma, see data/README.md
    int32_t start;        // first point index (inclusive) belonging to this cohort
    int32_t count;        // number of points in this cohort
    float   param;        // sphere/cylinder radius, meters; 0 for plane/edge
    float   axis_x, axis_y, axis_z;   // cylinder axis unit vector; 0,0,0 for non-cylinder cohorts
};

// ===========================================================================
// GPU kernel declarations — nvcc-only (see the file header for why: cl.exe,
// compiling reference_cpu.cpp, has never heard of __global__).
// ===========================================================================
#ifdef __CUDACC__

// ---- Voxel-hash index build (02.01 Method-B / 02.05 lineage, reused) ------

// compute_hash_keys_kernel — one thread per point: pack this point's voxel
// key at cell size kCellSizeM.
__global__ void compute_hash_keys_kernel(int n, const float* __restrict__ xyz,
                                         float cell, unsigned long long* __restrict__ keys);

// mark_boundaries_kernel — position 0, or any position whose key differs
// from its predecessor in the SORTED key array, starts a new voxel run.
__global__ void mark_boundaries_kernel(int n, const unsigned long long* __restrict__ keys_sorted,
                                       int* __restrict__ is_start);

// gather_unique_keys_kernel — one thread per occupied voxel v: copy that
// voxel's shared key into the dense unique_key[] array the KNN kernel
// binary-searches over.
__global__ void gather_unique_keys_kernel(int num_voxels, const int* __restrict__ seg_start,
                                          const unsigned long long* __restrict__ keys_sorted,
                                          unsigned long long* __restrict__ unique_key_out);

// ---- THE pipeline kernel ----------------------------------------------------

// estimate_normals_kernel — one thread per QUERY point q in [0, n): the
// fused KNN -> covariance -> eigensolve -> normal -> curvature ->
// degeneracy pipeline (file header, STEPS 2-7). Full documentation
// (thread mapping, register-pressure story, memory layout) sits with the
// definition in kernels.cu.
//
//   xyz            [n*3] IN: the point cloud (device), meters, sensor frame.
//   unique_key/seg_start/idx_sorted/n_sorted/num_voxels: the prebuilt voxel
//                  hash index (launch_build_voxel_index's output).
//   sensor         IN (by value): the sensor origin, meters, sensor frame —
//                  the viewpoint every normal is oriented toward (STEP 5).
//   out_normal     [n*3] OUT: unit, sensor-oriented surface normal.
//   out_eigenvalues[n*3] OUT: ascending lambda0<=lambda1<=lambda2, meters^2.
//   out_curvature  [n]   OUT: surface variation lambda0/(lambda0+lambda1+lambda2), unitless in [0, 1/3].
//   out_degeneracy [n]   OUT: kDegenClean/kDegenEdgeCorner/kDegenIsolated.
//   out_found      [n]   OUT: neighbors actually used (<=kK; ==kK unless isolated).
//   out_neighbor_ids [n*kK] OUT, OPTIONAL (pass nullptr to skip the write —
//                  the throughput pass does; the correctness pass does not,
//                  see kernels.cu): this point's kK nearest-neighbor point
//                  indices, ascending by knn_less, padded with -1 past
//                  out_found[q].
// ---------------------------------------------------------------------------
__global__ void estimate_normals_kernel(int n, const float* __restrict__ xyz,
                                        const unsigned long long* __restrict__ unique_key, int num_voxels,
                                        const int* __restrict__ seg_start,
                                        const int* __restrict__ idx_sorted, int n_sorted,
                                        float cell,
                                        float sensor_x, float sensor_y, float sensor_z,
                                        float* __restrict__ out_normal,
                                        float* __restrict__ out_eigenvalues,
                                        float* __restrict__ out_curvature,
                                        int32_t* __restrict__ out_degeneracy,
                                        int32_t* __restrict__ out_found,
                                        int32_t* __restrict__ out_neighbor_ids);

#endif // __CUDACC__

// ===========================================================================
// Host-callable launch wrappers (definitions in kernels.cu, which only nvcc
// compiles — but the DECLARATIONS below are plain C++, visible to main.cu).
// ===========================================================================

void launch_compute_hash_keys(int n, const float* d_xyz, float cell, unsigned long long* d_keys);

// launch_build_voxel_index — Thrust sort + boundary compaction (02.01
// Method-B / 02.05 pipeline, cited). d_keys_in [n] READ-ONLY; every other
// array is caller-provided scratch/output sized n. Returns the number of
// OCCUPIED voxels (== valid unique_key_out/seg_start entries).
int launch_build_voxel_index(int n, const unsigned long long* d_keys_in,
                             unsigned long long* d_keys_scratch, int* d_idx_scratch,
                             int* d_is_start_scratch, int* d_seg_start_out,
                             unsigned long long* d_unique_key_out);

void launch_estimate_normals(int n, const float* d_xyz,
                             const unsigned long long* d_unique_key, int num_voxels,
                             const int* d_seg_start, const int* d_idx_sorted, int n_sorted,
                             float cell, float sensor_x, float sensor_y, float sensor_z,
                             float* d_out_normal, float* d_out_eigenvalues, float* d_out_curvature,
                             int32_t* d_out_degeneracy, int32_t* d_out_found, int32_t* d_out_neighbor_ids);

// ===========================================================================
// CPU references (reference_cpu.cpp) — the correctness-oracle twins. All
// pointers below are HOST pointers. See reference_cpu.cpp's file header for
// the independence ruling.
// ===========================================================================

// jacobi_eigen_3x3_cpu — the CPU twin's OWN cyclic-Jacobi solve (same
// algorithm family as kernels.cu's __device__ version, independently typed
// — see reference_cpu.cpp). Shared here only as a DECLARATION (signature
// agreement); the two DEFINITIONS never call each other.
void jacobi_eigen_3x3_cpu(const float cov[6], float eigenvalues[3], float eigenvectors[3][3]);

// compute_hash_keys_cpu — the twin of compute_hash_keys_kernel (shared
// voxel-key FORMULA above; VERIFY-by-comparison, like 02.01/02.05).
void compute_hash_keys_cpu(int n, const float* xyz, float cell, unsigned long long* keys_out);

// HashMapCpu — the independent voxel-hash oracle's data structure: an
// std::unordered_map<uint64_t, std::vector<int>> voxel->points map — a
// DIFFERENT data structure than the GPU's sorted-array+binary-search index
// (02.04/02.05's identical "independent data structure" choice).
using HashMapCpu = std::unordered_map<unsigned long long, std::vector<int>>;
void build_hash_map_cpu(int n, const float* xyz, float cell, HashMapCpu& out_map);

// KnnResultCpu — one point's CPU-computed neighborhood + fitted geometry,
// the CPU twin's per-point output row (mirrors the GPU's flat output
// arrays; main.cu compares field by field).
struct KnnResultCpu {
    std::vector<int32_t> neighbor_ids;  // ascending by knn_less, size <= kK
    float normal[3];
    float eigenvalues[3];
    float curvature;
    int32_t degeneracy;
};

// estimate_normals_cpu — the twin of estimate_normals_kernel: an
// INDEPENDENT (retyped, not shared) sequential ring-expanding voxel-hash
// KNN + mean-shifted covariance + jacobi_eigen_3x3_cpu + sensor-oriented
// normal + curvature + degeneracy pipeline, one call per point.
void estimate_normals_cpu(int n, const float* xyz, const HashMapCpu& map, float cell,
                          float sensor_x, float sensor_y, float sensor_z,
                          int query_idx, KnnResultCpu& out);

// ---------------------------------------------------------------------------
// Brute-force oracle — THE third-tier independent verification gate
// (02.05's brute_force_anchor precedent): a hash-free O(n) linear scan per
// query, sharing no spatial-index code with EITHER twin above. Run only
// over a documented anchor subset (main.cu caps it — O(n) per query is
// deliberately never run over the full point set at this project's scale).
// ---------------------------------------------------------------------------
void estimate_normal_brute_force(int n, const float* xyz, float cell,
                                 float sensor_x, float sensor_y, float sensor_z,
                                 int query_idx, KnnResultCpu& out);

#endif // PROJECT_KERNELS_CUH
