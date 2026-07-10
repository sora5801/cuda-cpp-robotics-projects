// ===========================================================================
// kernels.cuh — interface for project 02.06
//               ICP: point-to-point → point-to-plane → GICP, all batched
//               (teaching core: point-to-point + point-to-plane, GPU brute
//               force; GICP documented as the third rung — see THEORY.md
//               "Where this sits in the real world")
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (the closed-loop ICP driver), kernels.cu (the
// GPU kernels), and reference_cpu.cpp (the CPU oracle twins). Everything all
// three must agree on — the point-cloud layout, the correspondence-record
// layout, the SE(3) pose representation, and the 27-scalar reduction layout
// — is defined HERE, once (CLAUDE.md §12: state layouts are single-sourced).
//
// ICP in six lines (THEORY.md derives the linearization properly)
// -----------------------------------------------------------------
//   1. Transform the source cloud by the current estimate T_est (GPU map).
//   2. For every transformed source point, find its nearest TARGET point
//      (GPU brute-force search — this project's perception workload).
//   3. Reject correspondences farther than a distance gate (bad matches).
//   4. Linearize the alignment error around T_est and accumulate a 6×6
//      Gauss-Newton normal system (GPU map + block reduction — the
//      project's central NEW GPU concept beyond thread-per-problem).
//   5. Solve the 6×6 system on the HOST (33.01-style Cholesky) for a twist
//      increment δ = [ω; v] and fold it into T_est via the SE(3) update.
//   6. Repeat until the twist increment is tiny or max_iters is hit.
// Steps 2 and 4 are >99% of the arithmetic and are embarrassingly parallel
// across points — the same thread-per-problem shape as 33.01/09.01/08.01,
// now applied to a PERCEPTION correspondence search AND a least-squares
// accumulation instead of a matrix solve or a rollout.
//
// Two ICP variants share every kernel below except the LAST one:
//   point-to-point — minimizes Euclidean distance to the matched point.
//   point-to-plane — minimizes distance along the matched point's SURFACE
//                    NORMAL only (computed once per target cloud, §PCA
//                    below); converges faster on planar scenes because it
//                    does not fight the "free slide" along a flat surface
//                    (THEORY.md §the-algorithm derives why).
// GICP (the catalog bullet's third rung — covariance-to-covariance, not
// point-to-plane) is taught in full in THEORY.md and shipped as documented
// milestone, not code (README §Limitations; CLAUDE.md §2/§13 reduced-scope
// rule for bundled catalog bullets).
//
// POINT CLOUD LAYOUT — float* xyz, interleaved, meters:
//     xyz[i*3 + 0] = x, xyz[i*3 + 1] = y, xyz[i*3 + 2] = z
// This mirrors docs/SYSTEM_DESIGN.md §3.6's `PointCloud` message sketch
// (itself a flattened sensor_msgs/PointCloud2) — the repo-wide convention
// every perception project speaks so chains like SYSTEM_DESIGN §4.1 Chain A
// (11.01 → 02.06 → 05.01) are literally the same struct shape end to end.
//
// NORMAL LAYOUT — float* nrm, SAME interleaved layout, unit vectors, computed
// ONCE per target cloud (§PCA below) and read-only for the rest of a run.
//
// CORRESPONDENCE RECORD — two parallel arrays, one entry per SOURCE point:
//     corr_idx[k]   int    index into the TARGET cloud, or -1 = rejected
//                          (no target point within max_dist_m — see
//                          launch_find_correspondences).
//     corr_dist2[k] float  squared Euclidean distance (m^2) to that match;
//                          only meaningful where corr_idx[k] >= 0. Feeds the
//                          per-iteration RMS logged to demo/out/convergence.csv.
//
// SE(3) POSE ESTIMATE — T_target_source (09.01's naming convention: "source
// cloud's frame, expressed in the target cloud's frame"; applying it to a
// source point lands that point in the target frame, which is exactly what
// ICP needs each iteration). Represented as the repo-standard pair:
//     float t[3]     translation (m), target frame
//     float q[4]     unit quaternion (w,x,y,z) — REPO ORDER (CLAUDE.md §12),
//                     kept normalized after every update (THEORY.md
//                     §numerics discusses the drift this guards against).
// main.cu owns this pair; kernels only ever see the DERIVED rotation matrix
// (see Rigid3 below) because every kernel needs R·p+t per point, not the
// quaternion itself.
//
// Rigid3 — how R,t reach the kernels. UNLIKE 09.01 (whose robot model is
// the SAME for an entire batch of thousands of configurations, justifying a
// one-time __constant__-memory upload via set_robot_model()), T_est CHANGES
// every ICP iteration here — re-uploading via cudaMemcpyToSymbol every
// iteration would be pure overhead for 48 bytes. Instead Rigid3 is passed BY
// VALUE as an ordinary kernel argument: the CUDA compiler places kernel
// parameters in a small dedicated constant-ish parameter bank that every
// thread reads with the same broadcast efficiency as __constant__ memory,
// with none of the upload boilerplate. This is the third point on the
// "how do 30000 threads read the same handful of floats" spectrum that runs
// through this repository: 09.01's __constant__ symbol (static, batch-wide
// config) → 08.01's uniform global read of u_nom (changes every tick, read
// many times per kernel) → here, kernel parameters (changes every tick,
// read exactly ONCE per thread) — cheapest to update, right for this shape.
//
// Why this header is CUDA-qualifier-free (no __CUDACC__ fence needed): every
// declaration below is a plain host-callable function or POD struct — no
// __global__/__device__ signatures live here at all (unlike the template's
// saxpy example). hidx() and blocks_for() are ordinary inline host
// functions, used by main.cu and reference_cpu.cpp for the SAME 27-scalar
// bookkeeping that kernels.cu's device code performs with literal indices
// (documented at the kernel definition) — this avoids needing __host__
// __device__ dual-compilation tricks for two tiny index-arithmetic helpers.
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

// ---------------------------------------------------------------------------
// Rigid3 — a rigid-body transform, passed by value to every kernel that
// needs "where is the source cloud right now". R is row-major (R[i*3+j] is
// row i, column j — CLAUDE.md §12 SI/right-handed convention); t is meters.
// x_target = R * p_source + t.
// ---------------------------------------------------------------------------
struct Rigid3 {
    float R[9];   // row-major 3x3 rotation (orthonormal — main.cu re-derives
                  // it from the quaternion estimate every iteration so it
                  // can never accumulate its OWN drift independent of q)
    float t[3];   // translation (m), target frame
};

// ---------------------------------------------------------------------------
// IcpMode — selects the residual model build_normal_system uses. Both modes
// share the correspondence search (Euclidean nearest-neighbor is the right
// MATCHING rule either way — Low 2004 and PCL's point-to-plane estimator
// both still match by Euclidean distance and only change the ERROR metric
// once matched); only the linearized system differs (THEORY.md §the-math).
// ---------------------------------------------------------------------------
enum IcpMode {
    kPointToPoint = 0,   // residual r = x - q (3 rows); minimizes |x-q|^2
    kPointToPlane = 1    // residual e = n·(x-q) (1 row); minimizes the
                         // distance projected onto the matched surface normal
};

// ---------------------------------------------------------------------------
// Launch-geometry constants (repo default block sizes, sized with reasoning
// that lives beside each kernel in kernels.cu).
// ---------------------------------------------------------------------------
constexpr int kThreadsPerBlock = 256;  // transform / correspondence / normals
constexpr int kThreadsReduce   = 128;  // build_normal_system — shared-memory
                                       // budget reasoning is in kernels.cu

// PCA normal estimation: how many nearest neighbors (including the query
// point itself) form the local neighborhood whose covariance we eigen-
// decompose, and how many Jacobi sweeps zero the off-diagonal 3x3 entries.
// k=16 is a standard PCL-style default (enough points to fit a stable local
// plane, few enough to stay a brute-force inner loop); 8 sweeps of a 3x3
// symmetric Jacobi eigensolve is generous overkill for 3x3 (textbook
// practice converges in 3-5 sweeps; THEORY.md §numerics quantifies it).
constexpr int kPcaK          = 16;
constexpr int kJacobiSweeps  = 8;

// Default correspondence-rejection gate (meters). Sized generously (see
// THEORY.md §the-algorithm) so that iteration-0's large initial misalignment
// (rotation-induced displacement can exceed the translation itself for
// points far from the rotation's pivot) still finds valid matches on most
// of the scene, while still rejecting points that landed on the wrong
// surface entirely (e.g. off the edge of a wall).
constexpr float kDefaultMaxCorrDist = 2.0f;   // m

// The 6x6 Gauss-Newton normal system's upper triangle (SYMMETRIC matrix —
// CLAUDE.md §12: the lower-left block is JUST the transpose of the upper-
// right block by construction, THEORY.md §the-math shows why) has
// 6+5+4+3+2+1 = 21 unique entries; plus the 6-entry right-hand side g,
// that is 27 scalars — the number every block-reduction kernel and every
// CPU twin below produces per point and accumulates per block.
//
// Parameter order (both H's rows/cols and g): [wx, wy, wz, vx, vy, vz] —
// rotation-first then translation, the twist ordering THEORY.md derives.
// wx..wz are a SMALL-ANGLE rotation vector (rad, about target-frame axes);
// vx..vz are a translation increment (m, target frame) — see the SE(3)
// update note in main.cu for exactly how they fold back into T_est.
constexpr int kReduceWidth = 27;   // 21 (H upper triangle) + 6 (g)

// hidx(i, j) — flatten the upper-triangle (i <= j) index of a 6x6 symmetric
// matrix into the 0..20 offset used by every H array below. Row i's valid
// columns are j = i..5 (6-i of them), so row i starts right after all
// earlier rows' entries: row_start = {0, 6, 11, 15, 18, 20} (row 0 has 6
// entries so row 1 starts at 6; row 1 has 5 so row 2 starts at 11; ...).
// Used by main.cu (unpacking the reduced 27-vector into a full 6x6 for the
// Cholesky solve) and reference_cpu.cpp (building H21 in the same order);
// kernels.cu's device code writes the SAME 27 slots directly by documented
// literal index (see the kernel's header comment) rather than calling this
// function from device code — avoiding a __host__ __device__ dual-
// compilation fence for one tiny piece of index arithmetic.
// ---------------------------------------------------------------------------
inline int hidx(int i, int j)
{
    // Precomputed row-start table (derivation in the comment above); a
    // small constant array is clearer than re-deriving the closed form
    // 6*i - i*(i-1)/2 at every call site, and costs nothing at these sizes.
    const int row_start[6] = { 0, 6, 11, 15, 18, 20 };
    return row_start[i] + (j - i);   // caller guarantees i <= j <= 5
}

// blocks_for(count, threads) — integer ceiling division, the same idiom
// 33.01/08.01 use: how many `threads`-wide blocks cover `count` independent
// problems. Declared here (not just inside kernels.cu) because main.cu must
// compute the IDENTICAL block count to size the block-partial download
// buffer for launch_build_normal_system — single source of truth prevents a
// silent off-by-one between the launcher and the caller.
inline int blocks_for(int count, int threads)
{
    return (count + threads - 1) / threads;
}

// ---------------------------------------------------------------------------
// launch_transform_cloud — GPU map: out[k] = T.R * src[k] + T.t for every
// source point k. The first stage of every ICP iteration (THEORY.md's step
// 1) and also how main.cu produces the "aligned" cloud for the plotting
// artifact once the loop finishes.
//
//   n       : point count (>= 0; 0 is a valid no-op).
//   d_src_xyz : DEVICE pointer, n*3 floats, layout above. Never written.
//   T       : the rigid transform to apply (passed BY VALUE — see the Rigid3
//             comment above for why).
//   d_out_xyz : DEVICE pointer, n*3 floats, OVERWRITTEN. May alias d_src_xyz
//               (each thread reads its own point fully before writing it).
//
// Launch: one thread per point, kThreadsPerBlock-thread blocks.
// ---------------------------------------------------------------------------
void launch_transform_cloud(int n, const float* d_src_xyz, Rigid3 T, float* d_out_xyz);

// ---------------------------------------------------------------------------
// launch_find_correspondences — GPU brute-force nearest-neighbor search:
// for every point in the (already transformed) source cloud, scan the WHOLE
// target cloud and keep the closest point within max_dist_m.
//
//   n_src, d_cur_xyz : the transformed source cloud (n_src*3 floats).
//   m_tgt, d_tgt_xyz : the target cloud being searched (m_tgt*3 floats).
//   max_dist_m       : correspondence-rejection gate (m); see
//                       kDefaultMaxCorrDist above for the sizing story.
//   d_corr_idx       : DEVICE pointer, n_src ints OUT (layout above).
//   d_corr_dist2     : DEVICE pointer, n_src floats OUT (layout above).
//
// Complexity: O(n_src * m_tgt), embarrassingly parallel across n_src — the
// honest teaching choice (a KD-tree/LBVH search is project 02.05's job and
// this project's README Exercise 1). Launch: one thread per SOURCE point;
// every thread in a warp scans the SAME target index at the SAME loop
// iteration (no data-dependent branching in the scan), so target reads are
// a broadcast, not a gather — kernels.cu's header comment measures this.
// ---------------------------------------------------------------------------
void launch_find_correspondences(int n_src, const float* d_cur_xyz,
                                 int m_tgt, const float* d_tgt_xyz,
                                 float max_dist_m,
                                 int* d_corr_idx, float* d_corr_dist2);

// ---------------------------------------------------------------------------
// launch_estimate_normals — per-target-point surface normal via PCA over
// its kPcaK nearest neighbors (brute-force, self included), eigen-decomposed
// with an in-register Jacobi sweep (kJacobiSweeps). Computed ONCE per target
// cloud (the target cloud is fixed for the whole ICP run — only the SOURCE
// cloud moves), then reused by every point-to-plane iteration.
//
//   m_tgt, d_tgt_xyz  : target cloud (m_tgt*3 floats).
//   ref_point         : a point "inside" the scanned volume (m, target
//                       frame), used ONLY to pick a consistent sign for each
//                       normal — see below. main.cu passes the target
//                       cloud's own centroid: for a mostly-enclosing shell
//                       of surface points (floor + walls + a box, this
//                       project's scene — or any room/vehicle-interior-like
//                       scan), the centroid of the SURFACE POINTS naturally
//                       falls in the interior, with no scene-specific
//                       tuning required.
//   d_tgt_normals     : DEVICE pointer, m_tgt*3 floats OUT, unit vectors.
//
// Launch: one thread per target point. Sign convention: THEORY.md §the-math
// proves the point-to-plane linear system is invariant to a PER-POINT sign
// flip of the normal (it appears twice, quadratically, in every term), so
// orientation is not needed for CORRECTNESS — we still orient every normal
// to point toward ref_point purely so the artifact plot and a learner's
// intuition are not confused by a randomly speckled normal field.
// ---------------------------------------------------------------------------
void launch_estimate_normals(int m_tgt, const float* d_tgt_xyz,
                             const float ref_point[3], float* d_tgt_normals);

// ---------------------------------------------------------------------------
// launch_build_normal_system — the project's central NEW GPU concept: turn
// n_src independent per-point contributions into ONE 6x6 Gauss-Newton
// normal system via a two-stage reduction.
//
//   n_src, d_cur_xyz  : transformed source cloud (the "x_i" of THEORY.md).
//   d_tgt_xyz         : target cloud (the "q_i", indexed via d_corr_idx).
//   d_tgt_normals     : target normals; REQUIRED for kPointToPlane, IGNORED
//                       (may be nullptr) for kPointToPoint.
//   d_corr_idx        : correspondences from launch_find_correspondences;
//                       entries with idx < 0 contribute nothing (rejected).
//   mode              : kPointToPoint or kPointToPlane (selects the
//                       per-point formulas — THEORY.md §the-math).
//   d_block_partials  : DEVICE pointer, blocks_for(n_src,kThreadsReduce)
//                       * kReduceWidth floats OUT — block b's 27-scalar
//                       PARTIAL sum (H upper triangle then g) lives at
//                       d_block_partials[b*kReduceWidth .. +27). Stage 1
//                       (this kernel) reduces WITHIN a block via shared
//                       memory; stage 2 (main.cu) sums the per-block rows
//                       on the host — the same "GPU partial reduce, host
//                       finishes it" split 08.01 uses for its softmin
//                       blend, applied here to a least-squares accumulator
//                       instead of a weighted average (kernels.cu explains
//                       the shared-memory layout and why it is coalesced).
//
// Launch: one thread per SOURCE point, kThreadsReduce-thread blocks (a
// SMALLER block than the repo default — see kernels.cu for the shared-
// memory budget that motivates it).
// ---------------------------------------------------------------------------
void launch_build_normal_system(int n_src, const float* d_cur_xyz,
                                const float* d_tgt_xyz,
                                const float* d_tgt_normals,
                                const int* d_corr_idx,
                                IcpMode mode,
                                float* d_block_partials);

// ---------------------------------------------------------------------------
// CPU references (reference_cpu.cpp) — the correctness oracle twins of the
// four launchers above. Same math, same layouts, plain single-threaded
// C++, no CUDA anywhere. main.cu runs these against the GPU kernels on
// iteration 0's exact inputs and requires agreement within documented
// tolerances (the §5 GPU-vs-CPU gate for this project — see main.cu's
// VERIFY stage). All pointers below are HOST pointers.
//
// build_normal_system_cpu deliberately returns DOUBLE H/g: the GPU path
// reduces in FP32 (block-tree reduction, then a host double sum of the
// block partials — see launch_build_normal_system above); the CPU oracle
// skips the block-partial step entirely and accumulates every point's
// contribution directly into a double accumulator, so it is the more
// precise of the two paths by construction. main.cu compares the GPU's
// (float-reduced, then double-summed) result against this oracle with a
// documented RELATIVE tolerance, never bit equality (THEORY.md §numerics —
// the same "reduction reordering changes the last few bits" story 08.01
// and 33.01 tell, applied to a tree reduction instead of RK4 chaining).
// ---------------------------------------------------------------------------
void transform_cloud_cpu(int n, const float* src_xyz, const Rigid3& T, float* out_xyz);

void find_correspondences_cpu(int n_src, const float* cur_xyz,
                              int m_tgt, const float* tgt_xyz,
                              float max_dist_m,
                              int* corr_idx, float* corr_dist2);

void estimate_normals_cpu(int m_tgt, const float* tgt_xyz,
                          const float ref_point[3], float* tgt_normals);

void build_normal_system_cpu(int n_src, const float* cur_xyz,
                             const float* tgt_xyz, const float* tgt_normals,
                             const int* corr_idx, IcpMode mode,
                             double H21[21], double g6[6]);

#endif // PROJECT_KERNELS_CUH
