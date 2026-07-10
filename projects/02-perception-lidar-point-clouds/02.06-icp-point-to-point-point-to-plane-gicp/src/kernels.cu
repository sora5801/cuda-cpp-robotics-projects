// ===========================================================================
// kernels.cu — GPU kernels for project 02.06
//              ICP: point-to-point → point-to-plane → GICP, all batched
//              (teaching core: point-to-point + point-to-plane, brute-force
//              GPU correspondence search + a two-stage least-squares
//              reduction — see kernels.cuh for the full contract)
//
// Four kernels, run in this order every ICP iteration (main.cu orchestrates
// the loop; kernels.cuh documents each launcher's contract in full):
//
//   1. transform_cloud_kernel     — map: move the source cloud by T_est.
//   2. find_correspondences_kernel— brute-force nearest-neighbor search.
//   3. estimate_normals_kernel    — PCA normals (ONCE per target cloud,
//                                   not every iteration — see kernels.cuh).
//   4. build_normal_system_kernel — the NEW pattern this project teaches:
//                                   per-point least-squares contributions,
//                                   reduced within a block via shared
//                                   memory into one 27-scalar partial sum.
//
// What is NEW here beyond 33.01/09.01/08.01's thread-per-problem pattern:
//   * a brute-force SEARCH inside the kernel (correspondence) instead of a
//     fixed-size closed-form computation — the inner loop bound (m_tgt) is
//     the SAME for every thread in a warp, so the search has no thread
//     divergence even though it is data-dependent (kernels.cuh calls this
//     out: every lane reads the SAME target index at the SAME loop step,
//     making the target-cloud reads a broadcast, not a gather);
//   * an in-register EIGENSOLVE (Jacobi sweeps on a 3x3 symmetric matrix)
//     feeding a per-thread small-linear-algebra step, echoing 33.01's
//     register-resident Cholesky but for an eigenproblem instead of a solve;
//   * a REAL two-stage reduction (shared-memory block partials → host final
//     sum) — 08.01's softmin blend also finishes on the host, but never
//     needed a *block*-level reduction first because it read whole arrays,
//     not per-thread partial results; this project's normal-system build is
//     the first place in the repo's flagships where a genuine tree
//     reduction inside a kernel is required, because separate SOURCE POINTS
//     independently contribute to the SAME 27-number answer.
//
// Read this after: kernels.cuh.  Companion oracle: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"      // CUDA_CHECK_LAST_ERROR (§6.1 rule 7)

#include <cfloat>                   // FLT_MAX — the "nothing found yet" sentinel
#include <cstdio>                   // fprintf for the invalid-mode abort message
#include <cstdlib>                  // std::exit

// ===========================================================================
// Kernel 1: transform_cloud_kernel — out[k] = T.R * src[k] + T.t
// ===========================================================================
// The simplest possible map (SAXPY's shape, applied to points instead of
// scalars): one thread per point, one read, nine FMAs, one write. Included
// here (rather than folded into the correspondence kernel) so the
// TRANSFORMED cloud is a materialized array other kernels and the artifact
// writer can all read — a small memory-for-simplicity trade the repo makes
// throughout (compare 08.01's separate rollout vs. plant-step functions).
// ---------------------------------------------------------------------------
__global__ void transform_cloud_kernel(const float* __restrict__ src_xyz,  // [n*3] source points (m)
                                       Rigid3 T,                           // by-value transform (kernels.cuh)
                                       float*       __restrict__ out_xyz,  // [n*3] OUT: transformed points (m)
                                       int n)                              // point count
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's point index
    if (k >= n) return;                                   // ragged-tail guard

    // One point in registers; row-major R means row i is R[i*3 .. i*3+2].
    const float px = src_xyz[k * 3 + 0];
    const float py = src_xyz[k * 3 + 1];
    const float pz = src_xyz[k * 3 + 2];

    out_xyz[k * 3 + 0] = fmaf(T.R[0], px, fmaf(T.R[1], py, fmaf(T.R[2], pz, T.t[0])));
    out_xyz[k * 3 + 1] = fmaf(T.R[3], px, fmaf(T.R[4], py, fmaf(T.R[5], pz, T.t[1])));
    out_xyz[k * 3 + 2] = fmaf(T.R[6], px, fmaf(T.R[7], py, fmaf(T.R[8], pz, T.t[2])));
}

// ===========================================================================
// Kernel 2: find_correspondences_kernel — brute-force nearest neighbor.
// ===========================================================================
// Each thread OWNS one transformed source point and scans the ENTIRE target
// cloud. This is O(n_src * m_tgt) work, but every lane of a warp executes
// the identical loop bound (m_tgt) with no data-dependent branch inside the
// loop body — so at loop step m, all 32 lanes read tgt_xyz[m*3 .. m*3+2] at
// once: the SAME address for every thread, a broadcast the L1/L2 cache
// serves in one transaction, not 32 scattered ones. This is the same
// "uniform read" shape as 08.01's u_nom[t] (kernels.cuh's header comment
// places it on that same spectrum). A KD-tree/BVH search (02.05's project)
// would cut the O(m_tgt) factor to O(log m_tgt) at the cost of a much more
// complex, DIVERGENT traversal — the honest teaching trade-off this project
// makes explicitly (README Exercise 1 asks the learner to go build it).
// ---------------------------------------------------------------------------
__global__ void find_correspondences_kernel(
    const float* __restrict__ cur_xyz,     // [n_src*3] transformed source points (m)
    int n_src,                             // source point count
    const float* __restrict__ tgt_xyz,     // [m_tgt*3] target cloud being searched (m)
    int m_tgt,                             // target point count
    float max_dist2,                       // rejection gate, SQUARED (m^2) — squared once on
                                            // the host so 30000 threads don't each take a sqrt
    int*   __restrict__ corr_idx,          // [n_src] OUT: nearest target index, or -1
    float* __restrict__ corr_dist2)        // [n_src] OUT: squared distance to that match (m^2)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_src) return;

    // The query point lives in registers for the WHOLE scan — read once,
    // reused m_tgt times; this is why the search is compute-bound on the
    // subtraction/FMA chain rather than memory-bound on re-reading cur_xyz.
    const float px = cur_xyz[k * 3 + 0];
    const float py = cur_xyz[k * 3 + 1];
    const float pz = cur_xyz[k * 3 + 2];

    float best_d2 = FLT_MAX;   // "nothing found yet" — any real distance beats this
    int   best_j  = -1;        // no match found yet

    // The brute-force scan. No shared memory: target points are NOT reused
    // across ITERATIONS of the outer ICP loop within this single kernel
    // call, and every thread needs the full target cloud, not a tile of it
    // (contrast with classic tiled GEMM, where shared memory earns its keep
    // because the SAME tile is reused by many threads across many output
    // elements — THEORY.md §GPU-mapping expands this comparison).
    for (int m = 0; m < m_tgt; ++m) {
        const float dx = tgt_xyz[m * 3 + 0] - px;
        const float dy = tgt_xyz[m * 3 + 1] - py;
        const float dz = tgt_xyz[m * 3 + 2] - pz;
        const float d2 = fmaf(dz, dz, fmaf(dy, dy, dx * dx));
        if (d2 < best_d2) { best_d2 = d2; best_j = m; }   // strict '<': first
                                                          // minimum wins on
                                                          // ties, matching
                                                          // reference_cpu.cpp
    }

    // Apply the rejection gate AFTER the search (searching for the true
    // nearest neighbor first, then gating, is both simpler and correct —
    // gating during the scan would only save a compare, not real work).
    if (best_j >= 0 && best_d2 <= max_dist2) {
        corr_idx[k] = best_j;
    } else {
        corr_idx[k] = -1;      // rejected: too far, or (m_tgt==0) no candidate
    }
    corr_dist2[k] = best_d2;   // kept even when rejected — useful for
                               // diagnosing the gate distance, never read
                               // downstream unless corr_idx[k] >= 0
}

// ===========================================================================
// jacobi_eigen_3x3 — in-register eigendecomposition of a symmetric 3x3.
// ===========================================================================
// The classical CYCLIC JACOBI method (Numerical Recipes §11.1): repeatedly
// pick an off-diagonal entry and apply a 2x2 rotation that zeroes it. For a
// FIXED 3x3, there are only three off-diagonal pairs — (0,1), (0,2), (1,2)
// — so "cyclic" here just means visiting all three, every sweep, in order.
// Each rotation may re-grow an entry a previous rotation had zeroed, but the
// SUM OF SQUARED off-diagonal entries strictly decreases every rotation (a
// classical convergence proof), so a handful of sweeps drives it to
// numerical noise — kJacobiSweeps=8 is generous for 3x3 (THEORY.md
// §numerics measures the residual after fewer sweeps).
//
// On exit: A's diagonal holds the three eigenvalues (in whatever order the
// sweeps left them — the caller picks the one it wants, see the normal
// kernel below); V's COLUMNS hold the corresponding eigenvectors.
//
// Why Jacobi over the closed-form cubic-equation eigenvalue formula? The
// cubic form needs a careful acos/trig branch near degenerate (repeated-
// eigenvalue) inputs to stay numerically sound; Jacobi has no special case
// at all — the same straight-line code handles every input, including the
// perfectly-flat local neighborhoods this project's PCA normals see
// constantly (a plane's covariance has ONE eigenvalue near zero and TWO
// nearly equal — exactly the configuration the cubic form struggles with).
// Teaching beats cleverness (CLAUDE.md §1): the simpler, robust method wins.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void jacobi_rotate(float A[3][3], float V[3][3], int p, int q)
{
    const float apq = A[p][q];
    // Skip a rotation that would do nothing (already-zero off-diagonal) —
    // guards the division below AND saves work on later sweeps once the
    // matrix has mostly converged.
    if (fabsf(apq) < 1e-12f) return;

    // Standard Jacobi rotation angle, computed via its numerically-stable
    // tangent form (Numerical Recipes) rather than a direct atan2/atan —
    // avoids the extra transcendental call and a branch for apq==0.
    const float theta = (A[q][q] - A[p][p]) / (2.0f * apq);
    const float t = (theta >= 0.0f ? 1.0f : -1.0f) / (fabsf(theta) + sqrtf(theta * theta + 1.0f));
    const float c = rsqrtf(t * t + 1.0f);   // cos(rotation angle)
    const float s = t * c;                  // sin(rotation angle)

    const float app = A[p][p], aqq = A[q][q];
    A[p][p] = app - t * apq;
    A[q][q] = aqq + t * apq;
    A[p][q] = 0.0f;
    A[q][p] = 0.0f;

    // The THIRD row/column (neither p nor q) also rotates — this is what
    // makes Jacobi iterative rather than one-shot for N > 2.
#pragma unroll
    for (int r = 0; r < 3; ++r) {
        if (r == p || r == q) continue;
        const float arp = A[r][p], arq = A[r][q];
        A[r][p] = c * arp - s * arq;  A[p][r] = A[r][p];   // symmetric: mirror both sides
        A[r][q] = s * arp + c * arq;  A[q][r] = A[r][q];
    }

    // Accumulate the SAME rotation into V so its columns end up holding
    // eigenVECTORS, not just tracking the eigenVALUES in A's diagonal.
#pragma unroll
    for (int r = 0; r < 3; ++r) {
        const float vrp = V[r][p], vrq = V[r][q];
        V[r][p] = c * vrp - s * vrq;
        V[r][q] = s * vrp + c * vrq;
    }
}

__device__ __forceinline__ void jacobi_eigen_3x3(float A[3][3], float V[3][3])
{
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j)
            V[i][j] = (i == j) ? 1.0f : 0.0f;   // V starts as the identity

#pragma unroll
    for (int sweep = 0; sweep < kJacobiSweeps; ++sweep) {
        jacobi_rotate(A, V, 0, 1);
        jacobi_rotate(A, V, 0, 2);
        jacobi_rotate(A, V, 1, 2);
    }
}

// ===========================================================================
// Kernel 3: estimate_normals_kernel — PCA surface normal per target point.
// ===========================================================================
// One thread per target point j: brute-force scan the target cloud for the
// kPcaK nearest neighbors (SAME broadcast-read shape as correspondence
// search above), form their 3x3 covariance about the local centroid, and
// take the eigenvector of the SMALLEST eigenvalue — the direction the
// neighborhood varies LEAST, i.e. the local surface normal (THEORY.md
// derives why PCA's minimum-variance direction is the normal).
//
// Memory: the k=16 neighbor candidates (distance + xyz) live in FOUR
// register arrays of 16 floats each (64 registers) PLUS the covariance/
// Jacobi scratch (~20 more) — a genuinely register-heavy kernel, the same
// honest trade-off 33.01's N=6 Cholesky makes: fewer resident warps per SM,
// but each thread does enough arithmetic per byte that occupancy is not the
// bottleneck (THEORY.md §GPU-mapping profiles this).
//
// Complexity per thread: O(m_tgt) for the scan (the dominant cost) plus
// O(m_tgt * kPcaK) worst case for neighbor-list maintenance (in practice
// far less — insertions become rare once the top-k stabilizes) plus O(1)
// (kPcaK-independent) for the covariance + Jacobi step. Run ONCE per target
// cloud, not per ICP iteration (kernels.cuh).
// ---------------------------------------------------------------------------
__global__ void estimate_normals_kernel(
    const float* __restrict__ tgt_xyz,   // [m_tgt*3] target cloud (m)
    int m_tgt,
    float ref0, float ref1, float ref2,  // orientation reference point (m) — see kernels.cuh
    float* __restrict__ tgt_normals)     // [m_tgt*3] OUT: unit normals
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= m_tgt) return;

    const float qx = tgt_xyz[j * 3 + 0];
    const float qy = tgt_xyz[j * 3 + 1];
    const float qz = tgt_xyz[j * 3 + 2];

    // The k nearest neighbors found so far: UNSORTED arrays plus a tracked
    // "worst" slot. Insertion policy: on a new closer-than-worst candidate,
    // overwrite the worst slot, then re-scan all k slots for the new worst
    // (O(k) — cheap next to the O(m_tgt) outer scan). A sorted array would
    // cost more per insertion (shifting) for no benefit here: we only ever
    // need "what's the worst of my current k", never their rank order.
    float nb_d2[kPcaK], nb_x[kPcaK], nb_y[kPcaK], nb_z[kPcaK];
#pragma unroll
    for (int i = 0; i < kPcaK; ++i) nb_d2[i] = FLT_MAX;
    int   worst    = 0;
    float worst_d2 = FLT_MAX;

    for (int m = 0; m < m_tgt; ++m) {
        const float mx = tgt_xyz[m * 3 + 0];
        const float my = tgt_xyz[m * 3 + 1];
        const float mz = tgt_xyz[m * 3 + 2];
        const float dx = mx - qx, dy = my - qy, dz = mz - qz;
        const float d2 = fmaf(dz, dz, fmaf(dy, dy, dx * dx));   // includes m==j (d2=0):
                                                                // the query point IS
                                                                // its own nearest
                                                                // neighbor and is
                                                                // deliberately kept
                                                                // in the neighborhood
                                                                // (standard PCA-normal
                                                                // practice, e.g. PCL)
        if (d2 < worst_d2) {
            nb_d2[worst] = d2; nb_x[worst] = mx; nb_y[worst] = my; nb_z[worst] = mz;
            // Re-find the worst of the (still size-k) neighbor set.
            worst = 0; worst_d2 = nb_d2[0];
#pragma unroll
            for (int i = 1; i < kPcaK; ++i) {
                if (nb_d2[i] > worst_d2) { worst_d2 = nb_d2[i]; worst = i; }
            }
        }
    }

    // Centroid of the k-neighborhood.
    float cx = 0.0f, cy = 0.0f, cz = 0.0f;
#pragma unroll
    for (int i = 0; i < kPcaK; ++i) { cx += nb_x[i]; cy += nb_y[i]; cz += nb_z[i]; }
    const float inv_k = 1.0f / static_cast<float>(kPcaK);
    cx *= inv_k; cy *= inv_k; cz *= inv_k;

    // 3x3 covariance about the centroid — the "shape" of the local
    // neighborhood. A flat patch has one near-zero eigenvalue (across the
    // surface's thickness) and two larger, roughly-equal eigenvalues
    // (spanning the surface) — THEORY.md derives this from first principles.
    float cxx = 0.0f, cyy = 0.0f, czz = 0.0f, cxy = 0.0f, cxz = 0.0f, cyz = 0.0f;
#pragma unroll
    for (int i = 0; i < kPcaK; ++i) {
        const float dx = nb_x[i] - cx, dy = nb_y[i] - cy, dz = nb_z[i] - cz;
        cxx += dx * dx; cyy += dy * dy; czz += dz * dz;
        cxy += dx * dy; cxz += dx * dz; cyz += dy * dz;
    }

    float A[3][3] = { { cxx, cxy, cxz }, { cxy, cyy, cyz }, { cxz, cyz, czz } };
    float V[3][3];
    jacobi_eigen_3x3(A, V);

    // Smallest eigenvalue's eigenvector (a COLUMN of V) is the normal.
    int lo = 0;
    if (A[1][1] < A[lo][lo]) lo = 1;
    if (A[2][2] < A[lo][lo]) lo = 2;
    float nx = V[0][lo], ny = V[1][lo], nz = V[2][lo];

    // Defensive renormalization: V's columns are orthonormal by
    // construction (a product of rotations), but ~24 chained float32
    // rotations can leave ~1e-6-level drift — cheap to clean up, and
    // point-to-plane's math ASSUMES a unit normal (kernels.cuh contract).
    const float inv_len = rsqrtf(fmaf(nz, nz, fmaf(ny, ny, nx * nx)));
    nx *= inv_len; ny *= inv_len; nz *= inv_len;

    // Orientation for plotting/intuition only (kernels.cuh: the ICP math
    // itself is provably invariant to a per-point sign flip — THEORY.md
    // §the-math). Point the normal toward ref_point.
    const float to_ref_x = ref0 - qx, to_ref_y = ref1 - qy, to_ref_z = ref2 - qz;
    if (fmaf(nz, to_ref_z, fmaf(ny, to_ref_y, nx * to_ref_x)) < 0.0f) {
        nx = -nx; ny = -ny; nz = -nz;
    }

    tgt_normals[j * 3 + 0] = nx;
    tgt_normals[j * 3 + 1] = ny;
    tgt_normals[j * 3 + 2] = nz;
}

// ===========================================================================
// Kernel 4: build_normal_system_kernel — the project's central NEW pattern.
// ===========================================================================
// Every VALID correspondence (source point k with corr_idx[k] >= 0)
// contributes 27 numbers — the upper triangle of a 6x6 outer-product-like
// matrix H plus a 6-vector g — to a SHARED linear system. Unlike every
// other kernel in this project (and most of the repo's flagships), this is
// not "one thread, one independent answer": thousands of threads' numbers
// must be SUMMED into one final H, g. That summation is a REDUCTION, and
// this kernel implements its first stage.
//
// Two variants share this kernel via a COMPILE-TIME template parameter
// (`if constexpr`, C++17): each instantiation contains ONLY the arithmetic
// for its mode — no runtime branch, no wasted work computing the other
// mode's terms. This mirrors 33.01's template<int N> dispatch: a runtime
// enum picked once by the LAUNCHER, not by every one of 30000 threads.
//
// THE REDUCTION, IN DETAIL (THEORY.md §GPU-mapping walks the derivation):
//   Stage 1 (THIS kernel): each thread computes its point's 27-scalar
//     contribution into REGISTERS (`local[27]`), then the whole block
//     cooperatively sums those contributions via SHARED MEMORY, using the
//     classic binary-tree reduction (halve the active thread count each
//     step). The block's single combined 27-scalar answer is written to
//     `block_partials[blockIdx.x]`.
//   Stage 2 (main.cu, HOST): sum the (few hundred) block partials into the
//     final H, g — small enough that a simple host loop is both correct and
//     the clearest possible teaching code (the same choice 08.01 makes for
//     its softmin blend: finish a genuinely small reduction on the host
//     rather than write a second, tiny kernel — README Exercise names the
//     GPU-side finisher as the natural next step).
//
// WHY CHANNEL-MAJOR SHARED MEMORY (`sdata[channel][thread]`, flattened as
// `sdata[c*blockDim.x + tid]`) INSTEAD OF THREAD-MAJOR (`sdata[thread][channel]`)?
// During the tree reduction, at a given step s, active thread `tid` reads
// `sdata[c*blockDim.x + tid]` and `sdata[c*blockDim.x + tid + s]` — for a
// FIXED channel c, consecutive threads address CONSECUTIVE shared-memory
// banks (stride 1), the conflict-free pattern. Thread-major layout would
// instead stride by 27 floats between consecutive threads' SAME channel —
// 27 and the hardware's 32 banks share no common factor greater than 1, so
// it would not be a clean conflict either, but it would break the
// COALESCED read into 27 sequential clean passes that channel-major gives
// for free. Channel-major costs one thing: the initial write from `local[]`
// into shared memory is 27 STRIDED stores per thread (`sdata[c*blockDim.x+tid]`
// for c=0..26) rather than one contiguous 27-float store — a fair trade
// for a conflict-free REDUCTION, which runs log2(128)=7 times, versus the
// write, which runs once.
//
// Shared-memory budget and the SMALLER block size (kThreadsReduce=128, not
// the repo default 256): 27 floats * 128 threads * 4 bytes = 13.5 KiB per
// block — comfortably under Turing's 48 KiB static-shared-memory-per-block
// default, while still allowing multiple blocks resident per SM for
// latency hiding. At 256 threads the buffer would double to 27 KiB,
// leaving room for only one resident block per SM on some configurations —
// a real occupancy cost for no benefit here (this kernel does O(1) work per
// point, so it is reduction- and launch-bound, not compute-bound; more
// active blocks hides more latency). THEORY.md §GPU-mapping profiles both.
// ===========================================================================
template <IcpMode MODE>
__global__ void build_normal_system_kernel(
    const float* __restrict__ cur_xyz,       // [n_src*3] transformed source points ("x_i")
    const float* __restrict__ tgt_xyz,       // [m_tgt*3] target cloud ("q_i", via corr_idx)
    const float* __restrict__ tgt_normals,   // [m_tgt*3] target normals; UNUSED when MODE==kPointToPoint
    const int*   __restrict__ corr_idx,      // [n_src] correspondence indices, -1 = rejected
    float*       __restrict__ block_partials,// [num_blocks*kReduceWidth] OUT
    int n_src)
{
    // Channel-major shared scratch, sized for EXACTLY kThreadsReduce
    // threads (the only block size this kernel is ever launched with — see
    // launch_build_normal_system below). Using the compile-time constant
    // rather than blockDim.x keeps the array a static __shared__ (no
    // dynamic-shared-memory launch argument to keep in sync).
    __shared__ float sdata[kReduceWidth * kThreadsReduce];

    const int tid = threadIdx.x;
    const int k   = blockIdx.x * kThreadsReduce + tid;   // this thread's SOURCE point index

    // Every thread's local contribution starts at zero — this is what lets
    // BOTH the ragged tail (k >= n_src) AND rejected correspondences
    // (corr_idx[k] < 0) fall through to "contribute nothing" without an
    // early return, so every thread still reaches the shared-memory
    // reduction below in lockstep (an early return here would deadlock the
    // block at the first __syncthreads() — the same reasoning 33.01's SPD
    // check uses to justify never returning early mid-kernel).
    float local[kReduceWidth];
#pragma unroll
    for (int c = 0; c < kReduceWidth; ++c) local[c] = 0.0f;

    if (k < n_src) {
        const int j = corr_idx[k];
        if (j >= 0) {
            const float x0 = cur_xyz[k * 3 + 0], x1 = cur_xyz[k * 3 + 1], x2 = cur_xyz[k * 3 + 2];
            const float q0 = tgt_xyz[j * 3 + 0], q1 = tgt_xyz[j * 3 + 1], q2 = tgt_xyz[j * 3 + 2];
            const float r0 = x0 - q0, r1 = x1 - q1, r2 = x2 - q2;   // point-to-point residual (m)

            if constexpr (MODE == kPointToPoint) {
                // --- Linearized point-to-point (THEORY.md §the-math) -----
                // Jacobian J = [-skew(x) | I] (3x6); H = J^T J, g = J^T r.
                // Closed forms (derived once in THEORY.md, reused here):
                //   H_rot_rot   = |x|^2 * I - x x^T        (symmetric 3x3)
                //   H_rot_trans = skew(x)                  (3x3, antisymmetric)
                //   H_trans_trans = I                       (every point contributes
                //                                            EXACTLY the identity —
                //                                            a clean, checkable fact)
                //   g_rot   = x × r
                //   g_trans = r
                const float xx = x0 * x0, yy = x1 * x1, zz = x2 * x2;
                const float xy = x0 * x1, xz = x0 * x2, yz = x1 * x2;
                const float x2n = xx + yy + zz;   // |x|^2

                local[0]  = x2n - xx;   // H(0,0)
                local[1]  = -xy;        // H(0,1)
                local[2]  = -xz;        // H(0,2)
                local[3]  = 0.0f;       // H(0,3)  = skew(x)[0][0]
                local[4]  = -x2;        // H(0,4)  = skew(x)[0][1] = -x2
                local[5]  = x1;         // H(0,5)  = skew(x)[0][2] =  x1
                local[6]  = x2n - yy;   // H(1,1)
                local[7]  = -yz;        // H(1,2)
                local[8]  = x2;         // H(1,3)  = skew(x)[1][0] =  x2
                local[9]  = 0.0f;       // H(1,4)  = skew(x)[1][1]
                local[10] = -x0;        // H(1,5)  = skew(x)[1][2] = -x0
                local[11] = x2n - zz;   // H(2,2)
                local[12] = -x1;        // H(2,3)  = skew(x)[2][0] = -x1
                local[13] = x0;         // H(2,4)  = skew(x)[2][1] =  x0
                local[14] = 0.0f;       // H(2,5)  = skew(x)[2][2]
                local[15] = 1.0f;       // H(3,3) = 1
                local[16] = 0.0f;       // H(3,4) = 0
                local[17] = 0.0f;       // H(3,5) = 0
                local[18] = 1.0f;       // H(4,4) = 1
                local[19] = 0.0f;       // H(4,5) = 0
                local[20] = 1.0f;       // H(5,5) = 1
                local[21] = x1 * r2 - x2 * r1;   // g_rot.x = (x × r).x
                local[22] = x2 * r0 - x0 * r2;   // g_rot.y = (x × r).y
                local[23] = x0 * r1 - x1 * r0;   // g_rot.z = (x × r).z
                local[24] = r0;                  // g_trans.x
                local[25] = r1;                  // g_trans.y
                local[26] = r2;                  // g_trans.z
            } else {
                // --- Linearized point-to-plane (Low 2004; THEORY.md §the-math) ---
                // Scalar residual e0 = n . (x - q); Jacobian row
                // J = [x×n | n] (1x6, so H=J^T J is RANK-1 per point);
                // g = J^T e0.
                const float n0 = tgt_normals[j * 3 + 0];
                const float n1 = tgt_normals[j * 3 + 1];
                const float n2 = tgt_normals[j * 3 + 2];
                const float e0 = fmaf(n2, r2, fmaf(n1, r1, n0 * r0));   // n . r

                const float a0 = x1 * n2 - x2 * n1;   // (x × n).x
                const float a1 = x2 * n0 - x0 * n2;   // (x × n).y
                const float a2 = x0 * n1 - x1 * n0;   // (x × n).z

                local[0]  = a0 * a0;  // H(0,0)
                local[1]  = a0 * a1;  // H(0,1)
                local[2]  = a0 * a2;  // H(0,2)
                local[3]  = a0 * n0;  // H(0,3)
                local[4]  = a0 * n1;  // H(0,4)
                local[5]  = a0 * n2;  // H(0,5)
                local[6]  = a1 * a1;  // H(1,1)
                local[7]  = a1 * a2;  // H(1,2)
                local[8]  = a1 * n0;  // H(1,3)
                local[9]  = a1 * n1;  // H(1,4)
                local[10] = a1 * n2;  // H(1,5)
                local[11] = a2 * a2;  // H(2,2)
                local[12] = a2 * n0;  // H(2,3)
                local[13] = a2 * n1;  // H(2,4)
                local[14] = a2 * n2;  // H(2,5)
                local[15] = n0 * n0;  // H(3,3)
                local[16] = n0 * n1;  // H(3,4)
                local[17] = n0 * n2;  // H(3,5)
                local[18] = n1 * n1;  // H(4,4)
                local[19] = n1 * n2;  // H(4,5)
                local[20] = n2 * n2;  // H(5,5)
                local[21] = a0 * e0;  // g.0
                local[22] = a1 * e0;  // g.1
                local[23] = a2 * e0;  // g.2
                local[24] = n0 * e0;  // g.3
                local[25] = n1 * e0;  // g.4
                local[26] = n2 * e0;  // g.5
            }
        }
    }

    // --- Stage 1a: scatter local[] into channel-major shared memory -------
#pragma unroll
    for (int c = 0; c < kReduceWidth; ++c) sdata[c * kThreadsReduce + tid] = local[c];
    __syncthreads();   // the whole block's contributions must be visible
                       // before the tree reduction below reads a neighbor's

    // --- Stage 1b: binary-tree reduction, all 27 channels together --------
    // Classic halving reduction: at each step, the first `s` threads add
    // the partial sum `s` slots to their right, then s halves. log2(128)=7
    // steps total. Every step is bank-conflict-free (see header comment).
#pragma unroll
    for (int s = kThreadsReduce / 2; s > 0; s >>= 1) {
        if (tid < s) {
#pragma unroll
            for (int c = 0; c < kReduceWidth; ++c)
                sdata[c * kThreadsReduce + tid] += sdata[c * kThreadsReduce + tid + s];
        }
        __syncthreads();   // every step must fully finish before the next halving reads it
    }

    // Thread 0 holds the block's total in sdata[c*kThreadsReduce + 0] for
    // every channel c — write the block's 27-scalar partial to global
    // memory. main.cu (stage 2) sums these `num_blocks` rows on the host.
    if (tid == 0) {
#pragma unroll
        for (int c = 0; c < kReduceWidth; ++c)
            block_partials[blockIdx.x * kReduceWidth + c] = sdata[c * kThreadsReduce];
    }
}

// ===========================================================================
// Host launchers (declared in kernels.cuh).
// ===========================================================================

void launch_transform_cloud(int n, const float* d_src_xyz, Rigid3 T, float* d_out_xyz)
{
    if (n <= 0) return;                          // empty cloud: valid no-op
    const int blocks = blocks_for(n, kThreadsPerBlock);
    transform_cloud_kernel<<<blocks, kThreadsPerBlock>>>(d_src_xyz, T, d_out_xyz, n);
    CUDA_CHECK_LAST_ERROR("transform_cloud_kernel launch");
}

void launch_find_correspondences(int n_src, const float* d_cur_xyz,
                                 int m_tgt, const float* d_tgt_xyz,
                                 float max_dist_m,
                                 int* d_corr_idx, float* d_corr_dist2)
{
    if (n_src <= 0) return;
    const int blocks = blocks_for(n_src, kThreadsPerBlock);
    // Square the gate ONCE on the host rather than m_tgt times per thread —
    // a trivial saving, but the kind of "don't repeat cheap work 30000x
    // per launch" habit that adds up across a whole ICP run.
    const float max_dist2 = max_dist_m * max_dist_m;
    find_correspondences_kernel<<<blocks, kThreadsPerBlock>>>(
        d_cur_xyz, n_src, d_tgt_xyz, m_tgt, max_dist2, d_corr_idx, d_corr_dist2);
    CUDA_CHECK_LAST_ERROR("find_correspondences_kernel launch");
}

void launch_estimate_normals(int m_tgt, const float* d_tgt_xyz,
                             const float ref_point[3], float* d_tgt_normals)
{
    if (m_tgt <= 0) return;
    const int blocks = blocks_for(m_tgt, kThreadsPerBlock);
    estimate_normals_kernel<<<blocks, kThreadsPerBlock>>>(
        d_tgt_xyz, m_tgt, ref_point[0], ref_point[1], ref_point[2], d_tgt_normals);
    CUDA_CHECK_LAST_ERROR("estimate_normals_kernel launch");
}

void launch_build_normal_system(int n_src, const float* d_cur_xyz,
                                const float* d_tgt_xyz,
                                const float* d_tgt_normals,
                                const int* d_corr_idx,
                                IcpMode mode,
                                float* d_block_partials)
{
    if (n_src <= 0) return;
    const int blocks = blocks_for(n_src, kThreadsReduce);

    // Runtime mode -> compile-time template dispatch, exhaustive-or-abort —
    // the same fail-fast policy 33.01 uses for its n=3/4/6 switch
    // (CLAUDE.md §13: no silent fallback to an unintended code path).
    switch (mode) {
    case kPointToPoint:
        build_normal_system_kernel<kPointToPoint><<<blocks, kThreadsReduce>>>(
            d_cur_xyz, d_tgt_xyz, d_tgt_normals, d_corr_idx, d_block_partials, n_src);
        CUDA_CHECK_LAST_ERROR("build_normal_system_kernel<kPointToPoint> launch");
        break;
    case kPointToPlane:
        build_normal_system_kernel<kPointToPlane><<<blocks, kThreadsReduce>>>(
            d_cur_xyz, d_tgt_xyz, d_tgt_normals, d_corr_idx, d_block_partials, n_src);
        CUDA_CHECK_LAST_ERROR("build_normal_system_kernel<kPointToPlane> launch");
        break;
    default:
        std::fprintf(stderr, "launch_build_normal_system: invalid IcpMode %d\n", static_cast<int>(mode));
        std::exit(EXIT_FAILURE);
    }
}
