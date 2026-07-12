// ===========================================================================
// kernels.cu — GPU kernels for project 02.07
//              NDT scan matching (Autoware-style map localizer)
//
// Two GPU parallelism regimes, one per algorithmic stage (THEORY.md "The
// GPU mapping" argues both in depth):
//
//   VOXEL BUILD (4 kernels below) — POINT-parallel accumulation into a
//     DENSE voxel grid via atomicAdd (02.01's Method-A atomic-hash idiom,
//     cited, but into direct array slots instead of a hash table — see
//     kernels.cuh's "dense, not hashed" justification), sandwiched around
//     two VOXEL-parallel finalize passes. Two-pass mean-then-covariance
//     (not Welford, not the naive one-pass raw-second-moment trick) —
//     THEORY.md "numerical considerations" explains the choice.
//
//   ASSEMBLY (ndt_assemble_kernel) — POINT-parallel scoring of every scan
//     point against its voxel's Gaussian, block-tree-reduced into a
//     28-scalar [H21|g6|score] record per block — 01.17's EXACT reduction
//     shape and reasoning (cited): tree reduction over shared memory beats
//     28-way atomic contention, and keeps the GPU path's OWN rounding
//     bit-reproducible within a block (only the host's cross-block double
//     sum is intentionally unordered).
//
// Read this after: main.cu, kernels.cuh.  Read this before: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"

// ===========================================================================
// STAGE 1 — voxel build: PASS 1 (sums) -> finalize means -> PASS 2 (cov) ->
// finalize cov (regularize + invert).
// ===========================================================================

// ndt_voxel_accum_sum_kernel — thread i owns map point i. atomicAdd its xyz
// (cast to DOUBLE) and a +1 count into its voxel's running sum.
//
// Why atomicAdd on DOUBLE, not float (02.01's Method-A uses float sums)?
// A voxel here can accumulate THOUSANDS of map points (the map is a dense
// survey scan, not a single LiDAR sweep); repeated float atomicAdd on a
// running sum of that many similarly-sized terms accumulates rounding error
// that grows with sqrt(count) in the worst case — visible at the covariance
// stage, which SUBTRACTS large numbers (THEORY.md "numerical
// considerations" does the arithmetic). atomicAdd(double*, double) has been
// a native SASS instruction since compute capability 6.0 (Pascal) — sm_75
// and above pay no emulation penalty for it, so there is no performance
// reason to stay in float here (unlike the ASSEMBLY kernel below, whose
// per-point cost is dominated by transcendental exp() where float IS the
// right, measured trade — see that kernel's own comment).
//
// Thread mapping: i = blockIdx.x*blockDim.x+threadIdx.x, guarded i<n_map.
// Memory: map_xyz read is coalesced (consecutive points, consecutive
// threads); the atomicAdd targets are scattered across the (small) voxel
// array — contention is real but the accumulation only runs ONCE, at map-
// build time, not in the per-iteration optimizer hot loop below.
__global__ void ndt_voxel_accum_sum_kernel(int n_map, const float* __restrict__ map_xyz, NdtGridGPU grid)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_map) return;

    const float p[3] = { map_xyz[i * 3 + 0], map_xyz[i * 3 + 1], map_xyz[i * 3 + 2] };
    const int vidx = voxel_index(p, grid.origin_x, grid.origin_y, grid.origin_z,
                                 grid.leaf, grid.nx, grid.ny, grid.nz);
    if (vidx < 0) return;   // outside the map's bounding box — dropped, not clamped (honest data loss)

    atomicAdd(&grid.count[vidx], 1);
    atomicAdd(&grid.sum_xyz[vidx * 3 + 0], static_cast<double>(p[0]));
    atomicAdd(&grid.sum_xyz[vidx * 3 + 1], static_cast<double>(p[1]));
    atomicAdd(&grid.sum_xyz[vidx * 3 + 2], static_cast<double>(p[2]));
}

// ndt_finalize_means_kernel — thread v owns voxel v (one thread per voxel,
// NOT per point — the grid is tiny, hundreds to low thousands of voxels, so
// this kernel is a rounding error next to the point-parallel passes either
// side of it). mean = sum/count; empty voxels get an explicit 0 (never read
// downstream because `valid` stays false for them, but writing a defined
// value keeps every array element initialized — no uninitialized-read UB).
__global__ void ndt_finalize_means_kernel(NdtGridGPU grid)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    const int capacity = grid.nx * grid.ny * grid.nz;
    if (v >= capacity) return;

    const int c = grid.count[v];
    if (c > 0) {
        grid.mean[v * 3 + 0] = static_cast<float>(grid.sum_xyz[v * 3 + 0] / c);
        grid.mean[v * 3 + 1] = static_cast<float>(grid.sum_xyz[v * 3 + 1] / c);
        grid.mean[v * 3 + 2] = static_cast<float>(grid.sum_xyz[v * 3 + 2] / c);
    } else {
        grid.mean[v * 3 + 0] = grid.mean[v * 3 + 1] = grid.mean[v * 3 + 2] = 0.0f;
    }
}

// ndt_voxel_accum_cov_kernel — PASS 2, requires ndt_finalize_means_kernel to
// have already run (reads grid.mean). Thread i owns map point i AGAIN (the
// "two-pass" in "two-pass covariance": every point is visited twice, once
// per pass, THEORY.md's numerics tradeoff). atomicAdd the CENTERED outer
// product (p-mean)(p-mean)^T's six unique entries, in double, same
// reasoning as PASS 1.
__global__ void ndt_voxel_accum_cov_kernel(int n_map, const float* __restrict__ map_xyz, NdtGridGPU grid)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_map) return;

    const float p[3] = { map_xyz[i * 3 + 0], map_xyz[i * 3 + 1], map_xyz[i * 3 + 2] };
    const int vidx = voxel_index(p, grid.origin_x, grid.origin_y, grid.origin_z,
                                 grid.leaf, grid.nx, grid.ny, grid.nz);
    if (vidx < 0) return;

    const float* mu = &grid.mean[vidx * 3];
    const double dx = static_cast<double>(p[0]) - mu[0];
    const double dy = static_cast<double>(p[1]) - mu[1];
    const double dz = static_cast<double>(p[2]) - mu[2];

    // Packed [xx,xy,xz,yy,yz,zz] — kernels.cuh's shared convention, read
    // back the same way by regularize_and_invert_cov3.
    atomicAdd(&grid.sum_cov6[vidx * 6 + 0], dx * dx);
    atomicAdd(&grid.sum_cov6[vidx * 6 + 1], dx * dy);
    atomicAdd(&grid.sum_cov6[vidx * 6 + 2], dx * dz);
    atomicAdd(&grid.sum_cov6[vidx * 6 + 3], dy * dy);
    atomicAdd(&grid.sum_cov6[vidx * 6 + 4], dy * dz);
    atomicAdd(&grid.sum_cov6[vidx * 6 + 5], dz * dz);
}

// ndt_finalize_cov_kernel — voxel-parallel: turn the raw double covariance
// sum into a REGULARIZED, INVERTED, float voxel Gaussian. Voxels below
// kMinPointsPerVoxel are marked invalid (the NDT analogue of ICP's rejected
// correspondence, kernels.cuh's file header) — their statistics would be
// dominated by sampling noise, not the true local surface shape.
__global__ void ndt_finalize_cov_kernel(NdtGridGPU grid, unsigned int* __restrict__ d_regularized_count)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    const int capacity = grid.nx * grid.ny * grid.nz;
    if (v >= capacity) return;

    const int c = grid.count[v];
    if (c < kMinPointsPerVoxel) {
        grid.valid[v] = 0;
        for (int k = 0; k < 6; ++k) grid.inv_cov6[v * 6 + k] = 0.0f;
        return;
    }

    // Unbiased sample covariance: divide by (count-1), not count — the
    // standard correction (a single point would otherwise claim zero
    // variance, which is meaningless, not just unlucky).
    float cov[6];
    const double denom = static_cast<double>(c - 1);
    for (int k = 0; k < 6; ++k) cov[k] = static_cast<float>(grid.sum_cov6[v * 6 + k] / denom);

    float inv_cov[6];
    const bool regularized = regularize_and_invert_cov3(cov, inv_cov);
    for (int k = 0; k < 6; ++k) grid.inv_cov6[v * 6 + k] = inv_cov[k];
    grid.valid[v] = 1;

    if (regularized) atomicAdd(d_regularized_count, 1u);   // the [info] honesty count (main.cu)
}

// ---------------------------------------------------------------------------
// launch_build_ndt_grid — the four kernels above, in the ONLY order that is
// correct (means must finalize before PASS 2 can center on them). Zeros the
// grid's accumulator arrays itself so a caller building the SAME device
// buffers at a second resolution (or re-running the twin gate) never
// silently reuses stale accumulator state.
// ---------------------------------------------------------------------------
void launch_build_ndt_grid(int n_map, const float* d_map_xyz, NdtGridGPU grid,
                           unsigned int* out_regularized_count)
{
    const int capacity = ndt_grid_capacity(grid);
    CUDA_CHECK(cudaMemset(grid.count, 0, static_cast<size_t>(capacity) * sizeof(int)));
    CUDA_CHECK(cudaMemset(grid.sum_xyz, 0, static_cast<size_t>(capacity) * 3 * sizeof(double)));
    CUDA_CHECK(cudaMemset(grid.sum_cov6, 0, static_cast<size_t>(capacity) * 6 * sizeof(double)));
    CUDA_CHECK(cudaMemset(grid.valid, 0, static_cast<size_t>(capacity) * sizeof(unsigned char)));

    unsigned int* d_reg_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_reg_count, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_reg_count, 0, sizeof(unsigned int)));

    const int blockP = kThreadsVoxel;
    const int gridP = blocks_for(n_map, blockP);
    const int blockV = kThreadsVoxel;
    const int gridV = blocks_for(capacity, blockV);

    ndt_voxel_accum_sum_kernel<<<gridP, blockP>>>(n_map, d_map_xyz, grid);
    CUDA_CHECK_LAST_ERROR("ndt_voxel_accum_sum_kernel launch");

    ndt_finalize_means_kernel<<<gridV, blockV>>>(grid);
    CUDA_CHECK_LAST_ERROR("ndt_finalize_means_kernel launch");

    ndt_voxel_accum_cov_kernel<<<gridP, blockP>>>(n_map, d_map_xyz, grid);
    CUDA_CHECK_LAST_ERROR("ndt_voxel_accum_cov_kernel launch");

    ndt_finalize_cov_kernel<<<gridV, blockV>>>(grid, d_reg_count);
    CUDA_CHECK_LAST_ERROR("ndt_finalize_cov_kernel launch");

    CUDA_CHECK(cudaMemcpy(out_regularized_count, d_reg_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_reg_count));
}

// ===========================================================================
// STAGE 2 — ndt_assemble_kernel: the project's central NEW GPU concept.
//
// Per-point math (THEORY.md "The math" derives every line; kernels.cuh's
// file header names the mixture-model score this implements):
//   y = R*x + t                          (transform scan point to map frame)
//   vidx = voxel_index(y)                (O(1) direct lookup — no search)
//   q = y - mean(vidx)                   (displacement from voxel's Gaussian)
//   m = q^T * invC * q                   (Mahalanobis distance^2)
//   f = exp(-d2/2 * m)                   (the Gaussian's shape factor)
//   score += -d1 * f
//   J (3x6) = [ -[R*x]_x | I_3 ]         (chain rule through R*x+t, 01.17's
//                                          exact rotation-Jacobian formula,
//                                          cited, reused for a 3-vector
//                                          residual instead of a 2-vector
//                                          reprojection)
//   b = invC * q
//   g   += d1*d2*f * J^T b
//   H   += d1*d2*f * ( J^T*invC*J  -  d2 * (J^T b)(J^T b)^T )     (Gauss-
//         Newton term MINUS the Gaussian's own curvature correction — the
//         term that can make H INDEFINITE far from the optimum, THEORY.md
//         "numerical considerations" and kernels.cuh's cholesky6_solve_flat
//         comment both discuss the consequence: flat Levenberg damping,
//         not Marquardt diag-scaling.)
//
// Thread-to-data mapping: thread i = blockIdx.x*blockDim.x+threadIdx.x owns
// scan point i, guarded i<n_scan. A point whose transformed position lands
// outside the grid OR in an invalid (too-sparse) voxel contributes an
// all-zero 28-scalar row — the NDT correspondence-rejection analogue of
// 02.06's -1 corr_idx / 01.17's residual gating.
//
// Memory hierarchy — IDENTICAL shape to 01.17's assemble_normal_equations_
// kernel (cited): GLOBAL reads of scan_xyz (coalesced, one point per
// thread) and the voxel grid's mean/inv_cov6/valid arrays (a data-dependent
// gather — different threads in a warp can land in different voxels, so
// this is NOT a broadcast read like 08.01's u_nom; THEORY.md "The GPU
// mapping" measures the resulting cache behavior); SHARED memory holds
// every thread's 28-float row for the tree reduction; block_partials gets
// one row per block, written by thread 0.
//
// Why FLOAT per-point math here, when the voxel-build kernels above chose
// DOUBLE atomics? This kernel's per-point cost is dominated by expf() (a
// transcendental, evaluated n_scan times PER NEWTON ITERATION — the hot
// loop the whole project's runtime lives in), not by a long-running
// summation of many same-sign terms (the voxel builder's actual FP32 risk).
// float keeps the transcendental cheap; the host still finishes the
// CROSS-BLOCK sum in double (the same "float per-point, double reduction"
// split 01.17/02.06 use), so accumulated rounding across MANY points is
// still controlled where it actually matters.
// ===========================================================================
__global__ void ndt_assemble_kernel(const float* __restrict__ scan_xyz, int n_scan,
                                    Rigid3 T, NdtGridGPU grid,
                                    double d1, double d2,
                                    float* __restrict__ block_partials)
{
    extern __shared__ float sdata[];   // blockDim.x * kReduceWidth floats

    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    float local[kReduceWidth];
    for (int k = 0; k < kReduceWidth; ++k) local[k] = 0.0f;

    if (i < n_scan) {
        const float x[3] = { scan_xyz[i * 3 + 0], scan_xyz[i * 3 + 1], scan_xyz[i * 3 + 2] };

        float RP[3];
        mat3_vec(T.R, x, RP);                                   // R * x (rotated-only point)
        const float y[3] = { RP[0] + T.t[0], RP[1] + T.t[1], RP[2] + T.t[2] };

        const int vidx = voxel_index(y, grid.origin_x, grid.origin_y, grid.origin_z,
                                     grid.leaf, grid.nx, grid.ny, grid.nz);

        if (vidx >= 0 && grid.valid[vidx]) {
            const float* mu    = &grid.mean[vidx * 3];
            const float* invC  = &grid.inv_cov6[vidx * 6];
            const float q[3] = { y[0] - mu[0], y[1] - mu[1], y[2] - mu[2] };

            const float d1f = static_cast<float>(d1);
            const float d2f = static_cast<float>(d2);

            const float m = sym3_quad_form(invC, q);
            const float f = expf(-0.5f * d2f * m);

            float b[3];
            sym3_vec(invC, q, b);   // b = invC * q

            // J (3x6, row-major): cols 0..2 = -[RP]_x, cols 3..5 = I_3
            // (01.17's exact rotation-Jacobian formula, cited — see this
            // kernel's own header comment for the chain-rule statement).
            float S[9];
            skew3(RP, S);
            float J[18] = { 0.0f };
            for (int r = 0; r < 3; ++r) {
                J[r * 6 + 0] = -S[r * 3 + 0];
                J[r * 6 + 1] = -S[r * 3 + 1];
                J[r * 6 + 2] = -S[r * 3 + 2];
                J[r * 6 + 3 + r] = 1.0f;
            }

            // Jt_b[k] = sum_r J[r][k] * b[r]  (== J^T b, a 6-vector).
            float Jtb[6];
            for (int k = 0; k < 6; ++k) {
                float acc = 0.0f;
                for (int r = 0; r < 3; ++r) acc += J[r * 6 + k] * b[r];
                Jtb[k] = acc;
            }

            // JtCJ[k][l] = J[:,k] . (invC * J[:,l])  — upper triangle only.
            for (int l = 0; l < 6; ++l) {
                const float Jcol_l[3] = { J[0 * 6 + l], J[1 * 6 + l], J[2 * 6 + l] };
                float cJ_l[3];
                sym3_vec(invC, Jcol_l, cJ_l);
                for (int k = 0; k <= l; ++k) {
                    const float Jcol_k[3] = { J[0 * 6 + k], J[1 * 6 + k], J[2 * 6 + k] };
                    const float dot = Jcol_k[0] * cJ_l[0] + Jcol_k[1] * cJ_l[1] + Jcol_k[2] * cJ_l[2];
                    // Gauss-Newton term MINUS the Gaussian-curvature correction (see
                    // this kernel's file header — the term that can make H indefinite).
                    local[hidx(k, l)] = d1f * d2f * f * (dot - d2f * Jtb[k] * Jtb[l]);
                }
            }
            for (int k = 0; k < 6; ++k) local[21 + k] = d1f * d2f * f * Jtb[k];
            local[27] = -d1f * f;
        }
        // vidx invalid/out-of-grid: local[] stays all-zero (rejected point).
    }
    // i>=n_scan: local[] stays all-zero (ragged-tail padding for the tree
    // reduction below — identical idiom to 01.17/the template's SAXPY).

    float* my_row = &sdata[threadIdx.x * kReduceWidth];
    for (int k = 0; k < kReduceWidth; ++k) my_row[k] = local[k];
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float* a = &sdata[threadIdx.x * kReduceWidth];
            float* b = &sdata[(threadIdx.x + stride) * kReduceWidth];
            for (int k = 0; k < kReduceWidth; ++k) a[k] += b[k];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float* out_row = &block_partials[blockIdx.x * kReduceWidth];
        for (int k = 0; k < kReduceWidth; ++k) out_row[k] = sdata[k];
    }
}

// ---------------------------------------------------------------------------
// launch_ndt_assemble — grid/block math + dynamic shared memory, 01.17's
// exact launch-configuration reasoning (cited): kThreadsAssemble (128) is a
// power of two (required by the stride-halving tree reduction) and a warp
// multiple; grid is EXACTLY blocks_for(n_scan, kThreadsAssemble) — one
// thread per scan point, no grid-stride loop, because each thread's row is
// tied to its block-local threadIdx.x.
// ---------------------------------------------------------------------------
int launch_ndt_assemble(const float* d_scan_xyz, int n_scan, Rigid3 T, NdtGridGPU grid,
                        double d1, double d2, float* d_block_partials)
{
    const int block = kThreadsAssemble;
    const int grid_dim = blocks_for(n_scan, block);
    const size_t shmem_bytes = static_cast<size_t>(block) * kReduceWidth * sizeof(float);

    ndt_assemble_kernel<<<grid_dim, block, shmem_bytes>>>(d_scan_xyz, n_scan, T, grid, d1, d2, d_block_partials);
    CUDA_CHECK_LAST_ERROR("ndt_assemble_kernel launch");
    return grid_dim;
}
