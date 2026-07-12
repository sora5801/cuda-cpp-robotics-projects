// ===========================================================================
// kernels.cu — GPU kernels for project 02.16 (Multi-LiDAR merging +
//              extrinsic refinement)
//
// Four GPU stages, four different points on the "how do many threads
// cooperate" spectrum (THEORY.md "The GPU mapping" argues the full story):
//
//   1) transform_points(_multi) — a pure MAP. Every output point is
//      independent; this is the "merging is easy" half of the project,
//      deliberately as simple as the template's own SAXPY (main.cu's
//      TRANSFORM_TWIN gate exercises it, but the REAL teaching content of
//      "merging" lives in stage 2 and the dedup pipeline below, not here).
//
//   2) accumulate_centroid_kernel / accumulate_covariance_kernel — a MAP
//      into a SCATTER-REDUCE via atomics. Unlike stage 4's per-iteration,
//      heavily-contended 28-scalar reduction (where 01.17/02.06 chose a
//      shared-memory tree specifically to AVOID atomic contention), this
//      is a ONE-SHOT reduction into just kNumSurfaces=6 tiny slots: the
//      contention is bounded (at most 6 "hot" addresses, thousands of
//      threads spread across them) and the whole pass runs once per
//      sensor-cohort, not once per LM iteration — atomics' simplicity wins
//      over a tree reduction's bookkeeping at this scale. Teaching contrast
//      with stage 4, worth reading side by side.
//
//   3) assemble_point_to_plane_kernel — the refinement's central NEW
//      concept: 01.17/02.06's block-tree-reduce-then-host-finishes split
//      (cited), applied to a 1-ROW (scalar point-to-plane) Jacobian instead
//      of 01.17's 2-row (pixel reprojection) or 02.06's 1-row-but-per-
//      correspondence-varying-target Jacobian. Correspondence here is FIXED
//      by zone membership (this point's surface_id), not searched for each
//      iteration — see the kernel's own header for why that is an honest,
//      documented simplification of 02.06's nearest-neighbor search.
//
//   4) the dedup pipeline (compute_hash_keys / mark_boundaries /
//      gather_representatives, glued together by Thrust's sort_by_key in
//      launch_dedup_voxel_grid) — 02.01/02.09's sort-then-compact spatial-
//      index pattern (cited), applied here not to build a KNN index but to
//      collapse near-duplicate points into one representative per voxel.
//
// Read this after: main.cu, kernels.cuh.  Read this before: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

// is_nonzero_flag — a named functor (NOT a lambda) for thrust::copy_if's
// predicate below. A plain device lambda would need the --extended-lambda
// nvcc flag this project's .vcxproj does not set (a deliberate simplicity
// choice, kernels.cuh's file header style); a named __host__ __device__
// functor needs no such flag and is the idiom Thrust's own examples use.
struct is_nonzero_flag {
    __host__ __device__ bool operator()(int32_t x) const { return x != 0; }
};

// ===========================================================================
// Stage 1 — transform_points / transform_points_multi (pure map).
// ===========================================================================

// transform_points_kernel — out[k] = T.R * src[k] + T.t. One thread per
// point, grid-stride NOT needed (n here never exceeds a few tens of
// thousands — the template's grid-stride caution is for million-point
// clouds; a straight one-thread-one-point launch with a bounds check is
// simpler to read and, at this project's scale, exactly as fast).
__global__ void transform_points_kernel(int n, const float* __restrict__ src_xyz, Rigid3 T, float* __restrict__ out_xyz)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float p[3] = { src_xyz[i * 3 + 0], src_xyz[i * 3 + 1], src_xyz[i * 3 + 2] };
    float Rp[3];
    mat3_vec(T.R, p, Rp);
    out_xyz[i * 3 + 0] = Rp[0] + T.t[0];
    out_xyz[i * 3 + 1] = Rp[1] + T.t[1];
    out_xyz[i * 3 + 2] = Rp[2] + T.t[2];
}

void launch_transform_points(int n, const float* d_src_xyz, Rigid3 T, float* d_out_xyz)
{
    const int block = kThreadsPerBlock;
    const int grid = blocks_for(n, block);
    transform_points_kernel<<<grid, block>>>(n, d_src_xyz, T, d_out_xyz);
    CUDA_CHECK_LAST_ERROR("transform_points_kernel launch");
}

// transform_points_multi_kernel — the actual MERGE step: point i picks its
// OWN transform via T_per_sensor[sensor_id[i]] (a coalesced read of a
// kNumSensors=3-entry table broadcast to every thread — negligible traffic
// next to the point read/write itself).
__global__ void transform_points_multi_kernel(int n, const float* __restrict__ src_xyz,
                                              const int32_t* __restrict__ sensor_id,
                                              const Rigid3* __restrict__ T_per_sensor,
                                              float* __restrict__ out_xyz)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Rigid3 T = T_per_sensor[sensor_id[i]];
    const float p[3] = { src_xyz[i * 3 + 0], src_xyz[i * 3 + 1], src_xyz[i * 3 + 2] };
    float Rp[3];
    mat3_vec(T.R, p, Rp);
    out_xyz[i * 3 + 0] = Rp[0] + T.t[0];
    out_xyz[i * 3 + 1] = Rp[1] + T.t[1];
    out_xyz[i * 3 + 2] = Rp[2] + T.t[2];
}

void launch_transform_points_multi(int n, const float* d_src_xyz, const int32_t* d_sensor_id,
                                   const Rigid3* d_T_per_sensor, float* d_out_xyz)
{
    const int block = kThreadsPerBlock;
    const int grid = blocks_for(n, block);
    transform_points_multi_kernel<<<grid, block>>>(n, d_src_xyz, d_sensor_id, d_T_per_sensor, d_out_xyz);
    CUDA_CHECK_LAST_ERROR("transform_points_multi_kernel launch");
}

// ===========================================================================
// Stage 2 — plane fitting: centroid pass, then mean-shifted covariance pass
// (02.09's two-pass precision lesson, cited — see kernels.cuh's Plane doc).
// Both kernels use atomicAdd into just kNumSurfaces=6 slots; see the file
// header for why that is the right tool here (contrast with stage 3).
// ===========================================================================

__global__ void accumulate_centroid_kernel(int n, const float* __restrict__ xyz,
                                           const int32_t* __restrict__ surface_id,
                                           float* __restrict__ sums, int32_t* __restrict__ counts)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int32_t s = surface_id[i];
    if (s < 0 || s >= kNumSurfaces) return;   // defensive: a malformed tag never corrupts another surface's slot
    atomicAdd(&sums[s * 3 + 0], xyz[i * 3 + 0]);
    atomicAdd(&sums[s * 3 + 1], xyz[i * 3 + 1]);
    atomicAdd(&sums[s * 3 + 2], xyz[i * 3 + 2]);
    atomicAdd(&counts[s], 1);
}

void launch_accumulate_centroid(int n, const float* d_xyz, const int32_t* d_surface_id,
                                float* d_sums, int32_t* d_counts)
{
    CUDA_CHECK(cudaMemset(d_sums, 0, sizeof(float) * kNumSurfaces * 3));
    CUDA_CHECK(cudaMemset(d_counts, 0, sizeof(int32_t) * kNumSurfaces));
    const int block = kThreadsPerBlock;
    const int grid = blocks_for(n, block);
    accumulate_centroid_kernel<<<grid, block>>>(n, d_xyz, d_surface_id, d_sums, d_counts);
    CUDA_CHECK_LAST_ERROR("accumulate_centroid_kernel launch");
}

// accumulate_covariance_kernel — the SECOND pass: needs each surface's
// CENTROID already computed (host divides pass 1's sums by counts and
// re-uploads — see main.cu's fit_planes_gpu). Mean-shifting BEFORE
// squaring is what avoids the textbook one-pass formula's catastrophic
// cancellation (02.09's exact numerical argument, cited; THEORY.md
// restates it for this project's own point magnitudes).
__global__ void accumulate_covariance_kernel(int n, const float* __restrict__ xyz,
                                             const int32_t* __restrict__ surface_id,
                                             const float* __restrict__ centroids,
                                             float* __restrict__ cov_sums)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int32_t s = surface_id[i];
    if (s < 0 || s >= kNumSurfaces) return;
    const float dx = xyz[i * 3 + 0] - centroids[s * 3 + 0];
    const float dy = xyz[i * 3 + 1] - centroids[s * 3 + 1];
    const float dz = xyz[i * 3 + 2] - centroids[s * 3 + 2];
    // Upper triangle (c00,c01,c02,c11,c12,c22) — kernels.cuh's Plane doc.
    atomicAdd(&cov_sums[s * 6 + 0], dx * dx);
    atomicAdd(&cov_sums[s * 6 + 1], dx * dy);
    atomicAdd(&cov_sums[s * 6 + 2], dx * dz);
    atomicAdd(&cov_sums[s * 6 + 3], dy * dy);
    atomicAdd(&cov_sums[s * 6 + 4], dy * dz);
    atomicAdd(&cov_sums[s * 6 + 5], dz * dz);
}

void launch_accumulate_covariance(int n, const float* d_xyz, const int32_t* d_surface_id,
                                  const float* d_centroids, float* d_cov_sums)
{
    CUDA_CHECK(cudaMemset(d_cov_sums, 0, sizeof(float) * kNumSurfaces * 6));
    const int block = kThreadsPerBlock;
    const int grid = blocks_for(n, block);
    accumulate_covariance_kernel<<<grid, block>>>(n, d_xyz, d_surface_id, d_centroids, d_cov_sums);
    CUDA_CHECK_LAST_ERROR("accumulate_covariance_kernel launch");
}

// ===========================================================================
// Stage 3 — assemble_point_to_plane_kernel: the refinement LM assembly.
// ===========================================================================

// Thread-to-data mapping: thread i owns SOURCE point i (i < n), exactly
// 01.17's assembly-kernel shape. Correspondence is NOT searched (unlike
// 02.06's nearest-neighbor ICP): point i's target is target_planes[surface_id[i]],
// a FIXED lookup, because this project's "correspondence" is decided once
// at data-generation time by which physical surface a point lies on — an
// honest, documented simplification of 02.06's per-iteration search (see
// kernels.cuh's file header "ZONE SETS" and THEORY.md "The algorithm" for
// why zone-membership correspondence is legitimate here: the alternative,
// nearest-neighbor matching THIS project's own fitted planes, would just
// rediscover the same zone assignment every iteration since the scene is
// planar and the points do not move between planes as T is refined).
//
// A point contributes to the reduction only if (a) its surface_id's target
// plane is valid (enough points were available to fit it) AND (b) the
// caller's zone_mask has that surface's bit set (kernels.cuh's ZONE SETS —
// this is how the SAME kernel serves the full-observability solve and the
// single-zone degenerate solve without duplicating any code).
//
// Memory hierarchy: identical to 01.17's assembly kernel (GLOBAL: p_src/
// surface_id read-only, coalesced by point index; SHARED: sdata[blockDim.x
// * kReduceWidth] floats, 128*28*4 = 14336 bytes at kThreadsReduce=128,
// comfortably under the 48 KiB default budget; REGISTERS: each thread's
// local[28] accumulator). See that project's kernels.cu for the full
// "why a tree reduction, not atomics" argument — contrast with stage 2
// above, which made the OPPOSITE choice for the OPPOSITE reason (few slots,
// one-shot vs. many slots... here it is the reverse: ONE 28-wide slot,
// hammered every LM iteration by potentially thousands of points — exactly
// the high-contention regime atomics handle badly and a tree reduction
// handles well).
// ---------------------------------------------------------------------------
__global__ void assemble_point_to_plane_kernel(
    const float* __restrict__ p_src, const int32_t* __restrict__ surface_id, int n,
    Rigid3 T, const Plane* __restrict__ target_planes, uint32_t zone_mask,
    float* __restrict__ block_partials)
{
    extern __shared__ float sdata[];   // blockDim.x * kReduceWidth floats

    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    float local[kReduceWidth];
    for (int k = 0; k < kReduceWidth; ++k) local[k] = 0.0f;

    if (i < n) {
        const int32_t s = surface_id[i];
        const bool zone_active = (s >= 0 && s < kNumSurfaces) && ((zone_mask >> s) & 1u);
        if (zone_active && target_planes[s].valid) {
            const float p[3] = { p_src[i * 3 + 0], p_src[i * 3 + 1], p_src[i * 3 + 2] };
            float r, J[6];
            point_to_plane_residual_and_jacobian(T, p, target_planes[s].normal, target_planes[s].centroid, r, J);

            // H = J^T J, upper triangle (J is 1x6 here, unlike 01.17's 2x6 —
            // the same hidx() packing works for any row count: H[a][b] is
            // always sum-over-rows-of J[row][a]*J[row][b], and there is
            // exactly one row).
            for (int a = 0; a < 6; ++a)
                for (int b = a; b < 6; ++b)
                    local[hidx(a, b)] = J[a] * J[b];

            for (int a = 0; a < 6; ++a) local[21 + a] = J[a] * r;
            local[27] = r * r;
        }
        // inactive/invalid points fall through with an all-zero local[] —
        // the same "ragged tail padding" idiom 01.17 uses for i>=n.
    }

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

int launch_assemble_point_to_plane(const float* d_p_src, const int32_t* d_surface_id, int n,
                                   Rigid3 T, const Plane* d_target_planes, uint32_t zone_mask,
                                   float* d_block_partials)
{
    const int block = kThreadsReduce;
    const int grid = blocks_for(n, block);
    const size_t shmem_bytes = static_cast<size_t>(block) * kReduceWidth * sizeof(float);

    assemble_point_to_plane_kernel<<<grid, block, shmem_bytes>>>(
        d_p_src, d_surface_id, n, T, d_target_planes, zone_mask, d_block_partials);
    CUDA_CHECK_LAST_ERROR("assemble_point_to_plane_kernel launch");
    return grid;
}

// ===========================================================================
// Stage 4 — the dedup pipeline: hash -> sort -> mark boundaries -> compact.
// ===========================================================================

// compute_hash_keys_kernel — one thread per point: this point's voxel key
// at cell size `cell` (kernels.cuh's pack_voxel_key/voxel_coord, cited).
__global__ void compute_hash_keys_kernel(int n, const float* __restrict__ xyz, float cell,
                                         unsigned long long* __restrict__ keys)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float p[3] = { xyz[i * 3 + 0], xyz[i * 3 + 1], xyz[i * 3 + 2] };
    keys[i] = point_voxel_key(p, cell);
}

// mark_boundaries_kernel — position 0, or any position whose SORTED key
// differs from its predecessor, starts a new voxel's run (02.01/02.09's
// identical boundary-scan idiom, cited).
__global__ void mark_boundaries_kernel(int n, const unsigned long long* __restrict__ keys_sorted,
                                       int32_t* __restrict__ is_start)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    is_start[i] = (i == 0 || keys_sorted[i] != keys_sorted[i - 1]) ? 1 : 0;
}

// gather_representatives_kernel — one thread per UNIQUE voxel j: look up
// which SORTED-ARRAY position that voxel's run starts at (positions[j],
// built on the host side by launch_dedup_voxel_grid via thrust::copy_if),
// then read idx_sorted at that position to recover the ORIGINAL point
// index. Because the sort upstream was a STABLE sort_by_key over keys with
// idx initialized to 0..n-1 (ascending), the first position in every run
// carries the SMALLEST original index among that voxel's points — the same
// deterministic tie-break dedup_voxel_grid_cpu's single ascending pass
// produces (kernels.cuh's file header states this explicitly), which is
// exactly what lets DEDUP_ACCOUNTING assert EXACT agreement, not just an
// equal count.
__global__ void gather_representatives_kernel(int num_unique, const int32_t* __restrict__ positions,
                                              const int32_t* __restrict__ idx_sorted,
                                              int32_t* __restrict__ representative_orig_idx)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= num_unique) return;
    representative_orig_idx[j] = idx_sorted[positions[j]];
}

// launch_dedup_voxel_grid — glues the three kernels above together with
// Thrust's sort_by_key for the one genuinely non-trivial step (a full
// comparison sort — hand-rolling a GPU radix/merge sort would teach
// sorting, not merging, so this project uses the library the way
// 02.01/02.09 do, and says so explicitly per CLAUDE.md §1's "no black
// boxes" rule: thrust::stable_sort_by_key here performs a segmented radix
// sort under the hood for integer keys, O(n) passes over the data; we use
// it INSTEAD OF hand-rolling because the sort itself is not this project's
// teaching content — the VOXEL-GRID DEDUP ALGORITHM built on top of it is).
//
// Scratch buffers are allocated INSIDE this function (not caller-provided,
// unlike the assembly kernel's block_partials): dedup runs a handful of
// times per demo on a few-thousand-point merged cloud, so the extra
// cudaMalloc/cudaFree cost is negligible next to the sort itself — CLAUDE.md
// §1's "teaching clarity over micro-optimization" default.
// ---------------------------------------------------------------------------
int launch_dedup_voxel_grid(int n, const float* d_xyz, float cell, int32_t* d_representative_idx)
{
    if (n <= 0) return 0;

    unsigned long long* d_keys = nullptr;
    int32_t* d_idx = nullptr;
    int32_t* d_is_start = nullptr;
    int32_t* d_positions = nullptr;   // sorted-array positions where a new voxel starts (upper bound: n)
    CUDA_CHECK(cudaMalloc(&d_keys, sizeof(unsigned long long) * static_cast<size_t>(n)));
    CUDA_CHECK(cudaMalloc(&d_idx, sizeof(int32_t) * static_cast<size_t>(n)));
    CUDA_CHECK(cudaMalloc(&d_is_start, sizeof(int32_t) * static_cast<size_t>(n)));
    CUDA_CHECK(cudaMalloc(&d_positions, sizeof(int32_t) * static_cast<size_t>(n)));

    const int block = kThreadsPerBlock;
    const int grid = blocks_for(n, block);
    compute_hash_keys_kernel<<<grid, block>>>(n, d_xyz, cell, d_keys);
    CUDA_CHECK_LAST_ERROR("compute_hash_keys_kernel launch");

    // thrust::sequence: d_idx[i] = i (the "original index" every downstream
    // step tracks through the sort). thrust::device_ptr wraps our raw CUDA
    // pointers so Thrust's algorithms know these are DEVICE addresses.
    thrust::device_ptr<int32_t> idx_ptr(d_idx);
    thrust::sequence(idx_ptr, idx_ptr + n, 0);

    // The one library call this pipeline leans on: sort (key, idx) pairs by
    // key, STABLE (ties keep their relative order — see the file header for
    // why that specific guarantee is load-bearing here, not incidental).
    thrust::device_ptr<unsigned long long> keys_ptr(d_keys);
    thrust::stable_sort_by_key(keys_ptr, keys_ptr + n, idx_ptr);

    mark_boundaries_kernel<<<grid, block>>>(n, d_keys, d_is_start);
    CUDA_CHECK_LAST_ERROR("mark_boundaries_kernel launch");

    // Compact: copy the POSITION i (0..n-1) wherever is_start[i] != 0, using
    // a counting_iterator as the source (thrust's idiom for "emit my own
    // index") and the STENCIL overload of copy_if with the named
    // is_nonzero_flag predicate above (avoiding any device-lambda /
    // --extended-lambda compiler-flag dependency — a small, deliberate
    // simplicity choice). The returned iterator's distance from
    // d_positions is exactly the number of unique voxels — no separate
    // count pass needed.
    thrust::device_ptr<int32_t> is_start_ptr(d_is_start);
    thrust::device_ptr<int32_t> positions_ptr(d_positions);
    auto end_it = thrust::copy_if(
        thrust::counting_iterator<int32_t>(0), thrust::counting_iterator<int32_t>(n),
        is_start_ptr, positions_ptr, is_nonzero_flag());
    const int num_unique = static_cast<int>(end_it - positions_ptr);

    const int grid_u = blocks_for(num_unique, block);
    if (num_unique > 0) {
        gather_representatives_kernel<<<grid_u, block>>>(num_unique, d_positions, d_idx, d_representative_idx);
        CUDA_CHECK_LAST_ERROR("gather_representatives_kernel launch");
    }

    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_idx));
    CUDA_CHECK(cudaFree(d_is_start));
    CUDA_CHECK(cudaFree(d_positions));
    return num_unique;
}
