// ===========================================================================
// kernels.cu — GPU kernels for project 02.03
//              (Ground segmentation: RANSAC plane fit; Patchwork++-style
//              GPU port)
//
// Role in the project
// -------------------
// All __global__ (GPU) code lives here, plus the small host-side launch
// wrappers that own grid/block math (kept beside the kernel they configure —
// CLAUDE.md §6.1). Two independent pipelines live in this one file, sharing
// only the eigensolver and a block-reduction helper:
//
//   MILESTONE 1 (RANSAC)         MILESTONE 2 (Patchwork++-style CZM)
//   ---------------------         -----------------------------------
//   generate_hypotheses            compute_patch_ids
//   evaluate_hypotheses             sort_and_index (Thrust)
//   accumulate_inliers               fit_and_classify
//   refine
//
// Read this after: main.cu, kernels.cuh.  Read this before: reference_cpu.cpp.
// ===========================================================================

#include <cfloat>                // FLT_MAX — the min-z reduction's identity element
#include <thrust/device_ptr.h>   // wraps a raw device pointer for Thrust dispatch (02.01's precedent)
#include <thrust/sequence.h>     // thrust::sequence — parallel iota
#include <thrust/sort.h>         // thrust::stable_sort_by_key — radix sort under the hood
#include <thrust/binary_search.h>// thrust::lower_bound (vectorized form) — patch boundary search
#include <thrust/device_vector.h>// a TINY (161-int) internally-owned scratch buffer, see launch_czm_sort_and_index

#include "kernels.cuh"           // our own interface — keeps decl/def in sync at compile time
#include "util/cuda_check.cuh"   // CUDA_CHECK / CUDA_CHECK_LAST_ERROR

// ===========================================================================
// __device__ transcriptions of kernels.cuh's plain (host-only) shared
// functions. Each one below is a LITERAL, token-for-token copy of its header
// twin, recompiled with __device__ so it may be called from inside a
// __global__ kernel (kernels.cuh's file header explains why this
// duplication is necessary and how drift is caught: every one of these is
// exercised by a VERIFY(...) gate in main.cu that compares this kernel
// file's output against reference_cpu.cpp's independent CPU computation).
// ===========================================================================

__device__ inline uint32_t xorshift32_step_dev(uint32_t state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

__device__ inline uint32_t hypothesis_seed_dev(uint32_t global_seed, int k, int attempt)
{
    uint32_t s = global_seed ^ (0x9E3779B9u * static_cast<uint32_t>(k * 8 + attempt + 1));
    if (s == 0u) s = 1u;
    s = xorshift32_step_dev(s);
    s = xorshift32_step_dev(s);
    return s;
}

__device__ inline void pick_triplet_indices_dev(uint32_t seed, int n, int& i0, int& i1, int& i2)
{
    uint32_t s = seed;
    s = xorshift32_step_dev(s); i0 = static_cast<int>(s % static_cast<uint32_t>(n));
    s = xorshift32_step_dev(s); i1 = static_cast<int>(s % static_cast<uint32_t>(n));
    s = xorshift32_step_dev(s); i2 = static_cast<int>(s % static_cast<uint32_t>(n));
}

__device__ inline bool plane_from_triplet_dev(const float p0[3], const float p1[3], const float p2[3], PlaneModel& out)
{
    const float e1x = p1[0] - p0[0], e1y = p1[1] - p0[1], e1z = p1[2] - p0[2];
    const float e2x = p2[0] - p0[0], e2y = p2[1] - p0[1], e2z = p2[2] - p0[2];
    const float cx = e1y * e2z - e1z * e2y;
    const float cy = e1z * e2x - e1x * e2z;
    const float cz = e1x * e2y - e1y * e2x;
    const float norm2 = cx * cx + cy * cy + cz * cz;
    if (norm2 < kRansacMinCrossNormM2 * kRansacMinCrossNormM2) return false;
    const float inv_norm = 1.0f / sqrtf(norm2);
    float nx = cx * inv_norm, ny = cy * inv_norm, nz = cz * inv_norm;
    if (nz < 0.0f) { nx = -nx; ny = -ny; nz = -nz; }
    out.nx = nx; out.ny = ny; out.nz = nz;
    out.d  = -(nx * p0[0] + ny * p0[1] + nz * p0[2]);
    return true;
}

__device__ inline float point_plane_signed_distance_dev(const PlaneModel& pl, const float p[3])
{
    return pl.nx * p[0] + pl.ny * p[1] + pl.nz * p[2] + pl.d;
}

// __device__ copies of kernels.cuh's kCzmZoneEdgesM / kCzmZoneSectors host
// arrays. WHY these need a separate device-memory copy (unlike this file's
// __device__ *functions*, which are plain literal transcriptions of their
// header twins): a plain `constexpr float arr[N]` declared in a header is a
// HOST global with internal linkage — nvcc can inline a device-code READ of
// one ELEMENT at a *compile-time-constant* index (the compiler just embeds
// the value as an immediate), but every access below indexes with a
// RUNTIME loop variable, which requires the array to actually EXIST in
// device-addressable memory. `__device__` gives it exactly that: a genuine
// device-global copy, values initialized to match kernels.cuh's host
// arrays (kept in sync by eye — the same "duplicated, cross-referenced"
// pattern as this file's __device__ function transcriptions; VERIFY
// (patch_ids) in main.cu is the drift-catching gate, comparing every
// point's GPU patch id against the CPU twin that reads the header's HOST
// arrays directly).
__device__ constexpr float kCzmZoneEdgesMDev[kCzmNumZones + 1] = { 0.5f, 4.0f, 8.0f, 14.0f, 20.0f };
__device__ constexpr int   kCzmZoneSectorsDev[kCzmNumZones]    = { 32, 24, 16, 8 };

__device__ inline int czm_column_offset_for_zone_dev(int zone)
{
    int off = 0;
    for (int z = 0; z < zone; ++z) off += kCzmZoneSectorsDev[z];
    return off;
}

__device__ inline int czm_compute_patch_id_dev(float x, float y)
{
    const float r = sqrtf(x * x + y * y);
    if (r < kCzmZoneEdgesMDev[0] || r >= kCzmZoneEdgesMDev[kCzmNumZones]) return -1;

    int zone = 0;
    for (int z = 0; z < kCzmNumZones; ++z) {
        if (r >= kCzmZoneEdgesMDev[z] && r < kCzmZoneEdgesMDev[z + 1]) { zone = z; break; }
    }
    const float zone_lo = kCzmZoneEdgesMDev[zone];
    const float zone_hi = kCzmZoneEdgesMDev[zone + 1];
    const float ring_width = (zone_hi - zone_lo) / static_cast<float>(kCzmRingsPerZone);
    int ring = static_cast<int>((r - zone_lo) / ring_width);
    if (ring >= kCzmRingsPerZone) ring = kCzmRingsPerZone - 1;

    const int sectors = kCzmZoneSectorsDev[zone];
    const float kPi = 3.14159265358979323846f;
    float az = atan2f(y, x);
    if (az < 0.0f) az += 2.0f * kPi;
    int sector = static_cast<int>(az / (2.0f * kPi / static_cast<float>(sectors)));
    if (sector >= sectors) sector = sectors - 1;
    if (sector < 0) sector = 0;

    const int column = czm_column_offset_for_zone_dev(zone) + sector;
    return column * kCzmRingsPerZone + ring;
}

// jacobi_eigen_3x3_dev — literal device transcription of kernels.cuh's
// jacobi_eigen_3x3 (see that function's comment for the algorithm: fixed
// 8-sweep cyclic Jacobi rotation on a symmetric 3x3). Duplicated rather
// than shared because it is called from inside __global__ kernels below.
__device__ inline void jacobi_eigen_3x3_dev(const float a_in[6], float eigenvalues[3], float eigenvectors[3][3])
{
    float A[3][3] = {
        { a_in[0], a_in[1], a_in[2] },
        { a_in[1], a_in[3], a_in[4] },
        { a_in[2], a_in[4], a_in[5] },
    };
    float V[3][3] = { {1,0,0}, {0,1,0}, {0,0,1} };

    const int kSweeps = 8;
    for (int sweep = 0; sweep < kSweeps; ++sweep) {
        const int pairs[3][2] = { {0,1}, {0,2}, {1,2} };
        for (int pi = 0; pi < 3; ++pi) {
            const int p = pairs[pi][0], q = pairs[pi][1];
            const float apq = A[p][q];
            if (fabsf(apq) < 1.0e-12f) continue;
            const float theta = (A[q][q] - A[p][p]) / (2.0f * apq);
            const float t = (theta >= 0.0f ? 1.0f : -1.0f) /
                            (fabsf(theta) + sqrtf(theta * theta + 1.0f));
            const float c = 1.0f / sqrtf(t * t + 1.0f);
            const float s = t * c;
            const float app = A[p][p], aqq = A[q][q];
            A[p][p] = app - t * apq;
            A[q][q] = aqq + t * apq;
            A[p][q] = 0.0f; A[q][p] = 0.0f;
            for (int i = 0; i < 3; ++i) {
                if (i != p && i != q) {
                    const float aip = A[i][p], aiq = A[i][q];
                    A[i][p] = c * aip - s * aiq; A[p][i] = A[i][p];
                    A[i][q] = s * aip + c * aiq; A[q][i] = A[i][q];
                }
                const float vip = V[i][p], viq = V[i][q];
                V[i][p] = c * vip - s * viq;
                V[i][q] = s * vip + c * viq;
            }
        }
    }

    float ev[3] = { A[0][0], A[1][1], A[2][2] };
    float vec[3][3] = {
        { V[0][0], V[1][0], V[2][0] },
        { V[0][1], V[1][1], V[2][1] },
        { V[0][2], V[1][2], V[2][2] },
    };
    for (int i = 0; i < 2; ++i) {
        int min_j = i;
        for (int j = i + 1; j < 3; ++j) if (ev[j] < ev[min_j]) min_j = j;
        if (min_j != i) {
            const float tmp_ev = ev[i]; ev[i] = ev[min_j]; ev[min_j] = tmp_ev;
            for (int c2 = 0; c2 < 3; ++c2) { const float t2 = vec[i][c2]; vec[i][c2] = vec[min_j][c2]; vec[min_j][c2] = t2; }
        }
    }
    for (int i = 0; i < 3; ++i) {
        eigenvalues[i] = ev[i];
        eigenvectors[i][0] = vec[i][0]; eigenvectors[i][1] = vec[i][1]; eigenvectors[i][2] = vec[i][2];
    }
}

__device__ inline bool fit_plane_from_cov_accum_dev(const CovAccum9& acc, PlaneModel& out)
{
    if (acc.count < 3u) return false;
    const float inv_n = 1.0f / static_cast<float>(acc.count);
    const float mx = acc.sx * inv_n, my = acc.sy * inv_n, mz = acc.sz * inv_n;
    const float cxx = acc.sxx * inv_n - mx * mx;
    const float cxy = acc.sxy * inv_n - mx * my;
    const float cxz = acc.sxz * inv_n - mx * mz;
    const float cyy = acc.syy * inv_n - my * my;
    const float cyz = acc.syz * inv_n - my * mz;
    const float czz = acc.szz * inv_n - mz * mz;
    const float packed[6] = { cxx, cxy, cxz, cyy, cyz, czz };
    float eigenvalues[3]; float eigenvectors[3][3];
    jacobi_eigen_3x3_dev(packed, eigenvalues, eigenvectors);
    float nx = eigenvectors[0][0], ny = eigenvectors[0][1], nz = eigenvectors[0][2];
    if (nz < 0.0f) { nx = -nx; ny = -ny; nz = -nz; }
    out.nx = nx; out.ny = ny; out.nz = nz;
    out.d  = -(nx * mx + ny * my + nz * mz);
    return true;
}

// ---------------------------------------------------------------------------
// block_reduce_sum_dev / block_reduce_min_dev — the standard shared-memory
// tree reduction (CLAUDE.md's expected "explain the pattern" comment): every
// thread contributes one value into shared[threadIdx.x], then successive
// __syncthreads()-guarded halving passes fold the array in place until
// shared[0] holds the block-wide result, which every thread then reads back
// (a BROADCAST — every caller gets the same answer, not just thread 0).
//
// `shared` MUST be sized >= blockDim.x by the caller (both CZM/RANSAC
// kernels below launch with a FIXED block size matching their static
// __shared__ array, so blockDim.x here is always a compile-time-known
// power of two: 128 for the CZM fit kernel, 256 for RANSAC evaluate).
// Called REPEATEDLY within one kernel invocation (once per accumulated
// quantity) — the trailing __syncthreads() before returning is what makes
// that safe: it guarantees every thread has finished READING shared[0]
// before any thread starts WRITING the next call's shared[threadIdx.x].
// ---------------------------------------------------------------------------
__device__ inline float block_reduce_sum_dev(float val, float* shared)
{
    const int tid = threadIdx.x;
    shared[tid] = val;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < static_cast<int>(s)) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    const float result = shared[0];
    __syncthreads();   // protect against the NEXT call's writes racing this call's readers
    return result;
}

__device__ inline float block_reduce_min_dev(float val, float* shared)
{
    const int tid = threadIdx.x;
    shared[tid] = val;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < static_cast<int>(s)) shared[tid] = fminf(shared[tid], shared[tid + s]);
        __syncthreads();
    }
    const float result = shared[0];
    __syncthreads();
    return result;
}

// ===========================================================================
// MILESTONE 1 — RANSAC plane fit
// ===========================================================================

// ransac_generate_hypotheses_kernel — see kernels.cuh for the full
// documentation of the counter-based per-hypothesis RNG design. Thread
// mapping: grid-stride over the K hypotheses (K=1024 fits in 4 blocks of
// 256 with zero remainder, but grid-stride is used anyway so this kernel
// stays correct if K ever changes — the same robustness argument the
// template's SAXPY placeholder makes for the grid-stride idiom generally).
__global__ void ransac_generate_hypotheses_kernel(int n, const float* __restrict__ xyz,
                                                   uint32_t global_seed, int k,
                                                   PlaneModel* __restrict__ hyp_plane,
                                                   uint8_t* __restrict__ hyp_valid)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int h = idx; h < k; h += stride) {
        bool found = false;
        PlaneModel pl;
        // Degenerate-triplet retry loop: a near-collinear/coincident draw
        // is rare on this scene's dense point cloud but not impossible
        // (most likely in the CZM's sparsest far-zone rings, which RANSAC
        // also samples from since it draws from the WHOLE cloud) — retry
        // with a freshly-mixed seed (hypothesis_seed_dev's `attempt`
        // parameter) rather than accept a numerically meaningless normal.
        for (int attempt = 0; attempt < kRansacMaxTripletAttempts && !found; ++attempt) {
            const uint32_t seed = hypothesis_seed_dev(global_seed, h, attempt);
            int i0, i1, i2;
            pick_triplet_indices_dev(seed, n, i0, i1, i2);
            const float p0[3] = { xyz[i0*3+0], xyz[i0*3+1], xyz[i0*3+2] };
            const float p1[3] = { xyz[i1*3+0], xyz[i1*3+1], xyz[i1*3+2] };
            const float p2[3] = { xyz[i2*3+0], xyz[i2*3+1], xyz[i2*3+2] };
            if (plane_from_triplet_dev(p0, p1, p2, pl)) found = true;
        }
        hyp_plane[h] = pl;
        hyp_valid[h] = found ? 1u : 0u;
    }
}

void launch_ransac_generate_hypotheses(int n, const float* d_xyz, uint32_t global_seed, int k,
                                       PlaneModel* d_hyp_plane, uint8_t* d_hyp_valid)
{
    const int block = kThreadsPerBlock;
    const int grid  = blocks_for(k, block);
    ransac_generate_hypotheses_kernel<<<grid, block>>>(n, d_xyz, global_seed, k, d_hyp_plane, d_hyp_valid);
    CUDA_CHECK_LAST_ERROR("ransac_generate_hypotheses_kernel launch");
}

// ---------------------------------------------------------------------------
// ransac_evaluate_hypotheses_kernel — the K x N batched heart of Milestone 1.
//
// GPU MAPPING CHOSEN: ONE BLOCK PER HYPOTHESIS (grid.x == k, block ==
// kThreadsPerBlock == 256). Every thread in hypothesis k's block strides
// over ALL n points, testing |distance to plane k| <= threshold, then a
// shared-memory tree reduction folds the block's 256 partial counts into
// one integer written to hyp_inlier_count[k].
//
// THE ALTERNATIVE MAPPING (thread-per-(hypothesis,chunk) with global
// atomics: launch a big 2-D-flattened grid where each thread owns one
// (hypothesis, point-chunk) pair and atomicAdd's into a GLOBAL
// hyp_inlier_count[k] array) trades this kernel's clean block-local
// reduction for finer-grained parallelism — useful if K were small and N
// were enormous, so a single block per hypothesis would not fill the GPU.
// Here it is the other way around: K=1024 already gives 1024 blocks,
// comfortably saturating an RTX 2080 SUPER's 46 SMs many times over (~22
// blocks/SM at peak occupancy), so the SIMPLER block-per-hypothesis mapping
// (no atomics, no contention, no cross-block synchronization) wins on both
// clarity and performance for this problem size — the same "farm the
// independent unit of work across blocks" framing 08.01's MPPI rollout
// kernel uses (K independent trajectories -> K independent hypotheses here).
//
// Invalid hypotheses (hyp_valid[k]==0, every triplet attempt was
// degenerate) short-circuit to inlier_count=-1 — a sentinel
// select_best_hypothesis (kernels.cuh) skips, never a candidate for "best".
// ---------------------------------------------------------------------------
__global__ void ransac_evaluate_hypotheses_kernel(int n, const float* __restrict__ xyz,
                                                   const PlaneModel* __restrict__ hyp_plane,
                                                   const uint8_t* __restrict__ hyp_valid,
                                                   float threshold,
                                                   int* __restrict__ hyp_inlier_count)
{
    const int k = blockIdx.x;   // this BLOCK's hypothesis index
    __shared__ int sdata_i[256];  // block == kThreadsPerBlock == 256 (fixed at launch below)

    if (!hyp_valid[k]) {
        if (threadIdx.x == 0) hyp_inlier_count[k] = -1;
        return;   // uniform across the whole block (hyp_valid[k] is the same value for every thread) — safe early exit
    }

    const PlaneModel pl = hyp_plane[k];
    int local_count = 0;
    // Grid-stride WITHIN the block (gridDim.x here is K, not a point-count
    // grid — each block covers the FULL point range itself, block-locally).
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float p[3] = { xyz[i*3+0], xyz[i*3+1], xyz[i*3+2] };
        const float dist = point_plane_signed_distance_dev(pl, p);
        if (fabsf(dist) <= threshold) ++local_count;
    }

    sdata_i[threadIdx.x] = local_count;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata_i[threadIdx.x] += sdata_i[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) hyp_inlier_count[k] = sdata_i[0];
}

void launch_ransac_evaluate_hypotheses(int n, const float* d_xyz, const PlaneModel* d_hyp_plane,
                                       const uint8_t* d_hyp_valid, float threshold, int k,
                                       int* d_hyp_inlier_count)
{
    // ONE BLOCK PER HYPOTHESIS — see the kernel's header comment for why.
    ransac_evaluate_hypotheses_kernel<<<k, kThreadsPerBlock>>>(n, d_xyz, d_hyp_plane, d_hyp_valid,
                                                               threshold, d_hyp_inlier_count);
    CUDA_CHECK_LAST_ERROR("ransac_evaluate_hypotheses_kernel launch");
}

// ---------------------------------------------------------------------------
// ransac_accumulate_inliers_kernel — one thread per point (grid-stride);
// inliers of `plane` atomicAdd their (x,y,z) and 6 products into the shared
// CovAccum9 accumulator.
//
// WHY ATOMICS HERE (an order-nondeterministic choice, unlike 02.01's
// Method-B-style fixed-order reduction): this step runs ONCE per RANSAC
// call (not thousands of times like hypothesis evaluation), refining a
// single already-good plane. The engineering trade this project makes —
// spend the simplicity of atomics here, pay for it with a TOLERANCE-based
// (not bit-exact) VERIFY(ransac_refine) gate in main.cu — mirrors 02.01's
// Method A exactly (that project's kernels.cuh "TWO METHODS, ONE ANSWER"
// comment names this identical trade-off). A Method-B-style sorted/
// fixed-order reduction WOULD be possible here too (stream-compact the
// inlier indices, then a single-thread-per-nothing sequential sum) but
// would not teach anything new this project's CZM milestone does not
// already cover with its OWN fixed-order, block-sequential reductions —
// so the simpler atomic path was chosen for THIS step specifically.
// ---------------------------------------------------------------------------
__global__ void ransac_accumulate_inliers_kernel(int n, const float* __restrict__ xyz,
                                                  PlaneModel plane, float threshold,
                                                  CovAccum9* __restrict__ accum,
                                                  uint8_t* __restrict__ point_inlier_mask)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float p[3] = { xyz[i*3+0], xyz[i*3+1], xyz[i*3+2] };
        const float dist = point_plane_signed_distance_dev(plane, p);
        const bool inlier = fabsf(dist) <= threshold;
        point_inlier_mask[i] = inlier ? 1u : 0u;
        if (inlier) {
            atomicAdd(&accum->sx, p[0]);  atomicAdd(&accum->sy, p[1]);  atomicAdd(&accum->sz, p[2]);
            atomicAdd(&accum->sxx, p[0]*p[0]); atomicAdd(&accum->sxy, p[0]*p[1]); atomicAdd(&accum->sxz, p[0]*p[2]);
            atomicAdd(&accum->syy, p[1]*p[1]); atomicAdd(&accum->syz, p[1]*p[2]); atomicAdd(&accum->szz, p[2]*p[2]);
            atomicAdd(&accum->count, 1u);
        }
    }
}

void launch_ransac_accumulate_inliers(int n, const float* d_xyz, PlaneModel plane, float threshold,
                                      CovAccum9* d_accum, uint8_t* d_point_inlier_mask)
{
    // Fresh accumulation every call — reset before the atomics start.
    CUDA_CHECK(cudaMemset(d_accum, 0, sizeof(CovAccum9)));
    const int block = kThreadsPerBlock;
    const int grid  = blocks_for(n, block) > 4096 ? 4096 : blocks_for(n, block);
    ransac_accumulate_inliers_kernel<<<grid, block>>>(n, d_xyz, plane, threshold, d_accum, d_point_inlier_mask);
    CUDA_CHECK_LAST_ERROR("ransac_accumulate_inliers_kernel launch");
}

// ransac_refine_kernel — <<<1,1>>>: DELIBERATELY serial. See kernels.cuh's
// declaration comment: this is a K=1 batched-solve, contrasted with the
// 160-way parallel solve czm_fit_and_classify_kernel performs below.
__global__ void ransac_refine_kernel(const CovAccum9* __restrict__ accum,
                                     PlaneModel raw_plane, PlaneModel* __restrict__ refined_plane,
                                     int* __restrict__ refined_ok)
{
    const CovAccum9 acc = *accum;
    PlaneModel out;
    const bool ok = fit_plane_from_cov_accum_dev(acc, out);
    *refined_plane = ok ? out : raw_plane;
    *refined_ok = ok ? 1 : 0;
}

void launch_ransac_refine(const CovAccum9* d_accum, PlaneModel raw_plane,
                          PlaneModel* d_refined_plane, int* d_refined_ok)
{
    ransac_refine_kernel<<<1, 1>>>(d_accum, raw_plane, d_refined_plane, d_refined_ok);
    CUDA_CHECK_LAST_ERROR("ransac_refine_kernel launch");
}

// ===========================================================================
// MILESTONE 2 — Patchwork++-style concentric-zone model (CZM)
// ===========================================================================

__global__ void czm_compute_patch_ids_kernel(int n, const float* __restrict__ xyz,
                                             int* __restrict__ patch_id)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        patch_id[i] = czm_compute_patch_id_dev(xyz[i*3+0], xyz[i*3+1]);
    }
}

void launch_czm_compute_patch_ids(int n, const float* d_xyz, int* d_patch_id)
{
    const int block = kThreadsPerBlock;
    const int grid  = blocks_for(n, block) > 4096 ? 4096 : blocks_for(n, block);
    czm_compute_patch_ids_kernel<<<grid, block>>>(n, d_xyz, d_patch_id);
    CUDA_CHECK_LAST_ERROR("czm_compute_patch_ids_kernel launch");
}

// ---------------------------------------------------------------------------
// launch_czm_sort_and_index — turn "every point's patch id" into "every
// patch's contiguous point range", via two Thrust calls (CLAUDE.md §6.1
// rule 6 — what each one computes and why it beats a hand-rolled kernel):
//
//   thrust::stable_sort_by_key(patch_id, patch_id+n, point_idx) — sorts the
//   KEY range (patch ids, -1..159) ascending and permutes the paired VALUE
//   range (point_idx, initialized 0..n-1 by thrust::sequence) the same way.
//   Internally a RADIX sort (cheap here: patch ids span only 161 distinct
//   values, so this is close to radix sort's best case — a handful of
//   passes over a small key range). STABLE means points sharing a patch
//   keep their original relative (index) order — irrelevant for correctness
//   here (unlike 02.01's Method B, nothing downstream depends on the
//   WITHIN-patch order), but free and harmless to keep.
//
//   thrust::lower_bound(keys_first, keys_last, values_first, values_last,
//   result) — the VECTORIZED binary search: for each of the
//   kCzmNumPatches+1 PROBE values 0,1,...,kCzmNumPatches (built by
//   thrust::sequence into a tiny internal device_vector), find the index of
//   the first sorted patch_id >= that probe, all in one device-parallel
//   call (one binary search per probe, O(log n) each, but only 161 probes
//   total — negligible next to a hand-rolled equivalent of 161 separate
//   kernel launches). probe 160 always resolves to n (no patch id reaches
//   160), which is exactly patch 159's end boundary.
//
// A HAND-ROLLED alternative (mark_boundaries_kernel + copy_if, 02.01's
// Method B technique) would also work, but that pattern is already taught
// there; the vectorized-binary-search idiom is a genuinely different,
// useful Thrust tool worth introducing once in this repository.
// ---------------------------------------------------------------------------
void launch_czm_sort_and_index(int n, int* d_patch_id_scratch, int* d_point_idx, int* d_patch_start)
{
    thrust::device_ptr<int> patch_id_ptr(d_patch_id_scratch);
    thrust::device_ptr<int> idx_ptr(d_point_idx);

    thrust::sequence(idx_ptr, idx_ptr + n);                          // idx[i] = i (identity permutation)
    thrust::stable_sort_by_key(patch_id_ptr, patch_id_ptr + n, idx_ptr);

    // A tiny (kCzmNumPatches+1 == 161 ints) internally-owned scratch buffer
    // — small and fixed-size enough that thrust::device_vector's RAII
    // convenience is worth the one exception to this repo's "caller owns
    // all allocations" convention (see kernels.cuh's launch_czm_sort_and_index
    // declaration comment).
    thrust::device_vector<int> probes(kCzmNumPatches + 1);
    thrust::sequence(probes.begin(), probes.end(), 0);                // 0,1,2,...,kCzmNumPatches

    thrust::device_ptr<int> patch_start_ptr(d_patch_start);
    thrust::lower_bound(patch_id_ptr, patch_id_ptr + n, probes.begin(), probes.end(), patch_start_ptr);
}

// ---------------------------------------------------------------------------
// czm_fit_and_classify_kernel — ONE BLOCK PER COLUMN (grid.x ==
// kCzmNumColumns == 80, block == 128 threads, matching the static
// __shared__ float sred[128] reduction scratch below).
//
// Per column, the block processes ring 0 then ring 1 IN ORDER (a genuine
// data dependency: ring 1's seed selection may use ring 0's fitted plane),
// each ring going through the same 4 collaborative passes over that
// PATCH'S point range [patch_start[p], patch_start[p+1)):
//
//   1) MIN-Z reduction  -> establishes the "no prior" seed rule's floor.
//   2) SEED-RANGE COVARIANCE reduction (10 quantities: sum x/y/z, the 6
//      unique xx/xy/xz/yy/yz/zz products, and the seed point count) over
//      points whose z falls in the ring's seed window — feeds the shared
//      eigensolver (fit_plane_from_cov_accum_dev) via thread 0.
//   3) FLATNESS residual reduction (sum of squared perpendicular distances
//      of the SAME seed points to the just-fitted plane) — only meaningful
//      if the plane tentatively passed the uprightness test; the block
//      still executes this pass unconditionally (see the comment below on
//      why every __syncthreads()-bearing branch here is BLOCK-UNIFORM).
//   4) CLASSIFICATION pass over ALL of the patch's points (not just the
//      seed subset) — writes point_ground[original point index] = 0/1.
//
// UNIFORM CONTROL FLOW: every branch that reaches a __syncthreads() (the
// npts/count/is_ground checks) tests a value computed IDENTICALLY by every
// thread in the block (loaded from the same patch_start[] entries or the
// same shared variable) — never a per-thread-varying condition. This is
// what makes it safe for those branches to contain block_reduce_*_dev
// calls (which themselves __syncthreads()): CUDA requires every thread in
// a block to reach the SAME __syncthreads() call, and a block-uniform
// branch guarantees exactly that.
//
// REGION GROWING (the CZM's answer to a ramp/plateau a single RANSAC plane
// cannot represent): after ring 0 fits successfully, its plane is evaluated
// at the seed centroid's (x,y) to predict a ground HEIGHT (not just reuse
// the seed's mean z — evaluating the actual fitted plane lets the
// prediction inherit the ramp's slope), and ring 1's seed window is
// centered on that predicted height instead of ring 1's own (blind)
// minimum-z. THEORY.md "The algorithm" walks through why this specific
// design recovers the ramp and plateau where single-plane RANSAC cannot.
// ---------------------------------------------------------------------------
__global__ void czm_fit_and_classify_kernel(const float* __restrict__ xyz,
                                            const int* __restrict__ idx_sorted,
                                            const int* __restrict__ patch_start,
                                            CzmPatchResult* __restrict__ patch_result,
                                            uint8_t* __restrict__ point_ground)
{
    const int col = blockIdx.x;   // this block's column (a (zone,sector) pair)
    const int tid = threadIdx.x;

    __shared__ float sred[128];     // reduction scratch, REUSED across every pass below (block size fixed at 128)
    __shared__ float s_plane[4];    // broadcast: the current ring's tentative/final plane (nx,ny,nz,d)
    __shared__ int   s_is_ground;   // broadcast: current ring's pass/fail verdict
    __shared__ float s_zlo, s_zhi;  // broadcast: current ring's seed z-window

    float carry_z = 0.0f;   // per-thread locals, but only thread 0's copy is ever read/written (see below)
    int   carry_valid = 0;

    for (int ring = 0; ring < kCzmRingsPerZone; ++ring) {
        const int patch = col * kCzmRingsPerZone + ring;
        const int begin = patch_start[patch];       // SAME value for every thread — a block-uniform read
        const int end   = patch_start[patch + 1];
        const int npts  = end - begin;

        // ---- Pass 1: MIN-Z (always executed, even if npts==0: the loop
        //      below then contributes nothing and local_min stays FLT_MAX —
        //      keeping this pass UNCONDITIONAL avoids any risk of a
        //      non-uniform branch around a __syncthreads()-bearing call). --
        float local_min = FLT_MAX;
        for (int k = begin + tid; k < end; k += blockDim.x) {
            const int p = idx_sorted[k];
            local_min = fminf(local_min, xyz[p*3+2]);
        }
        const float min_z = block_reduce_min_dev(local_min, sred);

        if (tid == 0) {
            if (carry_valid) { s_zlo = carry_z - kCzmHeightCarryBandM; s_zhi = carry_z + kCzmHeightCarryBandM; }
            else              { s_zlo = min_z; s_zhi = min_z + kCzmSeedHeightMarginM; }
        }
        __syncthreads();

        // ---- Pass 2: seed-range covariance (9 sums + count) -------------
        float lsx=0,lsy=0,lsz=0,lsxx=0,lsxy=0,lsxz=0,lsyy=0,lsyz=0,lszz=0,lcnt=0;
        for (int k = begin + tid; k < end; k += blockDim.x) {
            const int p = idx_sorted[k];
            const float x = xyz[p*3+0], y = xyz[p*3+1], z = xyz[p*3+2];
            if (z >= s_zlo && z <= s_zhi) {
                lsx += x; lsy += y; lsz += z;
                lsxx += x*x; lsxy += x*y; lsxz += x*z; lsyy += y*y; lsyz += y*z; lszz += z*z;
                lcnt += 1.0f;   // accumulated as float; exact for any realistic patch size (<< 2^24, float's exact-integer range)
            }
        }
        const float sx = block_reduce_sum_dev(lsx, sred);
        const float sy = block_reduce_sum_dev(lsy, sred);
        const float sz = block_reduce_sum_dev(lsz, sred);
        const float sxx = block_reduce_sum_dev(lsxx, sred);
        const float sxy = block_reduce_sum_dev(lsxy, sred);
        const float sxz = block_reduce_sum_dev(lsxz, sred);
        const float syy = block_reduce_sum_dev(lsyy, sred);
        const float syz = block_reduce_sum_dev(lsyz, sred);
        const float szz = block_reduce_sum_dev(lszz, sred);
        const float fcount = block_reduce_sum_dev(lcnt, sred);
        const unsigned int count = static_cast<unsigned int>(fcount + 0.5f);  // round-to-nearest guards float sum noise

        float upright_deg = 0.0f;
        if (tid == 0) {
            int tentative_ground = 0;
            PlaneModel pl;
            if (count >= kCzmMinPatchPoints) {
                CovAccum9 acc; acc.sx=sx; acc.sy=sy; acc.sz=sz; acc.sxx=sxx; acc.sxy=sxy; acc.sxz=sxz;
                acc.syy=syy; acc.syz=syz; acc.szz=szz; acc.count=count;
                if (fit_plane_from_cov_accum_dev(acc, pl)) {
                    upright_deg = acosf(fminf(fmaxf(pl.nz, -1.0f), 1.0f)) * (180.0f / 3.14159265358979323846f);
                    if (upright_deg <= kCzmUprightMaxDeg) tentative_ground = 1;
                }
            }
            s_plane[0] = pl.nx; s_plane[1] = pl.ny; s_plane[2] = pl.nz; s_plane[3] = pl.d;
            s_is_ground = tentative_ground;
        }
        __syncthreads();

        // ---- Pass 3: flatness residual (seed points vs. the tentative
        //      plane) — unconditional block_reduce call, per the uniform-
        //      control-flow note above; the accumulated value is simply
        //      unused downstream when s_is_ground is already 0. -----------
        float lsumsq = 0.0f;
        {
            const PlaneModel plane_b = { s_plane[0], s_plane[1], s_plane[2], s_plane[3] };
            for (int k = begin + tid; k < end; k += blockDim.x) {
                const int p = idx_sorted[k];
                const float x = xyz[p*3+0], y = xyz[p*3+1], z = xyz[p*3+2];
                if (z >= s_zlo && z <= s_zhi) {
                    const float pp[3] = { x, y, z };
                    const float dist = point_plane_signed_distance_dev(plane_b, pp);
                    lsumsq += dist * dist;
                }
            }
        }
        const float sumsq = block_reduce_sum_dev(lsumsq, sred);

        float rms = 0.0f;
        if (tid == 0 && s_is_ground) {
            rms = sqrtf(sumsq / fmaxf(static_cast<float>(count), 1.0f));
            if (rms > kCzmFlatnessMaxRmsM) s_is_ground = 0;   // fails flatness after all
        }
        __syncthreads();

        // ---- Pass 4: classify every point in the patch (not just seed) --
        {
            const PlaneModel plane_c = { s_plane[0], s_plane[1], s_plane[2], s_plane[3] };
            const int is_ground_final = s_is_ground;
            for (int k = begin + tid; k < end; k += blockDim.x) {
                const int p = idx_sorted[k];
                uint8_t g = 0u;
                if (is_ground_final) {
                    const float pp[3] = { xyz[p*3+0], xyz[p*3+1], xyz[p*3+2] };
                    const float dist = point_plane_signed_distance_dev(plane_c, pp);
                    g = (fabsf(dist) <= kCzmClassifyDistM) ? 1u : 0u;
                }
                point_ground[p] = g;
            }
        }

        // ---- Bookkeeping: write this patch's result, update the carry
        //      for the NEXT ring in this column (thread 0 only — carry_z/
        //      carry_valid are per-thread locals whose OTHER threads' copies
        //      are simply never read, see the kernel header note). ---------
        if (tid == 0) {
            CzmPatchResult res;
            res.plane.nx = s_plane[0]; res.plane.ny = s_plane[1]; res.plane.nz = s_plane[2]; res.plane.d = s_plane[3];
            res.is_ground = s_is_ground;
            res.patch_point_count = static_cast<unsigned int>(npts);
            res.seed_point_count = count;
            res.rms_residual_m = rms;
            res.uprightness_deg = upright_deg;
            res.used_prior = carry_valid ? 1 : 0;
            patch_result[patch] = res;

            if (s_is_ground) {
                // Evaluate the FITTED plane (not just the seed mean z) at
                // the seed centroid's (x,y): on the ramp this correctly
                // predicts a HIGHER ground height further out, letting
                // ring 1's seed window track the slope instead of missing
                // it (THEORY.md "The algorithm" walks through this step).
                const float mx = (count > 0) ? sx / static_cast<float>(count) : 0.0f;
                const float my = (count > 0) ? sy / static_cast<float>(count) : 0.0f;
                const float nz_safe = (fabsf(s_plane[2]) > 1.0e-3f) ? s_plane[2] : 1.0f;  // uprightness test bounds |nz| away from 0 in practice; guarded defensively
                carry_z = -(s_plane[3] + s_plane[0] * mx + s_plane[1] * my) / nz_safe;
                carry_valid = 1;
            }
            // else: leave carry_z/carry_valid UNCHANGED — "carry forward
            // the last known-good estimate" (only observable with >2 rings;
            // documented for whoever extends kCzmRingsPerZone later).
        }
        __syncthreads();   // whole block waits for thread 0's writes before the next ring reuses shared memory
    }
}

void launch_czm_fit_and_classify(const float* d_xyz, const int* d_idx_sorted, const int* d_patch_start,
                                 CzmPatchResult* d_patch_result, uint8_t* d_point_ground)
{
    // ONE BLOCK PER COLUMN — see the kernel's header comment. Block size
    // 128 is a fixed teaching choice: with kCzmNumColumns==80 blocks, this
    // launch alone does not saturate an RTX 2080 SUPER's 46 SMs the way the
    // K=1024-block RANSAC evaluation does — an honest, DELIBERATE contrast
    // (README/THEORY "The GPU mapping"): CZM's parallelism is bounded by
    // the number of PATCHES (a scene-design choice), not by N, so this
    // milestone teaches a smaller-scale, patch-parallel GPU mapping.
    czm_fit_and_classify_kernel<<<kCzmNumColumns, 128>>>(d_xyz, d_idx_sorted, d_patch_start,
                                                         d_patch_result, d_point_ground);
    CUDA_CHECK_LAST_ERROR("czm_fit_and_classify_kernel launch");
}
