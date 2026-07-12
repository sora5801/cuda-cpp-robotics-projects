// ===========================================================================
// kernels.cu - GPU kernels for project 02.18
//              Weather filtering: snow/rain/dust outlier removal (DROR/LIOR)
//
// Role in the project
// -------------------
// All __global__ (GPU) code lives here, together with the small host-side
// launch wrappers that own the grid/block math. Six kernels, in three
// STATISTIC/CLASSIFY pairs (kernels.cuh's file header): SOR's K-nearest mean
// distance, DROR's dynamic-radius neighbor count, LIOR's fixed-radius
// neighbor count - each followed by a trivial threshold-classify kernel.
//
// The GPU mapping, once, for all six kernels
// -------------------------------------------
// Every kernel here maps ONE THREAD TO ONE POINT. The STATISTIC kernels
// (sor_mean_knn_dist, dror_neighbor_count, lior_neighbor_count) each do an
// O(n) brute-force scan over every OTHER point from that one thread - a
// classic "all-pairs" GPU pattern: n points, n threads, each thread doing
// O(n) work, O(n^2) total - identical in spirit to project 02.13's beam-
// parallel DDA march (many independent workers, no communication between
// them) but with a dense all-pairs inner loop instead of a bounded ray
// march. It parallelizes trivially (every point's neighborhood is
// independent of every other point's) and needs no spatial index (kernels.cuh
// file header: that acceleration is project 02.05's/02.09's job) because n
// is small enough (roughly one to two thousand points per scan) that O(n^2)
// finishes in well under a millisecond on any current GPU (measured in
// README "Expected output"). The CLASSIFY kernels are pure independent-per-
// point maps over an already-computed per-point statistic - the simplest
// GPU pattern there is.
//
// Read this after: main.cu, kernels.cuh.  Read this before: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"           // our own interface - keeps decl/def in sync at compile time
#include "util/cuda_check.cuh"   // CUDA_CHECK_LAST_ERROR for post-launch error surfacing

// ---------------------------------------------------------------------------
// sor_mean_knn_dist_kernel - one thread per point: brute-force K-nearest
// search over every other point, then the mean of the K nearest distances.
//
// Thread-to-data mapping: thread i = blockIdx.x*blockDim.x + threadIdx.x
// owns point i; a grid-stride loop (see launch_* below) covers n > grid*block.
//
// K-nearest bookkeeping: a small INSERTION-SORTED array of size kSorK (8) -
// not a heap. With K this small, a heap buys nothing (its whole point is
// avoiding an O(K) shift on every insert, which at K=8 costs at most 8
// compare-and-maybe-shift steps - cheaper than a heap's own bookkeeping)
// and a sorted array is far easier for a reader to verify by eye (CLAUDE.md
// paragraph 5: "the reference is clever, it can be wrong" - the same
// reasoning applies to the GPU kernel when K is this small). best_d2[]/
// best_idx[] are kept in ASCENDING order under kernels.cuh's dist_less total
// order at all times; a new candidate is inserted only if it beats the
// current worst (best_d2[count-1]) or the array is not yet full.
//
// Memory behavior: xyz is read n times total by EVERY thread (all-pairs),
// so global-memory traffic is O(n^2) - the dominant cost at this kernel's
// point counts, and exactly why this project stays at "a few thousand
// points, brute force" scope rather than "millions of points, spatial
// index" scope (that scale-up is 02.05's/02.09's project, cited repeatedly).
// No shared memory: with n in the low thousands, the whole xyz array
// (n*3*4 bytes, well under 24 KB even at n=2,000) is small enough that the
// L1/L2 cache captures nearly all of the reuse a shared-memory TILE would
// buy - the tiled-into-shared-memory version is this project's Exercise,
// not its baseline (see README).
//
// Parameters: as declared in kernels.cuh.
// ---------------------------------------------------------------------------
__global__ void sor_mean_knn_dist_kernel(int n, const float* __restrict__ xyz,
                                          float* __restrict__ mean_dist)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float pi[3] = { xyz[i * 3 + 0], xyz[i * 3 + 1], xyz[i * 3 + 2] };

        float best_d2[kSorK];
        int32_t best_idx[kSorK];
        int count = 0;   // how many of the kSorK slots are filled so far

        for (int j = 0; j < n; ++j) {
            if (j == i) continue;   // a point is never its own neighbor
            const float pj[3] = { xyz[j * 3 + 0], xyz[j * 3 + 1], xyz[j * 3 + 2] };
            const float d2 = squared_distance3(pi, pj);

            if (count < kSorK) {
                // Insert into the still-filling sorted array: shift larger
                // entries right, drop the new candidate into its slot.
                int pos = count;
                while (pos > 0 && dist_less(d2, static_cast<int32_t>(j), best_d2[pos - 1], best_idx[pos - 1])) {
                    best_d2[pos] = best_d2[pos - 1];
                    best_idx[pos] = best_idx[pos - 1];
                    --pos;
                }
                best_d2[pos] = d2;
                best_idx[pos] = static_cast<int32_t>(j);
                ++count;
            } else if (dist_less(d2, static_cast<int32_t>(j), best_d2[kSorK - 1], best_idx[kSorK - 1])) {
                // Beats the current worst of the full K-set: shift it out.
                int pos = kSorK - 1;
                while (pos > 0 && dist_less(d2, static_cast<int32_t>(j), best_d2[pos - 1], best_idx[pos - 1])) {
                    best_d2[pos] = best_d2[pos - 1];
                    best_idx[pos] = best_idx[pos - 1];
                    --pos;
                }
                best_d2[pos] = d2;
                best_idx[pos] = static_cast<int32_t>(j);
            }
        }

        // Mean of sqrt(dist2) over whatever was found (fewer than kSorK
        // only when n-1 < kSorK, i.e. a degenerately tiny scan - guarded
        // honestly rather than dividing by a phantom kSorK).
        float sum = 0.0f;
        for (int k = 0; k < count; ++k) sum += sqrtf(best_d2[k]);
        mean_dist[i] = (count > 0) ? (sum / static_cast<float>(count)) : 0.0f;
    }
}

// ---------------------------------------------------------------------------
// sor_classify_kernel - pure independent-per-point map: threshold compare.
// ---------------------------------------------------------------------------
__global__ void sor_classify_kernel(int n, const float* __restrict__ mean_dist,
                                     float threshold, int32_t* __restrict__ mask_out)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        mask_out[i] = (mean_dist[i] > threshold) ? 1 : 0;
    }
}

// ---------------------------------------------------------------------------
// dror_neighbor_count_kernel - one thread per point: this point's OWN
// range-scaled radius (kernels.cuh's dror_search_radius, the physical heart
// of DROR - THEORY.md "The math" derives it from beam divergence + angular
// sampling), then a brute-force count of neighbors within it.
//
// Comparing SQUARED distance against radius^2 (kernels.cuh file header
// "squared_distance3") avoids a sqrtf() per candidate - this kernel and
// lior_neighbor_count_kernel below are the two places in this project where
// that micro-optimization actually matters (they run on every one of the
// n*(n-1) candidate pairs; sor_mean_knn_dist_kernel needs the true distance
// for its MEAN anyway, so it cannot avoid the sqrtf).
// ---------------------------------------------------------------------------
__global__ void dror_neighbor_count_kernel(int n, const float* __restrict__ xyz,
                                            int32_t* __restrict__ count_out)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float pi[3] = { xyz[i * 3 + 0], xyz[i * 3 + 1], xyz[i * 3 + 2] };
        const float radius = dror_search_radius(range3(pi));
        const float radius2 = radius * radius;

        int32_t cnt = 0;
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            const float pj[3] = { xyz[j * 3 + 0], xyz[j * 3 + 1], xyz[j * 3 + 2] };
            if (squared_distance3(pi, pj) <= radius2) ++cnt;
        }
        count_out[i] = cnt;
    }
}

// ---------------------------------------------------------------------------
// dror_classify_kernel - outlier iff fewer than kDrorKMin neighbors.
// ---------------------------------------------------------------------------
__global__ void dror_classify_kernel(int n, const int32_t* __restrict__ count,
                                      int32_t* __restrict__ mask_out)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        mask_out[i] = (count[i] < kDrorKMin) ? 1 : 0;
    }
}

// ---------------------------------------------------------------------------
// lior_neighbor_count_kernel - identical shape to dror_neighbor_count_kernel
// above, with ONE deliberate difference that IS the point of this kernel
// existing separately rather than calling a shared "radius_count(radius)"
// helper: the radius here is the FIXED kLiorRadius, never scaled by range.
// Keeping the two kernels textually separate (rather than one kernel taking
// a radius parameter) makes that contrast impossible to miss while reading
// the file top to bottom - kernels.cuh's file header names this as this
// project's deliberate teaching choice.
// ---------------------------------------------------------------------------
__global__ void lior_neighbor_count_kernel(int n, const float* __restrict__ xyz,
                                            int32_t* __restrict__ count_out)
{
    const float radius2 = kLiorRadius * kLiorRadius;   // FIXED for every point, unlike DROR's per-point radius

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float pi[3] = { xyz[i * 3 + 0], xyz[i * 3 + 1], xyz[i * 3 + 2] };

        int32_t cnt = 0;
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            const float pj[3] = { xyz[j * 3 + 0], xyz[j * 3 + 1], xyz[j * 3 + 2] };
            if (squared_distance3(pi, pj) <= radius2) ++cnt;
        }
        count_out[i] = cnt;
    }
}

// ---------------------------------------------------------------------------
// lior_classify_kernel - outlier iff BOTH dim (intensity below threshold)
// AND locally sparse (count below kLiorKMin) - kernels.cuh file header
// point 3's "dim AND sparse" rule, the guard against throwing away a real
// but genuinely dark, densely-sampled surface.
// ---------------------------------------------------------------------------
__global__ void lior_classify_kernel(int n, const float* __restrict__ intensity,
                                      const int32_t* __restrict__ count,
                                      int32_t* __restrict__ mask_out)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const bool dim = intensity[i] < kLiorIntensityThresh;
        const bool sparse = count[i] < kLiorKMin;
        mask_out[i] = (dim && sparse) ? 1 : 0;
    }
}

// ===========================================================================
// Host launch wrappers - each owns the grid/block math + the mandatory
// post-launch error check (CLAUDE.md paragraph 6, rule 7). All six kernels
// share the SAME launch shape (one grid-stride pass over n points), so the
// reasoning is written out once here and referenced by the rest.
//
// block = 256 threads: a warp multiple (mandatory for full warps), a solid
//   default on sm_75..sm_89 (this repo's floor-to-ceiling architecture
//   range) that balances occupancy against per-thread register pressure -
//   sor_mean_knn_dist_kernel's largest, at kSorK=8 registers for best_d2[]
//   plus 8 for best_idx[], still leaves comfortable headroom at this block
//   size.
// grid = ceil(n/block), capped at 4096 blocks: enough blocks to fill every
//   SM on any current GPU many times over (this repo's owner's RTX 2080 has
//   46 SMs); the grid-stride loop inside each kernel absorbs whatever n
//   does not evenly divide, so an exact grid size is never required for
//   correctness, only for keeping launch overhead sane at this project's
//   n (a few thousand points needs at most a handful of blocks anyway).
// ===========================================================================
static inline int grid_for(int n, int block)
{
    int g = (n + block - 1) / block;
    return g > 4096 ? 4096 : g;
}

void launch_sor_mean_knn_dist(int n, const float* d_xyz, float* d_mean_dist)
{
    const int block = 256;
    sor_mean_knn_dist_kernel<<<grid_for(n, block), block>>>(n, d_xyz, d_mean_dist);
    CUDA_CHECK_LAST_ERROR("sor_mean_knn_dist_kernel launch");
}

void launch_sor_classify(int n, const float* d_mean_dist, float threshold, int32_t* d_mask_out)
{
    const int block = 256;
    sor_classify_kernel<<<grid_for(n, block), block>>>(n, d_mean_dist, threshold, d_mask_out);
    CUDA_CHECK_LAST_ERROR("sor_classify_kernel launch");
}

void launch_dror_neighbor_count(int n, const float* d_xyz, int32_t* d_count_out)
{
    const int block = 256;
    dror_neighbor_count_kernel<<<grid_for(n, block), block>>>(n, d_xyz, d_count_out);
    CUDA_CHECK_LAST_ERROR("dror_neighbor_count_kernel launch");
}

void launch_dror_classify(int n, const int32_t* d_count, int32_t* d_mask_out)
{
    const int block = 256;
    dror_classify_kernel<<<grid_for(n, block), block>>>(n, d_count, d_mask_out);
    CUDA_CHECK_LAST_ERROR("dror_classify_kernel launch");
}

void launch_lior_neighbor_count(int n, const float* d_xyz, int32_t* d_count_out)
{
    const int block = 256;
    lior_neighbor_count_kernel<<<grid_for(n, block), block>>>(n, d_xyz, d_count_out);
    CUDA_CHECK_LAST_ERROR("lior_neighbor_count_kernel launch");
}

void launch_lior_classify(int n, const float* d_intensity, const int32_t* d_count, int32_t* d_mask_out)
{
    const int block = 256;
    lior_classify_kernel<<<grid_for(n, block), block>>>(n, d_intensity, d_count, d_mask_out);
    CUDA_CHECK_LAST_ERROR("lior_classify_kernel launch");
}
