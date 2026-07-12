// ===========================================================================
// kernels.cu - GPU kernels for project 02.20 (LiDAR intensity calibration
//              across channels)
//
// Role in the project
// -------------------
// All __global__ (GPU) code lives here, together with the small host-side
// launch wrappers that own the grid/block math, and SECTION 8's shared
// host-only 16x16 solve (kernels.cuh SECTION 8's precedent: too small a
// problem for a meaningful GPU mapping - see THEORY.md "The GPU mapping").
//
// Pipeline (kernels.cuh SECTION 2-6 derive the math; this file implements
// the four GPU stages main.cu calls, per scan):
//   1) point_features_kernel  - MAP: forward-model inversion + voxel key.
//   2) bin_accumulate_kernel  - SCATTER-REDUCE: per-(voxel,channel) stats.
//   3) assemble_ls_kernel     - SCATTER-REDUCE: least-squares normal
//                                equations, one thread per voxel.
//   4) apply_gain_kernel      - MAP: the "reason to exist" correction.
// Between (3) and the gates, main.cu calls solve_channel_gains() (SECTION 8
// below) exactly once, host-side, on the (already GPU-vs-CPU-verified) A/b.
//
// Read this after: main.cu, kernels.cuh.  Read this before: reference_cpu.cpp.
// ===========================================================================

#include <vector>
#include <cmath>
#include <cstdlib>

#include "kernels.cuh"
#include "util/cuda_check.cuh"

// Repo-default block size (warp multiple; good occupancy on sm_75..sm_89 -
// see 02.01/02.06/08.01's identical launch-configuration reasoning).
static constexpr int kThreadsPerBlock = 256;

static inline int blocks_for(int count, int threads)
{
    return (count + threads - 1) / threads;
}

// ---------------------------------------------------------------------------
// point_features_kernel - see kernels.cuh SECTION 6 for the full contract.
//
// Thread-to-data mapping: thread i owns point i (a pure map - no point's
// output depends on any other point's, kernels.cuh SECTION 3/6's forward
// model is entirely local arithmetic).
//
// Memory behavior: xyz/intensity are read once per point (coalesced: thread
// i touches xyz[3i..3i+2], adjacent threads touch adjacent addresses);
// log_intensity/voxel_idx are written once per point, also coalesced. No
// shared memory: nothing is reused between threads.
// ---------------------------------------------------------------------------
__global__ void point_features_kernel(int n,
                                       const float* __restrict__ xyz,
                                       const float* __restrict__ intensity,
                                       GridBounds grid,
                                       float* __restrict__ log_intensity,
                                       int32_t* __restrict__ voxel_idx)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float p[3] = { xyz[i * 3 + 0], xyz[i * 3 + 1], xyz[i * 3 + 2] };

    // kernels.cuh SECTION 2: this project's OWN geometric model of the known
    // scene - never the ground-truth surf_id column (main.cu loads that
    // separately, for gates only).
    float normal[3];
    classify_normal_family(p, normal);

    const float r = point_range(p);
    const float r_safe = r > 1e-6f ? r : 1e-6f;   // guard: a point exactly at the sensor origin (never happens
                                                    // here - every surface is >= ~4.6 m away - but a cheap,
                                                    // honest guard against a future scene that gets closer)
    // cos(theta) = |direction . normal| = |p . normal| / r (kernels.cuh
    // SECTION 3's forward model; direction = p/r since the sensor is the
    // ray origin).
    const float cos_theta = fabsf(p[0] * normal[0] + p[1] * normal[1] + p[2] * normal[2]) / r_safe;

    const float f_r = range_falloff(r);
    log_intensity[i] = corrected_log_intensity(intensity[i], f_r, cos_theta);

    const int32_t ix = voxel_coord(p[0], grid.leaf);
    const int32_t iy = voxel_coord(p[1], grid.leaf);
    const int32_t iz = voxel_coord(p[2], grid.leaf);
    voxel_idx[i] = flat_voxel_index(ix, iy, iz, grid);
}

void launch_point_features(int n, const float* d_xyz, const float* d_intensity,
                            GridBounds grid, float* d_log_intensity, int32_t* d_voxel_idx)
{
    const int blocks = blocks_for(n, kThreadsPerBlock);
    point_features_kernel<<<blocks, kThreadsPerBlock>>>(n, d_xyz, d_intensity, grid,
                                                          d_log_intensity, d_voxel_idx);
    CUDA_CHECK_LAST_ERROR("point_features_kernel launch");
}

// ---------------------------------------------------------------------------
// bin_accumulate_kernel - see kernels.cuh SECTION 6.
//
// Thread-to-data mapping: thread i owns point i; it atomicAdd's into TWO
// small global destination arrays keyed by (voxel_idx[i], channel[i]) - the
// classic SCATTER-REDUCE / histogram pattern (01.09's radial_bin_kernel is
// the 1-D precedent; this is the same idea 2-D keyed). Multiple points
// (different azimuth steps, sometimes different channels) land in the same
// (voxel,channel) slot, so the destination write pattern is NOT one-to-one -
// atomics are required, unlike point_features_kernel's pure map.
//
// Why atomicAdd on float (not double): kernels.cuh's independence-ruling
// discussion (reference_cpu.cpp) - the GPU path's atomic-order-dependent
// float sum is compared against an INDEPENDENT, double-accumulated,
// sequential-order CPU sum within a measured tolerance (main.cu's
// VERIFY(accumulate) stage), the same "give the oracle better precision"
// asymmetry 01.09/02.01 use for their own atomic reductions.
// ---------------------------------------------------------------------------
__global__ void bin_accumulate_kernel(int n,
                                       const int32_t* __restrict__ channel,
                                       const float* __restrict__ xyz,
                                       const float* __restrict__ log_intensity,
                                       const int32_t* __restrict__ voxel_idx,
                                       int numVoxels,
                                       float* __restrict__ sum_log,
                                       int32_t* __restrict__ count,
                                       int32_t* __restrict__ voxel_family)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int32_t vox = voxel_idx[i];
    if (vox < 0 || vox >= numVoxels) return;   // outside the grid: should not happen (main.cu sizes the
                                                 // grid from this SAME point cloud with margin) - a silent
                                                 // skip here is a correctness bug main.cu's [info] point-
                                                 // count reconciliation would catch, not a policy choice.

    const int32_t ch = channel[i];
    atomicAdd(&sum_log[vox * kNumBeams + ch], log_intensity[i]);
    atomicAdd(&count[vox * kNumBeams + ch], 1);

    // voxel_family: a RACY but BENIGN plain store (kernels.cuh SECTION 6 -
    // every point sharing a voxel on this project's scene comes from the
    // SAME plane by construction, so every possible writer stores the
    // IDENTICAL value; no atomic needed). Recomputed here rather than piped
    // from point_features_kernel to avoid a third [n]-sized array for a
    // value that only needs to exist once per VOXEL, not once per point.
    const float p[3] = { xyz[i * 3 + 0], xyz[i * 3 + 1], xyz[i * 3 + 2] };
    float normal_unused[3];
    voxel_family[vox] = classify_normal_family(p, normal_unused);
}

void launch_bin_accumulate(int n, const int32_t* d_channel, const float* d_xyz,
                            const float* d_log_intensity, const int32_t* d_voxel_idx,
                            int numVoxels, float* d_sum_log, int32_t* d_count, int32_t* d_voxel_family)
{
    const int blocks = blocks_for(n, kThreadsPerBlock);
    bin_accumulate_kernel<<<blocks, kThreadsPerBlock>>>(n, d_channel, d_xyz, d_log_intensity, d_voxel_idx,
                                                          numVoxels, d_sum_log, d_count, d_voxel_family);
    CUDA_CHECK_LAST_ERROR("bin_accumulate_kernel launch");
}

// ---------------------------------------------------------------------------
// assemble_ls_kernel - see kernels.cuh SECTION 5/6.
//
// Thread-to-data mapping: thread v owns voxel v (numVoxels threads - most
// exit immediately: a real scan's occupied-but-not-shared voxels outnumber
// shared ones several-to-one, kernels.cuh SECTION 4). A qualifying voxel
// (>= 2 channels present) computes its own k_v x k_v centering-projector
// contribution ENTIRELY IN REGISTERS (chans[]/y[] are small fixed-size
// local arrays, k_v <= kNumBeams = 16 - no shared memory needed, unlike a
// block-tree reduction: there is nothing to reduce WITHIN a voxel, only
// across voxels, which the atomics below handle), then atomicAdd's its
// O(k_v^2) terms into the GLOBAL 16x16 A / 16 b - contended only by the
// handful of voxels that qualify (measured: a few hundred at most on this
// project's scale), the same "small global atomic destination" pattern
// 01.09's roi_mean_reduce_kernel uses for a single accumulator, generalized
// here to 256+16 destinations (still tiny next to a real reduction kernel's
// footprint).
//
// This kernel does NOT call kernels.cuh's channel_ls_accumulate() directly:
// that shared helper accumulates into a LOCAL (non-atomic) A/b, correct for
// ONE voxel at a time on a single thread (reference_cpu.cpp's sequential CPU
// twin calls it exactly that way); here, MANY threads (voxels) accumulate
// into the SAME global A/b concurrently, so this kernel re-derives the
// IDENTICAL formula with atomicAdd at each term - the same "device carries
// its own atomic transcription of a shared formula" pattern project 02.01's
// kernels.cuh documents for its own pack_voxel_key/spatial_hash constants.
// ---------------------------------------------------------------------------
__global__ void assemble_ls_kernel(int numVoxels,
                                    const float* __restrict__ sum_log,
                                    const int32_t* __restrict__ count,
                                    float* __restrict__ A,
                                    float* __restrict__ b,
                                    int32_t* __restrict__ shared_voxel_count)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= numVoxels) return;

    int32_t chans[kNumBeams];
    float y[kNumBeams];
    int k = 0;
    for (int c = 0; c < kNumBeams; ++c) {
        const int32_t cnt = count[v * kNumBeams + c];
        if (cnt > 0) {
            chans[k] = c;
            y[k] = sum_log[v * kNumBeams + c] / static_cast<float>(cnt);   // this (voxel,channel)'s mean
            ++k;
        }
    }
    if (k < 2) return;   // not a SHARED voxel (kernels.cuh SECTION 4/5) - contributes nothing

    atomicAdd(shared_voxel_count, 1);

    // ybar_v (kernels.cuh SECTION 5): this voxel's own baseline, computed
    // from ITS OWN data only - no external reflectivity knowledge used.
    float ybar = 0.0f;
    for (int i = 0; i < k; ++i) ybar += y[i];
    ybar /= static_cast<float>(k);

    const float inv_k = 1.0f / static_cast<float>(k);
    for (int i = 0; i < k; ++i) {
        const int32_t ci = chans[i];
        const float r_i = y[i] - ybar;
        atomicAdd(&b[ci], r_i);
        for (int j = 0; j < k; ++j) {
            const int32_t cj = chans[j];
            const float pij = (i == j) ? (1.0f - inv_k) : (-inv_k);
            atomicAdd(&A[ci * kNumBeams + cj], pij);
        }
    }
}

void launch_assemble_ls(int numVoxels, const float* d_sum_log, const int32_t* d_count,
                         float* d_A, float* d_b, int32_t* d_shared_voxel_count)
{
    const int blocks = blocks_for(numVoxels, kThreadsPerBlock);
    assemble_ls_kernel<<<blocks, kThreadsPerBlock>>>(numVoxels, d_sum_log, d_count,
                                                       d_A, d_b, d_shared_voxel_count);
    CUDA_CHECK_LAST_ERROR("assemble_ls_kernel launch");
}

// ---------------------------------------------------------------------------
// apply_gain_kernel - see kernels.cuh SECTION 6. A pure map, the same
// pattern as point_features_kernel; the "reason this project exists" step
// (01.09's correction_kernel is the exact precedent: divide out the
// recovered per-pixel/per-channel nuisance).
// ---------------------------------------------------------------------------
__global__ void apply_gain_kernel(int n,
                                   const float* __restrict__ raw_intensity,
                                   const int32_t* __restrict__ channel,
                                   const float* __restrict__ gain,
                                   float* __restrict__ corrected)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int32_t ch = channel[i];
    corrected[i] = raw_intensity[i] / gain[ch];
}

void launch_apply_gain(int n, const float* d_raw_intensity, const int32_t* d_channel,
                        const float* d_gain, float* d_corrected)
{
    const int blocks = blocks_for(n, kThreadsPerBlock);
    apply_gain_kernel<<<blocks, kThreadsPerBlock>>>(n, d_raw_intensity, d_channel, d_gain, d_corrected);
    CUDA_CHECK_LAST_ERROR("apply_gain_kernel launch");
}

// ===========================================================================
// SECTION 8 - solve_channel_gains (kernels.cuh SECTION 8's contract; the
// 01.09 SECTION-5 "shared, host-only, no GPU mapping" precedent).
//
// Algorithm: kernels.cuh SECTION 5's graph-Laplacian argument is a claim
// about the WHOLE off-diagonal structure of A, not just its diagonal - a
// channel with A[c][c] > 0 but whose only edges lead into a SMALL, otherwise
// isolated cluster of channels (a real possibility once a gate restricts the
// shared-voxel population to a single material, THEORY.md "Numerical
// considerations" documents the measured case that caught this) is only
// observable RELATIVE TO THAT CLUSTER - comparing it against channels in a
// different, unconnected cluster is meaningless (there is a second,
// per-cluster gauge freedom the single global ridge term does not resolve,
// which shows up as a near-zero eigenvalue and a wildly amplified solve, not
// a clean singularity). This function therefore finds ALL connected
// components of the channel graph (union-find over A's nonzero off-diagonal
// entries, restricted to channels with a nonzero diagonal), solves ONLY the
// LARGEST component, and flags every channel outside it - including
// genuinely isolated ones (kernels.cuh SECTION 5's original, simpler case)
// AND channels stranded in a smaller separate cluster - as UNOBSERVABLE
// rather than emitting an ill-conditioned number for either kind.
//
// Within the chosen component: add the gauge-fixing ridge term
// lambda/m * ones(m,m) to the reduced m x m system (kernels.cuh SECTION 5
// derives why this exactly pins mean(log_gain)=0 without perturbing the
// rest of the solution - b is provably already orthogonal to the all-ones
// vector, and a single connected component has EXACTLY one such null
// direction), then Gaussian elimination with partial pivoting (double
// precision internally - m <= 16, negligible cost).
// ===========================================================================
int solve_channel_gains(const float* A, const float* b,
                         float* out_log_gain, int32_t* out_observable)
{
    for (int c = 0; c < kNumBeams; ++c) {
        out_observable[c] = 0;
        out_log_gain[c] = 0.0f;   // sentinel default for unobservable channels (caller must check out_observable)
    }

    // ---- connected components (union-find, kNumBeams <= 16: a plain O(n^2)
    // edge scan + naive union is simpler to read than any fancier structure
    // at this size, and costs nothing measurable). ------------------------
    int parent[kNumBeams];
    for (int c = 0; c < kNumBeams; ++c) parent[c] = c;
    // find_root: iterative (no recursion - this is still device-adjacent
    // host code compiled by nvcc; keep it simple and stack-free).
    auto find_root = [&](int x) {
        while (parent[x] != x) x = parent[x];
        return x;
    };
    bool has_diag[kNumBeams];
    for (int c = 0; c < kNumBeams; ++c) has_diag[c] = A[c * kNumBeams + c] > kObservableEps;
    for (int i = 0; i < kNumBeams; ++i) {
        if (!has_diag[i]) continue;
        for (int j = i + 1; j < kNumBeams; ++j) {
            if (!has_diag[j]) continue;
            if (std::fabs(A[i * kNumBeams + j]) > kObservableEps) {
                const int ri = find_root(i), rj = find_root(j);
                if (ri != rj) parent[ri] = rj;   // union by pointer-chase (fine at n<=16)
            }
        }
    }
    // Tally component sizes among has_diag channels; pick the LARGEST root.
    int comp_size[kNumBeams] = { 0 };
    for (int c = 0; c < kNumBeams; ++c) if (has_diag[c]) comp_size[find_root(c)]++;
    int best_root = -1, best_size = 0;
    for (int c = 0; c < kNumBeams; ++c) if (comp_size[c] > best_size) { best_size = comp_size[c]; best_root = c; }

    int idx_map[kNumBeams];
    int m = 0;
    for (int c = 0; c < kNumBeams; ++c) {
        const bool observable = has_diag[c] && best_root >= 0 && find_root(c) == best_root;
        out_observable[c] = observable ? 1 : 0;
        if (observable) idx_map[m++] = c;
    }
    if (m == 0) return 0;

    // Build the reduced, gauge-ridged system in double precision.
    std::vector<double> Ar(static_cast<size_t>(m) * m);
    std::vector<double> br(static_cast<size_t>(m));
    const double ridge = static_cast<double>(kGaugeLambda) / static_cast<double>(m);
    for (int i = 0; i < m; ++i) {
        const int ci = idx_map[i];
        br[static_cast<size_t>(i)] = static_cast<double>(b[ci]);
        for (int j = 0; j < m; ++j) {
            const int cj = idx_map[j];
            Ar[static_cast<size_t>(i) * m + j] = static_cast<double>(A[ci * kNumBeams + cj]) + ridge;
        }
    }

    // Gaussian elimination with partial pivoting (textbook; m <= 16, so this
    // O(m^3) solve is microseconds - see kernels.cuh SECTION 8's "no GPU
    // mapping" note).
    for (int col = 0; col < m; ++col) {
        int piv = col;
        double best = std::fabs(Ar[static_cast<size_t>(col) * m + col]);
        for (int r = col + 1; r < m; ++r) {
            const double v = std::fabs(Ar[static_cast<size_t>(r) * m + col]);
            if (v > best) { best = v; piv = r; }
        }
        if (piv != col) {
            for (int cc = 0; cc < m; ++cc)
                std::swap(Ar[static_cast<size_t>(col) * m + cc], Ar[static_cast<size_t>(piv) * m + cc]);
            std::swap(br[static_cast<size_t>(col)], br[static_cast<size_t>(piv)]);
        }
        double diag = Ar[static_cast<size_t>(col) * m + col];
        if (std::fabs(diag) < 1e-15) diag = (diag >= 0.0) ? 1e-15 : -1e-15;   // degenerate guard (should not
                                                                                // trigger given the ridge term)
        for (int r = col + 1; r < m; ++r) {
            const double factor = Ar[static_cast<size_t>(r) * m + col] / diag;
            if (factor == 0.0) continue;
            for (int cc = col; cc < m; ++cc)
                Ar[static_cast<size_t>(r) * m + cc] -= factor * Ar[static_cast<size_t>(col) * m + cc];
            br[static_cast<size_t>(r)] -= factor * br[static_cast<size_t>(col)];
        }
    }
    std::vector<double> x(static_cast<size_t>(m), 0.0);
    for (int r = m - 1; r >= 0; --r) {
        double sum = br[static_cast<size_t>(r)];
        for (int cc = r + 1; cc < m; ++cc) sum -= Ar[static_cast<size_t>(r) * m + cc] * x[static_cast<size_t>(cc)];
        double diag = Ar[static_cast<size_t>(r) * m + r];
        if (std::fabs(diag) < 1e-15) diag = (diag >= 0.0) ? 1e-15 : -1e-15;
        x[static_cast<size_t>(r)] = sum / diag;
    }

    for (int i = 0; i < m; ++i) out_log_gain[idx_map[i]] = static_cast<float>(x[static_cast<size_t>(i)]);
    return m;
}
