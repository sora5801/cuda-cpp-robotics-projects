// ===========================================================================
// kernels.cu — GPU kernels for project 02.19 (PointPillars/CenterPoint
//              voxelization + scatter kernels feeding TensorRT)
//
// Role in the project
// -------------------
// All __global__ (GPU) code lives here, plus the small host-side launch
// wrappers that own the grid/block math (CLAUDE.md §6.1 rule 2: the launch
// reasoning sits beside the code it configures). kernels.cuh is the
// contract every kernel here implements; read it first for the WHY behind
// every constant and layout decision below.
//
// Pipeline order this file implements, top to bottom (main.cu calls these
// launchers in exactly this order for the "sorted" Method-B path; the
// atomic Method-A path only runs the first few, for the determinism study):
//   keys -> [Method A: atomic_bin]  or  [Method B: sort_and_compact -> sorted_bin]
//        -> pfn_stats (mean + kept count) -> augment_features -> pfn_lite
//        -> scatter -> (gather, for the roundtrip check)
//        -> conv3x3 (smooth) -> elementwise_mul (gate) -> conv3x3 (sharpen)
//        -> peak_extract  (NMS itself runs on the host — see main.cu)
//
// Read this after: main.cu, kernels.cuh.  Read this before: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK / CUDA_CHECK_LAST_ERROR

// Thrust: header-only pieces of the CUDA Toolkit (CLAUDE.md §5 — no extra
// .lib needed, see build/*.vcxproj's comment). Reused, almost verbatim,
// from 02.01's Method B (cited throughout this file) — stable_sort_by_key
// for the pillar/voxel-key sort, copy_if for the invalid-point filter AND
// the segment-boundary compaction, reduce for the occupied-cell count.
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

// ---------------------------------------------------------------------------
// Small predicate functors for the Thrust calls below. CUDA 13.3's Thrust
// dropped thrust::identity (see 02.01's kernels.cu comment on the same
// issue) — hand-written two-line functors are simpler than chasing the
// CCCL rename and just as self-explanatory.
// ---------------------------------------------------------------------------
struct is_valid_key {                                  // "this point is inside the BEV/vertical window"
    __host__ __device__ bool operator()(int k) const { return k >= 0; }
};
struct is_nonzero {                                     // "this boundary-mask entry marks a segment start"
    __host__ __device__ bool operator()(int x) const { return x != 0; }
};

// ===========================================================================
// Device-side transcription of kernels.cuh's shared key-arithmetic helpers
// (pillar_key_of / voxel_key_of / z_in_range). WHY DUPLICATED: those header
// functions are plain (no __host__/__device__) so cl.exe (reference_cpu.cpp)
// can compile them too, which makes them HOST-only under nvcc's rules and
// therefore uncallable from a __global__ kernel — the exact situation
// 02.01's kernels.cuh file header names and permits as "shared token-for-
// token transcription", PROVIDED an independent gate catches drift. That
// gate is main.cu's VERIFY(keys): every point's GPU key (via the functions
// below) is compared, bit-exact, against reference_cpu.cpp's key (via
// kernels.cuh's shared host functions).
// ===========================================================================
__device__ __forceinline__ int device_pillar_key(float x, float y, float z)
{
    if (z < kZMin || z > kZMax) return -1;
    const int ix = static_cast<int>(floorf((x - kXMin) / kPillarSizeM));
    const int iy = static_cast<int>(floorf((y - kYMin) / kPillarSizeM));
    if (ix < 0 || ix >= kGridNX || iy < 0 || iy >= kGridNY) return -1;
    return iy * kGridNX + ix;
}

__device__ __forceinline__ int device_voxel_key(float x, float y, float z)
{
    const int ix = static_cast<int>(floorf((x - kXMin) / kPillarSizeM));
    const int iy = static_cast<int>(floorf((y - kYMin) / kPillarSizeM));
    const int iz = static_cast<int>(floorf((z - kZMin) / kZBandM));
    if (ix < 0 || ix >= kGridNX || iy < 0 || iy >= kGridNY || iz < 0 || iz >= kNumZBins) return -1;
    return iz * kNumPillars + (iy * kGridNX + ix);
}

// ===========================================================================
// 1) KEYS — one thread per point (the "voxelization" half of the catalog
//    bullet: deciding which cell a point belongs to).
// ===========================================================================
__global__ void compute_pillar_keys_kernel(int n, const float* __restrict__ points, int* __restrict__ keys)
{
    // Grid-stride loop (08.01/02.01 idiom): correct for any n, lets the
    // caller pick the grid size for occupancy instead of ceil(n/block).
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        // points[] is interleaved (x,y,z,intensity) — 4 coalesced floats per
        // point per warp lane, same access pattern 02.01's compute_keys uses
        // for its xyz-only layout (one extra float here, intensity, unused
        // by the key itself but loaded together for locality).
        const float x = points[i * 4 + 0];
        const float y = points[i * 4 + 1];
        const float z = points[i * 4 + 2];
        keys[i] = device_pillar_key(x, y, z);
    }
}

__global__ void compute_voxel_keys_kernel(int n, const float* __restrict__ points, int* __restrict__ keys)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float x = points[i * 4 + 0];
        const float y = points[i * 4 + 1];
        const float z = points[i * 4 + 2];
        keys[i] = device_voxel_key(x, y, z);
    }
}

// ===========================================================================
// 2) METHOD A — atomic per-pillar slot claim.
// ===========================================================================
__global__ void reset_counts_kernel(unsigned int* __restrict__ point_count, int num_cells)
{
    // One thread per CELL (not per point) — num_cells is small (40,000),
    // this runs in well under a microsecond of kernel time; re-run before
    // every fresh Method-A pass (main.cu calls this once per input-order
    // permutation in the cap_truncation determinism study).
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < num_cells; c += gridDim.x * blockDim.x)
        point_count[c] = 0u;
}

// atomic_bin_kernel — Method A's heart. Every point in an over-full pillar
// races every other point in that SAME pillar for slots 0..cap_n-1; the
// hardware serializes the atomicAdd calls in SOME order (never specified by
// the CUDA memory model — see kernels.cuh's file header for the full
// determinism story main.cu's cap_truncation gate measures), and whichever
// point's atomicAdd is serialized last among the first cap_n wins a slot;
// every later one silently loses (the counter still increments past cap_n,
// so point_count[cell] always reports the TRUE arrival count, never
// clamped — main.cu relies on this to detect overflow at all).
__global__ void atomic_bin_kernel(int n, const float* __restrict__ points,
                                  const int* __restrict__ keys, PillarBinGPU bin)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const int key = keys[i];
        if (key < 0) continue;   // out of the BEV/vertical window — never binned

        // atomicAdd returns the value BEFORE the add — i.e. "my claimed
        // slot index" — a classic GPU counting-allocator pattern (the same
        // idea 02.01's hash-insert claim loop and 07.09's atomic compaction
        // both use, specialized here to a dense per-cell counter instead of
        // a hash-table probe or a global compaction index).
        const unsigned int slot = atomicAdd(&bin.point_count[key], 1u);
        if (static_cast<int>(slot) < bin.cap_n) {
            float* dst = &bin.raw_points[static_cast<size_t>(key) * bin.cap_n * 4 + static_cast<size_t>(slot) * 4];
            dst[0] = points[i * 4 + 0];
            dst[1] = points[i * 4 + 1];
            dst[2] = points[i * 4 + 2];
            dst[3] = points[i * 4 + 3];
        }
        // slot >= cap_n: this point LOSES the race for a stored slot — the
        // drop. No write happens; the point is gone from the tensor, exactly
        // as if it had never arrived, except point_count still remembers it did.
    }
}

// ===========================================================================
// 3) METHOD B — sort + fixed-order truncation (the generic machinery,
//    reused unchanged for BOTH pillar and voxel keys — kernels.cuh's file
//    header "same machinery, different key function").
// ===========================================================================
__global__ void mark_boundaries_kernel(int n, const int* __restrict__ keys_sorted,
                                       int* __restrict__ is_start)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        // Position 0 always starts a segment; otherwise compare to the
        // PREVIOUS sorted position — a single neighbor read, no shared
        // memory needed at this scale (identical to 02.01's kernel).
        is_start[i] = (i == 0 || keys_sorted[i] != keys_sorted[i - 1]) ? 1 : 0;
    }
}

__global__ void gather_occupied_cell_kernel(int num_occupied, const int* __restrict__ seg_start,
                                            const int* __restrict__ keys_sorted,
                                            int* __restrict__ occupied_cell)
{
    for (int s = blockIdx.x * blockDim.x + threadIdx.x; s < num_occupied; s += gridDim.x * blockDim.x)
        occupied_cell[s] = keys_sorted[seg_start[s]];   // every point in this run shares this key by construction
}

// sorted_bin_kernel — one thread per OCCUPIED SEGMENT (not per point): walks
// its run in SORTED-ARRAY order, which — because the sort upstream is
// STABLE — is exactly ASCENDING ORIGINAL POINT INDEX order. Truncating to
// the FIRST cap_n points of a FIXED order is what makes Method B bit-exact
// (kernels.cuh's file header derives this; reference_cpu.cpp's
// sorted_pipeline_cpu reproduces the identical rule with std::stable_sort).
__global__ void sorted_bin_kernel(int num_occupied, const int* __restrict__ seg_start, int n_sorted,
                                  const int* __restrict__ idx_sorted, const float* __restrict__ points,
                                  const int* __restrict__ occupied_cell, PillarBinGPU bin)
{
    (void)n_sorted;   // seg_start[num_occupied] is the sentinel end (set by the caller); kept in the
                      // signature for documentation symmetry with reference_cpu.cpp's twin.
    for (int s = blockIdx.x * blockDim.x + threadIdx.x; s < num_occupied; s += gridDim.x * blockDim.x) {
        const int start = seg_start[s];
        const int end   = seg_start[s + 1];             // always valid: the caller sized seg_start to n+1
        const int run_len = end - start;
        const int cell = occupied_cell[s];
        const int kept = run_len < bin.cap_n ? run_len : bin.cap_n;

        bin.point_count[cell] = static_cast<unsigned int>(run_len);   // TRUE arrival count, never clamped
        for (int k = 0; k < kept; ++k) {
            const int idx = idx_sorted[start + k];       // original point index, ascending (stability)
            float* dst = &bin.raw_points[static_cast<size_t>(cell) * bin.cap_n * 4 + static_cast<size_t>(k) * 4];
            dst[0] = points[idx * 4 + 0];
            dst[1] = points[idx * 4 + 1];
            dst[2] = points[idx * 4 + 2];
            dst[3] = points[idx * 4 + 3];
        }
    }
}

// ===========================================================================
// 4) FEATURES + PFN-LITE — turning kept raw points into the network's
//    9-D-per-point input tensor, then a per-pillar feature vector.
// ===========================================================================

// pfn_stats_kernel — one thread per occupied pillar: mean (x,y,z) over its
// kept points, plus the kept count. Sequential per-thread loop over <=32
// points: at this scale (cap_n=32) a parallel reduction would cost MORE
// (block launch/sync overhead for 32 elements) than it saves — the
// "small, bounded work per unit -> don't parallelize the inner loop, only
// the outer one" call THEORY.md "The GPU mapping" discusses explicitly.
__global__ void pfn_stats_kernel(int num_occupied, const int* __restrict__ occupied_cell,
                                 PillarBinGPU bin, float* __restrict__ mean_xyz_out,
                                 unsigned int* __restrict__ kept_count_out)
{
    for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < num_occupied; p += gridDim.x * blockDim.x) {
        const int cell = occupied_cell[p];
        const unsigned int arrived = bin.point_count[cell];
        const unsigned int kept = arrived < static_cast<unsigned int>(bin.cap_n)
                                 ? arrived : static_cast<unsigned int>(bin.cap_n);
        float sx = 0.0f, sy = 0.0f, sz = 0.0f;
        for (unsigned int k = 0; k < kept; ++k) {
            const float* pt = &bin.raw_points[static_cast<size_t>(cell) * bin.cap_n * 4 + static_cast<size_t>(k) * 4];
            sx += pt[0]; sy += pt[1]; sz += pt[2];
        }
        const float inv = kept > 0 ? 1.0f / static_cast<float>(kept) : 0.0f;
        mean_xyz_out[p * 3 + 0] = sx * inv;
        mean_xyz_out[p * 3 + 1] = sy * inv;
        mean_xyz_out[p * 3 + 2] = sz * inv;
        kept_count_out[p] = kept;
    }
}

// augment_features_kernel — one thread per (occupied pillar, point SLOT)
// pair: the classic PointPillars 9-D feature (kernels.cuh derives every
// term; THEORY.md "The math" derives the geometry). Slots >= kept_count are
// ZERO-PADDED — the fixed [num_occupied, cap_n, 9] tensor shape a real
// network's static input requires (README "System context" / "Limitations"
// name the real network's further padding to a fixed P_max across frames,
// out of this project's scope).
__global__ void augment_features_kernel(int num_occupied, const int* __restrict__ occupied_cell,
                                        PillarBinGPU bin, const float* __restrict__ mean_xyz,
                                        float* __restrict__ features_out)
{
    const int total = num_occupied * bin.cap_n;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < total; t += gridDim.x * blockDim.x) {
        const int p = t / bin.cap_n;      // which pillar
        const int slot = t % bin.cap_n;   // which point slot within it
        const int cell = occupied_cell[p];
        const unsigned int arrived = bin.point_count[cell];
        const unsigned int kept = arrived < static_cast<unsigned int>(bin.cap_n)
                                 ? arrived : static_cast<unsigned int>(bin.cap_n);
        float* feat = &features_out[static_cast<size_t>(p) * bin.cap_n * kNumPointFeatures
                                   + static_cast<size_t>(slot) * kNumPointFeatures];
        if (static_cast<unsigned int>(slot) >= kept) {
            #pragma unroll
            for (int d = 0; d < kNumPointFeatures; ++d) feat[d] = 0.0f;   // zero-pad: no point here
            continue;
        }
        const float* pt = &bin.raw_points[static_cast<size_t>(cell) * bin.cap_n * 4 + static_cast<size_t>(slot) * 4];
        const float x = pt[0], y = pt[1], z = pt[2], inten = pt[3];

        // Pillar geometric center (constant per cell, independent of which
        // points happen to be inside it — the "absolute anchor" term).
        const int ix = cell % kGridNX;
        const int iy = cell / kGridNX;
        const float pcx = kXMin + (static_cast<float>(ix) + 0.5f) * kPillarSizeM;
        const float pcy = kYMin + (static_cast<float>(iy) + 0.5f) * kPillarSizeM;

        feat[0] = x;
        feat[1] = y;
        feat[2] = z;
        feat[3] = inten;
        feat[4] = x - mean_xyz[p * 3 + 0];   // xc: offset from this pillar's KEPT-point mean
        feat[5] = y - mean_xyz[p * 3 + 1];   // yc
        feat[6] = z - mean_xyz[p * 3 + 2];   // zc
        feat[7] = x - pcx;                   // xp: offset from the pillar's fixed geometric center
        feat[8] = y - pcy;                   // yp
    }
}

// pfn_lite_kernel — the tiny FIXED-weight PFN stand-in (kernels.cuh
// documents all kPfnChannels=6 output channels). Deliberately reads ONLY
// the standardized 9-D feature tensor, never `bin` or raw points directly —
// exactly the boundary a real learned PFN sits behind (it never sees raw
// LiDAR either; the augmented-feature tensor IS its input contract).
__global__ void pfn_lite_kernel(int num_occupied, const float* __restrict__ features,
                                const unsigned int* __restrict__ kept_count,
                                const float* __restrict__ lin_w, const float* __restrict__ lin_b,
                                float* __restrict__ pillar_feat_out)
{
    for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < num_occupied; p += gridDim.x * blockDim.x) {
        const unsigned int kept = kept_count[p];
        const float occupancy = static_cast<float>(kept) / static_cast<float>(kMaxPointsPerPillar);

        float z_min = 1e30f, z_max = -1e30f;
        float lin_acc[kPfnLinOut];
        #pragma unroll
        for (int c = 0; c < kPfnLinOut; ++c) lin_acc[c] = -1e30f;   // running max, pre-ReLU-floor at 0 below

        for (unsigned int k = 0; k < kept; ++k) {
            const float* f = &features[static_cast<size_t>(p) * kMaxPointsPerPillar * kNumPointFeatures
                                      + static_cast<size_t>(k) * kNumPointFeatures];
            const float z = f[2];
            if (z < z_min) z_min = z;
            if (z > z_max) z_max = z;

            // The "linear layer": one dot product per output channel. Small
            // (kPfnLinOut * kNumPointFeatures = 36 multiply-adds) — this is
            // where a REAL PFN's learned weight matrix would do the same
            // shape of work at much larger channel counts (64-128 typical).
            #pragma unroll
            for (int c = 0; c < kPfnLinOut; ++c) {
                float acc = lin_b[c];
                #pragma unroll
                for (int d = 0; d < kNumPointFeatures; ++d)
                    acc += lin_w[c * kNumPointFeatures + d] * f[d];
                const float relu = acc > 0.0f ? acc : 0.0f;
                // MAX-POOL across points: this is the permutation-invariance
                // step (THEORY.md "The math" — the Deep Sets argument) — the
                // pillar's feature does not depend on which SLOT a point
                // landed in, only on the SET of kept points (which, per
                // kernels.cuh's file header, is exactly where the cap
                // truncation policy re-enters: Method A's slot assignment
                // changes which points are IN that set).
                if (relu > lin_acc[c]) lin_acc[c] = relu;
            }
        }
        const float height_extent = kept > 0 ? (z_max - z_min) / kHeightNormM : 0.0f;
        const float height_extent_clamped = height_extent < 0.0f ? 0.0f : (height_extent > 1.0f ? 1.0f : height_extent);

        float* out = &pillar_feat_out[static_cast<size_t>(p) * kPfnChannels];
        out[0] = occupancy;
        out[1] = height_extent_clamped;
        #pragma unroll
        for (int c = 0; c < kPfnLinOut; ++c) out[2 + c] = kept > 0 ? lin_acc[c] : 0.0f;
    }
}

// ===========================================================================
// 5) SCATTER / GATHER — the catalog's second named kernel: sparse pillar
//    list -> dense [C,H,W] canvas a conv head can read (and back, for the
//    roundtrip bookkeeping check).
// ===========================================================================
__global__ void scatter_kernel(int num_occupied, const int* __restrict__ occupied_cell,
                               const float* __restrict__ pillar_feat, float* __restrict__ canvas)
{
    for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < num_occupied; p += gridDim.x * blockDim.x) {
        const int cell = occupied_cell[p];
        const int ix = cell % kGridNX;
        const int iy = cell / kGridNX;
        // NCHW layout (channel-major planes): this pillar's kPfnChannels
        // writes land in kPfnChannels DIFFERENT, far-apart planes — NOT
        // coalesced across channels within one thread, the price of NCHW.
        // We pay it here (num_occupied writes, once per frame) so the conv
        // stage below (H*W*9 reads EVERY pass, run twice) gets the layout
        // it wants instead: each conv thread's 3x3 neighborhood is 9
        // CONTIGUOUS-ish reads within ONE channel plane. THEORY.md "The GPU
        // mapping" names this trade explicitly — an HWC canvas would flip
        // which stage pays the coalescing cost, and the conv stage runs far
        // more times than the scatter, so NCHW wins here.
        #pragma unroll
        for (int c = 0; c < kPfnChannels; ++c) {
            canvas[static_cast<size_t>(c) * kGridNY * kGridNX + static_cast<size_t>(iy) * kGridNX + ix]
                = pillar_feat[static_cast<size_t>(p) * kPfnChannels + c];
        }
    }
}

__global__ void gather_kernel(int num_occupied, const int* __restrict__ occupied_cell,
                              const float* __restrict__ canvas, float* __restrict__ gathered_out)
{
    for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < num_occupied; p += gridDim.x * blockDim.x) {
        const int cell = occupied_cell[p];
        const int ix = cell % kGridNX;
        const int iy = cell / kGridNX;
        #pragma unroll
        for (int c = 0; c < kPfnChannels; ++c) {
            gathered_out[static_cast<size_t>(p) * kPfnChannels + c]
                = canvas[static_cast<size_t>(c) * kGridNY * kGridNX + static_cast<size_t>(iy) * kGridNX + ix];
        }
    }
}

// ===========================================================================
// 6) THE TOY HEAD — hand-rolled 3x3 conv-as-stencil, run twice with
//    DESIGNED (not trained) weights, plus an elementwise occupancy gate.
// ===========================================================================

// conv3x3_kernel — one thread per OUTPUT PIXEL (iy,ix): a 3x3 weighted sum
// over its own neighborhood, zero-padded at the canvas boundary (a pillar
// just outside the grid contributes 0 — physically "nothing detected
// there," the simplest honest boundary policy; THEORY.md "Numerical
// considerations" notes the alternative, edge-replication, and why it is
// not needed at this grid size). This is the STENCIL pattern: each output
// depends on a small, FIXED neighborhood of inputs — the same access
// pattern as a finite-difference PDE solver's Laplacian (this repo's
// stencil taxonomy, cited in THEORY.md), specialized to a 3x3 image kernel.
__global__ void conv3x3_kernel(const float* __restrict__ in_plane, int h, int w,
                               const float* __restrict__ kernel3x3, float bias,
                               float* __restrict__ out_plane)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= w || iy >= h) return;

    float acc = bias;
    #pragma unroll
    for (int ky = -1; ky <= 1; ++ky) {
        #pragma unroll
        for (int kx = -1; kx <= 1; ++kx) {
            const int nx = ix + kx, ny = iy + ky;
            const float v = (nx >= 0 && nx < w && ny >= 0 && ny < h) ? in_plane[ny * w + nx] : 0.0f;
            acc += v * kernel3x3[(ky + 1) * 3 + (kx + 1)];
        }
    }
    out_plane[iy * w + ix] = acc;
}

__global__ void elementwise_mul_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                       int size, float* __restrict__ out)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x)
        out[i] = a[i] * b[i];
}

// peak_extract_kernel — one thread per pixel: mark it a CANDIDATE peak iff
// its value clears `threshold` AND it is the (tie-broken) maximum in its
// own (2*window_r+1)^2 neighborhood. Tie-break rule: a strictly larger
// neighbor always wins; an EQUAL neighbor wins only if its flattened index
// is smaller — this makes the "am I the local max" test a well-defined,
// deterministic total order (no two pixels of equal value in one
// neighborhood can both claim the peak), which reference_cpu.cpp's
// independent twin reproduces with the identical rule.
__global__ void peak_extract_kernel(const float* __restrict__ heatmap, int h, int w, float threshold,
                                    int window_r, unsigned char* __restrict__ is_candidate)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= w || iy >= h) return;

    const int my_flat = iy * w + ix;
    const float v = heatmap[my_flat];
    if (v <= threshold) { is_candidate[my_flat] = 0; return; }

    for (int dy = -window_r; dy <= window_r; ++dy) {
        for (int dx = -window_r; dx <= window_r; ++dx) {
            const int nx = ix + dx, ny = iy + dy;
            if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
            const int n_flat = ny * w + nx;
            if (n_flat == my_flat) continue;
            const float nv = heatmap[n_flat];
            if (nv > v || (nv == v && n_flat < my_flat)) { is_candidate[my_flat] = 0; return; }
        }
    }
    is_candidate[my_flat] = 1;
}

// ===========================================================================
// Host-callable launch wrappers — grid/block math + the mandatory post-
// launch error check (CLAUDE.md §6.1 rule 7), kept beside each kernel's
// definition above.
// ===========================================================================

static constexpr int kBlock1D = 256;      // warp multiple, standard repo default (02.01/08.01 use the same)
static constexpr int kGridCap1D = 4096;   // enough blocks to fill any current GPU many times over; the
                                          // grid-stride loops above absorb whatever this cap leaves uncovered

static int grid1d(int n)
{
    int g = blocks_for(n, kBlock1D);
    return g > kGridCap1D ? kGridCap1D : (g < 1 ? 1 : g);
}

void launch_compute_pillar_keys(int n, const float* d_points, int* d_keys)
{
    compute_pillar_keys_kernel<<<grid1d(n), kBlock1D>>>(n, d_points, d_keys);
    CUDA_CHECK_LAST_ERROR("compute_pillar_keys_kernel launch");
}

void launch_compute_voxel_keys(int n, const float* d_points, int* d_keys)
{
    compute_voxel_keys_kernel<<<grid1d(n), kBlock1D>>>(n, d_points, d_keys);
    CUDA_CHECK_LAST_ERROR("compute_voxel_keys_kernel launch");
}

void launch_reset_counts(unsigned int* d_point_count, int num_cells)
{
    reset_counts_kernel<<<grid1d(num_cells), kBlock1D>>>(d_point_count, num_cells);
    CUDA_CHECK_LAST_ERROR("reset_counts_kernel launch");
}

void launch_atomic_bin(int n, const float* d_points, const int* d_keys, PillarBinGPU bin)
{
    atomic_bin_kernel<<<grid1d(n), kBlock1D>>>(n, d_points, d_keys, bin);
    CUDA_CHECK_LAST_ERROR("atomic_bin_kernel launch");
}

int launch_sort_and_compact(int n, const int* d_keys_in, int* d_keys_scratch, int* d_idx_scratch,
                            int* d_is_start_scratch, int* d_seg_start_out, int* d_occupied_cell_out,
                            int* n_valid_out)
{
    thrust::device_ptr<const int> keys_in_ptr(d_keys_in);
    thrust::device_ptr<int> keys_scratch_ptr(d_keys_scratch);
    thrust::device_ptr<int> idx_scratch_ptr(d_idx_scratch);

    // 1) Filter: keep only in-window points (key >= 0). thrust::copy_if with
    // a STENCIL (a second range that decides inclusion, here the same key
    // array used both as values and as its own stencil) is the standard
    // stream-compaction primitive — see 02.01's kernels.cu for the identical
    // idiom applied to segment boundaries instead of raw validity.
    auto keys_end = thrust::copy_if(keys_in_ptr, keys_in_ptr + n, keys_in_ptr, keys_scratch_ptr, is_valid_key());
    const int n_valid = static_cast<int>(keys_end - keys_scratch_ptr);
    thrust::copy_if(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(n),
                    keys_in_ptr, idx_scratch_ptr, is_valid_key());
    *n_valid_out = n_valid;

    if (n_valid == 0) {
        const int zero = 0;
        CUDA_CHECK(cudaMemcpy(d_seg_start_out, &zero, sizeof(int), cudaMemcpyHostToDevice));
        return 0;
    }

    // 2) Stable sort by key, carrying the original point index — see
    // 02.01's kernels.cu for the full "why stable sort => bit-exact truncation"
    // explanation; identical reasoning applies here verbatim.
    thrust::stable_sort_by_key(keys_scratch_ptr, keys_scratch_ptr + n_valid, idx_scratch_ptr);

    // 3) Mark segment boundaries, then reduce (count) and compact (offsets)
    // exactly as 02.01's Method B does.
    mark_boundaries_kernel<<<grid1d(n_valid), kBlock1D>>>(n_valid, d_keys_scratch, d_is_start_scratch);
    CUDA_CHECK_LAST_ERROR("mark_boundaries_kernel launch");

    thrust::device_ptr<int> is_start_ptr(d_is_start_scratch);
    const int num_occupied = thrust::reduce(is_start_ptr, is_start_ptr + n_valid, 0);

    thrust::device_ptr<int> seg_start_ptr(d_seg_start_out);
    thrust::copy_if(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(n_valid),
                    is_start_ptr, seg_start_ptr, is_nonzero());
    // Sentinel end: seg_start_out[num_occupied] = n_valid, so every segment's
    // length is seg_start[s+1]-seg_start[s] uniformly, including the last.
    CUDA_CHECK(cudaMemcpy(d_seg_start_out + num_occupied, &n_valid, sizeof(int), cudaMemcpyHostToDevice));

    if (num_occupied > 0) {
        gather_occupied_cell_kernel<<<grid1d(num_occupied), kBlock1D>>>(num_occupied, d_seg_start_out,
                                                                        d_keys_scratch, d_occupied_cell_out);
        CUDA_CHECK_LAST_ERROR("gather_occupied_cell_kernel launch");
    }
    return num_occupied;
}

void launch_sorted_bin(int num_occupied, const int* d_seg_start, int n_sorted, const int* d_idx_sorted,
                       const float* d_points, const int* d_occupied_cell, PillarBinGPU bin)
{
    if (num_occupied == 0) return;
    sorted_bin_kernel<<<grid1d(num_occupied), kBlock1D>>>(num_occupied, d_seg_start, n_sorted, d_idx_sorted,
                                                          d_points, d_occupied_cell, bin);
    CUDA_CHECK_LAST_ERROR("sorted_bin_kernel launch");
}

void launch_pfn_stats(int num_occupied, const int* d_occupied_cell, PillarBinGPU bin,
                      float* d_mean_xyz_out, unsigned int* d_kept_count_out)
{
    if (num_occupied == 0) return;
    pfn_stats_kernel<<<grid1d(num_occupied), kBlock1D>>>(num_occupied, d_occupied_cell, bin,
                                                         d_mean_xyz_out, d_kept_count_out);
    CUDA_CHECK_LAST_ERROR("pfn_stats_kernel launch");
}

void launch_augment_features(int num_occupied, const int* d_occupied_cell, PillarBinGPU bin,
                             const float* d_mean_xyz, float* d_features_out)
{
    if (num_occupied == 0) return;
    const int total = num_occupied * bin.cap_n;
    augment_features_kernel<<<grid1d(total), kBlock1D>>>(num_occupied, d_occupied_cell, bin,
                                                         d_mean_xyz, d_features_out);
    CUDA_CHECK_LAST_ERROR("augment_features_kernel launch");
}

void launch_pfn_lite(int num_occupied, const float* d_features, const unsigned int* d_kept_count,
                     const float* d_lin_w, const float* d_lin_b, float* d_pillar_feat_out)
{
    if (num_occupied == 0) return;
    pfn_lite_kernel<<<grid1d(num_occupied), kBlock1D>>>(num_occupied, d_features, d_kept_count,
                                                        d_lin_w, d_lin_b, d_pillar_feat_out);
    CUDA_CHECK_LAST_ERROR("pfn_lite_kernel launch");
}

void launch_scatter(int num_occupied, const int* d_occupied_cell, const float* d_pillar_feat,
                    float* d_canvas)
{
    if (num_occupied == 0) return;
    scatter_kernel<<<grid1d(num_occupied), kBlock1D>>>(num_occupied, d_occupied_cell, d_pillar_feat, d_canvas);
    CUDA_CHECK_LAST_ERROR("scatter_kernel launch");
}

void launch_gather(int num_occupied, const int* d_occupied_cell, const float* d_canvas,
                   float* d_gathered_out)
{
    if (num_occupied == 0) return;
    gather_kernel<<<grid1d(num_occupied), kBlock1D>>>(num_occupied, d_occupied_cell, d_canvas, d_gathered_out);
    CUDA_CHECK_LAST_ERROR("gather_kernel launch");
}

// 2-D launch configs (the BEV canvas, kGridNX x kGridNY = 200x200): 16x16
// thread blocks are the standard 2-D image-kernel default (256 threads/block,
// same total as the 1-D kernels above) — grid = ceil(200/16)=13 in each dim.
static const dim3 kBlock2D(16, 16);
static dim3 grid2d(int w, int h)
{
    return dim3(static_cast<unsigned>(blocks_for(w, kBlock2D.x)), static_cast<unsigned>(blocks_for(h, kBlock2D.y)));
}

void launch_conv3x3(const float* d_in_plane, int h, int w, const float* d_kernel3x3, float bias,
                    float* d_out_plane)
{
    conv3x3_kernel<<<grid2d(w, h), kBlock2D>>>(d_in_plane, h, w, d_kernel3x3, bias, d_out_plane);
    CUDA_CHECK_LAST_ERROR("conv3x3_kernel launch");
}

void launch_elementwise_mul(const float* d_a, const float* d_b, int size, float* d_out)
{
    elementwise_mul_kernel<<<grid1d(size), kBlock1D>>>(d_a, d_b, size, d_out);
    CUDA_CHECK_LAST_ERROR("elementwise_mul_kernel launch");
}

void launch_peak_extract(const float* d_heatmap, int h, int w, float threshold, int window_r,
                         unsigned char* d_is_candidate)
{
    peak_extract_kernel<<<grid2d(w, h), kBlock2D>>>(d_heatmap, h, w, threshold, window_r, d_is_candidate);
    CUDA_CHECK_LAST_ERROR("peak_extract_kernel launch");
}
