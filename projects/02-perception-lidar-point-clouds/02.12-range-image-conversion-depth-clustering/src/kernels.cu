// ===========================================================================
// kernels.cu — GPU kernels for project 02.12 (Range-image conversion +
//              depth-clustering segmentation)
//
// Role in the project
// -------------------
// All __global__ (GPU) code lives here, plus the small host-side launch
// wrappers that own the grid/block math and the Thrust orchestration
// (CLAUDE.md §6.1 rule 2: launch-configuration reasoning sits beside the
// code it configures). The six pipeline stages kernels.cuh describes are
// implemented in order below: (1) range-image conversion, both directions;
// (2) ground removal; (3) depth-clustering edges (the beta criterion);
// (4) generic lock-free union-find (02.04 Method A, cited); (5) Euclidean
// comparison clustering (voxel hash, 02.04 lineage, cited).
//
// Read this after: main.cu, kernels.cuh.  Read this before: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/iterator/counting_iterator.h>

// is_nonzero — the copy_if predicate launch_build_voxel_index uses to
// compact the 0/1 boundary mask into segment-start positions. CUDA 13.3's
// Thrust dropped thrust::identity (root-caused in 02.01), so every project
// that needs this predicate carries its own one-line functor, cited there.
struct is_nonzero {
    __host__ __device__ bool operator()(int x) const { return x != 0; }
};

// ===========================================================================
// DEVICE TRANSCRIPTIONS of kernels.cuh's plain-inline range/cell-index
// helpers (02.02's identical pattern, cited): organized_cell_index(),
// pack_range_index(), and unpack_point_index() are declared WITHOUT
// __host__/__device__ in kernels.cuh so reference_cpu.cpp (compiled by
// cl.exe) can call them too, which means nvcc treats them as HOST-only and
// refuses to call them from a __global__ kernel. Each has a one-line
// __device__ twin here, cross-referenced at every call site below — the
// exact discipline kernels.cuh's file header promises.
// ===========================================================================
__device__ __forceinline__ int d_organized_cell_index(int ring, int az_bin)
{
    return ring * kAzimuthBins + az_bin;
}

__device__ __forceinline__ uint32_t d_float_range_to_sortable_u32(float r)
{
    // __float_as_int reinterprets the bit pattern with no rounding — the
    // device-side equivalent of the header's std::memcpy trick (both are
    // exact bit reinterpretation; memcpy is the portable host spelling,
    // __float_as_int is CUDA's intrinsic for the identical operation).
    return static_cast<uint32_t>(__float_as_int(r));
}

__device__ __forceinline__ uint64_t d_pack_range_index(float range_m, uint32_t point_idx)
{
    return (static_cast<uint64_t>(d_float_range_to_sortable_u32(range_m)) << 32) | point_idx;
}

__device__ __forceinline__ uint32_t d_unpack_point_index(uint64_t encoded)
{
    return static_cast<uint32_t>(encoded & 0xFFFFFFFFu);
}

// ===========================================================================
// STAGE 1a — unorganized -> organized (range-image conversion).
//
// scatter_encode_kernel races every input point's (range, point_index)
// encoded key into its target cell via atomicMin — IDENTICAL technique to
// 02.02's scatter_to_organized_kernel (cited): organized_cell_index()/
// pack_range_index() live in kernels.cuh as plain inline (host-only)
// functions, so this kernel calls the __device__ transcriptions above
// instead (d_organized_cell_index/d_pack_range_index) — see kernels.cuh's
// "why CUDA-qualifier-free" note for why the duplication exists.
// ===========================================================================
__global__ void scatter_encode_kernel(int n_points,
                                      const int* __restrict__ ring, const int* __restrict__ az_bin,
                                      const float* __restrict__ range_m,
                                      unsigned long long* __restrict__ cell_encoded)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) return;

    const int cell = d_organized_cell_index(ring[i], az_bin[i]);
    const unsigned long long key = d_pack_range_index(range_m[i], static_cast<uint32_t>(i));

    // atomicMin on a 64-bit key with (range in the top 32 bits, point index
    // in the bottom 32) means "smallest encoded value" == "smallest range,
    // ties broken by index" — exactly the nearest-return-wins rule a real
    // range image needs, decided deterministically regardless of which
    // thread's atomic actually executes last (02.02's argument, cited).
    atomicMin(&cell_encoded[cell], key);
}

// finalize_organized_kernel — one thread per cell: decode the winner (if
// any) and materialize the organized arrays. An empty cell (no point ever
// targeted it) gets range=0 (this project's "no return" sentinel
// throughout) and truth_id=-1.
__global__ void finalize_organized_kernel(int num_cells,
                                          const unsigned long long* __restrict__ cell_encoded,
                                          const float* __restrict__ px, const float* __restrict__ py,
                                          const float* __restrict__ pz, const float* __restrict__ prange,
                                          const int* __restrict__ ptruth,
                                          float* __restrict__ range_img,
                                          float* __restrict__ xyz_img,
                                          int* __restrict__ truth_img,
                                          int* __restrict__ winner_idx_img)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= num_cells) return;

    const unsigned long long enc = cell_encoded[c];
    if (enc == kEmptyCellEncoded) {
        range_img[c] = 0.0f;
        xyz_img[c * 3 + 0] = 0.0f; xyz_img[c * 3 + 1] = 0.0f; xyz_img[c * 3 + 2] = 0.0f;
        truth_img[c] = -1;
        winner_idx_img[c] = -1;
        return;
    }
    const uint32_t idx = d_unpack_point_index(enc);
    range_img[c] = prange[idx];
    xyz_img[c * 3 + 0] = px[idx];
    xyz_img[c * 3 + 1] = py[idx];
    xyz_img[c * 3 + 2] = pz[idx];
    truth_img[c] = ptruth[idx];
    winner_idx_img[c] = static_cast<int>(idx);
}

void launch_scatter_to_organized(int n_points,
                                 const int* d_ring, const int* d_az_bin, const float* d_range_m,
                                 const float* d_px, const float* d_py, const float* d_pz,
                                 const int* d_truth,
                                 float* d_range_img, float* d_xyz_img,
                                 int* d_truth_img, int* d_winner_idx_img)
{
    unsigned long long* d_cell_encoded = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cell_encoded, static_cast<size_t>(kNumCells) * sizeof(unsigned long long)));
    // A single byte-fill (0xFF everywhere) sets every cell to kEmptyCellEncoded
    // in one memset — no kernel needed (02.02's identical trick, cited).
    CUDA_CHECK(cudaMemset(d_cell_encoded, 0xFF, static_cast<size_t>(kNumCells) * sizeof(unsigned long long)));

    {
        const int block = kThreadsPerBlock;
        const int grid  = blocks_for(n_points, block);
        scatter_encode_kernel<<<grid, block>>>(n_points, d_ring, d_az_bin, d_range_m, d_cell_encoded);
        CUDA_CHECK_LAST_ERROR("scatter_encode_kernel launch");
    }
    {
        const int block = kThreadsPerBlock;
        const int grid  = blocks_for(kNumCells, block);
        finalize_organized_kernel<<<grid, block>>>(kNumCells, d_cell_encoded,
                                                    d_px, d_py, d_pz, d_range_m, d_truth,
                                                    d_range_img, d_xyz_img, d_truth_img, d_winner_idx_img);
        CUDA_CHECK_LAST_ERROR("finalize_organized_kernel launch");
    }
    CUDA_CHECK(cudaFree(d_cell_encoded));
}

// ===========================================================================
// STAGE 1b — organized -> unorganized (obstacle compaction). A simple
// atomic-counter push, not 02.02's Blelloch-scan compaction: at
// kNumCells=16,384 elements this is a few microseconds either way, so the
// simpler mechanism is the honest choice here (02.02 is cited as the
// technique to reach for at real LiDAR point counts).
// ===========================================================================
__global__ void compact_obstacles_kernel(int num_cells,
                                         const float* __restrict__ range_img,
                                         const float* __restrict__ xyz_img,
                                         const int* __restrict__ obstacle_mask,
                                         const int* __restrict__ truth_img,
                                         float* __restrict__ out_xyz,
                                         int* __restrict__ out_cell_idx,
                                         int* __restrict__ out_truth,
                                         int* __restrict__ out_count)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= num_cells) return;
    if (range_img[c] <= 0.0f || !obstacle_mask[c]) return;   // ground or empty: not an obstacle point

    const int slot = atomicAdd(out_count, 1);   // claim a push_back slot (parallel stream compaction)
    out_xyz[slot * 3 + 0] = xyz_img[c * 3 + 0];
    out_xyz[slot * 3 + 1] = xyz_img[c * 3 + 1];
    out_xyz[slot * 3 + 2] = xyz_img[c * 3 + 2];
    out_cell_idx[slot] = c;
    out_truth[slot] = truth_img[c];
}

int launch_compact_obstacles(const float* d_range_img, const float* d_xyz_img,
                             const int* d_obstacle_mask, const int* d_truth_img,
                             float* d_out_xyz, int* d_out_cell_idx, int* d_out_truth)
{
    int* d_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

    const int block = kThreadsPerBlock;
    const int grid  = blocks_for(kNumCells, block);
    compact_obstacles_kernel<<<grid, block>>>(kNumCells, d_range_img, d_xyz_img, d_obstacle_mask, d_truth_img,
                                              d_out_xyz, d_out_cell_idx, d_out_truth, d_count);
    CUDA_CHECK_LAST_ERROR("compact_obstacles_kernel launch");

    int count = 0;
    CUDA_CHECK(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_count));
    return count;   // always <= kNumCells: capacity is never exceeded by construction
}

// ===========================================================================
// STAGE 2 — ground removal: one thread PER AZIMUTH COLUMN, a short serial
// walk over its kNumBeams=16 cells.
//
// __device__ transcription of kBeamElevRad (kernels.cuh's header explains
// WHY an array — unlike a scalar constexpr — needs its own device-side
// copy: it lives in host memory unless explicitly placed for the device).
// ===========================================================================
__constant__ float kBeamElevRadDev[kNumBeams] = {
    -0.26179939f, -0.22689280f, -0.19198622f, -0.15707963f,
    -0.12217305f, -0.08726646f, -0.05235988f, -0.01745329f,
     0.01745329f,  0.05235988f,  0.08726646f,  0.12217305f,
     0.15707963f,  0.19198622f,  0.22689280f,  0.26179939f
};

// ground_removal_kernel — see kernels.cuh's stage-2 declaration comment for
// the full column-walk description (bottom-up, virtual sensor-height
// reference for the first return). This kernel embodies THE choice named
// there: kAzimuthBins=1024 independent threads (embarrassingly parallel
// across columns), each running a 16-iteration SEQUENTIAL loop (no further
// parallel decomposition — 16 is far too short to amortize a Blelloch scan's
// setup cost; see THEORY.md "The GPU mapping" for the crossover argument).
__global__ void ground_removal_kernel(int num_columns,
                                      const float* __restrict__ range_img,
                                      const float* __restrict__ xyz_img,
                                      int* __restrict__ ground_label,
                                      int* __restrict__ obstacle_mask)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= num_columns) return;

    // The virtual reference point: directly below the sensor, at the known
    // ground height. rho (horizontal radius) is 0 there by definition — the
    // sensor's own vertical axis passes through it.
    float rho_prev = 0.0f;
    float z_prev = -kSensorHeightM;

    for (int ring = 0; ring < kNumBeams; ++ring) {
        const int cell = d_organized_cell_index(ring, col);
        const float r = range_img[cell];
        if (r <= 0.0f) continue;   // no return in this cell: nothing to classify, walk continues unmodified

        const float x = xyz_img[cell * 3 + 0];
        const float y = xyz_img[cell * 3 + 1];
        const float z = xyz_img[cell * 3 + 2];
        const float rho = sqrtf(x * x + y * y);

        const float d_rho = rho - rho_prev;
        const float d_z   = z - z_prev;
        // atan2f handles every sign combination, including d_rho<=0 (a
        // near-radial jump onto a farther-or-nearer surface at almost the
        // same bearing — exactly what a depth discontinuity produces): the
        // result then reads close to +-90 deg, correctly failing the
        // flatness test just like a genuine vertical face would.
        const float angle_deg = atan2f(d_z, d_rho) * (180.0f / kPi);

        const int is_ground = (fabsf(angle_deg) <= kGroundAngleThresholdDeg) ? 1 : 0;
        ground_label[cell] = is_ground;
        obstacle_mask[cell] = is_ground ? 0 : 1;

        // Advance the walk from THIS return regardless of its label — the
        // next step's slope is always measured against the immediately
        // preceding VALID return (kernels.cuh's stage-2 comment explains
        // why this stays correct even once the column has left the ground).
        rho_prev = rho;
        z_prev = z;
    }
}

void launch_ground_removal(const float* d_range_img, const float* d_xyz_img,
                           int* d_ground_label, int* d_obstacle_mask)
{
    const int block = kThreadsPerBlock;
    const int grid  = blocks_for(kAzimuthBins, block);
    ground_removal_kernel<<<grid, block>>>(kAzimuthBins, d_range_img, d_xyz_img, d_ground_label, d_obstacle_mask);
    CUDA_CHECK_LAST_ERROR("ground_removal_kernel launch");
}

// ===========================================================================
// STAGE 3 — depth-clustering edges: THE BETA CRITERION, evaluated on the
// two FORWARD image neighbors of every obstacle cell.
// ===========================================================================

// depth_edges_kernel — one thread per cell c=(ring,col). Tests:
//   (a) (ring+1, col)       — clamped: no wrap in the ring direction (there
//                             is no beam "above" ring 15 or "below" ring 0).
//   (b) (ring, col+1 mod W) — WRAPPED: the sensor spins a full 360 degrees,
//                             so column 1023's forward neighbor is column 0.
//                             This project's own scene (person+wall placed
//                             at azimuth 0) straddles exactly this seam, so
//                             the wrap is genuinely exercised, not just
//                             theoretically present (see
//                             scripts/make_synthetic.py's diagnostic
//                             printout: person/wall az_bin ranges span both
//                             ends of [0,1023]).
// Processing FORWARD neighbors only means every undirected image edge is
// considered exactly once (by its lower-ring/lower-column endpoint) — the
// same "only emit when j>i" halving idea 02.04's build_edges_kernel uses,
// specialized to a grid's fixed 2-neighbor forward stencil instead of an
// arbitrary point index comparison.
__global__ void depth_edges_kernel(int num_cells,
                                   const float* __restrict__ range_img,
                                   const int* __restrict__ obstacle_mask,
                                   int* __restrict__ edge_u, int* __restrict__ edge_v,
                                   int* __restrict__ edge_count)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= num_cells) return;
    if (range_img[c] <= 0.0f || !obstacle_mask[c]) return;   // only obstacle cells enter the depth-cluster graph

    const int ring = c / kAzimuthBins;
    const int col  = c % kAzimuthBins;
    const float r_c = range_img[c];

    // ---- Neighbor (a): ring+1, same column (elevation step) --------------
    if (ring + 1 < kNumBeams) {
        const int nc = d_organized_cell_index(ring + 1, col);
        if (range_img[nc] > 0.0f && obstacle_mask[nc]) {
            const float r_n = range_img[nc];
            const float alpha = kBeamElevRadDev[ring + 1] - kBeamElevRadDev[ring];   // > 0 by construction
            const float r1 = fmaxf(r_c, r_n);   // farther of the two
            const float r2 = fminf(r_c, r_n);   // nearer of the two
            const float beta = atan2f(r2 * sinf(alpha), r1 - r2 * cosf(alpha));
            if (beta >= kBetaThresholdRad) {
                const int slot = atomicAdd(edge_count, 1);
                // c < nc always here (ring+1 strictly raises the flat cell
                // index) -- ascending order falls out for free.
                edge_u[slot] = c;
                edge_v[slot] = nc;
            }
        }
    }
    // ---- Neighbor (b): same ring, column+1 WITH WRAP-AROUND ---------------
    {
        const int next_col = (col + 1 == kAzimuthBins) ? 0 : col + 1;
        const int nc = d_organized_cell_index(ring, next_col);
        if (range_img[nc] > 0.0f && obstacle_mask[nc]) {
            const float r_n = range_img[nc];
            const float r1 = fmaxf(r_c, r_n);
            const float r2 = fminf(r_c, r_n);
            const float beta = atan2f(r2 * sinf(kAzimuthStepRad), r1 - r2 * cosf(kAzimuthStepRad));
            if (beta >= kBetaThresholdRad) {
                const int slot = atomicAdd(edge_count, 1);
                // At the WRAP seam (col==kAzimuthBins-1 -> next_col==0), nc
                // is numerically SMALLER than c even though nc is the
                // "forward" neighbor -- store in ASCENDING (min,max) order
                // so every edge this kernel ever emits is already
                // canonical, matching depth_edges_cpu's identical
                // convention (main.cu's VERIFY(depth_edges) compares the
                // two edge sets after only a sort, no per-edge reordering).
                edge_u[slot] = min(c, nc);
                edge_v[slot] = max(c, nc);
            }
        }
    }
}

int launch_depth_edges(const float* d_range_img, const int* d_obstacle_mask,
                       int* d_edge_u, int* d_edge_v)
{
    int* d_edge_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_edge_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_edge_count, 0, sizeof(int)));

    const int block = kThreadsPerBlock;
    const int grid  = blocks_for(kNumCells, block);
    depth_edges_kernel<<<grid, block>>>(kNumCells, d_range_img, d_obstacle_mask, d_edge_u, d_edge_v, d_edge_count);
    CUDA_CHECK_LAST_ERROR("depth_edges_kernel launch");

    int edge_count = 0;
    CUDA_CHECK(cudaMemcpy(&edge_count, d_edge_count, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_edge_count));
    return edge_count;   // <= kNumCells*2 always: exact capacity, no overflow bookkeeping needed
}

// ===========================================================================
// STAGE 4 — generic lock-free GPU union-find (02.04 Method A, cited and
// reused near-verbatim: the algorithm is GENERIC over any edge list, so the
// SAME three kernels below cluster both the depth-image graph and the
// Euclidean-comparison graph — main.cu simply calls them twice).
// ===========================================================================

// d_uf_find_halve — path-halving find, the one primitive every union-find
// stage below calls. See kernels.cuh "THE UNION-FIND CHAPTER" (this
// project's own copy of the identical derivation 02.04 states in full) for
// the monotone-parent safety argument that makes the lone non-atomic store
// below race-safe.
__device__ __forceinline__ int d_uf_find_halve(int* __restrict__ parent, int x)
{
    while (true) {
        const int p = parent[x];
        const int gp = parent[p];
        if (p == gp) return p;        // p is already a root (parent[p]==p)
        parent[x] = gp;                // skip a link: point x at its grandparent
        x = gp;
    }
}

__global__ void uf_init_kernel(int n, int* __restrict__ parent)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    parent[i] = i;
}

void launch_uf_init(int n, int* d_parent)
{
    const int block = kThreadsPerBlock;
    const int grid  = blocks_for(n, block);
    uf_init_kernel<<<grid, block>>>(n, d_parent);
    CUDA_CHECK_LAST_ERROR("uf_init_kernel launch");
}

__global__ void uf_union_sweep_kernel(int num_edges, const int* __restrict__ edge_u,
                                      const int* __restrict__ edge_v,
                                      int* __restrict__ parent, int* __restrict__ changed)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_edges) return;

    int ru = d_uf_find_halve(parent, edge_u[e]);
    int rv = d_uf_find_halve(parent, edge_v[e]);

    while (ru != rv) {
        const int lo = min(ru, rv);
        const int hi = max(ru, rv);
        const int old = atomicCAS(&parent[hi], hi, lo);
        if (old == hi) {
            atomicOr(changed, 1);
            return;
        }
        ru = d_uf_find_halve(parent, lo);
        rv = d_uf_find_halve(parent, hi);
    }
}

bool launch_uf_union_sweep(int num_edges, const int* d_edge_u, const int* d_edge_v,
                           int* d_parent, int* d_changed)
{
    CUDA_CHECK(cudaMemset(d_changed, 0, sizeof(int)));
    if (num_edges > 0) {
        const int block = kThreadsPerBlock;
        const int grid  = blocks_for(num_edges, block);
        uf_union_sweep_kernel<<<grid, block>>>(num_edges, d_edge_u, d_edge_v, d_parent, d_changed);
        CUDA_CHECK_LAST_ERROR("uf_union_sweep_kernel launch");
    }
    int changed = 0;
    CUDA_CHECK(cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
    return changed != 0;
}

__global__ void uf_finalize_kernel(int n, int* __restrict__ parent)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int x = i;
    while (parent[x] != x) x = parent[x];
    parent[i] = x;
}

void launch_uf_finalize(int n, int* d_parent)
{
    const int block = kThreadsPerBlock;
    const int grid  = blocks_for(n, block);
    uf_finalize_kernel<<<grid, block>>>(n, d_parent);
    CUDA_CHECK_LAST_ERROR("uf_finalize_kernel launch");
}

// ===========================================================================
// STAGE 5 — Euclidean comparison clustering: voxel hash + 27-cell stencil
// (02.04 lineage, cited and adapted to this project's smaller, hollow-
// surface point count — see kernels.cuh kMaxEdgesPerPointEuclid's derivation).
// ===========================================================================

__device__ __forceinline__ int32_t d_voxel_coord(float p, float leaf)
{
    return static_cast<int32_t>(floorf(p / leaf));
}

__device__ __forceinline__ uint64_t d_pack_voxel_key(int32_t vx, int32_t vy, int32_t vz)
{
    const uint64_t ux = static_cast<uint64_t>(vx + kCoordBias) & kCoordMask21;
    const uint64_t uy = static_cast<uint64_t>(vy + kCoordBias) & kCoordMask21;
    const uint64_t uz = static_cast<uint64_t>(vz + kCoordBias) & kCoordMask21;
    return ux | (uy << 21) | (uz << 42);
}

__device__ __forceinline__ void d_unpack_voxel_key(uint64_t key, int32_t& vx, int32_t& vy, int32_t& vz)
{
    vx = static_cast<int32_t>(key & kCoordMask21) - kCoordBias;
    vy = static_cast<int32_t>((key >> 21) & kCoordMask21) - kCoordBias;
    vz = static_cast<int32_t>((key >> 42) & kCoordMask21) - kCoordBias;
}

// d_lower_bound — binary search over the ascending-sorted unique_key[]
// array (02.04's identical helper, cited): the smallest index i with
// unique_key[i] >= target, or `count` if none.
__device__ __forceinline__ int d_lower_bound(const unsigned long long* __restrict__ unique_key,
                                             int count, unsigned long long target)
{
    int lo = 0, hi = count;
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (unique_key[mid] < target) lo = mid + 1;
        else                          hi = mid;
    }
    return lo;
}

__global__ void compute_voxel_keys_kernel(int n, const float* __restrict__ xyz,
                                          unsigned long long* __restrict__ keys)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float px = xyz[i * 3 + 0], py = xyz[i * 3 + 1], pz = xyz[i * 3 + 2];
    const int32_t vx = d_voxel_coord(px, kEuclideanClusterToleranceM);
    const int32_t vy = d_voxel_coord(py, kEuclideanClusterToleranceM);
    const int32_t vz = d_voxel_coord(pz, kEuclideanClusterToleranceM);
    keys[i] = d_pack_voxel_key(vx, vy, vz);
}

void launch_compute_voxel_keys(int n, const float* d_xyz, unsigned long long* d_keys)
{
    const int block = kThreadsPerBlock;
    const int grid  = blocks_for(n, block);
    compute_voxel_keys_kernel<<<grid, block>>>(n, d_xyz, d_keys);
    CUDA_CHECK_LAST_ERROR("compute_voxel_keys_kernel launch");
}

__global__ void mark_boundaries_kernel(int n, const unsigned long long* __restrict__ keys_sorted,
                                       int* __restrict__ is_start)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    is_start[i] = (i == 0 || keys_sorted[i] != keys_sorted[i - 1]) ? 1 : 0;
}

__global__ void gather_unique_keys_kernel(int num_voxels, const int* __restrict__ seg_start,
                                          const unsigned long long* __restrict__ keys_sorted,
                                          unsigned long long* __restrict__ unique_key_out)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_voxels) return;
    unique_key_out[v] = keys_sorted[seg_start[v]];
}

// launch_build_voxel_index — 02.01/02.04's sort+compaction pipeline (cited,
// reused verbatim in structure): turn "who shares a voxel" into contiguous
// sorted-array runs plus a dense ascending list of the distinct keys.
int launch_build_voxel_index(int n, const unsigned long long* d_keys_in,
                             unsigned long long* d_keys_scratch, int* d_idx_sorted,
                             int* d_is_start_scratch, int* d_seg_start_out,
                             unsigned long long* d_unique_key_out)
{
    CUDA_CHECK(cudaMemcpy(d_keys_scratch, d_keys_in,
                          static_cast<size_t>(n) * sizeof(unsigned long long),
                          cudaMemcpyDeviceToDevice));

    thrust::device_ptr<unsigned long long> keys_ptr(d_keys_scratch);
    thrust::device_ptr<int> idx_ptr(d_idx_sorted);
    thrust::sequence(idx_ptr, idx_ptr + n);
    // thrust::stable_sort_by_key: radix-sorts the 64-bit voxel keys
    // ascending, carrying the point-index permutation along (STABLE, so
    // points that share a key keep their relative order — irrelevant for
    // correctness here, but a free, cheap guarantee).
    thrust::stable_sort_by_key(keys_ptr, keys_ptr + n, idx_ptr);

    {
        const int block = kThreadsPerBlock;
        const int grid  = blocks_for(n, block);
        mark_boundaries_kernel<<<grid, block>>>(n, d_keys_scratch, d_is_start_scratch);
        CUDA_CHECK_LAST_ERROR("mark_boundaries_kernel launch");
    }

    thrust::device_ptr<int> is_start_ptr(d_is_start_scratch);
    const int num_voxels = thrust::reduce(is_start_ptr, is_start_ptr + n, 0);

    thrust::device_ptr<int> seg_start_ptr(d_seg_start_out);
    thrust::copy_if(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(n),
                    is_start_ptr, seg_start_ptr, is_nonzero());

    {
        const int block = kThreadsPerBlock;
        const int grid  = blocks_for(num_voxels, block);
        gather_unique_keys_kernel<<<grid, block>>>(num_voxels, d_seg_start_out, d_keys_scratch, d_unique_key_out);
        CUDA_CHECK_LAST_ERROR("gather_unique_keys_kernel launch");
    }

    return num_voxels;
}

// build_edges_euclid_kernel — one thread per point i: walk the 27-cell
// voxel stencil around i's own voxel (kEuclideanClusterToleranceM's leaf==d
// choice, 02.04's proof cited in kernels.cuh, guarantees this stencil is
// SUFFICIENT), find each occupied neighbor via binary search, and emit an
// edge (i,j) for every j>i within the actual Euclidean distance. Identical
// structure to 02.04's build_edges_kernel; cited rather than re-derived
// here (see kernels.cuh's declaration comment for the full walkthrough).
__global__ void build_edges_euclid_kernel(int n, const float* __restrict__ xyz,
                                          const unsigned long long* __restrict__ point_key,
                                          const unsigned long long* __restrict__ unique_key, int num_voxels,
                                          const int* __restrict__ seg_start,
                                          const int* __restrict__ idx_sorted, int n_sorted,
                                          int* __restrict__ edge_u, int* __restrict__ edge_v, int edge_capacity,
                                          int* __restrict__ edge_count, int* __restrict__ overflow_count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float pi[3] = { xyz[i * 3 + 0], xyz[i * 3 + 1], xyz[i * 3 + 2] };
    const float d2 = kEuclideanClusterToleranceM * kEuclideanClusterToleranceM;

    int32_t vx, vy, vz;
    d_unpack_voxel_key(point_key[i], vx, vy, vz);

    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const uint64_t nkey = d_pack_voxel_key(vx + dx, vy + dy, vz + dz);
                const int v = d_lower_bound(unique_key, num_voxels, nkey);
                if (v >= num_voxels || unique_key[v] != nkey) continue;

                const int begin = seg_start[v];
                const int end   = (v + 1 < num_voxels) ? seg_start[v + 1] : n_sorted;
                for (int k = begin; k < end; ++k) {
                    const int j = idx_sorted[k];
                    if (j <= i) continue;   // dedup: emit each undirected edge once (i<j)

                    const float pj[3] = { xyz[j * 3 + 0], xyz[j * 3 + 1], xyz[j * 3 + 2] };
                    const float dxp = pi[0] - pj[0], dyp = pi[1] - pj[1], dzp = pi[2] - pj[2];
                    const float dist2 = dxp * dxp + dyp * dyp + dzp * dzp;
                    if (dist2 > d2) continue;

                    const int slot = atomicAdd(edge_count, 1);
                    if (slot < edge_capacity) {
                        edge_u[slot] = i;
                        edge_v[slot] = j;
                    } else {
                        atomicAdd(overflow_count, 1);   // honestly counted, never silently dropped
                    }
                }
            }
        }
    }
}

int launch_build_edges_euclid(int n, const float* d_xyz, const unsigned long long* d_point_key,
                              const unsigned long long* d_unique_key, int num_voxels,
                              const int* d_seg_start, const int* d_idx_sorted, int n_sorted,
                              int* d_edge_u, int* d_edge_v, int edge_capacity,
                              int* d_overflow_count)
{
    int* d_edge_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_edge_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_edge_count, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_overflow_count, 0, sizeof(int)));

    const int block = kThreadsPerBlock;
    const int grid  = blocks_for(n, block);
    build_edges_euclid_kernel<<<grid, block>>>(n, d_xyz, d_point_key, d_unique_key, num_voxels,
                                               d_seg_start, d_idx_sorted, n_sorted,
                                               d_edge_u, d_edge_v, edge_capacity, d_edge_count, d_overflow_count);
    CUDA_CHECK_LAST_ERROR("build_edges_euclid_kernel launch");

    int edge_count = 0;
    CUDA_CHECK(cudaMemcpy(&edge_count, d_edge_count, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_edge_count));
    return (edge_count < edge_capacity) ? edge_count : edge_capacity;
}
