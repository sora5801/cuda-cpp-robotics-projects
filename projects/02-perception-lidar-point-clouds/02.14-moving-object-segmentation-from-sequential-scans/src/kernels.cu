// ===========================================================================
// kernels.cu — GPU kernels for project 02.14
//              Moving-object segmentation from sequential scans (online MOS)
//
// Role in the project
// -------------------
// All __global__ (GPU) code lives here, plus the host-side launch wrappers
// that own the grid/block math (CLAUDE.md §6.2: launch reasoning sits beside
// the launch). Five stages, in the order kernels.cuh's file header names
// them: (1) organize the current scan, (2) reproject + organize each
// previous scan, (3-4) residual + multi-scan MIN-fusion, (5) range-image CCL
// cleanup via the generic lock-free union-find (02.04/02.12 lineage).
//
// Every kernel here is a variant of the SAME shape 02.02/02.12/02.13 use:
// one thread per POINT for the scatter stages (embarrassingly parallel,
// atomicMin races resolve collisions deterministically), one thread per
// CELL for the finalize/residual/edge stages (kNumCells = 5,760 — tiny, but
// the mapping is the teaching point, not the occupancy: the SAME kernel
// shape scales to a real sensor's 100k+ point scans without changing a line).
//
// Read this after: main.cu, kernels.cuh.  Read this before: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"

constexpr int kThreadsPerBlock = 256;
static inline int blocks_for(int count, int threads) { return (count + threads - 1) / threads; }

// ===========================================================================
// Stage 1: organize the CURRENT scan into its own range image.
// ===========================================================================

// scatter_current_kernel — one thread per current-scan input point. See
// kernels.cuh's declaration comment for the full contract.
//
// Thread-to-data mapping: thread i owns input point i; it never touches
// another thread's point directly — all cross-thread interaction is the
// atomicMin race into cell_encoded[cell], the SAME nearest-wins pattern
// 02.02 introduces and 02.12/02.13 reuse (cited in kernels.cuh). Memory
// behavior: ring/az_bin/range_m are read with a coalesced access pattern
// (adjacent threads read adjacent array slots); the atomicMin write target
// is data-dependent (scattered), which is exactly why this is a SCATTER
// kernel, not a plain map — some contention is expected and harmless
// (atomics serialize correctly; they just cost more when many points share
// a cell, which they do not here except by rare, intentional coincidence).
__global__ void scatter_current_kernel(int n_points,
                                       const int* __restrict__ ring, const int* __restrict__ az_bin,
                                       const float* __restrict__ range_m,
                                       unsigned long long* __restrict__ cell_encoded)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) return;

    const int cell = organized_cell_index(ring[i], az_bin[i]);
    const unsigned long long key = pack_range_index(range_m[i], static_cast<uint32_t>(i));
    // atomicMin on an unsigned 64-bit key: because float_range_to_sortable_u32
    // preserves ordering for r>=0 (larger range -> larger bit pattern), the
    // packed key's numeric order IS range order, so a plain atomicMin
    // correctly keeps the SMALLEST range (nearest point) at each cell,
    // ties broken by point index (02.02's proof, cited in kernels.cuh).
    atomicMin(&cell_encoded[cell], static_cast<unsigned long long>(key));
}

// finalize_current_kernel — one thread per CELL: decode the winner and copy
// its range plus ground-truth payload (kernels.cuh's declaration comment).
__global__ void finalize_current_kernel(int num_cells,
                                        const unsigned long long* __restrict__ cell_encoded,
                                        const float* __restrict__ prange, const int* __restrict__ pcohort,
                                        const int* __restrict__ ptruth, const int* __restrict__ pdisocc,
                                        float* __restrict__ range_img,
                                        int* __restrict__ cohort_img, int* __restrict__ truth_img,
                                        int* __restrict__ disocc_img)
{
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= num_cells) return;

    const unsigned long long enc = cell_encoded[cell];
    if (enc == kEmptyCellEncoded) {
        // No return landed here: the project's "no data" sentinel throughout.
        range_img[cell] = 0.0f;
        cohort_img[cell] = kCohortNone;
        truth_img[cell] = -1;
        disocc_img[cell] = 0;
        return;
    }
    const uint32_t idx = unpack_point_index(enc);
    range_img[cell] = prange[idx];
    cohort_img[cell] = pcohort[idx];
    truth_img[cell] = ptruth[idx];
    disocc_img[cell] = pdisocc[idx];
}

// ===========================================================================
// Stage 2: reproject + organize ONE previous scan into the CURRENT frame.
// ===========================================================================

// reproject_scatter_kernel — one thread per point in previous scan j.
// Unlike scatter_current_kernel, this thread must first COMPUTE where its
// point lands (reproject_point_to_current + cell_for_local_point, both HD
// helpers shared with reference_cpu.cpp — kernels.cuh file header) before
// racing it in; everything after that is the identical atomicMin pattern.
__global__ void reproject_scatter_kernel(int n_points,
                                         const int* __restrict__ ring, const int* __restrict__ az_bin,
                                         const float* __restrict__ range_m,
                                         Pose pose_j, Pose pose_cur,
                                         unsigned long long* __restrict__ cell_encoded)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) return;

    // This point's LOCAL coordinates in scan j's OWN sensor frame, derived
    // from its native (ring, az_bin, range) — the single-sourced formula
    // (kernels.cuh) every stage in this project uses to avoid storing xyz
    // redundantly.
    const Vec3 p_local_j = local_point_from_ring_az(ring[i], az_bin[i], range_m[i]);

    // THE reprojection: express that same physical point in the CURRENT
    // sensor's local frame (kernels.cuh's reproject_point_to_current —
    // 02.08's deskew formula, cited, applied here across SCANS instead of
    // across a single sweep's continuous trajectory samples).
    const Vec3 p_in_current = reproject_point_to_current(pose_j, pose_cur, p_local_j);

    // Where does that reprojected point land in the CURRENT range image?
    // NOT this point's own (ring,az_bin) — those describe scan j's beam
    // geometry, not the current sensor's (kernels.cuh "The problem"). We
    // must re-derive elevation/azimuth from the TRANSFORMED xyz and snap to
    // the current sensor's nearest beam — a quantization step that is the
    // physical source of this project's reprojection-aliasing lesson.
    int cell; float range_reproj;
    cell_for_local_point(p_in_current, cell, range_reproj);

    const unsigned long long key = pack_range_index(range_reproj, static_cast<uint32_t>(i));
    atomicMin(&cell_encoded[cell], static_cast<unsigned long long>(key));
}

// finalize_prev_kernel — one thread per CELL: decode into a plain range
// image (no ground-truth payload — a previous scan's own labels are
// irrelevant; only CURRENT points are ever classified). The range value
// itself is recovered directly from the encoded key's upper 32 bits
// (unpack_range_m, kernels.cuh) — reproject_scatter_kernel encoded the
// REPROJECTED range, which exists nowhere else, so there is no separate
// per-point array to look it up in (this kernel's declaration comment).
__global__ void finalize_prev_kernel(int num_cells, const unsigned long long* __restrict__ cell_encoded,
                                     float* __restrict__ range_img_prev)
{
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= num_cells) return;
    const unsigned long long enc = cell_encoded[cell];
    range_img_prev[cell] = (enc == kEmptyCellEncoded) ? 0.0f : unpack_range_m(enc);
}

// ===========================================================================
// Stages 3-4: residual + multi-scan MIN-fusion (kernels.cuh file header
// steps 3-4 — read that derivation first; this kernel is its direct
// transcription).
// ===========================================================================

// residual_fuse_kernel — one thread per CURRENT cell.
//
// WHY MIN, not MEAN or a majority VOTE (the fusion choices kernels.cuh's
// declaration comment names as the alternatives)? A genuine mover
// perturbs EVERY one of the M comparisons by roughly the same amount as
// long as it keeps moving at roughly constant velocity (THEORY.md "The
// math" derives this): each previous scan sees the mover (or the
// background it vacated/revealed) offset by an amount roughly proportional
// to elapsed time, so |residual_j| grows steadily with lag j and is BOUNDED
// BELOW by the smallest-lag (freshest) comparison, which is already large
// for a fast mover. A one-off DISOCCLUSION artifact (01.21's lesson, cited
// in kernels.cuh), by contrast, typically appears in only ONE or a minority
// of the M comparisons (whichever previous scan happened to have the
// occluder in a different position relative to THIS specific cell) — MEAN
// would dilute a genuine mover's evidence by averaging in the OTHER
// comparisons' smaller residuals (which still exist even for a real mover,
// just smaller at longer lag if it started close to its old position), and
// VOTE (majority of M comparisons exceed threshold) is a reasonable middle
// ground but does not give a continuous score for the threshold sweep this
// project's window_size study performs. MIN is the CONSERVATIVE, most
// disocclusion-resistant choice: main.cu's disocclusion_mitigation gate
// measures exactly this — the false-positive rate MIN-fusion achieves on
// the wall's disocclusion band, WITH (M=4) vs WITHOUT (M=1, no fusion at
// all, nothing to be conservative ABOUT) the multi-scan check. The
// documented COST of this choice appears in the temporal_boundary [info]
// report: MIN-fusion is unable to detect an object that has JUST become
// stationary, because the freshest (smallest-lag) comparison alone already
// drags the minimum near zero, regardless of how large M is (THEORY.md
// "Numerical considerations" derives this as a fundamental property of MIN
// fusion, not a bug).
__global__ void residual_fuse_kernel(const float* __restrict__ range_img_cur,
                                     const float* const* __restrict__ prev_range_imgs, int window_m,
                                     float threshold_m,
                                     float* __restrict__ fused_evidence_out,
                                     int* __restrict__ sign_out, int* __restrict__ candidate_out)
{
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= kNumCells) return;

    const float r_cur = range_img_cur[cell];
    if (r_cur <= 0.0f) {
        // No current-scan point here at all — nothing to classify.
        fused_evidence_out[cell] = -1.0f;
        sign_out[cell] = 0;
        candidate_out[cell] = 0;
        return;
    }

    float min_abs = -1.0f;      // MIN |residual_j| over valid, included j (file header step 4); -1 = none yet
    float nearest_signed = 0.0f;  // signed residual of the smallest-lag VALID comparison (the representative sign)
    bool have_nearest = false;

    // window_m in [1,kMaxWindowM]; prev_range_imgs[0] is lag-1 (freshest),
    // prev_range_imgs[window_m-1] is the oldest INCLUDED comparison — this
    // loop is the whole "how much history did we look at" knob the
    // window_size study sweeps (main.cu calls this kernel once per M).
    for (int lag = 0; lag < window_m; ++lag) {
        const float r_prev = prev_range_imgs[lag][cell];
        if (r_prev <= 0.0f) continue;   // that previous scan had no return here — no evidence from it

        const float residual = r_cur - r_prev;              // SIGNED (kernels.cuh file header step 3)
        const float abs_residual = fabsf(residual);
        if (min_abs < 0.0f || abs_residual < min_abs) min_abs = abs_residual;
        if (!have_nearest) { nearest_signed = residual; have_nearest = true; }
    }

    fused_evidence_out[cell] = min_abs;   // stays -1.0f if no included previous scan had evidence
    sign_out[cell] = have_nearest ? (nearest_signed > 0.0f ? 1 : (nearest_signed < 0.0f ? -1 : 0)) : 0;
    // Insufficient evidence (min_abs < 0) is treated as STATIC — a
    // conservative default matching every occupancy-evidence algorithm in
    // this repo (02.13's identical "no evidence => no removal" stance,
    // cited): a newly-observed cell with no previous-scan comparison at all
    // cannot be PROVEN to have moved, so it is not flagged (README
    // "Limitations" states this honestly).
    candidate_out[cell] = (min_abs >= 0.0f && min_abs >= threshold_m) ? 1 : 0;
}

// ===========================================================================
// Stage 5: range-image CCL cleanup (02.12's union-find lineage, cited).
// ===========================================================================

// build_moving_edges_kernel — one thread per CELL: mirrors 02.12's
// depth_edges_kernel FORWARD-neighbor pattern exactly (only the two
// "forward" neighbors — ring+1 same column; same ring, column+1 WITH
// WRAP-AROUND — so every undirected image edge is considered exactly once,
// by its lower-ring/lower-column endpoint), but the connect PREDICATE is
// "both endpoints are candidate_moving" instead of a beta test.
__global__ void build_moving_edges_kernel(int num_cells, const int* __restrict__ candidate,
                                          int* __restrict__ edge_u, int* __restrict__ edge_v,
                                          int* __restrict__ edge_count)
{
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= num_cells) return;
    if (!candidate[cell]) return;   // only candidate-moving cells can start an edge

    const int ring = cell / kAzimuthBins;
    const int az = cell % kAzimuthBins;

    // Forward neighbor 1: next ring, same azimuth column (no wrap — ring 15
    // has no ring 16, the sensor's vertical FOV is NOT a full circle).
    if (ring + 1 < kNumBeams) {
        const int nb = organized_cell_index(ring + 1, az);
        if (candidate[nb]) {
            const int e = atomicAdd(edge_count, 1);
            edge_u[e] = cell; edge_v[e] = nb;
        }
    }
    // Forward neighbor 2: same ring, next azimuth column, WRAPPING at
    // kAzimuthBins-1 -> 0 (the sensor spins a full circle — 02.12's
    // identical wrap-around discipline, cited).
    {
        const int naz = (az + 1) % kAzimuthBins;
        const int nb = organized_cell_index(ring, naz);
        if (candidate[nb]) {
            const int e = atomicAdd(edge_count, 1);
            edge_u[e] = cell; edge_v[e] = nb;
        }
    }
}

// ---- generic lock-free GPU union-find (02.04/02.12 Method A, cited) ------
//
// The three kernels below are GENERIC over any edge list — this project
// reuses them UNCHANGED from 02.12's design (retyped here per the repo's
// self-containment rule, CLAUDE.md §4) purely to cluster the moving-cell
// adjacency graph built above. See 02.12's kernels.cu for the full
// correctness argument (the monotone-parent property under union-by-min);
// summarized here: parent[] only ever decreases, so repeated sweeps
// provably converge to each component's minimum linear index as its root,
// regardless of scheduling order.

__global__ void uf_init_kernel(int n, int* __restrict__ parent)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) parent[i] = i;   // every element starts as its own singleton root
}

// find_with_path_halving — device-local helper: follow parent pointers to
// the root, halving the path as we go (each visited node's parent is
// snapped to its grandparent) — an amortized-cheap partial compression that
// keeps later find() calls fast without a second full pass.
__device__ __forceinline__ int find_with_path_halving(int* parent, int i)
{
    while (parent[i] != i) {
        const int gp = parent[parent[i]];
        parent[i] = gp;   // path halving: skip one level
        i = gp;
    }
    return i;
}

__global__ void uf_union_sweep_kernel(int num_edges, const int* __restrict__ edge_u,
                                      const int* __restrict__ edge_v,
                                      int* __restrict__ parent, int* __restrict__ changed)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_edges) return;

    int ru = find_with_path_halving(parent, edge_u[e]);
    int rv = find_with_path_halving(parent, edge_v[e]);
    if (ru == rv) return;   // already in the same component — nothing to do

    // Union-by-MIN via atomicCAS retry: always attach the LARGER root under
    // the SMALLER (never the reverse), which is what guarantees parent[]
    // only ever decreases over the whole sweep loop (the monotone property
    // 02.04/02.12 prove converges regardless of thread scheduling).
    if (ru > rv) { const int t = ru; ru = rv; rv = t; }
    const int prev = atomicCAS(&parent[rv], rv, ru);
    if (prev == rv) {
        *changed = 1;   // a union happened this sweep — the host's convergence loop keeps going
    }
    // If the CAS lost the race (another thread already relinked rv), this
    // edge simply gets picked up again on a LATER sweep — always safe,
    // never a correctness issue, only a (bounded) extra sweep.
}

__global__ void uf_finalize_kernel(int n, int* __restrict__ parent)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // Full find-to-fixpoint (no halving needed here — this runs ONCE, after
    // convergence): parent[i] becomes EXACTLY i's canonical root, the
    // postcondition every downstream consumer (main.cu's component-size
    // count) assumes.
    int r = i;
    while (parent[r] != r) r = parent[r];
    parent[i] = r;
}

// ===========================================================================
// Host launch wrappers — grid/block math + the mandatory post-launch error
// check (CLAUDE.md §6.1 rule 7). block=256: a warp multiple, the repo's
// standard default (good occupancy on sm_75..sm_89 without starving
// per-block resources — 08.01's identical launch-configuration reasoning).
// ===========================================================================

void launch_scatter_current(int n_points, const int* d_ring, const int* d_az_bin, const float* d_range_m,
                            unsigned long long* d_cell_encoded)
{
    scatter_current_kernel<<<blocks_for(n_points, kThreadsPerBlock), kThreadsPerBlock>>>(
        n_points, d_ring, d_az_bin, d_range_m, d_cell_encoded);
    CUDA_CHECK_LAST_ERROR("scatter_current_kernel launch");
}

void launch_finalize_current(int num_cells, const unsigned long long* d_cell_encoded,
                             const float* d_prange, const int* d_pcohort, const int* d_ptruth, const int* d_pdisocc,
                             float* d_range_img, int* d_cohort_img, int* d_truth_img, int* d_disocc_img)
{
    finalize_current_kernel<<<blocks_for(num_cells, kThreadsPerBlock), kThreadsPerBlock>>>(
        num_cells, d_cell_encoded, d_prange, d_pcohort, d_ptruth, d_pdisocc,
        d_range_img, d_cohort_img, d_truth_img, d_disocc_img);
    CUDA_CHECK_LAST_ERROR("finalize_current_kernel launch");
}

void launch_reproject_scatter(int n_points, const int* d_ring, const int* d_az_bin, const float* d_range_m,
                              Pose pose_j, Pose pose_cur, unsigned long long* d_cell_encoded)
{
    reproject_scatter_kernel<<<blocks_for(n_points, kThreadsPerBlock), kThreadsPerBlock>>>(
        n_points, d_ring, d_az_bin, d_range_m, pose_j, pose_cur, d_cell_encoded);
    CUDA_CHECK_LAST_ERROR("reproject_scatter_kernel launch");
}

void launch_finalize_prev(int num_cells, const unsigned long long* d_cell_encoded, float* d_range_img_prev)
{
    finalize_prev_kernel<<<blocks_for(num_cells, kThreadsPerBlock), kThreadsPerBlock>>>(
        num_cells, d_cell_encoded, d_range_img_prev);
    CUDA_CHECK_LAST_ERROR("finalize_prev_kernel launch");
}

void launch_residual_fuse(const float* d_range_img_cur, const float* const* d_prev_range_imgs, int window_m,
                          float threshold_m, float* d_fused_evidence, int* d_sign, int* d_candidate)
{
    residual_fuse_kernel<<<blocks_for(kNumCells, kThreadsPerBlock), kThreadsPerBlock>>>(
        d_range_img_cur, d_prev_range_imgs, window_m, threshold_m, d_fused_evidence, d_sign, d_candidate);
    CUDA_CHECK_LAST_ERROR("residual_fuse_kernel launch");
}

int launch_build_moving_edges(int num_cells, const int* d_candidate, int* d_edge_u, int* d_edge_v)
{
    int* d_edge_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_edge_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_edge_count, 0, sizeof(int)));
    build_moving_edges_kernel<<<blocks_for(num_cells, kThreadsPerBlock), kThreadsPerBlock>>>(
        num_cells, d_candidate, d_edge_u, d_edge_v, d_edge_count);
    CUDA_CHECK_LAST_ERROR("build_moving_edges_kernel launch");
    int count = 0;
    CUDA_CHECK(cudaMemcpy(&count, d_edge_count, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_edge_count));
    return count;
}

void launch_uf_init(int n, int* d_parent)
{
    uf_init_kernel<<<blocks_for(n, kThreadsPerBlock), kThreadsPerBlock>>>(n, d_parent);
    CUDA_CHECK_LAST_ERROR("uf_init_kernel launch");
}

bool launch_uf_union_sweep(int num_edges, const int* d_edge_u, const int* d_edge_v,
                           int* d_parent, int* d_changed)
{
    CUDA_CHECK(cudaMemset(d_changed, 0, sizeof(int)));
    if (num_edges > 0) {
        uf_union_sweep_kernel<<<blocks_for(num_edges, kThreadsPerBlock), kThreadsPerBlock>>>(
            num_edges, d_edge_u, d_edge_v, d_parent, d_changed);
        CUDA_CHECK_LAST_ERROR("uf_union_sweep_kernel launch");
    }
    int changed = 0;
    CUDA_CHECK(cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
    return changed != 0;
}

void launch_uf_finalize(int n, int* d_parent)
{
    uf_finalize_kernel<<<blocks_for(n, kThreadsPerBlock), kThreadsPerBlock>>>(n, d_parent);
    CUDA_CHECK_LAST_ERROR("uf_finalize_kernel launch");
}

