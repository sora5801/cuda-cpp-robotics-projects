// ===========================================================================
// kernels.cu — GPU implementation for project 02.11
//              Scan Context / ring-descriptor loop-closure search
//
// Three kernels, three GPU mapping ideas
// ---------------------------------------
//   sc_build_kernel        — a SCATTER: one thread per POINT, racing to
//                             claim a (ring,sector) cell's running max
//                             height via a CAS-loop atomic max on floats.
//   ring_key_kernel         — a small per-(scan,ring) REDUCE, one thread
//                             each — cheap enough that "one thread, one
//                             output element, sequential inner loop" is the
//                             whole story (no block cooperation needed).
//   sc_shift_distance_kernel — the project's hot loop: CANDIDATE x SHIFT
//                             parallelism, one BLOCK per pair, one THREAD
//                             per sector column, ending in a block-level
//                             REDUCE. This is the "GPU mapping" the catalog
//                             bullet names explicitly (kernels.cuh header).
//
// Read this after: kernels.cuh.  Companion oracle: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"

#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// atomicMaxFloat — a float-valued atomic max built from atomicCAS, the same
// CAS-loop-from-a-primitive teaching idiom this repo's 01.06 uses for its
// 64-bit atomicMin64/atomicMax64 (kernels.cu, cited): read the current bit
// pattern, reinterpret it as a float for the COMPARISON (not for ordering
// the bits themselves — the classic alternative "order-preserving uint
// encoding" trick avoids the reinterpret-per-iteration but obscures the
// actual comparison being made; this project prefers the direct, honest
// compare, "no black boxes" per CLAUDE.md paragraph 1), and retry the CAS
// only when some other thread's write raced us since the read.
//
// Why this is the right (and only) way to get atomicMax on floats: CUDA's
// native atomicMax intrinsic is defined for integer types only — there is
// no atomicMax(float*, float) in the hardware ISA. Every CUDA program that
// needs one builds it from atomicCAS, exactly like this.
//
// Correctness note: max() is ORDER-INDEPENDENT (unlike a running sum, whose
// float rounding depends on evaluation order), so however many threads race
// to update the SAME cell, in whatever order the scheduler interleaves
// them, the final value is EXACTLY max(all values ever attempted) — no
// rounding drift, no partial-sum artifacts. This is what lets
// VERIFY(scan_context) in main.cu compare the GPU matrix against a
// sequential CPU max almost bit-for-bit (kernels.cuh's file header states
// the small remaining source of disagreement: ring/sector BINNING, not the
// max itself, can rarely differ by one cell right at a boundary because
// atan2f/sqrtf and their std:: counterparts are not guaranteed bit-identical
// — THEORY.md "numerical considerations" quantifies how rarely).
// ---------------------------------------------------------------------------
__device__ inline float atomicMaxFloat(float* addr, float val)
{
    int* addr_as_int = reinterpret_cast<int*>(addr);
    int old = *addr_as_int, assumed;
    do {
        const float old_f = __int_as_float(old);
        if (old_f >= val) break;                          // already at least as large — nothing to do, no CAS spent
        assumed = old;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(val));
    } while (assumed != old);                              // old != assumed => another thread won the race; retry with the fresh value
    return __int_as_float(old);
}

// ---------------------------------------------------------------------------
// Device transcriptions of kernels.cuh's shared HOST-only inline formulas.
// Per 02.10's ruling (kernels.cuh's own header comment, cited): a plain
// `inline` function compiled by nvcc is HOST-only and cannot be called from
// a __global__ kernel, so a __device__ copy lives here, LITERALLY matching
// the host version line for line (the sinf/cosf vs sin/cos precision note
// in kernels.cuh's file header is the only place device and host math can
// legally disagree — everything else here is copy-identical on purpose).
// ---------------------------------------------------------------------------
__device__ inline int d_ring_index_from_range(float range_m)
{
    const float ring_width_m = kSensorMaxRangeM / static_cast<float>(kNumRing);
    int r = static_cast<int>(range_m / ring_width_m);
    if (r < 0) r = 0;
    if (r >= kNumRing) r = kNumRing - 1;
    return r;
}

__device__ inline int d_sector_index_from_xy(float x, float y)
{
    float az = atan2f(y, x);
    if (az < 0.0f) az += 2.0f * kPiF;
    const float sector_width_rad = (2.0f * kPiF) / static_cast<float>(kNumSector);
    int s = static_cast<int>(az / sector_width_rad);
    if (s < 0) s = 0;
    if (s >= kNumSector) s = kNumSector - 1;
    return s;
}

__device__ inline float d_column_cosine_distance(const float* sc_a, int col_a, const float* sc_b, int col_b)
{
    float dot = 0.0f, norm_a_sq = 0.0f, norm_b_sq = 0.0f;
    bool any_real_a = false, any_real_b = false;
    for (int r = 0; r < kNumRing; ++r) {
        const float raw_a = sc_a[r * kNumSector + col_a];
        const float raw_b = sc_b[r * kNumSector + col_b];
        const bool cell_a_empty = (raw_a <= kEmptyZ + 1.0f);
        const bool cell_b_empty = (raw_b <= kEmptyZ + 1.0f);
        const float a = cell_a_empty ? 0.0f : raw_a;   // mask: kEmptyZ must never enter the geometry directly
        const float b = cell_b_empty ? 0.0f : raw_b;
        if (!cell_a_empty) any_real_a = true;
        if (!cell_b_empty) any_real_b = true;
        dot       += a * b;
        norm_a_sq += a * a;
        norm_b_sq += b * b;
    }
    // Degenerate-column policy (kernels.cuh's column_cosine_distance carries
    // the full derivation, including the bug an earlier "raw-value-norm,
    // both empty -> max distance" version of this rule caused): agreeing-
    // empty is 0.0f, one-sided-empty is 1.0f.
    if (!any_real_a && !any_real_b) return 0.0f;
    if (!any_real_a || !any_real_b) return 1.0f;
    if (norm_a_sq < 1e-9f || norm_b_sq < 1e-9f) return 1.0f;
    return 1.0f - dot / (sqrtf(norm_a_sq) * sqrtf(norm_b_sq));
}

// ===========================================================================
// sc_init_kernel — set every cell of every one of n_scans matrices to the
// empty sentinel kEmptyZ (see kernels.cuh's file header for why a running
// MAX must seed from a value no real reading can ever beat — NOT 0.0f, the
// bug this project's own VERIFY sweep caught). Flat one-thread-per-cell map
// over n_scans*kScCells; nothing about this needs cooperation, so the
// simplest possible mapping is also the right one (contrast with the two
// kernels below, where the mapping IS the teaching point).
// ===========================================================================
__global__ void sc_init_kernel(int n_scans, float* __restrict__ sc_all)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_scans * kScCells;
    if (i >= total) return;
    sc_all[i] = kEmptyZ;
}

void launch_sc_init(int n_scans, float* d_sc_all)
{
    const int total = n_scans * kScCells;
    const int threads = 256;
    const int blocks = blocks_for(total, threads);
    sc_init_kernel<<<blocks, threads>>>(n_scans, d_sc_all);
    CUDA_CHECK_LAST_ERROR("sc_init_kernel launch");
}

// ===========================================================================
// sc_build_kernel — one thread per POINT (across ALL scans at once: the
// grid size is total_points, the sum of every scan's point count). Thread i
// owns point i; it reads its own (x,y,z), works out which scan it belongs
// to (point_scan_id[i], precomputed on the host from the ragged per-scan
// point counts — a simple prefix-sum bookkeeping job that gains nothing
// from a kernel of its own at this project's scale, main.cu does it once),
// computes its (ring,sector) cell, and atomically claims that cell's
// running max height.
//
// Memory behavior: xyz reads are coalesced (thread i reads xyz[3i..3i+2],
// consecutive threads read consecutive memory). The atomicMaxFloat WRITE
// target is data-dependent (two different points can land in the same
// cell, or in cells scattered across the whole sc_all array) — this is
// fundamentally a SCATTER, not a coalesced write, and that is fine: a
// scatter is the correct GPU pattern for "many independent producers, one
// shared accumulator per bucket", exactly the situation here (THEORY.md
// "the GPU mapping" names the CPU-side alternative this replaces: a
// sequential loop with no threading at all).
// ===========================================================================
__global__ void sc_build_kernel(int total_points,
                                const float* __restrict__ xyz,
                                const int32_t* __restrict__ point_scan_id,
                                float* __restrict__ sc_all)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_points) return;

    const float x = xyz[i * 3 + 0];
    const float y = xyz[i * 3 + 1];
    const float z = xyz[i * 3 + 2];

    const float range_m = sqrtf(x * x + y * y);       // PLANAR (horizontal) range — Scan Context bins by ground
                                                        // distance from the sensor, not full 3-D distance (THEORY.md)
    const int ring   = d_ring_index_from_range(range_m);
    const int sector = d_sector_index_from_xy(x, y);

    const int scan   = point_scan_id[i];
    const int cell   = scan * kScCells + ring * kNumSector + sector;

    atomicMaxFloat(&sc_all[cell], z);
}

void launch_sc_build(int total_points, const float* d_xyz, const int32_t* d_point_scan_id, float* d_sc_all)
{
    const int threads = 256;
    const int blocks = blocks_for(total_points, threads);
    sc_build_kernel<<<blocks, threads>>>(total_points, d_xyz, d_point_scan_id, d_sc_all);
    CUDA_CHECK_LAST_ERROR("sc_build_kernel launch");
}

// ===========================================================================
// ring_key_kernel — one thread per (scan, ring): walk that ring's
// kNumSector cells, count the non-empty ones, divide by kNumSector. A tiny
// sequential inner loop (60 iterations) per thread; n_scans*kNumRing
// threads total (128*20 = 2560 for the committed sample) — small enough
// that no block cooperation is worth the complexity it would add.
// ===========================================================================
__global__ void ring_key_kernel(int n_scans, const float* __restrict__ sc_all, float* __restrict__ ringkey_all)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // flattened (scan, ring) index
    const int total = n_scans * kNumRing;
    if (i >= total) return;

    const int scan = i / kNumRing;
    const int ring = i % kNumRing;
    const float* row = sc_all + scan * kScCells + ring * kNumSector;   // this ring's kNumSector cells

    int occupied = 0;
    for (int s = 0; s < kNumSector; ++s)
        if (row[s] > kEmptyZ + 1.0f) ++occupied;            // kEmptyZ is the documented empty sentinel (kernels.cuh)

    ringkey_all[i] = static_cast<float>(occupied) / static_cast<float>(kNumSector);
}

void launch_ring_key(int n_scans, const float* d_sc_all, float* d_ringkey_all)
{
    const int total = n_scans * kNumRing;
    const int threads = 256;
    const int blocks = blocks_for(total, threads);
    ring_key_kernel<<<blocks, threads>>>(n_scans, d_sc_all, d_ringkey_all);
    CUDA_CHECK_LAST_ERROR("ring_key_kernel launch");
}

// ===========================================================================
// sc_shift_distance_kernel — the project's hot loop and its named GPU
// mapping: CANDIDATE x SHIFT parallelism.
//
// Grid: dim3(num_candidates, kNumSector) blocks — one block per (candidate
// c, shift s) pair. Block: kThreadsPerBlock=64 threads (kNumSector=60
// padded up to the next warp multiple; the 4 extra threads are guarded off
// below — a cheap, standard way to keep every block warp-aligned without
// complicating the sector math).
//
// Thread-to-data mapping: thread t < kNumSector owns SECTOR t of the QUERY
// matrix. It compares that column against sector (t+shift) mod kNumSector
// of the CANDIDATE matrix (column_cosine_distance, the shared formula) —
// this is exactly the "column shift" THEORY.md derives as the rotation
// signature: sliding every query sector by `shift` and asking "does the
// candidate look like this rotated query" for every possible shift at once,
// one shift per block-column of the grid.
//
// Memory coalescing (why sector is the FAST-VARYING index in kernels.cuh's
// matrix layout): column_cosine_distance loops over kNumRing rings; at each
// ring r, thread t reads sc_query[r*kNumSector + t] — across the 60 (well,
// 64-padded) threads of the block, that is the CONTIGUOUS span
// sc_query[r*kNumSector .. r*kNumSector+59], one coalesced transaction per
// ring. The candidate read is a WARP-LEVEL PERMUTATION of the same span
// (thread t reads offset (t+shift) mod kNumSector instead of t) — the
// hardware coalesces on the SET of addresses a warp touches, not on which
// lane reads which offset, so this is still the same one-transaction-per-
// ring cost despite the wrap-around shift (THEORY.md "the GPU mapping"
// works this out in detail with a worked example).
//
// Reduction: each thread's column_cosine_distance() result goes into shared
// memory; a standard power-of-two tree reduction (64 is already a power of
// two, so no ragged-tail special case) sums them, and thread 0 divides by
// kNumSector and writes the block's one output float. This is a DIFFERENT
// summation order than reference_cpu.cpp's plain sequential sum — unlike
// the scatter-max above, a SUM is order-dependent at the ULP level, which
// is exactly why main.cu's VERIFY(shift_distance) gate uses a small
// tolerance instead of exact equality (kernels.cuh's file header flags this
// distinction on purpose).
// ===========================================================================
__global__ void sc_shift_distance_kernel(const float* __restrict__ sc_query,
                                         int num_candidates,
                                         const float* __restrict__ sc_candidates,
                                         float* __restrict__ out_dist)
{
    const int candidate = blockIdx.x;    // which candidate matrix this block scores
    const int shift     = blockIdx.y;    // which of the kNumSector column shifts this block tries
    const int t          = threadIdx.x;   // this thread's sector, 0..63 (60 real, 4 padding)

    __shared__ float sdata[64];          // one partial distance per thread; power-of-two size for a clean tree reduction

    if (t < kNumSector) {
        const float* cand = sc_candidates + static_cast<size_t>(candidate) * kScCells;
        const int shifted_col = (t + shift) % kNumSector;
        sdata[t] = d_column_cosine_distance(sc_query, t, cand, shifted_col);
    } else {
        sdata[t] = 0.0f;                 // padding lanes contribute nothing to the sum
    }
    __syncthreads();

    // Tree reduction: 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1. Deterministic
    // pairing (thread t combines with t+half while t<half) so every run on
    // every GPU sums in the SAME order — the reduction's own float-rounding
    // is therefore reproducible across runs, even though it differs from
    // the CPU oracle's sequential-sum order (see the file-header note above).
    for (int half = 32; half > 0; half >>= 1) {
        if (t < half) sdata[t] += sdata[t + half];
        __syncthreads();
    }

    if (t == 0)
        out_dist[static_cast<size_t>(candidate) * kNumSector + shift] = sdata[0] / static_cast<float>(kNumSector);
}

void launch_sc_shift_distance(const float* d_sc_query, int num_candidates,
                              const float* d_sc_candidates, float* d_out_dist)
{
    if (num_candidates < 1) {
        std::fprintf(stderr, "launch_sc_shift_distance: num_candidates must be >= 1 (got %d)\n", num_candidates);
        std::exit(EXIT_FAILURE);
    }
    const dim3 grid(static_cast<unsigned int>(num_candidates), static_cast<unsigned int>(kNumSector));
    const int threads = 64;   // kNumSector(60) padded to the next warp multiple — see the kernel's header comment
    sc_shift_distance_kernel<<<grid, threads>>>(d_sc_query, num_candidates, d_sc_candidates, d_out_dist);
    CUDA_CHECK_LAST_ERROR("sc_shift_distance_kernel launch");
}
