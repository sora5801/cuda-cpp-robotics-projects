// ===========================================================================
// kernels.cu — GPU kernels for project 02.02
//              ROI crop, passthrough, organized<->unorganized conversion
//              kernels — THE SCAN CHAPTER is the didactic heart; everything
//              else in this file (predicates, compaction, the organized
//              scatter) is a CONSUMER of the scan primitive built here.
//
// Read this after: main.cu, kernels.cuh.  Read this before: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"

#include <cstdio>
#include <cstdlib>
#include <vector>

// Thrust: header-only part of the CUDA Toolkit (no separate .lib — see
// build/*.vcxproj's Thrust comment, flags copied verbatim from 02.01's
// ratified precedent, cited there and in this project's README "Prior art").
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

// ===========================================================================
// DEVICE-SIDE DATA — __constant__ duplicates of kernels.cuh's ARRAY/STRUCT
// constexpr constants.
//
// A scalar `constexpr` (kPassthroughZMin, kFx, kAzimuthBins, ...) compiles
// straight into device code as an inline immediate — nvcc constant-folds
// any scalar literal-typed constexpr wherever it is read, host or device,
// with no qualifier needed. An ARRAY or STRUCT constexpr does NOT get this
// treatment the moment it is indexed/accessed at a non-compile-time-known
// offset (kBoxMin[axis], kTCameraLidar.R[...]): that requires an actual
// memory address, and a plain host `constexpr` array has no device-visible
// address — nvcc reports exactly this ("identifier ... is undefined in
// device code"), which is how this project's own build first caught the
// mistake (kept here as an honest record, not smoothed over). The fix,
// applied repo-wide wherever compound constants cross the host/device
// line: duplicate them here as __constant__ (fast, read-only, broadcast-
// friendly device memory — ideal for small lookup tables/matrices every
// thread reads identically) with a `d_` prefix, DEVICE TRANSCRIPTIONS of
// kernels.cuh's host values, kept in lockstep by hand.
// ===========================================================================
__device__ __constant__ float d_kBoxMin[3] = { -4.0f, -4.0f, -1.5f };
__device__ __constant__ float d_kBoxMax[3] = {  4.0f,  4.0f,  1.0f };

__device__ __constant__ float d_kBeamElevRad[kNumBeams] = {
    -0.26179939f, -0.22689280f, -0.19198622f, -0.15707963f,
    -0.12217305f, -0.08726646f, -0.05235988f, -0.01745329f,
     0.01745329f,  0.05235988f,  0.08726646f,  0.12217305f,
     0.15707963f,  0.19198622f,  0.22689280f,  0.26179939f
};

__device__ __constant__ Rigid3 d_kTCameraLidar = {
    { 0.0f, -1.0f,  0.0f,
      0.0f,  0.0f, -1.0f,
      1.0f,  0.0f,  0.0f },
    { 0.0f, -0.30f, -0.05f }
};

// ===========================================================================
// DEVICE TRANSCRIPTIONS of kernels.cuh's host-only predicate/geometry
// functions. kernels.cuh explains WHY these exist as duplicates rather than
// shared calls (its "why this header is CUDA-qualifier-free" note): a plain
// inline host function cannot be called from a __global__ kernel. Each
// function below is a byte-for-byte transcription of its kernels.cuh
// counterpart (reading the __constant__ duplicates above in place of the
// header's host constants) — kept in lockstep BY HAND, with the drift risk
// covered by main.cu's VERIFY(predicate_correctness) / GATE
// frustum_geometry / GATE collision_accounting, each of which compares a
// GPU answer (computed via these functions) against a CPU answer (computed
// via kernels.cuh's shared host functions, through reference_cpu.cpp's
// independent loops).
// ===========================================================================

// DEVICE TRANSCRIPTION of is_invalid_point().
__device__ __forceinline__ bool d_is_invalid_point(float x)
{
    return x != x;   // NaN self-inequality; see kernels.cuh for the convention
}

// DEVICE TRANSCRIPTION of is_passthrough().
__device__ __forceinline__ bool d_is_passthrough(float z)
{
    return z >= kPassthroughZMin && z <= kPassthroughZMax;
}

// DEVICE TRANSCRIPTION of is_in_box() — reads the d_kBoxMin/Max __constant__
// duplicates (see the file-header note on why arrays need this).
__device__ __forceinline__ bool d_is_in_box(float x, float y, float z)
{
    return x >= d_kBoxMin[0] && x <= d_kBoxMax[0] &&
          y >= d_kBoxMin[1] && y <= d_kBoxMax[1] &&
          z >= d_kBoxMin[2] && z <= d_kBoxMax[2];
}

// DEVICE TRANSCRIPTION of transform_to_camera() — reads the d_kTCameraLidar
// __constant__ duplicate.
__device__ __forceinline__ void d_transform_to_camera(float x, float y, float z,
                                                       float& cx, float& cy, float& cz)
{
    const Rigid3& T = d_kTCameraLidar;
    cx = T.R[0] * x + T.R[1] * y + T.R[2] * z + T.t[0];
    cy = T.R[3] * x + T.R[4] * y + T.R[5] * z + T.t[1];
    cz = T.R[6] * x + T.R[7] * y + T.R[8] * z + T.t[2];
}

// DEVICE TRANSCRIPTION of is_in_frustum() — the five-plane test.
__device__ __forceinline__ bool d_is_in_frustum(float x, float y, float z)
{
    float cx, cy, cz;
    d_transform_to_camera(x, y, z, cx, cy, cz);
    if (cz < kFrustumNearM) return false;
    if (kFx * cx + kCx * cz < 0.0f) return false;
    if (-kFx * cx + (static_cast<float>(kImgW - 1) - kCx) * cz < 0.0f) return false;
    if (kFy * cy + kCy * cz < 0.0f) return false;
    if (-kFy * cy + (static_cast<float>(kImgH - 1) - kCy) * cz < 0.0f) return false;
    return true;
}

// DEVICE TRANSCRIPTION of azimuth_bin_of() — atan2f is the CUDA math
// library's device-callable float arctangent-of-two-args, bit-for-bit the
// same algorithm family as std::atan2 (both IEEE-754-correctly-rounded to
// within the platform's documented ULP bound; THEORY.md "Numerical
// considerations" discusses why this project never needed to care about
// the sub-ULP difference between the two libraries' implementations).
__device__ __forceinline__ int d_azimuth_bin_of(float x, float y)
{
    float az = atan2f(y, x);
    if (az < 0.0f) az += 2.0f * kPi;
    int bin = static_cast<int>(az / (2.0f * kPi / static_cast<float>(kAzimuthBins)));
    if (bin >= kAzimuthBins) bin = kAzimuthBins - 1;
    if (bin < 0) bin = 0;
    return bin;
}

// DEVICE TRANSCRIPTION of nearest_ring_of() — reads the d_kBeamElevRad
// __constant__ duplicate.
__device__ __forceinline__ int d_nearest_ring_of(float x, float y, float z)
{
    const float horiz = sqrtf(x * x + y * y);
    const float el = atan2f(z, horiz);
    int best = 0;
    float best_diff = 1.0e30f;
    for (int i = 0; i < kNumBeams; ++i) {
        const float diff = fabsf(el - d_kBeamElevRad[i]);
        if (diff < best_diff) { best_diff = diff; best = i; }
    }
    return best;
}

// DEVICE TRANSCRIPTION of pack_range_index()/float_range_to_sortable_u32() —
// __float_as_uint is CUDA's device intrinsic for the exact bit
// reinterpretation std::memcpy performs on the host (no rounding, a single
// instruction on every architecture this project targets).
__device__ __forceinline__ uint64_t d_pack_range_index(float range_m, uint32_t point_idx)
{
    const uint32_t bits = __float_as_uint(range_m);
    return (static_cast<uint64_t>(bits) << 32) | point_idx;
}

// ===========================================================================
// Predicate kernels — one thread per POINT, pure MAP: flags[i] depends only
// on xyz[i]. Grid-stride is unnecessary at this project's problem sizes
// (tens of thousands of points, comfortably under one launch's max grid),
// so these use the simple "one thread, one point, guard the tail" form —
// contrast with SAXPY's grid-stride loop in the scaffold placeholder;
// THEORY.md "The GPU mapping" discusses when each form is the right choice.
// ===========================================================================

__global__ void passthrough_predicate_kernel(int n, const float* __restrict__ xyz, int* __restrict__ flags)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    flags[i] = d_is_passthrough(xyz[i * 3 + 2]) ? 1 : 0;
}

__global__ void box_predicate_kernel(int n, const float* __restrict__ xyz, int* __restrict__ flags)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    flags[i] = d_is_in_box(xyz[i * 3 + 0], xyz[i * 3 + 1], xyz[i * 3 + 2]) ? 1 : 0;
}

__global__ void frustum_predicate_kernel(int n, const float* __restrict__ xyz, int* __restrict__ flags)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    flags[i] = d_is_in_frustum(xyz[i * 3 + 0], xyz[i * 3 + 1], xyz[i * 3 + 2]) ? 1 : 0;
}

// fused_predicate_kernel — the FUSED filter: all three tests in ONE kernel
// launch, reading xyz ONCE. Contrast with the CHAINED pipeline (main.cu),
// which runs passthrough_predicate_kernel -> compact -> box_predicate_kernel
// -> compact -> frustum_predicate_kernel -> compact as three SEPARATE
// launches, each re-reading and re-writing a (shrinking) xyz array from
// global memory. GATE fused_vs_chained (main.cu) proves both reach the
// IDENTICAL surviving point set (predicate composition is associative and
// commutative for a logical AND, so this is a correctness invariant, not a
// coincidence) while measuring the memory-traffic difference — the
// classic kernel-fusion lesson (cited from 01.01/01.23's ISP-stage-fusion
// precedent in this repo).
__global__ void fused_predicate_kernel(int n, const float* __restrict__ xyz, int* __restrict__ flags)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = xyz[i * 3 + 0], y = xyz[i * 3 + 1], z = xyz[i * 3 + 2];
    flags[i] = (d_is_passthrough(z) && d_is_in_box(x, y, z) && d_is_in_frustum(x, y, z)) ? 1 : 0;
}

// valid_predicate_kernel — the organized->unorganized filter: "keep" means
// "not NaN". Applied to the FLATTENED organized grid (ring-major, see
// kernels.cuh), so compaction here is literally the same primitive as
// every ROI filter above, with the simplest possible predicate.
__global__ void valid_predicate_kernel(int n, const float* __restrict__ xyz, int* __restrict__ flags)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    flags[i] = d_is_invalid_point(xyz[i * 3 + 0]) ? 0 : 1;
}

// ===========================================================================
// THE SCAN CHAPTER — a work-efficient (Blelloch 1990) TWO-LEVEL exclusive
// prefix scan, implemented by hand.
//
// WHY A SCAN SOLVES COMPACTION (the one-paragraph proof; THEORY.md "The
// algorithm" gives the full derivation): let flags[0..n) be 0/1 "keep this
// element" bits. The EXCLUSIVE prefix sum scan[i] = flags[0]+...+flags[i-1]
// counts how many elements BEFORE i were kept — which is EXACTLY the
// destination slot a kept element i must scatter to, because every earlier
// kept element already claimed slots 0..scan[i]-1 and no later element can
// ever claim a slot before it (scan is monotonic non-decreasing). One scan
// + one scatter = a stable, order-preserving compaction. No hand-rolled
// nested loop, no locks, no atomics needed for the scatter step at all.
//
// WHY NOT THE NAIVE SERIAL SCAN ON THE GPU: a single thread computing
// scan[i] = scan[i-1] + flags[i-1] is O(n) WORK but also O(n) DEPTH (n
// sequential steps) — on a GPU with thousands of idle threads, that is a
// wasted machine. The classic Hillis-Steele scan trades EXTRA work for
// LESS depth (O(n log n) work, O(log n) depth) and is simple to write, but
// at n in the tens of thousands the constant-factor difference against a
// WORK-EFFICIENT scan (Blelloch: O(n) work, O(log n) depth — the same
// asymptotic work as the serial version, but parallel DEPTH) is real
// bandwidth left on the table. This project teaches the harder, better one
// on purpose (THEORY.md "The math" derives both bounds side by side).
//
// THE TWO-LEVEL COMPOSITION (why 2 kernels of 1 kind + 1 kernel of another
// solve an ARBITRARILY large array with a scan primitive that only handles
// kScanElemsPerBlock=512 elements natively):
//
//   Level 1: blelloch_block_scan_kernel, launched over CEIL(n/512) blocks.
//            Each block scans its OWN 512-element span in shared memory —
//            a LOCAL, per-block answer — and records its span's TOTAL SUM
//            into block_sums[blockIdx.x].
//   Level 2: THE SAME KERNEL, launched a SECOND time as a SINGLE block over
//            block_sums (valid as long as the number of blocks from level 1
//            is itself <= 512 — see launch_scan_blelloch's runtime guard).
//            This produces the EXCLUSIVE scan of the per-block totals: the
//            OFFSET each block's local answers must be shifted by to become
//            part of one GLOBAL scan.
//   Combine: add_block_offsets_kernel adds block b's offset (from level 2)
//            to every element level 1 wrote for block b.
//
// This "reduce locally, scan the reductions, broadcast back" pattern is the
// standard template for turning any block-local primitive into a
// whole-array primitive — you will meet it again in this repo wherever an
// operation needs global information a single block cannot hold (e.g.
// segmented reductions, histogram merges).
// ===========================================================================

// ---------------------------------------------------------------------------
// blelloch_block_scan_kernel — one block's local exclusive scan of up to
// kScanElemsPerBlock=512 elements, via the classic Blelloch up-sweep +
// down-sweep in SHARED MEMORY (global memory is far too slow for the
// O(log n) rounds of tiny read-modify-write steps this algorithm performs;
// shared memory's ~20x lower latency and per-SM locality is what makes an
// in-place tree scan practical at all).
//
// Thread-to-data mapping: thread `tid` (0..255) owns TWO shared-memory
// slots, `ai = tid` and `bi = tid + 256` — so 256 threads cover 512
// elements, the "2 elements per thread" Blelloch layout kernels.cuh's
// kScanElemsPerBlock comment explains. Global element indices are
// block_offset + ai / block_offset + bi; a guard treats any global index
// >= n as an implicit 0 (this project's padding strategy — no explicit
// padded array, just a per-thread guard, since n is rarely a multiple of
// 512 and allocating a padded copy would cost a whole extra buffer for no
// benefit).
//
//                     UP-SWEEP (reduce), 9 rounds for 512 elements
//   round 0 (offset=1):  pairs (0,1)(2,3)(4,5)...        -> partial sums at odd indices
//   round 1 (offset=2):  pairs (1,3)(5,7)...              -> partial sums at indices 3,7,11...
//   ...                  each round HALVES the active thread count and
//                        DOUBLES the stride between the pair being combined
//   round 8 (offset=256): pair (255,511)                  -> temp[511] now holds the BLOCK TOTAL
//
//                     DOWN-SWEEP (distribute), 9 rounds, mirror image
//   Before it starts: temp[511] (the total) is COPIED to block_sums[] and
//   then RESET TO 0 — this converts the tree from an INCLUSIVE reduction
//   into the root of an EXCLUSIVE scan (the classic "clear the last element"
//   trick: down-sweep repeatedly swaps a node's OLD value down to its left
//   child and writes (old left + old right) to the right child, which is
//   exactly how a 0 planted at the root propagates into the correct
//   exclusive prefix at every leaf after log2(512)=9 rounds).
//
// BANK-CONFLICT HONESTY (CLAUDE.md's explicit ask — no silent optimism):
// shared memory is organized into 32 banks; a WARP accessing 32
// consecutive 4-byte words hits 32 DIFFERENT banks (free, one transaction).
// This kernel's up-sweep/down-sweep indices `offset*(2*tid+1)-1` STRIDE by
// `offset` (1, 2, 4, ..., 256 across the rounds) — once offset >= 32, EVERY
// active thread in a warp lands on indices that are multiples of the SAME
// bank stride, causing up to 32-WAY bank conflicts in the later up-sweep
// rounds and earlier down-sweep rounds (serializing what should be one
// transaction into up to 32). The well-known fix (Harris/Sengupta/Owens,
// GPU Gems 3 ch.39) pads every shared-memory index by
// `idx + (idx >> LOG_NUM_BANKS)` so no two threads' indices ever share a
// bank at any stride. This project deliberately DOES NOT implement that
// padding — kScanElemsPerBlock=512 keeps the shared-memory footprint (2 KB)
// and the kernel body small enough to read in one sitting, and the
// conflicts cost real but bounded time at this project's problem sizes
// (THEORY.md "The GPU mapping" reports the measured cost); a learner who
// wants the padded, conflict-free version has the exact formula above and
// this comment as a starting point (CLAUDE.md section 1: "explain the
// faster version in comments" when choosing the simpler one to teach).
//
// Reused for BOTH scan levels (see the file-level "THE SCAN CHAPTER"
// comment): `block_sums` may be nullptr (level 2's single-block call has no
// further level to feed).
// ---------------------------------------------------------------------------
__global__ void blelloch_block_scan_kernel(int n, const int* __restrict__ in,
                                           int* __restrict__ out_exclusive,
                                           int* __restrict__ block_sums)
{
    __shared__ int temp[kScanElemsPerBlock];   // 512 ints = 2 KiB shared mem per block

    const int tid = threadIdx.x;                          // 0..255
    const int block_offset = blockIdx.x * kScanElemsPerBlock;

    const int ai = tid;                                   // this thread's FIRST owned slot, 0..255
    const int bi = tid + kScanBlockThreads;                // this thread's SECOND owned slot, 256..511
    const int global_ai = block_offset + ai;
    const int global_bi = block_offset + bi;

    // Load phase: out-of-range global indices contribute 0 (the padding
    // guard described above) rather than reading past the real array.
    temp[ai] = (global_ai < n) ? in[global_ai] : 0;
    temp[bi] = (global_bi < n) ? in[global_bi] : 0;

    // ---- up-sweep (reduce): build partial sums up the implicit tree ----
    int offset = 1;
    for (int d = kScanElemsPerBlock >> 1; d > 0; d >>= 1) {
        __syncthreads();   // every round depends on ALL of the previous round's writes
        if (tid < d) {
            const int idx_a = offset * (2 * tid + 1) - 1;
            const int idx_b = offset * (2 * tid + 2) - 1;
            temp[idx_b] += temp[idx_a];
        }
        offset *= 2;
    }

    // Root now holds this block's TOTAL sum. Hand it to the next level
    // (level-1 callers pass block_sums; level-2's single-block call passes
    // nullptr, since there is no further level to feed), then zero the
    // root — the "plant a 0" step that turns this into an EXCLUSIVE scan.
    if (tid == 0) {
        if (block_sums != nullptr) block_sums[blockIdx.x] = temp[kScanElemsPerBlock - 1];
        temp[kScanElemsPerBlock - 1] = 0;
    }

    // ---- down-sweep (distribute): push the exclusive prefix back down ----
    for (int d = 1; d < kScanElemsPerBlock; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            const int idx_a = offset * (2 * tid + 1) - 1;
            const int idx_b = offset * (2 * tid + 2) - 1;
            const int t = temp[idx_a];
            temp[idx_a] = temp[idx_b];        // left child gets the parent's old (exclusive) value
            temp[idx_b] += t;                  // right child gets parent's value + old left child
        }
    }
    __syncthreads();   // the whole block must see the finished tree before any thread writes out

    if (global_ai < n) out_exclusive[global_ai] = temp[ai];
    if (global_bi < n) out_exclusive[global_bi] = temp[bi];
}

// ---------------------------------------------------------------------------
// add_block_offsets_kernel — the "combine" step of the two-level scan
// (see the chapter comment above): add block b's GLOBAL offset (the
// EXCLUSIVE scan of block sums, computed by level 2) to every LOCAL scan
// value level 1 wrote for block b. Same thread-to-data mapping as the scan
// kernel (2 elements/thread) for symmetry, though this kernel does no tree
// work — a plain, coalesced read-add-write map.
// ---------------------------------------------------------------------------
__global__ void add_block_offsets_kernel(int n, int* __restrict__ out_exclusive,
                                         const int* __restrict__ scanned_block_sums)
{
    const int tid = threadIdx.x;
    const int block_offset = blockIdx.x * kScanElemsPerBlock;
    const int global_ai = block_offset + tid;
    const int global_bi = block_offset + tid + kScanBlockThreads;
    const int off = scanned_block_sums[blockIdx.x];   // this block's exclusive prefix among block totals

    if (global_ai < n) out_exclusive[global_ai] += off;
    if (global_bi < n) out_exclusive[global_bi] += off;
}

// ---------------------------------------------------------------------------
// launch_scan_blelloch — the two-level scan end to end (see kernels.cuh's
// declaration for the interface contract).
//
// Scratch ownership: this wrapper cudaMalloc/cudaFree's its own block_sums
// / block_sums_scanned buffers rather than requiring the caller to thread
// scratch pointers through every call site. Every call in this project's
// demo happens a small, fixed number of times (a handful of compactions +
// the dedicated scan-scaling stage), so the extra allocator overhead is
// immeasurable against the kernel work itself — CLAUDE.md's "teaching
// beats cleverness": a self-contained function signature is easier to
// read and call correctly than one with three scratch-pointer parameters
// whose sizes the caller must get exactly right.
//
// Two-level scaling limit (an honest, checked boundary, not silently
// assumed): the level-2 call scans block_sums in a SINGLE block, so it can
// only handle up to kScanElemsPerBlock=512 block sums — i.e. this
// implementation's largest exact array size is 512*512 = 262,144 elements.
// Every array this project scans (<=~65k, see main.cu's scan_scaling
// stage) is comfortably inside that bound; the check below fails LOUDLY
// rather than silently truncating if a future caller ever exceeds it
// (THEORY.md "Numerical considerations" names the fix: a THIRD scan level).
// ---------------------------------------------------------------------------
void launch_scan_blelloch(int n, const int* d_in, int* d_out_exclusive)
{
    const int num_blocks = blocks_for(n, kScanElemsPerBlock);

    if (num_blocks > kScanElemsPerBlock) {
        std::fprintf(stderr,
            "launch_scan_blelloch: n=%d needs %d level-1 blocks, which exceeds this "
            "two-level scan's %d-block limit (see kernels.cu's scaling-limit comment) — "
            "a third scan level would be required.\n", n, num_blocks, kScanElemsPerBlock);
        std::exit(EXIT_FAILURE);
    }

    int* d_block_sums = nullptr;
    int* d_block_sums_scanned = nullptr;
    CUDA_CHECK(cudaMalloc(&d_block_sums, static_cast<size_t>(num_blocks) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_block_sums_scanned, static_cast<size_t>(num_blocks) * sizeof(int)));

    // Level 1: scan every 512-element span, in parallel, recording each
    // span's total into d_block_sums.
    blelloch_block_scan_kernel<<<num_blocks, kScanBlockThreads>>>(n, d_in, d_out_exclusive, d_block_sums);
    CUDA_CHECK_LAST_ERROR("blelloch_block_scan_kernel (level 1) launch");

    // Level 2: scan the (small) block-sums array in ONE block — this is
    // the SAME kernel, called with grid=1 and block_sums=nullptr (no
    // further level to feed). Valid because num_blocks <= kScanElemsPerBlock
    // is guaranteed by the check above.
    blelloch_block_scan_kernel<<<1, kScanBlockThreads>>>(num_blocks, d_block_sums, d_block_sums_scanned, nullptr);
    CUDA_CHECK_LAST_ERROR("blelloch_block_scan_kernel (level 2) launch");

    // Combine: broadcast level 2's per-block offsets back into level 1's
    // local answers. Always called (even for num_blocks==1, where the
    // single offset is provably 0 — an exclusive scan's first element is
    // always 0) rather than special-cased, for a uniform code path.
    add_block_offsets_kernel<<<num_blocks, kScanBlockThreads>>>(n, d_out_exclusive, d_block_sums_scanned);
    CUDA_CHECK_LAST_ERROR("add_block_offsets_kernel launch");

    CUDA_CHECK(cudaFree(d_block_sums));
    CUDA_CHECK(cudaFree(d_block_sums_scanned));
}

// ---------------------------------------------------------------------------
// launch_scan_thrust — the SAME exclusive scan via thrust::exclusive_scan.
//
// What this computes (CLAUDE.md rule 6): thrust::exclusive_scan(first,
// last, result, init) writes result[i] = init + sum(first[0..i)) for every
// i — precisely the hand-rolled scan above, one call. Internally, for a
// device-resident range of an arithmetic type with the default `+`
// operator, Thrust dispatches to CUB (CUDA UnBound, NVIDIA's header-only
// library of block/warp/device-level primitives that ships as part of the
// CUDA Toolkit's CCCL — the C++ Core Compute Libraries) — specifically
// cub::DeviceScan::ExclusiveSum, which implements a SINGLE-PASS
// "decoupled look-back" chained scan: unlike this file's two-level,
// three-kernel-launch approach, CUB's scan uses ONE kernel launch whose
// blocks communicate their running prefix through global memory flags,
// each block spinning briefly on its LEFT NEIGHBOR's flag rather than
// waiting for a whole separate reduce-then-broadcast pass. That is a
// meaningfully different (and, at scale, faster) algorithm from the one
// taught above — THEORY.md "Where this sits in the real world" names it
// explicitly as what production LiDAR pipelines actually call.
//
// thrust::device_ptr<> wraps a raw device pointer so Thrust's dispatch
// logic can SEE it is device memory (raw pointers are ambiguous to Thrust —
// they could be host or device); this wrapper adds no runtime cost, it is
// purely a compile-time tag.
// ---------------------------------------------------------------------------
void launch_scan_thrust(int n, const int* d_in, int* d_out_exclusive)
{
    thrust::device_ptr<const int> in_begin(d_in);
    thrust::device_ptr<int> out_begin(d_out_exclusive);
    thrust::exclusive_scan(in_begin, in_begin + n, out_begin, 0);
}

// ---------------------------------------------------------------------------
// compact_scatter_kernel — the ORDER-PRESERVING scatter that consumes a
// scan's addresses: one thread per INPUT point; kept points copy their xyz
// (and, optionally, their original index) to their scan-computed
// destination. See kernels.cuh's declaration for the "why this preserves
// order" argument (scan_exclusive is monotonic among kept elements).
// ---------------------------------------------------------------------------
__global__ void compact_scatter_kernel(int n, const float* __restrict__ xyz_in,
                                       const int* __restrict__ flags,
                                       const int* __restrict__ scan_exclusive,
                                       float* __restrict__ xyz_out,
                                       int* __restrict__ orig_idx_out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!flags[i]) return;   // dropped point: nothing to write, no destination slot claimed

    const int dst = scan_exclusive[i];   // this point's exclusive rank among kept points 0..i
    xyz_out[dst * 3 + 0] = xyz_in[i * 3 + 0];
    xyz_out[dst * 3 + 1] = xyz_in[i * 3 + 1];
    xyz_out[dst * 3 + 2] = xyz_in[i * 3 + 2];
    if (orig_idx_out != nullptr) orig_idx_out[dst] = i;
}

// ---------------------------------------------------------------------------
// compact_with_flags — internal orchestration shared by every named
// compaction pipeline below: scan the (already-computed) flags array, read
// back the total kept-count, then scatter. Kept `static` (internal
// linkage, no header declaration) because it is an implementation detail
// of THIS file, not part of the project's public kernel contract.
// ---------------------------------------------------------------------------
static int compact_with_flags(int n, const float* d_xyz, const int* d_flags,
                              float* d_out_xyz, int* d_out_orig_idx)
{
    int* d_scan = nullptr;
    CUDA_CHECK(cudaMalloc(&d_scan, static_cast<size_t>(n) * sizeof(int)));

    launch_scan_blelloch(n, d_flags, d_scan);   // THE primitive, doing all the real work

    // Total kept count = exclusive-scan-of-last-element + last flag itself
    // (the "one past the end" trick: scan[n-1] counts everyone strictly
    // before n-1; adding flags[n-1] accounts for element n-1 itself).
    int last_scan = 0, last_flag = 0;
    CUDA_CHECK(cudaMemcpy(&last_scan, d_scan + (n - 1), sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last_flag, d_flags + (n - 1), sizeof(int), cudaMemcpyDeviceToHost));
    const int count = last_scan + last_flag;

    const int blocks = blocks_for(n, kThreadsPerBlock);
    compact_scatter_kernel<<<blocks, kThreadsPerBlock>>>(n, d_xyz, d_flags, d_scan, d_out_xyz, d_out_orig_idx);
    CUDA_CHECK_LAST_ERROR("compact_scatter_kernel launch");

    CUDA_CHECK(cudaFree(d_scan));
    return count;
}

// ---------------------------------------------------------------------------
// The five named compaction pipelines — predicate kernel, then
// compact_with_flags (scan + scatter). Each allocates and frees its own
// [n]-sized flags scratch buffer; see compact_with_flags's comment for why
// per-call allocation is the right trade at this project's call count.
// ---------------------------------------------------------------------------
int launch_passthrough_compact(int n, const float* d_xyz, float* d_out_xyz, int* d_out_orig_idx)
{
    int* d_flags = nullptr;
    CUDA_CHECK(cudaMalloc(&d_flags, static_cast<size_t>(n) * sizeof(int)));
    const int blocks = blocks_for(n, kThreadsPerBlock);
    passthrough_predicate_kernel<<<blocks, kThreadsPerBlock>>>(n, d_xyz, d_flags);
    CUDA_CHECK_LAST_ERROR("passthrough_predicate_kernel launch");
    const int count = compact_with_flags(n, d_xyz, d_flags, d_out_xyz, d_out_orig_idx);
    CUDA_CHECK(cudaFree(d_flags));
    return count;
}

int launch_box_compact(int n, const float* d_xyz, float* d_out_xyz, int* d_out_orig_idx)
{
    int* d_flags = nullptr;
    CUDA_CHECK(cudaMalloc(&d_flags, static_cast<size_t>(n) * sizeof(int)));
    const int blocks = blocks_for(n, kThreadsPerBlock);
    box_predicate_kernel<<<blocks, kThreadsPerBlock>>>(n, d_xyz, d_flags);
    CUDA_CHECK_LAST_ERROR("box_predicate_kernel launch");
    const int count = compact_with_flags(n, d_xyz, d_flags, d_out_xyz, d_out_orig_idx);
    CUDA_CHECK(cudaFree(d_flags));
    return count;
}

int launch_frustum_compact(int n, const float* d_xyz, float* d_out_xyz, int* d_out_orig_idx)
{
    int* d_flags = nullptr;
    CUDA_CHECK(cudaMalloc(&d_flags, static_cast<size_t>(n) * sizeof(int)));
    const int blocks = blocks_for(n, kThreadsPerBlock);
    frustum_predicate_kernel<<<blocks, kThreadsPerBlock>>>(n, d_xyz, d_flags);
    CUDA_CHECK_LAST_ERROR("frustum_predicate_kernel launch");
    const int count = compact_with_flags(n, d_xyz, d_flags, d_out_xyz, d_out_orig_idx);
    CUDA_CHECK(cudaFree(d_flags));
    return count;
}

int launch_fused_compact(int n, const float* d_xyz, float* d_out_xyz, int* d_out_orig_idx)
{
    int* d_flags = nullptr;
    CUDA_CHECK(cudaMalloc(&d_flags, static_cast<size_t>(n) * sizeof(int)));
    const int blocks = blocks_for(n, kThreadsPerBlock);
    fused_predicate_kernel<<<blocks, kThreadsPerBlock>>>(n, d_xyz, d_flags);
    CUDA_CHECK_LAST_ERROR("fused_predicate_kernel launch");
    const int count = compact_with_flags(n, d_xyz, d_flags, d_out_xyz, d_out_orig_idx);
    CUDA_CHECK(cudaFree(d_flags));
    return count;
}

int launch_valid_compact(int n, const float* d_xyz, float* d_out_xyz, int* d_out_orig_idx)
{
    int* d_flags = nullptr;
    CUDA_CHECK(cudaMalloc(&d_flags, static_cast<size_t>(n) * sizeof(int)));
    const int blocks = blocks_for(n, kThreadsPerBlock);
    valid_predicate_kernel<<<blocks, kThreadsPerBlock>>>(n, d_xyz, d_flags);
    CUDA_CHECK_LAST_ERROR("valid_predicate_kernel launch");
    const int count = compact_with_flags(n, d_xyz, d_flags, d_out_xyz, d_out_orig_idx);
    CUDA_CHECK(cudaFree(d_flags));
    return count;
}

// ===========================================================================
// Unorganized -> organized: the OPPOSITE direction. Every point SCATTERS
// (rather than being gathered by a predicate) into a computed cell, with
// collisions resolved by a nearest-wins atomicMin race on a 64-bit encoded
// (range, index) key — see kernels.cuh's extended discussion of 01.18's
// 32-bit precedent.
// ===========================================================================

// ---------------------------------------------------------------------------
// scatter_to_organized_kernel — one thread per INPUT point: compute its
// (ring, azimuth) cell, encode (range, this point's index), and atomicMin
// it into cell_encoded[cell]. `point_cell_out[i]` records the computed cell
// for point i regardless of whether it wins — main.cu's GATE
// collision_accounting cross-checks this against the winner recorded by
// finalize_organized_kernel below, a genuine two-different-traversals
// reconciliation (not a tautology): point-space ("which cell did I aim
// at?") versus cell-space ("who actually won me?").
//
// atomicMin on `unsigned long long int` requires compute capability >= 5.0
// (Maxwell) — this repo's floor is sm_75 (Turing, CC 7.5), so no fallback
// path is needed; noted here because it is exactly the kind of
// hardware-floor fact CLAUDE.md's GPU-mapping commentary expects named.
// ---------------------------------------------------------------------------
__global__ void scatter_to_organized_kernel(int n_points, const float* __restrict__ xyz,
                                            unsigned long long* __restrict__ cell_encoded)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) return;

    const float x = xyz[i * 3 + 0];
    const float y = xyz[i * 3 + 1];
    const float z = xyz[i * 3 + 2];

    const int ring = d_nearest_ring_of(x, y, z);
    const int az   = d_azimuth_bin_of(x, y);
    const int cell = ring * kAzimuthBins + az;

    const float range = sqrtf(x * x + y * y + z * z);
    const unsigned long long key = d_pack_range_index(range, static_cast<uint32_t>(i));

    // The race: many threads (points sharing a cell) may call this
    // concurrently; atomicMin guarantees the smallest key — smallest
    // range, ties broken by smallest index — is what survives, regardless
    // of execution order (THEORY.md "Numerical considerations" argues this
    // is why the result is DETERMINISTIC despite being a race).
    atomicMin(&cell_encoded[cell], key);
}

// ---------------------------------------------------------------------------
// finalize_organized_kernel — one thread per CELL: decode the winning
// point (if any) and materialize its xyz into the organized output; an
// empty cell gets the NaN sentinel (is_invalid_point's convention), so the
// output is itself a valid organized grid a downstream consumer could load
// unmodified (the roundtrip gate in main.cu exercises exactly this).
// nanf("") is CUDA's device-callable "construct a quiet NaN" — used instead
// of a raw hex bit-pattern cast for readability; both produce an IEEE-754
// qNaN and either satisfies is_invalid_point()'s x!=x test.
// ---------------------------------------------------------------------------
__global__ void finalize_organized_kernel(int num_cells,
                                          const unsigned long long* __restrict__ cell_encoded,
                                          const float* __restrict__ xyz_source,
                                          float* __restrict__ organized_xyz_out,
                                          int* __restrict__ winner_index_out)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= num_cells) return;

    const unsigned long long enc = cell_encoded[c];
    if (enc == kEmptyCellEncoded) {
        organized_xyz_out[c * 3 + 0] = nanf("");
        organized_xyz_out[c * 3 + 1] = nanf("");
        organized_xyz_out[c * 3 + 2] = nanf("");
        winner_index_out[c] = -1;
        return;
    }

    const uint32_t idx = static_cast<uint32_t>(enc & 0xFFFFFFFFull);
    organized_xyz_out[c * 3 + 0] = xyz_source[idx * 3 + 0];
    organized_xyz_out[c * 3 + 1] = xyz_source[idx * 3 + 1];
    organized_xyz_out[c * 3 + 2] = xyz_source[idx * 3 + 2];
    winner_index_out[c] = static_cast<int>(idx);
}

// ---------------------------------------------------------------------------
// scatter_to_organized_kernel also needs, per point, the CELL it targeted
// (for the collision_accounting reconciliation) — rather than recomputing
// ring/azimuth a second time on the host (a second source of potential
// drift), the kernel above is given an EXTRA output array by this small
// wrapper kernel's caller. To avoid changing scatter_to_organized_kernel's
// signature (and kernels.cuh's declaration) for a value only the launcher
// needs internally, launch_scatter_to_organized recomputes point_cell on
// the HOST from the already-copied-back winner array plus a second small
// device pass — see the launcher below for the exact sequencing.
// ---------------------------------------------------------------------------
__global__ void record_point_cell_kernel(int n_points, const float* __restrict__ xyz,
                                         int* __restrict__ point_cell_out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) return;
    const float x = xyz[i * 3 + 0], y = xyz[i * 3 + 1], z = xyz[i * 3 + 2];
    point_cell_out[i] = d_nearest_ring_of(x, y, z) * kAzimuthBins + d_azimuth_bin_of(x, y);
}

// ---------------------------------------------------------------------------
// launch_scatter_to_organized — the unorganized->organized pipeline end to
// end. See kernels.cuh's declaration for the returned struct's meaning.
// ---------------------------------------------------------------------------
OrganizedScatterResult launch_scatter_to_organized(int n_points, const float* d_xyz,
                                                   float* d_organized_xyz_out,
                                                   int* d_winner_index_out)
{
    unsigned long long* d_encoded = nullptr;
    int* d_point_cell = nullptr;
    CUDA_CHECK(cudaMalloc(&d_encoded, static_cast<size_t>(kOrganizedCells) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_point_cell, static_cast<size_t>(n_points) * sizeof(int)));

    // 0xFF byte-fill = all bits set = kEmptyCellEncoded (UINT64_MAX) — a
    // single memset, no kernel, because the sentinel is bit-pattern-trivial.
    CUDA_CHECK(cudaMemset(d_encoded, 0xFF, static_cast<size_t>(kOrganizedCells) * sizeof(unsigned long long)));

    const int blocks_pts = blocks_for(n_points, kThreadsPerBlock);
    scatter_to_organized_kernel<<<blocks_pts, kThreadsPerBlock>>>(n_points, d_xyz, d_encoded);
    CUDA_CHECK_LAST_ERROR("scatter_to_organized_kernel launch");

    record_point_cell_kernel<<<blocks_pts, kThreadsPerBlock>>>(n_points, d_xyz, d_point_cell);
    CUDA_CHECK_LAST_ERROR("record_point_cell_kernel launch");

    const int blocks_cells = blocks_for(kOrganizedCells, kThreadsPerBlock);
    finalize_organized_kernel<<<blocks_cells, kThreadsPerBlock>>>(
        kOrganizedCells, d_encoded, d_xyz, d_organized_xyz_out, d_winner_index_out);
    CUDA_CHECK_LAST_ERROR("finalize_organized_kernel launch");

    // Both reconciliation inputs are small (kOrganizedCells=16384 ints and
    // n_points typically <15k) — a plain host copy-back and loop is the
    // clearest way to compute the two independent counts.
    std::vector<int> h_winner(static_cast<size_t>(kOrganizedCells));
    std::vector<int> h_point_cell(static_cast<size_t>(n_points));
    CUDA_CHECK(cudaMemcpy(h_winner.data(), d_winner_index_out,
                          h_winner.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_point_cell.data(), d_point_cell,
                          h_point_cell.size() * sizeof(int), cudaMemcpyDeviceToHost));

    int occupied = 0;
    for (int c = 0; c < kOrganizedCells; ++c) if (h_winner[c] != -1) ++occupied;

    int collisions = 0;
    for (int i = 0; i < n_points; ++i) {
        if (h_winner[h_point_cell[i]] != i) ++collisions;   // this point aimed at a cell it did not win
    }

    CUDA_CHECK(cudaFree(d_encoded));
    CUDA_CHECK(cudaFree(d_point_cell));

    return OrganizedScatterResult{ occupied, collisions };
}
