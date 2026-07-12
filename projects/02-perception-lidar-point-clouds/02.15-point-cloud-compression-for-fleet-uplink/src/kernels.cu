// ===========================================================================
// kernels.cu — GPU kernels for project 02.15
//              Point cloud compression (octree/entropy) for fleet uplink
//
// File map (read top to bottom — each section is a stage from kernels.cuh's
// file-header essay):
//   A. Device transcriptions of kernels.cuh's shared host-only formulas.
//   B. STAGE 1a — Morton codes + Thrust sort.
//   C. THE SCAN CHAPTER — 02.02's two-level Blelloch scan, copied and cited
//      verbatim, plus the Thrust alternative used where the hand-rolled
//      scan's size bound would bite.
//   D. STAGE 1b — per-level octree construction (flags -> scan -> occupancy).
//   E. STAGE 2 — histogram, canonical Huffman table build, GPU encode
//      (length map -> scan -> bit-scatter).
//
// Read this after: main.cu, kernels.cuh.  Read this before: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <queue>
#include <algorithm>

// Thrust: header-only part of the CUDA Toolkit (no separate .lib — see
// build/*.vcxproj's Thrust comment, copied from 02.02's ratified precedent).
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

// ===========================================================================
// A. DEVICE TRANSCRIPTIONS of kernels.cuh's shared host-only inline
// functions. kernels.cuh's file header explains WHY these exist as
// duplicates (a plain inline host function cannot be called from a
// __global__ kernel) — each is a byte-for-byte transcription of its
// kernels.cuh counterpart, kept in lockstep BY HAND, with the drift risk
// covered by main.cu's VERIFY(morton)/VERIFY(octree_levels) gates (a GPU
// answer computed via these functions, compared against a CPU answer
// computed via kernels.cuh's shared host functions).
// ===========================================================================

__device__ __forceinline__ uint32_t d_quantize_axis(float p, float lo, float extent_m, int D)
{
    float t = (extent_m > 0.0f) ? (p - lo) / extent_m : 0.0f;
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    const uint32_t max_cell = (1u << D) - 1u;
    uint32_t cell = static_cast<uint32_t>(t * static_cast<float>(1u << D));
    if (cell > max_cell) cell = max_cell;
    return cell;
}

__device__ __forceinline__ uint64_t d_morton_encode(uint32_t ix, uint32_t iy, uint32_t iz, int D)
{
    uint64_t code = 0;
    for (int b = 0; b < D; ++b) {
        code |= (static_cast<uint64_t>((ix >> b) & 1u)) << (3 * b + 0);
        code |= (static_cast<uint64_t>((iy >> b) & 1u)) << (3 * b + 1);
        code |= (static_cast<uint64_t>((iz >> b) & 1u)) << (3 * b + 2);
    }
    return code;
}

__device__ __forceinline__ uint64_t d_point_to_code(float x, float y, float z,
                                                     SceneAABB aabb, int D)
{
    const uint32_t ix = d_quantize_axis(x, aabb.min[0], aabb.extent_m, D);
    const uint32_t iy = d_quantize_axis(y, aabb.min[1], aabb.extent_m, D);
    const uint32_t iz = d_quantize_axis(z, aabb.min[2], aabb.extent_m, D);
    return d_morton_encode(ix, iy, iz, D);
}

// ===========================================================================
// B. STAGE 1a — Morton codes + sort.
// ===========================================================================

// compute_codes_kernel — one thread per point, pure MAP (the simplest GPU
// pattern this repo teaches — see the SAXPY placeholder every project
// scaffolds from): out_codes[i] depends only on xyz[i]. Grid-stride is
// unnecessary at this project's scale (<= a few hundred thousand points,
// comfortably under one launch's max grid), so this uses the plain "one
// thread, one point, guard the tail" form.
__global__ void compute_codes_kernel(int n, const float* __restrict__ xyz,
                                     SceneAABB aabb, int D,
                                     uint64_t* __restrict__ out_codes)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out_codes[i] = d_point_to_code(xyz[i * 3 + 0], xyz[i * 3 + 1], xyz[i * 3 + 2], aabb, D);
}

void launch_compute_codes(int n, const float* d_xyz, const SceneAABB& aabb, int D,
                          uint64_t* d_codes)
{
    const int blocks = blocks_for(n, kThreadsPerBlock);
    compute_codes_kernel<<<blocks, kThreadsPerBlock>>>(n, d_xyz, aabb, D, d_codes);
    CUDA_CHECK_LAST_ERROR("compute_codes_kernel launch");
}

// launch_sort_codes — Stage 1's sort: thrust::sort ascending over the raw
// 64-bit codes, IN PLACE. What this computes (CLAUDE.md §6 rule 6):
// thrust::sort dispatches (for a device range of an arithmetic type with
// the default '<') to CUB's cub::DeviceRadixSort — an 8 passes-of-8-bits
// LSD radix sort for a 64-bit key, O(n) work, memory-bandwidth bound. No
// paired "value" array is uploaded because this codec never needs to
// recover which ORIGINAL point produced a given code (unlike 02.05's
// augmented-key LBVH, which must — see kernels.cuh's launch_sort_codes
// declaration for why an unstable sort over possibly-EQUAL keys is still
// correct here: multiple points sharing one leaf cell is routine and the
// octree only ever needs the code MULTISET, not point identity).
void launch_sort_codes(int n, uint64_t* d_codes_inout)
{
    thrust::device_ptr<uint64_t> begin(d_codes_inout);
    thrust::sort(begin, begin + n);
}

// ===========================================================================
// C. THE SCAN CHAPTER — 02.02's two-level Blelloch exclusive scan, copied
// verbatim (cited): see 02.02's kernels.cu for the full up-sweep/down-sweep
// derivation and the bank-conflict honesty note (not re-derived here — this
// project's own teaching point is the COMPOSITION: the identical scan
// primitive answers two different questions in this file, "which node do I
// belong to" in section D and "which bit do I start at" in section E).
// ===========================================================================

__global__ void blelloch_block_scan_kernel(int n, const int* __restrict__ in,
                                           int* __restrict__ out_exclusive,
                                           int* __restrict__ block_sums)
{
    __shared__ int temp[kScanElemsPerBlock];

    const int tid = threadIdx.x;
    const int block_offset = blockIdx.x * kScanElemsPerBlock;

    const int ai = tid;
    const int bi = tid + kScanBlockThreads;
    const int global_ai = block_offset + ai;
    const int global_bi = block_offset + bi;

    temp[ai] = (global_ai < n) ? in[global_ai] : 0;
    temp[bi] = (global_bi < n) ? in[global_bi] : 0;

    int offset = 1;
    for (int d = kScanElemsPerBlock >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            const int idx_a = offset * (2 * tid + 1) - 1;
            const int idx_b = offset * (2 * tid + 2) - 1;
            temp[idx_b] += temp[idx_a];
        }
        offset *= 2;
    }

    if (tid == 0) {
        if (block_sums != nullptr) block_sums[blockIdx.x] = temp[kScanElemsPerBlock - 1];
        temp[kScanElemsPerBlock - 1] = 0;
    }

    for (int d = 1; d < kScanElemsPerBlock; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            const int idx_a = offset * (2 * tid + 1) - 1;
            const int idx_b = offset * (2 * tid + 2) - 1;
            const int t = temp[idx_a];
            temp[idx_a] = temp[idx_b];
            temp[idx_b] += t;
        }
    }
    __syncthreads();

    if (global_ai < n) out_exclusive[global_ai] = temp[ai];
    if (global_bi < n) out_exclusive[global_bi] = temp[bi];
}

__global__ void add_block_offsets_kernel(int n, int* __restrict__ out_exclusive,
                                         const int* __restrict__ scanned_block_sums)
{
    const int tid = threadIdx.x;
    const int block_offset = blockIdx.x * kScanElemsPerBlock;
    const int global_ai = block_offset + tid;
    const int global_bi = block_offset + tid + kScanBlockThreads;
    const int off = scanned_block_sums[blockIdx.x];

    if (global_ai < n) out_exclusive[global_ai] += off;
    if (global_bi < n) out_exclusive[global_bi] += off;
}

// launch_scan_blelloch — see kernels.cuh's declaration for the 262,144-
// element bound this project relies on (both committed clouds are 200,000
// points each — a documented, checked margin, not an assumption: the guard
// below fails loudly rather than silently truncating, same as 02.02).
void launch_scan_blelloch(int n, const int* d_in, int* d_out_exclusive)
{
    const int num_blocks = blocks_for(n, kScanElemsPerBlock);

    if (num_blocks > kScanElemsPerBlock) {
        std::fprintf(stderr,
            "launch_scan_blelloch: n=%d needs %d level-1 blocks, which exceeds this "
            "two-level scan's %d-block limit (kernels.cuh's kScanElemsPerBlock comment) — "
            "a third scan level (or launch_scan_thrust) would be required.\n",
            n, num_blocks, kScanElemsPerBlock);
        std::exit(EXIT_FAILURE);
    }

    int* d_block_sums = nullptr;
    int* d_block_sums_scanned = nullptr;
    CUDA_CHECK(cudaMalloc(&d_block_sums, static_cast<size_t>(num_blocks) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_block_sums_scanned, static_cast<size_t>(num_blocks) * sizeof(int)));

    blelloch_block_scan_kernel<<<num_blocks, kScanBlockThreads>>>(n, d_in, d_out_exclusive, d_block_sums);
    CUDA_CHECK_LAST_ERROR("blelloch_block_scan_kernel (level 1) launch");

    blelloch_block_scan_kernel<<<1, kScanBlockThreads>>>(num_blocks, d_block_sums, d_block_sums_scanned, nullptr);
    CUDA_CHECK_LAST_ERROR("blelloch_block_scan_kernel (level 2) launch");

    add_block_offsets_kernel<<<num_blocks, kScanBlockThreads>>>(n, d_out_exclusive, d_block_sums_scanned);
    CUDA_CHECK_LAST_ERROR("add_block_offsets_kernel launch");

    CUDA_CHECK(cudaFree(d_block_sums));
    CUDA_CHECK(cudaFree(d_block_sums_scanned));
}

// launch_scan_thrust — thrust::exclusive_scan (dispatches to
// cub::DeviceScan::ExclusiveSum's single-pass decoupled-look-back chained
// scan — see 02.02's identical launcher for the full explanation of why
// that is algorithmically different from, and at scale faster than, the
// hand-rolled two-level version above). Used for STAGE 2's code-length scan
// below, whose input size M (total octree node count across all levels) is
// DATA-DEPENDENT and can exceed 262,144 on the pathological cohort at fine
// depths (a uniformly scattered cloud produces close to one node PER LEVEL
// PER POINT in the worst case — THEORY.md "Numerical considerations" works
// this bound) — exactly the situation launch_scan_blelloch's guard above is
// designed to refuse rather than silently mishandle.
void launch_scan_thrust(int n, const int* d_in, int* d_out_exclusive)
{
    thrust::device_ptr<const int> in_begin(d_in);
    thrust::device_ptr<int> out_begin(d_out_exclusive);
    thrust::exclusive_scan(in_begin, in_begin + n, out_begin, 0);
}

// ===========================================================================
// D. STAGE 1b — per-level octree construction: THE OCTREE AS A STRING
// PROBLEM, worked in three kernels (see kernels.cuh's file-header essay for
// the full derivation).
// ===========================================================================

// compute_level_flags_kernel — see kernels.cuh's declaration. Each thread
// reads its own and its LEFT NEIGHBOR's sorted code (a coalesced, if
// slightly overlapping, read pattern — every code is read by exactly 2
// threads, itself and its right neighbor, which is negligible bandwidth
// next to the codes array's total size).
__global__ void compute_level_flags_kernel(int n, const uint64_t* __restrict__ sorted_codes,
                                           int D, int level,
                                           int* __restrict__ out_is_start)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (i == 0) { out_is_start[0] = 1; return; }   // the first point always starts the first node
    const int shift = 3 * (D - level);
    const uint64_t prev_prefix = sorted_codes[i - 1] >> shift;
    const uint64_t cur_prefix  = sorted_codes[i]     >> shift;
    out_is_start[i] = (cur_prefix != prev_prefix) ? 1 : 0;
}

// compute_occupancy_kernel — see kernels.cuh's declaration (including why
// out_occupancy is uint32_t, not uint8_t: no native byte-granular
// atomicOr, and the scan_exclusive[i] + is_start[i] - 1 node-index
// correction — read that comment first, it explains a real bug this
// project's own VERIFY(octree_levels) gate caught during development).
// child_octant (which of the owning node's 8 children this point falls
// under, one level DEEPER) is all a thread additionally needs: "OR my bit
// into MY node's word" — the same "coordinate parallel threads through a
// shared structure via one atomic primitive" idiom 02.01's
// hash_insert_kernel and 02.05's propagate_aabb_kernel use, here at its
// simplest (one bit, one OR, no read-modify-write race on the VALUE beyond
// what atomicOr itself resolves).
__global__ void compute_occupancy_kernel(int n, const uint64_t* __restrict__ sorted_codes,
                                         int D, int level,
                                         const int* __restrict__ is_start,
                                         const int* __restrict__ scan_exclusive,
                                         uint32_t* __restrict__ out_occupancy)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int node_id = scan_exclusive[i] + is_start[i] - 1;   // the inclusive-count-minus-one correction
    const int shift2 = 3 * (D - level - 1);
    const uint32_t oct = static_cast<uint32_t>((sorted_codes[i] >> shift2) & 7ull);
    atomicOr(&out_occupancy[node_id], 1u << oct);
}

// launch_build_octree_level — see kernels.cuh's declaration. The three
// kernels above, chained: flags (a map) -> exclusive scan (this level's
// point-to-node LABELING, not a compaction — the same primitive, a
// different use) -> occupancy (an atomic scatter-reduce). The node count is
// read back with the identical "exclusive-scan-of-last + flag-of-last"
// trick 02.02's compact_with_flags uses for a kept-element COUNT — here it
// counts NODES instead of survivors, the same arithmetic idea repurposed.
OctreeLevelResult launch_build_octree_level(int n, const uint64_t* d_sorted_codes,
                                            int D, int level,
                                            int* d_is_start_scratch, int* d_node_id_scratch,
                                            uint32_t* d_occupancy_out)
{
    const int blocks = blocks_for(n, kThreadsPerBlock);

    compute_level_flags_kernel<<<blocks, kThreadsPerBlock>>>(
        n, d_sorted_codes, D, level, d_is_start_scratch);
    CUDA_CHECK_LAST_ERROR("compute_level_flags_kernel launch");

    launch_scan_blelloch(n, d_is_start_scratch, d_node_id_scratch);

    int last_scan = 0, last_flag = 0;
    CUDA_CHECK(cudaMemcpy(&last_scan, d_node_id_scratch + (n - 1), sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last_flag, d_is_start_scratch + (n - 1), sizeof(int), cudaMemcpyDeviceToHost));
    const int num_nodes = last_scan + last_flag;

    CUDA_CHECK(cudaMemset(d_occupancy_out, 0, static_cast<size_t>(num_nodes) * sizeof(uint32_t)));
    compute_occupancy_kernel<<<blocks, kThreadsPerBlock>>>(
        n, d_sorted_codes, D, level, d_is_start_scratch, d_node_id_scratch, d_occupancy_out);
    CUDA_CHECK_LAST_ERROR("compute_occupancy_kernel launch");

    return OctreeLevelResult{ num_nodes };
}

// ===========================================================================
// E. STAGE 2 — histogram, canonical Huffman table, GPU encode.
// ===========================================================================

// compute_histogram_kernel — one thread per SYMBOL, atomicAdd into its bin.
// 256 bins is small enough that shared-memory-privatized histogramming
// (the classic "bin per warp/block in shared memory, merge at the end"
// optimization for reducing global-memory atomic contention) would be the
// production move at large M — THEORY.md "The GPU mapping" names it; this
// teaching version goes straight to global-memory atomics because M tops
// out in the hundreds of thousands here, not the billions where contention
// actually dominates the runtime.
__global__ void compute_histogram_kernel(int m, const uint8_t* __restrict__ symbols,
                                         int* __restrict__ out_hist)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    atomicAdd(&out_hist[symbols[i]], 1);
}

void launch_compute_histogram(int m, const uint8_t* d_symbols, int* d_hist_out)
{
    CUDA_CHECK(cudaMemset(d_hist_out, 0, 256 * sizeof(int)));
    const int blocks = blocks_for(m, kThreadsPerBlock);
    compute_histogram_kernel<<<blocks, kThreadsPerBlock>>>(m, d_symbols, d_hist_out);
    CUDA_CHECK_LAST_ERROR("compute_histogram_kernel launch");
}

// ---------------------------------------------------------------------------
// build_huffman_table — the OPERATIONAL canonical-Huffman table builder.
// Host-side, heap-based (std::priority_queue). See kernels.cuh's
// huffman_merge_key comment for the shared deterministic tie-break every
// independent builder (this one and reference_cpu.cpp's) must follow for
// their outputs to provably agree.
//
// Algorithm (three passes over <=256 symbols — utterly dominated by the GPU
// kernels this file also defines, no GPU pattern to teach here):
//   1) Seed the heap with one leaf per OBSERVED symbol (hist[s] > 0), leaf
//      id = s (< 256), so a leaf always out-ranks an internal node of equal
//      frequency in huffman_merge_key's ordering.
//   2) Repeatedly pop the two smallest (by huffman_merge_key), merge into a
//      new internal node (id = 256 + creation_index, freq = sum), record
//      parent[] for both children, push the merged node back — classic
//      Huffman construction, O(k log k) for k observed symbols.
//   3) Walk parent[] from every leaf to the root to get its code LENGTH
//      (tree depth), then CANONICALIZE: sort observed symbols by
//      (length, symbol) ascending and assign codes in that order — first
//      code 0, each next code = (prev_code + 1) << (len_i - len_{i-1}) —
//      the standard canonical-Huffman assignment (DEFLATE/RFC 1951 uses the
//      identical rule). This is the step that makes the FINAL BIT PATTERNS
//      deterministic given only the code-length multiset, independent of
//      which tree-construction order produced them.
// ---------------------------------------------------------------------------
void build_huffman_table(const int hist[256], HuffmanTable& out)
{
    for (int s = 0; s < 256; ++s) { out.len[s] = 0; out.bits[s] = 0; }

    // parent/freq indexed by node id: leaves 0..255, internal nodes 256..510
    // (at most 255 merges for at most 256 leaves).
    std::vector<int> parent(511, -1);
    std::vector<long long> freq(511, 0);

    // Min-heap ordered by huffman_merge_key ascending — std::priority_queue
    // is a MAX-heap by default, so the comparator below inverts '<'.
    using KV = std::pair<uint64_t, int>;   // (merge_key, node_id)
    std::priority_queue<KV, std::vector<KV>, std::greater<KV>> heap;

    int num_observed = 0;
    int only_symbol = -1;
    for (int s = 0; s < 256; ++s) {
        if (hist[s] > 0) {
            freq[s] = hist[s];
            heap.push({ huffman_merge_key(hist[s], s), s });
            ++num_observed;
            only_symbol = s;
        }
    }

    if (num_observed == 0) return;   // empty stream: nothing to encode (defensive; never hit at n>0)

    if (num_observed == 1) {
        // Degenerate: a single distinct symbol carries zero Shannon
        // information (H=0), but a real bitstream cannot emit a 0-bit code
        // — the conventional fix (also DEFLATE's) is to force a 1-bit code.
        // THEORY.md "Numerical considerations" documents this edge case
        // honestly; it does not occur in this project's committed data
        // (every depth's occupancy stream uses far more than one pattern).
        out.len[only_symbol] = 1;
        out.bits[only_symbol] = 0;
        return;
    }

    int next_internal_id = 256;
    int root_id = -1;
    while (heap.size() > 1) {
        const KV a = heap.top(); heap.pop();
        const KV b = heap.top(); heap.pop();
        const int id = next_internal_id++;
        const long long f = freq[a.second] + freq[b.second];
        freq[id] = f;
        parent[a.second] = id;
        parent[b.second] = id;
        heap.push({ huffman_merge_key(static_cast<int>(f), id), id });
        root_id = id;
    }

    // Code length = depth from leaf to root (count parent-hops).
    std::vector<std::pair<int, int>> by_len_then_symbol;   // (length, symbol)
    for (int s = 0; s < 256; ++s) {
        if (hist[s] <= 0) continue;
        int depth = 0;
        int cur = s;
        while (cur != root_id) { cur = parent[cur]; ++depth; }
        by_len_then_symbol.push_back({ depth, s });
    }
    std::sort(by_len_then_symbol.begin(), by_len_then_symbol.end());

    // Canonical code assignment (see the function header's step 3).
    uint32_t code = 0;
    int prev_len = 0;
    for (size_t k = 0; k < by_len_then_symbol.size(); ++k) {
        const int len = by_len_then_symbol[k].first;
        const int sym = by_len_then_symbol[k].second;
        if (k > 0) code = (code + 1u) << (len - prev_len);
        out.len[sym] = len;
        out.bits[sym] = code;
        prev_len = len;
    }
}

// compute_code_lengths_kernel — see kernels.cuh's declaration. Reads the
// __constant__ table uploaded by launch_huffman_encode below — a small
// (256*4 bytes), read-only, broadcast-friendly lookup every thread reads
// identically, the textbook use case for __constant__ memory (02.02's
// d_kBoxMin/d_kTCameraLidar precedent, cited).
__constant__ int d_huff_len[256];

// d_huff_bits — the companion table (the actual code bits, right-justified
// in the low d_huff_len[s] bits — kernels.cuh's HuffmanTable convention).
// Declared here, ABOVE both kernels that read it, so bit_scatter_kernel
// below sees an ordinary in-TU __constant__ definition rather than an
// `extern` forward reference (CUDA constant memory is simplest to reason
// about — and to keep the linker happy — when every reader sees the same
// single definition, not a declare-then-define split across the file).
__constant__ uint32_t d_huff_bits[256];

__global__ void compute_code_lengths_kernel(int m, const uint8_t* __restrict__ symbols,
                                            int* __restrict__ out_len)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    out_len[i] = d_huff_len[symbols[i]];
}

// bit_scatter_kernel — see kernels.cuh's declaration. Per symbol, per bit:
// atomicOr one bit into its aligned 32-bit output word. WHY THIS RACE IS
// SAFE: bit_offset[] comes from an EXCLUSIVE scan of code LENGTHS, so every
// symbol's bit range [bit_offset[i], bit_offset[i]+len) is DISJOINT from
// every other symbol's range — two threads may target the SAME output
// uint32_t word (when a code crosses a 32-bit boundary, or several short
// codes share one word), but NEVER the same BIT within it, and atomicOr on
// disjoint bits of one word commutes regardless of execution order (the
// same "disjoint sub-word writes via a full-word atomic" argument that
// motivates kernels.cuh's uint32_t occupancy choice in section D above,
// reused here at BIT granularity instead of BYTE granularity).
//
// out_packed is addressed as an array of uint32_t words (out_packed_words);
// launch_huffman_encode sizes it in words, not bytes, and main.cu narrows
// the final result to a byte vector on the host once decoding is the only
// remaining consumer (bit position 0 of the STREAM is the MSB of word 0 —
// see the shift arithmetic below).
__global__ void bit_scatter_kernel(int m, const uint8_t* __restrict__ symbols,
                                   const int* __restrict__ bit_offset,
                                   uint32_t* __restrict__ out_packed_words)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    const uint8_t sym = symbols[i];
    const int len = d_huff_len[sym];
    if (len == 0) return;   // defensive: every symbol here was measured, so len>0 always in practice

    const uint32_t code = d_huff_bits[sym];

    int base = bit_offset[i];
    for (int j = 0; j < len; ++j) {
        // MSB-first within the symbol's code (bit len-1 is written first —
        // the documented convention kernels.cuh's HuffmanTable comment and
        // reference_cpu.cpp's huffman_encode_cpu both independently follow).
        const uint32_t bit_val = (code >> (len - 1 - j)) & 1u;
        if (bit_val == 0) continue;   // 0 bits need no write: the buffer starts zeroed
        const int g = base + j;                       // global bit index in the whole stream
        const int word_idx = g >> 5;                   // which uint32_t word (32 bits/word)
        const int bit_in_word = 31 - (g & 31);          // MSB-first within the word too
        atomicOr(&out_packed_words[word_idx], 1u << bit_in_word);
    }
}

// launch_huffman_encode — see kernels.cuh's declaration. Orchestrates the
// whole STAGE-2 encode: upload table -> length map -> Thrust exclusive scan
// (see section C for why Thrust, not the hand-rolled scan, is used HERE —
// m is data-dependent and can exceed the two-level scan's bound) -> allocate
// exactly enough packed words -> bit-scatter -> copy back.
EncodeResult launch_huffman_encode(int m, const uint8_t* d_symbols, const HuffmanTable& table)
{
    CUDA_CHECK(cudaMemcpyToSymbol(d_huff_len, table.len, 256 * sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_huff_bits, table.bits, 256 * sizeof(uint32_t)));

    int* d_len = nullptr;
    int* d_bit_offset = nullptr;
    CUDA_CHECK(cudaMalloc(&d_len, static_cast<size_t>(m) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bit_offset, static_cast<size_t>(m) * sizeof(int)));

    const int blocks = blocks_for(m, kThreadsPerBlock);
    compute_code_lengths_kernel<<<blocks, kThreadsPerBlock>>>(m, d_symbols, d_len);
    CUDA_CHECK_LAST_ERROR("compute_code_lengths_kernel launch");

    launch_scan_thrust(m, d_len, d_bit_offset);   // exclusive scan: bit_offset[i] = sum(len[0..i))

    int last_offset = 0, last_len = 0;
    CUDA_CHECK(cudaMemcpy(&last_offset, d_bit_offset + (m - 1), sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last_len, d_len + (m - 1), sizeof(int), cudaMemcpyDeviceToHost));
    const long long total_bits = static_cast<long long>(last_offset) + static_cast<long long>(last_len);
    const long long total_words = (total_bits + 31) / 32 + 1;   // +1 word of pure defensive headroom

    uint32_t* d_packed_words = nullptr;
    CUDA_CHECK(cudaMalloc(&d_packed_words, static_cast<size_t>(total_words) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_packed_words, 0, static_cast<size_t>(total_words) * sizeof(uint32_t)));

    bit_scatter_kernel<<<blocks, kThreadsPerBlock>>>(m, d_symbols, d_bit_offset, d_packed_words);
    CUDA_CHECK_LAST_ERROR("bit_scatter_kernel launch");

    std::vector<uint32_t> h_words(static_cast<size_t>(total_words));
    CUDA_CHECK(cudaMemcpy(h_words.data(), d_packed_words,
                          h_words.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_len));
    CUDA_CHECK(cudaFree(d_bit_offset));
    CUDA_CHECK(cudaFree(d_packed_words));

    // Narrow words -> bytes on the host (MSB-first within each word, so
    // byte 0 of word 0 is bits 31..24, matching a plain byte-stream reader).
    EncodeResult result;
    result.num_bits = total_bits;
    const long long total_bytes = (total_bits + 7) / 8;
    result.packed.resize(static_cast<size_t>(total_bytes));
    for (long long b = 0; b < total_bytes; ++b) {
        const uint32_t word = h_words[static_cast<size_t>(b / 4)];
        const int byte_in_word = static_cast<int>(b % 4);        // 0 = most-significant byte of the word
        const int shift = 24 - 8 * byte_in_word;
        result.packed[static_cast<size_t>(b)] = static_cast<uint8_t>((word >> shift) & 0xFFu);
    }
    return result;
}
