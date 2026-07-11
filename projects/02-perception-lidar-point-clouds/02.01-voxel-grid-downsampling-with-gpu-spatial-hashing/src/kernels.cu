// ===========================================================================
// kernels.cu — GPU kernels for project 02.01 (Voxel-grid downsampling with
//              GPU spatial hashing): Method A (atomic hash table) and
//              Method B (Thrust sort + fixed-order segmented reduction).
//
// Role in the project
// -------------------
// All __global__ (GPU) code lives here, plus the small host-side launch
// wrappers that own the grid/block math (the launch-configuration reasoning
// sits beside the code it configures, CLAUDE.md §6.1 rule 2).
//
// Read this after: main.cu, kernels.cuh.  Read this before: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK / CUDA_CHECK_LAST_ERROR

// Thrust: header-only pieces of the CUDA Toolkit (CLAUDE.md §5 — allowed
// without a separate .lib; see build/*.vcxproj's comment on why no extra
// AdditionalDependencies entry is needed for it). Method B uses exactly
// three Thrust algorithms; each is explained at its call site below
// (CLAUDE.md §6.1 rule 6: what it computes, why the library instead of
// hand-rolling, and the shape of its inputs/outputs).
#include <thrust/device_ptr.h>            // wraps our raw cudaMalloc'd pointers for Thrust's algorithms
#include <thrust/sort.h>                  // thrust::stable_sort_by_key
#include <thrust/reduce.h>                // thrust::reduce
#include <thrust/copy.h>                  // thrust::copy_if
#include <thrust/sequence.h>              // thrust::sequence
#include <thrust/iterator/counting_iterator.h>

// is_nonzero — the copy_if predicate launch_sort_based_downsample uses to
// compact the 0/1 boundary mask into segment-start positions (step 4
// below). CUDA 13.3's Thrust dropped thrust::identity (it now lives, under
// a different name, in the C++20-flavored <cuda/std/functional> that ships
// with CCCL) — a tiny hand-written functor is simpler than chasing that
// rename, and just as clear: "is this mask entry true".
struct is_nonzero {
    __host__ __device__ bool operator()(int x) const { return x != 0; }
};

// ===========================================================================
// Device-side transcription of kernels.cuh's shared key-arithmetic helpers.
//
// WHY DUPLICATED (read kernels.cuh's file header for the full reasoning):
// kernels.cuh's voxel_coord/pack_voxel_key/unpack_voxel_key/spatial_hash are
// PLAIN inline functions (no __host__/__device__) so reference_cpu.cpp's
// cl.exe compile can see them too — but that means nvcc treats them as
// HOST-only and refuses to call them from a __global__ kernel. The fix used
// throughout this repo (02.06's hidx()/blocks_for() precedent) is: keep ONE
// authoritative host version in the shared header, and give device code its
// own __device__ copy, written to match EXACTLY, with a comment saying so.
//
// This is the repo's "shared token-for-token transcription" case, which the
// reference_cpu.cpp independence ruling explicitly permits PROVIDED an
// independent verification gate exists that does not depend on the two
// copies agreeing. That gate is VERIFY(keys) in main.cu: every one of the
// N points' GPU-transcribed key (via these functions) is compared,
// bit-exact, against reference_cpu.cpp's key (via kernels.cuh's shared
// host functions). A drift between the two copies below and the header
// fails that gate immediately, on the very first mismatched point.
// ===========================================================================

// __forceinline__: these are called once per point in the hottest kernels
// below (compute_keys_kernel, hash_insert_kernel) — inlining removes the
// call overhead entirely, leaving just the arithmetic in the compiled SASS.
__device__ __forceinline__ int32_t d_voxel_coord(float p, float leaf)
{
    return static_cast<int32_t>(floorf(p / leaf));   // see kernels.cuh voxel_coord() for the floor-vs-truncate pitfall
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

__device__ __forceinline__ uint32_t d_spatial_hash(int32_t vx, int32_t vy, int32_t vz)
{
    return (static_cast<uint32_t>(vx) * kHashP1) ^
           (static_cast<uint32_t>(vy) * kHashP2) ^
           (static_cast<uint32_t>(vz) * kHashP3);
}

// ===========================================================================
// compute_keys_kernel — the shared first stage of BOTH methods: one thread
// per point, pack its voxel key. A pure MAP (each output depends only on
// its own input point) — the simplest GPU mapping in the repo's vocabulary,
// exactly like the SAXPY placeholder this project replaced.
//
// Thread-to-data mapping: thread (blockIdx.x, threadIdx.x) owns point
//     i = blockIdx.x * blockDim.x + threadIdx.x
// A grid-stride loop is NOT needed here (unlike the SAXPY placeholder):
// main.cu launches exactly ceil(n/threads) blocks (see launch_compute_keys),
// comfortably under the ~2^31 block-count ceiling for our N ~ 250k, so a
// single guarded launch is simpler to read and just as correct — a
// deliberate simplification the repo allows when grid-stride generality
// buys nothing (CLAUDE.md "teaching beats cleverness").
//
// Memory behavior: xyz reads are coalesced (adjacent threads read adjacent
// 12-byte (x,y,z) triples — not perfectly 128-byte-aligned per warp because
// of the stride-3 layout, but still sequential, unlike a random-access
// pattern); keys writes are one coalesced uint64 store per thread.
// ---------------------------------------------------------------------------
__global__ void compute_keys_kernel(int n, const float* __restrict__ xyz,
                                    float leaf, unsigned long long* __restrict__ keys)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Read this thread's own point (3 floats — never touched by any other
    // thread, so no synchronization or atomics are needed for this stage).
    const float px = xyz[i * 3 + 0];
    const float py = xyz[i * 3 + 1];
    const float pz = xyz[i * 3 + 2];

    const int32_t vx = d_voxel_coord(px, leaf);
    const int32_t vy = d_voxel_coord(py, leaf);
    const int32_t vz = d_voxel_coord(pz, leaf);
    keys[i] = d_pack_voxel_key(vx, vy, vz);
}

void launch_compute_keys(int n, const float* d_xyz, float leaf, unsigned long long* d_keys)
{
    const int block = kThreadsPerBlock;
    const int grid  = blocks_for(n, block);
    compute_keys_kernel<<<grid, block>>>(n, d_xyz, leaf, d_keys);
    CUDA_CHECK_LAST_ERROR("compute_keys_kernel launch");
}

// ===========================================================================
// METHOD A — atomic open-addressing hash table.
// ===========================================================================

// hash_reset_kernel — one thread per TABLE SLOT (capacity threads, not n):
// initialize every slot to "unclaimed" before a fresh insert pass. Run once
// per Method-A attempt (main.cu re-runs the whole reset->insert->compact
// sequence 3 times for the determinism study — each pass needs an empty
// table, or slots from the previous pass would double-count).
__global__ void hash_reset_kernel(HashTableGPU table)
{
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= table.capacity) return;
    table.keys[slot]  = kEmptyKey;   // the sentinel every insert's atomicCAS compares against
    table.sum_x[slot] = 0.0f;
    table.sum_y[slot] = 0.0f;
    table.sum_z[slot] = 0.0f;
    table.count[slot] = 0u;
}

void launch_hash_reset(HashTableGPU table)
{
    const int block = kThreadsPerBlock;
    const int grid  = blocks_for(table.capacity, block);
    hash_reset_kernel<<<grid, block>>>(table);
    CUDA_CHECK_LAST_ERROR("hash_reset_kernel launch");
}

// ---------------------------------------------------------------------------
// hash_insert_kernel — THE canonical GPU hash-insert pattern, walked through
// step by step (this is the project's central new GPU-programming idea).
//
// Every one of n points, IN PARALLEL, needs to either (a) find an empty
// slot for its voxel key and claim it, or (b) find the slot some other
// thread ALREADY claimed for the SAME key and add itself to it. With
// thousands of threads racing to do this simultaneously, a naive
// "check-then-write" (if table[slot]==EMPTY: table[slot]=key) has a classic
// TOCTOU race: two threads can both see EMPTY, both proceed to write, and
// one thread's claim silently overwrites the other's — a dropped point.
//
// atomicCAS(address, compare, val) is the hardware primitive that closes
// this race: it atomically reads *address, compares it to `compare`, and —
// ONLY if they matched — writes `val`, all as one indivisible operation
// (no other thread's atomicCAS on the same address can interleave inside
// it). It always returns the value that was AT *address before the
// operation, so the caller can tell whether its write happened.
//
// The claim-or-probe loop, in words:
//   1. Compute this point's home slot from its key's hash.
//   2. Try to claim that slot: atomicCAS(&keys[slot], EMPTY, my_key).
//      - old == EMPTY  -> we just claimed it FOR my_key. Done — accumulate.
//      - old == my_key -> someone else already claimed this exact slot for
//                          this exact voxel (a scheduling race, not a bug —
//                          many threads share a voxel key on real scans).
//                          Also done — accumulate into the SAME slot.
//      - otherwise     -> a DIFFERENT key already lives here (a hash
//                          collision between two different voxels landing
//                          in the same slot). Move to the next slot
//                          (linear probing: slot = (slot+1) & (capacity-1))
//                          and try again.
//   3. Repeat until case 1 or 2 fires (guaranteed to terminate within
//      `capacity` probes given the load factor main.cu sizes for — see
//      kTargetLoadFactor in kernels.cuh).
//
// Once a slot is claimed (by either case 1 or 2), the accumulation itself
// —atomicAdd on sum_x/sum_y/sum_z/count — is the SAME kind of race-free
// primitive: many threads add to the same 4 memory locations; atomicAdd
// serializes those writes (in an ORDER the hardware scheduler decides, not
// the program) so no addition is ever lost — but that scheduler-decided
// ORDER is exactly why Method A's float sums are only reproducible up to
// rounding (THEORY.md "Numerical considerations"; main.cu's
// determinism_method_a measurement quantifies it).
//
// probe_len[i] records how many probe steps point i's insert needed —
// pure bookkeeping for the hash_stats gate, costs one extra store.
// overflow_count increments (once, harmlessly redundantly if it ever
// fires more than once) if a point's insert exhausts every slot in the
// table without finding EMPTY or a match — should never happen given
// main.cu's capacity sizing; if it does, the whole run is invalid and
// main.cu's overflow check catches it loudly rather than silently losing
// a point's contribution.
// ---------------------------------------------------------------------------
__global__ void hash_insert_kernel(int n, const float* __restrict__ xyz,
                                   const unsigned long long* __restrict__ keys,
                                   HashTableGPU table,
                                   int* __restrict__ probe_len,
                                   unsigned int* __restrict__ overflow_count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const unsigned long long my_key = keys[i];
    int32_t vx, vy, vz;
    d_unpack_voxel_key(my_key, vx, vy, vz);

    // capacity is a power of two (next_pow2 in main.cu's sizing) so
    // "& (capacity-1)" is a cheap replacement for "% capacity".
    const unsigned int mask = static_cast<unsigned int>(table.capacity - 1);
    unsigned int slot = d_spatial_hash(vx, vy, vz) & mask;

    int probes = 0;
    for (; probes < table.capacity; ++probes) {
        // The one indivisible read-compare-write of the whole algorithm.
        const unsigned long long old = atomicCAS(&table.keys[slot], kEmptyKey, my_key);
        if (old == kEmptyKey || old == my_key) {
            // Either we just claimed this slot for my_key (old==EMPTY), or
            // it was already claimed for my_key by another thread
            // (old==my_key) — either way, this IS my voxel's slot now.
            atomicAdd(&table.sum_x[slot], xyz[i * 3 + 0]);
            atomicAdd(&table.sum_y[slot], xyz[i * 3 + 1]);
            atomicAdd(&table.sum_z[slot], xyz[i * 3 + 2]);
            atomicAdd(&table.count[slot], 1u);
            probe_len[i] = probes;
            return;
        }
        // A DIFFERENT voxel's key already lives here: linear probe onward.
        slot = (slot + 1u) & mask;
    }

    // Exhausted the entire table without claiming a slot — the capacity
    // sizing in main.cu is wrong or badly violated. Record it; do NOT
    // silently drop the point's contribution without a trace.
    atomicAdd(overflow_count, 1u);
    probe_len[i] = probes;
}

void launch_hash_insert(int n, const float* d_xyz, const unsigned long long* d_keys,
                        HashTableGPU table, int* d_probe_len, unsigned int* d_overflow_count)
{
    const int block = kThreadsPerBlock;
    const int grid  = blocks_for(n, block);
    hash_insert_kernel<<<grid, block>>>(n, d_xyz, d_keys, table, d_probe_len, d_overflow_count);
    CUDA_CHECK_LAST_ERROR("hash_insert_kernel launch");
}

// ---------------------------------------------------------------------------
// hash_compact_kernel — turn the sparse table (mostly-empty, `capacity`
// slots) into a dense output array (exactly `num_occupied` rows). One
// thread per SLOT; slots that were never claimed (key == EMPTY) do
// nothing. Claimed slots grab a dense output row via atomicAdd on a single
// shared counter (num_occupied) — the simplest possible GPU stream
// compaction, deliberately: a scan-based compaction (like Method B's
// boundary compaction below) is more WORK-efficient at scale, but this
// project already teaches that pattern in Method B, so Method A's
// compaction stays the simplest thing that works (CLAUDE.md "teaching
// beats cleverness" — and the atomic counter is O(1) contention per BLOCK
// in practice since most slots are empty, not O(capacity) serialized).
// ---------------------------------------------------------------------------
__global__ void hash_compact_kernel(HashTableGPU table,
                                    float* __restrict__ out_xyz,
                                    unsigned int* __restrict__ out_count,
                                    unsigned long long* __restrict__ out_key,
                                    unsigned int* __restrict__ num_occupied)
{
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= table.capacity) return;

    const unsigned long long key = table.keys[slot];
    if (key == kEmptyKey) return;   // never claimed — nothing to emit

    const unsigned int cnt = table.count[slot];
    // atomicAdd returns the value BEFORE the add — exactly the dense row
    // index this slot should write to (the classic "atomic counter as a
    // parallel push_back" idiom).
    const unsigned int row = atomicAdd(num_occupied, 1u);

    out_xyz[row * 3 + 0] = table.sum_x[slot] / static_cast<float>(cnt);
    out_xyz[row * 3 + 1] = table.sum_y[slot] / static_cast<float>(cnt);
    out_xyz[row * 3 + 2] = table.sum_z[slot] / static_cast<float>(cnt);
    out_count[row] = cnt;
    out_key[row]   = key;
}

void launch_hash_compact(HashTableGPU table, float* d_out_xyz, unsigned int* d_out_count,
                         unsigned long long* d_out_key, unsigned int* d_num_occupied)
{
    // Caller must have zeroed *d_num_occupied before this launch (main.cu
    // does so once per Method-A attempt, right after hash_reset).
    const int block = kThreadsPerBlock;
    const int grid  = blocks_for(table.capacity, block);
    hash_compact_kernel<<<grid, block>>>(table, d_out_xyz, d_out_count, d_out_key, d_num_occupied);
    CUDA_CHECK_LAST_ERROR("hash_compact_kernel launch");
}

// ===========================================================================
// METHOD B — Thrust sort + fixed-order segmented reduction.
// ===========================================================================

// mark_boundaries_kernel — one thread per SORTED-ARRAY position: is this
// where a new voxel's run of points begins? Position 0 always is; any
// later position is iff its key differs from the previous position's key
// (the array is sorted, so equal keys are always contiguous — this single
// neighbor comparison is enough to find every boundary).
__global__ void mark_boundaries_kernel(int n, const unsigned long long* __restrict__ keys_sorted,
                                       int* __restrict__ is_start)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    is_start[i] = (i == 0 || keys_sorted[i] != keys_sorted[i - 1]) ? 1 : 0;
}

// ---------------------------------------------------------------------------
// segmented_reduce_kernel — Method B's deterministic centroid computation.
//
// One thread per VOXEL (num_voxels threads, NOT n — a different grid size
// than every other kernel in this file, sized from a value computed at
// RUNTIME by the Thrust boundary count, see launch_sort_based_downsample).
// Thread v owns the sorted-array run [seg_start[v], seg_start[v+1)) (or
// [seg_start[v], n) for the last voxel) and walks it WITH A PLAIN
// SEQUENTIAL FOR LOOP, in ASCENDING SORTED-ARRAY-POSITION order, summing
// x/y/z in plain float — no atomics anywhere in this kernel.
//
// This is deliberately the LESS parallel-efficient design: real GPU
// segmented-reduction libraries (Thrust's own reduce_by_key, CUB's
// DeviceSegmentedReduce) spread ONE segment's work across MANY threads
// with a parallel tree reduction, which is faster for skewed segment
// sizes (exactly our adversarial dense cluster: one voxel can hold
// thousands of points while its neighbors hold one) — see THEORY.md
// "Where this sits in the real world" for the honest cost of this choice
// (load imbalance: the one thread covering the dense cluster's voxel does
// far more work than every other thread in its warp, which idles).
//
// The reason to accept that cost: a SEQUENTIAL, FIXED-ORDER sum in a
// single thread is the one design whose result is byte-for-byte
// reproducible against a CPU reference that performs the IDENTICAL
// sequence of additions (reference_cpu.cpp's sort_based_downsample_cpu) —
// IEEE-754 float addition is fully specified (round-to-nearest-even) given
// a fixed operand order, and neither this kernel nor the CPU twin uses any
// fused-multiply-add (there is no multiply here, only add) or fast-math
// flag that could perturb rounding (this repo never sets --use_fast_math,
// CLAUDE.md §5). That is the whole Method-B determinism story, achieved
// through ALGORITHM DESIGN rather than through hardware guarantees.
// ---------------------------------------------------------------------------
__global__ void segmented_reduce_kernel(int num_voxels, const int* __restrict__ seg_start, int n_total,
                                        const int* __restrict__ idx_sorted,
                                        const float* __restrict__ xyz,
                                        const unsigned long long* __restrict__ keys_sorted,
                                        float* __restrict__ out_xyz,
                                        unsigned int* __restrict__ out_count,
                                        unsigned long long* __restrict__ out_key)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_voxels) return;

    const int begin = seg_start[v];
    const int end   = (v + 1 < num_voxels) ? seg_start[v + 1] : n_total;

    // Plain float accumulators — see the kernel header comment above for
    // why NOT double here: bit-exactness against the CPU twin requires
    // matching precision exactly, and the twin also accumulates in float.
    float sx = 0.0f, sy = 0.0f, sz = 0.0f;
    for (int k = begin; k < end; ++k) {
        const int p = idx_sorted[k];   // original point index at sorted position k
        sx += xyz[p * 3 + 0];
        sy += xyz[p * 3 + 1];
        sz += xyz[p * 3 + 2];
    }

    const unsigned int cnt = static_cast<unsigned int>(end - begin);
    out_xyz[v * 3 + 0] = sx / static_cast<float>(cnt);
    out_xyz[v * 3 + 1] = sy / static_cast<float>(cnt);
    out_xyz[v * 3 + 2] = sz / static_cast<float>(cnt);
    out_count[v] = cnt;
    out_key[v]   = keys_sorted[begin];   // every point in [begin,end) shares this key by construction
}

// ---------------------------------------------------------------------------
// launch_sort_based_downsample — host orchestration of the whole Method-B
// pipeline: one Thrust sort, one boundary-marking kernel, two Thrust calls
// to compact the boundaries, one reduction kernel. See kernels.cuh for the
// full parameter documentation; this comment explains what each Thrust call
// computes and why it is used instead of a hand-rolled kernel (CLAUDE.md
// §6.1 rule 6 — "no black boxes").
// ---------------------------------------------------------------------------
int launch_sort_based_downsample(int n, const float* d_xyz, const unsigned long long* d_keys_in,
                                 unsigned long long* d_keys_scratch, int* d_idx_scratch,
                                 int* d_is_start_scratch, int* d_seg_start_out,
                                 float* d_out_xyz, unsigned int* d_out_count,
                                 unsigned long long* d_out_key)
{
    // 1) Copy the (shared, read-only) keys into a scratch buffer: Thrust's
    //    sort permutes its key range IN PLACE, and Method A / VERIFY(keys)
    //    still need d_keys_in untouched after this call returns.
    CUDA_CHECK(cudaMemcpy(d_keys_scratch, d_keys_in,
                          static_cast<size_t>(n) * sizeof(unsigned long long),
                          cudaMemcpyDeviceToDevice));

    // thrust::device_ptr<T> is a thin, zero-overhead WRAPPER around a raw
    // device pointer: it carries no data of its own, it just tells Thrust's
    // algorithm dispatch "this pointer lives in device memory, run the CUDA
    // backend" instead of assuming host memory. We keep raw cudaMalloc'd
    // pointers everywhere else in this repo (CLAUDE.md's teaching-
    // transparency preference over thrust::device_vector's RAII magic);
    // wrapping them only at the Thrust call site is the standard idiom for
    // getting that transparency AND Thrust's algorithms.
    thrust::device_ptr<unsigned long long> keys_ptr(d_keys_scratch);
    thrust::device_ptr<int>                idx_ptr(d_idx_scratch);

    // thrust::sequence(first, last) fills [first,last) with 0,1,2,...,n-1 —
    // a trivial parallel "iota" (one thread per output element). idx[i] = i
    // is the IDENTITY permutation: after the sort below, idx[k] will read
    // "the original point index now sitting at sorted position k".
    thrust::sequence(idx_ptr, idx_ptr + n);

    // thrust::stable_sort_by_key(keys_first, keys_last, values_first) sorts
    // the KEY range ascending and permutes the paired VALUE range (here,
    // idx) the same way — i.e. it sorts (key[i], idx[i]) pairs by key.
    // Internally, for integer keys like our uint64_t voxel keys, Thrust
    // dispatches a RADIX sort: repeated stable partitioning by 1 byte (or a
    // few bits) of the key at a time, from least to most significant,
    // O(n * key_width/radix_bits) work, fully data-parallel per pass (a
    // hand-rolled version would need a multi-pass counting-sort/prefix-sum
    // pipeline per byte — exactly what 22.01's counting-sort neighbor
    // binning does by hand at a smaller scale; Thrust's radix sort is that
    // idea, generalized to 64-bit keys and highly tuned). We use the
    // library here specifically for its STABILITY guarantee: "stable"
    // means equal keys keep their RELATIVE INPUT ORDER — since idx starts
    // as 0,1,2,...,n-1 (ascending original point index), a stable sort
    // guarantees that within any voxel's run, points appear in ASCENDING
    // ORIGINAL INDEX order. That specific, simple tie-break rule is exactly
    // what reference_cpu.cpp's std::stable_sort reproduces on the CPU —
    // making the two permutations IDENTICAL, which is the foundation the
    // whole Method-B bit-exactness story is built on.
    thrust::stable_sort_by_key(keys_ptr, keys_ptr + n, idx_ptr);

    // 2) Mark segment boundaries in the now-sorted key array (our own
    //    kernel — a plain map, no library needed for a single neighbor
    //    comparison per element).
    {
        const int block = kThreadsPerBlock;
        const int grid  = blocks_for(n, block);
        mark_boundaries_kernel<<<grid, block>>>(n, d_keys_scratch, d_is_start_scratch);
        CUDA_CHECK_LAST_ERROR("mark_boundaries_kernel launch");
    }

    // 3) thrust::reduce(first, last, init) sums a range — here, the 0/1
    // boundary mask, so the sum IS the number of distinct voxels (one
    // "1" per segment start). This one call runs a standard parallel tree
    // reduction on the device and returns the scalar RESULT to the HOST
    // (a synchronizing call — main.cu can use num_voxels immediately after
    // this line, no manual cudaMemcpy needed).
    thrust::device_ptr<int> is_start_ptr(d_is_start_scratch);
    const int num_voxels = thrust::reduce(is_start_ptr, is_start_ptr + n, 0);

    // 4) thrust::copy_if(first, last, stencil_first, result, pred) copies
    // every element of [first,last) whose corresponding STENCIL entry
    // satisfies pred into a compacted output range — the standard STREAM
    // COMPACTION primitive (keep only the "interesting" elements, densely
    // packed, in their original relative order). Here [first,last) is a
    // COUNTING iterator (0,1,2,...,n-1 generated on the fly, not stored —
    // "which sorted-array position am I") and the stencil is our boundary
    // mask, so the result is exactly seg_start[0..num_voxels) = the sorted-
    // array positions where each voxel's run begins, in ascending voxel-
    // key order. is_nonzero (defined near the top of this file) is the
    // predicate: "keep this position iff its stencil (boundary mask) entry
    // is nonzero" — CUDA 13.3's Thrust dropped thrust::identity, so a
    // two-line hand-written functor replaces it.
    thrust::device_ptr<int> seg_start_ptr(d_seg_start_out);
    thrust::copy_if(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(n),
                    is_start_ptr, seg_start_ptr, is_nonzero());

    // 5) The deterministic reduction itself — see segmented_reduce_kernel's
    // header comment. Grid sized from num_voxels (a RUNTIME value, unlike
    // every other launch in this file), so the ceiling division happens
    // here rather than being baked into a compile-time constant.
    {
        const int block = kThreadsPerBlock;
        const int grid  = blocks_for(num_voxels, block);
        segmented_reduce_kernel<<<grid, block>>>(num_voxels, d_seg_start_out, n,
                                                 d_idx_scratch, d_xyz, d_keys_scratch,
                                                 d_out_xyz, d_out_count, d_out_key);
        CUDA_CHECK_LAST_ERROR("segmented_reduce_kernel launch");
    }

    return num_voxels;
}
