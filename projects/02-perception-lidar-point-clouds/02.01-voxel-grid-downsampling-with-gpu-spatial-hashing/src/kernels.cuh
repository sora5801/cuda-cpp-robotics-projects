// ===========================================================================
// kernels.cuh — interface for project 02.01
//               Voxel-grid downsampling with GPU spatial hashing
//               (Method A: atomic open-addressing hash table.
//                Method B: Thrust sort + fixed-order segmented reduction.)
//
// Role in the project
// -------------------
// The CONTRACT shared by main.cu (orchestration + gates), kernels.cu (the
// GPU kernels + Thrust-based Method B pipeline), and reference_cpu.cpp (the
// CPU oracle twins). Everything all three must agree on — the point-cloud
// layout, the voxel-key packing, the hash function, the hash-table layout —
// is defined HERE, once (CLAUDE.md §12: data-layout contracts are
// single-sourced; see the "independence ruling" in reference_cpu.cpp).
//
// POINT CLOUD LAYOUT — float* xyz, interleaved, meters, "lidar" sensor frame:
//     xyz[i*3 + 0] = x, xyz[i*3 + 1] = y, xyz[i*3 + 2] = z
// This is the SAME convention project 02.06 (this domain's flagship, ICP)
// and docs/SYSTEM_DESIGN.md §3.6's `PointCloud` message sketch use — a
// flattened sensor_msgs/PointCloud2 — so this project's output slots
// directly into 02.06's `src_xyz` input with zero reshaping (README "System
// context" names this exact hand-off).
//
// THE PROBLEM in one paragraph
// -----------------------------
// N ~ 200k-500k raw LiDAR points arrive every scan; most robot consumers
// (ICP, NDT, clustering, mapping) do not want every raw point — they want
// ONE representative point per small cube ("voxel") of world space, because
// (a) LiDAR point DENSITY falls off as 1/r^2 with range (THEORY.md derives
// this from beam geometry), so near-field voxels are absurdly oversampled
// relative to far-field ones, and (b) fewer, evenly-spaced points make every
// downstream O(N) or O(N^2) algorithm cheaper without losing the scene's
// shape. "Downsample" here means: partition points into voxels of edge
// length L (kVoxelLeafM below), and replace each occupied voxel's points
// with their CENTROID (mean position) — see THEORY.md "The algorithm".
//
// TWO METHODS, ONE ANSWER (up to float rounding) — the project's teaching
// core, a lesson in GPU data structures and determinism:
//
//   METHOD A — atomic spatial hash table (kHashInsert below). Each point's
//   voxel key is inserted into an open-addressing hash table via an
//   atomicCAS claim-or-probe loop (the canonical GPU hash-insert pattern —
//   THEORY.md "The GPU mapping"), then atomicAdd accumulates that voxel's
//   running sum/count. FAST and simple, but the ORDER in which points'
//   atomicAdd calls interleave is a hardware-scheduling accident — different
//   every run — so the float SUMS (not the voxel SET, not the counts) are
//   only reproducible up to float-accumulation rounding. This is measured,
//   not asserted (THEORY.md "Numerical considerations").
//
//   METHOD B — sort + fixed-order segmented reduction (kSortBasedDownsample
//   below). Points are STABLE-sorted by voxel key (thrust::stable_sort_by_key
//   — a radix sort under the hood, CLAUDE.md rule 6 explains what that
//   computes in kernels.cu), which turns "which points share a voxel" into
//   "a contiguous run in a sorted array" — then ONE thread per voxel walks
//   its run SEQUENTIALLY, in a fixed order (ascending original point index,
//   guaranteed by sort STABILITY), summing floats one at a time. Because the
//   order is fixed and reproducible, both across GPU runs AND against a CPU
//   twin that reproduces the identical order, Method B is BIT-EXACT — the
//   determinism headline continuing 01.13/01.14's *integer*-arithmetic
//   determinism designs, but achieved here by fixing ORDER instead, since
//   the underlying quantity (a centroid) is inherently float (THEORY.md
//   "Numerical considerations" spells this parallel out explicitly).
//
// VOXEL KEY — floor(p / L) per axis, packed into a 64-bit integer (bias +
// 21-bit-per-axis packing, see pack_voxel_key below). floor(), not
// truncation: floor(-0.3 / 0.2) = floor(-1.5) = -2 is the CORRECT voxel
// index for a point at x = -0.3 m with 20 cm voxels (the voxel spanning
// [-0.4, -0.2)); the naive (int)(-0.3/0.2) = (int)(-1.5) = -1 truncates
// TOWARD ZERO and silently puts negative-side points in the wrong voxel —
// see voxel_coord() below and THEORY.md "The math" for the full derivation.
//
// Why this header is CUDA-qualifier-free (no __host__ __device__, following
// 02.06's precedent exactly): every function below is a PLAIN inline C++
// function, deliberately WITHOUT __host__/__device__ qualifiers, so it
// compiles cleanly under BOTH nvcc (main.cu) and cl.exe (reference_cpu.cpp,
// which never sees CUDA keywords). The cost: these plain functions are
// HOST-only under nvcc's rules and therefore CANNOT be called from inside a
// __global__ kernel. kernels.cu's device code therefore carries its OWN
// literal __device__ transcription of voxel_coord/pack_voxel_key/
// unpack_voxel_key/spatial_hash (commented as such at its definition) —
// this is the "shared token-for-token transcription" case the independence
// ruling in reference_cpu.cpp explicitly permits, PROVIDED an independent
// gate exists that would catch drift between the two copies. That gate is
// VERIFY(keys) in main.cu: it compares the GPU (device-transcribed) key for
// every one of the N points against the CPU (this header's shared function)
// key, bit-exact. A typo in either copy fails that gate immediately.
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cstdint>       // int32_t, uint32_t, uint64_t — exact-width integers for the key packing
#include <cmath>         // std::floor (float overload == floorf, see voxel_coord)
#include <unordered_map> // reference_cpu.cpp's independent Method-A oracle uses a hash map
                         // (a genuinely different data structure than Method A's GPU open-
                         // addressing table — see hashmap_downsample_cpu below)

// ===========================================================================
// Problem-scale constants
// ===========================================================================

// Voxel edge length L (meters). 20 cm is a standard indoor-AMR / warehouse
// leaf size (PCL's VoxelGrid tutorial default is 1 cm for tabletop scans,
// 5-20 cm is typical for room-scale mobile robots) — small enough to keep
// local geometry (a doorway, a box corner) resolvable, large enough to
// collapse this project's near-field oversampling by 1-2 orders of
// magnitude (README "Expected output" reports the measured ratio).
constexpr float kVoxelLeafM = 0.20f;

// Repo-default block size (warp multiple, good occupancy on sm_75..sm_89 —
// see kernels.cu's launch-configuration comments for the per-kernel story).
constexpr int kThreadsPerBlock = 256;

// ---------------------------------------------------------------------------
// Voxel key packing: 3 signed integer voxel coordinates -> one 64-bit key.
//
// Each axis gets 21 bits (63 bits total; bit 63 is always 0 for a valid
// key — see kEmptyKey below). A biased (offset-binary) encoding turns a
// SIGNED coordinate into an UNSIGNED field: adding kCoordBias (2^20) before
// masking shifts the representable range [-2^20, 2^20 - 1] to [0, 2^21 - 1].
//
// Overflow bound (THEORY.md "Numerical considerations" does this arithmetic
// in full): at L = 0.20 m, a 21-bit signed voxel coordinate covers
// +/- 2^20 * 0.20 m = +/- 209,715.2 m (~210 km) from the origin per axis —
// this project's scene spans tens of meters, so headroom is enormous; the
// bound exists so a learner can see exactly WHY 21 bits (not fewer) was a
// safe, deliberate choice rather than an arbitrary one.
// ---------------------------------------------------------------------------
constexpr int32_t  kCoordBias   = 1 << 20;           // 1,048,576 — recenters [-2^20,2^20-1] to [0,2^21-1]
constexpr uint64_t kCoordMask21 = (1ull << 21) - 1ull; // low 21 bits: 0x1FFFFF

// Sentinel marking an unclaimed hash-table slot. All bits set (bit 63
// included); every VALID packed key has bit 63 clear (three 21-bit fields
// span only bits 0..62), so this value can never collide with a real key —
// no separate "occupied" flag array is needed (Method A's kernels.cu comment
// explains the atomicCAS claim loop that relies on this property).
constexpr uint64_t kEmptyKey = ~0ull;

// ---------------------------------------------------------------------------
// voxel_coord — floor(p / leaf) as an integer voxel index along one axis.
//
// THE PITFALL THIS AVOIDS (spelled out because it is a real, easy-to-miss
// bug class): C++'s float-to-int conversion TRUNCATES TOWARD ZERO, not
// floor. For a positive p this happens to agree with floor (p=0.3m,
// leaf=0.2m: p/leaf=1.5, truncate=1=floor — fine). For a NEGATIVE p it does
// not: p=-0.3m, leaf=0.2m: p/leaf=-1.5; (int)(-1.5) truncates to -1, but
// floor(-1.5) = -2. The correct voxel for x=-0.3m with 20cm voxels is the
// one spanning [-0.4,-0.2) — voxel index -2 — not [-0.2,0.0) (index -1),
// which is where the point at x=-0.3 quite obviously does NOT lie. Using
// std::floor (== floorf for the float overload, available identically to
// both cl.exe and nvcc — no CUDA-specific intrinsic needed) gets this right
// unconditionally, for every sign of p.
//
// Parameters: p (m, any sign), leaf (m, > 0). Returns: the voxel index
// (integer, any sign) such that p lies in [index*leaf, (index+1)*leaf).
// ---------------------------------------------------------------------------
inline int32_t voxel_coord(float p, float leaf)
{
    return static_cast<int32_t>(std::floor(p / leaf));
}

// ---------------------------------------------------------------------------
// pack_voxel_key / unpack_voxel_key — the 64-bit <-> (vx,vy,vz) bijection
// every stage of both methods and both CPU twins shares (the "data-layout
// contract" the independence ruling requires to be single-sourced).
//
// Layout: key = ux | (uy << 21) | (uz << 42), where ux/uy/uz are the
// bias-shifted UNSIGNED 21-bit encodings of vx/vy/vz. This layout also
// happens to sort in (vz, vy, vx) lexicographic order when compared as a
// plain integer — irrelevant for correctness (Method B does not rely on any
// particular spatial ordering, only that EQUAL voxels get EQUAL keys and
// vice versa) but a pleasant, mentionable side effect.
// ---------------------------------------------------------------------------
inline uint64_t pack_voxel_key(int32_t vx, int32_t vy, int32_t vz)
{
    const uint64_t ux = static_cast<uint64_t>(vx + kCoordBias) & kCoordMask21;
    const uint64_t uy = static_cast<uint64_t>(vy + kCoordBias) & kCoordMask21;
    const uint64_t uz = static_cast<uint64_t>(vz + kCoordBias) & kCoordMask21;
    return ux | (uy << 21) | (uz << 42);
}

inline void unpack_voxel_key(uint64_t key, int32_t& vx, int32_t& vy, int32_t& vz)
{
    vx = static_cast<int32_t>(key & kCoordMask21) - kCoordBias;
    vy = static_cast<int32_t>((key >> 21) & kCoordMask21) - kCoordBias;
    vz = static_cast<int32_t>((key >> 42) & kCoordMask21) - kCoordBias;
}

// ---------------------------------------------------------------------------
// spatial_hash — the classic Teschner et al. 2003 spatial hash ("Optimized
// Spatial Hashing for Collision Detection of Deformable Objects"): XOR the
// three grid coordinates together, each first multiplied by a large prime.
//
// h(vx,vy,vz) = (vx * p1) XOR (vy * p2) XOR (vz * p3)
//
// The three primes (73,856,093 / 19,349,663 / 83,492,791) are the paper's
// published constants, chosen empirically so that small, spatially-nearby
// integer coordinate changes (exactly what a LiDAR scan produces — most
// voxels differ from a neighbor by +/-1 in one axis) scatter widely across
// the hash range instead of landing in a visibly clustered pattern. The
// multiply is done in UNSIGNED 32-bit arithmetic deliberately: unsigned
// overflow (wraparound mod 2^32) is well-defined in C++ and is not a bug
// here — the wraparound IS part of the mixing, exactly like it is in any
// multiplicative hash. Casting vx/vy/vz to uint32_t first (a defined,
// bit-preserving reinterpretation of a two's-complement negative number)
// keeps that multiply portable and identical between CPU and GPU.
//
// The caller reduces this 32-bit spread to a table slot via `& (capacity-1)`
// (capacity is a power of two — kernels.cu's launch/insert comments explain
// the sizing), a cheap AND in place of an expensive modulo.
// ---------------------------------------------------------------------------
constexpr uint32_t kHashP1 = 73856093u;
constexpr uint32_t kHashP2 = 19349663u;
constexpr uint32_t kHashP3 = 83492791u;

inline uint32_t spatial_hash(int32_t vx, int32_t vy, int32_t vz)
{
    return (static_cast<uint32_t>(vx) * kHashP1) ^
           (static_cast<uint32_t>(vy) * kHashP2) ^
           (static_cast<uint32_t>(vz) * kHashP3);
}

// next_pow2 — smallest power of two >= x (x >= 1). Used once, on the host,
// to size Method A's hash table from the point count (see kTargetLoadFactor
// below and main.cu's table-sizing comment). The classic "smear the highest
// set bit rightward, then +1" bit trick — O(log 32) = 5 shifts, not a loop
// over bits, and this repo's `07.09`/`33.01`-style "small closed-form beats
// a library call for something this cheap" preference.
inline uint32_t next_pow2(uint32_t x)
{
    if (x < 1u) return 1u;
    x--;
    x |= x >> 1;  x |= x >> 2;  x |= x >> 4;  x |= x >> 8;  x |= x >> 16;
    return x + 1u;
}

// blocks_for — integer ceiling division: how many `threads`-wide blocks
// cover `count` independent problems (the same idiom 02.06/08.01/33.01 use).
inline int blocks_for(int count, int threads)
{
    return (count + threads - 1) / threads;
}

// Method A's target load factor (occupied slots / capacity) at the point
// where main.cu sizes the table: occupied voxels can never exceed N points,
// so sizing capacity = next_pow2(N / kTargetLoadFactor) guarantees the
// REALIZED load factor is <= this value even in the (unrealistic) worst
// case that every point lands in its own voxel. THEORY.md "The math" ties
// this number to open-addressing probe-length theory (why 0.5, not 0.9).
constexpr float kTargetLoadFactor = 0.5f;

// ===========================================================================
// Method A device-side data: a small bundle of raw device pointers, passed
// BY VALUE to every Method-A launcher (cheap: 5 pointers + 1 int, ~44 bytes
// — the same "small POD by value" reasoning 02.06's Rigid3 comment gives).
// main.cu owns the allocations; these launchers only read/write through them.
// ===========================================================================
struct HashTableGPU {
    unsigned long long* keys;    // [capacity] packed voxel key, or kEmptyKey if the slot is unclaimed
    float*        sum_x;         // [capacity] atomicAdd accumulator, meters (x)
    float*        sum_y;         // [capacity] atomicAdd accumulator, meters (y)
    float*        sum_z;         // [capacity] atomicAdd accumulator, meters (z)
    unsigned int* count;         // [capacity] atomicAdd accumulator, points claimed by this slot
    int           capacity;      // power-of-two slot count (see next_pow2/kTargetLoadFactor above)
};

// ===========================================================================
// GPU kernel declarations — nvcc-only (see the file header for why this
// fence exists: cl.exe, compiling reference_cpu.cpp, has never heard of
// __global__ and must never see these).
// ===========================================================================
#ifdef __CUDACC__

// compute_keys_kernel — one thread per point: pack this point's voxel key.
// Shared by BOTH methods (the single-sourced key layout). in xyz [n*3]
// device floats; out keys [n] device uint64_t.
__global__ void compute_keys_kernel(int n, const float* __restrict__ xyz,
                                    float leaf, unsigned long long* __restrict__ keys);

// hash_reset_kernel — one thread per SLOT (not per point): initialize the
// table to the empty state before insertion. Must run before every fresh
// Method-A pass (main.cu re-runs it for the 3-run determinism study).
__global__ void hash_reset_kernel(HashTableGPU table);

// hash_insert_kernel — Method A's heart: one thread per point, atomicCAS
// claim-or-probe insert + atomicAdd accumulation (full walkthrough in
// kernels.cu). probe_len[i] records how many linear-probe steps point i's
// insert took (the hash_stats gate's raw material); overflow_count counts
// insertions that exhausted the whole table without success (should stay 0
// given main.cu's capacity sizing — a correctness assertion, not expected
// behavior).
__global__ void hash_insert_kernel(int n, const float* __restrict__ xyz,
                                   const unsigned long long* __restrict__ keys,
                                   HashTableGPU table,
                                   int* __restrict__ probe_len,
                                   unsigned int* __restrict__ overflow_count);

// hash_compact_kernel — one thread per TABLE SLOT: slots with a claimed key
// atomically grab a dense output index (num_occupied) and emit their
// centroid. out_xyz/out_count/out_key are sized >= N (an upper bound on
// occupied voxels) by the caller; only the first *num_occupied entries are
// valid on return.
__global__ void hash_compact_kernel(HashTableGPU table,
                                    float* __restrict__ out_xyz,
                                    unsigned int* __restrict__ out_count,
                                    unsigned long long* __restrict__ out_key,
                                    unsigned int* __restrict__ num_occupied);

// mark_boundaries_kernel — Method B: one thread per SORTED-ARRAY position;
// marks 1 where a new voxel segment begins (position 0, or key changed from
// the previous position), 0 otherwise. Feeds the Thrust compaction step in
// launch_sort_based_downsample (kernels.cu explains why Thrust, not a
// hand-rolled scan, does that step).
__global__ void mark_boundaries_kernel(int n, const unsigned long long* __restrict__ keys_sorted,
                                       int* __restrict__ is_start);

// segmented_reduce_kernel — Method B's deterministic reduction: one thread
// per VOXEL (not per point!), walking its point run in seg_start[v] ..
// (seg_start[v+1] or n) IN SORTED-ARRAY ORDER, summing floats sequentially.
// This fixed order is what makes Method B bit-exact against
// reference_cpu.cpp's sort_based_downsample_cpu twin — see that function's
// comment and THEORY.md "Numerical considerations".
__global__ void segmented_reduce_kernel(int num_voxels, const int* __restrict__ seg_start, int n_total,
                                        const int* __restrict__ idx_sorted,
                                        const float* __restrict__ xyz,
                                        const unsigned long long* __restrict__ keys_sorted,
                                        float* __restrict__ out_xyz,
                                        unsigned int* __restrict__ out_count,
                                        unsigned long long* __restrict__ out_key);

#endif // __CUDACC__

// ===========================================================================
// Host-callable launch wrappers (definitions in kernels.cu, which only nvcc
// compiles — but the DECLARATIONS below are plain C++, visible to main.cu).
// ===========================================================================

void launch_compute_keys(int n, const float* d_xyz, float leaf, unsigned long long* d_keys);

void launch_hash_reset(HashTableGPU table);

void launch_hash_insert(int n, const float* d_xyz, const unsigned long long* d_keys,
                        HashTableGPU table, int* d_probe_len, unsigned int* d_overflow_count);

void launch_hash_compact(HashTableGPU table, float* d_out_xyz, unsigned int* d_out_count,
                         unsigned long long* d_out_key, unsigned int* d_num_occupied);

// launch_sort_based_downsample — Method B end to end (Thrust sort +
// boundary compaction + fixed-order segmented reduction — all orchestrated
// on the device from a single host call). Every array below is sized [n]
// (or [n*3] for xyz) by the caller — a safe upper bound, since the number
// of occupied voxels can never exceed the number of points.
//
//   d_xyz            : [n*3] the point cloud (read-only).
//   d_keys_in         : [n] voxel keys from launch_compute_keys (read-only;
//                       shared with Method A — the single-sourced layout).
//   d_keys_scratch    : [n] SCRATCH — a sortable copy of d_keys_in (Thrust
//                       sorts in place; the input must stay untouched for
//                       Method A / the VERIFY(keys) gate to reuse it).
//   d_idx_scratch     : [n] SCRATCH — becomes the sorted permutation of
//                       point indices (idx_scratch[k] = original point index
//                       now at sorted position k).
//   d_is_start_scratch: [n] SCRATCH — the segment-boundary 0/1 mask.
//   d_seg_start_out   : [n] OUT (first return-value entries valid) — sorted-
//                       array offset where each voxel's run begins.
//   d_out_xyz/_count/_key : [n*3]/[n]/[n] OUT (first return-value entries
//                       valid) — the downsampled result, in ASCENDING
//                       voxel-key order (a side effect of the sort, not a
//                       promise the caller needs to rely on, but a
//                       convenient one for the determinism/positional
//                       comparisons in main.cu).
//
// Returns: the number of occupied voxels (== valid rows in every *_out
// array above).
int launch_sort_based_downsample(int n, const float* d_xyz, const unsigned long long* d_keys_in,
                                 unsigned long long* d_keys_scratch, int* d_idx_scratch,
                                 int* d_is_start_scratch, int* d_seg_start_out,
                                 float* d_out_xyz, unsigned int* d_out_count,
                                 unsigned long long* d_out_key);

// ===========================================================================
// CPU references (reference_cpu.cpp) — the correctness-oracle twins. All
// pointers below are HOST pointers. See reference_cpu.cpp's file header for
// the independence ruling each of these three follows (they are NOT all
// independent in the same way — read that comment before assuming so).
// ===========================================================================

// compute_keys_cpu — the twin of compute_keys_kernel, calling this header's
// OWN voxel_coord/pack_voxel_key (a shared data-layout formula, not a
// duplicated algorithm — see the file header). VERIFY(keys) in main.cu
// compares this, point for point, against the GPU's device-transcribed
// version — the gate that catches any drift between the two copies.
void compute_keys_cpu(int n, const float* xyz, float leaf, unsigned long long* keys_out);

// sort_based_downsample_cpu — Method B's BIT-EXACT twin: std::stable_sort
// by key (mirroring thrust::stable_sort_by_key's stability guarantee
// exactly), then the identical fixed-order (ascending original-index,
// within a voxel) sequential float accumulation the GPU's
// segmented_reduce_kernel performs. Output arrays sized >= n by the caller;
// returns the number of occupied voxels, in ascending voxel-key order —
// the same order the GPU path produces, so main.cu can compare the two
// POSITIONALLY (no key-lookup matching needed) for the bit-exactness gate.
int sort_based_downsample_cpu(int n, const float* xyz, float leaf,
                              float* out_xyz, unsigned int* out_count,
                              unsigned long long* out_key);

// VoxelAccumD — one voxel's running sum in reference_cpu.cpp's INDEPENDENT
// Method-A oracle below. DOUBLE precision deliberately: this CPU twin's job
// is to be the more-precise, differently-ordered, differently-structured
// reference that Method A's float/atomicAdd/hash-table GPU result is
// compared AGAINST — the same "give the oracle better precision than the
// thing under test" choice 02.06's build_normal_system_cpu makes.
struct VoxelAccumD {
    double sx = 0.0, sy = 0.0, sz = 0.0;  // running position sum (m), double precision
    unsigned int count = 0;               // points accumulated into this voxel
};

// hashmap_downsample_cpu — Method A's INDEPENDENT twin (genuinely different
// from the GPU path in THREE ways: data structure — std::unordered_map's
// chaining/open-hashing internals, not this project's open-addressing
// table; ACCUMULATION ORDER — sequential point index 0..n-1, not GPU thread-
// scheduling order; PRECISION — double, not float). Voxel SET and per-voxel
// COUNTS must match the GPU exactly (both are integer bookkeeping,
// independent of summation order or precision); centroid VALUES are
// compared within a measured-then-margined tolerance (THEORY.md "How we
// verify correctness" documents the measured number).
void hashmap_downsample_cpu(int n, const float* xyz, float leaf,
                            std::unordered_map<unsigned long long, VoxelAccumD>& out);

#endif // PROJECT_KERNELS_CUH
