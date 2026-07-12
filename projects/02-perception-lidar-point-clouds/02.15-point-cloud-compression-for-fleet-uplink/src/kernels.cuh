// ===========================================================================
// kernels.cuh — interface for project 02.15
//               Point cloud compression (octree/entropy) for fleet uplink
//               (occupancy-octree geometry coding + canonical-Huffman entropy
//                coding of the octree's byte stream — a two-stage lossy-then-
//                lossless codec taught end to end, with a measured
//                rate-distortion sweep as the payoff)
//
// Role in the project
// -------------------
// The CONTRACT shared by main.cu (load both committed clouds, run the VERIFY
// stage, sweep depths, write every artifact), kernels.cu (the GPU kernels),
// and reference_cpu.cpp (the independent CPU oracle twins + the ONLY decoder
// — decode is host-only by design, see "Why decode is host-only" below).
//
// POINT CLOUD LAYOUT — float* xyz, interleaved, meters, right-handed MAP
// frame (+x East-like, +y North-like, +z Up — this project's clouds are
// static MAP TILES, not single-viewpoint sensor scans, so "map frame" is the
// honest name; SYSTEM_DESIGN.md's frame convention applies unchanged):
// xyz[i*3+0..2] = x,y,z. Both committed clouds (the structured map tile and
// the pathological uniform-random cube) share this exact layout.
//
// THE TWO-STAGE CODEC (this project's teaching spine — every stage below is
// built once, here, and consumed by main.cu in this exact order; THEORY.md
// "The algorithm" derives each step from first principles):
//
//   STAGE 1 — GEOMETRY: THE OCTREE AS A STRING PROBLEM. Quantize the scene's
//   (cube-padded) bounding box to a D-bit-per-axis integer grid (leaf size =
//   cube_extent / 2^D) and Morton-encode each point's (ix,iy,iz) into one
//   3D-bit code (morton_encode below — the SAME bit-interleave idea 02.05
//   uses for its LBVH, generalized here to a variable, swept depth D instead
//   of a fixed 10 bits/axis, and widened to uint64_t because 3*11=33 bits
//   overflows uint32_t at this project's deepest swept D=11). SORTING the
//   codes ascending is the entire trick: because Morton bit groups are
//   ordered coarsest-first (bit 3*(D-1)..3*(D-1)+2 is the ROOT split, bits
//   0..2 are the finest/leaf split), a sorted code array is EXACTLY a
//   depth-first, octant-ascending traversal of the octree in disguise — two
//   points share an internal node at level l if and only if their codes
//   share the same TOP 3*l bits (node_prefix below). This turns "build the
//   octree" into "find where consecutive sorted codes' top-3l-bit prefixes
//   change" — a STRING problem (longest-common-prefix boundaries), solved
//   with the exact predicate-plus-scan-plus-compact machinery 02.02 built
//   for stream compaction, reused here for a different purpose: LABELING
//   points with their owning node index at each level, not just filtering
//   them. See kernels.cu's build_octree_level_kernel-family header for the
//   full level-by-level walkthrough and the OCCUPANCY BYTE construction
//   (one byte per internal node: bit c set iff child octant c is non-empty
//   — the classic occupancy-octree representation, THEORY.md "The math").
//
//   STAGE 2 — ENTROPY: CANONICAL HUFFMAN OVER THE 256 OCCUPANCY SYMBOLS.
//   The occupancy-byte stream this project's octree emits is the compressed
//   GEOMETRY code (1 byte per internal node, already far smaller than raw
//   float32 xyz — the R-D sweep quantifies exactly how much smaller). It is
//   ALSO skewed: real surfaces are 2-D manifolds, so most internal nodes see
//   only a handful of their 8 children occupied (THEORY.md "The problem"
//   derives why, and the pathological uniform-random cohort measures the
//   contrast). A measured 256-symbol HISTOGRAM feeds a HOST-SIDE canonical
//   Huffman table build (build_huffman_table below — inherently small/
//   serial, no GPU pattern to teach there); the table then drives a THIRD
//   parallel pattern on the GPU: per-symbol code-LENGTH lookup (a map) ->
//   EXCLUSIVE SCAN of those lengths into per-symbol BIT OFFSETS (the exact
//   same scan primitive 02.02 built for compaction, reused here to answer
//   "which bit does my variable-length code start at?" instead of "which
//   slot do I compact to?" — CLAUDE.md's "tie it together" in its purest
//   form) -> a BIT-SCATTER kernel that writes each symbol's code into the
//   packed output via atomicOr (bit-level, not byte-level, because a
//   variable-length code routinely straddles a byte boundary and multiple
//   symbols routinely share one output byte — see kernels.cu for the full
//   correctness argument for why disjoint-bit atomicOr races are safe).
//
// Why decode is host-only (a design choice, not an omission)
// ------------------------------------------------------------
// ENCODE parallelizes cleanly because every symbol's output BIT OFFSET is
// knowable in advance from a scan over LENGTHS alone — no symbol needs to
// know any OTHER symbol's decoded VALUE first. DECODE has the opposite
// shape: Huffman codes are prefix-free but variable-length, so decoding
// symbol i requires knowing exactly how many bits symbol i-1 consumed,
// which requires having decoded symbol i-1 — an inherently SERIAL
// dependency chain with no scan-shaped escape hatch (THEORY.md "The GPU
// mapping" names the real answer production decoders use: BLOCK-WISE
// decode with periodic byte-aligned restart points, sacrificing a little
// rate for parallelism — documented, not implemented, per this project's
// ratified scope). This project's decoder therefore lives entirely in
// reference_cpu.cpp as plain sequential C++: a bit-trie Huffman walk
// followed by a level-by-level octree expansion that reconstructs LEAF
// CENTERS — the reconstructed, quantized point cloud a fleet's cloud side
// would receive. Its correctness is checked by an END-TO-END round-trip
// gate (decode(encode(cloud)) reproduces the depth-D quantized cloud
// EXACTLY — main.cu's GATE lossless_roundtrip), the appropriate
// verification shape for a component with no GPU counterpart to twin
// against (the reference_cpu.cpp file header's "independence ruling"
// explicitly anticipates this case: "a component with no GPU counterpart
// is verified end-to-end instead of by twin comparison").
//
// Why this header is CUDA-qualifier-free where possible (02.01/02.05's
// precedent, reused verbatim): pure math/bit-arithmetic helpers below
// (quantize_axis, morton_encode, point_to_code, node_prefix, child_octant,
// decode_leaf_center) are PLAIN inline C++ — no __host__/__device__ — so
// they compile under BOTH nvcc (main.cu, kernels.cu's host-side code) and
// cl.exe (reference_cpu.cpp). Being unqualified, they are HOST-only under
// nvcc's rules and cannot be called from a __global__ kernel; kernels.cu
// therefore carries its own literal __device__ transcription of each one
// (commented as such at each copy) — exactly as 02.01/02.05's d_-prefixed
// device copies do. reference_cpu.cpp's file header explains precisely
// which parts of THIS project are shared data-layout arithmetic (permitted)
// and which are the independently retyped algorithmic core (the octree
// level-build loop, the Huffman table build, the bit-packing encoder, and
// every CPU oracle) — read that header next.
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cstdint>   // uint8_t/uint32_t/uint64_t — exact-width integers everywhere below
#include <cmath>     // std::floor, std::sqrt — identical overloads to cl.exe and nvcc's host pass
#include <vector>    // reference_cpu.cpp's independent oracle outputs + main.cu's host buffers
#include <string>

// ===========================================================================
// Problem-scale constants — the numbers every stage and every CPU twin below
// must agree on bit-for-bit.
// ===========================================================================

// Repo-default block size (warp multiple; good occupancy on sm_75..sm_89).
constexpr int kThreadsPerBlock = 256;

// kSweepDepths — the rate-distortion study's four octree depths (bits/axis).
// D=8 -> leaf = cube_extent/256 (coarse, cheap); D=11 -> leaf = cube_extent/2048
// (fine, expensive) — THEORY.md "Numerical considerations" works the leaf-size
// arithmetic at this project's scene scale. kMaxDepth=11 fixes the widest
// Morton code at 3*11=33 bits, which is why every code below is uint64_t, not
// uint32_t (02.05's 30-bit/uint32_t code would silently truncate at D>10).
constexpr int kSweepDepths[4] = { 8, 9, 10, 11 };
constexpr int kNumSweepDepths = 4;
constexpr int kCanonicalDepthIndex = 2;      // kSweepDepths[2] == 10: the VERIFY-stage depth
constexpr int kMaxDepth = 11;

// Two-level Blelloch exclusive-scan block size — copied verbatim from 02.02
// (cited throughout kernels.cu): 256 threads, 2 elements/thread -> 512
// elements/block. Used for the STAGE-1 per-level boundary scan, whose input
// size is always exactly the POINT COUNT (fixed per cloud, never per-node),
// safely under the two-level scan's 512*512=262,144-element hard limit for
// both this project's committed clouds (200,000 points each) — see
// kernels.cu's launch_build_octree_level for the margin arithmetic and why
// STAGE 2's encode-length scan uses thrust::exclusive_scan instead (that
// array's size is the NODE count, which is data-dependent and can exceed the
// hand-rolled scan's bound on the pathological cohort at fine depths).
constexpr int kScanBlockThreads = 256;
constexpr int kScanElemsPerBlock = 512;

// blocks_for — integer ceiling division: how many `threads`-wide blocks
// cover `count` independent problems (the 02.01/02.02/02.05 idiom).
inline int blocks_for(int count, int threads)
{
    return (count + threads - 1) / threads;
}

// ===========================================================================
// STAGE 1 helpers — cube-padded scene AABB, quantization, Morton encoding,
// node/child bit arithmetic. Shared, plain-inline data-layout arithmetic
// (host+device compilable) — see the file header's independence discussion.
// ===========================================================================

// SceneAABB — a cloud's bounding box, PADDED TO A CUBE (equal extent on all
// three axes) so every leaf cell at every swept depth is a literal cube, not
// an axis-anisotropic box. This single choice is what makes the quantization
// distortion bound a ONE-LINE formula (leaf half-diagonal = leaf_m*sqrt(3)/2
// — THEORY.md "The math" derives it) instead of a three-axis mess. main.cu's
// compute_cube_aabb (host-only: run ONCE per cloud before any device work,
// so a GPU reduction kernel would add complexity for zero benefit at this
// project's problem sizes) pads with a small margin so no point sits exactly
// on the [0, 2^D) quantization boundary.
struct SceneAABB {
    float min[3];
    float max[3];
    float extent_m;   // == max[i]-min[i] for every axis i (the cube side length, meters)
};

// quantize_axis — map a world coordinate into [0, 2^D - 1] given the cube's
// extent. Clamped at both ends: a point exactly on the padded boundary still
// produces a valid, if saturated, grid coordinate (02.05's identical
// quantize_axis precedent and its identical "ground truth is always the real
// float xyz, never the quantized code" caveat apply here too — EXCEPT that
// in THIS project the quantized leaf center genuinely IS the reconstructed
// output, by design: this is a LOSSY geometry codec, and the whole point of
// the R-D sweep is to quantify that loss, not to explain it away).
inline uint32_t quantize_axis(float p, float lo, float extent_m, int D)
{
    float t = (extent_m > 0.0f) ? (p - lo) / extent_m : 0.0f;
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    const uint32_t max_cell = (1u << D) - 1u;
    uint32_t cell = static_cast<uint32_t>(t * static_cast<float>(1u << D));
    if (cell > max_cell) cell = max_cell;   // guard t==1.0's exact-boundary rounding
    return cell;
}

// morton_encode — LSB-first, 3-bits-per-iteration bit interleave: bit b of
// ix/iy/iz lands at code bit 3b+0 / 3b+1 / 3b+2 respectively (x=bit0,
// y=bit1, z=bit2 within each triple — the SAME axis-to-bit-position
// convention 02.05's expand_bits10/morton_encode30 uses, cited here, just
// written as a plain loop instead of the "insert two zero bits" magic-number
// SIMD trick: 02.05's trick is a fixed-width (10 bits/axis, 32-bit result)
// optimization; THIS project sweeps D across 8..11, and a loop parameterized
// by D is one formula instead of four hand-tuned magic-number variants — the
// "teaching beats cleverness" call (CLAUDE.md §1), with 02.05's faster
// fixed-width version cited as the production alternative for a FIXED D
// (THEORY.md "The GPU mapping" names the exact trade). Because bit b=D-1
// (each axis's MOST significant used bit — the COARSEST split) lands at the
// HIGHEST code bit position (3*(D-1)..+2), sorting codes ascending sorts by
// the COARSEST octant first — exactly the property node_prefix below relies
// on. D in [1,11]; result uses the low 3*D bits of a uint64_t (3*11=33 bits,
// comfortably inside 64).
inline uint64_t morton_encode(uint32_t ix, uint32_t iy, uint32_t iz, int D)
{
    uint64_t code = 0;
    for (int b = 0; b < D; ++b) {
        code |= (static_cast<uint64_t>((ix >> b) & 1u)) << (3 * b + 0);
        code |= (static_cast<uint64_t>((iy >> b) & 1u)) << (3 * b + 1);
        code |= (static_cast<uint64_t>((iz >> b) & 1u)) << (3 * b + 2);
    }
    return code;
}

// point_to_code — the full STAGE-1 map: a world-frame point -> its 3D-bit
// Morton code at depth D, normalized against `aabb`. Pure function of its
// inputs — the single formula every device/host copy below calls.
inline uint64_t point_to_code(float x, float y, float z, const SceneAABB& aabb, int D)
{
    const uint32_t ix = quantize_axis(x, aabb.min[0], aabb.extent_m, D);
    const uint32_t iy = quantize_axis(y, aabb.min[1], aabb.extent_m, D);
    const uint32_t iz = quantize_axis(z, aabb.min[2], aabb.extent_m, D);
    return morton_encode(ix, iy, iz, D);
}

// node_prefix — the top 3*level bits of `code`, i.e. the IDENTITY of the
// node this code's point belongs to at tree depth `level` (level=0 -> 0,
// the single root; level=D -> the code itself, i.e. leaf identity). Two
// points sharing this value for a given level share that level's node —
// THE property the whole level-by-level construction (kernels.cu) leans on.
inline uint64_t node_prefix(uint64_t code, int D, int level)
{
    return code >> (3 * (D - level));
}

// child_octant — the 3 bits that distinguish a level-`level` node's 8
// children (the NEXT 3 bits below node_prefix(code,D,level)'s bottom bit):
// bit0=x, bit1=y, bit2=z (morton_encode's convention). Valid for
// level in [0, D-1].
inline uint32_t child_octant(uint64_t code, int D, int level)
{
    return static_cast<uint32_t>((code >> (3 * (D - level - 1))) & 7ull);
}

// decode_leaf_center — the inverse of point_to_code, given a FULL depth-D
// code (i.e. node_prefix(code,D,D) == code): extract (ix,iy,iz) bit by bit
// and return the leaf cell's CENTER in world coordinates — the reconstructed
// point every decoded leaf produces. leaf_m (the cube's cell side length) is
// aabb.extent_m / 2^D; cited by main.cu's distortion-bound gate as the input
// to the analytic half-diagonal formula (THEORY.md "The math").
inline void decode_leaf_center(uint64_t code, int D, const SceneAABB& aabb,
                               float& x, float& y, float& z)
{
    uint32_t ix = 0, iy = 0, iz = 0;
    for (int b = 0; b < D; ++b) {
        ix |= static_cast<uint32_t>((code >> (3 * b + 0)) & 1ull) << b;
        iy |= static_cast<uint32_t>((code >> (3 * b + 1)) & 1ull) << b;
        iz |= static_cast<uint32_t>((code >> (3 * b + 2)) & 1ull) << b;
    }
    const float leaf_m = aabb.extent_m / static_cast<float>(1u << D);
    x = aabb.min[0] + (static_cast<float>(ix) + 0.5f) * leaf_m;
    y = aabb.min[1] + (static_cast<float>(iy) + 0.5f) * leaf_m;
    z = aabb.min[2] + (static_cast<float>(iz) + 0.5f) * leaf_m;
}

// ===========================================================================
// STAGE 2 data layout — the canonical Huffman table over the 256 possible
// occupancy-byte symbols.
// ===========================================================================

// HuffmanTable — per-symbol code LENGTH (bits; 0 = symbol never observed, no
// code assigned) and the code itself, RIGHT-JUSTIFIED in the low `len` bits
// of `bits`, read MSB-FIRST when packed (bit (len-1) of `bits` is the FIRST
// bit written to the stream — see kernels.cu's bit_scatter_kernel and
// reference_cpu.cpp's huffman_encode_cpu, which share this documented FORMAT
// convention, independently implemented on each side per the ruling). At
// most 255 of the 256 code LENGTHS can be nonzero in this project's data
// (occupancy bytes with value 0 — "this node has zero children" — can never
// occur: every internal node exists BECAUSE it has at least one occupied
// child, by construction; kernels.cu's build_huffman_table asserts this).
struct HuffmanTable {
    int      len[256];    // code length in bits; 0 = unused symbol
    uint32_t bits[256];   // the code, right-justified in the low len[s] bits
};

// huffman_merge_key — the DETERMINISTIC total order both independent table
// builders (build_huffman_table's heap and build_huffman_table_cpu's linear
// scan) use to pick "the two smallest remaining nodes" when merging: rank
// by frequency first, then by `tiebreak_id` (leaves are numbered by their
// own symbol value 0..255; internal nodes are numbered 256+creation_index —
// always >= 256, so a leaf ALWAYS wins a frequency tie against an internal
// node, and two internal nodes tie-break by creation order, i.e. "merged
// earlier wins"). This single shared formula is what makes two
// STRUCTURALLY DIFFERENT construction algorithms provably converge on the
// IDENTICAL code-length assignment (a data-layout-level total-order
// contract, not "the algorithm" — the same category as 02.05's knn_less).
// Packed into one uint64_t so both implementations compare with a single
// integer '<' — freq in the high 32 bits dominates the comparison, id
// breaks ties in the low 32 bits.
inline uint64_t huffman_merge_key(int freq, int tiebreak_id)
{
    return (static_cast<uint64_t>(static_cast<uint32_t>(freq)) << 32)
         | static_cast<uint32_t>(tiebreak_id);
}

// ===========================================================================
// GPU kernel declarations — nvcc-only (see the file header for why: cl.exe,
// compiling reference_cpu.cpp, has never heard of __global__).
// ===========================================================================
#ifdef __CUDACC__

// ---- Stage 1: Morton codes -------------------------------------------------

// compute_codes_kernel — one thread per point: quantize + Morton-encode.
// xyz [n*3] device floats (meters, map frame); aabb by value (tiny, 28
// bytes); D the depth for THIS call (the whole pipeline reruns per swept
// depth — see kernels.cu's per-level design note for why re-running from
// scratch, rather than incrementally refining, is the right trade here).
// out_codes [n] device uint64_t, in ORIGINAL point-index order (pre-sort).
__global__ void compute_codes_kernel(int n, const float* __restrict__ xyz,
                                     SceneAABB aabb, int D,
                                     uint64_t* __restrict__ out_codes);

// ---- Stage 1: per-level construction --------------------------------------

// compute_level_flags_kernel — one thread per SORTED point i: is_start[i]=1
// iff i is the FIRST sorted point under a new level-`level` node (i==0, or
// its node_prefix differs from point i-1's). This IS the "unique-prefix
// boundary" predicate the file header's STAGE 1 essay describes; its
// exclusive scan (below) turns these 0/1 flags into a per-point NODE INDEX
// at this level — the same "boundary flags -> scan -> per-element label"
// idiom 02.01 Method B / 02.02 use for compaction, repurposed here for
// LABELING instead of filtering (see kernels.cu for the full argument).
__global__ void compute_level_flags_kernel(int n, const uint64_t* __restrict__ sorted_codes,
                                           int D, int level,
                                           int* __restrict__ out_is_start);

// compute_occupancy_kernel — one thread per SORTED point i: this point's
// child octant bit, OR'd (atomicOr) into its OWNING node's occupancy word.
//
// Node index arithmetic (a one-line correction worth spelling out, because
// getting it wrong is a genuine, easy-to-make off-by-one — root-caused
// during this project's own development via GATE octree_levels catching a
// GPU/CPU mismatch): `scan_exclusive` is compute_level_flags_kernel's
// is_start array, exclusive-SCANNED (see launch_build_octree_level) — for
// a point that STARTS a new node (is_start[i]==1), scan_exclusive[i] IS
// already that new node's correct 0-based index (it counts how many
// earlier starts occurred, which is exactly this node's ordinal). But for
// a CONTINUATION point (is_start[i]==0, same node as its predecessor),
// scan_exclusive[i] over-counts by one relative to the node it actually
// belongs to — an EXCLUSIVE scan counts starts STRICTLY BEFORE position i,
// so at a continuation point it has already "seen" that point's own node's
// start (which occurred at some earlier index), one step earlier than the
// node's zero-based index would suggest by itself; the correct owning node
// is the INCLUSIVE count of starts at-or-before i, minus one:
//     node_id[i] = scan_exclusive[i] + is_start[i] - 1
// (both branches collapse to this one formula — no divergent code path
// needed). This is computed HERE, inline, from the two small arrays
// compute_level_flags_kernel and launch_scan_blelloch already produced —
// no extra kernel pass, since both operands are already resident.
// out_occupancy must be zeroed by the caller (cudaMemset) before this
// launch; sized >= num_nodes_at_this_level (always <= n).
//
// Why uint32_t and not uint8_t: CUDA's atomicOr has no native byte-granular
// overload (only 32-/64-bit words) — the production fix is a manual
// "read the containing aligned 32-bit word, shift the mask into the target
// byte's position, atomicOr the word" trick (THEORY.md "The GPU mapping"
// names it). This project takes the simpler, equally-correct teaching
// choice instead: one FULL uint32_t per node (only the low 8 bits are ever
// meaningful — a node has at most 8 children). At this project's node
// counts (well under a few million even in the pathological worst case)
// the 4x memory cost is negligible, and it removes an entire class of
// alignment bugs from a file whose real teaching point is the scan/atomic
// COMPOSITION, not sub-word bit tricks. main.cu narrows each finished
// level's word array down to a packed uint8_t stream on the host (a
// trivial O(num_nodes) cast loop) before it is ever used as the STAGE-2
// symbol stream — so nothing downstream (histogram, Huffman, encode) ever
// sees or cares about this choice.
__global__ void compute_occupancy_kernel(int n, const uint64_t* __restrict__ sorted_codes,
                                         int D, int level,
                                         const int* __restrict__ is_start,
                                         const int* __restrict__ scan_exclusive,
                                         uint32_t* __restrict__ out_occupancy);

// ---- The scan primitive (02.02 lineage, cited and reused verbatim) --------

// blelloch_block_scan_kernel / add_block_offsets_kernel — the identical
// two-level Blelloch exclusive prefix scan 02.02 built for stream
// compaction (see that project's kernels.cu for the full up-sweep/down-
// sweep derivation and the bank-conflict honesty note — not re-derived
// here). Declared so launch_scan_blelloch below can call them; no other
// file needs the raw kernels.
__global__ void blelloch_block_scan_kernel(int n, const int* __restrict__ in,
                                           int* __restrict__ out_exclusive,
                                           int* __restrict__ block_sums);

__global__ void add_block_offsets_kernel(int n, int* __restrict__ out_exclusive,
                                         const int* __restrict__ scanned_block_sums);

// ---- Stage 2: histogram + GPU encode ---------------------------------------

// compute_histogram_kernel — one thread per SYMBOL i in [0, m): atomicAdd
// into out_hist[symbol]. out_hist [256] must be zeroed by the caller first.
__global__ void compute_histogram_kernel(int m, const uint8_t* __restrict__ symbols,
                                         int* __restrict__ out_hist);

// compute_code_lengths_kernel — the STAGE-2 per-symbol MAP: look up this
// symbol's Huffman code length from the __constant__ table
// launch_huffman_encode uploads (d_huff_len in kernels.cu) before this
// launch. out_len [m] feeds the exclusive scan that turns lengths into bit
// offsets (see the file header's STAGE 2 essay).
__global__ void compute_code_lengths_kernel(int m, const uint8_t* __restrict__ symbols,
                                            int* __restrict__ out_len);

// bit_scatter_kernel — one thread per SYMBOL i: writes its Huffman code's
// `len[i]` bits, MSB-first, starting at global bit index bit_offset[i]
// (the STAGE-2 scan's output), into the packed output byte array via
// per-bit atomicOr (see kernels.cu for why bit-level atomics, not a
// byte-level write, are the correct primitive here — different symbols'
// codes routinely share one output byte, but never the same BIT).
// out_packed must be zeroed (cudaMemset) and sized >= ceil(total_bits/8)
// bytes by the caller before this launch.
__global__ void bit_scatter_kernel(int m, const uint8_t* __restrict__ symbols,
                                   const int* __restrict__ bit_offset,
                                   uint8_t* __restrict__ out_packed);

#endif // __CUDACC__

// ===========================================================================
// Host-callable launch wrappers (definitions in kernels.cu, which only nvcc
// compiles — but the DECLARATIONS below are plain C++, visible to main.cu).
// ===========================================================================

void launch_compute_codes(int n, const float* d_xyz, const SceneAABB& aabb, int D,
                          uint64_t* d_codes);

// launch_sort_codes — Stage 2: thrust::sort ascending, IN PLACE (a GPU
// radix/merge sort over 64-bit unsigned keys — kernels.cu's call site names
// exactly what it computes and why no paired "value" array is needed: this
// codec never needs to recover which ORIGINAL point produced a given code,
// only the code's multiset, so an unstable sort is correct and sufficient).
void launch_sort_codes(int n, uint64_t* d_codes_inout);

// launch_scan_blelloch — the two-level exclusive prefix scan (02.02
// lineage, cited). See kernels.cuh's kScanElemsPerBlock comment for the
// 262,144-element bound and why STAGE 2's encode-length scan uses
// launch_scan_thrust instead.
void launch_scan_blelloch(int n, const int* d_in, int* d_out_exclusive);

// launch_scan_thrust — thrust::exclusive_scan: the SAME primitive, no size
// bound, used for STAGE 2's data-dependent-length code-length scan (see
// kernels.cuh's kScanElemsPerBlock comment for why THIS scan needs it and
// STAGE 1's does not).
void launch_scan_thrust(int n, const int* d_in, int* d_out_exclusive);

// OctreeLevelResult — one level's build output: how many nodes exist at
// this level (always in [1, n]) and that many occupancy words.
struct OctreeLevelResult {
    int num_nodes;
};

// launch_build_octree_level — STAGE 1's per-level construction end to end:
// flags -> scan -> node_id -> occupancy (see kernels.cuh's kernel
// declarations above for each stage). d_sorted_codes [n] is this cloud's
// depth-D Morton-sorted code array (read-only, shared across every level's
// call at this depth). d_is_start_scratch/d_node_id_scratch are caller-
// owned [n]-sized int scratch (reused across levels — no per-level alloc).
// d_occupancy_out is caller-owned, sized >= n uint32_t words (always
// enough: num_nodes <= n — see kernels.cuh's compute_occupancy_kernel
// comment for why uint32_t, not uint8_t). Returns the node count.
OctreeLevelResult launch_build_octree_level(int n, const uint64_t* d_sorted_codes,
                                            int D, int level,
                                            int* d_is_start_scratch, int* d_node_id_scratch,
                                            uint32_t* d_occupancy_out);

void launch_compute_histogram(int m, const uint8_t* d_symbols, int* d_hist_out /* [256] */);

// build_huffman_table — the OPERATIONAL canonical-Huffman table builder
// (host-side, small/serial by nature — no GPU pattern here, per the file
// header's STAGE 2 essay). Heap-based: classic "repeatedly merge the two
// smallest frequencies" construction, THEN canonical-ized (code lengths
// re-sorted by (length, symbol) and codes reassigned in that order — the
// deterministic step that makes two independent implementations agree
// bit-for-bit, "canonical form makes it deterministic"). Defined in
// kernels.cu; reference_cpu.cpp's build_huffman_table_cpu is an
// INDEPENDENTLY WRITTEN twin (a different selection strategy, same
// algorithm) that main.cu's VERIFY stage compares against this one,
// symbol-for-symbol, exact.
void build_huffman_table(const int hist[256], HuffmanTable& out);

// EncodeResult — the packed bitstream plus its exact bit length (the byte
// array is padded with zero bits at the end; num_bits is the AUTHORITATIVE
// length the decoder must use — trailing pad bits are not meaningful data).
struct EncodeResult {
    std::vector<uint8_t> packed;
    long long num_bits;
};

// launch_huffman_encode — STAGE 2's GPU encode end to end: code-length map
// -> thrust exclusive scan -> bit-scatter (see kernels.cuh's kernel
// declarations for each stage; kernels.cu's launcher owns uploading `table`
// to the __constant__ device table and every scratch allocation). d_symbols
// [m] is this depth's concatenated occupancy-byte stream (all levels, in
// level-then-node order — see launch_build_octree_level's call sequence in
// main.cu). Copies the result back to host (m is at most a few hundred
// thousand bytes at this project's scale — a host round trip is negligible
// next to the kernels it measures).
EncodeResult launch_huffman_encode(int m, const uint8_t* d_symbols, const HuffmanTable& table);

// ===========================================================================
// CPU references (reference_cpu.cpp) — the correctness-oracle twins AND the
// project's only decoder. See reference_cpu.cpp's file header for the
// independence ruling each of these follows.
// ===========================================================================

// compute_codes_cpu — the twin of compute_codes_kernel (shared
// point_to_code formula, per the ruling — VERIFY(morton) compares GPU vs
// CPU point-for-point, exact).
void compute_codes_cpu(int n, const float* xyz, const SceneAABB& aabb, int D,
                       uint64_t* codes_out);

// sort_codes_cpu — std::sort (algorithmically different from Thrust's GPU
// sort) on a COPY of the code array. VERIFY(sort) compares the full sorted
// array element-wise, exact (codes need not be pairwise distinct — repeated
// leaf occupancy is routine and harmless; sort correctness only requires a
// total order on possibly-equal keys, which std::sort/thrust::sort both give).
void sort_codes_cpu(int n, const uint64_t* codes_in, uint64_t* codes_sorted_out);

// build_octree_levels_cpu — a GENUINELY INDEPENDENT re-implementation of
// the whole STAGE-1 level-by-level construction: ONE sequential pass over
// the sorted codes, tracking (per level) the CURRENT node's prefix and
// occupancy byte with a plain running accumulator — no scan, no atomics, a
// structurally different algorithm from the GPU's flags/scan/atomicOr
// pipeline (per the independence ruling). Fills occ_per_level[level] with
// that level's occupancy bytes, in the SAME node order the GPU path
// produces (ascending sorted-prefix order — see kernels.cu for why this
// order is forced by construction, not a coincidence). VERIFY(octree_levels)
// compares node counts AND occupancy bytes, level by level, exact.
void build_octree_levels_cpu(int n, const uint64_t* sorted_codes, int D,
                             std::vector<std::vector<uint8_t>>& occ_per_level);

// compute_histogram_cpu — the twin of compute_histogram_kernel (a plain
// serial pass, hist[symbol]++). VERIFY(histogram) compares all 256 bins, exact.
void compute_histogram_cpu(int m, const uint8_t* symbols, int hist_out[256]);

// build_huffman_table_cpu — the INDEPENDENT twin of build_huffman_table:
// same canonical-Huffman algorithm, a DIFFERENT selection strategy (a plain
// O(256^2) "scan for the two smallest remaining frequencies" loop instead
// of a binary heap — a genuinely different implementation shape, apt at
// N=256 symbols). VERIFY(huffman_table) compares len[]/bits[] for all 256
// symbols, exact — "canonical form makes it deterministic" is the claim
// this gate proves, not just asserts.
void build_huffman_table_cpu(const int hist[256], HuffmanTable& out);

// huffman_encode_cpu — the INDEPENDENT twin of launch_huffman_encode: a
// plain SEQUENTIAL bit writer (accumulate bits into a running byte, no
// scan, no parallelism-shaped anything) — a structurally different
// algorithm from the GPU's map+scan+scatter, sharing only the documented
// MSB-first bit-packing FORMAT (kernels.cuh's HuffmanTable comment).
// VERIFY(encode_bitstream) compares the packed bytes AND num_bits, exact.
void huffman_encode_cpu(int m, const uint8_t* symbols, const HuffmanTable& table,
                        std::vector<uint8_t>& out_packed, long long& out_num_bits);

// ---------------------------------------------------------------------------
// The decoder — THE ONLY implementation (no GPU counterpart; see the file
// header's "Why decode is host-only"). Plain sequential C++.
// ---------------------------------------------------------------------------

// huffman_decode_cpu — inverse of huffman_encode_cpu / launch_huffman_encode:
// walk a bit-trie built from `table` bit by bit, emitting one symbol per
// leaf reached, until exactly m_symbols have been produced. Documented
// SERIAL by necessity (file header): symbol i's start bit is not knowable
// without having decoded symbol i-1's length first.
void huffman_decode_cpu(const std::vector<uint8_t>& packed, long long num_bits,
                        int m_symbols, const HuffmanTable& table,
                        std::vector<uint8_t>& out_symbols);

// octree_decode_leaf_centers — inverse of the STAGE-1 encode: given the
// FULL concatenated occupancy-byte stream (all levels, level-then-node
// order — the SAME order build_octree_levels_cpu/launch_build_octree_level
// produce) and depth D, replay the level-by-level EXPANSION (root's byte
// tells you level-1's occupied octants; each node's byte tells you its own
// children; a node's byte count at level l+1 is exactly the sum of set
// bits — popcount — over every level-l byte, so the decoder never needs an
// explicit "how many nodes at this level" side channel) to reconstruct
// every LEAF's (ix,iy,iz) path and its world-frame CENTER
// (decode_leaf_center). Output order is ascending Morton order — the same
// order octree_unique_leaf_codes below produces the GROUND TRUTH in, which
// is what makes GATE lossless_roundtrip an exact, order-sensitive
// comparison instead of a weaker set-equality check.
void octree_decode_leaf_centers(const std::vector<uint8_t>& occupancy_stream, int D,
                                const SceneAABB& aabb,
                                std::vector<float>& out_leaf_xyz);

// octree_unique_leaf_codes — the GROUND TRUTH decode target: the sorted,
// DEDUPLICATED depth-D codes (i.e. node_prefix(code,D,D) run through a
// "collapse consecutive duplicates" pass over the already-sorted array) —
// exactly the leaf set octree_decode_leaf_centers should reproduce, in the
// same order. A trivial O(n) pass, deliberately NOT sharing any code with
// the decoder above (the independent ground truth a decode bug cannot also
// corrupt, mirroring 02.05's brute-force-oracle tier).
void octree_unique_leaf_codes(int n, const uint64_t* sorted_codes, int D,
                              std::vector<uint64_t>& out_unique_leaf_codes);

#endif // PROJECT_KERNELS_CUH
