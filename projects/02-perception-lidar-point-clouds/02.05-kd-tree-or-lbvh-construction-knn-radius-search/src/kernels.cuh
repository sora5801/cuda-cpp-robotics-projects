// ===========================================================================
// kernels.cuh — interface for project 02.05
//               KD-tree or LBVH construction + KNN/radius search on GPU
//               (LBVH: a linear BVH built PARALLEL-FROM-SCRATCH every scan
//                via Morton codes + Karras's radix-tree construction, then
//                used for BOTH radius search and K-nearest-neighbor (KNN)
//                queries; contrasted honestly against the domain's existing
//                fixed-radius voxel-hash technique from 02.01/02.04.)
//
// Role in the project
// -------------------
// The CONTRACT shared by main.cu (load data, orchestrate every build/query
// stage, run every VERIFY/GATE, write artifacts), kernels.cu (the GPU
// kernels), and reference_cpu.cpp (the independent CPU oracle twins).
// Every data-layout decision all three must agree on — the point-cloud
// layout, the Morton/augmented-key packing, the LBVH node layout, the
// voxel-hash layout, the KNN heap ordering, the query result-buffer
// layout — is defined HERE, once (CLAUDE.md §12).
//
// POINT CLOUD LAYOUT — float* xyz, interleaved, meters, LiDAR "sensor" frame
// (origin at the sensor, +x forward, +y left, +z up), IDENTICAL to 02.01's
// convention: xyz[i*3+0..2] = x,y,z. QUERY POINTS use the exact same layout
// and live in the exact same frame — a query is just another xyz triple; it
// need not (and in this project's grid queries, usually does not) coincide
// with any point already in the cloud.
//
// THE FOUR-STAGE PIPELINE (this project's teaching spine; every stage below
// is built once, here, and consumed by main.cu in this exact order — see
// THEORY.md "The algorithm" for the full derivation of each step):
//
//   STAGE 1 — MORTON CODES. Each point's position is normalized into the
//   scene's bounding box and quantized to a 10-bit-per-axis integer grid,
//   then the three 10-bit integers are BIT-INTERLEAVED ("Z-order"/Morton
//   encoding) into one 30-bit code (morton_encode30 below). Sorting points
//   by this code approximates a spatial sort: two points close in Morton
//   order are (with the well-known exceptions THEORY.md discusses) usually
//   close in 3-D space too — the property the whole LBVH construction
//   leans on. DUPLICATE CODES (two points quantizing to the same 30-bit
//   cell — routine at this resolution over ~200k points) are handled by
//   AUGMENTING each 30-bit code with the point's own 32-bit ORIGINAL INDEX
//   in the low bits of a 64-bit key (augmented_key below) — the standard
//   fix named in Karras's original paper. Because every point's index is
//   unique, augmented keys are ALWAYS pairwise distinct, which removes an
//   entire class of degenerate-range special cases from Stage 3 for free
//   (see build_radix_tree's header comment in kernels.cu).
//
//   STAGE 2 — SORT. thrust::sort (GPU) / std::sort (CPU) on the augmented
//   64-bit key array, ascending. Because the key's low 32 bits already ARE
//   the original point index, the sorted key array doubles as the sorted
//   PERMUTATION — no separate paired "value" array is needed (a small but
//   real simplification versus 02.01/02.04's separate key+idx sort_by_key,
//   made possible because THIS project's key is already unique — worth
//   noticing as a design contrast, not a superior technique in general).
//
//   STAGE 3 — RADIX TREE (Karras 2012; THEORY.md "The math" proves the key
//   lemma). N points produce EXACTLY N-1 internal nodes and N leaves, laid
//   out in one flat array of 2N-1 LbvhNode's (LbvhNode layout below).
//   THE central idea, and the whole reason this construction is a genuinely
//   NEW GPU-programming pattern versus every top-down build in this repo
//   (11.01's median-split BVH, cited and contrasted throughout): internal
//   node i's range [first_i, last_i] over the SORTED array, and therefore
//   its two children, can be computed FROM THE SORTED KEY ARRAY ALONE, with
//   NO information from any other internal node — every one of the N-1
//   internal nodes is built by an INDEPENDENT thread, in ONE parallel pass,
//   with no barriers, no recursion, and no data dependency between threads
//   (kernels.cu's build_radix_tree_kernel walks through exactly how).
//
//   STAGE 4 — BOTTOM-UP AABB PROPAGATION. Once the tree's SHAPE exists,
//   every node's bounding box is the union of its subtree's points. Leaves
//   get a trivial single-point AABB; internal nodes climb from the leaves
//   upward via an ATOMIC-FLAG race: each of the N leaf threads walks its
//   own parent chain, and at each internal node atomically increments a
//   per-node counter — the thread that arrives FIRST (this node's sibling
//   subtree is not finished yet) stops; the thread that arrives SECOND (its
//   sibling just finished) is now guaranteed both children are ready, so it
//   computes this node's AABB and continues climbing. Exactly N-1 internal
//   nodes get visited by exactly one "second arrival" each — see
//   propagate_aabb_kernel's header comment in kernels.cu for the full
//   argument, and 02.01's hash_insert_kernel / 02.04's union-find atomicCAS
//   sections for this repo's other worked examples of "coordinate parallel
//   threads through a shared data structure via one atomic primitive".
//
// CONTRAST WITH 11.01's HAND-BUILT BVH (read that project's kernels.cuh
// header first if you have not — it is this project's closest sibling):
//   * 11.01 builds TOP-DOWN, ON THE HOST, ONCE, recursively: pick the
//     largest AABB axis, split triangles at the MEDIAN (by count), recurse
//     into two children, repeat. Depth is PROVABLY <= ceil(log2(N/leafsize))
//     because every split is EXACTLY balanced by construction.
//   * THIS project builds BOTTOM-UP-TOPOLOGY (well, the topology recipe
//     is computed per-node independently, not by top-down recursion at
//     all), ON THE DEVICE, IN PARALLEL, from a SORTED ARRAY: no recursion,
//     no host involvement, and (this is the honest cost) NO exact balance
//     guarantee — an internal node's range length depends on how many
//     CONSECUTIVE points in sorted order share a long common Morton prefix,
//     which depends on the DATA, not a count invariant. THEORY.md "The GPU
//     mapping" gives this project's own depth bound instead (a BIT-COUNT
//     argument, not a balance argument): the whole construction is exactly
//     a binary RADIX TRIE over the augmented keys, and a trie over B-bit
//     keys has depth <= B, because each level consumes a DIFFERENT,
//     strictly-later bit position (no bit position can gate a branch
//     twice on one root-to-leaf path). The augmented key is 64 bits wide,
//     but its TOP 2 bits (63 and 62) are structurally always 0 (the
//     30-bit Morton code occupies bits 61..32 of the high half — see
//     augmented_key below), so they can NEVER differ between two keys and
//     can never gate a branch: only the remaining 62 bits (61..0) can —
//     so root-to-leaf depth can never exceed 62. kBvhStackSize below is
//     set to 64: 2 nodes of pure headroom over the proven bound, the same
//     "prove it, then add a defensive margin anyway" discipline 11.01
//     applies to ITS depth bound (a median-split BALANCE argument there,
//     a bit-trie argument here — a coincidence that both land near the
//     same small numeral, not a copied constant).
//   * Both projects use the SAME traversal shape once the tree exists: a
//     small fixed-size per-thread stack, push-both-children, prune by an
//     AABB test (sphere-overlap here instead of ray-slab there) — see
//     radius_search_bvh_kernel / knn_search_bvh_kernel in kernels.cu, and
//     11.01's intersect_bvh for the ray-casting analogue.
//
// THE DOMAIN CONTRAST — LBVH vs. the voxel-hash technique 02.01/02.04
// already teach (cited, reimplemented compactly below, NOT re-derived):
//   02.01's Method B (sort-by-voxel-key + segmented reduction) and 02.04's
//   27-cell-stencil-plus-binary-search neighbor query are both built on the
//   SAME "sort points into a spatial hash, walk a small neighborhood" idea
//   — cheap, simple, and O(1) EXPECTED per query, but ONLY for a query
//   shape that matches the hash's fixed cell size (a FIXED search radius
//   r, with leaf = r; see kClusterToleranceM's proof in 02.04, reused
//   verbatim below for launch_hash_radius_search's stencil correctness).
//   It has no answer at all for K-NEAREST-NEIGHBOR queries (which radius to
//   search? THEORY.md "The problem" derives WHY that question has no fixed
//   answer for real LiDAR data — point density falls off as 1/r^2 with
//   range) and it gets systematically SLOWER at large or spatially-VARYING
//   radii (more cells to stencil-scan, more of them empty). The LBVH pays
//   a higher one-time build cost and a per-query traversal with real (if
//   shallow) divergence, but adapts to ANY query shape — including KNN,
//   which this project's kernels.cu "knn_search_bvh_kernel" implements via
//   a bounded max-heap with SHRINKING-RADIUS pruning (a genuinely new
//   traversal idea, not just radius search with an unbounded r). main.cu's
//   density_contrast gate measures both techniques on the SAME two extreme
//   regions of the SAME point cloud and reports the honest, sometimes
//   uncomfortable, numbers (CLAUDE.md "never fabricate a benchmark claim").
//
// Why this header is CUDA-qualifier-free where possible (02.01/02.04's
// precedent, reused verbatim): pure math/bit-arithmetic helpers below
// (morton/augmented-key packing, clz64_portable, squared_distance3,
// aabb_min_dist2, heap comparison) are PLAIN inline C++ — no
// __host__/__device__ — so they compile under BOTH nvcc (main.cu,
// kernels.cu's host-side code) and cl.exe (reference_cpu.cpp). Being
// unqualified, they are HOST-only under nvcc's rules and cannot be called
// from a __global__ kernel; kernels.cu therefore carries its own literal
// __device__ transcription of each one (commented as such at each copy),
// exactly as 02.01's d_voxel_coord/d_pack_voxel_key do for that project's
// shared arithmetic. The independence ruling in reference_cpu.cpp's file
// header explains precisely which parts of THIS project are shared
// data-layout arithmetic (permitted) and which are the independently
// retyped algorithmic core (the radix-tree range/split search, the BVH/
// hash traversal loops, and every CPU oracle) — read that header next.
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cstdint>       // int32_t, uint32_t, uint64_t — exact-width integers everywhere below
#include <cmath>         // std::floor, std::sqrt — identical overloads to cl.exe and nvcc's host pass
#include <vector>        // reference_cpu.cpp's independent oracle outputs
#include <unordered_map> // reference_cpu.cpp's independent voxel-hash oracle (02.04 hashmap-oracle lineage)
#include <algorithm>     // std::sort (reference_cpu.cpp's independent sort twin)

// ===========================================================================
// Problem-scale constants — the numbers every stage and every CPU twin below
// must agree on bit-for-bit (a mismatch here is a silent, hard-to-find bug
// class of its own, which is exactly why they live in ONE shared header).
// ===========================================================================

// Repo-default block size (warp multiple; good occupancy on sm_75..sm_89 —
// see kernels.cu's per-kernel launch-configuration comments).
constexpr int kThreadsPerBlock = 256;

// kQueryK — the "K" in K-nearest-neighbor. 8 is a common real-world default
// (PCL's KdTreeFLANN tutorials, Open3D's KNN examples, and normal/curvature
// estimation in this repo's own 02.09 all use K in the 6-20 range for local
// surface fitting) — small enough to keep the per-query heap cheap, large
// enough to give a real local neighborhood a plane/curvature estimator (or
// FPFH, 02.10) could use. README "System context" names both consumers.
constexpr int kQueryK = 8;

// kRadiusM — the FIXED search radius used by BOTH the BVH radius-search
// path and the voxel-hash baseline (so the "hash vs BVH" contrast in
// main.cu's density_contrast/GATE hash_vs_bvh_agreement is a fair,
// apples-to-apples comparison at the SAME r). 0.5 m is a realistic local-
// neighborhood radius for the LiDAR point density this project's synthetic
// scan produces (see scripts/make_synthetic.py and 02.01's cited beam
// model) — comparable to 02.04's kClusterToleranceM (0.40 m) but distinct
// on purpose (a different project, tuned for ITS OWN adversarial regions;
// see scripts/make_synthetic.py's module docstring for the exact tuning).
constexpr float kRadiusM = 0.5f;

// kMortonBitsPerAxis — 10 bits/axis * 3 axes = 30-bit interleaved Morton
// code (the classic width for a 32-bit-friendly bit-interleave — see
// expand_bits10 below). THEORY.md "Numerical considerations" does the
// quantization-resolution arithmetic at this project's scene scale (a
// ~20 m room -> ~2 cm per grid cell along each axis at 10 bits: fine
// enough that Morton-adjacent points are genuinely close, coarse enough
// that the whole scene fits one interleave pass with no per-axis loop).
constexpr int kMortonBitsPerAxis = 10;
constexpr uint32_t kMortonAxisMax = (1u << kMortonBitsPerAxis) - 1u;  // 1023: max quantized per-axis coordinate

// kBvhStackSize — the traversal stack depth bound. See the file header's
// "CONTRAST WITH 11.01" section for the full bit-trie proof: only 62 of the
// augmented key's 64 bits can ever differ between two distinct points (the
// top 2 bits, holding nothing but zero-padding above the 30-bit Morton
// code, never do) — so root-to-leaf depth is bounded by 62. 64 gives 2
// nodes of pure headroom, purely defensive (the traversal kernels also
// carry an explicit "would overflow -> stop pushing" guard, 11.01's
// identical defensive-but-should-never-fire discipline).
constexpr int kBvhStackSize = 64;

// kMaxRadiusResults — the per-query output-buffer capacity for radius
// search (both BVH and hash paths share this cap, again for a fair
// comparison). Sized against this project's own worst case: the synthetic
// scene's adversarial DENSE cluster (scripts/make_synthetic.py) places at
// most a few thousand points inside one small cube — 8192 is a documented,
// generously-margined ceiling (not a universal constant), and every
// traversal kernel below DETECTS an overflow (never silently truncates) via
// a per-query overflow flag main.cu gates on being all-zero — the same
// "size to a documented worst case, then detect and report overflow"
// discipline 02.01's hash-table capacity and 02.04's kMaxEdgesPerPoint use.
constexpr int kMaxRadiusResults = 8192;

// blocks_for — integer ceiling division: how many `threads`-wide blocks
// cover `count` independent problems (the 02.01/02.04/08.01 idiom).
inline int blocks_for(int count, int threads)
{
    return (count + threads - 1) / threads;
}

// ===========================================================================
// STAGE 1 helpers — Morton encoding + the augmented (index-tie-broken) key.
// Shared, plain-inline data-layout arithmetic (host+device compilable) —
// see the file header's independence discussion for why this class of
// function is permitted to be single-sourced.
// ===========================================================================

// SceneAABB — the point cloud's axis-aligned bounding box (computed ONCE on
// the host from the full point set, THEN used to quantize both the point
// cloud and every query into the same [0,1]^3 normalized cube). Padded by
// main.cu with a small margin (see its computation) so a query point that
// falls slightly outside the point cloud's own extent — routine for the
// grid queries scripts/make_synthetic.py places near room walls — still
// quantizes into a valid [0,1023] cell instead of clamping to the boundary
// cell for every such query (which would distort the BVH's own root AABB
// test, not just the Morton code).
struct SceneAABB {
    float min[3];
    float max[3];
};

// quantize_axis — map a world coordinate into [0, kMortonAxisMax] given the
// scene's extent on that axis. Clamped at both ends: a query point outside
// the (slightly padded) SceneAABB still produces a valid, if saturated,
// grid coordinate — Morton quantization is only ever used to APPROXIMATE
// spatial locality for the tree build and is never the source of ground
// truth (every distance test downstream uses the point's real float xyz,
// never its quantized code) — so saturating out-of-range coordinates costs
// nothing but a slightly less selective code for that one point.
inline uint32_t quantize_axis(float p, float lo, float hi)
{
    float t = (hi > lo) ? (p - lo) / (hi - lo) : 0.0f;  // normalize to ~[0,1]; guard a degenerate zero-extent axis
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    return static_cast<uint32_t>(t * static_cast<float>(kMortonAxisMax));
}

// expand_bits10 — the classic "magic number" bit-interleave trick: given a
// 10-bit value (bits b9..b0), spread it out so TWO ZERO BITS separate every
// original bit: result bit layout (MSB..LSB) = b9 0 0 b8 0 0 ... b0 0 0
// (30 bits used of the 32-bit word). Three axes' expanded codes, shifted by
// 0/1/2 bits respectively and OR'd together, interleave into one Morton
// code where bit 3k+axis is bit k of that axis's coordinate — the standard
// construction (THEORY.md "The math" derives WHY this ordering makes
// sorted-order neighbors spatially close: it recurses the same way an
// octree does, most-significant-bit-first).
// Reference: this is the well-known 5-step SIMD-within-a-register spread
// (see e.g. the "Insert two zero bits" technique catalogued for decades in
// Morton-code implementations; THEORY.md cites the bit-trie argument this
// project's depth bound relies on).
inline uint32_t expand_bits10(uint32_t v)
{
    v &= 0x000003FFu;                          // keep only the low 10 bits
    v = (v | (v << 16)) & 0x030000FFu;
    v = (v | (v << 8))  & 0x0300F00Fu;
    v = (v | (v << 4))  & 0x030C30C3u;
    v = (v | (v << 2))  & 0x09249249u;
    return v;
}

// morton_encode30 — the full Stage-1 map: a world-frame point -> its 30-bit
// interleaved Morton code, normalized against `aabb`. Pure function of its
// inputs (no state) — the single formula every device/host copy below
// calls (device copies transcribe it verbatim; see kernels.cu).
inline uint32_t morton_encode30(float x, float y, float z, const SceneAABB& aabb)
{
    const uint32_t xi = quantize_axis(x, aabb.min[0], aabb.max[0]);
    const uint32_t yi = quantize_axis(y, aabb.min[1], aabb.max[1]);
    const uint32_t zi = quantize_axis(z, aabb.min[2], aabb.max[2]);
    return expand_bits10(xi) | (expand_bits10(yi) << 1) | (expand_bits10(zi) << 2);
}

// augmented_key — pack a 30-bit Morton code and the point's ORIGINAL index
// into one 64-bit key: morton in the HIGH 32 bits (so sorting by this key
// sorts by Morton code first, exactly as intended), original point index
// in the LOW 32 bits as a DETERMINISTIC TIE-BREAK for points that quantize
// to the identical Morton cell (routine at 30-bit resolution over ~200k
// points — see THEORY.md "Numerical considerations" for a measured
// collision count on this project's own data). Because every point's index
// is unique, augmented keys are ALWAYS pairwise distinct — this is what
// removes the "firstCode==lastCode" degenerate case from Karras's original
// algorithm (kernels.cu's build_radix_tree_kernel comment explains exactly
// where that simplification pays off) and what makes STAGE 2's sort need
// no separate stability guarantee: a total order on a set of UNIQUE keys
// has exactly one valid result, so std::sort (unstable) and thrust::sort
// (unstable) are both correct, and IDENTICAL, without invoking any
// "stable" variant at all.
inline unsigned long long augmented_key(uint32_t morton, int32_t original_index)
{
    return (static_cast<unsigned long long>(morton) << 32) |
           static_cast<unsigned long long>(static_cast<uint32_t>(original_index));
}

// decode_point_index — the inverse of augmented_key's low half: recover the
// original point index from a sorted augmented key. Used everywhere the
// tree needs to know "which original point does sorted position k hold?"
inline int32_t decode_point_index(unsigned long long key)
{
    return static_cast<int32_t>(key & 0xFFFFFFFFull);
}

// clz64_portable — count LEADING ZERO bits of a 64-bit value, portable
// across cl.exe (reference_cpu.cpp; MSVC has no __builtin_clzll) and nvcc's
// HOST pass (kernels.cu's device code uses the hardware __clzll intrinsic
// instead — see d_clz64 in kernels.cu — for speed; this portable version
// exists purely so the CPU twin needs no compiler-specific intrinsic).
// Classic binary-search-the-leading-1-bit construction: repeatedly halve
// the search range, comparing against the value with that many low bits
// all set to 1 (i.e., "is the top half all zero?"). x==0 is defined here
// as 64 leading zeros (matching __clzll's documented behavior for 0),
// though delta_lcp below never actually calls this with x==0 (augmented
// keys are always pairwise distinct, so their XOR is always nonzero).
inline int clz64_portable(unsigned long long x)
{
    if (x == 0ull) return 64;
    int n = 0;
    if (x <= 0x00000000FFFFFFFFull) { n += 32; x <<= 32; }
    if (x <= 0x0000FFFFFFFFFFFFull) { n += 16; x <<= 16; }
    if (x <= 0x00FFFFFFFFFFFFFFull) { n += 8;  x <<= 8;  }
    if (x <= 0x0FFFFFFFFFFFFFFFull) { n += 4;  x <<= 4;  }
    if (x <= 0x3FFFFFFFFFFFFFFFull) { n += 2;  x <<= 2;  }
    if (x <= 0x7FFFFFFFFFFFFFFFull) { n += 1; }
    return n;
}

// delta_lcp — Karras's "delta" function: the length, in bits, of the
// longest common PREFIX shared by the augmented keys at SORTED positions i
// and j. Returns -1 (Karras's "-infinity" sentinel) if j falls outside
// [0,n) — the construction algorithm relies on this exact sentinel value
// to make the array boundary behave like an edge with "nothing in common"
// (kernels.cu's build_radix_tree_kernel comment walks through where each
// call site depends on this). Because augmented keys are pairwise distinct
// (see augmented_key above), i==j is the ONLY way the XOR below can be
// zero, and delta_lcp is never called with i==j by the construction
// algorithm — the guard exists purely for defensive clarity.
//
// This helper is deliberately SHARED (unlike the range/split SEARCH LOOPS
// that call it, which are independently retyped in kernels.cu vs.
// reference_cpu.cpp — see reference_cpu.cpp's file header for the ruling):
// it is pure bit arithmetic over a data layout already single-sourced
// above, exactly the same category as 02.01's pack_voxel_key.
inline int delta_lcp(const unsigned long long* sorted_key, int n, int i, int j)
{
    if (j < 0 || j >= n) return -1;
    return clz64_portable(sorted_key[i] ^ sorted_key[j]);
}

// ===========================================================================
// STAGE 3/4 data layout — the flattened LBVH node array.
//
// n points -> EXACTLY n-1 internal nodes (indices [0, n-2]) and n leaves
// (indices [n-1, 2n-2]), for 2n-1 nodes total, following the standard LBVH
// convention (Karras 2012; popularized as "Thinking Parallel, Part III:
// Tree Construction on the GPU", NVIDIA Developer Blog — README "Prior art"
// cites both). Leaf k (array index n-1+k) holds the point at SORTED
// position k, i.e. original point index decode_point_index(sorted_key[k]).
// Node 0 is ALWAYS the root: internal node 0's range, by construction,
// covers the WHOLE sorted array [0, n-1] (see build_radix_tree_kernel), so
// it is never assigned as a CHILD of any other node — the same "root is
// always node 0" convention 11.01's top-down builder also adopts, for an
// unrelated reason (there, it is simply the first node allocated).
// ===========================================================================
struct LbvhNode {
    float aabb_min[3];   // this node's subtree bounding box, meters, world/sensor frame
    float aabb_max[3];
    int   left;          // internal node: index of the LEFT child (leaf or internal). Leaf: -1 (unused).
    int   right;         // internal node: index of the RIGHT child. Leaf: -1 (unused).
    int   parent;        // index of this node's parent; -1 ONLY for the root (node 0).
    int   point_idx;      // leaf only: the ORIGINAL point index this leaf represents. Internal: -1 (unused).
};

// is_leaf_node / leaf_sorted_slot — the fixed index arithmetic every stage
// and every traversal kernel/twin uses to tell a leaf from an internal node
// (positional, per the layout above — no extra flag bit needed).
inline bool is_leaf_node(int node_idx, int n) { return node_idx >= (n - 1); }
inline int  leaf_sorted_slot(int node_idx, int n) { return node_idx - (n - 1); }

// ===========================================================================
// Query-time shared arithmetic: distance and AABB-pruning tests. Shared for
// the same reason delta_lcp is (data-layout-level, not "the algorithm" —
// the surrounding TRAVERSAL LOOPS that decide when to prune/descend/collect
// are independently retyped per traversal kernel/twin).
// ===========================================================================

// squared_distance3 — |p-q|^2 for two {x,y,z} arrays. Squared (never sqrt)
// everywhere a radius/heap comparison only needs the ORDERING or a
// threshold test: x -> x^2 is monotonic for x >= 0, so every "<=" or "<"
// comparison below is exactly preserved without paying a sqrt per test
// (the same free micro-optimization 02.04's squared_distance names).
inline float squared_distance3(const float p[3], const float q[3])
{
    const float dx = p[0] - q[0], dy = p[1] - q[1], dz = p[2] - q[2];
    return dx * dx + dy * dy + dz * dz;
}

// aabb_min_dist2 — the squared distance from point q to the NEAREST point
// of an axis-aligned box [bmin,bmax] (0 if q is inside the box). This is
// THE pruning test for every traversal below: for radius search, a node is
// prunable iff aabb_min_dist2 > r^2 (the box cannot contain anything within
// r); for KNN, a node is prunable iff aabb_min_dist2 > (current K-th best
// distance)^2 (the box cannot contain anything better than what the heap
// already holds). Per-axis: if q's coordinate is inside [lo,hi] on that
// axis, that axis contributes 0 (the box spans q there); otherwise it
// contributes the gap to the nearer face, squared. Summed across axes —
// the standard "clamp point to box, measure the clamp distance" identity.
inline float aabb_min_dist2(const float bmin[3], const float bmax[3], const float q[3])
{
    float d2 = 0.0f;
    for (int a = 0; a < 3; ++a) {
        float gap = 0.0f;
        if (q[a] < bmin[a])      gap = bmin[a] - q[a];
        else if (q[a] > bmax[a]) gap = q[a] - bmax[a];
        d2 += gap * gap;
    }
    return d2;
}

// ---------------------------------------------------------------------------
// KnnCandidate / the (dist2, idx) TOTAL ORDER every KNN heap (GPU, CPU twin,
// brute-force oracle) compares by, and the documented TIE-BREAK POLICY:
// smaller dist2 wins; on an EXACT dist2 tie, the SMALLER original point
// index wins. knn_less below is that single comparison, shared so every
// implementation orders candidates identically — the precondition for
// "KNN results exact on distance ties" (README "Expected output",
// THEORY.md "Numerical considerations"). Because the final K-heap contents
// are the K smallest elements under a FIXED total order regardless of the
// ORDER candidates are discovered in (a basic property of a correct
// bounded top-K selection, whether by heap or brute force), every
// implementation below is required to produce the IDENTICAL final K-set
// and the IDENTICAL sorted order — not just an equivalent one.
// ---------------------------------------------------------------------------
struct KnnCandidate {
    float dist2;   // squared distance to the query, meters^2
    int   idx;     // original point index
};

// knn_less — true iff `a` strictly precedes `b` in the shared total order.
inline bool knn_less(const KnnCandidate& a, const KnnCandidate& b)
{
    if (a.dist2 != b.dist2) return a.dist2 < b.dist2;
    return a.idx < b.idx;
}

// ===========================================================================
// GPU kernel declarations — nvcc-only (see the file header for why: cl.exe,
// compiling reference_cpu.cpp, has never heard of __global__).
// ===========================================================================
#ifdef __CUDACC__

// ---- Stage 1: Morton / augmented keys -------------------------------------

// compute_augmented_keys_kernel — one thread per point: encode+pack this
// point's augmented key. in xyz [n*3] device floats, aabb (by value, tiny);
// out keys [n] device uint64_t, in ORIGINAL point-index order (pre-sort).
__global__ void compute_augmented_keys_kernel(int n, const float* __restrict__ xyz,
                                              SceneAABB aabb,
                                              unsigned long long* __restrict__ keys);

// ---- Stage 3: Karras radix-tree construction ------------------------------

// build_radix_tree_kernel — one thread per INTERNAL node i in [0, n-2].
// sorted_key [n] is the STAGE-2 sorted augmented-key array (read-only,
// shared with every other stage that needs "which original point is at
// sorted position k"). Writes nodes[i].left/right and, for BOTH children,
// nodes[child].parent = i. See kernels.cu for the full Karras walkthrough.
__global__ void build_radix_tree_kernel(int n, const unsigned long long* __restrict__ sorted_key,
                                        LbvhNode* __restrict__ nodes);

// init_leaves_kernel — one thread per LEAF k in [0, n): nodes[n-1+k] gets
// its point_idx (decoded from sorted_key[k]) and its degenerate single-
// point AABB (aabb_min == aabb_max == that point's xyz). Also sets
// nodes[n-1+k].left = nodes[n-1+k].right = -1 (leaves have no children).
__global__ void init_leaves_kernel(int n, const float* __restrict__ xyz,
                                   const unsigned long long* __restrict__ sorted_key,
                                   LbvhNode* __restrict__ nodes);

// propagate_aabb_kernel — one thread per LEAF k in [0, n): climbs the
// parent chain via the SECOND-ARRIVAL atomic-flag trick (file header
// "STAGE 4"), writing each internal node's AABB exactly once, by exactly
// one thread. flags [n-1] (one per INTERNAL node) must be zeroed by the
// caller before this launch (main.cu does so via cudaMemset each build).
__global__ void propagate_aabb_kernel(int n, LbvhNode* __restrict__ nodes,
                                      unsigned int* __restrict__ flags);

// ---- Stage 5a: BVH radius search -------------------------------------------

// radius_search_bvh_kernel — one thread per QUERY q in [0, num_queries).
// Stack-based traversal from the root (node 0); collects every point
// within kRadiusM, sorted ascending by original index (a canonical order
// for the exact-set-equality gates — no atomics needed: one thread owns
// its ENTIRE output row, unlike 02.01/02.04's per-point atomic appends).
//   out_ids   [num_queries * kMaxRadiusResults] OUT: this query's result
//             point indices, ascending, in [0, out_count[q]).
//   out_count [num_queries] OUT: how many results this query found.
//   out_overflow [num_queries] OUT: 1 iff results would have exceeded
//             kMaxRadiusResults (never expected — see that constant).
//   out_nodes_visited [num_queries] OUT: traversal_stats raw material.
//   out_stack_hwm     [num_queries] OUT: this query's traversal stack HIGH-
//             WATER MARK (max simultaneous depth reached) — traversal_stats
//             compares this against kBvhStackSize's proven bound.
__global__ void radius_search_bvh_kernel(const LbvhNode* __restrict__ nodes, int n,
                                         const float* __restrict__ xyz,
                                         const float* __restrict__ queries, int num_queries,
                                         float radius,
                                         int* __restrict__ out_ids,
                                         int* __restrict__ out_count,
                                         int* __restrict__ out_overflow,
                                         int* __restrict__ out_nodes_visited,
                                         int* __restrict__ out_stack_hwm);

// ---- Stage 5b: BVH K-nearest-neighbor --------------------------------------

// knn_search_bvh_kernel — one thread per QUERY. Maintains a size-kQueryK
// bounded max-heap (by knn_less) in LOCAL/register memory, pruning any
// subtree whose aabb_min_dist2 exceeds the current worst-of-heap distance
// (only meaningful once the heap holds kQueryK candidates — before that,
// every reachable subtree must still be explored). Final heap is sorted
// ascending (insertion sort over kQueryK<=~dozens elements — cheap) into
// out_ids/out_dist2, in knn_less order (the documented tie-break).
//   out_ids   [num_queries * kQueryK] OUT: ascending-by-knn_less neighbor ids.
//   out_dist2 [num_queries * kQueryK] OUT: matching squared distances.
//   out_found [num_queries] OUT: how many neighbors were actually found
//             (== kQueryK unless n < kQueryK — never true at this
//             project's scale, but checked honestly, not assumed).
//   out_stack_hwm [num_queries] OUT: traversal stack high-water mark (see
//             radius_search_bvh_kernel's identical parameter).
__global__ void knn_search_bvh_kernel(const LbvhNode* __restrict__ nodes, int n,
                                      const float* __restrict__ xyz,
                                      const float* __restrict__ queries, int num_queries,
                                      int* __restrict__ out_ids,
                                      float* __restrict__ out_dist2,
                                      int* __restrict__ out_found,
                                      int* __restrict__ out_nodes_visited,
                                      int* __restrict__ out_stack_hwm);

// ---- The domain contrast: fixed-radius voxel-hash search ------------------
// (02.01 Method B / 02.04's sort + 27-cell-stencil + binary-search lineage,
// cited and reimplemented compactly here — see kernels.cu for the shared
// mark_boundaries_kernel this project reuses almost verbatim from both.)

// compute_hash_keys_kernel — one thread per point: pack this point's voxel
// key at leaf = kRadiusM (the SAME leaf==radius proof 02.04's
// kClusterToleranceM comment gives, applied here to a QUERY radius instead
// of a clustering tolerance — either way, a 3x3x3 stencil around a query's
// own voxel is provably sufficient to find every point within kRadiusM).
__global__ void compute_hash_keys_kernel(int n, const float* __restrict__ xyz,
                                         float leaf, unsigned long long* __restrict__ keys);

// mark_boundaries_kernel — 02.01 Method B's identical boundary-marking
// kernel, cited and reused verbatim (both projects need the same "which
// sorted positions start a new voxel key" primitive).
__global__ void mark_boundaries_kernel(int n, const unsigned long long* __restrict__ keys_sorted,
                                       int* __restrict__ is_start);

// radius_search_hash_kernel — one thread per QUERY: compute the query's own
// voxel (at leaf=kRadiusM), scan its 3x3x3 stencil, binary-search each
// candidate voxel key into unique_key[], walk its point run testing actual
// squared distance <= kRadiusM^2. Same output-buffer layout/cap as the BVH
// radius kernel above (a fair, directly comparable format).
__global__ void radius_search_hash_kernel(const float* __restrict__ xyz,
                                          const unsigned long long* __restrict__ unique_key, int num_voxels,
                                          const int* __restrict__ seg_start,
                                          const int* __restrict__ idx_sorted, int n_sorted,
                                          const float* __restrict__ queries, int num_queries,
                                          float leaf, float radius,
                                          int* __restrict__ out_ids,
                                          int* __restrict__ out_count,
                                          int* __restrict__ out_overflow);

#endif // __CUDACC__

// ===========================================================================
// Host-callable launch wrappers (definitions in kernels.cu, which only nvcc
// compiles — but the DECLARATIONS below are plain C++, visible to main.cu).
// ===========================================================================

void launch_compute_augmented_keys(int n, const float* d_xyz, SceneAABB aabb, unsigned long long* d_keys);

void launch_build_radix_tree(int n, const unsigned long long* d_sorted_key, LbvhNode* d_nodes);

void launch_init_leaves(int n, const float* d_xyz, const unsigned long long* d_sorted_key, LbvhNode* d_nodes);

void launch_propagate_aabb(int n, LbvhNode* d_nodes, unsigned int* d_flags);

void launch_radius_search_bvh(const LbvhNode* d_nodes, int n, const float* d_xyz,
                              const float* d_queries, int num_queries, float radius,
                              int* d_out_ids, int* d_out_count, int* d_out_overflow,
                              int* d_out_nodes_visited, int* d_out_stack_hwm);

void launch_knn_search_bvh(const LbvhNode* d_nodes, int n, const float* d_xyz,
                           const float* d_queries, int num_queries,
                           int* d_out_ids, float* d_out_dist2, int* d_out_found,
                           int* d_out_nodes_visited, int* d_out_stack_hwm);

// launch_sort_augmented_keys — Stage 2: sort the augmented-key array
// ascending, IN PLACE, via thrust::sort (a GPU radix sort for 64-bit
// unsigned keys — see kernels.cu's call site for what that computes and
// why no paired "value" array is needed, unlike the voxel-hash sort below).
void launch_sort_augmented_keys(int n, unsigned long long* d_keys_sorted_inout);

void launch_compute_hash_keys(int n, const float* d_xyz, float leaf, unsigned long long* d_keys);

// launch_build_voxel_index — Thrust sort + boundary compaction (02.01
// Method-B pipeline, cited). d_keys_in [n] READ-ONLY; every other array is
// caller-provided scratch/output sized n. Returns the number of OCCUPIED
// voxels (== valid unique_key_out/seg_start entries).
int launch_build_voxel_index(int n, const unsigned long long* d_keys_in,
                             unsigned long long* d_keys_scratch, int* d_idx_scratch,
                             int* d_is_start_scratch, int* d_seg_start_out,
                             unsigned long long* d_unique_key_out);

void launch_radius_search_hash(const float* d_xyz,
                               const unsigned long long* d_unique_key, int num_voxels,
                               const int* d_seg_start, const int* d_idx_sorted, int n_sorted,
                               const float* d_queries, int num_queries, float leaf, float radius,
                               int* d_out_ids, int* d_out_count, int* d_out_overflow);

// ===========================================================================
// CPU references (reference_cpu.cpp) — the correctness-oracle twins. All
// pointers below are HOST pointers. See reference_cpu.cpp's file header for
// the independence ruling each of these follows — NOT all identical.
// ===========================================================================

// compute_augmented_keys_cpu — the twin of compute_augmented_keys_kernel,
// calling this header's OWN morton_encode30/augmented_key (shared data-
// layout formula). VERIFY(morton) in main.cu compares this, point for
// point, against the GPU's device-transcribed version, bit-exact.
void compute_augmented_keys_cpu(int n, const float* xyz, const SceneAABB& aabb,
                                unsigned long long* keys_out);

// sort_keys_cpu — std::sort (an algorithmically DIFFERENT sort than
// Thrust's GPU radix sort) on a COPY of the augmented-key array. Because
// augmented keys are pairwise distinct (see augmented_key above), the
// sorted result is the UNIQUE correct total order — no stability question
// arises. VERIFY(sort) compares the full sorted array element-wise, exact.
void sort_keys_cpu(int n, const unsigned long long* keys_in, unsigned long long* keys_sorted_out);

// build_radix_tree_cpu — a GENUINELY INDEPENDENT re-implementation of
// Karras's construction: the SAME range-determination/split-finding
// algorithm, RETYPED as a plain sequential loop over internal nodes
// (not calling any shared "build" function — only the low-level delta_lcp
// bit-arithmetic is shared, per the independence ruling). VERIFY(topology)
// compares left/right/parent for every internal node against the GPU's
// result, exact.
void build_radix_tree_cpu(int n, const unsigned long long* sorted_key, LbvhNode* nodes);

void init_leaves_cpu(int n, const float* xyz, const unsigned long long* sorted_key, LbvhNode* nodes);

// propagate_aabb_cpu — an INDEPENDENT (non-atomic, single-threaded,
// iterative post-order) bottom-up AABB computation — genuinely different
// from the GPU's atomic-flag race, sharing no code with it. VERIFY(aabb)
// compares every node's aabb_min/aabb_max against the GPU result; documented
// as EXACT (not tolerance-bounded) because float min/max is an order-
// independent, non-rounding operation over a FIXED point set — see that
// gate's comment in main.cu and THEORY.md "Numerical considerations" for
// the full argument (extending 02.02's AABB order-independence lineage).
void propagate_aabb_cpu(int n, LbvhNode* nodes);

// radius_search_bvh_cpu — an INDEPENDENT (retyped, not shared) sequential
// stack traversal twin of radius_search_bvh_kernel, mirroring 11.01's
// lidar_raycast_cpu / intersect_bvh pairing. Results returned pre-sorted
// ascending by point index (the same canonical order the GPU produces).
void radius_search_bvh_cpu(const LbvhNode* nodes, int n, const float* xyz,
                           const float* query, float radius,
                           std::vector<int>& out_ids, bool& out_overflow);

// knn_search_bvh_cpu — an INDEPENDENT sequential BVH-KNN twin (its own
// heap logic, retyped; shares only the knn_less total-order comparison and
// aabb_min_dist2, both data-layout-level per the ruling).
void knn_search_bvh_cpu(const LbvhNode* nodes, int n, const float* xyz,
                        const float* query, std::vector<KnnCandidate>& out_sorted);

// compute_hash_keys_cpu — the twin of compute_hash_keys_kernel (shared
// voxel-key formula, same VERIFY-by-comparison story as compute_augmented_keys_cpu).
void compute_hash_keys_cpu(int n, const float* xyz, float leaf, unsigned long long* keys_out);

// HashMapCpu — the independent voxel-hash oracle's data structure: an
// std::unordered_map<uint64_t, std::vector<int>> voxel->points map — a
// completely different data structure than the GPU's sorted-array+binary-
// search index (the same "independent data structure" choice 02.04's
// build_edges_cpu makes). Built ONCE (build_hash_map_cpu, O(n)) and then
// queried up to Q times (radius_search_hash_cpu, O(1) expected per voxel
// probe) — main.cu calls the builder once, not once per query, the same
// "build once, query many" discipline every index structure in this
// project (the LBVH itself included) follows.
using HashMapCpu = std::unordered_map<unsigned long long, std::vector<int>>;

void build_hash_map_cpu(int n, const float* xyz, float leaf, HashMapCpu& out_map);

// radius_search_hash_cpu — one query against the prebuilt map: the 3x3x3
// stencil + actual-distance test, mirroring radius_search_hash_kernel.
void radius_search_hash_cpu(const HashMapCpu& map, float leaf, float radius,
                            const float* xyz, const float* query, std::vector<int>& out_ids);

// ---------------------------------------------------------------------------
// Brute-force oracles — THE independent verification gate the
// reference_cpu.cpp ruling requires: a THIRD implementation, sharing no
// tree/hash/traversal code with EITHER twin above, that cannot be fooled by
// a bug common to both the GPU kernel and its CPU twin (e.g. a shared
// misunderstanding of the algorithm). O(n) per query — deliberately never
// run over the full query set (main.cu caps it at the documented anchor
// subset; see kernels.cuh's kMaxRadiusResults-style "size to a documented
// scope" discipline applied to RUNTIME cost here instead of memory).
// ---------------------------------------------------------------------------
void radius_search_brute_force(int n, const float* xyz, const float* query, float radius,
                               std::vector<int>& out_ids);

void knn_search_brute_force(int n, const float* xyz, const float* query,
                            std::vector<KnnCandidate>& out_sorted);

#endif // PROJECT_KERNELS_CUH
