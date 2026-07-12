// ===========================================================================
// kernels.cu — GPU kernels for project 02.05 (KD-tree or LBVH construction +
//              KNN/radius search on GPU): Morton/sort/Karras-radix-tree/
//              AABB-propagation build, BVH radius+KNN traversal, and the
//              voxel-hash fixed-radius baseline (the domain contrast).
//
// Role in the project
// -------------------
// All __global__ (GPU) code lives here, plus the host-side launch wrappers
// that own the grid/block math (CLAUDE.md §6.1 rule 2). Every constant,
// struct, and shared arithmetic helper is defined ONCE in kernels.cuh —
// read that file's long header comment FIRST; it is the map of this one.
//
// Read this after: main.cu, kernels.cuh.  Read this before: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK / CUDA_CHECK_LAST_ERROR

// Thrust: header-only pieces of the CUDA Toolkit (CLAUDE.md §5). Used for
// exactly two things — sorting the augmented-key array (Stage 2) and the
// voxel-hash boundary compaction (02.01 Method-B lineage) — each explained
// at its call site (CLAUDE.md §6.1 rule 6).
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/iterator/counting_iterator.h>

// is_nonzero — the copy_if predicate for voxel-hash boundary compaction
// (02.01's kernels.cu comment explains why a hand-written functor replaces
// CUDA 13.3's removed thrust::identity).
struct is_nonzero {
    __host__ __device__ bool operator()(int x) const { return x != 0; }
};

// ===========================================================================
// Device transcriptions of kernels.cuh's shared plain-inline helpers.
//
// WHY DUPLICATED: kernels.cuh's morton/augmented-key/clz/distance/AABB
// helpers are unqualified so cl.exe (reference_cpu.cpp) can see them too —
// which means nvcc treats them as HOST-only and refuses to call them from a
// __global__ kernel. Each __device__ copy below is a deliberate, literal
// transcription (02.01's d_voxel_coord precedent), and every one of them
// feeds a VERIFY gate in main.cu that would catch a drift between a copy
// and its header original on the very first mismatched point/node/query.
// ===========================================================================

__device__ __forceinline__ uint32_t d_quantize_axis(float p, float lo, float hi)
{
    float t = (hi > lo) ? (p - lo) / (hi - lo) : 0.0f;
    t = fminf(fmaxf(t, 0.0f), 1.0f);
    return static_cast<uint32_t>(t * static_cast<float>(kMortonAxisMax));
}

__device__ __forceinline__ uint32_t d_expand_bits10(uint32_t v)
{
    v &= 0x000003FFu;
    v = (v | (v << 16)) & 0x030000FFu;
    v = (v | (v << 8))  & 0x0300F00Fu;
    v = (v | (v << 4))  & 0x030C30C3u;
    v = (v | (v << 2))  & 0x09249249u;
    return v;
}

__device__ __forceinline__ uint32_t d_morton_encode30(float x, float y, float z, SceneAABB aabb)
{
    const uint32_t xi = d_quantize_axis(x, aabb.min[0], aabb.max[0]);
    const uint32_t yi = d_quantize_axis(y, aabb.min[1], aabb.max[1]);
    const uint32_t zi = d_quantize_axis(z, aabb.min[2], aabb.max[2]);
    return d_expand_bits10(xi) | (d_expand_bits10(yi) << 1) | (d_expand_bits10(zi) << 2);
}

__device__ __forceinline__ unsigned long long d_augmented_key(uint32_t morton, int32_t idx)
{
    return (static_cast<unsigned long long>(morton) << 32) |
           static_cast<unsigned long long>(static_cast<uint32_t>(idx));
}

// d_clz64 — count-leading-zeros of a 64-bit value via the hardware
// intrinsic __clzll. Unlike the bit-packing formulas above (which encode a
// DESIGN DECISION that a typo could silently corrupt), clz is a
// mathematically unambiguous function — "how many leading zero bits does
// this integer have" has exactly one correct answer, so there is no
// meaningful "drift" between this intrinsic and kernels.cuh's portable
// clz64_portable() beyond both being correct. Used for speed only: this
// runs once per delta_lcp call, and delta_lcp runs O(log n) times per
// internal node during Stage 3's construction (~200k nodes) — a hot path.
__device__ __forceinline__ int d_clz64(unsigned long long x)
{
    return __clzll(static_cast<long long>(x));
}

// d_delta_lcp — the device transcription of kernels.cuh's delta_lcp.
__device__ __forceinline__ int d_delta_lcp(const unsigned long long* sorted_key, int n, int i, int j)
{
    if (j < 0 || j >= n) return -1;
    return d_clz64(sorted_key[i] ^ sorted_key[j]);
}

__device__ __forceinline__ float d_squared_distance3(const float p[3], const float q[3])
{
    const float dx = p[0] - q[0], dy = p[1] - q[1], dz = p[2] - q[2];
    return dx * dx + dy * dy + dz * dz;
}

__device__ __forceinline__ float d_aabb_min_dist2(const float bmin[3], const float bmax[3], const float q[3])
{
    float d2 = 0.0f;
    #pragma unroll
    for (int a = 0; a < 3; ++a) {
        float gap = 0.0f;
        if (q[a] < bmin[a])      gap = bmin[a] - q[a];
        else if (q[a] > bmax[a]) gap = q[a] - bmax[a];
        d2 += gap * gap;
    }
    return d2;
}

__device__ __forceinline__ bool d_knn_less(float da, int ia, float db, int ib)
{
    if (da != db) return da < db;
    return ia < ib;
}

// d_is_leaf_node — device transcription of kernels.cuh's is_leaf_node
// (plain positional index arithmetic: leaves live at [n-1, 2n-2]).
__device__ __forceinline__ bool d_is_leaf_node(int node_idx, int n) { return node_idx >= (n - 1); }

// ===========================================================================
// STAGE 1 — augmented Morton keys.
// ===========================================================================

// compute_augmented_keys_kernel — a pure MAP, one thread per point: encode
// this point's (Morton, index) key. Thread i owns point i exclusively
// (no other thread ever reads or writes point i's data here), so — exactly
// like 02.01's compute_keys_kernel — no synchronization is needed at all.
__global__ void compute_augmented_keys_kernel(int n, const float* __restrict__ xyz,
                                              SceneAABB aabb,
                                              unsigned long long* __restrict__ keys)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float px = xyz[i * 3 + 0], py = xyz[i * 3 + 1], pz = xyz[i * 3 + 2];
    const uint32_t morton = d_morton_encode30(px, py, pz, aabb);
    keys[i] = d_augmented_key(morton, i);
}

void launch_compute_augmented_keys(int n, const float* d_xyz, SceneAABB aabb, unsigned long long* d_keys)
{
    const int block = kThreadsPerBlock;
    const int grid  = blocks_for(n, block);
    compute_augmented_keys_kernel<<<grid, block>>>(n, d_xyz, aabb, d_keys);
    CUDA_CHECK_LAST_ERROR("compute_augmented_keys_kernel launch");
}

// ===========================================================================
// STAGE 3 — Karras radix-tree construction.
//
// build_radix_tree_kernel — THE central kernel of this project. One thread
// per INTERNAL node i in [0, n-2]. Every thread computes its node's
// children FROM THE SORTED KEY ARRAY ALONE — no other thread's result is
// ever read, so this launch has NO data dependency between threads and
// needs no synchronization, no barrier, and no multi-pass structure. That
// is the entire point of Karras's construction versus a top-down recursive
// build (11.01's median-split BVH, built sequentially ON THE HOST): a
// top-down build must finish splitting a node before it can even START
// splitting that node's children (each level depends on the previous one);
// this bottom-up-by-formula approach computes ALL levels SIMULTANEOUSLY,
// because it never actually walks down from a root — it walks OUTWARD from
// each internal node's own position i to find "how far does my shared
// prefix reach", which is a self-contained question every i can answer in
// parallel (THEORY.md "The GPU mapping" restates this as the "independence
// argument" and contrasts it explicitly with 11.01's approach).
//
// The two-part algorithm per internal node i (Karras 2012; the exact
// formulation below follows the widely-reproduced reference pseudocode
// from Tero Karras's NVIDIA Developer Blog article "Thinking Parallel,
// Part III: Tree Construction on the GPU" — README "Prior art" cites it):
//
//   PART A — determineRange(i): find the two ends [first,last] of the
//   CONTIGUOUS sorted-array range that node i is the root of. Node i's
//   range always starts AT i; the DIRECTION it grows (toward i-1 or i+1)
//   is decided by comparing how much prefix i shares with its LEFT
//   neighbor versus its RIGHT neighbor — node i grows toward whichever
//   neighbor it shares MORE prefix with (that is the neighbor "on i's
//   side" of the eventual split). Once the direction d is fixed, an
//   EXPONENTIAL (doubling) search finds a generous upper bound on the
//   range length, then a BINARY search narrows it to the exact far end j —
//   O(log(range length)) delta_lcp calls total, versus an O(range length)
//   linear scan; THEORY.md "The algorithm" gives the full complexity
//   argument.
//
//   PART B — findSplit(first,last): within the now-known range
//   [first,last], binary-search for the exact position `split` where the
//   shared-prefix length INCREASES beyond the whole range's own shared
//   prefix (commonPrefix = delta_lcp(first,last)) — this position is
//   provably where the LEFT child's range ends and the RIGHT child's range
//   begins (THEORY.md "The math" sketches why: it is exactly the highest
//   bit at which points in [first,last] disagree, so it is the natural
//   "next branch" of the induced binary radix trie).
//
//   Each resulting sub-range degenerates to a SINGLE INDEX (first==last)
//   exactly when it names a LEAF; otherwise it names another INTERNAL
//   node, whose OWN index is defined (by Karras's construction) to be
//   `split` (or `split+1`) — the arithmetic below turns each child range
//   into a single node-array index via is_leaf_node/leaf_sorted_slot.
//
// Duplicate-key simplification versus the textbook algorithm: Karras's
// original findSplit special-cases firstCode==lastCode (real Morton codes
// collide routinely). Because THIS project's keys are AUGMENTED with the
// point's unique index (kernels.cuh's augmented_key), sorted_key[first] ==
// sorted_key[last] is IMPOSSIBLE whenever first != last — so that branch
// is provably dead code here and is omitted, with this comment standing in
// its place (CLAUDE.md "no black boxes": the omission is explained, not
// silent).
// ---------------------------------------------------------------------------
__global__ void build_radix_tree_kernel(int n, const unsigned long long* __restrict__ sorted_key,
                                        LbvhNode* __restrict__ nodes)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n - 1) return;   // n-1 internal nodes: indices [0, n-2]

    // ---- PART A: determineRange -------------------------------------------
    // Direction: which neighbor does i share MORE prefix with? d=+1 grows
    // the range rightward (the common case for i=0, where i-1 is out of
    // range and delta_lcp returns -1, i.e. "-infinity" — the range always
    // grows right at the very start of the array, as it must).
    const int delta_right = d_delta_lcp(sorted_key, n, i, i + 1);
    const int delta_left  = d_delta_lcp(sorted_key, n, i, i - 1);
    const int d = (delta_right - delta_left >= 0) ? 1 : -1;

    // delta_min: the shared-prefix length on the "wrong" side (i-d) — the
    // threshold every candidate far-end must EXCEED to still belong to
    // this node's range (this is what makes the range "as long as possible
    // while still sharing MORE prefix with i than the wrong-side neighbor
    // does" — the defining property of node i's range).
    const int delta_min = d_delta_lcp(sorted_key, n, i, i - d);

    // Exponential search for an upper bound l_max on the range length: keep
    // doubling until stepping l_max further no longer beats delta_min —
    // O(log(true length)) iterations versus scanning one step at a time.
    int l_max = 2;
    while (d_delta_lcp(sorted_key, n, i, i + l_max * d) > delta_min) {
        l_max *= 2;
    }

    // Binary search within [0, l_max] for the exact range length l: at
    // each halving step, greedily accept the larger length if it still
    // shares MORE prefix than delta_min (a standard "binary search for the
    // last position satisfying a monotonic predicate" pattern).
    int l = 0;
    for (int t = l_max / 2; t >= 1; t /= 2) {
        if (d_delta_lcp(sorted_key, n, i, i + (l + t) * d) > delta_min) {
            l += t;
        }
    }
    const int j = i + l * d;                       // the range's far end
    const int first = (d > 0) ? i : j;              // normalize to ascending [first,last]
    const int last  = (d > 0) ? j : i;

    // ---- PART B: findSplit --------------------------------------------------
    // Binary search within [first,last] for the split position: the LAST
    // index whose shared prefix with `first` still exceeds the WHOLE
    // range's own common prefix (commonPrefix). This is exactly the
    // boundary the induced binary radix trie branches at (THEORY.md "The
    // math"). firstCode==lastCode can never hold here (see the header
    // comment's "duplicate-key simplification"), so no degenerate branch.
    const int common_prefix = d_delta_lcp(sorted_key, n, first, last);
    int split = first;
    int step = last - first;
    do {
        step = (step + 1) >> 1;                      // shrink geometrically, rounding UP so step==1 is reachable
        const int new_split = split + step;
        if (new_split < last) {
            const int split_prefix = d_delta_lcp(sorted_key, n, first, new_split);
            if (split_prefix > common_prefix) {
                split = new_split;                    // accept: still within the "same branch" as first
            }
        }
    } while (step > 1);

    // ---- Emit children --------------------------------------------------------
    // A child range that has COLLAPSED to a single index (split==first, or
    // split+1==last) names a LEAF; otherwise it names another internal
    // node, whose node-array index IS its own range's start (split or
    // split+1) — the defining recursive property of Karras's numbering
    // (every internal node's index equals the START of the range it owns).
    const int left_child  = (split == first)     ? (n - 1 + split)      : split;
    const int right_child = (split + 1 == last)  ? (n - 1 + split + 1)  : (split + 1);

    nodes[i].left  = left_child;
    nodes[i].right = right_child;
    nodes[left_child].parent  = i;
    nodes[right_child].parent = i;
    // nodes[i].parent is written by WHICHEVER other internal node (or
    // nothing, if i==0, the root) owns i as a child — never by node i
    // itself. main.cu pre-fills every parent slot to -1 before this launch,
    // so a root (never claimed as anyone's child) correctly keeps parent=-1.
}

void launch_build_radix_tree(int n, const unsigned long long* d_sorted_key, LbvhNode* d_nodes)
{
    if (n < 2) return;   // fewer than 2 points: no internal nodes exist at all (degenerate; see main.cu's guard)
    const int block = kThreadsPerBlock;
    const int grid  = blocks_for(n - 1, block);
    build_radix_tree_kernel<<<grid, block>>>(n, d_sorted_key, d_nodes);
    CUDA_CHECK_LAST_ERROR("build_radix_tree_kernel launch");
}

// init_leaves_kernel — one thread per leaf slot k in [0,n): resolve the
// original point index at sorted position k, stamp the leaf's degenerate
// (single-point) AABB, and mark it childless. A pure map, like Stage 1.
__global__ void init_leaves_kernel(int n, const float* __restrict__ xyz,
                                   const unsigned long long* __restrict__ sorted_key,
                                   LbvhNode* __restrict__ nodes)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;

    const int point_idx = static_cast<int>(sorted_key[k] & 0xFFFFFFFFull);
    const int node_idx  = (n - 1) + k;

    LbvhNode leaf;
    leaf.aabb_min[0] = leaf.aabb_max[0] = xyz[point_idx * 3 + 0];
    leaf.aabb_min[1] = leaf.aabb_max[1] = xyz[point_idx * 3 + 1];
    leaf.aabb_min[2] = leaf.aabb_max[2] = xyz[point_idx * 3 + 2];
    leaf.left = -1;
    leaf.right = -1;
    leaf.parent = -1;   // overwritten by build_radix_tree_kernel unless this leaf IS the root (n==1 only)
    leaf.point_idx = point_idx;
    nodes[node_idx] = leaf;
}

void launch_init_leaves(int n, const float* d_xyz, const unsigned long long* d_sorted_key, LbvhNode* d_nodes)
{
    const int block = kThreadsPerBlock;
    const int grid  = blocks_for(n, block);
    init_leaves_kernel<<<grid, block>>>(n, d_xyz, d_sorted_key, d_nodes);
    CUDA_CHECK_LAST_ERROR("init_leaves_kernel launch");
}

// ---------------------------------------------------------------------------
// propagate_aabb_kernel — STAGE 4, the bottom-up AABB fill.
//
// One thread per LEAF k in [0,n). Each thread climbs its OWN parent chain,
// using flags[] (one atomic counter per INTERNAL node, zeroed by the
// caller before this launch) to coordinate with its SIBLING'S thread — the
// classic LBVH "second-arrival" trick (Apetrei 2014 / the same NVIDIA blog
// series cited above; README "Prior art"):
//
//   At internal node p, atomicAdd(&flags[p], 1) returns the value BEFORE
//   the increment:
//     old == 0  ->  this thread is the FIRST of p's two children to finish.
//                   p's OTHER child's subtree is not necessarily done yet
//                   (that thread may still be several levels below p, or
//                   may not have reached p at all) — so p's AABB cannot be
//                   computed YET. This thread's job at p is done; RETURN.
//     old == 1  ->  this thread is the SECOND child to finish — the FIRST
//                   one already returned, which (by the atomic increment's
//                   total order) can only mean the sibling subtree's AABB
//                   was ALREADY WRITTEN before this thread's atomicAdd
//                   executed (every leaf writes its own AABB before this
//                   loop begins; every internal node writes its AABB
//                   before incrementing ITS parent's flag and continuing
//                   upward — an invariant maintained by induction from the
//                   leaves). So BOTH children are now guaranteed ready:
//                   union their AABBs into p, and CONTINUE climbing from p.
//
// Every internal node has EXACTLY two children and is visited by EXACTLY
// two leaf-rooted climbs (one through each child subtree) — so its flag is
// incremented exactly twice, guaranteeing exactly one thread sees old==1
// and computes it, and that computation happens exactly once, race-free,
// with no locks. This is the SAME "atomic counter as a rendezvous point"
// idiom 02.01's hash_compact_kernel and 02.04's union-find both use for
// DIFFERENT purposes (compaction, union) — the general pattern is
// "coordinate N parallel threads through shared state using ONE atomic
// primitive per shared decision", worth recognizing across all three.
//
// Divergence note: threads reach the root at DIFFERENT times (leaves at
// different depths climb different distances) — this loop is exactly as
// divergent as BVH traversal itself (THEORY.md "The GPU mapping" measures
// it), but it is embarrassingly parallel in the sense that no thread ever
// WAITS for another (a "first-arrival" thread simply stops, it never spins
// or blocks) — a lock-free, wait-free construction.
// ---------------------------------------------------------------------------
__global__ void propagate_aabb_kernel(int n, LbvhNode* __restrict__ nodes,
                                      unsigned int* __restrict__ flags)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    if (n < 2) return;   // single-point degenerate tree: the lone leaf IS the root, nothing to propagate

    int node = (n - 1) + k;   // this thread's leaf

    // Climb until we either stop (first arrival) or reach the root.
    while (true) {
        const int parent = nodes[node].parent;
        if (parent < 0) return;   // reached the root's parent sentinel: done

        // __threadfence() — THE bug this project's own test suite caught
        // during development, and the reason this comment is here instead
        // of a silent one-line fix: atomicAdd only guarantees ATOMICITY of
        // the counter update, not that this thread's PRIOR write of
        // node's own AABB is yet VISIBLE, in global memory, TO OTHER
        // THREADS. Without a fence, the GPU's memory system is free to let
        // the atomicAdd's effect (the flag increment) become visible to
        // another SM before this thread's plain AABB store does — so a
        // second-arrival thread on another SM could read a STALE AABB for
        // the sibling subtree it just got the "go-ahead" to combine.
        // __threadfence() blocks until all of THIS thread's prior global
        // writes are visible to every other thread in the grid, which is
        // exactly the guarantee the "second arrival sees a ready sibling"
        // argument (this kernel's header comment) silently assumed. This
        // is the same class of bug 02.01/02.04's atomicCAS/atomicAdd
        // sections warn about in general (CLAUDE.md "Numerical
        // considerations": race conditions, atomics, determinism) — THIS
        // project's own concrete instance of it, caught by GATE
        // brute_force_anchor (an independent oracle routing around the
        // tree entirely) when VERIFY(aabb) first failed during
        // development; see THEORY.md "Numerical considerations" for the
        // full writeup.
        __threadfence();

        const unsigned int old = atomicAdd(&flags[parent], 1u);
        if (old == 0u) {
            return;   // first arrival at `parent`: sibling not ready, stop climbing
        }

        // Second arrival: both children are ready. Union their AABBs.
        const int lc = nodes[parent].left;
        const int rc = nodes[parent].right;
        LbvhNode p = nodes[parent];
        #pragma unroll
        for (int a = 0; a < 3; ++a) {
            p.aabb_min[a] = fminf(nodes[lc].aabb_min[a], nodes[rc].aabb_min[a]);
            p.aabb_max[a] = fmaxf(nodes[lc].aabb_max[a], nodes[rc].aabb_max[a]);
        }
        nodes[parent] = p;

        node = parent;   // continue climbing from the node we just finished
    }
}

void launch_propagate_aabb(int n, LbvhNode* d_nodes, unsigned int* d_flags)
{
    if (n < 2) return;
    CUDA_CHECK(cudaMemset(d_flags, 0, static_cast<size_t>(n - 1) * sizeof(unsigned int)));
    const int block = kThreadsPerBlock;
    const int grid  = blocks_for(n, block);
    propagate_aabb_kernel<<<grid, block>>>(n, d_nodes, d_flags);
    CUDA_CHECK_LAST_ERROR("propagate_aabb_kernel launch");
}

// ===========================================================================
// STAGE 5a — BVH radius search.
//
// radius_search_bvh_kernel — one thread per query, small fixed-size local
// stack (kBvhStackSize; depth-bound proof in kernels.cuh's file header),
// push-both-children traversal — structurally the SAME shape as 11.01's
// intersect_bvh, with the AABB test swapped from "ray-slab intersection"
// to "AABB-sphere overlap" (aabb_min_dist2 <= r^2). Every thread owns its
// ENTIRE output row [q*kMaxRadiusResults, q*kMaxRadiusResults+count) —
// unlike 02.01/02.04's per-EDGE/per-VOXEL atomic appends (there, many
// threads race to extend ONE shared list), here each of the num_queries
// threads writes to a DISJOINT slice of the output array, so no atomics
// are needed anywhere in this kernel.
// ---------------------------------------------------------------------------
__global__ void radius_search_bvh_kernel(const LbvhNode* __restrict__ nodes, int n,
                                         const float* __restrict__ xyz,
                                         const float* __restrict__ queries, int num_queries,
                                         float radius,
                                         int* __restrict__ out_ids,
                                         int* __restrict__ out_count,
                                         int* __restrict__ out_overflow,
                                         int* __restrict__ out_nodes_visited,
                                         int* __restrict__ out_stack_hwm)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= num_queries) return;

    const float qp[3] = { queries[q * 3 + 0], queries[q * 3 + 1], queries[q * 3 + 2] };
    const float r2 = radius * radius;

    int stack[kBvhStackSize];   // PER-THREAD local array: registers while small, spilled-but-cached local memory otherwise
    int sp = 0;
    if (n >= 1) stack[sp++] = 0;   // root is always node 0 (n>=2 guaranteed by main.cu's build guard; n==1 handled below)

    int count = 0;
    bool overflow = false;
    int visited = 0;
    int stack_hwm = sp;   // high-water mark: the largest `sp` this traversal ever reached

    while (sp > 0) {
        const int node_idx = stack[--sp];
        ++visited;
        const LbvhNode node = nodes[node_idx];

        if (d_aabb_min_dist2(node.aabb_min, node.aabb_max, qp) > r2) {
            continue;   // this subtree cannot contain anything within radius: prune
        }

        if (d_is_leaf_node(node_idx, n)) {
            const float pp[3] = { xyz[node.point_idx * 3 + 0], xyz[node.point_idx * 3 + 1], xyz[node.point_idx * 3 + 2] };
            if (d_squared_distance3(pp, qp) <= r2) {
                if (count < kMaxRadiusResults) {
                    out_ids[q * kMaxRadiusResults + count] = node.point_idx;
                    ++count;
                } else {
                    overflow = true;   // documented cap exceeded: report, never silently drop (kMaxRadiusResults comment)
                }
            }
        } else {
            if (sp + 2 <= kBvhStackSize) {
                stack[sp++] = node.left;
                stack[sp++] = node.right;
                if (sp > stack_hwm) stack_hwm = sp;
            }
            // else: defensive guard against the (proven-unreachable at this
            // project's key width, but never ASSUMED in code) stack overflow —
            // 11.01's identical defensive discipline for its own depth bound.
        }
    }

    // Canonicalize: sort this query's result ids ascending (insertion sort —
    // count is small relative to n, and this makes the output a CANONICAL
    // representation so main.cu's exact-set-equality gates need no separate
    // "sort before compare" step; 08.01/02.0x precedent for "small, hot,
    // hand-rolled beats a library call" applies at this scale too).
    for (int a = 1; a < count; ++a) {
        const int key = out_ids[q * kMaxRadiusResults + a];
        int b = a - 1;
        while (b >= 0 && out_ids[q * kMaxRadiusResults + b] > key) {
            out_ids[q * kMaxRadiusResults + b + 1] = out_ids[q * kMaxRadiusResults + b];
            --b;
        }
        out_ids[q * kMaxRadiusResults + b + 1] = key;
    }

    out_count[q] = count;
    out_overflow[q] = overflow ? 1 : 0;
    out_nodes_visited[q] = visited;
    out_stack_hwm[q] = stack_hwm;
}

void launch_radius_search_bvh(const LbvhNode* d_nodes, int n, const float* d_xyz,
                              const float* d_queries, int num_queries, float radius,
                              int* d_out_ids, int* d_out_count, int* d_out_overflow,
                              int* d_out_nodes_visited, int* d_out_stack_hwm)
{
    const int block = kThreadsPerBlock;
    const int grid  = blocks_for(num_queries, block);
    radius_search_bvh_kernel<<<grid, block>>>(d_nodes, n, d_xyz, d_queries, num_queries, radius,
                                              d_out_ids, d_out_count, d_out_overflow, d_out_nodes_visited,
                                              d_out_stack_hwm);
    CUDA_CHECK_LAST_ERROR("radius_search_bvh_kernel launch");
}

// ---------------------------------------------------------------------------
// knn_search_bvh_kernel — STAGE 5b, K-nearest-neighbor via a bounded
// max-heap with SHRINKING-RADIUS pruning — this project's other genuinely
// new traversal idea (radius search prunes against a FIXED r; KNN prunes
// against a radius that gets TIGHTER as better candidates are found).
//
// heap[0..kQueryK) is a binary MAX-heap ordered by knn_less: heap[0] is
// always the WORST (largest-dist2, or index-tie-break) of the current best
// kQueryK candidates. Two operations:
//   * while heap not full: plain heap INSERT (append + sift-up) — every
//     candidate is provisionally "good enough" until the heap fills.
//   * once full: a new candidate only enters if it beats heap[0] (the
//     current worst) — REPLACE the root and sift-down. This is the
//     textbook "bounded top-K via a max-heap" pattern (the same shape a
//     std::priority_queue-based top-K uses, hand-rolled here in
//     registers/local memory because CUDA device code has no STL).
//
// The PRUNE test this enables: once the heap holds kQueryK candidates,
// heap[0].dist2 is an UPPER BOUND on "the Kth-best distance so far" — any
// subtree whose aabb_min_dist2 exceeds heap[0].dist2 provably cannot
// contain anything that would improve the heap, so it is skipped entirely.
// Early in the traversal (heap not yet full), NO pruning by distance is
// valid — every reachable node must still be explored, so the effective
// radius is +infinity until the heap fills. This is exactly the "shrinking
// search radius" optimization named in kernels.cuh's file header and
// derived in THEORY.md "The algorithm".
//
// ORDER INDEPENDENCE (why this is exact regardless of traversal order):
// the final heap always holds the GLOBALLY kQueryK-smallest (dist2,idx)
// pairs among every point actually visited, and a point is only EVER
// skipped (not visited) when its entire containing subtree is PROVABLY
// farther than every point already in the heap — so no traversal order can
// cause a true top-K point to be missed. This is why main.cu's VERIFY(knn_bvh)
// gate can demand EXACT equality against an independently-ordered CPU
// traversal, and GATE brute_force_anchor can demand exact equality against
// a traversal-free linear scan: all three are computing the unique correct
// answer under the shared knn_less total order (kernels.cuh).
// ---------------------------------------------------------------------------
__global__ void knn_search_bvh_kernel(const LbvhNode* __restrict__ nodes, int n,
                                      const float* __restrict__ xyz,
                                      const float* __restrict__ queries, int num_queries,
                                      int* __restrict__ out_ids,
                                      float* __restrict__ out_dist2,
                                      int* __restrict__ out_found,
                                      int* __restrict__ out_nodes_visited,
                                      int* __restrict__ out_stack_hwm)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= num_queries) return;

    const float qp[3] = { queries[q * 3 + 0], queries[q * 3 + 1], queries[q * 3 + 2] };

    float heap_d2[kQueryK];   // per-thread bounded max-heap, in registers/local memory
    int   heap_id[kQueryK];
    int heap_size = 0;

    int stack[kBvhStackSize];
    int sp = 0;
    if (n >= 1) stack[sp++] = 0;

    int visited = 0;
    int stack_hwm = sp;

    while (sp > 0) {
        const int node_idx = stack[--sp];
        ++visited;
        const LbvhNode node = nodes[node_idx];

        // Prune radius: +infinity until the heap is full (see header
        // comment) — implemented by only applying the test when full.
        if (heap_size == kQueryK) {
            if (d_aabb_min_dist2(node.aabb_min, node.aabb_max, qp) > heap_d2[0]) {
                continue;
            }
        }

        if (d_is_leaf_node(node_idx, n)) {
            const float pp[3] = { xyz[node.point_idx * 3 + 0], xyz[node.point_idx * 3 + 1], xyz[node.point_idx * 3 + 2] };
            const float d2 = d_squared_distance3(pp, qp);
            const int id = node.point_idx;

            if (heap_size < kQueryK) {
                // Plain insert: append then sift UP until the max-heap
                // property (parent >= both children, under knn_less) holds.
                int c = heap_size++;
                heap_d2[c] = d2; heap_id[c] = id;
                while (c > 0) {
                    const int parent = (c - 1) / 2;
                    if (d_knn_less(heap_d2[parent], heap_id[parent], heap_d2[c], heap_id[c])) {
                        // parent is "less" (better) than child -> child is
                        // worse than parent, which VIOLATES max-heap order
                        // (root must be the WORST) -> swap and continue up.
                        float td = heap_d2[parent]; heap_d2[parent] = heap_d2[c]; heap_d2[c] = td;
                        int ti = heap_id[parent]; heap_id[parent] = heap_id[c]; heap_id[c] = ti;
                        c = parent;
                    } else break;
                }
            } else if (d_knn_less(d2, id, heap_d2[0], heap_id[0])) {
                // Candidate beats the current worst: replace root, sift DOWN.
                heap_d2[0] = d2; heap_id[0] = id;
                int c = 0;
                while (true) {
                    const int l = 2 * c + 1, rr = 2 * c + 2;
                    int worst = c;
                    if (l < kQueryK && d_knn_less(heap_d2[worst], heap_id[worst], heap_d2[l], heap_id[l])) worst = l;
                    if (rr < kQueryK && d_knn_less(heap_d2[worst], heap_id[worst], heap_d2[rr], heap_id[rr])) worst = rr;
                    if (worst == c) break;
                    float td = heap_d2[worst]; heap_d2[worst] = heap_d2[c]; heap_d2[c] = td;
                    int ti = heap_id[worst]; heap_id[worst] = heap_id[c]; heap_id[c] = ti;
                    c = worst;
                }
            }
        } else {
            if (sp + 2 <= kBvhStackSize) {
                stack[sp++] = node.left;
                stack[sp++] = node.right;
                if (sp > stack_hwm) stack_hwm = sp;
            }
        }
    }

    // Sort the final (<=kQueryK) heap contents ascending by knn_less — a
    // plain insertion sort (kQueryK==8: trivially cheap) so the output is
    // in the CANONICAL order every CPU/brute-force twin also produces.
    for (int a = 1; a < heap_size; ++a) {
        const float kd = heap_d2[a]; const int ki = heap_id[a];
        int b = a - 1;
        while (b >= 0 && d_knn_less(kd, ki, heap_d2[b], heap_id[b])) {
            heap_d2[b + 1] = heap_d2[b]; heap_id[b + 1] = heap_id[b];
            --b;
        }
        heap_d2[b + 1] = kd; heap_id[b + 1] = ki;
    }

    for (int a = 0; a < heap_size; ++a) {
        out_ids[q * kQueryK + a]   = heap_id[a];
        out_dist2[q * kQueryK + a] = heap_d2[a];
    }
    out_found[q] = heap_size;
    out_nodes_visited[q] = visited;
    out_stack_hwm[q] = stack_hwm;
}

void launch_knn_search_bvh(const LbvhNode* d_nodes, int n, const float* d_xyz,
                           const float* d_queries, int num_queries,
                           int* d_out_ids, float* d_out_dist2, int* d_out_found,
                           int* d_out_nodes_visited, int* d_out_stack_hwm)
{
    const int block = kThreadsPerBlock;
    const int grid  = blocks_for(num_queries, block);
    knn_search_bvh_kernel<<<grid, block>>>(d_nodes, n, d_xyz, d_queries, num_queries,
                                           d_out_ids, d_out_dist2, d_out_found, d_out_nodes_visited,
                                           d_out_stack_hwm);
    CUDA_CHECK_LAST_ERROR("knn_search_bvh_kernel launch");
}

// launch_sort_augmented_keys — Stage 2's GPU sort. thrust::sort dispatches
// a RADIX sort for unsigned integer keys (repeated stable partitioning by a
// few bits at a time, LSB to MSB — the same algorithm family 02.01's
// kernels.cu explains for thrust::stable_sort_by_key, here with no paired
// "value" array because the augmented key's low 32 bits ALREADY are the
// value we need after sorting — kernels.cuh's augmented_key comment names
// this simplification explicitly).
void launch_sort_augmented_keys(int n, unsigned long long* d_keys_sorted_inout)
{
    thrust::device_ptr<unsigned long long> keys_ptr(d_keys_sorted_inout);
    thrust::sort(keys_ptr, keys_ptr + n);
}

// ===========================================================================
// THE DOMAIN CONTRAST — fixed-radius voxel-hash search (02.01 Method B /
// 02.04 lineage, cited and reimplemented compactly; see kernels.cuh's file
// header "THE DOMAIN CONTRAST" for what this technique can and cannot do).
// ===========================================================================

__device__ __forceinline__ int32_t d_hash_voxel_coord(float p, float leaf)
{
    return static_cast<int32_t>(floorf(p / leaf));
}

// Same 21-bit-per-axis biased packing as 02.01/02.04's kernels.cuh (cited,
// re-derived here rather than shared across project folders per the
// self-containment rule — CLAUDE.md §4).
constexpr int32_t  kHashCoordBias   = 1 << 20;
constexpr uint64_t kHashCoordMask21 = (1ull << 21) - 1ull;

__device__ __forceinline__ uint64_t d_pack_hash_key(int32_t vx, int32_t vy, int32_t vz)
{
    const uint64_t ux = static_cast<uint64_t>(vx + kHashCoordBias) & kHashCoordMask21;
    const uint64_t uy = static_cast<uint64_t>(vy + kHashCoordBias) & kHashCoordMask21;
    const uint64_t uz = static_cast<uint64_t>(vz + kHashCoordBias) & kHashCoordMask21;
    return ux | (uy << 21) | (uz << 42);
}

__global__ void compute_hash_keys_kernel(int n, const float* __restrict__ xyz,
                                         float leaf, unsigned long long* __restrict__ keys)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int32_t vx = d_hash_voxel_coord(xyz[i * 3 + 0], leaf);
    const int32_t vy = d_hash_voxel_coord(xyz[i * 3 + 1], leaf);
    const int32_t vz = d_hash_voxel_coord(xyz[i * 3 + 2], leaf);
    keys[i] = d_pack_hash_key(vx, vy, vz);
}

void launch_compute_hash_keys(int n, const float* d_xyz, float leaf, unsigned long long* d_keys)
{
    const int block = kThreadsPerBlock;
    const int grid  = blocks_for(n, block);
    compute_hash_keys_kernel<<<grid, block>>>(n, d_xyz, leaf, d_keys);
    CUDA_CHECK_LAST_ERROR("compute_hash_keys_kernel launch");
}

// mark_boundaries_kernel — verbatim reuse of 02.01 Method B's kernel (see
// that project's kernels.cu for the full comment): position 0, or any
// position whose key differs from its predecessor, starts a new voxel run.
__global__ void mark_boundaries_kernel(int n, const unsigned long long* __restrict__ keys_sorted,
                                       int* __restrict__ is_start)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    is_start[i] = (i == 0 || keys_sorted[i] != keys_sorted[i - 1]) ? 1 : 0;
}

// gather_unique_keys_kernel — one thread per occupied voxel v: copy that
// voxel's shared key (every point in its run has the same key, by
// construction) into the dense unique_key[] array radius_search_hash_kernel
// binary-searches over (02.04's identical gather_unique_keys_kernel, cited).
// Defined HERE, ahead of launch_build_voxel_index's call site, rather than
// forward-declared with a local `extern __global__` — nvcc's device-code
// name mangling is fussier about matching a forward declaration to its
// later definition across a kernel launch than plain host C++ is, so this
// repo's convention (matching every other file in the project) is simply
// to order kernel DEFINITIONS before the host wrapper that launches them.
__global__ void gather_unique_keys_kernel(int num_voxels, const int* __restrict__ seg_start,
                                          const unsigned long long* __restrict__ keys_sorted,
                                          unsigned long long* __restrict__ unique_key_out)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_voxels) return;
    unique_key_out[v] = keys_sorted[seg_start[v]];
}

int launch_build_voxel_index(int n, const unsigned long long* d_keys_in,
                             unsigned long long* d_keys_scratch, int* d_idx_scratch,
                             int* d_is_start_scratch, int* d_seg_start_out,
                             unsigned long long* d_unique_key_out)
{
    // Copy keys into scratch (Thrust sorts in place; d_keys_in stays intact
    // for any caller that still needs the un-sorted, per-point key array).
    CUDA_CHECK(cudaMemcpy(d_keys_scratch, d_keys_in, static_cast<size_t>(n) * sizeof(unsigned long long),
                          cudaMemcpyDeviceToDevice));

    thrust::device_ptr<unsigned long long> keys_ptr(d_keys_scratch);
    thrust::device_ptr<int> idx_ptr(d_idx_scratch);
    thrust::sequence(idx_ptr, idx_ptr + n);   // idx[i]=i, the identity permutation before sorting

    // thrust::stable_sort_by_key: radix-sorts the 64-bit voxel keys
    // ascending, carrying idx along as the paired permutation (02.01's
    // kernels.cu explains what "radix sort" computes and why STABILITY
    // matters when the key itself is NOT unique — unlike Stage 2's
    // augmented-key sort above, MANY points legitimately share one voxel
    // key here, so stability's "equal keys keep input order" guarantee is
    // load-bearing this time, not just harmless).
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

    // unique_key[v] = the (single, shared) key of every point in voxel v's
    // run = keys_sorted[seg_start[v]] — gathered on the HOST side by the
    // caller is possible too, but doing it here with one more small kernel
    // keeps everything on-device between launches (avoids a round-trip).
    {
        // thrust::gather semantics via a tiny lambda-free kernel would add
        // library ceremony for a one-line copy; a hand-rolled kernel
        // (defined above, ahead of this call site) is simpler and just as
        // clear (CLAUDE.md "no black boxes").
        const int block = kThreadsPerBlock;
        const int grid  = blocks_for(num_voxels, block);
        gather_unique_keys_kernel<<<grid, block>>>(num_voxels, d_seg_start_out, d_keys_scratch, d_unique_key_out);
        CUDA_CHECK_LAST_ERROR("gather_unique_keys_kernel launch");
    }

    return num_voxels;
}

// d_lower_bound — the smallest index in unique_key[0,count) whose value is
// >= target (standard binary search; 02.04's d_lower_bound, cited and
// reused verbatim: this project's fixed-radius voxel-hash baseline needs
// the exact same "is this neighbor voxel occupied?" query 02.04's
// build_edges_kernel needs for its own 27-cell stencil).
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

// ---------------------------------------------------------------------------
// radius_search_hash_kernel — one thread per query: the voxel-hash
// counterpart of radius_search_bvh_kernel, over the SAME queries and the
// SAME radius, so main.cu's GATE hash_vs_bvh_agreement is a fair, apples-
// to-apples comparison. leaf == radius (the caller passes kRadiusM for
// both), so the proof in 02.04's kClusterToleranceM comment applies
// UNCHANGED: any point within `radius` of the query lies in the query's
// own voxel or one of its 26 face/edge/corner neighbors — a 3x3x3 stencil,
// 27 candidate voxels, each resolved via ONE d_lower_bound call.
// ---------------------------------------------------------------------------
__global__ void radius_search_hash_kernel(const float* __restrict__ xyz,
                                          const unsigned long long* __restrict__ unique_key, int num_voxels,
                                          const int* __restrict__ seg_start,
                                          const int* __restrict__ idx_sorted, int n_sorted,
                                          const float* __restrict__ queries, int num_queries,
                                          float leaf, float radius,
                                          int* __restrict__ out_ids,
                                          int* __restrict__ out_count,
                                          int* __restrict__ out_overflow)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= num_queries) return;

    const float qp[3] = { queries[q * 3 + 0], queries[q * 3 + 1], queries[q * 3 + 2] };
    const float r2 = radius * radius;

    const int32_t cvx = d_hash_voxel_coord(qp[0], leaf);
    const int32_t cvy = d_hash_voxel_coord(qp[1], leaf);
    const int32_t cvz = d_hash_voxel_coord(qp[2], leaf);

    int count = 0;
    bool overflow = false;

    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const uint64_t key = d_pack_hash_key(cvx + dx, cvy + dy, cvz + dz);
                const int v = d_lower_bound(unique_key, num_voxels, key);
                if (v >= num_voxels || unique_key[v] != key) continue;   // this neighbor voxel is unoccupied

                const int begin = seg_start[v];
                const int end = (v + 1 < num_voxels) ? seg_start[v + 1] : n_sorted;
                for (int s = begin; s < end; ++s) {
                    const int p = idx_sorted[s];
                    const float pp[3] = { xyz[p * 3 + 0], xyz[p * 3 + 1], xyz[p * 3 + 2] };
                    if (d_squared_distance3(pp, qp) <= r2) {
                        if (count < kMaxRadiusResults) {
                            out_ids[q * kMaxRadiusResults + count] = p;
                            ++count;
                        } else {
                            overflow = true;
                        }
                    }
                }
            }
        }
    }

    // Canonicalize (ascending id) — the same reason radius_search_bvh_kernel does.
    for (int a = 1; a < count; ++a) {
        const int key = out_ids[q * kMaxRadiusResults + a];
        int b = a - 1;
        while (b >= 0 && out_ids[q * kMaxRadiusResults + b] > key) {
            out_ids[q * kMaxRadiusResults + b + 1] = out_ids[q * kMaxRadiusResults + b];
            --b;
        }
        out_ids[q * kMaxRadiusResults + b + 1] = key;
    }

    out_count[q] = count;
    out_overflow[q] = overflow ? 1 : 0;
}

void launch_radius_search_hash(const float* d_xyz,
                               const unsigned long long* d_unique_key, int num_voxels,
                               const int* d_seg_start, const int* d_idx_sorted, int n_sorted,
                               const float* d_queries, int num_queries, float leaf, float radius,
                               int* d_out_ids, int* d_out_count, int* d_out_overflow)
{
    const int block = kThreadsPerBlock;
    const int grid  = blocks_for(num_queries, block);
    radius_search_hash_kernel<<<grid, block>>>(d_xyz, d_unique_key, num_voxels, d_seg_start, d_idx_sorted, n_sorted,
                                               d_queries, num_queries, leaf, radius,
                                               d_out_ids, d_out_count, d_out_overflow);
    CUDA_CHECK_LAST_ERROR("radius_search_hash_kernel launch");
}
