// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 02.05
//                     (KD-tree or LBVH construction + KNN/radius search)
//
// WHY does a GPU repository ship a CPU implementation of everything?
// ------------------------------------------------------------------
// Two load-bearing reasons (CLAUDE.md §5): (1) the CORRECTNESS ORACLE — GPU
// code fails in ways CPU code cannot (wrong indexing, races, stale device
// memory) — a dead-simple sequential twin gives ground truth; (2) the
// TEACHING BASELINE — reading this file, then kernels.cu, shows exactly
// what parallelization changed.
//
// Independence ruling for THIS project (CLAUDE.md §5's general policy,
// applied concretely — read this before trusting any twin comparison here)
// ---------------------------------------------------------------------------
// This project has an unusually rich mix of shared vs. independent code,
// because it has TWO kinds of "oracle": twin re-implementations of the
// SAME algorithm (which prove the GPU is a faithful parallelization) and a
// completely algorithm-FREE brute-force oracle (which proves the algorithm
// itself, not just its parallelization, is correct). Both tiers exist
// because CLAUDE.md's ruling is explicit that a shared bug between a
// kernel and its twin is invisible to a twin-only comparison — this
// project's construction is intricate enough (Karras's range/split search)
// that a THIRD, structurally unrelated check is exactly the safety net the
// ruling asks for.
//
//   SHARED (single-sourced in kernels.cuh, data-layout/bit-arithmetic only,
//   never the interesting logic): morton_encode30, augmented_key,
//   clz64_portable, delta_lcp, squared_distance3, aabb_min_dist2,
//   knn_less. Each is a short, unambiguous FORMULA (there is exactly one
//   correct way to interleave bits, count leading zeros, or compute a
//   squared distance) — sharing them is the SAME choice 02.01 makes for
//   pack_voxel_key: a data-layout contract, not an algorithm.
//
//   INDEPENDENT (retyped below, calling NONE of kernels.cu's device code
//   and none of each other's twin): sort_keys_cpu (std::sort, an
//   algorithmically different sort than Thrust's GPU radix sort);
//   build_radix_tree_cpu (Karras's range/split SEARCH LOOPS, written fresh
//   here as a sequential host loop — only delta_lcp itself is shared, per
//   the rule above); propagate_aabb_cpu (a single-threaded iterative
//   post-order walk — NOT an atomic-flag race, a genuinely different
//   algorithm shape); radius_search_bvh_cpu / knn_search_bvh_cpu (their
//   own stack/heap traversal loops, retyped — mirroring 11.01's
//   lidar_raycast_cpu / intersect_bvh independence exactly);
//   radius_search_hash_cpu (an std::unordered_map voxel map — a genuinely
//   DIFFERENT data structure than the GPU's sorted-array+binary-search
//   index, the same choice 02.04's build_edges_cpu makes).
//
//   THE THIRD TIER — brute force (radius_search_brute_force /
//   knn_search_brute_force): no tree, no hash, no sorted array at all —
//   a linear scan over every point. This is the ONE function in the whole
//   project that shares literally nothing with either the GPU path or any
//   other CPU twin except the query-time distance formula and the KNN
//   tie-break order (both single-line, unambiguous). main.cu's GATE
//   brute_force_anchor is the independent gate the CLAUDE.md ruling
//   requires: even a conceptual bug present in BOTH the GPU kernel and its
//   CPU twin (e.g. a shared misreading of Karras's algorithm) would still
//   be caught here, because this function never calls anything related to
//   tree construction at all.
//
// Rules for this file: plain C++17, no CUDA headers, no OpenMP, no
// cleverness — clarity beats speed here, always (CLAUDE.md §5).
//
// Read this after: kernels.cu — then compare the two side by side.
// ===========================================================================

#include "kernels.cuh"
#include <cstdint>
#include <algorithm>
#include <vector>
#include <unordered_map>

// ===========================================================================
// Stage 1 twin — shared formula, per the ruling above.
// ===========================================================================
void compute_augmented_keys_cpu(int n, const float* xyz, const SceneAABB& aabb,
                                unsigned long long* keys_out)
{
    for (int i = 0; i < n; ++i) {
        const uint32_t morton = morton_encode30(xyz[i * 3 + 0], xyz[i * 3 + 1], xyz[i * 3 + 2], aabb);
        keys_out[i] = augmented_key(morton, i);
    }
}

// ===========================================================================
// Stage 2 twin — std::sort (introsort: quicksort + heapsort fallback + a
// final insertion-sort pass), a DIFFERENT algorithm family than Thrust's
// GPU radix sort. Because augmented keys are pairwise distinct (every
// point's own index lives in the low 32 bits), the sorted order is the
// UNIQUE correct total order regardless of which correct sorting algorithm
// produces it — so std::sort (not even stable_sort) is legitimate here,
// unlike the voxel-hash keys below where genuine ties exist.
// ===========================================================================
void sort_keys_cpu(int n, const unsigned long long* keys_in, unsigned long long* keys_sorted_out)
{
    std::copy(keys_in, keys_in + n, keys_sorted_out);
    std::sort(keys_sorted_out, keys_sorted_out + n);
}

// ===========================================================================
// Stage 3 twin — Karras radix-tree construction, retyped as a plain
// sequential loop (INDEPENDENT of build_radix_tree_kernel's device code;
// see the file header ruling). Same two-part algorithm (determineRange,
// findSplit), same duplicate-key simplification (augmented keys are always
// distinct, so the textbook firstCode==lastCode branch is dead code here
// too) — written fresh, not copy-pasted from kernels.cu, so a coding
// mistake in one is unlikely to reproduce identically in the other.
// ===========================================================================
void build_radix_tree_cpu(int n, const unsigned long long* sorted_key, LbvhNode* nodes)
{
    if (n < 2) return;   // no internal nodes exist for a 0- or 1-point cloud

    for (int i = 0; i < n - 1; ++i) {
        // Part A: determineRange.
        const int delta_right = delta_lcp(sorted_key, n, i, i + 1);
        const int delta_left  = delta_lcp(sorted_key, n, i, i - 1);
        const int d = (delta_right - delta_left >= 0) ? 1 : -1;

        const int delta_min = delta_lcp(sorted_key, n, i, i - d);
        int l_max = 2;
        while (delta_lcp(sorted_key, n, i, i + l_max * d) > delta_min) {
            l_max *= 2;
        }
        int l = 0;
        for (int t = l_max / 2; t >= 1; t /= 2) {
            if (delta_lcp(sorted_key, n, i, i + (l + t) * d) > delta_min) {
                l += t;
            }
        }
        const int j = i + l * d;
        const int first = (d > 0) ? i : j;
        const int last  = (d > 0) ? j : i;

        // Part B: findSplit.
        const int common_prefix = delta_lcp(sorted_key, n, first, last);
        int split = first;
        int step = last - first;
        do {
            step = (step + 1) / 2;
            const int new_split = split + step;
            if (new_split < last) {
                const int split_prefix = delta_lcp(sorted_key, n, first, new_split);
                if (split_prefix > common_prefix) {
                    split = new_split;
                }
            }
        } while (step > 1);

        const int left_child  = (split == first)    ? (n - 1 + split)     : split;
        const int right_child = (split + 1 == last) ? (n - 1 + split + 1) : (split + 1);

        nodes[i].left = left_child;
        nodes[i].right = right_child;
        nodes[left_child].parent = i;
        nodes[right_child].parent = i;
    }
}

void init_leaves_cpu(int n, const float* xyz, const unsigned long long* sorted_key, LbvhNode* nodes)
{
    for (int k = 0; k < n; ++k) {
        const int point_idx = decode_point_index(sorted_key[k]);
        const int node_idx = (n - 1) + k;
        LbvhNode& leaf = nodes[node_idx];
        leaf.aabb_min[0] = leaf.aabb_max[0] = xyz[point_idx * 3 + 0];
        leaf.aabb_min[1] = leaf.aabb_max[1] = xyz[point_idx * 3 + 1];
        leaf.aabb_min[2] = leaf.aabb_max[2] = xyz[point_idx * 3 + 2];
        leaf.left = -1;
        leaf.right = -1;
        leaf.parent = -1;
        leaf.point_idx = point_idx;
    }
}

// ===========================================================================
// Stage 4 twin — propagate_aabb_cpu: an ITERATIVE POST-ORDER traversal
// (an explicit stack of (node, child-visited-count) pairs — no recursion,
// so this scales to this project's ~200k-leaf trees without risking a
// C++ call-stack overflow) — a genuinely different ALGORITHM SHAPE from
// the GPU's atomic-flag race (no atomics, no race, single thread, visits
// every node exactly once in dependency order instead of "whichever
// thread arrives second"). Both are correct because AABB union is
// associative/commutative — see the header comment on VERIFY(aabb) in
// kernels.cuh for the exact-equality argument (extending 02.02's AABB
// order-independence lineage: float min/max never rounds, so the RESULT
// depends only on the SET of points in a subtree, never on the order they
// were combined in).
// ===========================================================================
void propagate_aabb_cpu(int n, LbvhNode* nodes)
{
    if (n < 2) return;   // single-leaf degenerate tree already has its AABB from init_leaves_cpu

    // visited_children[i] counts how many of internal node i's two children
    // have had their OWN AABB finalized — node i's AABB can only be
    // computed once this reaches 2 (both children ready), mirroring the
    // GPU's "second arrival" condition without any atomics.
    std::vector<int> visited_children(static_cast<size_t>(n - 1), 0);

    // Start the walk from every LEAF (exactly like the GPU: N independent
    // starting points), climbing toward the root. A leaf's AABB is already
    // final (init_leaves_cpu), so this is a plain sequential loop with no
    // stack of its own beyond the implicit "climb parent pointers" loop.
    for (int k = 0; k < n; ++k) {
        int node = (n - 1) + k;
        while (true) {
            const int parent = nodes[node].parent;
            if (parent < 0) break;   // reached the root

            visited_children[parent]++;
            if (visited_children[parent] < 2) {
                break;   // sibling not finalized yet from THIS traversal's perspective
            }

            // Both children finalized: compute this node's AABB now.
            const int lc = nodes[parent].left;
            const int rc = nodes[parent].right;
            for (int a = 0; a < 3; ++a) {
                nodes[parent].aabb_min[a] = std::min(nodes[lc].aabb_min[a], nodes[rc].aabb_min[a]);
                nodes[parent].aabb_max[a] = std::max(nodes[lc].aabb_max[a], nodes[rc].aabb_max[a]);
            }
            node = parent;   // continue climbing
        }
    }
}

// ===========================================================================
// Stage 5a twin — radius_search_bvh_cpu: an independent sequential stack
// traversal (11.01's lidar_raycast_cpu / intersect_bvh pairing, applied
// here to a sphere-overlap test instead of ray-slab intersection).
// ===========================================================================
void radius_search_bvh_cpu(const LbvhNode* nodes, int n, const float* xyz,
                           const float* query, float radius,
                           std::vector<int>& out_ids, bool& out_overflow)
{
    out_ids.clear();
    out_overflow = false;
    if (n < 1) return;

    const float r2 = radius * radius;
    std::vector<int> stack;
    stack.reserve(64);
    stack.push_back(0);   // root

    while (!stack.empty()) {
        const int node_idx = stack.back();
        stack.pop_back();
        const LbvhNode& node = nodes[node_idx];

        if (aabb_min_dist2(node.aabb_min, node.aabb_max, query) > r2) {
            continue;
        }

        if (is_leaf_node(node_idx, n)) {
            const float pp[3] = { xyz[node.point_idx * 3 + 0], xyz[node.point_idx * 3 + 1], xyz[node.point_idx * 3 + 2] };
            if (squared_distance3(pp, query) <= r2) {
                if (static_cast<int>(out_ids.size()) < kMaxRadiusResults) {
                    out_ids.push_back(node.point_idx);
                } else {
                    out_overflow = true;
                }
            }
        } else {
            stack.push_back(node.left);
            stack.push_back(node.right);
        }
    }

    std::sort(out_ids.begin(), out_ids.end());   // canonical order, matching the GPU's insertion-sorted output
}

// ===========================================================================
// Stage 5b twin — knn_search_bvh_cpu: an independent bounded max-heap
// traversal. Uses std::vector as the heap storage (no fixed-size register
// array needed on the host) but the SAME knn_less order and the SAME
// shrinking-radius pruning idea, written as its own insert/replace logic
// (not calling the GPU's sift-up/down code).
// ===========================================================================
void knn_search_bvh_cpu(const LbvhNode* nodes, int n, const float* xyz,
                        const float* query, std::vector<KnnCandidate>& out_sorted)
{
    out_sorted.clear();
    if (n < 1) return;

    // A simple comparator adapting knn_less to std::push_heap/pop_heap's
    // "max at front" convention: std::*_heap treats `comp` as "a should be
    // popped AFTER b" (i.e. comp(a,b) true means a is LOWER priority) — we
    // want the WORST candidate at the top for O(log K) replace, so we hand
    // it knn_less directly (a "less" candidate is lower priority to KEEP
    // as the max, matching std::*_heap's usual max-heap convention).
    auto heap_cmp = [](const KnnCandidate& a, const KnnCandidate& b) {
        return knn_less(a, b);
    };

    std::vector<KnnCandidate> heap;
    heap.reserve(kQueryK);

    std::vector<int> stack;
    stack.reserve(64);
    stack.push_back(0);

    while (!stack.empty()) {
        const int node_idx = stack.back();
        stack.pop_back();
        const LbvhNode& node = nodes[node_idx];

        if (static_cast<int>(heap.size()) == kQueryK) {
            if (aabb_min_dist2(node.aabb_min, node.aabb_max, query) > heap.front().dist2) {
                continue;
            }
        }

        if (is_leaf_node(node_idx, n)) {
            const float pp[3] = { xyz[node.point_idx * 3 + 0], xyz[node.point_idx * 3 + 1], xyz[node.point_idx * 3 + 2] };
            KnnCandidate cand{ squared_distance3(pp, query), node.point_idx };

            if (static_cast<int>(heap.size()) < kQueryK) {
                heap.push_back(cand);
                std::push_heap(heap.begin(), heap.end(), heap_cmp);
            } else if (knn_less(cand, heap.front())) {
                std::pop_heap(heap.begin(), heap.end(), heap_cmp);
                heap.back() = cand;
                std::push_heap(heap.begin(), heap.end(), heap_cmp);
            }
        } else {
            stack.push_back(node.left);
            stack.push_back(node.right);
        }
    }

    out_sorted = heap;
    std::sort(out_sorted.begin(), out_sorted.end(),
             [](const KnnCandidate& a, const KnnCandidate& b) { return knn_less(a, b); });
}

// ===========================================================================
// Voxel-hash twin — shared key formula (data layout), independent radius
// query (an std::unordered_map, genuinely different data structure than
// the GPU's sorted-array+binary-search index — 02.04's build_edges_cpu
// precedent for this exact choice).
// ===========================================================================
void compute_hash_keys_cpu(int n, const float* xyz, float leaf, unsigned long long* keys_out)
{
    constexpr int32_t  kBias = 1 << 20;
    constexpr uint64_t kMask21 = (1ull << 21) - 1ull;
    for (int i = 0; i < n; ++i) {
        const int32_t vx = static_cast<int32_t>(std::floor(xyz[i * 3 + 0] / leaf));
        const int32_t vy = static_cast<int32_t>(std::floor(xyz[i * 3 + 1] / leaf));
        const int32_t vz = static_cast<int32_t>(std::floor(xyz[i * 3 + 2] / leaf));
        const uint64_t ux = static_cast<uint64_t>(vx + kBias) & kMask21;
        const uint64_t uy = static_cast<uint64_t>(vy + kBias) & kMask21;
        const uint64_t uz = static_cast<uint64_t>(vz + kBias) & kMask21;
        keys_out[i] = ux | (uy << 21) | (uz << 42);
    }
}

// voxel_hash_key_cpu — shared bit-packing formula, used by both the map
// builder and the query below (kept file-local since it is pure key
// arithmetic, the same category as kernels.cuh's shared helpers).
static uint64_t voxel_hash_key_cpu(int32_t vx, int32_t vy, int32_t vz)
{
    constexpr int32_t  kBias = 1 << 20;
    constexpr uint64_t kMask21 = (1ull << 21) - 1ull;
    const uint64_t ux = static_cast<uint64_t>(vx + kBias) & kMask21;
    const uint64_t uy = static_cast<uint64_t>(vy + kBias) & kMask21;
    const uint64_t uz = static_cast<uint64_t>(vz + kBias) & kMask21;
    return ux | (uy << 21) | (uz << 42);
}

// build_hash_map_cpu — populate the independent voxel->points map in ONE
// sequential O(n) pass — a different data structure AND a different
// construction order than the GPU's sort+segment index (02.04 lineage).
// Built exactly ONCE by main.cu, then queried by every one of the Q
// radius_search_hash_cpu calls below (never rebuilt per query — see this
// function's declaration comment in kernels.cuh for why that would be a
// real performance bug, not just an inefficiency).
void build_hash_map_cpu(int n, const float* xyz, float leaf, HashMapCpu& out_map)
{
    out_map.clear();
    out_map.reserve(static_cast<size_t>(n) / 4 + 1);
    for (int i = 0; i < n; ++i) {
        const int32_t vx = static_cast<int32_t>(std::floor(xyz[i * 3 + 0] / leaf));
        const int32_t vy = static_cast<int32_t>(std::floor(xyz[i * 3 + 1] / leaf));
        const int32_t vz = static_cast<int32_t>(std::floor(xyz[i * 3 + 2] / leaf));
        out_map[voxel_hash_key_cpu(vx, vy, vz)].push_back(i);
    }
}

void radius_search_hash_cpu(const HashMapCpu& map, float leaf, float radius,
                            const float* xyz, const float* query, std::vector<int>& out_ids)
{
    const int32_t cvx = static_cast<int32_t>(std::floor(query[0] / leaf));
    const int32_t cvy = static_cast<int32_t>(std::floor(query[1] / leaf));
    const int32_t cvz = static_cast<int32_t>(std::floor(query[2] / leaf));
    const float r2 = radius * radius;

    out_ids.clear();
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                auto it = map.find(voxel_hash_key_cpu(cvx + dx, cvy + dy, cvz + dz));
                if (it == map.end()) continue;
                for (int p : it->second) {
                    const float pp[3] = { xyz[p * 3 + 0], xyz[p * 3 + 1], xyz[p * 3 + 2] };
                    if (squared_distance3(pp, query) <= r2) {
                        out_ids.push_back(p);
                    }
                }
            }
        }
    }
    std::sort(out_ids.begin(), out_ids.end());
}

// ===========================================================================
// THE brute-force oracles — no tree, no hash, no sorted array: a linear
// scan over every one of the n points. This is the project's THIRD,
// structurally independent verification tier (file header). Deliberately
// the simplest, most obviously-correct code in the whole project.
// ===========================================================================
void radius_search_brute_force(int n, const float* xyz, const float* query, float radius,
                               std::vector<int>& out_ids)
{
    out_ids.clear();
    const float r2 = radius * radius;
    for (int i = 0; i < n; ++i) {
        const float pp[3] = { xyz[i * 3 + 0], xyz[i * 3 + 1], xyz[i * 3 + 2] };
        if (squared_distance3(pp, query) <= r2) {
            out_ids.push_back(i);
        }
    }
    std::sort(out_ids.begin(), out_ids.end());
}

void knn_search_brute_force(int n, const float* xyz, const float* query,
                            std::vector<KnnCandidate>& out_sorted)
{
    // Compute every distance, then partial-sort the smallest kQueryK by the
    // shared knn_less order — O(n log n) here (a plain full sort; n is
    // small enough per call that std::partial_sort's extra complexity
    // would only obscure the point: this function's entire REASON to exist
    // is to be too simple to hide a bug in, not to be fast).
    std::vector<KnnCandidate> all;
    all.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const float pp[3] = { xyz[i * 3 + 0], xyz[i * 3 + 1], xyz[i * 3 + 2] };
        all.push_back(KnnCandidate{ squared_distance3(pp, query), i });
    }
    std::sort(all.begin(), all.end(), [](const KnnCandidate& a, const KnnCandidate& b) { return knn_less(a, b); });

    const int k = std::min(static_cast<int>(all.size()), kQueryK);
    out_sorted.assign(all.begin(), all.begin() + k);
}
