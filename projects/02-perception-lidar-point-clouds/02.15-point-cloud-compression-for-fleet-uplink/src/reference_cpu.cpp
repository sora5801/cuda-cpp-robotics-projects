// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 02.15
//                     (Point cloud compression (octree/entropy) for fleet uplink)
//
// WHY does a GPU repository ship a CPU implementation of everything?
// ------------------------------------------------------------------
// Two load-bearing reasons (CLAUDE.md §5): (1) the CORRECTNESS ORACLE — GPU
// code fails in ways CPU code cannot (wrong indexing, races, stale device
// memory) — a dead-simple sequential twin gives ground truth; (2) the
// TEACHING BASELINE — reading this file, then kernels.cu, shows exactly
// what parallelization changed.
//
// Independence ruling for THIS project (docs/PROJECT_TEMPLATE's general
// policy, applied concretely — read this before trusting any comparison here)
// ---------------------------------------------------------------------------
//   SHARED (single-sourced in kernels.cuh, data-layout/format formulas only,
//   never the interesting logic): point_to_code / quantize_axis /
//   morton_encode (there is exactly one correct way to interleave bits given
//   a fixed axis-to-bit convention — the same class as 02.05's
//   morton_encode30), node_prefix / child_octant (index arithmetic derived
//   from that same convention), decode_leaf_center (the inverse formula),
//   huffman_merge_key (a documented deterministic TIE-BREAK total order, not
//   an algorithm — the same category as 02.05's knn_less), and the MSB-first
//   bit-packing FORMAT convention (documented in kernels.cuh's HuffmanTable
//   comment; each side below implements its own encoder/decoder against
//   that shared spec, exactly as 02.02's point-cloud layout is a shared
//   CONTRACT, not shared CODE).
//
//   INDEPENDENT (retyped below, calling none of kernels.cu's device code and
//   none of each other's twin):
//     - sort_codes_cpu: std::sort, algorithmically different from Thrust's
//       GPU radix/merge sort.
//     - build_octree_levels_cpu: ONE sequential pass over the sorted codes
//       with a plain per-level running accumulator (current node's prefix +
//       occupancy byte) — no scan, no atomics, no node_id array at all — a
//       structurally different algorithm from kernels.cu's
//       flags -> scan -> atomicOr pipeline, even though both walk the same
//       sorted array once per level.
//     - build_huffman_table_cpu: the SAME canonical-Huffman algorithm as
//       kernels.cu's build_huffman_table, but a genuinely different
//       SELECTION strategy (a plain O(k^2) linear scan for the two smallest
//       remaining frequencies, vs. the GPU-side builder's binary heap) —
//       apt at k<=256 symbols, and independent in the sense that matters:
//       a bug in the heap's comparator would not also live in a linear scan.
//     - huffman_encode_cpu: a plain SEQUENTIAL bit writer (no scan, no
//       parallel-shaped anything) — structurally different from the GPU's
//       map+scan+scatter, sharing only the documented bit-packing FORMAT.
//
//   THE DECODER (huffman_decode_cpu, octree_decode_leaf_centers) has NO GPU
//   counterpart at all (kernels.cuh's file header explains why decode is
//   host-only by design: the serial dependency chain a variable-length
//   prefix code imposes has no scan-shaped parallel form). A component with
//   no twin to compare against is verified DIFFERENTLY: main.cu's GATE
//   lossless_roundtrip checks decode(encode(cloud)) END TO END against an
//   INDEPENDENT ground truth (octree_unique_leaf_codes below — a trivial
//   dedup pass over the sorted array, sharing no code with the decoder) —
//   the same "a third, structurally unrelated check" discipline 02.05's
//   brute-force oracle applies to its own hardest-to-verify stage.
//
// Rules for this file: plain C++17, no CUDA headers, no OpenMP, no
// cleverness — clarity beats speed here, always (CLAUDE.md §5).
//
// Read this after: kernels.cu — then compare the two side by side.
// ===========================================================================

#include "kernels.cuh"
#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

// ===========================================================================
// Stage 1 twins — shared formula (point_to_code), independent sort.
// ===========================================================================

void compute_codes_cpu(int n, const float* xyz, const SceneAABB& aabb, int D,
                       uint64_t* codes_out)
{
    for (int i = 0; i < n; ++i) {
        codes_out[i] = point_to_code(xyz[i * 3 + 0], xyz[i * 3 + 1], xyz[i * 3 + 2], aabb, D);
    }
}

void sort_codes_cpu(int n, const uint64_t* codes_in, uint64_t* codes_sorted_out)
{
    std::copy(codes_in, codes_in + n, codes_sorted_out);
    std::sort(codes_sorted_out, codes_sorted_out + n);   // a DIFFERENT sort algorithm than Thrust's GPU radix sort
}

// ---------------------------------------------------------------------------
// build_octree_levels_cpu — the independent, sequential twin of
// launch_build_octree_level's whole per-depth sweep (kernels.cu section D).
//
// Big idea: walk the sorted codes ONCE, and for EVERY level simultaneously
// (not one level at a time, unlike the GPU path, which reruns a fresh
// flags/scan/atomicOr pass per level — see kernels.cu's per-level design
// note), track the CURRENT node's identity (its prefix) and its occupancy
// byte with a plain running accumulator: when a new point's prefix differs
// from the level's current node, the current node is "closed" (push its
// finished byte) and a fresh one begins at value 0. Because the array is
// Morton-SORTED, a level's node boundaries are visited in exactly this
// left-to-right order — no lookahead, no second pass, no data structure
// beyond one small per-level "current prefix" scalar (current_prefix[D]).
//
// Complexity: O(n*D) — the same total work as the GPU path (D flag/scan/
// occupancy passes of O(n) each), done here as ONE pass with D units of
// work per point instead of D passes with 1 unit of work per point per
// pass. Output: occ_per_level[level] holds that level's occupancy bytes in
// ascending-node (== ascending-Morton-prefix) order — VERIFY(octree_levels)
// in main.cu compares this, level by level, against the GPU's uint32_t
// occupancy words (narrowed to bytes) for bit-exact agreement.
// ---------------------------------------------------------------------------
void build_octree_levels_cpu(int n, const uint64_t* sorted_codes, int D,
                             std::vector<std::vector<uint8_t>>& occ_per_level)
{
    occ_per_level.assign(static_cast<size_t>(D), {});
    // current_prefix[level] — the node currently being accumulated at that
    // level; -1 (never a valid prefix, since prefixes are unsigned) means
    // "no node opened yet" (only true before the very first point).
    std::vector<int64_t> current_prefix(static_cast<size_t>(D), -1);

    for (int i = 0; i < n; ++i) {
        const uint64_t code = sorted_codes[i];
        for (int level = 0; level < D; ++level) {
            const int shift = 3 * (D - level);                    // this level's prefix width in bits, complemented
            const uint64_t prefix = code >> shift;                // node_prefix(code, D, level)
            if (static_cast<int64_t>(prefix) != current_prefix[static_cast<size_t>(level)]) {
                occ_per_level[static_cast<size_t>(level)].push_back(0);   // open a fresh node's byte
                current_prefix[static_cast<size_t>(level)] = static_cast<int64_t>(prefix);
            }
            const uint32_t oct = static_cast<uint32_t>((code >> (shift - 3)) & 7ull);   // child_octant(code,D,level)
            occ_per_level[static_cast<size_t>(level)].back() |= static_cast<uint8_t>(1u << oct);
        }
    }
}

// ===========================================================================
// Stage 2 twins.
// ===========================================================================

void compute_histogram_cpu(int m, const uint8_t* symbols, int hist_out[256])
{
    for (int s = 0; s < 256; ++s) hist_out[s] = 0;
    for (int i = 0; i < m; ++i) hist_out[symbols[i]]++;
}

// ---------------------------------------------------------------------------
// build_huffman_table_cpu — INDEPENDENT twin of kernels.cu's
// build_huffman_table: the SAME canonical-Huffman algorithm (shared
// huffman_merge_key tie-break, shared canonical bit-assignment FORMULA —
// both are documented total-order/format contracts, not "the algorithm",
// per the ruling above), but merge SELECTION is a plain O(k^2) linear scan
// over the still-active node list instead of a binary heap — genuinely
// different code, apt at k<=256 observed symbols where O(k^2)=65,536 worst
// case is instant. If a bug ever lived in the HEAP's comparator plumbing
// (a classic C++ std::priority_queue footgun — min vs max heap direction,
// stale top() after pop()), this independent implementation would not
// reproduce it, so VERIFY(huffman_table)'s bit-for-bit table comparison is
// a genuine check, not a tautology.
// ---------------------------------------------------------------------------
void build_huffman_table_cpu(const int hist[256], HuffmanTable& out)
{
    for (int s = 0; s < 256; ++s) { out.len[s] = 0; out.bits[s] = 0; }

    std::vector<int> parent(511, -1);
    std::vector<long long> freq(511, 0);
    std::vector<int> active;   // node ids not yet merged into a parent

    int num_observed = 0;
    int only_symbol = -1;
    for (int s = 0; s < 256; ++s) {
        if (hist[s] > 0) {
            freq[static_cast<size_t>(s)] = hist[s];
            active.push_back(s);
            ++num_observed;
            only_symbol = s;
        }
    }
    if (num_observed == 0) return;
    if (num_observed == 1) {
        // Same degenerate-single-symbol fix as build_huffman_table (kernels.cu):
        // force a 1-bit code — a real bitstream cannot emit 0 bits per symbol.
        out.len[only_symbol] = 1;
        out.bits[only_symbol] = 0;
        return;
    }

    int next_internal_id = 256;
    int root_id = -1;
    while (active.size() > 1) {
        // Linear scan for the two smallest by huffman_merge_key — O(k) per
        // merge, O(k^2) total; a DIFFERENT selection mechanism than the
        // GPU-path builder's heap, per the independence ruling above.
        size_t best_pos = 0, second_pos = 1;
        uint64_t best_key = huffman_merge_key(static_cast<int>(freq[static_cast<size_t>(active[0])]), active[0]);
        uint64_t second_key = huffman_merge_key(static_cast<int>(freq[static_cast<size_t>(active[1])]), active[1]);
        if (second_key < best_key) { std::swap(best_key, second_key); std::swap(best_pos, second_pos); }
        for (size_t k = 2; k < active.size(); ++k) {
            const uint64_t key = huffman_merge_key(static_cast<int>(freq[static_cast<size_t>(active[k])]), active[k]);
            if (key < best_key) { second_key = best_key; second_pos = best_pos; best_key = key; best_pos = k; }
            else if (key < second_key) { second_key = key; second_pos = k; }
        }
        const int id_a = active[best_pos];
        const int id_b = active[second_pos];
        // Erase the larger index first so the smaller index stays valid.
        const size_t hi = std::max(best_pos, second_pos);
        const size_t lo = std::min(best_pos, second_pos);
        active.erase(active.begin() + static_cast<long>(hi));
        active.erase(active.begin() + static_cast<long>(lo));

        const int id = next_internal_id++;
        const long long f = freq[static_cast<size_t>(id_a)] + freq[static_cast<size_t>(id_b)];
        freq[static_cast<size_t>(id)] = f;
        parent[static_cast<size_t>(id_a)] = id;
        parent[static_cast<size_t>(id_b)] = id;
        active.push_back(id);
        root_id = id;
    }

    std::vector<std::pair<int, int>> by_len_then_symbol;   // (length, symbol)
    for (int s = 0; s < 256; ++s) {
        if (hist[s] <= 0) continue;
        int depth = 0;
        int cur = s;
        while (cur != root_id) { cur = parent[static_cast<size_t>(cur)]; ++depth; }
        by_len_then_symbol.push_back({ depth, s });
    }
    std::sort(by_len_then_symbol.begin(), by_len_then_symbol.end());

    // Canonical code assignment — the SAME documented formula
    // build_huffman_table (kernels.cu) uses: given only the sorted
    // (length, symbol) list, this step is fully determined, which is
    // exactly the property VERIFY(huffman_table) exercises.
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

// ---------------------------------------------------------------------------
// huffman_encode_cpu — INDEPENDENT twin of launch_huffman_encode: a plain
// sequential bit writer. No scan, no bit-offset precomputation — each
// symbol's code bits are appended directly to a running byte accumulator,
// MSB-first (the shared FORMAT convention), flushing a completed byte the
// moment 8 bits have accumulated. This is the textbook serial encoder every
// GPU pattern in kernels.cu's section E is a PARALLELIZATION of — reading
// the two side by side is the point (CLAUDE.md §5 "teaching baseline").
// ---------------------------------------------------------------------------
void huffman_encode_cpu(int m, const uint8_t* symbols, const HuffmanTable& table,
                        std::vector<uint8_t>& out_packed, long long& out_num_bits)
{
    out_packed.clear();
    uint8_t cur_byte = 0;
    int cur_bits = 0;
    long long total_bits = 0;

    for (int i = 0; i < m; ++i) {
        const int len = table.len[symbols[i]];
        const uint32_t code = table.bits[symbols[i]];
        for (int j = len - 1; j >= 0; --j) {          // MSB-first: bit (len-1) written first
            const int bit = (code >> j) & 1;
            cur_byte = static_cast<uint8_t>((cur_byte << 1) | bit);
            ++cur_bits;
            if (cur_bits == 8) {
                out_packed.push_back(cur_byte);
                cur_byte = 0;
                cur_bits = 0;
            }
        }
        total_bits += len;
    }
    if (cur_bits > 0) {
        cur_byte = static_cast<uint8_t>(cur_byte << (8 - cur_bits));   // pad the final partial byte with zero bits
        out_packed.push_back(cur_byte);
    }
    out_num_bits = total_bits;
}

// ===========================================================================
// The decoder — see this file's header and kernels.cuh's "Why decode is
// host-only" for why there is no GPU counterpart to compare against.
// ===========================================================================

namespace {

// HuffTrieNode — an explicit binary trie built from a HuffmanTable, used
// only by huffman_decode_cpu below. left/right are trie-node indices
// (-1 = absent); symbol is the decoded byte value once a leaf is reached
// (-1 = internal, not yet a complete code).
struct HuffTrieNode {
    int left = -1;
    int right = -1;
    int symbol = -1;
};

// build_huffman_trie — insert every observed symbol's code into a fresh
// bit-trie, walking MSB-first (matching huffman_encode_cpu's write order
// and bit_scatter_kernel's — the shared format). O(sum of code lengths),
// negligible next to decoding the actual stream.
//
// A real bug root-caused during this project's own development, worth
// leaving documented rather than silently fixed: an earlier version held
// `int& child = ...trie[cur].left/right;` — a REFERENCE into the vector —
// and then called `trie.push_back(...)` while that reference was still
// live. push_back may REALLOCATE the vector's backing storage, which
// invalidates every existing reference/pointer/iterator into it,
// `child` included; the subsequent `child = ...` then wrote through a
// dangling reference (undefined behavior — in practice, a silently
// dropped child pointer, leaving that trie slot at -1 forever). The
// symptom surfaced far downstream: `huffman_decode_cpu`'s bit-walk
// eventually followed a real encoded bit sequence into that never-linked
// -1 child and indexed the trie with it, an out-of-bounds access that
// crashed the whole demo. The fix below never keeps a reference alive
// across a push_back: it re-reads `trie[cur]` by INDEX after growing the
// vector, and writes the new child index through that fresh access.
std::vector<HuffTrieNode> build_huffman_trie(const HuffmanTable& table)
{
    std::vector<HuffTrieNode> trie(1);   // node 0 = the trie root
    for (int s = 0; s < 256; ++s) {
        const int len = table.len[s];
        if (len == 0) continue;
        int cur = 0;
        for (int j = len - 1; j >= 0; --j) {
            const int bit = (table.bits[s] >> j) & 1;
            int next = bit ? trie[static_cast<size_t>(cur)].right : trie[static_cast<size_t>(cur)].left;
            if (next == -1) {
                trie.push_back(HuffTrieNode{});             // may reallocate — no live references across this line
                next = static_cast<int>(trie.size()) - 1;
                if (bit) trie[static_cast<size_t>(cur)].right = next;   // re-indexed AFTER the growth, never a stale reference
                else     trie[static_cast<size_t>(cur)].left = next;
            }
            cur = next;
        }
        trie[static_cast<size_t>(cur)].symbol = s;
    }
    return trie;
}

} // namespace

// ---------------------------------------------------------------------------
// huffman_decode_cpu — see kernels.cuh's declaration and the file header's
// "Why decode is host-only": walk `trie` one bit at a time, MSB-first per
// byte (packed[]'s documented layout — the same convention every encoder
// above writes), descending left/right; each time a LEAF is reached, emit
// its symbol and restart from the root. The SERIAL DEPENDENCY this project
// keeps insisting on is right here in the code: bit i+1's meaning (which
// trie node it descends from) depends on every earlier bit having already
// been consumed — there is no way to start decoding symbol k without first
// knowing exactly how many bits symbols 0..k-1 consumed.
// ---------------------------------------------------------------------------
void huffman_decode_cpu(const std::vector<uint8_t>& packed, long long num_bits,
                        int m_symbols, const HuffmanTable& table,
                        std::vector<uint8_t>& out_symbols)
{
    const std::vector<HuffTrieNode> trie = build_huffman_trie(table);
    out_symbols.clear();
    out_symbols.reserve(static_cast<size_t>(m_symbols));

    int cur = 0;             // current trie node, starts at the root
    long long bitpos = 0;
    while (static_cast<int>(out_symbols.size()) < m_symbols && bitpos < num_bits) {
        const long long byte_idx = bitpos >> 3;
        const int bit_in_byte = 7 - static_cast<int>(bitpos & 7);   // MSB-first within the byte
        const int bit = (packed[static_cast<size_t>(byte_idx)] >> bit_in_byte) & 1;
        cur = bit ? trie[static_cast<size_t>(cur)].right : trie[static_cast<size_t>(cur)].left;
        ++bitpos;
        if (trie[static_cast<size_t>(cur)].symbol != -1) {
            out_symbols.push_back(static_cast<uint8_t>(trie[static_cast<size_t>(cur)].symbol));
            cur = 0;   // restart at the root for the next symbol (prefix-free code property)
        }
    }
}

// ---------------------------------------------------------------------------
// octree_decode_leaf_centers — see kernels.cuh's declaration: the
// level-by-level EXPANSION inverse of the level-by-level CONSTRUCTION
// (build_octree_levels_cpu above / kernels.cu section D). `parents` tracks
// every currently-active node's accumulated (ix,iy,iz) path bits; at each
// level we consume exactly len(parents) bytes from the stream (one per
// active node — the SAME count the encoder emitted at that level, because
// popcount of a byte at level l is exactly the number of level-(l+1) nodes
// that byte's node contributes — no side channel for "how many nodes at
// this level" is needed, it falls out of the stream itself) and expand each
// set bit into a child with one more path bit resolved.
// ---------------------------------------------------------------------------
void octree_decode_leaf_centers(const std::vector<uint8_t>& occupancy_stream, int D,
                                const SceneAABB& aabb,
                                std::vector<float>& out_leaf_xyz)
{
    struct Node { uint32_t ix = 0, iy = 0, iz = 0; };

    std::vector<Node> parents(1);   // the single implicit root, path bits all zero so far
    size_t stream_pos = 0;

    for (int level = 0; level < D; ++level) {
        std::vector<Node> children;
        children.reserve(parents.size() * 2);   // a loose but cheap upper bound (<=8x, usually far less)
        const int bit_pos = D - 1 - level;       // this level's octant bits land at THIS ix/iy/iz bit

        for (const Node& p : parents) {
            const uint8_t byte = occupancy_stream[stream_pos++];   // this parent's occupancy byte
            for (int c = 0; c < 8; ++c) {
                if (!(byte & (1u << c))) continue;   // child octant c absent
                Node child = p;
                if (c & 1) child.ix |= (1u << bit_pos);   // bit0 = x (morton_encode's convention)
                if (c & 2) child.iy |= (1u << bit_pos);   // bit1 = y
                if (c & 4) child.iz |= (1u << bit_pos);   // bit2 = z
                children.push_back(child);
            }
        }
        parents = std::move(children);
    }

    // `parents` now holds every LEAF's full-resolution integer path — convert
    // each to a world-frame center (decode_leaf_center's formula, inlined
    // here per-axis since we already have ix/iy/iz rather than a packed code).
    const float leaf_m = aabb.extent_m / static_cast<float>(1u << D);
    out_leaf_xyz.clear();
    out_leaf_xyz.reserve(parents.size() * 3);
    for (const Node& leaf : parents) {
        out_leaf_xyz.push_back(aabb.min[0] + (static_cast<float>(leaf.ix) + 0.5f) * leaf_m);
        out_leaf_xyz.push_back(aabb.min[1] + (static_cast<float>(leaf.iy) + 0.5f) * leaf_m);
        out_leaf_xyz.push_back(aabb.min[2] + (static_cast<float>(leaf.iz) + 0.5f) * leaf_m);
    }
}

// ---------------------------------------------------------------------------
// octree_unique_leaf_codes — the INDEPENDENT ground truth for
// GATE lossless_roundtrip: a trivial "collapse consecutive duplicates" pass
// over the already-sorted code array. Shares no code with the decoder above
// (not the trie, not the level expansion) — a bug that corrupted BOTH the
// encoder's occupancy-byte emission AND the decoder's expansion in exactly
// matching ways would still be caught here, because this function never
// looks at the occupancy stream at all, only at the ORIGINAL sorted codes.
// D is accepted for interface symmetry with kernels.cuh's declaration and
// as documentation (sorted_codes was produced AT depth D — the caller's
// responsibility) but is not otherwise needed: two full-depth codes are
// equal if and only if they name the same leaf, at any D.
// ---------------------------------------------------------------------------
void octree_unique_leaf_codes(int n, const uint64_t* sorted_codes, int /*D*/,
                              std::vector<uint64_t>& out_unique_leaf_codes)
{
    out_unique_leaf_codes.clear();
    if (n <= 0) return;
    out_unique_leaf_codes.push_back(sorted_codes[0]);
    for (int i = 1; i < n; ++i) {
        if (sorted_codes[i] != sorted_codes[i - 1]) out_unique_leaf_codes.push_back(sorted_codes[i]);
    }
}
