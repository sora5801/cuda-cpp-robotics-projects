// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 02.04
//                     (Euclidean clustering via GPU union-find / connected
//                     components)
//
// WHY does a GPU repository ship a CPU implementation of everything?
// ------------------------------------------------------------------
// Two load-bearing reasons (CLAUDE.md paragraph 5):
//
//   1) It is the CORRECTNESS ORACLE. GPU code fails in ways CPU code
//      cannot: wrong thread indexing, missed tail elements, race
//      conditions, stale device memory, bad transfers. A dead-simple
//      sequential version that a reader can verify BY EYE gives us ground
//      truth; main.cu runs both and asserts agreement.
//   2) It is the TEACHING BASELINE. Reading this file first, then
//      kernels.cu, shows exactly what parallelization changed.
//
// Independence ruling for THIS project specifically (see kernels.cuh's
// "Why this header is CUDA-qualifier-free" for the general repo policy)
// ---------------------------------------------------------------------------
// Union-find is EXACTLY the kind of "clever" algorithm the template's
// independence ruling warns about: a subtle bug in the union-by-rank/
// path-compression logic could easily be reproduced identically in a
// twin implementation that shares its structure with the GPU version,
// making the twin comparison worthless (it would faithfully confirm two
// wrong answers agree). So this file's two algorithmic cores are
// GENUINELY INDEPENDENT of kernels.cu, not transcriptions:
//
//   * build_edges_cpu uses an std::unordered_map<uint64_t, std::vector<int>>
//     voxel -> point-list map — a completely different data structure from
//     the GPU's sorted-array + binary-search index (kernels.cu's
//     launch_build_voxel_index / d_lower_bound). Same 27-cell stencil rule,
//     same i<j dedup, same distance test — but no shared code path with the
//     GPU's neighbor search beyond the single-sourced voxel-key FORMULA
//     (kernels.cuh's pack_voxel_key, a data-layout contract, not an
//     algorithm — see kernels.cuh's file header for that distinction).
//
//   * serial_union_find_cpu is an ordinary SEQUENTIAL, single-threaded
//     union-find: no sweeps, no atomics, no retry loops — the textbook
//     version a reader can step through by hand. It shares the union-by-
//     min RULE (attach the larger-valued root under the smaller) with the
//     GPU's uf_union_sweep_kernel because that rule is the DEFINITION of
//     the canonical-labeling convention this project promises (component
//     id = its minimum member), not an implementation detail — changing it
//     would change what "correct" means, not just how fast it runs.
//
// Why this is not paranoia (the template's own case study, cited): flagship
// 13.03 had an identical variable-shadowing bug live in BOTH the GPU path
// and a too-similar CPU twin; only an INDEPENDENT gate caught it. Here, the
// independent gate is stronger still: because union-by-min's FINAL
// partition is mathematically ORDER-INDEPENDENT (any correct sequence of
// unions over the same edge set converges to the same canonical roots,
// regardless of processing order), main.cu's VERIFY(union_find) demands
// BIT-EXACT agreement between this file's sequential result and the GPU's
// massively-parallel one — not a tolerance, an equality. See THEORY.md
// "How we verify correctness" for why that is a meaningful, not accidental,
// guarantee.
//
// Read this after: kernels.cu — then compare the two side by side.
// ===========================================================================

#include "kernels.cuh"
#include <cstdint>
#include <algorithm>   // std::sort — canonicalizes build_edges_cpu's output for VERIFY(edges)

// ---------------------------------------------------------------------------
// compute_voxel_keys_cpu — the twin of compute_voxel_keys_kernel, calling
// this header's OWN voxel_coord/pack_voxel_key (a single-sourced data-layout
// FORMULA, not a duplicated algorithm; kernels.cuh's file header explains
// why sharing this particular kind of code is deliberate, not an
// independence violation). VERIFY(keys) in main.cu compares this, point for
// point, against the GPU's device-transcribed version.
// ---------------------------------------------------------------------------
void compute_voxel_keys_cpu(int n, const float* xyz, float leaf, unsigned long long* keys_out)
{
    for (int i = 0; i < n; ++i) {
        const float px = xyz[i * 3 + 0], py = xyz[i * 3 + 1], pz = xyz[i * 3 + 2];
        const int32_t vx = voxel_coord(px, leaf);
        const int32_t vy = voxel_coord(py, leaf);
        const int32_t vz = voxel_coord(pz, leaf);
        keys_out[i] = pack_voxel_key(vx, vy, vz);
    }
}

// ---------------------------------------------------------------------------
// build_edges_cpu — INDEPENDENT neighbor-edge construction (see file header
// for exactly how it differs from the GPU path). Two passes over the point
// set:
//
//   Pass 1: bucket every point index into an unordered_map keyed by its
//   packed voxel key — std::unordered_map's internal chaining/open-hashing
//   is itself a THIRD data structure distinct from both the GPU's sorted
//   array (Method B style) and 02.01's open-addressing table (Method A
//   style), so this pass alone already exercises a different code path
//   than anything in kernels.cu.
//
//   Pass 2: for every point i, walk its own 27-cell voxel stencil, look up
//   each neighbor key in the map (map::find — a hash lookup, but through a
//   totally different table implementation than the GPU's binary search),
//   and for every candidate j > i in that bucket, test the SAME squared-
//   distance rule the GPU kernel uses. Complexity is close to linear in n
//   for a spatially well-distributed point set (each point visits a
//   constant number of buckets of roughly constant size) — this is a
//   TEACHING reference, not a performance target, so no further tuning is
//   attempted (CLAUDE.md "teaching beats cleverness").
//
// Returns the edge set already SORTED ascending by (u,v) (a plain
// std::vector sort at the end) so main.cu's VERIFY(edges) can compare it
// against the GPU's own sorted copy with a single vector::operator==.
// ---------------------------------------------------------------------------
std::vector<std::pair<int,int>> build_edges_cpu(int n, const float* xyz, float d)
{
    const float d2 = d * d;

    std::unordered_map<uint64_t, std::vector<int>> voxel_points;
    voxel_points.reserve(static_cast<size_t>(n) * 2);   // generous: occupied voxels <= n

    std::vector<uint64_t> point_key(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const float px = xyz[i * 3 + 0], py = xyz[i * 3 + 1], pz = xyz[i * 3 + 2];
        const int32_t vx = voxel_coord(px, d), vy = voxel_coord(py, d), vz = voxel_coord(pz, d);
        const uint64_t key = pack_voxel_key(vx, vy, vz);
        point_key[static_cast<size_t>(i)] = key;
        voxel_points[key].push_back(i);
    }

    std::vector<std::pair<int,int>> edges;
    for (int i = 0; i < n; ++i) {
        int32_t vx, vy, vz;
        unpack_voxel_key(point_key[static_cast<size_t>(i)], vx, vy, vz);
        const float pi[3] = { xyz[i * 3 + 0], xyz[i * 3 + 1], xyz[i * 3 + 2] };

        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const uint64_t nkey = pack_voxel_key(vx + dx, vy + dy, vz + dz);
                    const auto it = voxel_points.find(nkey);
                    if (it == voxel_points.end()) continue;   // neighbor voxel unoccupied
                    for (int j : it->second) {
                        if (j <= i) continue;   // dedup: identical i<j rule to the GPU kernel
                        const float pj[3] = { xyz[j * 3 + 0], xyz[j * 3 + 1], xyz[j * 3 + 2] };
                        if (squared_distance(pi, pj) <= d2) {
                            edges.emplace_back(i, j);
                        }
                    }
                }
            }
        }
    }

    std::sort(edges.begin(), edges.end());   // canonical order for main.cu's set-equality comparison
    return edges;
}

// ---------------------------------------------------------------------------
// serial_union_find_cpu — ordinary sequential union-find: the textbook
// version, processing edges 0..E-1 in the ORDER build_edges_cpu produced
// them (an arbitrary but FIXED order — irrelevant to the final partition,
// see the file header's order-independence argument). find_root() below
// performs full recursive path compression (a SECOND, different
// compression strategy from the GPU's iterative path HALVING — both are
// textbook-correct; using a different one here is itself a small extra
// piece of independence, since a bug specific to one compression style is
// unlikely to be replicated by the other).
// ---------------------------------------------------------------------------
static int find_root_recursive(std::vector<int>& parent, int x)
{
    if (parent[static_cast<size_t>(x)] == x) return x;
    // Full path compression: every node visited on the way to the root gets
    // repointed DIRECTLY at the root, via the recursive return value —
    // the classic two-line "textbook" compression a learner is most likely
    // to have already seen in an algorithms course.
    const int root = find_root_recursive(parent, parent[static_cast<size_t>(x)]);
    parent[static_cast<size_t>(x)] = root;
    return root;
}

void serial_union_find_cpu(int n, const std::vector<std::pair<int,int>>& edges, std::vector<int>& parent_out)
{
    parent_out.assign(static_cast<size_t>(n), 0);
    for (int i = 0; i < n; ++i) parent_out[static_cast<size_t>(i)] = i;   // every point its own root initially

    for (const auto& e : edges) {
        const int ru = find_root_recursive(parent_out, e.first);
        const int rv = find_root_recursive(parent_out, e.second);
        if (ru == rv) continue;
        // Union by MIN (not by rank/size): the same canonicalization
        // convention the GPU path promises — a component's root is always
        // its smallest member — so this sequential result is directly,
        // bit-exactly comparable to the GPU's finalized parent[] array.
        if (ru < rv) parent_out[static_cast<size_t>(rv)] = ru;
        else         parent_out[static_cast<size_t>(ru)] = rv;
    }

    // A final pass guarantees every entry points DIRECTLY at its root (some
    // nodes may not have been touched as a compression target above if they
    // were never the STARTING point of a find call) — the same "root
    // canonicalization" postcondition uf_finalize_kernel establishes on the
    // GPU side, so the two arrays are comparable entry-for-entry.
    for (int i = 0; i < n; ++i) {
        parent_out[static_cast<size_t>(i)] = find_root_recursive(parent_out, i);
    }
}
