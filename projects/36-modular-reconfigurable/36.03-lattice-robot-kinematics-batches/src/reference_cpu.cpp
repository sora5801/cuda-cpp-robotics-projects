// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 36.03
//                     Lattice-robot kinematics batches (sliding-cube model)
//
// Two DIFFERENT jobs live in this file (both declared in kernels.cuh):
//
//   1. The FOUR *_cpu functions — exact, line-by-line ORACLE TWINS of the
//      four GPU kernels in kernels.cu, sequential over k. main.cu runs
//      these against the GPU on the FULL K=4096 batch and requires
//      BIT-EXACT integer agreement (this project's all-integer §5 gate —
//      no tolerance anywhere, unlike 08.01/09.01's FP32 comparisons).
//
//   2. The TWO *_bruteforce_cpu functions — INDEPENDENTLY shaped oracles
//      that re-derive the same two answers (articulation points, move
//      legality) via a structurally different method, so a bug shared by
//      the "fast" algorithm and its line-by-line CPU twin (a shared
//      misunderstanding of the rules, not a translation slip) still gets
//      caught. main.cu runs these on a SUBSET of the batch (§6) — see
//      kernels.cuh for why a subset, not the full K, is enough here.
//
// Per CLAUDE.md §5's reference-file rule: plain C++17, no CUDA headers, no
// cleverness. The *_cpu functions ARE deliberately duplicated math (not
// shared via the header) with kernels.cu's __device__ functions — diff the
// two files side by side; only the __device__/__global__ spellings differ.
//
// Read this after: kernels.cu — then compare the two side by side.
// ===========================================================================

#include "kernels.cuh"   // shared constants + this file's function prototypes

#include <cstdlib>       // std::abs

// ===========================================================================
// Section 1 — host twins of kernels.cu's device geometry helpers. Same
// algorithms, same variable names, no __device__/__restrict__ — see the
// matching function in kernels.cu for the full commentary (not repeated
// here on purpose: the MATH must stay identical, the prose would just
// drift out of sync if written twice).
// ===========================================================================

static int manhattan(const int32_t* a, const int32_t* b)
{
    return std::abs(a[0] - b[0]) + std::abs(a[1] - b[1]) + std::abs(a[2] - b[2]);
}

static bool occupied(const int32_t* p, int32_t x, int32_t y, int32_t z)
{
    for (int m = 0; m < kM; ++m) {
        if (p[m * 3 + 0] == x && p[m * 3 + 1] == y && p[m * 3 + 2] == z) return true;
    }
    return false;
}

static int64_t pack_key(int32_t x, int32_t y, int32_t z)
{
    constexpr int64_t kBias = 1LL << 20;
    return ((int64_t)(x + kBias) << 42)
         | ((int64_t)(y + kBias) << 21)
         |  (int64_t)(z + kBias);
}

static void slide_delta(int dir, int32_t& dx, int32_t& dy, int32_t& dz)
{
    const int axis = dir >> 1;
    const int sign = (dir & 1) ? -1 : 1;
    dx = (axis == 0) ? sign : 0;
    dy = (axis == 1) ? sign : 0;
    dz = (axis == 2) ? sign : 0;
}

static void corner_axes(int c, int& e_dir, int& f_dir)
{
    const int pair = c / 4;
    const int combo = c % 4;
    const int e_sign = (combo < 2) ? 0 : 1;
    const int f_sign = (combo % 2 == 0) ? 0 : 1;
    if (pair == 0)      { e_dir = 0 + e_sign; f_dir = 2 + f_sign; }
    else if (pair == 1) { e_dir = 0 + e_sign; f_dir = 4 + f_sign; }
    else                { e_dir = 2 + e_sign; f_dir = 4 + f_sign; }
}

// ===========================================================================
// Section 2 — the four ORACLE TWINS (CLAUDE.md §5 gate #1: full-batch
// GPU-vs-CPU agreement). Each is a straight sequential "for k in 0..K"
// wrapper around exactly the single-configuration algorithm the matching
// GPU kernel runs per thread — see kernels.cu for the algorithm narrative.
// ===========================================================================

void validity_cpu(int K, const int32_t* pos, uint8_t* valid)
{
    for (int k = 0; k < K; ++k) {                          // sequential over configs — the kernel's K threads, one at a time
        const int32_t* p = pos + (size_t)k * kM * 3;        // this configuration's kM*3 ints, same slice the kernel's thread k reads

        int64_t keys[kM];                                    // one sortable key per module (see pack_key() above)
        for (int m = 0; m < kM; ++m)
            keys[m] = pack_key(p[m * 3 + 0], p[m * 3 + 1], p[m * 3 + 2]);

        // Insertion sort keys[] ascending — same algorithm as the kernel,
        // just running on one CPU core instead of a GPU thread.
        for (int i = 1; i < kM; ++i) {
            const int64_t key = keys[i];
            int j = i - 1;
            while (j >= 0 && keys[j] > key) { keys[j + 1] = keys[j]; --j; }
            keys[j + 1] = key;
        }

        // Duplicates are adjacent after sorting — one linear scan finds them.
        uint8_t ok = 1;
        for (int i = 1; i < kM; ++i) if (keys[i] == keys[i - 1]) { ok = 0; break; }
        valid[k] = ok;
    }
}

void connectivity_cpu(int K, const int32_t* pos, const uint8_t* valid, uint8_t* connected)
{
    (void)valid;   // see connectivity_kernel's comment in kernels.cu

    for (int k = 0; k < K; ++k) {
        const int32_t* p = pos + (size_t)k * kM * 3;

        bool visited[kM];                       // has the BFS reached module i yet?
        for (int i = 0; i < kM; ++i) visited[i] = false;

        int queue[kM];                          // array-based FIFO frontier: push at qt, pop at qh
        int qh = 0, qt = 0;
        visited[0] = true;
        queue[qt++] = 0;                        // module 0 is always real and unmoved — see main.cu §4
        int reached = 1;

        // Standard BFS: dequeue u, scan every OTHER module for face-adjacency,
        // enqueue any newly-discovered neighbour. O(kM) per dequeue, O(kM^2) total.
        while (qh < qt) {
            const int u = queue[qh++];
            for (int v = 0; v < kM; ++v) {
                if (visited[v]) continue;
                if (manhattan(p + u * 3, p + v * 3) == 1) {
                    visited[v] = true;
                    queue[qt++] = v;
                    ++reached;
                }
            }
        }
        connected[k] = (reached == kM) ? 1 : 0;   // did the BFS reach every module?
    }
}

void articulation_cpu(int K, const int32_t* pos,
                      const uint8_t* valid, const uint8_t* connected,
                      uint8_t* is_articulation, int32_t* num_articulation)
{
    (void)valid;
    (void)connected;

    for (int k = 0; k < K; ++k) {
        const int32_t* p = pos + (size_t)k * kM * 3;

        // disc/low/parent/next_child: Tarjan bookkeeping, one entry per
        // module — see kernels.cu's articulation_kernel for the full
        // algorithm narrative (discovery time, low-link, the two cut-vertex
        // tests). This function is that same algorithm, iteratively, on one
        // CPU core instead of one GPU thread — line-by-line identical logic.
        int disc[kM], low[kM], parent[kM], next_child[kM];
        uint8_t artic[kM];
        for (int i = 0; i < kM; ++i) {
            disc[i] = -1; low[i] = -1; parent[i] = -1; next_child[i] = 0; artic[i] = 0;
        }

        int stack[kM];             // explicit DFS stack (recursion-to-state-machine)
        int sp = 0;                // stack pointer / current DFS path depth
        int timer = 0;             // next discovery time to hand out
        int root_children = 0;     // module 0's DFS-tree children (root special case)

        disc[0] = low[0] = timer++;
        stack[sp++] = 0;

        while (sp > 0) {
            const int u = stack[sp - 1];               // peek — may resume u's neighbour scan
            if (next_child[u] < kM) {
                const int v = next_child[u]++;           // next candidate neighbour
                if (manhattan(p + u * 3, p + v * 3) != 1) continue;   // not adjacent: try the next v

                if (disc[v] == -1) {
                    // TREE EDGE: descend into the undiscovered neighbour v.
                    parent[v] = u;
                    if (u == 0) ++root_children;
                    disc[v] = low[v] = timer++;
                    stack[sp++] = v;
                } else if (v != parent[u]) {
                    // BACK EDGE to an ancestor: relax u's low-link.
                    if (disc[v] < low[u]) low[u] = disc[v];
                }
            } else {
                // u's neighbours are exhausted: pop, propagate low[u] to the
                // parent, and apply the non-root cut-vertex test.
                --sp;
                if (sp > 0) {
                    const int par = stack[sp - 1];
                    if (low[u] < low[par]) low[par] = low[u];
                    if (par != 0 && low[u] >= disc[par]) artic[par] = 1;
                }
            }
        }
        if (root_children > 1) artic[0] = 1;   // root cut-vertex test (2+ DFS-tree children)

        int cnt = 0;
        for (int i = 0; i < kM; ++i) {
            is_articulation[(size_t)k * kM + i] = artic[i];
            cnt += artic[i];
        }
        num_articulation[k] = cnt;
    }
}

void move_enum_cpu(int K, const int32_t* pos,
                   const uint8_t* valid, const uint8_t* connected,
                   const uint8_t* is_articulation,
                   uint8_t* legal_move, int32_t* move_count)
{
    (void)valid;
    (void)connected;

    for (int k = 0; k < K; ++k) {
        const int32_t* p = pos + (size_t)k * kM * 3;
        const uint8_t* artic = is_articulation + (size_t)k * kM;
        uint8_t* out = legal_move + (size_t)k * kM * kNumMoveDirs;

        int total = 0;
        for (int m = 0; m < kM; ++m) {
            const int32_t ax = p[m * 3 + 0], ay = p[m * 3 + 1], az = p[m * 3 + 2];
            const bool movable = (artic[m] == 0);

            for (int dir = 0; dir < kNumSlideDirs; ++dir) {
                uint8_t legal = 0;
                if (movable) {
                    int32_t dx, dy, dz;
                    slide_delta(dir, dx, dy, dz);
                    const int32_t bx = ax + dx, by = ay + dy, bz = az + dz;
                    if (!occupied(p, bx, by, bz)) {
                        // 2-module wall spanning A+f and B+f — see kernels.cu
                        // Stage 4 header comment for why one cell is not enough.
                        bool support = false;
                        for (int f = 0; f < kNumSlideDirs && !support; ++f) {
                            if (f / 2 == dir / 2) continue;
                            int32_t fx, fy, fz;
                            slide_delta(f, fx, fy, fz);
                            const bool wall_at_a = occupied(p, ax + fx, ay + fy, az + fz);
                            const bool wall_at_b = occupied(p, bx + fx, by + fy, bz + fz);
                            if (wall_at_a && wall_at_b) support = true;
                        }
                        legal = support ? 1 : 0;
                    }
                }
                out[m * kNumMoveDirs + dir] = legal;
                total += legal;
            }

            for (int c = 0; c < kNumCornerDirs; ++c) {
                uint8_t legal = 0;
                if (movable) {
                    int e_dir, f_dir;
                    corner_axes(c, e_dir, f_dir);
                    int32_t edx, edy, edz, fdx, fdy, fdz;
                    slide_delta(e_dir, edx, edy, edz);
                    slide_delta(f_dir, fdx, fdy, fdz);
                    const int32_t bx = ax + edx + fdx, by = ay + edy + fdy, bz = az + edz + fdz;
                    if (!occupied(p, bx, by, bz)) {
                        const bool p1 = occupied(p, ax + edx, ay + edy, az + edz);
                        const bool p2 = occupied(p, ax + fdx, ay + fdy, az + fdz);
                        legal = (p1 != p2) ? 1 : 0;
                    }
                }
                out[m * kNumMoveDirs + kNumSlideDirs + c] = legal;
                total += legal;
            }
        }
        move_count[k] = total;
    }
}

// ===========================================================================
// Section 3 — the two BRUTE-FORCE ORACLES (CLAUDE.md §5 gate #2: cross-
// algorithm verification on a subset — kernels.cuh explains why a subset
// suffices). Both are deliberately written a DIFFERENT way than Section 1's
// twins (an explicit occupied-cell array instead of on-the-fly scans for
// the move checker; a literal 12-row table instead of the corner_axes()
// formula) — not because either is faster (at kM=24 neither is), but so a
// conceptual bug in the "fast" understanding of the rules is unlikely to
// reproduce itself here by accident.
// ===========================================================================

// articulation_bruteforce_cpu — the textbook "remove and recheck" oracle:
// for each module m, physically drop it from the occupancy test (skip it
// in every comparison) and BFS the remaining kM-1 modules from any other
// module; m is a cut vertex iff that BFS does not reach all kM-1 survivors.
// O(kM) removals x O(kM^2) BFS = O(kM^3) — 13,824 operations for kM=24, a
// non-issue at the "subset of the batch" scale main.cu uses this at
// (kernels.cuh's SUBSET note). Entirely independent of Tarjan low-link.
void articulation_bruteforce_cpu(const int32_t* pos_one_config, uint8_t* is_articulation_out)
{
    for (int removed = 0; removed < kM; ++removed) {
        // Start the BFS from any module OTHER than `removed`. kM=24 >= 2
        // always leaves a valid start; module 0 unless IT is the one removed.
        const int start = (removed == 0) ? 1 : 0;

        bool visited[kM];
        for (int i = 0; i < kM; ++i) visited[i] = false;
        visited[removed] = true;   // mark "removed" as if it were never there — it can never be (re)visited

        int queue[kM];
        int qh = 0, qt = 0;
        visited[start] = true;
        queue[qt++] = start;
        int reached = 1;

        while (qh < qt) {
            const int u = queue[qh++];
            for (int v = 0; v < kM; ++v) {
                if (visited[v]) continue;   // already seen, or IS the removed module
                if (manhattan(pos_one_config + u * 3, pos_one_config + v * 3) == 1) {
                    visited[v] = true;
                    queue[qt++] = v;
                    ++reached;
                }
            }
        }

        // Survivors = kM - 1 (every module except `removed`). Articulation
        // iff the BFS from `start` did not reach ALL of them.
        is_articulation_out[removed] = (reached == kM - 1) ? 0 : 1;
    }
}

// The 12 edge-diagonal directions as a LITERAL table (e_dir, f_dir pairs
// using the same 0..5 slide-direction numbering as slide_delta), instead of
// corner_axes()'s arithmetic formula — hand-written once, cross-checked
// against the formula by construction (every row below was computed the
// same way the formula derives it, then transcribed as data, so a mistake
// in re-deriving the formula from scratch would show up as a mismatch).
static const int kBruteCornerTable[kNumCornerDirs][2] = {
    {0, 2}, {0, 3}, {1, 2}, {1, 3},   // x paired with y   (c = 0..3)
    {0, 4}, {0, 5}, {1, 4}, {1, 5},   // x paired with z   (c = 4..7)
    {2, 4}, {2, 5}, {3, 4}, {3, 5},   // y paired with z   (c = 8..11)
};

// move_precondition_bruteforce_cpu — re-derives Stage 4's legality flags
// for ONE configuration via an EXPLICIT occupied-cell array (built once,
// up front, rather than re-scanning `pos` inline on every query) and the
// literal corner table above instead of corner_axes()'s formula. Same
// preconditions, differently organized code — see kernels.cu's Stage 4
// header comment for the rules themselves (not repeated here).
void move_precondition_bruteforce_cpu(const int32_t* pos_one_config,
                                      const uint8_t* is_articulation_one_config,
                                      uint8_t* legal_move_out)
{
    // Build the explicit cell list up front — a deliberately different
    // shape than occupied()'s "scan pos[] directly" approach.
    int32_t cells[kM][3];
    for (int m = 0; m < kM; ++m) {
        cells[m][0] = pos_one_config[m * 3 + 0];
        cells[m][1] = pos_one_config[m * 3 + 1];
        cells[m][2] = pos_one_config[m * 3 + 2];
    }
    auto cell_occupied = [&](int32_t x, int32_t y, int32_t z) -> bool {
        for (int m = 0; m < kM; ++m)
            if (cells[m][0] == x && cells[m][1] == y && cells[m][2] == z) return true;
        return false;
    };

    for (int m = 0; m < kM; ++m) {
        const int32_t ax = cells[m][0], ay = cells[m][1], az = cells[m][2];
        const bool movable = (is_articulation_one_config[m] == 0);

        // ---- slide directions: explicit per-axis support enumeration
        // (no loop over a direction table — the two OTHER axes are named
        // directly, a hand-unrolled cross-check of the fast path's loop).
        for (int dir = 0; dir < kNumSlideDirs; ++dir) {
            int32_t dx, dy, dz;
            slide_delta(dir, dx, dy, dz);
            const int32_t bx = ax + dx, by = ay + dy, bz = az + dz;
            uint8_t legal = 0;
            if (movable && !cell_occupied(bx, by, bz)) {
                // A 2-module wall spanning BOTH A's and B's offset in some
                // perpendicular direction — explicit per-axis form (the two
                // OTHER axes named directly, both signs), independent of the
                // fast path's loop-over-slide_delta() formulation.
                const int axis = dir >> 1;
                bool support = false;
                if (axis != 0) {
                    support = support || (cell_occupied(ax + 1, ay, az) && cell_occupied(bx + 1, by, bz));
                    support = support || (cell_occupied(ax - 1, ay, az) && cell_occupied(bx - 1, by, bz));
                }
                if (axis != 1) {
                    support = support || (cell_occupied(ax, ay + 1, az) && cell_occupied(bx, by + 1, bz));
                    support = support || (cell_occupied(ax, ay - 1, az) && cell_occupied(bx, by - 1, bz));
                }
                if (axis != 2) {
                    support = support || (cell_occupied(ax, ay, az + 1) && cell_occupied(bx, by, bz + 1));
                    support = support || (cell_occupied(ax, ay, az - 1) && cell_occupied(bx, by, bz - 1));
                }
                legal = support ? 1 : 0;
            }
            legal_move_out[m * kNumMoveDirs + dir] = legal;
        }

        // ---- corner directions: literal table instead of corner_axes()
        for (int c = 0; c < kNumCornerDirs; ++c) {
            const int e_dir = kBruteCornerTable[c][0];
            const int f_dir = kBruteCornerTable[c][1];
            int32_t edx, edy, edz, fdx, fdy, fdz;
            slide_delta(e_dir, edx, edy, edz);
            slide_delta(f_dir, fdx, fdy, fdz);
            const int32_t bx = ax + edx + fdx, by = ay + edy + fdy, bz = az + edz + fdz;
            uint8_t legal = 0;
            if (movable && !cell_occupied(bx, by, bz)) {
                const bool p1 = cell_occupied(ax + edx, ay + edy, az + edz);
                const bool p2 = cell_occupied(ax + fdx, ay + fdy, az + fdz);
                legal = (p1 != p2) ? 1 : 0;
            }
            legal_move_out[m * kNumMoveDirs + kNumSlideDirs + c] = legal;
        }
    }
}
