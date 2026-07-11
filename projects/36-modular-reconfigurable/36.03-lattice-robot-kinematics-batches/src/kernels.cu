// ===========================================================================
// kernels.cu — GPU implementation for project 36.03
//              Lattice-robot kinematics batches (sliding-cube model)
//
// The big idea
// ------------
// Every other flagship in this repo parallelizes across INDEPENDENT PHYSICS
// (rollouts in 08.01, joint chains in 09.01, particles in 04.01...). This
// project parallelizes across INDEPENDENT SMALL GRAPHS: each of the K
// threads owns one whole lattice-robot CONFIGURATION (kM=24 modules) and
// runs a complete graph algorithm on it — duplicate detection, BFS
// connectivity, Tarjan articulation points, move-precondition enumeration —
// entirely in local arrays, with zero communication between threads. That
// is the "small-graph-per-thread regime": the graphs here (24 nodes) are
// far too small to profitably parallelize INTERNALLY (a parallel BFS needs
// frontier synchronization across threads/blocks that costs more than a
// single thread's serial 24-node BFS ever could — THEORY.md "GPU mapping"
// argues this with the same scale-honesty 25.01 uses for its own choice of
// per-cell serial work). What DOES parallelize beautifully is the BATCH:
// K=4096 independent configurations, one thread each — exactly 08.01's
// "K independent futures" pattern, with a small graph algorithm standing in
// for the ODE rollout.
//
// Four stages, four kernels, each gating the next (documented per-kernel
// below and in main.cu's pipeline comment):
//   1. validity_kernel      — any two modules sharing a cell?
//   2. connectivity_kernel  — does the face-adjacency graph span all kM?
//   3. articulation_kernel  — which modules are cut vertices?
//   4. move_enum_kernel     — which (module, direction) pairs are legal
//                             sliding-cube moves for NON-cut-vertex modules?
//
// ALL-INTEGER (CLAUDE.md §12 feature, see kernels.cuh header): no float
// appears anywhere in this file. Every comparison below is an exact integer
// predicate, so the GPU-vs-CPU verify gate in main.cu demands BIT-EXACT
// agreement with reference_cpu.cpp, not a tolerance — a deliberate contrast
// with 08.01/09.01's FP32 rel-tolerance gates (THEORY.md "Numerical
// considerations" makes the contrast explicit).
//
// Read this after: kernels.cuh.  Companion oracle: reference_cpu.cpp (a
// line-by-line twin of every device function below, compiled by cl.exe).
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"      // CUDA_CHECK / CUDA_CHECK_LAST_ERROR (§6.1 rule 7)

#include <cstdio>
#include <cstdlib>

// ===========================================================================
// Shared device geometry helpers — deliberately NOT hoisted into kernels.cuh
// as shared code: reference_cpu.cpp carries its own host-only twins of every
// one of these (same repo convention 08.01 uses for cartpole_deriv — a
// shared header cannot hold __device__ code AND stay includable by cl.exe
// without the __CUDACC__ fence, and duplicating four small pure functions is
// cheaper to keep honest than threading that fence through math this simple).
// ===========================================================================

// manhattan — |ax-bx| + |ay-by| + |az-bz| between two module cells (a, b are
// pointers to the FIRST of 3 packed int32_t coordinates). This is the face-
// adjacency test: two unit cubes on an integer lattice share a face iff
// their Manhattan distance is exactly 1 (distance 0 = same cell = overlap,
// not adjacency; distance 2 with two nonzero components = an edge-diagonal
// neighbor, which is NOT mechanically connected — only a corner MOVE target,
// never a connectivity edge).
__device__ __forceinline__ int manhattan(const int32_t* a, const int32_t* b)
{
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2]);
}

// occupied — does ANY of the kM modules of config `p` sit at cell (x,y,z)?
// O(kM) linear scan: with kM=24 there is no benefit to a hash/sorted lookup
// here (a sorted-key check is used in validity_kernel, where EVERY key is
// compared anyway; here we ask a handful of ad-hoc point queries per
// module, and 24 int32 compares is a handful of cycles versus the sort's
// own O(kM log kM) setup cost — THEORY.md "Numerical considerations"
// discusses this tiny-M complexity honesty explicitly).
__device__ __forceinline__ bool occupied(const int32_t* p, int32_t x, int32_t y, int32_t z)
{
    for (int m = 0; m < kM; ++m) {
        if (p[m * 3 + 0] == x && p[m * 3 + 1] == y && p[m * 3 + 2] == z) return true;
    }
    return false;
}

// pack_key — fold one module's (x,y,z) into a single sortable int64_t, MSB-
// first by axis (x most significant, z least), so lexicographic integer
// order on the key equals lexicographic order on (x,y,z). kBias re-centers
// negative coordinates into an unsigned-feeling range before packing (grid
// coordinates in this project range roughly +-10^4 after the disconnect
// corruption's translation — see main.cu §4 — so 21 bits per axis,
// [-2^20, 2^20), is generous headroom, not a tight fit).
__device__ __forceinline__ int64_t pack_key(int32_t x, int32_t y, int32_t z)
{
    constexpr int64_t kBias = 1LL << 20;               // 1,048,576 — see comment above
    return ((int64_t)(x + kBias) << 42)
         | ((int64_t)(y + kBias) << 21)
         |  (int64_t)(z + kBias);
}

// slide_delta — the unit step for slide direction `dir` in [0,6): dir 0..5
// = +x,-x,+y,-y,+z,-z (kernels.cuh's fixed numbering). `axis = dir>>1`
// picks the coordinate (0=x,1=y,2=z); `dir&1` is the sign bit (0=positive,
// 1=negative) — a tiny formula standing in for a lookup table (see the
// kernels.cuh header comment on why this project skips __constant__ memory
// for its direction table, unlike 09.01's runtime-loaded robot model: a
// FIXED lattice geometry needs no upload, just arithmetic).
__device__ __forceinline__ void slide_delta(int dir, int32_t& dx, int32_t& dy, int32_t& dz)
{
    const int axis = dir >> 1;              // 0=x, 1=y, 2=z
    const int sign = (dir & 1) ? -1 : 1;     // even dir = positive axis, odd = negative
    dx = (axis == 0) ? sign : 0;
    dy = (axis == 1) ? sign : 0;
    dz = (axis == 2) ? sign : 0;
}

// corner_axes — the two component slide directions (e_dir, f_dir) whose sum
// is corner direction `c` in [0,12). The 12 edge-diagonals of a cube are
// exactly the 3 axis-PAIRS (xy, xz, yz) times the 4 sign combinations of
// each pair's two axes (README/THEORY diagram every one): c/4 selects the
// pair, c%4 selects the (e sign, f sign) combo. Reusing slide_delta() for
// both components is the same "one formula, two derived tables" economy the
// header promises — there is exactly one place that knows what a "unit
// step" is.
__device__ __forceinline__ void corner_axes(int c, int& e_dir, int& f_dir)
{
    const int pair = c / 4;                       // 0=xy, 1=xz, 2=yz
    const int combo = c % 4;                       // 0:(+,+) 1:(+,-) 2:(-,+) 3:(-,-)
    const int e_sign = (combo < 2) ? 0 : 1;         // combo 0,1 -> +e ; combo 2,3 -> -e
    const int f_sign = (combo % 2 == 0) ? 0 : 1;    // combo 0,2 -> +f ; combo 1,3 -> -f
    if (pair == 0)      { e_dir = 0 + e_sign; f_dir = 2 + f_sign; }   // x paired with y
    else if (pair == 1) { e_dir = 0 + e_sign; f_dir = 4 + f_sign; }   // x paired with z
    else                { e_dir = 2 + e_sign; f_dir = 4 + f_sign; }   // y paired with z
}

// ===========================================================================
// STAGE 1 — validity_kernel: does any pair of the kM modules of a
// configuration share a lattice cell?
//
// Method: pack each module's cell into one sortable int64_t key (see
// pack_key above), INSERTION-SORT the kM=24 keys in a local array, then
// scan for adjacent equal keys. Insertion sort is O(kM^2)=576 worst-case
// compare/shifts — at kM=24 this beats setting up a hash table (no growth,
// no collisions to reason about, and the sorted order is a genuinely useful
// byproduct: a human reading a debugger dump of `keys[]` sees the modules
// in a canonical order). This is the catalog's "hash/sort-based duplicate
// detection per config" — sort-based, chosen over hashing for exactly that
// simplicity-at-this-M argument.
//
// Thread-to-data mapping: thread k = blockIdx.x*blockDim.x+threadIdx.x owns
// configuration k entirely — reads kM*3 int32 from GLOBAL memory (`pos`),
// keeps the kM int64 keys in a LOCAL array (24*8=192 bytes; kM is small but
// this spills registers into L1-cached local memory, unlike 08.01's 4-float
// state — THEORY.md is honest about that tradeoff), writes one uint8 out.
// ===========================================================================
__global__ void validity_kernel(const int32_t* __restrict__ pos,   // [K*kM*3] see kernels.cuh POSITION LAYOUT
                                int K,
                                uint8_t* __restrict__ valid)        // [K] OUT
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;                                  // ragged-tail guard

    const int32_t* p = pos + (size_t)k * kM * 3;         // this thread's configuration, kM*3 ints

    int64_t keys[kM];                                     // local scratch: one sortable key per module
    for (int m = 0; m < kM; ++m)
        keys[m] = pack_key(p[m * 3 + 0], p[m * 3 + 1], p[m * 3 + 2]);

    // Textbook insertion sort: keys[0..i) is sorted; insert keys[i] into
    // place by shifting larger keys right. Chosen for its simplicity over a
    // fancier O(kM log kM) sort — at kM=24 the constant factor dominates,
    // and insertion sort is the one every reader can verify by eye.
    for (int i = 1; i < kM; ++i) {
        const int64_t key = keys[i];
        int j = i - 1;
        while (j >= 0 && keys[j] > key) {
            keys[j + 1] = keys[j];
            --j;
        }
        keys[j + 1] = key;
    }

    // A duplicate cell shows up as two EQUAL keys, which sorting has made
    // ADJACENT — one linear scan finds every duplicate without an O(kM^2)
    // all-pairs comparison.
    uint8_t ok = 1;
    for (int i = 1; i < kM; ++i) {
        if (keys[i] == keys[i - 1]) { ok = 0; break; }
    }
    valid[k] = ok;
}

// ===========================================================================
// STAGE 2 — connectivity_kernel: does the face-adjacency graph reach all
// kM modules from module 0?
//
// Method: textbook array-based BFS. `visited[]` and an explicit `queue[]`
// both live in local arrays sized kM (no dynamic allocation — the whole
// graph fits in the thread's own tiny world). Neighbor discovery is a
// linear O(kM) scan per dequeued node (no adjacency list is built — with
// kM=24 that list would itself cost O(kM^2) to construct, so we simply pay
// the O(kM^2) BFS cost directly and skip the bookkeeping).
//
// `valid` is accepted but NOT used to skip work: connectivity is well-
// defined even on a config with a duplicate cell (two co-located modules
// are, by construction, adjacent to exactly the same neighbors — see
// kernels.cuh's discussion of the duplicate-corruption ground truth in
// main.cu §4) — computing it unconditionally lets Stage 2 be independently
// checkable on EVERY configuration, not just the ones Stage 1 already
// passed. main.cu's verify/success gates are the layer that decides which
// configs' connectivity result is trusted.
// ===========================================================================
__global__ void connectivity_kernel(const int32_t* __restrict__ pos, // [K*kM*3]
                                    int K,
                                    const uint8_t* __restrict__ valid, // [K] (unused — see comment above)
                                    uint8_t* __restrict__ connected)    // [K] OUT
{
    (void)valid;   // deliberately unconditional — documented above

    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    const int32_t* p = pos + (size_t)k * kM * 3;

    bool visited[kM];
    for (int i = 0; i < kM; ++i) visited[i] = false;

    int queue[kM];                     // BFS frontier, array-based (push at qt, pop at qh)
    int qh = 0, qt = 0;
    visited[0] = true;
    queue[qt++] = 0;                   // module 0 is always a REAL, uncorrupted module (see main.cu §4)
    int reached = 1;

    while (qh < qt) {
        const int u = queue[qh++];
        for (int v = 0; v < kM; ++v) {
            if (visited[v]) continue;
            // manhattan==1: face-adjacent. v==u gives distance 0 and is
            // automatically excluded — no separate self-check needed.
            if (manhattan(p + u * 3, p + v * 3) == 1) {
                visited[v] = true;
                queue[qt++] = v;
                ++reached;
            }
        }
    }

    connected[k] = (reached == kM) ? 1 : 0;
}

// ===========================================================================
// STAGE 3 — articulation_kernel: Tarjan's DFS low-link algorithm, one cut-
// vertex search per thread, over its own kM-node graph.
//
// THEORY.md teaches the algorithm's intuition end to end; the essence:
//   disc[u]  = DFS discovery time of u (the order the search first visits u)
//   low[u]   = the SMALLEST discovery time reachable from u's DFS subtree
//              using at most one "back edge" (an edge to an ANCESTOR, not a
//              tree edge) — i.e. how far up the tree u's subtree can reach
//              without going through u's own parent edge.
//   Non-root u is a cut vertex iff it has a DFS-tree child c with
//              low[c] >= disc[u] — meaning c's whole subtree has NO back
//              edge escaping above u, so removing u strands that subtree.
//   The DFS root is a cut vertex iff it has 2+ DFS-tree children (they can
//              only be connected to each other THROUGH the root).
//
// Implemented ITERATIVELY with an explicit stack (not recursively): CUDA
// device recursion works, but an explicit stack keeps the per-thread stack
// depth and memory footprint fully under this file's control, and it is
// the version most CUDA-to-C++ readers can map back onto their systems-
// programming muscle memory (an explicit stack machine, not a call stack).
// `next_child[u]` remembers where u's neighbor scan left off, so re-peeking
// u after processing a child resumes exactly where it stopped — the
// standard "convert recursion to a state machine" trick.
// ===========================================================================
__global__ void articulation_kernel(const int32_t* __restrict__ pos,   // [K*kM*3]
                                    int K,
                                    const uint8_t* __restrict__ valid,      // unused — see connectivity_kernel comment
                                    const uint8_t* __restrict__ connected,  // unused — computed unconditionally
                                    uint8_t* __restrict__ is_articulation,  // [K*kM] OUT
                                    int32_t* __restrict__ num_articulation) // [K] OUT
{
    (void)valid;
    (void)connected;

    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    const int32_t* p = pos + (size_t)k * kM * 3;

    // Per-thread DFS bookkeeping — five kM-sized local arrays plus a stack.
    // disc/low: -1 means "unvisited" (module 0's component is a cut vertex
    // question ONLY about the reachable component — see the Stage 2
    // comment on disconnected-config honesty).
    int disc[kM], low[kM], parent[kM], next_child[kM];
    uint8_t artic[kM];
    for (int i = 0; i < kM; ++i) {
        disc[i] = -1; low[i] = -1; parent[i] = -1; next_child[i] = 0; artic[i] = 0;
    }

    int stack[kM];
    int sp = 0;              // stack pointer (depth of the current DFS path)
    int timer = 0;           // next discovery time to hand out
    int root_children = 0;   // DFS-tree children of module 0 (root special case)

    disc[0] = low[0] = timer++;
    stack[sp++] = 0;

    while (sp > 0) {
        const int u = stack[sp - 1];       // PEEK, do not pop — we may resume u's scan
        if (next_child[u] < kM) {
            const int v = next_child[u]++;  // next candidate neighbor to try
            // v==u gives distance 0, never 1 — no separate self-skip needed.
            if (manhattan(p + u * 3, p + v * 3) != 1) continue;   // not adjacent: try the next v

            if (disc[v] == -1) {
                // TREE EDGE: v is undiscovered — descend into it.
                parent[v] = u;
                if (u == 0) ++root_children;
                disc[v] = low[v] = timer++;
                stack[sp++] = v;            // push: v's children are explored next
            } else if (v != parent[u]) {
                // BACK EDGE to an already-discovered ancestor (never to the
                // parent — that is the tree edge we descended through, and
                // walking back "up" it is not a back edge for low-link
                // purposes). Relax u's low-link with v's discovery time.
                if (disc[v] < low[u]) low[u] = disc[v];
            }
            // else: v is the parent — the tree edge back up; skip (no-op).
        } else {
            // u's neighbor scan is exhausted: POP u and propagate its
            // low-link up to its parent, then apply the cut-vertex test.
            --sp;
            if (sp > 0) {
                const int par = stack[sp - 1];
                if (low[u] < low[par]) low[par] = low[u];
                if (par != 0 && low[u] >= disc[par]) artic[par] = 1;
            }
        }
    }
    if (root_children > 1) artic[0] = 1;   // root special case (see header comment)

    int cnt = 0;
    for (int i = 0; i < kM; ++i) {
        is_articulation[(size_t)k * kM + i] = artic[i];
        cnt += artic[i];
    }
    num_articulation[k] = cnt;
}

// ===========================================================================
// STAGE 4 — move_enum_kernel: legal sliding-cube moves per (module,
// direction) pair, restricted to NON-articulation modules.
//
// Two move families, README/THEORY diagram both exhaustively:
//
//   SLIDE (directions 0..5): module m slides from A to the face-adjacent
//   EMPTY cell B=A+e. Requires a 2-MODULE WALL — modules at BOTH A+f and
//   B+f for some f on a DIFFERENT axis than e (4 candidates: the two signs
//   of each of the other two axes). Why two modules and not one: on a
//   bipartite grid graph no single cell is face-adjacent to two OTHER
//   face-adjacent cells (a face-adjacent pair never shares a common face-
//   neighbor — a short parity argument, spelled out in THEORY.md), so a
//   lone module at A+f loses contact with the slider the instant it
//   reaches B (distance becomes 2, an edge-diagonal, not a face). A wall
//   of two modules spanning BOTH A+f and B+f gives the slider a
//   continuous face to ride along for the whole 1-cell motion — the
//   physically honest discrete precondition (kernels.cuh / THEORY.md).
//
//   CORNER (directions 6..17): module m rotates from A to the edge-
//   diagonal EMPTY cell B=A+e+f (e, f on different axes) by pivoting
//   around the edge of a neighbor. In the 2x2 cell block {A, A+e, A+f, B},
//   EXACTLY ONE of the two "L-corner" cells (A+e, A+f) must be occupied
//   (the pivot the module swings around) and the OTHER must be EMPTY (the
//   cell the module's corner sweeps through on its way to B — occupied
//   there means the rotation collides with a third module). Both occupied
//   means the corner is boxed in (no legal pivot); both empty means there
//   is nothing to swing around.
//
// Only NON-articulation modules are ever marked movable — an articulation
// module's removal fractures the OTHER kM-1 modules before it lands
// anywhere, so it is never legal regardless of geometry (kernels.cuh's
// "for each non-articulation module" scoping, enforced right here).
// ===========================================================================
__global__ void move_enum_kernel(const int32_t* __restrict__ pos,      // [K*kM*3]
                                 int K,
                                 const uint8_t* __restrict__ valid,       // unused — see comment above
                                 const uint8_t* __restrict__ connected,   // unused
                                 const uint8_t* __restrict__ is_articulation, // [K*kM]
                                 uint8_t* __restrict__ legal_move,        // [K*kM*kNumMoveDirs] OUT
                                 int32_t* __restrict__ move_count)        // [K] OUT
{
    (void)valid;
    (void)connected;

    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    const int32_t* p = pos + (size_t)k * kM * 3;
    const uint8_t* artic = is_articulation + (size_t)k * kM;
    uint8_t* out = legal_move + (size_t)k * kM * kNumMoveDirs;

    int total = 0;
    for (int m = 0; m < kM; ++m) {
        const int32_t ax = p[m * 3 + 0], ay = p[m * 3 + 1], az = p[m * 3 + 2];
        const bool movable = (artic[m] == 0);   // articulation modules: every direction stays illegal

        // ---- slide directions 0..5 --------------------------------------
        for (int dir = 0; dir < kNumSlideDirs; ++dir) {
            uint8_t legal = 0;
            if (movable) {
                int32_t dx, dy, dz;
                slide_delta(dir, dx, dy, dz);
                const int32_t bx = ax + dx, by = ay + dy, bz = az + dz;
                if (!occupied(p, bx, by, bz)) {
                    // Look for a 2-module WALL: some perpendicular direction f
                    // (a different axis than dir) with BOTH A+f and B+f
                    // occupied — a continuous face spanning the whole slide
                    // (see the Stage 4 header comment for why one cell is not
                    // enough).
                    bool support = false;
                    for (int f = 0; f < kNumSlideDirs && !support; ++f) {
                        if (f / 2 == dir / 2) continue;      // same axis as the slide: not perpendicular
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

        // ---- corner directions 6..17 -------------------------------------
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
                    const bool p1 = occupied(p, ax + edx, ay + edy, az + edz);   // pivot candidate 1 (A+e)
                    const bool p2 = occupied(p, ax + fdx, ay + fdy, az + fdz);   // pivot candidate 2 (A+f)
                    legal = (p1 != p2) ? 1 : 0;   // exactly one occupied: a clear pivot, an empty sweep cell
                }
            }
            out[m * kNumMoveDirs + kNumSlideDirs + c] = legal;
            total += legal;
        }
    }
    move_count[k] = total;
}

// ===========================================================================
// Host launchers (declared in kernels.cuh). All four share ONE launch
// geometry — 256-thread blocks, ceil(K/256) blocks, the repo default
// (08.01/09.01/33.01) — because all four kernels share the SAME mapping:
// one thread per configuration, no shared memory, no atomics, no cross-
// thread communication whatsoever. Each function's only job is grid math
// plus the mandatory post-launch error check (CLAUDE.md §6.1 rule 7).
// ===========================================================================

void launch_validity(int K, const int32_t* d_pos, uint8_t* d_valid)
{
    if (K < 1 || !d_pos || !d_valid) {
        std::fprintf(stderr, "launch_validity: invalid arguments (K=%d)\n", K);
        std::exit(EXIT_FAILURE);
    }
    const int threads = 256;
    const int blocks = (K + threads - 1) / threads;
    validity_kernel<<<blocks, threads>>>(d_pos, K, d_valid);
    CUDA_CHECK_LAST_ERROR("validity_kernel launch");
}

void launch_connectivity(int K, const int32_t* d_pos, const uint8_t* d_valid,
                         uint8_t* d_connected)
{
    if (K < 1 || !d_pos || !d_valid || !d_connected) {
        std::fprintf(stderr, "launch_connectivity: invalid arguments (K=%d)\n", K);
        std::exit(EXIT_FAILURE);
    }
    const int threads = 256;
    const int blocks = (K + threads - 1) / threads;
    connectivity_kernel<<<blocks, threads>>>(d_pos, K, d_valid, d_connected);
    CUDA_CHECK_LAST_ERROR("connectivity_kernel launch");
}

void launch_articulation(int K, const int32_t* d_pos,
                         const uint8_t* d_valid, const uint8_t* d_connected,
                         uint8_t* d_is_articulation, int32_t* d_num_articulation)
{
    if (K < 1 || !d_pos || !d_valid || !d_connected || !d_is_articulation || !d_num_articulation) {
        std::fprintf(stderr, "launch_articulation: invalid arguments (K=%d)\n", K);
        std::exit(EXIT_FAILURE);
    }
    const int threads = 256;
    const int blocks = (K + threads - 1) / threads;
    articulation_kernel<<<blocks, threads>>>(d_pos, K, d_valid, d_connected,
                                             d_is_articulation, d_num_articulation);
    CUDA_CHECK_LAST_ERROR("articulation_kernel launch");
}

void launch_move_enum(int K, const int32_t* d_pos,
                      const uint8_t* d_valid, const uint8_t* d_connected,
                      const uint8_t* d_is_articulation,
                      uint8_t* d_legal_move, int32_t* d_move_count)
{
    if (K < 1 || !d_pos || !d_valid || !d_connected || !d_is_articulation
        || !d_legal_move || !d_move_count) {
        std::fprintf(stderr, "launch_move_enum: invalid arguments (K=%d)\n", K);
        std::exit(EXIT_FAILURE);
    }
    const int threads = 256;
    const int blocks = (K + threads - 1) / threads;
    move_enum_kernel<<<blocks, threads>>>(d_pos, K, d_valid, d_connected,
                                          d_is_articulation, d_legal_move, d_move_count);
    CUDA_CHECK_LAST_ERROR("move_enum_kernel launch");
}
