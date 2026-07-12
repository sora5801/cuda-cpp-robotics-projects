// ===========================================================================
// kernels.cu — GPU kernels for project 02.04 (Euclidean clustering via GPU
//              union-find / connected components): voxel binning, edge
//              construction, lock-free union-find, min-label propagation,
//              and cluster relabeling/stats.
//
// Role in the project
// -------------------
// All __global__ (GPU) code lives here, plus the small host-side launch
// wrappers that own the grid/block math and the Thrust orchestration
// (CLAUDE.md paragraph 6.1 rule 2: launch-configuration reasoning sits
// beside the code it configures).
//
// Read this after: main.cu, kernels.cuh.  Read this before: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK / CUDA_CHECK_LAST_ERROR

// Thrust: header-only pieces of the CUDA Toolkit (CLAUDE.md paragraph 5 —
// allowed without a separate .lib). Used for exactly the sort/scan/reduce
// primitives named at each call site below (CLAUDE.md paragraph 6.1 rule 6:
// what each computes, why the library instead of hand-rolling).
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/iterator/counting_iterator.h>

// is_nonzero — the copy_if predicate launch_build_voxel_index uses to
// compact the 0/1 boundary mask into segment-start positions (identical to
// 02.01's kernels.cu — CUDA 13.3's Thrust dropped thrust::identity, cited
// there; reproduced here for the same reason).
struct is_nonzero {
    __host__ __device__ bool operator()(int x) const { return x != 0; }
};

// ===========================================================================
// Device-side transcription of kernels.cuh's shared voxel-key helpers.
// WHY DUPLICATED: see kernels.cuh's file header ("Why this header is
// CUDA-qualifier-free") — the header's plain inline functions are HOST-only
// under nvcc's rules, so device code needs its own __device__ copy, written
// to match EXACTLY, cross-referenced here and caught by VERIFY(keys) in
// main.cu if the two ever drift (the identical discipline 02.01 established).
// ===========================================================================
__device__ __forceinline__ int32_t d_voxel_coord(float p, float leaf)
{
    return static_cast<int32_t>(floorf(p / leaf));
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

// ---------------------------------------------------------------------------
// d_lower_bound — a hand-rolled binary search over the ascending-sorted
// unique_key[0,count) array: the smallest index i with unique_key[i] >=
// target, or `count` if no such index exists (the standard std::lower_bound
// contract). Used by build_edges_kernel to test "is this neighbor voxel
// occupied?" in O(log V) per query instead of a second hash table — a
// DELIBERATELY different technique from 02.01's Method-A hash-insert/probe,
// so a learner sees two idiomatic ways to query the same kind of index
// (THEORY.md "The GPU mapping" compares their trade-offs: a hash lookup is
// O(1) expected but needs a second data structure; this binary search reuses
// the sorted array Method-B-style compaction already built, at O(log V)
// instead, with zero extra memory).
//
// This is a PLAIN device function (no library): CLAUDE.md's "no black
// boxes" stance, and binary search over ~10-100 elements is exactly the
// "small, hand-rolled beats a library call" case 02.03's jacobi_eigen_3x3
// and 33.01 make for similarly small, hot, elementary routines.
// ---------------------------------------------------------------------------
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
// d_atomic_min_float / d_atomic_max_float — atomic float min/max via the
// SAME atomicCAS-retry-loop idiom "THE UNION-FIND CHAPTER" (kernels.cuh)
// teaches for the union step: read the current value, and if it already
// beats the proposed one, stop; otherwise try to CAS it in, and on failure
// (someone else won the race) just re-read and retry. CUDA has no native
// atomicMin/atomicMax overload for float (only int/unsigned/long long), so
// this is the standard, portable, always-correct (any sign) construction —
// reinterpreting the float's bit pattern as an int purely so atomicCAS (an
// integer-only primitive) can operate on it; the VALUE COMPARISON is still
// done as float via __int_as_float, so IEEE-754 ordering (including
// negative numbers) is respected throughout, unlike the faster but sign-
// fragile "reinterpret as unsigned and flip" trick some CUDA samples use.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void d_atomic_min_float(float* addr, float value)
{
    int* addr_as_int = reinterpret_cast<int*>(addr);
    int old = *addr_as_int, assumed;
    do {
        if (__int_as_float(old) <= value) return;   // already the min: nothing to do
        assumed = old;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(value));
    } while (assumed != old);   // CAS failed (another thread updated first): retry with the fresh value
}

__device__ __forceinline__ void d_atomic_max_float(float* addr, float value)
{
    int* addr_as_int = reinterpret_cast<int*>(addr);
    int old = *addr_as_int, assumed;
    do {
        if (__int_as_float(old) >= value) return;
        assumed = old;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(value));
    } while (assumed != old);
}

// ---------------------------------------------------------------------------
// d_uf_find_halve — the union-find FIND primitive with PATH HALVING, the one
// device function every stage of Method A calls. See kernels.cuh "THE
// UNION-FIND CHAPTER" for the full walkthrough and the monotone-parent
// safety argument; this is that pseudocode, verbatim, in CUDA C++.
//
// Each iteration reads parent[x] and parent[parent[x]] (the grandparent),
// redirects parent[x] to point at the grandparent (skipping one link), and
// continues from the grandparent — so the path from any node to its root
// roughly HALVES in length every time find() walks it, which is exactly
// where "path halving" gets its name and exactly why repeated sweeps
// converge in O(log D) rather than O(D) (D = graph diameter; contrast
// lp_sweep_kernel below, which performs NO compression and therefore needs
// O(D) sweeps).
//
// Safety under concurrency: the single non-atomic store `parent[x] = gp`
// races freely against every other thread's own halving stores and against
// uf_union_sweep_kernel's atomicCAS unions — and this is PROVABLY safe
// because parent values only ever DECREASE over the run's lifetime
// (union-by-min never attaches a smaller root under a larger one). At the
// instant this thread reads parent[parent[x]], that value IS a true
// ancestor of x (possibly stale by the time of the write, but never
// WRONG — a stale ancestor is still an ancestor, just not yet the closest
// possible one; a later find() simply compresses further). No lost updates,
// no cycles, no torn reads are possible: every write here writes a value
// this thread itself just read from live memory.
// ---------------------------------------------------------------------------
__device__ __forceinline__ int d_uf_find_halve(int* __restrict__ parent, int x)
{
    while (true) {
        const int p = parent[x];
        if (p == x) return x;               // x IS the root: parent points at itself
        const int gp = parent[p];
        parent[x] = gp;                      // halve: skip x directly to its grandparent
        x = gp;
    }
}

// ===========================================================================
// Stage 1 — voxel keys (02.01 lineage; see that project's kernels.cu for the
// identical kernel with identical reasoning, cited rather than re-derived).
// ===========================================================================
__global__ void compute_voxel_keys_kernel(int n, const float* __restrict__ xyz,
                                          float leaf, unsigned long long* __restrict__ keys)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float px = xyz[i * 3 + 0], py = xyz[i * 3 + 1], pz = xyz[i * 3 + 2];
    const int32_t vx = d_voxel_coord(px, leaf);
    const int32_t vy = d_voxel_coord(py, leaf);
    const int32_t vz = d_voxel_coord(pz, leaf);
    keys[i] = d_pack_voxel_key(vx, vy, vz);
}

void launch_compute_voxel_keys(int n, const float* d_xyz, float leaf, unsigned long long* d_keys)
{
    const int block = kThreadsPerBlock;
    const int grid  = blocks_for(n, block);
    compute_voxel_keys_kernel<<<grid, block>>>(n, d_xyz, leaf, d_keys);
    CUDA_CHECK_LAST_ERROR("compute_voxel_keys_kernel launch");
}

// mark_boundaries_kernel — identical to 02.01's Method B (cited): one
// thread per sorted-array position, 1 iff a new voxel run starts here.
__global__ void mark_boundaries_kernel(int n, const unsigned long long* __restrict__ keys_sorted,
                                       int* __restrict__ is_start)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    is_start[i] = (i == 0 || keys_sorted[i] != keys_sorted[i - 1]) ? 1 : 0;
}

__global__ void gather_unique_keys_kernel(int num_voxels, const int* __restrict__ seg_start,
                                          const unsigned long long* __restrict__ keys_sorted,
                                          unsigned long long* __restrict__ unique_key_out)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_voxels) return;
    unique_key_out[v] = keys_sorted[seg_start[v]];   // every point in voxel v's run shares this key
}

// ---------------------------------------------------------------------------
// launch_build_voxel_index — 02.01 Method B's sort+compaction pipeline
// (cited, reused for the SAME purpose: turn "who shares a voxel" into
// contiguous sorted-array runs), plus one extra kernel (gather_unique_keys)
// this project's binary-search neighbor lookup needs that 02.01 did not.
// See kernels.cuh for the full parameter documentation.
// ---------------------------------------------------------------------------
int launch_build_voxel_index(int n, const unsigned long long* d_keys_in,
                             unsigned long long* d_keys_scratch, int* d_idx_sorted,
                             int* d_is_start_scratch, int* d_seg_start_out,
                             unsigned long long* d_unique_key_out)
{
    // Copy (never mutate) the shared, read-only key array — Thrust's sort
    // permutes its key range in place, and build_edges_kernel's `point_key`
    // argument needs the UNSORTED original-order copy to survive intact.
    CUDA_CHECK(cudaMemcpy(d_keys_scratch, d_keys_in,
                          static_cast<size_t>(n) * sizeof(unsigned long long),
                          cudaMemcpyDeviceToDevice));

    thrust::device_ptr<unsigned long long> keys_ptr(d_keys_scratch);
    thrust::device_ptr<int> idx_ptr(d_idx_sorted);

    // thrust::sequence: idx[i] = i, the identity permutation before sorting
    // (see 02.01's kernels.cu for the full explanation of this idiom).
    thrust::sequence(idx_ptr, idx_ptr + n);

    // thrust::stable_sort_by_key: radix-sorts the 64-bit keys ascending and
    // carries idx along as the paired permutation — after this call,
    // idx_sorted[k] is the ORIGINAL point index now at sorted position k
    // (02.01's kernels.cu explains what "radix sort" computes and why
    // Thrust's STABILITY guarantee matters; the same reasoning applies here
    // verbatim, cited rather than repeated in full).
    thrust::stable_sort_by_key(keys_ptr, keys_ptr + n, idx_ptr);

    {
        const int block = kThreadsPerBlock;
        const int grid  = blocks_for(n, block);
        mark_boundaries_kernel<<<grid, block>>>(n, d_keys_scratch, d_is_start_scratch);
        CUDA_CHECK_LAST_ERROR("mark_boundaries_kernel launch");
    }

    // thrust::reduce sums the 0/1 boundary mask -> the number of distinct
    // voxel keys (one "1" per run start).
    thrust::device_ptr<int> is_start_ptr(d_is_start_scratch);
    const int num_voxels = thrust::reduce(is_start_ptr, is_start_ptr + n, 0);

    // thrust::copy_if with a counting_iterator source: keep every sorted-
    // array position k whose boundary-mask entry is nonzero -> exactly the
    // ascending list of run-start offsets (02.01's kernels.cu explains this
    // "counting iterator + stencil = stream compaction" idiom in full).
    thrust::device_ptr<int> seg_start_ptr(d_seg_start_out);
    thrust::copy_if(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(n),
                    is_start_ptr, seg_start_ptr, is_nonzero());

    {
        const int block = kThreadsPerBlock;
        const int grid  = blocks_for(num_voxels, block);
        gather_unique_keys_kernel<<<grid, block>>>(num_voxels, d_seg_start_out, d_keys_scratch, d_unique_key_out);
        CUDA_CHECK_LAST_ERROR("gather_unique_keys_kernel launch");
    }

    return num_voxels;
}

// ===========================================================================
// Stage 2 — neighbor edges: the project's first genuinely new kernel.
// ===========================================================================

// ---------------------------------------------------------------------------
// build_edges_kernel — one thread per point i. Walks the 27-cell voxel
// stencil around i's own voxel (kClusterToleranceM's header comment proves
// this stencil is SUFFICIENT — no point within d of i can lie outside it),
// finds each occupied neighbor voxel via d_lower_bound, and for every
// candidate point j in that voxel's run tests the ACTUAL squared distance
// (voxel adjacency is necessary but not sufficient: two points in
// corner-adjacent voxels can be up to sqrt(3)*leaf apart, well over d) —
// only points that pass BOTH tests become an edge.
//
// Thread-to-data mapping: thread i owns point i and reads (never writes)
// every OTHER point's data — a classic "gather" kernel; the only WRITES are
// the atomic edge-list appends, which is why they need atomics at all
// (every other memory access here is read-only, hence race-free already).
//
// Dedup rule: an edge is only emitted when j > i (comparing ORIGINAL point
// indices, not sorted positions). This (a) halves the work relative to
// emitting both (i,j) and (j,i), since Euclidean-distance adjacency is
// symmetric, and (b) is what makes the CPU twin's independently-built edge
// set (reference_cpu.cpp's build_edges_cpu, using the identical i<j rule)
// directly, exactly comparable — VERIFY(edges) in main.cu would otherwise
// have to reconcile two differently-oriented edge sets.
//
// The atomic append is the SAME "atomic counter as a parallel push_back"
// idiom 02.01's hash_compact_kernel uses (cited): atomicAdd(edge_count,1)
// returns the slot THIS thread just claimed, before anyone else can reuse
// it; a bounds check against edge_capacity (never an unchecked write) turns
// a would-be buffer overrun into an honestly-counted overflow instead
// (kMaxEdgesPerPoint's header comment documents why this project's edge
// buffer is sized generously enough that overflow_count should always read
// 0 — main.cu gates on exactly that).
// ---------------------------------------------------------------------------
__global__ void build_edges_kernel(int n, const float* __restrict__ xyz,
                                   const unsigned long long* __restrict__ point_key,
                                   const unsigned long long* __restrict__ unique_key, int num_voxels,
                                   const int* __restrict__ seg_start,
                                   const int* __restrict__ idx_sorted, int n_sorted,
                                   float d, float d2,
                                   int* __restrict__ edge_u, int* __restrict__ edge_v, int edge_capacity,
                                   int* __restrict__ edge_count, int* __restrict__ overflow_count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float pi[3] = { xyz[i * 3 + 0], xyz[i * 3 + 1], xyz[i * 3 + 2] };

    int32_t vx, vy, vz;
    d_unpack_voxel_key(point_key[i], vx, vy, vz);

    // The 27-cell stencil: every combination of {-1,0,+1} along each axis.
    // A plain triple-nested loop (27 fixed iterations, no data-dependent
    // trip count) — deliberately unrolled-by-the-compiler-friendly rather
    // than a lookup table, since 27 is small and constant.
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const uint64_t nkey = d_pack_voxel_key(vx + dx, vy + dy, vz + dz);
                const int v = d_lower_bound(unique_key, num_voxels, nkey);
                if (v >= num_voxels || unique_key[v] != nkey) continue;   // neighbor voxel unoccupied

                const int begin = seg_start[v];
                const int end   = (v + 1 < num_voxels) ? seg_start[v + 1] : n_sorted;
                for (int k = begin; k < end; ++k) {
                    const int j = idx_sorted[k];       // original point index in this neighbor voxel
                    if (j <= i) continue;                // dedup: only i<j emits (see header comment)

                    const float pj[3] = { xyz[j * 3 + 0], xyz[j * 3 + 1], xyz[j * 3 + 2] };
                    const float dxp = pi[0] - pj[0], dyp = pi[1] - pj[1], dzp = pi[2] - pj[2];
                    const float dist2 = dxp * dxp + dyp * dyp + dzp * dzp;
                    if (dist2 > d2) continue;            // voxel-adjacent but geometrically too far

                    const int slot = atomicAdd(edge_count, 1);   // claim a push_back slot
                    if (slot < edge_capacity) {
                        edge_u[slot] = i;
                        edge_v[slot] = j;
                    } else {
                        atomicAdd(overflow_count, 1);      // honestly counted, never silently dropped
                    }
                }
            }
        }
    }
    (void)d;  // d itself (unsquared) is unused inside the kernel; kept in the signature for readability/logging parity with d2
}

int launch_build_edges(int n, const float* d_xyz, const unsigned long long* d_point_key,
                       const unsigned long long* d_unique_key, int num_voxels,
                       const int* d_seg_start, const int* d_idx_sorted, int n_sorted,
                       float d, int* d_edge_u, int* d_edge_v, int edge_capacity,
                       int* d_overflow_count)
{
    int* d_edge_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_edge_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_edge_count, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_overflow_count, 0, sizeof(int)));

    const float d2 = d * d;
    const int block = kThreadsPerBlock;
    const int grid  = blocks_for(n, block);
    build_edges_kernel<<<grid, block>>>(n, d_xyz, d_point_key, d_unique_key, num_voxels,
                                        d_seg_start, d_idx_sorted, n_sorted, d, d2,
                                        d_edge_u, d_edge_v, edge_capacity, d_edge_count, d_overflow_count);
    CUDA_CHECK_LAST_ERROR("build_edges_kernel launch");

    int edge_count = 0;
    CUDA_CHECK(cudaMemcpy(&edge_count, d_edge_count, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_edge_count));

    // Clamp the RETURNED count to capacity: entries beyond edge_capacity
    // were never written (guarded above), so the caller must never read
    // past them even though the atomic counter itself may have overshot.
    return (edge_count < edge_capacity) ? edge_count : edge_capacity;
}

// ===========================================================================
// Stage 3 — GPU union-find (Method A).
// ===========================================================================

__global__ void uf_init_kernel(int n, int* __restrict__ parent)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    parent[i] = i;   // every point starts as its own singleton component's root
}

void launch_uf_init(int n, int* d_parent)
{
    const int block = kThreadsPerBlock;
    const int grid  = blocks_for(n, block);
    uf_init_kernel<<<grid, block>>>(n, d_parent);
    CUDA_CHECK_LAST_ERROR("uf_init_kernel launch");
}

// uf_union_sweep_kernel — see kernels.cuh "THE UNION-FIND CHAPTER" for the
// full pseudocode this implements verbatim: find both endpoints' roots
// (path-halving as a side effect), and if they differ, attach the
// LARGER-valued root under the SMALLER via a lock-free atomicCAS retry
// loop. The while(ru!=rv) outer loop is what makes this lock-free rather
// than merely "hope the CAS wins first try": on a failed CAS, some OTHER
// thread already changed parent[hi] (won the race), so this thread simply
// re-finds both roots (cheap, thanks to halving) and tries again — this
// always terminates because parent values only ever decrease (the monotone
// argument), so ru and rv strictly shrink across retries.
__global__ void uf_union_sweep_kernel(int num_edges, const int* __restrict__ edge_u,
                                      const int* __restrict__ edge_v,
                                      int* __restrict__ parent, int* __restrict__ changed)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_edges) return;

    int ru = d_uf_find_halve(parent, edge_u[e]);
    int rv = d_uf_find_halve(parent, edge_v[e]);

    while (ru != rv) {
        const int lo = min(ru, rv);
        const int hi = max(ru, rv);
        // "hi is STILL a root (parent[hi]==hi) -> attach it under lo,
        // atomically". If parent[hi] changed since we read it (another
        // thread already unioned hi elsewhere), the CAS fails and `old`
        // tells us nothing useful beyond "retry" — re-finding from lo/hi is
        // simpler and just as cheap as trying to reuse `old` directly.
        const int old = atomicCAS(&parent[hi], hi, lo);
        if (old == hi) {
            atomicOr(changed, 1);   // a real union happened this sweep
            return;
        }
        ru = d_uf_find_halve(parent, lo);
        rv = d_uf_find_halve(parent, hi);
    }
    // ru == rv: u and v were already in the same component (possibly
    // because an earlier edge THIS SAME SWEEP, processed by another thread,
    // already merged them) — nothing to do, and nothing to report as a
    // "change" for THIS edge specifically (some other edge's union already
    // set the flag if that merge was new).
}

bool launch_uf_union_sweep(int num_edges, const int* d_edge_u, const int* d_edge_v,
                           int* d_parent, int* d_changed)
{
    CUDA_CHECK(cudaMemset(d_changed, 0, sizeof(int)));
    const int block = kThreadsPerBlock;
    const int grid  = blocks_for(num_edges, block);
    uf_union_sweep_kernel<<<grid, block>>>(num_edges, d_edge_u, d_edge_v, d_parent, d_changed);
    CUDA_CHECK_LAST_ERROR("uf_union_sweep_kernel launch");
    int changed = 0;
    CUDA_CHECK(cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
    return changed != 0;
}

// uf_finalize_kernel — ROOT CANONICALIZATION: after the sweep loop
// converges (no sweep changed anything), every component's tree is already
// nearly flat from repeated halving, but individual nodes may still be one
// or two hops from the true root. This pass guarantees the postcondition
// every downstream stage relies on: parent[i] == i's TRUE root, exactly,
// for every i, with no further indirection needed.
__global__ void uf_finalize_kernel(int n, int* __restrict__ parent)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int x = i;
    while (parent[x] != x) x = parent[x];   // walk to the true root (short walk: already near-flat)
    parent[i] = x;                          // full compression: point directly at the root
}

void launch_uf_finalize(int n, int* d_parent)
{
    const int block = kThreadsPerBlock;
    const int grid  = blocks_for(n, block);
    uf_finalize_kernel<<<grid, block>>>(n, d_parent);
    CUDA_CHECK_LAST_ERROR("uf_finalize_kernel launch");
}

// ===========================================================================
// Stage 4 — GPU label propagation (Method B; the 30.01/01.06/01.21 pattern,
// cited, adapted from a pixel grid to an arbitrary edge list).
// ===========================================================================

__global__ void lp_init_kernel(int n, int* __restrict__ label)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    label[i] = i;
}

void launch_lp_init(int n, int* d_label)
{
    const int block = kThreadsPerBlock;
    const int grid  = blocks_for(n, block);
    lp_init_kernel<<<grid, block>>>(n, d_label);
    CUDA_CHECK_LAST_ERROR("lp_init_kernel launch");
}

// lp_sweep_kernel — one thread per EDGE: flood the smaller of the two
// endpoint labels across the edge, in BOTH directions (an edge here is
// undirected — u might have the smaller label, or v might). Deliberately NO
// path compression of any kind: every sweep can only move a label ONE hop
// along ONE edge, so a component shaped like a long chain (the snake) needs
// as many sweeps as its longest shortest-path (its DIAMETER) before the
// global minimum label has flooded end to end — the O(D) bound THEORY.md
// derives and this project's snake is built specifically to make visible.
__global__ void lp_sweep_kernel(int num_edges, const int* __restrict__ edge_u,
                                const int* __restrict__ edge_v,
                                int* __restrict__ label, int* __restrict__ changed)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_edges) return;

    const int u = edge_u[e], v = edge_v[e];
    const int lu = label[u], lv = label[v];
    if (lu < lv) {
        atomicMin(&label[v], lu);
        atomicOr(changed, 1);
    } else if (lv < lu) {
        atomicMin(&label[u], lv);
        atomicOr(changed, 1);
    }
}

bool launch_lp_sweep(int num_edges, const int* d_edge_u, const int* d_edge_v,
                     int* d_label, int* d_changed)
{
    CUDA_CHECK(cudaMemset(d_changed, 0, sizeof(int)));
    const int block = kThreadsPerBlock;
    const int grid  = blocks_for(num_edges, block);
    lp_sweep_kernel<<<grid, block>>>(num_edges, d_edge_u, d_edge_v, d_label, d_changed);
    CUDA_CHECK_LAST_ERROR("lp_sweep_kernel launch");
    int changed = 0;
    CUDA_CHECK(cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
    return changed != 0;
}

// ===========================================================================
// Stage 5 — cluster relabeling (compact ids via scan) + per-cluster stats.
// ===========================================================================

__global__ void scatter_dense_id_kernel(int n, const int* __restrict__ idx_sorted,
                                        const int* __restrict__ dense_id_sorted,
                                        int* __restrict__ dense_id_out)
{
    // NOTE (see kernels.cuh's declaration comment): `dense_id_sorted` here
    // is actually the RAW 1-based inclusive-scan-of-boundaries value at
    // this sorted position — subtracting 1 turns "how many run-starts have
    // we seen up to and including here" into a 0-based dense cluster id.
    // Folding the -1 into this scatter avoids a separate elementwise pass.
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    dense_id_out[idx_sorted[k]] = dense_id_sorted[k] - 1;
}

// mark_root_boundaries_kernel — the int-keyed twin of mark_boundaries_kernel
// (which is typed for uint64_t voxel keys): one thread per sorted position,
// 1 iff a new ROOT VALUE begins here. File-local (not declared in
// kernels.cuh): it is a pure implementation detail of launch_relabel_
// clusters below, never called from main.cu or reference_cpu.cpp, so it
// does not belong in the shared cross-TU contract (kernels.cuh's job is the
// interface OTHER files need, not every internal helper — see that file's
// header for the same "declarations only, the real contract" framing).
__global__ void mark_root_boundaries_kernel(int n, const int* __restrict__ root_sorted,
                                            int* __restrict__ is_start)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    is_start[k] = (k == 0 || root_sorted[k] != root_sorted[k - 1]) ? 1 : 0;
}

// ---------------------------------------------------------------------------
// launch_relabel_clusters — "cluster relabeling (compact ids via scan)":
// turn the SPARSE canonical-root values every point carries (integers
// somewhere in [0,n), one value per distinct component, not contiguous)
// into a DENSE [0,K) id per point, so per-cluster stat arrays can be sized
// exactly K instead of n and indexed directly.
//
// The pipeline: sort (root, point_index) pairs by root ascending (Thrust
// stable_sort_by_key — the SAME primitive 02.01's Method B uses, cited),
// mark where each new root's run begins, then an INCLUSIVE SCAN (running
// sum) over that 0/1 mask turns "is this a boundary" into "how many
// boundaries have occurred up to and including here" — which, read at any
// sorted position, IS that position's 1-based dense cluster id. This is a
// different (and, for this exact job, slightly more direct) compaction
// idiom than 02.01 Method B's reduce+copy_if pair: no second array of
// "where do runs start" is even needed, because the scan output already
// answers "which dense id does THIS position belong to" for every position
// at once, not just the run-starts — thrust::inclusive_scan is exactly
// STD::PARTIAL_SUM's data-parallel cousin (each output = sum of all inputs
// at or before it), computed via a standard parallel work-efficient
// prefix-sum tree under the hood (THEORY.md "The GPU mapping" sketches it).
// ---------------------------------------------------------------------------
int launch_relabel_clusters(int n, const int* d_root,
                            int* d_root_scratch, int* d_idx_scratch,
                            int* d_is_start_scratch, int* d_scan_scratch,
                            int* d_dense_id_out)
{
    CUDA_CHECK(cudaMemcpy(d_root_scratch, d_root, static_cast<size_t>(n) * sizeof(int), cudaMemcpyDeviceToDevice));

    thrust::device_ptr<int> root_ptr(d_root_scratch);
    thrust::device_ptr<int> idx_ptr(d_idx_scratch);
    thrust::sequence(idx_ptr, idx_ptr + n);
    thrust::stable_sort_by_key(root_ptr, root_ptr + n, idx_ptr);

    {
        const int block = kThreadsPerBlock;
        const int grid  = blocks_for(n, block);
        mark_root_boundaries_kernel<<<grid, block>>>(n, d_root_scratch, d_is_start_scratch);
        CUDA_CHECK_LAST_ERROR("mark_root_boundaries_kernel launch");
    }

    thrust::device_ptr<int> is_start_ptr(d_is_start_scratch);
    thrust::device_ptr<int> scan_ptr(d_scan_scratch);
    // thrust::inclusive_scan(first,last,result): result[k] = sum(first[0..k])
    // -- a data-parallel running total, computed with a Blelloch-style
    // work-efficient parallel prefix sum internally (O(n) work, O(log n)
    // depth) rather than n sequential additions.
    thrust::inclusive_scan(is_start_ptr, is_start_ptr + n, scan_ptr);

    const int num_clusters = thrust::reduce(is_start_ptr, is_start_ptr + n, 0);

    {
        const int block = kThreadsPerBlock;
        const int grid  = blocks_for(n, block);
        scatter_dense_id_kernel<<<grid, block>>>(n, d_idx_scratch, d_scan_scratch, d_dense_id_out);
        CUDA_CHECK_LAST_ERROR("scatter_dense_id_kernel launch");
    }

    return num_clusters;
}

// ===========================================================================
// Per-cluster statistics: count, centroid (sum then divide), AABB (atomic
// float min/max via the CAS-loop helpers above).
// ===========================================================================

__global__ void stats_init_kernel(int n, int* __restrict__ count,
                                  float* __restrict__ sum_x, float* __restrict__ sum_y, float* __restrict__ sum_z,
                                  float* __restrict__ min_x, float* __restrict__ min_y, float* __restrict__ min_z,
                                  float* __restrict__ max_x, float* __restrict__ max_y, float* __restrict__ max_z)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    count[k] = 0;
    sum_x[k] = 0.0f; sum_y[k] = 0.0f; sum_z[k] = 0.0f;
    // +INFINITY / -INFINITY sentinels: the first real atomic_min/max always
    // overwrites them (any finite scene coordinate beats an infinite bound),
    // the same "identity element" role kEmptyKey plays for 02.01's hash slots.
    min_x[k] = min_y[k] = min_z[k] = INFINITY;
    max_x[k] = max_y[k] = max_z[k] = -INFINITY;
}

void launch_stats_init(int num_clusters_upper_bound, int* d_count,
                       float* d_sum_x, float* d_sum_y, float* d_sum_z,
                       float* d_min_x, float* d_min_y, float* d_min_z,
                       float* d_max_x, float* d_max_y, float* d_max_z)
{
    const int block = kThreadsPerBlock;
    const int grid  = blocks_for(num_clusters_upper_bound, block);
    stats_init_kernel<<<grid, block>>>(num_clusters_upper_bound, d_count,
                                       d_sum_x, d_sum_y, d_sum_z,
                                       d_min_x, d_min_y, d_min_z, d_max_x, d_max_y, d_max_z);
    CUDA_CHECK_LAST_ERROR("stats_init_kernel launch");
}

__global__ void stats_accumulate_kernel(int n, const float* __restrict__ xyz, const int* __restrict__ dense_id,
                                        int* __restrict__ count,
                                        float* __restrict__ sum_x, float* __restrict__ sum_y, float* __restrict__ sum_z,
                                        float* __restrict__ min_x, float* __restrict__ min_y, float* __restrict__ min_z,
                                        float* __restrict__ max_x, float* __restrict__ max_y, float* __restrict__ max_z)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int k = dense_id[i];
    const float px = xyz[i * 3 + 0], py = xyz[i * 3 + 1], pz = xyz[i * 3 + 2];
    atomicAdd(&count[k], 1);
    atomicAdd(&sum_x[k], px); atomicAdd(&sum_y[k], py); atomicAdd(&sum_z[k], pz);
    d_atomic_min_float(&min_x[k], px); d_atomic_min_float(&min_y[k], py); d_atomic_min_float(&min_z[k], pz);
    d_atomic_max_float(&max_x[k], px); d_atomic_max_float(&max_y[k], py); d_atomic_max_float(&max_z[k], pz);
}

void launch_stats_accumulate(int n, const float* d_xyz, const int* d_dense_id,
                             int* d_count,
                             float* d_sum_x, float* d_sum_y, float* d_sum_z,
                             float* d_min_x, float* d_min_y, float* d_min_z,
                             float* d_max_x, float* d_max_y, float* d_max_z)
{
    const int block = kThreadsPerBlock;
    const int grid  = blocks_for(n, block);
    stats_accumulate_kernel<<<grid, block>>>(n, d_xyz, d_dense_id, d_count,
                                             d_sum_x, d_sum_y, d_sum_z,
                                             d_min_x, d_min_y, d_min_z, d_max_x, d_max_y, d_max_z);
    CUDA_CHECK_LAST_ERROR("stats_accumulate_kernel launch");
}

__global__ void stats_finalize_kernel(int num_clusters, const int* __restrict__ count,
                                      float* __restrict__ sum_x, float* __restrict__ sum_y, float* __restrict__ sum_z)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= num_clusters) return;
    const int c = count[k];
    if (c <= 0) return;   // never populated (should not happen for k<num_clusters, guarded defensively)
    const float inv_c = 1.0f / static_cast<float>(c);
    sum_x[k] *= inv_c; sum_y[k] *= inv_c; sum_z[k] *= inv_c;   // sum -> centroid, in place
}

void launch_stats_finalize(int num_clusters, const int* d_count,
                           float* d_sum_x, float* d_sum_y, float* d_sum_z)
{
    const int block = kThreadsPerBlock;
    const int grid  = blocks_for(num_clusters, block);
    stats_finalize_kernel<<<grid, block>>>(num_clusters, d_count, d_sum_x, d_sum_y, d_sum_z);
    CUDA_CHECK_LAST_ERROR("stats_finalize_kernel launch");
}
