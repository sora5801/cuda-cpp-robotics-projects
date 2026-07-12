// ===========================================================================
// kernels.cuh — interface for project 02.04
//               Euclidean clustering via GPU union-find / connected components
//               (Method A: lock-free GPU union-find, path-halving + union-by-
//                min. Method B: iterative min-label propagation, the same
//                edges, the SAME final partition, a very different sweep
//                count — the project's teaching core.)
//
// Role in the project
// -------------------
// The CONTRACT shared by main.cu (orchestration + gates + artifacts),
// kernels.cu (the GPU kernels), and reference_cpu.cpp (the independent CPU
// oracle twins). Everything all three must agree on — point-cloud layout,
// the voxel-key packing, the edge-list layout, the union-find/label arrays —
// is defined HERE, once, following 02.01's "single-sourced data-layout
// contract" precedent (see that project's kernels.cuh header for the
// pattern this file reuses almost verbatim for the voxel machinery).
//
// POINT CLOUD LAYOUT — float* xyz, interleaved, meters, LiDAR "sensor" frame
// (origin at the sensor, +x forward, +y left, +z up — CLAUDE.md paragraph 12
// body convention): xyz[i*3+0..2] = x,y,z. This project's INPUT is already
// the NON-GROUND subset of a scan — i.e. exactly what project 02.03 (ground
// segmentation) would hand downstream after its own compaction step; see
// scripts/make_synthetic.py for how the committed sample manufactures that
// hand-off with ground truth, and README "System context" for the named
// 02.01 -> 02.03 -> 02.04 pipeline this project completes.
//
// THE PIPELINE (the project's teaching spine; every stage cited below is
// implemented once, here, and consumed by main.cu in this exact order):
//
//   1. VOXEL BINNING (02.01's spatial-hash/key machinery, cited and reused
//      almost token-for-token below) with voxel LEAF = the cluster tolerance
//      D (kClusterToleranceM). Setting leaf == d is not a coincidence: it is
//      the exact condition under which a 27-cell (3x3x3) stencil around a
//      point's own voxel is GUARANTEED to contain every other point within
//      Euclidean distance d — see the proof in the comment above
//      kClusterToleranceM below, and THEORY.md "The GPU mapping" for the
//      full derivation.
//   2. NEIGHBOR EDGES — for every point, the 27-cell stencil (via a sorted
//      voxel index + binary search, not a second hash table: a deliberately
//      DIFFERENT technique from 02.01's Method A, so a learner meets two
//      idiomatic ways to query a spatial hash) finds every voxel-neighbor
//      candidate, tests the ACTUAL Euclidean distance (voxel adjacency is
//      necessary but not sufficient — a corner-adjacent voxel pair can still
//      be farther apart than d), and appends a qualifying (i,j), i<j pair to
//      a shared edge list (build_edges_kernel below).
//   3. GPU UNION-FIND (Method A, THE new idea this project teaches) — see
//      "THE UNION-FIND CHAPTER" below.
//   4. GPU LABEL PROPAGATION (Method B, the technique 30.01/01.06/01.21
//      already use for image-grid connected components, cited and adapted
//      here to an arbitrary EDGE LIST instead of a pixel grid) — run on the
//      SAME edges, so the only variable between the two methods is the
//      ALGORITHM, not the data.
//   5. ROOT CANONICALIZATION + CLUSTER RELABELING (compact ids via a Thrust
//      scan — the same sort/mark-boundary/compact idiom 02.01's Method B
//      teaches for voxel compaction, reused here for CLUSTER compaction,
//      cited as such) -> per-cluster stats (count/centroid/AABB, atomics)
//      -> min-size filtering (noise rejection).
//
// THE UNION-FIND CHAPTER — lock-free GPU union-find, step by step
// -----------------------------------------------------------------
// Classical (sequential) union-find with path compression + union by rank
// achieves O(alpha(n)) AMORTIZED time per operation, alpha the inverse
// Ackermann function — for any n representable in this universe, alpha(n)<=4,
// so union-find is "effectively constant time" (Tarjan 1975; THEORY.md "The
// math" states the amortized-complexity result precisely and cites the
// argument). This project's GPU version cannot run one operation at a time —
// thousands of edges are processed IN PARALLEL — so it cannot literally BE
// the sequential algorithm; instead it processes EVERY edge, every SWEEP, in
// parallel, with the SAME find-compress-union primitives, and repeats sweeps
// until nothing changes. THEORY.md derives why this converges in O(log D)
// sweeps (D = the graph's diameter) rather than needing D sweeps outright —
// the path-HALVING each find() performs, done by EVERY thread EVERY sweep,
// roughly halves every remaining path length each round (the same
// "pointer-jumping" idea that turns list-ranking from O(n) into O(log n) on
// a PRAM). uf_union_sweep_kernel below is that one primitive, called
// row-by-row from main.cu's convergence loop.
//
//   find-root, with PATH HALVING (the compression step):
//       while parent[x] != x:
//           parent[x] = parent[parent[x]]     // "skip a link" — grandparent, not root
//           x = parent[x]
//       return x
//   Why HALVING, not full compression, inside a kernel every thread might
//   call concurrently: full compression needs TWO passes (find the root,
//   then walk again writing it into every node) — a second pass another
//   thread could be racing through the SAME nodes mid-write. Halving needs
//   only ONE pass, and each write only ever redirects parent[x] to
//   parent[parent[x]] — a value that WAS a valid ancestor of x at the moment
//   it was read. See the monotone-parent argument below for why that is
//   always safe, even racing against other threads' writes to the same node.
//
//   union, with UNION-BY-MIN + a lock-free atomicCAS retry loop:
//       ru = find(u);  rv = find(v)
//       if ru == rv: done (already the same component)
//       lo = min(ru, rv);  hi = max(ru, rv)
//       loop:
//           old = atomicCAS(&parent[hi], hi, lo)   // "hi is still a root -> attach it under lo"
//           if old == hi: done (the union succeeded)
//           hi = find(hi)                          // someone else moved hi meanwhile: refresh and retry
//           lo = min(lo, hi); if lo == hi: done (already unioned by another thread)
//   The CAS only succeeds if parent[hi] is STILL hi (i.e. hi is STILL a
//   root) at the instant of the atomic operation — exactly the same
//   "compare, then write, atomically" idea 02.01's hash_insert_kernel uses
//   for its claim-or-probe loop, applied here to a UNION instead of an
//   INSERT.
//
//   THE MONOTONE-PARENT ARGUMENT (why this is safe with NO locks) — the
//   property every step above relies on, stated once and cited everywhere
//   it is used (kernels.cu repeats the pointer to this paragraph at each
//   call site): union-by-min NEVER attaches a smaller-valued root under a
//   larger one — only ever the reverse. Consequently, for every node x,
//   parent[x] is MONOTONICALLY NON-INCREASING over the whole run's lifetime
//   (once parent[x] becomes some value p, it can later only become SMALLER,
//   never larger, because the only writer of parent[x] is a union that
//   proved x was a root and is attaching it under a SMALLER root). Path
//   halving therefore never "corrupts" the structure: redirecting
//   parent[x] from p to parent[p] can only ever move x's pointer to a value
//   that is EQUAL TO OR SMALLER than a true ancestor of x at read-time —
//   still a value on the true path to x's eventual root, or already the
//   root itself. No CAS is needed on the halving write specifically BECAUSE
//   of this: two threads racing to halve the SAME node can only race between
//   two values that are BOTH valid (one older, one more-compressed)
//   ancestors, so a plain (non-atomic) store cannot introduce a cycle or
//   lose the true root — the classic Anderson & Woll (1991) / Jayanti &
//   Tarjan (2016) lock-free union-find safety argument, taught here at the
//   level this repo's audience needs (THEORY.md "The math" gives the full
//   citation and a sketch of the proof).
//
// Why this header is CUDA-qualifier-free where possible (02.01's precedent)
// ---------------------------------------------------------------------------
// Pure math/bookkeeping functions below (voxel_coord, pack/unpack_voxel_key,
// spatial_hash) are declared as PLAIN inline C++ — no __host__/__device__ —
// so they compile under BOTH nvcc (main.cu, kernels.cu's host-side code, and
// reference_cpu.cpp's shared-formula calls) and cl.exe (reference_cpu.cpp
// otherwise). Being unqualified, they are HOST-only under nvcc's rules and
// CANNOT be called from inside a __global__/__device__ function; kernels.cu
// therefore carries its OWN literal __device__ transcription of each one,
// clearly cross-referenced in comments at both copies — the gate that
// catches drift between a header copy and its device transcription is
// VERIFY(keys) in main.cu (mirroring 02.01's identical gate exactly).
//
// The UNION-FIND and LABEL-PROPAGATION algorithmic cores are, per the
// reference_cpu.cpp independence ruling, NOT shared: reference_cpu.cpp's
// serial_union_find_cpu and build_edges_cpu are genuinely independent
// re-implementations (different data structures: std::unordered_map instead
// of a sorted-array + binary search; sequential instead of parallel-sweep;
// see reference_cpu.cpp's file header for the full ruling and why this
// project needs it more than most — union-find is exactly the kind of
// "clever" algorithm CLAUDE.md warns a shared implementation could smuggle
// the same bug into both sides of a twin comparison).
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cstdint>       // int32_t, uint32_t, uint64_t — exact-width integers for the key packing
#include <cmath>         // std::floor, std::sqrt — identical overloads to cl.exe and nvcc's host pass
#include <vector>        // reference_cpu.cpp's independent edge list / union-find output
#include <utility>       // std::pair<int,int> — the canonicalized (u<v) edge representation
#include <unordered_map> // reference_cpu.cpp's independent voxel->points map (build_edges_cpu)

// ===========================================================================
// Problem-scale constants
// ===========================================================================

// Repo-default block size (warp multiple, good occupancy on sm_75..sm_89 —
// see kernels.cu's launch-configuration comments for the per-kernel story).
constexpr int kThreadsPerBlock = 256;

// ---------------------------------------------------------------------------
// kClusterToleranceM ("d") — the Euclidean-clustering distance threshold AND
// (deliberately, see below) the voxel leaf size fed to 02.01's spatial-hash
// machinery. Two points closer than this are the "same object" by the
// single-linkage rule this project implements (THEORY.md "The math" states
// the definition precisely: clustering = connected components of the graph
// where an edge joins every pair of points within d of each other).
//
// WHY leaf == d EXACTLY (the proof the 27-cell stencil in kernels.cu leans
// on): let L be the voxel edge length and suppose L >= d. Take any two
// points p, q with |p-q| <= d. For EACH axis a, |p_a - q_a| <= |p-q| <= d
// <= L (a single coordinate difference is never more than the full
// Euclidean distance). Voxel indices are floor(coord/L); if two voxel
// indices along axis a differed by 2 or more, the two points would have to
// straddle at least one FULL voxel width between their cells, forcing
// |p_a-q_a| >= L -- contradicting |p_a-q_a| <= L only at the boundary and
// impossible for a strict difference of 2+. Hence every axis's voxel index
// differs by AT MOST 1, i.e. q's voxel lies within the 3x3x3 block of
// voxels centered on p's voxel -- the 27-cell stencil. This holds with
// EQUALITY (L == d) as well as any L > d; using the tightest bound (L = d)
// keeps voxels small, which keeps each voxel's point COUNT small, which
// keeps the per-point stencil-scan cheap (THEORY.md "The GPU mapping"
// derives the resulting expected work per point from the scene's point
// density). A learner should notice this is the SAME slab argument 02.01's
// THEORY.md "The math" makes for hash-collision probability, aimed instead
// at a coverage guarantee.
constexpr float kClusterToleranceM = 0.40f;  // d = leaf, meters (see the derivation above)

// ---------------------------------------------------------------------------
// kMinClusterSize — a connected component with FEWER than this many points
// is reported as unclustered NOISE, not a cluster (README "Expected output",
// THEORY.md "Numerical considerations" for why: a component of size 1 or 2
// carries essentially no shape evidence -- it could be a stray multipath
// return, calibration speckle, or a single grass blade, none of which a
// downstream tracker (04.xx) should instantiate a full object hypothesis
// for). Every scattered-noise point in the synthetic scene is, by
// construction, farther than d from every other point -- a SINGLETON
// component (size 1) -- so ANY threshold >= 2 filters it; 5 is chosen with
// margin so a learner can see the floor is not razor-thin (scripts/
// make_synthetic.py's module docstring restates this design decision).
// Stored in the sample file's header too (min_cluster_size) so main.cu can
// assert the data and the compiled pipeline were designed for each other --
// the same data/code consistency check 02.01 does for kVoxelLeafM.
// ---------------------------------------------------------------------------
constexpr int kMinClusterSize = 5;

// Sentinel: a point's FINAL cluster id after min-size filtering, when its
// raw connected component was rejected as noise. Never a valid dense id
// (dense ids start at 0), so a single sentinel check is unambiguous.
constexpr int32_t kNoCluster = -1;

// ---------------------------------------------------------------------------
// kMaxEdgesPerPoint — the per-point upper bound the edge buffer is sized
// with (capacity = n * kMaxEdgesPerPoint), the SAME "size to a documented
// worst case, then detect and report overflow rather than silently drop"
// discipline 02.01's hash table capacity (kTargetLoadFactor) uses.
//
// Worst-case derivation: the densest object in the synthetic scene is a
// FILLED voxel-grid block at point spacing g = 0.15 m (see make_synthetic.
// py's OBJECT_FILL_SPACING_M). A point deep inside such a block has every
// other filled-grid point within floor(d/g) = floor(0.40/0.15) = 2 grid
// steps of it in each axis as a geometric NEIGHBOR CANDIDATE; the count of
// integer lattice points inside a Euclidean ball of radius 2 grid steps is
// bounded by the CONTINUOUS ball volume (4/3)*pi*2^3 ~= 33.5, generously
// rounded up and margined against edge effects (a point at a block corner
// sees FEWER neighbors, an interior point sees close to this bound) to 96,
// then doubled again for headroom against a future scene edit -- 256 is a
// comfortable, documented ceiling matched to this specific scene's density,
// not a universal constant (a much denser cloud would need a larger bound,
// or the two-pass count-then-fill idiom THEORY.md "Where this sits in the
// real world" names as the production-grade alternative).
// ---------------------------------------------------------------------------
constexpr int kMaxEdgesPerPoint = 256;

// Safety caps on the sweep-loop convergence checks main.cu runs (never
// expected to be HIT -- both algorithms provably converge on a finite graph
// -- but a hard cap turns a latent bug into a loud, bounded failure instead
// of an infinite loop, the same honesty policy 02.03's kRansacMaxTripletAttempts
// and this repo's other iterative kernels apply).
constexpr int kMaxUfSweeps = 128;    // union-find: THEORY.md predicts O(log D) sweeps; D here is at most a few hundred
constexpr int kMaxLpSweeps = 4096;   // label propagation: THEORY.md predicts O(D) sweeps; the snake alone needs low hundreds

// blocks_for — integer ceiling division: how many `threads`-wide blocks
// cover `count` independent problems (the same idiom 02.01/02.03/08.01 use).
inline int blocks_for(int count, int threads)
{
    return (count + threads - 1) / threads;
}

// ===========================================================================
// Voxel key packing — IDENTICAL formulas to 02.01's kernels.cuh (cited, not
// reinvented): 3 signed integer voxel coordinates -> one 64-bit key, 21 bits
// per axis with a bias so a SIGNED coordinate packs into an UNSIGNED field.
// See 02.01's kernels.cuh for the full bit-budget derivation (the same
// +-210 km per-axis headroom argument applies verbatim at this project's
// leaf size). Re-declared here (rather than #included from 02.01) per
// CLAUDE.md's self-containment rule: projects never reference another
// project's folder at build time, so shared formulas are copied,
// deliberately and documented, not linked.
// ===========================================================================
constexpr int32_t  kCoordBias   = 1 << 20;             // 1,048,576 — recenters [-2^20,2^20-1] to [0,2^21-1]
constexpr uint64_t kCoordMask21 = (1ull << 21) - 1ull;  // low 21 bits: 0x1FFFFF
constexpr uint64_t kEmptyKey    = ~0ull;                // sentinel: bit 63 set, never a valid packed key

inline int32_t voxel_coord(float p, float leaf)
{
    // floor(), not truncation -- see 02.01's kernels.cuh voxel_coord() for
    // the negative-coordinate pitfall this avoids (identical reasoning,
    // identical fix, cited rather than re-derived here).
    return static_cast<int32_t>(std::floor(p / leaf));
}

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

// squared_distance — |p-q|^2 for two {x,y,z} arrays. Squared, not the actual
// distance, so every "within d" test in this project (build_edges_kernel,
// build_edges_cpu) compares against d*d and never pays a sqrt -- a classic,
// free micro-optimization worth NAMING once here so it is not mistaken for
// an approximation: x^2 is a monotonic transform of |x| for x>=0, so the
// ORDERING (and the <= d test) is exactly preserved.
inline float squared_distance(const float p[3], const float q[3])
{
    const float dx = p[0] - q[0], dy = p[1] - q[1], dz = p[2] - q[2];
    return dx * dx + dy * dy + dz * dz;
}

// ===========================================================================
// GPU kernel declarations — nvcc-only (see the file header for why this
// fence exists: cl.exe, compiling reference_cpu.cpp, has never heard of
// __global__ and must never see these).
// ===========================================================================
#ifdef __CUDACC__

// ---- Stage 1: voxel keys (02.01 lineage, cited above) --------------------

// compute_voxel_keys_kernel — one thread per point: pack this point's voxel
// key at leaf = kClusterToleranceM. in xyz [n*3] device floats; out keys [n]
// device uint64_t, in ORIGINAL point-index order (NOT sorted).
__global__ void compute_voxel_keys_kernel(int n, const float* __restrict__ xyz,
                                          float leaf, unsigned long long* __restrict__ keys);

// mark_boundaries_kernel — one thread per SORTED-ARRAY position: 1 where a
// new voxel's run of points begins (02.01 Method B's identical kernel,
// cited and reused verbatim -- both projects need the same "which sorted
// positions start a new key" primitive).
__global__ void mark_boundaries_kernel(int n, const unsigned long long* __restrict__ keys_sorted,
                                       int* __restrict__ is_start);

// gather_unique_keys_kernel — one thread per OCCUPIED VOXEL v: copy that
// voxel's key (every point in its run shares one key by construction) into
// a dense, ascending-sorted unique_key[v] array -- the array
// build_edges_kernel's device_lower_bound searches below.
__global__ void gather_unique_keys_kernel(int num_voxels, const int* __restrict__ seg_start,
                                          const unsigned long long* __restrict__ keys_sorted,
                                          unsigned long long* __restrict__ unique_key_out);

// ---- Stage 2: neighbor edges ----------------------------------------------

// build_edges_kernel — one thread per point i (the project's first genuinely
// new kernel): scan the 27 voxel-stencil neighbors of i's own voxel, find
// each occupied one via a device binary search over unique_key[0..num_voxels),
// walk that voxel's point run (idx_sorted[seg_start[v]..seg_start[v+1])),
// and atomically append an edge (i,j) for every candidate j > i with
// squared_distance(i,j) <= d*d. See kernels.cu for the full walkthrough,
// including the "why j > i" dedup rule and the atomic-counter-as-push_back
// idiom (02.01's hash_compact_kernel, cited).
//
//   xyz              — [n*3] the point cloud (read-only).
//   point_key         — [n] this point's OWN voxel key, ORIGINAL index order
//                       (from compute_voxel_keys_kernel; NOT the sorted copy).
//   unique_key         — [num_voxels] ascending-sorted distinct voxel keys.
//   seg_start          — [num_voxels] sorted-array offset where each voxel's
//                       point run begins (the (v+1)'th entry, or n if v is
//                       the last voxel, bounds the run's end).
//   idx_sorted          — [n] original point indices in ascending-voxel-key
//                       sorted order (the permutation the Thrust sort produced).
//   edge_u/edge_v         — [n*kMaxEdgesPerPoint] OUT: the appended edge list
//                       (only the first *edge_count entries are valid).
//   edge_count          — OUT: running/final total edge count (atomic counter).
//   overflow_count       — OUT: incremented if a point's local candidate list
//                       would exceed the per-point capacity check inside the
//                       kernel (an honest failure signal, never a silent drop).
__global__ void build_edges_kernel(int n, const float* __restrict__ xyz,
                                   const unsigned long long* __restrict__ point_key,
                                   const unsigned long long* __restrict__ unique_key, int num_voxels,
                                   const int* __restrict__ seg_start,
                                   const int* __restrict__ idx_sorted, int n_sorted,
                                   float d, float d2,
                                   int* __restrict__ edge_u, int* __restrict__ edge_v, int edge_capacity,
                                   int* __restrict__ edge_count, int* __restrict__ overflow_count);

// ---- Stage 3: GPU union-find (Method A) -----------------------------------

// uf_init_kernel — parent[i] = i for every point: every point starts as the
// root of its own singleton component (the standard union-find seed).
__global__ void uf_init_kernel(int n, int* __restrict__ parent);

// uf_union_sweep_kernel — ONE SWEEP of the convergence loop: one thread per
// EDGE. find() BOTH endpoints (path halving as a side effect on every node
// each find touches), and if they resolve to different roots, union them by
// min via the atomicCAS retry loop ("THE UNION-FIND CHAPTER" above). Sets
// *changed nonzero iff at least one union actually happened this sweep —
// main.cu's host loop calls this repeatedly until a sweep changes nothing.
__global__ void uf_union_sweep_kernel(int num_edges, const int* __restrict__ edge_u,
                                      const int* __restrict__ edge_v,
                                      int* __restrict__ parent, int* __restrict__ changed);

// uf_finalize_kernel — ROOT CANONICALIZATION: one thread per point, a full
// (loop-to-fixpoint) find with compression, so that on return parent[i] is
// EXACTLY that point's component root -- no further indirection. Every
// gate and every relabeling step downstream assumes this postcondition.
__global__ void uf_finalize_kernel(int n, int* __restrict__ parent);

// ---- Stage 4: GPU label propagation (Method B) ----------------------------

// lp_init_kernel — label[i] = i (30.01/01.06/01.21's convention, cited,
// adapted from a masked pixel grid to an unmasked point/edge graph: every
// point here IS foreground, so no kLabelNone sentinel is needed).
__global__ void lp_init_kernel(int n, int* __restrict__ label);

// lp_sweep_kernel — ONE SWEEP: one thread per EDGE (u,v). Each direction is
// a min-flood: if label[u] < label[v], atomicMin(&label[v], label[u]), and
// symmetrically for the other direction; *changed is set nonzero iff either
// update actually lowered a label. Unlike union-find's find(), THIS kernel
// performs NO compression -- every sweep only propagates a label ONE hop
// along the edge it processes, which is exactly why convergence needs
// O(graph diameter) sweeps instead of O(log diameter) (THEORY.md "The
// algorithm" derives the bound; the snake in the synthetic scene exists to
// make the difference measurable, not just assertable).
__global__ void lp_sweep_kernel(int num_edges, const int* __restrict__ edge_u,
                                const int* __restrict__ edge_v,
                                int* __restrict__ label, int* __restrict__ changed);

// ---- Stage 5: relabeling (compact ids via scan) + per-cluster stats ------

// scatter_dense_id_kernel — one thread per SORTED-ARRAY position k: write
// this position's dense cluster id (already computed on the host/via Thrust
// scan into dense_id_sorted) back into dense_id_out at the ORIGINAL point
// index idx_sorted[k] -- turning a sorted-order compaction back into a
// per-point lookup table. See launch_relabel_clusters in kernels.cu for the
// full sort -> mark-boundary -> scan pipeline this is the last step of.
__global__ void scatter_dense_id_kernel(int n, const int* __restrict__ idx_sorted,
                                        const int* __restrict__ dense_id_sorted,
                                        int* __restrict__ dense_id_out);

// stats_init_kernel — one thread per potential cluster slot k in [0,n):
// zero the accumulators before stats_accumulate_kernel's atomics run (n is
// used as the allocation upper bound -- occupied clusters K <= n always,
// the same "size to n, use the first K" discipline 02.01 applies).
__global__ void stats_init_kernel(int n, int* __restrict__ count,
                                  float* __restrict__ sum_x, float* __restrict__ sum_y, float* __restrict__ sum_z,
                                  float* __restrict__ min_x, float* __restrict__ min_y, float* __restrict__ min_z,
                                  float* __restrict__ max_x, float* __restrict__ max_y, float* __restrict__ max_z);

// stats_accumulate_kernel — one thread per POINT: atomically fold this
// point into its dense cluster's running count/sum (centroid numerator) and
// AABB (min/max via the atomicCAS-loop float helpers in kernels.cu).
__global__ void stats_accumulate_kernel(int n, const float* __restrict__ xyz, const int* __restrict__ dense_id,
                                        int* __restrict__ count,
                                        float* __restrict__ sum_x, float* __restrict__ sum_y, float* __restrict__ sum_z,
                                        float* __restrict__ min_x, float* __restrict__ min_y, float* __restrict__ min_z,
                                        float* __restrict__ max_x, float* __restrict__ max_y, float* __restrict__ max_z);

// stats_finalize_kernel — one thread per cluster slot k in [0,K): turn the
// running position SUM into a CENTROID (divide by count); no-op (count==0)
// slots are never read downstream (K is the exact occupied-cluster count).
__global__ void stats_finalize_kernel(int num_clusters, const int* __restrict__ count,
                                      float* __restrict__ sum_x, float* __restrict__ sum_y, float* __restrict__ sum_z);

#endif // __CUDACC__

// ===========================================================================
// Host-callable launch wrappers (definitions in kernels.cu, which only nvcc
// compiles — but the DECLARATIONS below are plain C++, visible to main.cu).
// ===========================================================================

void launch_compute_voxel_keys(int n, const float* d_xyz, float leaf, unsigned long long* d_keys);

// launch_build_voxel_index — Thrust sort_by_key(keys) + boundary compaction,
// the 02.01 Method-B pipeline (cited) reused here to produce the sorted
// point-index permutation and per-voxel run boundaries build_edges_kernel
// needs. d_keys_in [n] is READ-ONLY (shared with the un-sorted point_key
// argument build_edges_kernel also takes); every other array is scratch or
// output, sized n (voxels) / n (points) by the caller, upper bounds.
// Returns the number of OCCUPIED voxels (== valid unique_key_out/seg_start
// entries).
int launch_build_voxel_index(int n, const unsigned long long* d_keys_in,
                             unsigned long long* d_keys_scratch, int* d_idx_sorted,
                             int* d_is_start_scratch, int* d_seg_start_out,
                             unsigned long long* d_unique_key_out);

// launch_build_edges — build_edges_kernel's host wrapper: resets the atomic
// counters, launches the kernel, and reads back the final edge count and
// overflow count so main.cu can size its host-side copy and gate on
// overflow==0. Returns the edge count (<= n*kMaxEdgesPerPoint by construction).
int launch_build_edges(int n, const float* d_xyz, const unsigned long long* d_point_key,
                       const unsigned long long* d_unique_key, int num_voxels,
                       const int* d_seg_start, const int* d_idx_sorted, int n_sorted,
                       float d, int* d_edge_u, int* d_edge_v, int edge_capacity,
                       int* d_overflow_count);

void launch_uf_init(int n, int* d_parent);

// launch_uf_union_sweep — one sweep; returns true iff at least one union
// happened (i.e. the caller should sweep again). Resets the device changed
// flag itself.
bool launch_uf_union_sweep(int num_edges, const int* d_edge_u, const int* d_edge_v,
                           int* d_parent, int* d_changed);

void launch_uf_finalize(int n, int* d_parent);

void launch_lp_init(int n, int* d_label);

// launch_lp_sweep — the label-propagation twin of launch_uf_union_sweep;
// same return-value contract (true = keep sweeping).
bool launch_lp_sweep(int num_edges, const int* d_edge_u, const int* d_edge_v,
                     int* d_label, int* d_changed);

// launch_relabel_clusters — "cluster relabeling (compact ids via scan)":
// turn a canonical-root array (sparse values in [0,n)) into a DENSE
// [0,K) id per point via Thrust stable_sort_by_key + an inclusive_scan over
// the boundary mask (the scan IS the compaction — no copy_if needed here,
// unlike 02.01's Method B, giving a learner a second, related-but-different
// Thrust compaction idiom to compare against). d_root_scratch/d_idx_scratch/
// d_is_start_scratch/d_scan_scratch are caller-provided scratch, sized n.
// d_dense_id_out [n] receives the per-point dense id. Returns K, the number
// of distinct roots (== distinct raw connected components).
int launch_relabel_clusters(int n, const int* d_root,
                            int* d_root_scratch, int* d_idx_scratch,
                            int* d_is_start_scratch, int* d_scan_scratch,
                            int* d_dense_id_out);

void launch_stats_init(int num_clusters_upper_bound, int* d_count,
                       float* d_sum_x, float* d_sum_y, float* d_sum_z,
                       float* d_min_x, float* d_min_y, float* d_min_z,
                       float* d_max_x, float* d_max_y, float* d_max_z);

void launch_stats_accumulate(int n, const float* d_xyz, const int* d_dense_id,
                             int* d_count,
                             float* d_sum_x, float* d_sum_y, float* d_sum_z,
                             float* d_min_x, float* d_min_y, float* d_min_z,
                             float* d_max_x, float* d_max_y, float* d_max_z);

void launch_stats_finalize(int num_clusters, const int* d_count,
                           float* d_sum_x, float* d_sum_y, float* d_sum_z);

// ===========================================================================
// CPU references (reference_cpu.cpp) — the correctness-oracle twins. All
// pointers below are HOST pointers. See reference_cpu.cpp's file header for
// the independence ruling each of these follows.
// ===========================================================================

// compute_voxel_keys_cpu — the twin of compute_voxel_keys_kernel, calling
// this header's OWN voxel_coord/pack_voxel_key (a shared data-layout
// formula, not a duplicated algorithm — see the file header). VERIFY(keys)
// in main.cu compares this, point for point, against the GPU's
// device-transcribed version — the gate that catches drift between the two
// copies (mirroring 02.01's identical VERIFY(keys) gate).
void compute_voxel_keys_cpu(int n, const float* xyz, float leaf, unsigned long long* keys_out);

// build_edges_cpu — a GENUINELY INDEPENDENT re-implementation of neighbor-
// edge construction: an std::unordered_map<uint64_t, std::vector<int>>
// voxel->points map (a completely different data structure from the GPU's
// sorted-array + binary-search index), scanned with the SAME 27-cell
// stencil and the SAME i<j / distance<=d rule. Returns the edge set
// canonicalized as ascending (u,v), u<v pairs, itself sorted ascending —
// VERIFY(edges) in main.cu compares this against the GPU's edge set
// (canonicalized and sorted the same way) for EXACT set equality.
std::vector<std::pair<int,int>> build_edges_cpu(int n, const float* xyz, float d);

// serial_union_find_cpu — a GENUINELY INDEPENDENT re-implementation of
// Method A: ordinary SEQUENTIAL union-find (no threads, no atomics) over
// build_edges_cpu's edge list, with the identical union-by-min + path-
// compression rules (stated once more, sequentially, so a reader can watch
// the "textbook" version run without the parallel bookkeeping). Because
// union-by-min's FINAL partition (and its per-component root = min member)
// is mathematically order-independent, this CPU result is expected to
// match the GPU union-find's finalized parent[] EXACTLY, bit-for-bit,
// despite completely different execution orders — the strongest possible
// correctness statement main.cu's VERIFY(union_find) gate can make.
// parent_out[n] receives each point's canonical root (min index in its
// component); the SAME array also serves as the correctness oracle for
// VERIFY(label_propagation), since label propagation converges to the
// identical canonical partition (THEORY.md "The math").
void serial_union_find_cpu(int n, const std::vector<std::pair<int,int>>& edges, std::vector<int>& parent_out);

#endif // PROJECT_KERNELS_CUH
