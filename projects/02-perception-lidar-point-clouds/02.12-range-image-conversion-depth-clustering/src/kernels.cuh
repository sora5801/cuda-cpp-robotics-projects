// ===========================================================================
// kernels.cuh — interface & contract for project 02.12
//               Range-image conversion + depth-clustering segmentation
//
// Role in the project
// -------------------
// The CONTRACT shared by main.cu (orchestration + gates + artifacts),
// kernels.cu (the GPU kernels), and reference_cpu.cpp (the independent CPU
// oracle twins). Grid shape, the (ring,azimuth)->cell formula, the beta
// (depth-discontinuity) criterion, the ground-removal angle test, and the
// voxel-hash Euclidean-comparison machinery are single-sourced HERE, once,
// following 02.01's "single-sourced data-layout contract" precedent.
//
// THE THESIS (read this before anything else)
// ---------------------------------------------------------------------------
// A spinning LiDAR does not natively produce an "unorganized point cloud" —
// it produces one range value per (beam, azimuth-step), i.e. a 2-D RANGE
// IMAGE, exactly the way a rolling-shutter camera produces one intensity per
// (row, column). 02.02's kernels.cuh already names this organized grid and
// its ring-major layout (cited, reused verbatim below: kNumBeams,
// kAzimuthBins, kBeamElevRad, organized_cell_index). THIS project's thesis
// is what that organization buys you: two cells that are NEIGHBORS IN THE
// IMAGE (adjacent ring or adjacent azimuth column) are, almost always,
// neighbors ON THE SENSOR'S SPHERE OF VIEW — no 3-D k-nearest-neighbor
// search, no spatial hash, no sort. Compare against 02.04 and 02.05: those
// projects spend most of their kernel budget FINDING neighbors (voxel
// hashing + 27-cell stencil, or a k-d tree); here, neighbors are already
// sitting at index+-1 in a flat array. The one place this shortcut breaks is
// exactly the place a learner should notice: at RANGE DISCONTINUITIES, image
// neighbors can be meters apart in 3-D — but that break is not a bug, it is
// the signal object boundaries live in (Bogoslavskyi & Stachniss's central
// insight, "THE BETA CRITERION" below).
//
// THE PIPELINE (every stage single-sourced here, run in this order by
// main.cu; kernels.cu implements the GPU side, reference_cpu.cpp the
// independent CPU twin of each):
//
//   1. RANGE-IMAGE CONVERSION (both directions, 02.02 lineage, cited):
//      (a) unorganized -> organized: the committed sample is a flat list of
//          valid LiDAR returns, each already carrying its native (ring,
//          azimuth_bin) — exactly what a real driver packet reports (see
//          PRACTICE.md section 1) — scattered into the organized grid via
//          the SAME nearest-wins encoded-atomicMin race 02.02's
//          scatter_to_organized_kernel teaches (extended here only in that
//          the encoded key also carries enough to recover the point's TRUTH
//          label for the gates below).
//      (b) organized -> unorganized: after ground removal, the surviving
//          OBSTACLE cells are compacted back into a flat point list (an
//          atomic-counter append — a simpler compaction than 02.02's
//          Blelloch-scan version, appropriate at this project's cell count;
//          02.02 is cited as the scan-based alternative) so the Euclidean
//          comparison clustering (stage 6) can consume it as an ordinary
//          unorganized cloud, the same shape 02.04 expects.
//   2. GROUND REMOVAL (range-image-native, compact — contrast with 02.03's
//      full RANSAC/CZM treatment, cited): one thread per azimuth COLUMN,
//      walking its 16 rings bottom-up, testing the vertical angle between
//      consecutive returns (and a virtual point at the known sensor mount
//      height for the very first return) against a flatness threshold.
//   3. DEPTH-CLUSTERING EDGES — THE BETA CRITERION (Bogoslavskyi &
//      Stachniss, IROS 2016, cited in full in THEORY.md): for every
//      obstacle cell's two "forward" image neighbors (ring+1 same column;
//      same ring, column+1 WITH WRAP-AROUND — the sensor spins a full
//      circle), compute the angle beta the line between the two returns
//      makes with the closer return's own line of sight. Large beta =
//      continuous surface (connect); small beta = a range discontinuity =
//      an object boundary (cut). One threshold, one derivation, reused for
//      every pair in the image — see THEORY.md "The math" for the triangle.
//   4. UNION-FIND (Method A from 02.04, cited and reused near-verbatim: the
//      lock-free path-halving/union-by-min sweep loop is GENERIC over any
//      edge list, so the SAME three kernels below cluster BOTH the
//      depth-image graph (stage 3) and the Euclidean-comparison graph
//      (stage 6) — only the edges differ).
//   5. EUCLIDEAN COMPARISON CLUSTERING (02.04 lineage, cited and adapted to
//      this project's much smaller, HOLLOW-surface point count): voxel-hash
//      the compacted obstacle points at leaf = the cluster tolerance,
//      27-cell stencil + sorted-array binary search for neighbor
//      candidates, exact-distance test, union-find (stage 4's kernels
//      again) over the resulting edges. This is the "no image, pure 3-D
//      distance" baseline the project measures against.
//   6. RELABELING + STATS + MIN-SIZE FILTER — small bookkeeping over at most
//      kNumCells (16,384) or the compacted obstacle count (both trivially
//      small next to 02.04's regime), done on the HOST after the two timed
//      GPU stages above: a deliberate, documented scope choice (see the
//      note above launch_uf_finalize below) that keeps kernels.cu focused
//      on this project's two NOVEL kernel ideas (image-stencil beta edges,
//      column-serial ground removal) rather than re-deriving 02.04's
//      GPU relabel-via-scan machinery for data this small.
//
// Why this header is CUDA-qualifier-free where possible (02.01/02.04
// precedent, cited) — pure math/bookkeeping functions below (cell indexing,
// the beta formula, voxel-key packing, squared distance) are PLAIN inline
// C++ — no __host__/__device__ — so they compile under BOTH nvcc (main.cu,
// kernels.cu's host code, reference_cpu.cpp's shared-formula calls) and
// cl.exe (reference_cpu.cpp). Being unqualified, they are HOST-only under
// nvcc's rules; kernels.cu carries its OWN literal __device__ transcription
// of each one used inside a kernel, cross-referenced in comments at both
// copies. The independence ruling for the two "clever" algorithmic cores
// (union-find, and the beta/ground-removal walks) is stated in
// reference_cpu.cpp's file header, mirroring 02.04's identical ruling.
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cstdint>       // int32_t, uint32_t, uint64_t — exact-width integers for keys/encoding
#include <cmath>         // std::floor, std::sqrt, std::atan2, std::sin, std::cos — host math
#include <cstring>       // std::memcpy — float->uint32 bit reinterpretation (range encoding)
#include <vector>        // reference_cpu.cpp's independent edge lists / union-find output
#include <utility>       // std::pair<int,int> — canonicalized (u<v) edges
#include <unordered_map> // reference_cpu.cpp's independent voxel->points map (Euclidean edges)

// ===========================================================================
// Organized-grid shape — IDENTICAL to 02.02's kernels.cuh (cited, reused
// verbatim: same 16-beam elevation table, same ring-major cell formula) so a
// learner who has read 02.02 recognizes this instantly. MUST match
// scripts/make_synthetic.py's NUM_BEAMS / AZIMUTH_BINS; main.cu asserts this
// against the committed sample's header at load time (02.01's discipline).
// ===========================================================================
constexpr int kNumBeams       = 16;
constexpr int kAzimuthBins    = 1024;                          // 360/1024 = 0.3516 deg/column
constexpr int kNumCells       = kNumBeams * kAzimuthBins;       // 16,384 organized-grid cells

// 16-beam elevation table (RADIANS), cited verbatim from 02.01/02.02:
// -15..+15 deg in 2 deg steps. Index i = ring i; ring 0 is the BOTTOM beam
// (most negative elevation) — "walk rings bottom-up" (ground removal, stage
// 2) means increasing ring index.
constexpr float kBeamElevRad[kNumBeams] = {
    -0.26179939f, -0.22689280f, -0.19198622f, -0.15707963f,
    -0.12217305f, -0.08726646f, -0.05235988f, -0.01745329f,
     0.01745329f,  0.05235988f,  0.08726646f,  0.12217305f,
     0.15707963f,  0.19198622f,  0.22689280f,  0.26179939f
};  // = radians(-15), radians(-13), ..., radians(+15)

constexpr float kPi = 3.14159265358979323846f;

// organized_cell_index — the ring-major flat-index formula (02.02, cited).
inline int organized_cell_index(int ring, int az_bin)
{
    return ring * kAzimuthBins + az_bin;
}

// azimuth_step_rad — the constant angular step between adjacent COLUMNS
// (the "alpha" the beta criterion uses for an azimuth-direction edge). A
// plain constexpr SCALAR (unlike kBeamElevRad, an array) is usable directly
// from device code with no separate __device__ transcription needed — a
// compile-time immediate, not a host-memory lookup (contrast the array
// case, "why this header is CUDA-qualifier-free" above).
constexpr float kAzimuthStepRad = 2.0f * kPi / static_cast<float>(kAzimuthBins);

// ===========================================================================
// Repo-default launch configuration + ceiling-division helper (02.01/02.04
// idiom, cited).
// ===========================================================================
constexpr int kThreadsPerBlock = 256;

inline int blocks_for(int count, int threads)
{
    return (count + threads - 1) / threads;
}

// ===========================================================================
// Sensor + scene constants — MUST match scripts/make_synthetic.py's Python
// mirrors (documented match, not a machine-checked assert, following 02.03's
// precedent for geometry-DESIGN constants: main.cu DOES separately assert
// the safety-critical fields — grid shape, thresholds, tolerances — that ARE
// stored in the sample header; see load_scene() in main.cu).
// ===========================================================================

// kSensorHeightM — the LiDAR's mount height above the true ground plane
// (meters). Ground removal's column walk needs this as the VIRTUAL first
// reference point (rho=0, z=-kSensorHeightM) for the very first return in
// each column — see ground_removal_kernel's derivation in kernels.cu and
// THEORY.md "The math". This is the standard "known mounting height" prior
// every practical ground-angle method (Zermas et al. 2017; LeGO-LOAM's
// ground plane removal) relies on.
constexpr float kSensorHeightM = 1.5f;

// kMaxRangeM — beyond this, a beam that hit nothing reports NO RETURN (a
// physically honest dropout: open sky, or the ground too far away for a
// shallow beam to reach within sensor range — see scripts/make_synthetic.py
// module docstring for the exact geometry this produces).
constexpr float kMaxRangeM = 18.0f;

// ---------------------------------------------------------------------------
// kGroundAngleThresholdDeg — ground removal's flatness test (stage 2). The
// angle between consecutive column returns (or the first return and the
// virtual sensor-height reference) is compared against this: <= threshold
// means "close enough to horizontal to be ground", > threshold means
// "this return broke away from the ground plane" (an obstacle's near-
// vertical face). 10 deg comfortably separates our flat ground (angle ~= 0
// deg, only float/ray-cast noise) from any of the scene's standing objects
// (near-vertical faces read close to 90 deg) — see THEORY.md "The math" for
// the full derivation and the noise-sensitivity analysis.
// ---------------------------------------------------------------------------
constexpr float kGroundAngleThresholdDeg = 10.0f;

// ---------------------------------------------------------------------------
// kBetaThresholdDeg — THE depth-clustering decision boundary (stage 3):
// connect an image-adjacent pair iff beta >= this. 10 deg is the value most
// commonly cited from Bogoslavskyi & Stachniss's own reference
// implementation (github.com/PRBonn/depth_clustering); THEORY.md "The math"
// derives WHY a continuous, roughly sensor-facing surface produces beta
// approaching 90 deg regardless of the angular step, while a genuine range
// discontinuity (this project's depth-gap showcase cohort) produces beta
// within a few degrees of 0 — a wide, robust margin either side of 10.
// ---------------------------------------------------------------------------
constexpr float kBetaThresholdDeg = 10.0f;

// kMinDepthClusterSize — a depth-image connected component smaller than
// this many cells is reported as noise, not an object (mirrors 02.04's
// kMinClusterSize=5 reasoning: a 1-4 cell blob carries no shape evidence).
// This is also the floor scripts/make_synthetic.py's "thin pole" cohort is
// designed to sit near — see README "Expected output" for the measured
// outcome (the pole may or may not clear this floor; THEORY.md/README
// discuss the trade honestly either way).
constexpr int kMinDepthClusterSize = 5;

// kEuclideanClusterToleranceM — the Euclidean comparison's distance
// threshold AND (02.04's identical proof applies verbatim) the voxel leaf
// fed to its spatial hash, so the 27-cell stencil is a provably sufficient
// neighbor search (02.04's kernels.cuh derives the leaf==d coverage proof;
// cited, not re-derived here). Reused at 02.04's EXACT value on purpose:
// this project's "near-touching, depth-separated" showcase cohort (the
// person standing 0.30 m in front of a wall) is designed so that gap is
// SMALLER than this tolerance — Euclidean clustering merges person and
// wall; the beta criterion, being range-RATIO based rather than a fixed
// metric distance, does not (README "The comparison" states the measured
// outcome; never assumed, always the actual gate result).
constexpr float kEuclideanClusterToleranceM = 0.40f;

constexpr int kMinEuclideanClusterSize = 5;   // same floor, same reasoning, applied to the comparison baseline

// Sentinel: a cell/point's final cluster id once its raw component was
// rejected by the min-size filter (02.04's kNoCluster convention, reused).
constexpr int32_t kNoCluster = -1;

// kMaxEdgesPerPointEuclid — the per-point candidate-edge ceiling the
// Euclidean comparison's edge buffer is sized with (capacity =
// M * kMaxEdgesPerPointEuclid, M = compacted obstacle point count).
// Derivation: this project's DENSEST surface is the "wall_behind" panel
// (scripts/make_synthetic.py WALL_BEHIND), a 2 m x 2 m near-range (~2.45 m)
// flat panel. At that range, column spacing ~= r*kAzimuthStepRad ~= 0.015 m
// and ring spacing ~= r*2deg-in-rad ~= 0.086 m, so a disk of radius d=0.40 m
// on that surface contains roughly pi*d^2 / (0.015*0.086) ~= 390 grid
// samples — noticeably MORE than 02.04's filled-block estimate of ~34,
// because a near-range organized-grid surface sample is much denser than a
// uniform 3-D voxel fill. 1024 is a generous, documented ceiling over that
// estimate (matching 02.04's "size to a worst case, then detect and report
// overflow" discipline — overflow_count below is the honest safety net, not
// this bound).
constexpr int kMaxEdgesPerPointEuclid = 1024;

// Safety caps on the union-find convergence loop main.cu drives (never
// expected to be hit on graphs this small and shallow — a documented,
// bounded failure signal rather than a silent infinite loop, 02.04's
// identical policy).
constexpr int kMaxUfSweeps = 128;

// ===========================================================================
// Range-image conversion (stage 1a): 64-bit (range, index)-encoded atomicMin
// key — IDENTICAL technique to 02.02's kernels.cuh (cited, reused verbatim):
// packing the winning POINT INDEX alongside the range makes the "nearest
// wins" race deterministic even on an exact range tie (never exercised by
// the ray-cast points here, but exercised on purpose by two synthetic
// duplicate-target points scripts/make_synthetic.py adds specifically to
// exhaut the collision path — see that script's module docstring).
// ===========================================================================
inline uint32_t float_range_to_sortable_u32(float r)
{
    uint32_t bits;
    std::memcpy(&bits, &r, sizeof(bits));   // bit-for-bit reinterpretation, no rounding
    return bits;                            // valid for r >= 0 (a physical range): larger r -> larger bits
}

inline uint64_t pack_range_index(float range_m, uint32_t point_idx)
{
    return (static_cast<uint64_t>(float_range_to_sortable_u32(range_m)) << 32) | point_idx;
}

inline uint32_t unpack_point_index(uint64_t encoded)
{
    return static_cast<uint32_t>(encoded & 0xFFFFFFFFu);
}

// ALL BITS SET sentinel: larger than any valid encoded key (02.02's proof
// applies verbatim), so a single byte-fill (cudaMemset(ptr,0xFF,bytes))
// correctly initializes "empty" for every cell.
constexpr unsigned long long kEmptyCellEncoded = ~0ull;

// ===========================================================================
// Ground removal (stage 2) — the column-walk math, shared host formula.
// ===========================================================================

// ground_step_angle_deg — the vertical angle (DEGREES) the segment from
// (rho_prev,z_prev) to (rho,z) makes with the HORIZONTAL. atan2 handles
// every sign combination (including rho - rho_prev <= 0, the exact
// situation a near-radial depth jump produces — see THEORY.md "The math"):
// near 0 deg = flat/ground-like step, near +-90 deg = a steep/near-vertical
// step (an obstacle face, or a radial depth jump onto a farther surface).
inline float ground_step_angle_deg(float rho_prev, float z_prev, float rho, float z)
{
    const float d_rho = rho - rho_prev;
    const float d_z   = z - z_prev;
    return std::atan2(d_z, d_rho) * (180.0f / kPi);
}

// ===========================================================================
// Depth clustering (stage 3) — THE BETA CRITERION (Bogoslavskyi & Stachniss,
// cited in full in THEORY.md "The math", which derives this formula from
// the O-A-B line-of-sight triangle). r1 = the FARTHER of the two returns,
// r2 = the NEARER; alpha = the angular step between the two beams (either
// kAzimuthStepRad for a same-ring/adjacent-column pair, or the local ring
// elevation difference for a same-column/adjacent-ring pair). Returns beta
// in RADIANS; callers compare against kBetaThresholdDeg converted once.
//
// beta = atan2(r2*sin(alpha), r1 - r2*cos(alpha))
//
// Large beta (-> 90 deg as r1 -> r2): the two returns lie on a surface
// roughly PERPENDICULAR to the bisector of the two beams — a continuous
// object face. Small beta (-> 0 as r1 >> r2 or vice versa): the segment
// between the two returns points nearly ALONG the line of sight — a range
// discontinuity, almost always an occlusion boundary between two different
// objects (or object and background).
// ===========================================================================
inline float beta_criterion_rad(float r1, float r2, float alpha)
{
    return std::atan2(r2 * std::sin(alpha), r1 - r2 * std::cos(alpha));
}

constexpr float kBetaThresholdRad = kBetaThresholdDeg * (kPi / 180.0f);
constexpr float kGroundAngleThresholdRad = kGroundAngleThresholdDeg * (kPi / 180.0f);

// ===========================================================================
// Euclidean comparison clustering (stage 5) — voxel key packing IDENTICAL to
// 02.01/02.04's kernels.cuh (cited, not reinvented; re-declared here per the
// self-containment rule — CLAUDE.md section 4 — rather than included across
// project folders).
// ===========================================================================
constexpr int32_t  kCoordBias   = 1 << 20;             // recenters [-2^20,2^20-1] to [0,2^21-1]
constexpr uint64_t kCoordMask21 = (1ull << 21) - 1ull;  // low 21 bits
constexpr uint64_t kEmptyKey    = ~0ull;                // sentinel: never a valid packed key

inline int32_t voxel_coord(float p, float leaf)
{
    return static_cast<int32_t>(std::floor(p / leaf));   // floor, not truncation — negative-coordinate safe
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

inline float squared_distance(const float p[3], const float q[3])
{
    const float dx = p[0] - q[0], dy = p[1] - q[1], dz = p[2] - q[2];
    return dx * dx + dy * dy + dz * dz;   // monotonic in |p-q| for a nonnegative test — no sqrt needed
}

// ===========================================================================
// GPU kernel declarations — nvcc-only (cl.exe, compiling reference_cpu.cpp,
// has never heard of __global__ and must never see these).
// ===========================================================================
#ifdef __CUDACC__

// ---- Stage 1a: unorganized -> organized (range-image conversion) ---------

// scatter_encode_kernel — one thread per INPUT point: compute this point's
// cell (already carries its own ring/az_bin — a real driver reports these,
// see PRACTICE.md section 1), encode (range,point_idx), and atomicMin-race
// it into cell_encoded[cell]. Many points never collide (each ray-cast
// point already owns a unique cell) EXCEPT the two synthetic collision-test
// points scripts/make_synthetic.py adds on purpose — see that script and
// VERIFY(range_image) in main.cu.
__global__ void scatter_encode_kernel(int n_points,
                                      const int* __restrict__ ring, const int* __restrict__ az_bin,
                                      const float* __restrict__ range_m,
                                      unsigned long long* __restrict__ cell_encoded);

// finalize_organized_kernel — one thread per CELL: decode the winning point
// index (kEmptyCellEncoded => no return), and copy that point's
// x/y/z/range/truth_id into the organized arrays; an empty cell gets range=0
// (the "no return" sentinel this project uses throughout — see kernels.cuh
// file header) and truth_id=-1.
__global__ void finalize_organized_kernel(int num_cells,
                                          const unsigned long long* __restrict__ cell_encoded,
                                          const float* __restrict__ px, const float* __restrict__ py,
                                          const float* __restrict__ pz, const float* __restrict__ prange,
                                          const int* __restrict__ ptruth,
                                          float* __restrict__ range_img,
                                          float* __restrict__ xyz_img,
                                          int* __restrict__ truth_img,
                                          int* __restrict__ winner_idx_img);

// ---- Stage 1b: organized -> unorganized (obstacle compaction) ------------

// compact_obstacles_kernel — one thread per CELL: if obstacle_mask[cell] is
// set, atomically append its xyz + cell index + truth id to the flat output
// arrays (an atomic-counter push — a simpler compaction than 02.02's
// Blelloch-scan version, appropriate at this project's small cell count;
// see kernels.cuh file header). out_* arrays sized [num_cells] (a safe
// upper bound); *out_count is the true compacted length on return.
__global__ void compact_obstacles_kernel(int num_cells,
                                         const float* __restrict__ range_img,
                                         const float* __restrict__ xyz_img,
                                         const int* __restrict__ obstacle_mask,
                                         const int* __restrict__ truth_img,
                                         float* __restrict__ out_xyz,
                                         int* __restrict__ out_cell_idx,
                                         int* __restrict__ out_truth,
                                         int* __restrict__ out_count);

// ---- Stage 2: ground removal (column-serial walk) -------------------------

// ground_removal_kernel — one thread PER AZIMUTH COLUMN (kAzimuthBins
// threads total): walks its kNumBeams cells bottom-up (ring 0 -> 15),
// skipping cells with no return (range_img==0), applying
// ground_step_angle_deg() against the previous VALID return (or the virtual
// sensor-height reference for the column's first valid return). Writes
// ground_label[cell] (0/1, meaningful only where range_img[cell]>0) and
// obstacle_mask[cell] = (range_img[cell]>0 && !ground_label[cell]).
// Launch config: one thread per column is embarrassingly parallel across
// kAzimuthBins=1024 independent columns; the 16-iteration walk WITHIN a
// thread is inherently sequential (each step's classification needs the
// previous step's position) — a small, bounded serial chain, not a
// candidate for a further parallel scan at this length (see THEORY.md "The
// GPU mapping" for why 16 steps does not justify a Blelloch-style
// decomposition the way 02.02's scan chapter does for large arrays).
__global__ void ground_removal_kernel(int num_columns,
                                      const float* __restrict__ range_img,
                                      const float* __restrict__ xyz_img,
                                      int* __restrict__ ground_label,
                                      int* __restrict__ obstacle_mask);

// ---- Stage 3: depth-clustering edges (the beta criterion) -----------------

// depth_edges_kernel — one thread per CELL: tests the two FORWARD image
// neighbors only (ring+1 same column, clamped at the top ring; same ring,
// column+1 WITH WRAP-AROUND at kAzimuthBins-1 -> 0 — the sensor spins a
// full circle, so column 1023's "next" column is column 0, THEORY.md "The
// GPU mapping" discusses why this wrap must be explicit) — processing only
// forward neighbors means each undirected image edge is considered exactly
// once, by its lower-ring/lower-column endpoint. Both endpoints must be
// valid obstacle cells (range>0 && obstacle_mask); if so, computes beta via
// beta_criterion_rad() with r1/r2 assigned as the farther/nearer of the two
// ranges, and appends the edge (this_cell, neighbor_cell) iff
// beta >= kBetaThresholdRad. edge_u/edge_v sized [num_cells*2] (2 forward
// neighbors per cell, a tight, exact capacity — no overflow bookkeeping
// needed here, unlike the Euclidean path).
__global__ void depth_edges_kernel(int num_cells,
                                   const float* __restrict__ range_img,
                                   const int* __restrict__ obstacle_mask,
                                   int* __restrict__ edge_u, int* __restrict__ edge_v,
                                   int* __restrict__ edge_count);

// ---- Stage 4: generic lock-free GPU union-find (02.04 Method A, cited) ---

// uf_init_kernel — parent[i] = i for every element: every element starts as
// the root of its own singleton component.
__global__ void uf_init_kernel(int n, int* __restrict__ parent);

// uf_union_sweep_kernel — ONE SWEEP: one thread per EDGE; find() (path
// halving) both endpoints, union-by-min via an atomicCAS retry loop if the
// roots differ. Sets *changed nonzero iff a union happened this sweep — see
// 02.04's kernels.cuh "THE UNION-FIND CHAPTER" for the full correctness
// argument (the monotone-parent property), reused verbatim here since the
// algorithm is GENERIC over any edge list.
__global__ void uf_union_sweep_kernel(int num_edges, const int* __restrict__ edge_u,
                                      const int* __restrict__ edge_v,
                                      int* __restrict__ parent, int* __restrict__ changed);

// uf_finalize_kernel — one thread per element: a full find-to-fixpoint, so
// parent[i] is EXACTLY that element's canonical root on return (no further
// indirection) — the postcondition every downstream gate assumes.
__global__ void uf_finalize_kernel(int n, int* __restrict__ parent);

// ---- Stage 5: Euclidean comparison clustering (02.04 lineage, adapted) ---

// compute_voxel_keys_kernel — one thread per point: pack this point's voxel
// key at leaf = kEuclideanClusterToleranceM.
__global__ void compute_voxel_keys_kernel(int n, const float* __restrict__ xyz,
                                          unsigned long long* __restrict__ keys);

// mark_boundaries_kernel — one thread per SORTED-ARRAY position: 1 where a
// new voxel's run of points begins (02.01 Method B / 02.04, cited).
__global__ void mark_boundaries_kernel(int n, const unsigned long long* __restrict__ keys_sorted,
                                       int* __restrict__ is_start);

// gather_unique_keys_kernel — one thread per OCCUPIED voxel: copy its key
// into a dense ascending-sorted unique_key[] array (02.04, cited).
__global__ void gather_unique_keys_kernel(int num_voxels, const int* __restrict__ seg_start,
                                          const unsigned long long* __restrict__ keys_sorted,
                                          unsigned long long* __restrict__ unique_key_out);

// build_edges_euclid_kernel — one thread per point i: scan the 27-voxel
// stencil around i's own voxel via a device binary search over
// unique_key[0..num_voxels), walk each occupied voxel's point run, and
// atomically append an edge (i,j) for every candidate j>i with
// squared_distance<=d*d (02.04's kernels.cu, cited and reused near-verbatim
// — see kernels.cu for the full walkthrough). overflow_count is incremented
// (never silently dropped) if a point's edge count would exceed
// kMaxEdgesPerPointEuclid.
__global__ void build_edges_euclid_kernel(int n, const float* __restrict__ xyz,
                                          const unsigned long long* __restrict__ point_key,
                                          const unsigned long long* __restrict__ unique_key, int num_voxels,
                                          const int* __restrict__ seg_start,
                                          const int* __restrict__ idx_sorted, int n_sorted,
                                          int* __restrict__ edge_u, int* __restrict__ edge_v, int edge_capacity,
                                          int* __restrict__ edge_count, int* __restrict__ overflow_count);

#endif // __CUDACC__

// ===========================================================================
// Host-callable launch wrappers (definitions in kernels.cu, nvcc-only file;
// these DECLARATIONS are plain C++, visible to main.cu).
// ===========================================================================

void launch_scatter_to_organized(int n_points,
                                 const int* d_ring, const int* d_az_bin, const float* d_range_m,
                                 const float* d_px, const float* d_py, const float* d_pz,
                                 const int* d_truth,
                                 float* d_range_img, float* d_xyz_img,
                                 int* d_truth_img, int* d_winner_idx_img);

void launch_ground_removal(const float* d_range_img, const float* d_xyz_img,
                           int* d_ground_label, int* d_obstacle_mask);

// launch_compact_obstacles — the organized->unorganized direction end to
// end (memset the atomic counter, launch, read back the count). Returns the
// number of compacted obstacle points M (<= kNumCells).
int launch_compact_obstacles(const float* d_range_img, const float* d_xyz_img,
                             const int* d_obstacle_mask, const int* d_truth_img,
                             float* d_out_xyz, int* d_out_cell_idx, int* d_out_truth);

// launch_depth_edges — depth_edges_kernel's wrapper: resets the atomic edge
// counter, launches, reads back the count. Returns the edge count
// (<= num_cells*2 by construction — no overflow possible, see kernels.cuh).
int launch_depth_edges(const float* d_range_img, const int* d_obstacle_mask,
                       int* d_edge_u, int* d_edge_v);

void launch_uf_init(int n, int* d_parent);
bool launch_uf_union_sweep(int num_edges, const int* d_edge_u, const int* d_edge_v,
                           int* d_parent, int* d_changed);
void launch_uf_finalize(int n, int* d_parent);

// launch_build_voxel_index — Thrust sort_by_key(keys) + boundary compaction
// (02.01 Method-B / 02.04 lineage, cited). Returns the number of OCCUPIED
// voxels.
int launch_build_voxel_index(int n, const unsigned long long* d_keys_in,
                             unsigned long long* d_keys_scratch, int* d_idx_sorted,
                             int* d_is_start_scratch, int* d_seg_start_out,
                             unsigned long long* d_unique_key_out);

void launch_compute_voxel_keys(int n, const float* d_xyz, unsigned long long* d_keys);

// launch_build_edges_euclid — build_edges_euclid_kernel's wrapper. Returns
// the edge count (<= n*kMaxEdgesPerPointEuclid).
int launch_build_edges_euclid(int n, const float* d_xyz, const unsigned long long* d_point_key,
                              const unsigned long long* d_unique_key, int num_voxels,
                              const int* d_seg_start, const int* d_idx_sorted, int n_sorted,
                              int* d_edge_u, int* d_edge_v, int edge_capacity,
                              int* d_overflow_count);

// ===========================================================================
// CPU references (reference_cpu.cpp) — the correctness-oracle twins. All
// pointers below are HOST pointers. See reference_cpu.cpp's file header for
// the independence ruling each of these follows (mirroring 02.02's and
// 02.04's identical rulings for the analogous stages).
// ===========================================================================

// scatter_to_organized_cpu — INDEPENDENT twin of stage 1a: a plain
// per-cell "running minimum range" scan over all input points (no encoding,
// no atomics — see 02.02's identical CPU-twin technique, cited) — order-
// independent (min is commutative/associative) so it is directly, exactly
// comparable to the GPU's atomicMin race.
void scatter_to_organized_cpu(int n_points,
                              const int* ring, const int* az_bin, const float* range_m,
                              const float* px, const float* py, const float* pz, const int* truth,
                              float* range_img, float* xyz_img, int* truth_img, int* winner_idx_img);

// ground_removal_cpu — INDEPENDENT twin of stage 2: the same column-walk
// RULE (it is the definition of "ground" this project promises, not an
// implementation detail — see reference_cpu.cpp file header), but a
// completely separate, independently-written sequential loop.
void ground_removal_cpu(const float* range_img, const float* xyz_img,
                        int* ground_label, int* obstacle_mask);

// depth_edges_cpu — INDEPENDENT twin of stage 3: a plain double nested loop
// (cells x 2 forward neighbors), the SAME beta_criterion_rad() shared
// formula (a data-layout/definition formula, not an algorithm — see file
// header), returned as a canonicalized ascending-sorted edge vector for
// main.cu's set-equality VERIFY.
std::vector<std::pair<int,int>> depth_edges_cpu(const float* range_img, const int* obstacle_mask);

// build_edges_euclid_cpu — INDEPENDENT twin of stage 5's edge construction:
// an std::unordered_map<uint64_t,std::vector<int>> voxel->points map (a
// different data structure from the GPU's sorted-array+binary-search index
// — 02.04's identical independence argument, cited), same 27-cell stencil,
// same i<j/distance rule, returned sorted ascending.
std::vector<std::pair<int,int>> build_edges_euclid_cpu(int n, const float* xyz);

// serial_union_find_cpu — INDEPENDENT sequential union-find (02.04's
// identical twin, reused as a GENERIC function over any edge list — used to
// verify BOTH the depth-image graph and the Euclidean-comparison graph).
// Union-by-min's final partition is mathematically order-independent, so
// this is expected to match the GPU's finalized parent[] BIT-EXACT.
void serial_union_find_cpu(int n, const std::vector<std::pair<int,int>>& edges, std::vector<int>& parent_out);

#endif // PROJECT_KERNELS_CUH
