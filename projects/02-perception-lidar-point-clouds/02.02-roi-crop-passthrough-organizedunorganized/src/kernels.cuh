// ===========================================================================
// kernels.cuh — interface & contract for project 02.02
//               ROI crop, passthrough, organized<->unorganized conversion
//               kernels — a STREAM-COMPACTION masterclass, prefix scan first.
//
// Role in the project
// -------------------
// The CONTRACT shared by main.cu (orchestration + gates), kernels.cu (the
// GPU kernels), and reference_cpu.cpp (the CPU oracle twins). Data-layout
// constants, predicate formulas, and encoding schemes are SINGLE-SOURCED
// here (CLAUDE.md §12) — see the "independence ruling" in reference_cpu.cpp
// for exactly what is shared here versus reimplemented independently there.
//
// POINT CLOUD LAYOUT — float* xyz, interleaved, meters, "lidar" sensor
// frame: xyz[i*3+0]=x, xyz[i*3+1]=y, xyz[i*3+2]=z. IDENTICAL to 02.01 and
// 02.06 (this domain's precedent — cited) and docs/SYSTEM_DESIGN.md §3.6's
// `PointCloud` sketch, so this project's output slots directly into 02.06's
// `src_xyz`/`dst_xyz` inputs and 02.01's `d_xyz` input with zero reshaping.
//
// THE PROBLEM in one paragraph
// -----------------------------
// Every stage between a LiDAR driver and every consumer needs to (a) throw
// away points nobody downstream wants (range gating, a region of interest,
// "only what the camera can see") and (b) translate between the sensor's
// native ORGANIZED (ring x azimuth) grid and the UNORGANIZED flat point
// lists almost every algorithm in this domain expects. Both problems reduce
// to the SAME primitive: STREAM COMPACTION — decide, per element, keep or
// drop (a predicate), then pack the survivors into a dense array PRESERVING
// their relative order. The exclusive prefix sum ("scan") of the 0/1
// keep-flags is exactly the survivor's destination index — this file's
// docs/THEORY.md "The GPU mapping" derives that fact in full; kernels.cu
// implements it three ways (hand-rolled two-level Blelloch scan, Thrust,
// and a CPU serial loop) so a learner can see the SAME answer emerge from
// three genuinely different pieces of code.
//
// SIX KERNELS, ONE PRIMITIVE — the project's teaching arc:
//   1) PASSTHROUGH   — axis-range predicate (z in [zmin,zmax]).
//   2) BOX ROI CROP  — axis-aligned-box predicate.
//   3) FRUSTUM CROP  — camera-frustum predicate (the sensor-fusion use
//      case: keep only points the CAMERA can see — cited from 01.18's
//      camera<->LiDAR extrinsic and 02.17's downstream fusion use).
//   4) CHAINED vs FUSED — three predicates applied as three compaction
//      passes vs. one conjoined predicate in a single pass: bit-identical
//      answers, different memory traffic (the fusion lesson, cited from
//      01.01/01.23's ISP-stage-fusion precedent).
//   5) ORGANIZED -> UNORGANIZED — compaction with the "keep" predicate
//      being simply "not NaN": drops invalid cells out of the sensor's
//      native ring x azimuth grid.
//   6) UNORGANIZED -> ORGANIZED — the OPPOSITE direction: scatter each
//      point into its computed (ring,azimuth) cell, resolving collisions
//      with a NEAREST-WINS atomicMin race on a 64-bit encoded (range,
//      index) key — extending 01.18's uint-encoded atomicMin z-buffer
//      trick (cited) from 32 bits (range only) to 64 bits (range + a
//      point-index tiebreaker), so the winner is always a real, traceable
//      point index, never an ambiguous "whoever got there first".
//
// ORGANIZED GRID LAYOUT — ring-major flat index:
//     cell_index(ring, azimuth_bin) = ring * kAzimuthBins + azimuth_bin
// ring in [0, kNumBeams), azimuth_bin in [0, kAzimuthBins). An invalid cell
// (no return: geometric miss OR the independent absorption/glare dropout —
// THEORY.md "The problem" explains both) stores three IEEE-754 NaN floats;
// is_invalid_point() below tests the x-coordinate ONLY (by construction all
// three coordinates of an invalid cell are NaN together — one check
// suffices and is the convention every kernel/reference here relies on).
//
// Why this header is CUDA-qualifier-free (following 02.01's precedent
// exactly, cited) — every predicate/encoding function below is a PLAIN
// inline C++ function, deliberately WITHOUT __host__/__device__, so it
// compiles under BOTH nvcc (main.cu, reference_cpu.cpp's caller side is
// actually cl.exe — see next line) and cl.exe (reference_cpu.cpp). The
// cost: these are HOST-only under nvcc's rules and cannot be called from a
// __global__ kernel. kernels.cu's device code therefore carries its OWN
// literal __device__ transcription of every predicate/encoding formula
// (commented as such at each definition) — the "shared token-for-token
// transcription" case reference_cpu.cpp's independence ruling explicitly
// permits, PROVIDED an independent gate would catch drift between the two
// copies. That gate is main.cu's VERIFY(predicate_correctness) /
// GATE frustum_geometry / GATE collision_accounting — each compares a GPU
// result computed via kernels.cu's device transcription against a CPU
// result computed via THIS header's shared host functions (routed through
// reference_cpu.cpp's independently-structured serial algorithms).
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cstdint>   // uint32_t/uint64_t — the range/index encoding words
#include <cmath>     // std::atan2, std::sqrt, std::fabs — host-side geometry
#include <cstring>   // std::memcpy — the float->uint32 bit reinterpretation

// ===========================================================================
// Organized-grid shape. MUST match scripts/make_synthetic.py's NUM_BEAMS /
// AZIMUTH_BINS exactly — main.cu asserts this against the sample file's
// header at load time (the same discipline 02.01 applies to its leaf_m).
// ===========================================================================
constexpr int kNumBeams    = 16;
constexpr int kAzimuthBins = 1024;                          // 0.3516 deg/step
constexpr int kOrganizedCells = kNumBeams * kAzimuthBins;   // 16384 cells

// 16-beam elevation table (RADIANS), cited verbatim from 01.18's THEORY.md /
// reused from 02.01's make_synthetic.py: -15..+15 deg in 2 deg steps. Stored
// pre-converted to radians so nearest_ring_of() below does no per-call
// deg->rad conversion. Index i corresponds to ring i.
constexpr float kBeamElevRad[kNumBeams] = {
    -0.26179939f, -0.22689280f, -0.19198622f, -0.15707963f,
    -0.12217305f, -0.08726646f, -0.05235988f, -0.01745329f,
     0.01745329f,  0.05235988f,  0.08726646f,  0.12217305f,
     0.15707963f,  0.19198622f,  0.22689280f,  0.26179939f
};  // = radians(-15), radians(-13), ..., radians(+15)

constexpr float kPi = 3.14159265358979323846f;

// ===========================================================================
// Predicate bounds — MUST match scripts/make_synthetic.py's Python mirrors
// exactly (that script places the "edge cohort" points at +-EDGE_EPS around
// these same numbers, so main.cu's predicate_correctness gate can exercise
// the <=/>= boundary comparisons precisely). Meters, "lidar" frame.
// ===========================================================================

// PASSTHROUGH: keep points with z in [kPassthroughZMin, kPassthroughZMax] —
// a body-height band that drops ground clutter (z near -1.5 m, the floor)
// and overhangs/wall tops (z near +1.5 m) in one axis-range test.
constexpr float kPassthroughZMin = -1.0f;
constexpr float kPassthroughZMax =  0.5f;

// BOX ROI: an axis-aligned crop of the room's central footprint, keeping
// obstacles relevant to a robot operating near the sensor and discarding
// the far walls.
constexpr float kBoxMin[3] = { -4.0f, -4.0f, -1.5f };
constexpr float kBoxMax[3] = {  4.0f,  4.0f,  1.0f };

// FRUSTUM: camera intrinsics IDENTICAL to 01.18's teaching camera (cited):
// 160x120 px, fx=154, fy=152 (~56.5deg x ~44.7deg FOV), principal point at
// image center-ish. kFrustumNearM is this project's own near-plane choice
// (01.18 has no near-plane concept; a LiDAR-camera fusion crop needs one so
// points BEHIND the camera, or too close for either sensor to agree on,
// are rejected before the pixel-bounds test even applies).
constexpr float kFx = 154.0f, kFy = 152.0f, kCx = 80.0f, kCy = 60.0f;
constexpr int   kImgW = 160,  kImgH = 120;
constexpr float kFrustumNearM = 0.5f;

// Rigid3 — a rigid-body transform, IDENTICAL shape to 01.18's calibration
// output: P_dst = R * P_src + t, R stored ROW-MAJOR (R[r*3+c]). A pure
// data-layout struct (no member functions) — safe to include from both
// nvcc and cl.exe translation units with no __host__/__device__ decoration.
struct Rigid3 {
    float R[9];   // row-major 3x3 rotation
    float t[3];   // translation, meters, in the DESTINATION frame
};

// kTCameraLidar — P_camera = R * P_lidar + t. Reused VERBATIM (numerically
// identical rotation and translation) from 01.18's THEORY.md-derived
// extrinsic: a roof-mounted LiDAR (x-forward,y-left,z-up) above and behind
// a windshield-height camera, with a clean axis-permutation rotation
// (camera-z = lidar-x, camera-x = -lidar-y, camera-y = -lidar-z) — cited,
// not re-derived; see 01.18's THEORY.md "The math" for the full derivation
// and this project's own THEORY.md "The math" for the FRUSTUM PLANE
// derivation built on top of it (the part that IS new here).
constexpr Rigid3 kTCameraLidar = {
    { 0.0f, -1.0f,  0.0f,
      0.0f,  0.0f, -1.0f,
      1.0f,  0.0f,  0.0f },
    { 0.0f, -0.30f, -0.05f }
};

// Repo-default block size for simple per-point map kernels (predicates,
// scatter). A warp multiple with solid occupancy on sm_75..sm_89.
constexpr int kThreadsPerBlock = 256;

// ---------------------------------------------------------------------------
// blocks_for — integer ceiling division: how many `threads`-wide blocks
// cover `count` independent problems (the 02.01/02.06/08.01/33.01 idiom).
// ---------------------------------------------------------------------------
inline int blocks_for(int count, int threads)
{
    return (count + threads - 1) / threads;
}

// ===========================================================================
// Predicate & geometry helpers — the SHARED data-layout formulas (host-only;
// see the file header for why, and for the device-transcription rule).
// ===========================================================================

// is_invalid_point — the organized grid's NaN convention: x!=x is true iff
// x is NaN (the classic, portable NaN self-inequality test — works
// identically under cl.exe and nvcc with no <cmath> intrinsic needed). By
// construction every invalid cell has ALL THREE coordinates NaN together,
// so testing x alone is sufficient and is the convention every kernel here
// relies on (documented once, here, rather than at each call site).
inline bool is_invalid_point(float x)
{
    return x != x;
}

// is_passthrough — axis-range predicate on z alone. See kPassthroughZMin/Max.
inline bool is_passthrough(float z)
{
    return z >= kPassthroughZMin && z <= kPassthroughZMax;
}

// is_in_box — axis-aligned-box predicate (inclusive bounds both ends, so a
// point EXACTLY on a face is KEPT — the edge-cohort points at the exact
// bound value probe this choice; THEORY.md "Numerical considerations"
// discusses why inclusive-inclusive is the only convention that keeps
// "chained == fused" exactly true when boxes tile space edge-to-edge).
inline bool is_in_box(float x, float y, float z)
{
    return x >= kBoxMin[0] && x <= kBoxMax[0] &&
          y >= kBoxMin[1] && y <= kBoxMax[1] &&
          z >= kBoxMin[2] && z <= kBoxMax[2];
}

// transform_to_camera — p_cam = R * p_lidar + t via kTCameraLidar. Shared by
// is_in_frustum() below and by main.cu's frustum-crop top-view artifact.
inline void transform_to_camera(float x, float y, float z,
                                float& cx_out, float& cy_out, float& cz_out)
{
    const Rigid3& T = kTCameraLidar;
    cx_out = T.R[0] * x + T.R[1] * y + T.R[2] * z + T.t[0];
    cy_out = T.R[3] * x + T.R[4] * y + T.R[5] * z + T.t[1];
    cz_out = T.R[6] * x + T.R[7] * y + T.R[8] * z + T.t[2];
}

// is_in_frustum — the FIVE-PLANE test (near + left + right + top + bottom;
// "far" is deliberately omitted — passthrough/box already bound range, and
// a camera has no natural far plane the way a rasterizer's clip volume
// does — THEORY.md "The math" derives all five plane equations from
// fx/fy/cx/cy/W/H below). Every plane passes through the camera origin
// EXCEPT near; each test is one dot product/inequality, no division, no
// trig — cheap on purpose (THEORY.md "The GPU mapping" explains why this
// matters at 10-20 Hz).
inline bool is_in_frustum(float x, float y, float z)
{
    float cx, cy, cz;
    transform_to_camera(x, y, z, cx, cy, cz);
    if (cz < kFrustumNearM) return false;                                  // near
    if (kFx * cx + kCx * cz < 0.0f) return false;                          // left  (u >= 0)
    if (-kFx * cx + (static_cast<float>(kImgW - 1) - kCx) * cz < 0.0f) return false;  // right (u <= W-1)
    if (kFy * cy + kCy * cz < 0.0f) return false;                          // top   (v >= 0)
    if (-kFy * cy + (static_cast<float>(kImgH - 1) - kCy) * cz < 0.0f) return false;  // bottom(v <= H-1)
    return true;
}

// is_fused — the conjunction all three predicates: the FUSED single-pass
// filter that launch_fused_compact()/fused_compact_cpu() apply, contrasted
// against the CHAINED three-pass pipeline in main.cu's GATE fused_vs_chained.
inline bool is_fused(float x, float y, float z)
{
    return is_passthrough(z) && is_in_box(x, y, z) && is_in_frustum(x, y, z);
}

// azimuth_bin_of — atan2-based azimuth angle mapped to [0, kAzimuthBins).
// atan2 returns (-pi,pi]; we shift negative results by 2*pi to land in
// [0,2*pi) BEFORE dividing, so the bin index is always non-negative.
inline int azimuth_bin_of(float x, float y)
{
    float az = std::atan2(y, x);                 // (-pi, pi]
    if (az < 0.0f) az += 2.0f * kPi;              // [0, 2*pi)
    int bin = static_cast<int>(az / (2.0f * kPi / static_cast<float>(kAzimuthBins)));
    if (bin >= kAzimuthBins) bin = kAzimuthBins - 1;  // guard the az~=2*pi float-rounding edge
    if (bin < 0) bin = 0;                             // guard az~=0 rounding the other way
    return bin;
}

// nearest_ring_of — elevation angle mapped to the CLOSEST of the 16 known
// beam elevations (a 16-entry linear scan — cheap, and simplicity beats a
// binary search at this table size; CLAUDE.md "teaching beats cleverness").
// Every point this project re-bins into the organized grid was ORIGINALLY
// cast along one of these exact 16 directions (see kernels.cuh's callers),
// so "nearest" always lands EXACTLY on the source ring, up to float
// rounding in atan2/sqrt — never ambiguous in practice for this dataset.
inline int nearest_ring_of(float x, float y, float z)
{
    const float horiz = std::sqrt(x * x + y * y);
    const float el = std::atan2(z, horiz);
    int best = 0;
    float best_diff = 1.0e30f;
    for (int i = 0; i < kNumBeams; ++i) {
        const float diff = std::fabs(el - kBeamElevRad[i]);
        if (diff < best_diff) { best_diff = diff; best = i; }
    }
    return best;
}

// organized_cell_index — the ring-major flat-index formula (see file header).
inline int organized_cell_index(int ring, int azimuth_bin)
{
    return ring * kAzimuthBins + azimuth_bin;
}

// ---------------------------------------------------------------------------
// 64-bit (range, index)-encoded atomicMin key — extends 01.18's uint-
// encoded atomicMin z-buffer trick (cited) from 32 bits to 64.
//
// 01.18 packs ONLY the depth into 32 bits (atomicMin resolves "who is
// closer"), leaving TIES between simultaneous threads to whichever thread's
// atomicMin call the hardware scheduler happened to apply last — harmless
// there because 01.18 only needs the DEPTH VALUE, not which point produced
// it. This project's winner must be a POINT INDEX (so the finalize step
// knows which xyz to copy into the organized cell), so a tie is no longer
// harmless: two points at the identical encoded range would leave the
// "winner" ambiguous between hardware runs, breaking determinism. Packing
// the point INDEX into the low 32 bits removes the ambiguity: for any two
// DISTINCT points, their 64-bit keys are strictly ordered (equal ranges
// break the tie by index), so atomicMin always converges to the same
// answer regardless of thread scheduling — a genuinely extended technique,
// not a cosmetic 32->64 relabeling.
//
// float_range_to_sortable_u32 — reinterpret a POSITIVE, FINITE float's
// IEEE-754 bit pattern as an unsigned int. For any two positive finite
// floats, larger value implies larger bit pattern (exponent bits dominate
// the comparison and live in the high bits), so unsigned integer ordering
// matches float ordering with NO transformation needed — 01.18's THEORY.md
// derives this; every range in this project is a physical distance (>=0),
// so the general negative-float case 01.18 also documents does not apply.
// ---------------------------------------------------------------------------
inline uint32_t float_range_to_sortable_u32(float r)
{
    uint32_t bits;
    std::memcpy(&bits, &r, sizeof(bits));  // bit-for-bit reinterpretation, no rounding
    return bits;
}

inline uint64_t pack_range_index(float range_m, uint32_t point_idx)
{
    return (static_cast<uint64_t>(float_range_to_sortable_u32(range_m)) << 32) | point_idx;
}

inline uint32_t unpack_point_index(uint64_t encoded)
{
    return static_cast<uint32_t>(encoded & 0xFFFFFFFFu);
}

// Sentinel marking an empty organized-grid cell during the atomicMin race:
// ALL BITS SET is larger than any valid encoded key (a valid key's top 32
// bits are, at most, the bit pattern of a large-but-finite positive float,
// which is always < 0xFFFFFFFF), so cudaMemset(ptr, 0xFF, bytes) — a single
// byte-fill, no kernel needed — correctly initializes the whole array.
constexpr unsigned long long kEmptyCellEncoded = ~0ull;

// ===========================================================================
// Scan (prefix-sum) launch-configuration constants — see kernels.cu's
// "THE SCAN CHAPTER" for the full two-level Blelloch derivation.
// kScanElemsPerBlock = 2 * kScanBlockThreads: the classic Blelloch layout,
// each of kScanBlockThreads threads owns TWO shared-memory elements (the
// up-sweep/down-sweep tree needs a power-of-two element count per block,
// and processing 2 elements/thread halves the thread count needed for a
// given block "span" versus 1 element/thread — THEORY.md "The algorithm").
// ===========================================================================
constexpr int kScanBlockThreads   = 256;
constexpr int kScanElemsPerBlock  = 2 * kScanBlockThreads;  // 512

// ===========================================================================
// GPU kernel declarations — nvcc-only (cl.exe, compiling reference_cpu.cpp,
// has never heard of __global__ and must never see these).
// ===========================================================================
#ifdef __CUDACC__

// ---- Predicate kernels: one thread per point, write 0/1 into flags[i]. ----
__global__ void passthrough_predicate_kernel(int n, const float* __restrict__ xyz, int* __restrict__ flags);
__global__ void box_predicate_kernel(int n, const float* __restrict__ xyz, int* __restrict__ flags);
__global__ void frustum_predicate_kernel(int n, const float* __restrict__ xyz, int* __restrict__ flags);
__global__ void fused_predicate_kernel(int n, const float* __restrict__ xyz, int* __restrict__ flags);
__global__ void valid_predicate_kernel(int n, const float* __restrict__ xyz, int* __restrict__ flags);

// ---- THE SCAN CHAPTER: two-level work-efficient (Blelloch) exclusive scan. ----

// blelloch_block_scan_kernel — Kernel 1/2 of the two-level scan. Each block
// loads up to kScanElemsPerBlock flags into shared memory, runs an
// up-sweep (reduce) then down-sweep (distribute) pass to compute this
// block's LOCAL exclusive scan, writes it to out_exclusive[], and (if
// block_sums != nullptr) records this block's TOTAL sum into
// block_sums[blockIdx.x] — the raw material the second level scans next.
// Reused, unmodified, for BOTH levels of the two-level composition: level 1
// scans the (large) flags array across many blocks; level 2 scans the
// (small, <= kScanElemsPerBlock) block_sums array in a SINGLE block launch
// (block_sums passed as nullptr there — a single block has no "next level"
// to feed). See kernels.cu for the full up-sweep/down-sweep/bank-conflict
// commentary — the heart of this project's didactic core.
__global__ void blelloch_block_scan_kernel(int n, const int* __restrict__ in,
                                           int* __restrict__ out_exclusive,
                                           int* __restrict__ block_sums);

// add_block_offsets_kernel — Kernel 2/2 (level 1 only): adds this block's
// scanned offset (scanned_block_sums[blockIdx.x]) to every element in its
// range, turning kScanElemsPerBlock-sized LOCAL scans into one GLOBAL scan.
__global__ void add_block_offsets_kernel(int n, int* __restrict__ out_exclusive,
                                         const int* __restrict__ scanned_block_sums);

// ---- Compaction scatter: order-preserving pack using the scan's addresses. ----

// compact_scatter_kernel — one thread per INPUT point; if flags[i], writes
// xyz[i] to xyz_out[scan_exclusive[i]] (and, if orig_idx_out != nullptr,
// records the source index i there too — the raw material for main.cu's
// GATE order_preservation). scan_exclusive[i] is monotonically
// non-decreasing in i among kept elements, which is EXACTLY why this
// scatter preserves relative order — THEORY.md "The algorithm" proves it.
__global__ void compact_scatter_kernel(int n, const float* __restrict__ xyz_in,
                                       const int* __restrict__ flags,
                                       const int* __restrict__ scan_exclusive,
                                       float* __restrict__ xyz_out,
                                       int* __restrict__ orig_idx_out);

// ---- Unorganized -> organized: nearest-wins scatter via encoded atomicMin. ----

// scatter_to_organized_kernel — one thread per INPUT point: computes this
// point's (ring, azimuth) cell, encodes (range, point_index) per
// pack_range_index(), and atomicMin-races it into cell_encoded[cell]. Many
// threads may target the SAME cell (that IS the collision this project
// measures) — atomicMin guarantees the smallest encoded key (== smallest
// range, index as tiebreaker) survives, deterministically, regardless of
// which thread's atomic executes last.
__global__ void scatter_to_organized_kernel(int n_points, const float* __restrict__ xyz,
                                            unsigned long long* __restrict__ cell_encoded);

// finalize_organized_kernel — one thread per CELL: decode the winning point
// index (if any — cell_encoded[c] == kEmptyCellEncoded means no point
// landed here) and copy that point's xyz into organized_xyz_out[c*3..+2];
// an empty cell gets the NaN sentinel (is_invalid_point's convention),
// keeping the output a valid organized grid a downstream range-image
// consumer (02.12) could read unmodified. winner_index_out[c] records the
// winning ORIGINAL point index, or -1 for an empty cell — main.cu's
// GATE collision_accounting reduces over this array.
__global__ void finalize_organized_kernel(int num_cells,
                                          const unsigned long long* __restrict__ cell_encoded,
                                          const float* __restrict__ xyz_source,
                                          float* __restrict__ organized_xyz_out,
                                          int* __restrict__ winner_index_out);

#endif // __CUDACC__

// ===========================================================================
// Host-callable launch wrappers (definitions in kernels.cu, nvcc-only file,
// but these DECLARATIONS are plain C++ — visible to main.cu, which is also
// compiled by nvcc in this project, same as every project in the repo).
// ===========================================================================

// launch_scan_blelloch — the hand-rolled two-level scan end to end (both
// kernel launches, scratch allocation/free included — see kernels.cu for
// why managing scratch INSIDE this wrapper, rather than threading scratch
// pointers through main.cu, is the right trade for a function called only
// a handful of times per demo run: clarity over micro-optimizing away a
// couple of small cudaMalloc/cudaFree calls). in/out sized [n]; out[i] is
// the EXCLUSIVE prefix sum of in[0..i-1].
void launch_scan_blelloch(int n, const int* d_in, int* d_out_exclusive);

// launch_scan_thrust — the SAME exclusive scan via thrust::exclusive_scan
// (CLAUDE.md rule 6: kernels.cu's definition explains what this computes
// and names CUB, the library Thrust's scan wraps, explicitly).
void launch_scan_thrust(int n, const int* d_in, int* d_out_exclusive);

// launch_scan_cpu_style_on_gpu is intentionally ABSENT — the CPU serial
// scan lives only on the host (reference_cpu.cpp's scan_exclusive_cpu) so
// the three-way comparison (hand-rolled GPU / Thrust GPU / CPU) genuinely
// exercises three different pieces of code, not the same GPU kernel called
// from two call sites.

// ---- The five named compaction pipelines (predicate + scan + scatter). ----
// Every d_out_xyz/d_out_orig_idx buffer must be sized >= n by the caller (a
// safe upper bound: compaction can never grow the point count). Returns the
// number of points KEPT (== the compacted count, and the valid prefix
// length of both output arrays).
int launch_passthrough_compact(int n, const float* d_xyz, float* d_out_xyz, int* d_out_orig_idx);
int launch_box_compact(int n, const float* d_xyz, float* d_out_xyz, int* d_out_orig_idx);
int launch_frustum_compact(int n, const float* d_xyz, float* d_out_xyz, int* d_out_orig_idx);
int launch_fused_compact(int n, const float* d_xyz, float* d_out_xyz, int* d_out_orig_idx);
int launch_valid_compact(int n, const float* d_xyz, float* d_out_xyz, int* d_out_orig_idx);

// launch_scatter_to_organized — the unorganized->organized direction end to
// end: cudaMemset the encoded array to kEmptyCellEncoded, race every input
// point's encoded key in via scatter_to_organized_kernel, then materialize
// xyz + winner indices via finalize_organized_kernel. d_organized_xyz_out
// sized [kOrganizedCells*3]; d_winner_index_out sized [kOrganizedCells].
// Returns (occupied cell count, collision count) — collisions counted as
// (points that targeted an already-occupied-by-a-DIFFERENT-point cell) =
// n_points_that_lost_their_race, the bookkeeping main.cu's
// GATE collision_accounting reconciles against n_points - occupied.
struct OrganizedScatterResult { int occupied; int collisions; };
OrganizedScatterResult launch_scatter_to_organized(int n_points, const float* d_xyz,
                                                   float* d_organized_xyz_out,
                                                   int* d_winner_index_out);

// ===========================================================================
// CPU references (reference_cpu.cpp) — the correctness-oracle twins. All
// pointers below are HOST pointers. See reference_cpu.cpp's file header for
// the independence ruling: the GEOMETRIC FORMULAS (is_passthrough/
// is_in_box/is_in_frustum/nearest_ring_of/azimuth_bin_of/pack_range_index,
// all above) are single-sourced data-layout contracts and ARE shared by
// these functions; the ALGORITHMS below (serial filtering, serial scan, a
// simple running-min instead of an atomicMin race) are independently
// structured from the GPU's scan+scatter / atomicMin-race machinery.
// ===========================================================================

// scan_exclusive_cpu — the dead-simple serial oracle for BOTH GPU scans
// (hand-rolled Blelloch and Thrust). out[i] = sum(in[0..i-1]); out sized
// [n]. This is the file's simplest function on purpose (CLAUDE.md
// "clarity beats speed" — reference_cpu.cpp's rules).
void scan_exclusive_cpu(int n, const int* in, int* out_exclusive);

// The five CPU compaction twins — a plain serial "if predicate, push_back"
// loop each (see reference_cpu.cpp), genuinely independent of the GPU's
// scan-based compaction structure. out_xyz/out_orig_idx sized >= n by the
// caller. Returns the kept count.
int passthrough_compact_cpu(int n, const float* xyz, float* out_xyz, int* out_orig_idx);
int box_compact_cpu(int n, const float* xyz, float* out_xyz, int* out_orig_idx);
int frustum_compact_cpu(int n, const float* xyz, float* out_xyz, int* out_orig_idx);
int fused_compact_cpu(int n, const float* xyz, float* out_xyz, int* out_orig_idx);
int valid_compact_cpu(int n, const float* xyz, float* out_xyz, int* out_orig_idx);

// scatter_to_organized_cpu — the unorganized->organized CPU twin. Uses a
// plain "running minimum per cell" (no encoding, no atomics — a genuinely
// different mechanism from the GPU's atomicMin race) that is ALSO
// order-independent: the smallest range per cell is well-defined regardless
// of the order points are visited in, exactly mirroring why the GPU's
// atomicMin race converges to the same answer regardless of thread
// scheduling — this is WHY the two mechanisms can be compared bit-exact
// with no tolerance (THEORY.md "How we verify correctness" makes this
// argument explicitly: min is commutative/associative, sums are not).
// organized_xyz_out sized [kOrganizedCells*3]; winner_index_out sized
// [kOrganizedCells] (-1 = empty). Returns (occupied, collisions).
struct OrganizedScatterCpuResult { int occupied; int collisions; };
OrganizedScatterCpuResult scatter_to_organized_cpu(int n_points, const float* xyz,
                                                   float* organized_xyz_out,
                                                   int* winner_index_out);

#endif // PROJECT_KERNELS_CUH
