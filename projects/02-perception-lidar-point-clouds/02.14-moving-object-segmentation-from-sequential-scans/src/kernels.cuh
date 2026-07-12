// ===========================================================================
// kernels.cuh — interface & contract for project 02.14
//               Moving-object segmentation from sequential scans (online MOS)
//
// Role in the project
// -------------------
// The CONTRACT shared by main.cu (orchestration + gates + artifacts),
// kernels.cu (the GPU kernels), reference_cpu.cpp (its independent CPU
// twins), and scripts/make_synthetic.py (whose "MUST MATCH" comments mirror
// the beam-model constants below). Everything all four must agree on — the
// beam model, the range-image layout, the pose/reprojection formula, the
// residual-fusion rule, and the CCL edge rule — is defined HERE, once
// (CLAUDE.md §12).
//
// THE PROBLEM, contrasted with this project's offline dual (02.13)
// ---------------------------------------------------------------------------
// 02.13 ("dynamic point removal") answers: given a LONG history of K posed
// scans building a MAP, which map voxels were only ever transiently occupied
// (a car that drove through once, weeks of scans ago)? It works OFFLINE, in
// VOXEL space, and its evidence is a ledger accumulated over the WHOLE
// history before any point is judged.
//
// THIS project answers a different, harder-in-latency question: given ONLY
// the CURRENT scan plus a SHORT window of M=4 immediately preceding scans,
// which points in the CURRENT scan belong to something moving RIGHT NOW? It
// must run ONLINE, once per incoming scan, at the sensor's own rate (10-20
// Hz — README "System context") — there is no long history to lean on, and
// the answer must be ready before the NEXT scan arrives. The representation
// is the RANGE IMAGE (02.12's thesis, cited below), not a voxel grid: 02.12
// showed that a spinning LiDAR's own (ring, azimuth) organization gives
// free, O(1) spatial neighbors; this project additionally exploits the SAME
// organization across TIME — the previous scan, reprojected into the current
// sensor's range image, is directly comparable cell-by-cell to the current
// scan's own range image, no nearest-neighbor search needed in either space
// or time. This is the LiDAR-MOS lineage (Chen et al., IROS 2021, cited in
// full in THEORY.md "Where this sits in the real world"), taught classically
// (fixed residual threshold, no learned network) rather than the
// range-image residual images that paper trains a network on.
//
// THE METHOD, five steps (THEORY.md derives every one in full)
// ---------------------------------------------------------------------------
//   1. ORGANIZE the CURRENT scan into its own range image (02.02/02.12
//      lineage, cited): each current-scan point already carries its NATIVE
//      (ring, az_bin) — a real driver reports these — so this is a plain
//      nearest-wins atomicMin scatter, no reprojection needed.
//   2. REPROJECT each of the M previous scans INTO the current sensor's
//      frame and range image: transform every previous-scan point by the
//      RELATIVE POSE between its own capture instant and the current scan
//      (exactly 02.08's deskew_one_point formula, cited and reused: "project
//      point i's own instant into a reference instant" — here the reference
//      is "now" instead of "sweep end"), recompute which (ring, az_bin) cell
//      the transformed point falls into (a NEAREST-ELEVATION snap — the
//      previous scan's own physical beam directions do not, in general,
//      re-hit the current sensor's exact beam elevations after reprojection
//      — THEORY.md "The problem" derives why), and atomicMin-race the
//      reprojected range into that scan's own range image (02.02/02.12/02.13
//      nearest-wins technique, cited and reused verbatim).
//   3. RESIDUAL, per current-scan cell, per previous scan j:
//          residual_j = range_current - range_prev_reprojected_j
//      THE SIGN IS THE HEART OF THIS PROJECT (THEORY.md "The math" derives
//      both directions in full, diagrammed):
//        * residual_j < 0 (current CLOSER than what scan j saw here): this
//          cell is now occupied CLOSER than free-space evidence from j said
//          it was — an ARRIVAL. A crossing car sweeping into a previously-
//          open direction, or a car approaching (radially closer) along a
//          direction it already occupied, both produce this sign.
//        * residual_j > 0 (current FARTHER than what scan j saw here): the
//          surface scan j saw at this cell has RECEDED or LEFT — a
//          DEPARTURE / revealed-background signature. A car moving away
//          (radially farther) along the same line of sight it still
//          occupies produces this sign; so does a car that has fully left a
//          cell, revealing permanent background behind it.
//      The sign_semantics gate (main.cu) proves this derivation is actually
//      implemented, not accidentally satisfied: the oncoming-car cohort must
//      show negative residuals, the receding-car cohort positive ones.
//   4. MULTI-SCAN EVIDENCE FUSION: fused_evidence = MIN_j |residual_j| over
//      the M included previous scans (kernels.cu documents the choice in
//      full: MIN is the conservative, disocclusion-resistant rule — a cell
//      that shows a large residual against only ONE of the M comparisons,
//      while agreeing closely with the others, is far more likely a one-off
//      disocclusion-boundary artifact (01.21's lesson, cited) than a genuine
//      mover, which perturbs EVERY comparison in the window roughly equally
//      as long as it keeps moving). candidate_moving = fused_evidence >=
//      kResidualThresholdM (THEORY.md "The math" derives the threshold from
//      range noise; "Numerical considerations" reports the as-measured
//      value).
//   5. RANGE-IMAGE CCL CLEANUP (02.12's union-find lineage, cited and
//      reused): build edges between image-ADJACENT candidate_moving cells
//      (same forward-neighbor rule 02.12's beta-criterion edges use, minus
//      the beta test itself), run the SAME generic lock-free union-find
//      sweep 02.04/02.12 use for an entirely different edge list, and drop
//      components smaller than kMinMovingClusterSize as speckle — the final
//      per-cell MOVING/STATIC label, propagated 1:1 back onto the current
//      scan's points (every valid current-scan cell holds exactly one point).
//
// Why this header is CUDA-qualifier-free where possible, HD elsewhere
// ---------------------------------------------------------------------
// Range-image indexing, the pose-composition/reprojection algebra (Vec3/
// Quat and friends), and the encoded-atomicMin key packing are DATA-LAYOUT
// CONTRACTS — pure coordinate bookkeeping and the shared pose model, not the
// algorithm under test — so they are declared HD ("__host__ __device__"
// under nvcc, nothing under cl.exe — the same macro 01.10/02.08/02.13 use)
// and shared, token-for-token, by kernels.cu's kernels and
// reference_cpu.cpp's twins (per the independence ruling in
// reference_cpu.cpp's file header: sharing the quaternion/pose formulas is
// the same "camera/pose model" exception 02.08 claims for its own identical
// helpers). What stays INDEPENDENT (typed twice, the ruling's default) is
// the algorithmic core this project actually teaches: the reprojection
// SCATTER loop, the residual-fusion decision, and the CCL edge-build +
// union-find sweep — each written fresh in kernels.cu (GPU) and
// reference_cpu.cpp (CPU), and main.cu's gates additionally check the FINAL
// labels against ground truth that never touches this shared code
// (THEORY.md "How we verify correctness" spells out the two tiers).
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cmath>      // floorf/sqrtf/atan2f/asinf/roundf — used by the HD helpers below
#include <cstdint>    // uint32_t/uint64_t — exact-width integers for the encoded keys
#include <cstring>    // std::memcpy — float->uint32 bit reinterpretation (range encoding)
#include <vector>     // reference_cpu.cpp's independent edge lists / per-scan buffers
#include <utility>    // std::pair<int,int> — canonicalized (u<v) CCL edges

// ---------------------------------------------------------------------------
// HD — "__host__ __device__" under nvcc, nothing under cl.exe. Same trick as
// 01.10/02.08/02.13 (see those file headers for the full rationale): lets
// kernels.cu's device code and reference_cpu.cpp's host twin call the
// IDENTICAL compiled-twice primitive without either translation unit seeing
// a CUDA keyword it cannot parse.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ===========================================================================
// Beam model + range-image shape — MUST MATCH ../scripts/make_synthetic.py's
// module-level constants of the same name (main.cu asserts the data file's
// '#'-prefixed header against these at load time — the 02.08/02.13-style
// data/code consistency check).
//
// 16-beam elevation table, -15..+15 deg in 2 deg steps: the SAME table
// 02.02/02.08/02.12/02.13 all cite from 01.18's original derivation, reused
// verbatim here rather than invented fresh.
// ===========================================================================
constexpr int kNumBeams = 16;
constexpr float kBeamElevMinDeg = -15.0f;   // ring 0's elevation
constexpr float kBeamElevStepDeg = 2.0f;    // uniform spacing — lets ring lookup be closed-form (below)

// Azimuth resolution: 360 bins = 1 deg/bin — the SAME resolution 02.08 uses
// for its own spinning-LiDAR sweep (cited), a deliberately coarser choice
// than 02.12's 1024 bins (0.35 deg/bin): this project's lesson is about
// TEMPORAL residuals between scans, not fine within-scan clustering, and a
// coarser grid keeps the committed sample and the demo's range-image
// artifacts small (THEORY.md "The problem" also uses this exact resolution
// to derive the thin-pole aliasing story — a deliberately chosen teaching
// consequence, not an oversight).
constexpr int kAzimuthBins = 360;
constexpr int kNumCells = kNumBeams * kAzimuthBins;   // 5,760 organized-grid cells

constexpr float kPi = 3.14159265358979323846f;
constexpr float kAzimuthStepDeg = 360.0f / static_cast<float>(kAzimuthBins);

// organized_cell_index — the ring-major flat-index formula (02.02/02.12
// lineage, cited and reused verbatim).
HD inline int organized_cell_index(int ring, int az_bin) { return ring * kAzimuthBins + az_bin; }

// ===========================================================================
// Scene / sequence constants — MUST match ../scripts/make_synthetic.py's
// Python mirrors (documented match, 02.03/02.13 precedent: main.cu DOES
// separately assert the safety-critical header fields that ARE stored in
// the data file — grid shape, scan count, max range — see load_scans() in
// main.cu).
// ===========================================================================
constexpr float kMaxRangeM = 30.0f;          // beyond this, a beam is a no-return (organized cell stays empty)
constexpr float kRangeNoiseSigmaM = 0.02f;   // synthetic sensor's range-noise floor (2 cm), THEORY.md "The math"

// kNumScansWindow — the FULL sequence this project loads: the CURRENT scan
// plus kMaxWindowM = 4 PREVIOUS scans (the catalog bullet's "K posed scans"
// specialized to K=5, M=4 — README "System context"). Index kCurrentScanIdx
// is "now"; indices 0..3 are previous scans in scan-order (0 = oldest).
constexpr int kNumScansWindow = 5;
constexpr int kMaxWindowM = kNumScansWindow - 1;          // 4
constexpr int kCurrentScanIdx = kNumScansWindow - 1;      // index 4

// kPrevScanIdx[lag-1] — previous-scan index at LAG (1 = most recent previous
// scan, kMaxWindowM = oldest). THE window-size study (README "Expected
// output") uses M in {1,2,4}: "use window size M" means "include only
// kPrevScanIdx[0..M-1]", i.e. the M MOST RECENT previous scans — evidence is
// added from the freshest comparison outward, never skipped in the middle.
constexpr int kPrevScanIdx[kMaxWindowM] = { 3, 2, 1, 0 };   // lag 1,2,3,4

// kDynamicThresholdM — the fused-evidence decision boundary (step 4 above).
// Derived in THEORY.md "The math" from kRangeNoiseSigmaM (two independent
// noisy ranges combine to a residual noise std of sqrt(2)*sigma) plus a
// measured reprojection-quantization contribution; the AS-MEASURED value on
// this project's committed scene is reported by main.cu's
// noise_derivation [info] line and discussed in THEORY.md "Numerical
// considerations" (the same "theoretical bound printed alongside the
// measured operating value" honesty 01.21 practices, cited).
constexpr float kDynamicThresholdM = 0.20f;

// kMinMovingClusterSize — step 5's CCL min-size filter (mirrors 02.04/02.12's
// kMinClusterSize=5 reasoning: a 1-2 cell blob carries no shape evidence and
// is far more likely reprojection-quantization speckle than a real mover).
constexpr int kMinMovingClusterSize = 3;

// Safety cap on the union-find convergence loop main.cu drives (02.04/02.12's
// identical policy: never expected to be hit on a graph this small and
// shallow — a documented, bounded failure signal, not a silent infinite loop).
constexpr int kMaxUfSweeps = 64;

// ===========================================================================
// Ground-truth cohort ids — MUST MATCH ../scripts/make_synthetic.py's
// COHORT_* constants. Used ONLY by main.cu's gates/artifacts (never by the
// reprojection/residual/CCL algorithm itself — the same "no black-box
// cheating" rule 02.13's kernels.cuh states for its own cohort ids).
// ===========================================================================
enum CohortId : int {
    kCohortWall         = 0,   // static: the long back wall (also the disocclusion-band carrier)
    kCohortPole         = 1,   // static: the thin pole (discretization honesty cohort)
    kCohortCrossingCar  = 2,   // dynamic: lateral mover, also occludes/reveals the wall
    kCohortOncomingCar  = 3,   // dynamic: radial approach (the negative-residual showcase)
    kCohortRecedingCar  = 4,   // dynamic: radial departure (the positive-residual showcase)
    kCohortStoppedCar   = 5,   // dynamic: moving for scans 0-3, stationary between scan 3 and current (temporal boundary)
    kCohortNone         = -1   // no return (max range): no object, no cohort, no point
};
inline const char* cohort_name(int id)
{
    switch (id) {
        case kCohortWall:        return "wall";
        case kCohortPole:        return "pole";
        case kCohortCrossingCar: return "crossing_car";
        case kCohortOncomingCar: return "oncoming_car";
        case kCohortRecedingCar: return "receding_car";
        case kCohortStoppedCar:  return "stopped_car";
        default:                 return "none";
    }
}
// Dynamic (ground-truth mover) cohorts — everything else is static.
inline bool cohort_is_dynamic(int id)
{
    return id == kCohortCrossingCar || id == kCohortOncomingCar ||
           id == kCohortRecedingCar || id == kCohortStoppedCar;
}

// ===========================================================================
// Pose algebra — Vec3/Quat/quat_mul/quat_conj/quat_rotate, REPO ORDER
// (w,x,y,z), textbook formulas written out term-by-term (CLAUDE.md §1: no
// black-box quaternion class). Token-for-token the SAME formulas 02.08's
// kernels.cuh derives and cites in full there; reused here because this
// project's reprojection IS 02.08's deskew problem with "current scan" in
// the role of "reference instant" (file header). This project's demo scene
// keeps every scan's orientation at IDENTITY (see make_synthetic.py's
// module docstring — the same documented scope cut 02.13 makes, cited), but
// the algebra below is written fully general so the reprojection is
// correct for ANY rigid sensor trajectory, not just this demo's translation-
// only one (README Exercise: give the sensor a yaw sweep and rerun).
// ===========================================================================
struct Vec3 { float x, y, z; };
struct Quat { float w, x, y, z; };
struct Pose { Vec3 p; Quat q; };   // T_world_sensor: world position + orientation

HD inline Vec3 vec3_add(Vec3 a, Vec3 b) { return Vec3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
HD inline Vec3 vec3_sub(Vec3 a, Vec3 b) { return Vec3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
HD inline Vec3 vec3_scale(Vec3 a, float s) { return Vec3{ a.x * s, a.y * s, a.z * s }; }
HD inline float vec3_norm(Vec3 a) { return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z); }

HD inline Quat quat_conj(Quat q) { return Quat{ q.w, -q.x, -q.y, -q.z }; }   // = inverse, for a UNIT quaternion

HD inline Quat quat_mul(Quat a, Quat b)
{
    return Quat{
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w
    };
}

// quat_rotate — rotate vector v by unit quaternion q via the optimized
// "sandwich product" identity (02.08's kernels.cuh derives this expansion in
// full; reused verbatim): two cross products + a few FMAs instead of two
// full quaternion multiplies for the same result.
HD inline Vec3 quat_rotate(Quat q, Vec3 v)
{
    const Vec3 u{ q.x, q.y, q.z };
    const Vec3 cr1{ u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x };  // u x v
    const Vec3 t = vec3_scale(cr1, 2.0f);
    const Vec3 cr2{ u.y * t.z - u.z * t.y, u.z * t.x - u.x * t.z, u.x * t.y - u.y * t.x };  // u x t
    return vec3_add(vec3_add(v, vec3_scale(t, q.w)), cr2);
}

// reproject_point_to_current — THE core pose-composition formula (THEORY.md
// "The math" derives this from first principles): given a point measured in
// scan j's OWN local sensor frame, express it in the CURRENT scan's local
// sensor frame. Identical derivation to 02.08's deskew_one_point (cited):
//     P_world      = pose_j.p + R(pose_j.q) * p_local_j
//     P_in_current = R(conj(pose_cur.q)) * (P_world - pose_cur.p)
// Parameters: pose_j     — scan j's world pose (T_world_sensor_j).
//             pose_cur   — the CURRENT scan's world pose (T_world_sensor_cur).
//             p_local_j  — the point, in scan j's own local frame (meters).
// Returns: the point re-expressed in the current sensor's local frame.
HD inline Vec3 reproject_point_to_current(Pose pose_j, Pose pose_cur, Vec3 p_local_j)
{
    const Vec3 p_world = vec3_add(pose_j.p, quat_rotate(pose_j.q, p_local_j));
    const Quat q_cur_conj = quat_conj(pose_cur.q);
    return quat_rotate(q_cur_conj, vec3_sub(p_world, pose_cur.p));
}

// ===========================================================================
// Range-image geometry — the (ring, az_bin) <-> direction contract, shared
// by every stage that must agree on it (organizing the current scan,
// deriving a stored point's local xyz, and re-binning a reprojected point).
// ===========================================================================

// beam_dir_local — unit beam direction in the SENSOR's own local frame for
// elevation/azimuth in DEGREES (spherical convention: azimuth measured CCW
// from local +x in the local xy-plane, elevation up from that plane — the
// same x-forward/y-left/z-up body convention 02.13's beam_direction()
// uses, cited).
HD inline Vec3 beam_dir_local(float elev_deg, float az_deg)
{
    const float el = elev_deg * (kPi / 180.0f);
    const float az = az_deg * (kPi / 180.0f);
    return Vec3{ cosf(el) * cosf(az), cosf(el) * sinf(az), sinf(el) };
}

// local_point_from_ring_az — a stored point's local xyz, derived from its
// NATIVE (ring, az_bin, range) rather than stored a second time (the same
// "derive once, from the single-sourced formula" discipline 02.13's
// kernels.cuh states for its own beam_point()).
HD inline Vec3 local_point_from_ring_az(int ring, int az_bin, float range_m)
{
    const float elev_deg = kBeamElevMinDeg + static_cast<float>(ring) * kBeamElevStepDeg;
    const float az_deg = static_cast<float>(az_bin) * kAzimuthStepDeg;
    return vec3_scale(beam_dir_local(elev_deg, az_deg), range_m);
}

// nearest_ring_for_elev_deg — snap a CONTINUOUS elevation angle (degrees) to
// the nearest of the 16 discrete beam rings. Closed-form because the table
// is UNIFORMLY spaced (kBeamElevStepDeg apart) — a documented assumption
// that would break for an irregular beam table (THEORY.md "The problem"
// derives why a reprojected point, in general, does NOT land exactly on any
// of the sensor's real beam elevations, so this snap is itself a source of
// quantization error, not a bug to "fix").
HD inline int nearest_ring_for_elev_deg(float elev_deg)
{
    int ring = static_cast<int>(lroundf((elev_deg - kBeamElevMinDeg) / kBeamElevStepDeg));
    if (ring < 0) ring = 0;
    if (ring >= kNumBeams) ring = kNumBeams - 1;
    return ring;
}

// az_bin_for_az_deg — snap a continuous azimuth (any real value, will be
// wrapped) to the nearest of the kAzimuthBins columns, WRAPPING AROUND at
// 360 deg (the sensor spins a full circle — 02.12's identical wrap-around
// discipline for its own azimuth stencil, cited).
HD inline int az_bin_for_az_deg(float az_deg)
{
    float wrapped = fmodf(az_deg, 360.0f);
    if (wrapped < 0.0f) wrapped += 360.0f;
    int bin = static_cast<int>(lroundf(wrapped / kAzimuthStepDeg)) % kAzimuthBins;
    if (bin < 0) bin += kAzimuthBins;   // defensive: lroundf negative-zero edge case
    return bin;
}

// cell_for_local_point — the full "where does this local-frame point land in
// MY range image" formula: recover elevation/azimuth from xyz, snap to
// (ring, az_bin), return the flat cell index plus the point's own range
// (the caller needs both — range for the atomicMin key, the cell for where
// to race it in). Used by BOTH the current-scan organizer (ring/az already
// native, but this function re-derives them identically so a single formula
// governs every scatter in the project — no "two slightly different binning
// rules" bug class) and the reprojection scatter (ring/az computed fresh
// from the transformed point, THEORY.md "The problem").
HD inline void cell_for_local_point(Vec3 p_local, int& out_cell, float& out_range_m)
{
    const float range_m = vec3_norm(p_local);
    out_range_m = range_m;
    if (range_m < 1e-6f) { out_cell = organized_cell_index(0, 0); return; }   // degenerate guard: sensor origin itself
    // asinf is undefined outside [-1,1]; z/range is mathematically inside
    // that range by construction (z is one component of a vector of norm
    // range), but float rounding in the division can push it a hair past
    // +-1.0 at near-vertical elevations — the same defensive clamp
    // 02.12's beta-criterion acos-adjacent code and 08.01's rotation-angle
    // clamp both apply before an inverse trig call (cited pattern).
    float sin_elev = p_local.z / range_m;
    sin_elev = sin_elev < -1.0f ? -1.0f : (sin_elev > 1.0f ? 1.0f : sin_elev);
    const float elev_deg = asinf(sin_elev) * (180.0f / kPi);
    const float az_deg = atan2f(p_local.y, p_local.x) * (180.0f / kPi);
    const int ring = nearest_ring_for_elev_deg(elev_deg);
    const int az_bin = az_bin_for_az_deg(az_deg);
    out_cell = organized_cell_index(ring, az_bin);
}

// ===========================================================================
// Encoded (range, point-index) atomicMin key — IDENTICAL technique to
// 02.02/02.12/02.13's kernels.cuh (cited, reused verbatim): packing the
// winning point index alongside the range makes "nearest wins" a
// deterministic race even on an exact range tie.
// ===========================================================================
HD inline uint32_t float_range_to_sortable_u32(float r)
{
    uint32_t bits;
    std::memcpy(&bits, &r, sizeof(bits));   // bit-for-bit reinterpretation — valid for r >= 0
    return bits;
}
HD inline uint64_t pack_range_index(float range_m, uint32_t point_idx)
{
    return (static_cast<uint64_t>(float_range_to_sortable_u32(range_m)) << 32) | point_idx;
}
HD inline uint32_t unpack_point_index(uint64_t encoded) { return static_cast<uint32_t>(encoded & 0xFFFFFFFFu); }
// unpack_range_m — the inverse of float_range_to_sortable_u32: recover the
// RANGE VALUE (not just the winning index) directly from an encoded key's
// upper 32 bits. finalize_prev_kernel uses this: the reprojection scatter
// (kernels.cu) encodes the REPROJECTED range (not any input point's stored
// range) into the key, so that value only ever exists inside the key itself
// — decoding it here avoids carrying a second, redundant "reprojected range
// per point" array just to look it up post-hoc.
HD inline float unpack_range_m(uint64_t encoded)
{
    const uint32_t bits = static_cast<uint32_t>(encoded >> 32);
    float r;
    std::memcpy(&r, &bits, sizeof(r));
    return r;
}
constexpr unsigned long long kEmptyCellEncoded = ~0ull;   // larger than any valid key (02.02's proof, cited)

// ===========================================================================
// GPU kernel declarations — nvcc-only (cl.exe, compiling reference_cpu.cpp,
// has never heard of __global__ and must never see these).
// ===========================================================================
#ifdef __CUDACC__

// scatter_current_kernel — step 1: one thread per CURRENT-scan input point.
// This point already carries its NATIVE (ring, az_bin) (a real driver
// reports these); the kernel just encodes (range, point_idx) and races it
// into cell_encoded[cell] via atomicMin (02.02/02.12 lineage). No
// reprojection: the current scan is already in its own frame.
__global__ void scatter_current_kernel(int n_points,
                                       const int* __restrict__ ring, const int* __restrict__ az_bin,
                                       const float* __restrict__ range_m,
                                       unsigned long long* __restrict__ cell_encoded);

// finalize_current_kernel — one thread per CELL: decode the winning point
// index (kEmptyCellEncoded => no return -> range_img=0, the "no return"
// sentinel this project uses throughout), copy that point's range plus its
// GROUND-TRUTH cohort/truth/disocclusion-band fields (used only by main.cu's
// gates/artifacts, never by later algorithm stages — kernels.cuh file
// header "Ground truth").
__global__ void finalize_current_kernel(int num_cells,
                                        const unsigned long long* __restrict__ cell_encoded,
                                        const float* __restrict__ prange, const int* __restrict__ pcohort,
                                        const int* __restrict__ ptruth, const int* __restrict__ pdisocc,
                                        float* __restrict__ range_img,
                                        int* __restrict__ cohort_img, int* __restrict__ truth_img,
                                        int* __restrict__ disocc_img);

// reproject_scatter_kernel — step 2: one thread per point in PREVIOUS scan
// j. Reprojects the point into the current sensor's frame
// (reproject_point_to_current), re-derives its (ring,az_bin) cell in the
// CURRENT range image (cell_for_local_point — NOT the point's own native
// ring, which belongs to scan j's geometry, not the current one — THEORY.md
// "The problem"), and races (range,idx) into that scan's OWN prev range
// image via the same encoded atomicMin.
__global__ void reproject_scatter_kernel(int n_points,
                                         const int* __restrict__ ring, const int* __restrict__ az_bin,
                                         const float* __restrict__ range_m,
                                         Pose pose_j, Pose pose_cur,
                                         unsigned long long* __restrict__ cell_encoded);

// finalize_prev_kernel — one thread per CELL: decode the winner (or 0 for
// no return) into a plain range image for previous scan j. No ground-truth
// payload here — a previous scan's OWN points' truth is irrelevant; only
// the CURRENT scan's points are ever labeled.
__global__ void finalize_prev_kernel(int num_cells, const unsigned long long* __restrict__ cell_encoded,
                                     float* __restrict__ range_img_prev);

// residual_fuse_kernel — steps 3-4: one thread per CURRENT cell. Reads the
// current range image and up to window_m previous range images (passed as
// an array of device pointers, ordered nearest-lag-first per kPrevScanIdx),
// computes the signed residual against each VALID (range>0) previous cell,
// fuses via MIN(|residual|) (file header step 4), and writes:
//   fused_evidence_out[cell] : MIN |residual| over included, valid j; -1.0f
//                              if NO included previous scan had a valid
//                              return at this cell (insufficient evidence).
//   sign_out[cell]           : sign of the NEAREST-LAG valid comparison's
//                              signed residual (+1/-1/0), the representative
//                              sign the sign_semantics gate reads.
//   candidate_out[cell]      : 1 iff current has a return AND fused_evidence
//                              >= threshold_m; else 0.
// window_m in [1, kMaxWindowM]; prev_range_imgs[0..window_m) are the device
// pointers for lag 1..window_m (kPrevScanIdx order).
__global__ void residual_fuse_kernel(const float* __restrict__ range_img_cur,
                                     const float* const* __restrict__ prev_range_imgs, int window_m,
                                     float threshold_m,
                                     float* __restrict__ fused_evidence_out,
                                     int* __restrict__ sign_out, int* __restrict__ candidate_out);

// build_moving_edges_kernel — step 5a: one thread per CELL, mirrors 02.12's
// depth_edges_kernel FORWARD-neighbor pattern exactly (ring+1 same column;
// same ring, column+1 WITH WRAP-AROUND) but the "connect" predicate is
// "both cells are candidate_moving", not a beta test. edge_u/edge_v sized
// [num_cells*2] (2 forward neighbors per cell, an exact capacity — 02.12's
// identical sizing argument).
__global__ void build_moving_edges_kernel(int num_cells, const int* __restrict__ candidate,
                                          int* __restrict__ edge_u, int* __restrict__ edge_v,
                                          int* __restrict__ edge_count);

// ---- generic lock-free GPU union-find (02.04/02.12 Method A, cited) ------
__global__ void uf_init_kernel(int n, int* __restrict__ parent);
__global__ void uf_union_sweep_kernel(int num_edges, const int* __restrict__ edge_u,
                                      const int* __restrict__ edge_v,
                                      int* __restrict__ parent, int* __restrict__ changed);
__global__ void uf_finalize_kernel(int n, int* __restrict__ parent);

#endif // __CUDACC__

// ===========================================================================
// Host-callable launch wrappers (definitions in kernels.cu). Each owns its
// grid/block math + the mandatory post-launch error check (CLAUDE.md §6.1
// rule 7).
// ===========================================================================
void launch_scatter_current(int n_points, const int* d_ring, const int* d_az_bin, const float* d_range_m,
                            unsigned long long* d_cell_encoded);
void launch_finalize_current(int num_cells, const unsigned long long* d_cell_encoded,
                             const float* d_prange, const int* d_pcohort, const int* d_ptruth, const int* d_pdisocc,
                             float* d_range_img, int* d_cohort_img, int* d_truth_img, int* d_disocc_img);

void launch_reproject_scatter(int n_points, const int* d_ring, const int* d_az_bin, const float* d_range_m,
                              Pose pose_j, Pose pose_cur, unsigned long long* d_cell_encoded);
void launch_finalize_prev(int num_cells, const unsigned long long* d_cell_encoded, float* d_range_img_prev);

void launch_residual_fuse(const float* d_range_img_cur, const float* const* d_prev_range_imgs, int window_m,
                          float threshold_m, float* d_fused_evidence, int* d_sign, int* d_candidate);

// launch_build_moving_edges — wrapper: resets the atomic edge counter,
// launches, reads back the count (<= num_cells*2 by construction, 02.12's
// identical no-overflow-possible argument).
int launch_build_moving_edges(int num_cells, const int* d_candidate, int* d_edge_u, int* d_edge_v);

void launch_uf_init(int n, int* d_parent);
bool launch_uf_union_sweep(int num_edges, const int* d_edge_u, const int* d_edge_v,
                           int* d_parent, int* d_changed);
void launch_uf_finalize(int n, int* d_parent);

// ===========================================================================
// CPU references (reference_cpu.cpp) — the correctness-oracle twins. All
// pointers below are HOST pointers. See reference_cpu.cpp's file header for
// the independence ruling each of these follows.
// ===========================================================================

// scatter_current_cpu — INDEPENDENT twin of scatter+finalize_current: a
// plain per-cell "running minimum range" scan over all input points (no
// encoding, no atomics — 02.02/02.12's identical CPU-twin technique, cited)
// — order-independent, so directly comparable to the GPU's atomicMin race.
void scatter_current_cpu(int n_points, const int* ring, const int* az_bin, const float* range_m,
                         const int* cohort, const int* truth, const int* disocc,
                         float* range_img, int* cohort_img, int* truth_img, int* disocc_img);

// reproject_scatter_cpu — INDEPENDENT twin of reproject_scatter+finalize_prev
// for ONE previous scan: same per-cell running-minimum-range scan, using the
// SHARED reproject_point_to_current/cell_for_local_point formulas (the
// "camera/pose model" sharing exception — file header).
void reproject_scatter_cpu(int n_points, const int* ring, const int* az_bin, const float* range_m,
                           Pose pose_j, Pose pose_cur, float* range_img_prev);

// residual_fuse_cpu — INDEPENDENT twin of residual_fuse_kernel: identical
// MIN-fusion rule, typed fresh as a plain sequential loop over cells and,
// within each cell, over the included previous scans.
void residual_fuse_cpu(const float* range_img_cur, const std::vector<const float*>& prev_range_imgs,
                       int window_m, float threshold_m,
                       float* fused_evidence_out, int* sign_out, int* candidate_out);

// build_moving_edges_cpu — INDEPENDENT twin of step 5a: a plain double loop
// (cells x 2 forward neighbors), returned as an ascending-sorted canonical
// edge vector for main.cu's set-equality VERIFY (02.12's identical pattern).
std::vector<std::pair<int,int>> build_moving_edges_cpu(int num_cells, const int* candidate);

// serial_union_find_cpu — INDEPENDENT sequential union-find (02.04/02.12's
// identical twin, GENERIC over any edge list). Union-by-min's final
// partition is order-independent, so this is expected to match the GPU's
// finalized parent[] bit-exact.
void serial_union_find_cpu(int n, const std::vector<std::pair<int,int>>& edges, std::vector<int>& parent_out);

#endif // PROJECT_KERNELS_CUH
