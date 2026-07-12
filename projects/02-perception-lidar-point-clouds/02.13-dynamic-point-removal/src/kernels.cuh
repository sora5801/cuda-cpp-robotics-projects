// ===========================================================================
// kernels.cuh — interface for project 02.13
//               Dynamic point removal (raycast free-space carving)
//
// Role in the project
// -------------------
// The CONTRACT shared by main.cu (orchestration + gates), kernels.cu (the
// three GPU kernels), reference_cpu.cpp (its independent CPU twins), and
// scripts/make_synthetic.py (whose "MUST MATCH" comments mirror the beam
// model constants below). Everything all four must agree on — the beam
// model, the voxel grid layout, the ledger layout, and the classification
// rule — is defined HERE, once (CLAUDE.md §12).
//
// THE PRINCIPLE, in five lines (THEORY.md derives it properly)
// ---------------------------------------------------------------------------
//   1. A LiDAR beam that returns a HIT tells you one thing happened at ONE
//      point: something reflected the beam there.
//   2. The SAME beam ALSO tells you something the hit alone does not: every
//      voxel the beam passed through on the way there was, at that instant,
//      EMPTY (nothing blocked the beam earlier, or it would have hit that
//      instead). This is free-space EVIDENCE, and it is normally thrown
//      away the moment a mapping pipeline keeps only the endpoint.
//   3. Accumulate that evidence, per voxel, over many scans: a HIT ledger
//      (this voxel reflected a beam) and a PASS ledger (a beam went straight
//      through this voxel). A permanent wall accumulates hits and almost no
//      passes. A voxel a parked car once occupied accumulates one scan's
//      worth of hits, then passes forever after, once the car leaves.
//   4. score = passes / (hits + passes) is a per-voxel "how often did later
//      evidence contradict this voxel's occupancy" ratio. High score = the
//      voxel looks like something that MOVED.
//   5. Points landing in a high-score voxel are DYNAMIC — ghosts of
//      something that is no longer there. Remove them; what remains is a
//      map worth trusting (README "System context": this is what makes a
//      map REUSABLE).
// Step 3's traversal is the classic Amanatides & Woo (1987) voxel-marching
// DDA, the same voxel-grid-traversal idea 05.01 (TSDF fusion) applies at
// VOXEL granularity via per-voxel projective lookup; this project applies it
// at BEAM granularity via per-beam ray marching — cite 05.01 as this
// domain's "fuse evidence into a dense voxel grid" precedent, and note the
// honest difference: 05.01 never marches a ray through the grid (it
// projects each of 2.1M voxels INTO the depth image instead); this project
// introduces the actual DDA march because free-space carving's evidence
// only exists ALONG the beam, not at a single projected sample.
//
// THE LEDGER — three dense per-voxel counters (uint32, "why not float": an
// occupancy COUNT is intrinsically an integer, and integers sum losslessly
// and order-independently under concurrent atomicAdd — the SAME "order-
// independent integer accounting" property 02.02's counting kernels rely
// on, cited here for the identical reason):
//   hits[v]            : beams whose ENDPOINT landed in voxel v (excludes
//                        every voxel the beam merely passed through).
//   pass_from_hit[v]   : beams that HIT something ELSEWHERE but passed
//                        THROUGH v on the way (this voxel was empty when
//                        that beam fired, even though the beam eventually
//                        found a surface further on).
//   pass_from_maxrange[v]: beams that never hit anything at all (a MAX-RANGE
//                        return) but passed through v — free-space evidence
//                        with no accompanying hit anywhere on that beam.
// Kept SEPARATE (rather than one combined "passes[v]") solely so the
// max_range_carving verification gate (README "Expected output") can prove,
// per voxel, that carving evidence came from returns with no hit at all —
// the classification score itself uses their SUM (see kDynamicThreshold
// below); THEORY.md "The math" derives the ratio and its threshold.
//
// COUNTING SUBTLETIES (both load-bearing, both explained again at the DDA
// implementation in kernels.cu/reference_cpu.cpp):
//   * ENDPOINT EXCLUSION — the voxel a beam HITS is never also counted as a
//     PASS for that same beam: one beam contributes exactly one ledger
//     increment total (either one hits[v], or one pass_*[v] for every voxel
//     it crosses, INCLUDING its own terminal voxel when there is no hit).
//   * MAX-RANGE BEAMS CARVE BUT NEVER HIT — a beam with no return still
//     marches the full kMaxRangeM and marks every voxel it crosses,
//     including its terminal voxel, as PASS. It contributes to hits[] for
//     NO voxel, ever.
//   * SELF-CARVE GUARD — the voxel CONTAINING THE SENSOR is never marked
//     PASS for any beam (every beam's march starts counting only from the
//     FIRST voxel boundary it crosses, THEORY.md "Numerical considerations"
//     explains why marking the sensor's own voxel would be both trivial and
//     wrong: at K=10 scans it would accumulate ~28,800 passes trivially, in
//     a voxel that (being where the robot physically sits) was never
//     actually observed as free by any OTHER means).
//
// VOXEL GRID — a dense array, x fastest (the 05.01 layout convention):
//     linear index v = (iz * kGridNY + iy) * kGridNX + ix
//     voxel i's world-space extent: [origin + i*L, origin + (i+1)*L) per axis
// kGridNX * kGridNY * kGridNZ = 160*160*30 = 768,000 voxels; three uint32
// ledger arrays cost 768,000*4*3 ~= 9.2 MiB of device memory — trivial.
//
// BEAM MODEL — MUST MATCH ../scripts/make_synthetic.py's module-level
// constants of the same name (main.cu asserts the data file's header
// against these at load time, the 02.08-style data/code consistency check).
// 16-beam elevation table reused verbatim from 02.08 (itself citing 01.18).
//
// BEAM RECORD LAYOUT — parallel arrays (Structure-of-Arrays), one entry per
// beam, N = kNumScans * kNumBeams * kAzimuthSteps = 28,800:
//     scan_id[i] : which scan this beam belongs to (0..kNumScans-1)
//     dir[i*3+0..2] : unit direction, WORLD frame (sensor orientation is
//                     identity throughout this project's scenario — see
//                     make_synthetic.py's file header — so body-frame and
//                     world-frame directions coincide; a yaw-varying
//                     platform is a straightforward extension, README
//                     Exercise)
//     is_hit[i]  : 1 if this beam returned a hit, 0 if max-range (a uint8,
//                  not bool, for a portable, explicitly-sized ABI — the
//                  same "explicit width over convenient default" reasoning
//                  02.01 gives for uint64_t keys)
//     range[i]   : meters. If is_hit, the (noisy) measured range; if not,
//                  EXACTLY kMaxRangeM (make_synthetic.py's file header) —
//                  so the carving kernel always marches the SAME distance
//                  field regardless of hit/miss, and only the is_hit flag
//                  decides whether the final voxel becomes a HIT or a PASS.
// dir is stored INTERLEAVED (dir[i*3+0..2], not three separate arrays) to
// match this repo's PointCloud convention (02.01/02.06/SYSTEM_DESIGN.md
// §3.6) — an honest coalescing trade documented at the kernel launch site in
// kernels.cu: a Structure-of-Arrays split (dx[],dy[],dz[]) would coalesce
// each of the three reads independently; interleaved xyz coalesces less
// perfectly per read but keeps this project's beam records the same shape
// every other point-cloud project in the repo uses (README Exercise: try
// the split layout and re-profile).
//
// The point a HIT beam produced (only used post-hoc, for classification and
// for ground-truth cohort bookkeeping) is DERIVED, never stored twice:
//     P = origin(scan_id) + dir * range
// computed identically (same fmaf order) everywhere it is needed — the
// single-sourced formula in world_to_voxel()'s callers below.
//
// GROUND TRUTH — cohort[i]/truth_dynamic[i], loaded alongside the beam but
// used ONLY by main.cu's gates and artifacts, NEVER by the GPU/CPU carving
// or classification algorithms themselves (a real pipeline has no such
// oracle — CLAUDE.md §8 "never fabricate results" cuts both ways: the
// algorithm must not cheat by reading its own answer key).
//
// Why this header is CUDA-qualifier-free where possible, HD elsewhere
// ---------------------------------------------------------------------
// The voxel-INDEXING arithmetic below (world_to_voxel, voxel_index,
// voxel_in_bounds) is a DATA-LAYOUT CONTRACT — pure coordinate bookkeeping,
// not an algorithm — so it is declared HD ("__host__ __device__" under
// nvcc, nothing under cl.exe; the same macro 01.10/09.01/02.08 use) and
// SHARED, token-for-token, by kernels.cu's kernels and reference_cpu.cpp's
// twins: per the independence ruling in reference_cpu.cpp's file header,
// duplicating "floor divide, then pack three ints into a linear index" by
// hand would be pure transcription, exactly the exception 02.08/01.10 claim
// for their own shared math. What stays INDEPENDENT (typed twice, the
// ruling's default) is the actual DDA MARCH — the algorithm this project
// teaches — in kernels.cu's __device__ carve_one_beam_gpu and
// reference_cpu.cpp's carve_one_beam_cpu. Because sharing indexing math
// alone cannot catch an indexing bug, this project's independent gates
// (README "Expected output": ghost_removal/late_leaver/static_preservation)
// compare the FINAL classification against ground truth that never touches
// this header's functions at all — the required "gate that does not route
// through the shared code" (reference_cpu.cpp's file header explains why).
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cmath>      // floorf/fabsf — used by the HD helpers below (identical on host and device)
#include <cstdint>    // uint32_t etc. — exact-width integers for the ledger

// ---------------------------------------------------------------------------
// HD — "__host__ __device__" under nvcc, nothing under cl.exe. Same trick as
// 01.10/09.01/02.08 (see kernels.cuh's file header there for the full
// rationale): lets kernels.cu's device code and reference_cpu.cpp's host
// twin call the IDENTICAL compiled-twice indexing primitive without either
// translation unit seeing a CUDA keyword it cannot parse.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ===========================================================================
// Beam model constants — MUST MATCH ../scripts/make_synthetic.py's
// module-level constants of the same name (main.cu's loader asserts the
// data file's '#'-prefixed header lines against these).
// ===========================================================================
constexpr int kNumBeams = 16;              // elevation rings per sweep
constexpr float kBeamElevDeg[kNumBeams] = {
    -15.0f, -13.0f, -11.0f, -9.0f, -7.0f, -5.0f, -3.0f, -1.0f,
      1.0f,   3.0f,   5.0f,  7.0f,  9.0f, 11.0f, 13.0f, 15.0f
};                                          // reused verbatim from 02.08 (itself citing 01.18)

constexpr int   kAzimuthSteps = 180;        // 2 deg/step (README "Limitations": a documented
                                            // scope cut from 02.08's 1-deg convention — this
                                            // project's measurement is TEMPORAL evidence
                                            // accumulation, not angular resolution; 90 steps/
                                            // 4 deg was the first value tried and measured to
                                            // starve most static voxels of repeat hits across
                                            // the 10 scans — scripts/make_synthetic.py's
                                            // AZIMUTH_STEPS comment tells the measured story)
constexpr int   kNumScans     = 10;         // K in the catalog bullet's "K posed scans"
constexpr float kMaxRangeM    = 20.0f;      // sensor max range (m)

constexpr int kBeamsPerScan = kNumBeams * kAzimuthSteps;         // 2,880
constexpr int kTotalBeams   = kBeamsPerScan * kNumScans;         // 28,800

// The "late leaver" verify gate carves the first 5 scans only, then all 10 —
// see README "Expected output" and main.cu's late_leaver gate. Because
// scripts/make_synthetic.py writes beams SCAN-MAJOR (every one of scan k's
// beams before any of scan k+1's), "the first 5 scans' beams" is simply the
// first kBeamsPerScan*5 entries of the loaded array — no filtering needed.
constexpr int kLateLeaverScans = 5;
constexpr int kLateLeaverBeamCount = kBeamsPerScan * kLateLeaverScans;  // 7,200

// ===========================================================================
// Voxel grid — the local map this project carves (README "System context":
// bounded, like any real online mapping grid — a beam that exits these
// bounds before reaching its target simply stops carving, the honest
// behavior of a LOCAL map, not a bug).
// ===========================================================================
constexpr float kVoxelSizeM  = 0.20f;       // voxel edge (m) — same order of magnitude as 02.01's kVoxelLeafM
// Grid origin deliberately offset by 0.07 m from the "round" -16/-16/-2 that
// would otherwise look natural. WHY: scripts/make_synthetic.py's scene uses
// human-friendly round-number geometry (a wall face at y=7.8 m, a pole at
// x=-2 m, ...) — and 7.8, -2.0 etc. are all EXACT multiples of kVoxelSizeM
// relative to a origin of -16.0. An object surface sitting EXACTLY on a
// voxel boundary is the worst case for discretization: the sensor's 2 cm
// range noise then scatters roughly HALF of that surface's hit points
// across the boundary into the voxel in FRONT of the surface — a voxel that
// (being the final-approach voxel for every beam that ever reaches the
// wall) is a pass-count hotspot, not a hits==0 curiosity. The result, found
// empirically while building this project's demo: the plain WALL cohort
// alone showed a ~30% false-dynamic rate, dominated by boundary-straddled
// points nowhere near any INTENTIONAL edge (THEORY.md "The problem" derives
// this failure mode in full; it is the general reason production voxel
// algorithms sample at voxel CENTERS or jitter their grid, the same lesson
// 05.01's "+0.5 centers the sample" TSDF comment teaches from the opposite
// direction). Offsetting the ORIGIN by a sub-voxel amount that shares no
// common factor with the scene's round-number coordinates fixes every
// surface in the scene at once, without touching the didactic geometry
// itself — verified by rerunning static_preservation before/after (README
// "Expected output" states both numbers).
constexpr float kGridOriginX = -15.93f;     // world position of the grid's minimum corner (m)
constexpr float kGridOriginY = -15.93f;
constexpr float kGridOriginZ = -1.93f;
constexpr int   kGridNX = 160;              // 32 m / 0.20 m
constexpr int   kGridNY = 160;              // 32 m / 0.20 m
constexpr int   kGridNZ = 30;               //  6 m / 0.20 m  (covers every modeled object, sensor height 1.2 m)
constexpr int   kNumVoxels = kGridNX * kGridNY * kGridNZ;   // 768,000

// Safety cap on DDA march steps (both the bulk carve and the trace buffer
// below share this bound): the grid's own diagonal is
// sqrt(32^2+32^2+6^2)/0.20 ~= 227 voxel steps in the worst case a beam could
// possibly need before EITHER reaching its target OR exiting the grid — 300
// is comfortable headroom against an infinite loop from a future bug, not an
// expected path length (THEORY.md "Numerical considerations").
constexpr int kMaxDDASteps = 300;

// score = (pass_from_hit + pass_from_maxrange) / (hits + pass_from_hit +
// pass_from_maxrange); a point in a voxel with score >= this threshold is
// classified DYNAMIC. 0.6 = "clearly more evidence this spot was empty than
// that it was occupied", a small majority-plus margin rather than a bare
// majority (0.5) — measured (README "Expected output") to matter for
// exactly the voxels this project's ratio statistics are noisiest on: a
// voxel with a SINGLE real hit and a single incidental nearby pass scores
// EXACTLY 0.5 (a coin flip this project's low-repeat-count static cohorts
// hit often enough to move the static_preservation gate measurably); 0.6
// requires slightly more than a bare majority of passing evidence before
// removing a point, trading a small amount of ghost-removal aggressiveness
// (still comfortably over its floor at this setting) for meaningfully fewer
// false removals of well-observed static structure (THEORY.md "The math"
// derives the statistics and the threshold's sensitivity in full).
constexpr float kDynamicThreshold = 0.6f;

// ===========================================================================
// Ground-truth cohort ids — MUST MATCH ../scripts/make_synthetic.py's
// COHORT_* constants. Used ONLY by main.cu's gates/artifacts (file header
// "GROUND TRUTH" note) — never read by the carving/classification kernels.
// ===========================================================================
enum CohortId : int {
    kCohortWall       = 0,   // static: the long wall
    kCohortPole       = 1,   // static: the thin pole (discretization honesty cohort)
    kCohortWallEdge   = 2,   // static: the wall's free end (discretization honesty cohort)
    kCohortCar        = 3,   // dynamic: the ghost-trail maker
    kCohortPedestrian = 4,   // dynamic: the "late leaver" (temporarily static)
    kCohortGhost      = 5,   // dynamic: the isolated max-range-only carving proof
    kCohortNone       = -1   // max-range beam: no object, no cohort, no point
};
inline const char* cohort_name(int id)
{
    switch (id) {
        case kCohortWall:       return "wall";
        case kCohortPole:       return "pole";
        case kCohortWallEdge:   return "wall_edge";
        case kCohortCar:        return "car";
        case kCohortPedestrian: return "pedestrian";
        case kCohortGhost:      return "ghost";
        default:                return "none";
    }
}

// ===========================================================================
// Shared HD voxel-indexing primitives (the data-layout contract; see file
// header "Why this header is CUDA-qualifier-free where possible, HD
// elsewhere" for the independence-ruling justification).
// ===========================================================================

// voxel_coord_axis — floor((p - origin) / voxel_size) along one axis. Same
// "must floor, not truncate" pitfall 02.01's voxel_coord documents in full
// (a negative p near zero truncates the WRONG way with a plain int cast);
// floorf gets every sign right, identically on host and device.
HD inline int voxel_coord_axis(float p, float origin, float voxel_size)
{
    return static_cast<int>(floorf((p - origin) / voxel_size));
}

// world_to_voxel — the point (px,py,pz), meters, world frame -> integer
// voxel coordinates (ix,iy,iz). The SAME formula the DDA march's start/
// target voxels and the classify kernel's per-point lookup all use — single-
// sourced so a hit beam's recorded point and the voxel its own carve
// incremented can never silently disagree (the free_space_consistency gate,
// README "Expected output", is the programmatic check that they never do).
HD inline void world_to_voxel(float px, float py, float pz, int& ix, int& iy, int& iz)
{
    ix = voxel_coord_axis(px, kGridOriginX, kVoxelSizeM);
    iy = voxel_coord_axis(py, kGridOriginY, kVoxelSizeM);
    iz = voxel_coord_axis(pz, kGridOriginZ, kVoxelSizeM);
}

// voxel_in_bounds — is (ix,iy,iz) inside the local map? The DDA march's exit
// condition for beams that leave the mapped region before reaching their
// target (file header "Voxel grid": the honest local-map behavior).
HD inline bool voxel_in_bounds(int ix, int iy, int iz)
{
    return ix >= 0 && ix < kGridNX && iy >= 0 && iy < kGridNY && iz >= 0 && iz < kGridNZ;
}

// voxel_index — pack (ix,iy,iz) into the flat ledger index, x fastest (the
// 05.01 layout convention: adjacent-x threads touch adjacent addresses).
// Caller must have already checked voxel_in_bounds — this function does not
// re-check (it is called on every DDA step, and the bounds check already
// happened in the caller's loop; CLAUDE.md's "no black boxes" extends to
// "no redundant hidden work" in a function this hot).
HD inline int voxel_index(int ix, int iy, int iz)
{
    return (iz * kGridNY + iy) * kGridNX + ix;
}

// ===========================================================================
// Ledger — the three dense per-voxel counters (file header). A small POD of
// three device (or host, for the CPU twins) pointers, passed BY VALUE — the
// same "small pointer bundle, cheaper than an indirection" reasoning 02.01's
// HashTableGPU comment gives.
// ===========================================================================
struct Ledger {
    unsigned int* hits;              // [kNumVoxels] beams whose ENDPOINT landed here
    unsigned int* pass_from_hit;     // [kNumVoxels] beams that hit elsewhere but crossed here
    unsigned int* pass_from_maxrange;// [kNumVoxels] beams with NO hit anywhere that crossed here
};

// ===========================================================================
// GPU kernel declarations — nvcc-only (cl.exe, compiling reference_cpu.cpp,
// has never heard of __global__ and must never see these).
// ===========================================================================
#ifdef __CUDACC__

// ledger_clear_kernel — one thread per VOXEL: reset all three counters to 0.
// Full launch-configuration reasoning lives with the definition in kernels.cu.
__global__ void ledger_clear_kernel(Ledger ledger);

// carve_kernel — one thread per BEAM: march its DDA path and atomically
// update the ledger (the bulk carving pass; no per-beam sequence output).
//   n_beams  : how many of the arrays below to process, FROM THE START —
//              callers exploit the scan-major layout to carve "the first 5
//              scans" by simply passing kLateLeaverBeamCount (file header).
//   scan_id  : [n_beams] which scan each beam belongs to (indexes the
//              __constant__ scan-origin table set_scan_origins() uploads).
//   dir      : [n_beams*3] unit direction, world frame, INTERLEAVED xyz.
//   is_hit   : [n_beams] 1 = real return, 0 = max-range.
//   range    : [n_beams] meters (see file header "BEAM RECORD LAYOUT").
__global__ void carve_kernel(int n_beams,
                             const int* __restrict__ scan_id,
                             const float* __restrict__ dir,
                             const unsigned char* __restrict__ is_hit,
                             const float* __restrict__ range,
                             Ledger ledger);

// carve_trace_kernel — one thread per beam IN A SMALL SUBSET (the verify-
// stage "documented subset", README "Expected output"): march the SAME DDA
// path as carve_kernel but write the ORDERED sequence of visited voxel
// linear indices to trace_out instead of touching any ledger (this kernel
// takes no Ledger argument at all — it is purely an instrumentation path,
// see kernels.cu's shared __device__ helper for how the two kernels reuse
// one march implementation without duplicating it on the GPU side).
//   n_beams    : subset size (small — main.cu documents and prints it).
//   trace_out  : [n_beams * kMaxDDASteps] OUT, row per beam, this beam's
//                visited voxel indices in march order (unused tail entries
//                are undefined — trace_len says how many are valid).
//   trace_len  : [n_beams] OUT, how many of trace_out's kMaxDDASteps slots
//                this beam actually used.
__global__ void carve_trace_kernel(int n_beams,
                                   const int* __restrict__ scan_id,
                                   const float* __restrict__ dir,
                                   const unsigned char* __restrict__ is_hit,
                                   const float* __restrict__ range,
                                   int* __restrict__ trace_out,
                                   int* __restrict__ trace_len);

// classify_kernel — one thread per BEAM (file header: threads for max-range
// beams exit immediately — no point exists to classify; the "some threads
// inactive" cost is negligible next to the simplicity of a uniform launch
// over the same array carve_kernel already walks).
//   ledger    : the (already carved) ledger to score against.
//   threshold : kDynamicThreshold in every caller; exposed as a parameter so
//               main.cu's late_leaver gate can reuse this kernel unchanged
//               against TWO different ledgers (5-scan and 10-scan) without
//               a second copy.
//   score_out : [n_beams] OUT, the ratio (file header "THE PRINCIPLE" step
//               4); left at -1.0f for beams with is_hit == 0 (no point).
//   label_out : [n_beams] OUT, 1 = DYNAMIC (remove), 0 = STATIC (keep), -1
//               = not applicable (max-range beam, no point).
__global__ void classify_kernel(int n_beams,
                                const int* __restrict__ scan_id,
                                const float* __restrict__ dir,
                                const unsigned char* __restrict__ is_hit,
                                const float* __restrict__ range,
                                Ledger ledger, float threshold,
                                float* __restrict__ score_out,
                                int* __restrict__ label_out);

#endif // __CUDACC__

// ---------------------------------------------------------------------------
// set_scan_origins — upload the K sensor positions to GPU __constant__
// memory (kernels.cu owns the __constant__ symbol). Beams are scan-major
// (file header), so within any warp of carve_kernel/classify_kernel threads
// most lanes share the SAME scan_id and therefore the SAME origin read — a
// broadcast pattern, the same reasoning 02.08's set_trajectory() gives for
// its own __constant__ upload. Must be called once before any launch below.
//   host_origin_xyz : [kNumScans*3] host floats, interleaved xyz (m).
// ---------------------------------------------------------------------------
void set_scan_origins(const float* host_origin_xyz);

// ---------------------------------------------------------------------------
// Host launch wrappers (definitions in kernels.cu). Each owns its grid/block
// math + the mandatory post-launch error check (CLAUDE.md §6.1 rule 7).
// ---------------------------------------------------------------------------
void launch_ledger_clear(Ledger ledger);
void launch_carve(int n_beams, const int* d_scan_id, const float* d_dir,
                  const unsigned char* d_is_hit, const float* d_range,
                  Ledger ledger);
void launch_carve_trace(int n_beams, const int* d_scan_id, const float* d_dir,
                        const unsigned char* d_is_hit, const float* d_range,
                        int* d_trace_out, int* d_trace_len);
void launch_classify(int n_beams, const int* d_scan_id, const float* d_dir,
                     const unsigned char* d_is_hit, const float* d_range,
                     Ledger ledger, float threshold,
                     float* d_score_out, int* d_label_out);

// ===========================================================================
// CPU references (reference_cpu.cpp) — the correctness-oracle twins. All
// pointers below are HOST pointers; scan_origin_xyz is a plain host array
// (no __constant__ memory on a CPU — the same difference 02.08's kernels.cuh
// comment names for its own CPU twin). See reference_cpu.cpp's file header
// for the independence ruling each of these follows.
// ===========================================================================

// carve_cpu — the INDEPENDENT twin of carve_kernel: same DDA algorithm,
// typed fresh, sequential over beams (no atomics needed — a single thread
// can never race itself). main.cu compares its ledger against the GPU's,
// element-wise, EXACT (the "hit/pass ledgers EXACT" verify gate).
void carve_cpu(int n_beams, const int* scan_id, const float* dir,
               const unsigned char* is_hit, const float* range,
               const float* scan_origin_xyz,
               unsigned int* hits, unsigned int* pass_from_hit,
               unsigned int* pass_from_maxrange);

// carve_trace_one_beam_cpu — the INDEPENDENT twin of carve_trace_kernel, for
// exactly ONE beam (main.cu calls this once per beam in the documented
// verify subset, mirroring the GPU's one-thread-per-beam trace). Returns the
// visited voxel count (<= kMaxDDASteps); trace_out must have that capacity.
int carve_trace_one_beam_cpu(float ox, float oy, float oz,
                             float dx, float dy, float dz,
                             float range_m, bool is_hit,
                             int* trace_out);

// classify_cpu — the INDEPENDENT twin of classify_kernel: identical ratio
// formula, sequential over beams. main.cu compares its labels (and, with a
// small numerical-headroom tolerance, its scores) against the GPU's.
void classify_cpu(int n_beams, const int* scan_id, const float* dir,
                  const unsigned char* is_hit, const float* range,
                  const float* scan_origin_xyz,
                  const unsigned int* hits, const unsigned int* pass_from_hit,
                  const unsigned int* pass_from_maxrange, float threshold,
                  float* score_out, int* label_out);

#endif // PROJECT_KERNELS_CUH
