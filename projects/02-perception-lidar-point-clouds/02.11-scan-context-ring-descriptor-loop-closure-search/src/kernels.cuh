// ===========================================================================
// kernels.cuh — interface for project 02.11
//               Scan Context / ring-descriptor loop-closure search
//
// Role in the project
// -------------------
// The CONTRACT shared by main.cu (orchestration, every VERIFY/GATE, every
// artifact), kernels.cu (the GPU kernels), and reference_cpu.cpp (the
// independent CPU oracle twins). Every data-layout decision all three must
// agree on — the point layout, the Scan Context matrix layout, the ring-key
// layout, the shift-distance layout, and the small deterministic formulas
// that turn a point into a (ring, sector) cell — is defined HERE, once
// (CLAUDE.md paragraph 12).
//
// Scan Context in five lines (Kim & Kim, IROS 2018, "Scan Context: Egocentric
// Spatial Descriptor for Place Recognition within 3D Point Cloud Map" — the
// method this project reimplements didactically; THEORY.md derives every
// formula below from first principles and cites the paper's exact role):
//   1. Bin every point of a scan into a POLAR grid around the sensor: ring
//      from range, sector from azimuth — an EGOCENTRIC signature.
//   2. Each cell stores the MAX height (z) of the points that land in it —
//      a compact summary of "how tall is the vertical structure in this
//      direction, this far out" (the descriptor's whole information content).
//   3. Two scans of the SAME place, taken from DIFFERENT headings, produce
//      the SAME matrix with its COLUMNS CYCLICALLY SHIFTED — rotation
//      becomes a shift (THEORY.md "the math" derives this exactly).
//   4. So: search over all NUM_SECTOR shifts for the one that makes two
//      matrices look most alike (mean column-wise cosine distance); the
//      minimum distance is "how similar are these places", and the shift
//      that achieves it is a FREE relative-yaw estimate.
//   5. Comparing a new scan against every scan ever seen is too slow — a
//      cheap per-ring "how full is this ring" ROW KEY, compared by L1
//      distance, prefilters a small candidate set before step 4 ever runs
//      (the two-stage search every real deployment uses).
//
// SCAN CONTEXT MATRIX LAYOUT — float sc[kNumRing * kNumSector], ROW-MAJOR,
// sc[r * kNumSector + s] = max sensor-frame z (meters) among points whose
// (range, azimuth) fall in ring r, sector s; kEmptyZ if no point ever landed
// there (the EMPTY-CELL SENTINEL — see the numerics note below). Sector is
// the FAST-VARYING index deliberately: kernels.cu's shift-distance kernel
// reads one (ring, sector) row at a time across all kNumSector threads of a
// block, and consecutive sectors are consecutive floats in memory — a
// coalesced access pattern that only exists because of this layout choice
// (the same "pick the layout the hot kernel wants" lesson as 08.01's
// transposed noise array, cited).
//
// RING KEY LAYOUT — float ringkey[kNumRing]: ringkey[r] = the FRACTION of
// sector cells in ring r that are non-empty (a real number in [0,1]), NOT
// the paper's row-mean-of-heights. This project's deliberate simplification
// (documented, not hidden): an occupancy fraction is exactly the coarse
// "how much structure surrounds me at this range" fingerprint a prefilter
// needs, and — unlike a mean over cells that mixes real heights with the
// empty-cell sentinel — it never needs the sentinel's value to be
// meaningful, which keeps the empty-cell design decision isolated to the
// matrix itself (THEORY.md "numerical considerations").
//
// EMPTY-CELL SENTINEL — kEmptyZ = -1000.0f, the ORIGINAL Scan Context
// paper's own convention (Kim & Kim 2018, cited) — adopted here for a
// reason this project's own build process surfaced the hard way (THEORY.md
// "numerical considerations" tells the story in full): a running MAX must
// start from a value no real reading can ever beat. This project's sensor
// sits kSensorHeightM above the ground it rides on, so a GROUND hit reads a
// legitimate, informative NEGATIVE sensor-frame z (around -kSensorHeightM,
// not near zero) — an early version of this file used 0.0f as the sentinel,
// reasoning that "a real return is rarely near exactly 0.0"; that reasoning
// was correct but irrelevant, because 0.0f as a MAX-SEED silently discards
// every ground-only cell (-1.6 m never beats a 0.0 m seed), not just
// coincidental exact-zero returns — the descriptor was accidentally
// throwing away most of its own ground signal. kEmptyZ = -1000.0f is far
// below any physically reachable sensor-frame height in this project's
// world, so every real reading — ground included — always wins the max.
//
// Because kEmptyZ is a large NEGATIVE number, it must NEVER be fed directly
// into the cosine-distance arithmetic below (a column with one real 0.3 m
// reading and nineteen -1000.0f "empty" entries would have a cosine
// geometry dominated by -1000.0f, not by the one real number) — every
// consumer of sc[] MASKS kEmptyZ to a neutral 0.0 contribution first and
// separately tracks whether a ring cell was real vs. empty. This also
// creates the genuine DEGENERATE COLUMN case the numerics note promises: an
// entire sector-column with NO real return anywhere in range (e.g. looking
// straight down a wide open street beyond kSensorMaxRangeM) — handled
// explicitly by column_cosine_distance() below (a documented fallback, not
// a crash): two columns that are BOTH entirely empty AGREE ("nothing out
// there", in both scans) and score 0.0 distance; a column empty in only ONE
// of the two scans is a genuine conflict and scores the maximum, 1.0.
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cmath>      // std::atan2/std::sqrt/std::fabs — identical overloads under cl.exe and nvcc's host pass
#include <cstdint>

// ===========================================================================
// Problem-scale constants — the numbers every stage and every CPU twin below
// must agree on bit-for-bit (CLAUDE.md paragraph 12: one source of truth).
// ===========================================================================

// kNumRing / kNumSector — the classic Scan Context grid (Kim & Kim 2018's
// own choice, cited): 20 rings x 60 sectors. Sector width = 360/60 = 6 deg —
// this is the descriptor's fundamental YAW RESOLUTION: no shift search can
// ever report a rotation finer than 6 deg, a bound the rotation_invariance
// gate in main.cu explicitly measures against (THEORY.md "numerical
// considerations" derives the expected error distribution from it).
constexpr int kNumRing   = 20;
constexpr int kNumSector = 60;
constexpr int kScCells   = kNumRing * kNumSector;   // = 1200 floats per scan descriptor

// kSensorMaxRangeM — ring 0..19 evenly divide [0, kSensorMaxRangeM). MUST
// equal scripts/make_synthetic.py's MAX_RANGE_M (data<->pipeline contract,
// stated in both files per CLAUDE.md paragraph 12 — there is no build-time
// check spanning Python and CUDA, so a change here means a change there).
constexpr float kSensorMaxRangeM = 40.0f;

// kSensorHeightM — sensor mount height above the ground it rides on. MUST
// equal scripts/make_synthetic.py's SENSOR_HEIGHT_M. Used only for
// DOCUMENTATION/reasoning here (a ground hit's sensor-frame z is always
// approximately -kSensorHeightM) — no code below actually needs the value,
// since kEmptyZ just needs to sit safely below the lowest reachable z.
constexpr float kSensorHeightM = 1.6f;

// kEmptyZ — the empty-cell sentinel (see the file header's full derivation
// of why this must be far below every physically reachable sensor-frame z,
// not merely "unlikely to occur exactly").
constexpr float kEmptyZ = -1000.0f;

constexpr float kPiF = 3.14159265358979323846f;

// kMinLoopGapKeyframes — a query at index q may only be compared against
// candidates with index <= q - kMinLoopGapKeyframes. MUST equal
// scripts/make_synthetic.py's MIN_LOOP_GAP_KF (used there to decide which
// curated revisit pairs are even meaningful loop-closure test cases — see
// that file's build_loop_pairs() docstring). THEORY.md "the algorithm"
// explains why every real system enforces this: without it, a query would
// "loop-close" against the keyframe from one second ago, which is odometry,
// not place recognition.
constexpr int kMinLoopGapKeyframes = 15;

// kRingKeyPrefilterBudget — how many candidates the cheap ring-key L1
// prefilter hands to the expensive full shift-distance search. THEORY.md
// "the math" and main.cu's ringkey_prefilter gate quantify the recall this
// budget buys versus an exhaustive search over every valid candidate.
constexpr int kRingKeyPrefilterBudget = 12;

// kScDistanceThreshold — the operating point on the shift-distance PR curve
// (main.cu sweeps the full curve to demo/out/pr_curve.csv; this is the ONE
// value the loop_detection/rotation_invariance/negative_cohort gates apply).
// Chosen from a measured sweep of the committed sample, not the original
// paper's own value — this project's synthetic world, ray density, and
// un-normalized geometry are different enough from the paper's KITTI
// setting that reusing their number would be cargo-culting, not
// verification (README "Limitations"). The measured sweep (main.cu's
// diagnostic run, reproduced in THEORY.md "how we verify correctness")
// showed every genuine same-place revisit scoring <= 0.03 and the closest
// GENUINE aliasing confound (two different, structurally similar streets)
// scoring >= 0.13 — 0.10 sits with wide margin (CLAUDE.md paragraph 12:
// "success thresholds carry wide margins") in the middle of that gap,
// deliberately far from either edge rather than hugging the nearest
// negative example.
constexpr float kScDistanceThreshold = 0.10f;

// kPlaceRadiusM — the CONTINUOUS ground-truth definition used by the
// full-trajectory PR-curve sweep (as opposed to the curated cohort pairs,
// which are hand-labeled): two keyframes are "the same physical place" iff
// their true (x,y) positions are within this radius. Set slightly above the
// largest deliberately-authored lateral offset (2.6 m) so that cohort still
// counts as a true revisit for the sweep's ground truth.
constexpr float kPlaceRadiusM = 3.5f;

// ---------------------------------------------------------------------------
// blocks_for — integer ceiling division (the repo-wide idiom: 02.01/02.03/
// 02.10 all define this identically).
// ---------------------------------------------------------------------------
inline int blocks_for(int count, int threads)
{
    return (count + threads - 1) / threads;
}

// ---------------------------------------------------------------------------
// ring_index_from_range / sector_index_from_xy — the small, deterministic,
// formulaic pieces every implementation below must agree on bit-for-bit
// (02.10's darboux_triplet/angle_to_bin precedent, cited: shared as PLAIN
// host-callable inline C++ so reference_cpu.cpp calls them DIRECTLY; being
// unqualified they are HOST-ONLY under nvcc's rules, so kernels.cu carries
// its own literal __device__ transcription for use inside a kernel — cross-
// referenced in comments at both copies, per 02.10's ruling).
//
// THEORY.md "the math" derives both formulas; the short version: ring is a
// linear bin of range in [0, kSensorMaxRangeM), sector is a linear bin of
// azimuth atan2(y,x) folded from (-pi,pi] into [0,2pi).
// ---------------------------------------------------------------------------
inline int ring_index_from_range(float range_m)
{
    const float ring_width_m = kSensorMaxRangeM / static_cast<float>(kNumRing);
    int r = static_cast<int>(range_m / ring_width_m);
    if (r < 0) r = 0;
    if (r >= kNumRing) r = kNumRing - 1;   // clamp: a return exactly at/just past max range still lands in the last ring
    return r;
}

inline int sector_index_from_xy(float x, float y)
{
    float az = std::atan2(y, x);                    // (-pi, pi]
    if (az < 0.0f) az += 2.0f * kPiF;                // fold into [0, 2pi)
    const float sector_width_rad = (2.0f * kPiF) / static_cast<float>(kNumSector);
    int s = static_cast<int>(az / sector_width_rad);
    if (s < 0) s = 0;
    if (s >= kNumSector) s = kNumSector - 1;         // clamp: guards the az==2pi floating-point edge
    return s;
}

// ---------------------------------------------------------------------------
// column_cosine_distance — the shift-search's inner comparison: how
// dissimilar is column col_a of matrix sc_a from column col_b of matrix
// sc_b, reading kNumRing values strided by kNumSector out of each flat
// row-major matrix. Returns 1 - cosine_similarity, in [0, 2] (THEORY.md "the
// math" — this is the standard cosine DISTANCE, zero for identical
// directions, one for orthogonal, two for opposite).
//
// MASKING (the file header's promise): kEmptyZ must never enter the cosine
// arithmetic as a raw number — it is not a height, it is "no data". Every
// ring cell is read once and classified: a REAL cell contributes its actual
// value to dot/norm_a_sq/norm_b_sq; an EMPTY cell (== kEmptyZ) contributes
// 0.0 (a neutral no-op for a dot product / sum-of-squares) and is instead
// tallied into any_real_a/any_real_b.
//
// DEGENERATE COLUMN handling (the numerics note the file header promises;
// this is also the story behind a real bug this project's own VERIFY runs
// caught — THEORY.md "numerical considerations" tells it in full): cosine
// similarity is the undefined 0/0 whenever a column has NO real cells at
// all (every ring empty — a direction with no return anywhere within range,
// e.g. straight down a wide open street). Two SUB-CASES need DIFFERENT
// answers, not one blanket rule:
//   * BOTH columns entirely empty — both scans agree "nothing out there
//     this way". That is a genuine piece of AGREEING evidence (the same as
//     two scans independently returning a matching real height), so it
//     resolves to distance 0.0f. Treating "empty" as maximal disagreement
//     regardless of whether the OTHER side is also empty is the bug this
//     project's own VERIFY sweep caught: it silently inflated the distance
//     between two IDENTICAL revisits by roughly one point of distance per
//     mutually-empty sector, pushing true same-place revisits well above
//     any sane operating threshold — worth re-deriving in THEORY.md.
//   * Exactly ONE column entirely empty — a real, informative conflict (one
//     scan saw a wall in this direction, the other saw nothing) — resolves
//     to the MAXIMUM distance 1.0f.
//   * Both columns have SOME real cells but the resulting vectors still
//     have near-zero magnitude (physically shouldn't happen in this
//     project's world, since every real height is bounded away from 0 by
//     kSensorHeightM — kept as a defensive fallback, also 1.0f).
// This function is called kNumSector times per (candidate, shift) — its
// mean is what kernels.cu's sc_shift_distance_kernel and this file's CPU
// twin (reference_cpu.cpp) both compute.
// ---------------------------------------------------------------------------
inline float column_cosine_distance(const float* sc_a, int col_a, const float* sc_b, int col_b)
{
    float dot = 0.0f, norm_a_sq = 0.0f, norm_b_sq = 0.0f;
    bool any_real_a = false, any_real_b = false;
    for (int r = 0; r < kNumRing; ++r) {
        const float raw_a = sc_a[r * kNumSector + col_a];
        const float raw_b = sc_b[r * kNumSector + col_b];
        const bool cell_a_empty = (raw_a <= kEmptyZ + 1.0f);   // margin well below any physical z, above kEmptyZ itself
        const bool cell_b_empty = (raw_b <= kEmptyZ + 1.0f);
        const float a = cell_a_empty ? 0.0f : raw_a;           // mask: empty contributes nothing to the geometry
        const float b = cell_b_empty ? 0.0f : raw_b;
        if (!cell_a_empty) any_real_a = true;
        if (!cell_b_empty) any_real_b = true;
        dot       += a * b;
        norm_a_sq += a * a;
        norm_b_sq += b * b;
    }
    if (!any_real_a && !any_real_b) return 0.0f;   // both agree "nothing here" — see comment above
    if (!any_real_a || !any_real_b) return 1.0f;   // exactly one saw structure, the other did not — a real conflict
    if (norm_a_sq < 1e-9f || norm_b_sq < 1e-9f) return 1.0f;   // defensive fallback (see comment above)
    return 1.0f - dot / (std::sqrt(norm_a_sq) * std::sqrt(norm_b_sq));
}

// ---------------------------------------------------------------------------
// ring_key_l1_distance — the cheap prefilter metric: L1 distance between two
// kNumRing-dimensional ring-key vectors. Small and formulaic like the two
// functions above — shared for the same reason.
// ---------------------------------------------------------------------------
inline float ring_key_l1_distance(const float* key_a, const float* key_b)
{
    float d = 0.0f;
    for (int r = 0; r < kNumRing; ++r) d += std::fabs(key_a[r] - key_b[r]);
    return d;
}

// ===========================================================================
// GPU kernel declarations — nvcc-only (the __CUDACC__ fence; see the
// template's explanation, repeated in every project's kernels.cuh).
// ===========================================================================
#ifdef __CUDACC__

// sc_init_kernel — set n_scans Scan Context matrices to the empty sentinel
// (0.0f) before the scatter pass below claims any real cells. One thread
// per (scan, cell) — n_scans*kScCells is small (128*1200 ~ 153K) so a flat
// grid-stride-free launch is plenty (kernels.cu's launcher sizes it).
__global__ void sc_init_kernel(int n_scans, float* __restrict__ sc_all);

// sc_build_kernel — THE SCATTER: one thread per POINT (across every scan at
// once — total_points is the grid size). Each thread computes its own
// point's (ring, sector) cell and claims the cell's running max height via
// atomicMaxFloat (kernels.cu defines it; see that file for why a CAS loop,
// not the order-preserving-bitcast trick, is this project's teaching
// choice). point_scan_id[i] says which of the n_scans matrices point i's
// cell lives in — sc_all is laid out as n_scans contiguous kScCells blocks.
__global__ void sc_build_kernel(int total_points,
                                const float* __restrict__ xyz,          // [total_points*3], sensor frame, meters
                                const int32_t* __restrict__ point_scan_id,  // [total_points], which scan this point belongs to
                                float* __restrict__ sc_all);            // [n_scans*kScCells] IN/OUT (scattered into)

// ring_key_kernel — one thread per (scan, ring): count non-empty sector
// cells in that ring, divide by kNumSector. Reads sc_all (already fully
// built), writes ringkey_all[n_scans*kNumRing].
__global__ void ring_key_kernel(int n_scans,
                                const float* __restrict__ sc_all,
                                float* __restrict__ ringkey_all);

// sc_shift_distance_kernel — THE SEARCH: one BLOCK per (candidate, shift)
// pair (grid = dim3(num_candidates, kNumSector)), kNumSector-padded-to-64
// THREADS per block, one thread per SECTOR of the query matrix. Each thread
// computes column_cosine_distance() between its own query column and the
// shift-rotated candidate column, then the block reduces the kNumSector
// per-thread distances to their mean — kernels.cu's header derives the
// launch geometry and the coalescing argument for this layout in full.
// out_dist[c*kNumSector + shift] = mean column cosine distance.
__global__ void sc_shift_distance_kernel(const float* __restrict__ sc_query,      // [kScCells], ONE query matrix
                                         int num_candidates,
                                         const float* __restrict__ sc_candidates, // [num_candidates*kScCells]
                                         float* __restrict__ out_dist);           // [num_candidates*kNumSector] OUT

#endif // __CUDACC__

// ===========================================================================
// Host-callable launch wrappers (definitions in kernels.cu).
// ===========================================================================
void launch_sc_init(int n_scans, float* d_sc_all);
void launch_sc_build(int total_points, const float* d_xyz, const int32_t* d_point_scan_id, float* d_sc_all);
void launch_ring_key(int n_scans, const float* d_sc_all, float* d_ringkey_all);
void launch_sc_shift_distance(const float* d_sc_query, int num_candidates,
                              const float* d_sc_candidates, float* d_out_dist);

// ===========================================================================
// CPU references (reference_cpu.cpp) — the correctness-oracle twins. All
// pointers are HOST pointers. Per the repo's independence ruling
// (reference_cpu.cpp's file header states it in full): the small formulaic
// pieces above (ring/sector index, column cosine distance, ring-key L1) are
// SHARED — called directly, not re-derived — while the AGGREGATION LOOPS
// (the parallel atomic scatter vs. a sequential max; the per-block reduction
// vs. a sequential mean) are INDEPENDENTLY reimplemented, which is where a
// real GPU-only bug (a race, a wrong thread-to-cell mapping, a bad reduction)
// would actually show up.
// ===========================================================================
void sc_build_cpu(int total_points, const float* xyz, const int32_t* point_scan_id,
                  int n_scans, float* sc_all);
void ring_key_cpu(int n_scans, const float* sc_all, float* ringkey_all);
void sc_shift_distance_cpu(const float* sc_query, int num_candidates,
                           const float* sc_candidates, float* out_dist);

#endif // PROJECT_KERNELS_CUH
