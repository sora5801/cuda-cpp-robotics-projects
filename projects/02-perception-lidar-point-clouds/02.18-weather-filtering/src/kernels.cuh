// ===========================================================================
// kernels.cuh - interface for project 02.18
//               Weather filtering: snow/rain/dust outlier removal (DROR/LIOR)
//
// Role in the project
// -------------------
// The CONTRACT shared by main.cu (orchestration + gates), kernels.cu (the
// GPU kernels), reference_cpu.cpp (its independent CPU twins), and
// scripts/make_synthetic.py (whose "MUST MATCH" comments mirror the beam
// model constants below). Everything all four must agree on - the beam
// model, the three filters' parameters, and the point-record layout - is
// defined HERE, once (CLAUDE.md paragraph 12).
//
// THE THREE FILTERS, in five lines (THEORY.md derives every claim below)
// ---------------------------------------------------------------------------
//   1. SOR (Statistical Outlier Removal, the generic baseline): for every
//      point, the mean distance to its K nearest neighbors; a point is an
//      outlier if that mean distance exceeds the WHOLE CLOUD's
//      mean + kSorStdMult * std. This assumes point density is roughly
//      UNIFORM across the cloud - an assumption a spinning LiDAR's own
//      geometry violates (project 02.01's 1/r^2 density lesson, cited
//      throughout this project): a real, solid surface far from the sensor
//      is sampled by the SAME angular grid as a near one, so its points are
//      naturally farther apart. SOR cannot tell "far and sparse but real"
//      from "an isolated snowflake" - it flags both, which is exactly the
//      SOR_far_range_failure gate this project measures (README/THEORY).
//   2. DROR (Dynamic Radius Outlier Removal, Charron et al. 2018): fixes
//      SOR's blind spot by growing the SEARCH RADIUS with range instead of
//      using a fixed neighbor count/distance globally. r_search(r) =
//      max(beta * alpha * r, r_min), where alpha is the sensor's angular
//      resolution (radians) - the same quantity that sets a real surface's
//      point SPACING at range r (arc length ~= alpha * r). A point is an
//      outlier if fewer than kDrorKMin neighbors fall within its OWN
//      range-scaled radius. Because the radius grows exactly as fast as
//      real-surface spacing grows, a far real point still finds its
//      neighbors; an isolated airborne scatterer, whose neighbors (if any)
//      are other scatterers scattered randomly through 3-D space, usually
//      does not.
//   3. LIOR (Low-Intensity Outlier Removal, this project's teaching version
//      of the intensity-based family of weather filters): snow/rain/dust
//      returns are systematically DIM (THEORY.md derives why: partial beam
//      interception by a millimeter-scale particle, and - for real sensors -
//      the calibration gain 02.20 studies). A point is an outlier if its
//      intensity is below kLiorIntensityThresh AND it is locally sparse
//      within a small FIXED radius kLiorRadius (fewer than kLiorKMin
//      neighbors) - the density companion exists so a genuinely dim but
//      densely-and-coherently-sampled real surface (e.g. dark asphalt at a
//      grazing angle) is not thrown away just for being dark.
// All three share ONE brute-force building block: "how many of my n-1
// neighbors lie within radius R of me" (or its K-nearest cousin for SOR) -
// this project deliberately does NOT build a spatial index (that is project
// 02.05's/02.09's job, cited here, not re-taught); with n on the order of
// one to two thousand points per scan, O(n^2) brute force is well within a
// desktop GPU's (and even a single CPU core's) budget, and it keeps every
// line of this project about the FILTERING MATH, not spatial acceleration.
//
// POINT-RECORD LAYOUT - one weather scan at a time (main.cu loops over the
// three committed scans; every kernel below operates on ONE scan's n points):
//     xyz[i*3+0..2] : meters, SENSOR frame (sensor at the origin every scan -
//                     a deliberate scope cut, see scripts/make_synthetic.py's
//                     module docstring and README "Limitations": this
//                     project compares three INDEPENDENT captures of the
//                     SAME static scene under three atmospheres, not one
//                     continuously moving platform).
//     intensity[i]  : unitless, [0,1] - see scripts/make_synthetic.py's
//                     module docstring for the Lambertian-reflectance (real
//                     points) / partial-interception (scatterer points)
//                     forward model that produced it.
// xyz is stored INTERLEAVED (not split x[]/y[]/z[] arrays) to match this
// repo's PointCloud convention (02.01/02.06/02.13/SYSTEM_DESIGN.md section
// 3.6) - an honest coalescing trade documented at the kernel launch sites in
// kernels.cu, the same trade 02.13's kernels.cuh names for its own beam
// records (README Exercise: try the split layout and re-profile).
//
// GROUND TRUTH - is_real[i]/scatterer_type[i]/surf_cohort[i], loaded
// alongside the points but used ONLY by main.cu's gates and artifacts,
// NEVER by the filtering kernels themselves (CLAUDE.md paragraph 8 "never
// fabricate results" cuts both ways: the algorithms must not cheat by
// reading their own answer key - the same discipline 02.13's kernels.cuh
// states for its own cohort/truth fields).
//
// Range r_i = ||xyz_i|| (sensor is the origin) is DERIVED, never stored
// twice - computed inline everywhere DROR needs it (single-sourced, the
// same "P is derived" discipline 02.13's kernels.cuh states for its own
// beam-endpoint formula).
//
// Why this header is CUDA-qualifier-free where possible, HD elsewhere
// ---------------------------------------------------------------------
// The shared math below (squared_distance3, the (dist2,index) tie-break,
// the dynamic-radius formula) is DATA-LAYOUT / FORMULA bookkeeping, not the
// search ALGORITHM itself - declared HD ("__host__ __device__" under nvcc,
// nothing under cl.exe; the same macro 01.10/02.08/02.09/02.13 use) and
// SHARED, token-for-token, by kernels.cu's kernels and reference_cpu.cpp's
// twins, per the independence ruling reference_cpu.cpp's file header states
// in full: sharing a four-line formula is transcription, not the algorithm
// under test; the ACTUAL O(n) brute-force scan loop that consumes it is
// typed out INDEPENDENTLY in each file. Because the shared formulas alone
// cannot hide a search-loop bug, the CLASSIFY-stage verify gates below
// additionally re-run each classify kernel against the OTHER path's already-
// exact-verified per-point statistic array (see main.cu's VERIFY stage,
// 02.13's ledger-then-classify precedent) - a real cross-check, not a
// tautology.
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cmath>      // sqrtf/floorf - used by the HD helpers below (identical on host and device)
#include <cstdint>    // int32_t etc. - exact-width integers

// ---------------------------------------------------------------------------
// HD - "__host__ __device__" under nvcc, nothing under cl.exe. Same trick as
// 01.10/02.08/02.09/02.13 (see their kernels.cuh file headers for the full
// rationale): lets kernels.cu's device code and reference_cpu.cpp's host
// twin call the IDENTICAL compiled-twice formula without either translation
// unit seeing a CUDA keyword it cannot parse.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ===========================================================================
// Beam model - MUST MATCH ../scripts/make_synthetic.py's module-level
// constants of the same name (main.cu asserts the data file's '#'-prefixed
// header against these at load time, the 02.08/02.13-style data/code
// consistency check). Not used by the filtering kernels themselves (they
// only ever see already-formed points) - these exist so main.cu can sanity-
// check the committed sample and report the beam budget honestly.
// ===========================================================================
constexpr int kNumBeams = 16;                 // elevation rings per sweep
constexpr int kAzimuthSteps = 100;            // covers [-50, +49] deg, 1 deg/step
constexpr float kMaxRangeM = 45.0f;           // sensor max usable range (m)
constexpr int kBeamsPerScan = kNumBeams * kAzimuthSteps;   // 1,600

// Weather scan ids - MUST MATCH ../scripts/make_synthetic.py's WEATHER_*.
enum WeatherId : int32_t { kWeatherSnow = 0, kWeatherRain = 1, kWeatherDust = 2, kNumWeatherScans = 3 };
inline const char* weather_name(int32_t w)
{
    switch (w) {
        case kWeatherSnow: return "snow";
        case kWeatherRain: return "rain";
        case kWeatherDust: return "dust";
        default:           return "?";
    }
}

// Real-surface cohort ids - MUST MATCH ../scripts/make_synthetic.py's
// COHORT_*. GROUND TRUTH ONLY (file header) - never read by the filters.
enum SurfCohort : int32_t {
    kCohortGround   = 0,
    kCohortWallNear = 1,
    kCohortWallFar  = 2,
    kCohortCar      = 3,
    kCohortNone     = -1   // scatterer point: not applicable
};

// ===========================================================================
// SOR (Statistical Outlier Removal) parameters (file header point 1).
// kSorK: neighbor count for the local mean-distance statistic - a small,
// textbook value (PCL's own SOR filter defaults to a similar magnitude);
// large enough to average out single-neighbor noise, small enough to stay
// LOCAL (a large K starts measuring cloud-scale structure, not local
// density, defeating the whole idea of a *local* outlier test).
// kSorStdMult: how many standard deviations above the cloud-wide mean
// counts as "too far" - MEASURED (README "Expected output") against this
// project's committed sample: 0.5 is the value at which SOR's far-range
// real-point false-removal rate becomes large and clearly visible (the
// designed lesson) while its near-range false-removal rate stays exactly
// zero - a value like the PCL-typical 1.0 washes out the far/near contrast
// on this project's scene (see THEORY.md "How we verify correctness" for
// the swept comparison).
// ===========================================================================
constexpr int   kSorK = 8;
constexpr float kSorStdMult = 0.5f;

// ===========================================================================
// DROR (Dynamic Radius Outlier Removal, Charron et al. 2018) parameters
// (file header point 2). kDrorAlphaRad is the sensor's native ANGULAR
// resolution in radians - here the azimuth step (1 deg), the tighter of the
// two sampling axes and the one Charron et al.'s own formula uses (their
// "alpha" is the sensor's horizontal angular resolution). kDrorBeta is the
// paper's tunable safety multiplier (they report good results for beta in
// roughly 3-6; this project uses their lower, more conservative end).
// kDrorRMin floors the search radius so a point very near the sensor (r ->
// 0) still gets a sane, nonzero neighborhood. kDrorKMin is the minimum
// neighbor count (WITHIN that radius) for "not an outlier" - Charron et al.
// use a small integer; this project measured (README) that 3 clearly
// separates real structure from isolated scatterers on the committed scene.
// ===========================================================================
constexpr float kDrorAlphaRad = 0.0174533f;   // 1 deg in radians (== the azimuth step)
constexpr float kDrorBeta = 3.0f;
constexpr float kDrorRMin = 0.05f;            // meters
constexpr int   kDrorKMin = 3;

// ---------------------------------------------------------------------------
// dror_search_radius - r_search(r) = max(beta * alpha * r, r_min) (file
// header point 2; derived in full in THEORY.md "The math"). HD + shared
// (file header "Why this header is HD") because it is a four-term formula,
// not the search algorithm - both kernels.cu and reference_cpu.cpp call
// this SAME compiled-twice function so the radius itself can never drift
// between the GPU and CPU paths; only the O(n) scan that USES it is typed
// independently in each file.
// ---------------------------------------------------------------------------
HD inline float dror_search_radius(float range_m)
{
    const float r = kDrorBeta * kDrorAlphaRad * range_m;
    return r > kDrorRMin ? r : kDrorRMin;
}

// ===========================================================================
// LIOR (Low-Intensity Outlier Removal) parameters (file header point 3).
// kLiorIntensityThresh sits between the two intensity populations measured
// on the committed sample (README "Expected output" quotes the actual
// percentiles): real surfaces sit mostly at 0.02-0.55 (grazing-angle ground
// through near-normal car paint), scatterer returns sit mostly at 0.00-0.04
// with a long thin tail. kLiorRadius/kLiorKMin are the FIXED companion
// density check (file header point 3 explains why it exists); FIXED, not
// range-scaled like DROR's, is the deliberate simplicity/cost tradeoff LIOR
// makes over DROR - and, measured honestly, also the reason LIOR is the
// filter that (in this project's dust scene) struggles more on the dense
// plume core than DROR does: LIOR's fixed radius is larger than DROR's own
// range-scaled radius at the plume's short range, so it saturates into
// "looks locally dense enough to keep" SOONER as scatterer density rises
// (README "Expected output", THEORY.md "Numerical considerations" derive
// this in full - a genuinely non-obvious, honestly-measured finding, not
// the naively-expected "LIOR simply wins").
// ===========================================================================
constexpr float kLiorIntensityThresh = 0.05f;
constexpr float kLiorRadius = 0.35f;          // meters
constexpr int   kLiorKMin = 2;

// ===========================================================================
// Range-band boundaries shared by main.cu's range-stratified [info] report
// AND the sor_far_range_failure gate (README "Expected output"): near
// (< kRangeNearM), mid ([kRangeNearM, kRangeFarM)), far (>= kRangeFarM).
// ===========================================================================
constexpr float kRangeNearM = 12.0f;
constexpr float kRangeFarM  = 25.0f;

// ===========================================================================
// Shared HD helpers (file header "Why this header is HD"): pure formula
// bookkeeping, safe to share token-for-token between the GPU and CPU paths.
// ===========================================================================

// squared_distance3 - ||p - q||^2 for two interleaved-xyz points. Squared
// (not sqrt'd) distance is used for every THRESHOLD comparison in this
// project (radius^2 vs dist2) - comparing squares avoids one sqrtf() per
// candidate in the O(n^2) inner loops below, the classic "don't take a
// square root you're about to throw away" micro-optimization, applied
// honestly (THEORY.md "Numerical considerations" confirms it changes no
// comparison's outcome: x < y iff x^2 < y^2 for x,y >= 0).
HD inline float squared_distance3(const float p[3], const float q[3])
{
    const float dx = p[0] - q[0], dy = p[1] - q[1], dz = p[2] - q[2];
    return dx * dx + dy * dy + dz * dz;
}

// range3 - ||p|| for a point in SENSOR frame (sensor at the origin, file
// header "POINT-RECORD LAYOUT") - DROR's r_i, computed inline, never stored.
HD inline float range3(const float p[3])
{
    return sqrtf(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
}

// The shared total order SOR's K-nearest search compares by, and its
// documented TIE-BREAK: smaller dist2 wins; on an EXACT dist2 tie, the
// SMALLER original point index wins (02.05's knn_less / 02.09's knn_less,
// cited, reimplemented here as this project's own tiny K=8 version). Fixing
// a total order means every implementation below produces the IDENTICAL
// K-nearest SET under a fixed candidate-discovery order (both GPU and CPU
// scan candidates j = 0..n-1 in the same order) - the precondition for the
// SOR mean-distance verify gate's tight-tolerance (not merely "equivalent
// set") agreement.
HD inline bool dist_less(float da, int32_t ia, float db, int32_t ib)
{
    if (da != db) return da < db;
    return ia < ib;
}

#ifdef __CUDACC__  // ---- device-aware section: only nvcc sees this ---------

// sor_mean_knn_dist_kernel - one thread per point: the K-nearest mean
// distance SOR's global statistic is built from (file header point 1).
// Full documentation with the definition in kernels.cu.
//   n         : points in this scan.
//   xyz       : [n*3] interleaved, DEVICE pointer, sensor frame, meters.
//   mean_dist : [n] OUT, meters - mean distance to this point's kSorK
//               nearest neighbors (excluding itself).
__global__ void sor_mean_knn_dist_kernel(int n, const float* __restrict__ xyz,
                                          float* __restrict__ mean_dist);

// sor_classify_kernel - one thread per point: threshold mean_dist against
// the (host-computed, see main.cu) cloud-wide mu + kSorStdMult*sigma.
//   threshold : mu + kSorStdMult*sigma, a single scalar the caller computes
//               ONCE from the (already GPU-vs-CPU-verified) mean_dist array
//               - passed in rather than recomputed here so the CLASSIFY
//               verify gate compares "same inputs in, same mask out"
//               (kernels.cuh file header, 02.13's ledger-then-classify
//               precedent).
//   mask_out  : [n] OUT, 1 = outlier (remove), 0 = keep.
__global__ void sor_classify_kernel(int n, const float* __restrict__ mean_dist,
                                     float threshold, int32_t* __restrict__ mask_out);

// dror_neighbor_count_kernel - one thread per point: how many neighbors
// fall within THIS point's own range-scaled radius (file header point 2).
//   xyz         : [n*3] interleaved, DEVICE pointer.
//   count_out   : [n] OUT - neighbor count within dror_search_radius(r_i).
__global__ void dror_neighbor_count_kernel(int n, const float* __restrict__ xyz,
                                            int32_t* __restrict__ count_out);

// dror_classify_kernel - threshold count_out against kDrorKMin.
__global__ void dror_classify_kernel(int n, const int32_t* __restrict__ count,
                                      int32_t* __restrict__ mask_out);

// lior_neighbor_count_kernel - one thread per point: how many neighbors
// fall within the FIXED kLiorRadius (file header point 3 - deliberately NOT
// range-scaled, the LIOR/DROR contrast this project measures).
__global__ void lior_neighbor_count_kernel(int n, const float* __restrict__ xyz,
                                            int32_t* __restrict__ count_out);

// lior_classify_kernel - outlier iff intensity < kLiorIntensityThresh AND
// count < kLiorKMin (file header point 3's "dim AND sparse" rule).
__global__ void lior_classify_kernel(int n, const float* __restrict__ intensity,
                                      const int32_t* __restrict__ count,
                                      int32_t* __restrict__ mask_out);

#endif // __CUDACC__ --------------------------------------------------------

// ---------------------------------------------------------------------------
// Host launch wrappers (definitions in kernels.cu). Each owns its grid/block
// math + the mandatory post-launch error check (CLAUDE.md paragraph 6, rule 7).
// ---------------------------------------------------------------------------
void launch_sor_mean_knn_dist(int n, const float* d_xyz, float* d_mean_dist);
void launch_sor_classify(int n, const float* d_mean_dist, float threshold, int32_t* d_mask_out);
void launch_dror_neighbor_count(int n, const float* d_xyz, int32_t* d_count_out);
void launch_dror_classify(int n, const int32_t* d_count, int32_t* d_mask_out);
void launch_lior_neighbor_count(int n, const float* d_xyz, int32_t* d_count_out);
void launch_lior_classify(int n, const float* d_intensity, const int32_t* d_count, int32_t* d_mask_out);

// ===========================================================================
// CPU references (reference_cpu.cpp) - the correctness-oracle twins. All
// pointers below are HOST pointers. See reference_cpu.cpp's file header for
// the independence ruling each of these follows (kernels.cuh file header
// "Why this header is HD").
// ===========================================================================
void sor_mean_knn_dist_cpu(int n, const float* xyz, float* mean_dist);
void sor_classify_cpu(int n, const float* mean_dist, float threshold, int32_t* mask_out);
void dror_neighbor_count_cpu(int n, const float* xyz, int32_t* count_out);
void dror_classify_cpu(int n, const int32_t* count, int32_t* mask_out);
void lior_neighbor_count_cpu(int n, const float* xyz, int32_t* count_out);
void lior_classify_cpu(int n, const float* intensity, const int32_t* count, int32_t* mask_out);

#endif // PROJECT_KERNELS_CUH
