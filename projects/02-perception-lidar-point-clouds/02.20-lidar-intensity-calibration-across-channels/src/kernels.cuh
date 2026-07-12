// ===========================================================================
// kernels.cuh - interface for project 02.20
//               LiDAR intensity calibration across channels
//
// Role in the project
// -------------------
// The CONTRACT shared by main.cu (orchestration + gates), kernels.cu (the
// GPU kernels), reference_cpu.cpp (its independent CPU twins), and
// scripts/make_synthetic.py (whose "MUST MATCH" comments mirror the scene
// constants below). Everything all four must agree on - the forward model,
// the scene geometry, the voxel grid, and the point-record layout - is
// defined HERE, once (CLAUDE.md paragraph 12).
//
// THE PROJECT IN FOUR SENTENCES (THEORY.md derives every claim below)
// ---------------------------------------------------------------------------
// A 16-beam LiDAR's channels do not agree on how bright the same surface is:
// each beam's own laser/detector pair has a slightly different GAIN g_ch
// (aging, alignment, manufacturing variance). This project recovers all 16
// relative gains from ONE scan, with NO reflectance targets: it finds small
// patches of world space ("voxels") that MULTIPLE channels happened to
// observe, and - after dividing out the one part of the signal every
// channel already agrees on (how range and incidence angle affect return
// strength) - the remaining cross-channel disagreement in a shared patch can
// only be gain. A tiny per-channel least-squares solve turns thousands of
// such local disagreements into one global gain estimate per channel, with
// an explicit check for channels no other channel ever overlapped with.
//
// THE FORWARD MODEL (single-sourced here; make_synthetic.py's module
// docstring derives the physics in full; THEORY.md "The math" restates it
// with units):
//
//     I = g[ch] * R_surface * f(r) * cos(theta) + noise
//
//   g[ch]      - per-channel GAIN (the unknown this project recovers).
//   R_surface  - the struck surface's reflectivity (unknown to the
//                calibration algorithm; a "nuisance" the shared-voxel trick
//                eliminates without ever estimating it).
//   f(r)       - range_falloff() below: a 1/r^2 regime beyond kRangePlateauM,
//                clamped flat inside it (the near-range defocus plateau).
//   cos(theta) - Lambertian incidence-angle falloff from the struck
//                surface's own normal (classify_normal_family() below - this
//                teaching version uses KNOWN, axis-aligned scene planes
//                rather than a per-voxel plane fit; README "Limitations"
//                names 02.03/02.09's RANSAC/plane-fit as the production
//                replacement for an arbitrary scene).
//
// SCENE GEOMETRY - MUST MATCH ../scripts/make_synthetic.py's constants of
// the same name (main.cu can therefore recompute geometry-derived quantities
// - the incidence normal, the range - from a point's OWN xyz, exactly like a
// system calibrating against a KNOWN environment would; it never reads the
// ground-truth surf_id/R_true columns to do so - see "GROUND TRUTH" below).
// Four real planar surfaces plus one used only by the degenerate scan:
//     GROUND      - horizontal plane z = kGroundZM.
//     WALL_NEAR   - vertical plane x = kWallNearXM, a broad bright surface.
//     PANEL       - a small, brighter patch beside WALL_NEAR at the SAME
//                   range (the multi-material-at-similar-range cohort).
//     WALL_FAR    - vertical plane x = kWallFarXM, same normal family as
//                   WALL_NEAR (both are "near-plane" for cos(theta) purposes
//                   - no, WALL_FAR is the FAR plane family; see enum below),
//                   naturally sparser cross-channel overlap purely from beam
//                   angular spacing at range.
//     ISOLATED_TARGET - horizontal plane z = kIsolatedZM, reached ONLY by
//                   the degenerate scan's retargeted channel (see
//                   scan_degenerate.csv) - no other channel's ray ever gets
//                   near it, by construction.
//
// GROUND TRUTH - surf_id/R_true, loaded from the CSV alongside points, used
// ONLY by main.cu's gates/artifacts, NEVER by the calibration kernels
// (kernels.cu/reference_cpu.cpp) - the same discipline 02.13/02.18's
// kernels.cuh state for their own truth fields. classify_normal_family()
// below is NOT ground truth: it is the calibration algorithm's own
// (documented, scene-specific) geometric model, exactly as legitimate as a
// real system's stored extrinsic calibration against a known target room.
//
// VOXEL GRID - THE OBSERVABILITY-GRAPH CURRENCY (THEORY.md "The math" full
// derivation). voxel_coord() below bins a coordinate to the NEAREST multiple
// of kVoxelLeafM (round-half-up, via floor(v/leaf + 0.5)) rather than
// project 02.01's plain floor(v/leaf). This is a DELIBERATE, measured
// difference from 02.01's convention, not an inconsistency: 02.01's plain
// floor puts a bin BOUNDARY at every multiple of the leaf size, including
// exactly zero; this project's beam fan is symmetric about elevation 0, so
// with a plain floor the two beams straddling elevation 0 would sit on
// OPPOSITE sides of that boundary in EVERY azimuth column, for EVERY leaf
// size - a permanent, structural graph cut no amount of extra data heals
// (verified empirically while building this project's synthetic scene - see
// THEORY.md "Numerical considerations" for the measured before/after
// connectivity). Centering the bins on multiples of the leaf instead moves
// the boundary away from the physically special coordinate 0, which is all
// it takes to reconnect the whole 16-channel graph on this project's scene.
// A real deployment would use its OWN sensor's beam table to check for this
// kind of grid/geometry aliasing rather than relying on luck.
//
// Why this header is CUDA-qualifier-free where possible, HD elsewhere
// ---------------------------------------------------------------------
// Shared formula bookkeeping (range_falloff, classify_normal_family,
// voxel_coord, the projector-accumulation math) is DATA-LAYOUT / FORMULA,
// not the search/reduction ALGORITHM itself - declared HD ("__host__
// __device__" under nvcc, nothing under cl.exe; the same macro 01.10/02.08/
// 02.09/02.13/02.18 use) and SHARED, token-for-token, by kernels.cu's
// kernels and reference_cpu.cpp's twins, per the independence ruling
// reference_cpu.cpp's file header states in full: sharing a short formula is
// transcription, not the algorithm under test. The ACTUAL reduction
// (per-voxel-per-channel accumulation, the least-squares assembly) is typed
// out INDEPENDENTLY in each file - see reference_cpu.cpp.
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cmath>      // sqrtf/fabsf/floorf/logf - identical on host and device
#include <cstdint>    // int32_t etc. - exact-width integers

// ---------------------------------------------------------------------------
// HD - "__host__ __device__" under nvcc, nothing under cl.exe. See file
// header "Why this header is HD".
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ===========================================================================
// SECTION 1 - problem scale (single-sourced; MUST MATCH
// ../scripts/make_synthetic.py's NUM_BEAMS).
// ===========================================================================
constexpr int kNumBeams = 16;

// ===========================================================================
// SECTION 2 - scene geometry (MUST MATCH ../scripts/make_synthetic.py's
// constants of the same name - file header "SCENE GEOMETRY"). Units: meters,
// sensor frame (sensor at the origin).
// ===========================================================================
constexpr float kGroundZM = -1.2f;

constexpr float kWallNearXM = 8.0f;
constexpr float kWallNearYLo = -2.2f, kWallNearYHi = 2.2f;
constexpr float kWallNearZLo = -1.5f, kWallNearZHi = 3.0f;

constexpr float kPanelXM = 8.0f;
constexpr float kPanelYLo = 2.4f, kPanelYHi = 3.0f;
constexpr float kPanelZLo = -0.5f, kPanelZHi = 2.0f;

constexpr float kWallFarXM = 20.0f;
constexpr float kWallFarYLo = -11.0f, kWallFarYHi = 11.0f;
constexpr float kWallFarZLo = -3.0f, kWallFarZHi = 5.0f;

constexpr float kIsolatedZM = 6.0f;

constexpr float kMaxRangeM = 45.0f;

// Ground-truth surf_id values, matching ../scripts/make_synthetic.py's
// SURF_* exactly (used ONLY by main.cu's gates - file header "GROUND TRUTH").
enum GroundTruthSurfId : int32_t {
    kGtGround = 0, kGtWallNear = 1, kGtPanel = 2, kGtWallFar = 3, kGtIsolated = 4
};

// Normal-FAMILY ids the calibration algorithm's OWN geometric classifier
// (classify_normal_family below) can distinguish - coarser than
// GroundTruthSurfId because WALL_NEAR and PANEL share one plane (x =
// kWallNearXM) and therefore one normal; the algorithm has no reason to
// (and does not) tell them apart.
enum NormalFamily : int32_t {
    kFamilyGround = 0, kFamilyNearPlane = 1, kFamilyFarPlane = 2, kFamilyIsolated = 3,
    kFamilyUnknown = -1
};

// Plane-coordinate matching tolerance for classify_normal_family (meters).
// Generously larger than this project's range noise (make_synthetic.py's
// RANGE_NOISE_SIGMA_M = 0.02 m projected onto the plane coordinate, at most
// a few cm) - see THEORY.md "Numerical considerations" for the margin math.
constexpr float kSurfPlaneTolM = 0.5f;

// ---------------------------------------------------------------------------
// classify_normal_family - the calibration algorithm's OWN model of "which
// known plane did this point come from, and what is that plane's outward
// normal" (file header "SCENE GEOMETRY" / "Why this header is HD"). Derived
// PURELY from the point's own xyz against this header's scene constants -
// never from the ground-truth surf_id column (file header "GROUND TRUTH").
//
// Order matters: the two vertical planes are checked FIRST, each requiring
// the point's (y,z) to ALSO fall inside that plane's rectangular extent -
// not just an x-coordinate match - because a grazing-incidence ground point
// can coincidentally have almost any x. Requiring the full box match makes
// an accidental cross-family match astronomically unlikely at this
// project's point counts (measured: zero mismatches against the generator's
// own surf_id across both committed scans - main.cu's classify_sanity
// [info] line reports this).
//
// Parameters: p[3] - a point's xyz, sensor frame, meters.
// Returns: the matched NormalFamily (kFamilyUnknown if no plane matches,
// which should never happen for a point this project's own generator
// produced - a fail-loud signal, not silently ignored, if it ever does).
// Side effect: writes the matched plane's unit outward normal into n_out.
// ---------------------------------------------------------------------------
HD inline int32_t classify_normal_family(const float p[3], float n_out[3])
{
    if (fabsf(p[0] - kWallNearXM) < kSurfPlaneTolM &&
        p[1] >= kWallNearYLo - kSurfPlaneTolM && p[1] <= kWallNearYHi + kSurfPlaneTolM &&
        p[2] >= kWallNearZLo - kSurfPlaneTolM && p[2] <= kWallNearZHi + kSurfPlaneTolM) {
        n_out[0] = -1.0f; n_out[1] = 0.0f; n_out[2] = 0.0f;
        return kFamilyNearPlane;
    }
    if (fabsf(p[0] - kWallNearXM) < kSurfPlaneTolM &&
        p[1] >= kPanelYLo - kSurfPlaneTolM && p[1] <= kPanelYHi + kSurfPlaneTolM &&
        p[2] >= kPanelZLo - kSurfPlaneTolM && p[2] <= kPanelZHi + kSurfPlaneTolM) {
        n_out[0] = -1.0f; n_out[1] = 0.0f; n_out[2] = 0.0f;
        return kFamilyNearPlane;   // panel shares WALL_NEAR's plane/normal (file header)
    }
    if (fabsf(p[0] - kWallFarXM) < kSurfPlaneTolM &&
        p[1] >= kWallFarYLo - kSurfPlaneTolM && p[1] <= kWallFarYHi + kSurfPlaneTolM &&
        p[2] >= kWallFarZLo - kSurfPlaneTolM && p[2] <= kWallFarZHi + kSurfPlaneTolM) {
        n_out[0] = -1.0f; n_out[1] = 0.0f; n_out[2] = 0.0f;
        return kFamilyFarPlane;
    }
    if (fabsf(p[2] - kGroundZM) < kSurfPlaneTolM) {
        n_out[0] = 0.0f; n_out[1] = 0.0f; n_out[2] = 1.0f;
        return kFamilyGround;
    }
    if (fabsf(p[2] - kIsolatedZM) < kSurfPlaneTolM) {
        n_out[0] = 0.0f; n_out[1] = 0.0f; n_out[2] = -1.0f;
        return kFamilyIsolated;
    }
    n_out[0] = 0.0f; n_out[1] = 0.0f; n_out[2] = 1.0f;
    return kFamilyUnknown;
}

// ===========================================================================
// SECTION 3 - the forward model's known part: range and incidence-angle
// falloff (file header "THE FORWARD MODEL"). MUST MATCH
// ../scripts/make_synthetic.py's RANGE_PLATEAU_M / range_falloff().
// ===========================================================================
constexpr float kRangePlateauM = 4.0f;

// range_falloff - f(r) = (kRangePlateauM / max(r, kRangePlateauM))^2. Flat
// (f=1) for r <= kRangePlateauM (the near-range defocus plateau a real
// receiver's optics produce - THEORY.md "The problem"), 1/r^2 beyond it.
// This project's committed geometry stays entirely in the 1/r^2 regime
// (nearest surface ~8 m, well past the 4 m plateau radius) - README
// "Limitations" states this scope cut honestly.
HD inline float range_falloff(float r_m)
{
    const float r_eff = r_m > kRangePlateauM ? r_m : kRangePlateauM;
    const float ratio = kRangePlateauM / r_eff;
    return ratio * ratio;
}

// point_range - ||p|| for a point in SENSOR frame (sensor at the origin).
HD inline float point_range(const float p[3])
{
    return sqrtf(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
}

// Numerical guards (THEORY.md "Numerical considerations" derives both):
// kIntensityFloor keeps log() away from -inf for the near-zero, grazing-
// incidence ground returns this project's own scene produces (measured:
// some ground points clamp to exactly 0.0 intensity - see data/README.md);
// kDenomFloor keeps the f(r)*cos(theta) DIVISOR away from zero for the same
// grazing points (cos(theta) is itself floored at 0.02 by the generator,
// but f(r)*cos(theta) can still be tiny at long range).
constexpr float kIntensityFloor = 1.0e-4f;
constexpr float kDenomFloor = 1.0e-4f;

// corrected_log_intensity - log(max(I / max(f(r)*cos(theta), floor), floor)).
// The ONE formula every per-point stage of this project's pipeline computes
// (kernels.cu's bin_and_accumulate_kernel, reference_cpu.cpp's twin, and
// main.cu's post-hoc diagnostics all call this SAME function - the shared,
// single-sourced arithmetic the independence ruling permits).
// Returns: log(g[ch] * R_surface) + noise-in-log-space, per THEORY.md "The
// math" - the quantity whose cross-channel, same-voxel DIFFERENCES estimate
// relative log-gains once R_surface cancels.
HD inline float corrected_log_intensity(float intensity, float f_r, float cos_theta)
{
    float denom = f_r * cos_theta;
    if (denom < kDenomFloor) denom = kDenomFloor;
    float corrected = intensity / denom;
    if (corrected < kIntensityFloor) corrected = kIntensityFloor;
    return logf(corrected);
}

// ===========================================================================
// SECTION 4 - the voxel grid (file header "VOXEL GRID"). kVoxelLeafM was
// chosen (and the round-half-up binning adopted) by MEASURING channel-graph
// connectivity on this project's committed scene while building it - see
// THEORY.md "Numerical considerations" for the swept comparison.
// ===========================================================================
constexpr float kVoxelLeafM = 0.5f;

// voxel_coord - round v/leaf to the NEAREST integer (round-half-up via
// floor(x+0.5)), NOT project 02.01's plain floor(v/leaf) - file header
// "VOXEL GRID" explains why this project needs the offset. Parameters: p
// (m, any sign), leaf (m, > 0). Returns: the voxel index along one axis.
HD inline int32_t voxel_coord(float p, float leaf)
{
    return static_cast<int32_t>(floorf(p / leaf + 0.5f));
}

// GridBounds - the dense voxel grid's extent, computed ONCE per scan from
// that scan's own point cloud (main.cu, host-only; a data-layout parameter,
// not part of either "algorithm" twin - shared with both GPU and CPU paths,
// exactly like 01.09's kW/kH problem geometry is shared). A DENSE grid
// (flat 3-D array), not project 02.01's spatial HASH table, is deliberate
// here: this project's scene is small and bounded (at most a few hundred
// occupied voxels), so a dense array sized to the data's own bounding box
// (with a small margin) is simpler AND avoids hash collisions entirely -
// project 02.01's hashing IS the right tool at LiDAR-scan point counts
// (10^5-10^6), which this project is far below (README "Limitations" states
// the scope cut explicitly, the same "brute force is honestly enough at
// this scale" choice project 02.18 makes for its own search kernels).
struct GridBounds {
    int32_t ix_min, iy_min, iz_min;   // voxel-index origin (any sign)
    int32_t nx, ny, nz;               // grid dimensions (>= 1 each)
    float leaf;                       // == kVoxelLeafM, carried for convenience
};

// flat_voxel_index - (ix,iy,iz) -> a single non-negative index into a dense
// [nx*ny*nz] array, or -1 if the coordinate falls outside `grid` (should
// never happen for a point that was used to COMPUTE `grid`'s bounds with
// margin - main.cu's grid-building step guards this explicitly).
HD inline int32_t flat_voxel_index(int32_t ix, int32_t iy, int32_t iz, const GridBounds& grid)
{
    const int32_t lx = ix - grid.ix_min, ly = iy - grid.iy_min, lz = iz - grid.iz_min;
    if (lx < 0 || ly < 0 || lz < 0 || lx >= grid.nx || ly >= grid.ny || lz >= grid.nz) return -1;
    return (lx * grid.ny + ly) * grid.nz + lz;
}

// ===========================================================================
// SECTION 5 - the least-squares GAIN MODEL (THEORY.md "The math" derives
// this in full; this is the implementation-facing summary).
//
// For a shared voxel v with channel set C_v (|C_v| = k_v >= 2) and observed
// per-(voxel,channel) mean corrected_log_intensity y_{v,c} = log(g_c) +
// log(R_v) + noise, PROFILING OUT the per-voxel nuisance log(R_v)
// (analytically: the optimal log(R_v) given x is the mean of (y_{v,c}-x_c)
// over c in C_v) reduces the joint least-squares problem to a pure function
// of the 16 unknown log-gains x_c. Writing r_{v,c} = y_{v,c} - ybar_v (the
// voxel's own already-known, already-centered residual - ybar_v is the mean
// of y_{v,c} over C_v), the reduced objective is
//
//     J(x) = sum_v sum_{c in C_v} (r_{v,c} - (x_c - xbar_v(x)))^2
//
// whose normal equations accumulate, PER SHARED VOXEL, the k_v x k_v
// "centering projector" P_v = I - (1/k_v) * ones(k_v,k_v) (embedded at rows/
// cols C_v of a 16x16 matrix) into A, and r_v (embedded the same way) into
// b: A = sum_v P_v, b = sum_v r_v, solved as A x = b. This is EXACTLY a
// weighted graph Laplacian over the "which channels share a voxel" graph:
// A's off-diagonal structure IS that graph's adjacency (channel_ls_accumulate
// below), and A's rank deficiency is that graph's number of connected
// components (standard graph-Laplacian theory) - the "observability as
// graph connectivity" statement README/THEORY.md make explicit. A channel
// that never shares ANY voxel with another channel contributes NOTHING to
// A's off-diagonal terms and has A_cc == 0 EXACTLY (not just small) - the
// signal solve_channel_gains (kernels.cu/reference_cpu.cpp, a SHARED
// host-only function per the 01.09 SECTION-5 precedent below) uses to flag
// it UNOBSERVABLE instead of solving for a meaningless value.
//
// GAUGE FREEDOM: adding a constant to every log(g_c) and subtracting it from
// every log(R_v) leaves every residual unchanged - A's null space always
// contains the all-ones vector (over the OBSERVABLE channels). Fixed by the
// mean-log-gain=0 convention: solve_channel_gains adds a small ridge term
// lambda/m * ones(m,m) (m = observable channel count) to the reduced system
// - since ones is in A's null space (by symmetry, orthogonal to A's range),
// this pins mean(x)=0 without perturbing anything else (THEORY.md "The
// math" derives why the ridge term is exactly equivalent to that
// constraint).
// ===========================================================================

// channel_ls_accumulate - ONE shared voxel's contribution to the 16x16
// normal-equations matrix A and 16-vector b (this section's derivation).
// HD + shared (file header "Why this header is HD"): a short, closed-form
// formula (not the accumulation LOOP over many voxels, which stays
// independently typed in kernels.cu/reference_cpu.cpp - the twin-
// independence ruling).
//
// Parameters:
//   chans      - [k] the k>=2 channel ids present in this voxel (any order).
//   y          - [k] this voxel's per-channel mean corrected_log_intensity,
//                SAME order as chans.
//   k          - channel count in this voxel (>= 2 - caller filters k<2).
//   A          - [16*16] IN/OUT, ACCUMULATED into (row-major, A[c*16+c2]).
//   b          - [16]    IN/OUT, ACCUMULATED into.
// Side effects: adds this voxel's P_v (into A) and r_v (into b). Complexity:
// O(k^2) (k <= kNumBeams, so at most 256 term-writes per voxel).
HD inline void channel_ls_accumulate(const int32_t* chans, const float* y, int k,
                                      float* A, float* b)
{
    // ybar_v = mean_{c in C_v} y_{v,c} (this voxel's own material/gain
    // baseline - see this section's derivation for why it need not, and
    // must not, be known in advance).
    float ybar = 0.0f;
    for (int i = 0; i < k; ++i) ybar += y[i];
    ybar /= static_cast<float>(k);

    const float inv_k = 1.0f / static_cast<float>(k);
    for (int i = 0; i < k; ++i) {
        const int32_t ci = chans[i];
        const float r_i = y[i] - ybar;             // r_{v,ci}, already centered (sums to ~0 over C_v)
        b[ci] += r_i;                               // this voxel's contribution to b (this section)
        for (int j = 0; j < k; ++j) {
            const int32_t cj = chans[j];
            // P_v[i][j] = (i==j ? 1 - 1/k : -1/k) (the centering projector).
            const float pij = (i == j) ? (1.0f - inv_k) : (-inv_k);
            A[ci * kNumBeams + cj] += pij;
        }
    }
}

// ===========================================================================
// SECTION 6 - point-record layout (main.cu owns loading; every kernel below
// operates on ONE scan's n points at a time - main.cu calls the whole
// pipeline twice, once per committed scan).
//     channel[i]   : int32, [0, kNumBeams).
//     xyz[i*3+0..2]: meters, SENSOR frame (sensor at the origin - file
//                    header "SCENE GEOMETRY"). Interleaved, matching this
//                    repo's PointCloud convention (02.01/02.06/02.13/02.18/
//                    SYSTEM_DESIGN.md section 3.6).
//     intensity[i] : the RAW measured intensity (unitless, >= 0) - the
//                    forward model's output (file header).
// GROUND TRUTH - surf_id[i]/R_true[i], loaded alongside points, used ONLY by
// main.cu's gates/artifacts (file header "GROUND TRUTH").
// ===========================================================================

#ifdef __CUDACC__  // ---- device-aware section: only nvcc sees this ---------

// point_features_kernel - one thread per point: this project's SECTION-3
// forward-model inversion + SECTION-4 voxel binning, fused into one MAP
// (every point is independent - no communication between threads; the same
// "map" pattern as 01.09's stack_mean_kernel's per-pixel outer loop).
//
//   n            : points in this scan.
//   xyz          : [n*3] interleaved, DEVICE pointer, sensor frame, meters.
//   intensity    : [n] DEVICE pointer, raw measured intensity.
//   grid         : this scan's voxel-grid bounds (main.cu computes once,
//                  host-side, from this SAME point cloud - SECTION 4).
//   log_intensity: [n] OUT - corrected_log_intensity() per point (SECTION 3).
//   voxel_idx    : [n] OUT - flat_voxel_index() per point (SECTION 4); -1 if
//                  (should not happen) outside `grid`'s bounds.
__global__ void point_features_kernel(int n,
                                       const float* __restrict__ xyz,
                                       const float* __restrict__ intensity,
                                       GridBounds grid,
                                       float* __restrict__ log_intensity,
                                       int32_t* __restrict__ voxel_idx);

// bin_accumulate_kernel - one thread per point: SCATTER-REDUCE (the
// histogram pattern 01.09's radial_bin_kernel/02.18's DROR neighbor-count
// use, here 2-D keyed by (voxel, channel) instead of 1-D by radius).
//
//   channel   : [n] DEVICE pointer, this point's beam id [0,kNumBeams).
//   log_intensity, voxel_idx : [n] DEVICE pointers, point_features_kernel's
//                  output (this project's stage-1 -> stage-2 hand-off).
//   numVoxels : grid.nx*grid.ny*grid.nz (the dense array's element count).
//   sum_log   : [numVoxels*kNumBeams] OUT, atomicAdd-accumulated - sum of
//               log_intensity over every point of that (voxel,channel).
//   count     : [numVoxels*kNumBeams] OUT, atomicAdd-accumulated - how many
//               points contributed.
//   voxel_family : [numVoxels] OUT - this voxel's NormalFamily (every point
//               in a voxel shares one family by construction on this
//               project's scene - a RACY but BENIGN plain store: every
//               possible writer stores the IDENTICAL value, so no atomic is
//               needed - documented explicitly, not an oversight).
__global__ void bin_accumulate_kernel(int n,
                                       const int32_t* __restrict__ channel,
                                       const float* __restrict__ xyz,
                                       const float* __restrict__ log_intensity,
                                       const int32_t* __restrict__ voxel_idx,
                                       int numVoxels,
                                       float* __restrict__ sum_log,
                                       int32_t* __restrict__ count,
                                       int32_t* __restrict__ voxel_family);

// assemble_ls_kernel - one thread per VOXEL (numVoxels threads, most exit
// immediately - fewer than kNumBeams/2 typically qualify as "shared"): for
// voxels with >= 2 channels present, SECTION 5's channel_ls_accumulate(),
// atomicAdd'ed into the GLOBAL 16x16 A / 16 b (small, contended only by the
// handful of shared voxels - the same "small global atomic destination"
// pattern 01.09's roi_mean_reduce_kernel uses for its single double
// accumulator, generalized to 256+16 destinations).
//
//   sum_log, count : [numVoxels*kNumBeams] IN, bin_accumulate_kernel's output.
//   A : [kNumBeams*kNumBeams] OUT, atomicAdd-accumulated (row-major).
//   b : [kNumBeams] OUT, atomicAdd-accumulated.
//   shared_voxel_count : [1] OUT, atomicAdd-accumulated - how many voxels
//               qualified (>= 2 channels) - a diagnostic main.cu reports.
__global__ void assemble_ls_kernel(int numVoxels,
                                    const float* __restrict__ sum_log,
                                    const int32_t* __restrict__ count,
                                    float* __restrict__ A,
                                    float* __restrict__ b,
                                    int32_t* __restrict__ shared_voxel_count);

// apply_gain_kernel - the "reason to exist" MAP: divide out the RECOVERED
// per-channel gain from the raw intensity (main.cu's "AFTER" artifacts and
// the consistency_improvement gate). Channels flagged unobservable
// (observable[ch]==0) pass through with gain 1.0 (documented: "no
// correction applied", never a silent guess).
//   raw_intensity : [n] DEVICE pointer.
//   channel       : [n] DEVICE pointer.
//   gain          : [kNumBeams] DEVICE pointer, recovered (or 1.0) per channel.
//   corrected     : [n] OUT.
__global__ void apply_gain_kernel(int n,
                                   const float* __restrict__ raw_intensity,
                                   const int32_t* __restrict__ channel,
                                   const float* __restrict__ gain,
                                   float* __restrict__ corrected);

#endif // __CUDACC__ --------------------------------------------------------

// ---------------------------------------------------------------------------
// Host launch wrappers (definitions in kernels.cu). Each owns its grid/block
// math + the mandatory post-launch error check (CLAUDE.md paragraph 6 rule 7).
// ---------------------------------------------------------------------------
void launch_point_features(int n, const float* d_xyz, const float* d_intensity,
                            GridBounds grid, float* d_log_intensity, int32_t* d_voxel_idx);
void launch_bin_accumulate(int n, const int32_t* d_channel, const float* d_xyz,
                            const float* d_log_intensity, const int32_t* d_voxel_idx,
                            int numVoxels, float* d_sum_log, int32_t* d_count, int32_t* d_voxel_family);
void launch_assemble_ls(int numVoxels, const float* d_sum_log, const int32_t* d_count,
                         float* d_A, float* d_b, int32_t* d_shared_voxel_count);
void launch_apply_gain(int n, const float* d_raw_intensity, const int32_t* d_channel,
                        const float* d_gain, float* d_corrected);

// ===========================================================================
// SECTION 7 - CPU references (reference_cpu.cpp) - the correctness-oracle
// twins. All pointers below are HOST pointers. See reference_cpu.cpp's file
// header for the independence ruling each of these follows.
// ===========================================================================
void point_features_cpu(int n, const float* xyz, const float* intensity,
                         const GridBounds& grid, float* log_intensity, int32_t* voxel_idx);
void bin_accumulate_cpu(int n, const int32_t* channel, const float* log_intensity,
                         const int32_t* voxel_idx, int numVoxels,
                         double* sum_log_d, int32_t* count);
void assemble_ls_cpu(int numVoxels, const double* sum_log_d, const int32_t* count,
                      double* A_d, double* b_d, int32_t* shared_voxel_count);
void apply_gain_cpu(int n, const float* raw_intensity, const int32_t* channel,
                     const float* gain, float* corrected);

// ===========================================================================
// SECTION 8 - the SHARED, HOST-ONLY 16x16 solve (the 01.09 SECTION-5
// precedent: a dense micro-solve this small has no meaningful GPU mapping -
// project 33.01 is where GPU-side BATCHED small solves earn their keep, at a
// scale where batching pays for itself; a single 16x16 system does not).
// Because this solve is SHARED (called once, from the already GPU-vs-CPU-
// verified A/b), the twin comparison is BLIND to bugs inside it - which is
// why main.cu's gain_recovery gate checks the OUTPUT against independent
// ground truth (never a second implementation of this function) - SECTION 5
// derives the algorithm; this is Gaussian elimination with partial
// pivoting over the OBSERVABLE-channel subsystem, gauge-fixed by a small
// ridge term.
//
// Parameters:
//   A, b        - [kNumBeams*kNumBeams] / [kNumBeams] the assembled normal
//                 equations (SECTION 5), IN.
//   out_log_gain- [kNumBeams] OUT - recovered log(g_c) for observable
//                 channels; UNCHANGED (caller should treat as invalid) for
//                 unobservable ones - always check out_observable first.
//   out_observable - [kNumBeams] OUT - 1 if channel c belongs to the
//                 LARGEST connected component of the shared-voxel graph
//                 (A[c][c] > kObservableEps AND reachable, via nonzero
//                 off-diagonal entries, from that dominant component), 0
//                 otherwise. A channel that shares a voxel with ONLY a
//                 small, otherwise-isolated cluster of other channels is
//                 flagged unobservable too - comparing its gain against a
//                 different, unconnected cluster is meaningless (a second,
//                 unresolved gauge freedom, not the same case as true
//                 isolation, but requiring the same honest flag - see
//                 kernels.cu's solve_channel_gains for the connected-
//                 components argument this refines SECTION 5's original,
//                 single-component-assumed statement into).
// Returns: the number of observable channels solved (the largest
// component's size).
// ===========================================================================
constexpr float kObservableEps = 1.0e-6f;
constexpr float kGaugeLambda = 1.0e-2f;

int solve_channel_gains(const float* A, const float* b,
                         float* out_log_gain, int32_t* out_observable);

#endif // PROJECT_KERNELS_CUH
