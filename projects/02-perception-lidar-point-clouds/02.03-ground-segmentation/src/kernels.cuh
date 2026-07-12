// ===========================================================================
// kernels.cuh — interface for project 02.03
//               Ground segmentation: RANSAC plane fit; Patchwork++-style
//               GPU port (catalog bullet, BUNDLED per CLAUDE.md §2)
//
// Role in the project
// -------------------
// The CONTRACT shared by main.cu (orchestration + gates + artifacts),
// kernels.cu (the GPU kernels), and reference_cpu.cpp (the independent CPU
// oracle twins). Everything all three must agree on — point-cloud layout,
// plane representation, the RANSAC hypothesis-seeding formula, the CZM
// (concentric-zone model) patch geometry — is defined HERE, once, following
// 02.01's "single-sourced data-layout contract" precedent (see that
// project's kernels.cuh header for the pattern this one reuses).
//
// POINT CLOUD LAYOUT — float* xyz, interleaved, meters, LiDAR "sensor" frame
// (origin at the sensor, +x forward, +z up): xyz[i*3+0..2] = x,y,z. Same
// convention as 02.01 and 02.06 (this domain's flagships) — this project's
// output (a per-point ground/not-ground label) is exactly the kind of mask
// 02.04 (Euclidean clustering) and 14.02 (traversability costmaps) expect
// as an upstream input (README "System context" names this hand-off).
//
// TWO MILESTONES, ONE SCENE — the bundled project's teaching core
// -----------------------------------------------------------------
// MILESTONE 1 — GPU RANSAC plane fit. Fits ONE infinite plane to the WHOLE
// point cloud by sampling random 3-point hypotheses, counting inliers for
// each in parallel, refining the best one by least squares. Fast, simple,
// and correct exactly where its assumption holds: the ground is ONE plane.
//
// MILESTONE 2 — Patchwork++-style concentric-zone model (CZM). Partitions
// the scan into many small LOCAL patches (a polar grid centered on the
// sensor: zones by range, rings within a zone, sectors within a ring) and
// fits an INDEPENDENT small plane per patch, with a simple region-growing
// rule (a patch's seed height is predicted from its inward neighbor's
// fitted plane) that lets the recovered "ground" surface bend — handling
// the ramp and the two-level plateau that defeat a single global plane.
// This is an honest, REDUCED-SCOPE teaching version of Patchwork++
// (Lee et al., IEEE RA-L 2022): the real system adds adaptive ground
// likelihood estimation, temporal reversion, and a more elaborate seed
// selection than the height-margin rule used here — see THEORY.md "Where
// this sits in the real world" for exactly what is scoped out and why.
//
// THE DESIGNED SCENE (see ../scripts/make_synthetic.py) exists to make the
// two milestones' behavior DIFFER, measurably: a flat segment (RANSAC's
// home turf), an 8-degree ramp, and an elevated plateau — single-plane
// RANSAC can fit only the (majority) flat segment and must reject the
// ramp/plateau's true ground as "not ground"; the CZM's region-growing
// design should recover it. main.cu's `single_plane_failure` and
// `czm_recovery` gates assert this designed contrast numerically.
//
// Why this header is CUDA-qualifier-free where possible (02.01's precedent)
// ---------------------------------------------------------------------------
// Pure math/bookkeeping functions below (RNG stepping, triplet picking,
// plane fitting, point-to-plane distance, the CZM patch-id formula) are
// declared as PLAIN inline C++ — no __host__/__device__ — so they compile
// under BOTH nvcc (main.cu, and reference_cpu.cpp's shared-formula calls)
// and cl.exe (reference_cpu.cpp otherwise). Being unqualified, they are
// HOST-only under nvcc's rules and CANNOT be called from inside a
// __global__ kernel; kernels.cu therefore carries its OWN literal
// __device__ transcription of each one, clearly cross-referenced in
// comments at both copies. The gate that catches any drift between a
// header copy and its device transcription is named at each use site
// below (mirroring 02.01's VERIFY(keys) pattern) — e.g. VERIFY(patch_ids)
// and VERIFY(hypotheses) in main.cu.
//
// The one genuinely reusable numerical routine — the symmetric 3x3
// eigensolver used by BOTH milestones' plane fits — is likewise duplicated
// exactly this way: an unqualified `jacobi_eigen_3x3` here (used by
// reference_cpu.cpp), and a `__device__` transcription `jacobi_eigen_3x3_dev`
// in kernels.cu (used inside the RANSAC-refinement and CZM-fit kernels).
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cstdint>   // uint32_t etc. — exact-width integers for the RNG
#include <cmath>     // std::sqrt/std::fabs/std::atan2 — identical overloads to cl.exe and nvcc's host pass

// ===========================================================================
// Shared launch-configuration constant (same reasoning as every project in
// this repo — see e.g. 02.01/08.01's kernels.cuh: a warp multiple, good
// default occupancy on sm_75..sm_89).
// ===========================================================================
constexpr int kThreadsPerBlock = 256;

// blocks_for — integer ceiling division: how many `threads`-wide blocks
// cover `count` independent problems (the same idiom 02.01/02.06/08.01 use).
inline int blocks_for(int count, int threads)
{
    return (count + threads - 1) / threads;
}

// ===========================================================================
// Scene geometry constants — SHARED with scripts/make_synthetic.py (same
// values, documented there too; the LEAF_M / kVoxelLeafM precedent from
// 02.01 for why a Python generator and a C++ header carry the same numeric
// constant by convention rather than a machine-checked cross-file assert:
// these are geometry-DESIGN constants, not safety-critical data-format
// fields, so a documented match is sufficient — main.cu does not read them
// back from the binary file).
//
// The scene (full account in scripts/make_synthetic.py's module docstring):
// a flat ground segment everywhere EXCEPT a forward corridor between
// RAMP_X_START and PLATEAU_X_END within +-RAMP_Y_HALF_WIDTH of the sensor's
// forward axis; inside that corridor, ground RAMPS UP at RAMP_SLOPE_DEG over
// RAMP_LENGTH, then continues as a flat PLATEAU at the raised height. This
// three-level ground (flat / ramp / plateau) is what a single RANSAC plane
// cannot represent but the patch-local CZM can.
// ===========================================================================
constexpr float kSceneRampXStartM     = 4.0f;   // corridor: ramp begins at this forward range (m)
constexpr float kSceneRampLengthM     = 4.0f;   // ramp run length (m); ends at kSceneRampXStartM + this
constexpr float kSceneRampYHalfWidthM = 3.5f;   // corridor half-width (m); |y| beyond this is flat ground
constexpr float kSceneRampSlopeDeg    = 8.0f;   // ramp grade, degrees from horizontal
constexpr float kSceneSensorHeightM   = 1.5f;   // LiDAR mount height above the base floor (m) — main.cu's flat-plane truth check

// ===========================================================================
// MILESTONE 1 — RANSAC plane fit: shared config, plane representation, and
// the single-sourced math (RNG, triplet picking, plane-from-3-points,
// point-to-plane distance).
// ===========================================================================

// Number of random-triplet hypotheses evaluated per RANSAC run. Both the
// full-scene run and the flat-only run (README "The algorithm in brief")
// use this same K; main.cu's `ransac_formula` gate checks it against the
// classical k = log(1-p)/log(1-w^3) requirement for the MEASURED inlier
// ratio (THEORY.md "The math" derives the formula).
constexpr int   kRansacK                 = 1024;
constexpr float kRansacInlierThresholdM  = 0.08f;  // |point-to-plane distance| <= this => inlier (8 cm)
constexpr int   kRansacMaxTripletAttempts = 8;     // degenerate-triplet retries per hypothesis (see below)
constexpr float kRansacMinCrossNormM2    = 0.02f;  // reject a triplet whose |e1 x e2| (= 2*triangle area, m^2) is below this
constexpr double kRansacTargetSuccessProb = 0.999;  // p in k = log(1-p)/log(1-w^3) — README/THEORY "iteration count"

// A fitted plane: unit normal (nx,ny,nz) and offset d such that, for any
// point p on the plane, nx*px + ny*py + nz*pz + d == 0. The normal is
// oriented so nz >= 0 ("pointing up") purely so uprightness comparisons
// (angle vs +z) read naturally — inlier counting itself is sign-agnostic
// (it uses |distance|), so this orientation convention affects nothing
// about correctness, only readability of printed angles.
struct PlaneModel {
    float nx = 0.0f, ny = 0.0f, nz = 1.0f, d = 0.0f;
};

// ---------------------------------------------------------------------------
// xorshift32_step — one step of Marsaglia's 32-bit xorshift PRNG (the SAME
// three-shift/three-XOR core used repo-wide: see 02.01's make_synthetic.py
// module docstring, and 08.01/11.01's device generators). Full 2^32-1 period
// for any nonzero seed; degenerate (stays 0 forever) at seed 0, which is why
// every seeding function below guards against a zero result.
// ---------------------------------------------------------------------------
inline uint32_t xorshift32_step(uint32_t state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

// ---------------------------------------------------------------------------
// hypothesis_seed — COUNTER-BASED per-hypothesis RNG seed: turns (a global
// run seed, a hypothesis index k, a retry attempt) into one xorshift32
// starting state, with NO dependency on any other hypothesis's draws.
//
// Why counter-based, not one shared sequential stream (the teaching point):
// K=1024 hypotheses are generated by K INDEPENDENT GPU threads/blocks in an
// order the scheduler decides, not index order. A single shared "next()"
// stream would need a serialized critical section (an atomic RNG cursor) —
// exactly the kind of false dependency GPU parallel code must avoid. Instead
// each hypothesis derives its OWN starting state from its own index via a
// cheap multiplicative mix (the constant 0x9E3779B9 is the golden-ratio
// fractional constant popularized by splitmix/Fibonacci hashing — it has no
// small common factors with 2^32, so nearby k values scatter widely instead
// of producing near-identical seeds). Two xorshift32_step() mixing rounds
// then erase any remaining structure from the multiply. This is embarrassingly
// parallel BY CONSTRUCTION: thread k never needs to know what thread k-1 did.
// ---------------------------------------------------------------------------
inline uint32_t hypothesis_seed(uint32_t global_seed, int k, int attempt)
{
    uint32_t s = global_seed ^ (0x9E3779B9u * static_cast<uint32_t>(k * 8 + attempt + 1));
    if (s == 0u) s = 1u;              // xorshift32 is degenerate at exactly 0
    s = xorshift32_step(s);
    s = xorshift32_step(s);
    return s;
}

// ---------------------------------------------------------------------------
// pick_triplet_indices — draw 3 point indices in [0,n) from a single
// xorshift32 stream seeded by `seed`. Indices MAY collide (i0==i1, etc.);
// that is not filtered here because plane_from_triplet's degenerate check
// (near-zero cross product) already rejects a coincident/collinear triplet
// as a special case of "too small a triangle" — one check instead of two.
// ---------------------------------------------------------------------------
inline void pick_triplet_indices(uint32_t seed, int n, int& i0, int& i1, int& i2)
{
    uint32_t s = seed;
    s = xorshift32_step(s); i0 = static_cast<int>(s % static_cast<uint32_t>(n));
    s = xorshift32_step(s); i1 = static_cast<int>(s % static_cast<uint32_t>(n));
    s = xorshift32_step(s); i2 = static_cast<int>(s % static_cast<uint32_t>(n));
}

// ---------------------------------------------------------------------------
// plane_from_triplet — fit the unique plane through 3 points via the cross
// product of two edge vectors (the textbook normal-from-3-points formula:
// n = (p1-p0) x (p2-p0), then normalize). Returns false — a DEGENERATE
// triplet — when the 3 points are (nearly) collinear or coincident, which
// makes |n| (twice the triangle's area) vanish and the true normal
// direction numerically meaningless; callers must retry with a different
// triplet rather than accept a garbage plane (README/THEORY "numerical
// considerations": near-collinear triplets are the classic RANSAC-on-planes
// failure mode, most likely on this scene's SPARSE-looking far-range rings).
//
// Parameters: p0,p1,p2 — each a 3-element {x,y,z} array, meters.
//             out      — OUT: the fitted PlaneModel (oriented nz>=0), valid
//                        only when this function returns true.
// ---------------------------------------------------------------------------
inline bool plane_from_triplet(const float p0[3], const float p1[3], const float p2[3], PlaneModel& out)
{
    const float e1x = p1[0] - p0[0], e1y = p1[1] - p0[1], e1z = p1[2] - p0[2];
    const float e2x = p2[0] - p0[0], e2y = p2[1] - p0[1], e2z = p2[2] - p0[2];
    // Cross product e1 x e2 — its magnitude is twice the triangle's area, so
    // a tiny magnitude means "collinear or coincident", the degenerate case.
    const float cx = e1y * e2z - e1z * e2y;
    const float cy = e1z * e2x - e1x * e2z;
    const float cz = e1x * e2y - e1y * e2x;
    const float norm2 = cx * cx + cy * cy + cz * cz;      // |cross|^2 (units m^4)
    if (norm2 < kRansacMinCrossNormM2 * kRansacMinCrossNormM2) return false;  // degenerate: reject
    const float inv_norm = 1.0f / std::sqrt(norm2);
    float nx = cx * inv_norm, ny = cy * inv_norm, nz = cz * inv_norm;
    if (nz < 0.0f) { nx = -nx; ny = -ny; nz = -nz; }      // orient "up" (see PlaneModel comment)
    out.nx = nx; out.ny = ny; out.nz = nz;
    out.d  = -(nx * p0[0] + ny * p0[1] + nz * p0[2]);     // solve n.p0 + d = 0 for d
    return true;
}

// point_plane_signed_distance — n.p + d for a plane in PlaneModel form and a
// point p[3]. Positive on the side the normal points toward. Callers that
// only need "is this an inlier" take std::fabs() of the result.
inline float point_plane_signed_distance(const PlaneModel& pl, const float p[3])
{
    return pl.nx * p[0] + pl.ny * p[1] + pl.nz * p[2] + pl.d;
}

// select_best_hypothesis — argmax over K inlier counts, skipping hypotheses
// marked invalid (all triplet-generation attempts were degenerate). Ties
// break toward the LOWEST index (first occurrence wins under an ascending
// scan) — a fixed, deterministic rule so main.cu's GPU path and
// reference_cpu.cpp's independent twin, both scanning k=0..K-1 ascending,
// agree exactly even if two hypotheses happen to tie. Host-only: K=1024 is
// far too small to be worth a reduction kernel (the "know when NOT to
// parallelize" lesson — contrast with the genuinely-parallel evaluation
// kernel this selection consumes the output of).
inline int select_best_hypothesis(const int* hyp_inlier_count, const uint8_t* hyp_valid, int k)
{
    int best = -1;
    int best_count = -1;
    for (int i = 0; i < k; ++i) {
        if (!hyp_valid[i]) continue;
        if (hyp_inlier_count[i] > best_count) { best_count = hyp_inlier_count[i]; best = i; }
    }
    return best;
}

// ===========================================================================
// The shared symmetric-3x3 eigensolver — used by BOTH milestones' plane
// fits (RANSAC's least-squares refinement on inliers, and each CZM patch's
// PCA fit). Cyclic Jacobi rotation (Jacobi 1846; see any numerical linear
// algebra text, e.g. Golub & Van Loan "Matrix Computations" §8.4): for a
// FIXED number of "sweeps" (full passes over the 3 off-diagonal entries),
// apply a rotation that zeroes one off-diagonal entry at a time. For a 3x3
// symmetric matrix this converges to machine precision in a handful of
// sweeps — we run a fixed kSweeps (no convergence check needed, which also
// keeps this GPU-friendly: every thread that calls it does the SAME fixed
// amount of work, no data-dependent branching to cause warp divergence).
// This is the "small closed-form/iterative solve beats a heavyweight
// library call" choice 33.01 (batched small-matrix linalg, this repo's
// foundations flagship) teaches for exactly this matrix size — see that
// project for the LU/Cholesky/eigen alternatives at larger sizes.
//
// Parameters: a[6] — the UPPER TRIANGLE of the symmetric 3x3 input, packed
//             (a00,a01,a02,a11,a12,a22) — the covariance matrix's 6 unique
//             entries (caller's responsibility to build; see kernels.cu /
//             reference_cpu.cpp call sites).
//             eigenvalues[3] — OUT, ascending order.
//             eigenvectors[3][3] — OUT, eigenvectors[i] is the UNIT
//             eigenvector for eigenvalues[i], i.e. eigenvectors[0] is the
//             SMALLEST-eigenvalue eigenvector — the plane normal both
//             milestones want (THEORY.md "The math" derives why the
//             smallest-eigenvalue eigenvector of the point covariance is
//             the least-squares plane normal).
// ---------------------------------------------------------------------------
inline void jacobi_eigen_3x3(const float a_in[6], float eigenvalues[3], float eigenvectors[3][3])
{
    // Working copy as a full 3x3 (symmetric) matrix — simpler indexing below
    // than re-deriving the packed upper-triangle offsets at every access.
    float A[3][3] = {
        { a_in[0], a_in[1], a_in[2] },
        { a_in[1], a_in[3], a_in[4] },
        { a_in[2], a_in[4], a_in[5] },
    };
    // V accumulates the product of all Jacobi rotations -> converges to the
    // eigenvector matrix (columns = eigenvectors), starting from identity.
    float V[3][3] = { {1,0,0}, {0,1,0}, {0,0,1} };

    const int kSweeps = 8;  // 3x3 Jacobi converges to float precision well within 8 sweeps (measured; see THEORY.md)
    for (int sweep = 0; sweep < kSweeps; ++sweep) {
        // Zero each of the 3 off-diagonal pairs in turn: (0,1), (0,2), (1,2).
        const int pairs[3][2] = { {0,1}, {0,2}, {1,2} };
        for (int pi = 0; pi < 3; ++pi) {
            const int p = pairs[pi][0], q = pairs[pi][1];
            const float apq = A[p][q];
            if (std::fabs(apq) < 1.0e-12f) continue;  // already ~zero: skip the rotation (nothing to do)
            // Classic Jacobi rotation angle: cot(2*theta) = (aqq-app)/(2*apq).
            const float theta = (A[q][q] - A[p][p]) / (2.0f * apq);
            const float t = (theta >= 0.0f ? 1.0f : -1.0f) /
                            (std::fabs(theta) + std::sqrt(theta * theta + 1.0f));  // numerically stable tan(rotation)
            const float c = 1.0f / std::sqrt(t * t + 1.0f);
            const float s = t * c;
            // Apply the rotation to A (both rows/cols p,q) and accumulate into V.
            const float app = A[p][p], aqq = A[q][q];
            A[p][p] = app - t * apq;
            A[q][q] = aqq + t * apq;
            A[p][q] = 0.0f; A[q][p] = 0.0f;
            for (int i = 0; i < 3; ++i) {
                if (i != p && i != q) {
                    const float aip = A[i][p], aiq = A[i][q];
                    A[i][p] = c * aip - s * aiq; A[p][i] = A[i][p];
                    A[i][q] = s * aip + c * aiq; A[q][i] = A[i][q];
                }
                const float vip = V[i][p], viq = V[i][q];
                V[i][p] = c * vip - s * viq;
                V[i][q] = s * vip + c * viq;
            }
        }
    }

    // Extract eigenvalues (now on the diagonal of A) with their eigenvectors
    // (the matching COLUMN of V), then insertion-sort the 3 pairs ascending
    // — trivial for n=3, no need for a real sort routine.
    float ev[3] = { A[0][0], A[1][1], A[2][2] };
    float vec[3][3] = {
        { V[0][0], V[1][0], V[2][0] },
        { V[0][1], V[1][1], V[2][1] },
        { V[0][2], V[1][2], V[2][2] },
    };
    for (int i = 0; i < 2; ++i) {
        int min_j = i;
        for (int j = i + 1; j < 3; ++j) if (ev[j] < ev[min_j]) min_j = j;
        if (min_j != i) {
            const float tmp_ev = ev[i]; ev[i] = ev[min_j]; ev[min_j] = tmp_ev;
            for (int c2 = 0; c2 < 3; ++c2) { const float t2 = vec[i][c2]; vec[i][c2] = vec[min_j][c2]; vec[min_j][c2] = t2; }
        }
    }
    for (int i = 0; i < 3; ++i) {
        eigenvalues[i] = ev[i];
        eigenvectors[i][0] = vec[i][0]; eigenvectors[i][1] = vec[i][1]; eigenvectors[i][2] = vec[i][2];
    }
}

// CovAccum9 — the 9 running sums (+count) a least-squares plane fit needs:
// sum of x,y,z (for the mean) and sum of the 6 unique products xx,xy,xz,
// yy,yz,zz (for the covariance). Both milestones' plane fits reduce their
// inlier/patch points to exactly this struct before calling the eigensolver
// (see fit_plane_from_cov_accum below).
struct CovAccum9 {
    float sx = 0.0f, sy = 0.0f, sz = 0.0f;
    float sxx = 0.0f, sxy = 0.0f, sxz = 0.0f, syy = 0.0f, syz = 0.0f, szz = 0.0f;
    unsigned int count = 0;
};

// ---------------------------------------------------------------------------
// fit_plane_from_cov_accum — turn a CovAccum9 into a least-squares PlaneModel
// via mean + covariance + smallest-eigenvector normal (THEORY.md "The math"
// derives why this minimizes the sum of squared PERPENDICULAR distances,
// unlike a naive z = f(x,y) height-field fit which minimizes VERTICAL
// distance and is biased on a tilted patch like the ramp).
//
// Returns false (leaving `out` untouched) when count < 3 — too few points
// for a well-posed plane fit; callers must check.
// ---------------------------------------------------------------------------
inline bool fit_plane_from_cov_accum(const CovAccum9& acc, PlaneModel& out)
{
    if (acc.count < 3u) return false;
    const float inv_n = 1.0f / static_cast<float>(acc.count);
    const float mx = acc.sx * inv_n, my = acc.sy * inv_n, mz = acc.sz * inv_n;
    // Covariance = E[pp^T] - mean.mean^T (the standard one-pass formula —
    // THEORY.md "Numerical considerations" discusses its conditioning on
    // very thin/near-collinear patches vs. the safer two-pass alternative).
    const float cxx = acc.sxx * inv_n - mx * mx;
    const float cxy = acc.sxy * inv_n - mx * my;
    const float cxz = acc.sxz * inv_n - mx * mz;
    const float cyy = acc.syy * inv_n - my * my;
    const float cyz = acc.syz * inv_n - my * mz;
    const float czz = acc.szz * inv_n - mz * mz;
    const float packed[6] = { cxx, cxy, cxz, cyy, cyz, czz };
    float eigenvalues[3]; float eigenvectors[3][3];
    jacobi_eigen_3x3(packed, eigenvalues, eigenvectors);
    // eigenvectors[0] is the SMALLEST-eigenvalue eigenvector (ascending
    // order guaranteed by jacobi_eigen_3x3) — the least-squares plane normal.
    float nx = eigenvectors[0][0], ny = eigenvectors[0][1], nz = eigenvectors[0][2];
    if (nz < 0.0f) { nx = -nx; ny = -ny; nz = -nz; }   // orient "up", matching plane_from_triplet's convention
    out.nx = nx; out.ny = ny; out.nz = nz;
    out.d  = -(nx * mx + ny * my + nz * mz);           // plane passes through the mean
    return true;
}

// ===========================================================================
// MILESTONE 2 — Patchwork++-style concentric-zone model (CZM): shared
// config and the patch-id formula.
//
// The CZM partitions the scan into a POLAR grid centered on the sensor:
//   ZONES   — kCzmNumZones concentric range bands (near to far).
//   RINGS   — each zone is split into kCzmRingsPerZone radial sub-bands.
//   SECTORS — each ring is split into a PER-ZONE sector count, coarser
//             (fewer, wider sectors) in far zones. Why: LiDAR point density
//             falls off ~1/r^2 with range (derived in 02.01's THEORY.md,
//             cited here) — a near zone crammed into few, wide sectors
//             would each contain thousands of points spanning many meters
//             of azimuth (a poor local-planarity assumption AND a slow
//             per-patch reduction); a far zone split as finely as the near
//             zone would starve most patches of the >= kCzmMinPatchPoints
//             a plane fit needs. Density-adaptive sector counts (the real
//             Patchwork++'s design too) keep patch POINT COUNTS roughly
//             comparable across the whole scan.
// COLUMN  — one (zone,sector) pair, containing kCzmRingsPerZone patches
//             (ring 0 = near half of the zone, ring 1 = far half). Patches
//             within a column are processed ring 0 -> ring 1 IN ORDER (see
//             kernels.cu's czm_fit_and_classify_kernel): ring 1's seed
//             height is predicted from ring 0's fitted plane when ring 0
//             passed its tests — the region-growing rule that lets the
//             recovered ground follow the ramp instead of breaking at
//             every patch boundary (THEORY.md "The algorithm").
// PATCH   — one (column, ring) cell; patch_id = column_index*kCzmRingsPerZone
//             + ring, so points sort into ring-then-column-major order and
//             a column's two patches land at ADJACENT patch ids (used by
//             the fit kernel to find "my column's other ring" trivially).
// ===========================================================================
constexpr int kCzmNumZones = 4;
// Zone boundaries in range r = sqrt(x^2+y^2) from the sensor, meters. 5
// edges bound 4 zones: [0.5,4), [4,8), [8,14), [14,20). Points with
// r < 0.5 m (likely sensor-mount self-returns) or r >= 20 m (this scene's
// max range) get patch_id = -1 — outside the CZM footprint entirely,
// classified non-ground by definition (real Patchwork++ does the same: it
// only reasons about ground WITHIN its configured range).
constexpr float kCzmZoneEdgesM[kCzmNumZones + 1] = { 0.5f, 4.0f, 8.0f, 14.0f, 20.0f };
// Per-zone sector counts, near to far — see the density-adaptive rationale
// above. Sum = 32+24+16+8 = 80 columns; x kCzmRingsPerZone(2) = 160 patches.
constexpr int kCzmZoneSectors[kCzmNumZones] = { 32, 24, 16, 8 };
constexpr int kCzmRingsPerZone = 2;
constexpr int kCzmNumColumns = 32 + 24 + 16 + 8;              // = 80
constexpr int kCzmNumPatches = kCzmNumColumns * kCzmRingsPerZone;  // = 160

constexpr float kCzmSeedHeightMarginM = 0.20f;  // ring-0 (no prior) seed rule: points within [min_z, min_z+margin]
constexpr float kCzmHeightCarryBandM  = 0.30f;  // ring-1 (prior available) seed rule: points within +-band of predicted height
constexpr float kCzmUprightMaxDeg     = 30.0f;  // reject a patch plane tilted more than this from vertical (must clear the 8deg ramp with margin)
constexpr float kCzmFlatnessMaxRmsM   = 0.06f;  // reject a patch whose seed-point RMS residual to its own fitted plane exceeds this
constexpr float kCzmClassifyDistM     = 0.05f;  // final per-point |distance to patch plane| <= this => ground;
                                                 // tuned tighter than RANSAC's 0.08 m threshold (measured: a
                                                 // looser classify band disproportionately catches the base
                                                 // RIM of standing obstacles -- their lowest points sit only
                                                 // centimeters above true ground by construction, an inherent
                                                 // ambiguity real systems tune this exact parameter to manage;
                                                 // see PRACTICE.md "the curb problem")
constexpr unsigned int kCzmMinPatchPoints = 10u; // a patch's seed set must reach this size to attempt a fit at all

// ---------------------------------------------------------------------------
// czm_column_index_for_zone — the running column-offset for zone `zone`
// (sum of kCzmZoneSectors[0..zone)). A tiny unrolled loop (kCzmNumZones==4)
// — not worth a precomputed table for 4 entries (same "know when not to
// over-engineer" lesson as select_best_hypothesis above).
// ---------------------------------------------------------------------------
inline int czm_column_offset_for_zone(int zone)
{
    int off = 0;
    for (int z = 0; z < zone; ++z) off += kCzmZoneSectors[z];
    return off;
}

// ---------------------------------------------------------------------------
// czm_compute_patch_id — the single-sourced polar patch-assignment formula:
// range+azimuth -> (zone,ring,sector) -> a flat patch id in [0,kCzmNumPatches),
// or -1 if the point falls outside every zone's range band.
//
// Parameters: x,y — the point's sensor-frame horizontal coordinates (m); z
//             is NOT used here (a purely horizontal polar partition — the
//             per-patch height reasoning happens later, in the fit kernel).
// Complexity: O(kCzmNumZones) = O(1) for a fixed zone count.
// ---------------------------------------------------------------------------
inline int czm_compute_patch_id(float x, float y)
{
    const float r = std::sqrt(x * x + y * y);
    if (r < kCzmZoneEdgesM[0] || r >= kCzmZoneEdgesM[kCzmNumZones]) return -1;  // outside the CZM footprint

    int zone = 0;
    for (int z = 0; z < kCzmNumZones; ++z) {
        if (r >= kCzmZoneEdgesM[z] && r < kCzmZoneEdgesM[z + 1]) { zone = z; break; }
    }
    const float zone_lo = kCzmZoneEdgesM[zone];
    const float zone_hi = kCzmZoneEdgesM[zone + 1];
    const float ring_width = (zone_hi - zone_lo) / static_cast<float>(kCzmRingsPerZone);
    int ring = static_cast<int>((r - zone_lo) / ring_width);
    if (ring >= kCzmRingsPerZone) ring = kCzmRingsPerZone - 1;  // guard the r==zone_hi boundary (float rounding)

    const int sectors = kCzmZoneSectors[zone];
    // atan2 range is (-pi,pi]; shift to [0, 2*pi) before binning into sectors.
    const float kPi = 3.14159265358979323846f;
    float az = std::atan2(y, x);
    if (az < 0.0f) az += 2.0f * kPi;
    int sector = static_cast<int>(az / (2.0f * kPi / static_cast<float>(sectors)));
    if (sector >= sectors) sector = sectors - 1;   // guard az==2*pi boundary (float rounding)
    if (sector < 0) sector = 0;

    const int column = czm_column_offset_for_zone(zone) + sector;
    return column * kCzmRingsPerZone + ring;
}

// Per-patch fit result — the DEVICE output of the fit kernel, and the
// INDEPENDENT CPU twin's output, compared by main.cu's VERIFY(czm_fit) gate.
struct CzmPatchResult {
    PlaneModel plane;             // valid only when is_ground != 0
    int   is_ground = 0;          // 1 if this patch passed the uprightness+flatness tests
    unsigned int patch_point_count = 0;  // total points assigned to this patch (both seed and non-seed)
    unsigned int seed_point_count  = 0;  // points used for the plane fit itself
    float rms_residual_m = 0.0f;  // seed-point RMS distance to the fitted plane (the flatness test's statistic)
    float uprightness_deg = 0.0f; // angle between the fitted normal and +z (the uprightness test's statistic)
    int   used_prior = 0;         // 1 if this patch's seed used the height-carry rule (ring 1 with a passing ring 0)
};

// ===========================================================================
// GPU kernel declarations — nvcc-only (see the file header for why this
// fence exists: cl.exe, compiling reference_cpu.cpp, has never heard of
// __global__ and must never see these).
// ===========================================================================
#ifdef __CUDACC__

// ---- Milestone 1: RANSAC -------------------------------------------------

// ransac_generate_hypotheses_kernel — one thread per hypothesis (grid-stride
// over kRansacK). Picks a triplet (retrying up to kRansacMaxTripletAttempts
// times on a degenerate draw), fits a plane, and writes it + a valid flag.
__global__ void ransac_generate_hypotheses_kernel(int n, const float* __restrict__ xyz,
                                                   uint32_t global_seed, int k,
                                                   PlaneModel* __restrict__ hyp_plane,
                                                   uint8_t* __restrict__ hyp_valid);

// ransac_evaluate_hypotheses_kernel — the K x N batched heart of Milestone 1.
// ONE BLOCK PER HYPOTHESIS (grid.x == k): every thread in the block strides
// over all n points, testing |distance| <= threshold and reducing a local
// inlier count via shared memory to one integer per block. See kernels.cu
// for the alternative mapping (thread-per-(hypothesis,chunk) with atomics)
// and why hypothesis-per-block was chosen.
__global__ void ransac_evaluate_hypotheses_kernel(int n, const float* __restrict__ xyz,
                                                   const PlaneModel* __restrict__ hyp_plane,
                                                   const uint8_t* __restrict__ hyp_valid,
                                                   float threshold,
                                                   int* __restrict__ hyp_inlier_count);

// ransac_accumulate_inliers_kernel — one thread per point (grid-stride);
// points within `threshold` of `plane` atomicAdd their contribution into
// the 9 running sums + count of `accum` (a single CovAccum9 in device
// memory) and set point_inlier_mask[i]=1. See kernels.cu for why this one
// step uses atomics (order-nondeterministic, tolerance-compared) rather
// than Method-B-style fixed-order reduction — the same "know the
// determinism/performance trade" lesson 02.01 teaches with its two methods.
__global__ void ransac_accumulate_inliers_kernel(int n, const float* __restrict__ xyz,
                                                  PlaneModel plane, float threshold,
                                                  CovAccum9* __restrict__ accum,
                                                  uint8_t* __restrict__ point_inlier_mask);

// ransac_refine_kernel — <<<1,1>>>: a deliberately SERIAL single-thread
// kernel that turns the accumulated covariance into a refined plane via the
// shared eigensolver. K=1 solve; contrast with czm_fit_and_classify_kernel
// below, which runs 160 of these in parallel, one per patch.
__global__ void ransac_refine_kernel(const CovAccum9* __restrict__ accum,
                                     PlaneModel raw_plane, PlaneModel* __restrict__ refined_plane,
                                     int* __restrict__ refined_ok);

// ---- Milestone 2: Patchwork++-style CZM ----------------------------------

// czm_compute_patch_ids_kernel — one thread per point (grid-stride): the
// device transcription of czm_compute_patch_id above.
__global__ void czm_compute_patch_ids_kernel(int n, const float* __restrict__ xyz,
                                             int* __restrict__ patch_id);

// czm_fit_and_classify_kernel — ONE BLOCK PER COLUMN (grid.x ==
// kCzmNumColumns): processes that column's ring-0 patch, then its ring-1
// patch (in that order — ring 1 may use ring 0's fitted plane as its
// height prior), fitting each via block-wide reductions + the shared
// eigensolver, testing uprightness/flatness, and finally classifying every
// point in both patches against whichever plane (or "non-ground" if the
// patch failed its tests). See kernels.cu for the full block-level
// algorithm walkthrough.
//
// idx_sorted[n]  — point indices in ASCENDING patch-id order (from Thrust
//                  stable_sort_by_key in launch_czm_sort_and_index).
// patch_start[kCzmNumPatches+1] — patch p's points are
//                  idx_sorted[patch_start[p] .. patch_start[p+1]).
__global__ void czm_fit_and_classify_kernel(const float* __restrict__ xyz,
                                            const int* __restrict__ idx_sorted,
                                            const int* __restrict__ patch_start,
                                            CzmPatchResult* __restrict__ patch_result,
                                            uint8_t* __restrict__ point_ground);

#endif // __CUDACC__

// ===========================================================================
// Host-callable launch wrappers (definitions in kernels.cu, nvcc-only TU —
// but these DECLARATIONS are plain C++, visible to main.cu).
// ===========================================================================

void launch_ransac_generate_hypotheses(int n, const float* d_xyz, uint32_t global_seed, int k,
                                       PlaneModel* d_hyp_plane, uint8_t* d_hyp_valid);

void launch_ransac_evaluate_hypotheses(int n, const float* d_xyz, const PlaneModel* d_hyp_plane,
                                       const uint8_t* d_hyp_valid, float threshold, int k,
                                       int* d_hyp_inlier_count);

void launch_ransac_accumulate_inliers(int n, const float* d_xyz, PlaneModel plane, float threshold,
                                      CovAccum9* d_accum, uint8_t* d_point_inlier_mask);

void launch_ransac_refine(const CovAccum9* d_accum, PlaneModel raw_plane,
                          PlaneModel* d_refined_plane, int* d_refined_ok);

void launch_czm_compute_patch_ids(int n, const float* d_xyz, int* d_patch_id);

// launch_czm_sort_and_index — Thrust stable_sort_by_key(patch_id, point
// index) + a vectorized thrust::lower_bound probe for the kCzmNumPatches+1
// patch_start boundaries (see kernels.cu for what each Thrust call computes
// and why — CLAUDE.md §6.1 rule 6). d_patch_id_scratch[n] is consumed
// in-place (sorted); d_point_idx[n] becomes the sorted permutation;
// d_patch_start[kCzmNumPatches+1] receives the boundaries.
void launch_czm_sort_and_index(int n, int* d_patch_id_scratch, int* d_point_idx, int* d_patch_start);

void launch_czm_fit_and_classify(const float* d_xyz, const int* d_idx_sorted, const int* d_patch_start,
                                 CzmPatchResult* d_patch_result, uint8_t* d_point_ground);

// ===========================================================================
// CPU references (reference_cpu.cpp) — the correctness-oracle twins.
// All pointers are HOST pointers. See reference_cpu.cpp's file header for
// the independence ruling each follows (mirroring 02.01's: shared
// data-layout formulas are called directly from this header; genuinely
// algorithmic steps get an INDEPENDENT re-implementation with a different
// data structure and/or precision, per CLAUDE.md's "twin vs. shared" rule).
// ===========================================================================

// ransac_generate_hypotheses_cpu — regenerates the SAME K hypotheses the
// GPU kernel does, using this header's shared hypothesis_seed/pick_triplet_
// indices/plane_from_triplet functions directly (no duplication needed:
// these are plain host-callable functions already). VERIFY(hypotheses) in
// main.cu compares this against the GPU's device-transcribed result,
// bit-exact — the drift-catching gate the file header promises.
void ransac_generate_hypotheses_cpu(int n, const float* xyz, uint32_t global_seed, int k,
                                    PlaneModel* hyp_plane, uint8_t* hyp_valid);

// ransac_evaluate_hypotheses_cpu — an INDEPENDENT re-implementation of the
// K x N inlier count (own loop nesting, no shared-memory reduction —
// genuinely different code, not a transcription) compared bit-exact
// (VERIFY(ransac_eval)) against the GPU's per-hypothesis counts.
void ransac_evaluate_hypotheses_cpu(int n, const float* xyz, const PlaneModel* hyp_plane,
                                    const uint8_t* hyp_valid, float threshold, int k,
                                    int* hyp_inlier_count);

// ransac_refine_cpu — an INDEPENDENT least-squares refinement: DOUBLE
// PRECISION sequential accumulation (unlike the GPU's float atomics — see
// ransac_accumulate_inliers_kernel's comment for why that path is
// order-nondeterministic) over inliers of `plane`, then this header's
// jacobi_eigen_3x3. Compared within a measured-then-margined tolerance
// (VERIFY(ransac_refine)) — the same "independent, different precision and
// order" ruling 02.01's hashmap_downsample_cpu documents for Method A.
bool ransac_refine_cpu(int n, const float* xyz, PlaneModel plane, float threshold, PlaneModel& refined_out);

// czm_compute_patch_ids_cpu — a thin loop calling this header's shared
// czm_compute_patch_id per point (the CPU side of the SAME single-sourced
// formula the GPU device-transcribes) — VERIFY(patch_ids) compares this
// against the GPU kernel's output, bit-exact.
void czm_compute_patch_ids_cpu(int n, const float* xyz, int* patch_id);

// czm_fit_and_classify_cpu — a GENUINELY INDEPENDENT re-implementation of
// Milestone 2: builds per-patch point LISTS with std::vector<std::vector<int>>
// (a different data structure than the GPU's Thrust-sort-based contiguous
// layout), processes ring0->ring1 per column sequentially (same algorithm,
// independent code), and calls this header's jacobi_eigen_3x3. Compared
// against the GPU's CzmPatchResult array and point_ground labels within
// documented tolerances (VERIFY(czm_fit) — not bit-exact: summation order
// and float-vs-... differ, same ruling class as ransac_refine_cpu above).
void czm_fit_and_classify_cpu(int n, const float* xyz, const int* patch_id,
                              CzmPatchResult* patch_result, uint8_t* point_ground);

#endif // PROJECT_KERNELS_CUH
