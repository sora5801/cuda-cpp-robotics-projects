// ===========================================================================
// kernels.cuh — interface for project 02.10
//               FPFH descriptors + RANSAC global registration (align two
//               scans with NO initial guess: local geometry -> a pose-
//               invariant descriptor per point -> descriptor matching ->
//               RANSAC over correspondence triplets -> a few point-to-plane
//               ICP iterations to polish the result)
//
// Role in the project
// -------------------
// The CONTRACT shared by main.cu (orchestration, every VERIFY/GATE, every
// artifact), kernels.cu (the GPU kernels), and reference_cpu.cpp (the
// independent CPU oracle twins). Every data-layout decision all three must
// agree on — point-cloud layout, descriptor layout, correspondence-record
// layout, the RANSAC hypothesis representation, the SE(3) pose convention —
// is defined HERE, once (CLAUDE.md paragraph 12).
//
// POINT CLOUD LAYOUT — float* xyz, interleaved, meters, LiDAR "sensor" frame
// (origin at the sensor, +x forward, +y left, +z up — the same body
// convention 02.09 states explicitly): xyz[i*3+0..2] = x,y,z.
//
// NORMAL LAYOUT — float* nrm, same interleaved layout, unit vectors.
//
// DESCRIPTOR LAYOUT — float* fpfh, interleaved, kFpfhDim=33 floats/point:
// fpfh[i*33 + 0..10] = the alpha sub-histogram (11 bins), [11..21] = phi,
// [22..32] = theta (kernels.cu's compute_spfh/compute_fpfh headers derive
// what each angle means and why three SEPARATE 11-bin histograms, not one
// joint 11^3 histogram, are used — Rusu, Blodow & Beetz, ICRA 2009).
//
// THE PIPELINE, six stages (every stage below is its own kernel; the
// per-project "ratified scope" brief names this exact shape):
//
//   STAGE 1 — NORMALS (02.09-lineage, reimplemented COMPACTLY here per the
//   self-containment rule, cited not imported): brute-force KNN (this
//   project's point counts are ~1500-3200/scan — 02.06's own "the honest
//   teaching choice at this scale" reasoning applies verbatim; 02.09's
//   voxel hash is the right tool at ITS 1M+-point throughput target, not
//   here — see THEORY.md "Where this sits in the real world") -> mean-
//   shifted covariance -> Jacobi eigensolve -> smallest-eigenvalue
//   eigenvector, oriented toward the cloud's own centroid (02.06's
//   ref_point convention, cited).
//
//   STAGE 2 — SPFH (Simplified Point Feature Histogram, ONE per point):
//   for each of the point's K nearest neighbors, build the local DARBOUX
//   FRAME at the query point and read off three angles (alpha, phi, theta)
//   between the query/neighbor normals and the connecting line — a triplet
//   that depends ONLY on relative geometry (THEORY.md "The math" proves the
//   pose-invariance this whole method rests on). Histogram each of the
//   three angles into 11 bins SEPARATELY (33 dims total), normalize each
//   11-bin block to sum 1. O(n*K) total — already linear, because SPFH
//   fixes the query point as the frame's origin for every pair (no O(k^2)
//   all-pairs scan — see kernels.cu's compute_spfh_kernel header for the
//   contrast with true PFH).
//
//   STAGE 3 — FPFH (the "Fast" in FPFH): re-weight each point's OWN SPFH
//   with its K neighbors' ALREADY-COMPUTED SPFH values (kernels.cu's
//   compute_fpfh_kernel header derives the "two-ring" reading: ring 1 is
//   this point's own K neighbors; ring 2 arrives for free because each of
//   THOSE neighbors' SPFH already summarizes ITS OWN K-neighborhood — no
//   second explicit traversal). Still O(n*K): the whole reason FPFH exists
//   is to approximate true PFH's O(n*K^2) all-pairs cost this cheaply.
//   L1-normalized at the end (the ratified scope's explicit instruction).
//
//   STAGE 4 — MATCH: for every SOURCE point, brute-force nearest (L2, 33-D)
//   TARGET descriptor, with a documented ratio test (1st-nearest / 2nd-
//   nearest <= kMatchRatioMax) rejecting AMBIGUOUS matches — 01.04's
//   "feature matching without a distinctiveness check is a coin flip"
//   lesson, arriving in 3-D (README/THEORY.md cite it explicitly; this
//   project's flat floor/wall patches are, by design, locally
//   SELF-SIMILAR — exactly the aliasing hazard 01.04/01.05 teach for 2-D
//   image patches).
//
//   STAGE 5 — RANSAC over CORRESPONDENCE TRIPLETS (not over raw points —
//   02.03's plane-RANSAC teaches the pattern over 3 POINTS; this project
//   teaches it over 3 CORRESPONDENCE PAIRS, the natural generalization for
//   registration): sample 3 correspondences, PRESCREEN by pairwise edge-
//   length consistency (a rigid transform preserves inter-point distances —
//   THEORY.md derives this; it is what makes RANSAC-over-correspondences
//   fast, by rejecting most bad triplets for the cost of 3 subtractions,
//   BEFORE ever computing a fit), fit the minimal rigid transform (Horn
//   1987's closed-form quaternion solution, cited — 02.06/01.17's Rigid3/
//   T_target_source lineage, cited for the pose CONVENTION), score by
//   inlier count over the WHOLE correspondence set. Hypothesis-parallel on
//   GPU (02.03's "farm" pattern, cited): one thread per hypothesis.
//
//   STAGE 6 — REFIT + HANDOFF: the best hypothesis's inliers are refit
//   (same Horn solver, now over dozens-hundreds of points, not 3) for the
//   final RANSAC transform; then a FEW point-to-plane ICP iterations (02.06
//   lineage, cited, reimplemented compactly — THEORY.md "Where this sits in
//   the real world" contrasts the two projects' scope) polish it using the
//   STAGE 1 target normals already in hand. The registration_recovery /
//   icp_negative_control gates in main.cu measure, honestly, that RANSAC
//   finds what local ICP alone provably cannot at this relative pose.
//
// Why THIS header is CUDA-qualifier-free where possible (02.01/02.03/02.09's
// identical precedent): small, deterministic, formulaic pieces (the RNG,
// the edge-length prescreen, the Horn rigid fit + its 4x4 eigensolve, the
// Darboux-triplet-to-bin-index formula) are declared PLAIN inline C++ so
// BOTH nvcc (this project's .cu files) and cl.exe (reference_cpu.cpp) can
// see and call them directly. Being unqualified, they are HOST-only under
// nvcc's rules and cannot run inside a __global__ kernel — kernels.cu
// therefore carries its own literal __device__ transcription of each one,
// cross-referenced in comments at both copies (the drift-catching VERIFY
// gate is named at each use site below and in main.cu).
//
// Twin-vs-shared ruling for THIS project (reference_cpu.cpp's file header
// states the general rule; here is how it resolves per stage):
//   * STAGE 1 (normals): INDEPENDENT CPU eigensolve (02.09's stricter
//     choice, cited) — this is the explicitly-cited upstream lineage, and
//     TWO genuinely different Jacobi implementations catch more than one
//     shared header call would.
//   * STAGE 2/3 (SPFH/FPFH): the Darboux-triplet + binning FORMULA is
//     SHARED (small, deterministic, like 02.03's czm_compute_patch_id) —
//     the CPU twin calls it directly over its OWN independently-built
//     neighbor list; only the accumulation LOOP differs by construction.
//   * STAGE 5/6 (RANSAC): the per-hypothesis Horn FIT is SHARED (like
//     02.03's plane_from_triplet) so GPU-vs-CPU hypothesis generation is
//     bit-exact-checkable; the INLIER-COUNTING loop is independently
//     reimplemented (02.03's ransac_evaluate_hypotheses_cpu precedent); the
//     REFIT gets a fully INDEPENDENT double-precision reimplementation
//     (02.03's ransac_refine_cpu precedent) as the project's non-tautological
//     numerical cross-check. The registration_recovery / icp_negative_control
//     / descriptor_invariance gates compare against GROUND TRUTH, never
//     against the GPU/CPU peer — the "analytic/negative-control" tier the
//     independence ruling requires beyond twin agreement alone.
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cstdint>    // int32_t, uint32_t — exact-width integers everywhere below
#include <cmath>      // std::sqrt/std::fabs/std::atan2/std::floor — identical overloads to cl.exe and nvcc's host pass
#include <vector>     // reference_cpu.cpp's independent oracle outputs

// ===========================================================================
// Launch geometry + problem-scale constants — the numbers every stage and
// every CPU twin below must agree on bit-for-bit.
// ===========================================================================
constexpr int kThreadsPerBlock = 256;   // warp multiple; good default occupancy sm_75..sm_89 (repo-wide default)

// blocks_for — integer ceiling division: how many `threads`-wide blocks
// cover `count` independent problems (the 02.01/02.03/02.06/02.09 idiom).
inline int blocks_for(int count, int threads)
{
    return (count + threads - 1) / threads;
}

// kFpfhK — neighborhood size for BOTH the normal fit (STAGE 1) and SPFH
// pairing (STAGE 2), EXCLUDING the query point itself. One brute-force KNN
// pass per point serves both stages (main.cu computes it once, feeds two
// consumers) — 20 sits in PCL's typical 10-30 range for local surface
// description; THEORY.md's "Numerical considerations" derives the
// noise-vs-locality tradeoff.
constexpr int kFpfhK = 20;

// kFpfhBins / kFpfhDim — 11 bins per angle (Rusu et al.'s own choice,
// cited), 3 angles (alpha, phi, theta), concatenated: 33 total dims.
constexpr int kFpfhBins = 11;
constexpr int kFpfhDim  = 3 * kFpfhBins;   // = 33

// kJacobiSweeps3 — cyclic-Jacobi sweep count for the 3x3 normal-covariance
// eigensolve (02.03/02.09's measured-sufficient value for float32 3x3).
constexpr int kJacobiSweeps3 = 8;

// kJacobiSweeps4 — cyclic-Jacobi sweep count for Horn's 4x4 quaternion
// eigenproblem (STAGE 5/6). A 4x4 symmetric matrix has 6 off-diagonal pairs
// (vs. 3x3's 3), so convergence needs more sweeps to reach the same
// float32 precision; THEORY.md "Numerical considerations" reports the
// measured residual vs. sweep count that justifies this value.
constexpr int kJacobiSweeps4 = 14;

// kMatchRatioMax — Lowe-style ratio test (STAGE 4), compared on SQUARED
// distances (avoids a sqrt per one of the N_src*N_tgt comparisons — 01.04's
// "don't compute what a threshold test doesn't need" lesson): accept a
// match iff dist1_sq <= kMatchRatioMax^2 * dist2_sq. Looser than SIFT's
// classic 0.7-0.8 (THEORY.md explains why: a 33-D geometric histogram over
// a partially self-similar room is inherently less separable than a
// 128-D SIFT descriptor over a richly textured image — measured, not
// assumed, in main.cu's descriptor_distance_histogram.csv artifact).
constexpr float kMatchRatioMax = 0.95f;

// ---- STAGE 5/6: RANSAC over correspondence triplets -----------------------
// kRansacK — hypothesis budget. Sized against a REAL measured run (main.cu's
// ransac_formula gate): this scene's correspondence set (post ratio test)
// runs ~10% true-geometric-inlier fraction w (measured: 253/2416 on the
// committed pair1) because the room's flat floor/wall patches are locally
// self-similar (README/THEORY.md's 01.04/01.05 citation) and pass the ratio
// test more often than a richly-varied scene would allow. At w~0.10, the
// classical k=log(1-p)/log(1-w^3) formula for p=0.999 needs ~6000
// iterations; 8192 leaves real margin above that measured requirement.
constexpr int    kRansacK                  = 8192;
constexpr float  kRansacInlierThresholdM   = 0.15f;  // |T(src)-tgt| <= this => inlier, under a HYPOTHESIS transform (m)
constexpr float  kRansacMinEdgeLenM        = 1.5f;   // reject a triplet whose smallest pairwise SOURCE distance is below this (ill-conditioned minimal fit)
constexpr float  kRansacEdgeLenTolM        = 0.10f;  // reject a triplet if |src_pairwise_dist - tgt_pairwise_dist| exceeds this for ANY of its 3 pairs
constexpr int    kRansacMaxTripletAttempts = 8;      // per-hypothesis retries on a degenerate/duplicate draw (02.03 precedent)
constexpr double kRansacTargetSuccessProb  = 0.999;  // p in k = log(1-p)/log(1-w^3) — the ransac_formula gate

// ---- STAGE 6: point-to-plane ICP handoff -----------------------------------
constexpr int   kIcpMaxIters      = 10;      // small — this is a POLISH step, not the primary solver (THEORY.md)
constexpr float kIcpMaxCorrDistM  = 0.50f;   // correspondence-rejection gate (m); generous because RANSAC already landed close
constexpr float kIcpConvRotDeg    = 0.01f;   // "converged" twist thresholds — identical values to 02.06 (cited), same justification
constexpr float kIcpConvTransM    = 0.0001f;
constexpr double kIcpDampingLM    = 1.0e-3;  // Tikhonov damping added to the 6x6 system's diagonal (33.01/02.06 "JtJ + lambda*I" pattern, cited)

// ---------------------------------------------------------------------------
// Rigid3 — a rigid-body transform, passed by value to kernels that need
// "where is the source cloud right now" (02.06's identical Rigid3/by-value
// convention, cited: T changes every RANSAC hypothesis / every ICP
// iteration, so a kernel PARAMETER — not a __constant__ upload — is the
// cheapest place for 30000+ threads to read the SAME 48 bytes once each).
// R is ROW-MAJOR (R[i*3+j] = row i, col j). x_target = R * p_source + t.
// This is T_target_source (02.06/01.17's naming convention, cited): "the
// source cloud's frame, expressed in the target cloud's frame".
// ---------------------------------------------------------------------------
struct Rigid3 {
    float R[9];
    float t[3];
};

inline void apply_rigid(const Rigid3& T, const float p[3], float out[3])
{
    out[0] = T.R[0] * p[0] + T.R[1] * p[1] + T.R[2] * p[2] + T.t[0];
    out[1] = T.R[3] * p[0] + T.R[4] * p[1] + T.R[5] * p[2] + T.t[1];
    out[2] = T.R[6] * p[0] + T.R[7] * p[1] + T.R[8] * p[2] + T.t[2];
}

inline float squared_distance3(const float p[3], const float q[3])
{
    const float dx = p[0] - q[0], dy = p[1] - q[1], dz = p[2] - q[2];
    return dx * dx + dy * dy + dz * dz;
}

// hidx(i,j) — flatten the upper-triangle (i<=j) index of a 6x6 symmetric
// matrix into the 0..20 offset the STAGE 6 point-to-plane accumulator
// (kernels.cu's icp_accumulate_kernel) and main.cu's unpacking both use —
// 02.06's identical layout and derivation, cited: row i's valid columns are
// j=i..5, so row i starts right after all earlier rows' entries
// (row_start = {0,6,11,15,18,20}). Parameter order [wx,wy,wz,vx,vy,vz]
// (rotation-first twist, then translation) — 02.06's convention, reused
// here for the SAME point-to-plane linearization (THEORY.md re-derives it
// for this project's own Jacobian: J = [x_cur x n_tgt, n_tgt]).
inline int hidx(int i, int j)
{
    const int row_start[6] = { 0, 6, 11, 15, 18, 20 };
    return row_start[i] + (j - i);   // caller guarantees i <= j <= 5
}

// ---------------------------------------------------------------------------
// The shared total order every KNN heap (GPU, CPU twin, brute-force) sorts
// by, and its documented TIE-BREAK — 02.05/02.09's identical knn_less
// pattern, reused here for THIS project's own brute-force KNN: smaller
// dist2 wins; on an exact tie, the smaller point index wins. A fixed total
// order is what lets GPU and CPU twins agree on the EXACT K-set, not just
// an equivalent one.
// ---------------------------------------------------------------------------
inline bool knn_less(float da, int32_t ia, float db, int32_t ib)
{
    if (da != db) return da < db;
    return ia < ib;
}

// ===========================================================================
// RANSAC RNG — xorshift32 (Marsaglia 2003), the SAME three-shift/three-XOR
// core used repo-wide (02.01/02.03/08.01/11.01, cited), plus 02.03's
// counter-based per-hypothesis seeding (cited, reused verbatim): each
// hypothesis derives its OWN starting state from its own index, so K
// independent GPU threads never need a shared RNG cursor (embarrassingly
// parallel BY CONSTRUCTION — 02.03's kernels.cuh derives this in full).
// ===========================================================================
inline uint32_t xorshift32_step(uint32_t state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

inline uint32_t hypothesis_seed(uint32_t global_seed, int k, int attempt)
{
    uint32_t s = global_seed ^ (0x9E3779B9u * static_cast<uint32_t>(k * 8 + attempt + 1));
    if (s == 0u) s = 1u;
    s = xorshift32_step(s);
    s = xorshift32_step(s);
    return s;
}

// pick_correspondence_triplet — draw 3 DISTINCT correspondence indices in
// [0,nc) from one xorshift32 stream seeded by `seed`. Distinctness is
// enforced by simple rejection-and-reseed within the small fixed draw (nc
// is at most a few hundred here, so a collision is rare and cheap to retry
// inline — unlike 02.03's pick_triplet_indices, which lets plane_from_
// triplet's degenerate check absorb an index collision instead).
inline bool pick_correspondence_triplet(uint32_t seed, int nc, int& i0, int& i1, int& i2)
{
    if (nc < 3) return false;
    uint32_t s = seed;
    s = xorshift32_step(s); i0 = static_cast<int>(s % static_cast<uint32_t>(nc));
    for (int guard = 0; guard < 8; ++guard) {
        s = xorshift32_step(s); i1 = static_cast<int>(s % static_cast<uint32_t>(nc));
        if (i1 != i0) break;
    }
    for (int guard = 0; guard < 8; ++guard) {
        s = xorshift32_step(s); i2 = static_cast<int>(s % static_cast<uint32_t>(nc));
        if (i2 != i0 && i2 != i1) break;
    }
    return (i0 != i1) && (i0 != i2) && (i1 != i2);
}

// ---------------------------------------------------------------------------
// edge_length_prescreen — THE speed trick that makes correspondence RANSAC
// fast (STAGE 5's file-header promise; THEORY.md "The algorithm" derives
// why a rigid transform must preserve every pairwise distance): before
// ever fitting a candidate transform, check that the 3 SOURCE-side pairwise
// distances match the 3 TARGET-side pairwise distances of the SAME
// correspondence triplet within kRansacEdgeLenTolM, and that no pair is
// nearly coincident (kRansacMinEdgeLenM — a numerically healthy minimal fit
// needs well-separated points). A triplet containing even ONE wrong
// correspondence almost always fails this O(1) check (three subtractions
// and three compares) — rejected for a tiny fraction of a full Horn fit's
// cost. main.cu's prescreen_efficiency [info] measurement reports exactly
// what fraction of drawn triplets this catches.
// ---------------------------------------------------------------------------
inline bool edge_length_prescreen(const float s0[3], const float s1[3], const float s2[3],
                                  const float t0[3], const float t1[3], const float t2[3])
{
    const float ds01 = std::sqrt(squared_distance3(s0, s1));
    const float ds02 = std::sqrt(squared_distance3(s0, s2));
    const float ds12 = std::sqrt(squared_distance3(s1, s2));
    if (ds01 < kRansacMinEdgeLenM || ds02 < kRansacMinEdgeLenM || ds12 < kRansacMinEdgeLenM) return false;

    const float dt01 = std::sqrt(squared_distance3(t0, t1));
    const float dt02 = std::sqrt(squared_distance3(t0, t2));
    const float dt12 = std::sqrt(squared_distance3(t1, t2));
    if (std::fabs(ds01 - dt01) > kRansacEdgeLenTolM) return false;
    if (std::fabs(ds02 - dt02) > kRansacEdgeLenTolM) return false;
    if (std::fabs(ds12 - dt12) > kRansacEdgeLenTolM) return false;
    return true;
}

// ===========================================================================
// Horn's closed-form absolute orientation (Horn 1987, "Closed-form solution
// of absolute orientation using unit quaternions", JOSA A 4(4)) — the
// shared rigid-fit-from-correspondences routine used for BOTH the minimal
// 3-point RANSAC hypothesis fit and the final many-point inlier refit
// (same function, any count>=3; THEORY.md "The math" derives it in full).
//
// Algorithm: centroid-subtract both point sets, form the 3x3 cross-
// covariance M = sum (s_i-cs)(t_i-ct)^T, pack it into Horn's 4x4 SYMMETRIC
// "key matrix" N (below), and take the eigenvector of N's LARGEST
// eigenvalue as the optimal rotation QUATERNION (w,x,y,z) — a genuinely
// different construction from the 3x3 covariance-normal eigenproblem
// STAGE 1 solves (there the SMALLEST eigenvalue's eigenvector is wanted;
// here it is the LARGEST). Translation follows in closed form: t = ct -
// R*cs. This is a GLOBAL least-squares optimum, not an iterative
// approximation — the reason a single Horn fit (not a gradient step) is
// the right tool for the 3-point minimal case.
// ---------------------------------------------------------------------------
inline void horn_build_n_matrix(const float M[9], float N[16])
{
    const float Sxx = M[0], Sxy = M[1], Sxz = M[2];
    const float Syx = M[3], Syy = M[4], Syz = M[5];
    const float Szx = M[6], Szy = M[7], Szz = M[8];
    // Row-major 4x4, symmetric by construction (Horn 1987 eq. 33-ish; see
    // THEORY.md for the full derivation from the trace/antisymmetric parts
    // of M). Only the algebra changes between STAGE 1's 3x3 eigenproblem
    // and this one — the JACOBI MACHINERY that diagonalizes them is the
    // same idea at a different size (see jacobi_eigen_4x4 below).
    N[0]  = Sxx + Syy + Szz;   N[1]  = Syz - Szy;         N[2]  = Szx - Sxz;         N[3]  = Sxy - Syx;
    N[4]  = N[1];              N[5]  = Sxx - Syy - Szz;   N[6]  = Sxy + Syx;         N[7]  = Szx + Sxz;
    N[8]  = N[2];              N[9]  = N[6];              N[10] = -Sxx + Syy - Szz;  N[11] = Syz + Szy;
    N[12] = N[3];              N[13] = N[7];              N[14] = N[11];             N[15] = -Sxx - Syy + Szz;
}

// jacobi_eigen_4x4 — cyclic Jacobi eigensolve for a SYMMETRIC 4x4 matrix,
// packed as the 10-entry upper triangle (row-major reading order: [0][0],
// [0][1],[0][2],[0][3],[1][1],[1][2],[1][3],[2][2],[2][3],[3][3]). Same
// algorithm family as 02.03/02.09's 3x3 Jacobi (Golub & Van Loan
// "Matrix Computations" section 8.4, cited), generalized to 6 off-diagonal
// pairs: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3). eigenvalues[] and
// eigenvectors[][] are returned in NO particular order (unlike the 3x3
// normal solver, callers here want the LARGEST, found by a trivial max-scan
// — see rigid_fit_horn below — not a full sort).
inline void jacobi_eigen_4x4(const float a_in[10], float eigenvalues[4], float eigenvectors[4][4])
{
    float A[4][4] = {
        { a_in[0], a_in[1], a_in[2], a_in[3] },
        { a_in[1], a_in[4], a_in[5], a_in[6] },
        { a_in[2], a_in[5], a_in[7], a_in[8] },
        { a_in[3], a_in[6], a_in[8], a_in[9] },
    };
    float V[4][4] = { {1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {0,0,0,1} };

    const int pairs[6][2] = { {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3} };
    for (int sweep = 0; sweep < kJacobiSweeps4; ++sweep) {
        for (int pi = 0; pi < 6; ++pi) {
            const int p = pairs[pi][0], q = pairs[pi][1];
            const float apq = A[p][q];
            if (std::fabs(apq) < 1.0e-12f) continue;   // already ~zero: nothing to rotate
            const float theta = (A[q][q] - A[p][p]) / (2.0f * apq);
            const float t = (theta >= 0.0f ? 1.0f : -1.0f) /
                            (std::fabs(theta) + std::sqrt(theta * theta + 1.0f));
            const float c = 1.0f / std::sqrt(t * t + 1.0f);
            const float s = t * c;
            const float app = A[p][p], aqq = A[q][q];
            A[p][p] = app - t * apq;
            A[q][q] = aqq + t * apq;
            A[p][q] = 0.0f; A[q][p] = 0.0f;
            // Every OTHER row/col (2 of them, for a 4x4) mixes too — the
            // 3x3 case had exactly one such "remaining index"; here there
            // are two, so this is a small loop instead of one hand-picked
            // index (compare 02.09's d_jacobi_rotate, cited).
            for (int i = 0; i < 4; ++i) {
                if (i == p || i == q) continue;
                const float aip = A[i][p], aiq = A[i][q];
                A[i][p] = c * aip - s * aiq; A[p][i] = A[i][p];
                A[i][q] = s * aip + c * aiq; A[q][i] = A[i][q];
            }
            for (int i = 0; i < 4; ++i) {
                const float vip = V[i][p], viq = V[i][q];
                V[i][p] = c * vip - s * viq;
                V[i][q] = s * vip + c * viq;
            }
        }
    }

    for (int i = 0; i < 4; ++i) {
        eigenvalues[i] = A[i][i];
        for (int j = 0; j < 4; ++j) eigenvectors[i][j] = V[j][i];  // eigenvectors[i] = i-th COLUMN of V
    }
}

// rigid_fit_horn — the shared entry point: `count` correspondence pairs
// (src[i] <-> tgt[i], count>=3) -> the least-squares rigid transform R,t
// minimizing sum_i |R*src[i]+t - tgt[i]|^2. Returns false (leaving R,t
// untouched) if the centroid-subtracted point sets are degenerate (near-
// zero cross-covariance — e.g. all source points coincide after the
// prescreen somehow still slipped through; defensive, not expected to ever
// fire given edge_length_prescreen already ran upstream in STAGE 5).
// ---------------------------------------------------------------------------
inline bool rigid_fit_horn(int count, const float* src_xyz, const float* tgt_xyz, float R_out[9], float t_out[3])
{
    if (count < 3) return false;
    float cs[3] = { 0, 0, 0 }, ct[3] = { 0, 0, 0 };
    for (int i = 0; i < count; ++i) {
        cs[0] += src_xyz[i * 3 + 0]; cs[1] += src_xyz[i * 3 + 1]; cs[2] += src_xyz[i * 3 + 2];
        ct[0] += tgt_xyz[i * 3 + 0]; ct[1] += tgt_xyz[i * 3 + 1]; ct[2] += tgt_xyz[i * 3 + 2];
    }
    const float inv_n = 1.0f / static_cast<float>(count);
    for (int k = 0; k < 3; ++k) { cs[k] *= inv_n; ct[k] *= inv_n; }

    // Cross-covariance M = sum (s-cs)(t-ct)^T, row-major 3x3 (M[r*3+c]).
    float M[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    for (int i = 0; i < count; ++i) {
        const float sx = src_xyz[i * 3 + 0] - cs[0], sy = src_xyz[i * 3 + 1] - cs[1], sz = src_xyz[i * 3 + 2] - cs[2];
        const float tx = tgt_xyz[i * 3 + 0] - ct[0], ty = tgt_xyz[i * 3 + 1] - ct[1], tz = tgt_xyz[i * 3 + 2] - ct[2];
        M[0] += sx * tx; M[1] += sx * ty; M[2] += sx * tz;
        M[3] += sy * tx; M[4] += sy * ty; M[5] += sy * tz;
        M[6] += sz * tx; M[7] += sz * ty; M[8] += sz * tz;
    }

    float trace_abs = std::fabs(M[0]) + std::fabs(M[4]) + std::fabs(M[8]);
    if (trace_abs < 1.0e-9f) return false;   // degenerate: essentially no spread to fit against

    float N[16];
    horn_build_n_matrix(M, N);
    const float n_packed[10] = { N[0], N[1], N[2], N[3], N[5], N[6], N[7], N[10], N[11], N[15] };
    float eigenvalues[4]; float eigenvectors[4][4];
    jacobi_eigen_4x4(n_packed, eigenvalues, eigenvectors);

    int best = 0;
    for (int i = 1; i < 4; ++i) if (eigenvalues[i] > eigenvalues[best]) best = i;
    float qw = eigenvectors[best][0], qx = eigenvectors[best][1], qy = eigenvectors[best][2], qz = eigenvectors[best][3];
    const float qn = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
    if (qn < 1.0e-12f) return false;
    const float inv_qn = 1.0f / qn;
    qw *= inv_qn; qx *= inv_qn; qy *= inv_qn; qz *= inv_qn;

    // Quaternion -> row-major rotation matrix (02.06's quat_to_matrix
    // formula, cited, reimplemented inline here since this header has no
    // Quat struct of its own — a rotation MATRIX is what apply_rigid needs).
    R_out[0] = 1.0f - 2.0f * (qy * qy + qz * qz);  R_out[1] = 2.0f * (qx * qy - qw * qz);         R_out[2] = 2.0f * (qx * qz + qw * qy);
    R_out[3] = 2.0f * (qx * qy + qw * qz);         R_out[4] = 1.0f - 2.0f * (qx * qx + qz * qz);  R_out[5] = 2.0f * (qy * qz - qw * qx);
    R_out[6] = 2.0f * (qx * qz - qw * qy);         R_out[7] = 2.0f * (qy * qz + qw * qx);         R_out[8] = 1.0f - 2.0f * (qx * qx + qy * qy);

    t_out[0] = ct[0] - (R_out[0] * cs[0] + R_out[1] * cs[1] + R_out[2] * cs[2]);
    t_out[1] = ct[1] - (R_out[3] * cs[0] + R_out[4] * cs[1] + R_out[5] * cs[2]);
    t_out[2] = ct[2] - (R_out[6] * cs[0] + R_out[7] * cs[1] + R_out[8] * cs[2]);
    return true;
}

// ===========================================================================
// STAGE 2's shared Darboux-frame + binning formula (the data-layout/math
// contract every implementation below must agree on — 02.03's
// czm_compute_patch_id precedent: small, deterministic, host-callable,
// used DIRECTLY by the CPU twin, transcribed literally for the device).
//
// Given the QUERY point's normal n_q and a NEIGHBOR (position p_k, normal
// n_k), builds the local Darboux frame (u,v,w) rooted at the query and
// returns the pose-invariant angle triplet (alpha, phi, theta) — THEORY.md
// "The math" derives every line below from first principles (u=n_q by
// definition; v,w span the plane perpendicular to it; each angle is a dot
// or atan2 of unit vectors, hence independent of any RIGID transform
// applied to both points+normals together — translations cancel in the
// difference p_k-p_q, and rotations rotate every vector in the dot/atan2
// identically, leaving the angle itself unchanged).
// ---------------------------------------------------------------------------
inline void darboux_triplet(const float n_q[3], const float p_q[3], const float n_k[3], const float p_k[3],
                            float& alpha, float& phi, float& theta)
{
    float d[3] = { p_k[0] - p_q[0], p_k[1] - p_q[1], p_k[2] - p_q[2] };
    const float dist = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    const float inv_dist = (dist > 1.0e-9f) ? (1.0f / dist) : 0.0f;
    d[0] *= inv_dist; d[1] *= inv_dist; d[2] *= inv_dist;   // d is now the UNIT direction query->neighbor

    const float u[3] = { n_q[0], n_q[1], n_q[2] };
    // v = normalize(u x d) — degenerate only when d is parallel to u (the
    // neighbor sits almost exactly "above" the query along its own normal,
    // a rare near-grazing configuration). Fall back to an arbitrary stable
    // perpendicular (the same "pick the world axis least aligned with u"
    // trick 02.03/02.09 use for their own perpendicular-basis helpers) so
    // the frame stays well-defined for every neighbor, never NaN.
    float v[3] = { u[1] * d[2] - u[2] * d[1], u[2] * d[0] - u[0] * d[2], u[0] * d[1] - u[1] * d[0] };
    float vnorm = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (vnorm < 1.0e-6f) {
        const float helper[3] = { (std::fabs(u[0]) < 0.9f) ? 1.0f : 0.0f, (std::fabs(u[0]) < 0.9f) ? 0.0f : 1.0f, 0.0f };
        v[0] = u[1] * helper[2] - u[2] * helper[1];
        v[1] = u[2] * helper[0] - u[0] * helper[2];
        v[2] = u[0] * helper[1] - u[1] * helper[0];
        vnorm = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    }
    const float inv_vnorm = 1.0f / vnorm;
    v[0] *= inv_vnorm; v[1] *= inv_vnorm; v[2] *= inv_vnorm;
    const float w[3] = { u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0] };

    alpha = v[0] * n_k[0] + v[1] * n_k[1] + v[2] * n_k[2];               // in [-1,1]
    phi   = u[0] * d[0] + u[1] * d[1] + u[2] * d[2];                     // in [-1,1]
    const float wn = w[0] * n_k[0] + w[1] * n_k[1] + w[2] * n_k[2];
    const float un = u[0] * n_k[0] + u[1] * n_k[1] + u[2] * n_k[2];
    theta = std::atan2(wn, un);                                          // in (-pi, pi]
}

// angle_to_bin — map alpha/phi (range [-1,1]) or theta (range (-pi,pi]) to
// a bin index in [0, kFpfhBins). Shared by GPU/CPU/reference so a boundary
// value bins IDENTICALLY everywhere the SAME float32 formula runs (the
// "shared formula = bit-exact" precedent — 02.03's czm_compute_patch_id).
inline int angle_to_bin(float value, float lo, float hi)
{
    float frac = (value - lo) / (hi - lo);
    if (frac < 0.0f) frac = 0.0f;
    if (frac > 0.99999994f) frac = 0.99999994f;   // largest float < 1.0f: keeps value==hi in the LAST bin, never bin==kFpfhBins
    int bin = static_cast<int>(frac * static_cast<float>(kFpfhBins));
    if (bin < 0) bin = 0;
    if (bin >= kFpfhBins) bin = kFpfhBins - 1;
    return bin;
}

constexpr float kPiF = 3.14159265358979323846f;

// ===========================================================================
// GPU kernel declarations — nvcc-only.
// ===========================================================================
#ifdef __CUDACC__

// ---- STAGE 1 helper: brute-force KNN (this project's neighbor engine — see
// the file header for why brute force is the right tool at THIS point
// count, citing 02.06's identical reasoning) ---------------------------------
//
// One thread per QUERY point q: scan all n points, keep the kFpfhK nearest
// EXCLUDING self via a bounded max-heap (02.05/02.09's identical pattern).
// out_neighbor_ids[q*kFpfhK + a] / out_neighbor_dist[q*kFpfhK + a] are
// ascending by knn_less (a=0 closest); guaranteed exactly kFpfhK entries
// whenever n > kFpfhK (always true at this project's scale).
__global__ void knn_search_kernel(int n, const float* __restrict__ xyz,
                                  int32_t* __restrict__ out_neighbor_ids,
                                  float* __restrict__ out_neighbor_dist);

// ---- STAGE 1: normals -------------------------------------------------------
// One thread per point: mean-shifted covariance over its kFpfhK neighbors
// -> Jacobi eigensolve -> smallest-eigenvalue eigenvector, oriented toward
// ref_point (the cloud's own centroid — see launch_estimate_normals).
__global__ void estimate_normals_kernel(int n, const float* __restrict__ xyz,
                                        const int32_t* __restrict__ neighbor_ids,
                                        float ref_x, float ref_y, float ref_z,
                                        float* __restrict__ out_normal);

// ---- STAGE 2: SPFH -----------------------------------------------------------
// One thread per point: darboux_triplet against each of its kFpfhK
// neighbors, histogrammed into 3x11 bins, each block normalized to sum 1.
__global__ void compute_spfh_kernel(int n, const float* __restrict__ xyz, const float* __restrict__ normal,
                                    const int32_t* __restrict__ neighbor_ids,
                                    float* __restrict__ out_spfh);

// ---- STAGE 3: FPFH -----------------------------------------------------------
// One thread per point: FPFH(q) = SPFH(q) + (1/K) * sum_k (1/dist_k) *
// SPFH(neighbor_k), then L1-normalized. Reads the GLOBAL spfh[] array
// (already fully populated by STAGE 2 for every point) at its OWN
// neighbors' rows — no atomics: each thread reads many rows but writes
// only its OWN output row (kernels.cu's header justifies "per-thread-
// private output, read-only neighbor access" over any atomic scheme).
__global__ void compute_fpfh_kernel(int n, const float* __restrict__ spfh,
                                    const int32_t* __restrict__ neighbor_ids,
                                    const float* __restrict__ neighbor_dist,
                                    float* __restrict__ out_fpfh);

// ---- STAGE 4: descriptor matching + ratio test -------------------------------
// One thread per SOURCE point: brute-force nearest+second-nearest TARGET
// descriptor (squared L2 over 33 dims), ratio-test accept/reject.
__global__ void match_correspondences_kernel(int n_src, const float* __restrict__ fpfh_src,
                                             int n_tgt, const float* __restrict__ fpfh_tgt,
                                             uint8_t* __restrict__ out_matched,
                                             int32_t* __restrict__ out_best_idx,
                                             float* __restrict__ out_dist1_sq,
                                             float* __restrict__ out_dist2_sq);

// ---- STAGE 5: RANSAC hypothesis farm ------------------------------------------
// One thread per HYPOTHESIS (grid-stride over kRansacK): sample+prescreen+
// fit+score against the WHOLE gathered correspondence set (corr_src_xyz/
// corr_tgt_xyz, nc entries, gathered by main.cu after STAGE 4).
__global__ void ransac_hypotheses_kernel(int nc, const float* __restrict__ corr_src_xyz,
                                         const float* __restrict__ corr_tgt_xyz,
                                         uint32_t global_seed, int k,
                                         uint8_t* __restrict__ out_valid,
                                         Rigid3* __restrict__ out_transform,
                                         int32_t* __restrict__ out_inlier_count);

// ---- STAGE 6: ICP handoff kernels (02.06 lineage, cited, reimplemented
// compactly) ---------------------------------------------------------------
__global__ void transform_cloud_kernel(int n, const float* __restrict__ src_xyz, Rigid3 T,
                                       float* __restrict__ out_xyz);

// Brute-force nearest TARGET point for every (already transformed) SOURCE
// point, gated by max_dist_m (02.06's launch_find_correspondences, cited).
__global__ void icp_correspondences_kernel(int n_src, const float* __restrict__ cur_xyz,
                                           int n_tgt, const float* __restrict__ tgt_xyz,
                                           float max_dist_m,
                                           int32_t* __restrict__ out_corr_idx,
                                           float* __restrict__ out_corr_dist2);

// Point-to-plane Gauss-Newton accumulation (27 doubles: 21 upper-triangle
// H + 6 g, kernels.cu documents the packed layout) via atomicAdd(double) —
// a DOCUMENTED SIMPLIFICATION vs. 02.06's shared-memory block-tree
// reduction (THEORY.md "The GPU mapping" justifies: at this project's
// n_src ~ a few thousand, contention is cheap and the code is a third the
// size; 02.06 is the right citation for the higher-throughput block-
// reduction alternative once n_src grows into the millions).
__global__ void icp_accumulate_kernel(int n_src, const float* __restrict__ cur_xyz,
                                      const float* __restrict__ tgt_xyz, const float* __restrict__ tgt_normal,
                                      const int32_t* __restrict__ corr_idx,
                                      double* __restrict__ accum27);

#endif // __CUDACC__

// ===========================================================================
// Host-callable launch wrappers (definitions in kernels.cu).
// ===========================================================================
void launch_knn_search(int n, const float* d_xyz, int32_t* d_neighbor_ids, float* d_neighbor_dist);
void launch_estimate_normals(int n, const float* d_xyz, const int32_t* d_neighbor_ids,
                             float ref_x, float ref_y, float ref_z, float* d_out_normal);
void launch_compute_spfh(int n, const float* d_xyz, const float* d_normal, const int32_t* d_neighbor_ids,
                         float* d_out_spfh);
void launch_compute_fpfh(int n, const float* d_spfh, const int32_t* d_neighbor_ids, const float* d_neighbor_dist,
                         float* d_out_fpfh);
void launch_match_correspondences(int n_src, const float* d_fpfh_src, int n_tgt, const float* d_fpfh_tgt,
                                  uint8_t* d_out_matched, int32_t* d_out_best_idx,
                                  float* d_out_dist1_sq, float* d_out_dist2_sq);
void launch_ransac_hypotheses(int nc, const float* d_corr_src_xyz, const float* d_corr_tgt_xyz,
                              uint32_t global_seed, int k,
                              uint8_t* d_out_valid, Rigid3* d_out_transform, int32_t* d_out_inlier_count);
void launch_transform_cloud(int n, const float* d_src_xyz, Rigid3 T, float* d_out_xyz);
void launch_icp_correspondences(int n_src, const float* d_cur_xyz, int n_tgt, const float* d_tgt_xyz,
                                float max_dist_m, int32_t* d_out_corr_idx, float* d_out_corr_dist2);
void launch_icp_accumulate(int n_src, const float* d_cur_xyz, const float* d_tgt_xyz, const float* d_tgt_normal,
                           const int32_t* d_corr_idx, double* d_accum27);

// ===========================================================================
// CPU references (reference_cpu.cpp) — the correctness-oracle twins. All
// pointers are HOST pointers. See reference_cpu.cpp's file header + this
// header's own "Twin-vs-shared ruling" note above for what is shared vs.
// independently reimplemented at each stage.
// ===========================================================================
void knn_search_cpu(int n, const float* xyz, int32_t* neighbor_ids, float* neighbor_dist);

// jacobi_eigen_3x3_cpu — STAGE 1's INDEPENDENT eigensolve (02.09's stricter
// choice, cited): same algorithm FAMILY as any device transcription, its
// own separate typing/loop, so VERIFY(normals) is a real cross-check.
void jacobi_eigen_3x3_cpu(const float cov[6], float eigenvalues[3], float eigenvectors[3][3]);
void estimate_normals_cpu(int n, const float* xyz, const int32_t* neighbor_ids,
                          float ref_x, float ref_y, float ref_z, float* out_normal);

void compute_spfh_cpu(int n, const float* xyz, const float* normal, const int32_t* neighbor_ids, float* out_spfh);
void compute_fpfh_cpu(int n, const float* spfh, const int32_t* neighbor_ids, const float* neighbor_dist,
                      float* out_fpfh);

void match_correspondences_cpu(int n_src, const float* fpfh_src, int n_tgt, const float* fpfh_tgt,
                               uint8_t* out_matched, int32_t* out_best_idx,
                               float* out_dist1_sq, float* out_dist2_sq);

void ransac_hypotheses_cpu(int nc, const float* corr_src_xyz, const float* corr_tgt_xyz,
                           uint32_t global_seed, int k,
                           uint8_t* out_valid, Rigid3* out_transform, int32_t* out_inlier_count);

// ransac_refit_cpu — the FULLY INDEPENDENT double-precision refit oracle
// (02.03's ransac_refine_cpu precedent, cited): its OWN double-precision
// accumulation and its OWN jacobi_eigen_4x4_cpu (not this header's shared
// float rigid_fit_horn) over the supplied inlier correspondence set.
// Compared against the real (float, host, shared-function) refit within a
// documented tolerance — the project's designated non-tautological check
// on the Horn/Jacobi math itself, beyond twin agreement.
void jacobi_eigen_4x4_cpu(const double a_in[10], double eigenvalues[4], double eigenvectors[4][4]);
bool ransac_refit_cpu(int count, const float* src_xyz, const float* tgt_xyz, float R_out[9], float t_out[3]);

void transform_cloud_cpu(int n, const float* src_xyz, const Rigid3& T, float* out_xyz);
void icp_correspondences_cpu(int n_src, const float* cur_xyz, int n_tgt, const float* tgt_xyz,
                             float max_dist_m, int32_t* out_corr_idx, float* out_corr_dist2);
void icp_accumulate_cpu(int n_src, const float* cur_xyz, const float* tgt_xyz, const float* tgt_normal,
                        const int32_t* corr_idx, double accum27[27]);

#endif // PROJECT_KERNELS_CUH
