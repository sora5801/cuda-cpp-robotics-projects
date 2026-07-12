// ===========================================================================
// kernels.cuh — interface for project 02.07
//               NDT scan matching (Autoware-style map localizer)
//               (teaching core: distribution-to-point registration, taught
//                directly against 02.06's ICP correspondence-to-point
//                registration on the SAME scene)
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (staged verification + the multi-resolution
// driver), kernels.cu (the GPU voxel-build and score/gradient/Hessian
// kernels), and reference_cpu.cpp (independent CPU oracle twins + the
// compact ICP contrast baseline). Everything all three must agree on — the
// point-cloud layout, the SE(3) pose representation, the voxel-grid layout,
// and the 28-scalar reduction layout — is defined HERE, once (CLAUDE.md §12).
//
// NDT vs ICP — THE didactic contrast this project exists to teach
// -------------------------------------------------------------------------
// 02.06's ICP matches a transformed source POINT to its nearest TARGET
// POINT, then minimizes a point-to-point (or point-to-plane) residual — the
// objective is a sum of many small, locally flat "V-shaped" cost bowls, one
// per correspondence, and it is only differentiable where the nearest
// neighbor doesn't jump (it can, between iterations, causing local minima).
// NDT throws away individual target points entirely and instead pre-compiles
// the target ("map") cloud into a grid of local GAUSSIANS (one per voxel:
// mean + covariance) — see THEORY.md "physics-first" for why a small patch
// of LiDAR-scanned surface really is well modeled as Gaussian-distributed
// noise around a local plane. A transformed source point then scores against
// the SMOOTH, closed-form density of whichever voxel it lands in — no
// nearest-neighbor SEARCH at all (voxel lookup is O(1) direct indexing), and
// the objective is a sum of smooth Gaussian bumps instead of a sum of
// V-shaped correspondence bowls. THEORY.md "physics-first" and "the math"
// derive both consequences: (a) no correspondence search, and (b) a smoother
// (wider-basin, especially at coarse voxel size) optimization landscape.
// This project runs BOTH algorithms from the SAME perturbed initial poses on
// the SAME scene and MEASURES which claim actually holds (README "Expected
// output", main.cu STAGE G).
//
// POINT CLOUD LAYOUT — float* xyz, interleaved, meters — 02.06's exact
// convention (cited), itself docs/SYSTEM_DESIGN.md §3.6's flattened
// `PointCloud` message shape:  xyz[i*3+0]=x, xyz[i*3+1]=y, xyz[i*3+2]=z.
//
// SE(3) POSE / LOCAL PARAMETERIZATION — Rigid3 { float R[9] row-major;
// float t[3] } and the LOCAL 6-vector delta = [omega(3); v(3)] retraction
// (R_new = Exp(omega)*R, t_new = t+v) are 01.17's EXACT construction, REUSED
// HERE VERBATIM (cited, reimplemented per the repo's self-containment rule,
// CLAUDE.md §4 — never referenced across project folders at build/run time).
// The estimate is T_map_scan: x_map = R * x_scan + t (02.06's T_target_source
// naming convention, applied here with "map" playing target's role and
// "scan" playing source's role — the Autoware localizer's exact framing).
//
// Why this header is CUDA-qualifier-DUAL (NDT_HD), unlike 02.01/02.06's
// qualifier-free convention: the SE(3) retraction math (so3_exp, skew3,
// mat3_vec, mat3_mul, retract) and the 3x3 symmetric eigensolve
// (jacobi_eigen_symmetric3) are each called from BOTH a __global__ kernel
// (kernels.cu) and plain host code (reference_cpu.cpp's independent
// accumulation loops, main.cu's host orchestration) — duplicating closed-form
// linear algebra by hand a second time would be pure transcription (01.17's
// exact argument, cited). The NDT_HD macro (below) is 01.17's CALIB_HD,
// renamed. Per the twin-independence ruling this sharing requires (see
// reference_cpu.cpp's file header): the ACCUMULATION LOOPS that fold many
// points into one 28-scalar record are written independently in
// reference_cpu.cpp, and this project carries INDEPENDENT gates (jacobian
// check via central differences; convergence/accuracy against Python-
// generated ground truth) that would catch a bug hiding inside this shared
// math even though the twin comparison alone could not.
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cstdint>   // uint32_t, uint64_t
#include <cmath>     // sqrtf/sinf/cosf/fabsf — used by both host and device math below
#include <vector>    // std::vector — the CPU-side voxel grid representation (NdtVoxelCPU records)

// ---------------------------------------------------------------------------
// NDT_HD — "__host__ __device__" under nvcc, nothing under cl.exe (01.17's
// CALIB_HD macro, cited and renamed). reference_cpu.cpp is compiled by
// cl.exe and must never see a CUDA keyword; this is the standard trick that
// lets the handful of shared SE(3)/eigensolve primitives below compile as
// plain host functions there while ALSO compiling as dual host/device
// functions kernels.cu's __global__ kernels call directly.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define NDT_HD __host__ __device__
#else
#define NDT_HD
#endif

// ===========================================================================
// Rigid3 — a rigid-body transform, passed BY VALUE to every kernel/function
// that needs "the current pose estimate" (02.06/01.17's exact convention and
// argument for why: T changes every optimizer iteration, so a __constant__
// upload would be pure overhead for 48 bytes — see 02.06's kernels.cuh for
// the full three-way "how many threads read a few floats" spectrum).
// x_map = R * x_scan + t.
// ===========================================================================
struct Rigid3 {
    float R[9];   // row-major 3x3 rotation (orthonormal by construction —
                  // every update goes through so3_exp()'s exact Rodrigues
                  // formula, never an additive approximation, so it cannot
                  // drift off SO(3))
    float t[3];   // translation (m), MAP frame
};

constexpr Rigid3 kIdentityRigid3{
    { 1.0f, 0.0f, 0.0f,  0.0f, 1.0f, 0.0f,  0.0f, 0.0f, 1.0f },
    { 0.0f, 0.0f, 0.0f }
};

// ===========================================================================
// Shared SE(3) primitives — 01.17's exact formulas (cited; reimplemented
// here per this project's own copy, CLAUDE.md §4 self-containment rule).
// See 01.17's kernels.cuh for the full derivations; THEORY.md "The math"
// repeats the parts specific to how this project USES them (the NDT
// Jacobian chain rule through R*x+t into the Mahalanobis form).
// ===========================================================================

// skew3 — skew-symmetric (cross-product) matrix [v]_x, row-major.
NDT_HD inline void skew3(const float v[3], float S[9])
{
    S[0] =  0.0f;  S[1] = -v[2];  S[2] =  v[1];
    S[3] =  v[2];  S[4] =  0.0f;  S[5] = -v[0];
    S[6] = -v[1];  S[7] =  v[0];  S[8] =  0.0f;
}

// mat3_vec — out = R * p, row-major 3x3 times 3-vector.
NDT_HD inline void mat3_vec(const float R[9], const float p[3], float out[3])
{
    out[0] = R[0] * p[0] + R[1] * p[1] + R[2] * p[2];
    out[1] = R[3] * p[0] + R[4] * p[1] + R[5] * p[2];
    out[2] = R[6] * p[0] + R[7] * p[1] + R[8] * p[2];
}

// mat3_mul — out = A * B, row-major 3x3 times 3x3. out must not alias A/B.
NDT_HD inline void mat3_mul(const float A[9], const float B[9], float out[9])
{
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) {
            float acc = 0.0f;
            for (int k = 0; k < 3; ++k) acc += A[r * 3 + k] * B[k * 3 + c];
            out[r * 3 + c] = acc;
        }
}

// so3_exp — SO(3) exponential map (exact Rodrigues' formula): given an
// axis-angle vector omega (rad), return R = Exp([omega]_x). Small-angle
// branch avoids the 0/0 as theta -> 0 (01.17's exact bound: first-order
// Taylor error is O(theta^2), negligible below the 1e-8 threshold at the
// theta values this project's retraction ever calls this with).
NDT_HD inline void so3_exp(const float omega[3], float R[9])
{
    const float theta2 = omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2];
    const float theta  = sqrtf(theta2);

    float S[9];
    skew3(omega, S);

    if (theta < 1.0e-8f) {
        R[0] = 1.0f + S[0]; R[1] = S[1];        R[2] = S[2];
        R[3] = S[3];        R[4] = 1.0f + S[4]; R[5] = S[5];
        R[6] = S[6];        R[7] = S[7];        R[8] = 1.0f + S[8];
        return;
    }

    const float a = sinf(theta) / theta;
    const float b = (1.0f - cosf(theta)) / theta2;

    float S2[9];
    mat3_mul(S, S, S2);
    for (int i = 0; i < 9; ++i) {
        const float identity_i = (i == 0 || i == 4 || i == 8) ? 1.0f : 0.0f;
        R[i] = identity_i + a * S[i] + b * S2[i];
    }
}

// retract — apply a local 6-vector delta = [omega(3); v(3)] to T (01.17's
// decoupled SO(3) x R^3 retraction, cited): R_new = Exp(omega) * R (LEFT
// perturbation, exact Rodrigues update), t_new = t + v.
NDT_HD inline void retract(const Rigid3& T, const double delta[6], Rigid3& out)
{
    const float omega[3] = { static_cast<float>(delta[0]), static_cast<float>(delta[1]), static_cast<float>(delta[2]) };
    float dR[9];
    so3_exp(omega, dR);
    mat3_mul(dR, T.R, out.R);
    out.t[0] = T.t[0] + static_cast<float>(delta[3]);
    out.t[1] = T.t[1] + static_cast<float>(delta[4]);
    out.t[2] = T.t[2] + static_cast<float>(delta[5]);
}

// rotation_angle_deg / translation_error_m — 01.17's exact reporting
// helpers (cited): geodesic rotation distance and Euclidean translation
// distance between a recovered Rigid3 and ground truth. Host-only usage
// (main.cu's gates) but harmless to mark NDT_HD for consistency.
NDT_HD inline float rotation_angle_deg(const float R[9], const float R_gt[9])
{
    float Rt[9] = { R[0], R[3], R[6],  R[1], R[4], R[7],  R[2], R[5], R[8] };
    float Rerr[9];
    mat3_mul(Rt, R_gt, Rerr);
    float trace = Rerr[0] + Rerr[4] + Rerr[8];
    float c = (trace - 1.0f) * 0.5f;
    if (c > 1.0f) c = 1.0f;
    if (c < -1.0f) c = -1.0f;
    return acosf(c) * (180.0f / 3.14159265358979323846f);
}

NDT_HD inline float translation_error_m(const float t[3], const float t_gt[3])
{
    const float dx = t[0] - t_gt[0], dy = t[1] - t_gt[1], dz = t[2] - t_gt[2];
    return sqrtf(dx * dx + dy * dy + dz * dz);
}

// ===========================================================================
// Map / voxel-grid geometry — SINGLE-SOURCED so scripts/make_synthetic.py's
// scene (documented in that script's header — an L-shaped corridor opening
// into a room, the corridor being the project's deliberate degeneracy axis,
// THEORY.md "physics-first"), main.cu's grid allocation, and kernels.cu's
// device-side voxel indexing all agree on the same bounding box.
//
// The grid is DENSE (a flat array indexed directly by (ix,iy,iz)), NOT
// hashed like 02.01's spatial hash table. Justification (THEORY.md "The GPU
// mapping" repeats this): 02.01 hashes because a raw LiDAR SCAN's occupied
// voxel set is sparse and unbounded in principle (a streaming point cloud
// could, in general, span an unknown extent). An NDT MAP is the opposite: it
// is built ONCE from a bounded, known survey area (here ~17 x 9 x 3.5 m),
// so even the FINE 1.0 m grid is only 17*9*4 = 612 voxels — three orders of
// magnitude smaller than a hash table would ever need to be at this scale.
// Direct indexing is O(1) with no probe loop, no collision handling, and no
// atomicCAS insert path: strictly simpler AND faster than hashing here. A
// city-scale HD map (kilometers, not meters) would flip this trade back
// toward hashing/tiling — README "Limitations" says so explicitly.
// ===========================================================================
constexpr float kMapOriginX = -0.5f, kMapOriginY = -4.5f, kMapOriginZ = -0.5f;  // m, grid's (0,0,0) voxel corner
constexpr float kMapSizeX = 17.0f, kMapSizeY = 9.0f, kMapSizeZ = 3.5f;          // m, grid extent (covers the scene + margin)

constexpr float kLeafCoarse = 2.0f;   // m — first multi-resolution stage (wide basin, THEORY.md)
constexpr float kLeafFine   = 1.0f;   // m — second stage (sharp, accurate optimum)

// A voxel needs enough points to estimate a meaningful 3x3 covariance (6
// unique numbers) without being dominated by sampling noise — the same
// "small-sample statistics are unstable" caution 02.06's PCA normal
// estimation (kPcaK=16 neighbors) makes, applied here to a fixed spatial
// bucket instead of a k-NN neighborhood. Voxels with fewer points than this
// are marked INVALID and contribute nothing to the score (the NDT analogue
// of ICP's correspondence-rejection gate, 02.06's -1 corr_idx).
constexpr int kMinPointsPerVoxel = 5;

// Eigenvalue flooring ratio for covariance regularization (THEORY.md "The
// problem — physics & engineering first" derives WHY a flat wall patch
// needs this): a perfectly flat surface has near-zero variance ALONG ITS
// NORMAL, so the raw sample covariance is near-singular (smallest eigenvalue
// -> 0) — inverting it directly would blow the Mahalanobis distance up
// arbitrarily for any point with even a tiny offset along that direction.
// Flooring the smallest eigenvalue to this fraction of the largest one (a
// direct reimplementation of PCL/Autoware's NDT covariance regularization,
// cited by name in THEORY.md) keeps the voxel's Gaussian numerically well-
// conditioned while still expressing "this voxel is planar" through the
// SHAPE (not the raw magnitude) of its remaining anisotropy.
constexpr float kEigenFloorRatio = 0.01f;

// 3x3 symmetric Jacobi eigensolve sweep count — 02.06's exact number and
// reasoning (cited): "8 sweeps of a 3x3 symmetric Jacobi eigensolve is
// generous overkill for 3x3 (textbook practice converges in 3-5 sweeps)".
constexpr int kJacobiSweeps3 = 8;

// blocks_for — ceil(count/threads), the repo-wide idiom (02.06/08.01/01.17,
// cited). Declared here (not just in kernels.cu) so main.cu can compute the
// IDENTICAL block count when sizing block_partials download buffers.
inline int blocks_for(int count, int threads)
{
    return (count + threads - 1) / threads;
}

// voxel_index — map a MAP-FRAME point to its dense-grid linear voxel index,
// or -1 if the point falls outside the grid's bounding box. floor(), not
// truncation — 02.01's exact voxel_coord() argument (cited): naive
// (int)(p/leaf) truncates TOWARD ZERO and misplaces points on the negative
// side of an axis; std::floor gets every sign right unconditionally.
// Linear layout: idx = ix + nx*(iy + ny*iz) (x fastest-varying — the layout
// every build/lookup kernel and CPU twin below shares).
NDT_HD inline int voxel_index(const float p[3],
                              float origin_x, float origin_y, float origin_z,
                              float leaf, int nx, int ny, int nz)
{
    const int ix = static_cast<int>(floorf((p[0] - origin_x) / leaf));
    const int iy = static_cast<int>(floorf((p[1] - origin_y) / leaf));
    const int iz = static_cast<int>(floorf((p[2] - origin_z) / leaf));
    if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz) return -1;
    return ix + nx * (iy + ny * iz);
}

// grid_dims_for_leaf — how many voxels per axis a given leaf size needs to
// cover the fixed map bounding box above (ceiling division: a partial voxel
// at the far edge still needs a whole cell). Called once per resolution
// level, by both main.cu (to size device arrays) and reference_cpu.cpp (to
// size its own independent std::vector<Voxel>).
inline void grid_dims_for_leaf(float leaf, int& nx, int& ny, int& nz)
{
    nx = static_cast<int>(std::ceil(kMapSizeX / leaf));
    ny = static_cast<int>(std::ceil(kMapSizeY / leaf));
    nz = static_cast<int>(std::ceil(kMapSizeZ / leaf));
}

// ===========================================================================
// jacobi_eigen_symmetric3 — classic cyclic Jacobi eigenvalue algorithm
// (Golub & Van Loan), specialized to N=3 — 02.06's PCA-normal Jacobi
// solver, cited and reimplemented here per this project's own copy
// (self-containment rule) for a DIFFERENT job: regularizing and inverting a
// voxel's covariance instead of extracting a surface normal. NDT_HD because
// the voxel-finalize kernel calls this once per voxel on the GPU.
//
// A (IN/OUT) — 3x3 symmetric, row-major; overwritten with the diagonalized
//   matrix (the diagonal on return holds the eigenvalues, unordered).
// V (OUT) — 3x3, row-major; COLUMNS are the corresponding eigenvectors.
// ===========================================================================
NDT_HD inline void jacobi_eigen_symmetric3(float A[9], float V[9])
{
    for (int i = 0; i < 9; ++i) V[i] = (i == 0 || i == 4 || i == 8) ? 1.0f : 0.0f;

    for (int sweep = 0; sweep < kJacobiSweeps3; ++sweep) {
        for (int p = 0; p < 3; ++p) {
            for (int q = p + 1; q < 3; ++q) {
                const float apq = A[p * 3 + q];
                if (fabsf(apq) < 1.0e-12f) continue;   // already (numerically) zero
                const float app = A[p * 3 + p], aqq = A[q * 3 + q];
                const float theta = (aqq - app) / (2.0f * apq);
                const float t = (theta >= 0.0f ? 1.0f : -1.0f) / (fabsf(theta) + sqrtf(theta * theta + 1.0f));
                const float c = 1.0f / sqrtf(t * t + 1.0f);
                const float s = t * c;

                A[p * 3 + p] = c * c * app - 2.0f * s * c * apq + s * s * aqq;
                A[q * 3 + q] = s * s * app + 2.0f * s * c * apq + c * c * aqq;
                A[p * 3 + q] = A[q * 3 + p] = 0.0f;
                for (int k = 0; k < 3; ++k) {
                    if (k == p || k == q) continue;
                    const float akp = A[k * 3 + p], akq = A[k * 3 + q];
                    A[k * 3 + p] = A[p * 3 + k] = c * akp - s * akq;
                    A[k * 3 + q] = A[q * 3 + k] = s * akp + c * akq;
                }
                for (int k = 0; k < 3; ++k) {
                    const float vkp = V[k * 3 + p], vkq = V[k * 3 + q];
                    V[k * 3 + p] = c * vkp - s * vkq;
                    V[k * 3 + q] = s * vkp + c * vkq;
                }
            }
        }
    }
}

// regularize_and_invert_cov3 — the voxel-finalize step: given a RAW sample
// covariance (3x3 symmetric, row-major, from the two-pass accumulation —
// kernels.cu/reference_cpu.cpp's build routines), eigendecompose it, FLOOR
// the smallest eigenvalue(s) to kEigenFloorRatio * (largest eigenvalue),
// then reconstruct the INVERSE directly from the (orthonormal) eigenbasis:
// inv(C_reg) = V * diag(1/e') * V^T — cheaper than a general 3x3 inverse
// AND regularization in one step, since the eigendecomposition already
// diagonalizes C. Returns true if flooring actually changed an eigenvalue
// (the [info] "how many voxels needed regularizing" count main.cu reports —
// THEORY.md ties this to "which voxels are flat" honestly).
NDT_HD inline bool regularize_and_invert_cov3(const float cov_raw[6], float inv_cov_out[6])
{
    // cov_raw packed [xx,xy,xz,yy,yz,zz] -> full row-major 3x3 for the solver.
    float A[9] = {
        cov_raw[0], cov_raw[1], cov_raw[2],
        cov_raw[1], cov_raw[3], cov_raw[4],
        cov_raw[2], cov_raw[4], cov_raw[5]
    };
    float V[9];
    jacobi_eigen_symmetric3(A, V);   // A's diagonal is now the eigenvalues

    float e[3] = { A[0], A[4], A[8] };
    float e_max = e[0];
    if (e[1] > e_max) e_max = e[1];
    if (e[2] > e_max) e_max = e[2];
    const float e_floor = e_max * kEigenFloorRatio;

    bool regularized = false;
    for (int i = 0; i < 3; ++i) {
        if (e[i] < e_floor) { e[i] = e_floor; regularized = true; }
    }

    // inv(C_reg) = V * diag(1/e) * V^T, row-major. V's COLUMNS are
    // eigenvectors (jacobi_eigen_symmetric3's convention), so
    // inv(C_reg)[r][c] = sum_k V[r][k] * (1/e[k]) * V[c][k].
    for (int r = 0; r < 3; ++r) {
        for (int c = r; c < 3; ++c) {
            float acc = 0.0f;
            for (int k = 0; k < 3; ++k) acc += V[r * 3 + k] * (1.0f / e[k]) * V[c * 3 + k];
            // Pack upper triangle [xx,xy,xz,yy,yz,zz] to match cov_raw's layout.
            if (r == 0 && c == 0) inv_cov_out[0] = acc;
            else if (r == 0 && c == 1) inv_cov_out[1] = acc;
            else if (r == 0 && c == 2) inv_cov_out[2] = acc;
            else if (r == 1 && c == 1) inv_cov_out[3] = acc;
            else if (r == 1 && c == 2) inv_cov_out[4] = acc;
            else if (r == 2 && c == 2) inv_cov_out[5] = acc;
        }
    }
    return regularized;
}

// sym3_quad_form / sym3_vec — small helpers shared by the assembly kernel
// and its CPU twin: q^T * M * q (scalar) and M * q (3-vector), where M is a
// packed-symmetric [xx,xy,xz,yy,yz,zz] 3x3. Used with M = inv_cov (the
// Mahalanobis form, THEORY.md "the math") in both the score and gradient.
NDT_HD inline float sym3_quad_form(const float M[6], const float q[3])
{
    return M[0] * q[0] * q[0] + M[3] * q[1] * q[1] + M[5] * q[2] * q[2]
         + 2.0f * (M[1] * q[0] * q[1] + M[2] * q[0] * q[2] + M[4] * q[1] * q[2]);
}

NDT_HD inline void sym3_vec(const float M[6], const float q[3], float out[3])
{
    out[0] = M[0] * q[0] + M[1] * q[1] + M[2] * q[2];
    out[1] = M[1] * q[0] + M[3] * q[1] + M[4] * q[2];
    out[2] = M[2] * q[0] + M[4] * q[1] + M[5] * q[2];
}

// ===========================================================================
// NDT mixture-model constants (d1, d2) — the Biber & Straßer (2003) /
// Magnusson (2009 thesis) derivation THEORY.md "the math" reproduces in
// full, in the EXACT parameterization Autoware's ndt_scan_matcher and PCL's
// NormalDistributionsTransform use (named here so a learner can cross-
// reference either codebase directly):
//
//   p(x) = c1 * N(x; mu, Sigma) + c2                (Gaussian-plus-uniform-
//                                                     outlier mixture)
//   score(x) = -d1 * exp(-d2/2 * Mahalanobis(x)^2)   (the smooth, closed-form
//                                                      approximation to
//                                                      -log p(x) NDT actually
//                                                      optimizes)
//
// c1 = 10*(1-outlier_ratio), c2 = outlier_ratio / resolution^3 (a uniform
// density over one voxel's volume — resolution enters HERE, which is why
// d1/d2 are recomputed per resolution level, THEORY.md explains the units).
// d1, d2 are then solved by matching -log(p(x)) at two points (x=mu, and
// the point where the Gaussian's exponent is -1/2) — see THEORY.md for the
// full two-equation solve; the closed form below is the textbook result.
//
// outlier_ratio here is an ASSUMED robustness parameter (a tuning knob a
// real localizer sets WITHOUT knowing the true scene outlier fraction), NOT
// required to equal scripts/make_synthetic.py's TRUE injected outlier
// fraction (kTrueOutlierFraction, documented in that script) — README
// "Limitations" and THEORY.md "numerical considerations" discuss this
// deliberate mismatch honestly; it is how a real system is actually tuned.
// ===========================================================================
constexpr double kAssumedOutlierRatio = 0.40;   // the d1/d2 robustness knob (NOT the true injected fraction)

inline void ndt_compute_d1_d2(double resolution_m, double outlier_ratio, double& d1, double& d2)
{
    // Derivation (THEORY.md "the math" reproduces this in full, with the
    // sign check spelled out): let target(m) = -log(p(x)) shifted so it
    // vanishes as m->infinity (p(x) -> c2, the pure-outlier density).
    // Matching -d1*exp(-d2/2*m) to that shifted target AT m=0 and m=1
    // (one Mahalanobis sigma) gives, in closed form:
    //   d1 = log((c1+c2)/c2)                          > 0
    //   d2 = -2*log( log((c1/c2)*exp(-0.5)+1) / d1 )   > 0
    // Both d1 and d2 MUST be positive: score(x) = -d1*exp(-d2/2*m) has to
    // be MOST NEGATIVE at m=0 (best possible alignment) and rise toward 0
    // as m grows -- a negative d1 would flip that into a function that gets
    // BETTER the worse the alignment, silently inverting every gradient
    // and Hessian sign downstream. (This is exactly the bug this comment
    // now guards against — verified numerically against an independent
    // from-scratch re-derivation before being written down here.)
    const double c1 = 10.0 * (1.0 - outlier_ratio);
    const double c2 = outlier_ratio / (resolution_m * resolution_m * resolution_m);
    const double d3 = -std::log(c2);                                    // = log(1/c2), > 0
    d1 = std::log(c1 + c2) + d3;                                        // = log((c1+c2)/c2), > 0
    d2 = -2.0 * std::log((std::log(c1 * std::exp(-0.5) + c2) + d3) / d1);
}

// ===========================================================================
// 28-scalar reduction — 01.17's EXACT packing (cited): a 6x6 symmetric
// Hessian's 21 upper-triangle entries (hidx() below, row_start table),
// followed by the 6-entry gradient g, followed by ONE more scalar: the
// summed NDT score. One reduction produces everything a Newton/LM step (and
// the score_sanity gate) needs. Parameter order [wx,wy,wz,vx,vy,vz] —
// rotation-first, matching the retract() delta ordering above.
// ===========================================================================
NDT_HD inline int hidx(int i, int j)
{
    const int row_start[6] = { 0, 6, 11, 15, 18, 20 };
    return row_start[i] + (j - i);   // caller guarantees i <= j <= 5
}

constexpr int kReduceWidth      = 28;   // 21 (H upper triangle) + 6 (g) + 1 (score)
constexpr int kThreadsAssemble  = 128;  // 01.17/02.06's kThreadsReduce default
constexpr int kThreadsVoxel     = 256;  // repo default for point-parallel voxel-build kernels

// ---------------------------------------------------------------------------
// cholesky6_solve_flat — HOST-ONLY. Solves (H + lambda*diag(|H|)) delta =
// -g via Cholesky-Crout (H_damped = L L^T) plus forward/back substitution.
//
// SIGN-SAFE SCALED damping, NOT 01.17/02.06's Marquardt (1+lambda)*diag(H)
// variant, and NOT a flat lambda*I either — a DELIBERATE, MEASURED choice
// (THEORY.md "numerical considerations" tells the full story; summary
// here): 02.06's ICP and 01.17's calibration both assemble H = J^T J,
// guaranteed positive-semidefinite, so Marquardt's (1+lambda)*H_ii scaling
// is safe (it always PRESERVES H_ii's sign). NDT's Hessian (THEORY.md "the
// math" derives it) carries an EXTRA term from the Gaussian's own
// exponential curvature that can make H INDEFINITE far from the optimum —
// Marquardt scaling on a NEGATIVE diagonal entry would make it MORE
// negative, the wrong direction. A first version of this project used a
// FLAT lambda*I add instead (unconditionally sign-safe), but a flat add is
// only meaningful relative to each parameter's OWN natural scale — and
// NDT's 6 parameters do not share one: rotation entries carry units of
// "meters per radian" (via dy/domega ~ |R*x|, x ~ meters away) while
// translation entries are dimensionless-in-meters (dy/dv = I). Measured on
// this project's cohort, a flat lambda left ROTATION converging cleanly
// while TRANSLATION stalled within millimeters of its start (the same
// lambda was enormous relative to the translation block's naturally
// smaller diagonal, negligible relative to rotation's naturally larger
// one) — a real bug this project's own convergence gate caught during
// development. The fix combines both virtues: scale the damping added to
// row i by |H_ii| (Marquardt's per-parameter adaptivity) but ADD it as a
// positive quantity rather than multiply (Levenberg's sign-safety) —
// A_ii += lambda * max(|H_ii|, floor). This is always a POSITIVE addition
// regardless of H_ii's sign, so it still provably restores positive-
// definiteness for large enough lambda (Levenberg's guarantee), while
// remaining properly scaled per parameter (Marquardt's benefit).
// ---------------------------------------------------------------------------
inline bool cholesky6_solve_flat(const double H21[21], const double g6[6], double lambda, double out_delta[6])
{
    double A[6][6];
    for (int i = 0; i < 6; ++i)
        for (int j = i; j < 6; ++j) {
            const double hij = H21[hidx(i, j)];
            A[i][j] = hij;
            A[j][i] = hij;
        }
    for (int i = 0; i < 6; ++i) {
        const double diag_scale = std::fabs(A[i][i]) > 1.0e-9 ? std::fabs(A[i][i]) : 1.0;
        A[i][i] += lambda * diag_scale;   // sign-safe, per-parameter scaled damping (see header comment)
    }

    double L[6][6] = {};
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = A[i][j];
            for (int k = 0; k < j; ++k) sum -= L[i][k] * L[j][k];
            if (i == j) {
                if (sum <= 0.0) return false;   // not SPD at this lambda -- caller backs off
                L[i][i] = std::sqrt(sum);
            } else {
                L[i][j] = sum / L[j][j];
            }
        }
    }

    double y[6];
    for (int i = 0; i < 6; ++i) {
        double sum = -g6[i];
        for (int k = 0; k < i; ++k) sum -= L[i][k] * y[k];
        y[i] = sum / L[i][i];
    }
    for (int i = 5; i >= 0; --i) {
        double sum = y[i];
        for (int k = i + 1; k < 6; ++k) sum -= L[k][i] * out_delta[k];
        out_delta[i] = sum / L[i][i];
    }
    return true;
}

// Newton/LM hyperparameters — single-sourced (main.cu's GPU-orchestrated
// trajectory, reference_cpu.cpp's independent trajectory, and the
// multi-resolution schedule all step identically). THEORY.md "the
// algorithm" documents the coarse->fine choice: MORE iterations are allowed
// at fine resolution because coarse-stage steps are typically large (basin-
// finding), fine-stage steps are typically small refinements.
constexpr int    kMaxItersCoarse    = 12;
constexpr int    kMaxItersFine      = 15;
// (Measured during development: raising these to 20/25 did not change the
// cohort's convergence count at all — confirming the un-converged trials
// stall at genuine local minima, not from running out of iterations. Kept
// at the smaller, cheaper values; THEORY.md "numerical considerations"
// reports this negative result honestly.)

// kLambdaInit — the STARTING damping value fed to cholesky6_solve_flat's
// per-parameter-scaled A_ii += lambda*|H_ii| update (that function's own
// header explains the scaling; this constant is just the dimensionless
// starting point, since the scaling by |H_ii| already absorbs each
// parameter's natural units — no separate "scale by the Hessian's overall
// magnitude" step is needed, unlike a flat-lambda scheme would require).
// Small (well under 1) so the first step is close to a plain Newton step
// when H is well-conditioned, and grows via kLambdaUp when it is not.
constexpr double kLambdaInit        = 1.0e-2;
constexpr double kLambdaUp          = 10.0;
constexpr double kLambdaDown        = 0.5;
constexpr double kLambdaMin         = 1.0e-6;
constexpr double kConvergeDeltaNorm = 1.0e-7;   // ||delta|| (mixed rad/m, 01.17's exact caveat)

// kMaxAcceptRejectRetries — how many times ONE outer Newton iteration will
// re-damp (lambda *= kLambdaUp) the SAME H/g and retry before giving up on
// this iteration ENTIRELY (main.cu's run_ndt_stage_gpu and
// reference_cpu.cpp's run_ndt_stage_cpu both `break` the OUTER loop, not
// just this inner one, once this cap is hit with no accepted step --
// ending the whole coarse/fine stage early). Named (not a bare literal) so
// this project's tuning pass could sweep it directly: measured NO change
// to cohort convergence up to 40 retries (THEORY.md "numerical
// considerations" reports this honestly) -- the trials that give up early
// are giving up at a genuine stationary point, not an under-explored one.
constexpr int kMaxAcceptRejectRetries = 8;

// "Converged" classification for the basin/convergence gates — final pose
// within these bounds of ground truth. THEORY.md "how we verify
// correctness" explains the choice: at ~1-2 m voxels and 2 cm range noise,
// centimeter-level, not millimeter-level, is the honest achievable bound
// for a grid-based method (unlike 01.17's exact-geometry camera calibration).
constexpr float kConvergedTransM = 0.08f;   // 8 cm
constexpr float kConvergedRotDeg = 3.0f;    // 3 deg

// ===========================================================================
// NdtGridGPU — device-side voxel grid, passed BY VALUE (a handful of
// pointers + 7 scalars, ~80 bytes — 02.06/01.17's "small POD by value"
// convention). main.cu owns the allocations; kernels.cu's launchers only
// read/write through them. Per-voxel arrays are length nx*ny*nz, indexed by
// voxel_index() above (dense, x-fastest-varying).
// ===========================================================================
struct NdtGridGPU {
    float origin_x, origin_y, origin_z;   // m, grid (0,0,0) voxel corner (kMapOrigin*)
    float leaf;                           // m, this grid's resolution (kLeafCoarse or kLeafFine)
    int   nx, ny, nz;                     // voxel counts per axis (grid_dims_for_leaf)

    int*    count;         // [capacity] points accumulated into this voxel
    double* sum_xyz;       // [capacity*3] running position sum (double — see kernels.cu build-kernel
                            //  comment for why atomicAdd(double*) is used instead of float)
    double* sum_cov6;      // [capacity*6] running centered outer-product sum (pass 2), packed
                            //  [xx,xy,xz,yy,yz,zz], double
    float*  mean;          // [capacity*3] finalized mean (valid once count>0)
    float*  inv_cov6;      // [capacity*6] finalized REGULARIZED inverse covariance, packed as above
    unsigned char* valid;  // [capacity] 1 iff count>=kMinPointsPerVoxel and finalize succeeded
};

inline int ndt_grid_capacity(const NdtGridGPU& g) { return g.nx * g.ny * g.nz; }

// ===========================================================================
// GPU kernels (kernels.cu). __global__ signatures are __CUDACC__-fenced —
// only nvcc parses them, matching 01.17/02.06's established pattern.
// ===========================================================================
#ifdef __CUDACC__

// ndt_voxel_accum_sum_kernel — PASS 1 of the two-pass mean/covariance build
// (THEORY.md "numerical considerations" justifies two-pass over Welford/
// naive-one-pass): point-parallel, atomicAdd each map point's xyz into its
// voxel's running sum (+1 to count). Full documentation with kernels.cu.
__global__ void ndt_voxel_accum_sum_kernel(int n_map, const float* __restrict__ map_xyz, NdtGridGPU grid);

// ndt_finalize_means_kernel — voxel-parallel: mean = sum/count.
__global__ void ndt_finalize_means_kernel(NdtGridGPU grid);

// ndt_voxel_accum_cov_kernel — PASS 2: point-parallel, atomicAdd each map
// point's CENTERED outer product (x-mean)(x-mean)^T into its voxel's
// running covariance sum. Requires finalized means (must run after
// ndt_finalize_means_kernel).
__global__ void ndt_voxel_accum_cov_kernel(int n_map, const float* __restrict__ map_xyz, NdtGridGPU grid);

// ndt_finalize_cov_kernel — voxel-parallel: cov = sum_cov6/(count-1),
// eigen-regularize + invert (regularize_and_invert_cov3), mark valid.
// d_regularized_count[0] is atomicAdd-incremented once per voxel whose
// smallest eigenvalue needed flooring (the [info] honesty count).
__global__ void ndt_finalize_cov_kernel(NdtGridGPU grid, unsigned int* __restrict__ d_regularized_count);

// ndt_assemble_kernel — the project's central NEW GPU concept: POINT-
// parallel scoring of every (transformed) scan point against its voxel's
// Gaussian, block-tree-reduced into ONE 28-scalar [H21|g6|score] record per
// block (01.17's EXACT reduction shape, cited). Full documentation
// (thread mapping, shared-memory layout, the chain-rule formulas) sits with
// the definition in kernels.cu.
__global__ void ndt_assemble_kernel(const float* __restrict__ scan_xyz, int n_scan,
                                    Rigid3 T, NdtGridGPU grid,
                                    double d1, double d2,
                                    float* __restrict__ block_partials);

#endif // __CUDACC__

// ---------------------------------------------------------------------------
// Host launch wrappers (definitions in kernels.cu; declared outside the
// __CUDACC__ fence so any translation unit may call them).
// ---------------------------------------------------------------------------

// launch_build_ndt_grid — runs all four voxel-build kernels above in order
// (accumulate sums -> finalize means -> accumulate cov -> finalize cov)
// against an ALREADY-ALLOCATED, ALREADY-ZEROED grid (main.cu owns
// allocation/zeroing via cudaMemset so repeated calls at different
// resolutions are explicit about their own buffers). Returns via
// *out_regularized_count how many voxels needed eigenvalue flooring.
void launch_build_ndt_grid(int n_map, const float* d_map_xyz, NdtGridGPU grid,
                           unsigned int* out_regularized_count);

// launch_ndt_assemble — runs ndt_assemble_kernel and returns the number of
// blocks launched (== blocks_for(n_scan, kThreadsAssemble)), so the caller
// knows how many kReduceWidth-wide rows to download and sum in double.
int launch_ndt_assemble(const float* d_scan_xyz, int n_scan, Rigid3 T, NdtGridGPU grid,
                        double d1, double d2, float* d_block_partials);

// ===========================================================================
// CPU references (reference_cpu.cpp) — see that file's header for the full
// independence ruling (what is shared, what is independently reimplemented,
// and which gates remain blind-spot coverage for the shared math).
// ===========================================================================

// One voxel's build state, host-side (used by both the independent CPU
// grid-builder and main.cu's readback of the GPU grid for the twin gate).
struct NdtVoxelCPU {
    int count = 0;
    double mean[3] = { 0.0, 0.0, 0.0 };
    double cov6[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };   // raw (unregularized), for the twin comparison
    float inv_cov6[6] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    bool valid = false;
};

// build_ndt_grid_cpu — the INDEPENDENT twin of launch_build_ndt_grid: its
// OWN two-pass accumulation loop (sequential over points, DOUBLE precision
// throughout, no atomics/reduction-order questions at all — see the file
// header for why this makes it the more-precise oracle by construction),
// using ONLY the shared voxel_index()/regularize_and_invert_cov3() formulas
// (the data-layout contract, not the algorithm).
void build_ndt_grid_cpu(int n_map, const float* map_xyz, float leaf, int nx, int ny, int nz,
                        std::vector<NdtVoxelCPU>& out);

// ndt_assemble_cpu — the INDEPENDENT twin of ndt_assemble_kernel: same
// shared voxel_index/Mahalanobis math, but its own SEQUENTIAL accumulation
// loop (no block-reduction tree at all) directly into double H21/g6/score
// accumulators.
void ndt_assemble_cpu(const float* scan_xyz, int n_scan, const Rigid3& T,
                      const std::vector<NdtVoxelCPU>& grid,
                      float leaf, int nx, int ny, int nz,
                      double d1, double d2,
                      double H21[21], double g6[6], double* score_out);

// ndt_total_score_cpu — SCORE ONLY (no gradient/Hessian), used by (a) the
// jacobian_check gate's central-difference numeric gradient (perturbs T by
// a local delta and re-evaluates score — the calculus gate that is BLIND to
// the shared assembly code, per the file header's independence ruling), and
// (b) the score_sanity gate.
double ndt_total_score_cpu(const float* scan_xyz, int n_scan, const Rigid3& T,
                           const std::vector<NdtVoxelCPU>& grid,
                           float leaf, int nx, int ny, int nz,
                           double d1, double d2);

// run_ndt_multires_cpu — an INDEPENDENTLY-WRITTEN full coarse->fine Newton
// trajectory (own damping/accept-reject loop, own control flow — see
// reference_cpu.cpp), used by main.cu's "one full trajectory" twin gate
// against the GPU-assembly-driven host-orchestrated trajectory in main.cu.
void run_ndt_multires_cpu(const float* scan_xyz, int n_scan,
                          const std::vector<NdtVoxelCPU>& grid_coarse, float leaf_coarse, int cnx, int cny, int cnz,
                          const std::vector<NdtVoxelCPU>& grid_fine, float leaf_fine, int fnx, int fny, int fnz,
                          double d1_coarse, double d2_coarse, double d1_fine, double d2_fine,
                          Rigid3 T_init, Rigid3& out_T, double* loss_history, int& out_num_iters);

// ---------------------------------------------------------------------------
// Compact point-to-point ICP — the project's DIRECT CONTRAST baseline
// (README "Prior art", THEORY.md "where this sits in the real world" cite
// 02.06 for the full GPU point-to-point/point-to-plane/GICP treatment).
// Deliberately CPU-ONLY: this project's GPU-teaching payload is the NDT
// voxel build + assembly kernels above; re-deriving a second GPU brute-
// force correspondence-search kernel here would duplicate 02.06's own
// didactic content rather than adding new GPU-mapping lessons (README
// "Limitations" states this scoping choice honestly). ICP here reuses only
// the shared SE(3)/hidx/cholesky6_solve_flat machinery (all host code
// already), NOT any NDT-specific voxel/Mahalanobis math.
//
//   scan_xyz [n_scan*3]    : source points (scan frame)
//   target_xyz [n_target*3]: target points (map frame; a downsampled map
//                            cloud committed by scripts/make_synthetic.py —
//                            see data/README.md)
//   T_init                 : initial pose estimate
//   max_iters, max_corr_dist_m : ICP hyperparameters (kIcpMaxIters etc.)
//   out_T                  : final pose estimate
//   out_num_iters           : iterations actually taken (<= max_iters)
// ---------------------------------------------------------------------------
constexpr int   kIcpMaxIters      = 25;
constexpr float kIcpMaxCorrDistM  = 1.5f;
constexpr double kIcpLambdaInit   = 1.0e-3;   // Marquardt (diag-scaled) damping IS appropriate here:
constexpr double kIcpLambdaUp     = 10.0;     // point-to-point ICP's H = J^T J is PSD by construction
constexpr double kIcpLambdaDown   = 0.3;      // (unlike NDT's H above) — THEORY.md "numerical
constexpr double kIcpLambdaMin    = 1.0e-12;  // considerations" makes this contrast explicit.
constexpr double kIcpConvergeDeltaNorm = 1.0e-8;

void icp_point_to_point_cpu(const float* scan_xyz, int n_scan,
                            const float* target_xyz, int n_target,
                            Rigid3 T_init, int max_iters, float max_corr_dist_m,
                            Rigid3& out_T, int& out_num_iters);

#endif // PROJECT_KERNELS_CUH
