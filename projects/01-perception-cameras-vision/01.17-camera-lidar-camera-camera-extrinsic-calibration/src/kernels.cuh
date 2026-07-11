// ===========================================================================
// kernels.cuh — interface for project 01.17
//               Camera-LiDAR / camera-camera extrinsic calibration
//               (batched reprojection-error minimization, GPU Levenberg-
//               Marquardt)
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (the staged-verification driver), kernels.cu
// (the two GPU kernels), and reference_cpu.cpp (the independent CPU oracle
// twins). Everything all three must agree on — the correspondence layout,
// the SE(3) estimate representation, the 6-vector local parameterization,
// the LM hyperparameters, and the 28-scalar reduction layout — is defined
// HERE, once (CLAUDE.md §12: state layouts are single-sourced).
//
// THE ONE PROBLEM BOTH SCENARIOS SOLVE
// -------------------------------------
// Both "camera-LiDAR" and "camera-camera" extrinsic calibration are the
// SAME optimization problem wearing different clothes: given N correspon-
// dences, each a known 3-D point p_src in some SOURCE sensor's frame and an
// observed 2-D pixel uv_obs in some DEST camera, find the rigid transform
// T_dest_src = (R, t) that minimizes total reprojection error
//
//     E(R,t) = sum_i || project(R * p_src_i + t; K) - uv_obs_i ||^2
//
//   * Camera-LiDAR: p_src = a fiducial point's position AS MEASURED BY THE
//     LIDAR (lidar frame); uv_obs = that same fiducial's pixel AS DETECTED
//     BY THE CAMERA. Solving recovers T_camera_lidar.
//   * Camera-camera: p_src = a board point's position in CAMERA 1's frame
//     (known exactly, because it comes from the board's manufactured
//     geometry composed with camera 1's own, separately-solved, per-view
//     PnP pose — see README "Limitations"); uv_obs = that same point's pixel
//     AS DETECTED BY CAMERA 2. Solving recovers T_camera2_camera1.
//
// This is why one Correspondence struct, one residual/Jacobian formula, one
// LM solver, and one pair of GPU kernels serve BOTH catalog scenarios —
// "one optimizer" per the project brief. main.cu supplies two different
// correspondence SETS (loaded from two different data/sample/*.csv files)
// to the exact same machinery.
//
// PARAMETERIZATION — 6-vector local coordinates on SO(3) x R^3
// ---------------------------------------------------------------
// The estimate is stored as a Rigid3 (rotation matrix + translation, the
// same shape 02.06's ICP uses — cited below). A local perturbation is a
// 6-vector delta = [omega(3); v(3)]: omega is an so(3) LOG-ROTATION
// (Rodrigues axis-angle vector, radians, expressed in the frame R maps
// INTO — i.e. a LEFT/world-frame perturbation), v is a plain R^3 additive
// translation increment (meters). The RETRACTION (THEORY.md "The math"
// derives and names this choice explicitly, including how it differs from
// the fully-coupled se(3) exponential) is:
//
//     R(delta) = Exp(omega) * R          (Exp = so3_exp, exact Rodrigues)
//     t(delta) = t + v
//
// This is the standard "SO(3) x R^3" (decoupled) local parameterization
// used throughout vision bundle adjustment (as opposed to the fully-coupled
// se(3) exponential, which additionally couples v through the SO(3) V(omega)
// matrix) — simpler to differentiate, and the difference is a documented,
// honest simplification (THEORY.md "Numerical considerations").
//
// so3_exp() and the analytic 2x6 reprojection Jacobian in
// residual_and_jacobian() are the project's "system under test" — like
// 08.01's cart-pole dynamics, they are SHARED (__host__ __device__, guarded
// by the CALIB_HD macro below) between the GPU kernels and the CPU oracle,
// because duplicating a closed-form formula by hand would be pure
// transcription (reference_cpu.cpp's file header states the sharing ruling
// this project follows and names the INDEPENDENT gates — zero-noise
// recovery against an independently-Python-generated ground truth, and the
// numeric-vs-analytic Jacobian check — that remain blind to a bug hiding in
// this shared code).
//
// CORRESPONDENCE LAYOUT — Correspondence, TRUE (noise-free) as loaded from
// data/sample/*.csv:
//     p_src[3]   — 3-D point in the SOURCE sensor's frame, meters
//     uv_true[2] — the EXACT pixel this point projects to under the file's
//                  ground-truth extrinsic + kIntrinsics (no sensor noise)
// main.cu adds sensor noise itself, deterministically (xorshift32, seed 42),
// producing the flat OBSERVED arrays (p_obs[n*3], uv_obs[n*2]) that every
// kernel and CPU twin below actually consumes — this is what lets the same
// committed sample serve the zero-noise sanity gate (no noise added) and the
// noise-scaling gate (three documented sigmas) without three separate files.
//
// CAMERA MODEL — pinhole, intrinsics (fx, fy, cx, cy) in the 01.16 naming
// convention (cited), OPTICAL frame (z-forward depth axis, x-right, y-down —
// the documented REP-103 exception SYSTEM_DESIGN.md §3.2 allows for camera
// optics; every function below that touches a camera-frame point states
// this convention here, once).
//
// SE(3) POSE — Rigid3 { float R[9] row-major; float t[3] }, the SAME shape
// and by-value kernel-argument convention as 02.06's ICP (cited): T changes
// every LM iteration, so it rides as an ordinary kernel parameter rather
// than a __constant__ upload (02.06's kernels.cuh explains the three-way
// "how do many threads read a few floats" spectrum this project sits on).
//
// 28-SCALAR REDUCTION — the correspondence-parallel assembly kernel below
// produces, per block, the SAME kind of packed record 02.06's ICP produces
// for its 6x6 Gauss-Newton system (21 upper-triangle H entries + 6-entry g),
// PLUS one more scalar: the summed cost r^T r (so a single reduction yields
// everything main.cu needs to both solve AND report loss). hidx()'s
// row-start table and the upper-triangle packing are 02.06's convention,
// reused here unmodified (cited at the declaration below).
//
// TWO GPU PARALLELISM REGIMES (THEORY.md "The GPU mapping" argues when each
// dominates):
//   assemble_normal_equations_kernel — CORRESPONDENCE-parallel: one thread
//     per correspondence, block-level shared-memory tree reduction into one
//     block_partials row per block (host finishes the sum, exactly 02.06's
//     "GPU partial reduce, host finishes it" split). Used by main.cu's
//     host-orchestrated single-trajectory LM (one call per LM iteration) —
//     the natural mapping when N is what's large and there is ONE estimate.
//   multistart_lm_farm_kernel — OPTIMIZATION-parallel (the 08.01/01.12 farm
//     idiom, cited): one thread per INDEPENDENT LM run, K=kMultiStartK of
//     them, each looping its own up-to-kMaxLmIters iterations serially in
//     registers/local memory. The natural mapping when N is small (here,
//     48) and what's large is the number of INITIAL GUESSES you want to try
//     — the convergence-basin study this project's degeneracy/basin gates
//     are built on.
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cstdint>   // uint32_t — the xorshift32 RNG state word
#include <cmath>     // sqrtf/sinf/cosf/fabs — used by both host and device math below

// ---------------------------------------------------------------------------
// CALIB_HD — expands to "__host__ __device__" under nvcc, to NOTHING under
// cl.exe. reference_cpu.cpp is compiled by cl.exe and (per its own file
// header) must never see a CUDA keyword; this macro is how the handful of
// shared "camera model" primitives below compile as plain host functions
// there while ALSO compiling as dual host/device functions for kernels.cu's
// __global__ kernels to call directly — without reference_cpu.cpp including
// <cuda_runtime.h> at all. See the file header's "PARAMETERIZATION" note for
// which functions this applies to and why sharing them is the documented,
// deliberate choice (not an oversight).
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define CALIB_HD __host__ __device__
#else
#define CALIB_HD
#endif

// ===========================================================================
// Camera model — ONE pinhole camera model, reused for every projection in
// both scenarios (the camera in the camera-LiDAR pair, and camera 2 in the
// camera-camera pair). fx/fy/cx/cy naming follows 01.16's convention (cited
// in that project's Zhang-calibration code) — the intrinsics THIS project
// assumes are already known (01.16's job), consumed here by name.
// Image is deliberately small (160x120) so the committed sample PGM used for
// the demo's overlay artifact stays kilobytes, not megabytes (CLAUDE.md §8).
// ===========================================================================
constexpr int   kImageWidth  = 160;   // px
constexpr int   kImageHeight = 120;   // px
constexpr float kFx = 154.0f;         // px (focal length x, ~55 deg horizontal FOV at this width)
constexpr float kFy = 152.0f;         // px (focal length y)
constexpr float kCx = 80.0f;          // px (principal point x, image-center-ish)
constexpr float kCy = 60.0f;          // px (principal point y)

struct PinholeIntrinsics {
    float fx, fy, cx, cy;   // px; see the file header for the frame convention (optical, z-forward)
};
constexpr PinholeIntrinsics kIntrinsics{ kFx, kFy, kCx, kCy };

// ---------------------------------------------------------------------------
// Correspondence — ONE (3-D source point, 2-D dest pixel) pair, TRUE
// (noise-free) as committed in data/sample/*.csv. See the file header's
// "CORRESPONDENCE LAYOUT" note for why noise is added later, by main.cu, not
// baked into this struct.
// ---------------------------------------------------------------------------
struct Correspondence {
    float p_src[3];    // 3-D point in the SOURCE sensor's frame, meters (exact)
    float uv_true[2];  // exact pixel projection under the file's ground truth (px)
};

// Dataset shape shared by every data/sample/*.csv this project ships: a
// planar target with 4 retroreflector-style fiducials, observed across
// V=12 distinct target poses — the catalog bullet's exact numbers.
constexpr int kNumViews      = 12;
constexpr int kPointsPerView = 4;
constexpr int kNumCorr       = kNumViews * kPointsPerView;   // 48

// ---------------------------------------------------------------------------
// Rigid3 — a rigid-body transform, passed BY VALUE to every kernel that
// needs "the current estimate" (02.06's ICP convention, reused verbatim and
// cited: T changes every LM iteration, so a __constant__ upload would be
// pure overhead for 48 bytes — see that project's kernels.cuh for the full
// three-way "how many threads read a few floats" argument).
// x_dest = R * p_src + t.
// ---------------------------------------------------------------------------
struct Rigid3 {
    float R[9];   // row-major 3x3 rotation (orthonormal by construction — every
                  // update goes through so3_exp()'s exact Rodrigues formula,
                  // never an additive approximation, so it cannot drift)
    float t[3];   // translation (m), dest frame
};

constexpr Rigid3 kIdentityRigid3{
    { 1.0f, 0.0f, 0.0f,  0.0f, 1.0f, 0.0f,  0.0f, 0.0f, 1.0f },
    { 0.0f, 0.0f, 0.0f }
};

// ===========================================================================
// LM (Levenberg-Marquardt) hyperparameters — single-sourced so main.cu's
// host-orchestrated single trajectory, kernels.cu's two kernels, and
// reference_cpu.cpp's independent CPU trajectories all damp, adapt, and stop
// identically. THEORY.md "The algorithm" derives the Marquardt damping rule
// (lambda scales diag(H), not just adds lambda*I) and justifies these
// numbers.
// ===========================================================================
constexpr int    kMaxLmIters        = 20;     // hard iteration cap (catalog bullet's number)
constexpr double kLambdaInit        = 1.0e-3; // initial Marquardt damping (unitless scale on diag(H))
constexpr double kLambdaUp          = 10.0;   // multiply on a REJECTED step (back off)
constexpr double kLambdaDown        = 0.3;    // multiply on an ACCEPTED step (press forward)
constexpr double kLambdaMin         = 1.0e-12;
constexpr double kConvergeDeltaNorm = 1.0e-9; // ||delta|| (mixed rad/m; THEORY.md "Numerical considerations")
constexpr double kConvergeCostRel   = 1.0e-12;// relative cost-decrease-too-small stopping rule

// A deterministic, modest offset from ground truth used as the "rough
// prior" starting guess for the single-trajectory stages (twin check, noise
// scaling, degeneracy) — standing in for a real system's CAD/previous-
// calibration estimate. Chosen FIXED (not random) so those stages need no
// RNG at all; the genuinely-blind, no-prior-knowledge case is what the
// multi-start farm explores (see kMultiStartK below).
constexpr float kRoughPriorOmegaOffset[3] = { 0.05f, -0.04f, 0.03f };  // rad
constexpr float kRoughPriorTransOffset[3] = { 0.03f, 0.03f, -0.02f };  // m

// Multi-start farm (the convergence-basin study): K independent LM runs
// from randomized initial guesses, perturbed away from the IDENTITY
// transform (not ground truth — the whole point is these runs have no
// privileged information) by up to the stated rotation/translation
// magnitudes, drawn uniformly in [0, max] with a uniformly-random direction.
constexpr int   kMultiStartK    = 1024;   // catalog bullet's number
constexpr float kBasinMaxRotRad = 2.4f;   // ~137 deg — deliberately larger than "comfortable" so the
                                          // farm actually FINDS a basin boundary (measured: smaller
                                          // ranges converged 100% of the time, which taught nothing
                                          // about basin SIZE — THEORY.md reports the measured curve)
constexpr float kBasinMaxTransM = 1.20f;  // 1.2 m — likewise deliberately large

// "Converged" classification for the basin gate: final loss near the global
// minimum AND final parameters near the true answer (THEORY.md "How we
// verify correctness" explains why BOTH conditions are required — a run can
// reach a low-loss point that is not the metrically-true extrinsic on badly
// under-constrained data, which is precisely the degeneracy lesson).
constexpr float kBasinConvergedRotDeg = 1.0f;   // deg
constexpr float kBasinConvergedTransM = 0.01f;  // m (1 cm)

// ---------------------------------------------------------------------------
// NoiseLevel — the three documented sensor-noise settings used by the
// noise-scaling gate (and kNoiseMed is also the "realistic" level used by
// the recovery / basin / degeneracy gates). sigma_px is camera pixel-
// detection noise (std-dev, px); sigma_p_src_m is the SOURCE point's
// Cartesian position noise (std-dev per axis, m) — for camera-LiDAR this
// approximates LiDAR range/angular noise as isotropic Cartesian Gaussian
// (THEORY.md "Numerical considerations" names the real spherical-noise
// model this simplifies); for camera-camera it is UNUSED (sigma_p_src_m is
// simply not applied — see the file header's "known 3-D board points" note
// and README "Limitations").
// ---------------------------------------------------------------------------
struct NoiseLevel { float sigma_px; float sigma_p_src_m; const char* label; };
constexpr NoiseLevel kNoiseLow  { 0.10f, 0.003f, "low"  };
constexpr NoiseLevel kNoiseMed  { 0.50f, 0.010f, "med"  };
constexpr NoiseLevel kNoiseHigh { 2.00f, 0.040f, "high" };

// ===========================================================================
// xorshift32 + Box-Muller Gaussian — the repo's portable deterministic RNG
// (08.01's exact construction, cited), used here for BOTH host-side noise
// synthesis (main.cu) and device-side multi-start initial-guess sampling
// (kernels.cu) — CALIB_HD because the farm kernel calls it per-thread.
// ===========================================================================
CALIB_HD inline uint32_t xorshift32(uint32_t& state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

// (0,1] — never exactly 0, safe for log() inside gaussian()'s Box-Muller.
CALIB_HD inline float uniform01(uint32_t& state)
{
    return (xorshift32(state) >> 8) * (1.0f / 16777216.0f) + (0.5f / 16777216.0f);
}

// One N(0, sigma^2) draw, Box-Muller in double for the transcendental step
// (keeps the tails well-behaved in FP32 — 08.01's exact reasoning, cited).
CALIB_HD inline float gaussian(uint32_t& state, float sigma)
{
    const double u1 = static_cast<double>(uniform01(state));
    const double u2 = static_cast<double>(uniform01(state));
    const double z  = sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
    return sigma * static_cast<float>(z);
}

// ===========================================================================
// Shared "system under test" math primitives (CALIB_HD — see the macro
// comment above). These are the camera model + SO(3) retraction: exactly
// the kind of "dynamics model that IS the system under test" the twin-
// independence ruling (reference_cpu.cpp's header) names as the legitimate
// case for sharing. What is NOT shared: the per-iteration accumulation loop
// (kernels.cu's block reduction vs. reference_cpu.cpp's serial loop) and the
// LM control flow (main.cu's host orchestration vs. reference_cpu.cpp's own
// independent loops) — see reference_cpu.cpp's header for the full ruling
// as applied here, and the independent gates that remain blind to a bug
// hiding in this shared code.
// ===========================================================================

// skew3 — the skew-symmetric (cross-product) matrix [v]_x, row-major, such
// that [v]_x * w == v cross w for any w. Used twice below: inside so3_exp
// (Rodrigues' formula) and inside residual_and_jacobian (the rotation-
// perturbation Jacobian term).
CALIB_HD inline void skew3(const float v[3], float S[9])
{
    S[0] =  0.0f;  S[1] = -v[2];  S[2] =  v[1];
    S[3] =  v[2];  S[4] =  0.0f;  S[5] = -v[0];
    S[6] = -v[1];  S[7] =  v[0];  S[8] =  0.0f;
}

// mat3_vec — out = R * p, row-major 3x3 times 3-vector.
CALIB_HD inline void mat3_vec(const float R[9], const float p[3], float out[3])
{
    out[0] = R[0] * p[0] + R[1] * p[1] + R[2] * p[2];
    out[1] = R[3] * p[0] + R[4] * p[1] + R[5] * p[2];
    out[2] = R[6] * p[0] + R[7] * p[1] + R[8] * p[2];
}

// mat3_mul — out = A * B, row-major 3x3 times 3x3. out must not alias A or B.
CALIB_HD inline void mat3_mul(const float A[9], const float B[9], float out[9])
{
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            float acc = 0.0f;
            for (int k = 0; k < 3; ++k) acc += A[r * 3 + k] * B[k * 3 + c];
            out[r * 3 + c] = acc;
        }
    }
}

// so3_exp — the SO(3) exponential map (Rodrigues' rotation formula): given
// an axis-angle vector omega (rad; |omega| = rotation angle, direction =
// rotation axis), return R = Exp([omega]_x) EXACTLY (not the first-order
// approximation) as a row-major 3x3 matrix.
//
//     R = I + (sin(theta)/theta) [omega]_x + ((1-cos(theta))/theta^2) [omega]_x^2
//
// theta = |omega|. THEORY.md "The math" derives this from the ODE
// dR/dt = [omega]_x R; the small-angle branch below (theta < 1e-8) uses the
// first-order truncation R ~= I + [omega]_x, whose own error is O(theta^2)
// -- below the branch threshold, negligibly smaller than FP32 epsilon for
// the theta values this project's retraction ever calls this with (THEORY.md
// "Numerical considerations" bounds it).
CALIB_HD inline void so3_exp(const float omega[3], float R[9])
{
    const float theta2 = omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2];
    const float theta  = sqrtf(theta2);

    float S[9];
    skew3(omega, S);   // [omega]_x — used by both branches below

    if (theta < 1.0e-8f) {
        // Small-angle branch: avoids the 0/0 in sin(theta)/theta and
        // (1-cos theta)/theta^2 as theta -> 0. First-order Taylor is exact
        // enough here (see the function header for the error bound).
        R[0] = 1.0f + S[0]; R[1] = S[1];        R[2] = S[2];
        R[3] = S[3];        R[4] = 1.0f + S[4]; R[5] = S[5];
        R[6] = S[6];        R[7] = S[7];        R[8] = 1.0f + S[8];
        return;
    }

    const float a = sinf(theta) / theta;              // sin(theta)/theta
    const float b = (1.0f - cosf(theta)) / theta2;     // (1-cos theta)/theta^2

    float S2[9];
    mat3_mul(S, S, S2);   // [omega]_x^2

    for (int i = 0; i < 9; ++i) {
        const float identity_i = (i == 0 || i == 4 || i == 8) ? 1.0f : 0.0f;
        R[i] = identity_i + a * S[i] + b * S2[i];
    }
}

// pinhole_project — project a point ALREADY IN THE DEST CAMERA'S OPTICAL
// FRAME (z-forward, x-right, y-down; file header "CAMERA MODEL") to a pixel.
// Pcam[2] (depth) is assumed > 0 (point in front of the camera); callers
// that might violate this (multi-start's randomized initial guesses can, in
// principle, rotate a point behind the camera) tolerate the resulting large
// residual rather than special-casing it — THEORY.md "Numerical
// considerations" discusses why that is the honest, simple choice here.
CALIB_HD inline void pinhole_project(const float Pcam[3], PinholeIntrinsics K, float uv[2])
{
    const float invZ = 1.0f / Pcam[2];
    uv[0] = K.fx * Pcam[0] * invZ + K.cx;
    uv[1] = K.fy * Pcam[1] * invZ + K.cy;
}

// residual_and_jacobian — THE formula this whole project teaches: the
// reprojection residual and its analytic 2x6 Jacobian w.r.t. the LOCAL
// 6-vector [omega; v] at the CURRENT estimate T (file header
// "PARAMETERIZATION"). THEORY.md "The math" derives every line below from
// the chain rule; this is the "classic 2x6" the project brief calls out.
//
//   r = project(R*p_src + t; K) - uv_obs                      (2-vector, px)
//   Pcam = R*p_src + t;  RP = R*p_src  (= Pcam - t, the ROTATED-ONLY point)
//   d(Pcam)/d(omega) = -[RP]_x     (LEFT/world-frame perturbation, see the
//                                    file header's PARAMETERIZATION note —
//                                    NOT -[Pcam]_x; the rotation perturbs
//                                    R*p_src, translation is untouched by it)
//   d(Pcam)/d(v)     =  I_3
//   J_proj (2x3) = d(u,v)/d(Pcam):
//       [ fx/Zc,   0,     -fx*Xc/Zc^2 ]
//       [ 0,       fy/Zc, -fy*Yc/Zc^2 ]
//   J (2x6, row-major) = J_proj * [ -[RP]_x | I_3 ]
//
// Parameters:
//   T       — current estimate (Rigid3, dest<-src)
//   K       — dest camera intrinsics
//   p_src   — [3] the correspondence's OBSERVED (possibly noisy) source point
//   uv_obs  — [2] the correspondence's OBSERVED (possibly noisy) pixel
//   r  OUT  — [2] residual (px)
//   J  OUT  — [12] Jacobian, ROW-MAJOR 2x6: J[row*6+col], row in {0,1} (u,v),
//             col in {0..2}=omega, {3..5}=v
CALIB_HD inline void residual_and_jacobian(const Rigid3& T, PinholeIntrinsics K,
                                           const float p_src[3], const float uv_obs[2],
                                           float r[2], float J[12])
{
    float RP[3];
    mat3_vec(T.R, p_src, RP);                              // R * p_src (rotated-only point)
    const float Pcam[3] = { RP[0] + T.t[0], RP[1] + T.t[1], RP[2] + T.t[2] };

    float uv[2];
    pinhole_project(Pcam, K, uv);
    r[0] = uv[0] - uv_obs[0];
    r[1] = uv[1] - uv_obs[1];

    const float invZ  = 1.0f / Pcam[2];
    const float invZ2 = invZ * invZ;
    // J_proj, row-major 2x3: row 0 = d(u)/d(Pcam), row 1 = d(v)/d(Pcam).
    const float Jproj[6] = {
        K.fx * invZ, 0.0f,        -K.fx * Pcam[0] * invZ2,
        0.0f,        K.fy * invZ, -K.fy * Pcam[1] * invZ2
    };

    float S[9];
    skew3(RP, S);   // [RP]_x — RP, NOT Pcam (see the function header derivation)

    // J[:,0:3] = Jproj * (-S); J[:,3:6] = Jproj (since d(Pcam)/d(v) = I_3).
    for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < 3; ++col) {
            float acc = 0.0f;
            for (int k = 0; k < 3; ++k) acc += Jproj[row * 3 + k] * (-S[k * 3 + col]);
            J[row * 6 + col] = acc;
        }
        J[row * 6 + 3] = Jproj[row * 3 + 0];
        J[row * 6 + 4] = Jproj[row * 3 + 1];
        J[row * 6 + 5] = Jproj[row * 3 + 2];
    }
}

// retract — apply a local 6-vector delta = [omega(3); v(3)] to T via the
// file header's decoupled SO(3) x R^3 retraction: R_new = Exp(omega) * R
// (EXACT Rodrigues update, not the first-order approximation used to derive
// the Jacobian above — standard Gauss-Newton-on-a-manifold practice: linear
// Jacobian for the STEP, exact retraction for the UPDATE), t_new = t + v.
CALIB_HD inline void retract(const Rigid3& T, const double delta[6], Rigid3& out)
{
    const float omega[3] = { static_cast<float>(delta[0]), static_cast<float>(delta[1]), static_cast<float>(delta[2]) };
    float dR[9];
    so3_exp(omega, dR);
    mat3_mul(dR, T.R, out.R);
    out.t[0] = T.t[0] + static_cast<float>(delta[3]);
    out.t[1] = T.t[1] + static_cast<float>(delta[4]);
    out.t[2] = T.t[2] + static_cast<float>(delta[5]);
}

// rotation_angle_deg / translation_error_m — small reporting helpers used
// throughout main.cu's gates to turn a recovered Rigid3 into a human metric
// against ground truth. angle = arccos((trace(R_err)-1)/2), R_err = R^T R_gt
// (the standard rotation-matrix geodesic distance). CALIB_HD only because it
// is convenient to keep beside the rest of this math; never called from a
// kernel (host-only usage in main.cu).
CALIB_HD inline float rotation_angle_deg(const float R[9], const float R_gt[9])
{
    // R_err = R^T * R_gt (R is orthonormal, so R^T == R^-1: this is "how far
    // does R_gt differ from R, expressed as a single rotation").
    float Rt[9] = { R[0], R[3], R[6],  R[1], R[4], R[7],  R[2], R[5], R[8] };
    float Rerr[9];
    mat3_mul(Rt, R_gt, Rerr);
    float trace = Rerr[0] + Rerr[4] + Rerr[8];
    float c = (trace - 1.0f) * 0.5f;
    if (c > 1.0f) c = 1.0f;      // clamp: FP rounding can push |c| a hair past 1
    if (c < -1.0f) c = -1.0f;
    return acosf(c) * (180.0f / 3.14159265358979323846f);
}

CALIB_HD inline float translation_error_m(const float t[3], const float t_gt[3])
{
    const float dx = t[0] - t_gt[0], dy = t[1] - t_gt[1], dz = t[2] - t_gt[2];
    return sqrtf(dx * dx + dy * dy + dz * dz);
}

// ===========================================================================
// hidx / kReduceWidth — the 6x6 symmetric normal-matrix upper-triangle
// packing, 02.06's ICP convention REUSED VERBATIM (cited): row i's valid
// columns are j=i..5, row_start = {0,6,11,15,18,20}. This project extends
// the reduced record by ONE scalar beyond 02.06's 27 (H21+g6): index 27
// holds the summed cost r^T r, so a single reduction produces everything a
// caller needs to both SOLVE (H,g) and REPORT (cost) in one pass.
// Made CALIB_HD (unlike 02.06, which kept it host-only and had device code
// write literal indices instead) because this project already carries the
// CALIB_HD macro for the shared camera-model primitives above — extending it
// to this tiny index helper adds no new dual-compilation risk.
// ===========================================================================
CALIB_HD inline int hidx(int i, int j)
{
    const int row_start[6] = { 0, 6, 11, 15, 18, 20 };
    return row_start[i] + (j - i);   // caller guarantees i <= j <= 5
}

constexpr int kReduceWidth     = 28;   // 21 (H upper triangle) + 6 (g) + 1 (cost)
constexpr int kThreadsPerBlock = 128;  // assembly-kernel block size (02.06's kThreadsReduce default)

// blocks_for — ceil(count/threads), 02.06/08.01's shared idiom (cited).
// Declared here (not just in kernels.cu) because main.cu must compute the
// IDENTICAL block count to size the block_partials download buffer.
CALIB_HD inline int blocks_for(int count, int threads)
{
    return (count + threads - 1) / threads;
}

// ===========================================================================
// cholesky6_solve — HOST-ONLY (no CALIB_HD: this is NOT compiled for the
// device; the multi-start farm kernel carries its OWN independently-written
// device-side Cholesky in kernels.cu — deliberately NOT shared, unlike the
// camera-model primitives above, so the "multi-start batch subset" twin gate
// in main.cu is a genuine GPU-vs-CPU check of the whole farm behavior,
// linear solve included).
//
// Solves the Marquardt-damped normal equations (H + lambda*diag(H)) delta =
// -g for a 6x6 SYMMETRIC POSITIVE-(semi)DEFINITE H via Cholesky-Crout
// decomposition (H_damped = L L^T, L lower-triangular) plus forward/back
// substitution — the textbook small-dense-SPD solve (cite 33.01's batched
// small-matrix linalg project for the general pattern this specializes).
//
// This function is used by BOTH main.cu (the host-orchestrated single-
// trajectory LM's per-iteration solve, after the GPU assembly kernel hands
// back H/g) AND reference_cpu.cpp (its own independent LM trajectories).
// That sharing is NOT a twin-independence concern: neither caller is "the
// GPU path" — both are host code, and main.cu is the orchestrator, not one
// of the two oracle-compared implementations (reference_cpu.cpp's header
// explains the CPU-vs-GPU boundary this project draws the sharing line at).
//
// Parameters:
//   H21    — [21] upper-triangle of H = J^T J (hidx() packing), UNDAMPED
//   g6     — [6] g = J^T r
//   lambda — Marquardt damping scale (>= 0; 0 = plain Gauss-Newton)
//   out_delta — [6] OUT: solved step, valid only if this function returns true
// Returns false if H+lambda*diag(H) is not (numerically) positive definite
// — callers respond by increasing lambda and retrying (THEORY.md
// "Numerical considerations").
// ===========================================================================
inline bool cholesky6_solve(const double H21[21], const double g6[6], double lambda, double out_delta[6])
{
    double A[6][6];
    for (int i = 0; i < 6; ++i)
        for (int j = i; j < 6; ++j) {
            const double hij = H21[hidx(i, j)];
            A[i][j] = hij;
            A[j][i] = hij;
        }
    // Marquardt damping: scale the DIAGONAL by (1+lambda), not a flat
    // lambda*I add — this keeps the damping's units/scale matched to each
    // parameter's own curvature (THEORY.md "The algorithm" derives why this
    // variant converges more robustly than Levenberg's original lambda*I).
    for (int i = 0; i < 6; ++i) A[i][i] *= (1.0 + lambda);

    double L[6][6] = {};
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = A[i][j];
            for (int k = 0; k < j; ++k) sum -= L[i][k] * L[j][k];
            if (i == j) {
                if (sum <= 0.0) return false;   // not SPD — caller backs off lambda
                L[i][i] = sqrt(sum);
            } else {
                L[i][j] = sum / L[j][j];
            }
        }
    }

    // Forward-substitute L y = -g, then back-substitute L^T delta = y.
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

// ---------------------------------------------------------------------------
// jacobi_eigen_symmetric6 — classic cyclic Jacobi eigenvalue algorithm
// (Golub & Van Loan), specialized to N=6 — 01.16's exact construction and
// name, reimplemented here per this project's own copy (self-containment
// rule, CLAUDE.md §4: never reference another project's files, reimplement
// with credit). Used ONLY by main.cu's degeneracy gate, to turn a converged
// H=J^T J into a condition-number PROXY (max eigenvalue / min eigenvalue) —
// a one-shot, twice-per-run host computation, utterly dominated by the GPU
// kernels above (same "stays on the host" call 01.16 makes for its own
// one-shot 6x6 eigensolve).
// Parameters: A (IN/OUT — overwritten with the diagonalized matrix; the
// diagonal on return IS the eigenvalues, in no particular order), eigvecs
// (OUT — columns are eigenvectors, unused by the degeneracy gate but
// returned for completeness / the numeric-check exercise in README).
// ---------------------------------------------------------------------------
inline void jacobi_eigen_symmetric6(double A[6][6], double eigvecs[6][6])
{
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            eigvecs[i][j] = (i == j) ? 1.0 : 0.0;

    // 01.16's measured practice: 3-5 sweeps converges a 3x3; a 6x6 has more
    // off-diagonal pairs to zero, so this project budgets more sweeps
    // (THEORY.md "Numerical considerations" reports the measured residual
    // off-diagonal magnitude after this many sweeps).
    const int kSweeps = 12;
    for (int sweep = 0; sweep < kSweeps; ++sweep) {
        for (int p = 0; p < 6; ++p) {
            for (int q = p + 1; q < 6; ++q) {
                if (fabs(A[p][q]) < 1e-15) continue;   // already (numerically) zero
                const double theta = (A[q][q] - A[p][p]) / (2.0 * A[p][q]);
                const double t = (theta >= 0.0 ? 1.0 : -1.0) /
                                 (fabs(theta) + sqrt(theta * theta + 1.0));
                const double c = 1.0 / sqrt(t * t + 1.0);
                const double s = t * c;
                const double app = A[p][p], aqq = A[q][q], apq = A[p][q];
                A[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
                A[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
                A[p][q] = A[q][p] = 0.0;
                for (int k = 0; k < 6; ++k) {
                    if (k == p || k == q) continue;
                    const double akp = A[k][p], akq = A[k][q];
                    A[k][p] = A[p][k] = c * akp - s * akq;
                    A[k][q] = A[q][k] = s * akp + c * akq;
                }
                for (int k = 0; k < 6; ++k) {
                    const double vkp = eigvecs[k][p], vkq = eigvecs[k][q];
                    eigvecs[k][p] = c * vkp - s * vkq;
                    eigvecs[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }
}

// ===========================================================================
// GPU kernels (kernels.cu). __global__ signatures are __CUDACC__-fenced
// (only nvcc parses them — plain host TUs like reference_cpu.cpp never see
// the word __global__), matching the template's established pattern.
// ===========================================================================
#ifdef __CUDACC__

// assemble_normal_equations_kernel — CORRESPONDENCE-parallel: thread i
// computes correspondence i's residual+Jacobian (via the shared
// residual_and_jacobian above), folds its 2x6 contribution into a 28-scalar
// [H21|g6|cost] record, and the block tree-reduces (shared memory) all its
// threads' records into ONE row of block_partials. Full documentation
// (thread mapping, shared-memory budget, why no cross-block atomics) sits
// with the definition in kernels.cu.
__global__ void assemble_normal_equations_kernel(
    const float* __restrict__ p_obs, const float* __restrict__ uv_obs, int n,
    Rigid3 T, PinholeIntrinsics K,
    float* __restrict__ block_partials);

// multistart_lm_farm_kernel — OPTIMIZATION-parallel: thread k draws its own
// randomized initial guess and runs its own COMPLETE up-to-kMaxLmIters LM
// trajectory serially, entirely in registers/local memory (n=48
// correspondences is small enough that the whole inner loop lives on-chip).
// Full documentation sits with the definition in kernels.cu.
__global__ void multistart_lm_farm_kernel(
    const float* __restrict__ p_obs, const float* __restrict__ uv_obs, int n,
    PinholeIntrinsics K, Rigid3 T_seed,
    float max_rot_perturb_rad, float max_trans_perturb_m,
    uint32_t base_seed, int k_starts, int max_iters,
    Rigid3* __restrict__ out_T, double* __restrict__ out_loss,
    float* __restrict__ out_init_rot, float* __restrict__ out_init_trans);

#endif // __CUDACC__

// ---------------------------------------------------------------------------
// Host launch wrappers (defined in kernels.cu; declared outside the
// __CUDACC__ fence so any translation unit, including main.cu, may call
// them — only their DEFINITIONS need nvcc).
// ---------------------------------------------------------------------------

// launch_assemble_normal_equations — runs assemble_normal_equations_kernel
// and returns the number of blocks it launched (== blocks_for(n,
// kThreadsPerBlock)) so the caller knows how many kReduceWidth-wide rows to
// download from d_block_partials and sum (main.cu does that sum in double).
int launch_assemble_normal_equations(const float* d_p_obs, const float* d_uv_obs, int n,
                                     Rigid3 T, PinholeIntrinsics K,
                                     float* d_block_partials);

// launch_multistart_farm — runs multistart_lm_farm_kernel for k_starts
// independent optimizations (k_starts <= kMultiStartK; all output arrays
// must have room for k_starts entries).
void launch_multistart_farm(const float* d_p_obs, const float* d_uv_obs, int n,
                            PinholeIntrinsics K, Rigid3 T_seed,
                            float max_rot_perturb_rad, float max_trans_perturb_m,
                            uint32_t base_seed, int k_starts, int max_iters,
                            Rigid3* d_out_T, double* d_out_loss,
                            float* d_out_init_rot, float* d_out_init_trans);

// ===========================================================================
// CPU references (reference_cpu.cpp) — the independent oracle twins. All
// pointers are HOST pointers. See reference_cpu.cpp's file header for
// exactly what is and is not independently reimplemented here.
// ===========================================================================

// assemble_normal_equations_cpu — the twin of assemble_normal_equations_kernel:
// same shared residual_and_jacobian formula, but the ACCUMULATION LOOP
// (over all n correspondences, into H21/g6/cost) is written independently,
// sequentially, no reduction tree. main.cu compares this against the GPU
// kernel's (block-summed-then-host-double-summed) output at tight tolerance.
void assemble_normal_equations_cpu(const float* p_obs, const float* uv_obs, int n,
                                   const Rigid3& T, PinholeIntrinsics K,
                                   double H21[21], double g6[6], double* cost_out);

// run_lm_cpu — an INDEPENDENTLY-WRITTEN full LM trajectory (own loop, own
// damping/accept-reject flow — see reference_cpu.cpp), used by main.cu's
// "one full LM trajectory" twin gate against the GPU-assembly-driven
// host-orchestrated trajectory. Records up to max_iters+1 cost values
// (initial + one per iteration actually taken) into loss_history; returns
// the actual count via out_num_iters (<= max_iters+1).
void run_lm_cpu(const float* p_obs, const float* uv_obs, int n, PinholeIntrinsics K,
                Rigid3 T_init, int max_iters,
                Rigid3& out_T, double* loss_history, int& out_num_iters);

// multistart_lm_cpu — reproduces ONE multi-start thread's exact result on
// the CPU: same seed formula as multistart_lm_farm_kernel (so main.cu can
// hand it k=0..63 and expect the SAME initial guesses the GPU thread k
// drew), independently-written LM loop (own control flow; shares only
// residual_and_jacobian and the host cholesky6_solve, per the file header).
void multistart_lm_cpu(const float* p_obs, const float* uv_obs, int n,
                       PinholeIntrinsics K, Rigid3 T_seed,
                       float max_rot_perturb_rad, float max_trans_perturb_m,
                       uint32_t base_seed, int k, int max_iters,
                       Rigid3& out_T, double& out_loss,
                       float& out_init_rot, float& out_init_trans);

#endif // PROJECT_KERNELS_CUH
