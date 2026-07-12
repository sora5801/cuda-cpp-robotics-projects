// ===========================================================================
// kernels.cuh — interface for project 02.16
//               Multi-LiDAR merging + extrinsic refinement
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (the staged-verification driver), kernels.cu
// (the GPU kernels), and reference_cpu.cpp (the independent CPU oracle
// twins). Everything all three must agree on — the rig geometry, the point
// layout, the zone/plane representation, the SE(3) refinement parameters,
// and the 28-scalar reduction layout — is defined HERE, once (CLAUDE.md §12:
// data-layout contracts are single-sourced; scripts/make_synthetic.py, being
// Python, duplicates the numeric rig constants with a "must match" comment
// rather than sharing this header — see that file's own header for why the
// duplication is harmless here).
//
// THE RIG (top-down, not to scale; base frame: +x forward, +y left, +z up,
// origin at the vehicle's ground-projected center — CLAUDE.md §12)
// ---------------------------------------------------------------------------
//
//                              +x (forward)
//                                   ^
//                    LEFT FOV        |        RIGHT FOV
//                  (-15..165 deg)    |     (-165..15 deg)
//                        \           |           /
//                         \.         |         ,/
//                    L  ---o_________|_________o---  R      <- corner mounts
//                (1.8,0.9,0.5)   [ vehicle ]  (1.8,-0.9,0.5)   z=0.5 m, yawed
//                         :        body        :               +/-55 deg
//           +y <----------:-----+  o  +--------:----------  (left of vehicle)
//          (left)         :  MAIN (roof, x=0,y=0,z=1.8m,      :
//                         :   360 deg, no drift -- the        :
//                         :   trusted reference sensor)        :
//                         :________________________________:
//                                     |
//                                (rear, blind
//                              to LEFT+RIGHT --
//                               MAIN-only zone)
//
// Three LiDARs, one vehicle: MAIN spins the full 360 deg from the roof and
// never drifts (it anchors every other sensor's calibration — the role a
// fleet's factory-calibrated primary sensor plays in practice, PRACTICE.md
// §3). LEFT and RIGHT are front-corner units, angled ~55 deg outward, each
// covering a 180-deg wedge (kernels.cuh's FOV constants below) — together
// they cover the whole vehicle perimeter EXCEPT a ~30-deg wedge directly
// behind it (MAIN's job to fill — the classic "why does a car need a roof
// LiDAR AND corner LiDARs" blind-zone argument, README "System context").
// LEFT and RIGHT overlap each other only in a narrow +/-15 deg FORWARD cone.
//
// scripts/make_synthetic.py builds a small "yard" (a ground plane plus four
// walls at three MUTUALLY ORTHOGONAL orientations — front/rear walls normal
// along +/-x, left/right walls normal along +/-y, the ground normal along
// +z — plus decorative poles) and works out, from the FOV wedges above,
// which of the three sensors captures each world point. The resulting
// OVERLAP TABLE (surface x sensor-pair; "n" = committed-sample point counts,
// see data/README.md) is the geometric backbone of every stage below:
//
//     surface       MAIN&LEFT   MAIN&RIGHT   LEFT&RIGHT   all three
//     ground             737          737           33          33
//     wall_front         250          250           90          90
//     wall_left          680            0            0           0
//     wall_right           0          680            0           0
//     wall_rear (MAIN-only; a small LEFT/RIGHT sliver exists but is
//                unused by any zone set below — see "ZONE SETS")
//
// TWO COHORTS (data/sample/aligned.csv, drifted.csv) — the SAME world
// points, the SAME sensor assignment, differing ONLY in each sensor's TRUE
// mounting pose: "aligned" = every sensor exactly at its nominal (as-
// designed) pose (the CONTROL); "drifted" = LEFT and RIGHT carry a small,
// documented, DIFFERENT drift each (~0.8 deg / 3 cm — the "mounts creep"
// story 01.17's PRACTICE.md tells for camera rigs, cited, applied here to
// LiDAR mounts: vibration and thermal cycling loosen a bracket a hair over
// months of fleet service). MAIN never drifts. See NOMINAL_MOUNT/DRIFT below
// for the exact numbers and nominal_extrinsic()/true_extrinsic() for how
// they become rigid transforms.
//
// ZONE SETS — which fitted planes feed which refinement solve. LEFT's "full"
// solve uses {ground, wall_front, wall_left} (three MUTUALLY ORTHOGONAL
// normals: z, x, y — full-rank observability, THEORY.md derives why);
// LEFT's "degenerate" solve (the observability-contrast gate) uses
// {wall_front} ALONE (one normal direction — the coplanar-pose lesson
// 01.17's DEGENERACY gate teaches, recast from camera poses to LiDAR
// planes). RIGHT's sets are the mirror ({wall_right} not {wall_left}).
// wall_rear and pole are never used as refinement targets (wall_rear's
// LEFT/RIGHT coverage is too thin — see the table above — and poles are
// excluded from plane fitting on principle: a cylinder is not a plane).
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cstdint>   // int32_t, uint32_t, uint64_t — exact-width integers throughout
#include <cmath>     // sqrtf/sinf/cosf/fabs/floor — identical overloads to cl.exe and nvcc's host pass

// ---------------------------------------------------------------------------
// CALIB_HD — expands to "__host__ __device__" under nvcc, to NOTHING under
// cl.exe (01.17's exact macro and reasoning, cited): reference_cpu.cpp is
// compiled by cl.exe and must never see a CUDA keyword; this macro lets the
// handful of shared "rigid-transform primitive" functions below compile as
// plain host functions there while ALSO compiling as dual host/device
// functions for kernels.cu's __global__ kernels to call directly.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define CALIB_HD __host__ __device__
#else
#define CALIB_HD
#endif

// ===========================================================================
// Sensor / surface vocabulary — shared by every file, and by
// scripts/make_synthetic.py (duplicated there in Python; see its header).
// ===========================================================================
constexpr int32_t kSensorMain  = 0;
constexpr int32_t kSensorLeft  = 1;
constexpr int32_t kSensorRight = 2;
constexpr int      kNumSensors = 3;

constexpr int32_t kSurfaceGround    = 0;
constexpr int32_t kSurfaceWallFront = 1;
constexpr int32_t kSurfaceWallLeft  = 2;
constexpr int32_t kSurfaceWallRight = 3;
constexpr int32_t kSurfaceWallRear  = 4;
constexpr int32_t kSurfacePole      = 5;
constexpr int      kNumSurfaces     = 6;

// Zone masks — which surface_id bits a given refinement solve treats as
// active targets (file header "ZONE SETS"). A point whose surface_id bit is
// clear in the mask contributes nothing to that solve, regardless of
// whether a fitted plane for it exists (kernels.cu's assembly kernel checks
// both target_planes[...].valid AND this mask).
constexpr uint32_t kZoneMaskLeftFull  = (1u << kSurfaceGround) | (1u << kSurfaceWallFront) | (1u << kSurfaceWallLeft);
constexpr uint32_t kZoneMaskRightFull = (1u << kSurfaceGround) | (1u << kSurfaceWallFront) | (1u << kSurfaceWallRight);
constexpr uint32_t kZoneMaskWallFrontOnly = (1u << kSurfaceWallFront);   // the observability-contrast (degenerate) set
constexpr uint32_t kZoneMaskLeftRightDirect = (1u << kSurfaceGround) | (1u << kSurfaceWallFront);  // the only zones LEFT and RIGHT share directly

// ===========================================================================
// THE RIG — mount geometry + drift, mirrored in scripts/make_synthetic.py.
// Distances in meters, angles documented in degrees / stored in radians
// (CLAUDE.md §12). Indexed by sensor id (kSensorMain/Left/Right).
// ===========================================================================
struct MountSpec { float pos[3]; float yaw_rad; };
constexpr MountSpec kNominalMount[kNumSensors] = {
    /* MAIN  */ { {0.0f, 0.0f, 1.8f}, 0.0f },
    /* LEFT  */ { {1.8f, 0.9f, 0.5f}, 0.9599310886f },    // +55 deg
    /* RIGHT */ { {1.8f, -0.9f, 0.5f}, -0.9599310886f },  // -55 deg
};

// Drift: TRUE = Exp(drift_omega) * R_nominal (LEFT/world-frame perturbation,
// 01.17's retraction convention, cited), t_true = t_nominal + drift_t.
// omega in radians (deg noted alongside), t in meters. MAIN's drift is the
// zero vector — it never moves; it is this project's anchor sensor.
struct DriftSpec { float omega[3]; float t[3]; };
constexpr DriftSpec kDrift[kNumSensors] = {
    /* MAIN  */ { {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f} },
    /* LEFT  */ { {0.009309f, -0.009309f, 0.004655f},        // ~0.8 deg magnitude
                  {0.02592f, 0.01296f, -0.00778f} },          // 3 cm magnitude
    /* RIGHT */ { {-0.009502f, 0.009502f, 0.003800f},         // ~0.8 deg magnitude, DIFFERENT axis than LEFT
                  {-0.01418f, 0.02364f, 0.01182f} },          // 3 cm magnitude
};

// ===========================================================================
// Rigid3 — a rigid-body transform, passed BY VALUE to every kernel that
// needs "the current estimate" (01.17/02.06's shared convention, cited: T
// changes every LM iteration, so a __constant__ upload would be pure
// overhead for 48 bytes). x_base = R * p_sensor + t ("base, expressed from
// sensor" — 09.01's T_parent_child naming, cited).
// ---------------------------------------------------------------------------
struct Rigid3 {
    float R[9];   // row-major 3x3 rotation (orthonormal by construction —
                  // every update goes through so3_exp()'s exact Rodrigues
                  // formula, never an additive approximation)
    float t[3];   // translation (m), base frame
};

constexpr Rigid3 kIdentityRigid3{
    { 1.0f, 0.0f, 0.0f,  0.0f, 1.0f, 0.0f,  0.0f, 0.0f, 1.0f },
    { 0.0f, 0.0f, 0.0f }
};

// ===========================================================================
// Shared rigid-transform primitives (CALIB_HD — see the macro comment
// above). 01.17's exact skew3/mat3_vec/mat3_mul/so3_exp, reused verbatim
// (cited) because these are closed-form SO(3) formulas whose hand
// duplication would be pure transcription — the twin-independence ruling
// in reference_cpu.cpp's header names this as the legitimate case for
// sharing. What is NOT shared: the per-point accumulation loops (GPU block
// reduction vs. CPU serial loop) and the LM control flow (main.cu's host
// orchestration vs. reference_cpu.cpp's own independent loop) — see that
// file's header for the full ruling and the gates that stay blind to it.
// ===========================================================================

// skew3 — the skew-symmetric (cross-product) matrix [v]_x, row-major, such
// that [v]_x * w == v cross w for any w.
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

// mat3_transpose_vec — out = R^T * p (R orthonormal, so R^T == R^-1): used
// to go FROM base frame INTO a sensor's own frame, p_sensor = R^T*(p_base-t).
CALIB_HD inline void mat3_transpose_vec(const float R[9], const float p[3], float out[3])
{
    out[0] = R[0] * p[0] + R[3] * p[1] + R[6] * p[2];
    out[1] = R[1] * p[0] + R[4] * p[1] + R[7] * p[2];
    out[2] = R[2] * p[0] + R[5] * p[1] + R[8] * p[2];
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

// so3_exp — the SO(3) exponential map (Rodrigues' rotation formula), EXACT
// (not a first-order approximation). 01.17's formula and small-angle branch,
// cited verbatim; THEORY.md "The math" re-derives it for this project.
CALIB_HD inline void so3_exp(const float omega[3], float R[9])
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

// rigid3_compose — out = A composed with B: applying out to a point equals
// applying B then A (out.R = A.R*B.R, out.t = A.R*B.t + A.t). Used to build
// T_left_right_via_main = inv(T_main_left) composed with T_main_right (the
// loop-consistency check, main.cu Stage H).
CALIB_HD inline Rigid3 rigid3_compose(const Rigid3& A, const Rigid3& B)
{
    Rigid3 out;
    mat3_mul(A.R, B.R, out.R);
    float Rb_t[3];
    mat3_vec(A.R, B.t, Rb_t);
    out.t[0] = Rb_t[0] + A.t[0];
    out.t[1] = Rb_t[1] + A.t[1];
    out.t[2] = Rb_t[2] + A.t[2];
    return out;
}

// rigid3_inverse — out such that out composed with T == identity.
CALIB_HD inline Rigid3 rigid3_inverse(const Rigid3& T)
{
    Rigid3 out;
    out.R[0] = T.R[0]; out.R[1] = T.R[3]; out.R[2] = T.R[6];
    out.R[3] = T.R[1]; out.R[4] = T.R[4]; out.R[5] = T.R[7];
    out.R[6] = T.R[2]; out.R[7] = T.R[5]; out.R[8] = T.R[8];
    float Rt_t[3];
    mat3_vec(out.R, T.t, Rt_t);
    out.t[0] = -Rt_t[0]; out.t[1] = -Rt_t[1]; out.t[2] = -Rt_t[2];
    return out;
}

// retract — apply a local 6-vector delta = [omega(3); v(3)] to T via the
// decoupled SO(3) x R^3 retraction (01.17's convention, cited):
// R_new = Exp(omega) * R, t_new = t + v.
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

// nominal_extrinsic / true_extrinsic — turn the rig constants above into a
// base<-sensor Rigid3. Host-only (used by main.cu and scripts/*.py-adjacent
// bookkeeping only — never called from a kernel, so no CALIB_HD needed).
inline Rigid3 nominal_extrinsic(int32_t sensor_id)
{
    const MountSpec& m = kNominalMount[sensor_id];
    Rigid3 T;
    const float omega_yaw[3] = { 0.0f, 0.0f, m.yaw_rad };
    so3_exp(omega_yaw, T.R);
    T.t[0] = m.pos[0]; T.t[1] = m.pos[1]; T.t[2] = m.pos[2];
    return T;
}

inline Rigid3 true_extrinsic(int32_t sensor_id)
{
    const Rigid3 nom = nominal_extrinsic(sensor_id);
    const DriftSpec& d = kDrift[sensor_id];
    float dR_full[9];
    so3_exp(d.omega, dR_full);
    Rigid3 T;
    mat3_mul(dR_full, nom.R, T.R);      // R_true = Exp(drift_omega) * R_nominal
    T.t[0] = nom.t[0] + d.t[0];
    T.t[1] = nom.t[1] + d.t[1];
    T.t[2] = nom.t[2] + d.t[2];
    return T;
}

// rotation_angle_deg / translation_error_m — reporting helpers (01.17's
// exact formulas, cited): geodesic rotation distance and Euclidean
// translation distance, used throughout main.cu's gates.
CALIB_HD inline float rotation_angle_deg(const float R[9], const float R_gt[9])
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

CALIB_HD inline float translation_error_m(const float t[3], const float t_gt[3])
{
    const float dx = t[0] - t_gt[0], dy = t[1] - t_gt[1], dz = t[2] - t_gt[2];
    return sqrtf(dx * dx + dy * dy + dz * dz);
}

// ===========================================================================
// Plane representation — one fitted plane per (sensor, surface_id). normal
// is a UNIT vector, oriented via the single fixed interior reference point
// kPlaneOrientRef (see kernels.cu's fit kernels for the derivation of why
// one reference point correctly orients all of ground+3 walls at once).
// centroid is the mean of the points the plane was fit from (meters, base
// frame). valid=0 means "fewer than kMinPlanePoints points were available"
// — callers must never read normal/centroid when valid==0.
// ---------------------------------------------------------------------------
struct Plane {
    float normal[3];
    float centroid[3];
    int32_t valid;
    int32_t count;
};

constexpr Plane kInvalidPlane{ {0,0,0}, {0,0,0}, 0, 0 };

// The single interior reference point every plane's normal is oriented
// toward (base frame, meters) — see kernels.cu's accumulate/orient kernel
// header for why one point works for all four orthogonal surfaces here.
constexpr float kPlaneOrientRef[3] = { 0.0f, 0.0f, 1.0f };

constexpr int kMinPlanePoints = 8;   // fewer points than this: too little to trust a PCA fit (THEORY.md "Numerical considerations")

// point_to_plane_residual_and_jacobian — THE formula this project's
// refinement teaches (02.06's point-to-plane ICP linearization, cited,
// applied here to a FIXED zone-plane target instead of a per-iteration
// nearest-neighbor match — see kernels.cu's assembly kernel header for why
// that swap is an honest, documented simplification). Given the CURRENT
// estimate T (base<-source_sensor) and a target plane (normal n, point on
// plane c, both in BASE frame):
//
//   Pbase = R*p_src + t                      (source point transformed to base)
//   r = n . (Pbase - c)                       (signed point-to-plane distance, scalar, m)
//   d(Pbase)/d(omega) = -[R*p_src]_x          (LEFT/world-frame perturbation,
//                                               01.17's identical derivation)
//   d(Pbase)/d(v)     =  I_3
//   J (1x6, row-major) = n^T * [ -[R*p_src]_x | I_3 ]
//
// Parameters:
//   T        — current estimate (Rigid3, base<-source)
//   p_src    — [3] source point, in the SOURCE SENSOR's own raw frame
//   normal,centroid — [3] each: the TARGET plane, base frame
//   r   OUT  — residual (m)
//   J   OUT  — [6] Jacobian row, col 0..2 = d(r)/d(omega), col 3..5 = d(r)/d(v)
// ---------------------------------------------------------------------------
CALIB_HD inline void point_to_plane_residual_and_jacobian(
    const Rigid3& T, const float p_src[3],
    const float normal[3], const float centroid[3],
    float& r, float J[6])
{
    float Rp[3];
    mat3_vec(T.R, p_src, Rp);                       // R * p_src (rotated-only point)
    const float Pbase[3] = { Rp[0] + T.t[0], Rp[1] + T.t[1], Rp[2] + T.t[2] };

    r = normal[0] * (Pbase[0] - centroid[0]) +
        normal[1] * (Pbase[1] - centroid[1]) +
        normal[2] * (Pbase[2] - centroid[2]);

    float S[9];
    skew3(Rp, S);   // [R*p_src]_x

    // J[0:3] = -n^T * S  (row-vector times matrix = each column's dot with n)
    for (int col = 0; col < 3; ++col) {
        J[col] = -(normal[0] * S[0 * 3 + col] + normal[1] * S[1 * 3 + col] + normal[2] * S[2 * 3 + col]);
    }
    // J[3:6] = n^T * I_3 = n
    J[3] = normal[0]; J[4] = normal[1]; J[5] = normal[2];
}

// ===========================================================================
// hidx / kReduceWidth — the 6x6 symmetric normal-matrix upper-triangle
// packing PLUS the summed cost, 01.17's exact 28-scalar extension of
// 02.06's 27-scalar [H21|g6] convention (cited, reused verbatim): row i's
// valid columns are j=i..5, row_start = {0,6,11,15,18,20}; index 27 holds
// r^T r (here just r^2, since the residual is a SCALAR, not a 2-vector —
// the reduction record shape does not care how many rows a single point's
// Jacobian has, only that H=J^T J and g=J^T r end up in the same 21+6+1
// slots either way).
// ===========================================================================
CALIB_HD inline int hidx(int i, int j)
{
    const int row_start[6] = { 0, 6, 11, 15, 18, 20 };
    return row_start[i] + (j - i);   // caller guarantees i <= j <= 5
}

constexpr int kReduceWidth     = 28;   // 21 (H upper triangle) + 6 (g) + 1 (cost)
constexpr int kThreadsPerBlock = 256;  // repo-default block size: transform / centroid / covariance kernels
constexpr int kThreadsReduce   = 128;  // assembly kernel block size (01.17/02.06's shared-memory-budget default)

// blocks_for — ceil(count/threads), the repo-wide idiom (01.17/02.06/33.01, cited).
CALIB_HD inline int blocks_for(int count, int threads)
{
    return (count + threads - 1) / threads;
}

// ===========================================================================
// LM (Levenberg-Marquardt) hyperparameters for the refinement solve —
// 01.17's exact values and Marquardt-damping-on-the-diagonal rule (cited);
// this project's normal system is 6x6 just like 01.17's, so the same
// battle-tested numbers apply unchanged.
// ===========================================================================
constexpr int    kMaxLmIters        = 20;
constexpr double kLambdaInit        = 1.0e-3;
constexpr double kLambdaUp          = 10.0;
constexpr double kLambdaDown        = 0.3;
constexpr double kLambdaMin         = 1.0e-12;
constexpr double kConvergeDeltaNorm = 1.0e-9;
constexpr double kConvergeCostRel   = 1.0e-12;

// ---------------------------------------------------------------------------
// cholesky6_solve — HOST-ONLY (01.17's exact algorithm and reasoning,
// cited): Cholesky-Crout decomposition of the Marquardt-damped 6x6 normal
// system (H + lambda*diag(H)) delta = -g, plus forward/back substitution.
// Used by BOTH main.cu's GPU-orchestrated LM loop and reference_cpu.cpp's
// independent LM loop — not a twin-independence concern (per 01.17's
// header: neither caller is "the GPU path"; both are host code, and this
// is a generic textbook small-SPD solve, not the project's own formula —
// see 33.01 for the general batched pattern this specializes).
// ---------------------------------------------------------------------------
inline bool cholesky6_solve(const double H21[21], const double g6[6], double lambda, double out_delta[6])
{
    double A[6][6];
    for (int i = 0; i < 6; ++i)
        for (int j = i; j < 6; ++j) {
            const double hij = H21[hidx(i, j)];
            A[i][j] = hij;
            A[j][i] = hij;
        }
    for (int i = 0; i < 6; ++i) A[i][i] *= (1.0 + lambda);

    double L[6][6] = {};
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = A[i][j];
            for (int k = 0; k < j; ++k) sum -= L[i][k] * L[j][k];
            if (i == j) {
                if (sum <= 0.0) return false;
                L[i][i] = sqrt(sum);
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

// ---------------------------------------------------------------------------
// jacobi_eigen_symmetric6 — classic cyclic Jacobi eigenvalue algorithm
// (Golub & Van Loan), specialized to N=6 — 01.16/01.17's exact construction
// (cited, reimplemented per this project's own copy, self-containment
// rule). Used ONLY by main.cu's OBSERVABILITY gate, to turn a converged
// H=J^T J into a condition-number PROXY (max eigenvalue / min eigenvalue).
// ---------------------------------------------------------------------------
inline void jacobi_eigen_symmetric6(double A[6][6], double eigvecs[6][6])
{
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            eigvecs[i][j] = (i == j) ? 1.0 : 0.0;

    const int kSweeps = 12;
    for (int sweep = 0; sweep < kSweeps; ++sweep) {
        for (int p = 0; p < 6; ++p) {
            for (int q = p + 1; q < 6; ++q) {
                if (fabs(A[p][q]) < 1e-15) continue;
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
// jacobi_eigen_3x3 — cyclic Jacobi eigenvalue algorithm specialized to N=3
// (02.03/02.09/01.17's algorithm FAMILY, cited; a compact reimplementation
// per this project's own copy). Used by main.cu to decode a GPU-accumulated
// covariance (see kernels.cu's accumulate_covariance_kernel) into a plane
// normal. reference_cpu.cpp carries its OWN, independently-typed twin
// (jacobi_eigen_3x3_cpu) so PLANE_FIT_TWIN is a genuine cross-check, not a
// tautology through shared code (the twin-independence ruling, cited).
//
// Input:  cov[6] — upper triangle of the symmetric 3x3 covariance, packed
//         (c00,c01,c02,c11,c12,c22), meters^2.
// Output: eigenvalues[3] ASCENDING (lambda0 <= lambda1 <= lambda2);
//         eigenvectors[3][3], eigenvectors[i] the unit eigenvector for
//         eigenvalues[i] — eigenvectors[0] is the (unoriented) plane normal.
// ---------------------------------------------------------------------------
inline void jacobi_eigen_3x3(const float cov[6], float eigenvalues[3], float eigenvectors[3][3])
{
    double A[3][3] = {
        { cov[0], cov[1], cov[2] },
        { cov[1], cov[3], cov[4] },
        { cov[2], cov[4], cov[5] }
    };
    double V[3][3] = { {1,0,0}, {0,1,0}, {0,0,1} };

    const int kSweeps = 8;   // 02.03/02.09's measured-sufficient sweep count for a 3x3
    for (int sweep = 0; sweep < kSweeps; ++sweep) {
        for (int p = 0; p < 3; ++p) {
            for (int q = p + 1; q < 3; ++q) {
                if (fabs(A[p][q]) < 1e-18) continue;
                const double theta = (A[q][q] - A[p][p]) / (2.0 * A[p][q]);
                const double t = (theta >= 0.0 ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1.0));
                const double c = 1.0 / sqrt(t * t + 1.0);
                const double s = t * c;
                const double app = A[p][p], aqq = A[q][q], apq = A[p][q];
                A[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
                A[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
                A[p][q] = A[q][p] = 0.0;
                for (int k = 0; k < 3; ++k) {
                    if (k == p || k == q) continue;
                    const double akp = A[k][p], akq = A[k][q];
                    A[k][p] = A[p][k] = c * akp - s * akq;
                    A[k][q] = A[q][k] = s * akp + c * akq;
                }
                for (int k = 0; k < 3; ++k) {
                    const double vkp = V[k][p], vkq = V[k][q];
                    V[k][p] = c * vkp - s * vkq;
                    V[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }

    // Sort ascending by eigenvalue (a plain 3-element selection sort — the
    // simplest correct thing for exactly 3 elements).
    int order[3] = { 0, 1, 2 };
    for (int i = 0; i < 3; ++i)
        for (int j = i + 1; j < 3; ++j)
            if (A[order[j]][order[j]] < A[order[i]][order[i]]) { int tmp = order[i]; order[i] = order[j]; order[j] = tmp; }

    for (int i = 0; i < 3; ++i) {
        eigenvalues[i] = static_cast<float>(A[order[i]][order[i]]);
        for (int k = 0; k < 3; ++k) eigenvectors[i][k] = static_cast<float>(V[k][order[i]]);
    }
}

// ===========================================================================
// Voxel-hash key packing — the dedup grid's DATA-LAYOUT contract (shared,
// CALIB_HD, per 01.17's "hidx() is data layout, not the algorithm" ruling
// applied here: 02.01/02.09's key-packing SCHEME, cited, reimplemented per
// this project's own copy). What is independent between the GPU dedup path
// and the CPU oracle is the CONTAINER (sorted-array-plus-scan on GPU vs.
// std::unordered_map on CPU) and nothing else — see kernels.cu and
// reference_cpu.cpp.
// ---------------------------------------------------------------------------
constexpr float kDedupCellM = 0.12f;   // dedup voxel size (m) — THEORY.md "Numerical considerations" derives this against this project's noise (6 mm) and drift (cm-to-dm scale) magnitudes

CALIB_HD inline int32_t voxel_coord(float p, float cell)
{
    return static_cast<int32_t>(floorf(p / cell));
}

constexpr int32_t  kHashCoordBias   = 1 << 20;
constexpr uint64_t kHashCoordMask21 = (1ull << 21) - 1ull;

CALIB_HD inline unsigned long long pack_voxel_key(int32_t vx, int32_t vy, int32_t vz)
{
    const uint64_t ux = static_cast<uint64_t>(vx + kHashCoordBias) & kHashCoordMask21;
    const uint64_t uy = static_cast<uint64_t>(vy + kHashCoordBias) & kHashCoordMask21;
    const uint64_t uz = static_cast<uint64_t>(vz + kHashCoordBias) & kHashCoordMask21;
    return ux | (uy << 21) | (uz << 42);
}

CALIB_HD inline unsigned long long point_voxel_key(const float p[3], float cell)
{
    return pack_voxel_key(voxel_coord(p[0], cell), voxel_coord(p[1], cell), voxel_coord(p[2], cell));
}

// ===========================================================================
// GPU kernels (kernels.cu). __global__ signatures are __CUDACC__-fenced —
// only nvcc parses them, matching the template's established pattern.
// ===========================================================================
#ifdef __CUDACC__

// transform_points_kernel — the trivial merge primitive: out[k] = T.R *
// src[k] + T.t, ONE transform for ALL n points (a pure map — see kernels.cu
// for why this is deliberately the simplest kernel in the project, and
// where the REAL merging difficulty (dedup, below) lives instead).
__global__ void transform_points_kernel(int n, const float* __restrict__ src_xyz, Rigid3 T, float* __restrict__ out_xyz);

// transform_points_multi_kernel — the actual MERGE kernel: like the above,
// but each point selects ITS OWN transform via sensor_id[k] indexing into a
// small per-sensor Rigid3 table (kNumSensors=3 entries) — this is how
// main.cu produces the "before" (nominal-extrinsic) and "after" (refined-
// extrinsic) merged clouds from one flat multi-sensor point array.
__global__ void transform_points_multi_kernel(int n, const float* __restrict__ src_xyz,
                                              const int32_t* __restrict__ sensor_id,
                                              const Rigid3* __restrict__ T_per_sensor,
                                              float* __restrict__ out_xyz);

// accumulate_centroid_kernel — PASS 1 of the per-(sensor,surface) plane
// fit: every point atomically adds its position into sums[surface_id] and
// increments counts[surface_id]. Full documentation (why atomics, and why
// that is the RIGHT choice here unlike the refinement assembly kernel
// below) sits with the definition in kernels.cu.
__global__ void accumulate_centroid_kernel(int n, const float* __restrict__ xyz,
                                           const int32_t* __restrict__ surface_id,
                                           float* __restrict__ sums,      // [kNumSurfaces*3]
                                           int32_t* __restrict__ counts); // [kNumSurfaces]

// accumulate_covariance_kernel — PASS 2: given each surface's centroid
// (computed on the host from pass 1's output), every point atomically adds
// its MEAN-SHIFTED outer product into cov_sums[surface_id] (upper triangle,
// 6 floats per surface — 02.09's two-pass mean-shift precision lesson,
// cited, applied at zone granularity instead of per-point-KNN granularity).
__global__ void accumulate_covariance_kernel(int n, const float* __restrict__ xyz,
                                             const int32_t* __restrict__ surface_id,
                                             const float* __restrict__ centroids,  // [kNumSurfaces*3]
                                             float* __restrict__ cov_sums);        // [kNumSurfaces*6]

// assemble_point_to_plane_kernel — the refinement's central NEW GPU
// concept: turn n source points into ONE 6x6 Gauss-Newton normal system via
// a two-stage reduction (01.17/02.06's block-tree-reduce-then-host-finishes
// split, cited). Full documentation (thread mapping, shared-memory budget,
// zone masking) sits with the definition in kernels.cu.
__global__ void assemble_point_to_plane_kernel(
    const float* __restrict__ p_src, const int32_t* __restrict__ surface_id, int n,
    Rigid3 T, const Plane* __restrict__ target_planes, uint32_t zone_mask,
    float* __restrict__ block_partials);

// compute_hash_keys_kernel / mark_boundaries_kernel / gather_representatives_kernel
// — the GPU dedup pipeline's three small kernels (02.01/02.09's sort +
// boundary-scan + compact index build, cited). Full documentation sits with
// the definitions in kernels.cu.
__global__ void compute_hash_keys_kernel(int n, const float* __restrict__ xyz, float cell,
                                         unsigned long long* __restrict__ keys);

__global__ void mark_boundaries_kernel(int n, const unsigned long long* __restrict__ keys_sorted,
                                       int32_t* __restrict__ is_start);

__global__ void gather_representatives_kernel(int num_unique, const int32_t* __restrict__ positions,
                                              const int32_t* __restrict__ idx_sorted,
                                              int32_t* __restrict__ representative_orig_idx);

#endif // __CUDACC__

// ---------------------------------------------------------------------------
// Host launch wrappers (defined in kernels.cu; declared outside the
// __CUDACC__ fence so any translation unit, including main.cu, may call
// them — only their DEFINITIONS need nvcc).
// ---------------------------------------------------------------------------
void launch_transform_points(int n, const float* d_src_xyz, Rigid3 T, float* d_out_xyz);

void launch_transform_points_multi(int n, const float* d_src_xyz, const int32_t* d_sensor_id,
                                   const Rigid3* d_T_per_sensor, float* d_out_xyz);

void launch_accumulate_centroid(int n, const float* d_xyz, const int32_t* d_surface_id,
                                float* d_sums, int32_t* d_counts);

void launch_accumulate_covariance(int n, const float* d_xyz, const int32_t* d_surface_id,
                                  const float* d_centroids, float* d_cov_sums);

// launch_assemble_point_to_plane — returns the number of blocks launched
// (== blocks_for(n, kThreadsReduce)) so main.cu knows how many
// kReduceWidth-wide rows to download and host-sum.
int launch_assemble_point_to_plane(const float* d_p_src, const int32_t* d_surface_id, int n,
                                   Rigid3 T, const Plane* d_target_planes, uint32_t zone_mask,
                                   float* d_block_partials);

// launch_dedup_voxel_grid — the full GPU dedup pipeline (hash -> sort ->
// mark -> compact). d_xyz [n*3] IN; d_representative_idx [n] OUT (only the
// first `return value` entries are meaningful — the kept original indices,
// ascending). Returns the number of UNIQUE voxels (== kept point count).
// Uses Thrust's sort_by_key internally (kernels.cu documents why).
int launch_dedup_voxel_grid(int n, const float* d_xyz, float cell,
                            int32_t* d_representative_idx);

// ===========================================================================
// CPU references (reference_cpu.cpp) — the independent oracle twins. All
// pointers are HOST pointers. See reference_cpu.cpp's file header for
// exactly what is and is not independently reimplemented here.
// ===========================================================================
void transform_points_cpu(int n, const float* src_xyz, const Rigid3& T, float* out_xyz);

// fit_planes_cpu — the CPU twin of the GPU's two-pass accumulate + host
// jacobi_eigen_3x3: an INDEPENDENT single-pass-per-surface (still
// mean-shifted: this function does its OWN two internal passes) loop,
// producing kNumSurfaces Planes from one sensor's tagged point set.
void fit_planes_cpu(int n, const float* xyz, const int32_t* surface_id, Plane out_planes[kNumSurfaces]);

// jacobi_eigen_3x3_cpu — the CPU oracle's OWN, independently-written cyclic
// Jacobi solve (see kernels.cuh's jacobi_eigen_3x3 comment for why this is
// a SEPARATE implementation, not a shared call).
void jacobi_eigen_3x3_cpu(const float cov[6], float eigenvalues[3], float eigenvectors[3][3]);

// assemble_point_to_plane_cpu — the twin of assemble_point_to_plane_kernel:
// same shared point_to_plane_residual_and_jacobian formula, but the
// ACCUMULATION LOOP is independently written, sequential, no reduction tree.
void assemble_point_to_plane_cpu(const float* p_src, const int32_t* surface_id, int n,
                                 const Rigid3& T, const Plane target_planes[kNumSurfaces], uint32_t zone_mask,
                                 double H21[21], double g6[6], double* cost_out);

// run_refinement_lm_cpu — an INDEPENDENTLY-WRITTEN full LM trajectory (own
// loop, own damping/accept-reject flow — see reference_cpu.cpp), used by
// main.cu's TRAJECTORY_TWIN gate.
void run_refinement_lm_cpu(const float* p_src, const int32_t* surface_id, int n,
                           const Plane target_planes[kNumSurfaces], uint32_t zone_mask,
                           Rigid3 T_init, int max_iters,
                           Rigid3& out_T, double* loss_history, int& out_num_iters);

// dedup_voxel_grid_cpu — the independent oracle: an
// std::unordered_map<uint64_t,int32_t> voxel->smallest-original-index map
// (02.09's HashMapCpu precedent, cited), built with a SINGLE ascending pass
// so ties resolve to "smallest index" by construction — the SAME tie-break
// rule the GPU's stable_sort_by_key path produces (kernels.cu documents
// this explicitly), so the two paths' kept-index SETS must match exactly.
// out_representative_idx [n] OUT (first return-value entries meaningful,
// ascending by voxel-first-seen order — NOT necessarily sorted by index;
// main.cu sorts both sides before comparing). Returns unique voxel count.
int dedup_voxel_grid_cpu(int n, const float* xyz, float cell, int32_t* out_representative_idx);

#endif // PROJECT_KERNELS_CUH
