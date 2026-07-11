// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 01.12
//                     Visual servoing: image-Jacobian control loop entirely
//                     on GPU (teaching core: eye-in-hand IBVS)
//
// Two kinds of code live in this file (CLAUDE.md §5):
//
//   1) SHARED SETUP ("data", not algorithm): build_target_and_goal_cpu,
//      generate_batch_init_poses_cpu, generate_basin_grid_poses_cpu. These
//      derive the target/goal geometry and the batch's STARTING poses —
//      deterministic constants and an RNG stream, never touched by the
//      control law. main.cu calls these ONCE to build host arrays that are
//      then fed identically to BOTH the GPU kernel (uploaded) and the CPU
//      oracle below (used directly) — exactly like 08.01's exploration
//      noise, which is generated once on the host and fed to both paths.
//
//   2) THE ORACLE (independently reimplemented ALGORITHM): ibvs_compute_step_cpu
//      and ibvs_batch_cpu are the twins of kernels.cu's ibvs_compute_step /
//      ibvs_batch_kernel — same math, same layouts, written a SECOND time
//      here in plain sequential C++ so the comparison in main.cu tests
//      something real.
//
// Independence ruling (Phase-1 standards retrospective — quoted from
// docs/PROJECT_TEMPLATE/src/reference_cpu.cpp, load-bearing for every
// project in this repo):
//
//   * Data-layout contracts (structs, constants, indexing formulas) MUST be
//     single-sourced in kernels.cuh and shared. Divergent layouts between
//     the twins are a bug class of their own, not "independence".
//   * The ALGORITHMIC CORE should be written twice — independently, in the
//     simplest possible C++ here. That is the default, because the twin
//     comparison only catches bugs the two paths DON'T share.
//   * A shared __host__ __device__ helper is permitted when duplicating it
//     would be pure token-for-token transcription — but then the twin
//     comparison is BLIND to bugs inside that helper, so the project MUST
//     also carry at least one verification gate that does not route through
//     the shared code.
//
// This project's application of the ruling: the CONTROL LAW (quaternion
// math, the per-point interaction-matrix row, the 6x6 damped solve, the
// SE(3) integrator) is written TWICE, independently — once as __device__
// functions in kernels.cu, once as plain functions below — so the
// GPU-vs-CPU comparisons in main.cu (the single-loop trajectory twin, the
// Jacobian/pseudoinverse twin, the batch-statistics twin) are real tests.
// The target/goal geometry and the initial-pose RNG stream, by contrast,
// are pure SETUP data (never part of the control law under test) and are
// single-sourced here, matching the ruling's first bullet — duplicating a
// deterministic constant or an RNG formula would only risk the two paths
// silently disagreeing about the PROBLEM, not test anything about the
// CONTROLLER. On top of the twin comparisons, this project ALSO carries
// INDEPENDENT gates that route through neither twin's shared assumptions —
// the exponential_decay gate is a closed-form control-theory prediction
// (ṡ ≈ -λe locally), and the retreat_pathology gate is a known qualitative
// phenomenon from the visual-servoing literature, not a numeric target
// derived from either implementation (see THEORY.md §How we verify
// correctness) — satisfying the ruling's "at least one gate that does not
// route through the shared code" requirement with room to spare.
//
// Read this after: kernels.cuh.  Read this beside: kernels.cu.
// ===========================================================================

#include "kernels.cuh"

#include <cmath>      // std::sin, std::cos, std::sqrt, std::fabs — the float versions
#include <cstring>    // std::memset

// ===========================================================================
// PART 1 — shared setup ("data"): target/goal geometry, initial-pose RNG.
// ===========================================================================

// build_target_and_goal_cpu — see kernels.cuh for the full derivation. The
// 4 target points are placed by the compile-time constant kTargetHalfSize;
// s* is their projection from the goal pose p_goal=(0,0,-kGoalStandoff),
// q_goal=identity. Because the goal orientation is identity and the world
// frame shares the camera's optical-frame axis convention at the goal (the
// file header's documented simplification), the goal-frame projection has
// a closed form with no rotation involved at all: a world point (X,Y,0) is
// seen by the goal camera at camera-frame (X, Y, kGoalStandoff), so
// x* = X/d, y* = Y/d directly.
void build_target_and_goal_cpu(float target_pts_world[12], float s_star[8])
{
    const float a = kTargetHalfSize;
    // P0=(-a,-a,0) P1=(+a,-a,0) P2=(+a,+a,0) P3=(-a,+a,0) — a square, CCW
    // when viewed from +Z (i.e. from the goal camera looking toward +Z).
    const float pts[4][3] = {
        { -a, -a, 0.0f }, { a, -a, 0.0f }, { a, a, 0.0f }, { -a, a, 0.0f }
    };
    for (int i = 0; i < 4; ++i) {
        target_pts_world[i*3 + 0] = pts[i][0];
        target_pts_world[i*3 + 1] = pts[i][1];
        target_pts_world[i*3 + 2] = pts[i][2];
        // Closed-form goal projection (see comment above): camera-frame
        // point = (X, Y, kGoalStandoff), so x* = X/d, y* = Y/d.
        s_star[i*2 + 0] = pts[i][0] / kGoalStandoff;
        s_star[i*2 + 1] = pts[i][1] / kGoalStandoff;
    }
}

// ---------------------------------------------------------------------------
// xorshift32 / uniform01 — the repo's portable deterministic RNG (same
// formula as 08.01's exploration noise). Implemented ONCE, here, because
// initial-pose generation is SETUP data, not the algorithm under test (see
// file header). base_seed default 42 is chosen by main.cu.
// ---------------------------------------------------------------------------
static inline uint32_t xorshift32(uint32_t& state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

static inline float uniform01(uint32_t& state)   // (0,1] — never exactly 0
{
    return (xorshift32(state) >> 8) * (1.0f / 16777216.0f) + (0.5f / 16777216.0f);
}

// axis_angle_to_quat — build a unit quaternion rotating by `angle` radians
// about `axis` (need not be pre-normalized; normalized here). Falls back
// to the identity if axis is degenerate (near-zero, which only happens for
// astronomically unlikely RNG draws — defensive, not load-bearing).
static inline void axis_angle_to_quat(float ax, float ay, float az, float angle, float q[4])
{
    float n = std::sqrt(ax*ax + ay*ay + az*az);
    if (n < 1e-8f) { q[0] = 1.0f; q[1] = q[2] = q[3] = 0.0f; return; }
    ax /= n; ay /= n; az /= n;
    const float half = 0.5f * angle;
    const float s = std::sin(half);
    q[0] = std::cos(half);
    q[1] = ax * s; q[2] = ay * s; q[3] = az * s;
}

// generate_batch_init_poses_cpu — fill K initial poses across the three
// designed cohorts (kernels.cuh "COHORTS"). For EVERY loop index k,
// regardless of cohort, we draw exactly 7 sequential uniform01() values
// from that loop's own xorshift32 stream (seeded by mixing `seed` with k,
// the same odd-multiplier index-mixing 08.01 uses for its per-tick noise
// stream, here used for per-LOOP separation instead of per-TICK): a fixed
// draw count keeps the generator's code path identical for every k
// (simpler to read and to get right) even though a given cohort only
// consumes a subset of the 7 draws — the unused draws are documented, not
// silently wasted entropy that could confuse a future reader.
void generate_batch_init_poses_cpu(int K, int n_nominal, int n_decay, uint32_t seed, float* poses7)
{
    const int n_retreat_start = n_nominal + n_decay;   // >= this index -> RETREAT cohort
    const float nominalAngleMaxRad = kNominalAngleMaxDeg * (3.14159265358979323846f / 180.0f);
    const float retreatAngleMinRad = kRetreatAngleMinDeg * (3.14159265358979323846f / 180.0f);
    const float retreatAngleMaxRad = kRetreatAngleMaxDeg * (3.14159265358979323846f / 180.0f);

    for (int k = 0; k < K; ++k) {
        uint32_t s = seed + 1000003u * static_cast<uint32_t>(k + 1);   // per-loop stream separation
        if (s == 0) s = 1u;
        float u[7];
        for (int i = 0; i < 7; ++i) u[i] = uniform01(s);

        float dx = 0.0f, dy = 0.0f, dz = 0.0f;
        float q[4] = { 1.0f, 0.0f, 0.0f, 0.0f };

        if (k < n_nominal) {
            // NOMINAL: random box offset + random-axis rotation up to kNominalAngleMaxDeg.
            dx = (2.0f*u[0] - 1.0f) * kNominalPosRange;
            dy = (2.0f*u[1] - 1.0f) * kNominalPosRange;
            dz = (2.0f*u[2] - 1.0f) * kNominalPosRange;
            const float ax = 2.0f*u[3] - 1.0f, ay = 2.0f*u[4] - 1.0f, az = 2.0f*u[5] - 1.0f;
            const float angle = u[6] * nominalAngleMaxRad;
            axis_angle_to_quat(ax, ay, az, angle, q);
        } else if (k < n_retreat_start) {
            // DECAY: small PURE-TRANSLATION offset, zero rotation.
            const float dirx = 2.0f*u[0] - 1.0f, diry = 2.0f*u[1] - 1.0f, dirz = 2.0f*u[2] - 1.0f;
            float n = std::sqrt(dirx*dirx + diry*diry + dirz*dirz);
            if (n < 1e-6f) n = 1.0f;   // defensive: astronomically unlikely
            const float mag = kDecayPosMin + u[3] * (kDecayPosMax - kDecayPosMin);
            dx = dirx / n * mag; dy = diry / n * mag; dz = dirz / n * mag;
            // q stays identity: zero rotation is this cohort's whole point.
        } else {
            // RETREAT: near-180-degree rotation about the camera's OWN
            // optical (Z) axis (which, at zero rotation, coincides with
            // world +Z — see the file header's frame-convention note),
            // plus a small position jitter for realism/variety.
            const float angle = retreatAngleMinRad + u[0] * (retreatAngleMaxRad - retreatAngleMinRad);
            const float sign = (u[1] < 0.5f) ? -1.0f : 1.0f;
            axis_angle_to_quat(0.0f, 0.0f, sign, angle, q);
            dx = (2.0f*u[2] - 1.0f) * kRetreatPosJitter;
            dy = (2.0f*u[3] - 1.0f) * kRetreatPosJitter;
            dz = (2.0f*u[4] - 1.0f) * kRetreatPosJitter;
            // u[5], u[6] unused for this cohort — kept for the uniform 7-draw count (see header).
        }

        float* out = &poses7[k * 7];
        out[0] = dx;
        out[1] = dy;
        out[2] = -kGoalStandoff + dz;   // perturb standoff distance along Z
        out[3] = q[0]; out[4] = q[1]; out[5] = q[2]; out[6] = q[3];
    }
}

// generate_basin_grid_poses_cpu — deterministic (dx,dy) grid at dz=0, zero
// rotation, for the basin_map.ppm artifact. No RNG: a scenario grid is
// constants (the same reasoning 08.01 uses for its scenario CSV).
void generate_basin_grid_poses_cpu(int G, float* poses7)
{
    for (int iy = 0; iy < G; ++iy) {
        for (int ix = 0; ix < G; ++ix) {
            const int k = iy * G + ix;
            const float u = (G > 1) ? (float)ix / (float)(G - 1) : 0.5f;
            const float v = (G > 1) ? (float)iy / (float)(G - 1) : 0.5f;
            const float dx = (2.0f*u - 1.0f) * kBasinPosRange;
            const float dy = (2.0f*v - 1.0f) * kBasinPosRange;
            float* out = &poses7[k * 7];
            out[0] = dx; out[1] = dy; out[2] = -kGoalStandoff;
            out[3] = 1.0f; out[4] = 0.0f; out[5] = 0.0f; out[6] = 0.0f;   // identity: zero rotation
        }
    }
}

// ===========================================================================
// PART 2 — the ORACLE: an INDEPENDENT reimplementation of the control law
// (see the file header's ruling). Every helper below is written fresh —
// deliberately NOT calling anything in kernels.cu — even though the math
// is, of necessity, the same math (a rotation is a rotation); the
// IMPLEMENTATION (variable names, loop order, intermediate steps) is not
// shared code, so a bug introduced in one side's arithmetic has a real
// chance of NOT appearing on the other side, which is the entire point.
// ===========================================================================

static void quat_conj_cpu(const float q[4], float out[4])
{
    out[0] = q[0]; out[1] = -q[1]; out[2] = -q[2]; out[3] = -q[3];
}

static void quat_mul_cpu(const float a[4], const float b[4], float out[4])
{
    out[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3];
    out[1] = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2];
    out[2] = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1];
    out[3] = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0];
}

static void quat_normalize_cpu(float q[4])
{
    const float n = std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    const float inv = (n > 1e-20f) ? (1.0f / n) : 1.0f;
    q[0] *= inv; q[1] *= inv; q[2] *= inv; q[3] *= inv;
}

static void rotate_by_quat_cpu(const float q[4], const float v[3], float out[3])
{
    const float qvx = q[1], qvy = q[2], qvz = q[3];
    const float tx = qvy*v[2] - qvz*v[1];
    const float ty = qvz*v[0] - qvx*v[2];
    const float tz = qvx*v[1] - qvy*v[0];
    const float ux = qvy*tz - qvz*ty;
    const float uy = qvz*tx - qvx*tz;
    const float uz = qvx*ty - qvy*tx;
    out[0] = v[0] + 2.0f*q[0]*tx + 2.0f*ux;
    out[1] = v[1] + 2.0f*q[0]*ty + 2.0f*uy;
    out[2] = v[2] + 2.0f*q[0]*tz + 2.0f*uz;
}

static void rotate_by_conj_quat_cpu(const float q[4], const float v[3], float out[3])
{
    float qc[4];
    quat_conj_cpu(q, qc);
    rotate_by_quat_cpu(qc, v, out);
}

// Dense 6x6 Cholesky, in place — same textbook algorithm as kernels.cu's
// chol6_inplace, an independent transcription (plain double-loop, no
// #pragma unroll — this file is compiled by cl.exe, which does not use
// that pragma, and clarity over speed is this file's whole purpose).
static void chol6_inplace_cpu(float A[6][6])
{
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j <= i; ++j) {
            float sum = A[i][j];
            for (int k = 0; k < j; ++k) sum -= A[i][k] * A[j][k];
            if (i == j) {
                float d = sum > 1e-12f ? sum : 1e-12f;
                A[i][i] = std::sqrt(d);
            } else {
                A[i][j] = sum / A[j][j];
            }
        }
    }
}

static void chol6_solve_cpu(const float R[6][6], const float b[6], float x[6])
{
    float y[6];
    for (int i = 0; i < 6; ++i) {
        float sum = b[i];
        for (int k = 0; k < i; ++k) sum -= R[i][k] * y[k];
        y[i] = sum / R[i][i];
    }
    for (int i = 5; i >= 0; --i) {
        float sum = y[i];
        for (int k = i + 1; k < 6; ++k) sum -= R[k][i] * x[k];
        x[i] = sum / R[i][i];
    }
}

static float cholesky_diag_ratio_cpu(const float R[6][6])
{
    float lo = R[0][0], hi = R[0][0];
    for (int i = 1; i < 6; ++i) {
        lo = (R[i][i] < lo) ? R[i][i] : lo;
        hi = (R[i][i] > hi) ? R[i][i] : hi;
    }
    return lo / (hi > 1e-12f ? hi : 1e-12f);
}

// ibvs_compute_step_cpu — the oracle's per-step function; see kernels.cuh
// for the full parameter contract (identical to the GPU twin's).
void ibvs_compute_step_cpu(const float pose[7], int variant,
                           const float target_pts_world[12], const float s_star[8],
                           float v_out[6], float A_out[36], float b_out[6],
                           float feat_out[8], float* err_norm_out,
                           float* cond_proxy_out, float* zmax_out, float* featmax_out)
{
    const float p[3] = { pose[0], pose[1], pose[2] };
    const float q[4] = { pose[3], pose[4], pose[5], pose[6] };

    float A[6][6];
    for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) A[i][j] = 0.0f;
    float b[6] = { 0,0,0,0,0,0 };

    float sumSqErr = 0.0f, zmax = 0.0f, featmax = 0.0f;

    for (int i = 0; i < 4; ++i) {
        const float Pw[3] = {
            target_pts_world[i*3+0] - p[0],
            target_pts_world[i*3+1] - p[1],
            target_pts_world[i*3+2] - p[2]
        };
        float Pc[3];
        rotate_by_conj_quat_cpu(q, Pw, Pc);

        const float Z = Pc[2];
        const float x = Pc[0] / Z;
        const float y = Pc[1] / Z;
        feat_out[2*i+0] = x; feat_out[2*i+1] = y;

        const float ex = x - s_star[2*i+0];
        const float ey = y - s_star[2*i+1];
        sumSqErr += ex*ex + ey*ey;

        zmax    = (Z > zmax) ? Z : zmax;
        const float ax = std::fabs(x), ay = std::fabs(y);
        featmax = (ax > featmax) ? ax : featmax;
        featmax = (ay > featmax) ? ay : featmax;

        float xj, yj, Zj;
        if (variant == kVariantTrueDepth) {
            xj = x; yj = y; Zj = Z;
        } else if (variant == kVariantFixedDepth) {
            xj = x; yj = y; Zj = kGoalStandoff;
        } else {
            xj = s_star[2*i+0]; yj = s_star[2*i+1]; Zj = kGoalStandoff;
        }

        const float invZ = 1.0f / Zj;
        const float Lrow0[6] = { -invZ, 0.0f,  xj*invZ,    xj*yj,       -(1.0f+xj*xj), yj  };
        const float Lrow1[6] = {  0.0f, -invZ, yj*invZ,    1.0f+yj*yj,  -xj*yj,       -xj  };

        for (int r = 0; r < 6; ++r) {
            b[r] += Lrow0[r]*ex + Lrow1[r]*ey;
            for (int c = 0; c < 6; ++c)
                A[r][c] += Lrow0[r]*Lrow0[c] + Lrow1[r]*Lrow1[c];
        }
    }

    for (int i = 0; i < 6; ++i) A[i][i] += kDampingMu;

    if (A_out) for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) A_out[i*6+j] = A[i][j];
    if (b_out) for (int i = 0; i < 6; ++i) b_out[i] = b[i];

    chol6_inplace_cpu(A);
    float x[6];
    chol6_solve_cpu(A, b, x);
    const float condProxy = cholesky_diag_ratio_cpu(A);

    for (int i = 0; i < 6; ++i) v_out[i] = -kLambda * x[i];

    *err_norm_out = std::sqrt(sumSqErr);
    *cond_proxy_out = condProxy;
    *zmax_out = zmax;
    *featmax_out = featmax;
}

// Independent twin of kernels.cu's ibvs_feature_error_norm — see that
// function's header for why it exists (the non-converged exit path).
static float ibvs_feature_error_norm_cpu(const float pose[7],
                                         const float target_pts_world[12], const float s_star[8])
{
    const float p[3] = { pose[0], pose[1], pose[2] };
    const float q[4] = { pose[3], pose[4], pose[5], pose[6] };
    float sumSqErr = 0.0f;
    for (int i = 0; i < 4; ++i) {
        const float Pw[3] = {
            target_pts_world[i*3+0]-p[0], target_pts_world[i*3+1]-p[1], target_pts_world[i*3+2]-p[2]
        };
        float Pc[3];
        rotate_by_conj_quat_cpu(q, Pw, Pc);
        const float x = Pc[0]/Pc[2], y = Pc[1]/Pc[2];
        const float ex = x - s_star[i*2+0], ey = y - s_star[i*2+1];
        sumSqErr += ex*ex + ey*ey;
    }
    return std::sqrt(sumSqErr);
}

// Independent twin of kernels.cu's ibvs_integrate — see that function's
// header for the SE(3) integration scheme (exact rotation / first-order
// translation) this reproduces line-by-line, spelled with std:: functions.
static void ibvs_integrate_cpu(float pose[7], const float v[6], float dt)
{
    const float vlin[3] = { v[0], v[1], v[2] };
    const float wang[3] = { v[3], v[4], v[5] };
    const float q[4]    = { pose[3], pose[4], pose[5], pose[6] };

    float dp[3];
    rotate_by_quat_cpu(q, vlin, dp);
    pose[0] += dt * dp[0]; pose[1] += dt * dp[1]; pose[2] += dt * dp[2];

    const float theta[3] = { dt*wang[0], dt*wang[1], dt*wang[2] };
    const float angle = std::sqrt(theta[0]*theta[0] + theta[1]*theta[1] + theta[2]*theta[2]);
    float qDelta[4];
    if (angle > 1e-8f) {
        const float half = 0.5f * angle;
        const float s = std::sin(half) / angle;
        qDelta[0] = std::cos(half);
        qDelta[1] = theta[0]*s; qDelta[2] = theta[1]*s; qDelta[3] = theta[2]*s;
    } else {
        qDelta[0] = 1.0f; qDelta[1] = 0.5f*theta[0]; qDelta[2] = 0.5f*theta[1]; qDelta[3] = 0.5f*theta[2];
    }
    float qNew[4];
    quat_mul_cpu(q, qDelta, qNew);
    quat_normalize_cpu(qNew);
    pose[3] = qNew[0]; pose[4] = qNew[1]; pose[5] = qNew[2]; pose[6] = qNew[3];
}

// ibvs_batch_cpu — the oracle's twin of ibvs_batch_kernel: K independent
// loops, run sequentially, identical convergence/trace/output semantics.
void ibvs_batch_cpu(int K, int variant, const float* init_poses,
                    const float target_pts_world[12], const float s_star[8],
                    const int* trace_idx, int trace_count,
                    float* out_converged, float* out_steps, float* out_final_err,
                    float* out_cond_min, float* out_zmax, float* out_featmax,
                    float* out_trace)
{
    for (int k = 0; k < K; ++k) {
        float pose[7];
        for (int i = 0; i < 7; ++i) pose[i] = init_poses[k*7 + i];

        int traceSlot = -1;
        for (int i = 0; i < trace_count; ++i) if (trace_idx[i] == k) { traceSlot = i; break; }

        float condMin = 1.0f, zMax = 0.0f, featMax = 0.0f, lastErrNorm = 0.0f;
        bool converged = false;
        int stepsUsed = 0;

        for (int t = 0; t < kMaxSteps; ++t) {
            float v[6], feat[8], errNorm, condProxy, zStep, featStep;
            ibvs_compute_step_cpu(pose, variant, target_pts_world, s_star,
                                  v, nullptr, nullptr, feat, &errNorm, &condProxy, &zStep, &featStep);

            condMin = (condProxy < condMin) ? condProxy : condMin;
            zMax    = (zStep > zMax) ? zStep : zMax;
            featMax = (featStep > featMax) ? featStep : featMax;
            lastErrNorm = errNorm;

            if (traceSlot >= 0 && out_trace) {
                float* row = out_trace
                           + (size_t)traceSlot * (kMaxSteps + 1) * kTraceRowStride
                           + (size_t)t * kTraceRowStride;
                row[0] = (float)t;
                row[1] = pose[0]; row[2] = pose[1]; row[3] = pose[2];
                for (int i = 0; i < 8; ++i) row[4+i] = feat[i];
            }

            if (errNorm < kConvergeEps) { converged = true; stepsUsed = t; break; }

            ibvs_integrate_cpu(pose, v, kDt);
            stepsUsed = t + 1;
        }

        const float finalErr = converged ? lastErrNorm
                                         : ibvs_feature_error_norm_cpu(pose, target_pts_world, s_star);

        out_converged[k]  = converged ? 1.0f : 0.0f;
        out_steps[k]       = (float)stepsUsed;
        out_final_err[k]   = finalErr;
        out_cond_min[k]    = condMin;
        out_zmax[k]        = zMax;
        out_featmax[k]     = featMax;
    }
}
