// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 09.01
//                     Batched forward kinematics (10⁵ configurations —
//                     the foundation for everything above)
//
// WHY a CPU implementation in a GPU repo? (same two reasons as every project,
// CLAUDE.md §5): (1) the CORRECTNESS ORACLE — GPU code fails in ways CPU code
// cannot (wrong thread indexing, missed tails, stale device memory), and a
// dead-simple sequential twin is the ground truth main.cu compares against;
// (2) the TEACHING BASELINE — diff this file against kernels.cu to see that
// only the parallel plumbing differs: same recurrence, same helpers, same
// Shepperd conversion, in the same order, on purpose.
//
// One honest difference to know about: the trig. This file uses
// std::sin/std::cos; the kernel uses CUDA's sincosf. Both are correctly
// rounded to within ~1-2 ulp but NOT to identical bits — which is one of the
// reasons main.cu compares within a tolerance instead of bitwise
// (THEORY.md §Numerical considerations).
//
// All layout contracts (model rows, q, pose) live in kernels.cuh — the one
// place; this file just implements them for host memory, single-threaded,
// clarity over speed always.
//
// Read this after: kernels.cuh.  Read this beside: kernels.cu.
// ===========================================================================

#include "kernels.cuh"   // shared layout contracts + this function's signature

#include <cmath>         // std::sin, std::cos, std::sqrt

// ---------------------------------------------------------------------------
// Host twins of the device math helpers in kernels.cu (deliberate, documented
// duplication — the files are meant to be diffable side by side).
// ---------------------------------------------------------------------------

// C = A·B, row-major 3×3; C must not alias A or B.
static void mat3_mul(const float* A, const float* B, float* C)
{
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            C[i * 3 + j] = A[i * 3 + 0] * B[0 * 3 + j]
                         + A[i * 3 + 1] * B[1 * 3 + j]
                         + A[i * 3 + 2] * B[2 * 3 + j];
}

// Rotation matrix from a normalized quaternion (w,x,y,z) — see kernels.cu.
static void quat_to_mat3(float w, float x, float y, float z, float* R)
{
    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, xz = x * z, yz = y * z;
    const float wx = w * x, wy = w * y, wz = w * z;
    R[0] = 1.0f - 2.0f * (yy + zz); R[1] = 2.0f * (xy - wz);        R[2] = 2.0f * (xz + wy);
    R[3] = 2.0f * (xy + wz);        R[4] = 1.0f - 2.0f * (xx + zz); R[5] = 2.0f * (yz - wx);
    R[6] = 2.0f * (xz - wy);        R[7] = 2.0f * (yz + wx);        R[8] = 1.0f - 2.0f * (xx + yy);
}

// Rodrigues axis-angle → matrix; axis must be unit length (see kernels.cu).
static void rodrigues(float ax, float ay, float az, float ang, float* R)
{
    const float s = std::sin(ang), c = std::cos(ang);
    const float C = 1.0f - c;
    R[0] = c + ax * ax * C;        R[1] = ax * ay * C - az * s;   R[2] = ax * az * C + ay * s;
    R[3] = ay * ax * C + az * s;   R[4] = c + ay * ay * C;        R[5] = ay * az * C - ax * s;
    R[6] = az * ax * C - ay * s;   R[7] = az * ay * C + ax * s;   R[8] = c + az * az * C;
}

// Shepperd's stable matrix → quaternion (w,x,y,z); same four-branch
// largest-divisor selection as the kernel, same non-canonical hemisphere.
static void mat3_to_quat(const float* R, float* q)
{
    const float tr = R[0] + R[4] + R[8];
    float w, x, y, z;
    if (tr > 0.0f) {
        float s = std::sqrt(tr + 1.0f) * 2.0f;
        w = 0.25f * s;
        x = (R[7] - R[5]) / s;
        y = (R[2] - R[6]) / s;
        z = (R[3] - R[1]) / s;
    } else if (R[0] > R[4] && R[0] > R[8]) {
        float s = std::sqrt(1.0f + R[0] - R[4] - R[8]) * 2.0f;
        w = (R[7] - R[5]) / s;
        x = 0.25f * s;
        y = (R[1] + R[3]) / s;
        z = (R[2] + R[6]) / s;
    } else if (R[4] > R[8]) {
        float s = std::sqrt(1.0f + R[4] - R[0] - R[8]) * 2.0f;
        w = (R[2] - R[6]) / s;
        x = (R[1] + R[3]) / s;
        y = 0.25f * s;
        z = (R[5] + R[7]) / s;
    } else {
        float s = std::sqrt(1.0f + R[8] - R[0] - R[4]) * 2.0f;
        w = (R[3] - R[1]) / s;
        x = (R[2] + R[6]) / s;
        y = (R[5] + R[7]) / s;
        z = 0.25f * s;
    }
    // Same defensive renormalization as the kernel (1/sqrt instead of
    // rsqrtf — the host has no fast-approximate version, and doesn't need one).
    const float inv_n = 1.0f / std::sqrt(w * w + x * x + y * y + z * z);
    q[0] = w * inv_n; q[1] = x * inv_n; q[2] = y * inv_n; q[3] = z * inv_n;
}

// ---------------------------------------------------------------------------
// batched_fk_cpu — sequential FK over the whole batch.
//
// The GPU gives each configuration its own thread; this loop visits them one
// after another — same recurrence per configuration:
//     p ← p + R·t_j ;  R ← R·R_fix_j ;  R ← R·Rot(axis_j, q_j)
// The model arrives as a plain argument (CPUs have no constant memory —
// that difference between the twins is itself a lesson; see THEORY.md).
// ---------------------------------------------------------------------------
void batched_fk_cpu(int nj, const float* model,
                    int count, const float* q, float* pose)
{
    for (int k = 0; k < count; ++k) {          // one configuration at a time
        // Chain state: T_base_link = (R, p), starting at the identity.
        float R[9] = { 1.0f, 0.0f, 0.0f,
                       0.0f, 1.0f, 0.0f,
                       0.0f, 0.0f, 1.0f };
        float p[3] = { 0.0f, 0.0f, 0.0f };

        for (int j = 0; j < nj; ++j) {
            const float* m = &model[j * kModelStride];   // joint j's model row

            // (1) walk out along the link (rotate t_j into the base frame)
            const float tx = m[0], ty = m[1], tz = m[2];
            p[0] += R[0] * tx + R[1] * ty + R[2] * tz;
            p[1] += R[3] * tx + R[4] * ty + R[5] * tz;
            p[2] += R[6] * tx + R[7] * ty + R[8] * tz;

            // (2) the link's fixed rotation
            float Rf[9], Rt[9];
            quat_to_mat3(m[3], m[4], m[5], m[6], Rf);
            mat3_mul(R, Rf, Rt);

            // (3) the joint's motion
            float Rj[9];
            rodrigues(m[7], m[8], m[9], q[k * nj + j], Rj);
            mat3_mul(Rt, Rj, R);
        }

        // Emit the message-shaped pose (layout: kernels.cuh).
        float quat[4];
        mat3_to_quat(R, quat);
        float* out = &pose[k * kPoseStride];
        out[0] = p[0];    out[1] = p[1];    out[2] = p[2];
        out[3] = quat[0]; out[4] = quat[1]; out[5] = quat[2]; out[6] = quat[3];
    }
}
