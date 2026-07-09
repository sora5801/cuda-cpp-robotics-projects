// ===========================================================================
// kernels.cu — GPU implementation for project 09.01
//              Batched forward kinematics (10⁵ configurations — the
//              foundation for everything above)
//
// The big idea
// ------------
// Forward kinematics (FK) answers "where is the hand?": given joint angles
// q, compose the chain's transforms to get the end-effector pose. One FK
// evaluation is trivial (~200 flops for 6 joints). But sampling-based
// robotics never wants one: an IK solver with random restarts (09.05) does
// FK per seed per iteration; a grasp-reachability ranker (19.08) scores 10⁵
// candidate configurations; a sampling controller checks pose costs across
// rollouts. So the GPU mapping is the repo's foundational batch pattern,
// the same one project 33.01 teaches:
//
//     one thread = one configuration's whole FK chain, in registers.
//
// What is NEW here beyond 33.01:
//   * __constant__ memory — the robot model is identical for every thread,
//     and constant memory broadcasts a uniform read to the whole warp.
//   * Composing rigid transforms (R, p pairs) instead of raw matmuls.
//   * Rodrigues' rotation formula (axis-angle → matrix) — the workhorse of
//     kinematics.
//   * Numerically-stable rotation-matrix → quaternion conversion, with the
//     repo's (w,x,y,z) convention and the double-cover caveat.
//
// Frames and units (SYSTEM_DESIGN conventions): right-handed, SI. The chain
// state carried down the loop is T_base_linkj = (R, p): R is the 3×3
// rotation base←linkj (row-major), p the linkj origin in base coordinates,
// meters. The recurrence per joint (layout & math contract in kernels.cuh):
//
//     p ← p + R·t_j                (walk out along the link geometry)
//     R ← R · R_fix_j              (the link's fixed orientation change)
//     R ← R · Rot(axis_j, q_j)     (the joint's motion)
//
// Read this after: kernels.cuh.  Companion oracle: reference_cpu.cpp
// (a deliberate line-by-line twin — diff the two; only the plumbing differs).
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"      // CUDA_CHECK / CUDA_CHECK_LAST_ERROR (§6.1 rule 7)

#include <cstdio>                   // fprintf for loud aborts
#include <cstdlib>                  // std::exit

// ---------------------------------------------------------------------------
// The robot model in __constant__ memory.
//
// Why __constant__ and not kernel arguments or global memory? All threads
// read the SAME model values at the same loop iteration. Constant memory is
// backed by a small per-SM cache with a broadcast path: a warp reading one
// uniform address costs a single transaction — as cheap as a register read
// once cached. Kernel arguments could carry ~4 KB too, but a persistent
// symbol mirrors how a real stack loads its URDF once at startup, and
// teaches cudaMemcpyToSymbol. (If threads indexed DIFFERENT model entries,
// constant memory would serialize — its cache serves one address per cycle;
// that trap is worth remembering.)
// ---------------------------------------------------------------------------
__constant__ float c_model[kMaxJoints * kModelStride];  // [nj*10] model rows (layout: kernels.cuh)
__constant__ int   c_nj;                                // number of joints actually uploaded

// Host-side mirror of c_nj so the launcher can validate without a device
// read-back. File-scope static: private to this translation unit.
static int g_nj = 0;

// ---------------------------------------------------------------------------
// Small device math helpers.
//
// All are __forceinline__: they compile into the kernel's straight-line code
// with no call overhead, keeping every intermediate in registers. The CPU
// oracle carries identical plain-C++ twins (deliberate duplication — the
// pair is meant to be diffed; see reference_cpu.cpp's header).
// ---------------------------------------------------------------------------

// mat3_mul — C = A·B for row-major 3×3. 27 FMAs, fully unrolled.
// C must not alias A or B (call sites use a temporary).
__device__ __forceinline__ void mat3_mul(const float* A, const float* B, float* C)
{
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j)
            C[i * 3 + j] = fmaf(A[i * 3 + 0], B[0 * 3 + j],
                           fmaf(A[i * 3 + 1], B[1 * 3 + j],
                                A[i * 3 + 2] * B[2 * 3 + j]));
}

// quat_to_mat3 — rotation matrix from a normalized quaternion (w,x,y,z).
// Standard expansion of R(q) = (w²−v·v)I + 2vvᵀ + 2w[v]ₓ; the loader in
// main.cu guarantees ‖q‖=1, so no renormalization here (documented trust).
__device__ __forceinline__ void quat_to_mat3(float w, float x, float y, float z, float* R)
{
    // Products used by every entry — computed once, register-resident.
    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, xz = x * z, yz = y * z;
    const float wx = w * x, wy = w * y, wz = w * z;
    R[0] = 1.0f - 2.0f * (yy + zz); R[1] = 2.0f * (xy - wz);        R[2] = 2.0f * (xz + wy);
    R[3] = 2.0f * (xy + wz);        R[4] = 1.0f - 2.0f * (xx + zz); R[5] = 2.0f * (yz - wx);
    R[6] = 2.0f * (xz - wy);        R[7] = 2.0f * (yz + wx);        R[8] = 1.0f - 2.0f * (xx + yy);
}

// rodrigues — rotation matrix for angle `ang` (rad) about UNIT axis (ax,ay,az):
//     R = I + sinθ·K + (1−cosθ)·K²,   K = skew(axis)
// written out entry-wise (no K matrix materialized). This is THE kinematics
// workhorse: every revolute joint is one Rodrigues evaluation. The axis is
// unit-length by the loader's validation — an unnormalized axis would scale
// the rotation nonuniformly (a classic silent-corruption bug).
__device__ __forceinline__ void rodrigues(float ax, float ay, float az, float ang, float* R)
{
    float s, c;
    // sincosf computes both trig values in one call (they share the argument
    // reduction). We use the PRECISE library version, not the __sincosf fast
    // intrinsic: the intrinsic's error grows near |ang| = π, exactly where
    // wrapped joint angles live. The oracle uses std::sin/std::cos; rounding
    // still differs between the two paths — one more reason comparisons are
    // tolerance-based, never bitwise (THEORY.md §numerics).
    sincosf(ang, &s, &c);
    const float C = 1.0f - c;
    R[0] = c + ax * ax * C;        R[1] = ax * ay * C - az * s;   R[2] = ax * az * C + ay * s;
    R[3] = ay * ax * C + az * s;   R[4] = c + ay * ay * C;        R[5] = ay * az * C - ax * s;
    R[6] = az * ax * C - ay * s;   R[7] = az * ay * C + ax * s;   R[8] = c + az * az * C;
}

// mat3_to_quat — numerically-stable rotation-matrix → quaternion (w,x,y,z),
// Shepperd's method: four algebraically-equivalent formulas exist, each
// dividing by one quaternion component; pick the formula whose divisor is
// LARGEST (via the trace comparison) so we never divide by a near-zero
// number. Cost: a 4-way branch — warp lanes may diverge, but each path is
// ~10 flops, so divergence is negligible (and it affects speed only, never
// values — results stay deterministic).
// Output is normalized to ~FP32 precision but NOT hemisphere-canonicalized:
// q and −q encode the same rotation (the double cover), and which one this
// returns depends on the branch taken. Consumers compare accordingly
// (see the comparator in main.cu).
__device__ __forceinline__ void mat3_to_quat(const float* R, float* q)
{
    const float tr = R[0] + R[4] + R[8];        // trace = 4w² − 1 when w dominates
    float w, x, y, z;
    if (tr > 0.0f) {
        float s = sqrtf(tr + 1.0f) * 2.0f;      // s = 4w — safely large on this branch
        w = 0.25f * s;
        x = (R[7] - R[5]) / s;                  // off-diagonal differences encode the axis
        y = (R[2] - R[6]) / s;
        z = (R[3] - R[1]) / s;
    } else if (R[0] > R[4] && R[0] > R[8]) {    // r00 largest → x component dominates
        float s = sqrtf(1.0f + R[0] - R[4] - R[8]) * 2.0f;   // s = 4x
        w = (R[7] - R[5]) / s;
        x = 0.25f * s;
        y = (R[1] + R[3]) / s;                  // off-diagonal SUMS on the non-w branches
        z = (R[2] + R[6]) / s;
    } else if (R[4] > R[8]) {                   // r11 largest → y dominates
        float s = sqrtf(1.0f + R[4] - R[0] - R[8]) * 2.0f;   // s = 4y
        w = (R[2] - R[6]) / s;
        x = (R[1] + R[3]) / s;
        y = 0.25f * s;
        z = (R[5] + R[7]) / s;
    } else {                                    // r22 largest → z dominates
        float s = sqrtf(1.0f + R[8] - R[0] - R[4]) * 2.0f;   // s = 4z
        w = (R[3] - R[1]) / s;
        x = (R[2] + R[6]) / s;
        y = (R[5] + R[7]) / s;
        z = 0.25f * s;
    }
    // One defensive renormalization: six chained FP32 rotations drift ‖q‖
    // by ~1e-6; normalizing here keeps the OUTPUT contract ("normalized
    // quaternion") true to FP32 precision. rsqrtf's ~2-ulp error is fine at
    // this tolerance.
    const float inv_n = rsqrtf(fmaf(w, w, fmaf(x, x, fmaf(y, y, z * z))));
    q[0] = w * inv_n; q[1] = x * inv_n; q[2] = y * inv_n; q[3] = z * inv_n;
}

// ===========================================================================
// The FK kernel: one thread = one configuration.
//
// Thread-to-data mapping: thread k = blockIdx.x*blockDim.x + threadIdx.x
// owns configuration k. Grid: ceil(count/256) × 256 (repo default geometry;
// tail guard as always).
//
// Memory spaces per thread:
//   registers : the running (R, p) chain state + temporaries (~40 regs)
//   constant  : the robot model — uniform warp-broadcast reads (see above)
//   global    : q read once per joint (thread k reads q[k*nj + j]; at fixed
//               j a warp's reads sit 24 bytes apart for nj=6 — imperfectly
//               coalesced, small, and dwarfed by the arithmetic),
//               pose written once at the end (7 floats per thread).
// No shared memory: threads share nothing per-configuration (the model
// sharing is exactly what constant memory already handles better).
// ===========================================================================
__global__ void batched_fk_kernel(const float* __restrict__ q,      // [count*c_nj] joint angles (rad)
                                  float*       __restrict__ pose,   // [count*7] OUT poses (p, quat wxyz)
                                  int count)                        // number of configurations
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's configuration index
    if (k >= count) return;                               // ragged-tail guard

    // Chain state: T_base_link as (R, p), initialized to identity — link 0's
    // parent IS the base frame.
    float R[9] = { 1.0f, 0.0f, 0.0f,
                   0.0f, 1.0f, 0.0f,
                   0.0f, 0.0f, 1.0f };   // rotation base←current-link, row-major
    float p[3] = { 0.0f, 0.0f, 0.0f };   // current link origin in base frame (m)

    const int nj = c_nj;                  // one broadcast read, then a register

    // This loop has a RUNTIME trip count (nj), so it does not fully unroll —
    // fine, because all array indexing inside stays literal (the helpers
    // unroll internally). The chain is inherently SEQUENTIAL: joint j needs
    // joint j−1's frame. Parallelism lives ACROSS the batch, never along
    // the chain — the structural fact this whole project rests on.
    for (int j = 0; j < nj; ++j) {
        const float* m = &c_model[j * kModelStride];      // joint j's model row (uniform read → broadcast)

        // (1) Walk out along the link: p += R · t_j. t_j lives in the
        // PREVIOUS link's frame, so it is rotated into the base frame first.
        const float tx = m[0], ty = m[1], tz = m[2];
        p[0] = fmaf(R[0], tx, fmaf(R[1], ty, fmaf(R[2], tz, p[0])));
        p[1] = fmaf(R[3], tx, fmaf(R[4], ty, fmaf(R[5], tz, p[1])));
        p[2] = fmaf(R[6], tx, fmaf(R[7], ty, fmaf(R[8], tz, p[2])));

        // (2) Apply the link's fixed rotation: R ← R · R_fix_j.
        float Rf[9];                                      // R_fix_j from the model quaternion
        quat_to_mat3(m[3], m[4], m[5], m[6], Rf);
        float Rt[9];                                      // temporary: mat3_mul forbids aliasing
        mat3_mul(R, Rf, Rt);

        // (3) Apply the joint's motion: R ← R · Rot(axis_j, q_j).
        float Rj[9];                                      // the joint rotation about its local axis
        rodrigues(m[7], m[8], m[9], q[k * nj + j], Rj);
        mat3_mul(Rt, Rj, R);
    }

    // Convert the final orientation to the output quaternion and write the
    // message-shaped pose (layout contract in kernels.cuh). Consecutive
    // threads write 28 bytes apart — the same honest coalescing story as
    // project 33.01, equally dwarfed by arithmetic.
    float quat[4];
    mat3_to_quat(R, quat);
    float* out = &pose[k * kPoseStride];
    out[0] = p[0];    out[1] = p[1];    out[2] = p[2];
    out[3] = quat[0]; out[4] = quat[1]; out[5] = quat[2]; out[6] = quat[3];
}

// ===========================================================================
// Host-side API (declared in kernels.cuh).
// ===========================================================================

void set_robot_model(int nj, const float* model)
{
    if (nj < 1 || nj > kMaxJoints) {
        std::fprintf(stderr,
                     "set_robot_model: nj=%d outside [1, %d] — refusing (the "
                     "constant buffer is sized for kMaxJoints; see kernels.cuh)\n",
                     nj, kMaxJoints);
        std::exit(EXIT_FAILURE);
    }
    // cudaMemcpyToSymbol writes the __constant__ symbols. Symbols are
    // per-module device state — conceptually "the loaded robot", set once,
    // read by every launch after (like a stack loading its URDF at startup).
    CUDA_CHECK(cudaMemcpyToSymbol(c_model, model,
                                  static_cast<size_t>(nj) * kModelStride * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_nj, &nj, sizeof(int)));
    g_nj = nj;   // host mirror so the launcher can validate cheaply
}

void launch_batched_fk(int count, const float* d_q, float* d_pose)
{
    if (g_nj == 0) {
        // Launching FK with no robot loaded is a programming error, not a
        // runtime condition — fail loudly and immediately (no silent
        // identity-robot fallback; CLAUDE.md §13).
        std::fprintf(stderr, "launch_batched_fk: set_robot_model() was never called\n");
        std::exit(EXIT_FAILURE);
    }
    if (count <= 0) return;                        // empty batch: valid no-op

    // Repo-default geometry: 256 threads/block (warp multiple, moderate
    // register pressure), ceil-div grid; reasoning as in project 33.01.
    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;
    batched_fk_kernel<<<blocks, threads>>>(d_q, d_pose, count);
    CUDA_CHECK_LAST_ERROR("batched_fk_kernel launch");
}
