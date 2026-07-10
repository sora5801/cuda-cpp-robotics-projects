// ===========================================================================
// kernels.cu — GPU implementation for project 16.01
//              Thruster allocation for overactuated ROVs (batched QP)
//
// The big idea
// ------------
// A single 8-unknown, 6-equation box-constrained QP is FAR too small to
// parallelize internally (500 projected-gradient iterations x ~80 flops is
// nothing for one SM) — exactly the 33.01 lesson. A real ROV control loop
// does not need to solve ONE allocation problem, though: a whole planned
// trajectory's wrench sequence needs allocating in one shot, and a
// fault-tolerance sweep needs the SAME batch re-solved once per candidate
// thruster failure (main.cu's two demo stages). Both are thousands of
// INDEPENDENT small QPs — so, exactly like 33.01's batched Cholesky solve
// and 08.01's MPPI rollouts, the GPU mapping flips:
//
//     one thread  =  one whole QP, solved in registers.
//
// What is NEW here beyond 33.01/08.01:
//   * the per-thread "problem" is not a fixed linear system but an ITERATIVE
//     first-order optimizer (projected gradient descent) — the loop body is
//     a dense 8x8 matrix-vector product plus a box PROJECTION (a branchless
//     clamp), repeated kPgdIters times;
//   * the 8x8 Hessian H and 8x6 matrix BtW2 are IDENTICAL for every thread
//     in the batch (they depend only on the vehicle's fixed geometry, never
//     on the commanded wrench) — the textbook case for CUDA __constant__
//     memory: every thread reads the SAME address every iteration, which
//     the constant cache serves at close to register speed after the first
//     touch (broadcast to the whole warp in one transaction). This sits at
//     the "always-uniform" end of the same read-pattern spectrum 08.01
//     discusses for u_nom (uniform, L2-served) and 09.01 for its rotation
//     table (__constant__, this project's exact pattern);
//   * per-problem, PER-THRUSTER box limits (d_umax) — not a single global
//     bound — because the fault-tolerance sweep needs to lock individual
//     thrusters to 0 without touching this kernel at all (main.cu Stage 4).
//
// Why not cuSOLVER / a QP library? cusolverDn has no box-constrained QP
// primitive; production stacks reach for OSQP, qpOASES, or a custom active-
// set/interior-point solver (README "Prior art"). We hand-roll projected
// gradient descent because it IS the simplest solver whose behavior a
// learner can predict by hand (THEORY.md "the algorithm"), and because the
// PROJECTION step — a per-component clamp — is the one piece of QP theory
// this project exists to teach (CLAUDE.md §1: no black boxes).
//
// Read this after: kernels.cuh.  Companion oracle: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"      // CUDA_CHECK / CUDA_CHECK_LAST_ERROR (§6.1 rule 7)

#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// __constant__ memory: the vehicle-geometry-derived matrices EVERY thread in
// EVERY launch reads identically. 64 + 48 = 112 floats = 448 bytes — a tiny
// fraction of the 64 KiB constant-memory budget, cached on-chip once warmed.
//
//   c_H    : [kNThr*kNThr] = 2*(B^T W^2 B + eps*I), row-major — the QP's
//            quadratic term; grad J(u) = c_H*u - g (see the kernel below).
//   c_BtW2 : [kNThr*kNDof] = B^T W^2, row-major — turns a commanded wrench
//            into the QP's linear term: g = 2 * c_BtW2 * tau.
//
// Filled ONCE per program run by upload_allocation_constants (below), called
// from main.cu's one-time setup stage — never inside the per-tick/per-batch
// hot path, exactly like 08.01's persistent device buffers.
// ---------------------------------------------------------------------------
__constant__ float c_H[kNThr * kNThr];
__constant__ float c_BtW2[kNThr * kNDof];

void upload_allocation_constants(const float* H, const float* BtW2)
{
    // cudaMemcpyToSymbol resolves the __constant__ SYMBOL (not a device
    // pointer — there is no cudaMalloc for __constant__ memory; the linker
    // reserves it statically) and copies host bytes into it. This is the
    // idiomatic way to fill constant memory once at startup.
    CUDA_CHECK(cudaMemcpyToSymbol(c_H, H, sizeof(float) * kNThr * kNThr));
    CUDA_CHECK(cudaMemcpyToSymbol(c_BtW2, BtW2, sizeof(float) * kNThr * kNDof));
}

// ---------------------------------------------------------------------------
// Shared launch geometry: one thread per QP (the 33.01/08.01 default).
// ---------------------------------------------------------------------------
static constexpr int kThreadsPerBlock = 256;

static inline int blocks_for(int count)
{
    return (count + kThreadsPerBlock - 1) / kThreadsPerBlock;
}

// ===========================================================================
// pgd_allocate_kernel — solve one box-constrained QP per thread via a FIXED
// number of projected-gradient-descent steps.
//
// The math (THEORY.md "the algorithm" derives every line):
//   J(u)     = ||W(Bu - tau)||^2 + eps||u||^2
//            = 0.5 u^T c_H u - g^T u + const,   g = 2*c_BtW2*tau
//   grad J   = c_H*u - g
//   update   = clip( u - step*grad J(u),  -u_max,  +u_max )   (step = 1/L,
//              L = lambda_max(c_H), set once on the host — THEORY.md "the
//              math" derives why 1/L guarantees monotone descent: the
//              classical projected-gradient / ISTA convergence result for
//              an L-smooth convex objective, Beck & Teboulle 2009).
//
// Thread-to-data mapping: thread k = blockIdx.x*blockDim.x + threadIdx.x
// owns problem k. Grid: blocks_for(count) x kThreadsPerBlock (ragged tail
// guarded, as everywhere in this repo).
//
// Memory spaces per thread:
//   registers : tau[6], umax[8], g[8], u[8], grad[8] (~30 registers) — the
//               WHOLE optimization state lives here for all kPgdIters steps;
//   constant  : c_H (64 floats), c_BtW2 (48 floats) — UNIFORM reads (every
//               thread, every iteration, same address) — broadcast/cached,
//               the cheapest possible global-scope read pattern;
//   global    : tau[k], umax[k] read ONCE at the top (coalesced: consecutive
//               threads read consecutive problems' consecutive floats);
//               u_out[k] written ONCE at the bottom.
// No shared memory (problems share nothing beyond the read-only constants),
// no atomics, no divergence beyond the tail guard: kPgdIters is FIXED and
// identical for every thread, so — unlike an early-exit convergence check —
// every lane in a warp executes the SAME instruction stream for the SAME
// number of iterations (THEORY.md "GPU mapping" discusses the trade against
// early exit explicitly).
// ===========================================================================
__global__ void pgd_allocate_kernel(
    const float* __restrict__ tau,    // [count*kNDof] commanded wrenches (N, N, N, N*m, N*m, N*m)
    const float* __restrict__ umax,   // [count*kNThr] per-problem, per-thruster saturation limits (N)
    float*       __restrict__ u_out,  // [count*kNThr] OUT: solved thruster forces (N)
    int count,                        // number of independent QPs in this batch
    float step)                       // projected-gradient step size, 1/L (computed on host)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's problem index
    if (k >= count) return;                               // ragged-tail guard

    // ---- Load this problem's inputs into registers (coalesced reads: warp
    // lane t reads tau[(k)*6 + i] for fixed i across t -> consecutive
    // addresses across the warp, one 128-byte-class transaction per i). ----
    float t[kNDof];
#pragma unroll
    for (int i = 0; i < kNDof; ++i) t[i] = tau[k * kNDof + i];

    float um[kNThr];
#pragma unroll
    for (int i = 0; i < kNThr; ++i) um[i] = umax[k * kNThr + i];

    // ---- Linear term g = 2 * BtW2 * t (an 8x6 matvec against the UNIFORM
    // constant-memory matrix c_BtW2; every thread computes its OWN g from
    // its OWN t, but every thread reads the SAME c_BtW2 entries to do it). --
    float g[kNThr];
#pragma unroll
    for (int i = 0; i < kNThr; ++i) {
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < kNDof; ++j)
            acc = fmaf(c_BtW2[i * kNDof + j], t[j], acc);
        g[i] = 2.0f * acc;
    }

    // ---- Projected gradient descent, cold-started at zero thrust. Zero is
    // a defensible, DOCUMENTED start (README "Limitations"): it is not the
    // previous tick's solution (a real controller would warm-start from
    // there — README Exercise — for faster practical convergence), but it
    // keeps this kernel STATELESS and each call reproducible in isolation,
    // which matters for the GPU-vs-CPU verification gate (both paths must
    // start from the identical, documented point). ---------------------
    float u[kNThr];
#pragma unroll
    for (int i = 0; i < kNThr; ++i) u[i] = 0.0f;

    for (int it = 0; it < kPgdIters; ++it) {
        // grad = c_H * u - g   (8x8 matvec; c_H is UNIFORM constant memory,
        // u is this thread's PRIVATE register state — the classic "shared
        // read-only matrix x private vector" pattern).
        float grad[kNThr];
#pragma unroll
        for (int i = 0; i < kNThr; ++i) {
            float acc = 0.0f;
#pragma unroll
            for (int j = 0; j < kNThr; ++j)
                acc = fmaf(c_H[i * kNThr + j], u[j], acc);
            grad[i] = acc - g[i];
        }
        // Gradient step + PROJECTION onto the box [-um_i, +um_i] — the
        // entire "QP theory" this project teaches collapses to this one
        // clamp, because a BOX is the one constraint set whose Euclidean
        // projection is trivial (component-wise clip, no linear system to
        // solve — THEORY.md "the algorithm" contrasts this with a general
        // polytope, where projection itself would be another QP).
#pragma unroll
        for (int i = 0; i < kNThr; ++i) {
            const float cand = u[i] - step * grad[i];
            u[i] = fminf(fmaxf(cand, -um[i]), um[i]);
        }
    }

    // One coalesced write per output element; the host's optimality gates
    // and the demo's CSV artifacts consume u_out next (main.cu).
#pragma unroll
    for (int i = 0; i < kNThr; ++i) u_out[k * kNThr + i] = u[i];
}

// ===========================================================================
// Host launcher (declared in kernels.cuh).
// ===========================================================================
void launch_thruster_allocation(int count,
                                const float* d_tau, const float* d_umax,
                                float* d_u_out, float step)
{
    if (count <= 0) return;   // an empty batch is a valid no-op, not an error

    const int blocks = blocks_for(count);
    pgd_allocate_kernel<<<blocks, kThreadsPerBlock>>>(d_tau, d_umax, d_u_out, count, step);
    CUDA_CHECK_LAST_ERROR("pgd_allocate_kernel launch");
}
