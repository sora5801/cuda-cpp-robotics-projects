// ===========================================================================
// kernels.cu — GPU kernels for project 01.17 (Camera-LiDAR / camera-camera
//              extrinsic calibration — batched reprojection-error LM)
//
// Two kernels, two GPU parallelism regimes (kernels.cuh's file header names
// both; THEORY.md "The GPU mapping" argues when each dominates):
//
//   assemble_normal_equations_kernel — CORRESPONDENCE-parallel. One thread
//     per correspondence (n=48 here — tiny, but the kernel is written for
//     ANY n, and README Exercise 1 asks what changes at n=50,000). Threads
//     within a block tree-reduce their 28-scalar contributions through
//     shared memory into one row of block_partials; main.cu finishes the
//     sum across blocks on the host, in double (02.06's ICP established
//     this exact "GPU partial reduce, host finishes it" split, cited).
//
//   multistart_lm_farm_kernel — OPTIMIZATION-parallel. One thread per
//     INDEPENDENT Levenberg-Marquardt run (K=1024). Each thread owns its
//     whole trajectory: draw an initial guess, then iterate up to
//     kMaxLmIters times, each iteration re-scanning all n correspondences
//     itself (no cross-thread communication at all — the 08.01/01.12 "farm"
//     idiom, cited). This is the natural mapping when the THING that is
//     large is the number of trials, not the number of correspondences.
//
// Both kernels call the SHARED residual_and_jacobian()/so3_exp() primitives
// from kernels.cuh (the camera model — see that header's file comment for
// why sharing those specific functions is deliberate and documented).
//
// Read this after: main.cu, kernels.cuh.  Read this before: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"

// ---------------------------------------------------------------------------
// cholesky6_solve_device — the multi-start farm's OWN, independently-written
// device-side 6x6 damped-normal-equations solve. Deliberately NOT shared
// with kernels.cuh's host-only cholesky6_solve (see that function's header):
// this project wants at least one part of the "solve" step that a GPU/CPU
// disagreement could actually expose, and the "multi-start batch subset"
// twin gate in main.cu is built exactly to exercise this function against
// the CPU's cholesky6_solve on identical H/g inputs.
//
// Same algorithm as the host version (Cholesky-Crout + forward/back
// substitution on the Marquardt-damped 6x6), transcribed independently:
// double precision throughout (see the file header "why double" note below
// this file's multistart_lm_farm_kernel doc comment) — a 6x6 double solve
// is ~50 FLOPs, utterly negligible next to the 48-correspondence residual
// scan it feeds, so there is no performance reason to drop to float here.
// ---------------------------------------------------------------------------
__device__ bool cholesky6_solve_device(const double H21[21], const double g6[6],
                                       double lambda, double out_delta[6])
{
    double A[6][6];
    for (int i = 0; i < 6; ++i)
        for (int j = i; j < 6; ++j) {
            const double hij = H21[hidx(i, j)];
            A[i][j] = hij;
            A[j][i] = hij;
        }
    for (int i = 0; i < 6; ++i) A[i][i] *= (1.0 + lambda);

    double L[6][6];
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            L[i][j] = 0.0;

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
// assemble_normal_equations_kernel
//
// Thread-to-data mapping: thread (blockIdx.x, threadIdx.x) owns
// correspondence i = blockIdx.x*blockDim.x + threadIdx.x, guarded by i<n
// (n=48 here fits in under half a block at kThreadsPerBlock=128 — the
// kernel is still launched with the FULL grid-of-blocks math below so it
// stays correct if a caller ever points it at a much larger n, e.g. the
// n=50,000 hypothetical in README Exercise 1).
//
// Memory hierarchy:
//   GLOBAL  — p_obs/uv_obs (read-only, one correspondence per thread: each
//             warp's 32 threads read CONSECUTIVE 3-float / 2-float records,
//             a coalesced access pattern) and block_partials (one 28-float
//             row written per block, by thread 0 only).
//   SHARED  — sdata[blockDim.x * kReduceWidth] floats: every thread's own
//             28-scalar contribution, laid out so thread t's record starts
//             at sdata[t*kReduceWidth] — this is what the tree reduction
//             below halves and sums. At kThreadsPerBlock=128 that is
//             128*28*4 bytes = 14336 bytes, comfortably under the 48 KiB
//             default shared-memory budget on sm_75..sm_89 (room to spare
//             for occupancy — no dynamic-vs-static tradeoff needed here).
//   REGISTERS — each thread's own local[28] accumulator (built from a
//             SINGLE correspondence, so no loop-carried register pressure)
//             before the one copy into shared memory.
//
// Why a tree reduction and not per-thread atomicAdd into a single global
// H/g/cost? Two reasons taught together: (1) atomics on 28 separate float
// slots from up to 128 threads per block would serialize heavily (28
// separate contention points), while the tree reduction is O(log
// blockDim.x) parallel steps with no contention; (2) atomicAdd on float is
// a NON-DETERMINISTIC-ORDER sum (whichever thread's atomic lands first),
// which reorders floating-point rounding run to run — the block-partial +
// host-double-sum split (02.06's convention, cited) keeps the GPU path
// bit-reproducible in its OWN block-tree order and pushes the only
// intentionally-unordered step (summing block rows) into the double
// accumulator, where the reordering sensitivity is negligible.
// ---------------------------------------------------------------------------
__global__ void assemble_normal_equations_kernel(
    const float* __restrict__ p_obs, const float* __restrict__ uv_obs, int n,
    Rigid3 T, PinholeIntrinsics K,
    float* __restrict__ block_partials)
{
    extern __shared__ float sdata[];   // blockDim.x * kReduceWidth floats (see header)

    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's correspondence index

    // Build this thread's 28-scalar contribution in REGISTERS first — a
    // single correspondence's residual/Jacobian never needs to touch shared
    // memory until the final copy below.
    float local[kReduceWidth];
    for (int k = 0; k < kReduceWidth; ++k) local[k] = 0.0f;

    if (i < n) {
        const float p_src[3]  = { p_obs[i * 3 + 0], p_obs[i * 3 + 1], p_obs[i * 3 + 2] };
        const float uv_o[2]   = { uv_obs[i * 2 + 0], uv_obs[i * 2 + 1] };
        float r[2], J[12];
        residual_and_jacobian(T, K, p_src, uv_o, r, J);   // the shared camera-model formula

        // H = J^T J, upper triangle only (hidx() packing, 21 entries) —
        // J is 2x6, so H[a][b] = J[0][a]*J[0][b] + J[1][a]*J[1][b].
        for (int a = 0; a < 6; ++a)
            for (int b = a; b < 6; ++b)
                local[hidx(a, b)] = J[a] * J[b] + J[6 + a] * J[6 + b];

        // g = J^T r (6 entries).
        for (int a = 0; a < 6; ++a)
            local[21 + a] = J[a] * r[0] + J[6 + a] * r[1];

        // cost = r^T r (1 entry) — lets main.cu read the loss from the SAME
        // reduction that produces H/g, no second kernel pass needed.
        local[27] = r[0] * r[0] + r[1] * r[1];
    }
    // Threads with i>=n contribute all-zero rows — correct no-op padding
    // for the tree reduction below (the "ragged tail" guard, same idiom as
    // the template's saxpy grid-stride tail handling).

    float* my_row = &sdata[threadIdx.x * kReduceWidth];
    for (int k = 0; k < kReduceWidth; ++k) my_row[k] = local[k];
    __syncthreads();   // every thread's row must be written before the reduction reads any of them

    // Tree reduction: at each step, the first half of the still-active
    // threads adds the second half's row into its own, halving the active
    // count. log2(blockDim.x) steps total (7 for blockDim.x=128) instead of
    // blockDim.x-1 serial adds — the standard shared-memory reduction.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float* a = &sdata[threadIdx.x * kReduceWidth];
            float* b = &sdata[(threadIdx.x + stride) * kReduceWidth];
            for (int k = 0; k < kReduceWidth; ++k) a[k] += b[k];
        }
        __syncthreads();   // every stride's adds must finish before the next stride reads them
    }

    if (threadIdx.x == 0) {
        float* out_row = &block_partials[blockIdx.x * kReduceWidth];
        for (int k = 0; k < kReduceWidth; ++k) out_row[k] = sdata[k];
    }
}

// ---------------------------------------------------------------------------
// launch_assemble_normal_equations — grid/block math + launch + error check.
//
// Block size: kThreadsPerBlock (128), a power of two (required by the tree
// reduction's stride-halving above) and a warp multiple. Grid: exactly
// blocks_for(n, kThreadsPerBlock) blocks — UNLIKE the template's capped
// grid-stride SAXPY, this kernel wants EXACTLY one thread per correspondence
// (no grid-stride loop) because each thread's reduction slot is tied to its
// block-local threadIdx.x, so launching fewer blocks than needed would
// silently drop correspondences rather than looping over them.
// Dynamic shared memory: blockDim.x * kReduceWidth * sizeof(float).
// ---------------------------------------------------------------------------
int launch_assemble_normal_equations(const float* d_p_obs, const float* d_uv_obs, int n,
                                     Rigid3 T, PinholeIntrinsics K,
                                     float* d_block_partials)
{
    const int block = kThreadsPerBlock;
    const int grid  = blocks_for(n, block);
    const size_t shmem_bytes = static_cast<size_t>(block) * kReduceWidth * sizeof(float);

    assemble_normal_equations_kernel<<<grid, block, shmem_bytes>>>(d_p_obs, d_uv_obs, n, T, K, d_block_partials);
    CUDA_CHECK_LAST_ERROR("assemble_normal_equations_kernel launch");
    return grid;
}

// ---------------------------------------------------------------------------
// multistart_lm_farm_kernel
//
// Thread-to-data mapping: thread k = blockIdx.x*blockDim.x+threadIdx.x OWNS
// optimization k (k in [0, k_starts)); its ENTIRE trajectory — initial
// guess, every LM iteration, every correspondence scan inside every
// iteration — runs serially on that one thread with no synchronization or
// communication with any other thread. This is the opposite extreme from
// the assembly kernel above: there, many threads cooperate on ONE estimate;
// here, many threads each solve their OWN estimate independently. Both are
// "the natural GPU mapping" — for different axes of the same problem.
//
// Why double precision throughout this kernel (H/g/cost/lambda/T's role in
// the solve), unlike the assembly kernel's float accumulation? The
// DEGENERACY GATE deliberately feeds this same machinery a near-rank-
// deficient correspondence set (the coplanar cohort); FP32 rounding and
// ILL-CONDITIONING would then compound in a way that is hard to tell apart
// from the geometric effect this project is trying to TEACH. Using double
// throughout the LM bookkeeping isolates conditioning as a measured,
// reportable NUMBER (the condition-number proxy) rather than an FP32
// artifact. THEORY.md "Numerical considerations" is explicit that a
// production embedded/GPU calibration pipeline would likely stay in FP32
// with more careful scaling (Kahan summation, residual pre-scaling) for
// throughput — this project trades that throughput for a cleaner lesson,
// and says so.
//
// Occupancy note: each thread's per-iteration H21/g6 (21+6 doubles = 216
// bytes) plus T/candidate-T (2*12 floats) round-trips through registers and
// (once the register file is exhausted) thread-local memory — for a
// K=1024, 128-thread-block launch this is a deliberate throughput/registers
// trade the THEORY.md "GPU mapping" section measures and discusses; the
// kernel is correctness- and teaching-focused, not tuned for maximum
// occupancy (CLAUDE.md §1: "a slower kernel a learner can follow beats a
// fast one they cannot").
// ---------------------------------------------------------------------------
__global__ void multistart_lm_farm_kernel(
    const float* __restrict__ p_obs, const float* __restrict__ uv_obs, int n,
    PinholeIntrinsics K, Rigid3 T_seed,
    float max_rot_perturb_rad, float max_trans_perturb_m,
    uint32_t base_seed, int k_starts, int max_iters,
    Rigid3* __restrict__ out_T, double* __restrict__ out_loss,
    float* __restrict__ out_init_rot, float* __restrict__ out_init_trans)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= k_starts) return;

    // Per-thread RNG stream: multiply-mix the thread index into the base
    // seed (the same "odd multiplier for full-period stream separation"
    // reasoning 08.01 uses per control-step, applied here per THREAD) and
    // guard against the all-zero xorshift32 state (which would stay zero
    // forever — xorshift32's one degenerate fixed point).
    uint32_t seed = base_seed ^ (0x9E3779B9u * static_cast<uint32_t>(k + 1));
    if (seed == 0u) seed = 1u;

    // ---- draw this thread's randomized initial guess ----------------------
    // Random rotation AXIS: three Gaussian draws normalized to a unit
    // vector is the standard way to sample uniformly on a sphere (the
    // Gaussian's rotational symmetry means normalizing removes all
    // magnitude information, leaving only a uniform direction).
    float ax[3] = { gaussian(seed, 1.0f), gaussian(seed, 1.0f), gaussian(seed, 1.0f) };
    float axn = sqrtf(ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2]);
    if (axn < 1.0e-6f) axn = 1.0e-6f;   // guard: (near-)zero draw is astronomically unlikely but not impossible
    const float rot_mag = uniform01(seed) * max_rot_perturb_rad;   // magnitude uniform in [0, max]
    const float omega0[3] = { ax[0] / axn * rot_mag, ax[1] / axn * rot_mag, ax[2] / axn * rot_mag };

    float tx[3] = { gaussian(seed, 1.0f), gaussian(seed, 1.0f), gaussian(seed, 1.0f) };
    float txn = sqrtf(tx[0] * tx[0] + tx[1] * tx[1] + tx[2] * tx[2]);
    if (txn < 1.0e-6f) txn = 1.0e-6f;
    const float trans_mag = uniform01(seed) * max_trans_perturb_m;
    const float v0[3] = { tx[0] / txn * trans_mag, tx[1] / txn * trans_mag, tx[2] / txn * trans_mag };

    out_init_rot[k]   = rot_mag;
    out_init_trans[k] = trans_mag;

    Rigid3 T;
    float dR[9];
    so3_exp(omega0, dR);
    mat3_mul(dR, T_seed.R, T.R);
    T.t[0] = T_seed.t[0] + v0[0];
    T.t[1] = T_seed.t[1] + v0[1];
    T.t[2] = T_seed.t[2] + v0[2];

    // ---- this thread's own complete LM trajectory --------------------------
    double lambda = kLambdaInit;
    double cost   = 0.0;
    bool have_cost = false;

    for (int it = 0; it < max_iters; ++it) {
        // Assemble H21/g6/cost at the CURRENT T by scanning every
        // correspondence — a serial loop (this thread's own work, no
        // cooperation with any other thread; contrast with the assembly
        // kernel's parallel-then-reduce version of this same sum).
        double H21[21];
        double g6[6];
        double cost_cur = 0.0;
        for (int a = 0; a < 21; ++a) H21[a] = 0.0;
        for (int a = 0; a < 6; ++a) g6[a] = 0.0;

        for (int i = 0; i < n; ++i) {
            const float p_src[3] = { p_obs[i * 3 + 0], p_obs[i * 3 + 1], p_obs[i * 3 + 2] };
            const float uv_o[2]  = { uv_obs[i * 2 + 0], uv_obs[i * 2 + 1] };
            float r[2], J[12];
            residual_and_jacobian(T, K, p_src, uv_o, r, J);
            for (int a = 0; a < 6; ++a)
                for (int b = a; b < 6; ++b)
                    H21[hidx(a, b)] += static_cast<double>(J[a] * J[b] + J[6 + a] * J[6 + b]);
            for (int a = 0; a < 6; ++a)
                g6[a] += static_cast<double>(J[a] * r[0] + J[6 + a] * r[1]);
            cost_cur += static_cast<double>(r[0] * r[0] + r[1] * r[1]);
        }
        if (!have_cost) { cost = cost_cur; have_cost = true; }

        // Solve, backing lambda off (up to 5 tries) if the damped system is
        // not SPD — rare (H is a sum of outer products, PSD by
        // construction; damping with lambda>0 makes it SPD unless a pivot
        // underflows), but cheap to guard.
        double delta[6];
        bool ok = false;
        for (int attempt = 0; attempt < 5 && !ok; ++attempt) {
            ok = cholesky6_solve_device(H21, g6, lambda, delta);
            if (!ok) lambda *= kLambdaUp;
        }
        if (!ok) break;   // numerically stuck — this start is reported as non-converged

        Rigid3 T_new;
        retract(T, delta, T_new);

        // Cost at the CANDIDATE step (residual only; computing the full
        // Jacobian too is wasted work here, but reusing one formula for
        // both keeps the code simple and the cost is negligible at n=48 —
        // CLAUDE.md §1: teaching clarity over micro-optimization).
        double cost_new = 0.0;
        for (int i = 0; i < n; ++i) {
            const float p_src[3] = { p_obs[i * 3 + 0], p_obs[i * 3 + 1], p_obs[i * 3 + 2] };
            const float uv_o[2]  = { uv_obs[i * 2 + 0], uv_obs[i * 2 + 1] };
            float r[2], J[12];
            residual_and_jacobian(T_new, K, p_src, uv_o, r, J);
            cost_new += static_cast<double>(r[0] * r[0] + r[1] * r[1]);
        }

        const double delta_norm = sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2] +
                                       delta[3] * delta[3] + delta[4] * delta[4] + delta[5] * delta[5]);

        if (cost_new < cost) {
            // Accept: press forward (shrink damping toward Gauss-Newton).
            const double rel_change = fabs(cost - cost_new) / (cost_new + 1.0e-12);
            T = T_new;
            cost = cost_new;
            lambda *= kLambdaDown;
            if (lambda < kLambdaMin) lambda = kLambdaMin;
            if (delta_norm < kConvergeDeltaNorm || rel_change < kConvergeCostRel) break;
        } else {
            // Reject: back off (grow damping toward gradient descent) and
            // retry from the SAME T next iteration — T is left unchanged.
            lambda *= kLambdaUp;
        }
    }

    out_T[k]    = T;
    out_loss[k] = cost;
}

// ---------------------------------------------------------------------------
// launch_multistart_farm — grid/block math + launch + error check.
// Block size 128 (a warp multiple; no reduction here so no power-of-two
// requirement, but 128 keeps register pressure per SM reasonable given each
// thread's sizeable local state — see the kernel's occupancy note above).
// ---------------------------------------------------------------------------
void launch_multistart_farm(const float* d_p_obs, const float* d_uv_obs, int n,
                            PinholeIntrinsics K, Rigid3 T_seed,
                            float max_rot_perturb_rad, float max_trans_perturb_m,
                            uint32_t base_seed, int k_starts, int max_iters,
                            Rigid3* d_out_T, double* d_out_loss,
                            float* d_out_init_rot, float* d_out_init_trans)
{
    const int block = 128;
    const int grid  = blocks_for(k_starts, block);

    multistart_lm_farm_kernel<<<grid, block>>>(d_p_obs, d_uv_obs, n, K, T_seed,
                                               max_rot_perturb_rad, max_trans_perturb_m,
                                               base_seed, k_starts, max_iters,
                                               d_out_T, d_out_loss, d_out_init_rot, d_out_init_trans);
    CUDA_CHECK_LAST_ERROR("multistart_lm_farm_kernel launch");
}
