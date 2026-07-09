// ===========================================================================
// kernels.cu — GPU implementation for project 33.01
//              Batched small-matrix linear algebra (3×3, 4×4, 6×6 —
//              the robotics sizes)
//
// The big idea
// ------------
// Robotics almost never needs ONE big matrix factorized fast; it needs a
// MILLION tiny ones factorized simultaneously: a mass matrix per sampled
// configuration, a covariance per particle, a normal-equation solve per
// IK seed. At sizes 3/4/6 a single matrix is far too small to parallelize
// INTERNALLY (a 6×6 Cholesky is ~100 flops — less than one warp's worth),
// so the GPU mapping flips: parallelize ACROSS the batch instead.
//
//     one thread  =  one whole matrix problem, solved in registers.
//
// That is the pattern this whole file teaches, and it is the same pattern
// the repository reuses for FK (09.01), rollouts (08.01), particles (04.01):
// small independent problems → thread-per-problem → data in registers.
//
// Memory-hierarchy reasoning (THEORY.md §GPU-mapping expands this)
// ----------------------------------------------------------------
// A thread's matrix lives in a C array indexed only by compile-time-known
// indices after full unrolling — so the compiler places it in REGISTERS,
// the fastest storage on the chip (~0 cycles, private per thread). Sizes:
// N=6 needs 36 floats for L plus 6+6 for vectors ≈ 48 registers — heavy but
// under the 255/thread hardware cap; occupancy drops, yet each thread does
// so much arithmetic per byte loaded that the SMs stay busy anyway. This
// trade (fewer, fatter threads) is discussed honestly in THEORY.md.
//
// The load/store pattern is the honest weakness of the matrix-contiguous
// layout (see kernels.cuh): consecutive THREADS read words N*N apart, so a
// warp's 32 loads land in different 128-byte segments — poorly coalesced.
// We keep the layout anyway because (a) it is the natural way robotics code
// stores matrices, (b) the kernels are arithmetic-heavy enough at N=6 to
// hide much of it, and (c) fixing it (SoA / staging through shared memory,
// as cuBLAS gemmBatched does internally) is exactly Exercise 3 in README.md.
// Teaching beats peak throughput here (CLAUDE.md §1).
//
// Why not cuBLAS? cublasSgemmBatched / cusolverDn<t>potrsBatched do exactly
// this, faster, for large batches. We hand-roll because this project IS the
// explanation of what those calls do inside (no black boxes, CLAUDE.md §1);
// README.md §Prior-art points at them for production use.
//
// Read this after: kernels.cuh.  Companion oracle: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"              // the interface this file implements
#include "util/cuda_check.cuh"      // CUDA_CHECK_LAST_ERROR at every launch (§6.1 rule 7)

#include <cstdio>                   // fprintf for the invalid-n abort message
#include <cstdlib>                  // std::exit
#include <math_constants.h>         // CUDART_NAN_F: the canonical device NaN constant

// ---------------------------------------------------------------------------
// Shared launch geometry: one thread per matrix problem.
//
// 256 threads/block is the repository default: a multiple of the 32-thread
// warp, small enough to keep per-block register demand reasonable (256
// threads × ~48 registers ≈ 12k of an SM's 64k registers), large enough to
// give the scheduler warps to hide memory latency with. Nothing below
// depends on the exact value; it is a sane default, not a tuned one —
// Nsight-based tuning is Exercise 5 in README.md.
// ---------------------------------------------------------------------------
static constexpr int kThreadsPerBlock = 256;

// Integer ceiling division: how many blocks cover `count` problems at one
// thread each. Named as a function so the intent reads at the call sites.
// Example: count=1000 → (1000+255)/256 = 4 blocks, the last one ragged with
// 24 idle-exit threads — every kernel below guards `k >= count` for exactly
// that tail.
static inline int blocks_for(int count)
{
    return (count + kThreadsPerBlock - 1) / kThreadsPerBlock;
}

// ===========================================================================
// Kernel 1: batched matrix multiply  C_k = A_k · B_k
// ===========================================================================
// Template parameter N (3, 4, or 6) is COMPILE-TIME so that:
//   * a[N*N], b[N*N] are register arrays — after full unrolling every index
//     is a literal, which is what allows register placement at all (register
//     files are not runtime-indexable; a dynamic index forces a spill to
//     slow "local" memory, which physically lives in global DRAM);
//   * the triple loop unrolls into straight-line FMA code (N=3: 27 FMAs,
//     N=6: 216 FMAs per thread, zero loop overhead).
// The host launcher below maps the caller's runtime n onto these
// instantiations with an exhaustive switch.
//
// Thread-to-data mapping: thread with global index
//     k = blockIdx.x * blockDim.x + threadIdx.x
// owns matrix pair k. Grid: blocks_for(count) blocks × kThreadsPerBlock.
//
// Memory spaces: each thread reads its A_k and B_k from GLOBAL memory into
// REGISTERS once (N*N loads each) and writes C_k once — the minimum traffic
// per problem. There is NO data reuse across threads (k's matrices are k's
// alone), which is why SHARED memory earns no keep here and is absent.
// Contrast with large-matrix GEMM, where every element is reused across
// many threads and shared-memory tiling is the whole game — that contrast
// is half the point of this project (THEORY.md §GPU-mapping).
// ===========================================================================
template <int N>
__global__ void batched_matmul_kernel(const float* __restrict__ A,   // [count*N*N] left factors (layout: kernels.cuh)
                                      const float* __restrict__ B,   // [count*N*N] right factors
                                      float*       __restrict__ C,   // [count*N*N] OUT: products; must not alias A/B
                                      int count)                     // number of independent pairs
{
    // This thread's problem index; the guard retires the over-provisioned
    // threads of the ragged last block before they touch memory.
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= count) return;

    // __restrict__ above is our promise that A/B/C never alias — it lets the
    // compiler keep a[]/b[] live in registers across the computation instead
    // of conservatively re-loading after every store to C.

    // Base offset of matrix k (layout rule from kernels.cuh: matrix k is the
    // N*N-float slab starting at k*N*N, row-major inside the slab).
    const int base = k * N * N;

    float a[N * N];   // this thread's private copy of A_k, in registers
    float b[N * N];   // this thread's private copy of B_k
#pragma unroll
    for (int i = 0; i < N * N; ++i) a[i] = A[base + i];
#pragma unroll
    for (int i = 0; i < N * N; ++i) b[i] = B[base + i];

    // Classic row·column product, fully unrolled. fmaf(x, y, acc) makes the
    // fused multiply-add EXPLICIT: one rounding per term instead of two.
    // The CPU oracle may or may not contract to FMA (host compiler's choice)
    // — which is precisely why main.cu compares against a documented
    // TOLERANCE, never bit-for-bit (THEORY.md §numerics tells this story).
#pragma unroll
    for (int i = 0; i < N; ++i) {          // row of C
#pragma unroll
        for (int j = 0; j < N; ++j) {      // column of C
            float acc = 0.0f;              // c(i,j) accumulator, FP32 like the data
#pragma unroll
            for (int p = 0; p < N; ++p)    // contraction index: row i of A · column j of B
                acc = fmaf(a[i * N + p], b[p * N + j], acc);
            C[base + i * N + j] = acc;     // exactly one global store per output element
        }
    }
}

// ===========================================================================
// Kernel 2: batched Cholesky solve  A_k x_k = b_k  for SPD A_k
// ===========================================================================
// Why Cholesky and not general LU? For symmetric positive definite (SPD)
// matrices Cholesky is the standard answer: half the flops of LU (~N³/3),
// no pivoting needed (SPD guarantees positive pivots in exact arithmetic),
// and the positivity test doubles as the SPD-ness diagnostic. Robotics
// leans on it constantly: joint-space mass matrices M(q), damped normal
// equations JᵀJ + λI in Levenberg–Marquardt IK (project 09.05), covariance
// manipulation in estimators (domain 04).
//
// Per-thread algorithm (all in registers, ~N³/3 + 2N² flops):
//   1. factorize   A = L·Lᵀ     (lower-triangular L, positive diagonal)
//   2. forward     L·y = b      (solve for y — done in place in v[])
//   3. backward    Lᵀ·x = y     (solve for x — same v[])
//
// Thread mapping / grid / ragged-tail guard: identical to the matmul kernel;
// see the shared-geometry comment at the top of the file.
//
// Numerical honesty (THEORY.md §numerics): FP32 Cholesky is fine for the
// well-conditioned systems this demo generates (the generator adds N·I to a
// Gram matrix, bounding the condition number to single digits). Near
// kinematic singularities real mass/normal matrices become ill-conditioned
// and production code promotes to FP64 or regularizes — provoking that
// failure on purpose is Exercise 4 in README.md.
// ===========================================================================
template <int N>
__global__ void batched_cholesky_solve_kernel(const float* __restrict__ A,  // [count*N*N] SPD matrices; lower triangle read
                                              const float* __restrict__ b,  // [count*N] right-hand sides
                                              float*       __restrict__ x,  // [count*N] OUT: solutions (NaN-filled if not SPD)
                                              int count)                    // number of independent systems
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's system index
    if (k >= count) return;                         // ragged-tail guard, as always

    // Stage the lower triangle of A_k into a register array that is then
    // overwritten IN PLACE by L (standard trick: position (i,j) of A is
    // never needed again once l_ij is computed). The strict upper triangle
    // of the source is never read — interface note in kernels.cuh — and the
    // unused upper slots are zeroed so no lane computes on garbage.
    float m[N * N];   // A_k's lower triangle on entry; L on exit
#pragma unroll
    for (int i = 0; i < N; ++i)
#pragma unroll
        for (int j = 0; j < N; ++j)
            m[i * N + j] = (j <= i) ? A[k * N * N + i * N + j] : 0.0f;

    float v[N];       // b_k on entry → y after forward-sub → x after back-sub
#pragma unroll
    for (int i = 0; i < N; ++i) v[i] = b[k * N + i];

    // --- 1) Factorize: column-by-column classical Cholesky -----------------
    // Loop invariant entering column j: columns 0..j-1 of m already hold L.
    bool spd = true;  // remains true iff every pivot was strictly positive
#pragma unroll
    for (int j = 0; j < N; ++j) {
        // Diagonal: l_jj = sqrt( a_jj − Σ_{p<j} l_jp² ).
        float diag = m[j * N + j];
#pragma unroll
        for (int p = 0; p < j; ++p)
            diag = fmaf(-m[j * N + p], m[j * N + p], diag);   // subtract l_jp²

        // The SPD test IS this positivity test. On failure we do NOT return
        // early: all warp lanes keep marching through the same instructions
        // (the math below is harmless garbage for this k) and the NaN policy
        // at the end makes the failure unmissable. Early-exit would create
        // divergence for zero benefit at these tiny sizes.
        spd = spd && (diag > 0.0f);
        float l_jj = sqrtf(fmaxf(diag, 0.0f));  // fmaxf: keep sqrtf's argument legal even when !spd
        m[j * N + j] = l_jj;
        // One division, reused for the whole column below the diagonal —
        // divisions are ~10× the cost of multiplies; hoisting them matters.
        float inv_l_jj = (l_jj > 0.0f) ? (1.0f / l_jj) : 0.0f;

        // Below-diagonal column: l_ij = ( a_ij − Σ_{p<j} l_ip·l_jp ) / l_jj.
#pragma unroll
        for (int i = j + 1; i < N; ++i) {
            float s = m[i * N + j];
#pragma unroll
            for (int p = 0; p < j; ++p)
                s = fmaf(-m[i * N + p], m[j * N + p], s);
            m[i * N + j] = s * inv_l_jj;
        }
    }

    // --- 2) Forward substitution: L·y = b (v becomes y) ---------------------
    // Row i of L·y = b  ⇒  y_i = ( b_i − Σ_{p<i} l_ip·y_p ) / l_ii.
#pragma unroll
    for (int i = 0; i < N; ++i) {
        float s = v[i];
#pragma unroll
        for (int p = 0; p < i; ++p)
            s = fmaf(-m[i * N + p], v[p], s);
        v[i] = s / m[i * N + i];   // /0 only possible when !spd; result discarded below
    }

    // --- 3) Back substitution: Lᵀ·x = y (v becomes x) ------------------------
    // Lᵀ(i,p) is L(p,i) — hence the transposed index m[p*N + i].
#pragma unroll
    for (int i = N - 1; i >= 0; --i) {
        float s = v[i];
#pragma unroll
        for (int p = i + 1; p < N; ++p)
            s = fmaf(-m[p * N + i], v[p], s);
        v[i] = s / m[i * N + i];
    }

    // --- Write back, applying the documented non-SPD policy -----------------
    // NaN, deliberately: NaN propagates through any downstream arithmetic,
    // so a consumer CANNOT mistake a failed solve for a real answer (zeros
    // would look plausible). reference_cpu.cpp implements the identical
    // policy, keeping the GPU-vs-CPU comparison meaningful for bad inputs.
#pragma unroll
    for (int i = 0; i < N; ++i)
        x[k * N + i] = spd ? v[i] : CUDART_NAN_F;
}

// ===========================================================================
// Host-side launchers: runtime n → compile-time N dispatch + error checking.
// ===========================================================================
// The switch is deliberately EXHAUSTIVE-or-abort: silently routing an
// unexpected n to some slow generic path would betray the repository's
// no-silent-fallbacks stance (CLAUDE.md §13). The three cases are the only
// template instantiations that exist in the binary — an unsupported size
// cannot even be linked, let alone limp along.
// ---------------------------------------------------------------------------

// Loud, grep-able abort for an unsupported dimension: the learner who hits
// this finds the message text right here next to the reasoning.
static void abort_bad_dimension(const char* fn, int n)
{
    std::fprintf(stderr,
                 "%s: unsupported matrix size n=%d (this library instantiates "
                 "n = 3, 4, 6 only — see the dispatch comment in kernels.cu)\n",
                 fn, n);
    std::exit(EXIT_FAILURE);
}

void launch_batched_matmul(int n, int count,
                           const float* d_A, const float* d_B, float* d_C)
{
    if (count <= 0) return;              // an empty batch is a valid no-op, not an error
    const int blocks = blocks_for(count);

    // <<<...>>> enqueues asynchronously; CUDA_CHECK_LAST_ERROR immediately
    // after each launch catches launch-time failures (bad geometry, missing
    // sm_XX image) at the exact site. Faults INSIDE the kernel surface at
    // the caller's next synchronizing call — util/cuda_check.cuh tells that
    // whole story once, for the entire project.
    switch (n) {
    case 3:
        batched_matmul_kernel<3><<<blocks, kThreadsPerBlock>>>(d_A, d_B, d_C, count);
        CUDA_CHECK_LAST_ERROR("batched_matmul_kernel<3> launch");
        break;
    case 4:
        batched_matmul_kernel<4><<<blocks, kThreadsPerBlock>>>(d_A, d_B, d_C, count);
        CUDA_CHECK_LAST_ERROR("batched_matmul_kernel<4> launch");
        break;
    case 6:
        batched_matmul_kernel<6><<<blocks, kThreadsPerBlock>>>(d_A, d_B, d_C, count);
        CUDA_CHECK_LAST_ERROR("batched_matmul_kernel<6> launch");
        break;
    default:
        abort_bad_dimension("launch_batched_matmul", n);
    }
}

void launch_batched_cholesky_solve(int n, int count,
                                   const float* d_A, const float* d_b,
                                   float* d_x)
{
    if (count <= 0) return;              // empty batch: valid no-op
    const int blocks = blocks_for(count);

    switch (n) {
    case 3:
        batched_cholesky_solve_kernel<3><<<blocks, kThreadsPerBlock>>>(d_A, d_b, d_x, count);
        CUDA_CHECK_LAST_ERROR("batched_cholesky_solve_kernel<3> launch");
        break;
    case 4:
        batched_cholesky_solve_kernel<4><<<blocks, kThreadsPerBlock>>>(d_A, d_b, d_x, count);
        CUDA_CHECK_LAST_ERROR("batched_cholesky_solve_kernel<4> launch");
        break;
    case 6:
        batched_cholesky_solve_kernel<6><<<blocks, kThreadsPerBlock>>>(d_A, d_b, d_x, count);
        CUDA_CHECK_LAST_ERROR("batched_cholesky_solve_kernel<6> launch");
        break;
    default:
        abort_bad_dimension("launch_batched_cholesky_solve", n);
    }
}
