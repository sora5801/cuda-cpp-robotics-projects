// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 33.01
//                     Batched small-matrix linear algebra (3×3, 4×4, 6×6 —
//                     the robotics sizes)
//
// WHY does a GPU repository ship a CPU implementation of everything?
// ------------------------------------------------------------------
// Two load-bearing reasons (CLAUDE.md §5):
//
//   1) It is the CORRECTNESS ORACLE. GPU code fails in ways CPU code cannot:
//      wrong thread indexing, missed tail elements, race conditions, stale
//      device memory, bad transfers. A dead-simple sequential version that a
//      reader can verify by eye is the ground truth main.cu checks against.
//
//   2) It is the TEACHING BASELINE. Reading this file first, then
//      kernels.cu, shows exactly what the GPU version adds (thread mapping,
//      register staging, launch plumbing) and what stays identical (the
//      math — compare the loop bodies line by line; they match on purpose).
//
// Style contract for this file: the SIMPLEST correct C++ — no SIMD, no
// OpenMP, no cleverness. Slow is fine; the demo times it honestly and the
// speed-up line is a teaching artifact, not a benchmark (CLAUDE.md §12).
//
// The layout rule (matrix k's element (i,j) at ptr[k*n*n + i*n + j]) and the
// NaN-on-non-SPD policy are defined ONCE in kernels.cuh — this file
// implements the same contract for host memory, or the GPU-vs-CPU
// comparison in main.cu would be comparing different problems.
//
// Read this after: kernels.cuh.  Read this before (or beside): kernels.cu.
// ===========================================================================

#include "kernels.cuh"   // shared interface: signatures + layout/NaN contract

#include <cmath>         // std::sqrt for the Cholesky diagonal
#include <limits>        // std::numeric_limits<float>::quiet_NaN()

// ---------------------------------------------------------------------------
// batched_matmul_cpu — sequential  C_k = A_k · B_k  for k = 0..count-1.
//
// Mirrors launch_batched_matmul (kernels.cuh documents the shared contract).
// Unlike the GPU path, n is a plain runtime int here — the CPU has no
// register-allocation reason to fix it at compile time, and accepting any
// n >= 1 keeps the oracle usable for experiments beyond 3/4/6 (the GPU side
// is the deliberately restricted one).
//
// Cost: count · n³ multiply-adds on ONE core, one problem after another —
// exactly the serial baseline the batch pattern exists to beat.
// ---------------------------------------------------------------------------
void batched_matmul_cpu(int n, int count,
                        const float* A, const float* B, float* C)
{
    for (int k = 0; k < count; ++k) {          // each matrix pair in turn (the GPU gives each its own thread)
        const float* a = A + static_cast<size_t>(k) * n * n;  // matrix k of A (layout rule: kernels.cuh)
        const float* b = B + static_cast<size_t>(k) * n * n;  // matrix k of B
        float*       c = C + static_cast<size_t>(k) * n * n;  // matrix k of C (output)

        // Textbook triple loop — the same i/j/p roles as the unrolled GPU
        // version, in the same order, so the two files read side by side.
        for (int i = 0; i < n; ++i) {          // row of C
            for (int j = 0; j < n; ++j) {      // column of C
                float acc = 0.0f;              // c(i,j) accumulator
                for (int p = 0; p < n; ++p)    // row i of A · column j of B
                    acc += a[i * n + p] * b[p * n + j];
                c[i * n + j] = acc;
            }
        }
    }
    // Note on floating point: the GPU accumulates with explicit fmaf (one
    // rounding per term); this loop leaves contraction to the host compiler.
    // Same mathematical sum, possibly different rounding — which is WHY the
    // comparison in main.cu uses a documented tolerance, never equality.
}

// ---------------------------------------------------------------------------
// batched_cholesky_solve_cpu — sequential SPD solve  A_k x_k = b_k.
//
// Same three phases as the GPU kernel (factorize → forward → backward), same
// column-Cholesky formulation, same NaN policy on a non-positive pivot —
// deliberately a line-by-line CPU twin of batched_cholesky_solve_kernel so a
// learner can diff the two and see that ONLY the parallel plumbing differs.
//
// Scratch storage: small fixed-size stack arrays (kMaxN=8 comfortably covers
// the robotics sizes and the GPU path's 3/4/6). Stack arrays keep the oracle
// allocation-free and trivially thread-safe, at the price of a hard n cap —
// checked, not assumed.
// ---------------------------------------------------------------------------
void batched_cholesky_solve_cpu(int n, int count,
                                const float* A, const float* b, float* x)
{
    constexpr int kMaxN = 8;           // covers n=3/4/6 with margin; see header comment
    if (n < 1 || n > kMaxN) {
        // Fail LOUDLY on a size the scratch buffers cannot hold — silently
        // truncating would corrupt the oracle and poison every comparison.
        // (The GPU launcher enforces its own {3,4,6} restriction separately.)
        return;                        // n is validated by main.cu before we get here; belt and suspenders
    }

    const float nan_f = std::numeric_limits<float>::quiet_NaN();  // host twin of CUDART_NAN_F

    float m[kMaxN * kMaxN];            // working copy: A_k's lower triangle → L (in place)
    float v[kMaxN];                    // working vector: b_k → y → x (in place)

    for (int k = 0; k < count; ++k) {  // one system after another (GPU: one thread each)
        const float* a_k = A + static_cast<size_t>(k) * n * n;
        const float* b_k = b + static_cast<size_t>(k) * n;
        float*       x_k = x + static_cast<size_t>(k) * n;

        // Stage the lower triangle (upper entries zeroed — never read, same
        // as the GPU staging loop).
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                m[i * n + j] = (j <= i) ? a_k[i * n + j] : 0.0f;
        for (int i = 0; i < n; ++i) v[i] = b_k[i];

        // --- 1) Factorize A = L·Lᵀ, column by column ------------------------
        bool spd = true;               // every pivot must be strictly positive
        for (int j = 0; j < n; ++j) {
            // l_jj = sqrt( a_jj − Σ_{p<j} l_jp² )
            float diag = m[j * n + j];
            for (int p = 0; p < j; ++p)
                diag -= m[j * n + p] * m[j * n + p];

            spd = spd && (diag > 0.0f);
            float l_jj = std::sqrt(diag > 0.0f ? diag : 0.0f);  // clamp mirrors the GPU's fmaxf guard
            m[j * n + j] = l_jj;
            float inv_l_jj = (l_jj > 0.0f) ? (1.0f / l_jj) : 0.0f;

            // l_ij = ( a_ij − Σ_{p<j} l_ip·l_jp ) / l_jj  for the column below
            for (int i = j + 1; i < n; ++i) {
                float s = m[i * n + j];
                for (int p = 0; p < j; ++p)
                    s -= m[i * n + p] * m[j * n + p];
                m[i * n + j] = s * inv_l_jj;
            }
        }

        // --- 2) Forward substitution L·y = b (v becomes y) ------------------
        for (int i = 0; i < n; ++i) {
            float s = v[i];
            for (int p = 0; p < i; ++p)
                s -= m[i * n + p] * v[p];
            v[i] = s / m[i * n + i];   // division by 0 only when !spd; overwritten below
        }

        // --- 3) Back substitution Lᵀ·x = y (v becomes x) ---------------------
        for (int i = n - 1; i >= 0; --i) {
            float s = v[i];
            for (int p = i + 1; p < n; ++p)
                s -= m[p * n + i] * v[p];   // Lᵀ(i,p) = L(p,i): transposed index, like the GPU
            v[i] = s / m[i * n + i];
        }

        // --- Write back with the shared NaN-on-non-SPD policy ---------------
        for (int i = 0; i < n; ++i)
            x_k[i] = spd ? v[i] : nan_f;
    }
}
