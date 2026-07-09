// ===========================================================================
// kernels.cuh — interface for project 33.01
//               Batched small-matrix linear algebra (3×3, 4×4, 6×6 —
//               the robotics sizes)
//
// Role in the project
// -------------------
// This header is the CONTRACT between the three translation units:
//   * main.cu           — the driver: builds batches, calls both paths, compares
//   * kernels.cu        — the GPU implementation (nvcc-compiled)
//   * reference_cpu.cpp — the CPU correctness oracle (cl.exe-compiled)
// Declaring the GPU launcher and the CPU reference side by side, with the
// SAME shapes and layout rules, is what keeps the two paths comparable at
// compile time instead of drifting apart silently.
//
// The one data-layout rule everything here shares
// -----------------------------------------------
// A "batch" is `count` square N×N matrices stored CONTIGUOUSLY, row-major,
// matrix-after-matrix ("array of matrices"):
//
//     element (i,j) of matrix k lives at   ptr[k*N*N + i*N + j]
//
// and a batch of right-hand-side vectors stores vector k at ptr[k*N + i].
// This layout is documented ONCE, here, and every function below references
// it (CLAUDE.md §12: state layouts are single-sourced). THEORY.md §GPU-mapping
// discusses why this "matrix-contiguous" layout is chosen for clarity, what
// it costs in memory-coalescing terms, and the SoA alternative real libraries
// (cuBLAS gemmBatched, MAGMA) use to win those loads back.
//
// Why there is no __CUDACC__-fenced __global__ section here (unlike the
// repository template): every kernel launch happens inside kernels.cu, next
// to the kernel definitions, so no other translation unit ever needs to see
// a __global__ signature. The header stays host-only and any compiler —
// nvcc or cl.exe — can include it without fences. Simpler is better to read.
//
// Why N is a runtime `int` here but a compile-time template inside kernels.cu:
// callers (main.cu) want one function that handles the three robotics sizes;
// kernels want N known at compile time so matrices live in REGISTERS and all
// loops fully unroll. The launchers below bridge the two worlds with a
// switch(n) that dispatches to the N=3 / N=4 / N=6 instantiation, and reject
// any other size loudly (fail-fast beats a silently-wrong fallback).
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH   // classic include guard: safe on every compiler
#define PROJECT_KERNELS_CUH

// ---------------------------------------------------------------------------
// launch_batched_matmul — GPU batched product  C_k = A_k * B_k,  k = 0..count-1.
//
// Parameters
//   n       : matrix dimension; MUST be 3, 4, or 6 (any other value aborts
//             with a message — see the dispatch comment in kernels.cu).
//   count   : number of independent matrix pairs in the batch (>= 0).
//   d_A,d_B : DEVICE pointers, count*n*n floats each, layout as above. Never
//             written. Unitless in this demo; in a real consumer these carry
//             whatever the caller's units are (e.g. rotation composition —
//             dimensionless; inertia transforms — kg·m²).
//   d_C     : DEVICE pointer, count*n*n floats, OVERWRITTEN with the products.
//             May NOT alias d_A or d_B (each thread reads its full A_k/B_k
//             before writing C_k, but only within its own k; aliasing across
//             the batch is not defended against — documented, not checked).
//
// Launch configuration: one THREAD per matrix pair, 256-thread blocks,
// ceil(count/256) blocks — the reasoning lives with the kernel in kernels.cu.
// Synchronization: none; work is enqueued on the default stream. The caller
// times/synchronizes via cudaEvents (util/timer.cuh) or cudaMemcpy.
// Complexity: O(count · n³) multiply-adds, perfectly parallel across k.
// ---------------------------------------------------------------------------
void launch_batched_matmul(int n, int count,
                           const float* d_A, const float* d_B, float* d_C);

// ---------------------------------------------------------------------------
// launch_batched_cholesky_solve — GPU batched SPD solve  A_k x_k = b_k.
//
// Solves count independent linear systems whose matrices are symmetric
// positive definite (SPD) — the bread-and-butter robotics case: joint-space
// mass matrices M(q), Gauss-Newton normal equations JᵀJ + λI, covariance
// matrices. Method: in-register Cholesky factorization A = L·Lᵀ followed by
// forward substitution (L y = b) and back substitution (Lᵀ x = y).
//
// Parameters
//   n     : matrix dimension; MUST be 3, 4, or 6 (else abort, as above).
//   count : number of systems (>= 0).
//   d_A   : DEVICE pointer, count*n*n floats, SPD matrices (layout above).
//           Only the lower triangle including the diagonal is READ — exactly
//           what Cholesky consumes; the strict upper triangle is ignored, so
//           a caller that only filled the lower half is fine.
//   d_b   : DEVICE pointer, count*n floats, right-hand sides. Never written.
//   d_x   : DEVICE pointer, count*n floats, OVERWRITTEN with solutions.
//
// Non-SPD input: if a pivot is not strictly positive the matrix is not SPD
// (or is numerically singular); that system's x_k is filled with NaN so the
// failure is IMPOSSIBLE to miss downstream, and computation continues for
// the other k (one bad system must not poison a 100k-system batch). The CPU
// reference implements the identical policy so the comparison stays valid.
// ---------------------------------------------------------------------------
void launch_batched_cholesky_solve(int n, int count,
                                   const float* d_A, const float* d_b,
                                   float* d_x);

// ---------------------------------------------------------------------------
// CPU references (defined in reference_cpu.cpp — the correctness oracle).
//
// Same math, same layout rule, same NaN-on-non-SPD policy, plain single-
// threaded C++ with no CUDA anywhere — slow on purpose and easy to read.
// main.cu runs BOTH paths on identical inputs and asserts agreement within
// the tolerances documented in main.cu's output contract (CLAUDE.md §5:
// every project verifies GPU against CPU).
// All pointers are HOST pointers with the same shapes as the GPU twins.
// ---------------------------------------------------------------------------
void batched_matmul_cpu(int n, int count,
                        const float* A, const float* B, float* C);

void batched_cholesky_solve_cpu(int n, int count,
                                const float* A, const float* b, float* x);

#endif // PROJECT_KERNELS_CUH
