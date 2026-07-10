// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 27.04
//                     Composite layup optimization + Tsai-Wu failure envelope
//
// Two jobs in this project (both declared in kernels.cuh):
//
//   1. layup_sweep_cpu — the ORACLE twin of layup_sweep_kernel: same
//      decode_layup, same laminate_failure_factor (the exact SAME
//      __host__ __device__ inline functions the kernel calls — see
//      kernels.cuh's HD note), sequential over every (layup, case) pair.
//      main.cu runs it against the GPU on IDENTICAL inputs and requires
//      full-array agreement within a documented tolerance — the §5
//      GPU-vs-CPU gate. Unlike 18.01's stride-sampled spot check, this
//      project's problem sizes are tiny enough (at most kEnvGridN^2 =
//      16384 points) that the oracle recomputes EVERY element.
//
//   2. envelope_cpu — the ORACLE twin of envelope_kernel: same
//      envelope_factor_at, full 128x128 grid.
//
// Because the physics functions are HD (__host__ __device__) inline
// functions defined ONCE in kernels.cuh, this file does not re-implement
// any math — it only supplies the ORCHESTRATION loops that the kernel
// spreads across threads. Diffing kernels.cu against this file shows
// EXACTLY what parallelization changed: nothing about the math, only the
// loop-vs-thread mapping (the same lesson 08.01's SAXPY placeholder teaches
// at the smallest possible scale).
//
// Read this after: kernels.cuh.  Read this beside: kernels.cu.
// ===========================================================================

#include "kernels.cuh"   // for Lamina/LoadCase/AngleAlphabet/Layup8 + the shared HD physics functions

// ---------------------------------------------------------------------------
// layup_sweep_cpu — every (layup_id, case_id) pair, sequential (the GPU
// gives each its own thread). Uses the SAME flattening g = layup_id*n_cases
// + case_id the kernel writes — the oracle honors it too (kernels.cuh's
// file header: this flattening is a data contract, not a kernel detail).
// ---------------------------------------------------------------------------
void layup_sweep_cpu(const Lamina& mat, const AngleAlphabet& alpha,
                     const LoadCase* cases, int n_cases, float* out_factor)
{
    for (int layup_id = 0; layup_id < kNLayups; ++layup_id) {
        float angles[kNPlies];
        decode_layup(layup_id, alpha.deg, angles);
        for (int case_id = 0; case_id < n_cases; ++case_id) {
            const int g = layup_id * n_cases + case_id;
            out_factor[g] = laminate_failure_factor(mat, angles, kNPlies, cases[case_id]);
        }
    }
}

// ---------------------------------------------------------------------------
// envelope_cpu — every (i, j) grid point, sequential. Uses envelope_factor_at
// directly (the exact function the kernel calls) so the clamp-near-zero-load
// behavior cannot itself be a source of GPU-vs-CPU disagreement.
// ---------------------------------------------------------------------------
void envelope_cpu(const Lamina& mat, const Layup8& layup, float n_max_npm, float* out_factor)
{
    for (int i = 0; i < kEnvGridN; ++i) {
        for (int j = 0; j < kEnvGridN; ++j) {
            const int g = i * kEnvGridN + j;
            out_factor[g] = envelope_factor_at(mat, layup.deg, kNPlies, i, j, n_max_npm);
        }
    }
}
