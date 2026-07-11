// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project {{PROJECT_ID}}
//                     ({{PROJECT_NAME}})
//
// TEMPLATE PLACEHOLDER — replace with this project's real CPU reference.
// TODO(scaffold): implement the real algorithm here in the simplest, most
// readable C++ you can write — clarity beats speed in this file, always.
//
// WHY does a GPU repository ship a CPU implementation of everything?
// ------------------------------------------------------------------
// Two load-bearing reasons (CLAUDE.md §5):
//
//   1) It is the CORRECTNESS ORACLE. GPU code fails in ways CPU code cannot:
//      wrong thread indexing, missed tail elements, race conditions, stale
//      device memory, bad transfers. A dead-simple sequential version that a
//      reader can verify BY EYE gives us ground truth; main.cu runs both and
//      asserts element-wise agreement within a documented tolerance. If the
//      two disagree, the bug hunt starts with certainty that a bug exists.
//
//   2) It is the TEACHING BASELINE. The GPU version only makes sense as a
//      transformation OF something — this file is that something. Reading it
//      first, then kernels.cu, shows exactly what parallelization changed
//      (spoiler for SAXPY: the loop became threads; the body is identical).
//      It also makes the printed speed-up legible: same machine, same data,
//      same algorithm — one core vs. thousands of threads.
//
// Rules for this file: plain C++17, no CUDA headers, no hand-vectorization,
// no OpenMP, no cleverness. If the reference is clever, it can be wrong, and
// then the oracle lies. (This file is compiled by the HOST compiler, cl.exe;
// the __CUDACC__ fence in kernels.cuh hides device declarations from it.)
//
// Independence ruling (Phase-1 standards retrospective — load-bearing)
// --------------------------------------------------------------------
// How much code may this file SHARE with the GPU path? The flagships split
// both ways, and one of them settled the question the hard way:
//
//   * Data-layout contracts (structs, constants, indexing formulas) MUST be
//     single-sourced in kernels.cuh and shared. Divergent layouts between
//     the twins are a bug class of their own, not "independence".
//   * The ALGORITHMIC CORE should be written twice — independently, in the
//     simplest possible C++ here. That is the default, because the twin
//     comparison only catches bugs the two paths DON'T share.
//   * A shared __host__ __device__ helper is permitted when duplicating it
//     would be pure token-for-token transcription (e.g., the dynamics model
//     that IS the system under test in both paths) — but then the twin
//     comparison is BLIND to bugs inside that helper, so the project MUST
//     also carry at least one verification gate that does not route through
//     the shared code: a closed-form/analytic solution, a physical
//     invariant, or a negative control.
//
// Why this is not paranoia: in flagship 13.03 an identical variable-
// shadowing bug lived in BOTH the GPU path and this file's counterpart —
// the element-wise twin comparison passed perfectly, and only the analytic
// ramp-angle gate (15.00° known answer) exposed it. Twin agreement proves
// the parallelization is faithful to the reference; only an INDEPENDENT
// gate proves the reference itself is right. Every project needs both
// tiers (document yours in THEORY.md "How we verify correctness").
//
// Read this after: kernels.cu — then compare the two side by side.
// ===========================================================================

#include "kernels.cuh"  // for the saxpy_cpu prototype: compiler-enforced
                        // signature agreement with what main.cu calls.

// ---------------------------------------------------------------------------
// saxpy_cpu — sequential y[i] = a * x[i] + y[i] for i = 0 .. n-1.
//
// Parameters:
//   n — element count (> 0)
//   a — the SAXPY scalar
//   x — [n] host pointer, read-only input
//   y — [n] host pointer, read AND written in place (input y, output a*x+y)
//
// Complexity: O(n) time, O(1) extra space. Side effects: overwrites y.
//
// Numerical note (why GPU-vs-CPU comparison needs a tolerance): this line
// may be compiled as a separate multiply and add (two rounding steps) or as
// a fused multiply-add (one rounding step), depending on compiler flags; the
// GPU almost always fuses. Both are correct; they may differ in the last bit
// (~1 ULP). Hence main.cu's tolerance of 1e-6 rather than bit-equality —
// the honest way to compare floating point across compilers and devices.
// ---------------------------------------------------------------------------
void saxpy_cpu(int n, float a, const float* x, float* y)
{
    // One loop, one line — deliberately the simplest correct statement of
    // the computation. This is the exact loop the GPU kernel parallelizes:
    // in kernels.cu, "for each i" became "each thread owns some i".
    for (int i = 0; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}
