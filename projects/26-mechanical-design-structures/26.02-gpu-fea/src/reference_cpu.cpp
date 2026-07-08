// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 26.02
//                     (GPU FEA: static, modal (arm vibration modes), harmonic response)
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
