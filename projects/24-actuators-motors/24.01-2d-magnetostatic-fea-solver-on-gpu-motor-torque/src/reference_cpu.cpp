// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 24.01
//                     2D magnetostatic FEA solver on GPU -> motor torque-
//                     ripple/cogging parameter sweeps
//
// One job in this project (declared in kernels.cuh):
//
//   fea_solve_batch_cpu — the ORACLE twin of the GPU batched red-black SOR
//   kernel: the IDENTICAL FP32 update expression, the IDENTICAL red-then-
//   black pass order, sequential over (b, j, i) instead of one thread per
//   (b, i, j). main.cu runs it against the GPU on one representative motor
//   variant (the same nu/Jsrc inputs, the same n_sweeps and omega) and
//   requires the two vector-potential fields to agree within kTwinTolAbs —
//   the paragraph 5 GPU-vs-CPU gate for this project. It also serves as the
//   honest timing baseline printed on the VERIFY stage's [time] line:
//   "a CPU manages one variant at a time" is measured here, not asserted.
//
// Geometry rasterization, the permanent-magnet equivalent-current source,
// and the Maxwell-stress-tensor torque integral are NOT duplicated here —
// they are SETUP and POST-PROCESSING, not the solver, and CLAUDE.md's
// "deliberate, documented duplication" rule applies to the algorithm this
// project TEACHES (the stencil relaxation), not to every line of host code.
// main.cu builds the nu/Jsrc arrays once and feeds them IDENTICALLY into
// both this oracle and the GPU path — exactly 31.01's precedent (its
// initial condition l0 is built once in main.cu, not duplicated per path).
//
// The loop body below is a deliberate line-by-line twin of the per-cell
// update in kernels.cu's __global__ kernel — diff the two files: only the
// launch/thread-index machinery differs; the ARITHMETIC is identical so the
// twin comparison in main.cu actually means something.
//
// Read this after: kernels.cuh.  Read this beside: kernels.cu.
// ===========================================================================

#include "kernels.cuh"   // shared FeaGrid, layout contract, signatures

// ---------------------------------------------------------------------------
// fea_sor_pass_cpu — one RED or BLACK half-sweep, all B variants, all cells,
// one after another (the GPU gives each (variant, cell) pair its own
// thread; here a single core walks them in row-major order per variant,
// which is also the memory-friendly order — the same locality argument as
// GPU coalescing, in single-core form, per 31.01's precedent).
//
// The update expression is copy-pasted from kernels.cu's kernel body with
// only the indexing spelled as nested loops instead of thread coordinates —
// see that file for the full derivation commentary (not repeated here; the
// MATH must stay identical, expression for expression).
// ---------------------------------------------------------------------------
static void fea_sor_pass_cpu(const FeaGrid& g, int B, int color,
                             const float* nu, const float* Jsrc, float* A)
{
    const float h2 = g.h * g.h;
    for (int b = 0; b < B; ++b) {
        const int base = b * g.ny * g.nx;
        for (int j = 1; j < g.ny - 1; ++j) {          // interior rows only (border = Dirichlet A=0)
            for (int i = 1; i < g.nx - 1; ++i) {       // interior columns only
                if (((i + j) & 1) != color) continue;  // this pass's checkerboard color only

                const int idx = base + j * g.nx + i;

                const float nu_c = nu[idx];
                const float nu_e = 0.5f * (nu_c + nu[idx + 1]);
                const float nu_w = 0.5f * (nu_c + nu[idx - 1]);
                const float nu_n = 0.5f * (nu_c + nu[idx + g.nx]);
                const float nu_s = 0.5f * (nu_c + nu[idx - g.nx]);
                const float diag = nu_e + nu_w + nu_n + nu_s;

                const float Ae = A[idx + 1];
                const float Aw = A[idx - 1];
                const float An = A[idx + g.nx];
                const float As = A[idx - g.nx];

                const float gs_target =
                    (nu_e * Ae + nu_w * Aw + nu_n * An + nu_s * As + h2 * Jsrc[idx]) / diag;

                A[idx] = A[idx] + g.omega * (gs_target - A[idx]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// fea_solve_batch_cpu — n_sweeps red+black pass pairs, in place, over the
// whole batch. Mirrors launch_fea_solve_batch's loop in kernels.cu exactly
// (same pass order: red, then black, n_sweeps times) so the two are
// comparing the SAME sequence of arithmetic operations, not just the same
// converged answer — a bug in pass ORDER would otherwise still converge to
// a *different but plausible* field and slip past a looser check.
// ---------------------------------------------------------------------------
void fea_solve_batch_cpu(const FeaGrid& g, int B, int n_sweeps,
                         const float* nu, const float* Jsrc, float* A)
{
    for (int s = 0; s < n_sweeps; ++s) {
        fea_sor_pass_cpu(g, B, /*color=*/0, nu, Jsrc, A);   // red pass
        fea_sor_pass_cpu(g, B, /*color=*/1, nu, Jsrc, A);   // black pass
    }
}
