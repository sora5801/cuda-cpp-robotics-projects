// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 34.03
//                     Ergodic control: spectral multiscale coverage (SMC)
//
// Two jobs, like 08.01's reference_cpu.cpp:
//
//   1) The ORACLE twins of kernels.cu's two GPU paths:
//        phi_k_direct_cpu — an independent, NO-FFT, direct trapezoidal
//          cosine projection (precomputed cosine tables, so still fast —
//          see the comment above the function). Compared against
//          launch_build_phi_k's DCT-via-FFT result: this project's
//          TRANSFORM-CORRECTNESS gate (README §Expected output).
//        smc_step_cpu — sequential, line-by-line twin of smc_step_kernel.
//          Compared against launch_smc_step over a short window: the §5
//          GPU-vs-CPU VERIFY gate (CLAUDE.md §5).
//
//   2) integrate_agent_cpu — THE PLANT. The closed loop needs something to
//      actually move the agent each control step; a 2-state Euler
//      integrator + wall reflection is trivial serial work, so — exactly
//      like 08.01's cartpole_step_cpu is both oracle-adjacent code AND the
//      plant the controller drives — it lives here, once, host-only.
//
// Rules for this file (CLAUDE.md §5): plain C++17, NO CUDA/cuFFT headers —
// the correctness oracle must never depend on nvcc, so it independently
// proves the GPU pipeline right instead of assuming it. basis_norm_h_cpu,
// the cos/sin basis evaluation, and the Sobolev weight below are
// DELIBERATE, DOCUMENTED DUPLICATES of kernels.cu's __device__ versions —
// diff the two files: only the CUDA attributes and a few spellings differ,
// the math is identical line-for-line (the same discipline 08.01 uses for
// cartpole_deriv).
//
// Read this after: kernels.cuh.  Read this beside: kernels.cu.
// ===========================================================================

#include "kernels.cuh"   // shared constants, layouts, and the prototypes below

#include <cmath>         // std::cos, std::sin, std::sqrt
#include <vector>        // cos_table / w1d scratch in phi_k_direct_cpu

// ---------------------------------------------------------------------------
// basis_norm_h_cpu — host twin of kernels.cu's basis_norm_h_dev. See that
// function's comment for the derivation (h_k in {1, 1/sqrt2, 1/2}).
// ---------------------------------------------------------------------------
static double basis_norm_h_cpu(int k1, int k2)
{
    const bool z1 = (k1 == 0), z2 = (k2 == 0);
    if (z1 && z2) return 1.0;
    if (z1 || z2) return 0.70710678118654752440;   // 1/sqrt(2)
    return 0.5;
}

// ---------------------------------------------------------------------------
// phi_k_direct_cpu — the TRANSFORM-CORRECTNESS oracle: compute every
// phi_k directly from its definition (a trapezoidal double integral),
// WITHOUT any FFT — a completely independent code path from
// launch_build_phi_k's cuFFT pipeline.
//
// Naive nested loops (k1,k2,n,m) would be O(kPhiGridN^2 * kNumModes) =
// 129^2 * 1024 ~ 17M iterations, EACH calling cos() twice (~34M
// transcendental calls) — correct, but wastefully slow for something that
// runs once at startup. The fix used here costs the SAME O(N^2*K) work but
// with ZERO trig calls in the hot loop: precompute a small
// cos_table[k][n] = cos(k*pi*x_n) table ONCE (O(N*K) work, trivial), then
// the double integral becomes pure multiply-accumulate table lookups. This
// is still the DIRECT definition (no FFT, no O(N log N) trick) — only the
// REDUNDANT trig evaluations are removed, exactly the kind of honest
// "still O(N^2*K), just not wasteful" optimization CLAUDE.md §1's "teaching
// beats cleverness" allows without compromising this function's role as an
// independent oracle.
//
// Grid & weights: kPhiGridN points per axis, spacing h=1/(kPhiGridN-1),
// TRAPEZOIDAL weight w1d(n) = 0.5 at n=0 or n=N-1, else 1.0 — the same
// weighting the DCT-I identity in kernels.cu's file header derives from.
// ---------------------------------------------------------------------------
void phi_k_direct_cpu(const double* phi_grid, double* phi_k_out)
{
    const int N = kPhiGridN;
    const double h = 1.0 / static_cast<double>(N - 1);

    // cos_table[k*N + n] = cos(k*pi*x_n), x_n = n*h. Same table serves BOTH
    // axes (the grid and basis are identical along x1 and x2).
    std::vector<double> cos_table(static_cast<size_t>(kK) * N);
    for (int k = 0; k < kK; ++k) {
        for (int n = 0; n < N; ++n) {
            const double xn = static_cast<double>(n) * h;
            cos_table[static_cast<size_t>(k) * N + n] = std::cos(static_cast<double>(k) * kPi * xn);
        }
    }

    // Trapezoidal edge weights: 0.5 at the two endpoints, 1.0 interior —
    // the standard trapezoidal-rule quadrature weight, applied per axis.
    std::vector<double> w1d(N, 1.0);
    w1d[0] = 0.5;
    w1d[N - 1] = 0.5;

    // Direct double integral, per mode: I[k1,k2] = h^2 * sum_n sum_m
    // w1d(n)*w1d(m)*phi[n,m]*cos_table[k1,n]*cos_table[k2,m]; then
    // orthonormalize by h_k. This is EXACTLY the quantity
    // launch_build_phi_k computes via the DCT-I-via-FFT identity (kernels.cu
    // derives the equivalence in full) — two independent code paths that
    // must agree to near machine precision if both are correct.
    for (int k1 = 0; k1 < kK; ++k1) {
        for (int k2 = 0; k2 < kK; ++k2) {
            double acc = 0.0;
            for (int n = 0; n < N; ++n) {
                const double wn = w1d[n];
                const double cn = cos_table[static_cast<size_t>(k1) * N + n];
                if (wn == 0.0 || cn == 0.0) continue;   // (never true here; guard is defensive, not load-bearing)
                double row = 0.0;
                const double* phi_row = phi_grid + static_cast<size_t>(n) * N;
                const double* ck2 = &cos_table[static_cast<size_t>(k2) * N];
                for (int m = 0; m < N; ++m) {
                    row += w1d[m] * phi_row[m] * ck2[m];
                }
                acc += wn * cn * row;
            }
            const double I_k = acc * h * h;
            phi_k_out[static_cast<size_t>(k1) * kK + k2] = I_k / basis_norm_h_cpu(k1, k2);
        }
    }
}

// ---------------------------------------------------------------------------
// smc_step_cpu — sequential, line-by-line twin of kernels.cu's
// smc_step_kernel: one iteration of the loop below == one GPU thread.
// Same math, same running-sum state layout (the caller owns S[] exactly as
// it owns the GPU's d_S — see kernels.cuh's launch_smc_step doc comment).
// ---------------------------------------------------------------------------
void smc_step_cpu(double x1, double x2, const double* phi_k,
                  double* S, int n,
                  double* c, double* Bx, double* By)
{
    for (int idx = 0; idx < kNumModes; ++idx) {
        const int k1 = idx / kK;
        const int k2 = idx % kK;
        const double h = basis_norm_h_cpu(k1, k2);

        const double cx1 = std::cos(static_cast<double>(k1) * kPi * x1);
        const double cx2 = std::cos(static_cast<double>(k2) * kPi * x2);
        const double f = (cx1 * cx2) / h;

        const double s_new = S[idx] + f;
        S[idx] = s_new;
        const double c_val = s_new / static_cast<double>(n);

        const double diff = c_val - phi_k[idx];
        const double kk = static_cast<double>(k1 * k1 + k2 * k2);
        const double lambda = 1.0 / ((1.0 + kk) * std::sqrt(1.0 + kk));

        const double sx1 = std::sin(static_cast<double>(k1) * kPi * x1);
        const double sx2 = std::sin(static_cast<double>(k2) * kPi * x2);
        const double dfdx1 = (-static_cast<double>(k1) * kPi * sx1 * cx2) / h;
        const double dfdx2 = (-cx1 * static_cast<double>(k2) * kPi * sx2) / h;

        c[idx] = c_val;
        Bx[idx] = lambda * diff * dfdx1;
        By[idx] = lambda * diff * dfdx2;
    }
}

// ---------------------------------------------------------------------------
// integrate_agent_cpu — THE PLANT: one Euler step of xdot = u, dt seconds,
// then REFLECT off the [0,1]^2 domain walls.
//
// This is the project's SINGLE DEFINED BOUNDARY POINT (CLAUDE.md §12,
// mirroring 08.01's single defined angle-wrap point in cartpole_step_cpu).
// WHY REFLECT (not clamp, not wrap)? The cosine basis f_k = cos(k*pi*x) has
// ZERO DERIVATIVE at x=0 and x=1 — the Neumann ("no flux through the wall")
// boundary condition — so the basis itself already models a domain the
// agent cannot leave and does not wrap around. A billiard-style reflection
// is the trajectory behavior physically consistent with that assumption;
// clamping (sticking to the wall) or wrapping (teleporting to the far
// side) would each silently violate it. THEORY.md §numerics discusses the
// (small, at this kVmax*kDt step size) simplification of allowing at most
// ONE reflection per axis per step rather than looping to convergence.
// ---------------------------------------------------------------------------
void integrate_agent_cpu(double x[2], double u1, double u2, double dt)
{
    x[0] += dt * u1;
    x[1] += dt * u2;

    // Reflect through whichever wall was crossed. kVmax*kDt = 0.4*0.01 =
    // 0.004 << 1, so a position can overshoot a wall by at most that much
    // per step — a single reflection per axis always suffices; the trailing
    // clamp is a defensive belt-and-suspenders bound, not load-bearing.
    if (x[0] < 0.0) x[0] = -x[0];
    if (x[0] > 1.0) x[0] = 2.0 - x[0];
    if (x[1] < 0.0) x[1] = -x[1];
    if (x[1] > 1.0) x[1] = 2.0 - x[1];

    if (x[0] < 0.0) x[0] = 0.0; if (x[0] > 1.0) x[0] = 1.0;
    if (x[1] < 0.0) x[1] = 0.0; if (x[1] > 1.0) x[1] = 1.0;
}
