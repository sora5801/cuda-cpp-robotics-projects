// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 08.01
//                     MPPI controller (cart-pole teaching core)
//
// Two jobs in this project (both declared in kernels.cuh):
//
//   1. mppi_rollouts_cpu — the ORACLE twin of the GPU rollout kernel: same
//      dynamics, same RK4, same clamps, same cost, sequential over k.
//      main.cu runs it against the GPU on identical inputs (iteration 0)
//      and requires agreement within a relative tolerance — the §5
//      GPU-vs-CPU gate. It also serves as the honest timing baseline: "a
//      CPU manages dozens of rollouts" is measured here, not asserted.
//
//   2. cartpole_step_cpu — THE PLANT. The closed-loop demo needs a "real"
//      cart-pole for the controller to drive; simulating it on the host
//      with the same RK4 keeps the demo self-contained. (Sim-as-plant means
//      zero model mismatch — deliberately ideal conditions, stated honestly
//      in README §Limitations; robustness to mismatch is an exercise.)
//
// The dynamics/integrator/cost functions below are line-by-line twins of
// the __device__ versions in kernels.cu — deliberate, documented
// duplication (diff the files: only float-function spellings differ,
// std::sin vs sinf — one of the reasons the comparison is tolerance-based).
//
// Read this after: kernels.cuh.  Read this beside: kernels.cu.
// ===========================================================================

#include "kernels.cuh"   // shared model constants, layouts, signatures

#include <cmath>         // std::sin, std::cos, float versions

// ---------------------------------------------------------------------------
// Host twins of the device model functions (see kernels.cu for the full
// physics commentary — not repeated here; the MATH must stay identical).
// ---------------------------------------------------------------------------

static void cartpole_deriv(const float* x, float u, float* xdot)
{
    const float sin_th = std::sin(x[2]);
    const float cos_th = std::cos(x[2]);

    const float total_mass = kMassCart + kMassPole;
    const float ml = kMassPole * kPoleHalfLen;

    const float tmp = (u + ml * x[3] * x[3] * sin_th) / total_mass;
    const float th_acc = (kGravity * sin_th - cos_th * tmp)
        / (kPoleHalfLen * (4.0f / 3.0f - kMassPole * cos_th * cos_th / total_mass));
    const float p_acc = tmp - ml * th_acc * cos_th / total_mass;

    xdot[0] = x[1];
    xdot[1] = p_acc;
    xdot[2] = x[3];
    xdot[3] = th_acc;
}

static void rk4_step(float* x, float u, float dt)
{
    float k1[kNX], k2[kNX], k3[kNX], k4[kNX], xt[kNX];

    cartpole_deriv(x, u, k1);
    for (int i = 0; i < kNX; ++i) xt[i] = x[i] + 0.5f * dt * k1[i];
    cartpole_deriv(xt, u, k2);
    for (int i = 0; i < kNX; ++i) xt[i] = x[i] + 0.5f * dt * k2[i];
    cartpole_deriv(xt, u, k3);
    for (int i = 0; i < kNX; ++i) xt[i] = x[i] + dt * k3[i];
    cartpole_deriv(xt, u, k4);

    for (int i = 0; i < kNX; ++i)
        x[i] += dt * (1.0f / 6.0f) * (k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + k4[i]);
}

static float stage_cost(const float* x, float u)
{
    const float upright = 1.0f - std::cos(x[2]);
    return kWAngle * upright
         + kWThdot * x[3] * x[3]
         + kWPos   * x[0] * x[0]
         + kWPdot  * x[1] * x[1]
         + kWCtrl  * u * u;
}

// ---------------------------------------------------------------------------
// mppi_rollouts_cpu — all K rollouts, one after another (the GPU gives each
// its own thread). Reads the SAME transposed noise layout eps[t*K + k] the
// kernel reads — the layout is a data contract, not a GPU implementation
// detail, so the oracle honors it too.
// ---------------------------------------------------------------------------
void mppi_rollouts_cpu(int K, const float* x0,
                       const float* u_nom, const float* eps,
                       float* cost)
{
    for (int k = 0; k < K; ++k) {
        float x[kNX];
        for (int i = 0; i < kNX; ++i) x[i] = x0[i];

        float S = 0.0f;
        for (int t = 0; t < kHorizon; ++t) {
            float u = u_nom[t] + eps[t * K + k];
            // Same clamp as the kernel — fminf/fmaxf spelled the std:: way.
            u = u < -kUmax ? -kUmax : (u > kUmax ? kUmax : u);
            rk4_step(x, u, kDt);
            S += stage_cost(x, u);
        }
        cost[k] = S;
    }
}

// ---------------------------------------------------------------------------
// cartpole_step_cpu — the plant: one dt of "reality" under constant force.
//
// This is the project's SINGLE DEFINED WRAP POINT (CLAUDE.md §12): the
// plant state keeps theta in (-pi, pi] so logs and success checks read
// naturally; rollouts integrate unwrapped (their cost uses cos(theta),
// which does not care — see stage_cost).
// ---------------------------------------------------------------------------
void cartpole_step_cpu(float* x, float u, float dt)
{
    rk4_step(x, u, dt);

    // Wrap theta into (-pi, pi]. The loop form is transparent and never
    // iterates more than once here (|thdot|·dt << 2*pi at this dt).
    const float pi = 3.14159265358979323846f;
    while (x[2] >  pi) x[2] -= 2.0f * pi;
    while (x[2] <= -pi) x[2] += 2.0f * pi;
}
