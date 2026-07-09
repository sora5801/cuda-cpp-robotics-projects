// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 31.01
//                     Hamilton-Jacobi reachability (double-integrator core)
//
// Two jobs in this project (both declared in kernels.cuh):
//
//   1. hj_solve_cpu — the TWIN of the GPU solver: the identical FP32 update
//      expression, the identical ping-pong discipline, sequential over
//      cells. main.cu runs both from the same initial field and requires
//      max |V_gpu - V_cpu| <= kTwinTol — the §5 GPU-vs-CPU gate. It also
//      serves as the honest timing baseline for the [time] line.
//
//   2. min_time_to_origin — the ANALYTIC oracle: the closed-form bang-bang
//      minimum-time solution of the double integrator. This one is NOT a
//      twin of anything on the GPU — it is independent MATHEMATICS, in
//      double precision, that the PDE answer is checked against. Verifying
//      a numerical scheme against a known exact solution is the strongest
//      check this repository gets to make; this project exists partly to
//      show what that looks like (THEORY.md §verify).
//
// The sweep function below is a deliberate line-by-line twin of the
// __global__ kernel in kernels.cu — diff the files: only the launch
// scaffolding and fminf/fabsf spellings differ. Deliberate, documented
// duplication (CLAUDE.md §5): the twins must be diffable.
//
// Read this after: kernels.cuh.  Read this beside: kernels.cu.
// ===========================================================================

#include "kernels.cuh"   // shared HjGrid, layout contract, signatures

#include <cmath>         // std::fabs, std::sqrt, std::fmin — float & double overloads
#include <vector>        // the CPU-side ping-pong partner buffer

// ---------------------------------------------------------------------------
// hj_sweep_cell — one cell's PDE update: the CPU spelling of the body of
// hj_sweep_kernel (see kernels.cu for the full math commentary — not
// repeated here; the ARITHMETIC must stay identical, expression for
// expression, so the twin comparison stays meaningful).
//
//   in   : the value snapshot at tau (nx*nv floats, V[j*nx+i] layout)
//   i, j : the cell (position index, velocity index)
// Returns the cell's value at tau + dt.
// ---------------------------------------------------------------------------
static float hj_sweep_cell(const HjGrid& g, const float* in, int i, int j)
{
    const int idx = j * g.nx + i;
    const float c = in[idx];

    // Face neighbors with the same linear-extrapolation border ghosts.
    const float xm = (i > 0)        ? in[idx - 1]    : 2.0f * c - in[idx + 1];
    const float xp = (i < g.nx - 1) ? in[idx + 1]    : 2.0f * c - in[idx - 1];
    const float vm = (j > 0)        ? in[idx - g.nx] : 2.0f * c - in[idx + g.nx];
    const float vp = (j < g.nv - 1) ? in[idx + g.nx] : 2.0f * c - in[idx - g.nx];

    // One-sided difference pairs, Hamiltonian, dissipation — kernels.cu's
    // expressions with std:: spellings (float overloads throughout).
    const float pxm = (c - xm) / g.dx;
    const float pxp = (xp - c) / g.dx;
    const float pvm = (c - vm) / g.dv;
    const float pvp = (vp - c) / g.dv;

    const float vj = g.vmin + (float)j * g.dv;
    const float h  = 0.5f * (pxm + pxp) * vj
                   - g.umax * std::fabs(0.5f * (pvm + pvp));

    const float diss = 0.5f * std::fabs(vj) * (pxp - pxm)
                     + 0.5f * g.umax        * (pvp - pvm);

    // Freeze-and-step (tube semantics: the value may only decrease).
    return c + g.dt * std::fmin(0.0f, h + diss);
}

// ---------------------------------------------------------------------------
// hj_solve_cpu — all sweeps, all cells, one after another (the GPU gives
// each cell its own thread). The ping-pong here uses a std::vector partner
// exactly as the launcher uses a second device buffer — updating in place
// would corrupt neighbors mid-sweep on the CPU just as surely as on the
// GPU; the race is in the ALGORITHM, not in the parallelism.
// ---------------------------------------------------------------------------
void hj_solve_cpu(const HjGrid& g, int n_sweeps, float* V)
{
    const size_t total = (size_t)g.nx * g.nv;
    std::vector<float> pong(total);              // the CPU ping-pong partner

    float* in  = V;                              // caller's buffer holds l = V(tau=0)
    float* out = pong.data();

    for (int s = 0; s < n_sweeps; ++s) {
        // j outer / i inner matches the memory layout (i contiguous), so
        // the CPU walks memory sequentially — the same locality argument
        // as GPU coalescing, in single-core form.
        for (int j = 0; j < g.nv; ++j)
            for (int i = 0; i < g.nx; ++i)
                out[j * g.nx + i] = hj_sweep_cell(g, in, i, j);
        float* tmp = in; in = out; out = tmp;    // swap, same as the launcher
    }

    // Result must end in the caller's buffer regardless of sweep parity —
    // the same guarantee launch_hj_solve makes with its device copy.
    if (in != V)
        for (size_t k = 0; k < total; ++k) V[k] = in[k];
}

// ---------------------------------------------------------------------------
// min_time_to_origin — closed-form minimum time to drive the double
// integrator from (x, v) to the origin (0, 0) under |u| <= umax.
//
//   x    : position (m); v : velocity (m/s); umax : accel bound (m/s^2), > 0.
//   Returns the minimum time T* in seconds (>= 0; 0 exactly at the origin).
//
// The classical Pontryagin result (derived in THEORY.md §the-math): the
// time-optimal control is BANG-BANG with at most one switch — full thrust
// one way, then full thrust the other. The switch happens on the SWITCHING
// CURVE, the pair of parabolic arcs through the origin that full-thrust
// trajectories arrive along:
//
//        x = -v|v| / (2 umax)          (both branches in one formula)
//
//   * State RIGHT of the curve: thrust u = -umax first (drive left/brake),
//     coast down onto the u = +umax arrival parabola, ride it in:
//         T* = ( v + 2*sqrt(v^2/2 + umax*x) ) / umax
//   * State LEFT of the curve: the mirror image (u = +umax first):
//         T* = (-v + 2*sqrt(v^2/2 - umax*x) ) / umax
//   * ON the curve: pure arrival, no switch:  T* = |v| / umax.
//
// Double precision, no shortcuts: this oracle must be beyond suspicion,
// because the whole point of the demo's ANALYTIC stage is that the grid
// solver answers to mathematics, not to another copy of itself.
// ---------------------------------------------------------------------------
double min_time_to_origin(double x, double v, double umax)
{
    // Signed distance-like test against the switching curve: s > 0 means
    // the state lies to the RIGHT of x = -v|v|/(2 umax).
    const double s = x + 0.5 * v * std::fabs(v) / umax;

    if (s > 0.0) {
        // Right of the curve. The sqrt argument v^2/2 + umax*x is provably
        // >= 0 here (it equals umax*s plus a non-negative term); the fmax
        // guard only absorbs last-ulp rounding for states ON the curve.
        return (v + 2.0 * std::sqrt(std::fmax(0.0, 0.5 * v * v + umax * x))) / umax;
    }
    if (s < 0.0) {
        // Left of the curve — the (x,v) -> (-x,-v) mirror of the case above.
        return (-v + 2.0 * std::sqrt(std::fmax(0.0, 0.5 * v * v - umax * x))) / umax;
    }
    // Exactly on the switching curve: ride the arrival parabola to rest.
    return std::fabs(v) / umax;
}
