// ===========================================================================
// kernels.cuh — interface for project 31.01
//               Hamilton-Jacobi reachability: level-set grid solvers
//               (teaching core: double-integrator backward reachable tube)
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (the driver), kernels.cu (the GPU stencil
// solver), and reference_cpu.cpp (the CPU twin + the analytic minimum-time
// oracle). Everything all three must agree on — the grid layout, the value-
// function array layout, the PDE update rule, and the solver constants — is
// defined HERE, once (CLAUDE.md §12).
//
// HJ reachability in five lines (THEORY.md derives it properly):
//   1. Pick a TARGET set (states you want to be able to reach — "safe").
//   2. Encode it as the zero sublevel set of a level function l(x,v) <= 0.
//   3. Solve the Hamilton-Jacobi PDE  V_t + min(0, min_u grad(V)·f) = 0
//      BACKWARD in time from V(·,0) = l for a horizon T.
//   4. The zero sublevel set of V(·,-T) is the BACKWARD REACHABLE TUBE:
//      every state from which SOME control reaches the target within T.
//   5. That set IS a safety certificate — exhaustive over all controls,
//      which no finite set of sampled trajectories can ever be.
// Every grid cell updates independently from a 5-point neighborhood each
// sweep — the classic STENCIL pattern (07.09's pattern, with real PDE math
// in the stencil body). That is why the catalog calls this "GPU-perfect".
//
// The plant: a DOUBLE INTEGRATOR — position x, velocity v, acceleration
// command u with |u| <= umax:   xdot = v,  vdot = u.
// The canonical starter system for reachability because its minimum-time-
// to-origin solution is CLOSED FORM (bang-bang, one switch — the famous
// switching curves), so the PDE answer can be checked against mathematics,
// not just against another program (reference_cpu.cpp, THEORY.md §verify).
//
// STATE-SPACE GRID LAYOUT — one float per cell, documented once here:
//     V[j * nx + i]   value at cell (i, j)
//     i in [0, nx)    position index; x_i = xmin + i*dx   (m)      FAST axis
//     j in [0, nv)    velocity index; v_j = vmin + j*dv   (m/s)    slow axis
// i is the fast (contiguous) axis so a warp's 32 consecutive threads read
// 32 consecutive floats — the coalescing rule every grid project in this
// repo follows (07.09 taught it; swapping the axes is THE classic mistake).
// Node-centered grid: cell 0 sits ON xmin, cell nx-1 ON xmax, so
// dx = (xmax - xmin)/(nx - 1).
//
// SIGN CONVENTION for V (level-set standard, used by every file here):
//     V(x,v) <  0  : (x,v) can reach the target within the horizon so far
//     V(x,v) == 0  : the moving front (the set boundary)
//     V(x,v) >  0  : cannot (yet)
//
// All scenario numbers (grid size, domain, umax, target level, horizon)
// come from data/sample/double_integrator_scenario.csv — loaded by main.cu
// and passed through the HjGrid struct below, so the three implementations
// can never disagree about the problem they are solving.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

// ---------------------------------------------------------------------------
// Solver constants — shared verbatim by the GPU solver, the CPU twin, and
// main.cu's step-count computation (one source of truth).
// ---------------------------------------------------------------------------

// CFL safety factor. Explicit LF timestepping is stable only while
//   dt * ( max|v|/dx + umax/dv ) <= 1
// (information may not cross more than one cell per sweep — THEORY.md
// §numerics derives this). 0.5 doubles the sweep count over the limit in
// exchange for a comfortable stability margin and less per-step smearing.
constexpr float kCfl = 0.5f;

// Analytic-verification band half-width, in grid cells: cells whose
// CLOSED-FORM classification changes somewhere within a Chebyshev radius of
// kBandCells are excused from the analytic agreement check — a first-order
// scheme cannot place the front more precisely than a couple of cells, and
// pretending otherwise would just test rounding luck (THEORY.md §verify).
// Outside the band, agreement must be EXACT: every cell, no exceptions.
constexpr int kBandCells = 2;

// GPU-vs-CPU twin tolerance: max |V_gpu - V_cpu| over the whole final field.
// Both paths run the same FP32 arithmetic in the same order per cell; the
// only differences are FMA-contraction choices by the two compilers
// (~1e-7 per op) compounded over the sweeps. Measured worst on the demo
// scenario: ~1e-6; 1e-3 is ~1000x headroom, while an indexing/upwinding/
// layout bug shifts values at order 1 and blows past it instantly.
constexpr float kTwinTol = 1e-3f;

// ---------------------------------------------------------------------------
// HjGrid — the problem definition every solver function receives.
//
// A plain aggregate (no methods, no CUDA types) so the SAME struct compiles
// under nvcc (kernels.cu, main.cu) and cl.exe (reference_cpu.cpp), and can
// be passed to a kernel BY VALUE (it rides in the kernel-argument buffer —
// constant-cache-backed on arrival, ideal for uniform per-launch data).
// ---------------------------------------------------------------------------
struct HjGrid {
    int   nx;      // cells along x (position), >= 2; fast/contiguous axis
    int   nv;      // cells along v (velocity), >= 2; slow axis
    float xmin;    // position of cell i=0 (m)
    float dx;      // cell pitch in x (m); x_i = xmin + i*dx
    float vmin;    // velocity of cell j=0 (m/s)
    float dv;      // cell pitch in v (m/s); v_j = vmin + j*dv
    float umax;    // acceleration bound |u| <= umax (m/s^2), > 0
    float dt;      // sweep timestep (s), CFL-limited (main.cu computes it)
};

// ---------------------------------------------------------------------------
// launch_hj_solve — run n_sweeps explicit backward-time sweeps on the GPU.
//
//   g        : the problem definition (grid geometry + dynamics bound + dt).
//   n_sweeps : number of timesteps (>= 1); n_sweeps * g.dt = the horizon T.
//   d_V      : DEVICE pointer, g.nx * g.nv floats, layout V[j*nx + i].
//              IN:  the initial level function l (target set = {l <= 0}).
//              OUT: the value field after n_sweeps sweeps (result guaranteed
//              in THIS buffer regardless of ping-pong parity).
//
// Launch: one thread per grid cell, 16x16 tiles (grid math + reasoning with
// the kernel in kernels.cu). Allocates the ping-pong partner internally.
// ---------------------------------------------------------------------------
void launch_hj_solve(const HjGrid& g, int n_sweeps, float* d_V);

// ---------------------------------------------------------------------------
// CPU references (reference_cpu.cpp).
// ---------------------------------------------------------------------------

// hj_solve_cpu — the twin of the GPU solver: same update expression, same
// FP32 types, same ping-pong discipline, sequential over cells. main.cu
// runs both from the same initial field and requires max |V_gpu - V_cpu|
// <= kTwinTol (the §5 GPU-vs-CPU gate for this project).
// V is a HOST pointer, g.nx * g.nv floats, in/out like d_V above.
void hj_solve_cpu(const HjGrid& g, int n_sweeps, float* V);

// min_time_to_origin — the ANALYTIC oracle: the closed-form minimum time
// (s) for the double integrator to drive state (x [m], v [m/s]) to the
// origin (0,0) under |u| <= umax [m/s^2]. This is the textbook bang-bang
// solution (one switch on the switching curve x = -v|v|/(2*umax)); it lets
// the demo check the PDE against MATHEMATICS, not just against another
// implementation of the same scheme. Double precision throughout: the
// oracle must be beyond suspicion. Derivation: THEORY.md §the-math.
double min_time_to_origin(double x, double v, double umax);

#endif // PROJECT_KERNELS_CUH
