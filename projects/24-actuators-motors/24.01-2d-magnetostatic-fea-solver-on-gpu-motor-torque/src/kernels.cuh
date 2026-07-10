// ===========================================================================
// kernels.cuh — interface for project 24.01
//               2D magnetostatic FEA solver on GPU -> motor torque-ripple/
//               cogging parameter sweeps
//               (teaching core: batched red-black SOR relaxation of the
//               magnetic vector potential on a structured grid)
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (geometry rasterization, the sweep driver,
// torque post-processing, verification), kernels.cu (the GPU batched
// stencil solver), and reference_cpu.cpp (the CPU twin). Everything all
// three must agree on — the grid layout, the batching layout, and the
// solver's update rule — is defined HERE, once (CLAUDE.md paragraph 12).
//
// SCOPE NOTE (read this before anything else): "FEA" in the catalog bullet
// is used loosely, as it is throughout industry ("2D FEA motor design").
// What is actually implemented is a finite-DIFFERENCE / finite-VOLUME
// discretization of the SAME governing PDE a linear triangular-element FEA
// solver would assemble — a regular Cartesian grid with harmonic-mean
// interface coefficients, which is mathematically a low-order finite-volume
// method. This is the ratified, honestly-labeled teaching discretization
// (see the project's README "Limitations & honesty" and THEORY.md "Where
// this sits in the real world" for the unstructured-mesh FEA comparison —
// production tools like FEMM/Ansys Maxwell/JMAG use triangular/quad meshes
// that conform to the geometry instead of stair-stepping it on a fixed grid).
//
// 2D magnetostatics in five lines (THEORY.md derives it properly):
//   1. Every current and magnet in the cross-section is z-directed (out of
//      the page) or z-directed-equivalent (permanent magnets, via the bound
//      "equivalent magnetizing current" Jm = curl(M)) — so the vector
//      potential collapses to a SCALAR field A_z(x,y).
//   2. B = curl(A_z z-hat) = (dA/dy, -dA/dx, 0) — an in-plane field with a
//      built-in divergence-free guarantee (no monopoles, by construction).
//   3. Ampere's law becomes the elliptic PDE  -div(nu * grad(A_z)) = J_z,
//      nu = 1/mu the RELUCTIVITY — a variable-coefficient Poisson equation,
//      the coefficient jumping wherever iron meets air meets magnet.
//   4. Solve it on a grid: every cell updates from its 4-neighborhood through
//      a 5-point stencil weighted by HARMONIC-averaged face reluctivities —
//      the correct averaging for flux crossing a material interface in
//      series (THEORY.md derives why harmonic, not arithmetic).
//   5. Once A_z is known, B is known everywhere, and torque follows from the
//      Maxwell stress tensor integrated around a circular contour in the
//      air gap (computed in main.cu, a host-side post-processing step).
// Every grid cell updates independently from its 4 face neighbors each
// sweep — the classic STENCIL pattern (07.09/31.01's pattern, with real
// variable-coefficient elliptic PDE math in the stencil body).
//
// THE SWEEP — what makes this a "parameter sweep" project, not just a
// solver demo: the catalog bullet asks for torque-ripple/cogging sweeps.
// Cogging torque is measured at ZERO winding current: only the permanent
// magnets and the slotted stator's varying reluctance interact as the rotor
// turns. We sweep magnet POLE-ARC FRACTION (how much of each pole's angular
// span the magnet actually covers) times ROTOR ANGLE (one electrical/
// mechanical position per solve) and, for each arc fraction, BATCH all
// rotor-angle solves into ONE kernel-launch sequence — B independent grids
// solved simultaneously, one thread per (batch, cell). Memory budget for
// B=24 (see kDefaultSweepAngles below): 24 * 256*256 * 4B * 3 arrays
// (nu, Jsrc, A) ~= 18.9 MiB — trivially fits any current GPU.
//
// BATCHED GRID LAYOUT — one float per (variant, node), documented once here:
//     F[b*ny*nx + j*nx + i]   value at variant b, cell (i, j)
//     i in [0, nx)   x index; x_i = -half_w + i*h   (m)   FAST axis
//     j in [0, ny)   y index; y_j = -half_w + j*h   (m)   slow axis
//     b in [0, B)    batch/variant index (one rotor-angle solve)      SLOWEST axis
// i is the fast (contiguous) axis so a warp's 32 consecutive threads read 32
// consecutive floats within ONE variant's row — the coalescing rule every
// grid project in this repo follows (07.09/31.01 taught it for a single
// grid; here the batch axis rides OUTSIDE the per-variant layout so each
// variant stays exactly as coalesced as the single-grid case).
// Node-centered SQUARE grid: dx = dy = h (a genuine simplification — most
// production FEA meshes are unstructured and graded; THEORY.md discusses
// the consequence for air-gap resolution).
//
// SIGN / UNITS CONVENTIONS used by every file here:
//   A_z         : Wb/m (= T*m), the (single, scalar) component of the
//                 magnetic vector potential A = A_z * z-hat.
//   B = (Bx,By) : Tesla.  Bx = dA_z/dy,  By = -dA_z/dx.
//   nu          : reluctivity, 1/mu, units m/H (H = henry). nu = 1/(mu0*mu_r).
//   J_z         : A/m^2, total z-directed source current density — the sum
//                 of free (winding) current (ZERO throughout the cogging
//                 sweep — cogging is BY DEFINITION a zero-current quantity)
//                 and the permanent magnets' equivalent magnetizing current
//                 Jm = dMy/dx - dMx/dy (computed once per rotor angle in
//                 main.cu, a host-side setup step — see its comments).
//   Torque      : computed in main.cu via the Maxwell stress tensor,
//                 reported in N*m/m — newton-meters PER METER OF AXIAL
//                 STACK LENGTH. This is the honest 2D unit: a real motor's
//                 torque is this number times its actual stack length in
//                 meters (README documents an illustrative scale-up).
//
// All scenario numbers (geometry, materials, sweep parameters) come from
// data/sample/motor_scenario.csv — loaded by main.cu — so the committed
// sample and the solver can never silently disagree about the problem being
// solved (the same discipline as every other flagship's scenario file).
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

// ---------------------------------------------------------------------------
// Fixed grid resolution. The catalog bullet's ratified scope is a 256x256
// grid; it is a compile-time constant (not scenario-loaded) because it
// governs array-shape reasoning throughout this file's comments and the
// batching memory-budget arithmetic above.
// ---------------------------------------------------------------------------
constexpr int kGridN = 256;    // nx = ny = kGridN

// Vacuum permeability, mu0 (H/m) — the one universal physical constant this
// project needs. Every other permeability is expressed as a relative mu_r
// multiplying this (kernels.cuh's material constants live in main.cu's
// MotorScenario, loaded from the CSV, since they are "problem data" the way
// 08.01 treats its plant masses — but mu0 itself is physics, not data, so it
// belongs here, one source of truth for every translation unit).
constexpr float kMu0 = 4.0e-7f * 3.14159265358979323846f;   // H/m (exact, by SI definition pre-2019 redefinition; still the standard engineering value)

// GPU-vs-CPU twin tolerance: max |A_gpu - A_cpu| over the whole verify
// field, in Wb/m. Both paths run the same FP32 update expression in the
// same red/black order; only compiler FMA-contraction differences separate
// them (~1e-7 relative per op, compounded over ~3000 sweep passes). The
// tolerance and the MEASURED worst value are both printed by main.cu, and
// the true "did this catch a bug" evidence is that an indexing, layout, or
// averaging bug shifts the field at order of its own magnitude (~1e-3..1e-2
// Wb/m for this problem), not at the 1e-7-ish level FP32 rounding alone
// produces — see THEORY.md "How we verify correctness" for the measured
// number and the reasoning.
constexpr float kTwinTolAbs = 2.0e-5f;

// ---------------------------------------------------------------------------
// FeaGrid — the problem definition every solver function receives.
//
// A plain aggregate (no methods, no CUDA types) so it compiles under both
// nvcc (kernels.cu, main.cu) and cl.exe (reference_cpu.cpp) and rides in the
// kernel-argument buffer by value (constant-cache-backed on arrival — ideal
// for uniform per-launch data every thread reads, the same reasoning 31.01's
// HjGrid documents).
// ---------------------------------------------------------------------------
struct FeaGrid {
    int   nx;      // cells along x, == kGridN; FAST/contiguous axis
    int   ny;      // cells along y, == kGridN; slow axis
    float h;       // uniform cell pitch (m); dx = dy = h (square grid)
    float omega;   // SOR relaxation factor, 0 < omega < 2 (1 = plain Gauss-Seidel)
};

#ifdef __CUDACC__  // ---- device-aware section: only nvcc sees this ---------

// fea_sor_pass_kernel — one RED or BLACK half-sweep, over ALL B batched
// variants at once. Declared here for completeness (definition + full
// commentary in kernels.cu); main.cu never calls this directly, only the
// host launcher below.
__global__ void fea_sor_pass_kernel(FeaGrid g, int B, int color,
                                    const float* __restrict__ nu,
                                    const float* __restrict__ Jsrc,
                                    float*       __restrict__ A);

#endif // __CUDACC__ --------------------------------------------------------

// ---------------------------------------------------------------------------
// launch_fea_solve_batch — run n_sweeps red+black SOR sweep PAIRS on the GPU,
// across B independent batched variants at once (the "batch of independent
// solves" this project's parameter sweep needs — one kernel-launch sequence
// solves every rotor-angle variant for one arc fraction simultaneously).
//
//   g        : grid geometry (nx, ny, h, omega), shared by every variant —
//              every variant in a batch uses the SAME grid resolution and
//              domain size; only the material/source content differs.
//   B        : number of independent variants (grids) in this batch (>= 1).
//   n_sweeps : fixed number of red+black sweep PAIRS (>= 1). FIXED, not
//              residual-triggered, ON PURPOSE: a data-dependent per-cell or
//              per-variant "converged, stop early" test would force
//              divergent control flow across the batch and break the
//              uniform, lock-step launch this pattern depends on for
//              throughput (THEORY.md "The GPU mapping" explains the trade).
//   d_nu     : DEVICE pointer, B*ny*nx floats — reluctivity per node
//              (m/H), the BATCHED LAYOUT documented in the file header.
//   d_Jsrc   : DEVICE pointer, B*ny*nx floats — total source current
//              density J_z (A/m^2), same layout.
//   d_A      : DEVICE pointer, B*ny*nx floats, IN/OUT. IN: initial guess
//              (main.cu always uploads all-zero — a cold start; a warm
//              start from a neighboring rotor angle is README Exercise 4).
//              OUT: the solved vector potential field, same layout.
//              Dirichlet A=0 on every variant's outer border is enforced
//              IMPLICITLY: border nodes are never touched by the kernel, so
//              whatever the caller uploaded there (main.cu uploads zero)
//              persists — the "flux never crosses this boundary" condition
//              THEORY.md derives.
//
// Launch: one thread per (batch, cell), 16x16 tiles per variant, B tiles
// deep (blockIdx.z = variant) — grid math + reasoning live with the kernel
// in kernels.cu.
// ---------------------------------------------------------------------------
void launch_fea_solve_batch(const FeaGrid& g, int B, int n_sweeps,
                            const float* d_nu, const float* d_Jsrc, float* d_A);

// ---------------------------------------------------------------------------
// CPU reference (reference_cpu.cpp).
// ---------------------------------------------------------------------------

// fea_solve_batch_cpu — the oracle twin of the GPU solver: identical FP32
// update expression, identical red/black order, sequential over (b, j, i)
// instead of parallel. main.cu runs it against the GPU on one representative
// motor variant and requires agreement within kTwinTolAbs (the paragraph 5
// GPU-vs-CPU gate for this project). nu/Jsrc/A are HOST pointers, same
// batched layout as the device buffers above.
void fea_solve_batch_cpu(const FeaGrid& g, int B, int n_sweeps,
                         const float* nu, const float* Jsrc, float* A);

#endif // PROJECT_KERNELS_CUH
