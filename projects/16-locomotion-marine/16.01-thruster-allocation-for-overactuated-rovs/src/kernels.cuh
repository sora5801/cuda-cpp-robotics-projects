// ===========================================================================
// kernels.cuh — interface for project 16.01
//               Thruster allocation for overactuated ROVs (batched QP)
//
// Role in the project
// -------------------
// The CONTRACT shared by all three translation units:
//   * main.cu           — the driver: builds the allocation matrix once at
//                          startup, loads the wrench batch, calls both paths,
//                          runs the optimality/KKT gates, writes artifacts.
//   * kernels.cu         — the GPU projected-gradient-descent (PGD) kernel.
//   * reference_cpu.cpp  — the CPU twin of the kernel, PLUS the one-time
//                          host-side setup math (build B, form the QP, power
//                          iteration for the step size) and the closed-form
//                          pseudoinverse oracle used by the optimality gate.
// Every physical constant, every layout, and the QP itself is defined ONCE,
// here (CLAUDE.md §12: state layouts and units are single-sourced).
//
// THE PROBLEM, IN ONE PARAGRAPH (THEORY.md derives all of this properly)
// ------------------------------------------------------------------------
// An ROV has 8 thrusters but only 6 degrees of freedom (DOF) to control —
// it is OVERACTUATED, on purpose, for fault tolerance and station-keeping
// authority (THEORY.md "the problem"). Each thruster i sits at body-frame
// position r_i and pushes along a fixed unit direction d_i; thruster force
// u_i (signed, thrusters run both directions) produces a body WRENCH
// (force + moment) [d_i; r_i x d_i] * u_i. Stacking all 8 thrusters gives
// the 6x8 ALLOCATION MATRIX B such that the body wrench from a force vector
// u in R^8 is tau = B*u. Given a COMMANDED wrench tau_cmd (from an upstream
// controller — e.g. 16.09's docking-under-current MPPI, or a station-keeping
// PID), allocation asks: which u produces (approximately) tau_cmd, subject
// to each thruster's saturation limit |u_i| <= u_max_i? That is the QP:
//
//     minimize_u   || W (B u - tau_cmd) ||^2   +   eps * ||u||^2
//     subject to   -u_max_i <= u_i <= u_max_i     for i = 0..7
//
// W weights which wrench components matter most (identity here — README
// "Limitations"); eps is a small damping term that (a) keeps the problem
// well-posed even along B's null-space directions (8 unknowns, 6 equations
// -> a 2-dimensional space of "internal squeeze" force combinations that
// produce ZERO net wrench) and (b) is exactly what makes the unconstrained
// optimum a closed-form DAMPED WEIGHTED PSEUDOINVERSE — the ground truth the
// README §Expected-output "pseudoinverse gate" checks the QP against.
//
// BODY FRAME — A DELIBERATE DEVIATION FROM THE REPO DEFAULT (CLAUDE.md §12
// permits this: "x-forward/y-left/z-up UNLESS a domain standard says
// otherwise (state it)"). Marine robotics almost universally uses the SNAME/
// Fossen convention instead: x-FORWARD, y-STARBOARD (right), z-DOWN. We use
// that here because (a) it is what every marine-robotics textbook and ROV
// manual this project points to (README "Prior art") actually uses, and
// (b) "z down = toward the seabed" is the intuitive frame for a vehicle that
// spends its life underwater. Concretely: a THRUSTER pushing the vehicle
// UP (toward the surface) has direction d_z = -1 in this frame. Units:
// meters, Newtons, Newton-meters throughout (SI, as always).
//
// THRUSTER LAYOUT — the standard "vectored-octo" ROV configuration (the
// topology popularized by BlueROV2-class vehicles; our exact numbers are
// this project's own synthetic geometry, not reverse-engineered from any
// vendor's CAD): 4 HORIZONTAL thrusters mounted in an "X" pattern at +-45
// degrees to the body x-axis, all at the vehicle's CG depth (z=0) — they
// span surge/sway/yaw (Fx, Fy, Mz) and, being co-planar with the CG, add
// (by construction) ZERO roll/pitch moment. 4 VERTICAL thrusters mounted at
// the four corners, each pushing straight up/down — they span heave/roll/
// pitch (Fz, Mx, My) and (being purely vertical) add ZERO yaw moment. The
// result is a B matrix that is (nearly) BLOCK-DIAGONAL: columns 0-3 only
// populate rows Fx/Fy/Mz, columns 4-7 only populate rows Fz/Mx/My. Real
// hulls have small cross-coupling from CG offset and hydrodynamic
// interaction; the QP does not care — it works for ANY full-row-rank B
// (THEORY.md "numerical considerations" discusses the block structure and
// its conditioning benefit).
//
//     index  name  role         position r (m)            direction d (unit)
//       0     H1   horiz. FS    (+kHx, +kHy, 0)            ( c45, -c45, 0)
//       1     H2   horiz. FP    (+kHx, -kHy, 0)            ( c45, +c45, 0)
//       2     H3   horiz. AS    (-kHx, +kHy, 0)            ( c45, +c45, 0)
//       3     H4   horiz. AP    (-kHx, -kHy, 0)            ( c45, -c45, 0)
//       4     V1   vert.  FS    (+kVx, +kVy, 0)            (   0,    0,-1)
//       5     V2   vert.  FP    (+kVx, -kVy, 0)            (   0,    0,-1)
//       6     V3   vert.  AS    (-kVx, +kVy, 0)            (   0,    0,-1)
//       7     V4   vert.  AP    (-kVx, -kVy, 0)            (   0,    0,-1)
//   (F/A = fore/aft, S/P = starboard/port; c45 = cos(45 deg) = sin(45 deg))
//
// WRENCH LAYOUT — float tau[6], SI units, documented once here:
//     tau[0] = Fx  surge force   (N)      tau[3] = Mx  roll  moment (N*m)
//     tau[1] = Fy  sway force    (N)      tau[4] = My  pitch moment (N*m)
//     tau[2] = Fz  heave force   (N)      tau[5] = Mz  yaw   moment (N*m)
//
// FORCE LAYOUT — float u[8], one signed force per thruster (N), indexed as
// the table above (H1..H4 then V1..V4); |u_i| <= u_max_i.
//
// ALLOCATION-MATRIX LAYOUT — B is stored ROW-MAJOR, 6 rows x 8 columns:
// element (row, col) lives at B[row*8 + col]. The QP's dense 8x8 Hessian
// H = 2*(B^T W^2 B + eps*I) and the 8x6 matrix BtW2 = B^T W^2 are stored the
// same way: H[row*8+col], BtW2[row*6+col]. These three matrices are
// IDENTICAL for every problem in a batch (they depend only on the vehicle's
// geometry, not on the commanded wrench) — computed ONCE at startup on the
// host (build_allocation_matrix / build_qp_matrices below) and then either
// passed to the CPU oracle or uploaded to GPU __constant__ memory
// (kernels.cu) for a broadcast-cheap read by every thread in the batch.
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

// ---------------------------------------------------------------------------
// Vehicle geometry and thruster limits (SI units; the "plant" this project
// allocates for — shared verbatim by every path, exactly as 08.01 shares its
// cart-pole constants). Values are this project's own synthetic ROV, sized
// like a small/medium observation-class vehicle (hull roughly 0.4 x 0.3 m).
// ---------------------------------------------------------------------------
constexpr int kNThr = 8;      // number of thrusters
constexpr int kNDof = 6;      // body wrench dimension (Fx,Fy,Fz,Mx,My,Mz)

constexpr float kHx = 0.20f;  // horizontal-thruster fore/aft offset from CG (m)
constexpr float kHy = 0.15f;  // horizontal-thruster port/starboard offset (m)
constexpr float kVx = 0.15f;  // vertical-thruster fore/aft offset from CG (m)
constexpr float kVy = 0.20f;  // vertical-thruster port/starboard offset (m)

constexpr float kUMaxNominal = 40.0f; // per-thruster saturation limit (N),
                                      // symmetric +-, e.g. a T200-class
                                      // thruster's illustrative bollard
                                      // thrust at mid-voltage (PRACTICE.md
                                      // §2 has the real-part story; real
                                      // thrusters are FORWARD/REVERSE
                                      // asymmetric — README "Limitations").

// Thruster positions r_i (m) and unit directions d_i, laid out [8][3],
// row i = thruster i, columns = (x,y,z) in the body frame documented above.
// c45 = cos(45 deg) = sin(45 deg) = sqrt(2)/2, spelled out (not sqrtf) so
// this table is a plain compile-time constant usable from host AND device
// code without a device math-library call.
constexpr float kC45 = 0.70710678f;

constexpr float kThrusterPos[8][3] = {
    { +kHx, +kHy, 0.0f },   // H1 — horizontal, fore-starboard
    { +kHx, -kHy, 0.0f },   // H2 — horizontal, fore-port
    { -kHx, +kHy, 0.0f },   // H3 — horizontal, aft-starboard
    { -kHx, -kHy, 0.0f },   // H4 — horizontal, aft-port
    { +kVx, +kVy, 0.0f },   // V1 — vertical, fore-starboard corner
    { +kVx, -kVy, 0.0f },   // V2 — vertical, fore-port corner
    { -kVx, +kVy, 0.0f },   // V3 — vertical, aft-starboard corner
    { -kVx, -kVy, 0.0f },   // V4 — vertical, aft-port corner
};
constexpr float kThrusterDir[8][3] = {
    { kC45, -kC45, 0.0f },  // H1 — "X" vectoring: points fore-and-to-port
    { kC45, +kC45, 0.0f },  // H2 — points fore-and-to-starboard
    { kC45, +kC45, 0.0f },  // H3 — points fore-and-to-starboard
    { kC45, -kC45, 0.0f },  // H4 — points fore-and-to-port
    { 0.0f, 0.0f, -1.0f },  // V1 — straight up (z-down frame: -z is "up")
    { 0.0f, 0.0f, -1.0f },  // V2
    { 0.0f, 0.0f, -1.0f },  // V3
    { 0.0f, 0.0f, -1.0f },  // V4
};

// ---------------------------------------------------------------------------
// QP hyperparameters (THEORY.md "the math" derives the eps/step trade-off;
// THEORY.md "GPU mapping" derives the iteration-count/conditioning story).
// ---------------------------------------------------------------------------
// W: diagonal wrench-tracking weights (unitless scale factors), one per
// wrench component in the kWrench layout above. UNIFORM here — the simplest
// defensible choice (README "Limitations"; retuning is README Exercise 2).
constexpr float kWeight[6] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };

constexpr float kEpsReg = 0.10f;   // regularization eps (see header comment
                                   // above: keeps H = 2(B^T W^2 B + eps*I)
                                   // strongly convex/well-conditioned along
                                   // B's null space; THEORY.md quantifies
                                   // the resulting condition number, ~41).
constexpr int   kPgdIters = 500;   // FIXED projected-gradient iteration count
                                   // (no early exit — every thread in a warp
                                   // does the SAME number of iterations, so
                                   // there is no iteration-count divergence;
                                   // THEORY.md "GPU mapping" explains the
                                   // trade against early-exit convergence
                                   // checks). 500 is ~2x the iterations a
                                   // double-precision reference needs to hit
                                   // machine zero for this eps — generous
                                   // headroom for FP32 (measured in
                                   // THEORY.md "numerical considerations").
constexpr int   kPowerIters = 100; // host power-iteration steps used ONCE at
                                   // startup to estimate lambda_max(H) (see
                                   // power_iteration_lambda_max below).

// A single "motivating example" commanded wrench — the exact case
// THEORY.md's "why not just clip the pseudoinverse" worked example uses,
// re-used at RUNTIME as the monotonicity/KKT trace sample (main.cu "Stage
// 2c"). Keeping the prose and the code pointed at the same numbers is the
// whole point: a reader can reproduce the THEORY.md story by running the
// demo. Values: a demanding combined surge+yaw docking correction that
// saturates two of the four horizontal thrusters (THEORY.md derives why).
constexpr float kMotivatingWrench[6] = { -18.33f, 1.91f, 0.0f, 0.0f, 0.0f, -62.99f };

// ---------------------------------------------------------------------------
// launch_thruster_allocation — solve a BATCH of `count` independent
// box-constrained QPs on the GPU, one thread per problem.
//
//   count   : number of allocation problems (>= 0; 0 is a valid no-op).
//   d_tau   : DEVICE pointer, count*kNDof floats — commanded wrenches,
//             kWrench layout above, row-major (problem k's wrench at
//             d_tau[k*6 .. k*6+5]). Never written.
//   d_umax  : DEVICE pointer, count*kNThr floats — PER-PROBLEM, per-thruster
//             saturation limits (N), row-major (problem k's limits at
//             d_umax[k*8 .. k*8+7]). Per-problem (not a single global
//             u_max) is what lets the failure-analysis stage (main.cu) model
//             a LOCKED-OUT thruster by setting that one column to 0 for a
//             whole re-allocated batch, without touching the kernel.
//   d_u_out : DEVICE pointer, count*kNThr floats, OUT — solved thruster
//             forces (N), same layout as d_umax.
//   step    : the projected-gradient step size 1/L (see kernels.cuh header
//             and power_iteration_lambda_max), computed ONCE on the host at
//             startup and passed in — a plain kernel argument, not a
//             __constant__, because it is a single scalar (no benefit to
//             broadcasting it any other way).
//
// Prerequisite: upload_allocation_constants() (below) must have been called
// at least once before this launch — it is what fills the __constant__
// arrays c_H / c_BtW2 the kernel reads (kernels.cu owns their definition).
//
// Launch: one thread per QP, 256-thread blocks (grid math + reasoning live
// with the kernel in kernels.cu, matching the repo default geometry).
// ---------------------------------------------------------------------------
void upload_allocation_constants(const float* H, const float* BtW2);

void launch_thruster_allocation(int count,
                                const float* d_tau, const float* d_umax,
                                float* d_u_out, float step);

// ---------------------------------------------------------------------------
// Host-side setup math (defined in reference_cpu.cpp — plain C++, no CUDA;
// this is "the CPU's problem to solve once," not a competing implementation
// of the kernel, so it lives beside the CPU oracle rather than in kernels.cu;
// 08.01 sets the same precedent with cartpole_step_cpu doubling as "the
// plant"). main.cu calls these ONCE at startup; both the GPU launcher and
// the CPU oracle below then allocate against the SAME H/BtW2/step.
// ---------------------------------------------------------------------------

// build_allocation_matrix — compute B (row-major, kNDof x kNThr) from the
// kThrusterPos/kThrusterDir tables above: row 0-2 = d_i (the force
// contribution), row 3-5 = r_i x d_i (the moment contribution). Pure
// geometry, O(kNThr) cross products, run once.
void build_allocation_matrix(float* B /* [kNDof*kNThr] OUT */);

// build_qp_matrices — form the QP's dense Hessian H = 2*(B^T W^2 B + eps*I)
// (row-major kNThr x kNThr) and BtW2 = B^T W^2 (row-major kNThr x kNDof, so
// that g = 2*BtW2*tau is the QP's linear term for a given wrench). Also
// returns Q = B^T W^2 B + eps*I (row-major kNThr x kNThr, SPD) — the matrix
// the pseudoinverse-gate Cholesky solve (below) actually factors.
void build_qp_matrices(const float* B /* [kNDof*kNThr] */,
                       const float* W /* [kNDof] diagonal weights */,
                       float eps,
                       float* H   /* [kNThr*kNThr] OUT: 2*(B^T W^2 B + eps I) */,
                       float* BtW2/* [kNThr*kNDof] OUT: B^T W^2 */,
                       float* Q   /* [kNThr*kNThr] OUT: B^T W^2 B + eps I */);

// power_iteration_lambda_max — estimate the largest eigenvalue of a SYMMETRIC
// n x n matrix M (row-major) via `iters` steps of the classic power method,
// starting from a FIXED, documented vector (all-ones, normalized) so the
// result is bit-reproducible. This IS the Lipschitz constant L of the QP's
// gradient (THEORY.md "the math" derives why); main.cu calls this once with
// M = H to get L, then sets step = 1/L.
float power_iteration_lambda_max(const float* M, int n, int iters);

// cholesky_solve_spd — solve Q x = b for a SYMMETRIC POSITIVE DEFINITE Q
// (n x n, row-major) via in-place Cholesky factorization + forward/back
// substitution — the exact n=8 case of 33.01's batched algorithm, here
// solved ONCE (not batched) as the "ground truth" the optimality gate
// (main.cu Stage 2b) compares the QP's UNSATURATED solutions against: the
// closed-form damped weighted pseudoinverse x* = Q^-1 (B^T W^2 tau).
void cholesky_solve_spd(const float* Q /* [n*n] */, const float* b /* [n] */,
                        float* x /* [n] OUT */, int n);

// ---------------------------------------------------------------------------
// CPU references (reference_cpu.cpp — the correctness oracle).
// ---------------------------------------------------------------------------

// thruster_allocate_cpu — the oracle twin of the GPU kernel: same H/BtW2,
// same fixed-iteration projected-gradient loop, sequential over the batch.
// main.cu runs this against the GPU kernel on the SAME inputs and requires
// element-wise agreement (the §5 GPU-vs-CPU gate for this project).
void thruster_allocate_cpu(int count,
                           const float* tau, const float* umax,
                           const float* H, const float* BtW2, float step,
                           int iters, float* u_out);

// thruster_allocate_trace_cpu — identical algorithm, but for a SINGLE
// problem, and it additionally logs the QP objective J(u_k) = ||W(Bu_k-tau)||^2
// + eps*||u_k||^2 (and the raw wrench residual ||Bu_k - tau||) at every
// iteration into the caller's arrays. Used only by main.cu's Stage 2c
// (monotonicity gate) on kMotivatingWrench — never in the hot batch path,
// which is why it is a separate, simpler function rather than an option
// flag on thruster_allocate_cpu.
void thruster_allocate_trace_cpu(const float* tau, const float* umax,
                                 const float* B, const float* W,
                                 const float* H, const float* BtW2,
                                 float eps, float step, int iters,
                                 float* J_trace /* [iters+1] OUT */,
                                 float* residual_trace /* [iters+1] OUT */);

#endif // PROJECT_KERNELS_CUH
