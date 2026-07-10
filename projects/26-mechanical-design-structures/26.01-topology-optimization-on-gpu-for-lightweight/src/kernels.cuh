// ===========================================================================
// kernels.cuh — interface for project 26.01
//               Topology optimization (SIMP) on GPU for lightweight links and
//               brackets — flagship design project
//               (teaching core: matrix-free preconditioned-CG FEA solve +
//               SIMP compliance minimization via Optimality Criteria)
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (scenario loading, the SIMP outer loop,
// verification, artifacts), kernels.cu (the GPU matrix-free CG solver and
// the sensitivity/filter kernels), and reference_cpu.cpp (the CPU twin used
// for the §5 GPU-vs-CPU gate, and the KE_hat element-stiffness derivation
// shared by both paths). Everything all three must agree on — the mesh
// layout, the DOF numbering, the element stiffness matrix, and the filter
// weights — is defined HERE, once (CLAUDE.md §12).
//
// SCOPE NOTE (read before anything else): the catalog bullet offers
// "SIMP/level-set". This project implements the SIMP (Solid Isotropic
// Material with Penalization) compliance-minimization pipeline — the
// "99-line/88-line topopt" lineage (Sigmund 2001; Andreassen et al. 2011),
// GPU-ported and taught properly. Level-set topology optimization is the
// documented alternative (THEORY.md "Where this sits in the real world");
// it is not implemented here, per the ratified scope.
//
// SIMP topology optimization in six lines (THEORY.md derives it properly):
//   1. Design domain = a structured grid of bilinear (Q4) plane-stress
//      elements; each element owns one density rho_e in [0,1].
//   2. SIMP interpolates stiffness: E(rho_e) = Emin + rho_e^3 (E0 - Emin) —
//      the cube PENALIZES intermediate densities so the optimizer prefers
//      pure solid (rho=1) or pure void (rho=0), not a physically-meaningless
//      "50% material" (THEORY.md derives why p=3 works).
//   3. For the current density field, SOLVE the linear elasticity FEA
//      problem K(rho) U = F for the displacement field U — this project's
//      GPU workhorse: a MATRIX-FREE, element-by-element, node-GATHER
//      preconditioned conjugate-gradient solve (no global stiffness matrix
//      is ever assembled — see "The GPU mapping" below and in THEORY.md).
//   4. Compute compliance c = F^T U = sum_e E(rho_e) u_e^T KE_hat u_e (a
//      *stiffness*-per-unit-density SENSITIVITY dc/drho_e falls out of the
//      same per-element quadratic form — one more small GPU kernel).
//   5. FILTER the sensitivities over a small radius (Sigmund's classic
//      density-weighted filter) — the single fix for the checkerboard
//      pathology and the reason topology-optimized designs are
//      mesh-independent (THEORY.md tells this story in full; try Exercise 2
//      with the filter disabled to see the pathology yourself).
//   6. UPDATE every element's density with the Optimality Criteria (OC)
//      rule, bisecting a Lagrange multiplier so the volume constraint holds
//      exactly (host, small, cheap — THEORY.md derives the KKT condition
//      this heuristic satisfies). Repeat 3-6 until the design stops moving.
// Step 3 (the CG solve) is >95% of the arithmetic per outer iteration and is
// where every GPU idea in this project lives; steps 4-5 are small element-
// parallel kernels; step 6 is O(elements) host bookkeeping, deliberately
// kept off the GPU (same "keep the whole algorithm on one screen" choice
// 08.01 makes for its softmin blend).
//
// MESH & DOF LAYOUT — the single source of truth every file honors:
//     nelx, nely     : elements along x (FAST axis), y (SLOW axis)
//     nx = nelx+1    : nodes along x
//     ny = nely+1    : nodes along y
//     node id        n = j*nx + i,           i in [0,nx), j in [0,ny)
//     element id     e = ey*nelx + ex,       ex in [0,nelx), ey in [0,nely)
//     dof (2 per node, x then y)  dof_x = 2*n, dof_y = 2*n+1
//     ndof = 2*nx*ny
// Element (ex,ey) occupies the unit cell [ex,ex+1] x [ey,ey+1] in a
// NONDIMENSIONAL element-index coordinate system; its 4 corner nodes, in the
// CCW order the element stiffness matrix KE_hat below assumes, are:
//     local 0 = node(ex,   ey)      (this element's "min-x,min-y" corner)
//     local 1 = node(ex+1, ey)
//     local 2 = node(ex+1, ey+1)
//     local 3 = node(ex,   ey+1)
// i (x) is the FAST/contiguous axis for BOTH nodes and elements — every
// grid-stencil project in this repo (07.09/24.01/31.01) makes this same
// choice so a warp's 32 threads touch 32 consecutive floats.
//
// WHY ELEMENT SIZE h NEVER APPEARS IN THE FEA MATH (a genuine, checkable
// derivation, not a simplification swept under the rug — THEORY.md "The
// math" proves it): for a SQUARE Q4 plane-stress element, mapping the
// [-1,1]^2 parent domain to a physical h x h square contributes a Jacobian
// determinant h^2/4 to the area integral and a factor (2/h)^2 to every
// shape-function derivative inside B — the h^2 cancels the 1/h^2 EXACTLY.
// The element stiffness matrix for a UNIFORM SQUARE mesh therefore depends
// only on Young's modulus E, thickness t, and Poisson's ratio nu — never on
// h. That is why kernels.cuh's element stiffness constant is named
// "KE_hat": the DIMENSIONLESS unit-E, unit-t, unit-square stiffness matrix,
// and every element's real stiffness is exactly E(rho_e) * KE_hat (t=1,
// i.e. "per meter of out-of-plane thickness" — the same honest 2D
// convention 24.01 uses for torque). Physical domain size (README/PRACTICE)
// is therefore a LABEL, not an input the solver needs.
//
// UNITS (SI throughout, CLAUDE.md §12):
//   rho        : dimensionless density in [0,1] (SIMP design variable).
//   E, E0, Emin: Pa (N/m^2). Emin is a small floor, never exactly 0 — a
//                truly zero-stiffness element would make K(rho) singular.
//   U          : node displacement (m), 2 components per node (ux, uy).
//   F          : node force (N), 2 components per node.
//   c          : compliance = F^T U, Joules (N*m) — "how much the structure
//                deflects under its design load"; minimizing it (at fixed
//                material budget) is this project's whole objective.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cstdint>

// ---------------------------------------------------------------------------
// HOSTDEV — the standard trick (used throughout this repo, e.g. 24.01's
// FeaGrid) for a header shared between nvcc (kernels.cu, main.cu) and cl.exe
// (reference_cpu.cpp): __CUDACC__ is defined ONLY when nvcc is compiling, so
// plain cl.exe never sees the CUDA-only __host__ __device__ qualifiers.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HOSTDEV __host__ __device__
#else
#define HOSTDEV
#endif

// ---------------------------------------------------------------------------
// Fixed algorithm/physics constants — shared verbatim by every file so
// "the same SIMP problem" never silently drifts between the GPU path, the
// CPU oracle, and the host-side OC update.
// ---------------------------------------------------------------------------
constexpr float kPoissonNu = 0.3f;   // Poisson's ratio (dimensionless). 0.3 is the
                                     // generic structural-metal value used throughout
                                     // the classic 99-line/88-line topopt papers this
                                     // project teaches toward — using it here lets a
                                     // learner cross-check KE_hat against the literature
                                     // (real Al 6061 is closer to 0.33; README/PRACTICE
                                     // notes the difference honestly).
// SIMP penalization exponent p=3 is FIXED by the ratified scope (not a tunable
// scenario field): p=3 is the textbook value that makes intermediate
// densities structurally unattractive without making the optimization
// problem numerically pathological (THEORY.md derives why p<1 fails to
// penalize and large p causes convergence stalls). Because p is fixed, every
// E(rho)/dE(rho) call below hand-multiplies rho*rho*rho instead of calling
// powf(rho,3)/pow(rho,3) — one fewer transcendental function AND one fewer
// possible source of GPU-vs-CPU divergence (nvcc's powf and MSVC's pow are
// not guaranteed bit-identical; a hand-rolled cube is, up to FP rounding,
// which both paths do identically since it is pure multiplication).

// CG solver defaults (README documents the measured behavior these produce).
constexpr int   kMaxCgIters = 400;    // hard cap per FEA solve — the matrix-free
                                      // gather kernel is cheap enough that even a
                                      // "wasted" full-cap solve stays fast; the cap
                                      // exists so a pathological density field can
                                      // never hang the demo (THEORY.md "numerics").
constexpr float kCgRelTol   = 1.0e-3f; // stop when ||r|| / ||F|| falls below this —
                                       // adequate for SIMP's OUTER loop (the design
                                       // only needs approximately-converged
                                       // sensitivities each iteration; over-solving
                                       // early iterations wastes GPU time on a design
                                       // that is about to change anyway).

// Sensitivity filter radius (elements) — "~2.4" per the catalog bullet's
// mesh-independence lesson (Sigmund 1997/2001; THEORY.md tells the
// checkerboard story this filter fixes).
constexpr float kFilterRMin = 2.4f;
constexpr int   kFilterR    = 2;   // integer half-window: any offset with
                                   // |di|>2 or |dj|>2 has distance > 2.4 by
                                   // construction (min possible dist at |di|=3
                                   // is 3.0 > 2.4), so a 5x5 window is exact,
                                   // not an approximation of the true support.

// GPU-vs-CPU twin tolerances (documented + measured values live in
// README "Expected output" and THEORY "How we verify correctness"; see the
// header comment there for why a CAPPED-ITERATION float CG needs a
// residual-level tolerance rather than solution bit-equality).
constexpr float kTwinRelTolCompliance = 5.0e-3f;   // relative agreement on total compliance
constexpr float kTwinRelTolDisp       = 1.0e-2f;   // relative agreement, worst single DOF

// ---------------------------------------------------------------------------
// TopoGrid — the mesh definition every solver/kernel function receives.
// Plain aggregate (CLAUDE.md's usual reasoning, see 24.01's FeaGrid): rides
// in the kernel argument buffer by value (constant-cache-backed on arrival —
// every thread in a launch reads the SAME nelx/nely/nx/ny, a broadcast read).
// ---------------------------------------------------------------------------
struct TopoGrid {
    int nelx, nely;   // elements along x (fast), y (slow)
    int nx, ny;       // nodes = nelx+1, nely+1 (stored, not recomputed, so
                      // every kernel reads it once instead of an extra add)
};

// node_id / elem_id — the ONE place the flat-index arithmetic lives (the
// layout contract stated in the file header). HOSTDEV so main.cu (host),
// kernels.cu (device), and reference_cpu.cpp (host, via cl.exe) all call the
// exact same expression — eliminates an entire class of "the CPU oracle
// indexed it differently" bugs by construction.
inline HOSTDEV int node_id(const TopoGrid& g, int i, int j) { return j * g.nx + i; }
inline HOSTDEV int elem_id(const TopoGrid& g, int ex, int ey) { return ey * g.nelx + ex; }

// E_of_rho / dE_drho — the SIMP interpolation and its derivative (needed by
// the sensitivity kernel: dc/drho_e = dE/drho_e * u_e^T KE_hat u_e — the
// chain rule applied to c = sum_e E(rho_e) u_e^T KE_hat u_e, derived in
// THEORY.md "The math"). E0/Emin are Pa; both HOSTDEV for the same reason
// as node_id/elem_id above.
inline HOSTDEV float E_of_rho(float rho, float E0, float Emin)
{
    return Emin + rho * rho * rho * (E0 - Emin);   // p=3, hand-multiplied — see the constants comment above
}
inline HOSTDEV float dE_drho(float rho, float E0, float Emin)
{
    return 3.0f * rho * rho * (E0 - Emin);   // d/drho [Emin + rho^3 (E0-Emin)]
}

#ifdef __CUDACC__  // ---- device-aware section: only nvcc sees this ---------

// The element stiffness constant KE_hat (8x8, row-major, E=1 Pa, t=1 m,
// nu=kPoissonNu, unit square) lives in __constant__ memory: EVERY thread in
// EVERY kernel below reads the SAME 64 floats every launch — the textbook
// __constant__ use case (broadcast to all threads via the constant cache,
// the same reasoning 09.01/08.01 document for their own per-launch-uniform
// data). Populated once via upload_KE_hat() before the first solve.
//
// NOTE ON "extern __constant__": CUDA's __constant__ variables are NOT
// externally linkable across translation units without relocatable device
// code (-rdc=true, which this project's .vcxproj does not enable — see the
// CodeGeneration comment there). This project has exactly one .cu file that
// launches kernels (kernels.cu), so the symbols are declared AND defined
// there, privately (static storage is the default for __constant__); no
// other translation unit needs to name them directly — main.cu only ever
// calls upload_KE_hat()/upload_filter_weights() below, never the symbols.

// ---- device kernels (defined + fully commented in kernels.cu) -------------
__global__ void matvec_gather_kernel(TopoGrid g,
                                     const float* __restrict__ rho,
                                     const float* __restrict__ x,
                                     const uint8_t* __restrict__ fixed,
                                     float E0, float Emin,
                                     float* __restrict__ y);

__global__ void diag_gather_kernel(TopoGrid g,
                                   const float* __restrict__ rho,
                                   const uint8_t* __restrict__ fixed,
                                   float E0, float Emin,
                                   float* __restrict__ diag);

__global__ void elem_sensitivity_kernel(TopoGrid g,
                                        const float* __restrict__ rho,
                                        const float* __restrict__ U,
                                        float E0, float Emin,
                                        float* __restrict__ ce,
                                        float* __restrict__ dc_raw);

__global__ void density_filter_kernel(TopoGrid g,
                                      const float* __restrict__ rho,
                                      const float* __restrict__ dc_raw,
                                      float* __restrict__ dc_filt);

__global__ void vec_copy_kernel(int n, const float* __restrict__ src, float* __restrict__ dst);
__global__ void vec_combine_kernel(int n, float a, const float* __restrict__ x,
                                   float b, float* __restrict__ y);   // y = a*x + b*y
__global__ void vec_divide_kernel(int n, const float* __restrict__ num,
                                  const float* __restrict__ den, float* __restrict__ out);
__global__ void reduce_dot_kernel(int n, const float* __restrict__ a,
                                  const float* __restrict__ b, float* __restrict__ partial);

#endif // __CUDACC__ --------------------------------------------------------

// ---------------------------------------------------------------------------
// upload_KE_hat / upload_filter_weights — one-time host->constant-memory
// uploads (kernels.cu owns the __constant__ symbols; these are the only
// functions allowed to write them, keeping "who initializes the constants"
// to one obvious place per CLAUDE.md §12).
// ---------------------------------------------------------------------------
void upload_KE_hat(const float KE_hat[64]);
void upload_filter_weights(const float weights[(2 * kFilterR + 1) * (2 * kFilterR + 1)]);

// ---------------------------------------------------------------------------
// launch_topo_cg_solve — the project's GPU workhorse: solve K(rho) U = F for
// U via matrix-free, Jacobi-preconditioned conjugate gradient, entirely
// through the gather kernels above (no global stiffness matrix is ever
// assembled — THEORY.md "The GPU mapping" explains why assembly-free beats
// an assembled sparse solve on a GPU at this problem size).
//
//   g        : mesh (nelx, nely, nx, ny).
//   d_rho    : DEVICE [nelx*nely] current per-element density (IN, read-only).
//   d_F      : DEVICE [ndof] applied nodal force (N), zero at fixed dofs.
//   d_fixed  : DEVICE [ndof] 1 = Dirichlet-fixed dof (U forced to 0 by
//              construction — THEORY.md "numerics" explains how the CG
//              recursion enforces this WITHOUT ever modifying K), 0 = free.
//   d_U      : DEVICE [ndof] IN/OUT. IN: the WARM-START guess (main.cu
//              passes the previous outer iteration's solution — a
//              deliberate performance choice, not just a convenience:
//              consecutive SIMP iterations change rho only slightly once
//              the design stabilizes, so warm-starting collapses CG to a
//              handful of iterations late in the run; THEORY.md quantifies
//              the measured effect). OUT: the solved displacement field (m).
//   E0, Emin : material Young's modulus and SIMP floor (Pa).
//   out_iters, out_rel_resid : OPTIONAL (may be null) diagnostics: iterations
//              actually run and the final ||r||/||F|| achieved — main.cu
//              logs these for the VERIFY stage and for honesty about
//              whether the cap or the tolerance ended the solve.
//
// Scratch buffers (diag, r, z, p, Kp, dot-reduction partials) are allocated
// and freed INSIDE this call — see kernels.cu's comment on why that
// overhead is negligible next to the CG iterations themselves.
// ---------------------------------------------------------------------------
void launch_topo_cg_solve(const TopoGrid& g, const float* d_rho, const float* d_F,
                          const uint8_t* d_fixed, float* d_U, float E0, float Emin,
                          int max_iters, float rel_tol,
                          int* out_iters, float* out_rel_resid);

// launch_elem_sensitivity — host wrapper for elem_sensitivity_kernel (grid
// math + launch-error check, the repo's standard launcher shape).
void launch_elem_sensitivity(const TopoGrid& g, const float* d_rho, const float* d_U,
                             float E0, float Emin, float* d_ce, float* d_dc_raw);

// launch_density_filter — host wrapper for density_filter_kernel.
void launch_density_filter(const TopoGrid& g, const float* d_rho,
                           const float* d_dc_raw, float* d_dc_filt);

// ---------------------------------------------------------------------------
// CPU references (reference_cpu.cpp) — plain, sequential twins of every GPU
// stage above, used by main.cu's VERIFY stage (§5 gate) on one small
// representative problem.
// ---------------------------------------------------------------------------

// compute_KE_hat — derive the 8x8 unit-square, unit-E, unit-thickness plane-
// stress element stiffness matrix via 2x2 Gauss quadrature (THEORY.md "The
// math" walks the derivation this function performs numerically — no magic
// constant matrix is hardcoded anywhere in this project: it is COMPUTED,
// once, at startup, in double precision, then narrowed to float). Shared by
// BOTH paths: main.cu calls it once to build the __constant__ table AND to
// feed the CPU twin below, so "the same KE_hat" is not just asserted but
// structurally guaranteed (one function, one call site per path).
void compute_KE_hat(float nu, float KE_hat[64]);

// compute_filter_weights — derive the 5x5 distance-weighted filter kernel
// (w(di,dj) = max(0, rmin - sqrt(di^2+dj^2))) shared by the GPU filter
// kernel's __constant__ table and the CPU twin below.
void compute_filter_weights(float rmin, int r, float* weights /* [(2r+1)*(2r+1)] */);

// topo_cg_solve_cpu — sequential twin of launch_topo_cg_solve: identical
// gather math, identical Jacobi preconditioner, identical stopping rule,
// looped instead of parallel. Same signature spirit as the GPU launcher but
// all pointers are HOST pointers.
void topo_cg_solve_cpu(const TopoGrid& g, const float* rho, const float* F,
                       const uint8_t* fixed, float* U, float E0, float Emin,
                       int max_iters, float rel_tol, int* out_iters, float* out_rel_resid);

// topo_sensitivity_cpu — sequential twin of elem_sensitivity_kernel.
void topo_sensitivity_cpu(const TopoGrid& g, const float* rho, const float* U,
                          float E0, float Emin, float* ce, float* dc_raw);

// topo_filter_cpu — sequential twin of density_filter_kernel.
void topo_filter_cpu(const TopoGrid& g, const float* rho, const float* dc_raw, float* dc_filt);

#endif // PROJECT_KERNELS_CUH
