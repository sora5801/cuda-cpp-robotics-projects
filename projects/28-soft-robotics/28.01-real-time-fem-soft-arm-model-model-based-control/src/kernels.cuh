// ===========================================================================
// kernels.cuh — interface for project 28.01
//               Real-time FEM soft-arm model + model-based control
//               (teaching core: explicit corotational-linear FEM soft arm,
//                scatter+atomics assembly, identified-Jacobian tip control)
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (scenario setup, the verify/analytic-gate
// stages, the closed-loop controller, artifacts), kernels.cu (the GPU element
// -assembly + node-integration kernels), and reference_cpu.cpp (the CPU twin
// used for the §5 GPU-vs-CPU gate, plus the geometry/energy helpers shared by
// both paths). Everything all three must agree on — the mesh layout, the DOF
// numbering, the element stiffness matrix, the material/damping constants,
// and the tendon-actuation model — is defined HERE, once (CLAUDE.md §12).
//
// The project in six lines (THEORY.md derives every step properly):
//   1. A 2-D cantilevered soft arm = a nelx x nely grid of bilinear (Q4)
//      plane-stress elements, one uniform elastomer-class material, fixed
//      along its left edge (the base). One thread per NODE owns 2 DOF (x,y).
//   2. Each element's stiffness is COROTATIONAL-LINEAR: a per-element
//      rotation R_e is extracted from the deformation gradient (closed-form
//      2-D polar decomposition, THEORY.md "The math"), the element's elastic
//      force is computed in the UNROTATED frame (where it is exactly linear
//      elasticity) and rotated back — this is what keeps a "linear" material
//      law stable and artifact-free at the large bending rotations a soft
//      arm actually reaches (contrast: pure linear FEM explodes here).
//   3. Time integration is EXPLICIT SYMPLECTIC EULER (semi-implicit): update
//      velocity from the current force, then update position from the NEW
//      velocity. No linear solve, ever — the whole point of "real-time".
//   4. Two ANTAGONISTIC TENDONS (line forces distributed along the top and
//      bottom fibers) actuate the arm: differential tension bends it, like a
//      bimetallic strip's differential eigenstrain (README/THEORY explain
//      the honest distinction from a free frictionless cable).
//   5. Force ASSEMBLY is one thread per ELEMENT, scattering its 4 corner
//      nodes' contributions with atomicAdd — the race-prone dual of 26.01's
//      gather (THEORY.md "The GPU mapping" tells both stories side by side).
//   6. A model-based task-space controller PROBES the FEM model itself at
//      startup (a small tension step, let it settle, measure the tip) to
//      IDENTIFY a quasi-static gain, then drives a PI loop on tip error
//      through that identified gain — "model-based" in the honest sense: the
//      model is not assumed, it is measured from the very simulator it
//      controls.
//
// MESH & DOF LAYOUT — the single source of truth every file honors (the same
// convention 26.01 uses, so a reader who has seen that project reads this
// one for free):
//     nelx, nely     : elements along x (length axis, FAST), y (height, SLOW)
//     nx = nelx+1    : nodes along x
//     ny = nely+1    : nodes along y
//     node id        n = j*nx + i,           i in [0,nx), j in [0,ny)
//     element id     e = ey*nelx + ex,       ex in [0,nelx), ey in [0,nely)
//     dof (2 per node, x then y)  dof_x = 2*n, dof_y = 2*n+1
//     ndof = 2*nx*ny
// Element (ex,ey) occupies the physical square [ex*h,(ex+1)*h] x
// [ey*h,(ey+1)*h] (h = kElemSize_m, EVERY element the same size — a uniform
// structured mesh, never adaptive). Its 4 corner nodes, in the CCW order the
// element stiffness matrix below assumes (IDENTICAL convention to 26.01):
//     local 0 = node(ex,   ey)      (min-x, min-y corner — "SW")
//     local 1 = node(ex+1, ey)      (max-x, min-y — "SE")
//     local 2 = node(ex+1, ey+1)    (max-x, max-y — "NE")
//     local 3 = node(ex,   ey+1)    (min-x, max-y — "NW")
// The BASE is the i=0 node column (x=0): both DOFs of every base node are
// Dirichlet-fixed — the cantilever boundary condition. The TIP is the i=nelx
// node column (x=L); the CENTERLINE tip node (i=nelx, j=nely/2) is the
// project's single reference point for "the tip position" everywhere a
// scalar tip measurement is needed (control, the analytic gates) — nely=12
// is even, so a true centerline node row exists (no averaging needed).
//
// WHY ELEMENT SIZE h DOES NOT APPEAR IN THE STIFFNESS MATH (reusing 26.01's
// exact, checkable derivation — THEORY.md "The math" repeats it here): for a
// SQUARE Q4 plane-stress element, mapping the [-1,1]^2 parent domain to a
// physical h x h square contributes a Jacobian determinant h^2/4 to the area
// integral and a factor (2/h)^2 to every shape-function derivative inside
// the strain-displacement matrix B — the h^2 cancels the 1/h^2 EXACTLY. The
// element stiffness matrix for a uniform square mesh therefore depends only
// on Young's modulus E, thickness t (out-of-plane, meters), and Poisson's
// ratio nu — never on h. kKeHat below is that DIMENSIONLESS unit-E,
// unit-thickness, unit-square stiffness matrix; every element's REAL
// stiffness is exactly Et * kKeHat where Et = E * thickness (N, a single
// scalar "axial stiffness scale" — see upload_KE_hat / kEt below).
// h DOES still appear elsewhere: in the reference corner coordinates (the
// mesh's actual physical size), in the deformation-gradient shape-function
// GRADIENTS used only to EXTRACT the corotational rotation (a physical-space
// gradient genuinely needs the physical element size), in the CFL timestep
// derivation, and in the lumped node mass. Those uses are irreducible; the
// stiffness-matrix cancellation is the one genuine free lunch.
//
// UNITS (SI throughout, CLAUDE.md §12):
//   x, v        : node position (m), node velocity (m/s); 2 components/node.
//   E           : Pa (N/m^2). nu: dimensionless. rho: kg/m^3. thickness: m.
//   alpha, beta : Rayleigh mass-/stiffness-proportional damping coefficients
//                 (1/s and s respectively — C = alpha*M + beta*K).
//   dt          : s. T_top, T_bottom: N (tendon tensions). force: N.
//   node_mass   : kg. All angles (the extracted rotation theta): rad.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cstdint>

// ---------------------------------------------------------------------------
// HOSTDEV — shared-header trick (26.01/24.01 precedent): __CUDACC__ is
// defined only when nvcc compiles a translation unit, so plain cl.exe
// (reference_cpu.cpp) never sees the CUDA-only __host__ __device__
// qualifiers, while kernels.cu and main.cu (both nvcc) do.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HOSTDEV __host__ __device__
#else
#define HOSTDEV
#endif

// ---------------------------------------------------------------------------
// Mesh size (fixed by the ratified scope: "~120x12 elements").
// ---------------------------------------------------------------------------
constexpr int kNelx = 120;   // elements along the arm's length (x)
constexpr int kNely = 12;    // elements through the arm's height (y)

// ---------------------------------------------------------------------------
// Material & geometry — SYNTHETIC, elastomer-class, labeled everywhere this
// appears (CLAUDE.md §8). Chosen together (not independently) so the CFL
// timestep below lands at a teaching-legible ~1e-5 s scale — see kDt_s's
// derivation comment.
// ---------------------------------------------------------------------------
constexpr float kElemSize_m   = 0.002f;   // h: square element side (2 mm)
constexpr float kArmLength_m  = kElemSize_m * static_cast<float>(kNelx);  // 0.24 m
constexpr float kArmHeight_m  = kElemSize_m * static_cast<float>(kNely);  // 0.024 m
constexpr float kYoungsE_Pa   = 1.0e6f;   // 1 MPa — "elastomer-class", synthetic
constexpr float kPoissonNu    = 0.40f;    // NOT 0.49 (real silicone): nu near 0.5
                                          // volumetrically locks bilinear Q4
                                          // elements badly (THEORY.md "numerics");
                                          // 0.40 keeps the teaching mesh honest
                                          // without a mixed/B-bar formulation.
constexpr float kThickness_m  = 0.02f;    // out-of-plane depth (2 cm) — a REAL
                                          // physical thickness (unlike 26.01's
                                          // "per meter" convention: this project
                                          // needs true mass, so thickness is a
                                          // genuine input, not a bookkeeping unit).
constexpr float kDensity_kgm3 = 1100.0f;  // synthetic, silicone-like

// Et = E * thickness (N): the single scalar that scales the unit stiffness
// matrix kKeHat into this project's REAL element stiffness (see the file
// header's "why h never appears" derivation). Computed once, used every step.
constexpr float kEt_N = kYoungsE_Pa * kThickness_m;   // 1.0e6 * 0.02 = 2.0e4 N

// ---------------------------------------------------------------------------
// CFL-derived timestep (explicit dynamics has no implicit solve to save it
// from instability — THEORY.md "The math" derives this bound from the
// material's longitudinal wave speed):
//     c = sqrt(E/rho) = sqrt(1e6/1100) = 30.1511 m/s
//     dt_CFL = h / c  = 0.002 / 30.1511 = 6.6332e-5 s   (one wave crossing
//                        per element per step is the stability edge)
//     safety factor kCflSafety = 0.5 (a conventional, conservative margin
//                        for explicit FEM — THEORY.md "numerics" discusses
//                        why 2-D lumped-mass corotational elements do not
//                        get a tighter formal bound than this 1-D estimate
//                        in practice, at this mesh's aspect ratio)
//     dt = 0.5 * 6.6332e-5 = 3.317e-5 s
// kDt_s below is a clean, SLIGHTLY MORE conservative round number under that
// bound (safety ratio dt/dt_CFL = 0.452, i.e. more margin than the nominal
// 0.5) — chosen once, by hand, from the arithmetic above, not computed at
// runtime (CLAUDE.md's "derive it in a comment, hardcode the number" style,
// e.g. 08.01's kSigma). NOTE: this dt sits an order of magnitude below the
// catalog bullet's illustrative "~2e-4 s" — THEORY.md "numerics" explains
// honestly why this project's specific (E, h) choice tightens the bound
// rather than forcing an unstable timestep to hit a round number.
constexpr float kCflSafety = 0.5f;      // documented margin (see derivation above)
constexpr float kDt_s      = 3.0e-5f;   // 30 microseconds — the integration AND control sub-step period

// ---------------------------------------------------------------------------
// Rayleigh damping — C = alpha*M + beta*K (mass-proportional + stiffness-
// proportional), BOTH terms integrated explicitly. Explicit integration puts
// a hard STABILITY TAX on beta that every explicit-FEM practitioner learns
// the hard way (this project's own first draft learned it by producing NaN
// within ~350 steps — THEORY.md "numerics" tells that story honestly):
//
//   For one vibration mode omega under symplectic Euler, the velocity update
//   carries the factor (1 - dt*(alpha + beta*omega^2)); the mode explodes
//   once dt*(alpha + beta*omega^2) leaves the stability window (~2; THEORY.md
//   derives it). The HIGHEST mesh mode is the killer: omega_max ~ 2*c/h =
//   2*30.151/0.002 = 30,151 rad/s here, so beta multiplies omega_max^2 ~
//   9.1e8 — a beta sized for the FIRST bending mode (beta ~ 0.024 s for
//   zeta_1 = 0.15) gives dt*beta*omega_max^2 ~ 640: instant NaN.
//
// The stable, standard split (zeta(omega) = alpha/(2*omega) + beta*omega/2,
// derived in THEORY.md "The math"):
//   * alpha carries the LOW-mode damping (zeta ~ alpha/(2*omega) is largest
//     at low frequency — exactly the slow bending mode we want to settle):
//         omega_1 = 2*pi*f1 = 12.7498 rad/s  (f1 = 2.0292 Hz, the analytic
//                             cantilever first mode with THESE constants)
//         alpha = 2*zeta_1*omega_1 = 2*0.15*12.7498 = 3.825  ->  3.8 1/s
//         (zeta_1 = 3.8/(2*12.7498) = 0.149; stability cost dt*alpha =
//          1.1e-4 — nothing)
//   * beta stays SMALL, sized against the stability tax, and damps the
//     mesh-scale ringing alpha cannot reach (zeta ~ beta*omega/2 grows with
//     frequency — exactly where element-level numerical noise lives):
//         beta = 2.0e-5 s  ->  dt*beta*omega_max^2 = 3e-5*2e-5*9.09e8 = 0.55
//         (comfortably inside the bound), zeta(omega_max) ~ 0.30, and its
//         zeta_1 contribution ~ 1.3e-4 (negligible, as intended)
//
// beta is not optional garnish here — it is REQUIRED for stability of this
// force formulation. The classic warped-stiffness corotational force
// f = R*K*(R^T x - X) omits the dR/dx variation of the rotation, so its
// tangent is not symmetric, and the discrete system SELF-EXCITES even from
// exact rest (measured on this model: amplitude e-folds at ~6.4/s from a
// 1e-11 J rounding seed, at dt AND dt/2 — i.e., a property of the force
// field, not the integrator). beta*K damping suppresses that flutter
// completely at negligible physical cost; the truly variational corotational
// force (which needs no such crutch) is README Exercise territory, and
// THEORY.md "numerics" tells the full detective story.
//
// The analytic gates that want an "undamped" system (natural frequency,
// energy conservation) therefore pass alpha=0 with this minimal beta at the
// call site — mode 1 is undamped to within zeta_1 ~ 1.3e-4, which the gate
// bounds account for explicitly (main.cu documents the budget).
// ---------------------------------------------------------------------------
constexpr float kRayleighAlphaOn = 3.8f;     // mass-proportional term (1/s): zeta_1 ~ 0.149 on the first bending mode
constexpr float kRayleighBetaOn  = 2.0e-5f;  // stiffness-proportional term (s): high-mode damper; stability product 0.55

// ---------------------------------------------------------------------------
// Tendon actuation — the "standard tendon-driven soft-arm abstraction" the
// catalog bullet names, distinguished honestly from a pneumatic chamber
// (README/THEORY "where this sits in the real world"). Each tendon is
// modeled as a DISTRIBUTED AXIAL LINE FORCE: every node along its fiber
// (excluding the fixed base column) receives an equal share of the fiber's
// total tension, directed toward the base (-x, "contraction"). This is the
// lumped equivalent of a continuous fiber-reinforced/embedded actuator
// bonded along the arm's surface (as in fiber-reinforced elastomer
// actuators) — NOT a literal frictionless free cable, which (being taut and
// straight) would apply force only at its two endpoints; THEORY.md derives
// why a distributed model is the one that actually bends the arm here.
// Differential tension between the top and bottom fiber creates DIFFERENTIAL
// axial compression across the cross-section — exactly the bimetallic-strip
// mechanism that gives soft continuum arms and PneuNet-style bending
// actuators their curvature (README "Prior art").
// ---------------------------------------------------------------------------
constexpr int kTendonAttachNodes = kNelx;   // i = 1..kNelx on each fiber row (i=0 is the fixed base, excluded)
// Co-contraction bias: BOTH tendons carry this pretension at rest, so a
// differential can be commanded in either direction while both stay taut
// (a real cable cannot push). The bias is BOUNDED ABOVE by Euler buckling of
// the arm itself — tendon tension compresses the backbone axially, and a
// cantilever's critical compressive load is
//     P_cr = pi^2 * E * I / (2L)^2,  I = t*H^3/12 = 2.304e-8 m^4
//          = 9.8696 * 1e6 * 2.304e-8 / (0.48)^2 = 0.987 N        (THEORY.md)
// Total compression = T_top + T_bottom = 2*bias (the differential cancels),
// so 2*bias must sit WELL below 0.987 N: bias = 0.25 N puts the constant
// compression at 0.5 N ~ 51% of P_cr — bent-but-not-buckling territory, and
// a genuinely soft-robotic design constraint worth teaching (tendon-driven
// continuum arms really do buckle when over-tensioned; PRACTICE.md §1).
constexpr float kTendonBiasN     = 0.25f;   // baseline co-contraction pretension per tendon (N)

// Sentinel: no point force active this step (used by the analytic static-
// deflection gate only; every other phase passes this sentinel).
constexpr int kNoPointForce = -1;

// ---------------------------------------------------------------------------
// ArmGrid — the mesh definition every kernel/host function receives, by
// value (constant-cache-backed on the GPU side — every thread in a launch
// reads the same 4 ints, a broadcast read; same reasoning as 26.01's
// TopoGrid). Plain aggregate on purpose (CLAUDE.md's usual "no hidden
// behavior in POD" reasoning).
// ---------------------------------------------------------------------------
struct ArmGrid {
    int nelx, nely;   // elements along x (fast), y (slow)
    int nx, ny;       // nodes = nelx+1, nely+1 (stored, not recomputed)
};

inline HOSTDEV ArmGrid make_arm_grid()
{
    ArmGrid g;
    g.nelx = kNelx; g.nely = kNely;
    g.nx = kNelx + 1; g.ny = kNely + 1;
    return g;
}

// node_id / elem_id — THE flat-index arithmetic (the layout contract stated
// above). HOSTDEV so main.cu (host), kernels.cu (device), and
// reference_cpu.cpp (host, via cl.exe) all evaluate the identical expression
// — eliminates an entire class of "CPU oracle indexed it differently" bugs
// by construction (26.01's exact reasoning, reapplied).
inline HOSTDEV int node_id(const ArmGrid& g, int i, int j) { return j * g.nx + i; }
inline HOSTDEV int elem_id(const ArmGrid& g, int ex, int ey) { return ey * g.nelx + ex; }

// tip_node_index — the project's single reference point for "the tip":
// the CENTERLINE node at the free end (i = nelx, j = nely/2). nely is even
// (12), so this row exists exactly — no averaging of top/bottom needed.
inline HOSTDEV int tip_node_index(const ArmGrid& g) { return node_id(g, g.nelx, g.nely / 2); }

// corner_offset — local corner a (0..3, the CCW convention in the file
// header) -> its (cx,cy) offset in ELEMENT-INDEX units from the element's
// (ex,ey) origin corner. Used to compute both the corner's NODE id
// (ex+cx, ey+cy) and its REFERENCE physical position ((ex+cx)*h, (ey+cy)*h).
// switch/case (not a static array) — the same device-safety choice 26.01's
// quadrant_elem makes (function-local static arrays in device code are
// legal but this repo prefers the branch form for uniformity across files).
inline HOSTDEV void corner_offset(int a, int* cx, int* cy)
{
    switch (a) {
        case 0: *cx = 0; *cy = 0; break;   // SW
        case 1: *cx = 1; *cy = 0; break;   // SE
        case 2: *cx = 1; *cy = 1; break;   // NE
        default:*cx = 0; *cy = 1; break;   // NW (a == 3)
    }
}

// corner_parent_sign — local corner a -> its PARENT-DOMAIN coordinate signs
// (xi_i, eta_i) in {-1,+1}^2 (the bilinear Q4 shape-function corner table,
// same convention 26.01's compute_KE_hat uses). Needed to build both the
// unit stiffness matrix (reference_cpu.cpp's compute_KE_hat) and the
// physical-space shape-function gradients used for the corotational
// rotation extraction (kernels.cu / reference_cpu.cpp's fem_step twins).
inline HOSTDEV void corner_parent_sign(int a, float* xi_i, float* eta_i)
{
    switch (a) {
        case 0: *xi_i = -1.0f; *eta_i = -1.0f; break;
        case 1: *xi_i =  1.0f; *eta_i = -1.0f; break;
        case 2: *xi_i =  1.0f; *eta_i =  1.0f; break;
        default:*xi_i = -1.0f; *eta_i =  1.0f; break;   // a == 3
    }
}

// grad_n_physical — the CONSTANT (element-independent, for a uniform square
// mesh) physical-space gradient of corner a's bilinear shape function,
// evaluated at the element CENTER (parent xi=eta=0 — the only point this
// project's corotational rotation extraction needs, THEORY.md "The math").
// dN_a/dxi|_0 = xi_i/4, dN_a/deta|_0 = eta_i/4 (bilinear shape function
// derivative, evaluated at the center); the parent->physical Jacobian for a
// square element of side h is the constant diag(2/h, 2/h) (h/2 maps
// [-1,1]->[0,h]), so dN_a/dX = (xi_i/4)*(2/h) = xi_i/(2h), and likewise for Y.
inline HOSTDEV void grad_n_physical(int a, float h, float* dNdX, float* dNdY)
{
    float xi_i, eta_i;
    corner_parent_sign(a, &xi_i, &eta_i);
    *dNdX = xi_i  / (2.0f * h);
    *dNdY = eta_i / (2.0f * h);
}

#ifdef __CUDACC__  // ---- device-aware section: only nvcc sees this ---------

// The unit element stiffness matrix kKeHat (8x8, row-major, E=1 Pa, t=1 m,
// nu=kPoissonNu, unit square) lives in __constant__ memory: EVERY thread in
// EVERY launch of elem_force_kernel reads the SAME 64 floats — the textbook
// __constant__ broadcast case (08.01/26.01's identical reasoning). Populated
// once via upload_KE_hat() before the first step. See the "extern
// __constant__" note in 26.01's kernels.cuh for why these symbols are
// declared+defined privately in kernels.cu and never named outside it.

// ---- device kernels (defined + fully commented in kernels.cu) -------------

// elem_force_kernel — one thread per ELEMENT: extract the corotational
// rotation, compute the local (elastic + stiffness-proportional-damping)
// force, rotate it back to world, and SCATTER it (atomicAdd) into the 4
// corner nodes' entries of `force` — the race-prone dual of 26.01's gather
// (file-header point 5; full race-condition story in kernels.cu). `force`
// MUST hold all zeros on entry — guaranteed by the zero-after-consume
// contract below (node_integrate_kernel re-zeroes each entry it reads) plus
// one initial cudaMemset at allocation time (main.cu's ArmSim::init).
__global__ void elem_force_kernel(ArmGrid g,
                                  const float* __restrict__ x,      // [2*nnode] current position (m)
                                  const float* __restrict__ v,      // [2*nnode] current velocity (m/s)
                                  float Et, float h, float beta,
                                  float* __restrict__ force);       // [2*nnode] OUT (atomics target, zero on entry)

// node_integrate_kernel — one thread per NODE: read the assembled internal
// force, add mass-proportional damping + tendon actuation + (optionally) one
// analytic-gate point force, divide by node mass, and advance with symplectic
// (semi-implicit) Euler: v += dt*a; x += dt*v — velocity FIRST, using the
// OLD position's force, THEN position from the NEW velocity (the specific
// order that gives symplectic Euler its bounded-energy-drift property,
// THEORY.md "numerics"). Dirichlet-fixed nodes (i==0) are reset to their
// rest position with zero velocity instead of integrated.
// ZERO-AFTER-CONSUME: after reading its node's 2 force entries, the thread
// writes 0 back to them, leaving the buffer ready for the NEXT step's
// scatter — this replaces a per-step cudaMemset (one fewer host API call in
// the hot loop; measured effect in kernels.cu's launch_fem_step comment).
__global__ void node_integrate_kernel(ArmGrid g,
                                      float* __restrict__ x,             // [2*nnode] IN/OUT position (m)
                                      float* __restrict__ v,             // [2*nnode] IN/OUT velocity (m/s)
                                      float* __restrict__ force,         // [2*nnode] IN: assembled force (N); zeroed on the way out
                                      const float* __restrict__ node_mass, // [nnode] lumped node mass (kg)
                                      const uint8_t* __restrict__ fixed,   // [2*nnode] 1 = Dirichlet-fixed dof
                                      float h, float alpha, float dt,
                                      float T_top, float T_bottom,       // tendon tensions this step (N)
                                      int point_force_node,              // kNoPointForce, or a node index
                                      float point_force_x, float point_force_y); // N, used only if point_force_node >= 0

#endif // __CUDACC__ --------------------------------------------------------

// upload_KE_hat — one-time host->constant-memory upload (kernels.cu owns the
// __constant__ symbol; this is the only function allowed to write it).
void upload_KE_hat(const float KE_hat[64]);

// ---------------------------------------------------------------------------
// launch_fem_step — the project's GPU workhorse: ONE dt of explicit
// corotational FEM dynamics, entirely via the two kernels above (zero force
// -> scatter-assemble -> integrate). This is the function main.cu's tight
// physics loop calls, every kDt_s, thousands of times per phase — its
// measured steps-per-wall-second IS this project's real-time-factor claim
// (README "Expected output", THEORY "The GPU mapping").
//
//   g          : mesh (nelx, nely, nx, ny).
//   d_x, d_v   : DEVICE [2*nnode] IN/OUT — the arm's current state.
//   d_force    : DEVICE [2*nnode] SCRATCH — must be all-zero on the FIRST
//                call (one cudaMemset at allocation, main.cu); every call
//                leaves it zeroed for the next (the zero-after-consume
//                contract on node_integrate_kernel above).
//   d_node_mass: DEVICE [nnode] lumped mass (kg), precomputed once.
//   d_fixed    : DEVICE [2*nnode] Dirichlet mask, precomputed once.
//   Et, h      : material stiffness scale (N) and element size (m) — see the
//                file header's "why h never appears [in the stiffness]"
//                derivation for why only Et (not E and thickness separately)
//                is needed by the force kernel.
//   alpha, beta: Rayleigh damping coefficients THIS call uses (0 for the two
//                undamped analytic gates, kRayleighAlpha/BetaOn otherwise).
//   dt         : integration timestep (s) — always kDt_s in this project,
//                passed explicitly so the CPU twin's signature matches.
//   T_top, T_bottom: tendon tensions (N) held constant for this one step.
//   point_force_node/x/y: kNoPointForce to disable, or a node index + force
//                (N) for the static-deflection analytic gate.
// ---------------------------------------------------------------------------
void launch_fem_step(const ArmGrid& g,
                     float* d_x, float* d_v, float* d_force,
                     const float* d_node_mass, const uint8_t* d_fixed,
                     float Et, float h, float alpha, float beta, float dt,
                     float T_top, float T_bottom,
                     int point_force_node, float point_force_x, float point_force_y);

// ---------------------------------------------------------------------------
// CPU references (reference_cpu.cpp) — plain, sequential twins of every GPU
// stage above, used by main.cu's VERIFY stage (§5 gate), PLUS the small
// geometry/energy utilities BOTH paths share (one function, one call site
// per path — 26.01's "shared by construction" pattern).
// ---------------------------------------------------------------------------

// compute_KE_hat — derive the 8x8 unit-square (h=1), unit-thickness (t=1),
// unit-modulus (E=1 Pa) plane-stress element stiffness matrix via 2x2 Gauss
// quadrature, in double precision, then narrow to float (THEORY.md "The
// math" walks the derivation this function performs numerically — no magic
// constant matrix is hardcoded anywhere in this project). Shared by BOTH
// paths: main.cu calls it once to build the __constant__ table AND to feed
// the CPU twin, so "the same stiffness matrix" is structurally guaranteed.
void compute_KE_hat(float nu, float KE_hat[64]);

// compute_node_mass — lumped mass at every node: each element contributes
// rho*thickness*h^2 (its mass) split EQUALLY among its 4 corners (row-sum
// lumping, the standard consistent-to-lumped simplification for a uniform
// Q4 mesh — THEORY.md "numerics"). A node with up to 4 incident elements
// sums up to 4 such shares. Computed ONCE at startup, shared by GPU (upload
// the result) and CPU (use it directly) — never recomputed per step.
void compute_node_mass(const ArmGrid& g, float rho, float thickness, float h,
                       float* node_mass /* [nnode] OUT, kg */);

// build_fixed_mask — the cantilever boundary condition: both DOFs of every
// i==0 (base) node are Dirichlet-fixed (1); everything else is free (0).
void build_fixed_mask(const ArmGrid& g, uint8_t* fixed /* [2*nnode] OUT */);

// fem_step_cpu — sequential twin of launch_fem_step: identical rotation
// extraction, identical local-force formula, but element contributions are
// summed into `force` in FIXED element-index order (0..nelem-1) instead of
// via atomics — a deterministic reference sum the GPU's atomicAdd order is
// compared against within a documented, reassociation-aware tolerance (the
// §5 gate; THEORY.md "How we verify correctness" derives the tolerance).
void fem_step_cpu(const ArmGrid& g,
                  float* x, float* v, float* force,
                  const float* node_mass, const uint8_t* fixed,
                  float Et, float h, float alpha, float beta, float dt,
                  float T_top, float T_bottom,
                  int point_force_node, float point_force_x, float point_force_y);

// arm_kinetic_energy / arm_elastic_pe — DIAGNOSTIC energy functions (host-
// only; not part of the GPU-vs-CPU dueling pair, but reuse the identical
// per-element corotational math fem_step_cpu uses internally, so "energy"
// and "force" can never quietly disagree). Used by main.cu's energy-
// conservation analytic gate on state downloaded from either path.
//   arm_kinetic_energy: 0.5 * sum_node m_node * |v_node|^2               (J)
//   arm_elastic_pe    : 0.5 * sum_elem  u_local_e^T * (Et*KE_hat) * u_local_e (J)
double arm_kinetic_energy(const ArmGrid& g, const float* v, const float* node_mass);
double arm_elastic_pe(const ArmGrid& g, const float* x, float Et, float h);

#endif // PROJECT_KERNELS_CUH
