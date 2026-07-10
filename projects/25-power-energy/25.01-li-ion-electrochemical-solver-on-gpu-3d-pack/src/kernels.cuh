// ===========================================================================
// kernels.cuh — interface for project 25.01
//               Li-ion electrochemical (SPM tier of the P2D/SPMe ladder) +
//               3D pack thermal simulation + cooling-design sweeps
//               (teaching core: batched single-particle-model electrochemistry
//               coupled to a batched anisotropic 3D pack heat-equation solve)
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (scenario loading, the mission-profile driver,
// Butler-Volmer/OCV/voltage bookkeeping, the cooling-design sweep, artifacts,
// verification), kernels.cu (the two GPU stencil kernels), and
// reference_cpu.cpp (their CPU twins). Everything all three must agree on —
// array layouts, struct shapes, and constants — is defined HERE, once
// (CLAUDE.md §12).
//
// RATIFIED SCOPE (CLAUDE.md §2/§13 — read this before anything else): the
// catalog bullet names the electrochemical ladder "P2D/SPMe". This project
// ships the SPM (Single Particle Model) tier — the ladder's simplest rung
// that still contains real solid-phase diffusion PDEs and real Butler-Volmer
// kinetics — plus the FULL 3D pack thermal solver and cooling sweep the
// bullet also names. SPMe (adds electrolyte-phase concentration/potential)
// and full P2D (resolves the porous electrode with x-position-dependent
// particles) are documented, with their governing equations, in THEORY.md
// "Where this sits in the real world" — the ladder rung actually implemented
// is stated honestly in README §13, not silently upgraded in the text.
//
// Li-ion electrochemistry + pack thermal in eight lines (THEORY.md derives
// each properly):
//   1. Each electrode (anode, cathode) is modeled as ONE representative
//      spherical particle per CELL — lithium intercalates by SOLID-STATE
//      DIFFUSION, a genuine 1-D-in-r parabolic PDE, dc/dt = (1/r^2) d/dr
//      (D r^2 dc/dr), solved via a spherical finite-volume (FV) scheme.
//   2. The applied pack current sets a NEUMANN flux boundary condition at
//      each particle's outer shell (mol/(m^2 s)) — Li leaves the anode
//      particle and enters the cathode particle during discharge.
//   3. The particle's SURFACE concentration feeds Butler-Volmer kinetics:
//      an activation overpotential eta that, for the symmetric (alpha=0.5)
//      case used here, inverts in CLOSED FORM via eta = (2RT/F)*asinh(...).
//   4. Terminal voltage = OCV_cathode(x_c) - OCV_anode(x_a) + eta_c - eta_a
//      - I*R_ohm; heat generated per cell q = I*(OCV_cell - V_cell) (the
//      irreversible/ohmic heat; the reversible entropic term is a documented
//      omission, README §13).
//   5. Each of the kNCells=24 cells is a heat SOURCE in a 3-D anisotropic
//      pack medium — a second PDE, the heat equation, solved by an explicit
//      finite-volume stencil on a 32x24x16 voxel grid.
//   6. Cooling is a Robin (convective) boundary condition on exactly ONE
//      face of the pack (a bottom cold-plate or a side cold-plate, the
//      design choice being swept) with a design-specific coefficient h.
//   7. The coupling is TWO-WAY and LAGGED (documented, not hidden): local
//      cell temperature (read once per thermal step) Arrhenius-scales each
//      particle's diffusivity D(T) and reaction-rate prefactor k(T); the
//      resulting heat generation feeds back into the next thermal step's
//      source term.
//   8. THE SWEEP is the actual design question: 6 values of h x 2 cooling
//      faces = 12 independent pack designs, BATCHED into one sequence of
//      kernel launches (batch axis rides outside every array, exactly the
//      24.01/31.01 batching idiom), each driven through the SAME mission
//      profile, compared on peak temperature, cell-to-cell spread, and
//      end-of-mission voltage spread.
//
// WHY A SINGLE SHARED CURRENT ACROSS ALL 24 CELLS: the pack is modeled as
// one series string under simple (non-actively-balancing) BMS control, so
// every cell sees the SAME commanded pack current at every instant — a
// documented simplification (README §13). What DOES differ, cell to cell and
// design to design, is each cell's local TEMPERATURE (from its position in
// the 3-D thermal grid and the cooling design), which Arrhenius-scales that
// cell's D and k — this is the electro-thermal coupling the catalog bullet
// asks for, and it is exactly what makes the 12 designs diverge from an
// identical start.
//
// ARRAY LAYOUTS (one source of truth, shared verbatim by every file):
//
//   PARTICLE ("electrochemistry") STATE — flatten a "particle index"
//       p = (b * kNCells + cell) * 2 + e
//   for design b in [0,B), cell in [0,kNCells), electrode e in {0=anode,
//   1=cathode} — e is the FASTEST-varying axis of p (consecutive particles
//   alternate anode/cathode for the same cell), design is SLOWEST. Then the
//   concentration array is
//       c[p * kNShells + s]        s in [0, kNShells)   shell index,
//                                   FASTEST axis: r_s = (s+0.5)*R_p/kNShells
//                                   (cell-centered spherical FV shells)
//   Per-particle scalar arrays (diffusivity D, current cell temperature)
//   are simply D[p], T_cell[p] — same flattening, no shell axis.
//
//   THERMAL ("pack") STATE — flatten a voxel index
//       idx = ((b * kTNZ + k) * kTNY + j) * kTNX + i
//   for design b in [0,B), k in [0,kTNZ) (z, SLOWEST spatial axis), j in
//   [0,kTNY) (y), i in [0,kTNX) (x, FASTEST/contiguous axis — the repo's
//   standard coalescing choice, e.g. 24.01/31.01). Design b rides OUTSIDE
//   the whole per-design 3-D block, exactly the 24.01 batched-grid idiom
//   extended from 2-D to 3-D.
//   A cell (cx,cy,cz) in [0,kPackNX)x[0,kPackNY)x[0,kPackNZ) owns the
//   voxel block i in [cx*kVoxPerCellX, (cx+1)*kVoxPerCellX), and likewise
//   for j/k — kTNX/kPackNX = kVoxPerCellX voxels per cell per axis, exactly
//   (compile-time-checked below).
//
// UNITS & SIGN CONVENTIONS (used by every file here):
//   Concentration c        : mol/m^3.
//   Flux j (per electrode)  : mol/(m^2 s), POSITIVE = Li LEAVING that
//                             electrode's particle (delithiation). During
//                             pack discharge (I_cell > 0 by convention) the
//                             anode delithiates (j_a > 0) and the cathode
//                             lithiates (j_c < 0) — see main.cu's mission
//                             loop for the exact j_a/j_c <-> I_cell formula.
//   Temperature T           : kelvin, everywhere (never Celsius in code).
//   Voltage/OCV/overpotential: volts.  Current I: amps (pack/cell current;
//                             SPM treats the whole cell as one lumped
//                             circuit element, no current collector losses).
//   Heat source q            : watts (per cell, from main.cu's electro-
//                             thermal bookkeeping) or W/m^3 (volumetric,
//                             inside the thermal kernel — main.cu converts).
//   Thermal conductivity k   : W/(m K), ANISOTROPIC (kx, ky, kz) — a wound/
//                             stacked cell's in-plane conductivity is much
//                             higher than its through-plane conductivity
//                             (THEORY.md "The problem" explains why).
//
// All scenario numbers (electrode/particle parameters, pack thermal
// properties, the mission profile, the sweep's h values) come from
// data/sample/pack_scenario.csv, loaded once by main.cu, so the committed
// sample and the compiled solver can never silently disagree (the same
// discipline every flagship's scenario file follows).
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

// ---------------------------------------------------------------------------
// Universal physical constants — physics, not scenario data, so they live
// here once rather than in the CSV (the same reasoning 24.01 gives for mu0).
// ---------------------------------------------------------------------------
constexpr float kFaradayC   = 96485.33212f;  // Faraday constant, C/mol
constexpr float kGasConstJ  = 8.314462618f;  // universal gas constant, J/(mol K)
constexpr float kArrheniusRefT = 298.15f;    // Arrhenius reference temperature, K (25 C)

// ---------------------------------------------------------------------------
// Fixed problem sizes — compile-time constants (not scenario-loaded)
// because they govern every array-layout computation in every file's
// comments below, exactly as kGridN does in 24.01.
// ---------------------------------------------------------------------------
constexpr int kNShells  = 20;                 // radial FV shells per particle

constexpr int kPackNX = 4, kPackNY = 3, kPackNZ = 2;         // cells per axis
constexpr int kNCells  = kPackNX * kPackNY * kPackNZ;         // = 24

constexpr int kTNX = 32, kTNY = 24, kTNZ = 16;                // thermal voxels per axis
constexpr int kVoxPerCellX = kTNX / kPackNX;                  // = 8
constexpr int kVoxPerCellY = kTNY / kPackNY;                  // = 8
constexpr int kVoxPerCellZ = kTNZ / kPackNZ;                  // = 8
static_assert(kVoxPerCellX * kPackNX == kTNX, "thermal grid must tile the pack exactly in x");
static_assert(kVoxPerCellY * kPackNY == kTNY, "thermal grid must tile the pack exactly in y");
static_assert(kVoxPerCellZ * kPackNZ == kTNZ, "thermal grid must tile the pack exactly in z");

constexpr int kNSweepH     = 6;                // swept convective coefficients
constexpr int kNSweepFaces = 2;                // swept cooling-face choices
constexpr int kNDesigns    = kNSweepH * kNSweepFaces;   // = 12, the sweep batch size

// Which pack face carries the Robin (convective) cooling boundary condition
// for a given design. All FIVE other faces are adiabatic (zero-flux) —
// documented, not hidden: a real enclosure loses some heat everywhere, but
// "one dominant cooling path" is the honest first-order pack-design question
// (THEORY.md "The problem").
enum CoolFace : int { kCoolBottomZ = 0, kCoolSideX = 1 };

// ---------------------------------------------------------------------------
// ElectrodeGeom — one electrode's (anode's or cathode's) particle geometry,
// solid-diffusion, and reaction-kinetics parameters. A plain aggregate (no
// methods, no CUDA types) so it compiles under both nvcc and cl.exe and
// rides in the kernel-argument buffer by value — the same reasoning 24.01's
// FeaGrid documents.
//
// All numeric VALUES the demo uses are SYNTHETIC teaching parameters (README
// §Data, THEORY.md "The math") — shaped to be plausible (real activation
// energies, real diffusivity orders of magnitude, an OCV curve of the right
// qualitative shape) but never claimed to match any specific real cell or
// dataset.
// ---------------------------------------------------------------------------
struct ElectrodeGeom {
    float R_p;      // particle radius (m)
    float c_max;    // maximum lithium concentration this electrode can hold (mol/m^3)
    float D25;      // solid-phase diffusivity AT the Arrhenius reference temperature (m^2/s)
    float Ea_D;     // activation energy for diffusion (J/mol) — Arrhenius: D(T) = D25*exp(-Ea_D/R*(1/T - 1/Tref))
    float i0_ref;   // exchange-current-density prefactor AT the reference temperature (A/m^2);
                     // the full exchange current density is i0_ref * exp(Arrhenius term) * sqrt(x*(1-x))
                     // — the sqrt(x(1-x)) shape factor is the SPM's simplified stand-in for the
                     // full Butler-Volmer c_e^(1-a)*c_surf^a*(c_max-c_surf)^(1-a) dependence, valid
                     // under SPM's own assumption of a spatially uniform, constant electrolyte
                     // concentration (THEORY.md derives the reduction and names exactly where it breaks).
    float Ea_k;      // activation energy for the reaction-rate prefactor (J/mol)
    float A_surf;    // total active particle surface area for this electrode, PER CELL (m^2) —
                     // sets how a cell current [A] maps to a per-particle molar flux [mol/(m^2 s)]:
                     // j = I_cell / (F * A_surf). Physically this stands in for "particle count x
                     // one particle's area" without the demo needing to invent a particle count.
};

// ---------------------------------------------------------------------------
// PackThermalParams — the 3-D pack medium's anisotropic thermal properties
// and voxel geometry, shared by every design in the batch (only h and the
// cooling face vary per design — see DesignPoint below).
// ---------------------------------------------------------------------------
struct PackThermalParams {
    float rho_cp;     // volumetric heat capacity, rho*cp (J/(m^3 K))
    float kx, ky, kz; // ANISOTROPIC effective thermal conductivity (W/(m K)) —
                       // kx,ky ("in-plane") are set higher than kz ("through-
                       // plane") to teach the real wound/stacked-cell anisotropy
                       // (THEORY.md "The problem").
    float dx, dy, dz; // voxel pitch (m); dx = cell_Lx / kVoxPerCellX, etc.
    float T_coolant;  // coolant temperature at the cooling face (K), held fixed
};

// ---------------------------------------------------------------------------
// DesignPoint — one point in the 12-design cooling sweep: a convective
// coefficient and which face it acts on. main.cu builds the 12-entry array
// (6 h values x {bottom, side}) that both the batched launcher and the CPU
// twin read.
// ---------------------------------------------------------------------------
struct DesignPoint {
    float h;     // convective heat-transfer coefficient at the cooling face (W/(m^2 K))
    int   face;  // CoolFace — which single face carries this design's cooling
};

// ---------------------------------------------------------------------------
// Verification tolerances — measured, documented, not guessed (CLAUDE.md:
// never widen a tolerance without proof). See main.cu's VERIFY stage for the
// measured worst-case numbers this headroom is checked against.
// ---------------------------------------------------------------------------
constexpr float kTwinTolConc = 5.0e-2f;   // max |c_gpu - c_cpu|, mol/m^3, over the verify slice
constexpr float kTwinTolTemp = 5.0e-4f;   // max |T_gpu - T_cpu|, K, over the verify slice

#ifdef __CUDACC__  // ---- device-aware section: only nvcc sees this ---------

// electrochem_fv_kernel / thermal_step_kernel — declared here for
// completeness (full derivation + definition in kernels.cu); main.cu never
// calls these directly, only the host launchers below.
__global__ void electrochem_fv_kernel(int B,
                                      ElectrodeGeom geomA, ElectrodeGeom geomC,
                                      float j_a, float j_c, float dt_e,
                                      const float* __restrict__ D,
                                      const float* __restrict__ c_in,
                                      float*       __restrict__ c_out);

__global__ void thermal_step_kernel(int B, PackThermalParams p, float dt_thermal,
                                    const DesignPoint* __restrict__ designs,
                                    const float* __restrict__ q_vol,
                                    const float* __restrict__ T_in,
                                    float*       __restrict__ T_out);

#endif // __CUDACC__ --------------------------------------------------------

// ---------------------------------------------------------------------------
// launch_electrochem_substep — run ONE explicit FV timestep of solid-phase
// diffusion for every particle in the batch (B designs x kNCells cells x 2
// electrodes = B*kNCells*2 independent spherical diffusion problems).
//
//   B        : number of designs in this batch (>= 1).
//   geomA/geomC : anode/cathode particle geometry+kinetics (shared by every
//                 design and cell — only D and the per-cell temperature vary).
//   j_a, j_c : this step's surface flux boundary condition (mol/(m^2 s)),
//              SAME for every cell/design (the shared-mission-current
//              simplification documented above) — sign convention: positive
//              = leaving the particle.
//   dt_e     : electrochemistry substep size (s); main.cu derives it from
//              dt_thermal / n_sub and documents the diffusion-CFL headroom.
//   d_D      : DEVICE pointer, B*kNCells*2 floats — this step's Arrhenius-
//              scaled diffusivity per particle (main.cu recomputes it every
//              step from each particle's current cell temperature).
//   d_c_in   : DEVICE pointer, B*kNCells*2*kNShells floats — concentration
//              at the START of this step (the layout documented above).
//   d_c_out  : DEVICE pointer, same shape, OUT — concentration one dt_e
//              later. d_c_in and d_c_out must be DISTINCT buffers (ping-pong
//              — a stencil read of a still-being-written neighbor would race);
//              main.cu owns the two persistent buffers and swaps pointers
//              across the many calls in its mission loop (allocated ONCE
//              outside the loop — the 08.01 precedent for a hot per-tick call).
//
// Launch: one thread per (particle, shell), grid-stride-free flat 1-D launch
// (grid math + reasoning with the kernel in kernels.cu).
// ---------------------------------------------------------------------------
void launch_electrochem_substep(int B, ElectrodeGeom geomA, ElectrodeGeom geomC,
                                float j_a, float j_c, float dt_e,
                                const float* d_D,
                                const float* d_c_in, float* d_c_out);

// ---------------------------------------------------------------------------
// launch_thermal_substep — run ONE explicit FV timestep of the anisotropic
// 3-D pack heat equation for every design in the batch.
//
//   B        : number of designs in this batch (>= 1).
//   p        : shared thermal medium properties + voxel geometry.
//   d_designs: DEVICE pointer, B DesignPoint — this batch's h/cooling-face
//              per design (main.cu uploads it once; it never changes during
//              the mission).
//   d_q_vol  : DEVICE pointer, B*kTNZ*kTNY*kTNX floats — this step's
//              volumetric heat source (W/m^3) per voxel, per design
//              (main.cu rebuilds it every step from that step's per-cell
//              heat generation, uniformly spread over each cell's voxel
//              block — see main.cu's build_heat_source()).
//   d_T_in   : DEVICE pointer, same shape — temperature (K) at the START of
//              this step.
//   d_T_out  : DEVICE pointer, same shape, OUT — temperature one dt_thermal
//              later. Ping-pong, same discipline as the electrochem call.
//   dt_thermal: thermal substep size (s); main.cu checks it against the
//              explicit-FTCS CFL bound (kernels.cu derives the bound) before
//              the mission loop even starts.
//
// Launch: one thread per (design, voxel), 8x8 (i,j) tiles, blockIdx.z
// encoding (design, k) — grid math + reasoning with the kernel in kernels.cu.
// ---------------------------------------------------------------------------
void launch_thermal_substep(int B, const PackThermalParams& p, float dt_thermal,
                            const DesignPoint* d_designs,
                            const float* d_q_vol,
                            const float* d_T_in, float* d_T_out);

// ---------------------------------------------------------------------------
// CPU references (reference_cpu.cpp) — oracle twins of the two kernels
// above, sequential over every (particle,shell) / (design,voxel), same FP32
// update expressions. main.cu runs both against the GPU on a small
// representative slice and requires agreement within the tolerances above
// (the CLAUDE.md §5 GPU-vs-CPU gate for this project).
// ---------------------------------------------------------------------------
void electrochem_fv_cpu(int B, ElectrodeGeom geomA, ElectrodeGeom geomC,
                        float j_a, float j_c, float dt_e,
                        const float* D, const float* c_in, float* c_out);

void thermal_step_cpu(int B, const PackThermalParams& p, float dt_thermal,
                      const DesignPoint* designs,
                      const float* q_vol, const float* T_in, float* T_out);

#endif // PROJECT_KERNELS_CUH
