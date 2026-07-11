// ===========================================================================
// kernels.cuh — kernel & reference declarations for project 35.01
//               (Magnetic microrobot swarms: Biot-Savart field computation +
//               swarm dynamics — [R&D] catalog bullet, reduced-scope teaching
//               version; see README.md "Limitations & honesty" for scoping.)
//
// Role in the project
// --------------------
// This header is the ONE PLACE the coil geometry, the field-map memory
// layout, the swarm state layout, and every physical unit are defined —
// every other file (kernels.cu, reference_cpu.cpp, main.cu) includes this
// header and agrees with it by construction (CLAUDE.md §12: "every state
// vector documents its layout in one place and cross-references it").
//
// The pipeline this header describes, end to end (see THEORY.md "The GPU
// mapping" for the full reasoning):
//
//   1. biot_savart_basis_kernel  — brute-force Biot-Savart sum over ALL
//      coil segments, ONE call per coil, unit current on that coil only.
//      Produces 4 "basis" field maps B_c(x) (T per ampere-turn) on a
//      256x256 grid. This is the expensive step (256x256 grid x 720
//      segments = ~188k thread-segment products) and the catalog bullet's
//      named GPU hook — but it runs only ONCE per demo, not once per
//      control tick, because of step 2:
//   2. combine_field_kernel — exploits the LINEARITY of Maxwell's equations
//      in vacuum: the field of ANY coil-current vector I = (I_E,I_W,I_N,I_S)
//      is the linear combination  B(x) = sum_c I_c * B_c(x)  of the 4
//      basis maps. This is a trivial elementwise "map" kernel, and it is
//      what makes the swarm loop below cheap: no coil re-sums a single
//      Biot-Savart segment ever again after step 1.
//   3. gradient_b2_kernel — a STENCIL kernel (4-neighbor central difference)
//      that turns the combined (Bx,By) map into a (dB2x,dB2y) map holding
//      d(|B|^2)/dx and d(|B|^2)/dy at every grid cell. Precomputing this
//      ONCE per phase (not once per robot per step) is what makes step 4
//      cheap: 65536 stencil evaluations vs. 1000 robots x 300 steps x a
//      5-point finite difference if done per-robot-per-step.
//   4. swarm_step_kernel — one thread per ROBOT (the "agent farm" pattern
//      used throughout this repo, e.g. 08.01's rollouts, 22.01's agents).
//      Each thread bilinearly interpolates (dB2x,dB2y) at its own robot's
//      position, turns that into a force via the superparamagnetic-bead
//      model (THEORY.md "The math"), turns the force into a velocity via
//      Stokes drag (THEORY.md "The problem" — low-Reynolds-number regime,
//      so there is no separate "convert force to acceleration" step: at
//      Re << 1, velocity IS proportional to force, with no inertia), and
//      integrates position forward for `steps` explicit-Euler sub-steps
//      inside ONE kernel launch (so a whole schedule "phase" is one call).
//
// Why ".cuh"?
// -----------
// The repo convention (CLAUDE.md §12): .cuh headers may contain CUDA-
// specific constructs (__global__, kernel launches) and are included from
// nvcc-compiled .cu files; plain .h headers stay host-only. This header is
// ALSO included by reference_cpu.cpp, compiled by the HOST compiler
// (cl.exe), which does not know __global__ — so device-only declarations
// are fenced behind #ifdef __CUDACC__ (a macro only nvcc defines). This is
// the standard trick used throughout this repository (see the scaffold's
// own note, preserved below for continuity).
//
// A SECOND, related trick this file introduces: several small numerical
// helpers (the Biot-Savart per-segment contribution, bilinear sampling)
// are needed VERBATIM on both the GPU (kernels.cu) and the CPU oracle
// (reference_cpu.cpp) — and we want ONE implementation, not two that can
// silently drift apart. CUDA's __host__ __device__ qualifier lets a
// function compile for both targets, but reference_cpu.cpp is compiled by
// cl.exe, which has never heard of __host__/__device__ unless
// <cuda_runtime.h> is included. The HOSTDEV macro below resolves this: it
// expands to the real qualifiers under nvcc, and to plain `inline`
// everywhere else — so the SAME function body is reused by the GPU kernel
// and the CPU oracle, and a bug fixed in one is fixed in both.
//
// Units (CLAUDE.md §12, stated once, used everywhere):
//   lengths            meters (m)
//   magnetic field B   tesla (T)
//   current            amperes (A) — coil currents are AMPERE-TURNS
//                      (current x number of wire turns; see PRACTICE.md §2)
//   viscosity          pascal-seconds (Pa*s)
//   force              newtons (N)
//   time               seconds (s)
//   frame              right-handed 3D (x,y,z); the workspace/robots live
//                      in the z=0 plane; coil geometry is genuinely 3D.
//
// Read this after: main.cu.  Read this before: kernels.cu.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

// This header is SELF-CONTAINED: it uses sqrtf (biot_savart_contribution
// below) and must not rely on an includer having already pulled in the C
// math functions in the right order. reference_cpu.cpp, for instance,
// includes this header BEFORE its own <cmath> — a header that only works
// when included in one particular order is a bug waiting to bite the next
// file that includes it differently, so <cmath> is included right here.
#include <cmath>

// ---------------------------------------------------------------------------
// HOSTDEV — see the file header above. Under nvcc (__CUDACC__ is defined by
// the compiler itself, unconditionally, whenever nvcc processes a
// translation unit) this expands to CUDA's real dual-compilation qualifier;
// under cl.exe compiling reference_cpu.cpp, it degrades to a plain inline
// host function. Either way the FUNCTION BODY is identical — that identity
// is the whole point (one implementation, not two that can drift).
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HOSTDEV __host__ __device__ inline
#else
#define HOSTDEV inline
#endif

// ===========================================================================
// Fixed architectural constants.
//
// NUM_COILS is a true compile-time constant: the ratified teaching scope
// (CLAUDE.md §2, §13) is a 4-coil planar arrangement — two orthogonal
// "East/West" and "North/South" coil pairs — and every kernel signature
// below (Float4 per-coil currents, [NUM_COILS][grid_cells] basis-map
// layout) is written assuming exactly 4 coils. Everything ELSE about the
// scenario (grid resolution, coil radius, workspace size, fluid/bead
// parameters, the current schedule) is loaded at RUNTIME from
// data/sample/microswarm_scenario.csv into the SwarmScenario struct below —
// mirroring 24.01's MotorScenario pattern, and for the same reason: a
// physics demo's numbers should live in one runtime-loaded, regenerable
// place, not be duplicated as silently-driftable compile-time constants.
// ===========================================================================
constexpr int NUM_COILS = 4;   // coil id 0=East(+x) 1=West(-x) 2=North(+y) 3=South(-y) — see main.cu generate_coil_segments()

// mu0, the vacuum permeability (T*m/A). A physical constant, not a scenario
// parameter — it never varies, so (unlike coil geometry) it belongs here as
// a true compile-time constant. CODATA value to float precision.
constexpr float MU0_T_M_PER_A = 1.25663706212e-6f;

// ---------------------------------------------------------------------------
// CoilSegment — one straight sub-segment of a discretized circular coil.
//
// Each of the 4 coils is approximated as a regular polygon of
// `segs_per_coil` straight segments inscribed in the true circle (radius
// coil_radius_m) — the "180 current segments" the catalog bullet's ratified
// scope names. Biot-Savart is then a SUM over these straight elements
// (THEORY.md "The math"), which is exactly representable on a GPU as a
// map-then-reduce: one thread per field point, loop over every segment.
//
// Fields:
//   mx,my,mz   — the segment's MIDPOINT (m): the source point used in the
//                discretized Biot-Savart law (see biot_savart_contribution
//                below) — using the midpoint rather than an endpoint halves
//                the discretization error for a given segment count (the
//                midpoint rule is second-order accurate; an endpoint rule
//                is only first-order).
//   dlx,dly,dlz — the segment VECTOR (p1 - p0, m): its length is the
//                segment's arc-chord length, and its DIRECTION encodes the
//                coil's current sense — flip every segment's dl and the
//                coil's field reverses, exactly as flipping physical wire
//                current would.
//   coil_id    — which coil (0..NUM_COILS-1) this segment belongs to; the
//                basis-map kernel uses this to decide whether a segment
//                contributes to the basis map currently being built.
// ---------------------------------------------------------------------------
struct CoilSegment {
    float mx, my, mz;      // segment midpoint (m) — Biot-Savart source point
    float dlx, dly, dlz;   // segment vector p1-p0 (m) — length + current direction
    int   coil_id;         // 0..NUM_COILS-1
};

// ---------------------------------------------------------------------------
// Vec3 — a plain 3-float vector. Used only by the host-side analytic-gate
// machinery (main.cu's on-axis / Helmholtz / divergence checks call
// biot_savart_point_cpu, which returns one of these) — deliberately NOT
// used inside the hot GPU kernels below, which keep every field component
// as a separate named float (the repo's general style for kernel-local
// state: see 08.01 kernels.cuh's note on why registers, not structs,
// dominate kernel-local hot paths).
// ---------------------------------------------------------------------------
struct Vec3 { float x, y, z; };

// ---------------------------------------------------------------------------
// Float4 — a plain, CUDA-independent 4-float bundle for "one current per
// coil" (NUM_COILS=4). CUDA ships a built-in `float4` vector type, but it
// lives in <vector_types.h> (pulled in by <cuda_runtime.h>) — a header
// reference_cpu.cpp deliberately never includes (CLAUDE.md §5: the CPU
// oracle must not depend on nvcc/the CUDA toolkit at all). Defining our own
// 4-float struct keeps combine_field_kernel/_cpu callable from BOTH the
// nvcc-compiled kernels.cu and the cl.exe-compiled reference_cpu.cpp with
// the exact same type, no CUDA header required on the host-only side.
// ---------------------------------------------------------------------------
struct Float4 { float x, y, z, w; };   // per-coil currents: x=East y=West z=North w=South

// ---------------------------------------------------------------------------
// SwarmScenario — every runtime-loaded parameter of this project's demo,
// read from data/sample/microswarm_scenario.csv by main.cu's load_scenario()
// (reference_cpu.cpp and kernels.cu never touch the CSV; they take plain
// numbers, so the SAME scenario always drives the SAME GPU and CPU paths).
//
// Grouped by physical subsystem; every field documents its unit at the
// point of use elsewhere in this header, so only the non-obvious ones are
// re-noted here.
// ---------------------------------------------------------------------------
struct SwarmScenario {
    // -- coil geometry --
    int   grid_n            = 0;     // field-map resolution per axis (256 in the sample)
    float coil_radius_m     = 0.0f;  // R: radius of each of the 4 coils (m)
    float coil_offset_m     = 0.0f;  // distance from the workspace origin to each coil's center, along its own axis (m)
    int   segs_per_coil     = 0;     // straight segments approximating each coil's circle (180 in the sample)
    float workspace_half_m  = 0.0f;  // half-width of the square workspace the field map covers (m)

    // -- fluid + bead (THEORY.md "The problem" derives every one of these) --
    float mu_fluid_pa_s     = 0.0f;  // dynamic viscosity of the carrier fluid (Pa*s; water ~1e-3)
    float bead_radius_m     = 0.0f;  // a: microrobot bead (or bead-cluster) radius (m)
    float chi_eff           = 0.0f;  // dimensionless effective volume-susceptibility contrast (bead - fluid)

    // -- drive + schedule --
    float I0_ampere_turns   = 0.0f;  // per-coil drive magnitude used by every phase of the waypoint schedule (A, ampere-turns)
    float dt_s              = 0.0f;  // explicit-Euler integration step (s)
    int   steps_per_phase   = 0;     // Euler steps executed per waypoint-schedule phase

    // -- swarm --
    int          n_robots      = 0;     // number of simulated microrobots
    float        init_spread_m = 0.0f;  // std-dev of the initial Gaussian cluster around the workspace origin (m)
    unsigned int seed          = 0;     // host RNG seed for the initial cluster (determinism, CLAUDE.md §12)

    bool loaded = false;   // false until load_scenario() (main.cu) successfully parses every required row

    int grid_cells() const { return grid_n * grid_n; }             // total field-map cells
    int n_segments() const { return segs_per_coil * NUM_COILS; }   // total coil segments (720 in the sample)
};

// ===========================================================================
// Shared HOSTDEV numerical helpers — ONE implementation, used by both the
// GPU kernels (kernels.cu) and the CPU oracle (reference_cpu.cpp). See the
// file header for why this is safe across the host/device compiler split.
// ===========================================================================

// grid_to_world — map a grid index (0..grid_n-1) along one axis to its
// world-space coordinate (m), given the workspace is centered on the
// origin and spans [-half, +half]. Used identically for x and y (the
// workspace is square) — callers pass the axis-appropriate index.
HOSTDEV float grid_to_world(int idx, int grid_n, float half_m)
{
    // idx=0 -> -half, idx=grid_n-1 -> +half; linear in between. grid_n-1 in
    // the denominator (not grid_n) makes the two ENDPOINTS of the grid land
    // exactly on the two workspace edges — the standard "inclusive linspace"
    // convention (matches numpy.linspace, used to prototype this project).
    return -half_m + (2.0f * half_m) * (static_cast<float>(idx) / static_cast<float>(grid_n - 1));
}

// world_to_grid_frac — the inverse of grid_to_world: a world coordinate (m)
// to a FRACTIONAL grid index (e.g. 3.5 = "halfway between cell 3 and 4").
// Used by bilinear_sample below to locate the 2x2 neighborhood to blend.
HOSTDEV float world_to_grid_frac(float coord_m, int grid_n, float half_m)
{
    return (coord_m + half_m) / (2.0f * half_m) * static_cast<float>(grid_n - 1);
}

// biot_savart_contribution — the discretized Biot-Savart law for ONE
// straight current segment, evaluated at ONE field point:
//
//     dB = (mu0 / 4*pi) * I * (dl x r) / |r|^3
//
// exactly the formula the catalog bullet's ratified scope names. dl is the
// segment's vector (direction = current sense, magnitude = segment length);
// r is the vector FROM the segment's midpoint TO the field point; the cross
// product dl x r gives a field perpendicular to both, falling off as 1/r^2
// weighted by the perpendicular component of dl (the familiar right-hand
// rule "field circles the wire" picture, applied to one small straight
// piece of a much longer discretized loop).
//
// Numerical guard (THEORY.md "Numerical considerations" explains why this
// is defensive, not load-bearing, for THIS project's geometry): a tiny
// epsilon is added to r^2 before cubing, so a field point that ever lands
// exactly ON a segment's midpoint (r=0) produces a large-but-finite value
// instead of a NaN from 0/0. In this project's geometry the workspace
// (|coord| <= workspace_half_m) never gets closer than ~(coil_offset_m -
// workspace_half_m) to any coil segment, so the guard is never actually
// exercised at the values in the committed scenario — it is included
// because a general Biot-Savart routine should never trust its caller's
// geometry to keep it safe, and this is the standard technique (the same
// role a softening length plays in N-body gravity kernels).
HOSTDEV void biot_savart_contribution(
    float mx, float my, float mz,      // IN: segment midpoint (m)
    float dlx, float dly, float dlz,   // IN: segment vector (m); encodes current direction
    float I,                            // IN: current on this segment (A, ampere-turns; 0 if the segment's coil is inactive)
    float px, float py, float pz,      // IN: field evaluation point (m)
    float& Bx, float& By, float& Bz)   // OUT: accumulated INTO (caller zero-inits, we +=)
{
    const float rx = px - mx, ry = py - my, rz = pz - mz;   // r: midpoint -> field point (m)
    const float r2 = rx * rx + ry * ry + rz * rz + 1e-12f;  // |r|^2 + epsilon guard (m^2; epsilon negligible at our scales)
    const float inv_r3 = 1.0f / (r2 * sqrtf(r2));           // 1/|r|^3 (m^-3) — one sqrt, one divide, reused for all 3 components

    // Cross product dl x r (the "current element x displacement" that
    // Biot-Savart says a moving-charge/current element produces a field
    // perpendicular to).
    const float cx = dly * rz - dlz * ry;
    const float cy = dlz * rx - dlx * rz;
    const float cz = dlx * ry - dly * rx;

    const float scale = (MU0_T_M_PER_A / (4.0f * 3.14159265358979323846f)) * I * inv_r3;
    Bx += scale * cx;
    By += scale * cy;
    Bz += scale * cz;
}

// bilinear_sample — sample a grid_n x grid_n scalar field at an arbitrary
// world-space point (x,y) via bilinear interpolation of its 4 enclosing
// grid cells. `field` is row-major with index = iy*grid_n + ix (iy is the
// SLOWER-varying "row" index, matching the field-map layout every kernel
// below uses — documented once here, honored everywhere).
//
// Numerical guard: (x,y) is CLAMPED to the grid's valid coordinate range
// before converting to fractional indices. This is a SEPARATE guard from
// the Biot-Savart 1/r^3 guard above — it protects against an OUT-OF-BOUNDS
// MEMORY READ (a robot that has drifted past the mapped workspace edge)
// rather than a numerical singularity; THEORY.md "Numerical considerations"
// discusses both. This project's tuned parameters keep every simulated
// robot comfortably inside the mapped region (see README "Expected
// output"), so the clamp is, again, a defensive floor rather than a
// frequently-hit code path.
HOSTDEV float bilinear_sample(const float* field, int grid_n, float half_m, float x, float y)
{
    float fx = world_to_grid_frac(x, grid_n, half_m);
    float fy = world_to_grid_frac(y, grid_n, half_m);
    const float max_idx = static_cast<float>(grid_n - 1) - 1e-4f;  // leave room for the "+1" neighbor read below
    if (fx < 0.0f) fx = 0.0f; else if (fx > max_idx) fx = max_idx;
    if (fy < 0.0f) fy = 0.0f; else if (fy > max_idx) fy = max_idx;

    const int ix0 = static_cast<int>(fx);      // floor via truncation (fx >= 0 here, so this IS floor)
    const int iy0 = static_cast<int>(fy);
    const float tx = fx - static_cast<float>(ix0);   // fractional offset within the cell, x
    const float ty = fy - static_cast<float>(iy0);   // fractional offset within the cell, y

    const float v00 = field[iy0 * grid_n + ix0];
    const float v10 = field[iy0 * grid_n + (ix0 + 1)];
    const float v01 = field[(iy0 + 1) * grid_n + ix0];
    const float v11 = field[(iy0 + 1) * grid_n + (ix0 + 1)];

    // Standard bilinear blend: interpolate along x at both rows, then blend
    // the two rows along y. Four multiplies, three adds, no branches — a
    // clean, cheap operation to run twice (Bx,By fields become gx,gy fields
    // by the time this function is called from swarm_step) per robot per step.
    const float v0 = v00 * (1.0f - tx) + v10 * tx;
    const float v1 = v01 * (1.0f - tx) + v11 * tx;
    return v0 * (1.0f - ty) + v1 * ty;
}

// ===========================================================================
// GPU kernels (device-only declarations; nvcc sees these, cl.exe does not).
// Full documentation (thread mapping, launch configuration, memory
// behavior) lives with each DEFINITION in kernels.cu — headers carry the
// one-line summary from the pipeline overview above.
// ===========================================================================
#ifdef __CUDACC__

// One thread per field-evaluation grid cell; each thread sums the
// contribution of every one of the n_segs coil segments whose coil_id ==
// active_coil, at unit current — producing ONE basis map (Bx,By), T per
// ampere-turn, for that one coil. Called NUM_COILS times.
__global__ void biot_savart_basis_kernel(
    const CoilSegment* __restrict__ segs, int n_segs, int active_coil,
    int grid_n, float half_m,
    float* __restrict__ Bx, float* __restrict__ By);

// One thread per grid cell; combined = sum_c I_coil[c] * basis[c] — the
// linearity-of-Maxwell "map" kernel described in the file header.
// basisBx/basisBy are [NUM_COILS][grid_cells] flattened, coil-major.
__global__ void combine_field_kernel(
    const float* __restrict__ basisBx, const float* __restrict__ basisBy,
    Float4 I_coil, int grid_cells,
    float* __restrict__ Bx, float* __restrict__ By);

// One thread per grid cell; 4-neighbor central-difference stencil producing
// d(|B|^2)/dx and d(|B|^2)/dy (the force-generating gradient; THEORY.md
// "The math" derives why |B|^2's gradient, not B's, is what matters here).
__global__ void gradient_b2_kernel(
    const float* __restrict__ Bx, const float* __restrict__ By,
    int grid_n, float half_m,
    float* __restrict__ gx, float* __restrict__ gy);

// One thread per ROBOT; each thread runs `steps` explicit-Euler
// sub-steps in a register-resident loop (the agent-farm pattern this
// repo's swarm/rollout kernels share — see 08.01, 22.01).
__global__ void swarm_step_kernel(
    const float* __restrict__ gx, const float* __restrict__ gy,
    int grid_n, float half_m,
    float* __restrict__ rx, float* __restrict__ ry, int n_robots,
    float k_force, float gamma, float dt_s, int steps);

#endif // __CUDACC__

// ---------------------------------------------------------------------------
// Host launch wrappers — own the grid/block math + the post-launch error
// check (CLAUDE.md §6.1 rule 7), so main.cu's orchestration code never
// touches a <<<...>>> literal. Declared OUTSIDE the __CUDACC__ fence: these
// are plain host functions callable from any translation unit; only their
// DEFINITIONS (in kernels.cu) need nvcc.
// ---------------------------------------------------------------------------
void launch_biot_savart_basis(const CoilSegment* d_segs, int n_segs, int active_coil,
                              int grid_n, float half_m, float* d_Bx, float* d_By);
void launch_combine_field(const float* d_basisBx, const float* d_basisBy, Float4 I_coil,
                          int grid_n, float* d_Bx, float* d_By);
void launch_gradient_b2(const float* d_Bx, const float* d_By, int grid_n, float half_m,
                        float* d_gx, float* d_gy);
void launch_swarm_step(const float* d_gx, const float* d_gy, int grid_n, float half_m,
                       float* d_rx, float* d_ry, int n_robots,
                       float k_force, float gamma, float dt_s, int steps);

// ---------------------------------------------------------------------------
// CPU oracle twins (defined in reference_cpu.cpp) — line-by-line sequential
// versions of the 4 kernels above, sharing the HOSTDEV helpers so the ONLY
// algorithmic difference is "loop over cells/robots" vs. "one thread per
// cell/robot" (CLAUDE.md §5: the CPU reference is what makes the GPU
// speed-up legible, and the correctness oracle main.cu's VERIFY stages
// check the GPU kernels against).
// ---------------------------------------------------------------------------
void biot_savart_basis_cpu(const CoilSegment* segs, int n_segs, int active_coil,
                           int grid_n, float half_m, float* Bx, float* By);
void combine_field_cpu(const float* basisBx, const float* basisBy, Float4 I_coil,
                       int grid_n, float* Bx, float* By);
void gradient_b2_cpu(const float* Bx, const float* By, int grid_n, float half_m,
                     float* gx, float* gy);
void swarm_step_cpu(const float* gx, const float* gy, int grid_n, float half_m,
                    float* rx, float* ry, int n_robots,
                    float k_force, float gamma, float dt_s, int steps);

// biot_savart_point_cpu — the general-purpose, ungridded Biot-Savart sum at
// ONE arbitrary 3D point, over an arbitrary per-coil current vector. This
// is the building block main.cu's three ANALYTIC gates use (on-axis single
// loop, Helmholtz-pair flatness, full-3D divergence sanity) — each gate
// evaluates the SAME underlying physics at hand-picked points/configurations
// chosen to have a known closed-form or symmetry answer, independent of the
// 256x256 grid map entirely (so a grid-resolution bug could never
// accidentally make an analytic gate pass). biot_savart_basis_cpu (the
// gridded CPU oracle above) is implemented BY CALLING this function at
// every grid point — one code path, two use sites.
Vec3 biot_savart_point_cpu(const CoilSegment* segs, int n_segs,
                           const float I_coil[NUM_COILS],
                           float px, float py, float pz);

#endif // PROJECT_KERNELS_CUH
