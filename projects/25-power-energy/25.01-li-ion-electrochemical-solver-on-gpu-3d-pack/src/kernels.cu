// ===========================================================================
// kernels.cu — GPU implementation for project 25.01
//              Li-ion electrochemical (SPM) solver + 3D pack thermal solve
//              (teaching core: two batched explicit finite-volume stencils)
//
// The big idea
// ------------
// This project teaches TWO coupled PDEs, both solved the same way a GPU
// likes: turn "march a stencil forward in time, everywhere, every step"
// into "one thread per independent grid point, all threads at once"
// (07.09/31.01/24.01's pattern, applied twice):
//
//   1. electrochem_fv_kernel — solid-state lithium diffusion INSIDE one
//      spherical particle per electrode per cell. The "grid" here is a
//      1-D radial shell index; the BATCH axis folds together every design,
//      cell, and electrode (B*kNCells*2 independent spheres), so one flat
//      launch solves all of them at once — exactly 24.01's "many small
//      independent problems is as GPU-friendly as one big problem" lesson,
//      now with B*kNCells*2 = up to 576 problems instead of 24.01's ~24.
//   2. thermal_step_kernel — the pack's 3-D heat equation, batched the same
//      way 24.01 batches rotor angles: the SAME kx/ky/kz/rho_cp medium, B
//      independent (h, cooling-face) designs, all advanced one timestep in
//      a single launch.
//
// What is NEW here beyond 24.01/31.01's single-PDE stencils:
//   * a SPHERICAL (not Cartesian) finite-volume geometry — the face areas
//     and shell volumes grow with r^2/r^3, so the stencil coefficients are
//     NOT uniform (unlike a Cartesian Laplacian's constant 1/dx^2) — THEORY.md
//     derives the coefficients from first principles;
//   * a NEUMANN (flux) boundary condition carrying the actual applied
//     current, instead of the Dirichlet A=0 / freezing conditions the
//     other grid projects use;
//   * TWO independent batched PDEs in one project, coupled only through
//     host-side per-step bookkeeping (main.cu) — deliberately LOOSE/LAGGED
//     coupling, documented, not a monolithic fused kernel (THEORY.md
//     "The GPU mapping" argues why that would teach less, not more, here).
//
// All layouts and constants come from kernels.cuh — the single source
// shared with the CPU twin; both kernels below are deliberate line-by-line
// twins of the functions in reference_cpu.cpp.
//
// Read this after: kernels.cuh.  Companion twin: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"      // CUDA_CHECK / CUDA_CHECK_LAST_ERROR (CLAUDE.md §6.1 rule 7)

#include <cstdio>
#include <cstdlib>

// A device-side pi (host code in main.cu uses its own double kPi; the
// kernels only ever need float precision — FP32 throughout, per CLAUDE.md §5).
static constexpr float kPiF = 3.14159265358979323846f;

// ===========================================================================
// electrochem_fv_kernel — one thread = one (particle, shell) FV update.
//
// Thread-to-data mapping: flat index idx = blockIdx.x*blockDim.x+threadIdx.x
// covers [0, B*kNCells*2*kNShells). We decode it as
//     s = idx % kNShells                 shell index (FASTEST axis)
//     p = idx / kNShells                 particle index, p in [0, B*kNCells*2)
//     e = p % 2                          0 = anode, 1 = cathode
// exactly the kernels.cuh layout contract. Because s is the fastest axis and
// a warp's 32 consecutive threads therefore share the SAME particle (s runs
// 0..19 within one particle before p increments — kNShells=20 is close to
// a warp), most of a warp's global reads land in one contiguous 20-float
// span: not a full 128-byte coalesced transaction, but far better than a
// per-particle-strided layout would give, and simple enough that the
// derivation above stays legible (a genuinely coalescing-optimal layout
// would need shell-major-across-particles storage, which would scatter
// the physically-local shell-to-shell stencil reads instead — README
// Exercise 4 profiles the trade).
//
// The math this implements (full derivation in THEORY.md "The algorithm";
// summary here so the kernel reads standalone). Spherical finite volume
// with kNShells cell-centered shells of uniform thickness dr = R_p/kNShells:
// shell s occupies [s*dr, (s+1)*dr), volume V_s = (4/3)*pi*(r_out^3-r_in^3),
// with FACE areas A_in = 4*pi*r_in^2, A_out = 4*pi*r_out^2. Flux (mol/(m^2 s),
// positive in +r direction) at an INTERNAL face between two shell centers a
// distance dr apart is Fick's law F = -D*(c_far - c_near)/dr; at the
// particle's very center F=0 (symmetry: no flux crosses r=0); at the
// particle's outer surface F = j (the imposed Neumann boundary condition,
// this step's applied current translated to a molar flux by main.cu).
// Accumulation in shell s: V_s * dc_s/dt = A_in*F_in - A_out*F_out (flux IN
// through the inner face minus flux OUT through the outer face — the FV
// statement of Fick's second law in spherical coordinates).
//
// Memory spaces per thread:
//   registers : ~15 (D, j, dr, radii, both face areas, both fluxes)
//   global    : up to 3 reads from c_in (self, and the ONE existing
//               neighbor shell at each face — a boundary shell has only
//               one), 1 read from D, 1 write to c_out. No shared memory:
//               each shell value is reused by at most its 2 immediate
//               neighbors, and the L2 covers that reuse at this tiny
//               working-set size (kNShells=20 floats per particle) — the
//               same "L2 is enough at this size" call 07.09/31.01/24.01 make.
// No atomics; no divergence beyond the two boundary-shell branches (s==0,
// s==kNShells-1), which every particle takes at the SAME two shell indices,
// so within a warp (which shares one particle, see above) divergence only
// ever touches 1-2 of the 20 lanes — cheap, predicated, not serialized.
// ===========================================================================
__global__ void electrochem_fv_kernel(int B,
                                      ElectrodeGeom geomA, ElectrodeGeom geomC,
                                      float j_a, float j_c, float dt_e,
                                      const float* __restrict__ D,     // [B*kNCells*2] this step's Arrhenius-scaled diffusivity (m^2/s)
                                      const float* __restrict__ c_in,  // [B*kNCells*2*kNShells] concentration IN (mol/m^3)
                                      float*       __restrict__ c_out) // [B*kNCells*2*kNShells] concentration OUT (mol/m^3)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * kNCells * 2 * kNShells;
    if (idx >= total) return;                         // ragged-tail guard

    const int s = idx % kNShells;                      // this thread's shell (fast axis)
    const int p = idx / kNShells;                       // this thread's particle index
    const int e = p % 2;                                 // 0 = anode, 1 = cathode

    // Pick this particle's geometry/kinetics and boundary flux by electrode.
    // geomA/geomC arrive BY VALUE (tiny structs, constant-cache-backed —
    // the same reasoning 24.01's FeaGrid-by-value documents), so this branch
    // costs a predicated select, not a memory access.
    const ElectrodeGeom geom = (e == 0) ? geomA : geomC;
    const float j = (e == 0) ? j_a : j_c;
    const float Dp = D[p];                              // this particle's CURRENT diffusivity (Arrhenius-scaled by main.cu from its cell's temperature)

    const float dr = geom.R_p / static_cast<float>(kNShells);   // shell thickness (m)
    const float r_in  = static_cast<float>(s) * dr;              // this shell's inner face radius (m)
    const float r_out = static_cast<float>(s + 1) * dr;          // this shell's outer face radius (m)

    // Shell volume and face areas — the genuinely NEW ingredient versus a
    // Cartesian stencil (07.09/31.01/24.01): these are NOT constant across
    // shells, they grow with r^2 (area) / r^3 (volume), which is exactly
    // why lithium concentration profiles in a particle are curved even
    // under a perfectly uniform diffusivity (THEORY.md derives the
    // quasi-steady parabolic shape used by this project's analytic gate).
    const float V_s   = (4.0f / 3.0f) * kPiF * (r_out * r_out * r_out - r_in * r_in * r_in);
    const float A_in   = 4.0f * kPiF * r_in  * r_in;
    const float A_out  = 4.0f * kPiF * r_out * r_out;

    const float c0 = c_in[idx];                          // this shell's current concentration (mol/m^3)

    // Inner face flux: zero at the particle's geometric center (s==0 has no
    // inner neighbor — r_in==0 makes A_in==0 anyway, so this branch is only
    // needed to avoid reading c_in[idx-1] out of the particle's own block).
    float F_in = 0.0f;
    if (s > 0) F_in = -Dp * (c0 - c_in[idx - 1]) / dr;    // Fick's law between shell centers s-1 and s

    // Outer face flux: the imposed surface flux j at the LAST shell (the
    // Neumann boundary condition this step's applied current sets); Fick's
    // law between shell centers s and s+1 everywhere else.
    float F_out;
    if (s == kNShells - 1) F_out = j;
    else                    F_out = -Dp * (c_in[idx + 1] - c0) / dr;

    // FV accumulation: V*dc/dt = flux IN (inner face) - flux OUT (outer
    // face). Forward Euler over this call's dt_e (main.cu chooses dt_e well
    // inside the diffusion-CFL bound derived in THEORY.md "Numerical
    // considerations" — the innermost shell's coefficient, ~3D/dr^2, sets
    // the tightest limit because its volume shrinks fastest as r->0).
    const float dcdt = (A_in * F_in - A_out * F_out) / V_s;
    c_out[idx] = c0 + dt_e * dcdt;
}

// ===========================================================================
// thermal_step_kernel — one thread = one (design, voxel) FV update of the
// anisotropic 3-D pack heat equation.
//
// Thread-to-data mapping: thread (i, j, b, k) where i = blockIdx.x*blockDim.x
// +threadIdx.x (x, fast axis), j = blockIdx.y*blockDim.y+threadIdx.y (y),
// and blockIdx.z is DECODED into (b, k) = (blockIdx.z / kTNZ, blockIdx.z %
// kTNZ) — folding the batch axis and the z axis into CUDA's single z grid
// dimension, extending 24.01's "batch rides in blockIdx.z" idiom from 2-D to
// 3-D. idx = ((b*kTNZ+k)*kTNY+j)*kTNX+i, the kernels.cuh layout contract.
//
// The math (full derivation, including the boundary treatment, in
// THEORY.md "The algorithm"; summary here so the kernel reads standalone).
// Per axis, the FLUX DIVERGENCE (a volumetric term, W/m^3) is:
//   * INTERIOR voxel (a neighbor exists on both sides): the standard
//     anisotropic second difference,  k_axis*(T_minus - 2*T0 + T_plus)/d^2.
//   * BOUNDARY voxel on a face that is NOT this design's cooling face
//     (5 of the pack's 6 faces, always): a ZERO-GRADIENT ("adiabatic")
//     ghost, T_ghost = T0, which collapses the same formula to the
//     one-sided  k_axis*(T_neighbor - T0)/d^2 — algebraically the same
//     expression, just with the missing term dropped (substituting
//     T_ghost=T0 into (T_ghost - 2T0 + T_neighbor) gives exactly
//     (T_neighbor - T0)).
//   * BOUNDARY voxel on THIS design's cooling face: the one-sided
//     conduction term above PLUS a Robin convective term
//     h*(T_coolant - T0)/d — a volumetric-equivalent source spread over
//     one voxel's thickness, standard for a cell-centered (non-staggered)
//     grid (THEORY.md derives the units and the "half control volume
//     touches the coolant directly" reasoning).
// Only ONE of the 6 faces is ever a design's cooling face (kCoolBottomZ:
// k==0; kCoolSideX: i==0) — every other boundary voxel takes the adiabatic
// branch. Summing all three axes' contributions and this step's volumetric
// heat source q_vol, then dividing by rho*cp, gives dT/dt (K/s); forward
// Euler over dt_thermal (checked against the CFL bound derived below and in
// THEORY.md, and printed by main.cu before the mission loop even starts).
//
// Memory spaces per thread:
//   registers : T0 + up to 6 neighbor reads + a handful of scratch floats
//   global    : up to 6 reads from T_in (fewer at boundaries), 1 read from
//               q_vol, 1 read from DesignPoint (tiny, per-block-uniform —
//               every thread in a (i,j) tile at fixed b reads the SAME
//               designs[b], served by broadcast/L1 after the first touch),
//               1 write to T_out. No shared memory: each voxel's value is
//               reused by at most 6 neighbor threads and the L2 covers that
//               reuse at this grid size (07.09/31.01/24.01's same call).
// No atomics; divergence only at the domain's outer boundary voxels (a
// small fraction of threads at this grid size) and is cheap/predicated.
// ===========================================================================
__global__ void thermal_step_kernel(int B, PackThermalParams p, float dt_thermal,
                                    const DesignPoint* __restrict__ designs, // [B]
                                    const float* __restrict__ q_vol,          // [B*kTNZ*kTNY*kTNX] W/m^3
                                    const float* __restrict__ T_in,           // [B*kTNZ*kTNY*kTNX] K
                                    float*       __restrict__ T_out)          // [B*kTNZ*kTNY*kTNX] K, OUT
{
    const int i  = blockIdx.x * blockDim.x + threadIdx.x;   // x index (fast axis)
    const int j  = blockIdx.y * blockDim.y + threadIdx.y;   // y index
    const int bk = blockIdx.z;                               // encodes (b, k)
    const int k  = bk % kTNZ;
    const int b  = bk / kTNZ;
    if (i >= kTNX || j >= kTNY || b >= B) return;            // ragged tile / batch guard

    const int idx = ((b * kTNZ + k) * kTNY + j) * kTNX + i;   // this voxel's flat index (kernels.cuh layout)
    const float T0 = T_in[idx];
    const DesignPoint dp = designs[b];                          // this design's (h, cooling face)

    float flux_div = 0.0f;   // accumulated volumetric flux divergence (W/m^3); see header derivation

    // ---- x axis --------------------------------------------------------
    if (i > 0 && i < kTNX - 1) {
        flux_div += p.kx * (T_in[idx - 1] - 2.0f * T0 + T_in[idx + 1]) / (p.dx * p.dx);
    } else {
        // Exactly one neighbor exists; pick it.
        const float Tn = (i == 0) ? T_in[idx + 1] : T_in[idx - 1];
        flux_div += p.kx * (Tn - T0) / (p.dx * p.dx);
        // The x=0 face is THIS design's cooling face only if face==kCoolSideX
        // AND we are actually sitting on that face (i==0 — x=kTNX-1 stays
        // adiabatic even for a side-cooled design: a single side plate,
        // not two, is the design being taught, THEORY.md "The problem").
        if (i == 0 && dp.face == kCoolSideX)
            flux_div += dp.h * (p.T_coolant - T0) / p.dx;
    }

    // ---- y axis (always adiabatic — neither design cools a y face) -----
    if (j > 0 && j < kTNY - 1) {
        flux_div += p.ky * (T_in[idx - kTNX] - 2.0f * T0 + T_in[idx + kTNX]) / (p.dy * p.dy);
    } else {
        const float Tn = (j == 0) ? T_in[idx + kTNX] : T_in[idx - kTNX];
        flux_div += p.ky * (Tn - T0) / (p.dy * p.dy);
    }

    // ---- z axis ----------------------------------------------------------
    const int planeStride = kTNX * kTNY;
    if (k > 0 && k < kTNZ - 1) {
        flux_div += p.kz * (T_in[idx - planeStride] - 2.0f * T0 + T_in[idx + planeStride]) / (p.dz * p.dz);
    } else {
        const float Tn = (k == 0) ? T_in[idx + planeStride] : T_in[idx - planeStride];
        flux_div += p.kz * (Tn - T0) / (p.dz * p.dz);
        if (k == 0 && dp.face == kCoolBottomZ)
            flux_div += dp.h * (p.T_coolant - T0) / p.dz;
    }

    // Add this step's volumetric heat generation and integrate forward
    // (explicit Euler — the same choice 31.01 makes, for the same reason:
    // a stencil this cheap per cell is dominated by memory traffic, not
    // arithmetic, so an implicit solve would spend far more bandwidth for
    // a stability margin this project does not need — see the CFL check
    // main.cu prints before the mission loop starts).
    const float dTdt = (flux_div + q_vol[idx]) / p.rho_cp;
    T_out[idx] = T0 + dt_thermal * dTdt;
}

// ===========================================================================
// Host launchers (declared in kernels.cuh).
// ===========================================================================

void launch_electrochem_substep(int B, ElectrodeGeom geomA, ElectrodeGeom geomC,
                                float j_a, float j_c, float dt_e,
                                const float* d_D,
                                const float* d_c_in, float* d_c_out)
{
    if (B < 1 || !d_D || !d_c_in || !d_c_out || dt_e <= 0.0f) {
        std::fprintf(stderr, "launch_electrochem_substep: invalid arguments (B=%d dt_e=%g)\n",
                     B, static_cast<double>(dt_e));
        std::exit(EXIT_FAILURE);
    }
    const int total = B * kNCells * 2 * kNShells;
    const int threads = 256;                              // repo default block size (warp multiple)
    const int blocks = (total + threads - 1) / threads;    // ceil: cover every (particle,shell)

    electrochem_fv_kernel<<<blocks, threads>>>(B, geomA, geomC, j_a, j_c, dt_e, d_D, d_c_in, d_c_out);
    CUDA_CHECK_LAST_ERROR("electrochem_fv_kernel launch");
}

void launch_thermal_substep(int B, const PackThermalParams& p, float dt_thermal,
                            const DesignPoint* d_designs,
                            const float* d_q_vol,
                            const float* d_T_in, float* d_T_out)
{
    if (B < 1 || !d_designs || !d_q_vol || !d_T_in || !d_T_out || dt_thermal <= 0.0f) {
        std::fprintf(stderr, "launch_thermal_substep: invalid arguments (B=%d dt=%g)\n",
                     B, static_cast<double>(dt_thermal));
        std::exit(EXIT_FAILURE);
    }
    const dim3 block(8, 8, 1);                              // 8x8=64 threads/tile: kTNX=32,kTNY=24 tile exactly
    const dim3 grid(kTNX / 8, kTNY / 8, static_cast<unsigned int>(B) * kTNZ);   // z folds (design, k)

    thermal_step_kernel<<<grid, block>>>(B, p, dt_thermal, d_designs, d_q_vol, d_T_in, d_T_out);
    CUDA_CHECK_LAST_ERROR("thermal_step_kernel launch");
}
