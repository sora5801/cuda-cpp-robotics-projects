// ===========================================================================
// kernels.cu — GPU implementation for project 31.01
//              Hamilton-Jacobi reachability: level-set grid solvers
//              (teaching core: double-integrator backward reachable tube)
//
// The big idea
// ------------
// Reachability asks the SAFETY question exhaustively: "from which states can
// the robot still reach the target set within T seconds, over ALL possible
// controls?" Dynamic programming compresses that infinite family of
// trajectories into ONE scalar field V(x,v,t) obeying a Hamilton-Jacobi PDE;
// solving the PDE backward in time makes the zero sublevel set of V sweep
// outward from the target exactly along the optimal trajectories. No
// sampling, no gaps — the PDE *is* the proof (THEORY.md §problem).
//
// Numerically this is a STENCIL computation — 07.09's pattern with real PDE
// math in the stencil body: every sweep, every cell combines itself with its
// 4 face neighbors through the LAX-FRIEDRICHS numerical Hamiltonian and
// steps dt further back in time. One thread per cell, ping-pong buffers
// (read a consistent snapshot, write the next one), a few hundred sweeps.
//
// What is NEW here beyond 07.09's grid pattern:
//   * a real PDE: one-sided (upwind) difference PAIRS p^-, p^+ per axis,
//     a Hamiltonian evaluated at their average, and an artificial-
//     dissipation term that steers the scheme to the physically correct
//     (viscosity) solution at kinks — the front has corners ON the
//     bang-bang switching curve, and naive central differences would
//     oscillate and diverge there (THEORY.md §algorithm);
//   * a CFL-limited timestep: unlike JFA's "any schedule works", an
//     explicit PDE step is only STABLE while information moves less than
//     one cell per sweep — dt is not a knob, it is a law (§numerics);
//   * FREEZING (min with 0): the value may only DECREASE, which turns the
//     "reach exactly at time T" solution into the "reach within T" TUBE —
//     the set robotics actually wants (§the-math).
//
// All layouts and constants come from kernels.cuh — the single source
// shared with the CPU twin; the update expression below is a deliberate
// line-by-line twin of the one in reference_cpu.cpp.
//
// Read this after: kernels.cuh.  Companion twin: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"      // CUDA_CHECK / CUDA_CHECK_LAST_ERROR (§6.1 rule 7)

#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Launch geometry: 2-D blocks of 16x16 = 256 threads (the repo default
// thread count, arranged as a square tile because the DATA is 2-D and the
// stencil reaches along both axes). threadIdx.x maps to i (the fast/x axis)
// so each warp reads 32 consecutive floats of a row — coalesced; mapping
// threadIdx.x to j instead would stride reads nx floats apart (the classic
// grid-kernel mistake, called out in kernels.cuh and 07.09).
// ---------------------------------------------------------------------------
static constexpr int kTile = 16;

// ===========================================================================
// The HJ sweep kernel: one thread = one grid cell = one PDE update.
//
// Thread-to-data mapping: thread (i, j) = (blockIdx*blockDim + threadIdx)
// owns cell (i, j) — value V[j*nx + i]; guards handle grids that are not
// multiples of 16.
//
// The math this implements (full derivation in THEORY.md; summary here so
// the code reads standalone). We integrate the tube PDE forward in
// BACKWARD-time tau = -t:
//
//     dV/dtau = min( 0,  H(v, pbar) + dissipation )
//     H(v, p) = min over |u|<=umax of p.f(x,v,u) = px*v - umax*|pv|
//
//   * px, pv approximated by one-sided pairs (p^-, p^+) per axis; H is
//     evaluated at their AVERAGE pbar.
//   * dissipation = ax/2*(px+ - px-) + av/2*(pv+ - pv-), with
//     ax = |v_j| (EXACT: dH/dpx = v, and v is a grid coordinate, not part
//     of the solution — so this "local Lax-Friedrichs" is genuinely just
//     upwinding in x, zero excess smearing) and av = umax (bounds
//     |dH/dpv| = umax). Positive at valleys/kinks -> it rounds them the
//     way the true viscosity solution does, instead of oscillating.
//   * min(0, ...) is the FREEZING step: once a cell's value has fallen
//     (= the cell joined the reachable tube), it never rises again — sets
//     only grow as the horizon grows, matching "reach WITHIN T" semantics.
//
// Memory spaces per thread and per sweep:
//   registers : the 5 stencil values + difference pairs (~20 regs)
//   global    : 5 reads from `in` — center and the x-neighbors sit at
//               consecutive addresses (coalesced within the warp's row);
//               the +/-nx neighbors are whole-row strides whose reuse
//               across the block's rows the L2 serves;
//               1 coalesced write to `out`.
// No shared memory: each interior value is re-read by at most 4 neighbor
// threads and the L2 covers that reuse at this grid size; a shared-memory
// tile is README Exercise 4 (measure it — the honest 07.09 position).
// No atomics, no divergence beyond the tail guard and border selects.
//
// Boundary policy: LINEAR EXTRAPOLATION ghost cells — the missing neighbor
// is invented as 2*center - opposite_neighbor, which makes p^- == p^+ at
// the border (the one-sided slope carried straight through, so the border
// adds no dissipation of its own). Standard for level sets on truncated
// domains; harmless here because the scenario keeps the final front more
// than 20 cells away from every edge (main.cu's scenario comment).
// ===========================================================================
__global__ void hj_sweep_kernel(const float* __restrict__ in,   // [nx*nv] value snapshot at tau (s)
                                float*       __restrict__ out,  // [nx*nv] OUT: value at tau + dt
                                HjGrid g)                       // by value: uniform per-launch data
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's x (position) index
    const int j = blockIdx.y * blockDim.y + threadIdx.y;   // this thread's v (velocity) index
    if (i >= g.nx || j >= g.nv) return;                    // ragged edges of the 16x16 tiling

    const int idx = j * g.nx + i;      // flat index — the kernels.cuh layout contract
    const float c = in[idx];           // center value V(x_i, v_j) at this tau

    // The four face neighbors, with linear-extrapolation ghosts at borders.
    // Each border ghost only needs the OPPOSITE neighbor, which always
    // exists because the launcher validates nx, nv >= 2.
    const float xm = (i > 0)        ? in[idx - 1]    : 2.0f * c - in[idx + 1];     // V at (i-1, j)
    const float xp = (i < g.nx - 1) ? in[idx + 1]    : 2.0f * c - in[idx - 1];     // V at (i+1, j)
    const float vm = (j > 0)        ? in[idx - g.nx] : 2.0f * c - in[idx + g.nx];  // V at (i, j-1)
    const float vp = (j < g.nv - 1) ? in[idx + g.nx] : 2.0f * c - in[idx - g.nx];  // V at (i, j+1)

    // One-sided difference pairs. Units: V carries seconds (main.cu's
    // initial condition is a time-to-reach function), so px is s/m and
    // pv is s/(m/s) — and H below comes out in s/s = unitless rate.
    const float pxm = (c - xm) / g.dx;     // backward difference in x
    const float pxp = (xp - c) / g.dx;     // forward  difference in x
    const float pvm = (c - vm) / g.dv;     // backward difference in v
    const float pvp = (vp - c) / g.dv;     // forward  difference in v

    // The Hamiltonian at the averaged gradient. v_j is this ROW's velocity
    // coordinate — for the double integrator the x-transport speed IS the
    // velocity state, which is what couples the two axes.
    const float vj = g.vmin + (float)j * g.dv;                       // v of row j (m/s)
    const float h  = 0.5f * (pxm + pxp) * vj                         // px*v term (transport in x)
                   - g.umax * fabsf(0.5f * (pvm + pvp));             // -umax*|pv|: the OPTIMAL control
                                                                     // u = -umax*sign(pv) pulls V down
                                                                     // as fast as the bound allows

    // Lax-Friedrichs dissipation: ax = |vj| exact, av = umax (see header).
    const float diss = 0.5f * fabsf(vj)  * (pxp - pxm)
                     + 0.5f * g.umax     * (pvp - pvm);

    // Freeze-and-step: the value may only decrease (tube semantics). One
    // coalesced write; the next sweep reads the other buffer (ping-pong).
    out[idx] = c + g.dt * fminf(0.0f, h + diss);
}

// ===========================================================================
// Host launcher (declared in kernels.cuh): allocate the ping-pong partner,
// run the sweep schedule, guarantee the result lands in the caller's buffer.
// ===========================================================================
void launch_hj_solve(const HjGrid& g, int n_sweeps, float* d_V)
{
    if (g.nx < 2 || g.nv < 2 || n_sweeps < 1 || !d_V ||
        g.dx <= 0.0f || g.dv <= 0.0f || g.umax <= 0.0f || g.dt <= 0.0f) {
        std::fprintf(stderr, "launch_hj_solve: invalid arguments (nx=%d nv=%d sweeps=%d)\n",
                     g.nx, g.nv, n_sweeps);
        std::exit(EXIT_FAILURE);
    }
    const size_t total = (size_t)g.nx * g.nv;

    // The ping-pong partner. d_V (the caller's) is buffer A; we allocate B;
    // sweeps alternate A->B, B->A. Writing in place instead would let
    // half-updated neighbors leak into the same sweep's stencils — the
    // read-modify-write race 07.09 explains; the swap is the cure.
    float* d_pong = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pong, total * sizeof(float)));

    const dim3 block(kTile, kTile);                    // 16x16 tile (see kTile comment)
    const dim3 grid((g.nx + kTile - 1) / kTile,
                    (g.nv + kTile - 1) / kTile);

    float* in  = d_V;                                  // caller's buffer holds l = V(tau=0)
    float* out = d_pong;

    // March backward time: n_sweeps steps of dt cover the horizon exactly
    // (main.cu chose dt = T / n_sweeps under the CFL bound). Launches queue
    // asynchronously in stream 0; each reads the previous one's completed
    // output — stream order IS the dependency, no explicit syncs needed.
    for (int s = 0; s < n_sweeps; ++s) {
        hj_sweep_kernel<<<grid, block>>>(in, out, g);
        CUDA_CHECK_LAST_ERROR("hj_sweep_kernel launch");
        // Swap roles: this sweep's output is the next sweep's snapshot.
        // Pointer swap costs nothing — the buffers trade names, not contents.
        float* tmp = in; in = out; out = tmp;
    }

    // After the loop the freshest field lives in `in` (the last-written
    // buffer). If that is the pong buffer (odd sweep count), copy back into
    // the caller's d_V — one device-to-device copy at full memory bandwidth
    // (256 KiB for the demo grid; negligible next to the sweeps).
    if (in != d_V)
        CUDA_CHECK(cudaMemcpy(d_V, in, total * sizeof(float),
                              cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaFree(d_pong));
}
