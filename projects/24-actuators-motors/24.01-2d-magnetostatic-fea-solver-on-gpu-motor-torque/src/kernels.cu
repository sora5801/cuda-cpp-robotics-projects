// ===========================================================================
// kernels.cu — GPU implementation for project 24.01
//              2D magnetostatic FEA solver on GPU -> motor torque-ripple/
//              cogging parameter sweeps
//              (teaching core: batched red-black SOR relaxation)
//
// The big idea
// ------------
// The magnetostatic PDE  -div(nu * grad(A_z)) = J_z  is a variable-
// coefficient Poisson equation. Discretized on a grid it becomes, at every
// interior node, a 5-point stencil relating a cell to its 4 face neighbors
// through HARMONIC-averaged reluctivities (THEORY.md derives why harmonic
// averaging, not arithmetic, is the physically correct choice at a material
// interface — the classic finite-volume lesson). Solving that stencil
// everywhere simultaneously, over and over, until the field stops moving,
// IS the "FEA solve" this project teaches — 07.09/31.01's stencil pattern,
// now with a genuinely variable coefficient and an iterative linear solve
// in place of an explicit PDE march.
//
// RED-BLACK GAUSS-SEIDEL, not Jacobi: color the grid like a checkerboard by
// (i+j) parity. Every "red" cell's 4 neighbors are all "black" and vice
// versa, so updating all red cells in parallel — reading ONLY black
// (unmodified-this-pass) neighbors — is exactly equivalent to a sequential
// Gauss-Seidel sweep over the red cells, and vice versa for black. This is
// what lets a GPU do Gauss-Seidel (normally a strictly SEQUENTIAL algorithm
// — cell i needs cell i-1's *already updated* value) with zero race
// conditions and NO ping-pong buffer: red-black updates IN PLACE, halving
// the memory traffic 31.01's ping-pong Jacobi-style scheme pays. SOR
// (Successive Over-Relaxation, omega > 1) rides on top for free: it is the
// same stencil with an extra blend toward the update, and it is the
// difference between converging in hundreds vs. thousands of sweeps on a
// grid this size (THEORY.md "Numerical considerations" has the measured
// numbers that justify kOmega=1.97).
//
// BATCHING: blockIdx.z selects the variant (rotor angle). Every variant
// shares the SAME grid geometry (g) but has its own nu/Jsrc/A slice — B
// independent linear solves, running as one kernel launch instead of B
// separate ones, is this project's second GPU lesson (after the stencil
// itself): "many independent SMALL problems" is exactly as GPU-friendly as
// "one large problem", as long as the layout keeps each problem's own
// per-row memory access coalesced (see the BATCHED GRID LAYOUT comment in
// kernels.cuh — the batch axis rides OUTSIDE the per-variant (i,j) layout).
//
// What is NEW here beyond 07.09/31.01's single-grid stencils:
//   * a VARIABLE coefficient (nu varies per node — air/iron/magnet), with
//     face values computed ON THE FLY from the neighbor's node value (no
//     separate face-coefficient array — trading a few extra flops per cell
//     for one less array's worth of memory traffic, a deliberate choice
//     documented below);
//   * RED-BLACK color-splitting instead of ping-pong Jacobi — a genuinely
//     different (and here, faster-converging) way to parallelize a
//     sequential relaxation, teaching a second idiom alongside 31.01's;
//   * a BATCH dimension folded into the launch grid (blockIdx.z), so one
//     launch solves many independent problems — the parameter-sweep
//     project's central GPU lesson.
//
// All layouts and constants come from kernels.cuh — the single source
// shared with the CPU twin; the stencil expression below is a deliberate
// line-by-line twin of the one in reference_cpu.cpp.
//
// Read this after: kernels.cuh.  Companion twin: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"      // CUDA_CHECK / CUDA_CHECK_LAST_ERROR (paragraph 6.1 rule 7)

#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Launch geometry: 2-D tiles of 16x16 = 256 threads per (i,j) slice, one
// slice per batch variant along the z grid axis. 16x16 is the repo default
// square-tile shape for 2-D stencils (31.01 uses the same tile size and the
// same reasoning: threadIdx.x maps to i, the fast/contiguous axis, so every
// warp's 32 reads within a row are consecutive floats — coalesced).
// ---------------------------------------------------------------------------
static constexpr int kTile = 16;

// ===========================================================================
// fea_sor_pass_kernel — one RED (color=0) or BLACK (color=1) half-sweep of
// the batched red-black SOR solve. One thread = one (batch, cell) update.
//
// Thread-to-data mapping: thread (i, j, b) = (blockIdx.x*blockDim.x+
// threadIdx.x, blockIdx.y*blockDim.y+threadIdx.y, blockIdx.z) owns variant
// b's cell (i, j) — flat index idx = (b*g.ny + j)*g.nx + i (the kernels.cuh
// batched layout contract). Two guards, in order:
//   1) out-of-range / border guard — border nodes carry the Dirichlet A=0
//      condition and are simply never written (kernels.cuh explains why
//      that alone enforces the boundary condition);
//   2) checkerboard-color guard — a thread only updates cells of ITS half
//      of the checkerboard this launch; the other half is read-only this
//      pass (their values came from the LAST launch, still valid — that is
//      exactly what makes red-black race-free without a ping-pong buffer).
//
// The math (full derivation in THEORY.md; summary here so the kernel reads
// standalone). At interior node (i,j), discretizing -div(nu*grad(A)) = J on
// a square grid (dx=dy=h) with face reluctivities HARMONIC-averaged from
// the two adjacent nodes (nu_face = 0.5*(nu_here + nu_neighbor) — this
// arithmetic mean OF RELUCTIVITY is algebraically identical to the harmonic
// mean of the two nodes' PERMEABILITIES; THEORY.md proves the identity and
// why it, not a naive arithmetic mean of mu, is physically correct):
//
//     A_new = ( nu_E*A_E + nu_W*A_W + nu_N*A_N + nu_S*A_S + h^2*J ) / diag
//     diag  = nu_E + nu_W + nu_N + nu_S
//
// SOR blends the Gauss-Seidel update toward this target instead of jumping
// straight to it: A <- A + omega*(A_new - A), omega in (0,2). omega=1 is
// plain Gauss-Seidel; omega near 2 (this project uses 1.97) converges an
// order of magnitude faster on a grid this size — THEORY.md's measured
// sweep-count-vs-omega table justifies the exact value.
//
// Memory spaces per thread and per pass:
//   registers : nu_c/nu_e/nu_w/nu_n/nu_s, diag, the 4 neighbor A reads,
//               ~15 registers — a light, compute-cheap kernel per cell.
//   global    : 5 reads from `nu` (center + 4 face neighbors — center and
//               the i+-1 neighbors sit at CONSECUTIVE addresses within a
//               warp's row, coalesced; the j+-1 neighbors are whole-row
//               strides the L2 cache serves across the block's rows,
//               exactly 31.01's reasoning), 4 reads from `A` (same
//               pattern), 1 read from `Jsrc`, 1 write to `A`.
// No shared memory: each node's nu/A value is reused by at most 4 neighbor
// threads and the L2 covers that reuse at this grid size (07.09/31.01's
// same documented choice; a shared-memory tile is README Exercise 4).
// No atomics; no divergence beyond the two guards above (the checkerboard
// test IS branchy per-thread but every thread in a warp shares the same
// (i+j) parity pattern only along a ROW — within a warp of 32 CONSECUTIVE i
// values, exactly half take each branch, so a warp always has some threads
// return early on line 1 of the color test; this costs a predicated no-op
// for those lanes, not a serialized re-execution, because both branches are
// trivial (an early return) — cheap warp divergence, not the expensive kind).
// ===========================================================================
__global__ void fea_sor_pass_kernel(FeaGrid g, int B, int color,
                                    const float* __restrict__ nu,    // [B*ny*nx] reluctivity (m/H)
                                    const float* __restrict__ Jsrc,  // [B*ny*nx] source J_z (A/m^2)
                                    float*       __restrict__ A)     // [B*ny*nx] IN/OUT vector potential (Wb/m)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's x (fast) index
    const int j = blockIdx.y * blockDim.y + threadIdx.y;   // this thread's y (slow) index
    const int b = blockIdx.z;                               // this thread's batch/variant index

    if (i >= g.nx || j >= g.ny || b >= B) return;           // ragged tile edges / batch guard

    // Border nodes carry the Dirichlet A=0 boundary condition (kernels.cuh)
    // and are never updated — skip them so whatever the caller uploaded
    // there (main.cu: zero) persists for the whole solve.
    if (i == 0 || i == g.nx - 1 || j == 0 || j == g.ny - 1) return;

    // Checkerboard color test: only cells whose (i+j) parity matches this
    // launch's color get touched. See the header comment for why this is
    // race-free without a ping-pong buffer (the two-pass structure IS the
    // synchronization — the launcher enforces pass ordering via stream
    // order, exactly like 31.01's ping-pong swap).
    if (((i + j) & 1) != color) return;

    const int idx = (b * g.ny + j) * g.nx + i;   // this cell's flat index (kernels.cuh layout)

    // Center reluctivity, then the 4 face-averaged reluctivities computed
    // ON THE FLY from the neighboring nodes' values. Trading compute
    // (4 extra multiply-adds) for memory traffic (no separate face-
    // coefficient array to allocate, upload, and read every sweep) — the
    // classic GPU trade of recompute-vs-store, and the right one here
    // because `nu` is tiny (one float per node) and read-bandwidth, not
    // arithmetic, dominates this kernel's cost (THEORY.md "GPU mapping").
    const float nu_c = nu[idx];
    const float nu_e = 0.5f * (nu_c + nu[idx + 1]);          // face toward i+1 (safe: i<nx-1, guarded above)
    const float nu_w = 0.5f * (nu_c + nu[idx - 1]);          // face toward i-1 (safe: i>0)
    const float nu_n = 0.5f * (nu_c + nu[idx + g.nx]);       // face toward j+1 (safe: j<ny-1)
    const float nu_s = 0.5f * (nu_c + nu[idx - g.nx]);       // face toward j-1 (safe: j>0)
    const float diag = nu_e + nu_w + nu_n + nu_s;            // always > 0: nu > 0 everywhere (mu_r finite)

    const float Ae = A[idx + 1];
    const float Aw = A[idx - 1];
    const float An = A[idx + g.nx];
    const float As = A[idx - g.nx];

    const float h2 = g.h * g.h;
    const float gs_target =
        (nu_e * Ae + nu_w * Aw + nu_n * An + nu_s * As + h2 * Jsrc[idx]) / diag;

    // SOR blend: step omega of the way from the current value toward the
    // fresh Gauss-Seidel target. Writing back IN PLACE is what makes this
    // pass immediately visible to the OTHER color's pass launched right
    // after it (stream order is the only synchronization needed, as in
    // every ping-pong/color-split solver in this repo).
    A[idx] = A[idx] + g.omega * (gs_target - A[idx]);
}

// ===========================================================================
// Host launcher (declared in kernels.cuh): run n_sweeps red+black pass
// pairs, in place, over the whole batch.
// ===========================================================================
void launch_fea_solve_batch(const FeaGrid& g, int B, int n_sweeps,
                            const float* d_nu, const float* d_Jsrc, float* d_A)
{
    if (g.nx < 4 || g.ny < 4 || B < 1 || n_sweeps < 1 || !d_nu || !d_Jsrc || !d_A ||
        g.h <= 0.0f || g.omega <= 0.0f || g.omega >= 2.0f) {
        std::fprintf(stderr,
            "launch_fea_solve_batch: invalid arguments (nx=%d ny=%d B=%d sweeps=%d omega=%g)\n",
            g.nx, g.ny, B, n_sweeps, static_cast<double>(g.omega));
        std::exit(EXIT_FAILURE);
    }

    const dim3 block(kTile, kTile, 1);                       // 16x16 threads per (i,j) tile
    const dim3 grid((g.nx + kTile - 1) / kTile,               // tiles covering x
                    (g.ny + kTile - 1) / kTile,               // tiles covering y
                    static_cast<unsigned int>(B));            // one z-slice per batch variant

    // n_sweeps RED+BLACK pass pairs, launched back to back in the default
    // stream. Stream order guarantees each pass sees the FULL result of the
    // previous one (no explicit cudaDeviceSynchronize needed between passes
    // — exactly the dependency argument 31.01's ping-pong loop documents).
    for (int s = 0; s < n_sweeps; ++s) {
        fea_sor_pass_kernel<<<grid, block>>>(g, B, /*color=*/0, d_nu, d_Jsrc, d_A);  // red pass
        CUDA_CHECK_LAST_ERROR("fea_sor_pass_kernel (red) launch");
        fea_sor_pass_kernel<<<grid, block>>>(g, B, /*color=*/1, d_nu, d_Jsrc, d_A);  // black pass
        CUDA_CHECK_LAST_ERROR("fea_sor_pass_kernel (black) launch");
    }
}
