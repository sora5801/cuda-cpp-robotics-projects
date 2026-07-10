// ===========================================================================
// kernels.cu — GPU implementation for project 28.01
//              Real-time FEM soft-arm model + model-based control
//              (teaching core: explicit corotational-linear FEM, scatter
//               assembly with atomics, symplectic-Euler node integration)
//
// The big idea
// ------------
// Every dt, this project does exactly two GPU passes:
//   (1) elem_force_kernel  — one thread per ELEMENT computes that element's
//       internal (elastic + damping) force and SCATTERS it into its 4
//       corner nodes' shared force accumulator via atomicAdd.
//   (2) node_integrate_kernel — one thread per NODE turns the assembled
//       force into an acceleration and steps position/velocity forward.
// No global stiffness matrix, no linear solve — explicit dynamics is just
// "compute force from state, then integrate" repeated ~33,000 times/second
// of simulated arm motion. That is the whole real-time claim.
//
// SCATTER vs. GATHER — the race-condition story, told once, here (the
// deliberate contrast with 26.01's gather-only CG solver, kernels.cuh point
// 5): a node's stiffness neighborhood is its up-to-4 incident elements —
// EXACTLY the neighborhood 26.01's matvec_gather_kernel visits by READING
// from elements it does not own. This project visits the SAME neighborhood
// from the other side: one thread per ELEMENT computes ITS contribution and
// WRITES it into up to 4 nodes it does not own. Two elements sharing a node
// (every interior node has up to 4) can run on different SMs at the same
// instant and BOTH try to add into that node's force entry — a genuine data
// race if done with a plain "+=". atomicAdd serializes the contending
// writes so no update is lost; the price is contention (up to 4-way on an
// interior node) and a small but real slowdown versus an uncontended write,
// AND non-associative summation order that varies run to run (the reason
// the §5 GPU-vs-CPU gate compares within a tolerance, not bit-for-bit).
// WHY scatter here when 26.01 chose gather for the "same" neighborhood
// shape? Because THIS kernel's natural per-thread "problem" is an ELEMENT
// (the corotational rotation is an ELEMENT quantity — extracting it needs
// all 4 of an element's current corner positions at once, not a node's
// neighbor list), so the owner-writes-its-own-output-only trick that lets
// 26.01 avoid atomics does not apply for free here without redesigning the
// kernel around per-NODE rotation bookkeeping (a real but more complex
// option — README Exercise). Scatter+atomics is the honest, direct mapping
// of "one thread per element computes one element's answer"; this project
// teaches that half of the assembly-strategy spectrum on purpose.
//
// All layouts and constants come from kernels.cuh — the single source
// shared with the CPU twin; every kernel below is a deliberate line-by-line
// twin of the corresponding *_cpu function in reference_cpu.cpp (compare
// them side by side — that comparison IS the project's §5 gate).
//
// Read this after: kernels.cuh.  Companion twin: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"

#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// __constant__ storage: the unit element stiffness matrix. Defined here
// (kernels.cu owns the symbol — "who initializes the constants" stays in one
// obvious place, CLAUDE.md §12); populated once by upload_KE_hat(), called
// from main.cu before the first fem step.
// ---------------------------------------------------------------------------
__constant__ float d_KE_hat[64];

void upload_KE_hat(const float KE_hat[64])
{
    CUDA_CHECK(cudaMemcpyToSymbol(d_KE_hat, KE_hat, 64 * sizeof(float)));
}

// Launch geometry: 256-thread 1-D blocks, the repo default (08.01/26.01's
// vector-kernel choice) — both kernels below map a thread to a flat element
// or node index, so a 1-D grid-stride-free "one thread, tail-guarded" launch
// is the natural, simplest-correct shape at this problem's small size
// (<=1440 elements, <=1573 nodes — a handful of blocks either way).
static constexpr int kBlockSize = 256;

// ===========================================================================
// elem_force_kernel — one thread per ELEMENT (thread e = blockIdx.x*blockDim.x
// + threadIdx.x owns element (ex,ey) = (e % nelx, e / nelx)).
//
// Per-thread work, all in registers (no shared memory: elements do not
// share data with each other, only with nodes, via the atomic scatter):
//   1. Gather this element's 4 corner nodes' CURRENT position and velocity
//      (8 floats each) — the one read of `x`/`v` this thread needs.
//   2. Deformation gradient F (2x2) from the current positions and the
//      CONSTANT physical shape-function gradients (grad_n_physical,
//      kernels.cuh) — F = sum_a x_a (outer) grad(N_a); for a bilinear
//      element this identity gives F = I exactly at rest (THEORY.md proves
//      it), so no "+I" bookkeeping is needed.
//   3. Extract the rotation angle theta = atan2(F21-F12, F11+F22) — the
//      closed-form 2-D polar decomposition (THEORY.md "The math" derives
//      this from "the R in SO(2) that maximizes trace(R^T F)", i.e. the
//      rotation closest to F in Frobenius norm — exactly the corotational
//      frame we want, computed with ONE atan2, no eigendecomposition).
//   4. Local (corotated) displacement u_local_a = R^T x_a - X_a, local
//      velocity v_local_a = R^T v_a (no reference subtraction — velocities
//      have no rest value); combo = u_local + beta*v_local folds elastic
//      AND stiffness-proportional Rayleigh damping into ONE local force via
//      ONE 8x8 matvec: f_local = Et*KE_hat * combo (THEORY.md "The math").
//   5. Rotate f_local back to world (R * f_local_a per node) and SCATTER its
//      NEGATIVE (an internal force resists deformation: M*a = F_ext - F_int)
//      into `force` via atomicAdd — the race described in the file header.
//
// Memory spaces:
//   constant : d_KE_hat[64] — every thread, every launch, the same 64
//              floats (broadcast via the constant cache).
//   global   : 8 reads of x, 8 of v (this element's 4 corners); up to 8
//              atomicAdd writes to `force` (contended on shared nodes).
// No shared memory: nothing is reused BETWEEN threads in this kernel (unlike
// 26.01's gather, which reuses a node's data across its incident elements'
// gather passes) — the reuse here happens implicitly through `force`,
// mediated by the atomics.
// ===========================================================================
__global__ void elem_force_kernel(ArmGrid g,
                                  const float* __restrict__ x,
                                  const float* __restrict__ v,
                                  float Et, float h, float beta,
                                  float* __restrict__ force)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    const int nelem = g.nelx * g.nely;
    if (e >= nelem) return;                       // ragged-tail guard
    const int ex = e % g.nelx;                     // i (fast axis) is e's low bits
    const int ey = e / g.nelx;

    // ---- 1) gather this element's 4 corners: node ids, current x/v --------
    int node[4];
    float xa[4][2], va[4][2], Xa[4][2];             // current pos, current vel, REFERENCE pos
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        int cx, cy; corner_offset(a, &cx, &cy);
        node[a] = node_id(g, ex + cx, ey + cy);
        xa[a][0] = x[2 * node[a]];     xa[a][1] = x[2 * node[a] + 1];
        va[a][0] = v[2 * node[a]];     va[a][1] = v[2 * node[a] + 1];
        Xa[a][0] = static_cast<float>(ex + cx) * h;   // the mesh is a perfect
        Xa[a][1] = static_cast<float>(ey + cy) * h;   // regular grid at rest —
                                                       // reference corners are
                                                       // computed, never stored
    }

    // ---- 2) deformation gradient F at the element center (kernels.cuh's
    //         grad_n_physical: constant per-corner physical gradients for a
    //         uniform square mesh) ------------------------------------------
    float F11 = 0.0f, F12 = 0.0f, F21 = 0.0f, F22 = 0.0f;
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        float dNdX, dNdY;
        grad_n_physical(a, h, &dNdX, &dNdY);
        F11 += xa[a][0] * dNdX;   F12 += xa[a][0] * dNdY;
        F21 += xa[a][1] * dNdX;   F22 += xa[a][1] * dNdY;
    }

    // ---- 3) closed-form 2-D polar-decomposition rotation -------------------
    const float theta = atan2f(F21 - F12, F11 + F22);
    const float c = cosf(theta), s = sinf(theta);
    // R = [[c,-s],[s,c]]; R^T = [[c,s],[-s,c]] — applied inline below.

    // ---- 4) local combo = u_local + beta*v_local, per corner --------------
    float combo[8];
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        const float ux = xa[a][0], uy = xa[a][1];
        const float rx = c * ux + s * uy;          // R^T * x_a
        const float ry = -s * ux + c * uy;
        const float u_local_x = rx - Xa[a][0];
        const float u_local_y = ry - Xa[a][1];

        const float vx = va[a][0], vy = va[a][1];
        const float rvx = c * vx + s * vy;         // R^T * v_a (no reference term)
        const float rvy = -s * vx + c * vy;

        combo[2 * a]     = u_local_x + beta * rvx;
        combo[2 * a + 1] = u_local_y + beta * rvy;
    }

    // ---- 5) f_local = Et * KE_hat * combo (8x8 matvec, register-resident) -
    float f_local[8];
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        float acc = 0.0f;
        #pragma unroll
        for (int cc = 0; cc < 8; ++cc) acc += d_KE_hat[r * 8 + cc] * combo[cc];
        f_local[r] = Et * acc;
    }

    // ---- 6) rotate back to world and SCATTER the NEGATIVE (internal,
    //         resistive) force into the shared node accumulator ------------
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        const float flx = f_local[2 * a], fly = f_local[2 * a + 1];
        const float fwx = c * flx - s * fly;       // R * f_local_a
        const float fwy = s * flx + c * fly;
        atomicAdd(&force[2 * node[a]],     -fwx);   // THE race: up to 4 elements
        atomicAdd(&force[2 * node[a] + 1], -fwy);   // hit the same node this step
    }
}

// ===========================================================================
// node_integrate_kernel — one thread per NODE. Reads the force
// elem_force_kernel assembled, adds mass-proportional damping and this
// step's external forces (tendons + optional analytic-gate point load),
// and advances with SYMPLECTIC (semi-implicit) EULER:
//     v_new = v_old + dt * (force_total / m)      <- uses the OLD state's force
//     x_new = x_old + dt * v_new                   <- uses the NEW velocity
// This specific order (not the "naive" x_new = x_old + dt*v_old paired with
// v_new = v_old + dt*a, which is EXPLICIT Euler) is what gives the scheme
// its much better long-run energy behavior for oscillatory systems — the
// bounded-drift property THEORY.md "numerics" derives and the energy gate
// measures (same integrator family as 31.01's explicit stepping).
// Dirichlet-fixed nodes (the base column, i==0) are simply reset to their
// rest position with zero velocity — the boundary condition, enforced every
// step rather than solved for (the standard explicit-dynamics way to fix a
// DOF: it costs nothing extra and cannot drift).
// This kernel also ZEROES each force entry it consumes (the zero-after-
// consume contract in kernels.cuh): the next step's scatter then lands in a
// clean buffer without a per-step cudaMemset — one fewer host API call per
// step in the hot loop (the launcher comment below quantifies it). Safe
// because this kernel is the only reader and the next elem_force launch is
// separated from these writes by the stream's kernel-launch boundary.
// ===========================================================================
__global__ void node_integrate_kernel(ArmGrid g,
                                      float* __restrict__ x,
                                      float* __restrict__ v,
                                      float* __restrict__ force,
                                      const float* __restrict__ node_mass,
                                      const uint8_t* __restrict__ fixed,
                                      float h, float alpha, float dt,
                                      float T_top, float T_bottom,
                                      int point_force_node,
                                      float point_force_x, float point_force_y)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int nnode = g.nx * g.ny;
    if (n >= nnode) return;                        // ragged-tail guard
    const int i = n % g.nx;
    const int j = n / g.nx;

    // Consume this node's assembled force, then zero it for the next step
    // (done before the fixed-node early-out so base nodes' entries are
    // cleared too — elements DO scatter into base nodes, and stale force
    // there would silently corrupt step N+1's assembly).
    float fx = force[2 * n];
    float fy = force[2 * n + 1];
    force[2 * n]     = 0.0f;
    force[2 * n + 1] = 0.0f;

    if (fixed[2 * n]) {                             // the cantilever base (i==0):
        x[2 * n]     = 0.0f;                        // pinned to its rest position,
        x[2 * n + 1] = static_cast<float>(j) * h;    // zero velocity, every step
        v[2 * n]     = 0.0f;
        v[2 * n + 1] = 0.0f;
        return;
    }

    // Mass-proportional Rayleigh term (alpha = kRayleighAlphaOn = 3.8 1/s in
    // the tuned material — the LOW-mode half of the damping split derived in
    // kernels.cuh; the beta half lives in elem_force_kernel's combo).
    const float m = node_mass[n];
    fx += -alpha * m * v[2 * n];
    fy += -alpha * m * v[2 * n + 1];

    // Tendon line forces: every non-base node on the top row (j==nely) or
    // bottom row (j==0) carries an equal share of that fiber's tension,
    // pulling toward the base (-x). See kernels.cuh's "Tendon actuation"
    // comment for the honest distinction from a free cable.
    if (i >= 1 && i <= g.nelx) {
        if (j == g.nely) fx += -T_top    / static_cast<float>(kTendonAttachNodes);
        if (j == 0)      fx += -T_bottom / static_cast<float>(kTendonAttachNodes);
    }

    // Optional analytic-gate point force (static tip-deflection check only).
    if (point_force_node == n) {
        fx += point_force_x;
        fy += point_force_y;
    }

    const float ax = fx / m, ay = fy / m;
    const float vx_new = v[2 * n]     + dt * ax;    // velocity FIRST (symplectic Euler)
    const float vy_new = v[2 * n + 1] + dt * ay;
    v[2 * n]     = vx_new;
    v[2 * n + 1] = vy_new;
    x[2 * n]     += dt * vx_new;                     // THEN position, from the NEW velocity
    x[2 * n + 1] += dt * vy_new;
}

// ---------------------------------------------------------------------------
// launch_fem_step — host wrapper (declared in kernels.cuh): assemble
// (scatter), then integrate. Called every kDt_s by main.cu's physics loop —
// this function's measured steps/wall-second is this project's real-time-
// factor claim (README "Expected output").
//
// Note what is NOT here: no cudaMemset. The first draft zeroed d_force with
// cudaMemset every call (3 host API calls per step); the zero-after-consume
// contract (node_integrate_kernel re-zeroes what it reads) removes it. At
// this problem size the step cost is DOMINATED by per-call submission
// overhead, not GPU arithmetic, so cutting 3 calls to 2 is a direct saving
// on the hot loop (measured before/after on the reference RTX 2080 SUPER,
// 20k-step enqueue+sync loop: 27.9 -> 17.9 us/step; that launch-overhead-
// dominated regime is itself a teaching point — THEORY.md "The GPU
// mapping"). The caller provides the one initial all-zero buffer (a single
// cudaMemset at allocation, main.cu).
// ---------------------------------------------------------------------------
void launch_fem_step(const ArmGrid& g,
                     float* d_x, float* d_v, float* d_force,
                     const float* d_node_mass, const uint8_t* d_fixed,
                     float Et, float h, float alpha, float beta, float dt,
                     float T_top, float T_bottom,
                     int point_force_node, float point_force_x, float point_force_y)
{
    const int nnode = g.nx * g.ny;
    const int nelem = g.nelx * g.nely;

    const int elem_blocks = (nelem + kBlockSize - 1) / kBlockSize;
    elem_force_kernel<<<elem_blocks, kBlockSize>>>(g, d_x, d_v, Et, h, beta, d_force);
    CUDA_CHECK_LAST_ERROR("elem_force_kernel launch");

    const int node_blocks = (nnode + kBlockSize - 1) / kBlockSize;
    node_integrate_kernel<<<node_blocks, kBlockSize>>>(
        g, d_x, d_v, d_force, d_node_mass, d_fixed, h, alpha, dt,
        T_top, T_bottom, point_force_node, point_force_x, point_force_y);
    CUDA_CHECK_LAST_ERROR("node_integrate_kernel launch");
}
