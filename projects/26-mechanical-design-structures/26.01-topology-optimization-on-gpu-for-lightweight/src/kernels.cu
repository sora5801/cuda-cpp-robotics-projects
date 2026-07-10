// ===========================================================================
// kernels.cu — GPU implementation for project 26.01
//              Topology optimization (SIMP) on GPU for lightweight links and
//              brackets — flagship design project
//              (teaching core: matrix-free preconditioned-CG FEA solve)
//
// The big idea
// ------------
// Nearly all the arithmetic in SIMP topology optimization is "solve K U = F,
// over and over, for a K that changes a little each outer iteration." A
// direct solve would ASSEMBLE a global sparse stiffness matrix (expensive to
// build AND expensive to keep synchronized with a changing density field)
// and factor it (expensive, and a poor GPU fit — sparse factorization is
// bandwidth- and latency-bound in ways a GPU's SIMT model handles badly).
// This project instead solves K U = F with MATRIX-FREE conjugate gradient:
// CG only ever needs the ACTION of K on a vector (K*p), never K itself — so
// we compute that action on demand, per CG iteration, by GATHERING each
// node's contribution directly from its incident elements' densities and
// the (precomputed, tiny) unit element stiffness matrix. No matrix is ever
// assembled, stored, or synchronized: the density field IS the matrix.
//
// GATHER vs. SCATTER (the race-condition story, told once, here):
//   SCATTER would launch one thread per ELEMENT, each computing its 8x8
//   local contribution K_e * p_e and ADDING (atomicAdd) the results into a
//   shared global output vector — every element writes into up to 4 nodes
//   it does NOT own, so different threads race on the SAME output entries
//   and atomics are mandatory (a real cost: atomicAdd serializes contending
//   writes, and up to 4 elements contend on every interior node's 2 dofs).
//   GATHER instead launches one thread per NODE, and each thread reads
//   (never writes) its up-to-4 incident elements' data and OWNS the single
//   output entry it writes — zero atomics, zero write races, by
//   construction. The price is redundant READS (a node shared by 4 elements
//   gets read by all 4 of THEIR gather passes) — but reads are cheap and
//   cacheable (L2 covers this exactly as 24.01's FEA stencil documents),
//   while atomics are not. GATHER is the right default whenever the
//   "owner of the output" can be identified up front — exactly the case
//   here (a node's own displacement update needs only ITS incident
//   elements, a small, static, precomputable neighborhood). This project
//   teaches gather; a scatter+atomics variant is README Exercise 4.
//
// THE STENCIL: a 2D SCALAR Poisson problem (07.09/24.01/31.01) needs a
// 5-point stencil (4 neighbors + self). This project's PDE is VECTOR
// (2 dofs per node) and the shape functions are BILINEAR, so a node's true
// stiffness support is its full 3x3 neighborhood (9 nodes, 18 dofs) — a
// richer version of the same idea, arising from the SAME 4 incident
// elements every node in a structured Q4 mesh has (see kernels.cuh's
// "MESH & DOF LAYOUT" comment for the local-node-index convention every
// kernel below assumes).
//
// All layouts and constants come from kernels.cuh — the single source
// shared with the CPU twin; every kernel below is a deliberate line-by-line
// twin of the corresponding *_cpu function in reference_cpu.cpp.
//
// Read this after: kernels.cuh.  Companion twin: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// ---------------------------------------------------------------------------
// __constant__ storage (declared extern in kernels.cuh; DEFINED here, once —
// the "who owns the symbol" rule). Populated by upload_KE_hat /
// upload_filter_weights below, called once by main.cu before any solve.
// ---------------------------------------------------------------------------
__constant__ float d_KE_hat[64];
__constant__ float d_filter_w[(2 * kFilterR + 1) * (2 * kFilterR + 1)];

void upload_KE_hat(const float KE_hat[64])
{
    CUDA_CHECK(cudaMemcpyToSymbol(d_KE_hat, KE_hat, 64 * sizeof(float)));
}
void upload_filter_weights(const float weights[(2 * kFilterR + 1) * (2 * kFilterR + 1)])
{
    CUDA_CHECK(cudaMemcpyToSymbol(d_filter_w, weights,
        (2 * kFilterR + 1) * (2 * kFilterR + 1) * sizeof(float)));
}

// ---------------------------------------------------------------------------
// Launch geometry: 16x16 node/element tiles — the repo default square-tile
// shape for 2D grid kernels (same choice and reasoning as 24.01's kTile).
// ---------------------------------------------------------------------------
static constexpr int kTile = 16;

// ---------------------------------------------------------------------------
// gather_element_local — shared helper (device-inline) used by BOTH
// matvec_gather_kernel and diag_gather_kernel: given a node (i,j) and one of
// its up-to-4 incident elements (identified by which "quadrant" q it is —
// see the local-index table below), read that element's density and, for
// matvec, its 4 corner nodes' x-vector values into an 8-float local array.
//
// Quadrant table (q -> (ex,ey) offset from (i,j), and this node's LOCAL
// index L within that element — the CCW convention kernels.cuh fixes):
//   q=0: element (i-1,j-1), this node is its local corner 2 (max-x,max-y)
//   q=1: element (i,  j-1), this node is its local corner 3 (min-x,max-y)
//   q=2: element (i-1,j  ), this node is its local corner 1 (max-x,min-y)
//   q=3: element (i,  j  ), this node is its local corner 0 (min-x,min-y)
// Each is valid only if the element indices stay in [0,nelx)x[0,nely) — the
// bounds check every call site below performs before using q.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void quadrant_elem(int i, int j, int q, int* ex, int* ey, int* L)
{
    // clang-format off
    switch (q) {
        case 0: *ex = i - 1; *ey = j - 1; *L = 2; break;
        case 1: *ex = i;     *ey = j - 1; *L = 3; break;
        case 2: *ex = i - 1; *ey = j;     *L = 1; break;
        default:*ex = i;     *ey = j;     *L = 0; break;
    }
    // clang-format on
}

// ===========================================================================
// matvec_gather_kernel — the project's hot kernel: one thread per NODE,
// computing (K(rho) * x) at that node's 2 dofs by gathering over its up to
// 4 incident elements (see the file-header GATHER discussion).
//
// Thread-to-data mapping: thread (i,j) = (blockIdx.x*blockDim.x+threadIdx.x,
// blockIdx.y*blockDim.y+threadIdx.y) owns node (i,j)'s output y[2n], y[2n+1]
// where n = node_id(g,i,j).
//
// Per incident element e (density rho_e, local index L for THIS node):
//   1) gather the 4 corner nodes' x-vector values into a local 8-float array
//      x_e[0..7] = (x[node0].x, x[node0].y, x[node1].x, ..., x[node3].y);
//   2) the local contribution to THIS node's 2 output rows is the (2x8)
//      slice of KE_hat at rows [2L, 2L+1], scaled by E(rho_e):
//          y_local[c] += E(rho_e) * KE_hat[2L+c][k] * x_e[k]   for k=0..7, c=0,1
//   3) sum over all incident elements (up to 4).
// Then the Dirichlet fix: if THIS dof is marked fixed, force the output to
// EXACTLY zero — the standard matrix-free way to realize a reduced (free-
// dof-only) linear system without ever touching K itself (THEORY.md
// "Numerical considerations" proves this keeps the CG iterate identically
// zero on every fixed dof for every iteration, by induction on r and p).
//
// Memory spaces per thread:
//   constant : d_KE_hat[64] — read by every thread in every launch, the
//              textbook constant-cache broadcast case (kernels.cuh comment).
//   global   : up to 4 reads of `rho` (one per incident element — L2-cached,
//              each element's rho is re-read by up to 4 node-threads, the
//              accepted GATHER redundancy discussed in the file header);
//              up to 4*4=16 reads of `x` (again, heavily reused across
//              neighboring node-threads — L2 covers it at this problem size,
//              the same no-shared-memory call 24.01 makes and justifies);
//              1 read of `fixed`; 2 coalesced writes to `y` (i is the fast
//              axis, so adjacent threads write adjacent node's y-pairs).
// No atomics (GATHER, by construction — see file header). Divergence: the
// element-bounds checks at grid edges only affect the outermost ring of
// threads; interior threads (the overwhelming majority for any grid bigger
// than a few cells) take all 4 branches uniformly.
// ===========================================================================
__global__ void matvec_gather_kernel(TopoGrid g,
                                     const float* __restrict__ rho,
                                     const float* __restrict__ x,
                                     const uint8_t* __restrict__ fixed,
                                     float E0, float Emin,
                                     float* __restrict__ y)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= g.nx || j >= g.ny) return;

    float y0 = 0.0f, y1 = 0.0f;   // this node's two output accumulators (x, y dofs)

    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        int ex, ey, L;
        quadrant_elem(i, j, q, &ex, &ey, &L);
        if (ex < 0 || ex >= g.nelx || ey < 0 || ey >= g.nely) continue;   // off-domain quadrant

        const int e = elem_id(g, ex, ey);
        const float Ee = E_of_rho(rho[e], E0, Emin);

        // Gather this element's 4 corner nodes' x-values (8 floats, the
        // element's local DOF vector) in the CCW order kernels.cuh fixes.
        const int n0 = node_id(g, ex,     ey);
        const int n1 = node_id(g, ex + 1, ey);
        const int n2 = node_id(g, ex + 1, ey + 1);
        const int n3 = node_id(g, ex,     ey + 1);
        const float xe[8] = { x[2*n0], x[2*n0+1], x[2*n1], x[2*n1+1],
                              x[2*n2], x[2*n2+1], x[2*n3], x[2*n3+1] };

        // This node's 2 output rows are KE_hat's rows [2L, 2L+1] dotted
        // with the local vector — 16 multiply-adds, scaled by E(rho_e).
        const float* row0 = &d_KE_hat[(2*L)   * 8];
        const float* row1 = &d_KE_hat[(2*L+1) * 8];
        float s0 = 0.0f, s1 = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; ++k) { s0 += row0[k] * xe[k]; s1 += row1[k] * xe[k]; }
        y0 += Ee * s0;
        y1 += Ee * s1;
    }

    const int n = node_id(g, i, j);
    // Dirichlet fix: force EXACT zero on fixed dofs (see the kernel header's
    // induction argument — this alone realizes the reduced linear system).
    y[2*n]   = fixed[2*n]   ? 0.0f : y0;
    y[2*n+1] = fixed[2*n+1] ? 0.0f : y1;
}

// ===========================================================================
// diag_gather_kernel — the Jacobi preconditioner: diag(K(rho)) at every
// free dof (fixed dofs get a harmless nonzero placeholder — see below).
// Same gather structure as matvec_gather_kernel but MUCH cheaper: it needs
// only the (L,L)-diagonal 2x2 block of KE_hat per incident element (no
// 8-value local gather of a vector — diag(K) does not depend on any vector,
// only on the density field, so it is recomputed once per OUTER iteration,
// not once per CG iteration).
// ===========================================================================
__global__ void diag_gather_kernel(TopoGrid g,
                                   const float* __restrict__ rho,
                                   const uint8_t* __restrict__ fixed,
                                   float E0, float Emin,
                                   float* __restrict__ diag)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= g.nx || j >= g.ny) return;

    float d0 = 0.0f, d1 = 0.0f;
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        int ex, ey, L;
        quadrant_elem(i, j, q, &ex, &ey, &L);
        if (ex < 0 || ex >= g.nelx || ey < 0 || ey >= g.nely) continue;
        const float Ee = E_of_rho(rho[elem_id(g, ex, ey)], E0, Emin);
        d0 += Ee * d_KE_hat[(2*L)   * 8 + (2*L)];       // KE_hat[2L][2L]
        d1 += Ee * d_KE_hat[(2*L+1) * 8 + (2*L+1)];     // KE_hat[2L+1][2L+1]
    }

    const int n = node_id(g, i, j);
    // Fixed dofs: the matvec forces their K*x output to zero, so CG's
    // residual/search-direction stay identically zero there regardless of
    // what diag holds (kernel header's induction argument) — 1.0 is simply
    // a division-by-zero guard, never actually used to compute anything.
    diag[2*n]   = fixed[2*n]   ? 1.0f : fmaxf(d0, 1e-20f);
    diag[2*n+1] = fixed[2*n+1] ? 1.0f : fmaxf(d1, 1e-20f);
}

// ===========================================================================
// elem_sensitivity_kernel — one thread per ELEMENT (a different mapping
// from the two kernels above: sensitivities are an ELEMENT quantity, so the
// natural "problem" a thread owns here is one element, not one node — the
// same map-pattern shift 07.09's cell-vs-seed kernels illustrate).
//
// Reads this element's 4 corner nodes' SOLVED displacements (gathered, 8
// floats) and computes the quadratic form q_e = u_e^T KE_hat u_e (the
// element's unit-stiffness strain energy — 64 multiply-adds, small and
// register-resident). From q_e, THEORY.md's chain rule gives both outputs
// directly:
//     ce[e]     = E(rho_e) * q_e                    (this element's share of total compliance)
//     dc_raw[e] = -dE/drho_e * q_e = -3 rho_e^2 (E0-Emin) * q_e   (raw, UNFILTERED sensitivity)
// (Compliance is minimized, so a NEGATIVE dc/drho — "adding material here
// helps" — is what SIMP's OC update below responds to; the raw sign
// convention and the OC formula are matched in THEORY.md "The math".)
// ===========================================================================
__global__ void elem_sensitivity_kernel(TopoGrid g,
                                        const float* __restrict__ rho,
                                        const float* __restrict__ U,
                                        float E0, float Emin,
                                        float* __restrict__ ce,
                                        float* __restrict__ dc_raw)
{
    const int ex = blockIdx.x * blockDim.x + threadIdx.x;
    const int ey = blockIdx.y * blockDim.y + threadIdx.y;
    if (ex >= g.nelx || ey >= g.nely) return;

    const int n0 = node_id(g, ex,     ey);
    const int n1 = node_id(g, ex + 1, ey);
    const int n2 = node_id(g, ex + 1, ey + 1);
    const int n3 = node_id(g, ex,     ey + 1);
    const float ue[8] = { U[2*n0], U[2*n0+1], U[2*n1], U[2*n1+1],
                          U[2*n2], U[2*n2+1], U[2*n3], U[2*n3+1] };

    // q_e = ue^T KE_hat ue — a full 8x8 quadratic form, register-resident
    // (KE_hat lives in the constant cache, broadcast to every element
    // thread — the same read pattern matvec's rows use, here for the whole
    // matrix instead of two rows).
    float q = 0.0f;
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        float s = 0.0f;
        #pragma unroll
        for (int c = 0; c < 8; ++c) s += d_KE_hat[r*8 + c] * ue[c];
        q += ue[r] * s;
    }

    const int e = elem_id(g, ex, ey);
    const float rho_e = rho[e];
    ce[e]     = E_of_rho(rho_e, E0, Emin) * q;
    dc_raw[e] = -dE_drho(rho_e, E0, Emin) * q;
}

// ===========================================================================
// density_filter_kernel — one thread per element: Sigmund's classic
// density-weighted SENSITIVITY filter (THEORY.md derives why filtering the
// SENSITIVITY, weighted by neighbor density, fixes the checkerboard
// pathology and buys mesh-independence). Formula:
//
//     dc_filt[e] = ( sum_f w(e,f) * rho_f * dc_raw[f] )
//                / ( rho_e * sum_f w(e,f) )                (Sigmund 1997/2001)
//
// with w(e,f) = max(0, rmin - dist(e,f)) — the small precomputed 5x5
// constant table d_filter_w (kernels.cuh: exact for kFilterR=2 since any
// offset outside that window has distance > kFilterRMin by construction).
// rho_e in the denominator is floored (1e-3) to avoid dividing by a
// near-zero design variable at the void end of SIMP's range.
// ===========================================================================
__global__ void density_filter_kernel(TopoGrid g,
                                      const float* __restrict__ rho,
                                      const float* __restrict__ dc_raw,
                                      float* __restrict__ dc_filt)
{
    const int ex = blockIdx.x * blockDim.x + threadIdx.x;
    const int ey = blockIdx.y * blockDim.y + threadIdx.y;
    if (ex >= g.nelx || ey >= g.nely) return;

    float num = 0.0f, wsum = 0.0f;
    #pragma unroll
    for (int dj = -kFilterR; dj <= kFilterR; ++dj) {
        const int fy = ey + dj;
        if (fy < 0 || fy >= g.nely) continue;
        #pragma unroll
        for (int di = -kFilterR; di <= kFilterR; ++di) {
            const int fx = ex + di;
            if (fx < 0 || fx >= g.nelx) continue;
            const float w = d_filter_w[(dj + kFilterR) * (2*kFilterR+1) + (di + kFilterR)];
            if (w <= 0.0f) continue;   // outside the true radius (5x5 window is a superset)
            const int f = elem_id(g, fx, fy);
            num  += w * rho[f] * dc_raw[f];
            wsum += w;
        }
    }

    const int e = elem_id(g, ex, ey);
    const float rho_e_floored = fmaxf(rho[e], 1.0e-3f);
    dc_filt[e] = (wsum > 0.0f) ? (num / (rho_e_floored * wsum)) : dc_raw[e];
}

// ===========================================================================
// Small vector kernels — the CG solver's map/reduce building blocks. Each is
// a plain grid-stride map over ndof (or nelx*nely) floats; see 26.01's own
// scaffolded SAXPY placeholder for the pattern these generalize (this
// project's real AXPY-shaped kernel is vec_combine_kernel below — the
// toolchain smoke test the scaffold shipped turns out to be exactly the
// building block the real algorithm needs).
// ===========================================================================
__global__ void vec_copy_kernel(int n, const float* __restrict__ src, float* __restrict__ dst)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride) dst[i] = src[i];
}

// y = a*x + b*y, in place — covers every CG update this project needs:
// r=F-Kp (a=-1,b=1 after seeding y=F via vec_copy_kernel), U+=alpha*p
// (a=alpha,b=1), r-=alpha*Kp (a=-alpha,b=1), p=z+beta*p (a=1,b=beta).
__global__ void vec_combine_kernel(int n, float a, const float* __restrict__ x,
                                   float b, float* __restrict__ y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride) y[i] = a * x[i] + b * y[i];
}

__global__ void vec_divide_kernel(int n, const float* __restrict__ num,
                                  const float* __restrict__ den, float* __restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride) out[i] = num[i] / den[i];
}

// reduce_dot_kernel — block-level shared-memory tree reduction computing one
// PARTIAL sum per block of a[i]*b[i]; the host sums the (few dozen) block
// partials to finish the reduction (main.cu / kernels.cu's dot() helper
// below). This is the repo's "reduce" pattern (map/reduce/stencil/scan/
// batched-solve/sampling — CLAUDE.md README template's own vocabulary):
// each thread accumulates a grid-stride partial product, then a standard
// power-of-two shared-memory tree collapses blockDim.x partials to 1.
__global__ void reduce_dot_kernel(int n, const float* __restrict__ a,
                                  const float* __restrict__ b, float* __restrict__ partial)
{
    extern __shared__ float sdata[];              // blockDim.x floats, one per thread
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;

    float acc = 0.0f;
    for (; i < n; i += stride) acc += a[i] * b[i]; // grid-stride: each thread may sum several elements
    sdata[tid] = acc;
    __syncthreads();

    // Tree reduction within the block: at each step, half the still-active
    // threads add their neighbor's value; after log2(blockDim.x) steps
    // sdata[0] holds the whole block's sum.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

// ---------------------------------------------------------------------------
// Host-side launch wrappers — grid math + launch-error checks, kept beside
// their kernels (the repo's standard shape).
// ---------------------------------------------------------------------------
static void launch_matvec(const TopoGrid& g, const float* d_rho, const float* d_x,
                          const uint8_t* d_fixed, float E0, float Emin, float* d_y)
{
    dim3 block(kTile, kTile);
    dim3 grid((g.nx + kTile - 1) / kTile, (g.ny + kTile - 1) / kTile);
    matvec_gather_kernel<<<grid, block>>>(g, d_rho, d_x, d_fixed, E0, Emin, d_y);
    CUDA_CHECK_LAST_ERROR("matvec_gather_kernel launch");
}

static void launch_diag(const TopoGrid& g, const float* d_rho, const uint8_t* d_fixed,
                        float E0, float Emin, float* d_diag)
{
    dim3 block(kTile, kTile);
    dim3 grid((g.nx + kTile - 1) / kTile, (g.ny + kTile - 1) / kTile);
    diag_gather_kernel<<<grid, block>>>(g, d_rho, d_fixed, E0, Emin, d_diag);
    CUDA_CHECK_LAST_ERROR("diag_gather_kernel launch");
}

void launch_elem_sensitivity(const TopoGrid& g, const float* d_rho, const float* d_U,
                             float E0, float Emin, float* d_ce, float* d_dc_raw)
{
    dim3 block(kTile, kTile);
    dim3 grid((g.nelx + kTile - 1) / kTile, (g.nely + kTile - 1) / kTile);
    elem_sensitivity_kernel<<<grid, block>>>(g, d_rho, d_U, E0, Emin, d_ce, d_dc_raw);
    CUDA_CHECK_LAST_ERROR("elem_sensitivity_kernel launch");
}

void launch_density_filter(const TopoGrid& g, const float* d_rho,
                           const float* d_dc_raw, float* d_dc_filt)
{
    dim3 block(kTile, kTile);
    dim3 grid((g.nelx + kTile - 1) / kTile, (g.nely + kTile - 1) / kTile);
    density_filter_kernel<<<grid, block>>>(g, d_rho, d_dc_raw, d_dc_filt);
    CUDA_CHECK_LAST_ERROR("density_filter_kernel launch");
}

// 1D grid-stride launch geometry shared by the vector kernels: 256-thread
// blocks (repo default), capped at 4096 blocks — same reasoning as the
// scaffold's launch_saxpy (kernels.cu's ancestor comment for this project).
static void vec_launch_dims(int n, int* blocks, int* threads)
{
    *threads = 256;
    *blocks = (n + *threads - 1) / *threads;
    if (*blocks > 4096) *blocks = 4096;
}

static void vec_copy(int n, const float* src, float* dst)
{
    int b, t; vec_launch_dims(n, &b, &t);
    vec_copy_kernel<<<b, t>>>(n, src, dst);
    CUDA_CHECK_LAST_ERROR("vec_copy_kernel launch");
}
static void vec_combine(int n, float a, const float* x, float b, float* y)
{
    int bl, t; vec_launch_dims(n, &bl, &t);
    vec_combine_kernel<<<bl, t>>>(n, a, x, b, y);
    CUDA_CHECK_LAST_ERROR("vec_combine_kernel launch");
}
static void vec_divide(int n, const float* num, const float* den, float* out)
{
    int b, t; vec_launch_dims(n, &b, &t);
    vec_divide_kernel<<<b, t>>>(n, num, den, out);
    CUDA_CHECK_LAST_ERROR("vec_divide_kernel launch");
}

// dot() — the reduce_dot_kernel wrapper: launch the block-level reduction,
// copy the (few dozen) block partials back, and finish the sum on the host.
// Doing the LAST log2(numBlocks) reduction steps on the host — instead of a
// second reduction kernel — is a deliberate didactic choice matching 08.01's
// "keep the small trailing arithmetic on the host, in plain sight" call:
// numBlocks is at most 4096, a microsecond of host summation, and it keeps
// the CG loop below reading as ordinary C++ instead of a kernel-launch maze.
// (README Exercise 5: fuse this into a single-kernel two-level reduction and
// measure the per-iteration host round-trip this choice costs.)
static float dot(int n, const float* d_a, const float* d_b, std::vector<float>& scratch)
{
    int blocks, threads;
    vec_launch_dims(n, &blocks, &threads);
    scratch.resize(static_cast<size_t>(blocks));

    float* d_partial = nullptr;
    CUDA_CHECK(cudaMalloc(&d_partial, static_cast<size_t>(blocks) * sizeof(float)));
    reduce_dot_kernel<<<blocks, threads, threads * sizeof(float)>>>(n, d_a, d_b, d_partial);
    CUDA_CHECK_LAST_ERROR("reduce_dot_kernel launch");
    CUDA_CHECK(cudaMemcpy(scratch.data(), d_partial, static_cast<size_t>(blocks) * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_partial));

    double acc = 0.0;                              // double accumulator: up to 4096 partials summed,
    for (float v : scratch) acc += v;               // float would lose precision the way 08.01's softmin does
    return static_cast<float>(acc);
}

// ===========================================================================
// launch_topo_cg_solve — Jacobi-preconditioned CG, matrix-free, entirely via
// the gather kernels + vector kernels above. Algorithm derivation and the
// warm-start performance story live in THEORY.md and kernels.cuh's doc
// comment; this function is the direct transcription (compare line-by-line
// against topo_cg_solve_cpu in reference_cpu.cpp — that comparison IS the
// project's §5 GPU-vs-CPU gate).
// ===========================================================================
void launch_topo_cg_solve(const TopoGrid& g, const float* d_rho, const float* d_F,
                          const uint8_t* d_fixed, float* d_U, float E0, float Emin,
                          int max_iters, float rel_tol,
                          int* out_iters, float* out_rel_resid)
{
    const int ndof = 2 * g.nx * g.ny;
    float *d_diag = nullptr, *d_r = nullptr, *d_z = nullptr, *d_p = nullptr, *d_Kp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_diag, static_cast<size_t>(ndof) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_r,    static_cast<size_t>(ndof) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_z,    static_cast<size_t>(ndof) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p,    static_cast<size_t>(ndof) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Kp,   static_cast<size_t>(ndof) * sizeof(float)));
    std::vector<float> scratch;   // reused dot()-partial-sum buffer across every iteration

    // diag(K) depends only on rho, which is FIXED for this whole solve —
    // computed once, outside the CG loop (the "recompute once per outer
    // iteration, not once per CG iteration" saving kernels.cuh promises).
    launch_diag(g, d_rho, d_fixed, E0, Emin, d_diag);

    const float normF = std::sqrt(std::fmax(0.0f, dot(ndof, d_F, d_F, scratch)));
    const float denom = (normF > 1e-30f) ? normF : 1.0f;   // guard the (untestable) all-zero-force case

    // r = F - K*U0  (U0 is the caller's WARM-START guess, possibly nonzero).
    vec_copy(ndof, d_F, d_r);
    launch_matvec(g, d_rho, d_U, d_fixed, E0, Emin, d_Kp);
    vec_combine(ndof, -1.0f, d_Kp, 1.0f, d_r);

    vec_divide(ndof, d_r, d_diag, d_z);   // z = M^-1 r, M = diag(K)
    vec_copy(ndof, d_z, d_p);             // p = z

    float rz_old = dot(ndof, d_r, d_z, scratch);
    int iters_run = max_iters;
    float rel_resid = std::sqrt(std::fmax(0.0f, dot(ndof, d_r, d_r, scratch))) / denom;

    for (int it = 0; it < max_iters; ++it) {
        if (rel_resid < rel_tol) { iters_run = it; break; }

        launch_matvec(g, d_rho, d_p, d_fixed, E0, Emin, d_Kp);
        const float pKp = dot(ndof, d_p, d_Kp, scratch);
        if (std::fabs(pKp) < 1e-30f) { iters_run = it; break; }   // breakdown guard (rare: p -> 0)
        const float alpha = rz_old / pKp;

        vec_combine(ndof, alpha, d_p, 1.0f, d_U);     // U += alpha*p
        vec_combine(ndof, -alpha, d_Kp, 1.0f, d_r);   // r -= alpha*Kp

        rel_resid = std::sqrt(std::fmax(0.0f, dot(ndof, d_r, d_r, scratch))) / denom;
        if (rel_resid < rel_tol) { iters_run = it + 1; break; }

        vec_divide(ndof, d_r, d_diag, d_z);
        const float rz_new = dot(ndof, d_r, d_z, scratch);
        const float beta = rz_new / rz_old;
        vec_combine(ndof, 1.0f, d_z, beta, d_p);      // p = z + beta*p
        rz_old = rz_new;
    }

    if (out_iters) *out_iters = iters_run;
    if (out_rel_resid) *out_rel_resid = rel_resid;

    CUDA_CHECK(cudaFree(d_diag));
    CUDA_CHECK(cudaFree(d_r));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_Kp));
}
