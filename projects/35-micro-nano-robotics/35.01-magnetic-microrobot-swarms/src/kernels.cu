// ===========================================================================
// kernels.cu — GPU kernels for project 35.01 (Magnetic microrobot swarms:
//              Biot-Savart field computation + swarm dynamics)
//
// Role in the project
// --------------------
// The four __global__ kernels below implement the pipeline kernels.cuh's
// file header lays out: brute-force Biot-Savart -> linear combine -> stencil
// gradient -> agent-farm dynamics. Every kernel's numerical CORE (the
// physics/math) lives in a HOSTDEV helper shared with reference_cpu.cpp
// (kernels.cuh); what is unique to THIS file is exclusively the THREAD
// MAPPING and the launch-configuration reasoning — read this file to learn
// "how do I turn a nested loop into a kernel," not "what is Biot-Savart"
// (that is THEORY.md's job).
//
// Read this after: main.cu, kernels.cuh.  Read this before: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK_LAST_ERROR for post-launch error surfacing

// ===========================================================================
// Kernel 1/4 — biot_savart_basis_kernel
//
// Thread-to-data mapping: ONE THREAD PER FIELD-EVALUATION GRID CELL. With a
// 256x256 grid that is 65536 threads; grid = ceil(65536/256) = 256 blocks of
// 256 threads (a clean, occupancy-friendly 1D launch — no need for a 2D
// block/grid shape since the row-major index i=iy*grid_n+ix already flattens
// the 2D problem into a 1D one, exactly as biot_savart_basis_cpu's nested
// loop does with `int idx = iy*grid_n+ix`).
//
// Per thread: recover (ix,iy) from the flat index, convert to world (x,y)
// via grid_to_world, then loop over all n_segs segments — EXACTLY the
// catalog bullet's "sum over ALL segments of I dl x r / |r|^3" — accepting
// or skipping each one by comparing its coil_id against active_coil. This
// is the classic MAP+REDUCE-PER-THREAD pattern: independent output cells
// (map), each computed by reducing (summing) over a shared input array
// (the segment list) that every thread reads identically.
//
// Memory behavior: segs[] is READ IDENTICALLY by every thread in the grid
// (a broadcast access pattern) — small enough (720 segments x 28 bytes =
// ~20 KB) to live comfortably in L2/read-only cache across the whole
// kernel's lifetime, so after the first few blocks warm the cache, this
// kernel is COMPUTE-bound (the cross product + rsqrt per segment), not
// memory-bound, despite touching every segment 65536 times. Bx/By writes
// are fully coalesced: consecutive threads (consecutive flat idx) write
// consecutive addresses.
//
// This is the expensive kernel (256x256 grid x 720 segments = ~4.7e7
// segment evaluations) — and, by design (kernels.cuh step 2), the ONLY
// kernel that ever touches segs[] at all. It runs exactly NUM_COILS=4
// times for the whole demo.
// ===========================================================================
__global__ void biot_savart_basis_kernel(
    const CoilSegment* __restrict__ segs, int n_segs, int active_coil,
    int grid_n, float half_m,
    float* __restrict__ Bx, float* __restrict__ By)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;   // flat grid-cell index, 0..grid_n*grid_n-1
    const int total_cells = grid_n * grid_n;
    if (idx >= total_cells) return;   // guard: grid_n*grid_n may not be an exact multiple of blockDim.x

    const int ix = idx % grid_n;   // recover 2D coords from the flat index (row-major: idx = iy*grid_n+ix)
    const int iy = idx / grid_n;
    const float x = grid_to_world(ix, grid_n, half_m);   // this thread's field-evaluation point (z=0 plane)
    const float y = grid_to_world(iy, grid_n, half_m);

    // Local accumulators live in REGISTERS — private per thread, updated
    // n_segs times, never shared between threads (the "no shared memory
    // needed" case: every thread's work is fully independent, unlike
    // 24.01's SOR stencil which genuinely reads neighbors).
    float acc_x = 0.0f, acc_y = 0.0f, acc_z = 0.0f;

    for (int s = 0; s < n_segs; ++s) {
        const CoilSegment seg = segs[s];        // one 28-byte struct load per iteration, shared across all 65536 threads
        if (seg.coil_id != active_coil) continue;   // unit current lives ONLY on active_coil; others contribute zero
        biot_savart_contribution(seg.mx, seg.my, seg.mz, seg.dlx, seg.dly, seg.dlz,
                                 1.0f,   // unit ampere-turn — this call is building a PER-UNIT-CURRENT basis map
                                 x, y, 0.0f,
                                 acc_x, acc_y, acc_z);
    }

    Bx[idx] = acc_x;   // one coalesced write; Bz (acc_z) is intentionally not stored (kernels.cuh explains why)
    By[idx] = acc_y;
}

void launch_biot_savart_basis(const CoilSegment* d_segs, int n_segs, int active_coil,
                              int grid_n, float half_m, float* d_Bx, float* d_By)
{
    const int total_cells = grid_n * grid_n;
    const int block = 256;                                  // warp multiple, the repo default (08.01, 24.01 use the same)
    const int grid  = (total_cells + block - 1) / block;    // ceil: cover every cell exactly once (no grid-stride needed — 65536 is a modest, fixed size)
    biot_savart_basis_kernel<<<grid, block>>>(d_segs, n_segs, active_coil, grid_n, half_m, d_Bx, d_By);
    CUDA_CHECK_LAST_ERROR("biot_savart_basis_kernel launch");
}

// ===========================================================================
// Kernel 2/4 — combine_field_kernel
//
// Thread-to-data mapping: one thread per grid cell (same 256x256 -> 65536
// -> 256 blocks x 256 threads shape as kernel 1). Per thread: read the SAME
// cell index out of all 4 basis maps, form the linear combination
// B(x) = sum_c I_coil[c]*basis_c(x). This is a pure MAP — no segment loop,
// no neighbor reads — the cheapest kernel in the pipeline, and it is what
// buys the swarm loop its speed: every one of the 3 waypoint-schedule
// phases (and the illustrative field-magnitude artifact) calls this
// instead of re-running kernel 1's 4.7e7-segment-evaluation sum.
//
// Memory behavior: basisBx/basisBy are read with a STRIDE of `grid_cells`
// between the 4 coils (layout: [coil][cell], coil-major) — within one
// coil's slice, consecutive threads read consecutive addresses
// (coalesced); across coils, each thread makes 4 separate coalesced reads
// rather than 1 strided read, which is why the layout is coil-major and
// not cell-major (a cell-major [cell][coil] layout would make every
// thread's 4-float read for one cell coalesce beautifully AS A GROUP, but
// would break the coalescing of kernel 1's writes, which produce one whole
// coil's map contiguously; coil-major is the layout that keeps BOTH
// kernel 1's writes and kernel 2's reads coalesced, since kernel 1 must
// finish writing one full coil map per launch).
// ===========================================================================
__global__ void combine_field_kernel(
    const float* __restrict__ basisBx, const float* __restrict__ basisBy,
    Float4 I_coil, int grid_cells,
    float* __restrict__ Bx, float* __restrict__ By)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= grid_cells) return;

    // Unrolled 4-term dot product (NUM_COILS is a compile-time 4, so the
    // compiler fully unrolls this regardless — written out explicitly here
    // for readability, matching combine_field_cpu's loop line for line in
    // spirit if not in literal source form).
    const float bx = I_coil.x * basisBx[0 * grid_cells + i]
                    + I_coil.y * basisBx[1 * grid_cells + i]
                    + I_coil.z * basisBx[2 * grid_cells + i]
                    + I_coil.w * basisBx[3 * grid_cells + i];
    const float by = I_coil.x * basisBy[0 * grid_cells + i]
                    + I_coil.y * basisBy[1 * grid_cells + i]
                    + I_coil.z * basisBy[2 * grid_cells + i]
                    + I_coil.w * basisBy[3 * grid_cells + i];
    Bx[i] = bx;
    By[i] = by;
}

void launch_combine_field(const float* d_basisBx, const float* d_basisBy, Float4 I_coil,
                          int grid_n, float* d_Bx, float* d_By)
{
    const int cells = grid_n * grid_n;
    const int block = 256;
    const int grid  = (cells + block - 1) / block;
    combine_field_kernel<<<grid, block>>>(d_basisBx, d_basisBy, I_coil, cells, d_Bx, d_By);
    CUDA_CHECK_LAST_ERROR("combine_field_kernel launch");
}

// ===========================================================================
// Kernel 3/4 — gradient_b2_kernel
//
// Thread-to-data mapping: one thread per grid cell — but unlike kernels 1
// and 2, this is a genuine STENCIL: thread (ix,iy) reads its FOUR NEIGHBORS
// (ix±1,iy) and (ix,iy±1), not just its own cell. This is the same access
// pattern 24.01's SOR solver teaches (a 5-point stencil), applied here to a
// one-shot finite difference rather than an iterative relaxation.
//
// Memory behavior: no shared-memory tiling is used, deliberately — at
// 256x256 cells (2 KB rows), the whole Bx/By pair (2 * 65536 * 4 bytes =
// 512 KB) does not fit in one SM's shared memory, and this kernel runs only
// 3 times total (once per waypoint-schedule phase) rather than in a hot
// per-step loop, so the L2-cache reuse between neighboring threads' overlapping
// reads (thread ix's "+1" neighbor is thread ix+1's own cell) is already
// enough — this is the honest "measure before tiling" lesson (THEORY.md
// "The GPU mapping" elaborates; 24.01's SOR kernel is the sibling project
// where tiling WOULD start to matter, because it iterates thousands of times).
// ===========================================================================
__global__ void gradient_b2_kernel(
    const float* __restrict__ Bx, const float* __restrict__ By,
    int grid_n, float half_m,
    float* __restrict__ gx, float* __restrict__ gy)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_cells = grid_n * grid_n;
    if (idx >= total_cells) return;

    const int ix = idx % grid_n;
    const int iy = idx / grid_n;
    const float h = (2.0f * half_m) / static_cast<float>(grid_n - 1);   // grid spacing (m)

    // b2_at: |B|^2 at a given (clamped-implicitly-by-caller) grid index.
    // Written as a local lambda-like inline helper via a small macro-free
    // function pointer would be overkill in a kernel — a tiny local lambda
    // is fine in device code (nvcc supports C++17 lambdas in __global__
    // bodies) and keeps the four neighbor reads below readable.
    auto b2_at = [&](int gx_i, int gy_i) -> float {
        const int j = gy_i * grid_n + gx_i;
        const float bx = Bx[j], by = By[j];
        return bx * bx + by * by;
    };

    // Central difference where both neighbors exist; one-sided (half-step)
    // difference at the two edges — identical policy to gradient_b2_cpu, so
    // VERIFY_FIELD-style comparisons never see a boundary-handling mismatch.
    float dB2dx;
    if (ix > 0 && ix < grid_n - 1) {
        dB2dx = (b2_at(ix + 1, iy) - b2_at(ix - 1, iy)) / (2.0f * h);
    } else if (ix == 0) {
        dB2dx = (b2_at(ix + 1, iy) - b2_at(ix, iy)) / h;
    } else {
        dB2dx = (b2_at(ix, iy) - b2_at(ix - 1, iy)) / h;
    }

    float dB2dy;
    if (iy > 0 && iy < grid_n - 1) {
        dB2dy = (b2_at(ix, iy + 1) - b2_at(ix, iy - 1)) / (2.0f * h);
    } else if (iy == 0) {
        dB2dy = (b2_at(ix, iy + 1) - b2_at(ix, iy)) / h;
    } else {
        dB2dy = (b2_at(ix, iy) - b2_at(ix, iy - 1)) / h;
    }

    gx[idx] = dB2dx;
    gy[idx] = dB2dy;
}

void launch_gradient_b2(const float* d_Bx, const float* d_By, int grid_n, float half_m,
                        float* d_gx, float* d_gy)
{
    const int cells = grid_n * grid_n;
    const int block = 256;
    const int grid  = (cells + block - 1) / block;
    gradient_b2_kernel<<<grid, block>>>(d_Bx, d_By, grid_n, half_m, d_gx, d_gy);
    CUDA_CHECK_LAST_ERROR("gradient_b2_kernel launch");
}

// ===========================================================================
// Kernel 4/4 — swarm_step_kernel
//
// Thread-to-data mapping: ONE THREAD PER ROBOT — the "agent farm" pattern
// this repo uses for every independent-agent simulation (08.01's rollouts,
// 22.01's swarm agents): thread k owns robot k's (x,y) position in
// registers for the ENTIRE `steps`-iteration loop, exactly as 08.01's
// rollout kernel owns one cart-pole state for its whole T-step horizon.
// With n_robots=1000 and block=256, grid=ceil(1000/256)=4 blocks — a small
// launch (this repo's kernels routinely launch thousands of threads; 1000
// is intentionally on the small side, matching a real coil system's
// practical swarm size for a desktop demo, not a GPU occupancy target).
//
// Memory behavior: gx/gy (the precomputed gradient maps, kernel 3's output)
// are read via bilinear_sample — a scattered, per-robot-position-dependent
// 2x2 neighborhood read. Because DIFFERENT robots are usually at DIFFERENT
// positions, this access pattern does not coalesce the way kernels 1-3's
// per-cell writes do; it behaves like a small random-access texture lookup.
// At only 1000 robots x 4 reads each = 4000 scattered reads per step, this
// is nowhere near a bottleneck (gx/gy together are 512 KB, comfortably
// L2-resident) — but it IS the reason a texture-memory or __ldg()-cached
// read would be the next optimization on a much larger swarm (README
// Exercise territory), a fact worth knowing even though it is not needed here.
//
// No atomics, no inter-thread communication: every robot's trajectory is
// fully independent, by construction — the swarm interacts only through
// the SHARED field map it all reads, never through each other directly
// (this project does not model bead-bead magnetic dipole interactions or
// hydrodynamic coupling — an honest, named limitation, see README).
// ===========================================================================
__global__ void swarm_step_kernel(
    const float* __restrict__ gx, const float* __restrict__ gy,
    int grid_n, float half_m,
    float* __restrict__ rx, float* __restrict__ ry, int n_robots,
    float k_force, float gamma, float dt_s, int steps)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's robot index
    if (k >= n_robots) return;                              // guard the ragged last block (1000 % 256 != 0)

    // Register-resident state: loaded once, updated `steps` times, written
    // back once — exactly the register-residency discipline 08.01's rollout
    // kernel documents (CLAUDE.md §6.2's own worked example).
    float x = rx[k];
    float y = ry[k];

    for (int s = 0; s < steps; ++s) {
        const float dB2dx = bilinear_sample(gx, grid_n, half_m, x, y);
        const float dB2dy = bilinear_sample(gy, grid_n, half_m, x, y);
        const float Fx = k_force * 0.5f * dB2dx;   // F = k_force*grad(|B|^2/2); THEORY.md "The math" derives this
        const float Fy = k_force * 0.5f * dB2dy;
        const float vx = Fx / gamma;                // v = F/gamma (Stokes drag; Re<<1 => no inertia term)
        const float vy = Fy / gamma;
        x += vx * dt_s;                              // explicit Euler; THEORY.md "Numerical considerations" justifies dt_s
        y += vy * dt_s;
    }

    rx[k] = x;
    ry[k] = y;
}

void launch_swarm_step(const float* d_gx, const float* d_gy, int grid_n, float half_m,
                       float* d_rx, float* d_ry, int n_robots,
                       float k_force, float gamma, float dt_s, int steps)
{
    const int block = 256;
    const int grid  = (n_robots + block - 1) / block;
    swarm_step_kernel<<<grid, block>>>(d_gx, d_gy, grid_n, half_m, d_rx, d_ry, n_robots,
                                       k_force, gamma, dt_s, steps);
    CUDA_CHECK_LAST_ERROR("swarm_step_kernel launch");
}
