// ===========================================================================
// kernels.cu — GPU kernel for project 02.08
//              Per-point motion deskew with pose interpolation
//
// Role in the project
// -------------------
// The ONE __global__ kernel in this project, plus the small host-side pieces
// that feed it: the __constant__ trajectory buffer and its setter, and the
// launch wrapper owning the grid/block math. All of the actual MATH (SLERP,
// quaternion algebra, the rigid re-projection) lives in kernels.cuh as
// shared HD functions — this file is deliberately thin, because the whole
// point of the deskew kernel is that it is a PURE MAP: read a point + a
// uniform pose, call one function, write a point.
//
// Read this after: main.cu, kernels.cuh.  Read this before: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"

#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// g_traj — the currently-uploaded trajectory, in __constant__ memory.
//
// WHY constant memory (the same reasoning 09.01's robot model and 01.10's
// row LUT give, both cited in kernels.cuh): every thread in the grid reads
// the SAME kMaxTrajSamples*kTrajStride = 512 floats (2 KiB) for a GIVEN
// launch — a textbook broadcast access pattern. Constant memory is cached
// and serves a uniform read to an entire warp in one transaction; the
// alternative (a plain __device__ array read through a pointer) would work
// correctly but forgo that broadcast — measurably slower for an access this
// hot (every thread touches g_traj on every find_bracket_index step).
//
// Sized to kMaxTrajSamples (64), not kDenseSamples (21): the buffer is
// declared once at its ceiling; set_trajectory() below uploads only the
// FIRST n*kTrajStride floats that matter for a given call, and every reader
// (find_bracket_index, interpolate_pose) is told the true n explicitly — the
// unused tail of the buffer is simply never addressed. This headroom is what
// lets a learner's Exercise (denser trajectories) change ONLY the data, not
// this kernel or its buffer declaration.
// ---------------------------------------------------------------------------
__constant__ float g_traj[kMaxTrajSamples * kTrajStride];

// ---------------------------------------------------------------------------
// set_trajectory — see kernels.cuh for the full contract. Validates n before
// touching the device (a silent out-of-range n would either under-fill the
// buffer with stale bytes from a PREVIOUS regime's upload, or overflow it).
// ---------------------------------------------------------------------------
void set_trajectory(const float* host_traj, int n)
{
    if (n < 2 || n > kMaxTrajSamples) {
        std::fprintf(stderr,
            "set_trajectory: n=%d out of range [2,%d] (kMaxTrajSamples)\n",
            n, kMaxTrajSamples);
        std::exit(EXIT_FAILURE);
    }
    // cudaMemcpyToSymbol copies HOST bytes into a __constant__/__device__
    // symbol by NAME (the compiler resolves g_traj's device address at link
    // time) — the standard way to seed constant memory from the host; see
    // util/cuda_check.cuh for what CUDA_CHECK catches here (a malformed
    // symbol reference or a size that would overflow the declared array).
    CUDA_CHECK(cudaMemcpyToSymbol(g_traj, host_traj,
                                  static_cast<size_t>(n) * kTrajStride * sizeof(float)));
}

// ---------------------------------------------------------------------------
// deskew_kernel — one thread per point, a pure MAP (kernels.cuh's
// deskew_one_point derivation is the whole algorithm; this kernel is just
// the thread-to-data mapping around it).
//
// Thread-to-data mapping: thread (blockIdx.x, threadIdx.x) owns global point
// index i = blockIdx.x*blockDim.x + threadIdx.x, and does exactly ONE call
// into the shared math — no loop, no grid-stride (this project's N is at
// most a few thousand points per cohort, comfortably under one launch's
// natural grid size; a grid-stride loop, as 33.01/SAXPY use for very large
// N, would be the right upgrade if N ever grew past ~10 million — README
// Exercise).
//
// Memory behavior: t_points/xyz_local are read once per thread (coalesced:
// adjacent threads read adjacent point indices -> adjacent addresses,
// exactly like every other per-element kernel in this repo); g_traj is a
// UNIFORM broadcast read from constant memory (same address range for
// every thread in the grid, see the g_traj comment above); xyz_out is one
// coalesced write per thread. No shared memory (points share no data with
// each other) and no atomics (each point's output is independent — the
// embarrassingly-parallel property the catalog bullet names explicitly).
// ---------------------------------------------------------------------------
__global__ void deskew_kernel(int n_points, const float* __restrict__ t_points,
                              const float* __restrict__ xyz_local, int n_samples,
                              Pose ref_pose, float* __restrict__ xyz_out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) return;   // guard the ragged last block

    const Vec3 p_local{ xyz_local[i * 3 + 0], xyz_local[i * 3 + 1], xyz_local[i * 3 + 2] };
    const Vec3 p_out = deskew_one_point(g_traj, n_samples, ref_pose, t_points[i], p_local);
    xyz_out[i * 3 + 0] = p_out.x;
    xyz_out[i * 3 + 1] = p_out.y;
    xyz_out[i * 3 + 2] = p_out.z;
}

// ---------------------------------------------------------------------------
// launch_deskew — host wrapper owning the launch configuration.
//
// block = 256 threads (repo-default warp-multiple, good occupancy on
// sm_75..sm_89 — the same default 02.01/08.01/09.01 use). grid =
// ceil(n_points/256): every project cohort has a few thousand points, so
// this is at most a few dozen blocks — nowhere near saturating even one SM
// generation's block-scheduling capacity, which is exactly why NO grid cap
// or grid-stride loop is needed here (contrast SAXPY's 4096-block cap for
// million-element inputs — this kernel's N never approaches that regime;
// see the deskew_kernel comment above for the explicit exercise pointer).
// ---------------------------------------------------------------------------
void launch_deskew(int n_points, const float* d_t_points, const float* d_xyz_local,
                   int n_samples, Pose ref_pose, float* d_xyz_out)
{
    if (n_points <= 0) return;   // 0-point cohort is a valid (if degenerate) no-op

    const int block = 256;
    const int grid = (n_points + block - 1) / block;

    deskew_kernel<<<grid, block>>>(n_points, d_t_points, d_xyz_local, n_samples, ref_pose, d_xyz_out);

    // Kernel launches fail asynchronously (CLAUDE.md §6.1 rule 7) — surface
    // any launch-configuration error (bad grid/block dims, no compatible
    // device code for this GPU's sm_XX) HERE, at the launch site.
    CUDA_CHECK_LAST_ERROR("deskew_kernel launch");
}
