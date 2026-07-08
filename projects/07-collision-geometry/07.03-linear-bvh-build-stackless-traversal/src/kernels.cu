// ===========================================================================
// kernels.cu — GPU kernels for project 07.03 (Linear BVH (Morton codes) build + stackless traversal)
//
// TEMPLATE PLACEHOLDER — replace with this project's real kernels.
// TODO(scaffold): delete the SAXPY kernel and implement the real ones (one
// teaching-focused kernel per concept, each commented to the standard shown
// here and in CLAUDE.md §6.2 / docs/COMMENTING_STANDARD.md).
//
// Role in the project
// -------------------
// All __global__ (GPU) code lives here, together with the small host-side
// launch wrappers that own the grid/block math. Keeping the launch math next
// to the kernel means the launch-configuration reasoning (the comments the
// repo standard requires) sits beside the code it configures.
//
// Big idea of the placeholder kernel
// ----------------------------------
// SAXPY is a pure MAP: out[i] depends only on in[i]. The GPU mapping is
// therefore the simplest one that exists — one thread per element — written
// here in its robust production form, the GRID-STRIDE LOOP. Learn this
// pattern well: a large fraction of the kernels in this repository are maps
// or start from one.
//
// Read this after: main.cu, kernels.cuh.  Read this before: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"           // our own interface — keeps decl/def in sync at compile time
#include "util/cuda_check.cuh"   // CUDA_CHECK_LAST_ERROR for post-launch error surfacing

// ---------------------------------------------------------------------------
// saxpy_kernel — one grid-stride pass computing y[i] = a * x[i] + y[i].
//
// Thread-to-data mapping:
//   Thread (blockIdx.x, threadIdx.x) starts at global element
//       i0 = blockIdx.x * blockDim.x + threadIdx.x
//   and then strides by the TOTAL number of threads in the grid
//       stride = gridDim.x * blockDim.x
//   visiting i0, i0+stride, i0+2*stride, ... < n.
//
// Why a grid-stride loop instead of the naive "one thread = one element,
// return if i >= n"?
//   1) Correct for ANY n, even n larger than the maximum grid size —
//      the loop just runs more iterations per thread.
//   2) Lets the CALLER choose the grid size for occupancy reasons instead of
//      being forced to launch exactly ceil(n/block) blocks.
//   3) It is the idiom used throughout CUDA's own samples and libraries, so
//      learning it here pays off everywhere.
//   The cost: two extra registers and a loop branch — negligible for a
//   memory-bound kernel.
//
// Memory behavior (the whole performance story for SAXPY):
//   Adjacent threads (threadIdx.x, threadIdx.x+1) read adjacent addresses
//   x[i], x[i+1] — so each 32-thread warp touches one contiguous 128-byte
//   span, which the hardware COALESCES into the minimum number of memory
//   transactions. Coalescing is THE first-order GPU optimization; a strided
//   or random access pattern here could cost 10-30x. No shared memory is
//   used because no data is reused between threads — shared memory only pays
//   when threads share or revisit data (see THEORY.md "The GPU mapping").
//
// Parameters:
//   n   — element count (> 0); unitless placeholder data
//   a   — the SAXPY scalar (FP32, exactly representable 2.0 in the demo)
//   x   — [n] device pointer, read-only input. __restrict__ promises the
//         compiler x and y do not alias, unlocking wider loads/scheduling.
//   y   — [n] device pointer, read AND written in place (input y, output
//         a*x+y). In-place is safe because element i never reads element j.
//
// Numerical note: the compiler typically fuses a*x[i]+y[i] into one FMA
// (fused multiply-add, a single rounding step). The CPU reference may round
// twice (mul, then add). Max divergence ~1 ULP — which is exactly why
// main.cu compares with a small tolerance instead of demanding bit equality.
//
// Launch configuration: owned by launch_saxpy() below — see its comment.
// ---------------------------------------------------------------------------
__global__ void saxpy_kernel(int n,
                             float a,
                             const float* __restrict__ x,
                             float*       __restrict__ y)
{
    // This thread's first element, and the whole-grid stride (see mapping
    // note above). Both fit in int here because n is int; a real project
    // handling >2^31 elements would use long long / size_t indexing.
    int i      = blockIdx.x * blockDim.x + threadIdx.x;  // my starting element
    int stride = gridDim.x * blockDim.x;                 // total threads in the grid

    // Each iteration: 2 loads (x[i], y[i]), 1 FMA, 1 store — ~12 bytes moved
    // per 2 FLOPs. Memory-bound, as promised. The loop condition also guards
    // the ragged tail: threads whose i starts beyond n simply do nothing.
    for (; i < n; i += stride) {
        y[i] = a * x[i] + y[i];
    }
}

// ---------------------------------------------------------------------------
// launch_saxpy — host wrapper that owns the launch configuration.
//
// Purpose: keep the <<<grid, block>>> math, its reasoning, and the mandatory
// post-launch error check in ONE place, so callers (main.cu) stay clean and
// no launch in the codebase goes unchecked (CLAUDE.md §6.1 rule 7).
//
// Parameters: as saxpy_kernel, but x/y are DEVICE pointers the CALLER owns —
// this function allocates nothing, frees nothing, and synchronizes nothing
// (the caller times/syncs via events; see main.cu step 3).
//
// Launch configuration reasoning:
//   block = 256 threads — a solid default on sm_75..sm_89: a multiple of the
//     32-thread warp (mandatory for full warps), large enough for good
//     occupancy, small enough to keep per-block resources (registers) free.
//     Powers of two between 128 and 512 are all reasonable; measure before
//     believing any single number.
//   grid = ceil(n / block), capped at 4096 blocks — enough blocks to fill
//     every SM on any current GPU many times over (an RTX 2080 has 46 SMs);
//     beyond that, more blocks add scheduling overhead without adding
//     parallelism, and the grid-stride loop absorbs the remainder anyway.
//     The integer ceil idiom (n + block - 1) / block is used all over this
//     repo — it rounds UP so the last partial block is not lost.
// ---------------------------------------------------------------------------
void launch_saxpy(int n, float a, const float* d_x, float* d_y)
{
    const int block = 256;                              // threads per block (warp multiple; see above)
    int grid = (n + block - 1) / block;                 // ceil(n/block): cover every element
    if (grid > 4096) grid = 4096;                       // cap: grid-stride loop covers the rest

    saxpy_kernel<<<grid, block>>>(n, a, d_x, d_y);

    // Kernel launches return errors ASYNCHRONOUSLY: an invalid configuration
    // or a crashed kernel surfaces on a LATER call unless we ask. This macro
    // (util/cuda_check.cuh) calls cudaGetLastError() right away so a broken
    // launch is reported HERE, at the launch site, not three calls later.
    CUDA_CHECK_LAST_ERROR("saxpy_kernel launch");
}
