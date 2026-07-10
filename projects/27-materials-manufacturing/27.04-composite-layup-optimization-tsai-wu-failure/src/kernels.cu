// ===========================================================================
// kernels.cu — GPU kernels for project 27.04
//              Composite layup optimization + Tsai-Wu failure envelope sweeps
//
// The big idea
// ------------
// The GPU content this project teaches is TWO independent parameter sweeps,
// the same "thread-per-independent-problem" batched map that 08.01
// (rollouts), 18.01 (gaits), and 33.01 (small linear systems) all share —
// here, one thread solves ONE small (3x3 linear system + 8 quadratic-root
// evaluations) materials problem:
//
//   1. layup_sweep_kernel — one thread per (candidate layup, load case)
//      PAIR: decode the layup, score it against one load direction. This is
//      the literal "one thread per candidate layup x load case" mapping the
//      catalog's scope calls for.
//   2. envelope_kernel    — one thread per (Nx,Ny) grid point: score a FIXED
//      layup against every point of a 128x128 load-space grid, producing
//      the Tsai-Wu failure envelope as a field.
//
// Both kernels are THIN by design: all the actual physics (lamina_Q,
// transform_Qbar, the 3x3 Cramer solve, per-ply stress, the Tsai-Wu
// quadratic) lives in kernels.cuh as __host__ __device__ inline functions
// (the 18.01 pattern) — this file's job is only to decode a thread's index
// into a problem instance, call the shared physics, and write one result.
// That thinness guarantees the GPU path and the CPU oracle
// (reference_cpu.cpp) run the EXACT SAME code, so the §5 VERIFY gate
// measures only floating-point implementation differences (sinf/cosf/sqrtf),
// never two independently-typed algorithms drifting apart (CLAUDE.md §12).
//
// Memory behavior (both kernels): Lamina/AngleAlphabet/Layup8 arrive as
// KERNEL PARAMETERS (passed by value — broadcast to every thread at
// effectively zero cost, the same "uniform, parameter-memory" spot on
// 08.01's memory-spectrum note as u_nom[t]); the load-case array (sweep
// kernel only) is a small DEVICE pointer read — n_cases is at most a few
// dozen floats, read redundantly by every thread in a layup's block, served
// by the L2/read-only cache path after the first touch. Every thread's
// per-ply loop (kNPlies=8 iterations) lives entirely in REGISTERS: no
// shared memory, no atomics, no inter-thread communication anywhere — by
// construction, the whole point of an independent design-space sweep.
//
// Occupancy note: this project's whole problem (kNLayups*n_cases <= 4096,
// or kEnvGridN^2 = 16384) is FAR smaller than a "fill the GPU" workload —
// an RTX 2080 SUPER's 46 SMs can retire either sweep in a handful of
// microseconds of actual kernel time (measured in main.cu's [time] line).
// The teaching point here is the MAPPING (how a materials design search
// becomes threads), not raw throughput — unlike 08.01/18.01, this project
// never needed a block/grid cap; ceil(total/256) is always small.
//
// Read this after: kernels.cuh (the physics + the layup/load-case/envelope
// contracts). Companion oracle: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK / CUDA_CHECK_LAST_ERROR (CLAUDE.md §6.1 rule 7)

// ===========================================================================
// layup_sweep_kernel — one thread = one (layup_id, case_id) pair.
//
// Thread-to-data mapping: thread g = blockIdx.x*blockDim.x + threadIdx.x
// owns flattened index g; layup_id = g / n_cases, case_id = g % n_cases —
// layup slowest-varying, case fastest (matches the flattening documented in
// kernels.cuh and reused by main.cu's host-side per-layup MIN reduction).
//
// Per-thread work: decode_layup (a handful of integer ops) -> ONE call to
// laminate_failure_factor (assemble a 3x3 A matrix, one Cramer solve, 8
// ply-stress + Tsai-Wu-quadratic evaluations) -> ONE coalesced write. The
// heaviest reused local state is angles[kNPlies] (8 floats) — trivially
// register-resident, nowhere near the register pressure of 18.01's
// multi-thousand-step gait integrator.
// ===========================================================================
__global__ void layup_sweep_kernel(Lamina mat, AngleAlphabet alpha,
                                   const LoadCase* __restrict__ cases, int n_cases,
                                   float* __restrict__ out_factor, int total)
{
    const int g = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's flattened (layup, case) index
    if (g >= total) return;                                 // ragged-tail guard

    const int layup_id = g / n_cases;
    const int case_id  = g % n_cases;

    float angles[kNPlies];                    // this thread's 8 ply angles (degrees) — registers
    decode_layup(layup_id, alpha.deg, angles);

    // ONE coalesced write per thread — a warp of 32 consecutive g values
    // writes 32 consecutive floats (the textbook coalesced pattern, 33.01's
    // lesson, applied automatically here since g IS the output index).
    out_factor[g] = laminate_failure_factor(mat, angles, kNPlies, cases[case_id]);
}

// ===========================================================================
// envelope_kernel — one thread = one (Nx, Ny) grid point, for a FIXED layup.
//
// Thread-to-data mapping: thread g owns grid index g; row i = g / kEnvGridN,
// column j = g % kEnvGridN — the SAME row/column convention
// envelope_grid_point() (kernels.cuh) and the PGM writer (main.cu) use.
// ===========================================================================
__global__ void envelope_kernel(Lamina mat, Layup8 layup, float n_max_npm,
                                float* __restrict__ out_factor)
{
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = kEnvGridN * kEnvGridN;
    if (g >= total) return;

    const int i = g / kEnvGridN;   // row    -> Ny
    const int j = g % kEnvGridN;   // column -> Nx

    out_factor[g] = envelope_factor_at(mat, layup.deg, kNPlies, i, j, n_max_npm);
}

// ===========================================================================
// Host launchers (declared in kernels.cuh). Both allocate nothing and free
// nothing — the caller (main.cu) owns every device buffer, matching the
// 08.01/18.01 stateless-launcher convention.
//
// Launch configuration reasoning (both kernels): block = 256 threads — the
// repo default (a warp multiple, comfortable occupancy, small enough
// per-block register pressure). grid = ceil(total/block); no cap is needed
// (unlike 08.01's K=4096-rollout cap) because this project's largest
// problem (the envelope: kEnvGridN^2 = 16384 threads) is already small
// relative to what a single cap-free launch handles instantly on any
// current GPU — see the file header's occupancy note.
// ===========================================================================

void launch_layup_sweep(const Lamina& mat, const AngleAlphabet& alpha,
                        const LoadCase* d_cases, int n_cases,
                        float* d_factor)
{
    const int total = kNLayups * n_cases;
    const int block = 256;
    const int blocks = (total + block - 1) / block;

    layup_sweep_kernel<<<blocks, block>>>(mat, alpha, d_cases, n_cases, d_factor, total);
    CUDA_CHECK_LAST_ERROR("layup_sweep_kernel launch");
}

void launch_envelope(const Lamina& mat, const Layup8& layup, float n_max_npm,
                     float* d_factor)
{
    const int total = kEnvGridN * kEnvGridN;
    const int block = 256;
    const int blocks = (total + block - 1) / block;

    envelope_kernel<<<blocks, block>>>(mat, layup, n_max_npm, d_factor);
    CUDA_CHECK_LAST_ERROR("envelope_kernel launch");
}
