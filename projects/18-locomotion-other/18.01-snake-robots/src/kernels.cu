// ===========================================================================
// kernels.cu — GPU implementation for project 18.01
//              Snake robots: serpenoid gait sweeps (anisotropic friction)
//
// The big idea
// ------------
// The GPU content this project teaches is a PARAMETER SWEEP, the same
// "thread-per-independent-problem" batched-simulation pattern as 08.01's
// rollouts (one thread = one candidate control sequence) and 10.03's farm
// (one thread = one environment) — here, one thread = one CANDIDATE GAIT
// (amplitude, phase offset, temporal frequency). Every thread runs an
// independent multi-second physics simulation (thousands of internal dt
// steps) and reduces it to a handful of scalars (speed, straightness, cost
// of transport). No thread reads another thread's data at any point — by
// construction, the whole point of a design-space sweep — so this kernel
// has NO shared memory and NO atomics, exactly like 08.01/10.03's kernels.
//
// What is DIFFERENT from 08.01/10.03's kernels:
//   * Each thread's "problem" needs a small per-link SCRATCH ARRAY (see
//     snake_step in kernels.cuh) instead of a handful of scalars — the
//     first project in this repo's sampling family whose per-thread state
//     is a little N-link chain, not a point mass. Register footprint is
//     therefore larger (~kNLinks*6 + (kNLinks-1) floats); occupancy is
//     correspondingly lower than 08.01's 4-float rollout state, but the
//     workload (G <= a few thousand gaits) never needed thousands of
//     threads resident at once to saturate the GPU anyway — see the launch
//     wrapper's comment.
//   * The per-thread WORK is much larger (thousands of dt steps, each with
//     dozens of sinf/cosf calls) — this kernel is COMPUTE-bound on
//     transcendental throughput (the GPU's Special Function Units), not
//     memory-bound like SAXPY or bandwidth-bound like a rollout kernel with
//     a large noise array. THEORY.md §GPU-mapping walks the arithmetic.
//
// ALL the physics lives in kernels.cuh's snake_step()/simulate_gait() as
// __host__ __device__ inline functions (the 10.03 pattern) — this file is
// deliberately thin: decode this thread's gait, call simulate_gait, write
// the result. That thinness is not laziness, it is the point: it guarantees
// the GPU path and the CPU oracle (reference_cpu.cpp) run the EXACT SAME
// code, so the §5 VERIFY gate measures floating-point implementation
// differences (sinf/cosf), not two independently-typed algorithms drifting
// apart (CLAUDE.md §12).
//
// Read this after: kernels.cuh (the physics + the state/gait/result
// contracts). Companion oracle: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK / CUDA_CHECK_LAST_ERROR (CLAUDE.md §6.1 rule 7)

// ===========================================================================
// sweep_kernel — one thread = one gait.
//
// Thread-to-data mapping: thread g = blockIdx.x*blockDim.x + threadIdx.x
// owns sweep index g (the SAME flattening decode_gait() inverts everywhere
// in this project — kernels.cuh, reference_cpu.cpp, main.cu all agree).
//
// Memory behavior:
//   global reads  : grid/sim/mu_t/mu_n are KERNEL PARAMETERS (passed by
//                   value, live in constant-like parameter memory —
//                   broadcast to every thread at effectively zero cost,
//                   the same "uniform read" spot on 08.01's memory-spectrum
//                   note as u_nom[t], just promoted one level further by
//                   being parameters instead of a pointed-to array).
//   global writes : SIX coalesced writes at the very end (one float per
//                   output array per thread) — a warp's 32 threads write
//                   32 consecutive floats to each array, the textbook
//                   coalesced pattern (33.01's lesson).
//   registers/local: the ENTIRE multi-thousand-step simulation — every
//                   per-link scratch array inside snake_step(), the
//                   running body state — lives here for the kernel's whole
//                   lifetime. Nothing touches global memory in between
//                   (contrast 08.01, which re-reads eps[t*K+k] from global
//                   every step; this kernel's per-step "input" is a
//                   deterministic function of t, not read from memory).
// No shared memory (gaits share nothing), no atomics, no divergence beyond
// the ragged-tail guard: every thread runs the SAME number of steps
// (sim.n_steps), so — unlike 07.09's field kernels — there is not even the
// usual small amount of data-dependent branching inside the loop.
// ===========================================================================
__global__ void sweep_kernel(GaitGridParams grid, SimParams sim, float mu_t, float mu_n,
                             float* __restrict__ out_distance_m,
                             float* __restrict__ out_straightness,
                             float* __restrict__ out_cot,
                             float* __restrict__ out_effort_j,
                             float* __restrict__ out_final_x_m,
                             float* __restrict__ out_final_y_m,
                             int G)
{
    const int g = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's flattened sweep index
    if (g >= G) return;                                    // ragged-tail guard (G need not be a multiple of blockDim)

    // Decode -> simulate -> reduce: the whole kernel in three calls, all
    // three shared verbatim with the CPU oracle (kernels.cuh's file header
    // explains why that sharing matters for the §5 VERIFY gate).
    const GaitParams gp = decode_gait(g, grid, mu_t, mu_n);
    GaitResult res;
    simulate_gait(gp, sim, res);

    // One coalesced write per output array (see the header comment above).
    out_distance_m[g]   = res.distance_m;
    out_straightness[g] = res.straightness;
    out_cot[g]           = res.cot;
    out_effort_j[g]      = res.effort_j;
    out_final_x_m[g]     = res.final_x_m;
    out_final_y_m[g]      = res.final_y_m;
}

// ===========================================================================
// launch_sweep — host launcher (declared in kernels.cuh). All six output
// pointers are DEVICE pointers of G floats each, allocated by the caller
// (main.cu); this function allocates nothing and frees nothing.
//
// Launch configuration reasoning: block = 128, not the repo's usual 256.
// This kernel's per-thread register footprint is much larger than a rollout
// or farm-step thread (kNLinks-sized scratch arrays, THEORY.md §GPU-mapping
// counts the registers) — a smaller block trades a little scheduling
// overhead for headroom against register-limited occupancy, still a
// multiple of the 32-thread warp. G is typically a few thousand (the
// committed scenario: 32*32*8 = 8192), so even a handful of waves of
// 128-thread blocks comfortably covers an RTX 2080 SUPER's 46 SMs; unlike
// 08.01/10.03 this project never needed to reason about a 4096-block cap —
// G is nowhere near that large.
// ===========================================================================
void launch_sweep(const GaitGridParams& grid, const SimParams& sim, float mu_t, float mu_n,
                  float* d_distance_m, float* d_straightness, float* d_cot, float* d_effort_j,
                  float* d_final_x_m, float* d_final_y_m, int G)
{
    const int block = 128;                          // see the reasoning above
    const int blocks = (G + block - 1) / block;      // ceil(G/block): cover every gait, no cap needed

    sweep_kernel<<<blocks, block>>>(grid, sim, mu_t, mu_n,
                                    d_distance_m, d_straightness, d_cot, d_effort_j,
                                    d_final_x_m, d_final_y_m, G);
    CUDA_CHECK_LAST_ERROR("sweep_kernel launch");
}
