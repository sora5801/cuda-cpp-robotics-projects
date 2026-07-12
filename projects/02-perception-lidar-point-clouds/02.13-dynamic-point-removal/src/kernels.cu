// ===========================================================================
// kernels.cu — GPU implementation for project 02.13
//              Dynamic point removal (raycast free-space carving)
//
// The big idea
// ------------
// K posed scans arrive as 14,400 independent BEAMS. Every beam's evidence —
// "empty along here, occupied (or nothing) at the end" — is entirely its
// own: no beam needs to know about any other beam while it marches. That
// independence is this project's GPU mapping: one thread per BEAM, the
// classic thread-per-problem shape 08.01/09.01/33.01 all use, here applied
// to a voxel-grid RAY MARCH instead of an ODE rollout or a batched solve.
// Contention only appears at the WRITE side (many beams' marches converge on
// the same handful of voxels near the sensor — kernels.cu's carve_kernel
// comment below measures this; main.cu's [info] contention gate reports the
// actual numbers from this machine's run).
//
// What is NEW here beyond 05.01/08.01/09.01/33.01:
//   * the Amanatides & Woo (1987) voxel DDA march itself — a beam's path is
//     turned into an ORDERED SEQUENCE of integer voxel coordinates using
//     only integer step decisions driven by three per-axis "time to next
//     boundary" floats (tMaxX/Y/Z) — no per-step distance computation, no
//     resampling, no missed or double-counted voxel (the classic exact
//     voxel-traversal algorithm; THEORY.md "The algorithm" derives it in
//     full, kernels.cuh's file header gives the five-line summary);
//   * a THREE-WAY atomic ledger (hits / pass_from_hit / pass_from_maxrange)
//     instead of a single counter — chosen so main.cu's max_range_carving
//     gate can prove, per voxel, WHICH kind of evidence carved it;
//   * a kernel with NO Ledger argument at all (carve_trace_kernel) — pure
//     instrumentation, sharing its march logic with carve_kernel through one
//     __device__ helper (carve_one_beam) so the two kernels can never drift
//     apart on what "the DDA march" means, while main.cu's verify stage
//     still gets an ORDERED sequence to diff against the CPU twin, not just
//     a final ledger state.
//
// All layouts and constants come from kernels.cuh — the single source
// shared with the CPU twin; carve_one_beam below is NOT shared with
// reference_cpu.cpp's carve_one_beam_cpu (the independence ruling's
// default: the algorithmic core is typed twice). What IS shared (via
// kernels.cuh's HD functions) is only the voxel-indexing arithmetic — see
// kernels.cuh's file header for why that split is the correct one here.
//
// DETERMINISM: every per-axis tMax/tDelta below is built from an explicit
// fmaf() (device intrinsic) whose CPU twin uses std::fmaf identically (the
// same 05.01 determinism contract: nvcc would contract a*b+c into an fma
// anyway and MSVC would not, so writing it out makes both paths execute
// IDENTICAL IEEE-754 operations in the same order). Because the DDA march
// contains no transcendental function at all (only floor, multiply, add,
// divide, compare), this determinism is exact enough that main.cu's verify
// stage requires the GPU and CPU voxel SEQUENCES to match EXACTLY, integer
// for integer, not just within a tolerance (kernels.cuh's file header calls
// this out; THEORY.md "Numerical considerations" gives the full argument).
//
// Read this after: kernels.cuh.  Companion oracle: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"      // CUDA_CHECK / CUDA_CHECK_LAST_ERROR (§6.1 rule 7)

#include <cstdio>
#include <cstdlib>

// Repo default launch geometry: 256-thread 1-D blocks. Every kernel below is
// either a simple per-voxel fill or a per-beam march whose per-thread work
// (up to kMaxDDASteps atomic updates) varies with distance to the nearest
// surface — a workload imbalance INSIDE warps (some threads finish after a
// handful of steps, others march the full grid), the honest cost of mapping
// a variable-length algorithm onto fixed-width SIMT hardware (THEORY.md "The
// GPU mapping" discusses this divergence explicitly; it costs cycles, not
// correctness — every thread still visits every voxel ITS OWN beam requires).
static constexpr int kThreads = 256;

// ---------------------------------------------------------------------------
// __constant__ scan origins — the K sensor positions, set once per program
// run via set_scan_origins() (kernels.cuh's file header explains the
// broadcast-read reasoning: beams are scan-major, so most warps share one
// scan_id and therefore one origin). kNumScans*3 floats = 120 bytes, a
// vanishing fraction of the 64 KiB constant bank (same order of magnitude as
// 02.08's trajectory upload).
// ---------------------------------------------------------------------------
__constant__ float c_scan_origin[kNumScans * 3];

void set_scan_origins(const float* host_origin_xyz)
{
    CUDA_CHECK(cudaMemcpyToSymbol(c_scan_origin, host_origin_xyz,
                                  kNumScans * 3 * sizeof(float)));
}

// ===========================================================================
// carve_one_beam — the DDA march, shared by carve_kernel and
// carve_trace_kernel (both device-side call sites in THIS file — this is
// ordinary code reuse within the GPU translation unit, not the host/device
// sharing the independence ruling in kernels.cuh's file header is about;
// reference_cpu.cpp's twin below is typed completely independently).
//
// Parameters:
//   ox,oy,oz, dx,dy,dz : beam origin (m, world) and UNIT direction.
//   range_m            : how far this beam's ledger contribution should
//                         extend — the (possibly noisy) hit range, or
//                         EXACTLY kMaxRangeM for a miss (kernels.cuh "BEAM
//                         RECORD LAYOUT").
//   is_hit             : whether this beam's terminal voxel becomes a HIT
//                         (true) or one more PASS (false, the max-range case).
//   hits/pass_hit/pass_maxrange : the three ledger arrays, or ALL THREE
//                         nullptr to skip ledger updates entirely (the
//                         trace-only call from carve_trace_kernel).
//   trace_out/trace_cap/trace_len : if trace_out != nullptr, every voxel
//                         this march visits (in order, AFTER the start
//                         voxel — the self-carve guard) is appended here, up
//                         to trace_cap entries, and *trace_len is set to the
//                         count actually written.
//
// THE ALGORITHM (Amanatides & Woo 1987; THEORY.md "The algorithm" derives
// every line below from first principles — this comment is the summary):
//   1. cur       = the voxel containing the ORIGIN (never itself marked —
//                  the self-carve guard, kernels.cuh file header).
//   2. target    = the voxel containing the ENDPOINT (origin + dir*range_m),
//                  computed DIRECTLY (not by marching) — this is what makes
//                  the stopping condition an EXACT INTEGER comparison
//                  instead of a fragile floating-point distance check
//                  (THEORY.md "Numerical considerations" explains why that
//                  matters: two different float computations of "have I
//                  gone far enough" can disagree by one voxel right at a
//                  boundary; comparing already-decided INTEGER coordinates
//                  cannot).
//   3. Per axis, compute:
//        step[axis]   : which way the voxel INDEX moves when crossing a
//                       boundary on this axis (+1, -1, or 0 if dir[axis]==0)
//        tMax[axis]   : the ray parameter t at which the NEXT boundary on
//                       this axis is crossed (a "time to collision" per axis)
//        tDelta[axis] : how much t advances to cross ONE MORE whole voxel on
//                       this axis (constant per beam: voxel_size / |dir[axis]|)
//   4. Repeatedly advance along whichever axis has the SMALLEST tMax (that
//      is the next boundary the ray reaches, on ANY axis) — step that axis's
//      voxel index by step[axis], and add tDelta[axis] to that axis's tMax
//      so the next comparison is fair. This visits every voxel the ray
//      passes through, IN ORDER, exactly once, using only additions and
//      comparisons per step (no per-step sqrt/divide) — the algorithm's
//      whole appeal.
//   5. Stop when cur == target (mark it HIT if is_hit, else one more PASS)
//      or when cur exits the grid (the local-map boundary — kernels.cuh
//      file header "Voxel grid").
// ---------------------------------------------------------------------------
__device__ __forceinline__ void carve_one_beam(
    float ox, float oy, float oz,
    float dx, float dy, float dz,
    float range_m, bool is_hit,
    unsigned int* __restrict__ hits,
    unsigned int* __restrict__ pass_hit,
    unsigned int* __restrict__ pass_maxrange,
    int* __restrict__ trace_out, int trace_cap, int* __restrict__ trace_len)
{
    // Step 1: start voxel (never marked — self-carve guard).
    int cx = voxel_coord_axis(ox, kGridOriginX, kVoxelSizeM);
    int cy = voxel_coord_axis(oy, kGridOriginY, kVoxelSizeM);
    int cz = voxel_coord_axis(oz, kGridOriginZ, kVoxelSizeM);

    // Step 2: target voxel, computed DIRECTLY from the endpoint (not by
    // marching) — the exact-integer stopping condition (function comment).
    const float ex = fmaf(dx, range_m, ox);
    const float ey = fmaf(dy, range_m, oy);
    const float ez = fmaf(dz, range_m, oz);
    int tx, ty, tz;
    world_to_voxel(ex, ey, ez, tx, ty, tz);

    int n_trace = 0;

    // Degenerate case: origin and target share a voxel (a near-zero-range
    // hit — noise can occasionally produce one). Nothing to march; mark the
    // hit if this was a real return, emit an empty trace, and return.
    if (cx == tx && cy == ty && cz == tz) {
        if (is_hit && hits != nullptr && voxel_in_bounds(cx, cy, cz))
            atomicAdd(&hits[voxel_index(cx, cy, cz)], 1u);
        if (trace_len != nullptr) *trace_len = 0;
        return;
    }

    // Step 3: per-axis step/tMax/tDelta (Amanatides & Woo). fmaf() spells
    // every multiply-add explicitly for the GPU/CPU determinism contract
    // (file header). AXIS_SETUP is a local macro purely to avoid writing
    // this 12-line block three times with x/y/z substituted by hand — it
    // expands inline, there is no hidden control flow, and it is undef'd
    // immediately after use (never leaks into the rest of the file).
#define AXIS_SETUP(O, D, ORIGIN, CVOX, STEP, TMAX, TDELTA)                    \
    int STEP; float TMAX, TDELTA;                                            \
    if ((D) > 0.0f) {                                                        \
        STEP = 1;                                                            \
        const float boundary = fmaf(static_cast<float>((CVOX) + 1), kVoxelSizeM, (ORIGIN)); \
        TMAX = (boundary - (O)) / (D);                                       \
        TDELTA = kVoxelSizeM / (D);                                          \
    } else if ((D) < 0.0f) {                                                 \
        STEP = -1;                                                           \
        const float boundary = fmaf(static_cast<float>(CVOX), kVoxelSizeM, (ORIGIN));       \
        TMAX = (boundary - (O)) / (D);                                       \
        TDELTA = kVoxelSizeM / (-(D));                                       \
    } else {                                                                 \
        STEP = 0;                                                            \
        TMAX = 3.402823466e38f;   /* FLT_MAX: this axis never crosses a boundary */ \
        TDELTA = 3.402823466e38f;                                            \
    }
    AXIS_SETUP(ox, dx, kGridOriginX, cx, stepx, tMaxX, tDeltaX)
    AXIS_SETUP(oy, dy, kGridOriginY, cy, stepy, tMaxY, tDeltaY)
    AXIS_SETUP(oz, dz, kGridOriginZ, cz, stepz, tMaxZ, tDeltaZ)
#undef AXIS_SETUP

    // Step 4-5: march. kMaxDDASteps bounds the loop against any bug turning
    // this into an infinite spin (kernels.cuh's constant comment justifies
    // the number); a well-formed beam always stops earlier, by reaching
    // target or leaving the grid.
    for (int steps = 0; steps < kMaxDDASteps; ++steps) {
        // Advance along whichever axis reaches its next boundary soonest.
        if (tMaxX <= tMaxY && tMaxX <= tMaxZ) { cx += stepx; tMaxX += tDeltaX; }
        else if (tMaxY <= tMaxZ)              { cy += stepy; tMaxY += tDeltaY; }
        else                                   { cz += stepz; tMaxZ += tDeltaZ; }

        if (!voxel_in_bounds(cx, cy, cz))
            break;   // exited the local map — stop carving, the honest local-map behavior

        const bool reached = (cx == tx && cy == ty && cz == tz);
        const int v = voxel_index(cx, cy, cz);

        if (reached && is_hit) {
            // Endpoint exclusion in action: this voxel becomes a HIT, never
            // a pass, for this beam.
            if (hits != nullptr) atomicAdd(&hits[v], 1u);
            if (trace_out != nullptr && n_trace < trace_cap) trace_out[n_trace++] = v;
            break;
        }

        // Every other visited voxel (including the max-range beam's OWN
        // terminal voxel, per "max-range beams carve but never hit" in
        // kernels.cuh's file header) is a PASS, split by beam kind so
        // main.cu's max_range_carving gate can isolate the evidence source.
        if (is_hit) { if (pass_hit != nullptr) atomicAdd(&pass_hit[v], 1u); }
        else        { if (pass_maxrange != nullptr) atomicAdd(&pass_maxrange[v], 1u); }
        if (trace_out != nullptr && n_trace < trace_cap) trace_out[n_trace++] = v;

        if (reached) break;   // max-range beam: just marked its terminal voxel PASS, done
    }

    if (trace_len != nullptr) *trace_len = n_trace;
}

// ===========================================================================
// Kernel 0: reset the ledger. One thread per voxel, a pure fill (same shape
// as 05.01's volume_clear_kernel).
// ===========================================================================
__global__ void ledger_clear_kernel(Ledger ledger)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= kNumVoxels) return;
    ledger.hits[v] = 0u;
    ledger.pass_from_hit[v] = 0u;
    ledger.pass_from_maxrange[v] = 0u;
}

// ===========================================================================
// Kernel 1: carve_kernel — the bulk carving pass.
//
// Thread-to-data mapping: thread i = blockIdx.x*blockDim.x + threadIdx.x
// owns beam i; beams never interact except through the SHARED ledger, which
// is exactly why atomics are required (see the contention discussion below).
//
// Memory spaces per thread:
//   global (read)  : scan_id[i] (used once, to index c_scan_origin — see
//                    below), dir[i*3+0..2] (interleaved — kernels.cuh's file
//                    header documents the coalescing trade), is_hit[i],
//                    range[i]; all four arrays are read EXACTLY ONCE per
//                    thread, fully coalesced except dir's interleaving.
//   constant       : c_scan_origin[scan_id[i]*3+0..2] — a broadcast read
//                    within any warp whose threads share a scan_id (the
//                    scan-major layout makes that the common case).
//   global (write) : the ledger, via atomicAdd — see below.
//   registers      : the whole DDA march state (cx,cy,cz,tMaxX/Y/Z,...).
//
// CONTENTION, honestly (THEORY.md "The GPU mapping" measures this on real
// data; main.cu's [info] "contention" lines report the ACTUAL counts from
// this run, never asserted, always measured): every beam in a scan starts
// from the SAME sensor voxel and immediately fans out. In the first few
// marching steps, many beams are still close together — near-sensor voxels
// receive PASS increments from a large fraction of that scan's ~1,440
// beams, while a voxel 15 m away is typically touched by only the handful
// of beams whose direction actually threads that specific spot. Near-sensor
// voxels are therefore atomicAdd HOTSPOTS: many threads contending for the
// same memory address, serialized by the hardware's atomic unit. Far voxels
// see near-zero contention. This is the SAME "low contention except near
// the sensor" story every raycasting-into-a-grid kernel tells (OctoMap's
// own insertPointCloud documentation makes the identical observation) — the
// atomic-per-voxel design this project teaches is the right tool BECAUSE
// the hot region is small and the vast majority of the grid is uncontended,
// not despite it (an all-atomics design that was hot EVERYWHERE would need
// the shared-memory local-histogram trick 07.09/23.01 teach instead).
// ===========================================================================
__global__ void carve_kernel(int n_beams,
                             const int* __restrict__ scan_id,
                             const float* __restrict__ dir,
                             const unsigned char* __restrict__ is_hit,
                             const float* __restrict__ range,
                             Ledger ledger)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_beams) return;   // ragged-tail guard

    const int sid = scan_id[i];
    const float ox = c_scan_origin[sid * 3 + 0];
    const float oy = c_scan_origin[sid * 3 + 1];
    const float oz = c_scan_origin[sid * 3 + 2];

    carve_one_beam(ox, oy, oz,
                   dir[i * 3 + 0], dir[i * 3 + 1], dir[i * 3 + 2],
                   range[i], is_hit[i] != 0,
                   ledger.hits, ledger.pass_from_hit, ledger.pass_from_maxrange,
                   /*trace_out=*/nullptr, /*trace_cap=*/0, /*trace_len=*/nullptr);
}

// ===========================================================================
// Kernel 2: carve_trace_kernel — the verify-stage instrumentation pass.
// Same march (carve_one_beam), no ledger, ordered voxel sequence out
// instead. Launched only on a small documented subset (main.cu prints its
// size) — kernels.cuh's declaration comment gives the buffer layout.
// ===========================================================================
__global__ void carve_trace_kernel(int n_beams,
                                   const int* __restrict__ scan_id,
                                   const float* __restrict__ dir,
                                   const unsigned char* __restrict__ is_hit,
                                   const float* __restrict__ range,
                                   int* __restrict__ trace_out,
                                   int* __restrict__ trace_len)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_beams) return;

    const int sid = scan_id[i];
    const float ox = c_scan_origin[sid * 3 + 0];
    const float oy = c_scan_origin[sid * 3 + 1];
    const float oz = c_scan_origin[sid * 3 + 2];

    carve_one_beam(ox, oy, oz,
                   dir[i * 3 + 0], dir[i * 3 + 1], dir[i * 3 + 2],
                   range[i], is_hit[i] != 0,
                   /*hits=*/nullptr, /*pass_hit=*/nullptr, /*pass_maxrange=*/nullptr,
                   trace_out + static_cast<size_t>(i) * kMaxDDASteps, kMaxDDASteps,
                   &trace_len[i]);
}

// ===========================================================================
// Kernel 3: classify_kernel — one thread per BEAM; threads for max-range
// beams (no point) exit immediately after writing the sentinel.
//
// Memory spaces: the ledger is read-only here (three GATHER reads at the
// point's own voxel — scattered across the grid, no coalescing to exploit,
// the same honest stencil-adjacent-read story 05.01's marching-cubes corner
// loads tell); score_out/label_out are one coalesced write each.
// ===========================================================================
__global__ void classify_kernel(int n_beams,
                                const int* __restrict__ scan_id,
                                const float* __restrict__ dir,
                                const unsigned char* __restrict__ is_hit,
                                const float* __restrict__ range,
                                Ledger ledger, float threshold,
                                float* __restrict__ score_out,
                                int* __restrict__ label_out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_beams) return;

    if (is_hit[i] == 0) {
        score_out[i] = -1.0f;   // sentinel: no point, not applicable (kernels.cuh declaration comment)
        label_out[i] = -1;
        return;
    }

    const int sid = scan_id[i];
    const float px = fmaf(dir[i * 3 + 0], range[i], c_scan_origin[sid * 3 + 0]);
    const float py = fmaf(dir[i * 3 + 1], range[i], c_scan_origin[sid * 3 + 1]);
    const float pz = fmaf(dir[i * 3 + 2], range[i], c_scan_origin[sid * 3 + 2]);

    int ix, iy, iz;
    world_to_voxel(px, py, pz, ix, iy, iz);

    // Every point's own voxel was, by construction, incremented in hits[]
    // the moment its beam was carved (carve_one_beam's "reached && is_hit"
    // branch) — so hits[v] >= 1 always holds here; this IS the invariant
    // main.cu's free_space_consistency gate checks independently.
    float score = -1.0f;
    int label = -1;
    if (voxel_in_bounds(ix, iy, iz)) {
        const int v = voxel_index(ix, iy, iz);
        const unsigned int h = ledger.hits[v];
        const unsigned int p = ledger.pass_from_hit[v] + ledger.pass_from_maxrange[v];
        const unsigned int total = h + p;
        score = (total > 0u) ? (static_cast<float>(p) / static_cast<float>(total)) : 0.0f;
        label = (score >= threshold) ? 1 : 0;
    }
    score_out[i] = score;
    label_out[i] = label;
}

// ===========================================================================
// Host launchers (declared in kernels.cuh).
// ===========================================================================

void launch_ledger_clear(Ledger ledger)
{
    if (!ledger.hits || !ledger.pass_from_hit || !ledger.pass_from_maxrange) {
        std::fprintf(stderr, "launch_ledger_clear: null ledger pointer\n");
        std::exit(EXIT_FAILURE);
    }
    ledger_clear_kernel<<<(kNumVoxels + kThreads - 1) / kThreads, kThreads>>>(ledger);
    CUDA_CHECK_LAST_ERROR("ledger_clear_kernel launch");
}

void launch_carve(int n_beams, const int* d_scan_id, const float* d_dir,
                  const unsigned char* d_is_hit, const float* d_range,
                  Ledger ledger)
{
    if (n_beams < 1 || !d_scan_id || !d_dir || !d_is_hit || !d_range) {
        std::fprintf(stderr, "launch_carve: invalid arguments (n_beams=%d)\n", n_beams);
        std::exit(EXIT_FAILURE);
    }
    carve_kernel<<<(n_beams + kThreads - 1) / kThreads, kThreads>>>(
        n_beams, d_scan_id, d_dir, d_is_hit, d_range, ledger);
    CUDA_CHECK_LAST_ERROR("carve_kernel launch");
}

void launch_carve_trace(int n_beams, const int* d_scan_id, const float* d_dir,
                        const unsigned char* d_is_hit, const float* d_range,
                        int* d_trace_out, int* d_trace_len)
{
    if (n_beams < 1 || !d_scan_id || !d_dir || !d_is_hit || !d_range || !d_trace_out || !d_trace_len) {
        std::fprintf(stderr, "launch_carve_trace: invalid arguments (n_beams=%d)\n", n_beams);
        std::exit(EXIT_FAILURE);
    }
    // Small subset by construction (main.cu documents the size) — one block
    // is typical, but the same ceil-division geometry as every other launch
    // keeps this correct for any subset size.
    carve_trace_kernel<<<(n_beams + kThreads - 1) / kThreads, kThreads>>>(
        n_beams, d_scan_id, d_dir, d_is_hit, d_range, d_trace_out, d_trace_len);
    CUDA_CHECK_LAST_ERROR("carve_trace_kernel launch");
}

void launch_classify(int n_beams, const int* d_scan_id, const float* d_dir,
                     const unsigned char* d_is_hit, const float* d_range,
                     Ledger ledger, float threshold,
                     float* d_score_out, int* d_label_out)
{
    if (n_beams < 1 || !d_scan_id || !d_dir || !d_is_hit || !d_range || !d_score_out || !d_label_out) {
        std::fprintf(stderr, "launch_classify: invalid arguments (n_beams=%d)\n", n_beams);
        std::exit(EXIT_FAILURE);
    }
    classify_kernel<<<(n_beams + kThreads - 1) / kThreads, kThreads>>>(
        n_beams, d_scan_id, d_dir, d_is_hit, d_range, ledger, threshold, d_score_out, d_label_out);
    CUDA_CHECK_LAST_ERROR("classify_kernel launch");
}
