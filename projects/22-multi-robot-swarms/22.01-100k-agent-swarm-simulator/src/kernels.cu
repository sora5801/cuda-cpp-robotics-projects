// ===========================================================================
// kernels.cu — GPU implementation for project 22.01
//              100k-agent swarm simulator: flocking, pheromone grids,
//              stigmergy
//
// The big idea
// ------------
// A swarm is N independent agents whose behavior depends only on their LOCAL
// neighborhood — Reynolds' three boids rules (separation, alignment,
// cohesion) plus, here, a pheromone field agents write and read (stigmergy:
// coordination THROUGH the environment). Locality is the whole game:
//
//   * brute force neighbor search is O(N^2): at N = 100,000 that is 10^10
//     pair tests PER STEP — no machine does that at 20 Hz;
//   * a UNIFORM GRID with cell size == interaction radius makes it
//     O(N * ~30): each agent only inspects the 3x3 cells around it, which
//     provably contain every neighbor within the radius (kernels.cuh).
//
// The per-step GPU pipeline (order matters; main.cu drives it):
//
//   count -> [host exclusive scan] -> scatter -> flock step -> pheromone step
//   (histogram)   (65k ints, honest    (fill bins) (one thread   (one thread
//    atomicAdd     & simple on the                  per AGENT)    per CELL,
//    on ints)      host: THEORY.md)                               stencil)
//
// What is NEW here beyond the four foundations (33.01/09.01/07.09/08.01):
//   * ATOMICS used constructively: an integer histogram (bit-exact — int
//     addition associates) that double-serves as the pheromone DEPOSIT map,
//     and an atomic scatter cursor (whose ordering nondeterminism we
//     document rather than hide — kernels.cuh determinism contract);
//   * a COUNTING SORT built from two tiny kernels + a host scan — the
//     classic GPU spatial-binning pattern (PCL/Warp/Isaac all have one);
//   * a GATHER over variable-length bins — the first kernel in the repo
//     whose per-thread work depends on the DATA (divergence story below);
//   * two coupled layers ping-ponging together: agents (Lagrangian, moving)
//     and a field (Eulerian, fixed grid) — the same agent/field split as
//     production crowd/traffic/swarm simulators.
//
// All constants and layouts come from kernels.cuh — the single source shared
// with the CPU oracle; the per-agent rule math below is a deliberate
// line-by-line twin of reference_cpu.cpp (only the neighbor ITERATION
// differs: bins here, brute force there — that difference is the point of
// the oracle).
//
// Read this after: kernels.cuh.  Companion oracle: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"      // CUDA_CHECK / CUDA_CHECK_LAST_ERROR (§6.1 rule 7)

#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Launch geometry.
//
// Agent kernels (count/scatter/flock): 1-D, 256-thread blocks — the repo
// default; one thread per agent, ragged tail guarded.
// Pheromone kernel: 2-D, 16x16 = 256-thread blocks — 07.09's grid/stencil
// geometry: the data is 2-D and the neighbors are 2-D offsets, so 2-D blocks
// make the index math read like grid coordinates, and consecutive
// threadIdx.x along a row keeps row-major accesses coalesced (x must be the
// fast axis — swapping x/y is THE classic grid-kernel mistake).
// ---------------------------------------------------------------------------
static constexpr int kThreads = 256;   // 1-D block size for agent kernels
static constexpr int kTile    = 16;    // 2-D tile edge for the stencil kernel

// ---------------------------------------------------------------------------
// cell_coord — map a position coordinate (m) to a grid index, clamped.
//
// This tiny function is textually TWINNED with the CPU oracle
// (reference_cpu.cpp re-states it identically) — position -> cell must be
// bit-identical on both paths or the neighbor structures would differ by
// construction. One multiply then floorf: both are single correctly-rounded
// IEEE operations, so identical inputs give identical cells on any
// conforming compiler (THEORY.md §numerics).
// The clamp handles only the boundary case px == kArena (floor gives
// kGridDim, one past the last cell) — positions are already kept inside
// [0, kArena] by the integrator's clamp.
// ---------------------------------------------------------------------------
__device__ __forceinline__ int cell_coord(float p)
{
    int c = static_cast<int>(floorf(p * kInvCell));
    if (c < 0) c = 0;
    if (c > kGridDim - 1) c = kGridDim - 1;
    return c;
}

// ===========================================================================
// Kernel 1: histogram — count agents per cell.
//
// Thread-to-data mapping: thread i = blockIdx.x*blockDim.x + threadIdx.x
// owns agent i and increments its cell's counter. Grid: ceil(n/256) x 256.
//
// Memory spaces: px/py coalesced reads (SoA pays off); counts[] takes
// ATOMIC adds — many threads may hit the same cell. Two things to teach:
//   * correctness: without the atomic, two threads read-modify-writing the
//     same counter lose increments (the textbook data race);
//   * determinism: INTEGER addition is associative, so the finished
//     histogram is bit-exact regardless of thread order — which is exactly
//     why this histogram can double as the pheromone DEPOSIT map (every
//     agent deposits the same kDeposit, so deposit = kDeposit * count,
//     computed exactly; kernels.cuh determinism contract).
// Contention is mild: ~1.5 agents/cell on average, so most atomics are
// uncontended; sm_75+ resolve them in L2 without serializing the kernel.
// ===========================================================================
__global__ void bin_count_kernel(const float* __restrict__ px,     // [n] agent x (m)
                                 const float* __restrict__ py,     // [n] agent y (m)
                                 int*         __restrict__ counts, // [kNumCells] OUT (pre-zeroed by main.cu)
                                 int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's agent
    if (i >= n) return;                                    // ragged-tail guard
    const int c = cell_coord(py[i]) * kGridDim + cell_coord(px[i]);  // row-major cell id
    atomicAdd(&counts[c], 1);   // int atomic: race-free AND order-independent
}

// ===========================================================================
// Kernel 2: scatter — write each agent's index into its cell's bin.
//
// Thread i claims a slot in cell c by atomically bumping cursor[c] (the
// host pre-loads cursor with the exclusive-scan starts, so cell c's slots
// are starts[c] .. starts[c+1]-1). The RESULT is a valid counting sort; the
// ORDER of agents within one cell depends on which thread's atomic lands
// first — scheduling-dependent, hence the documented run-to-run ulp
// nondeterminism in downstream float sums (kernels.cuh). A stable sort
// would cost a second ranking pass or a full radix sort; for a teaching
// simulator the honest cursor + a SMOOTH force kernel (see
// accumulate_neighbor) is the better trade — THEORY.md §numerics weighs
// the alternatives.
// ===========================================================================
__global__ void bin_scatter_kernel(const float* __restrict__ px,         // [n] agent x (m)
                                   const float* __restrict__ py,         // [n] agent y (m)
                                   int*         __restrict__ cursor,     // [kNumCells] next free slot per cell
                                   int*         __restrict__ bin_agents, // [n] OUT: agent ids grouped by cell
                                   int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int c = cell_coord(py[i]) * kGridDim + cell_coord(px[i]);
    const int slot = atomicAdd(&cursor[c], 1);   // claim a unique slot in my cell's range
    bin_agents[slot] = i;
}

// ===========================================================================
// The per-agent rule math — textual twin of reference_cpu.cpp.
//
// NeighborAccum collects the running sums one agent gathers over its
// neighbors; accumulate_neighbor() folds in one candidate; finish_agent()
// turns the sums into an acceleration, integrates, and clamps. The GPU
// kernel and the CPU oracle differ ONLY in who calls accumulate_neighbor
// with which j and in what order — the math is identical, which is what
// makes the lockstep comparison meaningful (kernels.cuh: swarm_step_cpu).
// ===========================================================================
struct NeighborAccum {
    float w_sum;        // sum of hat weights w = 1 - d/r (unitless)
    float avx, avy;     // weighted sum of neighbor velocities (m/s * weight)
    float cenx, ceny;   // weighted sum of neighbor OFFSETS dx,dy (m * weight)
                        // (offsets, not absolute positions: the weighted mean
                        //  offset IS the vector to the local centroid, and
                        //  small differences of small numbers beat the
                        //  huge-minus-huge cancellation absolute positions
                        //  would suffer — a free numerics win)
    float sepx, sepy;   // separation push, unit-direction-weighted (unitless)
    int   nbr;          // plain neighbor count (for the [info] metric)
};

// ---------------------------------------------------------------------------
// accumulate_neighbor — fold candidate j into agent i's running sums.
//
// Params: agent i's position (pxi,pyi), candidate j's position/velocity,
// and the accumulator 'a' (updated in place). All quantities SI (m, m/s),
// arena frame (kernels.cuh layout contract).
//
// The HAT WEIGHT w = 1 - d/r is load-bearing for verification: every
// contribution goes SMOOTHLY to zero at the interaction radius, so when an
// ulp of rounding difference flips whether a borderline pair "is" a
// neighbor, the force changes by ~0 instead of jumping O(1) — without this,
// the GPU-vs-CPU lockstep gate would flake on radius-boundary coincidences
// (THEORY.md §numerics tells the full story; plain unweighted means, the
// textbook boids formulation, are README Exercise 2).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void accumulate_neighbor(
    float pxi, float pyi,
    float pxj, float pyj, float vxj, float vyj,
    NeighborAccum& a)
{
    const float dx = pxj - pxi;            // offset i -> j (m)
    const float dy = pyj - pyi;
    const float d2 = dx * dx + dy * dy;    // squared distance (m^2) — compare
                                           // squared vs squared: sqrt only for
                                           // actual neighbors (most candidates
                                           // in the 3x3 block are NOT within r)
    if (d2 >= kRNeighbor * kRNeighbor) return;

    const float d = sqrtf(d2);             // distance (m); sqrtf is IEEE
                                           // correctly-rounded on both paths
    const float w = 1.0f - d * kInvRNb;    // hat weight: 1 at d=0, 0 at d=r
    a.w_sum += w;
    a.avx  += w * vxj;   a.avy  += w * vyj;   // alignment: weighted velocity sum
    a.cenx += w * dx;    a.ceny += w * dy;    // cohesion: weighted offset sum
    a.nbr  += 1;

    if (d < kRSep) {
        // Separation: push AWAY from j, strength ramping from full at
        // contact to 0 at kRSep (smooth again). Dividing by d turns (dx,dy)
        // into a unit direction; the fmaxf floor guards the (measure-zero,
        // but fatal) exactly-coincident case against a 0/0 NaN.
        const float s = (1.0f - d * kInvRSep) / fmaxf(d, 1e-6f);
        a.sepx -= s * dx;
        a.sepy -= s * dy;
    }
}

// ---------------------------------------------------------------------------
// finish_agent — rules -> acceleration -> clamped integration, for agent i.
//
// Inputs: agent i's state (m, m/s), its finished NeighborAccum, and the
// pheromone gradient (gx,gy) in concentration/m sampled by the caller.
// Outputs (via pointers): the post-step state, and the alignment score
// (cosine between v_i and the local mean velocity, in [-1,1]; the
// kNoNeighborScore sentinel when undefined) — the demo's cohesion metric.
//
// Design notes, in execution order:
//   * WALL: a soft linear ramp inside kWallMargin — continuous everywhere,
//     unlike reflection, whose velocity sign-flip would make the lockstep
//     gate flaky for wall-grazing agents (THEORY.md §numerics). The
//     integrator's final position clamp is the belt-and-suspenders bound.
//   * RULES: alignment relaxes v toward the local weighted mean (gain 1/s),
//     cohesion is a spring toward the local weighted centroid (gain 1/s^2),
//     separation and the pheromone pull enter as direct accelerations.
//   * CLAMPS: acceleration then speed, both by SCALING the vector (which is
//     continuous and preserves direction), never by per-axis clipping
//     (which distorts direction). Actuator realism: rules PROPOSE, the
//     clamp DISPOSES — the same lesson as 08.01's force limit.
//   * INTEGRATION: semi-implicit Euler (v first, then x with the NEW v) —
//     the standard, stable choice for velocity-clamped agent models
//     (THEORY.md §numerics; nothing here is stiff, so no RK4 needed).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void finish_agent(
    float pxi, float pyi, float vxi, float vyi,
    const NeighborAccum& a,
    float gx, float gy,
    float* px_o, float* py_o, float* vx_o, float* vy_o,
    float* score_o)
{
    // --- wall force: linear ramp from kWWall at the wall to 0 at the margin
    float ax = 0.0f, ay = 0.0f;                       // acceleration (m/s^2)
    if (pxi < kWallMargin)          ax += kWWall * (1.0f - pxi / kWallMargin);
    if (pxi > kArena - kWallMargin) ax -= kWWall * (1.0f - (kArena - pxi) / kWallMargin);
    if (pyi < kWallMargin)          ay += kWWall * (1.0f - pyi / kWallMargin);
    if (pyi > kArena - kWallMargin) ay -= kWWall * (1.0f - (kArena - pyi) / kWallMargin);

    // --- boids rules (only defined when there was at least one neighbor)
    float score = kNoNeighborScore;                   // sentinel: no neighbors
    if (a.w_sum > 0.0f) {
        const float inv = 1.0f / a.w_sum;
        const float mvx = a.avx * inv;                // local mean velocity (m/s)
        const float mvy = a.avy * inv;
        ax += kWAli * (mvx - vxi) + kWCoh * (a.cenx * inv);
        ay += kWAli * (mvy - vyi) + kWCoh * (a.ceny * inv);

        // Alignment score for the cohesion metric: cos(angle between my
        // velocity and the local mean). Guards: either vector can be tiny
        // (a stalled agent, a perfectly-cancelling neighborhood) — then the
        // cosine is undefined and we keep the sentinel instead of a NaN.
        const float ni = sqrtf(vxi * vxi + vyi * vyi);
        const float nm = sqrtf(mvx * mvx + mvy * mvy);
        if (ni > 1e-6f && nm > 1e-6f)
            score = (vxi * mvx + vyi * mvy) / (ni * nm);
    }
    ax += kWSep * a.sepx;                             // separation (already directional)
    ay += kWSep * a.sepy;

    // --- stigmergy: weak pull UP the pheromone gradient (trail-following)
    ax += kWPher * gx;
    ay += kWPher * gy;

    // --- clamp acceleration (scale, don't clip)
    const float aa = ax * ax + ay * ay;
    if (aa > kAMax * kAMax) {
        const float s = kAMax / sqrtf(aa);
        ax *= s;  ay *= s;
    }

    // --- semi-implicit Euler: v first ...
    float vxn = vxi + ax * kDt;
    float vyn = vyi + ay * kDt;

    // --- clamp speed into [kVMin, kVMax] (scale, don't clip)
    const float vv = vxn * vxn + vyn * vyn;
    if (vv > kVMax * kVMax) {
        const float s = kVMax / sqrtf(vv);
        vxn *= s;  vyn *= s;
    } else if (vv < kVMin * kVMin) {
        const float sp = sqrtf(vv);
        if (sp > 1e-6f) {
            const float s = kVMin / sp;               // speed floor: boids don't stall
            vxn *= s;  vyn *= s;
        } else {
            vxn = kVMin;  vyn = 0.0f;                 // fully stalled (measure-zero
        }                                             // after init): deterministic kick
    }

    // --- ... then x with the NEW v; clamp into the arena as the hard bound.
    // The wall force does the real work — this clamp is the guarantee the
    // demo's BOUNDED check rests on, and clamping is CONTINUOUS, so it
    // cannot destabilize the lockstep comparison the way reflection would.
    float pxn = pxi + vxn * kDt;
    float pyn = pyi + vyn * kDt;
    pxn = fminf(fmaxf(pxn, 0.0f), kArena);
    pyn = fminf(fmaxf(pyn, 0.0f), kArena);

    *px_o = pxn;  *py_o = pyn;
    *vx_o = vxn;  *vy_o = vyn;
    *score_o = score;
}

// ===========================================================================
// Kernel 3: the flock step — one thread = one agent's whole decision.
//
// Thread-to-data mapping: thread i owns agent i: gather neighbors from the
// 3x3 cells around agent i's cell, fold them through accumulate_neighbor,
// sample the pheromone gradient, then finish_agent. Grid: ceil(n/256) x 256.
//
// Memory spaces per thread:
//   registers : the agent's state + the NeighborAccum sums (~40 regs)
//   global    : px/py/vx/vy[i]   — coalesced reads (SoA, consecutive i)
//               starts[c]        — 9+ reads into a 65k-entry array; a warp's
//                                  agents are scattered across the arena
//                                  (agent order never changes), so these
//                                  lean on L2
//               bin_agents[s], then px/py/vx/vy[j] — GATHER reads: j values
//                                  are scattered, so this is the kernel's
//                                  uncoalesced, L2-dependent part. Intrinsic
//                                  to neighbor search; production sims
//                                  REORDER agent state into bin order every
//                                  step so neighbors are contiguous in
//                                  memory (README Exercise 5 — measure it).
//               pher[...]        — 4 reads for the central-difference
//                                  gradient, cached like starts
//               *_nxt[i], align_score[i] — coalesced writes
//
// Divergence, honestly: bin sizes differ, so lanes in a warp loop different
// trip counts and the warp runs as long as its fullest bin (idle lanes wait).
// With ~1.5 agents/cell average density this costs tens of percent, not
// integer factors — the measured per-step numbers are in THEORY.md
// §GPU-mapping. No shared memory: bins are variable-length and a block's
// agents span many cells; the shared-memory tile variant is README
// Exercise 5's second half.
// ===========================================================================
__global__ void flock_step_kernel(const float* __restrict__ px,          // [n] x (m) at step start
                                  const float* __restrict__ py,          // [n] y (m)
                                  const float* __restrict__ vx,          // [n] vx (m/s)
                                  const float* __restrict__ vy,          // [n] vy (m/s)
                                  const int*   __restrict__ starts,      // [kNumCells+1] bin ranges
                                  const int*   __restrict__ bin_agents,  // [n] agent ids by cell
                                  const float* __restrict__ pher,        // [kNumCells] field at step start
                                  float*       __restrict__ px_o,        // [n] OUT x (m)
                                  float*       __restrict__ py_o,        // [n] OUT y (m)
                                  float*       __restrict__ vx_o,        // [n] OUT vx (m/s)
                                  float*       __restrict__ vy_o,        // [n] OUT vy (m/s)
                                  float*       __restrict__ align_score, // [n] OUT metric (kernels.cuh)
                                  int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // My state -> registers for the whole step (read once, use many times).
    const float pxi = px[i], pyi = py[i];
    const float vxi = vx[i], vyi = vy[i];
    const int cx = cell_coord(pxi);
    const int cy = cell_coord(pyi);

    // Gather neighbors from the 3x3 cell block (clamped at the arena edge —
    // there is nothing outside the walls, so edge agents just scan fewer
    // cells; the CPU oracle's brute force finds the IDENTICAL neighbor set
    // because kCellSize == kRNeighbor guarantees 3x3 coverage).
    NeighborAccum acc = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0 };
    const int x0 = (cx > 0) ? cx - 1 : 0, x1 = (cx < kGridDim - 1) ? cx + 1 : kGridDim - 1;
    const int y0 = (cy > 0) ? cy - 1 : 0, y1 = (cy < kGridDim - 1) ? cy + 1 : kGridDim - 1;
    for (int ky = y0; ky <= y1; ++ky) {
        for (int kx = x0; kx <= x1; ++kx) {
            const int c = ky * kGridDim + kx;
            const int s_end = starts[c + 1];              // bin range [starts[c], s_end)
            for (int s = starts[c]; s < s_end; ++s) {
                const int j = bin_agents[s];
                if (j == i) continue;                     // an agent is not its own neighbor
                accumulate_neighbor(pxi, pyi, px[j], py[j], vx[j], vy[j], acc);
            }
        }
    }

    // Pheromone gradient by central difference of the cell-sampled field,
    // indices clamped at the edges (which shortens the effective baseline
    // there — a boundary approximation both paths make identically;
    // THEORY.md §numerics).
    const int cxe = (cx < kGridDim - 1) ? cx + 1 : kGridDim - 1;
    const int cxw = (cx > 0) ? cx - 1 : 0;
    const int cyn = (cy < kGridDim - 1) ? cy + 1 : kGridDim - 1;
    const int cys = (cy > 0) ? cy - 1 : 0;
    const float gx = (pher[cy * kGridDim + cxe] - pher[cy * kGridDim + cxw]) * (0.5f * kInvCell);
    const float gy = (pher[cyn * kGridDim + cx] - pher[cys * kGridDim + cx]) * (0.5f * kInvCell);

    // Rules -> acceleration -> clamped integration (shared math, see above).
    finish_agent(pxi, pyi, vxi, vyi, acc, gx, gy,
                 &px_o[i], &py_o[i], &vx_o[i], &vy_o[i], &align_score[i]);
}

// ===========================================================================
// Kernel 4: the pheromone field step — deposit + diffuse + decay.
//
// Thread-to-data mapping: thread (cx, cy) = (blockIdx*blockDim + threadIdx)
// owns cell (cx, cy); 16x16 tiles; guards handle the exact 256x256 fit.
//
// The update (per cell c, one pass — derivation in THEORY.md §math):
//
//     lap    = pN + pS + pE + pW - 4*p              (5-point Laplacian)
//     out[c] = (1-kDecay) * (p + kDiffuse*lap)      (diffuse, then evaporate)
//              + kDeposit * counts[c]               (this step's deposits)
//
// Deposits come from the INT histogram, not float atomics — bit-exact and
// order-independent because every agent deposits the same amount (the
// design decision documented in kernels.cuh; variable per-agent deposits
// are README Exercise 4). Boundary is zero-flux: an out-of-range neighbor
// contributes the center value (equivalently: no gradient across the wall),
// so pheromone cannot leak out of the arena — matching the walls the agents
// themselves feel.
//
// Memory: the classic 5-point stencil — center coalesced; N/S one full row
// apart (coalesced too, different cache lines); E/W overlap the warp's own
// loads and come from L1/L2. One coalesced write. READ pher_in, WRITE
// pher_out: the same ping-pong discipline as 07.09 — an in-place stencil
// would mix this step's and last step's values depending on scheduling
// (a real data race, not just nondeterminism).
// ===========================================================================
__global__ void pheromone_step_kernel(const float* __restrict__ pher_in,  // [kNumCells] field at step start
                                      const int*   __restrict__ counts,   // [kNumCells] deposit map (agents/cell)
                                      float*       __restrict__ pher_out) // [kNumCells] OUT field after step
{
    const int cx = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's cell column
    const int cy = blockIdx.y * blockDim.y + threadIdx.y;   // this thread's cell row
    if (cx >= kGridDim || cy >= kGridDim) return;
    const int c = cy * kGridDim + cx;

    const float p  = pher_in[c];
    // Zero-flux boundary: missing neighbors mirror the center value, which
    // makes their (neighbor - center) contribution to the Laplacian zero.
    const float pn = (cy < kGridDim - 1) ? pher_in[c + kGridDim] : p;
    const float ps = (cy > 0)            ? pher_in[c - kGridDim] : p;
    const float pe = (cx < kGridDim - 1) ? pher_in[c + 1]        : p;
    const float pw = (cx > 0)            ? pher_in[c - 1]        : p;

    const float lap = pn + ps + pe + pw - 4.0f * p;
    pher_out[c] = (1.0f - kDecay) * (p + kDiffuse * lap)
                + kDeposit * static_cast<float>(counts[c]);
}

// ===========================================================================
// Host launchers (declared in kernels.cuh). Each owns its grid math and the
// post-launch error check; main.cu owns buffers, ordering, and the host scan.
// ===========================================================================

// Shared argument sanity check — a null device pointer here means main.cu's
// allocation logic broke; fail loudly at the launch site, not inside a kernel.
static void require_valid(const SwarmGpu& g, const char* who)
{
    if (g.n < 1 || !g.px_cur || !g.py_cur || !g.vx_cur || !g.vy_cur ||
        !g.px_nxt || !g.py_nxt || !g.vx_nxt || !g.vy_nxt ||
        !g.pher_cur || !g.pher_nxt || !g.counts || !g.starts ||
        !g.cursor || !g.bin_agents || !g.align_score) {
        std::fprintf(stderr, "%s: invalid SwarmGpu (n=%d or null device pointer)\n", who, g.n);
        std::exit(EXIT_FAILURE);
    }
}

void launch_bin_count(const SwarmGpu& g)
{
    require_valid(g, "launch_bin_count");
    const int blocks = (g.n + kThreads - 1) / kThreads;
    bin_count_kernel<<<blocks, kThreads>>>(g.px_cur, g.py_cur, g.counts, g.n);
    CUDA_CHECK_LAST_ERROR("bin_count_kernel launch");
}

void launch_bin_scatter(const SwarmGpu& g)
{
    require_valid(g, "launch_bin_scatter");
    const int blocks = (g.n + kThreads - 1) / kThreads;
    bin_scatter_kernel<<<blocks, kThreads>>>(g.px_cur, g.py_cur, g.cursor, g.bin_agents, g.n);
    CUDA_CHECK_LAST_ERROR("bin_scatter_kernel launch");
}

void launch_flock_step(const SwarmGpu& g)
{
    require_valid(g, "launch_flock_step");
    const int blocks = (g.n + kThreads - 1) / kThreads;
    flock_step_kernel<<<blocks, kThreads>>>(g.px_cur, g.py_cur, g.vx_cur, g.vy_cur,
                                            g.starts, g.bin_agents, g.pher_cur,
                                            g.px_nxt, g.py_nxt, g.vx_nxt, g.vy_nxt,
                                            g.align_score, g.n);
    CUDA_CHECK_LAST_ERROR("flock_step_kernel launch");
}

void launch_pheromone_step(const SwarmGpu& g)
{
    require_valid(g, "launch_pheromone_step");
    // 2-D launch over the 256x256 grid: 16x16 tiles -> a 16x16 grid of
    // blocks (an exact fit here, but the kernel guards anyway so the
    // geometry can change without a silent out-of-bounds).
    const dim3 block(kTile, kTile);
    const dim3 grid((kGridDim + kTile - 1) / kTile, (kGridDim + kTile - 1) / kTile);
    pheromone_step_kernel<<<grid, block>>>(g.pher_cur, g.counts, g.pher_nxt);
    CUDA_CHECK_LAST_ERROR("pheromone_step_kernel launch");
}
