// ===========================================================================
// kernels.cuh — interface for project 22.01
//               100k-agent swarm simulator: flocking, pheromone grids,
//               stigmergy
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (the simulation driver), kernels.cu (the GPU
// step), and reference_cpu.cpp (the brute-force oracle). Everything all
// three must agree on — the agent state layout, the arena and grid geometry,
// the boids rule math, the pheromone stencil, and the per-step pipeline
// order — is defined HERE, once (CLAUDE.md §12).
//
// The simulation in five lines (THEORY.md derives and justifies each):
//   1. BIN: count agents per grid cell (atomicAdd on ints), host exclusive
//      scan of the 256x256 histogram, scatter agent indices into bins.
//   2. FLOCK: per agent, gather neighbors from the 3x3 cells around it and
//      apply Reynolds' boids rules (separation / alignment / cohesion),
//      plus a soft wall force and a weak pull up the pheromone gradient.
//   3. INTEGRATE: clamp acceleration and speed, semi-implicit Euler.
//   4. STIGMERGY: every agent deposits pheromone into its cell (the cell
//      HISTOGRAM from step 1 *is* the deposit map — see the determinism
//      note below); a diffusion+decay stencil spreads and evaporates it.
//   5. Swap the ping-pong buffers, repeat.
// Steps 2 and 4 are the arithmetic; both are embarrassingly parallel — one
// thread per AGENT (2-3) and one thread per CELL (4). Step 1 is what makes
// step 2 affordable: brute-force neighbor search is O(N^2) = 10^10 pair
// tests at N = 100,000 — impossible at interactive rates on anything —
// while grid binning makes it O(N * ~30) (THEORY.md §algorithm).
//
// AGENT STATE LAYOUT — structure-of-arrays (SoA), SI units, defined once:
//     px[i], py[i]   position (m), arena frame: origin at the SW corner,
//                    x right, y up (right-handed with z out of the plane);
//                    always inside [0, kArena] (integration clamps)
//     vx[i], vy[i]   velocity (m/s), same frame; speed kept in
//                    [kVMin, kVMax] by the integrator
// SoA and not an Agent struct because every kernel reads the same field for
// consecutive i — SoA makes a warp's 32 reads CONSECUTIVE floats (coalesced);
// an array-of-structs would stride them 16 bytes apart (the 33.01 lesson).
//
// GRID LAYOUT — one uniform grid serves BOTH the neighbor search and the
// pheromone field (deliberate: one geometry to learn, one histogram reused):
//     kGridDim x kGridDim cells of kCellSize m, row-major index
//     cell(cx, cy) = cy * kGridDim + cx, cx = floor(px / kCellSize)
//     (clamped to [0, kGridDim-1]; the clamp only matters for px == kArena).
//     kCellSize == kRNeighbor, so the 3x3 cell block around an agent is
//     GUARANTEED to contain every neighbor within kRNeighbor — the property
//     that makes the grid gather EXACTLY equal to brute force (THEORY.md).
//
// BIN LAYOUT (counting sort, rebuilt every step):
//     counts[c]      int, number of agents in cell c (also the deposit map)
//     starts[c]      exclusive prefix sum of counts; starts[kNumCells] = N
//     bin_agents[s]  agent index at slot s; cell c's agents occupy slots
//                    starts[c] .. starts[c+1]-1, in ARBITRARY order (the
//                    scatter uses an atomic cursor; see determinism below)
//
// DETERMINISM CONTRACT (the honest part — THEORY.md §numerics):
//   * Integer atomics are ASSOCIATIVE — the cell histogram is bit-exact no
//     matter the thread order. Because every agent deposits the SAME amount
//     of pheromone per step, deposit[c] = kDeposit * counts[c] is therefore
//     also bit-exact. This is a deliberate design choice: per-agent variable
//     deposits would need float atomicAdd, whose result depends on thread
//     order (float addition does not associate). README Exercise 4.
//   * Float SUMS over neighbors are NOT order-independent, and the scatter
//     puts agents into bins in scheduling-dependent order — so neighbor
//     sums can differ in their last bits from run to run. All rule kernels
//     are SMOOTH at the interaction radius (hat-weighted, contribution -> 0
//     at r), so an ulp never flips into an O(1) force jump; trajectories
//     still diverge chaotically at ulp scale, which is why the demo's
//     verdict uses flock STATISTICS with wide margins and the GPU-vs-CPU
//     gate compares in LOCKSTEP, one step at a time (see swarm_step_cpu).
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

// ---------------------------------------------------------------------------
// Arena & grid geometry — shared verbatim by GPU kernels and CPU oracle.
// ---------------------------------------------------------------------------
constexpr int   kGridDim   = 256;              // cells per side (256x256 = 65,536 cells)
constexpr int   kNumCells  = kGridDim * kGridDim;
constexpr float kCellSize  = 1.0f;             // cell edge (m) — MUST equal kRNeighbor
constexpr float kArena     = kGridDim * kCellSize;  // arena edge (m): 256 m square
constexpr float kInvCell   = 1.0f / kCellSize; // precomputed reciprocal (both paths
                                               // use the SAME multiply-then-floor so
                                               // cell indices are bit-identical)

// ---------------------------------------------------------------------------
// Boids parameters (Reynolds 1987 rules, hat-kernel weighted — THEORY.md
// §algorithm tells the tuning story; these are the taught, tuned values).
// ---------------------------------------------------------------------------
constexpr float kRNeighbor = 1.0f;   // alignment/cohesion radius (m) == kCellSize
constexpr float kRSep      = 0.5f;   // separation radius (m) — must be <= kRNeighbor
constexpr float kInvRNb    = 1.0f / kRNeighbor;
constexpr float kInvRSep   = 1.0f / kRSep;

constexpr float kWSep  = 6.0f;   // separation gain (m/s^2 at full overlap): personal space
constexpr float kWAli  = 3.0f;   // alignment gain (1/s): relax v toward the local mean
constexpr float kWCoh  = 1.0f;   // cohesion gain (1/s^2): spring toward the local centroid
constexpr float kWWall = 8.0f;   // wall repulsion gain (m/s^2 at the wall itself)
constexpr float kWallMargin = 8.0f;  // wall force ramps linearly to 0 over this distance (m)
constexpr float kWPher = 0.4f;   // pheromone-gradient gain ((m/s^2)/(pher/m)) — WEAK by
                                 // design: stigmergy biases the flock, it must not
                                 // overpower flocking (THEORY.md §algorithm)

constexpr float kDt    = 0.05f;  // simulation step (s) -> 20 Hz, a realistic swarm
                                 // coordination rate (SYSTEM_DESIGN item 1)
constexpr float kVMax  = 2.0f;   // speed ceiling (m/s) — 0.1 m/step = 1/10 cell, so an
                                 // agent can never tunnel through a cell in one step
constexpr float kVMin  = 0.3f;   // speed floor (m/s): boids never stall (keeps the
                                 // polarization metric well-defined; guard in code)
constexpr float kAMax  = 8.0f;   // acceleration ceiling (m/s^2) — actuator realism:
                                 // rules PROPOSE, the clamp DISPOSES (08.01's lesson)

// ---------------------------------------------------------------------------
// Pheromone field parameters (diffusion + decay stencil, 07.09's grid
// pattern). Field value is unitless "concentration"; equilibrium mean is
// kDeposit * (N / kNumCells) / kDecay ~ 3.8 at N = 100k (THEORY.md §math).
// ---------------------------------------------------------------------------
constexpr float kDeposit = 0.05f;  // concentration added per agent per step
constexpr float kDiffuse = 0.15f;  // diffusion coefficient kappa (per step);
                                   // explicit 5-point stencil is stable for
                                   // kappa <= 0.25 (THEORY.md §numerics)
constexpr float kDecay   = 0.02f;  // evaporation fraction per step

// ---------------------------------------------------------------------------
// Problem sizes & verification configuration.
// ---------------------------------------------------------------------------
constexpr int kDefaultN     = 100000; // the headline swarm (scenario file may override)
constexpr int kDefaultSteps = 300;    // 300 steps @ 20 Hz = 15 s of swarm time
constexpr int kVerifyN      = 4096;   // small deterministic config for the CPU gate:
                                      // brute force is O(N^2), so the oracle must be
                                      // small enough to run in seconds
constexpr int kVerifySteps  = 100;    // lockstep-verified steps (see swarm_step_cpu)

// Lockstep tolerances (justification: THEORY.md §verification — observed
// deviations are ulp-scale ~1e-5; these carry ~100x headroom while real
// bugs, e.g. a wrong bin offset, blow past them on step 1):
constexpr float kTolPos  = 1e-3f;  // max |gpu-cpu| position error per step (m)
constexpr float kTolVel  = 1e-3f;  // max |gpu-cpu| velocity error per step (m/s)
constexpr float kTolPher = 1e-3f;  // max |gpu-cpu| pheromone error per step (abs)

// ---------------------------------------------------------------------------
// SwarmGpu — the device-side state owned by main.cu (allocated once, reused
// every step; a 20 Hz loop that reallocates spends its budget on the
// allocator — 08.01's lesson). All pointers are DEVICE pointers.
//
// Ping-pong discipline: kernels read pos/vel/pher "cur" and write "nxt";
// main.cu swaps the pointers after each step. Writing in place would let
// half-updated neighbors leak into this step's gathers (07.09's rule).
// ---------------------------------------------------------------------------
struct SwarmGpu {
    // agent state, SoA (layout contract above), [n] each
    float *px_cur, *py_cur, *vx_cur, *vy_cur;   // read by the step
    float *px_nxt, *py_nxt, *vx_nxt, *vy_nxt;   // written by the step
    // pheromone field, [kNumCells], row-major grid layout
    float *pher_cur, *pher_nxt;
    // uniform-grid neighbor structure, rebuilt every step
    int *counts;      // [kNumCells] histogram (also the deposit map)
    int *starts;      // [kNumCells+1] exclusive scan (uploaded by the host)
    int *cursor;      // [kNumCells] scatter cursors (device copy of starts)
    int *bin_agents;  // [n] agent indices grouped by cell
    // per-agent metric output (written by the flock kernel each step):
    // local alignment score in [-1,1], or kNoNeighborScore when the agent
    // had no neighbors this step (host skips those when averaging)
    float *align_score;
    int n;            // agent count this SwarmGpu was allocated for
};

// Sentinel for "agent had no neighbors" in align_score (outside [-1,1], so
// the host can filter it out of the cohesion metric unambiguously).
constexpr float kNoNeighborScore = 2.0f;

// ---------------------------------------------------------------------------
// launch_bin_count — kernel 1 of the binning pipeline: zero counts, then
// histogram agents into cells with atomicAdd(int).
//   g : device state; reads g.px_cur/g.py_cur, writes g.counts.
// Launch: one thread per agent, 256-thread blocks (repo default), after a
// cudaMemset of counts. Integer atomics => bit-exact histogram (header note).
// ---------------------------------------------------------------------------
void launch_bin_count(const SwarmGpu& g);

// ---------------------------------------------------------------------------
// launch_bin_scatter — kernel 2: place agent indices into their cell's bin.
//   g : device state; reads px/py + g.cursor (pre-loaded with starts),
//       writes g.bin_agents. main.cu copies starts -> cursor (D2D) first.
// Launch: one thread per agent. Slot order within a cell is scheduling-
// dependent (atomic cursor) — the documented ulp-level nondeterminism.
// ---------------------------------------------------------------------------
void launch_bin_scatter(const SwarmGpu& g);

// ---------------------------------------------------------------------------
// launch_flock_step — the heart: boids rules + wall + pheromone gradient +
// clamped semi-implicit Euler, one thread per agent.
//   g : device state; reads *_cur, starts, bin_agents, pher_cur;
//       writes *_nxt and align_score.
// Launch config, memory spaces, and the full rule math are documented at
// the definition (kernels.cu) — the header carries the summary, the
// definition carries the essay, so there is one place to keep deeply true.
// ---------------------------------------------------------------------------
void launch_flock_step(const SwarmGpu& g);

// ---------------------------------------------------------------------------
// launch_pheromone_step — deposit + diffuse + decay, one thread per CELL
// (07.09's 2-D grid/stencil pattern; 16x16 blocks).
//   g : device state; reads pher_cur and counts (the deposit map),
//       writes pher_nxt. Zero-flux boundary (edge cells reuse the center
//       value for out-of-range neighbors — pheromone cannot leave the arena).
// ---------------------------------------------------------------------------
void launch_pheromone_step(const SwarmGpu& g);

// ---------------------------------------------------------------------------
// CPU reference (reference_cpu.cpp).
//
// swarm_step_cpu — ONE full simulation step on the host, BRUTE FORCE:
// for every agent, test every other agent against the interaction radii
// (O(N^2)) — deliberately a DIFFERENT algorithm from the GPU's grid gather,
// like 07.09's exact oracle: if the binning/scatter/gather machinery has any
// bug (wrong cell, missed bin, off-by-one in starts), the neighbor SETS
// differ and the comparison fails loudly on step 1. Same rule math, same
// constants, same clamps; pheromone stencil is a line-by-line twin.
//
//   n                    : agent count
//   px,py,vx,vy          : [n] HOST arrays, state at step start (read)
//   pher                 : [kNumCells] HOST pheromone field at step start (read)
//   px_o,py_o,vx_o,vy_o  : [n] HOST arrays OUT: state after the step
//   pher_o               : [kNumCells] HOST OUT: field after the step
//
// main.cu runs it in LOCKSTEP with the GPU during the verify stage: both
// paths step from the SAME state, outputs are compared within kTol*, then
// the GPU output becomes the next shared state. Lockstep matters: flocking
// is chaotic, so free-running comparison would amplify benign ulp
// differences into meters within ~100 steps (THEORY.md §verification).
// ---------------------------------------------------------------------------
void swarm_step_cpu(int n,
                    const float* px, const float* py,
                    const float* vx, const float* vy,
                    const float* pher,
                    float* px_o, float* py_o,
                    float* vx_o, float* vy_o,
                    float* pher_o);

#endif // PROJECT_KERNELS_CUH
