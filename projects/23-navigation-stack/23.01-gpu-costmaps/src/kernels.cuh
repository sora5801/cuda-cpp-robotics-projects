// ===========================================================================
// kernels.cuh — interface for project 23.01
//               GPU costmaps: inflation, raytrace clearing, multi-layer fusion
//               + a DWA local-planner consumer, closed loop
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (the closed-loop driver), kernels.cu (the GPU
// costmap + DWA kernels), and reference_cpu.cpp (the oracle twins + the
// diff-drive PLANT). Everything all three must agree on — grid geometry,
// cost semantics, the LiDAR scan layout, the (v,w) sampling window layout,
// and every tuning constant — is defined HERE, once (CLAUDE.md §12).
//
// The pipeline in six lines (THEORY.md derives every piece properly):
//   1. Simulate a 360-beam LiDAR scan against the TRUE world (host, §Scan).
//   2. GPU: raytrace the scan into the OBSTACLE layer (mark hits, clear the
//      free cells each beam passed through) — one thread per BEAM.
//   3. GPU: INFLATE the union of static+obstacle lethal cells outward —
//      one thread per CELL, bounded-radius gather.
//   4. GPU: FUSE static + obstacle + inflation into the MASTER costmap by
//      per-cell max — one thread per CELL (the map pattern, again).
//   5. GPU: score a whole (v,w) sampling window against the master costmap
//      by forward-simulating each candidate — one thread per SAMPLE.
//   6. Host: pick the best admissible sample, drive the plant one tick,
//      repeat.
// Three genuinely different GPU access patterns share one project: a small
// per-BEAM raytrace (07.09-style grid walk, but a 1-D launch), a per-CELL
// bounded gather (07.09's stencil family), and a per-SAMPLE rollout
// (08.01's MPPI pattern, reused for a *scoring* pass instead of a control
// blend). Cross-references throughout kernels.cu and THEORY.md.
//
// GRID LAYOUT — row-major, cell (x, y) at index y*width + x, x rightward,
// y downward (image convention, same as 07.09). One cell = kResolutionM
// meters on a side; world-frame metric coordinates convert to cell indices
// by floor(coord_m / kResolutionM). This mirrors nav_msgs/OccupancyGrid
// (SYSTEM_DESIGN.md §3.6): resolution_m, origin at cell (0,0), row-major data.
//
// COST SEMANTICS — one byte per cell, 0..254, mirroring Nav2's costmap_2d
// convention exactly (so the production comparison in THEORY.md is a real
// one, not just an analogy):
//     kCostFree    =   0   traversable, no known hazard
//     1..252              inflation GRADIENT — decays with distance to the
//                          nearest lethal cell (falls off with SQUARED
//                          distance, not linear distance — see kernels.cu
//                          for why: it keeps the whole pipeline exact
//                          integer arithmetic, no sqrt/exp)
//     kCostInscribed = 253  within the robot's inscribed radius of a lethal
//                          cell — the robot's OWN footprint would already be
//                          touching the obstacle even though this exact
//                          cell is not the obstacle cell
//     kCostLethal  = 254   an actual obstacle cell (static wall or sensed
//                          hit) — DWA admissibility hard-rejects any sampled
//                          arc that touches a cell at this cost
// (Nav2 reserves 255 for NO_INFORMATION / unknown space; this project's
// world is fully known a priori, so 255 is never produced and is not a
// named constant here — stated so the omission reads as a choice, not a gap.)
//
// LAYERS — three device buffers, one master:
//     static_layer    [W*H] unsigned char   the FULL true world map, loaded
//                     once from data/sample/*.pgm and uploaded verbatim —
//                     "what a prebuilt SLAM map already knows" (Nav2's
//                     static_layer). Read-only after upload.
//     obstacle_layer  [W*H] int             THIS TICK's sensed obstacles
//                     only (Nav2's obstacle_layer) — reset to kCostFree and
//                     rebuilt from the current scan every tick by the
//                     raytrace kernel. int, not unsigned char: CUDA has no
//                     native 1-byte atomicMax, and this layer is the one
//                     buffer multiple GPU threads race to write (kernels.cu
//                     §the race). See kernels.cu for the honest trade.
//     inflation_layer [W*H] unsigned char   this tick's inflation gradient,
//                     computed from the UNION of static_layer and
//                     obstacle_layer lethal cells.
//     master_costmap  [W*H] unsigned char   per-cell max(static, obstacle,
//                     inflation, converted to a common byte scale) — what
//                     DWA actually scores against, and the demo's PGM artifact.
//
// LIDAR SCAN LAYOUT — kNumBeams beams, evenly spaced over the full circle
// (angle_i = i * 2*pi/kNumBeams, i = 0..kNumBeams-1 — a simplified,
// omnidirectional stand-in for a real 2D LiDAR's angle_min/angle_increment;
// see THEORY.md and 04.01's fuller sensor-model treatment). The scan is
// PRE-DISCRETIZED on the host into integer grid endpoints before either GPU
// or CPU ever sees it (kernels.cu explains why this is what makes the
// obstacle layer byte-exact):
//     end_ix[i], end_iy[i]  the CELL this beam's ray reaches — either the
//                           true obstacle cell (hit[i] != 0) or the cell at
//                           kMaxRangeCells with no obstacle in the way
//                           (hit[i] == 0, a pure "clear to here" beam).
//     hit[i]                1 if the beam found an obstacle within range, 0
//                           if it ran out to max range and saw nothing.
//
// (v, w) SAMPLING WINDOW LAYOUT — a flat kVSamples*kWSamples grid of
// candidate (linear, angular) velocity pairs, row-major over (vi, wi):
// sample k -> vi = k / kWSamples, wi = k % kWSamples, and
//     v = lerp(v_lo, v_hi, vi / (kVSamples-1))
//     w = lerp(w_lo, w_hi, wi / (kWSamples-1))
// [v_lo,v_hi] x [w_lo,w_hi] is the DYNAMIC WINDOW: velocities the robot can
// actually reach in one control period given its acceleration limits,
// centered on last tick's applied (v,w) — computed on the host each tick
// (main.cu) and passed to the kernel as four scalars. This is the "Dynamic"
// in Dynamic Window Approach — see THEORY.md §the-math.
//
// Read this after: README.md.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

// ---------------------------------------------------------------------------
// World / grid constants — the committed sample (data/sample/world_map.pgm)
// is generated to EXACTLY these dimensions; main.cu's loader is strict and
// refuses a map file whose header disagrees (CLAUDE.md §9: never silently
// accept malformed input).
// ---------------------------------------------------------------------------
constexpr int   kGridW        = 256;    // cells, x axis (columns)
constexpr int   kGridH        = 256;    // cells, y axis (rows)
constexpr float kResolutionM  = 0.05f;  // meters per cell edge -> 12.8 x 12.8 m world
constexpr int   kGridTotal    = kGridW * kGridH;

// ---------------------------------------------------------------------------
// Cost semantics — see the file header. unsigned char keeps the static,
// inflation, and master layers at exactly one byte per cell (65536 cells =
// 64 KiB per layer — the reason a PGM image and a costmap layer are almost
// the same object, and why demo/out/costmap.pgm needs no rescaling at all:
// a cost byte IS a gray level, verbatim).
// ---------------------------------------------------------------------------
constexpr unsigned char kCostFree      = 0;
constexpr unsigned char kCostInscribed = 253;
constexpr unsigned char kCostLethal    = 254;

// ---------------------------------------------------------------------------
// Inflation geometry (cells). kInflationRadiusCells = 10 is the catalog's
// own number ("radius ~10 cells", CLAUDE.md task brief) — at kResolutionM
// that is 0.50 m, a plausible robot-radius-plus-margin for a small AMR.
// kInscribedRadiusCells = 4 (0.20 m) stands in for the robot's own physical
// radius: any cell within this distance of a lethal cell is a cell the
// robot's FOOTPRINT would already be touching the obstacle from, even
// though the cell itself never registered a hit (PRACTICE.md §1 revisits
// this — footprint vs. point-robot planning is a real simplification).
// ---------------------------------------------------------------------------
constexpr int kInscribedRadiusCells  = 4;
constexpr int kInflationRadiusCells  = 10;
constexpr int kInscribedR2           = kInscribedRadiusCells * kInscribedRadiusCells;   // 16
constexpr int kInflationR2           = kInflationRadiusCells * kInflationRadiusCells;   // 100

// ---------------------------------------------------------------------------
// LiDAR model constants (the simulated sensor — see reference_cpu.cpp's
// twin-free host-only scan simulator, called from main.cu).
// ---------------------------------------------------------------------------
constexpr int   kNumBeams     = 360;              // one beam per degree, full circle
constexpr float kMaxRangeM    = 6.0f;             // sensor max range (m)
constexpr int   kMaxRangeCells = 120;             // kMaxRangeM / kResolutionM, precomputed
                                                   // (constexpr float division is legal but
                                                   // int truncation is the value we actually
                                                   // want spelled out, not left implicit)

// ---------------------------------------------------------------------------
// DWA sampling window (the (v,w) grid) and rollout geometry.
// ---------------------------------------------------------------------------
constexpr int   kVSamples        = 64;            // linear-velocity samples
constexpr int   kWSamples        = 64;            // angular-velocity samples
constexpr int   kNumDwaSamples   = kVSamples * kWSamples;  // 4096 — one thread each
constexpr float kVMax            = 0.6f;          // m/s, forward-only (see THEORY §algorithm)
constexpr float kVMin            = 0.0f;          // m/s — no reverse in this teaching scope
constexpr float kWMax            = 1.0f;          // rad/s, symmetric turn-rate limit
constexpr float kAccelV          = 0.8f;          // m/s^2, max linear acceleration
constexpr float kAccelW          = 2.0f;          // rad/s^2, max angular acceleration
constexpr float kDtControl       = 0.1f;          // s — control/costmap tick period (10 Hz;
                                                   // SYSTEM_DESIGN.md's 5-20 Hz costmap band)
constexpr float kHorizonS        = 2.0f;          // s — how far each candidate is simulated
constexpr int   kRolloutSubsteps = 20;             // RK4 steps per rollout -> dt_sub = 0.1 s
constexpr float kDtSub           = kHorizonS / static_cast<float>(kRolloutSubsteps);

// Stage-cost weights (unitless; the tuning story is THEORY.md's — same
// "document the tuning, don't hide it" convention as 08.01's kW*):
constexpr float kWObstacle = 6.0f;   // on mean sampled costmap cost / kCostLethal, in [0,1]
constexpr float kWGoalDist = 3.0f;   // on remaining distance to goal / mission-start distance
constexpr float kWHeading  = 1.5f;   // on (1 - cos(heading error)) — wrap-free, MPPI-style
constexpr float kWSpeed    = 1.0f;   // SUBTRACTED: on v / kVMax — rewards making progress

// Any candidate whose rollout touches a lethal cell (or leaves the mapped
// area) is INADMISSIBLE: its score is clamped to this large-but-finite
// sentinel so host argmin never selects it, yet no arithmetic on it can
// produce inf/NaN (CLAUDE.md §12 determinism spirit — finite beats infinite
// even for "this should never win").
constexpr float kInadmissibleScore = 1.0e6f;

constexpr float kGoalTolM = 0.3f;    // success radius (m) — "reached the goal"

#ifdef __CUDACC__  // ---- device-aware section: only nvcc sees this ---------

// raytrace_kernel — one thread per LiDAR beam: Bresenham-walk the COSTMAP
// grid from the robot's cell to this beam's precomputed endpoint cell,
// clearing every intermediate cell and marking the endpoint lethal iff the
// beam actually hit something. Writes obstacle_layer via atomicMax — the
// race and why atomicMax fixes it are kernels.cu's centerpiece comment.
//   robot_ix, robot_iy : the robot's current cell (int, grid frame)
//   end_ix, end_iy, hit : DEVICE pointers, [kNumBeams] each — this tick's
//                         precomputed scan (uploaded by launch_costmap_update)
//   obstacle_layer      : DEVICE pointer, [kGridW*kGridH] int, OUT (caller
//                         must have reset it to kCostFree before this launch)
__global__ void raytrace_kernel(int robot_ix, int robot_iy,
                                const int* __restrict__ end_ix,
                                const int* __restrict__ end_iy,
                                const unsigned char* __restrict__ hit,
                                int* __restrict__ obstacle_layer);

// inflation_kernel — one thread per CELL: bounded-radius gather over a
// (2*kInflationRadiusCells+1)^2 window, finds the nearest lethal cell in
// static_layer OR obstacle_layer, and writes a squared-distance decay cost.
//   static_layer, obstacle_layer : DEVICE pointers, [kGridW*kGridH], IN
//   inflation_layer              : DEVICE pointer, [kGridW*kGridH], OUT
__global__ void inflation_kernel(const unsigned char* __restrict__ static_layer,
                                 const int* __restrict__ obstacle_layer,
                                 unsigned char* __restrict__ inflation_layer);

// fusion_kernel — one thread per CELL: master = max(static, obstacle,
// inflation). The map pattern from the scaffold SAXPY placeholder, now
// doing the project's real work: three independent reads, one write, no
// interaction between cells or threads at all.
__global__ void fusion_kernel(const unsigned char* __restrict__ static_layer,
                              const int* __restrict__ obstacle_layer,
                              const unsigned char* __restrict__ inflation_layer,
                              unsigned char* __restrict__ master_costmap);

// dwa_score_kernel — one thread per (v,w) SAMPLE: forward-simulate the
// diff-drive unicycle for kHorizonS seconds under constant (v,w), sampling
// the master costmap along the way; score admissible arcs, sentinel-out
// arcs that touch a lethal cell or leave the map. See THEORY.md §the-math
// for the scoring derivation.
//   master        : DEVICE pointer, [kGridW*kGridH] unsigned char, IN
//   pose_x/y/theta: the robot's CURRENT pose (m, m, rad) — every sample
//                   rolls out from this SAME start, like MPPI's x0
//   goal_x/y      : the goal position (m), world frame
//   v_lo/hi, w_lo/hi : this tick's dynamic window bounds (m/s, rad/s)
//   mission_dist  : distance-to-goal AT MISSION START (m) — the fixed
//                   normalizer for the goal-progress term (kept constant
//                   across the whole run so the term's scale is stable,
//                   not recomputed from the shrinking current distance)
//   scores        : DEVICE pointer, [kNumDwaSamples] float, OUT
__global__ void dwa_score_kernel(const unsigned char* __restrict__ master,
                                 float pose_x, float pose_y, float pose_theta,
                                 float goal_x, float goal_y,
                                 float v_lo, float v_hi, float w_lo, float w_hi,
                                 float mission_dist,
                                 float* __restrict__ scores);

#endif // __CUDACC__ --------------------------------------------------------

// ---------------------------------------------------------------------------
// Host launchers (defined in kernels.cu). Both own their grid/block math,
// their internal scan upload (small per-tick data, uploaded inside — the
// same "stateless launcher" trade 08.01 makes for its 16-byte x0), and the
// mandatory post-launch error check.
//
// launch_costmap_update — runs the whole GPU costmap pipeline for one tick:
// reset obstacle_layer, raytrace the scan into it, inflate, fuse. All FOUR
// device buffers below are PERSISTENT — allocated once in main.cu, reused
// every tick (unlike the tiny per-call scan upload, W*H buffers are large
// enough that alloc/free every tick would be wasteful, not just "trivial").
//   robot_ix, robot_iy       : robot's current cell
//   end_ix, end_iy, hit      : HOST pointers, [kNumBeams] — this tick's scan
//                              (uploaded to device internally)
//   d_static                 : DEVICE pointer, [kGridW*kGridH] unsigned char,
//                              the (already-uploaded, unchanging) static layer
//   d_obstacle                : DEVICE pointer, [kGridW*kGridH] int, scratch
//                              (reset by this call)
//   d_inflation, d_master     : DEVICE pointers, [kGridW*kGridH] unsigned char, OUT
// ---------------------------------------------------------------------------
void launch_costmap_update(int robot_ix, int robot_iy,
                           const int* end_ix, const int* end_iy, const unsigned char* hit,
                           const unsigned char* d_static,
                           int* d_obstacle,
                           unsigned char* d_inflation,
                           unsigned char* d_master);

// launch_dwa_scores — runs dwa_score_kernel for the whole (v,w) window.
// d_master is a persistent device buffer (this tick's fused costmap);
// d_scores is a persistent [kNumDwaSamples] float OUT buffer.
void launch_dwa_scores(const unsigned char* d_master,
                       float pose_x, float pose_y, float pose_theta,
                       float goal_x, float goal_y,
                       float v_lo, float v_hi, float w_lo, float w_hi,
                       float mission_dist,
                       float* d_scores);

// ---------------------------------------------------------------------------
// CPU references (reference_cpu.cpp) — the oracle twins of the three GPU
// costmap kernels (fused into one call, matching launch_costmap_update's
// contract exactly) and of the DWA scoring kernel, plus the diff-drive
// PLANT (the closed loop's single defined angle-wrap point, CLAUDE.md §12 —
// mirroring 08.01's cartpole_step_cpu).
// ---------------------------------------------------------------------------

// costmap_update_cpu — sequential oracle twin of launch_costmap_update.
// All host pointers; obstacle_layer/inflation_layer/master are OUT (sized
// [kGridW*kGridH], caller-allocated). Uses the SAME max-combine rule the
// GPU's atomicMax computes (kernels.cu explains why "max, sequentially" is
// the correct oracle for "max, racily" — not just "any order that happens
// to work").
void costmap_update_cpu(int robot_ix, int robot_iy,
                        const int* end_ix, const int* end_iy, const unsigned char* hit,
                        const unsigned char* static_layer,
                        int* obstacle_layer,
                        unsigned char* inflation_layer,
                        unsigned char* master_costmap);

// dwa_scores_cpu — sequential oracle twin of dwa_score_kernel. scores is
// OUT, sized [kNumDwaSamples].
void dwa_scores_cpu(const unsigned char* master,
                    float pose_x, float pose_y, float pose_theta,
                    float goal_x, float goal_y,
                    float v_lo, float v_hi, float w_lo, float w_hi,
                    float mission_dist,
                    float* scores);

// diffdrive_step_cpu — THE PLANT: advance pose[3] = {x_m, y_m, theta_rad}
// by one dt under constant (v, w) using the same RK4 integrator the
// rollouts use internally, then wrap theta to (-pi, pi] — the project's
// SINGLE DEFINED WRAP POINT (CLAUDE.md §12). Rollouts (GPU and CPU) never
// wrap; only the real plant does, exactly mirroring 08.01's cartpole_step_cpu.
void diffdrive_step_cpu(float* pose, float v, float w, float dt);

#endif // PROJECT_KERNELS_CUH
