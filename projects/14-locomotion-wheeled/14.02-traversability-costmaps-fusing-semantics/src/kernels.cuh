// ===========================================================================
// kernels.cuh — interface for project 14.02
//               Traversability costmaps fusing semantics + geometry
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (terrain + semantics synthesis, orchestration),
// kernels.cu (the four GPU kernels), and reference_cpu.cpp (their four CPU
// oracle twins). Everything all three must agree on — the elevation-map and
// semantic-map layouts, the six-class palette and its prior costs, the cost
// semantics, and every tuned constant — is defined HERE, once (CLAUDE.md §12).
//
// The pipeline in five lines (THEORY.md derives every step properly):
//   1. GEOMETRIC LAYER: fit a least-squares plane to each cell's local window
//      of neighboring heights (slope, roughness — same idea as 13.03's
//      foothold scorer) PLUS a second, smaller window's max-min height swing
//      (step height) — three per-cell numbers from one windowed pass.
//   2. SEMANTIC LAYER: a per-cell class prior cost, PULLED TOWARD a
//      documented pessimistic fallback as the simulated segmentation
//      confidence drops — "I don't trust this label" degrades to "I don't
//      know", never to "I'll assume it's fine".
//   3. FUSION: two independent evidence channels become one cost, with HARD
//      VETOES (water; slope or step past a wheeled-vehicle limit) that no
//      blend weight can soften, and a WEIGHTED BLEND everywhere else, so a
//      confident semantic reading can "rescue" a geometrically noisy cell
//      (tall grass) and a confident geometric reading cannot rescue a cell
//      the semantic channel refuses on safety grounds (water).
//   4. SPEED LIMIT: a curvature-free, straight-line stopping-distance bound
//      turns the fused cost into a per-cell max safe speed — the number a
//      downstream sampling controller (14.01's MPPI, cited by name in
//      README) actually consumes.
// Steps 1-4 are all PER-CELL maps/stencils (one thread per one of the 65536
// grid cells) — a deliberately SIMPLER GPU-mapping shape than 13.03's mixed
// per-cell/per-query pipeline (THEORY.md §The GPU mapping explains why nothing
// here needs a batched-query stage: a costmap has no "queries", only cells).
//
// ELEVATION-MAP LAYOUT — float elevation_m[W*H], row-major (mirrors the
// flattened nav_msgs/OccupancyGrid-style local-terrain-patch convention in
// docs/SYSTEM_DESIGN.md §3.6, generalized from occupancy bytes to heights,
// exactly as 13.03's height_m does):
//     elevation_m[row*W + col]  height (m) of cell (row,col), MAP frame.
//     cell world coordinates:    x_m = col * kCellM   (map-local "x")
//                                y_m = row * kCellM   (map-local "y")
//     origin (0,0) at the map's (row=0,col=0) corner; the map spans
//     [0, kGridW*kCellM) x [0, kGridH*kCellM) meters, right-handed, z-up
//     (SYSTEM_DESIGN §3.2). No NaN/holes in this project (see README
//     Limitations — a deliberate scope choice that keeps the two-channel
//     FUSION story, not sensor dropout, the center of the teaching).
//
// SEMANTIC-MAP LAYOUT — parallel arrays, same row-major indexing:
//     semantic_class[row*W + col]   uint8_t in [0, kNumClasses), the
//                                   segmentation net's ARGMAX class ID
//                                   (the six-class palette below).
//     confidence[row*W + col]       float in [0,1], the ARGMAX class's
//                                   simulated softmax confidence — what a
//                                   real segmentation net (12.x/30.x) would
//                                   also hand a fusion stage.
//
// THE SIX-CLASS PALETE — one CLASS_* id, one documented PRIOR TRAVERSABILITY
// COST (0 = trivially easy, 1 = as bad as a hard veto), single-sourced here
// so kernels.cu, reference_cpu.cpp, main.cu and every doc quote the SAME
// numbers (CLAUDE.md §12's single-source-of-truth rule):
//     CLASS_DIRT        0   0.05   firm bare/compacted ground — the easy case
//     CLASS_GRAVEL       1   0.10   loose but firm, good drainage/traction
//     CLASS_GRASS        2   0.20   short/mown ground cover, mild traction hit
//     CLASS_VEGETATION   3   0.45   tall grass/brush — occludes the ground;
//                                   may hide small hazards (README/THEORY)
//     CLASS_WATER        4   1.00   HARD VETO regardless of confidence (see
//                                   fusion_kernel) — unknown depth/current
//     CLASS_UNKNOWN      5   0.65   no confident label at all
// kPessimisticPriorCost (the confidence fallback target, see below) is
// DELIBERATELY EQUAL to CLASS_UNKNOWN's own prior: a label we do not trust
// degrades toward "I don't know", never toward "I'll assume it's cheap".
//
// COST SEMANTICS (a deliberate NAMING CHANGE from 13.03's "score", to match
// this catalog bullet's own word "costmaps" and Nav2's costmap_2d
// convention, 23.01's closest analogue in this repo): every per-cell number
// this project calls a "cost" lives in [0,1] with 0 = free/trivial and
// 1 = LETHAL (hard-vetoed) — the OPPOSITE polarity from 13.03's "score"
// (1 = best). README §Limitations names this explicitly so a reader moving
// between the two sibling projects is not confused mid-sentence.
//
// UNITS (SYSTEM_DESIGN §3.1): meters, radians, m/s, unitless [0,1] costs and
// confidences. Every array is named with its unit as a suffix (…_m, …_rad,
// …_mps) except the two pure-cost/confidence arrays, which are unitless by
// construction and named accordingly.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cstdint>   // uint8_t — the semantic-class id's storage type

// ---------------------------------------------------------------------------
// Grid geometry (compile-time: this demo always studies one 256x256 map).
// 256x256 @ 0.10 m/cell = 25.6 x 25.6 m — the OFF-ROAD scale named in the
// catalog bullet: five times 13.03's indoor-quadruped 5.12x5.12 m patch at
// coarser resolution (0.10 m vs 0.02 m), because a wheeled vehicle's tire
// footprint and the terrain features it cares about (a ditch, a berm) are
// meters, not centimeters, across.
// ---------------------------------------------------------------------------
constexpr int   kGridW  = 256;    // columns (map-local x)
constexpr int   kGridH  = 256;    // rows    (map-local y)
constexpr float kCellM  = 0.10f;  // meters per cell edge

// ---------------------------------------------------------------------------
// The six-class semantic palette (README/THEORY table above — this is the
// single source of truth every file and doc quotes).
// ---------------------------------------------------------------------------
constexpr uint8_t CLASS_DIRT       = 0;
constexpr uint8_t CLASS_GRAVEL     = 1;
constexpr uint8_t CLASS_GRASS      = 2;
constexpr uint8_t CLASS_VEGETATION = 3;
constexpr uint8_t CLASS_WATER      = 4;
constexpr uint8_t CLASS_UNKNOWN    = 5;
constexpr int      kNumClasses     = 6;

// Per-class PRIOR traversability cost, indexed by the CLASS_* constants
// above — ONE literal list both nvcc (device code) and cl.exe (the CPU
// oracle) compile (CLAUDE.md §12 single-source rule). Runtime indexing
// happens inside a device kernel (semantic_layer_kernel/fusion_kernel), so
// under nvcc this needs real device-memory storage (__device__) — cl.exe
// does not know that keyword, hence the __CUDACC__ fence; the VALUES are
// written exactly once either way. THEORY.md §The problem derives every
// number's physical reasoning; PRACTICE.md dates and caveats them as
// illustrative.
#ifdef __CUDACC__
__device__
#endif
constexpr float kClassPriorCost[kNumClasses] = {
    0.05f,  // CLASS_DIRT
    0.10f,  // CLASS_GRAVEL
    0.20f,  // CLASS_GRASS
    0.45f,  // CLASS_VEGETATION
    1.00f,  // CLASS_WATER (also independently hard-vetoed in fusion_kernel —
            // the prior cost matters only if that veto is ever bypassed by a
            // future extension; THEORY.md discusses why the veto exists at
            // all instead of relying on this prior alone)
    0.65f,  // CLASS_UNKNOWN
};

// The confidence fallback target (semantic_layer_kernel below): as confidence
// -> 0, semantic_cost -> kPessimisticPriorCost, NOT toward the class's own
// (possibly cheap) prior. Deliberately equal to CLASS_UNKNOWN's own prior —
// "I don't trust this label" and "I have no label" converge to the same
// number (THEORY.md §The math derives the blend formula this constant feeds).
constexpr float kPessimisticPriorCost = kClassPriorCost[CLASS_UNKNOWN];  // 0.65f

// ---------------------------------------------------------------------------
// Tuned algorithm constants — geometry (windowed-gather sizes) and the
// wheeled-vehicle physical parameters THEORY.md §The problem derives the two
// hard-veto limits from. Like 13.03's kFrictionMu etc., these are the
// "tuned, taught setup" (CLAUDE.md §8's distinction from DATA, which lives
// in data/sample/traversability_scenario.csv).
// ---------------------------------------------------------------------------
constexpr int kFitRadiusCells = 3;   // slope/roughness plane-fit window
                                     // half-width, CELLS: a 7x7 = 0.7x0.7 m
                                     // window — approximately a small/mid
                                     // off-road UGV's wheelbase-and-track
                                     // footprint (README §algorithm; THEORY
                                     // contrasts this with 13.03's much
                                     // smaller foot-pad-sized 5x5 window).
constexpr int kStepRadiusCells = 2;  // step-height window half-width, CELLS:
                                     // a SMALLER 5x5 = 0.4x0.4 m window,
                                     // sized to a single wheel/track contact
                                     // patch — deliberately tighter than the
                                     // slope-fit window so a sharp discrete
                                     // edge (a ditch lip, a berm crest) is
                                     // localized instead of smoothed away by
                                     // the wider plane fit (THEORY.md §The
                                     // algorithm explains why one window
                                     // cannot serve both jobs well).

constexpr float kWheelRadiusM   = 0.20f;  // illustrative off-road UGV wheel
                                          // radius (m) — PRACTICE.md §2 dates
                                          // and caveats this as an example.
constexpr float kWheelMu        = 0.7f;   // tire-on-dirt/gravel Coulomb
                                          // friction coefficient — reused for
                                          // BOTH the traction slope limit and
                                          // the step-climb limit below
                                          // (THEORY.md §The problem derives
                                          // both from this single constant).
constexpr float kTrackWidthM    = 0.6f;   // illustrative track width (m):
                                          // lateral distance between the
                                          // left/right wheel contact lines.
constexpr float kCogHeightM     = 0.6f;   // illustrative center-of-gravity
                                          // height (m) above the ground
                                          // plane (elevated by a sensor mast
                                          // — THEORY.md's "why rollover, not
                                          // just friction, can govern" story).
constexpr float kGravityMps2    = 9.81f;  // standard gravity, m/s^2.

constexpr float kRoughnessMaxM = 0.012f; // roughness (m) at which rough_cost
                                         // saturates to 1 — the geometric
                                         // layer's third hazard signal,
                                         // alongside slope and step. Tuned
                                         // tighter than a first guess might
                                         // suggest: at this map's 0.10 m
                                         // resolution and kFitRadiusCells=3
                                         // window, a canopy-noise patch's
                                         // step-height signal saturates its
                                         // OWN limit (kStepRadiusCells's
                                         // tighter window) well before a
                                         // loose roughness threshold would
                                         // register anything — this value
                                         // keeps roughness a genuinely
                                         // independent, sensitive third
                                         // signal instead of a redundant,
                                         // rarely-triggered one (README
                                         // Exercise; THEORY.md §Numerical
                                         // considerations measures this).

// Geometric sub-cost blend weights (sum to 1.0 — THEORY.md §The math).
constexpr float kWeightSlope = 0.4f;
constexpr float kWeightStep  = 0.3f;
constexpr float kWeightRough = 0.3f;

// Fusion blend weights: geometry vs. semantics, EQUAL TRUST by default (sum
// to 1.0) — THEORY.md §The two-channel fusion problem discusses when a real
// system would weight these unevenly (per-channel measured reliability).
constexpr float kWeightGeo = 0.5f;
constexpr float kWeightSem = 0.5f;

// A fused cost at or below this value counts as VALID/drivable in the
// analytic gates and the printed summary; strictly a REPORTING/gating
// threshold, not a physical limit like the two hard vetoes above (compare
// 13.03's kValidThreshold, inverted here because this project uses COST, not
// SCORE, polarity — see the file header's "COST SEMANTICS" note).
constexpr float kMaxValidCost = 0.6f;

// ---------------------------------------------------------------------------
// Speed-limit kernel constants — the curvature-free stopping-distance
// argument THEORY.md §The math derives in full.
// ---------------------------------------------------------------------------
constexpr float kVMaxMps        = 2.5f;         // operator cruise-speed cap
                                                // (m/s) — the speed a cell
                                                // with cost=0 is allowed,
                                                // independent of terrain.
constexpr float kSafetyFraction = 1.0f / 3.0f;  // fraction of the wheel's
                                                // total friction budget this
                                                // bound RESERVES for
                                                // straight-line braking
                                                // (the rest is margin for
                                                // simultaneous steering — a
                                                // friction-circle argument,
                                                // THEORY.md §The math).
constexpr float kStopDistM      = 2.0f;         // fixed look-ahead/reaction
                                                // distance (m) the vehicle
                                                // must be able to fully stop
                                                // within — a property of the
                                                // SENSOR RANGE and control
                                                // rate, deliberately NOT of
                                                // any assumed path curvature
                                                // (README/THEORY "curvature-
                                                // free" discussion).

// Veto-reason bit flags (fusion_kernel's fourth output, veto_reason[]) — a
// small diagnostic breadcrumb the analytic gates and demo/out/layers.csv use
// to say WHY a cell was vetoed, not just THAT it was.
constexpr int32_t kVetoNone = 0;
constexpr int32_t kVetoGeo  = 1;   // bit 0: slope or step past its limit
constexpr int32_t kVetoSem  = 2;   // bit 1: semantic class is CLASS_WATER

#ifdef __CUDACC__  // ---- device-aware section: only nvcc sees this ---------

// Kernel declarations. Full documentation (thread mapping, memory spaces,
// numerics) sits with each definition in kernels.cu.

__global__ void geometric_layer_kernel(const float* __restrict__ elevation_m,
                                       float* __restrict__ slope_rad,
                                       float* __restrict__ step_height_m,
                                       float* __restrict__ roughness_m);

__global__ void semantic_layer_kernel(const uint8_t* __restrict__ semantic_class,
                                      const float* __restrict__ confidence,
                                      float* __restrict__ semantic_cost);

__global__ void fusion_kernel(const float* __restrict__ slope_rad,
                              const float* __restrict__ step_height_m,
                              const float* __restrict__ roughness_m,
                              const uint8_t* __restrict__ semantic_class,
                              const float* __restrict__ semantic_cost,
                              float slope_limit_rad,
                              float step_limit_m,
                              float* __restrict__ geo_cost,
                              float* __restrict__ fused_cost,
                              int32_t* __restrict__ veto_reason);

__global__ void speed_limit_kernel(const float* __restrict__ fused_cost,
                                   float* __restrict__ speed_limit_mps);

#endif // __CUDACC__ --------------------------------------------------------

// ---------------------------------------------------------------------------
// Host launch wrappers — own the grid/block math + post-launch error check
// (CLAUDE.md §6.1 rule 7). d_* are DEVICE pointers the caller allocated.
// All four operate on the fixed kGridW x kGridH grid.
// ---------------------------------------------------------------------------
void launch_geometric_layer(const float* d_elevation_m,
                            float* d_slope_rad, float* d_step_height_m,
                            float* d_roughness_m);

void launch_semantic_layer(const uint8_t* d_semantic_class,
                           const float* d_confidence, float* d_semantic_cost);

void launch_fusion(const float* d_slope_rad, const float* d_step_height_m,
                   const float* d_roughness_m, const uint8_t* d_semantic_class,
                   const float* d_semantic_cost, float slope_limit_rad,
                   float step_limit_m, float* d_geo_cost, float* d_fused_cost,
                   int32_t* d_veto_reason);

void launch_speed_limit(const float* d_fused_cost, float* d_speed_limit_mps);

// ---------------------------------------------------------------------------
// CPU references (reference_cpu.cpp) — the oracle twins of the four
// kernels above, sequential over cells. main.cu runs each against the GPU
// kernel with SHARED, PINNED upstream inputs (13.03's stage-isolation
// technique — see main.cu's file header) so every VERIFY gate isolates
// exactly one kernel's correctness.
// ---------------------------------------------------------------------------
void geometric_layer_cpu(const float* elevation_m,
                         float* slope_rad, float* step_height_m, float* roughness_m);

void semantic_layer_cpu(const uint8_t* semantic_class, const float* confidence,
                        float* semantic_cost);

void fusion_cpu(const float* slope_rad, const float* step_height_m,
                const float* roughness_m, const uint8_t* semantic_class,
                const float* semantic_cost, float slope_limit_rad,
                float step_limit_m, float* geo_cost, float* fused_cost,
                int32_t* veto_reason);

void speed_limit_cpu(const float* fused_cost, float* speed_limit_mps);

#endif // PROJECT_KERNELS_CUH
