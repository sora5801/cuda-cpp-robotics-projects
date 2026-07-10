// ===========================================================================
// kernels.cuh — interface for project 13.03
//               Foothold scoring kernels: slope, roughness, edge distance
//               from elevation maps
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (terrain synthesis + orchestration),
// kernels.cu (the four GPU kernels), and reference_cpu.cpp (their four CPU
// oracle twins). Everything all three must agree on — the elevation-map
// layout, the query/result record shapes, and the tuned algorithm constants
// — is defined HERE, once (CLAUDE.md §12).
//
// The pipeline in five lines (THEORY.md derives every step properly):
//   1. SLOPE + ROUGHNESS: fit a least-squares plane to each cell's local
//      window of neighboring heights; slope = angle of that plane's normal
//      from vertical, roughness = std-dev of the fit's residuals.
//   2. EDGE DISTANCE: mark cells "hazardous" (unknown height, too steep, or
//      too rough) and, for every other cell, find the distance to the
//      nearest hazard within a bounded search window.
//   3. FUSION: blend slope/roughness/edge-distance into one score in [0,1]
//      per cell, with two HARD VETOES (unknown height; slope past the
//      friction-derived limit) that force score = 0 regardless of weights.
//   4. FOOTHOLD SELECTION: for each of ~1000 nominal landing points along a
//      walking path, search a small disc of the score grid and return the
//      best-scoring valid cell — the map becomes a foot placement.
// Steps 1-3 are PER-CELL maps (one thread per of the 65536 grid cells);
// step 4 is a per-query batched search (one thread per query, ~1000
// threads) — two different GPU mapping shapes in one pipeline, contrasted
// in THEORY.md §The GPU mapping.
//
// ELEVATION-MAP LAYOUT — float height_m[W*H], row-major (mirrors the
// flattened nav_msgs/OccupancyGrid-style local-terrain-patch convention in
// docs/SYSTEM_DESIGN.md §3.6):
//     height_m[row*W + col]   height (m) of cell (row,col) in the MAP frame,
//                              NaN = unknown / sensor dropout (a hole)
//     cell world coordinates:  x_m = col * kCellM   (map-local "x", §3.2)
//                              y_m = row * kCellM   (map-local "y")
//     origin (0,0) sits at the map's (row=0,col=0) corner; the map spans
//     [0, kGridW*kCellM) x [0, kGridH*kCellM) meters — a right-handed,
//     z-up local patch (SYSTEM_DESIGN §3.2), NOT yet transformed into
//     `world`/`odom` — a real system applies T_world_map at the boundary.
//
// UNITS (SYSTEM_DESIGN §3.1): meters, radians, unitless [0,1] scores. Every
// array below is named with its unit as a suffix (…_m, …_rad).
//
// NUMERICS AT A GLANCE (THEORY.md §Numerical considerations has the full
// story): NaN PROPAGATES from a hole through the plane fit (a hole cell
// itself gets NaN slope/roughness; a cell whose *window* touches a hole
// simply has fewer samples, never a fabricated cliff) and then through
// FUSION's hard veto (score forced to exactly 0.0f) — never silently
// treated as zero height, which would invent a false cliff.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

// ---------------------------------------------------------------------------
// Grid geometry (compile-time: this demo always studies one 256x256 map).
// 256x256 @ 0.02 m/cell = 5.12 x 5.12 m — the footprint a 2.5-D elevation
// mapper like 05.05 would publish around a walking quadruped.
// ---------------------------------------------------------------------------
constexpr int   kGridW  = 256;    // columns (map-local x)
constexpr int   kGridH  = 256;    // rows    (map-local y)
constexpr float kCellM  = 0.02f;  // meters per cell edge

// ---------------------------------------------------------------------------
// Tuned algorithm constants — the "taught, tuned setup" (CLAUDE.md §8's
// distinction: terrain GEOMETRY is data in data/sample/terrain_scenario.csv;
// these THRESHOLDS/WEIGHTS are the algorithm and live here, like 08.01's
// kW* cost weights).
// ---------------------------------------------------------------------------
constexpr int   kFitRadius = 2;      // plane-fit window half-width, CELLS:
                                     // a (2*2+1)x(2*2+1) = 5x5 window, i.e. a
                                     // 0.1 m x 0.1 m patch — about the size
                                     // of a mid-size quadruped's foot pad
                                     // contact area (README §algorithm).
constexpr float kFrictionMu = 0.6f; // Coulomb friction coefficient, foot pad
                                     // on typical outdoor ground (rubber-like
                                     // pad on packed dirt/concrete) — THEORY
                                     // derives slope_limit_rad = atan(mu)
                                     // from the friction-cone condition.
constexpr float kRoughnessMaxM = 0.02f; // roughness (m) at which rough_score
                                     // saturates to 0 in the fused blend —
                                     // also the hazard threshold the
                                     // edge-distance kernel repels from.
constexpr float kEdgeSafeDistM = 0.10f; // distance (m) from the nearest
                                     // hazard at which edge_score saturates
                                     // to 1 — "far enough is far enough".
constexpr int   kEdgeSearchRadiusCells = 10; // bounded search window for the
                                     // distance-to-hazard gather (10 cells =
                                     // 0.2 m); THEORY.md ties this to 07.09's
                                     // (exact, unbounded) distance transform.
constexpr float kWeightSlope = 0.4f; // fusion blend weights (sum to 1.0);
constexpr float kWeightRough = 0.3f; // the tuning story is THEORY.md §fusion.
constexpr float kWeightEdge  = 0.3f;
constexpr float kValidThreshold = 0.5f; // minimum fused score to count as a
                                     // "valid" foothold in selection/gates.
constexpr float kFootholdSearchRadiusM = 0.10f; // foothold-selection search
                                     // disc radius (m) around each nominal
                                     // landing point (5 cells).

// ---------------------------------------------------------------------------
// FootholdQuery / FootholdResult — the consumer-facing records for step 4.
// Deliberately message-shaped (SYSTEM_DESIGN §3.6): a real gait/footstep
// planner (13.02/13.08) would receive FootholdResult as (part of) a
// geometry_msgs/PointStamped-like reply to a nominal swing-leg target.
// ---------------------------------------------------------------------------
struct FootholdQuery {
    float x_m, y_m;            // nominal landing point, map frame (meters)
};

struct FootholdResult {
    int   row, col;             // selected cell (-1,-1 if the disc held no
                                 // in-bounds cell at all — never happens here
                                 // since a query's own nominal cell is always
                                 // in range, but checked defensively)
    float score;                 // fused score at the selected cell, [0,1]
                                 // (0 if nothing better than the sentinel was
                                 // found — see kernels.cu)
    float dist_m;                 // Euclidean distance from (x_m,y_m) to the
                                 // selected cell's center (m) — must be
                                 // <= kFootholdSearchRadiusM by construction
    int   valid;                  // 1 iff score >= kValidThreshold
};

#ifdef __CUDACC__  // ---- device-aware section: only nvcc sees this ---------

// Kernel declarations. Full documentation (thread mapping, memory spaces,
// numerics) sits with each definition in kernels.cu — see the header
// comment there for why headers carry summaries and .cu files carry essays.

__global__ void slope_roughness_kernel(const float* __restrict__ height_m,
                                       float* __restrict__ slope_rad,
                                       float* __restrict__ roughness_m);

__global__ void edge_distance_kernel(const float* __restrict__ height_m,
                                     const float* __restrict__ slope_rad,
                                     const float* __restrict__ roughness_m,
                                     float slope_limit_rad,
                                     float* __restrict__ edge_dist_m);

__global__ void fusion_kernel(const float* __restrict__ height_m,
                              const float* __restrict__ slope_rad,
                              const float* __restrict__ roughness_m,
                              const float* __restrict__ edge_dist_m,
                              float slope_limit_rad,
                              float* __restrict__ score);

__global__ void foothold_selection_kernel(const float* __restrict__ score,
                                          const FootholdQuery* __restrict__ queries,
                                          int num_queries,
                                          FootholdResult* __restrict__ results);

#endif // __CUDACC__ --------------------------------------------------------

// ---------------------------------------------------------------------------
// Host launch wrappers — own the grid/block math + post-launch error check
// (CLAUDE.md §6.1 rule 7). d_* are DEVICE pointers the caller allocated.
// All four operate on the fixed kGridW x kGridH grid.
// ---------------------------------------------------------------------------
void launch_slope_roughness(const float* d_height_m,
                            float* d_slope_rad, float* d_roughness_m);

void launch_edge_distance(const float* d_height_m, const float* d_slope_rad,
                          const float* d_roughness_m, float slope_limit_rad,
                          float* d_edge_dist_m);

void launch_fusion(const float* d_height_m, const float* d_slope_rad,
                   const float* d_roughness_m, const float* d_edge_dist_m,
                   float slope_limit_rad, float* d_score);

void launch_foothold_selection(const float* d_score,
                               const FootholdQuery* d_queries, int num_queries,
                               FootholdResult* d_results);

// ---------------------------------------------------------------------------
// CPU references (reference_cpu.cpp) — the oracle twins of the four
// kernels above, sequential over cells/queries. main.cu runs each against
// the GPU kernel with SHARED, PINNED upstream inputs (never mixing a GPU
// stage's output into the CPU path or vice versa) so every gate isolates
// exactly one kernel's correctness — see THEORY.md §How we verify correctness.
// ---------------------------------------------------------------------------
void slope_roughness_cpu(const float* height_m,
                         float* slope_rad, float* roughness_m);

void edge_distance_cpu(const float* height_m, const float* slope_rad,
                       const float* roughness_m, float slope_limit_rad,
                       float* edge_dist_m);

void fusion_cpu(const float* height_m, const float* slope_rad,
                const float* roughness_m, const float* edge_dist_m,
                float slope_limit_rad, float* score);

void foothold_selection_cpu(const float* score,
                            const FootholdQuery* queries, int num_queries,
                            FootholdResult* results);

#endif // PROJECT_KERNELS_CUH
