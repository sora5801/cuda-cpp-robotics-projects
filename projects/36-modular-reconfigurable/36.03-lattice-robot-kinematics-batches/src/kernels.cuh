// ===========================================================================
// kernels.cuh — interface for project 36.03
//               Lattice-robot kinematics batches (sliding-cube model)
//               [R&D] catalog bullet, reduced-scope teaching version
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (batch generator + closed pipeline driver),
// kernels.cu (the four GPU stage kernels), and reference_cpu.cpp (the exact
// oracle twins PLUS two independent brute-force cross-checks). Everything
// all three must agree on — the lattice geometry, the position layout, the
// per-stage output layouts, and the move-direction numbering — is defined
// HERE, once (CLAUDE.md §12).
//
// The model in one paragraph (THEORY.md derives it properly)
// -------------------------------------------------------------------------
// A lattice robot is M unit-cube MODULES sitting on an integer 3D grid.
// Two modules are MECHANICALLY CONNECTED iff they occupy FACE-adjacent
// cells (Manhattan distance exactly 1 — sharing a cube face, the only way
// two real cubes can latch together). This is the "sliding-cube" abstract
// model used across the lattice self-reconfiguring robotics literature
// (crystalline/metamodule robots; see README "Prior art"): a module can
// change cells in exactly two ways —
//   SLIDE  (linear move, 1 cell): translate to a face-adjacent EMPTY cell,
//           sliding along the face of a perpendicular NEIGHBOR that acts as
//           a wall (README/THEORY diagram it in full).
//   CORNER (convex/pivot move, 1 edge-diagonal cell): rotate 90 degrees
//           around the edge of a face-adjacent PIVOT neighbor, landing on
//           an edge-diagonal cell, provided the "swept" corner cell is
//           empty (README/THEORY diagram it in full).
// A module that is a CUT VERTEX (articulation point) of the face-adjacency
// graph can never legally move — removing it would fracture the robot into
// two or more physically disconnected pieces before it lands anywhere.
//
// THE BATCH (the GPU content, CLAUDE.md's "batches" in the catalog title):
// K independent configurations of M modules each, processed one thread per
// CONFIGURATION through four staged kernels — validity, connectivity,
// articulation points, and legal-move enumeration — each stage's output
// gating the next (a config that fails validity is not trusted downstream;
// see main.cu's staged pipeline and THEORY.md "The GPU mapping" for why
// "one thread runs a whole small graph algorithm" beats parallelizing
// BFS/DFS itself at this M).
//
// ALL-INTEGER PROJECT (feature, not incidental — CLAUDE.md §12; contrast
// with 08.01/09.01's FP32 tolerance gates): every quantity here — cell
// coordinates, adjacency, validity, connectivity, articulation flags, move
// legality — is an exact integer predicate. There is no rounding anywhere
// in this pipeline, so the GPU-vs-CPU verify gate demands BIT-EXACT
// equality, not a tolerance. See THEORY.md "Numerical considerations".
//
// POSITION LAYOUT — int32_t pos[K * kM * 3], lattice CELLS (dimensionless
// integer grid coordinates; PRACTICE.md gives the nominal physical edge
// length a real module cell would carry):
//     pos[(k*kM + m)*3 + 0] = x of module m in configuration k
//     pos[(k*kM + m)*3 + 1] = y
//     pos[(k*kM + m)*3 + 2] = z
// Right-handed grid; no particular "up" axis is assumed by validity,
// connectivity, articulation, or move legality (the sliding-cube rules
// implemented here are ISOTROPIC — THEORY.md explains that scoping choice
// versus the gravity-biased variant common in hardware-focused papers).
//
// MOVE-DIRECTION NUMBERING — kNumMoveDirs = kNumSlideDirs + kNumCornerDirs
// = 6 + 12 = 18, one fixed global order every array below uses:
//     dir 0.. 5  SLIDE  : +x,-x,+y,-y,+z,-z            (see slide_delta())
//     dir 6..17  CORNER : the 12 edge-diagonal directions of a cube, 4 per
//                         axis-pair (xy, xz, yz) — see corner_axes() in
//                         kernels.cu / reference_cpu.cpp for the formula
//                         that turns a corner index into its two component
//                         slide directions (e, f) and hence its pivot
//                         cells. Diagrammed exhaustively in README/THEORY.
//
// PER-STAGE OUTPUT LAYOUTS (all length K unless noted; uint8_t flags are
// 0/1 booleans, chosen over `bool` because `bool` has no fixed cross-
// compiler layout guarantee for device<->host memcpy):
//     valid[K]            1 iff no two modules of config k share a cell
//     connected[K]        1 iff the face-adjacency graph of config k spans
//                          all kM modules (BFS from module 0) — computed
//                          regardless of `valid` (see kernels.cu header);
//                          only ASSERTED correct for valid&&connected
//                          configs by the verify gates in main.cu.
//     is_articulation[K*kM]   is_articulation[k*kM+m] = 1 iff module m is a
//                          cut vertex of config k's adjacency graph.
//     num_articulation[K] sum of the row above, per config.
//     legal_move[K*kM*kNumMoveDirs]  legal_move[(k*kM+m)*kNumMoveDirs+d]=1
//                          iff module m of config k has a mechanically
//                          legal move in direction d AND m is NOT an
//                          articulation point (the catalog's "for each
//                          non-articulation module" scoping, folded
//                          directly into this flag).
//     move_count[K]       sum of the kM*kNumMoveDirs row above, per config.
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cstdint>

// ---------------------------------------------------------------------------
// Fixed lattice-batch geometry (CLAUDE.md §12: shared constants live once).
// kM is a COMPILE-TIME constant (unlike K, which main.cu may size at the
// command line) because every per-thread local array below (adjacency
// scratch, DFS bookkeeping) is fixed-size — the same "small array baked
// into local/register storage" pattern 08.01 uses for its kNX=4 state, just
// bigger here because a whole graph algorithm, not an ODE step, runs per
// thread (THEORY.md "The GPU mapping" argues the size honestly).
// ---------------------------------------------------------------------------
constexpr int kM = 24;                 // modules per configuration (fixed)
constexpr int kNumSlideDirs = 6;       // +x,-x,+y,-y,+z,-z
constexpr int kNumCornerDirs = 12;     // the cube's 12 edge-diagonals
constexpr int kNumMoveDirs = kNumSlideDirs + kNumCornerDirs;  // 18

constexpr int kDefaultK = 4096;        // configurations per batch (repo default)

// ---------------------------------------------------------------------------
// launch_validity — Stage 1: per-config duplicate-position detection.
//
//   K        : number of configurations (>= 1).
//   d_pos    : DEVICE pointer, K*kM*3 int32_t — see POSITION LAYOUT above.
//   d_valid  : DEVICE pointer, K uint8_t OUT — 1 iff no two of the kM
//              modules share a cell.
//
// Launch: one thread per configuration (the repo's thread-per-problem
// pattern, e.g. 08.01/09.01/33.01); each thread sorts kM packed position
// keys in LOCAL memory and scans for adjacent duplicates (kernels.cu).
// ---------------------------------------------------------------------------
void launch_validity(int K, const int32_t* d_pos, uint8_t* d_valid);

// ---------------------------------------------------------------------------
// launch_connectivity — Stage 2: face-adjacency graph spans all kM modules?
//
//   K, d_pos : as above.
//   d_valid  : DEVICE pointer, K uint8_t — Stage 1's output (read-only here;
//              connectivity is computed unconditionally, but is only
//              ASSERTED correct downstream when valid[k]==1 — see the
//              header comment above).
//   d_connected : DEVICE pointer, K uint8_t OUT.
//
// Launch: one thread per configuration; each thread runs a BFS from module
// 0 over the kM-node graph, entirely in local arrays.
// ---------------------------------------------------------------------------
void launch_connectivity(int K, const int32_t* d_pos, const uint8_t* d_valid,
                         uint8_t* d_connected);

// ---------------------------------------------------------------------------
// launch_articulation — Stage 3: cut-vertex (articulation point) detection.
//
//   K, d_pos, d_valid, d_connected : as above.
//   d_is_articulation : DEVICE pointer, K*kM uint8_t OUT.
//   d_num_articulation : DEVICE pointer, K int32_t OUT (row sums).
//
// Launch: one thread per configuration; each thread runs the classic
// Tarjan DFS low-link algorithm (iterative, explicit stack — THEORY.md
// teaches it step by step) over its own kM-node graph.
// ---------------------------------------------------------------------------
void launch_articulation(int K, const int32_t* d_pos,
                         const uint8_t* d_valid, const uint8_t* d_connected,
                         uint8_t* d_is_articulation, int32_t* d_num_articulation);

// ---------------------------------------------------------------------------
// launch_move_enum — Stage 4: legal sliding/corner moves per module.
//
//   K, d_pos, d_valid, d_connected : as above.
//   d_is_articulation : DEVICE pointer, K*kM uint8_t — Stage 3's output;
//                       articulation modules are forced non-movable here.
//   d_legal_move : DEVICE pointer, K*kM*kNumMoveDirs uint8_t OUT.
//   d_move_count : DEVICE pointer, K int32_t OUT (row sums).
//
// Launch: one thread per configuration; up to kM*kNumMoveDirs = 432
// precondition checks per thread, each O(kM) — see kernels.cu.
// ---------------------------------------------------------------------------
void launch_move_enum(int K, const int32_t* d_pos,
                      const uint8_t* d_valid, const uint8_t* d_connected,
                      const uint8_t* d_is_articulation,
                      uint8_t* d_legal_move, int32_t* d_move_count);

// ---------------------------------------------------------------------------
// CPU references (reference_cpu.cpp) — exact oracle twins of the four
// launchers above (same signatures, HOST pointers, sequential over k).
// main.cu runs these against the GPU kernels over the FULL batch and
// requires BIT-EXACT integer agreement (the project's §5 gate; see the
// ALL-INTEGER note in the file header).
// ---------------------------------------------------------------------------
void validity_cpu(int K, const int32_t* pos, uint8_t* valid);
void connectivity_cpu(int K, const int32_t* pos, const uint8_t* valid,
                      uint8_t* connected);
void articulation_cpu(int K, const int32_t* pos,
                      const uint8_t* valid, const uint8_t* connected,
                      uint8_t* is_articulation, int32_t* num_articulation);
void move_enum_cpu(int K, const int32_t* pos,
                   const uint8_t* valid, const uint8_t* connected,
                   const uint8_t* is_articulation,
                   uint8_t* legal_move, int32_t* move_count);

// ---------------------------------------------------------------------------
// Brute-force ORACLES (reference_cpu.cpp) — independently coded, obviously-
// correct-by-inspection cross-checks used on a SUBSET of the batch (main.cu
// §6). These are NOT performance baselines (at kM=24 there is no complexity-
// class gap to demonstrate — THEORY.md is honest about that); they exist to
// catch a bug that the fast algorithm and its line-by-line CPU twin could
// share, by re-deriving the same answer a structurally DIFFERENT way.
// ---------------------------------------------------------------------------

// articulation_bruteforce_cpu — for each module m of ONE configuration,
// physically remove it (skip its cell) and BFS the remaining kM-1 modules
// from any other module; m is an articulation point iff that BFS does not
// reach all kM-1 survivors. O(kM^2) per configuration — the textbook
// "remove and recheck" oracle, independent of Tarjan low-link entirely.
void articulation_bruteforce_cpu(const int32_t* pos_one_config,
                                 uint8_t* is_articulation_out);

// move_precondition_bruteforce_cpu — for ONE configuration, re-derive
// legality for every (module, direction) pair by exhaustively rebuilding
// an explicit occupied-cell list first (rather than querying positions
// on the fly, as the fast path does) and testing each precondition
// against that list — a deliberately differently-shaped implementation of
// the same rules (README/THEORY diagram the rules; both implementations
// must reach the same verdict on every one of kM*kNumMoveDirs entries).
void move_precondition_bruteforce_cpu(const int32_t* pos_one_config,
                                      const uint8_t* is_articulation_one_config,
                                      uint8_t* legal_move_out);

#endif // PROJECT_KERNELS_CUH
