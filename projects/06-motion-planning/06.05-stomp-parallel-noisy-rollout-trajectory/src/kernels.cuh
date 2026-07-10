// ===========================================================================
// kernels.cuh — interface for project 06.05
//               STOMP: parallel noisy-rollout trajectory optimization
//               (teaching core: 2-D point robot through an obstacle field)
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (the closed optimization loop + M-matrix setup),
// kernels.cu (the GPU scoring kernel), and reference_cpu.cpp (the scoring
// oracle + the single-path evaluator). Everything all three must agree on —
// the trajectory representation, the cost function, the grid/field convention,
// and the noise-array layout — is defined HERE, once (CLAUDE.md §12).
//
// STOMP in five lines (THEORY.md derives it properly)
// ---------------------------------------------------
//   1. Keep a nominal trajectory: N interior waypoints between a fixed start
//      and a fixed goal (2-D positions in metres).
//   2. Sample K noisy trajectories theta~_k = theta + eps_k, where eps_k is
//      SMOOTH noise eps = M z (z = per-waypoint white noise; M derived from
//      R^-1, R = A^T A the finite-difference acceleration matrix). The smooth
//      construction is what makes STOMP STOMP — see main.cu build_M().
//   3. Score each noisy trajectory on the GPU — K independent rollouts — by
//      integrating an obstacle-cost field along its segments. One GPU thread
//      per rollout (K ~ thousands; the catalog calls STOMP "born for GPU").
//   4. UPDATE PER WAYPOINT: at each waypoint index j, softmin-weight the K
//      noisy perturbations by their LOCAL cost at j and blend; smooth the
//      whole update through M. (This per-waypoint weighting is STOMP's
//      signature — contrast MPPI's single per-WHOLE-TRAJECTORY softmin, 08.01.)
//   5. Iterate until the trajectory stops improving. Endpoints never move.
// Step 3 is the embarrassingly-parallel part — one thread per noisy rollout,
// the SAME mapping 08.01 uses for MPPI rollouts.
//
// The problem: a POINT ROBOT in a 2-D plane must travel from start to goal
// without hitting obstacles, along a short, smooth path. It is the classic
// teaching instance of trajectory optimization: rich enough to be genuinely
// non-convex (obstacles carve the cost landscape into basins), small enough
// to fit on one screen. A real arm/AMR planner is this same math in more
// dimensions over a distance field from the map (README §System context).
//
// TRAJECTORY LAYOUT — documented once here:
//   * kN interior waypoints, each a 2-D point. Stored as two parallel arrays
//     theta_x[kN], theta_y[kN] (structure-of-arrays: adjacent waypoints are
//     adjacent in memory, and x/y kept apart so each is a clean coalesced
//     stream on the GPU).
//   * The FULL path used for cost evaluation has kN+2 points:
//         P[0]      = start   (FIXED — never perturbed, never updated)
//         P[1..kN]  = interior waypoints  (P[m] = theta[m-1])
//         P[kN+1]   = goal    (FIXED)
//     So there are kN+1 segments. main.cu writes P[0..kN+1] to the artifact.
//   * Frame: world frame, right-handed, +x right / +y up, origin at the
//     map's lower-left corner. SI units (metres). Positions are expressed in
//     the world frame throughout; there is no moving frame here, so the
//     T_parent_child convention (CLAUDE.md §12) is trivially T_world_point.
//
// COST-FIELD / GRID convention — documented once here:
//   The obstacle-cost field is a dense gw x gh grid of FP32 costs, built on
//   the HOST at load time from the committed obstacle spec (main.cu
//   build_cost_field). Cell (ix, iy) covers world position
//       x = ix * cell_m,   y = iy * cell_m      (cell centre convention: the
//   sampler treats grid coordinate g = x / cell_m and bilinearly interpolates
//   between the four surrounding cells). Row-major: field[iy * gw + ix].
//   Cost is >= 0: ~0 in free space, rising smoothly inside an inflation band
//   around each obstacle, high inside obstacles (see build_cost_field).
//
// NOISE LAYOUT — eps is stored TRANSPOSED: eps[j*K + k], not eps[k*N + j].
//   Why: when the kernel processes waypoint index j, every thread k reads its
//   own eps[j][k]. With the transposed layout a warp's 32 reads are
//   CONSECUTIVE floats (perfectly coalesced); the "natural" per-rollout layout
//   eps[k*N + j] would stride reads N floats apart and waste ~90% of every
//   memory transaction. This is the exact same layout lesson 08.01 applies to
//   its control-noise array. There are two such arrays, epsx and epsy (one per
//   spatial dimension), each kN*K floats.
//
// Noise is generated ON THE HOST with the repo's portable xorshift32 and the
// smooth mixing eps = M z, then uploaded per iteration. Production STOMP/GPU
// planners generate noise on-device (cuRAND) to avoid the upload; we trade
// that for bit-reproducible demos — the trade is documented, not hidden
// (THEORY.md §numerics; on-device noise is README Exercise 4).
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

// ---------------------------------------------------------------------------
// Trajectory dimensions (shared verbatim by the GPU kernel, the CPU oracle,
// and the host optimizer — a mismatch here would silently plan the wrong
// trajectory).
// ---------------------------------------------------------------------------
constexpr int kN   = 64;   // number of interior (free) waypoints — the DOF STOMP optimizes
constexpr int kDim = 2;    // spatial dimensions (x, y) — a planar point robot
constexpr int kPathPoints = kN + 2;  // full path incl. fixed start + goal (66)

// ---------------------------------------------------------------------------
// STOMP hyperparameters (defaults; the scenario file supplies the geometry —
// map size, start, goal, obstacles — but these tuned knobs are part of the
// taught setup, so they are compile-time constants, not data).
// ---------------------------------------------------------------------------
constexpr int   kDefaultK   = 1024;  // noisy rollouts per iteration (real STOMP uses ~5-20;
                                     // we use far more because we CAN — the GPU makes it cheap,
                                     // and more samples give a smoother per-waypoint estimate)
constexpr int   kMaxIters   = 100;   // hard cap on optimization iterations (plateau stops earlier)
constexpr float kNoiseSigma = 4.0f;  // white-noise std-dev fed into eps = M z (tuned: THEORY §numerics)
constexpr float kSensitivity = 10.0f;// softmin sensitivity h in the per-waypoint weights (STOMP paper: 10)

// ---------------------------------------------------------------------------
// Cost-field shaping (the obstacle cost as a function of signed distance d to
// the nearest obstacle boundary, in metres; d < 0 inside an obstacle). These
// define build_cost_field() AND how "collision-free with margin" is judged.
//   d <= 0        : kCostCollision + penetration ramp        (inside — very costly)
//   0 < d < kInfl : kCostCollision * ((kInfl - d)/kInfl)^2   (smooth quadratic halo → 0 at kInfl)
//   d >= kInfl    : 0                                        (free space)
// The quadratic halo gives the field a gradient that pushes waypoints away
// from obstacles even before they touch — the same idea as a ROS costmap
// inflation layer or CHOMP's smooth obstacle cost.
// ---------------------------------------------------------------------------
constexpr float kInfl          = 0.60f;  // inflation radius (m): how far the cost halo reaches
constexpr float kCostCollision = 100.0f; // cost at an obstacle boundary (peak of the halo)
constexpr float kPenetration   = 50.0f;  // extra cost per metre INSIDE an obstacle (gradient out)

// ---------------------------------------------------------------------------
// Cost weights and sampling densities.
//   kWSmooth : weight on the sum-of-squared-accelerations smoothness term.
//              Kept small so the OBSTACLE term dominates the cost-reduction
//              story (STOMP's structural M-smoothing already keeps paths
//              smooth); tuned in THEORY §algorithm.
//   kSegSamples  : sub-samples per segment the SCORING path uses (kernel +
//                  oracle). Dense enough to catch a thin obstacle between two
//                  waypoints; both paths MUST use the same count so the
//                  GPU-vs-CPU verify is apples-to-apples.
//   kCheckSamples: denser sub-sampling the single-path EVALUATOR uses for the
//                  final collision verdict — strict about thin obstacles.
// ---------------------------------------------------------------------------
constexpr float kWSmooth      = 2.0f;
constexpr int   kSegSamples   = 8;   // scoring: samples along each segment (kernel + oracle)
constexpr int   kCheckSamples = 32;  // verdict: samples along each segment (evaluator only)

// ---------------------------------------------------------------------------
// Grid resolution of the cost field / the PGM artifact. Fixed (not from data)
// so the artifact dimensions are stable. 256x256 over a ~10 m map is ~0.04 m
// per cell — fine enough to resolve the sub-metre obstacles.
// ---------------------------------------------------------------------------
constexpr int kGridW = 256;
constexpr int kGridH = 256;

// ---------------------------------------------------------------------------
// launch_stomp_score — score all K noisy trajectories on the GPU.
//
//   K        : number of noisy rollouts (>= 1).
//   d_field  : DEVICE pointer, gw*gh floats — the obstacle-cost field
//              (row-major, cost >= 0; see the grid convention above).
//   gw, gh   : field dimensions (cells).
//   cell_m   : world size of one cell (m); world = grid_coord * cell_m.
//   start2   : HOST pointer, 2 floats {x, y} — the FIXED start (m). Uploaded
//              internally (it is 8 bytes; every thread reads the same value).
//   goal2    : HOST pointer, 2 floats {x, y} — the FIXED goal (m). Ditto.
//   d_theta_x/d_theta_y : DEVICE pointers, kN floats each — the current
//              nominal interior waypoints (uniform reads: same address, all
//              threads).
//   d_epsx/d_epsy : DEVICE pointers, kN*K floats each — the smooth noise,
//              TRANSPOSED layout eps[j*K + k] (see the header comment).
//   d_Sloc   : DEVICE pointer, kN*K floats OUT — per-waypoint LOCAL obstacle
//              cost, transposed [j*K + k]; consumed by the host per-waypoint
//              softmin update.
//   d_cost   : DEVICE pointer, K floats OUT — TOTAL trajectory cost per
//              rollout (obstacle path-integral + kWSmooth*smoothness); used by
//              the §5 GPU-vs-CPU verify gate.
//
// Launch: one thread per rollout, 256-thread blocks (grid math + reasoning
// live with the kernel in kernels.cu). The kernel is the project's hot loop.
// ---------------------------------------------------------------------------
void launch_stomp_score(int K,
                        const float* d_field, int gw, int gh, float cell_m,
                        const float* start2, const float* goal2,
                        const float* d_theta_x, const float* d_theta_y,
                        const float* d_epsx, const float* d_epsy,
                        float* d_Sloc, float* d_cost);

// ---------------------------------------------------------------------------
// CPU references (reference_cpu.cpp).
// ---------------------------------------------------------------------------

// stomp_rollouts_cpu — the ORACLE twin of the scoring kernel: same field
// sampler, same segment integral, same smoothness term, sequential over k.
// Fills cost[k] (the TOTAL per-trajectory cost) only — main.cu runs it against
// the GPU on iteration 0's inputs and requires agreement within a relative
// tolerance (the §5 GPU-vs-CPU gate for this project). All pointers are HOST
// pointers; it reads the SAME transposed noise layout eps[j*K + k].
void stomp_rollouts_cpu(int K,
                        const float* field, int gw, int gh, float cell_m,
                        const float* start2, const float* goal2,
                        const float* theta_x, const float* theta_y,
                        const float* epsx, const float* epsy,
                        float* cost);

// evaluate_path_cost — score ONE full path (P[0..npoints-1], including the
// fixed endpoints) on the host, for convergence monitoring and the final
// collision verdict. Uses the DENSE kCheckSamples sub-sampling. Returns the
// total cost (double, so long accumulations stay clean) and writes the maximum
// field value seen anywhere along the path into *out_max_field — the quantity
// the "collision-free with margin" check tests. Host pointers.
double evaluate_path_cost(const float* field, int gw, int gh, float cell_m,
                          const float* px, const float* py, int npoints,
                          float* out_max_field);

#endif // PROJECT_KERNELS_CUH
