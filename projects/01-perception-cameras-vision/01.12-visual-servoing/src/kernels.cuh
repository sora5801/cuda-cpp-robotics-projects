// ===========================================================================
// kernels.cuh — interface & shared contract for project 01.12
//               Visual servoing: image-Jacobian control loop entirely on GPU
//               (teaching core: eye-in-hand IBVS, batched convergence-basin study)
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (the driver), kernels.cu (the GPU rollout
// farm), and reference_cpu.cpp (the independent CPU oracle). Everything all
// three must agree on — target geometry, goal pose, controller constants,
// the pose/output layouts, and the cohort/variant enumerations — is defined
// HERE, once (CLAUDE.md §12). This project is a CONTROL project (a closed-
// loop feedback law) living in the vision domain; its nearest kin is
// project 08.01 (MPPI) — read that project's kernels.cuh/kernels.cu first if
// this is your entry point into the repo. The 6x6 damped normal-equations
// solve below is the same small-SPD-solve idiom taught in 33.01 (batched
// Cholesky). Quaternion order and the T_parent_child transform-naming
// convention follow 09.01 / docs/SYSTEM_DESIGN.md §interface conventions.
//
// IBVS in six lines (THEORY.md derives every step properly):
//   1. A camera rigidly attached to the end effector ("eye-in-hand") sees a
//      target of 4 known 3-D points; project them to normalized image
//      coordinates s = (x0,y0,...,x3,y3) in R^8 — the FEATURES.
//   2. A goal camera pose defines the DESIRED features s*.
//   3. The image Jacobian (interaction matrix) L(s,Z) linearly relates the
//      camera's body-frame twist v_c in R^6 to the features' rate of
//      change: ṡ = L(s,Z) v_c.
//   4. Drive the error e = s - s* to zero with the damped-least-squares
//      control law v_c = -λ · L̂⁺ · e, L̂⁺ solved via the 6x6 normal
//      equations (L̂ᵀL̂ + μI) x = L̂ᵀe  (Levenberg-damped Gauss-Newton).
//   5. Integrate the camera pose forward by v_c for one control tick.
//   6. Repeat until ‖e‖ is small (converged) or the step budget is spent.
// Step 5-6 repeated for T steps is ONE "rollout" of a closed LOOP (not an
// open-loop trajectory like 08.01's MPPI candidates) — K = 4096 such loops,
// from randomized/designed initial camera poses, are fully independent, so
// one GPU thread per loop is the natural mapping: the 08.01 rollout-farm
// idiom applied to a whole closed-loop controller instead of a single
// open-loop simulation. This is the project's GPU angle: a BATCHED
// CONVERGENCE-BASIN STUDY of a classic controller, not a novel kernel.
//
// FRAME CONVENTIONS (a deliberate, documented deviation from the repo
// default — CLAUDE.md §12 requires stating any deviation explicitly):
//   The repo default body convention is x-forward/y-left/z-up. Camera
//   geometry is instead taught here in the standard MACHINE-VISION OPTICAL
//   frame — x-right, y-down, z-forward (along the optical axis) — because
//   that is the frame in which the classical interaction-matrix derivation
//   (THEORY.md §the-math) is written in every reference (Chaumette &
//   Hutchinson 2006; Corke's "Robotics, Vision and Control"); translating
//   the derivation into the repo's default frame would only relabel axes,
//   not teach anything new, at the cost of not matching the literature.
//   The WORLD frame is defined to share this same axis convention at the
//   GOAL pose (see kGoalStandoff below) — i.e. the goal camera orientation
//   is the identity quaternion — which keeps the demo's linear algebra
//   focused on the servoing math rather than on a fixed extrinsic-
//   calibration offset. README §Limitations states this simplification.
//
// CAMERA POSE LAYOUT — float pose[7], SI units:
//     pose[0..2] = p        camera position in the WORLD frame (m):
//                            T_world_cam's translation part
//     pose[3..6] = (w,x,y,z) camera orientation quaternion — REPO ORDER,
//                            kept normalized (CLAUDE.md §12): T_world_cam's
//                            rotation part, i.e. rotate_by_quat(q, v)
//                            takes a vector expressed in the CAMERA frame
//                            and expresses it in the WORLD frame.
// A camera-frame point is therefore   P_cam = R_wcᵀ · (P_world − p)   —
// world-to-camera uses the quaternion CONJUGATE (world "child" -> camera
// "parent" is the inverse of T_world_cam; see kernels.cu rotate_by_conj_quat).
//
// CAMERA TWIST v_c = (vx,vy,vz, wx,wy,wz), SI units (m/s, rad/s), expressed
// in the CAMERA's OWN (moving, body) frame — the classical eye-in-hand
// convention and the frame the interaction matrix is derived in.
//
// TARGET — 4 coplanar 3-D points (a documented square, "fiducial-marker"
// sized) fixed in the WORLD frame, lying in the world Z=0 plane, centered
// at the world origin, facing +Z:
//     P0 = (-a,-a, 0)   P1 = (+a,-a, 0)   P2 = (+a,+a, 0)   P3 = (-a,+a, 0)
// with a = kTargetHalfSize. This is the industrial-norm "fiducial-based
// servoing" setup named in README §System context (upstream of this
// project: 01.04/01.06 feature/fiducial detection would produce these 4
// image points on a real robot; here they are simulated directly so the
// project can focus on the CONTROL LAW, not detection).
//
// GOAL POSE — the camera sits fronto-parallel, kGoalStandoff meters back
// along world +Z's negative side, looking straight at the target:
//     p_goal = (0, 0, -kGoalStandoff),  q_goal = (1,0,0,0)  (identity)
// s* (the desired features) is the deterministic projection of the 4
// target points from this pose — computed once by build_target_and_goal_cpu
// (shared setup, not the algorithmic core; see reference_cpu.cpp header for
// the twin-vs-shared ruling this project follows) and uploaded to GPU
// __constant__ memory once via set_target_and_goal (mirrors 09.01's
// set_robot_model: every thread reads the SAME 12+8 floats every step, so
// constant memory's broadcast-to-a-warp behavior is the textbook fit).
//
// OUTPUT LAYOUT — SIX parallel float[K] arrays (Structure-of-Arrays, one
// scalar per loop, so each is one coalesced array — no interleaving cost):
//     out_converged[k]  1.0f if ‖e‖ < kConvergeEps within kMaxSteps, else 0.0f
//     out_steps[k]      steps actually simulated before break (<= kMaxSteps)
//     out_final_err[k]  L2 norm of the FINAL feature error (8-vector)
//     out_cond_min[k]   worst (smallest) Cholesky-diagonal-ratio proxy
//                       encountered along the loop — see THEORY.md
//                       §numerics; NOT a true condition number, a cheap
//                       proxy computed from work already being done.
//     out_zmax[k]       max TRUE camera depth (m) reached — the retreat-
//                       pathology signal, tracked regardless of which
//                       variant's (possibly wrong) depth fed the Jacobian.
//     out_featmax[k]    max |normalized coordinate| reached across all 4
//                       points' x and y — the feature-excursion signal.
//
// TRACE LAYOUT — a small documented subset of loops (kTraceCount, indices
// chosen at runtime from the cohort boundaries — see main.cu) get their
// full per-step state logged for the plottable artifacts. Buffer:
//     out_trace[slot*(kMaxSteps+1)*kTraceRowStride + t*kTraceRowStride + f]
// row f = [ t_step, p.x, p.y, p.z, x0,y0, x1,y1, x2,y2, x3,y3 ]  (12 floats)
// A loop that converges at step t writes rows 0..t inclusive and leaves the
// rest of its slot at the caller's cudaMemset(0) value — the row count to
// actually read back is out_steps[trace_idx[slot]] (+1 if converged).
//
// COHORTS — the INITIAL poses are not uniform-random over one big box; they
// are drawn from three DESIGNED cohorts occupying disjoint index ranges of
// the K loops (documented precisely, and implemented ONCE — see the
// generate_*_cpu functions below, "data", not "algorithm", per the
// twin-vs-shared ruling in reference_cpu.cpp):
//     [0, n_nominal)                    NOMINAL — the everyday convergence-
//                                        basin region: random position
//                                        offset (each axis in
//                                        [-kNominalPosRange,+kNominalPosRange]
//                                        m) + random-axis rotation up to
//                                        kNominalAngleMaxDeg degrees.
//     [n_nominal, n_nominal+n_decay)    DECAY — small PURE-TRANSLATION
//                                        offsets (magnitude
//                                        [kDecayPosMin,kDecayPosMax] m,
//                                        random direction, zero rotation) —
//                                        the exponential_decay gate's cohort.
//     [n_nominal+n_decay, K)            RETREAT — near-180 DEGREE rotation
//                                        about the camera's own optical
//                                        (Z) axis, near-zero position
//                                        offset — the classic IBVS "camera
//                                        retreat" pathology cohort
//                                        (THEORY.md derives WHY geometrically).
// n_decay = round(K * kFracDecay), n_retreat = round(K * kFracRetreat),
// n_nominal = K - n_decay - n_retreat (computed once in main.cu, passed to
// both the host pose generator and used to slice per-cohort gate metrics).
//
// CONTROLLER VARIANTS — three run over the SAME initial-pose batch (the
// didactic comparison), selected by which (x,y,Z) triple feeds the
// per-point interaction-matrix ROW formula (the feature ERROR always uses
// the true current features — only the JACOBIAN's depth/point assumption
// changes):
//   kVariantTrueDepth      — L built from the CURRENT (x,y) and the EXACT
//                            current depth Z (ground truth available only
//                            in simulation) — the reference upper bound.
//   kVariantFixedDepth     — L built from the CURRENT (x,y) but a CONSTANT
//                            Z = kGoalStandoff (the classic practical
//                            approximation: real systems rarely have
//                            per-point depth without stereo/RGBD).
//   kVariantDesiredJacobian— L built ENTIRELY from the DESIRED features s*
//                            and Z = kGoalStandoff — i.e. the textbook
//                            "constant Jacobian" scheme L(s*,Z*), literally
//                            invariant over the whole loop. This
//                            implementation deliberately recomputes it every
//                            step anyway (see kernels.cu) to keep ONE
//                            uniform per-step code path across all three
//                            variants — memoizing it is README Exercise.
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cstdint>   // uint32_t — the xorshift32 RNG state type

// ---------------------------------------------------------------------------
// Problem-size constants (state-vector / feature-vector dimensions).
// ---------------------------------------------------------------------------
constexpr int kNumPoints = 4;                    // coplanar target points
constexpr int kFeatDim   = 2 * kNumPoints;        // 8: (x,y) per point
constexpr int kNV        = 6;                    // camera twist DOF
constexpr int kPoseDim   = 7;                    // p(3) + quaternion(4)

// ---------------------------------------------------------------------------
// Target geometry & goal pose (SI units; see the file header for the frame
// convention and the closed-form derivation of s*).
// ---------------------------------------------------------------------------
constexpr float kTargetHalfSize = 0.06f;   // m — target square half-side (12 cm square, fiducial-sized)
constexpr float kGoalStandoff   = 0.5f;    // m — goal camera distance along Z from the target plane

// ---------------------------------------------------------------------------
// Controller constants (the taught, tuned teaching setup — CLAUDE.md §12
// "state vectors/weights are single-sourced"; tuning story in THEORY.md).
// ---------------------------------------------------------------------------
constexpr float kLambda       = 2.0f;     // servo gain (1/s): sets the target exponential time-constant 1/λ
constexpr float kDampingMu    = 0.05f;    // Levenberg damping added to LᵀL before the 6x6 solve (guards near-singular L)
constexpr float kDt           = 0.01f;    // control period (s) -> 100 Hz simulated control loop
constexpr int   kMaxSteps     = 400;      // per-loop step budget (4.0 s of simulated time)
constexpr float kConvergeEps  = 0.006f;   // L2 norm of the 8-vector feature error counted as "converged"

// ---------------------------------------------------------------------------
// Batch / cohort constants.
// ---------------------------------------------------------------------------
constexpr int   kDefaultK       = 4096;   // loops per batch (one thread per loop)
constexpr int   kDefaultBasinG  = 64;     // basin-map grid side (G*G = 4096 grid loops)

constexpr float kFracDecay      = 1.0f / 16.0f;  // fraction of K in the DECAY cohort
constexpr float kFracRetreat    = 1.0f / 16.0f;  // fraction of K in the RETREAT cohort
                                                  // (remainder is NOMINAL)

constexpr float kNominalPosRange    = 0.15f;   // m — nominal cohort: dx,dy,dz each in [-range,+range]
constexpr float kNominalAngleMaxDeg = 15.0f;   // deg — nominal cohort: rotation angle in [0, max]

constexpr float kDecayPosMin = 0.02f;   // m — decay cohort: pure-translation offset magnitude, min
constexpr float kDecayPosMax = 0.05f;   // m — decay cohort: pure-translation offset magnitude, max

constexpr float kRetreatAngleMinDeg = 150.0f;  // deg — retreat cohort: rotation angle about optical axis, min
constexpr float kRetreatAngleMaxDeg = 180.0f;  // deg — retreat cohort: rotation angle about optical axis, max
constexpr float kRetreatPosJitter   = 0.01f;   // m — retreat cohort: small position jitter per axis

constexpr float kBasinPosRange = 0.30f;    // m — basin-map grid spans [-range,+range] in dx,dy at dz=0, no rotation

// Retreat-pathology detection: "camera physically retreated" if the true
// depth ever exceeds this multiple of the goal standoff.
constexpr float kRetreatZMultiple = 3.0f;

// ---------------------------------------------------------------------------
// Controller variant selector (see file header for the didactic meaning of
// each). Passed as a plain int to keep the launcher signature C-compatible
// across the nvcc/cl.exe boundary (an enum class would work too, but a
// plain int with named constants is the simplest thing that is still safe).
// ---------------------------------------------------------------------------
constexpr int kVariantTrueDepth       = 0;
constexpr int kVariantFixedDepth      = 1;
constexpr int kVariantDesiredJacobian = 2;
constexpr int kVariantCount           = 3;

// Human-readable names for report lines — single-sourced so GPU-run and
// CPU-twin reports never spell a variant two different ways.
inline const char* variant_name(int variant)
{
    switch (variant) {
        case kVariantTrueDepth:       return "true-depth";
        case kVariantFixedDepth:      return "fixed-depth";
        case kVariantDesiredJacobian: return "desired-jacobian";
        default:                      return "unknown-variant";
    }
}

// Trace-row layout (see file header "TRACE LAYOUT").
constexpr int kTraceRowStride = 1 /*t*/ + 3 /*p*/ + kFeatDim /*features*/;  // 12
constexpr int kTraceCount     = 8;   // the "small documented subset" traced for artifacts

// ---------------------------------------------------------------------------
// set_target_and_goal — upload the target's 4 world-frame points and the
// desired feature vector s* to GPU __constant__ memory (kernels.cu owns the
// __constant__ symbols; this is the setter, mirroring 09.01's
// set_robot_model — every thread of every loop reads the SAME 20 floats
// every step, the textbook constant-memory broadcast use case).
//
//   target_pts_world : HOST pointer, 12 floats = 4 points * (X,Y,Z), m.
//   s_star           : HOST pointer, 8 floats = 4 points * (x,y), normalized.
//
// Must be called once before any launch_ibvs_* call. Cheap (80 bytes).
// ---------------------------------------------------------------------------
void set_target_and_goal(const float target_pts_world[12], const float s_star[8]);

// ---------------------------------------------------------------------------
// launch_ibvs_batch — GPU: simulate K independent closed IBVS loops.
//
//   K             : number of loops (>= 1).
//   variant        : one of kVariant{TrueDepth,FixedDepth,DesiredJacobian}.
//   d_init_poses   : DEVICE pointer, K*kPoseDim floats — each loop's STARTING
//                    camera pose (layout above). Generated on the HOST by
//                    generate_batch_init_poses_cpu / generate_basin_grid_poses_cpu
//                    (deterministic "data", shared by both the GPU and CPU
//                    paths — see reference_cpu.cpp's twin-vs-shared ruling).
//   d_trace_idx    : DEVICE pointer, trace_count ints — loop indices to log
//                    full per-step trajectories for (may be nullptr if
//                    trace_count == 0).
//   trace_count    : number of traced loops (0..kTraceCount typically).
//   d_out_*        : DEVICE pointers, K floats each OUT — see "OUTPUT LAYOUT"
//                    in the file header.
//   d_out_trace    : DEVICE pointer, trace_count*(kMaxSteps+1)*kTraceRowStride
//                    floats OUT — see "TRACE LAYOUT". Caller must
//                    cudaMemset this to 0 before the call (rows past a
//                    loop's convergence step are intentionally left
//                    untouched, and read back as exactly zero).
//
// Launch: one THREAD per loop, 256-thread blocks, ceil(K/256) blocks — the
// same thread-per-problem pattern as 08.01/33.01/09.01. Reasoning with the
// kernel in kernels.cu.
// ---------------------------------------------------------------------------
void launch_ibvs_batch(int K, int variant, const float* d_init_poses,
                       const int* d_trace_idx, int trace_count,
                       float* d_out_converged, float* d_out_steps,
                       float* d_out_final_err, float* d_out_cond_min,
                       float* d_out_zmax, float* d_out_featmax,
                       float* d_out_trace);

// ---------------------------------------------------------------------------
// launch_ibvs_single_step — GPU: evaluate ONE control step from each of
// `count` given poses (no time integration). This is the project's
// "Jacobian entries + pseudoinverse solve" VERIFICATION path — it exposes
// the intermediate linear algebra (the assembled normal matrix A, the
// right-hand side b, and the resulting twist v) that a full-loop comparison
// would hide inside 400 compounding steps (CLAUDE.md §5 GPU-vs-CPU gate,
// applied at the finest useful grain).
//
//   count       : number of sampled poses (>= 1).
//   d_poses7    : DEVICE pointer, count*kPoseDim floats — the sampled poses.
//   variant     : which controller variant's Jacobian rule to evaluate.
//   d_out_v     : DEVICE pointer, count*kNV floats OUT — the computed twist.
//   d_out_A     : DEVICE pointer, count*kNV*kNV floats OUT — the damped
//                 normal matrix A = LᵀL + μI, row-major 6x6, PRE-factorization.
//   d_out_b     : DEVICE pointer, count*kNV floats OUT — the right-hand side b = Lᵀe.
//   d_out_e     : DEVICE pointer, count*kFeatDim floats OUT — the feature error s-s*.
// ---------------------------------------------------------------------------
void launch_ibvs_single_step(int count, const float* d_poses7, int variant,
                             float* d_out_v, float* d_out_A,
                             float* d_out_b, float* d_out_e);

// ===========================================================================
// CPU references (reference_cpu.cpp) — see that file's header for the
// twin-vs-shared independence ruling this project follows.
// ===========================================================================

// build_target_and_goal_cpu — deterministic SETUP (not the algorithmic
// core; single-sourced "data" per the ruling): derive the 4 world-frame
// target points and the desired feature vector s* from the compile-time
// constants above. Used by main.cu both to upload via set_target_and_goal
// AND to feed the CPU oracle below — the two paths must see the identical
// target/goal, so this is deliberately not duplicated.
void build_target_and_goal_cpu(float target_pts_world[12], float s_star[8]);

// generate_batch_init_poses_cpu — deterministic SETUP (single-sourced
// "data" — see file header "COHORTS"): fill K initial poses across the
// three designed cohorts. seed is the xorshift32 base seed (documented
// per-loop mixing formula lives in reference_cpu.cpp next to the
// implementation). poses7 is a HOST pointer, K*kPoseDim floats OUT.
void generate_batch_init_poses_cpu(int K, int n_nominal, int n_decay,
                                   uint32_t seed, float* poses7);

// generate_basin_grid_poses_cpu — deterministic SETUP (no RNG): fill G*G
// initial poses on a regular (dx,dy) grid at dz=0, zero rotation, spanning
// [-kBasinPosRange,+kBasinPosRange] — the basin_map.ppm artifact's sample
// points. poses7 is a HOST pointer, G*G*kPoseDim floats OUT.
void generate_basin_grid_poses_cpu(int G, float* poses7);

// ibvs_compute_step_cpu — the ORACLE twin of the GPU's per-step device
// function: one IBVS control step from `pose`, independently reimplemented
// in plain C++ (see reference_cpu.cpp). Exposes the same intermediate
// quantities as launch_ibvs_single_step for the tight Jacobian/pseudoinverse
// verification gate, and is also the building block ibvs_batch_cpu below
// uses internally for the full-loop oracle.
//
//   pose            : HOST pointer, kPoseDim floats — current camera pose.
//   variant         : which controller variant's Jacobian rule to evaluate.
//   target_pts_world: HOST pointer, 12 floats (CPU takes the model as an
//                     ordinary argument — no __constant__ memory on a CPU;
//                     that difference IS part of the lesson, per 09.01).
//   s_star          : HOST pointer, 8 floats.
//   v_out           : HOST pointer, 6 floats OUT — the computed twist.
//   A_out           : HOST pointer, 36 floats OUT (row-major 6x6), OR
//                     nullptr if the caller does not need it.
//   b_out           : HOST pointer, 6 floats OUT, OR nullptr.
//   feat_out        : HOST pointer, 8 floats OUT — current normalized features.
//   err_norm_out    : HOST pointer, 1 float OUT — L2 norm of (feat - s_star).
//   cond_proxy_out  : HOST pointer, 1 float OUT — this step's conditioning proxy.
//   zmax_out        : HOST pointer, 1 float OUT — max true depth this step (over the 4 points).
//   featmax_out     : HOST pointer, 1 float OUT — max |normalized coord| this step.
void ibvs_compute_step_cpu(const float pose[kPoseDim], int variant,
                           const float target_pts_world[12], const float s_star[8],
                           float v_out[kNV], float A_out[kNV * kNV], float b_out[kNV],
                           float feat_out[kFeatDim], float* err_norm_out,
                           float* cond_proxy_out, float* zmax_out, float* featmax_out);

// ibvs_batch_cpu — the ORACLE twin of launch_ibvs_batch: K independent
// closed loops, sequential, identical convergence/trace semantics. Used by
// main.cu for (a) the single-loop trajectory twin (K=1), (b) the batch
// statistics twin (a K=128 subset), both against the GPU's results on the
// SAME init_poses.
void ibvs_batch_cpu(int K, int variant, const float* init_poses,
                    const float target_pts_world[12], const float s_star[8],
                    const int* trace_idx, int trace_count,
                    float* out_converged, float* out_steps, float* out_final_err,
                    float* out_cond_min, float* out_zmax, float* out_featmax,
                    float* out_trace);

#endif // PROJECT_KERNELS_CUH
