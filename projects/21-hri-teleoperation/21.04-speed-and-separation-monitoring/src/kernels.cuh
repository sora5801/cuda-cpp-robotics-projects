// ===========================================================================
// kernels.cuh — the CONTRACT for project 21.04
//               Speed-and-separation monitoring: depth streams -> minimum-
//               distance fields at frame rate (ISO/TS 15066-style helper)
//
// >>> DIDACTIC IMPLEMENTATION -- NOT A CERTIFIED SAFETY FUNCTION. <<<
// This project computes metrics ADJACENT to ISO/TS 15066's speed-and-
// separation-monitoring concept, for teaching. It is not a certified
// implementation of any standard, is not safety-rated, and must never be
// used to guard a real robot. See README/THEORY/PRACTICE for the full
// caveat (CLAUDE.md §1, §8). The person in this scenario is modeled as an
// ANONYMOUS capsule pair (torso + reaching arm) -- there is no identity,
// tracking, or recognition anywhere in this project; the framing is
// collaborative safety (protecting a generic person near a robot), never
// surveillance of individuals.
//
// Role in the project
// --------------------
// Everything main.cu (orchestration), kernels.cu (GPU kernels), and
// reference_cpu.cpp (the independent CPU oracle) must agree on lives HERE,
// once (CLAUDE.md §12): the capsule representation, the depth-camera and
// cell geometry, the robot's forward-kinematics model constants, the human
// model constants, the ISO/TS-15066-STYLE separation-distance parameters,
// and the pixel-label / state-machine vocabulary.
//
// The pipeline in five lines (THEORY.md derives every step properly):
//   1. RENDER + CLASSIFY: an orthographic top-down depth camera looks down
//      on the cell; each pixel is labeled BACKGROUND / ROBOT / HUMAN by
//      comparing the sensed surface height against the robot's OWN known
//      pose (the "self-filter") and the empty-floor baseline.
//   2. RECONSTRUCT (fused into step 3): every HUMAN pixel's depth value
//      is turned into a 3-D point (x, y, z) in the cell frame.
//   3. MINIMUM-DISTANCE FIELD: each human point's distance to the nearest
//      ROBOT capsule is a point-capsule distance (derived in THEORY.md);
//      one thread evaluates one candidate pixel (map), and a block-level
//      shared-memory tree reduction finds the frame's d_min (reduce).
//   4. SSM DECISION: compare d_min against two ISO/TS-15066-STYLE
//      protective separation distances (S_p at full speed, S_p at reduced
//      speed) and drive a NORMAL / REDUCED / PROTECTIVE_STOP state machine
//      with documented hysteresis.
//   5. (Bonus, one frame): the SAME point-capsule distance function run
//      densely over every pixel gives the "clearance field" artifact the
//      catalog bullet names.
//
// Read this after: main.cu (orchestration + the analytic ground truth).
// Read this before: kernels.cu (the GPU kernels) and reference_cpu.cpp
// (their independent CPU oracle twins).
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cstdint>   // uint8_t pixel labels

// ===========================================================================
// SECTION 1 — geometry primitive: the capsule.
//
// A capsule is the set of points within `radius` of the 3-D segment
// A->B (a "swept sphere"). Every solid body in this project -- every robot
// link and the human's torso/arm -- is one capsule. Capsules are cheap to
// test (a handful of FLOPs) and, swept along a segment, approximate rounded
// link/limb geometry far better than a single sphere or a box, which is why
// they are the standard collision/clearance primitive in robotics (Drake,
// cuRobo, MoveIt's FCL backend all use them for exactly this reason).
//
// SCOPING CHOICE (load-bearing, see THEORY.md "The GPU mapping" and
// "Numerical considerations"): every capsule authored by this project's
// scene generator (main.cu's build_scene()) has an axis that is EITHER
//   * perfectly HORIZONTAL (az == bz, arbitrary x/y), or
//   * perfectly VERTICAL   (ax == bx AND ay == by, arbitrary z),
// tagged explicitly by `kind` below (never inferred by comparing floats for
// equality -- that would be fragile). This constraint makes the top-down
// depth RENDERING of a capsule's silhouette height EXACT in closed form
// (see capsule_top_at() in kernels.cu / reference_cpu.cpp) -- no iterative
// solve, no linearization error. A capsule with a tilted axis would need a
// harder 1-D optimization to find its rendered top height; THEORY.md
// derives why and treats a general-orientation renderer as exercise
// territory. This is a deliberate, DOCUMENTED reduced-scope choice
// (CLAUDE.md §13), not a hidden shortcut -- the SCARA-style arm and the
// human model below are built entirely from horizontal/vertical links,
// which is also a physically ordinary choice (SCARA arms literally are
// vertical-axis-jointed machines; an upright person is well approximated
// by a vertical torso capsule).
// ===========================================================================
struct Capsule {
    float ax, ay, az;   // endpoint A (m, cell frame -- see SECTION 2)
    float bx, by, bz;   // endpoint B (m). For VERTICAL capsules, B is
                        // ALWAYS the higher (larger-z) endpoint -- a
                        // convention every capsule in build_scene() obeys,
                        // so capsule_top_at() never has to branch on which
                        // endpoint is "up".
    float radius;       // m, > 0
    int   kind;          // 0 = HORIZONTAL (az == bz exactly, by construction)
                        // 1 = VERTICAL   (ax == bx && ay == by, by construction)
};

// ===========================================================================
// SECTION 2 — cell, camera, and depth-image geometry.
//
// The "cell" is a flat 4 m x 4 m floor patch (a small collaborative work
// cell footprint -- SYSTEM_DESIGN.md's manipulator work-cell reference
// robot). A single synthetic overhead depth camera looks straight down
// (orthographic projection, not perspective -- see THEORY.md for exactly
// what that simplifies and why it is honest for a teaching demo) and
// reports, at each pixel, the height of the highest surface below it.
// ===========================================================================
constexpr float kCellMinX = -2.0f;   // m, cell frame x lower bound
constexpr float kCellMaxX =  2.0f;   // m, cell frame x upper bound
constexpr float kCellMinY = -2.0f;   // m, cell frame y lower bound
constexpr float kCellMaxY =  2.0f;   // m, cell frame y upper bound

constexpr int   kImageW = 200;       // pixels, x direction
constexpr int   kImageH = 200;       // pixels, y direction
constexpr int   kNumPixels = kImageW * kImageH;   // 40,000

// Pixel pitch: (4.0 m) / 200 px = 0.02 m/px = 2 cm/px in both axes -- the
// SINGLE number that sets this pipeline's spatial quantization error (see
// kPixelQuantBound below and THEORY.md "Numerical considerations").
constexpr float kPixelSizeX = (kCellMaxX - kCellMinX) / static_cast<float>(kImageW);
constexpr float kPixelSizeY = (kCellMaxY - kCellMinY) / static_cast<float>(kImageH);

constexpr float kCamHeight = 3.0f;   // m, overhead camera mount height above
                                     // the floor (z=0) -- above the tallest
                                     // object in the scene (human head at
                                     // 1.5 m) with headroom, matching a
                                     // typical ceiling-truss SSM camera mount.

// Classification epsilons (m). kFloorEps separates "nothing there" from
// "something there" against the perfectly flat, perfectly known floor (a
// simplification -- THEORY.md discusses real sensor noise). kSelfFilterEps
// is the tolerance band for "this pixel's height matches what the robot's
// OWN known pose predicts" -- real self-filters need a nonzero band because
// encoder/calibration uncertainty means the commanded pose and the true
// pose never match exactly (PRACTICE.md §1 discusses the physical source).
constexpr float kFloorEps      = 0.01f;   // 1 cm: floor/background threshold
constexpr float kSelfFilterEps = 0.015f;  // 1.5 cm: robot self-filter band

// Pixel-quantization bound (m): the worst-case distance between a pixel's
// SAMPLED (x,y) center and the true continuous point on a surface nearest
// to it is half the pixel diagonal. THEORY.md proves this is also the
// worst-case OVERESTIMATE the pixel pipeline's d_min can have relative to
// the true (closed-form) minimum distance -- the central numerical result
// this project's verification gates rely on.
constexpr float kPixelQuantBound =
    0.70710678f * (kPixelSizeX);   // sqrt(2)/2 * pixel size (square pixels: kPixelSizeX == kPixelSizeY)

// ===========================================================================
// SECTION 3 — robot model: a SCARA-style 3-joint arm, 8 teaching capsules.
//
// All three joints are VERTICAL-AXIS (yaw) revolutes -- the defining trait
// of a real SCARA (Selective Compliance Assembly Robot Arm) architecture,
// not an invented simplification (THEORY.md "Where this sits in the real
// world" names real SCARA products). Each LINK is a horizontal capsule at
// its own fixed height; each JOINT is a small vertical "hub" capsule that
// steps the height down by a few cm between links -- a simplified version
// of the real mechanical reason SCARA links sit at slightly different
// heights (so rotating links never collide with each other). This choice
// is what keeps EVERY capsule in the arm horizontal or vertical (SECTION 1)
// while still producing a genuine, non-trivial 3-D silhouette.
//
// Forward kinematics (derived, not guessed -- THEORY.md "The math"):
//   reach_fraction(t) = 0.5*(1 - cos(2*pi*t / kSequenceDurationS))   in [0,1],
//     0 at t=0 and t=kSequenceDurationS, 1 at the midpoint -- a smooth
//     "there and back" profile shared (by deliberate scenario design, see
//     main.cu build_scene()) with the human's approach-retreat profile, so
//     both actors reach their closest configuration at the same instant --
//     a single, unambiguous closest-approach event for this teaching demo.
//   th1 = kJoint1MeanDeg + kJoint1AmpDeg * reach_fraction(t)      (shoulder yaw)
//   th2 = kJoint2MeanDeg + kJoint2AmpDeg * reach_fraction(t)      (elbow yaw, relative)
//   th3 = kJoint3MeanDeg + kJoint3AmpDeg * reach_fraction(t)      (wrist yaw, relative)
//   elbow  = shoulder + L1 * (cos th1,       sin th1)
//   wrist  = elbow    + L2 * (cos(th1+th2),  sin(th1+th2))
//   tool   = wrist    + L3 * (cos(th1+th2+th3), sin(th1+th2+th3))
// (all angles in radians when used; *Deg constants below are degrees for
// readability and converted once in main.cu's build_scene()).
// ===========================================================================
constexpr float kL1 = 0.50f;         // m, upper-arm reach (shoulder->elbow)
constexpr float kL2 = 0.45f;         // m, forearm reach (elbow->wrist)
constexpr float kL3 = 0.25f;         // m, tool reach (wrist->tool point)

constexpr float kZShoulder = 1.10f;  // m, upper-arm link height
constexpr float kZForearm  = 1.00f;  // m, forearm link height (10 cm lower)
constexpr float kZWrist    = 0.95f;  // m, tool link height (5 cm lower again)

constexpr float kJoint1MeanDeg =  10.0f, kJoint1AmpDeg = 20.0f;  // shoulder yaw
constexpr float kJoint2MeanDeg = -20.0f, kJoint2AmpDeg = 20.0f;  // elbow yaw (relative)
constexpr float kJoint3MeanDeg =  10.0f, kJoint3AmpDeg = 15.0f;  // wrist yaw (relative)

constexpr int kNumRobotCapsules = 8;   // "~8" per the catalog bullet
// Capsule radii (m), one per named link/hub -- indices match
// kRobotCapsuleNames below and main.cu's build_scene() emission order.
constexpr float kRBaseColumn  = 0.12f;  // fixed pedestal, floor to shoulder
constexpr float kRShoulderHub = 0.10f;  // shoulder joint housing
constexpr float kRUpperArm    = 0.09f;  // shoulder -> elbow link
constexpr float kRElbowHub    = 0.08f;  // elbow joint housing (height step)
constexpr float kRForearm     = 0.07f;  // elbow -> wrist link
constexpr float kRWristHub    = 0.06f;  // wrist joint housing (height step)
constexpr float kRToolLink    = 0.05f;  // wrist -> tool link
constexpr float kRGripperTip  = 0.03f;  // downward-pointing gripper fingertip

// Human-readable capsule names for [info] lines and the CSV artifact.
// `static constexpr` (not `inline constexpr`): nvcc's device-side front
// end rejects `inline` on a non-function declaration even though it is
// valid ISO C++17 (an "inline variable"); `static` gives each translation
// unit its own small copy instead, which is just as safe for a handful of
// string-literal pointers and portable across both cl.exe and nvcc.
static constexpr const char* kRobotCapsuleNames[kNumRobotCapsules] = {
    "base_column", "shoulder_hub", "upper_arm", "elbow_hub",
    "forearm", "wrist_hub", "tool_link", "gripper_tip"
};

// ===========================================================================
// SECTION 4 — human model: an ANONYMOUS torso+arm capsule pair.
//
// No identity, pose detail, or limb articulation beyond this: a vertical
// TORSO capsule (shoulder-width clearance radius, floor to overhead) and a
// horizontal ARM capsule (a conservative "always reaching toward the cell"
// posture, offset from the torso surface). This is the catalog bullet's
// "anonymous cylinder/capsule" framing, deliberately: the pipeline below
// never asks "who is this" or "which pixel is a hand vs a hip" -- only
// "is there an unexplained foreground surface, and how far is its nearest
// point from the robot."
//
// WHY kHumanTorsoRadius IS 0.18 m AND NOT A ROUNDER "generous" 0.22 m
// (load-bearing, see THEORY.md "Numerical considerations" and README
// "Limitations & honesty" -- this is a real finding from building this
// project, not a cosmetic tuning choice): a purely VERTICAL capsule's
// top-down rendering (capsule_top_at(), SECTION 1) can only ever report
// heights on its TOP hemisphere (from kHumanTorsoHeight up to
// kHumanTorsoHeight + radius) -- the entire cylindrical SIDE of a standing
// person, where a robot part at chest/waist height would actually make
// closest approach, is geometrically invisible to a purely overhead
// camera, at ANY pixel resolution (this is a visibility/occlusion limit,
// not a quantization one -- shrinking the pixel size to zero would not
// reveal it). Earlier development of this project used a larger torso
// radius under which the TRUE (analytic, closed-form) closest pair was
// occasionally the torso's invisible side rather than the arm -- and the
// pixel pipeline, unable to see that side at all, silently fell back to
// the arm pathway, breaking the d_min sandwich-bound gate by tens of
// centimeters (not the few-millimeter pixel-quantization gap the bound is
// about). This radius is chosen SMALL ENOUGH that the horizontal ARM
// capsule (which sits at a height close to the robot's operating heights
// and is therefore a much better top-down-visibility match) is
// ANALYTICALLY the closer human part on every frame of the committed
// scenario (verified by main.cu's own gates, every run) -- keeping the
// project's central numerical claim (the pixel pipeline's error is bounded
// by pixel quantization) actually true, rather than accidentally
// exercising a different, much larger error source. The invisible-side
// limitation itself is real and stays fully documented (README
// "Limitations & honesty", THEORY.md, PRACTICE.md §1) -- production SSM
// systems solve it with side-mounted or multiple/fused viewpoints, never a
// single overhead camera alone.
constexpr int kNumHumanCapsules = 2;
constexpr float kHumanTorsoRadius = 0.18f;  // m, shoulder/hip clearance (see the note above
                                            // for why this is not larger)
constexpr float kHumanTorsoHeight = 1.5f;   // m, floor to top-of-head allowance
constexpr float kHumanArmRadius   = 0.08f;  // m
constexpr float kHumanArmOffset   = 0.22f;  // m, arm starts at the torso surface
constexpr float kHumanArmLength   = 0.5f;   // m, reach length beyond the torso
constexpr float kHumanArmHeight   = 1.0f;   // m, shoulder-height reach

static constexpr const char* kHumanCapsuleNames[kNumHumanCapsules] = {
    "torso", "arm"
};

// ===========================================================================
// SECTION 5 — pixel labels and the SSM state machine.
// ===========================================================================
enum class PixelLabel : uint8_t { BACKGROUND = 0, ROBOT = 1, HUMAN = 2 };

// NORMAL: d_min > S_p(full speed)          -- robot runs at full speed.
// REDUCED: S_p(reduced) < d_min <= S_p(full) -- robot must be at/below its
//          reduced ("creep") speed for S_p(reduced) to remain valid.
// PROTECTIVE_STOP: d_min <= S_p(reduced)   -- robot must be stopped.
// (This three-zone structure -- computing S_p at TWO robot speeds to get
// two boundaries -- mirrors how real multi-zone SSM systems are built;
// THEORY.md "The math" derives both S_p values term by term.)
enum class SsmState : int { NORMAL = 0, REDUCED = 1, PROTECTIVE_STOP = 2 };

inline const char* ssm_state_name(SsmState s)
{
    switch (s) {
        case SsmState::NORMAL:          return "NORMAL";
        case SsmState::REDUCED:         return "REDUCED";
        case SsmState::PROTECTIVE_STOP: return "PROTECTIVE_STOP";
    }
    return "UNKNOWN";
}

// ===========================================================================
// SECTION 6 — ISO/TS-15066-STYLE protective separation distance, S_p.
//
// >>> ILLUSTRATIVE, NOT A REPRODUCTION OF THE STANDARD. <<< ISO/TS 15066
// is a copyrighted document this project does not reproduce; the formula
// below matches the PUBLISHED STRUCTURE described in secondary literature
// (a human-approach term, a robot-approach term, a fixed intrusion/reach
// allowance, and position-uncertainty terms) with round, documented,
// illustrative numbers chosen to fit this compact teaching cell -- not the
// standard's actual default constants. Consult the licensed standard text
// for anything resembling compliance work (THEORY.md, PRACTICE.md §4).
//
//   S_p(v_r) = v_h * (T_r + T_s(v_r))   -- human can close this much
//                                          ground during the robot's
//                                          reaction + stopping time
//            + v_r * T_r                -- robot closes this much ground
//                                          during its OWN reaction time
//                                          (before it even starts stopping)
//            + C_reach                  -- fixed reach/intrusion allowance
//            + Z_detection              -- sensor position uncertainty
//                                          (this project folds the pixel-
//                                          quantization bound in here --
//                                          see THEORY.md "Numerical
//                                          considerations")
//            + Z_robot                  -- robot position uncertainty
//
//   T_s(v_r) = v_r / a_stop             -- stopping time from a documented
//                                          illustrative maximum deceleration
//
// Evaluated at v_r = kVRobotFull  -> S_p_full     (the "must slow down" line)
// Evaluated at v_r = kVRobotReduced -> S_p_reduced (the "must stop" line)
// ===========================================================================
constexpr float kVHumanMax     = 1.6f;   // m/s, illustrative max human approach speed
constexpr float kVRobotFull    = 0.8f;   // m/s, illustrative rated max Cartesian tool speed
constexpr float kVRobotReduced = 0.2f;   // m/s, illustrative reduced/creep speed
constexpr float kTReaction     = 0.05f;  // s, illustrative sensor+processing+comms latency budget
constexpr float kAStop         = 6.0f;   // m/s^2, illustrative max protective-stop deceleration
constexpr float kCReach        = 0.10f;  // m, illustrative reach/intrusion allowance (reduced
                                         // from typical whole-body defaults to fit this compact
                                         // demo cell -- README/PRACTICE say so explicitly)
constexpr float kZDetection    = 0.02f;  // m, illustrative sensor position uncertainty
                                         // (covers the kPixelQuantBound ~= 0.0141 m derived above
                                         // plus a rounding margin for assumed depth noise)
constexpr float kZRobot        = 0.01f;  // m, illustrative robot position uncertainty

// compute_Sp — the formula above, evaluated at a given robot speed. Plain
// `inline` (not `__host__ __device__`): this function runs on the HOST
// only (main.cu's SSM decision logic; no kernel needs it), so it needs no
// CUDA qualifiers and is safe to define in a header included by cl.exe.
inline float compute_Sp(float v_r)
{
    const float t_stop = v_r / kAStop;
    return kVHumanMax * (kTReaction + t_stop) + v_r * kTReaction
         + kCReach + kZDetection + kZRobot;
}

// Hysteresis (THEORY.md "The algorithm" argues the asymmetry): ESCALATING
// to a more-restrictive state happens IMMEDIATELY (0-frame delay) -- safety
// must never wait. DE-ESCALATING to a less-restrictive state requires the
// less-restrictive condition to hold for this many CONSECUTIVE frames --
// availability may wait, safety may not.
constexpr int kHysteresisHoldFrames = 5;   // 5 frames @ 30 Hz = 167 ms

// ===========================================================================
// SECTION 7 — verification gate parameters (main.cu's closed-form checks).
//
// THE d_min SANDWICH BOUND, DERIVED IN TWO PARTS (THEORY.md "Numerical
// considerations" gives the full argument -- this is the short version):
//
//   analytic_dmin <= pipeline_dmin <= analytic_dmin + kPixelQuantBound
//                                                    + kSilhouetteSagBound
//
// LOWER bound: exact, always. Every pixel the pipeline calls HUMAN is
// reconstructed at a point that lies EXACTLY on the human capsule's true
// surface (capsule_top_at() is exact, not approximate, for the horizontal/
// vertical capsules this project uses -- kernels.cuh SECTION 1); a discrete
// sample of points on a surface can never find a smaller minimum distance
// than the true continuous minimum over that same surface.
//
// UPPER bound, two additive error sources:
//   kPixelQuantBound     -- horizontal pixel quantization: the nearest
//                           pixel center is within half a pixel diagonal
//                           of any given (x,y), and point-to-convex-set
//                           distance is 1-Lipschitz in the query point.
//   kSilhouetteSagBound  -- a SECOND, LARGER, and more interesting source
//                           this project's development surfaced: a
//                           top-down camera only ever sees a capsule's
//                           UPPER surface, so if the true globally-closest
//                           point on the human capsule sits slightly below
//                           its own axis height (because the nearby robot
//                           capsule is a little lower), the best the pixel
//                           pipeline can do is the visible point at the
//                           capsule's "equator" (axis height, radius
//                           offset) -- an extra, resolution-INDEPENDENT
//                           gap bounded by how far below the arm's own
//                           height a relevant robot capsule sits. This
//                           project's scenario keeps the ARM (not the
//                           TORSO -- see kHumanTorsoRadius's comment)
//                           analytically closest for exactly this reason,
//                           and every robot capsule it approaches
//                           (tool_link/wrist_hub at kZWrist, forearm at
//                           kZForearm) sits within 0.05 m of the arm's own
//                           height (kHumanArmHeight) -- so 0.05 m is a
//                           scenario-CALIBRATED bound, not a universal one
//                           (README Exercises names deriving the general
//                           case, which depends on which robot capsule is
//                           nearby, as a follow-on).
// ---------------------------------------------------------------------------
constexpr float kSilhouetteSagBound = 0.05f;   // m -- see the derivation above

// "No false stop": frames where the ANALYTIC (closed-form) distance exceeds
// S_p_full by more than this margin must never show PROTECTIVE_STOP (in
// fact must show NORMAL -- see main.cu). This gate only needs the LOWER
// half of the sandwich bound (pipeline_dmin >= analytic_dmin, exact,
// unconditionally) -- if the true distance is comfortably large, the
// pipeline's distance is AT LEAST as large, so a small margin (absorbing
// FP32/self-filter epsilon effects only) suffices regardless of the upper
// bound's size.
constexpr float kNoFalseStopMargin  = 0.05f;   // m
// "No missed stop": frames where the analytic distance is below S_p_reduced
// by more than this margin must show PROTECTIVE_STOP. This gate needs the
// UPPER half of the sandwich bound (the pipeline can OVERESTIMATE distance
// by up to kPixelQuantBound + kSilhouetteSagBound), so the margin must
// exceed that sum with headroom -- otherwise a genuinely dangerous frame
// could be measured as just barely outside PROTECTIVE_STOP range.
constexpr float kNoMissedStopMargin = 0.08f;   // m
// State transitions (escalation frames directly, de-escalation frames after
// adding the hysteresis hold) must land within this many frames of the
// analytic S_p crossing.
constexpr int kTransitionFrameTolerance = 1;   // frames
// d_min sandwich bound slack: kPixelQuantBound + kSilhouetteSagBound cover
// the PROVEN geometric bias (above); this small extra slack absorbs FP32
// rounding and the self-filter epsilon band, honestly labeled as slack
// rather than folded into the proven bound.
constexpr float kDminBoundSlack = 0.006f;      // m

// ===========================================================================
// SECTION 8 — the pipeline entry points (GPU launchers + CPU oracle twins).
//
// GPU-side capsule state lives in __constant__ memory inside kernels.cu
// (SECTION-level rationale in kernels.cu's header comment); upload_capsules
// must be called once per frame BEFORE any of the three launch_* calls
// below consume that frame's geometry.
// ===========================================================================

// upload_capsules — copy this frame's robot and human capsules into GPU
// constant memory (cudaMemcpyToSymbol). DEVICE-side effect only; both
// pointers are HOST arrays of the stated fixed sizes.
void upload_capsules(const Capsule robot[kNumRobotCapsules],
                     const Capsule human[kNumHumanCapsules]);

// launch_render_classify — stage 1 (+ fused stage 2): render the top-down
// depth image and classify every pixel BACKGROUND/ROBOT/HUMAN using the
// capsules most recently uploaded by upload_capsules(). d_depth/d_label are
// DEVICE pointers, [kNumPixels] each, caller-allocated (persistent buffers
// reused every frame -- see main.cu's SEQUENCE stage).
void launch_render_classify(float* d_depth, uint8_t* d_label);

// launch_human_min_distance — stage 3 (map + reduce): for every HUMAN
// pixel, the point-capsule distance to the nearest robot capsule; reduce
// to this frame's d_min and its closest capsule id. d_depth/d_label are
// DEVICE pointers (as produced by launch_render_classify); d_block_mins/
// d_block_ids are DEVICE scratch, [kReduceBlocks] each (see kernels.cu),
// caller-allocated and reused every frame. out_dmin/out_closest_capsule
// are HOST pointers this call fills in (it does the tiny final host-side
// reduction over the block partials itself, mirroring 08.01's "trivial
// host finish" pattern).
void launch_human_min_distance(const float* d_depth, const uint8_t* d_label,
                               float* d_block_mins, int* d_block_ids,
                               float* out_dmin, int* out_closest_capsule);

// launch_dense_distance_field — the bullet's dense variant: EVERY pixel's
// distance to the nearest robot capsule (a pure map, reusing the same
// device distance function as stage 3). d_depth is the DEVICE depth image
// (as produced by launch_render_classify); d_field is a DEVICE pointer,
// [kNumPixels], caller-allocated. Run once, for the artifact frame only
// (main.cu) -- it is not needed every SEQUENCE frame.
void launch_dense_distance_field(const float* d_depth, float* d_field);

// reduce_num_blocks — the compile-time-determined grid size
// launch_human_min_distance's reduction kernel uses (declared here so
// main.cu can size d_block_mins/d_block_ids identically without repeating
// the arithmetic -- see kernels.cu for the formula and its reasoning).
int reduce_num_blocks();

// ---------------------------------------------------------------------------
// CPU oracle twins (reference_cpu.cpp) — independent, plain-C++
// reimplementations of the three GPU stages above, used by main.cu's
// VERIFY stage to catch indexing/layout/threading bugs the GPU path alone
// could hide (CLAUDE.md §5). Unlike the GPU path, these take the robot
// capsule array as an explicit parameter (constant memory is a GPU-only
// concept) and run single-threaded, sequentially, pixel by pixel.
// ---------------------------------------------------------------------------
void render_classify_cpu(const Capsule robot[kNumRobotCapsules],
                         const Capsule human[kNumHumanCapsules],
                         float* depth, uint8_t* label);

void human_min_distance_cpu(const float* depth, const uint8_t* label,
                            const Capsule robot[kNumRobotCapsules],
                            float* out_dmin, int* out_closest_capsule);

void dense_distance_field_cpu(const float* depth,
                              const Capsule robot[kNumRobotCapsules],
                              float* field);

#endif // PROJECT_KERNELS_CUH
