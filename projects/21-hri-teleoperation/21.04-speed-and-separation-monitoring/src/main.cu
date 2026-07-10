// ===========================================================================
// main.cu — entry point for project 21.04
//           Speed-and-separation monitoring: depth streams -> minimum-
//           distance fields at frame rate (ISO/TS 15066-style helper)
//
// >>> DIDACTIC IMPLEMENTATION -- NOT A CERTIFIED SAFETY FUNCTION. <<<
// See kernels.cuh's header comment for the full caveat; the demo also
// prints a stable NOTICE: line saying exactly this, every run.
//
// What this program does, start to finish
// ----------------------------------------
//   1. Load the committed scenario (human path + frame count/rate) from
//      data/sample/, print the SSM thresholds it derives.
//   2. VERIFY STAGE (the §5 GPU-vs-CPU gate): at two representative frames
//      (the far start and the measured-by-construction closest-approach
//      instant), run all three GPU kernels AND their CPU oracle twins on
//      IDENTICAL scene geometry, and require agreement -- exact for pixel
//      labels, tight FP32 tolerance for depth/distance values.
//   3. SEQUENCE STAGE: run the GPU pipeline for every frame of the demo
//      (human approaches, the SSM state machine escalates NORMAL->
//      REDUCED->PROTECTIVE_STOP, the human retreats, it recovers), logging
//      d_min/state to demo/out/ssm_timeline.csv and rendering demo/out/
//      distance_field.pgm at the measured closest-approach frame.
//   4. FOUR VERIFICATION GATES, each checked against an ANALYTIC (closed-
//      form, double-precision, pixel-pipeline-independent) ground truth
//      computed directly from the scenario's parametric geometry:
//        NO-FALSE-STOP, NO-MISSED-STOP, TRANSITIONS (+/-1 frame), and the
//        proven d_min SANDWICH BOUND (THEORY.md derives all four).
//   5. Exit 0 only if VERIFY and all four gates pass.
//
// Three independent code paths meet in this file, and it matters which is
// which (THEORY.md "How we verify correctness" explains why three, not
// two): the GPU PIPELINE (kernels.cu) and its CPU ORACLE TWIN
// (reference_cpu.cpp) both start from the SAME rendered/reconstructed
// pixels -- comparing them catches indexing/threading/formula bugs in the
// pixel pipeline itself. The ANALYTIC ground truth (this file, namespace
// analytic::) never touches a pixel at all -- it evaluates the scenario's
// closed-form capsule geometry directly -- so comparing the PIPELINE
// against IT catches a different class of bug: "the pipeline agrees with
// itself but is measuring the wrong thing" (e.g. a systematic rendering
// bias, an occlusion the scenario should not have, or a threshold set
// against the wrong quantity).
//
// Output contract: stable lines are "[demo]", "NOTICE:", "PROBLEM:",
// "SCENARIO:", "VERIFY:", "ARTIFACT:", "GATE ...:", "RESULT:" --
// "[info]"/"[time]" lines are unchecked (see demo/expected_output.txt).
// Change a stable line => update demo/expected_output.txt in the same
// change.
//
// Read this first, then kernels.cuh -> reference_cpu.cpp -> kernels.cu.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
#include <direct.h>               // _mkdir (std::filesystem avoided in .cu -- see 07.09)
#else
#include <sys/stat.h>
#endif

// ===========================================================================
// Scenario loading -- the committed "task definition": how many frames the
// depth camera runs, at what rate, and where the human's walk starts/ends.
// Everything else (robot FK model, cell/camera geometry, capsule radii,
// SSM formula parameters) is the compile-time "model" in kernels.cuh --
// the same split 08.01 uses (x0/steps loaded; cart-pole physics and MPPI
// hyperparameters compiled in).
// ===========================================================================
struct Scenario {
    int   frames = 0;
    float rate_hz = 0.0f;
    float human_start_x = 0.0f, human_start_y = 0.0f;
    float human_closest_x = 0.0f, human_closest_y = 0.0f;
    bool  loaded = false;
};

static Scenario load_scenario(const std::string& path)
{
    Scenario sc;
    std::ifstream in(path);
    if (!in.is_open()) return sc;

    bool have_frames = false, have_rate = false, have_start = false, have_closest = false;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string label, cell;
        std::getline(ss, label, ',');
        if (label == "FRAMES") {
            if (!std::getline(ss, cell, ',')) { std::fprintf(stderr, "scenario: short FRAMES row\n"); return Scenario{}; }
            sc.frames = std::atoi(cell.c_str());
            have_frames = true;
        } else if (label == "RATE_HZ") {
            if (!std::getline(ss, cell, ',')) { std::fprintf(stderr, "scenario: short RATE_HZ row\n"); return Scenario{}; }
            sc.rate_hz = std::strtof(cell.c_str(), nullptr);
            have_rate = true;
        } else if (label == "HUMAN_START") {
            std::string cx, cy;
            if (!std::getline(ss, cx, ',') || !std::getline(ss, cy, ',')) { std::fprintf(stderr, "scenario: short HUMAN_START row\n"); return Scenario{}; }
            sc.human_start_x = std::strtof(cx.c_str(), nullptr);
            sc.human_start_y = std::strtof(cy.c_str(), nullptr);
            have_start = true;
        } else if (label == "HUMAN_CLOSEST") {
            std::string cx, cy;
            if (!std::getline(ss, cx, ',') || !std::getline(ss, cy, ',')) { std::fprintf(stderr, "scenario: short HUMAN_CLOSEST row\n"); return Scenario{}; }
            sc.human_closest_x = std::strtof(cx.c_str(), nullptr);
            sc.human_closest_y = std::strtof(cy.c_str(), nullptr);
            have_closest = true;
        } else {
            std::fprintf(stderr, "scenario: unknown row label '%s'\n", label.c_str());
            return Scenario{};
        }
    }
    if (!have_frames || !have_rate || !have_start || !have_closest || sc.frames < 2 || sc.rate_hz <= 0.0f) {
        std::fprintf(stderr, "scenario: missing or invalid fields\n");
        return Scenario{};
    }
    sc.loaded = true;
    return sc;
}

// ===========================================================================
// build_scene — the SHARED scenario geometry generator: turns a time
// instant into this frame's 8 robot capsules + 2 human capsules. Used
// IDENTICALLY by the GPU path, the CPU oracle, and the analytic ground
// truth below -- it is the scene DEFINITION, not a thing under test (the
// same role 08.01's plant-parameter constants play: shared input, not a
// GPU-vs-CPU comparison target). See kernels.cuh SECTION 3/4 for the
// forward-kinematics and human-path derivations this function implements.
// ===========================================================================
struct SceneCapsules {
    Capsule robot[kNumRobotCapsules];
    Capsule human[kNumHumanCapsules];
};

constexpr float kPi = 3.14159265358979323846f;

// reach_fraction -- the single 0->1->0 raised-cosine profile that drives
// BOTH the robot's synchronized reach cycle and the human's approach-
// retreat walk (kernels.cuh SECTION 3's comment explains why sharing one
// profile is a deliberate scenario choice, not an accident): 0 at t=0 and
// t=t_total, 1 at the midpoint, smooth (zero velocity at both ends and at
// the peak) everywhere in between.
static float reach_fraction(float t, float t_total)
{
    return 0.5f * (1.0f - cosf(2.0f * kPi * t / t_total));
}

static SceneCapsules build_scene(float t, float t_total, const Scenario& scenario)
{
    SceneCapsules sc;
    const float rf = reach_fraction(t, t_total);
    const float deg2rad = kPi / 180.0f;

    // ---- robot: SCARA forward kinematics (kernels.cuh SECTION 3) --------
    const float th1 = (kJoint1MeanDeg + kJoint1AmpDeg * rf) * deg2rad;
    const float th2 = (kJoint2MeanDeg + kJoint2AmpDeg * rf) * deg2rad;
    const float th3 = (kJoint3MeanDeg + kJoint3AmpDeg * rf) * deg2rad;

    const float shoulder_x = 0.0f, shoulder_y = 0.0f;   // robot base, fixed at the cell origin
    const float a1 = th1;
    const float elbow_x = shoulder_x + kL1 * cosf(a1);
    const float elbow_y = shoulder_y + kL1 * sinf(a1);
    const float a2 = a1 + th2;
    const float wrist_x = elbow_x + kL2 * cosf(a2);
    const float wrist_y = elbow_y + kL2 * sinf(a2);
    const float a3 = a2 + th3;
    const float tool_x = wrist_x + kL3 * cosf(a3);
    const float tool_y = wrist_y + kL3 * sinf(a3);

    // Emission order matches kRobotCapsuleNames exactly (kernels.cuh). Every
    // VERTICAL capsule below lists its HIGHER endpoint as B, the convention
    // capsule_top_at() relies on (kernels.cuh SECTION 1).
    int k = 0;
    sc.robot[k++] = Capsule{ shoulder_x, shoulder_y, 0.0f,
                             shoulder_x, shoulder_y, kZShoulder,
                             kRBaseColumn, 1 };                          // base_column (vertical pedestal)
    sc.robot[k++] = Capsule{ shoulder_x, shoulder_y, kZShoulder - 0.05f,
                             shoulder_x, shoulder_y, kZShoulder,
                             kRShoulderHub, 1 };                         // shoulder_hub (vertical)
    sc.robot[k++] = Capsule{ shoulder_x, shoulder_y, kZShoulder,
                             elbow_x, elbow_y, kZShoulder,
                             kRUpperArm, 0 };                            // upper_arm (horizontal)
    sc.robot[k++] = Capsule{ elbow_x, elbow_y, kZForearm,
                             elbow_x, elbow_y, kZShoulder,
                             kRElbowHub, 1 };                            // elbow_hub (vertical height step)
    sc.robot[k++] = Capsule{ elbow_x, elbow_y, kZForearm,
                             wrist_x, wrist_y, kZForearm,
                             kRForearm, 0 };                             // forearm (horizontal)
    sc.robot[k++] = Capsule{ wrist_x, wrist_y, kZWrist,
                             wrist_x, wrist_y, kZForearm,
                             kRWristHub, 1 };                            // wrist_hub (vertical height step)
    sc.robot[k++] = Capsule{ wrist_x, wrist_y, kZWrist,
                             tool_x, tool_y, kZWrist,
                             kRToolLink, 0 };                            // tool_link (horizontal)
    sc.robot[k++] = Capsule{ tool_x, tool_y, kZWrist - 0.15f,
                             tool_x, tool_y, kZWrist,
                             kRGripperTip, 1 };                          // gripper_tip (vertical, hangs down)

    // ---- human: anonymous torso + reaching-arm capsule pair --------------
    // (kernels.cuh SECTION 4). The SAME reach_fraction(t) drives the walk.
    const float hx = scenario.human_start_x + (scenario.human_closest_x - scenario.human_start_x) * rf;
    const float hy = scenario.human_start_y + (scenario.human_closest_y - scenario.human_start_y) * rf;
    sc.human[0] = Capsule{ hx, hy, 0.0f,
                           hx, hy, kHumanTorsoHeight,
                           kHumanTorsoRadius, 1 };                       // torso (vertical)
    sc.human[1] = Capsule{ hx + kHumanArmOffset, hy, kHumanArmHeight,
                           hx + kHumanArmOffset + kHumanArmLength, hy, kHumanArmHeight,
                           kHumanArmRadius, 0 };                         // arm (horizontal, fixed +x reach posture)
    return sc;
}

// ===========================================================================
// analytic:: — the CLOSED-FORM ground truth, entirely independent of the
// pixel pipeline (see this file's header comment for why a THIRD path is
// worth having). Runs in double precision throughout, deliberately more
// precise than the FP32 pipeline it verifies.
// ===========================================================================
namespace analytic {

struct Vec3 { double x, y, z; };
static Vec3   sub(Vec3 a, Vec3 b)   { return Vec3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static Vec3   add(Vec3 a, Vec3 b)   { return Vec3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static Vec3   scale(Vec3 a, double s) { return Vec3{ a.x * s, a.y * s, a.z * s }; }
static double dot(Vec3 a, Vec3 b)   { return a.x * b.x + a.y * b.y + a.z * b.z; }
static double norm(Vec3 a)          { return std::sqrt(dot(a, a)); }

// closest_seg_seg_distance -- the standard closed-form closest-distance-
// between-two-3-D-segments algorithm (Ericson, "Real-Time Collision
// Detection" §5.1.9; equivalently Lumelsky 1985 / Sunday's "distance3D_
// Segment_to_Segment"). This project REIMPLEMENTS the well-known algorithm
// didactically -- it is prior art, not an original derivation (README
// "Prior art & further reading" credits it). Minimizes
// |P1(s) - P2(t)|^2 = |(p1 + s*d1) - (p2 + t*d2)|^2 over s,t in [0,1] by
// solving the 2x2 linear system from setting the gradient to zero, with
// the standard degenerate-segment and out-of-range clamping fallbacks
// (THEORY.md "The math" walks the derivation in full).
static double closest_seg_seg_distance(Vec3 p1, Vec3 q1, Vec3 p2, Vec3 q2)
{
    const Vec3 d1 = sub(q1, p1), d2 = sub(q2, p2), r = sub(p1, p2);
    const double a = dot(d1, d1), e = dot(d2, d2), f = dot(d2, r);
    const double eps = 1e-12;
    double s, t;
    if (a <= eps && e <= eps) {
        s = 0.0; t = 0.0;                                        // both segments degenerate to points
    } else if (a <= eps) {
        s = 0.0; t = std::min(1.0, std::max(0.0, f / e));        // segment 1 is a point
    } else {
        const double c = dot(d1, r);
        if (e <= eps) {
            t = 0.0; s = std::min(1.0, std::max(0.0, -c / a));   // segment 2 is a point
        } else {
            const double b = dot(d1, d2);
            const double denom = a * e - b * b;                  // 0 exactly when the segments are parallel
            s = (denom != 0.0) ? std::min(1.0, std::max(0.0, (b * f - c * e) / denom)) : 0.0;
            t = (b * s + f) / e;
            if (t < 0.0) { t = 0.0; s = std::min(1.0, std::max(0.0, -c / a)); }
            else if (t > 1.0) { t = 1.0; s = std::min(1.0, std::max(0.0, (b - c) / a)); }
        }
    }
    const Vec3 c1 = add(p1, scale(d1, s));
    const Vec3 c2 = add(p2, scale(d2, t));
    return norm(sub(c1, c2));
}

static double capsule_pair_distance(const Capsule& a, const Capsule& b)
{
    const double d = closest_seg_seg_distance(
        Vec3{ a.ax, a.ay, a.az }, Vec3{ a.bx, a.by, a.bz },
        Vec3{ b.ax, b.ay, b.az }, Vec3{ b.bx, b.by, b.bz });
    const double out = d - static_cast<double>(a.radius) - static_cast<double>(b.radius);
    return out > 0.0 ? out : 0.0;
}

// scene_min_distance -- the TRUE minimum distance between every robot
// capsule and every human capsule (16 pairs -- trivial cost), computed
// directly from build_scene()'s closed-form geometry. NEVER touches a
// pixel, a depth image, or a label -- the independence that makes it a
// trustworthy ground truth for the pipeline's own verification gates.
struct Result { double dmin; int robot_idx; int human_idx; };
static Result scene_min_distance(const SceneCapsules& sc)
{
    Result best{ 1e18, -1, -1 };
    for (int i = 0; i < kNumRobotCapsules; ++i) {
        for (int j = 0; j < kNumHumanCapsules; ++j) {
            const double d = capsule_pair_distance(sc.robot[i], sc.human[j]);
            if (d < best.dmin) { best.dmin = d; best.robot_idx = i; best.human_idx = j; }
        }
    }
    return best;
}

} // namespace analytic

// ===========================================================================
// SSM decision logic: raw threshold classification + the hysteresis state
// machine (kernels.cuh SECTION 6 documents the escalate-immediately /
// de-escalate-after-hold asymmetry; THEORY.md "The algorithm" argues it).
// ===========================================================================
static SsmState classify_raw(float d, float sp_full, float sp_reduced)
{
    if (d <= sp_reduced) return SsmState::PROTECTIVE_STOP;
    if (d <= sp_full)    return SsmState::REDUCED;
    return SsmState::NORMAL;
}

struct HysteresisFsm {
    SsmState state = SsmState::NORMAL;
    int hold_count = 0;

    // step -- advance the state machine by one frame given this frame's
    // RAW classification. Returns true (and fills old_state/new_state) iff
    // the state actually changed this frame.
    //
    // ESCALATION (raw more restrictive than the current state): apply
    // IMMEDIATELY, reset the hold counter -- 0-frame delay, always.
    // DE-ESCALATION (raw less restrictive): increment a hold counter; only
    // relax once the counter reaches kHysteresisHoldFrames. Because the
    // counter starts at 1 on the FIRST qualifying frame, the transition
    // actually fires on the (kHysteresisHoldFrames)-th consecutive
    // qualifying frame -- i.e. at (first_qualifying_frame +
    // kHysteresisHoldFrames - 1). main.cu's TRANSITIONS gate uses this
    // exact offset when predicting de-escalation frames from the analytic
    // crossing (see below).
    bool step(SsmState raw, SsmState* old_state, SsmState* new_state)
    {
        const SsmState before = state;
        if (static_cast<int>(raw) > static_cast<int>(state)) {
            state = raw;
            hold_count = 0;
        } else if (static_cast<int>(raw) < static_cast<int>(state)) {
            ++hold_count;
            if (hold_count >= kHysteresisHoldFrames) {
                state = raw;
                hold_count = 0;
            }
        } else {
            hold_count = 0;
        }
        if (state != before) { *old_state = before; *new_state = state; return true; }
        return false;
    }
};

// ===========================================================================
// PGM artifact writer (the smallest real image format -- P5: one header
// line trio, then raw bytes; same idiom as 07.09's distance.pgm/voronoi.pgm).
// ===========================================================================
static bool write_pgm(const std::string& path, int width, int height,
                      const std::vector<uint8_t>& gray)
{
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) return false;
    out << "P5\n" << width << " " << height << "\n255\n";
    out.write(reinterpret_cast<const char*>(gray.data()),
              static_cast<std::streamsize>(gray.size()));
    return static_cast<bool>(out);
}

static bool ensure_dir(const std::string& path)
{
#ifdef _WIN32
    const int r = _mkdir(path.c_str());
#else
    const int r = mkdir(path.c_str(), 0755);
#endif
    return r == 0 || errno == EEXIST;
}

// Path helpers (same exe-relative resolution as 08.01/07.09: the exe sits
// at build/x64/<Config>/, three levels below the project root).
static std::string project_root_from(const char* argv0)
{
    std::string exe(argv0 ? argv0 : "");
    size_t cut = exe.find_last_of("/\\");
    if (cut == std::string::npos) return ".";
    return exe.substr(0, cut) + "/../../..";
}

static std::string find_scenario(const std::string& cli_path, const char* argv0)
{
    std::vector<std::string> candidates;
    if (!cli_path.empty()) candidates.push_back(cli_path);
    candidates.push_back(project_root_from(argv0) + "/data/sample/ssm_scenario.csv");
    candidates.push_back("data/sample/ssm_scenario.csv");
    candidates.push_back("../data/sample/ssm_scenario.csv");
    for (const auto& c : candidates)
        if (std::ifstream(c).is_open()) return c;
    return "";
}

// ---------------------------------------------------------------------------
// run_verify_frame -- run the GPU pipeline and the CPU oracle on IDENTICAL
// scene geometry at one frame and accumulate the worst-case disagreement
// into the caller's running totals. Used twice by the VERIFY stage (frame 0
// and the scenario's designed closest-approach midpoint).
// ---------------------------------------------------------------------------
struct VerifyTotals {
    long long label_mismatches = 0;
    float depth_max_diff = 0.0f;
    float dmin_max_diff = 0.0f;
    float field_max_diff = 0.0f;
};

static void run_verify_frame(const SceneCapsules& sc, VerifyTotals* totals)
{
    static std::vector<float>   depth_gpu(kNumPixels), depth_cpu(kNumPixels);
    static std::vector<uint8_t> label_gpu(kNumPixels), label_cpu(kNumPixels);
    static std::vector<float>   field_gpu(kNumPixels), field_cpu(kNumPixels);

    float *d_depth = nullptr, *d_field = nullptr;
    uint8_t* d_label = nullptr;
    float* d_block_mins = nullptr;
    int* d_block_ids = nullptr;
    const int blocks = reduce_num_blocks();

    CUDA_CHECK(cudaMalloc(&d_depth, kNumPixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_label, kNumPixels * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_field, kNumPixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_block_mins, blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_block_ids, blocks * sizeof(int)));

    // ---- GPU path -----------------------------------------------------------
    upload_capsules(sc.robot, sc.human);
    launch_render_classify(d_depth, d_label);
    float dmin_gpu = 0.0f; int cid_gpu = -1;
    launch_human_min_distance(d_depth, d_label, d_block_mins, d_block_ids, &dmin_gpu, &cid_gpu);
    launch_dense_distance_field(d_depth, d_field);
    CUDA_CHECK(cudaMemcpy(depth_gpu.data(), d_depth, kNumPixels * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(label_gpu.data(), d_label, kNumPixels * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(field_gpu.data(), d_field, kNumPixels * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_depth));
    CUDA_CHECK(cudaFree(d_label));
    CUDA_CHECK(cudaFree(d_field));
    CUDA_CHECK(cudaFree(d_block_mins));
    CUDA_CHECK(cudaFree(d_block_ids));

    // ---- CPU oracle -----------------------------------------------------------
    render_classify_cpu(sc.robot, sc.human, depth_cpu.data(), label_cpu.data());
    float dmin_cpu = 0.0f; int cid_cpu = -1;
    human_min_distance_cpu(depth_cpu.data(), label_cpu.data(), sc.robot, &dmin_cpu, &cid_cpu);
    dense_distance_field_cpu(depth_cpu.data(), sc.robot, field_cpu.data());

    // ---- compare ----------------------------------------------------------
    for (int i = 0; i < kNumPixels; ++i) {
        if (label_gpu[static_cast<size_t>(i)] != label_cpu[static_cast<size_t>(i)])
            totals->label_mismatches++;
        const float dd = std::fabs(depth_gpu[static_cast<size_t>(i)] - depth_cpu[static_cast<size_t>(i)]);
        if (dd > totals->depth_max_diff) totals->depth_max_diff = dd;
        const float fd = std::fabs(field_gpu[static_cast<size_t>(i)] - field_cpu[static_cast<size_t>(i)]);
        if (fd > totals->field_max_diff) totals->field_max_diff = fd;
    }
    const float dm = std::fabs(dmin_gpu - dmin_cpu);
    if (dm > totals->dmin_max_diff) totals->dmin_max_diff = dm;
    (void)cid_gpu; (void)cid_cpu;   // ids are informational only; distance agreement is the gate
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    std::string data_path;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--data") && i + 1 < argc) data_path = argv[++i];
        else {
            std::fprintf(stderr, "usage: %s [--data ssm_scenario.csv]\n", argv[0]);
            return 2;
        }
    }

    std::printf("[demo] speed-and-separation monitoring: depth streams -> minimum-distance fields (project 21.04)\n");
    print_device_info();
    std::printf("NOTICE: didactic implementation -- NOT a certified safety function (ISO/TS 15066-style metrics, illustrative only)\n");

    // ---- scenario -----------------------------------------------------------
    const std::string scenario_path = find_scenario(data_path, argv[0]);
    if (scenario_path.empty()) {
        std::printf("SCENARIO: NOT FOUND -- data/sample/ssm_scenario.csv missing (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (scenario missing)\n");
        return 1;
    }
    std::printf("[info] scenario file: %s\n", scenario_path.c_str());
    Scenario scenario = load_scenario(scenario_path);
    if (!scenario.loaded) {
        std::printf("SCENARIO: MALFORMED -- see stderr\n");
        std::printf("RESULT: FAIL (scenario malformed)\n");
        return 1;
    }
    const float t_total = static_cast<float>(scenario.frames) / scenario.rate_hz;

    std::printf("PROBLEM: SSM pipeline on synthetic top-down depth streams, %dx%d px over %.1fx%.1f m cell @ %.0f Hz, %d robot capsules + %d human capsules, %d frames\n",
               kImageW, kImageH, kCellMaxX - kCellMinX, kCellMaxY - kCellMinY,
               static_cast<double>(scenario.rate_hz), kNumRobotCapsules, kNumHumanCapsules, scenario.frames);
    std::printf("SCENARIO: human walks (%.2f,%.2f) -> (%.2f,%.2f) -> (%.2f,%.2f) while a SCARA-style arm performs a synchronized reach cycle [synthetic]\n",
               static_cast<double>(scenario.human_start_x), static_cast<double>(scenario.human_start_y),
               static_cast<double>(scenario.human_closest_x), static_cast<double>(scenario.human_closest_y),
               static_cast<double>(scenario.human_start_x), static_cast<double>(scenario.human_start_y));

    const float sp_full = compute_Sp(kVRobotFull);
    const float sp_reduced = compute_Sp(kVRobotReduced);
    std::printf("[info] SSM thresholds: Sp_full=%.4f m (v_r=%.2f m/s)  Sp_reduced=%.4f m (v_r=%.2f m/s)  hysteresis hold=%d frames (%.0f ms)  pixel size=%.4f m  pixel-quant bound=%.4f m\n",
               static_cast<double>(sp_full), static_cast<double>(kVRobotFull),
               static_cast<double>(sp_reduced), static_cast<double>(kVRobotReduced),
               kHysteresisHoldFrames, 1000.0 * kHysteresisHoldFrames / static_cast<double>(scenario.rate_hz),
               static_cast<double>(kPixelSizeX), static_cast<double>(kPixelQuantBound));

    // ======================= VERIFY STAGE ====================================
    // Two representative frames: t=0 (human far away) and the scenario's
    // designed midpoint (both the robot's reach and the human's walk peak
    // there by construction -- kernels.cuh SECTION 3). Spot-checking two
    // geometrically different frames catches bugs a single frame could hide
    // (e.g. an "always background" bug that a human-far frame would miss).
    VerifyTotals verify_totals;
    {
        const int verify_frames[2] = { 0, scenario.frames / 2 };
        for (int vf : verify_frames) {
            const float t = static_cast<float>(vf) / scenario.rate_hz;
            const SceneCapsules sc = build_scene(t, t_total, scenario);
            run_verify_frame(sc, &verify_totals);
        }
    }
    std::printf("[info] verify: %lld label mismatches over %d pixels x 2 frames; depth max|d|=%.3e m; d_min max|d|=%.3e m; dense-field max|d|=%.3e m\n",
               verify_totals.label_mismatches, kNumPixels,
               static_cast<double>(verify_totals.depth_max_diff),
               static_cast<double>(verify_totals.dmin_max_diff),
               static_cast<double>(verify_totals.field_max_diff));

    constexpr float kVerifyDepthTol = 1e-4f;   // m
    constexpr float kVerifyDminTol  = 1e-4f;   // m
    constexpr float kVerifyFieldTol = 1e-4f;   // m
    const bool verify_pass = (verify_totals.label_mismatches == 0)
                          && (verify_totals.depth_max_diff <= kVerifyDepthTol)
                          && (verify_totals.dmin_max_diff  <= kVerifyDminTol)
                          && (verify_totals.field_max_diff <= kVerifyFieldTol);
    std::printf("VERIFY: %s (GPU matches CPU reference: exact pixel labels; depth/d_min/dense-field agree within tol 1e-4 m)\n",
               verify_pass ? "PASS" : "FAIL");
    if (!verify_pass) {
        std::printf("RESULT: FAIL (GPU/CPU pipeline disagreement -- fix before trusting the SSM decision)\n");
        return 1;
    }

    // ======================= SEQUENCE STAGE ===================================
    // Persistent device buffers, allocated ONCE outside the frame loop (the
    // 08.01 lesson: cudaMalloc costs hundreds of microseconds, and a
    // per-frame loop that reallocates spends its budget on the allocator).
    float* d_depth = nullptr;
    uint8_t* d_label = nullptr;
    float* d_block_mins = nullptr;
    int* d_block_ids = nullptr;
    float* d_field = nullptr;
    const int blocks = reduce_num_blocks();
    CUDA_CHECK(cudaMalloc(&d_depth, kNumPixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_label, kNumPixels * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_block_mins, blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_block_ids, blocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_field, kNumPixels * sizeof(float)));

    const int N = scenario.frames;
    std::vector<float>    pipeline_dmin(static_cast<size_t>(N));
    std::vector<int>      pipeline_state(static_cast<size_t>(N));       // SsmState, stored as int
    std::vector<int>      pipeline_closest_cap(static_cast<size_t>(N));
    std::vector<double>   analytic_dmin(static_cast<size_t>(N));

    struct TransitionEvent { int frame; SsmState old_state; SsmState new_state; };
    std::vector<TransitionEvent> pipeline_transitions;

    HysteresisFsm fsm;
    double loop_gpu_ms = 0.0;

    for (int frame = 0; frame < N; ++frame) {
        const float t = static_cast<float>(frame) / scenario.rate_hz;
        const SceneCapsules sc = build_scene(t, t_total, scenario);

        GpuTimer gt;
        gt.begin();
        upload_capsules(sc.robot, sc.human);
        launch_render_classify(d_depth, d_label);
        float dmin = 0.0f; int cid = -1;
        launch_human_min_distance(d_depth, d_label, d_block_mins, d_block_ids, &dmin, &cid);
        loop_gpu_ms += static_cast<double>(gt.end_ms());

        const analytic::Result truth = analytic::scene_min_distance(sc);

        const SsmState raw = classify_raw(dmin, sp_full, sp_reduced);
        SsmState old_state, new_state;
        if (fsm.step(raw, &old_state, &new_state))
            pipeline_transitions.push_back(TransitionEvent{ frame, old_state, new_state });

        pipeline_dmin[static_cast<size_t>(frame)] = dmin;
        pipeline_state[static_cast<size_t>(frame)] = static_cast<int>(fsm.state);
        pipeline_closest_cap[static_cast<size_t>(frame)] = cid;
        analytic_dmin[static_cast<size_t>(frame)] = truth.dmin;
    }

    // ---- artifact frame: the MEASURED closest approach (not assumed) --------
    int closest_frame = 0;
    for (int f = 1; f < N; ++f)
        if (pipeline_dmin[static_cast<size_t>(f)] < pipeline_dmin[static_cast<size_t>(closest_frame)])
            closest_frame = f;

    {
        const float t = static_cast<float>(closest_frame) / scenario.rate_hz;
        const SceneCapsules sc = build_scene(t, t_total, scenario);
        upload_capsules(sc.robot, sc.human);
        launch_render_classify(d_depth, d_label);
        launch_dense_distance_field(d_depth, d_field);
        std::vector<float> field(kNumPixels);
        CUDA_CHECK(cudaMemcpy(field.data(), d_field, kNumPixels * sizeof(float), cudaMemcpyDeviceToHost));

        // The analytic (closed-form) ground truth at this SAME frame, for
        // an apples-to-apples [info] comparison right next to the pipeline's
        // own measurement -- and the only place kHumanCapsuleNames earns its
        // keep (kernels.cuh SECTION 4).
        const analytic::Result truth_here = analytic::scene_min_distance(sc);
        std::printf("[info] closed-form ground truth at frame %d: d_min=%.4f m between robot capsule '%s' and human capsule '%s'\n",
                   closest_frame, truth_here.dmin,
                   truth_here.robot_idx >= 0 ? kRobotCapsuleNames[truth_here.robot_idx] : "none",
                   truth_here.human_idx >= 0 ? kHumanCapsuleNames[truth_here.human_idx] : "none");

        constexpr float kDisplayMaxDist = 1.0f;   // m -- clamp for display; documented, fixed
        std::vector<uint8_t> gray(kNumPixels);
        for (int i = 0; i < kNumPixels; ++i) {
            float v = field[static_cast<size_t>(i)] / kDisplayMaxDist;
            v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
            // dark = close (dangerous), bright = far (safe) -- 07.09's convention.
            gray[static_cast<size_t>(i)] = static_cast<uint8_t>(v * 255.0f + 0.5f);
        }
        const std::string out_dir = project_root_from(argv[0]) + "/demo/out";
        const bool dir_ok = ensure_dir(out_dir);
        const bool pgm_ok = dir_ok && write_pgm(out_dir + "/distance_field.pgm", kImageW, kImageH, gray);

        // ---- ssm_timeline.csv: the per-frame teaching plot -------------------
        bool csv_ok = false;
        if (dir_ok) {
            std::ofstream f(out_dir + "/ssm_timeline.csv");
            csv_ok = f.is_open();
            if (csv_ok) {
                f << "# SYNTHETIC ssm timeline artifact for project 21.04 -- didactic, NOT a certified safety output\n";
                f << "frame,t_s,d_min_m,sp_full_m,sp_reduced_m,state,closest_robot_capsule\n";
                for (int fr = 0; fr < N; ++fr) {
                    const int cap = pipeline_closest_cap[static_cast<size_t>(fr)];
                    f << fr << ',' << (static_cast<float>(fr) / scenario.rate_hz) << ','
                      << pipeline_dmin[static_cast<size_t>(fr)] << ',' << sp_full << ',' << sp_reduced << ','
                      << ssm_state_name(static_cast<SsmState>(pipeline_state[static_cast<size_t>(fr)])) << ','
                      << (cap >= 0 ? kRobotCapsuleNames[cap] : "none") << '\n';
                }
            }
        }

        if (pgm_ok && csv_ok) {
            std::printf("ARTIFACT: wrote demo/out/distance_field.pgm (%dx%d) and demo/out/ssm_timeline.csv (%d rows)\n",
                       kImageW, kImageH, N);
        } else {
            std::printf("ARTIFACT: FAILED to write demo/out files\n");
        }
        std::printf("[info] artifact frame (measured closest approach): frame %d, t=%.3f s, pipeline d_min=%.4f m, closest capsule=%s\n",
                   closest_frame, static_cast<double>(closest_frame) / static_cast<double>(scenario.rate_hz),
                   static_cast<double>(pipeline_dmin[static_cast<size_t>(closest_frame)]),
                   pipeline_closest_cap[static_cast<size_t>(closest_frame)] >= 0
                       ? kRobotCapsuleNames[pipeline_closest_cap[static_cast<size_t>(closest_frame)]] : "none");
    }

    CUDA_CHECK(cudaFree(d_depth));
    CUDA_CHECK(cudaFree(d_label));
    CUDA_CHECK(cudaFree(d_block_mins));
    CUDA_CHECK(cudaFree(d_block_ids));
    CUDA_CHECK(cudaFree(d_field));

    std::printf("[time] SEQUENCE stage: %.4f ms average GPU kernel time per frame over %d frames (render + human-min-distance)\n",
               loop_gpu_ms / N, N);

    // ======================= VERIFICATION GATES ===============================
    bool all_gates_pass = true;

    // ---- GATE: NO-FALSE-STOP -------------------------------------------------
    {
        long long qualifying = 0, violations = 0;
        double tightest_margin = 1e18;
        for (int f = 0; f < N; ++f) {
            const double margin = analytic_dmin[static_cast<size_t>(f)] - static_cast<double>(sp_full);
            if (margin > kNoFalseStopMargin) {
                qualifying++;
                if (margin < tightest_margin) tightest_margin = margin;
                if (static_cast<SsmState>(pipeline_state[static_cast<size_t>(f)]) != SsmState::NORMAL)
                    violations++;
            }
        }
        const bool pass = (violations == 0) && (qualifying > 0);
        all_gates_pass = all_gates_pass && pass;
        std::printf("[info] no-false-stop: %lld/%d frames qualify (analytic d_min > Sp_full + %.3f m); %lld violation(s); tightest margin tested %.4f m\n",
                   qualifying, N, static_cast<double>(kNoFalseStopMargin), violations,
                   qualifying > 0 ? tightest_margin : 0.0);
        std::printf("GATE NO-FALSE-STOP: %s (no PROTECTIVE_STOP while the closed-form distance exceeds Sp_full by more than the documented margin)\n",
                   pass ? "PASS" : "FAIL");
    }

    // ---- GATE: NO-MISSED-STOP -------------------------------------------------
    {
        long long qualifying = 0, violations = 0;
        double tightest_gap = 1e18;
        for (int f = 0; f < N; ++f) {
            const double gap = static_cast<double>(sp_reduced) - kNoMissedStopMargin - analytic_dmin[static_cast<size_t>(f)];
            if (gap > 0.0) {
                qualifying++;
                if (gap < tightest_gap) tightest_gap = gap;
                if (static_cast<SsmState>(pipeline_state[static_cast<size_t>(f)]) != SsmState::PROTECTIVE_STOP)
                    violations++;
            }
        }
        const bool pass = (violations == 0) && (qualifying > 0);
        all_gates_pass = all_gates_pass && pass;
        std::printf("[info] no-missed-stop: %lld/%d frames qualify (analytic d_min < Sp_reduced - %.3f m); %lld violation(s); tightest gap tested %.4f m\n",
                   qualifying, N, static_cast<double>(kNoMissedStopMargin), violations,
                   qualifying > 0 ? tightest_gap : 0.0);
        std::printf("GATE NO-MISSED-STOP: %s (PROTECTIVE_STOP holds wherever the closed-form distance is below Sp_reduced by more than the documented margin)\n",
                   pass ? "PASS" : "FAIL");
    }

    // ---- GATE: TRANSITIONS (+/-1 frame of the closed-form S_p crossing) ------
    {
        // Analytic RAW crossings: frames where classify_raw(analytic_dmin)
        // changes, scanning the closed-form series directly (never the
        // pipeline's own thresholded output -- that would be circular).
        struct Crossing { int frame; SsmState old_state; SsmState new_state; };
        std::vector<Crossing> analytic_crossings;
        SsmState prev = classify_raw(static_cast<float>(analytic_dmin[0]), sp_full, sp_reduced);
        for (int f = 1; f < N; ++f) {
            const SsmState cur = classify_raw(static_cast<float>(analytic_dmin[static_cast<size_t>(f)]), sp_full, sp_reduced);
            if (cur != prev) { analytic_crossings.push_back(Crossing{ f, prev, cur }); prev = cur; }
        }

        bool pass = (analytic_crossings.size() == pipeline_transitions.size()) && !analytic_crossings.empty();
        int worst_abs_diff = 0;
        if (pass) {
            for (size_t k = 0; k < analytic_crossings.size(); ++k) {
                const Crossing& ac = analytic_crossings[k];
                const TransitionEvent& pt = pipeline_transitions[k];
                const bool escalation = static_cast<int>(pt.new_state) > static_cast<int>(pt.old_state);
                // Escalations fire on the SAME frame the raw series crosses
                // (0-frame design delay); de-escalations fire kHysteresisHoldFrames-1
                // frames LATER (HysteresisFsm::step's documented offset).
                const int predicted = escalation ? ac.frame : (ac.frame + kHysteresisHoldFrames - 1);
                const int diff = std::abs(pt.frame - predicted);
                if (diff > worst_abs_diff) worst_abs_diff = diff;
                if (diff > kTransitionFrameTolerance ||
                    pt.old_state != ac.old_state || pt.new_state != ac.new_state)
                    pass = false;
            }
        }
        all_gates_pass = all_gates_pass && pass;
        std::printf("[info] transitions: %zu analytic crossing(s), %zu pipeline transition(s); worst frame offset %d (tolerance %d)\n",
                   analytic_crossings.size(), pipeline_transitions.size(), worst_abs_diff, kTransitionFrameTolerance);
        for (size_t k = 0; k < pipeline_transitions.size(); ++k) {
            const TransitionEvent& pt = pipeline_transitions[k];
            std::printf("[info]   transition %zu: frame %d, %s -> %s\n",
                       k, pt.frame, ssm_state_name(pt.old_state), ssm_state_name(pt.new_state));
        }
        std::printf("GATE TRANSITIONS: %s (all state transitions land within +/-%d frame of the closed-form S_p crossing)\n",
                   pass ? "PASS" : "FAIL", kTransitionFrameTolerance);
    }

    // ---- GATE: D_MIN SANDWICH BOUND -------------------------------------------
    // Proven (kernels.cuh SECTION 7 derives both terms; THEORY.md "Numerical
    // considerations" gives the full argument):
    //   analytic_dmin <= pipeline_dmin
    //                  <= analytic_dmin + kPixelQuantBound + kSilhouetteSagBound
    // up to small FP32/self-filter slop (kDminBoundSlack). Checked on EVERY frame.
    {
        long long violations = 0;
        float worst_over = -1e9f, worst_under = 1e9f;
        constexpr float kFpSlack = 0.002f;   // m, FP32 rounding headroom below the proven lower bound
        const float upper_bound = kPixelQuantBound + kSilhouetteSagBound + kDminBoundSlack;
        for (int f = 0; f < N; ++f) {
            const float truth = static_cast<float>(analytic_dmin[static_cast<size_t>(f)]);
            const float measured = pipeline_dmin[static_cast<size_t>(f)];
            const float lower = truth - kFpSlack;
            const float upper = truth + upper_bound;
            if (measured < lower || measured > upper) violations++;
            const float over = measured - truth;
            if (over > worst_over) worst_over = over;
            if (over < worst_under) worst_under = over;
        }
        const bool pass = (violations == 0);
        all_gates_pass = all_gates_pass && pass;
        std::printf("[info] d_min bound: bound=[-%.4f, +%.4f] m (pixel-quant %.4f + silhouette-sag %.4f + slack %.4f) around the closed-form distance; worst observed offset [%.4f, %.4f] m over %d frames; %lld violation(s)\n",
                   static_cast<double>(kFpSlack), static_cast<double>(upper_bound),
                   static_cast<double>(kPixelQuantBound), static_cast<double>(kSilhouetteSagBound), static_cast<double>(kDminBoundSlack),
                   static_cast<double>(worst_under), static_cast<double>(worst_over), N, violations);
        std::printf("GATE D_MIN BOUND: %s (pipeline d_min stays within the documented pixel-quantization + silhouette-visibility bound of the closed-form distance on every frame)\n",
                   pass ? "PASS" : "FAIL");
    }

    // ---- verdict --------------------------------------------------------------
    if (all_gates_pass)
        std::printf("RESULT: PASS (VERIFY + all 4 gates passed)\n");
    else
        std::printf("RESULT: FAIL (one or more gates failed -- see GATE lines above)\n");
    return all_gates_pass ? 0 : 1;
}
