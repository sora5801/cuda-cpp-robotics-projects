// ===========================================================================
// main.cu — entry point for project 02.06
//           ICP: point-to-point → point-to-plane → GICP, all batched
//           (teaching core: point-to-point + point-to-plane closed-loop ICP)
//
// What this program does, start to finish
// -----------------------------------------
//   1. Print the banner + GPU info; load the two committed demo pairs
//      (data/sample/pairs_meta.csv + the four .bin point clouds).
//   2. Compute each pair's target-cloud normals ONCE (PCA + Jacobi, GPU).
//   3. VERIFY STAGE (the §5 GPU-vs-CPU gate, on pair 0 / iteration 0):
//        (a) correspondence indices match the CPU oracle EXACTLY;
//        (b) the point-to-point normal system matches within tolerance;
//        (c) the point-to-plane normal system matches within tolerance.
//   4. CLOSED LOOP: run BOTH ICP variants (point-to-point, point-to-plane)
//      on BOTH pairs — four runs total — logging pair 0's per-iteration
//      RMS to the convergence artifact.
//   5. GROUND-TRUTH GATE: every run's recovered pose must be within
//      documented rotation/translation error thresholds of the committed
//      ground truth, AND point-to-plane must converge in fewer iterations
//      than point-to-point on pair 0 (the wall-dominated scene) — the
//      taught superiority, measured, not asserted.
//   6. Write the plotting artifacts (aligned cloud, convergence curve) and
//      print the final RESULT line.
//
// The ICP update implemented here (derivation in THEORY.md §the-math):
//      x_i   = R_est * p_i + t_est                          (GPU: transform)
//      q_i   = nearest target point to x_i                   (GPU: correspond)
//      H,g   = Σ_i (per-point 6x6 contribution)              (GPU: reduce)
//      delta = solve H·delta = -g                             (host, double, 33.01-style Cholesky)
//      R_est ← Exp(delta.w) · R_est ; t_est ← t_est + delta.v (host, quaternion compose)
// This file owns everything EXCEPT the four GPU kernels (kernels.cu) and
// the CPU oracle (reference_cpu.cpp) — the SE(3) bookkeeping (quaternion
// math, the 6x6 Cholesky solve, the twist update) lives here, deliberately,
// the same way 08.01 keeps its softmin blend on the host: it is O(1) work
// per iteration, and keeping it in plain C++ next to the kernel calls puts
// the WHOLE algorithm on a few screens instead of scattered across files.
//
// Output contract (load-bearing!): stable lines are "[demo]", "PROBLEM:",
// "SCENARIO:", "VERIFY:", "CHECK:", "ARTIFACT:", "RESULT:" — "[info]"/
// "[time]" are NOT diffed (device names and measured numbers vary by
// machine/run). Change a stable line ⇒ update demo/expected_output.txt in
// the same change, and vice versa (CLAUDE.md §9).
//
// Read this first, then kernels.cuh → reference_cpu.cpp → kernels.cu.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
#include <direct.h>               // _mkdir (std::filesystem avoided in .cu — see 07.09)
#else
#include <sys/stat.h>
#endif

// ---------------------------------------------------------------------------
// Tunable constants — every one is a DETERMINISTIC compile-time value, so
// the PROBLEM/CHECK/RESULT lines that quote them are stable across
// machines. Measured behavior that justifies each threshold is recorded in
// the comment beside it (CLAUDE.md: "justify calibrated thresholds in code
// comments with measured values").
// ---------------------------------------------------------------------------

// ICP loop control.
static constexpr int   kMaxIcpIters      = 60;      // hard cap; measured convergence (both pairs, both variants; see the
                                                     // [info] lines) is point-to-plane 5-6 iterations, point-to-point
                                                     // 32-48 iterations — 60 leaves every measured run headroom to reach
                                                     // its OWN convergence threshold rather than being cut off by the cap
static constexpr float kConvDeltaRotDeg  = 0.01f;   // "converged" = the LAST twist increment's rotation part is below this...
static constexpr float kConvDeltaTransM  = 0.0001f; // ...AND its translation part is below this (0.1 mm — tiny next to the noise floor)
static constexpr double kDamping         = 1.0e-3;  // Tikhonov regularization added to H's diagonal before solving — a tiny,
                                                     // fixed safety margin against a near-singular system late in convergence
                                                     // (33.01 PRACTICE §... cites exactly this "JᵀJ + λI" pattern); negligible
                                                     // next to H's own diagonal magnitude (point-to-point's translation block
                                                     // alone accumulates ~n_valid, i.e. thousands, per pair).

// Ground-truth pose-error gate (applied to ALL four (pair, variant) runs).
// Margins are wide on purpose (CLAUDE.md: "success thresholds carry wide
// margins" — 08.01's phrase, reused here) — see the measured [info] lines
// for how far under threshold the actual runs land.
static constexpr float kGateRotErrDeg   = 1.0f;     // deg
static constexpr float kGateTransErrM   = 0.05f;    // m (5 cm — ~10x the 5 mm noise sigma)

// GPU-vs-CPU verification tolerances (§5 gate). Relative, floored at 1.0 so
// near-zero reference values do not force an unreasonably tight absolute
// check — the same shape 08.01/33.01 use.
static constexpr float kVerifyRelTolSystem = 1e-3f;   // normal-system H/g entries
static constexpr float kVerifyRelTolDist2  = 1e-4f;   // correspondence squared distances

// ---------------------------------------------------------------------------
// Small SE(3) helpers — quaternion (w,x,y,z), the repo order (CLAUDE.md
// §12), kept normalized after every update. These mirror 09.01's
// conventions but serve a different purpose: 09.01 COMPOSES a FIXED chain
// of joints; here we ITERATIVELY UPDATE a single running estimate by
// left-multiplying a small-angle increment onto it every ICP iteration.
// ---------------------------------------------------------------------------
struct Quat { float w, x, y, z; };

static Quat quat_normalize(Quat q)
{
    const float n = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    const float inv = (n > 0.0f) ? (1.0f / n) : 1.0f;   // guard: never divide by zero
    return { q.w * inv, q.x * inv, q.y * inv, q.z * inv };
}

// Hamilton product a*b — LEFT-multiplying b by a (applies a's rotation
// AFTER b's, in world/target-frame terms: the convention THEORY.md's
// linearization derives, matching "compose the new increment on the left").
static Quat quat_mul(Quat a, Quat b)
{
    return {
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w
    };
}

static void quat_to_matrix(Quat q, float R[9])
{
    const float w = q.w, x = q.x, y = q.y, z = q.z;
    R[0] = 1.0f - 2.0f * (y * y + z * z);  R[1] = 2.0f * (x * y - w * z);         R[2] = 2.0f * (x * z + w * y);
    R[3] = 2.0f * (x * y + w * z);         R[4] = 1.0f - 2.0f * (x * x + z * z);  R[5] = 2.0f * (y * z - w * x);
    R[6] = 2.0f * (x * z - w * y);         R[7] = 2.0f * (y * z + w * x);         R[8] = 1.0f - 2.0f * (x * x + y * y);
}

// axisangle_to_quat — the SE(3) exponential map's ROTATION half, exact (not
// just first-order) via Rodrigues' formula expressed through sin/cos of the
// half-angle: q = (cos(theta/2), sin(theta/2) * axis). theta = |w| may be
// LARGE on early iterations (this project's ground-truth rotations are
// 7-9 degrees, not infinitesimal), so the exact trig form — not a
// first-order (1, w/2) approximation — is used whenever theta is not
// vanishingly small (THEORY.md §numerics discusses the small-theta branch).
static Quat axisangle_to_quat(float wx, float wy, float wz)
{
    const float theta = std::sqrt(wx * wx + wy * wy + wz * wz);
    if (theta < 1e-8f) {
        // First-order fallback for a (near-)zero rotation: normalizing
        // afterward cleans up the truncation error, and this branch only
        // matters once ICP has essentially converged (theta this small).
        return quat_normalize(Quat{ 1.0f, wx * 0.5f, wy * 0.5f, wz * 0.5f });
    }
    const float half = 0.5f * theta;
    const float s = std::sin(half) / theta;   // sin(half)/theta, not sin(half)/|axis| — axis = w/theta
    return Quat{ std::cos(half), wx * s, wy * s, wz * s };
}

// rotation_angle_deg — the angle of a rotation MATRIX via its trace
// (standard identity: trace(R) = 1 + 2cos(theta)). Used for the
// ground-truth pose-error metric (main() below).
static float rotation_angle_deg(const float R[9])
{
    float c = (R[0] + R[4] + R[8] - 1.0f) * 0.5f;
    if (c > 1.0f) c = 1.0f;     // clamp against float rounding pushing |c| slightly past 1
    if (c < -1.0f) c = -1.0f;
    return std::acos(c) * (180.0f / 3.14159265358979323846f);
}

static void mat3_mul(const float A[9], const float B[9], float C[9])
{
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < 3; ++k) acc += A[i * 3 + k] * B[k * 3 + j];
            C[i * 3 + j] = acc;
        }
}

static void mat3_transpose(const float A[9], float At[9])
{
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            At[i * 3 + j] = A[j * 3 + i];
}

static void mat3_apply(const float R[9], const float v[3], float out[3])
{
    out[0] = R[0] * v[0] + R[1] * v[1] + R[2] * v[2];
    out[1] = R[3] * v[0] + R[4] * v[1] + R[5] * v[2];
    out[2] = R[6] * v[0] + R[7] * v[1] + R[8] * v[2];
}

// ---------------------------------------------------------------------------
// solve_6x6_spd — 33.01-STYLE Cholesky solve, ported from that project's
// in-register GPU kernel to a plain sequential HOST function in DOUBLE
// precision (not float): unlike 33.01's batch of thousands of independent
// small systems (where FP32 registers and massive parallelism are the
// whole point), ICP solves exactly ONE 6x6 system per iteration on the
// host — there is no performance reason to stay in float, and double
// removes any doubt about the LINEAR SOLVE'S OWN error competing with the
// least-squares system's own conditioning (THEORY.md §numerics).
//
// Same three steps as 33.01's kernel: factorize A = L·Lᵀ (lower-triangular,
// positive diagonal), forward-solve L·y=b, back-solve Lᵀ·x=y. Same NaN-on-
// non-SPD policy (33.01's kernels.cuh): a failed solve is IMPOSSIBLE to
// miss downstream rather than silently returning a plausible-looking zero.
// `damping` (kDamping above) is added to the diagonal BEFORE factorizing —
// the "JᵀJ + λI" pattern 33.01's own header comment names as the reason
// Cholesky solves exist in this repository at all.
// ---------------------------------------------------------------------------
static bool solve_6x6_spd(const double H[36], const double rhs[6], double damping, double x[6])
{
    double m[36];   // A's lower triangle on entry -> L on exit (33.01's in-place trick)
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            m[i * 6 + j] = (j <= i) ? H[i * 6 + j] : 0.0;
    for (int i = 0; i < 6; ++i) m[i * 6 + i] += damping;

    bool spd = true;
    for (int j = 0; j < 6; ++j) {
        double diag = m[j * 6 + j];
        for (int p = 0; p < j; ++p) diag -= m[j * 6 + p] * m[j * 6 + p];
        spd = spd && (diag > 0.0);
        const double ljj = std::sqrt(diag > 0.0 ? diag : 0.0);
        m[j * 6 + j] = ljj;
        const double inv_ljj = (ljj > 0.0) ? (1.0 / ljj) : 0.0;
        for (int i = j + 1; i < 6; ++i) {
            double s = m[i * 6 + j];
            for (int p = 0; p < j; ++p) s -= m[i * 6 + p] * m[j * 6 + p];
            m[i * 6 + j] = s * inv_ljj;
        }
    }

    double v[6];
    for (int i = 0; i < 6; ++i) v[i] = rhs[i];
    for (int i = 0; i < 6; ++i) {                       // forward: L y = rhs
        double s = v[i];
        for (int p = 0; p < i; ++p) s -= m[i * 6 + p] * v[p];
        v[i] = s / m[i * 6 + i];
    }
    for (int i = 5; i >= 0; --i) {                      // back: Lᵀ x = y
        double s = v[i];
        for (int p = i + 1; p < 6; ++p) s -= m[p * 6 + i] * v[p];
        v[i] = s / m[i * 6 + i];
    }

    for (int i = 0; i < 6; ++i)
        x[i] = spd ? v[i] : std::numeric_limits<double>::quiet_NaN();
    return spd;
}

// expand_h — unpack the 21-entry upper triangle (kernels.cuh's hidx layout)
// into a full symmetric 6x6 for solve_6x6_spd.
static void expand_h(const double H21[21], double Hfull[36])
{
    for (int i = 0; i < 6; ++i)
        for (int j = i; j < 6; ++j) {
            const double val = H21[hidx(i, j)];
            Hfull[i * 6 + j] = val;
            Hfull[j * 6 + i] = val;
        }
}

// ---------------------------------------------------------------------------
// Data loading: the binary cloud format + pairs_meta.csv (format documented
// byte-exactly in data/README.md and scripts/make_synthetic.py's header).
// ---------------------------------------------------------------------------
struct PairData {
    std::string name;
    std::vector<float> src_xyz;   // n_src*3
    std::vector<float> tgt_xyz;   // n_tgt*3
    int n_src = 0, n_tgt = 0;
    float gt_t[3] = { 0, 0, 0 };
    float gt_q[4] = { 1, 0, 0, 0 };   // (w,x,y,z)
};

static bool load_cloud_bin(const std::string& path, std::vector<float>& xyz, int& n)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    char magic[4] = { 0, 0, 0, 0 };
    f.read(magic, 4);
    if (f.gcount() != 4 || std::memcmp(magic, "PC01", 4) != 0) return false;

    uint32_t count = 0;
    f.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (f.gcount() != static_cast<std::streamsize>(sizeof(count))) return false;

    n = static_cast<int>(count);
    if (n < 0) return false;
    xyz.assign(static_cast<size_t>(n) * 3, 0.0f);
    const std::streamsize want = static_cast<std::streamsize>(xyz.size() * sizeof(float));
    f.read(reinterpret_cast<char*>(xyz.data()), want);
    return f.gcount() == want;   // exact byte count, not stream-flag guessing
}

// load_pairs_meta — strict row-labeled loader, same discipline as 08.01's
// load_scenario: unknown labels or short rows fail LOUDLY (CLAUDE.md §13:
// no silent fallback on malformed input).
static bool load_pairs_meta(const std::string& path, std::vector<PairData>& pairs, std::string& dir)
{
    std::ifstream in(path);
    if (!in.is_open()) return false;

    const size_t slash = path.find_last_of("/\\");
    dir = (slash == std::string::npos) ? "." : path.substr(0, slash);

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string label;
        std::getline(ss, label, ',');
        if (label != "PAIR") {
            std::fprintf(stderr, "pairs_meta: unknown row label '%s'\n", label.c_str());
            return false;
        }

        std::string cell;
        auto next = [&](const char* what) -> std::string {
            if (!std::getline(ss, cell, ',')) {
                std::fprintf(stderr, "pairs_meta: short PAIR row (missing %s)\n", what);
                cell.clear();
                return "";
            }
            return cell;
        };

        PairData pd;
        pd.name = next("name");
        const std::string src_file = next("source_file");
        const std::string tgt_file = next("target_file");
        const int meta_n_src = std::atoi(next("n_source").c_str());
        const int meta_n_tgt = std::atoi(next("n_target").c_str());
        pd.gt_q[0] = std::strtof(next("qw").c_str(), nullptr);
        pd.gt_q[1] = std::strtof(next("qx").c_str(), nullptr);
        pd.gt_q[2] = std::strtof(next("qy").c_str(), nullptr);
        pd.gt_q[3] = std::strtof(next("qz").c_str(), nullptr);
        pd.gt_t[0] = std::strtof(next("tx_m").c_str(), nullptr);
        pd.gt_t[1] = std::strtof(next("ty_m").c_str(), nullptr);
        pd.gt_t[2] = std::strtof(next("tz_m").c_str(), nullptr);
        next("noise_sigma_m");   // parsed for completeness; not used at runtime

        if (pd.name.empty() || src_file.empty() || tgt_file.empty()) {
            std::fprintf(stderr, "pairs_meta: malformed PAIR row for '%s'\n", pd.name.c_str());
            return false;
        }

        if (!load_cloud_bin(dir + "/" + src_file, pd.src_xyz, pd.n_src) ||
            !load_cloud_bin(dir + "/" + tgt_file, pd.tgt_xyz, pd.n_tgt)) {
            std::fprintf(stderr, "pairs_meta: failed to load cloud(s) for pair '%s'\n", pd.name.c_str());
            return false;
        }
        if (pd.n_src != meta_n_src || pd.n_tgt != meta_n_tgt) {
            std::fprintf(stderr, "pairs_meta: point-count mismatch for pair '%s' (meta says %d/%d, file has %d/%d)\n",
                        pd.name.c_str(), meta_n_src, meta_n_tgt, pd.n_src, pd.n_tgt);
            return false;
        }

        pairs.push_back(std::move(pd));
    }
    return !pairs.empty();
}

static std::string project_root_from(const char* argv0)
{
    std::string exe(argv0 ? argv0 : "");
    size_t cut = exe.find_last_of("/\\");
    if (cut == std::string::npos) return ".";
    return exe.substr(0, cut) + "/../../..";
}

static std::string find_meta_file(const std::string& cli_path, const char* argv0)
{
    std::vector<std::string> candidates;
    if (!cli_path.empty()) candidates.push_back(cli_path);
    candidates.push_back(project_root_from(argv0) + "/data/sample/pairs_meta.csv");
    candidates.push_back("data/sample/pairs_meta.csv");
    candidates.push_back("../data/sample/pairs_meta.csv");
    for (const auto& c : candidates)
        if (std::ifstream(c).is_open()) return c;
    return "";
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

// ---------------------------------------------------------------------------
// centroid3 — mean of an interleaved xyz cloud (host, double accumulation —
// this runs once per pair, precision is free). Feeds the point-to-plane
// normal-orientation reference (kernels.cuh: "a mostly-enclosing shell of
// surface points' centroid naturally falls in the interior").
// ---------------------------------------------------------------------------
static void centroid3(const std::vector<float>& xyz, int n, float out[3])
{
    double sx = 0.0, sy = 0.0, sz = 0.0;
    for (int i = 0; i < n; ++i) {
        sx += xyz[static_cast<size_t>(i) * 3 + 0];
        sy += xyz[static_cast<size_t>(i) * 3 + 1];
        sz += xyz[static_cast<size_t>(i) * 3 + 2];
    }
    const double inv_n = (n > 0) ? (1.0 / n) : 0.0;
    out[0] = static_cast<float>(sx * inv_n);
    out[1] = static_cast<float>(sy * inv_n);
    out[2] = static_cast<float>(sz * inv_n);
}

// ---------------------------------------------------------------------------
// ConvergenceRow — one logged iteration of the "teaching curve" artifact.
// ---------------------------------------------------------------------------
struct ConvergenceRow {
    int pair_index;
    int mode;        // IcpMode, stored as int for easy CSV printing
    int iter;
    float rms_m;
    int num_valid;
};

struct IcpResult {
    int iterations_used = 0;
    float final_rms_m = -1.0f;
    int final_num_valid = 0;
    double gpu_ms_total = 0.0;
    float t_est[3] = { 0, 0, 0 };
    float q_est[4] = { 1, 0, 0, 0 };
};

// ---------------------------------------------------------------------------
// run_icp — the closed loop described in the file header, for ONE pair and
// ONE variant. Allocates its own persistent device buffers (freed before
// returning) — called four times total by main() (2 pairs x 2 variants).
//
//   d_src_xyz    : DEVICE pointer, the pair's ORIGINAL (untransformed)
//                  source cloud — never modified; each iteration re-derives
//                  the transformed cloud from it and the CURRENT estimate,
//                  never from the previous iteration's transformed cloud
//                  (avoids compounding transform error across iterations).
//   d_tgt_xyz, d_tgt_normals : the pair's target cloud + precomputed normals
//                  (normals may be a dummy/unused pointer when mode is
//                  kPointToPoint — kernels.cuh documents this).
// ---------------------------------------------------------------------------
static IcpResult run_icp(int n_src, const float* d_src_xyz,
                         int n_tgt, const float* d_tgt_xyz, const float* d_tgt_normals,
                         IcpMode mode, float max_corr_dist_m,
                         int pair_index, std::vector<ConvergenceRow>* conv_log)
{
    IcpResult result;

    const int num_blocks = blocks_for(n_src, kThreadsReduce);
    float* d_cur_xyz = nullptr;
    int*   d_corr_idx = nullptr;
    float* d_corr_dist2 = nullptr;
    float* d_block_partials = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cur_xyz, static_cast<size_t>(n_src) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_corr_idx, static_cast<size_t>(n_src) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_corr_dist2, static_cast<size_t>(n_src) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_block_partials, static_cast<size_t>(num_blocks) * kReduceWidth * sizeof(float)));

    std::vector<int>   h_corr_idx(static_cast<size_t>(n_src));
    std::vector<float> h_corr_dist2(static_cast<size_t>(n_src));
    std::vector<float> h_block_partials(static_cast<size_t>(num_blocks) * kReduceWidth);

    float t_est[3] = { 0.0f, 0.0f, 0.0f };
    Quat  q_est = { 1.0f, 0.0f, 0.0f, 0.0f };   // identity: ICP starts from "no motion yet"

    int iter = 0;
    for (; iter < kMaxIcpIters; ++iter) {
        Rigid3 T;
        quat_to_matrix(q_est, T.R);
        T.t[0] = t_est[0]; T.t[1] = t_est[1]; T.t[2] = t_est[2];

        GpuTimer gt;
        gt.begin();
        launch_transform_cloud(n_src, d_src_xyz, T, d_cur_xyz);
        launch_find_correspondences(n_src, d_cur_xyz, n_tgt, d_tgt_xyz, max_corr_dist_m,
                                    d_corr_idx, d_corr_dist2);
        launch_build_normal_system(n_src, d_cur_xyz, d_tgt_xyz, d_tgt_normals, d_corr_idx,
                                   mode, d_block_partials);
        result.gpu_ms_total += static_cast<double>(gt.end_ms());

        CUDA_CHECK(cudaMemcpy(h_corr_idx.data(), d_corr_idx,
                              static_cast<size_t>(n_src) * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_corr_dist2.data(), d_corr_dist2,
                              static_cast<size_t>(n_src) * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_block_partials.data(), d_block_partials,
                              static_cast<size_t>(num_blocks) * kReduceWidth * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // --- Stage 2 of the reduction (kernels.cuh): sum the block
        // partials on the host, in double — the same "GPU partial reduce,
        // host finishes it" split as 08.01's softmin blend.
        double H21[21] = { 0.0 };
        double g6[6]   = { 0.0 };
        for (int b = 0; b < num_blocks; ++b) {
            const float* row = &h_block_partials[static_cast<size_t>(b) * kReduceWidth];
            for (int c = 0; c < 21; ++c) H21[c] += static_cast<double>(row[c]);
            for (int c = 0; c < 6;  ++c) g6[c]  += static_cast<double>(row[21 + c]);
        }

        // RMS + valid-match count over the CURRENT correspondences — the
        // convergence curve's per-iteration teaching signal.
        double sum_d2 = 0.0;
        int num_valid = 0;
        for (int k = 0; k < n_src; ++k) {
            if (h_corr_idx[static_cast<size_t>(k)] >= 0) {
                sum_d2 += static_cast<double>(h_corr_dist2[static_cast<size_t>(k)]);
                ++num_valid;
            }
        }
        const float rms_m = (num_valid > 0)
            ? static_cast<float>(std::sqrt(sum_d2 / static_cast<double>(num_valid)))
            : -1.0f;

        if (conv_log)
            conv_log->push_back({ pair_index, static_cast<int>(mode), iter, rms_m, num_valid });

        // --- Solve H·delta = -g (Gauss-Newton normal equations: g is
        // J^T r, NOT already negated — kernels.cuh's per-point formulas —
        // so the RHS handed to the solver is -g).
        double Hfull[36];
        expand_h(H21, Hfull);
        double rhs[6];
        for (int c = 0; c < 6; ++c) rhs[c] = -g6[c];
        double delta[6];
        const bool spd = solve_6x6_spd(Hfull, rhs, kDamping, delta);
        if (!spd) {
            // A non-SPD system (near-singular H) would mean the scene does
            // not constrain all 6 DOF at the current estimate — did not
            // happen in any measured run on this project's wall-dominated
            // scenes (kDamping's headroom), but is reported rather than
            // silently continuing with a NaN delta (CLAUDE.md §13).
            std::fprintf(stderr, "run_icp: pair %d mode %d iter %d: normal system not SPD\n",
                        pair_index, static_cast<int>(mode), iter);
            break;
        }

        const float wx = static_cast<float>(delta[0]), wy = static_cast<float>(delta[1]), wz = static_cast<float>(delta[2]);
        const float vx = static_cast<float>(delta[3]), vy = static_cast<float>(delta[4]), vz = static_cast<float>(delta[5]);

        // SE(3) update (THEORY.md §numerics names this explicitly): the
        // ROTATION composes via the exact quaternion exponential (LEFT
        // multiply — the increment is expressed in the TARGET/world frame,
        // matching the linearization); the TRANSLATION is the SIMPLER
        // additive retraction t += v (not the full coupled se(3) "V
        // matrix"), the standard simplification point-to-plane ICP
        // implementations use (Low 2004; PCL's TransformationEstimation
        // LLS) — correct here because Gauss-Newton RE-LINEARIZES every
        // iteration, self-correcting any first-order error the
        // simplification introduces. Contrast with 09.01, whose STATIC
        // joint-chain composition has no such iterative self-correction
        // and therefore needs the exact composition at every joint.
        const Quat q_delta = axisangle_to_quat(wx, wy, wz);
        q_est = quat_normalize(quat_mul(q_delta, q_est));
        t_est[0] += vx; t_est[1] += vy; t_est[2] += vz;

        result.iterations_used = iter + 1;
        result.final_rms_m = rms_m;
        result.final_num_valid = num_valid;

        const float rot_delta_deg = std::sqrt(wx * wx + wy * wy + wz * wz) * (180.0f / 3.14159265358979323846f);
        const float trans_delta_m = std::sqrt(vx * vx + vy * vy + vz * vz);
        if (rot_delta_deg < kConvDeltaRotDeg && trans_delta_m < kConvDeltaTransM) break;   // converged
    }

    result.t_est[0] = t_est[0]; result.t_est[1] = t_est[1]; result.t_est[2] = t_est[2];
    result.q_est[0] = q_est.w;  result.q_est[1] = q_est.x;  result.q_est[2] = q_est.y;  result.q_est[3] = q_est.z;

    CUDA_CHECK(cudaFree(d_cur_xyz));
    CUDA_CHECK(cudaFree(d_corr_idx));
    CUDA_CHECK(cudaFree(d_corr_dist2));
    CUDA_CHECK(cudaFree(d_block_partials));
    return result;
}

// ---------------------------------------------------------------------------
// pose_error — THEORY.md's ground-truth metric: compose the error transform
// T_err = T_est * T_gt^-1 and report its rotation ANGLE and translation
// NORM. (Naively differencing t_est - t_gt and the two quaternions'
// angle separately would double-count rotation-translation coupling —
// composing the error transform is the correct way, the same reasoning
// pose-graph SLAM literature uses for relative pose error.)
// ---------------------------------------------------------------------------
static void pose_error(const float q_est_arr[4], const float t_est[3],
                       const float q_gt_arr[4], const float t_gt[3],
                       float& rot_err_deg, float& trans_err_m)
{
    float R_est[9], R_gt[9], R_gt_T[9], R_err[9];
    quat_to_matrix(Quat{ q_est_arr[0], q_est_arr[1], q_est_arr[2], q_est_arr[3] }, R_est);
    quat_to_matrix(Quat{ q_gt_arr[0], q_gt_arr[1], q_gt_arr[2], q_gt_arr[3] }, R_gt);
    mat3_transpose(R_gt, R_gt_T);
    mat3_mul(R_est, R_gt_T, R_err);          // R_err = R_est * R_gt^T

    float Rerr_tgt[3];
    mat3_apply(R_err, t_gt, Rerr_tgt);       // R_err * t_gt
    const float t_err[3] = {
        t_est[0] - Rerr_tgt[0],
        t_est[1] - Rerr_tgt[1],
        t_est[2] - Rerr_tgt[2],
    };

    rot_err_deg = rotation_angle_deg(R_err);
    trans_err_m = std::sqrt(t_err[0] * t_err[0] + t_err[1] * t_err[1] + t_err[2] * t_err[2]);
}

static const char* mode_name(IcpMode m) { return (m == kPointToPoint) ? "point-to-point" : "point-to-plane"; }

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    std::string meta_path_arg;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--data") && i + 1 < argc) {
            meta_path_arg = argv[++i];
        } else {
            std::fprintf(stderr, "usage: %s [--data pairs_meta.csv]\n", argv[0]);
            return 2;
        }
    }

    std::printf("[demo] ICP point-to-point / point-to-plane registration (GICP documented) — project 02.06\n");
    print_device_info();
    std::printf("PROBLEM: brute-force GPU correspondence search + linearized Gauss-Newton ICP, "
               "max_iters=%d, corr_gate=%.1f m, damping=%.0e, FP32 kernels / FP64 host reduction\n",
               kMaxIcpIters, static_cast<double>(kDefaultMaxCorrDist), kDamping);

    // ---- load data -----------------------------------------------------------
    const std::string meta_path = find_meta_file(meta_path_arg, argv[0]);
    if (meta_path.empty()) {
        std::printf("SCENARIO: NOT FOUND — data/sample/pairs_meta.csv missing (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (data missing)\n");
        return 1;
    }
    std::printf("[info] metadata file: %s\n", meta_path.c_str());

    std::vector<PairData> pairs;
    std::string data_dir;
    if (!load_pairs_meta(meta_path, pairs, data_dir) || pairs.size() != 2) {
        std::printf("SCENARIO: MALFORMED — see stderr\n");
        std::printf("RESULT: FAIL (data malformed)\n");
        return 1;
    }
    for (const auto& pd : pairs) {
        const float t_mag = std::sqrt(pd.gt_t[0] * pd.gt_t[0] + pd.gt_t[1] * pd.gt_t[1] + pd.gt_t[2] * pd.gt_t[2]);
        std::printf("SCENARIO: %s: N_source=%d N_target=%d synthetic room (floor+2 walls+box), "
                   "ground-truth |t|=%.3f m [synthetic]\n",
                   pd.name.c_str(), pd.n_src, pd.n_tgt, static_cast<double>(t_mag));
    }

    // ---- upload clouds, compute target normals (ONCE per pair) ---------------
    struct PairDevice {
        float* d_src_xyz = nullptr;
        float* d_tgt_xyz = nullptr;
        float* d_tgt_normals = nullptr;
    };
    std::vector<PairDevice> dev(pairs.size());

    for (size_t p = 0; p < pairs.size(); ++p) {
        const PairData& pd = pairs[p];
        PairDevice& dv = dev[p];
        CUDA_CHECK(cudaMalloc(&dv.d_src_xyz, static_cast<size_t>(pd.n_src) * 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dv.d_tgt_xyz, static_cast<size_t>(pd.n_tgt) * 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dv.d_tgt_normals, static_cast<size_t>(pd.n_tgt) * 3 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dv.d_src_xyz, pd.src_xyz.data(),
                              static_cast<size_t>(pd.n_src) * 3 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dv.d_tgt_xyz, pd.tgt_xyz.data(),
                              static_cast<size_t>(pd.n_tgt) * 3 * sizeof(float), cudaMemcpyHostToDevice));

        float ref_point[3];
        centroid3(pd.tgt_xyz, pd.n_tgt, ref_point);   // orientation reference: target cloud's own centroid

        GpuTimer gt;
        gt.begin();
        launch_estimate_normals(pd.n_tgt, dv.d_tgt_xyz, ref_point, dv.d_tgt_normals);
        const float normals_ms = gt.end_ms();
        std::printf("[info] %s: target normals (PCA k=%d, %d Jacobi sweeps): %.2f ms\n",
                   pd.name.c_str(), kPcaK, kJacobiSweeps, static_cast<double>(normals_ms));
    }

    // ======================= VERIFY STAGE ====================================
    // Pair 0, identity transform (iteration 0's exact situation): GPU vs
    // CPU on correspondence indices AND both variants' normal systems.
    bool verify_pass = true;
    {
        const PairData& pd = pairs[0];
        const PairDevice& dv = dev[0];
        const int n = pd.n_src, m = pd.n_tgt;

        // Identity transform: "cur" cloud IS the source cloud.
        Rigid3 I{ { 1,0,0, 0,1,0, 0,0,1 }, { 0,0,0 } };
        float* d_cur = nullptr;
        CUDA_CHECK(cudaMalloc(&d_cur, static_cast<size_t>(n) * 3 * sizeof(float)));
        launch_transform_cloud(n, dv.d_src_xyz, I, d_cur);

        int*   d_idx = nullptr;
        float* d_d2  = nullptr;
        CUDA_CHECK(cudaMalloc(&d_idx, static_cast<size_t>(n) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_d2,  static_cast<size_t>(n) * sizeof(float)));
        launch_find_correspondences(n, d_cur, m, dv.d_tgt_xyz, kDefaultMaxCorrDist, d_idx, d_d2);

        std::vector<int>   h_idx_gpu(static_cast<size_t>(n)), h_idx_cpu(static_cast<size_t>(n));
        std::vector<float> h_d2_gpu(static_cast<size_t>(n)), h_d2_cpu(static_cast<size_t>(n));
        CUDA_CHECK(cudaMemcpy(h_idx_gpu.data(), d_idx, static_cast<size_t>(n) * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_d2_gpu.data(), d_d2, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost));

        find_correspondences_cpu(n, pd.src_xyz.data(), m, pd.tgt_xyz.data(), kDefaultMaxCorrDist,
                                 h_idx_cpu.data(), h_d2_cpu.data());

        int mismatches = 0;
        float worst_d2_rel = 0.0f;
        for (int k = 0; k < n; ++k) {
            if (h_idx_gpu[static_cast<size_t>(k)] != h_idx_cpu[static_cast<size_t>(k)]) ++mismatches;
            const float scale = std::fabs(h_d2_cpu[static_cast<size_t>(k)]) > 1.0f ? std::fabs(h_d2_cpu[static_cast<size_t>(k)]) : 1.0f;
            const float rel = std::fabs(h_d2_gpu[static_cast<size_t>(k)] - h_d2_cpu[static_cast<size_t>(k)]) / scale;
            if (rel > worst_d2_rel) worst_d2_rel = rel;
        }
        const bool corr_pass = (mismatches == 0) && (worst_d2_rel <= kVerifyRelTolDist2);
        std::printf("[info] correspondence check: %d/%d index mismatches, worst relative dist2 deviation %.3e\n",
                   mismatches, n, static_cast<double>(worst_d2_rel));
        std::printf("VERIFY: %s (correspondence indices match CPU reference exactly; dist2 within rel tol %.0e)\n",
                   corr_pass ? "PASS" : "FAIL", static_cast<double>(kVerifyRelTolDist2));
        verify_pass = verify_pass && corr_pass;

        // Normal-system check, both modes, using these SAME correspondences.
        // Download the target normals ONCE (point-to-plane's CPU twin needs
        // the real per-point normals; point-to-point's twin ignores the
        // pointer entirely — kernels.cuh documents both signatures).
        std::vector<float> h_normals(static_cast<size_t>(m) * 3);
        CUDA_CHECK(cudaMemcpy(h_normals.data(), dv.d_tgt_normals, h_normals.size() * sizeof(float), cudaMemcpyDeviceToHost));

        for (IcpMode mode : { kPointToPoint, kPointToPlane }) {
            const int num_blocks = blocks_for(n, kThreadsReduce);
            float* d_bp = nullptr;
            CUDA_CHECK(cudaMalloc(&d_bp, static_cast<size_t>(num_blocks) * kReduceWidth * sizeof(float)));
            launch_build_normal_system(n, d_cur, dv.d_tgt_xyz, dv.d_tgt_normals, d_idx, mode, d_bp);

            std::vector<float> h_bp(static_cast<size_t>(num_blocks) * kReduceWidth);
            CUDA_CHECK(cudaMemcpy(h_bp.data(), d_bp, h_bp.size() * sizeof(float), cudaMemcpyDeviceToHost));
            double H_gpu[21] = { 0.0 }, g_gpu[6] = { 0.0 };
            for (int b = 0; b < num_blocks; ++b) {
                const float* row = &h_bp[static_cast<size_t>(b) * kReduceWidth];
                for (int c = 0; c < 21; ++c) H_gpu[c] += static_cast<double>(row[c]);
                for (int c = 0; c < 6;  ++c) g_gpu[c] += static_cast<double>(row[21 + c]);
            }

            double H_cpu[21], g_cpu[6];
            build_normal_system_cpu(n, pd.src_xyz.data(), pd.tgt_xyz.data(), h_normals.data(),
                                    h_idx_cpu.data(), mode, H_cpu, g_cpu);

            float worst_rel = 0.0f;
            for (int c = 0; c < 21; ++c) {
                const double scale = std::fabs(H_cpu[c]) > 1.0 ? std::fabs(H_cpu[c]) : 1.0;
                const float rel = static_cast<float>(std::fabs(H_gpu[c] - H_cpu[c]) / scale);
                if (rel > worst_rel) worst_rel = rel;
            }
            for (int c = 0; c < 6; ++c) {
                const double scale = std::fabs(g_cpu[c]) > 1.0 ? std::fabs(g_cpu[c]) : 1.0;
                const float rel = static_cast<float>(std::fabs(g_gpu[c] - g_cpu[c]) / scale);
                if (rel > worst_rel) worst_rel = rel;
            }
            const bool sys_pass = worst_rel <= kVerifyRelTolSystem;
            std::printf("[info] %s normal system: worst relative H/g deviation %.3e (%d valid correspondences)\n",
                       mode_name(mode), static_cast<double>(worst_rel), n);
            std::printf("VERIFY: %s (%s normal system matches CPU reference within rel tol %.0e)\n",
                       sys_pass ? "PASS" : "FAIL", mode_name(mode), static_cast<double>(kVerifyRelTolSystem));
            verify_pass = verify_pass && sys_pass;

            CUDA_CHECK(cudaFree(d_bp));
        }

        CUDA_CHECK(cudaFree(d_cur));
        CUDA_CHECK(cudaFree(d_idx));
        CUDA_CHECK(cudaFree(d_d2));
    }
    if (!verify_pass) {
        std::printf("RESULT: FAIL (GPU/CPU disagreement — fix before trusting the ICP loop)\n");
        return 1;
    }

    // ======================= CLOSED LOOP, ALL FOUR RUNS =======================
    std::vector<ConvergenceRow> conv_log;
    IcpResult results[2][2];   // [pair_index][mode]
    for (size_t p = 0; p < pairs.size(); ++p) {
        for (IcpMode mode : { kPointToPoint, kPointToPlane }) {
            IcpResult r = run_icp(pairs[p].n_src, dev[p].d_src_xyz,
                                  pairs[p].n_tgt, dev[p].d_tgt_xyz, dev[p].d_tgt_normals,
                                  mode, kDefaultMaxCorrDist,
                                  static_cast<int>(p), (p == 0) ? &conv_log : nullptr);
            float rot_err_deg = 0.0f, trans_err_m = 0.0f;
            pose_error(r.q_est, r.t_est, pairs[p].gt_q, pairs[p].gt_t, rot_err_deg, trans_err_m);
            std::printf("[info] %s %s: %d iterations, final RMS=%.4f m (%d valid), "
                       "rot_err=%.4f deg, trans_err=%.4f m, GPU %.2f ms total (%.3f ms/iter)\n",
                       pairs[p].name.c_str(), mode_name(mode), r.iterations_used,
                       static_cast<double>(r.final_rms_m), r.final_num_valid,
                       static_cast<double>(rot_err_deg), static_cast<double>(trans_err_m),
                       r.gpu_ms_total, r.gpu_ms_total / (r.iterations_used > 0 ? r.iterations_used : 1));
            results[p][mode] = r;
        }
    }

    // ======================= GROUND-TRUTH GATE ================================
    bool gate_pass = true;
    for (size_t p = 0; p < pairs.size(); ++p) {
        for (IcpMode mode : { kPointToPoint, kPointToPlane }) {
            float rot_err_deg = 0.0f, trans_err_m = 0.0f;
            pose_error(results[p][mode].q_est, results[p][mode].t_est, pairs[p].gt_q, pairs[p].gt_t,
                      rot_err_deg, trans_err_m);
            const bool ok = (rot_err_deg < kGateRotErrDeg) && (trans_err_m < kGateTransErrM);
            std::printf("CHECK: %s (%s: rot_err < %.2f deg AND trans_err < %.3f m)\n",
                       ok ? "PASS" : "FAIL",
                       (pairs[p].name + " " + mode_name(mode)).c_str(),
                       static_cast<double>(kGateRotErrDeg), static_cast<double>(kGateTransErrM));
            gate_pass = gate_pass && ok;
        }
    }
    const bool plane_faster = results[0][kPointToPlane].iterations_used < results[0][kPointToPoint].iterations_used;
    std::printf("CHECK: %s (point-to-plane converges in fewer iterations than point-to-point on pair0, the wall-dominated scene)\n",
               plane_faster ? "PASS" : "FAIL");
    gate_pass = gate_pass && plane_faster;

    // ---- artifacts -------------------------------------------------------------
    const std::string out_dir = project_root_from(argv[0]) + "/demo/out";
    bool artifacts_ok = ensure_dir(out_dir);

    if (artifacts_ok) {
        // aligned.csv: pair0's point-to-plane result — subsampled aligned
        // source vs subsampled target, for plotting.
        const PairData& pd = pairs[0];
        Rigid3 T;
        quat_to_matrix(Quat{ results[0][kPointToPlane].q_est[0], results[0][kPointToPlane].q_est[1],
                            results[0][kPointToPlane].q_est[2], results[0][kPointToPlane].q_est[3] }, T.R);
        T.t[0] = results[0][kPointToPlane].t_est[0];
        T.t[1] = results[0][kPointToPlane].t_est[1];
        T.t[2] = results[0][kPointToPlane].t_est[2];

        std::vector<float> aligned(pd.src_xyz.size());
        transform_cloud_cpu(pd.n_src, pd.src_xyz.data(), T, aligned.data());   // reuse the CPU oracle map — see reference_cpu.cpp

        std::ofstream f(out_dir + "/aligned.csv");
        if (f.is_open()) {
            f << "cloud,x_m,y_m,z_m\n";
            const int stride_src = std::max(1, pd.n_src / 1500);
            for (int k = 0; k < pd.n_src; k += stride_src)
                f << "aligned," << aligned[static_cast<size_t>(k) * 3 + 0] << ',' << aligned[static_cast<size_t>(k) * 3 + 1] << ',' << aligned[static_cast<size_t>(k) * 3 + 2] << '\n';
            const int stride_tgt = std::max(1, pd.n_tgt / 1500);
            for (int k = 0; k < pd.n_tgt; k += stride_tgt)
                f << "target," << pd.tgt_xyz[static_cast<size_t>(k) * 3 + 0] << ',' << pd.tgt_xyz[static_cast<size_t>(k) * 3 + 1] << ',' << pd.tgt_xyz[static_cast<size_t>(k) * 3 + 2] << '\n';
            std::printf("ARTIFACT: wrote demo/out/aligned.csv (pair0, point-to-plane result)\n");
        } else {
            artifacts_ok = false;
        }
    }
    if (artifacts_ok) {
        std::ofstream f(out_dir + "/convergence.csv");
        if (f.is_open()) {
            f << "pair,mode,iter,rms_m,num_valid\n";
            for (const auto& row : conv_log)
                f << row.pair_index << ',' << mode_name(static_cast<IcpMode>(row.mode)) << ','
                  << row.iter << ',' << row.rms_m << ',' << row.num_valid << '\n';
            // Row count is NOT printed here: it is the ICP convergence path
            // length (iterations_used, summed over both variants) — a
            // MEASURED quantity, not a fixed input parameter, so it stays
            // out of this stable line for the same reason 08.01 keeps
            // trajectory numbers out of its stable output (THEORY.md
            // §numerics; [info] lines above carry the actual counts).
            std::printf("[info] convergence.csv rows: %zu (pair0, both variants, one row per ICP iteration)\n", conv_log.size());
            std::printf("ARTIFACT: wrote demo/out/convergence.csv (pair0, both variants)\n");
        } else {
            artifacts_ok = false;
        }
    }
    if (!artifacts_ok)
        std::printf("ARTIFACT: FAILED to write demo/out/ files\n");

    // ---- cleanup ----------------------------------------------------------------
    for (auto& dv : dev) {
        CUDA_CHECK(cudaFree(dv.d_src_xyz));
        CUDA_CHECK(cudaFree(dv.d_tgt_xyz));
        CUDA_CHECK(cudaFree(dv.d_tgt_normals));
    }

    const bool success = gate_pass && artifacts_ok;
    if (success)
        std::printf("RESULT: PASS (verify passed; all 4 runs met the pose-error gate; "
                   "point-to-plane converged faster than point-to-point on pair0)\n");
    else
        std::printf("RESULT: FAIL (see CHECK/ARTIFACT lines above)\n");
    return success ? 0 : 1;
}
