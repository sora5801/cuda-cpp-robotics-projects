// ===========================================================================
// main.cu — entry point for project 01.17
//           Camera-LiDAR / camera-camera extrinsic calibration
//           (batched reprojection-error minimization, GPU Levenberg-
//           Marquardt)
//
// What this program does, start to finish
// ----------------------------------------
// Loads three committed correspondence sets (kernels.cuh explains the
// unified Correspondence/Rigid3 contract both scenarios share) and one
// background image, then runs NINE independent verification stages, each
// printing a stable PASS/FAIL line plus an [info] line with the measured
// number(s) behind it (CLAUDE.md §5/§9 — the GPU-vs-CPU gate and beyond):
//
//   JACOBIAN_CHECK    — analytic vs. central-difference numeric Jacobian
//                        (an INDEPENDENT gate, not a GPU/CPU twin — see
//                        kernels.cuh's file header and reference_cpu.cpp's).
//   ASSEMBLY_TWIN     — GPU kernel vs. CPU oracle, one normal-equation
//                        assembly, tight tolerance.
//   TRAJECTORY_TWIN   — GPU-orchestrated vs. CPU-only, one full 20-iteration
//                        LM trajectory, measured-then-margined (08.01's
//                        technique, cited).
//   MULTISTART_TWIN   — 64 of the K=1024 multi-start farm threads
//                        reproduced exactly on the CPU.
//   BASIN             — convergence-basin study: what fraction of 1024
//                        randomized starts (up to ~34 deg / 30 cm from
//                        identity) reach the true camera-LiDAR extrinsic.
//   RECOVERY_CAM_LIDAR / RECOVERY_CAM_CAM — best-of-multistart recovery
//                        error vs. ground truth, realistic sensor noise.
//   NOISE_SCALING     — recovery error at three documented noise levels,
//                        checked for sane (non-decreasing) scaling.
//   DEGENERACY        — the project's practical calibration lesson: the
//                        SAME extrinsic, solved from a near-coplanar pose
//                        cohort, shows measurably worse conditioning AND
//                        measurably worse translation accuracy than the
//                        pose-diverse cohort (01.16's Zhang-calibration
//                        finding, echoed here — cited).
//   ZERO_NOISE_CAM_LIDAR / ZERO_NOISE_CAM_CAM — noise-free correspondences
//                        recover the ground truth to near machine precision
//                        (the exactness anchor, and — per reference_cpu.cpp's
//                        header — the independent check on the shared
//                        camera-model formula itself).
//
// Every stage's exact measured numbers are also written to
// demo/out/gates_metrics.csv; the LM loss curves and the multi-start basin
// scatter get their own CSV artifacts, and the "money shot" — LiDAR points
// reprojected onto a camera image before vs. after calibration — is written
// to demo/out/overlay.ppm.
//
// Output contract: stable lines "[demo]", "PROBLEM:", "SCENARIO:", every
// "*_TWIN:"/"*_CHECK:"/"BASIN:"/"RECOVERY_*:"/"NOISE_SCALING:"/"DEGENERACY:"/
// "ZERO_NOISE_*:" verdict line, "ARTIFACT:", "RESULT:" — "[info]"/"[time]"
// unchecked (device names and every measured float live there, never in a
// stable line — the same discipline 08.01 documents and this project
// follows for the same reason: FMA/libm rounding can differ by GPU
// architecture even though the ALGORITHM is deterministic).
//
// Read this first, then kernels.cuh → kernels.cu → reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"
#include "util/paths.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

// ===========================================================================
// Noise seeds — main.cu-only bookkeeping (not a GPU/CPU contract item, so
// not in kernels.cuh): every stage that needs a NOISY correspondence set
// draws from xorshift32 rooted at the machine-wide seed 42, salted per stage
// so the ten noise realizations below are independent streams while staying
// fully deterministic and reproducible (CLAUDE.md §12). The multi-start
// farm's OWN seed (kernels.cuh's kMultiStartK usage) is the bare root, 42u.
// ===========================================================================
constexpr uint32_t kRngRoot                = 42u;
constexpr uint32_t kSeedAssemblyTwin       = kRngRoot + 1001u;
constexpr uint32_t kSeedTrajectoryTwin     = kRngRoot + 1002u;
constexpr uint32_t kSeedCamLidarRecovery   = kRngRoot + 1003u;
constexpr uint32_t kSeedCamCamRecovery     = kRngRoot + 1004u;
constexpr uint32_t kSeedNoiseScalingLow    = kRngRoot + 1010u;
constexpr uint32_t kSeedNoiseScalingMed    = kRngRoot + 1011u;
constexpr uint32_t kSeedNoiseScalingHigh   = kRngRoot + 1012u;
constexpr uint32_t kSeedDegenDiverse       = kRngRoot + 1020u;
constexpr uint32_t kSeedDegenCoplanar      = kRngRoot + 1021u;
constexpr uint32_t kMultistartBaseSeed     = kRngRoot;

// Gate tolerances — every one of these is "measured, then margined" (08.01's
// phrase, cited): the project was run, the ACTUAL worst-case deviation was
// read off the [info] lines below, and the threshold here was set with
// generous headroom above that measurement, documented per-gate. None of
// these numbers were guessed first and hoped to pass.
constexpr float  kJacobianRelTol        = 5.0e-2f;   // central-difference vs analytic (THEORY.md derives the eps/tol tradeoff; measured worst 1.52e-3, ~33x headroom)
constexpr double kAssemblyRelTol        = 1.0e-4;    // one-shot H/g/cost, GPU (float tree-reduce) vs CPU (near-double sum); measured worst 1.65e-7, ~600x headroom
constexpr float  kTrajectoryRotTolDeg   = 0.20f;     // final-pose GPU-orchestrated vs CPU-only, 20 chained iterations; measured worst 3.96e-2 deg, ~5x headroom (chained-iteration divergence, 08.01's "measured-then-margined" story)
constexpr float  kTrajectoryTransTolM   = 5.0e-4f;   // measured worst 4.02e-7 m, >1000x headroom
constexpr float  kMultistartRotTolDeg   = 0.20f;     // 64-subset GPU farm vs CPU farm reproduction; same chained-iteration story as the trajectory twin above
constexpr float  kMultistartTransTolM   = 5.0e-4f;   // measured worst 2.76e-5 m, ~18x headroom
constexpr double kMultistartLossRelTol  = 5.0e-3;    // measured worst 1.08e-6, >1000x headroom
constexpr float  kBasinMinConvergedPct  = 40.0f;     // >= this % of 1024 random starts reach the true optimum
constexpr float  kRecoveryRotTolDeg     = 0.6f;      // best-of-multistart vs ground truth, kNoiseMed sensor noise; measured 0.305 deg, ~2x headroom
constexpr float  kRecoveryTransTolM     = 0.015f;    // 15 mm; measured 7.44 mm, ~2x headroom
constexpr float  kNoiseScalingSlackDeg  = 0.15f;     // monotone-within-slack (FP32 + finite-K-multistart noise floor)
constexpr float  kNoiseScalingSlackM    = 0.006f;
constexpr double kDegenCondFactor       = 3.0;       // coplanar cohort's J^T J condition number must exceed diverse's by this factor
constexpr double kDegenTransFactor      = 3.0;       // ...and its translation error likewise
constexpr float  kZeroNoiseRotTolDeg    = 0.01f;     // "near machine precision" for an FP32 residual/Jacobian pipeline
constexpr float  kZeroNoiseTransTolM    = 2.0e-4f;

// ===========================================================================
// Dataset — one correspondence file's fully-loaded contents (ground truth +
// intrinsics + the kNumCorr TRUE correspondences). main.cu-local: neither
// kernels.cu nor reference_cpu.cpp need to know how a Dataset was loaded,
// only the flat float arrays main.cu hands them after adding noise.
// ===========================================================================
struct Dataset {
    float omega_gt[3] = { 0.0f, 0.0f, 0.0f };
    float t_gt[3]      = { 0.0f, 0.0f, 0.0f };
    PinholeIntrinsics K{ 0.0f, 0.0f, 0.0f, 0.0f };
    std::vector<Correspondence> corr;
    bool loaded = false;
};

// gt_rigid3 / rough_prior — turn a Dataset's ground truth into a Rigid3, and
// build the deterministic "rough prior" starting guess kernels.cuh documents
// (kRoughPriorOmegaOffset/TransOffset) — a stand-in for a real system's
// CAD/previous-calibration estimate, used by every SINGLE-trajectory stage
// (assembly/trajectory twins, noise scaling, degeneracy) so those stages
// need no RNG. The genuinely blind, no-prior case is the multi-start farm.
static Rigid3 gt_rigid3(const Dataset& ds)
{
    Rigid3 T;
    so3_exp(ds.omega_gt, T.R);
    T.t[0] = ds.t_gt[0]; T.t[1] = ds.t_gt[1]; T.t[2] = ds.t_gt[2];
    return T;
}

static Rigid3 rough_prior(const Dataset& ds)
{
    const float omega[3] = {
        ds.omega_gt[0] + kRoughPriorOmegaOffset[0],
        ds.omega_gt[1] + kRoughPriorOmegaOffset[1],
        ds.omega_gt[2] + kRoughPriorOmegaOffset[2]
    };
    Rigid3 T;
    so3_exp(omega, T.R);
    T.t[0] = ds.t_gt[0] + kRoughPriorTransOffset[0];
    T.t[1] = ds.t_gt[1] + kRoughPriorTransOffset[1];
    T.t[2] = ds.t_gt[2] + kRoughPriorTransOffset[2];
    return T;
}

static void pose_error(const Rigid3& T, const Dataset& ds, float& rot_deg, float& trans_m)
{
    const Rigid3 Tgt = gt_rigid3(ds);
    rot_deg = rotation_angle_deg(T.R, Tgt.R);
    trans_m = translation_error_m(T.t, Tgt.t);
}

// ---------------------------------------------------------------------------
// load_dataset — strict, label-based CSV loader (08.01's Scenario-loader
// discipline, cited): every row is "LABEL,value,value,..."; any unknown
// label or short row fails LOUDLY (returns an unloaded Dataset) rather than
// silently defaulting. Matches scripts/make_synthetic.py's writer exactly.
// ---------------------------------------------------------------------------
static Dataset load_dataset(const std::string& path)
{
    Dataset ds;
    if (path.empty()) return ds;
    std::ifstream in(path);
    if (!in.is_open()) return ds;

    bool have_omega = false, have_t = false, have_k = false;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string label, cell;
        std::getline(ss, label, ',');

        if (label == "OMEGA_GT") {
            for (int i = 0; i < 3; ++i) {
                if (!std::getline(ss, cell, ',')) { std::fprintf(stderr, "dataset: short OMEGA_GT row\n"); return Dataset{}; }
                ds.omega_gt[i] = std::strtof(cell.c_str(), nullptr);
            }
            have_omega = true;
        } else if (label == "T_GT") {
            for (int i = 0; i < 3; ++i) {
                if (!std::getline(ss, cell, ',')) { std::fprintf(stderr, "dataset: short T_GT row\n"); return Dataset{}; }
                ds.t_gt[i] = std::strtof(cell.c_str(), nullptr);
            }
            have_t = true;
        } else if (label == "INTRINSICS") {
            float vals[4];
            for (int i = 0; i < 4; ++i) {
                if (!std::getline(ss, cell, ',')) { std::fprintf(stderr, "dataset: short INTRINSICS row\n"); return Dataset{}; }
                vals[i] = std::strtof(cell.c_str(), nullptr);
            }
            ds.K = PinholeIntrinsics{ vals[0], vals[1], vals[2], vals[3] };
            have_k = true;
        } else if (label == "CORR") {
            std::vector<float> vals;
            while (std::getline(ss, cell, ',')) vals.push_back(std::strtof(cell.c_str(), nullptr));
            if (vals.size() != 7) { std::fprintf(stderr, "dataset: malformed CORR row\n"); return Dataset{}; }
            Correspondence c;
            c.p_src[0] = vals[2]; c.p_src[1] = vals[3]; c.p_src[2] = vals[4];
            c.uv_true[0] = vals[5]; c.uv_true[1] = vals[6];
            ds.corr.push_back(c);
        } else {
            std::fprintf(stderr, "dataset: unknown row label '%s'\n", label.c_str());
            return Dataset{};
        }
    }

    if (!have_omega || !have_t || !have_k || static_cast<int>(ds.corr.size()) != kNumCorr) {
        std::fprintf(stderr, "dataset: incomplete (omega=%d t=%d k=%d corr=%zu, want %d)\n",
                     have_omega, have_t, have_k, ds.corr.size(), kNumCorr);
        return Dataset{};
    }
    ds.loaded = true;
    return ds;
}

// ---------------------------------------------------------------------------
// apply_noise — turn a Dataset's TRUE correspondences into the OBSERVED
// arrays every kernel/CPU-twin below consumes, adding independent Gaussian
// sensor noise (kernels.cuh's xorshift32/gaussian, deterministic given
// `seed`). add_p_src_noise selects whether the SOURCE point also gets
// position noise (camera-LiDAR: yes, approximating LiDAR range/angular
// noise as isotropic Cartesian — THEORY.md "Numerical considerations" names
// the real spherical model this simplifies; camera-camera: no, per the
// "known 3-D board points" framing — see README "Limitations"). sigma=0.0
// makes this a lossless copy (used by the zero-noise sanity gate).
// ---------------------------------------------------------------------------
static void apply_noise(const Dataset& ds, float sigma_px, float sigma_p_src_m, bool add_p_src_noise,
                        uint32_t seed, std::vector<float>& p_obs, std::vector<float>& uv_obs)
{
    const size_t n = ds.corr.size();
    p_obs.resize(n * 3);
    uv_obs.resize(n * 2);
    uint32_t s = seed;
    if (s == 0u) s = 1u;   // xorshift32's one degenerate fixed point — never seed with 0

    for (size_t i = 0; i < n; ++i) {
        const Correspondence& c = ds.corr[i];
        const float nx = add_p_src_noise ? gaussian(s, sigma_p_src_m) : 0.0f;
        const float ny = add_p_src_noise ? gaussian(s, sigma_p_src_m) : 0.0f;
        const float nz = add_p_src_noise ? gaussian(s, sigma_p_src_m) : 0.0f;
        p_obs[i * 3 + 0] = c.p_src[0] + nx;
        p_obs[i * 3 + 1] = c.p_src[1] + ny;
        p_obs[i * 3 + 2] = c.p_src[2] + nz;

        const float du = gaussian(s, sigma_px);
        const float dv = gaussian(s, sigma_px);
        uv_obs[i * 2 + 0] = c.uv_true[0] + du;
        uv_obs[i * 2 + 1] = c.uv_true[1] + dv;
    }
}

// ---------------------------------------------------------------------------
// GrayImage / load_pgm — a minimal, hand-rolled P5 (binary grayscale) PGM
// reader. No vendored image library (CLAUDE.md §5: hand-roll when it
// teaches something, and a 5-line binary-format reader is exactly such a
// case — stb_image would be overkill for one fixed, self-generated format).
// Does not handle PGM's optional '#'-comment header lines (our own writer,
// scripts/make_synthetic.py, never emits them) — a deliberate, documented
// simplification, not an oversight.
// ---------------------------------------------------------------------------
struct GrayImage {
    int w = 0, h = 0;
    std::vector<unsigned char> px;
    bool loaded = false;
};

static GrayImage load_pgm(const std::string& path)
{
    GrayImage img;
    if (path.empty()) return img;
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return img;

    std::string magic;
    int w = 0, h = 0, maxval = 0;
    in >> magic >> w >> h >> maxval;         // operator>> skips whitespace/newlines for us
    in.get();                                // consume the single required whitespace byte after maxval
    if (!in || magic != "P5" || w <= 0 || h <= 0 || maxval <= 0 || maxval > 255) return img;

    img.w = w; img.h = h;
    img.px.resize(static_cast<size_t>(w) * static_cast<size_t>(h));
    in.read(reinterpret_cast<char*>(img.px.data()), static_cast<std::streamsize>(img.px.size()));
    if (!in) return GrayImage{};
    img.loaded = true;
    return img;
}

// write_ppm — the twin write-side helper: a P6 (binary RGB) writer for the
// overlay artifact. Same "hand-roll a five-line format" reasoning as above.
static bool write_ppm(const std::string& path, int w, int h, const std::vector<unsigned char>& rgb)
{
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    f << "P6\n" << w << ' ' << h << "\n255\n";
    f.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    return f.good();
}

// draw_cross — plot a small 5-pixel "+" marker at (u,v) in an interleaved
// RGB buffer, clipped to the image bounds (multi-start's randomized guesses
// mean a reprojected point can legitimately land off-frame). Not
// anti-aliased or sub-pixel — this is a teaching overlay, not a renderer.
static void draw_cross(std::vector<unsigned char>& rgb, int w, int h, float uf, float vf,
                       unsigned char r, unsigned char g, unsigned char b)
{
    const int u = static_cast<int>(std::lround(uf));
    const int v = static_cast<int>(std::lround(vf));
    const int offsets[5][2] = { {0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1} };
    for (auto& o : offsets) {
        const int x = u + o[0], y = v + o[1];
        if (x < 0 || x >= w || y < 0 || y >= h) continue;
        const size_t idx = (static_cast<size_t>(y) * w + x) * 3;
        rgb[idx + 0] = r; rgb[idx + 1] = g; rgb[idx + 2] = b;
    }
}

// ---------------------------------------------------------------------------
// run_lm_gpu — the host-orchestrated single-trajectory LM: EVERY normal-
// equation assembly (both the "solve at current T" and the "evaluate cost
// at the candidate T" that accept/reject needs) goes through
// launch_assemble_normal_equations, the CORRESPONDENCE-parallel GPU kernel.
// The damping/accept-reject CONTROL FLOW below is main.cu's own (compare
// against reference_cpu.cpp's run_lm_cpu, written independently — see that
// file's header for exactly what independence buys the TRAJECTORY_TWIN
// gate). d_block_partials/h_block_partials are caller-owned scratch,
// reused across every call in this program (08.01's "allocate once, outside
// the loop" discipline, cited: a 20-iteration, several-times-per-demo LM
// loop that reallocated device memory every step would spend its budget on
// the allocator).
// ---------------------------------------------------------------------------
static void run_lm_gpu(const float* d_p_obs, const float* d_uv_obs, int n, PinholeIntrinsics K,
                       Rigid3 T_init, int max_iters,
                       float* d_block_partials, std::vector<float>& h_block_partials,
                       Rigid3& out_T, std::vector<double>& loss_history,
                       double out_H21[21], double out_g6[6])
{
    auto assemble = [&](Rigid3 Tq, double H21[21], double g6[6], double& cost) {
        const int num_blocks = launch_assemble_normal_equations(d_p_obs, d_uv_obs, n, Tq, K, d_block_partials);
        h_block_partials.resize(static_cast<size_t>(num_blocks) * kReduceWidth);
        CUDA_CHECK(cudaMemcpy(h_block_partials.data(), d_block_partials,
                              h_block_partials.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (int a = 0; a < 21; ++a) H21[a] = 0.0;
        for (int a = 0; a < 6; ++a) g6[a] = 0.0;
        cost = 0.0;
        // Host finishes the sum ACROSS BLOCKS in double (02.06's "GPU
        // partial reduce, host finishes it" split, cited in kernels.cuh).
        for (int b = 0; b < num_blocks; ++b) {
            const float* row = &h_block_partials[static_cast<size_t>(b) * kReduceWidth];
            for (int a = 0; a < 21; ++a) H21[a] += static_cast<double>(row[a]);
            for (int a = 0; a < 6; ++a) g6[a] += static_cast<double>(row[21 + a]);
            cost += static_cast<double>(row[27]);
        }
    };

    Rigid3 T = T_init;
    double lambda = kLambdaInit;
    double H21[21], g6[6], cost;
    assemble(T, H21, g6, cost);
    loss_history.clear();
    loss_history.push_back(cost);

    for (int it = 0; it < max_iters; ++it) {
        double delta[6];
        bool ok = false;
        for (int attempt = 0; attempt < 5 && !ok; ++attempt) {
            ok = cholesky6_solve(H21, g6, lambda, delta);
            if (!ok) lambda *= kLambdaUp;
        }
        if (!ok) break;

        Rigid3 T_cand;
        retract(T, delta, T_cand);
        double H21n[21], g6n[6], cost_new;
        assemble(T_cand, H21n, g6n, cost_new);

        const double delta_norm = std::sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2] +
                                            delta[3] * delta[3] + delta[4] * delta[4] + delta[5] * delta[5]);

        if (cost_new < cost) {
            const double rel_change = std::fabs(cost - cost_new) / (cost_new + 1.0e-12);
            T = T_cand;
            cost = cost_new;
            for (int a = 0; a < 21; ++a) H21[a] = H21n[a];
            for (int a = 0; a < 6; ++a) g6[a] = g6n[a];
            lambda *= kLambdaDown;
            if (lambda < kLambdaMin) lambda = kLambdaMin;
            loss_history.push_back(cost);
            if (delta_norm < kConvergeDeltaNorm || rel_change < kConvergeCostRel) break;
        } else {
            lambda *= kLambdaUp;
            loss_history.push_back(cost);
        }
    }

    out_T = T;
    for (int a = 0; a < 21; ++a) out_H21[a] = H21[a];
    for (int a = 0; a < 6; ++a) out_g6[a] = g6[a];
}

// ---------------------------------------------------------------------------
// jacobian_check — the CALCULUS gate: at several representative (state,
// correspondence) pairs, compare residual_and_jacobian's ANALYTIC Jacobian
// against a CENTRAL-DIFFERENCE NUMERIC one built by calling ONLY the
// residual half at retracted +/-eps states (kernels.cuh "retract"). This is
// independent of whether the PROJECTION formula matches reality (the
// zero-noise gate checks that) — it only checks that J is truly
// d(residual)/d(local-state) for whatever residual_and_jacobian computes,
// catching a transcription bug in the analytic-Jacobian lines specifically.
// eps=1e-3 (float, central difference): truncation error ~eps^2~1e-6,
// rounding error ~ulp/eps~1e-7/1e-3~1e-4 — rounding-dominated, so the
// achievable numeric-Jacobian accuracy is ~1e-4 absolute in a quantity of
// order 1-100 (px per unit radian/meter) -> kJacobianRelTol=5e-2 leaves
// generous headroom above that floor (THEORY.md "Numerical considerations"
// walks through this eps/tolerance tradeoff in full).
// ---------------------------------------------------------------------------
static bool jacobian_check(const Dataset& ds, float& out_worst_rel)
{
    float worst = 0.0f;
    const float eps = 1.0e-3f;

    // A handful of representative states: ground truth, the rough prior
    // (deterministic, not random — see rough_prior's own comment), and two
    // extra fixed offsets so the check exercises more of the state space
    // than a single point.
    const Rigid3 Tgt = gt_rigid3(ds);
    const Rigid3 states[3] = {
        Tgt,
        rough_prior(ds),
        [&] { float om[3] = { Tgt.R[0] * 0.0f + 0.20f, -0.15f, 0.10f }; Rigid3 T; so3_exp(om, T.R);
              T.t[0] = ds.t_gt[0] - 0.10f; T.t[1] = ds.t_gt[1] + 0.05f; T.t[2] = ds.t_gt[2] + 0.08f; return T; }()
    };

    for (const Rigid3& T : states) {
        // The first 6 correspondences are enough to exercise every column
        // of J at each state without ballooning the check's runtime.
        for (int i = 0; i < 6 && i < static_cast<int>(ds.corr.size()); ++i) {
            const Correspondence& c = ds.corr[static_cast<size_t>(i)];
            float r0[2], J[12];
            residual_and_jacobian(T, ds.K, c.p_src, c.uv_true, r0, J);

            for (int dim = 0; dim < 6; ++dim) {
                double delta_plus[6] = {0}, delta_minus[6] = {0};
                delta_plus[dim] = eps;
                delta_minus[dim] = -eps;
                Rigid3 Tp, Tm;
                retract(T, delta_plus, Tp);
                retract(T, delta_minus, Tm);
                float rp[2], rm[2], Jdummy[12];
                residual_and_jacobian(Tp, ds.K, c.p_src, c.uv_true, rp, Jdummy);
                residual_and_jacobian(Tm, ds.K, c.p_src, c.uv_true, rm, Jdummy);

                for (int row = 0; row < 2; ++row) {
                    const float numeric = (rp[row] - rm[row]) / (2.0f * eps);
                    const float analytic = J[row * 6 + dim];
                    const float scale = std::fabs(analytic) > 1.0f ? std::fabs(analytic) : 1.0f;
                    const float rel = std::fabs(numeric - analytic) / scale;
                    if (rel > worst) worst = rel;
                }
            }
        }
    }
    out_worst_rel = worst;
    return worst <= kJacobianRelTol;
}

// ===========================================================================
// main
// ===========================================================================
int main(int argc, char** argv)
{
    std::printf("[demo] camera-LiDAR / camera-camera extrinsic calibration (project 01.17): batched reprojection-error LM\n");
    print_device_info();
    std::printf("PROBLEM: batched Levenberg-Marquardt reprojection-error minimization, V=%d poses x %d fiducials = %d correspondences/scenario, "
               "image %dx%d px, intrinsics fx=%.1f fy=%.1f cx=%.1f cy=%.1f, K=%d multi-start restarts, max %d LM iters/run\n",
               kNumViews, kPointsPerView, kNumCorr, kImageWidth, kImageHeight,
               static_cast<double>(kFx), static_cast<double>(kFy), static_cast<double>(kCx), static_cast<double>(kCy),
               kMultiStartK, kMaxLmIters);

    // ---- load the three committed correspondence sets + background -------
    const std::string dl_diverse  = find_data_file("", argv[0], "cam_lidar_diverse.csv");
    const std::string dl_coplanar = find_data_file("", argv[0], "cam_lidar_coplanar.csv");
    const std::string dc_diverse  = find_data_file("", argv[0], "cam_cam_diverse.csv");
    const std::string bg_path     = find_data_file("", argv[0], "cam_background.pgm");

    Dataset ds_cl_diverse  = load_dataset(dl_diverse);
    Dataset ds_cl_coplanar = load_dataset(dl_coplanar);
    Dataset ds_cc_diverse  = load_dataset(dc_diverse);
    GrayImage bg           = load_pgm(bg_path);

    if (!ds_cl_diverse.loaded || !ds_cl_coplanar.loaded || !ds_cc_diverse.loaded || !bg.loaded) {
        std::printf("SCENARIO: NOT FOUND or MALFORMED — run scripts/make_synthetic.py to regenerate data/sample/\n");
        std::printf("RESULT: FAIL (data missing)\n");
        return 1;
    }
    std::printf("SCENARIO: camera-LiDAR (diverse + coplanar cohorts) and camera-camera (diverse cohort) extrinsic recovery [synthetic]\n");

    // ---- persistent device buffers, allocated once (08.01's discipline) ---
    float *d_p_obs = nullptr, *d_uv_obs = nullptr, *d_block_partials = nullptr;
    CUDA_CHECK(cudaMalloc(&d_p_obs, static_cast<size_t>(kNumCorr) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_uv_obs, static_cast<size_t>(kNumCorr) * 2 * sizeof(float)));
    const int max_blocks = blocks_for(kNumCorr, kThreadsPerBlock);
    CUDA_CHECK(cudaMalloc(&d_block_partials, static_cast<size_t>(max_blocks) * kReduceWidth * sizeof(float)));

    Rigid3* d_farm_T = nullptr;
    double* d_farm_loss = nullptr;
    float *d_farm_rot = nullptr, *d_farm_trans = nullptr;
    CUDA_CHECK(cudaMalloc(&d_farm_T, static_cast<size_t>(kMultiStartK) * sizeof(Rigid3)));
    CUDA_CHECK(cudaMalloc(&d_farm_loss, static_cast<size_t>(kMultiStartK) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_farm_rot, static_cast<size_t>(kMultiStartK) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_farm_trans, static_cast<size_t>(kMultiStartK) * sizeof(float)));

    std::vector<float> h_block_partials;
    std::vector<float> p_obs, uv_obs;
    bool all_pass = true;

    // Artifact accumulators, written once at the end.
    std::ostringstream gates_csv;
    gates_csv << "gate,metric,value\n";

    // =======================================================================
    // STAGE A — jacobian_check (the calculus gate; independent of GPU/CPU
    // twin agreement — see the function's own header)
    // =======================================================================
    float jac_worst = 0.0f;
    const bool jac_pass = jacobian_check(ds_cl_diverse, jac_worst);
    std::printf("[info] jacobian check: worst relative deviation %.4e over 3 states x 6 correspondences x 6 dims (eps=1e-3 central difference, tol %.3g)\n",
               static_cast<double>(jac_worst), static_cast<double>(kJacobianRelTol));
    std::printf("JACOBIAN_CHECK: %s (analytic vs. central-difference numeric Jacobian agree within tolerance)\n", jac_pass ? "PASS" : "FAIL");
    all_pass &= jac_pass;
    gates_csv << "jacobian_check,worst_rel_deviation," << jac_worst << "\n";

    // =======================================================================
    // STAGE B — ASSEMBLY_TWIN: one normal-equation assembly, GPU vs CPU,
    // tight tolerance (a SINGLE evaluation — no chained iterations yet).
    // =======================================================================
    apply_noise(ds_cl_diverse, kNoiseMed.sigma_px, kNoiseMed.sigma_p_src_m, true, kSeedAssemblyTwin, p_obs, uv_obs);
    CUDA_CHECK(cudaMemcpy(d_p_obs, p_obs.data(), p_obs.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_uv_obs, uv_obs.data(), uv_obs.size() * sizeof(float), cudaMemcpyHostToDevice));

    const Rigid3 T_probe = rough_prior(ds_cl_diverse);
    const int nb = launch_assemble_normal_equations(d_p_obs, d_uv_obs, kNumCorr, T_probe, ds_cl_diverse.K, d_block_partials);
    h_block_partials.resize(static_cast<size_t>(nb) * kReduceWidth);
    CUDA_CHECK(cudaMemcpy(h_block_partials.data(), d_block_partials, h_block_partials.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double H21_gpu[21] = {0}, g6_gpu[6] = {0}, cost_gpu = 0.0;
    for (int b = 0; b < nb; ++b) {
        const float* row = &h_block_partials[static_cast<size_t>(b) * kReduceWidth];
        for (int a = 0; a < 21; ++a) H21_gpu[a] += static_cast<double>(row[a]);
        for (int a = 0; a < 6; ++a) g6_gpu[a] += static_cast<double>(row[21 + a]);
        cost_gpu += static_cast<double>(row[27]);
    }
    double H21_cpu[21], g6_cpu[6], cost_cpu;
    assemble_normal_equations_cpu(p_obs.data(), uv_obs.data(), kNumCorr, T_probe, ds_cl_diverse.K, H21_cpu, g6_cpu, &cost_cpu);

    double assembly_worst = 0.0;
    for (int a = 0; a < 21; ++a) { const double s = std::fabs(H21_cpu[a]) > 1.0 ? std::fabs(H21_cpu[a]) : 1.0; assembly_worst = std::max(assembly_worst, std::fabs(H21_gpu[a] - H21_cpu[a]) / s); }
    for (int a = 0; a < 6; ++a)  { const double s = std::fabs(g6_cpu[a]) > 1.0 ? std::fabs(g6_cpu[a]) : 1.0;   assembly_worst = std::max(assembly_worst, std::fabs(g6_gpu[a] - g6_cpu[a]) / s); }
    { const double s = std::fabs(cost_cpu) > 1.0 ? std::fabs(cost_cpu) : 1.0; assembly_worst = std::max(assembly_worst, std::fabs(cost_gpu - cost_cpu) / s); }
    const bool assembly_pass = assembly_worst <= kAssemblyRelTol;
    std::printf("[info] assembly twin: worst relative deviation %.4e over 28 [H21|g6|cost] entries (tol %.3g)\n", assembly_worst, kAssemblyRelTol);
    std::printf("ASSEMBLY_TWIN: %s (GPU correspondence-parallel assembly matches CPU oracle)\n", assembly_pass ? "PASS" : "FAIL");
    all_pass &= assembly_pass;
    gates_csv << "assembly_twin,worst_rel_deviation," << assembly_worst << "\n";

    // =======================================================================
    // STAGE C — TRAJECTORY_TWIN: one full LM trajectory, GPU-orchestrated
    // (main.cu's run_lm_gpu, calling the assembly kernel every iteration)
    // vs. CPU-only (reference_cpu.cpp's independently-written run_lm_cpu).
    // Same starting guess as Stage B, a FRESH noise realization (its own
    // seed) so the trajectory gate is not just replaying Stage B's data.
    // =======================================================================
    apply_noise(ds_cl_diverse, kNoiseMed.sigma_px, kNoiseMed.sigma_p_src_m, true, kSeedTrajectoryTwin, p_obs, uv_obs);
    CUDA_CHECK(cudaMemcpy(d_p_obs, p_obs.data(), p_obs.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_uv_obs, uv_obs.data(), uv_obs.size() * sizeof(float), cudaMemcpyHostToDevice));

    Rigid3 T_gpu_traj;
    std::vector<double> loss_hist_gpu;
    double H21f[21], g6f[6];
    run_lm_gpu(d_p_obs, d_uv_obs, kNumCorr, ds_cl_diverse.K, T_probe, kMaxLmIters,
              d_block_partials, h_block_partials, T_gpu_traj, loss_hist_gpu, H21f, g6f);

    Rigid3 T_cpu_traj;
    std::vector<double> loss_hist_cpu(static_cast<size_t>(kMaxLmIters) + 1, 0.0);
    int n_iters_cpu = 0;
    run_lm_cpu(p_obs.data(), uv_obs.data(), kNumCorr, ds_cl_diverse.K, T_probe, kMaxLmIters, T_cpu_traj, loss_hist_cpu.data(), n_iters_cpu);
    loss_hist_cpu.resize(static_cast<size_t>(n_iters_cpu));

    float traj_rot_dev = 0.0f, traj_trans_dev = 0.0f;
    traj_rot_dev = rotation_angle_deg(T_gpu_traj.R, T_cpu_traj.R);
    traj_trans_dev = translation_error_m(T_gpu_traj.t, T_cpu_traj.t);
    const bool traj_pass = (traj_rot_dev <= kTrajectoryRotTolDeg) && (traj_trans_dev <= kTrajectoryTransTolM);
    std::printf("[info] trajectory twin: final-pose deviation %.4e deg rotation, %.4e m translation over %d/%d GPU/CPU iterations (tol %.3g deg, %.3g m)\n",
               static_cast<double>(traj_rot_dev), static_cast<double>(traj_trans_dev),
               static_cast<int>(loss_hist_gpu.size()) - 1, n_iters_cpu - 1,
               static_cast<double>(kTrajectoryRotTolDeg), static_cast<double>(kTrajectoryTransTolM));
    std::printf("TRAJECTORY_TWIN: %s (GPU-orchestrated LM trajectory matches CPU-only trajectory)\n", traj_pass ? "PASS" : "FAIL");
    all_pass &= traj_pass;
    gates_csv << "trajectory_twin,rot_deviation_deg," << traj_rot_dev << "\n";
    gates_csv << "trajectory_twin,trans_deviation_m," << traj_trans_dev << "\n";

    // convergence_curves.csv artifact (the shorter of the two histories sets
    // the row count — see demo/README.md for why they are expected to match).
    const std::string out_dir = resolve_out_dir(argv[0]);
    {
        std::ofstream f(out_dir + "/convergence_curves.csv");
        if (f.is_open()) {
            f << "iteration,gpu_loss,cpu_loss\n";
            const size_t rows = std::min(loss_hist_gpu.size(), loss_hist_cpu.size());
            for (size_t i = 0; i < rows; ++i) f << i << ',' << loss_hist_gpu[i] << ',' << loss_hist_cpu[i] << '\n';
        }
    }
    // Row count is NOT part of the stable line: it is the number of LM
    // iterations actually taken, a boundary-sensitive quantity (it depends
    // on a floating-point convergence-threshold comparison) that could
    // legitimately differ by one iteration on a different GPU architecture
    // (sm_86/sm_89 vs. this machine's sm_75) — exactly the class of
    // cross-platform ulp sensitivity CLAUDE.md §12 warns stable lines to
    // avoid. The FILE is still written and still useful; only the count
    // moves to an unchecked [info] line.
    std::printf("[info] convergence_curves.csv: %zu rows (min of GPU %zu / CPU %zu LM iterations taken)\n",
               std::min(loss_hist_gpu.size(), loss_hist_cpu.size()), loss_hist_gpu.size(), loss_hist_cpu.size());
    std::printf("ARTIFACT: wrote demo/out/convergence_curves.csv\n");

    // =======================================================================
    // STAGE D — multi-start farm on the camera-LiDAR diverse cohort: the
    // convergence-basin study (K=1024) AND, from its best result, the
    // camera-LiDAR recovery gate.
    // =======================================================================
    apply_noise(ds_cl_diverse, kNoiseMed.sigma_px, kNoiseMed.sigma_p_src_m, true, kSeedCamLidarRecovery, p_obs, uv_obs);
    CUDA_CHECK(cudaMemcpy(d_p_obs, p_obs.data(), p_obs.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_uv_obs, uv_obs.data(), uv_obs.size() * sizeof(float), cudaMemcpyHostToDevice));

    GpuTimer farm_timer;
    farm_timer.begin();
    launch_multistart_farm(d_p_obs, d_uv_obs, kNumCorr, ds_cl_diverse.K, kIdentityRigid3,
                           kBasinMaxRotRad, kBasinMaxTransM, kMultistartBaseSeed, kMultiStartK, kMaxLmIters,
                           d_farm_T, d_farm_loss, d_farm_rot, d_farm_trans);
    const float farm_ms = farm_timer.end_ms();

    std::vector<Rigid3> farm_T(kMultiStartK);
    std::vector<double> farm_loss(kMultiStartK);
    std::vector<float> farm_rot(kMultiStartK), farm_trans(kMultiStartK);
    CUDA_CHECK(cudaMemcpy(farm_T.data(), d_farm_T, farm_T.size() * sizeof(Rigid3), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(farm_loss.data(), d_farm_loss, farm_loss.size() * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(farm_rot.data(), d_farm_rot, farm_rot.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(farm_trans.data(), d_farm_trans, farm_trans.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::printf("[time] multi-start farm: K=%d independent LM trajectories in %.3f ms\n", kMultiStartK, static_cast<double>(farm_ms));

    int best_k = 0;
    for (int k = 1; k < kMultiStartK; ++k) if (farm_loss[static_cast<size_t>(k)] < farm_loss[static_cast<size_t>(best_k)]) best_k = k;
    const Rigid3 T_cl_recovered = farm_T[static_cast<size_t>(best_k)];
    float cl_rot_err = 0.0f, cl_trans_err = 0.0f;
    pose_error(T_cl_recovered, ds_cl_diverse, cl_rot_err, cl_trans_err);
    const bool recovery_cl_pass = (cl_rot_err <= kRecoveryRotTolDeg) && (cl_trans_err <= kRecoveryTransTolM);
    std::printf("[info] camera-LiDAR recovery (best of %d starts, k=%d, noise=%s): rotation error %.4f deg, translation error %.4f mm (tol %.3g deg, %.3g mm)\n",
               kMultiStartK, best_k, kNoiseMed.label, static_cast<double>(cl_rot_err), static_cast<double>(cl_trans_err) * 1000.0,
               static_cast<double>(kRecoveryRotTolDeg), static_cast<double>(kRecoveryTransTolM) * 1000.0);
    std::printf("RECOVERY_CAM_LIDAR: %s (best-of-multistart recovers T_camera_lidar within tolerance)\n", recovery_cl_pass ? "PASS" : "FAIL");
    all_pass &= recovery_cl_pass;
    gates_csv << "recovery_cam_lidar,rot_err_deg," << cl_rot_err << "\n";
    gates_csv << "recovery_cam_lidar,trans_err_m," << cl_trans_err << "\n";

    // Basin classification + basin_scatter.csv. "Converged" requires BOTH a
    // near-true rotation AND translation (kernels.cuh explains why loss
    // alone is not enough on badly under-constrained data — the degeneracy
    // gate below is exactly that failure mode).
    int n_converged = 0;
    {
        std::ofstream f(out_dir + "/basin_scatter.csv");
        const bool ok = f.is_open();
        if (ok) f << "k,init_rot_perturb_rad,init_trans_perturb_m,final_loss,rot_err_deg,trans_err_m,converged\n";
        for (int k = 0; k < kMultiStartK; ++k) {
            float rk, tk;
            pose_error(farm_T[static_cast<size_t>(k)], ds_cl_diverse, rk, tk);
            const bool converged = (rk <= kBasinConvergedRotDeg) && (tk <= kBasinConvergedTransM);
            if (converged) ++n_converged;
            if (ok) f << k << ',' << farm_rot[static_cast<size_t>(k)] << ',' << farm_trans[static_cast<size_t>(k)] << ','
                     << farm_loss[static_cast<size_t>(k)] << ',' << rk << ',' << tk << ',' << (converged ? 1 : 0) << '\n';
        }
    }
    const float basin_pct = 100.0f * static_cast<float>(n_converged) / static_cast<float>(kMultiStartK);
    const bool basin_pass = basin_pct >= kBasinMinConvergedPct;
    std::printf("[info] basin: %d/%d (%.1f%%) of randomized starts (<= %.2f rad / %.2f m from identity) converged to the true extrinsic (tol >= %.1f%%)\n",
               n_converged, kMultiStartK, static_cast<double>(basin_pct),
               static_cast<double>(kBasinMaxRotRad), static_cast<double>(kBasinMaxTransM), static_cast<double>(kBasinMinConvergedPct));
    std::printf("BASIN: %s (convergence-basin coverage over %d randomized multi-starts)\n", basin_pass ? "PASS" : "FAIL", kMultiStartK);
    all_pass &= basin_pass;
    gates_csv << "basin,converged_pct," << basin_pct << "\n";
    std::printf("ARTIFACT: wrote demo/out/basin_scatter.csv (%d rows)\n", kMultiStartK);

    // =======================================================================
    // STAGE D2 — MULTISTART_TWIN: reproduce GPU threads k=0..63 exactly on
    // the CPU (same seed formula, same perturbation draw — see
    // multistart_lm_cpu's header). kBasinMaxRotRad/TransM is now large
    // enough (see kernels.cuh) that MANY of these 64 starts are FAR from the
    // true optimum and converge nowhere in particular — comparing their raw
    // final pose would compare two independently-chaotic non-convergent
    // trajectories and could disagree by many degrees/meters for reasons
    // that have nothing to do with a bug (tiny float-order differences,
    // amplified by 20 iterations of an unstable, far-from-solution Newton
    // step). The MEANINGFUL twin question is therefore two-tiered:
    //   (a) do GPU and CPU AGREE on WHICH starts converge (same threshold
    //       the basin gate uses)? — this is well-defined and non-chaotic
    //       even for runs that individually diverge unpredictably;
    //   (b) for starts BOTH sides classify as converged, do they converge
    //       to the SAME tight final pose? — the assembly/trajectory twins'
    //       kind of check, restricted to the well-posed subset.
    // =======================================================================
    const int kSubset = 64;
    int agree_count = 0, both_converged = 0;
    float ms_worst_rot = 0.0f, ms_worst_trans = 0.0f;
    double ms_worst_loss_rel = 0.0;
    for (int k = 0; k < kSubset; ++k) {
        Rigid3 T_cpu_k; double loss_cpu_k; float init_rot_cpu, init_trans_cpu;
        multistart_lm_cpu(p_obs.data(), uv_obs.data(), kNumCorr, ds_cl_diverse.K, kIdentityRigid3,
                          kBasinMaxRotRad, kBasinMaxTransM, kMultistartBaseSeed, k, kMaxLmIters,
                          T_cpu_k, loss_cpu_k, init_rot_cpu, init_trans_cpu);

        float r_gpu, t_gpu, r_cpu, t_cpu;
        pose_error(farm_T[static_cast<size_t>(k)], ds_cl_diverse, r_gpu, t_gpu);
        pose_error(T_cpu_k, ds_cl_diverse, r_cpu, t_cpu);
        const bool conv_gpu = (r_gpu <= kBasinConvergedRotDeg) && (t_gpu <= kBasinConvergedTransM);
        const bool conv_cpu = (r_cpu <= kBasinConvergedRotDeg) && (t_cpu <= kBasinConvergedTransM);
        if (conv_gpu == conv_cpu) ++agree_count;

        if (conv_gpu && conv_cpu) {
            ++both_converged;
            const float rdev = rotation_angle_deg(T_cpu_k.R, farm_T[static_cast<size_t>(k)].R);
            const float tdev = translation_error_m(T_cpu_k.t, farm_T[static_cast<size_t>(k)].t);
            const double lscale = std::fabs(farm_loss[static_cast<size_t>(k)]) > 1.0 ? std::fabs(farm_loss[static_cast<size_t>(k)]) : 1.0;
            const double ldev = std::fabs(loss_cpu_k - farm_loss[static_cast<size_t>(k)]) / lscale;
            ms_worst_rot = std::max(ms_worst_rot, rdev);
            ms_worst_trans = std::max(ms_worst_trans, tdev);
            ms_worst_loss_rel = std::max(ms_worst_loss_rel, ldev);
        }
    }
    const bool multistart_twin_pass = (agree_count == kSubset) &&
                                      (ms_worst_rot <= kMultistartRotTolDeg) && (ms_worst_trans <= kMultistartTransTolM) &&
                                      (ms_worst_loss_rel <= kMultistartLossRelTol);
    std::printf("[info] multistart twin: converged/diverged classification agreed on %d/%d starts; of the %d both sides classified converged, "
               "worst deviation %.4e deg rotation, %.4e m translation, %.4e relative loss (tol %.3g deg, %.3g m, %.3g)\n",
               agree_count, kSubset, both_converged, static_cast<double>(ms_worst_rot), static_cast<double>(ms_worst_trans), ms_worst_loss_rel,
               static_cast<double>(kMultistartRotTolDeg), static_cast<double>(kMultistartTransTolM), kMultistartLossRelTol);
    std::printf("MULTISTART_TWIN: %s (GPU farm threads 0..%d reproduced exactly on the CPU)\n", multistart_twin_pass ? "PASS" : "FAIL", kSubset - 1);
    all_pass &= multistart_twin_pass;
    gates_csv << "multistart_twin,classification_agree_count," << agree_count << "\n";
    gates_csv << "multistart_twin,worst_rot_dev_deg," << ms_worst_rot << "\n";
    gates_csv << "multistart_twin,worst_trans_dev_m," << ms_worst_trans << "\n";

    // =======================================================================
    // STAGE E — RECOVERY_CAM_CAM: a fresh, smaller multi-start farm on the
    // camera-camera diverse cohort (no source-point noise — see apply_noise
    // and the file header on why camera-camera differs here).
    // =======================================================================
    const int kCamCamStarts = 256;
    apply_noise(ds_cc_diverse, kNoiseMed.sigma_px, kNoiseMed.sigma_p_src_m, /*add_p_src_noise=*/false, kSeedCamCamRecovery, p_obs, uv_obs);
    CUDA_CHECK(cudaMemcpy(d_p_obs, p_obs.data(), p_obs.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_uv_obs, uv_obs.data(), uv_obs.size() * sizeof(float), cudaMemcpyHostToDevice));
    launch_multistart_farm(d_p_obs, d_uv_obs, kNumCorr, ds_cc_diverse.K, kIdentityRigid3,
                           kBasinMaxRotRad, kBasinMaxTransM, kMultistartBaseSeed, kCamCamStarts, kMaxLmIters,
                           d_farm_T, d_farm_loss, d_farm_rot, d_farm_trans);
    std::vector<Rigid3> cc_T(kCamCamStarts);
    std::vector<double> cc_loss(kCamCamStarts);
    CUDA_CHECK(cudaMemcpy(cc_T.data(), d_farm_T, cc_T.size() * sizeof(Rigid3), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cc_loss.data(), d_farm_loss, cc_loss.size() * sizeof(double), cudaMemcpyDeviceToHost));

    int cc_best = 0;
    for (int k = 1; k < kCamCamStarts; ++k) if (cc_loss[static_cast<size_t>(k)] < cc_loss[static_cast<size_t>(cc_best)]) cc_best = k;
    const Rigid3 T_cc_recovered = cc_T[static_cast<size_t>(cc_best)];
    float cc_rot_err = 0.0f, cc_trans_err = 0.0f;
    pose_error(T_cc_recovered, ds_cc_diverse, cc_rot_err, cc_trans_err);
    const bool recovery_cc_pass = (cc_rot_err <= kRecoveryRotTolDeg) && (cc_trans_err <= kRecoveryTransTolM);
    std::printf("[info] camera-camera recovery (best of %d starts, k=%d, noise=%s, no source-point noise): rotation error %.4f deg, translation error %.4f mm (tol %.3g deg, %.3g mm)\n",
               kCamCamStarts, cc_best, kNoiseMed.label, static_cast<double>(cc_rot_err), static_cast<double>(cc_trans_err) * 1000.0,
               static_cast<double>(kRecoveryRotTolDeg), static_cast<double>(kRecoveryTransTolM) * 1000.0);
    std::printf("RECOVERY_CAM_CAM: %s (best-of-multistart recovers T_camera2_camera1 within tolerance)\n", recovery_cc_pass ? "PASS" : "FAIL");
    all_pass &= recovery_cc_pass;
    gates_csv << "recovery_cam_cam,rot_err_deg," << cc_rot_err << "\n";
    gates_csv << "recovery_cam_cam,trans_err_m," << cc_trans_err << "\n";

    // =======================================================================
    // STAGE F — NOISE_SCALING: single-trajectory recovery (run_lm_gpu from
    // the rough prior) at three documented noise levels, camera-LiDAR
    // diverse cohort. Checked for sane (non-decreasing-within-slack) scaling.
    // =======================================================================
    struct NoiseResult { const char* label; float rot_err_deg; float trans_err_m; };
    NoiseResult noise_results[3];
    const NoiseLevel levels[3] = { kNoiseLow, kNoiseMed, kNoiseHigh };
    const uint32_t level_seeds[3] = { kSeedNoiseScalingLow, kSeedNoiseScalingMed, kSeedNoiseScalingHigh };
    for (int i = 0; i < 3; ++i) {
        apply_noise(ds_cl_diverse, levels[i].sigma_px, levels[i].sigma_p_src_m, true, level_seeds[i], p_obs, uv_obs);
        CUDA_CHECK(cudaMemcpy(d_p_obs, p_obs.data(), p_obs.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_uv_obs, uv_obs.data(), uv_obs.size() * sizeof(float), cudaMemcpyHostToDevice));
        Rigid3 T_ns; std::vector<double> hist; double H21ns[21], g6ns[6];
        run_lm_gpu(d_p_obs, d_uv_obs, kNumCorr, ds_cl_diverse.K, rough_prior(ds_cl_diverse), kMaxLmIters,
                  d_block_partials, h_block_partials, T_ns, hist, H21ns, g6ns);
        float re, te;
        pose_error(T_ns, ds_cl_diverse, re, te);
        noise_results[i] = { levels[i].label, re, te };
        std::printf("[info] noise scaling (%s: sigma_px=%.2f px, sigma_lidar=%.4f m): rotation error %.4f deg, translation error %.4f mm\n",
                   levels[i].label, static_cast<double>(levels[i].sigma_px), static_cast<double>(levels[i].sigma_p_src_m),
                   static_cast<double>(re), static_cast<double>(te) * 1000.0);
        gates_csv << "noise_scaling_" << levels[i].label << ",rot_err_deg," << re << "\n";
        gates_csv << "noise_scaling_" << levels[i].label << ",trans_err_m," << te << "\n";
    }
    const bool noise_scaling_pass =
        (noise_results[0].rot_err_deg   <= noise_results[1].rot_err_deg   + kNoiseScalingSlackDeg) &&
        (noise_results[1].rot_err_deg   <= noise_results[2].rot_err_deg   + kNoiseScalingSlackDeg) &&
        (noise_results[0].trans_err_m   <= noise_results[1].trans_err_m   + kNoiseScalingSlackM) &&
        (noise_results[1].trans_err_m   <= noise_results[2].trans_err_m   + kNoiseScalingSlackM);
    std::printf("NOISE_SCALING: %s (recovery error scales sensibly, low <= med <= high within a documented slack)\n", noise_scaling_pass ? "PASS" : "FAIL");
    all_pass &= noise_scaling_pass;

    // =======================================================================
    // STAGE G — DEGENERACY: the practical calibration lesson. SAME ground
    // truth, SAME noise level, two pose cohorts — diverse (well-conditioned)
    // vs. coplanar (near-rank-deficient). Compares the FINAL J^T J's
    // condition-number proxy (jacobi_eigen_symmetric6, 01.16's construction,
    // cited) and the final translation error.
    // =======================================================================
    apply_noise(ds_cl_diverse, kNoiseMed.sigma_px, kNoiseMed.sigma_p_src_m, true, kSeedDegenDiverse, p_obs, uv_obs);
    CUDA_CHECK(cudaMemcpy(d_p_obs, p_obs.data(), p_obs.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_uv_obs, uv_obs.data(), uv_obs.size() * sizeof(float), cudaMemcpyHostToDevice));
    Rigid3 T_div; std::vector<double> hist_div; double H21_div[21], g6_div[6];
    run_lm_gpu(d_p_obs, d_uv_obs, kNumCorr, ds_cl_diverse.K, rough_prior(ds_cl_diverse), kMaxLmIters,
              d_block_partials, h_block_partials, T_div, hist_div, H21_div, g6_div);
    float div_rot_err, div_trans_err;
    pose_error(T_div, ds_cl_diverse, div_rot_err, div_trans_err);

    apply_noise(ds_cl_coplanar, kNoiseMed.sigma_px, kNoiseMed.sigma_p_src_m, true, kSeedDegenCoplanar, p_obs, uv_obs);
    CUDA_CHECK(cudaMemcpy(d_p_obs, p_obs.data(), p_obs.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_uv_obs, uv_obs.data(), uv_obs.size() * sizeof(float), cudaMemcpyHostToDevice));
    Rigid3 T_cop; std::vector<double> hist_cop; double H21_cop[21], g6_cop[6];
    run_lm_gpu(d_p_obs, d_uv_obs, kNumCorr, ds_cl_coplanar.K, rough_prior(ds_cl_coplanar), kMaxLmIters,
              d_block_partials, h_block_partials, T_cop, hist_cop, H21_cop, g6_cop);
    float cop_rot_err, cop_trans_err;
    pose_error(T_cop, ds_cl_coplanar, cop_rot_err, cop_trans_err);

    auto condition_number = [](const double H21[21]) -> double {
        double A[6][6];
        for (int i = 0; i < 6; ++i)
            for (int j = i; j < 6; ++j) { A[i][j] = H21[hidx(i, j)]; A[j][i] = A[i][j]; }
        double eigvecs[6][6];
        jacobi_eigen_symmetric6(A, eigvecs);
        double lo = 1.0e300, hi = -1.0e300;
        for (int i = 0; i < 6; ++i) { const double v = A[i][i]; if (v < lo) lo = v; if (v > hi) hi = v; }
        if (lo < 1.0e-6) lo = 1.0e-6;   // guard: a genuinely singular direction floors, not divides by ~0
        return hi / lo;
    };
    const double cond_div = condition_number(H21_div);
    const double cond_cop = condition_number(H21_cop);
    const double cond_ratio = cond_div > 1.0e-12 ? (cond_cop / cond_div) : 0.0;
    const double trans_ratio = static_cast<double>(div_trans_err) > 1.0e-9
                              ? static_cast<double>(cop_trans_err) / static_cast<double>(div_trans_err) : 0.0;
    const bool degeneracy_pass = (cond_ratio > kDegenCondFactor) && (trans_ratio > kDegenTransFactor);
    std::printf("[info] degeneracy: J^T J condition number proxy diverse=%.4e coplanar=%.4e (ratio %.2fx, tol > %.1fx); "
               "translation error diverse=%.4f mm coplanar=%.4f mm (ratio %.2fx, tol > %.1fx)\n",
               cond_div, cond_cop, cond_ratio, kDegenCondFactor,
               static_cast<double>(div_trans_err) * 1000.0, static_cast<double>(cop_trans_err) * 1000.0, trans_ratio, kDegenTransFactor);
    std::printf("DEGENERACY: %s (coplanar-pose cohort shows worse conditioning AND worse translation accuracy than the diverse cohort)\n",
               degeneracy_pass ? "PASS" : "FAIL");
    all_pass &= degeneracy_pass;
    gates_csv << "degeneracy,condition_ratio," << cond_ratio << "\n";
    gates_csv << "degeneracy,translation_error_ratio," << trans_ratio << "\n";

    // =======================================================================
    // STAGE H — ZERO_NOISE sanity: noise-free correspondences (sigma=0 —
    // apply_noise then becomes a lossless copy) must recover the ground
    // truth to near machine precision, for BOTH scenarios. Per
    // reference_cpu.cpp's header, this is also the independent check on the
    // shared C++ camera-model formula against the independent Python
    // generator that produced the file.
    // =======================================================================
    apply_noise(ds_cl_diverse, 0.0f, 0.0f, true, kSeedCamLidarRecovery, p_obs, uv_obs);
    CUDA_CHECK(cudaMemcpy(d_p_obs, p_obs.data(), p_obs.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_uv_obs, uv_obs.data(), uv_obs.size() * sizeof(float), cudaMemcpyHostToDevice));
    Rigid3 T_zn_cl; std::vector<double> hist_zn_cl; double H21zn[21], g6zn[6];
    run_lm_gpu(d_p_obs, d_uv_obs, kNumCorr, ds_cl_diverse.K, rough_prior(ds_cl_diverse), kMaxLmIters,
              d_block_partials, h_block_partials, T_zn_cl, hist_zn_cl, H21zn, g6zn);
    float zn_cl_rot, zn_cl_trans;
    pose_error(T_zn_cl, ds_cl_diverse, zn_cl_rot, zn_cl_trans);
    const bool zero_noise_cl_pass = (zn_cl_rot <= kZeroNoiseRotTolDeg) && (zn_cl_trans <= kZeroNoiseTransTolM);
    std::printf("[info] zero-noise (camera-LiDAR): rotation error %.4e deg, translation error %.4e mm (tol %.3g deg, %.3g mm)\n",
               static_cast<double>(zn_cl_rot), static_cast<double>(zn_cl_trans) * 1000.0,
               static_cast<double>(kZeroNoiseRotTolDeg), static_cast<double>(kZeroNoiseTransTolM) * 1000.0);
    std::printf("ZERO_NOISE_CAM_LIDAR: %s (noise-free correspondences recover T_camera_lidar to near machine precision)\n", zero_noise_cl_pass ? "PASS" : "FAIL");
    all_pass &= zero_noise_cl_pass;
    gates_csv << "zero_noise_cam_lidar,rot_err_deg," << zn_cl_rot << "\n";
    gates_csv << "zero_noise_cam_lidar,trans_err_m," << zn_cl_trans << "\n";

    apply_noise(ds_cc_diverse, 0.0f, 0.0f, false, kSeedCamCamRecovery, p_obs, uv_obs);
    CUDA_CHECK(cudaMemcpy(d_p_obs, p_obs.data(), p_obs.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_uv_obs, uv_obs.data(), uv_obs.size() * sizeof(float), cudaMemcpyHostToDevice));
    Rigid3 T_zn_cc; std::vector<double> hist_zn_cc; double H21zn2[21], g6zn2[6];
    run_lm_gpu(d_p_obs, d_uv_obs, kNumCorr, ds_cc_diverse.K, rough_prior(ds_cc_diverse), kMaxLmIters,
              d_block_partials, h_block_partials, T_zn_cc, hist_zn_cc, H21zn2, g6zn2);
    float zn_cc_rot, zn_cc_trans;
    pose_error(T_zn_cc, ds_cc_diverse, zn_cc_rot, zn_cc_trans);
    const bool zero_noise_cc_pass = (zn_cc_rot <= kZeroNoiseRotTolDeg) && (zn_cc_trans <= kZeroNoiseTransTolM);
    std::printf("[info] zero-noise (camera-camera): rotation error %.4e deg, translation error %.4e mm (tol %.3g deg, %.3g mm)\n",
               static_cast<double>(zn_cc_rot), static_cast<double>(zn_cc_trans) * 1000.0,
               static_cast<double>(kZeroNoiseRotTolDeg), static_cast<double>(kZeroNoiseTransTolM) * 1000.0);
    std::printf("ZERO_NOISE_CAM_CAM: %s (noise-free correspondences recover T_camera2_camera1 to near machine precision)\n", zero_noise_cc_pass ? "PASS" : "FAIL");
    all_pass &= zero_noise_cc_pass;
    gates_csv << "zero_noise_cam_cam,rot_err_deg," << zn_cc_rot << "\n";
    gates_csv << "zero_noise_cam_cam,trans_err_m," << zn_cc_trans << "\n";

    // =======================================================================
    // Artifacts — gates_metrics.csv, and the overlay.ppm "money shot":
    // camera-LiDAR points reprojected onto the committed background image,
    // BEFORE (the rough prior) vs. AFTER (Stage D's best-of-multistart
    // recovery) calibration, against the TRUE detected pixels.
    // =======================================================================
    {
        std::ofstream f(out_dir + "/gates_metrics.csv");
        if (f.is_open()) f << gates_csv.str();
    }
    std::printf("ARTIFACT: wrote demo/out/gates_metrics.csv\n");

    {
        std::vector<unsigned char> rgb(static_cast<size_t>(bg.w) * bg.h * 3);
        for (size_t i = 0; i < bg.px.size(); ++i) { rgb[i * 3 + 0] = bg.px[i]; rgb[i * 3 + 1] = bg.px[i]; rgb[i * 3 + 2] = bg.px[i]; }

        const Rigid3 T_before = rough_prior(ds_cl_diverse);
        for (const Correspondence& c : ds_cl_diverse.corr) {
            float uv_true[2] = { c.uv_true[0], c.uv_true[1] };
            draw_cross(rgb, bg.w, bg.h, uv_true[0], uv_true[1], 0, 255, 0);   // GREEN: true detected pixel

            float rb[2], Jb[12];
            residual_and_jacobian(T_before, ds_cl_diverse.K, c.p_src, uv_true, rb, Jb);
            draw_cross(rgb, bg.w, bg.h, uv_true[0] + rb[0], uv_true[1] + rb[1], 255, 40, 40);   // RED: before calibration

            float ra[2], Ja[12];
            residual_and_jacobian(T_cl_recovered, ds_cl_diverse.K, c.p_src, uv_true, ra, Ja);
            draw_cross(rgb, bg.w, bg.h, uv_true[0] + ra[0], uv_true[1] + ra[1], 60, 120, 255);  // BLUE: after calibration
        }
        const bool overlay_ok = write_ppm(out_dir + "/overlay.ppm", bg.w, bg.h, rgb);
        if (overlay_ok) std::printf("ARTIFACT: wrote demo/out/overlay.ppm (%dx%d, green=detected red=before-calibration blue=after-calibration)\n", bg.w, bg.h);
        else std::printf("ARTIFACT: FAILED to write demo/out/overlay.ppm\n");
        all_pass &= overlay_ok;
    }

    CUDA_CHECK(cudaFree(d_p_obs));
    CUDA_CHECK(cudaFree(d_uv_obs));
    CUDA_CHECK(cudaFree(d_block_partials));
    CUDA_CHECK(cudaFree(d_farm_T));
    CUDA_CHECK(cudaFree(d_farm_loss));
    CUDA_CHECK(cudaFree(d_farm_rot));
    CUDA_CHECK(cudaFree(d_farm_trans));

    if (all_pass) {
        std::printf("RESULT: PASS (all verification stages passed — jacobian, assembly/trajectory/multistart twins, basin, both recoveries, noise scaling, degeneracy, both zero-noise sanity checks)\n");
        return 0;
    }
    std::printf("RESULT: FAIL (see the stage verdict lines above for which stage(s) failed)\n");
    return 1;
}
