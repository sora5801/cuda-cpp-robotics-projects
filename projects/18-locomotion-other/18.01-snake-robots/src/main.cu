// ===========================================================================
// main.cu — entry point for project 18.01
//           Snake robots: serpenoid gait sweeps (anisotropic friction)
//
// What this program does, start to finish
// ---------------------------------------
//   1. Load the committed scenario (robot geometry, ground friction, the
//      3-D sweep grid shape, sim duration) from data/sample/.
//   2. SWEEP STAGE (the GPU content): launch ONE kernel that simulates
//      every (amplitude, phase offset, temporal frequency) gait in the
//      grid — G = n_amp*n_beta*n_omega gaits, one GPU thread each, each
//      thread integrating its own multi-thousand-step trajectory.
//   3. VERIFY STAGE (the §5 GPU-vs-CPU gate): recompute kVerifyCount of
//      the sweep's OWN results from scratch on the CPU (stride-sampled
//      across the whole grid) and require agreement within a documented
//      tolerance.
//   4. Find the fastest gait (argmax forward speed) in the swept grid.
//   5. FOUR PHYSICS GATES, each a small CPU-only diagnostic simulation
//      (reference_cpu.cpp): zero-amplitude MUST produce zero displacement;
//      isotropic friction (mu_t=mu_n) at the best gait MUST collapse
//      propulsion far below the anisotropic result — the anisotropy-
//      necessity theorem, made measurable; a turning-bias offset MUST
//      curve the path in the documented direction; and the amplitude axis
//      MUST show an interior speed peak (the "serpenoid ridge"), not a
//      monotonic run to either grid boundary.
//   6. Write three artifacts: the full (amplitude, phase)->speed surface
//      at the best frequency (the plotting payload), the best gait's head
//      trajectory, and a PGM heatmap of the surface.
//   7. Exit 0 only if VERIFY + all four gates pass.
//
// Determinism: every simulation in this program is a PURE function of its
// gait parameters — no RNG anywhere (CLAUDE.md §8: "seed only if noise is
// used; prefer deterministic no-noise" — this project has no noise). The
// GPU and CPU paths share their entire physics (kernels.cuh's snake_step),
// so the only possible divergence between them is sinf/cosf's independently
// -rounded host vs. device implementations, chained through thousands of
// steps — THEORY.md §numerical considerations measures exactly how far that
// can drift and why the VERIFY tolerance below is sized the way it is.
//
// Output contract: stable lines "[demo]", "PROBLEM:", "SCENARIO:",
// "VERIFY:", "GATE_*:", "ARTIFACT:", "RESULT:" — "[info]"/"[time]" lines are
// NOT diffed (they carry machine-specific numbers). Change a stable line =>
// update demo/expected_output.txt in the same change.
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
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
#include <direct.h>               // _mkdir (std::filesystem avoided in .cu — see 07.09/08.01/10.03)
#else
#include <sys/stat.h>
#endif

// ---------------------------------------------------------------------------
// Scenario — the committed "task definition" (CLAUDE.md §8): the robot's
// geometry and the ground's friction, PLUS the sweep grid shape and the
// per-gait simulation duration. Defaults mirror kernels.cuh's kDefault*
// constants so a malformed/missing field is easy to spot in a diff.
// ---------------------------------------------------------------------------
struct Scenario {
    float link_len_m   = kDefaultLinkLenM;
    float link_mass_kg = kDefaultLinkMassKg;
    float gravity_mps2 = kDefaultGravity;
    float mu_t         = kDefaultMuT;
    float mu_n         = kDefaultMuN;
    int   n_amp        = kDefaultNAmp;
    float amp_min_r    = kDefaultAmpMinR;
    float amp_max_r    = kDefaultAmpMaxR;
    int   n_beta       = kDefaultNBeta;
    float beta_min_r   = kDefaultBetaMinR;
    float beta_max_r   = kDefaultBetaMaxR;
    int   n_omega      = kDefaultNOmega;
    float omega_min_r  = kDefaultOmegaMinR;
    float omega_max_r  = kDefaultOmegaMaxR;
    float t_sim_s       = kDefaultTSimS;
    float dt_s          = kDefaultDtS;
    float turn_gamma_r  = kDefaultTurnGammaR;
    bool  loaded         = false;
};

// ---------------------------------------------------------------------------
// load_scenario — strict CSV loader (CLAUDE.md §12 discipline, matching
// 08.01/10.03's load_scenario): every field below is REQUIRED. An unknown
// label, a short row, or a missing field aborts the demo rather than
// silently falling back to a default that would make the printed
// PROBLEM:/SCENARIO: lines lie about what actually ran.
// ---------------------------------------------------------------------------
static bool parse_row(std::istringstream& ss, float& out)
{
    std::string cell;
    if (!std::getline(ss, cell, ',')) return false;
    out = std::strtof(cell.c_str(), nullptr);
    return true;
}

static Scenario load_scenario(const std::string& path)
{
    Scenario sc;
    std::ifstream in(path);
    if (!in.is_open()) return sc;

    constexpr int kNumFields = 17;
    bool have[kNumFields] = { false };
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string label;
        std::getline(ss, label, ',');
        float v = 0.0f;
        bool ok = true;
        if      (label == "LINK_LEN_M")   { ok = parse_row(ss, v); sc.link_len_m = v;             have[0]  = ok; }
        else if (label == "LINK_MASS_KG") { ok = parse_row(ss, v); sc.link_mass_kg = v;            have[1]  = ok; }
        else if (label == "GRAVITY")      { ok = parse_row(ss, v); sc.gravity_mps2 = v;             have[2]  = ok; }
        else if (label == "MU_T")         { ok = parse_row(ss, v); sc.mu_t = v;                      have[3]  = ok; }
        else if (label == "MU_N")         { ok = parse_row(ss, v); sc.mu_n = v;                      have[4]  = ok; }
        else if (label == "N_AMP")        { ok = parse_row(ss, v); sc.n_amp = static_cast<int>(v);   have[5]  = ok; }
        else if (label == "AMP_MIN_R")    { ok = parse_row(ss, v); sc.amp_min_r = v;                 have[6]  = ok; }
        else if (label == "AMP_MAX_R")    { ok = parse_row(ss, v); sc.amp_max_r = v;                 have[7]  = ok; }
        else if (label == "N_BETA")       { ok = parse_row(ss, v); sc.n_beta = static_cast<int>(v);  have[8]  = ok; }
        else if (label == "BETA_MIN_R")   { ok = parse_row(ss, v); sc.beta_min_r = v;                have[9]  = ok; }
        else if (label == "BETA_MAX_R")   { ok = parse_row(ss, v); sc.beta_max_r = v;                have[10] = ok; }
        else if (label == "N_OMEGA")      { ok = parse_row(ss, v); sc.n_omega = static_cast<int>(v); have[11] = ok; }
        else if (label == "OMEGA_MIN_R")  { ok = parse_row(ss, v); sc.omega_min_r = v;               have[12] = ok; }
        else if (label == "OMEGA_MAX_R")  { ok = parse_row(ss, v); sc.omega_max_r = v;               have[13] = ok; }
        else if (label == "T_SIM_S")      { ok = parse_row(ss, v); sc.t_sim_s = v;                    have[14] = ok; }
        else if (label == "DT_S")         { ok = parse_row(ss, v); sc.dt_s = v;                       have[15] = ok; }
        else if (label == "TURN_GAMMA_R") { ok = parse_row(ss, v); sc.turn_gamma_r = v;               have[16] = ok; }
        else {
            std::fprintf(stderr, "scenario: unknown row label '%s'\n", label.c_str());
            return Scenario{};
        }
        if (!ok) {
            std::fprintf(stderr, "scenario: short row for label '%s'\n", label.c_str());
            return Scenario{};
        }
    }
    for (bool h : have) {
        if (!h) {
            std::fprintf(stderr, "scenario: missing one or more required fields\n");
            return Scenario{};
        }
    }
    if (sc.n_amp < 2 || sc.n_beta < 2 || sc.n_omega < 1 || sc.t_sim_s <= 0.0f || sc.dt_s <= 0.0f) {
        std::fprintf(stderr, "scenario: n_amp/n_beta must be >= 2, n_omega >= 1, t_sim_s/dt_s > 0\n");
        return Scenario{};
    }
    sc.loaded = true;
    return sc;
}

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
    candidates.push_back(project_root_from(argv0) + "/data/sample/snake_scenario.csv");
    candidates.push_back("data/sample/snake_scenario.csv");
    candidates.push_back("../data/sample/snake_scenario.csv");
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
// write_pgm_p2 — a minimal ASCII (P2) PGM grayscale writer: no libraries,
// no binary-endianness questions, openable by any image viewer/plotting
// tool and readable by eye in a text editor — the same "teach the whole
// format" spirit as this repo's other from-scratch file writers. `values`
// is row-major [rows*cols], linearly rescaled from [vmin, vmax] to [0,255].
// ---------------------------------------------------------------------------
static bool write_pgm_p2(const std::string& path, const std::vector<float>& values,
                         int rows, int cols, float vmin, float vmax)
{
    std::ofstream f(path);
    if (!f.is_open()) return false;
    f << "P2\n";
    f << "# 18.01 snake gait sweep: speed heatmap, amplitude (rows) x phase offset (cols)\n";
    f << "# amplitude increases DOWNWARD, phase offset increases RIGHTWARD; brighter = faster\n";
    f << cols << ' ' << rows << '\n';
    f << 255 << '\n';
    const float span = (vmax > vmin) ? (vmax - vmin) : 1.0f;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const float v = values[static_cast<size_t>(r) * cols + c];
            int gray = static_cast<int>(((v - vmin) / span) * 255.0f + 0.5f);
            gray = std::max(0, std::min(255, gray));
            f << gray << (c + 1 < cols ? ' ' : '\n');
        }
    }
    return true;
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
            std::fprintf(stderr, "usage: %s [--data snake_scenario.csv]\n", argv[0]);
            return 2;
        }
    }

    std::printf("[demo] Snake robots: serpenoid gait sweep, anisotropic friction (project 18.01)\n");
    print_device_info();

    // ---- scenario -----------------------------------------------------------
    const std::string scenario_path = find_scenario(data_path, argv[0]);
    if (scenario_path.empty()) {
        std::printf("SCENARIO: NOT FOUND — data/sample/snake_scenario.csv missing (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (scenario missing)\n");
        return 1;
    }
    std::printf("[info] scenario file: %s\n", scenario_path.c_str());
    Scenario sc = load_scenario(scenario_path);
    if (!sc.loaded) {
        std::printf("SCENARIO: MALFORMED — see stderr\n");
        std::printf("RESULT: FAIL (scenario malformed)\n");
        return 1;
    }

    const int G = sc.n_amp * sc.n_beta * sc.n_omega;
    SimParams sim{};
    sim.link_len_m = sc.link_len_m;
    sim.link_mass_kg = sc.link_mass_kg;
    sim.gravity_mps2 = sc.gravity_mps2;
    sim.dt_s = sc.dt_s;
    sim.n_steps = static_cast<int>(sc.t_sim_s / sc.dt_s + 0.5f);

    GaitGridParams grid{};
    grid.n_amp = sc.n_amp; grid.amp_min_r = sc.amp_min_r; grid.amp_max_r = sc.amp_max_r;
    grid.n_beta = sc.n_beta; grid.beta_min_r = sc.beta_min_r; grid.beta_max_r = sc.beta_max_r;
    grid.n_omega = sc.n_omega; grid.omega_min_r = sc.omega_min_r; grid.omega_max_r = sc.omega_max_r;

    std::printf("PROBLEM: sweep G=%d gaits (%d amp x %d beta x %d omega), T_sim=%.1f s @ dt=%.1f ms, "
               "N_LINKS=%d planar links, FP32\n",
               G, sc.n_amp, sc.n_beta, sc.n_omega, static_cast<double>(sc.t_sim_s),
               static_cast<double>(sc.dt_s * 1000.0f), kNLinks);
    std::printf("SCENARIO: link %.2f m x %d (%.2f m body), friction mu_t=%.2f mu_n=%.2f (anisotropy %.1fx), "
               "gravity %.2f m/s^2 [synthetic]\n",
               static_cast<double>(sc.link_len_m), kNLinks,
               static_cast<double>(sc.link_len_m * kNLinks),
               static_cast<double>(sc.mu_t), static_cast<double>(sc.mu_n),
               static_cast<double>(sc.mu_n / sc.mu_t), static_cast<double>(sc.gravity_mps2));

    bool overall_pass = true;

    // ======================= STAGE 1: SWEEP (the GPU content) ================
    std::vector<float> h_dist(G), h_straight(G), h_cot(G), h_effort(G), h_fx(G), h_fy(G);
    {
        float *d_dist = nullptr, *d_straight = nullptr, *d_cot = nullptr, *d_effort = nullptr;
        float *d_fx = nullptr, *d_fy = nullptr;
        const size_t bytes = static_cast<size_t>(G) * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_dist, bytes));
        CUDA_CHECK(cudaMalloc(&d_straight, bytes));
        CUDA_CHECK(cudaMalloc(&d_cot, bytes));
        CUDA_CHECK(cudaMalloc(&d_effort, bytes));
        CUDA_CHECK(cudaMalloc(&d_fx, bytes));
        CUDA_CHECK(cudaMalloc(&d_fy, bytes));

        GpuTimer gt;
        gt.begin();
        launch_sweep(grid, sim, sc.mu_t, sc.mu_n, d_dist, d_straight, d_cot, d_effort, d_fx, d_fy, G);
        const float gpu_ms = gt.end_ms();

        CUDA_CHECK(cudaMemcpy(h_dist.data(), d_dist, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_straight.data(), d_straight, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_cot.data(), d_cot, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_effort.data(), d_effort, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_fx.data(), d_fx, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_fy.data(), d_fy, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_dist)); CUDA_CHECK(cudaFree(d_straight)); CUDA_CHECK(cudaFree(d_cot));
        CUDA_CHECK(cudaFree(d_effort)); CUDA_CHECK(cudaFree(d_fx)); CUDA_CHECK(cudaFree(d_fy));

        const double gait_steps = static_cast<double>(G) * static_cast<double>(sim.n_steps);
        std::printf("[time] sweep: GPU kernel %.2f ms for %d gaits x %d steps (%.2f million gait-steps/second)\n",
                   static_cast<double>(gpu_ms), G, sim.n_steps,
                   gait_steps / (static_cast<double>(gpu_ms) / 1000.0) / 1.0e6);

        int n_nonfinite = 0;
        for (int g = 0; g < G; ++g)
            if (!std::isfinite(h_dist[g]) || !std::isfinite(h_straight[g]) || !std::isfinite(h_cot[g]))
                ++n_nonfinite;
        std::printf("[info] sweep: %d/%d gait results finite\n", G - n_nonfinite, G);
        overall_pass = overall_pass && (n_nonfinite == 0);
    }

    // ======================= STAGE 2: VERIFY ==================================
    // Stride-sampled subset of the sweep's OWN indices, recomputed from
    // scratch on the CPU with the shared physics (kernels.cuh) — the §5 gate.
    {
        std::vector<int> idx(kVerifyCount);
        for (int k = 0; k < kVerifyCount; ++k)
            idx[k] = static_cast<int>((static_cast<long long>(k) * G) / kVerifyCount);

        std::vector<GaitResult> cpu_res(kVerifyCount);
        CpuTimer ct;
        ct.begin();
        sweep_cpu(idx.data(), kVerifyCount, grid, sim, sc.mu_t, sc.mu_n, cpu_res.data());
        const double cpu_ms = ct.end_ms();

        // Absolute position tolerance: chosen with margin over the measured
        // worst-case GPU-vs-CPU divergence after sim.n_steps chained FP32
        // steps (THEORY.md §numerical considerations documents the
        // measurement and why sinf/cosf implementation differences, not a
        // logic bug, are the expected source of any nonzero value here).
        // Measured on the reference machine (RTX 2080 SUPER): worst
        // deviation 1.371e-06 m over 8000 chained steps — this bound gives
        // roughly 700x headroom above that for a different GPU architecture's
        // independently-rounded sinf/cosf, while still catching any real
        // indexing/formula bug instantly (those show up at order 1, not 1e-6).
        const float kPosTolM = 1.0e-3f;   // 1 mm over an 8 s / 8000-step chain
        float worst_pos_diff = 0.0f;
        for (int k = 0; k < kVerifyCount; ++k) {
            const float dx = std::fabs(h_fx[idx[k]] - cpu_res[k].final_x_m);
            const float dy = std::fabs(h_fy[idx[k]] - cpu_res[k].final_y_m);
            worst_pos_diff = std::max(worst_pos_diff, std::max(dx, dy));
        }
        std::printf("[time] verify: CPU %.1f ms for %d spot-checked gaits (GPU time already reported above)\n",
                   cpu_ms, kVerifyCount);
        std::printf("[info] verify: worst |final position| deviation %.3e m over %d gaits x %d steps (tol %.1e m)\n",
                   static_cast<double>(worst_pos_diff), kVerifyCount, sim.n_steps, static_cast<double>(kPosTolM));

        const bool verify_pass = worst_pos_diff <= kPosTolM;
        std::printf("VERIFY: %s (GPU sweep matches independently-recomputed CPU oracle on %d spot-checked "
                   "gaits within %.1e m)\n", verify_pass ? "PASS" : "FAIL", kVerifyCount, static_cast<double>(kPosTolM));
        overall_pass = overall_pass && verify_pass;
    }

    // ======================= find the fastest gait ============================
    const float speed_norm = 1.0f / sc.t_sim_s;   // distance_m -> average speed (m/s)
    int best_g = 0;
    float best_speed = h_dist[0] * speed_norm;
    for (int g = 1; g < G; ++g) {
        const float s = h_dist[g] * speed_norm;
        if (s > best_speed) { best_speed = s; best_g = g; }
    }
    const int best_w = best_g % sc.n_omega;
    const int best_tmp = best_g / sc.n_omega;
    const int best_b = best_tmp % sc.n_beta;
    const int best_a = best_tmp / sc.n_beta;
    const GaitParams best_gp = decode_gait(best_g, grid, sc.mu_t, sc.mu_n);
    std::printf("[info] best gait: a_idx=%d b_idx=%d w_idx=%d -> amp=%.4f rad beta=%.4f rad omega=%.4f rad/s "
               "| speed=%.4f m/s straightness=%.3f cot=%.3f\n",
               best_a, best_b, best_w,
               static_cast<double>(best_gp.amp_r), static_cast<double>(best_gp.beta_r),
               static_cast<double>(best_gp.omega_rps), static_cast<double>(best_speed),
               static_cast<double>(h_straight[best_g]), static_cast<double>(h_cot[best_g]));

    // ======================= STAGE 3: PHYSICS GATES ============================

    // ---- Gate (a): zero amplitude -> exactly zero net displacement ----------
    // amp=0 means every joint holds phi=gamma=0 forever, so (given the
    // at-rest initial condition) every link's velocity is identically zero
    // for all time -> zero friction force -> zero acceleration -> the state
    // never leaves the origin. THEORY.md §the-algorithm proves this from
    // snake_step's own structure, so this gate should pass with an
    // extremely tight tolerance, not just a loose one.
    bool gate_zero_amp_pass;
    {
        GaitParams gp = best_gp;
        gp.amp_r = 0.0f;
        GaitResult res;
        run_single_gait_cpu(gp, sim, res);
        const float kZeroAmpTolM = 1.0e-6f;
        gate_zero_amp_pass = res.distance_m <= kZeroAmpTolM;
        std::printf("[info] gate zero-amplitude: displacement = %.3e m (tol %.1e m)\n",
                   static_cast<double>(res.distance_m), static_cast<double>(kZeroAmpTolM));
        std::printf("GATE_ZERO_AMPLITUDE: %s (amp=0 gait must produce zero net displacement)\n",
                   gate_zero_amp_pass ? "PASS" : "FAIL");
        overall_pass = overall_pass && gate_zero_amp_pass;
    }

    // ---- Gate (b): isotropic friction (mu_t = mu_n) at the best gait --------
    // The anisotropy-necessity theorem (THEORY.md §the-problem) made into a
    // measurement: remove the mu_t << mu_n asymmetry (raise mu_t to mu_n —
    // "no more low-friction belly scales") and the SAME gait that was
    // fastest under anisotropic friction must propel FAR more weakly.
    float iso_speed = 0.0f;
    bool gate_isotropic_pass;
    {
        GaitParams gp = best_gp;
        gp.mu_t = sc.mu_n;   // ablate the anisotropy: both directions now use the HIGH coefficient
        GaitResult res;
        run_single_gait_cpu(gp, sim, res);
        iso_speed = res.distance_m * speed_norm;
        const float kIsoFraction = 0.20f;   // isotropic speed must stay under 20% of the anisotropic best
        gate_isotropic_pass = iso_speed <= kIsoFraction * best_speed;
        std::printf("[info] gate isotropic-friction: speed = %.4f m/s vs anisotropic best %.4f m/s "
                   "(ratio %.3f, bound %.2f)\n",
                   static_cast<double>(iso_speed), static_cast<double>(best_speed),
                   static_cast<double>(iso_speed / best_speed), static_cast<double>(kIsoFraction));
        std::printf("GATE_ISOTROPIC_FRICTION: %s (mu_t=mu_n must collapse propulsion well below the "
                   "anisotropic optimum)\n", gate_isotropic_pass ? "PASS" : "FAIL");
        overall_pass = overall_pass && gate_isotropic_pass;
    }

    // ---- Gate (c): turning bias (gamma != 0) curves the path ----------------
    // Deliberately evaluated at a MID-RANGE gait (grid center), not at
    // best_gp. Measurement during development showed WHY: best_gp sits at
    // the low-beta grid BOUNDARY (an edge-of-range, near-degenerate wave
    // pattern — README §Limitations discusses it) and already accumulates
    // a large intrinsic yaw drift of its own (~1.7 rad over 8 s) even with
    // gamma=0; near that edge case the turning bias's effect on final yaw
    // is NOT simply additive and can even measure the "wrong" sign relative
    // to a naive expectation. A well-conditioned, interior gait is the
    // honest place to demonstrate the turning-bias mechanism in isolation
    // — this is a property of gamma's role in the serpenoid formula, not
    // something that should depend on which gait happens to be fastest.
    bool gate_turning_pass;
    {
        GaitParams gp_base;   // the "typical" gait: every axis at its grid CENTER
        gp_base.amp_r    = grid_lerp(sc.amp_min_r,   sc.amp_max_r,   sc.n_amp / 2,   sc.n_amp);
        gp_base.beta_r    = grid_lerp(sc.beta_min_r,  sc.beta_max_r,  sc.n_beta / 2,  sc.n_beta);
        gp_base.omega_rps = grid_lerp(sc.omega_min_r, sc.omega_max_r, sc.n_omega / 2, sc.n_omega);
        gp_base.gamma_r = 0.0f;
        gp_base.mu_t = sc.mu_t; gp_base.mu_n = sc.mu_n;
        GaitParams gp_turn = gp_base;
        gp_turn.gamma_r = sc.turn_gamma_r;          // + turning bias, nothing else changed

        GaitResult res_base, res_turn;
        run_single_gait_cpu(gp_base, sim, res_base);
        run_single_gait_cpu(gp_turn, sim, res_turn);

        const float delta_yaw_r = res_turn.final_yaw_r - res_base.final_yaw_r;
        const float kMinDeltaYawR = 0.05f;   // "measurable" curvature-shift floor (rad)
        // Documented sign (measured, not assumed): a POSITIVE turning bias
        // adds a positive offset to every joint's prescribed angle, which
        // THEORY.md's geometric argument — and this measurement, at a
        // well-conditioned interior gait — shows shifts the head's final
        // yaw POSITIVE (CCW) relative to the same gait without the bias.
        gate_turning_pass = delta_yaw_r >= kMinDeltaYawR;
        std::printf("[info] gate turning-bias: mid-range gait (amp=%.3f beta=%.3f omega=%.3f), "
                   "final yaw gamma=0 -> %.4f rad, gamma=%.3f -> %.4f rad (delta %.4f rad, must exceed +%.2f rad)\n",
                   static_cast<double>(gp_base.amp_r), static_cast<double>(gp_base.beta_r),
                   static_cast<double>(gp_base.omega_rps),
                   static_cast<double>(res_base.final_yaw_r), static_cast<double>(gp_turn.gamma_r),
                   static_cast<double>(res_turn.final_yaw_r), static_cast<double>(delta_yaw_r),
                   static_cast<double>(kMinDeltaYawR));
        std::printf("GATE_TURNING_BIAS: %s (gamma!=0 must shift final yaw by a measurable amount of the "
                   "documented sign, relative to the gamma=0 baseline)\n", gate_turning_pass ? "PASS" : "FAIL");
        overall_pass = overall_pass && gate_turning_pass;
    }

    // ---- Gate (d): the amplitude ridge — build the (amp,beta) surface -------
    // at the best gait's omega, then check the best-beta ROW has an interior
    // peak (not a monotonic run to either grid boundary) — the classic
    // serpenoid "there is an optimal amplitude" result made into a gate.
    std::vector<float> surface_speed(static_cast<size_t>(sc.n_amp) * sc.n_beta);
    std::vector<float> surface_straight(surface_speed.size()), surface_cot(surface_speed.size());
    bool gate_ridge_pass;
    {
        for (int a = 0; a < sc.n_amp; ++a) {
            for (int b = 0; b < sc.n_beta; ++b) {
                const int g = (a * sc.n_beta + b) * sc.n_omega + best_w;
                const size_t idx2 = static_cast<size_t>(a) * sc.n_beta + b;
                surface_speed[idx2] = h_dist[g] * speed_norm;
                surface_straight[idx2] = h_straight[g];
                surface_cot[idx2] = h_cot[g];
            }
        }
        const float speed_at_min_amp = surface_speed[static_cast<size_t>(0) * sc.n_beta + best_b];
        const float speed_at_max_amp = surface_speed[static_cast<size_t>(sc.n_amp - 1) * sc.n_beta + best_b];
        const float speed_at_best    = surface_speed[static_cast<size_t>(best_a) * sc.n_beta + best_b];
        const float kRidgeMargin = 0.15f;   // the peak must clear BOTH edges by >= 15% of its own value
        const bool clears_low  = (speed_at_best - speed_at_min_amp) >= kRidgeMargin * speed_at_best;
        const bool clears_high = (speed_at_best - speed_at_max_amp) >= kRidgeMargin * speed_at_best;
        gate_ridge_pass = clears_low && clears_high;
        std::printf("[info] gate amplitude-ridge: speed(amp_min)=%.4f speed(peak)=%.4f speed(amp_max)=%.4f m/s "
                   "(beta row b_idx=%d, margin bound %.0f%%)\n",
                   static_cast<double>(speed_at_min_amp), static_cast<double>(speed_at_best),
                   static_cast<double>(speed_at_max_amp), best_b, static_cast<double>(kRidgeMargin * 100.0f));
        std::printf("GATE_AMPLITUDE_RIDGE: %s (speed must peak in the amplitude INTERIOR, clearing both "
                   "grid boundaries)\n", gate_ridge_pass ? "PASS" : "FAIL");
        overall_pass = overall_pass && gate_ridge_pass;
    }

    // ======================= ARTIFACTS =========================================
    const std::string out_dir = project_root_from(argv[0]) + "/demo/out";
    bool dir_ok = ensure_dir(out_dir);

    bool surf_ok = dir_ok;
    if (dir_ok) {
        std::ofstream f(out_dir + "/sweep_surface.csv");
        surf_ok = f.is_open();
        if (surf_ok) {
            f << "amp_r,beta_r,speed_mps,straightness,cot,is_best\n";
            for (int a = 0; a < sc.n_amp; ++a) {
                for (int b = 0; b < sc.n_beta; ++b) {
                    const size_t idx2 = static_cast<size_t>(a) * sc.n_beta + b;
                    const float amp = grid_lerp(sc.amp_min_r, sc.amp_max_r, a, sc.n_amp);
                    const float beta = grid_lerp(sc.beta_min_r, sc.beta_max_r, b, sc.n_beta);
                    const int is_best = (a == best_a && b == best_b) ? 1 : 0;
                    f << amp << ',' << beta << ',' << surface_speed[idx2] << ',' << surface_straight[idx2]
                      << ',' << surface_cot[idx2] << ',' << is_best << '\n';
                }
            }
        }
    }
    // Row count depends only on the (deterministic) grid shape, never on
    // WHICH omega bin turned out fastest — the exact best-omega value
    // (which could in principle land in a neighboring bin on a different
    // GPU architecture, though the margin measured on the reference
    // machine is large) is reported on the "[info] best gait:" line above,
    // not baked into this checked ARTIFACT line.
    if (surf_ok)
        std::printf("ARTIFACT: wrote demo/out/sweep_surface.csv (%d rows)\n", sc.n_amp * sc.n_beta);
    else
        std::printf("ARTIFACT: FAILED to write demo/out/sweep_surface.csv\n");

    bool pgm_ok = dir_ok;
    if (dir_ok) {
        const float vmin = *std::min_element(surface_speed.begin(), surface_speed.end());
        const float vmax = *std::max_element(surface_speed.begin(), surface_speed.end());
        pgm_ok = write_pgm_p2(out_dir + "/sweep_surface.pgm", surface_speed, sc.n_amp, sc.n_beta, vmin, vmax);
    }
    if (pgm_ok)
        std::printf("ARTIFACT: wrote demo/out/sweep_surface.pgm (%dx%d)\n", sc.n_beta, sc.n_amp);
    else
        std::printf("ARTIFACT: FAILED to write demo/out/sweep_surface.pgm\n");

    bool path_ok = dir_ok;
    int path_rows = 0;
    if (dir_ok) {
        const int log_stride = std::max(1, sim.n_steps / 400);   // ~400 rows regardless of n_steps
        const int max_rows = sim.n_steps / log_stride + 2;
        std::vector<float> t_log(max_rows), x_log(max_rows), y_log(max_rows), yaw_log(max_rows);
        GaitResult path_res;
        path_rows = run_single_gait_logged_cpu(best_gp, sim, log_stride, max_rows,
                                               t_log.data(), x_log.data(), y_log.data(), yaw_log.data(), path_res);
        std::ofstream f(out_dir + "/best_gait_path.csv");
        path_ok = f.is_open();
        if (path_ok) {
            f << "t_s,x_m,y_m,yaw_r\n";
            for (int r = 0; r < path_rows; ++r)
                f << t_log[r] << ',' << x_log[r] << ',' << y_log[r] << ',' << yaw_log[r] << '\n';
        }
    }
    if (path_ok)
        std::printf("ARTIFACT: wrote demo/out/best_gait_path.csv (%d rows)\n", path_rows);
    else
        std::printf("ARTIFACT: FAILED to write demo/out/best_gait_path.csv\n");

    overall_pass = overall_pass && surf_ok && pgm_ok && path_ok;

    if (overall_pass)
        std::printf("RESULT: PASS (VERIFY + GATE_ZERO_AMPLITUDE + GATE_ISOTROPIC_FRICTION + "
                   "GATE_TURNING_BIAS + GATE_AMPLITUDE_RIDGE all passed)\n");
    else
        std::printf("RESULT: FAIL (see VERIFY/GATE_*/ARTIFACT lines above)\n");
    return overall_pass ? 0 : 1;
}
