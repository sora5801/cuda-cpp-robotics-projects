// ===========================================================================
// main.cu — entry point for project 10.03
//           Massively parallel robot sim (Isaac-Gym-style: one robot,
//           10,000 environments)
//
// What this program does, start to finish
// ---------------------------------------
//   1. Print the banner + GPU info; load the committed farm scenario
//      (N, T, episode cap, seed, domain-randomization ranges, controller
//      gains) from data/sample/.
//   2. VERIFY STAGE (the §5 GPU-vs-CPU gate): init + step kVerifyEnvs=256
//      environments for kVerifySteps=220 ticks (> the 200-step episode
//      cap, so every environment passes through exactly one CAP-triggered
//      reset) on BOTH the GPU kernels and the CPU oracle, with IDENTICAL
//      seeds/parameters, and require agreement within documented
//      tolerances.
//   3. FARM STAGE: init + step the FULL N-environment farm for T ticks —
//      ONE kernel launch covers the whole run (see kernels.cu's header for
//      why) — then check every environment's state is finite and every
//      environment's reset_count falls in a documented, largely provable
//      range (kernels.cuh: episode_cap=200, T=1000 => every environment
//      resets AT LEAST floor(T/episode_cap)=5 times). Writes
//      demo/out/env_metrics.csv.
//   4. ENERGY-CONSERVATION EXPERIMENT: a single undriven, unbounded
//      cart-pole trajectory (CPU-only; see reference_cpu.cpp) exposing the
//      RK4 integrator's own drift, independent of the farm. Writes
//      demo/out/energy_drift.csv.
//   5. Exit 0 only if VERIFY + FARM + ENERGY all pass.
//
// The three ingredients of a GPU RL training farm, and where each one
// lives in this file (THEORY.md derives all three properly):
//   - PARALLEL ENVIRONMENTS      -> the FarmBuffers allocated at farm scale
//                                    and stepped by launch_farm_step (§3).
//   - DOMAIN RANDOMIZATION       -> launch_farm_init's dr_mc/dr_mp/dr_l
//                                    arguments, loaded from the scenario.
//   - EPISODE RESET              -> fused into step_farm_kernel; this file
//                                    only reads the resulting reset_count.
// 12.06 (RL training kernels, cited in docs/SYSTEM_DESIGN.md Chain C) reuses
// exactly this pattern, replacing the fixed gains here with a policy that
// learns from steps_balanced-shaped rewards.
//
// Determinism: every random draw in this program (domain randomization,
// initial-angle resets) comes from the shared xorshift32 stream in
// kernels.cuh, seeded from the scenario's SEED field — the run is
// bit-reproducible on a GIVEN machine. Across DIFFERENT GPUs/driver
// versions, the device sinf/cosf implementation may differ in the last
// ULP from another device's (or from the host CRT's) — the same honest
// caveat 08.01 documents — so stable output lines below carry NO
// trajectory numbers, only PASS/FAIL against thresholds with measured,
// documented margin (THEORY.md §numerical considerations).
//
// Output contract: stable lines "[demo]", "PROBLEM:", "SCENARIO:",
// "VERIFY:", "FARM:", "ENERGY:", "ARTIFACT:", "RESULT:" — "[info]"/"[time]"
// unchecked. Change a stable line => update demo/expected_output.txt in
// the same commit.
//
// Read this first, then kernels.cuh -> reference_cpu.cpp -> kernels.cu.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

#include <algorithm>   // std::min/std::max — used throughout the reporting/gating code below
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
#include <direct.h>               // _mkdir (std::filesystem avoided in .cu — see 07.09/08.01)
#else
#include <sys/stat.h>
#endif

// ---------------------------------------------------------------------------
// FarmScenario — the committed "task definition" for this project: how
// many environments, how long to run, the domain-randomization envelope,
// and the fixed controller gains. This is data (CLAUDE.md §8), not a
// tuning knob buried in code — see data/README.md for the file format.
// Defaults mirror kernels.cuh's kDefault* constants so a malformed/missing
// field is easy to spot in a diff against the documented committed values.
// ---------------------------------------------------------------------------
struct FarmScenario {
    int      N            = kDefaultN;
    int      T_farm       = kDefaultTFarm;
    int      episode_cap  = kDefaultEpisodeCap;
    uint32_t seed          = kDefaultSeed;
    float    dr_mass_cart = kDefaultDrMassCart;
    float    dr_mass_pole = kDefaultDrMassPole;
    float    dr_len       = kDefaultDrLen;
    float    theta0_range = kDefaultTheta0Range;
    float    Kx           = kDefaultKx;
    float    Kxd          = kDefaultKxd;
    float    Kth          = kDefaultKth;
    float    Kthd         = kDefaultKthd;
    bool     loaded        = false;
};

// ---------------------------------------------------------------------------
// load_scenario — strict CSV loader (CLAUDE.md §12 discipline, matching
// 08.01's load_scenario): every field below is REQUIRED; an unknown label,
// a short row, or a missing field aborts the demo rather than silently
// falling back to a default that would make the "PROBLEM:"/"SCENARIO:"
// lines lie about what actually ran.
// ---------------------------------------------------------------------------
static bool parse_row(std::istringstream& ss, float& out)
{
    std::string cell;
    if (!std::getline(ss, cell, ',')) return false;
    out = std::strtof(cell.c_str(), nullptr);
    return true;
}

static FarmScenario load_scenario(const std::string& path)
{
    FarmScenario sc;
    std::ifstream in(path);
    if (!in.is_open()) return sc;

    // Track which required fields we have actually SEEN — a field left at
    // its struct-default without appearing in the file is a malformed
    // scenario, not a silently-accepted default (strict-loader discipline).
    bool have[12] = { false };
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string label;
        std::getline(ss, label, ',');
        float v = 0.0f;
        bool ok = true;
        if      (label == "N")             { ok = parse_row(ss, v); sc.N = static_cast<int>(v);            have[0]  = ok; }
        else if (label == "T_FARM")        { ok = parse_row(ss, v); sc.T_farm = static_cast<int>(v);       have[1]  = ok; }
        else if (label == "EPISODE_CAP")   { ok = parse_row(ss, v); sc.episode_cap = static_cast<int>(v);  have[2]  = ok; }
        else if (label == "SEED")          { ok = parse_row(ss, v); sc.seed = static_cast<uint32_t>(v);    have[3]  = ok; }
        else if (label == "DR_MASS_CART")  { ok = parse_row(ss, v); sc.dr_mass_cart = v;                    have[4]  = ok; }
        else if (label == "DR_MASS_POLE")  { ok = parse_row(ss, v); sc.dr_mass_pole = v;                    have[5]  = ok; }
        else if (label == "DR_LEN")        { ok = parse_row(ss, v); sc.dr_len = v;                          have[6]  = ok; }
        else if (label == "THETA0_RANGE")  { ok = parse_row(ss, v); sc.theta0_range = v;                    have[7]  = ok; }
        else if (label == "KX")            { ok = parse_row(ss, v); sc.Kx = v;                               have[8]  = ok; }
        else if (label == "KXD")           { ok = parse_row(ss, v); sc.Kxd = v;                              have[9]  = ok; }
        else if (label == "KTH")           { ok = parse_row(ss, v); sc.Kth = v;                              have[10] = ok; }
        else if (label == "KTHD")          { ok = parse_row(ss, v); sc.Kthd = v;                             have[11] = ok; }
        else {
            std::fprintf(stderr, "scenario: unknown row label '%s'\n", label.c_str());
            return FarmScenario{};
        }
        if (!ok) {
            std::fprintf(stderr, "scenario: short row for label '%s'\n", label.c_str());
            return FarmScenario{};
        }
    }
    for (bool h : have) {
        if (!h) {
            std::fprintf(stderr, "scenario: missing one or more required fields\n");
            return FarmScenario{};
        }
    }
    if (sc.N < kVerifyEnvs || sc.T_farm < 1 || sc.episode_cap < 1) {
        std::fprintf(stderr, "scenario: N must be >= %d, T_FARM and EPISODE_CAP must be >= 1\n", kVerifyEnvs);
        return FarmScenario{};
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
    candidates.push_back(project_root_from(argv0) + "/data/sample/farm_scenario.csv");
    candidates.push_back("data/sample/farm_scenario.csv");
    candidates.push_back("../data/sample/farm_scenario.csv");
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
// alloc_device_farm / alloc_host_farm / free_device_farm — small helpers
// that keep main() readable. Each FarmBuffers instance owns N-sized
// arrays for every field kernels.cuh declares; device and host versions
// share the exact same struct TYPE (kernels.cuh) with device vs. host
// pointers inside it.
// ---------------------------------------------------------------------------
static FarmBuffers alloc_device_farm(int N)
{
    FarmBuffers b{};
    CUDA_CHECK(cudaMalloc(&b.x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.xdot, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.theta, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.thdot, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.mass_cart, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.mass_pole, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.pole_half_len, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.rng_state, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&b.ep_step, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&b.steps_balanced, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&b.reset_count, N * sizeof(int)));
    return b;
}

static void free_device_farm(FarmBuffers& b)
{
    CUDA_CHECK(cudaFree(b.x));            CUDA_CHECK(cudaFree(b.xdot));
    CUDA_CHECK(cudaFree(b.theta));        CUDA_CHECK(cudaFree(b.thdot));
    CUDA_CHECK(cudaFree(b.mass_cart));    CUDA_CHECK(cudaFree(b.mass_pole));
    CUDA_CHECK(cudaFree(b.pole_half_len));CUDA_CHECK(cudaFree(b.rng_state));
    CUDA_CHECK(cudaFree(b.ep_step));      CUDA_CHECK(cudaFree(b.steps_balanced));
    CUDA_CHECK(cudaFree(b.reset_count));
}

// Host-side storage for a FarmBuffers: std::vectors own the memory; the
// FarmBuffers struct returned just points at vector.data() (the caller
// must keep the vectors alive as long as the FarmBuffers is used — a
// standard "view" idiom, documented here rather than hidden).
struct HostFarmStorage {
    std::vector<float> x, xdot, theta, thdot, mass_cart, mass_pole, pole_half_len;
    std::vector<uint32_t> rng_state;
    std::vector<int> ep_step, steps_balanced, reset_count;

    explicit HostFarmStorage(int N)
        : x(N), xdot(N), theta(N), thdot(N), mass_cart(N), mass_pole(N), pole_half_len(N),
          rng_state(N), ep_step(N), steps_balanced(N), reset_count(N) {}

    FarmBuffers view()
    {
        FarmBuffers b{};
        b.x = x.data(); b.xdot = xdot.data(); b.theta = theta.data(); b.thdot = thdot.data();
        b.mass_cart = mass_cart.data(); b.mass_pole = mass_pole.data(); b.pole_half_len = pole_half_len.data();
        b.rng_state = rng_state.data();
        b.ep_step = ep_step.data(); b.steps_balanced = steps_balanced.data(); b.reset_count = reset_count.data();
        return b;
    }
};

// device_to_host — copy every field of a device FarmBuffers into a
// HostFarmStorage of the same size (used to bring GPU results back for
// comparison/reporting; every copy is checked, as always).
static void device_to_host(const FarmBuffers& d, HostFarmStorage& h, int N)
{
    CUDA_CHECK(cudaMemcpy(h.x.data(), d.x, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.xdot.data(), d.xdot, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.theta.data(), d.theta, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.thdot.data(), d.thdot, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.mass_cart.data(), d.mass_cart, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.mass_pole.data(), d.mass_pole, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.pole_half_len.data(), d.pole_half_len, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.rng_state.data(), d.rng_state, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.ep_step.data(), d.ep_step, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.steps_balanced.data(), d.steps_balanced, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.reset_count.data(), d.reset_count, N * sizeof(int), cudaMemcpyDeviceToHost));
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
            std::fprintf(stderr, "usage: %s [--data farm_scenario.csv]\n", argv[0]);
            return 2;
        }
    }

    std::printf("[demo] Massively parallel robot sim: N-environment cart-pole farm (project 10.03)\n");
    print_device_info();

    // ---- scenario -------------------------------------------------------
    const std::string scenario_path = find_scenario(data_path, argv[0]);
    if (scenario_path.empty()) {
        std::printf("SCENARIO: NOT FOUND — data/sample/farm_scenario.csv missing (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (scenario missing)\n");
        return 1;
    }
    std::printf("[info] scenario file: %s\n", scenario_path.c_str());
    FarmScenario sc = load_scenario(scenario_path);
    if (!sc.loaded) {
        std::printf("SCENARIO: MALFORMED — see stderr\n");
        std::printf("RESULT: FAIL (scenario malformed)\n");
        return 1;
    }

    std::printf("PROBLEM: N=%d environments x T=%d steps @ dt=%.2f s, cart-pole farm (Isaac-Gym-style), FP32\n",
               sc.N, sc.T_farm, static_cast<double>(kDt));
    std::printf("SCENARIO: domain-randomized mc +/-%.0f%%, mp +/-%.0f%%, l +/-%.0f%%; episode cap %d steps; "
               "fixed pole-placement balance gains [synthetic]\n",
               static_cast<double>(sc.dr_mass_cart * 100.0f),
               static_cast<double>(sc.dr_mass_pole * 100.0f),
               static_cast<double>(sc.dr_len * 100.0f),
               sc.episode_cap);

    bool overall_pass = true;

    // ======================= STAGE 1: VERIFY =================================
    // A kVerifyEnvs-environment subset, stepped kVerifySteps (> episode_cap)
    // ticks on BOTH paths from IDENTICAL seeds — exercises domain
    // randomization, the controller, RK4 integration, AND (because
    // kVerifySteps > episode_cap) exactly one CAP-triggered reset per env.
    {
        FarmBuffers d_buf = alloc_device_farm(kVerifyEnvs);
        HostFarmStorage h_gpu(kVerifyEnvs);
        HostFarmStorage h_cpu(kVerifyEnvs);
        FarmBuffers cpu_view = h_cpu.view();

        GpuTimer gt;
        gt.begin();
        launch_farm_init(d_buf, kVerifyEnvs, sc.seed, sc.dr_mass_cart, sc.dr_mass_pole, sc.dr_len, sc.theta0_range);
        launch_farm_step(d_buf, kVerifyEnvs, kVerifySteps, sc.Kx, sc.Kxd, sc.Kth, sc.Kthd,
                         sc.theta0_range, sc.episode_cap);
        const float gpu_ms = gt.end_ms();
        device_to_host(d_buf, h_gpu, kVerifyEnvs);
        free_device_farm(d_buf);

        CpuTimer ct;
        ct.begin();
        farm_init_cpu(cpu_view, kVerifyEnvs, sc.seed, sc.dr_mass_cart, sc.dr_mass_pole, sc.dr_len, sc.theta0_range);
        farm_step_cpu(cpu_view, kVerifyEnvs, kVerifySteps, sc.Kx, sc.Kxd, sc.Kth, sc.Kthd,
                      sc.theta0_range, sc.episode_cap);
        const double cpu_ms = ct.end_ms();

        // (a) State comparison — ABSOLUTE tolerance (not relative): every
        // state component legitimately passes through ~0 during balancing
        // (e.g. xdot oscillates around zero), where a relative tolerance
        // is meaningless. kStateTol is calibrated below from the measured
        // worst case (THEORY.md §numerical considerations documents both
        // the number and the calibration).
        const float kStateTol = 1.0e-3f;   // see THEORY.md for the measured value this covers
        float worst_state_diff = 0.0f;
        for (int i = 0; i < kVerifyEnvs; ++i) {
            const float d0 = std::fabs(h_gpu.x[i] - h_cpu.x[i]);
            const float d1 = std::fabs(h_gpu.xdot[i] - h_cpu.xdot[i]);
            const float d2 = std::fabs(h_gpu.theta[i] - h_cpu.theta[i]);
            const float d3 = std::fabs(h_gpu.thdot[i] - h_cpu.thdot[i]);
            worst_state_diff = std::max(worst_state_diff, std::max(std::max(d0, d1), std::max(d2, d3)));
        }

        // (b) reset_count must match EXACTLY — it is driven by an INTEGER
        // step counter reaching episode_cap, which is bit-for-bit
        // identical on both paths regardless of any float rounding
        // difference (kernels.cuh's design note). A mismatch here means a
        // real logic bug (indexing, off-by-one in the cap check), not
        // floating-point noise — so this gate demands EQUALITY, not tolerance.
        int reset_mismatches = 0;
        for (int i = 0; i < kVerifyEnvs; ++i)
            if (h_gpu.reset_count[i] != h_cpu.reset_count[i]) ++reset_mismatches;

        // (c) steps_balanced may differ by a SMALL integer count: it is
        // classified from theta against kBalancedTheta every tick, and a
        // handful of ticks near that boundary CAN flip classification
        // under the sinf/cosf ULP-level divergence documented in
        // kernels.cuh — so this gate allows a small, documented slack.
        const int kBalancedSlack = 3;
        int worst_balanced_diff = 0;
        for (int i = 0; i < kVerifyEnvs; ++i)
            worst_balanced_diff = std::max(worst_balanced_diff,
                                           std::abs(h_gpu.steps_balanced[i] - h_cpu.steps_balanced[i]));

        std::printf("[info] verify: worst |state| deviation %.3e over %d envs x %d steps (tol %.1e)\n",
                   static_cast<double>(worst_state_diff), kVerifyEnvs, kVerifySteps,
                   static_cast<double>(kStateTol));
        std::printf("[info] verify: reset_count exact matches %d/%d; worst steps_balanced abs diff %d (slack %d)\n",
                   kVerifyEnvs - reset_mismatches, kVerifyEnvs, worst_balanced_diff, kBalancedSlack);
        std::printf("[time] verify (envs=%d, steps=%d): CPU %.1f ms | GPU %.3f ms | speed-up %.0fx "
                   "(teaching artifact; small problem size, launch overhead dominates)\n",
                   kVerifyEnvs, kVerifySteps, cpu_ms, static_cast<double>(gpu_ms),
                   cpu_ms / (static_cast<double>(gpu_ms) > 0.0 ? static_cast<double>(gpu_ms) : 1.0));

        const bool verify_pass = (worst_state_diff <= kStateTol)
                               && (reset_mismatches == 0)
                               && (worst_balanced_diff <= kBalancedSlack);
        std::printf("VERIFY: %s (GPU state matches CPU reference within tol %.1e; reset_count exact; "
                   "steps_balanced within %d)\n",
                   verify_pass ? "PASS" : "FAIL", static_cast<double>(kStateTol), kBalancedSlack);
        overall_pass = overall_pass && verify_pass;
    }

    // ======================= STAGE 2: FARM ====================================
    // The full N-environment farm, one kernel launch covering all T ticks
    // (kernels.cu's header explains why the whole run fuses into one call).
    std::vector<int> farm_steps_balanced, farm_reset_count;
    std::vector<float> farm_mass_cart, farm_mass_pole, farm_pole_half_len;
    double aggregate_env_steps_per_sec = 0.0;
    {
        FarmBuffers d_buf = alloc_device_farm(sc.N);
        HostFarmStorage h(sc.N);

        launch_farm_init(d_buf, sc.N, sc.seed, sc.dr_mass_cart, sc.dr_mass_pole, sc.dr_len, sc.theta0_range);

        // Time ONLY the step kernel — this single launch runs the WHOLE
        // farm (sc.N environments x sc.T_farm ticks each), so its elapsed
        // time is exactly what "aggregate env-steps/second" divides by
        // (contrast 08.01, which times one 20ms CONTROL TICK at a time).
        GpuTimer gt;
        gt.begin();
        launch_farm_step(d_buf, sc.N, sc.T_farm, sc.Kx, sc.Kxd, sc.Kth, sc.Kthd,
                         sc.theta0_range, sc.episode_cap);
        const float gpu_ms = gt.end_ms();

        device_to_host(d_buf, h, sc.N);
        free_device_farm(d_buf);

        const double total_env_steps = static_cast<double>(sc.N) * static_cast<double>(sc.T_farm);
        aggregate_env_steps_per_sec = total_env_steps / (static_cast<double>(gpu_ms) / 1000.0);

        // ---- gate (a): every environment's state must be finite ---------
        int n_nonfinite = 0;
        for (int i = 0; i < sc.N; ++i) {
            if (!std::isfinite(h.x[i]) || !std::isfinite(h.xdot[i]) ||
                !std::isfinite(h.theta[i]) || !std::isfinite(h.thdot[i]))
                ++n_nonfinite;
        }

        // ---- gate (b): reset_count in a documented deterministic range --
        // PROVABLE lower bound: an episode lasts AT MOST episode_cap ticks
        // (it resets the instant it hits the cap, if it has not failed
        // earlier), so over T_farm ticks every environment resets AT LEAST
        // floor(T_farm/episode_cap) times — this is a fact about the code's
        // control flow, true regardless of dynamics. kFarmResetMax is an
        // EMPIRICAL bound with margin (see THEORY.md for the measured
        // value this covers): the pole-placement controller is robust
        // enough across the domain-randomization envelope that failures
        // are rare-to-absent, so resets stay close to the provable minimum.
        const int kFarmResetMin = sc.T_farm / sc.episode_cap;   // provable, from the code's own logic
        // kFarmResetMax = 12: measured on the reference machine (RTX 2080
        // SUPER), EVERY one of the 10,000 environments resets EXACTLY 5
        // times (== kFarmResetMin) — the pole-placement controller is
        // robust across the entire domain-randomization envelope, so no
        // environment ever fails early. 12 is a >2x margin above that
        // measured value: generous enough to absorb the rare boundary
        // case a different GPU's sinf/cosf might flip, while still being
        // a REAL gate that fails if a change makes the controller
        // meaningfully less robust (THEORY.md documents the measurement).
        const int kFarmResetMax = 12;
        int rmin = h.reset_count[0], rmax = h.reset_count[0];
        long long rsum = 0;
        int steps_balanced_min = h.steps_balanced[0], steps_balanced_max = h.steps_balanced[0];
        long long steps_balanced_sum = 0;
        int n_out_of_range = 0;
        for (int i = 0; i < sc.N; ++i) {
            rmin = std::min(rmin, h.reset_count[i]);
            rmax = std::max(rmax, h.reset_count[i]);
            rsum += h.reset_count[i];
            steps_balanced_min = std::min(steps_balanced_min, h.steps_balanced[i]);
            steps_balanced_max = std::max(steps_balanced_max, h.steps_balanced[i]);
            steps_balanced_sum += h.steps_balanced[i];
            if (h.reset_count[i] < kFarmResetMin || h.reset_count[i] > kFarmResetMax) ++n_out_of_range;
        }

        std::printf("[info] farm: aggregate throughput %.2f million env-steps/second (%d envs x %d steps in %.3f ms)\n",
                   aggregate_env_steps_per_sec / 1.0e6, sc.N, sc.T_farm, static_cast<double>(gpu_ms));
        std::printf("[info] farm: reset_count min/avg/max = %d/%.2f/%d (provable min %d, documented max %d)\n",
                   rmin, static_cast<double>(rsum) / sc.N, rmax, kFarmResetMin, kFarmResetMax);
        std::printf("[info] farm: steps_balanced/T_farm min/avg/max = %.3f/%.3f/%.3f\n",
                   static_cast<double>(steps_balanced_min) / sc.T_farm,
                   (static_cast<double>(steps_balanced_sum) / sc.N) / sc.T_farm,
                   static_cast<double>(steps_balanced_max) / sc.T_farm);

        const bool farm_pass = (n_nonfinite == 0) && (n_out_of_range == 0);
        std::printf("FARM: %s (all %d environments finite; reset_count in [%d,%d] for all environments)\n",
                   farm_pass ? "PASS" : "FAIL", sc.N, kFarmResetMin, kFarmResetMax);
        overall_pass = overall_pass && farm_pass;

        // Stash the first 1000 environments' metrics + params for the CSV
        // artifact written after this block.
        const int n_report = std::min(sc.N, 1000);
        farm_steps_balanced.assign(h.steps_balanced.begin(), h.steps_balanced.begin() + n_report);
        farm_reset_count.assign(h.reset_count.begin(), h.reset_count.begin() + n_report);
        farm_mass_cart.assign(h.mass_cart.begin(), h.mass_cart.begin() + n_report);
        farm_mass_pole.assign(h.mass_pole.begin(), h.mass_pole.begin() + n_report);
        farm_pole_half_len.assign(h.pole_half_len.begin(), h.pole_half_len.begin() + n_report);
    }

    // ======================= STAGE 3: ENERGY CONSERVATION =====================
    // CPU-only: one undriven, unbounded trajectory (reference_cpu.cpp).
    std::vector<float> energy_trace(kEnergySteps + 1);
    {
        float final_state[4];
        energy_conservation_cpu(kEnergySteps, kEnergyTheta0,
                                kMassCartNominal, kMassPoleNominal, kPoleHalfLenNominal,
                                energy_trace.data(), final_state);

        const float e0 = energy_trace[0];
        float emin = e0, emax = e0;
        for (float e : energy_trace) { emin = std::min(emin, e); emax = std::max(emax, e); }
        const float max_abs_drift = std::max(std::fabs(emax - e0), std::fabs(e0 - emin));
        const float max_rel_drift = max_abs_drift / std::fabs(e0);

        // kEnergyRelDriftBound: calibrated from the MEASURED drift (see
        // THEORY.md §numerical considerations for the derivation and the
        // exact measured number this margin covers) — this is RK4's own
        // local-truncation-error signature made visible: an undriven,
        // frictionless system's energy is an exact invariant of the true
        // ODE, so ANY drift observed here is entirely integrator error,
        // not modeling error (there is no model error to attribute it to).
        const float kEnergyRelDriftBound = 1.0e-3f;
        const bool energy_pass = (max_rel_drift <= kEnergyRelDriftBound) && std::isfinite(max_rel_drift);

        std::printf("[info] energy: E0=%.6f J, Emin=%.6f J, Emax=%.6f J, final theta=%.4f rad\n",
                   static_cast<double>(e0), static_cast<double>(emin), static_cast<double>(emax),
                   static_cast<double>(final_state[2]));
        std::printf("[info] energy: max relative drift %.3e over %d undriven RK4 steps (dt=%.2f s)\n",
                   static_cast<double>(max_rel_drift), kEnergySteps, static_cast<double>(kDt));
        std::printf("ENERGY: %s (undriven cart-pole energy drift <= documented bound %.1e; "
                   "drift IS the RK4 truncation error made visible)\n",
                   energy_pass ? "PASS" : "FAIL", static_cast<double>(kEnergyRelDriftBound));
        overall_pass = overall_pass && energy_pass;
    }

    // ---- artifacts ------------------------------------------------------
    const std::string out_dir = project_root_from(argv[0]) + "/demo/out";
    bool dir_ok = ensure_dir(out_dir);

    bool env_csv_ok = dir_ok;
    if (dir_ok) {
        std::ofstream f(out_dir + "/env_metrics.csv");
        env_csv_ok = f.is_open();
        if (env_csv_ok) {
            f << "env_id,mass_cart_kg,mass_pole_kg,pole_half_len_m,steps_balanced,reset_count,balanced_fraction\n";
            for (size_t i = 0; i < farm_steps_balanced.size(); ++i) {
                f << i << ',' << farm_mass_cart[i] << ',' << farm_mass_pole[i] << ',' << farm_pole_half_len[i]
                  << ',' << farm_steps_balanced[i] << ',' << farm_reset_count[i]
                  << ',' << (static_cast<double>(farm_steps_balanced[i]) / static_cast<double>(sc.T_farm)) << '\n';
            }
        }
    }
    if (env_csv_ok)
        std::printf("ARTIFACT: wrote demo/out/env_metrics.csv (%zu rows)\n", farm_steps_balanced.size());
    else
        std::printf("ARTIFACT: FAILED to write demo/out/env_metrics.csv\n");

    bool energy_csv_ok = dir_ok;
    if (dir_ok) {
        std::ofstream f(out_dir + "/energy_drift.csv");
        energy_csv_ok = f.is_open();
        if (energy_csv_ok) {
            f << "step,t_s,energy_j,drift_rel\n";
            const float e0 = energy_trace[0];
            for (int t = 0; t <= kEnergySteps; ++t) {
                f << t << ',' << (static_cast<float>(t) * kDt) << ',' << energy_trace[static_cast<size_t>(t)]
                  << ',' << ((energy_trace[static_cast<size_t>(t)] - e0) / e0) << '\n';
            }
        }
    }
    if (energy_csv_ok)
        std::printf("ARTIFACT: wrote demo/out/energy_drift.csv (%d rows)\n", kEnergySteps + 1);
    else
        std::printf("ARTIFACT: FAILED to write demo/out/energy_drift.csv\n");

    overall_pass = overall_pass && env_csv_ok && energy_csv_ok;

    std::printf("[time] aggregate farm throughput: %.2f million env-steps/second\n",
               aggregate_env_steps_per_sec / 1.0e6);

    if (overall_pass)
        std::printf("RESULT: PASS (VERIFY + FARM + ENERGY all passed)\n");
    else
        std::printf("RESULT: FAIL (see VERIFY/FARM/ENERGY lines above)\n");
    return overall_pass ? 0 : 1;
}
