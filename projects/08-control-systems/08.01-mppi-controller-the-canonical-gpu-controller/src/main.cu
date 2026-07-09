// ===========================================================================
// main.cu — entry point for project 08.01
//           MPPI controller — the canonical GPU controller
//           (teaching core: force-limited cart-pole swing-up)
//
// What this program does, start to finish
// ---------------------------------------
//   1. Print the banner + GPU info; load the committed scenario (start
//      state + number of control steps) from data/sample/.
//   2. VERIFY STAGE (the §5 GPU-vs-CPU gate): on the very first control
//      iteration's inputs, compute ALL K rollout costs on the GPU kernel
//      AND the CPU oracle and require agreement within rel tol 1e-3.
//   3. CLOSED LOOP: run the MPPI controller for the scenario's 400 steps
//      at 50 Hz — swing the pole up from hanging and balance it — logging
//      every step to demo/out/trajectory.csv (the visual artifact; plot it).
//   4. SUCCESS CHECK: the pole must be upright (|theta| < 0.2 rad) for
//      every one of the final 100 steps. Exit 0 only if verify + success
//      both hold.
//
// The MPPI update implemented here (derivation in THEORY.md §the-math):
//      S_k        = cost of rollout k                    (GPU kernel)
//      w_k        = exp(-(S_k - S_min)/lambda)           (host, double accum)
//      u_nom[t]  += sum_k w_k * eps_k[t] / sum_k w_k     (host)
//      apply u_nom[0] to the plant; shift u_nom left; append 0.
// The host side is deliberately kept on the host: it is O(K·T) trivial
// arithmetic, and seeing the whole algorithm in ~40 lines of plain C++
// below the kernel call is worth more didactically than a fused reduction
// kernel (that optimization is README Exercise 3).
//
// Determinism: noise is host-generated Gaussian (xorshift32 + Box–Muller),
// seeded per control step — the run is bit-reproducible on this machine.
// (Host libm ulp differences may flip low bits across PLATFORMS; the
// stable output lines therefore carry no trajectory numbers, only
// PASS/FAIL against thresholds with wide margins. THEORY.md §numerics.)
//
// Output contract: stable lines "[demo]", "PROBLEM:", "SCENARIO:",
// "VERIFY:", "ARTIFACT:", "RESULT:" — "[info]"/"[time]" unchecked. Change a
// stable line ⇒ update demo/expected_output.txt in the same commit.
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
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
#include <direct.h>               // _mkdir (std::filesystem avoided in .cu — see 07.09)
#else
#include <sys/stat.h>
#endif

// ---------------------------------------------------------------------------
// Deterministic Gaussian noise: xorshift32 (the repo's portable generator)
// + Box–Muller. MPPI's derivation assumes Gaussian exploration noise; the
// weights' importance-sampling interpretation depends on it (THEORY.md).
// ---------------------------------------------------------------------------
static inline uint32_t xorshift32(uint32_t& state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

static inline float uniform01(uint32_t& state)      // (0,1] — never 0, safe for log()
{
    return (xorshift32(state) >> 8) * (1.0f / 16777216.0f) + (0.5f / 16777216.0f);
}

// One N(0, sigma^2) draw. Box–Muller in double for the transcendental step,
// cast at the end — the cheap way to keep the tails well-behaved in FP32.
static inline float gaussian(uint32_t& state, float sigma)
{
    const double u1 = static_cast<double>(uniform01(state));
    const double u2 = static_cast<double>(uniform01(state));
    const double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    return sigma * static_cast<float>(z);
}

// Fill the TRANSPOSED noise array eps[t*K + k] for one control step. The
// per-step seed mixes the base seed with the step index (odd multiplier →
// full-period stream separation) so every tick explores fresh noise —
// re-using noise across ticks is a classic MPPI bug that freezes exploration.
static void fill_noise(std::vector<float>& eps, int K, int step, uint32_t base_seed)
{
    uint32_t s = base_seed + 1000003u * static_cast<uint32_t>(step + 1);
    if (s == 0) s = 1u;
    for (float& e : eps) e = gaussian(s, kSigma);
}

// ---------------------------------------------------------------------------
// Scenario loading — the committed "task definition": where the plant
// starts and how long the controller runs. Tiny, but it is this project's
// data/sample content and follows the same strict-loader discipline.
// Rows: "X0,p,pdot,theta,thetadot" and "STEPS,n".
// ---------------------------------------------------------------------------
struct Scenario {
    float x0[kNX] = { 0.0f, 0.0f, 0.0f, 0.0f };
    int steps = 0;
    bool loaded = false;
};

static Scenario load_scenario(const std::string& path)
{
    Scenario sc;
    std::ifstream in(path);
    if (!in.is_open()) return sc;

    bool have_x0 = false, have_steps = false;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string label, cell;
        std::getline(ss, label, ',');
        if (label == "X0") {
            for (int i = 0; i < kNX; ++i) {
                if (!std::getline(ss, cell, ',')) { std::fprintf(stderr, "scenario: short X0 row\n"); return Scenario{}; }
                sc.x0[i] = std::strtof(cell.c_str(), nullptr);
            }
            have_x0 = true;
        } else if (label == "STEPS") {
            if (!std::getline(ss, cell, ',')) { std::fprintf(stderr, "scenario: short STEPS row\n"); return Scenario{}; }
            sc.steps = std::atoi(cell.c_str());
            have_steps = true;
        } else {
            std::fprintf(stderr, "scenario: unknown row label '%s'\n", label.c_str());
            return Scenario{};
        }
    }
    if (!have_x0 || !have_steps || sc.steps < 1) {
        std::fprintf(stderr, "scenario: missing X0 or STEPS\n");
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
    candidates.push_back(project_root_from(argv0) + "/data/sample/cartpole_scenario.csv");
    candidates.push_back("data/sample/cartpole_scenario.csv");
    candidates.push_back("../data/sample/cartpole_scenario.csv");
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
// main — verify stage, then the closed loop described in the file header.
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    int K = kDefaultK;             // rollouts per tick (CLI-overridable for experiments)
    std::string data_path;
    for (int i = 1; i < argc; ++i) {
        if      (!std::strcmp(argv[i], "--rollouts") && i + 1 < argc) K = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--data")     && i + 1 < argc) data_path = argv[++i];
        else {
            std::fprintf(stderr,
                "usage: %s [--rollouts K] [--data cartpole_scenario.csv]\n"
                "note: non-default K changes the PROBLEM line; the demo diff will flag it.\n",
                argv[0]);
            return 2;
        }
    }

    std::printf("[demo] MPPI controller: cart-pole swing-up (project 08.01)\n");
    print_device_info();
    std::printf("PROBLEM: sampling MPC, K=%d rollouts x T=%d steps @ dt=%.2f s (%.1f s horizon), force limit %.0f N, FP32\n",
                K, kHorizon, static_cast<double>(kDt),
                static_cast<double>(kHorizon * kDt), static_cast<double>(kUmax));

    // ---- scenario -----------------------------------------------------------
    const std::string scenario_path = find_scenario(data_path, argv[0]);
    if (scenario_path.empty()) {
        std::printf("SCENARIO: NOT FOUND — data/sample/cartpole_scenario.csv missing (run scripts/make_synthetic.py?)\n");
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
    std::printf("SCENARIO: start hanging (theta=pi), goal upright at p=0; %d control steps @ %.0f Hz [synthetic]\n",
                sc.steps, 1.0 / static_cast<double>(kDt));

    // ---- persistent device buffers ------------------------------------------
    // Allocated ONCE, outside the control loop — cudaMalloc costs hundreds of
    // microseconds, and a 50 Hz loop that reallocates every tick spends its
    // budget on the allocator instead of the rollouts. (The per-tick 16-byte
    // x0 upload inside the launcher is the deliberate, negligible exception.)
    const size_t eps_count = static_cast<size_t>(kHorizon) * K;
    float *d_u_nom = nullptr, *d_eps = nullptr, *d_cost = nullptr;
    CUDA_CHECK(cudaMalloc(&d_u_nom, kHorizon * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_eps, eps_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cost, static_cast<size_t>(K) * sizeof(float)));

    std::vector<float> u_nom(kHorizon, 0.0f);       // the nominal plan (starts at rest)
    std::vector<float> eps(eps_count);              // this tick's noise (transposed layout)
    std::vector<float> cost(static_cast<size_t>(K));

    // ======================= VERIFY STAGE ====================================
    // Iteration 0's exact inputs through both paths (the §5 gate). Tolerance
    // justification: 50 chained FP32 RK4 steps + sinf/cosf implementation
    // differences give ~1e-6..1e-5 relative cost divergence; 1e-3 is ~100×
    // headroom while indexing/clamp/layout bugs blow past it instantly.
    {
        fill_noise(eps, K, /*step=*/0, /*base_seed=*/42u);
        CUDA_CHECK(cudaMemcpy(d_eps, eps.data(), eps_count * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_u_nom, u_nom.data(), kHorizon * sizeof(float), cudaMemcpyHostToDevice));

        GpuTimer gt;
        gt.begin();
        launch_mppi_rollouts(K, sc.x0, d_u_nom, d_eps, d_cost);
        const float gpu_ms = gt.end_ms();
        CUDA_CHECK(cudaMemcpy(cost.data(), d_cost, static_cast<size_t>(K) * sizeof(float),
                              cudaMemcpyDeviceToHost));

        std::vector<float> cost_cpu(static_cast<size_t>(K));
        CpuTimer ct;
        ct.begin();
        mppi_rollouts_cpu(K, sc.x0, u_nom.data(), eps.data(), cost_cpu.data());
        const double cpu_ms = ct.end_ms();

        bool verify_pass = true;
        float worst = 0.0f;
        for (int k = 0; k < K; ++k) {
            const float scale = std::fabs(cost_cpu[static_cast<size_t>(k)]) > 1.0f
                              ? std::fabs(cost_cpu[static_cast<size_t>(k)]) : 1.0f;
            const float d = std::fabs(cost[static_cast<size_t>(k)] - cost_cpu[static_cast<size_t>(k)]) / scale;
            if (d > worst) worst = d;
            if (d > 1e-3f) verify_pass = false;
        }
        std::printf("[info] verify: worst relative cost deviation %.3e over %d rollouts\n",
                    static_cast<double>(worst), K);
        std::printf("[time] rollout set (K=%d, T=%d): CPU %.1f ms | GPU kernel %.3f ms | speed-up %.0fx (teaching artifact; kernel only)\n",
                    K, kHorizon, cpu_ms, static_cast<double>(gpu_ms),
                    cpu_ms / (static_cast<double>(gpu_ms) > 0.0 ? static_cast<double>(gpu_ms) : 1.0));
        std::printf("VERIFY: %s (GPU rollout costs match CPU reference within rel tol 1e-3)\n",
                    verify_pass ? "PASS" : "FAIL");
        if (!verify_pass) {
            std::printf("RESULT: FAIL (GPU/CPU rollout disagreement — fix before trusting the controller)\n");
            return 1;
        }
    }

    // ======================= CLOSED LOOP =====================================
    float x[kNX];                                    // the PLANT state ("reality")
    for (int i = 0; i < kNX; ++i) x[i] = sc.x0[i];

    std::vector<float> traj;                         // logged rows for the artifact
    traj.reserve(static_cast<size_t>(sc.steps) * 6);

    int balanced_streak = 0;                         // consecutive steps with |theta| < 0.2 rad
    int first_balanced_step = -1;                    // when the streak that lasted began ([info])
    double loop_gpu_ms = 0.0;                        // accumulated kernel time across the run

    for (int step = 0; step < sc.steps; ++step) {
        // (1) fresh exploration noise for this tick (seed varies per step).
        fill_noise(eps, K, step, 42u);
        CUDA_CHECK(cudaMemcpy(d_eps, eps.data(), eps_count * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_u_nom, u_nom.data(), kHorizon * sizeof(float), cudaMemcpyHostToDevice));

        // (2) K simulated futures, scored on the GPU.
        GpuTimer gt;
        gt.begin();
        launch_mppi_rollouts(K, x, d_u_nom, d_eps, d_cost);
        loop_gpu_ms += static_cast<double>(gt.end_ms());
        CUDA_CHECK(cudaMemcpy(cost.data(), d_cost, static_cast<size_t>(K) * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // (3) softmin weights. Subtracting S_min before exp() is the
        // standard overflow guard (exp of ~0 instead of exp of -10^4);
        // double accumulators keep the tiny-weight tail from vanishing.
        float s_min = cost[0];
        for (int k = 1; k < K; ++k) if (cost[static_cast<size_t>(k)] < s_min) s_min = cost[static_cast<size_t>(k)];
        double w_sum = 0.0;
        static std::vector<double> w;                // reused across ticks (no per-tick alloc)
        w.assign(static_cast<size_t>(K), 0.0);
        for (int k = 0; k < K; ++k) {
            w[static_cast<size_t>(k)] = std::exp(-static_cast<double>(cost[static_cast<size_t>(k)] - s_min) / kLambda);
            w_sum += w[static_cast<size_t>(k)];
        }

        // (4) blend the noise into the plan:  u_nom[t] += Σ_k w_k ε_k[t] / Σw.
        // Note the transposed eps layout pays off here too — the inner loop
        // over k walks consecutive memory.
        for (int t = 0; t < kHorizon; ++t) {
            double acc = 0.0;
            const float* eps_t = &eps[static_cast<size_t>(t) * K];
            for (int k = 0; k < K; ++k) acc += w[static_cast<size_t>(k)] * static_cast<double>(eps_t[k]);
            float u = u_nom[static_cast<size_t>(t)] + static_cast<float>(acc / w_sum);
            u_nom[static_cast<size_t>(t)] = u < -kUmax ? -kUmax : (u > kUmax ? kUmax : u);
        }

        // (5) act: apply the plan's first control to the plant for one tick.
        const float u_apply = u_nom[0];
        cartpole_step_cpu(x, u_apply, kDt);

        // (6) recede the horizon: shift the plan, hope zero at the far end.
        for (int t = 0; t + 1 < kHorizon; ++t) u_nom[static_cast<size_t>(t)] = u_nom[static_cast<size_t>(t) + 1];
        u_nom[kHorizon - 1] = 0.0f;

        // (7) bookkeeping: log + success tracking (plant theta is wrapped).
        traj.push_back(static_cast<float>(step) * kDt);
        traj.push_back(x[0]); traj.push_back(x[1]);
        traj.push_back(x[2]); traj.push_back(x[3]);
        traj.push_back(u_apply);
        if (std::fabs(x[2]) < 0.2f) {
            if (balanced_streak == 0) first_balanced_step = step;
            balanced_streak++;
        } else {
            balanced_streak = 0;
        }
    }

    CUDA_CHECK(cudaFree(d_u_nom));
    CUDA_CHECK(cudaFree(d_eps));
    CUDA_CHECK(cudaFree(d_cost));

    std::printf("[info] final state: p=%.3f m, pdot=%.3f m/s, theta=%.3f rad, thdot=%.3f rad/s\n",
                static_cast<double>(x[0]), static_cast<double>(x[1]),
                static_cast<double>(x[2]), static_cast<double>(x[3]));
    std::printf("[info] balanced streak at end: %d steps (threshold |theta| < 0.2 rad; streak began at step %d)\n",
                balanced_streak, first_balanced_step);
    std::printf("[time] closed loop: %.2f ms average GPU rollout kernel per control step over %d steps\n",
                loop_gpu_ms / sc.steps, sc.steps);

    // ---- artifact: the trajectory, plottable with anything -------------------
    const std::string out_dir = project_root_from(argv[0]) + "/demo/out";
    bool artifact_ok = ensure_dir(out_dir);
    if (artifact_ok) {
        std::ofstream f(out_dir + "/trajectory.csv");
        artifact_ok = f.is_open();
        if (artifact_ok) {
            f << "t_s,p_m,pdot_ms,theta_rad,thdot_rads,u_N\n";   // units in the header row (§12)
            for (int s = 0; s < sc.steps; ++s) {
                const float* r = &traj[static_cast<size_t>(s) * 6];
                f << r[0] << ',' << r[1] << ',' << r[2] << ',' << r[3] << ',' << r[4] << ',' << r[5] << '\n';
            }
        }
    }
    if (artifact_ok)
        std::printf("ARTIFACT: wrote demo/out/trajectory.csv (%d steps)\n", sc.steps);
    else
        std::printf("ARTIFACT: FAILED to write demo/out/trajectory.csv\n");

    // ---- success check (the stable verdict) ----------------------------------
    // "Swung up and stayed up": upright within 0.2 rad for every one of the
    // final 100 steps (2 s). Thresholds chosen with wide margin so platform
    // ulp differences in the noise stream cannot flip the verdict
    // (see the determinism note in the file header).
    const bool success = artifact_ok && (balanced_streak >= 100);
    if (success)
        std::printf("RESULT: PASS (swing-up achieved; pole balanced upright for the final 100 steps)\n");
    else
        std::printf("RESULT: FAIL (controller did not hold the pole upright for the final 100 steps — see [info] lines)\n");
    return success ? 0 : 1;
}
