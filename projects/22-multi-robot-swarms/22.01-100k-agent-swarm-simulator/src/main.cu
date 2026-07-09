// ===========================================================================
// main.cu — entry point for project 22.01
//           100k-agent swarm simulator: flocking, pheromone grids, stigmergy
//
// What this program does, start to finish
// ---------------------------------------
//   1. Print the banner + GPU info; load the committed scenario (agent
//      count, step count, spawn seed) from data/sample/.
//   2. VERIFY STAGE (the §5 GPU-vs-CPU gate): run a small deterministic
//      configuration (kVerifyN = 4096 agents) for kVerifySteps = 100 steps
//      in LOCKSTEP against the brute-force CPU oracle — both paths step
//      from the same state, outputs are compared within kTol* every step,
//      then the GPU output becomes the shared next state. Lockstep matters:
//      flocking is chaotic, so a free-running comparison would amplify
//      benign ulp differences into meters (THEORY.md §verification).
//   3. HEADLINE RUN: N = 100,000 agents for 300 steps (15 s of swarm time)
//      — bin, flock, deposit, diffuse, repeat — with per-phase timings.
//   4. METRICS + ARTIFACTS: check every agent stayed inside the arena,
//      compute the flock cohesion metric (mean local velocity alignment),
//      and write demo/out/density.pgm (agent density heatmap),
//      demo/out/pheromone.pgm (the stigmergy field), and
//      demo/out/positions.csv (first 1000 agents). Exit 0 only if verify +
//      bounded + cohesion all hold.
//
// The per-step pipeline (kernels.cuh contract; kernels in kernels.cu):
//      memset counts -> bin_count -> D2H counts -> HOST exclusive scan
//      -> H2D starts -> starts->cursor (D2D) -> bin_scatter
//      -> flock_step (reads cur, writes nxt) -> pheromone_step -> swap
// The host-side exclusive scan is a deliberate teaching choice: 65,536 ints
// is trivial CPU work, the round trip is honest and visible, and a GPU scan
// (Blelloch/CUB) is README Exercise 3.
//
// Determinism: the spawn is host-generated (xorshift32, the repo's portable
// generator, seed from the scenario file) — bit-reproducible everywhere.
// The SIMULATION is not bit-reproducible run to run: the scatter kernel's
// atomic cursor orders each cell's bin by thread scheduling, so neighbor
// sums differ in their last bits and chaos amplifies them (kernels.cuh
// determinism contract). The demo's verdict is engineered to survive that:
// the stable lines carry no trajectory numbers, and the success thresholds
// are statistics of 100k agents with wide margins.
//
// Output contract: stable lines "[demo]", "PROBLEM:", "SCENARIO:",
// "VERIFY:", "ARTIFACT:", "RESULT:" — "[info]"/"[time]" unchecked. Change a
// stable line => update demo/expected_output.txt in the same commit.
//
// Read this first, then kernels.cuh -> reference_cpu.cpp -> kernels.cu.
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
// Deterministic spawn — xorshift32 (the repo's portable generator; NEVER
// std::uniform_real_distribution, which is not bit-portable across standard
// libraries). Agents spawn uniformly inside the wall margin with random
// headings at 1 m/s — a maximally disordered start, so the cohesion metric
// measures ordering the DYNAMICS created, not ordering we baked in.
// ---------------------------------------------------------------------------
static inline uint32_t xorshift32(uint32_t& state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

static inline float uniform01(uint32_t& state)      // (0,1] — 24-bit mantissa-clean
{
    return (xorshift32(state) >> 8) * (1.0f / 16777216.0f) + (0.5f / 16777216.0f);
}

// Fill the four SoA state arrays for n agents. Headings use double trig for
// the transcendental step, cast at the end (same pattern as 08.01's
// Box-Muller): host libm ulp differences can flip trajectory low bits
// across PLATFORMS — absorbed by the statistics-based verdict (file header).
static void spawn_agents(int n, uint32_t seed,
                         std::vector<float>& px, std::vector<float>& py,
                         std::vector<float>& vx, std::vector<float>& vy)
{
    px.resize(n); py.resize(n); vx.resize(n); vy.resize(n);
    uint32_t s = seed ? seed : 1u;                  // xorshift32 must not start at 0
    const float lo = kWallMargin;                   // spawn inside the wall ramp so
    const float hi = kArena - kWallMargin;          // step 0 forces are pure flocking
    for (int i = 0; i < n; ++i) {
        px[i] = lo + (hi - lo) * uniform01(s);
        py[i] = lo + (hi - lo) * uniform01(s);
        const double ang = 6.283185307179586 * static_cast<double>(uniform01(s));
        vx[i] = static_cast<float>(std::cos(ang)); // |v| = 1 m/s: inside [kVMin,kVMax]
        vy[i] = static_cast<float>(std::sin(ang));
    }
}

// ---------------------------------------------------------------------------
// Scenario loading — the committed "task definition": how many agents, how
// many steps, which spawn seed. A swarm simulator's data IS its scenario
// (08.01 set the precedent); everything else is generated from it. Strict
// loader: unknown labels, short rows, or missing fields abort the demo.
// Rows: "N,<agents>", "STEPS,<steps>", "SEED,<uint>".
// ---------------------------------------------------------------------------
struct Scenario {
    int n = 0;
    int steps = 0;
    uint32_t seed = 0;
    bool loaded = false;
};

static Scenario load_scenario(const std::string& path)
{
    Scenario sc;
    std::ifstream in(path);
    if (!in.is_open()) return sc;

    bool have_n = false, have_steps = false, have_seed = false;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string label, cell;
        std::getline(ss, label, ',');
        if (!std::getline(ss, cell, ',')) {
            std::fprintf(stderr, "scenario: short '%s' row\n", label.c_str());
            return Scenario{};
        }
        if      (label == "N")     { sc.n = std::atoi(cell.c_str());     have_n = true; }
        else if (label == "STEPS") { sc.steps = std::atoi(cell.c_str()); have_steps = true; }
        else if (label == "SEED")  { sc.seed = static_cast<uint32_t>(std::strtoul(cell.c_str(), nullptr, 10)); have_seed = true; }
        else {
            std::fprintf(stderr, "scenario: unknown row label '%s'\n", label.c_str());
            return Scenario{};
        }
    }
    if (!have_n || !have_steps || !have_seed || sc.n < 1 || sc.steps < 1) {
        std::fprintf(stderr, "scenario: missing/invalid N, STEPS, or SEED\n");
        return Scenario{};
    }
    sc.loaded = true;
    return sc;
}

// Resolve paths relative to the executable (which lives three levels below
// the project root at build/x64/<Config>/), so the demo works from any CWD.
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
    candidates.push_back(project_root_from(argv0) + "/data/sample/swarm_scenario.csv");
    candidates.push_back("data/sample/swarm_scenario.csv");
    candidates.push_back("../data/sample/swarm_scenario.csv");
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

// PGM (P5) writer — the smallest real image format there is; viewable in
// any image tool, zero libraries (07.09's artifact pattern).
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

// ---------------------------------------------------------------------------
// Device-state lifecycle. Allocated ONCE per configuration and reused every
// step — cudaMalloc costs hundreds of microseconds; a 20 Hz loop that
// reallocates spends its budget on the allocator, not the swarm (08.01's
// lesson, repo-wide).
// ---------------------------------------------------------------------------
static SwarmGpu alloc_swarm(int n)
{
    SwarmGpu g = {};
    g.n = n;
    const size_t nb = static_cast<size_t>(n) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&g.px_cur, nb));  CUDA_CHECK(cudaMalloc(&g.py_cur, nb));
    CUDA_CHECK(cudaMalloc(&g.vx_cur, nb));  CUDA_CHECK(cudaMalloc(&g.vy_cur, nb));
    CUDA_CHECK(cudaMalloc(&g.px_nxt, nb));  CUDA_CHECK(cudaMalloc(&g.py_nxt, nb));
    CUDA_CHECK(cudaMalloc(&g.vx_nxt, nb));  CUDA_CHECK(cudaMalloc(&g.vy_nxt, nb));
    CUDA_CHECK(cudaMalloc(&g.pher_cur, kNumCells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.pher_nxt, kNumCells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.counts, kNumCells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g.starts, (kNumCells + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g.cursor, kNumCells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g.bin_agents, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g.align_score, nb));
    // Pheromone starts at exactly zero — the field is built by the agents.
    CUDA_CHECK(cudaMemset(g.pher_cur, 0, kNumCells * sizeof(float)));
    return g;
}

static void free_swarm(SwarmGpu& g)
{
    CUDA_CHECK(cudaFree(g.px_cur));  CUDA_CHECK(cudaFree(g.py_cur));
    CUDA_CHECK(cudaFree(g.vx_cur));  CUDA_CHECK(cudaFree(g.vy_cur));
    CUDA_CHECK(cudaFree(g.px_nxt));  CUDA_CHECK(cudaFree(g.py_nxt));
    CUDA_CHECK(cudaFree(g.vx_nxt));  CUDA_CHECK(cudaFree(g.vy_nxt));
    CUDA_CHECK(cudaFree(g.pher_cur)); CUDA_CHECK(cudaFree(g.pher_nxt));
    CUDA_CHECK(cudaFree(g.counts));   CUDA_CHECK(cudaFree(g.starts));
    CUDA_CHECK(cudaFree(g.cursor));   CUDA_CHECK(cudaFree(g.bin_agents));
    CUDA_CHECK(cudaFree(g.align_score));
    g = SwarmGpu{};
}

static void upload_state(SwarmGpu& g,
                         const std::vector<float>& px, const std::vector<float>& py,
                         const std::vector<float>& vx, const std::vector<float>& vy)
{
    const size_t nb = static_cast<size_t>(g.n) * sizeof(float);
    CUDA_CHECK(cudaMemcpy(g.px_cur, px.data(), nb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.py_cur, py.data(), nb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.vx_cur, vx.data(), nb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.vy_cur, vy.data(), nb, cudaMemcpyHostToDevice));
}

// Swap the ping-pong roles after a step: what was written becomes what is
// read. Pointer swaps only — no data moves.
static void swap_buffers(SwarmGpu& g)
{
    std::swap(g.px_cur, g.px_nxt);  std::swap(g.py_cur, g.py_nxt);
    std::swap(g.vx_cur, g.vx_nxt);  std::swap(g.vy_cur, g.vy_nxt);
    std::swap(g.pher_cur, g.pher_nxt);
}

// Per-step timing accumulators ([time] lines; teaching artifacts, never
// benchmark claims — CLAUDE.md §12).
struct StepTimes {
    double bin_ms = 0.0;    // wall time: count kernel + D2H + host scan + H2D + scatter
    double flock_ms = 0.0;  // GPU event time: the flock kernel
    double pher_ms = 0.0;   // GPU event time: the pheromone stencil
};

// ---------------------------------------------------------------------------
// gpu_step — one full simulation step on the device (the pipeline from the
// file header). h_counts/h_starts are caller-owned scratch so the hot loop
// does no host allocation. After this returns, *_nxt and pher_nxt hold the
// post-step state (caller swaps).
// ---------------------------------------------------------------------------
static void gpu_step(SwarmGpu& g, std::vector<int>& h_counts,
                     std::vector<int>& h_starts, StepTimes& t)
{
    // --- (1) bin: histogram -> host exclusive scan -> scatter --------------
    CpuTimer bt;                       // wall clock: this phase is dominated by
    bt.begin();                        // the two PCIe hops, which events on the
                                       // GPU timeline would under-report
    CUDA_CHECK(cudaMemset(g.counts, 0, kNumCells * sizeof(int)));
    launch_bin_count(g);
    CUDA_CHECK(cudaMemcpy(h_counts.data(), g.counts, kNumCells * sizeof(int),
                          cudaMemcpyDeviceToHost));

    // Exclusive scan on the host: starts[c] = number of agents in cells
    // before c; starts[kNumCells] = n (the total, so bin c is always
    // [starts[c], starts[c+1])). 65k additions — trivial, visible, honest;
    // the GPU scan that removes the round trip is README Exercise 3.
    h_starts[0] = 0;
    for (int c = 0; c < kNumCells; ++c) h_starts[c + 1] = h_starts[c] + h_counts[c];

    CUDA_CHECK(cudaMemcpy(g.starts, h_starts.data(), (kNumCells + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    // The scatter consumes a WORKING copy of starts (it bumps cursors);
    // starts itself must stay intact for the flock kernel's bin lookups.
    CUDA_CHECK(cudaMemcpy(g.cursor, g.starts, kNumCells * sizeof(int),
                          cudaMemcpyDeviceToDevice));
    launch_bin_scatter(g);
    CUDA_CHECK(cudaDeviceSynchronize());   // close the phase for honest wall timing
    t.bin_ms += bt.end_ms();

    // --- (2) flock: one thread per agent, reads cur / writes nxt -----------
    GpuTimer ft;
    ft.begin();
    launch_flock_step(g);
    t.flock_ms += static_cast<double>(ft.end_ms());

    // --- (3) pheromone: deposit (from counts) + diffuse + decay ------------
    GpuTimer pt;
    pt.begin();
    launch_pheromone_step(g);
    t.pher_ms += static_cast<double>(pt.end_ms());
}

// ---------------------------------------------------------------------------
// verify_lockstep — the §5 GPU-vs-CPU gate (stage 2 of the file header).
// Returns true on PASS and reports the worst deviations seen.
// ---------------------------------------------------------------------------
static bool verify_lockstep(const char* /*argv0*/, double& cpu_ms_total, double& gpu_ms_total)
{
    // Deterministic small config: seed fixed at 42 (independent of the
    // scenario, so the gate never changes when the scenario does).
    std::vector<float> px, py, vx, vy;
    spawn_agents(kVerifyN, 42u, px, py, vx, vy);
    std::vector<float> pher(kNumCells, 0.0f);

    SwarmGpu g = alloc_swarm(kVerifyN);
    upload_state(g, px, py, vx, vy);

    // Host-side buffers: CPU-step outputs, GPU-step downloads, and scratch.
    const size_t nb = static_cast<size_t>(kVerifyN) * sizeof(float);
    std::vector<float> cpx(kVerifyN), cpy(kVerifyN), cvx(kVerifyN), cvy(kVerifyN);
    std::vector<float> cpher(kNumCells);
    std::vector<float> gpx(kVerifyN), gpy(kVerifyN), gvx(kVerifyN), gvy(kVerifyN);
    std::vector<float> gpher(kNumCells);
    std::vector<int> h_counts(kNumCells), h_starts(kNumCells + 1);
    StepTimes t;

    float worst_pos = 0.0f, worst_vel = 0.0f, worst_pher = 0.0f;
    bool pass = true;
    cpu_ms_total = 0.0;

    for (int step = 0; step < kVerifySteps; ++step) {
        // GPU path: one step from the shared state, then download.
        gpu_step(g, h_counts, h_starts, t);
        CUDA_CHECK(cudaMemcpy(gpx.data(), g.px_nxt, nb, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gpy.data(), g.py_nxt, nb, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gvx.data(), g.vx_nxt, nb, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gvy.data(), g.vy_nxt, nb, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gpher.data(), g.pher_nxt, kNumCells * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // CPU oracle: the SAME step from the SAME state, brute force.
        CpuTimer ct;
        ct.begin();
        swarm_step_cpu(kVerifyN, px.data(), py.data(), vx.data(), vy.data(),
                       pher.data(), cpx.data(), cpy.data(), cvx.data(), cvy.data(),
                       cpher.data());
        cpu_ms_total += ct.end_ms();

        // Compare: max absolute deviations this step (units: m, m/s, conc).
        for (int i = 0; i < kVerifyN; ++i) {
            const float dp = std::fmax(std::fabs(gpx[i] - cpx[i]), std::fabs(gpy[i] - cpy[i]));
            const float dv = std::fmax(std::fabs(gvx[i] - cvx[i]), std::fabs(gvy[i] - cvy[i]));
            if (dp > worst_pos) worst_pos = dp;
            if (dv > worst_vel) worst_vel = dv;
        }
        for (int c = 0; c < kNumCells; ++c) {
            const float dq = std::fabs(gpher[c] - cpher[c]);
            if (dq > worst_pher) worst_pher = dq;
        }
        if (worst_pos > kTolPos || worst_vel > kTolVel || worst_pher > kTolPher) {
            pass = false;
            std::printf("[info] verify: FAILED at step %d (pos %.3e m, vel %.3e m/s, pher %.3e)\n",
                        step, static_cast<double>(worst_pos),
                        static_cast<double>(worst_vel), static_cast<double>(worst_pher));
            break;
        }

        // ADOPT the GPU output as the shared state for the next step — the
        // lockstep re-anchoring that keeps chaos out of the comparison.
        px = gpx;  py = gpy;  vx = gvx;  vy = gvy;  pher = gpher;
        swap_buffers(g);   // device side continues from the same (GPU) state
    }

    gpu_ms_total = t.bin_ms + t.flock_ms + t.pher_ms;
    std::printf("[info] verify: worst per-step deviation over %d lockstep steps: "
                "pos %.3e m, vel %.3e m/s, pheromone %.3e (tol %.0e each)\n",
                kVerifySteps, static_cast<double>(worst_pos),
                static_cast<double>(worst_vel), static_cast<double>(worst_pher),
                static_cast<double>(kTolPos));
    free_swarm(g);
    return pass;
}

// ---------------------------------------------------------------------------
// main — banner, scenario, verify stage, headline run, metrics, artifacts.
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // CLI: overrides exist for experimentation; the CHECKED demo runs with
    // no arguments (non-default sizes change the stable PROBLEM/SCENARIO
    // lines and the diff will flag it — by design).
    std::string data_path;
    int cli_n = 0, cli_steps = 0;
    for (int i = 1; i < argc; ++i) {
        if      (!std::strcmp(argv[i], "--agents") && i + 1 < argc) cli_n = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--steps")  && i + 1 < argc) cli_steps = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--data")   && i + 1 < argc) data_path = argv[++i];
        else {
            std::fprintf(stderr,
                "usage: %s [--agents N] [--steps T] [--data swarm_scenario.csv]\n"
                "note: non-default N/T change the stable output lines; the demo diff will flag it.\n",
                argv[0]);
            return 2;
        }
    }

    std::printf("[demo] 100k-agent swarm simulator: flocking + pheromone stigmergy (project 22.01)\n");
    print_device_info();

    // ---- scenario -----------------------------------------------------------
    const std::string scenario_path = find_scenario(data_path, argv[0]);
    if (scenario_path.empty()) {
        std::printf("SCENARIO: NOT FOUND — data/sample/swarm_scenario.csv missing (run scripts/make_synthetic.py?)\n");
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
    if (cli_n > 0) sc.n = cli_n;             // experimentation overrides
    if (cli_steps > 0) sc.steps = cli_steps;

    std::printf("PROBLEM: N=%d boids + %dx%d pheromone grid, uniform-grid neighbor search, %.0f m arena, FP32\n",
                sc.n, kGridDim, kGridDim, static_cast<double>(kArena));
    std::printf("SCENARIO: uniform spawn, random headings (seed %u); %d steps @ dt=%.2f s (%.0f s of swarm time) [synthetic]\n",
                sc.seed, sc.steps, static_cast<double>(kDt),
                static_cast<double>(sc.steps) * static_cast<double>(kDt));

    // ======================= VERIFY STAGE ====================================
    {
        double cpu_ms = 0.0, gpu_ms = 0.0;
        const bool ok = verify_lockstep(argv[0], cpu_ms, gpu_ms);
        std::printf("[time] verify: CPU brute-force oracle %.0f ms total vs GPU pipeline %.0f ms total "
                    "(N=%d, %d steps; O(N^2) vs grid — teaching artifact, not a benchmark)\n",
                    cpu_ms, gpu_ms, kVerifyN, kVerifySteps);
        std::printf("VERIFY: %s (GPU lockstep-matches CPU brute-force oracle: N=%d, %d steps, pos/vel/pheromone within tol 1e-3)\n",
                    ok ? "PASS" : "FAIL", kVerifyN, kVerifySteps);
        if (!ok) {
            std::printf("RESULT: FAIL (GPU/CPU disagreement — fix before trusting the swarm)\n");
            return 1;
        }
    }

    // ======================= HEADLINE RUN ====================================
    std::vector<float> px, py, vx, vy;
    spawn_agents(sc.n, sc.seed, px, py, vx, vy);

    SwarmGpu g = alloc_swarm(sc.n);
    upload_state(g, px, py, vx, vy);

    std::vector<int> h_counts(kNumCells), h_starts(kNumCells + 1);
    StepTimes t;
    for (int step = 0; step < sc.steps; ++step) {
        gpu_step(g, h_counts, h_starts, t);
        swap_buffers(g);
    }
    std::printf("[time] headline run (N=%d, %d steps): bin %.2f ms/step (incl. host scan round trip) | "
                "flock kernel %.2f ms/step | pheromone kernel %.3f ms/step\n",
                sc.n, sc.steps, t.bin_ms / sc.steps, t.flock_ms / sc.steps, t.pher_ms / sc.steps);

    // ---- final state + metrics ----------------------------------------------
    const size_t nb = static_cast<size_t>(sc.n) * sizeof(float);
    std::vector<float> score(sc.n), pher(kNumCells);
    CUDA_CHECK(cudaMemcpy(px.data(), g.px_cur, nb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(py.data(), g.py_cur, nb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vx.data(), g.vx_cur, nb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vy.data(), g.vy_cur, nb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(score.data(), g.align_score, nb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pher.data(), g.pher_cur, kNumCells * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Rebin the FINAL positions so the density artifact shows where agents
    // ENDED (counts currently reflect the last step's pre-update positions —
    // one step stale; cheap to fix, so fix it).
    CUDA_CHECK(cudaMemset(g.counts, 0, kNumCells * sizeof(int)));
    launch_bin_count(g);
    CUDA_CHECK(cudaMemcpy(h_counts.data(), g.counts, kNumCells * sizeof(int),
                          cudaMemcpyDeviceToHost));
    free_swarm(g);

    // BOUNDED: every agent finite and inside [0, kArena] — the integrator's
    // clamp guarantees this unless the dynamics produced a NaN (NaN fails
    // every comparison, so it lands in the else-branch and fails the check).
    int escaped = 0;
    for (int i = 0; i < sc.n; ++i) {
        const bool ok = std::isfinite(px[i]) && std::isfinite(py[i]) &&
                        px[i] >= 0.0f && px[i] <= kArena &&
                        py[i] >= 0.0f && py[i] <= kArena;
        if (!ok) ++escaped;
    }

    // COHESION: mean local velocity alignment — the average, over agents
    // that HAD neighbors on the final step, of cos(angle between the agent's
    // velocity and its local mean). Random headings score ~0; a flocked
    // swarm scores near 1. Double accumulator: 100k small floats.
    double align_sum = 0.0;
    long long align_cnt = 0;
    for (int i = 0; i < sc.n; ++i) {
        if (score[i] <= 1.0f) { align_sum += score[i]; ++align_cnt; }  // sentinel is 2.0
    }
    const double mean_align = (align_cnt > 0) ? align_sum / static_cast<double>(align_cnt) : 0.0;

    // Global polarization |sum of headings|/N — the classic Vicsek order
    // parameter, printed for context: it stays well below 1 here because
    // 15 s is far too short for ONE consensus heading to spread 256 m at a
    // 1 m interaction radius; local order (above) is the honest 15-s metric
    // (THEORY.md §algorithm).
    double sx = 0.0, sy = 0.0;
    for (int i = 0; i < sc.n; ++i) {
        const double sp = std::sqrt(static_cast<double>(vx[i]) * vx[i] +
                                    static_cast<double>(vy[i]) * vy[i]);
        if (sp > 1e-12) { sx += vx[i] / sp; sy += vy[i] / sp; }
    }
    const double polarization = std::sqrt(sx * sx + sy * sy) / sc.n;

    std::printf("[info] final metrics: mean local alignment %.3f (over %lld agents with neighbors, "
                "%.1f%% of swarm), global polarization %.3f, escaped/NaN agents %d\n",
                mean_align, align_cnt, 100.0 * align_cnt / sc.n, polarization, escaped);

    // ---- artifacts ------------------------------------------------------------
    const std::string out_dir = project_root_from(argv[0]) + "/demo/out";
    bool artifact_ok = ensure_dir(out_dir);
    if (artifact_ok) {
        // density.pgm — agents per cell, sqrt-scaled so sparse cells stay
        // visible next to dense flock cores. PGM row 0 is the TOP of the
        // image; arena y points UP — so image row r shows grid row
        // kGridDim-1-r (a y-flip, or every viewer shows the arena mirrored).
        int cmax = 1;
        for (int c = 0; c < kNumCells; ++c) if (h_counts[c] > cmax) cmax = h_counts[c];
        float pmax = 1e-6f;
        for (int c = 0; c < kNumCells; ++c) if (pher[c] > pmax) pmax = pher[c];
        std::vector<uint8_t> dens(kNumCells), pimg(kNumCells);
        for (int cy = 0; cy < kGridDim; ++cy) {
            for (int cx = 0; cx < kGridDim; ++cx) {
                const int src = cy * kGridDim + cx;
                const int dst = (kGridDim - 1 - cy) * kGridDim + cx;   // the y-flip
                dens[dst] = static_cast<uint8_t>(255.0 * std::sqrt(
                    static_cast<double>(h_counts[src]) / cmax) + 0.5);
                pimg[dst] = static_cast<uint8_t>(255.0 *
                    static_cast<double>(pher[src]) / pmax + 0.5);
            }
        }
        artifact_ok = write_pgm(out_dir + "/density.pgm", kGridDim, kGridDim, dens)
                   && write_pgm(out_dir + "/pheromone.pgm", kGridDim, kGridDim, pimg);

        // positions.csv — the first 1000 agents' final state, plottable
        // with anything (units in the header row, CLAUDE.md §12).
        if (artifact_ok) {
            std::ofstream f(out_dir + "/positions.csv");
            artifact_ok = f.is_open();
            if (artifact_ok) {
                f << "id,x_m,y_m,vx_ms,vy_ms\n";
                const int rows = sc.n < 1000 ? sc.n : 1000;
                for (int i = 0; i < rows; ++i)
                    f << i << ',' << px[i] << ',' << py[i] << ','
                      << vx[i] << ',' << vy[i] << '\n';
            }
        }
    }
    if (artifact_ok)
        std::printf("ARTIFACT: wrote demo/out/density.pgm, demo/out/pheromone.pgm, demo/out/positions.csv (first 1000 agents)\n");
    else
        std::printf("ARTIFACT: FAILED to write demo/out files\n");

    // ---- success check (the stable verdict) ----------------------------------
    // Thresholds carry wide margins (measured mean alignment ~0.9 vs the
    // 0.5 gate; random start scores ~0) so run-to-run atomic-order ulps and
    // cross-platform libm ulps cannot flip the verdict (file header).
    const bool bounded = (escaped == 0);
    const bool cohesive = (mean_align >= 0.5) && (align_cnt > 0);
    const bool success = artifact_ok && bounded && cohesive;
    if (success)
        std::printf("RESULT: PASS (all agents bounded in the arena; flocking emerged: mean local alignment >= 0.5)\n");
    else
        std::printf("RESULT: FAIL (bounded=%s cohesive=%s artifacts=%s — see [info] lines)\n",
                    bounded ? "yes" : "NO", cohesive ? "yes" : "NO", artifact_ok ? "yes" : "NO");
    return success ? 0 : 1;
}
