// ===========================================================================
// main.cu — entry point for project 32.02
//           CUDA Graphs for jitter-free fixed-rate perception-control loops
//
// What this program does, start to finish
// -----------------------------------------
// This is a LATENCY/JITTER MEASUREMENT STUDY, not a controller demo. One
// "tick" (kernels.cuh/kernels.cu: 8 kernels + 2 device-to-device "publish"
// copies, a small MPPI-shaped perception-control pipeline borrowed from
// project 08.01) is executed N times at a fixed 250 Hz pace, three
// different ways:
//
//   MODE A "stream"        — twelve individual CUDA API calls per tick,
//                             straight onto a stream (the naive way).
//   MODE B "graph"         — the identical twelve calls captured ONCE via
//                             cudaStreamBeginCapture/cudaStreamEndCapture,
//                             replayed with a single cudaGraphLaunch/tick.
//   MODE C "graph+update"  — the same DAG, built explicitly via
//                             cudaGraphAddKernelNode/cudaGraphAddMemcpyNode
//                             so that ONE node's input pointer can be
//                             double-buffered and repointed every tick via
//                             cudaGraphExecKernelNodeSetParams — the real
//                             technique for a captured node whose ARGUMENT
//                             (not just the memory behind it) must change.
//
// For each mode this program measures, per tick: host SUBMIT time (the API
// calls only, no waiting), end-to-end LATENCY (submit to GPU-completion-
// confirmed, via cudaEvent + QueryPerformanceCounter), and the achieved
// PACING PERIOD — then reports mean/p50/p95/p99/max. See THEORY.md
// "Measurement methodology" for why three different clocks are in play and
// why tail percentiles are reported as [info], never gated.
//
// Correctness has two independent layers (mirroring the repo's usual §5
// gate, doubled because this project ALSO claims something about identical
// GPU orchestration):
//   1. VERIFY  — tick 0's GPU path (mode A) vs. reference_cpu.cpp's plain
//      C++ twin of the WHOLE tick, within a documented relative tolerance.
//   2. CROSSMODE — mode B's and mode C's full 2000-tick output trajectories
//      compared BIT-FOR-BIT against mode A's. All three modes run the exact
//      same deterministic kernels on the exact same per-tick inputs with
//      the exact same launch geometry — the only thing that differs is HOW
//      the work reaches the GPU, so THEORY.md §numerics argues (and this
//      check proves) their outputs must be identical down to the last bit.
//
// Output contract: stable lines "[demo]", "PROBLEM:", "SCENARIO:",
// "VERIFY:", "CROSSMODE:", "GATE ...:", "ARTIFACT:", "RESULT:" — "[info]"/
// "[time]" lines are NOT diffed (machine-dependent numbers). Change a
// stable line -> update demo/expected_output.txt in the same change.
//
// Read this first, then kernels.cuh (the tick's contract) -> kernels.cu
// (the 8 kernels) -> reference_cpu.cpp (the CPU twin).
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <mmsystem.h>          // timeBeginPeriod/timeEndPeriod — see PacingClock below
#include <direct.h>            // _mkdir — used by ensure_dir() near the artifact writer
#pragma comment(lib, "winmm.lib")   // belt-and-suspenders alongside the .vcxproj's
                                    // AdditionalDependencies entry (CLAUDE.md §5 rule 2
                                    // asks the .vcxproj to document *why*; this pragma
                                    // just keeps the .cu file buildable if ever compiled
                                    // outside that project file, e.g. via a stray cl.exe
                                    // invocation while debugging)
#else
#include <sys/stat.h>
#endif

// ===========================================================================
// PacingClock — the hybrid sleep+spin fixed-rate scheduler.
//
// THE HONEST PROBLEM (derived, not asserted — see the [info] calibration
// line main() prints before the study runs): Windows' default system timer
// tick is ~15.6 ms (64 Hz) unless a process requests finer resolution.
// Sleep(1) on an UNADJUSTED system can therefore actually sleep for up to
// ~15 ms — more than three whole ticks of this project's 4 ms budget.
// timeBeginPeriod(1) requests 1 ms scheduler resolution (the best Win32
// offers without a real-time kernel), which brings Sleep(1)'s typical
// overshoot down to roughly 1-2 ms — STILL too coarse to hit a 4 ms
// deadline reliably by itself (a single Sleep(1) call can eat a third of
// the whole budget). The fix used everywhere in this project: sleep for
// most of the remaining time (cheap, yields the CPU, but imprecise), then
// BUSY-SPIN polling QueryPerformanceCounter for the final ~1.2 ms (precise,
// ~100 ns resolution on any modern PC, but burns a core). Neither
// technique alone is both efficient AND precise; the hybrid is standard
// practice for soft-real-time pacing on stock Windows (THEORY.md
// "Windows WDDM vs Linux/TCC vs Jetson" discusses what changes with a
// PREEMPT_RT kernel or a dedicated real-time core).
// ===========================================================================
#ifdef _WIN32
struct PacingClock {
    LARGE_INTEGER freq{};
    bool timer_period_set = false;

    void init()
    {
        QueryPerformanceFrequency(&freq);
        // Best-effort: some sandboxed/virtualized environments deny this: we
        // still function (just coarser), so we do not CUDA_CHECK-style abort.
        timer_period_set = (timeBeginPeriod(1) == TIMERR_NOERROR);
    }
    ~PacingClock() { if (timer_period_set) timeEndPeriod(1); }

    double now_us() const
    {
        LARGE_INTEGER c;
        QueryPerformanceCounter(&c);
        return static_cast<double>(c.QuadPart) * 1.0e6 / static_cast<double>(freq.QuadPart);
    }

    // Block (host thread) until the absolute deadline (in the same us
    // timebase as now_us()) has passed. See the file-header derivation for
    // why this is sleep-then-spin, not either alone.
    void sleep_until(double deadline_us) const
    {
        constexpr double kSpinMarginUs = 1200.0;   // measured-safe margin (see [info] calibration)
        for (;;) {
            const double remain = deadline_us - now_us();
            if (remain <= 0.0) return;                       // already late — do not oversleep further
            if (remain <= kSpinMarginUs) break;               // hand off to the spin loop below
            const DWORD ms = static_cast<DWORD>((remain - kSpinMarginUs) / 1000.0);
            if (ms >= 1) Sleep(ms); else break;
        }
        while (now_us() < deadline_us) { /* busy-spin: the precise final stretch */ }
    }
};
#else
// Portable fallback for the CMake/Linux best-effort path (CLAUDE.md §5).
// Linux has no equivalent of timeBeginPeriod; nanosleep()'s achievable
// granularity is typically better than stock Win32 Sleep() even without
// special privileges, and a REAL deployment would use clock_nanosleep with
// TIMER_ABSTIME on a SCHED_FIFO thread under PREEMPT_RT (THEORY.md
// "Windows WDDM vs Linux/TCC vs Jetson" names this explicitly) rather than
// this demo's best-effort std::this_thread::sleep_for.
struct PacingClock {
    void init() {}
    double now_us() const
    {
        return std::chrono::duration<double, std::micro>(
                   std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    void sleep_until(double deadline_us) const
    {
        constexpr double kSpinMarginUs = 500.0;
        for (;;) {
            const double remain = deadline_us - now_us();
            if (remain <= 0.0) return;
            if (remain <= kSpinMarginUs) break;
            std::this_thread::sleep_for(std::chrono::microseconds(
                static_cast<long long>(remain - kSpinMarginUs)));
        }
        while (now_us() < deadline_us) { /* busy-spin */ }
    }
};
#endif

// Measure Sleep(1)'s actual achieved duration a few times before the study
// starts — the honest "derive/measure" this project's brief asks for,
// rather than asserting the ~15.6 ms/1 ms numbers from documentation alone.
static void calibrate_sleep_granularity(const PacingClock& clk)
{
    const int trials = 20;
    double total_us = 0.0, worst_us = 0.0;
    for (int i = 0; i < trials; ++i) {
        const double t0 = clk.now_us();
#ifdef _WIN32
        Sleep(1);
#else
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
#endif
        const double dt = clk.now_us() - t0;
        total_us += dt;
        if (dt > worst_us) worst_us = dt;
    }
    std::printf("[info] measured Sleep(1) duration over %d trials: mean %.0f us, worst %.0f us "
                "(vs. the 4000 us tick budget) -- why pacing uses hybrid sleep+spin, not Sleep() alone\n",
                trials, total_us / trials, worst_us);
}

// ---------------------------------------------------------------------------
// Deterministic host RNG: xorshift32 + Box-Muller, the same portable
// generator 08.01 uses, so a reader who has studied that project recognizes
// it instantly. Two INDEPENDENT streams (eps, sensor) so the exploration
// noise and the synthetic sensor noise never share bits.
// ---------------------------------------------------------------------------
static inline uint32_t xorshift32(uint32_t& s)
{
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return s;
}
static inline float uniform01(uint32_t& s)
{
    return (xorshift32(s) >> 8) * (1.0f / 16777216.0f) + (0.5f / 16777216.0f);
}
static inline float gaussian(uint32_t& s, float sigma)
{
    const double u1 = static_cast<double>(uniform01(s));
    const double u2 = static_cast<double>(uniform01(s));
    const double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    return sigma * static_cast<float>(z);
}

// tick_inputs — everything ONE tick's GPU work reads that changes tick to
// tick, generated PURELY as a function of the global tick index `g` (never
// of wall-clock time or of which MODE is running) — the precondition for
// all three modes seeing bit-identical inputs at the same tick, which is in
// turn the precondition for the CROSSMODE bit-identical-output check.
static void tick_inputs(long long g, uint32_t seed_eps, uint32_t seed_sensor,
                        float* sensor_raw, float* eps)
{
    uint32_t s_eps = seed_eps + 1000003u * static_cast<uint32_t>(g + 1);   // same odd-multiplier
    if (s_eps == 0) s_eps = 1u;                                            // stream-separation trick
    for (int i = 0; i < kHorizon * kRollouts; ++i) eps[i] = gaussian(s_eps, kSigma);

    // Synthetic "range scan": a spatial triangle wave (so the smoothing
    // kernel has real neighbor structure to denoise) whose phase drifts
    // slowly with the tick index (so the pipeline sees a changing scene,
    // not a frozen one), plus per-tick Gaussian sensor noise. Honestly a
    // MADE-UP relationship (README §Limitations) — its only job is shape.
    uint32_t s_sensor = seed_sensor + 1000003u * static_cast<uint32_t>(g + 1);
    if (s_sensor == 0) s_sensor = 1u;
    for (int i = 0; i < kSensorN; ++i) {
        const int phase = (i + static_cast<int>((g * 3) % 64)) % 64;
        const float tri = static_cast<float>(phase < 32 ? phase : 63 - phase);   // 0..31, period 64
        const float base = 2000.0f + 60.0f * tri;                                // ~2000..3860 counts
        float v = base + gaussian(s_sensor, 40.0f);
        v = v < 0.0f ? 0.0f : (v > 4095.0f ? 4095.0f : v);                       // valid raw-count range
        sensor_raw[i] = v;
    }
}

// ---------------------------------------------------------------------------
// Scenario loading — the committed tick-pipeline + pacing configuration
// (data/sample/tick_scenario.csv), the same strict-loader discipline as
// 08.01's cartpole_scenario.csv. Regenerating it with scripts/make_synthetic.py
// with different arguments changes the PROBLEM: line (documented, same as
// 08.01's --rollouts note).
// ---------------------------------------------------------------------------
struct Scenario {
    float    x0[kNX] = { 0.0f, 0.0f, 0.3f, 0.0f };
    uint32_t seed_eps = 1u, seed_sensor = 1u;
    int      measured_ticks = 0;
    int      warmup_ticks = 0;
    double   pacing_hz = 0.0;
    bool     loaded = false;
};

static Scenario load_scenario(const std::string& path)
{
    Scenario sc;
    std::ifstream in(path);
    if (!in.is_open()) return sc;

    bool have_x0 = false, have_ticks = false, have_hz = false;
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
        } else if (label == "SEED_EPS") {
            if (!std::getline(ss, cell, ',')) return Scenario{};
            sc.seed_eps = static_cast<uint32_t>(std::strtoul(cell.c_str(), nullptr, 10));
        } else if (label == "SEED_SENSOR") {
            if (!std::getline(ss, cell, ',')) return Scenario{};
            sc.seed_sensor = static_cast<uint32_t>(std::strtoul(cell.c_str(), nullptr, 10));
        } else if (label == "MEASURED_TICKS") {
            if (!std::getline(ss, cell, ',')) return Scenario{};
            sc.measured_ticks = std::atoi(cell.c_str());
            have_ticks = true;
        } else if (label == "WARMUP_TICKS") {
            if (!std::getline(ss, cell, ',')) return Scenario{};
            sc.warmup_ticks = std::atoi(cell.c_str());
        } else if (label == "PACING_HZ") {
            if (!std::getline(ss, cell, ',')) return Scenario{};
            sc.pacing_hz = std::strtod(cell.c_str(), nullptr);
            have_hz = true;
        } else {
            std::fprintf(stderr, "scenario: unknown row label '%s'\n", label.c_str());
            return Scenario{};
        }
    }
    if (!have_x0 || !have_ticks || !have_hz || sc.measured_ticks < 1 || sc.pacing_hz <= 0.0) {
        std::fprintf(stderr, "scenario: missing/invalid required rows\n");
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
    candidates.push_back(project_root_from(argv0) + "/data/sample/tick_scenario.csv");
    candidates.push_back("data/sample/tick_scenario.csv");
    candidates.push_back("../data/sample/tick_scenario.csv");
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

// ===========================================================================
// Device buffers — every persistent allocation the tick pipeline touches,
// allocated ONCE (CLAUDE.md §12: cudaMalloc is expensive; a fixed-rate loop
// must not pay it every tick) and reused by all three modes in turn
// (reset_tick_state() below re-initializes the STATE, never the addresses —
// addresses staying fixed across the whole run is exactly what lets modes B
// and C capture a graph once and replay it thousands of times).
// d_sensor_raw[2]: TWO allocations, not one — only mode C actually
// alternates between them (its cudaGraphExecKernelNodeSetParams
// double-buffering demo); modes A and B always use index 0. Allocating both
// unconditionally keeps this struct and its alloc/free functions mode-
// agnostic.
// ===========================================================================
struct DeviceBuffers {
    float* d_sensor_raw[2] = { nullptr, nullptr };
    float* d_sensor_scaled = nullptr;
    float* d_sensor_smoothed = nullptr;
    float* d_state_pred = nullptr;
    float* d_state_est = nullptr;
    float* d_eps = nullptr;
    float* d_u_nom = nullptr;
    float* d_cost = nullptr;
    float* d_s_min = nullptr;
    float* d_weights = nullptr;
    float* d_w_sum = nullptr;
    float* d_u_published = nullptr;
    float* d_state_telemetry = nullptr;
};

static void alloc_device_buffers(DeviceBuffers& db)
{
    CUDA_CHECK(cudaMalloc(&db.d_sensor_raw[0], kSensorN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db.d_sensor_raw[1], kSensorN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db.d_sensor_scaled, kSensorN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db.d_sensor_smoothed, kSensorN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db.d_state_pred, kNX * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db.d_state_est, kNX * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db.d_eps, static_cast<size_t>(kHorizon) * kRollouts * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db.d_u_nom, kHorizon * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db.d_cost, kRollouts * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db.d_s_min, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db.d_weights, kRollouts * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db.d_w_sum, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db.d_u_published, kHorizon * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db.d_state_telemetry, kNX * sizeof(float)));
}

static void free_device_buffers(DeviceBuffers& db)
{
    CUDA_CHECK(cudaFree(db.d_sensor_raw[0]));
    CUDA_CHECK(cudaFree(db.d_sensor_raw[1]));
    CUDA_CHECK(cudaFree(db.d_sensor_scaled));
    CUDA_CHECK(cudaFree(db.d_sensor_smoothed));
    CUDA_CHECK(cudaFree(db.d_state_pred));
    CUDA_CHECK(cudaFree(db.d_state_est));
    CUDA_CHECK(cudaFree(db.d_eps));
    CUDA_CHECK(cudaFree(db.d_u_nom));
    CUDA_CHECK(cudaFree(db.d_cost));
    CUDA_CHECK(cudaFree(db.d_s_min));
    CUDA_CHECK(cudaFree(db.d_weights));
    CUDA_CHECK(cudaFree(db.d_w_sum));
    CUDA_CHECK(cudaFree(db.d_u_published));
    CUDA_CHECK(cudaFree(db.d_state_telemetry));
}

// Reset the tick pipeline's RECURRENT state (u_nom, state estimate) to the
// scenario's starting point. Called before EACH mode's run so all three
// modes see an IDENTICAL starting condition — required for the CROSSMODE
// bit-identical check (kernels.cuh's inductive argument only holds if tick
// 0's inputs AND starting state are identical across modes).
static void reset_tick_state(DeviceBuffers& db, const float* x0, cudaStream_t stream)
{
    CUDA_CHECK(cudaMemsetAsync(db.d_u_nom, 0, kHorizon * sizeof(float), stream));
    CUDA_CHECK(cudaMemcpyAsync(db.d_state_est, x0, kNX * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// ===========================================================================
// submit_naive — the twelve individual CUDA API calls that make up ONE
// tick, issued straight onto `stream`. This function is MODE A's entire
// per-tick body, called directly every tick; it is ALSO exactly what gets
// captured for MODE B (see capture_graph_stream below) — the same code,
// wrapped in cudaStreamBeginCapture/cudaStreamEndCapture, becomes a graph.
// That reuse is the point: CUDA Graphs do not require different code, only
// the SAME code recorded once instead of replayed by hand every tick.
// ===========================================================================
static void submit_naive(DeviceBuffers& db, const float* h_sensor, const float* h_eps, cudaStream_t stream)
{
    CUDA_CHECK(cudaMemcpyAsync(db.d_sensor_raw[0], h_sensor, kSensorN * sizeof(float),
                               cudaMemcpyHostToDevice, stream));                                    // call 1
    CUDA_CHECK(cudaMemcpyAsync(db.d_eps, h_eps, static_cast<size_t>(kHorizon) * kRollouts * sizeof(float),
                               cudaMemcpyHostToDevice, stream));                                    // call 2
    launch_sensor_scale_bias(db.d_sensor_raw[0], db.d_sensor_scaled, stream);                       // call 3
    launch_sensor_smooth(db.d_sensor_scaled, db.d_sensor_smoothed, stream);                          // call 4
    launch_state_predict(db.d_state_est, db.d_state_pred, stream);                                   // call 5
    launch_state_correct(db.d_sensor_smoothed, db.d_state_pred, db.d_state_est, stream);              // call 6
    launch_mppi_rollout(db.d_state_est, db.d_u_nom, db.d_eps, db.d_cost, stream);                    // call 7
    launch_cost_min(db.d_cost, db.d_s_min, stream);                                                  // call 8
    launch_softmin_weight(db.d_cost, db.d_s_min, db.d_weights, db.d_w_sum, stream);                  // call 9
    launch_control_blend(db.d_u_nom, db.d_eps, db.d_weights, db.d_w_sum, stream);                    // call 10
    CUDA_CHECK(cudaMemcpyAsync(db.d_u_published, db.d_u_nom, kHorizon * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream));                                  // call 11
    CUDA_CHECK(cudaMemcpyAsync(db.d_state_telemetry, db.d_state_est, kNX * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream));                                  // call 12
}

// ===========================================================================
// MODE B — capture submit_naive() ONCE via stream capture, instantiate.
//
// cudaStreamBeginCapture puts `stream` into "recording" mode: every API
// call enqueued on it (memcpys, kernel launches) becomes a GRAPH NODE
// instead of doing real work, and stream-ISSUE-ORDER becomes graph
// DEPENDENCY order (a linear chain here, since submit_naive issues
// everything on the one stream with no forking). cudaStreamEndCapture
// finalizes the recorded cudaGraph_t; cudaGraphInstantiate performs the
// (relatively expensive, one-time) validation + upload that turns a graph
// DESCRIPTION into a graph EXECUTABLE (cudaGraphExec_t) the driver can
// replay cheaply. See THEORY.md "What CUDA Graphs actually are" for the
// full API-family explanation.
// ===========================================================================
static void capture_graph_stream(DeviceBuffers& db, const float* h_sensor_pinned, const float* h_eps_pinned,
                                 cudaStream_t stream, cudaGraph_t& graph, cudaGraphExec_t& graphExec)
{
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    submit_naive(db, h_sensor_pinned, h_eps_pinned, stream);   // recorded, not executed
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, 0));
}

// ===========================================================================
// MODE C — the explicit node-construction API + cudaGraphExecKernelNodeSetParams.
//
// Built with cudaGraphAddKernelNode/cudaGraphAddMemcpyNode1D instead of
// stream capture because SetParams needs a cudaKernelNodeParams struct
// whose kernelParams array points at storage THIS PROGRAM owns and can
// mutate later (see GraphCState below) — stream capture does not hand that
// struct back to the caller. The chain is deliberately kept LINEAR (every
// node depends on only the one before it, exactly mirroring mode B's
// stream-captured shape) even where two stages have no real data
// dependency (e.g. sensor_scale_bias and state_predict) — see THEORY.md
// "Measurement methodology" for why: letting mode C exploit real
// intra-tick parallelism that mode B's single-stream capture cannot would
// make its gpu_exec_ms incomparable to the other two modes' for reasons
// that have nothing to do with the update mechanism being studied. A
// fancier multi-stream capture recovering that parallelism for ALL modes
// is README Exercise territory, not this demo's job.
//
// ONLY the sensor_scale_bias node (stage 1) is repointed every tick, at a
// deliberately alternating (double-buffered) input pointer — see
// submit_setparams() below and README "The algorithm in brief" for why
// double buffering, specifically, is the realistic motivating use case.
// ===========================================================================
struct GraphCState {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaGraphNode_t sensorNode = nullptr;
    cudaKernelNodeParams sensorParams{};
    const float* sensorXPtr = nullptr;     // mutated every tick (kernelParams[0] points HERE)
    float*       sensorYPtr = nullptr;     // fixed target buffer (kernelParams[1] points here, unchanged)
    void*        sensorArgs[2] = { nullptr, nullptr };
};

static void build_graph_setparams(DeviceBuffers& db, const float* h_eps_pinned,
                                  cudaStream_t /*stream*/, GraphCState& gc)
{
    CUDA_CHECK(cudaGraphCreate(&gc.graph, 0));

    // Node 1: eps upload — a root node (no dependencies), refreshed the same
    // way mode B's captured memcpy is: overwrite the FIXED host source
    // address (h_eps_pinned) before each cudaGraphLaunch; the memcpy node's
    // own src/dst addresses never change, so it needs no SetParams at all.
    cudaGraphNode_t nEps;
    CUDA_CHECK(cudaGraphAddMemcpyNode1D(&nEps, gc.graph, nullptr, 0,
        db.d_eps, h_eps_pinned, static_cast<size_t>(kHorizon) * kRollouts * sizeof(float),
        cudaMemcpyHostToDevice));

    // Node 2: sensor_scale_bias — THE node this project updates every tick.
    // kernelParams[0] points at gc.sensorXPtr (a member variable this
    // struct owns for the rest of the program's life), NOT at a stack
    // temporary — cudaGraphExecKernelNodeSetParams re-reads through this
    // pointer at every call, so the storage must outlive the whole study.
    gc.sensorXPtr = db.d_sensor_raw[0];
    gc.sensorYPtr = db.d_sensor_scaled;
    gc.sensorArgs[0] = &gc.sensorXPtr;
    gc.sensorArgs[1] = &gc.sensorYPtr;
    gc.sensorParams = cudaKernelNodeParams{};
    gc.sensorParams.func = reinterpret_cast<void*>(sensor_scale_bias_kernel);
    gc.sensorParams.gridDim = dim3(1, 1, 1);
    gc.sensorParams.blockDim = dim3(kSensorN, 1, 1);
    gc.sensorParams.sharedMemBytes = 0;
    gc.sensorParams.kernelParams = gc.sensorArgs;
    gc.sensorParams.extra = nullptr;
    {
        cudaGraphNode_t deps[1] = { nEps };   // chained after nEps for shape-parity with mode B only
        CUDA_CHECK(cudaGraphAddKernelNode(&gc.sensorNode, gc.graph, deps, 1, &gc.sensorParams));
    }

    // Node 3: sensor_smooth. AddKernelNode COPIES the argument VALUES it is
    // given at call time (the same semantics as an ordinary <<<>>> launch),
    // so plain local variables are fine here — only node 2's params, which
    // are touched again LATER via SetParams, need member-lifetime storage.
    cudaGraphNode_t nSmooth;
    {
        const float* p0 = db.d_sensor_scaled; float* p1 = db.d_sensor_smoothed;
        void* args[2] = { &p0, &p1 };
        cudaKernelNodeParams p{};
        p.func = reinterpret_cast<void*>(sensor_smooth_kernel);
        p.gridDim = dim3(1, 1, 1); p.blockDim = dim3(kSensorN, 1, 1); p.kernelParams = args;
        cudaGraphNode_t deps[1] = { gc.sensorNode };
        CUDA_CHECK(cudaGraphAddKernelNode(&nSmooth, gc.graph, deps, 1, &p));
    }

    // Node 4: state_predict.
    cudaGraphNode_t nPredict;
    {
        const float* p0 = db.d_state_est; float* p1 = db.d_state_pred;
        void* args[2] = { &p0, &p1 };
        cudaKernelNodeParams p{};
        p.func = reinterpret_cast<void*>(state_predict_kernel);
        p.gridDim = dim3(1, 1, 1); p.blockDim = dim3(kNX, 1, 1); p.kernelParams = args;
        cudaGraphNode_t deps[1] = { nSmooth };   // forced serial — see file-header comment
        CUDA_CHECK(cudaGraphAddKernelNode(&nPredict, gc.graph, deps, 1, &p));
    }

    // Node 5: state_correct.
    cudaGraphNode_t nCorrect;
    {
        const float* p0 = db.d_sensor_smoothed; const float* p1 = db.d_state_pred; float* p2 = db.d_state_est;
        void* args[3] = { &p0, &p1, &p2 };
        cudaKernelNodeParams p{};
        p.func = reinterpret_cast<void*>(state_correct_kernel);
        p.gridDim = dim3(1, 1, 1); p.blockDim = dim3(kSensorN, 1, 1); p.kernelParams = args;
        cudaGraphNode_t deps[1] = { nPredict };
        CUDA_CHECK(cudaGraphAddKernelNode(&nCorrect, gc.graph, deps, 1, &p));
    }

    // Node 6: mppi_rollout (the fat kernel).
    cudaGraphNode_t nRollout;
    {
        const float* p0 = db.d_state_est; const float* p1 = db.d_u_nom;
        const float* p2 = db.d_eps; float* p3 = db.d_cost;
        void* args[4] = { &p0, &p1, &p2, &p3 };
        cudaKernelNodeParams p{};
        p.func = reinterpret_cast<void*>(mppi_rollout_kernel);
        p.gridDim = dim3(1, 1, 1); p.blockDim = dim3(kRollouts, 1, 1); p.kernelParams = args;
        cudaGraphNode_t deps[1] = { nCorrect };
        CUDA_CHECK(cudaGraphAddKernelNode(&nRollout, gc.graph, deps, 1, &p));
    }

    // Node 7: cost_min.
    cudaGraphNode_t nCostMin;
    {
        const float* p0 = db.d_cost; float* p1 = db.d_s_min;
        void* args[2] = { &p0, &p1 };
        cudaKernelNodeParams p{};
        p.func = reinterpret_cast<void*>(cost_min_kernel);
        p.gridDim = dim3(1, 1, 1); p.blockDim = dim3(kRollouts, 1, 1); p.kernelParams = args;
        cudaGraphNode_t deps[1] = { nRollout };
        CUDA_CHECK(cudaGraphAddKernelNode(&nCostMin, gc.graph, deps, 1, &p));
    }

    // Node 8: softmin_weight.
    cudaGraphNode_t nSoftmin;
    {
        const float* p0 = db.d_cost; const float* p1 = db.d_s_min;
        float* p2 = db.d_weights; float* p3 = db.d_w_sum;
        void* args[4] = { &p0, &p1, &p2, &p3 };
        cudaKernelNodeParams p{};
        p.func = reinterpret_cast<void*>(softmin_weight_kernel);
        p.gridDim = dim3(1, 1, 1); p.blockDim = dim3(kRollouts, 1, 1); p.kernelParams = args;
        cudaGraphNode_t deps[1] = { nCostMin };
        CUDA_CHECK(cudaGraphAddKernelNode(&nSoftmin, gc.graph, deps, 1, &p));
    }

    // Node 9: control_blend (gridDim = kHorizon blocks).
    cudaGraphNode_t nBlend;
    {
        float* p0 = db.d_u_nom; const float* p1 = db.d_eps;
        const float* p2 = db.d_weights; const float* p3 = db.d_w_sum;
        void* args[4] = { &p0, &p1, &p2, &p3 };
        cudaKernelNodeParams p{};
        p.func = reinterpret_cast<void*>(control_blend_kernel);
        p.gridDim = dim3(kHorizon, 1, 1); p.blockDim = dim3(kRollouts, 1, 1); p.kernelParams = args;
        cudaGraphNode_t deps[1] = { nSoftmin };
        CUDA_CHECK(cudaGraphAddKernelNode(&nBlend, gc.graph, deps, 1, &p));
    }

    // Node 10-11: the two "publish" copies.
    cudaGraphNode_t nPub1, nPub2;
    {
        cudaGraphNode_t deps[1] = { nBlend };
        CUDA_CHECK(cudaGraphAddMemcpyNode1D(&nPub1, gc.graph, deps, 1,
            db.d_u_published, db.d_u_nom, kHorizon * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    {
        cudaGraphNode_t deps[1] = { nPub1 };   // forced serial (real dependency is on nCorrect only)
        CUDA_CHECK(cudaGraphAddMemcpyNode1D(&nPub2, gc.graph, deps, 1,
            db.d_state_telemetry, db.d_state_est, kNX * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    // Instantiate: validate the finished DAG and upload it to the driver as
    // a replayable executable. Same one-time cost as mode B's instantiate.
    CUDA_CHECK(cudaGraphInstantiate(&gc.exec, gc.graph, 0));
}

// submit_setparams — mode C's per-tick body: refresh the alternating
// device buffer, repoint the captured sensor node at it, launch. THREE
// host API calls (vs. mode A's twelve, mode B's one) — the honest middle
// ground the README quotes.
static void submit_setparams(DeviceBuffers& db, GraphCState& gc, const float* h_sensor_pinned,
                             int buf_index, cudaStream_t stream)
{
    CUDA_CHECK(cudaMemcpyAsync(db.d_sensor_raw[buf_index], h_sensor_pinned, kSensorN * sizeof(float),
                               cudaMemcpyHostToDevice, stream));                 // call 1: fill the OTHER buffer
    gc.sensorXPtr = db.d_sensor_raw[buf_index];                                 // mutate the owned local...
    CUDA_CHECK(cudaGraphExecKernelNodeSetParams(gc.exec, gc.sensorNode, &gc.sensorParams));  // ...call 2: push it in
    CUDA_CHECK(cudaGraphLaunch(gc.exec, stream));                               // call 3: replay
}

// ===========================================================================
// Statistics helpers — nearest-rank percentiles over a copied, sorted
// vector (simple, exact for the sample sizes here, and easy to audit by
// eye — no interpolation subtleties to get wrong in a measurement script).
// ===========================================================================
static double percentile_of_sorted(const std::vector<double>& sorted_v, double p)
{
    if (sorted_v.empty()) return 0.0;
    size_t idx = static_cast<size_t>(p * static_cast<double>(sorted_v.size() - 1) + 0.5);
    if (idx >= sorted_v.size()) idx = sorted_v.size() - 1;
    return sorted_v[idx];
}

static JitterStats compute_stats(std::vector<double> v)   // by value: sorted in place, harmlessly
{
    JitterStats s{};
    if (v.empty()) return s;
    double sum = 0.0;
    for (double x : v) sum += x;
    s.mean = sum / static_cast<double>(v.size());
    std::sort(v.begin(), v.end());
    s.p50 = percentile_of_sorted(v, 0.50);
    s.p95 = percentile_of_sorted(v, 0.95);
    s.p99 = percentile_of_sorted(v, 0.99);
    s.max = v.back();
    return s;
}

static std::vector<double> field(const std::vector<TickMeasurement>& v, double TickMeasurement::* f)
{
    std::vector<double> out; out.reserve(v.size());
    for (const auto& m : v) out.push_back(m.*f);
    return out;
}

// ===========================================================================
// run_mode — reset state, warm up (unpaced), then run kMeasuredTicks paced
// ticks, logging a TickMeasurement and the tick's two published outputs
// (for the later CROSSMODE comparison) every measured tick.
// ===========================================================================
enum class Mode { kStreamNaive, kGraphStatic, kGraphSetParams };

struct ModeRunResult {
    std::vector<TickMeasurement> ticks;
    std::vector<float> u_nom_log;     // [measured_ticks * kHorizon]
    std::vector<float> state_log;     // [measured_ticks * kNX]
};

static ModeRunResult run_mode(Mode mode, DeviceBuffers& db,
                              float* h_sensor_pinned, float* h_eps_pinned,
                              cudaGraphExec_t graphExecB, GraphCState* gcC,
                              const Scenario& sc, cudaStream_t stream,
                              cudaEvent_t evStart, cudaEvent_t evStop,
                              const PacingClock& clk)
{
    reset_tick_state(db, sc.x0, stream);

    // Warmup: unpaced, back-to-back, so first-launch module-load/JIT costs
    // and clock ramp-up land here, not in the measured window (CLAUDE.md
    // §12 timing discipline). Uses the SAME deterministic tick_inputs()
    // sequence (global indices 0..warmup_ticks-1) so the state the measured
    // window STARTS from is identical across all three modes.
    for (int w = 0; w < sc.warmup_ticks; ++w) {
        tick_inputs(w, sc.seed_eps, sc.seed_sensor, h_sensor_pinned, h_eps_pinned);
        if (mode == Mode::kStreamNaive)        submit_naive(db, h_sensor_pinned, h_eps_pinned, stream);
        else if (mode == Mode::kGraphStatic)   CUDA_CHECK(cudaGraphLaunch(graphExecB, stream));
        else                                    submit_setparams(db, *gcC, h_sensor_pinned, w % 2, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    ModeRunResult r;
    r.ticks.reserve(static_cast<size_t>(sc.measured_ticks));
    r.u_nom_log.assign(static_cast<size_t>(sc.measured_ticks) * kHorizon, 0.0f);
    r.state_log.assign(static_cast<size_t>(sc.measured_ticks) * kNX, 0.0f);

    const double period_us = 1.0e6 / sc.pacing_hz;
    double prev_tick_start = clk.now_us() - period_us;   // so tick 0's logged period reads ~period_us
    double next_deadline = clk.now_us() + period_us;

    for (int i = 0; i < sc.measured_ticks; ++i) {
        const int g = sc.warmup_ticks + i;   // global tick index — continues the deterministic sequence
        tick_inputs(g, sc.seed_eps, sc.seed_sensor, h_sensor_pinned, h_eps_pinned);

        const double t0 = clk.now_us();
        CUDA_CHECK(cudaEventRecord(evStart, stream));
        if (mode == Mode::kStreamNaive)        submit_naive(db, h_sensor_pinned, h_eps_pinned, stream);
        else if (mode == Mode::kGraphStatic)   CUDA_CHECK(cudaGraphLaunch(graphExecB, stream));
        else                                    submit_setparams(db, *gcC, h_sensor_pinned, g % 2, stream);
        CUDA_CHECK(cudaEventRecord(evStop, stream));
        const double t1 = clk.now_us();                 // submission done, NOT waiting for completion
        CUDA_CHECK(cudaEventSynchronize(evStop));        // block until the GPU confirms this tick is done
        const double t2 = clk.now_us();

        float gpu_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, evStart, evStop));

        // Untimed readback (after latency is already recorded) for the
        // CROSSMODE correctness check and the artifact — identical cost in
        // all three modes, so it cannot bias the comparison.
        CUDA_CHECK(cudaMemcpy(&r.u_nom_log[static_cast<size_t>(i) * kHorizon], db.d_u_published,
                              kHorizon * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&r.state_log[static_cast<size_t>(i) * kNX], db.d_state_telemetry,
                              kNX * sizeof(float), cudaMemcpyDeviceToHost));

        TickMeasurement tm{};
        tm.tick = i;
        tm.submit_us = t1 - t0;
        tm.latency_us = t2 - t0;
        tm.gpu_exec_ms = static_cast<double>(gpu_ms);
        tm.period_us = t0 - prev_tick_start;
        prev_tick_start = t0;
        r.ticks.push_back(tm);

        next_deadline += period_us;       // FIXED schedule: never re-anchor on the current time, or
        clk.sleep_until(next_deadline);   // a late tick would silently shrink every future period
    }
    return r;
}

// ===========================================================================
// main
// ===========================================================================
int main(int argc, char** argv)
{
    std::string data_path;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--data") && i + 1 < argc) data_path = argv[++i];
        else {
            std::fprintf(stderr, "usage: %s [--data tick_scenario.csv]\n", argv[0]);
            return 2;
        }
    }

    std::printf("[demo] CUDA Graphs for jitter-free fixed-rate perception-control loops (project 32.02)\n");
    print_device_info();

    // ---- scenario -----------------------------------------------------------
    const std::string scenario_path = find_scenario(data_path, argv[0]);
    if (scenario_path.empty()) {
        std::printf("SCENARIO: NOT FOUND -- data/sample/tick_scenario.csv missing (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (scenario missing)\n");
        return 1;
    }
    std::printf("[info] scenario file: %s\n", scenario_path.c_str());
    Scenario sc = load_scenario(scenario_path);
    if (!sc.loaded) {
        std::printf("SCENARIO: MALFORMED -- see stderr\n");
        std::printf("RESULT: FAIL (scenario malformed)\n");
        return 1;
    }
    std::printf("PROBLEM: tick = 8 kernels + 2 D2D copies (sensorN=%d, K=%d rollouts, T=%d horizon), "
                "3 modes x %d ticks @ %.0f Hz (%.1f ms period), %d warmup ticks/mode [synthetic]\n",
                kSensorN, kRollouts, kHorizon, sc.measured_ticks, sc.pacing_hz, 1000.0 / sc.pacing_hz,
                sc.warmup_ticks);
    std::printf("SCENARIO: x0=[p=%.2f, pdot=%.2f, theta=%.2f, thdot=%.2f], seeds=(eps=%u, sensor=%u)\n",
                static_cast<double>(sc.x0[0]), static_cast<double>(sc.x0[1]),
                static_cast<double>(sc.x0[2]), static_cast<double>(sc.x0[3]),
                sc.seed_eps, sc.seed_sensor);

    // ---- setup ----------------------------------------------------------------
    PacingClock clk;
    clk.init();
    calibrate_sleep_granularity(clk);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    cudaEvent_t evStart, evStop;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));

    DeviceBuffers db;
    alloc_device_buffers(db);

    // PINNED host staging buffers for the two per-tick uploads: pinned
    // memory is what makes cudaMemcpyAsync genuinely asynchronous (a
    // pageable-memory async copy silently falls back to a synchronous
    // staged copy inside the driver) — load-bearing for both the submit-
    // time measurement's honesty and for graph replay correctness (a
    // captured memcpy node re-reads its source address at EVERY replay;
    // that source had better be a real DMA-able address, not a pageable
    // buffer the OS could have swapped).
    float* h_sensor_pinned = nullptr;
    float* h_eps_pinned = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_sensor_pinned, kSensorN * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_eps_pinned, static_cast<size_t>(kHorizon) * kRollouts * sizeof(float)));

    // ======================= VERIFY STAGE ====================================
    // Tick 0's exact inputs through mode A's GPU path AND reference_cpu.cpp's
    // plain-C++ twin of the WHOLE tick. Tolerance: rel 1e-3 (08.01's
    // precedent) -- 08.01's THEORY.md measured ~1e-7 relative divergence
    // over 50 chained FP32 RK4 steps from FMA/trig-implementation
    // differences alone; this project's rollouts are shorter (T=16), so the
    // same tolerance carries even more headroom here (THEORY.md quantifies
    // this project's own measured worst case).
    bool verify_pass = false;
    {
        reset_tick_state(db, sc.x0, stream);
        tick_inputs(0, sc.seed_eps, sc.seed_sensor, h_sensor_pinned, h_eps_pinned);
        submit_naive(db, h_sensor_pinned, h_eps_pinned, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<float> gpu_cost(kRollouts), gpu_u_nom(kHorizon), gpu_state(kNX);
        CUDA_CHECK(cudaMemcpy(gpu_cost.data(), db.d_cost, kRollouts * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gpu_u_nom.data(), db.d_u_nom, kHorizon * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gpu_state.data(), db.d_state_est, kNX * sizeof(float), cudaMemcpyDeviceToHost));

        std::vector<float> cpu_cost(kRollouts);
        std::vector<float> cpu_u_nom(kHorizon, 0.0f);   // matches reset_tick_state's zero-initialized u_nom
        std::vector<float> cpu_state(kNX);
        tick_pipeline_cpu(h_sensor_pinned, h_eps_pinned, sc.x0, cpu_u_nom.data(), cpu_state.data(), cpu_cost.data());

        float worst_cost_rel = 0.0f;
        for (int k = 0; k < kRollouts; ++k) {
            const float scale = std::fabs(cpu_cost[k]) > 1.0f ? std::fabs(cpu_cost[k]) : 1.0f;
            const float d = std::fabs(gpu_cost[k] - cpu_cost[k]) / scale;
            if (d > worst_cost_rel) worst_cost_rel = d;
        }
        float worst_u_abs = 0.0f;
        for (int t = 0; t < kHorizon; ++t)
            worst_u_abs = std::max(worst_u_abs, std::fabs(gpu_u_nom[t] - cpu_u_nom[t]));
        float worst_state_abs = 0.0f;
        for (int i = 0; i < kNX; ++i)
            worst_state_abs = std::max(worst_state_abs, std::fabs(gpu_state[i] - cpu_state[i]));

        verify_pass = (worst_cost_rel <= 1e-3f) && (worst_u_abs <= 1e-3f) && (worst_state_abs <= 1e-3f);
        std::printf("[info] verify: worst rollout-cost rel deviation %.3e, worst u_nom abs %.3e, "
                    "worst state abs %.3e\n",
                    static_cast<double>(worst_cost_rel), static_cast<double>(worst_u_abs),
                    static_cast<double>(worst_state_abs));
        std::printf("VERIFY: %s (GPU tick 0 matches the CPU tick twin within documented tolerance)\n",
                    verify_pass ? "PASS" : "FAIL");
    }
    if (!verify_pass) {
        std::printf("RESULT: FAIL (GPU/CPU tick disagreement -- fix before trusting any timing number below)\n");
        free_device_buffers(db);
        return 1;
    }

    // ======================= THE THREE-MODE STUDY ============================
    cudaGraph_t graphB = nullptr; cudaGraphExec_t graphExecB = nullptr;
    capture_graph_stream(db, h_sensor_pinned, h_eps_pinned, stream, graphB, graphExecB);

    GraphCState gcC;
    build_graph_setparams(db, h_eps_pinned, stream, gcC);

    std::printf("[info] graphs instantiated: mode B (stream-captured, %d nodes incl. sensor memcpy), "
                "mode C (explicit-construction, 11 nodes, sensor node repointed via SetParams every tick)\n",
                12);

    ModeRunResult rA = run_mode(Mode::kStreamNaive, db, h_sensor_pinned, h_eps_pinned,
                                nullptr, nullptr, sc, stream, evStart, evStop, clk);
    std::printf("[info] mode A (stream, naive) complete: %d measured ticks\n", sc.measured_ticks);

    ModeRunResult rB = run_mode(Mode::kGraphStatic, db, h_sensor_pinned, h_eps_pinned,
                                graphExecB, nullptr, sc, stream, evStart, evStop, clk);
    std::printf("[info] mode B (graph, static) complete: %d measured ticks\n", sc.measured_ticks);

    ModeRunResult rC = run_mode(Mode::kGraphSetParams, db, h_sensor_pinned, h_eps_pinned,
                                nullptr, &gcC, sc, stream, evStart, evStop, clk);
    std::printf("[info] mode C (graph, SetParams) complete: %d measured ticks\n", sc.measured_ticks);

    // ======================= CROSSMODE CORRECTNESS ============================
    // All three modes ran the IDENTICAL deterministic kernels on IDENTICAL
    // per-tick inputs with IDENTICAL launch geometry -- only the
    // orchestration differed. THEORY.md §numerics argues their outputs
    // MUST therefore be bit-identical; this is where that claim is checked,
    // not asserted.
    auto arrays_equal = [](const std::vector<float>& a, const std::vector<float>& b,
                           size_t& mismatch_idx) -> bool {
        if (a.size() != b.size()) { mismatch_idx = 0; return false; }
        for (size_t i = 0; i < a.size(); ++i)
            if (a[i] != b[i]) { mismatch_idx = i; return false; }
        return true;
    };
    size_t mm = 0;
    const bool crossmode_b = arrays_equal(rA.u_nom_log, rB.u_nom_log, mm)
                           && arrays_equal(rA.state_log, rB.state_log, mm);
    const bool crossmode_c = arrays_equal(rA.u_nom_log, rC.u_nom_log, mm)
                           && arrays_equal(rA.state_log, rC.state_log, mm);
    const bool crossmode_pass = crossmode_b && crossmode_c;
    std::printf("CROSSMODE: %s (mode B and mode C outputs bit-identical to mode A over %d ticks x "
                "(%d control + %d state) floats)\n",
                crossmode_pass ? "PASS" : "FAIL", sc.measured_ticks, kHorizon, kNX);
    if (!crossmode_pass)
        std::printf("[info] first mismatch at flattened index %zu (mode B ok=%d, mode C ok=%d)\n",
                    mm, crossmode_b, crossmode_c);

    // ======================= MEASUREMENT GATES ================================
    const JitterStats submitA = compute_stats(field(rA.ticks, &TickMeasurement::submit_us));
    const JitterStats submitB = compute_stats(field(rB.ticks, &TickMeasurement::submit_us));
    const JitterStats submitC = compute_stats(field(rC.ticks, &TickMeasurement::submit_us));
    const JitterStats latA = compute_stats(field(rA.ticks, &TickMeasurement::latency_us));
    const JitterStats latB = compute_stats(field(rB.ticks, &TickMeasurement::latency_us));
    const JitterStats latC = compute_stats(field(rC.ticks, &TickMeasurement::latency_us));
    const JitterStats gpuA = compute_stats(field(rA.ticks, &TickMeasurement::gpu_exec_ms));
    const JitterStats gpuB = compute_stats(field(rB.ticks, &TickMeasurement::gpu_exec_ms));
    const JitterStats gpuC = compute_stats(field(rC.ticks, &TickMeasurement::gpu_exec_ms));
    const JitterStats perA = compute_stats(field(rA.ticks, &TickMeasurement::period_us));
    const JitterStats perB = compute_stats(field(rB.ticks, &TickMeasurement::period_us));
    const JitterStats perC = compute_stats(field(rC.ticks, &TickMeasurement::period_us));

    std::printf("[time] mode A stream : submit mean %.1f us (p50 %.1f, p95 %.1f, p99 %.1f, max %.1f) | "
                "latency mean %.1f us | gpu-exec mean %.4f ms\n",
                submitA.mean, submitA.p50, submitA.p95, submitA.p99, submitA.max, latA.mean, gpuA.mean);
    std::printf("[time] mode B graph  : submit mean %.1f us (p50 %.1f, p95 %.1f, p99 %.1f, max %.1f) | "
                "latency mean %.1f us | gpu-exec mean %.4f ms\n",
                submitB.mean, submitB.p50, submitB.p95, submitB.p99, submitB.max, latB.mean, gpuB.mean);
    std::printf("[time] mode C graph+setparams: submit mean %.1f us (p50 %.1f, p95 %.1f, p99 %.1f, max %.1f) | "
                "latency mean %.1f us | gpu-exec mean %.4f ms\n",
                submitC.mean, submitC.p50, submitC.p95, submitC.p99, submitC.max, latC.mean, gpuC.mean);
    std::printf("[info] achieved pacing period (target %.1f us): A p50=%.1f B p50=%.1f C p50=%.1f | "
                "tail (max) A=%.1f B=%.1f C=%.1f\n",
                1.0e6 / sc.pacing_hz, perA.p50, perB.p50, perC.p50, perA.max, perB.max, perC.max);

    // Gate 1: mean SUBMIT time is measurably lower for the graph modes than
    // for naive stream launches -- the one claim this project can make
    // reliably (README/THEORY explain why). Conservative margin: require at
    // least a 25% reduction, well under what CUDA Graphs typically deliver
    // for a 10+ call tick on WDDM (see the ACTUAL measured numbers above/in
    // README, which are usually a much larger factor).
    constexpr double kSubmitMarginFactor = 0.75;   // graph mean must be <= 75% of naive mean
    const bool gate_submit_b = submitB.mean <= submitA.mean * kSubmitMarginFactor;
    const bool gate_submit_c = submitC.mean <= submitA.mean * kSubmitMarginFactor;
    std::printf("[info] submit means: A(stream)=%.1f us, B(graph)=%.1f us, C(graph+setparams)=%.1f us\n",
                submitA.mean, submitB.mean, submitC.mean);
    std::printf("GATE submit-reduction: %s (both graph modes' mean submit time <= %.0f%% of the naive "
                "stream mode's mean -- see the [info] line above for the actual measured means)\n",
                (gate_submit_b && gate_submit_c) ? "PASS" : "FAIL", kSubmitMarginFactor * 100.0);

    // Gate 2: the GPU's own device-timeline work per tick is the SAME work
    // in every mode (identical kernels, identical data) -- means must agree
    // within a generous band; only WDDM scheduling noise between
    // separately-timed runs should separate them, not the orchestration
    // technique. Band picked wide (2x) precisely because it is a sanity
    // check on "same work happened," not a performance claim.
    constexpr double kGpuConsistencyFactor = 2.0;
    const double gpu_max = std::max({ gpuA.mean, gpuB.mean, gpuC.mean });
    const double gpu_min = std::min({ gpuA.mean, gpuB.mean, gpuC.mean });
    const bool gate_gpu_consistency = (gpu_min > 0.0) && (gpu_max / gpu_min <= kGpuConsistencyFactor);
    std::printf("[info] mean device-exec time: A=%.4f ms, B=%.4f ms, C=%.4f ms (max/min ratio %.2fx)\n",
                gpuA.mean, gpuB.mean, gpuC.mean, gpu_min > 0.0 ? gpu_max / gpu_min : -1.0);
    std::printf("GATE gpu-work-consistency: %s (mean device-timeline time per tick agrees across all "
                "three modes within %.1fx -- same kernels, same data, so it should -- see [info] above)\n",
                gate_gpu_consistency ? "PASS" : "FAIL", kGpuConsistencyFactor);

    // Gate 3: pacing sanity -- achieved p50 period within documented
    // tolerance of the 4 ms target, for every mode. Never gates on tail
    // percentiles (max/p99) -- those are reported as [info] above, exactly
    // per this project's brief, because WDDM/OS tail jitter is real and
    // this project's didactic job is to MEASURE it honestly, not hide it
    // behind a gate it might legitimately fail on a loaded machine.
    constexpr double kPacingToleranceUs = 600.0;   // +-15% of the 4000 us target
    const double target_us = 1.0e6 / sc.pacing_hz;
    const bool gate_pacing = std::fabs(perA.p50 - target_us) <= kPacingToleranceUs
                           && std::fabs(perB.p50 - target_us) <= kPacingToleranceUs
                           && std::fabs(perC.p50 - target_us) <= kPacingToleranceUs;
    std::printf("[info] achieved p50 period: A=%.1f us, B=%.1f us, C=%.1f us (target %.1f us)\n",
                perA.p50, perB.p50, perC.p50, target_us);
    std::printf("GATE pacing-accuracy: %s (every mode's achieved p50 tick period is within +-%.0f us "
                "of the %.0f Hz target -- see [info] above for the actual measured periods)\n",
                gate_pacing ? "PASS" : "FAIL", kPacingToleranceUs, sc.pacing_hz);

    // ======================= ARTIFACTS ========================================
    const std::string out_dir = project_root_from(argv[0]) + "/demo/out";
    bool artifact_ok = ensure_dir(out_dir);
    if (artifact_ok) {
        std::ofstream f(out_dir + "/latency_histogram.csv");
        artifact_ok = f.is_open();
        if (artifact_ok) {
            f << "mode,tick,submit_us,latency_us,gpu_exec_ms,period_us\n";
            auto dump = [&](const char* name, const std::vector<TickMeasurement>& v) {
                for (const auto& m : v)
                    f << name << ',' << m.tick << ',' << m.submit_us << ',' << m.latency_us << ','
                      << m.gpu_exec_ms << ',' << m.period_us << '\n';
            };
            dump("stream_naive", rA.ticks);
            dump("graph_static", rB.ticks);
            dump("graph_setparams", rC.ticks);
        }
    }
    if (artifact_ok)
        std::printf("ARTIFACT: wrote demo/out/latency_histogram.csv (%d rows)\n",
                    3 * sc.measured_ticks);
    else
        std::printf("ARTIFACT: FAILED to write demo/out/latency_histogram.csv\n");

    bool summary_ok = artifact_ok;
    if (summary_ok) {
        std::ofstream f(out_dir + "/jitter_summary.csv");
        summary_ok = f.is_open();
        if (summary_ok) {
            f << "mode,metric,mean,p50,p95,p99,max\n";
            auto row = [&](const char* mode, const char* metric, const JitterStats& s) {
                f << mode << ',' << metric << ',' << s.mean << ',' << s.p50 << ',' << s.p95 << ','
                  << s.p99 << ',' << s.max << '\n';
            };
            row("stream_naive", "submit_us", submitA);   row("stream_naive", "latency_us", latA);
            row("stream_naive", "period_us", perA);       row("stream_naive", "gpu_exec_ms", gpuA);
            row("graph_static", "submit_us", submitB);    row("graph_static", "latency_us", latB);
            row("graph_static", "period_us", perB);       row("graph_static", "gpu_exec_ms", gpuB);
            row("graph_setparams", "submit_us", submitC); row("graph_setparams", "latency_us", latC);
            row("graph_setparams", "period_us", perC);    row("graph_setparams", "gpu_exec_ms", gpuC);
        }
    }
    if (summary_ok)
        std::printf("ARTIFACT: wrote demo/out/jitter_summary.csv (12 rows)\n");
    else
        std::printf("ARTIFACT: FAILED to write demo/out/jitter_summary.csv\n");

    // ======================= CLEANUP ==========================================
    CUDA_CHECK(cudaGraphExecDestroy(graphExecB));
    CUDA_CHECK(cudaGraphDestroy(graphB));
    CUDA_CHECK(cudaGraphExecDestroy(gcC.exec));
    CUDA_CHECK(cudaGraphDestroy(gcC.graph));
    CUDA_CHECK(cudaEventDestroy(evStart));
    CUDA_CHECK(cudaEventDestroy(evStop));
    CUDA_CHECK(cudaFreeHost(h_sensor_pinned));
    CUDA_CHECK(cudaFreeHost(h_eps_pinned));
    free_device_buffers(db);
    CUDA_CHECK(cudaStreamDestroy(stream));

    // ======================= VERDICT ===========================================
    const bool success = verify_pass && crossmode_pass && artifact_ok && summary_ok
                       && gate_submit_b && gate_submit_c && gate_gpu_consistency && gate_pacing;
    if (success)
        std::printf("RESULT: PASS (tick correctness verified, all three modes bit-identical, "
                    "submit/consistency/pacing gates satisfied)\n");
    else
        std::printf("RESULT: FAIL (see the VERIFY/CROSSMODE/GATE lines above for which check failed)\n");
    return success ? 0 : 1;
}
