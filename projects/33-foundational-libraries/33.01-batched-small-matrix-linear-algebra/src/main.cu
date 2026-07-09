// ===========================================================================
// main.cu — entry point for project 33.01
//           Batched small-matrix linear algebra (3×3, 4×4, 6×6 —
//           the robotics sizes)
//
// What this program does, start to finish
// ---------------------------------------
//   1. Print the demo banner + GPU info.
//   2. SAMPLE STAGE: load the tiny committed sample (data/sample/
//      smallmat_sample.csv, synthetic, seed 42), run every problem in it on
//      BOTH the GPU kernels and the CPU oracle, and compare within the
//      documented tolerances. This is the offline, zero-download
//      reproducibility check CLAUDE.md §4 demands.
//   3. BATCH STAGE: generate a large deterministic batch IN MEMORY (fixed
//      xorshift32 seed — bit-identical on every platform, see rng notes
//      below), verify GPU == CPU on all of it, and time the n=6 kernels
//      against the single-core oracle for the demo's teaching speed-up line.
//   4. Exit 0 only if every comparison passed (the demo script and CI-style
//      wrappers key off this).
//
// The output contract (shared with demo/run_demo.ps1 + expected_output.txt)
// --------------------------------------------------------------------------
// STABLE lines — printed identically on every machine, and therefore safe
// for demo/expected_output.txt to check verbatim:
//     "[demo] ..."   the banner
//     "PROBLEM: ..." what is computed
//     "SAMPLE: ..."  sample composition (fixed by the committed file)
//     "SAMPLE RESULT: ..." PASS/FAIL + tolerances
//     "BATCH: ..."   batch composition (fixed by the default arguments)
//     "RESULT: ..."  the overall verdict
// UNSTABLE lines — machine/run dependent, deliberately NOT checked:
//     "[info] ..."   device name, file paths, worst deviations
//     "[time] ..."   timings and the teaching speed-up
// If you change any stable line, update demo/expected_output.txt in the
// same commit — run_demo.ps1 will fail loudly otherwise, by design.
//
// Tolerances (justified in THEORY.md §numerics, quoted here for the reader):
//   matmul: |gpu−cpu| ≤ 1e-5 absolute. Inputs are in [−1,1), so each output
//           is a ≤6-term dot product bounded by 6; FP32 rounding plus
//           FMA-contraction differences land near 1e-7 — 1e-5 is ~100×
//           headroom without masking real indexing bugs (those produce
//           errors of order 1, not 1e-6).
//   solve:  |gpu−cpu| ≤ 1e-4 · max(1, |cpu|) per element. The generator
//           guarantees A = G·Gᵀ + n·I (condition number single-digit), so
//           both factorizations agree to ~1e-6 relative; 1e-4 is safe
//           headroom, and NaN-vs-NaN counts as agreement (shared policy).
//
// Read this first, then kernels.cuh → reference_cpu.cpp → kernels.cu.
// ===========================================================================

#include "kernels.cuh"            // GPU launchers + CPU oracle signatures
#include "util/cuda_check.cuh"    // CUDA_CHECK / device banner (§6.1 rule 7)
#include "util/timer.cuh"         // GpuTimer (cudaEvent) + CpuTimer (chrono)

#include <cmath>                  // std::isnan, std::fabs
#include <cstdint>                // uint32_t for the deterministic RNG
#include <cstdio>                 // printf-family: the demo's entire UI
#include <cstring>                // strcmp for the tiny argv parser
#include <fstream>                // sample CSV loading
#include <map>                    // label → values from the sample file
#include <sstream>                // CSV row parsing
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Deterministic RNG: xorshift32 (Marsaglia 2003), hand-rolled on purpose.
//
// Why not std::mt19937 + std::uniform_real_distribution? The ENGINE is
// portable, but the DISTRIBUTION is not — the C++ standard lets libstdc++
// and MSVC's STL turn the same engine stream into different floats. This
// demo promises "BATCH: ... [seed 42]" reproduces bit-identical inputs on
// Windows and Linux, so it uses a 6-line generator whose every bit is
// specified right here. (cuRAND enters the repo in projects that generate
// noise ON the device — 08.01 MPPI; here the host generates once, so a tiny
// host RNG is the honest tool.)
// ---------------------------------------------------------------------------
static inline uint32_t xorshift32(uint32_t& state)
{
    // Three shift-xor steps; period 2^32−1 over nonzero states. Constants
    // (13,17,5) are Marsaglia's classic full-period triple.
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

// Uniform float in [−1, 1): take the top 24 bits (a float's full mantissa
// width) so the conversion to float is EXACT — no double rounding anywhere,
// hence bit-identical results on every IEEE-754 platform.
static inline float uniform_pm1(uint32_t& state)
{
    float u01 = (xorshift32(state) >> 8) * (1.0f / 16777216.0f);  // [0,1), 24-bit exact
    return 2.0f * u01 - 1.0f;
}

// Fill `v` with uniform [−1,1) values from a stream seeded by `seed`.
// Distinct arrays get distinct seeds (caller passes 42+tag) so no two
// arrays share a stream — cheap insurance against accidental correlation.
static void fill_uniform_pm1(std::vector<float>& v, uint32_t seed)
{
    uint32_t s = seed ? seed : 1u;      // xorshift32 must not start at 0 (fixed point)
    for (float& e : v) e = uniform_pm1(s);
}

// ---------------------------------------------------------------------------
// make_spd_batch — build `count` guaranteed-SPD n×n matrices: A = G·Gᵀ + n·I.
//
// Why this construction: G·Gᵀ is symmetric positive SEMI-definite for any G;
// adding n·I pushes every eigenvalue up by n, making A strictly positive
// definite with eigenvalues in [n, n + ‖G‖²] — condition number a single
// digit for entries in [−1,1). Well-conditioned inputs keep the FP32
// GPU-vs-CPU comparison about CODE differences, not about amplified
// rounding (the ill-conditioned story is README Exercise 4). The sample
// generator (scripts/make_synthetic.py) uses the identical construction.
// ---------------------------------------------------------------------------
static void make_spd_batch(int n, int count, std::vector<float>& A, uint32_t seed)
{
    A.assign(static_cast<size_t>(count) * n * n, 0.0f);
    std::vector<float> G(static_cast<size_t>(n) * n);   // scratch: one random factor per matrix
    uint32_t s = seed ? seed : 1u;

    for (int k = 0; k < count; ++k) {
        for (float& g : G) g = uniform_pm1(s);           // fresh random G for matrix k
        float* a = A.data() + static_cast<size_t>(k) * n * n;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                float acc = (i == j) ? static_cast<float>(n) : 0.0f;  // the +n·I term
                for (int p = 0; p < n; ++p)
                    acc += G[i * n + p] * G[j * n + p];  // (G·Gᵀ)(i,j) = row i of G · row j of G
                a[i * n + j] = acc;
            }
    }
}

// ---------------------------------------------------------------------------
// GPU round-trips: allocate → H2D → launch → D2H → free.
//
// Deliberately plain (no pooling, no streams, no pinned memory): this
// project teaches the KERNELS; transfer engineering has its own projects in
// domain 32. Every CUDA call is checked (§6.1 rule 7).
// ---------------------------------------------------------------------------

// Run the GPU batched matmul on host data; fills C_gpu. Optionally reports
// the kernel-only time (transfers excluded — the [time] line says so).
static void gpu_matmul(int n, int count, const std::vector<float>& A,
                       const std::vector<float>& B, std::vector<float>& C_gpu,
                       float* kernel_ms /*nullable: pass nullptr to skip timing*/)
{
    const size_t mat_bytes = static_cast<size_t>(count) * n * n * sizeof(float);
    C_gpu.assign(static_cast<size_t>(count) * n * n, 0.0f);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;   // d_ prefix: device pointers (repo convention)
    CUDA_CHECK(cudaMalloc(&d_A, mat_bytes));                 // may fail with cudaErrorMemoryAllocation on small GPUs
    CUDA_CHECK(cudaMalloc(&d_B, mat_bytes));
    CUDA_CHECK(cudaMalloc(&d_C, mat_bytes));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), mat_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), mat_bytes, cudaMemcpyHostToDevice));

    if (kernel_ms) {
        GpuTimer t;                                          // cudaEvent timing — util/timer.cuh explains why not chrono
        t.begin();
        launch_batched_matmul(n, count, d_A, d_B, d_C);
        *kernel_ms = t.end_ms();                             // synchronizes: kernel is done after this returns
    } else {
        launch_batched_matmul(n, count, d_A, d_B, d_C);
    }

    // This D2H copy doubles as the synchronization point where any in-kernel
    // fault would surface (see util/cuda_check.cuh's "sticky error" note).
    CUDA_CHECK(cudaMemcpy(C_gpu.data(), d_C, mat_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

// Run the GPU batched Cholesky solve on host data; fills x_gpu. Same shape
// as gpu_matmul — read that one first; only the buffer shapes differ.
static void gpu_solve(int n, int count, const std::vector<float>& A,
                      const std::vector<float>& b, std::vector<float>& x_gpu,
                      float* kernel_ms /*nullable*/)
{
    const size_t mat_bytes = static_cast<size_t>(count) * n * n * sizeof(float);
    const size_t vec_bytes = static_cast<size_t>(count) * n * sizeof(float);
    x_gpu.assign(static_cast<size_t>(count) * n, 0.0f);

    float *d_A = nullptr, *d_b = nullptr, *d_x = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, mat_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_x, vec_bytes));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), mat_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), vec_bytes, cudaMemcpyHostToDevice));

    if (kernel_ms) {
        GpuTimer t;
        t.begin();
        launch_batched_cholesky_solve(n, count, d_A, d_b, d_x);
        *kernel_ms = t.end_ms();
    } else {
        launch_batched_cholesky_solve(n, count, d_A, d_b, d_x);
    }

    CUDA_CHECK(cudaMemcpy(x_gpu.data(), d_x, vec_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_x));
}

// ---------------------------------------------------------------------------
// Comparators — GPU vs CPU within the documented tolerances (file header).
// Both return the worst deviation seen (for the honest [info] report) and
// clear `pass` on any element outside tolerance.
// ---------------------------------------------------------------------------

// Absolute tolerance (matmul outputs are O(1)-bounded; see file header).
static float max_abs_diff(const std::vector<float>& gpu,
                          const std::vector<float>& cpu, bool& pass, float tol)
{
    float worst = 0.0f;
    for (size_t i = 0; i < gpu.size(); ++i) {
        float d = std::fabs(gpu[i] - cpu[i]);
        if (d > worst) worst = d;
        if (d > tol) pass = false;
    }
    return worst;
}

// Relative-with-floor tolerance for solve results: |Δ| ≤ tol·max(1,|cpu|).
// The max(1,·) floor keeps near-zero solution entries from demanding
// impossible absolute precision. NaN-vs-NaN agrees (shared non-SPD policy);
// NaN on one side only is an automatic fail — that asymmetry is exactly the
// bug class this check exists to catch.
static float max_rel_diff(const std::vector<float>& gpu,
                          const std::vector<float>& cpu, bool& pass, float tol)
{
    float worst = 0.0f;
    for (size_t i = 0; i < gpu.size(); ++i) {
        bool gn = std::isnan(gpu[i]), cn = std::isnan(cpu[i]);
        if (gn || cn) {
            if (gn != cn) pass = false;   // one-sided NaN: the two policies diverged — a real bug
            continue;                     // both NaN: agreed failure, fine
        }
        float scale = std::fabs(cpu[i]) > 1.0f ? std::fabs(cpu[i]) : 1.0f;
        float d = std::fabs(gpu[i] - cpu[i]) / scale;
        if (d > worst) worst = d;
        if (d > tol) pass = false;
    }
    return worst;
}

// ---------------------------------------------------------------------------
// Sample-file handling.
//
// Format (written by scripts/make_synthetic.py; documented in data/README.md):
//   * '#' lines are comments (provenance, seed, the SYNTHETIC label).
//   * data rows:  label,index,v0,v1,...,v{m-1}   with label ∈
//     {A3,B3,A4,B4,A6,B6,S6,b6}; matrices row-major (m = n·n), vectors m = n.
// The loader is forgiving about row ORDER (labels partition the rows) but
// strict about VALUE COUNTS — a malformed row aborts the load, and the demo
// then fails its SAMPLE stable lines. Corrupt data must never quietly pass.
// ---------------------------------------------------------------------------
struct SampleData {
    std::map<std::string, std::vector<float>> values;  // label → concatenated values (file order)
    std::map<std::string, int> rows;                   // label → number of rows seen
    bool loaded = false;
};

static SampleData load_sample(const std::string& path)
{
    SampleData s;
    std::ifstream in(path);
    if (!in.is_open()) return s;                       // caller reports; the path search tries several candidates

    // Expected value count per label; anything else is a format error.
    // (n=3→9, n=4→16, n=6→36 matrix values; b6→6 vector values.)
    const std::map<std::string, size_t> expect = {
        {"A3", 9}, {"B3", 9}, {"A4", 16}, {"B4", 16},
        {"A6", 36}, {"B6", 36}, {"S6", 36}, {"b6", 6},
    };

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;  // comment/provenance lines
        std::stringstream ss(line);
        std::string cell, label;
        if (!std::getline(ss, label, ',')) continue;   // column 1: the row label
        auto it = expect.find(label);
        if (it == expect.end()) {
            std::fprintf(stderr, "sample: unknown row label '%s'\n", label.c_str());
            return SampleData{};                       // fail the whole load — no partial samples
        }
        std::getline(ss, cell, ',');                   // column 2: row index (informational; order not required)
        size_t got = 0;
        while (std::getline(ss, cell, ',')) {          // remaining columns: the values
            s.values[label].push_back(std::strtof(cell.c_str(), nullptr));
            ++got;
        }
        if (got != it->second) {
            std::fprintf(stderr, "sample: row '%s' has %zu values, expected %zu\n",
                         label.c_str(), got, it->second);
            return SampleData{};
        }
        s.rows[label] += 1;
    }

    // Pairwise consistency: every A needs its B, every S6 its b6.
    if (s.rows["A3"] != s.rows["B3"] || s.rows["A4"] != s.rows["B4"] ||
        s.rows["A6"] != s.rows["B6"] || s.rows["S6"] != s.rows["b6"]) {
        std::fprintf(stderr, "sample: mismatched pair counts\n");
        return SampleData{};
    }
    s.loaded = true;
    return s;
}

// Locate the sample file. Order: explicit --data path; then relative to the
// EXECUTABLE (argv[0]) — the exe lives at build/x64/<Config>/, so the
// project root is three directories up; then cwd-relative fallbacks for
// people running by hand from the project root or from build/.
static std::string find_sample(const std::string& cli_path, const char* argv0)
{
    std::vector<std::string> candidates;
    if (!cli_path.empty()) candidates.push_back(cli_path);

    std::string exe(argv0 ? argv0 : "");
    size_t cut = exe.find_last_of("/\\");             // strip the exe filename, keep its directory
    if (cut != std::string::npos) {
        std::string dir = exe.substr(0, cut);
        candidates.push_back(dir + "/../../../data/sample/smallmat_sample.csv");
    }
    candidates.push_back("data/sample/smallmat_sample.csv");        // run from the project root
    candidates.push_back("../data/sample/smallmat_sample.csv");     // run from build/

    for (const auto& c : candidates)
        if (std::ifstream(c).is_open()) return c;
    return "";
}

// ---------------------------------------------------------------------------
// main — orchestrates the two stages described in the file header.
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // ---- tiny argv parser (the defaults define the checked BATCH line) ----
    int   batch_pairs  = 200000;   // matmul pairs PER SIZE in the batch stage
    int   batch_solves = 100000;   // SPD systems (n=6) in the batch stage
    std::string data_path;         // optional --data override for the sample
    for (int i = 1; i < argc; ++i) {
        if      (!std::strcmp(argv[i], "--count")  && i + 1 < argc) batch_pairs  = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--solves") && i + 1 < argc) batch_solves = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--data")   && i + 1 < argc) data_path    = argv[++i];
        else {
            std::fprintf(stderr,
                "usage: %s [--count N_pairs_per_size] [--solves N_systems] [--data sample.csv]\n"
                "note: non-default sizes change the BATCH output line, which then no longer\n"
                "      matches demo/expected_output.txt — intended for experiments only.\n",
                argv[0]);
            return 2;
        }
    }

    // ---- banner (stable) + device info (unstable) --------------------------
    std::printf("[demo] batched small-matrix linear algebra (project 33.01)\n");
    print_device_info();   // also the earliest loud failure if no CUDA device/driver is present
    std::printf("PROBLEM: batched matmul C=A*B (n=3,4,6) + batched Cholesky solve A*x=b (n=6, SPD), FP32\n");

    const float kMatmulTol = 1e-5f;  // absolute — justification in the file header
    const float kSolveTol  = 1e-4f;  // relative-with-floor — ditto
    bool all_pass = true;

    // ======================= SAMPLE STAGE ===================================
    const std::string sample_path = find_sample(data_path, argv[0]);
    if (sample_path.empty()) {
        // Deliberate hard failure: the committed sample is part of the §4
        // reproducibility contract. Its absence means a broken checkout or a
        // packaging bug — the demo must not "pass" around it.
        std::printf("SAMPLE: NOT FOUND — data/sample/smallmat_sample.csv missing (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (sample data missing)\n");
        return 1;
    }
    std::printf("[info] sample file: %s\n", sample_path.c_str());

    SampleData sample = load_sample(sample_path);
    if (!sample.loaded) {
        std::printf("SAMPLE: MALFORMED — see stderr for the offending row\n");
        std::printf("RESULT: FAIL (sample data malformed)\n");
        return 1;
    }
    std::printf("SAMPLE: %d pairs n=3, %d pairs n=4, %d pairs n=6; %d SPD solves n=6 [synthetic, seed 42]\n",
                sample.rows["A3"], sample.rows["A4"], sample.rows["A6"], sample.rows["S6"]);

    // Matmul checks at each size — GPU vs CPU on the identical file data.
    bool sample_pass = true;
    float sample_worst_mm = 0.0f;   // worst matmul deviation across the three sizes (for the [info] line)
    const int sizes[3] = {3, 4, 6};
    for (int n : sizes) {
        const std::string a_lbl = "A" + std::to_string(n), b_lbl = "B" + std::to_string(n);
        int cnt = sample.rows[a_lbl];
        std::vector<float> C_gpu, C_cpu(sample.values[a_lbl].size());
        gpu_matmul(n, cnt, sample.values[a_lbl], sample.values[b_lbl], C_gpu, nullptr);
        batched_matmul_cpu(n, cnt, sample.values[a_lbl].data(),
                           sample.values[b_lbl].data(), C_cpu.data());
        float w = max_abs_diff(C_gpu, C_cpu, sample_pass, kMatmulTol);
        if (w > sample_worst_mm) sample_worst_mm = w;
    }

    // Cholesky solve check (the n=6 systems from the file).
    {
        int cnt = sample.rows["S6"];
        std::vector<float> x_gpu, x_cpu(static_cast<size_t>(cnt) * 6);
        gpu_solve(6, cnt, sample.values["S6"], sample.values["b6"], x_gpu, nullptr);
        batched_cholesky_solve_cpu(6, cnt, sample.values["S6"].data(),
                                   sample.values["b6"].data(), x_cpu.data());
        float w = max_rel_diff(x_gpu, x_cpu, sample_pass, kSolveTol);
        std::printf("[info] sample worst deviation: matmul %.3e abs, solve %.3e rel\n",
                    static_cast<double>(sample_worst_mm), static_cast<double>(w));
    }

    std::printf("SAMPLE RESULT: %s (GPU vs CPU within tol: matmul abs 1e-5, solve rel 1e-4)\n",
                sample_pass ? "PASS" : "FAIL");
    all_pass = all_pass && sample_pass;

    // ======================= BATCH STAGE ====================================
    // Large deterministic batch — same generators, fixed seeds, so this line
    // is stable and checkable. Seeds are 42+tag so no two arrays share a
    // stream (see fill_uniform_pm1).
    std::printf("BATCH: %d pairs per size (n=3,4,6) + %d SPD solves (n=6), generated in-memory [seed 42]\n",
                batch_pairs, batch_solves);

    bool batch_pass = true;
    for (int idx = 0; idx < 3; ++idx) {
        const int n = sizes[idx];
        std::vector<float> A(static_cast<size_t>(batch_pairs) * n * n);
        std::vector<float> B(A.size());
        fill_uniform_pm1(A, 42u + static_cast<uint32_t>(idx) * 2u);   // stream tags 42, 44, 46
        fill_uniform_pm1(B, 43u + static_cast<uint32_t>(idx) * 2u);   // stream tags 43, 45, 47

        std::vector<float> C_gpu, C_cpu(A.size());
        float gpu_ms = 0.0f;
        gpu_matmul(n, batch_pairs, A, B, C_gpu, (n == 6) ? &gpu_ms : nullptr);

        CpuTimer ct;                    // times the whole single-core reference loop (synchronous host code)
        ct.begin();
        batched_matmul_cpu(n, batch_pairs, A.data(), B.data(), C_cpu.data());
        double cpu_ms = ct.end_ms();

        max_abs_diff(C_gpu, C_cpu, batch_pass, kMatmulTol);
        if (n == 6) {
            // Single-shot numbers, kernel-only on the GPU side (transfers
            // excluded, and the line says so) — a teaching artifact, never a
            // benchmark claim (CLAUDE.md §12).
            std::printf("[time] matmul n=6 x %d: CPU %.1f ms | GPU kernel %.3f ms | speed-up %.0fx (teaching artifact; kernel only)\n",
                        batch_pairs, cpu_ms, static_cast<double>(gpu_ms),
                        cpu_ms / (static_cast<double>(gpu_ms) > 0.0 ? static_cast<double>(gpu_ms) : 1.0));
        }
    }

    {
        std::vector<float> A, b(static_cast<size_t>(batch_solves) * 6);
        make_spd_batch(6, batch_solves, A, 48u);      // guaranteed SPD (see make_spd_batch's comment)
        fill_uniform_pm1(b, 49u);

        std::vector<float> x_gpu, x_cpu(b.size());
        float gpu_ms = 0.0f;
        gpu_solve(6, batch_solves, A, b, x_gpu, &gpu_ms);

        CpuTimer ct;
        ct.begin();
        batched_cholesky_solve_cpu(6, batch_solves, A.data(), b.data(), x_cpu.data());
        double cpu_ms = ct.end_ms();

        max_rel_diff(x_gpu, x_cpu, batch_pass, kSolveTol);
        std::printf("[time] cholesky-solve n=6 x %d: CPU %.1f ms | GPU kernel %.3f ms | speed-up %.0fx (teaching artifact; kernel only)\n",
                    batch_solves, cpu_ms, static_cast<double>(gpu_ms),
                    cpu_ms / (static_cast<double>(gpu_ms) > 0.0 ? static_cast<double>(gpu_ms) : 1.0));
    }
    all_pass = all_pass && batch_pass;

    // ---- verdict (stable line; run_demo.ps1 and the exit code key off it) --
    if (all_pass)
        std::printf("RESULT: PASS (all GPU results match the CPU reference within documented tolerances)\n");
    else
        std::printf("RESULT: FAIL (at least one GPU result diverged from the CPU reference — see lines above)\n");
    return all_pass ? 0 : 1;
}
