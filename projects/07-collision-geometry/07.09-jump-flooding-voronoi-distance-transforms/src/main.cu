// ===========================================================================
// main.cu — entry point for project 07.09
//           Jump-flooding Voronoi/distance transforms (easy, visual, useful)
//
// What this program does, start to finish
// ---------------------------------------
//   1. Print the banner + GPU info.
//   2. SAMPLE STAGE: load the committed seed set (data/sample/
//      jfa_seeds.csv: 64 seeds on a 512×512 grid), run GPU JFA and the
//      EXACT CPU scan, and check the approximation bounds (see below).
//   3. ARTIFACT: write two viewable images into demo/out/ — voronoi.pgm
//      (regions colored by label) and distance.pgm (the clearance field) —
//      this project's result is inherently visual (CLAUDE.md §6.3).
//   4. BATCH STAGE: 1024×1024 grid with 128 deterministic in-memory seeds;
//      same bounds check, plus honest timing of both paths.
//   5. Exit 0 only if every stage passed.
//
// The verification contract (unusual — read this even if you read 33.01/09.01)
// ----------------------------------------------------------------------------
// JFA is APPROXIMATE; the CPU oracle is EXACT — so this demo does not check
// "same answer within rounding" but "the approximation keeps its documented
// promise":
//     label mismatches ≤ 0.5% of cells   (mismatch = labels differ AND the
//                                         squared distances differ; exact
//                                         ties are legitimately ambiguous
//                                         and count as agreement)
//     max |d_jfa − d_exact| ≤ 2.0 cells  (distance error of any mislabel)
// All distances are INTEGER squared distances until display — the entire
// comparison is exact arithmetic, bit-identical across machines, so the
// mismatch counts printed on [info] lines are deterministic, not noisy.
//
// Output contract: stable lines are "[demo]", "PROBLEM:", "SAMPLE:",
// "SAMPLE RESULT:", "ARTIFACT:", "BATCH:", "BATCH RESULT:", "RESULT:";
// "[info]"/"[time]" lines are unchecked. Change a stable line ⇒ update
// demo/expected_output.txt in the same commit.
//
// Read this first, then kernels.cuh → reference_cpu.cpp → kernels.cu.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

#include <cerrno>                 // EEXIST for the mkdir helper
#include <cmath>                  // std::sqrt for display-time distances
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#ifdef _WIN32
#include <direct.h>               // _mkdir — <filesystem> is deliberately
                                  // avoided in .cu files: nvcc's frontend on
                                  // MSVC chokes on it; a 5-line mkdir helper
                                  // is the portable, teachable alternative
#else
#include <sys/stat.h>             // mkdir (POSIX twin)
#endif
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Deterministic RNG — the repo's portable xorshift32 (rationale in 33.01).
// ---------------------------------------------------------------------------
static inline uint32_t xorshift32(uint32_t& state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

// ---------------------------------------------------------------------------
// Seed generation for the batch stage: n distinct cells on a W×H grid.
// Distinctness by rejection against an occupancy grid — duplicate seeds
// would violate the scatter kernel's no-collision contract (kernels.cu).
// ---------------------------------------------------------------------------
static std::vector<int> make_seeds(int width, int height, int n, uint32_t seed)
{
    std::vector<int> out;
    out.reserve(static_cast<size_t>(n) * kSeedStride);
    std::vector<uint8_t> taken(static_cast<size_t>(width) * height, 0);
    uint32_t s = seed ? seed : 1u;
    int id = 0;
    while (id < n) {
        // Rejection-sample a free cell. Modulo bias is irrelevant here (we
        // need scattered-and-deterministic, not statistically perfect).
        const int x = static_cast<int>(xorshift32(s) % static_cast<uint32_t>(width));
        const int y = static_cast<int>(xorshift32(s) % static_cast<uint32_t>(height));
        if (taken[static_cast<size_t>(y) * width + x]) continue;
        taken[static_cast<size_t>(y) * width + x] = 1;
        out.push_back(x); out.push_back(y); out.push_back(id);
        ++id;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Sample loading: rows "S,<id>,x,y" for a fixed 512×512 grid. Strict, like
// every loader in this repo: bad labels/counts/coords or duplicate cells
// abort — corrupt samples never quietly pass.
// ---------------------------------------------------------------------------
struct SampleData {
    std::vector<int> seeds;   // n*kSeedStride ints, ids re-checked to be 0..n-1
    int n = 0;
    bool loaded = false;
};

static SampleData load_sample(const std::string& path, int width, int height)
{
    SampleData s;
    std::ifstream in(path);
    if (!in.is_open()) return s;

    std::vector<uint8_t> taken(static_cast<size_t>(width) * height, 0);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string label, cell;
        std::getline(ss, label, ',');
        if (label != "S") {
            std::fprintf(stderr, "sample: unknown row label '%s'\n", label.c_str());
            return SampleData{};
        }
        int vals[3] = { -1, -1, -1 };   // id, x, y
        for (int i = 0; i < 3 && std::getline(ss, cell, ','); ++i)
            vals[i] = std::atoi(cell.c_str());
        const int id = vals[0], x = vals[1], y = vals[2];
        if (id != s.n || x < 0 || x >= width || y < 0 || y >= height) {
            std::fprintf(stderr, "sample: bad row (id=%d x=%d y=%d) — ids must be "
                                 "sequential and coords inside %dx%d\n", id, x, y, width, height);
            return SampleData{};
        }
        if (taken[static_cast<size_t>(y) * width + x]) {
            std::fprintf(stderr, "sample: duplicate seed cell (%d,%d)\n", x, y);
            return SampleData{};
        }
        taken[static_cast<size_t>(y) * width + x] = 1;
        s.seeds.push_back(x); s.seeds.push_back(y); s.seeds.push_back(id);
        s.n += 1;
    }
    s.loaded = (s.n >= 1);
    return s;
}

// Path helpers (same exe-relative resolution as 33.01/09.01: the exe sits at
// build/x64/<Config>/, three levels below the project root).
static std::string project_root_from(const char* argv0)
{
    std::string exe(argv0 ? argv0 : "");
    size_t cut = exe.find_last_of("/\\");
    if (cut == std::string::npos) return ".";
    return exe.substr(0, cut) + "/../../..";
}

static std::string find_sample(const std::string& cli_path, const char* argv0)
{
    std::vector<std::string> candidates;
    if (!cli_path.empty()) candidates.push_back(cli_path);
    candidates.push_back(project_root_from(argv0) + "/data/sample/jfa_seeds.csv");
    candidates.push_back("data/sample/jfa_seeds.csv");
    candidates.push_back("../data/sample/jfa_seeds.csv");
    for (const auto& c : candidates)
        if (std::ifstream(c).is_open()) return c;
    return "";
}

// ---------------------------------------------------------------------------
// GPU round-trip: allocate the cell buffer, run JFA, download the result.
// ---------------------------------------------------------------------------
static void gpu_jfa(int width, int height, const std::vector<int>& seeds, int n,
                    std::vector<int4>& cells_gpu, double* total_ms /*nullable*/)
{
    const size_t bytes = static_cast<size_t>(width) * height * sizeof(int4);
    cells_gpu.assign(static_cast<size_t>(width) * height, make_int4(-1, -1, -1, 0));

    int4* d_cells = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cells, bytes));

    if (total_ms) {
        // End-to-end wall time of launch_jfa (all passes + its internal
        // allocs/copies), fenced by a device sync — labeled exactly so on
        // the [time] line. Per-pass event timing is Exercise 5.
        CpuTimer t;
        t.begin();
        launch_jfa(width, height, seeds.data(), n, d_cells);
        CUDA_CHECK(cudaDeviceSynchronize());
        *total_ms = t.end_ms();
    } else {
        launch_jfa(width, height, seeds.data(), n, d_cells);
    }

    CUDA_CHECK(cudaMemcpy(cells_gpu.data(), d_cells, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_cells));
}

// ---------------------------------------------------------------------------
// The bounds check (the verification contract from the file header).
// ---------------------------------------------------------------------------
struct ApproxMetrics {
    long long mismatched = 0;   // cells where labels AND distances differ
    long long ties = 0;         // cells where labels differ but distances tie (fine)
    double max_dist_err = 0.0;  // max |sqrt(d2_jfa) − sqrt(d2_exact)| in cells
    long long unassigned = 0;   // cells JFA left at sentinel (must be 0)
};

static ApproxMetrics check_bounds(const std::vector<int4>& gpu,
                                  const std::vector<int4>& cpu,
                                  int width, int height,
                                  bool& pass, double frac_limit, double dist_limit)
{
    ApproxMetrics m;
    const long long total = static_cast<long long>(width) * height;
    for (long long i = 0; i < total; ++i) {
        const int4 g = gpu[static_cast<size_t>(i)];
        const int4 c = cpu[static_cast<size_t>(i)];
        const int x = static_cast<int>(i % width), y = static_cast<int>(i / width);
        if (g.x < 0) { m.unassigned++; pass = false; continue; }   // JFA must label every cell

        const long long dg = (long long)(x - g.x) * (x - g.x) + (long long)(y - g.y) * (y - g.y);
        const long long dc = (long long)(x - c.x) * (x - c.x) + (long long)(y - c.y) * (y - c.y);
        if (g.z == c.z) continue;                 // same label — agreement
        if (dg == dc) { m.ties++; continue; }     // different label, same distance — a true tie
        m.mismatched++;
        const double err = std::sqrt(static_cast<double>(dg)) - std::sqrt(static_cast<double>(dc));
        // dg >= dc always (the exact oracle is optimal) — err is the JFA's
        // overestimate of clearance at this cell; keep the worst.
        if (err > m.max_dist_err) m.max_dist_err = err;
    }
    if (m.mismatched > frac_limit * static_cast<double>(total)) pass = false;
    if (m.max_dist_err > dist_limit) pass = false;
    return m;
}

// ---------------------------------------------------------------------------
// PGM artifacts — the smallest real image format there is (P5: one header
// line trio, then raw bytes), viewable in any image tool, no libraries.
// ---------------------------------------------------------------------------
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

// Create one directory level; succeeding OR already-existing both count as
// success (the only two outcomes a demo cares about). demo/out sits directly
// under the existing demo/, so one level is all we ever need.
static bool ensure_dir(const std::string& path)
{
#ifdef _WIN32
    const int r = _mkdir(path.c_str());
#else
    const int r = mkdir(path.c_str(), 0755);
#endif
    return r == 0 || errno == EEXIST;
}

static bool write_artifacts(const std::string& out_dir, int width, int height,
                            const std::vector<int4>& cells)
{
    if (!ensure_dir(out_dir)) return false;   // demo/out/ is git-ignored scratch

    const size_t total = static_cast<size_t>(width) * height;
    std::vector<uint8_t> vor(total), dist(total);

    // Pass 1: find the max distance for display normalization.
    double dmax = 1.0;
    for (size_t i = 0; i < total; ++i) {
        const int4 c = cells[i];
        const int x = static_cast<int>(i % width), y = static_cast<int>(i / width);
        const double d = std::sqrt(static_cast<double>(
            (long long)(x - c.x) * (x - c.x) + (long long)(y - c.y) * (y - c.y)));
        if (d > dmax) dmax = d;
    }
    // Pass 2: fill both images. Voronoi gray = a multiplicative hash of the
    // label so adjacent regions get visually distinct tones (97 is coprime
    // with 256 ⇒ all labels map to distinct grays until they wrap).
    for (size_t i = 0; i < total; ++i) {
        const int4 c = cells[i];
        const int x = static_cast<int>(i % width), y = static_cast<int>(i / width);
        const double d = std::sqrt(static_cast<double>(
            (long long)(x - c.x) * (x - c.x) + (long long)(y - c.y) * (y - c.y)));
        vor[i]  = static_cast<uint8_t>(32 + (c.z * 97) % 224);      // labels → tones (avoid near-black)
        dist[i] = static_cast<uint8_t>(255.0 * d / dmax + 0.5);     // clearance: dark = at a seed
    }
    return write_pgm(out_dir + "/voronoi.pgm", width, height, vor)
        && write_pgm(out_dir + "/distance.pgm", width, height, dist);
}

// ---------------------------------------------------------------------------
// One full stage: JFA vs exact, bounds check, report. Returns stage pass.
// ---------------------------------------------------------------------------
static bool run_stage(const char* stage_name, int width, int height,
                      const std::vector<int>& seeds, int n,
                      std::vector<int4>* keep_cells /*nullable: for artifacts*/,
                      bool timed)
{
    std::vector<int4> cells_gpu;
    std::vector<int4> cells_cpu(static_cast<size_t>(width) * height);

    double gpu_ms = 0.0;
    gpu_jfa(width, height, seeds, n, cells_gpu, timed ? &gpu_ms : nullptr);

    CpuTimer ct;
    ct.begin();
    voronoi_exact_cpu(width, height, seeds.data(), n, cells_cpu.data());
    const double cpu_ms = ct.end_ms();

    bool pass = true;
    const ApproxMetrics m = check_bounds(cells_gpu, cells_cpu, width, height,
                                         pass, 0.005, 2.0);
    const long long total = static_cast<long long>(width) * height;
    std::printf("[info] %s: %lld/%lld cells mismatched (%.4f%%), %lld exact ties, "
                "max distance error %.3f cells, %lld unassigned\n",
                stage_name, m.mismatched, total,
                100.0 * static_cast<double>(m.mismatched) / static_cast<double>(total),
                m.ties, m.max_dist_err, m.unassigned);
    if (timed)
        std::printf("[time] %s: CPU exact scan %.1f ms | GPU JFA end-to-end %.2f ms "
                    "(all passes + internal alloc/copies) | speed-up %.0fx (teaching artifact)\n",
                    stage_name, cpu_ms, gpu_ms, cpu_ms / (gpu_ms > 0.0 ? gpu_ms : 1.0));

    if (keep_cells) *keep_cells = std::move(cells_gpu);
    return pass;
}

// ---------------------------------------------------------------------------
// main.
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    std::string data_path;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--data") && i + 1 < argc) data_path = argv[++i];
        else {
            std::fprintf(stderr, "usage: %s [--data jfa_seeds.csv]\n", argv[0]);
            return 2;
        }
    }

    std::printf("[demo] jump-flooding Voronoi + distance transform (project 07.09)\n");
    print_device_info();
    std::printf("PROBLEM: nearest-seed labels + distance field on a 2D grid; JFA (approx, GPU) vs exact scan (CPU)\n");

    bool all_pass = true;

    // ======================= SAMPLE STAGE ===================================
    const int W = 512, H = 512;   // the committed sample's grid (fixed — part of the stable SAMPLE line)
    const std::string sample_path = find_sample(data_path, argv[0]);
    if (sample_path.empty()) {
        std::printf("SAMPLE: NOT FOUND — data/sample/jfa_seeds.csv missing (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (sample data missing)\n");
        return 1;
    }
    std::printf("[info] sample file: %s\n", sample_path.c_str());
    SampleData sample = load_sample(sample_path, W, H);
    if (!sample.loaded) {
        std::printf("SAMPLE: MALFORMED — see stderr\n");
        std::printf("RESULT: FAIL (sample data malformed)\n");
        return 1;
    }
    std::printf("SAMPLE: %dx%d grid, %d seeds [synthetic, seed 42]\n", W, H, sample.n);

    std::vector<int4> sample_cells;   // kept for the artifact images
    const bool sample_pass = run_stage("sample", W, H, sample.seeds, sample.n,
                                       &sample_cells, /*timed=*/false);
    std::printf("SAMPLE RESULT: %s (label mismatches <= 0.5%% of cells; max |d_jfa - d_exact| <= 2 cells)\n",
                sample_pass ? "PASS" : "FAIL");
    all_pass = all_pass && sample_pass;

    // ======================= ARTIFACT =======================================
    // The result is inherently visual — ship it as images (CLAUDE.md §6.3).
    // demo/out/ is git-ignored run-time scratch; the images regenerate on
    // every run. Paths print relative on the stable line (machine-neutral),
    // absolute on the [info] line.
    const std::string out_dir = project_root_from(argv[0]) + "/demo/out";
    if (write_artifacts(out_dir, W, H, sample_cells)) {
        std::printf("[info] artifact dir: %s\n", out_dir.c_str());
        std::printf("ARTIFACT: wrote demo/out/voronoi.pgm and demo/out/distance.pgm (512x512)\n");
    } else {
        std::printf("ARTIFACT: FAILED to write demo/out images\n");
        all_pass = false;   // the visual artifact is part of this project's deliverable
    }

    // ======================= BATCH STAGE ====================================
    const int BW = 1024, BH = 1024, BN = 128;
    std::printf("BATCH: %dx%d grid, %d seeds, generated in-memory [seed 42]\n", BW, BH, BN);
    const std::vector<int> batch_seeds = make_seeds(BW, BH, BN, 42u);
    const bool batch_pass = run_stage("batch", BW, BH, batch_seeds, BN,
                                      nullptr, /*timed=*/true);
    std::printf("BATCH RESULT: %s (label mismatches <= 0.5%% of cells; max |d_jfa - d_exact| <= 2 cells)\n",
                batch_pass ? "PASS" : "FAIL");
    all_pass = all_pass && batch_pass;

    // ---- verdict ------------------------------------------------------------
    if (all_pass)
        std::printf("RESULT: PASS (JFA within documented approximation bounds of the exact CPU field on all stages)\n");
    else
        std::printf("RESULT: FAIL (a stage exceeded the documented approximation bounds — see lines above)\n");
    return all_pass ? 0 : 1;
}
