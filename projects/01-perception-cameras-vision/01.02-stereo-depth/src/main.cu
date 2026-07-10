// ===========================================================================
// main.cu — entry point for project 01.02
//           Stereo depth: block matching, then Semi-Global Matching (SGM)
//
// What this program does, start to finish
// ---------------------------------------
//   1. Print the banner + GPU info; load the committed synthetic stereo
//      pair + dense ground truth from data/sample/ (left.pgm, right.pgm,
//      gt_disparity.pgm, gt_valid.pgm — see ../scripts/make_synthetic.py).
//   2. GPU PIPELINE: census -> cost volume (shared by both methods), then
//      BLOCK MATCHING (winner-take-all -> left-right check — the teaching
//      BASELINE, fast and visibly streaky) and SEMI-GLOBAL MATCHING (4-path
//      aggregation -> winner-take-all -> left-right check -> 3x3 median —
//      the SAME cost volume, a smoothness prior added).
//   3. VERIFY STAGE (the §5 GPU-vs-CPU gate, made EXACT by this project's
//      all-integer math — see kernels.cuh): census, the cost volume, ONE
//      SGM aggregation path, and both FINAL disparity maps must match
//      reference_cpu.cpp bit-for-bit. Zero tolerance, because zero
//      floating point is involved anywhere in this pipeline.
//   4. GROUND-TRUTH GATE: for both BM and SGM, the "good-pixel rate"
//      (fraction of GT-valid, non-occluded pixels with |disp - gt| <= 1)
//      must clear a documented floor, AND SGM's rate must exceed BM's by
//      a documented margin — the whole reason this project builds BOTH
//      methods side by side instead of just SGM.
//   5. ARTIFACTS: demo/out/disparity_bm.pgm, disparity_sgm.pgm (both
//      scaled *4, matching gt_disparity.pgm's convention) and
//      error_map.pgm (SGM's per-pixel correctness against ground truth) —
//      the visual story a printed percentage cannot tell by itself.
//
// Output contract: stable lines are "[demo]", "PROBLEM:", "DATA:",
// "VERIFY:", "BM:", "SGM:", "ARTIFACT:", "RESULT:"; "[info]"/"[time]" lines
// are unchecked. Change a stable line => update demo/expected_output.txt in
// the same commit.
//
// Read this first, then kernels.cuh -> reference_cpu.cpp -> kernels.cu.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#ifdef _WIN32
#include <direct.h>               // _mkdir (std::filesystem avoided in .cu — see 07.09)
#else
#include <sys/stat.h>
#endif

// ---------------------------------------------------------------------------
// Ground-truth gate thresholds — calibrated from an ACTUAL measured run on
// this project's committed sample (numbers recorded in THEORY.md "How we
// verify correctness" and README "Expected output"; never asserted from
// theory alone, per CLAUDE.md §8 "never fabricate"). Wide margins below the
// measured values so the gate stays robust to the ~1-ULP-free but still
// platform-dependent double-precision arithmetic used ONLY in the
// percentage math below (the disparities feeding it are exact integers,
// bit-identical on every platform — see kernels.cuh).
//   Measured on the reference machine (RTX 2080 SUPER, sm_75):
//     BM good-pixel rate  = 63.35%
//     SGM good-pixel rate = 97.52%
//     SGM margin over BM  = 34.17 percentage points
// ---------------------------------------------------------------------------
static constexpr double kMinGoodPixelRateBM = 45.0;   // floor, ~18 pts below measured (63.35%)
static constexpr double kMinGoodPixelRateSGM = 85.0;  // floor, ~13 pts below measured (97.52%)
static constexpr double kMinSgmMarginOverBm = 15.0;   // floor, ~19 pts below measured margin (34.17)
static constexpr int kDispTolerance = 1;              // |disp - gt| <= this counts as "good"
static constexpr int kGtDispScale = 4;                // must match scripts/make_synthetic.py DISP_SCALE

// ---------------------------------------------------------------------------
// Minimal, STRICT PGM (P5, 8-bit binary grayscale) reader — matches exactly
// the format ../scripts/make_synthetic.py writes (and 07.09's write_pgm
// produces): "P5\n<W> <H>\n255\n" then W*H raw bytes. Not a general-purpose
// PGM parser (no support for the P2 ASCII variant or non-255 maxval) — this
// project only ever reads files its own generator wrote.
// ---------------------------------------------------------------------------
static bool read_pgm(const std::string& path, int& W, int& H, std::vector<unsigned char>& data)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return false;

    std::string magic;
    in >> magic;
    if (magic != "P5") return false;

    // read_int: skip whitespace and '#'-comment lines (the PGM spec allows
    // both between header tokens), then read one integer.
    auto read_int = [&](int& out) -> bool {
        for (;;) {
            const int c = in.peek();
            if (c == '#') { std::string line; std::getline(in, line); continue; }
            if (c != EOF && std::isspace(c)) { in.get(); continue; }
            break;
        }
        in >> out;
        return static_cast<bool>(in);
    };

    int maxval = 0;
    if (!read_int(W) || !read_int(H) || !read_int(maxval)) return false;
    if (maxval != 255 || W <= 0 || H <= 0) return false;
    in.get();   // consume the single mandatory whitespace byte after maxval (PGM spec)

    data.resize(static_cast<size_t>(W) * static_cast<size_t>(H));
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
    // Success iff we got EXACTLY the requested byte count — the robust way
    // to check (stream failbit/eofbit interaction around exact-length reads
    // is implementation-nuanced; gcount() is unambiguous).
    return in.gcount() == static_cast<std::streamsize>(data.size());
}

// ---------------------------------------------------------------------------
// Sample loading — all four committed PGMs, dimension-checked against each
// other (a strict loader per repo convention: any mismatch aborts rather
// than silently truncating or padding).
// ---------------------------------------------------------------------------
struct StereoSample {
    int W = 0, H = 0;
    std::vector<unsigned char> left, right, gt_disp, gt_valid;
    bool loaded = false;
};

static StereoSample load_sample(const std::string& dir)
{
    StereoSample s;
    int w2, h2, w3, h3, w4, h4;
    if (!read_pgm(dir + "/left.pgm", s.W, s.H, s.left)) { std::fprintf(stderr, "sample: failed to read left.pgm\n"); return StereoSample{}; }
    if (!read_pgm(dir + "/right.pgm", w2, h2, s.right)) { std::fprintf(stderr, "sample: failed to read right.pgm\n"); return StereoSample{}; }
    if (!read_pgm(dir + "/gt_disparity.pgm", w3, h3, s.gt_disp)) { std::fprintf(stderr, "sample: failed to read gt_disparity.pgm\n"); return StereoSample{}; }
    if (!read_pgm(dir + "/gt_valid.pgm", w4, h4, s.gt_valid)) { std::fprintf(stderr, "sample: failed to read gt_valid.pgm\n"); return StereoSample{}; }
    if (w2 != s.W || h2 != s.H || w3 != s.W || h3 != s.H || w4 != s.W || h4 != s.H) {
        std::fprintf(stderr, "sample: dimension mismatch across left/right/gt_disparity/gt_valid\n");
        return StereoSample{};
    }
    if (s.W <= 2 * kCensusHalf + kMaxDisp || s.H <= 2 * kCensusHalf) {
        std::fprintf(stderr, "sample: %dx%d too small for a %d-px census margin and D=%d\n",
                     s.W, s.H, kCensusHalf, kMaxDisp);
        return StereoSample{};
    }
    s.loaded = true;
    return s;
}

// Path helpers (same exe-relative resolution as 07.09/08.01: the exe sits
// at build/x64/<Config>/, three levels below the project root).
static std::string project_root_from(const char* argv0)
{
    std::string exe(argv0 ? argv0 : "");
    size_t cut = exe.find_last_of("/\\");
    if (cut == std::string::npos) return ".";
    return exe.substr(0, cut) + "/../../..";
}

static std::string find_data_dir(const std::string& cli_dir, const char* argv0)
{
    std::vector<std::string> candidates;
    if (!cli_dir.empty()) candidates.push_back(cli_dir);
    candidates.push_back(project_root_from(argv0) + "/data/sample");
    candidates.push_back("data/sample");
    candidates.push_back("../data/sample");
    for (const auto& c : candidates)
        if (std::ifstream(c + "/left.pgm").is_open()) return c;
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

static bool write_pgm(const std::string& path, int W, int H, const std::vector<unsigned char>& gray)
{
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) return false;
    out << "P5\n" << W << " " << H << "\n255\n";
    out.write(reinterpret_cast<const char*>(gray.data()), static_cast<std::streamsize>(gray.size()));
    return static_cast<bool>(out);
}

// ---------------------------------------------------------------------------
// exact_match — count element-wise mismatches between a GPU result
// (downloaded to host) and the CPU oracle. Used by every VERIFY checkpoint
// below; ALL comparisons in this project are exact-equality (kernels.cuh /
// reference_cpu.cpp explain why: every operation is integer arithmetic).
// ---------------------------------------------------------------------------
template <typename T>
static long long exact_match(const std::vector<T>& gpu, const std::vector<T>& cpu)
{
    long long mism = 0;
    const size_t n = gpu.size();
    for (size_t i = 0; i < n; ++i) if (gpu[i] != cpu[i]) ++mism;
    return mism;
}

// ---------------------------------------------------------------------------
// good_pixel_rate — the ground-truth gate metric: over pixels the SCENE
// says are genuinely visible in both views (gt_valid == 255), what fraction
// does this algorithm get right within kDispTolerance? disp is compared in
// the SAME *4 scaled domain gt_disp already uses (both sides are exact
// integers, so no rounding subtlety — see kGtDispScale above).
// ---------------------------------------------------------------------------
static double good_pixel_rate(const std::vector<unsigned char>& disp,
                              const StereoSample& sample, long long* n_valid_out)
{
    long long n_valid = 0, n_good = 0;
    const size_t n = disp.size();
    for (size_t i = 0; i < n; ++i) {
        if (sample.gt_valid[i] != 255) continue;
        ++n_valid;
        if (disp[i] == kInvalidDisp) continue;
        const int d_scaled = static_cast<int>(disp[i]) * kGtDispScale;
        const int diff = d_scaled - static_cast<int>(sample.gt_disp[i]);
        const int adiff = (diff < 0) ? -diff : diff;
        if (adiff <= kDispTolerance * kGtDispScale) ++n_good;
    }
    if (n_valid_out) *n_valid_out = n_valid;
    return n_valid > 0 ? (100.0 * static_cast<double>(n_good) / static_cast<double>(n_valid)) : 0.0;
}

// ---------------------------------------------------------------------------
// disparity_to_pgm — visualize a disparity map: valid pixels -> disp*4
// (matching gt_disparity.pgm's own scale, so the two are visually and
// numerically comparable side by side); kInvalidDisp -> 0 (black), a
// visually unambiguous "no answer" that can never collide with a real
// scaled disparity (the smallest real value, 0*4=0, only occurs at
// disparity exactly 0 — an acceptable, documented ambiguity for a debug
// visualization, called out in demo/README.md).
// ---------------------------------------------------------------------------
static std::vector<unsigned char> disparity_to_pgm(const std::vector<unsigned char>& disp)
{
    std::vector<unsigned char> out(disp.size());
    for (size_t i = 0; i < disp.size(); ++i) {
        out[i] = (disp[i] == kInvalidDisp) ? 0
               : static_cast<unsigned char>(std::min(255, static_cast<int>(disp[i]) * kGtDispScale));
    }
    return out;
}

// ---------------------------------------------------------------------------
// error_map_pgm — per-pixel correctness against ground truth, for the demo
// artifact (the visual story a percentage cannot tell): 255 (white) = good
// match on a GT-scored pixel; 80 (dark gray) = wrong or no-answer on a
// GT-scored pixel; 0 (black) = not scored (occluded/border in ground truth).
// ---------------------------------------------------------------------------
static std::vector<unsigned char> error_map_pgm(const std::vector<unsigned char>& disp,
                                                 const StereoSample& sample)
{
    std::vector<unsigned char> out(disp.size(), 0);
    for (size_t i = 0; i < disp.size(); ++i) {
        if (sample.gt_valid[i] != 255) { out[i] = 0; continue; }
        if (disp[i] == kInvalidDisp) { out[i] = 80; continue; }
        const int d_scaled = static_cast<int>(disp[i]) * kGtDispScale;
        const int diff = d_scaled - static_cast<int>(sample.gt_disp[i]);
        const int adiff = (diff < 0) ? -diff : diff;
        out[i] = (adiff <= kDispTolerance * kGtDispScale) ? 255 : 80;
    }
    return out;
}

// ---------------------------------------------------------------------------
// main.
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    std::string data_dir;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--data") && i + 1 < argc) data_dir = argv[++i];
        else {
            std::fprintf(stderr, "usage: %s [--data path/to/data/sample]\n", argv[0]);
            return 2;
        }
    }

    std::printf("[demo] stereo depth: census + Hamming cost volume, block matching vs Semi-Global Matching (project 01.02)\n");
    print_device_info();
    std::printf("PROBLEM: stereo matching, D=%d disparities, %dx%d census window (%d bits), P1=%d P2=%d, 4-path SGM\n",
               kMaxDisp, 2 * kCensusHalf + 1, 2 * kCensusHalf + 1, kCensusBits, kP1, kP2);

    // ---- data ----------------------------------------------------------------
    const std::string dir = find_data_dir(data_dir, argv[0]);
    if (dir.empty()) {
        std::printf("DATA: NOT FOUND — data/sample/left.pgm missing (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (sample data missing)\n");
        return 1;
    }
    std::printf("[info] data dir: %s\n", dir.c_str());
    StereoSample sample = load_sample(dir);
    if (!sample.loaded) {
        std::printf("DATA: MALFORMED — see stderr\n");
        std::printf("RESULT: FAIL (sample data malformed)\n");
        return 1;
    }
    const int W = sample.W, H = sample.H;
    const size_t pix_n = static_cast<size_t>(W) * H;
    const size_t vol_n = static_cast<size_t>(kMaxDisp) * pix_n;
    std::printf("DATA: %dx%d rectified synthetic stereo pair, dense ground truth [synthetic, seed 42]\n", W, H);

    // ======================= device buffers, shared stages ====================
    unsigned char *d_left = nullptr, *d_right = nullptr;
    unsigned long long *d_census_l = nullptr, *d_census_r = nullptr;
    unsigned char* d_cost = nullptr;
    CUDA_CHECK(cudaMalloc(&d_left, pix_n));
    CUDA_CHECK(cudaMalloc(&d_right, pix_n));
    CUDA_CHECK(cudaMalloc(&d_census_l, pix_n * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_census_r, pix_n * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_cost, vol_n));
    CUDA_CHECK(cudaMemcpy(d_left, sample.left.data(), pix_n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_right, sample.right.data(), pix_n, cudaMemcpyHostToDevice));

    GpuTimer gt_shared; gt_shared.begin();
    launch_census(d_left, d_census_l, W, H);
    launch_census(d_right, d_census_r, W, H);
    launch_cost_volume(d_census_l, d_census_r, d_cost, W, H);
    const float shared_ms = gt_shared.end_ms();

    // ======================= GPU: BLOCK MATCHING ===============================
    unsigned char *d_bm_l = nullptr, *d_bm_r = nullptr, *d_disp_bm = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bm_l, pix_n));
    CUDA_CHECK(cudaMalloc(&d_bm_r, pix_n));
    CUDA_CHECK(cudaMalloc(&d_disp_bm, pix_n));

    GpuTimer gt_bm; gt_bm.begin();
    launch_wta_bm(d_cost, d_bm_l, W, H);
    launch_wta_bm_right(d_cost, d_bm_r, W, H);
    launch_lr_check(d_bm_l, d_bm_r, d_disp_bm, W, H, kLrCheckTolerance);
    const float bm_ms = gt_bm.end_ms();

    std::vector<unsigned char> disp_bm_gpu(pix_n);
    CUDA_CHECK(cudaMemcpy(disp_bm_gpu.data(), d_disp_bm, pix_n, cudaMemcpyDeviceToHost));

    // ======================= GPU: SEMI-GLOBAL MATCHING ==========================
    int* d_lsum = nullptr;
    CUDA_CHECK(cudaMalloc(&d_lsum, vol_n * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_lsum, 0, vol_n * sizeof(int)));

    GpuTimer gt_sgm; gt_sgm.begin();
    launch_sgm_path(d_cost, d_lsum, W, H, kP1, kP2, +1, 0);   // L->R
    launch_sgm_path(d_cost, d_lsum, W, H, kP1, kP2, -1, 0);   // R->L
    launch_sgm_path(d_cost, d_lsum, W, H, kP1, kP2, 0, +1);   // T->B
    launch_sgm_path(d_cost, d_lsum, W, H, kP1, kP2, 0, -1);   // B->T

    unsigned char *d_sgm_l = nullptr, *d_sgm_r = nullptr, *d_sgm_lr = nullptr, *d_disp_sgm = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sgm_l, pix_n));
    CUDA_CHECK(cudaMalloc(&d_sgm_r, pix_n));
    CUDA_CHECK(cudaMalloc(&d_sgm_lr, pix_n));
    CUDA_CHECK(cudaMalloc(&d_disp_sgm, pix_n));
    launch_wta_sgm(d_lsum, d_sgm_l, W, H);
    launch_wta_sgm_right(d_lsum, d_sgm_r, W, H);
    launch_lr_check(d_sgm_l, d_sgm_r, d_sgm_lr, W, H, kLrCheckTolerance);
    launch_median3(d_sgm_lr, d_disp_sgm, W, H);
    const float sgm_ms = gt_sgm.end_ms();

    std::vector<unsigned char> disp_sgm_gpu(pix_n);
    CUDA_CHECK(cudaMemcpy(disp_sgm_gpu.data(), d_disp_sgm, pix_n, cudaMemcpyDeviceToHost));

    std::printf("[time] GPU kernels: census+cost %.3f ms | BM (wta+lr) %.3f ms | SGM (4-path+wta+lr+median) %.3f ms\n",
               static_cast<double>(shared_ms), static_cast<double>(bm_ms), static_cast<double>(sgm_ms));

    // ======================= VERIFY STAGE (CPU oracle, exact) ===================
    // Every checkpoint below is EXACT equality — kernels.cuh explains why:
    // census/Hamming/the SGM recurrence are all integer arithmetic, so there
    // is no rounding for a tolerance to paper over.
    bool verify_pass = true;
    {
        CpuTimer ct; ct.begin();

        // ---- census + cost volume ---------------------------------------------
        std::vector<unsigned long long> census_l_gpu(pix_n), census_r_gpu(pix_n);
        CUDA_CHECK(cudaMemcpy(census_l_gpu.data(), d_census_l, pix_n * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(census_r_gpu.data(), d_census_r, pix_n * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        std::vector<unsigned char> cost_gpu(vol_n);
        CUDA_CHECK(cudaMemcpy(cost_gpu.data(), d_cost, vol_n, cudaMemcpyDeviceToHost));

        std::vector<unsigned long long> census_l_cpu(pix_n), census_r_cpu(pix_n);
        census_cpu(sample.left.data(), census_l_cpu.data(), W, H);
        census_cpu(sample.right.data(), census_r_cpu.data(), W, H);
        std::vector<unsigned char> cost_cpu(vol_n);
        cost_volume_cpu(census_l_cpu.data(), census_r_cpu.data(), cost_cpu.data(), W, H);

        const long long mism_census = exact_match(census_l_gpu, census_l_cpu) + exact_match(census_r_gpu, census_r_cpu);
        const long long mism_cost = exact_match(cost_gpu, cost_cpu);
        std::printf("[info] verify(census): %lld mismatches over %zu signatures (both images)\n", mism_census, pix_n * 2);
        std::printf("[info] verify(cost volume): %lld mismatches over %zu entries\n", mism_cost, vol_n);
        if (mism_census != 0 || mism_cost != 0) verify_pass = false;

        // ---- one aggregation path (L->R, dx=+1,dy=0) ---------------------------
        int* d_lsum_check = nullptr;
        CUDA_CHECK(cudaMalloc(&d_lsum_check, vol_n * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_lsum_check, 0, vol_n * sizeof(int)));
        launch_sgm_path(d_cost, d_lsum_check, W, H, kP1, kP2, +1, 0);
        std::vector<int> lsum_gpu(vol_n);
        CUDA_CHECK(cudaMemcpy(lsum_gpu.data(), d_lsum_check, vol_n * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_lsum_check));

        std::vector<int> lsum_cpu_check(vol_n, 0);
        sgm_path_cpu(cost_cpu.data(), lsum_cpu_check.data(), W, H, kP1, kP2, +1, 0);
        const long long mism_path = exact_match(lsum_gpu, lsum_cpu_check);
        std::printf("[info] verify(one SGM path, L->R): %lld mismatches over %zu entries\n", mism_path, vol_n);
        if (mism_path != 0) verify_pass = false;

        // ---- final disparity, both pipelines ------------------------------------
        // The CPU oracle re-derives everything from raw pixels (its OWN
        // census + cost + aggregation), not from the GPU's intermediate
        // arrays above — a truly independent second implementation, per
        // CLAUDE.md §5's "the oracle must be able to catch anything".
        std::vector<unsigned char> bm_l_cpu(pix_n), bm_r_cpu(pix_n), disp_bm_cpu(pix_n);
        wta_bm_cpu(cost_cpu.data(), bm_l_cpu.data(), W, H);
        wta_bm_right_cpu(cost_cpu.data(), bm_r_cpu.data(), W, H);
        lr_check_cpu(bm_l_cpu.data(), bm_r_cpu.data(), disp_bm_cpu.data(), W, H, kLrCheckTolerance);

        std::vector<int> lsum_cpu(vol_n, 0);
        sgm_path_cpu(cost_cpu.data(), lsum_cpu.data(), W, H, kP1, kP2, +1, 0);
        sgm_path_cpu(cost_cpu.data(), lsum_cpu.data(), W, H, kP1, kP2, -1, 0);
        sgm_path_cpu(cost_cpu.data(), lsum_cpu.data(), W, H, kP1, kP2, 0, +1);
        sgm_path_cpu(cost_cpu.data(), lsum_cpu.data(), W, H, kP1, kP2, 0, -1);
        std::vector<unsigned char> sgm_l_cpu(pix_n), sgm_r_cpu(pix_n), sgm_lr_cpu(pix_n), disp_sgm_cpu(pix_n);
        wta_sgm_cpu(lsum_cpu.data(), sgm_l_cpu.data(), W, H);
        wta_sgm_right_cpu(lsum_cpu.data(), sgm_r_cpu.data(), W, H);
        lr_check_cpu(sgm_l_cpu.data(), sgm_r_cpu.data(), sgm_lr_cpu.data(), W, H, kLrCheckTolerance);
        median3_cpu(sgm_lr_cpu.data(), disp_sgm_cpu.data(), W, H);

        const double cpu_ms = ct.end_ms();

        const long long mism_bm = exact_match(disp_bm_gpu, disp_bm_cpu);
        const long long mism_sgm = exact_match(disp_sgm_gpu, disp_sgm_cpu);
        std::printf("[info] verify(final disparity, BM):  %lld mismatches over %zu pixels\n", mism_bm, pix_n);
        std::printf("[info] verify(final disparity, SGM): %lld mismatches over %zu pixels\n", mism_sgm, pix_n);
        std::printf("[time] full CPU oracle (census+cost+BM+SGM, all checkpoints): %.1f ms\n", cpu_ms);
        if (mism_bm != 0 || mism_sgm != 0) verify_pass = false;
    }
    std::printf("VERIFY: %s (GPU matches CPU reference EXACTLY: census, cost volume, one SGM path, BM final disparity, SGM final disparity)\n",
               verify_pass ? "PASS" : "FAIL");
    if (!verify_pass) {
        std::printf("RESULT: FAIL (GPU/CPU disagreement — fix before trusting either disparity map)\n");
        return 1;
    }

    // ======================= GROUND-TRUTH GATE ==================================
    long long n_valid_bm = 0, n_valid_sgm = 0;
    const double rate_bm = good_pixel_rate(disp_bm_gpu, sample, &n_valid_bm);
    const double rate_sgm = good_pixel_rate(disp_sgm_gpu, sample, &n_valid_sgm);
    const double margin = rate_sgm - rate_bm;

    std::printf("BM: good-pixel rate (|d-gt|<=%d) = %.2f%% over %lld GT-valid pixels\n", kDispTolerance, rate_bm, n_valid_bm);
    std::printf("SGM: good-pixel rate (|d-gt|<=%d) = %.2f%% over %lld GT-valid pixels\n", kDispTolerance, rate_sgm, n_valid_sgm);
    std::printf("[info] SGM margin over BM: %.2f percentage points\n", margin);

    // ======================= ARTIFACTS ===========================================
    const std::string out_dir = project_root_from(argv[0]) + "/demo/out";
    bool artifact_ok = ensure_dir(out_dir);
    if (artifact_ok) {
        artifact_ok = write_pgm(out_dir + "/disparity_bm.pgm", W, H, disparity_to_pgm(disp_bm_gpu))
                    && write_pgm(out_dir + "/disparity_sgm.pgm", W, H, disparity_to_pgm(disp_sgm_gpu))
                    && write_pgm(out_dir + "/error_map.pgm", W, H, error_map_pgm(disp_sgm_gpu, sample));
    }
    if (artifact_ok)
        std::printf("ARTIFACT: wrote demo/out/disparity_bm.pgm, demo/out/disparity_sgm.pgm, demo/out/error_map.pgm\n");
    else
        std::printf("ARTIFACT: FAILED to write demo/out images\n");

    // ---- cleanup ----------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_left));   CUDA_CHECK(cudaFree(d_right));
    CUDA_CHECK(cudaFree(d_census_l)); CUDA_CHECK(cudaFree(d_census_r));
    CUDA_CHECK(cudaFree(d_cost));
    CUDA_CHECK(cudaFree(d_bm_l));  CUDA_CHECK(cudaFree(d_bm_r));  CUDA_CHECK(cudaFree(d_disp_bm));
    CUDA_CHECK(cudaFree(d_lsum));
    CUDA_CHECK(cudaFree(d_sgm_l)); CUDA_CHECK(cudaFree(d_sgm_r)); CUDA_CHECK(cudaFree(d_sgm_lr)); CUDA_CHECK(cudaFree(d_disp_sgm));

    // ---- verdict ------------------------------------------------------------------
    const bool gate_bm = rate_bm >= kMinGoodPixelRateBM;
    const bool gate_sgm = rate_sgm >= kMinGoodPixelRateSGM;
    const bool gate_margin = margin >= kMinSgmMarginOverBm;
    const bool success = artifact_ok && gate_bm && gate_sgm && gate_margin;

    if (success) {
        std::printf("RESULT: PASS (BM >= %.0f%%, SGM >= %.0f%%, SGM exceeds BM by >= %.0f points — SGM measurably fixes BM's streaks)\n",
                   kMinGoodPixelRateBM, kMinGoodPixelRateSGM, kMinSgmMarginOverBm);
    } else {
        std::printf("RESULT: FAIL (a ground-truth gate was not met — see BM:/SGM:/[info] lines above)\n");
    }
    return success ? 0 : 1;
}
