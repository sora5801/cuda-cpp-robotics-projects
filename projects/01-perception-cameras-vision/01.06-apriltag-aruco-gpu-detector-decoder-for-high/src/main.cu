// ===========================================================================
// main.cu — entry point for project 01.06
//           AprilTag / ArUco GPU detector-decoder for high-rate fiducial
//           localization
//
// What this program does, start to finish
// -----------------------------------------
//   1. Load the committed dictionary (32 codes + metadata) and three
//      committed synthetic scenes (scene_main.pgm: 6 tags, full perspective;
//      scene_distractor.pgm: no tags, adversarial; scene_robustness.pgm: 4
//      tags with deliberately corrupted payload bits) plus their ground
//      truth CSVs, all written by scripts/make_synthetic.py.
//   2. For EACH scene, run the full 6-stage pipeline TWICE: once on the GPU
//      (kernels.cu) and once on the CPU (reference_cpu.cpp) — pixel-parallel
//      stages 1-2 first, then the host-side candidate filter (small,
//      O(H*W) but sequential — same division of labor as 30.01/08.01), then
//      candidate-parallel stages 3-6.
//   3. VERIFY: compare every GPU intermediate against its CPU twin — local
//      mean (float), mask (exact), CCL labels (exact, per kernels.cu's
//      convergence argument), candidate stats (exact), refined corners
//      (float tolerance), homography (double tolerance), decode + pose
//      (tolerance) — see the kTol* constants below for the measured-and-
//      margined value behind every number.
//   4. FIVE INDEPENDENT GATES (none of which is just "VERIFY again" — see
//      kernels.cuh/reference_cpu.cpp's twin-independence ruling): detection
//      (all 6 main-scene tags found, correct ID, no extras), corner-accuracy
//      (max corner error vs the RENDERER's ground truth), decode-robustness
//      (bit flips AT capacity decode correctly, BEYOND capacity are
//      rejected — the dictionary's own negative control), false-positive
//      (the tag-free distractor scene yields zero accepted detections), and
//      pose (rotation/translation error vs the renderer's ground truth
//      camera pose — an ANALYTIC check that never routes through this
//      project's own DLT/CCL/threshold code, per the independence ruling).
//   5. ARTIFACTS: demo/out/{detections_overlay.ppm, decoded_grid_debug.ppm,
//      detections.csv, gates_metrics.csv}.
//
// Output contract (load-bearing!): stable lines are "[demo]", "PROBLEM:",
// "DATA:", "VERIFY:", every "GATE <name>:" line, "ARTIFACT:", and "RESULT:"
// — no embedded numbers, GPU-architecture independent. Measured numbers live
// on "[info]"/"[time]" lines, deliberately NOT diffed by demo/run_demo.*
// (this project's floating-point pipeline can differ by a few ULP across
// sm_75/86/89 — THEORY.md "Numerical considerations"). Change a stable line
// => update demo/expected_output.txt in the same change.
//
// Read this first, then kernels.cuh -> kernels.cu -> reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"
#include "util/paths.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// ===========================================================================
// Tolerances — every number below is either a numerical argument (documented
// inline; see THEORY.md "Numerical considerations" for the full derivation)
// or a floor/ceiling calibrated from an ACTUAL measured run on this
// project's committed sample (CLAUDE.md paragraph 8: "never fabricate" —
// see README "Expected output" for the measured values this project
// actually produced on the reference machine), with margin so the gate
// stays robust to legitimate cross-GPU float differences.
// ===========================================================================
static constexpr double kPiForDegrees = 3.14159265358979323846;  // MSVC's <cmath> does not define M_PI
                                                                  // without _USE_MATH_DEFINES; spelling our
                                                                  // own avoids depending on that compiler
                                                                  // quirk (used only by the pose gate's
                                                                  // rotation-error math below).
static constexpr double kTolLocalMean   = 0.02;   // local_mean: GPU vs CPU, 0..255 scale (box-sum is exact
                                                   // integer float32 arithmetic -- see kernels.cu's file
                                                   // header -- so this is generous headroom, not an
                                                   // expectation of real drift)
static constexpr int    kMaxMaskMismatch  = 8;    // mask: count of differing pixels out of W*H (should be 0;
                                                   // small floor for near-threshold-boundary FP order effects)
static constexpr int    kMaxLabelMismatch = 8;    // CCL label: count of differing pixels (see kernels.cu's
                                                   // convergence-argument comment: should also be 0)
static constexpr double kTolCentroidPx    = 0.01; // candidate centroid, px (integer sums / integer count => exact)
// kTolCornerPx / kTolHomography / kTolPoseR: all three are DOWNSTREAM of the
// same root cause, measured on this project's committed scene (README
// "Expected output" records the exact numbers) -- the corner-refinement
// radial search (kernels.cu's refine_one_corner) walks a FIXED number of
// sample steps and picks the LAST dark->light crossing; GPU (nvcc, permitted
// to contract multiply-add into FMA) and CPU (cl.exe, no contraction by
// default) can therefore land on a DIFFERENT step as "last" when a ray
// grazes a threshold boundary within about 0.15 px of a step edge -- rare,
// but when it happens the DLT solve (which has NO smoothing; it fits the 4
// corners EXACTLY) amplifies that sub-pixel input difference into a
// homography-entry difference of order 1, and the pose decomposition
// amplifies THAT further into a rotation-matrix difference of order 1e-2.
// This is the same "small input perturbation, exact-fit solver, amplified
// output" story as an ill-conditioned linear system -- expected, not a bug
// (THEORY.md "Numerical considerations" derives the amplification chain in
// full). The tolerances below have real margin over the measured maxima.
static constexpr double kTolCornerPx      = 0.30;  // refined quad corner, px
static constexpr double kTolHomography    = 5.0;   // homography h[] entries (mixed units/scales -- see above)
static constexpr int    kMaxHammingDrift  = 1;      // decode hamming_distance: GPU vs CPU may differ by <=1 bit
                                                    // if a sampled point sits within ULP noise of the local
                                                    // threshold -- correction capacity absorbs it either way
static constexpr double kTolPoseR         = 0.02;   // pose rotation matrix entries (dimensionless, unit vectors)
static constexpr double kTolPoseT         = 5e-3;   // pose translation, meters

// -- Gate tolerances (measured-and-margined; see README "Expected output"
// for the actual numbers this project's committed sample produced) --------
static constexpr double kCornerGateTolPx      = 3.5;    // corner-accuracy gate ceiling, px
// Pose-from-homography without IPPE refinement has REAL angular sensitivity
// to a few pixels of corner noise on a moderately-tilted, moderately-sized
// tag -- a well-known limitation of this project's chosen (teaching-scope)
// pose method, named honestly in README "Limitations & honesty" and
// THEORY.md "Where this sits in the real world" (IPPE is the production
// fix). The ceilings below carry real margin over the measured maxima
// (README "Expected output"), not zero margin.
static constexpr double kPoseRotGateTolDeg    = 13.0;   // pose gate rotation-error ceiling, degrees
static constexpr double kPoseTransGateTolPct  = 16.0;   // pose gate translation-error ceiling, % of tag size

// ===========================================================================
// Minimal strict PGM (P5) / PPM (P6) readers/writers — same discipline as
// 01.01's read_pnm: this project only ever reads files its own generator
// wrote, so a strict reader that ABORTS on anything unexpected is the right
// choice (a silent partial-read would corrupt every downstream stage).
// ===========================================================================
static bool read_pgm(const std::string& path, int& W, int& H, std::vector<unsigned char>& data)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return false;
    std::string magic;
    in >> magic;
    if (magic != "P5") return false;
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
    in.get();
    data.resize(static_cast<size_t>(W) * static_cast<size_t>(H));
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
    return in.gcount() == static_cast<std::streamsize>(data.size());
}
static bool write_ppm(const std::string& path, int W, int H, const std::vector<unsigned char>& rgb)
{
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) return false;
    out << "P6\n" << W << " " << H << "\n255\n";
    out.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    return static_cast<bool>(out);
}

// ===========================================================================
// Dictionary loader — reads data/sample/dictionary.bin, written by
// scripts/make_synthetic.py as: 5x int32 header (num_codes, bits_per_code,
// grid_n, min_distance, correction_capacity) followed by num_codes uint16
// codes, all little-endian (native on every machine this repo targets).
// ===========================================================================
struct DictionaryData {
    std::vector<uint16_t> codes;
    int bits_per_code = 0, grid_n = 0, min_distance = 0, correction_capacity = 0;
    bool loaded = false;
};
static DictionaryData load_dictionary(const std::string& path)
{
    DictionaryData d;
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return d;
    int32_t header[5];
    in.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!in) return d;
    const int num_codes = header[0];
    d.bits_per_code = header[1]; d.grid_n = header[2];
    d.min_distance = header[3]; d.correction_capacity = header[4];
    if (num_codes <= 0 || num_codes > 10000) return d;
    d.codes.resize(static_cast<size_t>(num_codes));
    in.read(reinterpret_cast<char*>(d.codes.data()), static_cast<std::streamsize>(num_codes * sizeof(uint16_t)));
    if (!in) return d;
    d.loaded = true;
    return d;
}

// ===========================================================================
// Ground-truth CSV readers — hand-rolled comma-split (these files are
// written by scripts/make_synthetic.py in one fixed, simple shape; a full
// RFC-4180 CSV parser would be solving a problem this project does not have).
// ===========================================================================
static std::vector<std::string> split_csv(const std::string& line)
{
    std::vector<std::string> out;
    std::string cur;
    for (char ch : line) {
        if (ch == ',') { out.push_back(cur); cur.clear(); }
        else cur.push_back(ch);
    }
    out.push_back(cur);
    return out;
}

struct MainGroundTruthTag {
    int tag_index = 0, dict_id = -1;
    float cx[4] = {}, cy[4] = {};
    float R[9] = {};
    float t[3] = {};
};
static bool load_main_ground_truth(const std::string& path, std::vector<MainGroundTruthTag>& out)
{
    std::ifstream in(path);
    if (!in.is_open()) return false;
    std::string line;
    if (!std::getline(in, line)) return false;   // header
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto f = split_csv(line);
        if (f.size() < 22) return false;
        MainGroundTruthTag g;
        g.tag_index = std::stoi(f[0]);
        g.dict_id   = std::stoi(f[1]);
        for (int k = 0; k < 4; ++k) { g.cx[k] = std::stof(f[2 + 2 * k]); g.cy[k] = std::stof(f[3 + 2 * k]); }
        for (int k = 0; k < 9; ++k) g.R[k] = std::stof(f[10 + k]);
        for (int k = 0; k < 3; ++k) g.t[k] = std::stof(f[19 + k]);
        out.push_back(g);
    }
    return true;
}

struct RobustnessGroundTruthTag {
    int tag_index = 0, true_dict_id = -1, num_flips = 0;
    bool expect_accept = false;
    float cx[4] = {}, cy[4] = {};
};
static bool load_robustness_ground_truth(const std::string& path, std::vector<RobustnessGroundTruthTag>& out)
{
    std::ifstream in(path);
    if (!in.is_open()) return false;
    std::string line;
    if (!std::getline(in, line)) return false;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto f = split_csv(line);
        if (f.size() < 12) return false;
        RobustnessGroundTruthTag g;
        g.tag_index     = std::stoi(f[0]);
        g.true_dict_id  = std::stoi(f[1]);
        g.num_flips     = std::stoi(f[2]);
        g.expect_accept = (f[3] == "accept");
        for (int k = 0; k < 4; ++k) { g.cx[k] = std::stof(f[4 + 2 * k]); g.cy[k] = std::stof(f[5 + 2 * k]); }
        out.push_back(g);
    }
    return true;
}

// ===========================================================================
// PipelineResult — everything one full run (GPU or CPU) of the pipeline on
// one scene produces, kept together so main() can VERIFY and gate on it.
// ===========================================================================
struct PipelineResult {
    std::vector<unsigned char> mask;
    std::vector<int> label;
    std::vector<float> local_mean;
    std::vector<CandidateComponent> candidates;
    std::vector<QuadCorners> quads;
    std::vector<Homography> homs;
    std::vector<Detection> dets;
    int ccl_sweeps = 0;
};

// ---------------------------------------------------------------------------
// filter_candidates_gpu_path — the HOST-side compaction scan for the GPU
// path: scans the dense [H*W] stat arrays the GPU's atomic scatter produced
// for canonical roots (mask[p] && label[p]==p) and applies the same filter
// CONSTANTS reference_cpu.cpp's build_candidates_cpu uses. This function is
// deliberately SEPARATE code from build_candidates_cpu (not shared) — see
// kernels.cuh's twin-independence discussion: sharing only the numeric
// threshold constants (data, not algorithm) keeps the two candidate lists
// an honest cross-check rather than a foregone conclusion.
// ---------------------------------------------------------------------------
static std::vector<CandidateComponent> filter_candidates_gpu_path(
    const std::vector<unsigned char>& mask, const std::vector<int>& label, int W, int H,
    const std::vector<int>& count, const std::vector<unsigned long long>& sum_x,
    const std::vector<unsigned long long>& sum_y,
    const std::vector<int>& min_x, const std::vector<int>& max_x,
    const std::vector<int>& min_y, const std::vector<int>& max_y,
    const std::vector<unsigned long long>& key_min_sum, const std::vector<unsigned long long>& key_max_sum,
    const std::vector<unsigned long long>& key_min_diff, const std::vector<unsigned long long>& key_max_diff)
{
    std::vector<CandidateComponent> out;
    for (int p = 0; p < W * H && static_cast<int>(out.size()) < kMaxCandidates; ++p) {
        if (!mask[p] || label[p] != p || count[p] <= 0) continue;
        const int pc = count[p];
        const int bw = max_x[p] - min_x[p] + 1, bh = max_y[p] - min_y[p] + 1;
        const float fill = static_cast<float>(pc) / static_cast<float>(bw * bh);
        if (pc < kMinComponentPixels || pc > kMaxComponentPixels) continue;
        if (fill < kMinFillRatio || fill > kMaxFillRatio) continue;
        if (bw < kMinBBoxSidePx || bw > kMaxBBoxSidePx) continue;
        if (bh < kMinBBoxSidePx || bh > kMaxBBoxSidePx) continue;

        CandidateComponent c{};
        c.label = p;
        c.pixel_count = pc;
        c.centroid_x = static_cast<float>(sum_x[p]) / static_cast<float>(pc);
        c.centroid_y = static_cast<float>(sum_y[p]) / static_cast<float>(pc);
        c.bbox_min_x = min_x[p]; c.bbox_max_x = max_x[p];
        c.bbox_min_y = min_y[p]; c.bbox_max_y = max_y[p];

        const int i0 = unpack_corner_index(key_min_sum[p]);
        const int i1 = unpack_corner_index(key_max_diff[p]);
        const int i2 = unpack_corner_index(key_max_sum[p]);
        const int i3 = unpack_corner_index(key_min_diff[p]);
        c.raw_corner_x[0] = static_cast<float>(i0 % W); c.raw_corner_y[0] = static_cast<float>(i0 / W);
        c.raw_corner_x[1] = static_cast<float>(i1 % W); c.raw_corner_y[1] = static_cast<float>(i1 / W);
        c.raw_corner_x[2] = static_cast<float>(i2 % W); c.raw_corner_y[2] = static_cast<float>(i2 / W);
        c.raw_corner_x[3] = static_cast<float>(i3 % W); c.raw_corner_y[3] = static_cast<float>(i3 / W);
        out.push_back(c);
    }
    return out;
}

// ---------------------------------------------------------------------------
// run_gpu_pipeline — the full 6-stage GPU pipeline on one WxH gray scene.
// pixel_ms / candidate_ms are OUT parameters: the two timed regions this
// project's whole "pixel-parallel vs candidate-parallel" lesson rests on
// (THEORY.md "The GPU mapping" reports the actual measured numbers).
// ---------------------------------------------------------------------------
static PipelineResult run_gpu_pipeline(const std::vector<unsigned char>& gray, int W, int H,
                                       const uint16_t* d_dict, int num_dict_codes, int capacity,
                                       double& pixel_ms, double& candidate_ms)
{
    PipelineResult r;
    const size_t N = static_cast<size_t>(W) * H;

    unsigned char* d_gray = nullptr; CUDA_CHECK(cudaMalloc(&d_gray, N));
    CUDA_CHECK(cudaMemcpy(d_gray, gray.data(), N, cudaMemcpyHostToDevice));
    float* d_row_sum = nullptr;    CUDA_CHECK(cudaMalloc(&d_row_sum, N * sizeof(float)));
    float* d_local_mean = nullptr; CUDA_CHECK(cudaMalloc(&d_local_mean, N * sizeof(float)));
    unsigned char* d_mask = nullptr; CUDA_CHECK(cudaMalloc(&d_mask, N));
    int* d_label = nullptr;        CUDA_CHECK(cudaMalloc(&d_label, N * sizeof(int)));
    int* d_changed = nullptr;      CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));

    int* d_count = nullptr;      CUDA_CHECK(cudaMalloc(&d_count, N * sizeof(int)));
    unsigned long long *d_sum_x = nullptr, *d_sum_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sum_x, N * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_sum_y, N * sizeof(unsigned long long)));
    int *d_min_x = nullptr, *d_max_x = nullptr, *d_min_y = nullptr, *d_max_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_min_x, N * sizeof(int))); CUDA_CHECK(cudaMalloc(&d_max_x, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_min_y, N * sizeof(int))); CUDA_CHECK(cudaMalloc(&d_max_y, N * sizeof(int)));
    unsigned long long *d_key_min_sum = nullptr, *d_key_max_sum = nullptr, *d_key_min_diff = nullptr, *d_key_max_diff = nullptr;
    CUDA_CHECK(cudaMalloc(&d_key_min_sum, N * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_key_max_sum, N * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_key_min_diff, N * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_key_max_diff, N * sizeof(unsigned long long)));

    GpuTimer gt_pixel; gt_pixel.begin();
    launch_box_sum_h(d_gray, d_row_sum, W, H);
    launch_box_sum_v(d_row_sum, d_local_mean, W, H);
    launch_adaptive_threshold(d_gray, d_local_mean, d_mask, W, H);
    launch_ccl_init(d_mask, d_label, W, H);
    int sweeps = 0;
    for (;;) {
        const int zero = 0;
        CUDA_CHECK(cudaMemcpy(d_changed, &zero, sizeof(int), cudaMemcpyHostToDevice));
        launch_ccl_propagate_sweep(d_mask, d_label, W, H, d_changed);
        int changed_host = 0;
        CUDA_CHECK(cudaMemcpy(&changed_host, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        ++sweeps;
        if (!changed_host || sweeps >= kMaxCclSweeps) break;
    }
    launch_component_stats_init(d_count, d_sum_x, d_sum_y, d_min_x, d_max_x, d_min_y, d_max_y,
                                d_key_min_sum, d_key_max_sum, d_key_min_diff, d_key_max_diff, W, H);
    launch_component_stats_accumulate(d_mask, d_label, d_count, d_sum_x, d_sum_y, d_min_x, d_max_x, d_min_y, d_max_y,
                                      d_key_min_sum, d_key_max_sum, d_key_min_diff, d_key_max_diff, W, H);
    pixel_ms = static_cast<double>(gt_pixel.end_ms());
    r.ccl_sweeps = sweeps;

    r.mask.resize(N); r.label.resize(N); r.local_mean.resize(N);
    CUDA_CHECK(cudaMemcpy(r.mask.data(), d_mask, N, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(r.label.data(), d_label, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(r.local_mean.data(), d_local_mean, N * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<int> h_count(N); std::vector<unsigned long long> h_sum_x(N), h_sum_y(N);
    std::vector<int> h_min_x(N), h_max_x(N), h_min_y(N), h_max_y(N);
    std::vector<unsigned long long> h_kms(N), h_kxs(N), h_kmd(N), h_kxd(N);
    CUDA_CHECK(cudaMemcpy(h_count.data(), d_count, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sum_x.data(), d_sum_x, N * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sum_y.data(), d_sum_y, N * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_min_x.data(), d_min_x, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_max_x.data(), d_max_x, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_min_y.data(), d_min_y, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_max_y.data(), d_max_y, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_kms.data(), d_key_min_sum, N * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_kxs.data(), d_key_max_sum, N * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_kmd.data(), d_key_min_diff, N * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_kxd.data(), d_key_max_diff, N * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    r.candidates = filter_candidates_gpu_path(r.mask, r.label, W, H, h_count, h_sum_x, h_sum_y,
                                              h_min_x, h_max_x, h_min_y, h_max_y, h_kms, h_kxs, h_kmd, h_kxd);
    const int n = static_cast<int>(r.candidates.size());

    CandidateComponent* d_cands = nullptr;
    QuadCorners* d_quads = nullptr;
    Homography* d_homs = nullptr;
    Detection* d_dets = nullptr;
    if (n > 0) {
        CUDA_CHECK(cudaMalloc(&d_cands, static_cast<size_t>(n) * sizeof(CandidateComponent)));
        CUDA_CHECK(cudaMemcpy(d_cands, r.candidates.data(), static_cast<size_t>(n) * sizeof(CandidateComponent), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_quads, static_cast<size_t>(n) * sizeof(QuadCorners)));
        CUDA_CHECK(cudaMalloc(&d_homs, static_cast<size_t>(n) * sizeof(Homography)));
        CUDA_CHECK(cudaMalloc(&d_dets, static_cast<size_t>(n) * sizeof(Detection)));

        GpuTimer gt_cand; gt_cand.begin();
        launch_corner_refine(d_cands, n, d_gray, d_local_mean, W, H, d_quads);
        launch_homography_solve(d_quads, n, d_homs);
        launch_grid_decode(d_homs, n, d_gray, d_local_mean, W, H, d_dict, num_dict_codes, capacity, d_dets);
        launch_pose_from_homography(d_homs, n, d_dets);
        candidate_ms = static_cast<double>(gt_cand.end_ms());

        r.quads.resize(n); r.homs.resize(n); r.dets.resize(n);
        CUDA_CHECK(cudaMemcpy(r.quads.data(), d_quads, static_cast<size_t>(n) * sizeof(QuadCorners), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(r.homs.data(), d_homs, static_cast<size_t>(n) * sizeof(Homography), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(r.dets.data(), d_dets, static_cast<size_t>(n) * sizeof(Detection), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_cands)); CUDA_CHECK(cudaFree(d_quads));
        CUDA_CHECK(cudaFree(d_homs));  CUDA_CHECK(cudaFree(d_dets));
    } else {
        candidate_ms = 0.0;
    }

    CUDA_CHECK(cudaFree(d_gray)); CUDA_CHECK(cudaFree(d_row_sum)); CUDA_CHECK(cudaFree(d_local_mean));
    CUDA_CHECK(cudaFree(d_mask)); CUDA_CHECK(cudaFree(d_label));   CUDA_CHECK(cudaFree(d_changed));
    CUDA_CHECK(cudaFree(d_count)); CUDA_CHECK(cudaFree(d_sum_x));  CUDA_CHECK(cudaFree(d_sum_y));
    CUDA_CHECK(cudaFree(d_min_x)); CUDA_CHECK(cudaFree(d_max_x));  CUDA_CHECK(cudaFree(d_min_y)); CUDA_CHECK(cudaFree(d_max_y));
    CUDA_CHECK(cudaFree(d_key_min_sum)); CUDA_CHECK(cudaFree(d_key_max_sum));
    CUDA_CHECK(cudaFree(d_key_min_diff)); CUDA_CHECK(cudaFree(d_key_max_diff));
    return r;
}

// ---------------------------------------------------------------------------
// run_cpu_pipeline — the full 6-stage CPU oracle, calling ONLY the
// independently-typed reference_cpu.cpp functions.
// ---------------------------------------------------------------------------
static PipelineResult run_cpu_pipeline(const std::vector<unsigned char>& gray, int W, int H,
                                       const std::vector<uint16_t>& dictionary, int capacity,
                                       double& cpu_ms)
{
    PipelineResult r;
    const size_t N = static_cast<size_t>(W) * H;
    CpuTimer t; t.begin();

    std::vector<float> row_sum(N);
    r.local_mean.resize(N);
    r.mask.resize(N);
    r.label.resize(N);
    box_sum_h_cpu(gray.data(), row_sum.data(), W, H);
    box_sum_v_cpu(row_sum.data(), r.local_mean.data(), W, H);
    adaptive_threshold_cpu(gray.data(), r.local_mean.data(), r.mask.data(), W, H);
    ccl_union_find_cpu(r.mask.data(), r.label.data(), W, H);

    std::vector<CandidateComponent> cand_buf(kMaxCandidates);
    const int n = build_candidates_cpu(r.mask.data(), r.label.data(), W, H, cand_buf.data());
    r.candidates.assign(cand_buf.begin(), cand_buf.begin() + n);

    r.quads.resize(n); r.homs.resize(n); r.dets.resize(n);
    for (int i = 0; i < n; ++i) {
        r.quads[i] = corner_refine_one_cpu(r.candidates[i], gray.data(), r.local_mean.data(), W, H);
        r.homs[i] = homography_solve_one_cpu(r.quads[i]);
        r.dets[i] = grid_decode_one_cpu(i, r.homs[i], gray.data(), r.local_mean.data(), W, H,
                                        dictionary.data(), static_cast<int>(dictionary.size()), capacity);
        pose_from_homography_one_cpu(r.homs[i], r.dets[i]);
    }
    cpu_ms = t.end_ms();
    return r;
}

// ===========================================================================
// Comparison helpers for VERIFY.
// ===========================================================================
static double max_abs_diff_f(const std::vector<float>& a, const std::vector<float>& b)
{
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, static_cast<double>(std::fabs(a[i] - b[i])));
    return m;
}
static int count_mismatch_u8(const std::vector<unsigned char>& a, const std::vector<unsigned char>& b)
{
    int n = 0;
    for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) ++n;
    return n;
}
static int count_mismatch_i(const std::vector<int>& a, const std::vector<int>& b)
{
    int n = 0;
    for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) ++n;
    return n;
}
static double max_abs_diff_h(const Homography& a, const Homography& b)
{
    double m = 0.0;
    for (int k = 0; k < 9; ++k) m = std::max(m, std::fabs(a.h[k] - b.h[k]));
    return m;
}

// ===========================================================================
// verify_scene — runs the seven per-stage VERIFY checks for one scene's GPU
// vs CPU PipelineResult, printing one "[info] verify(...)" line per check
// and returning overall pass/fail. `label` names the scene in the printed
// lines.
// ===========================================================================
static bool verify_scene(const char* label, const PipelineResult& gpu, const PipelineResult& cpu)
{
    bool pass = true;

    const double d_mean = max_abs_diff_f(gpu.local_mean, cpu.local_mean);
    std::printf("[info] verify(%s.local_mean): max|gpu-cpu| = %.6f (tol %.2f)\n", label, d_mean, kTolLocalMean);
    if (d_mean > kTolLocalMean) pass = false;

    const int d_mask = count_mismatch_u8(gpu.mask, cpu.mask);
    std::printf("[info] verify(%s.mask): mismatched pixels = %d / %zu (tol <= %d)\n",
               label, d_mask, gpu.mask.size(), kMaxMaskMismatch);
    if (d_mask > kMaxMaskMismatch) pass = false;

    const int d_label = count_mismatch_i(gpu.label, cpu.label);
    std::printf("[info] verify(%s.ccl_label): mismatched pixels = %d / %zu (tol <= %d)\n",
               label, d_label, gpu.label.size(), kMaxLabelMismatch);
    if (d_label > kMaxLabelMismatch) pass = false;

    const bool count_ok = (gpu.candidates.size() == cpu.candidates.size());
    std::printf("[info] verify(%s.candidate_count): gpu = %zu, cpu = %zu\n",
               label, gpu.candidates.size(), cpu.candidates.size());
    if (!count_ok) { pass = false; return pass; }   // index-wise comparisons below need equal length

    const int n = static_cast<int>(gpu.candidates.size());
    double max_centroid_d = 0.0, max_corner_d = 0.0, max_hom_d = 0.0, max_pose_r_d = 0.0, max_pose_t_d = 0.0;
    int max_hamming_drift = 0;
    int tag_id_mismatches = 0, accepted_mismatches = 0;
    for (int i = 0; i < n; ++i) {
        const auto& gc = gpu.candidates[i]; const auto& cc = cpu.candidates[i];
        max_centroid_d = std::max({ max_centroid_d,
                                   std::fabs(static_cast<double>(gc.centroid_x) - cc.centroid_x),
                                   std::fabs(static_cast<double>(gc.centroid_y) - cc.centroid_y) });
        const auto& gq = gpu.quads[i]; const auto& cq = cpu.quads[i];
        for (int k = 0; k < 4; ++k) {
            max_corner_d = std::max({ max_corner_d,
                                     std::fabs(static_cast<double>(gq.x[k]) - cq.x[k]),
                                     std::fabs(static_cast<double>(gq.y[k]) - cq.y[k]) });
        }
        max_hom_d = std::max(max_hom_d, max_abs_diff_h(gpu.homs[i], cpu.homs[i]));

        const auto& gd = gpu.dets[i]; const auto& cd = cpu.dets[i];
        if (gd.tag_id != cd.tag_id) ++tag_id_mismatches;
        if (gd.accepted != cd.accepted) ++accepted_mismatches;
        max_hamming_drift = std::max(max_hamming_drift, std::abs(gd.hamming_distance - cd.hamming_distance));
        if (gd.pose_valid && cd.pose_valid) {
            for (int k = 0; k < 9; ++k) max_pose_r_d = std::max(max_pose_r_d, std::fabs(static_cast<double>(gd.R[k]) - cd.R[k]));
            for (int k = 0; k < 3; ++k) max_pose_t_d = std::max(max_pose_t_d, std::fabs(static_cast<double>(gd.t[k]) - cd.t[k]));
        }
    }
    std::printf("[info] verify(%s.centroid): max|gpu-cpu| = %.6f px (tol %.3f)\n", label, max_centroid_d, kTolCentroidPx);
    if (max_centroid_d > kTolCentroidPx) pass = false;
    std::printf("[info] verify(%s.corner): max|gpu-cpu| = %.6f px (tol %.3f)\n", label, max_corner_d, kTolCornerPx);
    if (max_corner_d > kTolCornerPx) pass = false;
    std::printf("[info] verify(%s.homography): max|gpu-cpu| = %.8f (tol %.6f)\n", label, max_hom_d, kTolHomography);
    if (max_hom_d > kTolHomography) pass = false;
    std::printf("[info] verify(%s.decode): tag_id mismatches = %d, accepted mismatches = %d, "
               "max hamming drift = %d (tol <= %d)\n",
               label, tag_id_mismatches, accepted_mismatches, max_hamming_drift, kMaxHammingDrift);
    if (tag_id_mismatches > 0 || accepted_mismatches > 0 || max_hamming_drift > kMaxHammingDrift) pass = false;
    std::printf("[info] verify(%s.pose): max|gpu-cpu| R = %.6f (tol %.4f), t = %.6f m (tol %.4f)\n",
               label, max_pose_r_d, kTolPoseR, max_pose_t_d, kTolPoseT);
    if (max_pose_r_d > kTolPoseR || max_pose_t_d > kTolPoseT) pass = false;

    return pass;
}

// ===========================================================================
// best_shift_corner_error — the corner-accuracy gate needs to compare a
// DETECTED quad (corner order fixed by this project's arbitrary-but-
// consistent extreme-corner assignment, kernels.cuh's file header) against
// GROUND TRUTH corners (fixed canonical tag-model order). The two orders can
// differ by an UNKNOWN cyclic shift of 0-3 quarter turns (which physical
// corner the detector happened to call "corner 0"). Rather than hand-derive
// the sign/direction of that shift from the rotation bookkeeping (risking a
// convention bug), this function tries all 4 cyclic shifts of the ground
// truth order and reports the shift that MINIMIZES the max corner distance
// — an honest, robust alignment that does not assume a convention.
// ---------------------------------------------------------------------------
static double best_shift_corner_error(const float det_x[4], const float det_y[4],
                                      const float gt_x[4], const float gt_y[4], int* best_shift_out)
{
    double best_err = 1e30;
    int best_shift = 0;
    for (int shift = 0; shift < 4; ++shift) {
        double err = 0.0;
        for (int k = 0; k < 4; ++k) {
            const int gk = (k + shift) % 4;
            const double dx = static_cast<double>(det_x[k]) - gt_x[gk];
            const double dy = static_cast<double>(det_y[k]) - gt_y[gk];
            err = std::max(err, std::sqrt(dx * dx + dy * dy));
        }
        if (err < best_err) { best_err = err; best_shift = shift; }
    }
    if (best_shift_out) *best_shift_out = best_shift;
    return best_err;
}

// rotation_angle_deg_between — geodesic angle (degrees) between two 3x3
// rotation matrices, row-major: angle = acos((trace(R1^T R2) - 1) / 2).
static double rotation_angle_deg_between(const float A[9], const float B[9])
{
    // trace(A^T B) = sum_{i,j} A[i][j]*B[i][j] for ROW-MAJOR 3x3 (A^T B's
    // trace equals the elementwise dot product of the two matrices' entries
    // when both are stored the same way — a standard identity, cheaper than
    // building A^T explicitly).
    double tr = 0.0;
    for (int k = 0; k < 9; ++k) tr += static_cast<double>(A[k]) * static_cast<double>(B[k]);
    double c = (tr - 1.0) * 0.5;
    c = std::min(1.0, std::max(-1.0, c));   // guard acos domain against float roundoff
    return std::acos(c) * 180.0 / kPiForDegrees;
}
// Rz(theta) applied on the RIGHT of R (rotating the tag's own local X/Y
// axes about its own Z by theta) — row-major 3x3 result of R * Rz(theta).
static void right_multiply_rz(const float R[9], double theta_rad, float out[9])
{
    const double c = std::cos(theta_rad), s = std::sin(theta_rad);
    // Rz = [c -s 0; s c 0; 0 0 1] (row-major)
    for (int i = 0; i < 3; ++i) {
        const double r0 = R[i * 3 + 0], r1 = R[i * 3 + 1], r2 = R[i * 3 + 2];
        out[i * 3 + 0] = static_cast<float>(r0 * c + r1 * s);
        out[i * 3 + 1] = static_cast<float>(-r0 * s + r1 * c);
        out[i * 3 + 2] = static_cast<float>(r2);
    }
}
// best_rotation_error_deg — same "try all 4 cyclic corner-labelings" idea
// as best_shift_corner_error, applied to the POSE: search the 4 candidate
// in-plane relabelings of the tag's own frame (0/90/180/270 degrees about
// its own Z) and report the SMALLEST geodesic rotation error against ground
// truth. Independent of best_shift_corner_error's search (no assumption
// that the two shifts must agree in sign/direction — see this file's
// header note on why a robust search beats a hand-derived convention here).
static double best_rotation_error_deg(const float R_det[9], const float R_gt[9])
{
    double best = 1e30;
    for (int shift = 0; shift < 4; ++shift) {
        float R_gt_rot[9];
        right_multiply_rz(R_gt, shift * (kPiForDegrees / 2.0), R_gt_rot);
        best = std::min(best, rotation_angle_deg_between(R_det, R_gt_rot));
    }
    return best;
}

// ===========================================================================
// Small pixel-font digit drawing + line drawing for the detections overlay
// artifact (debug/teaching visualization only — never consumed by the
// pipeline or the gates above).
// ===========================================================================
static const unsigned char kDigitFont[10][5] = {
    {0b111,0b101,0b101,0b101,0b111}, {0b010,0b110,0b010,0b010,0b111},
    {0b111,0b001,0b111,0b100,0b111}, {0b111,0b001,0b111,0b001,0b111},
    {0b101,0b101,0b111,0b001,0b001}, {0b111,0b100,0b111,0b001,0b111},
    {0b111,0b100,0b111,0b101,0b111}, {0b111,0b001,0b001,0b001,0b001},
    {0b111,0b101,0b111,0b101,0b111}, {0b111,0b101,0b111,0b001,0b111},
};
static void put_px(std::vector<unsigned char>& rgb, int W, int H, int x, int y, unsigned char r, unsigned char g, unsigned char b)
{
    if (x < 0 || x >= W || y < 0 || y >= H) return;
    const size_t i = (static_cast<size_t>(y) * W + x) * 3;
    rgb[i] = r; rgb[i + 1] = g; rgb[i + 2] = b;
}
static void draw_line(std::vector<unsigned char>& rgb, int W, int H, float x0, float y0, float x1, float y1,
                      unsigned char r, unsigned char g, unsigned char b)
{
    const float dx = x1 - x0, dy = y1 - y0;
    const int steps = std::max(1, static_cast<int>(std::sqrt(dx * dx + dy * dy)) * 2);
    for (int s = 0; s <= steps; ++s) {
        const float t = static_cast<float>(s) / static_cast<float>(steps);
        put_px(rgb, W, H, static_cast<int>(x0 + dx * t + 0.5f), static_cast<int>(y0 + dy * t + 0.5f), r, g, b);
    }
}
static void draw_digit(std::vector<unsigned char>& rgb, int W, int H, int x0, int y0, int digit, int scale,
                       unsigned char r, unsigned char g, unsigned char b)
{
    if (digit < 0 || digit > 9) return;
    for (int row = 0; row < 5; ++row) {
        for (int col = 0; col < 3; ++col) {
            if ((kDigitFont[digit][row] >> (2 - col)) & 1) {
                for (int sy = 0; sy < scale; ++sy)
                    for (int sx = 0; sx < scale; ++sx)
                        put_px(rgb, W, H, x0 + col * scale + sx, y0 + row * scale + sy, r, g, b);
            }
        }
    }
}
static void draw_number(std::vector<unsigned char>& rgb, int W, int H, int x0, int y0, int value, int scale,
                        unsigned char r, unsigned char g, unsigned char b)
{
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%d", value);
    int x = x0;
    for (const char* p = buf; *p; ++p) {
        if (*p >= '0' && *p <= '9') draw_digit(rgb, W, H, x, y0, *p - '0', scale, r, g, b);
        x += (3 * scale + scale);
    }
}

// ===========================================================================
// gates_metrics.csv / detections.csv writers.
// ===========================================================================
struct CsvRow { std::string gate, metric, value, tol, pass; };
static std::string fmt(double v, int prec = 6) { char buf[64]; std::snprintf(buf, sizeof(buf), "%.*f", prec, v); return std::string(buf); }
static bool write_gates_csv(const std::string& path, const std::vector<CsvRow>& rows)
{
    std::ofstream out(path);
    if (!out.is_open()) return false;
    out << "gate,metric,value,tolerance,pass\n";
    for (const auto& r : rows) out << r.gate << "," << r.metric << "," << r.value << "," << r.tol << "," << r.pass << "\n";
    return static_cast<bool>(out);
}
static bool write_detections_csv(const std::string& path,
                                 const std::vector<std::pair<std::string, const PipelineResult*>>& scenes)
{
    std::ofstream out(path);
    if (!out.is_open()) return false;
    out << "scene,candidate_index,border_ok,accepted,tag_id,rotation,hamming_distance,"
       << "corner0_x,corner0_y,corner1_x,corner1_y,corner2_x,corner2_y,corner3_x,corner3_y,"
       << "pose_valid,t0,t1,t2\n";
    for (const auto& sp : scenes) {
        for (const auto& d : sp.second->dets) {
            out << sp.first << "," << d.candidate_index << "," << (d.border_ok ? 1 : 0) << ","
               << (d.accepted ? 1 : 0) << "," << d.tag_id << "," << d.rotation << "," << d.hamming_distance << ",";
            for (int k = 0; k < 4; ++k) out << fmt(d.corners_x[k], 3) << "," << fmt(d.corners_y[k], 3) << (k < 3 ? "," : ",");
            out << (d.pose_valid ? 1 : 0) << "," << fmt(d.t[0], 4) << "," << fmt(d.t[1], 4) << "," << fmt(d.t[2], 4) << "\n";
        }
    }
    return static_cast<bool>(out);
}

// ===========================================================================
// main.
// ===========================================================================
int main(int argc, char** argv)
{
    std::string data_dir;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--data") && i + 1 < argc) data_dir = argv[++i];
        else { std::fprintf(stderr, "usage: %s [--data path/to/data/sample]\n", argv[0]); return 2; }
    }

    std::printf("[demo] AprilTag/ArUco GPU fiducial detector-decoder: adaptive threshold -> CCL -> "
               "quad extraction -> DLT homography -> grid decode -> pose (project 01.06)\n");
    print_device_info();

    // ---- load dictionary + scenes ------------------------------------------
    const std::string dict_path = find_data_file(data_dir, argv[0], "dictionary.bin");
    DictionaryData dict = dict_path.empty() ? DictionaryData{} : load_dictionary(dict_path);
    if (!dict.loaded) {
        std::printf("DATA: NOT FOUND or MALFORMED (dictionary.bin -- run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (sample data missing or malformed)\n");
        return 1;
    }

    struct SceneFile { const char* pgm; const char* gt; };
    const std::string main_pgm_path = find_data_file(data_dir, argv[0], "scene_main.pgm");
    const std::string main_gt_path  = find_data_file(data_dir, argv[0], "scene_main_ground_truth.csv");
    const std::string dis_pgm_path  = find_data_file(data_dir, argv[0], "scene_distractor.pgm");
    const std::string rob_pgm_path  = find_data_file(data_dir, argv[0], "scene_robustness.pgm");
    const std::string rob_gt_path   = find_data_file(data_dir, argv[0], "scene_robustness_ground_truth.csv");

    std::vector<unsigned char> gray_main, gray_dis, gray_rob;
    int wA = 0, hA = 0, wB = 0, hB = 0, wC = 0, hC = 0;
    std::vector<MainGroundTruthTag> main_gt;
    std::vector<RobustnessGroundTruthTag> rob_gt;

    const bool data_ok =
        !main_pgm_path.empty() && !dis_pgm_path.empty() && !rob_pgm_path.empty() &&
        !main_gt_path.empty() && !rob_gt_path.empty() &&
        read_pgm(main_pgm_path, wA, hA, gray_main) &&
        read_pgm(dis_pgm_path, wB, hB, gray_dis) &&
        read_pgm(rob_pgm_path, wC, hC, gray_rob) &&
        wA == kFullW && hA == kFullH && wB == kFullW && hB == kFullH && wC == kFullW && hC == kFullH &&
        load_main_ground_truth(main_gt_path, main_gt) &&
        load_robustness_ground_truth(rob_gt_path, rob_gt);

    if (!data_ok) {
        std::printf("DATA: NOT FOUND or MALFORMED (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (sample data missing or malformed)\n");
        return 1;
    }
    std::printf("PROBLEM: %dx%d scenes, %d-code dictionary (bits=%d, grid=%dx%d, min_dist=%d, "
               "correction_capacity=%d), 3 scenes (main: %zu tags, distractor: 0 tags, robustness: %zu tags)\n",
               kFullW, kFullH, static_cast<int>(dict.codes.size()), dict.bits_per_code, dict.grid_n, dict.grid_n,
               dict.min_distance, dict.correction_capacity, main_gt.size(), rob_gt.size());
    std::printf("DATA: 3 synthetic PGM scenes + dictionary.bin + 2 ground-truth CSVs "
               "[synthetic, seed 42, xorshift32] loaded from data/sample/\n");

    // ---- upload dictionary once, shared by every scene's GPU run ----------
    uint16_t* d_dict = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dict, dict.codes.size() * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpy(d_dict, dict.codes.data(), dict.codes.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));

    // ---- run GPU + CPU pipelines on all three scenes -----------------------
    double px_ms_main = 0, cd_ms_main = 0, cpu_ms_main = 0;
    double px_ms_dis = 0, cd_ms_dis = 0, cpu_ms_dis = 0;
    double px_ms_rob = 0, cd_ms_rob = 0, cpu_ms_rob = 0;

    PipelineResult gpu_main = run_gpu_pipeline(gray_main, kFullW, kFullH, d_dict, static_cast<int>(dict.codes.size()),
                                               dict.correction_capacity, px_ms_main, cd_ms_main);
    PipelineResult cpu_main = run_cpu_pipeline(gray_main, kFullW, kFullH, dict.codes, dict.correction_capacity, cpu_ms_main);

    PipelineResult gpu_dis = run_gpu_pipeline(gray_dis, kFullW, kFullH, d_dict, static_cast<int>(dict.codes.size()),
                                              dict.correction_capacity, px_ms_dis, cd_ms_dis);
    PipelineResult cpu_dis = run_cpu_pipeline(gray_dis, kFullW, kFullH, dict.codes, dict.correction_capacity, cpu_ms_dis);

    PipelineResult gpu_rob = run_gpu_pipeline(gray_rob, kFullW, kFullH, d_dict, static_cast<int>(dict.codes.size()),
                                              dict.correction_capacity, px_ms_rob, cd_ms_rob);
    PipelineResult cpu_rob = run_cpu_pipeline(gray_rob, kFullW, kFullH, dict.codes, dict.correction_capacity, cpu_ms_rob);

    CUDA_CHECK(cudaFree(d_dict));

    std::printf("[time] scene_main:        GPU pixel-parallel %.3f ms | GPU candidate-parallel (%zu cand) %.3f ms | "
               "CPU (all stages) %.3f ms | CCL sweeps %d\n",
               px_ms_main, gpu_main.candidates.size(), cd_ms_main, cpu_ms_main, gpu_main.ccl_sweeps);
    std::printf("[time] scene_distractor:  GPU pixel-parallel %.3f ms | GPU candidate-parallel (%zu cand) %.3f ms | "
               "CPU (all stages) %.3f ms | CCL sweeps %d\n",
               px_ms_dis, gpu_dis.candidates.size(), cd_ms_dis, cpu_ms_dis, gpu_dis.ccl_sweeps);
    std::printf("[time] scene_robustness:  GPU pixel-parallel %.3f ms | GPU candidate-parallel (%zu cand) %.3f ms | "
               "CPU (all stages) %.3f ms | CCL sweeps %d\n",
               px_ms_rob, gpu_rob.candidates.size(), cd_ms_rob, cpu_ms_rob, gpu_rob.ccl_sweeps);

    // ---- VERIFY -------------------------------------------------------------
    const bool v_main = verify_scene("scene_main", gpu_main, cpu_main);
    const bool v_dis  = verify_scene("scene_distractor", gpu_dis, cpu_dis);
    const bool v_rob  = verify_scene("scene_robustness", gpu_rob, cpu_rob);
    const bool verify_pass = v_main && v_dis && v_rob;
    std::printf("VERIFY: %s (GPU matches CPU reference within documented per-stage tolerance across all 3 scenes: "
               "local_mean, mask, ccl_label, candidate_count, centroid, corner, homography, decode, pose)\n",
               verify_pass ? "PASS" : "FAIL");

    std::vector<CsvRow> csv;

    // ---- GATE 1: detection (scene_main) -------------------------------------
    std::vector<bool> gt_matched(main_gt.size(), false);
    int detection_matches = 0;
    int accepted_count_main = 0;
    for (const auto& d : gpu_main.dets) if (d.accepted) ++accepted_count_main;
    for (size_t gi = 0; gi < main_gt.size(); ++gi) {
        const auto& g = main_gt[gi];
        float gcx = 0.0f, gcy = 0.0f;
        for (int k = 0; k < 4; ++k) { gcx += g.cx[k]; gcy += g.cy[k]; }
        gcx *= 0.25f; gcy *= 0.25f;
        double best_d = 1e30; int best_i = -1;
        for (size_t di = 0; di < gpu_main.dets.size(); ++di) {
            const auto& d = gpu_main.dets[di];
            if (!d.accepted) continue;
            float dcx = 0.0f, dcy = 0.0f;
            for (int k = 0; k < 4; ++k) { dcx += d.corners_x[k]; dcy += d.corners_y[k]; }
            dcx *= 0.25f; dcy *= 0.25f;
            const double dd = std::hypot(static_cast<double>(dcx) - gcx, static_cast<double>(dcy) - gcy);
            if (dd < best_d) { best_d = dd; best_i = static_cast<int>(di); }
        }
        if (best_i >= 0 && best_d < 40.0 && gpu_main.dets[best_i].tag_id == g.dict_id) {
            gt_matched[gi] = true;
            ++detection_matches;
        }
    }
    const bool gate_detection = (detection_matches == static_cast<int>(main_gt.size())) &&
                               (accepted_count_main == static_cast<int>(main_gt.size()));
    std::printf("GATE detection: %s\n", gate_detection ? "PASS" : "FAIL");
    std::printf("[info] detection: %d/%zu ground-truth tags matched with correct ID, %d total accepted "
               "detections in scene_main (want == %zu, no misses, no extras)\n",
               detection_matches, main_gt.size(), accepted_count_main, main_gt.size());
    csv.push_back({ "detection", "matched_of_total",
                   fmt(detection_matches, 0) + "/" + fmt(static_cast<double>(main_gt.size()), 0),
                   "6/6", gate_detection ? "PASS" : "FAIL" });

    // ---- GATE 2: corner-accuracy (scene_main, matched tags only) -----------
    double max_corner_err_px = 0.0;
    int corner_matched = 0;
    for (size_t gi = 0; gi < main_gt.size(); ++gi) {
        if (!gt_matched[gi]) continue;
        const auto& g = main_gt[gi];
        float gcx = 0.0f, gcy = 0.0f;
        for (int k = 0; k < 4; ++k) { gcx += g.cx[k]; gcy += g.cy[k]; }
        gcx *= 0.25f; gcy *= 0.25f;
        double best_d = 1e30; int best_i = -1;
        for (size_t di = 0; di < gpu_main.dets.size(); ++di) {
            const auto& d = gpu_main.dets[di];
            if (!d.accepted || d.tag_id != g.dict_id) continue;
            float dcx = 0.0f, dcy = 0.0f;
            for (int k = 0; k < 4; ++k) { dcx += d.corners_x[k]; dcy += d.corners_y[k]; }
            dcx *= 0.25f; dcy *= 0.25f;
            const double dd = std::hypot(static_cast<double>(dcx) - gcx, static_cast<double>(dcy) - gcy);
            if (dd < best_d) { best_d = dd; best_i = static_cast<int>(di); }
        }
        if (best_i < 0) continue;
        int shift = 0;
        const double err = best_shift_corner_error(gpu_main.dets[best_i].corners_x, gpu_main.dets[best_i].corners_y,
                                                   g.cx, g.cy, &shift);
        max_corner_err_px = std::max(max_corner_err_px, err);
        ++corner_matched;
    }
    const bool gate_corner = (corner_matched == detection_matches) && (max_corner_err_px <= kCornerGateTolPx);
    std::printf("GATE corner_accuracy: %s\n", gate_corner ? "PASS" : "FAIL");
    std::printf("[info] corner_accuracy: max corner error over %d matched tags = %.4f px (tol %.2f px; "
               "best-of-4-cyclic-shift alignment against renderer ground truth)\n",
               corner_matched, max_corner_err_px, kCornerGateTolPx);
    csv.push_back({ "corner_accuracy", "max_corner_error_px", fmt(max_corner_err_px, 4), fmt(kCornerGateTolPx, 2), gate_corner ? "PASS" : "FAIL" });

    // ---- GATE 5 (computed here alongside corners): pose --------------------
    double max_rot_err_deg = 0.0, max_trans_err_pct = 0.0;
    int pose_matched = 0;
    for (size_t gi = 0; gi < main_gt.size(); ++gi) {
        if (!gt_matched[gi]) continue;
        const auto& g = main_gt[gi];
        float gcx = 0.0f, gcy = 0.0f;
        for (int k = 0; k < 4; ++k) { gcx += g.cx[k]; gcy += g.cy[k]; }
        gcx *= 0.25f; gcy *= 0.25f;
        double best_d = 1e30; int best_i = -1;
        for (size_t di = 0; di < gpu_main.dets.size(); ++di) {
            const auto& d = gpu_main.dets[di];
            if (!d.accepted || d.tag_id != g.dict_id || !d.pose_valid) continue;
            float dcx = 0.0f, dcy = 0.0f;
            for (int k = 0; k < 4; ++k) { dcx += d.corners_x[k]; dcy += d.corners_y[k]; }
            dcx *= 0.25f; dcy *= 0.25f;
            const double dd = std::hypot(static_cast<double>(dcx) - gcx, static_cast<double>(dcy) - gcy);
            if (dd < best_d) { best_d = dd; best_i = static_cast<int>(di); }
        }
        if (best_i < 0) continue;
        const auto& d = gpu_main.dets[best_i];
        const double rot_err = best_rotation_error_deg(d.R, g.R);
        const double dx = static_cast<double>(d.t[0]) - g.t[0];
        const double dy = static_cast<double>(d.t[1]) - g.t[1];
        const double dz = static_cast<double>(d.t[2]) - g.t[2];
        const double trans_err_m = std::sqrt(dx * dx + dy * dy + dz * dz);
        const double trans_err_pct = 100.0 * trans_err_m / kTagSizeM;
        max_rot_err_deg = std::max(max_rot_err_deg, rot_err);
        max_trans_err_pct = std::max(max_trans_err_pct, trans_err_pct);
        ++pose_matched;
    }
    const bool gate_pose = (pose_matched == detection_matches) &&
                          (max_rot_err_deg <= kPoseRotGateTolDeg) && (max_trans_err_pct <= kPoseTransGateTolPct);
    std::printf("GATE pose: %s\n", gate_pose ? "PASS" : "FAIL");
    std::printf("[info] pose: over %d matched tags, max rotation error = %.3f deg (tol %.2f), "
               "max translation error = %.3f%% of tag size (tol %.2f%%) "
               "[analytic ground truth from scripts/make_synthetic.py, independent of this pipeline's own code]\n",
               pose_matched, max_rot_err_deg, kPoseRotGateTolDeg, max_trans_err_pct, kPoseTransGateTolPct);
    csv.push_back({ "pose", "max_rotation_error_deg", fmt(max_rot_err_deg, 3), fmt(kPoseRotGateTolDeg, 2), gate_pose ? "PASS" : "FAIL" });
    csv.push_back({ "pose", "max_translation_error_pct", fmt(max_trans_err_pct, 3), fmt(kPoseTransGateTolPct, 2), gate_pose ? "PASS" : "FAIL" });

    // ---- GATE 3: decode-robustness (scene_robustness) -----------------------
    int robustness_ok = 0;
    for (const auto& g : rob_gt) {
        float gcx = 0.0f, gcy = 0.0f;
        for (int k = 0; k < 4; ++k) { gcx += g.cx[k]; gcy += g.cy[k]; }
        gcx *= 0.25f; gcy *= 0.25f;
        double best_d = 1e30; int best_i = -1;
        for (size_t di = 0; di < gpu_rob.dets.size(); ++di) {
            const auto& d = gpu_rob.dets[di];
            float dcx = 0.0f, dcy = 0.0f;
            for (int k = 0; k < 4; ++k) { dcx += d.corners_x[k]; dcy += d.corners_y[k]; }
            dcx *= 0.25f; dcy *= 0.25f;
            const double dd = std::hypot(static_cast<double>(dcx) - gcx, static_cast<double>(dcy) - gcy);
            if (dd < best_d) { best_d = dd; best_i = static_cast<int>(di); }
        }
        bool ok = false;
        if (best_i >= 0 && best_d < 40.0) {
            const auto& d = gpu_rob.dets[best_i];
            if (g.expect_accept) ok = d.accepted && (d.tag_id == g.true_dict_id);
            else                 ok = !d.accepted;   // beyond capacity: must NOT be accepted (at all, under any id)
        }
        std::printf("[info] decode_robustness: tag_index=%d true_dict_id=%d num_flips=%d expected=%s -> %s\n",
                   g.tag_index, g.true_dict_id, g.num_flips, g.expect_accept ? "accept" : "reject", ok ? "OK" : "MISMATCH");
        if (ok) ++robustness_ok;
    }
    const bool gate_robustness = (robustness_ok == static_cast<int>(rob_gt.size()));
    std::printf("GATE decode_robustness: %s\n", gate_robustness ? "PASS" : "FAIL");
    std::printf("[info] decode_robustness: %d/%zu tags behaved as expected (at-capacity accept, beyond-capacity reject; "
               "correction_capacity = %d)\n", robustness_ok, rob_gt.size(), dict.correction_capacity);
    csv.push_back({ "decode_robustness", "correct_of_total",
                   fmt(robustness_ok, 0) + "/" + fmt(static_cast<double>(rob_gt.size()), 0),
                   "4/4", gate_robustness ? "PASS" : "FAIL" });

    // ---- GATE 4: false-positive (scene_distractor) --------------------------
    int accepted_count_dis = 0;
    for (const auto& d : gpu_dis.dets) if (d.accepted) ++accepted_count_dis;
    const bool gate_false_positive = (accepted_count_dis == 0);
    std::printf("GATE false_positive: %s\n", gate_false_positive ? "PASS" : "FAIL");
    std::printf("[info] false_positive: %d accepted detections in the tag-free distractor scene "
               "(%zu candidate components reached quad extraction; want 0 accepted)\n",
               accepted_count_dis, gpu_dis.candidates.size());
    csv.push_back({ "false_positive", "accepted_detections", fmt(accepted_count_dis, 0), "0", gate_false_positive ? "PASS" : "FAIL" });

    // ======================= ARTIFACTS ============================================
    const std::string out_dir = resolve_out_dir(argv[0]);
    bool artifact_ok = !out_dir.empty();

    // detections_overlay.ppm: scene_main in RGB with quad edges + ID drawn.
    std::vector<unsigned char> overlay(static_cast<size_t>(kFullW) * kFullH * 3);
    for (size_t i = 0; i < gray_main.size(); ++i) { overlay[i * 3] = overlay[i * 3 + 1] = overlay[i * 3 + 2] = gray_main[i]; }
    for (const auto& d : gpu_main.dets) {
        const unsigned char r = d.accepted ? 40 : 220, g = d.accepted ? 220 : 40, b = 40;
        for (int k = 0; k < 4; ++k) {
            const int k2 = (k + 1) % 4;
            draw_line(overlay, kFullW, kFullH, d.corners_x[k], d.corners_y[k], d.corners_x[k2], d.corners_y[k2], r, g, b);
        }
        if (d.accepted) {
            float cx = 0, cy = 0;
            for (int k = 0; k < 4; ++k) { cx += d.corners_x[k]; cy += d.corners_y[k]; }
            draw_number(overlay, kFullW, kFullH, static_cast<int>(cx / 4.0f) - 6, static_cast<int>(cy / 4.0f) - 3,
                       d.tag_id, 2, 255, 255, 0);
        }
    }
    artifact_ok = artifact_ok && write_ppm(out_dir + "/detections_overlay.ppm", kFullW, kFullH, overlay);

    // decoded_grid_debug.ppm: re-sample the FIRST accepted main-scene
    // detection's 6x6 grid at a fixed display resolution (debug-only code,
    // independent of both the GPU and CPU decode paths — it exists purely
    // to let a learner SEE what the decoder saw).
    {
        const Detection* dbg = nullptr;
        const Homography* dbg_h = nullptr;
        for (size_t i = 0; i < gpu_main.dets.size(); ++i) {
            if (gpu_main.dets[i].accepted) { dbg = &gpu_main.dets[i]; dbg_h = &gpu_main.homs[i]; break; }
        }
        const int cellPx = 30, dispN = kGridN * cellPx;
        std::vector<unsigned char> grid_img(static_cast<size_t>(dispN) * dispN * 3, 128);
        if (dbg && dbg_h && dbg_h->valid) {
            const double half = kTagHalfM, cell = kTagSizeM / kGridN;
            for (int r = 0; r < kGridN; ++r) {
                for (int c = 0; c < kGridN; ++c) {
                    const double X = -half + (c + 0.5) * cell, Y = -half + (r + 0.5) * cell;
                    const double w = dbg_h->h[6] * X + dbg_h->h[7] * Y + dbg_h->h[8];
                    const double px = (dbg_h->h[0] * X + dbg_h->h[1] * Y + dbg_h->h[2]) / w;
                    const double py = (dbg_h->h[3] * X + dbg_h->h[4] * Y + dbg_h->h[5]) / w;
                    const int xi = std::min(std::max(static_cast<int>(px), 0), kFullW - 1);
                    const int yi = std::min(std::max(static_cast<int>(py), 0), kFullH - 1);
                    const unsigned char sample = gray_main[static_cast<size_t>(yi) * kFullW + xi];
                    const float local_m = gpu_main.local_mean[static_cast<size_t>(yi) * kFullW + xi];
                    const bool dark = static_cast<float>(sample) < (local_m - kThreshBiasC);
                    const unsigned char shade = dark ? 20 : 235;
                    const unsigned char rr = is_border_cell(r, c) ? 200 : shade;
                    const unsigned char gg = is_border_cell(r, c) ? 80 : shade;
                    const unsigned char bb = is_border_cell(r, c) ? 80 : shade;
                    for (int yy = 0; yy < cellPx; ++yy)
                        for (int xx = 0; xx < cellPx; ++xx)
                            put_px(grid_img, dispN, dispN, c * cellPx + xx, r * cellPx + yy, rr, gg, bb);
                }
            }
        }
        artifact_ok = artifact_ok && write_ppm(out_dir + "/decoded_grid_debug.ppm", dispN, dispN, grid_img);
    }

    std::vector<std::pair<std::string, const PipelineResult*>> scene_dets = {
        { "scene_main", &gpu_main }, { "scene_distractor", &gpu_dis }, { "scene_robustness", &gpu_rob }
    };
    artifact_ok = artifact_ok && write_detections_csv(out_dir + "/detections.csv", scene_dets);
    artifact_ok = artifact_ok && write_gates_csv(out_dir + "/gates_metrics.csv", csv);

    if (artifact_ok)
        std::printf("ARTIFACT: wrote demo/out/{detections_overlay.ppm, decoded_grid_debug.ppm, detections.csv, gates_metrics.csv}\n");
    else
        std::printf("ARTIFACT: FAILED to write one or more demo/out files\n");

    // ---- verdict --------------------------------------------------------------
    const bool success = verify_pass && gate_detection && gate_corner && gate_robustness &&
                        gate_false_positive && gate_pose && artifact_ok;
    if (success) {
        std::printf("RESULT: PASS (VERIFY + all 5 gates passed: detection, corner_accuracy, decode_robustness, "
                   "false_positive, pose)\n");
    } else {
        std::printf("RESULT: FAIL (VERIFY or a gate above did not pass -- see GATE:/VERIFY:/[info] lines)\n");
    }
    return success ? 0 : 1;
}
