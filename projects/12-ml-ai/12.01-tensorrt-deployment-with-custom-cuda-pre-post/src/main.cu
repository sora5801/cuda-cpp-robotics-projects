// ===========================================================================
// main.cu — entry point for project 12.01
//           TensorRT deployment with custom CUDA pre/post kernels:
//           NMS, argmax decode, keypoint extraction
//
// What this program does, start to finish
// ---------------------------------------
//   1. Print the banner + GPU info; load the committed weights, test image,
//      and ground-truth object list from data/sample/ (strict loaders).
//   2. Upload weights + image once; run the FALLBACK (default, no
//      TensorRT) CUDA pipeline: preprocess -> conv1 -> conv2 -> head ->
//      argmax decode -> threshold+box decode -> (host sort) -> IoU matrix
//      -> (host greedy NMS) -> keypoint extract.
//   3. Run the SAME pipeline on the CPU (reference_cpu.cpp) as an
//      independent oracle.
//   4. VERIFY STAGE (the §5 GPU-vs-CPU gate, applied at EVERY stage, not
//      just the end): diff the preprocessed tensor, both conv activations,
//      the head's raw output, the pre-/post-NMS candidate counts, and the
//      final detections, each within a documented tolerance.
//   5. GROUND-TRUTH GATE: match the GPU path's final detections against
//      the scene's known objects (center distance + IoU bounds), count
//      false positives, and check the NMS reduction factor.
//   6. Write demo/out/detections.pgm (source image with boxes + keypoints
//      burned in) and demo/out/detections.csv (one row per detection).
//   7. RESULT: PASS only if VERIFY and the ground-truth gate both hold.
//
// TensorRT (README "Build" documents the opt-in): this file calls
// tensorrt_path_available() unconditionally (always linked, see
// tensorrt_path.cpp) to report status, and — ONLY when compiled with
// -DUSE_TENSORRT — additionally invokes the optional, best-effort
// TensorRT demonstration. Neither of those paths touches the STABLE
// output lines below: the checked demo/expected_output.txt contract is
// the fallback (plain-CUDA) path, always, on every build configuration
// (the ratified design rule — CLAUDE.md §5, this project's README).
//
// Determinism: there is NO run-time randomness anywhere in this program —
// the scene's mild background dither was baked into the committed
// data/sample/test_scene.ppm bytes by scripts/make_synthetic.py (seed 42);
// every kernel here is a deterministic function of its inputs. All stable
// output lines are therefore bit-reproducible on any machine; the exact
// MEASURED numbers (candidate counts, timings, center errors) live only in
// "[info]"/"[time]" lines, per this repo's convention (CLAUDE.md §12).
//
// Output contract: stable lines are "[demo]", "PROBLEM:", "WEIGHTS:",
// "SCENE:", "VERIFY:", "GROUNDTRUTH:", "ARTIFACT:", "RESULT:" —
// "[info]"/"[time]" are unchecked. Change a stable line -> update
// demo/expected_output.txt in the same change.
//
// Read this first, then kernels.cuh -> kernels.cu -> reference_cpu.cpp.
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
#include <direct.h>               // _mkdir (std::filesystem avoided in .cu — see 07.09)
#else
#include <sys/stat.h>
#endif

// ---------------------------------------------------------------------------
// Ground-truth object record (loaded from data/sample/ground_truth.csv):
// one synthetic rectangle placed by scripts/make_synthetic.py, in SOURCE-
// image pixel coordinates (int, top-left corner + size).
// ---------------------------------------------------------------------------
struct GtObject {
    int class_id;      // 0 = red, 1 = blue
    int x0, y0, w, h;  // SOURCE-image pixels
};

// ---------------------------------------------------------------------------
// Verification / ground-truth-gate tolerances — TEST-HARNESS constants,
// not part of the algorithm's contract (that is kernels.cuh's job), so
// they live here, exactly like 08.01 keeps its success thresholds in
// main.cu rather than kernels.cuh. Margins are wide relative to the
// measured values (see README "Expected output" and THEORY.md "How we
// verify correctness") so ordinary FP32 rounding differences across GPU
// architectures cannot flip the verdict.
// ---------------------------------------------------------------------------
static const float kStageTol             = 1e-3f;  // relative tol, GPU-vs-CPU tensor stages (floor 1.0)
static const float kDetFieldAbsTol       = 1e-2f;  // abs tol, GPU-vs-CPU final-detection scalar fields
static const float kMatchCenterTolNetPx  = 6.0f;   // max center distance (network-input px) to call a match
static const float kMatchIouMinForMatch  = 0.30f;  // min IoU (network-input space) to call a match
static const int   kMaxFalsePositives    = 1;      // documented upper bound (measured: 0)
static const float kMinNmsReductionFactor = 3.0f;  // pre-NMS/post-NMS must be at least this (measured: ~7x)

// ---------------------------------------------------------------------------
// max_rel_diff — GPU-vs-CPU tensor comparison, the same pattern 08.01 uses
// for its rollout-cost gate: relative difference with a floor of 1.0 (so
// near-zero reference values do not blow up the ratio), max over all n
// elements. Reused for every pipeline stage below.
// ---------------------------------------------------------------------------
static float max_rel_diff(const float* gpu, const float* cpu, int n)
{
    float worst = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float scale = std::fabs(cpu[i]) > 1.0f ? std::fabs(cpu[i]) : 1.0f;
        const float d = std::fabs(gpu[i] - cpu[i]) / scale;
        if (d > worst) worst = d;
    }
    return worst;
}

// ---------------------------------------------------------------------------
// The deterministic sort/greedy-NMS the GPU PATH performs on the host,
// between launch_threshold_box_decode and launch_iou_matrix / after
// launch_iou_matrix. This is a SEPARATE implementation from
// reference_cpu.cpp's nms_cpu (deliberate duplication — see that file's
// header): the comparator is trivial bookkeeping (not "the algorithm"), so
// it is fine for both to use the identical rule, but the SUPPRESSION SCAN
// here reads a GPU-COMPUTED IoU matrix, while nms_cpu computes IoUs
// on the fly — genuinely different code paths converging on the same
// verified answer.
// ---------------------------------------------------------------------------
static void sort_candidates_by_score(std::vector<Detection>& v)
{
    std::stable_sort(v.begin(), v.end(), [](const Detection& a, const Detection& b) {
        if (a.score != b.score) return a.score > b.score;
        return a.cell_index < b.cell_index;
    });
}

// Consumes a precomputed n*n row-major IoU matrix (sorted-by-score order)
// and performs the SEQUENTIAL greedy suppression scan described in
// kernels.cu KERNEL 5's header — this loop is the "parallelism tension"
// discussion made concrete: it cannot start on candidate j until every
// higher-scored candidate i<j has already decided survive-or-suppress.
static std::vector<Detection> greedy_nms_from_matrix(const std::vector<Detection>& sorted,
                                                      const std::vector<float>& iou)
{
    const int n = static_cast<int>(sorted.size());
    std::vector<bool> suppressed(static_cast<size_t>(n), false);
    std::vector<Detection> kept;
    kept.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        if (suppressed[static_cast<size_t>(i)]) continue;
        kept.push_back(sorted[static_cast<size_t>(i)]);
        for (int j = i + 1; j < n; ++j) {
            if (suppressed[static_cast<size_t>(j)]) continue;
            if (sorted[static_cast<size_t>(j)].class_id != sorted[static_cast<size_t>(i)].class_id) continue;
            if (iou[static_cast<size_t>(i) * n + static_cast<size_t>(j)] > kNmsIouThreshold)
                suppressed[static_cast<size_t>(j)] = true;
        }
    }
    return kept;
}

// ---------------------------------------------------------------------------
// load_weight_blob — strict loader for data/sample/weights.bin. Reads each
// array with its OWN ifstream::read() call, in the exact order documented
// in kernels.cuh SECTION 3 — never one bulk struct-sized read (portable
// across any struct-packing behavior). Aborts (returns false) on any
// magic/version/size mismatch or short read: a malformed weight file must
// never silently produce garbage detections.
// ---------------------------------------------------------------------------
static bool load_weight_blob(const std::string& path, WeightBlob& out)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return false;

    char magic[8];
    in.read(magic, 8);
    if (!in || std::memcmp(magic, "RCWTPK01", 8) != 0) {
        std::fprintf(stderr, "weights: bad magic in '%s'\n", path.c_str());
        return false;
    }
    uint32_t version = 0;
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (!in || version != 1) {
        std::fprintf(stderr, "weights: unsupported format_version %u in '%s'\n",
                     static_cast<unsigned>(version), path.c_str());
        return false;
    }

    in.read(reinterpret_cast<char*>(&out.conv1_w[0][0][0][0]), sizeof(out.conv1_w));
    in.read(reinterpret_cast<char*>(&out.conv1_b[0]),          sizeof(out.conv1_b));
    in.read(reinterpret_cast<char*>(&out.conv2_w[0][0][0][0]), sizeof(out.conv2_w));
    in.read(reinterpret_cast<char*>(&out.conv2_b[0]),          sizeof(out.conv2_b));
    in.read(reinterpret_cast<char*>(&out.head_w[0][0][0][0]),  sizeof(out.head_w));
    in.read(reinterpret_cast<char*>(&out.head_b[0]),           sizeof(out.head_b));
    if (!in) {
        std::fprintf(stderr, "weights: short read in '%s' (truncated file?)\n", path.c_str());
        return false;
    }
    // Confirm there is no trailing garbage: a well-formed file ends exactly
    // here. Reading one more byte must fail.
    char extra;
    in.read(&extra, 1);
    if (in.gcount() != 0) {
        std::fprintf(stderr, "weights: '%s' longer than the documented format\n", path.c_str());
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// load_ppm — strict loader for the committed P6 (binary RGB) test image.
// Only accepts EXACTLY the format scripts/make_synthetic.py writes: ASCII
// header tokens "P6", width, height, maxval(=255), then a single
// whitespace byte, then width*height*3 raw bytes — this is not a general
// PPM parser (real PPM allows '#' comments and flexible whitespace; we
// don't need that generality here and a strict check catches a corrupted
// or hand-edited sample immediately instead of misreading it).
// ---------------------------------------------------------------------------
static bool load_ppm(const std::string& path, std::vector<uint8_t>& out_hwc)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return false;

    std::string magic;
    int w = 0, h = 0, maxval = 0;
    in >> magic >> w >> h >> maxval;
    if (!in || magic != "P6" || w != kSrcW || h != kSrcH || maxval != 255) {
        std::fprintf(stderr,
            "test image: expected 'P6 %d %d 255', got '%s %d %d %d' in '%s'\n",
            kSrcW, kSrcH, magic.c_str(), w, h, maxval, path.c_str());
        return false;
    }
    in.get();   // consume the single mandatory whitespace byte after maxval

    out_hwc.resize(static_cast<size_t>(kSrcW) * kSrcH * 3);
    in.read(reinterpret_cast<char*>(out_hwc.data()), static_cast<std::streamsize>(out_hwc.size()));
    if (!in) {
        std::fprintf(stderr, "test image: short read in '%s'\n", path.c_str());
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// load_ground_truth — strict loader for data/sample/ground_truth.csv, the
// same "# comments / labeled rows / abort on anything unknown" discipline
// as 08.01's scenario loader.
// ---------------------------------------------------------------------------
static bool load_ground_truth(const std::string& path, std::vector<GtObject>& out)
{
    std::ifstream in(path);
    if (!in.is_open()) return false;

    out.clear();
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string label, cell;
        std::getline(ss, label, ',');
        if (label != "OBJ") {
            std::fprintf(stderr, "ground_truth: unknown row label '%s'\n", label.c_str());
            return false;
        }
        GtObject g{};
        int fields[5];
        for (int i = 0; i < 5; ++i) {
            if (!std::getline(ss, cell, ',')) {
                std::fprintf(stderr, "ground_truth: short OBJ row\n");
                return false;
            }
            fields[i] = std::atoi(cell.c_str());
        }
        g.class_id = fields[0];
        g.x0 = fields[1]; g.y0 = fields[2]; g.w = fields[3]; g.h = fields[4];
        if (g.class_id < 0 || g.class_id >= kNumClasses) {
            std::fprintf(stderr, "ground_truth: class_id %d out of range\n", g.class_id);
            return false;
        }
        out.push_back(g);
    }
    return !out.empty();
}

// ---------------------------------------------------------------------------
// Path resolution — identical strategy to 08.01's find_scenario: try a CLI
// override, then paths relative to the executable, then relative to CWD.
// ---------------------------------------------------------------------------
static std::string project_root_from(const char* argv0)
{
    std::string exe(argv0 ? argv0 : "");
    size_t cut = exe.find_last_of("/\\");
    if (cut == std::string::npos) return ".";
    return exe.substr(0, cut) + "/../../..";
}

static std::string find_data_file(const std::string& cli_dir, const char* argv0, const char* filename)
{
    std::vector<std::string> candidates;
    if (!cli_dir.empty()) candidates.push_back(cli_dir + "/" + filename);
    candidates.push_back(project_root_from(argv0) + "/data/sample/" + filename);
    candidates.push_back(std::string("data/sample/") + filename);
    candidates.push_back(std::string("../data/sample/") + filename);
    for (const auto& c : candidates)
        if (std::ifstream(c, std::ios::binary).is_open()) return c;
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

// resolve_out_dir — the artifact directory needs the SAME multi-candidate
// resolution find_data_file already applies to input files: the
// project_root_from(argv0) candidate assumes the VS OutDir convention
// (build/x64/Release, 3 directories below the project root — this repo's
// REQUIRED build); the optional CMake path's default multi-config layout
// (build-cmake/Release, only 2 directories deep) resolves one level too
// high through that formula. A CWD-relative candidate fixes it for BOTH
// run_demo scripts, which invoke the executable with the project root as
// the working directory. ensure_dir()'s single-level _mkdir only succeeds
// when its immediate parent ("demo/") already exists — true for every
// candidate below, since demo/ is a committed folder in every project.
static std::string resolve_out_dir(const char* argv0)
{
    const std::vector<std::string> candidates = {
        project_root_from(argv0) + "/demo/out",   // VS OutDir layout (required build)
        "demo/out",                                // CWD-relative (both run_demo scripts)
        "../demo/out",                              // one level up, defensive fallback
    };
    for (const auto& c : candidates)
        if (ensure_dir(c)) return c;
    return candidates.front();   // exhausted; caller reports the write failure honestly
}

// ---------------------------------------------------------------------------
// class_name — small lookup used by the artifact writer and log lines.
// ---------------------------------------------------------------------------
static const char* class_name(int class_id)
{
    return class_id == 0 ? "red" : (class_id == 1 ? "blue" : "?");
}

// ---------------------------------------------------------------------------
// iou_net — host helper for the ground-truth gate (network-input pixel
// space). A THIRD independent IoU implementation in this file (after
// kernels.cu's iou_device and reference_cpu.cpp's iou_host) is deliberate:
// this one is test-harness code checking the ALGORITHM's output against
// ground truth, not part of the algorithm itself — see the file header.
// ---------------------------------------------------------------------------
static float iou_net(float ax0, float ay0, float ax1, float ay1,
                     float bx0, float by0, float bx1, float by1)
{
    const float ix0 = std::max(ax0, bx0), iy0 = std::max(ay0, by0);
    const float ix1 = std::min(ax1, bx1), iy1 = std::min(ay1, by1);
    const float iw = std::max(0.0f, ix1 - ix0), ih = std::max(0.0f, iy1 - iy0);
    const float inter = iw * ih;
    const float area_a = std::max(0.0f, ax1 - ax0) * std::max(0.0f, ay1 - ay0);
    const float area_b = std::max(0.0f, bx1 - bx0) * std::max(0.0f, by1 - by0);
    const float uni = area_a + area_b - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

// ===========================================================================
// main
// ===========================================================================
int main(int argc, char** argv)
{
    std::string data_dir;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--data") && i + 1 < argc) data_dir = argv[++i];
        else {
            std::fprintf(stderr, "usage: %s [--data data/sample/]\n", argv[0]);
            return 2;
        }
    }

    std::printf("[demo] TensorRT deployment demo: custom CUDA pre/post kernels around a fixed synthetic detector (project 12.01)\n");
    print_device_info();
    std::printf("PROBLEM: fixed synthetic 2-class detector, input %dx%dx3 -> net %dx%d -> conv(%d->%d)->conv(%d->%d)->head(%d->%d) -> grid %dx%d, FP32\n",
               kSrcW, kSrcH, kNetW, kNetH, kConv1In, kConv1Out, kConv2In, kConv2Out,
               kHeadIn, kHeadOut, kGridW, kGridH);
    std::printf("[info] TensorRT optional path: %s (see README Build section to enable; USE_TENSORRT define)\n",
               tensorrt_path_available() ? "COMPILED IN (not exercised by this demo's checked output)"
                                        : "not compiled (default build — no TensorRT headers/libs required)");

    // ---- load committed sample data ----------------------------------------
    const std::string weights_path = find_data_file(data_dir, argv[0], "weights.bin");
    const std::string image_path   = find_data_file(data_dir, argv[0], "test_scene.ppm");
    const std::string gt_path      = find_data_file(data_dir, argv[0], "ground_truth.csv");
    if (weights_path.empty() || image_path.empty() || gt_path.empty()) {
        std::printf("WEIGHTS: NOT FOUND — data/sample/{weights.bin,test_scene.ppm,ground_truth.csv} missing (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (sample data missing)\n");
        return 1;
    }

    // "[info]" lines below carry the absolute file paths this run resolved
    // to — those are NOT stable across machines/clone locations, so they
    // are deliberately kept out of the checked "WEIGHTS:"/"SCENE:" lines
    // (same split as 08.01's "[info] scenario file: %s" vs. its path-free
    // "SCENARIO:" line — see main.cu's output-contract comment above).
    std::printf("[info] weights file: %s\n", weights_path.c_str());
    WeightBlob weights{};
    if (!load_weight_blob(weights_path, weights)) {
        std::printf("WEIGHTS: MALFORMED — see stderr\n");
        std::printf("RESULT: FAIL (weights malformed)\n");
        return 1;
    }
    std::printf("WEIGHTS: loaded (460 bytes, format v1, synthetic fixed weights, seed 42) [synthetic]\n");

    std::printf("[info] test image file: %s\n", image_path.c_str());
    std::vector<uint8_t> src_hwc;
    if (!load_ppm(image_path, src_hwc)) {
        std::printf("SCENE: MALFORMED — see stderr\n");
        std::printf("RESULT: FAIL (test image malformed)\n");
        return 1;
    }
    std::printf("[info] ground truth file: %s\n", gt_path.c_str());
    std::vector<GtObject> gt;
    if (!load_ground_truth(gt_path, gt)) {
        std::printf("SCENE: MALFORMED — see stderr\n");
        std::printf("RESULT: FAIL (ground truth malformed)\n");
        return 1;
    }
    std::printf("SCENE: %dx%d RGB test image with %d ground-truth object(s) [synthetic]\n",
               kSrcW, kSrcH, static_cast<int>(gt.size()));

    // ======================= GPU PATH (the fallback pipeline) ===============
    const int ncells = kGridH * kGridW;

    uint8_t* d_src_hwc = nullptr;
    float* d_net_chw = nullptr;
    float* d_conv1_w = nullptr, * d_conv1_b = nullptr, * d_conv1_out = nullptr;
    float* d_conv2_w = nullptr, * d_conv2_b = nullptr, * d_conv2_out = nullptr;
    float* d_head_w = nullptr, * d_head_b = nullptr, * d_head_out = nullptr;
    int* d_best_class = nullptr;
    float* d_best_score = nullptr;
    Detection* d_candidates = nullptr;
    int* d_count = nullptr;

    CUDA_CHECK(cudaMalloc(&d_src_hwc, src_hwc.size()));
    CUDA_CHECK(cudaMalloc(&d_net_chw, 3 * kNetH * kNetW * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv1_w, sizeof(weights.conv1_w)));
    CUDA_CHECK(cudaMalloc(&d_conv1_b, sizeof(weights.conv1_b)));
    CUDA_CHECK(cudaMalloc(&d_conv1_out, static_cast<size_t>(kConv1Out) * kConv1OutH * kConv1OutW * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv2_w, sizeof(weights.conv2_w)));
    CUDA_CHECK(cudaMalloc(&d_conv2_b, sizeof(weights.conv2_b)));
    CUDA_CHECK(cudaMalloc(&d_conv2_out, static_cast<size_t>(kConv2Out) * kConv2OutH * kConv2OutW * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_head_w, sizeof(weights.head_w)));
    CUDA_CHECK(cudaMalloc(&d_head_b, sizeof(weights.head_b)));
    CUDA_CHECK(cudaMalloc(&d_head_out, static_cast<size_t>(kHeadOut) * kHeadOutH * kHeadOutW * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_best_class, static_cast<size_t>(ncells) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_best_score, static_cast<size_t>(ncells) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_candidates, static_cast<size_t>(kMaxCandidates) * sizeof(Detection)));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_src_hwc, src_hwc.data(), src_hwc.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conv1_w, &weights.conv1_w[0][0][0][0], sizeof(weights.conv1_w), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conv1_b, &weights.conv1_b[0], sizeof(weights.conv1_b), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conv2_w, &weights.conv2_w[0][0][0][0], sizeof(weights.conv2_w), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conv2_b, &weights.conv2_b[0], sizeof(weights.conv2_b), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_head_w, &weights.head_w[0][0][0][0], sizeof(weights.head_w), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_head_b, &weights.head_b[0], sizeof(weights.head_b), cudaMemcpyHostToDevice));

    GpuTimer gpu_pipeline_timer;
    gpu_pipeline_timer.begin();

    launch_preprocess(d_src_hwc, d_net_chw);
    launch_conv2d(d_net_chw, d_conv1_w, d_conv1_b, d_conv1_out,
                 kConv1In, kConv1Out, kNetH, kNetW, kConv1K, kConv1Stride, kConv1Pad,
                 kConv1OutH, kConv1OutW, /*relu=*/true);
    launch_conv2d(d_conv1_out, d_conv2_w, d_conv2_b, d_conv2_out,
                 kConv2In, kConv2Out, kConv1OutH, kConv1OutW, kConv2K, kConv2Stride, kConv2Pad,
                 kConv2OutH, kConv2OutW, /*relu=*/true);
    launch_conv2d(d_conv2_out, d_head_w, d_head_b, d_head_out,
                 kHeadIn, kHeadOut, kConv2OutH, kConv2OutW, kHeadK, kHeadStride, kHeadPad,
                 kHeadOutH, kHeadOutW, /*relu=*/false);
    launch_argmax_decode(d_head_out, d_best_class, d_best_score);
    launch_threshold_box_decode(d_best_class, d_best_score, d_head_out, d_candidates, d_count);

    const float conv_pipeline_ms = gpu_pipeline_timer.end_ms();

    int n_pre_nms = 0;
    CUDA_CHECK(cudaMemcpy(&n_pre_nms, d_count, sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<Detection> gpu_candidates(static_cast<size_t>(n_pre_nms));
    if (n_pre_nms > 0)
        CUDA_CHECK(cudaMemcpy(gpu_candidates.data(), d_candidates,
                              static_cast<size_t>(n_pre_nms) * sizeof(Detection), cudaMemcpyDeviceToHost));

    // (host) deterministic sort — see the file-header note on why this is
    // bookkeeping, not "the algorithm".
    sort_candidates_by_score(gpu_candidates);

    // Re-upload the SORTED candidates, compute the all-pairs IoU matrix on
    // the GPU (the genuinely parallel half of NMS — kernels.cu KERNEL 5),
    // then run the inherently-sequential greedy scan on the host.
    Detection* d_sorted = nullptr;
    float* d_iou = nullptr;
    std::vector<float> iou_matrix;
    if (n_pre_nms > 0) {
        CUDA_CHECK(cudaMalloc(&d_sorted, static_cast<size_t>(n_pre_nms) * sizeof(Detection)));
        CUDA_CHECK(cudaMemcpy(d_sorted, gpu_candidates.data(),
                              static_cast<size_t>(n_pre_nms) * sizeof(Detection), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_iou, static_cast<size_t>(n_pre_nms) * n_pre_nms * sizeof(float)));
        launch_iou_matrix(d_sorted, n_pre_nms, d_iou);
        iou_matrix.resize(static_cast<size_t>(n_pre_nms) * n_pre_nms);
        CUDA_CHECK(cudaMemcpy(iou_matrix.data(), d_iou, iou_matrix.size() * sizeof(float), cudaMemcpyDeviceToHost));
    }
    std::vector<Detection> gpu_kept = greedy_nms_from_matrix(gpu_candidates, iou_matrix);
    const int n_post_nms = static_cast<int>(gpu_kept.size());

    // Keypoint extraction on the survivors only.
    Detection* d_kept = nullptr;
    if (n_post_nms > 0) {
        CUDA_CHECK(cudaMalloc(&d_kept, static_cast<size_t>(n_post_nms) * sizeof(Detection)));
        CUDA_CHECK(cudaMemcpy(d_kept, gpu_kept.data(),
                              static_cast<size_t>(n_post_nms) * sizeof(Detection), cudaMemcpyHostToDevice));
        launch_keypoint_extract(d_kept, n_post_nms, d_head_out);
        CUDA_CHECK(cudaMemcpy(gpu_kept.data(), d_kept,
                              static_cast<size_t>(n_post_nms) * sizeof(Detection), cudaMemcpyDeviceToHost));
    }

    // Download the intermediate tensors for the stage-wise VERIFY gate.
    std::vector<float> gpu_net_chw(3 * kNetH * kNetW);
    std::vector<float> gpu_conv1(static_cast<size_t>(kConv1Out) * kConv1OutH * kConv1OutW);
    std::vector<float> gpu_conv2(static_cast<size_t>(kConv2Out) * kConv2OutH * kConv2OutW);
    std::vector<float> gpu_head(static_cast<size_t>(kHeadOut) * kHeadOutH * kHeadOutW);
    CUDA_CHECK(cudaMemcpy(gpu_net_chw.data(), d_net_chw, gpu_net_chw.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_conv1.data(), d_conv1_out, gpu_conv1.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_conv2.data(), d_conv2_out, gpu_conv2.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_head.data(), d_head_out, gpu_head.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // ======================= CPU PATH (the oracle) ===========================
    CpuTimer cpu_timer;
    cpu_timer.begin();

    std::vector<float> cpu_net_chw(3 * kNetH * kNetW);
    preprocess_cpu(src_hwc.data(), cpu_net_chw.data());

    std::vector<float> cpu_conv1(static_cast<size_t>(kConv1Out) * kConv1OutH * kConv1OutW);
    conv2d_cpu(cpu_net_chw.data(), &weights.conv1_w[0][0][0][0], &weights.conv1_b[0], cpu_conv1.data(),
              kConv1In, kConv1Out, kNetH, kNetW, kConv1K, kConv1Stride, kConv1Pad,
              kConv1OutH, kConv1OutW, /*relu=*/true);

    std::vector<float> cpu_conv2(static_cast<size_t>(kConv2Out) * kConv2OutH * kConv2OutW);
    conv2d_cpu(cpu_conv1.data(), &weights.conv2_w[0][0][0][0], &weights.conv2_b[0], cpu_conv2.data(),
              kConv2In, kConv2Out, kConv1OutH, kConv1OutW, kConv2K, kConv2Stride, kConv2Pad,
              kConv2OutH, kConv2OutW, /*relu=*/true);

    std::vector<float> cpu_head(static_cast<size_t>(kHeadOut) * kHeadOutH * kHeadOutW);
    conv2d_cpu(cpu_conv2.data(), &weights.head_w[0][0][0][0], &weights.head_b[0], cpu_head.data(),
              kHeadIn, kHeadOut, kConv2OutH, kConv2OutW, kHeadK, kHeadStride, kHeadPad,
              kHeadOutH, kHeadOutW, /*relu=*/false);

    std::vector<int> cpu_best_class(static_cast<size_t>(ncells));
    std::vector<float> cpu_best_score(static_cast<size_t>(ncells));
    argmax_decode_cpu(cpu_head.data(), cpu_best_class.data(), cpu_best_score.data());

    DetectionList cpu_candidates{};
    threshold_box_decode_cpu(cpu_best_class.data(), cpu_best_score.data(), cpu_head.data(), &cpu_candidates);

    DetectionList cpu_kept{};
    nms_cpu(&cpu_candidates, kNmsIouThreshold, &cpu_kept);
    keypoint_extract_cpu(&cpu_kept, cpu_head.data());

    const double cpu_ms = cpu_timer.end_ms();

    // ======================= VERIFY STAGE =====================================
    // Tolerance justification: at most 27 chained FP32 multiply-adds per
    // output element (conv1: Cin*K*K = 3*3*3 = 27; conv2 and the head are
    // shallower), plus a resize's 4-tap bilinear blend — orders of
    // magnitude below where FP32 rounding differences become visible at
    // 1e-3 relative. See THEORY.md "How we verify correctness" for the
    // measured worst-case numbers.
    bool verify_pass = true;
    const float d_pre  = max_rel_diff(gpu_net_chw.data(), cpu_net_chw.data(), static_cast<int>(gpu_net_chw.size()));
    const float d_c1   = max_rel_diff(gpu_conv1.data(), cpu_conv1.data(), static_cast<int>(gpu_conv1.size()));
    const float d_c2   = max_rel_diff(gpu_conv2.data(), cpu_conv2.data(), static_cast<int>(gpu_conv2.size()));
    const float d_head = max_rel_diff(gpu_head.data(), cpu_head.data(), static_cast<int>(gpu_head.size()));
    if (d_pre > kStageTol || d_c1 > kStageTol || d_c2 > kStageTol || d_head > kStageTol) verify_pass = false;

    std::printf("[info] verify: worst rel. diff  preprocess=%.3e  conv1=%.3e  conv2=%.3e  head=%.3e  (tol %.1e)\n",
               static_cast<double>(d_pre), static_cast<double>(d_c1),
               static_cast<double>(d_c2), static_cast<double>(d_head), static_cast<double>(kStageTol));

    // Candidate/detection COUNTS must match exactly — an integer, not a
    // tolerance, check: any mismatch here is a genuine indexing/threshold/
    // NMS bug, not FP32 rounding (CLAUDE.md §5 gate, applied to decode too).
    if (n_pre_nms != cpu_candidates.count) verify_pass = false;
    if (n_post_nms != cpu_kept.count) verify_pass = false;
    std::printf("[info] verify: candidate counts  GPU pre-NMS=%d post-NMS=%d | CPU pre-NMS=%d post-NMS=%d\n",
               n_pre_nms, n_post_nms, cpu_candidates.count, cpu_kept.count);

    // Field-by-field comparison of the final, matched-order detections
    // (both paths sort identically — see the file-header note).
    float worst_det_diff = 0.0f;
    if (n_post_nms == cpu_kept.count) {
        for (int i = 0; i < n_post_nms; ++i) {
            const Detection& g = gpu_kept[static_cast<size_t>(i)];
            const Detection& c = cpu_kept.items[i];
            if (g.class_id != c.class_id) verify_pass = false;
            const float diffs[6] = {
                std::fabs(g.x0 - c.x0), std::fabs(g.y0 - c.y0),
                std::fabs(g.x1 - c.x1), std::fabs(g.y1 - c.y1),
                std::fabs(g.kp_x - c.kp_x), std::fabs(g.kp_y - c.kp_y)
            };
            for (float d : diffs) {
                if (d > worst_det_diff) worst_det_diff = d;
                if (d > kDetFieldAbsTol) verify_pass = false;
            }
        }
    }
    std::printf("[info] verify: worst final-detection field abs diff = %.3e px (tol %.1e px)\n",
               static_cast<double>(worst_det_diff), static_cast<double>(kDetFieldAbsTol));
    std::printf("[time] pipeline: CPU reference %.2f ms | GPU kernels (conv chain only) %.3f ms (teaching artifact, single-shot)\n",
               cpu_ms, static_cast<double>(conv_pipeline_ms));
    std::printf("VERIFY: %s (GPU matches CPU reference at every pipeline stage within documented tolerance)\n",
               verify_pass ? "PASS" : "FAIL");
    if (!verify_pass) {
        std::printf("RESULT: FAIL (GPU/CPU pipeline disagreement — fix before trusting any detection)\n");
        return 1;
    }

    // ======================= GROUND-TRUTH GATE ================================
    // Greedy one-to-one matching: for each ground-truth object (in file
    // order), pick the closest UNUSED same-class GPU detection; a match
    // requires BOTH a tight center distance AND a minimum IoU (kernels.cuh
    // constants converted to network-input space via kNetScale). Any
    // detection never claimed as a match is a false positive.
    std::vector<bool> det_used(static_cast<size_t>(n_post_nms), false);
    int matched = 0;
    float worst_center_err = 0.0f, worst_iou = 1.0f;
    for (const GtObject& g : gt) {
        const float gx0 = static_cast<float>(g.x0) * kNetScale;
        const float gy0 = static_cast<float>(g.y0) * kNetScale;
        const float gx1 = static_cast<float>(g.x0 + g.w) * kNetScale;
        const float gy1 = static_cast<float>(g.y0 + g.h) * kNetScale;
        const float gcx = (gx0 + gx1) * 0.5f, gcy = (gy0 + gy1) * 0.5f;

        int best_i = -1;
        float best_dist2 = 0.0f;
        for (int i = 0; i < n_post_nms; ++i) {
            if (det_used[static_cast<size_t>(i)]) continue;
            const Detection& d = gpu_kept[static_cast<size_t>(i)];
            if (d.class_id != g.class_id) continue;
            const float dcx = (d.x0 + d.x1) * 0.5f, dcy = (d.y0 + d.y1) * 0.5f;
            const float dist2 = (dcx - gcx) * (dcx - gcx) + (dcy - gcy) * (dcy - gcy);
            if (best_i < 0 || dist2 < best_dist2) { best_i = i; best_dist2 = dist2; }
        }
        if (best_i < 0) continue;   // no same-class detection left at all

        const Detection& d = gpu_kept[static_cast<size_t>(best_i)];
        const float center_err = std::sqrt(best_dist2);
        const float iou = iou_net(d.x0, d.y0, d.x1, d.y1, gx0, gy0, gx1, gy1);
        if (center_err <= kMatchCenterTolNetPx && iou >= kMatchIouMinForMatch) {
            det_used[static_cast<size_t>(best_i)] = true;
            matched++;
            if (center_err > worst_center_err) worst_center_err = center_err;
            if (iou < worst_iou) worst_iou = iou;
        }
    }
    int false_positives = 0;
    for (bool used : det_used) if (!used) false_positives++;

    const float reduction_factor = n_post_nms > 0
        ? static_cast<float>(n_pre_nms) / static_cast<float>(n_post_nms)
        : 0.0f;

    std::printf("[info] groundtruth: matched %d/%d objects | worst center error %.2f net-px (tol %.1f) | worst IoU %.3f (min %.2f)\n",
               matched, static_cast<int>(gt.size()), static_cast<double>(worst_center_err),
               static_cast<double>(kMatchCenterTolNetPx), static_cast<double>(worst_iou),
               static_cast<double>(kMatchIouMinForMatch));
    std::printf("[info] groundtruth: false positives=%d (bound <=%d) | NMS reduction %d->%d = %.1fx (min %.1fx)\n",
               false_positives, kMaxFalsePositives, n_pre_nms, n_post_nms,
               static_cast<double>(reduction_factor), static_cast<double>(kMinNmsReductionFactor));

    const bool groundtruth_pass = (matched == static_cast<int>(gt.size()))
        && (false_positives <= kMaxFalsePositives)
        && (reduction_factor >= kMinNmsReductionFactor);
    std::printf("GROUNDTRUTH: %s (%d/%d objects detected within tolerance, false positives within bound, NMS reduction within bound)\n",
               groundtruth_pass ? "PASS" : "FAIL", matched, static_cast<int>(gt.size()));

    // ======================= ARTIFACTS ========================================
    const std::string out_dir = resolve_out_dir(argv[0]);
    bool artifact_ok = ensure_dir(out_dir);

    if (artifact_ok) {
        // ---- detections.pgm: grayscale render + burned-in boxes/keypoints ---
        std::vector<uint8_t> gray(static_cast<size_t>(kSrcW) * kSrcH);
        for (int y = 0; y < kSrcH; ++y)
            for (int x = 0; x < kSrcW; ++x) {
                const uint8_t* p = &src_hwc[static_cast<size_t>(y * kSrcW + x) * 3];
                // Simple average, not perceptual luma weights — a
                // documented simplification (THEORY.md "Numerical
                // considerations"): this artifact is for visual sanity-
                // checking, not photometric accuracy.
                gray[static_cast<size_t>(y * kSrcW + x)] =
                    static_cast<uint8_t>((p[0] + p[1] + p[2]) / 3);
            }

        auto set_px = [&](int x, int y, uint8_t v) {
            if (x >= 0 && x < kSrcW && y >= 0 && y < kSrcH) gray[static_cast<size_t>(y * kSrcW + x)] = v;
        };
        for (int i = 0; i < n_post_nms; ++i) {
            const Detection& d = gpu_kept[static_cast<size_t>(i)];
            // Map NETWORK-input pixel coords back to SOURCE-image pixel
            // coords for drawing (divide by kNetScale — the resize's
            // forward scale factor; see kernels.cuh SECTION 1).
            const int sx0 = static_cast<int>(d.x0 / kNetScale + 0.5f);
            const int sy0 = static_cast<int>(d.y0 / kNetScale + 0.5f);
            const int sx1 = static_cast<int>(d.x1 / kNetScale + 0.5f);
            const int sy1 = static_cast<int>(d.y1 / kNetScale + 0.5f);
            for (int x = sx0; x <= sx1; ++x) { set_px(x, sy0, 255); set_px(x, sy1, 255); }
            for (int y = sy0; y <= sy1; ++y) { set_px(sx0, y, 255); set_px(sx1, y, 255); }
            const int kpx = static_cast<int>(d.kp_x / kNetScale + 0.5f);
            const int kpy = static_cast<int>(d.kp_y / kNetScale + 0.5f);
            for (int t = -2; t <= 2; ++t) { set_px(kpx + t, kpy, 0); set_px(kpx, kpy + t, 0); }
        }

        std::ofstream pgm(out_dir + "/detections.pgm", std::ios::binary);
        artifact_ok = pgm.is_open();
        if (artifact_ok) {
            pgm << "P5\n" << kSrcW << " " << kSrcH << "\n255\n";
            pgm.write(reinterpret_cast<const char*>(gray.data()), static_cast<std::streamsize>(gray.size()));
        }
    }

    if (artifact_ok) {
        // ---- detections.csv: one row per surviving GPU-path detection ------
        std::ofstream csv(out_dir + "/detections.csv");
        artifact_ok = csv.is_open();
        if (artifact_ok) {
            csv << "class_id,class_name,score,box_x0_px,box_y0_px,box_x1_px,box_y1_px,kp_x_px,kp_y_px\n";
            for (int i = 0; i < n_post_nms; ++i) {
                const Detection& d = gpu_kept[static_cast<size_t>(i)];
                csv << d.class_id << ',' << class_name(d.class_id) << ',' << d.score << ','
                    << (d.x0 / kNetScale) << ',' << (d.y0 / kNetScale) << ','
                    << (d.x1 / kNetScale) << ',' << (d.y1 / kNetScale) << ','
                    << (d.kp_x / kNetScale) << ',' << (d.kp_y / kNetScale) << '\n';
            }
        }
    }

    if (artifact_ok)
        std::printf("ARTIFACT: wrote demo/out/detections.pgm and demo/out/detections.csv\n");
    else
        std::printf("ARTIFACT: FAILED to write demo/out/detections.pgm and/or demo/out/detections.csv\n");

    // ---- cleanup --------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_src_hwc));   CUDA_CHECK(cudaFree(d_net_chw));
    CUDA_CHECK(cudaFree(d_conv1_w));   CUDA_CHECK(cudaFree(d_conv1_b));   CUDA_CHECK(cudaFree(d_conv1_out));
    CUDA_CHECK(cudaFree(d_conv2_w));   CUDA_CHECK(cudaFree(d_conv2_b));   CUDA_CHECK(cudaFree(d_conv2_out));
    CUDA_CHECK(cudaFree(d_head_w));    CUDA_CHECK(cudaFree(d_head_b));    CUDA_CHECK(cudaFree(d_head_out));
    CUDA_CHECK(cudaFree(d_best_class)); CUDA_CHECK(cudaFree(d_best_score));
    CUDA_CHECK(cudaFree(d_candidates)); CUDA_CHECK(cudaFree(d_count));
    if (d_sorted) CUDA_CHECK(cudaFree(d_sorted));
    if (d_iou) CUDA_CHECK(cudaFree(d_iou));
    if (d_kept) CUDA_CHECK(cudaFree(d_kept));

    const bool success = artifact_ok && groundtruth_pass;
    if (success)
        std::printf("RESULT: PASS (fallback CUDA pipeline verified end-to-end against the CPU reference and the known scene)\n");
    else
        std::printf("RESULT: FAIL (ground-truth gate or artifact write failed — see [info] lines above)\n");
    return success ? 0 : 1;
}
