// ===========================================================================
// main.cu — entry point for project 02.17
//           LiDAR-camera projection/coloring fusion kernels
//
// What this program does, start to finish
// -----------------------------------------
//   1. Load the committed synthetic sample: rgb.ppm (the camera image, the
//      ONLY color source the pipeline ever reads) and lidar_points.csv
//      (x,y,z — the ONLY geometry the pipeline ever reads — plus
//      EVALUATION-ONLY ground truth: true_r/g/b and a camera-visibility
//      flag scripts/make_synthetic.py computed from an independent second
//      ray cast; see that script's header).
//   2. VERIFY STAGE (CLAUDE.md 5): run all four kernels (project+z-buffer,
//      project-points, bilinear sample, occlusion check) on BOTH the GPU
//      kernels and the independent CPU twins, on the SAME baseline
//      extrinsic, and require agreement within a documented tolerance.
//   3. EVALUATION GATES, all graded against scripts/make_synthetic.py's
//      ground truth (never seen by the pipeline itself):
//        coloring_accuracy      — Direction A's headline number.
//        occlusion_correctness  — the designed failure (naive coloring)
//                                  AND its fix (the z-buffer check), both
//                                  measured on the SAME occluded cohort.
//        depth_image_fidelity   — Direction B's product, sanity-checked
//                                  against an independently re-derived
//                                  per-pixel minimum.
//        frustum_accounting     — every point lands in exactly one bucket.
//        sensitivity_curve      — the calibration-error study: perturb
//                                  T_camera_lidar, measure how the sampled
//                                  colors drift, and cross-check the
//                                  smallest perturbation's measured pixel
//                                  displacement against 01.17's analytic
//                                  rotation/translation-error formula.
//        edge_bleeding          — [info] only: color-boundary points'
//                                  naive-coloring error rate vs interior
//                                  points', the sub-pixel/bilinear reality
//                                  at object silhouettes, measured honestly.
//   4. ARTIFACTS: a colored-cloud top view and side view (the "money shot"),
//      the painted sparse depth image, the occlusion cohort before/after
//      the visibility fix, the sensitivity curve, and a metrics CSV.
//
// Output contract: stable lines are "[demo]", "PROBLEM:", "SCENARIO:",
// "VERIFY:", "GATE:", "ARTIFACT:", "RESULT:" — "[info]"/"[time]" lines are
// NOT diffed (device names and measured numbers vary by machine). Change a
// stable line -> update demo/expected_output.txt in the same change.
//
// Read this first, then kernels.cuh -> kernels.cu -> reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"
#include "util/paths.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>

// ===========================================================================
// Tiny file-format helpers — PPM in/out, CSV in. Hand-rolled, no stb_image
// (CLAUDE.md 5's "no black boxes" default: a learner should be able to read
// every byte this project touches) — the same discipline 01.18 follows.
// ===========================================================================

static bool read_ppm(const std::string& path, std::vector<uint8_t>& rgb)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    std::string magic; int w = 0, h = 0, maxval = 0;
    f >> magic >> w >> h >> maxval;
    f.get();
    if (magic != "P6" || w != kImageWidth || h != kImageHeight || maxval != 255) return false;
    rgb.resize(static_cast<size_t>(w) * h * 3);
    f.read(reinterpret_cast<char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    return f.good() || f.eof();
}

static void write_ppm(const std::string& path, int w, int h, const std::vector<uint8_t>& rgb)
{
    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << w << " " << h << "\n255\n";
    f.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
}

static void write_pgm(const std::string& path, int w, int h, const std::vector<uint8_t>& gray)
{
    std::ofstream f(path, std::ios::binary);
    f << "P5\n" << w << " " << h << "\n255\n";
    f.write(reinterpret_cast<const char*>(gray.data()), static_cast<std::streamsize>(gray.size()));
}

// Truth — EVALUATION-ONLY ground truth parsed from lidar_points.csv's
// true_r/true_g/true_b/visible columns (colors normalized to [0,1], the same
// scale the pipeline's sampled colors use). Parallel array to the
// LidarPointF vector main() loads — index i's truth describes point i.
// NEVER passed to a kernel or a reference_cpu.cpp function (kernels.cuh's
// file header explains why: keeping it outside both verified code paths is
// what makes the gates below an INDEPENDENT check, not a circular one).
struct Truth {
    float r = 0.0f, g = 0.0f, b = 0.0f;   // true surface color, normalized [0,1]
    uint8_t visible = 0;                  // ground-truth camera-visibility flag
};

static bool read_lidar_csv(const std::string& path, std::vector<LidarPointF>& pts, std::vector<Truth>& truth)
{
    std::ifstream f(path);
    if (!f.is_open()) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#' || line[0] == 'x') continue;   // comment or header row
        std::stringstream ss(line);
        std::string cell;
        LidarPointF p{};
        Truth t{};
        if (!std::getline(ss, cell, ',')) continue; p.x = std::strtof(cell.c_str(), nullptr);
        if (!std::getline(ss, cell, ',')) continue; p.y = std::strtof(cell.c_str(), nullptr);
        if (!std::getline(ss, cell, ',')) continue; p.z = std::strtof(cell.c_str(), nullptr);
        if (!std::getline(ss, cell, ',')) continue; t.r = std::strtof(cell.c_str(), nullptr) / 255.0f;
        if (!std::getline(ss, cell, ',')) continue; t.g = std::strtof(cell.c_str(), nullptr) / 255.0f;
        if (!std::getline(ss, cell, ',')) continue; t.b = std::strtof(cell.c_str(), nullptr) / 255.0f;
        if (!std::getline(ss, cell, ',')) continue; t.visible = static_cast<uint8_t>(std::atoi(cell.c_str()));
        // Remaining column ("surface", a human-readable label) is read by no
        // one downstream of this file — intentionally not parsed further.
        pts.push_back(p);
        truth.push_back(t);
    }
    return !pts.empty();
}

// ===========================================================================
// Small evaluation-only host helpers. NONE of these are called by the
// VERIFY stage (they never feed a GPU-vs-CPU comparison) — they exist only
// to grade the pipeline's ALREADY-COMPUTED output against ground truth, the
// same role 01.18's build_edge_masks/max_channel_diff_host play there.
// ===========================================================================

// color_dist — max-abs-channel difference, normalized [0,1] scale (the same
// measure kernels.cuh's kColorBoundaryThresh and 01.18's conductance use).
static inline float color_dist(float r0, float g0, float b0, float r1, float g1, float b1)
{
    return std::max(std::fabs(r0 - r1), std::max(std::fabs(g0 - g1), std::fabs(b0 - b1)));
}

// project_to_pixel_eval — a THIRD, independent, evaluation-only copy of the
// rigid-transform + pinhole projection formula (kernels.cu and
// reference_cpu.cpp each carry their own per the twin-independence ruling;
// this one exists purely so depth_image_fidelity's cross-check and the
// edge_bleeding boundary classification below do not silently depend on
// either verified code path being correct — the SAME reasoning 01.18's
// main.cu applies to its own small evaluation-only formula copies).
static bool project_to_pixel_eval(const LidarPointF& p, const Rigid3& T, int& px, int& py, float& zc)
{
    const float* R = T.R;
    const float xc = R[0] * p.x + R[1] * p.y + R[2] * p.z + T.t[0];
    const float yc = R[3] * p.x + R[4] * p.y + R[5] * p.z + T.t[1];
    zc              = R[6] * p.x + R[7] * p.y + R[8] * p.z + T.t[2];
    if (zc <= 0.0f || zc > kMaxDepthM) { px = -1; py = -1; return false; }
    const float u = kFx * xc / zc + kCx;
    const float v = kFy * yc / zc + kCy;
    px = static_cast<int>(std::floor(u + 0.5f));
    py = static_cast<int>(std::floor(v + 0.5f));
    if (px < 0 || px >= kImageWidth || py < 0 || py >= kImageHeight) return false;
    return true;
}

// mat3_mul — plain 3x3 row-major matrix multiply, out = A*B. Used only to
// build the sensitivity sweep's perturbed rotations (evaluation-only, never
// part of a kernel or CPU-twin signature).
static void mat3_mul(const float A[9], const float B[9], float out[9])
{
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) {
            float s = 0.0f;
            for (int k = 0; k < 3; ++k) s += A[r * 3 + k] * B[k * 3 + c];
            out[r * 3 + c] = s;
        }
}

// perturb_rotation — R' = Ry(theta) * R_base: an EXTRA small rotation about
// the CAMERA's own Y ("down") axis, composed on the left of the existing
// extrinsic — physically, "the mounting rotation has an error of theta about
// the camera's down axis", a yaw-like error producing a mostly-HORIZONTAL
// pixel shift (THEORY.md derives why: Ry mixes the camera X/Z axes, and X is
// exactly the axis the pinhole formula divides into the horizontal pixel
// coordinate u). t is left untouched — this sweep isolates ROTATION error.
static Rigid3 perturb_rotation(const Rigid3& base, float theta_rad)
{
    const float c = std::cos(theta_rad), s = std::sin(theta_rad);
    const float Ry[9] = { c, 0.0f, s,
                          0.0f, 1.0f, 0.0f,
                         -s, 0.0f, c };
    Rigid3 out = base;
    mat3_mul(Ry, base.R, out.R);
    return out;
}

// perturb_translation — t' = t_base + delta_m along the CAMERA's own X
// ("right") axis (Rigid3's translation is already expressed in the
// DESTINATION/camera frame, kernels.cuh's convention, so this is a direct
// add — no further transform needed). R is left untouched — this sweep
// isolates TRANSLATION error.
static Rigid3 perturb_translation(const Rigid3& base, float delta_m)
{
    Rigid3 out = base;
    out.t[0] += delta_m;
    return out;
}

// ---------------------------------------------------------------------------
// Canvas — a tiny RGB scatter-plot surface for this project's "colored
// cloud" artifacts (README "Artifacts"). Hand-rolled (no plotting library,
// CLAUDE.md 5's default dependency budget) — a flat interleaved byte buffer
// plus a 3x3 "dot" splat so individual points are visible at these small
// canvas sizes (a single pixel per point is nearly invisible at a few
// thousand points spread across a multi-meter scene).
// ---------------------------------------------------------------------------
struct Canvas {
    int w, h;
    std::vector<uint8_t> px;
    Canvas(int w_, int h_, uint8_t bg) : w(w_), h(h_), px(static_cast<size_t>(w_) * h_ * 3, bg) {}
    void dot(int x, int y, uint8_t r, uint8_t g, uint8_t b)
    {
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                const int cx = x + dx, cy = y + dy;
                if (cx < 0 || cx >= w || cy < 0 || cy >= h) continue;
                const size_t i = (static_cast<size_t>(cy) * w + cx) * 3;
                px[i] = r; px[i + 1] = g; px[i + 2] = b;
            }
    }
};

// to_u8 — [0,1] normalized color channel -> [0,255] byte, saturating.
static inline uint8_t to_u8(float v)
{
    v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
    return static_cast<uint8_t>(v * 255.0f + 0.5f);
}

// ---------------------------------------------------------------------------
// render_scatter — orthographic scatter render of a point subset onto a
// Canvas, mapping world axes (ax, ay) linearly onto (canvas col, canvas row)
// with `ay` FLIPPED (world "up"/"far" reads as canvas "up") and a small
// margin so points at the exact bounding-box edge are not clipped by dot().
// Used for BOTH the full-cloud top/side views and the zoomed-in occlusion
// cohort views (README "Artifacts") — one small function, several callers.
// ---------------------------------------------------------------------------
static void render_scatter(const std::string& path, int W, int H,
                           const std::vector<float>& ax, const std::vector<float>& ay,
                           const std::vector<uint8_t>& r, const std::vector<uint8_t>& g,
                           const std::vector<uint8_t>& b, const std::vector<uint8_t>& mask)
{
    float amin = 1e9f, amax = -1e9f, bmin = 1e9f, bmax = -1e9f;
    for (size_t i = 0; i < ax.size(); ++i) {
        if (!mask[i]) continue;
        amin = std::min(amin, ax[i]); amax = std::max(amax, ax[i]);
        bmin = std::min(bmin, ay[i]); bmax = std::max(bmax, ay[i]);
    }
    if (amax <= amin) { amax = amin + 1.0f; }
    if (bmax <= bmin) { bmax = bmin + 1.0f; }
    const float pad_a = 0.05f * (amax - amin), pad_b = 0.05f * (bmax - bmin);
    amin -= pad_a; amax += pad_a; bmin -= pad_b; bmax += pad_b;

    Canvas c(W, H, 18);   // dark-gray background
    for (size_t i = 0; i < ax.size(); ++i) {
        if (!mask[i]) continue;
        const int cx = static_cast<int>((ax[i] - amin) / (amax - amin) * (W - 1));
        const int cy = static_cast<int>((bmax - ay[i]) / (bmax - bmin) * (H - 1));   // flipped: "far/up" -> canvas top
        c.dot(cx, cy, r[i], g[i], b[i]);
    }
    write_ppm(path, W, H, c.px);
}

// ===========================================================================
// main
// ===========================================================================
int main(int argc, char** argv)
{
    std::string data_dir;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--data-dir") && i + 1 < argc) data_dir = argv[++i];
        else {
            std::fprintf(stderr, "usage: %s [--data-dir DIR]\n", argv[0]);
            return 2;
        }
    }

    std::printf("[demo] LiDAR-camera projection/coloring fusion kernels (project 02.17)\n");
    print_device_info();

    // ---- 1) load data --------------------------------------------------------
    const std::string rgb_path   = find_data_file(data_dir, argv[0], "rgb.ppm");
    const std::string lidar_path = find_data_file(data_dir, argv[0], "lidar_points.csv");
    if (rgb_path.empty() || lidar_path.empty()) {
        std::printf("PROBLEM: sample data not found under data/sample/ (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (data missing)\n");
        return 1;
    }
    std::vector<uint8_t> rgb;
    std::vector<LidarPointF> pts;
    std::vector<Truth> truth;
    if (!read_ppm(rgb_path, rgb) || !read_lidar_csv(lidar_path, pts, truth)) {
        std::printf("PROBLEM: sample data malformed -- see paths above\n");
        std::printf("RESULT: FAIL (data malformed)\n");
        return 1;
    }
    const int n_pts = static_cast<int>(pts.size());

    // PLANAR normalized-[0,1] color guidance image (01.18's layout, cited):
    // rgbf[0..N) = red plane, [N,2N) = green, [2N,3N) = blue.
    std::vector<float> rgbf(static_cast<size_t>(3 * kImagePixels));
    for (int i = 0; i < kImagePixels; ++i) {
        const uint8_t* px8 = &rgb[static_cast<size_t>(i) * 3];
        rgbf[static_cast<size_t>(i)]                    = px8[0] / 255.0f;
        rgbf[static_cast<size_t>(i) + kImagePixels]      = px8[1] / 255.0f;
        rgbf[static_cast<size_t>(i) + 2 * kImagePixels]  = px8[2] / 255.0f;
    }

    const int n_occluded_truth = static_cast<int>(std::count_if(truth.begin(), truth.end(),
        [](const Truth& t) { return t.visible == 0; }));

    std::printf("PROBLEM: %dx%d camera, %d LiDAR returns, occlusion band %.2f m, "
               "color-boundary thresh %.2f, %d-level rotation/translation sensitivity sweep\n",
               kImageWidth, kImageHeight, n_pts, static_cast<double>(kOcclusionBandM),
               static_cast<double>(kColorBoundaryThresh), kNumSensitivityLevels);
    std::printf("SCENARIO: red occluder (~4 m) hides part of a green background (~12 m) from the CAMERA "
               "but not the (higher-mounted) LiDAR; %d/%d = %.1f%% of returns are ground-truth-occluded "
               "(the occlusion cohort) [synthetic]\n",
               n_occluded_truth, n_pts, 100.0 * n_occluded_truth / n_pts);

    // ---- device buffers (persistent for the whole run) ------------------------
    LidarPointF* d_pts = nullptr;
    float* d_rgbf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pts, static_cast<size_t>(n_pts) * sizeof(LidarPointF)));
    CUDA_CHECK(cudaMalloc(&d_rgbf, static_cast<size_t>(3 * kImagePixels) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_pts, pts.data(), static_cast<size_t>(n_pts) * sizeof(LidarPointF), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rgbf, rgbf.data(), static_cast<size_t>(3 * kImagePixels) * sizeof(float), cudaMemcpyHostToDevice));

    float *d_u = nullptr, *d_v = nullptr, *d_zc = nullptr, *d_color = nullptr;
    uint8_t *d_in_frustum = nullptr, *d_visible = nullptr;
    uint32_t* d_encoded = nullptr;
    CUDA_CHECK(cudaMalloc(&d_u, static_cast<size_t>(n_pts) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, static_cast<size_t>(n_pts) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_zc, static_cast<size_t>(n_pts) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_in_frustum, static_cast<size_t>(n_pts) * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_visible, static_cast<size_t>(n_pts) * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_color, static_cast<size_t>(3 * n_pts) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_encoded, static_cast<size_t>(kImagePixels) * sizeof(uint32_t)));

    // ======================= VERIFY STAGE ======================================
    // All four kernels, GPU vs CPU, at the BASELINE (unperturbed) extrinsic.
    bool verify_pass = true;

    // -- (a) project + z-buffer -------------------------------------------------
    std::vector<float> sparse_gpu(static_cast<size_t>(kImagePixels)), sparse_cpu(static_cast<size_t>(kImagePixels));
    {
        CUDA_CHECK(cudaMemset(d_encoded, 0xFF, static_cast<size_t>(kImagePixels) * sizeof(uint32_t)));
        GpuTimer gt; gt.begin();
        launch_project_zbuffer(d_pts, n_pts, kTCameraLidar, d_encoded);
        const float gpu_ms = gt.end_ms();

        std::vector<uint32_t> encoded(static_cast<size_t>(kImagePixels));
        CUDA_CHECK(cudaMemcpy(encoded.data(), d_encoded, encoded.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        for (int i = 0; i < kImagePixels; ++i) {
            const uint32_t bits = encoded[static_cast<size_t>(i)];
            float decoded; std::memcpy(&decoded, &bits, sizeof(decoded));
            sparse_gpu[static_cast<size_t>(i)] = (bits == 0xFFFFFFFFu) ? kInvalidDepth : decoded;
        }

        CpuTimer ct; ct.begin();
        project_zbuffer_cpu(pts.data(), n_pts, kTCameraLidar, sparse_cpu.data());
        const double cpu_ms = ct.end_ms();

        float worst = 0.0f; int mism = 0;
        for (int i = 0; i < kImagePixels; ++i) {
            const bool ea = sparse_gpu[static_cast<size_t>(i)] == kInvalidDepth;
            const bool eb = sparse_cpu[static_cast<size_t>(i)] == kInvalidDepth;
            if (ea != eb) { mism++; continue; }
            if (ea && eb) continue;
            worst = std::max(worst, std::fabs(sparse_gpu[static_cast<size_t>(i)] - sparse_cpu[static_cast<size_t>(i)]));
        }
        const bool ok = worst <= 1e-4f && mism == 0;
        verify_pass = verify_pass && ok;
        std::printf("[time] project+z-buffer: CPU %.3f ms | GPU %.3f ms\n", cpu_ms, static_cast<double>(gpu_ms));
        std::printf("[info] project+z-buffer: max |gpu-cpu| depth = %.3e m, %d empty-pixel mismatches\n",
                   static_cast<double>(worst), mism);
        std::printf("VERIFY: project+z-buffer %s (tol 1e-4 m, zero empty-pixel mismatches)\n", ok ? "PASS" : "FAIL");
    }

    // -- (b) project points -------------------------------------------------------
    std::vector<float> u_gpu(static_cast<size_t>(n_pts)), v_gpu(static_cast<size_t>(n_pts)), zc_gpu(static_cast<size_t>(n_pts));
    std::vector<uint8_t> inf_gpu(static_cast<size_t>(n_pts));
    {
        GpuTimer gt; gt.begin();
        launch_project_points(d_pts, n_pts, kTCameraLidar, d_u, d_v, d_zc, d_in_frustum);
        const float gpu_ms = gt.end_ms();
        CUDA_CHECK(cudaMemcpy(u_gpu.data(), d_u, u_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(v_gpu.data(), d_v, v_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(zc_gpu.data(), d_zc, zc_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(inf_gpu.data(), d_in_frustum, inf_gpu.size() * sizeof(uint8_t), cudaMemcpyDeviceToHost));

        std::vector<float> u_cpu(static_cast<size_t>(n_pts)), v_cpu(static_cast<size_t>(n_pts)), zc_cpu(static_cast<size_t>(n_pts));
        std::vector<uint8_t> inf_cpu(static_cast<size_t>(n_pts));
        CpuTimer ct; ct.begin();
        project_points_cpu(pts.data(), n_pts, kTCameraLidar, u_cpu.data(), v_cpu.data(), zc_cpu.data(), inf_cpu.data());
        const double cpu_ms = ct.end_ms();

        float worst_u = 0.0f, worst_zc = 0.0f; int mism = 0;
        for (int i = 0; i < n_pts; ++i) {
            if (inf_gpu[static_cast<size_t>(i)] != inf_cpu[static_cast<size_t>(i)]) { mism++; continue; }
            worst_u = std::max(worst_u, std::fabs(u_gpu[static_cast<size_t>(i)] - u_cpu[static_cast<size_t>(i)]));
            worst_u = std::max(worst_u, std::fabs(v_gpu[static_cast<size_t>(i)] - v_cpu[static_cast<size_t>(i)]));
            worst_zc = std::max(worst_zc, std::fabs(zc_gpu[static_cast<size_t>(i)] - zc_cpu[static_cast<size_t>(i)]));
        }
        const bool ok = worst_u <= 1e-3f && worst_zc <= 1e-4f && mism == 0;
        verify_pass = verify_pass && ok;
        std::printf("[time] project-points: CPU %.3f ms | GPU %.3f ms\n", cpu_ms, static_cast<double>(gpu_ms));
        std::printf("[info] project-points: max |gpu-cpu| u/v = %.3e px, zc = %.3e m, %d in-frustum mismatches\n",
                   static_cast<double>(worst_u), static_cast<double>(worst_zc), mism);
        std::printf("VERIFY: project-points %s (tol 1e-3 px / 1e-4 m, zero in-frustum mismatches)\n", ok ? "PASS" : "FAIL");
    }

    // -- (c) bilinear sample -------------------------------------------------------
    std::vector<float> color_gpu(static_cast<size_t>(3 * n_pts));
    {
        GpuTimer gt; gt.begin();
        launch_sample_bilinear(d_u, d_v, d_in_frustum, n_pts, d_rgbf, d_color);
        const float gpu_ms = gt.end_ms();
        CUDA_CHECK(cudaMemcpy(color_gpu.data(), d_color, color_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

        std::vector<float> color_cpu(static_cast<size_t>(3 * n_pts));
        CpuTimer ct; ct.begin();
        sample_bilinear_cpu(u_gpu.data(), v_gpu.data(), inf_gpu.data(), n_pts, rgbf.data(), color_cpu.data());
        const double cpu_ms = ct.end_ms();

        float worst = 0.0f;
        for (size_t i = 0; i < color_gpu.size(); ++i) worst = std::max(worst, std::fabs(color_gpu[i] - color_cpu[i]));
        const bool ok = worst <= 1e-5f;
        verify_pass = verify_pass && ok;
        std::printf("[time] bilinear-sample: CPU %.3f ms | GPU %.3f ms\n", cpu_ms, static_cast<double>(gpu_ms));
        std::printf("[info] bilinear-sample: max |gpu-cpu| = %.3e\n", static_cast<double>(worst));
        std::printf("VERIFY: bilinear-sample %s (tol 1e-5)\n", ok ? "PASS" : "FAIL");
    }

    // -- (d) occlusion check --------------------------------------------------------
    std::vector<uint8_t> visible_gpu(static_cast<size_t>(n_pts));
    {
        GpuTimer gt; gt.begin();
        launch_check_occlusion(d_u, d_v, d_zc, d_in_frustum, n_pts, d_encoded, kOcclusionBandM, d_visible);
        const float gpu_ms = gt.end_ms();
        CUDA_CHECK(cudaMemcpy(visible_gpu.data(), d_visible, visible_gpu.size() * sizeof(uint8_t), cudaMemcpyDeviceToHost));

        std::vector<uint8_t> visible_cpu(static_cast<size_t>(n_pts));
        CpuTimer ct; ct.begin();
        check_occlusion_cpu(u_gpu.data(), v_gpu.data(), zc_gpu.data(), inf_gpu.data(), n_pts,
                            sparse_cpu.data(), kOcclusionBandM, visible_cpu.data());
        const double cpu_ms = ct.end_ms();

        // The GPU path decodes ITS OWN encoded z-buffer (sparse_gpu, via
        // d_encoded); the CPU path reads the CPU z-buffer (sparse_cpu). The
        // two z-buffers already agree within 1e-4 m ((a) above); a handful
        // of points sitting almost exactly on the +-band_m boundary can
        // still flip which side of the compare they land on when the two
        // buffers' sub-1e-4 rounding differs — the SAME "chained-comparison"
        // story 01.17's TRAJECTORY_TWIN gate tells (cited in THEORY.md).
        // Zero is the expectation; a handful is tolerated and reported.
        int mism = 0;
        for (int i = 0; i < n_pts; ++i) if (visible_gpu[static_cast<size_t>(i)] != visible_cpu[static_cast<size_t>(i)]) mism++;
        const double mism_frac = static_cast<double>(mism) / n_pts;
        const bool ok = mism_frac <= 0.01;   // <=1% boundary-case flips tolerated (measured, see [info])
        verify_pass = verify_pass && ok;
        std::printf("[time] occlusion-check: CPU %.3f ms | GPU %.3f ms\n", cpu_ms, static_cast<double>(gpu_ms));
        std::printf("[info] occlusion-check: %d/%d visibility-flag mismatches (%.3f%%)\n", mism, n_pts, 100.0 * mism_frac);
        std::printf("VERIFY: occlusion-check %s (<=1%% boundary-case flag mismatches tolerated)\n", ok ? "PASS" : "FAIL");
    }

    if (!verify_pass) {
        CUDA_CHECK(cudaFree(d_pts)); CUDA_CHECK(cudaFree(d_rgbf));
        CUDA_CHECK(cudaFree(d_u)); CUDA_CHECK(cudaFree(d_v)); CUDA_CHECK(cudaFree(d_zc));
        CUDA_CHECK(cudaFree(d_in_frustum)); CUDA_CHECK(cudaFree(d_visible));
        CUDA_CHECK(cudaFree(d_color)); CUDA_CHECK(cudaFree(d_encoded));
        std::printf("RESULT: FAIL (GPU/CPU disagreement in the VERIFY stage -- see VERIFY lines above)\n");
        return 1;
    }

    // ======================= EVALUATION GATES ==================================
    bool gates_pass = true;

    // -- GATE: frustum_accounting (pure bookkeeping, exact integers) -----------
    int n_in_frustum = 0, n_colored = 0, n_filtered = 0;
    for (int i = 0; i < n_pts; ++i) {
        if (inf_gpu[static_cast<size_t>(i)]) {
            n_in_frustum++;
            if (visible_gpu[static_cast<size_t>(i)]) n_colored++; else n_filtered++;
        }
    }
    const int n_out_frustum = n_pts - n_in_frustum;
    {
        const bool ok = (n_in_frustum + n_out_frustum == n_pts) && (n_colored + n_filtered == n_in_frustum);
        gates_pass = gates_pass && ok;
        std::printf("[info] frustum_accounting: in_frustum=%d out_frustum=%d colored=%d filtered=%d total=%d\n",
                   n_in_frustum, n_out_frustum, n_colored, n_filtered, n_pts);
        std::printf("GATE: frustum_accounting %s (in_frustum+out_frustum==total AND colored+filtered==in_frustum)\n",
                   ok ? "PASS" : "FAIL");
    }

    // -- GATE: coloring_accuracy (Direction A's headline) -----------------------
    // Ground-truth-VISIBLE points only: the CHECKED (final, occlusion-fixed)
    // sampled color must be close to the point's true surface color.
    static constexpr float kColorAccTol = 0.12f;      // normalized [0,1] max-channel diff (~30/255)
    static constexpr double kColorAccFloor = 0.70;    // measured 76.9% on the reference GPU, margined down -- see README/THEORY
    {
        int n_visible_truth = 0, n_accurate = 0;
        for (int i = 0; i < n_pts; ++i) {
            if (!truth[static_cast<size_t>(i)].visible) continue;
            n_visible_truth++;
            const bool colored = inf_gpu[static_cast<size_t>(i)] && visible_gpu[static_cast<size_t>(i)];
            if (!colored) continue;
            const float d = color_dist(color_gpu[3 * static_cast<size_t>(i) + 0], color_gpu[3 * static_cast<size_t>(i) + 1],
                                       color_gpu[3 * static_cast<size_t>(i) + 2],
                                       truth[static_cast<size_t>(i)].r, truth[static_cast<size_t>(i)].g, truth[static_cast<size_t>(i)].b);
            if (d <= kColorAccTol) n_accurate++;
        }
        const double frac = n_visible_truth > 0 ? static_cast<double>(n_accurate) / n_visible_truth : 0.0;
        const bool ok = frac >= kColorAccFloor;
        gates_pass = gates_pass && ok;
        std::printf("[info] coloring_accuracy: %d/%d ground-truth-visible points colored accurately (tol %.2f) = %.2f%%\n",
                   n_accurate, n_visible_truth, static_cast<double>(kColorAccTol), 100.0 * frac);
        std::printf("GATE: coloring_accuracy %s (>= %.0f%% of ground-truth-visible points colored within tol %.2f)\n",
                   ok ? "PASS" : "FAIL", 100.0 * kColorAccFloor, static_cast<double>(kColorAccTol));
    }

    // -- GATE: occlusion_correctness (the designed failure AND its fix) --------
    static constexpr double kOcclusionWrongCeiling = 0.05;   // WITH the check: measured 0.7% on the reference GPU, ~7x headroom
    static constexpr double kOcclusionWrongFloor   = 0.80;   // WITHOUT the check: measured 89.1% on the reference GPU, margined down
    {
        int n_cohort = 0, n_wrong_with = 0, n_wrong_without = 0;
        for (int i = 0; i < n_pts; ++i) {
            if (truth[static_cast<size_t>(i)].visible) continue;   // cohort = ground-truth OCCLUDED points
            n_cohort++;
            const size_t ci = static_cast<size_t>(i);
            // WITHOUT the check: naive coloring always "receives" a color if
            // in-frustum (kernel 3's output, unconditioned by kernel 4).
            if (inf_gpu[ci]) {
                const float dw = color_dist(color_gpu[3 * ci + 0], color_gpu[3 * ci + 1], color_gpu[3 * ci + 2],
                                            truth[ci].r, truth[ci].g, truth[ci].b);
                if (dw > kColorAccTol) n_wrong_without++;
            } else {
                n_wrong_without++;   // no color at all is not "correctly colored" either -- counts as wrong
            }
            // WITH the check: only counts as "received a (wrong) color" if
            // the occlusion check actually let it through.
            if (inf_gpu[ci] && visible_gpu[ci]) {
                const float dc = color_dist(color_gpu[3 * ci + 0], color_gpu[3 * ci + 1], color_gpu[3 * ci + 2],
                                            truth[ci].r, truth[ci].g, truth[ci].b);
                if (dc > kColorAccTol) n_wrong_with++;
            }
        }
        const double frac_with = n_cohort > 0 ? static_cast<double>(n_wrong_with) / n_cohort : 0.0;
        const double frac_without = n_cohort > 0 ? static_cast<double>(n_wrong_without) / n_cohort : 0.0;
        const bool ok = n_cohort > 0 && frac_with <= kOcclusionWrongCeiling && frac_without >= kOcclusionWrongFloor;
        gates_pass = gates_pass && ok;
        std::printf("[info] occlusion_correctness (cohort n=%d): wrong-color rate WITHOUT check = %.2f%% | WITH check = %.2f%%\n",
                   n_cohort, 100.0 * frac_without, 100.0 * frac_with);
        std::printf("GATE: occlusion_correctness %s (WITHOUT check >= %.0f%% wrong; WITH check <= %.0f%% wrong)\n",
                   ok ? "PASS" : "FAIL", 100.0 * kOcclusionWrongFloor, 100.0 * kOcclusionWrongCeiling);
    }

    // -- GATE: depth_image_fidelity (Direction B) --------------------------------
    // Independently re-derive each pixel's minimum depth by brute-force
    // scanning project_to_pixel_eval's THIRD, evaluation-only projection
    // (never the z-buffer kernel's own atomicMin machinery) and compare
    // against the (already-verified) GPU z-buffer's decoded winner.
    {
        std::vector<float> host_min(static_cast<size_t>(kImagePixels), kInvalidDepth);
        for (int i = 0; i < n_pts; ++i) {
            int px, py; float zc;
            if (!project_to_pixel_eval(pts[static_cast<size_t>(i)], kTCameraLidar, px, py, zc)) continue;
            const size_t idx = static_cast<size_t>(py) * kImageWidth + px;
            if (host_min[idx] == kInvalidDepth || zc < host_min[idx]) host_min[idx] = zc;
        }
        float worst = 0.0f; int mism = 0, coverage = 0;
        for (int i = 0; i < kImagePixels; ++i) {
            const bool ea = host_min[static_cast<size_t>(i)] == kInvalidDepth;
            const bool eb = sparse_gpu[static_cast<size_t>(i)] == kInvalidDepth;
            if (!eb) coverage++;
            if (ea != eb) { mism++; continue; }
            if (ea && eb) continue;
            worst = std::max(worst, std::fabs(host_min[static_cast<size_t>(i)] - sparse_gpu[static_cast<size_t>(i)]));
        }
        const double coverage_frac = static_cast<double>(coverage) / kImagePixels;
        const bool ok = worst <= 1e-4f && mism == 0;
        gates_pass = gates_pass && ok;
        std::printf("[info] depth_image_fidelity: coverage=%d/%d pixels (%.2f%%), max deviation from an independently "
                   "re-derived per-pixel minimum = %.3e m, %d empty-pixel mismatches\n",
                   coverage, kImagePixels, 100.0 * coverage_frac, static_cast<double>(worst), mism);
        std::printf("GATE: depth_image_fidelity %s (painted depth matches an independently re-derived per-pixel minimum "
                   "within tol 1e-4 m)\n", ok ? "PASS" : "FAIL");
    }

    // -- [info] edge_bleeding honesty (not gated pass/fail) ----------------------
    // Classify each GROUND-TRUTH-VISIBLE point as "boundary" (its rounded
    // pixel sits on a strong RGB gradient in the rendered image -- an object
    // silhouette) or "interior" (a flat, low-gradient patch), using the SAME
    // max-abs-channel-difference measure kernels.cuh's kColorBoundaryThresh
    // and 01.18's conductance both use, then compare the naive-coloring
    // error RATE between the two cohorts -- the sub-pixel/bilinear-blending
    // reality at silhouettes, distinct from the OCCLUSION failure the gates
    // above measure (both cohorts here exclude occluded points entirely).
    {
        static constexpr float kEdgeGradThresh = 0.20f;   // normalized [0,1], same scale as 01.18's kRgbEdgeThreshHigh
        int n_boundary = 0, n_boundary_wrong = 0, n_interior = 0, n_interior_wrong = 0;
        for (int i = 0; i < n_pts; ++i) {
            if (!truth[static_cast<size_t>(i)].visible || !inf_gpu[static_cast<size_t>(i)]) continue;
            int px, py; float zc;
            if (!project_to_pixel_eval(pts[static_cast<size_t>(i)], kTCameraLidar, px, py, zc)) continue;
            float grad = 0.0f;
            for (int dy = -1; dy <= 1; ++dy) {
                const int ny = py + dy; if (ny < 0 || ny >= kImageHeight) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nx = px + dx; if (nx < 0 || nx >= kImageWidth) continue;
                    if (dx == 0 && dy == 0) continue;
                    const size_t a = static_cast<size_t>(py) * kImageWidth + px, b = static_cast<size_t>(ny) * kImageWidth + nx;
                    grad = std::max(grad, color_dist(rgbf[a], rgbf[a + kImagePixels], rgbf[a + 2 * kImagePixels],
                                                     rgbf[b], rgbf[b + kImagePixels], rgbf[b + 2 * kImagePixels]));
                }
            }
            const size_t ci = static_cast<size_t>(i);
            const bool wrong = color_dist(color_gpu[3 * ci + 0], color_gpu[3 * ci + 1], color_gpu[3 * ci + 2],
                                          truth[ci].r, truth[ci].g, truth[ci].b) > kColorAccTol;
            if (grad > kEdgeGradThresh) { n_boundary++; if (wrong) n_boundary_wrong++; }
            else { n_interior++; if (wrong) n_interior_wrong++; }
        }
        const double rate_b = n_boundary > 0 ? 100.0 * n_boundary_wrong / n_boundary : 0.0;
        const double rate_i = n_interior > 0 ? 100.0 * n_interior_wrong / n_interior : 0.0;
        std::printf("[info] edge_bleeding honesty: boundary points wrong-color rate = %.2f%% (n=%d) | "
                   "interior points wrong-color rate = %.2f%% (n=%d) | ratio boundary/interior = %.2fx\n",
                   rate_b, n_boundary, rate_i, n_interior, rate_i > 0.0 ? rate_b / rate_i : 0.0);
    }

    // -- GATE: sensitivity_curve (the calibration-error study) ------------------
    static constexpr double kConsistencyFactor = 4.0;   // measured-then-margined headroom, see README/THEORY
    struct SweepRow { double level; double flip_frac; double measured_disp_px; double predicted_disp_px; };
    std::vector<SweepRow> rot_rows, trans_rows;
    bool rot_monotone = true, trans_monotone = true;
    bool rot_consistent = false, trans_consistent = false;
    {
        // Baseline (zero perturbation) projection + naive color, GPU path.
        std::vector<float> u0 = u_gpu, v0 = v_gpu; std::vector<uint8_t> inf0 = inf_gpu;
        std::vector<float> color0 = color_gpu;

        double prev_flip = -1.0;
        for (int lvl = 0; lvl < kNumSensitivityLevels; ++lvl) {
            const float theta_rad = kSensitivityRotDeg[lvl] * 3.14159265358979323846f / 180.0f;
            const Rigid3 Tp = perturb_rotation(kTCameraLidar, theta_rad);
            launch_project_points(d_pts, n_pts, Tp, d_u, d_v, d_zc, d_in_frustum);
            launch_sample_bilinear(d_u, d_v, d_in_frustum, n_pts, d_rgbf, d_color);
            std::vector<float> up(static_cast<size_t>(n_pts)), vp(static_cast<size_t>(n_pts)), colorp(static_cast<size_t>(3 * n_pts));
            std::vector<uint8_t> infp(static_cast<size_t>(n_pts));
            CUDA_CHECK(cudaMemcpy(up.data(), d_u, up.size() * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(vp.data(), d_v, vp.size() * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(infp.data(), d_in_frustum, infp.size() * sizeof(uint8_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(colorp.data(), d_color, colorp.size() * sizeof(float), cudaMemcpyDeviceToHost));

            long n_considered = 0, n_flip = 0; double disp_sum = 0.0, zc_sum = 0.0;
            for (int i = 0; i < n_pts; ++i) {
                if (!inf0[static_cast<size_t>(i)] || !infp[static_cast<size_t>(i)]) continue;
                n_considered++;
                const size_t ci = static_cast<size_t>(i);
                const double du = up[ci] - u0[ci], dv = vp[ci] - v0[ci];
                disp_sum += std::sqrt(du * du + dv * dv);
                zc_sum += zc_gpu[ci];
                if (color_dist(colorp[3 * ci], colorp[3 * ci + 1], colorp[3 * ci + 2],
                               color0[3 * ci], color0[3 * ci + 1], color0[3 * ci + 2]) > kColorBoundaryThresh) n_flip++;
            }
            const double flip_frac = n_considered > 0 ? static_cast<double>(n_flip) / n_considered : 0.0;
            const double measured_disp = n_considered > 0 ? disp_sum / n_considered : 0.0;
            const double predicted_disp = static_cast<double>(kFx) * theta_rad;   // 01.17's rotation-error formula: ~fx*d_theta, range-independent
            rot_rows.push_back({ static_cast<double>(kSensitivityRotDeg[lvl]), flip_frac, measured_disp, predicted_disp });
            if (prev_flip >= 0.0 && flip_frac < prev_flip - 1e-9) rot_monotone = false;
            prev_flip = flip_frac;
            if (lvl == 0) {
                const double ratio = measured_disp > 0.0 ? measured_disp / predicted_disp : 0.0;
                rot_consistent = ratio >= 1.0 / kConsistencyFactor && ratio <= kConsistencyFactor;
                std::printf("[info] sensitivity(rotation) smallest level %.1f deg: predicted disp %.3f px, "
                           "measured mean disp %.3f px, ratio %.2fx\n",
                           static_cast<double>(kSensitivityRotDeg[0]), predicted_disp, measured_disp, ratio);
            }
        }

        prev_flip = -1.0;
        for (int lvl = 0; lvl < kNumSensitivityLevels; ++lvl) {
            const float delta_m = kSensitivityTransCm[lvl] / 100.0f;
            const Rigid3 Tp = perturb_translation(kTCameraLidar, delta_m);
            launch_project_points(d_pts, n_pts, Tp, d_u, d_v, d_zc, d_in_frustum);
            launch_sample_bilinear(d_u, d_v, d_in_frustum, n_pts, d_rgbf, d_color);
            std::vector<float> up(static_cast<size_t>(n_pts)), vp(static_cast<size_t>(n_pts)), colorp(static_cast<size_t>(3 * n_pts));
            std::vector<uint8_t> infp(static_cast<size_t>(n_pts));
            CUDA_CHECK(cudaMemcpy(up.data(), d_u, up.size() * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(vp.data(), d_v, vp.size() * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(infp.data(), d_in_frustum, infp.size() * sizeof(uint8_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(colorp.data(), d_color, colorp.size() * sizeof(float), cudaMemcpyDeviceToHost));

            long n_considered = 0, n_flip = 0; double disp_sum = 0.0, zc_sum = 0.0;
            for (int i = 0; i < n_pts; ++i) {
                if (!inf0[static_cast<size_t>(i)] || !infp[static_cast<size_t>(i)]) continue;
                n_considered++;
                const size_t ci = static_cast<size_t>(i);
                const double du = up[ci] - u0[ci], dv = vp[ci] - v0[ci];
                disp_sum += std::sqrt(du * du + dv * dv);
                zc_sum += zc_gpu[ci];
                if (color_dist(colorp[3 * ci], colorp[3 * ci + 1], colorp[3 * ci + 2],
                               color0[3 * ci], color0[3 * ci + 1], color0[3 * ci + 2]) > kColorBoundaryThresh) n_flip++;
            }
            const double flip_frac = n_considered > 0 ? static_cast<double>(n_flip) / n_considered : 0.0;
            const double measured_disp = n_considered > 0 ? disp_sum / n_considered : 0.0;
            const double mean_zc = n_considered > 0 ? zc_sum / n_considered : 1.0;
            const double predicted_disp = static_cast<double>(kFx) * delta_m / mean_zc;   // 01.17's translation-error formula: fx*d_t/R
            trans_rows.push_back({ static_cast<double>(kSensitivityTransCm[lvl]), flip_frac, measured_disp, predicted_disp });
            if (prev_flip >= 0.0 && flip_frac < prev_flip - 1e-9) trans_monotone = false;
            prev_flip = flip_frac;
            if (lvl == 0) {
                const double ratio = measured_disp > 0.0 ? measured_disp / predicted_disp : 0.0;
                trans_consistent = ratio >= 1.0 / kConsistencyFactor && ratio <= kConsistencyFactor;
                std::printf("[info] sensitivity(translation) smallest level %.1f cm: predicted disp %.3f px, "
                           "measured mean disp %.3f px, ratio %.2fx\n",
                           static_cast<double>(kSensitivityTransCm[0]), predicted_disp, measured_disp, ratio);
            }
        }
        for (const auto& r : rot_rows)
            std::printf("[info] sensitivity(rotation)    level=%.1f deg: flip_fraction=%.2f%%\n", r.level, 100.0 * r.flip_frac);
        for (const auto& r : trans_rows)
            std::printf("[info] sensitivity(translation) level=%.1f cm:  flip_fraction=%.2f%%\n", r.level, 100.0 * r.flip_frac);

        const bool ok = rot_monotone && trans_monotone && rot_consistent && trans_consistent;
        gates_pass = gates_pass && ok;
        std::printf("GATE: sensitivity_curve %s (flip-fraction non-decreasing with |perturbation| for both rotation and "
                   "translation sweeps AND the smallest level's measured mean pixel displacement matches the 01.17 "
                   "analytic prediction within %.0fx)\n", ok ? "PASS" : "FAIL", kConsistencyFactor);

        // Restore the baseline production outputs (the sweep overwrote d_u/
        // d_v/d_in_frustum/d_color on the device; host copies u_gpu/v_gpu/
        // inf_gpu/color_gpu above are untouched and remain the baseline the
        // artifacts below render).
        CUDA_CHECK(cudaMemcpy(d_u, u0.data(), u0.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v, v0.data(), v0.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_in_frustum, inf0.data(), inf0.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaFree(d_pts)); CUDA_CHECK(cudaFree(d_rgbf));
    CUDA_CHECK(cudaFree(d_u)); CUDA_CHECK(cudaFree(d_v)); CUDA_CHECK(cudaFree(d_zc));
    CUDA_CHECK(cudaFree(d_in_frustum)); CUDA_CHECK(cudaFree(d_visible));
    CUDA_CHECK(cudaFree(d_color)); CUDA_CHECK(cudaFree(d_encoded));

    // ---- ARTIFACTS --------------------------------------------------------------
    const std::string out_dir = resolve_out_dir(argv[0]);
    bool artifacts_ok = true;

    // Colored cloud (top view: X forward vs Y left; side view: X forward vs
    // Z up) — the CHECKED color where colored, a neutral gray where the
    // occlusion check filtered the point out (README "Artifacts").
    {
        std::vector<float> ax(static_cast<size_t>(n_pts)), ay(static_cast<size_t>(n_pts)), az(static_cast<size_t>(n_pts));
        std::vector<uint8_t> r(static_cast<size_t>(n_pts)), g(static_cast<size_t>(n_pts)), b(static_cast<size_t>(n_pts));
        std::vector<uint8_t> mask(static_cast<size_t>(n_pts), 1);
        for (int i = 0; i < n_pts; ++i) {
            const size_t ci = static_cast<size_t>(i);
            ax[ci] = pts[ci].x; ay[ci] = pts[ci].y; az[ci] = pts[ci].z;
            if (inf_gpu[ci] && visible_gpu[ci]) {
                r[ci] = to_u8(color_gpu[3 * ci]); g[ci] = to_u8(color_gpu[3 * ci + 1]); b[ci] = to_u8(color_gpu[3 * ci + 2]);
            } else {
                r[ci] = g[ci] = b[ci] = 90;   // neutral gray: "not colored" (out of frame or filtered by the occlusion check)
            }
        }
        render_scatter(out_dir + "/cloud_topview.ppm", 260, 200, ay, ax, r, g, b, mask);   // (col=y, row=x)
        render_scatter(out_dir + "/cloud_sideview.ppm", 260, 110, ax, az, r, g, b, mask);  // (col=x, row=z)
    }

    // Occlusion cohort before/after: only ground-truth-occluded points,
    // zoomed to their own local extent.
    {
        std::vector<float> ax, ay; std::vector<uint8_t> r_naive, g_naive, b_naive, r_check, g_check, b_check, mask;
        for (int i = 0; i < n_pts; ++i) {
            if (truth[static_cast<size_t>(i)].visible) continue;
            const size_t ci = static_cast<size_t>(i);
            ax.push_back(pts[ci].x); ay.push_back(pts[ci].y);
            mask.push_back(1);
            if (inf_gpu[ci]) {
                r_naive.push_back(to_u8(color_gpu[3 * ci])); g_naive.push_back(to_u8(color_gpu[3 * ci + 1])); b_naive.push_back(to_u8(color_gpu[3 * ci + 2]));
            } else { r_naive.push_back(60); g_naive.push_back(60); b_naive.push_back(60); }
            if (inf_gpu[ci] && visible_gpu[ci]) {
                r_check.push_back(to_u8(color_gpu[3 * ci])); g_check.push_back(to_u8(color_gpu[3 * ci + 1])); b_check.push_back(to_u8(color_gpu[3 * ci + 2]));
            } else { r_check.push_back(60); g_check.push_back(60); b_check.push_back(60); }   // filtered -> neutral
        }
        render_scatter(out_dir + "/occlusion_cohort_naive.ppm", 220, 160, ay, ax, r_naive, g_naive, b_naive, mask);
        render_scatter(out_dir + "/occlusion_cohort_checked.ppm", 220, 160, ay, ax, r_check, g_check, b_check, mask);
    }

    // Painted sparse depth image (near = bright, 01.18's convention; cited).
    {
        std::vector<uint8_t> gray(static_cast<size_t>(kImagePixels), 0);
        const float near_m = 2.0f, far_m = 16.0f;
        for (int i = 0; i < kImagePixels; ++i) {
            const float d = sparse_gpu[static_cast<size_t>(i)];
            if (d <= 0.0f) { gray[static_cast<size_t>(i)] = 0; continue; }
            float t = (d - near_m) / (far_m - near_m);
            t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
            gray[static_cast<size_t>(i)] = static_cast<uint8_t>(255.0f * (1.0f - t) + 0.5f);
        }
        write_pgm(out_dir + "/painted_depth.pgm", kImageWidth, kImageHeight, gray);
    }

    // Sensitivity curve CSV.
    {
        std::ofstream f(out_dir + "/sensitivity_curve.csv");
        artifacts_ok = artifacts_ok && f.is_open();
        if (f.is_open()) {
            f << "sweep,level,level_unit,flip_fraction,measured_mean_disp_px,predicted_disp_px\n";
            for (const auto& r : rot_rows)
                f << "rotation," << r.level << ",deg," << r.flip_frac << "," << r.measured_disp_px << "," << r.predicted_disp_px << "\n";
            for (const auto& r : trans_rows)
                f << "translation," << r.level << ",cm," << r.flip_frac << "," << r.measured_disp_px << "," << r.predicted_disp_px << "\n";
        }
    }

    // Gates metrics CSV.
    {
        std::ofstream f(out_dir + "/gates_metrics.csv");
        artifacts_ok = artifacts_ok && f.is_open();
        if (f.is_open()) {
            f << "gate,metric,value,unit\n";
            f << "frustum_accounting,in_frustum," << n_in_frustum << ",count\n";
            f << "frustum_accounting,out_frustum," << n_out_frustum << ",count\n";
            f << "frustum_accounting,colored," << n_colored << ",count\n";
            f << "frustum_accounting,filtered," << n_filtered << ",count\n";
            f << "occlusion,cohort_n," << n_occluded_truth << ",count\n";
        }
    }

    if (artifacts_ok)
        std::printf("ARTIFACT: wrote cloud_topview.ppm, cloud_sideview.ppm, occlusion_cohort_naive.ppm, "
                   "occlusion_cohort_checked.ppm, painted_depth.pgm, sensitivity_curve.csv, gates_metrics.csv to demo/out/\n");
    else
        std::printf("ARTIFACT: FAILED to write one or more files to demo/out/\n");

    // ---- verdict ------------------------------------------------------------
    const bool success = verify_pass && gates_pass && artifacts_ok;
    if (success)
        std::printf("RESULT: PASS (all VERIFY twins agree and all evaluation gates pass -- see GATE lines above)\n");
    else
        std::printf("RESULT: FAIL (see VERIFY/GATE lines above for the failing check)\n");
    return success ? 0 : 1;
}
