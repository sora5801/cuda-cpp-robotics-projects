// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 01.05
//                     (SIFT on GPU: Gaussian scale space, DoG extrema,
//                     warp-level orientation/descriptor histograms,
//                     brute-force L2 matching)
//
// WHY does a GPU repository ship a CPU implementation of everything? Two
// load-bearing reasons (CLAUDE.md §5) — the CORRECTNESS ORACLE (a dead-
// simple sequential version a reader can verify by eye) and the TEACHING
// BASELINE (reading this file, then kernels.cu, shows exactly what
// parallelization changed). See docs/PROJECT_TEMPLATE/src/reference_cpu.cpp
// for the repo-wide version of this argument in full, including the
// independence ruling this file follows:
//
//   * Data-LAYOUT contracts (structs, constants, the Gaussian weight-table
//     builder) are single-sourced in kernels.cuh and SHARED — divergent
//     layouts would be a bug class of their own, not "independence".
//   * The ALGORITHMIC CORE (the convolution loop, the extrema test, the
//     sub-pixel solve, the histogram accumulation, the matcher) is written
//     TWICE, independently, in the simplest possible C++ HERE.
//
// This project's twin strategy, stage by stage (see kernels.cuh's header
// for the full argument, and main.cu's VERIFY block for the measured
// numbers each choice produces):
//
//   scale space (blur + DoG)   — FLOAT TOLERANCE. Both sides consume the
//     SAME Gaussian weight table (shared data, see kernels.cuh), so any
//     measured difference isolates summation-order / FMA-fusion effects,
//     not "two different Gaussians".
//   DoG extrema                — candidate SETS compared, not element-wise.
//     Because the two blur pipelines are float-tolerance-close but not
//     bit-identical, a pixel sitting exactly AT the strict-inequality
//     boundary of the 3x3x3 test can legitimately flip between "extremum"
//     and "not" between the two pipelines — an honest, expected, and
//     SMALL-COUNT effect (main.cu measures and reports it), not a bug.
//   refine (sub-pixel/sub-scale) — FLOAT precision here too (deliberately
//     NOT double, unlike 01.04's harris_response_cpu use of double
//     accumulation): refinement is an ITERATIVE, RE-CENTERING loop, so a
//     different precision could change which INTEGER neighbor an
//     iteration re-centers onto — a qualitatively different divergence
//     (a different search PATH, not just a different rounding of the same
//     path) that would make the detect-stage comparison much harder to
//     reason about. Matching precision keeps the divergence to the
//     "independent implementation, same numeric family" kind this
//     project's detect-stage tolerance is built to describe.
//   orientation / describe     — FLOAT TOLERANCE, and this is THE
//     numerics lesson the catalog's "warp-level reductions" hook is
//     built around: kernels.cu's orientation_kernel/describe_kernel sum
//     each histogram bin via a WARP-SHUFFLE TREE (32 partial sums folded
//     pairwise in 5 steps — see that file), while this file sums the
//     SAME quantities SEQUENTIALLY, one sample at a time, in raster
//     order. Floating-point addition is NOT associative
//     ((a+b)+c != a+(b+c) in general, once rounding is involved) — so
//     these two summation ORDERS can (and, measured on the committed
//     sample, DO) disagree in the last few bits, even though every
//     individual per-sample contribution is computed identically on both
//     sides. main.cu's VERIFY tolerance for these two stages is derived
//     directly from this effect, measured, not guessed.
//   match (L2)                 — near-bit-exact in practice: both sides
//     run on the SAME (already GPU-computed, shared) descriptor arrays,
//     and both accumulate the 128 squared differences in the SAME
//     dimension order (0..127) — the one case in this file where "written
//     twice" still lands very close to identical, because there is no
//     reduction-ORDER freedom left to diverge on.
//
// Rules for this file: plain C++17, no CUDA headers, no hand-vectorization,
// no OpenMP, no cleverness. If the reference is clever, it can be wrong,
// and then the oracle lies.
//
// Read this after: kernels.cu — then compare the two side by side.
// ===========================================================================

#include "kernels.cuh"

#include <algorithm>   // std::min/max, std::sort
#include <cmath>       // std::atan2, std::exp, std::sqrt, std::floor, std::lround
#include <vector>

// ===========================================================================
// STAGE 1: Gaussian blur + DoG — independent host re-implementation of
// gaussian_blur_h_kernel + gaussian_blur_v_kernel (fused into ONE function,
// matching launch_gaussian_blur's "one call = one full 2-D blur" contract)
// and dog_subtract_kernel.
// ===========================================================================
void gaussian_blur_cpu(const float* src, float* dst, int W, int H, const float* weights, int radius)
{
    // A local temp buffer for the intermediate (horizontally-blurred)
    // image — the CPU has no equivalent of a caller-supplied device
    // scratch buffer, so it simply owns one locally (std::vector, freed
    // automatically on return).
    std::vector<float> tmp(static_cast<size_t>(W) * static_cast<size_t>(H));

    // Horizontal pass.
    for (int y = 0; y < H; ++y) {
        const float* row = src + static_cast<size_t>(y) * W;
        for (int x = 0; x < W; ++x) {
            float acc = 0.0f;
            for (int i = -radius; i <= radius; ++i) {
                int sx = x + i;
                if (sx < 0) sx = 0;
                if (sx >= W) sx = W - 1;   // clamp-to-edge (see kernels.cuh's header)
                acc += weights[i + radius] * row[sx];
            }
            tmp[static_cast<size_t>(y) * W + x] = acc;
        }
    }
    // Vertical pass.
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float acc = 0.0f;
            for (int i = -radius; i <= radius; ++i) {
                int sy = y + i;
                if (sy < 0) sy = 0;
                if (sy >= H) sy = H - 1;
                acc += weights[i + radius] * tmp[static_cast<size_t>(sy) * W + x];
            }
            dst[static_cast<size_t>(y) * W + x] = acc;
        }
    }
}

void dog_subtract_cpu(const float* a, const float* b, float* dst, int W, int H)
{
    const size_t n = static_cast<size_t>(W) * static_cast<size_t>(H);
    for (size_t i = 0; i < n; ++i) dst[i] = a[i] - b[i];
}

// downsample2x_cpu — independent host twin of downsample2x_kernel (see
// that kernel's header for why nearest-neighbor decimation is safe here).
void downsample2x_cpu(const float* src, int srcW, int srcH, float* dst)
{
    const int dstW = srcW / 2, dstH = srcH / 2;
    for (int y = 0; y < dstH; ++y)
        for (int x = 0; x < dstW; ++x)
            dst[static_cast<size_t>(y) * dstW + x] = src[static_cast<size_t>(2 * y) * srcW + (2 * x)];
}

// ===========================================================================
// STAGE 2: DoG extrema — independent host 3x3x3 local-extremum scan. Same
// STRICT-inequality decision rule as dog_extrema_candidates_kernel (a tie
// with any of the 26 neighbors suppresses the candidate on BOTH sides —
// see kernels.cu's header for why that convention must be identical), but
// written as a single sequential double loop with an explicit accumulate-
// into-std::vector (a GPU kernel cannot do that — its equivalent is the
// atomic-counter compaction in kernels.cu; this is where "the same
// algorithm, twice" looks most different in SHAPE while computing the
// same answer, mirroring 01.04's fast_nms_select_cpu precedent).
// ===========================================================================
int dog_extrema_cpu(const float* dog_below, const float* dog_center, const float* dog_above,
                    int W, int H, int octave, int layer, DogCandidate* out, int max_candidates)
{
    int count = 0;
    for (int y = kExtremaBorder; y < H - kExtremaBorder; ++y) {
        for (int x = kExtremaBorder; x < W - kExtremaBorder; ++x) {
            const float center = dog_center[static_cast<size_t>(y) * W + x];
            if (std::fabs(center) < kContrastThreshold) continue;

            bool is_max = true, is_min = true;
            for (int dy = -1; dy <= 1 && (is_max || is_min); ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const size_t nidx = static_cast<size_t>(y + dy) * W + (x + dx);
                    if (!(dx == 0 && dy == 0)) {
                        const float v = dog_center[nidx];
                        if (v >= center) is_max = false;
                        if (v <= center) is_min = false;
                    }
                    const float vb = dog_below[nidx];
                    if (vb >= center) is_max = false;
                    if (vb <= center) is_min = false;
                    const float va = dog_above[nidx];
                    if (va >= center) is_max = false;
                    if (va <= center) is_min = false;
                }
            }
            if (!is_max && !is_min) continue;

            if (count < max_candidates) out[count] = DogCandidate{ octave, layer, x, y };
            ++count;   // keep counting past capacity so the caller can detect+report an overflow, same convention as the GPU path
        }
    }
    return std::min(count, max_candidates);
}

// ===========================================================================
// STAGE 3: refine — independent host iterative quadratic-Taylor solve.
// Deliberately FLOAT precision (see this file's header for why, in
// contrast with 01.04's Harris double-accumulation precedent) and its OWN
// hand-typed 3x3 Cramer's-rule solver (solve_3x3_host below) — a SEPARATE
// piece of code from kernels.cu's solve_3x3_device, so a transcription
// bug in one cannot hide behind the other (the twin-independence ruling's
// whole point for exactly this kind of small numerical core).
// ===========================================================================
static double det3x3_host(const double M[3][3])
{
    return M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
         - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
         + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
}

// solve_3x3_host — the CPU twin's OWN 3x3 solve, hand-typed independently
// from kernels.cu's solve_3x3_device. Internal arithmetic uses double (the
// determinant/cofactor expansion is cheap and doing it in double costs
// nothing extra here — a small, LOCAL precision choice, unrelated to the
// file-header's "keep refinement float overall" decision, which is about
// the ITERATION LOOP's convergence path, not this one solve's internal
// rounding); results are cast back to float on return, matching the
// float-typed candidate/keypoint fields everywhere else in this stage.
static bool solve_3x3_host(const float Hf[3][3], const float rhsf[3], float z[3])
{
    double H[3][3], rhs[3];
    for (int r = 0; r < 3; ++r) { rhs[r] = static_cast<double>(rhsf[r]); for (int c = 0; c < 3; ++c) H[r][c] = static_cast<double>(Hf[r][c]); }

    const double det = det3x3_host(H);
    if (std::fabs(det) < 1e-12) return false;
    const double inv_det = 1.0 / det;

    double Hx[3][3] = { {rhs[0], H[0][1], H[0][2]}, {rhs[1], H[1][1], H[1][2]}, {rhs[2], H[2][1], H[2][2]} };
    double Hy[3][3] = { {H[0][0], rhs[0], H[0][2]}, {H[1][0], rhs[1], H[1][2]}, {H[2][0], rhs[2], H[2][2]} };
    double Hz[3][3] = { {H[0][0], H[0][1], rhs[0]}, {H[1][0], H[1][1], rhs[1]}, {H[2][0], H[2][1], rhs[2]} };
    z[0] = static_cast<float>(det3x3_host(Hx) * inv_det);
    z[1] = static_cast<float>(det3x3_host(Hy) * inv_det);
    z[2] = static_cast<float>(det3x3_host(Hz) * inv_det);
    return true;
}

int refine_keypoints_cpu(const float* dog, int W, int H,
                         const DogCandidate* candidates, int n, SiftKeypoint* out)
{
    int accepted_count = 0;
    for (int i = 0; i < n; ++i) {
        const DogCandidate& c = candidates[i];
        int x = c.x, y = c.y, layer = c.layer;
        float z[3] = { 0.0f, 0.0f, 0.0f };
        bool accepted = false;

        for (int iter = 0; iter < kMaxRefineIters && !accepted; ++iter) {
            if (x < 1 || x >= W - 1 || y < 1 || y >= H - 1 || layer < 1 || layer >= kDogPerOctave - 1) break;

            const float* Dm = dog + static_cast<size_t>(layer - 1) * W * H;
            const float* D0 = dog + static_cast<size_t>(layer) * W * H;
            const float* Dp = dog + static_cast<size_t>(layer + 1) * W * H;

            const float Dx = 0.5f * (D0[static_cast<size_t>(y) * W + (x + 1)] - D0[static_cast<size_t>(y) * W + (x - 1)]);
            const float Dy = 0.5f * (D0[static_cast<size_t>(y + 1) * W + x] - D0[static_cast<size_t>(y - 1) * W + x]);
            const float Ds = 0.5f * (Dp[static_cast<size_t>(y) * W + x] - Dm[static_cast<size_t>(y) * W + x]);

            const float d0 = D0[static_cast<size_t>(y) * W + x];
            const float Dxx = D0[static_cast<size_t>(y) * W + (x + 1)] - 2.0f * d0 + D0[static_cast<size_t>(y) * W + (x - 1)];
            const float Dyy = D0[static_cast<size_t>(y + 1) * W + x] - 2.0f * d0 + D0[static_cast<size_t>(y - 1) * W + x];
            const float Dss = Dp[static_cast<size_t>(y) * W + x] - 2.0f * d0 + Dm[static_cast<size_t>(y) * W + x];
            const float Dxy = 0.25f * (D0[static_cast<size_t>(y + 1) * W + (x + 1)] - D0[static_cast<size_t>(y - 1) * W + (x + 1)]
                                      - D0[static_cast<size_t>(y + 1) * W + (x - 1)] + D0[static_cast<size_t>(y - 1) * W + (x - 1)]);
            const float Dxs = 0.25f * (Dp[static_cast<size_t>(y) * W + (x + 1)] - Dp[static_cast<size_t>(y) * W + (x - 1)]
                                      - Dm[static_cast<size_t>(y) * W + (x + 1)] + Dm[static_cast<size_t>(y) * W + (x - 1)]);
            const float Dys = 0.25f * (Dp[static_cast<size_t>(y + 1) * W + x] - Dp[static_cast<size_t>(y - 1) * W + x]
                                      - Dm[static_cast<size_t>(y + 1) * W + x] + Dm[static_cast<size_t>(y - 1) * W + x]);

            const float Hmat[3][3] = { {Dxx, Dxy, Dxs}, {Dxy, Dyy, Dys}, {Dxs, Dys, Dss} };
            const float neg_grad[3] = { -Dx, -Dy, -Ds };

            if (!solve_3x3_host(Hmat, neg_grad, z)) break;

            if (std::fabs(z[0]) < kRefineConvergeTol && std::fabs(z[1]) < kRefineConvergeTol && std::fabs(z[2]) < kRefineConvergeTol) {
                const float d_hat = d0 + 0.5f * (Dx * z[0] + Dy * z[1] + Ds * z[2]);
                if (std::fabs(d_hat) < kContrastThreshold) break;

                const float tr = Dxx + Dyy;
                const float det2 = Dxx * Dyy - Dxy * Dxy;
                if (det2 <= 0.0f) break;
                const float ratio_test = (tr * tr) / det2;
                const float ratio_bound = (kEdgeRatioR + 1.0f) * (kEdgeRatioR + 1.0f) / kEdgeRatioR;
                if (ratio_test >= ratio_bound) break;

                SiftKeypoint kp;
                kp.octave = c.octave;
                kp.layer = layer;
                kp.x_oct = static_cast<float>(x) + z[0];
                kp.y_oct = static_cast<float>(y) + z[1];
                kp.ds = z[2];
                const float scale2x = static_cast<float>(1 << c.octave);
                kp.x_img = kp.x_oct * scale2x;
                kp.y_img = kp.y_oct * scale2x;
                kp.sigma_oct = sigma_at(static_cast<float>(layer) + z[2]);
                kp.sigma_img = kp.sigma_oct * scale2x;
                kp.contrast = std::fabs(d_hat);
                out[accepted_count++] = kp;
                accepted = true;
                break;
            }

            x += static_cast<int>(std::lround(static_cast<double>(z[0])));
            y += static_cast<int>(std::lround(static_cast<double>(z[1])));
            layer += static_cast<int>(std::lround(static_cast<double>(z[2])));
        }
    }
    return accepted_count;
}

// ===========================================================================
// STAGE 4: orientation — independent host 36-bin histogram, SEQUENTIAL
// raster-order accumulation (no warp, no shuffle, no per-lane partials --
// this IS the point: see this file's header for the summation-order
// numerics lesson this stage exists to demonstrate against kernels.cu's
// orientation_kernel).
// ===========================================================================
static float parabolic_peak_offset_cpu(float left, float center, float right)
{
    const float denom = left - 2.0f * center + right;
    if (std::fabs(denom) < 1e-12f) return 0.0f;
    return 0.5f * (left - right) / denom;
}

static void emit_oriented_keypoint_cpu(const SiftKeypoint& kp, float bin_interp, int num_bins,
                                       OrientedKeypoint* out, int& count, int out_capacity)
{
    float theta = bin_interp * (2.0f * kPi / static_cast<float>(num_bins));
    if (theta < 0.0f) theta += 2.0f * kPi;
    if (theta >= 2.0f * kPi) theta -= 2.0f * kPi;
    if (count < out_capacity) { out[count].kp = kp; out[count].theta = theta; }
    ++count;
}

int orientation_cpu(const float* gauss_oct, int W, int H, const SiftKeypoint* kps, int n,
                    OrientedKeypoint* out, int out_capacity)
{
    int count = 0;
    for (int k = 0; k < n; ++k) {
        const SiftKeypoint& kp = kps[k];
        const float* img = gauss_oct + static_cast<size_t>(kp.layer) * W * H;

        const float sigma_w = kOriSigmaFactor * kp.sigma_oct;
        const int radius = std::max(1, static_cast<int>(std::lround(static_cast<double>(kOriRadiusFactor * sigma_w))));
        const int cx = static_cast<int>(std::lround(static_cast<double>(kp.x_oct)));
        const int cy = static_cast<int>(std::lround(static_cast<double>(kp.y_oct)));
        const float two_sigma_w_sq = 2.0f * sigma_w * sigma_w;

        // SEQUENTIAL accumulation, raster order (y outer, x inner) --
        // deliberately a DIFFERENT traversal shape from kernels.cu's
        // "flattened, warp-strided" order (see this file's header).
        float hist[kOriHistBins] = { 0.0f };
        for (int dy = -radius; dy <= radius; ++dy) {
            const int y = cy + dy;
            if (y < 1 || y >= H - 1) continue;
            for (int dx = -radius; dx <= radius; ++dx) {
                if (dx * dx + dy * dy > radius * radius) continue;
                const int x = cx + dx;
                if (x < 1 || x >= W - 1) continue;

                const float gx = img[static_cast<size_t>(y) * W + (x + 1)] - img[static_cast<size_t>(y) * W + (x - 1)];
                const float gy = img[static_cast<size_t>(y - 1) * W + x] - img[static_cast<size_t>(y + 1) * W + x];
                const float mag = std::sqrt(gx * gx + gy * gy);
                float angle = std::atan2(gy, gx);
                if (angle < 0.0f) angle += 2.0f * kPi;

                const float weight = std::exp(-static_cast<float>(dx * dx + dy * dy) / two_sigma_w_sq);
                int bin = static_cast<int>(std::floor(angle * (static_cast<float>(kOriHistBins) / (2.0f * kPi))));
                if (bin < 0) bin = 0;
                if (bin >= kOriHistBins) bin = kOriHistBins - 1;
                hist[bin] += mag * weight;
            }
        }

        float smoothed[kOriHistBins];
        for (int b = 0; b < kOriHistBins; ++b) {
            const int m2 = (b - 2 + kOriHistBins) % kOriHistBins;
            const int m1 = (b - 1 + kOriHistBins) % kOriHistBins;
            const int p1 = (b + 1) % kOriHistBins;
            const int p2 = (b + 2) % kOriHistBins;
            smoothed[b] = (hist[m2] + hist[p2]) * (1.0f / 16.0f) + (hist[m1] + hist[p1]) * (4.0f / 16.0f) + hist[b] * (6.0f / 16.0f);
        }

        float max_val = 0.0f; int max_bin = 0;
        for (int b = 0; b < kOriHistBins; ++b) if (smoothed[b] > max_val) { max_val = smoothed[b]; max_bin = b; }
        if (max_val <= 0.0f) continue;   // no gradient signal -- drop this keypoint (same rare edge case as the GPU path)

        {
            const int m1 = (max_bin - 1 + kOriHistBins) % kOriHistBins;
            const int p1 = (max_bin + 1) % kOriHistBins;
            const float off = parabolic_peak_offset_cpu(smoothed[m1], smoothed[max_bin], smoothed[p1]);
            emit_oriented_keypoint_cpu(kp, static_cast<float>(max_bin) + off, kOriHistBins, out, count, out_capacity);
        }
        int spawned = 1;
        for (int b = 0; b < kOriHistBins && spawned < kMaxOrientedPerKeypoint; ++b) {
            if (b == max_bin) continue;
            const int m1 = (b - 1 + kOriHistBins) % kOriHistBins;
            const int p1 = (b + 1) % kOriHistBins;
            const float v = smoothed[b];
            if (v > smoothed[m1] && v > smoothed[p1] && v >= kOriPeakRatio * max_val) {
                const float off = parabolic_peak_offset_cpu(smoothed[m1], v, smoothed[p1]);
                emit_oriented_keypoint_cpu(kp, static_cast<float>(b) + off, kOriHistBins, out, count, out_capacity);
                ++spawned;
            }
        }
    }
    return std::min(count, out_capacity);
}

// ===========================================================================
// STAGE 5: describe — independent host 128-bin trilinear histogram, same
// SEQUENTIAL-accumulation-vs-warp-shuffle contrast as orientation_cpu.
// ===========================================================================
void describe_cpu(const float* gauss_oct, int W, int H, const OrientedKeypoint* kps, int n, SiftDescriptor* desc_out)
{
    for (int k = 0; k < n; ++k) {
        const OrientedKeypoint& okp = kps[k];
        const SiftKeypoint& kp = okp.kp;
        const float* img = gauss_oct + static_cast<size_t>(kp.layer) * W * H;

        const float cos_t = std::cos(okp.theta);
        const float sin_t = std::sin(okp.theta);
        const float hist_width = kDescScaleFactor * kp.sigma_oct;

        int radius = static_cast<int>(std::lround(static_cast<double>(hist_width) * 1.41421356 * (kDescGridSize + 1) * 0.5));
        radius = std::max(1, std::min(radius, kDescMaxRadius));
        const int cx = static_cast<int>(std::lround(static_cast<double>(kp.x_oct)));
        const int cy = static_cast<int>(std::lround(static_cast<double>(kp.y_oct)));

        const float half_d = kDescGridSize * 0.5f;
        const float two_sigma_desc_sq = 2.0f * half_d * half_d;

        float hist[kDescDims] = { 0.0f };

        for (int dy = -radius; dy <= radius; ++dy) {
            const int y = cy + dy;
            if (y < 1 || y >= H - 1) continue;
            for (int dx = -radius; dx <= radius; ++dx) {
                const int x = cx + dx;
                if (x < 1 || x >= W - 1) continue;

                const float rx = (cos_t * dx + sin_t * dy) / hist_width;
                const float ry = (-sin_t * dx + cos_t * dy) / hist_width;
                const float rbin = rx + half_d - 0.5f;
                const float cbin = ry + half_d - 0.5f;
                if (rbin <= -1.0f || rbin >= kDescGridSize || cbin <= -1.0f || cbin >= kDescGridSize) continue;

                const float gx = img[static_cast<size_t>(y) * W + (x + 1)] - img[static_cast<size_t>(y) * W + (x - 1)];
                const float gy = img[static_cast<size_t>(y - 1) * W + x] - img[static_cast<size_t>(y + 1) * W + x];
                const float mag = std::sqrt(gx * gx + gy * gy);
                float angle = std::atan2(gy, gx);
                if (angle < 0.0f) angle += 2.0f * kPi;

                float rel_angle = angle - okp.theta;
                if (rel_angle < 0.0f) rel_angle += 2.0f * kPi;
                if (rel_angle >= 2.0f * kPi) rel_angle -= 2.0f * kPi;
                const float obin = rel_angle * (static_cast<float>(kDescOriBins) / (2.0f * kPi));

                const float gauss_w = std::exp(-(rx * rx + ry * ry) / two_sigma_desc_sq);
                const float w = gauss_w * mag;

                const int r0 = static_cast<int>(std::floor(rbin));
                const int c0 = static_cast<int>(std::floor(cbin));
                const int o0 = static_cast<int>(std::floor(obin));
                const float rfrac = rbin - r0, cfrac = cbin - c0, ofrac = obin - o0;

                for (int dr = 0; dr <= 1; ++dr) {
                    const int rr = r0 + dr;
                    if (rr < 0 || rr >= kDescGridSize) continue;
                    const float wr = dr ? rfrac : (1.0f - rfrac);
                    for (int dc = 0; dc <= 1; ++dc) {
                        const int cc = c0 + dc;
                        if (cc < 0 || cc >= kDescGridSize) continue;
                        const float wc = dc ? cfrac : (1.0f - cfrac);
                        for (int doo = 0; doo <= 1; ++doo) {
                            int oo = (o0 + doo) % kDescOriBins;
                            if (oo < 0) oo += kDescOriBins;
                            const float wo = doo ? ofrac : (1.0f - ofrac);
                            const int bin_idx = (rr * kDescGridSize + cc) * kDescOriBins + oo;
                            hist[bin_idx] += w * wr * wc * wo;
                        }
                    }
                }
            }
        }

        float norm_sq = 0.0f;
        for (int b = 0; b < kDescDims; ++b) norm_sq += hist[b] * hist[b];
        const float inv_norm = (norm_sq > 1e-20f) ? (1.0f / std::sqrt(norm_sq)) : 0.0f;

        float clipped[kDescDims];
        float norm2_sq = 0.0f;
        for (int b = 0; b < kDescDims; ++b) {
            const float v = std::min(hist[b] * inv_norm, kDescClipValue);
            clipped[b] = v;
            norm2_sq += v * v;
        }
        const float inv_norm2 = (norm2_sq > 1e-20f) ? (1.0f / std::sqrt(norm2_sq)) : 0.0f;
        for (int b = 0; b < kDescDims; ++b) desc_out[k].v[b] = clipped[b] * inv_norm2;
    }
}

// ===========================================================================
// STAGE 6: brute-force squared-L2 match — independent host best/second-best
// scan. popcount32_portable's role in 01.04 (an explicit "what the
// hardware primitive computes" reference) has no analogue here: L2 has no
// hardware intrinsic to shadow, so this is simply the straightforward loop.
// ===========================================================================
void match_l2_cpu(const SiftDescriptor* query, int nQuery, const SiftDescriptor* train, int nTrain,
                  float* best1_dist_sq, int* best1_idx, float* best2_dist_sq, int* best2_idx)
{
    for (int qi = 0; qi < nQuery; ++qi) {
        const SiftDescriptor& q = query[qi];
        float b1 = 1.0e30f; int i1 = -1;
        float b2 = 1.0e30f; int i2 = -1;

        for (int ti = 0; ti < nTrain; ++ti) {
            const SiftDescriptor& t = train[ti];
            float dist_sq = 0.0f;
            for (int d = 0; d < kDescDims; ++d) {
                const float diff = q.v[d] - t.v[d];
                dist_sq += diff * diff;
            }
            if (dist_sq < b1) {
                b2 = b1; i2 = i1;
                b1 = dist_sq; i1 = ti;
            } else if (dist_sq < b2) {
                b2 = dist_sq; i2 = ti;
            }
        }

        best1_dist_sq[qi] = b1; best1_idx[qi] = i1;
        best2_dist_sq[qi] = b2; best2_idx[qi] = i2;
    }
}
