// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 01.06
//                     (AprilTag / ArUco GPU detector-decoder for high-rate
//                     fiducial localization)
//
// WHY does a GPU repository ship a CPU implementation of everything?
// ------------------------------------------------------------------
// Two load-bearing reasons (CLAUDE.md paragraph 5):
//   1) The CORRECTNESS ORACLE — main.cu runs both paths and asserts
//      element-wise / record-wise agreement within a documented tolerance.
//   2) The TEACHING BASELINE — reading this file first, then kernels.cu,
//      shows exactly what the GPU mapping changed for each of the six
//      stages (a pixel-parallel map/stencil/scatter here becomes a thread
//      per pixel; a tiny per-component solve here becomes a thread per
//      candidate — see kernels.cuh's file header for the full contrast).
//
// Independence ruling (see docs/PROJECT_TEMPLATE/src/reference_cpu.cpp's
// canonical header — reproduced and applied here):
//   * Data-layout contracts (structs, constants, indexing formulas) are
//     single-sourced in kernels.cuh and SHARED — is_border_cell(),
//     payload_bit_index(), rotate_payload_90(), popcount16(),
//     pack_corner_key()/unpack_corner_index(). These are bookkeeping, not
//     "the algorithm under test": a disagreement here would be a layout bug,
//     not a numerics bug, and sharing them removes that whole bug class.
//   * The ALGORITHMIC CORE is written TWICE, independently: the box filter,
//     CCL (a genuinely DIFFERENT algorithm — union-find, not label
//     propagation — see ccl_union_find_cpu's own header), quad extraction,
//     the DLT Gaussian-elimination solve, grid decoding, and pose
//     decomposition are all re-typed from scratch below, not copy-pasted
//     from kernels.cu.
//   * main.cu additionally carries gates that route through NEITHER copy:
//     the pose gate compares against scripts/make_synthetic.py's own
//     (Python, independently re-derived) ground-truth camera pose, and the
//     decode-robustness gate reasons directly about known, hand-computed
//     bit-flip counts.
//
// Rules for this file: plain C++17, no CUDA headers, no hand-vectorization,
// no OpenMP, no cleverness — clarity beats speed here, always.
//
// Read this after: kernels.cu — then compare the two side by side.
// ===========================================================================

#include "kernels.cuh"

#include <cmath>
#include <algorithm>
#include <vector>

// ===========================================================================
// Stage 1 — adaptive threshold (independent nested-loop box filter).
// ===========================================================================

// box_sum_h_cpu — same clamp-boundary rule as the GPU kernel (see
// kernels.cu's box_sum_h_kernel header for why clamp, not zero-pad), coded
// as a plain double-nested loop rather than a flat linear-index loop —
// genuinely different code shape, same result.
void box_sum_h_cpu(const unsigned char* gray, float* row_sum, int W, int H)
{
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float s = 0.0f;
            for (int dx = -kBoxRadius; dx <= kBoxRadius; ++dx) {
                int nx = x + dx;
                if (nx < 0) nx = 0;
                if (nx >= W) nx = W - 1;
                s += static_cast<float>(gray[y * W + nx]);
            }
            row_sum[y * W + x] = s;
        }
    }
}

void box_sum_v_cpu(const float* row_sum, float* local_mean, int W, int H)
{
    const float area = static_cast<float>((2 * kBoxRadius + 1) * (2 * kBoxRadius + 1));
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float s = 0.0f;
            for (int dy = -kBoxRadius; dy <= kBoxRadius; ++dy) {
                int ny = y + dy;
                if (ny < 0) ny = 0;
                if (ny >= H) ny = H - 1;
                s += row_sum[ny * W + x];
            }
            local_mean[y * W + x] = s / area;
        }
    }
}

void adaptive_threshold_cpu(const unsigned char* gray, const float* local_mean,
                            unsigned char* mask, int W, int H)
{
    const int N = W * H;
    for (int i = 0; i < N; ++i)
        mask[i] = (static_cast<float>(gray[i]) < local_mean[i] - kThreshBiasC) ? 1u : 0u;
}

// ===========================================================================
// Stage 2 — connected-component labeling via classic Rosenfeld two-pass
// union-find, DELIBERATELY a different ALGORITHM from the GPU's iterative
// label propagation (kernels.cu's ccl_propagate_sweep_kernel header proves
// both converge to the SAME unique fixed point, which is exactly what makes
// comparing two different algorithms' outputs a meaningful correctness
// check rather than a foregone conclusion). 4-connectivity, matching the
// GPU's connectivity choice.
// ===========================================================================
static int uf_find(std::vector<int>& parent, int x)
{
    // Path-halving: point every visited node at its grandparent as we walk
    // to the root — a cheap, allocation-free approximation of full path
    // compression that still keeps subsequent find() calls fast.
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}
static void uf_union_toward_smaller(std::vector<int>& parent, int a, int b)
{
    a = uf_find(parent, a);
    b = uf_find(parent, b);
    if (a == b) return;
    // Always attach the LARGER root under the SMALLER one, so every
    // component's root converges to its minimum linear pixel index — the
    // exact canonical-label convention the GPU's propagation independently
    // converges to (see kernels.cu's file header proof).
    if (a < b) parent[b] = a; else parent[a] = b;
}

void ccl_union_find_cpu(const unsigned char* mask, int* label, int W, int H)
{
    std::vector<int> parent(static_cast<size_t>(W) * H);
    for (int i = 0; i < W * H; ++i) parent[i] = i;

    // Pass 1: for every foreground pixel, union with its WEST and NORTH
    // foreground neighbors (already-visited neighbors in raster order —
    // the standard two-neighbor Rosenfeld scan; EAST/SOUTH will be picked
    // up when THOSE pixels run their own west/north union).
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int i = y * W + x;
            if (!mask[i]) continue;
            if (x > 0 && mask[i - 1]) uf_union_toward_smaller(parent, i, i - 1);
            if (y > 0 && mask[i - W]) uf_union_toward_smaller(parent, i, i - W);
        }
    }
    // Pass 2: resolve every foreground pixel to its component's canonical
    // (minimum-index) root; background stays kLabelNone.
    for (int i = 0; i < W * H; ++i)
        label[i] = mask[i] ? uf_find(parent, i) : kLabelNone;
}

// ===========================================================================
// build_candidates_cpu — single sequential pass accumulating the same nine
// per-component quantities the GPU's atomic scatter computes (count,
// centroid sums, bbox, four packed corner-extreme keys), then a second pass
// that finds canonical roots and applies the SAME filter constants
// (kMinComponentPixels etc. — shared compile-time thresholds, not
// algorithm). No atomics are needed: a single thread visiting pixels in a
// fixed order can just compare-and-replace.
// ===========================================================================
int build_candidates_cpu(const unsigned char* mask, const int* label, int W, int H,
                         CandidateComponent* out)
{
    const size_t N = static_cast<size_t>(W) * H;
    std::vector<int> count(N, 0);
    std::vector<unsigned long long> sum_x(N, 0ull), sum_y(N, 0ull);
    std::vector<int> min_x(N, W), max_x(N, -1), min_y(N, H), max_y(N, -1);
    std::vector<unsigned long long> key_min_sum(N, ~0ull), key_max_sum(N, 0ull);
    std::vector<unsigned long long> key_min_diff(N, ~0ull), key_max_diff(N, 0ull);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int i = y * W + x;
            if (!mask[i]) continue;
            const int L = label[i];
            count[L] += 1;
            sum_x[L] += static_cast<unsigned long long>(x);
            sum_y[L] += static_cast<unsigned long long>(y);
            min_x[L] = std::min(min_x[L], x); max_x[L] = std::max(max_x[L], x);
            min_y[L] = std::min(min_y[L], y); max_y[L] = std::max(max_y[L], y);

            const long long s = static_cast<long long>(x) + static_cast<long long>(y);
            const long long d = static_cast<long long>(x) - static_cast<long long>(y);
            key_min_sum[L]  = std::min(key_min_sum[L],  pack_corner_key(s, i));
            key_max_sum[L]  = std::max(key_max_sum[L],  pack_corner_key(s, i));
            key_min_diff[L] = std::min(key_min_diff[L], pack_corner_key(d, i));
            key_max_diff[L] = std::max(key_max_diff[L], pack_corner_key(d, i));
        }
    }

    int n_out = 0;
    for (int p = 0; p < static_cast<int>(N) && n_out < kMaxCandidates; ++p) {
        if (!mask[p] || label[p] != p || count[p] <= 0) continue;   // not a canonical root
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

        const int idx_min_sum  = unpack_corner_index(key_min_sum[p]);
        const int idx_max_sum  = unpack_corner_index(key_max_sum[p]);
        const int idx_min_diff = unpack_corner_index(key_min_diff[p]);
        const int idx_max_diff = unpack_corner_index(key_max_diff[p]);
        c.raw_corner_x[0] = static_cast<float>(idx_min_sum  % W); c.raw_corner_y[0] = static_cast<float>(idx_min_sum  / W);
        c.raw_corner_x[1] = static_cast<float>(idx_max_diff % W); c.raw_corner_y[1] = static_cast<float>(idx_max_diff / W);
        c.raw_corner_x[2] = static_cast<float>(idx_max_sum  % W); c.raw_corner_y[2] = static_cast<float>(idx_max_sum  / W);
        c.raw_corner_x[3] = static_cast<float>(idx_min_diff % W); c.raw_corner_y[3] = static_cast<float>(idx_min_diff / W);

        out[n_out++] = c;
    }
    return n_out;
}

// ===========================================================================
// Stage 3 — quad extraction (corner refinement), independent bilinear
// sampling + radial search — same ALGORITHM FAMILY as kernels.cu's
// refine_one_corner (this project's own chosen teaching method, see
// kernels.cuh's file header for why it is honestly weaker than production),
// but every line below is typed fresh, not copied.
// ===========================================================================
static float bilerp_u8_cpu(const unsigned char* img, int W, int H, float px, float py)
{
    px = std::min(std::max(px, 0.0f), static_cast<float>(W - 1));
    py = std::min(std::max(py, 0.0f), static_cast<float>(H - 1));
    const int x0 = static_cast<int>(std::floor(px));
    const int y0 = static_cast<int>(std::floor(py));
    const int x1 = std::min(x0 + 1, W - 1);
    const int y1 = std::min(y0 + 1, H - 1);
    const float tx = px - static_cast<float>(x0), ty = py - static_cast<float>(y0);
    const float a = static_cast<float>(img[y0 * W + x0]) + tx * (static_cast<float>(img[y0 * W + x1]) - static_cast<float>(img[y0 * W + x0]));
    const float b = static_cast<float>(img[y1 * W + x0]) + tx * (static_cast<float>(img[y1 * W + x1]) - static_cast<float>(img[y1 * W + x0]));
    return a + ty * (b - a);
}
static float bilerp_f32_cpu(const float* img, int W, int H, float px, float py)
{
    px = std::min(std::max(px, 0.0f), static_cast<float>(W - 1));
    py = std::min(std::max(py, 0.0f), static_cast<float>(H - 1));
    const int x0 = static_cast<int>(std::floor(px));
    const int y0 = static_cast<int>(std::floor(py));
    const int x1 = std::min(x0 + 1, W - 1);
    const int y1 = std::min(y0 + 1, H - 1);
    const float tx = px - static_cast<float>(x0), ty = py - static_cast<float>(y0);
    const float a = img[y0 * W + x0] + tx * (img[y0 * W + x1] - img[y0 * W + x0]);
    const float b = img[y1 * W + x0] + tx * (img[y1 * W + x1] - img[y1 * W + x0]);
    return a + ty * (b - a);
}

QuadCorners corner_refine_one_cpu(const CandidateComponent& cand,
                                  const unsigned char* gray, const float* local_mean, int W, int H)
{
    QuadCorners q;
    q.valid = true;
    constexpr int kSteps = 64;
    for (int k = 0; k < 4; ++k) {
        const float dx = cand.raw_corner_x[k] - cand.centroid_x;
        const float dy = cand.raw_corner_y[k] - cand.centroid_y;
        const float raw_dist = std::sqrt(dx * dx + dy * dy);
        if (raw_dist < 1.0f) { q.valid = false; q.x[k] = cand.raw_corner_x[k]; q.y[k] = cand.raw_corner_y[k]; continue; }
        const float ux = dx / raw_dist, uy = dy / raw_dist;
        // Small FIXED additive search margin, not a multiplicative one — see
        // kernels.cu's refine_one_corner header for the bug this fixes (an
        // earlier multiplicative margin let the search wander tens of
        // pixels past small/large tags alike and lock onto unrelated
        // content far from the true corner).
        const float kCornerSearchMarginPx = 6.0f;
        const float max_t = raw_dist + kCornerSearchMarginPx;
        const float step = max_t / static_cast<float>(kSteps);

        float best_t = raw_dist;
        float f_prev = 0.0f;
        for (int s = 0; s <= kSteps; ++s) {
            const float t = step * static_cast<float>(s);
            const float px = cand.centroid_x + ux * t, py = cand.centroid_y + uy * t;
            const float g = bilerp_u8_cpu(gray, W, H, px, py);
            const float m = bilerp_f32_cpu(local_mean, W, H, px, py);
            const float f = (m - kThreshBiasC) - g;
            if (s > 0 && f_prev > 0.0f && f <= 0.0f) {
                const float t_prev = step * static_cast<float>(s - 1);
                const float frac = f_prev / (f_prev - f);
                best_t = t_prev + frac * step;
            }
            f_prev = f;
        }
        q.x[k] = cand.centroid_x + ux * best_t;
        q.y[k] = cand.centroid_y + uy * best_t;
    }
    return q;
}

// ===========================================================================
// Stage 4 — DLT homography via an independently-typed Gaussian elimination
// (double precision, partial pivoting) — same algorithm FAMILY as
// kernels.cu's solve_8x8_partial_pivot (the well-posed 8x8 system has a
// unique solution regardless of implementation details, so both sides
// should agree to near-machine precision; see main.cu's VERIFY tolerance).
// ===========================================================================
Homography homography_solve_one_cpu(const QuadCorners& quad)
{
    Homography H{};
    H.valid = false;
    if (!quad.valid) return H;

    const double half = static_cast<double>(kTagHalfM);
    const double MX[4] = { -half,  half, half, -half };
    const double MY[4] = { -half, -half, half,  half };

    // 8x9 augmented matrix, built as a flat std::vector-of-vectors here
    // (rather than kernels.cu's fixed C array) — a genuinely different data
    // structure, same numbers.
    std::vector<std::vector<double>> A(8, std::vector<double>(9, 0.0));
    for (int k = 0; k < 4; ++k) {
        const double X = MX[k], Y = MY[k];
        const double x = static_cast<double>(quad.x[k]), y = static_cast<double>(quad.y[k]);
        A[2 * k]     = { X, Y, 1.0, 0.0, 0.0, 0.0, -x * X, -x * Y, x };
        A[2 * k + 1] = { 0.0, 0.0, 0.0, X, Y, 1.0, -y * X, -y * Y, y };
    }

    constexpr double kPivotEps = 1e-9;
    for (int col = 0; col < 8; ++col) {
        int piv = col;
        double best = std::fabs(A[col][col]);
        for (int r = col + 1; r < 8; ++r) {
            const double v = std::fabs(A[r][col]);
            if (v > best) { best = v; piv = r; }
        }
        if (best < kPivotEps) return H;   // singular/degenerate — report invalid, honestly
        std::swap(A[col], A[piv]);
        for (int r = col + 1; r < 8; ++r) {
            const double f = A[r][col] / A[col][col];
            if (f == 0.0) continue;
            for (int k = col; k < 9; ++k) A[r][k] -= f * A[col][k];
        }
    }
    double h8[8];
    for (int r = 7; r >= 0; --r) {
        double s = A[r][8];
        for (int k = r + 1; k < 8; ++k) s -= A[r][k] * h8[k];
        h8[r] = s / A[r][r];
    }
    for (int k = 0; k < 8; ++k) H.h[k] = h8[k];
    H.h[8] = 1.0;
    H.valid = true;
    return H;
}

// ===========================================================================
// Stage 5 — perspective grid sampling + dictionary decode, independently
// typed (own sampling loop, own bit assembly, own dictionary search).
// ===========================================================================
Detection grid_decode_one_cpu(int candidate_index, const Homography& H,
                              const unsigned char* gray, const float* local_mean, int W, int Himg,
                              const uint16_t* dictionary, int num_dict_codes, int correction_capacity)
{
    Detection d{};
    d.candidate_index = candidate_index;
    d.border_ok = false;
    d.accepted = false;
    d.tag_id = -1;
    d.rotation = 0;
    d.hamming_distance = 999;
    d.pose_valid = false;
    for (int k = 0; k < 4; ++k) { d.corners_x[k] = 0.0f; d.corners_y[k] = 0.0f; }
    for (int k = 0; k < 9; ++k) d.R[k] = 0.0f;
    d.t[0] = d.t[1] = d.t[2] = 0.0f;
    if (!H.valid) return d;

    const double cell = static_cast<double>(kTagSizeM) / static_cast<double>(kGridN);
    const double half = static_cast<double>(kTagHalfM);

    int border_errors = 0;
    uint16_t payload = 0;
    for (int r = 0; r < kGridN; ++r) {
        for (int c = 0; c < kGridN; ++c) {
            const double X = -half + (static_cast<double>(c) + 0.5) * cell;
            const double Y = -half + (static_cast<double>(r) + 0.5) * cell;
            const double w  = H.h[6] * X + H.h[7] * Y + H.h[8];
            const double px = (H.h[0] * X + H.h[1] * Y + H.h[2]) / w;
            const double py = (H.h[3] * X + H.h[4] * Y + H.h[5]) / w;
            const float g = bilerp_u8_cpu(gray, W, Himg, static_cast<float>(px), static_cast<float>(py));
            const float m = bilerp_f32_cpu(local_mean, W, Himg, static_cast<float>(px), static_cast<float>(py));
            const bool dark = g < (m - kThreshBiasC);
            if (is_border_cell(r, c)) {
                if (!dark) ++border_errors;   // tolerated up to kMaxBorderErrors -- see its doc comment
            } else if (dark) {
                payload = static_cast<uint16_t>(payload | (1u << payload_bit_index(r, c)));
            }
        }
    }
    d.border_ok = (border_errors <= kMaxBorderErrors);

    const double MX[4] = { -half,  half, half, -half };
    const double MY[4] = { -half, -half, half,  half };
    for (int k = 0; k < 4; ++k) {
        const double w = H.h[6] * MX[k] + H.h[7] * MY[k] + H.h[8];
        d.corners_x[k] = static_cast<float>((H.h[0] * MX[k] + H.h[1] * MY[k] + H.h[2]) / w);
        d.corners_y[k] = static_cast<float>((H.h[3] * MX[k] + H.h[4] * MY[k] + H.h[5]) / w);
    }

    if (!d.border_ok) return d;
    const int ones = popcount16(payload);
    if (ones == 0 || ones == kPayloadBits) return d;

    uint16_t rot[4];
    rot[0] = payload;
    rot[1] = rotate_payload_90(rot[0]);
    rot[2] = rotate_payload_90(rot[1]);
    rot[3] = rotate_payload_90(rot[2]);

    int best_dist = 999, best_code = -1, best_rot = 0;
    for (int j = 0; j < num_dict_codes; ++j) {
        for (int rIdx = 0; rIdx < 4; ++rIdx) {
            const int hd = popcount16(static_cast<uint16_t>(rot[rIdx] ^ dictionary[j]));
            if (hd < best_dist) { best_dist = hd; best_code = j; best_rot = rIdx; }
        }
    }
    d.hamming_distance = best_dist;
    d.tag_id = best_code;
    d.rotation = best_rot;
    d.accepted = (best_dist <= correction_capacity);
    if (!d.accepted) d.tag_id = -1;
    return d;
}

// ===========================================================================
// Stage 6 — pose from homography, independently typed decomposition (same
// K^-1*H column-normalization method, THEORY.md "The math" derives it once
// for both sides — the METHOD is the thing being taught, not a hidden
// implementation detail, so both sides deliberately implement the SAME
// documented method; independence here means "typed twice", per the ruling).
// ===========================================================================
void pose_from_homography_one_cpu(const Homography& H, Detection& d)
{
    d.pose_valid = false;
    if (!H.valid) return;

    const double fx = static_cast<double>(kFx), fy = static_cast<double>(kFy);
    const double cx = static_cast<double>(kCx), cy = static_cast<double>(kCy);

    double M[3][3];
    M[0][0] = (H.h[0] - cx * H.h[6]) / fx; M[0][1] = (H.h[1] - cx * H.h[7]) / fx; M[0][2] = (H.h[2] - cx * H.h[8]) / fx;
    M[1][0] = (H.h[3] - cy * H.h[6]) / fy; M[1][1] = (H.h[4] - cy * H.h[7]) / fy; M[1][2] = (H.h[5] - cy * H.h[8]) / fy;
    M[2][0] = H.h[6];                      M[2][1] = H.h[7];                      M[2][2] = H.h[8];

    double m1[3] = { M[0][0], M[1][0], M[2][0] };
    double m2[3] = { M[0][1], M[1][1], M[2][1] };
    double m3[3] = { M[0][2], M[1][2], M[2][2] };
    const double n1 = std::sqrt(m1[0]*m1[0] + m1[1]*m1[1] + m1[2]*m1[2]);
    const double n2 = std::sqrt(m2[0]*m2[0] + m2[1]*m2[1] + m2[2]*m2[2]);
    if (n1 < 1e-9 || n2 < 1e-9) return;

    double scale = 2.0 / (n1 + n2);
    double r1[3] = { m1[0]*scale, m1[1]*scale, m1[2]*scale };
    double r2[3] = { m2[0]*scale, m2[1]*scale, m2[2]*scale };
    double t[3]  = { m3[0]*scale, m3[1]*scale, m3[2]*scale };

    if (t[2] < 0.0) {
        scale = -scale;
        for (int k = 0; k < 3; ++k) { r1[k] = m1[k]*scale; r2[k] = m2[k]*scale; t[k] = m3[k]*scale; }
    }

    const double dot12 = r1[0]*r2[0] + r1[1]*r2[1] + r1[2]*r2[2];
    double r2o[3] = { r2[0] - dot12*r1[0], r2[1] - dot12*r1[1], r2[2] - dot12*r1[2] };
    const double n2o = std::sqrt(r2o[0]*r2o[0] + r2o[1]*r2o[1] + r2o[2]*r2o[2]);
    if (n2o < 1e-9) return;
    r2o[0] /= n2o; r2o[1] /= n2o; r2o[2] /= n2o;

    const double r3[3] = {
        r1[1]*r2o[2] - r1[2]*r2o[1],
        r1[2]*r2o[0] - r1[0]*r2o[2],
        r1[0]*r2o[1] - r1[1]*r2o[0]
    };

    d.R[0] = static_cast<float>(r1[0]); d.R[1] = static_cast<float>(r2o[0]); d.R[2] = static_cast<float>(r3[0]);
    d.R[3] = static_cast<float>(r1[1]); d.R[4] = static_cast<float>(r2o[1]); d.R[5] = static_cast<float>(r3[1]);
    d.R[6] = static_cast<float>(r1[2]); d.R[7] = static_cast<float>(r2o[2]); d.R[8] = static_cast<float>(r3[2]);
    d.t[0] = static_cast<float>(t[0]); d.t[1] = static_cast<float>(t[1]); d.t[2] = static_cast<float>(t[2]);
    d.pose_valid = true;
}
