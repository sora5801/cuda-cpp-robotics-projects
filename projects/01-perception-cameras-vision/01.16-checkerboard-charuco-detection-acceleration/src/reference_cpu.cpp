// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 01.16
//                     (Checkerboard/ChArUco detection acceleration for
//                     auto-calibration rigs)
//
// WHY does a GPU repository ship a CPU implementation of everything?
// ------------------------------------------------------------------
// Two load-bearing reasons (CLAUDE.md paragraph 5): the CORRECTNESS ORACLE
// (main.cu runs both paths and asserts agreement within documented
// tolerance) and the TEACHING BASELINE (read this file, then kernels.cu, to
// see exactly what the GPU mapping changed for each stage).
//
// Independence ruling for THIS project (see docs/PROJECT_TEMPLATE/src/
// reference_cpu.cpp's canonical header, and kernels.cuh's file header for
// how it applies here):
//   * TWINNED (independently typed below AND in kernels.cu): saddle
//     response, NMS candidate extraction, sub-pixel refinement, marker
//     decode. Every one of these is re-derived from scratch in this file --
//     none of it calls into kernels.cu, and none of kernels.cu calls here.
//   * SHARED, NOT twinned (single-sourced HERE, called by main.cu on
//     whichever corner set it is validating): order_grid_for_view (the
//     RETIRED plain-checkerboard path, kept only as the ambiguity_lesson
//     comparison baseline), order_grid_marker_first_for_view (THIS
//     project's pipeline output of record -- see kernels.cuh's file header
//     and this file's own comment at that function's definition below),
//     solve_dlt_homography, solve_zhang_calibration, jacobi_eigen_symmetric6.
//     These are cheap, serial, host-only bookkeeping over an ALREADY-
//     verified corner set (the twin comparison above already proved GPU and
//     CPU corners agree); duplicating a DLT Gaussian-elimination solve or a
//     Jacobi eigenvalue sweep a second time, byte-for-byte the same
//     algorithm either way, would be pure transcription, not independent
//     verification. Per the ruling, code shared this way MUST be checked by
//     an INDEPENDENT gate that routes through neither copy: main.cu's
//     grid_ordering / ambiguity_lesson / occlusion gates compare against
//     scripts/make_synthetic.py's own corner ground truth (a THIRD,
//     Python, independently-written source of truth), and mini_calibration
//     compares the recovered intrinsics against make_synthetic.py's own
//     recorded camera constants -- never against anything computed in this
//     file. order_grid_marker_first_for_view additionally CALLS
//     decode_one_hypothesis_cpu (TWIN 4's own CPU half, defined just below)
//     directly, at new (homography, sample-position) arguments -- kernels.
//     cuh's independence note explains why this does not reopen the
//     twin-independence question (decode_one_hypothesis_cpu is pure,
//     homography-agnostic arithmetic, already proven GPU-vs-CPU correct).
//
// Rules for this file: plain C++17, no CUDA headers, no hand-vectorization,
// no OpenMP, no cleverness -- clarity beats speed here, always.
//
// Read this after: kernels.cu -- then compare the two side by side.
// ===========================================================================

#define DEBUG_ORDER_PRINT
#include "kernels.cuh"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>

// ===========================================================================
// Small numerical helpers used by BOTH the twinned stages below and the
// shared host stages further down this file. Living in this one file (not
// duplicated again) is fine: the twin-independence boundary this project
// cares about is the GPU/CPU boundary, not helper reuse WITHIN one side of
// it (kernels.cu has its own, separately-typed bilinear_sample_device /
// apply_homography_device -- see that file).
// ===========================================================================

// bilinear_sample_cpu — independent (from kernels.cu's bilinear_sample_device)
// re-derivation of the same clamp-to-edge bilinear interpolation.
static float bilinear_sample_cpu(const unsigned char* gray, int view, float px, float py)
{
    if (px < 0.0f) px = 0.0f;
    if (px > kImgW - 1.0f) px = kImgW - 1.0f;
    if (py < 0.0f) py = 0.0f;
    if (py > kImgH - 1.0f) py = kImgH - 1.0f;

    const int x0 = static_cast<int>(px);
    const int y0 = static_cast<int>(py);
    const int x1 = (x0 + 1 < kImgW) ? x0 + 1 : x0;
    const int y1 = (y0 + 1 < kImgH) ? y0 + 1 : y0;
    const float fx = px - static_cast<float>(x0);
    const float fy = py - static_cast<float>(y0);

    const float i00 = static_cast<float>(gray[batch_pixel_index(view, x0, y0)]);
    const float i10 = static_cast<float>(gray[batch_pixel_index(view, x1, y0)]);
    const float i01 = static_cast<float>(gray[batch_pixel_index(view, x0, y1)]);
    const float i11 = static_cast<float>(gray[batch_pixel_index(view, x1, y1)]);

    const float top = i00 + (i10 - i00) * fx;
    const float bot = i01 + (i11 - i01) * fx;
    return top + (bot - top) * fy;
}

// apply_homography_cpu — board-plane meters -> pixel, through a row-major
// 3x3 Homography (double precision). Returns false on a degenerate weight.
static bool apply_homography_cpu(const Homography& H, double X, double Y, double& u, double& v)
{
    const double w = H.h[6] * X + H.h[7] * Y + H.h[8];
    if (std::fabs(w) < 1e-12) return false;
    u = (H.h[0] * X + H.h[1] * Y + H.h[2]) / w;
    v = (H.h[3] * X + H.h[4] * Y + H.h[5]) / w;
    return true;
}

// ===========================================================================
// TWIN 1 — saddle_response_cpu. Independent re-derivation of the Hessian-
// determinant saddle test (see kernels.cu's saddle_response_kernel for the
// full derivation comment; identical formula, different code shape: a plain
// quadruple-nested loop instead of a flat grid-stride index).
// ===========================================================================
void saddle_response_cpu(const unsigned char* gray, float* resp, int num_views)
{
    const int s = kSaddleStep;
    for (int view = 0; view < num_views; ++view) {
        for (int y = 0; y < kImgH; ++y) {
            for (int x = 0; x < kImgW; ++x) {
                const int idx = batch_pixel_index(view, x, y);
                if (x < s || x >= kImgW - s || y < s || y >= kImgH - s) {
                    resp[idx] = 0.0f;
                    continue;
                }
                const float Ic  = static_cast<float>(gray[batch_pixel_index(view, x,     y    )]);
                const float Ixp = static_cast<float>(gray[batch_pixel_index(view, x + s, y    )]);
                const float Ixm = static_cast<float>(gray[batch_pixel_index(view, x - s, y    )]);
                const float Iyp = static_cast<float>(gray[batch_pixel_index(view, x,     y + s)]);
                const float Iym = static_cast<float>(gray[batch_pixel_index(view, x,     y - s)]);
                const float Ipp = static_cast<float>(gray[batch_pixel_index(view, x + s, y + s)]);
                const float Ipm = static_cast<float>(gray[batch_pixel_index(view, x + s, y - s)]);
                const float Imp = static_cast<float>(gray[batch_pixel_index(view, x - s, y + s)]);
                const float Imm = static_cast<float>(gray[batch_pixel_index(view, x - s, y - s)]);

                const float Ixx = Ixp - 2.0f * Ic + Ixm;
                const float Iyy = Iyp - 2.0f * Ic + Iym;
                const float Ixy = (Ipp - Ipm - Imp + Imm) * 0.25f;
                const float det = Ixx * Iyy - Ixy * Ixy;
                const bool axes_ok = (std::fabs(Ixx) >= kMinAxisCurvature) && (std::fabs(Iyy) >= kMinAxisCurvature);
                const bool diag_ok = (std::fabs(Ipp - Imm) <= kMaxDiagonalAsymmetry) && (std::fabs(Ipm - Imp) <= kMaxDiagonalAsymmetry);
                const bool is_saddle = (det < 0.0f) && axes_ok && diag_ok;
                resp[idx] = is_saddle ? -det : 0.0f;
            }
        }
    }
}

// ===========================================================================
// TWIN 2 — nms_candidates_cpu. Independent re-derivation of the strict-
// local-maximum test (kernels.cu's nms_candidates_kernel). Writes into a
// PER-VIEW slice of `out` sized [kNumViews*max_per_view] (the SAME layout
// convention the GPU path uses), returns the TOTAL count actually written
// (<= kNumViews*max_per_view; any view whose true candidate count exceeds
// max_per_view is silently capped exactly like the GPU path, and main.cu
// can compare the two paths' PER-VIEW counts to catch any drift).
// ===========================================================================
int nms_candidates_cpu(const float* resp, RawCandidate* out, int max_per_view, int num_views, int* out_view_counts)
{
    const int margin = kNmsRadius + kSaddleStep;
    int total = 0;
    for (int view = 0; view < num_views; ++view) {
        int count_this_view = 0;
        for (int y = margin; y < kImgH - margin; ++y) {
            for (int x = margin; x < kImgW - margin; ++x) {
                const float center = resp[batch_pixel_index(view, x, y)];
                if (center <= kSaddleRespThresh) continue;
                bool is_max = true;
                for (int dy = -kNmsRadius; dy <= kNmsRadius && is_max; ++dy) {
                    for (int dx = -kNmsRadius; dx <= kNmsRadius; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        const float other = resp[batch_pixel_index(view, x + dx, y + dy)];
                        if (other >= center) { is_max = false; break; }
                    }
                }
                if (!is_max) continue;
                if (count_this_view < max_per_view) {
                    RawCandidate c;
                    c.view = view; c.x = x; c.y = y; c.score = center;
                    out[view * max_per_view + count_this_view] = c;
                    ++total;
                }
                ++count_this_view;   // counted even past capacity, mirroring the GPU path's view_counts[]
            }
        }
        out_view_counts[view] = count_this_view;
    }
    return total;
}

// ===========================================================================
// TWIN 3 — subpixel_refine_one_cpu. Independent re-derivation of the
// gradient-orthogonality iteration (kernels.cu's subpixel_refine_kernel).
// One candidate in, one refined corner out -- main.cu calls this once per
// candidate, mirroring how the GPU launches one thread per candidate.
// ===========================================================================
RefinedCorner subpixel_refine_one_cpu(const unsigned char* gray, const RawCandidate& cand)
{
    double cx = static_cast<double>(cand.x);
    double cy = static_cast<double>(cand.y);
    bool valid = true;

    for (int iter = 0; iter < kRefineIters; ++iter) {
        double G00 = 0.0, G01 = 0.0, G11 = 0.0, bxv = 0.0, byv = 0.0;
        for (int dy = -kRefineWinRadius; dy <= kRefineWinRadius; ++dy) {
            for (int dx = -kRefineWinRadius; dx <= kRefineWinRadius; ++dx) {
                if (dx == 0 && dy == 0) continue;
                const double qx = cx + dx;
                const double qy = cy + dy;
                const float gx = (bilinear_sample_cpu(gray, cand.view, static_cast<float>(qx + 1.0), static_cast<float>(qy)) -
                                 bilinear_sample_cpu(gray, cand.view, static_cast<float>(qx - 1.0), static_cast<float>(qy))) * 0.5f;
                const float gy = (bilinear_sample_cpu(gray, cand.view, static_cast<float>(qx), static_cast<float>(qy + 1.0)) -
                                 bilinear_sample_cpu(gray, cand.view, static_cast<float>(qx), static_cast<float>(qy - 1.0))) * 0.5f;
                const double gxd = static_cast<double>(gx), gyd = static_cast<double>(gy);
                G00 += gxd * gxd; G01 += gxd * gyd; G11 += gyd * gyd;
                bxv += gxd * gxd * qx + gxd * gyd * qy;
                byv += gxd * gyd * qx + gyd * gyd * qy;
            }
        }
        const double det = G00 * G11 - G01 * G01;
        if (std::fabs(det) < 1e-6) { valid = false; break; }
        cx = (bxv * G11 - G01 * byv) / det;
        cy = (G00 * byv - G01 * bxv) / det;
    }

    RefinedCorner r;
    r.view = cand.view; r.x = static_cast<float>(cx); r.y = static_cast<float>(cy); r.valid = valid;
    return r;
}

// ===========================================================================
// TWIN 4 — marker_decode_one_cpu. Independent re-derivation of the two-
// hypothesis marker decode (kernels.cu's marker_decode_kernel /
// decode_one_hypothesis). Coded as nested loops directly, rather than
// factored into a shared sub-function, so this really is a from-scratch
// second implementation, not the GPU device function recompiled for host.
// ===========================================================================
static void decode_one_hypothesis_cpu(const unsigned char* gray, int view, const Homography& H,
                                      int sample_bx, int sample_by, bool mirrored,
                                      uint16_t true_code, int correction_capacity,
                                      bool& border_ok, bool& accepted, int& hamming_out)
{
    float scx, scy;
    square_center_board_xy(sample_bx, sample_by, scx, scy);
    const double half = static_cast<double>(kMarkerFillFrac) * kSquareSizeM * 0.5;
    const double cell = (2.0 * half) / static_cast<double>(kMarkerGridN);

    bool cell_black[kMarkerGridN][kMarkerGridN];
    for (int r = 0; r < kMarkerGridN; ++r) {
        for (int c = 0; c < kMarkerGridN; ++c) {
            const double lx = (static_cast<double>(scx) - sample_bx * kSquareSizeM) - half + (c + 0.5) * cell;
            const double ly = (static_cast<double>(scy) - sample_by * kSquareSizeM) - half + (r + 0.5) * cell;
            const double boardX = sample_bx * static_cast<double>(kSquareSizeM) + lx;
            const double boardY = sample_by * static_cast<double>(kSquareSizeM) + ly;
            double u = 0.0, v = 0.0;
            bool black = true;
            if (apply_homography_cpu(H, boardX, boardY, u, v)) {
                const float val = bilinear_sample_cpu(gray, view, static_cast<float>(u), static_cast<float>(v));
                black = (val < kInkMidThreshold);
            }
            const int lr = mirrored ? (kMarkerGridN - 1 - r) : r;
            const int lc = mirrored ? (kMarkerGridN - 1 - c) : c;
            cell_black[lr][lc] = black;
        }
    }

    int border_errors = 0;
    for (int r = 0; r < kMarkerGridN; ++r)
        for (int c = 0; c < kMarkerGridN; ++c)
            if (marker_is_border_cell(r, c) && !cell_black[r][c]) ++border_errors;
    border_ok = (border_errors <= kMaxMarkerBorderErrors);

    uint16_t code = 0;
    for (int r = 1; r < kMarkerGridN - 1; ++r)
        for (int c = 1; c < kMarkerGridN - 1; ++c)
            if (cell_black[r][c]) code = static_cast<uint16_t>(code | (1u << marker_payload_bit_index(r - 1, c - 1)));

    hamming_out = popcount_u32(static_cast<unsigned int>(code ^ true_code));
    accepted = border_ok && (hamming_out <= correction_capacity);
}

MarkerDecodeResult marker_decode_one_cpu(const unsigned char* gray, const Homography& H,
                                         int view, int marker_id,
                                         const uint16_t* true_codes, int correction_capacity)
{
    MarkerDecodeResult r;
    r.view = view; r.marker_id = marker_id;
    r.border_ok_identity = false; r.border_ok_mirrored = false;
    r.accepted = false; r.hyp_mirrored = false; r.hamming_distance = kMarkerPayloadBits;

    if (!H.valid) return r;

    int bx, by; square_of_marker_id(marker_id, bx, by);
    const uint16_t true_code = true_codes[marker_id];

    bool border_id, acc_id; int ham_id;
    decode_one_hypothesis_cpu(gray, view, H, bx, by, false, true_code, correction_capacity, border_id, acc_id, ham_id);
    r.border_ok_identity = border_id;

    int mbx, mby; mirror_square(bx, by, mbx, mby);
    bool border_mir, acc_mir; int ham_mir;
    decode_one_hypothesis_cpu(gray, view, H, mbx, mby, true, true_code, correction_capacity, border_mir, acc_mir, ham_mir);
    r.border_ok_mirrored = border_mir;

    if (acc_id) {
        r.accepted = true; r.hyp_mirrored = false; r.hamming_distance = ham_id;
    } else if (acc_mir) {
        r.accepted = true; r.hyp_mirrored = true; r.hamming_distance = ham_mir;
    } else {
        if (ham_id <= ham_mir) { r.hyp_mirrored = false; r.hamming_distance = ham_id; }
        else                    { r.hyp_mirrored = true;  r.hamming_distance = ham_mir; }
    }
    return r;
}

// ===========================================================================
// SHARED (not twinned) — small dense linear algebra: 3x3 matrix ops and an
// 8x8 Gaussian-elimination-with-partial-pivoting solver, the same "batched
// small dense solve" teaching pattern as 33.01 (cited in kernels.cuh),
// applied here to homography DLT instead of a robot-arm Jacobian.
// ===========================================================================
struct Mat3 { double m[9]; };  // row-major 3x3

static Mat3 mat3_mul(const Mat3& A, const Mat3& B)
{
    Mat3 C{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double s = 0.0;
            for (int k = 0; k < 3; ++k) s += A.m[i * 3 + k] * B.m[k * 3 + j];
            C.m[i * 3 + j] = s;
        }
    return C;
}

static Mat3 mat3_inverse(const Mat3& M)
{
    const double a = M.m[0], b = M.m[1], c = M.m[2];
    const double d = M.m[3], e = M.m[4], f = M.m[5];
    const double g = M.m[6], h = M.m[7], i = M.m[8];
    const double A =  (e * i - f * h), B = -(d * i - f * g), C =  (d * h - e * g);
    const double D = -(b * i - c * h), E =  (a * i - c * g), F = -(a * h - b * g);
    const double G =  (b * f - c * e), H = -(a * f - c * d), I =  (a * e - b * d);
    const double det = a * A + b * B + c * C;
    const double inv_det = (std::fabs(det) > 1e-15) ? (1.0 / det) : 0.0;
    Mat3 R{};
    R.m[0] = A * inv_det; R.m[1] = D * inv_det; R.m[2] = G * inv_det;
    R.m[3] = B * inv_det; R.m[4] = E * inv_det; R.m[5] = H * inv_det;
    R.m[6] = C * inv_det; R.m[7] = F * inv_det; R.m[8] = I * inv_det;
    return R;
}

// solve_gauss_partial_pivot — solve A x = b for an NxN dense system via
// Gaussian elimination with partial pivoting (the same algorithm 01.06's
// DLT solve uses, generalized here to a runtime N so this one function
// serves both the 8-unknown DLT solve below). Operates on a LOCAL copy of
// A/b (the caller's arrays are untouched). Returns false if the matrix is
// (numerically) singular.
static bool solve_gauss_partial_pivot(std::vector<double> A, std::vector<double> b, int N, double* x)
{
    // A is row-major N*N, b is length N (both passed by value -- small N
    // here (8), so the copy is cheap and keeps this function side-effect-free).
    for (int col = 0; col < N; ++col) {
        int pivot_row = col;
        double best = std::fabs(A[col * N + col]);
        for (int r = col + 1; r < N; ++r) {
            const double v = std::fabs(A[r * N + col]);
            if (v > best) { best = v; pivot_row = r; }
        }
        if (best < 1e-14) return false;   // singular (or effectively so)
        if (pivot_row != col) {
            for (int k = 0; k < N; ++k) std::swap(A[col * N + k], A[pivot_row * N + k]);
            std::swap(b[col], b[pivot_row]);
        }
        const double piv = A[col * N + col];
        for (int r = col + 1; r < N; ++r) {
            const double factor = A[r * N + col] / piv;
            if (factor == 0.0) continue;
            for (int k = col; k < N; ++k) A[r * N + k] -= factor * A[col * N + k];
            b[r] -= factor * b[col];
        }
    }
    for (int r = N - 1; r >= 0; --r) {
        double s = b[r];
        for (int k = r + 1; k < N; ++k) s -= A[r * N + k] * x[k];
        x[r] = s / A[r * N + r];
    }
    return true;
}

// ===========================================================================
// SHARED — solve_dlt_homography: Hartley-normalized DLT (THEORY.md
// "Numerical considerations" derives why normalization matters: without it,
// the design matrix mixes terms of wildly different scale -- meters (~0.01-
// 0.24) against pixels (~0-320) -- and the least-squares solve is needlessly
// ill-conditioned; Hartley 1997 shows that translating each point set to
// its own centroid and rescaling to average distance sqrt(2) from it fixes
// this completely, for free).
// ===========================================================================
Homography solve_dlt_homography(const float* board_x, const float* board_y,
                                const float* px_x, const float* px_y, int n)
{
    Homography out{}; out.valid = false;
    if (n < 4) return out;

    // ---- normalize both point sets -----------------------------------------
    double bcx = 0, bcy = 0, pcx = 0, pcy = 0;
    for (int k = 0; k < n; ++k) { bcx += board_x[k]; bcy += board_y[k]; pcx += px_x[k]; pcy += px_y[k]; }
    bcx /= n; bcy /= n; pcx /= n; pcy /= n;

    double bavg = 0, pavg = 0;
    for (int k = 0; k < n; ++k) {
        bavg += std::sqrt((board_x[k] - bcx) * (board_x[k] - bcx) + (board_y[k] - bcy) * (board_y[k] - bcy));
        pavg += std::sqrt((px_x[k] - pcx) * (px_x[k] - pcx) + (px_y[k] - pcy) * (px_y[k] - pcy));
    }
    bavg /= n; pavg /= n;
    if (bavg < 1e-12 || pavg < 1e-12) return out;   // degenerate (all points coincide)
    const double bs = std::sqrt(2.0) / bavg;
    const double ps = std::sqrt(2.0) / pavg;

    std::vector<double> Xn(n), Yn(n), Un(n), Vn(n);
    for (int k = 0; k < n; ++k) {
        Xn[k] = (board_x[k] - bcx) * bs; Yn[k] = (board_y[k] - bcy) * bs;
        Un[k] = (px_x[k] - pcx) * ps;    Vn[k] = (px_y[k] - pcy) * ps;
    }

    // ---- accumulate the 8x8 normal-equation system (h33 fixed to 1 in
    // NORMALIZED coordinates -- safe because normalization keeps every
    // point away from the origin, so h33=1 is never close to the true
    // degenerate case the way an UN-normalized fixed-scale choice could be). ---
    std::vector<double> AtA(64, 0.0);
    std::vector<double> Atb(8, 0.0);
    for (int k = 0; k < n; ++k) {
        // Row for u: [X,Y,1,0,0,0,-Xu,-Yu] . h = u
        double ru[8] = { Xn[k], Yn[k], 1.0, 0.0, 0.0, 0.0, -Xn[k] * Un[k], -Yn[k] * Un[k] };
        double bu = Un[k];
        // Row for v: [0,0,0,X,Y,1,-Xv,-Yv] . h = v
        double rv[8] = { 0.0, 0.0, 0.0, Xn[k], Yn[k], 1.0, -Xn[k] * Vn[k], -Yn[k] * Vn[k] };
        double bv = Vn[k];
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) AtA[i * 8 + j] += ru[i] * ru[j] + rv[i] * rv[j];
            Atb[i] += ru[i] * bu + rv[i] * bv;
        }
    }

    double h[8] = {};
    if (!solve_gauss_partial_pivot(AtA, Atb, 8, h)) return out;

    Mat3 Hn{};
    Hn.m[0] = h[0]; Hn.m[1] = h[1]; Hn.m[2] = h[2];
    Hn.m[3] = h[3]; Hn.m[4] = h[4]; Hn.m[5] = h[5];
    Hn.m[6] = h[6]; Hn.m[7] = h[7]; Hn.m[8] = 1.0;

    // ---- denormalize: H = T2^-1 * Hn * T1, where T1 maps board->normalized
    // board, T2 maps pixel->normalized pixel (both similarity transforms). ---
    Mat3 T1{}; T1.m[0] = bs; T1.m[2] = -bs * bcx; T1.m[4] = bs; T1.m[5] = -bs * bcy; T1.m[8] = 1.0;
    Mat3 T2{}; T2.m[0] = ps; T2.m[2] = -ps * pcx; T2.m[4] = ps; T2.m[5] = -ps * pcy; T2.m[8] = 1.0;
    const Mat3 T2inv = mat3_inverse(T2);
    const Mat3 Hpix = mat3_mul(mat3_mul(T2inv, Hn), T1);

    for (int k = 0; k < 9; ++k) out.h[k] = Hpix.m[k];
    out.valid = true;
    return out;
}

// ===========================================================================
// SHARED — order_grid_for_view (THEORY.md "The algorithm" walks every step).
// ===========================================================================
namespace {

struct Pt { float x, y; int src_index; };

// try_order_from_seed — one attempt of the whole procedure, rooted at a
// GIVEN seed point. Returns the number of corners successfully labeled (0
// on total failure, e.g. could not establish two independent directions).
int try_order_from_seed(const std::vector<Pt>& pts, int seed_idx, GridLabel* out, Homography& out_hom)
{
    const int n = static_cast<int>(pts.size());
    std::vector<bool> used(n, false);
    used[seed_idx] = true;

    auto dist = [&](int a, int b) {
        const double dx = pts[a].x - pts[b].x, dy = pts[a].y - pts[b].y;
        return std::sqrt(dx * dx + dy * dy);
    };

    // ---- find the nearest neighbor (direction 1) ---------------------------
    int n1 = -1; double best_d1 = 1e30;
    for (int k = 0; k < n; ++k) {
        if (k == seed_idx) continue;
        const double d = dist(seed_idx, k);
        if (d < best_d1) { best_d1 = d; n1 = k; }
    }
    if (n1 < 0) return 0;
    double d1x = pts[n1].x - pts[seed_idx].x, d1y = pts[n1].y - pts[seed_idx].y;
    const double d1len = std::sqrt(d1x * d1x + d1y * d1y);
    if (d1len < 1e-6) return 0;
    d1x /= d1len; d1y /= d1len;

    // ---- find the most-orthogonal-to-d1 neighbor AMONG POINTS AT A
    // COMPARABLE STEP LENGTH to n1 (direction 2). Why the magnitude band
    // matters (a real bug this project's own build caught -- CLAUDE.md §6
    // "narrate the thought process, including the one that failed"): an
    // earlier version of this search ranked every one of the nearest dozen
    // points by orthogonality ALONE, with no regard for distance. On a
    // regular lattice, TWO neighbors are often near-perfectly orthogonal to
    // d1 at once -- the true adjacent corner at ~1 grid step away, AND a
    // corner two full steps away lying on the exact same grid line (same
    // near-zero cosine, since both sit on a line perpendicular to d1). A
    // strict "smallest cosine wins" tie-break can let float noise hand
    // victory to the FARTHER of the two, silently doubling this axis's grid
    // spacing and corrupting every label downstream. Restricting the search
    // to candidates within [0.5, 1.8] x n1's own step length first (falling
    // back to the unrestricted search only if that band is empty, e.g. the
    // true neighbor is occluded) fixes this: it is the same physical
    // assumption place_tags_no_overlap-style neighbor searches make
    // elsewhere in this repo -- adjacent lattice points are adjacent in
    // BOTH distance and direction, not direction alone.
    // ---------------------------------------------------------------------
    std::vector<int> order_by_dist;
    for (int k = 0; k < n; ++k) if (k != seed_idx && k != n1) order_by_dist.push_back(k);
    std::sort(order_by_dist.begin(), order_by_dist.end(),
             [&](int a, int b) { return dist(seed_idx, a) < dist(seed_idx, b); });
    const int consider = std::min<int>(12, static_cast<int>(order_by_dist.size()));

    auto search_n2 = [&](bool enforce_band) {
        int best = -1; double best_abs_cos = 2.0;
        for (int t = 0; t < consider; ++t) {
            const int k = order_by_dist[t];
            double dx = pts[k].x - pts[seed_idx].x, dy = pts[k].y - pts[seed_idx].y;
            const double dl = std::sqrt(dx * dx + dy * dy);
            if (dl < 1e-6) continue;
            if (enforce_band && (dl < 0.5 * d1len || dl > 1.8 * d1len)) continue;
            dx /= dl; dy /= dl;
            const double abs_cos = std::fabs(dx * d1x + dy * d1y);
            if (abs_cos < best_abs_cos) { best_abs_cos = abs_cos; best = k; }
        }
        return best;
    };
    int n2 = search_n2(/*enforce_band=*/true);
    if (n2 < 0) n2 = search_n2(/*enforce_band=*/false);   // band empty (e.g. occlusion) -- fall back, honestly weaker
    if (n2 < 0) return 0;
    double d2x = pts[n2].x - pts[seed_idx].x, d2y = pts[n2].y - pts[seed_idx].y;
    const double d2len = std::sqrt(d2x * d2x + d2y * d2y);
    d2x /= d2len; d2y /= d2len;

    // ---- walk ONE direction from the seed, re-estimating the step vector
    // from the last TWO accepted points each time (handles perspective
    // foreshortening -- a constant step would drift off a tilted board).
    // Returns the chain of NEWLY found points only (seed itself excluded --
    // walk_bidirectional() below stitches the seed back in once, at the
    // junction between the two directions). --------------------------------
    auto walk_one_dir = [&](double dirx, double diry, double step_len) {
        std::vector<int> chain;
        double cx = pts[seed_idx].x, cy = pts[seed_idx].y;
        double stepx = dirx * step_len, stepy = diry * step_len;
        for (int step = 0; step < std::max(kBoardCornersX, kBoardCornersY); ++step) {
            const double px = cx + stepx, py = cy + stepy;
            const double tol = kGridMatchTolFactor * std::sqrt(stepx * stepx + stepy * stepy) + 3.0;
            int best = -1; double best_d = tol;
            for (int k = 0; k < n; ++k) {
                if (used[k]) continue;
                const double dx = pts[k].x - px, dy = pts[k].y - py;
                const double d = std::sqrt(dx * dx + dy * dy);
                if (d < best_d) { best_d = d; best = k; }
            }
            if (best < 0) break;
            used[best] = true;
            stepx = pts[best].x - cx; stepy = pts[best].y - cy;   // re-estimate from the last accepted hop
            cx = pts[best].x; cy = pts[best].y;
            chain.push_back(best);
        }
        return chain;
    };

    // walk_bidirectional -- walk BOTH +dir and -dir from the seed and
    // splice the results into one chain with the seed in the middle. Why
    // bidirectional (a real robustness gap this project's own build
    // caught): the very FIRST seed choice (top-left-most detected corner)
    // is usually near a true grid boundary, so a forward-only walk from it
    // reaches nearly the full axis length -- but every RETRY seed (tried
    // when the first choice fails, or reached while probing direction 2
    // from an interior point) can sit anywhere in the interior of an axis,
    // where a forward-only walk finds at most "however many corners are
    // left in that one direction", silently UNDER-COUNTING that axis and
    // defeating the length-based i-vs-j disambiguation below (a length-5
    // chain that is really 3 own-side + 2 far-side points of the 7-corner
    // axis, undercounted, can tie or lose against the true 5-corner axis).
    // Returns the stitched chain AND `seed_offset`, the seed's own index
    // within it (0 if nothing was found in the negative direction).
    // ------------------------------------------------------------------------
    auto walk_bidirectional = [&](double dirx, double diry, double step_len, int& seed_offset) {
        std::vector<int> pos = walk_one_dir(dirx, diry, step_len);
        std::vector<int> neg = walk_one_dir(-dirx, -diry, step_len);
        std::vector<int> chain(neg.rbegin(), neg.rend());
        chain.push_back(seed_idx);
        chain.insert(chain.end(), pos.begin(), pos.end());
        seed_offset = static_cast<int>(neg.size());
        return chain;
    };

    used[n1] = false; used[n2] = false;   // walk_one_dir() re-marks the seed and re-discovers n1/n2 itself
    int row_offset = 0, col_offset = 0;
    std::vector<int> row_chain = walk_bidirectional(d1x, d1y, d1len, row_offset);
    used[seed_idx] = true;
    std::vector<int> col_chain = walk_bidirectional(d2x, d2y, d2len, col_offset);

    // ---- disambiguate WHICH walked direction is the "i" (X, up to
    // kBoardCornersX=7) axis vs the "j" (Y, up to kBoardCornersY=5) axis --
    // a real bug this project's own build caught (CLAUDE.md §6 "narrate the
    // thought process, including the one that failed"): nothing about
    // finding "the nearest neighbor" (d1) or "the most orthogonal neighbor"
    // (d2) has any idea which physical board axis it just walked -- d1
    // might legitimately be the vertical direction if it happens to be
    // fractionally closer to the seed than the horizontal one is, which
    // depends on this view's own tilt and is NOT predictable in advance.
    // Labeling whichever chain came from d1 as "i" regardless produced a
    // SILENT transpose (i<->j) relative to scripts/make_synthetic.py's own
    // corner_board_xy(i,j) convention on roughly half of this project's
    // views before this fix -- every downstream label was self-consistent
    // (the DLT fit and marker decode do not care which axis is which) but
    // WRONG relative to ground truth. The board is NOT square
    // (kBoardCornersX=7 != kBoardCornersY=5, unlike this project's marker
    // grid), so chain LENGTH disambiguates: the i-axis can walk up to 7
    // real corners, the j-axis at most 5 -- whichever chain is LONGER is
    // the i-axis. Bidirectional walking (above) is what makes this
    // heuristic reliable in practice: a forward-only chain routinely tied
    // or lost on length by pure bad luck (an interior seed's "remaining"
    // count on the true 7-axis can easily undercut the true 5-axis's own
    // full count); walking both ways nearly always recovers each axis's
    // TRUE extent, so ties are now genuinely rare (only under real
    // occlusion of one whole side). A residual tie keeps the d1-first
    // assignment -- an honest, documented limitation (README "Limitations
    // & honesty").
    if (col_chain.size() > row_chain.size()) {
        std::swap(row_chain, col_chain);
        std::swap(row_offset, col_offset);
        std::swap(d1x, d2x); std::swap(d1y, d2y);   // keep (d1,d2) matched to (row,col) after the swap
    }

    // ---- enforce a consistent HANDEDNESS between the i-direction (d1) and
    // j-direction (d2) -- a second, distinct bug from the axis-identity
    // swap above (CLAUDE.md §6, again: caught by this project's own build,
    // not anticipated in the design). Axis IDENTITY (which chain is i vs
    // j) is now correct, but each chain's own SIGN (which end is "small
    // index", which is "large") is still whatever the nearest-neighbor
    // search happened to find first -- entirely unconstrained. A full
    // 180-degree board rotation flips BOTH signs together (d1 -> -d1 AND
    // d2 -> -d2), which is the intended, physically real ambiguity this
    // project's marker anchoring resolves (README "System context"). But
    // flipping only ONE sign is not a rotation at all -- it is a
    // reflection, which never happens to a rigid board viewed from the
    // front -- so if only one sign is flipped, this attempt has combined
    // an unrelated pair of directions into a self-consistent-looking but
    // PHYSICALLY IMPOSSIBLE labeling (observed as a clean "j" or "i" axis
    // running backwards on this project's own committed views). The 2-D
    // cross product d1 x d2 has a FIXED sign for every orientation-
    // preserving view of the board (any rotation, including 180 degrees)
    // and the OPPOSITE sign whenever exactly one axis has been flipped --
    // so pinning that sign to the image convention's expected value
    // (i rightward, j downward -> a right-handed x-right/y-down frame ->
    // cross > 0) and re-reversing d2/col_chain whenever it is negative
    // removes this failure mode outright, leaving ONLY the legitimate,
    // both-axes 180-degree case for the marker-anchoring stage to resolve.
    const double cross = d1x * d2y - d1y * d2x;
    if (cross < 0.0) {
        std::reverse(col_chain.begin(), col_chain.end());
        col_offset = static_cast<int>(col_chain.size()) - 1 - col_offset;
        d2x = -d2x; d2y = -d2y;
    }

    // Need at least 2 points in each direction (a line needs 2 to exist) and
    // at least 4 correspondences total for a DLT fit.
    if (row_chain.size() < 2 || col_chain.size() < 2) return 0;

    // ---- finish_with_offsets: build correspondences, fit H0, predict every
    // remaining grid slot, and refit -- PARAMETERIZED by the anchor offsets
    // (not just the ones walk_bidirectional happened to measure). Why this
    // needs to be a re-triable function, not a one-shot tail (a real gap
    // this project's own build exposed -- CLAUDE.md §6): row_offset/
    // col_offset only tell us where the SEED sits WITHIN whatever this
    // chain actually found -- they say nothing about where that chain sits
    // within the board's true 7 (or 5) corner extent whenever the chain is
    // SHORTER than the axis's own known length (occlusion, a missed
    // detection at the true boundary, ...). A length-6 row chain could
    // legitimately be true i=[0..5] OR i=[1..6] -- both are consistent with
    // everything the chain itself observed -- and guessing wrong shifts
    // every label in that chain by a constant, which then cascades into the
    // predict-remaining phase too. The caller below tries EVERY plausible
    // shift for both axes and keeps whichever placed the most corners --
    // the same "try several hypotheses, keep the best-supported one"
    // strategy this function already uses for the SEED itself, one level
    // up (order_grid_for_view), applied here to the anchor instead.
    // -------------------------------------------------------------------------
    auto finish_with_offsets = [&](int row_off, int col_off, GridLabel* out_local, Homography& hom_local) -> int {
        std::vector<float> bx_list, by_list, px_list, py_list;
        for (size_t k = 0; k < row_chain.size(); ++k) {
            const int i = static_cast<int>(k) - row_off;
            if (i < 0 || i >= kBoardCornersX) continue;
            float X, Y; corner_board_xy(i, 0, X, Y);
            bx_list.push_back(X); by_list.push_back(Y);
            px_list.push_back(pts[row_chain[k]].x); py_list.push_back(pts[row_chain[k]].y);
        }
        for (size_t k = 0; k < col_chain.size(); ++k) {
            const int j = static_cast<int>(k) - col_off;
            if (static_cast<int>(k) == col_off) continue;   // the seed -- already added via row_chain above
            if (j < 0 || j >= kBoardCornersY) continue;
            float X, Y; corner_board_xy(0, j, X, Y);
            bx_list.push_back(X); by_list.push_back(Y);
            px_list.push_back(pts[col_chain[k]].x); py_list.push_back(pts[col_chain[k]].y);
        }
        if (static_cast<int>(bx_list.size()) < 4) return 0;

        Homography H0 = solve_dlt_homography(bx_list.data(), by_list.data(), px_list.data(), py_list.data(),
                                             static_cast<int>(bx_list.size()));
        if (!H0.valid) return 0;

        for (int k = 0; k < n; ++k) out_local[k] = GridLabel{};
        bool already[kBoardCornersX][kBoardCornersY] = {};
        int placed = 0;
        std::vector<float> final_bx, final_by, final_px, final_py;
        for (size_t k = 0; k < row_chain.size(); ++k) {
            const int i = static_cast<int>(k) - row_off;
            if (i < 0 || i >= kBoardCornersX) continue;
            out_local[pts[row_chain[k]].src_index] = GridLabel{ i, 0 };
            already[i][0] = true;
            float X, Y; corner_board_xy(i, 0, X, Y);
            final_bx.push_back(X); final_by.push_back(Y);
            final_px.push_back(pts[row_chain[k]].x); final_py.push_back(pts[row_chain[k]].y);
            ++placed;
        }
        for (size_t k = 0; k < col_chain.size(); ++k) {
            if (static_cast<int>(k) == col_off) continue;
            const int j = static_cast<int>(k) - col_off;
            if (j < 0 || j >= kBoardCornersY) continue;
            out_local[pts[col_chain[k]].src_index] = GridLabel{ 0, j };
            already[0][j] = true;
            float X, Y; corner_board_xy(0, j, X, Y);
            final_bx.push_back(X); final_by.push_back(Y);
            final_px.push_back(pts[col_chain[k]].x); final_py.push_back(pts[col_chain[k]].y);
            ++placed;
        }

        std::vector<bool> claimed(n, false);
        for (int idx : row_chain) claimed[idx] = true;
        for (int idx : col_chain) claimed[idx] = true;

        const double typical_spacing = (d1len + d2len) * 0.5;
        // Deliberately TIGHT (not the 2x-spacing tolerance an earlier
        // version used -- CLAUDE.md §6, again): H0 is already a decent fit
        // at this point (10+ correspondences), so a real grid slot's
        // prediction should land within a fraction of one grid step. A
        // generous tolerance instead let a genuinely UNDETECTED corner's
        // slot silently "steal" its NEIGHBOR's real detection (whichever
        // unclaimed candidate was nearest the failed prediction, one full
        // grid step away, was still "close enough") -- rows through the
        // stolen point then cascade an off-by-one down the rest of that
        // line. A tight tolerance instead leaves a truly missing corner's
        // slot honestly unassigned rather than mislabeling its neighbor.
        const double tol = kGridMatchTolFactor * typical_spacing * 0.5 + 2.5;

        for (int i = 0; i < kBoardCornersX; ++i) {
            for (int j = 0; j < kBoardCornersY; ++j) {
                if (already[i][j]) continue;
                float X, Y; corner_board_xy(i, j, X, Y);
                double u, v;
                if (!apply_homography_cpu(H0, X, Y, u, v)) continue;
                int best = -1; double best_d = tol;
                for (int k = 0; k < n; ++k) {
                    if (claimed[k]) continue;
                    const double dx = pts[k].x - u, dy = pts[k].y - v;
                    const double d = std::sqrt(dx * dx + dy * dy);
                    if (d < best_d) { best_d = d; best = k; }
                }
                if (best < 0) continue;
                claimed[best] = true;
                out_local[pts[best].src_index] = GridLabel{ i, j };
                final_bx.push_back(X); final_by.push_back(Y);
                final_px.push_back(pts[best].x); final_py.push_back(pts[best].y);
                ++placed;
            }
        }

        // Refit the homography from EVERY matched correspondence (more
        // accurate than the first-row/col-only H0) -- this is the
        // homography main.cu actually uses downstream for marker decode
        // and the Zhang solve.
        Homography Hfinal = solve_dlt_homography(final_bx.data(), final_by.data(), final_px.data(), final_py.data(),
                                                 static_cast<int>(final_bx.size()));
        if (!Hfinal.valid) return 0;
        hom_local = Hfinal;
        return placed;
    };

    // ---- try every plausible anchor shift for both axes (usually just the
    // ONE walk_bidirectional already measured, when a chain reached its
    // axis's full known length; up to a handful more when it did not -- see
    // finish_with_offsets()'s own header) and keep the best. -------------------
    const int row_slack = kBoardCornersX - static_cast<int>(row_chain.size());
    const int col_slack = kBoardCornersY - static_cast<int>(col_chain.size());
    std::vector<GridLabel> best_local(n);
    Homography best_hom_local{};
    int best_placed = 0;
    for (int rs = 0; rs <= std::max(0, row_slack); ++rs) {
        for (int cs = 0; cs <= std::max(0, col_slack); ++cs) {
            std::vector<GridLabel> attempt(n);
            Homography attempt_hom{};
            const int placed = finish_with_offsets(row_offset - rs, col_offset - cs, attempt.data(), attempt_hom);
            if (placed > best_placed) { best_placed = placed; best_local = attempt; best_hom_local = attempt_hom; }
        }
    }
    if (best_placed <= 0) return 0;
    for (int k = 0; k < n; ++k) out[k] = best_local[k];
    out_hom = best_hom_local;
    return best_placed;
}

}  // namespace

int order_grid_for_view(const float* cx, const float* cy, int n, GridLabel* out, Homography& out_hom)
{
    for (int k = 0; k < n; ++k) out[k] = GridLabel{};
    if (n < 4) return 0;

    std::vector<Pt> pts(n);
    for (int k = 0; k < n; ++k) pts[k] = Pt{ cx[k], cy[k], k };

    // Candidate seeds, ranked "most top-left-ish" first (smallest x+y) --
    // THEORY.md "The algorithm" names this the algorithm's one real
    // robustness limit: if the TRUE top-left-most corner is occluded or
    // missing, the next-best candidate is tried instead (a handful of
    // retries, never silently giving up on the first failure).
    std::vector<int> seed_order(n);
    for (int k = 0; k < n; ++k) seed_order[k] = k;
    std::sort(seed_order.begin(), seed_order.end(),
             [&](int a, int b) { return (pts[a].x + pts[a].y) < (pts[b].x + pts[b].y); });

    // Try every plausible seed (up to 12, or all of n if fewer) and keep
    // whichever attempt places the MOST corners -- deliberately NOT an
    // early-exit-on-"good enough" loop (an earlier version stopped at the
    // first attempt clearing kMinCornersForBoard, which could -- and did,
    // on this project's own committed views, CLAUDE.md §6 "narrate the
    // thought process, including the one that failed" -- lock in a wrong-
    // but-plausible-looking labeling from a lucky-but-incorrect seed before
    // a later, correct seed ever got a chance). Exhausting the small seed
    // budget every time is cheap (n <= a few dozen candidates per view).
    const int max_seed_attempts = std::min<int>(12, n);
    std::vector<GridLabel> best_labels(n);
    Homography best_hom{};
    int best_count = 0;
    for (int attempt = 0; attempt < max_seed_attempts; ++attempt) {
        std::vector<GridLabel> attempt_out(n);
        Homography attempt_hom{};
        const int placed = try_order_from_seed(pts, seed_order[attempt], attempt_out.data(), attempt_hom);
        if (placed > best_count) {
            best_count = placed;
            best_labels = attempt_out;
            best_hom = attempt_hom;
        }
    }

    for (int k = 0; k < n; ++k) out[k] = best_labels[k];
    out_hom = best_hom;
    return best_count;
}

// ===========================================================================
// SHARED — order_grid_marker_first_for_view: the MARKER-FIRST replacement
// for order_grid_for_view's role as "the pipeline's output of record"
// (kernels.cuh's file header names the four steps; this is the full
// derivation of step 2's label assignment, the part that is easy to get
// backwards).
//
// Why markers FIRST, and why this fixes what order_grid_for_view could
// not (THEORY.md "Numerical considerations" tells the full, honest story
// of the RETIRED path's failure modes): order_grid_for_view resolves the
// 180-degree ambiguity and axis handedness with GLOBAL cues (a chain's
// LENGTH disambiguates i vs j; a vote ACROSS the whole view decides
// identity-vs-mirrored) -- both of which degrade exactly when the view is
// hardest (large tilt shortens chains; occlusion removes points a chain
// needs; a 180-degree rotation combined with either compounds the two).
// Production ChArUco detectors (README "Prior art") sidestep this by
// finding markers FIRST: a marker's identity is a LOCAL, self-contained
// fact (read four corners, decode 24 possibilities), so it never depends
// on how much of the REST of the board is visible or how a global seed
// walk happened to start. This function reimplements that idea at the
// smallest scope this project's board supports.
//
// DERIVING THE LOCAL LABEL ASSIGNMENT (the one part worth writing out in
// full -- CLAUDE.md paragraph 6 "narrate the thought process"): a local
// quad has 4 corners (c00 = seed, c10 = the neighbor tentatively playing
// the board-X role, c01 = the neighbor tentatively playing board-Y, c11 =
// their predicted diagonal). Physical square (bx,by) -- see kernels.cuh's
// corner_board_xy -- has corners at inner-corner indices i in {bx-1,bx},
// j in {by-1,by}: (bx-1,by-1) is its "low X, low Y" corner. IF the quad's
// own (c10-role, c01-role) directions happen to align with the board's own
// (+i, +j) directions (the "identity" decode hypothesis accepts), then
// c00 IS that low corner: c00=(bx-1,by-1), c10=(bx,by-1), c01=(bx-1,by),
// c11=(bx,by). IF instead the quad's directions are the full 180-degree
// opposite (the "mirrored" hypothesis accepts -- a single square rotated
// 180 about ITS OWN center, physically identical to the whole-board
// rotation order_grid_for_view's own ambiguity lesson teaches, since only
// a whole-board rotation, never a per-square one, is physically
// achievable -- THEORY.md "The problem"), the assignment reverses:
// c00=(bx,by), c10=(bx-1,by), c01=(bx,by-1), c11=(bx-1,by-1).
//
// A THIRD possibility -- swapping WHICH neighbor plays the c10 (X) role
// vs the c01 (Y) role, keeping their physical directions fixed -- is a
// genuine, different-from-180-degrees ambiguity this function faces that
// order_grid_for_view's GLOBAL algorithm does not: order_grid_for_view
// disambiguates axis identity from chain LENGTH (the board is 7x5, not
// square), a cue that needs many points along an axis and simply does not
// exist for a single 2x2 local quad. Swapping which physical direction is
// "X" vs "Y" is a coordinate TRANSPOSE (determinant -1 relative to the
// other choice) -- mathematically indistinguishable from a REFLECTION at
// the level of raw corner geometry alone, even though nothing about the
// physical scene is actually mirrored.
//
// AN EARLY VERSION of this function tried to resolve THAT third ambiguity
// the same brute-force way as the 180-degree one: attempt BOTH axis
// assignments per quad and let the dictionary decode pick the winner.
// MEASURED (not assumed -- CLAUDE.md paragraph 8), this failed badly: this
// project's marker dictionary (scripts/make_synthetic.py's own
// generate_marker_dictionary() docstring says so explicitly) is built to
// separate its 24 codes from each other ONLY under the identity/180-degree
// reading -- it was never asked to also survive a TRANSPOSE. A one-off
// audit of every code's transpose+mirror combination against the
// committed dictionary (data/sample/marker_dictionary.csv) found EXACT
// (Hamming-0) cross-code collisions (e.g. marker 5's transpose+mirror
// reading is bit-identical to marker 23's own code) -- so brute-forcing
// the transpose hypothesis does not just risk occasional noise-driven
// false accepts, it can accept a WRONG marker with total confidence. The
// fix kept here has two parts: (1) resolve axis identity GEOMETRICALLY,
// once per view, from a robust, walk-free statistic over EVERY detected
// corner (estimate_view_axes(), below) -- so only the 180-degree
// hypothesis (which the dictionary's min-Hamming-distance design DOES
// protect, and which a one-off audit of the mirror-alone case, unlike
// transpose, found ZERO exact cross-code collisions for) is ever tested
// per quad; and (2) require an EXACT (Hamming-0) dictionary match for this
// local search specifically (ignoring the dictionary's own, more
// permissive correction_capacity, which exists to tolerate SENSOR noise
// at a single, already-known-correct hypothesis -- not to survive a
// brute-force search across 24 candidates, where the same slack makes an
// accidental match dramatically more likely, measured at ~47% per
// attempt for this dictionary's code density). Both together make marker-
// first ordering reliable in practice (README "Expected output" reports
// the measured result).
// ===========================================================================
namespace {

// estimate_view_axes — ONE robust, walk-free estimate of a view's own two
// lattice directions AND which is the board's i-axis (7 corners) vs j-axis
// (5 corners), computed from EVERY detected corner's own local nearest-
// neighbor direction and the point cloud's own extent -- never from a
// single fragile chain (contrast order_grid_for_view's per-seed walk,
// which shortens exactly when the view is hardest -- THEORY.md "Numerical
// considerations"). This is what lets the per-quad search below test only
// the 180-degree hypothesis instead of also brute-forcing which physical
// direction is "X" (this section's header comment above explains why that
// brute force is unsafe against this project's own dictionary).
//
// Method: (1) every corner's own nearest-neighbor direction, represented
// as an AXIAL (undirected-line) quantity via the doubled-angle trick (see
// step 1's own comment below -- this is the part an earlier version of
// this function got wrong); (2) two clusters, split by the sign of a dot
// product against a reference direction (the first corner's own) in that
// doubled-angle space -- a real lattice has exactly two axis directions
// roughly 90 degrees apart (180 degrees apart once doubled), so this
// simple sign test separates them robustly; (3) project every corner onto
// each cluster's own averaged axis and measure the RANGE divided by that
// axis's median step length -- an estimate of "how many grid steps does
// this axis span". The axis spanning MORE steps is the i-axis
// (kBoardCornersX=7 > kBoardCornersY=5, the board's own non-square aspect
// ratio -- the SAME fact order_grid_for_view's chain-length disambiguation
// exploits, applied here to the WHOLE point cloud's extent instead of one
// walked chain, so it degrades gracefully rather than catastrophically
// when some corners are missing).
//
// Returns false if fewer than 4 corners, or the two clusters cannot be
// separated (e.g. every direction points the same way -- a degenerate
// point set, such as the negative control's near-empty candidate set).
bool estimate_view_axes(const std::vector<Pt>& pts, double& ix, double& iy, double& jx, double& jy)
{
    const int n = static_cast<int>(pts.size());
    if (n < 4) return false;

    // step 1: every corner's own nearest-neighbor direction, represented
    // [0, pi) is NOT enough by itself: two corners can legitimately find
    // their nearest neighbor along the SAME physical lattice line but in
    // OPPOSITE senses (one seed's neighbor sits to its "east", another's to
    // its "west") -- naively averaging their raw (cos,sin) unit vectors
    // then CANCELS instead of reinforcing (measured, not assumed: an
    // earlier version of this function did exactly that and produced
    // garbage axis estimates on several of this project's own 8 views).
    // The fix is the standard circular-statistics trick for AXIAL data
    // (an undirected line, not a direction): work in DOUBLED-angle space,
    // (cos(2*theta), sin(2*theta)) -- theta and theta+pi (the same line,
    // opposite senses) map to the IDENTICAL doubled point, so averaging
    // never cancels, and the final axis direction is recovered by halving
    // the averaged angle back down.
    std::vector<double> dux(n, 0.0), duy(n, 0.0), stepd(n, 0.0);
    std::vector<bool> has(n, false);
    for (int a = 0; a < n; ++a) {
        int best = -1; double best_d = 1e30;
        for (int b = 0; b < n; ++b) {
            if (a == b) continue;
            const double dx = pts[a].x - pts[b].x, dy = pts[a].y - pts[b].y;
            const double d = std::sqrt(dx * dx + dy * dy);
            if (d < best_d) { best_d = d; best = b; }
        }
        if (best < 0) continue;
        const double theta = std::atan2(pts[best].y - pts[a].y, pts[best].x - pts[a].x);
        dux[a] = std::cos(2.0 * theta); duy[a] = std::sin(2.0 * theta);
        stepd[a] = best_d; has[a] = true;
    }

    // step 2: two clusters in DOUBLED-angle space, gated against the first
    // valid direction found -- a real lattice's two axes sit ~90 degrees
    // apart in ORIGINAL angle, i.e. ~180 degrees apart (opposite points) in
    // DOUBLED-angle space, so a plain "which reference is closer" dot-
    // product sign test cleanly separates them.
    int ref = -1;
    for (int k = 0; k < n; ++k) if (has[k]) { ref = k; break; }
    if (ref < 0) return false;
    double sum_ax = 0.0, sum_ay = 0.0, sum_bx = 0.0, sum_by = 0.0;
    std::vector<double> steps_a, steps_b;
    for (int k = 0; k < n; ++k) {
        if (!has[k]) continue;
        const double dot = dux[k] * dux[ref] + duy[k] * duy[ref];
        if (dot >= 0.0) { sum_ax += dux[k]; sum_ay += duy[k]; steps_a.push_back(stepd[k]); }
        else            { sum_bx += dux[k]; sum_by += duy[k]; steps_b.push_back(stepd[k]); }
    }
    if (steps_a.size() < 2 || steps_b.size() < 2) return false;   // could not separate two axes robustly
    if (std::sqrt(sum_ax * sum_ax + sum_ay * sum_ay) < 1e-9) return false;
    if (std::sqrt(sum_bx * sum_bx + sum_by * sum_by) < 1e-9) return false;
    // Halve the averaged DOUBLED angle back down to an ORIGINAL-space
    // direction (either of the two opposite halving solutions is an
    // equally valid representative -- callers only ever use fabs(dot),
    // so an overall sign flip here changes nothing downstream).
    const double phi_a = std::atan2(sum_ay, sum_ax), phi_b = std::atan2(sum_by, sum_bx);
    const double ax = std::cos(phi_a * 0.5), ay = std::sin(phi_a * 0.5);
    const double bx = std::cos(phi_b * 0.5), by = std::sin(phi_b * 0.5);

    // step 3: extent along each axis, measured in grid steps (pixel range
    // divided by that axis's own median step length).
    auto median_of = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
    const double step_a = median_of(steps_a), step_b = median_of(steps_b);
    double amin = 1e30, amax = -1e30, bmin = 1e30, bmax = -1e30;
    for (int k = 0; k < n; ++k) {
        const double pa = pts[k].x * ax + pts[k].y * ay;
        const double pb = pts[k].x * bx + pts[k].y * by;
        amin = std::min(amin, pa); amax = std::max(amax, pa);
        bmin = std::min(bmin, pb); bmax = std::max(bmax, pb);
    }
    const double span_a = (step_a > 1e-6) ? (amax - amin) / step_a : 0.0;
    const double span_b = (step_b > 1e-6) ? (bmax - bmin) / step_b : 0.0;

    if (span_a >= span_b) { ix = ax; iy = ay; jx = bx; jy = by; }
    else                  { ix = bx; iy = by; jx = ax; jy = ay; }
    return true;
}

// find_local_quad — LOCAL geometric search only (no global walk, no
// seed-retry budget): reuses try_order_from_seed's own nearest-neighbor /
// most-orthogonal-neighbor searches (pure local geometry, safe to reuse
// verbatim) to find a candidate 2x2 corner cluster anchored at `seed`, then
// predicts the 4th (diagonal) corner and snaps it to the nearest ACTUAL
// detection within a tight tolerance (same formula finish_with_offsets
// uses, for the same reason: a generous tolerance risks silently matching
// the wrong point). Deliberately does NOT attempt to determine axis
// identity or sign here -- see this file's comment above for why that is
// resolved by brute-force marker decode in the caller instead of geometry.
// Returns false if no plausible local quad exists (board edge, sparse
// detections, or a genuinely isolated point).
bool find_local_quad(const std::vector<Pt>& pts, int seed, int& out_n1, int& out_n2, int& out_n3)
{
    const int n = static_cast<int>(pts.size());
    auto dist = [&](int a, int b) {
        const double dx = pts[a].x - pts[b].x, dy = pts[a].y - pts[b].y;
        return std::sqrt(dx * dx + dy * dy);
    };

    // direction 1: nearest neighbor to the seed.
    int n1 = -1; double best_d1 = 1e30;
    for (int k = 0; k < n; ++k) {
        if (k == seed) continue;
        const double d = dist(seed, k);
        if (d < best_d1) { best_d1 = d; n1 = k; }
    }
    if (n1 < 0) return false;
    double d1x = pts[n1].x - pts[seed].x, d1y = pts[n1].y - pts[seed].y;
    const double d1len = std::sqrt(d1x * d1x + d1y * d1y);
    if (d1len < 1e-6) return false;
    d1x /= d1len; d1y /= d1len;

    // direction 2: the most-orthogonal-to-d1 neighbor at a comparable step
    // length (same magnitude-band reasoning try_order_from_seed's own
    // search_n2 uses -- a farther point lying on the exact same grid line
    // can look just as orthogonal to d1 as the true adjacent corner does).
    std::vector<int> order_by_dist;
    for (int k = 0; k < n; ++k) if (k != seed && k != n1) order_by_dist.push_back(k);
    std::sort(order_by_dist.begin(), order_by_dist.end(),
             [&](int a, int b) { return dist(seed, a) < dist(seed, b); });
    const int consider = std::min<int>(12, static_cast<int>(order_by_dist.size()));

    // MEASURED, not assumed (CLAUDE.md paragraph 8): an earlier version of
    // this function copied try_order_from_seed's OWN fallback verbatim --
    // "if the magnitude-banded search finds nothing, retry unrestricted"
    // (that function's own comment: "band empty (e.g. occlusion) -- fall
    // back, honestly weaker"). For a WALKED CHAIN, a weaker fallback point
    // is a minor accuracy hit. For a LOCAL QUAD, it is much more damaging:
    // when the true axis-2 neighbor is missing (a corner the saddle
    // detector missed, or partial occlusion), the unrestricted fallback
    // regularly locked onto a DIAGONAL point instead (one full step along
    // EACH axis, not one step along a single axis) -- silently building a
    // "quad" that is not a real unit cell at all. The resulting DLT
    // homography is subtly wrong (scaled/sheared to fit a diagonal as if
    // it were an axis step), which reads every marker cell a fraction of a
    // cell off -- exactly the symptom this project's own build diagnosed:
    // every attempted quad on one of the 8 committed views landing 1-2
    // payload bits short of ANY dictionary code, never zero, on EVERY
    // seed, regardless of axis assignment. The fix: NO fallback. A local
    // quad this function cannot confirm as a genuine unit cell is refused
    // outright (the seed is simply skipped -- main.cu's own per-seed loop
    // tries every OTHER corner as a seed too, so a skipped seed here costs
    // yield, never correctness).
    int n2 = -1; double best_abs_cos = 2.0;
    for (int t = 0; t < consider; ++t) {
        const int k = order_by_dist[t];
        double dx = pts[k].x - pts[seed].x, dy = pts[k].y - pts[seed].y;
        const double dl = std::sqrt(dx * dx + dy * dy);
        if (dl < 1e-6) continue;
        if (dl < 0.5 * d1len || dl > 1.8 * d1len) continue;   // magnitude band, NOT relaxed on failure
        dx /= dl; dy /= dl;
        const double abs_cos = std::fabs(dx * d1x + dy * d1y);
        if (abs_cos < best_abs_cos) { best_abs_cos = abs_cos; n2 = k; }
    }
    // A second sanity floor even within the band: the two step directions
    // of a real board square are close to perpendicular (some deviation is
    // expected from perspective shear, but a near-45-degree "orthogonal
    // best" is the signature of the diagonal-point failure mode above, not
    // a sheared true square).
    if (n2 < 0 || best_abs_cos > 0.5) return false;
    const double d2x = pts[n2].x - pts[seed].x, d2y = pts[n2].y - pts[seed].y;
    const double d2len = std::sqrt(d2x * d2x + d2y * d2y);

    // predict the diagonal 4th corner (seed + d1-step + d2-step) and snap
    // to the nearest ACTUAL detection within a tight tolerance.
    const double px = pts[seed].x + (pts[n1].x - pts[seed].x) + d2x;
    const double py = pts[seed].y + (pts[n1].y - pts[seed].y) + d2y;
    const double typical = (d1len + d2len) * 0.5;
    const double tol = kGridMatchTolFactor * typical * 0.5 + 2.5;   // same formula as finish_with_offsets

    int best = -1; double best_d = tol;
    for (int k = 0; k < n; ++k) {
        if (k == seed || k == n1 || k == n2) continue;
        const double dx = pts[k].x - px, dy = pts[k].y - py;
        const double d = std::sqrt(dx * dx + dy * dy);
        if (d < best_d) { best_d = d; best = k; }
    }
    if (best < 0) return false;

    out_n1 = n1; out_n2 = n2; out_n3 = best;
    return true;
}

}  // namespace

int order_grid_marker_first_for_view(const float* cx, const float* cy, int n,
                                     const unsigned char* gray, int view,
                                     const uint16_t* true_codes, int correction_capacity,
                                     GridLabel* out, Homography& out_hom,
                                     int* out_quads_decoded, int* out_anchor_conflicts)
{
    for (int k = 0; k < n; ++k) out[k] = GridLabel{};
    out_hom = Homography{};
    if (out_quads_decoded) *out_quads_decoded = 0;
    if (out_anchor_conflicts) *out_anchor_conflicts = 0;
    if (n < 4) return 0;

    std::vector<Pt> pts(n);
    for (int k = 0; k < n; ++k) pts[k] = Pt{ cx[k], cy[k], k };

    // ---- Step 0: fix axis identity ONCE for the whole view (see
    // estimate_view_axes' own header for why this replaces a per-quad
    // brute-force search -- this project's own dictionary cannot safely
    // resolve a transpose ambiguity, only a 180-degree one). -----------------
    double ix = 0, iy = 0, jx = 0, jy = 0;
    if (!estimate_view_axes(pts, ix, iy, jx, jy)) return 0;   // degenerate point set -- honest failure

    // code_symmetric[c] -- a PURE, image-independent fact about dictionary
    // code c, computed once: does its 9-bit payload read bit-for-bit
    // IDENTICAL under a 180-degree mirror? Two of this dictionary's own 24
    // codes are (found by a one-off audit -- this section's header cites
    // it). This matters below: for such a code, no image evidence can EVER
    // tell identity from mirrored apart (the sampled bits are the same
    // either way) -- that is a fact about the CODE, not a measurement
    // failure, so it needs a DIFFERENT handling than a genuinely ambiguous
    // (ie. wrong) decode.
    std::vector<bool> code_symmetric(kNumMarkerCodes, false);
    for (int c = 0; c < kNumMarkerCodes; ++c) {
        bool sym = true;
        for (int pr = 0; pr < kMarkerPayloadN && sym; ++pr) {
            for (int pc = 0; pc < kMarkerPayloadN; ++pc) {
                const int bit  = (true_codes[c] >> marker_payload_bit_index(pr, pc)) & 1;
                const int mbit = (true_codes[c] >> marker_payload_bit_index(kMarkerPayloadN - 1 - pr, kMarkerPayloadN - 1 - pc)) & 1;
                if (bit != mbit) { sym = false; break; }
            }
        }
        code_symmetric[c] = sym;
    }

    // ---- Step 1+2: for every corner as a candidate quad seed, find its
    // local quad (if any), assign axis roles from the Step-0 estimate (no
    // brute force), and try the ONE remaining ambiguity the dictionary DOES
    // protect against -- identity vs 180-degree-mirrored -- over every
    // dictionary code (kNumMarkerCodes=24, so <= 48 decode attempts per
    // seed, each a handful of bilinear samples). NOT negligible the way the
    // retired algorithm's grid ordering was: this search, across all 8
    // views, measures ~6 ms total -- MORE than the combined GPU pixel-
    // parallel stages (~0.7-0.9 ms) -- an honest Amdahl update, not the
    // "utterly dominated" claim the retired algorithm could make (THEORY.md
    // "The GPU mapping" and README "Expected output" report the measured
    // numbers and name the GPU-port opportunity this now represents). Each
    // successfully-decoded quad is
    // stored, not yet turned into (i,j) proposals -- a self-mirror-
    // symmetric marker (code_symmetric[]) can identify ITS OWN square with
    // full confidence while leaving its OWN orientation genuinely
    // undetermined; resolving that needs the OTHER, unambiguous quads'
    // consensus first (Step 2b, below the seed loop). -------------------------
    struct QuadResult { int seed, c10_idx, c01_idx, n3, marker_id; bool orientation_known, mirrored; };
    std::vector<QuadResult> quad_results;

    for (int seed = 0; seed < n; ++seed) {
        int n1, n2, n3;
        if (!find_local_quad(pts, seed, n1, n2, n3)) continue;

        // Classify n1/n2 by which is more parallel to the view's own
        // i-axis (ix,iy) -- the SIGN doesn't matter here (fabs), only
        // which physical direction is i-like vs j-like; the sign/180
        // question is exactly what the mirrored decode hypothesis below
        // resolves.
        auto axis_score_i = [&](int idx) {
            double dx = pts[idx].x - pts[seed].x, dy = pts[idx].y - pts[seed].y;
            const double dl = std::sqrt(dx * dx + dy * dy);
            if (dl < 1e-9) return 0.0;
            dx /= dl; dy /= dl;
            return std::fabs(dx * ix + dy * iy);
        };
        const double score1 = axis_score_i(n1), score2 = axis_score_i(n2);
        // MEASURED, not assumed (CLAUDE.md paragraph 8): when n1's and n2's
        // own axis scores are too CLOSE to call, this quad's local geometry
        // does not confidently say which neighbor is the i-role -- normally
        // harmless (a wrong axis assignment almost never decodes at all,
        // this section's header explains why), EXCEPT for a marker code
        // that happens to read IDENTICALLY under a transpose (this
        // project's own dictionary has two such self-transpose-symmetric
        // codes, found by the same one-off audit this file's header cites)
        // -- there, a low-confidence WRONG axis pick can still decode
        // cleanly, silently swapping that one quad's i/j roles (observed on
        // two of this project's own 8 committed views as a clean
        // NEIGHBOR-SWAP pair, e.g. truth (0,1)/(1,1) trading labels).
        // Refusing any quad whose axis pick is not clearly won removes this
        // failure mode at the source, at the cost of a little yield.
        if (std::fabs(score1 - score2) < 0.3) continue;
        const int c10_idx = (score1 >= score2) ? n1 : n2;   // the i(X)-role neighbor
        const int c01_idx = (c10_idx == n1) ? n2 : n1;      // the j(Y)-role neighbor

        // HANDEDNESS (a hard geometric constraint, not something the
        // decode step can ever resolve -- a real, distinct bug this
        // project's own build caught, CLAUDE.md paragraph 6 "narrate the
        // thought process, including the one that failed"): axis_score_i
        // above only asks "which neighbor is MORE i-like" -- it does NOT
        // check that (c10-seed, c01-seed) forms a PROPER, non-reflected
        // pair. A real camera image of a flat board is NEVER reflected (a
        // rigid rotation, 0 or 180 degrees in-plane, is the only physical
        // freedom -- THEORY.md "The problem"), so cross(i-direction,
        // j-direction) has a FIXED sign, matching corner_board_xy's own
        // (i+,j+) convention under this project's image axes (x-right,
        // y-down): a frontal view (view00) MEASURES this cross product as
        // positive. If axis_score_i's independent "which is more i-like"
        // pick happens to land on the WRONG-HANDED neighbor pairing (can
        // happen: nothing about "most parallel to the i-axis" guarantees
        // proper orientation), the resulting local frame is a REFLECTION,
        // not a rotation -- decode's identity/mirrored test only ever
        // covers the two PROPER cases (0 or 180 degrees), so a reflected
        // frame can silently "succeed" by accident (observed directly on
        // this project's own view06: a marker whose bit pattern happens to
        // be symmetric under this specific single-axis flip). Refusing any
        // quad whose (c10,c01) pair is not properly handed removes this
        // failure mode at its true geometric source.
        {
            const double cx10 = pts[c10_idx].x - pts[seed].x, cy10 = pts[c10_idx].y - pts[seed].y;
            const double cx01 = pts[c01_idx].x - pts[seed].x, cy01 = pts[c01_idx].y - pts[seed].y;
            const double cross = cx10 * cy01 - cy10 * cx01;
            if (cross <= 0.0) continue;
        }

        // Board-plane correspondences for THIS quad's own tiny DLT fit:
        // local origin (this quad's own seed) at board (0,0), the X-role
        // neighbor at (kSquareSizeM,0), the Y-role neighbor at
        // (0,kSquareSizeM), the predicted diagonal at (kSquareSizeM,
        // kSquareSizeM) -- a ONE-SQUARE-large local board frame, distinct
        // from (and much smaller than) order_grid_for_view's own
        // whole-board frame.
        float bxq[4] = { 0.0f, kSquareSizeM, 0.0f, kSquareSizeM };
        float byq[4] = { 0.0f, 0.0f, kSquareSizeM, kSquareSizeM };
        float pxq[4] = { pts[seed].x, pts[c10_idx].x, pts[c01_idx].x, pts[n3].x };
        float pyq[4] = { pts[seed].y, pts[c10_idx].y, pts[c01_idx].y, pts[n3].y };
        const Homography Hq = solve_dlt_homography(bxq, byq, pxq, pyq, 4);
        if (!Hq.valid) continue;

        // Sample position is ALWAYS this quad's own local (0,0) square for
        // BOTH hypotheses -- unlike the GLOBAL marker-decode path
        // (marker_decode_one_cpu), which shifts the sample position to
        // mirror_square(bx,by) for its mirrored hypothesis because ITS
        // homography spans many squares under one shared labeling
        // convention. A lone local quad has no "other square index" to
        // confuse itself with -- only its own reading orientation is in
        // question (this section's header derives this in full).
        // EXACT-MATCH acceptance (this section's header derives WHY a
        // plain "accept if hamming <= dictionary correction_capacity" test
        // is unsafe here): require Hamming-0 -- a bit-for-bit exact read --
        // ignoring the dictionary's own, more permissive correction_
        // capacity entirely (that capacity exists to tolerate SENSOR noise
        // at a single, already-known-correct hypothesis; searched across
        // 24 candidates x 2 hypotheses instead, that same slack becomes a
        // false-accept liability, measured directly on this project's own
        // committed views: relaxing to Hamming<=1 here, even gated behind
        // "only if the winner is unique", was tried and MEASURABLY made
        // several already-correct views worse, not better -- an honest
        // negative result kept here as a comment, not silently discarded).
        int best_id = -1; bool best_mirrored = false; int best_ham = 999;
        for (int cand = 0; cand < kNumMarkerCodes; ++cand) {
            bool border_id, acc_id; int ham_id;
            decode_one_hypothesis_cpu(gray, view, Hq, 0, 0, /*mirrored=*/false,
                                      true_codes[cand], /*correction_capacity=*/0, border_id, acc_id, ham_id);
            if (acc_id && ham_id < best_ham) { best_ham = ham_id; best_id = cand; best_mirrored = false; }
            bool border_mir, acc_mir; int ham_mir;
            decode_one_hypothesis_cpu(gray, view, Hq, 0, 0, /*mirrored=*/true,
                                      true_codes[cand], /*correction_capacity=*/0, border_mir, acc_mir, ham_mir);
            if (acc_mir && ham_mir < best_ham) { best_ham = ham_mir; best_id = cand; best_mirrored = true; }
        }
        if (best_id < 0) continue;   // no dictionary code decoded cleanly -- black square, occlusion, or too degraded

        // MEASURED, not assumed (CLAUDE.md paragraph 8): a SECOND pass,
        // counting how many (candidate, hypothesis) pairs tie at best_ham,
        // catches a real failure mode the single pass above cannot see --
        // two of this dictionary's own 24 codes are MIRROR-SELF-SYMMETRIC
        // (code_symmetric[], computed above): their 9-bit payload reads
        // bit-for-bit IDENTICAL whether sampled identity or mirrored. Two
        // tie outcomes are possible, and they mean DIFFERENT things:
        //   * the tie is the SAME candidate id under both hypotheses, AND
        //     that code is a KNOWN self-symmetric one -- this quad
        //     correctly, confidently identified ITS OWN square (best_id),
        //     it just cannot -- CANNOT, not "failed to" -- tell its own
        //     orientation from image evidence alone. Recorded with
        //     orientation_known=false; Step 2b below resolves it from the
        //     view's OTHER, unambiguous quads.
        //   * any other tie (a different candidate ties, or more than 2
        //     hypotheses tie) is a genuine, unexplained ambiguity -- this
        //     was OBSERVED directly on two of this project's own 8
        //     committed views before this split existed (a clean, symmetric
        //     NEIGHBOR-SWAP pair, e.g. truth (0,1)/(1,1) trading labels,
        //     traced to exactly this kind of quad being silently resolved
        //     the wrong way) -- refused outright, never guessed.
        int tie_count = 0; bool tie_all_same_id = true;
        for (int cand = 0; cand < kNumMarkerCodes; ++cand) {
            bool border_id, acc_id; int ham_id;
            decode_one_hypothesis_cpu(gray, view, Hq, 0, 0, /*mirrored=*/false,
                                      true_codes[cand], /*correction_capacity=*/0, border_id, acc_id, ham_id);
            if (acc_id && ham_id == best_ham) { ++tie_count; if (cand != best_id) tie_all_same_id = false; }
            bool border_mir, acc_mir; int ham_mir;
            decode_one_hypothesis_cpu(gray, view, Hq, 0, 0, /*mirrored=*/true,
                                      true_codes[cand], /*correction_capacity=*/0, border_mir, acc_mir, ham_mir);
            if (acc_mir && ham_mir == best_ham) { ++tie_count; if (cand != best_id) tie_all_same_id = false; }
        }
        if (tie_count == 1) {
            quad_results.push_back(QuadResult{ seed, c10_idx, c01_idx, n3, best_id, /*orientation_known=*/true, best_mirrored });
        } else if (tie_count == 2 && tie_all_same_id && code_symmetric[static_cast<size_t>(best_id)]) {
            quad_results.push_back(QuadResult{ seed, c10_idx, c01_idx, n3, best_id, /*orientation_known=*/false, false });
        }
        // else: genuinely ambiguous (a different candidate tied, or more
        // than 2 hypotheses tied) -- refuse, never guess.
    }

    // ---- Step 2b: resolve orientation for self-symmetric-marker quads
    // from the view's OWN majority of UNAMBIGUOUS quads -- the same
    // physical fact order_grid_for_view's own ambiguity lesson rests on
    // (a rigid board is either presented normally or rotated 180 degrees,
    // uniformly, never per-square), applied here as a simple vote instead
    // of a walked/anchored global homography, so it inherits none of that
    // path's fragility. If EVERY decoded quad in this view happens to be
    // self-symmetric (no unambiguous quad to vote from -- vanishingly
    // rare, and never observed on this project's own 8 committed views),
    // the fallback below defaults to "identity", an honest, arbitrary
    // choice for a case with no evidence at all either way.
    // -------------------------------------------------------------------------
    int votes_identity = 0, votes_mirrored = 0;
    for (const auto& q : quad_results)
        if (q.orientation_known) { if (q.mirrored) ++votes_mirrored; else ++votes_identity; }
    const bool orientation_fallback_mirrored = votes_mirrored > votes_identity;

    int quads_decoded = 0;
    struct Proposal { int idx; GridLabel lbl; };
    std::vector<Proposal> proposals;
    // A quad seeded next to a board-EDGE square proposes a label outside
    // [0,kBoardCornersX) x [0,kBoardCornersY) for the corner that would sit
    // past the board's own outer silhouette (e.g. bx=0's l00.i = -1: no
    // real inner corner exists "to its own left"). Drop only those OUT-OF-
    // RANGE proposals -- the quad's other, in-range corners are still
    // perfectly good anchors.
    auto push_if_valid = [&](int idx, GridLabel lbl) {
        if (lbl.i < 0 || lbl.i >= kBoardCornersX || lbl.j < 0 || lbl.j >= kBoardCornersY) return;
        proposals.push_back(Proposal{ idx, lbl });
    };
    for (const auto& q : quad_results) {
        ++quads_decoded;
        const bool mirrored = q.orientation_known ? q.mirrored : orientation_fallback_mirrored;
        // ---- label assignment -- see this section's header comment for
        // the full geometric derivation of both branches. --------------------
        int bx, by; square_of_marker_id(q.marker_id, bx, by);
        GridLabel l00, l10, l01, l11;
        if (!mirrored) {
            l00 = GridLabel{ bx - 1, by - 1 }; l10 = GridLabel{ bx, by - 1 };
            l01 = GridLabel{ bx - 1, by };     l11 = GridLabel{ bx, by };
        } else {
            l00 = GridLabel{ bx, by };         l10 = GridLabel{ bx - 1, by };
            l01 = GridLabel{ bx, by - 1 };     l11 = GridLabel{ bx - 1, by - 1 };
        }
        push_if_valid(q.seed, l00);
        push_if_valid(q.c10_idx, l10);
        push_if_valid(q.c01_idx, l01);
        push_if_valid(q.n3, l11);
    }
    if (out_quads_decoded) *out_quads_decoded = quads_decoded;

    // ---- Step 3: vote. A corner reached by more than one decoded quad
    // (every interior corner touches up to 4 squares, so this is common)
    // must see EVERY proposal agree, or a STRICT majority, before it is
    // trusted -- a tie is left unanchored rather than guessed (never
    // fabricate a label; the predict-remaining phase below will still
    // usually recover it from its neighbors' own anchors). -----------------
    std::vector<bool> anchored(n, false);
    std::vector<GridLabel> anchor_label(n, GridLabel{});
    std::vector<int> anchor_votes(n, 0);   // winning proposal count -- used below to break a DIFFERENT kind of conflict
    int anchor_conflicts = 0;
    for (int idx = 0; idx < n; ++idx) {
        std::vector<GridLabel> distinct; std::vector<int> counts; int total = 0;
        for (const auto& p : proposals) {
            if (p.idx != idx) continue;
            ++total;
            bool found = false;
            for (size_t k = 0; k < distinct.size(); ++k) {
                if (distinct[k].i == p.lbl.i && distinct[k].j == p.lbl.j) { ++counts[k]; found = true; break; }
            }
            if (!found) { distinct.push_back(p.lbl); counts.push_back(1); }
        }
        if (total == 0) continue;
        int best_k = 0;
        for (size_t k = 1; k < distinct.size(); ++k) if (counts[k] > counts[best_k]) best_k = static_cast<int>(k);
        if (distinct.size() > 1) ++anchor_conflicts;
        if (counts[best_k] * 2 <= total) continue;   // no strict majority -- leave unanchored, honestly
        anchored[idx] = true;
        anchor_label[idx] = distinct[static_cast<size_t>(best_k)];
        anchor_votes[idx] = counts[static_cast<size_t>(best_k)];
    }

    // ---- Step 3b: a SECOND, DIFFERENT kind of conflict the per-index vote
    // above cannot see (measured, not assumed -- CLAUDE.md paragraph 8:
    // this project's own build produced it on two of its 8 committed
    // views): two DIFFERENT corner indices, each individually winning ITS
    // OWN majority vote, can still end up claiming the SAME (i,j) label --
    // a genuinely inconsistent quad (usually the diagonal corner of one
    // quad accidentally coinciding with a neighboring quad's own seed)
    // wins a local majority without anything checking it against the
    // board's own "one corner per (i,j)" constraint. Group anchored
    // corners by their label; wherever more than one corner claims the
    // SAME (i,j), keep only the strongest-supported one (most winning
    // votes) and un-anchor the rest -- the predict-remaining phase below
    // will try to recover them honestly instead.
    {
        std::vector<int> owner(static_cast<size_t>(kBoardCornersX) * kBoardCornersY, -1);
        for (int idx = 0; idx < n; ++idx) {
            if (!anchored[idx]) continue;
            const size_t slot = static_cast<size_t>(anchor_label[idx].i) * kBoardCornersY + anchor_label[idx].j;
            if (owner[slot] < 0) { owner[slot] = idx; continue; }
            // a second (or later) claimant on the same slot -- keep whichever has more votes.
            ++anchor_conflicts;
            const int prev = owner[slot];
            if (anchor_votes[idx] > anchor_votes[prev]) {
                anchored[prev] = false; owner[slot] = idx;
            } else {
                anchored[idx] = false;
            }
        }
    }
    if (out_anchor_conflicts) *out_anchor_conflicts = anchor_conflicts;

    // ---- Step 4: fit ONE global homography from every marker-anchored
    // correspondence, then predict + snap every remaining grid slot to an
    // unclaimed detected corner (same tight-tolerance discipline as
    // order_grid_for_view's own finish_with_offsets), then refit ONCE more
    // from every final correspondence ("refine once"). ----------------------
    std::vector<float> bxl, byl, pxl, pyl;
    for (int k = 0; k < n; ++k) {
        if (!anchored[k]) continue;
        float X, Y; corner_board_xy(anchor_label[k].i, anchor_label[k].j, X, Y);
        bxl.push_back(X); byl.push_back(Y); pxl.push_back(pts[k].x); pyl.push_back(pts[k].y);
    }
    const int n_anchored = static_cast<int>(bxl.size());
    if (n_anchored < 4) return 0;   // no markers decoded at all -- honest failure (e.g. the negative control)

    const Homography Hg = solve_dlt_homography(bxl.data(), byl.data(), pxl.data(), pyl.data(), n_anchored);
    if (!Hg.valid) return 0;

    for (int k = 0; k < n; ++k) if (anchored[k]) out[pts[k].src_index] = anchor_label[k];

    // typical local corner spacing, estimated from every point's own
    // nearest-neighbor distance -- scale-adaptive across this project's 8
    // views' differing depths, same discipline order_grid_for_view uses.
    double typical_spacing = 20.0;
    {
        double sum_nn = 0.0; int cnt = 0;
        for (int a = 0; a < n; ++a) {
            double best = 1e30;
            for (int b = 0; b < n; ++b) {
                if (a == b) continue;
                const double dx = pts[a].x - pts[b].x, dy = pts[a].y - pts[b].y;
                const double d = std::sqrt(dx * dx + dy * dy);
                if (d < best) best = d;
            }
            if (best < 1e29) { sum_nn += best; ++cnt; }
        }
        if (cnt > 0) typical_spacing = sum_nn / cnt;
    }
    const double tol = kGridMatchTolFactor * typical_spacing * 0.5 + 2.5;

    bool already[kBoardCornersX][kBoardCornersY] = {};
    for (int k = 0; k < n; ++k)
        if (out[k].i >= 0 && out[k].i < kBoardCornersX && out[k].j >= 0 && out[k].j < kBoardCornersY)
            already[out[k].i][out[k].j] = true;
    std::vector<bool> claimed(n, false);
    for (int k = 0; k < n; ++k) if (anchored[k]) claimed[k] = true;

    int placed = n_anchored;
    for (int i = 0; i < kBoardCornersX; ++i) {
        for (int j = 0; j < kBoardCornersY; ++j) {
            if (already[i][j]) continue;
            float X, Y; corner_board_xy(i, j, X, Y);
            double u, v;
            if (!apply_homography_cpu(Hg, X, Y, u, v)) continue;
            int best = -1; double best_d = tol;
            for (int k = 0; k < n; ++k) {
                if (claimed[k]) continue;
                const double dx = pts[k].x - u, dy = pts[k].y - v;
                const double d = std::sqrt(dx * dx + dy * dy);
                if (d < best_d) { best_d = d; best = k; }
            }
            if (best < 0) continue;
            claimed[best] = true;
            out[pts[best].src_index] = GridLabel{ i, j };
            ++placed;
        }
    }

    // ---- refine once: refit from every final correspondence -- this is
    // the homography main.cu carries forward into the Zhang solve. ---------
    std::vector<float> fbx, fby, fpx, fpy;
    for (int k = 0; k < n; ++k) {
        if (out[k].i < 0) continue;
        float X, Y; corner_board_xy(out[k].i, out[k].j, X, Y);
        fbx.push_back(X); fby.push_back(Y); fpx.push_back(pts[k].x); fpy.push_back(pts[k].y);
    }
    const Homography Hfinal = solve_dlt_homography(fbx.data(), fby.data(), fpx.data(), fpy.data(),
                                                    static_cast<int>(fbx.size()));
    out_hom = Hfinal.valid ? Hfinal : Hg;
    return placed;
}

// ===========================================================================
// SHARED — Zhang's mini-calibration.
// ===========================================================================

// jacobi_eigen_symmetric6 — classic cyclic Jacobi eigenvalue sweep (Golub &
// Van Loan, "Matrix Computations", the standard reference algorithm): at
// each step, pick an off-diagonal pair (p,q), compute the rotation angle
// that zeroes A[p][q], apply it to A (a similarity transform, so
// eigenvalues are preserved) and accumulate it into eigvecs (so its columns
// converge to the eigenvectors). Repeated sweeps drive every off-diagonal
// entry to ~0; the diagonal then holds the eigenvalues. THEORY.md "The
// math" walks the rotation-angle derivation.
void jacobi_eigen_symmetric6(double A[6][6], double eigvecs[6][6])
{
    constexpr int N = 6;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            eigvecs[i][j] = (i == j) ? 1.0 : 0.0;

    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < N; ++p)
            for (int q = p + 1; q < N; ++q) off += A[p][q] * A[p][q];
        if (off < 1e-30) break;   // numerically diagonal already

        for (int p = 0; p < N - 1; ++p) {
            for (int q = p + 1; q < N; ++q) {
                if (std::fabs(A[p][q]) < 1e-300) continue;
                const double theta = (A[q][q] - A[p][p]) / (2.0 * A[p][q]);
                const double t = (theta >= 0.0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;
                const double app = A[p][p], aqq = A[q][q], apq = A[p][q];
                A[p][p] = app - t * apq;
                A[q][q] = aqq + t * apq;
                A[p][q] = 0.0; A[q][p] = 0.0;
                for (int k = 0; k < N; ++k) {
                    if (k == p || k == q) continue;
                    const double akp = A[k][p], akq = A[k][q];
                    A[k][p] = A[p][k] = c * akp - s * akq;
                    A[k][q] = A[q][k] = s * akp + c * akq;
                }
                for (int k = 0; k < N; ++k) {
                    const double vkp = eigvecs[k][p], vkq = eigvecs[k][q];
                    eigvecs[k][p] = c * vkp - s * vkq;
                    eigvecs[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }
}

// v_pq — Zhang's constraint-row builder (Zhang 2000, eq. 8): for a
// homography H (row-major 3x3), returns the length-6 vector v such that
// h_p^T * omega * h_q = v . b, where h_p is H's COLUMN p (0-indexed) and
// b = [B11,B12,B22,B13,B23,B33] are omega's upper-triangular entries.
// THEORY.md "The math" derives this from omega = K^-T K^-1 and the
// orthonormality of a rotation matrix's own columns.
static void v_pq(const Homography& H, int p, int q, double out[6])
{
    const double hp[3] = { H.h[0 * 3 + p], H.h[1 * 3 + p], H.h[2 * 3 + p] };
    const double hq[3] = { H.h[0 * 3 + q], H.h[1 * 3 + q], H.h[2 * 3 + q] };
    out[0] = hp[0] * hq[0];
    out[1] = hp[0] * hq[1] + hp[1] * hq[0];
    out[2] = hp[1] * hq[1];
    out[3] = hp[0] * hq[2] + hp[2] * hq[0];
    out[4] = hp[1] * hq[2] + hp[2] * hq[1];
    out[5] = hp[2] * hq[2];
}

ZhangResult solve_zhang_calibration(const Homography* homs, int n)
{
    ZhangResult result;

    // ---- stack 2 constraint rows per VALID homography into A^T A directly
    // (never materializing the full 2n x 6 A -- for n<=8 this is a trivial
    // amount of arithmetic either way, but accumulating straight into the
    // 6x6 normal matrix is the natural, memory-light way to do it). --------
    double AtA[6][6] = {};
    int used = 0;
    for (int k = 0; k < n; ++k) {
        if (!homs[k].valid) continue;
        double v01[6], vdiff[6], v00[6], v11[6];
        v_pq(homs[k], 0, 1, v01);
        v_pq(homs[k], 0, 0, v00);
        v_pq(homs[k], 1, 1, v11);
        for (int i = 0; i < 6; ++i) vdiff[i] = v00[i] - v11[i];
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 6; ++j)
                AtA[i][j] += v01[i] * v01[j] + vdiff[i] * vdiff[j];
        ++used;
    }
    if (used < 3) { result.valid = false; return result; }   // Zhang's method needs >= 3 views in general

    double eigvecs[6][6];
    jacobi_eigen_symmetric6(AtA, eigvecs);   // AtA is now (numerically) diagonal; its diagonal = eigenvalues

    int min_idx = 0;
    for (int i = 1; i < 6; ++i) if (AtA[i][i] < AtA[min_idx][min_idx]) min_idx = i;

    double b[6];
    for (int i = 0; i < 6; ++i) b[i] = eigvecs[i][min_idx];
    if (b[0] < 0.0) for (int i = 0; i < 6; ++i) b[i] = -b[i];   // B11 must be positive (see THEORY.md)

    const double B11 = b[0], B12 = b[1], B22 = b[2], B13 = b[3], B23 = b[4], B33 = b[5];
    const double denom = B11 * B22 - B12 * B12;
    if (std::fabs(denom) < 1e-18 || B11 <= 0.0) { result.valid = false; return result; }

    const double v0 = (B12 * B13 - B11 * B23) / denom;
    const double lambda = B33 - (B13 * B13 + v0 * (B12 * B13 - B11 * B23)) / B11;
    if (lambda <= 0.0) { result.valid = false; return result; }

    const double alpha2 = lambda / B11;
    const double beta2 = lambda * B11 / denom;
    if (alpha2 <= 0.0 || beta2 <= 0.0) { result.valid = false; return result; }

    result.fx = std::sqrt(alpha2);
    result.fy = std::sqrt(beta2);
    result.skew = -B12 * alpha2 * result.fy / lambda;
    result.cy = v0;
    result.cx = result.skew * v0 / result.fy - B13 * alpha2 / lambda;
    result.valid = true;
    return result;
}
