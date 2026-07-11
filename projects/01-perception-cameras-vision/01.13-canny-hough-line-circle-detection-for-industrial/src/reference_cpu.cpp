// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 01.13
//                     (Canny + Hough line/circle detection for industrial
//                     alignment)
//
// WHY does a GPU repository ship a CPU implementation of everything? See
// docs/PROJECT_TEMPLATE/src/reference_cpu.cpp's header for the two load-
// bearing reasons (correctness oracle + teaching baseline) and the
// independence ruling this file follows precisely:
//
//   * Data-layout contracts (kernels.cuh's constants, structs, the Q16 theta
//     table) are single-sourced and SHARED.
//   * The ALGORITHMIC CORE of every twinned stage is written TWICE,
//     independently, in the simplest correct C++. In particular:
//       - hysteresis here uses an explicit QUEUE-BASED FLOOD FILL, not the
//         GPU's synchronous repeated-sweep scan — a genuinely different
//         algorithm reaching the SAME fixed point (see the convergence
//         argument in kernels.cu's hysteresis kernel comment).
//       - the Hough accumulation loops here are plain nested for-loops with
//         ordinary "accum[cell]++" — no atomics needed on one CPU thread,
//         which is itself the point: the GPU's atomicAdd exists ONLY to make
//         concurrent writes safe, and integer addition's exact associativity
//         is what guarantees this sequential loop and the GPU's concurrent
//         one land on the IDENTICAL final counts (kernels.cu's file header).
//
// What IS and IS NOT twinned in this project (see THEORY.md "How we verify
// correctness" for the full table):
//   TWINNED, float tolerance:  Gaussian blur, Sobel gradients, NMS.
//   TWINNED, EXACT (integer):  hysteresis edge state, Hough LINE accumulator
//                               (thanks to the shared Q16 table — see
//                               kernels.cuh SECTION 5).
//   TWINNED, tolerance (peak): Hough CIRCLE accumulator — inherits float
//                               tolerance from the Sobel gradients its vote
//                               DIRECTION depends on; verified at the peak
//                               level, not element-wise (documented, honest
//                               scoping — see kernels.cu's circle kernel
//                               comment and main.cu's verify stage).
//   NOT twinned at all (single-sourced, downstream analysis, exactly the
//   pattern flagship 08.01 uses for its host-only softmin blend): peak
//   extraction/sub-bin refinement and the rigid-alignment least-squares
//   solve, both in main.cu — their correctness is checked by the
//   INDEPENDENT gates (line_recovery, circle_recovery, alignment,
//   edge_quality, hysteresis_lesson, negative_control), per the template's
//   ruling that shared downstream code needs an independent gate in place
//   of a twin comparison.
//
// Rules for this file: plain C++17, no CUDA headers, no hand-vectorization,
// no OpenMP, no cleverness — clarity beats speed here, always.
//
// Read this after: kernels.cu — compare stage by stage.
// ===========================================================================

#include "kernels.cuh"

#include <cmath>
#include <cstring>
#include <queue>
#include <vector>

// ---------------------------------------------------------------------------
// clampi_cpu — identical clamp-to-edge boundary rule as kernels.cu's
// device-side clampi(). Re-typed independently (it is one line; sharing it
// would buy nothing and the independence ruling default is "write it
// twice") but must implement the SAME rule, or the float-tolerance
// comparison would fail at every border pixel.
// ---------------------------------------------------------------------------
static inline int clampi_cpu(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

// ===========================================================================
// STAGE 1 — separable Gaussian blur (CPU twin of gaussian_blur_h/v_kernel).
// One sequential double loop, horizontal pass into a temporary buffer, then
// vertical pass into the output — the exact separable structure kernels.cu
// documents, just executed by one core instead of 76,800 threads.
// ---------------------------------------------------------------------------
void gaussian_blur_cpu(const uint8_t* img, int W, int H, float* blurred)
{
    std::vector<float> tmp(static_cast<size_t>(W) * H);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float acc = 0.0f;
            for (int k = -GAUSS_RADIUS; k <= GAUSS_RADIUS; ++k) {
                const int xs = clampi_cpu(x + k, 0, W - 1);
                acc += GAUSS_WEIGHTS[k + GAUSS_RADIUS] * static_cast<float>(img[y * W + xs]);
            }
            tmp[static_cast<size_t>(y) * W + x] = acc;
        }
    }
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float acc = 0.0f;
            for (int k = -GAUSS_RADIUS; k <= GAUSS_RADIUS; ++k) {
                const int ys = clampi_cpu(y + k, 0, H - 1);
                acc += GAUSS_WEIGHTS[k + GAUSS_RADIUS] * tmp[static_cast<size_t>(ys) * W + x];
            }
            blurred[static_cast<size_t>(y) * W + x] = acc;
        }
    }
}

// ===========================================================================
// STAGE 2 — Sobel gradients (CPU twin of sobel_gradient_kernel). Same 3x3
// stencil, same SOBEL_SCALE normalization (kernels.cuh SECTION 4) — get the
// scale right here too, independently, or the two paths would silently
// disagree by a factor of 4 despite "looking" like they compute the same
// direction and NMS/thresholds would need different tuning per path.
// ---------------------------------------------------------------------------
void sobel_gradient_cpu(const float* blurred, int W, int H, float* gx, float* gy)
{
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float p[3][3];
            for (int dy = -1; dy <= 1; ++dy) {
                const int ys = clampi_cpu(y + dy, 0, H - 1);
                for (int dx = -1; dx <= 1; ++dx) {
                    const int xs = clampi_cpu(x + dx, 0, W - 1);
                    p[dy + 1][dx + 1] = blurred[static_cast<size_t>(ys) * W + xs];
                }
            }
            const float raw_gx = (p[0][2] + 2.0f * p[1][2] + p[2][2]) - (p[0][0] + 2.0f * p[1][0] + p[2][0]);
            const float raw_gy = (p[2][0] + 2.0f * p[2][1] + p[2][2]) - (p[0][0] + 2.0f * p[0][1] + p[0][2]);
            const size_t i = static_cast<size_t>(y) * W + x;
            gx[i] = raw_gx * SOBEL_SCALE;
            gy[i] = raw_gy * SOBEL_SCALE;
        }
    }
}

// ===========================================================================
// STAGE 3 — non-max suppression (CPU twin of nms_kernel). Same 4-sector
// quantization rule, same border-is-not-an-edge policy, so the float-
// tolerance comparison in main.cu is comparing like with like.
// ---------------------------------------------------------------------------
static inline void nms_offsets_cpu(int dir, int& ox, int& oy)
{
    switch (dir) {
        case 0: ox = 1; oy = 0;  break;
        case 1: ox = 1; oy = 1;  break;
        case 2: ox = 0; oy = 1;  break;
        default: ox = 1; oy = -1; break;
    }
}

void nms_cpu(const float* gx, const float* gy, int W, int H, float* suppressed_mag)
{
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const size_t i = static_cast<size_t>(y) * W + x;
            if (x == 0 || x == W - 1 || y == 0 || y == H - 1) { suppressed_mag[i] = 0.0f; continue; }

            const float dx = gx[i], dy = gy[i];
            const float mag = std::sqrt(dx * dx + dy * dy);

            float angle_deg = std::atan2(dy, dx) * (180.0f / PI_F);
            if (angle_deg < 0.0f) angle_deg += 360.0f;
            const int octant = static_cast<int>((angle_deg + 22.5f) * (1.0f / 45.0f)) & 7;
            int ox, oy;
            nms_offsets_cpu(octant & 3, ox, oy);

            const int xa = x + ox, ya = y + oy;
            const int xb = x - ox, yb = y - oy;
            const size_t ia = static_cast<size_t>(ya) * W + xa;
            const size_t ib = static_cast<size_t>(yb) * W + xb;
            const float mag_a = std::sqrt(gx[ia] * gx[ia] + gy[ia] * gy[ia]);
            const float mag_b = std::sqrt(gx[ib] * gx[ib] + gy[ib] * gy[ib]);

            suppressed_mag[i] = (mag >= mag_a && mag >= mag_b) ? mag : 0.0f;
        }
    }
}

// ===========================================================================
// STAGE 4 — double-threshold classification (CPU twin of
// classify_threshold_kernel). Trivial per-pixel map.
// ---------------------------------------------------------------------------
void classify_threshold_cpu(const float* suppressed_mag, int W, int H,
                            float t_low, float t_high, unsigned char* state)
{
    const int N = W * H;
    for (int i = 0; i < N; ++i) {
        const float m = suppressed_mag[i];
        state[i] = (m >= t_high) ? EDGE_STRONG : (m >= t_low ? EDGE_WEAK : EDGE_NONE);
    }
}

// ===========================================================================
// STAGE 5 — hysteresis promotion (CPU twin of the GPU's repeated-sweep
// kernel) — DELIBERATELY A DIFFERENT ALGORITHM: a QUEUE-BASED FLOOD FILL.
//
// Seed the queue with every already-STRONG pixel. Pop a pixel, examine its
// 8 neighbors; any WEAK neighbor is promoted to STRONG and pushed onto the
// queue. This visits each pixel's promotion exactly once (O(W*H) total,
// versus the GPU's O(sweeps * W*H) re-scan of the whole image every sweep —
// the sequential algorithm gets to be smarter because it does not need
// every thread to make independent progress every cycle).
//
// Why this is the RIGHT independent twin, not a cheat: kernels.cu's header
// comment proves the promotion rule's fixed point is UNIQUE regardless of
// visit order (state only increases, bounded above, so it converges to
// "every weak pixel 8-connected — by any path — to a strong pixel becomes
// strong", full stop). This flood fill and the GPU's synchronous sweeps are
// two different paths to providably the SAME fixed point — exactly the kind
// of structurally-independent verification the repo's twin-independence
// ruling is asking for (reference_cpu.cpp's own file header, and the 13.03
// cautionary tale it cites: shared code can hide a shared bug; a genuinely
// different algorithm cannot share that bug).
// ---------------------------------------------------------------------------
void hysteresis_propagate_cpu(unsigned char* state, int W, int H)
{
    std::queue<int> q;
    const int N = W * H;
    for (int i = 0; i < N; ++i)
        if (state[i] == EDGE_STRONG) q.push(i);

    while (!q.empty()) {
        const int i = q.front(); q.pop();
        const int x = i % W, y = i / W;
        for (int dy = -1; dy <= 1; ++dy) {
            const int ny = y + dy;
            if (ny < 0 || ny >= H) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                const int nx = x + dx;
                if (nx < 0 || nx >= W) continue;
                const int ni = ny * W + nx;
                if (state[ni] == EDGE_WEAK) {
                    state[ni] = EDGE_STRONG;   // promote once
                    q.push(ni);                // and let IT propagate further
                }
            }
        }
    }
}

// ===========================================================================
// STAGE 6 — finalize edge map (CPU twin of finalize_edge_map_kernel).
// ---------------------------------------------------------------------------
void finalize_edge_map_cpu(const unsigned char* state, int W, int H, unsigned char* edge_map)
{
    const int N = W * H;
    for (int i = 0; i < N; ++i)
        edge_map[i] = (state[i] == EDGE_STRONG) ? 255u : 0u;
}

// ===========================================================================
// STAGE 7 — Hough LINE accumulation (CPU twin of hough_lines_vote_kernel).
//
// Plain triple-nested loop: for every edge pixel, for every theta bin,
// compute the SAME fixed-point rho as the GPU (using the IDENTICAL shared
// cos_fixed/sin_fixed table — see kernels.cuh SECTION 5 and this file's
// header) and increment accum[cell] with an ordinary "++", no atomic
// needed on a single sequential core. Integer addition's exact
// associativity is what makes this loop's final counts identical to the
// GPU's concurrent atomicAdd version, cell for cell, regardless of the
// millions of possible thread interleavings on the GPU side.
// ---------------------------------------------------------------------------
void hough_lines_accum_cpu(const unsigned char* edge_map, int W, int H,
                           const int32_t* cos_fixed, const int32_t* sin_fixed, int* accum)
{
    std::memset(accum, 0, sizeof(int) * static_cast<size_t>(HOUGH_LINE_ACCUM_CELLS));

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (edge_map[static_cast<size_t>(y) * W + x] == 0) continue;
            for (int t = 0; t < HOUGH_THETA_BINS; ++t) {
                const int32_t rho_fixed = x * cos_fixed[t] + y * sin_fixed[t];
                const int32_t rho = (rho_fixed >= 0)
                    ? (rho_fixed + (HOUGH_FIXED_SCALE / 2)) >> HOUGH_FIXED_SHIFT
                    : -(((-rho_fixed) + (HOUGH_FIXED_SCALE / 2)) >> HOUGH_FIXED_SHIFT);
                const int rho_bin = static_cast<int>(rho) + HOUGH_RHO_MAX;
                if (rho_bin < 0 || rho_bin >= HOUGH_RHO_BINS) continue;
                accum[static_cast<size_t>(t) * HOUGH_RHO_BINS + rho_bin]++;
            }
        }
    }
}

// ===========================================================================
// STAGE 8 — Hough CIRCLE accumulation (CPU twin of
// hough_circles_vote_kernel). Same known-radius, gradient-directed voting
// rule; see kernels.cu's kernel comment for why this stage is verified at
// the peak level rather than bit-exact (it depends on the float-tolerance
// Sobel gradients, not the fixed-point theta table).
// ---------------------------------------------------------------------------
void hough_circles_accum_cpu(const unsigned char* edge_map, const float* gx, const float* gy,
                             int W, int H, int* accum)
{
    std::memset(accum, 0, sizeof(int) * static_cast<size_t>(HOUGH_CIRCLE_ACCUM_CELLS));

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const size_t i = static_cast<size_t>(y) * W + x;
            if (edge_map[i] == 0) continue;

            const float dx = gx[i], dy = gy[i];
            const float mag = std::sqrt(dx * dx + dy * dy);
            if (mag < 1e-3f) continue;
            const float nx = dx / mag, ny = dy / mag;

            for (int k = 0; k < NUM_HOLES; ++k) {
                const float r = HOLE_RADIUS[k];
                for (int sign = -1; sign <= 1; sign += 2) {
                    const float fcx = static_cast<float>(x) + static_cast<float>(sign) * r * nx;
                    const float fcy = static_cast<float>(y) + static_cast<float>(sign) * r * ny;
                    const int cx = static_cast<int>(std::lround(fcx));
                    const int cy = static_cast<int>(std::lround(fcy));
                    if (cx < 0 || cx >= W || cy < 0 || cy >= H) continue;
                    accum[(static_cast<size_t>(k) * H + cy) * W + cx]++;
                }
            }
        }
    }
}
