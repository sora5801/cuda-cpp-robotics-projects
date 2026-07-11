// ===========================================================================
// kernels.cu — GPU kernels for project 01.13
//              (Canny + Hough line/circle detection for industrial alignment)
//
// Role in the project
// -------------------
// The full GPU pipeline lives here, stage by stage, in the exact order
// main.cu calls them: Gaussian blur (2 passes) -> Sobel gradients -> NMS ->
// double-threshold classify -> hysteresis promotion (repeated sweeps) ->
// finalize edge map -> Hough line voting -> Hough circle voting. Every
// kernel is a MAP or a bounded STENCIL over the image except the two Hough
// voting kernels, which are the project's SCATTER-with-atomics case study.
//
// One thread per PIXEL is the mapping for every stage through the edge map
// (IMG_W*IMG_H = 76,800 threads — tiny by GPU standards, launched as flat
// 1-D grids of 256-thread blocks, exactly the block/grid idiom used
// throughout this repo, e.g. 01.06's CCL kernels). The two Hough kernels
// keep the SAME one-thread-per-pixel mapping but each thread that owns an
// EDGE pixel fans out into many accumulator writes — see each kernel's own
// comment for the fan-out count and memory-access story.
//
// Read this after: kernels.cuh (the contracts). Read this before:
// reference_cpu.cpp (the independent CPU twins) and main.cu (orchestration).
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"

#include <cstdio>

// ---------------------------------------------------------------------------
// flat_launch — the one launch-configuration idiom this whole file reuses:
// N independent per-pixel (or per-cell) threads, 256 per block (a warp
// multiple, the repo's standard default — see docs/PROJECT_TEMPLATE's SAXPY
// comment for the full occupancy reasoning), enough blocks to cover N with
// no grid-stride loop (N here never exceeds IMG_W*IMG_H = 76,800, far under
// any grid-size limit, so the extra grid-stride complexity would teach
// nothing new over 01.06/08.01's flat kernels).
// ---------------------------------------------------------------------------
static inline dim3 flat_grid(int n, int block) { return dim3((n + block - 1) / block); }
static const int kBlock = 256;

// ===========================================================================
// __constant__ MEMORY — the Hough fixed-point theta table and known hole
// radii. Constant memory is the right home for these: EVERY thread in the
// hough_lines_vote_kernel launch reads the SAME 180 entries (broadcast
// access pattern), and constant memory is backed by a small per-SM cache
// tuned exactly for "every thread reads the same address on the same
// cycle" — a single broadcast fetch instead of 32 separate global loads per
// warp. At 180*4*2 = 1440 bytes total it is a tiny fraction of the 64 KiB
// constant window. See THEORY.md "The GPU mapping" for the alternative
// (global memory + L1/L2 caching) and why constant memory wins here.
// ---------------------------------------------------------------------------
__constant__ int32_t g_cos_fixed[HOUGH_THETA_BINS];
__constant__ int32_t g_sin_fixed[HOUGH_THETA_BINS];
__constant__ float   g_hole_radius[NUM_HOLES];

void upload_hough_constants(const int32_t* cos_fixed, const int32_t* sin_fixed)
{
    CUDA_CHECK(cudaMemcpyToSymbol(g_cos_fixed, cos_fixed, sizeof(int32_t) * HOUGH_THETA_BINS));
    CUDA_CHECK(cudaMemcpyToSymbol(g_sin_fixed, sin_fixed, sizeof(int32_t) * HOUGH_THETA_BINS));
    // HOLE_RADIUS is the compile-time-known nominal radius set from
    // kernels.cuh SECTION 2 — uploaded here (not passed per-launch) because
    // it is fixed for the whole program run, exactly like the theta table.
    CUDA_CHECK(cudaMemcpyToSymbol(g_hole_radius, HOLE_RADIUS, sizeof(float) * NUM_HOLES));
}

// ===========================================================================
// STAGE 1 — separable Gaussian blur (horizontal pass, then vertical pass).
//
// Why separable: a 2-D 5x5 Gaussian convolution costs 25 multiply-adds per
// pixel; splitting it into a 1-D horizontal 5-tap pass followed by a 1-D
// vertical 5-tap pass costs 5+5 = 10 — a 2.5x arithmetic reduction that is
// EXACT (not an approximation) because the Gaussian kernel is a true outer
// product of two 1-D Gaussians. This is the identical separability lesson
// project 01.05's Gaussian pyramid relies on (cite:
// projects/01-perception-cameras-vision/01.05-sift-surf-on-gpu) — re-derived
// here for THIS project's 5-tap binomial stencil (kernels.cuh SECTION 4).
//
// Boundary handling: CLAMP-TO-EDGE (repeat the nearest valid pixel) via
// clampi() below — simple, deterministic, and identical on GPU and CPU
// (reference_cpu.cpp uses the exact same clamp), which is what the
// GPU-vs-CPU float-tolerance comparison in main.cu depends on.
// ---------------------------------------------------------------------------
__device__ __forceinline__ int clampi(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

__global__ void gaussian_blur_h_kernel(const uint8_t* __restrict__ img, int W, int H,
                                       float* __restrict__ tmp)
{
    // Thread (bx,tx) owns ONE output pixel i = blockIdx.x*blockDim.x+threadIdx.x,
    // decoded to (x,y) via the row-major layout in kernels.cuh. Each thread
    // reads 5 INPUT pixels (a horizontal window) and writes 1 OUTPUT float —
    // a classic 1-D stencil map, no data sharing between threads (so no
    // shared-memory tiling here; see THEORY.md for why tiling would help at
    // larger radii but is not worth the complexity at radius 2).
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= W * H) return;
    const int x = i % W, y = i / W;

    float acc = 0.0f;
    for (int k = -GAUSS_RADIUS; k <= GAUSS_RADIUS; ++k) {
        const int xs = clampi(x + k, 0, W - 1);                 // clamp-to-edge boundary
        acc += GAUSS_WEIGHTS[k + GAUSS_RADIUS] * static_cast<float>(img[y * W + xs]);
    }
    tmp[i] = acc;   // intermediate result kept in FLOAT (not re-quantized to uint8) —
                    // see kernels.cuh SECTION 4's numerics note.
}

__global__ void gaussian_blur_v_kernel(const float* __restrict__ tmp, int W, int H,
                                       float* __restrict__ blurred)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= W * H) return;
    const int x = i % W, y = i / W;

    float acc = 0.0f;
    for (int k = -GAUSS_RADIUS; k <= GAUSS_RADIUS; ++k) {
        const int ys = clampi(y + k, 0, H - 1);
        acc += GAUSS_WEIGHTS[k + GAUSS_RADIUS] * tmp[ys * W + x];
    }
    blurred[i] = acc;
}

void launch_gaussian_blur(const uint8_t* d_img, int W, int H, float* d_tmp, float* d_blurred)
{
    const int N = W * H;
    gaussian_blur_h_kernel<<<flat_grid(N, kBlock), kBlock>>>(d_img, W, H, d_tmp);
    CUDA_CHECK_LAST_ERROR("gaussian_blur_h_kernel launch");
    gaussian_blur_v_kernel<<<flat_grid(N, kBlock), kBlock>>>(d_tmp, W, H, d_blurred);
    CUDA_CHECK_LAST_ERROR("gaussian_blur_v_kernel launch");
}

// ===========================================================================
// STAGE 2 — Sobel gradients (a 3x3 STENCIL map, one thread per pixel).
//
// Gx = [-1 0 1; -2 0 2; -1 0 1], Gy = Gx^T. Each thread reads a full 3x3
// neighborhood (9 loads, all clamp-to-edge) and writes gx[i], gy[i].
//
// THE SCALING LESSON (cite project 01.03's x32 Scharr case study, this
// project's own version — see kernels.cuh SECTION 4): the RAW convolution
// sum is 4x too large relative to true per-pixel intensity gradient units,
// because Gx/Gy's positive-side weights sum to 1+2+1=4, not 1. We correct
// it HERE, at the source, by multiplying by SOBEL_SCALE = 1/4 — every
// downstream consumer (NMS's magnitude, the T_LOW/T_HIGH thresholds, the
// circle kernel's unit-direction normalization) then works in genuine
// "intensity levels per pixel of edge steepness" units, not "4x that and a
// silent trap for whoever reuses this code". Get the scaling right ONCE.
// ---------------------------------------------------------------------------
__global__ void sobel_gradient_kernel(const float* __restrict__ blurred, int W, int H,
                                      float* __restrict__ gx, float* __restrict__ gy)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= W * H) return;
    const int x = i % W, y = i / W;

    // Load the 3x3 neighborhood once into registers (9 values), then reuse
    // for both Gx and Gy — halves the memory traffic versus computing Gx and
    // Gy in two separate kernels that would each re-read the same 9 pixels.
    float p[3][3];
    for (int dy = -1; dy <= 1; ++dy) {
        const int ys = clampi(y + dy, 0, H - 1);
        for (int dx = -1; dx <= 1; ++dx) {
            const int xs = clampi(x + dx, 0, W - 1);
            p[dy + 1][dx + 1] = blurred[ys * W + xs];
        }
    }

    const float raw_gx = (p[0][2] + 2.0f * p[1][2] + p[2][2]) - (p[0][0] + 2.0f * p[1][0] + p[2][0]);
    const float raw_gy = (p[2][0] + 2.0f * p[2][1] + p[2][2]) - (p[0][0] + 2.0f * p[0][1] + p[0][2]);

    gx[i] = raw_gx * SOBEL_SCALE;   // scaled to true per-pixel gradient units — see header note
    gy[i] = raw_gy * SOBEL_SCALE;
}

void launch_sobel_gradient(const float* d_blurred, int W, int H, float* d_gx, float* d_gy)
{
    const int N = W * H;
    sobel_gradient_kernel<<<flat_grid(N, kBlock), kBlock>>>(d_blurred, W, H, d_gx, d_gy);
    CUDA_CHECK_LAST_ERROR("sobel_gradient_kernel launch");
}

// ===========================================================================
// STAGE 3 — gradient-direction non-max suppression (NMS).
//
// Idea: a "real" edge pixel is a LOCAL MAXIMUM of gradient magnitude ALONG
// the gradient direction (i.e., walking perpendicular to the edge, straight
// across it). NMS keeps a pixel only if its magnitude is >= both neighbors
// one step forward and one step backward along that direction; otherwise it
// is part of the edge's "shoulder" (a multi-pixel-wide blur ridge), not its
// crest, and is suppressed to 0.
//
// THE HONEST SIMPLIFICATION this project makes (state it, don't hide it):
// textbook Canny NMS interpolates the two off-grid neighbor magnitudes
// bilinearly for sub-pixel accuracy at arbitrary gradient angles. This
// kernel instead uses the cheaper, classic 4-SECTOR QUANTIZATION: the
// gradient direction is rounded to the nearest of 4 axis/diagonal
// directions (0/45/90/135 degrees) and compared against the two INTEGER
// neighbor pixels that direction implies — no interpolation. The cost: a
// systematic angular error of up to 22.5 degrees can occasionally keep a
// shoulder pixel that true interpolated NMS would suppress (or vice versa),
// thickening some edges by roughly one extra pixel at unlucky angles. The
// benefit: no extra loads/lerps and a much simpler kernel a learner can
// read start to finish. THEORY.md "Numerical considerations" quantifies
// this trade-off against this project's own edge_quality gate.
//
// Direction-to-neighbor-pair table (see THEORY.md "The math" for the
// derivation of why atan2(gy,gx) quantized this way picks these exact
// pairs): index by octant = floor((angle_deg + 22.5) / 45) mod 8, then
// dir = octant mod 4 folds opposite octants (0<->4, 1<->5, ...) onto the
// same neighbor PAIR, because NMS always compares symmetric +/- offsets.
//   dir 0 (near 0/180 deg,  horizontal gradient) -> compare (x-1,y),(x+1,y)
//   dir 1 (near 45/225 deg, "\" gradient)         -> compare (x-1,y-1),(x+1,y+1)
//   dir 2 (near 90/270 deg, vertical gradient)    -> compare (x,y-1),(x,y+1)
//   dir 3 (near 135/315 deg,"/" gradient)         -> compare (x+1,y-1),(x-1,y+1)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void nms_offsets(int dir, int& ox, int& oy)
{
    // Compile-time-constant small switch — nvcc turns this into a handful of
    // predicated moves, not a branchy jump table, at these sizes.
    switch (dir) {
        case 0: ox = 1; oy = 0;  break;
        case 1: ox = 1; oy = 1;  break;
        case 2: ox = 0; oy = 1;  break;
        default: ox = 1; oy = -1; break; // dir 3
    }
}

__global__ void nms_kernel(const float* __restrict__ gx, const float* __restrict__ gy,
                           int W, int H, float* __restrict__ suppressed_mag)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= W * H) return;
    const int x = i % W, y = i / W;

    // Border pixels are treated as non-edges outright (see kernels.cuh's
    // note: none of this scene's features sit within 1 px of the image
    // border, so this costs nothing and avoids a second layer of clamping).
    if (x == 0 || x == W - 1 || y == 0 || y == H - 1) { suppressed_mag[i] = 0.0f; return; }

    const float dx = gx[i], dy = gy[i];
    const float mag = sqrtf(dx * dx + dy * dy);

    // atan2f returns radians in (-pi, pi]; convert to degrees in [0,360)
    // for the octant table above (fmodf handles the rare exact-(-pi) case).
    float angle_deg = atan2f(dy, dx) * (180.0f / PI_F);
    if (angle_deg < 0.0f) angle_deg += 360.0f;
    const int octant = static_cast<int>((angle_deg + 22.5f) * (1.0f / 45.0f)) & 7;
    int ox, oy;
    nms_offsets(octant & 3, ox, oy);

    const int xa = x + ox, ya = y + oy;   // "forward" neighbor along the gradient
    const int xb = x - ox, yb = y - oy;   // "backward" neighbor
    const float ga_x = gx[ya * W + xa], ga_y = gy[ya * W + xa];
    const float gb_x = gx[yb * W + xb], gb_y = gy[yb * W + xb];
    const float mag_a = sqrtf(ga_x * ga_x + ga_y * ga_y);
    const float mag_b = sqrtf(gb_x * gb_x + gb_y * gb_y);

    // Keep only if this pixel is a local maximum along the gradient
    // direction; >= (not >) so a perfectly flat ridge keeps its center
    // pixel deterministically rather than suppressing every candidate.
    suppressed_mag[i] = (mag >= mag_a && mag >= mag_b) ? mag : 0.0f;
}

void launch_nms(const float* d_gx, const float* d_gy, int W, int H, float* d_suppressed_mag)
{
    const int N = W * H;
    nms_kernel<<<flat_grid(N, kBlock), kBlock>>>(d_gx, d_gy, W, H, d_suppressed_mag);
    CUDA_CHECK_LAST_ERROR("nms_kernel launch");
}

// ===========================================================================
// STAGE 4 — double-threshold classification (ALSO used for the single-
// threshold comparison — see main.cu, which calls this with t_low==t_high).
//
// Pure per-pixel map: no neighbor reads at all, the simplest kernel in this
// file. Kept as its own kernel (rather than fused into NMS) because main.cu
// calls it TWICE per demo run with different threshold pairs — the
// hysteresis-lesson comparison this project is built to teach.
// ---------------------------------------------------------------------------
__global__ void classify_threshold_kernel(const float* __restrict__ suppressed_mag,
                                          int W, int H, float t_low, float t_high,
                                          unsigned char* __restrict__ state)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= W * H) return;
    const float m = suppressed_mag[i];
    state[i] = (m >= t_high) ? EDGE_STRONG : (m >= t_low ? EDGE_WEAK : EDGE_NONE);
}

void launch_classify_threshold(const float* d_suppressed_mag, int W, int H,
                               float t_low, float t_high, unsigned char* d_state)
{
    const int N = W * H;
    classify_threshold_kernel<<<flat_grid(N, kBlock), kBlock>>>(d_suppressed_mag, W, H, t_low, t_high, d_state);
    CUDA_CHECK_LAST_ERROR("classify_threshold_kernel launch");
}

// ===========================================================================
// STAGE 5 — hysteresis promotion, ONE SWEEP (CCL-style iterative
// propagation; cite: 01.06-apriltag-aruco's ccl_propagate_sweep_kernel and
// 30.01's stage 4, same algorithmic shape re-typed independently here for
// this project's edge_state layout — CLAUDE.md's cross-project duplication
// norm, not the within-project twin-independence ruling).
//
// Rule: a WEAK pixel (state 1) is promoted to STRONG (state 2) if ANY of
// its 8 neighbors is already STRONG. Every promoted pixel can, on the NEXT
// sweep, promote its own weak neighbors — exactly a flood fill, run in
// lockstep across the whole image instead of with an explicit queue.
//
// CONVERGENCE ARGUMENT (mirrors 01.06's atomicMin monotonicity argument,
// restated for atomicOr promotion): state[i] only ever INCREASES
// (EDGE_NONE < EDGE_WEAK < EDGE_STRONG, and this kernel only ever writes
// EDGE_STRONG, never demotes), and is bounded above by EDGE_STRONG. A
// monotonically non-decreasing, bounded sequence of per-pixel updates
// converges in finitely many sweeps to a UNIQUE fixed point: "every weak
// pixel 8-connected, by any path of weak pixels, to a strong pixel is now
// strong." That fixed point does not depend on sweep order or which thread
// ran first — which is exactly why reference_cpu.cpp can reach the SAME
// fixed point with a completely different algorithm (a queue-based flood
// fill) and main.cu can compare the two for EXACT equality.
// ---------------------------------------------------------------------------
__global__ void hysteresis_propagate_sweep_kernel(unsigned char* __restrict__ state,
                                                   int W, int H, int* __restrict__ changed)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= W * H) return;
    if (state[i] != EDGE_WEAK) return;   // only weak pixels are candidates for promotion

    const int x = i % W, y = i / W;
    bool has_strong_neighbor = false;
    // Unrolled 8-neighbor check with bounds guards (image border pixels have
    // fewer than 8 neighbors) — the same defensive pattern 01.06 uses for
    // its 4-neighbor CCL sweep, extended to 8-connectivity here because a
    // diagonal-only connection between the plate's strong boundary and the
    // scratch mark's weak pixels is a realistic geometry this scene creates.
    for (int dy = -1; dy <= 1 && !has_strong_neighbor; ++dy) {
        const int ny = y + dy;
        if (ny < 0 || ny >= H) continue;
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            const int nx = x + dx;
            if (nx < 0 || nx >= W) continue;
            if (state[ny * W + nx] == EDGE_STRONG) { has_strong_neighbor = true; break; }
        }
    }

    if (has_strong_neighbor) {
        state[i] = EDGE_STRONG;         // monotonic promotion — never demoted afterward
        atomicOr(changed, 1);           // tell the host loop "do another sweep"
    }
}

bool launch_hysteresis_sweep(unsigned char* d_state, int W, int H)
{
    // A single device int, reset to 0 before the sweep, atomicOr'd to 1 by
    // any thread that promotes a pixel. One tiny D2H copy per sweep (4
    // bytes) — negligible next to the sweep's own kernel launch overhead.
    static int* d_changed = nullptr;
    if (!d_changed) CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_changed, 0, sizeof(int)));

    const int N = W * H;
    hysteresis_propagate_sweep_kernel<<<flat_grid(N, kBlock), kBlock>>>(d_state, W, H, d_changed);
    CUDA_CHECK_LAST_ERROR("hysteresis_propagate_sweep_kernel launch");

    int changed = 0;
    CUDA_CHECK(cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
    return changed != 0;
}

// ===========================================================================
// STAGE 6 — state -> binary edge map (0 / 255), a trivial map kept as its
// own kernel so the Hough stages take one clean unsigned-char input rather
// than reaching back into the 3-valued state buffer.
// ---------------------------------------------------------------------------
__global__ void finalize_edge_map_kernel(const unsigned char* __restrict__ state,
                                         int W, int H, unsigned char* __restrict__ edge_map)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= W * H) return;
    edge_map[i] = (state[i] == EDGE_STRONG) ? 255u : 0u;
}

void launch_finalize_edge_map(const unsigned char* d_state, int W, int H, unsigned char* d_edge_map)
{
    const int N = W * H;
    finalize_edge_map_kernel<<<flat_grid(N, kBlock), kBlock>>>(d_state, W, H, d_edge_map);
    CUDA_CHECK_LAST_ERROR("finalize_edge_map_kernel launch");
}

// ===========================================================================
// STAGE 7 — Hough LINE voting: the project's SCATTER-with-atomics case
// study.
//
// One thread per PIXEL (same mapping as every stage above), but only edge
// pixels (edge_map[i] != 0) do any work — a genuinely DATA-DEPENDENT
// workload per thread, unlike every prior stage's uniform per-pixel cost.
// Each active thread then FANS OUT into HOUGH_THETA_BINS (180) accumulator
// writes: for every candidate line direction theta, compute the rho this
// pixel would contribute to a line at that angle, and atomicAdd 1 into
// accum[theta_bin][rho_bin].
//
// Why SCATTER, not GATHER: the natural "gather" mapping here would be one
// thread PER ACCUMULATOR CELL, each looping over every edge pixel asking
// "do I get a vote from you?" — but that is O(144,180 cells * edge_pixels),
// while the scatter mapping used here is O(edge_pixels * 180 bins), and
// edge_pixels is typically a few hundred, not 144,180. Scatter wins whenever
// the "owning" side (edge pixels) is far smaller than the "target" side
// (accumulator cells) — the general trade THEORY.md's "The GPU mapping"
// names explicitly (see also the contrast with a dense reduction like
// 26.01's structural FEA assembly, cited there for the opposite case where
// gather wins because the target side is small).
//
// WHY INTEGER ATOMICS -> ORDER-INDEPENDENT -> BIT-EXACT TWIN: atomicAdd on
// int is a hardware read-modify-write with a GUARANTEED total order among
// colliding threads (the hardware serializes conflicting atomics on the
// same address) and, critically, INTEGER ADDITION IS ASSOCIATIVE AND
// COMMUTATIVE EXACTLY (no rounding, unlike float addition, whose order-
// dependent rounding is why some other projects in this repo document
// float atomics as NOT reproducible run-to-run). However many edge pixels
// vote for cell (t, r), and in whatever order the scheduler happens to run
// them, the FINAL COUNT is the same every single time — which is exactly
// why main.cu can compare this kernel's accumulator against
// reference_cpu.cpp's completely different (purely sequential, no atomics
// needed) accumulator loop for BIT-EXACT equality (kernels.cuh SECTION 5's
// fixed-point table removes the OTHER source of GPU/CPU divergence, the
// vote-address computation itself).
//
// Memory pattern: g_cos_fixed/g_sin_fixed are __constant__ (broadcast read,
// see the file-header note); accum lives in GLOBAL memory and IS the
// contended resource — up to 320*240 = 76,800 threads each issuing 180
// atomics into a 144,180-cell table means genuine collisions cluster around
// whichever (theta,rho) bins the scene's real lines pass through (that
// clustering IS the signal: peaks are exactly where contention was highest).
// ---------------------------------------------------------------------------
__global__ void hough_lines_vote_kernel(const unsigned char* __restrict__ edge_map,
                                        int W, int H, int* __restrict__ accum)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= W * H) return;
    if (edge_map[i] == 0) return;   // only edge pixels vote — the data-dependent fan-out

    const int x = i % W, y = i / W;

    // Every one of the 180 candidate line directions gets one vote from
    // this pixel. Fixed-point integer arithmetic throughout (see the file
    // header and kernels.cuh SECTION 5): rho_fixed is exact Q16, and the
    // final round-to-nearest-int is a pure integer bias-and-shift — no
    // floating point anywhere in this address computation.
    for (int t = 0; t < HOUGH_THETA_BINS; ++t) {
        const int32_t rho_fixed = x * g_cos_fixed[t] + y * g_sin_fixed[t];   // Q16, fits int32 (see kernels.cuh)
        // Round rho_fixed/65536 to nearest integer via bias-then-shift:
        // adding +/- half a unit (32768) before an ARITHMETIC right shift
        // is the standard branch-free integer round-to-nearest for a value
        // whose sign is unknown ahead of time.
        const int32_t rho = (rho_fixed >= 0) ? (rho_fixed + (HOUGH_FIXED_SCALE / 2)) >> HOUGH_FIXED_SHIFT
                                              : -(((-rho_fixed) + (HOUGH_FIXED_SCALE / 2)) >> HOUGH_FIXED_SHIFT);
        const int rho_bin = static_cast<int>(rho) + HOUGH_RHO_MAX;
        if (rho_bin < 0 || rho_bin >= HOUGH_RHO_BINS) continue;   // rho out of the table's range (rare, corner pixels)
        atomicAdd(&accum[t * HOUGH_RHO_BINS + rho_bin], 1);
    }
}

void launch_hough_lines_vote(const unsigned char* d_edge_map, int W, int H, int* d_accum)
{
    CUDA_CHECK(cudaMemset(d_accum, 0, sizeof(int) * static_cast<size_t>(HOUGH_LINE_ACCUM_CELLS)));
    const int N = W * H;
    hough_lines_vote_kernel<<<flat_grid(N, kBlock), kBlock>>>(d_edge_map, W, H, d_accum);
    CUDA_CHECK_LAST_ERROR("hough_lines_vote_kernel launch");
}

// ===========================================================================
// STAGE 8 — Hough CIRCLE voting: known-radius, gradient-directed, the
// "memory-lean trick" that tames the generically-3-D circle Hough
// transform.
//
// A GENERIC circle Hough transform accumulates over (cx, cy, r) — a 3-D
// volume of size W*H*R_range, and every edge pixel would have to vote at
// EVERY candidate radius along a full circle of candidate centers (an O(R)
// fan-out per pixel PER radius, i.e. O(R^2) work per pixel in the worst
// naive form). This project's industrial premise removes that entirely:
// the drilled holes have KNOWN, DISTINCT nominal radii (kernels.cuh
// HOLE_RADIUS) — machinists do not drill mystery holes. That collapses the
// accumulator to NUM_HOLES=3 independent 2-D (cx,cy) PLANES (kernels.cuh
// SECTION 6), and — the second trick — GRADIENT-DIRECTED voting means each
// edge pixel votes at only 2 CANDIDATE CENTERS PER RADIUS PLANE (one for
// each sign of "which side of the boundary am I on"), not an entire
// candidate circle of centers. Total fan-out per edge pixel: NUM_HOLES * 2
// = 6 atomics, versus a generic transform's O(360) per radius per pixel.
//
// WHY GRADIENT-DIRECTED VOTING WORKS: on a circle boundary, the intensity
// gradient at any edge pixel points RADIALLY — directly toward or away from
// the true center, depending on whether the boundary is lighter-inside or
// darker-inside. So "walk distance r along the gradient direction, in
// EITHER sign" lands within about 1 px of the true center for a genuine
// circle edge pixel of that radius, while for edge pixels belonging to
// something else (a straight plate edge, texture noise) the two candidate
// points land nowhere in particular and spread their votes thinly across
// the accumulator instead of piling onto one cell — exactly the same
// "true structure concentrates votes, noise does not" argument that makes
// the line Hough transform work.
//
// NOT bit-exact against the CPU twin (an honest, documented exception —
// see THEORY.md "Numerical considerations" and reference_cpu.cpp): the
// vote position here depends on gx[i]/gy[i], which are only FLOAT-tolerant
// between GPU and CPU (Stage 2's Sobel kernel, not a fixed-point table like
// the line kernel's theta). main.cu therefore verifies this accumulator via
// PEAK-LEVEL tolerance (do the two independently-built accumulators agree
// on where their loudest cell is, within a small pixel/vote margin), not
// element-wise equality — see main.cu's verify stage.
// ---------------------------------------------------------------------------
__global__ void hough_circles_vote_kernel(const unsigned char* __restrict__ edge_map,
                                          const float* __restrict__ gx,
                                          const float* __restrict__ gy,
                                          int W, int H, int* __restrict__ accum)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= W * H) return;
    if (edge_map[i] == 0) return;

    const int x = i % W, y = i / W;
    const float dx = gx[i], dy = gy[i];
    const float mag = sqrtf(dx * dx + dy * dy);
    // A flat (near-zero-gradient) "edge" pixel has no reliable direction to
    // walk along; this should not occur for genuine hysteresis-surviving
    // edges (which are, by construction, local gradient maxima), but the
    // guard keeps the kernel well-defined against any future threshold
    // change instead of dividing by ~0.
    if (mag < 1e-3f) return;
    const float nx = dx / mag, ny = dy / mag;   // unit gradient direction (points across the boundary)

    for (int k = 0; k < NUM_HOLES; ++k) {
        const float r = g_hole_radius[k];
        // Both signs: the true center could be either "ahead of" or
        // "behind" this pixel along the gradient, depending on whether the
        // hole is darker-inside (center behind, against the gradient) or
        // lighter-inside (center ahead) — this scene renders holes DARKER
        // than the plate, but voting both directions makes the kernel
        // correct regardless, at the cost of exactly 2x the votes (the
        // "wrong" direction's votes land on essentially random cells and
        // do not concentrate, per the header note).
        for (int sign = -1; sign <= 1; sign += 2) {
            const float fcx = static_cast<float>(x) + static_cast<float>(sign) * r * nx;
            const float fcy = static_cast<float>(y) + static_cast<float>(sign) * r * ny;
            const int cx = static_cast<int>(lroundf(fcx));
            const int cy = static_cast<int>(lroundf(fcy));
            if (cx < 0 || cx >= W || cy < 0 || cy >= H) continue;
            atomicAdd(&accum[(static_cast<long long>(k) * H + cy) * W + cx], 1);
        }
    }
}

void launch_hough_circles_vote(const unsigned char* d_edge_map, const float* d_gx, const float* d_gy,
                               int W, int H, int* d_accum)
{
    CUDA_CHECK(cudaMemset(d_accum, 0, sizeof(int) * static_cast<size_t>(HOUGH_CIRCLE_ACCUM_CELLS)));
    const int N = W * H;
    hough_circles_vote_kernel<<<flat_grid(N, kBlock), kBlock>>>(d_edge_map, d_gx, d_gy, W, H, d_accum);
    CUDA_CHECK_LAST_ERROR("hough_circles_vote_kernel launch");
}
