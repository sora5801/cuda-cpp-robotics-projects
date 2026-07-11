// ===========================================================================
// kernels.cu — GPU kernels for project 01.03
//              (Optical flow: dense pyramidal Lucas-Kanade + census-
//              transform block-matching flow)
//
// Big idea (the whole project in one paragraph)
// -----------------------------------------------
// Every kernel below is a MAP or a small STENCIL over an independent unit of
// work — a pixel, almost always — which is exactly why DENSE optical flow is
// such a natural GPU fit: every one of kW*kH pixels gets its own flow vector,
// entirely independently of every other pixel, WITHIN one pyramid level or
// one census pass. The one place parallelism runs out is BETWEEN pyramid
// levels: level L+1's flow field is only known once level L has finished
// (level L+1 initializes FROM level L's result — see upsample_flow_kernel),
// so the level loop is host-orchestrated and SEQUENTIAL (run_pyramidal_lk_gpu
// below), while every kernel WITHIN a level is embarrassingly parallel. This
// "parallel within a level, sequential across levels" shape is the single
// most important GPU-mapping idea in this file — THEORY.md's "GPU mapping"
// section names it explicitly.
//
// Kernel families:
//   1. downsample_area2x_kernel, scharr_gradient_kernel, structure_tensor_
//      kernel, census_transform_kernel — per-pixel MAPS/STENCILS: one thread
//      per pixel, no shared memory (a teaching simplification named
//      honestly in each kernel's header; THEORY.md's "GPU mapping" derives
//      the shared-memory-tiled faster versions as exercises).
//   2. lk_iterate_kernel, upsample_flow_kernel — per-pixel maps that ALSO
//      read another kernel's OUTPUT as a spatially-varying per-pixel
//      "control" input (the running flow estimate) — still one thread per
//      pixel, still no cross-thread communication.
//   3. census_match_kernel — a per-pixel map with a SEARCH LOOP inside each
//      thread (169 candidate displacements): still embarrassingly parallel
//      across pixels, but now compute-heavier per thread than 1-3 above —
//      see that kernel's header for the shared-memory-tiling argument this
//      project deliberately leaves as an exercise.
//   4. census_consistency_kernel — a per-pixel map reading two OTHER
//      kernels' full output arrays (forward and backward flow fields).
//
// All shared layouts, constants, and the census offset table are single-
// sourced in kernels.cuh — read that file's header first; it explains the
// bit-exact-vs-tolerance twin strategy this file and reference_cpu.cpp both
// depend on, and the exact per-milestone border arithmetic every kernel here
// respects.
//
// Read this after: kernels.cuh.  Companion oracle: reference_cpu.cpp (an
// INDEPENDENT re-implementation of every algorithmic core below — diff them
// side by side to see what "the same algorithm, twice" looks like).
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK / CUDA_CHECK_LAST_ERROR (§6.1 rule 7)

#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Launch geometry shared by every "one thread per pixel" kernel in this
// file: a 16x16 2-D block (256 threads, a warp multiple), grid sized to
// cover the CALLER-SUPPLIED W x H (pyramid levels vary in size, unlike
// 01.04's fixed kW/kH — every kernel here takes W,H as runtime parameters)
// with a ragged-tail guard inside each kernel body — the 01.04/01.02 idiom.
// ---------------------------------------------------------------------------
static constexpr int kBlock2D = 16;

static inline dim3 grid2d(int W, int H)
{
    return dim3((W + kBlock2D - 1) / kBlock2D, (H + kBlock2D - 1) / kBlock2D);
}

// ---------------------------------------------------------------------------
// Device-visible copy of the census offset table. kernels.cuh defines the
// VALUES once as a macro (CENSUS_DX_INIT/CENSUS_DY_INIT) precisely so this
// __constant__-memory array and reference_cpu.cpp's plain host array
// (kCensusDx/kCensusDy) can never drift apart — see that header's comment
// for the CUDA-language reason device code cannot simply read the host
// array directly. __constant__ (not plain __device__ global) memory is the
// right home: every thread in every block reads the SAME 24 offsets on
// every launch, so the constant cache broadcasts one read to a whole warp
// in one transaction — the textbook constant-memory use case.
// ---------------------------------------------------------------------------
__constant__ int kCensusDxDev[kCensusBits] = CENSUS_DX_INIT;
__constant__ int kCensusDyDev[kCensusBits] = CENSUS_DY_INIT;

// ---------------------------------------------------------------------------
// bilinear_sample_u8 — sample a uint8 grayscale image at a fractional
// coordinate (x, y), clamp-to-edge outside the image. __device__-only.
//
// Deliberately NOT shared with reference_cpu.cpp's independent host copy —
// the same design decision project 01.01's kernels.cuh documents in full
// for its bilinear_sample_rgb helper (cited here rather than re-derived):
// bilinear sampling is exactly the kind of "small, easy to get subtly
// wrong" arithmetic where an independently-written twin is worth more than
// a shared helper would save.
//
// Clamping strategy: clamp the CONTINUOUS coordinate first, then floor —
// so a coordinate 0.3 px outside the image samples the edge pixel with
// partial (not zero) weight, rather than reading out of bounds. This
// project always calls it with (x,y) already close to in-bounds (the LK
// warp offset rarely exceeds a few pixels once the pyramid has converged),
// so the clamp is a safety net, not a load-bearing approximation.
// ---------------------------------------------------------------------------
__device__ inline float bilinear_sample_u8(const uint8_t* __restrict__ img, int W, int H, float x, float y)
{
    x = fminf(fmaxf(x, 0.0f), static_cast<float>(W - 1));
    y = fminf(fmaxf(y, 0.0f), static_cast<float>(H - 1));

    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    const int x1 = min(x0 + 1, W - 1);
    const int y1 = min(y0 + 1, H - 1);
    const float fx = x - static_cast<float>(x0);   // fractional part, in [0,1)
    const float fy = y - static_cast<float>(y0);

    const float i00 = static_cast<float>(img[y0 * W + x0]);
    const float i10 = static_cast<float>(img[y0 * W + x1]);
    const float i01 = static_cast<float>(img[y1 * W + x0]);
    const float i11 = static_cast<float>(img[y1 * W + x1]);

    // Separable bilinear blend: interpolate along x on both rows, then
    // along y between the two row results — the standard 4-tap formula.
    const float top = i00 + (i10 - i00) * fx;
    const float bot = i01 + (i11 - i01) * fx;
    return top + (bot - top) * fy;
}

// ===========================================================================
// 1) DOWNSAMPLE — exact 2x area-average box filter, the single-channel twin
//    of project 01.01's resize_area2x_kernel. One thread per OUTPUT pixel;
//    each thread reads exactly 4 input texels and averages them.
//
// Why area-average (not bilinear, not nearest, not a Gaussian blur+
// subsample) is the CORRECT anti-aliasing filter for an INTEGER 2x
// decimation: a box filter whose support exactly matches the decimation
// ratio means every input pixel contributes to EXACTLY ONE output pixel
// with EQUAL weight — no input sample is dropped (nearest-neighbor would
// alias high-frequency texture straight into the coarser level, which is
// exactly the kind of aliasing this project's hashed multi-scale texture
// scene is designed to have PLENTY of) or double-counted. 01.01's THEORY.md
// derives the aliasing argument in full; this project cites rather than
// re-derives it, and gets a second payoff specific to optical flow: a
// coarser level's PIXEL GRID needs to represent the SAME physical motion
// at half the spatial resolution, and averaging is exactly the operation
// whose result is scale-consistent with "the image, viewed from twice as
// far away" — the physical model the pyramid's coarse-to-fine story relies
// on (THEORY.md "The algorithm").
// ===========================================================================
__global__ void downsample_area2x_kernel(const uint8_t* __restrict__ in, int inW, int inH,
                                         uint8_t* __restrict__ out)
{
    const int outW = inW / 2, outH = inH / 2;
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= outW || oy >= outH) return;

    const int ix = ox * 2, iy = oy * 2;   // this output pixel's 2x2 input block's top-left corner
    const int sum = static_cast<int>(in[ iy      * inW + ix    ]) + static_cast<int>(in[ iy      * inW + ix + 1]) +
                    static_cast<int>(in[(iy + 1) * inW + ix    ]) + static_cast<int>(in[(iy + 1) * inW + ix + 1]);
    // Integer average of 4 uint8 values (max sum 1020): round-to-nearest via
    // +2 before the /4 shift — exact integer arithmetic, no float involved,
    // so this stage is BIT-EXACT between the GPU and CPU paths (see
    // downsample_area2x_cpu's header for the twin-strategy statement).
    out[oy * outW + ox] = static_cast<uint8_t>((sum + 2) / 4);
}

void launch_downsample_area2x(const uint8_t* d_in, int inW, int inH, uint8_t* d_out)
{
    downsample_area2x_kernel<<<grid2d(inW / 2, inH / 2), dim3(kBlock2D, kBlock2D)>>>(d_in, inW, inH, d_out);
    CUDA_CHECK_LAST_ERROR("downsample_area2x_kernel launch");
}

// ===========================================================================
// 2) SCHARR GRADIENTS — a small STENCIL kernel (3x3 neighborhood -> two
//    derivative estimates). Border kGradBorder.
//
// Why Scharr instead of the plain Sobel stencil project 01.04 uses: Scharr's
// coefficients (3,10,3 rather than Sobel's 1,2,1) are the unique 3x3
// integer stencil that MINIMIZES rotational-symmetry error for a first-
// derivative estimate (Scharr 2000) — i.e. the estimated gradient DIRECTION
// is closer to the true continuous-image gradient direction regardless of
// the edge's orientation. Sobel's cheaper coefficients are adequate for
// FAST/Harris (01.04), which only ever ASKS "is there a corner here" — but
// Lucas-Kanade's structure tensor and mismatch vector are built directly
// FROM the gradient direction, and this project's rotation+zoom test scene
// specifically exercises motion at every possible edge orientation, so the
// small extra accuracy is worth the identical 3x3 footprint and cost
// (THEORY.md "Numerical considerations" shows the coefficient derivation).
//
// Numerics: every Scharr weight is a small integer (-10..10), input is
// uint8 (0..255), so the RAW convolution sums are EXACT integers up to
// magnitude 16*255=4080 — representable without rounding error in float32
// (exact up to 2^24).
//
// NORMALIZATION (load-bearing — this is not cosmetic): the raw integer
// convolution over-reports the true per-pixel intensity derivative by a
// factor of 32 (the Scharr stencil's positive-side weights sum to 16, and
// a central difference spans 2 pixels, so a unit-slope ramp produces a raw
// sum of 16*2=32 where the true derivative is 1). Lucas-Kanade's normal
// equations are NOT scale-invariant to this: M = [[Sxx,Sxy],[Sxy,Syy]] is
// QUADRATIC in the gradient (scales as k^2 for a gradient scaled by k),
// while b=(bx,by) is LINEAR in the gradient (scales as k^1, since It is a
// raw intensity difference that does not scale with k) — so the solved
// step dp = M^-1*b scales as k^2/k... inverted, i.e. as 1/k. An
// UNNORMALIZED (k=32) gradient therefore shrinks every computed LK
// increment by 32x, a severe undershoot that looks like "slow convergence"
// but is actually a wrong step size (root-caused empirically while
// building this project — see THEORY.md "Numerical considerations" for
// the worked 1-D derivation and the measured before/after numbers).
// Dividing by exactly 32.0f (a power of two) is an EXACT float32 operation
// (only the exponent changes, no rounding, no underflow at these
// magnitudes) — so this normalization costs NOTHING in the bit-exactness
// this stage's VERIFY claims: GPU and CPU still agree to the last bit.
// ===========================================================================
__global__ void scharr_gradient_kernel(const uint8_t* __restrict__ img, int W, int H,
                                       float* __restrict__ gx_out, float* __restrict__ gy_out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const int idx = y * W + x;

    if (x < kGradBorder || x >= W - kGradBorder || y < kGradBorder || y >= H - kGradBorder) {
        gx_out[idx] = 0.0f;
        gy_out[idx] = 0.0f;
        return;
    }

    const int i00 = img[(y - 1) * W + (x - 1)], i01 = img[(y - 1) * W + x], i02 = img[(y - 1) * W + (x + 1)];
    const int i10 = img[ y      * W + (x - 1)],                             i12 = img[ y      * W + (x + 1)];
    const int i20 = img[(y + 1) * W + (x - 1)], i21 = img[(y + 1) * W + x], i22 = img[(y + 1) * W + (x + 1)];

    // Scharr 3x3 kernels (the rotationally-optimal integer stencil, see
    // header):  Gx = [-3 0 3; -10 0 10; -3 0 3]   Gy = [-3 -10 -3; 0 0 0; 3 10 3]
    const int gx = (3 * i02 + 10 * i12 + 3 * i22) - (3 * i00 + 10 * i10 + 3 * i20);
    const int gy = (3 * i20 + 10 * i21 + 3 * i22) - (3 * i00 + 10 * i01 + 3 * i02);

    gx_out[idx] = static_cast<float>(gx) * (1.0f / 32.0f);   // exact power-of-two scale — see header numerics note
    gy_out[idx] = static_cast<float>(gy) * (1.0f / 32.0f);
}

void launch_scharr_gradient(const uint8_t* d_img, int W, int H, float* d_gx, float* d_gy)
{
    scharr_gradient_kernel<<<grid2d(W, H), dim3(kBlock2D, kBlock2D)>>>(d_img, W, H, d_gx, d_gy);
    CUDA_CHECK_LAST_ERROR("scharr_gradient_kernel launch");
}

// ===========================================================================
// 3) STRUCTURE TENSOR + CONFIDENCE — a STENCIL kernel over the gradient
//    images: thread (x,y) sums (Gx^2,Gy^2,Gx*Gy) over a (2*kLkWindowRadius+1)^2
//    = 5x5 = 25-tap box window, forming
//        M = [ Sxx Sxy ]     Sxx=sum(Gx^2), Syy=sum(Gy^2), Sxy=sum(Gx*Gy)
//            [ Sxy Syy ]
//    then BOTH of M's eigenvalues via the closed-form 2x2 symmetric formula
//        lambda = trace/2 +- sqrt((trace/2)^2 - det)
//    and reports the SMALL one as this project's per-pixel CONFIDENCE.
//
// Why the small eigenvalue IS the aperture problem, made numeric (THEORY.md
// derives this from first principles; the summary a reader needs at the
// kernel level): M's eigenvectors are the window's principal gradient
// directions; a shift ALONG the eigenvector with eigenvalue lambda changes
// the window's sum-of-squared-difference cost by ~lambda*shift^2 (a
// standard second-order Taylor argument on the SSD surface E(u,v) —
// THEORY.md "The math"). If the SMALL eigenvalue is tiny, there EXISTS a
// direction in which the window's content barely changes under a shift —
// motion along that direction is invisible to this window, exactly project
// 01.04's THEORY.md's aperture-problem argument (cited there for the
// two-eigenvalue corner/edge/flat taxonomy this project's confidence output
// makes numeric and per-pixel instead of a binary corner/not-corner call).
// A LARGE small-eigenvalue means BOTH directions are well constrained
// (texture, not just an edge) — the flow estimate there can be trusted.
//
// Numerics: the sqrt's argument (trace/2)^2 - det is mathematically >= 0 for
// any real symmetric 2x2 matrix (M's eigenvalues are always real), but
// floating-point rounding in the box-sum can occasionally produce a
// microscopically negative value for a near-flat window — clamped to 0
// before the sqrt (fmaxf) rather than let a NaN propagate silently.
// ===========================================================================
__global__ void structure_tensor_kernel(const float* __restrict__ gx, const float* __restrict__ gy,
                                        int W, int H,
                                        float* __restrict__ sxx_out, float* __restrict__ syy_out,
                                        float* __restrict__ sxy_out, float* __restrict__ min_eig_out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const int idx = y * W + x;

    if (x < kLkBorder || x >= W - kLkBorder || y < kLkBorder || y >= H - kLkBorder) {
        sxx_out[idx] = syy_out[idx] = sxy_out[idx] = min_eig_out[idx] = 0.0f;
        return;
    }

    float Sxx = 0.0f, Syy = 0.0f, Sxy = 0.0f;
    #pragma unroll
    for (int wy = -kLkWindowRadius; wy <= kLkWindowRadius; ++wy) {
        #pragma unroll
        for (int wx = -kLkWindowRadius; wx <= kLkWindowRadius; ++wx) {
            const int widx = (y + wy) * W + (x + wx);   // valid: kLkBorder >= kLkWindowRadius + kGradBorder, see kernels.cuh
            const float gxv = gx[widx];
            const float gyv = gy[widx];
            Sxx += gxv * gxv;
            Syy += gyv * gyv;
            Sxy += gxv * gyv;
        }
    }
    sxx_out[idx] = Sxx; syy_out[idx] = Syy; sxy_out[idx] = Sxy;

    // Closed-form eigenvalues of a symmetric 2x2 matrix (see header). The
    // SMALL one is min(lambda1,lambda2) = half_trace - sqrt(...), since
    // the sqrt term is always <= half_trace for a positive-semidefinite M
    // (Sxx,Syy>=0 and det=Sxx*Syy-Sxy^2>=0 by the Cauchy-Schwarz inequality
    // applied to the gradient samples — M is a sum of outer products, which
    // is always PSD by construction, so both eigenvalues are >= 0).
    const float half_trace = 0.5f * (Sxx + Syy);
    const float det = Sxx * Syy - Sxy * Sxy;
    const float disc = fmaxf(half_trace * half_trace - det, 0.0f);   // clamp: see header numerics note
    min_eig_out[idx] = half_trace - sqrtf(disc);
}

void launch_structure_tensor(const float* d_gx, const float* d_gy, int W, int H,
                             float* d_sxx, float* d_syy, float* d_sxy, float* d_min_eig)
{
    structure_tensor_kernel<<<grid2d(W, H), dim3(kBlock2D, kBlock2D)>>>(d_gx, d_gy, W, H, d_sxx, d_syy, d_sxy, d_min_eig);
    CUDA_CHECK_LAST_ERROR("structure_tensor_kernel launch");
}

// ===========================================================================
// 4) LK ITERATE — one forward-additive Lucas-Kanade refinement step, thread
//    per pixel. This is the kernel run kLkIterationsPerLevel times per
//    pyramid level (see run_pyramidal_lk_gpu below).
//
// The derivation (full algebra in THEORY.md "The math"; summary here): we
// seek an INCREMENT (ddu,ddv) to add to the running estimate (u,v) that
// minimizes, over the 5x5 window, the squared re-warp residual
//     sum_w [ I1(x+dx+u+ddu, y+dy+v+ddv) - I0(x+dx,y+dy) ]^2 .
// Linearizing I1's warped value in (ddu,ddv) via a first-order Taylor
// expansion (I1's spatial derivative is APPROXIMATED by I0's precomputed
// gradient — the standard forward-additive approximation, valid because
// (ddu,ddv) is small once the pyramid has provided a good starting point)
// gives the normal equations   M * [ddu;ddv] = -b,   b = [bx;by],
//     bx = sum_w Ix(x+dx,y+dy) * It(x+dx,y+dy),   It = I1_warped - I0,
//     by = sum_w Iy(x+dx,y+dy) * It(x+dx,y+dy),
// with M = [[Sxx,Sxy],[Sxy,Syy]] the SAME structure tensor computed once
// per level (this is exactly why Sxx/Syy/Sxy do not need recomputing every
// iteration: they depend only on I0's gradient, never on the current warp).
// Solving the 2x2 system in closed form (Cramer's rule):
//     ddu = -(Syy*bx - Sxy*by) / det,   ddv = -(-Sxy*bx + Sxx*by) / det,
//     det = Sxx*Syy - Sxy^2.
//
// Thread-to-data mapping: thread (x,y) owns pixel (x,y); the 5x5 window sum
// (25 bilinear samples of I1, 25 reads each of I0/Ix/Iy) is entirely local
// to this thread — no shared memory, no cross-thread communication, exactly
// like every other stencil kernel in this file (THEORY.md's "GPU mapping"
// argues why tiling I1 into shared memory would help here MORE than in the
// simpler stencils above, since 25 bilinear samples per thread means more
// reuse between neighboring threads' overlapping windows — named as the
// project's optimization exercise, not implemented, per the repo's
// teaching-first-then-name-the-optimization convention).
// ===========================================================================
__global__ void lk_iterate_kernel(const uint8_t* __restrict__ img0, const uint8_t* __restrict__ img1,
                                  int W, int H,
                                  const float* __restrict__ gx, const float* __restrict__ gy,
                                  const float* __restrict__ sxx, const float* __restrict__ syy,
                                  const float* __restrict__ sxy,
                                  float* __restrict__ flow_u, float* __restrict__ flow_v)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const int idx = y * W + x;
    if (x < kLkBorder || x >= W - kLkBorder || y < kLkBorder || y >= H - kLkBorder) return;   // flow left untouched (pyramid-propagated value stands)

    const float Sxx = sxx[idx], Syy = syy[idx], Sxy = sxy[idx];
    const float det = Sxx * Syy - Sxy * Sxy;
    if (det < kLkDetEpsilon) return;   // near-singular window (aperture problem / flat region) — do not inject noise

    const float u = flow_u[idx], v = flow_v[idx];   // the RUNNING estimate this iteration refines

    float bx = 0.0f, by = 0.0f;
    #pragma unroll
    for (int wy = -kLkWindowRadius; wy <= kLkWindowRadius; ++wy) {
        #pragma unroll
        for (int wx = -kLkWindowRadius; wx <= kLkWindowRadius; ++wx) {
            const int widx = (y + wy) * W + (x + wx);
            const float sample_x = static_cast<float>(x + wx) + u;
            const float sample_y = static_cast<float>(y + wy) + v;
            const float i1w = bilinear_sample_u8(img1, W, H, sample_x, sample_y);
            const float it = i1w - static_cast<float>(img0[widx]);   // the mismatch this iteration tries to null out
            bx += gx[widx] * it;
            by += gy[widx] * it;
        }
    }

    // Cramer's-rule solve (see header derivation) + per-iteration step clamp
    // (kLkMaxStepPerIterPx — see kernels.cuh's comment for why this exists).
    float ddu = -(Syy * bx - Sxy * by) / det;
    float ddv = -(-Sxy * bx + Sxx * by) / det;
    ddu = clamp_f(ddu, -kLkMaxStepPerIterPx, kLkMaxStepPerIterPx);
    ddv = clamp_f(ddv, -kLkMaxStepPerIterPx, kLkMaxStepPerIterPx);

    flow_u[idx] = u + ddu;
    flow_v[idx] = v + ddv;
}

void launch_lk_iterate(const uint8_t* d_img0, const uint8_t* d_img1, int W, int H,
                       const float* d_gx, const float* d_gy,
                       const float* d_sxx, const float* d_syy, const float* d_sxy,
                       float* d_flow_u, float* d_flow_v)
{
    lk_iterate_kernel<<<grid2d(W, H), dim3(kBlock2D, kBlock2D)>>>(
        d_img0, d_img1, W, H, d_gx, d_gy, d_sxx, d_syy, d_sxy, d_flow_u, d_flow_v);
    CUDA_CHECK_LAST_ERROR("lk_iterate_kernel launch");
}

// ===========================================================================
// 5) UPSAMPLE FLOW — bilinear-upsample a coarse-level flow field to the
//    next finer resolution AND double its magnitude. One thread per FINE
//    output pixel.
//
// Why double the magnitude (not just the resolution): a flow vector is a
// DISPLACEMENT IN PIXELS. The same physical motion that moved a scene point
// by N pixels at half resolution moves it by 2N pixels once the image is
// twice as large (twice as many pixels span the same physical/scene
// extent) — so propagating a coarse flow estimate to the next finer level
// requires scaling the VALUE, not just resampling its spatial grid. This is
// the mechanism that lets a large-magnitude motion (more pixels than the
// finest level's 5x5 window could ever capture directly — see THEORY.md's
// worked numeric example on the rotation+zoom scene) get discovered
// entirely at coarse, small-in-pixel-terms resolutions and then carried
// down as a good INITIAL GUESS for the fine levels' small, local
// refinements — the entire reason a pyramid earns its keep over a single
// level (README's pyramid_advantage gate measures this directly).
// ===========================================================================
__global__ void upsample_flow_kernel(const float* __restrict__ coarse_u, const float* __restrict__ coarse_v,
                                     int coarseW, int coarseH,
                                     float* __restrict__ fine_u, float* __restrict__ fine_v,
                                     int fineW, int fineH)
{
    const int fx = blockIdx.x * blockDim.x + threadIdx.x;
    const int fy = blockIdx.y * blockDim.y + threadIdx.y;
    if (fx >= fineW || fy >= fineH) return;

    // Map the fine pixel back to coarse-grid coordinates (half the position,
    // since the coarse grid spans the same scene at half the pixel density),
    // then bilinear-sample the coarse flow field there — a plain per-channel
    // bilinear read (u and v interpolated independently and identically).
    const float cx = (static_cast<float>(fx) + 0.5f) * 0.5f - 0.5f;
    const float cy = (static_cast<float>(fy) + 0.5f) * 0.5f - 0.5f;
    const float ccx = fminf(fmaxf(cx, 0.0f), static_cast<float>(coarseW - 1));
    const float ccy = fminf(fmaxf(cy, 0.0f), static_cast<float>(coarseH - 1));

    const int x0 = static_cast<int>(floorf(ccx)), y0 = static_cast<int>(floorf(ccy));
    const int x1 = min(x0 + 1, coarseW - 1), y1 = min(y0 + 1, coarseH - 1);
    const float wx = ccx - static_cast<float>(x0), wy = ccy - static_cast<float>(y0);

    auto bilerp = [&](const float* __restrict__ field) -> float {
        const float v00 = field[y0 * coarseW + x0], v10 = field[y0 * coarseW + x1];
        const float v01 = field[y1 * coarseW + x0], v11 = field[y1 * coarseW + x1];
        const float top = v00 + (v10 - v00) * wx;
        const float bot = v01 + (v11 - v01) * wx;
        return top + (bot - top) * wy;
    };

    const int fidx = fy * fineW + fx;
    fine_u[fidx] = 2.0f * bilerp(coarse_u);   // the x2 magnitude scale — see header
    fine_v[fidx] = 2.0f * bilerp(coarse_v);
}

void launch_upsample_flow(const float* d_coarse_u, const float* d_coarse_v, int coarseW, int coarseH,
                          float* d_fine_u, float* d_fine_v, int fineW, int fineH)
{
    upsample_flow_kernel<<<grid2d(fineW, fineH), dim3(kBlock2D, kBlock2D)>>>(
        d_coarse_u, d_coarse_v, coarseW, coarseH, d_fine_u, d_fine_v, fineW, fineH);
    CUDA_CHECK_LAST_ERROR("upsample_flow_kernel launch");
}

// ===========================================================================
// 6) CENSUS TRANSFORM — per-pixel 24-bit rank-order signature, a STENCIL
//    kernel (each thread reads its own 5x5 neighborhood). Border
//    kCensusRadius.
//
// The rank-order invariance argument (full derivation in THEORY.md "The
// problem"; the essential idea here): bit k of a pixel's signature encodes
// ONLY the ORDER relationship "is neighbor k brighter-or-equal to the
// center", never the actual intensity VALUES or their difference. Apply
// ANY strictly monotonically increasing function f (a brightness curve, a
// gamma change, a uniform gain, an additive constant, auto-exposure) to
// EVERY pixel in the image: since f preserves order (a < b implies
// f(a) < f(b) for any monotonic increasing f), every one of the 24
// brighter-or-equal comparisons yields the IDENTICAL bit both before and
// after the transform. The signature — and hence the Hamming distance
// between any two signatures, and hence the match this project's block
// matcher finds — is UNCHANGED. This is a much stronger guarantee than
// FAST's/ORB's "survives a uniform additive/multiplicative shift" claim
// (project 01.04's THEORY.md) because it holds for ANY monotonic curve,
// not just affine ones — the property this project's brightness-robustness
// gate (main.cu, scene (c)'s smooth spatial gradient) exercises directly.
// ===========================================================================
__global__ void census_transform_kernel(const uint8_t* __restrict__ img, int W, int H,
                                        uint32_t* __restrict__ census_out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const int idx = y * W + x;

    if (x < kCensusRadius || x >= W - kCensusRadius || y < kCensusRadius || y >= H - kCensusRadius) {
        census_out[idx] = 0u;
        return;
    }

    const int center = static_cast<int>(img[idx]);
    uint32_t sig = 0u;
    #pragma unroll
    for (int k = 0; k < kCensusBits; ++k) {
        const int nx = x + kCensusDxDev[k], ny = y + kCensusDyDev[k];
        const int neighbor = static_cast<int>(img[ny * W + nx]);
        // bit=1 means "neighbor >= center" (see header for the rank-order
        // invariance this convention delivers; the polarity itself — >=
        // here vs. < — is an arbitrary but FIXED choice, single-sourced by
        // being written identically in census_transform_cpu).
        if (neighbor >= center) sig |= (1u << k);
    }
    census_out[idx] = sig;
}

void launch_census_transform(const uint8_t* d_img, int W, int H, uint32_t* d_census)
{
    census_transform_kernel<<<grid2d(W, H), dim3(kBlock2D, kBlock2D)>>>(d_img, W, H, d_census);
    CUDA_CHECK_LAST_ERROR("census_transform_kernel launch");
}

// ===========================================================================
// 7) CENSUS BLOCK MATCH — thread per REFERENCE pixel: brute-force search of
//    all (2R+1)^2 = 169 candidate displacements IN-THREAD, Hamming winner-
//    take-all via __popc(), then per-axis parabolic sub-pixel refinement.
//    Border kCensusBorder.
//
// GPU-mapping honesty (THEORY.md "GPU mapping" expands this): this kernel
// is compute-heavier per thread than the stencils above (169 candidates x
// 1 popcount each, plus up to 4 more for sub-pixel refinement) but is STILL
// embarrassingly parallel ACROSS pixels — no thread's search depends on any
// other thread's result. The natural OPTIMIZATION (not implemented here,
// named as an exercise) is shared-memory TILING: threads in the same block
// search overlapping regions of census_tgt (a block of pixels near each
// other in census_ref search near-identical neighborhoods of census_tgt),
// so cooperatively staging a (blockDim+2R)^2 tile of census_tgt into shared
// memory once per BLOCK, instead of every thread independently re-reading
// overlapping global-memory neighborhoods, would cut global traffic
// roughly (2R+1)^2-fold in the interior of a block — the same "read once,
// reuse across threads" argument every tiled-stencil kernel in this repo
// makes, scaled up by this kernel's much larger window. Left as README's
// census-tiling exercise deliberately, so a learner derives the tile-size
// arithmetic (block size + 2*kCensusSearchRadius halo) themselves.
//
// Sub-pixel refinement (only when the winner is INTERIOR to the search
// window — see the guard below, which avoids needing extra border margin
// for the 4 extra Hamming evaluations): a 1-D parabola through 3 Hamming
// costs (bd-1, bd, bd+1) along each axis independently gives a closed-form
// offset  0.5*(c(-1)-c(+1)) / (c(-1) - 2*c(0) + c(+1))  — the vertex of the
// unique parabola through those 3 points. THEORY.md "Numerical
// considerations" discusses why this SEPARABLE (x-then-y) approximation to
// the true 2-D cost surface is a documented simplification, not a full 2-D
// quadratic fit (named as an exercise).
// ===========================================================================
__global__ void census_match_kernel(const uint32_t* __restrict__ census_ref,
                                    const uint32_t* __restrict__ census_tgt,
                                    int W, int H,
                                    float* __restrict__ flow_u, float* __restrict__ flow_v,
                                    int* __restrict__ cost_min_out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const int idx = y * W + x;

    if (x < kCensusBorder || x >= W - kCensusBorder || y < kCensusBorder || y >= H - kCensusBorder) {
        flow_u[idx] = 0.0f; flow_v[idx] = 0.0f; cost_min_out[idx] = kCensusBits + 1;
        return;
    }

    const uint32_t ref_sig = census_ref[idx];

    // In-thread search loop over every candidate displacement (see header).
    int best_cost = kCensusBits + 1, best_dx = 0, best_dy = 0;
    #pragma unroll 1   // a 13x13=169-iteration loop: unrolling fully would bloat code size for no benefit
    for (int dy = -kCensusSearchRadius; dy <= kCensusSearchRadius; ++dy) {
        for (int dx = -kCensusSearchRadius; dx <= kCensusSearchRadius; ++dx) {
            const uint32_t tgt_sig = census_tgt[(y + dy) * W + (x + dx)];
            const int cost = __popc(ref_sig ^ tgt_sig);   // Hamming distance: hardware population-count of the XOR (see 01.04's identical lesson)
            if (cost < best_cost) { best_cost = cost; best_dx = dx; best_dy = dy; }
        }
    }
    cost_min_out[idx] = best_cost;

    // Sub-pixel parabolic refinement (see header) — only when the winner is
    // strictly interior to the search window, so all 4 extra samples below
    // stay within the SAME already-validated [-R,R] search range (no extra
    // border margin needed; a winner AT the search boundary is honestly
    // left un-refined — an edge case the LR consistency check downstream
    // tends to reject anyway, since a true match rarely sits exactly at an
    // arbitrarily-chosen search radius).
    float sub_dx = 0.0f, sub_dy = 0.0f;
    if (best_dx > -kCensusSearchRadius && best_dx < kCensusSearchRadius &&
        best_dy > -kCensusSearchRadius && best_dy < kCensusSearchRadius) {
        const int c_xm = __popc(ref_sig ^ census_tgt[(y + best_dy) * W + (x + best_dx - 1)]);
        const int c_xp = __popc(ref_sig ^ census_tgt[(y + best_dy) * W + (x + best_dx + 1)]);
        const int c_ym = __popc(ref_sig ^ census_tgt[(y + best_dy - 1) * W + (x + best_dx)]);
        const int c_yp = __popc(ref_sig ^ census_tgt[(y + best_dy + 1) * W + (x + best_dx)]);

        const float denom_x = static_cast<float>(c_xm - 2 * best_cost + c_xp);
        if (denom_x > 1e-3f) sub_dx = 0.5f * static_cast<float>(c_xm - c_xp) / denom_x;
        const float denom_y = static_cast<float>(c_ym - 2 * best_cost + c_yp);
        if (denom_y > 1e-3f) sub_dy = 0.5f * static_cast<float>(c_ym - c_yp) / denom_y;
    }

    flow_u[idx] = static_cast<float>(best_dx) + sub_dx;
    flow_v[idx] = static_cast<float>(best_dy) + sub_dy;
}

void launch_census_match(const uint32_t* d_census_ref, const uint32_t* d_census_tgt, int W, int H,
                         float* d_flow_u, float* d_flow_v, int* d_cost_min)
{
    census_match_kernel<<<grid2d(W, H), dim3(kBlock2D, kBlock2D)>>>(
        d_census_ref, d_census_tgt, W, H, d_flow_u, d_flow_v, d_cost_min);
    CUDA_CHECK_LAST_ERROR("census_match_kernel launch");
}

// ===========================================================================
// 8) CENSUS CONSISTENCY — per-pixel forward/backward (left-right) check, a
//    plain map reading two flow fields.
//
// Geometric argument: if forward flow correctly matched scene point p (in
// image0) to point q = p + fwd(p) (in image1), then a correctly-matched
// BACKWARD search FROM q should find its way back to p — i.e.
// bwd(q) ~= -fwd(p). Occluded points, repeated/ambiguous texture, and
// search-radius truncation all tend to break this round trip (the backward
// match lands somewhere else), which is exactly why it is a good validity
// signal that needs NO ground truth to compute (THEORY.md "How we verify
// correctness" contrasts this against the ground-truth-based gates, which
// DO need the synthetic scene's known transform).
//
// bwd is sampled with NEAREST-NEIGHBOR lookup at the rounded target pixel
// (not bilinear) — a documented simplification: q is already a sub-pixel
// location, and correctly interpolating a FLOW FIELD (as opposed to an
// intensity image) at a fractional pixel is itself a modeling choice
// production optical-flow implementations spend real effort on (occlusion-
// aware interpolation, edge-preserving weights); nearest-neighbor is the
// honest teaching default here (THEORY.md names bilinear flow-field
// interpolation as the refinement).
// ===========================================================================
__global__ void census_consistency_kernel(const float* __restrict__ fwd_u, const float* __restrict__ fwd_v,
                                          const float* __restrict__ bwd_u, const float* __restrict__ bwd_v,
                                          int W, int H,
                                          uint8_t* __restrict__ valid_out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const int idx = y * W + x;

    if (x < kCensusBorder || x >= W - kCensusBorder || y < kCensusBorder || y >= H - kCensusBorder) {
        valid_out[idx] = 0u;
        return;
    }

    const float fu = fwd_u[idx], fv = fwd_v[idx];
    // Target pixel, nearest-neighbor rounded and clamped to the census-
    // eligible interior (a target landing outside it has no valid backward
    // flow to check against — correctly treated as inconsistent, i.e. 0).
    const int qx = clampi(static_cast<int>(lroundf(static_cast<float>(x) + fu)), kCensusBorder, W - 1 - kCensusBorder);
    const int qy = clampi(static_cast<int>(lroundf(static_cast<float>(y) + fv)), kCensusBorder, H - 1 - kCensusBorder);

    const float bu = bwd_u[qy * W + qx], bv = bwd_v[qy * W + qx];
    const float res_x = fu + bu, res_y = fv + bv;   // should be ~(0,0) for a consistent match — see header
    const float residual = sqrtf(res_x * res_x + res_y * res_y);

    valid_out[idx] = (residual <= kCensusConsistencyTolPx) ? 1u : 0u;
}

void launch_census_consistency(const float* d_fwd_u, const float* d_fwd_v,
                               const float* d_bwd_u, const float* d_bwd_v, int W, int H,
                               uint8_t* d_valid)
{
    census_consistency_kernel<<<grid2d(W, H), dim3(kBlock2D, kBlock2D)>>>(
        d_fwd_u, d_fwd_v, d_bwd_u, d_bwd_v, W, H, d_valid);
    CUDA_CHECK_LAST_ERROR("census_consistency_kernel launch");
}

// ===========================================================================
// ORCHESTRATION — the two host functions that own device scratch buffers
// across multiple kernel launches (kernels.cuh's header explains why these
// live here rather than in main.cu). Both are PLAIN HOST CODE (no <<<>>>
// of their own) that sequences the kernels above.
// ===========================================================================

// run_pyramidal_lk_gpu — see kernels.cuh for the parameter contract. Always
// BUILDS the full kNumLevels pyramid (cheap — a couple of downsample
// launches on a 160x120 image) but only USES `num_levels` of them in the
// coarse-to-fine loop, so main.cu can request num_levels=1 for the
// pyramid_advantage ablation without a second code path.
void run_pyramidal_lk_gpu(const uint8_t* d_img0_full, const uint8_t* d_img1_full,
                          int num_levels, int iters_per_level,
                          float* d_flow_u_out, float* d_flow_v_out, float* d_min_eig_out)
{
    // ---- allocate one buffer set per pyramid level -------------------------
    uint8_t* d_img0[kNumLevels]; uint8_t* d_img1[kNumLevels];
    float* d_gx[kNumLevels];     float* d_gy[kNumLevels];
    float* d_sxx[kNumLevels];    float* d_syy[kNumLevels];   float* d_sxy[kNumLevels];
    float* d_min_eig[kNumLevels];
    float* d_flow_u[kNumLevels]; float* d_flow_v[kNumLevels];

    for (int L = 0; L < kNumLevels; ++L) {
        const size_t n = static_cast<size_t>(level_w(L)) * level_h(L);
        CUDA_CHECK(cudaMalloc(&d_img0[L], n));
        CUDA_CHECK(cudaMalloc(&d_img1[L], n));
        CUDA_CHECK(cudaMalloc(&d_gx[L], n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_gy[L], n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sxx[L], n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_syy[L], n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sxy[L], n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_min_eig[L], n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_flow_u[L], n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_flow_v[L], n * sizeof(float)));
    }

    // ---- build the pyramid: level 0 = the caller's full-res frames --------
    CUDA_CHECK(cudaMemcpy(d_img0[0], d_img0_full, static_cast<size_t>(kW) * kH, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_img1[0], d_img1_full, static_cast<size_t>(kW) * kH, cudaMemcpyDeviceToDevice));
    for (int L = 1; L < kNumLevels; ++L) {
        launch_downsample_area2x(d_img0[L - 1], level_w(L - 1), level_h(L - 1), d_img0[L]);
        launch_downsample_area2x(d_img1[L - 1], level_w(L - 1), level_h(L - 1), d_img1[L]);
    }

    // ---- coarse-to-fine loop (SEQUENTIAL across levels — see file header) --
    const int start_level = num_levels - 1;   // the coarsest level actually used
    CUDA_CHECK(cudaMemset(d_flow_u[start_level], 0, static_cast<size_t>(level_w(start_level)) * level_h(start_level) * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_flow_v[start_level], 0, static_cast<size_t>(level_w(start_level)) * level_h(start_level) * sizeof(float)));

    for (int L = start_level; L >= 0; --L) {
        const int Wl = level_w(L), Hl = level_h(L);
        launch_scharr_gradient(d_img0[L], Wl, Hl, d_gx[L], d_gy[L]);
        launch_structure_tensor(d_gx[L], d_gy[L], Wl, Hl, d_sxx[L], d_syy[L], d_sxy[L], d_min_eig[L]);
        for (int it = 0; it < iters_per_level; ++it) {
            launch_lk_iterate(d_img0[L], d_img1[L], Wl, Hl, d_gx[L], d_gy[L], d_sxx[L], d_syy[L], d_sxy[L],
                              d_flow_u[L], d_flow_v[L]);
        }
        if (L > 0) {
            launch_upsample_flow(d_flow_u[L], d_flow_v[L], Wl, Hl,
                                 d_flow_u[L - 1], d_flow_v[L - 1], level_w(L - 1), level_h(L - 1));
        }
    }

    // ---- copy out the finest (level 0) result, then free every scratch buffer
    const size_t n0 = static_cast<size_t>(kW) * kH;
    CUDA_CHECK(cudaMemcpy(d_flow_u_out, d_flow_u[0], n0 * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_flow_v_out, d_flow_v[0], n0 * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_min_eig_out, d_min_eig[0], n0 * sizeof(float), cudaMemcpyDeviceToDevice));

    for (int L = 0; L < kNumLevels; ++L) {
        CUDA_CHECK(cudaFree(d_img0[L])); CUDA_CHECK(cudaFree(d_img1[L]));
        CUDA_CHECK(cudaFree(d_gx[L]));   CUDA_CHECK(cudaFree(d_gy[L]));
        CUDA_CHECK(cudaFree(d_sxx[L]));  CUDA_CHECK(cudaFree(d_syy[L])); CUDA_CHECK(cudaFree(d_sxy[L]));
        CUDA_CHECK(cudaFree(d_min_eig[L]));
        CUDA_CHECK(cudaFree(d_flow_u[L])); CUDA_CHECK(cudaFree(d_flow_v[L]));
    }
}

// run_census_flow_gpu — census-transform both frames once, match FORWARD
// (0 as reference, 1 as target) and BACKWARD (1 as reference, 0 as target —
// the SAME kernel, arguments swapped, see census_match_kernel's header),
// then consistency-check. Every stage here is a single kernel launch (no
// per-level loop — census needs no pyramid, see kernels.cuh's file header
// for why), so this orchestrator is much shorter than the LK one above.
void run_census_flow_gpu(const uint8_t* d_img0, const uint8_t* d_img1,
                         float* d_flow_u_out, float* d_flow_v_out, uint8_t* d_valid_out)
{
    const size_t n = static_cast<size_t>(kW) * kH;

    uint32_t *d_census0 = nullptr, *d_census1 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_census0, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_census1, n * sizeof(uint32_t)));
    launch_census_transform(d_img0, kW, kH, d_census0);
    launch_census_transform(d_img1, kW, kH, d_census1);

    int* d_cost_fwd = nullptr; int* d_cost_bwd = nullptr;
    float *d_bwd_u = nullptr, *d_bwd_v = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cost_fwd, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cost_bwd, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bwd_u, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bwd_v, n * sizeof(float)));

    launch_census_match(d_census0, d_census1, kW, kH, d_flow_u_out, d_flow_v_out, d_cost_fwd);   // forward: 0 -> 1
    launch_census_match(d_census1, d_census0, kW, kH, d_bwd_u, d_bwd_v, d_cost_bwd);             // backward: 1 -> 0
    launch_census_consistency(d_flow_u_out, d_flow_v_out, d_bwd_u, d_bwd_v, kW, kH, d_valid_out);

    CUDA_CHECK(cudaFree(d_census0)); CUDA_CHECK(cudaFree(d_census1));
    CUDA_CHECK(cudaFree(d_cost_fwd)); CUDA_CHECK(cudaFree(d_cost_bwd));
    CUDA_CHECK(cudaFree(d_bwd_u)); CUDA_CHECK(cudaFree(d_bwd_v));
}
