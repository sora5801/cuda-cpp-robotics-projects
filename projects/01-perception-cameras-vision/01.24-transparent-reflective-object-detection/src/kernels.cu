// ===========================================================================
// kernels.cu — GPU kernels for project 01.24
//              Transparent/reflective object detection via polarization
//              imaging
//
// Role in the project
// -------------------
// All __global__ (GPU) code lives here, together with the small host-side
// launch wrappers that own the grid/block math (the launch-configuration
// reasoning sits beside the code it configures, per CLAUDE.md §6.1).
//
// Pipeline shape (kernels.cuh Sections 2-4 derive every constant used
// below): mosaic -> demosaic (map) -> Stokes (map) -> DoLP/AoLP (map) ->
// Malus residual (map, a FREE self-consistency check) -> {threshold -> 3x3
// morphological open -> connected-component labeling -> size filter}, the
// last four run TWICE (once on DoLP, once on an intensity-contrast signal)
// by main.cu, which owns the two-signal orchestration.
//
// Every kernel here except demosaic is a pure MAP or a small fixed-radius
// STENCIL — this project's GPU story is deliberately "boring" (no reduction,
// no batched solve): the interesting parallelism is that FOUR independent
// polarizer measurements per super-pixel collapse into thousands of
// independent per-pixel physics evaluations, each embarrassingly parallel.
// Contrast this with 01.21/30.01's scatter-heavy stages (accumulation into
// shared bins) — this project's only atomics are in the CCL / component-size
// kernels, copied from that same lineage.
//
// Read this after: main.cu, kernels.cuh.  Read this before: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"

// ---------------------------------------------------------------------------
// kBlock1D — thread count for every 1-D (per-pixel or per-element) map
// kernel below. 256 is the repo-standard default (a warp multiple, good
// occupancy, small footprint) — see 08.01's launch_saxpy comment for the
// fuller reasoning; every kernel in this file reuses it rather than
// re-deriving the same tradeoff seven times.
// ---------------------------------------------------------------------------
static constexpr int kBlock1D = 256;
static constexpr int kBlock2D = 16;    // 16x16 = 256 threads/block for the 2-D stencils (erode/dilate)

static inline int  grid1d(int n, int block) { return (n + block - 1) / block; }
static inline dim3 grid2d(int W, int H)
{
    return dim3((W + kBlock2D - 1) / kBlock2D, (H + kBlock2D - 1) / kBlock2D);
}

// ===========================================================================
// STAGE 1 — DEMOSAIC. Reconstruct all 4 polarizer-angle channels at every
// pixel from the single-channel DoFP mosaic (kernels.cuh Section 2's phase
// layout). Kinship (cited, re-typed fresh): project 01.23's
// demosaic_bilinear_kernel does the identical THING (recover N virtual
// full-resolution channels from an N-way spatially-multiplexed mosaic) for
// a 3-channel Bayer CFA; here N=4 polarizer angles instead of R/G/G/B.
//
// One thread per OUTPUT pixel (not per mosaic sample: every thread computes
// all 4 channels for its pixel, a "wide map"). Own channel is copied
// directly (it IS what the sensor measured there); the other 3 are
// RECOVERED by bilinear interpolation across that channel's own
// spacing-2 sub-lattice (kernels.cuh's PhaseSample derives the footprint;
// the 4-corner weighted blend below is this project's own, independently
// written interpolation step — see reference_cpu.cpp's CPU twin, which
// computes the SAME 4 outputs from a hand-written loop, not by calling this
// kernel's logic).
//
// HONESTY NOTE (documented here AND in THEORY.md "Numerical
// considerations"): this per-angle bilinear reconstruction is the textbook
// baseline, not the state of the art — it treats each angle channel as an
// independent image, ignoring the fact that all 4 come from the SAME
// underlying scene radiance. Real DoFP ISPs (README "Prior art") use
// edge-aware or intensity-correlated demosaicing (the polarization analogue
// of 01.23's Malvar-He-Cutler upgrade over plain bilinear) to reduce the
// "instantaneous FOV" artifact: each 2x2 super-pixel's four measurements
// come from four DIFFERENT physical photosites (not the same point in the
// scene), so any single-pixel-wide edge in the true scene is smeared across
// up to a 2-pixel neighborhood after reconstruction — visible as the mild
// blur this project's own stokes_accuracy gate margins for near object
// boundaries (kernels.cuh's kInteriorMarginPx exists BECAUSE of this).
// ---------------------------------------------------------------------------
__global__ void demosaic_polarization_kernel(const float* __restrict__ mosaic,
                                             float* __restrict__ channels4,
                                             int W, int H)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= W * H) return;
    const int x = i % W, y = i / W;               // this pixel's 2-D coordinate
    const int px = x & 1, py = y & 1;              // this pixel's OWN super-pixel phase (0/1 each axis)
    const int own_c = dofp_channel_for_phase(px, py);

    #pragma unroll
    for (int c = 0; c < kNumChannels; ++c) {
        float v;
        if (c == own_c) {
            // The direct measurement — no interpolation needed or wanted.
            v = mosaic[i];
        } else {
            // Recover channel c from ITS OWN phase's spacing-2 sub-lattice,
            // bracketing (x,y) with the 4 nearest same-phase samples.
            int tpx, tpy;
            dofp_phase_for_channel(c, tpx, tpy);
            const PhaseSample s = phase_sample_at(x, y, tpx, tpy, W, H);
            const float v00 = mosaic[s.y0 * W + s.x0];
            const float v10 = mosaic[s.y0 * W + s.x1];
            const float v01 = mosaic[s.y1 * W + s.x0];
            const float v11 = mosaic[s.y1 * W + s.x1];
            // Standard 4-corner bilinear blend — the ALGORITHMIC step this
            // project's twin-independence ruling requires be written twice;
            // this arithmetic form is THIS file's own (reference_cpu.cpp
            // writes an independently-structured nested-nearest-sample loop).
            v = (1.0f - s.wx) * (1.0f - s.wy) * v00
              +         s.wx  * (1.0f - s.wy) * v10
              + (1.0f - s.wx) *         s.wy  * v01
              +         s.wx  *         s.wy  * v11;
        }
        channels4[i * kNumChannels + c] = v;
    }
}
void launch_demosaic_polarization(const float* d_mosaic, float* d_channels4, int W, int H)
{
    const int n = W * H;
    demosaic_polarization_kernel<<<grid1d(n, kBlock1D), kBlock1D>>>(d_mosaic, d_channels4, W, H);
    CUDA_CHECK_LAST_ERROR("demosaic_polarization_kernel launch");
}

// ===========================================================================
// STAGE 2 — STOKES estimation. THEORY.md "The math" derives WHY this is the
// (unweighted) least-squares solution of Malus's law (*) sampled at
// theta=0,45,90,135 deg: I0=S0/2+S1/2, I90=S0/2-S1/2 give TWO independent
// estimates of S0 (I0+I90 and, symmetrically, I45+I135); averaging them
// (the /2 after summing all four) is the optimal combination under equal-
// variance per-channel noise. S1 and S2 each have only ONE estimate (no
// redundancy), hence the pure differences below.
// ---------------------------------------------------------------------------
__global__ void stokes_kernel(const float* __restrict__ channels4,
                              float* __restrict__ s0, float* __restrict__ s1, float* __restrict__ s2, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float I0   = channels4[i * kNumChannels + 0];
    const float I45  = channels4[i * kNumChannels + 1];
    const float I90  = channels4[i * kNumChannels + 2];
    const float I135 = channels4[i * kNumChannels + 3];
    s0[i] = 0.5f * (I0 + I45 + I90 + I135);   // two redundant S0 estimates, averaged
    s1[i] = I0 - I90;                          // the ONLY estimate of S1
    s2[i] = I45 - I135;                        // the ONLY estimate of S2
}
void launch_stokes(const float* d_channels4, float* d_s0, float* d_s1, float* d_s2, int n)
{
    stokes_kernel<<<grid1d(n, kBlock1D), kBlock1D>>>(d_channels4, d_s0, d_s1, d_s2, n);
    CUDA_CHECK_LAST_ERROR("stokes_kernel launch");
}

// ===========================================================================
// STAGE 3 — DoLP / AoLP. DoLP = sqrt(S1^2+S2^2)/S0 (an epsilon-guarded
// division — THEORY.md "Numerical considerations" derives the LOW-S0 bias
// this guard introduces and why it is harmless here: S0 in this scene never
// approaches zero). AoLP = 0.5*atan2(S2,S1), THEN WRAPPED into [0, pi): a
// linear polarizer at angle theta and theta+pi are physically the SAME axis
// (Malus's law (*) is pi-periodic in theta, not 2*pi), but atan2's natural
// range is (-pi,pi], so the raw 0.5*atan2(...) result lands in (-pi/2,pi/2]
// — the "half-angle wrap" THEORY.md teaches: add pi to any negative result.
// ---------------------------------------------------------------------------
__global__ void dolp_aolp_kernel(const float* __restrict__ s0, const float* __restrict__ s1,
                                 const float* __restrict__ s2,
                                 float* __restrict__ dolp, float* __restrict__ aolp_rad, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float S0 = s0[i], S1 = s1[i], S2 = s2[i];
    const float mag = sqrtf(S1 * S1 + S2 * S2);
    const float s0_safe = fmaxf(S0, 1.0e-3f);   // epsilon floor: guards div-by-~0 without biasing this scene's real S0 range
    dolp[i] = mag / s0_safe;
    float a = 0.5f * atan2f(S2, S1);            // in (-pi/2, pi/2]
    if (a < 0.0f) a += kPi;                      // the half-angle wrap -> [0, pi)
    aolp_rad[i] = a;
}
void launch_dolp_aolp(const float* d_s0, const float* d_s1, const float* d_s2,
                      float* d_dolp, float* d_aolp_rad, int n)
{
    dolp_aolp_kernel<<<grid1d(n, kBlock1D), kBlock1D>>>(d_s0, d_s1, d_s2, d_dolp, d_aolp_rad, n);
    CUDA_CHECK_LAST_ERROR("dolp_aolp_kernel launch");
}

// ===========================================================================
// STAGE 4 — the Malus self-consistency residual: THE FREE invariant.
// 4 raw measurements (I0,I45,I90,I135) feed a 3-parameter model (S0,S1,S2)
// -> 1 degree of freedom is NOT used to fit anything, and is therefore
// available to CHECK the four measurements against each other with no
// external ground truth at all. Concretely: the model predicts TWO
// independent ways to recover S0 (I0+I90 and I45+I135); in noise-free
// physics they are EXACTLY equal (THEORY.md "The math" proves this from
// (*)), so their difference is a pure self-consistency residual — zero
// unless something is inconsistent (sensor noise, a demosaic/registration
// bug, or a nonlinearity Malus's law does not model).
// ---------------------------------------------------------------------------
__global__ void malus_residual_kernel(const float* __restrict__ channels4, float* __restrict__ residual, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float I0   = channels4[i * kNumChannels + 0];
    const float I45  = channels4[i * kNumChannels + 1];
    const float I90  = channels4[i * kNumChannels + 2];
    const float I135 = channels4[i * kNumChannels + 3];
    residual[i] = (I0 + I90) - (I45 + I135);
}
void launch_malus_residual(const float* d_channels4, float* d_residual, int n)
{
    malus_residual_kernel<<<grid1d(n, kBlock1D), kBlock1D>>>(d_channels4, d_residual, n);
    CUDA_CHECK_LAST_ERROR("malus_residual_kernel launch");
}

// ===========================================================================
// STAGE 5 — DETECTION. Five small kernels main.cu composes into ONE
// pipeline, called TWICE with two different (signal, threshold) pairs —
// once for DoLP, once for the intensity-contrast baseline (main.cu builds
// that second signal with abs_diff_scalar_kernel below, feeding it S0 and
// S0's own image-mean). This is the literal GPU implementation of the
// README's "designed comparison": same five kernels, two signals, two very
// different outcomes on the glass objects.
// ===========================================================================

// abs_diff_scalar_kernel — out[i] = |signal[i] - ref_scalar|. Pure map; the
// "intensity contrast" signal is exactly this applied to (S0, mean(S0)).
__global__ void abs_diff_scalar_kernel(const float* __restrict__ signal, float ref_scalar,
                                       float* __restrict__ out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = fabsf(signal[i] - ref_scalar);
}
void launch_abs_diff_scalar(const float* d_signal, float ref_scalar, float* d_out, int n)
{
    abs_diff_scalar_kernel<<<grid1d(n, kBlock1D), kBlock1D>>>(d_signal, ref_scalar, d_out, n);
    CUDA_CHECK_LAST_ERROR("abs_diff_scalar_kernel launch");
}

// threshold_kernel — mask[i] = signal[i] >= thresh. Pure map.
__global__ void threshold_kernel(const float* __restrict__ signal, float thresh,
                                 uint8_t* __restrict__ mask_out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    mask_out[i] = (signal[i] >= thresh) ? 1u : 0u;
}
void launch_threshold(const float* d_signal, float thresh, uint8_t* d_mask_out, int n)
{
    threshold_kernel<<<grid1d(n, kBlock1D), kBlock1D>>>(d_signal, thresh, d_mask_out, n);
    CUDA_CHECK_LAST_ERROR("threshold_kernel launch");
}

// erode3x3_kernel / dilate3x3_kernel — 01.21's cited morphological-opening
// pair (itself citing 30.01), re-typed fresh for this project's uint8_t
// mask layout. erode: output=1 iff EVERY pixel in the 3x3 window (out-of-
// bounds reads as 0) is 1. dilate: output=1 iff ANY pixel is 1. Erode-then-
// dilate (an OPENING) removes isolated few-pixel false positives (stray
// noise-driven threshold crossings) without shrinking a surviving blob's
// silhouette — see launch_morphological_open below.
__global__ void erode3x3_kernel(const uint8_t* __restrict__ in, int W, int H, uint8_t* __restrict__ out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    uint8_t v = 1u;
    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        #pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            const int nx = x + dx, ny = y + dy;
            const uint8_t nb = (nx < 0 || nx >= W || ny < 0 || ny >= H) ? 0u : in[ny * W + nx];
            v &= nb;
        }
    }
    out[y * W + x] = v;
}
__global__ void dilate3x3_kernel(const uint8_t* __restrict__ in, int W, int H, uint8_t* __restrict__ out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    uint8_t v = 0u;
    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        #pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            const int nx = x + dx, ny = y + dy;
            const uint8_t nb = (nx < 0 || nx >= W || ny < 0 || ny >= H) ? 0u : in[ny * W + nx];
            v |= nb;
        }
    }
    out[y * W + x] = v;
}
void launch_morphological_open(uint8_t* d_mask_inout, int W, int H)
{
    uint8_t* d_scratch = nullptr;
    CUDA_CHECK(cudaMalloc(&d_scratch, static_cast<size_t>(W) * H));
    erode3x3_kernel<<<grid2d(W, H), dim3(kBlock2D, kBlock2D)>>>(d_mask_inout, W, H, d_scratch);
    CUDA_CHECK_LAST_ERROR("erode3x3_kernel launch");
    dilate3x3_kernel<<<grid2d(W, H), dim3(kBlock2D, kBlock2D)>>>(d_scratch, W, H, d_mask_inout);
    CUDA_CHECK_LAST_ERROR("dilate3x3_kernel launch");
    CUDA_CHECK(cudaFree(d_scratch));
}

// ccl_init_kernel / ccl_propagate_sweep_kernel — 01.21's cited (itself
// citing 01.06) label-propagation connected-component labeling, re-typed
// fresh: every foreground pixel starts labeled with its OWN linear index
// (the only label it can be sure of before any neighbor information
// propagates); each sweep replaces a pixel's label with the MINIMUM of its
// own and its 4-connected foreground neighbors' CURRENT labels via
// atomicMin. Labels only ever DECREASE and are bounded below by 0, so
// repeated sweeps converge (main.cu's/this file's host loop) to the UNIQUE
// fixed point label[p] = min linear index over p's connected component —
// independent of scheduling order, which is exactly why this GPU algorithm
// and reference_cpu.cpp's completely different union-find algorithm can be
// held to a BIT-EXACT tolerance (both converge to the same canonical label).
__global__ void ccl_init_kernel(const uint8_t* __restrict__ mask, int* __restrict__ label, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    label[i] = mask[i] ? i : -1;   // -1 = "no label" (background); matches reference_cpu.cpp's kLabelNone
}

__global__ void ccl_propagate_sweep_kernel(const uint8_t* __restrict__ mask, int* __restrict__ label,
                                           int W, int H, int* __restrict__ changed)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= W * H) return;
    if (!mask[i]) return;
    const int x = i % W, y = i / W;
    int best = label[i];
    if (x > 0     && mask[i - 1]) best = min(best, label[i - 1]);
    if (x < W - 1 && mask[i + 1]) best = min(best, label[i + 1]);
    if (y > 0     && mask[i - W]) best = min(best, label[i - W]);
    if (y < H - 1 && mask[i + W]) best = min(best, label[i + W]);
    if (best < label[i]) {
        atomicMin(&label[i], best);
        atomicOr(changed, 1);
    }
}

// launch_connected_components — owns the host-side convergence loop: keep
// sweeping while ANY pixel's label changed last round, up to kMaxCclSweeps
// (kernels.cuh's derivation of that safety cap). Returns the sweep count
// actually used, which main.cu reports as an [info] convergence diagnostic
// (mirrors 01.21's "[info] connected_components" line).
int launch_connected_components(const uint8_t* d_mask, int* d_label, int W, int H)
{
    const int n = W * H;
    ccl_init_kernel<<<grid1d(n, kBlock1D), kBlock1D>>>(d_mask, d_label, n);
    CUDA_CHECK_LAST_ERROR("ccl_init_kernel launch");

    int* d_changed = nullptr;
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));
    int sweeps = 0;
    for (; sweeps < kMaxCclSweeps; ++sweeps) {
        CUDA_CHECK(cudaMemset(d_changed, 0, sizeof(int)));
        ccl_propagate_sweep_kernel<<<grid1d(n, kBlock1D), kBlock1D>>>(d_mask, d_label, W, H, d_changed);
        CUDA_CHECK_LAST_ERROR("ccl_propagate_sweep_kernel launch");
        int h_changed = 0;
        CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        if (!h_changed) { ++sweeps; break; }   // count the CONFIRMING (no-op) sweep too, matching the CPU twin's convention
    }
    CUDA_CHECK(cudaFree(d_changed));
    return sweeps;
}

// component_size_count_kernel / component_filter_kernel — 01.21's cited
// atomic-scatter size accumulation (itself citing 01.06/30.01's dense-
// accumulator-keyed-by-label idea), re-typed fresh: every foreground pixel
// atomically adds 1 to its canonical label's size bucket, then a second map
// keeps a pixel only if its OWN component cleared kMinComponentSizePx.
__global__ void component_size_count_kernel(const uint8_t* __restrict__ mask, const int* __restrict__ label,
                                            int* __restrict__ size_out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!mask[i]) return;
    atomicAdd(&size_out[label[i]], 1);
}
__global__ void component_filter_kernel(const uint8_t* __restrict__ mask_in, const int* __restrict__ label,
                                        const int* __restrict__ size, int min_size,
                                        uint8_t* __restrict__ mask_out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    mask_out[i] = (mask_in[i] && size[label[i]] >= min_size) ? 1u : 0u;
}
void launch_component_size_filter(const uint8_t* d_mask_in, const int* d_label, int min_size_px,
                                  uint8_t* d_mask_out, int n)
{
    int* d_size = nullptr;
    CUDA_CHECK(cudaMalloc(&d_size, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_size, 0, static_cast<size_t>(n) * sizeof(int)));
    component_size_count_kernel<<<grid1d(n, kBlock1D), kBlock1D>>>(d_mask_in, d_label, d_size, n);
    CUDA_CHECK_LAST_ERROR("component_size_count_kernel launch");
    component_filter_kernel<<<grid1d(n, kBlock1D), kBlock1D>>>(d_mask_in, d_label, d_size, min_size_px, d_mask_out, n);
    CUDA_CHECK_LAST_ERROR("component_filter_kernel launch");
    CUDA_CHECK(cudaFree(d_size));
}
