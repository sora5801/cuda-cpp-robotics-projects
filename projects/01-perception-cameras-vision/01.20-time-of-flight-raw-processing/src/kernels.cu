// ===========================================================================
// kernels.cu — GPU kernels for project 01.20
//              Time-of-flight raw processing: phase unwrapping, flying-pixel
//              removal (continuous-wave indirect ToF)
//
// Role in the project
// -------------------
// Six kernels, one per pipeline stage. Five of them (Stages 1-3, 4, 6) are
// pure per-pixel MAPS, exactly like every kernel in 01.19 — no thread ever
// reads or writes another thread's pixel, so no shared memory, no atomics
// (kernels.cuh "Why no atomics anywhere", inherited unchanged from 01.19).
// Stage 5 (`flying_pixel_detect_kernel`) is this project's ONE genuine
// STENCIL kernel: each thread GATHERS its 3x3 neighborhood of ALREADY-
// COMPUTED depths/amplitudes. It is still race-free (every thread writes
// only its own output pixel; it only READS neighbors), so no atomics are
// needed even here — but it is the first kernel in this project's lineage
// (and unlike ANY kernel in 01.19) whose output genuinely depends on more
// than one pixel's input, the new GPU-mapping idea this project adds
// (THEORY.md "The GPU mapping" contrasts the two projects explicitly).
//
// Read this after: kernels.cuh. Companion oracle: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"      // CUDA_CHECK_LAST_ERROR (CLAUDE.md §6.1 rule 7)

#include <cstdio>
#include <cstdlib>

// Shared launch geometry: 256 threads/block, the repo default (warp
// multiple, good occupancy on sm_75..sm_89 without starving the register
// file — the same choice 01.19/08.01 make). grid = ceil(n/block) covers
// every pixel exactly once; n is at most kNPix = 19,200 here, comfortably
// under any grid-dimension limit, so no grid-stride loop is needed.
static inline int grid_for(int n, int block) { return (n + block - 1) / block; }

// ===========================================================================
// Stage 1 — extract_phase_amplitude_kernel: 4-tap correlation -> phase +
// amplitude. One thread per camera pixel; called ONCE PER FREQUENCY by
// main.cu (this kernel does not know or care which frequency its tap frames
// came from — the frequency only matters once a DEPTH is computed, in Stage
// 2/3 below).
//
// Thread-to-data mapping: pix = blockIdx.x*blockDim.x + threadIdx.x owns
// camera pixel `pix` (row-major, pix = row*kCamW + col).
//
// Memory per thread: reads 4 floats at FIXED, compile-time-known offsets
// (0,1,2,3)*n — not even a runtime loop, so all four are back-to-back
// coalesced 128-byte-per-warp transactions (kernels.cuh "Pattern-stack
// memory layout", the same argument 01.19's phase_decode_kernel makes),
// writes 2 floats. No shared memory (nothing reused across threads), no
// atomics (kernels.cuh "atomics" note).
//
// Math (kernels.cuh file header derives this from the C_k(phi) = A + B*cos
// (phi + k*pi/2) forward model): with C0..C3 the four taps,
//     phase     = atan2(C3-C1, C0-C2)     wrapped to [0, 2*pi)
//     amplitude = 0.5*sqrt((C3-C1)^2 + (C0-C2)^2)      (= B, the confidence)
// atan2f/sqrtf (not the fast __-prefixed intrinsics) are used deliberately:
// this kernel's output feeds a hard PASS/FAIL amplitude-floor threshold
// downstream (confidence_mask_kernel) and every depth in the demo — the
// same "accuracy over a few-ULP speedup" call 01.19's phase_decode_kernel
// makes, for the identical reason (this kernel is bandwidth-bound, not
// compute-bound: THEORY.md "The GPU mapping" makes the quantitative case).
// ===========================================================================
__global__ void extract_phase_amplitude_kernel(const float* __restrict__ d_taps,
                                               float*       __restrict__ d_phase,
                                               float*       __restrict__ d_amplitude,
                                               int n)
{
    const int pix = blockIdx.x * blockDim.x + threadIdx.x;
    if (pix >= n) return;

    const float c0 = d_taps[0 * n + pix];
    const float c1 = d_taps[1 * n + pix];
    const float c2 = d_taps[2 * n + pix];
    const float c3 = d_taps[3 * n + pix];

    const float num = c3 - c1;   // =  2*B*sin(phi) under this project's C_k(phi)=A+B*cos(phi+k*pi/2) convention
    const float den = c0 - c2;   // =  2*B*cos(phi) — the AMBIENT term A cancels EXACTLY in both differences

    float phi = atan2f(num, den);                       // (-pi, pi]
    if (phi < 0.0f) phi += 6.28318530717958647692f;     // this project's [0,2pi) convention (kernels.cuh)

    d_phase[pix]     = phi;
    d_amplitude[pix] = 0.5f * sqrtf(num * num + den * den);   // = B
}

void launch_extract_phase_amplitude(const float* d_taps, float* d_phase, float* d_amplitude, int n)
{
    const int block = 256;
    extract_phase_amplitude_kernel<<<grid_for(n, block), block>>>(d_taps, d_phase, d_amplitude, n);
    CUDA_CHECK_LAST_ERROR("extract_phase_amplitude_kernel launch");
}

// ===========================================================================
// Stage 2 — single_freq_depth_kernel: the WRAPPED, single-frequency depth
// estimate depth = (phase/2pi) * ambiguity_range. One thread per pixel;
// deliberately the simplest possible kernel in this project — its entire
// pedagogical purpose is to be compared against ground truth by the
// aliasing_demo gate (main.cu), which shows this naive estimate is badly
// WRONG for any pixel whose true depth exceeds `ambiguity_range_m`: the
// "designed aliasing demonstration" the catalog bullet asks for.
// ===========================================================================
__global__ void single_freq_depth_kernel(const float* __restrict__ d_phase,
                                         float ambiguity_range_m,
                                         float*       __restrict__ d_depth,
                                         int n)
{
    const int pix = blockIdx.x * blockDim.x + threadIdx.x;
    if (pix >= n) return;

    d_depth[pix] = (d_phase[pix] * (1.0f / 6.28318530717958647692f)) * ambiguity_range_m;
}

void launch_single_freq_depth(const float* d_phase, float ambiguity_range_m, float* d_depth, int n)
{
    const int block = 256;
    single_freq_depth_kernel<<<grid_for(n, block), block>>>(d_phase, ambiguity_range_m, d_depth, n);
    CUDA_CHECK_LAST_ERROR("single_freq_depth_kernel launch");
}

// ===========================================================================
// Stage 3 — dual_freq_unwrap_kernel: the CRT-style integer-wrap consistency
// search (kernels.cuh "Two frequencies, then unwrapping" derives the full
// argument). One thread per camera pixel.
//
// For each candidate wrap count n1 in [0, kMaxWraps1) of the FINE channel
// and n2 in [0, kMaxWraps2) of the COARSE channel, compute the two
// candidate depths
//     z1(n1) = (phase1/2pi + n1) * kAmbig1M
//     z2(n2) = (phase2/2pi + n2) * kAmbig2M
// and keep the (n1, n2) pair whose depths AGREE most closely (minimum
// |z1-z2|). The winning z1(n1) — the FINE channel's depth at its winning
// wrap — is reported as this pixel's unwrapped depth, because per the
// noise-scaling law (THEORY.md "Numerical considerations",
// sigma_d ~ c/(4*pi*f)*sigma_phi/B) the higher-frequency channel gives
// PROPORTIONALLY BETTER raw precision; the coarse channel's only job is
// resolving WHICH wrap, exactly 01.19's "Gray resolves the period, phase
// refines it" pattern with (coarse CW, fine CW) standing in for (Gray code,
// phase-shift).
//
// The search is a tiny, compile-time-bounded double loop
// (kMaxWraps1*kMaxWraps2 <= 3 iterations on this project's committed scene)
// — negligible compute next to the kernel's 2-float-read/1-float-write
// memory traffic; still bandwidth-bound, not compute-bound.
// ===========================================================================
__global__ void dual_freq_unwrap_kernel(const float* __restrict__ d_phase1,
                                        const float* __restrict__ d_phase2,
                                        float*         __restrict__ d_depth,
                                        int*           __restrict__ d_wrap_count,
                                        int n)
{
    const int pix = blockIdx.x * blockDim.x + threadIdx.x;
    if (pix >= n) return;

    const float frac1 = d_phase1[pix] * (1.0f / 6.28318530717958647692f);   // phase1/2pi, in [0,1)
    const float frac2 = d_phase2[pix] * (1.0f / 6.28318530717958647692f);   // phase2/2pi, in [0,1)

    float   best_diff = 3.4e38f;   // FLT_MAX-ish sentinel; the search always finds a finite candidate
    float   best_z1    = 0.0f;
    int     best_n1     = 0;
#pragma unroll
    for (int n1 = 0; n1 < kMaxWraps1; ++n1) {
        const float z1 = (frac1 + static_cast<float>(n1)) * kAmbig1M;
#pragma unroll
        for (int n2 = 0; n2 < kMaxWraps2; ++n2) {
            const float z2 = (frac2 + static_cast<float>(n2)) * kAmbig2M;
            const float diff = fabsf(z1 - z2);
            if (diff < best_diff) {
                best_diff = diff;
                best_z1 = z1;
                best_n1 = n1;
            }
        }
    }
    d_depth[pix]      = best_z1;
    d_wrap_count[pix] = best_n1;
}

void launch_dual_freq_unwrap(const float* d_phase1, const float* d_phase2,
                             float* d_depth, int* d_wrap_count, int n)
{
    const int block = 256;
    dual_freq_unwrap_kernel<<<grid_for(n, block), block>>>(d_phase1, d_phase2, d_depth, d_wrap_count, n);
    CUDA_CHECK_LAST_ERROR("dual_freq_unwrap_kernel launch");
}

// ===========================================================================
// Stage 4 — confidence_mask_kernel: amplitude-floor threshold. One thread
// per pixel; deliberately trivial (same "atan2 conditioning collapses as
// B->0" justification as 01.19's hybrid_combine_kernel confidence check —
// kernels.cuh "Confidence / validity").
// ===========================================================================
__global__ void confidence_mask_kernel(const float* __restrict__ d_amplitude,
                                       float amplitude_floor,
                                       unsigned char* __restrict__ d_valid,
                                       int n)
{
    const int pix = blockIdx.x * blockDim.x + threadIdx.x;
    if (pix >= n) return;

    d_valid[pix] = (d_amplitude[pix] >= amplitude_floor) ? 1 : 0;
}

void launch_confidence_mask(const float* d_amplitude, float amplitude_floor, unsigned char* d_valid, int n)
{
    const int block = 256;
    confidence_mask_kernel<<<grid_for(n, block), block>>>(d_amplitude, amplitude_floor, d_valid, n);
    CUDA_CHECK_LAST_ERROR("confidence_mask_kernel launch");
}

// ===========================================================================
// Stage 5 — flying_pixel_detect_kernel: THIS PROJECT'S ONE STENCIL KERNEL.
//
// Thread-to-data mapping: identical to every other kernel here (one thread
// per camera pixel `pix = row*w + col`), but each thread now GATHERS its
// (up to) 8 immediate neighbors' `d_depth`/`d_amplitude`/`d_confidence_valid`
// — a genuine 3x3 stencil, the first in this project's pipeline (kernels.cu
// file header contrasts this with 01.19, which needs none).
//
// TWO physically-motivated tests (kernels.cuh "Flying-pixel detection
// thresholds" derives both numerically; data/README.md "How the sample was
// tuned" records the precision/recall sweep that picked the two constants):
//   (a) depth-discontinuity test: the RANGE (max-min) of valid neighbor
//       depths exceeds `kFlyingDepthJumpM`. On its own this ALSO flags the
//       clean pixels sitting immediately on either side of a genuine depth
//       step (their neighbor sets span the step too) — expected, and why a
//       second, independent test is needed.
//   (b) amplitude-ratio test: THIS pixel's own amplitude, divided by the
//       strongest valid neighbor's amplitude, falls below
//       `kFlyingAmplitudeRatio`. A mixed-return pixel's phasor is the SUM
//       of two out-of-phase constituent phasors and is, by the triangle
//       inequality, generally WEAKER than either alone (kernels.cuh file
//       header) — a signature no CLEAN single-surface pixel shares, even
//       one sitting right next to a sharp step (its own amplitude tracks
//       only its own surface's albedo).
// A pixel is flagged FLYING only when BOTH tests fire — narrowing the
// depth-jump test's many neighbor-of-a-step false positives down to the
// pixels that are THEMSELVES a mixed return (measured precision/recall in
// main.cu's flying_pixel gate).
//
// Memory: up to 8 neighbor reads of `d_depth`/`d_amplitude`/
// `d_confidence_valid` PLUS the thread's own pixel — global memory only, no
// shared memory. At this project's problem size (19,200 pixels) a 3x3 halo
// exchange via shared-memory tiling would save a modest amount of redundant
// global traffic (each interior pixel's value is read up to 9 times across
// its neighbors) but adds real complexity (halo boundary handling); THEORY.md
// "The GPU mapping" makes the case that this kernel, like the rest of the
// pipeline, stays comfortably bandwidth-light at this size and is not worth
// the added complexity here — a genuine engineering judgment call, not an
// oversight (contrast with a real 1-20 MP sensor frame, where tiling would
// start to matter).
// ===========================================================================
__global__ void flying_pixel_detect_kernel(const float*         __restrict__ d_depth,
                                           const float*         __restrict__ d_amplitude,
                                           const unsigned char* __restrict__ d_confidence_valid,
                                           unsigned char* __restrict__ d_flying,
                                           int w, int h)
{
    const int pix = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = w * h;
    if (pix >= n) return;

    d_flying[pix] = 0;   // default: not flagged (early-outs below leave this as the answer)

    const int row = pix / w;
    const int col = pix - row * w;   // == pix % w, avoids a second integer divide (01.19's triangulate_kernel trick)

    // Gather the (up to) 8 immediate neighbors that are themselves
    // confidence-valid (Stage 4) — an invalid neighbor's depth/amplitude
    // carries no information (it may be an untrusted, near-random phase),
    // so it is simply excluded from the stencil rather than treated as a
    // (wrong) zero.
    float min_depth = 3.4e38f, max_depth = -3.4e38f, max_amp = 0.0f;
    int valid_neighbors = 0;
#pragma unroll
    for (int dr = -1; dr <= 1; ++dr) {
#pragma unroll
        for (int dc = -1; dc <= 1; ++dc) {
            if (dr == 0 && dc == 0) continue;           // skip self — this is a NEIGHBOR stencil
            const int r = row + dr, c = col + dc;
            if (r < 0 || r >= h || c < 0 || c >= w) continue;   // image-border guard
            const int npix = r * w + c;
            if (!d_confidence_valid[npix]) continue;
            const float nd = d_depth[npix];
            const float na = d_amplitude[npix];
            if (nd < min_depth) min_depth = nd;
            if (nd > max_depth) max_depth = nd;
            if (na > max_amp) max_amp = na;
            ++valid_neighbors;
        }
    }

    if (valid_neighbors < kFlyingMinValidNeighbors) return;   // too few neighbors to judge — leave unflagged
    if ((max_depth - min_depth) <= kFlyingDepthJumpM) return; // test (a): no local depth discontinuity
    if (max_amp <= 0.0f) return;                              // degenerate: no positive neighbor amplitude to compare against
    if ((d_amplitude[pix] / max_amp) >= kFlyingAmplitudeRatio) return;   // test (b): not amplitude-suppressed

    d_flying[pix] = 1;   // BOTH tests fired: flag this pixel as a flying (mixed-return) pixel
}

void launch_flying_pixel_detect(const float* d_depth, const float* d_amplitude,
                                const unsigned char* d_confidence_valid, unsigned char* d_flying, int w, int h)
{
    const int block = 256;
    const int n = w * h;
    flying_pixel_detect_kernel<<<grid_for(n, block), block>>>(d_depth, d_amplitude, d_confidence_valid, d_flying, w, h);
    CUDA_CHECK_LAST_ERROR("flying_pixel_detect_kernel launch");
}

// ===========================================================================
// Stage 6 — backproject_kernel: pinhole back-projection to a metric point
// cloud. One thread per camera pixel; a pure MAP again (Stage 5 was the one
// exception). Reuses 01.19/01.17's pixel-center camera-ray convention
// exactly: dx=(col+0.5-cx)/fx, dy=(row+0.5-cy)/fy, P = depth*(dx,dy,1) — no
// projector-plane geometry is needed here (unlike 01.19's Stage 4): a ToF
// camera measures a per-pixel RANGE directly along its own ray, not a
// correspondence to a second device, so back-projection is a single scalar
// multiply per axis, no ray/plane intersection at all.
// ===========================================================================
__global__ void backproject_kernel(const float*         __restrict__ d_depth,
                                   const unsigned char* __restrict__ d_final_valid,
                                   float*         __restrict__ d_xyz,
                                   int n)
{
    const int pix = blockIdx.x * blockDim.x + threadIdx.x;
    if (pix >= n) return;

    d_xyz[pix * 3 + 0] = 0.0f;
    d_xyz[pix * 3 + 1] = 0.0f;
    d_xyz[pix * 3 + 2] = 0.0f;
    if (!d_final_valid[pix]) return;

    const int row = pix / kCamW;
    const int col = pix - row * kCamW;
    const float dx = (static_cast<float>(col) + 0.5f - kCamCx) / kCamFx;
    const float dy = (static_cast<float>(row) + 0.5f - kCamCy) / kCamFy;
    const float z  = d_depth[pix];

    d_xyz[pix * 3 + 0] = dx * z;
    d_xyz[pix * 3 + 1] = dy * z;
    d_xyz[pix * 3 + 2] = z;
}

void launch_backproject(const float* d_depth, const unsigned char* d_final_valid, float* d_xyz, int n)
{
    const int block = 256;
    backproject_kernel<<<grid_for(n, block), block>>>(d_depth, d_final_valid, d_xyz, n);
    CUDA_CHECK_LAST_ERROR("backproject_kernel launch");
}
