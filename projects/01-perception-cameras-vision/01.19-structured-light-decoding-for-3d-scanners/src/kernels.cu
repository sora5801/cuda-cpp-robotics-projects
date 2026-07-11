// ===========================================================================
// kernels.cu — GPU kernels for project 01.19
//              Structured-light decoding (Gray code, phase shift) for 3D
//              scanners (Gray-code + phase-shift HYBRID scanner)
//
// Role in the project
// -------------------
// Five kernels, one per pipeline stage, EVERY one a pure per-pixel (or per-
// sample) MAP — no kernel here ever reads or writes another thread's output
// (kernels.cuh "Why no atomics anywhere" explains why that is possible for
// structured light specifically). All five share the pattern-stack layout,
// scanner geometry, and code parameters fixed in kernels.cuh; none of that
// is repeated here beyond a one-line reminder at each use.
//
// What is NEW here beyond 08.01/33.01/07.09 (the repo's other per-thread-
// independent-problem flagships):
//   * the PATTERN DIMENSION as the kernel's own INNER loop (kernels.cuh
//     "Pattern-stack memory layout" argues the coalescing case for the
//     [pattern][pixel] layout that makes this loop's every iteration one
//     coalesced 128-byte transaction per warp — the same lesson 08.01's
//     transposed noise array teaches, applied to images instead of
//     rollouts);
//   * a genuine two-STAGE decode (Gray, then phase) that only becomes a
//     3-D point on a THIRD stage (triangulate) — three kernels chained by
//     small per-pixel scalars (an int column, a float phase+confidence, a
//     float sub-pixel column) rather than one monolithic kernel, so each
//     stage is independently readable, independently testable against
//     kernels.cuh's independent-gate philosophy, AND independently
//     profilable;
//   * a kernel (boundary_stress_kernel) whose "problem" is not a camera
//     pixel at all but a synthetic 1-D probe — the same GPU mapping
//     (thousands of independent tiny computations, one per thread) applied
//     to a designed EXPERIMENT rather than to real sensor data.
//
// Read this after: kernels.cuh. Companion oracle: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"      // CUDA_CHECK_LAST_ERROR (§6.1 rule 7)

#include <cstdio>
#include <cstdlib>

// Shared launch geometry for every kernel below: 256 threads/block is the
// repo default (warp multiple, good occupancy on sm_75..sm_89 without
// starving the register file); grid = ceil(n/block) covers every element
// exactly once (n is at most kNPix = 30000 or kBoundarySamples = 20000 here,
// comfortably under any grid-dimension limit, so no grid-stride loop is
// needed — contrast with the template's SAXPY placeholder, which chose a
// grid-stride loop specifically to handle arbitrarily large n).
static inline int grid_for(int n, int block) { return (n + block - 1) / block; }

// ===========================================================================
// Stage 1 — Gray-code decode (device helpers + kernel).
//
// gray_code_of / decode_gray_to_binary are __device__-only and are also
// reused, unmodified, by boundary_stress_kernel below (Stage 5) — sharing a
// device helper WITHIN kernels.cu is unrelated to the reference_cpu.cpp
// independence ruling (which only governs the CPU/GPU boundary; see that
// file's header), and here it is exactly the right call: both kernels need
// the IDENTICAL Gray encode/decode bit manipulation, and duplicating it
// inside this one file would only invite the two copies to drift.
// ===========================================================================
__device__ __forceinline__ int gray_code_of(int col)
{
    return col ^ (col >> 1);          // the standard binary -> Gray map
}

// Gray-to-binary: prefix-XOR unreflection, MSB (bit 0) first. Same recurrence
// as reference_cpu.cpp's gray_decode_cpu, typed independently there — see
// this project's independence ruling for why that duplication is deliberate.
__device__ __forceinline__ int decode_gray_to_binary(int g)
{
    int running = (g >> (kGrayBits - 1)) & 1;
    int binary = running;
#pragma unroll
    for (int k = 1; k < kGrayBits; ++k) {
        const int gk = (g >> (kGrayBits - 1 - k)) & 1;
        running ^= gk;
        binary = (binary << 1) | running;
    }
    return binary;
}

// ---------------------------------------------------------------------------
// gray_decode_kernel — one thread per camera pixel.
//
// Thread-to-data mapping: pix = blockIdx.x*blockDim.x + threadIdx.x owns
// camera pixel `pix` (row-major, pix = row*kCamW + col).
//
// Memory per thread: reads kGrayBits*2 floats (direct+inverse, one pattern
// at a time — the INNER loop over `bit` is the pattern dimension; every
// warp's threads are on CONSECUTIVE pixels, so at fixed `bit` the read
// d_direct[bit*n + pix] is one coalesced 128-byte transaction per warp —
// kernels.cuh "Pattern-stack memory layout"), writes one int. No shared
// memory (nothing is reused across threads), no atomics (kernels.cuh "Why
// no atomics anywhere").
// ---------------------------------------------------------------------------
__global__ void gray_decode_kernel(const float* __restrict__ d_direct,
                                   const float* __restrict__ d_inverse,
                                   int*         __restrict__ d_gray_col,
                                   int n)
{
    const int pix = blockIdx.x * blockDim.x + threadIdx.x;
    if (pix >= n) return;

    // Assemble the measured Gray codeword bit by bit: direct>inverse cancels
    // ambient+albedo (kernels.cuh Stage 1 doc), leaving only illumination
    // sign — robust without any per-pixel calibrated threshold.
    int g = 0;
#pragma unroll
    for (int bit = 0; bit < kGrayBits; ++bit) {
        const float d = d_direct[bit * n + pix];
        const float v = d_inverse[bit * n + pix];
        const int   b = (d > v) ? 1 : 0;
        g = (g << 1) | b;
    }
    d_gray_col[pix] = decode_gray_to_binary(g);
}

void launch_gray_decode(const float* d_direct, const float* d_inverse, int* d_gray_col, int n)
{
    const int block = 256;
    gray_decode_kernel<<<grid_for(n, block), block>>>(d_direct, d_inverse, d_gray_col, n);
    CUDA_CHECK_LAST_ERROR("gray_decode_kernel launch");
}

// ===========================================================================
// Stage 2 — 4-step phase-shift decode. One thread per camera pixel.
//
// Same coalescing argument as Stage 1: the four phase-step reads happen at
// FIXED, compile-time-known offsets (0,1,2,3)*n — not even a runtime loop —
// so all four are back-to-back coalesced transactions with no branching.
// atan2f/sqrtf are single-instruction-throughput transcendentals on sm_75+;
// this kernel is bandwidth-bound (4 reads, 2 writes per pixel) same as
// Stage 1, not compute-bound — THEORY.md "The GPU mapping" makes this case
// quantitatively.
// ===========================================================================
__global__ void phase_decode_kernel(const float* __restrict__ d_phase,
                                    float*       __restrict__ d_phase_out,
                                    float*       __restrict__ d_confidence,
                                    int n)
{
    const int pix = blockIdx.x * blockDim.x + threadIdx.x;
    if (pix >= n) return;

    const float i0 = d_phase[0 * n + pix];
    const float i1 = d_phase[1 * n + pix];
    const float i2 = d_phase[2 * n + pix];
    const float i3 = d_phase[3 * n + pix];

    const float num = i1 - i3;   // = 2B sin(phi) — ambient A cancels exactly
    const float den = i0 - i2;   // = 2B cos(phi) — same cancellation

    // Precise atan2f/sqrtf (not the __-prefixed fast intrinsics): this
    // kernel's whole purpose is a geometrically meaningful angle and a
    // confidence used for a hard PASS/FAIL threshold downstream — trading
    // accuracy for the intrinsics' few-ULP speedup would buy nothing here
    // (bandwidth-bound, per the header comment) and could nudge borderline
    // confidence values across kDefaultConfidenceFloor (THEORY.md
    // "Numerical considerations").
    float phi = atan2f(num, den);                       // (-pi, pi]
    if (phi < 0.0f) phi += 6.28318530717958647692f;     // project's [0,2pi) convention (kernels.cuh)

    d_phase_out[pix] = phi;
    d_confidence[pix] = 0.5f * sqrtf(num * num + den * den);   // = B, the confidence signal
}

void launch_phase_decode(const float* d_phase, float* d_phase_out, float* d_confidence, int n)
{
    const int block = 256;
    phase_decode_kernel<<<grid_for(n, block), block>>>(d_phase, d_phase_out, d_confidence, n);
    CUDA_CHECK_LAST_ERROR("phase_decode_kernel launch");
}

// ===========================================================================
// Stage 3 — hybrid combine (Gray period + phase-guided period snap).
// One thread per camera pixel; see kernels.cuh / reference_cpu.cpp for the
// full derivation of the snapping rule — this kernel is its direct
// translation, unrolled into scalar float ops with no branching beyond the
// confidence gate (a single divergent `if` per thread; on sm_75+ this costs
// at most one extra instruction-issue slot per warp, negligible next to the
// bandwidth cost of the four stages combined).
// ===========================================================================
__global__ void hybrid_combine_kernel(const int*   __restrict__ d_gray_col,
                                      const float* __restrict__ d_phase,
                                      const float* __restrict__ d_confidence,
                                      float confidence_floor,
                                      float*         __restrict__ d_hybrid_col,
                                      unsigned char* __restrict__ d_valid,
                                      int n)
{
    const int pix = blockIdx.x * blockDim.x + threadIdx.x;
    if (pix >= n) return;

    if (d_confidence[pix] < confidence_floor) {
        // Untrusted phase (THEORY.md "Numerical considerations": atan2's
        // conditioning collapses as B -> 0) — mask, do not guess.
        d_hybrid_col[pix] = kInvalidColumnF;
        d_valid[pix] = 0;
        return;
    }

    const float frac = (d_phase[pix] / 6.28318530717958647692f) * kPhasePeriodCols;
    const float period = roundf((static_cast<float>(d_gray_col[pix]) - frac) / kPhasePeriodCols);
    float hybrid = period * kPhasePeriodCols + frac;
    hybrid = fminf(fmaxf(hybrid, 0.0f), static_cast<float>(kProjCols - 1));  // sanity clamp, see reference_cpu.cpp

    d_hybrid_col[pix] = hybrid;
    d_valid[pix] = 1;
}

void launch_hybrid_combine(const int* d_gray_col, const float* d_phase, const float* d_confidence,
                           float confidence_floor, float* d_hybrid_col, unsigned char* d_valid, int n)
{
    const int block = 256;
    hybrid_combine_kernel<<<grid_for(n, block), block>>>(
        d_gray_col, d_phase, d_confidence, confidence_floor, d_hybrid_col, d_valid, n);
    CUDA_CHECK_LAST_ERROR("hybrid_combine_kernel launch");
}

// ===========================================================================
// Stage 4 — ray / projector-plane triangulation. One thread per camera
// pixel. See reference_cpu.cpp for the full plane-intersection derivation
// (identical here, independently typed — the twin ruling).
// ===========================================================================
__global__ void triangulate_kernel(const float*         __restrict__ d_hybrid_col,
                                   const unsigned char* __restrict__ d_valid,
                                   float*         __restrict__ d_xyz,
                                   unsigned char* __restrict__ d_point_valid,
                                   int n)
{
    const int pix = blockIdx.x * blockDim.x + threadIdx.x;
    if (pix >= n) return;

    d_xyz[pix * 3 + 0] = 0.0f;
    d_xyz[pix * 3 + 1] = 0.0f;
    d_xyz[pix * 3 + 2] = 0.0f;
    d_point_valid[pix] = 0;
    if (!d_valid[pix]) return;

    const int row = pix / kCamW;
    const int col = pix - row * kCamW;   // == pix % kCamW, but avoids a second integer divide
    const float dx = (static_cast<float>(col) + 0.5f - kCamCx) / kCamFx;
    const float dy = (static_cast<float>(row) + 0.5f - kCamCy) / kCamFy;

    const float m = (d_hybrid_col[pix] - kProjCx) / kProjFx;
    const float np0 = 1.0f, np1 = 0.0f, np2 = -m;

    // n_cam = kRcp * n_p (row-major 3x3 * 3x1) — kRcp/kTcp live in constant
    // memory implicitly (they are compile-time __constant__-folded literals
    // from kernels.cuh, not a runtime __constant__ array; see THEORY.md "The
    // GPU mapping" for why that distinction matters at this problem size).
    const float ncx = kRcp[0] * np0 + kRcp[1] * np1 + kRcp[2] * np2;
    const float ncy = kRcp[3] * np0 + kRcp[4] * np1 + kRcp[5] * np2;
    const float ncz = kRcp[6] * np0 + kRcp[7] * np1 + kRcp[8] * np2;

    const float denom = ncx * dx + ncy * dy + ncz * 1.0f;
    if (fabsf(denom) < 1e-6f) return;   // near-parallel ray/plane — degenerate (THEORY.md)

    const float numer = ncx * kTcp[0] + ncy * kTcp[1] + ncz * kTcp[2];
    const float t = numer / denom;
    if (t <= 0.0f) return;

    d_xyz[pix * 3 + 0] = t * dx;
    d_xyz[pix * 3 + 1] = t * dy;
    d_xyz[pix * 3 + 2] = t;
    d_point_valid[pix] = 1;
}

void launch_triangulate(const float* d_hybrid_col, const unsigned char* d_valid,
                        float* d_xyz, unsigned char* d_point_valid, int n)
{
    const int block = 256;
    triangulate_kernel<<<grid_for(n, block), block>>>(d_hybrid_col, d_valid, d_xyz, d_point_valid, n);
    CUDA_CHECK_LAST_ERROR("triangulate_kernel launch");
}

// ===========================================================================
// Stage 5 — the Gray-vs-plain-binary boundary stress test.
//
// THE MECHANISM THIS REPRODUCES: a real optical system BLURS two physically
// adjacent projector columns together at their shared boundary (finite lens
// PSF / projector defocus — the same optics THEORY.md's "The problem"
// section teaches). A camera pixel sitting exactly on such a boundary sees,
// for EVERY bit plane, an intensity that is a BLEND of that bit's ideal
// value at column c0 and at column c0+1 — modeled below as a linear
// interpolation (box-blur-of-radius-0.5, the simplest honest model of "this
// pixel straddles exactly one boundary"), plus ordinary sensor noise, then
// thresholded exactly as Stage 1 thresholds a real capture.
//
// WHY THIS IS A FAIR (non-cherry-picked) MEASUREMENT: `d_true_x` is drawn
// UNIFORMLY over [0, kProjCols-1) by main.cu (xorshift32, seeded), so most
// probes land mid-cell (both codes trivially correct) and only the fraction
// genuinely near a boundary are at risk — exactly the mix a real scanline
// would present. The PROOF this is exploiting (THEORY.md "The math"):
// consecutive integers' GRAY codes differ in exactly one bit; consecutive
// integers' plain BINARY codes can differ in up to kGrayBits bits at once
// (worst case: the MSB boundary, e.g. 63 -> 64 flips all 7 — 0111111 vs
// 1000000). At a boundary, Gray code therefore puts AT MOST ONE bit at risk
// of corruption; plain binary can put ALL of them at risk simultaneously —
// this kernel measures exactly that gap, on the same hardware, same noise
// MODEL, same threshold rule, differing only in which bit-plane assignment
// (Gray vs binary) the projector used.
//
// gray_code_of / decode_gray_to_binary are the SAME __device__ helpers
// Stage 1 uses above (see that section's header for why sharing them here
// is fine and not a twin-independence violation).
// ===========================================================================
__device__ __forceinline__ int ideal_bit_gray(int col, int bit)
{
    const int g = gray_code_of(col);
    return (g >> (kGrayBits - 1 - bit)) & 1;
}
__device__ __forceinline__ int ideal_bit_binary(int col, int bit)
{
    return (col >> (kGrayBits - 1 - bit)) & 1;
}

__global__ void boundary_stress_kernel(const float* __restrict__ d_true_x,
                                       const float* __restrict__ d_noise,
                                       int* __restrict__ d_decoded_gray,
                                       int* __restrict__ d_decoded_binary,
                                       int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float x = d_true_x[i];
    const int   c0 = static_cast<int>(floorf(x));
    const float frac = x - static_cast<float>(c0);
    const int   c1 = c0 + 1;
    const float* my_noise = d_noise + static_cast<size_t>(i) * (2 * kGrayBits);

    int g = 0;
#pragma unroll
    for (int bit = 0; bit < kGrayBits; ++bit) {
        const float ideal = (1.0f - frac) * static_cast<float>(ideal_bit_gray(c0, bit))
                           + frac          * static_cast<float>(ideal_bit_gray(c1, bit));
        const float noisy = ideal + my_noise[bit];
        const int   b = (noisy > 0.5f) ? 1 : 0;
        g = (g << 1) | b;
    }
    d_decoded_gray[i] = decode_gray_to_binary(g);

    int b_val = 0;
#pragma unroll
    for (int bit = 0; bit < kGrayBits; ++bit) {
        const float ideal = (1.0f - frac) * static_cast<float>(ideal_bit_binary(c0, bit))
                           + frac          * static_cast<float>(ideal_bit_binary(c1, bit));
        const float noisy = ideal + my_noise[kGrayBits + bit];
        const int   b = (noisy > 0.5f) ? 1 : 0;
        b_val = (b_val << 1) | b;
    }
    d_decoded_binary[i] = b_val;
}

void launch_boundary_stress(const float* d_true_x, const float* d_noise,
                            int* d_decoded_gray, int* d_decoded_binary, int n)
{
    const int block = 256;
    boundary_stress_kernel<<<grid_for(n, block), block>>>(d_true_x, d_noise, d_decoded_gray, d_decoded_binary, n);
    CUDA_CHECK_LAST_ERROR("boundary_stress_kernel launch");
}
