// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 01.20
//                     (Time-of-flight raw processing: phase unwrapping,
//                     flying-pixel removal)
//
// WHY does a GPU repository ship a CPU implementation of everything?
// ------------------------------------------------------------------
// The correctness ORACLE (main.cu runs both and asserts agreement) and the
// TEACHING BASELINE (read this file, then kernels.cu, to see exactly what
// "one thread per pixel" changed — for five of six stages, almost nothing;
// see kernels.cu's file header for the one stencil exception). Plain
// C++17, no CUDA headers, no cleverness.
//
// Independence ruling (template's reference_cpu.cpp header, restated per
// 01.19's precedent — load-bearing; not re-derived here). Applied as
// follows in this project:
//   * Data-layout contracts (tap-stack indexing, all sensor/scene constants)
//     are single-sourced in kernels.cuh and shared.
//   * The ALGORITHMIC CORE of every stage below is written INDEPENDENTLY of
//     kernels.cu — separate loops, separate variable names, typed out fresh
//     from THEORY.md's math, not copy-pasted and s/__global__//.
//   * This project's INDEPENDENT-of-both-paths verification layer (the
//     third bullet of the ruling) is main.cu's set of GATES that compare
//     against scripts/make_synthetic.py's synthetic ground truth (a THIRD,
//     Python, codebase) — phase_extraction, offset_invariance,
//     aliasing_demo, unwrap_recovery, flying_pixel, the reconstruction
//     trio, and dark_cohort all check against Python-computed truth, not
//     against each other. A bug shared by BOTH this file and kernels.cu (a
//     wrong sign, a swapped tap index) would still be caught by those
//     gates even though the GPU-vs-CPU VERIFY lines above them would pass.
//
// Read this after: kernels.cu. Companion contract: kernels.cuh.
// ===========================================================================

#include "kernels.cuh"

#include <cmath>
#include <cstddef>

// ===========================================================================
// Stage 1 — extract_phase_amplitude_cpu.
//
// Per pixel: read the four correlation taps C0..C3 (tap-major layout,
// kernels.cuh), reconstruct the phase and modulation amplitude from the
// forward model C_k(phi) = A + B*cos(phi + k*pi/2) (kernels.cuh file
// header derives this). Expanding the four samples algebraically:
//   C0 = A + B*cos(phi)          C1 = A - B*sin(phi)
//   C2 = A - B*cos(phi)          C3 = A + B*sin(phi)
// so C3-C1 = 2*B*sin(phi) and C0-C2 = 2*B*cos(phi) — the ambient term A
// cancels EXACTLY in both differences (this project's offset_invariance
// gate exploits precisely this cancellation), leaving phi recoverable via
// atan2 and B (the confidence signal) via the vector length.
// ===========================================================================
void extract_phase_amplitude_cpu(const float* taps, float* phase, float* amplitude, int n)
{
    for (int pix = 0; pix < n; ++pix) {
        const float c0 = taps[0 * n + pix];
        const float c1 = taps[1 * n + pix];
        const float c2 = taps[2 * n + pix];
        const float c3 = taps[3 * n + pix];

        const float num = c3 - c1;   // = 2*B*sin(phi), ambient-free
        const float den = c0 - c2;   // = 2*B*cos(phi), ambient-free

        float phi = std::atan2(num, den);                 // in (-pi, pi]
        if (phi < 0.0f) phi += 6.28318530717958647692f;   // this project's [0,2pi) convention (kernels.cuh)

        phase[pix] = phi;
        amplitude[pix] = 0.5f * std::sqrt(num * num + den * den);   // = B
    }
}

// ===========================================================================
// Stage 2 — single_freq_depth_cpu: the naive, WRAPPED single-frequency
// depth estimate. depth = (phase / 2*pi) * ambiguity_range — the direct
// inversion of kernels.cuh's "distance(phi) = c*phi/(4*pi*f)" formula
// (ambiguity_range = c/(2*f) is passed in already computed, so this
// function need not know which frequency it was called for — mirrors
// kernels.cu's single_freq_depth_kernel exactly in spirit).
// ===========================================================================
void single_freq_depth_cpu(const float* phase, float ambiguity_range_m, float* depth, int n)
{
    const float two_pi = 6.28318530717958647692f;
    for (int pix = 0; pix < n; ++pix) {
        depth[pix] = (phase[pix] / two_pi) * ambiguity_range_m;
    }
}

// ===========================================================================
// Stage 3 — dual_freq_unwrap_cpu: the CRT-style integer-wrap consistency
// search (THEORY.md "The math" derives this in full; kernels.cu's kernel
// header restates the search for the GPU path).
//
// The fine channel's phase gives candidate depths z1(n1) = (phase1/2pi +
// n1) * kAmbig1M for each integer wrap n1 in [0, kMaxWraps1); the coarse
// channel similarly gives z2(n2). We search every (n1, n2) pair and keep
// the one whose two candidate depths agree most closely — physically, "the
// wrap hypothesis under which BOTH frequencies are describing the same
// point in space". The fine channel's depth at the WINNING n1 is reported
// (better raw precision per THEORY.md's noise-scaling derivation); n1
// itself is reported for the wrap_count-correctness gate.
// ===========================================================================
void dual_freq_unwrap_cpu(const float* phase1, const float* phase2, float* depth, int* wrap_count, int n)
{
    const float two_pi = 6.28318530717958647692f;
    for (int pix = 0; pix < n; ++pix) {
        const float frac1 = phase1[pix] / two_pi;   // in [0,1)
        const float frac2 = phase2[pix] / two_pi;   // in [0,1)

        float best_diff = 3.4e38f;
        float best_z1 = 0.0f;
        int best_n1 = 0;
        for (int n1 = 0; n1 < kMaxWraps1; ++n1) {
            const float z1 = (frac1 + static_cast<float>(n1)) * kAmbig1M;
            for (int n2 = 0; n2 < kMaxWraps2; ++n2) {
                const float z2 = (frac2 + static_cast<float>(n2)) * kAmbig2M;
                const float diff = std::fabs(z1 - z2);
                // Strict '<' (not '<='), matching kernels.cu's kernel bit-
                // for-bit, so ties resolve identically on both paths — a
                // deliberate detail that keeps this gate's GPU-vs-CPU
                // comparison exact rather than "usually exact".
                if (diff < best_diff) {
                    best_diff = diff;
                    best_z1 = z1;
                    best_n1 = n1;
                }
            }
        }
        depth[pix] = best_z1;
        wrap_count[pix] = best_n1;
    }
}

// ===========================================================================
// Stage 4 — confidence_mask_cpu: amplitude-floor threshold.
// ===========================================================================
void confidence_mask_cpu(const float* amplitude, float amplitude_floor, unsigned char* valid, int n)
{
    for (int pix = 0; pix < n; ++pix) {
        valid[pix] = (amplitude[pix] >= amplitude_floor) ? 1 : 0;
    }
}

// ===========================================================================
// Stage 5 — flying_pixel_detect_cpu: this project's one STENCIL stage,
// typed independently from kernels.cu's flying_pixel_detect_kernel (same
// two physically-motivated tests — kernels.cu's kernel header derives both
// in full; not re-derived here). w, h are the image dimensions (kCamW,
// kCamH in every real call, but kept as parameters so this function, like
// its GPU twin, does not hard-code the image shape).
// ===========================================================================
void flying_pixel_detect_cpu(const float* depth, const float* amplitude, const unsigned char* confidence_valid,
                             unsigned char* flying, int w, int h)
{
    const int n = w * h;
    for (int pix = 0; pix < n; ++pix) {
        flying[pix] = 0;

        const int row = pix / w;
        const int col = pix % w;

        float min_depth = 3.4e38f, max_depth = -3.4e38f, max_amp = 0.0f;
        int valid_neighbors = 0;
        for (int dr = -1; dr <= 1; ++dr) {
            for (int dc = -1; dc <= 1; ++dc) {
                if (dr == 0 && dc == 0) continue;   // skip self
                const int r = row + dr, c = col + dc;
                if (r < 0 || r >= h || c < 0 || c >= w) continue;   // image-border guard
                const int npix = r * w + c;
                if (!confidence_valid[npix]) continue;
                const float nd = depth[npix];
                const float na = amplitude[npix];
                if (nd < min_depth) min_depth = nd;
                if (nd > max_depth) max_depth = nd;
                if (na > max_amp) max_amp = na;
                ++valid_neighbors;
            }
        }

        if (valid_neighbors < kFlyingMinValidNeighbors) continue;        // too few neighbors to judge
        if ((max_depth - min_depth) <= kFlyingDepthJumpM) continue;      // test (a): no depth discontinuity
        if (max_amp <= 0.0f) continue;                                    // degenerate
        if ((amplitude[pix] / max_amp) >= kFlyingAmplitudeRatio) continue; // test (b): not amplitude-suppressed

        flying[pix] = 1;   // both tests fired
    }
}

// ===========================================================================
// Stage 6 — backproject_cpu: pinhole back-projection to a metric point
// cloud, identical convention to 01.19/01.17 (pixel-center camera rays).
// ===========================================================================
void backproject_cpu(const float* depth, const unsigned char* final_valid, float* xyz, int n)
{
    for (int pix = 0; pix < n; ++pix) {
        xyz[pix * 3 + 0] = 0.0f;
        xyz[pix * 3 + 1] = 0.0f;
        xyz[pix * 3 + 2] = 0.0f;
        if (!final_valid[pix]) continue;

        const int row = pix / kCamW;
        const int col = pix % kCamW;
        const float dx = (static_cast<float>(col) + 0.5f - kCamCx) / kCamFx;
        const float dy = (static_cast<float>(row) + 0.5f - kCamCy) / kCamFy;
        const float z  = depth[pix];

        xyz[pix * 3 + 0] = dx * z;
        xyz[pix * 3 + 1] = dy * z;
        xyz[pix * 3 + 2] = z;
    }
}
