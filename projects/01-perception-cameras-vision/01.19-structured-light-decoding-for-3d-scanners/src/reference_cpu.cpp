// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 01.19
//                     (Structured-light decoding: Gray code, phase shift,
//                     for 3D scanners)
//
// WHY does a GPU repository ship a CPU implementation of everything?
// ------------------------------------------------------------------
// The correctness ORACLE (main.cu runs both and asserts agreement) and the
// TEACHING BASELINE (read this file, then kernels.cu, to see exactly what
// parallelizing "one thread per pixel" changed — spoiler: almost nothing,
// because every stage here is already a pure per-pixel map; see kernels.cuh
// "Why no atomics anywhere"). Plain C++17, no CUDA headers, no cleverness.
//
// Independence ruling (template's reference_cpu.cpp header — load-bearing;
// read it once, it is not repeated per project). Applied here as follows:
//   * Data-layout contracts (pattern-stack indexing, state layouts, all
//     scanner/code constants) are single-sourced in kernels.cuh and shared.
//   * The ALGORITHMIC CORE of every stage below is written INDEPENDENTLY of
//     kernels.cu — separate loops, separate variable names, typed out fresh
//     from the same math in THEORY.md, not copy-pasted and s/__global__//.
//   * This project's INDEPENDENT-of-both-paths verification layer (per the
//     ruling's third bullet) is the set of gates that compare against the
//     synthetic ground truth (a THIRD, Python, codebase) rather than GPU-vs-
//     CPU agreement alone: gray_decode, hybrid_subpixel, the three
//     reconstruction gates, and dark_stripe honesty all check against
//     make_synthetic.py's truth files, not against each other — a shared
//     bug living in BOTH this file and kernels.cu (the 13.03 cautionary
//     tale the template cites) would still be caught by those gates.
//
// Read this after: kernels.cu. Companion contract: kernels.cuh.
// ===========================================================================

#include "kernels.cuh"

#include <cmath>
#include <cstddef>

// ===========================================================================
// Stage 1 — Gray-code decode.
//
// For each pixel: read its kGrayBits (direct, inverse) frame pairs, decide
// each bit by direct>inverse (the "captured + photometric inverse" trick —
// THEORY.md "The problem" explains why this beats a single fixed threshold:
// ambient light and surface albedo scale BOTH captures by the same factor,
// so the comparison cancels them, leaving only the sign of the illumination
// difference — robust to a surface being generally bright or dark).
// Then Gray-to-binary: b[0] = g[0]; b[k] = b[k-1] XOR g[k] — the standard
// prefix-XOR unreflection (THEORY.md "The math" proves this recovers the
// binary column whose Gray code is g).
// ===========================================================================
void gray_decode_cpu(const float* direct, const float* inverse, int* gray_col, int n)
{
    for (int pix = 0; pix < n; ++pix) {
        // Read kGrayBits bits, MSB (bit 0) first, and build the measured
        // Gray codeword g directly (no separate array needed: shifting a
        // running integer left and OR-ing in each new bit is the simplest
        // correct way to assemble a fixed-width code from MSB to LSB).
        int g = 0;
        for (int bit = 0; bit < kGrayBits; ++bit) {
            const float d = direct[bit * n + pix];
            const float v = inverse[bit * n + pix];
            const int   b = (d > v) ? 1 : 0;
            g = (g << 1) | b;
        }
        // Gray-to-binary: the k-th binary bit is the XOR of the Gray bit at
        // position k with EVERY Gray bit before it (equivalently, the
        // running-XOR recurrence below, which computes the same thing in
        // one pass instead of a bit-count-triangle of XORs).
        int binary = 0;
        int running = (g >> (kGrayBits - 1)) & 1;   // b[0] = g[0] (the MSB)
        binary = running;
        for (int k = 1; k < kGrayBits; ++k) {
            const int gk = (g >> (kGrayBits - 1 - k)) & 1;
            running ^= gk;                            // b[k] = b[k-1] XOR g[k]
            binary = (binary << 1) | running;
        }
        gray_col[pix] = binary;   // in [0, kProjCols) by construction (7 bits, 128 codes)
    }
}

// ===========================================================================
// Stage 2 — 4-step phase-shift decode.
//
// I_k = A + B*cos(phi - k*pi/2), k = 0..3 (THEORY.md derives this from the
// projector emitting a sinusoidal column intensity and the camera receiving
// albedo*intensity + ambient). Expanding the four samples algebraically:
//   I0 = A + B cos(phi),  I1 = A + B sin(phi),
//   I2 = A - B cos(phi),  I3 = A - B sin(phi)
// so  I1 - I3 = 2B sin(phi)  and  I0 - I2 = 2B cos(phi)  — the AMBIENT term
// A cancels EXACTLY in both differences (the "ambient/albedo cancellation"
// the catalog bullet asks to derive; phase_ambient_invariance gate exploits
// exactly this). phi = atan2(I1-I3, I0-I2) recovers the phase; the vector
// LENGTH of (I1-I3, I0-I2) is 2B, so B itself is the "modulation amplitude"
// used everywhere in this project as the per-pixel CONFIDENCE signal.
//
// DEVIATION FROM THE REPO DEFAULT ANGLE CONVENTION (documented, not hidden —
// kernels.cuh flags it too): CLAUDE.md's default wraps angles to (-pi, pi];
// this project instead wraps phi to [0, 2*pi) because it must map MONOTONE-
// ICALLY onto a projector column offset in [0, kPhasePeriodCols) for the
// hybrid-combine stage — a negative offset would have no physical meaning
// here. atan2 returns (-pi, pi]; a single "+= 2*pi if negative" fixes it up.
// ===========================================================================
void phase_decode_cpu(const float* phase, float* phase_out, float* confidence, int n)
{
    for (int pix = 0; pix < n; ++pix) {
        const float i0 = phase[0 * n + pix];
        const float i1 = phase[1 * n + pix];
        const float i2 = phase[2 * n + pix];
        const float i3 = phase[3 * n + pix];

        const float num = i1 - i3;   // = 2B sin(phi), ambient-free
        const float den = i0 - i2;   // = 2B cos(phi), ambient-free

        float phi = std::atan2(num, den);         // in (-pi, pi]
        if (phi < 0.0f) phi += 6.28318530717958647692f;  // project's [0,2pi) convention (see above)

        phase_out[pix] = phi;
        confidence[pix] = 0.5f * std::sqrt(num * num + den * den);  // = B, the modulation amplitude
    }
}

// ===========================================================================
// Stage 3 — hybrid combine: Gray resolves the PERIOD, phase resolves the
// SUB-PIXEL POSITION WITHIN it.
//
// frac = (phi / 2pi) * kPhasePeriodCols is the phase's own estimate of "how
// far into the current period am I", in [0, kPhasePeriodCols).
//
// The naive combination would take period = floor(gray_col / P) and output
// period*P + frac. That FAILS exactly at period boundaries: if the true
// position is, say, 7.98 (period 0, near the edge), Gray code might measure
// 8 (one period too far, off-by-one quantization) while phase correctly
// measures frac ~= 7.98 mod 8 ~= 7.98 -> floor(8/8) = period 1, producing
// 1*8 + 7.98 = 15.98 — nearly a full period (8 columns) of error from a
// Gray code answer that was off by just ONE integer!
//
// The fix — PHASE-GUIDED PERIOD SNAPPING (this project's documented rule,
// as the catalog bullet asks for): instead of trusting Gray code's period
// directly, recompute which period is CONSISTENT with the precise phase
// fraction: period = round((gray_col - frac) / P). Continuing the example:
// round((8 - 7.98)/8) = round(0.0025) = 0 — correctly snaps back to period
// 0, and the output 0*8 + 7.98 = 7.98 is right where phase said it should
// be. Gray code only has to get the period RIGHT TO THE NEAREST PERIOD, not
// exactly — exactly the robustness a coarse-but-absolute code should buy a
// fine-but-wrapped one (THEORY.md "The algorithm" walks the general case).
// ===========================================================================
void hybrid_combine_cpu(const int* gray_col, const float* phase, const float* confidence,
                        float confidence_floor, float* hybrid_col, unsigned char* valid, int n)
{
    for (int pix = 0; pix < n; ++pix) {
        if (confidence[pix] < confidence_floor) {
            // Low modulation: this pixel's phase (and hence any sub-pixel
            // refinement built on it) is untrustworthy — THEORY.md "Numerical
            // considerations" explains why atan2's conditioning collapses as
            // B -> 0. Mask it out rather than emit a plausible-looking wrong
            // number (the dark_stripe honesty gate checks this happens).
            hybrid_col[pix] = kInvalidColumnF;
            valid[pix] = 0;
            continue;
        }
        const float frac = (phase[pix] / 6.28318530717958647692f) * kPhasePeriodCols;
        const float period = std::round((static_cast<float>(gray_col[pix]) - frac) / kPhasePeriodCols);
        float hybrid = period * kPhasePeriodCols + frac;
        // Clamp to the projector's physical column range: a period-snap at
        // the very first/last period plus noise can walk a hair outside
        // [0, kProjCols) — clamping is a display/sanity guard, not a fix for
        // a wrong answer (the underlying period arithmetic already did the
        // real correction above).
        if (hybrid < 0.0f) hybrid = 0.0f;
        if (hybrid > static_cast<float>(kProjCols - 1)) hybrid = static_cast<float>(kProjCols - 1);
        hybrid_col[pix] = hybrid;
        valid[pix] = 1;
    }
}

// ===========================================================================
// Stage 4 — ray / projector-plane triangulation.
//
// A projector COLUMN (this project never encodes rows) does not pick out a
// single ray — it picks out every ray sharing that column, i.e. a PLANE.
// In the projector's own frame, that plane is Xp = m*Zp (m = (up-cxp)/fxp),
// i.e. all points with Xp - m*Zp = 0 — a plane through the projector's
// optical center with normal n_p = (1, 0, -m). Transformed into the camera
// frame via the projector's pose (P_proj = R_cp^T (P_cam - t_cp)), the same
// plane is  n_cam . (P_cam - t_cp) = 0  with  n_cam = R_cp * n_p  (THEORY.md
// "The math" derives this transform in full; 33.01 is cited there for the
// "thousands of independent tiny closed-form calculations, one per thread"
// GPU-mapping pattern this triangulation shares, even though — unlike
// 33.01's linear solves — a SINGLE plane/ray intersection has a closed form
// with no iteration or matrix inversion needed at all).
//
// Intersecting the camera ray P = t*d (d = (dx,dy,1), pixel-center
// convention) with that plane and solving for t:
//     n_cam . (t*d - t_cp) = 0   =>   t = (n_cam . t_cp) / (n_cam . d)
// NUMERICS NOTE (THEORY.md expands this): this formula is invariant to
// scaling n_cam by any nonzero constant (both numerator and denominator
// scale together and cancel) — so n_cam is deliberately NOT normalized: one
// fewer sqrt per pixel, for free, and a clean example of not paying for
// precision a formula does not need.
// ===========================================================================
void triangulate_cpu(const float* hybrid_col, const unsigned char* valid,
                     float* xyz, unsigned char* point_valid, int n)
{
    for (int pix = 0; pix < n; ++pix) {
        xyz[pix * 3 + 0] = 0.0f;
        xyz[pix * 3 + 1] = 0.0f;
        xyz[pix * 3 + 2] = 0.0f;
        point_valid[pix] = 0;
        if (!valid[pix]) continue;   // no trustworthy correspondence — see hybrid_combine_cpu

        const int row = pix / kCamW;
        const int col = pix % kCamW;
        // Pixel-center camera ray direction (kernels.cuh convention, shared
        // bit-for-bit with make_synthetic.py's ground-truth ray casting).
        const float dx = (static_cast<float>(col) + 0.5f - kCamCx) / kCamFx;
        const float dy = (static_cast<float>(row) + 0.5f - kCamCy) / kCamFy;

        // The projector-plane normal in the PROJECTOR's own frame: a fixed
        // column up picks out Xp = m*Zp, i.e. n_p = (1, 0, -m).
        const float m = (hybrid_col[pix] - kProjCx) / kProjFx;
        const float np0 = 1.0f, np1 = 0.0f, np2 = -m;

        // Rotate into the camera frame: n_cam = kRcp * n_p (kRcp row-major).
        // Written as a full 3x3 matrix-vector product (not simplified away
        // even though kRcp is identity here) so this code is correct for
        // ANY rig a learner substitutes (README Exercise).
        const float ncx = kRcp[0] * np0 + kRcp[1] * np1 + kRcp[2] * np2;
        const float ncy = kRcp[3] * np0 + kRcp[4] * np1 + kRcp[5] * np2;
        const float ncz = kRcp[6] * np0 + kRcp[7] * np1 + kRcp[8] * np2;

        // Ray/plane intersection: t = (n_cam . t_cp) / (n_cam . d).
        const float denom = ncx * dx + ncy * dy + ncz * 1.0f;
        // Denominator -> 0 means the camera ray is nearly PARALLEL to the
        // projector plane: a small-baseline-angle degeneracy (THEORY.md
        // "Numerical considerations" derives when this happens physically —
        // essentially, camera and projector "agreeing" on a direction).
        if (std::fabs(denom) < 1e-6f) continue;

        const float numer = ncx * kTcp[0] + ncy * kTcp[1] + ncz * kTcp[2];
        const float t = numer / denom;
        if (t <= 0.0f) continue;   // behind the camera: not a physical point

        xyz[pix * 3 + 0] = t * dx;
        xyz[pix * 3 + 1] = t * dy;
        xyz[pix * 3 + 2] = t;
        point_valid[pix] = 1;
    }
}

// ===========================================================================
// Stage 5 — the Gray-vs-plain-binary boundary stress test.
//
// See kernels.cuh's declaration comment for the noise-input design (pre-
// drawn on the host so GPU/CPU agreement does not depend on transcendental-
// function agreement) and kernels.cu's kernel header comment for the full
// derivation of WHY this 1-D blur+noise+threshold model reproduces the real
// mechanism (optical blur smearing two adjacent codewords together) behind
// Gray code's single-bit-adjacency advantage. This function is the plain
// twin: same math, sequential over samples, no CUDA.
// ===========================================================================
static int gray_code_of(int col) { return col ^ (col >> 1); }

// ideal_bit_gray/binary: the NOISE-FREE illumination of bit `bit` (0=MSB)
// of column `col`'s Gray code / plain binary code. Private to this file
// (ordinary host helpers, not a device/host-shared helper), mirroring the
// private helpers kernels.cu keeps on its own side — the independence
// ruling only governs the CPU-vs-GPU boundary, not helpers within one file.
static int ideal_bit_gray(int col, int bit)
{
    const int g = gray_code_of(col);
    return (g >> (kGrayBits - 1 - bit)) & 1;
}
static int ideal_bit_binary(int col, int bit)
{
    return (col >> (kGrayBits - 1 - bit)) & 1;
}

void boundary_stress_cpu(const float* true_x, const float* noise,
                         int* decoded_gray, int* decoded_binary, int n)
{
    for (int i = 0; i < n; ++i) {
        const float x = true_x[i];
        const int   c0 = static_cast<int>(std::floor(x));
        const float frac = x - static_cast<float>(c0);
        const int   c1 = c0 + 1;   // the boundary's far side; both codes are only
                                   // ever defined by BLENDING these two neighbors
        const float* my_noise = noise + static_cast<size_t>(i) * (2 * kGrayBits);

        // Gray-path bits: analog = lerp(ideal(c0), ideal(c1), frac) + noise,
        // thresholded at 0.5. Gray code differs between c0 and c1 in EXACTLY
        // one bit (THEORY.md proves this), so at most one bit here is ever
        // truly "ambiguous" (its ideal value near 0.5); the rest sit near 0
        // or 1 on BOTH sides of the boundary and survive the noise easily.
        int g = 0;
        for (int bit = 0; bit < kGrayBits; ++bit) {
            const float ideal = (1.0f - frac) * static_cast<float>(ideal_bit_gray(c0, bit))
                               + frac          * static_cast<float>(ideal_bit_gray(c1, bit));
            const float noisy = ideal + my_noise[bit];
            const int   b = (noisy > 0.5f) ? 1 : 0;
            g = (g << 1) | b;
        }
        int binary_from_gray = 0;
        int running = (g >> (kGrayBits - 1)) & 1;
        binary_from_gray = running;
        for (int k = 1; k < kGrayBits; ++k) {
            const int gk = (g >> (kGrayBits - 1 - k)) & 1;
            running ^= gk;
            binary_from_gray = (binary_from_gray << 1) | running;
        }
        decoded_gray[i] = binary_from_gray;

        // Binary-path bits: SAME blur+noise+threshold recipe, but plain
        // reflected binary can differ between c0 and c1 in UP TO kGrayBits
        // bits at once (worst case: the MSB boundary, e.g. 63 -> 64 flips
        // all 7) — every one of those bits is simultaneously near-0.5 and
        // simultaneously vulnerable to the SAME noise draw pattern this
        // Gray-path loop just weathered with only one bit at risk.
        int b_val = 0;
        for (int bit = 0; bit < kGrayBits; ++bit) {
            const float ideal = (1.0f - frac) * static_cast<float>(ideal_bit_binary(c0, bit))
                               + frac          * static_cast<float>(ideal_bit_binary(c1, bit));
            const float noisy = ideal + my_noise[kGrayBits + bit];
            const int   b = (noisy > 0.5f) ? 1 : 0;
            b_val = (b_val << 1) | b;
        }
        decoded_binary[i] = b_val;
    }
}
