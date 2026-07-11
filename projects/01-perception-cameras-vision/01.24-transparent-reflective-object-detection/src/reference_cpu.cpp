// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 01.24
//                     Transparent/reflective object detection via
//                     polarization imaging
//
// WHY does a GPU repository ship a CPU implementation of everything?
// (CLAUDE.md §5 — restated from the template): this file is BOTH the
// correctness oracle main.cu's VERIFY step compares the GPU path against,
// AND the teaching baseline that makes "what did parallelizing this change"
// legible.
//
// INDEPENDENCE RULING applied throughout this file (kernels.cuh's header,
// restated here per the template's convention of stating it in both
// places): shared DATA-LAYOUT contracts (canvas geometry, the DoFP phase/
// channel map, PhaseSample's coordinate footprint, object rectangles, the
// Fresnel closed-form helper) live ONCE in kernels.cuh and are used as-is —
// duplicating an index formula would not be independence, only a second
// hiding place for the same bug. The ALGORITHMIC core of every pipeline
// stage below — the demosaic 4-corner blend, the Stokes/DoLP/AoLP pointwise
// formulas, the Malus residual, morphological open, and (most
// consequentially) connected-component labeling — is written A SECOND TIME
// here, independently of kernels.cu:
//   * demosaic/Stokes/DoLP/AoLP/Malus residual/threshold/morphology: the
//     SAME textbook formulas, but re-typed from scratch in a separate
//     function body, the 01.22 precedent (naive_inverse_cpu/wiener_cpu are
//     that project's analogous "simple formula, still independently typed
//     twice" stages).
//   * connected-component labeling: a GENUINELY DIFFERENT algorithm —
//     classic Rosenfeld two-pass UNION-FIND, not the GPU's iterative label-
//     propagation sweep — the STRONGEST form of independence this repo
//     practices (01.21's precedent, cited by name below), because a bug
//     shared by "the same algorithm typed twice" cannot hide behind two
//     unrelated algorithms that must still agree.
// Independent GATEs in main.cu (never routed through this file OR
// kernels.cu) compare against ground truth and a closed-form physics
// prediction — see main.cu's file header for that third leg of the ruling.
//
// Rules for this file: plain C++17, no CUDA headers (kernels.cuh's
// __CUDACC__ fence hides every __global__ declaration from cl.exe), no
// hand-vectorization, no OpenMP, no cleverness — clarity beats speed here,
// always, per the template.
//
// Read this after: kernels.cu — then compare the two side by side.
// ===========================================================================

#include <cmath>
#include <vector>
#include <algorithm>   // std::min/max

#include "kernels.cuh"

// ===========================================================================
// STAGE 1 — demosaic (CPU twin of demosaic_polarization_kernel). Same
// PhaseSample footprint arithmetic (shared data contract, kernels.cuh), but
// this function's 4-corner blend is its OWN independent expression — a
// small helper function called per corner-set instead of the kernel's
// inlined arithmetic, so a transcription slip in either file's algebra
// would show up as a genuine VERIFY(demosaic) disagreement, not just a
// re-derivation of the identical typo.
// ===========================================================================
namespace {
float bilinear_blend(float v00, float v10, float v01, float v11, float wx, float wy)
{
    const float top = v00 + (v10 - v00) * wx;      // interpolate along x at y0
    const float bot = v01 + (v11 - v01) * wx;      // interpolate along x at y1
    return top + (bot - top) * wy;                  // interpolate along y between the two
}
} // namespace

void demosaic_polarization_cpu(const float* mosaic, float* channels4, int W, int H)
{
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int i = y * W + x;
            const int px = x & 1, py = y & 1;
            const int own_c = dofp_channel_for_phase(px, py);
            for (int c = 0; c < kNumChannels; ++c) {
                float v;
                if (c == own_c) {
                    v = mosaic[i];
                } else {
                    int tpx = 0, tpy = 0;
                    dofp_phase_for_channel(c, tpx, tpy);
                    const PhaseSample s = phase_sample_at(x, y, tpx, tpy, W, H);
                    v = bilinear_blend(mosaic[s.y0 * W + s.x0], mosaic[s.y0 * W + s.x1],
                                       mosaic[s.y1 * W + s.x0], mosaic[s.y1 * W + s.x1],
                                       s.wx, s.wy);
                }
                channels4[i * kNumChannels + c] = v;
            }
        }
    }
}

// ===========================================================================
// STAGE 2/3/4 — Stokes / DoLP-AoLP / Malus residual CPU twins. Independently
// re-typed pointwise formulas (kernels.cuh Section 2/THEORY.md "The math"
// derive them; kernels.cu's stage-2/3/4 kernels state the same derivation
// from the GPU side).
// ===========================================================================
void stokes_cpu(const float* channels4, float* s0, float* s1, float* s2, int n)
{
    for (int i = 0; i < n; ++i) {
        const float I0   = channels4[i * kNumChannels + 0];
        const float I45  = channels4[i * kNumChannels + 1];
        const float I90  = channels4[i * kNumChannels + 2];
        const float I135 = channels4[i * kNumChannels + 3];
        s0[i] = 0.5f * (I0 + I45 + I90 + I135);
        s1[i] = I0 - I90;
        s2[i] = I45 - I135;
    }
}

void dolp_aolp_cpu(const float* s0, const float* s1, const float* s2, float* dolp, float* aolp_rad, int n)
{
    for (int i = 0; i < n; ++i) {
        const float S0 = s0[i], S1 = s1[i], S2 = s2[i];
        const float mag = std::sqrt(S1 * S1 + S2 * S2);
        const float s0_safe = std::max(S0, 1.0e-3f);
        dolp[i] = mag / s0_safe;
        float a = 0.5f * std::atan2(S2, S1);
        if (a < 0.0f) a += kPi;
        aolp_rad[i] = a;
    }
}

void malus_residual_cpu(const float* channels4, float* residual, int n)
{
    for (int i = 0; i < n; ++i) {
        const float I0   = channels4[i * kNumChannels + 0];
        const float I45  = channels4[i * kNumChannels + 1];
        const float I90  = channels4[i * kNumChannels + 2];
        const float I135 = channels4[i * kNumChannels + 3];
        residual[i] = (I0 + I90) - (I45 + I135);
    }
}

// ===========================================================================
// STAGE 5 — detection CPU twins.
//
// VERIFY-ISOLATION NOTE (the 01.22 "IBP CPU twin seeded from the GPU's
// shift-and-add result" pattern, applied here): main.cu feeds these
// functions the GPU'S OWN dolp/intensity-contrast arrays (already verified
// to agree with THIS file's own stokes_cpu/dolp_aolp_cpu within float
// tolerance one stage earlier) rather than re-deriving that signal from
// scratch a second time. That isolates VERIFY(detection_*) to test ONLY the
// threshold/morphology/CCL/filter algorithm's agreement — deterministic
// integer/boolean operations on an IDENTICAL float input are expected to be
// BIT-EXACT, not merely close, and main.cu gates on exactly that.
// ===========================================================================
void abs_diff_scalar_cpu(const float* signal, float ref_scalar, float* out, int n)
{
    for (int i = 0; i < n; ++i) out[i] = std::fabs(signal[i] - ref_scalar);
}

void threshold_cpu(const float* signal, float thresh, uint8_t* mask_out, int n)
{
    for (int i = 0; i < n; ++i) mask_out[i] = (signal[i] >= thresh) ? 1u : 0u;
}

// erode3x3_cpu / dilate3x3_cpu / morphological_open_cpu — independently
// re-typed twin of kernels.cu's stencil pair (01.21's cited precedent).
namespace {
void erode3x3_cpu(const uint8_t* in, int W, int H, uint8_t* out)
{
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            uint8_t v = 1u;
            for (int dy = -1; dy <= 1 && v; ++dy) {
                for (int dx = -1; dx <= 1 && v; ++dx) {
                    const int nx = x + dx, ny = y + dy;
                    const uint8_t nb = (nx < 0 || nx >= W || ny < 0 || ny >= H) ? 0u : in[ny * W + nx];
                    v = static_cast<uint8_t>(v & nb);
                }
            }
            out[y * W + x] = v;
        }
    }
}
void dilate3x3_cpu(const uint8_t* in, int W, int H, uint8_t* out)
{
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            uint8_t v = 0u;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nx = x + dx, ny = y + dy;
                    const uint8_t nb = (nx < 0 || nx >= W || ny < 0 || ny >= H) ? 0u : in[ny * W + nx];
                    v = static_cast<uint8_t>(v | nb);
                }
            }
            out[y * W + x] = v;
        }
    }
}
} // namespace

void morphological_open_cpu(uint8_t* mask_inout, int W, int H)
{
    std::vector<uint8_t> scratch(static_cast<size_t>(W) * H);
    erode3x3_cpu(mask_inout, W, H, scratch.data());
    dilate3x3_cpu(scratch.data(), W, H, mask_inout);
}

// ---------------------------------------------------------------------------
// connected_components_cpu — classic Rosenfeld two-pass UNION-FIND (01.21's
// cited precedent, itself citing 01.06's uf_find/uf_union_toward_smaller,
// re-typed fresh here): pass 1 unions every foreground pixel with its WEST
// and NORTH foreground neighbors (EAST/SOUTH get swept up when THOSE pixels
// run their own west/north union — the standard two-neighbor Rosenfeld
// scan); pass 2 resolves every foreground pixel to its component's
// canonical root. "attach the larger root under the smaller" makes every
// component's root converge to its MINIMUM linear pixel index — the exact
// canonical-label convention kernels.cu's ccl_propagate_sweep_kernel
// independently converges to (that kernel's own header proves this), which
// is WHY these two structurally unrelated algorithms can be held to a
// bit-exact tolerance in main.cu.
// ---------------------------------------------------------------------------
namespace {
int uf_find(std::vector<int>& parent, int x)
{
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];   // path-halving: point at grandparent while walking to the root
        x = parent[x];
    }
    return x;
}
void uf_union_toward_smaller(std::vector<int>& parent, int a, int b)
{
    a = uf_find(parent, a);
    b = uf_find(parent, b);
    if (a == b) return;
    if (a < b) parent[b] = a; else parent[a] = b;
}
} // namespace

void connected_components_cpu(const uint8_t* mask, int* label, int W, int H)
{
    std::vector<int> parent(static_cast<size_t>(W) * H);
    for (int i = 0; i < W * H; ++i) parent[i] = i;

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int i = y * W + x;
            if (!mask[i]) continue;
            if (x > 0 && mask[i - 1]) uf_union_toward_smaller(parent, i, i - 1);
            if (y > 0 && mask[i - W]) uf_union_toward_smaller(parent, i, i - W);
        }
    }
    for (int i = 0; i < W * H; ++i)
        label[i] = mask[i] ? uf_find(parent, i) : -1;   // -1 = kernels.cu's ccl_init_kernel's "no label" convention
}

// component_size_filter_cpu — the CPU twin of launch_component_size_filter:
// a direct sequential count-then-filter (no atomics needed for a single
// thread — 01.06/01.21's identical simplification for their own GPU
// atomic-scatter counterparts, cited).
void component_size_filter_cpu(const uint8_t* mask_in, const int* label, int min_size_px,
                               uint8_t* mask_out, int n)
{
    std::vector<int> size(static_cast<size_t>(n), 0);
    for (int i = 0; i < n; ++i)
        if (mask_in[i]) size[label[i]] += 1;
    for (int i = 0; i < n; ++i)
        mask_out[i] = (mask_in[i] && size[label[i]] >= min_size_px) ? 1u : 0u;
}
