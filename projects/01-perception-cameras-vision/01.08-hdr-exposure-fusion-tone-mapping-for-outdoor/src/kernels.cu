// ===========================================================================
// kernels.cu — GPU kernels + host orchestration for project 01.08
//              (HDR exposure fusion + tone mapping for outdoor robots)
//
// Role in the project
// --------------------
// This file has THREE layers, read in this order:
//
//   1) A dozen small, REUSABLE __global__ kernels (SECTION 1) — each one a
//      single map/stencil/reduce primitive. None of them "is" HDR fusion or
//      tone mapping by itself; they are the alphabet the two HDR paths are
//      spelled with. Reusing the SAME reduce/expand/affine/weighted-sum
//      primitives across both paths is a deliberate teaching choice: it
//      shows that "Gaussian pyramid", "Laplacian pyramid", and "weighted
//      blend" are not path-specific tricks but a handful of general-purpose
//      building blocks that recur across completely different algorithms
//      (Burt & Adelson's 1983 pyramid formalism underlies BOTH classic
//      answers to the HDR problem taught here).
//
//   2) Host launch wrappers (SECTION 2) — one per kernel, owning the grid/
//      block math and the mandatory post-launch error check.
//
//   3) Host ORCHESTRATION functions (SECTION 3) — run_reinhard_global_gpu,
//      run_local_tonemap_gpu, run_mertens_gpu — each a sequence of the
//      primitives above (allocate scratch, launch in order, free scratch),
//      the exact "run_..._gpu" pattern 01.03-optical-flow's pyramidal
//      Lucas-Kanade orchestration uses for the same reason: a multi-stage
//      GPU algorithm is SEQUENTIAL across stages (each stage's kernels must
//      finish before the next reads their output) even though every
//      individual kernel is itself massively parallel across pixels.
//
//   4) The shared, HOST-ONLY Debevec-Malik CRF solver (SECTION 4) — see its
//      declaration in kernels.cuh for why this one piece of "the algorithm"
//      is deliberately NOT duplicated in reference_cpu.cpp.
//
// PATH A vs PATH B, in terms of these primitives:
//   PATH A (radiance reconstruction + tone mapping):
//     radiance_merge_kernel -> {run_reinhard_global_gpu | run_local_tonemap_gpu}
//   PATH B (Mertens exposure fusion, no CRF, no radiance):
//     run_mertens_gpu (also produces the "naive single-scale blend" failure
//     case in the SAME function, since both share every weight computation
//     — only the LAST step, single-scale vs. multiscale blending, differs).
//
// Read this after: main.cu, kernels.cuh.  Read this before: reference_cpu.cpp.
// ===========================================================================

#include <cmath>                 // logf/expf/fabsf/fminf/fmaxf — host+device math
#include <cstdio>
#include <vector>

#include "kernels.cuh"
#include "util/cuda_check.cuh"

// ===========================================================================
// Launch-configuration constants (repo-wide convention — see 01.01/01.03).
//   kBlock1D — threads/block for flat (n-element) kernels: a warp multiple,
//              generous for occupancy, small enough to keep register
//              pressure low. Our largest n is kN = 19,200 (full-res image),
//              so a single 1D grid (no grid-stride loop needed — unlike the
//              SAXPY placeholder's teaching example, every buffer here is
//              small enough that ceil(n/256) never approaches any grid-size
//              limit) keeps every kernel here simple to read.
//   kBlock2D — side length of a square 2D thread block for STENCIL kernels
//              (gaussian_reduce, bilinear_expand, mertens_raw_weight), which
//              need (x, y) indices, not a flat index, to reach neighbors.
// ===========================================================================
static constexpr int kBlock1D = 256;
static constexpr int kBlock2D = 16;   // 16x16 = 256 threads/block, matches kBlock1D's occupancy target

static inline int grid1d(int n) { return (n + kBlock1D - 1) / kBlock1D; }
static inline dim3 grid2d(int w, int h)
{
    return dim3((w + kBlock2D - 1) / kBlock2D, (h + kBlock2D - 1) / kBlock2D);
}

// ---------------------------------------------------------------------------
// g_crf_table — the recovered camera response function, g[z] = ln(exposure)
// for 8-bit code value z, in GPU __constant__ memory (kCrfBins = 256 floats
// = 1 KiB, comfortably inside the 64 KiB constant-memory budget every CUDA
// device since Kepler provides).
//
// Why constant memory here, honestly assessed: constant memory is fastest
// when every thread in a warp reads the SAME address (a true broadcast) —
// that is NOT quite our access pattern, since neighboring pixels usually
// have similar but not identical brightness, so a warp's 32 threads may
// touch a handful of distinct table entries rather than one. Even so, the
// constant cache (a small, low-latency cache backing this address space)
// comfortably holds all 256 entries and serves any access pattern within a
// warp far faster than an equivalent global-memory lookup table would once
// warmed — the standard, idiomatic home for a small per-pixel LUT in CUDA
// image pipelines, and a fine illustration of "not a true broadcast, still
// clearly the right memory space" (see THEORY.md "The GPU mapping").
// ---------------------------------------------------------------------------
__constant__ float g_crf_table[kCrfBins];

void upload_crf_table(const float* h_g256)
{
    // cudaMemcpyToSymbol resolves g_crf_table by NAME at compile time (it is
    // a genuine device-resident symbol, not a regular pointer) and copies
    // into it directly — no cudaMalloc/cudaFree pair needed for __constant__
    // storage; its lifetime is the whole program, like a static global.
    CUDA_CHECK(cudaMemcpyToSymbol(g_crf_table, h_g256, kCrfBins * sizeof(float)));
}

// ===========================================================================
// SECTION 1 — the __global__ kernels.
// ===========================================================================

// ---------------------------------------------------------------------------
// hat_weight_device — Debevec & Malik's triangular ("hat") weighting
// function: w(z) = min(z, 255-z). Peaks at z=127/128 (weight 127), is ZERO
// at z=0 and z=255. Purpose: near-black and near-white samples are the ones
// most likely to be CLIPPED or dominated by sensor noise (a dark pixel's
// signal is buried in read noise; a bright pixel may already be saturated),
// so the merge below trusts mid-toned samples more — automatically, with no
// explicit clipping test (THEORY.md "The math" derives this in full).
// __forceinline__: this one-line function is called 4x per pixel in the hot
// loop below; inlining removes the call overhead entirely.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float hat_weight_device(int z)
{
    return fminf(static_cast<float>(z), static_cast<float>(255 - z));
}

// ---------------------------------------------------------------------------
// radiance_merge_kernel — Debevec-Malik radiance RECONSTRUCTION, the second
// half of PATH A (the first half, CRF recovery, is crf_solve_debevec — a
// HOST-only one-time calibration step; see SECTION 4). A pure per-pixel MAP:
// thread i owns pixel i, reads its 4 LDR samples + the recovered CRF, writes
// one radiance estimate. No pixel depends on any other — the natural GPU
// mapping is one thread per pixel, exactly like every other map kernel in
// this repository (01.01's debayer, 01.03's Scharr gradient, ...).
//
// The math (see THEORY.md "The math" for the full derivation): each
// exposure j contributes an independent estimate of ln(radiance):
//     ln(E) ~= g(Z_ij) - ln(t_j)
// (because g(Z) = ln(exposure) = ln(R*t) = ln(R) + ln(t)). We combine the
// kNumExposures=4 estimates with hat-weighted averaging in the LOG domain
// (averaging logs, not linear values, keeps the estimate well-behaved
// across an exposure range spanning orders of magnitude — see "Numerical
// considerations").
//
// The CLIPPED-EVERYWHERE fallback (report honestly, never silently wrong):
// if every one of the 4 samples is EXACTLY 0 or 255, every hat weight is
// zero and the weighted average is 0/0. Real cameras face the same wall —
// a pixel that saturates every exposure in the bracket, or reads pure black
// in every exposure, has no INFORMATION about its true radiance beyond "it
// is at least this bright" or "at most this dark". We fall back to the
// SINGLE most-informative exposure: darkest fallback uses the LONGEST
// exposure (best chance of a nonzero reading of a dim scene), brightest
// fallback uses the SHORTEST (best chance of not saturating a bright
// scene) — found by comparing the 4 ln(t) arguments at RUNTIME rather than
// assuming which index is longest/shortest, so this kernel stays correct
// even if a caller passes exposures in a different order.
//
// Parameters:
//   z0..z3      — [n] DEVICE pointers, the four LDR exposures, uint8 0..255.
//   n           — pixel count (kN in this project).
//   ln_t0..ln_t3 — ln(exposure time), one per image, SECONDS-based (the
//                  caller takes the log once on the host; see
//                  launch_radiance_merge).
//   out_radiance — [n] OUT: linear-domain radiance estimate, > 0.
// Launch: one thread per pixel (grid1d(n), kBlock1D) — see launch wrapper.
// Memory: z0..z3 and out_radiance are coalesced (thread i reads/writes
// index i); g_crf_table is the small constant-memory LUT above.
// ---------------------------------------------------------------------------
__global__ void radiance_merge_kernel(const uint8_t* __restrict__ z0,
                                      const uint8_t* __restrict__ z1,
                                      const uint8_t* __restrict__ z2,
                                      const uint8_t* __restrict__ z3,
                                      int n,
                                      float ln_t0, float ln_t1, float ln_t2, float ln_t3,
                                      float* __restrict__ out_radiance)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int  zz[4] = { z0[i], z1[i], z2[i], z3[i] };
    const float lt[4] = { ln_t0, ln_t1, ln_t2, ln_t3 };

    float wsum = 0.0f, acc = 0.0f;
    int z_max = -1, z_min = 256;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float w = hat_weight_device(zz[j]);
        acc  += w * (g_crf_table[zz[j]] - lt[j]);   // w * (g(Z) - ln t) = w * ln(E) contribution
        wsum += w;
        if (zz[j] > z_max) z_max = zz[j];
        if (zz[j] < z_min) z_min = zz[j];
    }

    float ln_e;
    if (wsum > 1e-6f) {
        ln_e = acc / wsum;                          // the normal, well-supported case
    } else {
        // Clipped-everywhere fallback (see header): every z is 0 or 255.
        // z_max tells us which extreme: if the brightest of the 4 samples
        // is still 0, the pixel was black in EVERY exposure -> use the
        // LONGEST exposure's ln(t) (largest lt[]); if z_max is 255, at
        // least one exposure saturated white -> since wsum==0 the rest
        // must also be 0 or 255, meaning this pixel is at best only
        // constrained from the WHITE side -> use the SHORTEST exposure's
        // ln(t) (smallest lt[]).
        float lt_longest = lt[0], lt_shortest = lt[0];
        #pragma unroll
        for (int j = 1; j < 4; ++j) {
            if (lt[j] > lt_longest)  lt_longest  = lt[j];
            if (lt[j] < lt_shortest) lt_shortest = lt[j];
        }
        ln_e = (z_max == 0) ? (g_crf_table[0]   - lt_longest)
                            : (g_crf_table[255] - lt_shortest);
    }
    out_radiance[i] = expf(ln_e);
}

// ---------------------------------------------------------------------------
// luminance_log_sum_kernel — GPU REDUCTION computing sum_i ln(eps + E_i),
// the numerator of the Reinhard "log-average luminance" (THEORY.md "The
// math"). Classic two-phase reduction pattern:
//   phase 1 (per block, IN THIS KERNEL): each thread loads and takes ln() of
//     its element into SHARED memory, then a binary-tree reduction halves
//     the live thread count each step (s = s/2) until partial[0] holds the
//     block's total. Shared memory is used here because it is FAST,
//     on-chip, and every thread in the block needs to read partial sums
//     written by OTHER threads — global memory could do this but would be
//     10-30x slower per access (THEORY.md "The GPU mapping").
//   phase 2 (ONE atomicAdd per block): rather than writing 75-ish partial
//     sums back to global memory for a second kernel launch to finish, each
//     block's single float32-reduced-in-shared-mem total is atomically
//     added into ONE global double accumulator. This is the "reduce in
//     shared memory, then ONE atomic per block" pattern — atomics contend
//     only across ~75 blocks (for our 19,200-pixel images), not across
//     19,200 threads, keeping contention negligible while avoiding a
//     second kernel launch entirely.
//
// double accumulator: 75 blocks summing ~19,200 float32 partial sums each
// O(1e2) in magnitude could accumulate visible float32 rounding drift;
// atomicAdd on a double (natively supported since compute capability 6.0,
// well within this project's sm_75 floor) keeps the FINAL sum accurate
// without needing per-thread double precision (THEORY.md "Numerical
// considerations" discusses the general float-accumulation story).
//
// Parameters: radiance [n], eps (avoids ln(0) for a hypothetical zero
// pixel — radiance_merge_kernel's output is always > 0 in practice, this
// is defensive), d_sum_accum — OUT, caller must cudaMemset it to 0 first.
// ---------------------------------------------------------------------------
__global__ void luminance_log_sum_kernel(const float* __restrict__ radiance,
                                         int n, float eps,
                                         double* __restrict__ d_sum_accum)
{
    __shared__ float partial[kBlock1D];   // one slot per thread in this block

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

    // Out-of-range threads (the ragged last block) contribute 0 — the
    // additive identity, so they participate harmlessly in the tree below.
    partial[tid] = (i < n) ? logf(eps + radiance[i]) : 0.0f;
    __syncthreads();   // every thread's write must be visible before anyone reduces

    // Binary-tree reduction: at each step, the first `s` threads add the
    // element `s` away from them; s halves every iteration. After
    // log2(kBlock1D) steps, partial[0] holds this BLOCK's total.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) partial[tid] += partial[tid + s];
        __syncthreads();   // must finish this level before starting the next
    }

    if (tid == 0) {
        atomicAdd(d_sum_accum, static_cast<double>(partial[0]));
    }
}

// ---------------------------------------------------------------------------
// reinhard_map_kernel — the display-referred step of Reinhard's global
// photographic tone-reproduction operator (Reinhard et al. 2002):
//     Ld = Lscaled / (1 + Lscaled),   Lscaled = key_over_lavg * E
// Strictly increasing and strictly bounded in [0, 1) for any E >= 0 — see
// THEORY.md "The math" for why this project uses the SIMPLE form (no
// separate white-point burnout term) specifically so the tone_map_range
// gate's "output in [0,1)" contract holds by construction, not by clamping.
// A pure per-pixel MAP: key_over_lavg is a SCALAR the host already computed
// from run_reinhard_global_gpu's reduction step.
// ---------------------------------------------------------------------------
__global__ void reinhard_map_kernel(const float* __restrict__ radiance,
                                    int n, float key_over_lavg,
                                    float* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float l_scaled = key_over_lavg * radiance[i];
    out[i] = l_scaled / (1.0f + l_scaled);
}

// ---------------------------------------------------------------------------
// gaussian_reduce_kernel — pyramid REDUCE: blur with a small 2D Gaussian
// then downsample by 2x. This is Burt & Adelson's REDUCE operator (1983),
// the construction step of a Gaussian pyramid, used here by BOTH HDR paths
// (local tone mapping's base/detail split, and Mertens' multiresolution
// blend of images AND weight maps).
//
// The kernel is the outer product of the classic 5-tap binomial
// approximation to a Gaussian, [1,4,6,4,1]/16 (this is literally row 4 of
// Pascal's triangle, normalized — a cheap, separable-in-spirit stand-in for
// a true Gaussian that is exact enough for pyramid construction and cheap
// to hand-derive; see THEORY.md). We apply it as a DIRECT 5x5 2D
// convolution (25 multiply-adds per output pixel) rather than the more
// efficient SEPARABLE two-pass form (two 1D passes of 5 taps each, 10
// multiply-adds) — a deliberate simplification for a stencil this small on
// images this size (see README "Exercises" for separable convolution as a
// follow-up optimization exercise; THEORY.md "The GPU mapping" discusses
// the bandwidth trade-off in full).
//
// Thread-to-data mapping: thread (ox, oy) owns ONE output pixel and reads a
// 5x5 neighborhood of the INPUT centered at (2*ox, 2*oy) (the standard
// "output pixel o maps to input pixel 2*o" for a 2x decimating REDUCE).
// Border handling: CLAMP-TO-EDGE (out-of-range input coordinates are
// clamped into [0, inW-1] / [0, inH-1]) — the simplest defensible border
// rule, and adequate here since our regions of interest for every gate sit
// well away from the image border (see scripts/make_synthetic.py's ROI
// placement comments).
//
// Parameters: in [inW*inH], out [(inW/2)*(inH/2)]. inW, inH assumed EVEN
// (guaranteed by this project's 160x120 -> 80x60 -> 40x30 pyramid geometry;
// see kernels.cuh's level_w/level_h).
// ---------------------------------------------------------------------------
__global__ void gaussian_reduce_kernel(const float* __restrict__ in, int inW, int inH,
                                       float* __restrict__ out)
{
    const int outW = inW / 2, outH = inH / 2;
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= outW || oy >= outH) return;

    // 5-tap binomial weights (sum 16); the full 2D kernel is the outer
    // product of this with itself (sum 256) — see the header for why we
    // apply it directly rather than as two separable 1D passes.
    const float tap[5] = { 1.0f, 4.0f, 6.0f, 4.0f, 1.0f };

    const int cx = ox * 2, cy = oy * 2;   // this output pixel's center in INPUT coordinates
    float acc = 0.0f;
    #pragma unroll
    for (int dy = -2; dy <= 2; ++dy) {
        // Clamp-to-edge in y, computed once per row of the 5x5 window.
        int sy = cy + dy;
        sy = sy < 0 ? 0 : (sy >= inH ? inH - 1 : sy);
        #pragma unroll
        for (int dx = -2; dx <= 2; ++dx) {
            int sx = cx + dx;
            sx = sx < 0 ? 0 : (sx >= inW ? inW - 1 : sx);
            acc += tap[dy + 2] * tap[dx + 2] * in[sy * inW + sx];
        }
    }
    out[oy * outW + ox] = acc / 256.0f;   // normalize by the 2D kernel's total weight (16*16)
}

// ---------------------------------------------------------------------------
// bilinear_expand_kernel — pyramid EXPAND: upsample a coarser level back to
// an exact target resolution via bilinear interpolation. This is a
// SIMPLIFIED stand-in for Burt & Adelson's true EXPAND operator (which
// inserts zero rows/columns and convolves with 4x the REDUCE kernel — a
// "polyphase" upsample that better matches REDUCE's frequency response).
// Bilinear interpolation is cheaper to reason about and to implement
// correctly, and is entirely adequate for this project's didactic goal
// (visibly demonstrating multiscale blending's halo reduction vs. a naive
// single-scale blend — the halo_check gate); THEORY.md "Numerical
// considerations" states this simplification and its consequence honestly.
//
// Thread-to-data mapping: thread (ox, oy) owns ONE output pixel. We use the
// standard "pixel-center" resampling convention (the same one image-resize
// code across this repository uses, e.g. 01.01's remap stage): map the
// output pixel CENTER back into input space, so the four sampled corners
// bracket it symmetrically rather than being biased toward one edge.
//
// Parameters: in [inW*inH], out [outW*outH]. outW/outH are normally 2x
// inW/inH (one pyramid level up) but the kernel imposes no such
// restriction — it works for any target size, which is exactly what lets
// run_local_tonemap_gpu chain TWO expands (level 2 -> level 1 -> level 0)
// through this SAME kernel.
// ---------------------------------------------------------------------------
__global__ void bilinear_expand_kernel(const float* __restrict__ in, int inW, int inH,
                                       float* __restrict__ out, int outW, int outH)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= outW || oy >= outH) return;

    // Map the output pixel CENTER into input coordinates: scale by the
    // size ratio, then the classic "+0.5 ... -0.5" half-pixel correction so
    // integer output pixel 0's center (0.5 in output space) lands at the
    // correct fractional input position rather than being shifted by a
    // half input pixel (the textbook resampling-alignment fix).
    const float sx = (static_cast<float>(ox) + 0.5f) * (static_cast<float>(inW) / outW) - 0.5f;
    const float sy = (static_cast<float>(oy) + 0.5f) * (static_cast<float>(inH) / outH) - 0.5f;

    // Clamp into the valid sampling range so the 4-tap gather never reads
    // out of bounds; edges naturally repeat the border value (clamp-to-edge).
    const float sxc = sx < 0.0f ? 0.0f : (sx > inW - 1.0f ? inW - 1.0f : sx);
    const float syc = sy < 0.0f ? 0.0f : (sy > inH - 1.0f ? inH - 1.0f : sy);

    const int x0 = static_cast<int>(sxc), y0 = static_cast<int>(syc);
    const int x1 = x0 + 1 < inW ? x0 + 1 : x0;   // clamp the "+1" neighbor too, at the far edge
    const int y1 = y0 + 1 < inH ? y0 + 1 : y0;
    const float fx = sxc - x0, fy = syc - y0;    // fractional position within the 2x2 cell

    const float v00 = in[y0 * inW + x0], v10 = in[y0 * inW + x1];
    const float v01 = in[y1 * inW + x0], v11 = in[y1 * inW + x1];
    const float top = v00 + (v10 - v00) * fx;    // interpolate along x on the top row...
    const float bot = v01 + (v11 - v01) * fx;    // ...and the bottom row...
    out[oy * outW + ox] = top + (bot - top) * fy; // ...then along y between them.
}

// ---------------------------------------------------------------------------
// elementwise_sub_kernel / elementwise_add_kernel — generic per-element
// combine over a FLAT n-element array (no 2D structure needed: the two
// inputs and the output all share the same indexing, whatever it means).
// Reused for: Laplacian-pyramid band construction (sub: detail = fine -
// expand(coarse)), local-tonemap's log-domain composite (add: base +
// boosted detail), and Laplacian-pyramid RECONSTRUCTION (add: band +
// expand(coarser reconstruction)). One kernel, three call sites — the
// generic-primitive reuse this file's header describes.
// ---------------------------------------------------------------------------
__global__ void elementwise_sub_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                       int n, float* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] - b[i];
}
__global__ void elementwise_add_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                       int n, float* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

// ---------------------------------------------------------------------------
// affine_kernel — out[i] = scale*in[i] + offset. The single most-reused
// primitive in this project: local tone mapping's base-layer dynamic-range
// COMPRESSION (scale=compression_factor, offset=(1-compression_factor)*
// meanBase), its detail BOOST (scale=detail_boost, offset=0), and its final
// display-range MIN-MAX NORMALIZE (scale=1/(hi-lo), offset=-lo/(hi-lo)) are
// all this one kernel with different scalars — see run_local_tonemap_gpu.
// ---------------------------------------------------------------------------
__global__ void affine_kernel(const float* __restrict__ in, int n,
                              float scale, float offset, float* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = scale * in[i] + offset;
}

// ---------------------------------------------------------------------------
// log_kernel — out[i] = ln(in[i] + eps). The first step of local tone
// mapping: working in LOG radiance compresses the ~5-decade scene range
// into an ADDITIVE (rather than multiplicative) quantity, which is what
// lets a simple affine rescale (affine_kernel) act as a dynamic-range
// compressor at all (THEORY.md "The math" derives why log-domain
// compression is the natural choice here).
// ---------------------------------------------------------------------------
__global__ void log_kernel(const float* __restrict__ in, int n, float eps,
                           float* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = logf(in[i] + eps);
}

// ---------------------------------------------------------------------------
// u8_to_unit_kernel — out[i] = in[i] / 255. Converts an LDR exposure from
// its 8-bit storage representation into the normalized [0,1] float domain
// Mertens fusion (PATH B) operates in throughout — Mertens never touches
// radiance or the CRF at all, only these normalized pixel VALUES.
// ---------------------------------------------------------------------------
__global__ void u8_to_unit_kernel(const uint8_t* __restrict__ in, int n,
                                  float* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = static_cast<float>(in[i]) * (1.0f / 255.0f);
}

// ---------------------------------------------------------------------------
// mertens_raw_weight_kernel — Mertens, Kautz & Van Reeth's (2007) per-pixel,
// per-exposure "quality" measure, TWO of its three classic terms:
//
//   contrast(x,y)      = |laplacian3x3(img)(x,y)|     — a STENCIL: high
//     near edges/texture, ~0 in flat regions. Rewards exposures that show
//     LOCAL DETAIL at this pixel.
//   wellexposedness(x,y) = exp(-(img(x,y)-0.5)^2 / (2*sigma^2))  — a MAP:
//     peaks at mid-gray (0.5), falls off toward black or white. Rewards
//     exposures that are neither clipped nor buried in shadow HERE.
//   raw_weight = contrast^wc * wellexposedness^we
//
// The THIRD classic Mertens term — SATURATION, the standard deviation of a
// pixel's R,G,B channels — is DELIBERATELY OMITTED: this project's scenes
// are single-channel (see README "Limitations & honesty" and kernels.cuh's
// file header), and saturation-across-channels has no meaning for a
// grayscale image. Rather than invent a fake substitute that merely LOOKS
// like Mertens' formula, this kernel honestly computes contrast x
// well-exposedness only (wc=we=1.0 in this project's calls — see
// run_mertens_gpu) — a real, if reduced, two-term version of the published
// algorithm, not a fabrication.
//
// Thread-to-data mapping: one thread per pixel (2D grid, matches every
// other stencil in this file); the 3x3 Laplacian needs the four
// axis-neighbors, clamp-to-edge at the border.
// ---------------------------------------------------------------------------
__global__ void mertens_raw_weight_kernel(const float* __restrict__ img01, int W, int H,
                                          float wc, float we, float sigma,
                                          float* __restrict__ out_weight)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    const int xm = x > 0 ? x - 1 : 0, xp = x < W - 1 ? x + 1 : W - 1;
    const int ym = y > 0 ? y - 1 : 0, yp = y < H - 1 ? y + 1 : H - 1;
    const float center = img01[y * W + x];

    // 3x3 Laplacian stencil [0,1,0;1,-4,1;0,1,0] — the classic discrete
    // approximation to the continuous Laplacian operator, isotropic to
    // first order and the standard "blob/edge detector" used in Mertens'
    // own reference implementation.
    const float lap = img01[y * W + xm] + img01[y * W + xp]
                     + img01[ym * W + x] + img01[yp * W + x]
                     - 4.0f * center;
    const float contrast = fabsf(lap);

    const float d = center - 0.5f;
    const float wellexposed = expf(-(d * d) / (2.0f * sigma * sigma));

    out_weight[y * W + x] = powf(contrast, wc) * powf(wellexposed, we);
}

// ---------------------------------------------------------------------------
// normalize_weights4_kernel — rescale 4 raw per-exposure weight maps so
// they sum to exactly 1 at every pixel (a per-pixel softmax-like
// normalization, but linear rather than exponential — Mertens' own
// formulation, not a softmax). Fallback: if all 4 raw weights are
// (numerically) zero at a pixel — every exposure agrees this pixel is both
// flat AND poorly exposed, a genuine edge case — fall back to a UNIFORM
// 1/4 split rather than propagate a 0/0 division, so every downstream
// consumer always receives a valid partition of unity.
// ---------------------------------------------------------------------------
__global__ void normalize_weights4_kernel(const float* __restrict__ w0, const float* __restrict__ w1,
                                          const float* __restrict__ w2, const float* __restrict__ w3,
                                          int n,
                                          float* __restrict__ o0, float* __restrict__ o1,
                                          float* __restrict__ o2, float* __restrict__ o3)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float a0 = w0[i], a1 = w1[i], a2 = w2[i], a3 = w3[i];
    const float s = a0 + a1 + a2 + a3;
    if (s > 1e-6f) {
        const float inv = 1.0f / s;
        o0[i] = a0 * inv; o1[i] = a1 * inv; o2[i] = a2 * inv; o3[i] = a3 * inv;
    } else {
        o0[i] = o1[i] = o2[i] = o3[i] = 0.25f;   // uniform fallback (see header)
    }
}

// ---------------------------------------------------------------------------
// weighted_sum4_kernel — out[i] = a0[i]*w0[i] + a1[i]*w1[i] + a2[i]*w2[i] +
// a3[i]*w3[i]. THE workhorse of this project's Mertens path: called ONCE
// at full resolution for the naive single-scale blend (the failure-case
// artifact the halo_check gate quantifies), and called ONCE PER PYRAMID
// LEVEL (with each level's own Laplacian bands and Gaussian-blurred
// weights) to build the fused Laplacian pyramid in the real, multiscale
// Mertens blend — see run_mertens_gpu.
// ---------------------------------------------------------------------------
__global__ void weighted_sum4_kernel(const float* __restrict__ a0, const float* __restrict__ w0,
                                     const float* __restrict__ a1, const float* __restrict__ w1,
                                     const float* __restrict__ a2, const float* __restrict__ w2,
                                     const float* __restrict__ a3, const float* __restrict__ w3,
                                     int n, float* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a0[i] * w0[i] + a1[i] * w1[i] + a2[i] * w2[i] + a3[i] * w3[i];
}

// ===========================================================================
// SECTION 2 — host launch wrappers. Each owns its grid/block math and the
// mandatory post-launch error check (CLAUDE.md §6.1 rule 7). Thin on
// purpose — the teaching content lives with the kernel definitions above.
// ===========================================================================
void launch_radiance_merge(const uint8_t* d_z0, const uint8_t* d_z1,
                           const uint8_t* d_z2, const uint8_t* d_z3,
                           int n, float ln_t0, float ln_t1, float ln_t2, float ln_t3,
                           float* d_out_radiance)
{
    radiance_merge_kernel<<<grid1d(n), kBlock1D>>>(d_z0, d_z1, d_z2, d_z3, n,
                                                   ln_t0, ln_t1, ln_t2, ln_t3, d_out_radiance);
    CUDA_CHECK_LAST_ERROR("radiance_merge_kernel launch");
}

void launch_luminance_log_sum(const float* d_radiance, int n, float eps, double* d_sum_accum)
{
    luminance_log_sum_kernel<<<grid1d(n), kBlock1D>>>(d_radiance, n, eps, d_sum_accum);
    CUDA_CHECK_LAST_ERROR("luminance_log_sum_kernel launch");
}

void launch_reinhard_map(const float* d_radiance, int n, float key_over_lavg, float* d_out)
{
    reinhard_map_kernel<<<grid1d(n), kBlock1D>>>(d_radiance, n, key_over_lavg, d_out);
    CUDA_CHECK_LAST_ERROR("reinhard_map_kernel launch");
}

void launch_gaussian_reduce(const float* d_in, int inW, int inH, float* d_out)
{
    gaussian_reduce_kernel<<<grid2d(inW / 2, inH / 2), dim3(kBlock2D, kBlock2D)>>>(d_in, inW, inH, d_out);
    CUDA_CHECK_LAST_ERROR("gaussian_reduce_kernel launch");
}

void launch_bilinear_expand(const float* d_in, int inW, int inH, float* d_out, int outW, int outH)
{
    bilinear_expand_kernel<<<grid2d(outW, outH), dim3(kBlock2D, kBlock2D)>>>(d_in, inW, inH, d_out, outW, outH);
    CUDA_CHECK_LAST_ERROR("bilinear_expand_kernel launch");
}

void launch_elementwise_sub(const float* d_a, const float* d_b, int n, float* d_out)
{
    elementwise_sub_kernel<<<grid1d(n), kBlock1D>>>(d_a, d_b, n, d_out);
    CUDA_CHECK_LAST_ERROR("elementwise_sub_kernel launch");
}
void launch_elementwise_add(const float* d_a, const float* d_b, int n, float* d_out)
{
    elementwise_add_kernel<<<grid1d(n), kBlock1D>>>(d_a, d_b, n, d_out);
    CUDA_CHECK_LAST_ERROR("elementwise_add_kernel launch");
}
void launch_affine(const float* d_in, int n, float scale, float offset, float* d_out)
{
    affine_kernel<<<grid1d(n), kBlock1D>>>(d_in, n, scale, offset, d_out);
    CUDA_CHECK_LAST_ERROR("affine_kernel launch");
}
void launch_log(const float* d_in, int n, float eps, float* d_out)
{
    log_kernel<<<grid1d(n), kBlock1D>>>(d_in, n, eps, d_out);
    CUDA_CHECK_LAST_ERROR("log_kernel launch");
}
void launch_u8_to_unit(const uint8_t* d_in, int n, float* d_out)
{
    u8_to_unit_kernel<<<grid1d(n), kBlock1D>>>(d_in, n, d_out);
    CUDA_CHECK_LAST_ERROR("u8_to_unit_kernel launch");
}
void launch_mertens_raw_weight(const float* d_img01, int W, int H, float wc, float we, float sigma,
                               float* d_out_weight)
{
    mertens_raw_weight_kernel<<<grid2d(W, H), dim3(kBlock2D, kBlock2D)>>>(d_img01, W, H, wc, we, sigma, d_out_weight);
    CUDA_CHECK_LAST_ERROR("mertens_raw_weight_kernel launch");
}
void launch_normalize_weights4(const float* d_w0, const float* d_w1, const float* d_w2, const float* d_w3,
                               int n, float* d_o0, float* d_o1, float* d_o2, float* d_o3)
{
    normalize_weights4_kernel<<<grid1d(n), kBlock1D>>>(d_w0, d_w1, d_w2, d_w3, n, d_o0, d_o1, d_o2, d_o3);
    CUDA_CHECK_LAST_ERROR("normalize_weights4_kernel launch");
}
void launch_weighted_sum4(const float* d_a0, const float* d_w0, const float* d_a1, const float* d_w1,
                          const float* d_a2, const float* d_w2, const float* d_a3, const float* d_w3,
                          int n, float* d_out)
{
    weighted_sum4_kernel<<<grid1d(n), kBlock1D>>>(d_a0, d_w0, d_a1, d_w1, d_a2, d_w2, d_a3, d_w3, n, d_out);
    CUDA_CHECK_LAST_ERROR("weighted_sum4_kernel launch");
}

// ===========================================================================
// SECTION 3 — host ORCHESTRATION: the two HDR paths, assembled from the
// primitives above. Each function allocates its own scratch device memory,
// launches the full sequence, frees the scratch, and leaves ONLY its
// documented output buffer populated — the caller (main.cu) supplies that
// output buffer and everything device-resident besides.
// ===========================================================================

// ---------------------------------------------------------------------------
// run_reinhard_global_gpu — PATH A, global tone mapping (README "The
// algorithm in brief"). Two stages: (1) a REDUCTION computing the
// log-average luminance L_avg (luminance_log_sum_kernel, see its header for
// the shared-memory + atomics pattern), finished with a TINY host round-
// trip (one double, not worth a second kernel); (2) a per-pixel MAP
// applying the Reinhard formula with the now-known scalar key/L_avg.
//
// Parameters: d_radiance [n] linear HDR radiance; key — the "middle gray"
// exposure target (README/THEORY.md document the chosen value, 0.18 is the
// photographic convention this project uses); d_out_reinhard [n] OUT, in
// [0, 1) by construction (see reinhard_map_kernel's header).
// ---------------------------------------------------------------------------
void run_reinhard_global_gpu(const float* d_radiance, int n, float key, float* d_out_reinhard)
{
    double* d_sum = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(double)));   // the atomicAdd accumulator MUST start at 0

    const float eps = 1e-6f;   // avoids ln(0) if radiance is ever exactly 0 (defensive; see THEORY.md)
    launch_luminance_log_sum(d_radiance, n, eps, d_sum);

    double h_sum = 0.0;
    CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_sum));

    // L_avg = exp(mean of ln(eps+E)) — the GEOMETRIC mean of radiance, the
    // photographically standard "average scene luminance" estimator
    // (Reinhard et al. 2002; THEORY.md derives why geometric, not
    // arithmetic, mean is used here).
    const double l_avg = std::exp(h_sum / static_cast<double>(n));
    const float key_over_lavg = static_cast<float>(static_cast<double>(key) / l_avg);

    launch_reinhard_map(d_radiance, n, key_over_lavg, d_out_reinhard);
}

// ---------------------------------------------------------------------------
// run_local_tonemap_gpu — PATH A, LOCAL tone mapping via a "bilateral-grid-
// lite" base/detail split (README "The algorithm in brief"; THEORY.md "The
// math" derives every step below). Unlike a true bilateral filter, the
// "base" layer here is a plain GAUSSIAN low-pass (via two pyramid REDUCEs
// and two EXPANDs) — cheaper and simpler to reason about, at the honestly-
// documented cost of blurring across sharp albedo edges too (see THEORY.md
// "Numerical considerations" and README "Limitations & honesty").
//
// Pipeline (log-radiance domain throughout, until the very last step):
//   logL = ln(E + eps)
//   G1 = REDUCE(logL), G2 = REDUCE(G1)                  [Gaussian pyramid]
//   base_full = EXPAND(EXPAND(G2))                       [re-expand to full res]
//   detail = logL - base_full                            [everything REDUCE blurred away]
//   base_compressed = compression_factor*base_full + (1-compression_factor)*mean(G2)
//   detail_boosted = detail_boost * detail
//   composite = base_compressed + detail_boosted
//   output = min-max normalize(composite) to [0, 1]       [display-range squash]
//
// The min-max normalize's lo/hi come from a SMALL host round-trip over the
// full-resolution composite (19,200 floats, ~77 KiB — negligible next to a
// dedicated reduction kernel; see the comment at the call site for why this
// project does not add two more GPU min/max reduction kernels here, having
// already taught that pattern in run_reinhard_global_gpu).
// ---------------------------------------------------------------------------
void run_local_tonemap_gpu(const float* d_radiance, int W, int H,
                           float compression_factor, float detail_boost,
                           float* d_out_tonemap)
{
    const int n = W * H;
    const int w1 = W / 2, h1 = H / 2;     // level 1 dims
    const int w2 = w1 / 2, h2 = h1 / 2;   // level 2 dims (coarsest)
    const int n1 = w1 * h1, n2 = w2 * h2;

    float *d_logL = nullptr, *d_g1 = nullptr, *d_g2 = nullptr;
    float *d_baseMid = nullptr, *d_baseFull = nullptr, *d_detail = nullptr;
    float *d_baseComp = nullptr, *d_detailBoost = nullptr, *d_composite = nullptr;
    CUDA_CHECK(cudaMalloc(&d_logL, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g1, n1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g2, n2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_baseMid, n1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_baseFull, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_detail, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_baseComp, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_detailBoost, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_composite, n * sizeof(float)));

    launch_log(d_radiance, n, 1e-6f, d_logL);
    launch_gaussian_reduce(d_logL, W, H, d_g1);
    launch_gaussian_reduce(d_g1, w1, h1, d_g2);
    launch_bilinear_expand(d_g2, w2, h2, d_baseMid, w1, h1);
    launch_bilinear_expand(d_baseMid, w1, h1, d_baseFull, W, H);
    launch_elementwise_sub(d_logL, d_baseFull, n, d_detail);

    // Tiny host round-trip: mean of the COARSEST level (n2 = 40*30 = 1,200
    // floats — trivial to copy back and average on the CPU; this scalar
    // recenters the compression below so it shrinks the RANGE of base_full
    // around the scene's own overall log-brightness rather than around 0).
    std::vector<float> h_g2(static_cast<size_t>(n2));
    CUDA_CHECK(cudaMemcpy(h_g2.data(), d_g2, n2 * sizeof(float), cudaMemcpyDeviceToHost));
    double mean_g2 = 0.0;
    for (float v : h_g2) mean_g2 += v;
    mean_g2 /= static_cast<double>(n2);

    const float offset = static_cast<float>((1.0 - static_cast<double>(compression_factor)) * mean_g2);
    launch_affine(d_baseFull, n, compression_factor, offset, d_baseComp);
    launch_affine(d_detail, n, detail_boost, 0.0f, d_detailBoost);
    launch_elementwise_add(d_baseComp, d_detailBoost, n, d_composite);

    // Second tiny host round-trip: min/max of the FULL-resolution composite
    // (19,200 floats, ~77 KiB) to derive an EXACT min-max normalize to
    // [0, 1] — guaranteeing the output range by construction rather than by
    // an assumed/clamped bound (see the file header for why this is done
    // host-side rather than with two more dedicated reduction kernels).
    std::vector<float> h_comp(static_cast<size_t>(n));
    CUDA_CHECK(cudaMemcpy(h_comp.data(), d_composite, n * sizeof(float), cudaMemcpyDeviceToHost));
    float lo = h_comp[0], hi = h_comp[0];
    for (float v : h_comp) { if (v < lo) lo = v; if (v > hi) hi = v; }
    const float range = (hi - lo) > 1e-6f ? (hi - lo) : 1.0f;   // guard a degenerate flat scene
    launch_affine(d_composite, n, 1.0f / range, -lo / range, d_out_tonemap);

    CUDA_CHECK(cudaFree(d_logL));      CUDA_CHECK(cudaFree(d_g1));         CUDA_CHECK(cudaFree(d_g2));
    CUDA_CHECK(cudaFree(d_baseMid));   CUDA_CHECK(cudaFree(d_baseFull));   CUDA_CHECK(cudaFree(d_detail));
    CUDA_CHECK(cudaFree(d_baseComp));  CUDA_CHECK(cudaFree(d_detailBoost)); CUDA_CHECK(cudaFree(d_composite));
}

// ---------------------------------------------------------------------------
// run_mertens_gpu — PATH B, exposure fusion (README "The algorithm in
// brief"; THEORY.md "The algorithm"). Produces BOTH the real multiscale
// Mertens blend (d_out_fused) AND the naive single-scale blend
// (d_out_naive) that the halo_check gate compares it against — they share
// every weight computation, differing only in the LAST step (a single
// full-resolution weighted_sum4 vs. a Laplacian-pyramid blend-and-
// reconstruct), which is exactly the point this project's halo_check gate
// makes quantitative: multiscale blending is the fix for single-scale
// blending's haloing, not a different weighting scheme.
//
// Pipeline:
//   1) normalize each LDR exposure to [0,1]           (u8_to_unit_kernel x4)
//   2) raw weight = contrast x well-exposedness         (mertens_raw_weight_kernel x4)
//   3) normalize the 4 raw weights to sum to 1           (normalize_weights4_kernel)
//   4) NAIVE blend = single full-res weighted sum        (weighted_sum4_kernel) -> d_out_naive
//   5) build a 3-level Gaussian pyramid of EACH normalized image (GI) and
//      EACH normalized weight map (GW)                   (gaussian_reduce_kernel x2 per exposure, x2)
//   6) derive each image's LAPLACIAN bands from its Gaussian pyramid:
//        LI[l] = GI[l] - EXPAND(GI[l+1])  for l=0,1;  LI[2] = GI[2] (coarsest, no subtraction)
//   7) fuse EACH pyramid level: FL[l] = sum_j GW_j[l] * LI_j[l]     (weighted_sum4_kernel x3 levels)
//   8) reconstruct top-down: R2=FL[2]; R1=FL[1]+EXPAND(R2); R0=FL[0]+EXPAND(R1) -> d_out_fused
//
// Parameters: d_z0..z3 [W*H] uint8 LDR exposures; wc, we, sigma — the
// Mertens weight-formula parameters (see mertens_raw_weight_kernel);
// d_out_naive, d_out_fused [W*H] OUT, both roughly in [0,1] (Laplacian
// reconstruction can slightly overshoot at very sharp edges — see
// THEORY.md "Numerical considerations"; main.cu clamps on PGM write).
// ---------------------------------------------------------------------------
void run_mertens_gpu(const uint8_t* d_z0, const uint8_t* d_z1, const uint8_t* d_z2, const uint8_t* d_z3,
                     int W, int H, float wc, float we, float sigma,
                     float* d_out_naive, float* d_out_fused)
{
    const uint8_t* z[kNumExposures] = { d_z0, d_z1, d_z2, d_z3 };
    const int dims_w[kNumLevels] = { W, W / 2, W / 4 };
    const int dims_h[kNumLevels] = { H, H / 2, H / 4 };
    const int n = W * H;

    // ---- step 1+2+3: normalize, raw weight, normalize-to-sum-1 -----------
    float* d_img[kNumExposures];      // normalized [0,1] LDR images, level 0
    float* d_rawW[kNumExposures];
    float* d_W0lvl[kNumExposures];    // normalized weights, level 0 (fed into the Gaussian weight pyramid)
    for (int j = 0; j < kNumExposures; ++j) {
        CUDA_CHECK(cudaMalloc(&d_img[j], n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rawW[j], n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_W0lvl[j], n * sizeof(float)));
        launch_u8_to_unit(z[j], n, d_img[j]);
        launch_mertens_raw_weight(d_img[j], W, H, wc, we, sigma, d_rawW[j]);
    }
    launch_normalize_weights4(d_rawW[0], d_rawW[1], d_rawW[2], d_rawW[3], n,
                              d_W0lvl[0], d_W0lvl[1], d_W0lvl[2], d_W0lvl[3]);
    for (int j = 0; j < kNumExposures; ++j) CUDA_CHECK(cudaFree(d_rawW[j]));   // raw weights no longer needed

    // ---- step 4: NAIVE single-scale blend (the failure-case baseline) ----
    launch_weighted_sum4(d_img[0], d_W0lvl[0], d_img[1], d_W0lvl[1],
                         d_img[2], d_W0lvl[2], d_img[3], d_W0lvl[3], n, d_out_naive);

    // ---- step 5: Gaussian pyramids of images (GI) and weights (GW) -------
    // GI[j][0]/GW[j][0] ALIAS d_img[j]/d_W0lvl[j] (level 0 is just the
    // full-res buffer already computed above) — only levels 1 and 2 need
    // fresh allocations.
    float* d_GI[kNumExposures][kNumLevels];
    float* d_GW[kNumExposures][kNumLevels];
    for (int j = 0; j < kNumExposures; ++j) {
        d_GI[j][0] = d_img[j];
        d_GW[j][0] = d_W0lvl[j];
        for (int l = 1; l < kNumLevels; ++l) {
            const int nl = dims_w[l] * dims_h[l];
            CUDA_CHECK(cudaMalloc(&d_GI[j][l], nl * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_GW[j][l], nl * sizeof(float)));
            launch_gaussian_reduce(d_GI[j][l - 1], dims_w[l - 1], dims_h[l - 1], d_GI[j][l]);
            launch_gaussian_reduce(d_GW[j][l - 1], dims_w[l - 1], dims_h[l - 1], d_GW[j][l]);
        }
    }

    // ---- step 6: Laplacian bands LI[j][0..1] = GI[j][l] - EXPAND(GI[j][l+1]);
    //      LI[j][2] is just GI[j][2] itself (the coarsest Gaussian level IS
    //      the coarsest Laplacian-pyramid "band" by definition — no finer
    //      level exists to expand and subtract; see THEORY.md). -----------
    float* d_LI[kNumExposures][kNumLevels];
    for (int j = 0; j < kNumExposures; ++j) {
        d_LI[j][kNumLevels - 1] = d_GI[j][kNumLevels - 1];   // alias: coarsest band == coarsest Gaussian level
        for (int l = 0; l < kNumLevels - 1; ++l) {
            const int nl = dims_w[l] * dims_h[l];
            CUDA_CHECK(cudaMalloc(&d_LI[j][l], nl * sizeof(float)));
            float* d_expanded = nullptr;
            CUDA_CHECK(cudaMalloc(&d_expanded, nl * sizeof(float)));
            launch_bilinear_expand(d_GI[j][l + 1], dims_w[l + 1], dims_h[l + 1], d_expanded, dims_w[l], dims_h[l]);
            launch_elementwise_sub(d_GI[j][l], d_expanded, nl, d_LI[j][l]);
            CUDA_CHECK(cudaFree(d_expanded));
        }
    }

    // ---- step 7: fuse each level: FL[l] = sum_j GW[j][l] * LI[j][l] ------
    float* d_FL[kNumLevels];
    for (int l = 0; l < kNumLevels; ++l) {
        const int nl = dims_w[l] * dims_h[l];
        CUDA_CHECK(cudaMalloc(&d_FL[l], nl * sizeof(float)));
        launch_weighted_sum4(d_LI[0][l], d_GW[0][l], d_LI[1][l], d_GW[1][l],
                             d_LI[2][l], d_GW[2][l], d_LI[3][l], d_GW[3][l], nl, d_FL[l]);
    }

    // ---- step 8: reconstruct coarse-to-fine: R = FL[l] + EXPAND(R_coarser)
    float* d_recon = d_FL[kNumLevels - 1];   // coarsest reconstruction IS the coarsest fused band
    for (int l = kNumLevels - 2; l >= 0; --l) {
        const int nl = dims_w[l] * dims_h[l];
        float* d_expanded = nullptr;
        CUDA_CHECK(cudaMalloc(&d_expanded, nl * sizeof(float)));
        launch_bilinear_expand(d_recon, dims_w[l + 1], dims_h[l + 1], d_expanded, dims_w[l], dims_h[l]);
        float* d_next = (l == 0) ? d_out_fused : nullptr;
        if (l != 0) CUDA_CHECK(cudaMalloc(&d_next, nl * sizeof(float)));
        launch_elementwise_add(d_FL[l], d_expanded, nl, d_next);
        CUDA_CHECK(cudaFree(d_expanded));
        if (d_recon != d_FL[kNumLevels - 1]) CUDA_CHECK(cudaFree(d_recon));   // free the PREVIOUS level's recon (not FL, freed below)
        d_recon = d_next;
    }

    // ---- free every scratch buffer (level-0 GI/GW alias d_img/d_W0lvl,
    //      freed once via those; d_LI[j][kNumLevels-1] aliases d_GI[j]
    //      [kNumLevels-1], also freed once via d_GI below) ----------------
    for (int j = 0; j < kNumExposures; ++j) {
        CUDA_CHECK(cudaFree(d_img[j]));
        CUDA_CHECK(cudaFree(d_W0lvl[j]));
        for (int l = 1; l < kNumLevels; ++l) { CUDA_CHECK(cudaFree(d_GI[j][l])); CUDA_CHECK(cudaFree(d_GW[j][l])); }
        for (int l = 0; l < kNumLevels - 1; ++l) CUDA_CHECK(cudaFree(d_LI[j][l]));
    }
    for (int l = 0; l < kNumLevels; ++l) CUDA_CHECK(cudaFree(d_FL[l]));
}

// ===========================================================================
// SECTION 4 — the shared, HOST-ONLY Debevec-Malik CRF solver. See its
// declaration in kernels.cuh for the full statement of why this ONE
// function is shared across the "GPU path" and "CPU path" rather than
// duplicated (the twin-independence ruling's documented exception).
// ===========================================================================

// ---------------------------------------------------------------------------
// hat_weight_host — the host-side twin of hat_weight_device above (see that
// kernel's header for the formula and the rationale). Kept as a SEPARATE,
// independently-typed function (not a shared __host__ __device__ helper)
// even though it is one line, simply because crf_solve_debevec is itself
// entirely host code with no device counterpart to share with.
// ---------------------------------------------------------------------------
static inline double hat_weight_host(int z)
{
    return static_cast<double>(z < (255 - z) ? z : (255 - z));
}

// ---------------------------------------------------------------------------
// solve_linear_system — Gaussian elimination with partial pivoting, double
// precision, in place. A x = b, both A (n x n, row-major) and b (n) are
// OVERWRITTEN during elimination; x is written to a caller-provided output.
//
// Why hand-rolled Gaussian elimination rather than a library call: this
// repo's dependency policy (CLAUDE.md §5) allows cuSOLVER/cuBLAS only in
// projects that are explicitly ABOUT them; this project is about HDR
// imaging, not linear algebra, so a small (~320x320) dense solve is
// implemented directly, exactly as project 33.01 (batched small-matrix
// linalg, this repo's foundational-libraries flagship for exactly this
// class of problem) teaches the underlying operation. See THEORY.md "The
// GPU mapping" for why this system is NOT a good GPU-parallelization
// candidate at this size (a single ~320x320 solve, run ONCE per capture —
// see the file header — has far too little work to amortize a kernel
// launch, let alone justify cuSOLVER's dependency weight).
//
// Complexity: O(n^3) time (elimination) + O(n^2) (back-substitution),
// O(n^2) extra space for A (already allocated by the caller). For this
// project's n = 256 + P (P=64 sample pixels -> n=320), that is ~3.3e7
// double-precision operations — comfortably under a millisecond even
// unoptimized (measured in demo/expected_output.txt's [time] line).
//
// Parameters: A [n*n] row-major, IN/OUT (destroyed); b [n] IN/OUT
// (destroyed); n — system size; x [n] OUT — the solution.
// Returns: false if a pivot smaller than a numerical-singularity floor is
// found (should not happen given this project's pin equation guarantees
// full rank — see crf_solve_debevec — but checked and reported rather than
// silently producing garbage, CLAUDE.md §13 honesty).
// ---------------------------------------------------------------------------
static bool solve_linear_system(std::vector<double>& A, std::vector<double>& b, int n, std::vector<double>& x)
{
    for (int col = 0; col < n; ++col) {
        // Partial pivoting: swap in the row (at or below `col`) with the
        // LARGEST magnitude in this column, improving numerical stability
        // (small/zero pivots amplify rounding error catastrophically).
        int pivot = col;
        double maxval = std::fabs(A[col * n + col]);
        for (int r = col + 1; r < n; ++r) {
            const double v = std::fabs(A[r * n + col]);
            if (v > maxval) { maxval = v; pivot = r; }
        }
        if (maxval < 1e-12) return false;   // numerically singular — should not occur (see header)
        if (pivot != col) {
            for (int c = 0; c < n; ++c) std::swap(A[col * n + c], A[pivot * n + c]);
            std::swap(b[col], b[pivot]);
        }
        // Eliminate this column from every row BELOW the pivot.
        for (int r = col + 1; r < n; ++r) {
            const double factor = A[r * n + col] / A[col * n + col];
            if (factor == 0.0) continue;   // already zero — skip the wasted row pass
            for (int c = col; c < n; ++c) A[r * n + c] -= factor * A[col * n + c];
            b[r] -= factor * b[col];
        }
    }
    // Back-substitution: solve the now-upper-triangular system bottom-up.
    x.assign(static_cast<size_t>(n), 0.0);
    for (int row = n - 1; row >= 0; --row) {
        double sum = b[row];
        for (int c = row + 1; c < n; ++c) sum -= A[row * n + c] * x[c];
        x[row] = sum / A[row * n + row];
    }
    return true;
}

// ---------------------------------------------------------------------------
// crf_solve_debevec — see the full parameter/purpose documentation in
// kernels.cuh (SECTION 5). This is Debevec & Malik's gsolve algorithm
// (SIGGRAPH 1997), built as WEIGHTED NORMAL EQUATIONS (A^T A x = A^T b)
// rather than the paper's own SVD-based least-squares solve — a choice
// this project makes deliberately so the "small dense system, Gaussian
// elimination" teaching goal (see the task brief this project was built
// from, and THEORY.md "The algorithm") has a concrete, hand-rollable
// implementation instead of an opaque SVD library call.
//
// Because every equation below touches at most THREE unknowns (a data term
// touches g[z] and lnE[p]; a smoothness term touches three neighboring
// g[] entries; the pin touches one), we accumulate A^T A and A^T b
// directly via nested loops — never materializing the full (sparse,
// hundreds-of-rows) design matrix A itself — exactly the standard trick
// for forming normal equations from a SPARSE system without a sparse-
// matrix library (THEORY.md "The algorithm" walks the derivation in full).
// ---------------------------------------------------------------------------
void crf_solve_debevec(const uint8_t* z0, const uint8_t* z1, const uint8_t* z2, const uint8_t* z3,
                       int W, int H, float t0, float t1, float t2, float t3,
                       int grid_n, int margin, float lambda,
                       float* out_g256)
{
    const uint8_t* images[kNumExposures] = { z0, z1, z2, z3 };
    const double ln_t[kNumExposures] = { std::log(static_cast<double>(t0)), std::log(static_cast<double>(t1)),
                                         std::log(static_cast<double>(t2)), std::log(static_cast<double>(t3)) };

    // ---- choose the P = grid_n*grid_n sample pixel coordinates -----------
    // A regular grid across the image (not random pixels, unlike Debevec's
    // own paper): deterministic, reproducible with no RNG, and — given this
    // project's scene layout (horizontal brightness BANDS, see
    // scripts/make_synthetic.py) — a regular grid crosses every band
    // (sky/shade/concrete/shadow/gradient strip), giving the solve good
    // brightness coverage without needing to special-case region selection.
    const int P = grid_n * grid_n;
    std::vector<int> sample_x(static_cast<size_t>(P)), sample_y(static_cast<size_t>(P));
    for (int iy = 0; iy < grid_n; ++iy) {
        for (int ix = 0; ix < grid_n; ++ix) {
            const int idx = iy * grid_n + ix;
            sample_x[static_cast<size_t>(idx)] = margin + (grid_n > 1
                ? (ix * (W - 1 - 2 * margin)) / (grid_n - 1) : 0);
            sample_y[static_cast<size_t>(idx)] = margin + (grid_n > 1
                ? (iy * (H - 1 - 2 * margin)) / (grid_n - 1) : 0);
        }
    }

    // ---- unknowns: 256 CRF entries g[0..255] + P log-irradiances lnE[p] --
    const int nUnk = kCrfBins + P;
    std::vector<double> ATA(static_cast<size_t>(nUnk) * static_cast<size_t>(nUnk), 0.0);
    std::vector<double> ATb(static_cast<size_t>(nUnk), 0.0);

    // add_row — accumulate ONE weighted equation "coef . x = target" into
    // the normal equations (see the file header for the derivation:
    // ATA += w^2 * outer(coef, coef), ATb += w^2 * coef * target). `terms`
    // is a small list of (unknown_index, coefficient) pairs — at most 3 for
    // every equation this solver ever builds.
    auto add_row = [&](std::initializer_list<std::pair<int, double>> terms, double target, double w) {
        const double w2 = w * w;
        for (auto& t1 : terms) {
            ATb[static_cast<size_t>(t1.first)] += w2 * t1.second * target;
            for (auto& t2 : terms) {
                ATA[static_cast<size_t>(t1.first) * nUnk + t2.first] += w2 * t1.second * t2.second;
            }
        }
    };

    // ---- data terms: g(Zij) - lnE_i - ln(t_j) = 0, weighted by w(Zij) ----
    for (int p = 0; p < P; ++p) {
        const int pix = sample_y[static_cast<size_t>(p)] * W + sample_x[static_cast<size_t>(p)];
        for (int j = 0; j < kNumExposures; ++j) {
            const int z = images[j][pix];
            const double w = hat_weight_host(z);
            if (w <= 0.0) continue;   // zero-weight rows contribute nothing — skip (see header)
            add_row({ { z, 1.0 }, { kCrfBins + p, -1.0 } }, ln_t[j], w);
        }
    }

    // ---- smoothness prior: penalize the discrete second derivative of g,
    //      weighted by w(z) so the prior matters most where DATA support is
    //      weakest (near z=0/255) — Debevec & Malik's own formulation. ----
    for (int z = 1; z < kCrfBins - 1; ++z) {
        const double w = hat_weight_host(z);
        add_row({ { z - 1, 1.0 }, { z, -2.0 }, { z + 1, 1.0 } }, 0.0, static_cast<double>(lambda) * w);
    }

    // ---- pin g(128) = 0: removes the 1-parameter scale ambiguity (adding
    //      a constant to every g and every lnE leaves every data-term
    //      residual UNCHANGED — see kernels.cuh SECTION 5's file header). -
    add_row({ { 128, 1.0 } }, 0.0, 1.0);

    // ---- solve and unpack g[0..255] (the lnE[p] values are discarded —
    //      radiance_merge_kernel/_cpu re-derive radiance for EVERY pixel,
    //      not just these P samples, from g alone). --------------------
    std::vector<double> x;
    const bool ok = solve_linear_system(ATA, ATb, nUnk, x);
    if (!ok) {
        // Should not happen (the pin equation guarantees full rank); if it
        // ever does, fail LOUDLY rather than silently returning garbage
        // (CLAUDE.md §13 honesty) — the caller's crf_recovery gate would
        // otherwise report a bogus curve-fit error a learner would waste
        // time chasing in the wrong place.
        std::fprintf(stderr, "crf_solve_debevec: FATAL — normal-equations system is numerically singular\n");
        for (int z = 0; z < kCrfBins; ++z) out_g256[z] = 0.0f;
        return;
    }
    for (int z = 0; z < kCrfBins; ++z) out_g256[z] = static_cast<float>(x[static_cast<size_t>(z)]);
}
