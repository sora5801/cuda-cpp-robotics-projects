// ===========================================================================
// kernels.cu — GPU implementation for project 01.11
//              Low-light denoising (bilateral, non-local means, BM3D-lite)
//
// Five denoisers, five GPU patterns (kernels.cuh Sections 4-6 give the
// per-method parameters; THEORY.md "The GPU mapping" gives the full essay
// on each):
//   1. bilateral_naive_kernel  — a STENCIL: one thread per output pixel,
//      reads its 9x9 neighborhood from GLOBAL memory every time.
//   2. bilateral_tiled_kernel  — the SAME stencil, SHARED-memory tiled: a
//      block cooperatively loads its neighborhood (with halo) ONCE, then
//      every thread in the block reuses it. Same loop, same order, same
//      arithmetic as (1) -> bit-identical output, faster memory traffic
//      (main.cu measures and reports the difference: THIS project's
//      "tiling lesson, quantified", per the task brief).
//   3. gaussian_blur_kernel    — bilateral's spatial term ALONE: the
//      DESIGNED NEGATIVE CONTROL (kernels.cuh Section 4's header explains
//      why it is a separate kernel, not a bilateral call with sigma_range
//      set to infinity — so a future bilateral edit cannot silently change
//      the control's behavior too).
//   4. nlm_kernel              — a SEARCH: one thread per output pixel,
//      each comparing its 5x5 patch against 169 candidate patches in a
//      13x13 window. The expensive kernel (O(search_area * patch_area) per
//      pixel) — THEORY.md documents the integral-image speedup real NLM
//      implementations use and this project deliberately does not build.
//   5. bm3d_group_kernel + bm3d_finalize_kernel — BM3D-lite: one thread per
//      REFERENCE GROUP (not per pixel!) does block-matching, a separable
//      3-D transform (2-D DCT + 1-D Haar), hard-thresholding, the inverse
//      transform, and scatter-accumulates its 16 denoised patches into two
//      shared kN-sized buffers via atomicAdd (many groups' patches overlap
//      in pixel space — that overlap IS the aggregation, the reason BM3D
//      denoises better than a single non-overlapping tiling would). The
//      finalize kernel then divides sum by weight, one thread per pixel.
//
// Border handling (all five methods, one convention): CLAMP-TO-EDGE — any
// neighbor/patch/candidate coordinate that would fall outside [0,W)x[0,H)
// is clamped into range instead (clamp_coord() below). This is the simplest
// correct choice for a teaching kernel and is applied IDENTICALLY on the
// CPU twins in reference_cpu.cpp, so it never becomes a VERIFY-stage
// disagreement.
//
// Read this after: kernels.cuh.  Companion oracle: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"

#include <cfloat>    // FLT_MAX — the "no candidate yet" sentinel in BM3D-lite's top-16 search
#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// clamp_coord — clamp a possibly out-of-range pixel coordinate into [0, n).
// Shared by every kernel below (device-only; reference_cpu.cpp defines its
// OWN independent host copy — see that file's header for why duplicating
// this one-line function is the repo's rule, not laziness).
// ---------------------------------------------------------------------------
__device__ __forceinline__ int clamp_coord(int v, int n)
{
    return v < 0 ? 0 : (v >= n ? n - 1 : v);
}

// ===========================================================================
// 1) BILATERAL — naive global-memory stencil.
//
// Thread-to-data mapping: thread (bx*bdx+tx, by*bdy+ty) owns output pixel
// (x, y); a 2-D grid over the WxH image (the natural mapping for any
// per-pixel stencil — contrast MPPI's 1-D "one thread per rollout").
//
// Memory behavior: EVERY thread re-reads its own 9x9=81-pixel neighborhood
// from GLOBAL memory. Neighboring threads' neighborhoods overlap heavily
// (an 8-pixel-radius overlap band on every side) — each interior pixel of
// the image is therefore read up to 81 times total across all threads that
// need it. That redundant global traffic is exactly what (2) below removes.
//
// Numerics: weight = exp(spatial_term + range_term) — computed as ONE expf
// call on the SUM of the two exponents (mathematically identical to, and
// one transcendental call cheaper than, exp(spatial)*exp(range); THEORY.md
// "Numerical considerations" discusses the weight-underflow floor this
// implies for range_term when |diff| is large).
// ---------------------------------------------------------------------------
__global__ void bilateral_naive_kernel(const float* __restrict__ img, int W, int H,
                                       float* __restrict__ out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's output column
    const int y = blockIdx.y * blockDim.y + threadIdx.y;   // this thread's output row
    if (x >= W || y >= H) return;                          // ragged-tile guard (2-D)

    const float center = img[y * W + x];        // I(x,y): the range term compares every neighbor to THIS
    const float inv2ss = 1.0f / (2.0f * kBilateralSigmaSpatial * kBilateralSigmaSpatial);
    const float inv2sr = 1.0f / (2.0f * kBilateralSigmaRange * kBilateralSigmaRange);

    float wsum = 0.0f;   // normalizer: sum of weights (never zero — dx=dy=0 always contributes weight 1)
    float vsum = 0.0f;   // weighted intensity accumulator

    // Fixed raster order (dy outer, dx inner) — DELIBERATELY the same order
    // the tiled kernel below uses, so both kernels perform the identical
    // sequence of floating-point operations and therefore produce
    // BIT-IDENTICAL output (verified in main.cu's dedicated tiling check;
    // see that kernel's header for the full argument).
    for (int dy = -kBilateralRadius; dy <= kBilateralRadius; ++dy) {
        const int sy = clamp_coord(y + dy, H);
        for (int dx = -kBilateralRadius; dx <= kBilateralRadius; ++dx) {
            const int sx = clamp_coord(x + dx, W);
            const float v = img[sy * W + sx];
            const float spatial_term = -static_cast<float>(dx * dx + dy * dy) * inv2ss;
            const float diff = v - center;
            const float range_term = -(diff * diff) * inv2sr;
            const float w = expf(spatial_term + range_term);
            wsum += w;
            vsum += w * v;
        }
    }
    out[y * W + x] = vsum / wsum;
}

// ===========================================================================
// 2) BILATERAL — shared-memory TILED stencil (same math as (1) above).
//
// The idea: have the BLOCK cooperatively load its neighborhood — the
// blockDim.x x blockDim.y output tile PLUS an 8-pixel halo (kBilateralRadius
// on every side) — into shared memory ONCE, then every thread reads its
// 9x9 window from that fast on-chip cache instead of re-issuing 81 global
// loads. THEORY.md "The GPU mapping" derives the traffic reduction exactly:
// each pixel is read from GLOBAL memory ~once per block that touches it
// (a handful of times) instead of up to 81 times.
//
// Launch configuration (see launch_bilateral_tiled): 16x16 thread blocks;
// the shared tile is (16+2*4) x (16+2*4) = 24x24 floats = 2.25 KiB — a
// trivial fraction of a 48-128 KiB shared-memory budget, so occupancy is
// unaffected by the allocation (THEORY.md quantifies this).
//
// Correctness-by-construction: the per-thread double loop below is BYTE-FOR-
// BYTE the same nest as bilateral_naive_kernel's — same bounds, same order,
// same expf(spatial+range) formula — the ONLY difference is where `v` and
// `center` are read FROM (shared tile vs. global array). Since the values
// read are identical either way, and IEEE-754 arithmetic is deterministic
// given an identical operation sequence, the two kernels' outputs are
// BIT-IDENTICAL (0 ULP) — not merely "close". main.cu's tiling-speedup
// check verifies this with max_abs_diff == 0.0 exactly, and documents the
// reasoning (not just the number) for a reader who doubts it.
// ---------------------------------------------------------------------------
__global__ void bilateral_tiled_kernel(const float* __restrict__ img, int W, int H,
                                       float* __restrict__ out)
{
    extern __shared__ float tile[];   // dynamic shared memory: tileW*tileH floats (sized by the launcher)

    const int R = kBilateralRadius;                 // 4: halo width on every side
    const int tileW = blockDim.x + 2 * R;            // this block's tile width, including halo
    const int tileH = blockDim.y + 2 * R;            // this block's tile height, including halo

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;

    // ---- Cooperative halo load ---------------------------------------------
    // The tile is generally LARGER than the block (24x24 vs 16x16 = 576 vs
    // 256 cells), so each thread may load more than one tile cell — a
    // grid-stride-style loop over the tile's own local coordinate space.
    // Every load clamps its SOURCE pixel the same way the naive kernel
    // clamps its neighbor reads, so the two kernels see identical border
    // behavior too (part of the bit-identical argument above).
    for (int ly = ty; ly < tileH; ly += blockDim.y) {
        const int sy = clamp_coord(blockIdx.y * blockDim.y + ly - R, H);
        for (int lx = tx; lx < tileW; lx += blockDim.x) {
            const int sx = clamp_coord(blockIdx.x * blockDim.x + lx - R, W);
            tile[ly * tileW + lx] = img[sy * W + sx];
        }
    }
    __syncthreads();   // every thread's window read below must see a FULLY loaded tile

    if (x >= W || y >= H) return;   // ragged-tile guard AFTER the load: idle threads still helped load

    const float center = tile[(ty + R) * tileW + (tx + R)];   // this thread's own pixel, from the tile
    const float inv2ss = 1.0f / (2.0f * kBilateralSigmaSpatial * kBilateralSigmaSpatial);
    const float inv2sr = 1.0f / (2.0f * kBilateralSigmaRange * kBilateralSigmaRange);

    float wsum = 0.0f, vsum = 0.0f;
    for (int dy = -R; dy <= R; ++dy) {          // IDENTICAL nest to bilateral_naive_kernel
        for (int dx = -R; dx <= R; ++dx) {
            const float v = tile[(ty + R + dy) * tileW + (tx + R + dx)];
            const float spatial_term = -static_cast<float>(dx * dx + dy * dy) * inv2ss;
            const float diff = v - center;
            const float range_term = -(diff * diff) * inv2sr;
            const float w = expf(spatial_term + range_term);
            wsum += w;
            vsum += w * v;
        }
    }
    out[y * W + x] = vsum / wsum;
}

// ===========================================================================
// 3) GAUSSIAN BLUR — the negative control: bilateral's spatial term ALONE
// (no range/photometric weighting at all — every pixel in the 9x9 window
// is trusted purely by DISTANCE, so the filter cannot tell "same surface,
// different noise" from "different surface" and mixes across edges by
// construction). Same window, same sigma_spatial as bilateral, so any
// difference in behavior between the two is attributable ENTIRELY to the
// missing range term — the controlled-experiment framing README/THEORY use.
// ---------------------------------------------------------------------------
__global__ void gaussian_blur_kernel(const float* __restrict__ img, int W, int H,
                                     float* __restrict__ out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    const float inv2ss = 1.0f / (2.0f * kBilateralSigmaSpatial * kBilateralSigmaSpatial);
    float wsum = 0.0f, vsum = 0.0f;
    for (int dy = -kBilateralRadius; dy <= kBilateralRadius; ++dy) {
        const int sy = clamp_coord(y + dy, H);
        for (int dx = -kBilateralRadius; dx <= kBilateralRadius; ++dx) {
            const int sx = clamp_coord(x + dx, W);
            const float v = img[sy * W + sx];
            const float w = expf(-static_cast<float>(dx * dx + dy * dy) * inv2ss);
            wsum += w;
            vsum += w * v;
        }
    }
    out[y * W + x] = vsum / wsum;
}

// ===========================================================================
// 4) NON-LOCAL MEANS — one thread per output pixel; the expensive kernel.
//
// For every candidate q in a 13x13 SEARCH window around p=(x,y), compare
// the 5x5 PATCH centered at p to the 5x5 patch centered at q via mean
// squared difference; weight q by exp(-patchDist / h^2) (THEORY.md "The
// math" derives this from the self-similarity prior). Cost per output
// pixel: |search|*|patch| = 169*25 = 4,225 squared differences — the
// O(search_area x patch_area) cost THEORY.md's complexity section names,
// and the reason this is the slowest of the five kernels (main.cu's [time]
// line measures it directly).
//
// Memory behavior: every one of the 169 candidate patches re-reads its own
// 25 pixels from global memory — no reuse across candidates, let alone
// across threads. A production NLM avoids this with an INTEGRAL IMAGE (a
// summed-area table of squared differences per candidate OFFSET, turning
// each patch SSD into 4 array reads instead of 25) — documented, not built,
// here (THEORY.md "The GPU mapping" explains why and what it would take).
// ---------------------------------------------------------------------------
__global__ void nlm_kernel(const float* __restrict__ img, int W, int H,
                           float* __restrict__ out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    const int PR = kNlmPatchRadius;                          // 2 -> 5x5 patch
    const int SR = kNlmSearchRadius;                          // 6 -> 13x13 search window
    const float patchN = static_cast<float>((2 * PR + 1) * (2 * PR + 1));  // 25: patch pixel count
    const float invH2 = 1.0f / (kNlmH * kNlmH);

    float wsum = 0.0f, vsum = 0.0f;
    for (int oy = -SR; oy <= SR; ++oy) {
        const int qy = clamp_coord(y + oy, H);
        for (int ox = -SR; ox <= SR; ++ox) {
            const int qx = clamp_coord(x + ox, W);

            // Patch distance: mean squared difference over the 5x5
            // neighborhoods of p=(x,y) and q=(qx,qy) — every patch pixel is
            // independently clamped (a patch straddling the image border
            // near a corner needs its OWN clamp per pixel, not just the
            // patch center's).
            float ssd = 0.0f;
            for (int py = -PR; py <= PR; ++py) {
                const int ay = clamp_coord(y + py, H), by = clamp_coord(qy + py, H);
                for (int px = -PR; px <= PR; ++px) {
                    const int ax = clamp_coord(x + px, W), bx = clamp_coord(qx + px, W);
                    const float diff = img[ay * W + ax] - img[by * W + bx];
                    ssd += diff * diff;
                }
            }
            const float patch_dist = ssd / patchN;             // mean squared diff (DN^2)
            const float w = expf(-patch_dist * invH2);          // the NLM weight
            const float qv = img[qy * W + qx];
            wsum += w;
            vsum += w * qv;
        }
    }
    out[y * W + x] = vsum / wsum;
}

// ===========================================================================
// BM3D-LITE — shared device helpers (block-matching, the 2-D DCT-II via a
// dynamically-built 8x8 basis, and the 1-D Haar transform across a 16-stack).
// Each is a small, textbook building block; reference_cpu.cpp re-derives
// the SAME three transforms independently (per the twin-independence
// ruling, kernels.cuh's file header) — sharing them here would make the
// GPU-vs-CPU VERIFY comparison blind to a bug living inside any one of them.
// ===========================================================================

// dct8_basis — build the 8x8 orthonormal DCT-II basis matrix:
//   C[k][n] = alpha(k) * cos(pi/8 * (n+0.5) * k),  alpha(0)=sqrt(1/8),
//                                                   alpha(k>0)=sqrt(2/8).
// Recomputed by EVERY thread via cosf (64 transcendental calls) rather than
// loaded from __constant__ memory — a documented simplification: the cost
// is negligible next to this kernel's block-matching work (169 candidates x
// 64-pixel SSDs, THEORY.md's dominant term), and keeping the kernel
// self-contained avoids an extra host-side cudaMemcpyToSymbol setup step
// for a one-project teaching kernel (README "Limitations & honesty" names
// __constant__ memory as the production fix, following 09.01's precedent).
__device__ __forceinline__ void dct8_basis(float basis[8][8])
{
    const float kPi = 3.14159265358979323846f;
    for (int k = 0; k < 8; ++k) {
        const float alpha = (k == 0) ? sqrtf(1.0f / 8.0f) : sqrtf(2.0f / 8.0f);
        for (int n = 0; n < 8; ++n)
            basis[k][n] = alpha * cosf(kPi / 8.0f * (static_cast<float>(n) + 0.5f) * static_cast<float>(k));
    }
}

// dct2d_forward — X = C * P * C^T (separable: rows then columns). Since C
// is orthonormal, this transform preserves total energy (Parseval) and, for
// i.i.d. noise, maps noise variance sigma^2 in pixel space to variance
// sigma^2 in EVERY output coefficient (THEORY.md "Numerical considerations"
// proves this — the property kBm3dThreshold's whole design leans on).
__device__ __forceinline__ void dct2d_forward(const float basis[8][8], float p[8][8])
{
    float tmp[8][8];
    for (int u = 0; u < 8; ++u)
        for (int col = 0; col < 8; ++col) {
            float s = 0.0f;
            for (int n = 0; n < 8; ++n) s += basis[u][n] * p[n][col];   // (C * P)[u][col]
            tmp[u][col] = s;
        }
    for (int u = 0; u < 8; ++u)
        for (int v = 0; v < 8; ++v) {
            float s = 0.0f;
            for (int n = 0; n < 8; ++n) s += tmp[u][n] * basis[v][n];   // (.. * C^T)[u][v]
            p[u][v] = s;
        }
}

// dct2d_inverse — P = C^T * X * C, the exact inverse of dct2d_forward
// because C is orthonormal (C^{-1} = C^T) — no separate "inverse basis" is
// needed, just the transpose usage (index-swapped multiplies) below.
__device__ __forceinline__ void dct2d_inverse(const float basis[8][8], float p[8][8])
{
    float tmp[8][8];
    for (int row = 0; row < 8; ++row)
        for (int v = 0; v < 8; ++v) {
            float s = 0.0f;
            for (int u = 0; u < 8; ++u) s += basis[u][row] * p[u][v];   // (C^T * X)[row][v]
            tmp[row][v] = s;
        }
    for (int row = 0; row < 8; ++row)
        for (int col = 0; col < 8; ++col) {
            float s = 0.0f;
            for (int v = 0; v < 8; ++v) s += tmp[row][v] * basis[v][col];  // (.. * C)[row][col]
            p[row][col] = s;
        }
}

// haar_forward16 / haar_inverse16 — the FULL (4-level) orthonormal 1-D Haar
// decomposition of a length-16 vector, the "3rd dimension" transform BM3D
// applies ACROSS the stack (one call per (u,v) DCT-coefficient position,
// kBm3dStackSize=16=2^4 values deep). Standard fast-wavelet pyramid: each
// level halves the working length, replacing PAIRS (a,b) with a
// (sum,difference)/sqrt(2) (orthonormal — sqrt(2) is the norm-preserving
// scale, THEORY.md derives it) and leaves the difference half of the
// buffer untouched by later levels — the classic Mallat layout, so a
// single length-16 buffer holds the whole multi-resolution decomposition
// with no extra storage.
__device__ __forceinline__ void haar_forward16(float v[16])
{
    float tmp[16];
    for (int len = 16; len > 1; len /= 2) {
        const int half = len / 2;
        for (int i = 0; i < half; ++i) {
            const float a = v[2 * i], b = v[2 * i + 1];
            tmp[i] = (a + b) * 0.70710678118654752f;          // 1/sqrt(2): running average (low-pass)
            tmp[half + i] = (a - b) * 0.70710678118654752f;   // this level's detail (high-pass)
        }
        for (int i = 0; i < len; ++i) v[i] = tmp[i];
    }
}

// haar_inverse16 — undo haar_forward16 by replaying the levels in REVERSE
// (length 2 up to 16), each step reconstructing a doubled-length prefix
// from its (average, detail) halves — the exact algebraic inverse of the
// forward step above (verified for length 2 in kernels.cuh's file header
// derivation and by induction for the full pyramid; main.cu's VERIFY stage
// also confirms it empirically via GPU-vs-CPU agreement on real data).
__device__ __forceinline__ void haar_inverse16(float v[16])
{
    float tmp[16];
    for (int len = 2; len <= 16; len *= 2) {
        const int half = len / 2;
        for (int i = 0; i < half; ++i) {
            const float a = v[i], d = v[half + i];
            tmp[2 * i] = (a + d) * 0.70710678118654752f;
            tmp[2 * i + 1] = (a - d) * 0.70710678118654752f;
        }
        for (int i = 0; i < len; ++i) v[i] = tmp[i];
    }
}

// ===========================================================================
// 5) BM3D-LITE, stage 1 — one thread per REFERENCE GROUP (kBm3dNumGroups =
// 1,813 for this project's 200x150 frame — a MUCH smaller grid than the
// per-pixel kernels above; THEORY.md "The GPU mapping" contrasts this
// "one thread = one whole group's work" pattern with MPPI's "one thread =
// one whole rollout" (08.01) — same shape, different payload).
//
// Per thread: (a) block-match up to 169 candidate 8x8 patches around this
// group's reference anchor, keep the best kBm3dStackSize=16 by SSD; (b)
// gather them into a local 16x8x8 stack; (c) 2-D DCT each patch, 1-D Haar
// across the stack, HARD-THRESHOLD every one of the 1,024 coefficients,
// invert both transforms; (d) atomically scatter-accumulate the 16
// denoised patches — each at ITS OWN matched location, not just the
// reference location — into shared out_sum/out_weight buffers, weighted by
// this group's sparsity (fewer surviving coefficients = more confidently
// denoised = trusted more, the standard BM3D aggregation weight).
//
// Register/local-memory honesty (THEORY.md "The GPU mapping" repeats this):
// the local stack[16][8][8] alone is 1,024 floats (4 KiB) per thread — far
// beyond a healthy register budget, so nvcc SPILLS it to per-thread local
// memory (backed by global memory, L1/L2-cached). For this project's tiny
// 200x150 demo (1,813 threads total) that is still fast in absolute terms;
// it is NOT how a production BM3D-GPU implementation would shape this
// kernel (one BLOCK per group, threads cooperating over shared memory, is
// the fix — named here and in THEORY.md, not built, an honest scope cut
// consistent with this milestone's "-lite" name).
//
// Why atomics are unavoidable: with an 8-pixel patch on a 4-pixel stride,
// interior pixels are covered by MANY reference groups' matched patches
// (that is BM3D's whole point — richer aggregation than one filter pass
// per pixel), and those groups run on different threads with no ordering
// guarantee, so accumulation must be atomic (THEORY.md "Numerical
// considerations" discusses the resulting non-associative summation order
// and why main.cu's VERIFY tolerance for this method is therefore looser
// than bilateral's/NLM's deterministic pointwise kernels).
// ---------------------------------------------------------------------------
__global__ void bm3d_group_kernel(const float* __restrict__ img, int W, int H,
                                  float* __restrict__ out_sum, float* __restrict__ out_weight)
{
    const int numX = bm3d_num_positions(W);
    const int numY = bm3d_num_positions(H);
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's GROUP index
    if (gid >= numX * numY) return;

    const int ix = gid % numX, iy = gid / numX;
    const int gx = bm3d_position(ix, W);   // this group's reference-patch anchor (top-left), clamped in-bounds
    const int gy = bm3d_position(iy, H);

    // ---- (a) block matching: keep the best kBm3dStackSize by ascending SSD.
    // Small local top-K arrays (K=16) maintained by linear insertion — O(169
    // candidates x 16 slots) = 2,704 comparisons worst case, trivial next to
    // the SSD work itself.
    float best_ssd[kBm3dStackSize];
    int best_cx[kBm3dStackSize], best_cy[kBm3dStackSize];
    for (int i = 0; i < kBm3dStackSize; ++i) best_ssd[i] = FLT_MAX;   // "no candidate yet" sentinel

    for (int oy = -kBm3dSearchRadius; oy <= kBm3dSearchRadius; ++oy) {
        const int cy = clamp_coord(gy + oy, H - kBm3dPatch + 1);   // clamp so the WHOLE patch stays in-bounds
        for (int ox = -kBm3dSearchRadius; ox <= kBm3dSearchRadius; ++ox) {
            const int cx = clamp_coord(gx + ox, W - kBm3dPatch + 1);

            float ssd = 0.0f;
            for (int r = 0; r < kBm3dPatch; ++r)
                for (int c = 0; c < kBm3dPatch; ++c) {
                    const float diff = img[(gy + r) * W + (gx + c)] - img[(cy + r) * W + (cx + c)];
                    ssd += diff * diff;
                }

            // Insert into the sorted top-16 if this candidate beats the
            // current worst kept (best_ssd[kBm3dStackSize-1], since the
            // array is kept sorted ascending at all times).
            if (ssd < best_ssd[kBm3dStackSize - 1]) {
                int slot = kBm3dStackSize - 1;
                while (slot > 0 && best_ssd[slot - 1] > ssd) {
                    best_ssd[slot] = best_ssd[slot - 1];
                    best_cx[slot] = best_cx[slot - 1];
                    best_cy[slot] = best_cy[slot - 1];
                    --slot;
                }
                best_ssd[slot] = ssd;
                best_cx[slot] = cx;
                best_cy[slot] = cy;
            }
        }
    }

    // ---- (b) gather the 16 matched patches into a local stack. -------------
    float stack[kBm3dStackSize][kBm3dPatch][kBm3dPatch];
    for (int p = 0; p < kBm3dStackSize; ++p)
        for (int r = 0; r < kBm3dPatch; ++r)
            for (int c = 0; c < kBm3dPatch; ++c)
                stack[p][r][c] = img[(best_cy[p] + r) * W + (best_cx[p] + c)];

    // ---- (c) forward transform: 2-D DCT per patch, then 1-D Haar across
    // the stack (one call per of the 64 (u,v) coefficient positions). -------
    float basis[8][8];
    dct8_basis(basis);
    for (int p = 0; p < kBm3dStackSize; ++p) dct2d_forward(basis, stack[p]);

    for (int u = 0; u < kBm3dPatch; ++u)
        for (int v = 0; v < kBm3dPatch; ++v) {
            float vec[kBm3dStackSize];
            for (int p = 0; p < kBm3dStackSize; ++p) vec[p] = stack[p][u][v];
            haar_forward16(vec);
            for (int p = 0; p < kBm3dStackSize; ++p) stack[p][u][v] = vec[p];
        }

    // ---- HARD THRESHOLD every one of the 1,024 transform-domain
    // coefficients (both transforms are orthonormal, so noise variance is
    // preserved coefficient-for-coefficient — kBm3dThreshold applies
    // uniformly, no per-position scaling needed; see dct2d_forward's
    // comment and THEORY.md "Numerical considerations"). Count survivors
    // for the sparsity-based aggregation weight below.
    int nonzero = 0;
    for (int p = 0; p < kBm3dStackSize; ++p)
        for (int u = 0; u < kBm3dPatch; ++u)
            for (int v = 0; v < kBm3dPatch; ++v) {
                if (fabsf(stack[p][u][v]) < kBm3dThreshold) stack[p][u][v] = 0.0f;
                else ++nonzero;
            }

    // ---- (d) inverse transform: Haar first (undoing the LAST forward
    // step first — standard transform-pair symmetry), then 2-D DCT. --------
    for (int u = 0; u < kBm3dPatch; ++u)
        for (int v = 0; v < kBm3dPatch; ++v) {
            float vec[kBm3dStackSize];
            for (int p = 0; p < kBm3dStackSize; ++p) vec[p] = stack[p][u][v];
            haar_inverse16(vec);
            for (int p = 0; p < kBm3dStackSize; ++p) stack[p][u][v] = vec[p];
        }
    for (int p = 0; p < kBm3dStackSize; ++p) dct2d_inverse(basis, stack[p]);

    // ---- scatter-accumulate: sparsity-weighted (BM3D's standard hard-
    // threshold aggregation weight — a group whose coefficients mostly
    // vanished is judged CONFIDENTLY denoised and trusted more). Every one
    // of the 16 patches is written at ITS OWN matched location, not the
    // reference location — the collaborative-filtering step (THEORY.md
    // "The math") that makes BM3D more than a per-block average.
    const float weight = 1.0f / (1.0f + static_cast<float>(nonzero));
    for (int p = 0; p < kBm3dStackSize; ++p)
        for (int r = 0; r < kBm3dPatch; ++r)
            for (int c = 0; c < kBm3dPatch; ++c) {
                const int px = best_cx[p] + c, py = best_cy[p] + r;
                atomicAdd(&out_sum[py * W + px], weight * stack[p][r][c]);
                atomicAdd(&out_weight[py * W + px], weight);
            }
}

// ---------------------------------------------------------------------------
// 6) BM3D-LITE, stage 2 — finalize: one thread per PIXEL (a plain map,
// the simplest pattern in this file, deliberately, right after the most
// complex one). out_weight[i] is >0 for every pixel by construction (the
// bm3d_num_positions/bm3d_position coverage guarantee, kernels.cuh Section
// 6) — the img[] fallback is a defensive guard, never expected to trigger,
// and documented as such rather than silently hidden.
// ---------------------------------------------------------------------------
__global__ void bm3d_finalize_kernel(const float* __restrict__ img,
                                     const float* __restrict__ out_sum,
                                     const float* __restrict__ out_weight,
                                     int n, float* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float w = out_weight[i];
    out[i] = (w > 1e-6f) ? (out_sum[i] / w) : img[i];
}

// ===========================================================================
// Host launch wrappers (declared in kernels.cuh). Each owns its grid/block
// math, its mandatory post-launch error check, and — for BM3D-lite — the
// two scratch buffers the two-stage pipeline needs.
// ===========================================================================

void launch_bilateral_naive(const float* d_img, int W, int H, float* d_out)
{
    const dim3 block(16, 16);                                          // warp-multiple 2-D block, repo default
    const dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    bilateral_naive_kernel<<<grid, block>>>(d_img, W, H, d_out);
    CUDA_CHECK_LAST_ERROR("bilateral_naive_kernel launch");
}

void launch_bilateral_tiled(const float* d_img, int W, int H, float* d_out)
{
    const dim3 block(16, 16);
    const dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    const int tileW = static_cast<int>(block.x) + 2 * kBilateralRadius;   // 24
    const int tileH = static_cast<int>(block.y) + 2 * kBilateralRadius;   // 24
    const size_t shmem_bytes = static_cast<size_t>(tileW) * static_cast<size_t>(tileH) * sizeof(float);
    bilateral_tiled_kernel<<<grid, block, shmem_bytes>>>(d_img, W, H, d_out);
    CUDA_CHECK_LAST_ERROR("bilateral_tiled_kernel launch");
}

void launch_gaussian_blur(const float* d_img, int W, int H, float* d_out)
{
    const dim3 block(16, 16);
    const dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    gaussian_blur_kernel<<<grid, block>>>(d_img, W, H, d_out);
    CUDA_CHECK_LAST_ERROR("gaussian_blur_kernel launch");
}

void launch_nlm(const float* d_img, int W, int H, float* d_out)
{
    const dim3 block(16, 16);
    const dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    nlm_kernel<<<grid, block>>>(d_img, W, H, d_out);
    CUDA_CHECK_LAST_ERROR("nlm_kernel launch");
}

void launch_bm3d_lite(const float* d_img, int W, int H, float* d_out)
{
    // Two kN-sized scratch accumulators, owned entirely by this call (the
    // launch_mppi_rollouts precedent, 08.01: a self-contained, stateless
    // wrapper — alloc/free cost here is trivial next to the kernel work).
    float* d_sum = nullptr;
    float* d_weight = nullptr;
    const size_t bytes = static_cast<size_t>(W) * static_cast<size_t>(H) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_sum, bytes));
    CUDA_CHECK(cudaMalloc(&d_weight, bytes));
    CUDA_CHECK(cudaMemset(d_sum, 0, bytes));
    CUDA_CHECK(cudaMemset(d_weight, 0, bytes));

    // Stage 1: one thread per reference GROUP. A 1-D grid — the group count
    // (1,813 for this project's fixed 200x150 frame) is far smaller than
    // the pixel count, so a 1-D launch is simpler and just as correct as a
    // 2-D one here. block=64 (not the usual 256): each thread's local
    // footprint is large (kernels.cuh Section 6 / this file's kernel
    // header), so a smaller block keeps per-SM resident-thread counts
    // sane despite the local-memory spill.
    const int numX = bm3d_num_positions(W);
    const int numY = bm3d_num_positions(H);
    const int numGroups = numX * numY;
    const int block1 = 64;
    const int grid1 = (numGroups + block1 - 1) / block1;
    bm3d_group_kernel<<<grid1, block1>>>(d_img, W, H, d_sum, d_weight);
    CUDA_CHECK_LAST_ERROR("bm3d_group_kernel launch");

    // Stage 2: one thread per pixel, the ordinary 256-thread map.
    const int n = W * H;
    const int block2 = 256;
    const int grid2 = (n + block2 - 1) / block2;
    bm3d_finalize_kernel<<<grid2, block2>>>(d_img, d_sum, d_weight, n, d_out);
    CUDA_CHECK_LAST_ERROR("bm3d_finalize_kernel launch");

    CUDA_CHECK(cudaFree(d_sum));
    CUDA_CHECK(cudaFree(d_weight));
}
