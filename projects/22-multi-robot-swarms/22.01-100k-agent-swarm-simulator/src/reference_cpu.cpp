// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 22.01
//                     (100k-agent swarm simulator: flocking, pheromone
//                     grids, stigmergy)
//
// One job in this project (declared in kernels.cuh): swarm_step_cpu — ONE
// full simulation step, computed the DUMBEST correct way:
//
//   * neighbor search by BRUTE FORCE — every agent tests every other agent
//     against the interaction radius, O(N^2). This is deliberately a
//     DIFFERENT algorithm from the GPU's uniform-grid gather (07.09's
//     exact-oracle pattern): if the counting sort, the exclusive scan, the
//     scatter, or the 3x3 gather has ANY bug — wrong cell index, missed
//     bin, off-by-one in starts — the two paths see different neighbor
//     SETS and the lockstep comparison in main.cu fails loudly on step 1.
//     An oracle that reused the grid would inherit the grid's bugs.
//   * the pheromone stencil as a plain double loop over cells — a
//     line-by-line twin of pheromone_step_kernel.
//
// The per-agent rule math (cell_coord / accumulate_neighbor / finish_agent)
// is a textual TWIN of kernels.cu — diff the files: the functions are
// identical minus __device__ qualifiers. Only the neighbor ITERATION
// differs (index order here, bin order there), which is why the comparison
// is tolerance-based, not bitwise: float sums taken in different orders
// differ in their last bits (THEORY.md §numerics; the tolerances in
// kernels.cuh carry ~100x headroom over that).
//
// Why O(N^2) is acceptable HERE: the verify stage runs it at kVerifyN=4096
// (16.8M pair tests/step — seconds of CPU time for 100 steps), never at the
// headline N=100,000 (10^10 pair tests/step — the number that motivates the
// whole grid structure; README §what-this-computes).
//
// Rules for this file: plain C++17, no CUDA headers beyond kernels.cuh's
// constants, no hand-vectorization, no OpenMP, no cleverness. If the
// reference is clever, it can be wrong — and then the oracle lies.
//
// Read this after: kernels.cuh.  Read this beside: kernels.cu.
// ===========================================================================

#include "kernels.cuh"   // shared constants, layouts, signatures

#include <cmath>         // std::sqrt/std::floor (float overloads) — the host
                         // spellings of the kernel's sqrtf/floorf; both are
                         // IEEE correctly-rounded, so same input -> same bits

// ---------------------------------------------------------------------------
// Host twins of the device helpers (see kernels.cu for the full commentary —
// not repeated here; the MATH must stay line-for-line identical).
// ---------------------------------------------------------------------------

// Twin of kernels.cu cell_coord: position (m) -> clamped grid index. One
// multiply + one floor, both correctly rounded => bit-identical cells on
// both paths for identical position bits (the property the grid-vs-brute-
// force equivalence rests on).
static int cell_coord(float p)
{
    int c = static_cast<int>(std::floor(p * kInvCell));
    if (c < 0) c = 0;
    if (c > kGridDim - 1) c = kGridDim - 1;
    return c;
}

// Twin of kernels.cu NeighborAccum (same fields, same meanings, same units).
struct NeighborAccum {
    float w_sum;        // sum of hat weights w = 1 - d/r (unitless)
    float avx, avy;     // weighted sum of neighbor velocities (m/s * weight)
    float cenx, ceny;   // weighted sum of neighbor offsets (m * weight)
    float sepx, sepy;   // separation push (unit-direction-weighted)
    int   nbr;          // plain neighbor count
};

// Twin of kernels.cu accumulate_neighbor — the smooth (hat-weighted) rule
// sums; every contribution goes to zero at the radius so ulp-level rounding
// can never flip a border neighbor into an O(1) force jump.
static void accumulate_neighbor(float pxi, float pyi,
                                float pxj, float pyj, float vxj, float vyj,
                                NeighborAccum& a)
{
    const float dx = pxj - pxi;
    const float dy = pyj - pyi;
    const float d2 = dx * dx + dy * dy;
    if (d2 >= kRNeighbor * kRNeighbor) return;

    const float d = std::sqrt(d2);
    const float w = 1.0f - d * kInvRNb;
    a.w_sum += w;
    a.avx  += w * vxj;   a.avy  += w * vyj;
    a.cenx += w * dx;    a.ceny += w * dy;
    a.nbr  += 1;

    if (d < kRSep) {
        const float s = (1.0f - d * kInvRSep) / std::fmax(d, 1e-6f);
        a.sepx -= s * dx;
        a.sepy -= s * dy;
    }
}

// Twin of kernels.cu finish_agent — wall force, rules, pheromone pull,
// clamped semi-implicit Euler, arena clamp, alignment score. See kernels.cu
// for the design commentary on every block.
static void finish_agent(float pxi, float pyi, float vxi, float vyi,
                         const NeighborAccum& a,
                         float gx, float gy,
                         float* px_o, float* py_o, float* vx_o, float* vy_o,
                         float* score_o)
{
    float ax = 0.0f, ay = 0.0f;
    if (pxi < kWallMargin)          ax += kWWall * (1.0f - pxi / kWallMargin);
    if (pxi > kArena - kWallMargin) ax -= kWWall * (1.0f - (kArena - pxi) / kWallMargin);
    if (pyi < kWallMargin)          ay += kWWall * (1.0f - pyi / kWallMargin);
    if (pyi > kArena - kWallMargin) ay -= kWWall * (1.0f - (kArena - pyi) / kWallMargin);

    float score = kNoNeighborScore;
    if (a.w_sum > 0.0f) {
        const float inv = 1.0f / a.w_sum;
        const float mvx = a.avx * inv;
        const float mvy = a.avy * inv;
        ax += kWAli * (mvx - vxi) + kWCoh * (a.cenx * inv);
        ay += kWAli * (mvy - vyi) + kWCoh * (a.ceny * inv);

        const float ni = std::sqrt(vxi * vxi + vyi * vyi);
        const float nm = std::sqrt(mvx * mvx + mvy * mvy);
        if (ni > 1e-6f && nm > 1e-6f)
            score = (vxi * mvx + vyi * mvy) / (ni * nm);
    }
    ax += kWSep * a.sepx;
    ay += kWSep * a.sepy;

    ax += kWPher * gx;
    ay += kWPher * gy;

    const float aa = ax * ax + ay * ay;
    if (aa > kAMax * kAMax) {
        const float s = kAMax / std::sqrt(aa);
        ax *= s;  ay *= s;
    }

    float vxn = vxi + ax * kDt;
    float vyn = vyi + ay * kDt;

    const float vv = vxn * vxn + vyn * vyn;
    if (vv > kVMax * kVMax) {
        const float s = kVMax / std::sqrt(vv);
        vxn *= s;  vyn *= s;
    } else if (vv < kVMin * kVMin) {
        const float sp = std::sqrt(vv);
        if (sp > 1e-6f) {
            const float s = kVMin / sp;
            vxn *= s;  vyn *= s;
        } else {
            vxn = kVMin;  vyn = 0.0f;
        }
    }

    float pxn = pxi + vxn * kDt;
    float pyn = pyi + vyn * kDt;
    pxn = std::fmin(std::fmax(pxn, 0.0f), kArena);
    pyn = std::fmin(std::fmax(pyn, 0.0f), kArena);

    *px_o = pxn;  *py_o = pyn;
    *vx_o = vxn;  *vy_o = vyn;
    *score_o = score;
}

// ---------------------------------------------------------------------------
// swarm_step_cpu — one full step: brute-force flocking + pheromone stencil.
//
// Structure mirrors the GPU pipeline's OBSERVABLE effect exactly, in the
// same order (kernels.cuh contract):
//   1. cell histogram of the CURRENT positions (needed twice: the flock
//      step's pheromone deposits and — on the GPU — the bins; here counted
//      with plain ++, no atomics needed sequentially);
//   2. per-agent: brute-force neighbor sums, pheromone gradient at the
//      agent's cell, finish_agent — all reading the step-start state
//      (matching the GPU's read-cur/write-nxt ping-pong: no agent ever
//      sees a half-updated neighbor);
//   3. per-cell: the diffuse+decay+deposit stencil into pher_o.
//
// The alignment-score output of finish_agent is discarded here — the metric
// is a demo-side observable, not part of the verified state (main.cu takes
// it from the GPU); a local sink variable keeps the twin functions
// signature-identical.
// ---------------------------------------------------------------------------
void swarm_step_cpu(int n,
                    const float* px, const float* py,
                    const float* vx, const float* vy,
                    const float* pher,
                    float* px_o, float* py_o,
                    float* vx_o, float* vy_o,
                    float* pher_o)
{
    // --- 1) cell histogram (the deposit map), sequential so no atomics ----
    // Heap-allocated once per call; at kVerifyN call rates this is noise
    // next to the O(N^2) loop below.
    static_assert(kNumCells == kGridDim * kGridDim, "grid layout contract");
    int* counts = new int[kNumCells];
    for (int c = 0; c < kNumCells; ++c) counts[c] = 0;
    for (int i = 0; i < n; ++i)
        counts[cell_coord(py[i]) * kGridDim + cell_coord(px[i])] += 1;

    // --- 2) per-agent brute force (the oracle's whole point) --------------
    for (int i = 0; i < n; ++i) {
        const float pxi = px[i], pyi = py[i];
        const float vxi = vx[i], vyi = vy[i];

        // EVERY other agent is a candidate — no grid, no bins, nothing to
        // get subtly wrong. accumulate_neighbor applies the same radius
        // test the GPU applies, so the accepted neighbor SET is identical;
        // only the summation ORDER differs (index order here vs bin order
        // there) — the tolerance's job.
        NeighborAccum acc = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0 };
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            accumulate_neighbor(pxi, pyi, px[j], py[j], vx[j], vy[j], acc);
        }

        // Pheromone gradient — identical clamped central difference.
        const int cx = cell_coord(pxi);
        const int cy = cell_coord(pyi);
        const int cxe = (cx < kGridDim - 1) ? cx + 1 : kGridDim - 1;
        const int cxw = (cx > 0) ? cx - 1 : 0;
        const int cyn = (cy < kGridDim - 1) ? cy + 1 : kGridDim - 1;
        const int cys = (cy > 0) ? cy - 1 : 0;
        const float gx = (pher[cy * kGridDim + cxe] - pher[cy * kGridDim + cxw]) * (0.5f * kInvCell);
        const float gy = (pher[cyn * kGridDim + cx] - pher[cys * kGridDim + cx]) * (0.5f * kInvCell);

        float score_sink;   // discarded (see function header)
        finish_agent(pxi, pyi, vxi, vyi, acc, gx, gy,
                     &px_o[i], &py_o[i], &vx_o[i], &vy_o[i], &score_sink);
    }

    // --- 3) pheromone stencil — twin of pheromone_step_kernel -------------
    for (int cy = 0; cy < kGridDim; ++cy) {
        for (int cx = 0; cx < kGridDim; ++cx) {
            const int c = cy * kGridDim + cx;
            const float p  = pher[c];
            // Zero-flux boundary: missing neighbors mirror the center.
            const float pn = (cy < kGridDim - 1) ? pher[c + kGridDim] : p;
            const float ps = (cy > 0)            ? pher[c - kGridDim] : p;
            const float pe = (cx < kGridDim - 1) ? pher[c + 1]        : p;
            const float pw = (cx > 0)            ? pher[c - 1]        : p;
            const float lap = pn + ps + pe + pw - 4.0f * p;
            pher_o[c] = (1.0f - kDecay) * (p + kDiffuse * lap)
                      + kDeposit * static_cast<float>(counts[c]);
        }
    }

    delete[] counts;
}
