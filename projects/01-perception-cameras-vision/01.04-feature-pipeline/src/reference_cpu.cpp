// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 01.04
//                     (Feature pipeline: FAST/Harris detection, ORB
//                     descriptors, brute-force Hamming matcher)
//
// WHY does a GPU repository ship a CPU implementation of everything? Two
// load-bearing reasons (CLAUDE.md §5) — the CORRECTNESS ORACLE (a dead-
// simple sequential version a reader can verify by eye) and the TEACHING
// BASELINE (reading this file, then kernels.cu, shows exactly what
// parallelization changed). See docs/PROJECT_TEMPLATE/src/reference_cpu.cpp
// for the repo-wide version of this argument in full, including the
// independence ruling this file follows:
//
//   * Data-LAYOUT contracts (structs, constants, the rotated-pattern
//     table, the base pair table, the per-keypoint angle-bin index) are
//     single-sourced in kernels.cuh and SHARED — divergent layouts would
//     be a bug class of their own, not "independence".
//   * The ALGORITHMIC CORE (score computation, NMS, orientation centroid,
//     descriptor bit-packing, Hamming reduction) is written TWICE,
//     independently, in the simplest possible C++ HERE.
//
// This project's specific twin strategy (see kernels.cuh's header for the
// full argument): FAST is all-integer end to end, so its score map AND its
// final (sorted) keypoint list are BIT-EXACT twins. Harris's response map
// is float, so it is a TOLERANCE twin. Orientation (intensity centroid ->
// atan2) is float, so it is a TOLERANCE twin — but the table/bin index
// consumed by describe_cpu() below is DELIBERATELY the SAME data the GPU
// path used (main.cu quantizes the GPU-measured angle into a bin ONCE and
// hands that integer to both pipelines), which is what lets the descriptor
// bits stay BIT-EXACT despite the tolerant angle upstream. Hamming
// distances are pure integer popcount + min-reduction, so they are exact.
//
// Rules for this file: plain C++17, no CUDA headers, no hand-vectorization,
// no OpenMP, no cleverness. If the reference is clever, it can be wrong,
// and then the oracle lies.
//
// Read this after: kernels.cu — then compare the two side by side.
// ===========================================================================

#include "kernels.cuh"

#include <algorithm>   // std::sort, std::min/max
#include <cmath>       // std::atan2, std::lround
#include <vector>

// ===========================================================================
// STAGE 1a: FAST-9 — independent host re-implementation of fast_score_kernel.
// Same algorithm (Bresenham ring, high-speed quad pre-filter, contiguous-
// arc test, worst-point-margin score), written as ONE sequential double
// loop instead of one GPU thread per pixel — the exact "the loop became
// threads" contrast CLAUDE.md's commenting standard asks every project to
// make legible.
// ===========================================================================
void fast_score_cpu(const uint8_t* img, int* score_out)
{
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
            const int idx = y * kW + x;

            if (x < kDetectBorder || x >= kW - kDetectBorder ||
                y < kDetectBorder || y >= kH - kDetectBorder) {
                score_out[idx] = 0;
                continue;
            }

            const int Ip = static_cast<int>(img[idx]);

            int ring[kFastCircleN];
            for (int i = 0; i < kFastCircleN; ++i) {
                ring[i] = static_cast<int>(img[(y + kFastCircleY[i]) * kW + (x + kFastCircleX[i])]);
            }

            int quad_bright = 0, quad_dark = 0;
            for (int q = 0; q < 4; ++q) {
                const int v = ring[kFastQuadIdx[q]];
                if (v > Ip + kFastThreshold) ++quad_bright;
                else if (v < Ip - kFastThreshold) ++quad_dark;
            }
            if (quad_bright < 3 && quad_dark < 3) {
                score_out[idx] = 0;
                continue;
            }

            int best_margin = 0;
            for (int polarity = 0; polarity < 2; ++polarity) {
                const bool bright = (polarity == 0);
                if (bright && quad_bright < 3) continue;
                if (!bright && quad_dark < 3) continue;

                for (int start = 0; start < kFastCircleN; ++start) {
                    int margin = 2147483647;
                    bool ok = true;
                    for (int j = 0; j < kFastArcLen; ++j) {
                        const int v = ring[(start + j) % kFastCircleN];
                        const int m = bright ? (v - Ip - kFastThreshold) : (Ip - v - kFastThreshold);
                        if (m <= 0) { ok = false; break; }
                        if (m < margin) margin = m;
                    }
                    if (ok && margin > best_margin) best_margin = margin;
                }
            }

            score_out[idx] = best_margin;
        }
    }
}

// ---------------------------------------------------------------------------
// fast_nms_select_cpu — independent host NMS + deterministic sort/truncate.
//
// Same decision rule as nms_select_fast_kernel (strict local max over the
// 3x3 neighborhood, a tie with any neighbor suppresses both), collected
// into a std::vector (a GPU kernel cannot use one — its equivalent is the
// atomic-counter compaction in kernels.cu; this is where "the same
// algorithm, twice" looks most different in SHAPE while computing the
// same answer). The final sort-by-(score desc, y asc, x asc) is the exact
// convention main.cu applies to the GPU path's (unordered, atomics-
// produced) candidate list before the two are compared — see kernels.cuh's
// header for why that convention must be identical on both sides.
//
// Parameters: score — [kW*kH] IN, already-verified-bit-exact score map.
//             out — [max_out] OUT, sorted keypoints.
//             max_out — capacity of out (kTopNFast in practice).
// Returns: number of keypoints written (<= max_out).
// ---------------------------------------------------------------------------
int fast_nms_select_cpu(const int* score, Keypoint* out, int max_out)
{
    std::vector<Keypoint> candidates;
    candidates.reserve(1024);

    for (int y = kBorder; y < kH - kBorder; ++y) {
        for (int x = kBorder; x < kW - kBorder; ++x) {
            const int s = score[y * kW + x];
            if (s <= 0) continue;

            bool is_max = true;
            for (int dy = -1; dy <= 1 && is_max; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    if (score[(y + dy) * kW + (x + dx)] >= s) { is_max = false; break; }
                }
            }
            if (!is_max) continue;

            candidates.push_back(Keypoint{ x, y, static_cast<float>(s) });
        }
    }

    // Deterministic ranking: highest score first; ties broken by (y, x) so
    // the order is reproducible independent of insertion order (which here
    // happens to already be raster order, but sorting explicitly documents
    // the CONTRACT rather than relying on that incidental fact).
    std::sort(candidates.begin(), candidates.end(), [](const Keypoint& a, const Keypoint& b) {
        if (a.score != b.score) return a.score > b.score;
        if (a.y != b.y) return a.y < b.y;
        return a.x < b.x;
    });

    const int n = std::min(static_cast<int>(candidates.size()), max_out);
    for (int i = 0; i < n; ++i) out[i] = candidates[static_cast<size_t>(i)];
    return n;
}

// ===========================================================================
// STAGE 1b: Harris — independent host Sobel + structure-tensor response.
// ===========================================================================
void sobel_gradient_cpu(const uint8_t* img, float* gx_out, float* gy_out)
{
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
            const int idx = y * kW + x;
            if (x < kSobelBorder || x >= kW - kSobelBorder ||
                y < kSobelBorder || y >= kH - kSobelBorder) {
                gx_out[idx] = 0.0f;
                gy_out[idx] = 0.0f;
                continue;
            }
            // Read the 3x3 neighborhood via a small local array — a
            // deliberately different shape from kernels.cu's nine named
            // locals (both are "read the 3x3 window"; the point of an
            // independent implementation is to catch INDEXING mistakes,
            // and a loop-based read exercises that differently than nine
            // hand-written offsets do).
            int p[3][3];
            for (int wy = -1; wy <= 1; ++wy)
                for (int wx = -1; wx <= 1; ++wx)
                    p[wy + 1][wx + 1] = static_cast<int>(img[(y + wy) * kW + (x + wx)]);

            const int gx = (p[0][2] + 2 * p[1][2] + p[2][2]) - (p[0][0] + 2 * p[1][0] + p[2][0]);
            const int gy = (p[2][0] + 2 * p[2][1] + p[2][2]) - (p[0][0] + 2 * p[0][1] + p[0][2]);
            gx_out[idx] = static_cast<float>(gx);
            gy_out[idx] = static_cast<float>(gy);
        }
    }
}

void harris_response_cpu(const float* gx, const float* gy, float* response_out)
{
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
            const int idx = y * kW + x;
            if (x < kDetectBorder || x >= kW - kDetectBorder ||
                y < kDetectBorder || y >= kH - kDetectBorder) {
                response_out[idx] = 0.0f;
                continue;
            }
            double sxx = 0.0, syy = 0.0, sxy = 0.0;   // double accumulation: an independent numerical PATH from
                                                        // kernels.cu's float accumulation, on purpose (see main.cu's
                                                        // VERIFY tolerance derivation for why this is allowed to differ)
            for (int wy = -kHarrisWinRadius; wy <= kHarrisWinRadius; ++wy) {
                for (int wx = -kHarrisWinRadius; wx <= kHarrisWinRadius; ++wx) {
                    const int widx = (y + wy) * kW + (x + wx);
                    const double gxv = static_cast<double>(gx[widx]);
                    const double gyv = static_cast<double>(gy[widx]);
                    sxx += gxv * gxv;
                    syy += gyv * gyv;
                    sxy += gxv * gyv;
                }
            }
            const double det = sxx * syy - sxy * sxy;
            const double trace = sxx + syy;
            const double r = det - static_cast<double>(kHarrisK) * trace * trace;
            response_out[idx] = static_cast<float>(r);
        }
    }
}

// ===========================================================================
// STAGE 2a: orientation — independent host intensity centroid.
// ===========================================================================
void orientation_cpu(const uint8_t* img, const Keypoint* kps, int n, float* theta_out)
{
    const int R = kOrientPatchRadius;
    for (int i = 0; i < n; ++i) {
        const int cx = kps[i].x;
        const int cy = kps[i].y;

        long long m10 = 0, m01 = 0;   // exact integer accumulators, same magnitude argument as orientation_kernel
        for (int dy = -R; dy <= R; ++dy) {
            const int dy2 = dy * dy;
            for (int dx = -R; dx <= R; ++dx) {
                if (dx * dx + dy2 > R * R) continue;
                const long long I = static_cast<long long>(img[(cy + dy) * kW + (cx + dx)]);
                m10 += dx * I;
                m01 += dy * I;
            }
        }
        // std::atan2(float,float) resolves to the float overload (C++11
        // <cmath>), matching atan2f's precision family on the GPU side —
        // any remaining difference is purely the two platforms' distinct
        // libm/hardware atan2 IMPLEMENTATIONS, exactly what the "orientation
        // tolerance" twin in main.cu is designed to accept (see kernels.cuh
        // header: the tolerance is ~200x smaller than the 12-degree
        // orientation-bin width consumed downstream, so it never matters
        // for descriptor bit-exactness).
        theta_out[i] = std::atan2(static_cast<float>(m01), static_cast<float>(m10));
    }
}

// ===========================================================================
// STAGE 2b: describe — independent host rBRIEF bit-packing.
//
// table/bin_idx are SHARED, already-validated data (see this file's header
// and kernels.cuh) — what is independently re-derived here is the LOOP
// that walks 256 pairs, samples two pixels, compares, and packs a bit.
// ===========================================================================
void describe_cpu(const uint8_t* img, const Keypoint* kps, const int* bin_idx, int n,
                  const RotatedOffset* table, OrbDescriptor* desc_out)
{
    for (int i = 0; i < n; ++i) {
        const int cx = kps[i].x;
        const int cy = kps[i].y;
        const RotatedOffset* row = table + bin_idx[i] * kOrbNumPairs;

        OrbDescriptor d;
        for (int w = 0; w < kOrbDescWords; ++w) d.w[w] = 0u;

        for (int k = 0; k < kOrbNumPairs; ++k) {
            const RotatedOffset& o = row[k];
            int xa = cx + o.dx1, ya = cy + o.dy1;
            int xb = cx + o.dx2, yb = cy + o.dy2;
            xa = std::min(std::max(xa, 0), kW - 1); ya = std::min(std::max(ya, 0), kH - 1);
            xb = std::min(std::max(xb, 0), kW - 1); yb = std::min(std::max(yb, 0), kH - 1);

            const int Ia = static_cast<int>(img[ya * kW + xa]);
            const int Ib = static_cast<int>(img[yb * kW + xb]);
            if (Ia < Ib) {
                const int word = k / 32;
                const int bit  = k % 32;
                d.w[word] |= (1u << bit);
            }
        }
        desc_out[i] = d;
    }
}

// ===========================================================================
// STAGE 3: brute-force Hamming — independent host popcount + reduction.
// popcount32_portable (kernels.cuh) is the SWAR bit-trick "what the
// hardware POPC instruction is doing under the hood" — see that function's
// comment for the full derivation.
// ===========================================================================
void hamming_match_cpu(const OrbDescriptor* query, int nQuery, const OrbDescriptor* train, int nTrain,
                       int* best1_dist, int* best1_idx, int* best2_dist, int* best2_idx)
{
    for (int qi = 0; qi < nQuery; ++qi) {
        const OrbDescriptor& q = query[qi];
        int b1 = kOrbNumPairs + 1, i1 = -1;
        int b2 = kOrbNumPairs + 1, i2 = -1;

        for (int ti = 0; ti < nTrain; ++ti) {
            const OrbDescriptor& t = train[ti];
            int dist = 0;
            for (int w = 0; w < kOrbDescWords; ++w) {
                dist += popcount32_portable(q.w[w] ^ t.w[w]);
            }
            if (dist < b1) {
                b2 = b1; i2 = i1;
                b1 = dist; i1 = ti;
            } else if (dist < b2) {
                b2 = dist; i2 = ti;
            }
        }

        best1_dist[qi] = b1; best1_idx[qi] = i1;
        best2_dist[qi] = b2; best2_idx[qi] = i2;
    }
}
