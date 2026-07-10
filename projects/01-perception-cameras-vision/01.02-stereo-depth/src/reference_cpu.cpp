// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 01.02
//                     Stereo depth: block matching, then Semi-Global
//                     Matching (SGM) kernels
//
// WHY does a GPU repository ship a CPU implementation of everything?
// ------------------------------------------------------------------
// Two load-bearing reasons (CLAUDE.md §5), and this project leans on both
// harder than most because EVERY function here is pure integer arithmetic:
//
//   1) It is the CORRECTNESS ORACLE, and an EXACT one. Census comparisons,
//      Hamming distances (popcount of XOR), and the SGM min/add recurrence
//      involve no floating point anywhere — so unlike most GPU-vs-CPU
//      stories in this repo (which need a relative tolerance for FMA/trig
//      rounding differences), this project's VERIFY stage in main.cu
//      demands BIT-FOR-BIT equality between the GPU and this file. If they
//      ever disagree, it is unambiguously a bug — an indexing mistake, an
//      off-by-one in the D-major stride, a border case handled differently
//      — never "rounding".
//
//   2) It is the TEACHING BASELINE. Every function here is a direct
//      sequential transcription of its __device__/__global__ twin in
//      kernels.cu — same variable names, same order of operations, so a
//      reader can diff the two files side by side and see EXACTLY what
//      "putting it on the GPU" changed (spoiler, same as every project in
//      this repo: independent loop iterations became independent threads;
//      the arithmetic inside each iteration is identical).
//
// Rules for this file: plain C++17, no CUDA headers, no SIMD intrinsics, no
// OpenMP. If the reference is clever, it can be wrong, and then the oracle
// lies (CLAUDE.md §5). Compiled by the HOST compiler (cl.exe); kernels.cuh
// has no __CUDACC__-gated declarations for this project (see that file) so
// no host/device fencing is needed here.
//
// Read this after: kernels.cu — then compare the two side by side.
// ===========================================================================

#include "kernels.cuh"

// ---------------------------------------------------------------------------
// popcount64_cpu — population count (number of set bits) of a 64-bit word,
// written out BY HAND as the classic SWAR ("SIMD within a register") bit-
// trick: mask-and-add pairs of bits, then nibbles, then bytes, then sum the
// bytes with one multiply. This is deliberately NOT std::popcount (C++20,
// outside this repo's C++17 floor) and not a compiler builtin — it is the
// literal "what would it take to write by hand" answer CLAUDE.md §1 asks
// every library call to carry; the GPU side spends one hardware POPC
// instruction (__popcll) computing the exact same 48-bit-relevant count.
// ---------------------------------------------------------------------------
static inline int popcount64_cpu(unsigned long long v)
{
    v = v - ((v >> 1) & 0x5555555555555555ULL);                       // pairs: count of set bits in each 2-bit group
    v = (v & 0x3333333333333333ULL) + ((v >> 2) & 0x3333333333333333ULL); // nibbles
    v = (v + (v >> 4)) & 0x0F0F0F0F0F0F0F0FULL;                        // bytes (each byte now holds its own popcount, <= 8)
    return static_cast<int>((v * 0x0101010101010101ULL) >> 56);        // horizontal byte sum via multiply-by-ones + top-byte extract
}

// ===========================================================================
// 1) census_cpu — line-by-line twin of census_kernel (kernels.cu).
// ===========================================================================
void census_cpu(const unsigned char* img, unsigned long long* census, int W, int H)
{
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int idx = y * W + x;
            if (x < kCensusHalf || x >= W - kCensusHalf || y < kCensusHalf || y >= H - kCensusHalf) {
                census[idx] = kCensusInvalid;
                continue;
            }
            const unsigned char center = img[idx];
            unsigned long long bits = 0ULL;
            int bit = 0;
            for (int wy = -kCensusHalf; wy <= kCensusHalf; ++wy) {
                for (int wx = -kCensusHalf; wx <= kCensusHalf; ++wx) {
                    if (wx == 0 && wy == 0) continue;
                    const unsigned char neighbor = img[(y + wy) * W + (x + wx)];
                    if (neighbor < center) bits |= (1ULL << bit);
                    ++bit;
                }
            }
            census[idx] = bits;
        }
    }
}

// ===========================================================================
// 2) cost_volume_cpu — line-by-line twin of cost_volume_kernel.
// ===========================================================================
void cost_volume_cpu(const unsigned long long* census_l, const unsigned long long* census_r,
                     unsigned char* cost, int W, int H)
{
    const size_t plane = static_cast<size_t>(H) * static_cast<size_t>(W);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int pix = y * W + x;
            const unsigned long long cl = census_l[pix];
            if (cl == kCensusInvalid) {
                for (int d = 0; d < kMaxDisp; ++d) cost[static_cast<size_t>(d) * plane + pix] = kCostInvalid;
                continue;
            }
            for (int d = 0; d < kMaxDisp; ++d) {
                const int xr = x - d;
                unsigned char c;
                if (xr < 0) {
                    c = kCostInvalid;
                } else {
                    const unsigned long long cr = census_r[y * W + xr];
                    if (cr == kCensusInvalid) {
                        c = kCostInvalid;
                    } else {
                        c = static_cast<unsigned char>(popcount64_cpu(cl ^ cr));
                    }
                }
                cost[static_cast<size_t>(d) * plane + pix] = c;
            }
        }
    }
}

// ===========================================================================
// 3) sgm_path_cpu — line-by-line twin of sgm_path_kernel. Same D-major
//    indexing, same running-min recurrence, same additive accumulation into
//    lsum — the only difference from the GPU version is that "one thread
//    per scanline" becomes "one outer loop iteration per scanline" here.
// ===========================================================================
void sgm_path_cpu(const unsigned char* cost, int* lsum, int W, int H, int P1, int P2, int dx, int dy)
{
    const size_t plane = static_cast<size_t>(H) * static_cast<size_t>(W);
    const int lines = (dx != 0) ? H : W;

    for (int line = 0; line < lines; ++line) {
        int start, end, step;
        if (dx != 0) { start = (dx > 0) ? 0 : (W - 1); end = (dx > 0) ? W : -1; step = dx; }
        else         { start = (dy > 0) ? 0 : (H - 1); end = (dy > 0) ? H : -1; step = dy; }

        int prev[kMaxDisp];
        bool first = true;

        for (int t = start; t != end; t += step) {
            const int x = (dx != 0) ? t : line;
            const int y = (dx != 0) ? line : t;
            const int pix = y * W + x;

            if (first) {
                for (int d = 0; d < kMaxDisp; ++d) {
                    const int c = cost[static_cast<size_t>(d) * plane + pix];
                    prev[d] = c;
                    lsum[static_cast<size_t>(d) * plane + pix] += c;
                }
                first = false;
                continue;
            }

            int prev_min = prev[0];
            for (int d = 1; d < kMaxDisp; ++d) if (prev[d] < prev_min) prev_min = prev[d];

            int cur[kMaxDisp];
            for (int d = 0; d < kMaxDisp; ++d) {
                const int e0 = prev[d];
                const int e1 = (d > 0)            ? prev[d - 1] + P1 : (prev_min + P2);
                const int e2 = (d < kMaxDisp - 1) ? prev[d + 1] + P1 : (prev_min + P2);
                int m = e0;
                if (e1 < m) m = e1;
                if (e2 < m) m = e2;
                const int e3 = prev_min + P2;
                if (e3 < m) m = e3;
                const int c = cost[static_cast<size_t>(d) * plane + pix];
                cur[d] = c + m - prev_min;
            }
            for (int d = 0; d < kMaxDisp; ++d) {
                prev[d] = cur[d];
                lsum[static_cast<size_t>(d) * plane + pix] += cur[d];
            }
        }
    }
}

// ===========================================================================
// 4) Winner-take-all — twins of wta_bm_kernel / wta_bm_right_kernel /
//    wta_sgm_kernel / wta_sgm_right_kernel.
// ===========================================================================
void wta_bm_cpu(const unsigned char* cost, unsigned char* disp, int W, int H)
{
    const size_t plane = static_cast<size_t>(H) * static_cast<size_t>(W);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int pix = y * W + x;
            if (x < kCensusHalf || x >= W - kCensusHalf || y < kCensusHalf || y >= H - kCensusHalf) {
                disp[pix] = kInvalidDisp;
                continue;
            }
            int best = 256, best_d = 0;
            for (int d = 0; d < kMaxDisp; ++d) {
                const int c = cost[static_cast<size_t>(d) * plane + pix];
                if (c < best) { best = c; best_d = d; }
            }
            disp[pix] = (best >= kCostInvalid) ? kInvalidDisp : static_cast<unsigned char>(best_d);
        }
    }
}

void wta_bm_right_cpu(const unsigned char* cost, unsigned char* disp_r, int W, int H)
{
    const size_t plane = static_cast<size_t>(H) * static_cast<size_t>(W);
    for (int y = 0; y < H; ++y) {
        for (int xr = 0; xr < W; ++xr) {
            const int pix_r = y * W + xr;
            if (xr < kCensusHalf || xr >= W - kCensusHalf || y < kCensusHalf || y >= H - kCensusHalf) {
                disp_r[pix_r] = kInvalidDisp;
                continue;
            }
            int best = 256, best_d = 0;
            for (int d = 0; d < kMaxDisp; ++d) {
                const int xl = xr + d;
                if (xl >= W) break;
                const int c = cost[static_cast<size_t>(d) * plane + y * W + xl];
                if (c < best) { best = c; best_d = d; }
            }
            disp_r[pix_r] = (best >= kCostInvalid) ? kInvalidDisp : static_cast<unsigned char>(best_d);
        }
    }
}

void wta_sgm_cpu(const int* lsum, unsigned char* disp, int W, int H)
{
    const size_t plane = static_cast<size_t>(H) * static_cast<size_t>(W);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int pix = y * W + x;
            if (x < kCensusHalf || x >= W - kCensusHalf || y < kCensusHalf || y >= H - kCensusHalf) {
                disp[pix] = kInvalidDisp;
                continue;
            }
            long long best = -1;
            int best_d = 0;
            for (int d = 0; d < kMaxDisp; ++d) {
                const long long c = lsum[static_cast<size_t>(d) * plane + pix];
                if (best < 0 || c < best) { best = c; best_d = d; }
            }
            disp[pix] = static_cast<unsigned char>(best_d);
        }
    }
}

void wta_sgm_right_cpu(const int* lsum, unsigned char* disp_r, int W, int H)
{
    const size_t plane = static_cast<size_t>(H) * static_cast<size_t>(W);
    for (int y = 0; y < H; ++y) {
        for (int xr = 0; xr < W; ++xr) {
            const int pix_r = y * W + xr;
            if (xr < kCensusHalf || xr >= W - kCensusHalf || y < kCensusHalf || y >= H - kCensusHalf) {
                disp_r[pix_r] = kInvalidDisp;
                continue;
            }
            long long best = -1;
            int best_d = 0;
            for (int d = 0; d < kMaxDisp; ++d) {
                const int xl = xr + d;
                if (xl >= W) break;
                const long long c = lsum[static_cast<size_t>(d) * plane + y * W + xl];
                if (best < 0 || c < best) { best = c; best_d = d; }
            }
            disp_r[pix_r] = static_cast<unsigned char>(best_d);
        }
    }
}

// ===========================================================================
// 5) lr_check_cpu — twin of lr_check_kernel.
// ===========================================================================
void lr_check_cpu(const unsigned char* disp_l, const unsigned char* disp_r,
                  unsigned char* disp_out, int W, int H, int tol)
{
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int pix = y * W + x;
            const unsigned char dl = disp_l[pix];
            if (dl == kInvalidDisp) { disp_out[pix] = kInvalidDisp; continue; }

            const int xr = x - static_cast<int>(dl);
            if (xr < 0 || xr >= W) { disp_out[pix] = kInvalidDisp; continue; }

            const unsigned char dr = disp_r[y * W + xr];
            if (dr == kInvalidDisp) { disp_out[pix] = kInvalidDisp; continue; }

            const int diff = static_cast<int>(dl) - static_cast<int>(dr);
            const int adiff = (diff < 0) ? -diff : diff;
            disp_out[pix] = (adiff <= tol) ? dl : kInvalidDisp;
        }
    }
}

// ===========================================================================
// 6) median3_cpu — twin of median3_kernel.
// ===========================================================================
void median3_cpu(const unsigned char* disp_in, unsigned char* disp_out, int W, int H)
{
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int pix = y * W + x;
            const unsigned char center = disp_in[pix];
            if (center == kInvalidDisp) { disp_out[pix] = kInvalidDisp; continue; }

            unsigned char vals[9];
            int n = 0;
            for (int wy = -1; wy <= 1; ++wy) {
                const int ny = y + wy;
                if (ny < 0 || ny >= H) continue;
                for (int wx = -1; wx <= 1; ++wx) {
                    const int nx = x + wx;
                    if (nx < 0 || nx >= W) continue;
                    const unsigned char v = disp_in[ny * W + nx];
                    if (v != kInvalidDisp) vals[n++] = v;
                }
            }
            for (int i = 1; i < n; ++i) {
                const unsigned char key = vals[i];
                int j = i - 1;
                while (j >= 0 && vals[j] > key) { vals[j + 1] = vals[j]; --j; }
                vals[j + 1] = key;
            }
            disp_out[pix] = vals[n / 2];
        }
    }
}
