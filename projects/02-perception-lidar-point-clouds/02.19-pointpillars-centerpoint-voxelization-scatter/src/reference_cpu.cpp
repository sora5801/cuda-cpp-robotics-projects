// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 02.19
//                     (PointPillars/CenterPoint voxelization + scatter
//                     kernels feeding TensorRT)
//
// WHY does a GPU repository ship a CPU implementation of everything?
// (CLAUDE.md §5 — the full two-reason case lives in docs/PROJECT_TEMPLATE's
// copy of this file; not repeated here.) In THIS project specifically, the
// CPU oracle matters more than usual: this is a data-LAYOUT project, and
// layout bugs (an off-by-one pillar index, a swapped x/y, a wrong zero-pad
// boundary) are exactly the class of bug that "runs without crashing and
// produces plausible-looking garbage" — a CPU twin that a human can read
// top to bottom is the only thing standing between "the demo runs" and
// "the demo is right".
//
// Independence ruling applied here (see docs/PROJECT_TEMPLATE/src/
// reference_cpu.cpp's file header for the full three-tier policy this
// repeats): kernels.cuh's key formulas (pillar_key_of / voxel_key_of /
// z_in_range) and the 9-D feature LAYOUT are data-layout contracts — shared,
// single-sourced, called directly from this file. The SORT itself uses
// std::stable_sort (this file) vs. thrust::stable_sort_by_key (kernels.cu)
// — genuinely different libraries/algorithms, not a transcription. The
// per-pillar reduction (mean, max-pool), the conv stencil, and the peak
// search + NMS are each written here in their OWN loop structure, not a
// port of the corresponding kernel. Where the CPU and GPU loops walk
// points in the identical, well-defined order (ascending original index
// within a pillar — guaranteed by BOTH stable_sort variants), their
// floating-point sums are expected to be BIT-EXACT (no reordering, no
// fused-multiply-add divergence to hide behind) — main.cu's VERIFY gates
// say so explicitly where they hold that bar, and fall back to a
// documented tight tolerance where they do not (the PFN's linear+ReLU is a
// multiply-add chain the GPU's compiler may fuse differently than cl.exe;
// see THEORY.md "Numerical considerations").
//
// Read this after: kernels.cu — then compare the two side by side.
// ===========================================================================

#include "kernels.cuh"

#include <algorithm>   // std::stable_sort
#include <cmath>       // (kept for parity with the template; no transcendentals used here)
#include <utility>     // std::pair
#include <vector>

// ---------------------------------------------------------------------------
// pillar_keys_cpu / voxel_keys_cpu — twins of compute_pillar_keys_kernel /
// compute_voxel_keys_kernel. Both simply call kernels.cuh's SHARED,
// CUDA-qualifier-free formulas (pillar_key_of / voxel_key_of / z_in_range)
// — the "data-layout contract must be single-sourced" half of the ruling.
// main.cu's VERIFY(keys) gate compares these, point for point, against
// kernels.cu's independent __device__ transcription of the SAME formulas —
// the gate that would catch a typo in either copy.
// ---------------------------------------------------------------------------
void pillar_keys_cpu(int n, const float* points, int* keys_out)
{
    for (int i = 0; i < n; ++i) {
        const float x = points[i * 4 + 0];
        const float y = points[i * 4 + 1];
        const float z = points[i * 4 + 2];
        keys_out[i] = z_in_range(z) ? pillar_key_of(x, y) : -1;
    }
}

void voxel_keys_cpu(int n, const float* points, int* keys_out)
{
    for (int i = 0; i < n; ++i) {
        const float x = points[i * 4 + 0];
        const float y = points[i * 4 + 1];
        const float z = points[i * 4 + 2];
        keys_out[i] = voxel_key_of(x, y, z);
    }
}

// ---------------------------------------------------------------------------
// count_occupied_cpu — a GENUINELY DIFFERENT algorithm from the GPU's
// sort-based occupied count: a plain presence array, one linear pass, no
// sorting at all. main.cu's pillar_vs_voxel comparison uses this as an
// independent cross-check of launch_sort_and_compact's returned count for
// BOTH grids — a gate that cannot share a bug with the machinery it checks.
// ---------------------------------------------------------------------------
int count_occupied_cpu(int n, const float* points, bool use_voxel_keys, int num_cells)
{
    std::vector<unsigned char> present(static_cast<size_t>(num_cells), 0);
    for (int i = 0; i < n; ++i) {
        const float x = points[i * 4 + 0];
        const float y = points[i * 4 + 1];
        const float z = points[i * 4 + 2];
        const int key = use_voxel_keys ? voxel_key_of(x, y, z)
                                       : (z_in_range(z) ? pillar_key_of(x, y) : -1);
        if (key >= 0 && key < num_cells) present[static_cast<size_t>(key)] = 1;
    }
    int count = 0;
    for (unsigned char p : present) count += p;
    return count;
}

// ---------------------------------------------------------------------------
// sorted_bin_cpu — the independent Method-B twin. std::stable_sort (NOT
// Thrust) by pillar_key_of; truncation keeps the first kMaxPointsPerPillar
// points of that stable order — the SAME rule sorted_bin_kernel enforces
// via thrust::stable_sort_by_key + a device kernel, reached here by a
// completely different code path (a sorted std::vector<pair<key,idx>>, no
// GPU, no Thrust).
//
// Storage mirrors PillarBinGPU's dense per-cell shape (the shared
// data-layout contract — see this file's header) so main.cu can compare
// GPU and CPU pillar-by-pillar without any extra bookkeeping.
//
// Order-preservation note (why the mean ends up BIT-EXACT vs the GPU, not
// just "close"): `valid` below is built by scanning i=0..n-1 IN ORDER, so
// before sorting it already holds valid points in ascending original index.
// std::stable_sort then reorders ONLY by key, preserving that ascending-
// index order within each equal-key run — token-for-token the same
// guarantee thrust::stable_sort_by_key gives kernels.cu. Both therefore
// visit a pillar's kept points in the IDENTICAL order and sum them with
// plain sequential float `+=` (no reduction tree, no FMA) — IEEE-754
// float addition is deterministic given identical operands and order, so
// the two sums round identically, bit for bit.
// ---------------------------------------------------------------------------
int sorted_bin_cpu(int n, const float* points,
                   unsigned int* point_count_dense, float* raw_points_dense,
                   int* occupied_cell_out, unsigned int* kept_count_out, float* mean_xyz_out)
{
    std::vector<int> keys(static_cast<size_t>(n));
    pillar_keys_cpu(n, points, keys.data());

    std::vector<std::pair<int, int>> valid;   // (pillar key, original point index)
    valid.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        if (keys[static_cast<size_t>(i)] >= 0) valid.emplace_back(keys[static_cast<size_t>(i)], i);

    std::stable_sort(valid.begin(), valid.end(),
                     [](const std::pair<int, int>& a, const std::pair<int, int>& b) { return a.first < b.first; });

    int num_occupied = 0;
    size_t i = 0;
    while (i < valid.size()) {
        size_t j = i;
        const int cell = valid[i].first;
        while (j < valid.size() && valid[j].first == cell) ++j;
        const int run_len = static_cast<int>(j - i);
        const unsigned int kept = static_cast<unsigned int>(run_len < kMaxPointsPerPillar ? run_len : kMaxPointsPerPillar);

        point_count_dense[static_cast<size_t>(cell)] = static_cast<unsigned int>(run_len);

        // Plain sequential float accumulation, ascending original index —
        // see the file-header note above for why this matches the GPU bit
        // for bit rather than merely "closely".
        float sx = 0.0f, sy = 0.0f, sz = 0.0f;
        for (unsigned int k = 0; k < kept; ++k) {
            const int idx = valid[i + k].second;
            float* dst = &raw_points_dense[static_cast<size_t>(cell) * kMaxPointsPerPillar * 4 + static_cast<size_t>(k) * 4];
            dst[0] = points[idx * 4 + 0];
            dst[1] = points[idx * 4 + 1];
            dst[2] = points[idx * 4 + 2];
            dst[3] = points[idx * 4 + 3];
            sx += dst[0];
            sy += dst[1];
            sz += dst[2];
        }
        const float inv = kept > 0 ? 1.0f / static_cast<float>(kept) : 0.0f;

        occupied_cell_out[static_cast<size_t>(num_occupied)] = cell;
        kept_count_out[static_cast<size_t>(num_occupied)] = kept;
        mean_xyz_out[static_cast<size_t>(num_occupied) * 3 + 0] = sx * inv;
        mean_xyz_out[static_cast<size_t>(num_occupied) * 3 + 1] = sy * inv;
        mean_xyz_out[static_cast<size_t>(num_occupied) * 3 + 2] = sz * inv;

        ++num_occupied;
        i = j;
    }
    return num_occupied;
}

// ---------------------------------------------------------------------------
// augment_features_cpu — independent twin of augment_features_kernel. Same
// formula (a shared data-layout contract — the 9-D vector IS the network's
// input shape, not an algorithm to reinvent per implementation), its own
// loop nesting. Identical zero-padding policy for slots >= kept_count.
// ---------------------------------------------------------------------------
void augment_features_cpu(int num_occupied, const int* occupied_cell, const unsigned int* kept_count,
                          const float* mean_xyz, const float* raw_points_dense, float* features_out)
{
    for (int p = 0; p < num_occupied; ++p) {
        const int cell = occupied_cell[p];
        const unsigned int kept = kept_count[p];
        const int ix = cell % kGridNX;
        const int iy = cell / kGridNX;
        const float pcx = kXMin + (static_cast<float>(ix) + 0.5f) * kPillarSizeM;
        const float pcy = kYMin + (static_cast<float>(iy) + 0.5f) * kPillarSizeM;

        for (int slot = 0; slot < kMaxPointsPerPillar; ++slot) {
            float* feat = &features_out[static_cast<size_t>(p) * kMaxPointsPerPillar * kNumPointFeatures
                                       + static_cast<size_t>(slot) * kNumPointFeatures];
            if (static_cast<unsigned int>(slot) >= kept) {
                for (int d = 0; d < kNumPointFeatures; ++d) feat[d] = 0.0f;
                continue;
            }
            const float* pt = &raw_points_dense[static_cast<size_t>(cell) * kMaxPointsPerPillar * 4 + static_cast<size_t>(slot) * 4];
            const float x = pt[0], y = pt[1], z = pt[2], inten = pt[3];
            feat[0] = x;
            feat[1] = y;
            feat[2] = z;
            feat[3] = inten;
            feat[4] = x - mean_xyz[p * 3 + 0];
            feat[5] = y - mean_xyz[p * 3 + 1];
            feat[6] = z - mean_xyz[p * 3 + 2];
            feat[7] = x - pcx;
            feat[8] = y - pcy;
        }
    }
}

// ---------------------------------------------------------------------------
// pfn_lite_cpu — independent twin of pfn_lite_kernel: same fixed-weight
// formula (the PFN-lite IS the thing under test, so both sides necessarily
// compute the same math — the independence here is in the loop structure
// and in never calling into kernels.cu), its own accumulation loop.
// ---------------------------------------------------------------------------
void pfn_lite_cpu(int num_occupied, const float* features, const unsigned int* kept_count,
                  const float* lin_w, const float* lin_b, float* pillar_feat_out)
{
    for (int p = 0; p < num_occupied; ++p) {
        const unsigned int kept = kept_count[p];
        const float occupancy = static_cast<float>(kept) / static_cast<float>(kMaxPointsPerPillar);

        float z_min = 1e30f, z_max = -1e30f;
        float lin_acc[kPfnLinOut];
        for (int c = 0; c < kPfnLinOut; ++c) lin_acc[c] = -1e30f;

        for (unsigned int k = 0; k < kept; ++k) {
            const float* f = &features[static_cast<size_t>(p) * kMaxPointsPerPillar * kNumPointFeatures
                                      + static_cast<size_t>(k) * kNumPointFeatures];
            const float z = f[2];
            if (z < z_min) z_min = z;
            if (z > z_max) z_max = z;

            for (int c = 0; c < kPfnLinOut; ++c) {
                float acc = lin_b[c];
                for (int d = 0; d < kNumPointFeatures; ++d)
                    acc += lin_w[c * kNumPointFeatures + d] * f[d];
                const float relu = acc > 0.0f ? acc : 0.0f;
                if (relu > lin_acc[c]) lin_acc[c] = relu;
            }
        }
        const float height_extent = kept > 0 ? (z_max - z_min) / kHeightNormM : 0.0f;
        const float height_extent_clamped = height_extent < 0.0f ? 0.0f : (height_extent > 1.0f ? 1.0f : height_extent);

        float* out = &pillar_feat_out[static_cast<size_t>(p) * kPfnChannels];
        out[0] = occupancy;
        out[1] = height_extent_clamped;
        for (int c = 0; c < kPfnLinOut; ++c) out[2 + c] = kept > 0 ? lin_acc[c] : 0.0f;
    }
}

// ---------------------------------------------------------------------------
// scatter_cpu — independent twin of scatter_kernel: a plain copy loop.
// ---------------------------------------------------------------------------
void scatter_cpu(int num_occupied, const int* occupied_cell, const float* pillar_feat, float* canvas_out)
{
    for (int p = 0; p < num_occupied; ++p) {
        const int cell = occupied_cell[p];
        const int ix = cell % kGridNX;
        const int iy = cell / kGridNX;
        for (int c = 0; c < kPfnChannels; ++c) {
            canvas_out[static_cast<size_t>(c) * kGridNY * kGridNX + static_cast<size_t>(iy) * kGridNX + ix]
                = pillar_feat[static_cast<size_t>(p) * kPfnChannels + c];
        }
    }
}

// ---------------------------------------------------------------------------
// conv3x3_cpu — independent twin of conv3x3_kernel: same zero-padding
// boundary policy, its own nested loop (row-major output scan rather than
// the GPU's per-thread-per-pixel mapping).
// ---------------------------------------------------------------------------
void conv3x3_cpu(const float* in_plane, int h, int w, const float* kernel3x3, float bias, float* out_plane)
{
    for (int iy = 0; iy < h; ++iy) {
        for (int ix = 0; ix < w; ++ix) {
            float acc = bias;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    const int nx = ix + kx, ny = iy + ky;
                    const float v = (nx >= 0 && nx < w && ny >= 0 && ny < h) ? in_plane[ny * w + nx] : 0.0f;
                    acc += v * kernel3x3[(ky + 1) * 3 + (kx + 1)];
                }
            }
            out_plane[iy * w + ix] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// peak_extract_and_nms_cpu — independent twin of peak_extract_kernel's
// local-max rule (identical tie-break, so the CANDIDATE SET is bit-exact
// against the GPU's is_candidate mask) plus NMS (greedy: highest score
// first, suppress any remaining candidate within nms_radius_pillars,
// Euclidean, in pillar units). This exact NMS logic is reimplemented
// independently in main.cu for the GPU path (candidate counts are tiny
// after thresholding, so neither path needs a kernel for it) — the two
// are not the same function, but they necessarily agree on the algorithm
// (NMS has no ambiguity once the tie-break and radius are fixed), so
// main.cu's detection_closure gate is a real, independent double-check.
// ---------------------------------------------------------------------------
void peak_extract_and_nms_cpu(const float* heatmap, int h, int w, float threshold, int window_r,
                              int nms_radius_pillars, std::vector<PeakCPU>& peaks_out)
{
    std::vector<PeakCPU> candidates;
    for (int iy = 0; iy < h; ++iy) {
        for (int ix = 0; ix < w; ++ix) {
            const float v = heatmap[iy * w + ix];
            if (v <= threshold) continue;

            bool is_max = true;
            const int my_flat = iy * w + ix;
            for (int dy = -window_r; dy <= window_r && is_max; ++dy) {
                for (int dx = -window_r; dx <= window_r; ++dx) {
                    const int nx = ix + dx, ny = iy + dy;
                    if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                    const int n_flat = ny * w + nx;
                    if (n_flat == my_flat) continue;
                    const float nv = heatmap[n_flat];
                    if (nv > v || (nv == v && n_flat < my_flat)) { is_max = false; break; }
                }
            }
            if (is_max) candidates.push_back(PeakCPU{iy, ix, v});
        }
    }

    // Greedy NMS: highest score first (stable_sort keeps a well-defined,
    // reproducible order among exact score ties — the row-major scan order
    // above), suppress weaker candidates within the radius.
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const PeakCPU& a, const PeakCPU& b) { return a.score > b.score; });
    std::vector<unsigned char> suppressed(candidates.size(), 0);
    const int r2 = nms_radius_pillars * nms_radius_pillars;
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) continue;
        peaks_out.push_back(candidates[i]);
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (suppressed[j]) continue;
            const int ddy = candidates[j].iy - candidates[i].iy;
            const int ddx = candidates[j].ix - candidates[i].ix;
            if (ddy * ddy + ddx * ddx <= r2) suppressed[j] = 1;
        }
    }
}
