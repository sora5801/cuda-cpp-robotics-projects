// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 02.12
//                     (Range-image conversion + depth-clustering segmentation)
//
// WHY does a GPU repository ship a CPU implementation of everything?
// ------------------------------------------------------------------
// Two load-bearing reasons (CLAUDE.md §5): (1) it is the CORRECTNESS ORACLE
// — main.cu runs both and asserts agreement, so a GPU bug (wrong indexing,
// a missed tail element, a race) is caught, not silently trusted; (2) it is
// the TEACHING BASELINE — read this file first, then kernels.cu, and see
// exactly what parallelization changed.
//
// Independence ruling for THIS project specifically (mirrors 02.02's and
// 02.04's identical rulings for the analogous stages, cited)
// ---------------------------------------------------------------------------
// Two kinds of code live in this file:
//
//   SHARED DATA-LAYOUT FORMULAS (not an independence violation): the range/
//   voxel key packing, organized_cell_index, beta_criterion_rad, and
//   ground_step_angle_deg all live once, in kernels.cuh, as plain inline
//   functions. They are DEFINITIONS this project promises to compute a
//   specific way — kernels.cu's device transcriptions must match them
//   EXACTLY, and a VERIFY gate exists precisely to catch any drift (02.01's
//   established pattern, cited).
//
//   GENUINELY INDEPENDENT ALGORITHMIC CORES: ground_removal_cpu,
//   depth_edges_cpu, build_edges_euclid_cpu, and serial_union_find_cpu are
//   all separately-written, structurally different implementations from
//   their kernels.cu counterparts:
//     - scatter_to_organized_cpu uses a plain PER-CELL RUNNING MINIMUM (no
//       encoding, no atomics) instead of the GPU's encoded-atomicMin race —
//       02.02's identical twin technique, cited. Order-independent (min is
//       commutative/associative), so it is directly, exactly comparable.
//     - ground_removal_cpu is a fresh sequential per-column loop, sharing
//       only the ANGLE-TEST RULE (the definition of "ground" this project
//       promises), not any code structure, with ground_removal_kernel.
//     - depth_edges_cpu is a fresh double loop over cells x 2 neighbors,
//       sharing only the beta FORMULA (kernels.cuh) with depth_edges_kernel.
//     - build_edges_euclid_cpu uses an std::unordered_map<uint64_t,
//       std::vector<int>> voxel->points map — a THIRD data structure
//       distinct from both 02.01's open-addressing hash and this project's
//       own GPU sorted-array+binary-search index (02.04's identical
//       argument, cited).
//     - serial_union_find_cpu is ordinary sequential union-find with full
//       recursive path compression (a DIFFERENT compression strategy from
//       the GPU's iterative path halving) — 02.04's identical twin, reused
//       here as a GENERIC function over any edge list (both graphs this
//       project builds are verified with the SAME oracle function).
//
// Why this is not paranoia: flagship 13.03 had an identical bug live in
// BOTH the GPU path and a too-similar CPU twin; only a genuinely
// independent gate caught it (cited in 02.04's identical file header).
// Union-find and the beta/ground-removal walks are exactly this project's
// "clever enough to hide a shared bug" code, so they get the same
// treatment here.
//
// Read this after: kernels.cu — then compare the two side by side.
// ===========================================================================

#include "kernels.cuh"
#include <cstdint>
#include <cmath>
#include <algorithm>   // std::sort — canonicalizes edge-list output for VERIFY

// ---------------------------------------------------------------------------
// scatter_to_organized_cpu — INDEPENDENT twin of stage 1a. For every cell,
// keep the point with the SMALLEST range that targets it — a plain running
// minimum over ALL input points (O(n_points * kNumCells) is far too slow
// for a real driver, but this is a reference oracle read for correctness,
// not speed: CLAUDE.md "clarity beats speed" in this file specifically).
// A cheaper, still-independent alternative (bucket by cell first) would
// blur the "genuinely different mechanism" line this file's header draws;
// the running minimum is the simplest statement of the DEFINITION
// ("nearest return wins each cell"), which is exactly what an oracle should
// state as plainly as possible.
// ---------------------------------------------------------------------------
void scatter_to_organized_cpu(int n_points,
                              const int* ring, const int* az_bin, const float* range_m,
                              const float* px, const float* py, const float* pz, const int* truth,
                              float* range_img, float* xyz_img, int* truth_img, int* winner_idx_img)
{
    for (int c = 0; c < kNumCells; ++c) {
        range_img[c] = 0.0f;
        xyz_img[c * 3 + 0] = 0.0f; xyz_img[c * 3 + 1] = 0.0f; xyz_img[c * 3 + 2] = 0.0f;
        truth_img[c] = -1;
        winner_idx_img[c] = -1;
    }
    // best_range[c] tracks the current winner's range per cell; initialized
    // conceptually to +infinity (the "no winner yet" identity element for a
    // minimum, the same role 02.04's stats AABB sentinels play).
    std::vector<float> best_range(static_cast<size_t>(kNumCells), 1.0e30f);

    for (int i = 0; i < n_points; ++i) {
        const int cell = organized_cell_index(ring[i], az_bin[i]);
        if (range_m[i] < best_range[static_cast<size_t>(cell)]) {
            best_range[static_cast<size_t>(cell)] = range_m[i];
            range_img[cell] = range_m[i];
            xyz_img[cell * 3 + 0] = px[i];
            xyz_img[cell * 3 + 1] = py[i];
            xyz_img[cell * 3 + 2] = pz[i];
            truth_img[cell] = truth[i];
            winner_idx_img[cell] = i;
        }
    }
}

// ---------------------------------------------------------------------------
// ground_removal_cpu — INDEPENDENT twin of stage 2 (see file header). Same
// column-walk RULE, freshly written: for each of the kAzimuthBins columns,
// walk rings 0..15, testing the vertical angle against the previous VALID
// return (or the virtual (rho=0, z=-kSensorHeightM) reference for the
// column's first valid return).
// ---------------------------------------------------------------------------
void ground_removal_cpu(const float* range_img, const float* xyz_img,
                        int* ground_label, int* obstacle_mask)
{
    for (int col = 0; col < kAzimuthBins; ++col) {
        float rho_prev = 0.0f;
        float z_prev = -kSensorHeightM;

        for (int ring = 0; ring < kNumBeams; ++ring) {
            const int cell = organized_cell_index(ring, col);
            const float r = range_img[cell];
            if (r <= 0.0f) {
                ground_label[cell] = 0;
                obstacle_mask[cell] = 0;
                continue;
            }
            const float x = xyz_img[cell * 3 + 0];
            const float y = xyz_img[cell * 3 + 1];
            const float z = xyz_img[cell * 3 + 2];
            const float rho = std::sqrt(x * x + y * y);

            const float angle_deg = ground_step_angle_deg(rho_prev, z_prev, rho, z) ;
            const bool is_ground = std::fabs(angle_deg) <= kGroundAngleThresholdDeg;
            ground_label[cell] = is_ground ? 1 : 0;
            obstacle_mask[cell] = is_ground ? 0 : 1;

            rho_prev = rho;
            z_prev = z;
        }
    }
}

// ---------------------------------------------------------------------------
// depth_edges_cpu — INDEPENDENT twin of stage 3: a plain nested loop over
// every cell and its two forward neighbors (ring+1 same column; same ring,
// column+1 with wrap), sharing only beta_criterion_rad() (the shared
// FORMULA, kernels.cuh) with depth_edges_kernel. Returns the edge set
// canonicalized ascending (u<v both cell indices, vector sorted) for
// main.cu's set-equality comparison against the GPU's own canonicalized
// edge list.
// ---------------------------------------------------------------------------
std::vector<std::pair<int,int>> depth_edges_cpu(const float* range_img, const int* obstacle_mask)
{
    std::vector<std::pair<int,int>> edges;

    for (int ring = 0; ring < kNumBeams; ++ring) {
        for (int col = 0; col < kAzimuthBins; ++col) {
            const int c = organized_cell_index(ring, col);
            if (range_img[c] <= 0.0f || !obstacle_mask[c]) continue;
            const float r_c = range_img[c];

            if (ring + 1 < kNumBeams) {
                const int nc = organized_cell_index(ring + 1, col);
                if (range_img[nc] > 0.0f && obstacle_mask[nc]) {
                    const float r_n = range_img[nc];
                    const float alpha = kBeamElevRad[ring + 1] - kBeamElevRad[ring];
                    const float r1 = std::max(r_c, r_n);
                    const float r2 = std::min(r_c, r_n);
                    const float beta = beta_criterion_rad(r1, r2, alpha);
                    if (beta >= kBetaThresholdRad) edges.emplace_back(c, nc);
                }
            }
            {
                const int next_col = (col + 1 == kAzimuthBins) ? 0 : col + 1;
                const int nc = organized_cell_index(ring, next_col);
                if (range_img[nc] > 0.0f && obstacle_mask[nc]) {
                    const float r_n = range_img[nc];
                    const float r1 = std::max(r_c, r_n);
                    const float r2 = std::min(r_c, r_n);
                    const float beta = beta_criterion_rad(r1, r2, kAzimuthStepRad);
                    // Canonicalize (min,max): at the wrap seam (col ==
                    // kAzimuthBins-1) nc < c even though nc is the
                    // "forward" neighbor -- see kernels.cu's identical note
                    // at depth_edges_kernel's matching call site.
                    if (beta >= kBetaThresholdRad)
                        edges.emplace_back(std::min(c, nc), std::max(c, nc));
                }
            }
        }
    }

    std::sort(edges.begin(), edges.end());
    return edges;
}

// ---------------------------------------------------------------------------
// build_edges_euclid_cpu — INDEPENDENT twin of stage 5's edge construction
// (see file header for how it differs from the GPU's sorted-array index).
// Two passes: bucket points into an unordered_map keyed by voxel key, then
// for every point walk its 27-cell stencil and test the actual distance —
// the SAME structure 02.04's build_edges_cpu uses, cited.
// ---------------------------------------------------------------------------
std::vector<std::pair<int,int>> build_edges_euclid_cpu(int n, const float* xyz)
{
    const float d = kEuclideanClusterToleranceM;
    const float d2 = d * d;

    std::unordered_map<uint64_t, std::vector<int>> voxel_points;
    voxel_points.reserve(static_cast<size_t>(n) * 2);

    std::vector<uint64_t> point_key(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const float px = xyz[i * 3 + 0], py = xyz[i * 3 + 1], pz = xyz[i * 3 + 2];
        const int32_t vx = voxel_coord(px, d), vy = voxel_coord(py, d), vz = voxel_coord(pz, d);
        const uint64_t key = pack_voxel_key(vx, vy, vz);
        point_key[static_cast<size_t>(i)] = key;
        voxel_points[key].push_back(i);
    }

    std::vector<std::pair<int,int>> edges;
    for (int i = 0; i < n; ++i) {
        int32_t vx, vy, vz;
        unpack_voxel_key(point_key[static_cast<size_t>(i)], vx, vy, vz);
        const float pi[3] = { xyz[i * 3 + 0], xyz[i * 3 + 1], xyz[i * 3 + 2] };

        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const uint64_t nkey = pack_voxel_key(vx + dx, vy + dy, vz + dz);
                    const auto it = voxel_points.find(nkey);
                    if (it == voxel_points.end()) continue;
                    for (int j : it->second) {
                        if (j <= i) continue;
                        const float pj[3] = { xyz[j * 3 + 0], xyz[j * 3 + 1], xyz[j * 3 + 2] };
                        if (squared_distance(pi, pj) <= d2) {
                            edges.emplace_back(i, j);
                        }
                    }
                }
            }
        }
    }

    std::sort(edges.begin(), edges.end());
    return edges;
}

// ---------------------------------------------------------------------------
// serial_union_find_cpu — ordinary sequential union-find (02.04's identical
// twin, reused here as a GENERIC oracle over any edge list — main.cu calls
// this once for the depth-image graph and once for the Euclidean-comparison
// graph). find_root_recursive performs full recursive path compression, a
// DIFFERENT compression strategy from the GPU's iterative path halving —
// both are textbook-correct, and using a different one here is itself a
// small extra piece of independence (02.04's identical reasoning).
// ---------------------------------------------------------------------------
static int find_root_recursive(std::vector<int>& parent, int x)
{
    if (parent[static_cast<size_t>(x)] == x) return x;
    const int root = find_root_recursive(parent, parent[static_cast<size_t>(x)]);
    parent[static_cast<size_t>(x)] = root;
    return root;
}

void serial_union_find_cpu(int n, const std::vector<std::pair<int,int>>& edges, std::vector<int>& parent_out)
{
    parent_out.assign(static_cast<size_t>(n), 0);
    for (int i = 0; i < n; ++i) parent_out[static_cast<size_t>(i)] = i;

    for (const auto& e : edges) {
        const int ru = find_root_recursive(parent_out, e.first);
        const int rv = find_root_recursive(parent_out, e.second);
        if (ru == rv) continue;
        // Union by MIN: the canonical-root convention this project promises
        // (a component's root is always its smallest member index) — the
        // SAME convention the GPU's uf_union_sweep_kernel promises, so the
        // two results are directly, bit-exactly comparable.
        if (ru < rv) parent_out[static_cast<size_t>(rv)] = ru;
        else         parent_out[static_cast<size_t>(ru)] = rv;
    }

    for (int i = 0; i < n; ++i) {
        parent_out[static_cast<size_t>(i)] = find_root_recursive(parent_out, i);
    }
}
