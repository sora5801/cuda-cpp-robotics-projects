// ===========================================================================
// main.cu — entry point for project 11.01
//           GPU LiDAR simulator: hand-built BVH raycasting + beam
//           divergence, intensity, dropout noise
//
// What this program does, start to finish
// ---------------------------------------
//   1. Print the banner + GPU info; load the committed warehouse mesh +
//      materials, the sensor's scan configuration, and its world pose from
//      data/sample/ (all hand-rolled loaders — no mesh/CSV library).
//   2. BUILD the BVH on the HOST, once (median-split by triangle count —
//      see kernels.cuh for why that specific split rule matters).
//   3. VERIFY STAGE (the §5 GPU-vs-CPU gate): raycast the FULL demo frame
//      (channels x azimuth_steps beams) through the GPU kernel AND the CPU
//      oracle; require hit/dropped to match EXACTLY and range/intensity to
//      agree within a documented relative tolerance.
//   4. ANALYTIC GATES (run through the CPU oracle only — see kernels.cuh's
//      long comment on lidar_raycast_cpu for why that is sufficient):
//      (a) a beam aimed at the open floor must return the closed-form
//          range h/sin(|elevation|); (b) normal-incidence intensity at R
//          vs 2R must ratio exactly 4:1 (inverse-square law); (c) the
//          empirical dropout rate over many i.i.d. beams must match the
//          configured probability within a binomial statistical bound.
//   5. FRAME-LEVEL checks on the GPU's full-frame result: hit fraction and
//      mean range must fall in documented, MEASURED ranges.
//   6. Compact the surviving (hit & not dropped) beams into a PointCloud-
//      shaped buffer and write the two demo artifacts: a plottable CSV and
//      a range image (the LiDAR's native picture, THEORY.md explains why).
//
// Output contract (load-bearing!): demo/run_demo.ps1 diffs the STABLE lines
// of this program's stdout against demo/expected_output.txt — "[demo]",
// "SCENE:", "BVH:", "PROBLEM:", "VERIFY:", every "CHECK:", every
// "ARTIFACT:", and "RESULT:". "[info]"/"[time]" lines are NOT diffed (they
// carry machine-specific numbers). Change a stable line here => update
// demo/expected_output.txt in the same change, and vice versa.
//
// Read this first, then kernels.cuh (the contract) -> kernels.cu (the
// kernel) -> reference_cpu.cpp (the oracle twin).
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
#include <direct.h>               // _mkdir (std::filesystem avoided in .cu — see 07.09)
#else
#include <sys/stat.h>
#endif

// ---------------------------------------------------------------------------
// Verification constants. kVerifyTol mirrors the repo's standard §5 relative
// tolerance (08.01/02.06: 1e-3, ~100x headroom over the measured ulp-level
// GPU/CPU divergence from FP32 chained arithmetic + independent sin/cos
// implementations — see THEORY.md "Numerical considerations").
//
// The two "gate heights" and the ground-gate elevation define the SYNTHETIC
// single-beam probes the analytic gates fire (NOT the main demo frame):
// kGateHeight1M MUST match data/sample/sensor_poses.csv frame 0's z (both
// the ground-range gate and the dropout-statistics gate assume the real
// sensor sits at exactly this height) — if one changes, change the other
// and re-derive the expected numbers by hand before committing.
// kGateGroundElevationRad is chosen so the beam's horizontal reach
// (height / tan(|elevation|) =~ 7.4 m) stays inside the +x corridor
// scripts/make_synthetic.py deliberately keeps clear of obstacles.
// ---------------------------------------------------------------------------
// Two SEPARATE tolerances, not one blanket number — see THEORY.md "Numerical
// considerations" for the full measurement, summarized here: intensity's
// worst measured GPU/CPU deviation is 1.95e-4 (ordinary FP32 chained-
// arithmetic drift, same character as 08.01/02.06's ~1e-6..1e-3 stories),
// so kVerifyIntensityTol keeps the repo-standard 1e-3 with real headroom.
// Range is DIFFERENT: this kernel's divergence bundle picks the ARGMIN
// range across 5 independent rays (central + 4 subrays); near a geometric
// silhouette edge two of those rays can hit genuinely different surfaces at
// nearly-tied distances, and an ulp-level rounding difference between the
// GPU and CPU paths can flip which one "wins" — moving the reported range
// by the GEOMETRIC gap between the two surfaces (centimeters), not by an
// ulp. Measured on the committed scene: 5 of 23,340 hit beams (0.02%)
// exceed rel 1e-3, worst case 1.166e-2 — kVerifyRangeTol carries ~1.7x
// headroom over that measured worst case. A real indexing/logic bug would
// push error to O(1) (a wrong triangle across most of the frame), not
// stay clustered at O(1e-2) on a handful of edge beams, so this remains a
// meaningful gate, not a rubber stamp.
constexpr float kVerifyRangeTol = 2e-2f;
constexpr float kVerifyIntensityTol = 1e-3f;
constexpr float kGateHeight1M = 1.5f;
constexpr float kGateHeight2M = 3.0f;
constexpr float kGateGroundElevationRad = -0.20f;   // ~-11.46 deg
constexpr int   kGateDropoutBeams = 20000;
constexpr double kGateDropoutSigmaMultiple = 5.0;   // binomial-bound width (documented, not tuned to pass)

// ===========================================================================
// SECTION 1 — hand-rolled mesh / materials / sensor-config / pose loaders.
// Every loader is STRICT: an unrecognized label or a short row aborts with
// a message (the repo's strict-loader convention — 08.01's load_scenario,
// 02.06's cloud loader — catches a scene-authoring bug immediately instead
// of silently mis-simulating).
// ===========================================================================

struct RawMesh {
    std::vector<std::array<float, 3>> verts;
    std::vector<std::array<int, 3>>   faces;   // 0-indexed vertex ids
};

// load_obj — hand-rolled Wavefront OBJ reader: only "v x y z" and
// "f a b c" lines (bare 1-indexed integers, no texture/normal indices — the
// generator in scripts/make_synthetic.py never emits anything richer, and
// CLAUDE.md §5 says hand-roll rather than link a mesh library here).
static bool load_obj(const std::string& path, RawMesh& mesh)
{
    std::ifstream in(path);
    if (!in.is_open()) return false;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "v") {
            float x, y, z;
            if (!(ss >> x >> y >> z)) { std::fprintf(stderr, "obj: malformed 'v' line\n"); return false; }
            mesh.verts.push_back({ x, y, z });
        } else if (tag == "f") {
            int a, b, c;
            if (!(ss >> a >> b >> c)) { std::fprintf(stderr, "obj: malformed 'f' line\n"); return false; }
            mesh.faces.push_back({ a - 1, b - 1, c - 1 });   // OBJ is 1-indexed
        } else if (!tag.empty()) {
            std::fprintf(stderr, "obj: unsupported tag '%s' (this loader hand-rolls v/f only)\n", tag.c_str());
            return false;
        }
    }
    return true;
}

struct MaterialRange { int start, end, mat_id; };
struct MaterialTable {
    std::vector<Material> materials;
    std::vector<MaterialRange> ranges;
};

// load_materials — parses MATERIAL/RANGE rows (see data/sample/materials.csv
// and scripts/make_synthetic.py's write_materials()).
static bool load_materials(const std::string& path, MaterialTable& out)
{
    std::ifstream in(path);
    if (!in.is_open()) return false;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string label, cell;
        std::getline(ss, label, ',');
        if (label == "MATERIAL") {
            if (!std::getline(ss, cell, ',')) { std::fprintf(stderr, "materials: short MATERIAL row\n"); return false; }
            const int id = std::atoi(cell.c_str());
            if (!std::getline(ss, cell, ',')) { std::fprintf(stderr, "materials: short MATERIAL row\n"); return false; }
            const float albedo = std::strtof(cell.c_str(), nullptr);
            if (id != static_cast<int>(out.materials.size())) {
                std::fprintf(stderr, "materials: MATERIAL id %d out of order (expected %zu)\n", id, out.materials.size());
                return false;
            }
            out.materials.push_back(Material{ albedo });
        } else if (label == "RANGE") {
            int vals[3];
            for (int i = 0; i < 3; ++i) {
                if (!std::getline(ss, cell, ',')) { std::fprintf(stderr, "materials: short RANGE row\n"); return false; }
                vals[i] = std::atoi(cell.c_str());
            }
            out.ranges.push_back(MaterialRange{ vals[0], vals[1], vals[2] });
        } else {
            std::fprintf(stderr, "materials: unknown label '%s'\n", label.c_str());
            return false;
        }
    }
    return true;
}

// assign_materials — fold the mesh + material table into the Triangle array
// kernels.cuh's contract expects: one linear scan over the (small) RANGE
// list per triangle (O(triangles * ranges), trivial at this project's
// scale — 2264 triangles x 4 ranges), aborting loudly if any triangle is
// uncovered or a RANGE references an undefined material (a scene-authoring
// bug the generator itself also self-checks — see make_synthetic.py's
// coverage assertion).
static bool assign_materials(const RawMesh& mesh, const MaterialTable& mt, std::vector<Triangle>& out_tris)
{
    out_tris.resize(mesh.faces.size());
    for (size_t i = 0; i < mesh.faces.size(); ++i) {
        int mat_id = -1;
        for (const auto& r : mt.ranges) {
            if (static_cast<int>(i) >= r.start && static_cast<int>(i) <= r.end) { mat_id = r.mat_id; break; }
        }
        if (mat_id < 0) { std::fprintf(stderr, "materials: triangle %zu not covered by any RANGE\n", i); return false; }
        if (mat_id >= static_cast<int>(mt.materials.size())) {
            std::fprintf(stderr, "materials: RANGE references undefined material %d\n", mat_id);
            return false;
        }
        const auto& f = mesh.faces[i];
        Triangle t{};
        for (int a = 0; a < 3; ++a) {
            t.v0[a] = mesh.verts[static_cast<size_t>(f[0])][a];
            t.v1[a] = mesh.verts[static_cast<size_t>(f[1])][a];
            t.v2[a] = mesh.verts[static_cast<size_t>(f[2])][a];
        }
        t.material_id = mat_id;
        out_tris[i] = t;
    }
    return true;
}

// load_sensor_config — label,value rows -> SensorConfig (degrees/mrad
// converted to radians here, once; kernels.cuh's structs are radians-only).
static bool load_sensor_config(const std::string& path, SensorConfig& cfg)
{
    std::ifstream in(path);
    if (!in.is_open()) return false;
    std::map<std::string, double> vals;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string label, cell;
        std::getline(ss, label, ',');
        if (!std::getline(ss, cell, ',')) { std::fprintf(stderr, "sensor_config: short row '%s'\n", label.c_str()); return false; }
        vals[label] = std::strtod(cell.c_str(), nullptr);
    }
    static const char* kRequired[] = {
        "CHANNELS", "AZIMUTH_STEPS", "ELEVATION_MIN_DEG", "ELEVATION_MAX_DEG",
        "AZIMUTH_START_DEG", "RANGE_MIN_M", "RANGE_MAX_M", "DIVERGENCE_HALF_ANGLE_MRAD",
        "SUBRAY_COUNT", "INTENSITY_GAIN", "RANGE_NOISE_BASE_M", "RANGE_NOISE_PER_M",
        "DROPOUT_BASE", "DROPOUT_RANGE_COEFF", "DROPOUT_INCIDENCE_COEFF", "SEED"
    };
    for (const char* key : kRequired) {
        if (vals.find(key) == vals.end()) { std::fprintf(stderr, "sensor_config: missing '%s'\n", key); return false; }
    }
    const double kDegToRad = kPiD / 180.0;
    const double kMradToRad = 0.001;
    cfg.channels = static_cast<int>(vals["CHANNELS"]);
    cfg.azimuth_steps = static_cast<int>(vals["AZIMUTH_STEPS"]);
    cfg.elevation_min_rad = static_cast<float>(vals["ELEVATION_MIN_DEG"] * kDegToRad);
    cfg.elevation_max_rad = static_cast<float>(vals["ELEVATION_MAX_DEG"] * kDegToRad);
    cfg.azimuth_start_rad = static_cast<float>(vals["AZIMUTH_START_DEG"] * kDegToRad);
    cfg.range_min_m = static_cast<float>(vals["RANGE_MIN_M"]);
    cfg.range_max_m = static_cast<float>(vals["RANGE_MAX_M"]);
    cfg.divergence_half_angle_rad = static_cast<float>(vals["DIVERGENCE_HALF_ANGLE_MRAD"] * kMradToRad);
    cfg.subray_count = static_cast<int>(vals["SUBRAY_COUNT"]);
    cfg.intensity_gain = static_cast<float>(vals["INTENSITY_GAIN"]);
    cfg.range_noise_base_m = static_cast<float>(vals["RANGE_NOISE_BASE_M"]);
    cfg.range_noise_per_m = static_cast<float>(vals["RANGE_NOISE_PER_M"]);
    cfg.dropout_base = static_cast<float>(vals["DROPOUT_BASE"]);
    cfg.dropout_range_coeff = static_cast<float>(vals["DROPOUT_RANGE_COEFF"]);
    cfg.dropout_incidence_coeff = static_cast<float>(vals["DROPOUT_INCIDENCE_COEFF"]);
    cfg.seed = static_cast<unsigned int>(vals["SEED"]);
    return true;
}

struct RawPose { int frame_idx; float x, y, z, qw, qx, qy, qz; };

// load_sensor_poses — "POSE,frame_idx,x,y,z,qw,qx,qy,qz" rows (repo
// quaternion order, SYSTEM_DESIGN.md §3.4). v1 uses only frame 0.
static bool load_sensor_poses(const std::string& path, std::vector<RawPose>& out)
{
    std::ifstream in(path);
    if (!in.is_open()) return false;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string label, cell;
        std::getline(ss, label, ',');
        if (label != "POSE") { std::fprintf(stderr, "sensor_poses: unknown label '%s'\n", label.c_str()); return false; }
        double v[8];
        for (double& vi : v) {
            if (!std::getline(ss, cell, ',')) { std::fprintf(stderr, "sensor_poses: short POSE row\n"); return false; }
            vi = std::strtod(cell.c_str(), nullptr);
        }
        RawPose p{};
        p.frame_idx = static_cast<int>(v[0]);
        p.x = static_cast<float>(v[1]); p.y = static_cast<float>(v[2]); p.z = static_cast<float>(v[3]);
        p.qw = static_cast<float>(v[4]); p.qx = static_cast<float>(v[5]);
        p.qy = static_cast<float>(v[6]); p.qz = static_cast<float>(v[7]);
        out.push_back(p);
    }
    return !out.empty();
}

// quat_to_rot — unit quaternion (repo order w,x,y,z) -> row-major 3x3
// rotation (05.01's quat_to_rot pattern, applied here). Defensive
// renormalization guards a hand-edited pose file with a slightly non-unit
// quaternion (SYSTEM_DESIGN.md §3.4: quaternions are "kept normalized").
static void quat_to_rot(float qw, float qx, float qy, float qz, float R[9])
{
    const float n = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
    if (n > 0.0f) { qw /= n; qx /= n; qy /= n; qz /= n; }
    R[0] = 1.0f - 2.0f * (qy * qy + qz * qz);  R[1] = 2.0f * (qx * qy - qz * qw);        R[2] = 2.0f * (qx * qz + qy * qw);
    R[3] = 2.0f * (qx * qy + qz * qw);         R[4] = 1.0f - 2.0f * (qx * qx + qz * qz); R[5] = 2.0f * (qy * qz - qx * qw);
    R[6] = 2.0f * (qx * qz - qy * qw);         R[7] = 2.0f * (qy * qz + qx * qw);        R[8] = 1.0f - 2.0f * (qx * qx + qy * qy);
}

static void identity_R(float R[9])
{
    R[0] = 1; R[1] = 0; R[2] = 0;
    R[3] = 0; R[4] = 1; R[5] = 0;
    R[6] = 0; R[7] = 0; R[8] = 1;
}

// ===========================================================================
// SECTION 2 — the host BVH builder (median-split by triangle COUNT; see
// kernels.cuh's long comment for the depth-guarantee argument this scheme
// buys). GPU BVH construction is project 07.03's dedicated subject; this
// project builds the tree once, correctly, on the host, and spends its
// teaching budget on the RAYCAST that walks it (README §Limitations is
// explicit about this scoping choice).
// ===========================================================================

struct BvhBuilder {
    const std::vector<Triangle>& tris;
    std::vector<int>& tri_indices;             // permutation, reordered in place during the build
    std::vector<BvhNode>& nodes;                // grows via alloc_node(); reserved up front, never reallocates
    std::vector<std::array<float, 3>> centroid; // per ORIGINAL triangle index, precomputed once
    int max_depth = 0;

    BvhBuilder(const std::vector<Triangle>& t, std::vector<int>& ti, std::vector<BvhNode>& n)
        : tris(t), tri_indices(ti), nodes(n)
    {
        centroid.resize(tris.size());
        for (size_t i = 0; i < tris.size(); ++i) {
            const Triangle& tr = tris[i];
            for (int a = 0; a < 3; ++a) centroid[i][a] = (tr.v0[a] + tr.v1[a] + tr.v2[a]) / 3.0f;
        }
    }

    // alloc_node — append a fresh node and return its index. Callers must
    // NEVER hold a BvhNode& across this call: `nodes` was `reserve()`d to
    // its worst-case size up front (see build_bvh_median_split below), so
    // emplace_back never reallocates and INDICES stay valid — but a
    // reference taken before an emplace_back that happened to run before
    // that guarantee was in place would be a classic dangling-reference bug,
    // so this file always re-indexes nodes[idx] fresh instead of caching a
    // reference (see build() below).
    int alloc_node() { nodes.emplace_back(); return static_cast<int>(nodes.size()) - 1; }

    void compute_bounds(int first, int count, float out_min[3], float out_max[3]) const
    {
        for (int a = 0; a < 3; ++a) { out_min[a] = FLT_MAX; out_max[a] = -FLT_MAX; }
        for (int i = first; i < first + count; ++i) {
            const Triangle& t = tris[static_cast<size_t>(tri_indices[i])];
            const float* verts[3] = { t.v0, t.v1, t.v2 };
            for (const float* v : verts)
                for (int a = 0; a < 3; ++a) {
                    out_min[a] = std::min(out_min[a], v[a]);
                    out_max[a] = std::max(out_max[a], v[a]);
                }
        }
    }

    // build — recursively subdivide tri_indices[first, first+count) into
    // node_idx's subtree. THE key step: nth_element partitions the range
    // around its EXACT MIDPOINT by centroid coordinate along the axis with
    // the largest centroid-bounds extent — an O(count) partial sort
    // (average case) that guarantees the triangle COUNT splits in half
    // regardless of how the geometry is arranged in space (kernels.cuh
    // derives why this is what bounds the traversal stack).
    void build(int node_idx, int first, int count, int depth)
    {
        max_depth = std::max(max_depth, depth);

        float amin[3], amax[3];
        compute_bounds(first, count, amin, amax);
        for (int a = 0; a < 3; ++a) { nodes[node_idx].aabb_min[a] = amin[a]; nodes[node_idx].aabb_max[a] = amax[a]; }

        if (count <= kBvhLeafSize) {
            nodes[node_idx].left_first = first;
            nodes[node_idx].tri_count = count;
            return;
        }

        // Split axis = the CENTROID bounds' largest extent (not the full
        // triangle AABB's extent — centroids are what nth_element actually
        // partitions on, so their spread is the relevant measure).
        float cmin[3], cmax[3];
        for (int a = 0; a < 3; ++a) { cmin[a] = FLT_MAX; cmax[a] = -FLT_MAX; }
        for (int i = first; i < first + count; ++i) {
            const auto& c = centroid[static_cast<size_t>(tri_indices[i])];
            for (int a = 0; a < 3; ++a) { cmin[a] = std::min(cmin[a], c[a]); cmax[a] = std::max(cmax[a], c[a]); }
        }
        int axis = 0;
        float best_extent = cmax[0] - cmin[0];
        for (int a = 1; a < 3; ++a) {
            const float e = cmax[a] - cmin[a];
            if (e > best_extent) { best_extent = e; axis = a; }
        }

        const int mid = first + count / 2;   // EXACT half split by count -> the depth guarantee
        std::nth_element(tri_indices.begin() + first, tri_indices.begin() + mid, tri_indices.begin() + first + count,
            [this, axis](int ia, int ib) { return centroid[static_cast<size_t>(ia)][axis] < centroid[static_cast<size_t>(ib)][axis]; });

        const int left = alloc_node();
        const int right = alloc_node();   // always left+1 by construction — kernels.cuh's "children allocated in pairs" contract
        nodes[node_idx].left_first = left;
        nodes[node_idx].tri_count = 0;
        build(left, first, count / 2, depth + 1);
        build(right, mid, count - count / 2, depth + 1);
    }
};

// build_bvh_median_split — the project's one BVH build entry point.
static void build_bvh_median_split(const std::vector<Triangle>& tris, std::vector<BvhNode>& nodes,
                                   std::vector<int>& tri_indices, int& out_max_depth)
{
    const int n = static_cast<int>(tris.size());
    tri_indices.resize(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) tri_indices[static_cast<size_t>(i)] = i;

    nodes.clear();
    // Worst-case node count for a binary tree over n leaves is 2n-1 (every
    // leaf holding exactly 1 triangle); reserving 2n keeps alloc_node()'s
    // emplace_back from EVER reallocating mid-build, which is what makes
    // the "never hold a BvhNode& across alloc_node()" discipline in build()
    // sufficient rather than merely convenient.
    nodes.reserve(static_cast<size_t>(std::max(1, 2 * n)));

    BvhBuilder builder(tris, tri_indices, nodes);
    const int root = builder.alloc_node();
    builder.build(root, 0, n, 0);
    out_max_depth = builder.max_depth;
}

// ===========================================================================
// SECTION 3 — small utilities: data-file discovery, directory creation,
// the sensor-frame beam-direction formula (used ONLY for point-cloud
// packing — see kernels.cuh's output-layout comment for why that is not
// part of the GPU-vs-CPU numeric verification), and the two artifact
// writers.
// ===========================================================================

// looks_like_project_root — a cheap, reliable marker check: every project
// folder in this repo has its own CMakeLists.txt at the root (CLAUDE.md §4
// layout), so its presence is a good signal "this directory IS the project
// root", independent of which build system produced the running executable.
static bool looks_like_project_root(const std::string& dir)
{
    return std::ifstream(dir + "/CMakeLists.txt").is_open();
}

// project_root_from — find the project root relative to the running exe.
// The REQUIRED Visual Studio layout puts the exe 3 directories down
// (build/x64/Release/<exe>.exe — the .vcxproj's pinned OutDir); the OPTIONAL
// CMake path (demo/run_demo.sh) puts it only 2 down for multi-config
// generators (build-cmake/Release/<exe>) or 1 down for single-config ones
// (build-cmake/<exe>). Rather than hardcoding one depth (which would silently
// break artifact-writing under the other build system), try each plausible
// depth and keep the first that actually looks like the project root.
static std::string project_root_from(const char* argv0)
{
    std::string exe(argv0 ? argv0 : "");
    const size_t cut = exe.find_last_of("/\\");
    const std::string exe_dir = (cut == std::string::npos) ? "." : exe.substr(0, cut);

    static const char* kUps[] = { "/..", "/../..", "/../../.." };
    for (const char* up : kUps) {
        const std::string candidate = exe_dir + up;
        if (looks_like_project_root(candidate)) return candidate;
    }
    // Fall back to the required VS layout's depth even if the marker check
    // failed for some reason — find_data_file()'s CWD-relative candidates
    // still cover data loading; only artifact writing depends on this path.
    return exe_dir + "/../../..";
}

static std::string find_data_file(const std::string& name, const std::string& data_dir_override, const char* argv0)
{
    std::vector<std::string> candidates;
    if (!data_dir_override.empty()) candidates.push_back(data_dir_override + "/" + name);
    candidates.push_back(project_root_from(argv0) + "/data/sample/" + name);
    candidates.push_back("data/sample/" + name);
    candidates.push_back("../data/sample/" + name);
    for (const auto& c : candidates)
        if (std::ifstream(c).is_open()) return c;
    return "";
}

static bool ensure_dir(const std::string& path)
{
#ifdef _WIN32
    const int r = _mkdir(path.c_str());
#else
    const int r = mkdir(path.c_str(), 0755);
#endif
    return r == 0 || errno == EEXIST;
}

// beam_direction_sensor — the SAME elevation/azimuth -> direction formula
// simulate_beam() uses (kernels.cu / reference_cpu.cpp), evaluated here in
// the SENSOR frame only (no pose rotation) so main.cu can pack surviving
// beams into a "lidar"-frame PointCloud (SYSTEM_DESIGN.md §3.6) without
// re-deriving the sensor's world pose per point. This is the formula's
// THIRD appearance (GPU, CPU oracle, and here) — deliberately: it is pure,
// branchless trigonometry with no BVH/RNG involved, so a third copy cannot
// introduce a divergent bug the §5 gate would need to catch; it only feeds
// artifact packing, never a checked numeric comparison.
static void beam_direction_sensor(int channel, int az_idx, const SensorConfig& cfg, float dir_out[3])
{
    const double frac = (cfg.channels > 1)
        ? static_cast<double>(channel) / static_cast<double>(cfg.channels - 1)
        : 0.0;
    const double elevation = static_cast<double>(cfg.elevation_min_rad)
        + frac * (static_cast<double>(cfg.elevation_max_rad) - static_cast<double>(cfg.elevation_min_rad));
    const double azimuth = static_cast<double>(cfg.azimuth_start_rad)
        + static_cast<double>(az_idx) * (2.0 * kPiD / static_cast<double>(cfg.azimuth_steps));
    const double ce = std::cos(elevation), se = std::sin(elevation);
    const double ca = std::cos(azimuth), sa = std::sin(azimuth);
    dir_out[0] = static_cast<float>(ce * ca);
    dir_out[1] = static_cast<float>(ce * sa);
    dir_out[2] = static_cast<float>(se);
}

// write_cloud_csv — the plottable artifact: one row per surviving beam
// (hit && !dropped), SENSOR frame (SYSTEM_DESIGN.md §3.6's PointCloud,
// header.frame_id = "lidar" in spirit — the sensor does not know its own
// world pose, only its own ray directions and measured ranges).
static bool write_cloud_csv(const std::string& path, const SensorConfig& cfg,
                            const std::vector<float>& range, const std::vector<float>& intensity,
                            const std::vector<int>& hit, const std::vector<int>& dropped,
                            int& out_point_count)
{
    std::ofstream f(path);
    if (!f.is_open()) return false;
    f << "x_m,y_m,z_m,intensity,ring\n";
    int count = 0;
    for (int channel = 0; channel < cfg.channels; ++channel) {
        for (int az = 0; az < cfg.azimuth_steps; ++az) {
            const int beam = channel * cfg.azimuth_steps + az;
            if (!hit[static_cast<size_t>(beam)] || dropped[static_cast<size_t>(beam)]) continue;
            float dir[3];
            beam_direction_sensor(channel, az, cfg, dir);
            const float r = range[static_cast<size_t>(beam)];
            f << (dir[0] * r) << ',' << (dir[1] * r) << ',' << (dir[2] * r) << ','
              << intensity[static_cast<size_t>(beam)] << ',' << channel << '\n';
            ++count;
        }
    }
    out_point_count = count;
    return f.good();
}

// write_range_image_pgm — a channels x azimuth_steps grayscale range image,
// the LiDAR's own native "picture" (THEORY.md explains why organized
// point clouds are naturally an image). Binary PGM (P5): a tiny, fully
// hand-rolled format (3-line ASCII header, then raw bytes — CLAUDE.md §5's
// "no black boxes" stance applied to image I/O too). Pixel value = 255 *
// (1 - range/range_max): CLOSER surfaces are BRIGHTER; no-return beams
// (missed geometry or dropped) are BLACK (0).
static bool write_range_image_pgm(const std::string& path, const SensorConfig& cfg,
                                  const std::vector<float>& range,
                                  const std::vector<int>& hit, const std::vector<int>& dropped)
{
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    f << "P5\n" << cfg.azimuth_steps << ' ' << cfg.channels << "\n255\n";
    std::vector<unsigned char> pixels(static_cast<size_t>(cfg.channels) * static_cast<size_t>(cfg.azimuth_steps), 0);
    for (size_t beam = 0; beam < pixels.size(); ++beam) {
        if (!hit[beam] || dropped[beam]) { pixels[beam] = 0; continue; }
        float frac = 1.0f - (range[beam] / cfg.range_max_m);
        frac = std::min(std::max(frac, 0.0f), 1.0f);
        pixels[beam] = static_cast<unsigned char>(frac * 255.0f + 0.5f);
    }
    f.write(reinterpret_cast<const char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
    return f.good();
}

// run_cpu — thin helper: size the four SoA output arrays for cfg's beam
// count and call lidar_raycast_cpu. Used by every analytic gate below so
// each gate reads as "build a tiny SensorConfig, run_cpu, check the
// numbers" without repeating the allocation boilerplate.
static void run_cpu(const std::vector<Triangle>& tris, const std::vector<Material>& materials,
                    const std::vector<BvhNode>& nodes, const std::vector<int>& tri_indices,
                    const SensorConfig& cfg, const SensorPose& pose,
                    std::vector<float>& range, std::vector<float>& intensity,
                    std::vector<int>& hit, std::vector<int>& dropped)
{
    const int n = cfg.channels * cfg.azimuth_steps;
    range.assign(static_cast<size_t>(n), 0.0f);
    intensity.assign(static_cast<size_t>(n), 0.0f);
    hit.assign(static_cast<size_t>(n), 0);
    dropped.assign(static_cast<size_t>(n), 0);
    lidar_raycast_cpu(tris.data(), materials.data(), nodes.data(), tri_indices.data(),
                      cfg, pose, range.data(), intensity.data(), hit.data(), dropped.data());
}

// ---------------------------------------------------------------------------
// main — load, build, verify, gate, pack, report. See the file header for
// the numbered stages this function walks through.
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    std::string data_dir_override;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--data") && i + 1 < argc) data_dir_override = argv[++i];
        else {
            std::fprintf(stderr, "usage: %s [--data data/sample]\n", argv[0]);
            return 2;
        }
    }

    std::printf("[demo] GPU LiDAR simulator: BVH raycasting + beam divergence, intensity, dropout noise (project 11.01)\n");
    print_device_info();

    // ======================= LOAD =============================================
    const std::string obj_path = find_data_file("warehouse_scene.obj", data_dir_override, argv[0]);
    const std::string mat_path = find_data_file("materials.csv", data_dir_override, argv[0]);
    const std::string cfg_path = find_data_file("sensor_config.csv", data_dir_override, argv[0]);
    const std::string pose_path = find_data_file("sensor_poses.csv", data_dir_override, argv[0]);
    if (obj_path.empty() || mat_path.empty() || cfg_path.empty() || pose_path.empty()) {
        std::printf("SCENE: NOT FOUND — data/sample/*.{obj,csv} missing (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (sample data missing)\n");
        return 1;
    }

    RawMesh mesh;
    MaterialTable mat_table;
    std::vector<Triangle> tris;
    if (!load_obj(obj_path, mesh) || !load_materials(mat_path, mat_table) || !assign_materials(mesh, mat_table, tris)) {
        std::printf("SCENE: MALFORMED — see stderr\n");
        std::printf("RESULT: FAIL (scene malformed)\n");
        return 1;
    }
    std::printf("SCENE: %zu triangles, %zu vertices, %zu materials [synthetic]\n",
               tris.size(), mesh.verts.size(), mat_table.materials.size());

    SensorConfig cfg{};
    if (!load_sensor_config(cfg_path, cfg)) {
        std::printf("PROBLEM: SENSOR CONFIG MALFORMED — see stderr\n");
        std::printf("RESULT: FAIL (sensor config malformed)\n");
        return 1;
    }
    std::vector<RawPose> poses;
    if (!load_sensor_poses(pose_path, poses)) {
        std::printf("PROBLEM: SENSOR POSE MALFORMED — see stderr\n");
        std::printf("RESULT: FAIL (sensor pose malformed)\n");
        return 1;
    }
    SensorPose pose{};
    quat_to_rot(poses[0].qw, poses[0].qx, poses[0].qy, poses[0].qz, pose.R);
    pose.t[0] = poses[0].x; pose.t[1] = poses[0].y; pose.t[2] = poses[0].z;

    // ======================= BUILD THE BVH (host, once) =======================
    std::vector<BvhNode> nodes;
    std::vector<int> tri_indices;
    int max_depth = 0;
    CpuTimer build_timer;
    build_timer.begin();
    build_bvh_median_split(tris, nodes, tri_indices, max_depth);
    const double build_ms = build_timer.end_ms();
    int num_leaves = 0;
    for (const auto& n : nodes) if (n.tri_count > 0) ++num_leaves;
    const int depth_bound = static_cast<int>(std::ceil(std::log2(std::max(1, num_leaves))));
    std::printf("BVH: %zu nodes, %d leaves, depth %d (guaranteed bound %d), median-split by count, built on the host\n",
               nodes.size(), num_leaves, max_depth, depth_bound);
    std::printf("[time] BVH build: %.3f ms (host, one-time, at load — not part of the per-frame raycast cost)\n", build_ms);

    const int num_beams = cfg.channels * cfg.azimuth_steps;
    std::printf("PROBLEM: spinning LiDAR raycast, channels=%d x azimuth_steps=%d = %d beams/frame, BVH over %zu triangles, FP32\n",
               cfg.channels, cfg.azimuth_steps, num_beams, tris.size());

    // ======================= UPLOAD (persistent, one frame's worth) ===========
    Triangle* d_tris = nullptr;
    Material* d_materials = nullptr;
    BvhNode* d_nodes = nullptr;
    int* d_tri_indices = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tris, tris.size() * sizeof(Triangle)));
    CUDA_CHECK(cudaMalloc(&d_materials, mat_table.materials.size() * sizeof(Material)));
    CUDA_CHECK(cudaMalloc(&d_nodes, nodes.size() * sizeof(BvhNode)));
    CUDA_CHECK(cudaMalloc(&d_tri_indices, tri_indices.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_tris, tris.data(), tris.size() * sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_materials, mat_table.materials.data(), mat_table.materials.size() * sizeof(Material), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nodes, nodes.data(), nodes.size() * sizeof(BvhNode), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tri_indices, tri_indices.data(), tri_indices.size() * sizeof(int), cudaMemcpyHostToDevice));

    float* d_range = nullptr; float* d_intensity = nullptr;
    int* d_hit = nullptr; int* d_dropped = nullptr;
    CUDA_CHECK(cudaMalloc(&d_range, static_cast<size_t>(num_beams) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_intensity, static_cast<size_t>(num_beams) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hit, static_cast<size_t>(num_beams) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dropped, static_cast<size_t>(num_beams) * sizeof(int)));

    // ======================= VERIFY STAGE (§5 GPU-vs-CPU gate) ================
    // Full demo frame through BOTH paths. hit/dropped are DETERMINISTIC
    // integer decisions (geometry + an RNG stream that is bit-identical
    // between GPU and CPU by construction) and must match EXACTLY; range/
    // intensity are FP32 arithmetic chained through a tree traversal +
    // trig, and are compared within kVerifyTol (THEORY.md quantifies the
    // measured divergence).
    std::vector<float> range_gpu(static_cast<size_t>(num_beams)), intensity_gpu(static_cast<size_t>(num_beams));
    std::vector<int> hit_gpu(static_cast<size_t>(num_beams)), dropped_gpu(static_cast<size_t>(num_beams));
    GpuTimer gt;
    gt.begin();
    launch_lidar_raycast(d_tris, d_materials, d_nodes, d_tri_indices, cfg, pose,
                        d_range, d_intensity, d_hit, d_dropped);
    const float gpu_ms = gt.end_ms();
    CUDA_CHECK(cudaMemcpy(range_gpu.data(), d_range, range_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(intensity_gpu.data(), d_intensity, intensity_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hit_gpu.data(), d_hit, hit_gpu.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dropped_gpu.data(), d_dropped, dropped_gpu.size() * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<float> range_cpu, intensity_cpu;
    std::vector<int> hit_cpu, dropped_cpu;
    CpuTimer ct;
    ct.begin();
    run_cpu(tris, mat_table.materials, nodes, tri_indices, cfg, pose, range_cpu, intensity_cpu, hit_cpu, dropped_cpu);
    const double cpu_ms = ct.end_ms();

    int hit_mismatches = 0, drop_mismatches = 0;
    float worst_range_rel = 0.0f, worst_intensity_rel = 0.0f;
    int hits_total = 0, range_outliers = 0;   // outliers: beams exceeding kVerifyRangeTol — the "near-tie edge" beams (see the tolerance comment above)
    for (int i = 0; i < num_beams; ++i) {
        if (hit_gpu[static_cast<size_t>(i)] != hit_cpu[static_cast<size_t>(i)]) ++hit_mismatches;
        if (dropped_gpu[static_cast<size_t>(i)] != dropped_cpu[static_cast<size_t>(i)]) ++drop_mismatches;
        if (hit_cpu[static_cast<size_t>(i)]) {
            ++hits_total;
            const float rc = range_cpu[static_cast<size_t>(i)];
            const float scale_r = std::fabs(rc) > 1.0f ? std::fabs(rc) : 1.0f;
            const float dr = std::fabs(range_gpu[static_cast<size_t>(i)] - rc) / scale_r;
            if (dr > worst_range_rel) worst_range_rel = dr;
            if (dr > kVerifyRangeTol) ++range_outliers;

            const float ic = intensity_cpu[static_cast<size_t>(i)];
            const float scale_i = std::fabs(ic) > 1.0f ? std::fabs(ic) : 1.0f;
            const float di = std::fabs(intensity_gpu[static_cast<size_t>(i)] - ic) / scale_i;
            if (di > worst_intensity_rel) worst_intensity_rel = di;
        }
    }
    const bool verify_pass = (hit_mismatches == 0) && (drop_mismatches == 0)
                           && (range_outliers == 0) && (worst_intensity_rel < kVerifyIntensityTol);
    std::printf("[info] verify: %d/%d hit mismatches, %d/%d dropped mismatches, worst relative deviation range=%.3e (of %d hit beams) intensity=%.3e\n",
               hit_mismatches, num_beams, drop_mismatches, num_beams,
               static_cast<double>(worst_range_rel), hits_total, static_cast<double>(worst_intensity_rel));
    std::printf("[time] full frame (%d beams): CPU %.1f ms | GPU kernel %.3f ms | speed-up %.0fx (teaching artifact; kernel only)\n",
               num_beams, cpu_ms, static_cast<double>(gpu_ms),
               cpu_ms / (static_cast<double>(gpu_ms) > 0.0 ? static_cast<double>(gpu_ms) : 1.0));
    std::printf("VERIFY: %s (GPU raycast matches CPU reference: hit/dropped exact, intensity within rel tol %.0e, range within rel tol %.0e for all but documented near-tie edge beams)\n",
               verify_pass ? "PASS" : "FAIL", static_cast<double>(kVerifyIntensityTol), static_cast<double>(kVerifyRangeTol));

    // ======================= ANALYTIC GATES (CPU oracle only) =================
    // See kernels.cuh's long comment on lidar_raycast_cpu for why running
    // these through the CPU function alone is sufficient: the §5 gate just
    // above already proves the GPU kernel matches this function bit-for-bit
    // (within tolerance); these gates validate the PHYSICS MODEL itself.

    // --- Gate A: ground-plane range vs the closed form h/sin(|elevation|) ---
    bool gateA_pass = false;
    {
        SensorConfig g{};
        g.channels = 1; g.azimuth_steps = 1;
        g.elevation_min_rad = g.elevation_max_rad = kGateGroundElevationRad;
        g.azimuth_start_rad = 0.0f;   // +x — the corridor make_synthetic.py keeps clear
        g.range_min_m = 0.05f; g.range_max_m = 50.0f;
        g.divergence_half_angle_rad = 0.0f; g.subray_count = 0;   // "effects disabled" — pure geometry
        g.intensity_gain = cfg.intensity_gain;
        g.range_noise_base_m = 0.0f; g.range_noise_per_m = 0.0f;
        g.dropout_base = 0.0f; g.dropout_range_coeff = 0.0f; g.dropout_incidence_coeff = 0.0f;
        g.seed = 42u;
        SensorPose p{}; identity_R(p.R); p.t[0] = 0.0f; p.t[1] = 0.0f; p.t[2] = kGateHeight1M;

        std::vector<float> r, in; std::vector<int> h, d;
        run_cpu(tris, mat_table.materials, nodes, tri_indices, g, p, r, in, h, d);

        const double expected = static_cast<double>(kGateHeight1M) / std::sin(std::fabs(static_cast<double>(kGateGroundElevationRad)));
        const double rel_err = std::fabs(static_cast<double>(r[0]) - expected) / expected;
        gateA_pass = (h[0] == 1) && (d[0] == 0) && (rel_err < 1e-3);
        std::printf("[info] gate A: measured range %.6f m, closed-form h/sin(|elevation|) = %.6f m, rel error %.3e\n",
                   static_cast<double>(r[0]), expected, rel_err);
        std::printf("CHECK: ground-plane range matches closed form within rel tol 1e-3 -> %s\n", gateA_pass ? "PASS" : "FAIL");
    }

    // --- Gate B: normal-incidence intensity, R vs 2R -> exact 4:1 ratio -----
    bool gateB_pass = false;
    {
        auto make_nadir = [&](float height, SensorConfig& g, SensorPose& p) {
            g = SensorConfig{};
            g.channels = 1; g.azimuth_steps = 1;
            g.elevation_min_rad = g.elevation_max_rad = static_cast<float>(-kPiD * 0.5);   // straight down
            g.azimuth_start_rad = 0.0f;
            g.range_min_m = 0.05f; g.range_max_m = 50.0f;
            g.divergence_half_angle_rad = 0.0f; g.subray_count = 0;
            g.intensity_gain = cfg.intensity_gain;
            g.range_noise_base_m = 0.0f; g.range_noise_per_m = 0.0f;
            g.dropout_base = 0.0f; g.dropout_range_coeff = 0.0f; g.dropout_incidence_coeff = 0.0f;
            g.seed = 42u;
            p = SensorPose{}; identity_R(p.R); p.t[0] = 0.0f; p.t[1] = 0.0f; p.t[2] = height;
        };
        SensorConfig g1, g2; SensorPose p1, p2;
        make_nadir(kGateHeight1M, g1, p1);
        make_nadir(kGateHeight2M, g2, p2);

        std::vector<float> r1, in1; std::vector<int> h1, d1;
        std::vector<float> r2, in2; std::vector<int> h2, d2;
        run_cpu(tris, mat_table.materials, nodes, tri_indices, g1, p1, r1, in1, h1, d1);
        run_cpu(tris, mat_table.materials, nodes, tri_indices, g2, p2, r2, in2, h2, d2);

        const double ratio = static_cast<double>(in1[0]) / static_cast<double>(in2[0]);
        gateB_pass = (h1[0] == 1) && (h2[0] == 1) && (d1[0] == 0) && (d2[0] == 0) && (std::fabs(ratio - 4.0) < 0.01);
        std::printf("[info] gate B: intensity(R=%.1fm)=%.6f, intensity(R=%.1fm)=%.6f, ratio=%.6f (expect 4.000)\n",
                   static_cast<double>(kGateHeight1M), static_cast<double>(in1[0]),
                   static_cast<double>(kGateHeight2M), static_cast<double>(in2[0]), ratio);
        std::printf("CHECK: normal-incidence intensity ratio (R vs 2R) matches inverse-square 4:1 within tol 0.01 -> %s\n",
                   gateB_pass ? "PASS" : "FAIL");
    }

    // --- Gate C: dropout rate over many i.i.d. beams vs the configured model
    bool gateC_pass = false;
    {
        SensorConfig g{};
        g.channels = 1; g.azimuth_steps = kGateDropoutBeams;
        g.elevation_min_rad = g.elevation_max_rad = static_cast<float>(-kPiD * 0.5);   // straight down: same R, same cos_theta=1 for every beam
        g.azimuth_start_rad = 0.0f;
        g.range_min_m = 0.05f; g.range_max_m = cfg.range_max_m;   // reuse the REAL range_max_m so the dropout formula's range term matches the deployed model
        g.divergence_half_angle_rad = 0.0f; g.subray_count = 0;
        g.intensity_gain = cfg.intensity_gain;
        g.range_noise_base_m = 0.0f; g.range_noise_per_m = 0.0f;   // keep R exactly kGateHeight1M for a clean theoretical p
        g.dropout_base = cfg.dropout_base;
        g.dropout_range_coeff = cfg.dropout_range_coeff;
        g.dropout_incidence_coeff = cfg.dropout_incidence_coeff;
        g.seed = 42u;
        SensorPose p{}; identity_R(p.R); p.t[0] = 0.0f; p.t[1] = 0.0f; p.t[2] = kGateHeight1M;

        std::vector<float> r, in; std::vector<int> h, d;
        run_cpu(tris, mat_table.materials, nodes, tri_indices, g, p, r, in, h, d);

        long hits = 0, drops = 0;
        for (int i = 0; i < kGateDropoutBeams; ++i) {
            if (h[static_cast<size_t>(i)]) { ++hits; if (d[static_cast<size_t>(i)]) ++drops; }
        }
        double p_theory = static_cast<double>(cfg.dropout_base)
            + static_cast<double>(cfg.dropout_range_coeff) * (static_cast<double>(kGateHeight1M) / static_cast<double>(cfg.range_max_m))
            + static_cast<double>(cfg.dropout_incidence_coeff) * (1.0 - 1.0);
        p_theory = std::min(std::max(p_theory, 0.0), 1.0);
        const double p_emp = hits > 0 ? static_cast<double>(drops) / static_cast<double>(hits) : -1.0;
        const double sigma = hits > 0 ? std::sqrt(p_theory * (1.0 - p_theory) / static_cast<double>(hits)) : 1.0;
        const double bound = kGateDropoutSigmaMultiple * sigma;
        gateC_pass = (hits == kGateDropoutBeams) && (std::fabs(p_emp - p_theory) < bound);
        std::printf("[info] gate C: %ld/%d beams hit, empirical dropout rate %.5f, theoretical %.5f, %ldsigma bound +-%.5f\n",
                   hits, kGateDropoutBeams, p_emp, p_theory, static_cast<long>(kGateDropoutSigmaMultiple), bound);
        std::printf("CHECK: dropout rate matches configured model within a %.0f-sigma binomial bound -> %s\n",
                   kGateDropoutSigmaMultiple, gateC_pass ? "PASS" : "FAIL");
    }

    // ======================= FRAME-LEVEL CHECKS (from the GPU main frame) =====
    long hit_count = 0, valid_count = 0;
    double range_sum = 0.0;
    for (int i = 0; i < num_beams; ++i) {
        if (hit_gpu[static_cast<size_t>(i)]) {
            ++hit_count;
            if (!dropped_gpu[static_cast<size_t>(i)]) { ++valid_count; range_sum += static_cast<double>(range_gpu[static_cast<size_t>(i)]); }
        }
    }
    const double hit_fraction = static_cast<double>(hit_count) / static_cast<double>(num_beams);
    const double mean_range = valid_count > 0 ? range_sum / static_cast<double>(valid_count) : 0.0;
    // Bounds below are MEASURED on the committed scene/config (RTX 2080
    // SUPER, this seed) with generous documented margin — not asserted from
    // theory, because "how much of a warehouse a centered LiDAR sees" has
    // no closed form; see README §Expected output for the measured numbers.
    const bool gateD_pass = (hit_fraction > 0.40) && (hit_fraction < 0.95);
    const bool gateE_pass = (mean_range > 1.0) && (mean_range < 20.0);
    std::printf("[info] frame stats: hit fraction %.4f (%ld/%d), mean range of returned points %.3f m (%ld points)\n",
               hit_fraction, hit_count, num_beams, mean_range, valid_count);
    std::printf("CHECK: frame hit fraction in documented range (0.40, 0.95) -> %s\n", gateD_pass ? "PASS" : "FAIL");
    std::printf("CHECK: frame mean range in documented range (1.0, 20.0) m -> %s\n", gateE_pass ? "PASS" : "FAIL");

    // ======================= ARTIFACTS =========================================
    const std::string out_dir = project_root_from(argv[0]) + "/demo/out";
    bool artifacts_ok = ensure_dir(out_dir);
    int point_count = 0;
    if (artifacts_ok) {
        artifacts_ok = write_cloud_csv(out_dir + "/cloud.csv", cfg, range_gpu, intensity_gpu, hit_gpu, dropped_gpu, point_count);
    }
    if (artifacts_ok) {
        // The exact point count is a MEASURED result (it depends on the
        // hit/dropout decisions at every beam, some of which sit at
        // near-tie FP boundaries — see the VERIFY-tolerance comment above);
        // it is genuinely informative but not the kind of number this
        // project's stable output contract commits to across every
        // machine, so it lives on an unchecked "[info]" line, not the
        // checked "ARTIFACT:" line (contrast with 08.01's step count,
        // which is a fixed INPUT parameter and safe to check verbatim).
        std::printf("[info] cloud.csv: %d points\n", point_count);
        std::printf("ARTIFACT: wrote demo/out/cloud.csv\n");
    } else {
        std::printf("ARTIFACT: FAILED to write demo/out/cloud.csv\n");
    }

    bool pgm_ok = artifacts_ok && write_range_image_pgm(out_dir + "/range_image.pgm", cfg, range_gpu, hit_gpu, dropped_gpu);
    if (pgm_ok)
        std::printf("ARTIFACT: wrote demo/out/range_image.pgm (%dx%d)\n", cfg.azimuth_steps, cfg.channels);
    else
        std::printf("ARTIFACT: FAILED to write demo/out/range_image.pgm\n");

    // ======================= CLEANUP + VERDICT =================================
    CUDA_CHECK(cudaFree(d_tris));
    CUDA_CHECK(cudaFree(d_materials));
    CUDA_CHECK(cudaFree(d_nodes));
    CUDA_CHECK(cudaFree(d_tri_indices));
    CUDA_CHECK(cudaFree(d_range));
    CUDA_CHECK(cudaFree(d_intensity));
    CUDA_CHECK(cudaFree(d_hit));
    CUDA_CHECK(cudaFree(d_dropped));

    const bool success = verify_pass && gateA_pass && gateB_pass && gateC_pass && gateD_pass && gateE_pass && artifacts_ok && pgm_ok;
    if (success)
        std::printf("RESULT: PASS (GPU/CPU agree; all analytic and frame-level gates passed; artifacts written)\n");
    else
        std::printf("RESULT: FAIL (see VERIFY/CHECK/ARTIFACT lines above for which gate failed)\n");
    return success ? 0 : 1;
}
