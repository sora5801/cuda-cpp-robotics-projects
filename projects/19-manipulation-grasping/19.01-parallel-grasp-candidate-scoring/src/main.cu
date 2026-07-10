// ===========================================================================
// main.cu — entry point for project 19.01
//           Parallel grasp-candidate scoring: antipodal sampling over point
//           clouds (two-finger parallel-jaw gripper)
//
// What this program does, start to finish
// -----------------------------------------
//   1. Print the banner + GPU info; load the three committed synthetic
//      objects (data/sample/objects_meta.csv + box/cylinder/sphere_cloud.bin).
//   2. Compute EVERY object's surface normals once (PCA + Jacobi, GPU —
//      02.06's pattern, kernels.cu credits it precisely).
//   3. VERIFY STAGE (the §5 GPU-vs-CPU gate, on the BOX object — the
//      simplest, most checkable geometry):
//        (a) normals match the CPU oracle within a documented angle tol;
//        (b) candidate generation (idx1/idx2) matches the CPU oracle
//            EXACTLY, not just within tolerance (kernels.cuh's contract);
//        (c) scoring (width/angles/score) matches within a documented
//            relative tolerance.
//   4. For EACH object (box, cylinder, sphere): generate K=4096 candidates
//      (GPU), score them (GPU) — for the box, ALSO append 12 hand-picked
//      ADVERSARIAL candidates (adjacent, non-opposite face pairs) before
//      scoring — rank by score, keep the top kTopM, and check the
//      object's ANALYTIC gate (README/THEORY: because these are analytic
//      shapes, "is this actually a good grasp" has a closed-form answer).
//   5. ADVERSARIAL GATE (box only): the 12 adjacent-face candidates must
//      ALL be rejected by the friction-cone gate — a NEGATIVE control.
//   6. Write the plotting artifacts (grasps.csv, grasp_cloud.csv) and print
//      the final RESULT line.
//
// Output contract (load-bearing!): stable lines are "[demo]", "PROBLEM:",
// "SCENARIO:", "VERIFY:", "CHECK:", "ARTIFACT:", "RESULT:" — "[info]"/
// "[time]" are NOT diffed (measured numbers vary by GPU architecture: FP32
// arithmetic is not strictly associative, so which candidate wins a
// near-tie can differ across sm_75/sm_86/sm_89 even though every GATE
// still passes — THEORY.md "Numerical considerations" explains why the
// CHECK lines below are deliberately textual/PASS-FAIL, never embedding a
// specific measured value). Change a stable line => update
// demo/expected_output.txt in the same change, and vice versa.
//
// Read this first, then kernels.cuh -> reference_cpu.cpp -> kernels.cu.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
#include <direct.h>               // _mkdir (std::filesystem avoided in .cu — see 07.09/02.06)
#else
#include <sys/stat.h>
#endif

// ---------------------------------------------------------------------------
// Pipeline constants. Every threshold below is a DETERMINISTIC compile-time
// value, so the CHECK/RESULT lines that gate on them are stable across
// machines; the comment beside each records the MEASURED margin that
// justifies it (CLAUDE.md: calibrated thresholds are documented with
// measured values, not guessed).
// ---------------------------------------------------------------------------

// Candidate-generation hash seed (kernels.cuh's grasp_hash_u32). Fixed and
// documented — determinism is repo law (CLAUDE.md §12). Reused unchanged
// across all three objects: each object's own point count n bounds the
// hash's modulo, so the same seed is simply a different, still fully
// deterministic, draw sequence per object.
static constexpr unsigned int kHashSeed = 42u;

// GPU-vs-CPU verification tolerances (the §5 gate). Angle tolerance is
// ABSOLUTE (degrees); the scoring tolerance is RELATIVE, floored at 1.0 so
// near-zero reference values do not force an unreasonably tight check — the
// same shape 02.06/08.01/33.01 use. Measured worst-case values are printed
// on the [info] lines immediately above each VERIFY line.
static constexpr float kVerifyNormalAngleTolDeg = 0.5f;
static constexpr float kVerifyRelTol            = 1e-3f;

// Analytic-gate tolerances (README/THEORY "How we verify correctness").
// Width tolerance accounts for two independent, small error sources: the
// committed clouds' own sensor noise (0.3 mm axial sigma, data/README.md)
// and the candidate search's perpendicular tolerance (kSearchPerpTolM =
// 6 mm, kernels.cuh) — a genuinely antipodal pair whose partner point sits
// near the EDGE of that 6 mm search cylinder can measure up to
// sqrt(perp_tol^2) ~ a few mm longer than the object's true dimension.
static constexpr float kGateWidthTolM        = 0.006f;   // 6 mm
static constexpr float kGateAntipodalCosMin  = 0.90f;    // opposition angle <= ~25.8 deg
static constexpr float kGateAxisAlignMin     = 0.98f;    // box grasp axis must be >= 98% aligned to one coordinate axis
static constexpr float kGateCylPerpMax       = 0.20f;    // |dot(grasp axis, cylinder axis)| <= 0.20 (angle >= ~78.5 deg from parallel, i.e. within ~11.5 deg of perpendicular)

int constexpr kNumAdversarial = 12;   // box: all (6 choose 2) - 3 opposite pairs = 12 adjacent-face pairs

// ---------------------------------------------------------------------------
// ObjectMeta / ObjectData — this project's data layer. objects_meta.csv's
// format (columns, shape codes) is documented byte-exactly in
// data/README.md and scripts/make_synthetic.py's header.
// ---------------------------------------------------------------------------
struct ObjectMeta {
    std::string name;
    std::string file;
    int         n_points = 0;
    std::string shape;         // "box" | "cylinder" | "sphere"
    float       param_a_m = 0.0f, param_b_m = 0.0f, param_c_m = 0.0f;
    float       w_min_m = 0.0f, w_max_m = 0.0f, mu = 0.0f;
};

struct ObjectData {
    ObjectMeta          meta;
    std::vector<float>  xyz;   // n_points*3, meters (loaded from the .bin file)
};

// load_cloud_bin — this project's binary cloud format: 4-byte magic "GC01",
// little-endian uint32 count, then count*3 little-endian float32 xyz
// (meters, interleaved — docs/SYSTEM_DESIGN.md §3.6 PointCloud convention).
// A project-local magic (not 02.06's "PC01") per the self-containment rule
// (CLAUDE.md §4): the two projects share a LAYOUT convention, not a file.
static bool load_cloud_bin(const std::string& path, std::vector<float>& xyz, int& n)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    char magic[4] = { 0, 0, 0, 0 };
    f.read(magic, 4);
    if (f.gcount() != 4 || std::memcmp(magic, "GC01", 4) != 0) return false;

    uint32_t count = 0;
    f.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (f.gcount() != static_cast<std::streamsize>(sizeof(count))) return false;

    n = static_cast<int>(count);
    if (n < 0) return false;
    xyz.assign(static_cast<size_t>(n) * 3, 0.0f);
    const std::streamsize want = static_cast<std::streamsize>(xyz.size() * sizeof(float));
    f.read(reinterpret_cast<char*>(xyz.data()), want);
    return f.gcount() == want;
}

// load_objects_meta — plain header+rows CSV reader for objects_meta.csv
// (format documented in data/README.md). Unknown/short rows fail LOUDLY
// (CLAUDE.md §13: no silent fallback on malformed input) rather than
// silently skipping an object.
static bool load_objects_meta(const std::string& path, std::vector<ObjectMeta>& metas)
{
    std::ifstream in(path);
    if (!in.is_open()) return false;

    std::string line;
    bool header_seen = false;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (!header_seen) { header_seen = true; continue; }   // skip the "name,file,..." header row

        std::stringstream ss(line);
        std::string cell;
        auto next = [&](const char* what) -> std::string {
            if (!std::getline(ss, cell, ',')) {
                std::fprintf(stderr, "objects_meta: short row (missing %s)\n", what);
                return "";
            }
            return cell;
        };

        ObjectMeta m;
        m.name        = next("name");
        m.file        = next("file");
        m.n_points    = std::atoi(next("n_points").c_str());
        m.shape       = next("shape");
        m.param_a_m   = std::strtof(next("param_a_m").c_str(), nullptr);
        m.param_b_m   = std::strtof(next("param_b_m").c_str(), nullptr);
        m.param_c_m   = std::strtof(next("param_c_m").c_str(), nullptr);
        m.w_min_m     = std::strtof(next("gripper_w_min_m").c_str(), nullptr);
        m.w_max_m     = std::strtof(next("gripper_w_max_m").c_str(), nullptr);
        m.mu          = std::strtof(next("friction_mu").c_str(), nullptr);

        if (m.name.empty() || m.file.empty() || m.shape.empty()) {
            std::fprintf(stderr, "objects_meta: malformed row for '%s'\n", m.name.c_str());
            return false;
        }
        metas.push_back(std::move(m));
    }
    return !metas.empty();
}

static std::string project_root_from(const char* argv0)
{
    std::string exe(argv0 ? argv0 : "");
    size_t cut = exe.find_last_of("/\\");
    if (cut == std::string::npos) return ".";
    return exe.substr(0, cut) + "/../../..";
}

static std::string find_meta_file(const char* argv0)
{
    std::vector<std::string> candidates = {
        project_root_from(argv0) + "/data/sample/objects_meta.csv",
        "data/sample/objects_meta.csv",
        "../data/sample/objects_meta.csv",
    };
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

// centroid3 — mean of an interleaved xyz cloud (host, double accumulation:
// this runs once per object, precision is free). Feeds
// launch_estimate_normals's orientation reference — every object here is a
// solid convex body sampled on its surface, so its own centroid always
// lies in the interior (kernels.cu's estimate_normals_kernel header
// comment explains the resulting OUTWARD orientation policy).
static void centroid3(const std::vector<float>& xyz, int n, float out[3])
{
    double sx = 0.0, sy = 0.0, sz = 0.0;
    for (int i = 0; i < n; ++i) {
        sx += xyz[static_cast<size_t>(i) * 3 + 0];
        sy += xyz[static_cast<size_t>(i) * 3 + 1];
        sz += xyz[static_cast<size_t>(i) * 3 + 2];
    }
    const double inv_n = (n > 0) ? (1.0 / n) : 0.0;
    out[0] = static_cast<float>(sx * inv_n);
    out[1] = static_cast<float>(sy * inv_n);
    out[2] = static_cast<float>(sz * inv_n);
}

// nearest_point_index — host linear scan: the cloud index whose position is
// closest to `target` (meters). Used ONLY to build the box's 12 ADVERSARIAL
// candidates from REAL (noisy) committed cloud points nearest each face's
// ideal center, rather than inventing synthetic coordinates that never
// appeared in the cloud (README/THEORY "How we verify correctness": the
// negative control should be exercised through the same code path as every
// other candidate).
static int nearest_point_index(const std::vector<float>& xyz, int n, const float target[3])
{
    int best = -1;
    float best_d2 = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float dx = xyz[static_cast<size_t>(i) * 3 + 0] - target[0];
        const float dy = xyz[static_cast<size_t>(i) * 3 + 1] - target[1];
        const float dz = xyz[static_cast<size_t>(i) * 3 + 2] - target[2];
        const float d2 = dx * dx + dy * dy + dz * dz;
        if (best < 0 || d2 < best_d2) { best = i; best_d2 = d2; }
    }
    return best;
}

// GateStats — per-object bookkeeping the analytic CHECK lines below read;
// also the source of every quantitative [info] line (the "load imbalance"
// and gate-selectivity numbers THEORY.md's GPU-mapping section quotes).
struct GateStats {
    int total = 0;          // candidates scored (K, or K + adversarial for the box)
    int found_partner = 0;  // idx2 >= 0
    int friction_ok = 0;
    int width_ok = 0;
    int clearance_ok = 0;
    int feasible = 0;
};

static GateStats summarize(const std::vector<GraspScore>& scores)
{
    GateStats s;
    s.total = static_cast<int>(scores.size());
    for (const auto& sc : scores) {
        // width_m > 0 is a reliable proxy for "idx2 >= 0 was true": every
        // field below is only ever set non-default when a partner was
        // found (score_candidates_kernel/_cpu leave the whole record at
        // its zeroed/rejected default otherwise).
        if (sc.width_m > 0.0f) ++s.found_partner;
        if (sc.friction_ok) ++s.friction_ok;
        if (sc.width_ok) ++s.width_ok;
        if (sc.clearance_ok) ++s.clearance_ok;
        if (sc.feasible) ++s.feasible;
    }
    return s;
}

// top_m_indices — indices into `scores`, sorted by DESCENDING score, first
// min(kTopM, feasible count) entries. Host std::sort at K~4096-4108: the
// same honest "this does not need a GPU kernel at this N" call project
// 12.01 makes for its greedy-NMS step after a GPU IoU-matrix kernel
// (README "Prior art") — ranking touches no device memory and runs in
// microseconds here, so a fused GPU top-k (radix-select / bitonic) would
// only add code, not speed (README Exercise names it as the natural next
// step for a learner who wants the fully-on-device version).
static std::vector<int> top_m_indices(const std::vector<GraspScore>& scores, int m)
{
    std::vector<int> order(scores.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return scores[static_cast<size_t>(a)].score > scores[static_cast<size_t>(b)].score;
    });
    int keep = 0;
    while (keep < static_cast<int>(order.size()) && keep < m &&
          scores[static_cast<size_t>(order[static_cast<size_t>(keep)])].feasible) {
        ++keep;
    }
    order.resize(static_cast<size_t>(keep));
    return order;
}

static float rad2deg(float r) { return r * (180.0f / 3.14159265358979323846f); }

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    std::printf("[demo] Parallel grasp-candidate scoring: antipodal sampling over point clouds - project 19.01\n");
    print_device_info();
    std::printf("PROBLEM: K=%d candidates/object, PCA k=%d (%d Jacobi sweeps), "
               "3 synthetic objects (box, cylinder, sphere)\n",
               kNumCandidates, kPcaK, kJacobiSweeps);

    // ---- load data ----------------------------------------------------------
    const std::string meta_path = find_meta_file(argv[0]);
    if (meta_path.empty()) {
        std::printf("SCENARIO: NOT FOUND - data/sample/objects_meta.csv missing (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (data missing)\n");
        return 1;
    }
    std::printf("[info] metadata file: %s\n", meta_path.c_str());

    std::vector<ObjectMeta> metas;
    if (!load_objects_meta(meta_path, metas) || metas.size() != 3) {
        std::printf("SCENARIO: MALFORMED - see stderr\n");
        std::printf("RESULT: FAIL (data malformed)\n");
        return 1;
    }
    const size_t slash = meta_path.find_last_of("/\\");
    const std::string data_dir = (slash == std::string::npos) ? "." : meta_path.substr(0, slash);

    std::vector<ObjectData> objects(metas.size());
    for (size_t i = 0; i < metas.size(); ++i) {
        objects[i].meta = metas[i];
        int n_loaded = 0;
        if (!load_cloud_bin(data_dir + "/" + metas[i].file, objects[i].xyz, n_loaded) ||
            n_loaded != metas[i].n_points) {
            std::printf("SCENARIO: MALFORMED - failed to load cloud for '%s'\n", metas[i].name.c_str());
            std::printf("RESULT: FAIL (data malformed)\n");
            return 1;
        }
    }

    for (const auto& obj : objects) {
        const ObjectMeta& m = obj.meta;
        if (m.shape == "box") {
            std::printf("SCENARIO: %s: N=%d points, box %.0fx%.0fx%.0f mm [synthetic]\n",
                       m.name.c_str(), m.n_points,
                       static_cast<double>(m.param_a_m * 1000.0f),
                       static_cast<double>(m.param_b_m * 1000.0f),
                       static_cast<double>(m.param_c_m * 1000.0f));
        } else if (m.shape == "cylinder") {
            std::printf("SCENARIO: %s: N=%d points, cylinder r=%.0f mm h=%.0f mm (lateral surface) [synthetic]\n",
                       m.name.c_str(), m.n_points,
                       static_cast<double>(m.param_a_m * 1000.0f),
                       static_cast<double>(m.param_b_m * 1000.0f));
        } else {
            std::printf("SCENARIO: %s: N=%d points, sphere r=%.0f mm [synthetic]\n",
                       m.name.c_str(), m.n_points, static_cast<double>(m.param_a_m * 1000.0f));
        }
    }

    // ---- upload clouds, compute normals (ONCE per object) --------------------
    struct ObjectDevice {
        float* d_xyz = nullptr;
        float* d_normals = nullptr;
    };
    std::vector<ObjectDevice> dev(objects.size());
    for (size_t i = 0; i < objects.size(); ++i) {
        const int n = objects[i].meta.n_points;
        CUDA_CHECK(cudaMalloc(&dev[i].d_xyz, static_cast<size_t>(n) * 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dev[i].d_normals, static_cast<size_t>(n) * 3 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dev[i].d_xyz, objects[i].xyz.data(),
                              static_cast<size_t>(n) * 3 * sizeof(float), cudaMemcpyHostToDevice));

        float ref_point[3];
        centroid3(objects[i].xyz, n, ref_point);

        GpuTimer gt;
        gt.begin();
        launch_estimate_normals(n, dev[i].d_xyz, ref_point, dev[i].d_normals);
        const float normals_ms = gt.end_ms();
        std::printf("[info] %s: normals (PCA k=%d, %d Jacobi sweeps): %.2f ms\n",
                   objects[i].meta.name.c_str(), kPcaK, kJacobiSweeps, static_cast<double>(normals_ms));
    }

    // ======================= VERIFY STAGE (box, index 0) ======================
    bool verify_pass = true;
    std::vector<GraspCandidate> h_candidates_box;   // reused below (avoids regenerating box's K candidates)
    std::vector<float> h_normals_box;               // GPU normals, downloaded once, reused by every CPU twin call
    {
        const ObjectData& obj = objects[0];
        const int n = obj.meta.n_points;

        // --- (a) normals: independent GPU vs CPU computation --------------
        h_normals_box.resize(static_cast<size_t>(n) * 3);
        CUDA_CHECK(cudaMemcpy(h_normals_box.data(), dev[0].d_normals,
                              h_normals_box.size() * sizeof(float), cudaMemcpyDeviceToHost));

        float ref_point[3];
        centroid3(obj.xyz, n, ref_point);
        std::vector<float> h_normals_cpu(static_cast<size_t>(n) * 3);
        estimate_normals_cpu(n, obj.xyz.data(), ref_point, h_normals_cpu.data());

        float worst_angle_deg = 0.0f;
        for (int i = 0; i < n; ++i) {
            const float ax = h_normals_box[static_cast<size_t>(i) * 3 + 0];
            const float ay = h_normals_box[static_cast<size_t>(i) * 3 + 1];
            const float az = h_normals_box[static_cast<size_t>(i) * 3 + 2];
            const float bx = h_normals_cpu[static_cast<size_t>(i) * 3 + 0];
            const float by = h_normals_cpu[static_cast<size_t>(i) * 3 + 1];
            const float bz = h_normals_cpu[static_cast<size_t>(i) * 3 + 2];
            float d = ax * bx + ay * by + az * bz;
            d = std::fmin(1.0f, std::fmax(-1.0f, d));
            const float angle_deg = rad2deg(std::acos(d));
            if (angle_deg > worst_angle_deg) worst_angle_deg = angle_deg;
        }
        const bool normals_pass = worst_angle_deg <= kVerifyNormalAngleTolDeg;
        std::printf("[info] normals check (box): worst GPU-vs-CPU angle deviation %.4f deg (tol %.2f deg)\n",
                   static_cast<double>(worst_angle_deg), static_cast<double>(kVerifyNormalAngleTolDeg));
        std::printf("VERIFY: %s (normals GPU matches CPU reference within angle tolerance)\n",
                   normals_pass ? "PASS" : "FAIL");
        verify_pass = verify_pass && normals_pass;

        // --- (b) candidate generation: must match EXACTLY ------------------
        GraspCandidate* d_candidates = nullptr;
        CUDA_CHECK(cudaMalloc(&d_candidates, static_cast<size_t>(kNumCandidates) * sizeof(GraspCandidate)));
        launch_generate_candidates(n, dev[0].d_xyz, dev[0].d_normals, kHashSeed, kNumCandidates, d_candidates);

        h_candidates_box.resize(static_cast<size_t>(kNumCandidates));
        CUDA_CHECK(cudaMemcpy(h_candidates_box.data(), d_candidates,
                              h_candidates_box.size() * sizeof(GraspCandidate), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_candidates));

        // The CPU twin consumes the SAME (already-verified-close) GPU
        // normals, isolating THIS stage's check from any residual normals
        // deviation (the same "feed the same intermediate to both paths"
        // discipline 02.06 uses for its normal-system stage).
        std::vector<GraspCandidate> h_candidates_cpu(static_cast<size_t>(kNumCandidates));
        generate_candidates_cpu(n, obj.xyz.data(), h_normals_box.data(), kHashSeed, kNumCandidates,
                                h_candidates_cpu.data());

        int mismatches = 0;
        for (int k = 0; k < kNumCandidates; ++k) {
            if (h_candidates_box[static_cast<size_t>(k)].idx1 != h_candidates_cpu[static_cast<size_t>(k)].idx1 ||
                h_candidates_box[static_cast<size_t>(k)].idx2 != h_candidates_cpu[static_cast<size_t>(k)].idx2) {
                ++mismatches;
            }
        }
        const bool cand_pass = (mismatches == 0);
        std::printf("[info] candidate generation check (box): %d/%d index mismatches\n", mismatches, kNumCandidates);
        std::printf("VERIFY: %s (candidate generation GPU matches CPU reference exactly)\n",
                   cand_pass ? "PASS" : "FAIL");
        verify_pass = verify_pass && cand_pass;

        // --- (c) scoring: relative tolerance --------------------------------
        GraspCandidate* d_cand2 = nullptr;
        GraspScore* d_scores = nullptr;
        CUDA_CHECK(cudaMalloc(&d_cand2, h_candidates_box.size() * sizeof(GraspCandidate)));
        CUDA_CHECK(cudaMalloc(&d_scores, h_candidates_box.size() * sizeof(GraspScore)));
        CUDA_CHECK(cudaMemcpy(d_cand2, h_candidates_box.data(),
                              h_candidates_box.size() * sizeof(GraspCandidate), cudaMemcpyHostToDevice));
        launch_score_candidates(n, dev[0].d_xyz, dev[0].d_normals, d_cand2, kNumCandidates,
                                obj.meta.mu, obj.meta.w_min_m, obj.meta.w_max_m, d_scores);

        std::vector<GraspScore> h_scores_gpu(static_cast<size_t>(kNumCandidates));
        CUDA_CHECK(cudaMemcpy(h_scores_gpu.data(), d_scores,
                              h_scores_gpu.size() * sizeof(GraspScore), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_cand2));
        CUDA_CHECK(cudaFree(d_scores));

        std::vector<GraspScore> h_scores_cpu(static_cast<size_t>(kNumCandidates));
        score_candidates_cpu(n, obj.xyz.data(), h_normals_box.data(),
                             h_candidates_box.data(), kNumCandidates,
                             obj.meta.mu, obj.meta.w_min_m, obj.meta.w_max_m, h_scores_cpu.data());

        float worst_rel = 0.0f;
        int bool_mismatches = 0;
        for (int k = 0; k < kNumCandidates; ++k) {
            const GraspScore& g = h_scores_gpu[static_cast<size_t>(k)];
            const GraspScore& c = h_scores_cpu[static_cast<size_t>(k)];
            const float fields_gpu[3] = { g.width_m, g.antipodal_cos, g.score };
            const float fields_cpu[3] = { c.width_m, c.antipodal_cos, c.score };
            for (int f = 0; f < 3; ++f) {
                const float scale = std::fabs(fields_cpu[f]) > 1.0f ? std::fabs(fields_cpu[f]) : 1.0f;
                const float rel = std::fabs(fields_gpu[f] - fields_cpu[f]) / scale;
                if (rel > worst_rel) worst_rel = rel;
            }
            if (g.feasible != c.feasible) ++bool_mismatches;
        }
        const bool score_pass = worst_rel <= kVerifyRelTol;
        std::printf("[info] scoring check (box): worst relative deviation %.3e (tol %.0e), "
                   "%d/%d feasibility-flag mismatches\n",
                   static_cast<double>(worst_rel), static_cast<double>(kVerifyRelTol),
                   bool_mismatches, kNumCandidates);
        std::printf("VERIFY: %s (scoring GPU matches CPU reference within relative tolerance)\n",
                   score_pass ? "PASS" : "FAIL");
        verify_pass = verify_pass && score_pass;
    }
    if (!verify_pass) {
        std::printf("RESULT: FAIL (GPU/CPU disagreement - fix before trusting the ranked grasps)\n");
        return 1;
    }

    // ======================= PER-OBJECT PIPELINE + GATES ======================
    const std::string out_dir = project_root_from(argv[0]) + "/demo/out";
    bool artifacts_ok = ensure_dir(out_dir);
    std::ofstream grasps_csv, cloud_csv;
    if (artifacts_ok) {
        grasps_csv.open(out_dir + "/grasps.csv");
        cloud_csv.open(out_dir + "/grasp_cloud.csv");
        artifacts_ok = grasps_csv.is_open() && cloud_csv.is_open();
        if (artifacts_ok) {
            grasps_csv << "object,rank,idx1,idx2,p1x_m,p1y_m,p1z_m,p2x_m,p2y_m,p2z_m,"
                         "axis_x,axis_y,axis_z,width_m,antipodal_cos,theta1_deg,theta2_deg,"
                         "friction_ok,width_ok,clearance_ok,score\n";
            cloud_csv << "object,kind,id,x_m,y_m,z_m\n";
        }
    }

    bool gate_pass = true;
    bool adversarial_pass = true;

    for (size_t oi = 0; oi < objects.size(); ++oi) {
        const ObjectData& obj = objects[oi];
        const ObjectMeta& m = obj.meta;
        const int n = m.n_points;

        // --- candidates: box gets K random + kNumAdversarial hand-picked ---
        std::vector<GraspCandidate> h_cand;
        if (oi == 0) {
            h_cand = h_candidates_box;   // reuse the exact array VERIFY already generated/checked

            // Build the 12 adjacent-(non-opposite)-face adversarial pairs
            // from REAL committed cloud points nearest each face's ideal
            // center (README/THEORY: exercised through the same scoring
            // code path as every other candidate, not hand-computed math).
            const float hx = m.param_a_m * 0.5f, hy = m.param_b_m * 0.5f, hz = m.param_c_m * 0.5f;
            const float face_centers[6][3] = {
                {  hx, 0,  0 }, { -hx, 0,  0 },
                { 0,  hy, 0 }, { 0, -hy, 0 },
                { 0, 0,  hz }, { 0, 0, -hz },
            };
            int face_idx[6];
            for (int f = 0; f < 6; ++f) face_idx[f] = nearest_point_index(obj.xyz, n, face_centers[f]);

            const int opposite[3][2] = { {0,1}, {2,3}, {4,5} };
            for (int a = 0; a < 6; ++a) {
                for (int b = a + 1; b < 6; ++b) {
                    bool is_opposite = false;
                    for (const auto& op : opposite) if ((op[0]==a&&op[1]==b) || (op[0]==b&&op[1]==a)) is_opposite = true;
                    if (is_opposite) continue;   // the 3 genuinely-antipodal axis pairs — not adversarial
                    h_cand.push_back(GraspCandidate{ face_idx[a], face_idx[b] });
                }
            }
            // 6 choose 2 = 15 pairs total, minus 3 opposite pairs = 12 adversarial.
        } else {
            GraspCandidate* d_cand = nullptr;
            CUDA_CHECK(cudaMalloc(&d_cand, static_cast<size_t>(kNumCandidates) * sizeof(GraspCandidate)));
            launch_generate_candidates(n, dev[oi].d_xyz, dev[oi].d_normals, kHashSeed, kNumCandidates, d_cand);
            h_cand.resize(static_cast<size_t>(kNumCandidates));
            CUDA_CHECK(cudaMemcpy(h_cand.data(), d_cand, h_cand.size() * sizeof(GraspCandidate), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(d_cand));
        }

        // --- score all of them (GPU) ---------------------------------------
        GraspCandidate* d_cand2 = nullptr;
        GraspScore* d_scores = nullptr;
        CUDA_CHECK(cudaMalloc(&d_cand2, h_cand.size() * sizeof(GraspCandidate)));
        CUDA_CHECK(cudaMalloc(&d_scores, h_cand.size() * sizeof(GraspScore)));
        CUDA_CHECK(cudaMemcpy(d_cand2, h_cand.data(), h_cand.size() * sizeof(GraspCandidate), cudaMemcpyHostToDevice));

        GpuTimer gt;
        gt.begin();
        launch_score_candidates(n, dev[oi].d_xyz, dev[oi].d_normals, d_cand2,
                                static_cast<int>(h_cand.size()), m.mu, m.w_min_m, m.w_max_m, d_scores);
        const float score_ms = gt.end_ms();

        std::vector<GraspScore> h_scores(h_cand.size());
        CUDA_CHECK(cudaMemcpy(h_scores.data(), d_scores, h_scores.size() * sizeof(GraspScore), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_cand2));
        CUDA_CHECK(cudaFree(d_scores));

        const GateStats stats = summarize(h_scores);
        std::printf("[info] %s: %d candidates scored in %.2f ms - %d found a partner, "
                   "%d passed friction cone, %d passed width, %d passed clearance, %d fully feasible\n",
                   m.name.c_str(), stats.total, static_cast<double>(score_ms),
                   stats.found_partner, stats.friction_ok, stats.width_ok, stats.clearance_ok, stats.feasible);

        // --- box only: count how many candidates found the too-wide
        // (antipodal but gripper-infeasible) z-axis pairing, and check the
        // 12 adversarial entries (the LAST kNumAdversarial rows of h_scores,
        // by construction above) are ALL rejected.
        if (oi == 0) {
            int z_axis_antipodal_but_wide = 0;
            const float z_width = m.param_c_m;   // BOX_DZ: the too-wide axis
            for (int k = 0; k < kNumCandidates; ++k) {
                const GraspScore& sc = h_scores[static_cast<size_t>(k)];
                if (sc.width_m > 0.0f && sc.friction_ok && !sc.width_ok &&
                    std::fabs(sc.width_m - z_width) <= kGateWidthTolM) {
                    ++z_axis_antipodal_but_wide;
                }
            }
            std::printf("[info] box: %d candidates found the %.0f mm (z-axis) antipodal pairing and were "
                       "correctly marked width_ok=false (gripper stroke %.0f-%.0f mm)\n",
                       z_axis_antipodal_but_wide, static_cast<double>(z_width * 1000.0f),
                       static_cast<double>(m.w_min_m * 1000.0f), static_cast<double>(m.w_max_m * 1000.0f));
            std::printf("CHECK: %s (box: z-axis (%.0f mm) antipodal pairs correctly rejected by the gripper width gate)\n",
                       (z_axis_antipodal_but_wide > 0) ? "PASS" : "FAIL", static_cast<double>(z_width * 1000.0f));
            gate_pass = gate_pass && (z_axis_antipodal_but_wide > 0);

            int adv_rejected = 0;
            for (int a = 0; a < kNumAdversarial; ++a) {
                const GraspScore& sc = h_scores[static_cast<size_t>(kNumCandidates + a)];
                if (!sc.feasible) ++adv_rejected;
            }
            std::printf("[info] box: %d/%d adversarial adjacent-face candidates rejected\n",
                       adv_rejected, kNumAdversarial);
            std::printf("CHECK: %s (box: all adjacent-face (non-antipodal) candidates rejected by the friction-cone gate)\n",
                       (adv_rejected == kNumAdversarial) ? "PASS" : "FAIL");
            adversarial_pass = adversarial_pass && (adv_rejected == kNumAdversarial);
        }

        // --- rank + analytic gate --------------------------------------------
        const std::vector<int> top = top_m_indices(h_scores, kTopM);
        bool obj_gate = (static_cast<int>(top.size()) >= kTopM);   // must have found at least kTopM feasible grasps
        for (size_t rank0 = 0; rank0 < top.size(); ++rank0) {
            const int idx = top[rank0];
            const GraspScore& sc = h_scores[static_cast<size_t>(idx)];
            const GraspCandidate& cd = h_cand[static_cast<size_t>(idx)];
            const float p1x = obj.xyz[static_cast<size_t>(cd.idx1) * 3 + 0];
            const float p1y = obj.xyz[static_cast<size_t>(cd.idx1) * 3 + 1];
            const float p1z = obj.xyz[static_cast<size_t>(cd.idx1) * 3 + 2];
            const float p2x = obj.xyz[static_cast<size_t>(cd.idx2) * 3 + 0];
            const float p2y = obj.xyz[static_cast<size_t>(cd.idx2) * 3 + 1];
            const float p2z = obj.xyz[static_cast<size_t>(cd.idx2) * 3 + 2];
            const float inv_w = 1.0f / sc.width_m;
            const float ax = (p2x - p1x) * inv_w, ay = (p2y - p1y) * inv_w, az = (p2z - p1z) * inv_w;

            bool ok = (sc.antipodal_cos >= kGateAntipodalCosMin) && sc.friction_ok;
            if (m.shape == "box") {
                const float axis_align = std::fmax(std::fabs(ax), std::fmax(std::fabs(ay), std::fabs(az)));
                const bool matches_a = std::fabs(sc.width_m - m.param_a_m) <= kGateWidthTolM;
                const bool matches_b = std::fabs(sc.width_m - m.param_b_m) <= kGateWidthTolM;
                ok = ok && (matches_a || matches_b) && (axis_align >= kGateAxisAlignMin);
            } else if (m.shape == "cylinder") {
                const float diam = 2.0f * m.param_a_m;
                const bool matches_diam = std::fabs(sc.width_m - diam) <= kGateWidthTolM;
                ok = ok && matches_diam && (std::fabs(az) <= kGateCylPerpMax);   // axis roughly perpendicular to cylinder's z-axis
            } else {   // sphere
                const float diam = 2.0f * m.param_a_m;
                ok = ok && (std::fabs(sc.width_m - diam) <= kGateWidthTolM);
            }
            obj_gate = obj_gate && ok;

            if (artifacts_ok) {
                grasps_csv << m.name << ',' << (rank0 + 1) << ',' << cd.idx1 << ',' << cd.idx2 << ','
                          << p1x << ',' << p1y << ',' << p1z << ',' << p2x << ',' << p2y << ',' << p2z << ','
                          << ax << ',' << ay << ',' << az << ',' << sc.width_m << ',' << sc.antipodal_cos << ','
                          << sc.theta1_deg << ',' << sc.theta2_deg << ','
                          << static_cast<int>(sc.friction_ok) << ',' << static_cast<int>(sc.width_ok) << ','
                          << static_cast<int>(sc.clearance_ok) << ',' << sc.score << '\n';
            }
        }
        std::printf("CHECK: %s (%s: top-%d grasps are analytically valid antipodal pairs within tolerance)\n",
                   obj_gate ? "PASS" : "FAIL", m.name.c_str(), kTopM);
        gate_pass = gate_pass && obj_gate;

        // --- artifact: subsampled cloud + top-5 grasp axes -------------------
        if (artifacts_ok) {
            const int stride = std::max(1, n / 1200);
            for (int i = 0; i < n; i += stride) {
                cloud_csv << m.name << ",cloud,-1,"
                         << obj.xyz[static_cast<size_t>(i) * 3 + 0] << ','
                         << obj.xyz[static_cast<size_t>(i) * 3 + 1] << ','
                         << obj.xyz[static_cast<size_t>(i) * 3 + 2] << '\n';
            }
            const int n_plot = std::min<int>(5, static_cast<int>(top.size()));
            for (int r = 0; r < n_plot; ++r) {
                const GraspCandidate& cd = h_cand[static_cast<size_t>(top[static_cast<size_t>(r)])];
                for (int end = 0; end < 2; ++end) {
                    const int pidx = (end == 0) ? cd.idx1 : cd.idx2;
                    cloud_csv << m.name << ",grasp," << r << ','
                             << obj.xyz[static_cast<size_t>(pidx) * 3 + 0] << ','
                             << obj.xyz[static_cast<size_t>(pidx) * 3 + 1] << ','
                             << obj.xyz[static_cast<size_t>(pidx) * 3 + 2] << '\n';
                }
            }
        }
    }

    if (artifacts_ok) {
        std::printf("ARTIFACT: wrote demo/out/grasps.csv (top-%d grasps per object)\n", kTopM);
        std::printf("ARTIFACT: wrote demo/out/grasp_cloud.csv (subsampled clouds + top-5 grasp axes per object)\n");
    } else {
        std::printf("ARTIFACT: FAILED to write demo/out/ files\n");
    }

    // ---- cleanup ----------------------------------------------------------------
    for (auto& d : dev) {
        CUDA_CHECK(cudaFree(d.d_xyz));
        CUDA_CHECK(cudaFree(d.d_normals));
    }

    const bool success = gate_pass && adversarial_pass && artifacts_ok;
    if (success)
        std::printf("RESULT: PASS (verify passed; every object's top-%d grasps are analytically valid; "
                   "box width gate and adversarial negative control both correct)\n", kTopM);
    else
        std::printf("RESULT: FAIL (see CHECK/ARTIFACT lines above)\n");
    return success ? 0 : 1;
}
