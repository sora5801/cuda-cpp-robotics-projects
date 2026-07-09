// ===========================================================================
// main.cu — entry point for project 09.01
//           Batched forward kinematics (10⁵ configurations — the foundation
//           for everything above)
//
// What this program does, start to finish
// ---------------------------------------
//   1. Print the demo banner + GPU info.
//   2. Load the committed synthetic sample (data/sample/fk_sample.csv): a
//      6-joint robot model plus 64 joint configurations. Validate the model
//      ONCE (normalized quaternions, unit axes) and upload it to GPU
//      constant memory via set_robot_model().
//   3. SAMPLE STAGE: run FK on the sample configurations on BOTH the GPU
//      kernel and the CPU oracle; compare within documented tolerances.
//   4. BATCH STAGE: generate 200,000 configurations in memory (fixed
//      xorshift32 seed, angles in (−π, π]), verify GPU == CPU on all of
//      them, and time both paths for the teaching speed-up line.
//   5. Exit 0 only if every comparison passed.
//
// The output contract (shared with demo/run_demo.ps1 + expected_output.txt)
// --------------------------------------------------------------------------
// STABLE lines (checked verbatim): "[demo]", "PROBLEM:", "MODEL:",
// "SAMPLE:", "SAMPLE RESULT:", "BATCH:", "RESULT:". UNSTABLE lines (never
// checked): "[info]", "[time]". Change a stable line ⇒ update
// demo/expected_output.txt in the same commit.
//
// Tolerances (justified in THEORY.md §numerics; quoted for the reader):
//   position   : |Δp| ≤ 1e-4 m per component. The arm spans ~1 m; six
//                chained FP32 rotations plus trig-implementation differences
//                (sincosf vs std::sin) produce ~1e-6 m of legitimate
//                disagreement; 1e-4 gives ~100× headroom while still
//                catching any real chain-composition bug (those show up at
//                centimeter-to-meter scale).
//   quaternion : |Δq| ≤ 1e-4 per component AFTER hemisphere alignment.
//                q and −q are the same rotation (double cover) and Shepperd
//                branch selection may differ between paths near trace
//                boundaries, so the comparator first flips one side if
//                their dot product is negative — comparing rotations, not
//                sign conventions. Norm invariant ‖q‖ = 1 ± 1e-4 is checked
//                on both paths as a self-consistency gate.
//
// Read this first, then kernels.cuh → reference_cpu.cpp → kernels.cu.
// ===========================================================================

#include "kernels.cuh"            // layouts + GPU launchers + CPU oracle
#include "util/cuda_check.cuh"    // CUDA_CHECK / device banner
#include "util/timer.cuh"         // GpuTimer / CpuTimer

#include <cmath>                  // std::fabs, std::sqrt
#include <cstdint>                // uint32_t for the RNG
#include <cstdio>
#include <cstring>                // strcmp for the argv parser
#include <fstream>                // sample CSV loading
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Deterministic RNG — xorshift32, identical to project 33.01's and for the
// same reason: std::uniform_real_distribution is not bit-portable across
// standard libraries, and the BATCH line promises reproducible inputs on
// every platform. See 33.01's main.cu for the full rationale; the recap:
// top 24 bits → exact float in [0,1) → affine map, all exactly rounded.
// ---------------------------------------------------------------------------
static inline uint32_t xorshift32(uint32_t& state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

// Uniform angle in (−π, π] — the repo's canonical wrapped-angle interval
// (CLAUDE.md §12). Built from the exact [0,1) draw; the sign flip maps 0 to
// +π-side, keeping the interval half-open on the right as specified.
static inline float uniform_angle(uint32_t& state)
{
    const float u01 = (xorshift32(state) >> 8) * (1.0f / 16777216.0f);  // [0,1), 24-bit exact
    return 3.14159265358979323846f * (1.0f - 2.0f * u01);               // (−π, π]
}

// ---------------------------------------------------------------------------
// Sample-file handling.
//
// Format (written by scripts/make_synthetic.py; documented in data/README.md):
//   '#'-prefixed lines: comments (provenance, the SYNTHETIC label).
//   J,<idx>,tx,ty,tz,qw,qx,qy,qz,ax,ay,az   — one model row per joint, in
//                                             chain order (kModelStride=10)
//   Q,<idx>,q0,...,q{nj-1}                  — one configuration per row (rad)
// Strict loader: unknown labels, wrong value counts, or a Q row whose length
// disagrees with the number of J rows abort the load — corrupt samples must
// never quietly pass (same stance as 33.01).
// ---------------------------------------------------------------------------
struct SampleData {
    std::vector<float> model;   // nj*10 floats, chain order
    std::vector<float> q;       // nconf*nj joint angles (rad)
    int nj = 0;                 // number of joints (J rows seen)
    int nconf = 0;              // number of configurations (Q rows seen)
    bool loaded = false;
};

static SampleData load_sample(const std::string& path)
{
    SampleData s;
    std::ifstream in(path);
    if (!in.is_open()) return s;

    std::vector<std::vector<float>> q_rows;   // staged until nj is known
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string label, cell;
        if (!std::getline(ss, label, ',')) continue;
        std::getline(ss, cell, ',');          // the informational index column
        std::vector<float> vals;
        while (std::getline(ss, cell, ','))
            vals.push_back(std::strtof(cell.c_str(), nullptr));

        if (label == "J") {
            if (vals.size() != kModelStride) {
                std::fprintf(stderr, "sample: J row has %zu values, expected %d\n",
                             vals.size(), kModelStride);
                return SampleData{};
            }
            s.model.insert(s.model.end(), vals.begin(), vals.end());
            s.nj += 1;
        } else if (label == "Q") {
            q_rows.push_back(std::move(vals));
        } else {
            std::fprintf(stderr, "sample: unknown row label '%s'\n", label.c_str());
            return SampleData{};
        }
    }

    if (s.nj < 1 || s.nj > kMaxJoints) {
        std::fprintf(stderr, "sample: %d J rows (expected 1..%d)\n", s.nj, kMaxJoints);
        return SampleData{};
    }
    for (const auto& row : q_rows) {
        if (static_cast<int>(row.size()) != s.nj) {
            std::fprintf(stderr, "sample: Q row has %zu angles, expected %d\n",
                         row.size(), s.nj);
            return SampleData{};
        }
        s.q.insert(s.q.end(), row.begin(), row.end());
    }
    s.nconf = static_cast<int>(q_rows.size());
    s.loaded = true;
    return s;
}

// Validate-and-normalize the model IN PLACE, once, at load time — the per-
// thread inner loops then trust ‖quat‖=1 and ‖axis‖=1 (contract in
// kernels.cuh). Tolerant of ~1e-3 file-format rounding, loud beyond it:
// a wildly non-unit axis means a malformed model, not rounding.
static bool validate_model(std::vector<float>& model, int nj)
{
    for (int j = 0; j < nj; ++j) {
        float* m = &model[static_cast<size_t>(j) * kModelStride];
        // fixed-rotation quaternion (w,x,y,z) at offsets 3..6
        float qn = std::sqrt(m[3]*m[3] + m[4]*m[4] + m[5]*m[5] + m[6]*m[6]);
        // joint axis at offsets 7..9
        float an = std::sqrt(m[7]*m[7] + m[8]*m[8] + m[9]*m[9]);
        if (std::fabs(qn - 1.0f) > 1e-3f || std::fabs(an - 1.0f) > 1e-3f) {
            std::fprintf(stderr,
                         "model: joint %d not normalized (|quat|=%g, |axis|=%g)\n",
                         j, static_cast<double>(qn), static_cast<double>(an));
            return false;
        }
        for (int i = 3; i <= 6; ++i) m[i] /= qn;   // exact-ify to FP32 precision
        for (int i = 7; i <= 9; ++i) m[i] /= an;
    }
    return true;
}

// Locate the sample (same candidate order as 33.01: --data, exe-relative
// three-up, then cwd-relative fallbacks).
static std::string find_sample(const std::string& cli_path, const char* argv0)
{
    std::vector<std::string> candidates;
    if (!cli_path.empty()) candidates.push_back(cli_path);
    std::string exe(argv0 ? argv0 : "");
    size_t cut = exe.find_last_of("/\\");
    if (cut != std::string::npos)
        candidates.push_back(exe.substr(0, cut) + "/../../../data/sample/fk_sample.csv");
    candidates.push_back("data/sample/fk_sample.csv");
    candidates.push_back("../data/sample/fk_sample.csv");
    for (const auto& c : candidates)
        if (std::ifstream(c).is_open()) return c;
    return "";
}

// ---------------------------------------------------------------------------
// GPU round-trip: upload q, run the kernel, download poses. Plain and
// checked, like 33.01's — transfer engineering is domain 32's business.
// ---------------------------------------------------------------------------
static void gpu_fk(int nj, int count, const std::vector<float>& q,
                   std::vector<float>& pose_gpu, float* kernel_ms /*nullable*/)
{
    const size_t q_bytes = static_cast<size_t>(count) * nj * sizeof(float);
    const size_t p_bytes = static_cast<size_t>(count) * kPoseStride * sizeof(float);
    pose_gpu.assign(static_cast<size_t>(count) * kPoseStride, 0.0f);

    float *d_q = nullptr, *d_pose = nullptr;      // d_ prefix: device pointers
    CUDA_CHECK(cudaMalloc(&d_q, q_bytes));
    CUDA_CHECK(cudaMalloc(&d_pose, p_bytes));
    CUDA_CHECK(cudaMemcpy(d_q, q.data(), q_bytes, cudaMemcpyHostToDevice));

    if (kernel_ms) {
        GpuTimer t;
        t.begin();
        launch_batched_fk(count, d_q, d_pose);
        *kernel_ms = t.end_ms();                  // synchronizes (see util/timer.cuh)
    } else {
        launch_batched_fk(count, d_q, d_pose);
    }

    CUDA_CHECK(cudaMemcpy(pose_gpu.data(), d_pose, p_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_pose));
}

// ---------------------------------------------------------------------------
// Comparator — poses within tolerance, quaternions modulo the double cover.
// Returns worst deviations (for the [info] line); clears `pass` on failure.
// ---------------------------------------------------------------------------
struct PoseDeviation { float pos = 0.0f, quat = 0.0f, norm = 0.0f; };

static PoseDeviation compare_poses(const std::vector<float>& gpu,
                                   const std::vector<float>& cpu,
                                   int count, bool& pass,
                                   float pos_tol, float quat_tol)
{
    PoseDeviation worst;
    for (int k = 0; k < count; ++k) {
        const float* g = &gpu[static_cast<size_t>(k) * kPoseStride];
        const float* c = &cpu[static_cast<size_t>(k) * kPoseStride];

        // Position: plain per-component absolute difference (meters).
        for (int i = 0; i < 3; ++i) {
            float d = std::fabs(g[i] - c[i]);
            if (d > worst.pos) worst.pos = d;
            if (d > pos_tol) pass = false;
        }

        // Quaternion norm invariant on both paths (a self-consistency gate:
        // a wildly non-unit quaternion means conversion bugs even if the two
        // paths happen to agree with each other).
        auto norm4 = [](const float* v) {
            return std::sqrt(v[3]*v[3] + v[4]*v[4] + v[5]*v[5] + v[6]*v[6]);
        };
        float ng = std::fabs(norm4(g) - 1.0f), nc = std::fabs(norm4(c) - 1.0f);
        float nworst = ng > nc ? ng : nc;
        if (nworst > worst.norm) worst.norm = nworst;
        if (nworst > quat_tol) pass = false;

        // Quaternion values, hemisphere-aligned first: q and −q are the same
        // rotation, and the two paths may land on opposite covers (Shepperd
        // branch boundaries + rounding). Flip via the dot-product sign, then
        // compare per component.
        float dot = g[3]*c[3] + g[4]*c[4] + g[5]*c[5] + g[6]*c[6];
        float sign = (dot < 0.0f) ? -1.0f : 1.0f;
        for (int i = 3; i < 7; ++i) {
            float d = std::fabs(g[i] - sign * c[i]);
            if (d > worst.quat) worst.quat = d;
            if (d > quat_tol) pass = false;
        }
    }
    return worst;
}

// ---------------------------------------------------------------------------
// main — the two stages described in the file header.
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    int batch_count = 200000;      // configurations in the batch stage (default defines the BATCH line)
    std::string data_path;         // optional --data override
    for (int i = 1; i < argc; ++i) {
        if      (!std::strcmp(argv[i], "--count") && i + 1 < argc) batch_count = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--data")  && i + 1 < argc) data_path   = argv[++i];
        else {
            std::fprintf(stderr,
                "usage: %s [--count N_configurations] [--data fk_sample.csv]\n"
                "note: a non-default count changes the BATCH line and the demo diff will\n"
                "      flag it — intended for experiments only.\n", argv[0]);
            return 2;
        }
    }

    std::printf("[demo] batched forward kinematics (project 09.01)\n");
    print_device_info();
    std::printf("PROBLEM: serial-chain FK T_base_ee(q) -> pose (p [m], quat (w,x,y,z)), FP32\n");

    const float kPosTol  = 1e-4f;   // meters — see file header
    const float kQuatTol = 1e-4f;   // per component, hemisphere-aligned
    bool all_pass = true;

    // ---- sample: load, validate, upload the robot --------------------------
    const std::string sample_path = find_sample(data_path, argv[0]);
    if (sample_path.empty()) {
        std::printf("MODEL: NOT FOUND — data/sample/fk_sample.csv missing (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (sample data missing)\n");
        return 1;   // the committed sample is part of the §4 reproducibility contract
    }
    std::printf("[info] sample file: %s\n", sample_path.c_str());

    SampleData sample = load_sample(sample_path);
    if (!sample.loaded || !validate_model(sample.model, sample.nj)) {
        std::printf("MODEL: MALFORMED — see stderr\n");
        std::printf("RESULT: FAIL (sample data malformed)\n");
        return 1;
    }
    std::printf("MODEL: %d revolute joints [synthetic 6-DoF arm]\n", sample.nj);
    std::printf("SAMPLE: %d configurations [synthetic, seed 42]\n", sample.nconf);

    // One upload, many launches — the URDF-at-startup pattern (kernels.cuh).
    set_robot_model(sample.nj, sample.model.data());

    // ---- SAMPLE STAGE -------------------------------------------------------
    {
        std::vector<float> pose_gpu;
        std::vector<float> pose_cpu(static_cast<size_t>(sample.nconf) * kPoseStride);
        gpu_fk(sample.nj, sample.nconf, sample.q, pose_gpu, nullptr);
        batched_fk_cpu(sample.nj, sample.model.data(), sample.nconf,
                       sample.q.data(), pose_cpu.data());

        bool sample_pass = true;
        PoseDeviation w = compare_poses(pose_gpu, pose_cpu, sample.nconf,
                                        sample_pass, kPosTol, kQuatTol);
        std::printf("[info] sample worst deviation: position %.3e m, quaternion %.3e, |norm-1| %.3e\n",
                    static_cast<double>(w.pos), static_cast<double>(w.quat),
                    static_cast<double>(w.norm));
        std::printf("SAMPLE RESULT: %s (GPU vs CPU within tol: position abs 1e-4 m, quaternion abs 1e-4)\n",
                    sample_pass ? "PASS" : "FAIL");
        all_pass = all_pass && sample_pass;
    }

    // ---- BATCH STAGE --------------------------------------------------------
    std::printf("BATCH: %d configurations, generated in-memory [seed 42]\n", batch_count);
    {
        // Deterministic batch: angles uniform in (−π, π], one stream, seed 42.
        std::vector<float> q(static_cast<size_t>(batch_count) * sample.nj);
        uint32_t s = 42u;
        for (float& a : q) a = uniform_angle(s);

        std::vector<float> pose_gpu;
        std::vector<float> pose_cpu(static_cast<size_t>(batch_count) * kPoseStride);

        float gpu_ms = 0.0f;
        gpu_fk(sample.nj, batch_count, q, pose_gpu, &gpu_ms);

        CpuTimer ct;
        ct.begin();
        batched_fk_cpu(sample.nj, sample.model.data(), batch_count,
                       q.data(), pose_cpu.data());
        double cpu_ms = ct.end_ms();

        bool batch_pass = true;
        compare_poses(pose_gpu, pose_cpu, batch_count, batch_pass, kPosTol, kQuatTol);

        // Single-shot, kernel-only vs one CPU core — a teaching artifact,
        // never a benchmark claim (CLAUDE.md §12).
        std::printf("[time] fk x %d configs: CPU %.1f ms | GPU kernel %.3f ms | speed-up %.0fx (teaching artifact; kernel only)\n",
                    batch_count, cpu_ms, static_cast<double>(gpu_ms),
                    cpu_ms / (static_cast<double>(gpu_ms) > 0.0 ? static_cast<double>(gpu_ms) : 1.0));
        all_pass = all_pass && batch_pass;
    }

    if (all_pass)
        std::printf("RESULT: PASS (all GPU poses match the CPU reference within documented tolerances)\n");
    else
        std::printf("RESULT: FAIL (at least one GPU pose diverged from the CPU reference — see lines above)\n");
    return all_pass ? 0 : 1;
}
