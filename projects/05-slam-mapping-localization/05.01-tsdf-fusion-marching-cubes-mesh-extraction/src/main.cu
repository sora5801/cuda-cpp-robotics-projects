// ===========================================================================
// main.cu — entry point for project 05.01
//           TSDF fusion (KinectFusion clone) + marching-cubes mesh
//           extraction
//
// What this program does, start to finish
// ---------------------------------------
//   1. Print the banner + GPU info; load the committed sample (pinhole
//      intrinsics + 24 camera poses on a circle) from data/sample/.
//   2. RENDER the input: 24 synthetic depth frames, ray-cast on the host
//      from the ANALYTIC scene (a sphere floating above a ground plane —
//      kernels.cuh). The scene's signed distance function has a closed
//      form, which is the whole point: the fused volume can be checked
//      against exact ground truth, not against a fixture file.
//   3. VERIFY STAGE (the §5 GPU-vs-CPU gate): fuse the first 4 frames
//      through the GPU kernel AND the CPU twin into separate volumes;
//      require voxel-wise agreement (abs tol 1e-5 on TSDF, exact on
//      weights — the two paths are engineered to be bit-identical,
//      kernels.cuh §determinism).
//   4. FULL FUSION: fuse all 24 frames on the GPU.
//   5. GROUND-TRUTH CHECK: compare the fused TSDF against the analytic
//      scene SDF near the surface, within documented bounds (the bounds
//      story — projective bias, pixel quantization — is THEORY.md's).
//   6. MARCHING CUBES: extract the iso-surface mesh on the GPU
//      (atomic-append), check the triangle count against a CPU recount
//      (must match EXACTLY) and against a stable expected range, and check
//      every emitted vertex against the analytic SDF.
//   7. ARTIFACTS: demo/out/mesh.obj (open it in any 3-D viewer!) and
//      demo/out/tsdf_slice.pgm (a vertical slice through the volume).
//      Exit 0 only if every check above held.
//
// NO RNG anywhere in this project: the poses are closed-form constants, the
// depth is exact ray casting, fusion is deterministic arithmetic. The one
// nondeterminism in the whole pipeline is the ORDER of marching-cubes
// triangles in the buffer (atomic append) — the set and count are
// deterministic, and no check depends on order.
//
// Output contract: stable lines "[demo]", "PROBLEM:", "SAMPLE:", "VERIFY:",
// "GROUND TRUTH:", "MESH:", "ARTIFACT:", "RESULT:" — "[info]"/"[time]"
// unchecked (machine-dependent). Change a stable line ⇒ update
// demo/expected_output.txt in the same commit.
//
// Read this first, then kernels.cuh → reference_cpu.cpp → kernels.cu
// (mc_tables.h when marching cubes gets interesting).
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
#include <direct.h>               // _mkdir (std::filesystem avoided in .cu — see 07.09)
#else
#include <sys/stat.h>
#endif

// ---------------------------------------------------------------------------
// The analytic scene SDF — the project's ground truth (host-only; the
// geometry constants live in kernels.cuh so every file agrees on them).
//
//   sdf(p) = min( |p - sphere_center| - r ,  p.z - plane_z )
//
// Positive outside, negative inside, zero exactly ON the surface — and
// EXACT within the truncation band because the sphere floats 0.25 m above
// the plane, more than 2*mu of clearance (kernels.cuh scene note). Units: m.
// ---------------------------------------------------------------------------
static float scene_sdf(float x, float y, float z)
{
    const float dx = x - kSphereCX, dy = y - kSphereCY, dz = z - kSphereCZ;
    const float d_sphere = std::sqrt(dx * dx + dy * dy + dz * dz) - kSphereR;
    const float d_plane  = z - kPlaneZ;
    return d_sphere < d_plane ? d_sphere : d_plane;
}

// ---------------------------------------------------------------------------
// Depth rendering — the project's synthetic sensor.
//
// For every pixel, cast the ray through its center and intersect it with
// the analytic scene in CLOSED FORM (quadratic for the sphere, one division
// for the plane) — no marching, no approximation, so the rendered depth is
// exact to FP rounding. Rendered ONCE on the host and fed to BOTH the GPU
// and CPU integration paths, so the §5 comparison sees identical inputs.
//
// Depth convention (kernels.cuh): z-depth in meters along the optical axis.
// With the ray parameterized as p_cam = s * (x_n, y_n, 1) — the UNNORMALIZED
// direction whose z-component is exactly 1 — the ray parameter s IS the
// z-depth, which is why no normalization appears below. Pixels whose rays
// escape to infinity (above the horizon) get depth 0 = "no return", exactly
// how a real sensor reports dropouts.
// ---------------------------------------------------------------------------
static void render_depth(const Intrinsics& K, const PoseRt& T_world_cam,
                         std::vector<float>& depth /* [h*w] OUT, meters */)
{
    depth.assign(static_cast<size_t>(K.width) * K.height, 0.0f);
    const float* R = T_world_cam.r;              // camera axes, rows of R^T = columns of R
    const float ox = T_world_cam.t[0];           // camera position in world (m)
    const float oy = T_world_cam.t[1];
    const float oz = T_world_cam.t[2];

    for (int v = 0; v < K.height; ++v) {
        for (int u = 0; u < K.width; ++u) {
            // Ray direction through this pixel: camera frame, z = 1.
            const float xn = (static_cast<float>(u) - K.cx) / K.fx;
            const float yn = (static_cast<float>(v) - K.cy) / K.fy;
            // Rotate into the world frame: d_world = R_world_cam * d_cam.
            const float dxw = R[0] * xn + R[1] * yn + R[2];
            const float dyw = R[3] * xn + R[4] * yn + R[5];
            const float dzw = R[6] * xn + R[7] * yn + R[8];

            float s_hit = -1.0f;                 // smallest positive hit parameter (= z-depth)

            // Sphere: |o + s*d - c|^2 = r^2 → a*s^2 + b*s + c0 = 0.
            {
                const float mx = ox - kSphereCX, my = oy - kSphereCY, mz = oz - kSphereCZ;
                const float a  = dxw * dxw + dyw * dyw + dzw * dzw;
                const float b  = 2.0f * (mx * dxw + my * dyw + mz * dzw);
                const float c0 = mx * mx + my * my + mz * mz - kSphereR * kSphereR;
                const float disc = b * b - 4.0f * a * c0;
                if (disc >= 0.0f) {
                    // Near root first; cameras sit outside the sphere, so the
                    // near root is the visible surface when it is positive.
                    const float s = (-b - std::sqrt(disc)) / (2.0f * a);
                    if (s > 0.0f) s_hit = s;
                }
            }

            // Ground plane z = kPlaneZ (solid below): hit only when looking
            // downward from above — rays parallel to or away from the plane
            // miss (that is the horizon line visible in the depth images).
            if (dzw < 0.0f && oz > kPlaneZ) {
                const float s = (kPlaneZ - oz) / dzw;
                if (s > 0.0f && (s_hit < 0.0f || s < s_hit)) s_hit = s;  // nearest surface wins (occlusion)
            }

            if (s_hit > 0.0f)
                depth[static_cast<size_t>(v) * K.width + u] = s_hit;   // z-depth (m)
        }
    }
}

// ---------------------------------------------------------------------------
// quat_to_rot — unit quaternion (w,x,y,z — repo order, CLAUDE.md §12) to a
// row-major rotation matrix. Normalizes defensively first: a quaternion
// that drifted 1e-9 from unit length in the CSV's 9 significant digits
// would otherwise shear the rotation (the classic quaternion-drift hazard).
// ---------------------------------------------------------------------------
static void quat_to_rot(float w, float x, float y, float z, float r[9])
{
    const float n = std::sqrt(w * w + x * x + y * y + z * z);
    w /= n; x /= n; y /= n; z /= n;
    r[0] = 1 - 2 * (y * y + z * z); r[1] = 2 * (x * y - w * z);     r[2] = 2 * (x * z + w * y);
    r[3] = 2 * (x * y + w * z);     r[4] = 1 - 2 * (x * x + z * z); r[5] = 2 * (y * z - w * x);
    r[6] = 2 * (x * z - w * y);     r[7] = 2 * (y * z + w * x);     r[8] = 1 - 2 * (x * x + y * y);
}

// ---------------------------------------------------------------------------
// invert_pose — T_cam_world from T_world_cam. Rigid transforms invert in
// closed form (R is orthonormal): R⁻¹ = Rᵀ, t⁻¹ = −Rᵀ t. Done ONCE per
// frame on the host so the kernel never repeats it 2 million times.
// ---------------------------------------------------------------------------
static PoseRt invert_pose(const PoseRt& T)
{
    PoseRt inv;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            inv.r[i * 3 + j] = T.r[j * 3 + i];                      // transpose
    for (int i = 0; i < 3; ++i)
        inv.t[i] = -(inv.r[i * 3 + 0] * T.t[0] + inv.r[i * 3 + 1] * T.t[1]
                     + inv.r[i * 3 + 2] * T.t[2]);                  // -Rᵀ t
    return inv;
}

// ---------------------------------------------------------------------------
// Sample loading — the committed camera path: one CAM row (intrinsics) and
// N POSE rows (T_world_cam as translation + quaternion). Strict: unknown
// labels, short rows, or a missing CAM abort the demo (repo loader
// discipline — a half-read sample must never half-run).
// ---------------------------------------------------------------------------
struct Sample {
    Intrinsics K = {};
    std::vector<PoseRt> poses;      // T_world_cam per frame, file order
    bool loaded = false;
};

static Sample load_sample(const std::string& path)
{
    Sample sc;
    std::ifstream in(path);
    if (!in.is_open()) return sc;

    bool have_cam = false;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;      // comments carry provenance
        std::stringstream ss(line);
        std::string label, cell;
        std::getline(ss, label, ',');
        if (label == "CAM") {
            float vals[6];                                  // width,height,fx,fy,cx,cy
            for (int i = 0; i < 6; ++i) {
                if (!std::getline(ss, cell, ',')) { std::fprintf(stderr, "sample: short CAM row\n"); return Sample{}; }
                vals[i] = std::strtof(cell.c_str(), nullptr);
            }
            sc.K.width  = static_cast<int>(vals[0]);
            sc.K.height = static_cast<int>(vals[1]);
            sc.K.fx = vals[2]; sc.K.fy = vals[3];
            sc.K.cx = vals[4]; sc.K.cy = vals[5];
            have_cam = true;
        } else if (label == "POSE") {
            float vals[8];                                  // idx,tx,ty,tz,qw,qx,qy,qz
            for (int i = 0; i < 8; ++i) {
                if (!std::getline(ss, cell, ',')) { std::fprintf(stderr, "sample: short POSE row\n"); return Sample{}; }
                vals[i] = std::strtof(cell.c_str(), nullptr);
            }
            PoseRt T;
            quat_to_rot(vals[4], vals[5], vals[6], vals[7], T.r);   // (w,x,y,z)
            T.t[0] = vals[1]; T.t[1] = vals[2]; T.t[2] = vals[3];
            sc.poses.push_back(T);
        } else {
            std::fprintf(stderr, "sample: unknown row label '%s'\n", label.c_str());
            return Sample{};
        }
    }
    if (!have_cam || sc.poses.empty() ||
        sc.K.width < 1 || sc.K.height < 1 || sc.K.fx <= 0.0f || sc.K.fy <= 0.0f) {
        std::fprintf(stderr, "sample: missing/invalid CAM or no POSE rows\n");
        return Sample{};
    }
    sc.loaded = true;
    return sc;
}

// Exe-relative project root (the exe sits at build/x64/<Config>/, three
// levels below the project folder — the repo's standard data-path trick).
static std::string project_root_from(const char* argv0)
{
    std::string exe(argv0 ? argv0 : "");
    size_t cut = exe.find_last_of("/\\");
    if (cut == std::string::npos) return ".";
    return exe.substr(0, cut) + "/../../..";
}

static std::string find_sample(const std::string& cli_path, const char* argv0)
{
    std::vector<std::string> candidates;
    if (!cli_path.empty()) candidates.push_back(cli_path);
    candidates.push_back(project_root_from(argv0) + "/data/sample/camera_path.csv");
    candidates.push_back("data/sample/camera_path.csv");
    candidates.push_back("../data/sample/camera_path.csv");
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

// ---------------------------------------------------------------------------
// write_obj — the mesh artifact, in Wavefront OBJ: three "v x y z" lines
// per triangle followed later by "f" index triples (1-based, OBJ's rule).
// Deliberately UN-indexed (every triangle carries its own vertices): shared-
// vertex welding is a post-process this teaching kernel does not do —
// README Exercise 3 adds it and measures the size win.
// ---------------------------------------------------------------------------
static bool write_obj(const std::string& path, const std::vector<float>& verts, int n_tris)
{
    std::ofstream f(path);
    if (!f.is_open()) return false;
    f << "# mesh.obj - iso-surface extracted by project 05.01 (SYNTHETIC scene: sphere over plane)\n";
    f << "# " << n_tris << " triangles, un-indexed (3 unique vertices each); units: meters, world frame (z up)\n";
    for (int t = 0; t < n_tris; ++t) {
        const float* p = &verts[static_cast<size_t>(t) * 9];
        for (int k = 0; k < 3; ++k)
            f << "v " << p[k * 3] << ' ' << p[k * 3 + 1] << ' ' << p[k * 3 + 2] << '\n';
    }
    for (int t = 0; t < n_tris; ++t)                       // OBJ indices are 1-based
        f << "f " << t * 3 + 1 << ' ' << t * 3 + 2 << ' ' << t * 3 + 3 << '\n';
    return f.good();
}

// ---------------------------------------------------------------------------
// write_slice_pgm — a vertical (x–z) slice through the fused volume at the
// sphere's y, as an 8-bit PGM (openable everywhere, like 07.09's fields):
//   black (0)        = never observed
//   dark → light     = inside → free space (25 + (tsdf+1)/2 * 230)
//   mid-gray (~140)  = the zero crossing — the surface itself
// You should SEE the circle of the sphere, the line of the plane, and the
// black unobserved shadow beneath both. demo/README.md annotates it.
// ---------------------------------------------------------------------------
static bool write_slice_pgm(const std::string& path,
                            const std::vector<float>& tsdf, const std::vector<float>& weight)
{
    const int iy = kVolN / 2;      // slice plane: y ≈ 0.01 m — through the sphere center
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    f << "P5\n" << kVolN << ' ' << kVolN << "\n255\n";     // binary 8-bit PGM header
    for (int iz = kVolN - 1; iz >= 0; --iz) {              // top row = highest z (image "up" = world up)
        for (int ix = 0; ix < kVolN; ++ix) {
            const size_t v = (static_cast<size_t>(iz) * kVolN + iy) * kVolN + ix;
            unsigned char g = 0;                           // unobserved = black
            if (weight[v] > 0.0f)
                g = static_cast<unsigned char>(25.0f + (tsdf[v] + 1.0f) * 0.5f * 230.0f);
            f.put(static_cast<char>(g));
        }
    }
    return f.good();
}

// ---------------------------------------------------------------------------
// main — the seven stages described in the file header.
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    std::string data_path;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--data") && i + 1 < argc) data_path = argv[++i];
        else {
            std::fprintf(stderr, "usage: %s [--data camera_path.csv]\n", argv[0]);
            return 2;
        }
    }

    std::printf("[demo] TSDF fusion + marching cubes: sphere-over-plane scene (project 05.01)\n");
    print_device_info();

    // ---- stage 1: the committed sample --------------------------------------
    const std::string sample_path = find_sample(data_path, argv[0]);
    if (sample_path.empty()) {
        std::printf("SAMPLE: NOT FOUND — data/sample/camera_path.csv missing (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (sample missing)\n");
        return 1;
    }
    std::printf("[info] sample file: %s\n", sample_path.c_str());
    Sample sc = load_sample(sample_path);
    if (!sc.loaded) {
        std::printf("SAMPLE: MALFORMED — see stderr\n");
        std::printf("RESULT: FAIL (sample malformed)\n");
        return 1;
    }
    const int n_frames = static_cast<int>(sc.poses.size());

    std::printf("PROBLEM: TSDF fusion, %d^3 voxels @ %.2f m (%.2f m cube), truncation mu=%.2f m, %d depth frames %dx%d, FP32\n",
                kVolN, static_cast<double>(kVoxelSize),
                static_cast<double>(kVolN * kVoxelSize), static_cast<double>(kTruncation),
                n_frames, sc.K.width, sc.K.height);
    std::printf("SAMPLE: %d camera poses on a circle + pinhole intrinsics [synthetic]; depth rendered in-code by ray casting the analytic scene\n",
                n_frames);

    // ---- stage 2: render the synthetic depth frames (host, exact) -----------
    const size_t px_per_frame = static_cast<size_t>(sc.K.width) * sc.K.height;
    std::vector<std::vector<float>> frames(static_cast<size_t>(n_frames));
    {
        CpuTimer rt;
        rt.begin();
        for (int i = 0; i < n_frames; ++i)
            render_depth(sc.K, sc.poses[static_cast<size_t>(i)], frames[static_cast<size_t>(i)]);
        std::printf("[time] rendered %d synthetic depth frames on the host in %.1f ms (closed-form ray casts)\n",
                    n_frames, rt.end_ms());
    }

    // ---- persistent device buffers (allocated once — 2x 8 MiB volumes,
    //      one reusable depth frame, the mesh buffer + its counter) ----------
    const size_t n_vox = static_cast<size_t>(kVolN) * kVolN * kVolN;
    float *d_tsdf = nullptr, *d_weight = nullptr, *d_depth = nullptr, *d_verts = nullptr;
    int* d_tri_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tsdf, n_vox * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weight, n_vox * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_depth, px_per_frame * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_verts, static_cast<size_t>(kMaxTriangles) * 9 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tri_count, sizeof(int)));

    std::vector<float> tsdf(n_vox), weight(n_vox);        // host copies (downloads)

    // ======================= VERIFY STAGE ====================================
    // The first 4 frames through both paths, into separate volumes.
    // Tolerance justification: both paths are engineered to execute the same
    // IEEE operations (explicit fmaf twins, kernels.cuh §determinism), so the
    // expected deviation is ZERO; 1e-5 abs on TSDF is pure headroom against
    // compiler surprises, while any indexing/projection/layout bug shifts
    // voxels by order 0.1–2.0. Weights count identically or something is
    // deeply wrong — compared exactly.
    const int n_verify = n_frames < 4 ? n_frames : 4;
    {
        launch_volume_clear(d_tsdf, d_weight);

        double gpu_ms = 0.0;
        for (int i = 0; i < n_verify; ++i) {
            CUDA_CHECK(cudaMemcpy(d_depth, frames[static_cast<size_t>(i)].data(),
                                  px_per_frame * sizeof(float), cudaMemcpyHostToDevice));
            GpuTimer gt;
            gt.begin();
            launch_tsdf_integrate(d_depth, sc.K, invert_pose(sc.poses[static_cast<size_t>(i)]),
                                  d_tsdf, d_weight);
            gpu_ms += static_cast<double>(gt.end_ms());
        }
        CUDA_CHECK(cudaMemcpy(tsdf.data(), d_tsdf, n_vox * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(weight.data(), d_weight, n_vox * sizeof(float), cudaMemcpyDeviceToHost));

        // The CPU twin: same frames, same order, fresh host volume with the
        // same defined initial state (+1 / 0) the clear kernel writes.
        std::vector<float> tsdf_cpu(n_vox, 1.0f), weight_cpu(n_vox, 0.0f);
        CpuTimer ct;
        ct.begin();
        for (int i = 0; i < n_verify; ++i)
            tsdf_integrate_cpu(frames[static_cast<size_t>(i)].data(), sc.K,
                               invert_pose(sc.poses[static_cast<size_t>(i)]),
                               tsdf_cpu.data(), weight_cpu.data());
        const double cpu_ms = ct.end_ms();

        float worst_t = 0.0f, worst_w = 0.0f;
        size_t observed = 0;
        for (size_t v = 0; v < n_vox; ++v) {
            const float dt = std::fabs(tsdf[v] - tsdf_cpu[v]);
            const float dw = std::fabs(weight[v] - weight_cpu[v]);
            if (dt > worst_t) worst_t = dt;
            if (dw > worst_w) worst_w = dw;
            if (weight_cpu[v] > 0.0f) ++observed;
        }
        const bool verify_pass = (worst_t <= 1e-5f) && (worst_w == 0.0f);
        std::printf("[info] verify: worst |tsdf_gpu - tsdf_cpu| = %.3e, worst weight diff = %.1f, %zu voxels observed after %d frames\n",
                    static_cast<double>(worst_t), static_cast<double>(worst_w), observed, n_verify);
        std::printf("[time] integration per frame (%d-frame subset): CPU %.1f ms | GPU kernel %.3f ms | speed-up %.0fx (teaching artifact; kernel only)\n",
                    n_verify, cpu_ms / n_verify, gpu_ms / n_verify,
                    (cpu_ms / n_verify) / ((gpu_ms / n_verify) > 0.0 ? (gpu_ms / n_verify) : 1.0));
        std::printf("VERIFY: %s (GPU TSDF matches CPU reference on the %d-frame subset within abs tol 1e-5)\n",
                    verify_pass ? "PASS" : "FAIL", n_verify);
        if (!verify_pass) {
            std::printf("RESULT: FAIL (GPU/CPU integration disagreement — fix before trusting the map)\n");
            return 1;
        }
    }

    // ======================= FULL FUSION =====================================
    {
        launch_volume_clear(d_tsdf, d_weight);
        double gpu_ms = 0.0;
        for (int i = 0; i < n_frames; ++i) {
            CUDA_CHECK(cudaMemcpy(d_depth, frames[static_cast<size_t>(i)].data(),
                                  px_per_frame * sizeof(float), cudaMemcpyHostToDevice));
            GpuTimer gt;
            gt.begin();
            launch_tsdf_integrate(d_depth, sc.K, invert_pose(sc.poses[static_cast<size_t>(i)]),
                                  d_tsdf, d_weight);
            gpu_ms += static_cast<double>(gt.end_ms());
        }
        std::printf("[time] full fusion: %d frames, %.3f ms average GPU integration kernel per frame\n",
                    n_frames, gpu_ms / n_frames);
        CUDA_CHECK(cudaMemcpy(tsdf.data(), d_tsdf, n_vox * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(weight.data(), d_weight, n_vox * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // ======================= GROUND-TRUTH CHECK ==============================
    // The fused TSDF against the analytic scene SDF, two shells:
    //   SURFACE shell (|sdf_gt| <= voxel/2): where the mesh will land — the
    //     fused field must localize the surface accurately here.
    //   BAND shell (|sdf_gt| <= mu/2): the trusted half-band — dominated by
    //     the projective 1/cos(incidence) bias on the obliquely-viewed
    //     plane, so the bound is wider (THEORY.md §algorithm quantifies).
    // Bounds below are 3-6x above values measured on the reference machine
    // (RTX 2080 SUPER, 2026-07: surface max ≈ 6.5e-3 m, band max ≈ 1.6e-2 m)
    // — wide enough that FP ulp drift across platforms cannot flip the
    // verdict, tight enough that a projection or truncation bug (order
    // 0.1 m+) fails instantly.
    {
        double sum_surf = 0.0, max_surf = 0.0;   // |fused*mu - sdf_gt| (m), surface shell
        double sum_band = 0.0, max_band = 0.0;   // same, band shell
        size_t n_surf = 0, n_band = 0;
        for (int iz = 0; iz < kVolN; ++iz)
            for (int iy = 0; iy < kVolN; ++iy)
                for (int ix = 0; ix < kVolN; ++ix) {
                    const size_t v = (static_cast<size_t>(iz) * kVolN + iy) * kVolN + ix;
                    if (weight[v] <= 0.0f) continue;                     // never observed
                    const float x = kVolOriginX + (static_cast<float>(ix) + 0.5f) * kVoxelSize;
                    const float y = kVolOriginY + (static_cast<float>(iy) + 0.5f) * kVoxelSize;
                    const float z = kVolOriginZ + (static_cast<float>(iz) + 0.5f) * kVoxelSize;
                    const float gt = scene_sdf(x, y, z);                 // exact within the band
                    const double err = std::fabs(static_cast<double>(tsdf[v]) * kTruncation - gt);
                    if (std::fabs(gt) <= 0.5f * kVoxelSize) {
                        sum_surf += err; ++n_surf; if (err > max_surf) max_surf = err;
                    }
                    if (std::fabs(gt) <= 0.5f * kTruncation) {
                        sum_band += err; ++n_band; if (err > max_band) max_band = err;
                    }
                }
        std::printf("[info] ground truth: surface shell (|sdf|<=%.3f m): %zu voxels, mean err %.2e m, max %.2e m\n",
                    static_cast<double>(0.5f * kVoxelSize), n_surf,
                    n_surf ? sum_surf / n_surf : 0.0, max_surf);
        std::printf("[info] ground truth: half band     (|sdf|<=%.3f m): %zu voxels, mean err %.2e m, max %.2e m\n",
                    static_cast<double>(0.5f * kTruncation), n_band,
                    n_band ? sum_band / n_band : 0.0, max_band);
        const bool gt_pass = n_surf > 1000                 // the shell must actually be observed
                          && max_surf <= 0.020             // 1 voxel     (measured max ~6.5e-3)
                          && max_band <= 0.060             // mu/2        (measured max ~1.6e-2)
                          && (sum_surf / n_surf) <= 0.010; // half voxel  (measured mean ~1.5e-3)
        std::printf("GROUND TRUTH: %s (fused TSDF vs analytic scene SDF: surface max err <= 0.02 m, band max err <= 0.06 m)\n",
                    gt_pass ? "PASS" : "FAIL");
        if (!gt_pass) {
            std::printf("RESULT: FAIL (fused TSDF strayed from analytic ground truth — see [info] lines)\n");
            return 1;
        }
    }

    // ======================= MARCHING CUBES ==================================
    int n_tris = 0;
    std::vector<float> verts;
    {
        CUDA_CHECK(cudaMemset(d_tri_count, 0, sizeof(int)));   // the append counter starts at 0
        GpuTimer gt;
        gt.begin();
        launch_marching_cubes(d_tsdf, d_weight, kMaxTriangles, d_verts, d_tri_count);
        const float mc_ms = gt.end_ms();
        CUDA_CHECK(cudaMemcpy(&n_tris, d_tri_count, sizeof(int), cudaMemcpyDeviceToHost));
        std::printf("[time] marching cubes kernel: %.3f ms over %d^3 cells\n", mc_ms, kVolN - 1);

        if (n_tris > kMaxTriangles) {
            // The counter passed the buffer capacity — geometry was dropped.
            // With 4x headroom this means a real bug (or a changed scene);
            // fail loudly rather than write a silently incomplete mesh.
            std::printf("MESH: FAIL (triangle count %d exceeds buffer capacity %d)\n", n_tris, kMaxTriangles);
            std::printf("RESULT: FAIL (marching-cubes output overflow)\n");
            return 1;
        }

        // Check 1 — EXACT count match against the CPU recount (order-free
        // invariant of the atomic append; reference_cpu.cpp explains).
        const long long cpu_tris = marching_cubes_count_cpu(tsdf.data(), weight.data());

        // Check 2 — every emitted vertex must lie ON the analytic surface
        // (within interpolation + fusion error). The strongest geometric
        // check available: a wrong table row, edge mapping, or interpolation
        // would throw vertices centimeters off instantly. Bound: 1 voxel
        // (measured max on the reference machine: ~4.4e-3 m).
        verts.assign(static_cast<size_t>(n_tris) * 9, 0.0f);
        if (n_tris > 0)
            CUDA_CHECK(cudaMemcpy(verts.data(), d_verts,
                                  static_cast<size_t>(n_tris) * 9 * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        double max_vert_err = 0.0;
        for (size_t i = 0; i + 2 < verts.size(); i += 3) {
            const double e = std::fabs(static_cast<double>(scene_sdf(verts[i], verts[i + 1], verts[i + 2])));
            if (e > max_vert_err) max_vert_err = e;
        }

        // Check 3 — the count itself sits in a stable expected range. The
        // exact number is deterministic per machine but may drift by a few
        // triangles across GPU generations (borderline cells); the committed
        // scene yields ~66k (reference machine), so [40000, 100000] is wide
        // margin in both directions while a broken pass (0, or millions)
        // fails instantly.
        std::printf("[info] mesh: %d triangles (GPU) vs %lld (CPU recount); max |sdf(vertex)| = %.2e m\n",
                    n_tris, cpu_tris, max_vert_err);
        const bool mesh_pass = (static_cast<long long>(n_tris) == cpu_tris)
                            && (n_tris >= 40000 && n_tris <= 100000)
                            && (max_vert_err <= 0.020);
        std::printf("MESH: %s (GPU count == CPU recount, count in [40000, 100000], vertices on the analytic surface within 0.02 m)\n",
                    mesh_pass ? "PASS" : "FAIL");
        if (!mesh_pass) {
            std::printf("RESULT: FAIL (marching-cubes check failed — see [info] line)\n");
            return 1;
        }
    }

    CUDA_CHECK(cudaFree(d_tsdf));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_depth));
    CUDA_CHECK(cudaFree(d_verts));
    CUDA_CHECK(cudaFree(d_tri_count));

    // ======================= ARTIFACTS =======================================
    // demo/out/ is git-ignored scratch; both artifacts regenerate every run.
    const std::string out_dir = project_root_from(argv[0]) + "/demo/out";
    bool artifacts_ok = ensure_dir(out_dir);
    if (artifacts_ok && write_obj(out_dir + "/mesh.obj", verts, n_tris))
        std::printf("ARTIFACT: wrote demo/out/mesh.obj\n");
    else { std::printf("ARTIFACT: FAILED to write demo/out/mesh.obj\n"); artifacts_ok = false; }
    if (artifacts_ok && write_slice_pgm(out_dir + "/tsdf_slice.pgm", tsdf, weight))
        std::printf("ARTIFACT: wrote demo/out/tsdf_slice.pgm\n");
    else { std::printf("ARTIFACT: FAILED to write demo/out/tsdf_slice.pgm\n"); artifacts_ok = false; }

    // ---- the stable verdict ---------------------------------------------------
    if (artifacts_ok) {
        std::printf("RESULT: PASS (fusion verified against CPU twin and analytic ground truth; mesh extracted)\n");
        return 0;
    }
    std::printf("RESULT: FAIL (artifact write failed)\n");
    return 1;
}
