// ===========================================================================
// main.cu — entry point for project 06.05
//           STOMP: parallel noisy-rollout trajectory optimization
//           (teaching core: 2-D point robot through an obstacle field)
//
// What this program does, start to finish
// ---------------------------------------
//   1. Print the banner + GPU info; load the committed scenario (map size,
//      start, goal, obstacles) from data/sample/, and build the obstacle-cost
//      FIELD from it on the host (distance-based inflation).
//   2. Precompute the smoothing matrix M from R = A^T A (the finite-difference
//      acceleration operator) — the piece that makes STOMP's noise SMOOTH.
//   3. VERIFY STAGE (the §5 GPU-vs-CPU gate): on iteration 0's exact inputs,
//      score ALL K noisy trajectories on the GPU kernel AND the CPU oracle,
//      and require the total costs to agree within a relative tolerance.
//   4. OPTIMIZATION LOOP: run STOMP for up to kMaxIters iterations — sample K
//      smooth-noise trajectories, score them on the GPU, and update EACH
//      waypoint by a per-waypoint softmin blend (STOMP's signature move),
//      smoothed through M. Stop early when the cost plateaus.
//   5. VERDICT: the final trajectory must be collision-free with margin (max
//      field value along it below a documented threshold) AND its total cost
//      must be well below the straight-line initialization. Write the final
//      path (trajectory.csv) and the cost field with the path burned in
//      (costfield.pgm). Exit 0 only if verify + verdict both hold.
//
// The STOMP update implemented here (derivation in THEORY.md §the-math):
//      score       -> Sloc[j][k] = local obstacle cost at waypoint j of rollout k
//      per waypoint j:  w_k = exp(-h * (Sloc[j][k]-min_k)/(max_k-min_k))    (softmin)
//                       raw_dtheta[j] = sum_k (w_k/sum w) * eps_k[j]
//      dtheta      = M * raw_dtheta          (smooth the update; keeps ends fixed)
//      theta      += dtheta                  (start & goal never move)
// The PER-WAYPOINT softmin is the whole point: MPPI (08.01) weights each WHOLE
// trajectory by one number; STOMP weights each waypoint SEPARATELY by its own
// local cost, so a rollout can be "good here, bad there" and contribute only
// where it helps. Keeping this blend on the host (it is O(K*N) trivial
// arithmetic) puts the entire STOMP algorithm in ~40 readable lines below.
//
// Determinism: noise is host-generated (xorshift32 + Box-Muller, smoothed by
// M), seeded per iteration — the run is bit-reproducible on this machine.
// Host libm ulp differences and GPU low-bit scoring differences can perturb
// the trajectory across PLATFORMS; the stable output lines therefore carry no
// trajectory numbers, only PASS/FAIL against thresholds with wide margins
// (THEORY.md §numerics).
//
// Output contract: stable lines "[demo]", "PROBLEM:", "SCENARIO:", "VERIFY:",
// "ARTIFACT:", "RESULT:" — "[info]"/"[time]" unchecked. Change a stable line
// => update demo/expected_output.txt in the same commit.
//
// Read this first, then kernels.cuh -> reference_cpu.cpp -> kernels.cu.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>                 // std::atoi, std::strtof, std::exit
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>                 // std::swap (Gauss-Jordan row swap in build_M)
#include <vector>
#ifdef _WIN32
#include <direct.h>               // _mkdir (std::filesystem avoided in .cu — see 07.09)
#else
#include <sys/stat.h>
#endif

// ---------------------------------------------------------------------------
// Deterministic Gaussian noise: xorshift32 (the repo's portable generator)
// + Box-Muller. STOMP's exploration noise is Gaussian; we then MIX it through
// M to make it smooth (fill_smooth_noise below). Same generator as 08.01 so
// the whole repo shares one reproducible RNG story.
// ---------------------------------------------------------------------------
static inline uint32_t xorshift32(uint32_t& state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

static inline float uniform01(uint32_t& state)      // (0,1] — never 0, safe for log()
{
    return (xorshift32(state) >> 8) * (1.0f / 16777216.0f) + (0.5f / 16777216.0f);
}

// One N(0, sigma^2) draw. Box-Muller in double for the transcendental step,
// cast at the end — the cheap way to keep the tails well-behaved in FP32.
static inline float gaussian(uint32_t& state, float sigma)
{
    const double u1 = static_cast<double>(uniform01(state));
    const double u2 = static_cast<double>(uniform01(state));
    const double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    return sigma * static_cast<float>(z);
}

// ===========================================================================
// Scenario: the committed "task definition" — the map, the endpoints, and the
// obstacles. Loaded from data/sample/obstacle_scenario.csv (strict loader).
// ===========================================================================
struct Circle { float cx, cy, r; };   // obstacle: centre (m) + radius (m)

struct Scenario {
    float map_w = 0.0f, map_h = 0.0f;  // world size (m); this project assumes a SQUARE map (one cell_m)
    float start[2] = { 0.0f, 0.0f };   // fixed start (m)
    float goal[2]  = { 0.0f, 0.0f };   // fixed goal (m)
    std::vector<Circle> obstacles;     // circular obstacles
    bool loaded = false;
};

// Rows: "MAP,w,h", "START,x,y", "GOAL,x,y", "OBST,cx,cy,r" (>=1). '#' comments.
static Scenario load_scenario(const std::string& path)
{
    Scenario sc;
    std::ifstream in(path);
    if (!in.is_open()) return sc;

    bool have_map = false, have_start = false, have_goal = false;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string label, cell;
        std::getline(ss, label, ',');

        auto next_float = [&](float& out) -> bool {
            if (!std::getline(ss, cell, ',')) return false;
            out = std::strtof(cell.c_str(), nullptr);
            return true;
        };

        if (label == "MAP") {
            if (!next_float(sc.map_w) || !next_float(sc.map_h)) { std::fprintf(stderr, "scenario: short MAP row\n"); return Scenario{}; }
            have_map = true;
        } else if (label == "START") {
            if (!next_float(sc.start[0]) || !next_float(sc.start[1])) { std::fprintf(stderr, "scenario: short START row\n"); return Scenario{}; }
            have_start = true;
        } else if (label == "GOAL") {
            if (!next_float(sc.goal[0]) || !next_float(sc.goal[1])) { std::fprintf(stderr, "scenario: short GOAL row\n"); return Scenario{}; }
            have_goal = true;
        } else if (label == "OBST") {
            Circle c{};
            if (!next_float(c.cx) || !next_float(c.cy) || !next_float(c.r)) { std::fprintf(stderr, "scenario: short OBST row\n"); return Scenario{}; }
            sc.obstacles.push_back(c);
        } else {
            std::fprintf(stderr, "scenario: unknown row label '%s'\n", label.c_str());
            return Scenario{};
        }
    }
    if (!have_map || !have_start || !have_goal || sc.obstacles.empty() ||
        sc.map_w <= 0.0f || sc.map_h <= 0.0f) {
        std::fprintf(stderr, "scenario: missing MAP/START/GOAL or no obstacles\n");
        return Scenario{};
    }
    sc.loaded = true;
    return sc;
}

// ---------------------------------------------------------------------------
// build_cost_field — inflate the analytic obstacles into a dense cost grid.
//
// For every grid cell we compute the SIGNED distance d to the nearest obstacle
// boundary (negative inside), then map d to a cost by the falloff defined in
// kernels.cuh: high inside, a smooth quadratic halo out to kInfl, zero beyond.
// Doing this from analytic circles (d = |p - c| - r) keeps the project
// self-contained — no external SDF library, no dependency on the jump-flooding
// project 07.09 (which computes exactly this field from an occupancy grid; on
// a real robot that is where this field comes from — README §System context).
// The field is the ONLY thing the GPU kernel needs to know about obstacles.
// ---------------------------------------------------------------------------
static void build_cost_field(const Scenario& sc, std::vector<float>& field,
                             int gw, int gh, float cell_m)
{
    field.assign(static_cast<size_t>(gw) * gh, 0.0f);
    for (int iy = 0; iy < gh; ++iy) {
        const float y = iy * cell_m;                 // world y of this cell (m)
        for (int ix = 0; ix < gw; ++ix) {
            const float x = ix * cell_m;             // world x of this cell (m)

            // Signed distance to the NEAREST obstacle boundary.
            float d = 1e30f;
            for (const Circle& c : sc.obstacles) {
                const float dx = x - c.cx, dy = y - c.cy;
                const float dc = std::sqrt(dx * dx + dy * dy) - c.r;   // <0 inside this circle
                if (dc < d) d = dc;
            }

            // Distance -> cost (the falloff from kernels.cuh).
            float cost;
            if (d <= 0.0f) {
                cost = kCostCollision + (-d) * kPenetration;           // inside: peak + gradient outward
            } else if (d < kInfl) {
                const float t = (kInfl - d) / kInfl;                   // 1 at boundary, 0 at kInfl
                cost = kCostCollision * t * t;                          // smooth quadratic halo
            } else {
                cost = 0.0f;                                           // free space
            }
            field[static_cast<size_t>(iy) * gw + ix] = cost;
        }
    }
}

// ===========================================================================
// The smoothing matrix M — the mathematical heart of STOMP (THEORY.md §math).
//
// A is the finite-difference ACCELERATION operator on the kN interior
// waypoints (Dirichlet/zero boundary): row i is the second difference stencil
//     A[i][i-1] = 1,  A[i][i] = -2,  A[i][i+1] = 1
// (boundary terms dropped at i=0 and i=kN-1). R = A^T A is symmetric positive
// definite and banded; its inverse R^-1 is DENSE and SMOOTH — its columns are
// samples of the Green's function of the discrete biharmonic operator, i.e.
// smooth bumps that decay to ~0 at the ends. That is exactly what we want:
//   * Smooth NOISE:  eps = M z turns per-waypoint white noise z into a
//     spatially-smooth perturbation (each column of M is a smooth basis
//     function; independent per-waypoint noise would give a jagged path).
//   * Endpoint safety: because M's columns decay at the boundary, the noise
//     and the update barely move the waypoints next to the fixed start/goal.
// M = R^-1 with each COLUMN scaled so its largest entry is 1/N (the STOMP
// paper's scaling — keeps the perturbation magnitude sane). Computed in DOUBLE
// (R's condition number ~ N^4 ~ 1.7e7; double carries ~15 digits, so we keep
// ~8 — plenty), stored as FP32 for the runtime blends.
// ===========================================================================
static void build_M(std::vector<float>& M_out)
{
    const int n = kN;

    // A (n x n), then R = A^T A (n x n), both in double.
    std::vector<double> A(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) {
        if (i - 1 >= 0) A[static_cast<size_t>(i) * n + (i - 1)] = 1.0;
        A[static_cast<size_t>(i) * n + i] = -2.0;
        if (i + 1 < n)  A[static_cast<size_t>(i) * n + (i + 1)] = 1.0;
    }
    std::vector<double> R(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double s = 0.0;
            for (int m = 0; m < n; ++m)                          // R[i][j] = sum_m A[m][i] A[m][j]
                s += A[static_cast<size_t>(m) * n + i] * A[static_cast<size_t>(m) * n + j];
            R[static_cast<size_t>(i) * n + j] = s;
        }

    // Invert R by Gauss-Jordan with partial pivoting on the augmented [R | I].
    // n = 64, so this ~n^3 work is microseconds; clarity beats a fancy solver.
    std::vector<double> aug(static_cast<size_t>(n) * (2 * n), 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) aug[static_cast<size_t>(i) * (2 * n) + j] = R[static_cast<size_t>(i) * n + j];
        aug[static_cast<size_t>(i) * (2 * n) + (n + i)] = 1.0;    // identity on the right
    }
    for (int col = 0; col < n; ++col) {
        // Partial pivot: swap in the row with the largest |pivot| for stability.
        int piv = col;
        double best = std::fabs(aug[static_cast<size_t>(col) * (2 * n) + col]);
        for (int r = col + 1; r < n; ++r) {
            const double v = std::fabs(aug[static_cast<size_t>(r) * (2 * n) + col]);
            if (v > best) { best = v; piv = r; }
        }
        if (piv != col)
            for (int j = 0; j < 2 * n; ++j)
                std::swap(aug[static_cast<size_t>(col) * (2 * n) + j], aug[static_cast<size_t>(piv) * (2 * n) + j]);

        // Normalize the pivot row, then eliminate the column from all others.
        const double pv = aug[static_cast<size_t>(col) * (2 * n) + col];
        for (int j = 0; j < 2 * n; ++j) aug[static_cast<size_t>(col) * (2 * n) + j] /= pv;
        for (int r = 0; r < n; ++r) {
            if (r == col) continue;
            const double f = aug[static_cast<size_t>(r) * (2 * n) + col];
            if (f == 0.0) continue;
            for (int j = 0; j < 2 * n; ++j)
                aug[static_cast<size_t>(r) * (2 * n) + j] -= f * aug[static_cast<size_t>(col) * (2 * n) + j];
        }
    }

    // Extract R^-1, then scale each COLUMN so its max |entry| is 1/N.
    std::vector<double> Rinv(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            Rinv[static_cast<size_t>(i) * n + j] = aug[static_cast<size_t>(i) * (2 * n) + (n + j)];

    M_out.assign(static_cast<size_t>(n) * n, 0.0f);
    for (int j = 0; j < n; ++j) {
        double colmax = 0.0;
        for (int i = 0; i < n; ++i) {
            const double v = std::fabs(Rinv[static_cast<size_t>(i) * n + j]);
            if (v > colmax) colmax = v;
        }
        const double scale = (colmax > 0.0) ? (1.0 / (static_cast<double>(n) * colmax)) : 0.0;
        for (int i = 0; i < n; ++i)
            M_out[static_cast<size_t>(i) * n + j] = static_cast<float>(Rinv[static_cast<size_t>(i) * n + j] * scale);
    }
}

// matvec: out[i] = sum_j M[i][j] * v[j], for an n x n M (row-major) and length-n v.
static void matvec(const std::vector<float>& M, const float* v, float* out, int n)
{
    for (int i = 0; i < n; ++i) {
        double acc = 0.0;                                       // double accumulator: n=64 tiny terms
        const float* row = &M[static_cast<size_t>(i) * n];
        for (int j = 0; j < n; ++j) acc += static_cast<double>(row[j]) * v[j];
        out[i] = static_cast<float>(acc);
    }
}

// ---------------------------------------------------------------------------
// fill_smooth_noise — draw white noise z and mix it to SMOOTH noise eps = M z,
// writing eps into the TRANSPOSED arrays epsx[j*K+k], epsy[j*K+k].
//
// The per-iteration seed mixes the base seed with the iteration index (odd
// multiplier -> full-period stream separation) so every iteration explores
// fresh noise. For each rollout k and each dimension we draw kN white samples
// and pass them through M — the single step that distinguishes STOMP's
// spatially-correlated exploration from a naive per-waypoint jitter.
// ---------------------------------------------------------------------------
static void fill_smooth_noise(std::vector<float>& epsx, std::vector<float>& epsy,
                              int K, int iter, uint32_t base_seed,
                              const std::vector<float>& M)
{
    uint32_t s = base_seed + 1000003u * static_cast<uint32_t>(iter + 1);
    if (s == 0) s = 1u;

    std::vector<float> z(kN), e(kN);                            // white draw + its smoothed image
    for (int k = 0; k < K; ++k) {
        // x dimension
        for (int j = 0; j < kN; ++j) z[j] = gaussian(s, kNoiseSigma);
        matvec(M, z.data(), e.data(), kN);
        for (int j = 0; j < kN; ++j) epsx[static_cast<size_t>(j) * K + k] = e[j];   // transposed store
        // y dimension
        for (int j = 0; j < kN; ++j) z[j] = gaussian(s, kNoiseSigma);
        matvec(M, z.data(), e.data(), kN);
        for (int j = 0; j < kN; ++j) epsy[static_cast<size_t>(j) * K + k] = e[j];
    }
}

// ---------------------------------------------------------------------------
// stomp_update — the per-waypoint softmin blend + M-smoothing (host side).
//
// For each interior waypoint j: normalize the K local costs Sloc[j][k] to
// [0,1] by (S-min)/(max-min) (STOMP's per-timestep normalization keeps the
// softmin well-scaled regardless of absolute cost), exponentiate with
// sensitivity h to get importance weights, and blend the K perturbations at
// that waypoint. The RAW per-waypoint deltas are then smoothed through M
// (dtheta = M * raw) so the update is a smooth, boundary-respecting curve —
// this is where STOMP guarantees its updates stay smooth. theta is updated in
// place; start and goal are not part of theta and never move.
// ---------------------------------------------------------------------------
static void stomp_update(std::vector<float>& theta_x, std::vector<float>& theta_y,
                         const std::vector<float>& Sloc,
                         const std::vector<float>& epsx, const std::vector<float>& epsy,
                         int K, const std::vector<float>& M,
                         std::vector<double>& w /*scratch [K]*/)
{
    std::vector<float> raw_x(kN), raw_y(kN);
    for (int j = 0; j < kN; ++j) {
        const float* Srow  = &Sloc[static_cast<size_t>(j) * K];       // local costs at waypoint j
        const float* ex    = &epsx[static_cast<size_t>(j) * K];       // perturbations at waypoint j
        const float* ey    = &epsy[static_cast<size_t>(j) * K];

        // Per-waypoint min/max for the [0,1] normalization.
        float smin = Srow[0], smax = Srow[0];
        for (int k = 1; k < K; ++k) { smin = Srow[k] < smin ? Srow[k] : smin;
                                      smax = Srow[k] > smax ? Srow[k] : smax; }
        const float denom = smax - smin;

        double wsum = 0.0;
        if (denom <= 1e-12f) {
            // All rollouts equal here (e.g., all in free space): uniform weights.
            for (int k = 0; k < K; ++k) { w[k] = 1.0; wsum += 1.0; }
        } else {
            const double inv = 1.0 / static_cast<double>(denom);
            for (int k = 0; k < K; ++k) {
                w[k] = std::exp(-static_cast<double>(kSensitivity) *
                                (static_cast<double>(Srow[k] - smin) * inv));
                wsum += w[k];
            }
        }

        double accx = 0.0, accy = 0.0;
        for (int k = 0; k < K; ++k) {
            const double pk = w[k] / wsum;                            // normalized probability
            accx += pk * static_cast<double>(ex[k]);
            accy += pk * static_cast<double>(ey[k]);
        }
        raw_x[j] = static_cast<float>(accx);
        raw_y[j] = static_cast<float>(accy);
    }

    // Smooth the raw update through M, then apply it.
    std::vector<float> dx(kN), dy(kN);
    matvec(M, raw_x.data(), dx.data(), kN);
    matvec(M, raw_y.data(), dy.data(), kN);
    for (int j = 0; j < kN; ++j) { theta_x[j] += dx[j]; theta_y[j] += dy[j]; }
}

// ---------------------------------------------------------------------------
// Path assembly + small I/O helpers.
// ---------------------------------------------------------------------------

// Build the full path P[0..kN+1] (start, interior waypoints, goal) into px/py.
static void assemble_path(const Scenario& sc,
                          const std::vector<float>& theta_x, const std::vector<float>& theta_y,
                          std::vector<float>& px, std::vector<float>& py)
{
    px.resize(kPathPoints); py.resize(kPathPoints);
    px[0] = sc.start[0]; py[0] = sc.start[1];
    for (int j = 0; j < kN; ++j) { px[j + 1] = theta_x[j]; py[j + 1] = theta_y[j]; }
    px[kN + 1] = sc.goal[0]; py[kN + 1] = sc.goal[1];
}

static std::string project_root_from(const char* argv0)
{
    std::string exe(argv0 ? argv0 : "");
    size_t cut = exe.find_last_of("/\\");
    if (cut == std::string::npos) return ".";
    return exe.substr(0, cut) + "/../../..";
}

static std::string find_scenario(const std::string& cli_path, const char* argv0)
{
    std::vector<std::string> candidates;
    if (!cli_path.empty()) candidates.push_back(cli_path);
    candidates.push_back(project_root_from(argv0) + "/data/sample/obstacle_scenario.csv");
    candidates.push_back("data/sample/obstacle_scenario.csv");
    candidates.push_back("../data/sample/obstacle_scenario.csv");
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

// write_pgm — P5 (binary) grayscale, the smallest real image format there is:
// one header trio, then raw bytes (07.09's idiom). Rows are emitted TOP-DOWN
// while our buffer is stored y-UP, so we flip vertically here to make the
// saved image read like a map (y increasing upward).
static bool write_pgm(const std::string& path, int w, int h, const std::vector<uint8_t>& gray)
{
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) return false;
    out << "P5\n" << w << " " << h << "\n255\n";
    for (int r = 0; r < h; ++r) {
        const int iy = h - 1 - r;                                     // flip: image row 0 = top = max y
        out.write(reinterpret_cast<const char*>(&gray[static_cast<size_t>(iy) * w]),
                  static_cast<std::streamsize>(w));
    }
    return static_cast<bool>(out);
}

// Render the cost field to grayscale (free = light, obstacle = dark) and burn
// the final path into it as black (0), thickened by one pixel so a thin line
// is visible. The result is this project's inherently-visual artifact.
static bool write_costfield_pgm(const std::string& path, const std::vector<float>& field,
                                int gw, int gh, float cell_m,
                                const std::vector<float>& px, const std::vector<float>& py)
{
    std::vector<uint8_t> img(static_cast<size_t>(gw) * gh);
    // Field -> gray: cost 0 -> 235 (light free space), cost >= kCostCollision -> 30 (dark obstacle).
    for (size_t i = 0; i < img.size(); ++i) {
        float c = field[i];
        if (c > kCostCollision) c = kCostCollision;
        const float t = c / kCostCollision;                          // 0 (free) .. 1 (obstacle)
        img[i] = static_cast<uint8_t>(235.0f - t * (235.0f - 30.0f) + 0.5f);
    }
    // Burn the path: sample each segment densely in pixels, stamp a 3x3 dot.
    for (int s = 0; s + 1 < static_cast<int>(px.size()); ++s) {
        const float x0 = px[s] / cell_m, y0 = py[s] / cell_m;         // world -> pixel coords
        const float x1 = px[s + 1] / cell_m, y1 = py[s + 1] / cell_m;
        const int steps = 1 + static_cast<int>(std::fabs(x1 - x0) + std::fabs(y1 - y0)) * 2;
        for (int q = 0; q <= steps; ++q) {
            const float t = static_cast<float>(q) / static_cast<float>(steps);
            const int cx = static_cast<int>(x0 + t * (x1 - x0) + 0.5f);
            const int cy = static_cast<int>(y0 + t * (y1 - y0) + 0.5f);
            for (int ddy = -1; ddy <= 1; ++ddy)
                for (int ddx = -1; ddx <= 1; ++ddx) {
                    const int ix = cx + ddx, iy = cy + ddy;
                    if (ix >= 0 && ix < gw && iy >= 0 && iy < gh)
                        img[static_cast<size_t>(iy) * gw + ix] = 0;   // path pixel: black
                }
        }
    }
    return write_pgm(path, gw, gh, img);
}

// ---------------------------------------------------------------------------
// main — build the field, precompute M, verify, then run the STOMP loop.
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    int K = kDefaultK;             // noisy rollouts per iteration (CLI-overridable for experiments)
    std::string data_path;
    for (int i = 1; i < argc; ++i) {
        if      (!std::strcmp(argv[i], "--rollouts") && i + 1 < argc) K = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--data")     && i + 1 < argc) data_path = argv[++i];
        else {
            std::fprintf(stderr,
                "usage: %s [--rollouts K] [--data obstacle_scenario.csv]\n"
                "note: non-default K changes the PROBLEM line; the demo diff will flag it.\n",
                argv[0]);
            return 2;
        }
    }
    if (K < 1) { std::fprintf(stderr, "K must be >= 1\n"); return 2; }

    std::printf("[demo] STOMP planner: 2-D trajectory optimization through an obstacle field (project 06.05)\n");
    print_device_info();
    std::printf("PROBLEM: STOMP, K=%d noisy rollouts x N=%d waypoints (2-D), %d iters max, obstacle+smoothness cost, FP32\n",
                K, kN, kMaxIters);

    // ---- scenario + cost field ----------------------------------------------
    const std::string scenario_path = find_scenario(data_path, argv[0]);
    if (scenario_path.empty()) {
        std::printf("SCENARIO: NOT FOUND — data/sample/obstacle_scenario.csv missing (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (scenario missing)\n");
        return 1;
    }
    std::printf("[info] scenario file: %s\n", scenario_path.c_str());
    Scenario sc = load_scenario(scenario_path);
    if (!sc.loaded) {
        std::printf("SCENARIO: MALFORMED — see stderr\n");
        std::printf("RESULT: FAIL (scenario malformed)\n");
        return 1;
    }
    std::printf("SCENARIO: start (%.1f, %.1f) m -> goal (%.1f, %.1f) m through %zu circular obstacles on a %.0fx%.0f m map [synthetic]\n",
                static_cast<double>(sc.start[0]), static_cast<double>(sc.start[1]),
                static_cast<double>(sc.goal[0]),  static_cast<double>(sc.goal[1]),
                sc.obstacles.size(),
                static_cast<double>(sc.map_w), static_cast<double>(sc.map_h));

    const float cell_m = sc.map_w / static_cast<float>(kGridW);   // square-map assumption (see Scenario)
    std::vector<float> field;
    build_cost_field(sc, field, kGridW, kGridH, cell_m);

    // ---- precompute M -------------------------------------------------------
    std::vector<float> M;
    build_M(M);

    // ---- device buffers (allocated once) ------------------------------------
    const size_t eps_count = static_cast<size_t>(kN) * K;
    float *d_field = nullptr, *d_theta_x = nullptr, *d_theta_y = nullptr;
    float *d_epsx = nullptr, *d_epsy = nullptr, *d_Sloc = nullptr, *d_cost = nullptr;
    CUDA_CHECK(cudaMalloc(&d_field, field.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_theta_x, kN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_theta_y, kN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_epsx, eps_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_epsy, eps_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Sloc, eps_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cost, static_cast<size_t>(K) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_field, field.data(), field.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Straight-line initialization: waypoint j sits at fraction (j+1)/(N+1)
    // along the start->goal line. This DELIBERATELY drives through the
    // obstacles (that is the whole point — STOMP must route around them).
    std::vector<float> theta_x(kN), theta_y(kN);
    for (int j = 0; j < kN; ++j) {
        const float f = static_cast<float>(j + 1) / static_cast<float>(kN + 1);
        theta_x[j] = sc.start[0] + f * (sc.goal[0] - sc.start[0]);
        theta_y[j] = sc.start[1] + f * (sc.goal[1] - sc.start[1]);
    }

    std::vector<float> epsx(eps_count), epsy(eps_count);
    std::vector<float> Sloc(eps_count), cost(static_cast<size_t>(K));

    // Initial (straight-line) cost + max field, for the reduction/verdict later.
    std::vector<float> px, py;
    assemble_path(sc, theta_x, theta_y, px, py);
    float maxf_init = 0.0f;
    const double cost_init = evaluate_path_cost(field.data(), kGridW, kGridH, cell_m,
                                                px.data(), py.data(), kPathPoints, &maxf_init);

    // ======================= VERIFY STAGE ====================================
    // Iteration 0's exact inputs through both paths (the §5 gate). Tolerance
    // justification: the total cost is a sum of ~(kN+1)*kSegSamples ~ 520 FP32
    // bilinear samples plus kN smoothness terms; kernel and oracle do the same
    // ops in the same per-rollout order, so they differ only by FMA-contraction
    // (nvcc fuses a*b+c; MSVC may not) — ~1 ulp per op, accumulating to ~1e-6
    // relative. 1e-3 is ~1000x headroom while any indexing/layout/sampling bug
    // shifts costs at order 1, not 1e-6.
    fill_smooth_noise(epsx, epsy, K, /*iter=*/0, /*base_seed=*/42u, M);
    CUDA_CHECK(cudaMemcpy(d_theta_x, theta_x.data(), kN * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_theta_y, theta_y.data(), kN * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_epsx, epsx.data(), eps_count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_epsy, epsy.data(), eps_count * sizeof(float), cudaMemcpyHostToDevice));

    GpuTimer gt;
    gt.begin();
    launch_stomp_score(K, d_field, kGridW, kGridH, cell_m, sc.start, sc.goal,
                       d_theta_x, d_theta_y, d_epsx, d_epsy, d_Sloc, d_cost);
    const float gpu_ms0 = gt.end_ms();
    CUDA_CHECK(cudaMemcpy(cost.data(), d_cost, static_cast<size_t>(K) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Sloc.data(), d_Sloc, eps_count * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> cost_cpu(static_cast<size_t>(K));
    CpuTimer ct;
    ct.begin();
    stomp_rollouts_cpu(K, field.data(), kGridW, kGridH, cell_m, sc.start, sc.goal,
                       theta_x.data(), theta_y.data(), epsx.data(), epsy.data(), cost_cpu.data());
    const double cpu_ms = ct.end_ms();

    bool verify_pass = true;
    float worst = 0.0f;
    for (int k = 0; k < K; ++k) {
        const float scale = std::fabs(cost_cpu[k]) > 1.0f ? std::fabs(cost_cpu[k]) : 1.0f;
        const float d = std::fabs(cost[k] - cost_cpu[k]) / scale;
        if (d > worst) worst = d;
        if (d > 1e-3f) verify_pass = false;
    }
    std::printf("[info] verify: worst relative cost deviation %.3e over %d rollouts\n",
                static_cast<double>(worst), K);
    std::printf("[time] rollout set (K=%d, N=%d): CPU %.1f ms | GPU kernel %.3f ms | speed-up %.0fx (teaching artifact; kernel only)\n",
                K, kN, cpu_ms, static_cast<double>(gpu_ms0),
                cpu_ms / (static_cast<double>(gpu_ms0) > 0.0 ? static_cast<double>(gpu_ms0) : 1.0));
    std::printf("VERIFY: %s (GPU rollout costs match CPU reference within rel tol 1e-3)\n",
                verify_pass ? "PASS" : "FAIL");
    if (!verify_pass) {
        std::printf("RESULT: FAIL (GPU/CPU rollout disagreement — fix before trusting the planner)\n");
        return 1;
    }

    // ======================= OPTIMIZATION LOOP ===============================
    // We already scored iteration 0 above; apply its update, then continue.
    // Convergence: stop when the relative cost improvement stays below
    // kRelImprove for kPlateau consecutive iterations (or at kMaxIters).
    const double kRelImprove = 1e-3;   // "no meaningful improvement" threshold
    const int    kPlateau    = 5;      // consecutive such iterations => converged

    std::vector<double> wscratch(static_cast<size_t>(K));
    double loop_gpu_ms = static_cast<double>(gpu_ms0);
    double prev_cost = cost_init;
    int plateau_run = 0;
    int iters_run = 0;
    int converged_at = -1;

    for (int it = 0; it < kMaxIters; ++it) {
        // Iteration 0 already has its noise + scores from the verify stage;
        // later iterations draw fresh noise and re-score on the GPU.
        if (it > 0) {
            fill_smooth_noise(epsx, epsy, K, it, 42u, M);
            CUDA_CHECK(cudaMemcpy(d_theta_x, theta_x.data(), kN * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_theta_y, theta_y.data(), kN * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_epsx, epsx.data(), eps_count * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_epsy, epsy.data(), eps_count * sizeof(float), cudaMemcpyHostToDevice));

            GpuTimer gti;
            gti.begin();
            launch_stomp_score(K, d_field, kGridW, kGridH, cell_m, sc.start, sc.goal,
                               d_theta_x, d_theta_y, d_epsx, d_epsy, d_Sloc, d_cost);
            loop_gpu_ms += static_cast<double>(gti.end_ms());
            CUDA_CHECK(cudaMemcpy(Sloc.data(), d_Sloc, eps_count * sizeof(float), cudaMemcpyDeviceToHost));
        }

        // Per-waypoint softmin update, smoothed through M (STOMP's signature).
        stomp_update(theta_x, theta_y, Sloc, epsx, epsy, K, M, wscratch);

        // Convergence check on the NOMINAL trajectory cost (host-evaluated).
        assemble_path(sc, theta_x, theta_y, px, py);
        float maxf_cur = 0.0f;
        const double cur_cost = evaluate_path_cost(field.data(), kGridW, kGridH, cell_m,
                                                   px.data(), py.data(), kPathPoints, &maxf_cur);
        const double improve = (prev_cost > 0.0) ? (prev_cost - cur_cost) / prev_cost : 0.0;
        prev_cost = cur_cost;
        iters_run = it + 1;

        if (improve < kRelImprove) { if (++plateau_run >= kPlateau) { converged_at = iters_run; break; } }
        else plateau_run = 0;
    }

    // ---- final path + verdict -----------------------------------------------
    assemble_path(sc, theta_x, theta_y, px, py);
    float maxf_final = 0.0f;
    const double cost_final = evaluate_path_cost(field.data(), kGridW, kGridH, cell_m,
                                                 px.data(), py.data(), kPathPoints, &maxf_final);

    CUDA_CHECK(cudaFree(d_field));   CUDA_CHECK(cudaFree(d_theta_x)); CUDA_CHECK(cudaFree(d_theta_y));
    CUDA_CHECK(cudaFree(d_epsx));    CUDA_CHECK(cudaFree(d_epsy));
    CUDA_CHECK(cudaFree(d_Sloc));    CUDA_CHECK(cudaFree(d_cost));

    // The straight-line init drives through every obstacle, so its cost is
    // almost entirely COLLISION cost; the optimized path eliminates that,
    // leaving only a negligible smoothness residual. cost_final is therefore
    // ~0, which makes a raw init/final ratio an unstable astronomically-large
    // number — so we certify the reduction as a stable FRACTION (final as a
    // percentage of initial) and report both real numbers, not a fragile
    // factor. (Measured on this machine: 591.2 -> ~0.001, i.e. final is
    // ~0.0002% of initial — collision cost fully eliminated.)
    const double final_frac = (cost_init > 0.0) ? (cost_final / cost_init) : 1.0;
    if (converged_at > 0)
        std::printf("[info] plateau: converged at iteration %d (rel cost improvement < %.0e for %d consecutive iters)\n",
                    converged_at, kRelImprove, kPlateau);
    else
        std::printf("[info] plateau: ran the full %d iterations without an early plateau\n", iters_run);
    std::printf("[info] cost: initial %.2f, final %.4f (final is %.4f%% of initial — collision cost eliminated)\n",
                cost_init, cost_final, 100.0 * final_frac);
    std::printf("[info] max field value along path: initial %.1f -> final %.3f (0 = fully clear of all obstacle halos)\n",
                static_cast<double>(maxf_init), static_cast<double>(maxf_final));
    std::printf("[time] optimization loop: %.3f ms average GPU scoring kernel per iteration over %d iterations\n",
                loop_gpu_ms / (iters_run > 0 ? iters_run : 1), iters_run);

    // ---- artifacts ----------------------------------------------------------
    const std::string out_dir = project_root_from(argv[0]) + "/demo/out";
    bool artifact_ok = ensure_dir(out_dir);
    if (artifact_ok) {
        std::ofstream f(out_dir + "/trajectory.csv");
        artifact_ok = f.is_open();
        if (artifact_ok) {
            f << "idx,x_m,y_m\n";                                    // units in the header (§12)
            for (int s = 0; s < kPathPoints; ++s) f << s << ',' << px[s] << ',' << py[s] << '\n';
        }
    }
    if (artifact_ok)
        artifact_ok = write_costfield_pgm(out_dir + "/costfield.pgm", field, kGridW, kGridH, cell_m, px, py);
    if (artifact_ok)
        std::printf("ARTIFACT: wrote demo/out/trajectory.csv (%d points) and demo/out/costfield.pgm (%dx%d)\n",
                    kPathPoints, kGridW, kGridH);
    else
        std::printf("ARTIFACT: FAILED to write demo/out artifacts\n");

    // ---- verdict (the stable RESULT line) -----------------------------------
    // Two conditions, both with wide margins so platform low-bit differences
    // cannot flip the verdict (see the determinism note in the file header):
    //   (1) collision-free with margin: the max field value anywhere along the
    //       final path is below kCollisionThresh. The field is 0 in free space
    //       and rises to kCostCollision(=100) at an obstacle boundary; the
    //       threshold 25 corresponds to staying ~0.30 m clear of every
    //       obstacle (d such that 100*((0.6-d)/0.6)^2 = 25 => d = 0.30 m).
    //   (2) real improvement: the final total cost is under kMaxFinalFrac of
    //       the straight-line cost — a stable "at least 20x reduction" test
    //       that does not divide by the ~0 final cost.
    // Measured on this machine (RTX 2080 SUPER): max field along the final path
    // 0.000 (fully clear, i.e. >=0.6 m from every obstacle) and final cost
    // ~0.0002% of initial — both far inside the margins.
    const float  kCollisionThresh = 25.0f;   // field value; below it => >=0.30 m clear (see above)
    const double kMaxFinalFrac    = 0.05;    // final cost must be < 5% of initial (>=20x reduction)
    const bool success = artifact_ok &&
                         (maxf_final < kCollisionThresh) &&
                         (final_frac < kMaxFinalFrac);
    if (success)
        std::printf("RESULT: PASS (final trajectory collision-free with margin; total cost reduced vs straight-line init)\n");
    else
        std::printf("RESULT: FAIL (final trajectory hit the collision/reduction thresholds — see [info] lines)\n");
    return success ? 0 : 1;
}
