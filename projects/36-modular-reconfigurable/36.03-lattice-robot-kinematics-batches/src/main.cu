// ===========================================================================
// main.cu — entry point for project 36.03
//           Lattice-robot kinematics batches (sliding-cube model)
//
// What this program does, start to finish
// -----------------------------------------------------------------------
//   1. Load the committed SCENARIO (batch size K, RNG seed, corruption
//      fraction, vignette step budget) from data/sample/.
//   2. GENERATE the batch: K seeded-accretion connected configurations of
//      kM=24 modules each, then deterministically CORRUPT a documented
//      fraction with either a duplicate-position defect or a disconnection
//      defect — the negative controls the VERIFY gates below must catch.
//   3. Run the four GPU stage kernels (validity -> connectivity ->
//      articulation -> move enumeration) over the WHOLE batch.
//   4. VERIFY STAGE 1 (the repo's §5 gate): run the four CPU oracle twins
//      over the same batch and require BIT-EXACT integer agreement with
//      the GPU (this project's all-integer identity — no tolerance).
//   5. VERIFY STAGE 2 (corruption gate): every injected defect must be
//      caught, every clean configuration must pass, zero exceptions.
//   6. VERIFY STAGE 3 (brute-force cross-checks): on a subset, an
//      independently-coded "remove and recheck" articulation oracle and an
//      independently-coded move-precondition oracle must agree with the
//      fast algorithm exactly.
//   7. Write demo/out/batch_stats.csv (the batch statistics artifact).
//   8. THE VIGNETTE: take ONE 24-module straight-line configuration and
//      greedily execute legal, connectivity-preserving moves that reduce a
//      fixed-centroid compactness potential — verifying every intermediate
//      state stays valid+connected — logging demo/out/vignette_frames.csv.
//   9. Render demo/out/config_render.pgm (a 2D projection of one batch
//      configuration) and print the final RESULT line.
//
// Why this is NOT a reconfiguration PLANNER: the vignette's move choice is
// a GREEDY, single-step-lookahead heuristic (steepest descent on the
// compactness potential) — not a search, not guaranteed optimal, and it
// can get stuck in a local optimum before reaching a fully compact shape.
// A real planner (project 36.01, named throughout this project's docs) is
// the documented research step this teaching core deliberately does not
// attempt (CLAUDE.md §2/§13 [R&D] scoping).
//
// Output contract (load-bearing!): demo/run_demo.ps1/.sh diff the STABLE
// lines below against demo/expected_output.txt — "[demo]"/"PROBLEM:"/
// "SCENARIO:"/"VERIFY:"/"CORRUPTION-GATE:"/"ARTICULATION-BRUTEFORCE:"/
// "MOVE-PRECONDITION-BRUTEFORCE:"/"ARTIFACT:"/"VIGNETTE:"/"RESULT:" lines.
// "[info]"/"[time]" lines are NOT checked (machine/run dependent). Change a
// stable line here -> update demo/expected_output.txt in the same change.
//
// Read this first, then kernels.cuh -> reference_cpu.cpp -> kernels.cu.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
// Demo-level constants (batch-generation and vignette parameters — NOT part
// of the kernels.cuh contract because they describe THIS DEMO's problem
// instance, not the lattice geometry every kernel must agree on).
// ---------------------------------------------------------------------------
static const float   kDefaultCorruptFrac = 0.10f;   // fraction of K injected as negative controls
static const int     kArticulationSubset = 128;      // configs cross-checked against the brute-force oracle
static const int     kMoveSubset         = 128;      // configs cross-checked for move preconditions
static const int     kVignetteMaxSteps   = 600;      // greedy step budget (a cap, not a target — see §8)
static const int32_t kDisconnectOffset   = 100000;   // translation (lattice cells) for the disconnect corruption

static const int kLabelClean      = 0;
static const int kLabelDuplicate  = 1;
static const int kLabelDisconnect = 2;

// ---------------------------------------------------------------------------
// Local geometry helpers — a THIRD deliberate duplication of slide_delta /
// corner_axes (kernels.cu carries the __device__ copy, reference_cpu.cpp
// the CPU-oracle copy). main.cu is neither: it is the DATA GENERATOR and
// DEMO DRIVER (matching 08.01's fill_noise/gaussian, which are main.cu-
// local for the same reason — problem SETUP is not part of the algorithm
// under test, so it earns its own small, honestly-duplicated copy rather
// than reaching into kernels.cu/reference_cpu.cpp's internals).
// ---------------------------------------------------------------------------
static void slide_delta_host(int dir, int32_t& dx, int32_t& dy, int32_t& dz)
{
    const int axis = dir >> 1;
    const int sign = (dir & 1) ? -1 : 1;
    dx = (axis == 0) ? sign : 0;
    dy = (axis == 1) ? sign : 0;
    dz = (axis == 2) ? sign : 0;
}

static void corner_axes_host(int c, int& e_dir, int& f_dir)
{
    const int pair = c / 4;
    const int combo = c % 4;
    const int e_sign = (combo < 2) ? 0 : 1;
    const int f_sign = (combo % 2 == 0) ? 0 : 1;
    if (pair == 0)      { e_dir = 0 + e_sign; f_dir = 2 + f_sign; }
    else if (pair == 1) { e_dir = 0 + e_sign; f_dir = 4 + f_sign; }
    else                { e_dir = 2 + e_sign; f_dir = 4 + f_sign; }
}

// move_destination — where module A lands under move direction `dir`
// (0..17, the kernels.cuh numbering). Used only by the vignette (§8): the
// batch stages above never need to MATERIALIZE a destination, only test
// its legality, so this function has no counterpart in kernels.cu.
static void move_destination(const int32_t A[3], int dir, int32_t B[3])
{
    if (dir < kNumSlideDirs) {
        int32_t dx, dy, dz;
        slide_delta_host(dir, dx, dy, dz);
        B[0] = A[0] + dx; B[1] = A[1] + dy; B[2] = A[2] + dz;
    } else {
        int e_dir, f_dir;
        corner_axes_host(dir - kNumSlideDirs, e_dir, f_dir);
        int32_t edx, edy, edz, fdx, fdy, fdz;
        slide_delta_host(e_dir, edx, edy, edz);
        slide_delta_host(f_dir, fdx, fdy, fdz);
        B[0] = A[0] + edx + fdx; B[1] = A[1] + edy + fdy; B[2] = A[2] + edz + fdz;
    }
}

// ---------------------------------------------------------------------------
// xorshift32 — the repo's portable deterministic PRNG (byte-identical
// sequence on every platform given the same seed — see 08.01's identical
// generator). Used ONLY for batch generation (which modules attach where,
// which configs get corrupted how); it never touches the lattice math
// under test, so its output feeding into "PASS" is a data-generation
// choice, not part of the thing being verified.
// ---------------------------------------------------------------------------
static inline uint32_t xorshift32(uint32_t& state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

// rand_below — uniform-ish integer in [0, n). Simple modulo reduction (a
// small, DOCUMENTED bias for n not dividing 2^32 evenly — irrelevant here:
// n is always a tiny bound like kM or kNumSlideDirs, and this value only
// steers WHERE a module attaches during synthetic generation, never a
// verified numeric result).
static inline uint32_t rand_below(uint32_t& state, uint32_t n)
{
    return xorshift32(state) % n;
}

// ---------------------------------------------------------------------------
// generate_config — ONE seeded-accretion connected configuration of kM
// modules (README/THEORY "The algorithm"): start module 0 at the origin;
// each subsequent module attaches to a uniformly random ALREADY-PLACED
// module's uniformly random free face-neighbor cell. Connectivity holds BY
// CONSTRUCTION (every module attaches to something already in the graph),
// so this generator is the reason the batch's baseline validity/
// connectivity rate is exactly 100% before corruption (README "Data").
//
// Bounded random retries (64 attempts) handle the common case cheaply; the
// deterministic fallback scan (which tries every placed module's every
// direction in a fixed order) GUARANTEES termination — a connected set of
// fewer than kM cells on an infinite lattice always has at least one free
// neighbor, so the fallback can never fail. Both paths use the SAME
// continuing RNG stream `s`, so the whole configuration (and everything
// derived from it) is a pure function of the input seed.
// ---------------------------------------------------------------------------
static void generate_config(uint32_t seed, int32_t* pos_out /* [kM*3] */)
{
    uint32_t s = seed ? seed : 1u;   // xorshift32 must never see a zero state

    int32_t cx[kM], cy[kM], cz[kM];
    cx[0] = cy[0] = cz[0] = 0;       // module 0 always anchors the origin
    int placed = 1;

    while (placed < kM) {
        bool ok = false;

        // ---- fast path: random parent + random direction, bounded retries
        for (int attempt = 0; attempt < 64 && !ok; ++attempt) {
            const int parent = static_cast<int>(rand_below(s, static_cast<uint32_t>(placed)));
            const int dir = static_cast<int>(rand_below(s, static_cast<uint32_t>(kNumSlideDirs)));
            int32_t dx, dy, dz;
            slide_delta_host(dir, dx, dy, dz);
            const int32_t nx = cx[parent] + dx, ny = cy[parent] + dy, nz = cz[parent] + dz;

            bool occ = false;
            for (int i = 0; i < placed; ++i)
                if (cx[i] == nx && cy[i] == ny && cz[i] == nz) { occ = true; break; }
            if (!occ) { cx[placed] = nx; cy[placed] = ny; cz[placed] = nz; ++placed; ok = true; }
        }

        // ---- guaranteed fallback: deterministic scan, first free cell wins
        if (!ok) {
            for (int i = 0; i < placed && !ok; ++i) {
                for (int dir = 0; dir < kNumSlideDirs && !ok; ++dir) {
                    int32_t dx, dy, dz;
                    slide_delta_host(dir, dx, dy, dz);
                    const int32_t nx = cx[i] + dx, ny = cy[i] + dy, nz = cz[i] + dz;
                    bool occ = false;
                    for (int j = 0; j < placed; ++j)
                        if (cx[j] == nx && cy[j] == ny && cz[j] == nz) { occ = true; break; }
                    if (!occ) { cx[placed] = nx; cy[placed] = ny; cz[placed] = nz; ++placed; ok = true; }
                }
            }
            if (!ok) {
                // Cannot happen (see header comment) — a loud, honest abort
                // rather than a silently wrong batch if this invariant ever breaks.
                std::fprintf(stderr, "generate_config: accretion stalled at placed=%d (should be impossible)\n", placed);
                std::exit(EXIT_FAILURE);
            }
        }
    }

    for (int m = 0; m < kM; ++m) {
        pos_out[m * 3 + 0] = cx[m];
        pos_out[m * 3 + 1] = cy[m];
        pos_out[m * 3 + 2] = cz[m];
    }
}

// corrupt_duplicate — overwrite the LAST accreted module's cell with a
// copy of a random EARLIER module's cell. Module (kM-1) was, by
// construction, the most recently attached leaf of the accretion process,
// so removing/relocating it can never disconnect modules 0..kM-2 — this
// corruption is therefore CLEAN: it breaks validity only (README "Data").
static void corrupt_duplicate(uint32_t seed, int32_t* pos /* [kM*3] */)
{
    uint32_t s = seed ? seed : 1u;
    const int j = static_cast<int>(rand_below(s, static_cast<uint32_t>(kM - 1)));   // any module != kM-1
    pos[(kM - 1) * 3 + 0] = pos[j * 3 + 0];
    pos[(kM - 1) * 3 + 1] = pos[j * 3 + 1];
    pos[(kM - 1) * 3 + 2] = pos[j * 3 + 2];
}

// corrupt_disconnect — split the kM modules at a random accretion-order
// cut point and translate the later group by a huge, fixed offset in x.
// The translated group's internal relative positions (hence its internal
// validity) are untouched, and kDisconnectOffset=100000 lattice cells is
// far beyond any accretion cluster's extent — so this corruption is also
// CLEAN: it breaks connectivity only, never validity (README "Data").
static void corrupt_disconnect(uint32_t seed, int32_t* pos /* [kM*3] */)
{
    uint32_t s = seed ? seed : 1u;
    const int cut = 1 + static_cast<int>(rand_below(s, static_cast<uint32_t>(kM - 1)));   // in [1, kM-1]
    for (int m = cut; m < kM; ++m) pos[m * 3 + 0] += kDisconnectOffset;
}

// ---------------------------------------------------------------------------
// Scenario loading (the committed "task definition" — generator params and
// vignette budget, NOT the configurations themselves; see data/README.md
// for why the sample is parameters, not a multi-megabyte position dump —
// the same "scenario, not recordings" choice 08.01 makes).
// ---------------------------------------------------------------------------
struct Scenario {
    int K = kDefaultK;
    uint32_t seed = 42u;
    float corrupt_frac = kDefaultCorruptFrac;
    int vignette_max_steps = kVignetteMaxSteps;
    bool loaded = false;
};

static Scenario load_scenario(const std::string& path)
{
    Scenario sc;
    std::ifstream in(path);
    if (!in.is_open()) return sc;

    bool have_k = false, have_seed = false;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string label, cell;
        std::getline(ss, label, ',');
        if (!std::getline(ss, cell, ',')) { std::fprintf(stderr, "scenario: short row '%s'\n", label.c_str()); return Scenario{}; }
        if (label == "K") { sc.K = std::atoi(cell.c_str()); have_k = true; }
        else if (label == "SEED") { sc.seed = static_cast<uint32_t>(std::strtoul(cell.c_str(), nullptr, 10)); have_seed = true; }
        else if (label == "CORRUPT_FRAC") { sc.corrupt_frac = std::strtof(cell.c_str(), nullptr); }
        else if (label == "VIGNETTE_MAX_STEPS") { sc.vignette_max_steps = std::atoi(cell.c_str()); }
        else { std::fprintf(stderr, "scenario: unknown row label '%s'\n", label.c_str()); return Scenario{}; }
    }
    if (!have_k || !have_seed || sc.K < 8) {
        std::fprintf(stderr, "scenario: missing K or SEED (or K too small)\n");
        return Scenario{};
    }
    sc.loaded = true;
    return sc;
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
    candidates.push_back(project_root_from(argv0) + "/data/sample/lattice_scenario.csv");
    candidates.push_back("data/sample/lattice_scenario.csv");
    candidates.push_back("../data/sample/lattice_scenario.csv");
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
// write_batch_stats_csv — the batch artifact (README "Expected output"):
// one row per configuration, the numbers a learner would plot a histogram
// of (module/move/articulation counts) — see demo/README.md.
// ---------------------------------------------------------------------------
static bool write_batch_stats_csv(const std::string& path, int K,
                                  const std::vector<int>& label,
                                  const std::vector<uint8_t>& valid,
                                  const std::vector<uint8_t>& connected,
                                  const std::vector<int32_t>& num_artic,
                                  const std::vector<int32_t>& move_count)
{
    std::ofstream f(path);
    if (!f.is_open()) return false;
    f << "config_id,label,valid,connected,num_articulation,num_legal_moves\n";
    static const char* kLabelName[3] = { "clean", "duplicate", "disconnect" };
    for (int k = 0; k < K; ++k) {
        f << k << ',' << kLabelName[label[static_cast<size_t>(k)]] << ','
          << static_cast<int>(valid[static_cast<size_t>(k)]) << ','
          << static_cast<int>(connected[static_cast<size_t>(k)]) << ','
          << num_artic[static_cast<size_t>(k)] << ','
          << move_count[static_cast<size_t>(k)] << '\n';
    }
    return true;
}

// ---------------------------------------------------------------------------
// write_pgm — a 2D top-down (XY, z ignored) projection of ONE configuration
// as an ASCII PGM (P2): text-editable, zero dependencies, the simplest
// image format that teaches the format itself while it's at it. Each
// occupied (x,y) COLUMN (any z) paints one scale x scale white block;
// row 0 of the image is the MAXIMUM y (standard "y increases upward on
// screen, decreases going down the file" convention) — documented here and
// in demo/README.md so a learner opening the file in a text editor and a
// learner opening it in an image viewer see the same orientation.
// ---------------------------------------------------------------------------
static bool write_pgm(const std::string& path, const int32_t* pos_one_config)
{
    int32_t minx = pos_one_config[0], maxx = pos_one_config[0];
    int32_t miny = pos_one_config[1], maxy = pos_one_config[1];
    for (int m = 0; m < kM; ++m) {
        const int32_t x = pos_one_config[m * 3 + 0], y = pos_one_config[m * 3 + 1];
        if (x < minx) minx = x; if (x > maxx) maxx = x;
        if (y < miny) miny = y; if (y > maxy) maxy = y;
    }
    const int margin = 1;                                   // cells of black border
    const int scale = 6;                                     // pixels per lattice cell
    const int w_cells = (maxx - minx + 1) + 2 * margin;
    const int h_cells = (maxy - miny + 1) + 2 * margin;
    const int W = w_cells * scale, H = h_cells * scale;

    std::vector<uint8_t> img(static_cast<size_t>(W) * static_cast<size_t>(H), 0);  // 0 = black background
    for (int m = 0; m < kM; ++m) {
        const int32_t x = pos_one_config[m * 3 + 0], y = pos_one_config[m * 3 + 1];
        const int cellx = static_cast<int>(x - minx) + margin;
        const int celly_from_bottom = static_cast<int>(y - miny) + margin;   // y grows "up" in lattice space
        const int celly = (h_cells - 1) - celly_from_bottom;                 // flip: row 0 = max y (see header)
        for (int py = 0; py < scale; ++py)
            for (int px = 0; px < scale; ++px)
                img[static_cast<size_t>(celly * scale + py) * static_cast<size_t>(W)
                    + static_cast<size_t>(cellx * scale + px)] = 255;
    }

    std::ofstream f(path);
    if (!f.is_open()) return false;
    f << "P2\n" << W << ' ' << H << "\n255\n";
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) f << static_cast<int>(img[static_cast<size_t>(y) * W + x]) << ' ';
        f << '\n';
    }
    return true;
}

// ---------------------------------------------------------------------------
// write_vignette_csv — long-format positions per step, the animation
// payload (README "Expected output" / demo/README.md).
// ---------------------------------------------------------------------------
static bool write_vignette_csv(const std::string& path,
                               const std::vector<std::vector<int32_t>>& frames /* [step][kM*3] */)
{
    std::ofstream f(path);
    if (!f.is_open()) return false;
    f << "step,module,x,y,z\n";
    for (size_t s = 0; s < frames.size(); ++s) {
        const std::vector<int32_t>& fr = frames[s];
        for (int m = 0; m < kM; ++m)
            f << s << ',' << m << ',' << fr[m * 3 + 0] << ',' << fr[m * 3 + 1] << ',' << fr[m * 3 + 2] << '\n';
    }
    return true;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    int K = -1;                 // -1 = "use the scenario's K" (CLI override changes the PROBLEM line)
    std::string data_path;
    for (int i = 1; i < argc; ++i) {
        if      (!std::strcmp(argv[i], "--configs") && i + 1 < argc) K = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--data")    && i + 1 < argc) data_path = argv[++i];
        else {
            std::fprintf(stderr,
                "usage: %s [--configs K] [--data lattice_scenario.csv]\n"
                "note: non-default K changes the PROBLEM/SCENARIO lines; the demo diff will flag it.\n",
                argv[0]);
            return 2;
        }
    }

    std::printf("[demo] Lattice-robot kinematics batches (project 36.03): sliding-cube model batch pipeline\n");
    print_device_info();

    // ---- 1) scenario ---------------------------------------------------------
    const std::string scenario_path = find_scenario(data_path, argv[0]);
    if (scenario_path.empty()) {
        std::printf("SCENARIO: NOT FOUND — data/sample/lattice_scenario.csv missing (run scripts/make_synthetic.py?)\n");
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
    if (K < 8) K = sc.K;   // apply CLI override only if a sane value was given

    const int corrupt_count = static_cast<int>(static_cast<float>(K) * sc.corrupt_frac + 0.5f);
    const int dup_count = corrupt_count / 2;
    const int disc_count = corrupt_count - dup_count;
    const int clean_count = K - corrupt_count;

    std::printf("PROBLEM: K=%d configurations x M=%d modules, sliding-cube lattice model, all-integer\n", K, kM);
    std::printf("SCENARIO: seeded accretion (seed=%u) [synthetic]; corrupt_frac=%.2f -> %d corrupted (%d duplicate + %d disconnect), %d clean\n",
                sc.seed, static_cast<double>(sc.corrupt_frac), corrupt_count, dup_count, disc_count, clean_count);

    // ---- 2) generate the batch -------------------------------------------------
    std::vector<int32_t> h_pos(static_cast<size_t>(K) * kM * 3);
    std::vector<int> h_label(static_cast<size_t>(K), kLabelClean);

    CpuTimer gen_timer;
    gen_timer.begin();
    for (int k = 0; k < K; ++k) {
        const uint32_t cfg_seed = sc.seed + 1000003u * static_cast<uint32_t>(k + 1);   // 08.01's per-item seed mixing
        int32_t* p = &h_pos[static_cast<size_t>(k) * kM * 3];
        generate_config(cfg_seed, p);

        // The LAST clean_count..K-1 indices are corrupted (see the header
        // comment): duplicate-corrupted first, disconnect-corrupted after —
        // both draw from a seed derived from the SAME per-config stream so
        // the whole batch remains a pure function of sc.seed.
        if (k >= clean_count && k < clean_count + dup_count) {
            corrupt_duplicate(cfg_seed ^ 0x9E3779B9u, p);
            h_label[static_cast<size_t>(k)] = kLabelDuplicate;
        } else if (k >= clean_count + dup_count) {
            corrupt_disconnect(cfg_seed ^ 0x9E3779B9u, p);
            h_label[static_cast<size_t>(k)] = kLabelDisconnect;
        }
    }
    const double gen_ms = gen_timer.end_ms();
    std::printf("[time] batch generation (seeded accretion + corruption): %.2f ms for K=%d\n", gen_ms, K);

    // ---- 3) GPU pipeline: four stages, one thread per configuration each ------
    int32_t* d_pos = nullptr;
    uint8_t* d_valid = nullptr;
    uint8_t* d_connected = nullptr;
    uint8_t* d_is_artic = nullptr;
    int32_t* d_num_artic = nullptr;
    uint8_t* d_legal_move = nullptr;
    int32_t* d_move_count = nullptr;

    const size_t pos_count = static_cast<size_t>(K) * kM * 3;
    const size_t artic_count = static_cast<size_t>(K) * kM;
    const size_t move_count_total = static_cast<size_t>(K) * kM * kNumMoveDirs;

    CUDA_CHECK(cudaMalloc(&d_pos, pos_count * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_valid, static_cast<size_t>(K) * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_connected, static_cast<size_t>(K) * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_is_artic, artic_count * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_num_artic, static_cast<size_t>(K) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_legal_move, move_count_total * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_move_count, static_cast<size_t>(K) * sizeof(int32_t)));

    CUDA_CHECK(cudaMemcpy(d_pos, h_pos.data(), pos_count * sizeof(int32_t), cudaMemcpyHostToDevice));

    GpuTimer gpu_timer;
    gpu_timer.begin();
    launch_validity(K, d_pos, d_valid);
    launch_connectivity(K, d_pos, d_valid, d_connected);
    launch_articulation(K, d_pos, d_valid, d_connected, d_is_artic, d_num_artic);
    launch_move_enum(K, d_pos, d_valid, d_connected, d_is_artic, d_legal_move, d_move_count);
    const float gpu_ms = gpu_timer.end_ms();

    std::vector<uint8_t> h_valid(static_cast<size_t>(K)), h_connected(static_cast<size_t>(K));
    std::vector<uint8_t> h_is_artic(artic_count);
    std::vector<int32_t> h_num_artic(static_cast<size_t>(K));
    std::vector<uint8_t> h_legal_move(move_count_total);
    std::vector<int32_t> h_move_count(static_cast<size_t>(K));

    CUDA_CHECK(cudaMemcpy(h_valid.data(), d_valid, static_cast<size_t>(K) * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_connected.data(), d_connected, static_cast<size_t>(K) * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_is_artic.data(), d_is_artic, artic_count * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_num_artic.data(), d_num_artic, static_cast<size_t>(K) * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_legal_move.data(), d_legal_move, move_count_total * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_move_count.data(), d_move_count, static_cast<size_t>(K) * sizeof(int32_t), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_pos));
    CUDA_CHECK(cudaFree(d_valid));
    CUDA_CHECK(cudaFree(d_connected));
    CUDA_CHECK(cudaFree(d_is_artic));
    CUDA_CHECK(cudaFree(d_num_artic));
    CUDA_CHECK(cudaFree(d_legal_move));
    CUDA_CHECK(cudaFree(d_move_count));

    // ---- 4) VERIFY STAGE 1: CPU oracle twins, BIT-EXACT agreement -------------
    std::vector<uint8_t> c_valid(static_cast<size_t>(K)), c_connected(static_cast<size_t>(K));
    std::vector<uint8_t> c_is_artic(artic_count);
    std::vector<int32_t> c_num_artic(static_cast<size_t>(K));
    std::vector<uint8_t> c_legal_move(move_count_total);
    std::vector<int32_t> c_move_count(static_cast<size_t>(K));

    CpuTimer cpu_timer;
    cpu_timer.begin();
    validity_cpu(K, h_pos.data(), c_valid.data());
    connectivity_cpu(K, h_pos.data(), c_valid.data(), c_connected.data());
    articulation_cpu(K, h_pos.data(), c_valid.data(), c_connected.data(), c_is_artic.data(), c_num_artic.data());
    move_enum_cpu(K, h_pos.data(), c_valid.data(), c_connected.data(), c_is_artic.data(),
                  c_legal_move.data(), c_move_count.data());
    const double cpu_ms = cpu_timer.end_ms();

    std::printf("[time] GPU 4-stage pipeline: %.3f ms | CPU reference (all 4 stages): %.1f ms | speed-up %.0fx (teaching artifact)\n",
                static_cast<double>(gpu_ms), cpu_ms, cpu_ms / (static_cast<double>(gpu_ms) > 0.0 ? static_cast<double>(gpu_ms) : 1.0));

    bool verify_pass = true;
    int mismatch_valid = 0, mismatch_connected = 0, mismatch_artic = 0, mismatch_num_artic = 0;
    int mismatch_move = 0, mismatch_move_count = 0;
    for (int k = 0; k < K; ++k) {
        if (h_valid[static_cast<size_t>(k)] != c_valid[static_cast<size_t>(k)]) ++mismatch_valid;
        if (h_connected[static_cast<size_t>(k)] != c_connected[static_cast<size_t>(k)]) ++mismatch_connected;
        if (h_num_artic[static_cast<size_t>(k)] != c_num_artic[static_cast<size_t>(k)]) ++mismatch_num_artic;
        if (h_move_count[static_cast<size_t>(k)] != c_move_count[static_cast<size_t>(k)]) ++mismatch_move_count;
        for (int m = 0; m < kM; ++m)
            if (h_is_artic[static_cast<size_t>(k) * kM + m] != c_is_artic[static_cast<size_t>(k) * kM + m]) ++mismatch_artic;
        for (int i = 0; i < kM * kNumMoveDirs; ++i)
            if (h_legal_move[static_cast<size_t>(k) * kM * kNumMoveDirs + static_cast<size_t>(i)]
                != c_legal_move[static_cast<size_t>(k) * kM * kNumMoveDirs + static_cast<size_t>(i)]) ++mismatch_move;
    }
    const int total_mismatch = mismatch_valid + mismatch_connected + mismatch_artic
                              + mismatch_num_artic + mismatch_move + mismatch_move_count;
    verify_pass = (total_mismatch == 0);
    std::printf("[info] GPU-vs-CPU mismatches: valid=%d connected=%d is_articulation=%d num_articulation=%d legal_move=%d move_count=%d\n",
                mismatch_valid, mismatch_connected, mismatch_artic, mismatch_num_artic, mismatch_move, mismatch_move_count);
    std::printf("VERIFY: %s (GPU matches CPU reference BIT-EXACT over all 4 stages, K=%d configurations, all-integer)\n",
                verify_pass ? "PASS" : "FAIL", K);
    if (!verify_pass) {
        std::printf("RESULT: FAIL (GPU/CPU disagreement — fix before trusting any gate below)\n");
        return 1;
    }

    // ---- 5) VERIFY STAGE 2: corruption gate ------------------------------------
    int caught = 0, missed = 0, false_alarms = 0;
    for (int k = 0; k < K; ++k) {
        const bool ok = (c_valid[static_cast<size_t>(k)] == 1 && c_connected[static_cast<size_t>(k)] == 1);
        const int lbl = h_label[static_cast<size_t>(k)];
        if (lbl == kLabelClean) {
            if (!ok) ++false_alarms;
        } else if (lbl == kLabelDuplicate) {
            if (c_valid[static_cast<size_t>(k)] == 0) ++caught; else ++missed;
        } else { // disconnect
            if (c_valid[static_cast<size_t>(k)] == 1 && c_connected[static_cast<size_t>(k)] == 0) ++caught; else ++missed;
        }
    }
    const bool corruption_pass = (missed == 0 && false_alarms == 0);
    std::printf("CORRUPTION-GATE: %s (%d/%d injected configs caught, %d/%d false alarms on clean configs)\n",
                corruption_pass ? "PASS" : "FAIL", caught, corrupt_count, false_alarms, clean_count);
    if (!corruption_pass) {
        std::printf("RESULT: FAIL (corruption gate failed — see [info] mismatch counts above)\n");
        return 1;
    }

    // ---- 6) VERIFY STAGE 3: brute-force cross-checks on a subset ---------------
    // The subset is the FIRST clean_count indices, which are guaranteed
    // clean by construction (§5's split), so a subset of them is always
    // valid+connected — exactly what these two oracles assume.
    const int artic_subset = (kArticulationSubset < clean_count) ? kArticulationSubset : clean_count;
    int artic_bf_mismatch = 0;
    for (int k = 0; k < artic_subset; ++k) {
        uint8_t bf[kM];
        articulation_bruteforce_cpu(&h_pos[static_cast<size_t>(k) * kM * 3], bf);
        for (int m = 0; m < kM; ++m)
            if (bf[m] != c_is_artic[static_cast<size_t>(k) * kM + m]) ++artic_bf_mismatch;
    }
    const bool artic_bf_pass = (artic_bf_mismatch == 0);
    std::printf("ARTICULATION-BRUTEFORCE: %s (%d/%d subset configs cross-checked against the remove-and-recheck oracle, %d module mismatches)\n",
                artic_bf_pass ? "PASS" : "FAIL", artic_subset, artic_subset, artic_bf_mismatch);

    const int move_subset = (kMoveSubset < clean_count) ? kMoveSubset : clean_count;
    int move_bf_mismatch = 0;
    for (int k = 0; k < move_subset; ++k) {
        uint8_t bf[kM * kNumMoveDirs];
        move_precondition_bruteforce_cpu(&h_pos[static_cast<size_t>(k) * kM * 3],
                                         &c_is_artic[static_cast<size_t>(k) * kM], bf);
        for (int i = 0; i < kM * kNumMoveDirs; ++i)
            if (bf[i] != c_legal_move[static_cast<size_t>(k) * kM * kNumMoveDirs + static_cast<size_t>(i)]) ++move_bf_mismatch;
    }
    const bool move_bf_pass = (move_bf_mismatch == 0);
    std::printf("MOVE-PRECONDITION-BRUTEFORCE: %s (%d/%d subset configs cross-checked against the independent oracle, %d entry mismatches out of %d)\n",
                move_bf_pass ? "PASS" : "FAIL", move_subset, move_subset, move_bf_mismatch, move_subset * kM * kNumMoveDirs);

    if (!artic_bf_pass || !move_bf_pass) {
        std::printf("RESULT: FAIL (a brute-force cross-check disagreed with the fast algorithm)\n");
        return 1;
    }

    // ---- 7) artifact: batch_stats.csv ------------------------------------------
    const std::string out_dir = project_root_from(argv[0]) + "/demo/out";
    bool artifact_ok = ensure_dir(out_dir);
    if (artifact_ok)
        artifact_ok = write_batch_stats_csv(out_dir + "/batch_stats.csv", K, h_label,
                                            c_valid, c_connected, c_num_artic, c_move_count);
    if (artifact_ok)
        std::printf("ARTIFACT: wrote demo/out/batch_stats.csv (%d rows)\n", K);
    else {
        std::printf("ARTIFACT: FAILED to write demo/out/batch_stats.csv\n");
        std::printf("RESULT: FAIL (artifact write failed)\n");
        return 1;
    }

    // ============================ 8) THE VIGNETTE ===============================
    // ONE 24-module straight line, greedily compacted toward a FIXED integer
    // reference point (the start configuration's centroid, truncated to
    // integer) via steepest-descent on Phi = sum of squared distances to
    // that point — an all-integer potential, kept in lockstep with this
    // project's all-integer identity (no float enters the pass/fail path).
    // Every candidate move is required to (a) satisfy the Stage 4
    // precondition for a NON-articulation module, AND (b) leave the WHOLE
    // kM-module robot valid+connected afterward — (b) is NOT implied by (a)
    // alone (README/THEORY explain why: a mechanically legal move can still
    // strand the rest of the robot if the destination happens to touch
    // nothing else), so the vignette re-verifies it explicitly, every step.
    // ------------------------------------------------------------------------
    std::vector<int32_t> vig(kM * 3);
    for (int m = 0; m < kM; ++m) { vig[static_cast<size_t>(m) * 3 + 0] = m; vig[static_cast<size_t>(m) * 3 + 1] = 0; vig[static_cast<size_t>(m) * 3 + 2] = 0; }

    int64_t cx_sum = 0, cy_sum = 0, cz_sum = 0;
    for (int m = 0; m < kM; ++m) { cx_sum += vig[static_cast<size_t>(m) * 3 + 0]; cy_sum += vig[static_cast<size_t>(m) * 3 + 1]; cz_sum += vig[static_cast<size_t>(m) * 3 + 2]; }
    const int32_t Cx = static_cast<int32_t>(cx_sum / kM), Cy = static_cast<int32_t>(cy_sum / kM), Cz = static_cast<int32_t>(cz_sum / kM);

    auto potential = [&](const int32_t* pos) -> int64_t {
        int64_t phi = 0;
        for (int m = 0; m < kM; ++m) {
            const int64_t dx = pos[m * 3 + 0] - Cx, dy = pos[m * 3 + 1] - Cy, dz = pos[m * 3 + 2] - Cz;
            phi += dx * dx + dy * dy + dz * dz;
        }
        return phi;
    };

    std::vector<std::vector<int32_t>> frames;
    frames.push_back(vig);   // step 0: the starting line

    const int64_t phi_start = potential(vig.data());
    int64_t phi_cur = phi_start;
    int step = 0;
    bool stuck = false;
    int slide_moves = 0, corner_moves = 0;   // tallied from best_dir below — which move FAMILY the greedy picked each step

    for (step = 1; step <= sc.vignette_max_steps; ++step) {
        uint8_t v1, c1, artic1[kM];
        int32_t na1;
        validity_cpu(1, vig.data(), &v1);
        connectivity_cpu(1, vig.data(), &v1, &c1);
        articulation_cpu(1, vig.data(), &v1, &c1, artic1, &na1);
        if (v1 != 1 || c1 != 1) {
            std::fprintf(stderr, "vignette: state invalid/disconnected entering step %d — internal bug\n", step);
            std::exit(EXIT_FAILURE);
        }
        uint8_t legal1[kM * kNumMoveDirs];
        int32_t mc1;
        move_enum_cpu(1, vig.data(), &v1, &c1, artic1, legal1, &mc1);

        int best_m = -1, best_dir = -1;
        int64_t best_phi = phi_cur;   // require STRICT improvement
        int32_t best_dest[3] = { 0, 0, 0 };

        for (int m = 0; m < kM; ++m) {
            for (int dir = 0; dir < kNumMoveDirs; ++dir) {
                if (!legal1[m * kNumMoveDirs + dir]) continue;
                int32_t dest[3];
                move_destination(&vig[static_cast<size_t>(m) * 3], dir, dest);

                std::vector<int32_t> cand = vig;
                cand[static_cast<size_t>(m) * 3 + 0] = dest[0];
                cand[static_cast<size_t>(m) * 3 + 1] = dest[1];
                cand[static_cast<size_t>(m) * 3 + 2] = dest[2];

                // Re-verify the WHOLE robot, not just this module's move —
                // the honesty check the header comment promises.
                uint8_t cv, cc;
                validity_cpu(1, cand.data(), &cv);
                connectivity_cpu(1, cand.data(), &cv, &cc);
                if (cv != 1 || cc != 1) continue;

                const int64_t cand_phi = potential(cand.data());
                if (cand_phi < best_phi) {
                    best_phi = cand_phi; best_m = m; best_dir = dir;
                    best_dest[0] = dest[0]; best_dest[1] = dest[1]; best_dest[2] = dest[2];
                }
            }
        }

        if (best_m < 0) { stuck = true; break; }   // local optimum: no improving, legal, connectivity-preserving move

        // best_dir < kNumSlideDirs (6) means the greedy's winning candidate
        // was a SLIDE; otherwise it was a CORNER pivot — a cheap, honest use
        // of the direction index beyond just building best_dest above (the
        // per-family tally prints in the [info] line below).
        if (best_dir < kNumSlideDirs) ++slide_moves; else ++corner_moves;

        vig[static_cast<size_t>(best_m) * 3 + 0] = best_dest[0];
        vig[static_cast<size_t>(best_m) * 3 + 1] = best_dest[1];
        vig[static_cast<size_t>(best_m) * 3 + 2] = best_dest[2];
        phi_cur = best_phi;
        frames.push_back(vig);
    }
    const int steps_taken = static_cast<int>(frames.size()) - 1;

    std::printf("[info] vignette: start=straight line of %d modules, fixed reference (%d,%d,%d), Phi %lld -> %lld over %d steps (%s)\n",
                kM, Cx, Cy, Cz, static_cast<long long>(phi_start), static_cast<long long>(phi_cur), steps_taken,
                stuck ? "converged to a local optimum" : "step budget reached");
    std::printf("[info] vignette: move mix chosen by the greedy: %d slide + %d corner = %d total\n",
                slide_moves, corner_moves, slide_moves + corner_moves);

    // Bounding-box shrink is the human-readable "line -> compact blob"
    // story: report it, but the CHECKED criterion is the integer potential
    // decrease plus the per-step valid+connected re-verification above.
    int32_t minx = vig[0], maxx = vig[0], miny = vig[1], maxy = vig[1], minz = vig[2], maxz = vig[2];
    for (int m = 0; m < kM; ++m) {
        const int32_t x = vig[static_cast<size_t>(m) * 3 + 0], y = vig[static_cast<size_t>(m) * 3 + 1], z = vig[static_cast<size_t>(m) * 3 + 2];
        if (x < minx) minx = x; if (x > maxx) maxx = x;
        if (y < miny) miny = y; if (y > maxy) maxy = y;
        if (z < minz) minz = z; if (z > maxz) maxz = z;
    }
    std::printf("[info] vignette: final bounding box %d x %d x %d (started 24 x 1 x 1)\n",
                maxx - minx + 1, maxy - miny + 1, maxz - minz + 1);

    const bool vignette_pass = (steps_taken > 0) && (phi_cur < phi_start);
    std::printf("VIGNETTE: %s (%d legal moves executed, Phi %lld -> %lld, every intermediate state re-verified valid+connected)\n",
                vignette_pass ? "PASS" : "FAIL", steps_taken, static_cast<long long>(phi_start), static_cast<long long>(phi_cur));

    bool vignette_artifact_ok = write_vignette_csv(out_dir + "/vignette_frames.csv", frames);
    if (vignette_artifact_ok)
        std::printf("ARTIFACT: wrote demo/out/vignette_frames.csv (%d steps)\n", steps_taken + 1);
    else
        std::printf("ARTIFACT: FAILED to write demo/out/vignette_frames.csv\n");

    // ---- 9) artifact: config_render.pgm (batch configuration 0, guaranteed clean)
    bool pgm_ok = write_pgm(out_dir + "/config_render.pgm", &h_pos[0]);
    if (pgm_ok)
        std::printf("ARTIFACT: wrote demo/out/config_render.pgm (configuration 0)\n");
    else
        std::printf("ARTIFACT: FAILED to write demo/out/config_render.pgm\n");

    const bool success = verify_pass && corruption_pass && artic_bf_pass && move_bf_pass
                        && artifact_ok && vignette_pass && vignette_artifact_ok && pgm_ok;
    if (success)
        std::printf("RESULT: PASS (all gates passed: GPU-vs-CPU exact, corruption detection, brute-force cross-checks, vignette)\n");
    else
        std::printf("RESULT: FAIL (see the gate lines above for which check failed)\n");
    return success ? 0 : 1;
}
