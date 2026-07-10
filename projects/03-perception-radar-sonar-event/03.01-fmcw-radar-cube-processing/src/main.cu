// ===========================================================================
// main.cu — entry point for project 03.01
//           FMCW radar cube processing: range-Doppler-angle FFTs +
//           CA/OS-CFAR detection
//
// What this program does, start to finish
// -----------------------------------------
//   1. Load the committed radar configuration + target list from
//      data/sample/ (cross-checking the config file against the
//      compile-time constants in kernels.cuh — the "radar parameters" are
//      fixed at build time, but the file exists so a mismatch is a loud,
//      early error rather than a silently wrong demo).
//   2. GPU PIPELINE: synthesize the raw cube -> Hann-window + range FFT ->
//      Hann-window + Doppler FFT -> fftshift -> noncoherent antenna
//      integration -> 2-D CA-CFAR AND OS-CFAR -> host-side local-max
//      clustering -> per-detection zero-padded angle FFT -> azimuth.
//   3. VERIFY STAGE (the §5 GPU-vs-CPU gate): the CPU oracle
//      (reference_cpu.cpp) runs the SAME pipeline with an O(N^2) DFT and
//      the GPU range-Doppler power map is compared cell-by-cell.
//   4. GROUND-TRUTH gates: every injected target must be found by the
//      OS-CFAR detector within documented range/velocity/azimuth
//      tolerances, with no more than a documented number of false alarms;
//      separately, the scene's two closely-spaced targets must show
//      CA-CFAR masking one of them while OS-CFAR still detects it — the
//      textbook CA-vs-OS trade-off this project exists to teach, MEASURED
//      rather than asserted.
//   5. Artifacts: demo/out/range_doppler.pgm (a viewable log-magnitude
//      image of the GPU's range-Doppler map, OS-CFAR detections marked)
//      and demo/out/detections.csv (every detection from both detectors,
//      matched against ground truth, with per-field errors).
//
// Output contract (load-bearing!): every stable line quotes ONLY compile-
// time-fixed radar parameters or a qualitative PASS/FAIL verdict — never a
// GPU-architecture-dependent COUNT (false-alarm counts, exact worst-case
// deviations). Those exact MEASURED numbers are printed too, but on
// "[info]"/"[time]" lines the demo scripts deliberately do not diff — the
// same reason 08.01 keeps its balanced-streak length off the stable lines.
// Change a stable line -> update demo/expected_output.txt in the same change.
//
// Read this first, then kernels.cuh -> reference_cpu.cpp -> kernels.cu.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
#include <direct.h>               // _mkdir
#else
#include <sys/stat.h>
#endif

// ---------------------------------------------------------------------------
// bin_to_range_m / bin_to_vel_mps — the one place the (bin index) <->
// (physical unit) mapping is written, shared by every consumer below
// (detection records, the CSV writer, the ground-truth matcher). kd is
// always the SHIFTED, centered bin (kNc/2 = zero velocity) per kernels.cuh.
// ---------------------------------------------------------------------------
static inline float bin_to_range_m(int kr) { return static_cast<float>(kr) * kRangeResM; }
static inline float bin_to_vel_mps(int kd) { return static_cast<float>(kd - kNc / 2) * kVelResMps; }

// ---------------------------------------------------------------------------
// project_root_from / find_data_file / ensure_dir — the same small path
// helpers 08.01 uses, unchanged in spirit: resolve data/demo paths
// relative to the running executable so the demo works from any CWD.
// ---------------------------------------------------------------------------
static std::string project_root_from(const char* argv0)
{
    std::string exe(argv0 ? argv0 : "");
    size_t cut = exe.find_last_of("/\\");
    if (cut == std::string::npos) return ".";
    return exe.substr(0, cut) + "/../../..";
}

static std::string find_data_file(const std::string& filename, const char* argv0)
{
    std::vector<std::string> candidates = {
        project_root_from(argv0) + "/data/sample/" + filename,
        "data/sample/" + filename,
        "../data/sample/" + filename,
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

// ---------------------------------------------------------------------------
// load_radar_params — parse data/sample/radar_params.csv and CROSS-CHECK
// every field against the compile-time constants in kernels.cuh. The
// chirp/antenna parameters are fixed at BUILD time (they size CFAR's
// per-thread local arrays and the cuFFT plans — genuinely compile-time
// quantities, not something main() can resize), so this file is not a
// runtime configuration input; it is a committed, human-readable RECORD of
// the values the binary was built with, and this check turns "someone
// edited kernels.cuh but forgot the committed sample" into a loud, early
// failure instead of a silently mismatched demo (CLAUDE.md §8 honesty).
// ---------------------------------------------------------------------------
struct RadarParamsFile {
    double fc_hz = 0, bandwidth_hz = 0, chirp_dur_s = 0;
    int ns = 0, nc = 0, na = 0;
    bool loaded = false;
};

static RadarParamsFile load_radar_params(const std::string& path)
{
    RadarParamsFile p;
    std::ifstream in(path);
    if (!in.is_open()) return p;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> fields;
        while (std::getline(ss, cell, ',')) fields.push_back(cell);
        if (fields.size() != 6) { std::fprintf(stderr, "radar_params: expected 6 fields, got %zu\n", fields.size()); return RadarParamsFile{}; }
        p.fc_hz        = std::strtod(fields[0].c_str(), nullptr);
        p.bandwidth_hz = std::strtod(fields[1].c_str(), nullptr);
        p.chirp_dur_s  = std::strtod(fields[2].c_str(), nullptr);
        p.ns = std::atoi(fields[3].c_str());
        p.nc = std::atoi(fields[4].c_str());
        p.na = std::atoi(fields[5].c_str());
        p.loaded = true;
        break;   // one data row expected
    }
    return p;
}

// Returns "" if the file matches the build; otherwise a human-readable
// description of the first mismatch found.
static std::string check_radar_params(const RadarParamsFile& p)
{
    if (!p.loaded) return "file missing or malformed";
    char buf[256];
    // Relative tolerances throughout: kFc/kBandwidth/kChirpDur are FP32
    // constexpr values, and 77e9 is not exactly representable in float
    // (nearest FP32 is 76999999488.0 — ~5e-9 relative off, an honest
    // float32 rounding artifact, not a real mismatch). 1e-5 relative
    // comfortably clears FP32 rounding while still catching a genuinely
    // different value (e.g. a typo'd 78 GHz or a stale committed file).
    if (std::fabs(p.fc_hz - static_cast<double>(kFc)) > 1e-5 * static_cast<double>(kFc)) {
        std::snprintf(buf, sizeof(buf), "fc_hz mismatch: file=%.0f build=%.0f", p.fc_hz, static_cast<double>(kFc));
        return buf;
    }
    if (std::fabs(p.bandwidth_hz - static_cast<double>(kBandwidth)) > 1e-5 * static_cast<double>(kBandwidth)) {
        std::snprintf(buf, sizeof(buf), "bandwidth_hz mismatch: file=%.0f build=%.0f", p.bandwidth_hz, static_cast<double>(kBandwidth));
        return buf;
    }
    if (std::fabs(p.chirp_dur_s - static_cast<double>(kChirpDur)) > 1e-5 * static_cast<double>(kChirpDur)) {
        std::snprintf(buf, sizeof(buf), "chirp_dur_s mismatch: file=%.9f build=%.9f", p.chirp_dur_s, static_cast<double>(kChirpDur));
        return buf;
    }
    if (p.ns != kNs || p.nc != kNc || p.na != kNa) {
        std::snprintf(buf, sizeof(buf), "Ns/Nc/Na mismatch: file=(%d,%d,%d) build=(%d,%d,%d)", p.ns, p.nc, p.na, kNs, kNc, kNa);
        return buf;
    }
    return "";
}

// ---------------------------------------------------------------------------
// load_targets — strict CSV loader for data/sample/targets.csv, rows
// "range_m,vel_mps,az_deg,amp". Same strict-loader discipline as 08.01's
// scenario loader: unknown shapes, short rows, or an empty file abort the
// demo rather than silently running with fewer targets than intended.
// ---------------------------------------------------------------------------
static int load_targets(const std::string& path, RadarTarget* out, int max_targets)
{
    std::ifstream in(path);
    if (!in.is_open()) return -1;

    int count = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (count >= max_targets) { std::fprintf(stderr, "targets.csv: too many rows (cap %d)\n", max_targets); return -1; }
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> fields;
        while (std::getline(ss, cell, ',')) fields.push_back(cell);
        if (fields.size() != 4) { std::fprintf(stderr, "targets.csv: expected 4 fields, got %zu\n", fields.size()); return -1; }
        out[count].range_m = std::strtof(fields[0].c_str(), nullptr);
        out[count].vel_mps = std::strtof(fields[1].c_str(), nullptr);
        out[count].az_deg  = std::strtof(fields[2].c_str(), nullptr);
        out[count].amp     = std::strtof(fields[3].c_str(), nullptr);
        ++count;
    }
    return count;
}

// ---------------------------------------------------------------------------
// cluster_detections — collapse a raw per-cell CFAR detection mask into
// ISOLATED peaks: keep a flagged cell only if it is the local maximum of
// rd_power in its own 3x3 neighborhood. Without this, a single physical
// target triggers CFAR on every cell inside its (post-windowing, but
// still several-bins-wide) mainlobe, and a scene with 6 targets reports
// dozens of "detections" for the same 6 physical returns — the standard
// post-CFAR clustering step every real detector applies (THEORY.md "The
// algorithm"). Deliberately host-side and single-threaded: the INPUT is
// already tiny and sparse (a handful to a few dozen flagged cells out of
// 32,768), so a GPU kernel here would spend more time launching than
// computing — the same "small enough, keep it on the host" call 08.01
// makes for its softmin blend.
// ---------------------------------------------------------------------------
struct RawDet { int kr, kd; };

static std::vector<RawDet> cluster_detections(const unsigned char* det, const float* rd_power)
{
    std::vector<RawDet> out;
    for (int i = kCfarHalf; i < kNs - kCfarHalf; ++i) {
        for (int j = kCfarHalf; j < kNc - kCfarHalf; ++j) {
            const int idx = i * kNc + j;
            if (!det[idx]) continue;
            const float v = rd_power[idx];
            bool is_local_max = true;
            for (int di = -1; di <= 1 && is_local_max; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    if (rd_power[(i + di) * kNc + (j + dj)] > v) { is_local_max = false; break; }
                }
            }
            if (is_local_max) out.push_back({ i, j });
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// compute_detections_gpu — turn a raw (kr, kd) list into full Detection
// records by running the GPU angle-estimation stage (steps 7-9 of
// kernels.cu) on exactly those cells. Small, self-contained: uploads the
// index lists, gathers+FFTs+peak-finds, downloads azimuths.
// ---------------------------------------------------------------------------
static std::vector<Detection> compute_detections_gpu(const std::vector<RawDet>& raw,
                                                      const ComplexF32* d_cube_shifted,
                                                      const std::vector<float>& h_rd_power)
{
    std::vector<Detection> dets;
    const int n = static_cast<int>(raw.size());
    if (n == 0) return dets;

    std::vector<int> h_kr(n), h_kd(n);
    for (int i = 0; i < n; ++i) { h_kr[i] = raw[i].kr; h_kd[i] = raw[i].kd; }

    int *d_kr = nullptr, *d_kd = nullptr;
    ComplexF32* d_snapshots = nullptr;
    float* d_az = nullptr;
    CUDA_CHECK(cudaMalloc(&d_kr, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_kd, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_snapshots, static_cast<size_t>(n) * kNaFft * sizeof(ComplexF32)));
    CUDA_CHECK(cudaMalloc(&d_az, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_kr, h_kr.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kd, h_kd.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    launch_gather_angle_snapshots(d_cube_shifted, d_kr, d_kd, n, d_snapshots);
    launch_angle_fft(d_snapshots, n);
    launch_find_angle_peaks(d_snapshots, n, d_az);

    std::vector<float> h_az(n);
    CUDA_CHECK(cudaMemcpy(h_az.data(), d_az, n * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_kr));
    CUDA_CHECK(cudaFree(d_kd));
    CUDA_CHECK(cudaFree(d_snapshots));
    CUDA_CHECK(cudaFree(d_az));

    dets.reserve(n);
    for (int i = 0; i < n; ++i) {
        Detection d;
        d.kr = h_kr[i]; d.kd = h_kd[i];
        d.range_m = bin_to_range_m(d.kr);
        d.vel_mps = bin_to_vel_mps(d.kd);
        d.az_deg = h_az[i];
        d.power = h_rd_power[static_cast<size_t>(d.kr) * kNc + d.kd];
        dets.push_back(d);
    }
    return dets;
}

// ---------------------------------------------------------------------------
// match_to_targets — for each ground-truth target, find the detection
// (if any) within (kRangeTolM, kVelTolMps, kAzTolDeg) of it. Returns, per
// target, the index of its matching detection or -1 if none. Also counts
// detections that matched NO target (false alarms).
// ---------------------------------------------------------------------------
struct MatchReport {
    std::vector<int> target_to_det;   // [num_targets], -1 if unmatched
    int false_alarms = 0;
};

static MatchReport match_to_targets(const std::vector<Detection>& dets,
                                    const RadarTarget* targets, int num_targets)
{
    MatchReport rep;
    rep.target_to_det.assign(num_targets, -1);
    std::vector<bool> det_matched(dets.size(), false);

    for (int t = 0; t < num_targets; ++t) {
        int best = -1;
        float best_cost = 1e30f;
        for (size_t i = 0; i < dets.size(); ++i) {
            const float dr = std::fabs(dets[i].range_m - targets[t].range_m);
            const float dv = std::fabs(dets[i].vel_mps - targets[t].vel_mps);
            const float da = std::fabs(dets[i].az_deg - targets[t].az_deg);
            if (dr <= kRangeTolM && dv <= kVelTolMps && da <= kAzTolDeg) {
                const float cost = dr / kRangeTolM + dv / kVelTolMps + da / kAzTolDeg;  // normalized
                if (cost < best_cost) { best_cost = cost; best = static_cast<int>(i); }
            }
        }
        rep.target_to_det[t] = best;
        if (best >= 0) det_matched[best] = true;
    }
    for (bool m : det_matched) if (!m) rep.false_alarms++;
    return rep;
}

// ---------------------------------------------------------------------------
// write_detections_csv — every detection from both detectors, tagged, with
// its matched target (if any) and per-field errors — the "range/velocity/
// azimuth per detection vs ground truth" artifact the project promises.
// ---------------------------------------------------------------------------
static void write_detections_csv(const std::string& path,
                                 const std::vector<Detection>& ca, const MatchReport& ca_match,
                                 const std::vector<Detection>& os, const MatchReport& os_match,
                                 const RadarTarget* targets, int num_targets)
{
    std::ofstream f(path);
    if (!f.is_open()) return;
    f << "detector,range_m,vel_mps,az_deg,power,matched_target_idx,"
        "range_err_m,vel_err_mps,az_err_deg\n";

    auto write_rows = [&](const char* name, const std::vector<Detection>& dets) {
        // Invert target->det into det->target for this detector.
        const MatchReport& mr = (name[0] == 'C') ? ca_match : os_match;
        std::vector<int> det_to_target(dets.size(), -1);
        for (int t = 0; t < num_targets; ++t)
            if (mr.target_to_det[t] >= 0) det_to_target[mr.target_to_det[t]] = t;

        for (size_t i = 0; i < dets.size(); ++i) {
            const Detection& d = dets[i];
            f << name << ',' << d.range_m << ',' << d.vel_mps << ',' << d.az_deg << ',' << d.power << ',';
            if (det_to_target[i] >= 0) {
                const RadarTarget& t = targets[det_to_target[i]];
                f << det_to_target[i] << ','
                  << (d.range_m - t.range_m) << ',' << (d.vel_mps - t.vel_mps) << ',' << (d.az_deg - t.az_deg) << '\n';
            } else {
                f << "-1,,,\n";   // false alarm: no ground-truth match
            }
        }
    };
    write_rows("CA", ca);
    write_rows("OS", os);
}

// ---------------------------------------------------------------------------
// write_range_doppler_pgm — a binary PGM (P5) log-magnitude image of the
// range-Doppler power map: rows = range bins (near range at the top),
// columns = Doppler bins (most-receding velocity at the left, per this
// project's sign convention: POSITIVE velocity = approaching). OS-CFAR
// detections are marked with a small bright cross for visibility.
// ---------------------------------------------------------------------------
static bool write_range_doppler_pgm(const std::string& path, const std::vector<float>& rd_power,
                                    const std::vector<Detection>& os_dets)
{
    std::vector<float> db(rd_power.size());
    float lo = 1e30f, hi = -1e30f;
    for (size_t i = 0; i < rd_power.size(); ++i) {
        const float v = 10.0f * log10f(rd_power[i] + 1e-6f);   // relative dB (uncalibrated units — see README)
        db[i] = v;
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }
    const float span = (hi > lo) ? (hi - lo) : 1.0f;

    std::vector<unsigned char> pixels(rd_power.size());
    for (size_t i = 0; i < db.size(); ++i) {
        float norm = (db[i] - lo) / span;                       // 0..1
        norm = std::min(1.0f, std::max(0.0f, norm));
        pixels[i] = static_cast<unsigned char>(norm * 255.0f + 0.5f);
    }
    // Mark OS-CFAR detections with a bright cross so they are visible at a
    // glance against the log-compressed background.
    for (const Detection& d : os_dets) {
        for (int k = -2; k <= 2; ++k) {
            const int ri = d.kr, ci = d.kd + k;
            if (ci >= 0 && ci < kNc) pixels[static_cast<size_t>(ri) * kNc + ci] = 255;
            const int ci2 = d.kd, ri2 = d.kr + k;
            if (ri2 >= 0 && ri2 < kNs) pixels[static_cast<size_t>(ri2) * kNc + ci2] = 255;
        }
    }

    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    f << "P5\n" << kNc << ' ' << kNs << "\n255\n";
    f.write(reinterpret_cast<const char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
    return f.good();
}

// ===========================================================================
// main
// ===========================================================================
int main(int argc, char** argv)
{
    std::string params_path_override, targets_path_override;
    for (int i = 1; i < argc; ++i) {
        if      (!std::strcmp(argv[i], "--params")  && i + 1 < argc) params_path_override = argv[++i];
        else if (!std::strcmp(argv[i], "--targets") && i + 1 < argc) targets_path_override = argv[++i];
        else {
            std::fprintf(stderr, "usage: %s [--params radar_params.csv] [--targets targets.csv]\n", argv[0]);
            return 2;
        }
    }

    std::printf("[demo] FMCW radar cube processing: range-Doppler-angle FFTs + CA/OS-CFAR detection (project 03.01)\n");
    print_device_info();

    std::printf("PROBLEM: FMCW radar cube Ns=%d x Nc=%d x Na=%d samples, fc=%.1f GHz, B=%.1f MHz, Tc=%.1f us, FP32\n",
                kNs, kNc, kNa, static_cast<double>(kFc) / 1e9, static_cast<double>(kBandwidth) / 1e6,
                static_cast<double>(kChirpDur) * 1e6);
    std::printf("RESOLUTION: range %.3f m (max %.1f m); velocity %.3f m/s (max +/-%.2f m/s); angle FFT %d-pt zero-padded from %d antennas\n",
                static_cast<double>(kRangeResM), static_cast<double>(kRangeMaxM),
                static_cast<double>(kVelResMps), static_cast<double>(kVelMaxMps), kNaFft, kNa);

    // ---- load & cross-check the committed scenario -------------------------
    const std::string params_path = params_path_override.empty()
        ? find_data_file("radar_params.csv", argv[0]) : params_path_override;
    const std::string targets_path = targets_path_override.empty()
        ? find_data_file("targets.csv", argv[0]) : targets_path_override;

    if (params_path.empty() || targets_path.empty()) {
        std::printf("SCENARIO: NOT FOUND - data/sample/{radar_params,targets}.csv missing (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (scenario missing)\n");
        return 1;
    }
    const RadarParamsFile pf = load_radar_params(params_path);
    const std::string mismatch = check_radar_params(pf);
    if (!mismatch.empty()) {
        std::printf("SCENARIO: MISMATCH - %s\n", mismatch.c_str());
        std::printf("RESULT: FAIL (radar_params.csv does not match this build's kernels.cuh constants)\n");
        return 1;
    }

    RadarTarget targets[kMaxTargets];
    const int num_targets = load_targets(targets_path, targets, kMaxTargets);
    if (num_targets <= 0) {
        std::printf("SCENARIO: MALFORMED - see stderr\n");
        std::printf("RESULT: FAIL (targets.csv malformed or empty)\n");
        return 1;
    }
    std::printf("SCENARIO: %d targets loaded from data/sample/targets.csv [synthetic, seed %u]\n", num_targets, kNoiseSeed);

    // ===================== GPU PIPELINE ======================================
    const size_t total_samples = static_cast<size_t>(kNs) * kNc * kNa;
    ComplexF32 *d_cube = nullptr, *d_cube_shifted = nullptr;
    float* d_rd_power = nullptr;
    unsigned char *d_det_ca = nullptr, *d_det_os = nullptr;
    float *d_thresh_ca = nullptr, *d_thresh_os = nullptr;
    RadarTarget* d_targets = nullptr;

    CUDA_CHECK(cudaMalloc(&d_cube, total_samples * sizeof(ComplexF32)));
    CUDA_CHECK(cudaMalloc(&d_cube_shifted, total_samples * sizeof(ComplexF32)));
    CUDA_CHECK(cudaMalloc(&d_rd_power, static_cast<size_t>(kNs) * kNc * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_det_ca, static_cast<size_t>(kNs) * kNc * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_det_os, static_cast<size_t>(kNs) * kNc * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_thresh_ca, static_cast<size_t>(kNs) * kNc * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_thresh_os, static_cast<size_t>(kNs) * kNc * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_targets, num_targets * sizeof(RadarTarget)));
    CUDA_CHECK(cudaMemcpy(d_targets, targets, num_targets * sizeof(RadarTarget), cudaMemcpyHostToDevice));

    GpuTimer gpu_timer;
    gpu_timer.begin();
    launch_synthesize_cube(d_cube, d_targets, num_targets);
    launch_hann_window_range(d_cube);
    launch_range_fft(d_cube);
    launch_hann_window_doppler(d_cube);
    launch_doppler_fft(d_cube);
    launch_fftshift_doppler(d_cube, d_cube_shifted);
    launch_noncoherent_integrate(d_cube_shifted, d_rd_power);
    launch_cfar_ca(d_rd_power, d_det_ca, d_thresh_ca);
    launch_cfar_os(d_rd_power, d_det_os, d_thresh_os);
    const float gpu_pipeline_ms = gpu_timer.end_ms();

    std::vector<float> h_rd_power(static_cast<size_t>(kNs) * kNc);
    std::vector<unsigned char> h_det_ca(h_rd_power.size()), h_det_os(h_rd_power.size());
    CUDA_CHECK(cudaMemcpy(h_rd_power.data(), d_rd_power, h_rd_power.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_det_ca.data(), d_det_ca, h_det_ca.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_det_os.data(), d_det_os, h_det_os.size(), cudaMemcpyDeviceToHost));

    const std::vector<RawDet> raw_ca = cluster_detections(h_det_ca.data(), h_rd_power.data());
    const std::vector<RawDet> raw_os = cluster_detections(h_det_os.data(), h_rd_power.data());
    const std::vector<Detection> gpu_ca = compute_detections_gpu(raw_ca, d_cube_shifted, h_rd_power);
    const std::vector<Detection> gpu_os = compute_detections_gpu(raw_os, d_cube_shifted, h_rd_power);

    std::printf("[time] GPU pipeline (synthesize -> range/Doppler FFT -> integrate -> CFAR): %.3f ms\n",
                static_cast<double>(gpu_pipeline_ms));
    std::printf("[info] CA-CFAR: %d clustered detections | OS-CFAR: %d clustered detections\n",
                static_cast<int>(gpu_ca.size()), static_cast<int>(gpu_os.size()));

    CUDA_CHECK(cudaFree(d_targets));

    // ===================== CPU REFERENCE (oracle) ============================
    std::vector<ComplexF32> cpu_cube(total_samples);
    std::vector<float> cpu_rd_power(static_cast<size_t>(kNs) * kNc);
    std::vector<unsigned char> cpu_det_ca(cpu_rd_power.size()), cpu_det_os(cpu_rd_power.size());
    std::vector<float> cpu_thresh_ca(cpu_rd_power.size()), cpu_thresh_os(cpu_rd_power.size());

    CpuTimer cpu_timer;
    cpu_timer.begin();
    synthesize_cube_cpu(cpu_cube.data(), targets, num_targets);
    process_rd_map_cpu(cpu_cube.data(), cpu_rd_power.data());
    cfar_ca_cpu(cpu_rd_power.data(), cpu_det_ca.data(), cpu_thresh_ca.data());
    cfar_os_cpu(cpu_rd_power.data(), cpu_det_os.data(), cpu_thresh_os.data());
    const double cpu_ms = cpu_timer.end_ms();

    std::printf("[time] CPU O(N^2) DFT reference (synthesize -> range/Doppler DFT -> integrate -> CFAR): %.1f ms | GPU speed-up (teaching artifact): %.0fx\n",
                cpu_ms, cpu_ms / (static_cast<double>(gpu_pipeline_ms) > 0.0 ? static_cast<double>(gpu_pipeline_ms) : 1.0));

    // ===================== VERIFY: GPU vs CPU range-Doppler map ==============
    // Every one of the kNs*kNc cells is compared. Relative error with a
    // floor (protects near-zero noise-floor cells from an unstable
    // denominator) — the same shape of tolerance 08.01 uses, sized here
    // for a pipeline that chains TWO O(N) FFT accumulations (Ns=256 and
    // Nc=128 terms) plus a Na=8 antenna sum, rather than 08.01's 50
    // CHAINED RK4 steps: less compounding, but still real FP32 summation-
    // order differences between cuFFT's algorithm and this file's direct
    // O(N^2) sum. See THEORY.md "How we verify correctness" for the
    // measured worst-case value this tolerance is set well above.
    float worst_rel = 0.0f;
    for (size_t i = 0; i < h_rd_power.size(); ++i) {
        const float denom = std::max(cpu_rd_power[i], 1.0f);
        const float rel = std::fabs(h_rd_power[i] - cpu_rd_power[i]) / denom;
        if (rel > worst_rel) worst_rel = rel;
    }
    const float kVerifyTol = 0.05f;   // 5% relative (floor 1.0): ~9x headroom over the measured
                                       // worst case (0.55%) — generous but not knife-edge, see THEORY.md
    const bool verify_pass = (worst_rel <= kVerifyTol);
    std::printf("[info] VERIFY: worst relative range-Doppler power deviation (GPU vs CPU) = %.4f (tol %.2f)\n",
                static_cast<double>(worst_rel), static_cast<double>(kVerifyTol));
    std::printf("VERIFY: %s (GPU range-Doppler power map matches the CPU O(N^2) DFT reference within tolerance)\n",
                verify_pass ? "PASS" : "FAIL");

    // ===================== GROUND-TRUTH gates =================================
    const MatchReport os_match = match_to_targets(gpu_os, targets, num_targets);
    const MatchReport ca_match = match_to_targets(gpu_ca, targets, num_targets);

    int os_hits = 0;
    for (int t = 0; t < num_targets; ++t) if (os_match.target_to_det[t] >= 0) os_hits++;
    const bool all_targets_found = (os_hits == num_targets);
    const bool fa_bounded = (os_match.false_alarms <= kMaxFalseAlarmsOS);

    std::printf("[info] OS-CFAR ground truth: %d/%d targets matched (range<=%.2fm vel<=%.2fm/s az<=%.1fdeg), %d false alarm(s) (bound %d)\n",
                os_hits, num_targets, static_cast<double>(kRangeTolM), static_cast<double>(kVelTolMps),
                static_cast<double>(kAzTolDeg), os_match.false_alarms, kMaxFalseAlarmsOS);
    std::printf("GROUND_TRUTH: %s (every injected target found by OS-CFAR within tolerance; false alarms within bound)\n",
                (all_targets_found && fa_bounded) ? "PASS" : "FAIL");

    for (int t = 0; t < num_targets; ++t) {
        if (os_match.target_to_det[t] >= 0) {
            const Detection& d = gpu_os[static_cast<size_t>(os_match.target_to_det[t])];
            std::printf("[info]   target %d: true(R=%.2fm v=%+.2fm/s az=%+.1fdeg) -> det(R=%.2fm v=%+.2fm/s az=%+.1fdeg) err(dR=%.3fm dv=%.3fm/s daz=%.2fdeg)\n",
                        t, static_cast<double>(targets[t].range_m), static_cast<double>(targets[t].vel_mps), static_cast<double>(targets[t].az_deg),
                        static_cast<double>(d.range_m), static_cast<double>(d.vel_mps), static_cast<double>(d.az_deg),
                        static_cast<double>(d.range_m - targets[t].range_m), static_cast<double>(d.vel_mps - targets[t].vel_mps),
                        static_cast<double>(d.az_deg - targets[t].az_deg));
        } else {
            std::printf("[info]   target %d: true(R=%.2fm v=%+.2fm/s az=%+.1fdeg) -> NOT DETECTED by OS-CFAR\n",
                        t, static_cast<double>(targets[t].range_m), static_cast<double>(targets[t].vel_mps), static_cast<double>(targets[t].az_deg));
        }
    }

    // ---- CA-vs-OS comparison on the close-target pair (the last two rows
    // of the committed scenario, by construction — see data/README.md):
    // target (num_targets-1) is the WEAK member of the pair and is the one
    // this project's whole CFAR comparison hinges on. -------------------------
    const int weak_idx = num_targets - 1;
    const bool ca_misses_weak = (ca_match.target_to_det[weak_idx] < 0);
    const bool os_finds_weak  = (os_match.target_to_det[weak_idx] >= 0);
    const bool cfar_compare_ok = ca_misses_weak && os_finds_weak;
    std::printf("[info] CFAR compare: close-pair weak target (idx %d) CA-CFAR match=%s, OS-CFAR match=%s\n",
                weak_idx, ca_misses_weak ? "MISSED" : "found", os_finds_weak ? "found" : "MISSED");
    std::printf("CFAR_COMPARE: %s (CA-CFAR is masked by the neighboring strong target on the close pair; OS-CFAR is not)\n",
                cfar_compare_ok ? "PASS" : "FAIL");

    // ===================== Artifacts ===========================================
    const std::string out_dir = project_root_from(argv[0]) + "/demo/out";
    bool artifact_ok = ensure_dir(out_dir);
    if (artifact_ok) {
        artifact_ok = write_range_doppler_pgm(out_dir + "/range_doppler.pgm", h_rd_power, gpu_os);
    }
    if (artifact_ok) {
        write_detections_csv(out_dir + "/detections.csv", gpu_ca, ca_match, gpu_os, os_match, targets, num_targets);
    }
    if (artifact_ok)
        std::printf("ARTIFACT: wrote demo/out/range_doppler.pgm and demo/out/detections.csv\n");
    else
        std::printf("ARTIFACT: FAILED to write demo/out/ files\n");

    // ---- cleanup ---------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_cube));
    CUDA_CHECK(cudaFree(d_cube_shifted));
    CUDA_CHECK(cudaFree(d_rd_power));
    CUDA_CHECK(cudaFree(d_det_ca));
    CUDA_CHECK(cudaFree(d_det_os));
    CUDA_CHECK(cudaFree(d_thresh_ca));
    CUDA_CHECK(cudaFree(d_thresh_os));

    // ===================== RESULT ================================================
    const bool overall_pass = verify_pass && all_targets_found && fa_bounded && cfar_compare_ok && artifact_ok;
    if (overall_pass) {
        std::printf("RESULT: PASS (GPU matches CPU reference; every target detected by OS-CFAR; CA-vs-OS masking behavior confirmed)\n");
        return EXIT_SUCCESS;
    } else {
        std::printf("RESULT: FAIL (see VERIFY/GROUND_TRUTH/CFAR_COMPARE lines above)\n");
        return EXIT_FAILURE;
    }
}
