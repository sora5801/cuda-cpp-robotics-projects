// ===========================================================================
// main.cu — entry point for project 20.01
//           GelSight/DIGIT processing: contact patch, shear field via
//           optical flow, slip detection in real time
//
// What this program does, start to finish
// ---------------------------------------
//   0. RENDER: this file ALSO contains the synthetic gel-sensor renderer —
//      unlike a project that loads recorded images, a tactile sensor's
//      "dataset" here is a SCENARIO (data/sample/tactile_scenario.csv, in
//      08.01's spirit: a task definition, not recordings) plus the fixed
//      physical model in kernels.cuh. Every one of the 100 frames is
//      rendered IN-CODE, deterministically, from that scenario — a sphere
//      presses in, then shears, then partially slips (kernels.cuh's file
//      header has the whole physical story). Rendering is host-side data
//      GENERATION (like the template's make_input() or 08.01's plant
//      stepper), not something a kernel needs to compute.
//   1. Per frame: upload the rendered frame, run the 5-kernel GPU pipeline
//      (contact mask -> open -> patch stats -> detect markers -> track
//      markers), and run the CPU oracle on the SAME frame bytes — every one
//      of the 100 frames is cross-checked EXACTLY (kernels.cuh: every
//      operation here is integer/threshold arithmetic, so GPU and CPU must
//      agree bit-for-bit; no tolerance needed anywhere in this stage).
//   2. Host: a tiny (O(num markers)) rigid-transform fit turns each frame's
//      marker displacements into a single slip score (kernels.cuh; the same
//      "keep the cheap part on the host" call 08.01 makes for its softmin
//      blend).
//   3. THREE GROUND-TRUTH GATES, each comparing the algorithm's measurement
//      against the PHYSICS that generated the scene (not against itself):
//      CONTACT (patch area/centroid vs. the Hertzian footprint), SHEAR
//      (tracked displacement vs. the commanded translation), SLIP (detected
//      onset frame vs. the modeled Cattaneo-Mindlin onset, plus "no false
//      slip while stuck").
//   4. ARTIFACTS: demo/out/contact_mask.pgm, shear_field.csv,
//      slip_timeline.csv — the visual/tabular story the printed percentages
//      cannot tell alone.
//
// Output contract: stable lines are "[demo]", "PROBLEM:", "SCENARIO:",
// "VERIFY:", "CONTACT:", "SHEAR:", "SLIP:", "ARTIFACT:", "RESULT:";
// "[info]"/"[time]" lines are unchecked. Change a stable line => update
// demo/expected_output.txt in the same change.
//
// Read this first, then kernels.cuh (the sensor-model + pipeline contract),
// then kernels.cu, then reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

#include <algorithm>
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
// Ground-truth gate thresholds — calibrated from an ACTUAL measured run on
// this project's committed sample (numbers recorded in THEORY.md "How we
// verify correctness" and README "Expected output"; never asserted from
// theory alone, per CLAUDE.md §8 "never fabricate"). Wide margins below the
// measured values so the gates stay robust to ordinary FP rounding in the
// physics/fit math (the pipeline's own kernel outputs are exact integers —
// see kernels.cuh — only this GATE math, and the scene renderer, use floats).
//   Measured on the reference machine (RTX 2080 SUPER, sm_75):
//     CONTACT area rel error (press-hold frames)   = 1.3% mean, 1.3% max (8 frames)
//     CONTACT centroid abs error (px)               = 0.13 px max (8 frames)
//     SHEAR mean-displacement abs error (px)         = 0.00 px max (12 frames — exact; see THEORY.md
//                                                       "Numerical considerations" for why full-stick
//                                                       shear is provably exact under this project's
//                                                       integer-pixel marker rendering/detection)
//     SLIP onset frame vs modeled                    = frame 85 measured vs. 86 modeled (|err| = 1)
// ---------------------------------------------------------------------------
static constexpr double kPi = 3.14159265358979323846;    // avoids relying on MSVC's non-standard M_PI macro
static constexpr double kMaxContactAreaRelErr   = 0.05;  // 5%, ~4x the measured 1.3% mean/max
static constexpr double kMaxContactCentroidErrPx = 1.0;  // px, ~7.7x the measured 0.13px max
static constexpr double kMaxShearMeanErrPx       = 0.5;  // px, generous given the measured exact (0.00px) result
static constexpr int    kMaxSlipOnsetErrFrames   = 2;    // +/- frames vs. the modeled onset (measured |err|=1)

// Which press/shear frames count as "-hold" (steady-state, the most stable
// frames of each ramp-then-hold phase — see kernels.cuh's phase layout).
static bool is_press_hold(int t)  { return t >= kPressStart + kNPressRamp && t < kShearStart; }
static bool is_shear_hold(int t)  { return t >= kShearStart + kNShearRamp && t < kSlipStart; }

// Fixed artifact frames — one representative frame per phase, chosen at the
// middle of each "-hold" sub-phase (the most visually/numerically stable
// point to snapshot; kernels.cuh's constants make these exact, not magic).
static constexpr int kArtifactContactFrame = kPressStart + kNPressRamp + kNPressHold / 2;   // 26
static constexpr int kArtifactShearFrame   = kShearStart + kNShearRamp + kNShearHold / 2;   // 54

// ===========================================================================
// SECTION A — the synthetic sensor: scenario, physics, rendering.
// (kernels.cuh's file header derives every formula used below; this section
// only EVALUATES them.)
// ===========================================================================

enum class IndenterShape { Sphere, Edge };

// ---------------------------------------------------------------------------
// Scenario — the committed "task definition" (data/sample/tactile_scenario.csv):
// which indenter shape presses in, and the RNG seed for the gel's fixed
// micro-texture. Every other number (image size, marker grid, contact
// physics, phase lengths...) is the single-sourced, fixed teaching setup in
// kernels.cuh — only these two vary between regenerations (README Exercises).
// ---------------------------------------------------------------------------
struct Scenario {
    IndenterShape shape = IndenterShape::Sphere;
    unsigned seed = 42;
    bool loaded = false;
};

static Scenario load_scenario(const std::string& path)
{
    Scenario sc;
    std::ifstream in(path);
    if (!in.is_open()) return sc;

    bool have_indenter = false, have_seed = false;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string label, cell;
        std::getline(ss, label, ',');
        if (label == "INDENTER") {
            if (!std::getline(ss, cell, ',')) { std::fprintf(stderr, "scenario: short INDENTER row\n"); return Scenario{}; }
            if (cell == "sphere") sc.shape = IndenterShape::Sphere;
            else if (cell == "edge") sc.shape = IndenterShape::Edge;
            else { std::fprintf(stderr, "scenario: unknown INDENTER '%s' (want sphere|edge)\n", cell.c_str()); return Scenario{}; }
            have_indenter = true;
        } else if (label == "SEED") {
            if (!std::getline(ss, cell, ',')) { std::fprintf(stderr, "scenario: short SEED row\n"); return Scenario{}; }
            sc.seed = static_cast<unsigned>(std::strtoul(cell.c_str(), nullptr, 10));
            have_seed = true;
        } else {
            std::fprintf(stderr, "scenario: unknown row label '%s'\n", label.c_str());
            return Scenario{};
        }
    }
    if (!have_indenter || !have_seed) {
        std::fprintf(stderr, "scenario: missing INDENTER or SEED\n");
        return Scenario{};
    }
    sc.loaded = true;
    return sc;
}

// Indenter "radius" (mm) feeding the small-deflection contact-radius formula
// a = sqrt(R*depth) (kernels.cuh's file header derives this for the sphere;
// the edge/cylinder case reuses the identical parabolic-height derivation
// with the cylinder's own radius — see THEORY.md "The math").
static constexpr float kEdgeRadiusMm = 4.0f;
static float indenter_radius_mm(IndenterShape s) { return (s == IndenterShape::Sphere) ? kSphereRadiusMm : kEdgeRadiusMm; }

// hertz_contact_radius_px — the TRUE mechanical contact radius/half-width a
// (px) at mutual-approach depth (mm), via a = sqrt(R*depth) (kernels.cuh).
static float hertz_contact_radius_px(float depth_mm, IndenterShape shape)
{
    if (depth_mm <= 0.0f) return 0.0f;
    return kPxPerMm * std::sqrt(indenter_radius_mm(shape) * depth_mm);
}

// detectable_radius_px — the SMALLER, threshold-crossing radius a real
// intensity-threshold contact detector can actually see, derived in closed
// form from the shading model (kernels.cuh's file header, "SHADING MODEL"):
// darkening(r) = kShadeGainPerMm*depth*(1-(r/a)^2) hits kContactMaskThreshold
// exactly at r = a*sqrt(1 - threshold/(gain*depth)) — an EXACT consequence
// of the paraboloid shading profile, not an approximation, and the reason
// the CONTACT gate below compares against THIS radius, not the raw Hertz
// one (THEORY.md "How we verify correctness" names this explicitly).
static float detectable_radius_px(float depth_mm, IndenterShape shape)
{
    if (depth_mm <= 0.0f) return 0.0f;
    const float center_darkening = kShadeGainPerMm * depth_mm;
    if (center_darkening <= static_cast<float>(kContactMaskThreshold)) return 0.0f;  // never crosses threshold anywhere
    const float a = hertz_contact_radius_px(depth_mm, shape);
    const float frac = 1.0f - static_cast<float>(kContactMaskThreshold) / center_darkening;
    return a * std::sqrt(frac);
}

// FrameState — everything about frame t's physical scene that the renderer
// and the ground-truth gates both need; computed ONCE per frame by
// compute_frame_state() below so every consumer reads the same numbers.
struct FrameState {
    int phase = 0;                  // 0=BASELINE 1=PRESS 2=SHEAR 3=SLIP (see kernels.cuh phase layout)
    IndenterShape shape = IndenterShape::Sphere;
    float depth_mm = 0.0f;          // mutual-approach depth at THIS frame (mm)
    float contact_a_px = 0.0f;      // true Hertzian contact radius/half-width (px)
    float detect_a_px = 0.0f;       // threshold-visible radius (px) — see detectable_radius_px
    float center_x = kContactCenterX, center_y = kContactCenterY;  // current contact center (px)
    float stick_c_px = 0.0f;        // Cattaneo-Mindlin stick-zone radius (px); == contact_a_px when fully stuck
    float commanded_shear_px = 0.0f;// the object's own rigid lateral offset from rest THIS frame (px, +x)
};

static FrameState compute_frame_state(int t, IndenterShape shape)
{
    FrameState fs;
    fs.shape = shape;

    if (t < kPressStart) {
        // ---- BASELINE: no contact anywhere; also the reference frame. ----
        fs.phase = 0;
        return fs;   // depth=0, contact_a=0, detect_a=0, center=rest, stick_c=0, shear=0 — all defaults
    }

    if (t < kShearStart) {
        // ---- PRESS: indentation ramps in, then holds; no shear at all —
        // commanded_shear_px stays 0, so marker_displacement() below is 0
        // everywhere automatically (no special-casing needed anywhere). ----
        fs.phase = 1;
        const int local = t - kPressStart;
        fs.depth_mm = (local < kNPressRamp)
                     ? kIndentDepthMaxMm * static_cast<float>(local) / static_cast<float>(kNPressRamp - 1)
                     : kIndentDepthMaxMm;
        fs.contact_a_px = hertz_contact_radius_px(fs.depth_mm, shape);
        fs.detect_a_px = detectable_radius_px(fs.depth_mm, shape);
        fs.stick_c_px = fs.contact_a_px;   // no shear commanded yet => fully "stuck" trivially
        return fs;
    }

    if (t < kSlipStart) {
        // ---- SHEAR: depth already at max and held; the object translates,
        // dragging the WHOLE contact patch (fully stuck: c == a, s == 0) —
        // every in-contact marker gets exactly the commanded translation. ----
        fs.phase = 2;
        fs.depth_mm = kIndentDepthMaxMm;
        fs.contact_a_px = hertz_contact_radius_px(fs.depth_mm, shape);
        fs.detect_a_px = detectable_radius_px(fs.depth_mm, shape);
        const int local = t - kShearStart;
        fs.commanded_shear_px = (local < kNShearRamp)
                               ? kShearTotalPx * static_cast<float>(local) / static_cast<float>(kNShearRamp - 1)
                               : kShearTotalPx;
        fs.center_x = kContactCenterX + fs.commanded_shear_px;
        fs.stick_c_px = fs.contact_a_px;   // fully stuck throughout SHEAR by construction
        return fs;
    }

    // ---- SLIP: object position HELD at the final shear offset; the
    // Cattaneo-Mindlin load fraction s ramps 0->1, shrinking the stick
    // radius c(s) = a*(1-s)^(1/3) (kernels.cuh derivation). ----
    fs.phase = 3;
    fs.depth_mm = kIndentDepthMaxMm;
    fs.contact_a_px = hertz_contact_radius_px(fs.depth_mm, shape);
    fs.detect_a_px = detectable_radius_px(fs.depth_mm, shape);
    fs.commanded_shear_px = kShearTotalPx;
    fs.center_x = kContactCenterX + kShearTotalPx;
    const int local = t - kSlipStart;
    const float s = (kNSlip > 1) ? static_cast<float>(local) / static_cast<float>(kNSlip - 1) : 1.0f;
    fs.stick_c_px = fs.contact_a_px * std::cbrt(std::max(0.0f, 1.0f - s));
    return fs;
}

// contact_distance_px — "how far is this point from the touched patch's
// axis", in the metric the CURRENT indenter shape actually uses: radial for
// a sphere (a point contact), purely horizontal for an edge (a line contact
// running the full frame height — see README "Exercises" for regenerating
// with --indenter edge). Both the shading model and the marker stick/slip
// model read distance through THIS one function, so supporting a second
// indenter shape costs a single branch, not a duplicated physics model.
static float contact_distance_px(float px_x, float px_y, const FrameState& fs)
{
    const float dx = px_x - fs.center_x;
    if (fs.shape == IndenterShape::Sphere) {
        const float dy = px_y - fs.center_y;
        return std::sqrt(dx * dx + dy * dy);
    }
    return std::fabs(dx);   // Edge: contact band is vertical, distance is purely horizontal
}

// shading_darkening — gray-level darkening at image point (px_x,px_y) this
// frame, from the paraboloid depth profile (kernels.cuh "SHADING MODEL").
static float shading_darkening(float px_x, float px_y, const FrameState& fs)
{
    if (fs.contact_a_px <= 0.0f) return 0.0f;
    const float r = contact_distance_px(px_x, px_y, fs);
    if (r > fs.contact_a_px) return 0.0f;
    const float depth_r_mm = fs.depth_mm * (1.0f - (r * r) / (fs.contact_a_px * fs.contact_a_px));
    return kShadeGainPerMm * depth_r_mm;
}

// marker_displacement — this marker's (rest_x,rest_y) lateral motion (px)
// this frame, from the Cattaneo-Mindlin stick/slip model (kernels.cuh):
// full commanded motion inside the stick radius, a linearly-interpolated
// fraction (down to kStickResidualFrac) in the slipping annulus, zero
// outside the contact footprint entirely.
static Vec2f marker_displacement(float rest_x, float rest_y, const FrameState& fs)
{
    if (fs.contact_a_px <= 0.0f) return {0.0f, 0.0f};
    const float r = contact_distance_px(rest_x, rest_y, fs);
    if (r > fs.contact_a_px) return {0.0f, 0.0f};       // never touched

    float frac;
    if (r <= fs.stick_c_px) {
        frac = 1.0f;                                    // fully stuck: rides with the object exactly
    } else {
        const float denom = fs.contact_a_px - fs.stick_c_px;
        const float u = (denom > 1e-6f) ? (fs.contact_a_px - r) / denom : 0.0f;  // 1 at c, 0 at a
        frac = kStickResidualFrac + (1.0f - kStickResidualFrac) * u;
    }
    return { fs.commanded_shear_px * frac, 0.0f };       // this scenario's commanded motion is pure +x
}

// ---------------------------------------------------------------------------
// Deterministic hash -> [0,1) texture noise (identical construction to
// 01.02's _hash01 — a fixed function of (ix,iy,seed), no RNG state, so it is
// reproducible from these three integers alone regardless of call order).
// This models the gel's FIXED manufacturing micro-texture, not per-frame
// sensor read noise — it is the SAME every frame by design, which is
// exactly what makes background subtraction (frame - baseline) cleanly
// isolate contact-induced change and nothing else (THEORY.md "Numerical
// considerations").
// ---------------------------------------------------------------------------
static float hash01(int ix, int iy, unsigned seed)
{
    unsigned h = static_cast<unsigned>(ix * 374761393 + iy * 668265263) + seed * 2654435761u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= h >> 16;
    return static_cast<float>(h & 0xFFFFFFu) / static_cast<float>(1u << 24);   // top 24 bits -> [0,1)
}

static int texture_noise_gray(int x, int y, unsigned seed)
{
    const float u = hash01(x, y, seed);                 // in [0,1)
    return static_cast<int>(std::lround((u - 0.5f) * 2.0f * kTextureNoiseAmplitude));  // in [-A, +A]
}

static float smoothstep(float t) { return t * t * (3.0f - 2.0f * t); }

// build_marker_rest_positions — the fixed kNumMarkers x,y lattice
// (kernels.cuh "MARKER GRID"). Computed once; every frame's rendering and
// every kernel launch reuses this SAME array.
static std::vector<Vec2f> build_marker_rest_positions()
{
    std::vector<Vec2f> pos;
    pos.reserve(static_cast<size_t>(kNumMarkers));
    for (int row = 0; row < kMarkerNy; ++row) {
        for (int col = 0; col < kMarkerNx; ++col) {
            pos.push_back({
                static_cast<float>(kMarkerMarginPx + col * kMarkerSpacingPx),
                static_cast<float>(kMarkerMarginPx + row * kMarkerSpacingPx)
            });
        }
    }
    return pos;
}

// render_frame — fill h_frame[kImgW*kImgH] with frame t's synthetic image.
// Two passes (kernels.cuh's file header explains why): (1) the gel
// background, per pixel — baseline gray, minus contact shading, plus fixed
// texture noise; (2) each marker, blended radially into whatever background
// value pass (1) already wrote there, from full kMarkerDarkGray at its exact
// (rounded) center out to the background value at kMarkerRadiusPx — a
// smooth "bowl" whose minimum is PROVABLY unique to the center pixel (every
// other pixel in the gel, marker or not, is verified >= ~90 gray levels
// above kMarkerDarkGray=60's floor — see the file's tail comment), which is
// exactly what makes detect_markers_kernel's argmin search well-posed.
static void render_frame(int t, const Scenario& sc, const std::vector<Vec2f>& rest_pos,
                         std::vector<unsigned char>& h_frame)
{
    const FrameState fs = compute_frame_state(t, sc.shape);
    h_frame.assign(static_cast<size_t>(kImgW) * kImgH, 0);

    // ---- pass 1: gel background (shading + fixed texture noise) ----------
    for (int y = 0; y < kImgH; ++y) {
        for (int x = 0; x < kImgW; ++x) {
            const float shade = shading_darkening(static_cast<float>(x), static_cast<float>(y), fs);
            const int noise = texture_noise_gray(x, y, sc.seed);
            int v = kGelBaselineGray - static_cast<int>(std::lround(shade)) + noise;
            v = std::min(255, std::max(0, v));
            h_frame[static_cast<size_t>(y) * kImgW + x] = static_cast<unsigned char>(v);
        }
    }

    // ---- pass 2: markers, blended on top of pass 1's background ----------
    const int R = static_cast<int>(std::ceil(kMarkerRadiusPx));
    for (const Vec2f& rest : rest_pos) {
        const Vec2f d = marker_displacement(rest.x, rest.y, fs);
        const int mx = static_cast<int>(std::lround(rest.x + d.x));
        const int my = static_cast<int>(std::lround(rest.y + d.y));
        for (int dy = -R; dy <= R; ++dy) {
            for (int dx = -R; dx <= R; ++dx) {
                const float r = std::sqrt(static_cast<float>(dx * dx + dy * dy));
                if (r > kMarkerRadiusPx) continue;
                const int px = mx + dx, py = my + dy;
                if (px < 0 || px >= kImgW || py < 0 || py >= kImgH) continue;
                const size_t idx = static_cast<size_t>(py) * kImgW + px;
                const float blend = 1.0f - smoothstep(r / kMarkerRadiusPx);   // 1 at center, 0 at the rim
                const float gel_val = static_cast<float>(h_frame[idx]);      // pass 1's value here
                const float marker_val = gel_val * (1.0f - blend) + static_cast<float>(kMarkerDarkGray) * blend;
                h_frame[idx] = static_cast<unsigned char>(std::min(255, std::max(0, static_cast<int>(std::lround(marker_val)))));
            }
        }
    }
}

// ===========================================================================
// SECTION B — the host-side rigid fit + slip score (kernels.cuh: this is
// the O(num markers) piece deliberately kept off the GPU — see main()'s
// per-frame loop for why, mirroring 08.01's softmin blend).
// ===========================================================================

// rigid_fit_and_slip — closed-form 2-D Procrustes (rotation + translation,
// no scale) over the CURRENT frame's in-contact, valid markers, via the
// complex-number formulation: treating each centered point as a complex
// number, the optimal rotation is the argument of Sum(q'_i * conj(p'_i))
// (THEORY.md "The math" derives this in full; it needs no SVD in 2-D).
// Returns the fitted angle (diagnostic only — this scenario is pure
// translation, so it should stay near 0), the slip score (fraction of
// active markers whose fit RESIDUAL exceeds kResidualSlipThresholdPx), and
// how many markers were active (denominator; <3 is treated as "no
// meaningful patch yet" -> slip_score 0, matching true baseline/no-contact
// frames automatically).
static void rigid_fit_and_slip(const std::vector<Vec2f>& rest, const std::vector<Vec2f>& detected,
                               const std::vector<unsigned char>& in_contact,
                               const std::vector<unsigned char>& valid,
                               float* out_theta_rad, float* out_slip_score, int* out_n_active)
{
    std::vector<int> idxs;
    idxs.reserve(rest.size());
    for (size_t i = 0; i < rest.size(); ++i)
        if (in_contact[i] && valid[i]) idxs.push_back(static_cast<int>(i));

    const int n = static_cast<int>(idxs.size());
    *out_n_active = n;
    if (n < 3) { *out_theta_rad = 0.0f; *out_slip_score = 0.0f; return; }   // too few points for a meaningful fit

    // Centroids (double accumulators: n is at most kNumMarkers=221, but
    // double costs nothing and keeps the trig below well-conditioned).
    double cpx = 0.0, cpy = 0.0, cqx = 0.0, cqy = 0.0;
    for (int idx : idxs) {
        cpx += rest[idx].x; cpy += rest[idx].y;
        cqx += detected[idx].x; cqy += detected[idx].y;
    }
    cpx /= n; cpy /= n; cqx /= n; cqy /= n;

    // Optimal rotation via the complex-number trick: z_p = p'+i p'', z_q =
    // q'+i q''; Sum(z_q * conj(z_p)) = Re + i*Im; theta = atan2(Im, Re).
    double re = 0.0, im = 0.0;
    for (int idx : idxs) {
        const double px = rest[idx].x - cpx, py = rest[idx].y - cpy;
        const double qx = detected[idx].x - cqx, qy = detected[idx].y - cqy;
        re += qx * px + qy * py;
        im += qy * px - qx * py;
    }
    const double theta = std::atan2(im, re);
    const double ct = std::cos(theta), st = std::sin(theta);
    const double tx = cqx - (ct * cpx - st * cpy);
    const double ty = cqy - (st * cpx + ct * cpy);

    int n_slip = 0;
    for (int idx : idxs) {
        const double pred_x = ct * rest[idx].x - st * rest[idx].y + tx;
        const double pred_y = st * rest[idx].x + ct * rest[idx].y + ty;
        const double res = std::sqrt((detected[idx].x - pred_x) * (detected[idx].x - pred_x) +
                                     (detected[idx].y - pred_y) * (detected[idx].y - pred_y));
        if (res > kResidualSlipThresholdPx) ++n_slip;
    }

    *out_theta_rad = static_cast<float>(theta);
    *out_slip_score = static_cast<float>(n_slip) / static_cast<float>(n);
}

// ===========================================================================
// SECTION C — small I/O helpers (PGM/CSV writers, path resolution) — same
// idioms as 01.02/08.01/07.09 (std::filesystem deliberately avoided in .cu).
// ===========================================================================

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
    candidates.push_back(project_root_from(argv0) + "/data/sample/tactile_scenario.csv");
    candidates.push_back("data/sample/tactile_scenario.csv");
    candidates.push_back("../data/sample/tactile_scenario.csv");
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

static bool write_pgm(const std::string& path, int W, int H, const std::vector<unsigned char>& gray)
{
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) return false;
    out << "P5\n" << W << " " << H << "\n255\n";
    out.write(reinterpret_cast<const char*>(gray.data()), static_cast<std::streamsize>(gray.size()));
    return static_cast<bool>(out);
}

static const char* phase_name(int phase)
{
    switch (phase) {
        case 0: return "BASELINE";
        case 1: return "PRESS";
        case 2: return "SHEAR";
        default: return "SLIP";
    }
}

// exact_match — count element-wise mismatches (used for every VERIFY
// checkpoint below; every comparison in this project is exact — kernels.cuh
// explains why: integer/threshold arithmetic on a shared uint8 input).
template <typename T>
static long long exact_match(const std::vector<T>& a, const std::vector<T>& b)
{
    long long mism = 0;
    const size_t n = a.size();
    for (size_t i = 0; i < n; ++i) if (!(a[i] == b[i])) ++mism;
    return mism;
}

// Vec2f has no operator== by default (plain aggregate, kernels.cuh) — define
// one HERE, scoped to this translation unit's use in exact_match above,
// rather than growing the shared header with a convenience operator no
// device code needs.
static bool operator==(const Vec2f& a, const Vec2f& b) { return a.x == b.x && a.y == b.y; }

// ===========================================================================
// main.
// ===========================================================================
int main(int argc, char** argv)
{
    std::string data_path;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--data") && i + 1 < argc) data_path = argv[++i];
        else {
            std::fprintf(stderr, "usage: %s [--data path/to/tactile_scenario.csv]\n", argv[0]);
            return 2;
        }
    }

    std::printf("[demo] GelSight/DIGIT tactile processing: contact patch, shear field, slip detection (project 20.01)\n");
    print_device_info();
    std::printf("PROBLEM: synthetic gel-sensor sequence %dx%d, %d frames (%d baseline + %d press + %d shear + %d slip), %d markers\n",
               kImgW, kImgH, kNumFrames, kNBaseline, kNPress, kNShear, kNSlip, kNumMarkers);

    // ---- scenario --------------------------------------------------------
    const std::string scenario_path = find_scenario(data_path, argv[0]);
    if (scenario_path.empty()) {
        std::printf("SCENARIO: NOT FOUND - data/sample/tactile_scenario.csv missing (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (scenario missing)\n");
        return 1;
    }
    Scenario sc = load_scenario(scenario_path);
    if (!sc.loaded) {
        std::printf("SCENARIO: MALFORMED - see stderr\n");
        std::printf("RESULT: FAIL (scenario malformed)\n");
        return 1;
    }
    std::printf("[info] scenario file: %s\n", scenario_path.c_str());
    const float a_max = hertz_contact_radius_px(kIndentDepthMaxMm, sc.shape);
    std::printf("SCENARIO: [synthetic, seed %u] indenter=%s, R=%.1fmm, depth_max=%.2fmm (contact radius ~%.1fpx), commanded shear %.1fpx, Cattaneo-Mindlin partial slip\n",
               sc.seed, sc.shape == IndenterShape::Sphere ? "sphere" : "edge",
               static_cast<double>(indenter_radius_mm(sc.shape)), static_cast<double>(kIndentDepthMaxMm),
               static_cast<double>(a_max), static_cast<double>(kShearTotalPx));

    // ---- marker rest grid (fixed for the whole run) -----------------------
    const std::vector<Vec2f> rest_pos = build_marker_rest_positions();

    // ---- persistent device buffers -----------------------------------------
    const size_t pix_n = static_cast<size_t>(kImgW) * kImgH;
    unsigned char *d_frame = nullptr, *d_baseline = nullptr;
    unsigned char *d_mask_raw = nullptr, *d_mask_eroded = nullptr, *d_mask = nullptr;
    unsigned long long *d_area = nullptr, *d_sumx = nullptr, *d_sumy = nullptr;
    Vec2f *d_rest_pos = nullptr, *d_detected_pos = nullptr, *d_displacement = nullptr;
    int* d_min_intensity = nullptr;
    unsigned char *d_valid = nullptr, *d_in_contact = nullptr;

    CUDA_CHECK(cudaMalloc(&d_frame, pix_n));
    CUDA_CHECK(cudaMalloc(&d_baseline, pix_n));
    CUDA_CHECK(cudaMalloc(&d_mask_raw, pix_n));
    CUDA_CHECK(cudaMalloc(&d_mask_eroded, pix_n));
    CUDA_CHECK(cudaMalloc(&d_mask, pix_n));
    CUDA_CHECK(cudaMalloc(&d_area, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_sumx, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_sumy, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_rest_pos, kNumMarkers * sizeof(Vec2f)));
    CUDA_CHECK(cudaMalloc(&d_detected_pos, kNumMarkers * sizeof(Vec2f)));
    CUDA_CHECK(cudaMalloc(&d_displacement, kNumMarkers * sizeof(Vec2f)));
    CUDA_CHECK(cudaMalloc(&d_min_intensity, kNumMarkers * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_valid, kNumMarkers));
    CUDA_CHECK(cudaMalloc(&d_in_contact, kNumMarkers));
    CUDA_CHECK(cudaMemcpy(d_rest_pos, rest_pos.data(), kNumMarkers * sizeof(Vec2f), cudaMemcpyHostToDevice));

    // ---- baseline frame: rendered once (frame 0, guaranteed no-contact),
    // uploaded once, and reused as the background-subtraction reference for
    // every one of the 100 frames (a real sensor's calibration shot). -------
    std::vector<unsigned char> h_baseline;
    render_frame(kBaselineStart, sc, rest_pos, h_baseline);
    CUDA_CHECK(cudaMemcpy(d_baseline, h_baseline.data(), pix_n, cudaMemcpyHostToDevice));

    // ---- per-frame host buffers (reused across the loop; no per-frame
    // heap churn beyond render_frame's own working vectors) -----------------
    std::vector<unsigned char> h_frame(pix_n), h_mask_gpu(pix_n), h_mask_cpu(pix_n);
    std::vector<unsigned char> h_mask_raw_cpu(pix_n), h_mask_er_cpu(pix_n);
    std::vector<Vec2f> h_detected_gpu(kNumMarkers), h_detected_cpu(kNumMarkers);
    std::vector<Vec2f> h_displacement_gpu(kNumMarkers), h_displacement_cpu(kNumMarkers);
    std::vector<int> h_min_intensity_gpu(kNumMarkers), h_min_intensity_cpu(kNumMarkers);
    std::vector<unsigned char> h_valid_gpu(kNumMarkers), h_valid_cpu(kNumMarkers);
    std::vector<unsigned char> h_in_contact_gpu(kNumMarkers), h_in_contact_cpu(kNumMarkers);

    // ---- running verification + gate accumulators --------------------------
    long long total_mismatches = 0;
    double gpu_ms_total = 0.0;

    std::vector<double> contact_area_rel_err, contact_centroid_err_px;
    std::vector<double> shear_mean_err_px;
    int measured_slip_onset = -1;
    bool false_slip_in_stick_phase = false;

    struct SlipRow { int frame; int phase; float score; bool declared; };
    std::vector<SlipRow> slip_timeline;
    slip_timeline.reserve(kNumFrames);

    std::vector<unsigned char> artifact_mask;                 // saved at kArtifactContactFrame
    std::vector<Vec2f> artifact_rest, artifact_detected;       // saved at kArtifactShearFrame
    std::vector<unsigned char> artifact_in_contact, artifact_valid;

    // =========================== per-frame loop ==============================
    for (int t = 0; t < kNumFrames; ++t) {
        const FrameState fs = compute_frame_state(t, sc.shape);
        render_frame(t, sc, rest_pos, h_frame);
        CUDA_CHECK(cudaMemcpy(d_frame, h_frame.data(), pix_n, cudaMemcpyHostToDevice));

        // ---- GPU pipeline ---------------------------------------------------
        GpuTimer gt; gt.begin();
        launch_contact_mask(d_frame, d_baseline, d_mask_raw, kImgW, kImgH, kContactMaskThreshold);
        launch_erode3(d_mask_raw, d_mask_eroded, kImgW, kImgH);
        launch_dilate3(d_mask_eroded, d_mask, kImgW, kImgH);

        CUDA_CHECK(cudaMemset(d_area, 0, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_sumx, 0, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_sumy, 0, sizeof(unsigned long long)));
        launch_patch_stats(d_mask, kImgW, kImgH, d_area, d_sumx, d_sumy);

        launch_detect_markers(d_frame, d_rest_pos, kNumMarkers, kImgW, kImgH, kSearchRadiusPx,
                              d_detected_pos, d_min_intensity);
        launch_track_markers(d_detected_pos, d_min_intensity, d_rest_pos, d_mask,
                             kNumMarkers, kImgW, kImgH, kMarkerDetectThreshold,
                             d_displacement, d_valid, d_in_contact);
        gpu_ms_total += static_cast<double>(gt.end_ms());

        // ---- download everything this frame needs (mask + marker arrays;
        // all tiny except the mask, and even that is 76,800 bytes) -----------
        unsigned long long area_gpu = 0, sumx_gpu = 0, sumy_gpu = 0;
        CUDA_CHECK(cudaMemcpy(&area_gpu, d_area, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&sumx_gpu, d_sumx, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&sumy_gpu, d_sumy, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_mask_gpu.data(), d_mask, pix_n, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_detected_gpu.data(), d_detected_pos, kNumMarkers * sizeof(Vec2f), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_displacement_gpu.data(), d_displacement, kNumMarkers * sizeof(Vec2f), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_min_intensity_gpu.data(), d_min_intensity, kNumMarkers * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_valid_gpu.data(), d_valid, kNumMarkers, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_in_contact_gpu.data(), d_in_contact, kNumMarkers, cudaMemcpyDeviceToHost));

        // ---- CPU oracle, same frame bytes, same everything -------------------
        contact_mask_cpu(h_frame.data(), h_baseline.data(), h_mask_raw_cpu.data(), kImgW, kImgH, kContactMaskThreshold);
        erode3_cpu(h_mask_raw_cpu.data(), h_mask_er_cpu.data(), kImgW, kImgH);
        dilate3_cpu(h_mask_er_cpu.data(), h_mask_cpu.data(), kImgW, kImgH);
        unsigned long long area_cpu = 0, sumx_cpu = 0, sumy_cpu = 0;
        patch_stats_cpu(h_mask_cpu.data(), kImgW, kImgH, &area_cpu, &sumx_cpu, &sumy_cpu);
        detect_markers_cpu(h_frame.data(), rest_pos.data(), kNumMarkers, kImgW, kImgH, kSearchRadiusPx,
                          h_detected_cpu.data(), h_min_intensity_cpu.data());
        track_markers_cpu(h_detected_cpu.data(), h_min_intensity_cpu.data(), rest_pos.data(), h_mask_cpu.data(),
                         kNumMarkers, kImgW, kImgH, kMarkerDetectThreshold,
                         h_displacement_cpu.data(), h_valid_cpu.data(), h_in_contact_cpu.data());

        // ---- EXACT verify, this frame -----------------------------------------
        long long mism = exact_match(h_mask_gpu, h_mask_cpu);
        mism += (area_gpu != area_cpu) + (sumx_gpu != sumx_cpu) + (sumy_gpu != sumy_cpu);
        mism += exact_match(h_detected_gpu, h_detected_cpu);
        mism += exact_match(h_min_intensity_gpu, h_min_intensity_cpu);
        mism += exact_match(h_displacement_gpu, h_displacement_cpu);
        mism += exact_match(h_valid_gpu, h_valid_cpu);
        mism += exact_match(h_in_contact_gpu, h_in_contact_cpu);
        total_mismatches += mism;

        // ---- ground-truth bookkeeping (uses the GPU's own answers — the
        // demo's actual output — now that it is known to match the oracle) ----
        const double area_expected = kPi * static_cast<double>(fs.detect_a_px) * static_cast<double>(fs.detect_a_px);
        if (is_press_hold(t)) {
            const double rel_err = (area_expected > 0.0)
                                  ? std::fabs(static_cast<double>(area_gpu) - area_expected) / area_expected
                                  : 0.0;
            contact_area_rel_err.push_back(rel_err);
            if (area_gpu > 0) {
                const double cx = static_cast<double>(sumx_gpu) / static_cast<double>(area_gpu);
                const double cy = static_cast<double>(sumy_gpu) / static_cast<double>(area_gpu);
                const double dcx = cx - static_cast<double>(kContactCenterX);
                const double dcy = cy - static_cast<double>(kContactCenterY);
                contact_centroid_err_px.push_back(std::sqrt(dcx * dcx + dcy * dcy));
            }
        }
        if (is_shear_hold(t)) {
            double sum_dx = 0.0; int n_active = 0;
            for (int i = 0; i < kNumMarkers; ++i) {
                if (h_in_contact_gpu[i] && h_valid_gpu[i]) { sum_dx += h_displacement_gpu[i].x; ++n_active; }
            }
            if (n_active > 0)
                shear_mean_err_px.push_back(std::fabs(sum_dx / n_active - static_cast<double>(kShearTotalPx)));
        }

        float theta = 0.0f, slip_score = 0.0f; int n_active = 0;
        rigid_fit_and_slip(rest_pos, h_detected_gpu, h_in_contact_gpu, h_valid_gpu, &theta, &slip_score, &n_active);
        const bool declared = slip_score >= kSlipScoreDeclareThreshold;
        slip_timeline.push_back({ t, fs.phase, slip_score, declared });
        if (t < kSlipStart && declared) false_slip_in_stick_phase = true;
        if (t >= kSlipStart && declared && measured_slip_onset < 0) measured_slip_onset = t;

        if (t == kArtifactContactFrame) artifact_mask = h_mask_gpu;
        if (t == kArtifactShearFrame) {
            artifact_rest = rest_pos;
            artifact_detected = h_detected_gpu;
            artifact_in_contact = h_in_contact_gpu;
            artifact_valid = h_valid_gpu;
        }
    }

    CUDA_CHECK(cudaFree(d_frame));      CUDA_CHECK(cudaFree(d_baseline));
    CUDA_CHECK(cudaFree(d_mask_raw));   CUDA_CHECK(cudaFree(d_mask_eroded)); CUDA_CHECK(cudaFree(d_mask));
    CUDA_CHECK(cudaFree(d_area));       CUDA_CHECK(cudaFree(d_sumx));        CUDA_CHECK(cudaFree(d_sumy));
    CUDA_CHECK(cudaFree(d_rest_pos));   CUDA_CHECK(cudaFree(d_detected_pos)); CUDA_CHECK(cudaFree(d_displacement));
    CUDA_CHECK(cudaFree(d_min_intensity)); CUDA_CHECK(cudaFree(d_valid));    CUDA_CHECK(cudaFree(d_in_contact));

    std::printf("[info] verify: %lld mismatches over %d frames (contact mask+morphology, patch stats, marker detect+track)\n",
               total_mismatches, kNumFrames);
    std::printf("[time] GPU pipeline: %.3f ms total, %.4f ms/frame average over %d frames (5 kernels/frame)\n",
               gpu_ms_total, gpu_ms_total / kNumFrames, kNumFrames);
    const bool verify_pass = (total_mismatches == 0);
    std::printf("VERIFY: %s (GPU matches CPU reference EXACTLY on every frame: contact mask+morphology, patch stats, marker detect+track)\n",
               verify_pass ? "PASS" : "FAIL");
    if (!verify_pass) {
        std::printf("RESULT: FAIL (GPU/CPU disagreement - fix before trusting any ground-truth gate)\n");
        return 1;
    }

    // ======================= GROUND-TRUTH GATES ================================
    auto mean = [](const std::vector<double>& v) { double s = 0; for (double x : v) s += x; return v.empty() ? 0.0 : s / v.size(); };
    auto max_of = [](const std::vector<double>& v) { double m = 0; for (double x : v) m = std::max(m, x); return m; };

    const double contact_area_mean_rel = mean(contact_area_rel_err);
    const double contact_area_max_rel = max_of(contact_area_rel_err);
    const double contact_centroid_max = max_of(contact_centroid_err_px);
    const bool gate_contact = (contact_area_max_rel <= kMaxContactAreaRelErr) &&
                              (contact_centroid_max <= kMaxContactCentroidErrPx) &&
                              !contact_area_rel_err.empty();
    std::printf("CONTACT: patch area mean rel err %.1f%% (max %.1f%%) vs Hertzian footprint, centroid max err %.2fpx, over %zu press-hold frames\n",
               contact_area_mean_rel * 100.0, contact_area_max_rel * 100.0, contact_centroid_max, contact_area_rel_err.size());

    const double shear_max_err = max_of(shear_mean_err_px);
    const bool gate_shear = (shear_max_err <= kMaxShearMeanErrPx) && !shear_mean_err_px.empty();
    std::printf("SHEAR: mean tracked displacement max err %.2fpx vs commanded %.1fpx, over %zu shear-hold frames\n",
               shear_max_err, static_cast<double>(kShearTotalPx), shear_mean_err_px.size());

    // Modeled slip onset — computed from the SAME Cattaneo-Mindlin area-
    // fraction formula the scenario is built from, not hardcoded (kernels.cuh
    // "SHEAR / STICK-SLIP MODEL"; area-fraction-in-annulus = 1-(c/a)^2 = 1-(1-s)^(2/3)).
    int modeled_onset_local = -1;
    for (int local = 0; local < kNSlip; ++local) {
        const float s = (kNSlip > 1) ? static_cast<float>(local) / static_cast<float>(kNSlip - 1) : 1.0f;
        const float score_gt = 1.0f - std::pow(1.0f - s, 2.0f / 3.0f);
        if (score_gt >= kSlipScoreDeclareThreshold) { modeled_onset_local = local; break; }
    }
    const int modeled_onset_frame = (modeled_onset_local >= 0) ? kSlipStart + modeled_onset_local : -1;
    const int onset_err = (measured_slip_onset >= 0 && modeled_onset_frame >= 0)
                         ? std::abs(measured_slip_onset - modeled_onset_frame) : 999;
    const bool gate_slip = !false_slip_in_stick_phase && measured_slip_onset >= 0 &&
                           onset_err <= kMaxSlipOnsetErrFrames;
    std::printf("SLIP: detected onset at frame %d (modeled %d, |err|=%d, tol +/-%d); false slip during stick phase (frames %d-%d): %s\n",
               measured_slip_onset, modeled_onset_frame, onset_err, kMaxSlipOnsetErrFrames,
               kBaselineStart, kSlipStart - 1, false_slip_in_stick_phase ? "YES" : "no");

    // ======================= ARTIFACTS ===========================================
    const std::string out_dir = project_root_from(argv[0]) + "/demo/out";
    bool artifact_ok = ensure_dir(out_dir);
    if (artifact_ok) artifact_ok = write_pgm(out_dir + "/contact_mask.pgm", kImgW, kImgH, artifact_mask);
    if (artifact_ok) {
        std::ofstream f(out_dir + "/shear_field.csv");
        artifact_ok = f.is_open();
        if (artifact_ok) {
            f << "marker_col,marker_row,rest_x_px,rest_y_px,measured_dx_px,measured_dy_px,in_contact,valid\n";
            for (int row = 0; row < kMarkerNy; ++row) {
                for (int col = 0; col < kMarkerNx; ++col) {
                    const int i = row * kMarkerNx + col;
                    const float dx = artifact_detected[i].x - artifact_rest[i].x;
                    const float dy = artifact_detected[i].y - artifact_rest[i].y;
                    f << col << ',' << row << ',' << artifact_rest[i].x << ',' << artifact_rest[i].y << ','
                      << dx << ',' << dy << ',' << static_cast<int>(artifact_in_contact[i]) << ','
                      << static_cast<int>(artifact_valid[i]) << '\n';
                }
            }
        }
    }
    if (artifact_ok) {
        std::ofstream f(out_dir + "/slip_timeline.csv");
        artifact_ok = f.is_open();
        if (artifact_ok) {
            f << "frame,phase,slip_score,slip_declared\n";
            for (const SlipRow& row : slip_timeline)
                f << row.frame << ',' << phase_name(row.phase) << ',' << row.score << ',' << (row.declared ? 1 : 0) << '\n';
        }
    }
    if (artifact_ok)
        std::printf("ARTIFACT: wrote demo/out/contact_mask.pgm, demo/out/shear_field.csv, demo/out/slip_timeline.csv\n");
    else
        std::printf("ARTIFACT: FAILED to write demo/out artifacts\n");

    // ---- verdict ----------------------------------------------------------------
    const bool success = artifact_ok && gate_contact && gate_shear && gate_slip;
    if (success) {
        std::printf("RESULT: PASS (contact patch, shear field, and slip-onset ground-truth gates all met)\n");
    } else {
        std::printf("RESULT: FAIL (a ground-truth gate was not met - see CONTACT:/SHEAR:/SLIP: lines above)\n");
    }
    return success ? 0 : 1;
}
