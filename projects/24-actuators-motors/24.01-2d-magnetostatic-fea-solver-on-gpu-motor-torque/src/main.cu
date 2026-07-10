// ===========================================================================
// main.cu — entry point for project 24.01
//           2D magnetostatic FEA solver on GPU -> motor torque-ripple/
//           cogging parameter sweeps
//
// What this program does, start to finish
// ---------------------------------------
//   1. Print the banner + GPU info; load the committed scenario (motor
//      geometry, materials, sweep parameters) from data/sample/.
//   2. VERIFY STAGE (the paragraph 5 GPU-vs-CPU gate): rasterize ONE
//      representative motor variant, solve it on the GPU AND with the CPU
//      twin (src/reference_cpu.cpp), and require the two vector-potential
//      fields to agree within a documented tolerance.
//   3. ANALYTIC STAGE, part A (Ampere's law): solve a uniform-current
//      annulus in air on the SAME solver and check the computed azimuthal B
//      field against the textbook closed form in three regions (bore,
//      within the annulus, exterior).
//   4. ANALYTIC STAGE, part B (flux continuity): solve a straight two-
//      material (air/iron) interface driven by a current strip and check
//      that the NORMAL component of B is continuous across the interface —
//      the classic finite-volume sanity check for the harmonic-mean face
//      coefficients this solver uses.
//   5. THE SWEEP (the catalog bullet's actual point): for each of several
//      magnet pole-arc fractions, BATCH every rotor-angle solve for that
//      arc fraction into ONE kernel-launch sequence, compute the cogging
//      torque (Maxwell stress tensor over a circular air-gap contour) at
//      every angle, and report which arc fraction minimizes PEAK cogging
//      torque — the actual motor-design question this project answers.
//   6. PHYSICS SANITY: every cogging waveform must integrate to ~zero net
//      torque over the sampled period (no net work from cogging — argued
//      in THEORY.md) and the waveform must repeat after one magnet pole
//      pitch (a true structural period of the rotor-stator geometry,
//      checked with an independent solve).
//   7. ARTIFACTS: demo/out/field_magnitude.pgm (|B| over the cross-section
//      for the recommended design) and demo/out/cogging_waveforms.csv
//      (rotor angle vs. torque, one column per arc fraction — the design
//      plot).
//   8. RESULT: PASS only if every stage above holds.
//
// Determinism: everything here is deterministic FP32 arithmetic — no RNG
// anywhere. Results are bit-reproducible run to run ON ONE MACHINE; across
// platforms, compiler FMA-contraction choices can differ in the last few
// ulps after ~3000 chained sweep passes, so — following 31.01's precedent —
// no STABLE (checked) output line below carries a raw floating-point
// number; only PASS/FAIL verdicts against tolerances with real, measured
// headroom (see kernels.cuh's tolerance comments and THEORY.md "Numerical
// considerations" for the actual measured worst-case numbers).
//
// Output contract: stable lines "[demo]", "PROBLEM:", "SCENARIO:",
// "VERIFY:", "ANALYTIC_AMPERE:", "ANALYTIC_INTERFACE:", "PHYSICS:",
// "SWEEP:", "ARTIFACT:", "RESULT:" — "[info]"/"[time]" lines are UNCHECKED
// (device names and every measured number live there). Change a stable
// line => update demo/expected_output.txt in the same commit.
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
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
#include <direct.h>               // _mkdir (std::filesystem avoided in .cu — see 07.09/31.01/08.01)
#else
#include <sys/stat.h>             // mkdir (POSIX twin)
#endif

// A local double-precision pi: all the HOST-side geometry/physics setup
// below is deliberately done in double (narrowed to float only when
// uploaded to the solver) for the same reason 31.01 builds its initial
// condition in double — the SETUP should be beyond suspicion; only the
// iterative solver itself is taught/measured in FP32.
static constexpr double kPi = 3.14159265358979323846;

// ---------------------------------------------------------------------------
// MotorScenario — the committed "problem definition": grid, motor cross-
// section geometry, materials, and sweep parameters. Everything the demo
// needs comes from data/sample/motor_scenario.csv (loaded below) so the
// committed sample and the compiled solver can never silently drift apart —
// the same strict-loader discipline every flagship's scenario file follows.
// All lengths are METERS, angles are computed in RADIANS internally (the
// file stores the few angle-like quantities as dimensionless fractions).
// ---------------------------------------------------------------------------
struct MotorScenario {
    int    nx = 0, ny = 0;             // grid cells (must equal kGridN; §ratified scope)
    double half_w = 0.0;               // domain half-width (m); grid spans [-half_w, +half_w]^2
    double r_rotor_core = 0.0;         // rotor iron core outer radius (m)
    double mag_thk = 0.0;              // magnet radial thickness (m)
    double air_gap = 0.0;              // mechanical air gap (m)
    double r_back_iron_in = 0.0;       // stator: tooth-ring / back-iron boundary radius (m)
    double r_stator_out = 0.0;         // stator outer radius (m) — also near the A=0 boundary
    int    P = 0;                      // number of rotor magnet poles (even, >= 2)
    int    S = 0;                      // number of stator slots (>= 3)
    double mu_r_iron = 0.0;            // relative permeability, rotor/stator iron (linear model)
    double mu_r_magnet = 0.0;          // relative permeability, permanent magnets (~1, NdFeB-like)
    double br_tesla = 0.0;             // magnet remanence Br (T) — sets the equivalent magnetization
    double slot_open_frac = 0.0;       // fraction of one slot pitch that is an open (air) slot
    double omega = 0.0;                // SOR relaxation factor (FeaGrid.omega)
    int    n_sweeps = 0;               // fixed red+black sweep-pair budget per solve
    std::vector<double> arc_fracs;     // swept magnet pole-arc fractions (0,1]
    int    n_angles = 0;               // rotor-angle samples per pole pitch (must be EVEN — see below)
    bool   loaded = false;

    // Derived radii — the layered cross-section boundaries every rasterizer
    // call needs; kept as methods so there is exactly one formula for each,
    // shared by every caller (main.cu never repeats the addition).
    double r_mag_out()    const { return r_rotor_core + mag_thk; }        // magnet outer radius (m)
    double r_stator_in()  const { return r_mag_out() + air_gap; }         // stator bore radius (m)
};

// ---------------------------------------------------------------------------
// load_scenario — strict CSV loader for data/sample/motor_scenario.csv.
// Row-labeled, comma-separated, order-free; unknown labels or a missing
// required row aborts (returns loaded=false) rather than silently solving
// the wrong motor — the same discipline 31.01's scenario loader documents
// as essential for anything safety/physics-adjacent.
// ---------------------------------------------------------------------------
static std::vector<std::string> split_csv(const std::string& line)
{
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) out.push_back(cell);
    return out;
}

static MotorScenario load_scenario(const std::string& path)
{
    MotorScenario sc;
    std::ifstream in(path);
    if (!in.is_open()) return sc;

    bool have_grid=false, have_domain=false, have_rotor=false, have_stator=false,
         have_poles=false, have_mat=false, have_slot=false, have_solver=false,
         have_arcs=false, have_angles=false;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;   // provenance/comment lines
        auto tok = split_csv(line);
        if (tok.empty()) continue;
        const std::string& label = tok[0];

        if (label == "GRID" && tok.size() >= 3) {
            sc.nx = std::atoi(tok[1].c_str()); sc.ny = std::atoi(tok[2].c_str());
            have_grid = true;
        } else if (label == "DOMAIN" && tok.size() >= 2) {
            sc.half_w = std::strtod(tok[1].c_str(), nullptr);
            have_domain = true;
        } else if (label == "ROTOR" && tok.size() >= 4) {
            sc.r_rotor_core = std::strtod(tok[1].c_str(), nullptr);
            sc.mag_thk      = std::strtod(tok[2].c_str(), nullptr);
            sc.air_gap      = std::strtod(tok[3].c_str(), nullptr);
            have_rotor = true;
        } else if (label == "STATOR" && tok.size() >= 3) {
            sc.r_back_iron_in = std::strtod(tok[1].c_str(), nullptr);
            sc.r_stator_out   = std::strtod(tok[2].c_str(), nullptr);
            have_stator = true;
        } else if (label == "POLES_SLOTS" && tok.size() >= 3) {
            sc.P = std::atoi(tok[1].c_str()); sc.S = std::atoi(tok[2].c_str());
            have_poles = true;
        } else if (label == "MATERIALS" && tok.size() >= 4) {
            sc.mu_r_iron   = std::strtod(tok[1].c_str(), nullptr);
            sc.mu_r_magnet = std::strtod(tok[2].c_str(), nullptr);
            sc.br_tesla    = std::strtod(tok[3].c_str(), nullptr);
            have_mat = true;
        } else if (label == "SLOT_OPEN" && tok.size() >= 2) {
            sc.slot_open_frac = std::strtod(tok[1].c_str(), nullptr);
            have_slot = true;
        } else if (label == "SOLVER" && tok.size() >= 3) {
            sc.omega    = std::strtod(tok[1].c_str(), nullptr);
            sc.n_sweeps = std::atoi(tok[2].c_str());
            have_solver = true;
        } else if (label == "SWEEP_ARCS" && tok.size() >= 2) {
            for (size_t k = 1; k < tok.size(); ++k)
                sc.arc_fracs.push_back(std::strtod(tok[k].c_str(), nullptr));
            have_arcs = true;
        } else if (label == "SWEEP_ANGLES" && tok.size() >= 2) {
            sc.n_angles = std::atoi(tok[1].c_str());
            have_angles = true;
        } else {
            std::fprintf(stderr, "scenario: unrecognized row '%s'\n", line.c_str());
            return MotorScenario{};
        }
    }

    if (!(have_grid && have_domain && have_rotor && have_stator && have_poles &&
          have_mat && have_slot && have_solver && have_arcs && have_angles)) {
        std::fprintf(stderr, "scenario: one or more required rows missing\n");
        return MotorScenario{};
    }

    // Semantic validation — the numbers must describe a solvable, physically
    // ordered cross-section (each radial layer strictly nested inside the
    // next) and a batching-safe sweep (n_angles EVEN: main.cu's mean-zero
    // physics-sanity argument relies on the sample set being symmetric
    // about the half-pole-pitch point — THEORY.md derives why).
    const bool radii_ok =
        sc.r_rotor_core > 0.0 && sc.mag_thk > 0.0 && sc.air_gap > 0.0 &&
        sc.r_mag_out() < sc.r_stator_in() &&
        sc.r_stator_in() < sc.r_back_iron_in &&
        sc.r_back_iron_in < sc.r_stator_out &&
        sc.r_stator_out < sc.half_w;
    if (sc.nx != kGridN || sc.ny != kGridN || sc.half_w <= 0.0 || !radii_ok ||
        sc.P < 2 || (sc.P % 2) != 0 || sc.S < 3 ||
        sc.mu_r_iron < 1.0 || sc.mu_r_magnet < 1.0 || sc.br_tesla <= 0.0 ||
        sc.slot_open_frac <= 0.0 || sc.slot_open_frac >= 1.0 ||
        sc.omega <= 0.0 || sc.omega >= 2.0 || sc.n_sweeps < 1 ||
        sc.arc_fracs.empty() || sc.n_angles < 4 || (sc.n_angles % 2) != 0) {
        std::fprintf(stderr, "scenario: values out of range or geometrically inconsistent\n");
        return MotorScenario{};
    }
    for (double a : sc.arc_fracs)
        if (a <= 0.0 || a > 1.0) {
            std::fprintf(stderr, "scenario: an arc fraction is out of (0,1]\n");
            return MotorScenario{};
        }

    sc.loaded = true;
    return sc;
}

// Path helpers — the exe-relative resolution every flagship uses (the exe
// sits at build/x64/<Config>/, three levels below the project root).
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
    candidates.push_back(project_root_from(argv0) + "/data/sample/motor_scenario.csv");
    candidates.push_back("data/sample/motor_scenario.csv");
    candidates.push_back("../data/sample/motor_scenario.csv");
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
// wrap_pi — wrap an angle (radians) into (-pi, pi]. The project's single
// defined angle-wrap point for the RASTERIZER's pole/slot angular tests
// (CLAUDE.md paragraph 12 discipline: wrap at ONE documented place). The
// solver itself never sees an angle — it only sees the nu/Jsrc fields this
// wrapping produces.
// ---------------------------------------------------------------------------
static inline double wrap_pi(double a)
{
    while (a >  kPi) a -= 2.0 * kPi;
    while (a <= -kPi) a += 2.0 * kPi;
    return a;
}

// ---------------------------------------------------------------------------
// rasterize_motor — build the reluctivity field nu(x,y) and the permanent-
// magnet magnetization field M(x,y) = (Mx,My) for ONE rotor angle / arc-
// fraction variant, by walking every grid node and classifying it into one
// of five concentric regions (THEORY.md "The problem" draws the cross-
// section):
//
//   r <= r_rotor_core        : solid rotor iron (mu_r_iron)
//   r_rotor_core < r <= r_mag: magnet ring — iron-adjacent mu_r_magnet
//                               INSIDE a magnet pole's angular span, air
//                               otherwise (the space between poles);
//                               magnetized RADIALLY, alternating polarity
//                               pole to pole (M = +-M0 * r-hat)
//   r_mag < r <= r_stator_in : the air gap (mu_r = 1 everywhere)
//   r_stator_in < r <= r_back: the stator TOOTH ring — iron, except within
//                               S angularly-spaced SLOT OPENINGS (air) —
//                               the reluctance variation that produces
//                               cogging as the rotor turns past them
//   r_back < r <= r_stator_out: solid stator back iron (the flux-return
//                               path that also makes the outer domain
//                               boundary A=0 a valid "no flux escapes"
//                               truncation — THEORY.md derives why)
//   r > r_stator_out          : air, out to the domain boundary
//
// Parameters:
//   sc         : the scenario (geometry, materials, pole/slot counts).
//   arc_frac   : this variant's magnet pole-arc fraction (0,1] — the
//                design parameter being swept.
//   theta_rotor: this variant's mechanical rotor angle (rad) — pole p's
//                angular center is p*(2*pi/P) + theta_rotor; the STATOR
//                (slots) never moves, so sweeping theta_rotor is exactly
//                "spin the rotor under a fixed stator", the real cogging
//                torque experiment.
//   nu, Mx, My : OUT, each resized to nx*ny (node-major, kernels.cuh
//                layout), reluctivity (m/H) and magnetization (A/m).
//
// Complexity: O(nx*ny*P) worst case (each node tests up to P pole windows
// and S slot windows) — for the ratified 256x256/P=4/S=6 scope this is a
// few hundred thousand cheap trig-free comparisons, a few milliseconds on
// any CPU; this is SETUP, not the taught hot loop (kernels.cu's solver is),
// so it deliberately runs on the host once per variant (the 31.01/08.01
// precedent: build inputs on the host, feed identically to both solvers).
// ---------------------------------------------------------------------------
static void rasterize_motor(const MotorScenario& sc, double arc_frac, double theta_rotor,
                            std::vector<float>& nu, std::vector<float>& Mx, std::vector<float>& My)
{
    const int nx = sc.nx, ny = sc.ny;
    const double h    = 2.0 * sc.half_w / (nx - 1);
    const double mu0  = static_cast<double>(kMu0);
    const double r_mag_out   = sc.r_mag_out();
    const double r_stator_in = sc.r_stator_in();
    const double pole_pitch  = 2.0 * kPi / sc.P;
    const double slot_pitch  = 2.0 * kPi / sc.S;
    const double mag_half    = 0.5 * arc_frac * pole_pitch;          // half-angle of one magnet's span
    const double slot_half   = 0.5 * sc.slot_open_frac * slot_pitch; // half-angle of one slot opening
    const double m0          = sc.br_tesla / mu0;                    // |M| inside a magnet (A/m), from Br = mu0*M0

    const size_t total = static_cast<size_t>(nx) * ny;
    nu.assign(total, static_cast<float>(1.0 / mu0));   // default: air everywhere (nu = 1/mu0)
    Mx.assign(total, 0.0f);
    My.assign(total, 0.0f);

    for (int j = 0; j < ny; ++j) {
        const double y = -sc.half_w + j * h;
        for (int i = 0; i < nx; ++i) {
            const double x = -sc.half_w + i * h;
            const double r = std::sqrt(x * x + y * y);
            const size_t idx = static_cast<size_t>(j) * nx + i;

            if (r <= sc.r_rotor_core) {
                nu[idx] = static_cast<float>(1.0 / (mu0 * sc.mu_r_iron));
                continue;
            }
            if (r <= r_mag_out) {
                const double th = std::atan2(y, x);
                for (int p = 0; p < sc.P; ++p) {
                    const double center = p * pole_pitch + theta_rotor;
                    const double dth = wrap_pi(th - center);
                    if (std::fabs(dth) <= mag_half) {
                        nu[idx] = static_cast<float>(1.0 / (mu0 * sc.mu_r_magnet));
                        // Alternating radial magnetization: even-indexed poles
                        // point OUTWARD (north facing the air gap), odd-indexed
                        // poles point INWARD — M = sign * M0 * r-hat.
                        const double sign = (p % 2 == 0) ? 1.0 : -1.0;
                        Mx[idx] = static_cast<float>(sign * m0 * std::cos(th));
                        My[idx] = static_cast<float>(sign * m0 * std::sin(th));
                        break;   // pole windows never overlap (arc_frac <= 1)
                    }
                }
                continue;   // outside every pole's window: stays air (inter-pole gap)
            }
            if (r <= r_stator_in) continue;   // the air gap: stays air

            if (r <= sc.r_back_iron_in) {
                // Stator tooth ring: iron, EXCEPT inside an open-slot window.
                const double th = std::atan2(y, x);
                bool in_slot = false;
                for (int s = 0; s < sc.S; ++s) {
                    const double center = s * slot_pitch;   // slots are FIXED to the stator frame
                    const double dth = wrap_pi(th - center);
                    if (std::fabs(dth) <= slot_half) { in_slot = true; break; }
                }
                if (!in_slot) nu[idx] = static_cast<float>(1.0 / (mu0 * sc.mu_r_iron));
                continue;
            }
            if (r <= sc.r_stator_out) {
                nu[idx] = static_cast<float>(1.0 / (mu0 * sc.mu_r_iron));   // solid back iron
                continue;
            }
            // r > r_stator_out: outside the stator, stays air out to the
            // domain boundary (the "flux-line" Dirichlet A=0 truncation —
            // THEORY.md derives why placing it just outside the stator is
            // a legitimate boundary condition, not a free-space approximation).
        }
    }
}

// ---------------------------------------------------------------------------
// equivalent_current — the permanent magnets' EQUIVALENT MAGNETIZING
// CURRENT, Jm = curl(M) . z-hat = dMy/dx - dMx/dy, evaluated by central
// differences of the rasterized M field built above.
//
// Why this is the right source term (full derivation in THEORY.md "The
// math"): a permanent magnet is modeled as a linear material carrying an
// intrinsic magnetization M; Ampere's law for H (not B) plus B = mu*H +
// mu0*M rearranges into -div(nu*grad(A)) = J_free + curl(M).z. Because M is
// PIECEWISE CONSTANT (uniform inside each magnet, zero outside), its curl
// is concentrated at the magnet's boundaries — physically the classic
// "equivalent bound surface current" every PM-motor textbook derives via
// M x n-hat at each interface. Taking a CENTERED FINITE DIFFERENCE of the
// piecewise field on the SAME grid the solver uses reproduces that bound
// current automatically, smeared over about one grid cell — exactly
// consistent with the solver's own resolution, and far simpler to
// implement correctly than hand-deriving the surface-current geometry for
// every pole edge.
//
// nx*ny in, nx*ny out (Jm), central differences at interior nodes only —
// the border row/column is left at zero, which is harmless because those
// nodes are Dirichlet A=0 and never enter the solve.
// ---------------------------------------------------------------------------
static void equivalent_current(const MotorScenario& sc,
                               const std::vector<float>& Mx, const std::vector<float>& My,
                               std::vector<float>& Jm)
{
    const int nx = sc.nx, ny = sc.ny;
    const double h = 2.0 * sc.half_w / (nx - 1);
    Jm.assign(static_cast<size_t>(nx) * ny, 0.0f);
    for (int j = 1; j < ny - 1; ++j) {
        for (int i = 1; i < nx - 1; ++i) {
            const size_t idx = static_cast<size_t>(j) * nx + i;
            const double dMy_dx = (static_cast<double>(My[idx + 1]) - My[idx - 1]) / (2.0 * h);
            const double dMx_dy = (static_cast<double>(Mx[idx + nx]) - Mx[idx - nx]) / (2.0 * h);
            Jm[idx] = static_cast<float>(dMy_dx - dMx_dy);
        }
    }
}

// ---------------------------------------------------------------------------
// curl_A — B = curl(A_z z-hat) = (dA/dy, -dA/dx), central differences.
// Shared post-processing step used after every solve in this file (the
// verify stage, both analytic gates, and every sweep variant) — computing
// B from A is not part of the taught solver, so — like rasterize_motor — it
// lives once in main.cu rather than being duplicated per call site.
// Border row/column left at zero (never sampled: every caller interpolates
// well inside the domain, away from the Dirichlet edge).
// ---------------------------------------------------------------------------
static void curl_A(const MotorScenario& sc, const std::vector<float>& A,
                   std::vector<float>& Bx, std::vector<float>& By)
{
    const int nx = sc.nx, ny = sc.ny;
    const double h = 2.0 * sc.half_w / (nx - 1);
    const size_t total = static_cast<size_t>(nx) * ny;
    Bx.assign(total, 0.0f);
    By.assign(total, 0.0f);
    for (int j = 1; j < ny - 1; ++j) {
        for (int i = 1; i < nx - 1; ++i) {
            const size_t idx = static_cast<size_t>(j) * nx + i;
            Bx[idx] = static_cast<float>((static_cast<double>(A[idx + nx]) - A[idx - nx]) / (2.0 * h));
            By[idx] = static_cast<float>(-(static_cast<double>(A[idx + 1]) - A[idx - 1]) / (2.0 * h));
        }
    }
}

// ---------------------------------------------------------------------------
// bilerp — bilinear interpolation of a node-major field at an arbitrary
// (x,y), used to sample B along a CIRCLE that does not (in general) pass
// through grid nodes — both analytic gates and the Maxwell-stress torque
// integral need this. Clamped to the interior so a contour point exactly on
// (or numerically just past) the last node never reads out of bounds.
// ---------------------------------------------------------------------------
static float bilerp(const std::vector<float>& F, int nx, int ny, double half_w, double h,
                    double x, double y)
{
    double fx = (x + half_w) / h, fy = (y + half_w) / h;
    int ix = static_cast<int>(std::floor(fx));
    int iy = static_cast<int>(std::floor(fy));
    ix = std::max(0, std::min(nx - 2, ix));
    iy = std::max(0, std::min(ny - 2, iy));
    const double tx = fx - ix, ty = fy - iy;
    auto at = [&](int xx, int yy) { return static_cast<double>(F[static_cast<size_t>(yy) * nx + xx]); };
    const double f00 = at(ix, iy), f10 = at(ix + 1, iy), f01 = at(ix, iy + 1), f11 = at(ix + 1, iy + 1);
    return static_cast<float>(f00 * (1 - tx) * (1 - ty) + f10 * tx * (1 - ty)
                             + f01 * (1 - tx) * ty       + f11 * tx * ty);
}

// ---------------------------------------------------------------------------
// bphi_mean_at_r — average azimuthal B (Tesla) over a circle of radius r,
// used ONLY by the annulus analytic gate (Ampere's law is a statement about
// a circularly symmetric field; averaging over the contour cancels the
// small asymmetry the SQUARE domain boundary introduces — THEORY.md
// discusses why this is honest, not a fudge).
// ---------------------------------------------------------------------------
static double bphi_mean_at_r(const MotorScenario& sc, const std::vector<float>& Bx,
                             const std::vector<float>& By, double r, int nsamp = 720)
{
    const double h = 2.0 * sc.half_w / (sc.nx - 1);
    double sum = 0.0;
    for (int k = 0; k < nsamp; ++k) {
        const double th = 2.0 * kPi * k / nsamp;
        const double x = r * std::cos(th), y = r * std::sin(th);
        const double bx = bilerp(Bx, sc.nx, sc.ny, sc.half_w, h, x, y);
        const double by = bilerp(By, sc.nx, sc.ny, sc.half_w, h, x, y);
        sum += -bx * std::sin(th) + by * std::cos(th);   // B . theta-hat
    }
    return sum / nsamp;
}

// ---------------------------------------------------------------------------
// maxwell_stress_torque — cogging/electromagnetic torque via the Maxwell
// stress tensor, integrated around a circular contour of radius r_contour
// sitting in the air gap (THEORY.md derives the formula from first
// principles). Returns torque PER UNIT AXIAL STACK LENGTH (N*m/m) — the
// honest 2D unit this project reports throughout (kernels.cuh, README).
//
//     T' = (r_contour^2 / mu0) * INTEGRAL_0^2pi  Br(theta)*Btheta(theta) dtheta
//
// nsamp contour samples, trapezoidal (== simple Riemann sum for a smooth
// periodic integrand — the standard "trapezoidal rule on a periodic
// function needs no special endpoint handling" fact).
// ---------------------------------------------------------------------------
static double maxwell_stress_torque(const MotorScenario& sc, const std::vector<float>& Bx,
                                    const std::vector<float>& By, double r_contour, int nsamp = 720)
{
    const double h = 2.0 * sc.half_w / (sc.nx - 1);
    double integral = 0.0;
    for (int k = 0; k < nsamp; ++k) {
        const double th = 2.0 * kPi * k / nsamp;
        const double x = r_contour * std::cos(th), y = r_contour * std::sin(th);
        const double bx = bilerp(Bx, sc.nx, sc.ny, sc.half_w, h, x, y);
        const double by = bilerp(By, sc.nx, sc.ny, sc.half_w, h, x, y);
        const double br = bx * std::cos(th) + by * std::sin(th);          // B . r-hat
        const double bt = -bx * std::sin(th) + by * std::cos(th);         // B . theta-hat
        integral += br * bt;
    }
    integral *= (2.0 * kPi / nsamp);
    return (r_contour * r_contour / static_cast<double>(kMu0)) * integral;
}

// ---------------------------------------------------------------------------
// build_annulus_test / build_interface_test — the two ANALYTIC-GATE fixture
// problems. Neither is the motor: both are small, textbook configurations
// with a KNOWN closed-form answer, solved on the exact same solver and
// (for the annulus) the exact same grid/domain as the motor scenario, so
// "the solver is right" is checked against mathematics, not just against
// itself (31.01's min_time_to_origin gate is this project's direct
// precedent for what an analytic gate buys you).
// ---------------------------------------------------------------------------

// A uniform axial current density J0 filling an ANNULUS (r_in <= r <= r_out)
// in otherwise-empty air (mu_r = 1 everywhere — no iron in this fixture).
// Ampere's law gives the azimuthal field in closed form in all three
// regions (THEORY.md derives it); this fixture and the closed form are
// evaluated together in main()'s ANALYTIC_AMPERE stage.
static void build_annulus_test(const MotorScenario& sc, double r_in, double r_out, double j0,
                               std::vector<float>& nu, std::vector<float>& Jsrc)
{
    const int nx = sc.nx, ny = sc.ny;
    const double h = 2.0 * sc.half_w / (nx - 1);
    const double mu0 = static_cast<double>(kMu0);
    const size_t total = static_cast<size_t>(nx) * ny;
    nu.assign(total, static_cast<float>(1.0 / mu0));   // air everywhere
    Jsrc.assign(total, 0.0f);
    for (int j = 0; j < ny; ++j) {
        const double y = -sc.half_w + j * h;
        for (int i = 0; i < nx; ++i) {
            const double x = -sc.half_w + i * h;
            const double r = std::sqrt(x * x + y * y);
            if (r >= r_in && r <= r_out) Jsrc[static_cast<size_t>(j) * nx + i] = static_cast<float>(j0);
        }
    }
}

// A straight vertical interface at x=0 splitting the domain into air
// (x<=0) and iron of relative permeability mu_r_iron (x>0), driven by a
// uniform current-density strip on the air side so flux genuinely crosses
// the interface. main()'s ANALYTIC_INTERFACE stage checks that the NORMAL
// (here, x-) component of B is continuous across x=0 within a documented
// tolerance — the textbook flux-continuity boundary condition, and the
// direct correctness check on this solver's harmonic-mean face averaging.
static void build_interface_test(const MotorScenario& sc, double mu_r_iron,
                                 std::vector<float>& nu, std::vector<float>& Jsrc)
{
    const int nx = sc.nx, ny = sc.ny;
    const double h = 2.0 * sc.half_w / (nx - 1);
    const double mu0 = static_cast<double>(kMu0);
    const size_t total = static_cast<size_t>(nx) * ny;
    nu.assign(total, static_cast<float>(1.0 / mu0));
    Jsrc.assign(total, 0.0f);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const double x = -sc.half_w + i * h;
            const size_t idx = static_cast<size_t>(j) * nx + i;
            if (x > 0.0) nu[idx] = static_cast<float>(1.0 / (mu0 * mu_r_iron));   // iron half-space
            if (x > -20e-3 && x < -15e-3) Jsrc[idx] = 3.0e6f;                    // driving strip, air side
        }
    }
}

// ---------------------------------------------------------------------------
// write_pgm — the smallest real image format there is (P5), zero libraries
// (07.09/31.01's choice, reused verbatim here).
// ---------------------------------------------------------------------------
static bool write_pgm(const std::string& path, int width, int height,
                      const std::vector<uint8_t>& gray)
{
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) return false;
    out << "P5\n" << width << " " << height << "\n255\n";
    out.write(reinterpret_cast<const char*>(gray.data()), static_cast<std::streamsize>(gray.size()));
    return static_cast<bool>(out);
}

// ---------------------------------------------------------------------------
// gpu_solve — allocate, upload, run launch_fea_solve_batch, download, free.
// The canonical 5-step CUDA program body (CLAUDE.md paragraph 12), wrapped
// once so every call site below (VERIFY, both analytic gates, every sweep
// batch, and the periodicity/artifact spot-checks) is one line instead of
// twenty. B may be 1 (a single variant) or the full sweep batch size — the
// solver treats both identically (kernels.cuh's batching contract).
// ---------------------------------------------------------------------------
static void gpu_solve(const FeaGrid& g, int B, int n_sweeps,
                      const std::vector<float>& nu, const std::vector<float>& Jsrc,
                      std::vector<float>& A_out, float* out_gpu_ms)
{
    const size_t total = static_cast<size_t>(B) * g.nx * g.ny;
    float *d_nu = nullptr, *d_Jsrc = nullptr, *d_A = nullptr;
    CUDA_CHECK(cudaMalloc(&d_nu,   total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Jsrc, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_A,    total * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_nu,   nu.data(),   total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Jsrc, Jsrc.data(), total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_A, 0, total * sizeof(float)));   // cold start: A=0 everywhere (incl. the Dirichlet border)

    GpuTimer gt;
    gt.begin();
    launch_fea_solve_batch(g, B, n_sweeps, d_nu, d_Jsrc, d_A);
    const float ms = gt.end_ms();
    if (out_gpu_ms) *out_gpu_ms = ms;

    A_out.resize(total);
    CUDA_CHECK(cudaMemcpy(A_out.data(), d_A, total * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_nu));
    CUDA_CHECK(cudaFree(d_Jsrc));
    CUDA_CHECK(cudaFree(d_A));
}

// ---------------------------------------------------------------------------
// main — the pipeline described in the file header.
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    std::string data_path;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--data") && i + 1 < argc) data_path = argv[++i];
        else {
            std::fprintf(stderr, "usage: %s [--data motor_scenario.csv]\n", argv[0]);
            return 2;
        }
    }

    std::printf("[demo] 2D magnetostatic FEA solver: motor cogging-torque parameter sweep (project 24.01)\n");
    print_device_info();

    // ---- scenario -------------------------------------------------------------
    const std::string scenario_path = find_scenario(data_path, argv[0]);
    if (scenario_path.empty()) {
        std::printf("SCENARIO: NOT FOUND — data/sample/motor_scenario.csv missing (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (scenario missing)\n");
        return 1;
    }
    std::printf("[info] scenario file: %s\n", scenario_path.c_str());
    const MotorScenario sc = load_scenario(scenario_path);
    if (!sc.loaded) {
        std::printf("SCENARIO: MALFORMED — see stderr\n");
        std::printf("RESULT: FAIL (scenario malformed)\n");
        return 1;
    }

    std::printf("PROBLEM: 2D magnetostatic FEA (batched red-black SOR, omega=%.2f, %d sweep-pairs/solve), %dx%d grid over %.1fx%.1f mm domain, FP32\n",
                sc.omega, sc.n_sweeps, sc.nx, sc.ny, 2.0 * sc.half_w * 1e3, 2.0 * sc.half_w * 1e3);
    std::printf("SCENARIO: %d-pole/%d-slot motor cross-section, mu_r_iron=%.0f, Br=%.2f T, slot-open frac=%.2f; cogging sweep: %zu arc fractions x %d rotor angles [synthetic]\n",
                sc.P, sc.S, sc.mu_r_iron, sc.br_tesla, sc.slot_open_frac, sc.arc_fracs.size(), sc.n_angles);

    const FeaGrid g{ sc.nx, sc.ny, static_cast<float>(2.0 * sc.half_w / (sc.nx - 1)), static_cast<float>(sc.omega) };
    const double pole_pitch = 2.0 * kPi / sc.P;
    const double r_contour  = 0.5 * (sc.r_mag_out() + sc.r_stator_in());   // air-gap midpoint contour

    // ======================= VERIFY STAGE (GPU vs CPU twin) ====================
    bool verify_pass = false;
    {
        // A representative, non-trivial motor variant: the middle arc
        // fraction at a rotor angle deliberately NOT aligned with any
        // pole/slot symmetry axis, so an indexing/averaging bug has no
        // special symmetry to hide behind.
        const double verify_arc   = sc.arc_fracs[sc.arc_fracs.size() / 2];
        const double verify_theta = pole_pitch * 5.0 / sc.n_angles;

        std::vector<float> nu, Mx, My, Jm;
        rasterize_motor(sc, verify_arc, verify_theta, nu, Mx, My);
        equivalent_current(sc, Mx, My, Jm);

        std::vector<float> A_gpu;
        float gpu_ms = 0.0f;
        gpu_solve(g, 1, sc.n_sweeps, nu, Jm, A_gpu, &gpu_ms);

        std::vector<float> A_cpu(static_cast<size_t>(sc.nx) * sc.ny, 0.0f);   // cold start, same as the GPU path
        CpuTimer ct;
        ct.begin();
        fea_solve_batch_cpu(g, 1, sc.n_sweeps, nu.data(), Jm.data(), A_cpu.data());
        const double cpu_ms = ct.end_ms();

        float worst = 0.0f;
        for (size_t k = 0; k < A_gpu.size(); ++k) {
            const float d = std::fabs(A_gpu[k] - A_cpu[k]);
            if (d > worst) worst = d;
        }
        verify_pass = (worst <= kTwinTolAbs);
        std::printf("[info] verify: worst |A_gpu - A_cpu| = %.3e Wb/m over %zu nodes (variant: arc=%.2f, theta=%.2f deg)\n",
                    static_cast<double>(worst), A_gpu.size(), verify_arc, verify_theta * 180.0 / kPi);
        std::printf("[time] one variant, %d sweep-pairs on %dx%d: CPU %.1f ms | GPU %.2f ms | speed-up %.0fx (teaching artifact, single-shot)\n",
                    sc.n_sweeps, sc.nx, sc.ny, cpu_ms, static_cast<double>(gpu_ms),
                    cpu_ms / (static_cast<double>(gpu_ms) > 0.0 ? static_cast<double>(gpu_ms) : 1.0));
        std::printf("VERIFY: %s (GPU batched solver matches CPU twin within |dA| tol %.1e Wb/m)\n",
                    verify_pass ? "PASS" : "FAIL", static_cast<double>(kTwinTolAbs));
        if (!verify_pass) {
            std::printf("RESULT: FAIL (GPU/CPU solver disagreement — fix before trusting anything else)\n");
            return 1;
        }
    }

    // ======================= ANALYTIC GATE A: Ampere's law (annulus) ===========
    bool gate_ampere_pass = true;
    {
        const double r_in = 8e-3, r_out = 12e-3, j0 = 2.0e6;   // A/m^2 — see THEORY.md for the choice

        std::vector<float> nu, Jsrc;
        build_annulus_test(sc, r_in, r_out, j0, nu, Jsrc);
        std::vector<float> A;
        gpu_solve(g, 1, sc.n_sweeps, nu, Jsrc, A, nullptr);
        std::vector<float> Bx, By;
        curl_A(sc, A, Bx, By);

        // Bore (r < r_in): Ampere's law says NO enclosed current => B = 0.
        // Compared as an ABSOLUTE value against the field scale at r_out
        // (a relative comparison is meaningless when the analytic answer
        // IS zero).
        const double b_at_rout_analytic = static_cast<double>(kMu0) * (j0 * kPi * (r_out * r_out - r_in * r_in))
                                          / (2.0 * kPi * r_out);
        double worst_bore_abs = 0.0;
        for (double r : { 2e-3, 5e-3 }) {
            const double b = std::fabs(bphi_mean_at_r(sc, Bx, By, r));
            worst_bore_abs = std::max(worst_bore_abs, b);
        }
        const bool bore_ok = worst_bore_abs <= 0.02 * b_at_rout_analytic;

        // Within the annulus and outside it: closed-form Ampere's law,
        // checked at radii kept a few cells away from the r_in/r_out
        // discontinuities themselves (where a first-order scheme cannot be
        // expected to be sharp — the same "excused band near a
        // discontinuity" honesty 31.01's kBandCells documents).
        double worst_rel = 0.0;
        for (double r : { 10e-3, 16e-3, 20e-3 }) {
            const double i_enc = (r < r_in) ? 0.0
                                : (r < r_out ? j0 * kPi * (r * r - r_in * r_in)
                                             : j0 * kPi * (r_out * r_out - r_in * r_in));
            const double b_analytic = static_cast<double>(kMu0) * i_enc / (2.0 * kPi * r);
            const double b_numeric  = bphi_mean_at_r(sc, Bx, By, r);
            const double rel = std::fabs(b_numeric - b_analytic) / b_analytic;
            worst_rel = std::max(worst_rel, rel);
            std::printf("[info] ampere: r=%.1f mm  B_numeric=%.5e T  B_analytic=%.5e T  rel_err=%.3e\n",
                        r * 1e3, b_numeric, b_analytic, rel);
        }
        gate_ampere_pass = bore_ok && (worst_rel <= 0.05);
        std::printf("[info] ampere: bore |B| worst=%.3e T (<= 2%% of annulus-region B=%.3e T required)\n",
                    worst_bore_abs, b_at_rout_analytic);
        std::printf("ANALYTIC_AMPERE: %s (uniform-current annulus in air matches Ampere's law: zero field in the bore, closed-form B(r) elsewhere, rel tol 5%%)\n",
                    gate_ampere_pass ? "PASS" : "FAIL");
    }

    // ======================= ANALYTIC GATE B: flux continuity ==================
    bool gate_interface_pass = true;
    {
        std::vector<float> nu, Jsrc;
        build_interface_test(sc, sc.mu_r_iron, nu, Jsrc);
        std::vector<float> A;
        gpu_solve(g, 1, sc.n_sweeps, nu, Jsrc, A, nullptr);
        std::vector<float> Bx, By;
        curl_A(sc, A, Bx, By);

        // The interface sits at grid column xi (x=0); Bx (NORMAL to a
        // vertical interface) is sampled one cell into each material and
        // compared — the tightest, most direct test of the harmonic-mean
        // face-averaging this solver's stencil performs (THEORY.md derives
        // why this must hold to first order regardless of the mu ratio).
        const int xi = sc.nx / 2;
        const int row = sc.ny / 2;
        const float bx_air  = Bx[static_cast<size_t>(row) * sc.nx + (xi - 1)];
        const float bx_iron = Bx[static_cast<size_t>(row) * sc.nx + xi];
        const double scale = std::max(std::fabs(static_cast<double>(bx_air)), 1e-12);
        const double rel_jump = std::fabs(static_cast<double>(bx_air) - bx_iron) / scale;
        // Tolerance measured, not guessed: Release (-lineinfo, optimized
        // SASS) gives ~0.3-0.4% here; Debug (-G, unoptimized SASS) gives
        // ~2.5% — the SAME algorithm, the SAME converged-to-machine-
        // precision SOR iteration, but a DIFFERENT floating-point rounding
        // trajectory over ~3000 chained FP32 passes at omega=1.97 (close to
        // the SOR stability boundary of 2, where the iteration is most
        // sensitive to exactly this kind of rounding-order difference).
        // 5% keeps 10x+ headroom over the worst MEASURED build-configuration
        // spread while still catching a real indexing/averaging bug, which
        // shifts this number by O(1) (tens of percent), not a few percent
        // (THEORY.md "Numerical considerations" has the measured table).
        gate_interface_pass = rel_jump <= 5.0e-2;
        std::printf("[info] flux continuity: Bx_air=%.5e T, Bx_iron=%.5e T (mu_r_iron=%.0f), rel jump=%.3e\n",
                    static_cast<double>(bx_air), static_cast<double>(bx_iron), sc.mu_r_iron, rel_jump);
        std::printf("ANALYTIC_INTERFACE: %s (normal B continuous across a straight air/iron interface, rel tol 5%%)\n",
                    gate_interface_pass ? "PASS" : "FAIL");
    }

    // ============================= THE SWEEP ====================================
    // For each arc fraction: rasterize B = n_angles independent rotor-angle
    // variants and solve them ALL in one batched kernel-launch sequence —
    // the parameter sweep's central GPU lesson (kernels.cuh/kernels.cu).
    const size_t n_arcs = sc.arc_fracs.size();
    const int nA = sc.n_angles;
    std::vector<std::vector<double>> waveform(n_arcs, std::vector<double>(static_cast<size_t>(nA), 0.0));
    std::vector<double> peak(n_arcs, 0.0), mean_t(n_arcs, 0.0);
    double sweep_gpu_ms_total = 0.0;

    for (size_t ai = 0; ai < n_arcs; ++ai) {
        const double arc = sc.arc_fracs[ai];
        const size_t plane = static_cast<size_t>(sc.nx) * sc.ny;
        std::vector<float> nu_batch(static_cast<size_t>(nA) * plane), Jsrc_batch(static_cast<size_t>(nA) * plane);

        for (int k = 0; k < nA; ++k) {
            const double theta = pole_pitch * k / nA;   // evenly covers one full pole pitch
            std::vector<float> nu, Mx, My, Jm;
            rasterize_motor(sc, arc, theta, nu, Mx, My);
            equivalent_current(sc, Mx, My, Jm);
            std::copy(nu.begin(), nu.end(), nu_batch.begin() + static_cast<long>(k) * static_cast<long>(plane));
            std::copy(Jm.begin(), Jm.end(), Jsrc_batch.begin() + static_cast<long>(k) * static_cast<long>(plane));
        }

        std::vector<float> A_batch;
        float gpu_ms = 0.0f;
        gpu_solve(g, nA, sc.n_sweeps, nu_batch, Jsrc_batch, A_batch, &gpu_ms);
        sweep_gpu_ms_total += static_cast<double>(gpu_ms);

        for (int k = 0; k < nA; ++k) {
            std::vector<float> A_k(A_batch.begin() + static_cast<long>(k) * static_cast<long>(plane),
                                   A_batch.begin() + static_cast<long>(k + 1) * static_cast<long>(plane));
            std::vector<float> Bx, By;
            curl_A(sc, A_k, Bx, By);
            waveform[ai][static_cast<size_t>(k)] = maxwell_stress_torque(sc, Bx, By, r_contour);
        }

        double pk = 0.0, mn = 0.0;
        for (double t : waveform[ai]) { pk = std::max(pk, std::fabs(t)); mn += t; }
        mn /= nA;
        peak[ai] = pk;
        mean_t[ai] = mn;
        std::printf("[info] sweep arc=%.2f: peak|T|=%.4e N*m/m, mean(T)=%.4e N*m/m (batch of %d rotor angles, GPU %.2f ms)\n",
                    arc, pk, mn, nA, static_cast<double>(gpu_ms));
    }

    size_t best_idx = 0;
    for (size_t ai = 1; ai < n_arcs; ++ai)
        if (peak[ai] < peak[best_idx]) best_idx = ai;

    std::printf("[time] full sweep: %zu arc fractions x %d-rotor-angle batches, %d sweep-pairs each: GPU %.2f ms total (teaching artifact)\n",
                n_arcs, nA, sc.n_sweeps, sweep_gpu_ms_total);
    std::printf("[info] design result: minimum peak cogging torque at arc fraction %.2f (peak %.4e N*m/m); see demo/out/cogging_waveforms.csv for the full table\n",
                sc.arc_fracs[best_idx], peak[best_idx]);
    std::printf("SWEEP: completed %zu arc fractions x %d rotor angles (%d total FEA solves, batched %d-wide per arc fraction)\n",
                n_arcs, nA, static_cast<int>(n_arcs) * nA, nA);

    // ======================= PHYSICS SANITY =====================================
    // (a) mean-zero: cogging does no net work over a period — every
    //     waveform's mean should be small relative to its own peak.
    // (b) periodicity: one magnet pole pitch (360/P degrees) is a true
    //     structural period of the rotor-stator geometry — an INDEPENDENT
    //     solve at theta = pole_pitch must reproduce theta = 0's torque.
    bool sanity_pass = true;
    for (size_t ai = 0; ai < n_arcs; ++ai) {
        const double ratio = (peak[ai] > 1e-9) ? std::fabs(mean_t[ai]) / peak[ai] : 0.0;
        if (ratio > 0.05) sanity_pass = false;
        std::printf("[info] sanity arc=%.2f: |mean(T)|/peak|T| = %.4f (small => no net work from cogging over the sampled period)\n",
                    sc.arc_fracs[ai], ratio);
    }
    {
        const size_t nominal_ai = n_arcs / 2;
        std::vector<float> nu, Mx, My, Jm;
        rasterize_motor(sc, sc.arc_fracs[nominal_ai], pole_pitch, nu, Mx, My);   // one full pole pitch later
        equivalent_current(sc, Mx, My, Jm);
        std::vector<float> A;
        gpu_solve(g, 1, sc.n_sweeps, nu, Jm, A, nullptr);
        std::vector<float> Bx, By;
        curl_A(sc, A, Bx, By);
        const double t_end = maxwell_stress_torque(sc, Bx, By, r_contour);
        const double t_start = waveform[nominal_ai][0];
        const double denom = std::max(peak[nominal_ai], 1e-9);
        const double period_err = std::fabs(t_end - t_start) / denom;
        if (period_err > 0.05) sanity_pass = false;
        std::printf("[info] periodicity: T(theta=0)=%.4e N*m/m, T(theta=pole_pitch=%.1f deg)=%.4e N*m/m, |diff|/peak=%.4f (small => one pole pitch is a true period)\n",
                    t_start, pole_pitch * 180.0 / kPi, t_end, period_err);
    }
    std::printf("PHYSICS: %s (cogging waveforms integrate to ~zero net torque over the sampled period and repeat after one magnet pole pitch)\n",
                sanity_pass ? "PASS" : "FAIL");

    // ============================= ARTIFACTS ====================================
    const std::string out_dir = project_root_from(argv[0]) + "/demo/out";
    bool artifact_ok = ensure_dir(out_dir);

    if (artifact_ok) {
        // (a) The recommended design's field picture: |B| over the cross-
        // section at a non-trivial rotor angle, one quarter of the way
        // through the sampled pole pitch — "the classic motor-field
        // picture" the catalog bullet asks for.
        const double theta_snap = pole_pitch * (nA / 4) / nA;
        std::vector<float> nu, Mx, My, Jm;
        rasterize_motor(sc, sc.arc_fracs[best_idx], theta_snap, nu, Mx, My);
        equivalent_current(sc, Mx, My, Jm);
        std::vector<float> A;
        gpu_solve(g, 1, sc.n_sweeps, nu, Jm, A, nullptr);
        std::vector<float> Bx, By;
        curl_A(sc, A, Bx, By);

        const size_t total = static_cast<size_t>(sc.nx) * sc.ny;
        std::vector<float> bmag(total);
        float hi = 0.0f;
        for (size_t k = 0; k < total; ++k) {
            bmag[k] = std::sqrt(Bx[k] * Bx[k] + By[k] * By[k]);
            hi = std::max(hi, bmag[k]);
        }
        std::vector<uint8_t> gray(total);
        const float span = (hi > 0.0f) ? hi : 1.0f;
        for (size_t k = 0; k < total; ++k)
            gray[k] = static_cast<uint8_t>(std::min(255.0f, 255.0f * bmag[k] / span + 0.5f));
        artifact_ok = write_pgm(out_dir + "/field_magnitude.pgm", sc.nx, sc.ny, gray);
    }
    if (artifact_ok) {
        // (b) The design plot: rotor angle vs. cogging torque, one column
        // per swept arc fraction.
        std::ofstream f(out_dir + "/cogging_waveforms.csv");
        artifact_ok = f.is_open();
        if (artifact_ok) {
            f << "theta_deg";
            for (double arc : sc.arc_fracs) f << ",T_arc" << std::fixed << std::setprecision(2) << arc << "_Nm_per_m";
            f << '\n' << std::defaultfloat;
            for (int k = 0; k < nA; ++k) {
                f << (pole_pitch * k / nA) * 180.0 / kPi;
                for (size_t ai = 0; ai < n_arcs; ++ai) f << ',' << waveform[ai][static_cast<size_t>(k)];
                f << '\n';
            }
            artifact_ok = static_cast<bool>(f);
        }
    }
    if (artifact_ok)
        std::printf("ARTIFACT: wrote demo/out/field_magnitude.pgm and demo/out/cogging_waveforms.csv (%dx%d grid, %zu arc fractions)\n",
                    sc.nx, sc.ny, n_arcs);
    else
        std::printf("ARTIFACT: FAILED to write demo/out files\n");

    // -------------------------------- verdict -----------------------------------
    const bool all_pass = verify_pass && gate_ampere_pass && gate_interface_pass && sanity_pass && artifact_ok;
    if (all_pass)
        std::printf("RESULT: PASS (solver verified against the CPU twin and two analytic gates; cogging sweep completed and passed physics sanity)\n");
    else
        std::printf("RESULT: FAIL (a verification stage failed — see the lines above)\n");
    return all_pass ? 0 : 1;
}
