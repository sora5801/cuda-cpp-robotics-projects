// ===========================================================================
// main.cu — entry point for project 25.01
//           Li-ion electrochemical (SPM) solver + 3D pack thermal simulation
//           + cooling-design sweeps
//
// What this program does, start to finish
// ---------------------------------------
//   1. Print the banner + GPU info; load the committed scenario (electrode
//      params, pack thermal properties, the mission profile, the sweep's h
//      values) from data/sample/.
//   2. VERIFY STAGE (CLAUDE.md §5 GPU-vs-CPU gate): drive ONE design through
//      200 mission steps (20 s) on the GPU kernels AND their CPU twins from
//      an IDENTICAL input sequence, and require the concentration and
//      temperature fields to agree within kernels.cuh's tolerances.
//   3. ANALYTIC_DIFFUSION gate: an isolated single-particle constant-flux
//      diffusion fixture, run to quasi-steady state, checked against the
//      closed-form c_surf - c_avg -> j*R/(5D) (derived in THEORY.md).
//   4. ANALYTIC_COULOMB gate: the SAME fixture's total mole change checked
//      against the exactly-integrable applied flux (charge conservation).
//   5. ANALYTIC_THERMAL gate: an isolated uniform-heat, single-cooling-face
//      thermal fixture run to steady state, checked against the EXACT
//      energy-balance identity P_total = h*A_face*(T_face_avg - T_coolant)
//      (THEORY.md derives why this holds exactly, not approximately).
//   6. THE SWEEP (the catalog bullet's actual point): all 12 cooling designs
//      (6 h values x {bottom, side} cold plate) driven through the SAME
//      20-minute AMR duty-cycle mission profile, BATCHED into one sequence
//      of kernel launches, each design's electro-thermal state evolving
//      independently as its own cooling shapes its own cell temperatures.
//   7. PHYSICS gate: per-design running energy balance (heat generated -
//      heat convected away - thermal energy stored) over the whole mission,
//      checked near-exactly (the scheme is conservative by construction,
//      THEORY.md derives the telescoping argument) for every design.
//   8. ARTIFACTS: demo/out/pack_temps.csv (decimated per-cell T history for
//      the best + worst design), demo/out/design_sweep.csv (the 12-design
//      comparison table), demo/out/pack_slice.pgm (a mid-pack temperature
//      slice of the worst design at its peak).
//   9. RESULT: PASS only if every stage above holds.
//
// Determinism: everything here is deterministic FP32 arithmetic (the
// electrochemistry/thermal PDE state) plus deterministic FP64 host
// bookkeeping (mission profile, OCV/BV/voltage) — no RNG anywhere. Following
// 24.01/31.01's precedent, no STABLE (checked) line below carries a raw
// floating-point number that could shift with compiler FMA-contraction
// choices across platforms; only PASS/FAIL verdicts against tolerances with
// real, measured headroom (see kernels.cuh's tolerance comments).
//
// Output contract: stable lines "[demo]", "PROBLEM:", "SCENARIO:", "VERIFY:",
// "ANALYTIC_DIFFUSION:", "ANALYTIC_COULOMB:", "ANALYTIC_THERMAL:", "SWEEP:",
// "PHYSICS:", "ARTIFACT:", "RESULT:" — "[info]"/"[time]" lines are UNCHECKED.
// Change a stable line => update demo/expected_output.txt in the same commit.
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
#include <direct.h>               // _mkdir (std::filesystem avoided in .cu — 07.09/31.01/24.01/08.01 precedent)
#else
#include <sys/stat.h>
#endif

// Host-side double pi/constants: all SETUP and bookkeeping math below is
// deliberately done in double (narrowed to float only at the PDE-kernel
// boundary) — the same "setup beyond suspicion, only the solver is FP32
// taught/measured" reasoning 24.01's kPi documents.
static constexpr double kPi = 3.14159265358979323846;
static constexpr double kF  = 96485.33212;     // Faraday constant, C/mol (double twin of kFaradayC)
static constexpr double kR  = 8.314462618;     // gas constant, J/(mol K) (double twin of kGasConstJ)
static constexpr double kTrefK = 298.15;       // Arrhenius reference temperature, K

// ---------------------------------------------------------------------------
// PackScenario — the committed "problem definition": electrode particle
// geometry/kinetics, pack thermal properties, the mission's duty-cycle
// segments, and the sweep's h values. Everything the demo needs comes from
// data/sample/pack_scenario.csv (loaded below) — the same strict-loader
// discipline every flagship's scenario file follows (CLAUDE.md §12).
// ---------------------------------------------------------------------------
struct MissionSeg {
    std::string kind;     // human label only ("accelerate"/"cruise"/"idle"/"charge") — not parsed for logic
    double duration_s;    // this segment's length (s)
    double current_A;     // constant CELL current during this segment (A); + = discharge, - = charge
};

struct PackScenario {
    // THERMAL row
    double rho_cp = 0.0, kx = 0.0, ky = 0.0, kz = 0.0, T_init = 0.0, T_coolant = 0.0;
    // CELL_DIMS row (m) — one cell's footprint; tiles exactly into the pack domain
    double cell_Lx = 0.0, cell_Ly = 0.0, cell_Lz = 0.0;
    // ANODE / CATHODE rows
    ElectrodeGeom anode{}, cathode{};
    double anode_c0_frac = 0.0, cathode_c0_frac = 0.0;   // initial stoichiometry (dimensionless, c0/c_max)
    // ELEC row
    double R_ohm = 0.0;
    // MISSION row
    double dt_thermal = 0.0;
    int    n_sub = 0;
    double duration_s = 0.0;
    std::vector<MissionSeg> segs;    // SEG rows, in file order (one duty cycle)
    // SWEEP_H row
    std::vector<double> sweep_h;
    bool loaded = false;

    double cycle_length_s() const {
        double t = 0.0;
        for (const auto& s : segs) t += s.duration_s;
        return t;
    }
};

static std::vector<std::string> split_csv(const std::string& line)
{
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) out.push_back(cell);
    return out;
}

static PackScenario load_scenario(const std::string& path)
{
    PackScenario sc;
    std::ifstream in(path);
    if (!in.is_open()) return sc;

    bool have_thermal = false, have_dims = false, have_anode = false, have_cathode = false,
         have_elec = false, have_mission = false, have_sweep = false;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto tok = split_csv(line);
        if (tok.empty()) continue;
        const std::string& label = tok[0];

        if (label == "THERMAL" && tok.size() >= 7) {
            sc.rho_cp    = std::strtod(tok[1].c_str(), nullptr);
            sc.kx        = std::strtod(tok[2].c_str(), nullptr);
            sc.ky        = std::strtod(tok[3].c_str(), nullptr);
            sc.kz        = std::strtod(tok[4].c_str(), nullptr);
            sc.T_init    = std::strtod(tok[5].c_str(), nullptr);
            sc.T_coolant = std::strtod(tok[6].c_str(), nullptr);
            have_thermal = true;
        } else if (label == "CELL_DIMS" && tok.size() >= 4) {
            sc.cell_Lx = std::strtod(tok[1].c_str(), nullptr);
            sc.cell_Ly = std::strtod(tok[2].c_str(), nullptr);
            sc.cell_Lz = std::strtod(tok[3].c_str(), nullptr);
            have_dims = true;
        } else if ((label == "ANODE" || label == "CATHODE") && tok.size() >= 9) {
            ElectrodeGeom g{};
            g.R_p     = static_cast<float>(std::strtod(tok[1].c_str(), nullptr));
            g.D25     = static_cast<float>(std::strtod(tok[2].c_str(), nullptr));
            g.Ea_D    = static_cast<float>(std::strtod(tok[3].c_str(), nullptr));
            g.c_max   = static_cast<float>(std::strtod(tok[4].c_str(), nullptr));
            // tok[5] is c0_frac — stored separately below (not part of ElectrodeGeom,
            // which is the kernel-facing struct; c0_frac only matters for the
            // initial condition, a main.cu-only concern).
            g.i0_ref  = static_cast<float>(std::strtod(tok[6].c_str(), nullptr));
            g.Ea_k    = static_cast<float>(std::strtod(tok[7].c_str(), nullptr));
            g.A_surf  = static_cast<float>(std::strtod(tok[8].c_str(), nullptr));
            const double c0_frac = std::strtod(tok[5].c_str(), nullptr);
            if (label == "ANODE")   { sc.anode = g;   sc.anode_c0_frac = c0_frac;   have_anode = true; }
            else                     { sc.cathode = g; sc.cathode_c0_frac = c0_frac; have_cathode = true; }
        } else if (label == "ELEC" && tok.size() >= 2) {
            sc.R_ohm = std::strtod(tok[1].c_str(), nullptr);
            have_elec = true;
        } else if (label == "MISSION" && tok.size() >= 4) {
            sc.dt_thermal = std::strtod(tok[1].c_str(), nullptr);
            sc.n_sub      = std::atoi(tok[2].c_str());
            sc.duration_s = std::strtod(tok[3].c_str(), nullptr);
            have_mission = true;
        } else if (label == "SEG" && tok.size() >= 4) {
            MissionSeg seg;
            seg.kind = tok[1];
            seg.duration_s = std::strtod(tok[2].c_str(), nullptr);
            seg.current_A  = std::strtod(tok[3].c_str(), nullptr);
            sc.segs.push_back(seg);
        } else if (label == "SWEEP_H" && tok.size() >= 2) {
            for (size_t k = 1; k < tok.size(); ++k)
                sc.sweep_h.push_back(std::strtod(tok[k].c_str(), nullptr));
            have_sweep = true;
        } else {
            std::fprintf(stderr, "scenario: unrecognized row '%s'\n", line.c_str());
            return PackScenario{};
        }
    }

    if (!(have_thermal && have_dims && have_anode && have_cathode && have_elec &&
          have_mission && have_sweep) || sc.segs.empty()) {
        std::fprintf(stderr, "scenario: one or more required rows missing\n");
        return PackScenario{};
    }
    if (static_cast<int>(sc.sweep_h.size()) != kNSweepH) {
        std::fprintf(stderr, "scenario: SWEEP_H must list exactly %d values\n", kNSweepH);
        return PackScenario{};
    }
    const bool ranges_ok =
        sc.rho_cp > 0 && sc.kx > 0 && sc.ky > 0 && sc.kz > 0 && sc.T_init > 0 && sc.T_coolant > 0 &&
        sc.cell_Lx > 0 && sc.cell_Ly > 0 && sc.cell_Lz > 0 &&
        sc.anode.R_p > 0 && sc.anode.D25 > 0 && sc.anode.c_max > 0 && sc.anode.A_surf > 0 &&
        sc.cathode.R_p > 0 && sc.cathode.D25 > 0 && sc.cathode.c_max > 0 && sc.cathode.A_surf > 0 &&
        sc.anode_c0_frac > 0 && sc.anode_c0_frac < 1 && sc.cathode_c0_frac > 0 && sc.cathode_c0_frac < 1 &&
        sc.R_ohm > 0 && sc.dt_thermal > 0 && sc.n_sub >= 1 && sc.duration_s > 0;
    if (!ranges_ok) {
        std::fprintf(stderr, "scenario: values out of range\n");
        return PackScenario{};
    }
    for (double h : sc.sweep_h)
        if (h <= 0.0) { std::fprintf(stderr, "scenario: sweep h must be positive\n"); return PackScenario{}; }

    sc.loaded = true;
    return sc;
}

// Path helpers — the exe-relative resolution every flagship uses.
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
    candidates.push_back(project_root_from(argv0) + "/data/sample/pack_scenario.csv");
    candidates.push_back("data/sample/pack_scenario.csv");
    candidates.push_back("../data/sample/pack_scenario.csv");
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

// ===========================================================================
// Electrochemistry bookkeeping — pure host math shared by EVERY stage below
// (verify slice, both analytic fixtures, and the real sweep). Not part of
// the GPU-vs-CPU comparison target (only the two PDE kernels are — see
// kernels.cuh) — the same role 24.01's rasterize_motor/curl_A/Maxwell-stress
// functions play: shared, single-sourced, host-only setup/post-processing
// used identically by every downstream consumer.
// ===========================================================================

// SYNTHETIC teaching OCV curves (README §Data / §Limitations: shaped to be
// qualitatively plausible — monotonic, right order of magnitude, graphite's
// characteristic low flat plateau vs. an NMC-like cathode's smoother slope —
// but NOT fit to any real dataset. x is stoichiometry (c_surf/c_max), in [0,1].
static double ocv_cathode(double x)
{
    return 4.2 - 0.6 * x - 0.6 * std::pow(x, 8.0);
}
static double ocv_anode(double x)
{
    return 0.1 + 0.5 * std::exp(-30.0 * x) + 0.05 * (1.0 - x);
}

// arrhenius — scale a rate/diffusivity from the reference temperature Tref
// to T: A(T) = A25 * exp(-Ea/R * (1/T - 1/Tref)). THEORY.md "Numerical
// considerations" derives why this is evaluated in double (T is close to
// Tref, so 1/T - 1/Tref is a small difference of close numbers — the
// classic catastrophic-cancellation hazard float alone would make noisy).
static double arrhenius(double A25, double Ea, double T_kelvin)
{
    return A25 * std::exp(-Ea / kR * (1.0 / T_kelvin - 1.0 / kTrefK));
}

// bv_overpotential — closed-form Butler-Volmer inversion for the symmetric
// (alpha=0.5) case this project uses throughout: i = 2*i0*sinh(F*eta/(2RT))
// inverts exactly to eta = (2RT/F)*asinh(i/(2*i0)) — no Newton iteration
// needed (THEORY.md "The math" derives the inversion).
static double bv_overpotential(double i0, double i_density, double T_kelvin)
{
    const double i0_safe = std::max(i0, 1e-9);   // guard against a pathological i0<=0 (should not occur in-range)
    return (2.0 * kR * T_kelvin / kF) * std::asinh(i_density / (2.0 * i0_safe));
}

// cell_coords — canonical mapping cell index [0,kNCells) <-> (cx,cy,cz),
// the SAME ordering the thermal grid's voxel-block layout uses (kernels.cuh):
// cx fastest, cy next, cz slowest.
static inline void cell_coords(int cell, int& cx, int& cy, int& cz)
{
    cx = cell % kPackNX;
    cy = (cell / kPackNX) % kPackNY;
    cz = cell / (kPackNX * kPackNY);
}

// mission_current — the AMR duty-cycle current (A) at mission time t (s),
// cycling the scenario's SEG list forever (README documents the profile:
// accelerate/cruise/idle/charge, repeating for the mission's duration_s).
static double mission_current(const PackScenario& sc, double t_s)
{
    const double cyc = sc.cycle_length_s();
    double tm = std::fmod(t_s, cyc);
    if (tm < 0.0) tm += cyc;
    for (const auto& seg : sc.segs) {
        if (tm < seg.duration_s) return seg.current_A;
        tm -= seg.duration_s;
    }
    return sc.segs.back().current_A;   // floating-point edge case at tm==cyc
}

// build_thermal_params — the shared pack medium description every design's
// thermal solve uses (only h/face differ per design — kernels.cuh DesignPoint).
static PackThermalParams build_thermal_params(const PackScenario& sc)
{
    PackThermalParams p{};
    p.rho_cp   = static_cast<float>(sc.rho_cp);
    p.kx = static_cast<float>(sc.kx); p.ky = static_cast<float>(sc.ky); p.kz = static_cast<float>(sc.kz);
    p.dx = static_cast<float>(sc.cell_Lx / kVoxPerCellX);
    p.dy = static_cast<float>(sc.cell_Ly / kVoxPerCellY);
    p.dz = static_cast<float>(sc.cell_Lz / kVoxPerCellZ);
    p.T_coolant = static_cast<float>(sc.T_coolant);
    return p;
}

// thermal_cfl_margin — how far dt_thermal sits below the explicit-FTCS
// stability bound (THEORY.md "Numerical considerations" derives the bound:
// dt <= rho_cp / (2*(kx/dx^2 + ky/dy^2 + kz/dz^2))). Returned as a ratio
// dt_max/dt_used (>1 means stable, with that much headroom) — main() prints
// and gates on this BEFORE running a single step, the same "check the CFL
// before trusting the run" discipline 31.01's kCfl documents.
static double thermal_cfl_margin(const PackScenario& sc, const PackThermalParams& p)
{
    const double sum = static_cast<double>(p.kx) / (static_cast<double>(p.dx) * p.dx)
                     + static_cast<double>(p.ky) / (static_cast<double>(p.dy) * p.dy)
                     + static_cast<double>(p.kz) / (static_cast<double>(p.dz) * p.dz);
    const double dt_max = static_cast<double>(p.rho_cp) / (2.0 * sum);
    return dt_max / sc.dt_thermal;
}

// ---------------------------------------------------------------------------
// StepBookkeeping — per-(design,cell) results of ONE thermal-step's worth of
// electrochemical bookkeeping (OCV/BV/voltage/heat), computed from that
// cell's CURRENT surface stoichiometry and temperature. Shared verbatim by
// the verify slice and the real sweep (the only difference between them is
// how many designs/cells they loop over and where the PDE state itself lives).
// ---------------------------------------------------------------------------
struct StepBookkeeping {
    std::vector<float> D;        // [B*kNCells*2] Arrhenius-scaled diffusivity, THIS step (particle layout)
    std::vector<double> V_cell;  // [B*kNCells] terminal voltage (V)
    std::vector<double> q_cell;  // [B*kNCells] heat generation (W)
};

// compute_bookkeeping — the electro-thermal coupling itself: OCV + Butler-
// Volmer overpotential (closed-form) + ohmic drop -> terminal voltage, and
// I*(OCV_cell - V_cell) -> irreversible heat (THEORY.md "The math" derives
// every line). j_a/j_c are the SAME for every cell (the shared-current
// simplification kernels.cuh documents); only D and the overpotentials vary
// per cell, through that cell's own temperature.
static void compute_bookkeeping(const PackScenario& sc, int B,
                                double I_cmd_A, double j_a, double j_c,
                                const std::vector<float>& x_a,  // [B*kNCells] surface stoichiometry, anode
                                const std::vector<float>& x_c,  // [B*kNCells] surface stoichiometry, cathode
                                const std::vector<float>& T_cell, // [B*kNCells] K
                                StepBookkeeping& out)
{
    out.D.assign(static_cast<size_t>(B) * kNCells * 2, 0.0f);
    out.V_cell.assign(static_cast<size_t>(B) * kNCells, 0.0);
    out.q_cell.assign(static_cast<size_t>(B) * kNCells, 0.0);

    const double i_a = kF * j_a;     // reaction current density at the anode (A/m^2)
    const double i_c = kF * j_c;     // reaction current density at the cathode (A/m^2)

    for (int b = 0; b < B; ++b) {
        for (int cell = 0; cell < kNCells; ++cell) {
            const int bc = b * kNCells + cell;
            const double T = static_cast<double>(T_cell[static_cast<size_t>(bc)]);
            const double xa = std::min(0.999, std::max(0.001, static_cast<double>(x_a[static_cast<size_t>(bc)])));
            const double xc = std::min(0.999, std::max(0.001, static_cast<double>(x_c[static_cast<size_t>(bc)])));

            const double Da = arrhenius(sc.anode.D25,   sc.anode.Ea_D,   T);
            const double Dc = arrhenius(sc.cathode.D25, sc.cathode.Ea_D, T);
            const int p_a = bc * 2 + 0, p_c = bc * 2 + 1;   // particle-layout indices (kernels.cuh)
            out.D[static_cast<size_t>(p_a)] = static_cast<float>(Da);
            out.D[static_cast<size_t>(p_c)] = static_cast<float>(Dc);

            // Exchange current densities: Arrhenius-scaled prefactor times
            // the SPM shape factor sqrt(x*(1-x)) (kernels.cuh ElectrodeGeom
            // documents why — vanishes as an electrode approaches full/empty,
            // the qualitatively-correct BV behavior without an electrolyte
            // concentration state).
            const double i0_a = arrhenius(sc.anode.i0_ref,   sc.anode.Ea_k,   T) * std::sqrt(xa * (1.0 - xa));
            const double i0_c = arrhenius(sc.cathode.i0_ref, sc.cathode.Ea_k, T) * std::sqrt(xc * (1.0 - xc));

            const double eta_a = bv_overpotential(i0_a, i_a, T);
            const double eta_c = bv_overpotential(i0_c, i_c, T);

            const double ocv_a = ocv_anode(xa);
            const double ocv_c = ocv_cathode(xc);
            const double v_ocv = ocv_c - ocv_a;
            // Terminal voltage: OCV difference, PLUS the two overpotentials
            // with their natural signs (eta_c<0, eta_a>0 during discharge —
            // see kernels.cuh's flux sign-convention derivation), MINUS the
            // ohmic drop I*R_ohm.
            const double v_cell = v_ocv + eta_c - eta_a - I_cmd_A * sc.R_ohm;
            // Heat = I*(OCV - V): the IRREVERSIBLE heat only (ohmic + BV
            // activation losses); the reversible entropic term dS/dx*T*I/F
            // is a documented omission (README §Limitations — it needs an
            // entropy-coefficient curve this teaching model does not carry).
            const double q = I_cmd_A * (v_ocv - v_cell);

            out.V_cell[static_cast<size_t>(bc)] = v_cell;
            out.q_cell[static_cast<size_t>(bc)] = q;
        }
    }
}

// extract_surface_stoich — turn a compact [B*kNCells*2] array of surface
// (outermost-shell) concentrations into separate anode/cathode stoichiometry
// arrays x = c_surf/c_max, [B*kNCells] each. The compact array's layout
// mirrors the particle index p=(b*kNCells+cell)*2+e (kernels.cuh).
static void extract_surface_stoich(const PackScenario& sc, int B,
                                   const std::vector<float>& c_surf_compact,
                                   std::vector<float>& x_a, std::vector<float>& x_c)
{
    x_a.assign(static_cast<size_t>(B) * kNCells, 0.0f);
    x_c.assign(static_cast<size_t>(B) * kNCells, 0.0f);
    for (int bc = 0; bc < B * kNCells; ++bc) {
        x_a[static_cast<size_t>(bc)] = c_surf_compact[static_cast<size_t>(bc) * 2 + 0] / sc.anode.c_max;
        x_c[static_cast<size_t>(bc)] = c_surf_compact[static_cast<size_t>(bc) * 2 + 1] / sc.cathode.c_max;
    }
}

// extract_cell_temps — this step's representative temperature per (design,
// cell), taken as the CENTER voxel of that cell's kVoxPerCellX x Y x Z
// block (documented simplification, kernels.cuh header — a single scalar
// stands in for the whole block's internal gradient, which is small at
// this project's Biot numbers; THEORY.md discusses the approximation).
static void extract_cell_temps(int B, const std::vector<float>& T_full, std::vector<float>& T_cell)
{
    T_cell.assign(static_cast<size_t>(B) * kNCells, 0.0f);
    for (int b = 0; b < B; ++b) {
        for (int cell = 0; cell < kNCells; ++cell) {
            int cx, cy, cz;
            cell_coords(cell, cx, cy, cz);
            const int i = cx * kVoxPerCellX + kVoxPerCellX / 2;
            const int j = cy * kVoxPerCellY + kVoxPerCellY / 2;
            const int k = cz * kVoxPerCellZ + kVoxPerCellZ / 2;
            const size_t idx = ((static_cast<size_t>(b) * kTNZ + k) * kTNY + j) * kTNX + i;
            T_cell[static_cast<size_t>(b) * kNCells + cell] = T_full[idx];
        }
    }
}

// build_designs — the 12-point cooling sweep: designs [0,kNSweepH) are
// bottom-cooled at the scenario's 6 h values, designs [kNSweepH,2*kNSweepH)
// are side-cooled at the SAME 6 h values (README documents this ordering).
static std::vector<DesignPoint> build_designs(const PackScenario& sc)
{
    std::vector<DesignPoint> d;
    d.reserve(kNDesigns);
    for (int hIdx = 0; hIdx < kNSweepH; ++hIdx)
        d.push_back(DesignPoint{ static_cast<float>(sc.sweep_h[static_cast<size_t>(hIdx)]), kCoolBottomZ });
    for (int hIdx = 0; hIdx < kNSweepH; ++hIdx)
        d.push_back(DesignPoint{ static_cast<float>(sc.sweep_h[static_cast<size_t>(hIdx)]), kCoolSideX });
    return d;
}

// build_heat_source — spread each (design,cell)'s heat generation q_cell [W]
// UNIFORMLY over that cell's kVoxPerCellX*Y*Z voxel block, as a volumetric
// source [W/m^3] (main.cu's electro-thermal coupling: this step's chemistry
// becomes NEXT step's thermal source term — THEORY.md discusses the
// resulting one-step lag and why it is negligible at this dt).
static void build_heat_source(const PackScenario& sc, int B,
                              const std::vector<double>& q_cell, std::vector<float>& q_vol)
{
    q_vol.assign(static_cast<size_t>(B) * kTNZ * kTNY * kTNX, 0.0f);
    const double cell_vol = sc.cell_Lx * sc.cell_Ly * sc.cell_Lz;   // m^3, one cell's block volume
    for (int b = 0; b < B; ++b) {
        for (int cell = 0; cell < kNCells; ++cell) {
            const float qdens = static_cast<float>(q_cell[static_cast<size_t>(b) * kNCells + cell] / cell_vol);
            int cx, cy, cz;
            cell_coords(cell, cx, cy, cz);
            for (int dk = 0; dk < kVoxPerCellZ; ++dk) {
                const int k = cz * kVoxPerCellZ + dk;
                for (int dj = 0; dj < kVoxPerCellY; ++dj) {
                    const int j = cy * kVoxPerCellY + dj;
                    const size_t rowBase = ((static_cast<size_t>(b) * kTNZ + k) * kTNY + j) * kTNX
                                          + static_cast<size_t>(cx) * kVoxPerCellX;
                    for (int di = 0; di < kVoxPerCellX; ++di)
                        q_vol[rowBase + di] = qdens;
                }
            }
        }
    }
}

// ===========================================================================
// SweepDiagnostics — everything main() needs, per design, to answer "which
// cooling design keeps the pack balanced" plus the running energy-balance
// bookkeeping the PHYSICS gate checks.
// ===========================================================================
struct SweepDiagnostics {
    std::vector<double> peak_T;          // [B] K — max cell temperature ever reached
    std::vector<double> max_spread;      // [B] K — max (hottest-coldest cell) ever reached
    std::vector<double> initial_V_avg;   // [B] V — pack-average cell voltage at t=0
    std::vector<double> final_V_avg;     // [B] V — pack-average cell voltage at t=duration
    std::vector<double> final_V_spread;  // [B] V — hottest-minus-coldest cell VOLTAGE at t=duration
    std::vector<double> energy_in_J;     // [B] — running sum of generated heat, J
    std::vector<double> energy_out_J;    // [B] — running sum of heat convected to coolant, J
    std::vector<double> energy_stored_J; // [B] — final total thermal energy above T_init, J
    std::vector<float> log_time_s;                    // decimated mission-time samples (s)
    std::vector<std::vector<float>> log_cellT;         // [B][nSamples*kNCells] decimated per-cell T (K)
    std::vector<std::vector<float>> T_at_own_peak;     // [B][kTNZ*kTNY*kTNX] full-field snapshot at THIS design's peak step
    std::vector<float> T_final;          // [B*kTNZ*kTNY*kTNX] field at mission end
};

// ===========================================================================
// run_sweep_gpu — the mission loop, GPU-driven, for B batched designs. This
// is the catalog bullet's actual point: B independent (h, face) designs,
// advanced together every step by the two batched kernels, diverging only
// because their different cooling shapes their cells' temperatures
// differently (kernels.cuh's electro-thermal coupling).
//
// Per-step orchestration (documented HERE rather than re-explained inline,
// since main() also uses this exact sequence for the verify slice):
//   1. read this step's applied current from the mission profile;
//   2. read the CURRENT (start-of-step) surface stoichiometry and cell
//      temperature back from the device (a cudaMemcpy2D strided read for
//      the concentration surface — 20x less traffic than copying whole
//      particles — and a full-array read for temperature, since a scattered
//      per-cell gather would cost more round-trips than it saves at this
//      array size; README Exercise 5 profiles a gather-kernel alternative);
//   3. run the bookkeeping (OCV/BV/voltage/heat, and this step's Arrhenius-
//      scaled D) — pure host math, shared with every other stage;
//   4. spread the resulting per-cell heat into a volumetric source, upload
//      it and this step's D array, and advance BOTH PDEs by one step
//      (n_sub electrochemistry substeps, one thermal step) — this ordering
//      is a documented LAGGED coupling: this step's thermal source uses
//      chemistry computed from the START of the step, one dt behind the
//      thermal state it feeds (THEORY.md "Numerical considerations" bounds
//      the resulting error, negligible at dt_thermal=0.1s against both
//      PDEs' much longer time constants).
// ===========================================================================
static void run_sweep_gpu(const PackScenario& sc, const std::vector<DesignPoint>& designs,
                          SweepDiagnostics& diag, double* out_gpu_ms)
{
    const int B = static_cast<int>(designs.size());
    const PackThermalParams thermalP = build_thermal_params(sc);
    const size_t nParticles = static_cast<size_t>(B) * kNCells * 2;
    const size_t cSize = nParticles * kNShells;
    const size_t tPlane = static_cast<size_t>(kTNZ) * kTNY * kTNX;
    const size_t tSize = static_cast<size_t>(B) * tPlane;

    // ---- host initial conditions: uniform stoichiometry, uniform T_init ----
    std::vector<float> c_host(cSize);
    for (int bc = 0; bc < B * kNCells; ++bc) {
        const float c0a = static_cast<float>(sc.anode_c0_frac * sc.anode.c_max);
        const float c0c = static_cast<float>(sc.cathode_c0_frac * sc.cathode.c_max);
        for (int s = 0; s < kNShells; ++s) {
            c_host[static_cast<size_t>(bc) * 2 * kNShells + s] = c0a;
            c_host[static_cast<size_t>(bc) * 2 * kNShells + kNShells + s] = c0c;
        }
    }
    std::vector<float> T_host(tSize, static_cast<float>(sc.T_init));

    // ---- persistent device buffers (allocated ONCE outside the hot loop —
    // the 08.01 precedent for a call made many thousands of times) ----------
    float *d_c[2] = { nullptr, nullptr };
    float *d_T[2] = { nullptr, nullptr };
    float *d_D = nullptr, *d_qvol = nullptr;
    DesignPoint* d_designs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_c[0], cSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c[1], cSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_T[0], tSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_T[1], tSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_D, nParticles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_qvol, tSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_designs, static_cast<size_t>(B) * sizeof(DesignPoint)));
    CUDA_CHECK(cudaMemcpy(d_c[0], c_host.data(), cSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_T[0], T_host.data(), tSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_designs, designs.data(), static_cast<size_t>(B) * sizeof(DesignPoint), cudaMemcpyHostToDevice));
    int c_cur = 0, T_cur = 0;

    diag = SweepDiagnostics{};
    diag.peak_T.assign(static_cast<size_t>(B), sc.T_init);
    diag.max_spread.assign(static_cast<size_t>(B), 0.0);
    diag.initial_V_avg.assign(static_cast<size_t>(B), 0.0);
    diag.final_V_avg.assign(static_cast<size_t>(B), 0.0);
    diag.final_V_spread.assign(static_cast<size_t>(B), 0.0);
    diag.energy_in_J.assign(static_cast<size_t>(B), 0.0);
    diag.energy_out_J.assign(static_cast<size_t>(B), 0.0);
    diag.T_at_own_peak.assign(static_cast<size_t>(B), std::vector<float>(tPlane, static_cast<float>(sc.T_init)));
    diag.log_cellT.assign(static_cast<size_t>(B), {});

    const int nSteps = static_cast<int>(std::llround(sc.duration_s / sc.dt_thermal));
    const double dt_e = sc.dt_thermal / sc.n_sub;
    const int logStride = 60;   // every 60 steps = 6 s at dt_thermal=0.1s -> 200 samples over 1200s

    const double areaBottom = static_cast<double>(thermalP.dx) * thermalP.dy;   // one voxel's bottom-face area (m^2)
    const double areaSide   = static_cast<double>(thermalP.dy) * thermalP.dz;   // one voxel's side-face area (m^2)

    double gpu_ms_total = 0.0;
    std::vector<float> c_surf_compact(nParticles), x_a, x_c, T_cell, q_vol_host;
    StepBookkeeping bk;

    for (int step = 0; step < nSteps; ++step) {
        const double t = static_cast<double>(step) * sc.dt_thermal;
        const double I_cmd = mission_current(sc, t);
        const double j_a =  I_cmd / (kF * sc.anode.A_surf);
        const double j_c = -I_cmd / (kF * sc.cathode.A_surf);

        // Strided D2H: pull ONLY the outermost shell of every particle
        // (kernels.cuh layout: shell s is the fastest axis, so the surface
        // shell sits at a constant kNShells-1 offset within every particle's
        // block) — 2*B*kNCells floats instead of the full B*kNCells*2*kNShells.
        CUDA_CHECK(cudaMemcpy2D(c_surf_compact.data(), sizeof(float),
                                d_c[c_cur] + (kNShells - 1), kNShells * sizeof(float),
                                sizeof(float), nParticles, cudaMemcpyDeviceToHost));
        extract_surface_stoich(sc, B, c_surf_compact, x_a, x_c);

        CUDA_CHECK(cudaMemcpy(T_host.data(), d_T[T_cur], tSize * sizeof(float), cudaMemcpyDeviceToHost));
        extract_cell_temps(B, T_host, T_cell);

        compute_bookkeeping(sc, B, I_cmd, j_a, j_c, x_a, x_c, T_cell, bk);

        if (step == 0) {
            for (int b = 0; b < B; ++b) {
                double sum = 0.0;
                for (int cell = 0; cell < kNCells; ++cell) sum += bk.V_cell[static_cast<size_t>(b) * kNCells + cell];
                diag.initial_V_avg[static_cast<size_t>(b)] = sum / kNCells;
            }
        }

        build_heat_source(sc, B, bk.q_cell, q_vol_host);

        // Energy bookkeeping for the PHYSICS gate — independent host-side
        // accounting of what THIS step generates and convects away, using
        // the SAME start-of-step values the PDE update itself consumes.
        for (int b = 0; b < B; ++b) {
            double qsum = 0.0;
            for (int cell = 0; cell < kNCells; ++cell) qsum += bk.q_cell[static_cast<size_t>(b) * kNCells + cell];
            diag.energy_in_J[static_cast<size_t>(b)] += qsum * sc.dt_thermal;

            double qout = 0.0;
            const DesignPoint dpz = designs[static_cast<size_t>(b)];
            if (dpz.face == kCoolBottomZ) {
                for (int j = 0; j < kTNY; ++j)
                    for (int i = 0; i < kTNX; ++i) {
                        const size_t idx = (static_cast<size_t>(b) * kTNZ + 0) * kTNY * kTNX + static_cast<size_t>(j) * kTNX + i;
                        qout += dpz.h * (static_cast<double>(T_host[idx]) - sc.T_coolant) * areaBottom;
                    }
            } else {
                for (int k = 0; k < kTNZ; ++k)
                    for (int j = 0; j < kTNY; ++j) {
                        const size_t idx = (static_cast<size_t>(b) * kTNZ + k) * kTNY * kTNX + static_cast<size_t>(j) * kTNX + 0;
                        qout += dpz.h * (static_cast<double>(T_host[idx]) - sc.T_coolant) * areaSide;
                    }
            }
            diag.energy_out_J[static_cast<size_t>(b)] += qout * sc.dt_thermal;
        }

        // Peak / spread tracking + decimated logging.
        for (int b = 0; b < B; ++b) {
            double mn = 1e18, mx = -1e18;
            for (int cell = 0; cell < kNCells; ++cell) {
                const double Tv = T_cell[static_cast<size_t>(b) * kNCells + cell];
                mn = std::min(mn, Tv); mx = std::max(mx, Tv);
            }
            diag.max_spread[static_cast<size_t>(b)] = std::max(diag.max_spread[static_cast<size_t>(b)], mx - mn);
            if (mx > diag.peak_T[static_cast<size_t>(b)]) {
                diag.peak_T[static_cast<size_t>(b)] = mx;
                std::copy(T_host.begin() + static_cast<long>(b) * static_cast<long>(tPlane),
                         T_host.begin() + static_cast<long>(b + 1) * static_cast<long>(tPlane),
                         diag.T_at_own_peak[static_cast<size_t>(b)].begin());
            }
        }
        if (step % logStride == 0) {
            diag.log_time_s.push_back(static_cast<float>(t));
            for (int b = 0; b < B; ++b)
                for (int cell = 0; cell < kNCells; ++cell)
                    diag.log_cellT[static_cast<size_t>(b)].push_back(T_cell[static_cast<size_t>(b) * kNCells + cell]);
        }

        // Upload this step's D and heat source, then advance both PDEs.
        CUDA_CHECK(cudaMemcpy(d_D, bk.D.data(), nParticles * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_qvol, q_vol_host.data(), tSize * sizeof(float), cudaMemcpyHostToDevice));

        GpuTimer gt;
        gt.begin();
        for (int ss = 0; ss < sc.n_sub; ++ss) {
            launch_electrochem_substep(B, sc.anode, sc.cathode,
                                       static_cast<float>(j_a), static_cast<float>(j_c), static_cast<float>(dt_e),
                                       d_D, d_c[c_cur], d_c[1 - c_cur]);
            c_cur = 1 - c_cur;
        }
        launch_thermal_substep(B, thermalP, static_cast<float>(sc.dt_thermal), d_designs, d_qvol, d_T[T_cur], d_T[1 - T_cur]);
        T_cur = 1 - T_cur;
        gpu_ms_total += static_cast<double>(gt.end_ms());
    }

    // ---- final state + end-of-mission voltage bookkeeping -----------------
    CUDA_CHECK(cudaMemcpy(T_host.data(), d_T[T_cur], tSize * sizeof(float), cudaMemcpyDeviceToHost));
    diag.T_final = T_host;
    for (int b = 0; b < B; ++b) {
        double sumStored = 0.0;
        for (size_t v = 0; v < tPlane; ++v) sumStored += (static_cast<double>(T_host[static_cast<size_t>(b) * tPlane + v]) - sc.T_init);
        const double voxVol = static_cast<double>(thermalP.dx) * thermalP.dy * thermalP.dz;
        diag.energy_stored_J.push_back(sumStored * thermalP.rho_cp * voxVol);
    }

    CUDA_CHECK(cudaMemcpy2D(c_surf_compact.data(), sizeof(float),
                            d_c[c_cur] + (kNShells - 1), kNShells * sizeof(float),
                            sizeof(float), nParticles, cudaMemcpyDeviceToHost));
    extract_surface_stoich(sc, B, c_surf_compact, x_a, x_c);
    extract_cell_temps(B, T_host, T_cell);
    const double I_final = mission_current(sc, sc.duration_s);
    const double j_a_f =  I_final / (kF * sc.anode.A_surf);
    const double j_c_f = -I_final / (kF * sc.cathode.A_surf);
    compute_bookkeeping(sc, B, I_final, j_a_f, j_c_f, x_a, x_c, T_cell, bk);
    for (int b = 0; b < B; ++b) {
        double sum = 0.0, mn = 1e18, mx = -1e18;
        for (int cell = 0; cell < kNCells; ++cell) {
            const double v = bk.V_cell[static_cast<size_t>(b) * kNCells + cell];
            sum += v; mn = std::min(mn, v); mx = std::max(mx, v);
        }
        diag.final_V_avg[static_cast<size_t>(b)] = sum / kNCells;
        diag.final_V_spread[static_cast<size_t>(b)] = mx - mn;
    }

    CUDA_CHECK(cudaFree(d_c[0])); CUDA_CHECK(cudaFree(d_c[1]));
    CUDA_CHECK(cudaFree(d_T[0])); CUDA_CHECK(cudaFree(d_T[1]));
    CUDA_CHECK(cudaFree(d_D));    CUDA_CHECK(cudaFree(d_qvol));
    CUDA_CHECK(cudaFree(d_designs));

    if (out_gpu_ms) *out_gpu_ms = gpu_ms_total;
}

// ===========================================================================
// run_verify_slice — the CLAUDE.md §5 GPU-vs-CPU gate. ONE design, a small
// number of mission steps (not the full 20-minute/12-design sweep — running
// the CPU twin over that would take minutes on a single core and defeats
// the point of measuring a speed-up), GPU and CPU integrated INDEPENDENTLY
// from the same initial state, each step's inputs (j_a/j_c/D/q_vol) derived
// from the CPU trajectory's own state so both paths see IDENTICAL inputs
// every step — any divergence at the end is attributable ONLY to the two
// PDE implementations (kernels.cu vs. reference_cpu.cpp), not to different
// driving data.
// ===========================================================================
static bool run_verify_slice(const PackScenario& sc, int nSteps,
                             double* out_gpu_ms, double* out_cpu_ms,
                             float* out_conc_dev, float* out_temp_dev)
{
    const int B = 1;
    const std::vector<DesignPoint> designs = { DesignPoint{ static_cast<float>(sc.sweep_h[0]), kCoolBottomZ } };
    const PackThermalParams thermalP = build_thermal_params(sc);
    const size_t nParticles = static_cast<size_t>(B) * kNCells * 2;
    const size_t cSize = nParticles * kNShells;
    const size_t tSize = static_cast<size_t>(B) * kTNZ * kTNY * kTNX;

    std::vector<float> c_cpu[2], T_cpu[2];
    c_cpu[0].assign(cSize, 0.0f); c_cpu[1].assign(cSize, 0.0f);
    T_cpu[0].assign(tSize, static_cast<float>(sc.T_init)); T_cpu[1].assign(tSize, static_cast<float>(sc.T_init));
    for (int bc = 0; bc < B * kNCells; ++bc) {
        const float c0a = static_cast<float>(sc.anode_c0_frac * sc.anode.c_max);
        const float c0c = static_cast<float>(sc.cathode_c0_frac * sc.cathode.c_max);
        for (int s = 0; s < kNShells; ++s) {
            c_cpu[0][static_cast<size_t>(bc) * 2 * kNShells + s] = c0a;
            c_cpu[0][static_cast<size_t>(bc) * 2 * kNShells + kNShells + s] = c0c;
        }
    }
    int cc = 0, cT = 0;   // CPU ping-pong indices

    float *d_c[2] = { nullptr, nullptr }, *d_T[2] = { nullptr, nullptr }, *d_D = nullptr, *d_qvol = nullptr;
    DesignPoint* d_designs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_c[0], cSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c[1], cSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_T[0], tSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_T[1], tSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_D, nParticles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_qvol, tSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_designs, sizeof(DesignPoint)));
    CUDA_CHECK(cudaMemcpy(d_c[0], c_cpu[0].data(), cSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_T[0], T_cpu[0].data(), tSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_designs, designs.data(), sizeof(DesignPoint), cudaMemcpyHostToDevice));
    int gc = 0, gT = 0;   // GPU ping-pong indices

    const double dt_e = sc.dt_thermal / sc.n_sub;
    double gpu_ms = 0.0, cpu_ms = 0.0;
    std::vector<float> x_a(static_cast<size_t>(B) * kNCells), x_c(static_cast<size_t>(B) * kNCells), T_cell, q_vol_host;
    StepBookkeeping bk;

    for (int step = 0; step < nSteps; ++step) {
        const double t = static_cast<double>(step) * sc.dt_thermal;
        const double I_cmd = mission_current(sc, t);
        const double j_a =  I_cmd / (kF * sc.anode.A_surf);
        const double j_c = -I_cmd / (kF * sc.cathode.A_surf);

        // Bookkeeping inputs come from the CPU trajectory's own state
        // (direct host indexing — no device round trip needed here).
        for (int bc = 0; bc < B * kNCells; ++bc) {
            const int p_a = bc * 2 + 0, p_c = bc * 2 + 1;
            x_a[static_cast<size_t>(bc)] = c_cpu[cc][static_cast<size_t>(p_a) * kNShells + (kNShells - 1)] / sc.anode.c_max;
            x_c[static_cast<size_t>(bc)] = c_cpu[cc][static_cast<size_t>(p_c) * kNShells + (kNShells - 1)] / sc.cathode.c_max;
        }
        extract_cell_temps(B, T_cpu[cT], T_cell);
        compute_bookkeeping(sc, B, I_cmd, j_a, j_c, x_a, x_c, T_cell, bk);
        build_heat_source(sc, B, bk.q_cell, q_vol_host);

        // CPU step.
        CpuTimer ct; ct.begin();
        for (int ss = 0; ss < sc.n_sub; ++ss) {
            electrochem_fv_cpu(B, sc.anode, sc.cathode, static_cast<float>(j_a), static_cast<float>(j_c),
                               static_cast<float>(dt_e), bk.D.data(), c_cpu[cc].data(), c_cpu[1 - cc].data());
            cc = 1 - cc;
        }
        thermal_step_cpu(B, thermalP, static_cast<float>(sc.dt_thermal), designs.data(), q_vol_host.data(),
                         T_cpu[cT].data(), T_cpu[1 - cT].data());
        cT = 1 - cT;
        cpu_ms += ct.end_ms();

        // GPU step, SAME inputs.
        CUDA_CHECK(cudaMemcpy(d_D, bk.D.data(), nParticles * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_qvol, q_vol_host.data(), tSize * sizeof(float), cudaMemcpyHostToDevice));
        GpuTimer gt; gt.begin();
        for (int ss = 0; ss < sc.n_sub; ++ss) {
            launch_electrochem_substep(B, sc.anode, sc.cathode, static_cast<float>(j_a), static_cast<float>(j_c),
                                       static_cast<float>(dt_e), d_D, d_c[gc], d_c[1 - gc]);
            gc = 1 - gc;
        }
        launch_thermal_substep(B, thermalP, static_cast<float>(sc.dt_thermal), d_designs, d_qvol, d_T[gT], d_T[1 - gT]);
        gT = 1 - gT;
        gpu_ms += static_cast<double>(gt.end_ms());
    }

    std::vector<float> c_gpu_final(cSize), T_gpu_final(tSize);
    CUDA_CHECK(cudaMemcpy(c_gpu_final.data(), d_c[gc], cSize * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(T_gpu_final.data(), d_T[gT], tSize * sizeof(float), cudaMemcpyDeviceToHost));

    float worst_c = 0.0f, worst_T = 0.0f;
    for (size_t k = 0; k < cSize; ++k) worst_c = std::max(worst_c, std::fabs(c_gpu_final[k] - c_cpu[cc][k]));
    for (size_t k = 0; k < tSize; ++k) worst_T = std::max(worst_T, std::fabs(T_gpu_final[k] - T_cpu[cT][k]));

    CUDA_CHECK(cudaFree(d_c[0])); CUDA_CHECK(cudaFree(d_c[1]));
    CUDA_CHECK(cudaFree(d_T[0])); CUDA_CHECK(cudaFree(d_T[1]));
    CUDA_CHECK(cudaFree(d_D));    CUDA_CHECK(cudaFree(d_qvol));
    CUDA_CHECK(cudaFree(d_designs));

    if (out_gpu_ms) *out_gpu_ms = gpu_ms;
    if (out_cpu_ms) *out_cpu_ms = cpu_ms;
    *out_conc_dev = worst_c;
    *out_temp_dev = worst_T;
    return (worst_c <= kTwinTolConc) && (worst_T <= kTwinTolTemp);
}

// ===========================================================================
// DiffusionFixtureResult — output of the standalone single-particle,
// constant-flux fixture that BOTH analytic electrochemistry gates read
// (i: the quasi-steady surface-minus-average closed form; ii: Coulomb
// counting / mass conservation). Run once, checked twice — the same
// "one fixture, several checks" economy 24.01's annulus test shows.
// ===========================================================================
struct DiffusionFixtureResult {
    double c_surf_minus_avg = 0.0;     // measured, mol/m^3
    double quasi_steady_expected = 0.0; // (-j)*R_p/(5D), mol/m^3 (THEORY.md derives the sign)
    double moles_measured_delta = 0.0;  // moles_final - moles_initial
    double moles_implied_delta = 0.0;   // -j * A_particle * total_time (exact, by construction — THEORY.md)
};

// run_diffusion_fixture — an ISOLATED sphere (not the pack, not the mission
// current — a standalone textbook diffusion problem run on the SAME FV
// scheme kernels.cu implements) under a constant surface flux, integrated
// long enough (several times the particle's own diffusion time constant
// R_p^2/D) that the concentration PROFILE reaches its quasi-steady shape
// even as its average keeps drifting (THEORY.md "The math" derives both
// closed forms this fixture checks against).
static DiffusionFixtureResult run_diffusion_fixture(const PackScenario& sc)
{
    DiffusionFixtureResult res;
    const ElectrodeGeom geom = sc.anode;              // reuse the anode's real particle geometry
    const double D = static_cast<double>(geom.D25);   // T = Tref throughout -> Arrhenius factor 1, D = D25 exactly
    const double R_p = static_cast<double>(geom.R_p);
    const double j = -2.0e-6;                          // mol/(m^2 s), NEGATIVE = insertion (kernels.cuh sign convention)
    const double dt_e = 0.1;                            // s — well inside the diffusion CFL (THEORY.md derives the bound)
    // MEASURED (a standalone Python re-implementation of this exact FV
    // scheme, THEORY.md "How we verify correctness"): the quasi-steady
    // SHAPE is reached almost immediately (t = 2*R_p^2/D already matches
    // t = 80*R_p^2/D to 4 significant figures) — the remaining ~12% gap
    // from the closed form is NOT a convergence-in-time issue, it is an
    // O(1/kNShells) FINITE-VOLUME DISCRETIZATION bias (measured: 10 shells
    // -> 23%, 20 shells -> 12%, 40 shells -> 6% — halves each time shells
    // double, the textbook signature of a first-order boundary-representation
    // error: the outermost/innermost shells are CENTERED half a shell
    // inside the true r=0/r=R_p, so their values run slightly behind the
    // continuum profile's actual endpoints). kNShells=20 is this project's
    // ratified scope (CLAUDE.md's catalog bullet says "~20 shells"); the
    // gate's tolerance below is set from this MEASURED bias, not guessed.
    const double total_time = 3.0 * (R_p * R_p / D);    // 3x relaxation time constant — ample given the measurement above
    const int nSteps = static_cast<int>(std::llround(total_time / dt_e));

    const int B = 1;   // electrochem_fv_cpu always advances kNCells*2 particles; every one gets IDENTICAL
                        // geometry/flux here, so they are 48 exact copies of the same 1-particle problem —
                        // only particle 0 is inspected below.
    const size_t nParticles = static_cast<size_t>(B) * kNCells * 2;
    const size_t cSize = nParticles * kNShells;
    const double c0 = 0.5 * static_cast<double>(geom.c_max);   // mid-range start: safe headroom either direction

    std::vector<float> c[2];
    c[0].assign(cSize, static_cast<float>(c0));
    c[1].assign(cSize, 0.0f);
    std::vector<float> Darr(nParticles, static_cast<float>(D));
    int cur = 0;
    for (int s = 0; s < nSteps; ++s) {
        electrochem_fv_cpu(B, geom, geom, static_cast<float>(j), static_cast<float>(j), static_cast<float>(dt_e),
                           Darr.data(), c[cur].data(), c[1 - cur].data());
        cur = 1 - cur;
    }

    // Particle 0's shells: recompute the FV shell volumes (mirrors kernels.cu
    // exactly) to integrate the true volume-weighted average.
    const double dr = R_p / kNShells;
    double moles0 = 0.0, moles_final = 0.0, vol_total = 0.0;
    for (int s = 0; s < kNShells; ++s) {
        const double r_in = s * dr, r_out = (s + 1) * dr;
        const double V_s = (4.0 / 3.0) * kPi * (r_out * r_out * r_out - r_in * r_in * r_in);
        vol_total += V_s;
        moles0 += c0 * V_s;
        moles_final += static_cast<double>(c[cur][static_cast<size_t>(s)]) * V_s;
    }
    const double c_avg_final = moles_final / vol_total;
    const double c_surf_final = static_cast<double>(c[cur][kNShells - 1]);

    res.c_surf_minus_avg = c_surf_final - c_avg_final;
    res.quasi_steady_expected = (-j) * R_p / (5.0 * D);   // THEORY.md derives this sign for our "leaving-positive" j
    res.moles_measured_delta = moles_final - moles0;
    const double A_particle = 4.0 * kPi * R_p * R_p;       // this ONE particle's own surface area (not the scenario's A_surf)
    res.moles_implied_delta = -j * A_particle * (nSteps * dt_e);  // exact by the telescoping argument, THEORY.md
    return res;
}

// ===========================================================================
// ThermalFixtureResult / run_thermal_fixture — an isolated, UNIFORMLY heated
// pack-shaped domain with a single cooling face, run to (near) steady state.
// At true steady state, energy conservation is EXACT regardless of internal
// gradients: every watt generated must leave through the one active face,
// so P_total = h*A_face*(T_face_avg - T_coolant) holds exactly — not a
// lumped-Biot APPROXIMATION but a genuine closed-form consequence of global
// energy balance (THEORY.md "How we verify correctness" derives this).
// Fixture-only thermal properties (NOT the scenario's) are used so the
// domain reaches steady state in a tractable number of explicit steps —
// documented, not hidden (see the comment at the call site).
// ===========================================================================
struct ThermalFixtureResult {
    double dT_measured = 0.0;    // T_face_avg - T_coolant at the end of the run, K
    double dT_expected = 0.0;    // q0*domain_volume/(h*A_face), K
    double residual_ratio = 0.0; // |final total dT/dt| relative to the steady-state rate scale — how converged we are
};

static ThermalFixtureResult run_thermal_fixture(const PackScenario& sc)
{
    ThermalFixtureResult res;
    PackThermalParams p{};
    p.rho_cp = 4.0e4f;                              // FIXTURE-ONLY value (much smaller than the real pack's
                                                     // ~2e6): shrinks the thermal time constant so steady state
                                                     // is reached in a tractable number of EXPLICIT steps — see
                                                     // the CFL comment below for why this stays stable.
    p.kx = p.ky = p.kz = 3.0f;                       // isotropic here: this fixture tests the BOUNDARY term, not anisotropy
    p.dx = static_cast<float>(sc.cell_Lx / kVoxPerCellX);
    p.dy = static_cast<float>(sc.cell_Ly / kVoxPerCellY);
    p.dz = static_cast<float>(sc.cell_Lz / kVoxPerCellZ);
    p.T_coolant = static_cast<float>(sc.T_coolant);

    const double h = 100.0;
    const std::vector<DesignPoint> designs = { DesignPoint{ static_cast<float>(h), kCoolBottomZ } };
    const double q0 = 4000.0;   // W/m^3, uniform — chosen so the expected steady rise is a visible ~10 K (see file header math)
    const double dt = 0.1;      // matches the real scenario's dt_thermal; CFL-checked below like the real run is

    const size_t tSize = static_cast<size_t>(kTNZ) * kTNY * kTNX;
    std::vector<float> T[2];
    T[0].assign(tSize, static_cast<float>(sc.T_coolant));
    T[1].assign(tSize, 0.0f);
    std::vector<float> qvol(tSize, static_cast<float>(q0));

    const int nSteps = 12000;   // ~1.56x the fixture's own thermal time constant (rho_cp*Lz^2/kz) — long enough to converge
    int cur = 0;
    for (int s = 0; s < nSteps; ++s) {
        thermal_step_cpu(1, p, static_cast<float>(dt), designs.data(), qvol.data(), T[cur].data(), T[1 - cur].data());
        cur = 1 - cur;
    }

    double sumFace = 0.0;
    for (int j = 0; j < kTNY; ++j)
        for (int i = 0; i < kTNX; ++i)
            sumFace += static_cast<double>(T[cur][static_cast<size_t>(j) * kTNX + i]);   // k=0 plane
    const double T_face_avg = sumFace / (kTNX * kTNY);
    res.dT_measured = T_face_avg - sc.T_coolant;

    const double domainVol = static_cast<double>(kPackNX) * sc.cell_Lx
                            * static_cast<double>(kPackNY) * sc.cell_Ly
                            * static_cast<double>(kPackNZ) * sc.cell_Lz;
    const double areaFace = static_cast<double>(kPackNX) * sc.cell_Lx * static_cast<double>(kPackNY) * sc.cell_Ly;
    res.dT_expected = q0 * domainVol / (h * areaFace);

    // Convergence residual: compare the last step's max |dT| to the
    // expected temperature scale — small means "close enough to steady".
    double maxDT = 0.0;
    for (size_t k = 0; k < tSize; ++k) maxDT = std::max(maxDT, std::fabs(static_cast<double>(T[cur][k] - T[1 - cur][k])));
    res.residual_ratio = maxDT / std::max(res.dT_expected, 1e-9);
    return res;
}

// ---------------------------------------------------------------------------
// write_pgm — the smallest real image format there is (P5), zero libraries
// (07.09/24.01/31.01's shared choice, reused verbatim here).
// ---------------------------------------------------------------------------
static bool write_pgm(const std::string& path, int width, int height, const std::vector<uint8_t>& gray)
{
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) return false;
    out << "P5\n" << width << " " << height << "\n255\n";
    out.write(reinterpret_cast<const char*>(gray.data()), static_cast<std::streamsize>(gray.size()));
    return static_cast<bool>(out);
}

// ===========================================================================
// main — the pipeline described in the file header.
// ===========================================================================
int main(int argc, char** argv)
{
    std::string data_path;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--data") && i + 1 < argc) data_path = argv[++i];
        else {
            std::fprintf(stderr, "usage: %s [--data pack_scenario.csv]\n", argv[0]);
            return 2;
        }
    }

    std::printf("[demo] Li-ion SPM electrochemistry + 3D pack thermal cooling-design sweep (project 25.01)\n");
    print_device_info();

    // ---- scenario -----------------------------------------------------------
    const std::string scenario_path = find_scenario(data_path, argv[0]);
    if (scenario_path.empty()) {
        std::printf("SCENARIO: NOT FOUND — data/sample/pack_scenario.csv missing (run scripts/make_synthetic.py?)\n");
        std::printf("RESULT: FAIL (scenario missing)\n");
        return 1;
    }
    std::printf("[info] scenario file: %s\n", scenario_path.c_str());
    const PackScenario sc = load_scenario(scenario_path);
    if (!sc.loaded) {
        std::printf("SCENARIO: MALFORMED — see stderr\n");
        std::printf("RESULT: FAIL (scenario malformed)\n");
        return 1;
    }

    std::printf("PROBLEM: SPM electrochemistry (%d radial shells/particle) + anisotropic 3D pack heat equation "
               "(%dx%dx%d voxels, %d cells), %d-design cooling sweep, FP32\n",
               kNShells, kTNX, kTNY, kTNZ, kNCells, kNDesigns);
    std::printf("SCENARIO: %.0f s AMR duty-cycle mission (%zu segments/cycle), dt_thermal=%.2f s, n_sub=%d, "
               "sweep h in [%.0f, %.0f] W/(m^2 K) x {bottom,side} [synthetic]\n",
               sc.duration_s, sc.segs.size(), sc.dt_thermal, sc.n_sub,
               sc.sweep_h.front(), sc.sweep_h.back());

    const PackThermalParams thermalP = build_thermal_params(sc);
    const double cflMargin = thermal_cfl_margin(sc, thermalP);
    std::printf("[info] thermal explicit-FTCS CFL: dt_thermal=%.3f s, stability bound=%.3f s, margin=%.1fx\n",
               sc.dt_thermal, sc.dt_thermal * cflMargin, cflMargin);
    if (cflMargin < 1.0) {
        std::printf("RESULT: FAIL (thermal timestep violates the explicit-FTCS CFL bound — see [info] line)\n");
        return 1;
    }

    // ======================= VERIFY STAGE (GPU vs CPU twin) ====================
    bool verify_pass = false;
    {
        double gpu_ms = 0.0, cpu_ms = 0.0;
        float worst_c = 0.0f, worst_T = 0.0f;
        const int nVerifySteps = 200;   // 20 s of mission (spans the accelerate segment and into cruise)
        verify_pass = run_verify_slice(sc, nVerifySteps, &gpu_ms, &cpu_ms, &worst_c, &worst_T);
        std::printf("[info] verify: worst |c_gpu-c_cpu|=%.3e mol/m^3 (tol %.1e), worst |T_gpu-T_cpu|=%.3e K (tol %.1e), over %d steps\n",
                   static_cast<double>(worst_c), static_cast<double>(kTwinTolConc),
                   static_cast<double>(worst_T), static_cast<double>(kTwinTolTemp), nVerifySteps);
        std::printf("[time] verify slice (1 design, %d steps): CPU %.1f ms | GPU %.2f ms | speed-up %.1fx (teaching artifact)\n",
                   nVerifySteps, cpu_ms, gpu_ms, cpu_ms / std::max(gpu_ms, 1e-6));
        std::printf("VERIFY: %s (GPU electrochemistry+thermal solvers match their CPU twins within documented tolerance)\n",
                   verify_pass ? "PASS" : "FAIL");
        if (!verify_pass) {
            std::printf("RESULT: FAIL (GPU/CPU solver disagreement — fix before trusting anything else)\n");
            return 1;
        }
    }

    // ======================= ANALYTIC GATES A+B: diffusion fixture =============
    bool gate_diffusion_pass = false, gate_coulomb_pass = false;
    {
        const DiffusionFixtureResult r = run_diffusion_fixture(sc);
        const double relDiff = std::fabs(r.c_surf_minus_avg - r.quasi_steady_expected) / std::fabs(r.quasi_steady_expected);
        // Tolerance 15%: MEASURED headroom over the ~12.1% O(1/kNShells)
        // finite-volume discretization bias documented at this fixture's
        // definition above (a real, understood, shell-count-limited effect,
        // not a bug — it shrinks to ~6% at 40 shells) — 15% catches an
        // actual indexing/sign/coefficient bug (which shifts this number by
        // order 1, not by a few points) while honestly accommodating the
        // scheme's real resolution at the ratified kNShells=20.
        gate_diffusion_pass = relDiff <= 0.15;
        std::printf("[info] diffusion fixture: c_surf-c_avg=%.4f mol/m^3, quasi-steady closed form=%.4f mol/m^3, rel_err=%.4f\n",
                   r.c_surf_minus_avg, r.quasi_steady_expected, relDiff);
        std::printf("ANALYTIC_DIFFUSION: %s (constant-flux sphere's quasi-steady c_surf-c_avg matches the closed form j*R/(5D), rel tol 15%% — measured kNShells=20 discretization bias ~12%%)\n",
                   gate_diffusion_pass ? "PASS" : "FAIL");

        const double scale = std::max(std::fabs(r.moles_implied_delta), 1e-30);
        const double relCoulomb = std::fabs(r.moles_measured_delta - r.moles_implied_delta) / scale;
        gate_coulomb_pass = relCoulomb <= 1.0e-2;
        std::printf("[info] coulomb counting: measured delta-moles=%.6e mol, implied (integrated flux)=%.6e mol, rel_err=%.4e\n",
                   r.moles_measured_delta, r.moles_implied_delta, relCoulomb);
        std::printf("ANALYTIC_COULOMB: %s (total mole change equals the exactly-integrated applied flux, rel tol 1%%)\n",
                   gate_coulomb_pass ? "PASS" : "FAIL");
    }

    // ======================= ANALYTIC GATE C: thermal lumped energy balance ====
    bool gate_thermal_pass = false;
    {
        const ThermalFixtureResult r = run_thermal_fixture(sc);
        const double relErr = std::fabs(r.dT_measured - r.dT_expected) / std::max(r.dT_expected, 1e-9);
        gate_thermal_pass = relErr <= 0.05;
        std::printf("[info] thermal fixture: dT_measured=%.4f K, dT_expected (P/(h*A))=%.4f K, rel_err=%.4f, convergence residual=%.4f\n",
                   r.dT_measured, r.dT_expected, relErr, r.residual_ratio);
        std::printf("ANALYTIC_THERMAL: %s (steady uniform-heat, single-cooling-face pack matches the exact energy-balance identity P=h*A*dT, rel tol 5%%)\n",
                   gate_thermal_pass ? "PASS" : "FAIL");
    }

    // ============================= THE SWEEP ====================================
    const std::vector<DesignPoint> designs = build_designs(sc);
    SweepDiagnostics diag;
    double sweep_gpu_ms = 0.0;
    {
        GpuTimer wallish; // not used for the stable line, just an [info] figure via CpuTimer instead
        CpuTimer wall; wall.begin();
        run_sweep_gpu(sc, designs, diag, &sweep_gpu_ms);
        const double wall_ms = wall.end_ms();
        std::printf("[time] full sweep: %d designs x %.0f s mission: GPU kernel time %.1f ms, wall time %.1f ms (teaching artifact)\n",
                   kNDesigns, sc.duration_s, sweep_gpu_ms, wall_ms);
    }

    int best = 0, worst = 0;
    for (int b = 1; b < kNDesigns; ++b) {
        if (diag.peak_T[static_cast<size_t>(b)] < diag.peak_T[static_cast<size_t>(best)]) best = b;
        if (diag.peak_T[static_cast<size_t>(b)] > diag.peak_T[static_cast<size_t>(worst)]) worst = b;
    }
    std::printf("[info] design result: BEST (lowest peak T) = design %d (h=%.0f W/m^2K, face=%s): peak=%.2f K, spread=%.3f K, final V spread=%.4f V\n",
               best, designs[static_cast<size_t>(best)].h, designs[static_cast<size_t>(best)].face == kCoolBottomZ ? "bottom" : "side",
               diag.peak_T[static_cast<size_t>(best)], diag.max_spread[static_cast<size_t>(best)], diag.final_V_spread[static_cast<size_t>(best)]);
    std::printf("[info] design result: WORST (highest peak T) = design %d (h=%.0f W/m^2K, face=%s): peak=%.2f K, spread=%.3f K, final V spread=%.4f V\n",
               worst, designs[static_cast<size_t>(worst)].h, designs[static_cast<size_t>(worst)].face == kCoolBottomZ ? "bottom" : "side",
               diag.peak_T[static_cast<size_t>(worst)], diag.max_spread[static_cast<size_t>(worst)], diag.final_V_spread[static_cast<size_t>(worst)]);
    std::printf("SWEEP: completed %d cooling designs (%d h-values x 2 faces) over the full %.0f s mission, batched per step\n",
               kNDesigns, kNSweepH, sc.duration_s);

    // ======================= PHYSICS: energy conservation =======================
    bool physics_pass = true;
    double worstEnergyRelErr = 0.0;
    for (int b = 0; b < kNDesigns; ++b) {
        const double predicted = diag.energy_in_J[static_cast<size_t>(b)] - diag.energy_out_J[static_cast<size_t>(b)];
        const double scale = std::max(std::fabs(diag.energy_in_J[static_cast<size_t>(b)]), 1.0);
        const double relErr = std::fabs(predicted - diag.energy_stored_J[static_cast<size_t>(b)]) / scale;
        worstEnergyRelErr = std::max(worstEnergyRelErr, relErr);
    }
    physics_pass = worstEnergyRelErr <= 0.02;
    std::printf("[info] physics: worst-case (over 12 designs) |((heat_in - heat_out) - stored)| / heat_in = %.4f\n", worstEnergyRelErr);
    std::printf("PHYSICS: %s (thermal grid energy balance — generated minus convected minus stored — holds for every design, rel tol 2%%)\n",
               physics_pass ? "PASS" : "FAIL");

    // ============================= ARTIFACTS ====================================
    const std::string out_dir = project_root_from(argv[0]) + "/demo/out";
    bool artifact_ok = ensure_dir(out_dir);

    if (artifact_ok) {
        std::ofstream f(out_dir + "/design_sweep.csv");
        artifact_ok = f.is_open();
        if (artifact_ok) {
            f << "design_id,h_W_m2K,face,peak_T_K,max_spread_K,final_voltage_spread_V,initial_V_avg_V,final_V_avg_V\n";
            for (int b = 0; b < kNDesigns; ++b) {
                f << b << ',' << designs[static_cast<size_t>(b)].h << ','
                  << (designs[static_cast<size_t>(b)].face == kCoolBottomZ ? "bottom" : "side") << ','
                  << diag.peak_T[static_cast<size_t>(b)] << ',' << diag.max_spread[static_cast<size_t>(b)] << ','
                  << diag.final_V_spread[static_cast<size_t>(b)] << ',' << diag.initial_V_avg[static_cast<size_t>(b)] << ','
                  << diag.final_V_avg[static_cast<size_t>(b)] << '\n';
            }
            artifact_ok = static_cast<bool>(f);
        }
    }

    if (artifact_ok) {
        std::ofstream f(out_dir + "/pack_temps.csv");
        artifact_ok = f.is_open();
        if (artifact_ok) {
            f << "t_s";
            for (int cell = 0; cell < kNCells; ++cell) f << ",best_cell" << std::setfill('0') << std::setw(2) << cell << "_T_K";
            for (int cell = 0; cell < kNCells; ++cell) f << ",worst_cell" << std::setfill('0') << std::setw(2) << cell << "_T_K";
            f << '\n' << std::setfill(' ');
            const size_t nSamples = diag.log_time_s.size();
            for (size_t s = 0; s < nSamples; ++s) {
                f << diag.log_time_s[s];
                for (int cell = 0; cell < kNCells; ++cell) f << ',' << diag.log_cellT[static_cast<size_t>(best)][s * kNCells + static_cast<size_t>(cell)];
                for (int cell = 0; cell < kNCells; ++cell) f << ',' << diag.log_cellT[static_cast<size_t>(worst)][s * kNCells + static_cast<size_t>(cell)];
                f << '\n';
            }
            artifact_ok = static_cast<bool>(f);
        }
    }

    if (artifact_ok) {
        // A mid-pack (k = kTNZ/2) temperature slice of the WORST design at
        // ITS OWN peak-temperature step — the visual "which design runs hot" picture.
        const int kMid = kTNZ / 2;
        std::vector<uint8_t> gray(static_cast<size_t>(kTNX) * kTNY);
        const std::vector<float>& field = diag.T_at_own_peak[static_cast<size_t>(worst)];
        float lo = field[static_cast<size_t>(kMid) * kTNY * kTNX], hi = lo;
        for (int j = 0; j < kTNY; ++j)
            for (int i = 0; i < kTNX; ++i) {
                const float v = field[(static_cast<size_t>(kMid) * kTNY + j) * kTNX + i];
                lo = std::min(lo, v); hi = std::max(hi, v);
            }
        const float span = (hi > lo) ? (hi - lo) : 1.0f;
        for (int j = 0; j < kTNY; ++j)
            for (int i = 0; i < kTNX; ++i) {
                const float v = field[(static_cast<size_t>(kMid) * kTNY + j) * kTNX + i];
                gray[static_cast<size_t>(j) * kTNX + i] = static_cast<uint8_t>(std::min(255.0f, 255.0f * (v - lo) / span + 0.5f));
            }
        artifact_ok = write_pgm(out_dir + "/pack_slice.pgm", kTNX, kTNY, gray);
    }

    if (artifact_ok)
        std::printf("ARTIFACT: wrote demo/out/pack_temps.csv, demo/out/design_sweep.csv, demo/out/pack_slice.pgm (%d designs, %d cells)\n",
                   kNDesigns, kNCells);
    else
        std::printf("ARTIFACT: FAILED to write demo/out files\n");

    // -------------------------------- verdict -----------------------------------
    const bool all_pass = verify_pass && gate_diffusion_pass && gate_coulomb_pass && gate_thermal_pass
                         && physics_pass && artifact_ok;
    if (all_pass)
        std::printf("RESULT: PASS (solvers verified against CPU twins and three analytic gates; 12-design cooling sweep completed and passed the energy-conservation check)\n");
    else
        std::printf("RESULT: FAIL (a verification stage failed — see the lines above)\n");
    return all_pass ? 0 : 1;
}
