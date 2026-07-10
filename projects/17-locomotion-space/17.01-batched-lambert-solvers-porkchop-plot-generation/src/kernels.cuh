// ===========================================================================
// kernels.cuh — interface for project 17.01
//               Batched Lambert solvers + porkchop plot generation
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (the driver), kernels.cu (the GPU batched
// Lambert-solver kernel), and reference_cpu.cpp (the sequential oracle
// twin). Everything all three must agree on — canonical units, the scenario
// layout, the cell-status codes, and the algorithm's fixed-iteration
// bisection knobs — is defined HERE, once (CLAUDE.md §12).
//
// The problem in one line: for every (departure epoch t1, arrival epoch t2)
// cell on a 2D grid, solve Lambert's problem between two bodies on KNOWN
// circular heliocentric orbits and report the total impulsive delta-v of
// that transfer. Plotting delta-v over the (t1, t2) grid is a PORKCHOP PLOT
// — the standard interplanetary-mission-design tool (THEORY.md derives why
// it has its shape). One GPU thread per grid cell: K = 512*512 = 262,144
// completely independent nonlinear root-finds — a BATCHED SOLVE, the same
// "one thread owns one small numerical problem" pattern as 33.01's batched
// linear algebra, applied to a transcendental equation instead of a matrix.
//
// CANONICAL UNITS — the whole computation runs in a unit system where the
// Sun's gravitational parameter mu = 1 exactly. This is standard practice
// in orbital mechanics (it removes GM_sun = 1.327e20 m^3/s^2 from every
// formula and keeps FP32 magnitudes near 1.0, not 1e11). Two unit choices
// pin the system down; every other unit follows from mu = 1:
//
//     length unit (LU) := 1 AU = 1.495978707e11 m               (IAU 2012)
//     mu = G*M_sun      := 1  (LU^3 / TU^2)  by DEFINITION
//     time unit (TU)    := sqrt(LU^3 / GM_sun_SI)                (derived)
//
// Solving for TU with GM_sun_SI = 1.32712440018e20 m^3/s^2 (DE-series value)
// gives TU = 5,022,642.89 s = 58.132441 mean solar days = 0.159158 Julian
// years. A circular orbit at r = 1 LU therefore has period 2*pi TU (Kepler's
// third law, T = 2*pi*sqrt(a^3/mu), a = mu = 1) — matching Earth's ~1 year
// orbit almost exactly (2*pi TU = 365.25 days to 4 significant figures,
// because 1 AU and 1 year were historically chosen to make this nearly
// true; the small residual is why real ephemerides are not perfectly
// circular and this project's orbits are a teaching idealization).
//
//     SI CONVERSION TABLE (canonical -> SI; also see PRACTICE.md §1)
//     ----------------------------------------------------------------
//     1 LU (length)      = 1.495978707e11 m            = 1 AU
//     1 TU (time)         = 5.022642891e6 s              = 58.132441 days
//     1 LU/TU (velocity)  = 2.978603e4 m/s                = 29.786 km/s
//                           (Earth's own orbital speed, sqrt(mu/1 LU) = 1
//                           LU/TU exactly — a built-in sanity check)
//     1 LU/TU^2 (accel)   = 5.930e-3 m/s^2
//
// WHY NO EPHEMERIS DATA: this project studies two SYNTHETIC coplanar
// circular heliocentric orbits — an Earth-like body at r1 = 1.000 LU and a
// Mars-like body at r2 = 1.524 LU (Mars' real semi-major axis, in AU) —
// each body's position/velocity is the closed-form circular-orbit formula
// or_orbit(r, n, t) below. No JPL ephemeris, no SPICE kernel, no download:
// the whole scenario is five numbers (CLAUDE.md §8 synthetic-first). The
// real-ephemeris path (real planet positions from a SPICE kernel or JPL
// Horizons) is documented as the production next step in README §11 and
// THEORY.md §real-world; it changes body_state()'s inputs, not the Lambert
// solver.
//
// PHASE CONVENTION — both bodies are placed at orbital phase angle 0 (i.e.
// on the +x axis of the heliocentric frame) at canonical time t = 0. This
// is a modeling choice, not a physical fact (real planets are almost never
// aligned) — but because the departure-epoch axis below sweeps a full
// synodic period, EVERY relative phase angle between the two bodies is
// sampled somewhere in the grid regardless of the t=0 alignment, so the
// choice does not bias which transfers are representable (THEORY.md
// §the-math works this out precisely, including exactly where in the grid
// the Hohmann-optimal alignment recurs).
//
// GRID LAYOUT — row-major, cell (i, j) at flat index j*grid_n + i:
//     i = departure-epoch index, t1 = i * (window_tu / grid_n)   [0, grid_n)
//     j = arrival-epoch   index, t2 = j * (window_tu / grid_n)   [0, grid_n)
// Both axes span the SAME window [0, window_tu) TU — the classic porkchop
// convention (departure and arrival dates on comparable scales) — so a
// single dt = window_tu / grid_n applies to both. This matches the PGM
// artifact's row-major image layout directly (row j = one arrival epoch,
// same index arithmetic 07.09 uses for its distance-field images).
//
// SCENARIO — the "task definition" loaded from data/sample/ (a committed,
// synthetic, no-RNG record — CLAUDE.md §8): both orbit radii, the window,
// the accepted time-of-flight band, and the grid resolution. mu is NOT in
// the file: mu = 1 is an axiom of canonical units, not scenario data.
//
// CELL STATUS — every cell gets exactly one status code, the NaN POLICY
// this project uses (documented like 33.01's):
//     kStatusOk            (0) — delta_v[cell] holds a real number (LU/TU)
//     kStatusMaskedTof      (1) — t2-t1 outside [min_tof_tu, max_tof_tu):
//                                 a STRUCTURAL exclusion (too fast to be
//                                 physical or too slow to be a sane mission
//                                 candidate) — not a solver failure.
//     kStatusLongWay        (2) — the prograde transfer angle exceeds pi:
//                                 a SCOPE exclusion (this project solves
//                                 only the short-way/Type-I branch in v1 —
//                                 the long-way/Type-II branch is README
//                                 Exercise territory) — not a solver failure.
//     kStatusNearSingular   (3) — |transfer angle - pi| < kEpsSingularRad:
//                                 the universal-variable Lambert equations
//                                 are MATHEMATICALLY singular at a transfer
//                                 angle of exactly 180 deg (the transfer
//                                 plane through Sun-r1-r2 is undefined when
//                                 r1, r2, and the Sun are collinear) — this
//                                 is the TRUE "degenerate" NaN case, and
//                                 (delightfully/frustratingly) it sits right
//                                 on top of the Hohmann optimum — THEORY.md
//                                 §numerics makes this the centerpiece.
//     kStatusNonConverged   (4) — the fixed-bracket bisection never found a
//                                 sign change (should be vanishingly rare
//                                 given the bracket in this file; counted,
//                                 never silently ignored).
// main.cu's NaN-policy gate measures (kStatusNearSingular +
// kStatusNonConverged) as a fraction of ATTEMPTED cells (status != masked,
// status != long-way — i.e. cells where a short-way solve was actually
// tried) and requires that fraction to stay small (measured value printed
// on an [info] line; the bound is documented at the assertion site).
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

// ---------------------------------------------------------------------------
// Vec2 — a plain 2D vector in the shared orbital plane (z = 0 always: both
// orbits are coplanar by construction, so THEORY.md's 3D Lambert derivation
// specializes to 2D here — no third component to carry or discard). Units
// are LU (position) or LU/TU (velocity), documented at every call site.
//
// A hand-rolled POD struct rather than CUDA's float2: this file is included
// by BOTH nvcc (kernels.cu) and cl.exe (reference_cpu.cpp, main.cu is also
// nvcc but reference_cpu.cpp is plain C++17) — a bare struct needs no CUDA
// headers on either side, and "Vec2" reads as orbital-mechanics vocabulary
// rather than a GPU implementation detail (07.09 uses CUDA's int4 for the
// same reason float2 would work here; either choice is fine — this project
// picks the header-light one).
// ---------------------------------------------------------------------------
struct Vec2 {
    float x, y;
};

// ---------------------------------------------------------------------------
// LambertScenario — the runtime "task definition" loaded from
// data/sample/lambert_scenario.csv (main.cu's load_scenario()). Passed BY
// VALUE into the kernel (20 bytes — far under any kernel-parameter limit,
// so no device upload is needed, unlike 08.01's per-tick x0 buffer).
// ---------------------------------------------------------------------------
struct LambertScenario {
    float r1_au;       // body 1 ("Earth-like") circular orbit radius, LU (1 LU = 1 AU)
    float r2_au;       // body 2 ("Mars-like") circular orbit radius, LU
    float window_tu;   // BOTH epoch axes span [0, window_tu), TU (canonical time)
    float min_tof_tu;  // accepted time-of-flight band, lower bound, TU (excl.)
    float max_tof_tu;  // accepted time-of-flight band, upper bound, TU (excl.)
    int   grid_n;       // grid resolution per axis: grid_n * grid_n total cells
};

// ---------------------------------------------------------------------------
// Cell status codes — see the big file-header comment for the full NaN
// policy. Plain ints (not an enum class) so they drop straight into the
// int status[] output array without a cast at every use site.
// ---------------------------------------------------------------------------
constexpr int kStatusOk            = 0;
constexpr int kStatusMaskedTof     = 1;
constexpr int kStatusLongWay       = 2;
constexpr int kStatusNearSingular  = 3;
constexpr int kStatusNonConverged  = 4;

// ---------------------------------------------------------------------------
// Algorithm constants — the universal-variable Lambert solver's fixed,
// GPU/CPU-IDENTICAL iteration scheme (CLAUDE.md §5 gate: same algorithm on
// both paths so the GPU-vs-CPU comparison means something). Tuned once,
// documented, never silently changed per-cell — see THEORY.md §the-algorithm
// and §numerical-considerations for the derivation of each number.
// ---------------------------------------------------------------------------
constexpr float kPi = 3.14159265358979323846f;

// Bisection runs on the universal (Stumpff) anomaly z. The bracket must
// contain the true root for every (r1n, r2n, transfer angle, TOF) this
// project's scenario can produce; [-60, 39] is generous on both ends (the
// upper bound stays clear of the C(z) pole at z = 4*pi^2 ~= 39.478 — the
// elliptical multi-revolution boundary — THEORY.md derives why that pole
// exists) and was validated against this project's own scenario in
// THEORY.md §how-we-verify-correctness.
constexpr float kBisectZLo = -60.0f;
constexpr float kBisectZHi = 39.0f;

// Fixed iteration count — no early exit, ever, on EITHER path (a Newton
// method could converge in far fewer steps, but a variable iteration count
// would make "GPU thread k took N_k iterations" itself a source of
// GPU-vs-CPU divergence risk; bisection halves the bracket every step, so
// 60 steps shrinks the initial width-99 bracket to 99/2^60 ~= 8.6e-17 —
// vastly below FP32's ~1.2e-7 relative epsilon, i.e. deliberate, harmless
// overkill for a clean, comparable iteration count (mirrors 08.01's fixed
// RK4 step count for the same "identical scheme" reason).
constexpr int kBisectIters = 60;

// y(z) (the "universal" auxiliary quantity, THEORY.md §the-math) must stay
// positive for sqrt(y) to be real; this floor guards the FAR ends of the
// search bracket (where the true root is never expected to sit) against a
// transient negative y during bisection, without perturbing the converged
// answer near the actual root (THEORY.md §numerical-considerations).
constexpr float kYFloor = 1e-6f;

// Cells within this many radians of a transfer angle of exactly pi are
// flagged kStatusNearSingular rather than solved (see the big header
// comment). 2 degrees, converted to radians once here so both kernels.cu
// and reference_cpu.cpp read the identical literal.
constexpr float kEpsSingularRad = 0.034906585f;   // 2 deg in radians

// ---------------------------------------------------------------------------
// launch_lambert_grid — solve every cell of sc's grid on the GPU.
//
//   sc         : the scenario (by value — see the struct comment above).
//   d_deltav   : DEVICE pointer, grid_n*grid_n floats OUT — total transfer
//                delta-v per cell (LU/TU), or NaN where status != kStatusOk.
//   d_status   : DEVICE pointer, grid_n*grid_n ints OUT — the cell status
//                code (kStatus* above) — main.cu's NaN-policy accounting
//                and the artifact writer both read this array.
//
// Launch: one thread per cell, 256-thread blocks (grid math + reasoning
// live with the kernel in kernels.cu, next to the code they configure).
// ---------------------------------------------------------------------------
void launch_lambert_grid(const LambertScenario& sc, float* d_deltav, int* d_status);

// ---------------------------------------------------------------------------
// CPU reference (reference_cpu.cpp) — the oracle twin of the kernel: the
// SAME per-cell algorithm (body_state, Stumpff series, bisection with the
// identical bracket/iteration count above), evaluated sequentially. Output
// layout identical to the GPU path. main.cu runs both on the committed
// scenario and requires agreement (the §5 GPU-vs-CPU gate for this
// project) — see main.cu's VERIFY stage for the exact tolerance.
// ---------------------------------------------------------------------------
void lambert_grid_cpu(const LambertScenario& sc, float* deltav, int* status);

#endif // PROJECT_KERNELS_CUH
