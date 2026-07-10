// ===========================================================================
// kernels.cuh — interface & single-source contract for project 34.03
//               Ergodic control: spectral multiscale coverage (SMC), FFT-based
//               (teaching core: single 2-D first-order agent, reduced-scope
//               [R&D] implementation — see README §Limitations & THEORY.md
//               §Where this sits in the real world for the full-scope story)
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (closed-loop driver + artifacts), kernels.cu
// (the GPU DCT + per-mode SMC update), and reference_cpu.cpp (the CPU twin
// of every GPU computation, PLUS the "plant" that actually moves the agent).
// Everything all three must agree on — the domain, the mode count, the
// target-density shape, the ergodic-metric weights, and the agent's speed
// budget — is defined HERE, once (CLAUDE.md §12).
//
// ERGODIC COVERAGE IN ONE PARAGRAPH (THEORY.md derives all of this):
// A robot exploring a workspace should not sweep it uniformly (a raster/
// lawnmower scan) — it should spend TIME in proportion to where the
// information is, described by a target density phi(x) on the workspace
// [0,1]^2. "Ergodic" means the agent's TIME-AVERAGED occupancy statistics
// converge to phi's statistics. Spectral Multiscale Coverage (SMC; Mathew &
// Mezic 2011) makes this checkable and controllable by comparing FOURIER
// coefficients: phi_k (the target's, computed once) against c_k(t) (the
// trajectory's running time-average, updated every control step), weighted
// by a Sobolev weight Lambda_k that favors getting the SMOOTH (large-scale)
// part of the coverage right first. The controller steers by the NEGATIVE
// GRADIENT of the resulting mismatch metric — no path planning, just a
// per-step direction, at a constant speed budget.
//
// WHY COSINES (Neumann basis)? THEORY.md derives this in full; in short:
// cos(k*pi*x) has ZERO SPATIAL DERIVATIVE at x=0 and x=1 — the "no flux
// through the wall" (Neumann) boundary condition, physically the right one
// for a bounded workspace an agent cannot leave. It is also why this
// project's agent REFLECTS off the domain walls (integrate_agent_cpu below)
// rather than clamping or wrapping: reflection is the boundary condition the
// basis itself assumes.
//
// REDUCED SCOPE (this is an [R&D] catalog bullet, CLAUDE.md §2/§13): ONE
// agent, first-order (single-integrator) dynamics, a fixed bimodal target,
// K=32x32=1024 modes, a fixed 60 s / 6000-step run. Multi-agent SMC,
// second-order (double-integrator) dynamics, obstacles, and adaptive mode
// truncation are the full research version — documented, not implemented
// (README §Limitations, THEORY.md §Where this sits in the real world).
//
// MODE INDEXING — flat index idx = k1*kK + k2, k1,k2 in [0, kK).
// f_k(x1,x2) = (1/h_k) * cos(k1*pi*x1) * cos(k2*pi*x2)   (the basis)
// h_k        = basis L2-norm on [0,1]^2 (1, 1/sqrt2, or 1/2 — see kernels.cu)
// Lambda_k   = (1 + k1^2 + k2^2)^(-s), s = (d+1)/2 = 1.5 for d=2 (Sobolev
//              weight — THEORY.md derives why this exponent and why raw
//              integer-index ||k||^2, not the angular wavenumber (pi*k)^2,
//              is used here: a documented scaling convention).
//
// STATE the closed loop carries between steps:
//   x[2]        — agent position (domain-normalized [0,1]^2, unitless)
//   S[kNumModes]— RUNNING SUM (not yet divided by n) of f_k(x(t)) samples,
//                 persisted in DEVICE memory across steps so the c_k update
//                 kernel below never re-walks trajectory history — the
//                 "single running accumulator" pattern also used by an
//                 online mean. c_k(t) = S_k(t) / n where n = samples so far.
//
// PRECISION: this project computes coefficients in DOUBLE precision
// throughout (phi_k, c_k, S, Lambda_k, the cuFFT plan itself is Z2Z). At
// K=1024 modes and a 256x256 one-time FFT, the throughput cost of double
// vs. float is negligible on any sm_75+ GPU (THEORY.md §GPU mapping
// quantifies this) — and it removes float32 running-sum drift as a source
// of GPU-vs-CPU disagreement, so the §5 verify gate's tolerance measures
// only genuine algorithmic differences (documented in THEORY §numerics).
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cmath>   // std::exp — target_phi_shape() below

// ===========================================================================
// Constants — the single source of truth for the domain, the basis, the
// target density, the agent, and the closed-loop schedule. GPU kernels,
// their CPU twins, and main.cu's artifact/verification code all read these.
// ===========================================================================

constexpr double kPi = 3.14159265358979323846;

// ---- Spectral basis -------------------------------------------------------
constexpr int    kK          = 32;         // modes per axis (k1, k2 in [0, kK))
constexpr int    kNumModes   = kK * kK;    // 1024 total modes — the SMC kernel's thread count
constexpr double kSobolevS   = 1.5;        // s = (d+1)/2, d=2 spatial dimensions (THEORY.md derives)

// ---- Grid used to (a) compute phi_k (via both the GPU DCT and the CPU
// direct-sum oracle) and (b) numerically integrate each hotspot's target
// probability mass for the coverage gate. kPhiGridN points -> kPhiGridN-1
// trapezoidal CELLS per axis; kDctM is the even-extension FFT length the
// DCT-I-via-FFT trick needs (kernels.cu derives this in full): a mirrored
// period of 2*(N-1) samples. 129 -> 128 cells -> 256-point FFT: a clean
// power of two, and dense enough that the two Gaussian hotspots (sigma
// 0.07-0.09, see kMu*/kSigma* below) are resolved by MANY grid cells
// (the trapezoidal quadrature error is far below the tolerances this
// project checks against — THEORY.md §numerics quantifies it).
constexpr int    kPhiGridN   = 129;
constexpr int    kDctM       = 2 * (kPhiGridN - 1);   // = 256

// Visualization-only grid for the PGM artifacts (target_phi.pgm,
// empirical_coverage.pgm) and for the empirical-coverage histogram. Kept
// separate from kPhiGridN/kDctM on purpose: the PGM images are a picture,
// not a numerically-verified quantity, so resolution is chosen for
// LEGIBILITY, not precision. kNSteps=6000 samples spread over a finer grid
// (e.g. 64x64=4096 cells) average under 1.5 samples/cell — a raw visit
// COUNT histogram at that resolution is dominated by binning noise, not
// signal (main.cu's file header and demo/README.md discuss this). 24x24
// = 576 cells gives ~10 samples/cell on average: coarse, but the two-
// hotspot SHAPE the SMC controller traces out is legible, which is the
// artifact's whole job (a picture, not a measurement — the numerical
// COVERAGE gate, computed at full precision from the trajectory directly,
// is what main.cu actually verifies).
constexpr int    kVisGrid    = 24;

// ---- Target density: two Gaussian "information hotspots" + a small
// uniform "washout" floor (so phi never touches exactly zero — every point
// in the domain is at least a LITTLE bit interesting, the honest way to
// model "background information value" alongside two strong hotspots).
// Raw (unnormalized) shape; main.cu integrates it on the kPhiGridN grid and
// divides by that integral so the committed phi truly satisfies
// integral_domain phi dx = 1 (required for phi_(0,0) = 1, matching the
// basis's DC mode f_(0,0) = 1 exactly — see THEORY.md §the math).
constexpr double kMu1X = 0.25, kMu1Y = 0.70, kSigma1 = 0.07, kW1 = 0.45;   // hotspot 1 (upper-left)
constexpr double kMu2X = 0.75, kMu2Y = 0.30, kSigma2 = 0.09, kW2 = 0.45;   // hotspot 2 (lower-right)
constexpr double kWBg  = 0.10;                                            // uniform washout weight

// Coverage-gate basin: a disk of this radius around each hotspot's center.
// The TARGET mass in each basin is computed numerically (never fabricated)
// by the same trapezoidal grid that builds phi_k; the EMPIRICAL mass is the
// fraction of the closed-loop run the agent's trajectory spends inside it.
constexpr double kBasinRadius = 0.15;

// ---- Agent: first-order (single-integrator) dynamics, xdot = u,
// |u| <= kVmax (a pure SPEED budget — direction is the controller's only
// freedom, matching the bang-bang SMC control law in THEORY.md §the math).
constexpr double kVmax   = 0.4;     // domain-units / s (unitless workspace; PRACTICE.md §2 scales to meters)
constexpr double kTTotal = 60.0;    // s, total closed-loop run
constexpr double kDt     = 0.01;    // s, control/integration period -> 100 Hz
constexpr int    kNSteps = 6000;    // = kTTotal / kDt (int, exact: 60/0.01)

constexpr double kX0_1 = 0.50, kX0_2 = 0.50;   // default start: domain center (unbiased toward either hotspot)

// Control-law normalization epsilon: u = -kVmax * B / (||B|| + kBEps).
// Guards the near-ergodic state B -> 0 (division by a genuine zero would be
// undefined; THEORY.md §numerics discusses the resulting low-amplitude
// direction chatter this regularization trades for).
constexpr double kBEps = 1.0e-9;

// Steps used by the closed-form GPU-vs-CPU VERIFY gate (the §5 gate): a
// short window run from a FRESH state on both paths, comparing every
// mode's c_k/Bx/By every step (CLAUDE.md §5, mirroring 08.01's design).
constexpr int kVerifyWindow = 50;

// ---------------------------------------------------------------------------
// target_phi_shape — the RAW (unnormalized) target-density shape: a uniform
// floor plus two Gaussian bumps. Plain host inline function (no CUDA
// attributes needed — it runs ONLY on the host, in main.cu, to build the
// kPhiGridN and kVisGrid grids; neither kernels.cu nor reference_cpu.cpp
// calls it directly, they consume the grid main.cu built and normalized).
//
// Parameters: x1, x2 in [0,1] (domain-normalized workspace coordinates).
// Returns: unnormalized density (>= kWBg > 0 everywhere — never exactly
// zero, so no point of the workspace is modeled as "zero information",
// the honest choice for a washout floor).
// ---------------------------------------------------------------------------
inline double target_phi_shape(double x1, double x2)
{
    const double dx1 = x1 - kMu1X, dy1 = x2 - kMu1Y;
    const double dx2 = x1 - kMu2X, dy2 = x2 - kMu2Y;
    const double g1 = std::exp(-(dx1 * dx1 + dy1 * dy1) / (2.0 * kSigma1 * kSigma1));
    const double g2 = std::exp(-(dx2 * dx2 + dy2 * dy2) / (2.0 * kSigma2 * kSigma2));
    return kWBg + kW1 * g1 + kW2 * g2;
}

// ===========================================================================
// GPU launchers (defined in kernels.cu). Every one owns its own launch
// configuration and post-launch CUDA_CHECK_LAST_ERROR call (CLAUDE.md §6.1
// rule 7). main.cu calls only these, never a __global__ kernel directly —
// the __global__ kernels themselves are private to kernels.cu (no header
// declaration needed: nothing outside that translation unit launches them).
// ===========================================================================

// launch_build_phi_k — compute ALL kNumModes target Fourier coefficients
// ONCE via the DCT-via-FFT pipeline (mirror -> cufftPlan2d Z2Z -> extract +
// normalize). This is the project's named GPU/cuFFT hook (catalog bullet:
// "FFT-based — very GPU-friendly"); kernels.cu's file header explains why a
// single 2-D transform of a small array is still the right teaching example
// even though, at this K and grid size, the wall-clock cost is negligible
// either way (an honesty note THEORY.md repeats).
//   d_phi_grid : DEVICE [kPhiGridN*kPhiGridN] doubles, row-major
//                (n*kPhiGridN+m), the NORMALIZED target density on the
//                trapezoidal grid (integral over the domain == 1).
//   d_phi_k    : DEVICE [kNumModes] doubles OUT, flat index k1*kK+k2.
void launch_build_phi_k(const double* d_phi_grid, double* d_phi_k);

// launch_smc_step — ONE control step's per-mode update: advance the running
// sum S_k, form this step's c_k, and the per-mode contribution to the
// ergodic-descent direction B = sum_k Lambda_k*(c_k-phi_k)*grad f_k(x).
// ONE THREAD PER MODE (kNumModes = 1024 threads) — see kernels.cu for the
// full launch-configuration reasoning and the "small-but-real parallelism
// at this K" honesty note.
//   x1, x2  : the agent's CURRENT position (passed BY VALUE — see kernels.cu
//             for why this project does not need a device pointer/upload
//             for the state, unlike 08.01's 4-float MPPI state).
//   d_phi_k : DEVICE [kNumModes], the target coefficients (read-only, built
//             once by launch_build_phi_k).
//   d_S     : DEVICE [kNumModes] IN/OUT — the running sum S_k; persists
//             across calls (the caller owns its lifetime and must zero it
//             at the start of any fresh run — see main.cu's reset points).
//   n       : sample count INCLUDING this step (>= 1) — the running-average
//             denominator c_k = S_k / n.
//   d_c, d_Bx, d_By : DEVICE [kNumModes] OUT — this step's c_k and the two
//             components of its per-mode gradient contribution. main.cu
//             downloads these (a few KB) and reduces Bx=sum(d_Bx) etc. on
//             the HOST — the same "keep the tiny reduction in plain C++
//             beside the kernel call" choice 08.01 makes, documented there
//             and repeated in this project's THEORY.md §GPU mapping.
void launch_smc_step(double x1, double x2, const double* d_phi_k,
                     double* d_S, int n,
                     double* d_c, double* d_Bx, double* d_By);

// ===========================================================================
// CPU references (reference_cpu.cpp). Two independent jobs, like 08.01's
// reference_cpu.cpp:
//   1) The ORACLE twins of the two GPU paths above (phi_k_direct_cpu,
//      smc_step_cpu) — line-by-line duplicates the §5 VERIFY gate compares
//      against the GPU results.
//   2) integrate_agent_cpu — THE PLANT: the closed loop's actual agent
//      integrator (there is no "GPU plant"; a first-order 2-state Euler
//      step is trivial serial work, so it lives on the host exactly like
//      08.01's cartpole_step_cpu is both oracle-adjacent AND the plant).
// ===========================================================================

// phi_k_direct_cpu — the O(kPhiGridN^2 * kNumModes) DIRECT trapezoidal
// cosine projection (cosine VALUES precomputed into small tables so the
// inner loop is pure multiply-add, no repeated trig — see reference_cpu.cpp)
// — mathematically the SAME double integral launch_build_phi_k computes via
// the DCT-I-via-FFT identity, but via an entirely different code path with
// NO FFT anywhere. Comparing the two is this project's TRANSFORM-CORRECTNESS
// gate (README §Expected output; THEORY.md §How we verify correctness).
//   phi_grid    : HOST [kPhiGridN*kPhiGridN], same normalized grid as above.
//   phi_k_out   : HOST [kNumModes] OUT.
void phi_k_direct_cpu(const double* phi_grid, double* phi_k_out);

// smc_step_cpu — sequential, line-by-line twin of smc_step_kernel (loops
// idx = 0..kNumModes-1 instead of one GPU thread per idx). Same signature
// shape as launch_smc_step but with HOST pointers for S/c/Bx/By.
void smc_step_cpu(double x1, double x2, const double* phi_k,
                  double* S, int n,
                  double* c, double* Bx, double* By);

// integrate_agent_cpu — advance the agent one control step: Euler-integrate
// xdot=u over dt, then REFLECT off the domain walls (the project's single
// defined boundary point — THEORY.md explains why reflection, not clamping
// or wrapping, is the physically consistent choice for the Neumann/cosine
// basis this project uses everywhere else).
//   x  : HOST [2] IN/OUT, domain-normalized agent position.
//   u1, u2, dt : the applied control (|.| implicitly <= kVmax by
//                construction of the caller) and the step size (s).
void integrate_agent_cpu(double x[2], double u1, double u2, double dt);

#endif // PROJECT_KERNELS_CUH
