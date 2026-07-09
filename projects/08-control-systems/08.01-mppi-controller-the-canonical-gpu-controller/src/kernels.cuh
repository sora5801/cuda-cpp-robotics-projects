// ===========================================================================
// kernels.cuh — interface for project 08.01
//               MPPI controller — the canonical GPU controller
//               (teaching core: force-limited cart-pole swing-up)
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (the closed-loop driver), kernels.cu (the
// GPU rollout kernel), and reference_cpu.cpp (the rollout oracle + the
// plant stepper). Everything all three must agree on — the plant model and
// its parameters, the state layout, the cost function, and the noise-array
// layout — is defined HERE, once (CLAUDE.md §12).
//
// MPPI in five lines (THEORY.md derives it properly):
//   1. Keep a nominal control sequence u_nom[0..T).
//   2. Sample K perturbed sequences u_k = clamp(u_nom + eps_k).
//   3. Simulate each — K independent rollouts — and total its cost S_k.
//   4. Weight each rollout by softmin(S_k) and blend: u_nom += Σ w_k eps_k.
//   5. Apply u_nom[0] to the real plant, shift the sequence, repeat.
// Step 3 is >99% of the arithmetic and is embarrassingly parallel across
// k — one GPU thread per rollout. That single fact is why MPPI is "the
// canonical GPU controller" (the catalog's words).
//
// The plant: a CART-POLE (pendulum on a cart, force-limited, frictionless).
// The classic teaching plant because swing-up is genuinely nonlinear — no
// linear controller does it — while the whole model fits in a screen of
// code. The catalog bullet names the ladder "cart-pole → quadrotor → AGV →
// off-road racer"; this project builds the first rung completely and
// documents the rest (README §Limitations, THEORY §real-world).
//
// STATE LAYOUT — float x[4], SI units, documented once here:
//     x[0] = p      cart position (m), +right
//     x[1] = pdot   cart velocity (m/s)
//     x[2] = theta  pole angle (rad), 0 = UPRIGHT, pi = hanging down,
//                   wrapped to (-pi, pi] by the PLANT step only (the single
//                   defined wrap point, CLAUDE.md §12; rollouts integrate
//                   unwrapped and cost uses cos(theta), which never wraps)
//     x[3] = thdot  pole angular velocity (rad/s)
// Control u: horizontal force on the cart (N), clamped to [-kUmax, +kUmax].
//
// NOISE LAYOUT — eps is stored TRANSPOSED: eps[t*K + k], not eps[k*T + t].
// Why: at simulation step t, every thread k reads its own eps — with the
// transposed layout a warp's 32 reads are CONSECUTIVE floats (perfectly
// coalesced); the "natural" per-rollout layout would stride reads T floats
// apart. 33.01 taught the honest-suboptimal layout and its cost; this
// project shows the fix applied from the start. (kernels.cu measures the
// point in its header comment.)
//
// Noise is generated ON THE HOST with the repo's portable xorshift32 and
// uploaded per control step. Production MPPI generates noise on-device with
// cuRAND (no 800 KB upload per tick); we trade that for bit-reproducible
// demos across platforms — the trade is documented, not hidden
// (THEORY.md §numerics; on-device cuRAND is README Exercise 4).
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

// ---------------------------------------------------------------------------
// Plant parameters — shared verbatim by the GPU dynamics, the CPU oracle,
// and the plant stepper (one source of truth; a mismatch here would make
// "model-based control" quietly model the wrong plant).
// Values are the classic Barto/Sutton cart-pole, force-limited for swing-up.
// ---------------------------------------------------------------------------
constexpr float kMassCart = 1.0f;    // cart mass (kg)
constexpr float kMassPole = 0.1f;    // pole mass (kg)
constexpr float kPoleHalfLen = 0.5f; // pole HALF-length l (m) — the standard
                                     // parameterization: dynamics use l, the
                                     // physical pole is 2l = 1 m long
constexpr float kGravity = 9.81f;    // m/s^2
constexpr float kUmax = 10.0f;       // force limit (N) — small enough that
                                     // swing-up needs pumping, the point of
                                     // the demo (a big enough force would
                                     // just yank the pole up in one move)

// ---------------------------------------------------------------------------
// MPPI hyperparameters (defaults; the scenario file may override K only —
// cost weights and horizon are part of the taught, tuned teaching setup).
// ---------------------------------------------------------------------------
constexpr int   kDefaultK = 4096;    // rollouts per control step
constexpr int   kHorizon = 50;       // steps per rollout (T)
constexpr float kDt = 0.02f;         // integration/control period (s) → 50 Hz, 1 s horizon
constexpr float kSigma = 2.5f;       // exploration noise std-dev (N)
constexpr float kLambda = 0.5f;      // softmin temperature (unitless; ↓ = greedier)

// Stage-cost weights (unitless; per-step, tuned for swing-up + balance —
// the tuning story is THEORY.md §algorithm):
constexpr float kWAngle = 10.0f;     // on (1 - cos theta): 0 upright, 20 hanging — smooth, wrap-free
constexpr float kWThdot = 0.1f;      // on thdot^2: damp the pole near the top
constexpr float kWPos   = 0.5f;      // on p^2: keep the cart near the origin
constexpr float kWPdot  = 0.05f;     // on pdot^2: discourage runaway cart speed
constexpr float kWCtrl  = 0.001f;    // on u^2: mild effort penalty

constexpr int kNX = 4;               // state dimension (layout above)

// ---------------------------------------------------------------------------
// launch_mppi_rollouts — evaluate all K rollout costs on the GPU.
//
//   K       : number of rollouts (>= 1).
//   x0      : HOST pointer, kNX floats — the CURRENT plant state every
//             rollout starts from (uploaded internally; it is 16 bytes).
//   d_u_nom : DEVICE pointer, kHorizon floats — the nominal control sequence.
//   d_eps   : DEVICE pointer, kHorizon*K floats — noise, TRANSPOSED layout
//             eps[t*K + k] (see header comment).
//   d_cost  : DEVICE pointer, K floats OUT — total trajectory cost per
//             rollout (unitless, weighted sum defined by the kW* above).
//
// Launch: one thread per rollout, 256-thread blocks (grid math + reasoning
// with the kernel). The kernel is the project's hot loop: T RK4 steps × K.
// ---------------------------------------------------------------------------
void launch_mppi_rollouts(int K, const float* x0,
                          const float* d_u_nom, const float* d_eps,
                          float* d_cost);

// ---------------------------------------------------------------------------
// CPU references (reference_cpu.cpp).
// ---------------------------------------------------------------------------

// mppi_rollouts_cpu — the oracle twin of the kernel: same dynamics, same
// RK4, same cost, same clamps, sequential over k. main.cu runs it against
// the GPU on iteration 0's inputs and requires agreement within a relative
// tolerance (the §5 GPU-vs-CPU gate for this project).
void mppi_rollouts_cpu(int K, const float* x0,
                       const float* u_nom, const float* eps,
                       float* cost);

// cartpole_step_cpu — advance one state by dt under constant force u, RK4.
// Used by main.cu as THE PLANT (the "real" cart-pole the controller drives)
// and by the oracle internally. Wraps theta to (-pi, pi] — the plant step
// is the project's single defined wrap point.
void cartpole_step_cpu(float* x, float u, float dt);

#endif // PROJECT_KERNELS_CUH
