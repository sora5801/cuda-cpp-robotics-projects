// ===========================================================================
// kernels.cuh — interface for project 10.03
//               Massively parallel robot sim (Isaac-Gym-style: one robot,
//               10,000 environments)
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (the orchestrator), kernels.cu (the GPU farm
// kernels), and reference_cpu.cpp (the CPU oracle + the energy-conservation
// diagnostic). Everything all three must agree on — the plant model, the
// SoA state layout, the controller, the episode/reset rules, and the
// per-env RNG scheme — is defined HERE, once (CLAUDE.md §12).
//
// The pattern in five lines (THEORY.md derives it properly):
//   1. ONE robot model (a force-limited cart-pole — 08.01's plant, reused).
//   2. N = 10,000 independent COPIES of it ("environments"), each with its
//      own randomized mass/length (domain randomization) and its own
//      episode clock.
//   3. One GPU thread steps ONE environment; all N threads step in
//      LOCKSTEP (same dt, same instant) — the defining trait of an
//      Isaac-Gym-style training farm, as opposed to 08.01's single plant
//      sampling many CANDIDATE futures.
//   4. Each environment runs a fixed control policy (here: hand-tuned
//      pole-placement gains — a stand-in for the network 12.06 would train)
//      and independently fails/succeeds/resets on its own clock.
//   5. The whole T-step run is ONE kernel launch: no per-tick host
//      round-trip is needed (nothing here depends on a host decision),
//      so the aggregate throughput — env-steps per second — is the number
//      that matters (THEORY.md §GPU mapping), not a per-tick latency.
//
// The plant: the SAME cart-pole as 08.01 (frictionless, force-limited),
// generalized so mass/length are PER-ENVIRONMENT runtime values instead of
// compile-time constants — that one change is what makes domain
// randomization possible. See 08.01's kernels.cuh for the plant's physical
// story; this file does not repeat it.
//
// STATE LAYOUT — STRUCTURE OF ARRAYS (SoA), not the array-of-structs each
// thread privately owned in 08.01/09.01. Four parallel arrays, one float
// per environment:
//     x[i]      cart position (m),           env i
//     xdot[i]   cart velocity (m/s),          env i
//     theta[i]  pole angle (rad), 0=upright,  env i
//     thdot[i]  pole angular velocity (rad/s),env i
// WHY SoA HERE, when 08.01's rollout kernel kept x[4] entirely in one
// thread's REGISTERS? Both are "one thread owns one environment", but the
// two projects differ in what happens BETWEEN kernel calls:
//   - 08.01: the kernel is a stateless pure function called once per 20 ms
//     control tick; state lives in registers for the kernel's lifetime,
//     is read from a FRESH x0 each call, and never needs to persist.
//   - 10.03: N=10,000 environments must survive ACROSS a 1000-step run
//     (and, on a real training farm, across THOUSANDS of RL update calls
//     that read/write this state between kernel launches — e.g. rewards
//     computed on the host, policy weights updated, then the same buffers
//     stepped again). The state MUST live in GLOBAL MEMORY, addressable by
//     environment index, between launches.
//   Given state must live in global memory, the layout choice is the whole
//   ballgame: with SoA, thread i's read of x[i] sits in the SAME cache
//   line as thread i+1's read of x[i+1] — a warp's 32 threads issue ONE
//   128-byte coalesced transaction per array, four transactions per step
//   (x, xdot, theta, thdot). The alternative, AoS — one struct
//   {x,xdot,theta,thdot} per environment, indexed env[i] — would put a
//   single thread's four floats ADJACENT and every other thread's floats
//   16 bytes further on: a warp's 32 threads reading env[i].theta then
//   scatter across 32 separate 16-byte-strided addresses, wasting 3 of
//   every 4 sectors fetched per transaction (each 32-byte or 128-byte DRAM
//   burst carries mostly OTHER threads' x/xdot/thdot that this particular
//   load does not need). SoA is the coalescing-correct layout precisely
//   BECAUSE every thread in lockstep touches the SAME field at the SAME
//   time — the 33.01 lesson, applied to a genuinely stateful, resident
//   simulation instead of a one-shot kernel (kernels.cu measures the
//   contrast in its header comment).
//
// PER-ENVIRONMENT PARAMETERS (domain randomization) — three more parallel
// arrays, drawn ONCE per environment at farm init and held fixed for the
// entire run (never re-randomized on episode reset — only the INITIAL
// ANGLE is re-drawn on reset; see THEORY.md §domain randomization for why
// these are different kinds of randomness):
//     mass_cart[i], mass_pole[i], pole_half_len[i]  — SI units, per env i
//
// CONTROLLER — a FIXED linear state-feedback law, identical gains for
// every environment (it stands in for "one trained policy", stress-tested
// across N randomized copies of the plant — the Isaac-Gym farm's actual
// job): u = Kx*x + Kxd*xdot + Kth*theta + Kthd*thdot, clamped to
// [-kUmax,+kUmax]. THEORY.md derives the gains via pole placement on the
// nominal linearization and reports the measured robustness margin across
// the randomization envelope.
//
// EPISODE / RESET SEMANTICS — the third RL-farm ingredient:
//   FAIL   : |x| > kXFail  OR  |theta| > kThetaFail        (pole fell / cart ran off the track)
//   CAP    : steps-in-episode >= episode_cap                (successful episode, truncated)
//   on FAIL or CAP: reset_count[i]++, then draw a fresh initial angle from
//   env i's OWN persistent RNG stream and zero the rest of the state.
//   This reset logic is fused INSIDE the per-env step loop (one kernel),
//   not a separate "reset kernel" launch — see kernels.cu's header comment
//   for why that is both the correct AND the higher-performance choice.
//
// PER-ENV RNG — xorshift32, the SAME portable generator 08.01 used, but
// used differently: 08.01 generated noise ON THE HOST (for cross-platform
// bit-reproducibility) and only ever READ it on the device. This project's
// whole point is a RESIDENT farm — no per-tick host round-trip — so the
// RNG must run ON THE DEVICE, with each environment owning a persistent
// stream (rng_state[i]) seeded once at init and advanced by every
// domain-randomization draw and every reset. Because xorshift32 and its
// uniform01() mapping are PURE INTEGER/BIT operations (no transcendental
// library call), the SAME function compiled for host and device produces
// BIT-IDENTICAL output — the one part of this project's GPU-vs-CPU
// verification that is not merely "close within tolerance" but exactly
// reproducible (THEORY.md §numerical considerations).
//
// SHARING DYNAMICS ACROSS HOST AND DEVICE (a deliberate departure from
// 08.01's rule). 08.01 hand-duplicated its dynamics into kernels.cu
// (__device__) and reference_cpu.cpp (plain host) on purpose, so the CPU
// oracle file reads as ordinary C++ with no CUDA concept in sight. This
// project needs something 08.01 did not: the SAME algorithm — dynamics,
// controller, fail/cap/reset logic, AND the RNG — running verbatim on both
// the device (inside the kernel) and the host (inside the CPU oracle and
// the energy-conservation diagnostic), because any hand-copied divergence
// between two independently-typed implementations would be indistinguishable
// from a real bug when the two paths are compared. So the functions below
// are written ONCE, as small `HD` (host-and-device) inline functions in
// THIS header, and reused unchanged by kernels.cu and reference_cpu.cpp.
// The `HD` macro (defined below) expands to `__host__ __device__` only
// when nvcc is compiling (`__CUDACC__` defined) and to NOTHING when cl.exe
// compiles reference_cpu.cpp directly — so reference_cpu.cpp still never
// sees a CUDA keyword and stays plain C++17, honoring 08.01's rule in
// spirit while satisfying this project's correctness requirement in
// practice. Sharing the source does NOT make the two paths bit-identical
// by itself (nvcc's device sinf/cosf and the host CRT's sinf/cosf remain
// two different, independently-rounded implementations) — it only removes
// hand-copy bugs as a possible EXPLANATION for any measured divergence,
// which is exactly what lets §5's tolerance number mean something.
//
// Read this after: 08.01's kernels.cuh (the plant this one reuses).
// Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cstdint>   // uint32_t — the RNG state type, portable on host and device
#include <cmath>     // sinf/cosf/fabsf/fminf/fmaxf/fmaf — declared for the host path;
                     // nvcc supplies device overloads of the SAME names for the device path

// ---------------------------------------------------------------------------
// HD — expands to "__host__ __device__" when nvcc compiles this header
// (main.cu, kernels.cu — both .cu files), and to NOTHING when cl.exe
// compiles it directly (reference_cpu.cpp includes this header but is
// never touched by nvcc). This is the standard trick for sharing inline
// math between the two compilers without leaking CUDA syntax into a file
// that must stay plain C++ (see the file header's "sharing dynamics" note).
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ---------------------------------------------------------------------------
// Nominal plant parameters — the SAME classic Barto/Sutton cart-pole 08.01
// uses (so a reader who studied 08.01 recognizes every number). Every
// environment's ACTUAL mass_cart[i]/mass_pole[i]/pole_half_len[i] is a
// randomized draw centered on these nominal values (see FarmScenario
// below); kGravity and kUmax are never randomized (gravity is a universal
// constant here, and the force LIMIT is a hardware spec of the actuator,
// not a modeling uncertainty).
// ---------------------------------------------------------------------------
constexpr float kMassCartNominal     = 1.0f;   // kg
constexpr float kMassPoleNominal     = 0.1f;   // kg
constexpr float kPoleHalfLenNominal  = 0.5f;   // m (half-length l; physical pole is 2l = 1 m)
constexpr float kGravity             = 9.81f;  // m/s^2
constexpr float kUmax                = 10.0f;  // N — actuator force limit, same for every env
constexpr float kDt                  = 0.02f;  // s — 50 Hz physics/control tick (matches 08.01)

// ---------------------------------------------------------------------------
// Episode thresholds — compile-time because they are part of the TAUGHT,
// TUNED task definition (same status as 08.01's cost weights), not a
// per-run experimental knob. kXFail/kThetaFail match the classic
// Barto/Sutton/Gym "CartPole-v1" fail box (12 deg, 2.4 m) so a reader who
// has seen that benchmark recognizes the numbers immediately.
// kBalancedTheta is a TIGHTER band used only for the steps-balanced
// quality metric (README/THEORY explain why it differs from kThetaFail).
// ---------------------------------------------------------------------------
constexpr float kThetaFail      = 0.20943951f; // rad (12 deg) — episode FAILS past this
constexpr float kXFail          = 2.4f;        // m            — episode FAILS past this
constexpr float kBalancedTheta  = 0.10471976f; // rad (6 deg)  — "well balanced" quality band

// ---------------------------------------------------------------------------
// Farm-shape DEFAULTS. The committed scenario file may override every one
// of these (CLAUDE.md §8: "N, T, randomization ranges, controller gains,
// seeds" are the task's data); the constants here exist so the demo has a
// documented fallback and so THEORY.md/README.md can cite fixed numbers.
// kVerifyEnvs/kVerifySteps are NOT scenario-overridable: they size the §5
// GPU-vs-CPU gate, which is part of the verification harness, not the task.
// ---------------------------------------------------------------------------
constexpr int      kDefaultN            = 10000;  // environments in the full farm
constexpr int      kDefaultTFarm        = 1000;   // ticks in the full farm run
constexpr int      kDefaultEpisodeCap   = 200;    // ticks per episode before a "successful" reset
constexpr uint32_t kDefaultSeed         = 42u;    // base RNG seed (repo tradition, see 08.01)
constexpr float    kDefaultDrMassCart   = 0.20f;  // +/-20% domain randomization on mass_cart
constexpr float    kDefaultDrMassPole   = 0.30f;  // +/-30% domain randomization on mass_pole
constexpr float    kDefaultDrLen        = 0.15f;  // +/-15% domain randomization on pole_half_len
constexpr float    kDefaultTheta0Range  = 0.15f;  // rad — initial-angle draw range at every reset
constexpr float    kDefaultKx           = 12.0f;  // N/m      balance-controller gain on x
constexpr float    kDefaultKxd          = 14.0f;  // N.s/m    balance-controller gain on xdot
constexpr float    kDefaultKth          = 73.0f;  // N/rad    balance-controller gain on theta
constexpr float    kDefaultKthd         = 19.0f;  // N.s/rad  balance-controller gain on thdot

constexpr int      kVerifyEnvs  = 256;  // size of the §5 GPU-vs-CPU subset gate
constexpr int      kVerifySteps = 220;  // > episode cap (200): forces exactly one CAP-triggered
                                        // reset per env inside the verify window, so the gate
                                        // exercises the reset path, not just free integration.

constexpr int   kEnergySteps = 1000;   // ticks in the undriven energy-conservation experiment
constexpr float kEnergyTheta0 = 0.5f;  // rad — starting angle for that experiment (a real swing,
                                        // not a small-angle wobble, so RK4 does genuine work)

constexpr int kNX = 4;   // state dimension per environment (documented layout above);
                          // NOT how memory is laid out here (that is SoA, four separate
                          // arrays) — kNX is used only for small fixed-size scratch arrays
                          // inside rk4_step below.

// ---------------------------------------------------------------------------
// FarmBuffers — a plain bag of pointers describing ONE farm's persistent
// state + per-env parameters + per-env metrics. The SAME struct type is
// used for DEVICE pointers (passed by value into the kernels in kernels.cu
// — cheap: it is 11 pointers, well under any kernel-parameter-size limit)
// and for HOST pointers (used directly by reference_cpu.cpp's CPU oracle).
// The struct owns nothing: main.cu allocates and frees every array (device
// via cudaMalloc/cudaFree, host via std::vector) and is responsible for
// keeping the two populations (verify-subset vs full-farm) in separate
// FarmBuffers instances (main.cu's two stages never share one).
// ---------------------------------------------------------------------------
struct FarmBuffers {
    float*    x;               // [N] cart position, m
    float*    xdot;             // [N] cart velocity, m/s
    float*    theta;            // [N] pole angle, rad (0 = upright)
    float*    thdot;            // [N] pole angular velocity, rad/s
    float*    mass_cart;        // [N] per-env randomized cart mass, kg (fixed after init)
    float*    mass_pole;        // [N] per-env randomized pole mass, kg (fixed after init)
    float*    pole_half_len;    // [N] per-env randomized pole half-length, m (fixed after init)
    uint32_t* rng_state;        // [N] persistent per-env xorshift32 stream
    int*      ep_step;          // [N] steps elapsed in the CURRENT episode
    int*      steps_balanced;   // [N] OUT metric: cumulative ticks with |theta|<kBalancedTheta
                                 //     and in-bounds (the "return-like" quality signal)
    int*      reset_count;      // [N] OUT metric: cumulative episode resets (fail + cap-out)
};

// ---------------------------------------------------------------------------
// xorshift32 / uniform01 / uniform_range — the repo's portable RNG
// (identical algorithm to 08.01's host-side generator), promoted to a
// shared HD inline function because THIS project runs it on the device.
// Pure integer/bit arithmetic + one fixed-point-style scale-and-shift: no
// transcendental call, so host and device produce BIT-IDENTICAL streams
// for the same seed (THEORY.md §numerical considerations relies on this).
// ---------------------------------------------------------------------------
HD inline uint32_t xorshift32(uint32_t& state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

// uniform01 — (0,1], never exactly 0 (safe for any future log()-based use;
// kept even though this project only needs uniform ranges, for parity with
// 08.01's helper and because "never 0" is the cheap, safe default).
HD inline float uniform01(uint32_t& state)
{
    return (xorshift32(state) >> 8) * (1.0f / 16777216.0f) + (0.5f / 16777216.0f);
}

// uniform_range — one draw in [lo, hi). Used for BOTH kinds of randomness
// in this project: the once-at-init domain-randomization draws (mass/
// length) and the every-reset initial-angle draw (THEORY.md explains why
// those are conceptually different even though they share this helper).
HD inline float uniform_range(uint32_t& state, float lo, float hi)
{
    return lo + (hi - lo) * uniform01(state);
}

// env_seed — turn a base seed + environment index into a well-mixed,
// never-zero per-env stream seed. 2654435761u is Knuth's multiplicative
// hash constant (2^32 * the golden ratio, rounded to an odd integer) —
// a different odd mixing constant than 08.01's 1000003u (which mixed a
// STEP index into a shared seed) because here we are mixing an
// ENVIRONMENT index into a per-run base seed; either constant would work,
// the point is "some large odd constant", documented rather than magic.
HD inline uint32_t env_seed(uint32_t base_seed, int env_id)
{
    uint32_t s = base_seed + 2654435761u * static_cast<uint32_t>(env_id + 1);
    if (s == 0) s = 1u;   // xorshift32 is fixed at 0 forever if seeded with 0 — guard it
    return s;
}

// ---------------------------------------------------------------------------
// cartpole_deriv — the plant's equations of motion, xdot = f(x, u), THE
// SAME formulas 08.01 derives from the Lagrangian (see 08.01/THEORY.md;
// not re-derived here) — generalized so mc/mp/l are RUNTIME per-env values
// instead of compile-time constants (the one change domain randomization
// needs). x/xdot_out are 4-element arrays in the kernels.cuh state order
// [p, pdot, theta, thetadot].
// ---------------------------------------------------------------------------
HD inline void cartpole_deriv(const float x[kNX], float u, float mc, float mp, float l,
                              float xdot_out[kNX])
{
    const float sin_th = sinf(x[2]);   // precise sinf/cosf — same reasoning as 08.01/09.01:
    const float cos_th = cosf(x[2]);   // this project's angles stay small (kThetaFail = 12 deg)
                                       // so intrinsic-vs-precise barely matters here, but using
                                       // the precise path keeps this file's numerics story
                                       // identical to 08.01's and avoids a second thing to justify

    const float total_mass = mc + mp;
    const float ml = mp * l;

    // tmp = acceleration the cart would have if the pole rode along as a
    // point mass (force + the swinging pole's centrifugal reaction).
    const float tmp = (u + ml * x[3] * x[3] * sin_th) / total_mass;

    // Pole angular acceleration: gravity torque vs. the cart's reaction.
    const float th_acc = (kGravity * sin_th - cos_th * tmp)
        / (l * (4.0f / 3.0f - mp * cos_th * cos_th / total_mass));

    // Cart acceleration: tmp minus the pole's back-reaction.
    const float p_acc = tmp - ml * th_acc * cos_th / total_mass;

    xdot_out[0] = x[1];    // pdot
    xdot_out[1] = p_acc;   // pddot
    xdot_out[2] = x[3];    // thetadot
    xdot_out[3] = th_acc;  // thetaddot
}

// ---------------------------------------------------------------------------
// rk4_step — classic 4th-order Runge-Kutta under zero-order hold (u held
// constant across the step — how a 50 Hz controller actually drives an
// actuator; same choice and justification as 08.01/THEORY.md).
//
// Every multiply-add that blends a derivative into a state uses fmaf() —
// a SINGLE, IEEE-754-correctly-rounded fused multiply-add — INSTEAD of a
// separate multiply then add (which would round TWICE). fmaf() is a
// standard C library function available identically on host (CRT/libm)
// and device (CUDA's device runtime); because "correctly rounded" has a
// UNIQUE answer for given inputs, host fmaf() and device fmaf() should
// return the SAME bit pattern for the same inputs — unlike a bare `a*b+c`,
// which nvcc's device compiler contracts into an fma by default while
// cl.exe's host compiler does NOT (unless /fp:fast) — the exact,
// documented source of the ~1 ULP SAXPY divergence in the scaffold
// placeholder this project replaced. Writing fmaf() explicitly everywhere
// removes that whole class of divergence from THIS file, leaving sinf/
// cosf's independent host/device implementations as the only remaining,
// harder-to-remove source (THEORY.md §numerical considerations measures
// what is left after this discipline).
// ---------------------------------------------------------------------------
HD inline void rk4_step(float x[kNX], float u, float mc, float mp, float l, float dt)
{
    float k1[kNX], k2[kNX], k3[kNX], k4[kNX], xt[kNX];

    cartpole_deriv(x, u, mc, mp, l, k1);
    for (int i = 0; i < kNX; ++i) xt[i] = fmaf(0.5f * dt, k1[i], x[i]);
    cartpole_deriv(xt, u, mc, mp, l, k2);
    for (int i = 0; i < kNX; ++i) xt[i] = fmaf(0.5f * dt, k2[i], x[i]);
    cartpole_deriv(xt, u, mc, mp, l, k3);
    for (int i = 0; i < kNX; ++i) xt[i] = fmaf(dt, k3[i], x[i]);
    cartpole_deriv(xt, u, mc, mp, l, k4);

    // x += dt/6 * (k1 + 2k2 + 2k3 + k4) — folded into ONE fmaf per
    // component (the sum is plain adds; only the final scale-and-add to x
    // is a rounding-sensitive step, so that is the one we fuse).
    for (int i = 0; i < kNX; ++i) {
        const float sum = k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + k4[i];
        x[i] = fmaf(dt * (1.0f / 6.0f), sum, x[i]);
    }
}

// ---------------------------------------------------------------------------
// reset_episode — draw a fresh initial angle from env i's OWN persistent
// RNG stream and zero the rest of the state. Used both for the very FIRST
// episode (called once from farm_init_one_env) and every subsequent
// mid-run reset (called from farm_step_one_env on FAIL or CAP) — the
// "reset kernel" logic from the catalog bullet, fused into one function
// reused by both call sites rather than duplicated.
// ---------------------------------------------------------------------------
HD inline void reset_episode(float& x, float& xdot, float& theta, float& thdot,
                             int& ep_step, uint32_t& rng, float theta0_range)
{
    x = 0.0f;
    xdot = 0.0f;
    thdot = 0.0f;
    theta = uniform_range(rng, -theta0_range, theta0_range);
    ep_step = 0;
}

// ---------------------------------------------------------------------------
// farm_init_one_env — ONE environment's t=0 setup: seed its RNG stream,
// draw its domain-randomized mass_cart/mass_pole/pole_half_len (ONCE, for
// the whole run — THEORY.md explains why these are drawn only here and
// never again, unlike the initial angle), zero its metrics, and take its
// FIRST episode reset (draws the first initial angle).
// ---------------------------------------------------------------------------
HD inline void farm_init_one_env(float& x, float& xdot, float& theta, float& thdot,
                                 float& mc, float& mp, float& l,
                                 int& ep_step, uint32_t& rng,
                                 int& steps_balanced, int& reset_count,
                                 uint32_t seed_i,
                                 float dr_mc, float dr_mp, float dr_l, float theta0_range)
{
    rng = seed_i;
    mc = kMassCartNominal    * uniform_range(rng, 1.0f - dr_mc, 1.0f + dr_mc);
    mp = kMassPoleNominal    * uniform_range(rng, 1.0f - dr_mp, 1.0f + dr_mp);
    l  = kPoleHalfLenNominal * uniform_range(rng, 1.0f - dr_l,  1.0f + dr_l);
    steps_balanced = 0;
    reset_count = 0;
    reset_episode(x, xdot, theta, thdot, ep_step, rng, theta0_range);
}

// ---------------------------------------------------------------------------
// farm_step_one_env — ONE environment, ONE physics/control tick:
// evaluate the fixed balance controller, clamp to the actuator limit,
// integrate one RK4 step, classify the tick (failed / balanced-quality),
// advance the episode clock, and reset in place if the episode just ended
// (FAIL or CAP). This single function is called by BOTH the GPU kernel's
// per-thread T-step loop (kernels.cu) and the CPU oracle's per-env loop
// (reference_cpu.cpp) — see the file header's "sharing dynamics" note.
// ---------------------------------------------------------------------------
HD inline void farm_step_one_env(float& x, float& xdot, float& theta, float& thdot,
                                 float mc, float mp, float l,
                                 int& ep_step, uint32_t& rng,
                                 int& steps_balanced, int& reset_count,
                                 float Kx, float Kxd, float Kth, float Kthd,
                                 float theta0_range, int episode_cap)
{
    // Fixed linear state feedback around upright — ONE policy, applied
    // identically in every environment (THEORY.md derives Kx/Kxd/Kth/Kthd
    // via pole placement on the nominal linearization and measures how far
    // the SAME gains generalize across the randomized mc/mp/l envelope).
    float u = Kx * x + Kxd * xdot + Kth * theta + Kthd * thdot;
    u = fminf(fmaxf(u, -kUmax), kUmax);   // actuator saturation — same clamp-before-integrate
                                          // discipline as 08.01 (the controller must experience
                                          // the same limit the real actuator would impose)

    float xs[kNX] = { x, xdot, theta, thdot };
    rk4_step(xs, u, mc, mp, l, kDt);
    x = xs[0]; xdot = xs[1]; theta = xs[2]; thdot = xs[3];
    ep_step += 1;

    // FAIL: pole fell past the classic 12-degree box, or the cart ran off
    // a 2.4 m track — the standard CartPole failure definition, reused so
    // this project's thresholds are recognizable, not invented from
    // scratch (THEORY.md §the problem cites the source).
    const bool failed = (fabsf(x) > kXFail) || (fabsf(theta) > kThetaFail);

    // "Balanced" is a TIGHTER band than "not failed": it is the
    // return-like quality signal written to env_metrics.csv, distinct
    // from the wider fail box that merely ends the episode.
    if (!failed && fabsf(theta) < kBalancedTheta) steps_balanced += 1;

    // CAP: the episode ran its full scheduled length without failing —
    // a SUCCESSFUL truncation, exactly like a fixed-horizon RL rollout.
    const bool capped = (ep_step >= episode_cap);

    if (failed || capped) {
        reset_count += 1;
        reset_episode(x, xdot, theta, thdot, ep_step, rng, theta0_range);
    }
}

// ---------------------------------------------------------------------------
// cartpole_energy — total mechanical energy (kinetic + potential) of the
// UNDRIVEN cart-pole in state x, used only by the energy-conservation
// diagnostic (main.cu + reference_cpu.cpp). Derived from the same
// Lagrangian 08.01/THEORY.md cites for the equations of motion (rigid rod
// of half-length l, mass mp, pivoting on a cart of mass mc); the full
// derivation is in THEORY.md §the math. No device call site needs this
// function (the farm kernel never computes energy — it is a CPU-only
// verification tool) but it is still written HD and shared here so the
// SAME formula that is documented is the one that is checked — no black
// box, CLAUDE.md §1.
// ---------------------------------------------------------------------------
HD inline float cartpole_energy(float x, float xdot, float theta, float thdot,
                                float mc, float mp, float l)
{
    const float sin_th = sinf(theta);
    const float cos_th = cosf(theta);

    // Pole center-of-mass position and velocity in the WORLD frame (cart
    // frame's x plus the pole's swing; theta=0 upright => y = +l at top).
    const float xdot_pole = xdot + l * cos_th * thdot;
    const float ydot_pole = -l * sin_th * thdot;
    const float y_pole = l * cos_th;

    const float ke_cart = 0.5f * mc * xdot * xdot;
    // Rigid-rod moment of inertia about its OWN center of mass:
    // I_cm = (1/12) * mp * (2l)^2 = (1/3) * mp * l^2 (the same "4/3" family
    // of constants that appears in cartpole_deriv's th_acc denominator —
    // THEORY.md ties the two together).
    const float i_cm = (1.0f / 3.0f) * mp * l * l;
    const float ke_pole = 0.5f * mp * (xdot_pole * xdot_pole + ydot_pole * ydot_pole)
                         + 0.5f * i_cm * thdot * thdot;
    const float pe_pole = mp * kGravity * y_pole;   // cart's track is the y=0 reference

    return ke_cart + ke_pole + pe_pole;
}

// ---------------------------------------------------------------------------
// Host-callable launchers (defined in kernels.cu). buf's device pointers
// must already be allocated (N floats/ints each) by the caller (main.cu);
// these functions allocate nothing and free nothing.
// ---------------------------------------------------------------------------

// launch_farm_init — one thread per environment: seed, randomize
// mass/length, take the first episode reset. Must be called exactly once
// per FarmBuffers instance before launch_farm_step.
void launch_farm_init(const FarmBuffers& buf, int N, uint32_t base_seed,
                      float dr_mc, float dr_mp, float dr_l, float theta0_range);

// launch_farm_step — one thread per environment, each running `steps`
// ticks INTERNALLY in a loop (the whole run is ONE kernel launch — see
// kernels.cu's header comment for why). Safe to call more than once on
// the same buffers (e.g. to checkpoint/measure partway through a run);
// this project calls it once for the §5 verify subset and once for the
// full farm.
void launch_farm_step(const FarmBuffers& buf, int N, int steps,
                      float Kx, float Kxd, float Kth, float Kthd,
                      float theta0_range, int episode_cap);

// ---------------------------------------------------------------------------
// CPU references (reference_cpu.cpp).
// ---------------------------------------------------------------------------

// farm_init_cpu / farm_step_cpu — the oracle twins of the two launchers
// above: same per-env logic (farm_init_one_env / farm_step_one_env,
// literally shared via this header), sequential over environments instead
// of parallel. main.cu runs these against the GPU on the kVerifyEnvs
// subset and requires agreement within documented tolerances (the §5 gate).
void farm_init_cpu(const FarmBuffers& buf, int N, uint32_t base_seed,
                   float dr_mc, float dr_mp, float dr_l, float theta0_range);
void farm_step_cpu(const FarmBuffers& buf, int N, int steps,
                   float Kx, float Kxd, float Kth, float Kthd,
                   float theta0_range, int episode_cap);

// energy_conservation_cpu — the physics-invariant diagnostic: integrate
// ONE undriven (u=0), unbounded (no fail box, no reset) cart-pole for
// `steps` ticks from initial angle theta0 (nominal mass/length), writing
// [steps+1] total-energy samples (including the t=0 sample) to energy_out
// and the final state to final_state_out[4] (may be nullptr if unwanted).
// CPU-only by design: this checks the INTEGRATOR, not the farm, and a
// single trajectory needs no GPU parallelism (THEORY.md §how we verify).
void energy_conservation_cpu(int steps, float theta0,
                             float mc, float mp, float l,
                             float* energy_out, float* final_state_out);

#endif // PROJECT_KERNELS_CUH
