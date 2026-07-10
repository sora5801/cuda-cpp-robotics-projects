// ===========================================================================
// kernels.cuh — interface for project 18.01
//               Snake robots: serpenoid gait sweeps (anisotropic-friction
//               lateral undulation on flat ground; granular/DEM coupling
//               documented as the 10.10 milestone, not implemented here)
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (the sweep orchestrator + gate checker),
// kernels.cu (the GPU sweep kernel), and reference_cpu.cpp (the CPU oracle
// + the single-gait diagnostics). Everything all three must agree on — the
// snake's geometry, the serpenoid gait formula, the anisotropic-friction
// force law, the prescribed-joint dynamics, and the sweep-grid indexing —
// is defined HERE, once (CLAUDE.md §12).
//
// The physics in five lines (THEORY.md derives it properly):
//   1. A planar N=12-link chain. Joint j's angle is PRESCRIBED exactly by
//      the serpenoid wave phi_j(t) = A*sin(omega*t + j*beta) + gamma — no
//      joint dynamics to solve, the shape is a known function of time.
//   2. That leaves exactly 3 DYNAMIC degrees of freedom: the head link's
//      pose (x, y, yaw) — everything else is forward kinematics off it.
//   3. Every link drags on the ground with ANISOTROPIC Coulomb friction
//      (mu_t along its own axis, mu_n across it, mu_t << mu_n — THIS
//      asymmetry is what a symmetric wiggle turns into net forward thrust;
//      THEORY.md proves isotropic friction cannot).
//   4. Sum every link's friction force/torque about the head -> Newton's
//      law for the WHOLE snake (constant total mass + a nominal moment of
//      inertia) gives the head's acceleration -> semi-implicit Euler.
//   5. The GPU content is the SWEEP: one thread per (amplitude, phase
//      offset, temporal frequency) triple, each simulating its own T-second
//      gait independently — an embarrassingly parallel batched-simulation
//      map, the same shape as 08.01's rollouts / 10.03's environment farm,
//      here applied to a design-space search instead of control or RL.
//
// THE PRESCRIBED-JOINT TRICK (why this is tractable at all):
// A full N=12-link planar multibody snake has 3+11=14 DOF and needs a
// generalized mass matrix + constraint solve (Featherstone-class dynamics)
// every step — real robotics-sim engineering, not a first CUDA project.
// This project's catalog bullet asks for "serpenoid gait sweeps", i.e. the
// SHAPE is not something we are solving for — it is EXACTLY what the gait
// formula says at every instant (real snake-robot joint controllers track
// their reference angle to a few degrees; this project idealizes that
// tracking as PERFECT, a documented simplification, THEORY.md §numerics).
// With the shape known, the only unknowns are how the whole shape's pose
// drifts through the world — 3 ODOF (x, y, yaw), 6 first-order ODEs
// (x, y, yaw, vx, vy, yaw_rate) — small enough to integrate in registers,
// once per simulated millisecond, for thousands of gaits in parallel.
//
// STATE LAYOUT (per simulated gait) — SI units, right-handed, +z out of
// the page (world top-down view; CLAUDE.md §12 body convention):
//     x, y        head-link CENTER position (m), world frame
//     yaw         head-link heading (rad), 0 = +x axis, CCW positive
//     vx, vy      head-link CENTER velocity (m/s), world frame
//     yaw_rate    head-link angular velocity (rad/s)
// Every other link's pose/velocity is FORWARD KINEMATICS off this state —
// see snake_step() below — never stored as independent dynamic state.
//
// GAIT PARAMETERIZATION (Hirose's serpenoid curve, THEORY.md §the-math):
//     phi_j(t) = A * sin(omega*t + j*beta) + gamma,   j = 0 .. kNLinks-2
//         A     amplitude (rad)      — how far each joint swings
//         beta  phase offset (rad)   — sets the body wavelength
//         omega temporal frequency (rad/s) — how fast the wave travels
//         gamma turning bias (rad)  — a common offset added to every
//               joint; ONLY used by the turning-bias verification gate,
//               always 0 in the main (A, beta, omega) sweep (straight-
//               line locomotion is what the sweep searches for).
//
// THE SWEEP GRID — one thread per (a_idx, b_idx, w_idx) triple, flattened
// as g = (a_idx * n_beta + b_idx) * n_omega + w_idx (row-major, omega
// fastest-varying — matches the nested-loop order a reader would write by
// hand). Grid shape and ranges are SCENARIO DATA (kDefault* below are the
// documented fallback; data/sample/snake_scenario.csv is the committed
// values — CLAUDE.md §8: "snake parameters + sweep ranges + friction
// coefficients" are the task's data, not baked-in constants).
//
// WHY N_LINKS STAYS COMPILE-TIME: every gait's forward-kinematics pass
// needs small FIXED-SIZE scratch arrays (per-link position/force/tangent)
// living in registers/local memory inside the kernel — CUDA device code
// cannot size a local array from a runtime value. kNLinks is therefore a
// constexpr, not a scenario field, unlike the physical/gait parameters
// around it (documented scope limit; README Exercise 5 discusses lifting
// it via a template parameter or dynamic global scratch).
//
// Read this after: README.md/THEORY.md. Read this before: kernels.cu.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cstdint>
#include <cmath>     // sinf/cosf/sqrtf/fabsf/fmaxf — host declarations; nvcc
                     // supplies device overloads of the SAME names below

// ---------------------------------------------------------------------------
// HD — "__host__ __device__" under nvcc, nothing under cl.exe. The SAME
// trick 10.03 uses (see that project's kernels.cuh for the full rationale):
// it lets snake_step() — the one place the physics is written — be called
// UNCHANGED from the GPU sweep kernel (kernels.cu), the CPU oracle
// (reference_cpu.cpp, compiled by cl.exe, never sees a CUDA keyword), and
// main.cu's diagnostic single-gait runs (main.cu IS a .cu file, so it sees
// the __host__ __device__ version and simply calls it as ordinary C++).
// Sharing the source removes hand-copy divergence as an explanation for any
// GPU-vs-CPU difference the §5 VERIFY gate measures — what is left over is
// purely sinf/cosf's independently-rounded host vs. device implementations
// (THEORY.md §numerical considerations).
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ---------------------------------------------------------------------------
// kNLinks — number of rigid links in the snake (compile-time; see the file
// header's "why N_LINKS stays compile-time" note). 12 links of 0.10 m each
// gives a 1.2 m snake — comparable to real research snake robots (CMU
// modsnake, ACM-R5) and long enough to show several wavelengths of a
// serpenoid wave without the per-thread scratch arrays getting unwieldy.
// ---------------------------------------------------------------------------
constexpr int kNLinks = 12;

// ---------------------------------------------------------------------------
// Numerical-method constant (NOT scenario data — this is a discretization
// choice, like 08.01's "RK4 not Euler", not a property of the robot or the
// experiment). Coulomb friction's true force law is DISCONTINUOUS at zero
// relative velocity (a signum function) — integrating that directly at any
// finite dt makes the friction force chatter (flip sign every step once
// velocity oscillates near zero, injecting numerical energy). The standard,
// honest fix is a SMOOTHED signum: v / sqrt(v^2 + eps^2), which is C^1,
// approaches true Coulomb friction as |v| >> eps, and costs one sqrt.
// kFrictionEpsMps is that eps, in m/s — chosen well below the gaits' typical
// link speeds (~0.05-0.5 m/s, THEORY.md §numerics measures the ratio) so
// the smoothing barely perturbs genuine sliding while killing the chatter.
// ---------------------------------------------------------------------------
constexpr float kFrictionEpsMps = 0.01f;

// ---------------------------------------------------------------------------
// Scenario DEFAULTS. The committed data/sample/snake_scenario.csv overrides
// every one of these (CLAUDE.md §8); the constants exist so THEORY.md/
// README.md can cite fixed numbers and so a missing/malformed scenario file
// has a documented fallback shape. kVerifyGaits/kVerifyStride are NOT
// scenario-overridable: they size the §5 GPU-vs-CPU gate, part of the
// verification harness, not the experiment (mirrors 10.03's kVerifyEnvs).
// ---------------------------------------------------------------------------
constexpr float kDefaultLinkLenM   = 0.10f;   // per-link length (m); 12 links -> 1.2 m snake
constexpr float kDefaultLinkMassKg = 0.15f;   // per-link mass (kg); 12 links -> 1.8 kg total
constexpr float kDefaultGravity    = 9.81f;   // m/s^2
constexpr float kDefaultMuT        = 0.10f;   // tangential (along-link) Coulomb coefficient — LOW
constexpr float kDefaultMuN        = 0.70f;   // normal (across-link) Coulomb coefficient — HIGH;
                                              // mu_n/mu_t = 7 is the anisotropy ratio THEORY.md
                                              // proves is necessary for net propulsion

constexpr int   kDefaultNAmp     = 32;    // amplitude grid points
constexpr float kDefaultAmpMinR  = 0.05f; // rad (~2.9 deg) — smallest joint swing tested
constexpr float kDefaultAmpMaxR  = 1.05f; // rad (~60.2 deg) — largest joint swing tested
constexpr int   kDefaultNBeta    = 32;    // phase-offset grid points
constexpr float kDefaultBetaMinR = 0.10f; // rad — near-zero inter-joint phase (long wavelength)
constexpr float kDefaultBetaMaxR = 3.00f; // rad — near-pi inter-joint phase (short wavelength,
                                          // close to alternating left/right)
constexpr int   kDefaultNOmega    = 8;    // temporal-frequency grid points
constexpr float kDefaultOmegaMinR = 1.0f; // rad/s -> 6.28 s period
constexpr float kDefaultOmegaMaxR = 6.0f; // rad/s -> 1.05 s period

constexpr float kDefaultTSimS = 8.0f;    // simulated seconds per gait (long enough for several
                                         // gait cycles even at the slowest omega in range)
constexpr float kDefaultDtS   = 0.001f;  // 1 ms integration step (THEORY.md §numerics justifies
                                         // this against the Coulomb-friction stiffness above)
constexpr float kDefaultTurnGammaR = 0.15f; // rad (~8.6 deg) — turning-bias gate's test offset

constexpr int kVerifyCount = 32;  // how many of the G sweep results the §5 gate spot-checks
                                  // against a from-scratch CPU recomputation (stride-sampled
                                  // across the whole grid — see main.cu)

// ---------------------------------------------------------------------------
// GaitGridParams — the SWEEP's shape: how many points along each of the 3
// gait axes, and the inclusive [min, max] range each axis spans. Plain POD
// so it can be passed BY VALUE into the kernel launch (a few dozen bytes,
// well under any parameter-size limit — the same choice 10.03's smaller
// value-structs make).
// ---------------------------------------------------------------------------
struct GaitGridParams {
    int   n_amp;    float amp_min_r,   amp_max_r;    // rad
    int   n_beta;   float beta_min_r,  beta_max_r;    // rad
    int   n_omega;  float omega_min_r, omega_max_r;   // rad/s
};

// ---------------------------------------------------------------------------
// SimParams — the ROBOT + INTEGRATION settings shared by every gait in a
// run: link geometry/mass, gravity, and the fixed integration step/count.
// Friction coefficients are deliberately NOT here (see GaitParams below) —
// the isotropic-friction verification gate re-runs a gait with DIFFERENT
// mu_t/mu_n, so friction is per-gait, not per-simulation-run.
// ---------------------------------------------------------------------------
struct SimParams {
    float link_len_m;
    float link_mass_kg;
    float gravity_mps2;
    float dt_s;
    int   n_steps;    // = round(T_sim_s / dt_s); computed once on the host
};

// ---------------------------------------------------------------------------
// GaitParams — ONE fully-specified gait: the four serpenoid parameters plus
// the friction pair it experiences. Produced either by decode_gait() (for
// a sweep-grid index) or written by hand (the zero-amplitude / isotropic-
// friction / turning-bias verification gates in main.cu / reference_cpu.cpp).
// ---------------------------------------------------------------------------
struct GaitParams {
    float amp_r;      // A     (rad)
    float beta_r;      // beta  (rad)
    float omega_rps;   // omega (rad/s)
    float gamma_r;      // gamma (rad) — turning bias; 0 for the main sweep
    float mu_t;         // tangential friction coefficient (unitless)
    float mu_n;         // normal friction coefficient (unitless)
};

// ---------------------------------------------------------------------------
// GaitResult — everything the sweep records about one simulated gait.
// distance/path_length/straightness/effort/cot are all DERIVED at the end
// of simulate_gait() from the raw trajectory; nothing here needs to persist
// per-timestep data once the gait has been simulated.
// ---------------------------------------------------------------------------
struct GaitResult {
    float final_x_m, final_y_m, final_yaw_r; // head pose at t = T_sim
    float distance_m;       // |final_x, final_y| — net straight-line displacement from the origin
    float path_length_m;    // arc length actually traveled (sum of |v|*dt every step)
    float straightness;     // distance_m / path_length_m, in (0, 1] (1 = perfectly straight)
    float effort_j;         // sum_j |tau_j * phidot_j| dt — the raw actuation-effort proxy (J)
    float cot;              // effort_j / (total_weight_N * distance_m) — normalized cost of
                            // transport (THEORY.md §the-math; unitless, lower is better)
};

// ---------------------------------------------------------------------------
// grid_lerp — map an integer grid index i in [0, n) onto a float in
// [lo, hi], inclusive at both ends (i=0 -> lo, i=n-1 -> hi). n<=1 returns
// lo (degenerate single-point "sweep" — defensive, never hit by the
// committed scenario but cheap to guard against a malformed one).
// ---------------------------------------------------------------------------
HD inline float grid_lerp(float lo, float hi, int i, int n)
{
    if (n <= 1) return lo;
    return lo + (hi - lo) * (static_cast<float>(i) / static_cast<float>(n - 1));
}

// ---------------------------------------------------------------------------
// decode_gait — turn a flattened sweep index g into the GaitParams it names
// (gamma fixed at 0: the main sweep only ever searches straight-line
// gaits). Inverse of the flattening g = (a_idx*n_beta + b_idx)*n_omega +
// w_idx used everywhere else in this project (kernels.cu, reference_cpu.cpp,
// main.cu all decode the SAME way, from this ONE function).
// ---------------------------------------------------------------------------
HD inline GaitParams decode_gait(int g, const GaitGridParams& grid, float mu_t, float mu_n)
{
    const int w_idx = g % grid.n_omega;
    const int tmp    = g / grid.n_omega;
    const int b_idx  = tmp % grid.n_beta;
    const int a_idx  = tmp / grid.n_beta;

    GaitParams p;
    p.amp_r    = grid_lerp(grid.amp_min_r,   grid.amp_max_r,   a_idx, grid.n_amp);
    p.beta_r    = grid_lerp(grid.beta_min_r,  grid.beta_max_r,  b_idx, grid.n_beta);
    p.omega_rps = grid_lerp(grid.omega_min_r, grid.omega_max_r, w_idx, grid.n_omega);
    p.gamma_r   = 0.0f;
    p.mu_t = mu_t;
    p.mu_n = mu_n;
    return p;
}

// ---------------------------------------------------------------------------
// link_friction_force — the anisotropic Coulomb friction law for ONE link,
// evaluated from its CENTER velocity (vpx, vpy) and its tangent direction
// (tx, ty) = (cos theta_i, sin theta_i) (world frame; the link's normal is
// therefore (-ty, tx), a quarter-turn CCW — THEORY.md draws the diagram).
//
//   v_t = v . t_hat   (component ALONG the link's long axis)
//   v_n = v . n_hat   (component ACROSS the link's long axis)
//   f_t = -mu_t * (m*g) * smoothsign(v_t)     LOW resistance: the snake's
//                                              belly slides easily fore/aft
//   f_n = -mu_n * (m*g) * smoothsign(v_n)     HIGH resistance: the snake's
//                                              side "bites" into the ground
//
// smoothsign(v) = v / sqrt(v^2 + eps^2) — see kFrictionEpsMps's comment.
// m*g is the per-link NORMAL LOAD (weight) that sets the Coulomb force
// scale — this project has no vertical dynamics (flat ground, top-down
// view), so weight is simply the static per-link weight, never recomputed.
//
// THIS FUNCTION IS THE HEART OF THE WHOLE PROJECT (THEORY.md §the-problem):
// mu_t != mu_n is the ONE assumption that turns a side-to-side wiggle into
// forward thrust; set mu_t = mu_n and every link's friction force becomes
// (for straight sliding) exactly anti-parallel to its own velocity with no
// preferred direction — the isotropic-friction verification gate (main.cu)
// re-runs the best gait with mu_t=mu_n and MEASURES the propulsion collapse
// this predicts.
// ---------------------------------------------------------------------------
HD inline void link_friction_force(float vpx, float vpy, float tx, float ty,
                                   float mu_t, float mu_n, float link_mass_kg, float gravity_mps2,
                                   float& fx, float& fy)
{
    const float nx = -ty, ny = tx;                 // link's NORMAL direction (quarter-turn CCW of tangent)
    const float vt = vpx * tx + vpy * ty;           // velocity component ALONG the link (m/s)
    const float vn = vpx * nx + vpy * ny;           // velocity component ACROSS the link (m/s)
    const float weight_n = link_mass_kg * gravity_mps2;  // per-link normal load (N) — the Coulomb force scale

    // Smoothed signum (see kFrictionEpsMps): sqrtf, not rsqrtf — rsqrtf is a
    // CUDA-only device intrinsic with no MSVC host equivalent, and this
    // function must compile identically under cl.exe (reference_cpu.cpp)
    // and nvcc (kernels.cu) — portability over the last few percent of speed.
    const float ft = -mu_t * weight_n * vt / sqrtf(vt * vt + kFrictionEpsMps * kFrictionEpsMps);
    const float fn = -mu_n * weight_n * vn / sqrtf(vn * vn + kFrictionEpsMps * kFrictionEpsMps);

    fx = ft * tx + fn * nx;   // rotate (f_t, f_n) from the link's own (tangent, normal) frame
    fy = ft * ty + fn * ny;   // back into world coordinates
}

// ---------------------------------------------------------------------------
// snake_step — advance the 3-DOF body state (x, y, yaw, vx, vy, yaw_rate)
// by ONE dt, under the prescribed serpenoid shape evaluated at time t.
//
// Algorithm (THEORY.md §the-algorithm walks this with a diagram):
//   1. FORWARD PASS (link 0 .. kNLinks-1): walk the chain from the head,
//      computing each link's orientation/position/velocity from the
//      PRESCRIBED joint angles/rates and the current body state, then its
//      anisotropic friction force — accumulating the NET force and the NET
//      torque about the head as we go. O(kNLinks), registers only.
//   2. BACKWARD SUFFIX PASS (link kNLinks-1 .. 1): recover an ESTIMATED
//      torque at every joint via a free-body "cut" argument (THEORY.md
//      derives it), used only for the cost-of-transport effort integral.
//   3. NEWTON'S LAW for the whole snake (constant total mass + a nominal
//      moment of inertia about the head) -> linear/angular acceleration.
//   4. SEMI-IMPLICIT (symplectic) EULER: update velocity from force FIRST,
//      then position from the NEW velocity — unconditionally more stable
//      than explicit Euler for this kind of velocity-dependent force law,
//      and it is what keeps path_len_accum EXACT (see the inline note).
//
// Parameters:
//   x,y,yaw,vx,vy,yaw_rate  — IN/OUT: the 3-DOF body state (SI; see the
//                             file header's state-layout note).
//   t                       — current simulated time (s); ONLY used to
//                             evaluate the prescribed joint angles/rates —
//                             never accumulated across calls (each call
//                             computes t*omega fresh, so there is no time-
//                             accumulation drift; see THEORY.md §numerics).
//   dt                      — integration step (s).
//   amp,beta,omega,gamma    — this gait's serpenoid parameters (rad, rad,
//                             rad/s, rad).
//   link_len_m,link_mass_kg,gravity_mps2,mu_t,mu_n — robot + ground params.
//   path_len_accum          — IN/OUT: running sum of |v|*dt (m) — the
//                             actual polyline length of the head's path.
//   effort_accum            — IN/OUT: running sum of |tau_j*phidot_j|*dt
//                             over all joints (J) — the raw COT numerator.
// ---------------------------------------------------------------------------
HD inline void snake_step(float& x, float& y, float& yaw,
                          float& vx, float& vy, float& yaw_rate,
                          float t, float dt,
                          float amp, float beta, float omega, float gamma,
                          float link_len_m, float link_mass_kg, float gravity_mps2,
                          float mu_t, float mu_n,
                          float& path_len_accum, float& effort_accum)
{
    // Per-link scratch, kept small on purpose (see the file header's
    // register-footprint note): position and friction force (needed by
    // BOTH passes) plus the tangent direction (needed to locate each joint
    // in the backward pass). Velocity is used only transiently within the
    // forward pass and is therefore NOT stored per-link.
    float PX[kNLinks], PY[kNLinks];      // link CENTER positions (m, world frame)
    float FX[kNLinks], FY[kNLinks];      // per-link friction force (N, world frame)
    float TX[kNLinks], TY[kNLinks];      // per-link tangent unit vector (unitless)
    float phidot[kNLinks - 1];           // joint angular RATES (rad/s) — reused by the backward pass

    const float half_len = 0.5f * link_len_m;

    // ---- link 0 (the head): seeded directly from the body state --------
    float theta = yaw;          // running link orientation (rad)
    float theta_dot = yaw_rate; // running link angular velocity (rad/s)
    float px = x, py = y;       // running link CENTER position (m)
    float pvx = vx, pvy = vy;   // running link CENTER velocity (m/s)

    TX[0] = cosf(theta);
    TY[0] = sinf(theta);
    PX[0] = px;
    PY[0] = py;
    link_friction_force(pvx, pvy, TX[0], TY[0], mu_t, mu_n, link_mass_kg, gravity_mps2, FX[0], FY[0]);

    // ---- links 1 .. kNLinks-1: walk the chain, one JOINT at a time ------
    // Joint j (0-indexed, connecting link j and link j+1) carries the
    // serpenoid angle phi_j(t) = A*sin(omega*t + j*beta) + gamma — this IS
    // the catalog's formula with the paper's 1-indexed "(i-1)" turned into
    // the 0-indexed "j" used throughout this codebase.
    for (int i = 1; i < kNLinks; ++i) {
        const int j = i - 1;                         // joint index (0 .. kNLinks-2)
        const float arg = omega * t + static_cast<float>(j) * beta;
        const float sp = sinf(arg), cp = cosf(arg);
        const float phi  = amp * sp + gamma;          // this joint's PRESCRIBED angle (rad)
        const float phid = amp * omega * cp;          // its PRESCRIBED angular rate (rad/s) — d(phi)/dt
        phidot[j] = phid;                             // stashed for the backward (torque) pass below

        // Link (i-1)'s tangent/normal BEFORE we advance theta — needed for
        // the "far end of link i-1" term in the position/velocity update.
        const float tx_prev = TX[i - 1], ty_prev = TY[i - 1];
        const float nx_prev = -ty_prev,  ny_prev = tx_prev;
        const float theta_dot_prev = theta_dot;       // link (i-1)'s angular velocity, captured

        theta     += phi;    // link i's orientation = link (i-1)'s + this joint's prescribed angle
        theta_dot += phid;   // link i's angular velocity, by the same chain rule (d/dt of the line above)

        const float s = sinf(theta), c = cosf(theta);
        TX[i] = c; TY[i] = s;
        const float nx = -s, ny = c;                  // link i's normal (quarter-turn CCW of its tangent)

        // p_i = p_{i-1} + (L/2)*t_{i-1} + (L/2)*t_i  — the two link HALVES
        // meeting at the joint between them (rigid-rod chain kinematics).
        px += half_len * tx_prev + half_len * c;
        py += half_len * ty_prev + half_len * s;
        // v_i = d/dt[p_i] — the EXACT time-derivative of the line above:
        // each (L/2)*t_k term differentiates to (L/2)*theta_dot_k*n_k.
        pvx += half_len * theta_dot_prev * nx_prev + half_len * theta_dot * nx;
        pvy += half_len * theta_dot_prev * ny_prev + half_len * theta_dot * ny;

        PX[i] = px; PY[i] = py;
        link_friction_force(pvx, pvy, TX[i], TY[i], mu_t, mu_n, link_mass_kg, gravity_mps2, FX[i], FY[i]);
    }

    // ---- net force / torque about the HEAD's position (x, y) -----------
    // (x, y) are the state values from the START of this step — the
    // reference point the whole-snake Newton-Euler balance is taken about.
    float fx_net = 0.0f, fy_net = 0.0f, torque_net = 0.0f;
#ifdef __CUDACC__
#pragma unroll   // nvcc-only hint (kNLinks=12 is small and fixed) — cl.exe
#endif           // does not understand this pragma, hence the guard: this
                 // function is compiled by BOTH compilers (see the HD note above)
    for (int i = 0; i < kNLinks; ++i) {
        fx_net += FX[i];
        fy_net += FY[i];
        // 2D cross product (r x F)_z = r.x*F.y - r.y*F.x, r = link center
        // relative to the head — THEORY.md's right-handed-frame convention.
        torque_net += (PX[i] - x) * FY[i] - (PY[i] - y) * FX[i];
    }

    // ---- backward suffix pass: per-JOINT torque estimate for the COT ---
    // tau_k (k = joint index, 0..kNLinks-2) is estimated by "cutting" the
    // chain at joint k and summing the friction torque of every link
    // DOWNSTREAM of it (i = k+1 .. kNLinks-1) about the joint's own
    // location — a standard free-body argument (THEORY.md derives it),
    // computed here in O(kNLinks) via a running SUFFIX sum instead of the
    // naive O(kNLinks^2) all-pairs sum.
    float sx = 0.0f, sy = 0.0f, s_pxf = 0.0f;   // suffix force sum + suffix (p x F) about the WORLD origin
    float effort_delta = 0.0f;                  // this step's |tau*phidot| sum over all joints (J worth, pre-dt)
    for (int i = kNLinks - 1; i >= 1; --i) {
        sx += FX[i];
        sy += FY[i];
        s_pxf += PX[i] * FY[i] - PY[i] * FX[i];
        const int k = i - 1;                     // joint k sits between link k and link (k+1) = link i
        const float jx = PX[k] + half_len * TX[k];  // joint k's position = link k's FAR end
        const float jy = PY[k] + half_len * TY[k];
        // Torque of the downstream sum ABOUT the joint = (torque about the
        // world origin) minus (joint position) x (downstream force sum) —
        // the standard "shift the moment reference point" identity.
        const float tau_k = s_pxf - (jx * sy - jy * sx);
        effort_delta += fabsf(tau_k * phidot[k]);
    }
    effort_accum += effort_delta * dt;

    // ---- Newton's law for the whole snake, semi-implicit Euler ----------
    const float total_mass_kg = static_cast<float>(kNLinks) * link_mass_kg;
    // Nominal moment of inertia of a UNIFORM ROD of the snake's full length
    // (kNLinks*link_len_m), about ITS OWN END — matching the reference
    // point (the head, link 0's near end sits at the very front of the
    // chain). I_end = (1/3)*M*Ltot^2 (parallel-axis from the rod's
    // about-center (1/12)*M*Ltot^2 + M*(Ltot/2)^2). Held CONSTANT rather
    // than recomputed from the instantaneous bent shape — a documented
    // simplification (THEORY.md §numerics quantifies the error it costs).
    const float total_len_m = static_cast<float>(kNLinks) * link_len_m;
    const float i_eff = (1.0f / 3.0f) * total_mass_kg * total_len_m * total_len_m;

    const float ax = fx_net / total_mass_kg;
    const float ay = fy_net / total_mass_kg;
    const float alpha = torque_net / i_eff;

    // Semi-implicit (symplectic) Euler: velocity from force FIRST...
    vx += ax * dt;
    vy += ay * dt;
    yaw_rate += alpha * dt;
    // ...then position from the just-updated velocity. This ordering is
    // WHY path_len_accum below is the EXACT polyline length of the path
    // actually taken (dx, dy) = (vx, vy)*dt is not an approximation of the
    // step displacement, it IS the step displacement.
    x += vx * dt;
    y += vy * dt;
    yaw += yaw_rate * dt;

    path_len_accum += sqrtf(vx * vx + vy * vy) * dt;
}

// ---------------------------------------------------------------------------
// simulate_gait — run ONE gait from rest at the origin for n_steps, and
// reduce the resulting trajectory into a GaitResult. This is the ENTIRE
// per-thread body of the sweep kernel (kernels.cu) and is called directly,
// unchanged, by main.cu's diagnostic single-gait runs (zero-amplitude,
// isotropic-friction, turning-bias) and by reference_cpu.cpp's oracle —
// ONE simulation loop, four call sites, CLAUDE.md §12's single-source rule.
//
// Initial condition: x=y=yaw=vx=vy=yaw_rate=0 at t=0 — note the JOINTS are
// generally NOT at rest at t=0 (phi_j(0) = A*sin(j*beta) + gamma is
// whatever the serpenoid formula says), i.e. every gait starts "mid-wave"
// rather than from a dead-straight pose. This is deliberate: there is no
// physical reason a gait must begin straight, and starting mid-wave avoids
// biasing the measured average speed with an artificial straight-to-wavy
// startup transient (THEORY.md §how-we-verify).
// ---------------------------------------------------------------------------
HD inline void simulate_gait(const GaitParams& gp, const SimParams& sim, GaitResult& out)
{
    float x = 0.0f, y = 0.0f, yaw = 0.0f;
    float vx = 0.0f, vy = 0.0f, yaw_rate = 0.0f;
    float path_len = 0.0f, effort = 0.0f;

    for (int s = 0; s < sim.n_steps; ++s) {
        const float t = static_cast<float>(s) * sim.dt_s;   // fresh product each step: no drift in t itself
        snake_step(x, y, yaw, vx, vy, yaw_rate, t, sim.dt_s,
                  gp.amp_r, gp.beta_r, gp.omega_rps, gp.gamma_r,
                  sim.link_len_m, sim.link_mass_kg, sim.gravity_mps2, gp.mu_t, gp.mu_n,
                  path_len, effort);
    }

    out.final_x_m = x;
    out.final_y_m = y;
    out.final_yaw_r = yaw;
    out.distance_m = sqrtf(x * x + y * y);
    out.path_length_m = path_len;
    // Guard both ratios against division by ~0 — a near-stationary gait
    // (e.g. the zero-amplitude gate) legitimately has path_length_m == 0.
    out.straightness = (path_len > 1.0e-9f) ? (out.distance_m / path_len) : 0.0f;
    out.effort_j = effort;
    const float total_weight_n = static_cast<float>(kNLinks) * sim.link_mass_kg * sim.gravity_mps2;
    out.cot = effort / (total_weight_n * fmaxf(out.distance_m, 1.0e-6f));
}

// ---------------------------------------------------------------------------
// Host-callable launcher (defined in kernels.cu). Device output arrays are
// allocated by the caller (main.cu), G floats each; this function allocates
// nothing and frees nothing (stateless, like 08.01's launch_mppi_rollouts).
// ---------------------------------------------------------------------------
void launch_sweep(const GaitGridParams& grid, const SimParams& sim, float mu_t, float mu_n,
                  float* d_distance_m, float* d_straightness, float* d_cot, float* d_effort_j,
                  float* d_final_x_m, float* d_final_y_m, int G);

// ---------------------------------------------------------------------------
// CPU references (reference_cpu.cpp).
// ---------------------------------------------------------------------------

// sweep_cpu — the oracle twin of launch_sweep, but only for the listed
// indices (the §5 VERIFY gate's spot-check subset — see kVerifyCount).
// Decodes each index with the SAME decode_gait() the kernel uses, so any
// indexing/flattening bug shows up as a real disagreement, not a silent
// mismatch of what is being compared.
void sweep_cpu(const int* indices, int n_indices, const GaitGridParams& grid, const SimParams& sim,
              float mu_t, float mu_n, GaitResult* out);

// run_single_gait_cpu — simulate ONE fully-specified gait (used by the
// zero-amplitude / isotropic-friction / turning-bias verification gates in
// main.cu, where the friction coefficients or gamma are deliberately NOT
// the sweep's defaults).
void run_single_gait_cpu(const GaitParams& gp, const SimParams& sim, GaitResult& out);

// run_single_gait_logged_cpu — same simulation as run_single_gait_cpu, but
// also samples (t, x, y, yaw) every log_stride steps into the caller's
// buffers (capacity max_log_rows; raw pointers, not std::vector — kept
// consistent with this header's STL-free style). Returns the number of
// rows actually written. Used once, for the best-gait trajectory artifact.
int run_single_gait_logged_cpu(const GaitParams& gp, const SimParams& sim, int log_stride,
                               int max_log_rows, float* t_log, float* x_log, float* y_log,
                               float* yaw_log, GaitResult& out);

#endif // PROJECT_KERNELS_CUH
