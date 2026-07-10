// ===========================================================================
// kernels.cu — GPU implementation for project 23.01
//              GPU costmaps: inflation, raytrace clearing, multi-layer fusion
//              + a DWA local-planner consumer, closed loop
//
// Four kernels, three GPU access patterns, one costmap pipeline
// ---------------------------------------------------------------
//   raytrace_kernel   — one thread per LiDAR BEAM (360 threads): a Bresenham
//                       grid walk that both MARKS an obstacle cell and
//                       CLEARS everything in front of it — and races with
//                       every other beam that happens to cross the same
//                       cell. This file's centerpiece comment (below) works
//                       through that race honestly and shows why atomicMax
//                       resolves it deterministically and SAFELY.
//   inflation_kernel  — one thread per CELL (65536 threads): a bounded
//                       (2R+1)^2 gather, the same "stencil" family as
//                       07.09's jump-flooding pass — but brute-force, not
//                       propagated, because R=10 is small enough that
//                       O(W*H*R^2) is trivial on a GPU and the point here
//                       is inflation, not approximate nearest-seed search
//                       (07.09 is the project for that; this one is
//                       deliberately self-contained, per the task brief).
//   fusion_kernel     — one thread per CELL: the scaffold's SAXPY map
//                       pattern, doing real work — per-cell max of three
//                       independent layers.
//   dwa_score_kernel  — one thread per (v,w) SAMPLE (4096 threads): 08.01's
//                       MPPI rollout pattern reused for SCORING instead of
//                       control-blending — every thread independently
//                       simulates one candidate arc against the master
//                       costmap this tick just produced.
//
// Read this after: kernels.cuh.  Companion oracle: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"      // CUDA_CHECK / CUDA_CHECK_LAST_ERROR (§6.1 rule 7)

#include <cstdio>
#include <cstdlib>
#include <cmath>

// ===========================================================================
// raytrace_kernel — mark + clear one LiDAR beam into the obstacle layer.
//
// THE RACE, worked through honestly (CLAUDE.md task brief: "TEACH the race
// honestly")
// ----------------------------------------------------------------------
// Two DIFFERENT beams can legitimately disagree about the SAME cell in the
// SAME tick. Concretely: beam A's ray hits an obstacle at cell X (3 m out);
// beam B is one degree over and, because 360 beams over a full circle means
// beams only ~3-4 cm apart in angle at 6 m range, B's discretized Bresenham
// path can clip the same cell X on its way to a DIFFERENT, farther hit (or
// to max range with no hit at all) — a routine rasterization-aliasing
// event near any obstacle's silhouette, not a contrived edge case. Beam A
// wants to WRITE cell X = kCostLethal (254); beam B wants to WRITE the same
// cell X = kCostFree (0), because as far as B's ray is concerned, X is just
// a cell it passed through on the way to somewhere else.
//
// If both writes were plain assignments (obstacle_layer[idx] = value;),
// CUDA's execution model gives NO ordering guarantee between threads in
// different warps/blocks — whichever thread's store physically lands in
// memory LAST wins, and that order is unspecified by the language and can
// vary between runs, GPU architectures, or even occupancy-driven scheduling
// decisions on the SAME GPU. If B's 0 lands after A's 254, cell X — a REAL
// obstacle — silently reads back as free space. That is not a performance
// bug; it is a correctness bug with safety consequences (a planner could
// route the robot straight through it).
//
// THE FIX exploits the one fact that makes this problem special: our two
// candidate values are not arbitrary — kCostLethal (254) > kCostFree (0),
// and "an obstacle was detected here" should ALWAYS beat "a beam passed
// through here," no matter which thread's write happens to execute last.
// atomicMax(&obstacle_layer[idx], value) makes that priority the ONLY
// possible outcome: max(254, 0) = 254 regardless of which operand arrives
// first. The race still happens — both threads still touch the same
// address concurrently — but its OUTCOME becomes deterministic and, by
// construction, the safe one. This is the same idea 07.09's PRACTICE.md
// half-teaches through ping-pong buffering (make races impossible by
// construction); here we cannot avoid the race — two independent beams
// truly do contend for one cell — so instead we make its result provably
// safe. Compare with 09.01's uniform reads (no race possible, nothing to
// resolve) and 07.09's ping-pong (races avoided by buffer separation): this
// is the spectrum's third case — a race that is RESOLVED, not avoided.
//
// Why obstacle_layer is `int`, not `unsigned char`
// -------------------------------------------------
// CUDA has no native atomicMax for 1-byte types (only (u)int and
// (unsigned) long long, plus float/double via a CAS trick on recent
// architectures). Packing 4 cells per 32-bit word and doing byte-lane
// atomics with a compare-and-swap loop is a real production technique —
// and a good exercise (README) — but for a teaching kernel it would bury
// the mark/clear lesson under bit-packing plumbing. The honest trade made
// here: one int (4 bytes) per cell for this ONE scratch layer costs 256 KiB
// at 256x256 — trivial — in exchange for a one-line atomicMax that reads
// exactly like the story above.
// ===========================================================================
__global__ void raytrace_kernel(int robot_ix, int robot_iy,
                                const int* __restrict__ end_ix,
                                const int* __restrict__ end_iy,
                                const unsigned char* __restrict__ hit,
                                int* __restrict__ obstacle_layer)
{
    const int beam = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's beam index
    if (beam >= kNumBeams) return;                            // ragged-tail guard (360 not a multiple of 128)

    // Bresenham's integer line algorithm (Bresenham 1965) — PURE integer
    // arithmetic from here to the end of the walk. That purity is what
    // guarantees the GPU and CPU oracle visit the EXACT same sequence of
    // cells for the exact same (robot_ix,robot_iy)->(end_ix,end_iy) pair:
    // no sinf/cosf, no rounding, nothing that could differ between a
    // device and a host math library. The one float-to-int decision in this
    // whole pipeline (which cell a beam's continuous range/angle lands on)
    // already happened on the HOST, once, before this kernel ever launched
    // (see reference_cpu.cpp's simulate_lidar_scan, called from main.cu) —
    // this kernel only ever sees the resulting INTEGER endpoint.
    int x0 = robot_ix, y0 = robot_iy;
    const int x1 = end_ix[beam], y1 = end_iy[beam];
    const unsigned char beam_hit = hit[beam];

    int dx = x1 - x0; if (dx < 0) dx = -dx;
    int dy = y1 - y0; if (dy < 0) dy = -dy;
    const int sx = (x0 < x1) ? 1 : -1;
    const int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;   // classic Bresenham error accumulator (dx,-dy form)

    for (;;) {
        const bool at_end = (x0 == x1 && y0 == y1);
        // Intermediate cells are always "this beam saw clear space here";
        // the endpoint cell is lethal ONLY if the beam actually hit
        // something (beam_hit != 0) — a max-range beam's endpoint is just
        // another clear cell, the farthest one this beam can vouch for.
        const int write_value = (at_end && beam_hit) ? static_cast<int>(kCostLethal)
                                                       : static_cast<int>(kCostFree);
        // THE race-safe write — see the file-header essay above.
        atomicMax(&obstacle_layer[y0 * kGridW + x0], write_value);
        if (at_end) break;

        // Bresenham step: e2 compares the accumulated error against each
        // axis's half-step to decide whether x, y, or both advance this
        // iteration — the standard formulation, unchanged since 1965
        // because there is nothing to improve for a single-pixel-wide walk.
        const int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x0 += sx; }
        if (e2 <  dx) { err += dx; y0 += sy; }
    }
}

// ===========================================================================
// inflation_kernel — bounded-radius gather: for every cell, find the
// nearest LETHAL cell (in either static_layer or obstacle_layer) within
// kInflationRadiusCells, and write a cost that decays with distance.
//
// Thread-to-data mapping: 2-D, thread (x,y) owns cell (x,y) — the same
// tiling discipline 07.09 uses (x is the FAST axis; a warp's 32 threads
// share y and walk consecutive x, keeping every layer read/write coalesced).
//
// Why squared distance, not distance — the exactness story
// ----------------------------------------------------------
// Nav2's real inflation layer decays roughly as
// cost = (INSCRIBED-1) * exp(-scale * (distance - inscribed_radius)) — a
// continuous exponential in Euclidean DISTANCE, which needs a sqrt to get
// distance from the integer squared-distance the gather naturally computes,
// and an exp() on top. Both sqrtf/expf on the GPU and their std:: cousins
// on the CPU are accurate but NOT guaranteed bit-identical (different
// library implementations, different rounding in the last ULP) — which
// would make this layer a *tolerance* comparison, like DWA's float scoring.
// This project instead decays LINEARLY IN SQUARED DISTANCE:
//     cost = kCostInscribed * (kInflationR2 - dist2) / (kInflationR2 - kInscribedR2)
// Every term is a 32-bit integer; the divide truncates identically on any
// IEEE-conformant integer ALU (there is no "rounding mode" for integer
// division — it is exactly defined by the C++ and CUDA standards). The
// shape is still a smooth, monotonically-decreasing falloff (concave near
// the obstacle, tapering to 0 at the radius edge) — just parameterized by
// dist^2 instead of dist, which nobody driving the robot can tell apart by
// eye (THEORY.md plots both). The payoff: static+obstacle+inflation+fusion
// — the WHOLE costmap pipeline — becomes byte-exact GPU-vs-CPU, not just
// tolerance-close. Only DWA's trig-heavy scoring needs a tolerance
// (kernels.cu's dwa_score_kernel, main.cu's VERIFY DWA stage).
// ===========================================================================
static constexpr int kTile = 16;   // 16x16 = 256 threads/block, repo default (07.09's choice)

__global__ void inflation_kernel(const unsigned char* __restrict__ static_layer,
                                 const int* __restrict__ obstacle_layer,
                                 unsigned char* __restrict__ inflation_layer)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;   // fast axis
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= kGridW || y >= kGridH) return;

    // best_d2 tracks the nearest LETHAL seed's squared distance seen so
    // far; start one past the radius so "nothing found" naturally decays
    // to kCostFree below without a separate branch.
    int best_d2 = kInflationR2 + 1;

    // The (2R+1)^2 window, clipped at the grid border. R=10 -> 441 samples
    // per cell, 65536 cells -> ~28.9M distance checks total: a few
    // milliseconds of GPU time (measured in main.cu's [time] line) — the
    // brute-force cost this project's scope deliberately accepts in
    // exchange for staying self-contained (no JFA dependency, per the task
    // brief; 07.09 is where the O(log) propagation trick lives).
    for (int dy = -kInflationRadiusCells; dy <= kInflationRadiusCells; ++dy) {
        const int ny = y + dy;
        if (ny < 0 || ny >= kGridH) continue;
        for (int dx = -kInflationRadiusCells; dx <= kInflationRadiusCells; ++dx) {
            const int nx = x + dx;
            if (nx < 0 || nx >= kGridW) continue;

            const int d2 = dx * dx + dy * dy;
            if (d2 >= best_d2) continue;   // cannot improve — and enforces the CIRCULAR
                                           // window (a square (2R+1)^2 loop would otherwise
                                           // let corner cells beyond radius R contribute)
            if (d2 > kInflationR2) continue;

            const int idx = ny * kGridW + nx;
            // A seed if EITHER layer calls this neighbor lethal — the
            // fusion this kernel performs on ITS OWN inputs, ahead of the
            // separate fusion_kernel that combines all three OUTPUT layers.
            const bool lethal = (static_layer[idx] == kCostLethal) ||
                                (obstacle_layer[idx] >= static_cast<int>(kCostLethal));
            if (lethal) best_d2 = d2;
        }
    }

    unsigned char cost;
    if (best_d2 <= kInscribedR2) {
        // Within the robot's own inscribed radius of a lethal cell: the
        // robot's FOOTPRINT would already be overlapping the obstacle from
        // here, even though this cell itself never registered a hit.
        cost = kCostInscribed;
    } else if (best_d2 <= kInflationR2) {
        // Integer-exact linear-in-squared-distance ramp (see file header).
        const int span = kInflationR2 - kInscribedR2;   // > 0 by construction (100-16=84)
        cost = static_cast<unsigned char>(
            (static_cast<int>(kCostInscribed) * (kInflationR2 - best_d2)) / span);
    } else {
        cost = kCostFree;   // no lethal cell within the inflation radius
    }
    inflation_layer[y * kGridW + x] = cost;
}

// ===========================================================================
// fusion_kernel — the scaffold's SAXPY map pattern, now fusing three real
// layers instead of computing a*x+y. One thread per cell, one coalesced
// read of each of the three inputs, one coalesced write. No shared memory,
// no atomics, no interaction between threads at all — the "boring" kernel
// in this project, deliberately: not every GPU kernel needs to be clever,
// and a trivial map is exactly what per-cell layer combination is.
// ===========================================================================
__global__ void fusion_kernel(const unsigned char* __restrict__ static_layer,
                              const int* __restrict__ obstacle_layer,
                              const unsigned char* __restrict__ inflation_layer,
                              unsigned char* __restrict__ master_costmap)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kGridTotal) return;

    // obstacle_layer only ever holds kCostFree or kCostLethal (raytrace_kernel
    // never writes anything else — see its header), so the int->unsigned char
    // narrowing below is exact, never a lossy clamp.
    unsigned char m = static_layer[i];
    const unsigned char obs = static_cast<unsigned char>(obstacle_layer[i]);
    if (obs > m) m = obs;
    const unsigned char infl = inflation_layer[i];
    if (infl > m) m = infl;
    master_costmap[i] = m;
}

// ===========================================================================
// The diff-drive (unicycle) kinematic model — shared shape with 08.01's
// cart-pole: a __device__ derivative function feeding an RK4 stepper.
//
//     xdot     = v * cos(theta)
//     ydot     = v * sin(theta)
//     thetadot = w
//
// v, w are held CONSTANT for the whole rollout (the classic DWA
// assumption: "if I commit to this (v,w) and hold it for kHorizonS
// seconds, where do I end up and what do I hit along the way?" — unlike
// MPPI's per-step-varying noisy control sequence). Because thetadot is
// constant, theta(t) integrates EXACTLY under RK4 (a linear ODE has zero
// truncation error at any order >= 1); only x,y accumulate the usual
// O(dt^5)-local-error RK4 approximation as theta curves the path. A
// closed-form circular-arc solution exists for constant (v,w) — README
// Exercise 3 asks you to derive and substitute it — but numeric
// integration is used here deliberately: it is what generalizes the moment
// you add acceleration limits, wheel slip, or any dynamics term that
// breaks the closed form, and it keeps this project's rollout code
// shaped exactly like 08.01's (dynamics fn -> RK4 stepper -> per-step cost).
// ===========================================================================
__device__ __forceinline__ void unicycle_deriv(const float* pose, float v, float w, float* dpose)
{
    dpose[0] = v * cosf(pose[2]);   // xdot (m/s)
    dpose[1] = v * sinf(pose[2]);   // ydot (m/s)
    dpose[2] = w;                   // thetadot (rad/s)
}

__device__ __forceinline__ void unicycle_rk4_step(float* pose, float v, float w, float dt)
{
    float k1[3], k2[3], k3[3], k4[3], pt[3];

    unicycle_deriv(pose, v, w, k1);
#pragma unroll
    for (int i = 0; i < 3; ++i) pt[i] = fmaf(0.5f * dt, k1[i], pose[i]);
    unicycle_deriv(pt, v, w, k2);
#pragma unroll
    for (int i = 0; i < 3; ++i) pt[i] = fmaf(0.5f * dt, k2[i], pose[i]);
    unicycle_deriv(pt, v, w, k3);
#pragma unroll
    for (int i = 0; i < 3; ++i) pt[i] = fmaf(dt, k3[i], pose[i]);
    unicycle_deriv(pt, v, w, k4);

#pragma unroll
    for (int i = 0; i < 3; ++i)
        pose[i] += dt * (1.0f / 6.0f) * (k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + k4[i]);
    // Deliberately NO theta wrap here — see kernels.cuh's file header and
    // diffdrive_step_cpu's comment: rollouts integrate theta unwrapped
    // (the score's heading term uses cosf, which does not care); only the
    // real PLANT step wraps, and that happens on the host, once per tick.
}

// ===========================================================================
// dwa_score_kernel — one thread per (v,w) sample: simulate kHorizonS
// seconds under a constant candidate, sampling the master costmap along the
// way, and reduce that to one scalar score. See kernels.cuh for the (v,w)
// window layout and THEORY.md for the scoring derivation.
//
// Memory behavior: pose_x/y/theta, goal_x/y, and the window bounds are
// UNIFORM reads (kernel arguments, not device-memory loads — every thread
// gets the identical values straight from constant/parameter space, the
// same broadcast-cheap access 08.01's u_nom enjoys). master[] reads are the
// interesting case: each thread visits a DIFFERENT, DATA-DEPENDENT sequence
// of cells (wherever ITS candidate arc happens to go) — neither the
// uniform broadcast of u_nom nor the perfectly coalesced transposed-noise
// reads of MPPI's eps, and not 07.09's fixed neighbor-offset pattern
// either. It is essentially random access into a read-only grid shared by
// every thread — the fourth point on the repo's memory-access spectrum
// (09.01 constant broadcast -> 08.01 mixed uniform+coalesced -> 07.09
// divergent fixed-offset reads -> HERE: divergent DATA-DEPENDENT reads).
// L2 caching still helps (nearby rollouts visit nearby cells), but there is
// no coalescing trick available when the addresses are decided by physics,
// not by thread index — an honest limit, not a bug.
// ===========================================================================
__global__ void dwa_score_kernel(const unsigned char* __restrict__ master,
                                 float pose_x, float pose_y, float pose_theta,
                                 float goal_x, float goal_y,
                                 float v_lo, float v_hi, float w_lo, float w_hi,
                                 float mission_dist,
                                 float* __restrict__ scores)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= kNumDwaSamples) return;

    // Decode this thread's (v, w) sample from the flat index — the layout
    // contract from kernels.cuh: k = vi*kWSamples + wi.
    const int vi = k / kWSamples;
    const int wi = k % kWSamples;
    const float v = (kVSamples > 1) ? v_lo + (v_hi - v_lo) * (static_cast<float>(vi) / (kVSamples - 1)) : v_lo;
    const float w = (kWSamples > 1) ? w_lo + (w_hi - w_lo) * (static_cast<float>(wi) / (kWSamples - 1)) : w_lo;

    float pose[3] = { pose_x, pose_y, pose_theta };   // this thread's own simulated future, in registers
    float obstacle_sum = 0.0f;                        // running sum of sampled costmap cost (unitless byte scale)
    bool blocked = false;                              // true iff this arc is INADMISSIBLE

    for (int s = 0; s < kRolloutSubsteps; ++s) {
        unicycle_rk4_step(pose, v, w, kDtSub);

        // World meters -> grid cell (floor, matching the OccupancyGrid
        // convention in kernels.cuh / SYSTEM_DESIGN.md §3.6).
        const int ix = static_cast<int>(floorf(pose[0] / kResolutionM));
        const int iy = static_cast<int>(floorf(pose[1] / kResolutionM));

        unsigned char c;
        if (ix < 0 || ix >= kGridW || iy < 0 || iy >= kGridH) {
            c = kCostLethal;   // leaving the known map is treated as unsafe, not "free"
            blocked = true;
        } else {
            c = master[iy * kGridW + ix];
            if (c >= kCostLethal) blocked = true;
        }
        obstacle_sum += static_cast<float>(c);
    }

    if (blocked) {
        scores[k] = kInadmissibleScore;
        return;
    }

    // Goal-progress and heading terms, evaluated at the ARC'S END pose.
    const float dx = goal_x - pose[0];
    const float dy = goal_y - pose[1];
    const float dist_to_goal = sqrtf(dx * dx + dy * dy);
    const float bearing = atan2f(dy, dx);
    // Wrap-free heading error, the exact same trick as 08.01's
    // (1-cos(theta)) upright cost: cos(a-b) is periodic, so raw unwrapped
    // subtraction is safe and correct — no wrap_to_pi needed here at all.
    const float heading_term = 1.0f - cosf(bearing - pose[2]);

    const float score =
        kWObstacle * (obstacle_sum / static_cast<float>(kRolloutSubsteps)) / static_cast<float>(kCostLethal)
      + kWGoalDist * (dist_to_goal / mission_dist)
      + kWHeading  * heading_term
      - kWSpeed    * (v / kVMax);

    scores[k] = score;
}

// ===========================================================================
// Host launchers (declared in kernels.cuh).
// ===========================================================================

void launch_costmap_update(int robot_ix, int robot_iy,
                           const int* end_ix, const int* end_iy, const unsigned char* hit,
                           const unsigned char* d_static,
                           int* d_obstacle,
                           unsigned char* d_inflation,
                           unsigned char* d_master)
{
    if (!end_ix || !end_iy || !hit || !d_static || !d_obstacle || !d_inflation || !d_master) {
        std::fprintf(stderr, "launch_costmap_update: null argument\n");
        std::exit(EXIT_FAILURE);
    }

    // Reset the obstacle layer to kCostFree (0) before this tick's beams
    // race to write it. cudaMemset is a byte-fill primitive; since
    // kCostFree == 0, zero-filling IS the reset — there is no reason to
    // hand-write a kernel for what the driver's bandwidth-optimal memset
    // already does (contrast with jfa_clear_kernel in 07.09, which needed
    // a kernel because its sentinel is (-1,-1,-1), not zero).
    CUDA_CHECK(cudaMemset(d_obstacle, 0, static_cast<size_t>(kGridTotal) * sizeof(int)));

    // Upload this tick's scan (small: 360 ints + 360 ints + 360 bytes =
    // ~2.9 KiB). Allocated and freed HERE, matching 08.01's x0 pattern: the
    // cost is trivial next to the kernels this call launches, and keeping
    // the launcher stateless means main.cu never manages scan device memory.
    int* d_end_ix = nullptr; int* d_end_iy = nullptr; unsigned char* d_hit = nullptr;
    CUDA_CHECK(cudaMalloc(&d_end_ix, kNumBeams * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_end_iy, kNumBeams * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_hit,    kNumBeams * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemcpy(d_end_ix, end_ix, kNumBeams * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_end_iy, end_iy, kNumBeams * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hit,    hit,    kNumBeams * sizeof(unsigned char), cudaMemcpyHostToDevice));

    // Pass 1: raytrace — one thread per beam. 360 threads is a tiny launch
    // (fewer than one full block of 256 alone would cover), same "small
    // launch, still one-thread-per-item" shape as 07.09's seed-scatter kernel.
    {
        const int threads = 128;
        const int blocks = (kNumBeams + threads - 1) / threads;
        raytrace_kernel<<<blocks, threads>>>(robot_ix, robot_iy, d_end_ix, d_end_iy, d_hit, d_obstacle);
        CUDA_CHECK_LAST_ERROR("raytrace_kernel launch");
    }

    CUDA_CHECK(cudaFree(d_end_ix));
    CUDA_CHECK(cudaFree(d_end_iy));
    CUDA_CHECK(cudaFree(d_hit));

    // Pass 2: inflation — one thread per cell, 2-D tiling (see kTile above).
    {
        const dim3 block(kTile, kTile);
        const dim3 grid((kGridW + kTile - 1) / kTile, (kGridH + kTile - 1) / kTile);
        inflation_kernel<<<grid, block>>>(d_static, d_obstacle, d_inflation);
        CUDA_CHECK_LAST_ERROR("inflation_kernel launch");
    }

    // Pass 3: fusion — one thread per cell, flat 1-D (the plain map shape).
    {
        const int threads = 256;
        const int blocks = (kGridTotal + threads - 1) / threads;
        fusion_kernel<<<blocks, threads>>>(d_static, d_obstacle, d_inflation, d_master);
        CUDA_CHECK_LAST_ERROR("fusion_kernel launch");
    }
}

void launch_dwa_scores(const unsigned char* d_master,
                       float pose_x, float pose_y, float pose_theta,
                       float goal_x, float goal_y,
                       float v_lo, float v_hi, float w_lo, float w_hi,
                       float mission_dist,
                       float* d_scores)
{
    if (!d_master || !d_scores) {
        std::fprintf(stderr, "launch_dwa_scores: null argument\n");
        std::exit(EXIT_FAILURE);
    }
    const int threads = 256;                                  // repo default geometry
    const int blocks = (kNumDwaSamples + threads - 1) / threads;
    dwa_score_kernel<<<blocks, threads>>>(d_master, pose_x, pose_y, pose_theta,
                                          goal_x, goal_y, v_lo, v_hi, w_lo, w_hi,
                                          mission_dist, d_scores);
    CUDA_CHECK_LAST_ERROR("dwa_score_kernel launch");
}
