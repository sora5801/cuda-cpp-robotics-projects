// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 23.01
//                     GPU costmaps: inflation, raytrace clearing, multi-layer
//                     fusion + a DWA local-planner consumer
//
// Three jobs in this project (all declared in kernels.cuh):
//
//   1. costmap_update_cpu — the ORACLE twin of the whole GPU costmap
//      pipeline (raytrace + inflation + fusion in one call, matching
//      launch_costmap_update's contract exactly). Every layer here is
//      PURE INTEGER arithmetic (kernels.cu explains why), so main.cu's
//      VERIFY stage checks this against the GPU master costmap for BYTE
//      EQUALITY, not a tolerance — the strongest verification this repo's
//      conventions allow, and this project earns it honestly.
//
//   2. dwa_scores_cpu — the ORACLE twin of dwa_score_kernel: same RK4
//      unicycle integration, same costmap sampling, same scoring formula,
//      sequential over all 4096 candidates. Trig-heavy (cosf/atan2f), so
//      main.cu compares this against the GPU scores within a documented
//      relative tolerance — the same honest reason 08.01's rollout costs
//      need one.
//
//   3. diffdrive_step_cpu — THE PLANT. The closed-loop demo needs a "real"
//      differential-drive robot for the DWA-chosen (v,w) to be applied to;
//      simulating it on the host with the same RK4 keeps the demo
//      self-contained and gives a zero-model-mismatch plant — the same
//      deliberate idealization 08.01 makes, stated honestly in this
//      project's README §Limitations too.
//
// The mark/clear, inflation-decay, and RK4/scoring math below are
// line-by-line twins of the __device__ versions in kernels.cu — deliberate,
// documented duplication (diff the files side by side; only the
// atomic-vs-sequential mark/clear strategy and float-function spellings
// differ, and both are called out where they matter).
//
// Read this after: kernels.cuh.  Read this beside: kernels.cu.
// ===========================================================================

#include "kernels.cuh"   // shared constants, layouts, signatures

#include <algorithm>      // std::max — the sequential twin of atomicMax
#include <cmath>          // std::sin/cos/atan2/sqrt/floor (double... no, float overloads used)

// ---------------------------------------------------------------------------
// raytrace_beam_cpu — sequential oracle twin of one thread's work inside
// raytrace_kernel: the same Bresenham walk, the same mark/clear values, but
// combined with std::max instead of atomicMax.
//
// WHY std::max and not a plain assignment, even though this loop is
// single-threaded and therefore race-free by construction: the ORACLE must
// compute the same REDUCTION the GPU computes (per cell: the max over every
// beam that ever wrote there), not merely "some value that happened to
// survive a particular sequential visitation order." A plain assignment
// here would make the CPU path's answer depend on which beam this function
// happens to process last for a given cell — coincidentally matching the
// GPU's answer on some inputs and silently diverging on others. std::max
// makes the CPU path compute the identical MATHEMATICAL result
// (max over all beams touching this cell) regardless of visitation order,
// which is what makes the byte-exact GPU-vs-CPU comparison in main.cu a
// meaningful proof of correctness rather than a coincidence.
// ---------------------------------------------------------------------------
static void raytrace_beam_cpu(int robot_ix, int robot_iy,
                              int end_x, int end_y, unsigned char beam_hit,
                              int* obstacle_layer)
{
    int x0 = robot_ix, y0 = robot_iy;
    const int x1 = end_x, y1 = end_y;

    int dx = x1 - x0; if (dx < 0) dx = -dx;
    int dy = y1 - y0; if (dy < 0) dy = -dy;
    const int sx = (x0 < x1) ? 1 : -1;
    const int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;

    for (;;) {
        const bool at_end = (x0 == x1 && y0 == y1);
        const int write_value = (at_end && beam_hit) ? static_cast<int>(kCostLethal)
                                                       : static_cast<int>(kCostFree);
        int& cell = obstacle_layer[y0 * kGridW + x0];
        cell = std::max(cell, write_value);   // the sequential twin of atomicMax
        if (at_end) break;

        const int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x0 += sx; }
        if (e2 <  dx) { err += dx; y0 += sy; }
    }
}

// ---------------------------------------------------------------------------
// inflation_cell_cpu — sequential oracle twin of inflation_kernel for ONE
// cell (called from the double loop in costmap_update_cpu below). Same
// bounded gather, same integer squared-distance decay formula — see
// kernels.cu's file header for why this is exact, not tolerance-compared.
// ---------------------------------------------------------------------------
static unsigned char inflation_cell_cpu(int x, int y,
                                        const unsigned char* static_layer,
                                        const int* obstacle_layer)
{
    int best_d2 = kInflationR2 + 1;

    for (int dy = -kInflationRadiusCells; dy <= kInflationRadiusCells; ++dy) {
        const int ny = y + dy;
        if (ny < 0 || ny >= kGridH) continue;
        for (int dx = -kInflationRadiusCells; dx <= kInflationRadiusCells; ++dx) {
            const int nx = x + dx;
            if (nx < 0 || nx >= kGridW) continue;

            const int d2 = dx * dx + dy * dy;
            if (d2 >= best_d2) continue;
            if (d2 > kInflationR2) continue;

            const int idx = ny * kGridW + nx;
            const bool lethal = (static_layer[idx] == kCostLethal) ||
                                (obstacle_layer[idx] >= static_cast<int>(kCostLethal));
            if (lethal) best_d2 = d2;
        }
    }

    if (best_d2 <= kInscribedR2) return kCostInscribed;
    if (best_d2 <= kInflationR2) {
        const int span = kInflationR2 - kInscribedR2;
        return static_cast<unsigned char>(
            (static_cast<int>(kCostInscribed) * (kInflationR2 - best_d2)) / span);
    }
    return kCostFree;
}

// ---------------------------------------------------------------------------
// costmap_update_cpu — the whole pipeline, sequentially: reset, raytrace
// every beam, inflate every cell, fuse every cell. Mirrors
// launch_costmap_update's three-kernel sequence one-for-one.
// ---------------------------------------------------------------------------
void costmap_update_cpu(int robot_ix, int robot_iy,
                        const int* end_ix, const int* end_iy, const unsigned char* hit,
                        const unsigned char* static_layer,
                        int* obstacle_layer,
                        unsigned char* inflation_layer,
                        unsigned char* master_costmap)
{
    // Pass 0: reset (the sequential twin of the GPU's cudaMemset).
    for (int i = 0; i < kGridTotal; ++i) obstacle_layer[i] = static_cast<int>(kCostFree);

    // Pass 1: raytrace every beam, in order — order does not matter for
    // the RESULT (std::max makes the reduction order-independent, same as
    // atomicMax does on the GPU), only for how many times any one cell
    // gets touched along the way.
    for (int b = 0; b < kNumBeams; ++b)
        raytrace_beam_cpu(robot_ix, robot_iy, end_ix[b], end_iy[b], hit[b], obstacle_layer);

    // Pass 2: inflate every cell.
    for (int y = 0; y < kGridH; ++y)
        for (int x = 0; x < kGridW; ++x)
            inflation_layer[y * kGridW + x] = inflation_cell_cpu(x, y, static_layer, obstacle_layer);

    // Pass 3: fuse every cell (per-cell max of the three layers).
    for (int i = 0; i < kGridTotal; ++i) {
        unsigned char m = static_layer[i];
        const unsigned char obs = static_cast<unsigned char>(obstacle_layer[i]);
        if (obs > m) m = obs;
        const unsigned char infl = inflation_layer[i];
        if (infl > m) m = infl;
        master_costmap[i] = m;
    }
}

// ---------------------------------------------------------------------------
// Host twins of the device unicycle model (see kernels.cu for the physics
// commentary — not repeated here; the MATH must stay identical). Free
// functions, not exported — only diffdrive_step_cpu and dwa_scores_cpu
// need them, both in this file.
// ---------------------------------------------------------------------------
static void unicycle_deriv_cpu(const float* pose, float v, float w, float* dpose)
{
    dpose[0] = v * std::cos(pose[2]);
    dpose[1] = v * std::sin(pose[2]);
    dpose[2] = w;
}

static void unicycle_rk4_step_cpu(float* pose, float v, float w, float dt)
{
    float k1[3], k2[3], k3[3], k4[3], pt[3];

    unicycle_deriv_cpu(pose, v, w, k1);
    for (int i = 0; i < 3; ++i) pt[i] = pose[i] + 0.5f * dt * k1[i];
    unicycle_deriv_cpu(pt, v, w, k2);
    for (int i = 0; i < 3; ++i) pt[i] = pose[i] + 0.5f * dt * k2[i];
    unicycle_deriv_cpu(pt, v, w, k3);
    for (int i = 0; i < 3; ++i) pt[i] = pose[i] + dt * k3[i];
    unicycle_deriv_cpu(pt, v, w, k4);

    for (int i = 0; i < 3; ++i)
        pose[i] += dt * (1.0f / 6.0f) * (k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + k4[i]);
}

// ---------------------------------------------------------------------------
// dwa_scores_cpu — sequential oracle twin of dwa_score_kernel, one sample
// after another (the GPU gives each its own thread).
// ---------------------------------------------------------------------------
void dwa_scores_cpu(const unsigned char* master,
                    float pose_x, float pose_y, float pose_theta,
                    float goal_x, float goal_y,
                    float v_lo, float v_hi, float w_lo, float w_hi,
                    float mission_dist,
                    float* scores)
{
    for (int k = 0; k < kNumDwaSamples; ++k) {
        const int vi = k / kWSamples;
        const int wi = k % kWSamples;
        const float v = (kVSamples > 1) ? v_lo + (v_hi - v_lo) * (static_cast<float>(vi) / (kVSamples - 1)) : v_lo;
        const float w = (kWSamples > 1) ? w_lo + (w_hi - w_lo) * (static_cast<float>(wi) / (kWSamples - 1)) : w_lo;

        float pose[3] = { pose_x, pose_y, pose_theta };
        float obstacle_sum = 0.0f;
        bool blocked = false;

        for (int s = 0; s < kRolloutSubsteps; ++s) {
            unicycle_rk4_step_cpu(pose, v, w, kDtSub);

            const int ix = static_cast<int>(std::floor(pose[0] / kResolutionM));
            const int iy = static_cast<int>(std::floor(pose[1] / kResolutionM));

            unsigned char c;
            if (ix < 0 || ix >= kGridW || iy < 0 || iy >= kGridH) {
                c = kCostLethal;
                blocked = true;
            } else {
                c = master[iy * kGridW + ix];
                if (c >= kCostLethal) blocked = true;
            }
            obstacle_sum += static_cast<float>(c);
        }

        if (blocked) {
            scores[k] = kInadmissibleScore;
            continue;
        }

        const float dx = goal_x - pose[0];
        const float dy = goal_y - pose[1];
        const float dist_to_goal = std::sqrt(dx * dx + dy * dy);
        const float bearing = std::atan2(dy, dx);
        const float heading_term = 1.0f - std::cos(bearing - pose[2]);

        scores[k] =
            kWObstacle * (obstacle_sum / static_cast<float>(kRolloutSubsteps)) / static_cast<float>(kCostLethal)
          + kWGoalDist * (dist_to_goal / mission_dist)
          + kWHeading  * heading_term
          - kWSpeed    * (v / kVMax);
    }
}

// ---------------------------------------------------------------------------
// diffdrive_step_cpu — THE PLANT: one dt of "reality" under constant (v,w).
//
// This is the project's SINGLE DEFINED WRAP POINT (CLAUDE.md §12,
// SYSTEM_DESIGN.md §3.7): the plant's pose keeps theta in (-pi, pi] so logs
// and the goal-reached check read naturally; rollouts (both GPU and CPU,
// above) integrate theta unwrapped — their heading term uses cosf/cos,
// which does not care — mirroring 08.01's cartpole_step_cpu exactly.
// ---------------------------------------------------------------------------
void diffdrive_step_cpu(float* pose, float v, float w, float dt)
{
    unicycle_rk4_step_cpu(pose, v, w, dt);

    // Wrap theta into (-pi, pi]. The loop form is transparent and never
    // iterates more than once in practice here (|w|*dt << 2*pi at this dt).
    const float pi = 3.14159265358979323846f;
    while (pose[2] >  pi) pose[2] -= 2.0f * pi;
    while (pose[2] <= -pi) pose[2] += 2.0f * pi;
}
