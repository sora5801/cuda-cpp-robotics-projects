// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 35.01
//                     (Magnetic microrobot swarms: Biot-Savart field
//                     computation + swarm dynamics)
//
// WHY does a GPU repository ship a CPU implementation of everything?
// ------------------------------------------------------------------
// Two load-bearing reasons (CLAUDE.md §5):
//
//   1) CORRECTNESS ORACLE. A field solver's bugs (wrong segment indexing, a
//      transposed grid axis, a sign error in the cross product) do not
//      announce themselves — they produce a field that LOOKS plausible but
//      is wrong. A dead-simple sequential version a reader can verify by
//      eye is ground truth; main.cu runs both and asserts agreement within
//      a documented tolerance.
//   2) TEACHING BASELINE. Every function here shares its numerical core
//      (biot_savart_contribution, bilinear_sample — kernels.cuh's HOSTDEV
//      helpers) with the GPU kernels in kernels.cu. Reading the two side by
//      side shows EXACTLY what parallelization changed: a "for every grid
//      cell" loop became "one thread per grid cell"; a "for every robot"
//      loop became "one thread per robot, looping over its own steps." The
//      arithmetic is identical — only who does the looping changed.
//
// This file is compiled by the HOST compiler (cl.exe); the __CUDACC__
// fence in kernels.cuh hides the __global__ kernel declarations from it,
// while the HOSTDEV helper functions (biot_savart_contribution,
// bilinear_sample, grid_to_world, world_to_grid_frac) compile here as plain
// `inline` host functions — the SAME bodies kernels.cu compiles as
// __host__ __device__. That sharing is why this file is short: it is
// mostly loop structure around helpers it does not have to re-derive.
//
// Rules for this file: plain C++17, no CUDA headers, no hand-vectorization,
// no OpenMP, no cleverness. If the reference is clever, it can be wrong,
// and then the oracle lies.
//
// Read this after: kernels.cu — then compare the two side by side.
// ===========================================================================

#include "kernels.cuh"   // pulls in <cmath> itself (sqrtf, used by the shared HOSTDEV helpers) — see its own header comment

// ---------------------------------------------------------------------------
// biot_savart_point_cpu — see kernels.cuh for the full role/contract
// commentary. Sums biot_savart_contribution over every one of the n_segs
// segments, looking up each segment's OWN coil's current from I_coil[] —
// this is the "sum over ALL segments" the catalog bullet's ratified scope
// names, spelled out with no gridding, no interpolation, nothing but the
// raw physics. main.cu's three analytic gates (on-axis, Helmholtz,
// divergence) call this directly; biot_savart_basis_cpu below is built on
// top of it.
// ---------------------------------------------------------------------------
Vec3 biot_savart_point_cpu(const CoilSegment* segs, int n_segs,
                           const float I_coil[NUM_COILS],
                           float px, float py, float pz)
{
    float Bx = 0.0f, By = 0.0f, Bz = 0.0f;   // accumulator, zero-inited (biot_savart_contribution uses +=)
    for (int s = 0; s < n_segs; ++s) {
        const CoilSegment& seg = segs[s];
        const float I = I_coil[seg.coil_id];   // 0 for coils not part of this configuration — cheap to skip, but
        if (I == 0.0f) continue;               // skipping avoids 720-3*180 wasted cross products for one-hot basis calls
        biot_savart_contribution(seg.mx, seg.my, seg.mz, seg.dlx, seg.dly, seg.dlz,
                                 I, px, py, pz, Bx, By, Bz);
    }
    return Vec3{Bx, By, Bz};
}

// ---------------------------------------------------------------------------
// biot_savart_basis_cpu — the gridded CPU oracle for ONE coil's per-unit-
// current basis map. Sequential twin of biot_savart_basis_kernel: same
// grid_to_world mapping, same per-cell physics (via biot_savart_point_cpu),
// just a "for every cell" loop instead of "one thread per cell."
//
// Complexity: O(grid_n^2 * n_segs) — for the committed scenario,
// 256*256*720 ~= 4.7e7 segment evaluations for ONE coil (main.cu's VERIFY
// stage checks exactly one representative coil, not all 4, to keep the
// CPU-side check fast — see main.cu "VERIFY_FIELD").
// ---------------------------------------------------------------------------
void biot_savart_basis_cpu(const CoilSegment* segs, int n_segs, int active_coil,
                           int grid_n, float half_m, float* Bx, float* By)
{
    float I_coil[NUM_COILS] = {0.0f, 0.0f, 0.0f, 0.0f};
    I_coil[active_coil] = 1.0f;   // unit ampere-turn: this IS what makes it a "per unit current" basis map

    for (int iy = 0; iy < grid_n; ++iy) {
        const float y = grid_to_world(iy, grid_n, half_m);
        for (int ix = 0; ix < grid_n; ++ix) {
            const float x = grid_to_world(ix, grid_n, half_m);
            const Vec3 B = biot_savart_point_cpu(segs, n_segs, I_coil, x, y, 0.0f);
            const int idx = iy * grid_n + ix;    // row-major, iy slow-varying — the layout kernels.cuh documents once
            Bx[idx] = B.x;
            By[idx] = B.y;
            // B.z is discarded here (as in the GPU basis map): the field map
            // this project steers robots with is the IN-PLANE (Bx,By) field
            // at z=0. Bz is generally nonzero off-axis but plays no role in
            // the 2D gradient-pulling force this project computes — see
            // THEORY.md "The math" for why F depends only on the in-plane
            // gradient of the FULL |B|^2 = Bx^2+By^2+Bz^2, and why dropping
            // Bz's small contribution here is an honest, documented
            // simplification (THEORY.md "Numerical considerations").
        }
    }
}

// ---------------------------------------------------------------------------
// combine_field_cpu — sequential twin of combine_field_kernel: the
// linearity-of-Maxwell elementwise combination, basisBx/basisBy laid out
// [NUM_COILS][grid_cells] (coil-major, contiguous per coil).
// ---------------------------------------------------------------------------
void combine_field_cpu(const float* basisBx, const float* basisBy, Float4 I_coil,
                       int grid_n, float* Bx, float* By)
{
    const int cells = grid_n * grid_n;
    const float I[NUM_COILS] = {I_coil.x, I_coil.y, I_coil.z, I_coil.w};
    for (int i = 0; i < cells; ++i) {
        float bx = 0.0f, by = 0.0f;
        for (int c = 0; c < NUM_COILS; ++c) {
            bx += I[c] * basisBx[c * cells + i];
            by += I[c] * basisBy[c * cells + i];
        }
        Bx[i] = bx;
        By[i] = by;
    }
}

// ---------------------------------------------------------------------------
// gradient_b2_cpu — sequential twin of gradient_b2_kernel: 4-neighbor
// central-difference stencil of |B|^2 = Bx^2+By^2 over the grid. Edge cells
// fall back to a one-sided difference (documented once here; the GPU
// kernel makes the identical choice) rather than reading out of bounds —
// this project's swarm never gets near the edge (README "Expected
// output"), so the choice of one-sided-vs-clamped-vs-mirrored boundary
// handling does not affect any reported number, only robustness.
// ---------------------------------------------------------------------------
void gradient_b2_cpu(const float* Bx, const float* By, int grid_n, float half_m,
                     float* gx, float* gy)
{
    const float h = (2.0f * half_m) / static_cast<float>(grid_n - 1);   // grid spacing (m), same along x and y (square grid)

    auto b2_at = [&](int ix, int iy) -> float {
        const int idx = iy * grid_n + ix;
        return Bx[idx] * Bx[idx] + By[idx] * By[idx];
    };

    for (int iy = 0; iy < grid_n; ++iy) {
        for (int ix = 0; ix < grid_n; ++ix) {
            // x-direction difference: central where both neighbors exist,
            // one-sided (half the usual step) at either edge.
            float dB2dx;
            if (ix > 0 && ix < grid_n - 1) {
                dB2dx = (b2_at(ix + 1, iy) - b2_at(ix - 1, iy)) / (2.0f * h);
            } else if (ix == 0) {
                dB2dx = (b2_at(ix + 1, iy) - b2_at(ix, iy)) / h;
            } else {
                dB2dx = (b2_at(ix, iy) - b2_at(ix - 1, iy)) / h;
            }

            float dB2dy;
            if (iy > 0 && iy < grid_n - 1) {
                dB2dy = (b2_at(ix, iy + 1) - b2_at(ix, iy - 1)) / (2.0f * h);
            } else if (iy == 0) {
                dB2dy = (b2_at(ix, iy + 1) - b2_at(ix, iy)) / h;
            } else {
                dB2dy = (b2_at(ix, iy) - b2_at(ix, iy - 1)) / h;
            }

            const int idx = iy * grid_n + ix;
            gx[idx] = dB2dx;
            gy[idx] = dB2dy;
        }
    }
}

// ---------------------------------------------------------------------------
// swarm_step_cpu — sequential twin of swarm_step_kernel: for every robot,
// run `steps` explicit-Euler sub-steps of the low-Reynolds-number,
// gradient-pulling dynamics (THEORY.md "The math" derives every term):
//
//     F = k_force * 0.5 * grad(|B|^2)        (superparamagnetic force)
//     v = F / gamma                           (Stokes drag; Re << 1 => no
//                                              inertia term — velocity IS
//                                              proportional to force)
//     r_{t+1} = r_t + v * dt_s                (first-order/explicit Euler)
//
// k_force and gamma are passed in as single scalars (not recomputed per
// robot): every robot in this project's swarm is IDENTICAL (same bead
// radius, same susceptibility) — a heterogeneous-swarm extension is named
// honestly in THEORY.md "Where this sits in the real world" as an open
// research direction this teaching version does not implement.
// ---------------------------------------------------------------------------
void swarm_step_cpu(const float* gx, const float* gy, int grid_n, float half_m,
                    float* rx, float* ry, int n_robots,
                    float k_force, float gamma, float dt_s, int steps)
{
    for (int k = 0; k < n_robots; ++k) {
        float x = rx[k], y = ry[k];   // load this robot's state into locals (mirrors the GPU thread's registers)
        for (int s = 0; s < steps; ++s) {
            const float dB2dx = bilinear_sample(gx, grid_n, half_m, x, y);
            const float dB2dy = bilinear_sample(gy, grid_n, half_m, x, y);
            const float Fx = k_force * 0.5f * dB2dx;   // F = k_force * grad(|B|^2/2) = k_force*0.5*grad(|B|^2)
            const float Fy = k_force * 0.5f * dB2dy;
            const float vx = Fx / gamma;                // Stokes drag: v = F/gamma, no inertia (Re << 1)
            const float vy = Fy / gamma;
            x += vx * dt_s;                              // explicit Euler
            y += vy * dt_s;
        }
        rx[k] = x;
        ry[k] = y;   // write back once per robot, after all `steps` sub-steps — matches the GPU kernel's register-resident loop
    }
}
