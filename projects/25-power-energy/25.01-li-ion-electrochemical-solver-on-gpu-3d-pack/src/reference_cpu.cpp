// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 25.01
//                     Li-ion electrochemical (SPM) + 3D pack thermal solve
//
// The oracle twin of BOTH kernels in kernels.cu: same FV update expressions,
// same boundary treatment, same FP32 arithmetic, just sequential loops
// instead of one-thread-per-(particle,shell) / one-thread-per-(design,voxel).
// main.cu runs both against the GPU on a small representative slice (a
// handful of designs/steps — running the FULL 12-design/20-minute mission
// twice would defeat the point of "GPU vs CPU speed" and cost minutes on a
// single core) and requires agreement within kernels.cuh's documented
// tolerances — the CLAUDE.md §5 GPU-vs-CPU gate for this project.
//
// Deliberate, DOCUMENTED duplication (CLAUDE.md §4): diff this file against
// kernels.cu's two kernels and the arithmetic matches line for line; only
// the indexing (loops here, thread ID there) differs.
//
// Read this after: kernels.cu — then compare the two side by side.
// ===========================================================================

#include "kernels.cuh"   // shared structs, constants, and these functions' prototypes

#include <cmath>         // M_PI is not standard; we define our own constant below

static constexpr float kPiF = 3.14159265358979323846f;

// ---------------------------------------------------------------------------
// electrochem_fv_cpu — sequential twin of electrochem_fv_kernel. Loops over
// every particle p in [0, B*kNCells*2) and every shell s in [0, kNShells) —
// see kernels.cu's kernel comment for the full FV derivation; only the
// loop structure differs here.
// ---------------------------------------------------------------------------
void electrochem_fv_cpu(int B, ElectrodeGeom geomA, ElectrodeGeom geomC,
                        float j_a, float j_c, float dt_e,
                        const float* D, const float* c_in, float* c_out)
{
    const int nParticles = B * kNCells * 2;
    for (int p = 0; p < nParticles; ++p) {
        const int e = p % 2;                                  // 0 = anode, 1 = cathode
        const ElectrodeGeom geom = (e == 0) ? geomA : geomC;
        const float j = (e == 0) ? j_a : j_c;
        const float Dp = D[p];
        const float dr = geom.R_p / static_cast<float>(kNShells);

        for (int s = 0; s < kNShells; ++s) {
            const int idx = p * kNShells + s;
            const float r_in  = static_cast<float>(s) * dr;
            const float r_out = static_cast<float>(s + 1) * dr;
            const float V_s  = (4.0f / 3.0f) * kPiF * (r_out * r_out * r_out - r_in * r_in * r_in);
            const float A_in  = 4.0f * kPiF * r_in  * r_in;
            const float A_out = 4.0f * kPiF * r_out * r_out;

            const float c0 = c_in[idx];

            float F_in = 0.0f;
            if (s > 0) F_in = -Dp * (c0 - c_in[idx - 1]) / dr;

            float F_out;
            if (s == kNShells - 1) F_out = j;
            else                    F_out = -Dp * (c_in[idx + 1] - c0) / dr;

            const float dcdt = (A_in * F_in - A_out * F_out) / V_s;
            c_out[idx] = c0 + dt_e * dcdt;
        }
    }
}

// ---------------------------------------------------------------------------
// thermal_step_cpu — sequential twin of thermal_step_kernel. Loops over
// every design b, then every voxel (k, j, i) — see kernels.cu's kernel
// comment for the full boundary-condition derivation.
// ---------------------------------------------------------------------------
void thermal_step_cpu(int B, const PackThermalParams& p, float dt_thermal,
                      const DesignPoint* designs,
                      const float* q_vol, const float* T_in, float* T_out)
{
    const int planeStride = kTNX * kTNY;
    for (int b = 0; b < B; ++b) {
        const DesignPoint dp = designs[b];
        for (int k = 0; k < kTNZ; ++k) {
            for (int j = 0; j < kTNY; ++j) {
                for (int i = 0; i < kTNX; ++i) {
                    const int idx = ((b * kTNZ + k) * kTNY + j) * kTNX + i;
                    const float T0 = T_in[idx];
                    float flux_div = 0.0f;

                    // x axis
                    if (i > 0 && i < kTNX - 1) {
                        flux_div += p.kx * (T_in[idx - 1] - 2.0f * T0 + T_in[idx + 1]) / (p.dx * p.dx);
                    } else {
                        const float Tn = (i == 0) ? T_in[idx + 1] : T_in[idx - 1];
                        flux_div += p.kx * (Tn - T0) / (p.dx * p.dx);
                        if (i == 0 && dp.face == kCoolSideX)
                            flux_div += dp.h * (p.T_coolant - T0) / p.dx;
                    }

                    // y axis (always adiabatic)
                    if (j > 0 && j < kTNY - 1) {
                        flux_div += p.ky * (T_in[idx - kTNX] - 2.0f * T0 + T_in[idx + kTNX]) / (p.dy * p.dy);
                    } else {
                        const float Tn = (j == 0) ? T_in[idx + kTNX] : T_in[idx - kTNX];
                        flux_div += p.ky * (Tn - T0) / (p.dy * p.dy);
                    }

                    // z axis
                    if (k > 0 && k < kTNZ - 1) {
                        flux_div += p.kz * (T_in[idx - planeStride] - 2.0f * T0 + T_in[idx + planeStride]) / (p.dz * p.dz);
                    } else {
                        const float Tn = (k == 0) ? T_in[idx + planeStride] : T_in[idx - planeStride];
                        flux_div += p.kz * (Tn - T0) / (p.dz * p.dz);
                        if (k == 0 && dp.face == kCoolBottomZ)
                            flux_div += dp.h * (p.T_coolant - T0) / p.dz;
                    }

                    const float dTdt = (flux_div + q_vol[idx]) / p.rho_cp;
                    T_out[idx] = T0 + dt_thermal * dTdt;
                }
            }
        }
    }
}
