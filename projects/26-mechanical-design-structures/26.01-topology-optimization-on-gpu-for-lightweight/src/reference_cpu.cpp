// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 26.01
//                     Topology optimization (SIMP) on GPU for lightweight
//                     links and brackets
//
// Three jobs in this file (all declared in kernels.cuh):
//
//   1. compute_KE_hat / compute_filter_weights — the shared DERIVATIONS
//      both the GPU path and this CPU path build their constants from (no
//      magic numbers anywhere in this project — THEORY.md "The math" walks
//      the same 2x2 Gauss-quadrature integral this function evaluates
//      numerically, in double precision, at startup).
//
//   2. topo_cg_solve_cpu / topo_sensitivity_cpu / topo_filter_cpu — the
//      ORACLE twins of kernels.cu's GPU stages: identical gather math,
//      identical Jacobi-preconditioned CG recursion, sequential instead of
//      parallel. main.cu's VERIFY stage runs both paths on one small
//      representative problem and requires agreement within a documented
//      tolerance (the §5 GPU-vs-CPU gate).
//
// Rules for this file (CLAUDE.md §5): plain C++17, no CUDA headers, no
// hand-vectorization, no OpenMP. If the reference is clever, it can be
// wrong, and then the oracle lies.
//
// Read this after: kernels.cu — then compare the two side by side; the
// matvec/diag/sensitivity/filter functions below are deliberately shaped to
// mirror their __global__ counterparts function-for-function.
// ===========================================================================

#include "kernels.cuh"

#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// compute_KE_hat — derive the 8x8 unit-square (side 1), unit-thickness
// (t=1), unit-modulus (E=1 Pa) plane-stress element stiffness matrix via
// 2x2 Gauss quadrature. Every physical KE is exactly E(rho)*t*KE_hat
// (kernels.cuh's "why h never appears" derivation) — this function performs
// the textbook FEA element-stiffness assembly (shape functions -> B matrix
// -> B^T D B -> Gauss-sum) that a hand-written FEA course teaches, computed
// numerically instead of hardcoded, so nothing here is a black box.
//
// Nodes (parent coordinates, CCW, matching kernels.cuh's local-node
// convention): node0=(-1,-1) node1=(1,-1) node2=(1,1) node3=(-1,1).
// Mapping to the physical UNIT square [0,1]^2: x=(xi+1)/2, y=(eta+1)/2, so
// the Jacobian is the constant diagonal matrix diag(0.5,0.5), det(J)=0.25,
// and every derivative w.r.t. physical (x,y) is 2x the derivative w.r.t.
// parent (xi,eta) — dx/dxi=0.5 => dxi/dx=2.
// ---------------------------------------------------------------------------
void compute_KE_hat(float nu, float KE_hat[64])
{
    // Bilinear shape-function corner signs (xi_i, eta_i) in the CCW order.
    static const double xi_i[4]  = { -1.0,  1.0, 1.0, -1.0 };
    static const double eta_i[4] = { -1.0, -1.0, 1.0,  1.0 };

    // Plane-stress material matrix D for E=1, given Poisson's ratio nu
    // (THEORY.md "The math" derives this 3x3 from Hooke's law under the
    // plane-stress assumption sigma_zz=0).
    const double d11 = 1.0 / (1.0 - nu * nu);
    const double d12 = nu / (1.0 - nu * nu);
    const double d33 = (1.0 - nu) / 2.0 / (1.0 - nu * nu);
    // D = [[d11,d12,0],[d12,d11,0],[0,0,d33]]

    double KE[8][8] = {};   // double accumulator: this runs ONCE at startup,
                            // so there is no reason to accept float rounding
                            // in a constant every element in the domain will
                            // reuse for the whole optimization.

    // 2x2 Gauss quadrature: 2 points per axis at +/- 1/sqrt(3), weight 1
    // each (the exact rule for the bilinear x bilinear integrand this
    // element produces — no quadrature error at all for a Q4 element).
    const double gp = 1.0 / std::sqrt(3.0);
    const double gauss_pts[2] = { -gp, gp };

    for (double xi : gauss_pts) {
        for (double eta : gauss_pts) {
            // Shape-function derivatives w.r.t. PARENT coords, then scaled
            // by dxi/dx = deta/dy = 2 (the Jacobian inverse derived above)
            // to get derivatives w.r.t. PHYSICAL coords.
            double dNdx[4], dNdy[4];
            for (int a = 0; a < 4; ++a) {
                dNdx[a] = 0.25 * xi_i[a]  * (1.0 + eta_i[a] * eta) * 2.0;
                dNdy[a] = 0.25 * eta_i[a] * (1.0 + xi_i[a]  * xi)  * 2.0;
            }

            // B: 3x8 strain-displacement matrix (rows exx, eyy, gamma_xy;
            // columns node0.x,node0.y,node1.x,...,node3.y).
            double B[3][8];
            for (int a = 0; a < 4; ++a) {
                B[0][2*a]   = dNdx[a]; B[0][2*a+1] = 0.0;
                B[1][2*a]   = 0.0;     B[1][2*a+1] = dNdy[a];
                B[2][2*a]   = dNdy[a]; B[2][2*a+1] = dNdx[a];
            }

            // DB = D * B (3x8), then KE += w*w * det(J) * B^T * (D*B).
            // det(J) = 0.25 (constant over the element — computed above);
            // Gauss weights are 1*1 for this 2-point rule.
            double DB[3][8];
            for (int c = 0; c < 8; ++c) {
                DB[0][c] = d11*B[0][c] + d12*B[1][c];
                DB[1][c] = d12*B[0][c] + d11*B[1][c];
                DB[2][c] = d33*B[2][c];
            }
            const double wJ = 1.0 * 1.0 * 0.25;
            for (int r = 0; r < 8; ++r)
                for (int c = 0; c < 8; ++c)
                    KE[r][c] += wJ * (B[0][r]*DB[0][c] + B[1][r]*DB[1][c] + B[2][r]*DB[2][c]);
        }
    }

    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 8; ++c)
            KE_hat[r*8 + c] = static_cast<float>(KE[r][c]);   // narrow to float ONCE, at the end
}

// ---------------------------------------------------------------------------
// compute_filter_weights — derive the (2r+1)x(2r+1) distance-weighted
// filter table w(di,dj) = max(0, rmin - sqrt(di^2+dj^2)) (Sigmund's classic
// linear kernel — THEORY.md "The algorithm" explains why LINEAR-in-distance,
// not Gaussian or uniform: it vanishes smoothly at rmin, giving a filter
// whose support is EXACTLY the elements within rmin, no truncation error).
// ---------------------------------------------------------------------------
void compute_filter_weights(float rmin, int r, float* weights)
{
    const int side = 2 * r + 1;
    for (int dj = -r; dj <= r; ++dj) {
        for (int di = -r; di <= r; ++di) {
            const double dist = std::sqrt(static_cast<double>(di) * di + static_cast<double>(dj) * dj);
            const double w = rmin - dist;
            weights[(dj + r) * side + (di + r)] = static_cast<float>(w > 0.0 ? w : 0.0);
        }
    }
}

// ---------------------------------------------------------------------------
// cpu_ke_hat / cpu_filter_w — function-local cached copies, computed ONCE
// (via the exact same compute_* functions the GPU path calls to fill its
// __constant__ tables) so every CPU function below reads identical
// constants to the kernels — "shared by construction", not by convention.
// ---------------------------------------------------------------------------
static const float* cpu_ke_hat()
{
    static float ke[64];
    static bool ready = false;
    if (!ready) { compute_KE_hat(kPoissonNu, ke); ready = true; }
    return ke;
}
static const float* cpu_filter_w()
{
    static float w[(2 * kFilterR + 1) * (2 * kFilterR + 1)];
    static bool ready = false;
    if (!ready) { compute_filter_weights(kFilterRMin, kFilterR, w); ready = true; }
    return w;
}

// ---------------------------------------------------------------------------
// cpu_quadrant_elem — sequential twin of kernels.cu's quadrant_elem(): same
// CCW local-index table, same up-to-4-incident-elements enumeration.
// ---------------------------------------------------------------------------
static void cpu_quadrant_elem(int i, int j, int q, int* ex, int* ey, int* L)
{
    switch (q) {
        case 0: *ex = i - 1; *ey = j - 1; *L = 2; break;
        case 1: *ex = i;     *ey = j - 1; *L = 3; break;
        case 2: *ex = i - 1; *ey = j;     *L = 1; break;
        default:*ex = i;     *ey = j;     *L = 0; break;
    }
}

// cpu_matvec — sequential twin of matvec_gather_kernel: same node-gather
// loop, run for EVERY node in a plain nested for (the GPU version runs it
// once per thread, in parallel; the math is identical line-for-line).
static void cpu_matvec(const TopoGrid& g, const float* rho, const float* x,
                       const uint8_t* fixed, float E0, float Emin, float* y)
{
    const float* KE = cpu_ke_hat();
    for (int j = 0; j < g.ny; ++j) {
        for (int i = 0; i < g.nx; ++i) {
            float y0 = 0.0f, y1 = 0.0f;
            for (int q = 0; q < 4; ++q) {
                int ex, ey, L;
                cpu_quadrant_elem(i, j, q, &ex, &ey, &L);
                if (ex < 0 || ex >= g.nelx || ey < 0 || ey >= g.nely) continue;
                const float Ee = E_of_rho(rho[elem_id(g, ex, ey)], E0, Emin);
                const int n0 = node_id(g, ex, ey), n1 = node_id(g, ex + 1, ey);
                const int n2 = node_id(g, ex + 1, ey + 1), n3 = node_id(g, ex, ey + 1);
                const float xe[8] = { x[2*n0], x[2*n0+1], x[2*n1], x[2*n1+1],
                                      x[2*n2], x[2*n2+1], x[2*n3], x[2*n3+1] };
                const float* row0 = &KE[(2*L)   * 8];
                const float* row1 = &KE[(2*L+1) * 8];
                float s0 = 0.0f, s1 = 0.0f;
                for (int k = 0; k < 8; ++k) { s0 += row0[k]*xe[k]; s1 += row1[k]*xe[k]; }
                y0 += Ee * s0; y1 += Ee * s1;
            }
            const int n = node_id(g, i, j);
            y[2*n]   = fixed[2*n]   ? 0.0f : y0;
            y[2*n+1] = fixed[2*n+1] ? 0.0f : y1;
        }
    }
}

// cpu_diag — sequential twin of diag_gather_kernel.
static void cpu_diag(const TopoGrid& g, const float* rho, const uint8_t* fixed,
                     float E0, float Emin, float* diag)
{
    const float* KE = cpu_ke_hat();
    for (int j = 0; j < g.ny; ++j) {
        for (int i = 0; i < g.nx; ++i) {
            float d0 = 0.0f, d1 = 0.0f;
            for (int q = 0; q < 4; ++q) {
                int ex, ey, L;
                cpu_quadrant_elem(i, j, q, &ex, &ey, &L);
                if (ex < 0 || ex >= g.nelx || ey < 0 || ey >= g.nely) continue;
                const float Ee = E_of_rho(rho[elem_id(g, ex, ey)], E0, Emin);
                d0 += Ee * KE[(2*L)*8 + (2*L)];
                d1 += Ee * KE[(2*L+1)*8 + (2*L+1)];
            }
            const int n = node_id(g, i, j);
            diag[2*n]   = fixed[2*n]   ? 1.0f : std::fmax(d0, 1e-20f);
            diag[2*n+1] = fixed[2*n+1] ? 1.0f : std::fmax(d1, 1e-20f);
        }
    }
}

static double cpu_dot(int n, const float* a, const float* b)
{
    double acc = 0.0;               // double accumulator, matching kernels.cu's
    for (int i = 0; i < n; ++i) acc += static_cast<double>(a[i]) * b[i];  // block-partial-sum-in-double discipline
    return acc;
}

// ---------------------------------------------------------------------------
// topo_cg_solve_cpu — sequential Jacobi-PCG twin of launch_topo_cg_solve.
// Same recursion, same stopping rule; every vector update below is the
// direct scalar transcription of the GPU's vec_combine/vec_divide kernels.
// ---------------------------------------------------------------------------
void topo_cg_solve_cpu(const TopoGrid& g, const float* rho, const float* F,
                       const uint8_t* fixed, float* U, float E0, float Emin,
                       int max_iters, float rel_tol, int* out_iters, float* out_rel_resid)
{
    const int ndof = 2 * g.nx * g.ny;
    std::vector<float> diag(ndof), r(ndof), z(ndof), p(ndof), Kp(ndof);

    cpu_diag(g, rho, fixed, E0, Emin, diag.data());

    const float normF = static_cast<float>(std::sqrt(std::fmax(0.0, cpu_dot(ndof, F, F))));
    const float denom = (normF > 1e-30f) ? normF : 1.0f;

    cpu_matvec(g, rho, U, fixed, E0, Emin, Kp.data());
    for (int i = 0; i < ndof; ++i) r[i] = F[i] - Kp[i];
    for (int i = 0; i < ndof; ++i) z[i] = r[i] / diag[i];
    for (int i = 0; i < ndof; ++i) p[i] = z[i];

    float rz_old = static_cast<float>(cpu_dot(ndof, r.data(), z.data()));
    int iters_run = max_iters;
    float rel_resid = static_cast<float>(std::sqrt(std::fmax(0.0, cpu_dot(ndof, r.data(), r.data())))) / denom;

    for (int it = 0; it < max_iters; ++it) {
        if (rel_resid < rel_tol) { iters_run = it; break; }

        cpu_matvec(g, rho, p.data(), fixed, E0, Emin, Kp.data());
        const float pKp = static_cast<float>(cpu_dot(ndof, p.data(), Kp.data()));
        if (std::fabs(pKp) < 1e-30f) { iters_run = it; break; }
        const float alpha = rz_old / pKp;

        for (int i = 0; i < ndof; ++i) U[i] += alpha * p[i];
        for (int i = 0; i < ndof; ++i) r[i] -= alpha * Kp[i];

        rel_resid = static_cast<float>(std::sqrt(std::fmax(0.0, cpu_dot(ndof, r.data(), r.data())))) / denom;
        if (rel_resid < rel_tol) { iters_run = it + 1; break; }

        for (int i = 0; i < ndof; ++i) z[i] = r[i] / diag[i];
        const float rz_new = static_cast<float>(cpu_dot(ndof, r.data(), z.data()));
        const float beta = rz_new / rz_old;
        for (int i = 0; i < ndof; ++i) p[i] = z[i] + beta * p[i];
        rz_old = rz_new;
    }

    if (out_iters) *out_iters = iters_run;
    if (out_rel_resid) *out_rel_resid = rel_resid;
}

// topo_sensitivity_cpu — sequential twin of elem_sensitivity_kernel.
void topo_sensitivity_cpu(const TopoGrid& g, const float* rho, const float* U,
                          float E0, float Emin, float* ce, float* dc_raw)
{
    const float* KE = cpu_ke_hat();
    for (int ey = 0; ey < g.nely; ++ey) {
        for (int ex = 0; ex < g.nelx; ++ex) {
            const int n0 = node_id(g, ex, ey), n1 = node_id(g, ex + 1, ey);
            const int n2 = node_id(g, ex + 1, ey + 1), n3 = node_id(g, ex, ey + 1);
            const float ue[8] = { U[2*n0], U[2*n0+1], U[2*n1], U[2*n1+1],
                                  U[2*n2], U[2*n2+1], U[2*n3], U[2*n3+1] };
            float q = 0.0f;
            for (int r = 0; r < 8; ++r) {
                float s = 0.0f;
                for (int c = 0; c < 8; ++c) s += KE[r*8 + c] * ue[c];
                q += ue[r] * s;
            }
            const int e = elem_id(g, ex, ey);
            const float rho_e = rho[e];
            ce[e]     = E_of_rho(rho_e, E0, Emin) * q;
            dc_raw[e] = -dE_drho(rho_e, E0, Emin) * q;
        }
    }
}

// topo_filter_cpu — sequential twin of density_filter_kernel.
void topo_filter_cpu(const TopoGrid& g, const float* rho, const float* dc_raw, float* dc_filt)
{
    const float* W = cpu_filter_w();
    const int side = 2 * kFilterR + 1;
    for (int ey = 0; ey < g.nely; ++ey) {
        for (int ex = 0; ex < g.nelx; ++ex) {
            float num = 0.0f, wsum = 0.0f;
            for (int dj = -kFilterR; dj <= kFilterR; ++dj) {
                const int fy = ey + dj;
                if (fy < 0 || fy >= g.nely) continue;
                for (int di = -kFilterR; di <= kFilterR; ++di) {
                    const int fx = ex + di;
                    if (fx < 0 || fx >= g.nelx) continue;
                    const float w = W[(dj + kFilterR) * side + (di + kFilterR)];
                    if (w <= 0.0f) continue;
                    const int f = elem_id(g, fx, fy);
                    num += w * rho[f] * dc_raw[f];
                    wsum += w;
                }
            }
            const int e = elem_id(g, ex, ey);
            const float rho_e_floored = std::fmax(rho[e], 1.0e-3f);
            dc_filt[e] = (wsum > 0.0f) ? (num / (rho_e_floored * wsum)) : dc_raw[e];
        }
    }
}
