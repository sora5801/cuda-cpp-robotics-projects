// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 28.01
//                     Real-time FEM soft-arm model + model-based control
//
// Four jobs in this file (all declared in kernels.cuh):
//
//   1. compute_KE_hat — the shared DERIVATION both the GPU path and this CPU
//      path build their unit stiffness matrix from (THEORY.md "The math"
//      walks the same 2x2 Gauss-quadrature integral this function evaluates
//      numerically, in double precision, at startup — the exact method
//      26.01 uses, self-contained-reimplemented here per CLAUDE.md §4).
//
//   2. compute_node_mass / build_fixed_mask — small geometry precomputations
//      shared by both paths (mass and the Dirichlet mask never change once
//      built, so they are built ONCE, here, and reused every step).
//
//   3. fem_step_cpu — the ORACLE twin of kernels.cu's two-kernel step:
//      identical rotation extraction, identical local force formula, but
//      element contributions are summed into `force` in FIXED, sequential
//      element-index order (0..nelem-1) instead of via atomics — the
//      deterministic reference the GPU's atomicAdd order is compared
//      against (§5 gate; THEORY.md "How we verify correctness").
//
//   4. arm_kinetic_energy / arm_elastic_pe — diagnostic energy functions
//      used by main.cu's energy-conservation analytic gate, reusing the
//      identical per-element math fem_step_cpu computes internally.
//
// Rules for this file (CLAUDE.md §5): plain C++17, no CUDA headers, no
// hand-vectorization, no OpenMP, no cleverness. If the reference is clever,
// it can be wrong, and then the oracle lies.
//
// Read this after: kernels.cu — the two files are meant to be read side by
// side; the per-element math in fem_step_cpu is a deliberate line-by-line
// twin of elem_force_kernel (only the summation ORDER differs, on purpose).
// ===========================================================================

#include "kernels.cuh"

#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// compute_KE_hat — derive the 8x8 unit-square (h=1), unit-thickness (t=1),
// unit-modulus (E=1 Pa) plane-stress element stiffness matrix via 2x2 Gauss
// quadrature. Every element's REAL stiffness is exactly Et*KE_hat
// (kernels.cuh's "why h never appears" derivation) — this function performs
// the textbook FEA element-stiffness assembly (shape functions -> B matrix
// -> B^T D B -> Gauss-sum) that a hand-written FEA course teaches, computed
// numerically instead of hardcoded, so nothing here is a black box.
//
// Nodes (parent coordinates, CCW, matching kernels.cuh's corner_parent_sign):
// node0=(-1,-1) node1=(1,-1) node2=(1,1) node3=(-1,1). Mapping to the
// physical UNIT square [0,1]^2: x=(xi+1)/2, y=(eta+1)/2, so the Jacobian is
// the constant diagonal matrix diag(0.5,0.5), det(J)=0.25, and every
// derivative w.r.t. physical (x,y) is 2x the derivative w.r.t. parent
// (xi,eta) — dx/dxi=0.5 => dxi/dx=2 (the same "2/h" factor grad_n_physical
// uses for a general h, here specialized to h=1).
// ---------------------------------------------------------------------------
void compute_KE_hat(float nu, float KE_hat[64])
{
    static const double xi_i[4]  = { -1.0,  1.0, 1.0, -1.0 };
    static const double eta_i[4] = { -1.0, -1.0, 1.0,  1.0 };

    // Plane-stress material matrix D for E=1, given Poisson's ratio nu
    // (THEORY.md "The math" derives this 3x3 from Hooke's law under the
    // plane-stress assumption sigma_zz=0).
    const double d11 = 1.0 / (1.0 - nu * nu);
    const double d12 = nu / (1.0 - nu * nu);
    const double d33 = (1.0 - nu) / 2.0 / (1.0 - nu * nu);

    double KE[8][8] = {};   // double accumulator: this runs ONCE at startup,
                            // so there is no reason to accept float rounding
                            // in a constant every element in the domain will
                            // reuse for the whole simulation.

    // 2x2 Gauss quadrature: 2 points per axis at +/- 1/sqrt(3), weight 1
    // each — the EXACT rule for the bilinear x bilinear integrand a Q4
    // element produces (no quadrature error at all).
    const double gp = 1.0 / std::sqrt(3.0);
    const double gauss_pts[2] = { -gp, gp };

    for (double xi : gauss_pts) {
        for (double eta : gauss_pts) {
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

            double DB[3][8];
            for (int c = 0; c < 8; ++c) {
                DB[0][c] = d11*B[0][c] + d12*B[1][c];
                DB[1][c] = d12*B[0][c] + d11*B[1][c];
                DB[2][c] = d33*B[2][c];
            }
            const double wJ = 1.0 * 1.0 * 0.25;   // Gauss weights (1*1) * det(J) (0.25)
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
// compute_node_mass — lumped mass: each element's mass rho*thickness*h^2 is
// split equally (row-sum lumping) among its 4 corners; a node accumulates a
// share from each of its up to 4 incident elements. O(nelem) — run once.
// ---------------------------------------------------------------------------
void compute_node_mass(const ArmGrid& g, float rho, float thickness, float h,
                       float* node_mass)
{
    const int nnode = g.nx * g.ny;
    for (int n = 0; n < nnode; ++n) node_mass[n] = 0.0f;

    const float elem_mass = rho * thickness * h * h;   // total mass of ONE element (kg)
    const float share = elem_mass * 0.25f;             // equal quarter-share per corner

    for (int ey = 0; ey < g.nely; ++ey) {
        for (int ex = 0; ex < g.nelx; ++ex) {
            for (int a = 0; a < 4; ++a) {
                int cx, cy; corner_offset(a, &cx, &cy);
                const int n = node_id(g, ex + cx, ey + cy);
                node_mass[n] += share;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// build_fixed_mask — the cantilever boundary condition: i==0 (the base
// column) is Dirichlet-fixed in both DOFs; everything else is free.
// ---------------------------------------------------------------------------
void build_fixed_mask(const ArmGrid& g, uint8_t* fixed)
{
    for (int j = 0; j < g.ny; ++j) {
        for (int i = 0; i < g.nx; ++i) {
            const int n = node_id(g, i, j);
            const uint8_t f = (i == 0) ? 1 : 0;
            fixed[2 * n] = f;
            fixed[2 * n + 1] = f;
        }
    }
}

// ---------------------------------------------------------------------------
// cpu_ke_hat — function-local cached copy of the unit stiffness matrix,
// computed ONCE via the exact same compute_KE_hat() the GPU path calls to
// fill its __constant__ table (26.01's "shared by construction" pattern —
// main.cu is expected to call compute_KE_hat itself too and pass the result
// explicitly where a caller-supplied matrix is clearer; this cached copy
// exists so arm_elastic_pe/fem_step_cpu below never need a matrix parameter
// threaded through every call, matching the __constant__-table convenience
// the GPU path gets "for free").
// ---------------------------------------------------------------------------
static const float* cpu_ke_hat()
{
    static float ke[64];
    static bool ready = false;
    if (!ready) { compute_KE_hat(kPoissonNu, ke); ready = true; }
    return ke;
}

// ---------------------------------------------------------------------------
// cpu_elem_local_force — shared helper: given one element's 4 corner node
// ids, positions, and velocities, compute its LOCAL 8-vector force
// Et*KE_hat*(u_local + beta*v_local) and the rotation (c,s) used to bring it
// back to world. A direct, sequential twin of elem_force_kernel's steps
// 2-4 — kept as a small function here (rather than inlined into
// fem_step_cpu's loop) purely for readability; the GPU kernel inlines the
// same steps because a __global__ function cannot call a __device__ helper
// defined in a different translation unit without extra plumbing this
// project does not need.
// ---------------------------------------------------------------------------
static void cpu_elem_local_force(const ArmGrid& g, int ex, int ey,
                                 const float* x, const float* v,
                                 float Et, float h, float beta,
                                 int node[4], float f_local[8], float* out_c, float* out_s)
{
    float xa[4][2], va[4][2], Xa[4][2];
    for (int a = 0; a < 4; ++a) {
        int cx, cy; corner_offset(a, &cx, &cy);
        node[a] = node_id(g, ex + cx, ey + cy);
        xa[a][0] = x[2 * node[a]];     xa[a][1] = x[2 * node[a] + 1];
        va[a][0] = v[2 * node[a]];     va[a][1] = v[2 * node[a] + 1];
        Xa[a][0] = static_cast<float>(ex + cx) * h;
        Xa[a][1] = static_cast<float>(ey + cy) * h;
    }

    float F11 = 0.0f, F12 = 0.0f, F21 = 0.0f, F22 = 0.0f;
    for (int a = 0; a < 4; ++a) {
        float dNdX, dNdY;
        grad_n_physical(a, h, &dNdX, &dNdY);
        F11 += xa[a][0] * dNdX;   F12 += xa[a][0] * dNdY;
        F21 += xa[a][1] * dNdX;   F22 += xa[a][1] * dNdY;
    }

    const float theta = std::atan2(F21 - F12, F11 + F22);
    const float c = std::cos(theta), s = std::sin(theta);

    float combo[8];
    for (int a = 0; a < 4; ++a) {
        const float ux = xa[a][0], uy = xa[a][1];
        const float rx = c * ux + s * uy;
        const float ry = -s * ux + c * uy;
        const float u_local_x = rx - Xa[a][0];
        const float u_local_y = ry - Xa[a][1];

        const float vx = va[a][0], vy = va[a][1];
        const float rvx = c * vx + s * vy;
        const float rvy = -s * vx + c * vy;

        combo[2 * a]     = u_local_x + beta * rvx;
        combo[2 * a + 1] = u_local_y + beta * rvy;
    }

    const float* KE = cpu_ke_hat();
    for (int r = 0; r < 8; ++r) {
        float acc = 0.0f;
        for (int cc = 0; cc < 8; ++cc) acc += KE[r * 8 + cc] * combo[cc];
        f_local[r] = Et * acc;
    }
    *out_c = c; *out_s = s;
}

// ---------------------------------------------------------------------------
// fem_step_cpu — sequential twin of launch_fem_step. Element contributions
// are summed into `force` in FIXED order e=0..nelem-1 (a simple "+=", never
// an atomic — single-threaded code has no race to guard against), giving a
// deterministic reference the GPU's atomicAdd-order sum is compared against
// within a documented tolerance (THEORY.md "How we verify correctness").
// ---------------------------------------------------------------------------
void fem_step_cpu(const ArmGrid& g,
                  float* x, float* v, float* force,
                  const float* node_mass, const uint8_t* fixed,
                  float Et, float h, float alpha, float beta, float dt,
                  float T_top, float T_bottom,
                  int point_force_node, float point_force_x, float point_force_y)
{
    const int nnode = g.nx * g.ny;
    const int ndof = 2 * nnode;
    for (int d = 0; d < ndof; ++d) force[d] = 0.0f;

    // ---- assemble: sequential over elements, deterministic order ----------
    for (int ey = 0; ey < g.nely; ++ey) {
        for (int ex = 0; ex < g.nelx; ++ex) {
            int node[4]; float f_local[8], c, s;
            cpu_elem_local_force(g, ex, ey, x, v, Et, h, beta, node, f_local, &c, &s);
            for (int a = 0; a < 4; ++a) {
                const float flx = f_local[2 * a], fly = f_local[2 * a + 1];
                const float fwx = c * flx - s * fly;
                const float fwy = s * flx + c * fly;
                force[2 * node[a]]     += -fwx;   // plain += : sequential, no race
                force[2 * node[a] + 1] += -fwy;
            }
        }
    }

    // ---- integrate: sequential over nodes ----------------------------------
    for (int j = 0; j < g.ny; ++j) {
        for (int i = 0; i < g.nx; ++i) {
            const int n = node_id(g, i, j);
            if (fixed[2 * n]) {
                x[2 * n]     = 0.0f;
                x[2 * n + 1] = static_cast<float>(j) * h;
                v[2 * n]     = 0.0f;
                v[2 * n + 1] = 0.0f;
                continue;
            }
            float fx = force[2 * n];
            float fy = force[2 * n + 1];
            const float m = node_mass[n];
            fx += -alpha * m * v[2 * n];
            fy += -alpha * m * v[2 * n + 1];

            if (i >= 1 && i <= g.nelx) {
                if (j == g.nely) fx += -T_top    / static_cast<float>(kTendonAttachNodes);
                if (j == 0)      fx += -T_bottom / static_cast<float>(kTendonAttachNodes);
            }
            if (point_force_node == n) {
                fx += point_force_x;
                fy += point_force_y;
            }

            const float ax = fx / m, ay = fy / m;
            const float vx_new = v[2 * n]     + dt * ax;
            const float vy_new = v[2 * n + 1] + dt * ay;
            v[2 * n]     = vx_new;
            v[2 * n + 1] = vy_new;
            x[2 * n]     += dt * vx_new;
            x[2 * n + 1] += dt * vy_new;
        }
    }
}

// ---------------------------------------------------------------------------
// arm_kinetic_energy — 0.5 * sum_node m_node * |v_node|^2, in double (many
// small positive terms summed — double keeps the accumulation itself from
// being the noise floor of the energy-conservation gate).
// ---------------------------------------------------------------------------
double arm_kinetic_energy(const ArmGrid& g, const float* v, const float* node_mass)
{
    const int nnode = g.nx * g.ny;
    double ke = 0.0;
    for (int n = 0; n < nnode; ++n) {
        const double vx = v[2 * n], vy = v[2 * n + 1];
        ke += 0.5 * static_cast<double>(node_mass[n]) * (vx * vx + vy * vy);
    }
    return ke;
}

// ---------------------------------------------------------------------------
// arm_elastic_pe — 0.5 * sum_elem u_local_e^T (Et*KE_hat) u_local_e: the
// SAME quadratic form 26.01's compliance sensitivity uses (elem_sensitivity_
// kernel there), here evaluated on the COROTATIONAL local displacement
// (velocity term dropped: PE depends only on displacement, not velocity —
// call cpu_elem_local_force with beta=0 so combo == u_local exactly).
// ---------------------------------------------------------------------------
double arm_elastic_pe(const ArmGrid& g, const float* x, float Et, float h)
{
    static std::vector<float> zero_v;   // velocity is irrelevant to PE; reuse
    const int nnode = g.nx * g.ny;      // a zeroed buffer sized once, cached
    if (static_cast<int>(zero_v.size()) != 2 * nnode) zero_v.assign(2 * nnode, 0.0f);

    double pe = 0.0;
    for (int ey = 0; ey < g.nely; ++ey) {
        for (int ex = 0; ex < g.nelx; ++ex) {
            int node[4]; float f_local[8], c, s;
            // beta=0.0f: combo reduces to u_local exactly (no damping term);
            // f_local = Et*KE_hat*u_local, so u_local^T*f_local = 2*PE_elem.
            cpu_elem_local_force(g, ex, ey, x, zero_v.data(), Et, h, 0.0f, node, f_local, &c, &s);

            // Recompute u_local directly (cpu_elem_local_force does not
            // return it) — cheap, and keeps this function's signature simple.
            float xa[4][2], Xa[4][2];
            for (int a = 0; a < 4; ++a) {
                int cx, cy; corner_offset(a, &cx, &cy);
                const int nid = node[a];
                xa[a][0] = x[2 * nid];   xa[a][1] = x[2 * nid + 1];
                Xa[a][0] = static_cast<float>(ex + cx) * h;
                Xa[a][1] = static_cast<float>(ey + cy) * h;
            }
            double elem_pe = 0.0;
            for (int a = 0; a < 4; ++a) {
                const float ux = xa[a][0], uy = xa[a][1];
                const float rx = c * ux + s * uy;
                const float ry = -s * ux + c * uy;
                const float ulx = rx - Xa[a][0];
                const float uly = ry - Xa[a][1];
                elem_pe += static_cast<double>(ulx) * f_local[2 * a]
                         + static_cast<double>(uly) * f_local[2 * a + 1];
            }
            pe += 0.5 * elem_pe;
        }
    }
    return pe;
}
