// ===========================================================================
// kernels.cuh — interface for project 27.04
//               Composite layup optimization + Tsai-Wu failure envelope sweeps
//               (teaching core: classical laminate theory (CLT) + Tsai-Wu
//               first-ply-failure, swept over 256 symmetric 8-ply layups and
//               a 128x128 (Nx,Ny) failure-envelope grid)
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (the sweep/envelope orchestrator + analytic
// gate checker), kernels.cu (the two GPU sweep kernels), and reference_cpu.cpp
// (the CPU oracle, computed with the SAME physics functions). Everything all
// three must agree on — the lamina/laminate records and their UNITS, the
// stack-encoding scheme, the load-case format, and the CLT/Tsai-Wu math
// itself — is defined HERE, once (CLAUDE.md §12).
//
// The physics in six lines (THEORY.md derives every step properly):
//   1. A ply is an ORTHOTROPIC sheet: stiff along its fibers (E1), soft
//      across them (E2), with independent shear stiffness G12. Rotating a
//      ply by angle theta transforms its stiffness Q into laminate axes
//      Qbar(theta) — off-axis plies "leak" normal load into shear response.
//   2. Stacking N plies and summing thickness-weighted Qbar gives the
//      laminate's EXTENSIONAL stiffness matrix A. For a SYMMETRIC stack
//      (mirrored about the midplane — every layup this project builds), the
//      bending-extension coupling matrix B is EXACTLY zero.
//   3. B=0 means an in-plane load N=(Nx,Ny,Nxy) produces a UNIFORM midplane
//      strain eps0 with ZERO curvature — solve A*eps0=N (a 3x3 linear
//      system, Cramer's rule, in registers) and every ply sees the SAME eps0.
//   4. Each ply transforms eps0 into ITS OWN material axes and computes its
//      own stress (sigma1, sigma2, tau12) via the UNROTATED ply stiffness Q.
//   5. The Tsai-Wu quadratic failure criterion, evaluated per ply, is
//      quadratic in a LOAD-SCALING FACTOR lambda (because stress is linear
//      in load): solve it in closed form for the ply's own failure lambda;
//      the laminate's first-ply-failure factor is the MINIMUM over its plies.
//   6. The GPU content is the SWEEP: one thread per (layup, load case) pair
//      scores a candidate stacking sequence against a load-case set (the
//      ranking sweep), and one thread per (Nx,Ny) grid point maps the whole
//      failure envelope as a field (the envelope sweep) — both are
//      embarrassingly parallel "thread = one independent small linear-
//      algebra + quadratic-root problem" maps, the same shape as 33.01's
//      batched small-matrix pattern applied to a materials design search.
//
// WHY STACKING SEQUENCE DOES NOT MATTER HERE (an honest, load-bearing fact):
// A_ij = sum_k Qbar_ij(ply k) * t_ply depends only on the MULTISET of ply
// angles, never their ORDER — two layups that use the same four angles in a
// different sequence have IDENTICAL A matrices and therefore IDENTICAL
// first-ply-failure factors under any in-plane (membrane) load. Stacking
// sequence only changes the bending stiffness D (irrelevant to the pure
// membrane loads this project scores) and secondary effects (delamination
// resistance, thermal warpage) outside this project's scope. The sweep still
// enumerates all 256 ORDERED sequences (the catalog bullet's literal ask),
// but expect — and this project's THEORY.md/README discuss it as a genuine
// finding, not a bug — many exact ties among permutations of the same angle
// multiset (main.cu reports how many layups share the winning score).
//
// UNITS (SI throughout, documented once, CLAUDE.md §12):
//   Modulus / stiffness / strength   : Pa  (E1_pa, G12_pa, Xt_pa, Q11 ...)
//   Thickness                        : m   (t_ply_m)
//   In-plane load resultant N        : N/m (force per unit width — the
//                                      standard CLT load unit; NOT a stress)
//   Angle                            : degrees in data/API surfaces the
//                                      learner reads (angles_deg[]); radians
//                                      only inside trig calls (kDegToRad
//                                      converts at the one defined point per
//                                      function, CLAUDE.md §12 angle-wrap
//                                      discipline generalized to "convert
//                                      once, document the point")
//   Failure load factor lambda       : unitless — the scalar that multiplies
//                                      a reference load to reach first-ply
//                                      failure (>1 = safe margin, <1 = the
//                                      load already exceeds first-ply
//                                      capacity, =1 = exactly on the
//                                      Tsai-Wu envelope boundary)
//
// STACK ENCODING — a "layup_id" in [0, kNLayups) names ONE symmetric 8-ply
// stack via base-kNAngleAlphabet digits (decode_gait()-style, see 18.01):
//     layup_id = ((a0*4 + a1)*4 + a2)*4 + a3        a0..a3 in [0,4)
//     angles_deg[0..3] = alphabet[a0..a3]             (the 4 INDEPENDENT plies)
//     angles_deg[4..7] = angles_deg[3..0]             (mirrored -> symmetric)
// This is the ONE flattening every file in this project uses (decode_layup()
// below; CLAUDE.md §12 single-source rule).
//
// LOAD-CASE FORMAT — a LoadCase is (Nx_npm, Ny_npm, Nxy_npm), N/m, the three
// in-plane stress-resultant components of classical laminate theory. Two
// documented CASE SETS are loaded from data/sample/ (never hard-coded here —
// they are this project's "task data", CLAUDE.md §8): MIXED (16 combined
// Nx/Ny/Nxy directions sampling the whole load-angle circle) — the set that
// makes quasi-isotropic-like layups win — and ALIGNED (2 pure +-Nx cases) —
// the set that makes 0-heavy layups win. THEORY.md §the-problem walks why.
//
// Read this after: README.md/THEORY.md. Read this before: kernels.cu.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cstdint>
#include <cmath>     // sinf/cosf/sqrtf/fabsf/fminf — host declarations; nvcc
                     // supplies device overloads of the SAME names below

// ---------------------------------------------------------------------------
// HD — "__host__ __device__" under nvcc, nothing under cl.exe. The same
// trick 18.01/10.03 use: every physics function below is written ONCE and
// called UNCHANGED from the GPU kernels (kernels.cu), the CPU oracle
// (reference_cpu.cpp, compiled by cl.exe — never sees a CUDA keyword), and
// main.cu's analytic-gate diagnostics (main.cu IS a .cu file, so it simply
// calls the __host__ __device__ version as ordinary C++). Sharing the source
// removes hand-copy divergence as an explanation for any GPU-vs-CPU
// difference the §5 VERIFY gate measures — what is left over is purely
// sinf/cosf/sqrtf's independently-rounded host vs. device implementations
// (THEORY.md §numerical considerations).
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ---------------------------------------------------------------------------
// kDegToRad — the ONE conversion point angles pass through before any trig
// call (CLAUDE.md §12: angle units documented and converted at a defined
// point). Every angle stored in this project's data structures is DEGREES
// (human-readable in CSVs/logs); every angle handed to sinf/cosf is radians.
// ---------------------------------------------------------------------------
constexpr float kPi = 3.14159265358979323846f;
constexpr float kDegToRad = kPi / 180.0f;

// ===========================================================================
// Structural (compile-time) constants — NOT scenario data. These fix the
// SHAPE of the design space the catalog bullet names ("4^4 = 256 stack
// sequences" from the alphabet {0, +-45, 90}); like 18.01's kNLinks, they
// stay constexpr because the per-thread physics functions below use small
// FIXED-SIZE loops over them (register-resident, no device-side dynamic
// sizing — CUDA device code cannot size a runtime array).
// ===========================================================================
constexpr int kNPlies         = 8;   // full symmetric laminate ply count (this project's scope)
constexpr int kNIndepPlies    = kNPlies / 2;  // 4 independent angle slots define the symmetric stack
constexpr int kNAngleAlphabet = 4;   // {0, +45, -45, 90} — the catalog's documented alphabet
constexpr int kNLayups        = 256; // kNAngleAlphabet ^ kNIndepPlies = 4^4 (exhaustive enumeration)

constexpr int kEnvGridN = 128;       // envelope grid resolution per axis -> 128*128 = 16384 points/envelope

// Storage/display clamp for the envelope field (THEORY.md §numerics): the
// failure-load-FACTOR field diverges to +infinity as the applied load ->
// (0,0) (zero load can never fail), which is neither storable nor
// paintable. Clamping at a value far above the interesting factor=1
// boundary (kEnvFactorClamp >> 1) does not move the boundary at all — only
// flattens the field far inside the safe region, which is honestly
// documented wherever the clamp is applied (kernels.cu, reference_cpu.cpp,
// and the PGM/CSV writers in main.cu all use this ONE constant).
constexpr float kEnvFactorClamp   = 5.0f;   // unitless factor ceiling for storage/display
constexpr float kEnvFactorMinLoad = 1.0f;   // N/m: below this |N| the grid point is treated as "no load" -> clamp

// ---------------------------------------------------------------------------
// Verification-harness-only constants (NOT scenario data — mirrors 18.01's
// kVerifyCount precedent: these size an ANALYTIC GATE, not the experiment).
// ---------------------------------------------------------------------------
// Gate (ii): the isotropic-degenerate material. An ALUMINUM-like teaching
// value (E, nu) plus an equal-in-every-direction strength F0 — chosen so
// G12 = E/(2*(1+nu)) (the ISOTROPIC relation being tested: an orthotropic
// lamina with E1=E2=E and this G12 is fully rotation-invariant, THEORY.md
// derives why) and so the Tsai-Wu quadratic form is ALSO rotation-invariant
// (which additionally requires S12 = sqrt(F0^2/3) — i.e. F66 = 3*F11 —
// THEORY.md §the-math derives this from first principles: a genuinely
// interesting, non-obvious property of the -1/2 F12 normalization).
constexpr float kIsoE_Pa  = 70.0e9f;    // aluminum-class isotropic modulus (Pa) — SYNTHETIC teaching value
constexpr float kIsoNu    = 0.30f;      // isotropic Poisson ratio
constexpr float kIsoF0_Pa = 300.0e6f;   // equal Xt=Xc=Yt=Yc strength (Pa) — SYNTHETIC teaching value
constexpr int   kIsoGateDirs = 16;      // load directions swept around the circle for gate (ii)

// ===========================================================================
// Lamina — ONE ply's material properties: elastic constants + strengths.
// Shared, unmodified, by every ply of a laminate in this project (a single
// material system throughout — real hybrid laminates mix materials per ply;
// out of scope, README §Limitations).
// ===========================================================================
struct Lamina {
    // Elastic constants (plane-stress orthotropic lamina, material axes 1=fiber, 2=transverse).
    float E1_pa;    // longitudinal (fiber-direction) Young's modulus, Pa
    float E2_pa;    // transverse Young's modulus, Pa
    float G12_pa;   // in-plane shear modulus, Pa
    float nu12;     // major Poisson ratio (unitless): strain in 2 per unit strain in 1 under uniaxial sigma1
    float t_ply_m;  // single-ply cured thickness, m

    // Tsai-Wu strength parameters — MAGNITUDES (Pa), sign convention applied
    // inside tsaiwu_F() (THEORY.md §the-math derives F1..F66 from these five).
    float Xt_pa;    // longitudinal tensile strength
    float Xc_pa;    // longitudinal compressive strength (magnitude)
    float Yt_pa;    // transverse tensile strength
    float Yc_pa;    // transverse compressive strength (magnitude)
    float S12_pa;   // in-plane shear strength
};

// ---------------------------------------------------------------------------
// LoadCase — one in-plane load: the three CLT stress-resultant components
// (force per unit laminate width, N/m — NOT a stress; THEORY.md derives the
// distinction). Nxy=0 for every envelope-grid point (the ratified scope's
// "(Nx,Ny) plane" restriction); general (Nx,Ny,Nxy) triples populate the
// ranking sweep's MIXED case set.
// ---------------------------------------------------------------------------
struct LoadCase {
    float Nx_npm;   // N/m
    float Ny_npm;   // N/m
    float Nxy_npm;  // N/m
};

// ---------------------------------------------------------------------------
// AngleAlphabet — the 4 candidate ply angles (degrees), passed BY VALUE into
// the layup-sweep kernel (16 bytes — trivial parameter-memory cost, the same
// "small POD struct by value" choice 18.01's GaitGridParams makes).
// ---------------------------------------------------------------------------
struct AngleAlphabet {
    float deg[kNAngleAlphabet];
};

// ---------------------------------------------------------------------------
// Layup8 — one fully-decoded 8-ply symmetric stack (degrees), passed BY
// VALUE into the envelope kernel (32 bytes) so that kernel needs no separate
// device allocation just to name which laminate it is scoring.
// ---------------------------------------------------------------------------
struct Layup8 {
    float deg[kNPlies];
};

// ===========================================================================
// Physics — HD inline functions, the single source shared by the GPU
// kernels, the CPU oracle, and main.cu's analytic gates (file header above).
// Every function here is O(1) (a handful of plies, never a loop over time),
// so — unlike 08.01/18.01's per-step integrators — there is no accumulated
// numerical drift to reason about, only single-pass rounding (THEORY.md
// §numerics quantifies the whole chain end to end).
// ===========================================================================

// ---------------------------------------------------------------------------
// lamina_Q — the ply's stiffness matrix in ITS OWN material axes (1=fiber,
// 2=transverse), plane-stress reduced form (THEORY.md §the-math derives
// every line from the compliance matrix S = [[1/E1,-nu12/E1,0],[-nu21/E2,
// 1/E2,0],[0,0,1/G12]] inverted under plane stress, sigma3=tau13=tau23=0 —
// the standard CLT assumption for a thin laminate).
//
//   nu21 = nu12 * E2/E1     Maxwell-Betti reciprocity (nu21/E2 = nu12/E1) —
//                            NOT an independent material constant; computing
//                            it here (rather than storing it) keeps Lamina
//                            from carrying a redundant, potentially-
//                            inconsistent field.
//   Q11 = E1 / (1 - nu12*nu21)     Q22 = E2 / (1 - nu12*nu21)
//   Q12 = nu12*E2 / (1 - nu12*nu21) = nu21*E1 / (1 - nu12*nu21)  (symmetric)
//   Q66 = G12                       shear is UNCOUPLED from normal response
//                                    in material axes — off-axis coupling
//                                    (Q16, Q26) appears only after rotation
//                                    (transform_Qbar below).
// ---------------------------------------------------------------------------
HD inline void lamina_Q(const Lamina& m, float& Q11, float& Q12, float& Q22, float& Q66)
{
    const float nu21 = m.nu12 * m.E2_pa / m.E1_pa;   // reciprocal Poisson ratio (unitless)
    const float denom = 1.0f - m.nu12 * nu21;         // > 0 for any physically realizable orthotropic lamina
    Q11 = m.E1_pa / denom;
    Q22 = m.E2_pa / denom;
    Q12 = m.nu12 * m.E2_pa / denom;
    Q66 = m.G12_pa;
}

// ---------------------------------------------------------------------------
// transform_Qbar — rotate a ply's material-axis stiffness Q into LAMINATE
// (x-y) axes by angle theta (radians; kDegToRad converts at the call site).
// The standard "transformed reduced stiffness" identities (Jones, Mechanics
// of Composite Materials; THEORY.md re-derives them from the strain- and
// stress-transformation tensors so nothing here is a black box).
//
// theta is measured CCW from the laminate x-axis to the ply's fiber (1) axis
// — the right-handed, documented convention this whole project uses.
// At theta=0: Qbar == Q exactly (c=1,s=0 kills every s-bearing term) —
// GATE_CLT_SANITY (main.cu) checks this identity as a build-in-public sanity
// test of this very function.
// ---------------------------------------------------------------------------
HD inline void transform_Qbar(float Q11, float Q12, float Q22, float Q66, float theta_rad,
                              float& Qb11, float& Qb12, float& Qb16,
                              float& Qb22, float& Qb26, float& Qb66)
{
    const float c = cosf(theta_rad), s = sinf(theta_rad);
    const float c2 = c * c, s2 = s * s;
    const float c4 = c2 * c2, s4 = s2 * s2, s2c2 = s2 * c2;
    const float Qcomb = Q12 + 2.0f * Q66;   // appears in both Qb11 and Qb22 — computed once

    Qb11 = Q11 * c4 + 2.0f * Qcomb * s2c2 + Q22 * s4;
    Qb22 = Q11 * s4 + 2.0f * Qcomb * s2c2 + Q22 * c4;
    Qb12 = (Q11 + Q22 - 4.0f * Q66) * s2c2 + Q12 * (s4 + c4);
    Qb66 = (Q11 + Q22 - 2.0f * Q12 - 2.0f * Q66) * s2c2 + Q66 * (s4 + c4);
    // Qb16/Qb26: the OFF-AXIS shear-coupling terms — zero only at theta = 0
    // or 90 deg (c*s = 0). This coupling is exactly why a [+45] ply "leaks"
    // an applied Nx into shear strain, and why quasi-isotropic-like stacks
    // (which cancel it on average across angles) resist MIXED loading
    // better than a single-direction stack (THEORY.md §the-problem).
    Qb16 = (Q11 - Q12 - 2.0f * Q66) * c * c * c * s - (Q22 - Q12 - 2.0f * Q66) * c * s * s * s;
    Qb26 = (Q11 - Q12 - 2.0f * Q66) * c * s * s * s - (Q22 - Q12 - 2.0f * Q66) * c * c * c * s;
}

// ---------------------------------------------------------------------------
// solve3x3_sym — Cramer's-rule solve of a SYMMETRIC 3x3 linear system,
// fully unrolled and register-resident (the "33.01-style in-register
// small-batched-solve" this project's scope calls for, specialized to
// symmetric matrices since CLT's A matrix is ALWAYS symmetric — only 6 of
// its 9 entries are ever stored, here or anywhere in this project).
//
//   [a11 a12 a13] [x1]   [b1]
//   [a12 a22 a23] [x2] = [b2]
//   [a13 a23 a33] [x3]   [b3]
//
// Returns false (and zeros x) on a near-singular matrix — defensive; a
// physically valid laminate (nonzero ply thickness, valid elastic
// constants) always yields a strictly positive-definite A, so this branch
// should never fire on real data (CLAUDE.md error-handling discipline: fail
// loudly and visibly rather than silently propagate a NaN).
// ---------------------------------------------------------------------------
HD inline bool solve3x3_sym(float a11, float a12, float a13, float a22, float a23, float a33,
                            float b1, float b2, float b3,
                            float& x1, float& x2, float& x3)
{
    // det(A) via cofactor expansion along the first row.
    const float det = a11 * (a22 * a33 - a23 * a23)
                     - a12 * (a12 * a33 - a23 * a13)
                     + a13 * (a12 * a23 - a22 * a13);
    if (fabsf(det) < 1.0e-6f) { x1 = x2 = x3 = 0.0f; return false; }

    const float inv_det = 1.0f / det;
    // Cramer's rule: x_i = det(A with column i replaced by b) / det(A).
    x1 = (b1 * (a22 * a33 - a23 * a23) - a12 * (b2 * a33 - a23 * b3) + a13 * (b2 * a23 - a22 * b3)) * inv_det;
    x2 = (a11 * (b2 * a33 - a23 * b3) - b1 * (a12 * a33 - a23 * a13) + a13 * (a12 * b3 - b2 * a13)) * inv_det;
    x3 = (a11 * (a22 * b3 - b2 * a23) - a12 * (a12 * b3 - b2 * a13) + b1 * (a12 * a23 - a22 * a13)) * inv_det;
    return true;
}

// ---------------------------------------------------------------------------
// assemble_A — sum thickness-weighted Qbar over every ply into the
// laminate's 3x3 (symmetric) extensional stiffness matrix A.
//
//   A_ij = sum_k Qbar_ij(ply k) * t_ply
//
// No B (bending-extension coupling) or D (bending stiffness) matrix is
// computed anywhere in this project: this project's laminates are always
// SYMMETRIC about the midplane (B=0 exactly by construction — THEORY.md
// proves it from the z^2 term in the CLT integral canceling in mirrored
// pairs), and its loads are always pure in-plane (D never enters a
// zero-curvature membrane problem). GATE_CLT_SANITY checks A11=A22 for the
// cross-ply baseline as a symmetry sanity test of THIS function.
// ---------------------------------------------------------------------------
HD inline void assemble_A(const Lamina& mat, const float* angles_deg, int n_plies,
                          float& A11, float& A12, float& A16,
                          float& A22, float& A26, float& A66)
{
    float Q11, Q12, Q22, Q66;
    lamina_Q(mat, Q11, Q12, Q22, Q66);   // material-axis stiffness — SAME for every ply (one material system)

    A11 = A12 = A16 = A22 = A26 = A66 = 0.0f;
    for (int k = 0; k < n_plies; ++k) {
        float Qb11, Qb12, Qb16, Qb22, Qb26, Qb66;
        const float theta_rad = angles_deg[k] * kDegToRad;
        transform_Qbar(Q11, Q12, Q22, Q66, theta_rad, Qb11, Qb12, Qb16, Qb22, Qb26, Qb66);
        A11 += Qb11 * mat.t_ply_m; A12 += Qb12 * mat.t_ply_m; A16 += Qb16 * mat.t_ply_m;
        A22 += Qb22 * mat.t_ply_m; A26 += Qb26 * mat.t_ply_m; A66 += Qb66 * mat.t_ply_m;
    }
}

// ---------------------------------------------------------------------------
// ply_stress — given the (shared) midplane strain and ONE ply's orientation,
// return that ply's stress in ITS OWN material axes.
//
// Step 1 — strain transformation (global x-y -> material 1-2), ENGINEERING
// shear-strain convention (gamma_xy, not tensor strain gamma/2 — the CLT
// textbook convention throughout, e.g. Jones eq. 2.75-2.77):
//     eps1    =  epsx*c^2 + epsy*s^2 + gxy*s*c
//     eps2    =  epsx*s^2 + epsy*c^2 - gxy*s*c
//     gamma12 =  2*(epsy-epsx)*s*c + gxy*(c^2-s^2)
// Step 2 — Hooke's law in MATERIAL axes with the UNROTATED Q (not Qbar —
// each ply "sees" its own local stiffness, THEORY.md draws the diagram):
//     sigma1 = Q11*eps1 + Q12*eps2
//     sigma2 = Q12*eps1 + Q22*eps2
//     tau12  = Q66*gamma12
// ---------------------------------------------------------------------------
HD inline void ply_stress(float Q11, float Q12, float Q22, float Q66, float theta_rad,
                          float epsx, float epsy, float gxy,
                          float& s1, float& s2, float& t12)
{
    const float c = cosf(theta_rad), s = sinf(theta_rad);
    // NOTE: named "sq" (not "s2") on purpose — this function's OUTPUT
    // parameter is already named s2 (ply stress sigma2); reusing s2 for
    // sin^2(theta) here would shadow it and silently write the wrong thing.
    const float c2 = c * c, sq = s * s, sc = s * c;

    const float eps1 = epsx * c2 + epsy * sq + gxy * sc;
    const float eps2 = epsx * sq + epsy * c2 - gxy * sc;
    const float g12  = 2.0f * (epsy - epsx) * sc + gxy * (c2 - sq);

    s1  = Q11 * eps1 + Q12 * eps2;
    s2  = Q12 * eps1 + Q22 * eps2;
    t12 = Q66 * g12;
}

// ---------------------------------------------------------------------------
// tsaiwu_F — the six Tsai-Wu strength parameters, derived ONCE per laminate
// (they depend only on the shared material's five strengths, never on ply
// orientation or load — computing them outside the per-ply loop avoids
// redundant divides/sqrt, a small but real efficiency the comment calls
// out honestly rather than silently, CLAUDE.md §6.1 rule 6).
//
// THE CRITERION (THEORY.md §the-math derives every term from a general
// quadratic failure surface F_i*sigma_i + F_ij*sigma_i*sigma_j = 1):
//     F1  = 1/Xt - 1/Xc      F2  = 1/Yt - 1/Yc     (LINEAR terms — nonzero
//         only when tensile and compressive strength differ, i.e. the
//         failure surface is NOT centered on sigma=0)
//     F11 = 1/(Xt*Xc)        F22 = 1/(Yt*Yc)       (QUADRATIC terms)
//     F66 = 1/S12^2                                 (shear quadratic term)
//     F12 = -0.5*sqrt(F11*F22)                      (the INTERACTION term —
//         THE STANDARD "-1/2" NORMALIZATION: it is the value that makes the
//         failure surface a closed, bounded ellipsoid for the widest range
//         of biaxial strength ratios without extra experimental biaxial
//         data — Tsai & Hahn's classic recommendation, not derivable from
//         Xt/Xc/Yt/Yc/S12 alone; THEORY.md discusses the alternative
//         (measuring F12 from a real biaxial test) and why -1/2 is used
//         here.  Gate (ii) shows this SAME normalization is also exactly
//         what a rotation-invariant (isotropic) failure surface requires
//         when combined with F66 = 3*F11 — a genuinely elegant property.)
// ---------------------------------------------------------------------------
HD inline void tsaiwu_F(const Lamina& m, float& F1, float& F2, float& F11, float& F22, float& F66, float& F12)
{
    F1  = 1.0f / m.Xt_pa - 1.0f / m.Xc_pa;
    F2  = 1.0f / m.Yt_pa - 1.0f / m.Yc_pa;
    F11 = 1.0f / (m.Xt_pa * m.Xc_pa);
    F22 = 1.0f / (m.Yt_pa * m.Yc_pa);
    F66 = 1.0f / (m.S12_pa * m.S12_pa);
    F12 = -0.5f * sqrtf(F11 * F22);
}

// ---------------------------------------------------------------------------
// tsaiwu_ab — the Tsai-Wu criterion evaluated at a UNIT-scale stress state,
// packaged as the (a, b) coefficients of the quadratic solve_lambda() below.
//
// Because ply stress is LINEAR in the applied load (Hooke's law all the way
// through), scaling the load by lambda scales (s1,s2,t12) by lambda too.
// Substituting into the Tsai-Wu criterion (= 1 at failure):
//     F1*(lam*s1) + F2*(lam*s2) + F11*(lam*s1)^2 + F22*(lam*s2)^2
//         + F66*(lam*t12)^2 + 2*F12*(lam*s1)*(lam*s2)  =  1
//     =>  a*lam^2 + b*lam - 1 = 0
//     a = F11*s1^2 + F22*s2^2 + F66*t12^2 + 2*F12*s1*s2      (QUADRATIC terms)
//     b = F1*s1 + F2*s2                                       (LINEAR terms)
// solve_lambda() below finds the ply's own failure load factor from these —
// no per-load-magnitude re-derivation needed (THEORY.md §the-math).
// ---------------------------------------------------------------------------
HD inline void tsaiwu_ab(float F1, float F2, float F11, float F22, float F66, float F12,
                         float s1, float s2, float t12, float& a, float& b)
{
    a = F11 * s1 * s1 + F22 * s2 * s2 + F66 * t12 * t12 + 2.0f * F12 * s1 * s2;
    b = F1 * s1 + F2 * s2;
}

// ---------------------------------------------------------------------------
// solve_lambda — the positive root of a*lambda^2 + b*lambda - 1 = 0: the
// EXACT, closed-form load-scaling factor at which this ply's Tsai-Wu index
// reaches 1 (first-ply failure). No Newton iteration is needed anywhere in
// this project — the linear-elastic assumption behind CLT buys a genuine
// algebraic closed form (contrast a nonlinear-material FEA package, which
// would need an iterative solve here; THEORY.md §the-algorithm makes this
// contrast explicit).
//
// Two guarded degenerate cases (both physically rare — near-zero load
// component along a direction this ply cannot "feel" — but always possible
// at grid/sweep boundaries, so both are handled rather than left as a NaN):
//   * a ~ 0 (near-linear criterion): fall back to the linear equation
//     b*lambda = 1.
//   * a and b both ~ 0 (this ply feels essentially zero stress under this
//     load): return the sentinel kEnvFactorClamp-scale "practically
//     infinite, never fails" value (1e30) — the caller clamps for storage.
// The quadratic in general has two real roots (a "load in this direction"
// root and a "load in the exact opposite direction" root, since Tsai-Wu's
// surface is centered off the origin only through the LINEAR F1,F2 terms);
// this project always applies load in the +lambda sense, so the SMALLEST
// POSITIVE root is the one that matters.
// ---------------------------------------------------------------------------
HD inline float solve_lambda(float a, float b)
{
    constexpr float kHuge = 1.0e30f;   // "practically never fails in this direction" sentinel
    constexpr float kEps  = 1.0e-20f;  // near-zero guard (Pa^-2 scale coefficients are tiny; see THEORY.md §numerics)

    if (fabsf(a) < kEps) {
        if (fabsf(b) < kEps) return kHuge;          // no stress at all in this ply/direction
        const float lam = 1.0f / b;
        return (lam > 0.0f) ? lam : kHuge;           // negative root: this ply cannot fail in the +lambda sense
    }
    const float disc = b * b + 4.0f * a;              // discriminant of a*L^2+b*L-1=0 (b^2 - 4*a*(-1))
    if (disc < 0.0f) return kHuge;                    // no real root (rare; guarded)
    const float sq = sqrtf(disc);
    const float lam1 = (-b + sq) / (2.0f * a);
    const float lam2 = (-b - sq) / (2.0f * a);
    float lam = kHuge;
    if (lam1 > 0.0f) lam = fminf(lam, lam1);
    if (lam2 > 0.0f) lam = fminf(lam, lam2);
    return lam;
}

// ---------------------------------------------------------------------------
// laminate_failure_factor — THE per-thread computation: assemble A, solve
// for midplane strain, evaluate every ply's Tsai-Wu failure factor, return
// the MINIMUM (first-ply failure — the laminate fails as soon as its first
// ply does; THEORY.md §the-problem justifies first-ply-failure as the
// standard, conservative design criterion this project teaches, versus the
// more involved progressive/last-ply-failure analyses production tools add).
//
// n_plies is a RUNTIME parameter (<= kNPlies) so this ONE function serves
// both the main 8-ply sweep/envelope paths AND the analytic gates' 1-ply
// closed-form and isotropic checks (main.cu) — reusing the exact tested
// code path is the strongest form of verification available (CLAUDE.md §9).
// ---------------------------------------------------------------------------
HD inline float laminate_failure_factor(const Lamina& mat, const float* angles_deg, int n_plies,
                                        const LoadCase& lc)
{
    float Q11, Q12, Q22, Q66;
    lamina_Q(mat, Q11, Q12, Q22, Q66);

    float A11, A12, A16, A22, A26, A66;
    assemble_A(mat, angles_deg, n_plies, A11, A12, A16, A22, A26, A66);

    // B=0 (symmetric laminate) -> a pure membrane load produces UNIFORM
    // midplane strain with zero curvature: solve A*eps0 = N once, every ply
    // shares this eps0 (THEORY.md §the-math walks the CLT integral that
    // proves this).
    float epsx, epsy, gxy;
    solve3x3_sym(A11, A12, A16, A22, A26, A66, lc.Nx_npm, lc.Ny_npm, lc.Nxy_npm, epsx, epsy, gxy);

    // Tsai-Wu strength parameters, computed ONCE (they do not depend on ply
    // angle or load — see tsaiwu_F's comment).
    float F1, F2, F11, F22, F66, F12;
    tsaiwu_F(mat, F1, F2, F11, F22, F66, F12);

    // First-ply failure = MIN over plies of each ply's own failure lambda.
    float worst_lambda = 1.0e30f;
    for (int k = 0; k < n_plies; ++k) {
        const float theta_rad = angles_deg[k] * kDegToRad;
        float s1, s2, t12;
        ply_stress(Q11, Q12, Q22, Q66, theta_rad, epsx, epsy, gxy, s1, s2, t12);
        float a, b;
        tsaiwu_ab(F1, F2, F11, F22, F66, F12, s1, s2, t12, a, b);
        const float lam = solve_lambda(a, b);
        worst_lambda = fminf(worst_lambda, lam);
    }
    return worst_lambda;
}

// ---------------------------------------------------------------------------
// decode_layup — turn a flattened layup_id in [0, kNLayups) into its 8
// physical ply angles (degrees), mirrored for symmetry. The ONE flattening
// every file in this project uses (file header's STACK ENCODING note);
// kernels.cu, reference_cpu.cpp, and main.cu's ranking/report code all
// decode this exact same way, from this ONE function (CLAUDE.md §12).
// ---------------------------------------------------------------------------
HD inline void decode_layup(int layup_id, const float* alphabet_deg, float* angles_deg_out)
{
    int rem = layup_id;
    const int a3 = rem % kNAngleAlphabet; rem /= kNAngleAlphabet;
    const int a2 = rem % kNAngleAlphabet; rem /= kNAngleAlphabet;
    const int a1 = rem % kNAngleAlphabet; rem /= kNAngleAlphabet;
    const int a0 = rem % kNAngleAlphabet;

    const float t0 = alphabet_deg[a0], t1 = alphabet_deg[a1];
    const float t2 = alphabet_deg[a2], t3 = alphabet_deg[a3];

    // [t0/t1/t2/t3]_s — the "_s" (symmetric) notation: mirror the 4
    // independent plies about the laminate midplane.
    angles_deg_out[0] = t0; angles_deg_out[1] = t1; angles_deg_out[2] = t2; angles_deg_out[3] = t3;
    angles_deg_out[4] = t3; angles_deg_out[5] = t2; angles_deg_out[6] = t1; angles_deg_out[7] = t0;
}

// ---------------------------------------------------------------------------
// envelope_grid_point — map a 128x128 grid index (i=row, j=column) to the
// (Nx,Ny) load it represents (Nxy=0 — the ratified scope's "(Nx,Ny) plane"),
// spanning [-n_max_npm, +n_max_npm] inclusive at both ends of each axis.
// Row i -> Ny (increases DOWNWARD in the PGM, matching image row order —
// documented at the PGM writer in main.cu); column j -> Nx (increases
// RIGHTWARD). This is the ONE grid-index convention every envelope-related
// file in this project uses.
// ---------------------------------------------------------------------------
HD inline void envelope_grid_point(int i, int j, float n_max_npm, LoadCase& out)
{
    const float frac_x = static_cast<float>(j) / static_cast<float>(kEnvGridN - 1);
    const float frac_y = static_cast<float>(i) / static_cast<float>(kEnvGridN - 1);
    out.Nx_npm  = -n_max_npm + (2.0f * n_max_npm) * frac_x;
    out.Ny_npm  = -n_max_npm + (2.0f * n_max_npm) * frac_y;
    out.Nxy_npm = 0.0f;
}

// ---------------------------------------------------------------------------
// envelope_factor_at — laminate_failure_factor() specialized for one
// envelope grid point, with the near-zero-load clamp applied (file header's
// kEnvFactorClamp note). Shared by the envelope kernel AND its CPU oracle so
// the clamp logic itself cannot be a source of GPU-vs-CPU disagreement.
// ---------------------------------------------------------------------------
HD inline float envelope_factor_at(const Lamina& mat, const float* angles_deg, int n_plies,
                                   int i, int j, float n_max_npm)
{
    LoadCase lc;
    envelope_grid_point(i, j, n_max_npm, lc);
    const float mag2 = lc.Nx_npm * lc.Nx_npm + lc.Ny_npm * lc.Ny_npm;
    if (mag2 < kEnvFactorMinLoad * kEnvFactorMinLoad) return kEnvFactorClamp;   // "no load": treat as maximally safe
    return fminf(laminate_failure_factor(mat, angles_deg, n_plies, lc), kEnvFactorClamp);
}

// ===========================================================================
// GPU launchers (defined in kernels.cu). Device output arrays are allocated
// by the caller (main.cu); these functions allocate nothing and free
// nothing (stateless, the 08.01/18.01 convention).
// ===========================================================================

// launch_layup_sweep — one thread per (layup_id, case_id) PAIR (the file
// header's SWEEP): d_cases is a DEVICE array of n_cases LoadCase, d_factor
// is a DEVICE array of kNLayups*n_cases floats OUT, flattened
// g = layup_id*n_cases + case_id (layup slowest-varying, case fastest —
// the ONE flattening main.cu's host-side per-layup reduction also uses).
void launch_layup_sweep(const Lamina& mat, const AngleAlphabet& alpha,
                        const LoadCase* d_cases, int n_cases,
                        float* d_factor);

// launch_envelope — one thread per (Nx,Ny) grid point, g = i*kEnvGridN + j.
// d_factor is a DEVICE array of kEnvGridN*kEnvGridN floats OUT (clamped —
// see envelope_factor_at above).
void launch_envelope(const Lamina& mat, const Layup8& layup, float n_max_npm,
                     float* d_factor);

// ---------------------------------------------------------------------------
// CPU references (reference_cpu.cpp) — FULL brute-force recomputation (every
// problem size in this project is small enough — at most kEnvGridN^2 =
// 16384 points — that a full oracle is cheap; unlike 18.01's stride-sampled
// spot check, this project's §5 VERIFY gate checks EVERY element).
// ---------------------------------------------------------------------------
void layup_sweep_cpu(const Lamina& mat, const AngleAlphabet& alpha,
                     const LoadCase* cases, int n_cases, float* out_factor);

void envelope_cpu(const Lamina& mat, const Layup8& layup, float n_max_npm, float* out_factor);

#endif // PROJECT_KERNELS_CUH
