# 26.01 — Topology optimization (SIMP/level-set) on GPU for lightweight links and brackets — flagship design project

**Difficulty:** ★ beginner · **Domain:** 26. Mechanical Design & Structures

> Catalog bullet (source of truth, verbatim): `★ Topology optimization (SIMP/level-set) on GPU for lightweight links and brackets — flagship design project`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

**This project answers a question every mechanical engineer on a robotics team asks: given a load
and a place to bolt something down, where should the material actually go?** It implements SIMP
(Solid Isotropic Material with Penalization) compliance-minimization topology optimization — the
"99-line/88-line topopt" algorithm lineage, ported to the GPU and taught from first principles —
on a structured plane-stress finite-element mesh. Each outer iteration solves the linear-elasticity
equations for the current material layout (a matrix-free, GPU-accelerated conjugate-gradient FEA
solve — this project's computational heart), computes how much each tiny patch of material is
contributing to structural stiffness, smooths that signal to avoid a well-known numerical
pathology, and nudges the design toward a stiffer, lighter shape. Run it long enough and a uniform
gray slab of "maybe material" resolves into a crisp lattice of struts — the same organic-looking
shapes that show up on 3D-printed aerospace brackets and, increasingly, robot links.

Two load cases ship: the classic **half-MBB beam** (the field's textbook validation case — its
expected diagonal-strut result is famous enough to recognize on sight) and a **robot L-bracket**
(an L-shaped domain bolted to a frame along one edge and loaded by a motor flange at the other —
the design-for-robotics story this flagship exists to tell). Both are fully implemented and run
end to end; the catalog's "level-set" alternative is documented, not implemented, per this
project's ratified scope (`THEORY.md` "The math" explains the trade-off between the two families).

## What this computes & why the GPU helps

Per outer optimization iteration, the dominant cost is solving `K(rho) U = F` — a sparse linear
system with tens of thousands of unknowns whose matrix changes every iteration (because the
material layout `rho` just changed). This project solves it **matrix-free**: conjugate gradient
never needs the matrix itself, only its ACTION on a vector, computed on the fly by every node
GATHERING contributions from its (at most 4) incident elements' densities and a small, precomputed
element stiffness matrix — no global stiffness matrix is ever assembled.

- **Pattern:** the FEA solve is a **batched stencil + reduce** (each CG iteration: one gather-stencil
  matvec kernel + several small reduction kernels for the dot products); the sensitivity and
  density-filter stages are pure **map** kernels (one thread per element).
- **Measured reality:** two full 80-iteration SIMP optimizations (MBB: 120x40 elements; bracket:
  80x80 elements) plus three independent verification stages all complete in **~8 seconds** on the
  reference GPU — comfortably inside this project's documented time budget, and fast enough that
  the whole pipeline (FEA solve, sensitivity, filter, Optimality-Criteria update) genuinely runs on
  the GPU where it dominates, not just the toy kernel a slower demo might hide behind.
- **Warm-starting matters:** the same CG solve needs the full 400-iteration cap on a cold start but
  as few as 44 iterations once the design has settled — see `THEORY.md` "The GPU mapping" for why
  this is not a minor detail but the reason the demo fits its time budget at all.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy/design stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial
whole (see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** cross-cutting **mechanical design & structures** — this is an OFFLINE design
  tool, not a runtime component in the sense/plan/act loop; it runs once (or a few times, per
  design iteration) long before a robot exists, producing a part geometry that everything else in
  the stack then treats as a fixed, given structure.
- **Upstream inputs:** the LOAD CASE this project optimizes against comes from earlier design-time
  analysis — most directly, **09.03 (GPU Featherstone: batched ABA forward dynamics + RNEA inverse
  dynamics)**, whose inverse-dynamics (RNEA) pass computes the joint torques/forces a planned motion
  profile demands, which become exactly the point loads this project's `LOAD` scenario rows encode.
  A real design process would also sweep multiple load cases (different poses, worst-case
  accelerations, safety-factored static loads) — this project's demo runs one representative case
  per scenario, and README "Limitations" states that scoping choice honestly.
- **Downstream consumers:** the optimized density field is a MANUFACTURING input, not a runtime
  message. Most directly, **27.05 (3D printing: GPU slicing, support generation, FDM warp/thermal
  sim, SLM melt-pool sim)** — topology-optimized brackets are THE canonical additive-manufacturing
  use case, because their organic strut geometry is often unmachinable by traditional subtractive
  methods but straightforward to 3D print (PRACTICE §1 makes this concrete). The density field also
  feeds a CAD reinterpretation step (PRACTICE §3) before it becomes a manufacturable part.
- **Rate / latency budget:** **not a real-time component at all** — this is an offline design tool
  that runs at design time, not on the robot. This project's own demo runs both load cases end to
  end in ~8 seconds; a production optimization on a finer, unstructured mesh with multiple load
  cases can take minutes to hours, still entirely offline (SYSTEM_DESIGN item 1's rate/latency
  table does not apply here — the honest answer is "whatever the design cycle can afford").
- **Reference robot(s):** the **6-DoF manipulator work cell** (SYSTEM_DESIGN's reference robot with
  domains 01/19/09/06/07/08/21/24) — its links and joint brackets are exactly what this project
  designs — and the **quadruped** (domains 13/04/05/10/12/24/25) — leg-link mass sits directly in
  the dynamics this project's upstream 09.03 load cases would characterize, and legged-robot link
  design is one of the most mass-sensitive structural problems in robotics.
- **In production:** a shipping robotics company runs this class of optimization inside a
  commercial tool — Altair OptiStruct, Dassault TOSCA/Simulia, nTopology, or Ansys Mechanical's
  topology module — on an unstructured, adaptively-refined mesh with manufacturing constraints
  (minimum feature size, 3D-print overhang angles, milling draw directions) built directly into the
  optimization (`THEORY.md` "Where this sits in the real world"; `PRACTICE.md` §2 names illustrative
  tiers). This project's from-scratch structured-grid solver is the teaching core those tools
  industrialize.
- **Owning team:** **mechanical design engineering** (SYSTEM_DESIGN item 5) — working closely with
  structural/FEA analysis and, once a design is finalized, manufacturing engineering (PRACTICE §4).

## The algorithm in brief

- **Plane-stress Q4 finite elements** — bilinear shape functions, 2x2 Gauss-quadrature element
  stiffness (derived numerically at startup, not hardcoded). → [THEORY.md](THEORY.md) §The math.
- **SIMP density interpolation** `E(rho) = Emin + rho^3(E0-Emin)` — penalizes intermediate
  densities so the optimizer prefers pure solid or pure void. → THEORY §The math.
- **Matrix-free, Jacobi-preconditioned conjugate gradient** — the GPU workhorse: `K*p` computed by
  node-GATHER over incident elements, no global matrix ever assembled. → THEORY §The GPU mapping.
- **Adjoint-free compliance sensitivity** `dc/drho_e = -dE/drho_e * u_e^T KE_hat u_e` — one cheap
  per-element kernel once `U` is known. → THEORY §The math.
- **Sigmund's density-weighted sensitivity filter** — the checkerboard-pathology fix and the
  mesh-independence guarantee. → THEORY §The math.
- **Optimality Criteria update with bisection** on the volume-constraint Lagrange multiplier (host).
  → THEORY §The math.
- **Level-set topology optimization** — the catalog's named alternative; documented, not
  implemented (ratified scope). → THEORY §Where this sits in the real world.

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/topology-optimization-on-gpu-for-lightweight.sln`](build/topology-optimization-on-gpu-for-lightweight.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/topology-optimization-on-gpu-for-lightweight.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none** — CUDA runtime + C++17 standard library only.
No cuSOLVER/cuBLAS/etc.: the whole point of the matrix-free CG solver is that it needs nothing
beyond a handful of hand-written kernels (THEORY.md "The GPU mapping" explains why that is the
right call at this problem's scale, not a missed optimization).

## Run the demo

One command, from this folder (builds first if needed, runs both load cases plus three verification
stages, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) — including
**the two design images to open** (`demo/out/topology_mbb.pgm`, `demo/out/topology_bracket.pgm`)
and the convergence-history CSV to plot.

## Data

The committed samples are **problem definitions, not measurements**: `data/sample/mbb_scenario.csv`
and `data/sample/bracket_scenario.csv` (~550 bytes each, synthetic, no RNG — every field is a
deliberately chosen constant, written by `scripts/make_synthetic.py`) — mesh size, material
(Aluminum 6061-T6, illustrative), volume target, and boundary conditions for the two load cases
described above. No public dataset applies (there is nothing to synthesize FROM — the "data" is the
optimization problem itself); `scripts/download_data.ps1` is an honest no-op. Full field
documentation and checksums: [`data/README.md`](data/README.md).

## Expected output

Eleven stable lines — banner, `PROBLEM:`, five independent `<STAGE>: PASS` gates (`VERIFY`,
`PATCH`, `BEAM`, `MBB`, `BRACKET`), two `SCENARIO:` lines, `ARTIFACT:`, and the aggregate
`RESULT: PASS` — checked as a subset diff by
[`demo/expected_output.txt`](demo/expected_output.txt). Four independent verification ideas, each
gating on a MEASURED value with a documented margin (`THEORY.md` "How we verify correctness" has
the full derivation and numbers for every one of these):

1. **The §5 GPU-vs-CPU gate (`VERIFY`)** — one full inner iteration (CG + sensitivity + filter) on
   an intermediate-density problem, GPU kernels vs. the CPU oracle. Measured worst relative
   displacement deviation: `4.3e-3` (gate `1e-2`); compliance deviation `5.2e-8` (gate `5e-3`).
2. **The patch test (`PATCH`)** — a solid strip under uniform tension must reproduce the *exact*
   closed-form linear displacement field (the standard FEM correctness check). Measured relative
   error: `2.8e-6` (gate `1e-3`).
3. **The cantilever beam gate (`BEAM`)** — a solid cantilever's tip deflection vs. Euler-Bernoulli/
   Timoshenko beam theory. Measured: `0.8%` below the Timoshenko value (gate `5%`) — the honest,
   named residual is fully-integrated Q4 element shear locking, not a bug.
4. **Optimization sanity (`MBB`/`BRACKET`)** — volume fraction within `0.02` of target (both cases:
   exact `0.4000`); compliance monotone non-increasing from iteration 6 onward within `0.5%` slack
   (measured worst uptick: `~0.07%`); largest connected solid component holds `>= 90%` of material
   (both cases measured: **100%**).

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — the mesh/DOF layout, the SIMP interpolation, and (the
   project's most important derivation) why element size `h` never appears in the FEA math.
2. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — start here for the MATH: `compute_KE_hat()`
   derives the element stiffness matrix via Gauss quadrature (no magic numbers), and the sequential
   `topo_*_cpu` functions are the plainest possible statement of every algorithm this project runs.
3. [`src/kernels.cu`](src/kernels.cu) — the heart: the matrix-free GATHER matvec kernel (read the
   file header's GATHER-vs-SCATTER discussion first), the CG solver built from small vector
   kernels, and the sensitivity/filter kernels.
4. [`src/main.cu`](src/main.cu) — the orchestration: scenario loading, the SIMP outer loop
   (`run_simp`), the Optimality-Criteria bisection (`oc_update`), the three verification stages, and
   the PGM/CSV artifact writers. The single most interesting thing to look at: `run_simp()`'s
   warm-start comment, and then the measured `[info]` line it produces.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers (copied, not shared — §4 rule).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **Sigmund (2001), "A 99 line topology optimization code written in Matlab"** and **Andreassen et
  al. (2011), "Efficient topology optimization in MATLAB using 88 lines of code"** — the exact
  algorithm lineage this project ports to the GPU; `compute_KE_hat()`'s cross-check against their
  published constant is a direct, checkable link.
- **Bendsoe & Sigmund, "Topology Optimization: Theory, Methods, and Applications"** — the field's
  standard textbook; owns the OC-update KKT derivation this project's THEORY.md summarizes.
- **Altair OptiStruct, Dassault TOSCA/Simulia, nTopology, Ansys Mechanical (topology module)** —
  the commercial-grade descendants: unstructured adaptive meshes, manufacturing-aware constraints,
  industrial sparse/multigrid solvers. Study what production adds; this project's from-scratch
  structured solver is the teaching core they industrialize.
- **Wang, Wang & Guo (2003) / Allaire, Jouve & Toader (2004)** — the level-set topology-
  optimization lineage this project documents (THEORY.md) but does not implement.
- **cuRobo / Isaac Lab / MuJoCo (via 09.03/10.03)** — the dynamics simulators whose inverse-
  dynamics output is exactly this project's upstream load-case source (README "System context").

## Exercises

1. **Plot the artifacts:** `demo/out/convergence.csv` → compliance and volume-fraction vs.
   iteration for both cases; overlay the two on one plot and compare convergence speed. Open
   `demo/out/topology_mbb.pgm` and `topology_bracket.pgm` in any image viewer.
2. **Break the filter:** set `kFilterR=0` (or skip the filter kernel entirely) in `kernels.cuh`/
   `main.cu` and re-run the MBB case — watch the checkerboard pathology THEORY.md describes
   reappear, then explain why from the filter's derivation.
3. **Feel shear locking:** in `main.cu`'s beam gate, reduce `nely` from 8 to 2 or 3 and observe how
   much further the FEA deflection falls below the Timoshenko prediction — this is the signature of
   Q4 element locking, not numerical error.
4. **GATHER vs. SCATTER:** implement a one-thread-per-element, atomicAdd-based SCATTER variant of
   `matvec_gather_kernel` and measure its runtime against the gather version at this project's mesh
   sizes — quantify the atomic-contention cost the file header predicts.
5. **Fuse the reduction:** replace `kernels.cu`'s host-finishing `dot()` helper with a device-side
   two-level reduction (no host round trip) and measure the per-CG-iteration time this removes.

## Limitations & honesty

- **Structured grid, not unstructured/adaptive.** Production tools mesh the design domain with
  triangles/quads that conform to (and refine near) the evolving boundary; this project's fixed
  Cartesian grid stair-steps curved/diagonal features and cannot locally refine — the honest
  teaching trade-off for a matrix-free GPU solver whose regular memory access pattern depends on
  exactly this structure (THEORY.md "The GPU mapping").
- **`Emin/E0 = 1e-3`, not the more common `1e-9`.** A documented conditioning trade for a
  Jacobi-only (not multigrid) preconditioner — THEORY.md "Numerical considerations" derives the
  reasoning and cites the measured CG iteration counts that justify it.
- **One load case per scenario.** A real bracket design would optimize against several load cases
  (different poses, worst-case accelerations, safety factors) simultaneously; this project
  demonstrates the pipeline on one representative case per scenario (README "System context").
- **2D plane stress only** — a real bracket can fail by out-of-plane buckling even when its in-plane
  compliance (what this project optimizes) is excellent; PRACTICE §1 and §3 discuss the physical
  and validation-testing implications.
- **SIMP only; level-set is documented, not implemented** — the ratified scope for this project
  (THEORY.md "The math" and "Where this sits in the real world" cover the alternative honestly).
- **Sim-validated only, not a manufacturing release.** This project's output is a DENSITY FIELD, not
  a manufacturable part — turning it into one requires a CAD reinterpretation and independent
  structural validation step (PRACTICE §3) that this project does not perform. No claim is made that
  either committed design (MBB coupon or bracket) is fit to manufacture or load-bear as-is; nothing
  here is safety-certified (CLAUDE.md §1), and any real hardware use of a topology-optimized bracket
  demands the full engineering validation process PRACTICE.md describes.
