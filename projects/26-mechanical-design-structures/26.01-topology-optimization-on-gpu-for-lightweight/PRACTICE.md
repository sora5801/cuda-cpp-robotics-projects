# 26.01 — Topology optimization (SIMP/level-set) on GPU for lightweight links and brackets — flagship design project: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

**This project's physical carrier is the bracket or link its density field describes** — a solid
part cut, cast, milled, or printed from the material this project's `E0_PA` names. Two very
different manufacturing realities apply depending on how the optimized shape gets made:

- **Machining (subtractive) a topology-optimized part.** The organic, diagonal-strut shapes this
  project produces (both the MBB coupon and the L-bracket — see `demo/out/*.pgm`) are the opposite
  of what a 3-axis CNC mill wants: undercuts, non-planar internal faces, and struts at arbitrary
  angles all demand either a 5-axis machine, custom fixturing, or manual reorientation between
  operations — all of which raise cost and cycle time sharply relative to a simple prismatic
  bracket. Production topology-optimization tools address this with **manufacturing constraints**
  built into the optimization itself (draw-direction constraints, minimum feature size, planar-face
  preferences) — this project's from-scratch solver does not implement any of them (README
  "Limitations"), so its raw output should be read as a STARTING POINT for a manufacturable
  redesign, not a final part geometry.
- **3D printing (additive) a topology-optimized part** is the natural match: an FDM, SLS, or metal
  SLM/DMLS printer builds up material layer by layer and does not care whether a strut is diagonal,
  curved, or internal — which is exactly why topology optimization and additive manufacturing grew
  up together in industry (this project's downstream cross-reference is **27.05**, 3D printing:
  slicing, support generation, and warp/melt-pool simulation). Printing still imposes real
  constraints this project's solver does not model: **overhang angles** (unsupported material below
  ~45° from vertical typically needs sacrificial support structures, adding cost and post-processing
  — see the diagonal struts in `demo/out/topology_bracket.pgm` and imagine which build orientation
  minimizes them), minimum wall thickness (thin struts near the SIMP density threshold can print as
  fragile or fail entirely), and for metal printing, residual thermal stress from rapid layer-by-
  layer solidification (27.05's warp/thermal simulation exists specifically to predict this).
- **What breaks in the field:** a topology-optimized bracket's failure mode is usually NOT the
  strut geometry itself (which this project's compliance objective directly optimizes) but the
  **reentrant (concave) corner** where a notch begins — this project's L-bracket case has exactly
  one, where the "L" shape's inner corner sits. Continuum elasticity has a genuine stress
  singularity there (stress is unbounded in the idealized model as the corner radius goes to zero);
  every real manufacturing process fillets it to a finite radius, and that fillet radius, chosen at
  the CAD-reinterpretation step (§3), is a direct, checkable fatigue-life lever a design engineer
  controls that this project's continuum FEA model does not represent.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-10. All parts, materials, and prices named below are **illustrative examples,
never endorsements**; they go stale — verify current before relying on any of them.*

This project produces a PART, not a control-loop or sensor pipeline — so "hardware" here means the
manufacturing/material stack that turns its density field into a physical bracket, tiered by
process:

| Tier | Illustrative choice (2026) | Role |
|---|---|---|
| Material (this project's default) | Aluminum 6061-T6 (`E0_PA=6.89e10`, this project's `data/sample/*.csv`) | Good stiffness-to-weight, widely machinable/weldable, the default "generic structural metal" of the mechanical-design world |
| Hobby-tier manufacture | Consumer FDM 3D printing (PLA/PETG/nylon, ~US$200–2000 printers) | Fast, cheap iteration on FIT and rough geometry — not load-bearing validation (plastic, not the aluminum this project's material constants model) |
| Research-tier manufacture | Desktop metal printing (bound-metal FDM, e.g. sinter-and-print systems) or 3-axis CNC from billet (~US$1k–10k per part in low volume) | Load-bearing prototypes in the actual specified material |
| Industrial-tier manufacture | SLM/DMLS metal additive (aerospace-grade Ti/Al powders) or 5-axis CNC production runs (~US$1k–10k+ NRE tooling, then per-part cost drops with volume) | Flight/production hardware; the tier where 27.05's warp/thermal simulation and support-generation genuinely matter |
| Analysis/validation compute | The same GPU class this project runs on (desktop RTX-class or Jetson-class for embedded design tools) for the topology-optimization and downstream FEA-validation software | Design-time compute, not runtime robot compute — this project's own reference machine (RTX 2080 SUPER) is representative |
| Post-optimization validation | Commercial unstructured-mesh FEA (Ansys, Abaqus, NX Nastran) or open alternatives (CalculiX, code_aster) | The independent structural check every optimized design needs before trust (§3) |

## 3. Installation & integration — putting it on a real robot

**This project's output never runs ON a robot — it produces a PART.** "Integration" here means how
that part enters a robot's bill of materials, not a runtime deployment:

- **From density field to CAD.** This project's raw output (`demo/out/topology_*.pgm`, a per-
  element density) is NOT a manufacturable geometry. The standard next step (this project does not
  perform it) is **CAD reinterpretation**: import the density field into a tool that extracts a
  smooth solid boundary (marching-squares/cubes-style isosurfacing at, e.g., `rho=0.5`, followed by
  manual or automated surface cleanup), fillet the reentrant corner discussed in §1, and add
  manufacturing features (bolt holes, chamfers, datum faces) the topology optimizer knows nothing
  about. Commercial tools (nTopology, OptiStruct's own post-processor) automate much of this;
  smaller teams often do it by hand in general CAD software.
- **Independent structural validation.** Before trusting a reinterpreted design, re-run FEA on the
  ACTUAL CAD geometry (not the coarse structured grid this project used to generate it) in an
  independent, unstructured-mesh solver (§2's validation tier) — this closes the loop between "the
  optimizer's simplified model says this is stiff" and "the real manufactured part, with its
  fillets and bolt holes, actually is." A real validation pass also checks load cases this project's
  single-case demo does not: fatigue (cyclic loading over the robot's service life), and — for a 2D
  plane-stress model like this project's — OUT-OF-PLANE buckling, which the compliance objective
  optimized here does not see at all (README "Limitations").
- **Where it lands on the robot's BOM.** Once validated, the design becomes an ordinary mechanical
  part: a CAD file, a manufacturing drawing (or print-ready file for additive), and a line item in
  the robot's bill of materials, procured and QC'd like any other structural component — no ROS 2
  node, no runtime software, no bus or protocol (this section is N/A for those items: this project's
  artifact is consumed entirely at design/manufacturing time, never at robot runtime).
- **The safe hardware-testing ladder, adapted for a structural part** (CLAUDE.md §1's ladder, in its
  mechanical-design form): (1) *simulation* — this project's FEA, plus the independent revalidation
  above; (2) *bench testing* — a coupon or a single printed/machined sample, statically loaded on an
  instrumented test rig to the design load (and beyond, to find the actual margin) BEFORE it goes
  anywhere near a robot; (3) *tethered/limited installation* — fit-check and a low-load functional
  test on the actual robot, current/torque-limited, with the same E-stop discipline every other
  project in this repo requires; (4) *full duty-cycle field use* — only after (2) and (3) pass with
  margin, and only for a use case where a structural failure's consequences have been assessed. This
  project's own outputs have been through NONE of these rungs — they are teaching artifacts.

## 4. Business & regulatory context

- **Who needs this capability.** Any robotics company shipping physical hardware needs SOME answer
  to "how do we make our links and brackets as light as possible without breaking" — legged-robot
  and drone companies most acutely (mass multiplies through the whole kinematic/power chain: a
  gram saved on a quadruped's shin is a gram the hip motor never has to accelerate, every stride,
  for the robot's whole service life), but also manipulator-arm and mobile-robot makers wherever
  payload capacity or reach is competitively constrained. This work sits inside **mechanical
  engineering** (SYSTEM_DESIGN §5.1's org map — mechanical owns domains 26/27/28/36), typically
  under a structural/mechanical-design engineer role, working closely with manufacturing engineering
  (who owns the DFM feedback this project's §1 discusses) once a design is finalized.
- **The players.** Commercial topology-optimization software (Altair OptiStruct, Dassault TOSCA/
  Simulia, nTopology, Ansys Mechanical's topology module — README "Prior art") is a mature,
  licensed-software market; open-source alternatives exist (this project's own SIMP lineage, plus
  frameworks like TopOpt.jl) but rarely match commercial tools' manufacturing-constraint integration
  and unstructured-mesh solvers. Build-vs-buy (SYSTEM_DESIGN §5.3): most robotics companies BUY the
  optimization tool (it is not their differentiator) and instead differentiate on WHAT they design
  with it and how tightly they close the loop to manufacturing — exactly the workflow §3 describes.
- **What getting it wrong costs.** An under-designed bracket that looks fine in a compliance-only
  simulation (this project's exact objective) but fails to a load case, a fatigue cycle count, or an
  out-of-plane buckling mode the simulation never modeled is a FIELD FAILURE — at minimum a costly
  recall/retrofit, at worst a safety incident if the bracket was load-bearing near a person or
  carried a payload that then falls. The mitigation is architectural, not aspirational: the §3
  validation ladder (independent unstructured FEA re-check, physical bench testing to and past
  design load) exists specifically because a topology optimizer's simplified model is a DESIGN AID,
  never a certification.
- **Regulatory.** Structural components on robots are not usually regulated in their own right, but
  they inherit the regulatory reality of the MACHINE they are part of (SYSTEM_DESIGN item 6's map):
  a bracket in an industrial-arm work cell sits inside that arm's **ISO 10218** / **ISO/TS 15066**
  safety case (structural integrity under the arm's rated payload and speed is exactly the kind of
  evidence such a case must include); cross-cutting every robot type, EU **machinery-directive**
  conformity and general workplace-safety law both expect a documented, traceable design-validation
  process — which is precisely why §3's independent re-validation step is not optional in a real
  product, only in this teaching demo. This project computes a compliance-minimizing shape
  didactically; it makes no certification claim of any kind (CLAUDE.md §1).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
