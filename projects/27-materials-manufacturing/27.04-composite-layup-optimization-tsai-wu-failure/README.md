# 27.04 — Composite layup optimization + Tsai-Wu failure envelope sweeps

**Difficulty:** intermediate · **Domain:** 27. Materials Science & Manufacturing

> Catalog bullet (source of truth, verbatim): `Composite layup optimization + Tsai-Wu failure envelope sweeps`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

A fiber-composite part is not one material — it is a **stack of thin, directional sheets**, and the
engineer's real design variable is *which angle goes where*. This project builds the two classical
tools that answer that question on a GPU: **classical laminate theory (CLT)**, which turns a stack
of plies into one effective 2-D stiffness and back into per-ply stress under an applied in-plane
load, and the **Tsai-Wu failure criterion**, which turns each ply's stress state into a single
"how close to failure" number. The GPU content is a **design-space sweep**: one thread scores one
(candidate 8-ply stacking sequence, load case) pair, enumerating all 256 symmetric layups buildable
from the angle alphabet `{0°, +45°, -45°, 90°}` against two different load-case sets, and a second
sweep maps the full `(Nx, Ny)` **failure envelope** — the boundary in load space where a layup first
fails — as a field, for the winning layup and a standard `[0/90/0/90]s` cross-ply baseline. The demo
reproduces the textbook result: a quasi-isotropic-like stack (one ply of each angle) wins when the
load direction is unknown, while a 0-degree-heavy stack wins when the load is known to be aligned —
and it is a real, deterministic, closed-form-checkable computation the whole way through, not a
qualitative claim. Everything (material, strengths, load cases) is implemented; nothing about this
bullet is left documented-only.

## What this computes & why the GPU helps

Per layup-sweep launch: `kNLayups=256` candidate stacks × up to 16 load cases, each evaluation a
small (3×3 linear solve + 8 ply-level Tsai-Wu quadratics) closed-form computation — no iteration, no
shared state between problems. Per envelope launch: 128×128 = 16,384 independent grid points, same
per-point cost.

- **Pattern:** batched **sampling over a design space** — one thread = one independent (layup, load
  case) or (Nx, Ny) evaluation; the same "thread = one small linear-algebra problem" shape as
  33.01's batched small-matrix library, applied here to a materials-engineering search instead of a
  robotics kinematics chain.
- **Why the GPU helps:** a *design* sweep like this is the textbook case for embarrassing
  parallelism — every candidate is scored independently, so 4,608 (layup×case) + 32,768
  (2 envelopes × 16,384 points) evaluations run as one flat grid of threads instead of a host-side
  loop. Measured on the reference machine: the entire ranking sweep completes in ~1 ms of GPU kernel
  time, and both envelopes in a further ~0.1–0.5 ms — a scan of hundreds of candidate designs that
  never needs to leave the initial "explore many options" phase of a real design study.
- **The linear algebra is hand-rolled, not library-dispatched:** every solve is a fully-unrolled 3×3
  Cramer's-rule evaluation in registers (`src/kernels.cuh`'s `solve3x3_sym`) — deliberately, per
  CLAUDE.md §5: a 3×3 system is exactly the scale where hand-rolling teaches more than a cuSOLVER
  call would.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** **cross-cutting infrastructure** — specifically the *mechanical design*
  discipline that determines the physical structure everything else in the stack is bolted to. It
  runs entirely at **design time**, offline, long before any sensor/planner/controller loop exists;
  it never appears in a robot's runtime data flow at all.
- **Upstream inputs:** load requirements from a structural design study — e.g. **26.01's topology
  optimization** hands this project the exact loads (bending moment, in-plane force) its optimized
  bracket geometry must carry at its mounting flange; those loads become this project's `Nx/Ny/Nxy`
  load cases (message shape: a design-load table, the mechanical-engineering equivalent of a
  `geometry_msgs/Wrench` sampled over the part's operating envelope).
- **Downstream consumers:** the manufacturing process that lays up and cures the winning stack
  (**27.05**-class printing/layup process projects; PRACTICE §1 walks the physical process), and the
  QA/coupon-testing pipeline that verifies the built part matches this analysis (PRACTICE §3).
- **Rate / latency budget:** honestly, **none in the runtime sense** — this is a design-time
  calculation, run once (or a handful of times) per part revision, not a per-tick robot computation.
  The "budget" that matters is an engineer's iteration loop: this sweep's ~1 ms GPU time means a
  256-layup, two-load-case-set, two-envelope study is fast enough to run **interactively** while
  changing material or load assumptions — the actual value the GPU adds here (SYSTEM_DESIGN item 1's
  "design/offline" band, at the opposite end from control's kHz loops).
- **Reference robot(s):** the **drone airframe** (composite spars/skins are *the* standard
  lightweight-structure material for multirotor arms and fixed-wing spars) and the **manipulator
  work cell's arm links** (a lightweight composite forearm link reduces reflected inertia at the
  actuator, the same reason legged-robot limb designers reach for composites) — both from
  SYSTEM_DESIGN's five reference robots, wherever "make this structural member as light and stiff as
  possible for a known load" is the design question.
- **In production:** hand calculations and spreadsheets for simple parts; for anything load-bearing
  and certified, dedicated laminate tools (see Prior art below) replace this teaching sweep, and
  physical coupon testing (PRACTICE §3) always has the final word over any analysis, this one included.
- **Owning team:** mechanical/structures engineering, specifically a **materials & structures** or
  **composites** specialist role — adjacent to the topology-optimization/FEA team (26.x) upstream and
  manufacturing engineering (27.x) downstream (SYSTEM_DESIGN item 5).

## The algorithm in brief

- **Classical laminate theory (CLT)** — per-ply stiffness `Q` from engineering constants, rotated to
  `Qbar(theta)`, summed (thickness-weighted) into the laminate's extensional stiffness matrix `A`;
  for a **symmetric** stack `B=0` exactly, so a pure in-plane load gives one uniform midplane strain
  solved from a 3×3 linear system. → [THEORY.md §the-math](THEORY.md#the-math)
- **In-register 3×3 Cramer's-rule solve** (`solve3x3_sym`) — the 33.01-style small fixed-size linear
  solve, specialized to the always-symmetric CLT `A` matrix. → [THEORY.md §the-algorithm](THEORY.md#the-algorithm)
- **Per-ply stress recovery** — the shared midplane strain transformed into each ply's own material
  axes, then Hooke's law with the *unrotated* `Q`. → [THEORY.md §the-math](THEORY.md#the-math)
- **Tsai-Wu first-ply failure** — the quadratic interactive failure criterion (with the standard
  `-1/2` `F12` normalization), solved in **closed form** for the load-scaling factor at which each
  ply reaches its failure surface; the laminate's factor is the minimum over its plies.
  → [THEORY.md §the-math](THEORY.md#the-math)
- **The two sweeps** — one thread per (layup, load case) ranks all 256 stacking sequences; one
  thread per `(Nx,Ny)` grid point maps the full failure envelope as a field.
  → [THEORY.md §the-GPU-mapping](THEORY.md#the-gpu-mapping)

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/composite-layup-optimization-tsai-wu-failure.sln`](build/composite-layup-optimization-tsai-wu-failure.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/composite-layup-optimization-tsai-wu-failure.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none** — CUDA runtime + C++17 standard library only.
Every linear solve is a hand-rolled 3×3 Cramer's rule (no cuSOLVER, cuBLAS, or any other CUDA library
is linked — see `CMakeLists.txt`'s comment and `THEORY.md` §the-GPU-mapping for why this problem size
does not call for one).

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) — including what
each artifact shows and how to read the envelope PGMs.

## Data

The committed sample, `data/sample/laminate_scenario.csv` (~1.6 KiB), is a **task definition**, not a
recording: a synthetic carbon/epoxy-class lamina's elastic constants and Tsai-Wu strengths (labeled
synthetic, in the ballpark of a real aerospace tape but not sourced from any datasheet), the 4-angle
stacking alphabet, and the two generated load-case sets. `scripts/make_synthetic.py` regenerates it
deterministically (no seed needed — every computation here is a pure function of its inputs, CLAUDE.md
§8). No public dataset applies (this is a design-parameter set, not measured data);
`scripts/download_data.ps1` is an honest no-op. Full field-by-field documentation, checksum, and the
load-case generation formula: [`data/README.md`](data/README.md).

## Expected output

Fourteen stable lines — banner, `PROBLEM:`, `SCENARIO:`, `VERIFY:`, four `GATE_*:` lines, five
`ARTIFACT:` lines, and `RESULT:` — checked as a subset diff by
[`demo/expected_output.txt`](demo/expected_output.txt). Two kinds of verification, run every time:

1. **The §5 GPU-vs-CPU gate (`VERIFY:`):** the GPU sweep+envelope results are recomputed **in full**
   (every one of the 4,608 sweep points and 32,768 envelope points, not a spot-checked subset — this
   project's problem sizes are small enough to afford it) by the CPU oracle and must agree within
   relative tolerance `1e-3` (measured worst case: `1.76e-6` — three orders of magnitude of margin;
   THEORY.md explains why this project's tolerance can be so much tighter than a project with a
   chained-timestep integrator).
2. **Four analytic gates**, each checked against an independently-derivable closed-form or physical
   prediction rather than against the sweep's own output:
   `GATE_SINGLE_PLY_CLOSED_FORM` (a single 0° ply under pure `Nx`/`-Nx` must fail at exactly `Xt·t` /
   `Xc·t` — measured relative error `~1e-7`), `GATE_ISOTROPIC_ENVELOPE` (an isotropic-degenerate
   material's failure strength must not depend on load orientation — measured spread `~1e-7`),
   `GATE_CLT_SANITY` (the `[0/90/0/90]s` baseline's `A11=A22` and `Qbar(0)=Q` exactly), and
   `GATE_LOAD_HOMOGENEITY` (failure factor must scale exactly inversely with load magnitude).

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — the project's real content: every CLT/Tsai-Wu physics
   function (`lamina_Q`, `transform_Qbar`, `solve3x3_sym`, `assemble_A`, `ply_stress`, `tsaiwu_F`/
   `tsaiwu_ab`, `solve_lambda`, `laminate_failure_factor`), the `Lamina`/`LoadCase`/stack-encoding
   contracts, and every unit convention — read this first and most closely.
2. [`src/kernels.cu`](src/kernels.cu) — deliberately thin: `layup_sweep_kernel` and `envelope_kernel`
   each just decode a thread index and call the shared physics from (1).
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the full-array CPU oracle; note it calls the
   exact same `kernels.cuh` functions the GPU kernels do (no re-implementation to drift).
4. [`src/main.cu`](src/main.cu) — orchestration: load the scenario, run both sweeps, rank layups,
   run both envelopes, verify, run the four analytic gates, write five artifacts. The single most
   interesting thing to look at: the `GATE_ISOTROPIC_ENVELOPE` block's comment, which documents a
   real physics subtlety this project's own development process caught (why sweeping a load
   *direction* with `Nxy=0` does not test what it looks like it tests — THEORY.md §the-math tells the
   full story).
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers (copied, not shared — §4 rule).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **Jones, *Mechanics of Composite Materials*** — the standard CLT/Tsai-Wu textbook; every formula in
  `THEORY.md` traces back to this book's notation.
- **Tsai & Wu (1971), "A General Theory of Strength for Anisotropic Materials"** — the original
  failure-criterion paper; the `-1/2` `F12` normalization this project uses is Tsai & Hahn's later
  practical recommendation, discussed honestly in THEORY.md alongside its known controversy.
- **HyperSizer** — the commercial structural-sizing tool this project's "sweep many layups, rank by
  margin" pattern most directly resembles, at production scale (thousands of load cases, many failure
  modes, manufacturing constraints).
- **ANSYS Composite PrepPost (ACP)** / **Altair Hyperlaminate** — FEA-integrated laminate tools; where
  this project solves one uniform-strain membrane problem, these couple CLT to a full finite-element
  mesh (varying strain field, real boundary conditions).
- **classical hand calculations** (e.g. NASA/military handbook methods, MIL-HDBK-17) — the
  spreadsheet-and-slide-rule ancestor of every tool above; this project *is* essentially that
  calculation, done 256×16 times per second instead of once by hand.

## Exercises

1. **Plot the envelope:** load `demo/out/envelope_best_contour.csv` and `envelope_cross_contour.csv`
   into any plotting tool as a scatter — compare the two boundary shapes directly, and compare against
   `envelope_best.pgm`/`envelope_cross.pgm` opened in an image viewer.
2. **Break the isotropy gate on purpose:** in `main.cu`'s `GATE_ISOTROPIC_ENVELOPE` block, change
   `iso.S12_pa` to any value other than `kIsoF0_Pa/sqrt(3)` and rerun — watch the spread stop being
   near-zero, and connect the measured spread back to the `F66 = 3*F11` derivation in THEORY.md.
3. **Widen the alphabet:** add `+-30`/`+-60` to `ANGLE_ALPHABET_DEG` in `scripts/make_synthetic.py`
   (note: `kNAngleAlphabet` in `kernels.cuh` must match the alphabet size, and `kNLayups` must be
   updated to match — both are compile-time constants, documented in kernels.cuh's file header)
   and see whether a different quasi-isotropic-like combination wins the `MIXED` set.
4. **Add a last-ply-failure or progressive-degradation mode:** THEORY.md §where-this-sits-in-the-real-world
   describes what production tools add beyond first-ply-failure — implement a simple "degrade a failed
   ply's transverse/shear stiffness to near-zero and re-solve" loop as a second failure-load metric.
5. **Sweep a third case set:** add a `SHEAR`-dominated load-case set (large `Nxy`, small `Nx/Ny`) to
   `make_synthetic.py` and see which layup wins — predict the answer from `THEORY.md` before running.

## Limitations & honesty

- **First-ply failure only, no progressive damage.** Real laminates often carry meaningfully more
  load after their first ply fails (matrix cracking redistributes load to intact plies) — production
  tools model this with progressive ply-discount schemes; this project stops at the conservative,
  standard "first ply reaches its Tsai-Wu surface" criterion (Exercise 4 sketches the extension).
- **Stacking sequence is enumerated but does not affect the score.** As designed, `A_ij` depends only
  on the *multiset* of ply angles, never their order — so many of the 256 enumerated sequences tie
  exactly (measured: 24 of 256 for the `MIXED` winner, exactly `4!` permutations of one-of-each-angle).
  This is an honest CLT fact, not a bug (`kernels.cuh`'s file header and `THEORY.md` explain it) — a
  full design study would also weigh stacking-sequence-dependent effects (delamination resistance,
  thermal warpage, impact tolerance) this project's pure-membrane scope does not reach.
- **No bending, no out-of-plane loads.** Every load case here is a pure in-plane membrane load
  (`Nx, Ny, Nxy`); the laminate's bending stiffness `D` is never assembled or used.
- **Static strengths only, no environmental knockdown.** Real strengths degrade with temperature,
  moisture, and fatigue cycling — production allowables apply large "knockdown factors" this teaching
  material does not model (PRACTICE §2 discusses the real-hardware version honestly).
- **Synthetic material, not a datasheet.** `data/README.md` states this plainly; treat the exact
  numbers as illustrative, never as design values for a real part.
- **Timings are teaching artifacts** — single-shot, one machine, kernel-only where labeled.
- **No hardware-motion caveat needed here:** this project's output never commands a robot directly —
  it is an offline structural design calculation (System context above); the repo-wide sim-validated-
  only caveat (CLAUDE.md §1) is not applicable in the "moving hardware" sense, though PRACTICE §3's
  testing ladder (coupon → sub-component → full part) is exactly the analogous discipline for
  *trusting a structural analysis* before it flies.
