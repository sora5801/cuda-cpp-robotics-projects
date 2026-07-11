# 36.03 — Lattice-robot kinematics batches

**Difficulty:** [R&D] research · **Domain:** 36. Modular & Self-Reconfigurable Robots

> Catalog bullet (source of truth, verbatim): `Lattice-robot kinematics batches [R&D]`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

A **lattice robot** is a swarm of identical unit-cube modules that latch face-to-face on an integer
3D grid and reconfigure by sliding or pivoting, one module at a time, into new shapes — the
"self-reconfiguring modular robot" idea. This project builds a **GPU batch pipeline** over the
standard abstraction used to reason about such robots, the **sliding-cube model**: given
K = 4096 candidate 24-module configurations, four staged kernels compute, per configuration and
entirely in parallel, (1) whether any two modules overlap, (2) whether the face-adjacency graph is
connected, (3) which modules are **articulation points** (cut vertices — modules that can never move
without fracturing the robot), and (4) which of the remaining modules have a **legal sliding or
pivoting move** under the model's exact geometric preconditions. A tenth of the batch is
deliberately corrupted (duplicate positions, severed connectivity) as negative controls the
pipeline must catch with zero false alarms. A closing **vignette** takes one 24-module straight line
and greedily executes real, legality-checked moves that compact it toward a blob — the lattice-robot
story made concrete, while honestly stopping short of a full reconfiguration *planner* (that is
catalog project **36.01**, named throughout this project's docs, not reimplemented here).

**Scoping for this [R&D] catalog bullet (CLAUDE.md §2/§13):** the reduced-scope teaching core
implemented here is the full four-stage KINEMATICS pipeline (validity, connectivity, articulation,
move enumeration) plus a single greedy reconfiguration vignette. **Not** implemented, and explicitly
the documented research frontier (THEORY.md "Where this sits in the real world"): a real search-based
reconfiguration *planner* that finds a legal move SEQUENCE between two arbitrary target shapes
(project 36.01), and distributed/decentralized control where each module only has local information
(project 36.05). This project answers "is this shape legal, and what can each module do right now?" —
36.01 answers "how do I get from shape A to shape B?"

## What this computes & why the GPU helps

Four kernels, each K=4096-wide, one thread per **configuration** (not per module — see "System
context" and THEORY.md "The GPU mapping" for why):

- **Pattern:** batched small-graph analysis — every thread runs a *complete* graph algorithm
  (duplicate-key sort, BFS, Tarjan low-link DFS, or an O(kM·18) precondition sweep) over its own
  24-module graph, entirely in local arrays, with zero cross-thread communication. K independent
  small graphs in parallel is the same "K independent problems, one thread each" shape 08.01 uses
  for K independent ODE rollouts — here the "problem" is a whole graph algorithm, not an integration
  step.
- **Measured reality (RTX 2080 SUPER, Release):** all four stages over K=4096 configurations run in
  ~9.5–10 ms of GPU kernel time versus ~260–264 ms for the same four stages run sequentially on one
  CPU core — roughly 26–28x, a teaching artifact from one machine, not a benchmark claim.
- **All-integer, by design:** every quantity in this pipeline — lattice cells, adjacency, validity,
  connectivity, articulation flags, move legality — is an exact integer predicate. There is no
  rounding anywhere, so the GPU-vs-CPU verify gate below demands **bit-exact** agreement, not a
  tolerance — a deliberate contrast with 08.01/09.01's FP32 relative-tolerance gates (see
  [`THEORY.md`](THEORY.md) "Numerical considerations").

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** **planning support / world-model validity checking** — below the reconfiguration
  planner, above raw perception. It is the kinematic "legal-move oracle" a planner queries, analogous
  to how 07.09's distance fields feed a motion planner without being the planner itself.
- **Upstream inputs:** a candidate lattice configuration (module cell occupancy — the modular-robot
  analogue of `sensor_msgs/JointState`) and, ultimately, a **shape goal** from task planning (e.g. "form
  a bridge," "compact into a transport pose") that a reconfiguration planner would decompose into
  intermediate target shapes for this pipeline to validate.
- **Downstream consumers:** **project 36.01** ("Reconfiguration planning over enormous state spaces
  (GPU search)") — a real planner needs exactly this pipeline's outputs (legal moves, articulation
  constraints) as its per-state expansion oracle in a search over the astronomically large space of
  configurations; **project 36.05** ("Emergent distributed control experiments at scale") — a
  decentralized controller running independently on every physical module needs the SAME legality
  rules, evaluated locally rather than batched.
- **Rate / latency budget:** this is **planning-time analysis**, not a real-time control loop —
  legality checking for a whole batch of candidate configurations happens offline or between
  reconfiguration decisions (seconds of planning budget is typical in the research literature); actual
  module MOVES on physical hardware take on the order of **seconds each** (connector latch/release,
  actuator sweep — see [`PRACTICE.md`](PRACTICE.md) §1), nothing like the kHz control loops elsewhere
  in this repo. Stated honestly: there is no "30 Hz" or "1 kHz" number to give here, and inventing one
  would misrepresent the domain.
- **Reference robot(s):** none of SYSTEM_DESIGN.md's five reference robots is a lattice robot — this
  project's reference architecture is the **lattice-robot archetype** itself, a sixth family whose
  physical lineage is documented research hardware: **M-TRAN** (AIST, hinge-based lattice modules that
  also self-reconfigure), **Roombots** (EPFL, connector-and-rotational-joint modules for
  reconfigurable furniture/structures), and the broader crystalline/lattice-robot research line
  (chain- and lattice-type self-reconfiguring modular robots). These are cited as **documented research
  systems**, not products — see [`PRACTICE.md`](PRACTICE.md) §4 for the field's maturity level.
- **In production:** nothing — this is a research-only field (see "Limitations & honesty" and
  [`PRACTICE.md`](PRACTICE.md)). No lattice-reconfiguring robot ships as a commercial product today;
  the open problems (scalable planning, hardware reliability at module counts beyond dozens,
  self-repair) are named in [`THEORY.md`](THEORY.md) "Where this sits in the real world."
- **Owning team:** this capability lives entirely in **research** (academic robotics labs, and the
  handful of industrial research groups exploring modular robotics) — there is no "modular robotics
  product team" analogue elsewhere in this repo's SYSTEM_DESIGN.md item 5 org map to point to
  honestly; see [`PRACTICE.md`](PRACTICE.md) §4.

## The algorithm in brief

- **Seeded accretion batch generator** — grow each of K connected 24-module configurations by
  repeatedly attaching a new module to a random free face-neighbor of an already-placed module.
  → [THEORY.md](THEORY.md) "The algorithm".
- **Stage 1 — validity** — pack each module's cell into a sortable key, insertion-sort kM=24 keys,
  scan for adjacent duplicates. → THEORY §The algorithm.
- **Stage 2 — connectivity** — array-based BFS over the face-adjacency graph from module 0.
- **Stage 3 — articulation points** — Tarjan's DFS low-link algorithm (iterative, explicit stack),
  taught step by step. → THEORY §The algorithm.
- **Stage 4 — move enumeration** — the **sliding-cube model**'s two move families (linear SLIDE,
  edge-diagonal CORNER/pivot), each with an exact geometric precondition, diagrammed in full in
  THEORY §The math.
- **Reconfiguration vignette** — greedy steepest-descent on an all-integer compactness potential,
  re-verifying validity+connectivity after every candidate move. → THEORY §How we verify correctness.

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/lattice-robot-kinematics-batches.sln`](build/lattice-robot-kinematics-batches.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/lattice-robot-kinematics-batches.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none** — CUDA runtime + C++17 standard library only. No
cuBLAS/cuFFT/Thrust/etc. is linked: every stage is small, per-thread, integer graph work that a
library would not help with (see THEORY.md "The GPU mapping").

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) — including how
to read the three artifacts the demo writes.

## Data

The committed sample is a **scenario** (generator parameters), not recordings or pre-built
configurations — the same "scenario, not recordings" choice 08.01 makes for its cart-pole start
state: `data/sample/lattice_scenario.csv` (four rows: `K`, `SEED`, `CORRUPT_FRAC`,
`VIGNETTE_MAX_STEPS`; regenerated by `scripts/make_synthetic.py`, seed 42). `src/main.cu`'s seeded
accretion generator regenerates the full K=4096-configuration batch deterministically from `SEED`
every run — there is nothing else to download or cache. `scripts/download_data.ps1`/`.sh` are honest
no-ops (no public dataset applies to synthetic lattice configurations). Full field documentation,
provenance, and the sample's SHA-256 checksum: [`data/README.md`](data/README.md).

## Expected output

Twelve stable lines — banner, `PROBLEM:`, `SCENARIO:`, `VERIFY:`, `CORRUPTION-GATE:`,
`ARTICULATION-BRUTEFORCE:`, `MOVE-PRECONDITION-BRUTEFORCE:`, three `ARTIFACT:` lines, `VIGNETTE:`, and
`RESULT:` — checked as a subset diff by [`demo/expected_output.txt`](demo/expected_output.txt). Four
independent verification layers, all measured on the reference machine (RTX 2080 SUPER):

1. **GPU-vs-CPU, bit-exact (VERIFY):** all four stages, all K=4096 configurations, zero mismatches —
   `valid=0 connected=0 is_articulation=0 num_articulation=0 legal_move=0 move_count=0`. No tolerance:
   see "Numerical considerations" in [`THEORY.md`](THEORY.md).
2. **Injected-corruption detection (CORRUPTION-GATE):** 410/410 injected duplicate/disconnect
   configurations caught, 0/3686 false alarms on clean configurations.
3. **Brute-force cross-checks on a 128-configuration subset:** the Tarjan articulation result matches
   an independently-coded "remove each module, re-run connectivity" O(kM³) oracle with **0** module
   mismatches; the move-precondition result matches an independently-coded oracle with **0** mismatches
   out of 128×24×18 = 55,296 checked entries.
4. **The vignette (VIGNETTE):** a 24-module straight line, greedily reconfigured under real legality
   checks, reduces its compactness potential Phi from **1156 to 47** over **127** legal moves (49
   slide + 78 corner), converging to a local optimum with a final 5×3×3 bounding box (started
   24×1×1) — **every one of the 127 intermediate configurations independently re-verified
   valid+connected**.

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — the lattice geometry contract: position layout, the fixed
   move-direction numbering, and every stage's output layout, all defined once.
2. [`src/kernels.cu`](src/kernels.cu) — the four GPU stage kernels; start with `validity_kernel` (the
   simplest), then `connectivity_kernel`, then the big one, `articulation_kernel` (Tarjan low-link,
   iteratively), then `move_enum_kernel` (the sliding-cube preconditions, diagrammed in the header
   comment) — the single most interesting thing to read here.
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the four oracle twins PLUS the two
   independently-shaped brute-force cross-checkers (`articulation_bruteforce_cpu`,
   `move_precondition_bruteforce_cpu`) — a second kind of verification this repo's other flagships
   don't need, because at kM=24 a genuinely different algorithm is cheap enough to run for real.
4. [`src/main.cu`](src/main.cu) — the batch generator (seeded accretion + corruption), the staged
   pipeline driver, all three verify gates, and the vignette's greedy search loop.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers (copied, not shared — §4 rule).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **M-TRAN (AIST, Murata et al.)** — hinge-and-latch lattice-capable modular robots with real
  self-reconfiguration demonstrations; the physical lineage of the "connector + latch" hardware this
  project's abstract moves stand in for (see [`PRACTICE.md`](PRACTICE.md) §1–2).
- **Roombots (EPFL, Ijspeert/Mondada et al.)** — connector-and-rotational-joint modules built for
  reconfigurable furniture and structures; a second real hardware lineage for the same abstraction.
- **Crystalline / lattice-type self-reconfiguring robot literature** (the broader research area this
  project's sliding-cube model comes from) — study the move-legality and reconfiguration-algorithm
  papers in this space for the exact preconditions various groups use (this project documents its own
  self-consistent choice in [`THEORY.md`](THEORY.md) "The math", and is honest that conventions vary).
- **Modular robot reconfiguration complexity theory** (the "how hard is reconfiguration, in general"
  question) — the PSPACE-hardness results and known polynomial special cases this project's move
  legality feeds into; see [`THEORY.md`](THEORY.md) "Where this sits in the real world" for the exact
  framing and honest scoping.
- **Tarjan (1972), "Depth-First Search and Linear Graph Algorithms"** — the DFS low-link algorithm
  Stage 3 implements; the original source for the articulation-point / biconnectivity technique this
  project teaches from first principles.
- **cuGraph / NetworkX** — production graph libraries with articulation-point and connectivity
  primitives at MUCH larger scale (millions of nodes, ONE graph); contrast their "one huge graph,
  parallelize the algorithm" regime with this project's "many tiny graphs, parallelize the batch"
  regime (THEORY.md "The GPU mapping" makes the contrast explicit).

## Exercises

1. **Plot the artifacts:** `demo/out/batch_stats.csv` → histogram `num_legal_moves` and
   `num_articulation` across the clean configurations. `demo/out/vignette_frames.csv` → animate the
   24 module positions frame by frame and watch the line fold into a blob.
2. **View the render:** open `demo/out/config_render.pgm` in any image viewer that reads PGM (or a
   text editor — it's ASCII) and compare it against the bounding-box numbers in `[info]` output.
3. **Break the corruption gate on purpose:** set `CORRUPT_FRAC` to 0.0 in
   `data/sample/lattice_scenario.csv`, rerun, and confirm the gate still reports 0 false alarms with
   nothing to catch — then set it to 0.9 and watch the batch skew toward corrupted configurations.
4. **Change the vignette's target:** replace the fixed centroid reference with a moving target (e.g.
   the running centroid) and observe whether the greedy converges faster, slower, or to a different
   final shape — document what changed and why in your own notes.
5. **Climb toward 36.01:** replace the vignette's single-step greedy search with a beam search (keep
   the top-B candidate configurations at each step instead of only the best one) and measure whether
   it escapes the local optimum this project's plain greedy gets stuck in (Phi=47, not 0).

## Limitations & honesty

- **Reduced-scope [R&D] teaching version (CLAUDE.md §2/§13):** this project implements the
  KINEMATICS layer (validity, connectivity, articulation, move legality) and ONE greedy vignette —
  not a reconfiguration planner. Finding a legal move sequence between two arbitrary target shapes is
  the documented research step (project **36.01**); this project deliberately stops at "what is legal
  right now," which is 36.01's per-state building block, not its replacement.
- **The sliding-cube model's exact preconditions are THIS project's self-consistent choice, not a
  single universally agreed standard** — the literature's various formalizations differ in detail
  (gravity-biased vs. isotropic, single-cell vs. multi-cell wall support). [`THEORY.md`](THEORY.md)
  documents the physical reasoning behind every precondition used here (including a mid-development
  correction: an earlier single-support-cube slide rule was replaced with the physically correct
  2-module wall requirement once the geometry was checked carefully — a live example of the
  "no black boxes, verify everything" principle this repo asks for).
- **The greedy vignette is not a planner and is not guaranteed to reach a specific target shape** — it
  performs steepest-descent on a compactness potential and honestly reports getting stuck in a local
  optimum (Phi 1156 → 47, not → 0) rather than claiming a perfect result it did not achieve.
- **Isotropic, not gravity-biased:** this teaching version does not model gravity or a "floor" — a
  scoping choice discussed in THEORY.md, common in the theoretical reconfiguration-complexity
  literature but a simplification relative to most gravity-affected physical hardware demonstrations.
- **kM=24 modules, K=4096 configurations are teaching-scale, chosen for the in-thread graph-algorithm
  argument to hold cleanly (THEORY.md "The GPU mapping")** — real research hardware fleets to date are
  of a similar order (tens of modules); nothing about the algorithms here assumes this scale, but
  larger M would eventually favor a different (multi-thread-per-graph) GPU mapping, documented as
  future work, not implemented.
- **Sim-only, not safety-certified (CLAUDE.md §1):** nothing here commands real hardware; this
  project's output is a data structure (legality flags, move lists), never a physical actuation
  command — see [`PRACTICE.md`](PRACTICE.md) §3 for the honest gap between this pipeline and any real
  module's actuator bring-up.
- **Timings are teaching artifacts** — single-shot, one machine (RTX 2080 SUPER), never a benchmark
  claim.
