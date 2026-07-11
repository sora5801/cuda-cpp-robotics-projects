# Push note — 2026-07-10-20: flagship 36.03 lattice robots — Phase-1 flagships complete

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Flagship 36.03 — **lattice-robot kinematics batches** — is done, closing **batch 1h** (32.02,
34.03, 35.01, 36.03) and with it **all 36 Phase-1 flagships** (**36/505**): every domain in the
catalog now has one fully-verified, best-in-class study project. 36.03 is an [R&D] reduced-scope
teaching version of modular-robot analysis built on the sliding-cube abstraction: K=4,096
configurations of M=24 modules run a four-stage GPU pipeline — validity (duplicate-cell
detection), connectivity (in-thread BFS), articulation modules (iterative Tarjan low-link, taught
as the classic it is), and legal-move enumeration under the exact slide/corner preconditions —
one *thread per configuration*, everything all-integer, so GPU-vs-CPU verification is
**bit-exact, no tolerance**. The build's best moment was a physics catch: the builder's first
slide-precondition rule (single support cube) was proven impossible by a parity argument — no
lattice cell is face-adjacent to two mutually-adjacent cells — and corrected to the published
two-cube-wall rule before any documentation shipped. Verification goes a layer beyond the repo's
usual twin: two **independent brute-force oracles** (remove-and-recheck articulation; an explicit
precondition table) cross-check the fast algorithms exactly, and 410 deliberately-corrupted
configurations are all caught with zero false alarms. A reconfiguration vignette makes the domain
concrete: a 24-module line greedily compacts into a 5×3×3 block over 127 verified-legal moves.

## What changed

- **[projects/36-modular-reconfigurable/36.03-lattice-robot-kinematics-batches/](../projects/36-modular-reconfigurable/36.03-lattice-robot-kinematics-batches/)** —
  complete: batch generator (seeded accretion + documented corruption injectors), four-stage
  config-parallel GPU pipeline, CPU twin, two brute-force oracles, corruption gate, greedy
  compaction vignette, batch-stats CSV + vignette-frames CSV + configuration PGM artifacts,
  full README / THEORY / PRACTICE.
- **[docs/STATUS.md](../docs/STATUS.md)** — 36.03 → `done` (**36/505; flagships 36/36**).

## New projects (didactic blurbs)

**36.03 — Lattice-robot kinematics batches** ([R&D], domain 36, flagship). What "kinematics"
means when the robot is a *set* of cubes: not joint angles but occupancy, adjacency, and legal
moves. The project teaches the sliding-cube model with its exact preconditions (ASCII diagrams
per move type), why articulation points are the modules that must not move (cut-vertex theory via
Tarjan low-link, in-thread on a 24-node graph — the small-graph-per-thread regime argued
honestly), and how the mechatronic reality of real lattice hardware (M-TRAN, Roombots lineage —
latching connectors, alignment funnels, power through faces) makes the abstraction generous. Full
reconfiguration *planning* is the documented research frontier (36.01 named); PSPACE-hardness of
the general problem is stated with the known polynomial special cases. The single most
interesting thing to look at: `demo/out/vignette_frames.csv` — a line of 24 cubes folding itself
into a block, one verified move at a time.

## How to build & run

```powershell
projects\36-modular-reconfigurable\36.03-lattice-robot-kinematics-batches\demo\run_demo.ps1
# then read demo\out\batch_stats.csv and animate demo\out\vignette_frames.csv
```

## What to study here

`THEORY.md` §The problem (what real lattice modules must physically contain — the mechatronics
each abstract "slide" hides) → the move preconditions with their diagrams → the articulation-point
algorithm → `src/kernels.cu` (one thread per configuration; why in-thread graph algorithms beat
parallel-BFS at M=24). First exercise: change the greedy potential's reference point and watch the
vignette converge to a different local optimum — then read why that is exactly the argument for
real planning (36.01).

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-10), re-run independently by the lead after the builder's self-gate:

- `Release|x64` **and** `Debug|x64` rebuild from clean with **zero errors, zero warnings**.
- `demo/run_demo.ps1` passes end to end: all 12 stable lines matched, exit 0.
- **Twin gate:** GPU matches CPU **bit-exact** across all 4 stages and all 6 output arrays,
  K=4,096 (all-integer — no tolerance needed, and the docs say why).
- **Negative controls:** 410/410 injected corruptions (205 duplicate + 205 disconnect) caught;
  0/3,686 false alarms on clean configurations.
- **Brute-force oracles:** articulation 128/128 subset configs, 0 module mismatches (3,072
  checks); move preconditions 128/128, 0 mismatches out of 55,296 entries.
- **Vignette:** 127 legal moves (49 slide + 78 corner), potential Φ 1156 → 47, bounding box
  24×1×1 → 5×3×3, every intermediate state re-verified valid+connected.
- Timing (teaching artifact): GPU 4-stage pipeline ≈ 8.6 ms vs CPU ≈ 253 ms (~29×).
- `tools/verify_project.py`: **all structural gates PASS**; comment density 0.58; no changes
  outside the project folder.

## Known limitations / TODOs

- [R&D] reduced scope stated in README §13: sliding-cube abstraction (real hardware documented,
  not simulated), analysis + greedy vignette only — a true reconfiguration planner is 36.01's
  documented research step; fixed M=24; the greedy is honestly framed as a hill-climber that
  stops at local optima.

## Next push preview

The §11 standards retrospective (its own push, already prepared): template-wide LNK4099 fix,
`util/paths.h` ratifying 12.01's path resolver, the twin-vs-shared verification ruling from the
13.03 case study, and BUILD_GUIDE troubleshooting additions. Then Phase 2 opens: the remaining
469 projects, domain by domain, easiest-first.
