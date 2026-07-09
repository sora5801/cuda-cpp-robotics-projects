# Push note — 2026-07-08-03: flagship 07.09 jump flooding

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Foundation flagship #3: **07.09 jump-flooding Voronoi/distance transforms** — the repository's
first **grid/stencil-pattern** project (2-D blocks, ping-pong double buffering) and its first with
a **visual artifact** (the demo writes the Voronoi regions and the clearance field as viewable PGM
images). It is also the first project whose oracle verifies an *algorithm's promise* rather than a
ported computation: the GPU runs approximate JFA, the CPU runs the exact brute-force scan, and the
demo checks documented approximation bounds every run. That check earned its keep immediately —
see Verification.

## What changed

- **[projects/07-collision-geometry/07.09-jump-flooding-voronoi-distance-transforms/](../projects/07-collision-geometry/07.09-jump-flooding-voronoi-distance-transforms/)** —
  complete: JFA kernels + 1+JFA launcher
  ([`src/kernels.cu`](../projects/07-collision-geometry/07.09-jump-flooding-voronoi-distance-transforms/src/kernels.cu)),
  exact CPU oracle ([`src/reference_cpu.cpp`](../projects/07-collision-geometry/07.09-jump-flooding-voronoi-distance-transforms/src/reference_cpu.cpp)),
  two-stage driver with bounds-check comparator + PGM artifact writer
  ([`src/main.cu`](../projects/07-collision-geometry/07.09-jump-flooding-voronoi-distance-transforms/src/main.cu)),
  synthetic seed sample (seed 42) + generator, full README / THEORY / PRACTICE, all markers resolved.
- **[docs/STATUS.md](../docs/STATUS.md)** — 07.09 → `done` (**3/505**).

## New projects (didactic blurbs)

**07.09 — Jump-flooding Voronoi/distance transforms** (★ beginner, domain 07, flagship). Teaches
the grid/stencil GPU pattern: one thread per cell, 16×16 blocks, neighbors gathered at
exponentially-shrinking offsets, ping-pong buffers for pass consistency. Robotics reading: seeds
are obstacle cells; the output distance field is the clearance map costmaps/planners/safety
monitors consume, and the labels are Voronoi regions (free-space skeletons, coverage partitions).
Deliberately contrasts with 09.01 on memory spaces: uniform reads → constant memory (09.01);
divergent reads → global memory (this project). The single most interesting thing to look at: the
**1+JFA priming pass** in `src/kernels.cu` and the comment recording why it exists.

## How to build & run

```powershell
projects\07-collision-geometry\07.09-jump-flooding-voronoi-distance-transforms\demo\run_demo.ps1
# then open demo\out\voronoi.pgm and demo\out\distance.pgm in any image viewer
```

## What to study here

Project `README.md` → `THEORY.md` §The problem (why clearance fields are what safety arguments are
written in) → `src/kernels.cu` (the pass kernel + ping-pong launcher) → `src/main.cu`'s comparator
(tie handling, bounds). Exercises to try first: draw 512 seeds and look at the images (Exercise 1);
implement the brute-force GPU kernel and find the crossover (Exercise 2).

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-08):

- `Release|x64` **and** `Debug|x64` build with **zero errors, zero warnings**.
- `demo/run_demo.ps1` passes end to end: all 8 stable lines matched, exit 0; both PGM artifacts
  written. Deterministic (integer-exact) accuracy metrics: 512²/64-seed sample **0 mismatches vs
  the exact field**; 1024²/128-seed batch **4/1,048,576 cells (0.0004%)** mismatched, max distance
  error **0.88 cells** — inside the ≤0.5% / ≤2-cell bounds.
- Timing (single-shot teaching artifact): exact CPU scan ≈ 155 ms vs GPU JFA end-to-end ≈ 3–6 ms
  (all passes + internal alloc/copies) on the 1024² batch.
- `tools/verify_project.py`: **all structural gates PASS**.
- **The oracle caught a real defect during development:** plain JFA passed the 512² sample
  perfectly but exceeded the bounds on the 1024² batch (13 mislabeled cells, max error 3.9 cells
  > 2.0). Upgraded to the published **1+JFA** variant (one extra step-1 priming pass), which
  brought it to 4 cells / 0.88. Recorded in the kernel comments and THEORY.md §verification as a
  worked example of oracle-first development.

## Known limitations / TODOs

- JFA remains approximate (documented and bounds-checked every run); exact-field consumers should
  use separable exact transforms (README Exercise 5). 2-D, point seeds, Euclidean metric only.
- Worker dispatch still blocked by the session limit; the lead continues building inline.

## Next push preview

Foundation flagship #4 closes the set: 08.01 MPPI — the canonical GPU controller (cart-pole
teaching core), bringing cuRAND-free deterministic rollouts, softmin weighting, and the repo's
first control-loop demo.
