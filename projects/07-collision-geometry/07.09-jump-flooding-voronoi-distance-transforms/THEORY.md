# 07.09 — Jump-flooding Voronoi/distance transforms (easy, visual, useful): Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2).

## The problem — physics & engineering first

This project is **purely computational** — so, per the repo contract, we teach the physics of its
nearest physical carrier: the **occupancy grid** and the clearance question a moving robot asks of
it.

A mobile robot's world model at the planning layer is a grid of cells marked occupied/free, built
from range sensors. The physics enters through *how those cells get marked*: a LiDAR return says
"solid surface at this range ± noise" (centimeter-class σ for time-of-flight), a depth camera
similar with worse tails, sonar with wide cones — so an "obstacle cell" is really "evidence
crossed a threshold here" (occupancy mapping is domain 05's business; we inherit its output). The
robot's own physics then turns geometry into urgency: a base moving at `v` with maximum
deceleration `a` needs `v²/2a` meters of clearance to stop — at 1.5 m/s and 1 m/s², that is
1.1 m plus the robot's own footprint radius. **Distance-to-nearest-obstacle is therefore not a
convenience; it is the quantity safety arguments are written in** (speed-and-separation monitoring,
21.04, is literally a rulebook over this field). Engineering constraints: costmaps refresh at
5–20 Hz over grids of 10⁵–10⁷ cells (a 50 m × 50 m map at 5 cm/cell is a million cells), and the
field must be *dense* — every cell a planner might sample needs an answer, which is what makes
this a grid problem rather than a per-query one.

The **Voronoi labels** (which obstacle is nearest, not just how far) carry their own robotics
meaning: cells equidistant from two obstacles form the **generalized Voronoi diagram** — the
maximally-clear "skeleton" of free space that topological navigation follows (05.16), and the
partition structure coverage/multi-robot allocation builds on (22.04).

Units and frames: the grid is row-major, x rightward, y downward (image convention, stated in
[`src/kernels.cuh`](src/kernels.cuh)); one cell = one unit here, and a consumer multiplies by its
map resolution (m/cell) to get meters — the demo stays unitless on purpose so the scaling step is
conscious, not hidden.

## The math

**Problem.** Given seed set `S = {s₁…s_N} ⊂ G` on grid `G` of W×H cells, compute for every cell
`p`: the **distance transform** `D(p) = min_i ‖p − s_i‖` and the **Voronoi label**
`L(p) = argmin_i ‖p − s_i‖` (ties: any minimizer is valid — a genuinely ambiguous cell).

**Metric.** Euclidean on integer coordinates; all comparisons use **integer squared distances**
`d²(p,s) = (px−sx)² + (py−sy)²` (exact in 64-bit for any real grid), with `√` applied only for
display and error reporting. Comparing squared distances is order-preserving because `√` is
monotone — the standard trick that keeps the whole pipeline in exact arithmetic.

**Costs.** Exact scan: `O(W·H·N)`. Exact separable algorithms (Felzenszwalb–Huttenlocher) reach
`O(W·H)` for distance-only. JFA: `(log₂P + 2)` passes of `O(W·H)` gathers (P = grid size padded to
a power of two; +2 = the 1+JFA priming pass and the step-1 finish), **independent of N** — the
property that makes it attractive when obstacles are dense.

**Why JFA can err.** JFA maintains, per cell, one *hypothesis* (the best seed seen so far) and
propagates hypotheses through a fixed 9-sample funnel per pass. A cell `p` ends correct if some
sample chain carries its true seed to it; the classic failure is a seed whose region is *thin* along
the sampling directions — the hypothesis gets overwritten before the last hop. Errors concentrate
on region boundaries, mislabeling to the *second-nearest* seed, so the distance error is the gap
between best and second-best — small precisely where mislabeling is possible (boundary cells have
near-equal distances). That intuition is what our bounds check quantifies.

## The algorithm

1. **Init:** every cell ← sentinel "no seed"; each seed claims its own cell (scatter).
2. **1+JFA priming pass (step = 1):** each cell gathers from its 8 immediate neighbors — every
   seed's one-ring now knows about it before information starts teleporting.
3. **The halving schedule:** for step = P/2, P/4, …, 1: every cell examines itself + 8 neighbors at
   offsets `{−step, 0, +step}²` and keeps the candidate seed with smallest squared distance **to
   itself** (strictly smaller — ties keep the incumbent).
4. **Read out:** each cell's stored seed coords give the label (id) and distance.

Every pass is a pure **gather** into a fresh buffer (ping-pong): within a pass, all cells read the
same consistent snapshot. Complexity: `O(W·H·log P)` work, `O(log P)` sequential passes (the span);
brute force has more work but *zero* pass-to-pass dependencies — a genuine trade, not a free lunch.

## The GPU mapping

```
one thread = one CELL              (grid/stencil pattern — new vs 33.01/09.01)
block = 16×16 tile                 (2-D block ⇔ 2-D data; threadIdx.x along
grid  = ⌈W/16⌉ × ⌈H/16⌉             the row = the fast axis = coalescing)
per pass:  read `in` snapshot  →  write `out`  →  swap pointers
```

- **Why 2-D blocks now:** the neighbors are 2-D offsets; a square tile makes the index arithmetic
  read like grid coordinates. The repo's 256-thread default just changes shape (16×16).
  Swapping which index is the fast axis is *the* classic coalescing bug in grid kernels — called
  out in the kernel comment.
- **Ping-pong double buffering** is the load-bearing correctness idea: updating in place would let
  a cell read a *this-pass* value from a neighbor (a stale/fresh race in the read-modify-write
  sense). Two buffers + pointer swap per pass costs one extra grid of memory and zero copies
  (the final device-to-device copy only fixes up buffer parity for the caller).
- **Memory behavior — the honest stencil story:** small-step passes are cache-friendly (neighbors
  near the tile); the *large*-step passes gather from cells hundreds of rows away — effectively
  random access that leans on L2. That is intrinsic to JFA's teleporting information flow: the
  price of `O(log)` passes is early-pass locality. `int4` cell states keep every access one
  16-byte vector transaction (why the state is padded to 16 bytes).
- **No shared memory** — a tile only covers the ±step ring when step < tile size (the last pass or
  two); speculative tiling is Exercise 4, to be *measured*, not assumed.
- **Memory-space contrast with 09.01** (the two projects are a deliberate pair): 09.01's robot
  model is read *uniformly* → `__constant__` broadcast is perfect; this project's seed list is
  read *divergently* (each scatter thread a different seed) → constant memory would serialize;
  plain global memory is correct. **Choose the memory space by access pattern, not data size.**
- **Determinism:** every pass is a pure gather with strict-`<` acceptance — no atomics, no
  reductions, no float math. Result: bit-identical output on every run, machine, and thread
  schedule; even the mismatch *counts* against the oracle are fixed numbers.

## Numerical considerations

There is deliberately **no floating point** in the algorithm — squared distances are int64, so:
no rounding, no FMA-contraction story, no tolerance. The considerations that remain:

- **Overflow discipline:** `(x−sx)²` fits int32 only up to ~46k-cell grids; the code computes in
  `long long` (int64) unconditionally — cheap insurance with the reasoning in a comment.
- **Ties are semantics, not noise:** two seeds exactly equidistant from a cell are *both* correct
  answers. The GPU keeps the incumbent on ties (strict `<`); the CPU oracle keeps the lowest id
  (scan order). The comparator therefore treats equal-distance label disagreements as agreement —
  a policy decision that belongs in the verification contract, and does.
- **The approximation itself** is the "numerics" of this project: plain JFA on the 1024²/128-seed
  batch produced 13 mislabeled cells with up to 3.9-cell distance error — over the documented
  2-cell bound; 1+JFA brought it to 4 cells / 0.88. The bound is *empirically calibrated to the
  variant*, and the check runs every demo, so a regression (or a "harmless refactor" that breaks
  the funnel) trips the demo immediately.
- **Display quantization:** the PGM artifacts quantize distances to 8-bit for viewing — display
  only, never fed back into any computation.

## How we verify correctness

**The oracle is a different algorithm, on purpose.** 33.01/09.01 verified a *port* (same math,
serial twin). Here the GPU runs an approximate algorithm, so the CPU runs the *exact* one
(brute-force scan, [`src/reference_cpu.cpp`](src/reference_cpu.cpp)) and verification checks the
**approximation's documented promise**:

- every cell labeled (no sentinels survive) — catches schedule/buffer-parity bugs;
- label mismatches (labels differ AND distances differ) ≤ **0.5% of cells**;
- max distance error `√d²_jfa − √d²_exact` ≤ **2 cells** — mislabels must be near-ties, not
  gross errors;
- exact-distance ties counted as agreement (see §Numerical considerations).

Because everything is integer-exact, these metrics are deterministic — the demo's `[info]` lines
print the same counts on every machine (512² sample: 0 mismatches; 1024² batch: 4 cells, 0.88).

**The bounds check earned its keep during development:** the first (plain-JFA) implementation
passed the sample stage *perfectly* and failed the batch stage (13 cells, 3.9 > 2.0). A
tolerance-free eyeball test would have shipped it; the exact oracle + explicit bound caught it,
and the fix (1+JFA) is now a commented decision in `kernels.cu`. This is the strongest argument
this repository can make for oracle-first development — it happened, in this project, on push
2026-07-08-03.

Two-stage structure as usual: committed 512²/64-seed sample (offline reproducibility, §4) + a
1024²/128-seed deterministic in-memory batch (scale + timing).

## Where this sits in the real world

- **Rong & Tan (2006)** introduced JFA for exactly this pair of outputs; their paper also gives
  the variants (1+JFA, JFA²) and the error analysis our bounds echo. JFA went on to become a
  graphics workhorse (soft shadows, SDF generation for fonts/meshes — project 07.04's cousin).
- **Exact alternatives:** Felzenszwalb–Huttenlocher's separable O(W·H) transform is the strongest
  CPU answer for distance-only (Exercise 5); OpenCV's `distanceTransform` ships the classic
  chamfer/exact family. When a safety argument needs *exact* clearance, use them and pay the pass
  structure (the separable method parallelizes per-row/per-column — also GPU-friendly, just less
  simple).
- **Production robotics:** Nav2's costmap_2d *inflates* incrementally around changed cells rather
  than recomputing fields — the engineering trade against a 4 ms full recompute is update
  locality vs. worst-case latency. In 3-D, ESDF builders (nvblox on GPU, Voxblox on CPU) serve
  the same clearance queries to drone/manipulator planners; JFA's 3-D extension is one of the
  standard GPU routes, and project 05.02 picks that thread up.
- **The GVD connection:** keep the labels and mark cells whose neighbors disagree — that boundary
  *is* the generalized Voronoi diagram, the free-space skeleton (05.16 builds it with a GPU
  brushfire, this project's sibling technique).
- **Scope honesty:** real costmaps want obstacle-*footprint* distance (rasterized polygons as
  seeds — a preprocessing step, same algorithm), often anisotropic or time-varying costs, and
  incremental updates; none change the core taught here.
