# Push note — 2026-07-09-07: flagship 23.01 costmaps dwa batch 1b complete

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Flagship 23.01 — **GPU costmaps + DWA** — is done, and with it **worker batch 1b is complete**
(06.05, 15.01, 17.01, 23.01; **12/505 overall, 12 of 36 flagships**). This is the
navigation-stack centerpiece: per 20 ms tick, a simulated 360-beam LiDAR scan is raytraced into an
obstacle layer (per-beam Bresenham with an honestly-taught atomicMax mark/clear race), inflated
(bounded-radius integer-decay gather), fused, and consumed by a DWA local planner scoring 4,096
(v,ω) rollouts — driving a differential-drive robot to its goal in 288 steps with zero lethal-cell
entries. Verification highlight: the entire costmap pipeline uses pure integer arithmetic, so the
GPU-vs-CPU check is **byte-exact equality** (0/65,536 cells differ), not a tolerance. Design
honesty highlight: the first synthetic world triggered DWA's textbook local-minima failure; the
builder redesigned the world to be reactive-solvable, kept the failing world as README Exercise 3,
and taught the failure mode in THEORY.md rather than hiding it.

## What changed

- **[projects/23-navigation-stack/23.01-gpu-costmaps/](../projects/23-navigation-stack/23.01-gpu-costmaps/)** —
  complete: raytrace/inflation/fusion kernels + DWA scoring kernel, CPU twin of all four,
  closed-loop driver, BFS-solvability-checked scenario generator, costmap PGM + path CSV
  artifacts, full README / THEORY / PRACTICE.
- **[docs/STATUS.md](../docs/STATUS.md)** — 23.01 → `done` (**12/505**).

## New projects (didactic blurbs)

**23.01 — GPU costmaps + DWA** (★ beginner, domain 23, flagship). Three GPU patterns in one
project: per-beam raytracing (with the mark/clear race resolved by cost-ordering via atomicMax),
per-cell bounded gathers (inflation), and per-sample rollouts (DWA — 08.01's pattern reused for
argmin). Teaches why costmaps exist (uncertainty buried into inflation; stopping distance sets the
radius) and where DWA breaks (local minima — demonstrated, not just asserted). The single most
interesting thing to look at: `demo/out/costmap.pgm` — inflation halos around every pillar, with
the driven path threading between them in `demo/out/path.csv`.

## How to build & run

```powershell
projects\23-navigation-stack\23.01-gpu-costmaps\demo\run_demo.ps1
# then open demo\out\costmap.pgm and plot demo\out\path.csv
```

## What to study here

Batch 1b as a set completes a planning arc: STOMP optimizes a trajectory (06.05), min-snap makes
it dynamically meaningful (15.01), Lambert plans in orbital time (17.01), and 23.01 closes the
loop on a live costmap. Within 23.01: `THEORY.md` §The algorithm (the DWA scoring trade-offs) →
`src/kernels.cu` (the three patterns back to back). First exercise: swap in the slalom world and
watch DWA get stuck — then read why.

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-09), re-run independently by the lead after the builder's self-gate:

- `Release|x64` **and** `Debug|x64` build with **zero errors, zero warnings**.
- `demo/run_demo.ps1` passes end to end: all 9 stable lines matched, exit 0.
- **Costmap gate:** **byte-exact** — 0/65,536 cells differ between GPU and CPU over a full update
  cycle (integer-arithmetic pipeline by design).
- **DWA gate:** worst relative score deviation 2.2e-07 over 4,096 samples (tol 1e-3).
- **Closed loop:** goal reached in 288/500 steps; 0 lethal-cell entries; 0 emergency brakes.
- Timing (teaching artifacts): costmap update ≈ 0.8 ms GPU vs ≈ 24 ms CPU; DWA scoring ≈ 0.2 ms
  vs ≈ 2.4 ms; in-loop averages 0.31 + 0.05 ms/tick.
- `tools/verify_project.py`: **all structural gates PASS**; no changes outside the project folder.

## Known limitations / TODOs

- Reactive planner only (no global plan — the documented reason the world must be
  reactive-solvable); static world; simulated scans against the true map (no sensor noise model
  beyond discretization). Nav2's incremental-update trade is documented against this full-recompute
  design.

## Next push preview

Batch 1c begins on the remaining 24 flagships, starting with the perception column: 01.02 stereo
SGM, 02.06 GPU ICP, 03.01 FMCW radar cube + CFAR.
