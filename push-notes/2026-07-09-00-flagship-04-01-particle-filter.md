# Push note — 2026-07-09-00: flagship 04.01 particle filter

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

The first **worker-built** flagship: 04.01 — massive particle filter localization. A hundred
thousand particles localize a simulated 2-D robot on an occupancy grid at 10 Hz, with per-particle
range-beam raycasting on the GPU. Also the first project produced by the batch pipeline: a build
worker wrote the implementation, was cut off by an account session limit before ever compiling it,
and a finisher agent then built, tested, and documented it; the lead re-ran every gate
independently before merging. The multi-agent process (and its budget constraints) is now
calibrated — details under Known limitations.

## What changed

- **[projects/04-sensor-fusion-state-estimation/04.01-massive-particle-filter-localization/](../projects/04-sensor-fusion-state-estimation/04.01-massive-particle-filter-localization/)** —
  complete: predict + weight kernels (one thread per particle; in-kernel xorshift32 noise, no
  cuRAND — deterministic), host systematic resampling, strict sample loader, two-stage
  verification, trajectory artifact, full README / THEORY / PRACTICE.
- **[docs/STATUS.md](../docs/STATUS.md)** — 04.01 → `done` (**5/505**).

## New projects (didactic blurbs)

**04.01 — Massive particle filter localization** (★ beginner, domain 04, flagship). Teaches the
Bayes → particle filter chain concretely: predict (odometry + per-particle noise), weight
(raycast 16 beams per particle into the grid map, Gaussian range likelihood), systematic
resampling, weighted-mean estimate. GPU pattern: thread-per-particle with divergent map reads —
the honest contrast to constant-memory broadcast (09.01). The single most interesting thing to
look at: the weight kernel's per-particle DDA raycast in `src/kernels.cu`, and the effective-
sample-size discussion in THEORY §numerics (weight degeneracy is *the* particle-filter failure
mode).

## How to build & run

```powershell
projects\04-sensor-fusion-state-estimation\04.01-massive-particle-filter-localization\demo\run_demo.ps1
# then plot demo\out\trajectory_est.csv (estimated vs true path)
```

## What to study here

Project `README.md` → `THEORY.md` (Bayes filter math → particle approximation → ESS/degeneracy) →
`src/kernels.cu` (predict, then the raycasting weight kernel) → `src/main.cu` (systematic
resampling on the host, and why that split is honest v1 design). Exercise to try first: the GPU
prefix-sum resampler the README proposes.

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-09), re-run independently by the lead after the finisher's self-gate:

- `Release|x64` **and** `Debug|x64` build with **zero errors, zero warnings** (clean rebuilds).
- `demo/run_demo.ps1` passes end to end: all 6 stable lines matched, exit 0.
- **GPU-vs-CPU gates:** predict worst deviation 4.8e-07 abs; weight worst deviation 2.4e-07 rel
  (tolerances 1e-4 / 1e-3).
- **Closed loop (K=100,000, 120 steps @ 10 Hz):** position RMSE **0.0193 m** (threshold 0.15 m),
  heading RMSE 0.0042 rad; ESS min 1,376 / mean 38,016; weight kernel ≈ 1.0 ms GPU vs ≈ 358 ms
  single-core CPU (teaching artifact).
- `tools/verify_project.py`: **all structural gates PASS**; no changes outside the project folder.

## Known limitations / TODOs

- v1 scope as ratified: known map, systematic resampling on host, 2-D — extensions documented as
  exercises; kidnapped-robot/global-localization scenarios are siblings in domain 04.
- **Process note (§10 deviation, ratified):** workers run in the shared tree with strict
  disjoint-folder ownership instead of per-worker git branches — single-machine orchestration;
  the lead commits and pushes centrally. Also: account session limits killed two full worker
  batches mid-build (8-worker and 4-worker); the pipeline now runs ~2 builders per session window
  with finisher agents completing interrupted work. **This push also carries the interrupted,
  UNVERIFIED partial `src/` for 05.01, 22.01, and 31.01** (a checkpoint commit — never compiled,
  no docs); their STATUS rows say `in-progress`, and `verify_project.py` correctly reports them
  not-done. Finishers are completing them next.

## Next push preview

Finishers are already running for 05.01 TSDF fusion + marching cubes; 22.01 and 31.01 follow, then
batch 1b (06.05 STOMP, 15.01 min-snap, 17.01 Lambert, 23.01 costmaps+DWA).
