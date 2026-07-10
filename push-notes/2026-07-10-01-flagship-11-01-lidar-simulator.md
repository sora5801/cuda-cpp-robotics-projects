# Push note — 2026-07-10-01: flagship 11.01 lidar simulator

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Flagship 11.01 — the **GPU LiDAR simulator** — opens batch 1d: a 32-channel spinning LiDAR
raycasts 32,768 beams per frame through a hand-built BVH over a synthetic warehouse mesh, with the
three effect models the catalog names (beam divergence via a 5-ray bundle, Lambertian cos·/R²
intensity, range/incidence-dependent dropout + noise). This project heads SYSTEM_DESIGN's
composition Chain A (11.01 → 02.06 → 05.01 → …), and its output buffer is deliberately
PointCloud-shaped so that chain is real, not rhetorical. Verification is the strongest analytic
set yet: ground-plane ranges match the closed form at 7.7e-08 relative, the inverse-square
intensity ratio comes out 4.000000 exactly, dropout matches its binomial bound at 5σ — and the
BVH's depth bound is *proved by induction in THEORY.md*, then measured to land exactly on the
bound (depth 10 of 10).

## What changed

- **[projects/11-sensor-sim-digital-twins/11.01-gpu-lidar-simulator/](../projects/11-sensor-sim-digital-twins/11.01-gpu-lidar-simulator/)** —
  complete: host median-split BVH (flattened 32-byte nodes) + fixed-stack traversal,
  Möller–Trumbore kernel (derived in comments), divergence/intensity/dropout models, CPU twin,
  three analytic gates + frame gates, cloud CSV + range-image PGM artifacts, full README /
  THEORY / PRACTICE.
- **[docs/STATUS.md](../docs/STATUS.md)** — 11.01 → `done` (**17/505**).

## New projects (didactic blurbs)

**11.01 — GPU LiDAR simulator** (★ beginner, domain 11, flagship). Build-your-own-BVH before
touching OptiX (the §5 stance made concrete): median-split construction with a provable depth
bound, linearized nodes, and the divergent-traversal honesty every ray-tracing GPU programmer
needs. The radiometry derivation distinguishes LiDAR's inverse-square from radar's inverse-fourth
(03.01's companion lesson). The single most interesting thing to look at:
`demo/out/range_image.pgm` — the LiDAR's native picture, 1024×32, shelving racks and crates
clearly legible.

## How to build & run

```powershell
projects\11-sensor-sim-digital-twins\11.01-gpu-lidar-simulator\demo\run_demo.ps1
# then open demo\out\range_image.pgm and plot demo\out\cloud.csv
```

## What to study here

`THEORY.md` §The problem (how a real spinning LiDAR actually fires) → the BVH depth-bound proof →
`src/kernels.cu` (Möller–Trumbore + traversal). Then feed `demo/out/cloud.csv` to your own eyes
next to 02.06 — Chain A's first hop, visible. First exercise: swap median-split for a surface-area
heuristic and measure traversal-step counts.

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-10), re-run independently by the lead after the builder's self-gate:

- `Release|x64` **and** `Debug|x64` build with **zero errors, zero warnings**.
- `demo/run_demo.ps1` passes end to end: all 13 stable lines matched, exit 0.
- **GPU-vs-CPU gate:** hit/dropped decisions exact (0/32,768 mismatches both); intensity worst
  deviation 2.0e-04 (tol 1e-3); range worst 1.2e-02 confined to 5 documented silhouette-edge
  beams (0.02% — the 5-ray argmin's discontinuity, measured and explained, not hidden).
- **Analytic gates:** ground-plane closed form 7.7e-08 rel; inverse-square ratio exactly 4.000000;
  dropout 0.02405 vs 0.02375 theoretical (5σ bound ±0.00538).
- **Frame gates:** hit fraction 0.712, mean range 11.1 m — within documented ranges.
- Timing (teaching artifact): 32k beams ≈ 1.9 ms GPU vs ≈ 232 ms CPU; BVH build 0.56 ms host,
  one-time.
- `tools/verify_project.py`: **all structural gates PASS**; no changes outside the project folder.

## Known limitations / TODOs

- Single return per beam (no multi-return), no atmospheric effects, host-built BVH (GPU LBVH is
  07.03, cross-referenced), static scene — all documented with production counterparts (Isaac Sim
  RTX LiDAR, CARLA, OptiX).

## Next push preview

12.01 TensorRT deployment — the repo's first heavy-SDK project, which must ship the §5 fallback
demo path so the Definition of Done holds on a clean VS+CUDA machine. Then 13.03 foothold scoring
and 14.02 traversability costmaps.
