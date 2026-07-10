# Push note — 2026-07-09-08: flagship 01.02 stereo sgm

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Flagship 01.02 — **stereo depth: block matching, then SGM** — opens batch 1c and the perception
column. The catalog bullet asks for a *progression*, and the demo delivers it as a measured fact:
on the same synthetic rectified pair with dense ground truth, winner-take-all block matching
scores **63.35%** good pixels and 4-path Semi-Global Matching scores **97.52%** — a 34-point gap
you can see in the committed error maps. The entire disparity pipeline (census transform, Hamming
cost volume, SGM aggregation) is deliberately integer-only, so GPU-vs-CPU verification is **exact
equality across five checkpoints** (~14 million values, zero mismatches). THEORY.md is honest
about SGM's famous weakness on GPUs: the path-sequential aggregation drops parallelism from ~110k
threads to ~350, and the 62 ms it costs (vs 1 ms for the cost volume) is the measured price.

## What changed

- **[projects/01-perception-cameras-vision/01.02-stereo-depth/](../projects/01-perception-cameras-vision/01.02-stereo-depth/)** —
  complete: census / cost-volume / BM / 4-path-SGM kernels + LR-check + median, CPU twin of every
  stage, z-buffer-derived synthetic stereo data with exact occlusion masks, disparity + error-map
  artifacts, full README / THEORY / PRACTICE.
- **[docs/STATUS.md](../docs/STATUS.md)** — 01.02 → `done` (**13/505**).

## New projects (didactic blurbs)

**01.02 — Stereo depth: BM → SGM** (★ beginner, domain 01, flagship). Teaches how two cameras
become a depth sensor: epipolar geometry reduces matching to 1-D, `Z = fB/d` turns disparity into
meters (with the Z² accuracy falloff derived), census beats SAD because real cameras disagree
radiometrically, and SGM's 1-D path aggregation approximates a 2-D smoothness energy well enough
to erase BM's streaks. GPU lessons: the D-major cost-volume layout argument, and the
parallelism tension between per-pixel stages and path-sequential aggregation. The single most
interesting thing to look at: `demo/out/error_map.pgm` next to the two disparity maps.

## How to build & run

```powershell
projects\01-perception-cameras-vision\01.02-stereo-depth\demo\run_demo.ps1
# then open demo\out\disparity_bm.pgm, disparity_sgm.pgm, error_map.pgm side by side
```

## What to study here

`THEORY.md` §The problem (epipolar geometry, the Z² falloff) → §The algorithm (the SGM energy
model and P1/P2 semantics) → `src/kernels.cu` in pipeline order. First exercise: raise P2 and
watch over-smoothing eat the depth discontinuities — the classic SGM tuning lesson.

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-09), re-run independently by the lead after the builder's self-gate:

- `Release|x64` **and** `Debug|x64` build with **zero errors, zero warnings** (the builder also
  root-caused a pre-existing LNK4099 PDB race in the CUDA MSBuild integration — it reproduces on
  earlier flagships under forced clean rebuilds — and suppressed it with documentation; flagged
  for the §11 standards retrospective).
- `demo/run_demo.ps1` passes end to end: all 8 stable lines matched, exit 0.
- **GPU-vs-CPU gate: exact equality** — 0 mismatches over 221,184 census signatures, 7,077,888
  cost-volume entries, 7,077,888 SGM-path entries, and both 110,592-pixel disparity maps.
- **Ground-truth gates:** BM 63.35% ≥ 45%; SGM 97.52% ≥ 85%; SGM−BM margin 34.17 ≥ 15 points.
- Timing (teaching artifacts): census+cost ≈ 1.0 ms; BM ≈ 0.1 ms; SGM 4-path ≈ 62 ms (the taught
  parallelism tension); full CPU oracle ≈ 539 ms.
- `tools/verify_project.py`: **all structural gates PASS**; no changes outside the project folder.

## Known limitations / TODOs

- 4 SGM paths (8-path documented), D=64, synthetic pre-rectified input (rectification is
  01.01/01.07's job; Middlebury/KITTI licensing favors synthetic v1 — documented in
  data/README.md), fronto-parallel-dominant scene. All in README §Limitations.

## Next push preview

Batch 1c continues: 02.06 GPU ICP (point-to-plane), then 03.01 FMCW radar cube + CFAR, 10.03
massively-parallel robot sim.
