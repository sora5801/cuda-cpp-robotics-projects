# Push note — 2026-07-10-02: flagship 12.01 tensorrt deploy

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Flagship 12.01 — **TensorRT deployment with custom CUDA pre/post kernels** — is done, and it is
the repository's first heavy-SDK project, built to prove the §5 dependency policy works: **the
default build compiles zero TensorRT code and the committed demo runs on a clean VS+CUDA
machine**, while the TensorRT integration ships as documented, compile-ready, off-by-default code
(this machine has no TensorRT — verified — so that path is honestly labeled not-run). The
implemented heart is what the catalog bullet actually names: the custom kernels around inference —
bilinear-resize/normalize/transpose preprocessing, argmax class decode, threshold + box decode
with atomic stream compaction, a genuinely parallel IoU matrix with host-side greedy NMS (the
taught parallelism tension), and heatmap keypoint extraction — wrapped around a hand-designed,
deterministic 3-layer synthetic detector whose arithmetic THEORY.md works through by hand.

## What changed

- **[projects/12-ml-ai/12.01-tensorrt-deployment-with-custom-cuda-pre-post/](../projects/12-ml-ai/12.01-tensorrt-deployment-with-custom-cuda-pre-post/)** —
  complete: six pre/post + conv kernels, full-pipeline CPU twin with stage-by-stage verification,
  hand-designed committed weights (460 bytes, format documented byte-exactly), ground-truth
  detection gates, TensorRT path behind `USE_TENSORRT` (README documents the exact opt-in), boxes
  PGM + detections CSV artifacts, full README / THEORY / PRACTICE.
- **[docs/STATUS.md](../docs/STATUS.md)** — 12.01 → `done` (**18/505**).

## New projects (didactic blurbs)

**12.01 — TensorRT deploy + custom kernels** (★ beginner, domain 12, flagship). Teaches the
deployment side of robot ML — the part that is *always* plain CUDA no matter what runs the
network: layout transposes (why NCHW), anchor/box decode arithmetic, IoU/NMS theory including why
greedy suppression resists parallelization, and what TensorRT actually does underneath (fusion,
tactics, precision) — explained precisely, exercised optionally. The single most interesting
thing to look at: `THEORY.md`'s hand-worked forward pass — a "neural network" small enough to
verify with a pencil, which is exactly what makes the deployment plumbing verifiable.

## How to build & run

```powershell
projects\12-ml-ai\12.01-tensorrt-deployment-with-custom-cuda-pre-post\demo\run_demo.ps1
# no TensorRT required — the default path is self-contained; see README §Build to enable TRT
```

## What to study here

`README.md` §Build (the two-path design — the §5 policy in action) → `THEORY.md` §The algorithm
(anchor decode + NMS) → `src/kernels.cu` (the six kernels in pipeline order). First exercise:
enable `USE_TENSORRT` on a machine that has it and compare outputs against the fallback path —
the project is pre-wired for exactly that experiment.

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, **no TensorRT installed — the point**, 2026-07-10), re-run independently by the lead:

- `Release|x64` **and** `Debug|x64` build with **zero errors and zero new warnings**, with no
  TensorRT headers/libs present.
- `demo/run_demo.ps1` passes end to end: all 8 stable lines matched, exit 0.
- **Stage-by-stage GPU-vs-CPU gates:** preprocess exact (0.0); conv1/conv2/head worst 2.2e-07
  (tol 1e-3); candidate counts identical (21 pre-NMS / 3 post-NMS); final detection fields exact.
- **Ground-truth gates:** 3/3 objects matched (worst center error 2.40 px vs tol 6.0; worst IoU
  0.601 vs min 0.30); 0 false positives; NMS reduction 7.0× (min 3.0×).
- The builder also fixed a real artifact-path bug under the optional CMake layout and verified
  `run_demo.sh` end to end.
- `tools/verify_project.py`: **all structural gates PASS**; no changes outside the project folder.

## Known limitations / TODOs

- The network is a fixed synthetic tensor program, not a trained model (training is 12.06's job —
  deployment plumbing is this one's; stated in README §13); TensorRT path compile-ready but
  not executed here (no SDK on the machine — labeled, never claimed).

## Next push preview

13.03 foothold scoring kernels (legged locomotion's terrain evaluation), then 14.02
traversability costmaps closes batch 1d.
