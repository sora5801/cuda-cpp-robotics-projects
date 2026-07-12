# Demo — 02.17 LiDAR-camera projection/coloring fusion kernels

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

Two directions of LiDAR-camera fusion on one synthetic scene (`../data/README.md`):

- **Direction A — point coloring.** Every LiDAR point is projected into the camera image and given a
  bilinear-sampled color, TWICE: once naively (whatever pixel it lands on) and once through an honest
  z-buffer occlusion check. The demo measures the naive path's designed failure (points hidden behind
  the red occluder get painted the occluder's color instead of the green background's) and the checked
  path's fix, on the SAME ground-truth-occluded cohort.
- **Direction B — depth painting.** The same z-buffer pass IS a sparse depth image — the demo paints it
  to `painted_depth.pgm` and checks its coverage/fidelity independently.
- **The calibration-error sensitivity sweep.** `T_camera_lidar` is perturbed by documented
  rotation/translation errors and the resulting color drift is measured into a curve, cross-checked
  against 01.17's analytic pixel-displacement formula.

Every stage runs on both the GPU kernels and an independent CPU twin (`VERIFY:` lines); every
evaluation gate (`GATE:` lines) grades the GPU's output against ground truth
`scripts/make_synthetic.py` computed independently (never seen by the pipeline itself).

## Artifacts (`out/`, written every run)

| File | What it shows |
|---|---|
| `cloud_topview.ppm` / `cloud_sideview.ppm` | The full colored point cloud (top-down X-Y and side X-Z orthographic scatter renders) — the "money shot": each point in its checked-coloring result, neutral gray where the occlusion check filtered it out. |
| `occlusion_cohort_naive.ppm` / `occlusion_cohort_checked.ppm` | The ground-truth-occluded cohort ONLY, zoomed in — colored naively (before) vs. colored with the occlusion check applied (after). Compare the two: naive is mostly red (wrong); checked is mostly neutral gray (correctly filtered). |
| `painted_depth.pgm` | The sparse depth image (Direction B's product); near = bright, matching 01.18's convention. |
| `sensitivity_curve.csv` | Every calibration-error sweep level: flip fraction, measured mean pixel displacement, and the 01.17-predicted displacement. |
| `gates_metrics.csv` | Every gate's key bookkeeping numbers. |

## How to read the console output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, measured numbers behind each gate — varies by machine/run. | No. |
| `PROBLEM:` / `SCENARIO:` | The problem instance and the synthetic scene's shape. | Yes — stable. |
| `[time]`    | CPU reference ms vs. GPU kernel ms — a **teaching artifact, never a benchmark claim** (single-shot; first launches pay one-time init costs). | No. |
| `VERIFY:`   | GPU-vs-CPU twin agreement per kernel (tolerances documented in `../src/main.cu` and `THEORY.md`). | Yes — stable. |
| `GATE:`     | An independent evaluation gate against ground truth (never routes through the twin comparison). | Yes — stable. |
| `ARTIFACT:` | Confirms every file in `out/` was written. | Yes — stable. |
| `RESULT:`   | `PASS`/`FAIL` — requires every `VERIFY:` and every `GATE:` to pass. Nonzero exit on `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info) are allowed. `#`-prefixed lines in that file are comments.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list).
- **`RESULT: FAIL`:** the GPU result disagreed with the CPU oracle — a real bug. Start in
  `../src/kernels.cu` and compare against `../src/reference_cpu.cpp`.
- **Expected-line mismatch only:** the program passed its own check but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The two
  are a contract; fix them together.
