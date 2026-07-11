# Demo — 01.21 Scene flow from RGB-D pairs

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

The full scene-flow pipeline (README §"The algorithm in brief") on a synthetic 128×96 RGB-D pair:
dense 2-level pyramidal Lucas-Kanade flow → 3-D lifting via back-projection (with a depth-
consistency guard) → a robust (IRLS + Horn/Kabsch) fit of the dominant camera ego-motion → residual
segmentation of the one independently moving object (threshold → 3×3 morphological open →
connected-component size filter) → a robust (IRLS+Tukey), fixed-rotation fit of the object's own
motion offset. It runs on **two** data pairs: the main dynamic pair (camera moves AND the box moves)
and a static negative-control pair (camera moves, box does NOT) — the second pair exists purely to
prove the pipeline does not hallucinate movers under camera motion alone.

Along the way it VERIFIES every GPU kernel against its independent CPU twin on the real loaded
data (not synthetic toy inputs), then reports 7 EVALUATION GATES against the scene's known ground
truth: `flow_2d`, `scene_flow_3d`, `ego_motion` (+ `ego_motion_robustness`, the designed
naive-vs-robust comparison), `object_segmentation`, `static_negative_control`, and
`noise_derivation`. `object_motion` is reported `[info]`-only (not gated — see README
"Limitations & honesty" for why).

**Artifacts** (written to `out/`, all from the main dynamic pair):

| File | What it shows |
|------|----------------|
| `flow_2d.ppm` | dense 2-D flow, HSV color-wheel encoded (hue = direction, brightness = magnitude) |
| `scene_flow_magnitude.pgm` | raw 3-D scene-flow magnitude `\|P2-P1\|` per pixel, grayscale |
| `residual_map.pgm` | post-ego-motion residual magnitude `\|T(P1)-P2\|` per pixel — this is what gets thresholded |
| `moving_mask_postmorph.pgm` | the segmented mask AFTER threshold + morphological open, BEFORE the connected-component size filter |
| `moving_mask.pgm` | the FINAL mask — `moving_mask_postmorph.pgm` after the size filter; compare the two to see what it removed |
| `truth_mask.pgm` | the known ground-truth object mask, for visual comparison |
| `overlay.ppm` | the final segmented mask's outline drawn in green over the frame0 RGB image |
| `gates_metrics.csv` | every gate's measured numeric value, machine-readable |

**What to notice:** the moving box's outline in `overlay.ppm` roughly traces the real box, but a
real, coherent block of false positives survives immediately ADJACENT to it even after the size
filter — that is the demo's own honest evidence of a disocclusion-boundary artifact (background
revealed/occluded by the moving box, `\|P2-P1\|` there is genuinely large but WRONG) that happens to
be roughly the same size as the object's own largest surviving fragment. A pixel-count size floor
alone cannot separate a coherent wrong-shaped blob from a coherent right-shaped one — see README
"Limitations & honesty" and THEORY.md "Numerical considerations" for the full, measured story
(before/after component filtering, and why `object_motion`'s direction recovers well while its
magnitude still does not). It is why several gates carry generous, MEASURED-not-aspirational bounds.

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name and compute capability — varies by machine. | No. |
| `PROBLEM:`  | The exact problem instance (sizes, parameters). | Yes — stable (demo runs with no args). |
| `[time]`    | CPU reference ms, GPU kernel ms, and a speed-up figure — a **teaching artifact, never a benchmark claim** (single-shot, kernel-only vs. one CPU core; first launches pay one-time init costs). | No. |
| `RESULT:`   | `PASS`/`FAIL` verdict of the GPU-vs-CPU check (tolerance documented in `../src/main.cu` and `THEORY.md`). The program exits nonzero on `FAIL`. | Yes — stable. |

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
