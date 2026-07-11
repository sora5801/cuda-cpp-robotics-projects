# Demo — 01.05 SIFT/SURF on GPU (harder, warp-level reductions)

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

The full SIFT-on-GPU pipeline, run on the committed synthetic scene pair (`scene_a.pgm`/`scene_b.pgm`,
related by a KNOWN 1.5x-zoom + 20deg-rotation similarity transform) plus a negative-control scene
(`neg_scene_c.pgm`):

1. **Builds** a 2-octave x 2-interval Gaussian scale space and Difference-of-Gaussian pyramid for each
   image, on the GPU AND independently on the CPU, comparing them (float tolerance — see THEORY.md).
2. **Detects** DoG extrema (3x3x3 stencil + atomic compaction), refines them to sub-pixel/sub-scale
   precision (an iterative 3x3 linear solve per candidate), and rejects low-contrast/edge-like
   candidates — GPU-authoritative, CPU-compared at the candidate-SET level.
3. **Assigns orientation** via a 36-bin gradient histogram, ONE WARP PER KEYPOINT, using
   `__shfl_down_sync` tree reduction (the catalog's "warp-level reductions" hook) — then builds the
   classic 128-D (4x4x8) SIFT descriptor the same way.
4. **Matches** B against A (and, separately, the unrelated C against A) via brute-force squared-L2,
   Lowe ratio test + mutual cross-check.
5. **Checks SIX independent gates** against the KNOWN ground truth (none of which route through the
   GPU-vs-CPU comparison): `scale_recovery` (the headline gate — median matched-pair scale ratio should
   recover the real 1.5x zoom, something a single-scale detector like 01.04's FAST cannot do),
   `rotation_recovery`, `transform_inlier`, `scale_repeatability`, `negative_control`, and
   `descriptor_normalization`.
6. **Writes artifacts** to `demo/out/`: `keypoints_A.ppm`/`keypoints_B.ppm` (scale circles + orientation
   ticks overlaid on each scene), `matches.ppm` (side-by-side A|B canvas with green/red lines for
   ground-truth-correct/incorrect accepted matches), `matches.csv` (every accepted match's full geometry),
   and `gates_metrics.csv` (every gate's measured value, tolerance, and verdict).

What to notice: the `[info] verify(...)` lines show the GPU and CPU pipelines agreeing to within a few
times `10^-7` at every stage (float32 machine-precision noise, not a real disagreement) — proof the
warp-shuffle-tree reduction computes the SAME answer as a sequential CPU sum, just in a different order.
Then the `GATE ...` lines show the pipeline recovering the KNOWN 1.5x scale change and 20deg rotation
purely from the image pair, with no access to the transform that produced them.

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
