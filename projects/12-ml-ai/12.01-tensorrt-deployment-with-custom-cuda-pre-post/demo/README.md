# Demo — 12.01 TensorRT deployment with custom CUDA pre/post kernels: NMS, argmax decode, keypoint extraction

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3, NO TensorRT needed)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

The **fallback (default, no-TensorRT) pipeline**, end to end, on a single synthetic 80×80 test
image: a bilinear-resize + normalize + transpose preprocessing kernel, a tiny fixed 3-layer
"detector" (2 conv+ReLU layers and a 1×1 detection head — all hand-designed, deterministic weights,
not trained), then the catalog bullet's three named post-processing kernels — **argmax class
decode**, **score-threshold + anchor-arithmetic box decode**, **NMS** (a real GPU IoU-matrix kernel
plus a documented sequential greedy-suppression scan), and **keypoint extraction** (a local-window
heatmap argmax). Every stage runs on the GPU AND on a plain-C++ CPU oracle
([`../src/reference_cpu.cpp`](../src/reference_cpu.cpp)); the demo diffs every intermediate tensor
between the two before trusting the final answer.

The scene contains **3 known objects** (2 red rectangles, 1 blue) at fixed positions
(`../data/sample/ground_truth.csv`). The demo's `GROUNDTRUTH:` gate requires all 3 to be detected
within a documented center-distance/IoU tolerance, checks the false-positive count against a
documented bound, and checks that NMS actually reduced the candidate count by a meaningful factor —
**measured: 21 pre-NMS candidates → 3 post-NMS detections (7.0x), 0 false positives, worst center
error 2.40 px, worst IoU 0.601** (see `[info]` lines; the exact numbers are not part of the checked
contract — only the PASS/FAIL verdicts are, per this repo's convention).

**This demo writes two artifacts** (git-ignored, regenerated each run):

- `out/detections.pgm` — a grayscale render of the test image with detection box outlines (white)
  and keypoint markers (small black crosses) burned in. Open it with any image viewer that reads
  PGM (GIMP, IrfanView, `feh`, or convert with ImageMagick: `magick out/detections.pgm out.png`).
- `out/detections.csv` — one row per surviving detection: class, score, box corners, keypoint —
  all in SOURCE-image pixel coordinates for direct comparison against `ground_truth.csv`.

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, resolved file paths, and every MEASURED number (candidate counts, center errors, IoUs, TensorRT availability) — varies by machine/GPU/TensorRT-build. | No. |
| `PROBLEM:`  | The fixed architecture/shape summary (compile-time constants only). | Yes — stable. |
| `WEIGHTS:` / `SCENE:` | Confirmation that the sample data loaded, with no machine-specific paths. | Yes — stable. |
| `[time]`    | CPU reference ms and GPU kernel ms — a **teaching artifact, never a benchmark claim** (single-shot; the pipeline's kernels are tiny at this teaching image size — see THEORY.md "GPU mapping" for why that is honest and expected). | No. |
| `VERIFY:`   | PASS/FAIL of the GPU-vs-CPU pipeline agreement (every stage, tolerances documented in `../src/main.cu` and `THEORY.md`). | Yes — stable. |
| `GROUNDTRUTH:` | PASS/FAIL of the detection-quality gate against the known scene. | Yes — stable. |
| `ARTIFACT:` | Confirms both output files were written. | Yes — stable. |
| `RESULT:`   | Final PASS/FAIL. The program exits nonzero on `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info, measured numbers) are allowed. `#`-prefixed lines in that file are comments.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list). Building this project needs **no TensorRT** —
  if your build system is trying to find TensorRT headers, you (or a downstream tool) enabled the
  optional path; see README "Build" to disable it again.
- **`VERIFY: FAIL`:** the GPU result disagreed with the CPU oracle at some pipeline stage — a real
  bug. The `[info] verify:` line names which stage (`preprocess`/`conv1`/`conv2`/`head`) first
  diverged, or whether the pre-/post-NMS candidate COUNTS themselves disagreed. Start in
  `../src/kernels.cu` and compare against `../src/reference_cpu.cpp` stage by stage.
  `../src/main.cu` performs this comparison.
- **`GROUNDTRUTH: FAIL`:** the pipeline is internally consistent (VERIFY passed) but the fallback
  path's own detections drifted off the known scene — check `out/detections.pgm` visually and the
  `[info] groundtruth:` line for which bound (match count, false positives, NMS reduction) failed.
- **Expected-line mismatch only:** the program passed its own checks but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The two
  are a contract; fix them together.
