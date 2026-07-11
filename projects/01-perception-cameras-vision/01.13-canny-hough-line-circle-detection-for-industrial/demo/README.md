# Demo — 01.13 Canny + Hough line/circle detection for industrial alignment

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

The full industrial-vision pipeline, twice over: once on `data/sample/scene.pgm` (a synthetic machined
plate rendered under a KNOWN in-plane offset+rotation) and once on `negative_control.pgm` (the same
background with no part at all). It runs Canny edge detection (Gaussian blur -> Sobel -> non-max
suppression -> double-threshold hysteresis) entirely on the GPU, verifies every stage bit-for-bit or
within a documented float tolerance against an independent CPU implementation, then runs two Hough
transforms (lines via integer-atomic voting on a fixed-point theta table — genuinely BIT-EXACT between
GPU and CPU; circles via known-radius gradient-directed voting), extracts the detected lines/circles,
and solves a small least-squares rigid alignment to recover the plate's offset — the actual measurement
a factory inspection station reports. Six independent gates then check: are all 4 edges found
(`line_recovery`)? all 3 holes (`circle_recovery`)? is the recovered offset close to the truth
(`alignment` — the business gate)? is the Canny map itself accurate (`edge_quality`)? does the
deliberately weak scratch mark survive double-threshold hysteresis but NOT a single high threshold
(`hysteresis_lesson` — the whole reason hysteresis exists, quantified)? and does the part-free image
report nothing at all (`negative_control`)?

**Artifacts** (written to `demo/out/`, git-ignored — regenerate by re-running the demo):

| File | What it is |
|------|------------|
| `edges.pgm` | The double-threshold Canny edge map (0/255 grayscale) — open it in any PGM viewer. |
| `hough_lines_accum.pgm` | The (theta, rho) line accumulator, log-stretched to 0-255 — each bright horizontal streak is one real line's sinusoid family converging on its vote peak; read the axes as rho (columns, 801 bins) x theta (rows, 180 bins, 1 degree each). |
| `overlay.ppm` | The scene with detected lines (green), detected circles (red), and the recovered alignment vector (yellow, drawn from nominal to recovered center) drawn on top — open with any PPM-capable viewer (GIMP, IrfanView, `convert` from ImageMagick, or `python -c "from PIL import Image; Image.open('overlay.ppm').show()"`). |
| `gates_metrics.csv` | Every gate's measured value(s), bound(s), and PASS/FAIL — for the learner to inspect or plot outside the demo. |

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, measured tolerances/sweep counts/detections — informative, varies by machine/run. | No. |
| `PROBLEM:`  | The exact problem instance (image size, pipeline stages, Hough dimensions). | Yes — stable (demo runs with no args). |
| `DATA:`     | Confirms the synthetic scene/negative-control/truth files loaded. | Yes — stable. |
| `[time]`    | GPU pipeline ms, CPU pipeline ms, and a speed-up figure — a **teaching artifact, never a benchmark claim** (single-shot; first launches pay one-time init costs). | No. |
| `VERIFY:`   | One line per twin category (Gaussian/Sobel/NMS float-tolerance; hysteresis edge map EXACT; Hough line accumulator BIT-EXACT; Hough circle accumulator peak-tolerance) — see `THEORY.md` "How we verify correctness". | Yes — stable. |
| `GATE <name>:` | One of the 6 independent gates described above. | Yes — stable. |
| `ARTIFACT:` | Confirms the 4 output files were written. | Yes — stable. |
| `RESULT:`   | `PASS`/`FAIL` verdict of the whole run (every VERIFY and every GATE). The program exits nonzero on `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info) are allowed. `#`-prefixed lines in that file are comments.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list).
- **`RESULT: FAIL` with a failing `VERIFY:` line:** the GPU result disagreed with the CPU oracle — a
  real pipeline bug. Start in `../src/kernels.cu` and compare against `../src/reference_cpu.cpp`.
- **`RESULT: FAIL` with a failing `GATE`:** the two paths agree with each other but the DETECTION
  itself missed something (or the negative control found something it should not have). Check
  `demo/out/overlay.ppm` and `gates_metrics.csv` first, then `THEORY.md` "How we verify correctness"
  for how each gate's bound was measured.
- **Expected-line mismatch only:** the program passed its own check but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The two
  are a contract; fix them together.
