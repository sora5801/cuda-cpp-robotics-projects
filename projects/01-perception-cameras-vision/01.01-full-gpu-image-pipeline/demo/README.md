# Demo — 01.01 Full GPU image pipeline: debayer → undistort → rectify → resize → normalize, zero CPU copies

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

The demo runs the full five-stage pipeline (debayer -> undistort+rectify -> resize -> normalize) on
the committed synthetic Bayer scene TWICE: once **STAGED** (one kernel per stage, full
intermediate images in global memory) and once **FUSED** (undistort+rectify+resize collapsed into
one kernel). It then:

1. Verifies every GPU kernel against an independent CPU twin (`VERIFY:` line).
2. Runs **seven physical gates** that check things the twin comparison cannot — camera-model
   self-consistency (`roundtrip`), that rectification actually straightens a known-straight
   checkerboard edge (`straightness_rectified`) while the SAME edge measures curved in the
   uncorrected raw image (`distortion_negative_control` — the built-in negative control), that the
   rectified output matches the analytic ground truth away from edges (`color_fidelity`), that
   resizing conserves the image's mean (`resize_conservation`), that the final tensor is genuinely
   zero-mean/unit-std (`normalize`), and that the fused and staged pipelines agree (`fused_vs_staged`).
3. Prints a derived (not profiler-measured) memory-traffic comparison and the measured kernel times
   for both pipelines — the kernel-fusion lesson this project exists to teach.
4. Writes every stage's image to `demo/out/` plus a `gates_metrics.csv` record of every measured
   number.

**Look at the artifacts** (viewable in any image tool — GIMP, IrfanView, VS Code's image preview, or
convert with any PPM/PGM-aware viewer): `bayer_input.pgm` (the raw mosaic — visibly speckled),
`debayered.ppm` (full color, but still tilted/curved — distortion+rectification not yet removed),
`rectified.ppm` (the checkerboard is now square and axis-aligned — THIS is what undistort+rectify
did), `resized.ppm` / `fused_resized.ppm` (half-resolution, staged vs. fused — visually
indistinguishable, numerically almost identical), and `normalized_vis.ppm` (the final float tensor,
rescaled back to a viewable image for display only).

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, and every MEASURED number behind a gate/verify verdict — varies slightly by GPU architecture (FMA-contraction differences; see `../src/main.cu`'s output-contract comment). | No. |
| `PROBLEM:`  | The exact problem instance: image size, camera parameters, resize factor. | Yes — stable (demo runs with no args). |
| `DATA:`     | Which sample was loaded and its provenance. | Yes — stable. |
| `[time]`    | Per-stage and total kernel times for both pipelines, and the CPU oracle time — **teaching artifacts, never benchmark claims** (single-shot, first-launch JIT costs included). | No. |
| `VERIFY:`   | `PASS`/`FAIL` — every GPU kernel agrees with `reference_cpu.cpp`'s independent twin within the documented per-stage tolerance. | Yes — stable. |
| `GATE <name>:` | `PASS`/`FAIL` verdict of one of the seven physical gates (see "What the demo demonstrates" above); the measured number behind each verdict is on the following `[info]` line. | Yes — stable (7 lines). |
| `ARTIFACT:` | Confirms every `demo/out/` file was written. | Yes — stable. |
| `RESULT:`   | Overall `PASS`/`FAIL` — `VERIFY` AND all seven gates AND the artifact write must all succeed. The program exits nonzero on `FAIL`. | Yes — stable. |

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
