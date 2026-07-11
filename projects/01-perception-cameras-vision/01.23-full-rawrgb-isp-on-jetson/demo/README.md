# Demo — 01.23 Full RAW→RGB ISP on Jetson (Argus + custom CUDA stages)

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

The demo runs the complete RAW→RGB ISP — black level, lens-shading correction, defective-pixel
correction, white balance (gray-world AND white-patch estimators), demosaic (Malvar-He-Cutler AND a
bilinear baseline), the color-correction matrix, and gamma — on a synthetic 160×120 RAW10 sensor
frame, twice (a D65-lit scene and a tungsten-lit scene), both staged (one kernel per stage) and
fused (stages 1–4 in one kernel). It then runs **ten independent physical gates** (see
[`../README.md`](../README.md) "Expected output" for the full list and measured numbers) that check
things a plain GPU-vs-CPU comparison cannot: whether shading correction is actually flat across
radius, whether the committed defect pixels are actually recovered, whether MHC actually beats
bilinear demosaic (by a measured dB margin), whether gray-world AWB actually fails on a red-heavy
crop (a designed negative control), and whether the whole pipeline actually reconstructs the
original scene's colors.

**Visual artifacts** land in `demo/out/`: `raw_vis_d65.pgm` (the raw mosaic, viewable), `shading_corrected_d65.pgm`,
`demosaiced_mhc_d65.ppm` / `demosaiced_bilinear_d65.ppm` (compare these two side by side — the MHC
one should look visibly cleaner at the hashed-texture region's edges), `white_balanced_d65.ppm`,
`final_d65.ppm` / `final_tungsten.ppm` / `final_tungsten_wrong_awb.ppm` (the last one shows the
uncorrected color cast — compare against `final_tungsten.ppm`), `chart_crop_d65.ppm` (just the
24-patch chart, cropped, for a close look at color accuracy), and `gates_metrics.csv` (every gate's
measured value, tolerance, and verdict in one machine-readable table).

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, and every measured stat behind a gate (gains, residuals, dB, byte counts) — varies by machine/GPU. | No. |
| `PROBLEM:`  | The exact problem instance (resolution, sensor constants). | Yes — stable (demo runs with no args). |
| `DATA:`     | What sample was loaded. | Yes — stable. |
| `[time]`    | Kernel and CPU-oracle timings — a **teaching artifact, never a benchmark claim** (single-shot, one machine). | No. |
| `VERIFY:`   | GPU-vs-CPU agreement across every stage (tolerance documented in `../src/main.cu`). | Yes — stable. |
| `GATE <name>:` | One of the ten physical gates (PASS/FAIL, no embedded numbers — the number lives on the paired `[info]` line). | Yes — stable. |
| `ARTIFACT:` | Confirms the `demo/out/` files were written. | Yes — stable. |
| `RESULT:`   | Overall `PASS`/`FAIL` verdict. The program exits nonzero on `FAIL`. | Yes — stable. |

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
