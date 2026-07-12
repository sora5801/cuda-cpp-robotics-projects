# Demo — 02.14 Moving-object segmentation from sequential scans

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

The demo loads a 5-scan window (the CURRENT scan plus its 4 immediately preceding scans) from
`data/sample/`, organizes the current scan into a 16x360 range image, reprojects each of the 4
previous scans into the current sensor's frame, computes the signed per-cell residual against each,
fuses them via MIN(|residual|), thresholds, cleans up with range-image connected-component labeling,
and grades the result against ground truth built into four differently-moving car cohorts (a lateral
crosser, a radial approach, a radial departure, and a car that just stopped) plus two static
honesty cohorts (a wall and a thin pole).

**What to look at:**

- `demo/out/residual_image.pgm` — the signature visual: a grayscale rendering of the fused
  |residual| at every cell. The four cars appear as bright blobs against a mostly-dark (near-zero
  residual) static background — literally "movers glow".
- `demo/out/label_vs_truth.ppm` — a confusion map over the same range-image layout: TP green
  (correctly flagged moving), FP red (falsely flagged), FN orange (missed mover), TN gray (correctly
  static).
- `demo/out/disocclusion_band.ppm` — highlights, in magenta, exactly which WALL cells the crossing
  car's passage occluded/revealed somewhere in the 5-scan window (the ground-truth
  `disocclusion_band` flag) versus clean wall cells (gray) — compare this against where
  `label_vs_truth.ppm` shows any false positives on the wall.
- `demo/out/per_cohort_metrics.csv` and `demo/out/gates_metrics.csv` — the numeric backbone behind
  every `[info]`/`GATE:` line the program prints.

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, and every MEASURED number this project reports (recall/precision/IoU per cohort, sign-consistency fractions, false-positive rates, the window-size study) — varies run to run only in the *number*, never the presence of the line. | No. |
| `PROBLEM:`  | The exact problem instance (window size, range-image shape, threshold). | Yes — stable (demo runs with no args). |
| `VERIFY:`   | GPU-vs-CPU agreement for one of the four pipeline stages (organize / reproject / residual-fuse / CCL). | Yes — stable text, no numbers. |
| `[time]`    | The full canonical MOS pass's GPU kernel time versus the 20 Hz per-scan budget — a **teaching artifact, never a benchmark claim** (this project's demo scale is tiny; a real sensor's full point count would matter more to occupancy/bandwidth). | No. |
| `GATE:`     | `PASS`/`FAIL` verdict for one of the five independent correctness gates (mover_detection, sign_semantics, static_precision, disocclusion_mitigation, timing). | Yes — stable text, no numbers. |
| `ARTIFACT:` | Confirms a demo/out/ file was written. | Yes — stable. |
| `RESULT:`   | Overall `PASS`/`FAIL` — every VERIFY stage AND every GATE must pass. The program exits nonzero on `FAIL`. | Yes — stable. |

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
