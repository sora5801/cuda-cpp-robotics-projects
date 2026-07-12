# Demo — 02.08 Per-point motion deskew with pose interpolation

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

Six independent checks on the deskew pipeline, run against the committed 4-cohort sample
(`../data/sample/deskew_scan.bin`), in order:

1. **VERIFY** — the §5 GPU-vs-CPU gate: every cohort × both interpolation regimes (dense/sparse),
   compared point-by-point against `../src/reference_cpu.cpp`'s twin.
2. **SLERP_CORRECTNESS** — a data-independent unit test of `quat_slerp` itself: a >90° quaternion
   pair's geodesic-angle progression, plus double-cover sign-flip invariance.
3. **IDENTITY_CONTROL** — the stationary cohort: a correct deskew must be a no-op.
4. **RESTORATION** — the three moving cohorts: dense-sampled deskew must recover the analytic
   instantaneous truth, with the undeskewed (raw) scan reported alongside as the negative-control
   baseline (it should be, and is, an order of magnitude worse).
5. **SAMPLING_LESSON** — the wiggle cohort's sparse-vs-dense error gap (the aliasing lesson), and the
   straight cohort's dense-vs-sparse *agreement* (the "constant velocity is exact" consistency check).
6. **DOWNSTREAM_PAYOFF** — a compact plane-fit RMS on a wall slice of the straight cohort: the wall
   measurably *thickens* in the raw scan and *tightens* back down after deskew.

**Artifacts written to `out/`** (git-ignored, regenerated every run):

- `triptych_wiggle.pgm` — a top-view scatter, three panels side by side (truth | skewed | deskewed) for
  the wiggle cohort — the "money shot": open it in GIMP/IrfanView, or convert with
  `magick out/triptych_wiggle.pgm out/triptych_wiggle.png` (ImageMagick) if you want a PNG.
- `errors_<cohort>.csv` — per-point error breakdown (raw / dense-deskewed / sparse-deskewed vs. truth)
  for each of the 4 cohorts — plot a histogram or a time series against `t_s`.
- `gates_metrics.csv` — every measured number behind every gate above, in one table.

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, per-cohort point counts, and every gate's MEASURED numbers (means, maxima, ratios, RMS values) — floats that can differ in the last bit across GPU architectures. | No. |
| `PROBLEM:` / `DATA:` | The exact problem instance and the loaded sample's cohort/point counts (both fully deterministic — baked into the committed file, never recomputed at run time). | Yes — stable. |
| `[time]`    | CPU/GPU timings — a **teaching artifact, never a benchmark claim** (this pipeline's N is small; the number that matters is "well under the sweep period", not a speed-up multiplier). | No. |
| `VERIFY:` / `SLERP_CORRECTNESS:` / `IDENTITY_CONTROL:` / `RESTORATION:` / `SAMPLING_LESSON:` / `DOWNSTREAM_PAYOFF:` | Each gate's `PASS`/`FAIL` verdict — deliberately free of precise floats (see `../src/main.cu`'s gate-tolerance comment block for why: thresholds carry wide margins over the measured numbers on the `[info]` lines above them). | Yes — stable. |
| `ARTIFACT:` | Confirms each `out/` file was written. | Yes — stable. |
| `RESULT:`   | Overall `PASS`/`FAIL` — every gate above must pass. The program exits nonzero on `FAIL`. | Yes — stable. |

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
