# Demo — 01.11 Low-light denoising (bilateral, non-local means, fast BM3D variant)

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

Five GPU denoisers race a Gaussian-blur **negative control** on one heavily noisy 200x150 low-light
frame (peak signal 40 electrons at code value 255 — visibly grainy by design): **bilateral** (naive
global-memory + shared-memory tiled, bit-identical outputs), **NLM**, **BM3D-lite** (block-match ->
2-D DCT + 1-D Haar -> hard threshold -> invert -> aggregate), and the Gaussian baseline (bilateral's
spatial term alone). Every method runs on the GPU AND on an independent CPU oracle (VERIFY), then
every output is graded against the committed `clean.pgm` ground truth by five INDEPENDENT gates —
`psnr_improvement`, `edge_preservation` (where the Gaussian baseline is *required* to fail — the
designed negative control), `flat_noise_floor`, `method_ordering` (reported honestly, not forced),
and `noise_model_sanity` (checks the synthetic noise itself against the analytic Poisson+read-noise
prediction). Ten PGM images plus `gates_metrics.csv` land in `demo/out/` — open the four
`denoised_*.pgm`/`gaussian_baseline.pgm` files side by side against `noisy.pgm` and `clean.pgm`, and
the four `residual_*.pgm` heatmaps to see exactly WHERE each method still disagrees with ground truth
(mid-gray = zero error).

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, measured tolerances/PSNRs/fractions — varies by machine and is NOT a pass/fail line itself. | No. |
| `PROBLEM:`  | The exact problem instance (sizes, noise parameters). | Yes — stable (demo runs with no args). |
| `DATA:`     | What the committed sample is (synthetic, seeded). | Yes — stable. |
| `VERIFY(method):` | GPU-vs-CPU agreement for one method, within a documented tolerance (`../src/main.cu`). | Yes — stable. |
| `[time]`    | CPU reference ms, GPU kernel ms per method, tiling speed-up — a **teaching artifact, never a benchmark claim** (single-shot; first launches pay one-time init costs). | No. |
| `GATE name:` | `PASS`/`FAIL` verdict of one independent ground-truth check (`../THEORY.md` "How we verify correctness" explains each). | Yes — stable. |
| `ARTIFACT:` | Confirms every `demo/out/` file was written. | Yes — stable. |
| `RESULT:`   | Overall `PASS`/`FAIL` — VERIFY + all 5 gates. The program exits nonzero on `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info, per-gate `[info]` detail) are allowed. `#`-prefixed lines in that file are
comments.

## The artifacts (`demo/out/`)

| File | What it shows |
|---|---|
| `clean.pgm` / `noisy.pgm` | The committed ground truth and the noisy input, copied here for easy side-by-side viewing. |
| `denoised_bilateral.pgm` / `denoised_nlm.pgm` / `denoised_bm3d_lite.pgm` / `gaussian_baseline.pgm` | Each method's output. |
| `residual_*.pgm` | `denoised - clean`, mid-gray = 0, +-64 DN maps to black/white (`[info]` prints the exact cap and the RMS residual — the same number PSNR is computed from). |
| `gates_metrics.csv` | Every measured quantity behind every gate, machine-readable — one row per metric, with its tolerance and PASS/FAIL. |

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list).
- **`RESULT: FAIL`:** the GPU result disagreed with the CPU oracle — a real bug. Start in
  `../src/kernels.cu` and compare against `../src/reference_cpu.cpp`.
- **Expected-line mismatch only:** the program passed its own check but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The two
  are a contract; fix them together.
