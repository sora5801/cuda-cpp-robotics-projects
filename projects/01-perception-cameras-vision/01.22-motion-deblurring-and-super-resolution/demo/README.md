# Demo — 01.22 Motion deblurring and super-resolution for inspection zoom

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

Two inverse problems on the SAME synthetic inspection scene (`data/sample/truth.pgm`):

**Milestone 1 — motion deblurring.** `blurred.pgm` (the scene convolved with a known 9px/20deg
line PSF plus sensor noise) is restored three ways: a naive frequency-domain inverse filter (a
DESIGNED FAILURE — it amplifies noise catastrophically at the PSF's spectral near-zeros, ending up
*worse* than the blurred input), a Wiener filter (the same inverse, regularized — a clean
improvement), and Richardson-Lucy (a spatial-domain iterative alternative). A fourth run
deconvolves the identical blurred frame with a PSF rotated 25 degrees from the truth, showing how
badly non-blind deconvolution degrades when the assumed motion is wrong.

**Milestone 2 — multi-frame super-resolution.** 8 low-resolution frames, each a genuinely aliased
capture at a known quarter-pixel shift, are combined by shift-and-add onto a 2x grid and refined
by iterative back-projection (IBP), then compared against bicubic upscaling of a single frame. The
"money shot" is `bar_chart_comparison.pgm`: a bar pattern below the low-res grid's Nyquist limit
that only multi-frame SR can resolve correctly — bicubic reproduces a similar-looking but WRONG
(aliased/moire) pattern, which is why the `sr_resolution` gate checks pattern *correlation* against
ground truth, not raw contrast (see `../src/main.cu`'s `bar_pattern_correlation()` comment for the
measurement that surfaced this).

Everything the demo writes lands in `demo/out/`:

| File | What it shows |
|------|----------------|
| `truth.pgm` | The ground-truth inspection scene (never seen by any restoration method). |
| `blurred.pgm` | Milestone 1's input: truth convolved with the line PSF + Gaussian noise. |
| `naive_inverse.pgm` | The designed failure — visibly explodes into noise. |
| `wiener.pgm` | The regularized recovery — sharper than blurred, without the explosion. |
| `rl.pgm` | Richardson-Lucy's spatial-domain recovery after 30 iterations. |
| `lr_frame_0.pgm` | One of the 8 low-res captures (the zero-shift reference frame). |
| `bicubic.pgm` | Milestone 2's baseline: single-frame bicubic upscale of `lr_frame_0.pgm`. |
| `sr.pgm` | The multi-frame super-resolved result (shift-and-add + 12 IBP iterations). |
| `bar_chart_comparison.pgm` | Truth / bicubic / SR crops of the aliased bar group, side by side. |
| `rl_convergence.csv` | Richardson-Lucy's per-iteration data-fidelity MSE. |
| `ibp_convergence.csv` | IBP's per-iteration reprojection RMS (must fall monotonically). |
| `gates_metrics.csv` | Every VERIFY/GATE measurement in one machine-readable table. |

## How to read the output

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, and every MEASURED number (PSNRs, correlations, noise stds, tolerances) — varies slightly by machine/GPU architecture. | No. |
| `PROBLEM:`  | The exact problem instance (sizes, PSF, frame count). | Yes — stable (demo runs with no args). |
| `DATA:`     | What the committed sample contains. | Yes — stable. |
| `[time]`    | CPU reference ms, GPU kernel ms — a **teaching artifact, never a benchmark claim** (single-shot; first launches pay one-time init/cuFFT-plan costs). | No. |
| `VERIFY(method):` | GPU-vs-independent-CPU-twin agreement for one restoration method (tolerance documented in `../src/main.cu`). | Yes — stable (PASS/FAIL text; the numeric diff sits on the paired `[info]` line). |
| `GATE name:` | One independent, ground-truth-based correctness/honesty check (never routed through the shared FFT/bilinear machinery — see `../src/reference_cpu.cpp`'s header). | Yes — stable. |
| `ARTIFACT:` | Confirms every `demo/out/` file was written. | Yes — stable. |
| `RESULT:`   | `PASS`/`FAIL` verdict: all 6 VERIFYs + all 6 gates + artifacts. The program exits nonzero on `FAIL`. | Yes — stable. |

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
