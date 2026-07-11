# Demo — 01.09 Photometric/vignetting calibration kernels

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

Loads 16 dark frames + 16 flat frames + one natural test scene (`../data/sample/`), runs the GPU
calibration pipeline (dark-stack mean -> flat-stack mean -> dark-subtract -> center-normalize -> radial
histogram -> parametric least-squares fit -> per-pixel correction), cross-checks every GPU stage against
an independent CPU reference, and grades the result against six gates tied to the scene's known ground
truth: how well the additive DSNU field was recovered, how well the multiplicative gain field was
recovered, whether the fitted radial curve matches the true cos^4 vignette (and whether its residual is
consistent with being PRNU, not fit error), whether averaging N=1/4/16 frames reduces noise like the
textbook `1/sqrt(N)` law, whether five identical-radiance swatches (one center, four corners) read as
equal AFTER correction (they read up to ~26% apart before it), and whether a corrected flat frame reads
uniform.

**What to look at:** open `out/scene_uncorrected.pgm` next to `out/scene_corrected.pgm` — the corners are
visibly darker before correction and visually uniform after. `out/radial_profile.csv` plots the true
cos^4 curve, the raw (nonparametric) binned gain profile, and the fitted parametric curve together — they
should overlay almost exactly. `out/gates_metrics.csv` has every measured number behind every `GATE` line
below.

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, and every measured number (per-stage diffs, the fitted radial coefficients, per-gate metrics) — varies by machine/run. | No. |
| `PROBLEM:`  | The exact problem instance (image size, sensor model, stack sizes). | Yes — stable (demo runs with no args). |
| `DATA:`     | A one-line description of the loaded calibration rig + test scene. | Yes — stable. |
| `[time]`    | GPU-pipeline and CPU-oracle timings — a **teaching artifact, never a benchmark claim** (these images are tiny; both paths finish in well under a millisecond of kernel time). | No. |
| `VERIFY:`   | `PASS`/`FAIL` verdict of the GPU-vs-CPU cross-check across all pipeline stages (tolerances documented in `../src/main.cu` and `THEORY.md`). | Yes — stable. |
| `GATE ...:` | `PASS`/`FAIL` for each of the six independent ground-truth gates (`dsnu_recovery`, `gain_recovery`, `radial_fit`, `noise_averaging`, `correction_efficacy`, `flatness`). | Yes — stable (six lines). |
| `ARTIFACT:` | Confirms every file in `out/` was written. | Yes — stable. |
| `RESULT:`   | `PASS`/`FAIL` verdict of the WHOLE demo (VERIFY + all six gates). The program exits nonzero on `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info) are allowed. `#`-prefixed lines in that file are comments.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list).
- **`VERIFY: FAIL`:** the GPU result disagreed with the CPU oracle on at least one stage — a real bug.
  Start in `../src/kernels.cu` and compare the failing stage against its twin in `../src/reference_cpu.cpp`.
- **A `GATE ...: FAIL`:** VERIFY passed (GPU and CPU agree with each other) but the AGREED result
  disagrees with ground truth — read that gate's `[info]` line and its section in `../src/main.cu` and
  `../THEORY.md` "How we verify correctness" for what it measures and why.
- **Expected-line mismatch only:** the program passed its own check but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The two
  are a contract; fix them together.
