# Demo — 01.10 Rolling-shutter correction using IMU rates

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

The demo loads a captured rolling-shutter (RS) frame plus a clean and a degraded 200 Hz gyro trace,
integrates each gyro trace into a per-row rotation lookup table (host), runs the ONE GPU kernel in this
project (`rs_correct_kernel`) to resolve every output pixel's row-time fixed point and bilinearly resample
the RS frame, cross-checks the GPU result against an independent CPU twin, then runs eight independent
gates that check the correction is geometrically real (not just numerically self-consistent), converged,
and still helps even when the gyro is imperfect. It writes seven artifacts to `out/` (git-ignored,
regenerated every run):

- `rs_input.pgm` / `ground_truth_gs.pgm` — the two committed input frames, copied through for convenience.
- `corrected.pgm` — this project's reconstructed global-shutter-reference image (clean gyro).
- `uncorrected_diff.pgm` / `corrected_diff.pgm` — grayscale `|image - ground_truth|` heatmaps. Open both
  side by side: `uncorrected_diff.pgm` shows a bright, curved band tracing the scene's sheared marker
  line and cell-boundary edges; `corrected_diff.pgm` is visibly darker and flatter — the correction's
  effect, made directly visible.
- `rotation_profile.csv` — one row per output row: how many degrees of relative rotation that row needed
  (0 near the reference/middle row, growing toward the top/bottom — the RS skew's own shape).
- `gates_metrics.csv` — every gate's measured value, tolerance, and pass/fail, one row each.

**Reading the pictures**: `rs_input.pgm`'s single bright vertical marker line reads as a visible S-curve
(the "jello" effect, THEORY.md's taxonomy); `corrected.pgm`'s SAME line reads straight. That single
visual comparison is this project's whole lesson.

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, and every gate's measured number — varies by machine/run. | No. |
| `PROBLEM:`  | The exact problem instance (sizes, timing, gyro rate). | Yes — stable. |
| `DATA:`     | What sample was loaded. | Yes — stable. |
| `[time]`    | Host setup / GPU kernel / CPU twin timings — a **teaching artifact, never a benchmark claim**. | No. |
| `VERIFY:`   | GPU-vs-CPU agreement verdict (both gyro variants). | Yes — stable. |
| `GATE <name>:` | One of eight independent PASS/FAIL verdicts (README "Expected output" describes each). | Yes — stable. |
| `ARTIFACT:` | Confirms the seven `out/` files were written. | Yes — stable. |
| `RESULT:`   | Overall PASS/FAIL. The program exits nonzero on FAIL. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info, every `[info]` measurement) are allowed. `#`-prefixed lines in that file are
comments.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list).
- **`RESULT: FAIL`:** check which `GATE <name>:` line failed and read its `[info]` line for the measured
  number — `../THEORY.md` "How we verify correctness" explains what each gate proves and
  `../src/main.cu`'s tolerance-block comment records the reference-machine measurements the current
  tolerances were calibrated against.
- **`VERIFY: FAIL` specifically:** a real GPU-vs-CPU disagreement — start in `../src/kernels.cu`
  (`rs_correct_kernel`) and compare it side by side with `../src/reference_cpu.cpp` (`rs_correct_cpu`).
- **Expected-line mismatch only:** the program passed its own check but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The two
  are a contract; fix them together.
