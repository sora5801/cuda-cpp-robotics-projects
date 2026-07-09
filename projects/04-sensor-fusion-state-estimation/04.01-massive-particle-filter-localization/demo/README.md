# Demo — 04.01 Massive particle filter localization (10⁵–10⁶ particles, GPU likelihoods + resampling)

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

**A GPU keeping 100,000 pose hypotheses alive at sensor rate.** The demo runs a bootstrap
particle filter closed-loop on the committed synthetic sample: a 64×64 occupancy-grid world and a
120-step loop drive with noisy odometry and noisy 16-beam range scans. Every 100 ms step, the GPU
**predicts** all 100,000 particles through the odometry twist (each with its own noise, from an
in-kernel counter-based RNG) and **weights** each one by ray-casting 16 beams into the map —
~10⁸ occupancy lookups per scan, ~1.3 ms of kernel time where one CPU core needs ~340 ms. The
host then takes the weighted mean as the pose estimate and systematically resamples. Try
`--particles 1000000` to see the catalog's upper bound run (~6.4 ms/scan measured here).

Two checks gate the verdict:

1. **VERIFY** — the §5 GPU-vs-CPU gate, kernel by kernel on step 0's inputs: predict poses must
   agree within abs 1e-4 (measured ~5e-7), and weight log-likelihoods on identical poses within
   rel 1e-3 (measured ~2e-7).
2. **RESULT** — the estimation check: position RMSE of the estimate vs the (synthetic,
   fully-known) ground truth over all 120 steps must beat 0.15 m (it typically lands near
   0.02 m).

**This demo writes an artifact**: `out/trajectory_est.csv` (git-ignored, regenerated each run)
with columns `step, t_s, gt_x_m, gt_y_m, gt_theta_rad, est_x_m, est_y_m, est_theta_rad,
err_pos_m` — plot `est_x/est_y` over `gt_x/gt_y` with any tool and you can *see* the estimate
hug the true rounded-square loop; plot `err_pos_m` vs `t_s` to watch the initial 0.3 m cloud
collapse within the first few scans. That plot is the picture this project exists to produce.

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, verify deviations, RMSE/ESS diagnostics — vary by machine. | No. |
| `PROBLEM:`  | The exact problem instance (K, beams, map size). | Yes — stable (demo runs with no args). |
| `SAMPLE:`   | What was loaded from `../data/sample/` (labeled synthetic). | Yes — stable. |
| `[time]`    | CPU reference ms, GPU kernel ms, and a speed-up figure — a **teaching artifact, never a benchmark claim** (single-shot, kernel-only vs. one CPU core; first launches pay one-time init costs). | No. |
| `VERIFY:`   | `PASS`/`FAIL` of the GPU-vs-CPU gate (tolerances documented in `../src/main.cu` and `THEORY.md`). Nonzero exit on `FAIL`. | Yes — stable. |
| `ARTIFACT:` | The CSV written to `out/`. | Yes — stable. |
| `RESULT:`   | `PASS`/`FAIL` of the closed-loop RMSE gate. The program exits nonzero on `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info) are allowed. `#`-prefixed lines in that file are comments.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list).
- **`VERIFY: FAIL`:** the GPU kernels disagreed with the CPU oracle — a real bug. Start in
  `../src/kernels.cu` and diff against `../src/reference_cpu.cpp` (they are line-by-line twins).
- **`RESULT: FAIL` with `VERIFY: PASS`:** every kernel is right but the *filter* lost the robot —
  look at the `[info]` ESS line (weight collapse?) and plot the artifact to see where the
  estimate diverged; the usual suspects are the resampler and the noise sigmas.
- **Expected-line mismatch only:** the program passed its own checks but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The
  two are a contract; fix them together.
