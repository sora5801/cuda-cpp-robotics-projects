# Demo — 01.17 Camera-LiDAR / camera-camera extrinsic calibration (batched reprojection-error optimization)

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

Nine verification stages, run in sequence against three committed correspondence sets (README "Data"):
an independent calculus check (analytic vs. numeric Jacobian), three GPU-vs-CPU twin comparisons at
increasing scope (one assembly, one full trajectory, a multi-start subset), the convergence-basin study
(1024 randomized LM restarts), two extrinsic-recovery accuracy gates (camera-LiDAR and camera-camera),
a noise-scaling sanity check, the project's central DEGENERACY lesson (pose-diverse vs. near-coplanar
target poses), and a zero-noise exactness anchor. Every gate prints one stable `PASS`/`FAIL` line; the
exact measured number behind it is on the adjacent `[info]` line (never in the stable line itself — see
`../src/main.cu`'s "Output contract" comment for why).

Four artifacts land in `out/` (git-ignored; regenerated every run):

| Artifact | Contents |
|----------|----------|
| `convergence_curves.csv` | LM loss vs. iteration, GPU-orchestrated and CPU-only trajectories side by side (Stage C) |
| `basin_scatter.csv` | 1024 rows: each multi-start's initial rotation/translation perturbation magnitude, final loss, final pose error, and whether it converged (Stage D) |
| `gates_metrics.csv` | every gate's key numeric result, one row per metric |
| `overlay.ppm` | the "money shot" — the synthetic camera background with the camera-LiDAR correspondences' TRUE detected pixels (green), the PRE-calibration reprojection using a deliberately offset "rough prior" (red), and the POST-calibration reprojection using the best-of-1024-multistart recovered extrinsic (blue) |

**What to notice in `overlay.ppm`:** open it in any PPM-capable viewer (or convert with
`python -c "from PIL import Image; Image.open('out/overlay.ppm').save('overlay.png')"`). The blue
crosses should sit almost exactly on top of the green crosses (calibration recovered the extrinsic to
within a few pixels — the measured recovery error, `../THEORY.md`'s results table, corresponds to
roughly 1-2 px of reprojection error at this project's camera resolution), while the red crosses are
visibly scattered away from green (the rough-prior offset before calibration). Where blue fully
overlaps green, only a small green fringe or nothing peeks out — that overlap IS the calibration working.

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, and every gate's actual measured number (deviation, error, condition ratio, etc.) — deterministic on a given GPU architecture, but not diffed since FP rounding can shift by a bit or two across architectures (`../src/main.cu`'s "Output contract"). | No. |
| `PROBLEM:`  | The exact problem instance (view/fiducial/correspondence counts, image size, intrinsics, LM/multi-start settings) — all compile-time constants, so fully stable. | Yes — stable. |
| `SCENARIO:` | Names the three cohorts and confirms `[synthetic]`. | Yes — stable. |
| `[time]`    | The multi-start farm's GPU kernel time (K=1024 trajectories) — a **teaching artifact, never a benchmark claim**. | No. |
| `*_CHECK:` / `*_TWIN:` / `BASIN:` / `RECOVERY_*:` / `NOISE_SCALING:` / `DEGENERACY:` / `ZERO_NOISE_*:` | Each verification stage's `PASS`/`FAIL` verdict, with a plain-text description of what it checks (no numbers — see the `[info]` line immediately above each for the measured value and tolerance). | Yes — stable. |
| `ARTIFACT:` | Confirms a file was written to `out/`. | Yes — stable (row counts that depend on floating-point convergence behavior, like the LM iteration count, are reported on a separate `[info]` line instead — see `../src/main.cu`). |
| `RESULT:`   | Overall `PASS`/`FAIL` — `PASS` only if all nine stages passed. The program exits nonzero on `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info, every measured `[info]` number) are allowed. `#`-prefixed lines in that file are
comments — including a summary of the actual measured numbers behind every gate, for a reader who wants
them without running the demo.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list).
- **A `_TWIN` gate fails:** the GPU result disagreed with the CPU oracle by more than its documented
  (measured-then-margined) tolerance — a real bug. Start in `../src/kernels.cu` (the GPU kernel) and
  compare against `../src/reference_cpu.cpp` (the CPU oracle); both files' headers explain exactly what
  is and is not shared between them.
- **`JACOBIAN_CHECK` fails:** the analytic Jacobian formula in `residual_and_jacobian()`
  (`../src/kernels.cuh`) disagrees with a numeric central difference — check the derivation in
  `../THEORY.md` "The math" against the code line by line.
- **`RECOVERY_*`/`BASIN`/`DEGENERACY`/`NOISE_SCALING` fails:** the LM solver itself is behaving
  differently than measured (different GPU architecture, changed hyperparameters in
  `../src/kernels.cuh`, or a changed correspondence-generation seed in
  `../scripts/make_synthetic.py`) — re-read the `[info]` lines for the actual numbers and compare
  against `../THEORY.md`'s "How we verify correctness" results table.
- **Expected-line mismatch only:** the program passed its own checks but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The two
  are a contract; fix them together.
