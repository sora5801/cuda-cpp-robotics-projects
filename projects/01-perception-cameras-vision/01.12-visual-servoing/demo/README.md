# Demo — 01.12 Visual servoing: image-Jacobian control loop entirely on GPU

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

**A classic robot controller, studied 4096 ways at once.** The demo simulates 4096 independent
eye-in-hand IBVS servo loops in parallel on the GPU — one thread per loop, up to 400 control steps
each — from camera poses spread across three designed cohorts (an everyday "nominal" convergence
region, a small pure-translation cohort for measuring the exponential convergence rate, and a
near-180-degree-rotation cohort designed to trigger the classic IBVS "camera retreat" failure). It
runs the SAME batch through three controller variants (true-depth, fixed-depth, and the
desired-Jacobian scheme) to make the fixed-depth approximation's robustness quantitative rather than
anecdotal.

Two kinds of checks gate the verdict:

1. **VERIFY** — the §5 GPU-vs-CPU gate at three grains: a single loop's full trajectory, the
   Jacobian/pseudoinverse linear algebra at 16 sampled poses, and a 128-loop batch-statistics subset.
2. **GATE** — three INDEPENDENT checks that do not route through either twin (they check the
   controller against control-theory predictions and the visual-servoing literature, not against the
   other implementation): `exponential_decay`, `convergence_basin`, `retreat_pathology`.

**This demo writes five artifacts** into `out/` (git-ignored, regenerated each run):

| File | What it shows |
|------|----------------|
| `image_plane_traces.ppm` | 8 documented loops' feature paths in the normalized image plane — green squares mark the goal features, colored paths show each traced loop's 4 points converging (or, for the retreat cohort, doing something stranger). |
| `basin_map.ppm` | A 64x64 grid over initial (dx,dy) position offset, colored by convergence (green) / non-convergence (red) and shaded by steps-to-converge — the convergence basin, made visible. **Measured finding, stated honestly:** at this controller's tuning, PURE TRANSLATION (this artifact's slice — zero rotation, ±0.30 m) converges 100% of the time out to the edge of the grid; it stays entirely green. Contrast that with the RETREAT cohort (near-180-degree ROTATION, `image_plane_traces.ppm`), which fails ~100% of the time — a clean, measured illustration that for this controller, translation alone is forgiving and rotation is where IBVS's real difficulty lives (`THEORY.md` derives why). |
| `error_decay.csv` | The exponential-decay cohort's mean feature-error norm per step, alongside the theoretical exp(-lambda*t) prediction. |
| `batch_stats.csv` | Per-variant, per-cohort convergence percentage and median steps-to-converge. |
| `gates_metrics.csv` | Flat provenance table: every gate's measured value, threshold, and pass/fail. |

Open the PPMs with any viewer that reads binary PPM (P6) directly, or convert with any tool (e.g.
`magick convert image_plane_traces.ppm image_plane_traces.png`).

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, and every MEASURED number this run produced (fit rates, percentages, correlations) — varies run to run and platform to platform (see the determinism note in `../src/main.cu`). | No. |
| `PROBLEM:` / `SCENARIO:` | The exact problem instance (sizes, cohort split, controller constants). | Yes — stable (demo runs with no args). |
| `[time]`    | GPU kernel time across all three variants — a **teaching artifact, never a benchmark claim**. | No. |
| `VERIFY:`   | `PASS`/`FAIL` verdict of a GPU-vs-CPU twin (tolerances documented in `../src/main.cu` and `THEORY.md`). | Yes — stable. |
| `GATE <name>:` | `PASS`/`FAIL` verdict of an independent control-theory/literature gate. | Yes — stable. |
| `ARTIFACT:` | Confirms a file was written to `demo/out/`. | Yes — stable (the filenames; not the byte contents). |
| `RESULT:`   | Overall `PASS`/`FAIL`. The program exits nonzero on `FAIL`. | Yes — stable. |

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
