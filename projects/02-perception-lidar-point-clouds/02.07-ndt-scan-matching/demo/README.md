# Demo — 02.07 NDT scan matching (Autoware-style map localizer)

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

The demo registers a synthetic 16-beam LiDAR scan against a pre-built NDT voxel map of an
L-shaped corridor-into-a-room, using a coarse (2.0 m) then fine (1.0 m) multi-resolution Newton
optimizer on the GPU — and runs the SAME registration problem through a compact point-to-point
ICP contrast (CPU-only) from the SAME 240 perturbed initial poses, so the two algorithms' basins of
convergence can be measured side by side, honestly, on identical data.

It runs nine verification stages in order (`VOXEL_STATS_TWIN`, `JACOBIAN_CHECK`, `ASSEMBLY_TWIN`,
`TRAJECTORY_TWIN`, `SCORE_SANITY`, `CONVERGENCE`, `ACCURACY`, `BASIN_CONTRAST`,
`OUTLIER_ROBUSTNESS`), plus two **honesty-only** `[info]` reports (`failure_diagnosis` /
`bin0_corridor_axis_split` and `degenerate_axis`) that are never gated pass/fail. See
`../THEORY.md` "How we verify correctness" for what each stage checks and why.

**Artifacts** written to `demo/out/` (git-ignored; regenerated every run):

| File | What it is |
|------|------------|
| `registration_topview.csv` | Top-view (x,y) points labeled `MAP` / `SCAN_INITIAL` / `SCAN_REGISTERED` — the classic ICP/NDT before-after plot, for one representative cohort trial. |
| `basin_curve.csv` | Per-magnitude-bin converged-% for all three methods (`ndt_multires`, `ndt_fine_only`, `icp`) — plot this to see the basin-width story. |
| `convergence_trajectories.csv` | Per-iteration score for the STAGE D twin trial, both the GPU-orchestrated and the independent CPU trajectory. |
| `gates_metrics.csv` | Every gate's measured number, `gate,metric,value` rows — the machine-readable twin of the `[info]` lines below. |

**What to notice in the numbers** (measured on the reference machine; see
`expected_output.txt`'s header comment and `gates_metrics.csv` for the exact figures): the
smallest-perturbation bin (0.2 m / 5°) converges 65% of the time with the full multi-resolution
schedule versus **7.5%** running fine resolution alone with the identical total iteration budget —
the single clearest demonstration in this project that coarse voxels genuinely widen the basin of
attraction, not just add iterations. The `bin0_corridor_axis_split` `[info]` line then splits that
65% further: 0% for initial guesses offset along the corridor's own degenerate sliding axis, 84%
for every other direction — the smallest bin's shortfall from "near 100%" is almost entirely one
specific, measured, physically-explained geometric axis, not a diffuse optimizer weakness.
ICP is very strong at the smallest perturbations (100% at the same bin) and, measured honestly on
this project's small, simple scene, ICP also converges MORE of the whole cohort than NDT overall
(24.2% vs. 13.3%) — this project does not hide that result to tell a cleaner "NDT wins" story; see
`../THEORY.md` "Where this sits in the real world" and README "Limitations & honesty" for why, and
what scale would flip the comparison.

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, voxel-grid stats, per-gate measured numbers, the degenerate-axis report — varies by machine/run. | No. |
| `PROBLEM:` / `SCENARIO:` | The exact problem instance and data provenance. | Yes — stable. |
| `[time]`    | Voxel-build and ICP wall-clock timings — a **teaching artifact, never a benchmark claim**. | No. |
| `VOXEL_STATS_TWIN:` … `OUTLIER_ROBUSTNESS:` | Each stage's `PASS`/`FAIL` verdict. | Yes — stable. |
| `ARTIFACT:` | Confirms the four `demo/out/` files were written. | Yes — stable. |
| `RESULT:`   | Overall verdict; the program exits nonzero on any stage `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info, per-gate `[info]` numbers) are allowed. `#`-prefixed lines in that file are
comments — including the header block that records every gate's actual measured number, so the
numbers are documented without being part of the stability contract itself.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list).
- **A `_TWIN` stage `FAIL`s:** the GPU result disagreed with the CPU oracle — a real bug. Start in
  `../src/kernels.cu` and compare against `../src/reference_cpu.cpp`.
- **`CONVERGENCE`/`ACCURACY`/`BASIN_CONTRAST`/`OUTLIER_ROBUSTNESS` `FAIL`s:** the optimizer's
  behavior regressed relative to the measured-then-margined thresholds baked into `../src/main.cu`
  — see `../THEORY.md` "Numerical considerations" for the two real bugs this project's own
  development process caught this way (a damping-scale bug and a damping-symmetry bug), which is
  exactly the kind of regression these gates exist to catch.
- **Expected-line mismatch only:** the program passed its own checks but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The
  two are a contract; fix them together.
