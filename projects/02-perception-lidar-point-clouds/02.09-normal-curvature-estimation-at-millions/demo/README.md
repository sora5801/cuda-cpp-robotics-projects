# Demo — 02.09 Normal + curvature estimation at millions of points/sec

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

Two passes, back to back:

1. **Correctness pass** (N = 8,400 points, `data/sample/normals_scan.bin` — 12 analytic-surface
   cohorts: plane / sphere / cylinder / edge × none / low(3 mm) / high(15 mm) noise). Builds the
   voxel-hash neighbor index, runs the fused per-point KNN → covariance → Jacobi eigensolve → normal →
   curvature → degeneracy pipeline on the GPU, runs THREE independently-coded CPU/brute-force oracles,
   and checks every `VERIFY`/`GATE` line — including angular error against the surfaces' **closed-form**
   analytic normals and curvatures, a cylinder-axis fit from the estimated normals alone, sensor-facing
   orientation, the plane<cylinder<sphere<edge curvature ordering, and degeneracy-flag rates.
2. **Throughput pass** (N = 1,050,000 points — 125 translated+jittered copies of the same committed
   sample, `build_throughput_cloud()` in `../src/main.cu`; methodology stated honestly, not disguised as
   a fresh scan). GPU-only (no CPU twin at this scale — correctness is already established in pass 1);
   measures Mpts/s for the index build and the fused pipeline, the catalog's "millions of points/sec"
   promise, made measurable.

Artifacts written to `out/` (git-ignored, regenerated every run):

- `normal_map.ppm` — top-view point splat colored by estimated normal (`(nx,ny,nz)*0.5+0.5 -> RGB`),
  the classic normal-map visualization.
- `curvature_heatmap.ppm` — top-view, colored blue (flat) to red (high surface variation), clamped at
  0.15 — chosen against this project's own measured curvature scale (see `THEORY.md`).
- `degeneracy_map.ppm` — top-view, gray = clean, red = edge/corner-flagged, yellow = isolated
  (insufficient neighbors even after ring-`kMaxRing` expansion).
- `per_cohort_errors.csv` — every cohort's mean/max angular error, mean/median curvature, and
  degeneracy counts, in one table.
- `gates_metrics.csv` — every measured number behind every `VERIFY`/`GATE`/`[info]` line, for anyone who
  wants the raw numbers without re-running.

All three PPMs use the SAME top-view frame (world x/y of the committed sample, auto-fit with a 5%
margin) — open them side by side to see how normal direction, curvature, and degeneracy flags line up
spatially. Each analytic-surface patch is only a few meters across against a scene spanning tens of
meters, so the patches appear as small clusters — zoom in on any image viewer to see individual points.

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, and every MEASURED number (angular-error diffs, per-cohort means, throughput Mpts/s) — deliberately excluded from the diff even where deterministic, because a few are architecture/timing-sensitive (CLAUDE.md §12). | No. |
| `PROBLEM:` / `DATA:` | The exact problem instance and data provenance. | Yes — stable. |
| `VERIFY(...)`| GPU-vs-independent-CPU-twin agreement (KNN sets, eigenvalues, normals, curvature, degeneracy) within a documented tolerance. | Yes — the PASS/FAIL verdict and tolerance text; the measured diff itself is a separate `[info]` line. |
| `GATE ...`  | An INDEPENDENT check against analytic ground truth or a structural/aggregate invariant (brute-force anchor, per-surface angular error, cylinder-axis fit, orientation, curvature ordering, degeneracy rates, throughput). | Yes — the PASS/FAIL verdict, thresholds, and any exact integer counts; volatile float measurements are on `[info]` lines. |
| `[time]`    | Stage timings, ms — a **teaching artifact, never a benchmark claim** (single-shot; first launches pay one-time init costs). | No. |
| `ARTIFACT:` | Which files were written to `out/`. | Yes — stable. |
| `RESULT:`   | Overall `PASS`/`FAIL` verdict. The program exits nonzero on `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, info) are allowed. `#`-prefixed lines in that file are comments.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list). This project's `kernels.cu` uses Thrust, which
  needs the `/Zc:preprocessor /Zc:__cplusplus` flags already wired into `../build/*.vcxproj` — see that
  file's `CudaCompile` comments if you are retargeting the project elsewhere.
- **`RESULT: FAIL`:** a `VERIFY` line names a GPU-vs-CPU disagreement (a real implementation bug — start
  in `../src/kernels.cu` and compare against `../src/reference_cpu.cpp`); a `GATE` line names a
  disagreement with analytic truth or an aggregate invariant (check `../src/main.cu`'s gate logic and
  `../scripts/make_synthetic.py`'s geometry first).
- **Expected-line mismatch only:** the program passed its own check but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The two
  are a contract; fix them together.
