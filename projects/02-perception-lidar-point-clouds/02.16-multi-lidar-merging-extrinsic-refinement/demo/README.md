# Demo — 02.16 Multi-LiDAR merging + extrinsic refinement

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

Loads two committed cohorts — `aligned.csv` (a 3-LiDAR rig with every sensor exactly at its
as-designed mounting pose) and `drifted.csv` (the SAME yard scene, the SAME sensor assignment, but
the two front-corner LiDARs — LEFT and RIGHT — carry a small, known, undisclosed-to-the-solver
mounting drift, ~0.8° / 3 cm each, a different drift vector per sensor) — then runs ten independent
verification stages: two basic GPU-vs-CPU correctness twins (the merge transform, and per-zone plane
fitting), a **drift-detection** gate (plane-pair residuals expose the drifted rig and clear the
aligned rig), two more twins on the point-to-plane refinement machinery (one-shot assembly, and a
full 20-iteration Levenberg-Marquardt trajectory), the **recovery** headline (does refinement
actually find each sensor's true drift?), a **validation loop** (do the plane residuals fall back
under threshold after refinement?), an **observability** contrast (one planar zone vs three
mutually-orthogonal zones — the same "coplanar poses" lesson 01.17 teaches for camera calibration,
recast for LiDAR planes), a **zero-drift control** (refining the ALREADY-aligned rig must not
invent a correction), and **dedup accounting** (does voxel-grid deduplication of the merged cloud
agree exactly between GPU and CPU?).

**Artifacts** (all under `demo/out/`, written every run):

- `topview_before.ppm` / `topview_after.ppm` — a top-down (bird's-eye) view of the full merged
  drifted-rig cloud, colored by sensor (gray = MAIN, red = LEFT, blue = RIGHT), using the
  currently-believed (nominal) extrinsics vs. the refined ones. At this 26-meter scale the
  centimeter-scale drift is genuinely sub-pixel — these two are for SCENE CONTEXT.
- `topview_zoom_before.ppm` / `topview_zoom_after.ppm` — **the money shot**: a tight zoom on
  `wall_front`'s shared patch (the one surface all three sensors see). Before refinement, LEFT's red
  dots and RIGHT's blue dots visibly sit apart from MAIN's gray line; after refinement, all three
  collapse onto (very nearly) the same line.
- `plane_residuals.csv` — every plane-pair angle/offset residual computed during detection and
  validation, one row per (stage, cohort, sensor, surface).
- `gates_metrics.csv` — every gate's exact measured number(s), one row per metric.

If it prints `RESULT: PASS`, every stage agreed with its independent check, on the numbers actually
measured on this run.

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name and compute capability — varies by machine. | No. |
| `PROBLEM:`  | The exact problem instance (sizes, parameters). | Yes — stable (demo runs with no args). |
| `[time]`    | CPU reference ms, GPU kernel ms, and a speed-up figure — a **teaching artifact, never a benchmark claim** (single-shot, kernel-only vs. one CPU core; first launches pay one-time init costs). | No. |
| `RESULT:`   | `PASS`/`FAIL` verdict of the GPU-vs-CPU check (tolerance documented in `../src/main.cu` and `THEORY.md`). The program exits nonzero on `FAIL`. | Yes — stable. |

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
