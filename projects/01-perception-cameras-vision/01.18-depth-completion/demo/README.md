# Demo — 01.18 Depth completion: sparse LiDAR + RGB → dense depth

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

The demo loads the committed synthetic sample (a camera image, a sparse LiDAR point cloud, and —
evaluation-only — the scene's exact dense depth), runs the full pipeline on both the GPU and an
independent CPU oracle, and then grades the result against ground truth with five named gates:

1. **VERIFY** (four lines) — projection+z-buffer, conductance, diffusion, and IDW each run on the GPU
   AND on a from-scratch CPU implementation; the two must agree within a documented tolerance
   (CLAUDE.md §5). This is a correctness check, not an accuracy claim.
2. **STABILITY** — the forward-Euler diffusion step's CFL-style bound (`dt <= 0.25`) is enforced at
   *compile time* by a `static_assert` in `src/kernels.cuh`; this line just reports the numbers.
3. **GATE: overall_accuracy** — the edge-aware diffusion method's mean absolute error against the
   scene's exact truth, over every pixel that has one.
4. **GATE: edge_quality** — the *reason this method exists*: at real depth discontinuities the RGB
   image also shows, diffusion must beat the RGB-blind IDW baseline by a measured margin.
5. **GATE: texture_trap** — on a flat, high-contrast checkerboard patch, the edge-aware method must
   NOT hallucinate depth structure from texture (bounded degradation vs. the texture-blind IDW).
6. **GATE: camo_edge_honesty** — NOT a "guided is good" claim: on a low-contrast REAL depth edge (two
   identically-colored surfaces at different depths), error must measurably *exceed* the ordinary
   flat-region error — proving the demo actually exhibits the "RGB edge implies depth edge" prior's
   failure mode rather than hiding it.
7. **GATE: input_fidelity** — every pixel with a real LiDAR sample must keep that exact value
   (Dirichlet anchoring holds).

`[info]` lines (not diffed) report the actual measured numbers behind every gate — MAE/RMSE for both
methods, region pixel counts, and a `[info]`-only density sweep (~1.7% / ~4.4% / ~6.0% LiDAR density)
showing accuracy improves monotonically with density.

**Artifacts** (`demo/out/`, all regenerated every run): `rgb.ppm` (the input image),
`sparse_depth_vis.pgm` (the raw LiDAR samples, dilated for visibility), `completed_guided.pgm` /
`completed_idw.pgm` (the two densified depth fields), `truth_depth.pgm` (ground truth),
`error_guided.pgm` / `error_idw.pgm` (per-pixel `|completed - truth|` heatmaps), and
`gates_metrics.csv` (every gate's raw numbers). All depth PGMs share one convention: **near = bright**
(linear map over 2–20 m; no-data pixels are black). Error PGMs use the opposite sense on purpose
(**bright = more error**) so a depth map and an error map can never be mistaken for each other.

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, measured MAE/RMSE, region pixel counts, density sweep — carries every actual NUMBER. | No — GPU-measured values can vary by architecture. |
| `PROBLEM:`  | The exact problem instance (image size, LiDAR return count, diffusion iterations). | Yes — stable (demo runs with no args). |
| `[time]`    | CPU reference ms, GPU kernel ms, and a speed-up figure — a **teaching artifact, never a benchmark claim** (single-shot; first launches pay one-time init costs). | No. |
| `VERIFY:`   | GPU-vs-CPU twin agreement per pipeline stage (correctness, not accuracy). Numbers live on the preceding `[info]` line; this line is verdict-only text so it stays identical across GPUs. | Yes — stable. |
| `STABILITY:`| The diffusion forward-Euler CFL bound check (both numbers are compile-time constants). | Yes — stable. |
| `GATE:`     | One of five evaluation gates (see above) — verdict-only text; numbers live on the paired `[info]` line. | Yes — stable. |
| `ARTIFACT:` | Confirms the `demo/out/` files were written. | Yes — stable. |
| `RESULT:`   | Overall `PASS`/`FAIL` — every VERIFY twin agreed AND every GATE passed. The program exits nonzero on `FAIL`. | Yes — stable. |

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
