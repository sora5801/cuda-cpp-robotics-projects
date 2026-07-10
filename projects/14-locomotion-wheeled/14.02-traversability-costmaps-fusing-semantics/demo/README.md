# Demo — 14.02 Traversability costmaps fusing semantics + geometry

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

The demo builds a 256x256 (25.6x25.6 m) off-road scenario from `data/sample/traversability_scenario.csv`
— rolling terrain, an 18-degree berm, a 0.5 m V-shaped ditch, a rock patch, a water pool sitting inside
geometrically flat ground, and a vegetation patch that is geometrically noisy but semantically benign —
then runs the four-kernel geometric+semantic fusion pipeline on the GPU, checks every kernel against a
CPU oracle (`VERIFY:`), and checks the FUSED result against the scenario's own known ground truth,
including the two DESIGNED-DISAGREEMENT cases that are this project's whole point: the water pool must
be vetoed despite near-perfect geometry, and the vegetation patch must be rescued to a valid, reduced-
speed cell despite meaningfully bad geometry. Three artifacts land in `demo/out/` (git-ignored, rebuilt
every run):

- **`traversability.pgm`** — a 256x256 grayscale image of the fused cost: WHITE (255) = free (cost 0),
  BLACK (0) = lethal/vetoed (cost 1). Open it in any image viewer; the water pool and the ditch wall
  should read as clearly dark, the vegetation patch as a mid-gray (valid but costed), the control/rock
  bands as mostly bright.
- **`speed_limit.pgm`** — the same 256x256 layout, grayscale-mapped from 0 (black, forced stop) to
  `kVMaxMps` (white, full 2.5 m/s cruise). Compare it side by side with `traversability.pgm`: every
  black cell in one is black in the other, but the speed map additionally shows GRADATION where the
  cost map alone would only show "valid vs. not" — exactly the point of the speed-limit layer.
- **`layers.csv`** — the teaching artifact: every one of ~301 points along a hand-designed transect
  that walks through all six scenario features (control -> berm -> ditch -> rock patch -> water pool ->
  vegetation patch), with EVERY intermediate layer's value at that point (elevation, slope, step
  height, roughness, geo_cost, semantic class + confidence + semantic_cost, fused_cost, veto_reason,
  speed_limit). Plot any column against `sample_index` to watch the two channels diverge exactly where
  the scenario designs them to (README Exercise 1).

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, scenario file path, per-stage VERIFY/gate diagnostics with the actual measured numbers — varies by machine and is where the two designed-disagreement cases' real numbers are printed. | No. |
| `PROBLEM:`  | The exact problem instance (grid size, windows, the two derived hard-veto limits, fusion weights). | Yes — stable (demo runs with no args). |
| `SCENARIO:` | The parsed scenario summary (berm angle, ditch depth, rock count, semantic region count). | Yes — stable. |
| `[time]`    | CPU reference ms, GPU kernel ms per stage — a **teaching artifact, never a benchmark claim** (single-shot, kernel-only vs. one CPU core; first launches pay one-time init costs). | No. |
| `VERIFY:`   | `PASS`/`FAIL` verdict of the four stage-isolated GPU-vs-CPU kernel checks. | Yes — stable. |
| `ARTIFACT:` | Confirms the three `demo/out/` files were written, with row/pixel counts. | Yes — stable. |
| `RESULT:`   | Final `PASS`/`FAIL` verdict (VERIFY plus all four analytic gates plus artifact writes). The program exits nonzero on `FAIL`. | Yes — stable. |

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
