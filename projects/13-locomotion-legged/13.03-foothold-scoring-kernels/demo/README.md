# Demo — 13.03 Foothold scoring kernels: slope, roughness, edge distance from elevation maps

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

**A quadruped's foot planner deciding where NOT to step.** The demo builds a synthetic 256x256
(5.12x5.12 m) elevation map — flat ground, a 15-degree ramp, a 12 cm step, 16 scattered rocks, and a
rectangular sensor dropout (a NaN "hole") — then runs the four scoring kernels (slope, roughness,
edge distance, fusion) over every one of its 65536 cells, and finally scores 1000 candidate footholds
along a path that deliberately crosses every one of those features. You can *watch* the pipeline
refuse to stand on the step's edge, discount the rocky patch, and steer every query near the hole to
solid ground nearby — that behavior is the entire point of the project.

**Two artifacts, both regenerated every run (git-ignored, `out/`):**
- `out/foothold_score.pgm` — the fused [0,1] score map as a 256x256 grayscale image (white = great
  foothold, black = vetoed). Open it in any image viewer; the ramp, step, rocks, and hole are all
  visible as bright/dark regions.
- `out/selected_footholds.csv` — one row per query: nominal point, selected cell, its score, whether
  it was valid, and how far the selection moved from the nominal point. Plot `x_nom_m,y_nom_m` against
  `x_sel_m,y_sel_m` to see the selector's corrections.

Four independent VERIFY gates (one per kernel, each fed identical pinned inputs on the GPU and CPU
paths — see `../src/main.cu`'s file header for why) must pass, THEN four analytic gates against the
terrain's own known ground truth (a flat region's near-zero slope, the ramp's constructed angle, the
step's hazard band, every selection's validity) must pass, before `RESULT: PASS`.

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, scenario path, and per-stage measured diffs/gate readouts — the actual numbers vary slightly by GPU/compiler. | No. |
| `PROBLEM:`  | The fixed problem instance (grid size, fit radius, friction-derived slope limit, search radii). | Yes — stable (demo runs with no args). |
| `SCENARIO:` | The composed terrain and query count loaded from `data/sample/terrain_scenario.csv`. | Yes — stable. |
| `VERIFY:`   | `PASS`/`FAIL` verdict of the four stage-isolated GPU-vs-CPU kernel gates (tolerances documented in `../src/main.cu` and `THEORY.md`). | Yes — stable. |
| `ARTIFACT:` | Confirms both output files were written, with their sizes/row counts. | Yes — stable. |
| `RESULT:`   | Final `PASS`/`FAIL`: VERIFY **and** the four analytic terrain gates **and** the artifact writes. The program exits nonzero on `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info) are allowed. `#`-prefixed lines in that file are comments.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list).
- **`RESULT: FAIL`:** either `VERIFY: FAIL` (a GPU kernel disagreed with its CPU oracle — start in
  `../src/kernels.cu` and compare against `../src/reference_cpu.cpp`) or an analytic terrain gate
  failed (the `[info] gate A/B/C/D` lines print the measured value against its bound — start in
  `../src/main.cu`'s ANALYTIC GATES section).
- **Expected-line mismatch only:** the program passed its own check but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The two
  are a contract; fix them together.
