# Demo — 22.01 100k-agent swarm simulator: flocking, pheromone grids, stigmergy

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

**100,000 boids finding each other with no leader, plus a pheromone trail they leave behind.** The
demo first spawns 100,000 agents at random positions with random headings inside a 256 m arena — a
maximally disordered start — then runs 300 steps (15 s of simulated swarm time) of: bin agents into a
uniform grid, apply Reynolds' separation/alignment/cohesion rules plus a weak pull along the pheromone
gradient, integrate, then diffuse-and-decay the pheromone field. What starts as noise visibly
organizes into moving flocks; the mean local velocity alignment climbs from ~0 (random) to ~0.97.

Before the headline run, the demo runs a **lockstep GPU-vs-CPU verification** at a small, fast
configuration (N = 4,096, 100 steps) — every step, both paths start from the identical shared state,
and their outputs are compared within a documented tolerance. This is what makes `VERIFY: PASS` a real
correctness proof rather than a vibe check: see [`../THEORY.md`](../THEORY.md) §How we verify
correctness for why lockstep (rather than free-running comparison) is necessary for a chaotic system.

**This demo writes three artifacts** to `out/` (git-ignored, regenerated each run):

- `density.pgm` — a 256×256 grayscale heatmap of agent density per cell (sqrt-scaled so sparse regions
  stay visible next to dense flock cores). Open it in any image viewer to *see* where the flocks ended
  up.
- `pheromone.pgm` — the stigmergy field itself: brighter cells are where agents have recently and
  repeatedly passed. Compare it side-by-side with `density.pgm` — the pheromone trail traces where
  flocks have *been*, the density map shows where they *are*.
- `positions.csv` — the first 1,000 agents' final state (`id,x_m,y_m,vx_ms,vy_ms`), plottable with
  anything, for a quantitative look instead of a heatmap.

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, scenario file path, and measured verify/final-metric numbers — all vary by machine or by the documented atomic-ordering nondeterminism (`../THEORY.md` §Numerical considerations). | No. |
| `PROBLEM:`  | The exact problem instance (agent count, grid size, arena size). | Yes — stable (demo runs with no args). |
| `SCENARIO:` | The scenario file's contents (spawn seed, step count). | Yes — stable. |
| `[time]`    | CPU oracle ms, GPU pipeline ms, per-kernel timings — a **teaching artifact, never a benchmark claim** (single-shot, one machine). | No. |
| `VERIFY:`   | `PASS`/`FAIL` verdict of the lockstep GPU-vs-CPU gate (N=4096, tolerance in `../src/kernels.cuh`). | Yes — stable. |
| `ARTIFACT:` | Confirms the three `out/` files were written. | Yes — stable. |
| `RESULT:`   | `PASS`/`FAIL` verdict of the headline check: all agents bounded in the arena, mean local alignment ≥ 0.5. The program exits nonzero on `FAIL`. | Yes — stable. |

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
