# Demo — 17.01 Batched Lambert solvers + porkchop plot generation

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

**A GPU solving 262,144 independent trajectory problems in one kernel launch.** For every
(departure epoch, arrival epoch) cell on a 512×512 grid, the demo solves Lambert's problem between
two synthetic coplanar circular heliocentric orbits (Earth-like at 1 AU, Mars-like at 1.524 AU) and
reports the total impulsive delta-v of that transfer — the classic **porkchop plot** mission
designers use to pick a launch window. Three independent checks gate the verdict:

1. **VERIFY** — the §5 GPU-vs-CPU gate: every cell's classification and delta-v computed on the
   kernel and on [`../src/reference_cpu.cpp`](../src/reference_cpu.cpp) must agree (tolerance and
   justification in `../src/main.cu` and `THEORY.md`).
2. **NAN POLICY** — the fraction of *attempted* cells (short-way, valid time-of-flight) that hit the
   Lambert equations' genuine mathematical singularity (a transfer angle of exactly 180°) or failed
   to converge must stay small — measured and printed on an `[info]` line.
3. **ANALYTIC** — verification against **pure mathematics**: for two coplanar circular orbits, the
   delta-v-optimal transfer is provably the Hohmann transfer (closed-form, `THEORY.md` derives it
   from vis-viva). The grid's own minimum must land within a documented small window of that
   closed-form value — a third, independent computation that shares no code with the Lambert solver.

**This demo writes two artifacts** into `out/` (git-ignored, regenerated each run):

- `porkchop.pgm` — the classic picture: a grayscale image where brighter pixels are cheaper
  transfers (lower delta-v) and black pixels are excluded cells (masked time-of-flight, the
  long-way/Type-II transfers this project's v1 does not solve, or the near-singular ring around the
  Hohmann geometry). Open it in any image viewer.
- `minimum.csv` — the winning cell's departure/arrival epochs, time-of-flight, and delta-v, alongside
  the closed-form Hohmann values and the measured gap between them.

## How to read the output

| Line prefix   | Meaning | Checked against `expected_output.txt`? |
|----------------|---------|----------------------------------------|
| `[demo]`       | Which project/demo this is. | Yes — stable. |
| `[info]`       | GPU name, scenario path, cell census, and the measured gaps — varies by machine/build. | No. |
| `PROBLEM:`     | The exact problem instance (grid size, canonical units). | Yes — stable (demo runs with no args). |
| `SCENARIO:`    | The two orbit radii, epoch window, and time-of-flight band loaded from `data/sample/`. | Yes — stable. |
| `[time]`       | CPU reference ms, GPU kernel ms, and a speed-up figure — a **teaching artifact, never a benchmark claim**. | No. |
| `VERIFY:`      | The §5 GPU-vs-CPU gate verdict. | Yes — stable. |
| `NAN POLICY:`  | The degenerate/non-converged cell-fraction gate verdict. | Yes — stable. |
| `ANALYTIC:`    | The closed-form Hohmann verification verdict. | Yes — stable. |
| `ARTIFACT:`    | Confirms `porkchop.pgm` and `minimum.csv` were written. | Yes — stable. |
| `RESULT:`      | The combined `PASS`/`FAIL` verdict. The program exits nonzero on `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info, the cell census) are allowed. `#`-prefixed lines in that file are comments.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list).
- **`VERIFY: FAIL`:** the GPU result disagreed with the CPU oracle — a real bug. Start in
  `../src/kernels.cu` and compare against `../src/reference_cpu.cpp` line by line.
- **`NAN POLICY: FAIL`:** the near-singular/non-converged fraction exceeded the documented bound —
  check the `[info]` cell-census line; a wider `kEpsSingularRad` or a bracket that no longer covers
  this scenario's `z` range (`kernels.cuh`) are the usual suspects.
- **`ANALYTIC: FAIL`:** the grid minimum drifted too far from the closed-form Hohmann value — check
  `minimum.csv`'s reported gap; a legitimate cause is a scenario edit (`GRID_N`, the epoch window)
  that changes how closely the grid can approach the continuous optimum (`THEORY.md` §how-we-verify).
- **Expected-line mismatch only:** the program passed its own checks but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The two
  are a contract; fix them together.
