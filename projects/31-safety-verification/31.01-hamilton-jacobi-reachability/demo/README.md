# Demo — 31.01 Hamilton-Jacobi reachability: level-set grid solvers (stencil ops — GPU-perfect)

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

**A GPU PDE solver whose answer is checked against pure mathematics, not just against itself.**
The demo solves the Hamilton-Jacobi backward-reachable-tube equation for a double integrator
(`|u| <= 0.8 m/s^2`) on a 256x256 state-space grid over 109 explicit sweeps (a 0.4 s horizon on top
of a 0.6 s target, 1.0 s total elapsed-time budget), producing a value field `V(x,v)` whose zero
sublevel set is the backward reachable tube — every state from which *some* control sequence
reaches the origin within that budget. Two independent checks gate the verdict:

1. **VERIFY** — the repo-standard §5 gate: the GPU field and a plain-C++ CPU twin, solved from the
   identical initial condition with the identical per-cell update, must agree within
   `max|V_gpu-V_cpu| <= 1e-3` (measured worst case: `~1.7e-5`).
2. **ANALYTIC** — the check this project exists to feature: every cell's `V <= 0` classification is
   compared against the **closed-form bang-bang minimum-time solution** of the double integrator
   (a textbook Pontryagin result, not another program) — required to match exactly everywhere
   except a documented, measured 3-grid-cell boundary band (measured: 0 disagreements outside the
   band, 230 inside it out of 1,556 band cells).

**This demo writes two artifacts** into `out/` (git-ignored, regenerated each run):

- `value_function.pgm` — the value field as an 8-bit grayscale image (P5 PGM, viewable in any tool
  that reads it, or convertible with ImageMagick/GIMP/Python's Pillow). The reachable tube is the
  **dark region**; the target's deepest interior is black; far unreachable states are white. Row
  `j=0` (`v = vmin`) is the top image row.
- `brs_boundary.csv` — every boundary cell of the numeric tube, with columns `x_m, v_mps, t_min_s`
  (the analytic minimum time at that cell). Plot `t_min_s` along the boundary and watch it hug the
  near-constant value `t0 + T = 1.0 s` — that near-constancy IS the analytic check, visible to the
  eye (README Exercise 1).

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, file paths, and the full numeric detail behind VERIFY/ANALYTIC (worst error, tube size, sup-error) — varies by machine and is not diffed, but is where the interesting numbers live. | No. |
| `PROBLEM:`  | The exact problem instance (grid, domain, `umax`, horizon). | Yes — stable (demo runs with no args). |
| `SCENARIO:` | The target definition and the derived sweep schedule (`n_sweeps`, `dt`, CFL). | Yes — stable. |
| `[time]`    | CPU reference ms, GPU kernel ms, and a speed-up figure — a **teaching artifact, never a benchmark claim** (single-shot, kernel-only vs. one CPU core; varies run to run). | No. |
| `VERIFY:`   | `PASS`/`FAIL` verdict of the GPU-vs-CPU twin check. | Yes — stable. |
| `ANALYTIC:` | `PASS`/`FAIL` verdict of the closed-form-mathematics check — this project's headline verification. | Yes — stable. |
| `ARTIFACT:` | Confirms the PGM/CSV were written. | Yes — stable. |
| `RESULT:`   | Overall `PASS`/`FAIL` — PASS requires VERIFY, ANALYTIC, and the artifact writes to all succeed. The program exits nonzero on `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info, the `[info]` detail lines) are allowed. `#`-prefixed lines in that file are
comments explaining the diff rules, not part of the expected output itself.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list).
- **`VERIFY: FAIL`:** the GPU field disagreed with the CPU oracle — an indexing/layout/sign bug.
  Start in `../src/kernels.cu` and diff it line-by-line against `../src/reference_cpu.cpp`'s
  `hj_sweep_cell` (they are meant to read almost identically).
- **`ANALYTIC: FAIL`:** the numeric field disagrees with the closed-form minimum-time solution
  outside the documented band. Per this project's own investigation (`THEORY.md §Numerical
  considerations`), check sign conventions and the upwinding direction FIRST — never widen
  `kBandCells` in `../src/kernels.cuh` without measuring the actual required band (README
  Exercise 2/3 show you how) and writing the justification down, the same discipline this
  project's own scenario parameters were chosen with.
- **Expected-line mismatch only:** the program passed its own checks but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The
  two are a contract; fix them together.
