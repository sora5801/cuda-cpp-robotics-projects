# Demo — 15.01 Minimum-snap trajectory optimization batched over waypoint sets

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

**10,000 independent minimum-snap trajectory problems, solved on the GPU with one thread per
problem, each doing its own 32×32 Gaussian elimination.** Every waypoint set is 5 (x,y) points a
quadrotor must fly through; the demo builds a batch from 5 hand-designed shapes (`data/sample/`)
plus 9,995 seeded-random shapes, solves the whole batch on the GPU and, independently, on the CPU
(`src/reference_cpu.cpp`), and then checks the result TWO ways:

1. **VERIFY** — the §5 GPU-vs-CPU gate: every one of the 640,000 solved coefficients (10,000 sets ×
   2 axes × 32 coefficients) must agree between the two paths within a documented relative
   tolerance (measured worst case: ~6.4e-4, tolerance 5e-3 — ~8× headroom).
2. **CONSTRAINTS** — a check against the *mathematical definition* of minimum snap, independent of
   the CPU oracle: for every one of the 10,000 sets, re-evaluate the solved polynomials (a
   double-precision, separately-coded evaluator — see `src/main.cu`'s `eval_segment_derivs`) and
   measure waypoint-interpolation error, endpoint zero-derivative error, interior continuity jumps,
   and the analytic snap-cost integral (must be finite and non-negative for every set). All four
   pass with wide, documented margins.

**This demo writes two artifacts** (git-ignored, regenerated each run), both from the `slalom`
waypoint set (the zig-zag shape in `data/sample/waypoint_sets.csv`):

- `out/trajectory.csv` — 201 dense samples (columns `t_s, x_m, y_m, vx_ms, vy_ms, ax_ms2, ay_ms2`)
  across the whole 4-second flight. Plot `x_m` vs `y_m` to see the zig-zag path; plot `vx_ms`/`vy_ms`
  vs `t_s` to see the smooth, snap-continuous speed profile pumping through each turn.
- `out/slalom_path.pgm` — a 256×256 grayscale raster of the same path (mid-gray line, bright
  crosses at the 5 waypoints) — open it in any image viewer that reads PGM (GIMP, IrfanView, VS
  Code with an image extension) for an instant visual sanity check with zero plotting setup.

## How to read the output

| Line prefix    | Meaning | Checked against `expected_output.txt`? |
|-----------------|---------|----------------------------------------|
| `[demo]`        | Which project/demo this is. | Yes — stable. |
| `[info]`        | GPU name, sample-file path, and every MEASURED residual/cost number — varies by machine/GPU architecture. | No. |
| `PROBLEM:`      | The exact problem instance (batch size, segment/coefficient shape, box/spacing parameters). | Yes — stable (demo runs with no args). |
| `SAMPLE:`       | Confirms the 5 hand-designed sets loaded, including `slalom`. | Yes — stable. |
| `[time]`        | CPU reference ms, GPU kernel ms, and a speed-up figure — a **teaching artifact, never a benchmark claim**. | No. |
| `VERIFY:`       | `PASS`/`FAIL` verdict of the GPU-vs-CPU coefficient check (tolerance documented in `../src/main.cu` and `THEORY.md`). | Yes — stable. |
| `CONSTRAINTS:`  | `PASS`/`FAIL` verdict of the against-the-definition residual and snap-cost checks. | Yes — stable. |
| `ARTIFACT:`     | Confirms both output files were written. | Yes — stable. |
| `RESULT:`       | Overall verdict; the program exits nonzero on any `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info, measured residuals) are allowed. `#`-prefixed lines in that file are
comments.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list).
- **`VERIFY: FAIL`:** the GPU result disagreed with the CPU oracle — a real bug. Start in
  `../src/kernels.cu` and compare against `../src/reference_cpu.cpp` (they should be line-by-line
  twins other than `fabsf`-vs-`std::fabs` spellings).
- **`CONSTRAINTS: FAIL`:** the solved coefficients do not satisfy the minimum-snap constraints they
  were solved for — check the row-index arithmetic in `assemble_minsnap_system` against the layout
  documented in `../src/kernels.cuh`'s header comment.
- **Expected-line mismatch only:** the program passed its own checks but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The two
  are a contract; fix them together.
