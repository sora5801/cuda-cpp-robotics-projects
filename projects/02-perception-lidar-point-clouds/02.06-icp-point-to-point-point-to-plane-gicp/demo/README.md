# Demo — 02.06 ICP: point-to-point → point-to-plane → GICP, all batched

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

The full ICP teaching pipeline, end to end, on two committed synthetic pairs (`data/sample/`): brute-
force GPU nearest-neighbor correspondence search, PCA/Jacobi surface-normal estimation, and a two-stage
GPU reduction that builds a 6x6 Gauss-Newton normal system every iteration — run in **both** the
point-to-point and point-to-plane variants, on both pairs (four closed-loop registrations total). The
demo verifies its own GPU kernels against a CPU oracle (`VERIFY:` lines), then checks every run's
recovered pose against the COMMITTED GROUND TRUTH (`CHECK:` lines), and — the project's central taught
result — checks that point-to-plane needed measurably FEWER iterations than point-to-point to converge
on the wall-dominated main pair.

Two artifacts land in `demo/out/` (git-ignored; regenerated every run):

- **`aligned.csv`** — pair 0's point-to-plane result: the source cloud after applying the recovered
  transform, subsampled to ~1500 points, alongside an equally-subsampled copy of the target cloud, both
  labeled by a `cloud` column (`aligned` / `target`). Plot both as 3D scatter, colored by `cloud` — a
  correct registration shows them coincide on the floor, the two walls, and the box.
- **`convergence.csv`** — pair 0's per-iteration RMS correspondence error and valid-match count, for
  BOTH variants (`mode` column). Plot `rms_m` vs `iter`, one line per mode: point-to-plane should reach
  its floor in a handful of iterations; point-to-point takes visibly longer to get there (README
  Exercise 1 asks you to explain the shape from THEORY.md's "sliding along the plane" argument).

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, timings, and measured errors/iteration-counts — varies by machine/run. | No. |
| `PROBLEM:`  | The exact problem instance (iteration cap, correspondence gate, damping). | Yes — stable. |
| `SCENARIO:` | Each pair's point counts and ground-truth translation magnitude (from the committed data — deterministic, not machine-dependent). | Yes — stable. |
| `VERIFY:`   | GPU-vs-CPU agreement for correspondences and both normal-system variants (tolerances only — measured deviations are in the preceding `[info]` line). | Yes — stable. |
| `CHECK:`    | Ground-truth pose-error gate (both pairs, both variants) plus the point-to-plane-converges-faster claim — thresholds only, no measured numbers. | Yes — stable. |
| `ARTIFACT:` | Confirms `aligned.csv` / `convergence.csv` were written. | Yes — stable. |
| `RESULT:`   | Final `PASS`/`FAIL` verdict. The program exits nonzero on `FAIL`. | Yes — stable. |

Every `[info]` line quoted above carries the actual MEASURED numbers (iteration counts, rotation/
translation error, RMS, GPU timings) — read those to see how far under threshold the real runs land.

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
