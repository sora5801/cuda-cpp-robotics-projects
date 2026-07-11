# Demo — 01.08 HDR exposure fusion + tone mapping for outdoor robots

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

Loads a four-exposure HDR bracket of a synthetic outdoor scene (`../data/sample/`), recovers the
camera's response function from nothing but the pixel data (Debevec-Malik), reconstructs linear-domain
radiance, tone-maps it two ways (global Reinhard, local pyramid-based), and separately fuses the same
four exposures with Mertens' multiscale blend (plus a naive single-scale blend for comparison) — all on
the GPU, all cross-checked against an independent CPU reference, and all graded against six gates tied to
the scene's known ground truth (recovered CRF vs. the true curve, recovered radiance vs. the exact
synthetic radiance, tone-map range/monotonicity, dynamic-range coverage vs. every single input exposure,
local detail preservation in a shadow and a highlight region, and a haloing comparison between the naive
and multiscale blends).

**What to look at:** open `out/naive_blend.pgm` next to `out/mertens_fusion.pgm` — the naive version shows
a visible seam/halo at the painted-line/concrete boundary that the multiscale version smooths away, the
entire lesson this project teaches made visible. `out/crf_curve.csv` plots the recovered CRF against the
scene's known true curve (after accounting for the algorithm's documented scale ambiguity — see
`../THEORY.md` "Numerical considerations"). `out/gates_metrics.csv` has every measured number behind
every `GATE` line below.

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, and every measured number (CRF offset, per-stage diffs, per-gate metrics) — varies by machine/run. | No. |
| `PROBLEM:`  | The exact problem instance (image size, exposure bracket, CRF parameters, pyramid depth). | Yes — stable (demo runs with no args). |
| `DATA:`     | A one-line description of the loaded sample scene. | Yes — stable. |
| `[time]`    | CRF-solve, per-kernel GPU, and CPU-oracle timings — a **teaching artifact, never a benchmark claim** (these images are tiny; both paths finish in single-digit milliseconds). | No. |
| `VERIFY:`   | `PASS`/`FAIL` verdict of the GPU-vs-CPU cross-check across all five pipeline stages (tolerances documented in `../src/main.cu` and `THEORY.md`). | Yes — stable. |
| `GATE ...:` | `PASS`/`FAIL` for each of the six independent ground-truth gates (`crf_recovery`, `radiance_reconstruction`, `tone_map_range`, `dynamic_range_coverage`, `detail_preservation`, `halo_check`). | Yes — stable (six lines). |
| `ARTIFACT:` | Confirms every file in `out/` was written. | Yes — stable. |
| `RESULT:`   | `PASS`/`FAIL` verdict of the WHOLE demo (VERIFY + all six gates). The program exits nonzero on `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info) are allowed. `#`-prefixed lines in that file are comments.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list).
- **`VERIFY: FAIL`:** the GPU result disagreed with the CPU oracle on at least one stage — a real bug.
  Start in `../src/kernels.cu` and compare the failing stage against its twin in `../src/reference_cpu.cpp`.
- **A `GATE ...: FAIL`:** VERIFY passed (GPU and CPU agree with each other) but the AGREED result
  disagrees with ground truth — read that gate's `[info]` line and its section in `../src/main.cu` and
  `../THEORY.md` "How we verify correctness" for what it measures and why.
- **Expected-line mismatch only:** the program passed its own check but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The two
  are a contract; fix them together.
