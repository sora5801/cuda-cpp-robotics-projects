# Demo — 32.02 CUDA Graphs for jitter-free fixed-rate perception-control loops

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

One "tick" of a small, realistic perception-control DAG (8 kernels + 2 device-to-device copies,
`../src/kernels.cu`) run **2000 times at a fixed 250 Hz**, three different ways — naive per-kernel
stream launches, a captured CUDA Graph replayed every tick, and the same graph updated every tick
via `cudaGraphExecKernelNodeSetParams` — while measuring host submission time, end-to-end latency,
and pacing accuracy for each. It verifies two independent things (the GPU tick matches a plain-C++
CPU twin, AND all three orchestration modes produce bit-identical outputs over the whole run), then
reports the honest, measured answer to "do CUDA Graphs actually help here" — including where they do
not (see the root [`README.md`](../README.md) "Expected output" section for the full, numbers-
included story). Total runtime is ~24 s.

**Artifacts** (written to `out/`, git-ignored, regenerated every run):

- `out/latency_histogram.csv` — one row per tick per mode (6000 rows): `submit_us`, `latency_us`,
  `gpu_exec_ms`, `period_us`. The plotting payload — Exercise 1 in the root README asks you to
  histogram `latency_us` per mode and see the tail visually.
- `out/jitter_summary.csv` — one row per (mode, metric) pair (12 rows): `mean, p50, p95, p99, max`.
  The numbers behind every `[time]`/`[info]` line the program prints.

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, scenario path, Sleep(1) calibration, per-mode progress, and the ACTUAL measured numbers behind each `GATE` verdict — varies by machine and run. | No. |
| `PROBLEM:` / `SCENARIO:` | The exact tick shape and scenario loaded from `data/sample/tick_scenario.csv`. | Yes — stable (demo runs with no args). |
| `[time]`    | Per-mode submit/latency/gpu-exec means and percentiles — **teaching artifacts, never a benchmark claim** (single-shot, one machine, one run). | No. |
| `VERIFY:`   | GPU tick 0 vs. the CPU tick twin (`../src/reference_cpu.cpp`), within documented tolerance. | Yes — stable. |
| `CROSSMODE:`| Modes B/C's full-run outputs bit-identical to mode A's. | Yes — stable. |
| `GATE ...:` | Three measurement sanity gates (submit-reduction, gpu-work-consistency, pacing-accuracy) — the verdict text states the threshold POLICY only; the measured numbers are on the preceding `[info]` line. | Yes — stable (numbers deliberately excluded). |
| `ARTIFACT:` | Confirms the two CSVs were written, with their row counts. | Yes — stable. |
| `RESULT:`   | Overall `PASS`/`FAIL` — requires VERIFY, CROSSMODE, and all three gates to pass. The program exits nonzero on `FAIL`. | Yes — stable. |

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
