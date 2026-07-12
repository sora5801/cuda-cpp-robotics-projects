# Demo — 02.15 Point cloud compression (octree/entropy) for fleet uplink

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

The demo runs this project's whole two-stage codec (octree geometry coding, then canonical-Huffman
entropy coding — see `../README.md` "The algorithm in brief") end to end, on the GPU, and proves it
correct before trusting a single measured number:

1. **VERIFY stage** — at the canonical depth D=10, on the structured map tile, seven independent
   GPU-vs-CPU comparisons (Morton codes, sort, per-level octree construction, histogram, canonical
   Huffman table, encoded bitstream, decode round trip). Any mismatch aborts before the sweep runs.
2. **Sweep stage** — the same, now-verified GPU pipeline run for both committed clouds
   (`structured_map.bin`, `pathological_cube.bin`) across depths D=8,9,10,11 — the rate-distortion
   study this project exists to produce.
3. **Gates** — six pass/fail checks (`lossless_roundtrip`, `distortion_bound`, `rate_monotonic`,
   `entropy_payoff`, `entropy_bound`, `timing`) plus an `[info]`-only fleet-uplink arithmetic block.
4. **Artifacts**, written to `out/` (git-ignored; regenerated every run):
   - `rd_curve.csv` — the full 8-row rate-distortion sweep (both cohorts × all depths): node count,
     bits/point (raw and after Huffman), mean/max reconstruction error, the analytic distortion
     bound, the end-to-end compression ratio vs. raw float32 xyz, and the measured Shannon entropy
     vs. Huffman average code length.
   - `occupancy_histogram.csv` — the measured 256-symbol occupancy-byte histogram for both cohorts
     at the canonical depth — the compressibility physics made visible as numbers.
   - `topview_original.pgm`, `topview_recon_d8.pgm`, `topview_recon_d11.pgm` — 256×256 top-down
     (X-Y) point renders of the structured cloud: the original, and its reconstruction at the
     sweep's coarsest and finest depths — the visual distortion story (viewable in any PGM-capable
     image viewer, e.g. GIMP or IrfanView).
   - `gates_metrics.csv` — every gate's measured value, threshold, and verdict.

**What to notice:** the `[info] sweep ...` lines print, per (cohort, depth) row, the node count and
bits/point — watch the STRUCTURED cohort need roughly half the octree nodes (and bits) of the
PATHOLOGICAL cohort at every depth, for the *same* 200,000 points. That gap — not the entropy
coder's internal ratio, which behaves counter-intuitively (see `../THEORY.md` "How we verify
correctness") — is where the "surfaces are 2-D manifolds" argument actually pays off.

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, cube geometry, per-row sweep metrics, fleet arithmetic — every MEASURED number lives here. | No. |
| `PROBLEM:` / `DATA:` | The exact problem instance and loaded point counts. | Yes — stable (demo runs with no args). |
| `VERIFY <stage>:` | GPU-vs-CPU agreement for one of the seven verify-stage checks. | Yes — stable (PASS/FAIL only, no numbers). |
| `[time]`    | Wall-clock timings — a **teaching artifact, never a benchmark claim** (single machine, single run). | No. |
| `GATE <name>:` | PASS/FAIL verdict for one of the six rate-distortion/correctness gates (measured numbers live in `out/gates_metrics.csv`, not on this line). | Yes — stable. |
| `ARTIFACT:` | Confirms a file was written under `out/`. | Yes — stable. |
| `RESULT:`   | Overall PASS/FAIL — PASS only if the verify stage, every gate, and every artifact write succeeded. The program exits nonzero on `FAIL`. | Yes — stable. |

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
