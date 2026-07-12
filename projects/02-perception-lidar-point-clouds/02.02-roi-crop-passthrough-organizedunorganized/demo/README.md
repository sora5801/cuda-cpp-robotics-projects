# Demo — 02.02 ROI crop, passthrough, organized↔unorganized conversion kernels

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

The demo loads the committed single-revolution 16-beam organized LiDAR scan
(`data/sample/roi_scan.bin`) and runs every kernel this project teaches, in order:

1. **Organized -> unorganized**: compacts the 16,384-cell ring x azimuth grid down to its ~12,361
   valid points, GPU and CPU, and checks them bit-exact against each other AND against the Python
   generator's own independent tally (three differently-timed, differently-languaged counts of the
   same thing).
2. Builds the **predicate test cloud** (those valid points + a 39-point "edge cohort" straddling
   every passthrough/box/frustum boundary) and runs all four named compactions — **passthrough**,
   **box ROI**, **frustum crop**, and **fused** — GPU vs. CPU.
3. Runs the SAME three filters as a **chained** 3-pass pipeline and checks it against the **fused**
   single-pass result: bit-identical, with an analytical memory-traffic estimate printed alongside.
4. **Round-trips** the unorganized cloud back through **unorganized -> organized** and checks
   identity against the original grid on every originally-valid cell.
5. Builds a **collision test** cloud (the valid points plus 200 "ghost" second-echo duplicates at
   known cells) and exercises the 64-bit-encoded-atomicMin nearest-wins scatter, reconciling
   `valid_in == occupied + collisions` for both GPU and CPU.
6. Compares the **hand-rolled two-level Blelloch scan** against `thrust::exclusive_scan` and a CPU
   serial scan — bit-exact, integer arithmetic, no tolerance needed — then times both across three
   synthetic array sizes (`[info] scan_scaling`).

Every comparison above prints a `VERIFY(...)`/`GATE ...:` line with a `PASS`/`FAIL` verdict; the
program's exit code is nonzero if any of them fail. `RESULT: PASS` means every one of them agreed.

### Artifacts (`demo/out/`)

| File | What it shows |
|------|----------------|
| `full_topview.ppm` | Top-down (looking down -z) render of the whole predicate test cloud. |
| `box_topview.ppm` | Same view, dim-gray backdrop + the box-ROI survivors highlighted in **green** — the crop boundary should be visible as a sharp rectangle of green points against gray. |
| `frustum_topview.ppm` | Same view, backdrop + the frustum-crop survivors highlighted in **cyan** — a wedge shape fanning out from the camera/LiDAR origin. |
| `organized_occupancy_before.pgm` | The organized grid's validity mask (white = valid, black = invalid), 1024x16 px — dark horizontal bands are open-sky misses over the (ceiling-less) walls; scattered single-pixel dropouts are the 5% absorption/glare model. |
| `organized_occupancy_after.pgm` | The SAME mask reconstructed via the round-trip (organized -> unorganized -> organized) — should be visually IDENTICAL to `_before.pgm` (`GATE roundtrip` checks this bit-exact, not just visually). |
| `gates_metrics.csv` | Every measured number behind every printed line, in one machine-readable file. |

PPM/PGM are viewable in most image viewers, GIMP, or IrfanView; VS Code's "PBM/PPM/PGM Viewer"
extension also opens them directly.

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name and compute capability — varies by machine. | No. |
| `PROBLEM:`  | The exact problem instance (sizes, parameters). | Yes — stable (demo runs with no args). |
| `[time]`    | CPU reference ms, GPU kernel ms, and a speed-up figure — a **teaching artifact, never a benchmark claim** (single-shot, kernel-only vs. one CPU core; first launches pay one-time init costs). | No. |
| `RESULT:`   | `PASS`/`FAIL` verdict of the GPU-vs-CPU check (tolerance documented in `../src/main.cu` and `THEORY.md`). The program exits nonzero on `FAIL`. | Yes — stable. |

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
