# Demo — 01.02 Stereo depth: block matching, then Semi-Global Matching (SGM) kernels

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

**The same cost volume, two winner strategies, one measurable difference.** The demo builds a
census-transform + Hamming-distance cost volume once from the committed synthetic stereo pair, then
runs it through two competing disparity extractors: **block matching** (per-pixel winner-take-all —
fast, no context) and **Semi-Global Matching** (4-direction path aggregation before winner-take-all —
a smoothness prior added to the SAME data). Both are scored against dense, exact ground truth on the
same 95,448 unoccluded pixels, so the improvement is a *measured percentage*, not an assertion:

```
BM:  good-pixel rate (|d-gt|<=1) = 63.35% over 95448 GT-valid pixels
SGM: good-pixel rate (|d-gt|<=1) = 97.52% over 95448 GT-valid pixels
```

**This demo writes three artifacts** into `out/` (git-ignored, regenerated each run) — the visual
story the percentages above cannot tell by themselves:

| File | What it shows |
|------|----------------|
| `disparity_bm.pgm` | Block matching's raw disparity map (×4 scaled, matching `gt_disparity.pgm`'s convention). Open it and look for SPECKLE and STREAKS — isolated pixels or short runs that jump to the wrong disparity, worst in the coarse-textured background where many local windows look nearly identical. |
| `disparity_sgm.pgm` | SGM's disparity map, same scale, same scene. The streaks are visibly gone — neighboring pixels agree because the path aggregation makes them pay a penalty for disagreeing. |
| `error_map.pgm` | Per-pixel correctness against ground truth: **white (255)** = correct within tolerance, **dark gray (80)** = wrong or no answer, **black (0)** = not scored (occluded / census-border pixel in the ground truth). Compare this against `disparity_bm.pgm`'s speckle by eye — the gray pixels in `error_map.pgm` for the BM pass cluster exactly where the streaks are. |

Open `left.pgm` / `right.pgm` (in `../data/sample/`) alongside these three to see the actual scene: a
coarse-textured ground plane and three fronto-parallel rectangles at different depths, two of them
partially overlapping (an object-vs-object occlusion edge, not just object-vs-background).

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, per-checkpoint mismatch counts — informative, not gating on its own. | No. |
| `PROBLEM:`  | The exact problem instance (D, census window, SGM penalties). | Yes — stable. |
| `DATA:`     | The loaded sample's dimensions and provenance. | Yes — stable. |
| `[time]`    | CPU/GPU kernel timings — a **teaching artifact, never a benchmark claim** (single-shot, one machine; the SGM path kernel is deliberately low-parallelism — one thread per SCANLINE, not per pixel — see `../THEORY.md` "The GPU mapping" for why that is an honest, named trade-off, not a bug). | No. |
| `VERIFY:`   | The GPU-vs-CPU EXACT-equality gate (census, cost volume, one SGM path, both final disparity maps) — this project's math is all-integer, so this is bit-for-bit equality, not a tolerance. Exits nonzero on any mismatch before the ground-truth gate even runs. | Yes — stable. |
| `BM:` / `SGM:` | The ground-truth good-pixel rates that decide `RESULT:` — deterministic percentages (see `expected_output.txt`'s header comment for why these numbers are part of the stable contract, unlike every other timing/hardware line in this repo). | Yes — stable. |
| `ARTIFACT:` | Confirms the three PGM artifacts above were written. | Yes — stable. |
| `RESULT:`   | `PASS`/`FAIL` verdict of the three ground-truth gates (BM floor, SGM floor, SGM-over-BM margin — thresholds and their measured headroom are documented at the top of `../src/main.cu`). The program exits nonzero on `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info) are allowed. `#`-prefixed lines in that file are comments.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list).
- **`VERIFY: FAIL`:** the GPU result disagreed with the CPU oracle — a real bug, and (because every
  operation here is integer arithmetic) never a rounding artifact. Start in `../src/kernels.cu` and
  diff against `../src/reference_cpu.cpp` function by function; the `[info] verify(...)` lines name
  exactly which stage first disagreed.
- **`RESULT: FAIL` with `VERIFY: PASS`:** the pipeline is internally consistent but a ground-truth gate
  was not met — look at the `BM:`/`SGM:` percentages against the thresholds documented at the top of
  `../src/main.cu`, and at `disparity_sgm.pgm`/`error_map.pgm` to see WHERE it went wrong.
- **Expected-line mismatch only:** the program passed its own checks but printed different stable
  lines — someone changed the output (or the data, or the tuning constants) without updating
  `expected_output.txt` (or vice versa). The two are a contract; fix them together.
