# Demo — 02.11 Scan Context / ring-descriptor loop-closure search

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

The demo loads a synthetic 160-keyframe trajectory through a small 4x3-block "town" (`../data/README.md`)
and, for every keyframe, builds a Scan Context descriptor (a 20-ring x 60-sector polar height-max matrix)
on the GPU. It then runs the two-stage loop-closure search — a cheap ring-key L1 prefilter followed by
the full column-shift cosine-distance search — against every temporally-valid earlier keyframe, and
checks the result against curated ground truth: 8 same-heading revisits, 4 heading-reversed ("rotated")
revisits, 4 laterally-offset revisits, and 8 genuinely new places that must never fire a false loop
closure. Six independent gates score the result (see `../README.md` "Expected output" for the floor on
each), and an `[info]` illustration hands the recovered rotation off to a compact ICP to show it actually
helps a downstream aligner converge faster.

**Artifacts** (written to `demo/out/`, created if missing):

| File | What it shows |
|------|----------------|
| `sc_heatmap_revisit_{query,match}.pgm` | Two Scan Context matrices for a genuine same-place revisit — visually near-identical. |
| `sc_heatmap_nonpair_{query,candidate}.pgm` | A negative-cohort query next to its nearest (but wrong-place) candidate — visually different, yet close enough in distance to illustrate the aliasing problem THEORY.md names. |
| `trajectory_view.ppm` | Top-down view of the whole town + trajectory, with every detected loop closure drawn as a chord (green = correct place, red = wrong place). |
| `pr_curve.csv` | Precision/recall swept across 61 threshold values (0.00–1.20) over the FULL trajectory, continuous ground truth — the complete picture behind the one operating threshold the gates use. |
| `gates_metrics.csv` | Every VERIFY/GATE/`[info]` number this run printed, machine-readable. |

PGM/PPM are viewable with any image tool that reads plain NetPBM (GIMP, IrfanView, `convert` from
ImageMagick, or `matplotlib.pyplot.imread`).

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, measured statistics, and the diagnostic numbers behind each gate — informative but not diffed (varies slightly by GPU architecture/driver). | No. |
| `PROBLEM:`  | The fixed problem instance (grid size, gap, prefilter budget). | Yes — stable. |
| `DATA:`     | What was loaded from `data/sample/` (keyframe/building/point counts). | Yes — stable. |
| `[time]`    | CPU vs. GPU timings and a speed-up figure — a **teaching artifact, never a benchmark claim** (single-shot; first launches pay one-time init costs). | No. |
| `VERIFY(...):` | GPU-vs-CPU correctness-oracle gates (scan_context, ring_key, shift_distance). | Yes — stable. |
| `GATE ...:` | The independent, ground-truth-scored gates (loop_detection, rotation_invariance, lateral_sensitivity, negative_cohort, ringkey_prefilter). | Yes — stable. |
| `ARTIFACT:` | Confirmation that each `demo/out/` file was written. | Yes — stable. |
| `RESULT:`   | Overall `PASS`/`FAIL` — every VERIFY and GATE must pass. The program exits nonzero on `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info) are allowed. `#`-prefixed lines in that file are comments.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list).
- **A `VERIFY(...)` line fails:** the GPU result disagreed with the CPU oracle — a real bug. Start in
  `../src/kernels.cu` and compare against `../src/reference_cpu.cpp`; `kernels.cuh`'s file header
  documents the one place a small, bounded disagreement (rare ring/sector boundary ties) is expected
  and tolerated.
- **A `GATE ...` line fails:** the pipeline is internally consistent (GPU matches CPU) but disagrees
  with GROUND TRUTH — check `data/sample/loop_pairs.csv` against what `main.cu` actually detected, and
  read `THEORY.md` "numerical considerations" for the two real bugs (an empty-cell sentinel bug and an
  anchor-position asymmetry bug) this project's own gates caught during development.
- **Expected-line mismatch only:** the program passed its own checks but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The two
  are a contract; fix them together.
