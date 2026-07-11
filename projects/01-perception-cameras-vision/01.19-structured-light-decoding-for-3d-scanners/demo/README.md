# Demo — 01.19 Structured-light decoding (Gray code, phase shift) for 3D scanners

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

A full Gray-code + phase-shift **hybrid** structured-light scan of a synthetic scene (a tilted
background plane, a sphere, and a box's raised top face — a step edge), captured through the
20-pattern stack committed in `../data/sample/`. The program runs all **five pipeline stages** —
Gray decode, phase decode, hybrid combine, triangulation, and a Gray-vs-plain-binary boundary
stress test — on **both** the GPU and an independent CPU oracle, checks agreement, then runs
**eight independent gates** against the synthetic ground truth: how accurately Gray code alone
recovers the projector column, how much sharper the phase-refined *hybrid* answer is, how much
worse plain binary coding fails at code boundaries than Gray code under identical noise, whether
the phase decode is truly immune to added ambient light (the algebra `../THEORY.md` derives),
whether the reconstructed plane/sphere/step-edge match their known analytic ground truth, and
whether the confidence mask correctly *rejects* a deliberately low-albedo "dark stripe" region
instead of silently reconstructing garbage there.

It writes six labeled artifacts into `out/` (git-ignored, regenerated every run):

| File | What it shows |
|------|----------------|
| `gray_pattern_sample.pgm` | One committed Gray-code pattern (bit 3, direct illumination) — the vertical-stripe binary code as the camera actually sees it. |
| `phase_pattern_sample.pgm` | One committed phase-shift pattern (step 0 of 4) — the smooth sinusoidal fringes. |
| `decoded_column_map.pgm` | The final HYBRID sub-pixel projector column per pixel, normalized to `[1,255]`; `0` (pure black) marks pixels the confidence mask rejected. |
| `confidence_map.pgm` | The phase modulation amplitude (`B`) per pixel, in intensity counts — brighter = more trustworthy; this is literally the signal `hybrid_combine_kernel` thresholds. |
| `point_cloud.csv` | `x_m,y_m,z_m,surface_id_truth` for every triangulated point (camera-frame meters; `surface_id_truth` is included from the SYNTHETIC ground truth purely to color this visualization — it is never fed back into decoding). |
| `profile_view.ppm` | An orthographic X-Z scatter render — the "money shot": the sphere reads as a round bump and the box as a rectangular step, both poking UP out of the tilted background band (closer = higher in this projection; `../src/main.cu`'s artifact-writing section explains the convention). |
| `gates_metrics.csv` | Every gate's measured value and verdict, full precision (not diffed — a reference for plotting/inspection). |

**Placeholder status:** none — this is the project's real implementation; the scaffolded SAXPY
smoke test has been fully replaced.

## How to read the console output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, measured floating-point numbers behind every VERIFY/GATE line, and non-gated diagnostics. Varies by machine/run. | No. |
| `PROBLEM:` / `SCENARIO:` | The exact scanner + scene instance. | Yes — stable (demo runs with no args). |
| `[time]`    | CPU vs. GPU timing for the main pipeline and the boundary stress test — a **teaching artifact, never a benchmark claim** (single-shot; first launches pay one-time init costs). | No. |
| `VERIFY:`   | One line per pipeline stage: does the GPU kernel agree with the independent CPU oracle, within the stated tolerance? | Yes — stable. |
| `GATE:`     | One line per independent gate: does the (GPU-vs-CPU-verified) result agree with the SYNTHETIC GROUND TRUTH, within the stated floor/bound? | Yes — stable. |
| `ARTIFACT:` | Confirms each `out/` file was written (with a deterministic count where relevant). | Yes — stable. |
| `RESULT:`   | Final `PASS`/`FAIL` — all 5 VERIFYs and all 8 GATEs must pass. The program exits nonzero on `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info, measured numbers) are allowed. `#`-prefixed lines in that file are comments.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list).
- **A `VERIFY:` line FAILs:** the GPU kernel disagreed with the CPU oracle for that stage — a real
  bug. Start in `../src/kernels.cu` at that stage's kernel and compare line-by-line against the
  matching function in `../src/reference_cpu.cpp`.
- **A `GATE:` line FAILs but every `VERIFY:` PASSes:** the GPU and CPU agree with EACH OTHER but
  not with the synthetic ground truth — look for a bug shared by both twins (a data-layout mixup,
  a sign error in the geometry) or, if you changed `scripts/make_synthetic.py`, a scene parameter
  that pushed a gate's measured value past its (measured-then-margined) floor/bound — see
  `../data/README.md` "How the sample was tuned".
- **Expected-line mismatch only:** the program passed its own checks but printed different stable
  lines than `expected_output.txt` records — someone changed the output without updating the file
  (or vice versa). The two are a contract; regenerate one from the other.
