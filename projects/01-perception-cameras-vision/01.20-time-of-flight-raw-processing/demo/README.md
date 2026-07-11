# Demo — 01.20 Time-of-flight raw processing: phase unwrapping, flying-pixel removal

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

A full continuous-wave indirect-ToF (iToF) raw-processing pipeline on a synthetic room-scale scene
(a tilted background wall ~5 m away, a sphere, and a box's raised top face — a step edge), captured
through the 8-frame tap stack (2 modulation frequencies x 4 taps) committed in `../data/sample/`. The
program runs all **six pipeline stages** — 4-tap phase/amplitude extraction, single-frequency
(wrapped) depth, dual-frequency phase unwrapping, amplitude confidence masking, flying-pixel
detection, and pinhole back-projection — on **both** the GPU and an independent CPU oracle (as two
separate end-to-end cascades), checks stage-by-stage agreement, then runs **nine independent gates**
against the synthetic ground truth: how accurately the phase/amplitude extraction recovers the
analytic truth, whether the phase decode is truly immune to added ambient IR (the algebra
`../THEORY.md` derives), whether the naive single-frequency depth genuinely ALIASES on the far wall
(the designed demonstration), how well dual-frequency unwrapping recovers metric depth and the
correct integer wrap count, how precisely/how completely the flying-pixel detector catches the
scene's designed mixed-return edge pixels (scored against an INDEPENDENT ground-truth label the
detector never sees), whether the reconstructed wall/sphere/step-edge match their known analytic
geometry, and whether the amplitude mask correctly *rejects* a deliberately low-reflectivity "dark
cohort" instead of silently reconstructing garbage there.

It writes nine labeled artifacts into `out/` (git-ignored, regenerated every run):

| File | What it shows |
|------|----------------|
| `tap_sample_f1.pgm` | One committed raw correlation tap (frequency 1 / 60 MHz, tap 0 / 0 deg) — what the sensor's charge well actually reads before any decode math. |
| `tap_sample_f2.pgm` | The same, for frequency 2 (20 MHz, the coarse/unwrapping channel). |
| `wrapped_depth_f1.pgm` | The naive SINGLE-frequency depth (freq1 only) — the "aliasing visual": the far wall shows repeating bands where the true depth wrapped around the ambiguity range. |
| `unwrapped_depth.pgm` | The dual-frequency UNWRAPPED depth over the whole scene range — the wall is now one smooth gradient, no wraparound. |
| `flying_pixel_mask.pgm` | White = a pixel the detector flagged as a mixed (flying) return — a thin outline around the sphere and box silhouettes. |
| `point_cloud.csv` | `x_m,y_m,z_m,surface_id_truth` for every triangulated point AFTER flying-pixel removal (camera-frame meters; `surface_id_truth` is included from the SYNTHETIC ground truth purely to color this visualization — it is never fed back into decoding). |
| `profile_view_before.ppm` | An orthographic X-Z scatter render BEFORE flying-pixel removal (confidence-valid pixels only) — look for points hanging in space BETWEEN the sphere/box and the far wall: the "flying pixel" effect, visible. |
| `profile_view_after.ppm` | The SAME render AFTER flying-pixel removal — the hanging points are gone, the sphere/box/wall read as three clean, separated surfaces. The before/after pair is the demo's "money shot". |
| `gates_metrics.csv` | Every gate's measured value and verdict, full precision (not diffed — a reference for plotting/inspection). |

**Placeholder status:** none — this is the project's real implementation; the scaffolded SAXPY
smoke test has been fully replaced.

## How to read the console output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, measured floating-point numbers behind every VERIFY/GATE line, and the non-gated `noise_scaling` diagnostic. Varies by machine/run. | No. |
| `PROBLEM:` / `SCENARIO:` | The exact sensor + scene instance. | Yes — stable (demo runs with no args). |
| `[time]`    | CPU vs. GPU timing for the full pipeline — a **teaching artifact, never a benchmark claim** (single-shot; first launches pay one-time init costs). | No. |
| `VERIFY:`   | One line per pipeline stage (6 total): does the GPU kernel agree with the independent CPU oracle, within the stated tolerance/allowance? | Yes — stable. |
| `GATE:`     | One line per independent gate (9 total): does the (GPU-vs-CPU-verified) result agree with the SYNTHETIC GROUND TRUTH, within the stated floor/bound? | Yes — stable. |
| `ARTIFACT:` | Confirms each `out/` file was written. | Yes — stable. |
| `RESULT:`   | Final `PASS`/`FAIL` — all 6 VERIFYs and all 9 GATEs must pass. The program exits nonzero on `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info, measured numbers) are allowed. `#`-prefixed lines in that file are comments.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list).
- **A `VERIFY:` line FAILs:** the GPU kernel disagreed with the CPU oracle for that stage beyond its
  documented allowance — a real bug. Start in `../src/kernels.cu` at that stage's kernel and compare
  line-by-line against the matching function in `../src/reference_cpu.cpp`.
- **A `GATE:` line FAILs but every `VERIFY:` PASSes:** the GPU and CPU agree with EACH OTHER but not
  with the synthetic ground truth — look for a bug shared by both twins (a sign error, a swapped tap
  index) or, if you changed `scripts/make_synthetic.py`, a scene parameter that pushed a gate's
  measured value past its (measured-then-margined) floor/bound — see `../data/README.md` "How the
  sample was tuned".
- **Expected-line mismatch only:** the program passed its own checks but printed different stable
  lines than `expected_output.txt` records — someone changed the output without updating the file
  (or vice versa). The two are a contract; regenerate one from the other.
