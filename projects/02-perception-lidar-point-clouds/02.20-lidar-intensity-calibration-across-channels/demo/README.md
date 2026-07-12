# Demo — 02.20 LiDAR intensity calibration across channels

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

A 16-beam spinning LiDAR ray-casts a structured scene (a ground plane, a near wall, a small brighter
test panel beside it, and a far wall — `../scripts/make_synthetic.py`) with 16 DIFFERENT, hidden
per-channel gains baked into the forward model. The demo recovers all 16 relative gains from that ONE
scan alone — no reflectance targets — by finding small voxels of world space multiple channels
happened to observe, dividing out the one part of the signal every channel already agrees on (range
falloff and incidence angle), and solving a small least-squares system over the remaining
cross-channel disagreement. It:

1. **Verifies** the four-stage GPU pipeline (point features → voxel binning/accumulation →
   least-squares assembly → gain correction) against independently-typed CPU twins on the primary
   scan: per-point log-intensity (tight float tolerance), voxel indices (exact integers), per-voxel
   statistics and the assembled 16×16 system (atomic-order tolerance), and the final correction
   (tight tolerance).
2. **Solves once** (a shared, host-only 16×16 system — too small a problem for a meaningful GPU
   mapping) and reports which of the 16 channels are observable: connected, via a shared voxel, to
   the DOMINANT observation graph.
3. **Runs 4 gates**: `gain_recovery` (recovered gains vs. ground truth, gauge-aligned — the
   headline), `consistency_improvement` (the reason this project exists: cross-channel intensity
   spread collapses after calibration, plus a compact LIOR-style decision-flip demo closing project
   02.18's cited dependency), `multi_material_robustness` (mixing four different-reflectivity
   surfaces into the shared-voxel pool does not measurably hurt recovery vs. a single-material-only
   solve), and `unobservable_channel` (on a SECOND scan where one channel is deliberately retargeted
   to see nothing anyone else sees, the solver flags it rather than hallucinating a gain).
4. **Writes six artifacts** to `demo/out/` (git-ignored, regenerated every run):
   - `intensity_hist_before.csv` / `intensity_hist_after.csv` — per-channel histograms of the
     range-and-incidence-compensated intensity, before and after gain correction: 16 visibly
     discordant distributions (before) noticeably converging (after). Plot with anything.
   - `range_image_before.ppm` / `range_image_after.ppm` — a channel × azimuth "range image" colored
     by intensity (dark blue = no return at that azimuth for that channel). Visible horizontal
     banding in the BEFORE image (each row a different average brightness) mostly disappears in the
     AFTER image. Open with any image viewer that reads PPM (GIMP, IrfanView, `pillow`'s
     `Image.open`), or convert (`magick range_image_before.ppm range_image_before.png`).
   - `range_profile.csv` — the bonus milestone: nonparametric range-falloff shape recovery vs. the
     generator's true curve, reported as an `[info]` line only (not gated).
   - `gates_metrics.csv` — every gate's measured value, threshold, and verdict, for the record.

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, voxel-grid size, every MEASURED number (VERIFY deltas, recovered gains, gate metrics, the bootstrap noise-floor, the range-profile deviation) — informative but not diffed, per the output contract in `../src/main.cu`. | No. |
| `PROBLEM:`  | The exact problem instance (channel count, forward model). | Yes — stable (demo runs with no args). |
| `DATA:`     | What was loaded and how many points. | Yes — stable. |
| `[time]`    | CPU vs. GPU kernel timing — a **teaching artifact, never a benchmark claim**. | No. |
| `VERIFY:`   | GPU-vs-CPU agreement verdict across all four pipeline stages. | Yes — stable. |
| `GATE ...:` | Pass/fail verdict for each of the 4 gates (measured numbers live on the paired `[info]` line just above/below). | Yes — stable. |
| `ARTIFACT:` | Confirms every artifact file above was written. | Yes — stable. |
| `RESULT:`   | Overall `PASS`/`FAIL`. The program exits nonzero on `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info, every `[info]` measurement) are allowed. `#`-prefixed lines in that file are
comments.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list).
- **`VERIFY: FAIL`:** the GPU pipeline disagreed with its CPU oracle at one of the four stages — a
  real bug. Start in `../src/kernels.cu` and compare the matching function side by side against
  `../src/reference_cpu.cpp`.
- **A `GATE` line fails but `VERIFY` passes:** the pipeline is internally consistent but the measured
  behavior on this scene crossed a threshold — read the paired `[info]` line for the actual number,
  and see README "Expected output" / THEORY.md for what each gate's threshold means and how it was
  measured-then-margined.
- **Expected-line mismatch only:** the program passed its own checks but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The two
  are a contract; fix them together.
