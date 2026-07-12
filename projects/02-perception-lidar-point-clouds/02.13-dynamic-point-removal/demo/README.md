# Demo — 02.13 Dynamic point removal (raycast free-space carving)

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

Ten posed LiDAR scans (28,800 beams total) through a small synthetic scene — a wall, a thin pole,
a car that drives across the sensor's path in scans 1-4, and a pedestrian who stands still for
scans 0-4 then leaves — are carved into a shared voxel ledger by the Amanatides-Woo DDA march
(`../src/kernels.cu`), then every recorded point is classified STATIC or DYNAMIC from that ledger's
hit/pass ratio. The demo:

1. **Verifies** the DDA march (a documented subset, integer voxel sequences, exact), the full
   14-scan-equivalent carve ledger (exact, element-wise), and the classification (exact given the
   ledger) — GPU against an independently-typed CPU twin, three separate gates.
2. **Runs the five headline gates** (`ghost_removal`, `late_leaver`, `static_preservation`,
   `free_space_consistency`, `max_range_carving`) against ground truth loaded from
   `data/sample/beams.csv` — never seen by the carving/classification algorithm itself.
3. **Writes three artifacts** to `demo/out/` (git-ignored, regenerated every run):
   - `triptych.ppm` — the "money shot": three top-view panels side by side. LEFT is every point
     ever recorded, colored by ground truth (red = something that moved, gray = permanent
     structure) — the car's four-scan ghost trail is plainly visible as a red smear. MIDDLE is the
     CLEANED map (only points this project's algorithm decided to keep), colored by whether that
     decision was right (gray) or a miss (orange — a truth-dynamic point that was NOT removed,
     shown honestly). RIGHT is the ground-truth-static-only map, the answer key. Open the PPM with
     any image viewer that reads the format (GIMP, IrfanView, `pillow`'s `Image.open`), or convert
     it (`magick triptych.ppm triptych.png`).
   - `pedestrian_evidence.csv` — the "late leaver" evidence, scan by scan: the pedestrian's own
     voxel accumulates almost no free-space evidence while it stands there (scans 0-4), then the
     score climbs sharply once it leaves and later scans carve straight through (scans 5-9) — open
     it in a spreadsheet or plot `score` against `scan_id`.
   - `gates_metrics.csv` — every gate's measured value, threshold, and verdict, for the record.

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, data paths, and every MEASURED number (percentages, counts, contention stats) — informative but not diffed, per the output contract in `../src/main.cu`. | No. |
| `PROBLEM:`  | The exact problem instance (scan/beam/voxel counts). | Yes — stable (demo runs with no args). |
| `VERIFY:`   | GPU-vs-CPU agreement verdicts for the DDA trace, the ledger, and the classification — all EXACT (no tolerance; the DDA march has no transcendental functions, so bit-exactness is the expected outcome, not luck). | Yes — stable. |
| `[time]`    | CPU vs. GPU carve timing and a speed-up figure — a **teaching artifact, never a benchmark claim**. | No. |
| `GATE:`     | Pass/fail verdict for each of the five headline gates (measured numbers live on the paired `[info]` line just above). | Yes — stable. |
| `ARTIFACT:` | Confirms each of the three files above was written. | Yes — stable. |
| `RESULT:`   | Overall `PASS`/`FAIL`. The program exits nonzero on `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info, every `[info]` measurement) are allowed. `#`-prefixed lines in that file are
comments.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list).
- **A `VERIFY:` line fails:** the GPU DDA march, ledger, or classification disagreed with the CPU
  oracle — a real bug. Start in `../src/kernels.cu`'s `carve_one_beam` and compare against
  `../src/reference_cpu.cpp`'s `carve_one_beam_cpu` side by side.
- **A `GATE:` line fails but every `VERIFY:` line passes:** the algorithm is internally consistent
  but the measured behavior on this scene crossed a threshold — read the paired `[info]` line for
  the actual number, and see README "Expected output" / THEORY.md for what each gate's threshold
  means and how it was set.
- **Expected-line mismatch only:** the program passed its own checks but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The two
  are a contract; fix them together.
