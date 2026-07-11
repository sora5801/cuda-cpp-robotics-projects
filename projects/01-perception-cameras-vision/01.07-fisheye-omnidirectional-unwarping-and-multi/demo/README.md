# Demo — 01.07 Fisheye/omnidirectional unwarping and multi-camera surround-view stitching

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

The demo runs BOTH halves of the project on the 4 committed synthetic fisheye renders
(`data/sample/fisheye_{front,left,right,rear}.ppm`):

1. **Half 1 — single-camera unwarp.** The FRONT camera's fisheye image is re-projected onto a
   rectilinear (pinhole) sub-FOV and a wide cylindrical panorama, each via a precomputed
   inverse-mapping LUT + bilinear gather (the same two-stage pattern as sibling flagship 01.01).
2. **Half 2 — 4-camera bird's-eye view.** One kernel, one thread per output pixel, blends all 4
   cameras' contributions (through the shared rig geometry) into a single top-down stitched image,
   with a per-pixel camera-coverage bitmask.
3. Every kernel is checked against `../src/reference_cpu.cpp`'s independent CPU twin (`VERIFY:`).
4. **Seven physical gates** run next — see `../README.md`'s "The algorithm in brief" for what each
   proves; in short: the fisheye model round-trips exactly (`model_roundtrip`), a real straight edge
   comes out straight after rectilinear unwarp but curved in the raw fisheye
   (`straightness_rectilinear` / `distortion_negative_control`), the BEV matches ground truth on flat
   ground (`bev_ground_truth`) but visibly, deliberately fails near tall objects
   (`flat_ground_assumption` — the project's central teaching point), the 4 cameras agree with each
   other in their overlap zones (`seam_consistency`), and the rig covers its design radius
   (`coverage`).
5. Writes 7 artifacts to `demo/out/` (see below).

**Look at the artifacts** (viewable in any PPM/PGM-aware image tool — GIMP, IrfanView, VS Code's
image preview): `fisheye_front.ppm` (the raw circular fisheye view — sky, ground, a dashed lane
line, a boundary edge, and a red object, all visibly CURVED); `rectilinear.ppm` (a narrow forward
sub-FOV — the same boundary edge is now visibly STRAIGHT); `cylindrical.ppm` (a much wider
panorama, still without rectilinear's extreme edge stretching); `bev.ppm` (the stitched top-down
view — look for the lane lines, the light "loading zone", and the RADIAL GHOSTING/smearing around
each tall object, stretching away from whichever camera saw it — this is the flat-ground assumption
failing, on purpose); `coverage_map.pgm` (which camera(s) cover each BEV pixel — see below);
`error_heatmap.pgm` (BEV-vs-ground-truth error, brightest exactly at the ghosted object regions).

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, and every MEASURED number behind a gate/verify verdict — varies slightly by GPU architecture (bilinear FMA-contraction differences; see `../src/main.cu`'s output-contract comment). | No. |
| `PROBLEM:`  | The exact problem instance: fisheye/rectilinear/cylindrical/BEV sizes and camera parameters. | Yes — stable. |
| `DATA:`     | Which sample was loaded and its provenance. | Yes — stable. |
| `[time]`    | LUT-build, unwarp, and BEV-compose kernel times, plus the CPU oracle time — **teaching artifacts, never benchmark claims** (single-shot, first-launch JIT costs included). | No. |
| `VERIFY:`   | `PASS`/`FAIL` — every GPU kernel agrees with `reference_cpu.cpp`'s independent twin within the documented per-stage tolerance. | Yes — stable. |
| `GATE <name>:` | `PASS`/`FAIL` verdict of one of the seven physical gates; the measured number behind each verdict is on the following `[info]` line. | Yes — stable (7 lines). |
| `ARTIFACT:` | Confirms every `demo/out/` file was written. | Yes — stable. |
| `RESULT:`   | Overall `PASS`/`FAIL` — `VERIFY` AND all seven gates AND the artifact write must all succeed. The program exits nonzero on `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info) are allowed. `#`-prefixed lines in that file are comments.

### Reading `coverage_map.pgm`

The RAW coverage buffer every gate computes from is a 4-bit-per-pixel bitmask (bit 0 = front, bit 1
= left, bit 2 = right, bit 3 = rear — see `../src/kernels.cuh`'s `kCamFront..kCamRear`). The
committed artifact is a **display-rescaled** copy (each value multiplied by 17, so the 16 possible
bitmask values spread across the full 0-255 range and are distinguishable by eye) — `gates_metrics.csv`
and every gate operate on the true, unscaled bitmask internally.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list).
- **`RESULT: FAIL`:** the GPU result disagreed with the CPU oracle, or a gate's measured number
  fell outside its documented tolerance — a real bug (or, for `bev_ground_truth`/`seam_consistency`,
  possibly a legitimately different fisheye-resolution/rig-geometry choice; `../src/main.cu`'s
  tolerance block explains what these two gates' numbers are physically dominated by). Start in
  `../src/kernels.cu` and compare against `../src/reference_cpu.cpp`.
- **Expected-line mismatch only:** the program passed its own check but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The two
  are a contract; fix them together.
