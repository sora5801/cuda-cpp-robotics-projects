# Demo — 01.06 AprilTag / ArUco GPU detector-decoder for high-rate fiducial localization

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

The demo runs the full six-stage GPU pipeline (adaptive threshold -> connected-component labeling ->
quad extraction -> DLT homography -> perspective grid decode -> pose from homography) on **three**
committed synthetic scenes:

- **`scene_main.pgm`** — 6 tags under full perspective (varied depth, tilt, and in-plane rotation).
- **`scene_distractor.pgm`** — **no tags at all**: a checkerboard block and several filled disks,
  deliberately corner-rich, to stress the false-positive defenses.
- **`scene_robustness.pgm`** — 4 tags with payload bits deliberately flipped: two at exactly the
  dictionary's correction capacity (must still decode correctly), two one bit beyond it (must be
  rejected) — the dictionary's own built-in negative control.

For each scene it runs the pipeline TWICE — once on the GPU, once on an independent CPU oracle — and:

1. **VERIFIES** every intermediate (local mean, foreground mask, connected-component labels,
   candidate statistics, refined corners, homography, decoded ID, pose) agrees within a documented
   tolerance (`VERIFY:` line).
2. Runs **five independent gates**, none of which is just "VERIFY again" (they check the pipeline
   against ground truth the pipeline's own code never touches): `detection` (all 6 main-scene tags
   found, correct ID, no extras), `corner_accuracy` (max corner error vs. the renderer's ground
   truth), `pose` (rotation/translation error vs. the renderer's analytic camera pose),
   `decode_robustness` (the bit-flip negative control), and `false_positive` (zero accepted
   detections on the tag-free distractor scene).
3. Writes the artifacts below to `demo/out/`.

**Look at the artifacts** (PPM files open in GIMP, IrfanView, VS Code's image preview, or any
PPM-aware viewer): `detections_overlay.ppm` (scene_main with each accepted tag's quad outlined in
green and its decoded ID drawn in yellow) and `decoded_grid_debug.ppm` (the 6x6 grid the decoder
actually sampled for one tag, upsampled for inspection — the outer ring drawn in red so you can see
the border-black requirement, the interior in black/white as sampled).

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, and every MEASURED number behind a VERIFY/gate verdict (max GPU-vs-CPU diffs, per-tag corner/pose errors). Varies slightly by GPU architecture (FMA-contraction differences in the corner-refinement search — see `../src/main.cu`'s output-contract comment and `THEORY.md` "Numerical considerations"). | No. |
| `PROBLEM:`  | The exact problem instance: scene size, dictionary size/geometry/correction capacity, tag counts per scene. | Yes — stable (demo runs with no args). |
| `DATA:`     | Which sample was loaded and its provenance. | Yes — stable. |
| `[time]`    | Per-scene GPU pixel-parallel time, GPU candidate-parallel time, CPU oracle time, and CCL sweep count — **teaching artifacts, never benchmark claims** (single-shot, first-launch JIT costs included). | No. |
| `VERIFY:`   | `PASS`/`FAIL` — every GPU stage agrees with `reference_cpu.cpp`'s independent twin within its documented tolerance, across all 3 scenes. | Yes — stable. |
| `GATE <name>:` | `PASS`/`FAIL` verdict of one of the five gates (see "What the demo demonstrates" above); the measured number(s) behind each verdict are on the following `[info]` line. | Yes — stable (5 lines). |
| `ARTIFACT:` | Confirms every `demo/out/` file was written. | Yes — stable. |
| `RESULT:`   | Overall `PASS`/`FAIL` — `VERIFY` AND all five gates AND the artifact write must all succeed. The program exits nonzero on `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info, per-tag measured numbers) are allowed. `#`-prefixed lines in that file are
comments.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list).
- **`RESULT: FAIL`:** the GPU result disagreed with the CPU oracle, or a gate failed against ground
  truth — a real bug. Start in `../src/kernels.cu` and compare against `../src/reference_cpu.cpp`;
  `demo/out/detections.csv` and the overlay PPM are the fastest way to see WHICH tag/scene is wrong.
- **Expected-line mismatch only:** the program passed its own check but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The two
  are a contract; fix them together.
