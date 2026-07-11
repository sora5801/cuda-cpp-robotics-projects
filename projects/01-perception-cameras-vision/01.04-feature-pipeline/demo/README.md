# Demo — 01.04 Feature pipeline: FAST/Harris detection, ORB descriptors, brute-force Hamming matcher

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

The full classical sparse-feature front end of visual odometry / SLAM, taught end to end on a
ground-truth-known synthetic image pair:

1. **DETECT** — FAST-9 corners on `scene_a.pgm` and `scene_b.pgm` (a rotated/translated/brightened
   view of the SAME scene) and, side by side, Harris corners on `scene_a.pgm` only.
2. **DESCRIBE** — per-keypoint intensity-centroid orientation, then a 256-bit oriented rBRIEF (ORB)
   descriptor.
3. **MATCH** — brute-force Hamming matching (both directions), Lowe ratio test + mutual-consistency
   cross-check.
4. **VERIFY** every GPU stage against an independently-written CPU reference (bit-exact where the
   math is all-integer, tolerance-checked where it is float — see `THEORY.md`).
5. **GATE** the result against FOUR independent, ground-truth-based checks that a twin comparison
   cannot provide (does the matcher recover the KNOWN transform? the KNOWN rotation? are keypoints
   repeatable under it? does matching against an UNRELATED scene correctly find nothing?).

**Artifacts written to `out/`** (git-ignored; regenerated every run):

| File | What it shows |
|------|----------------|
| `keypoints_A.ppm` | `scene_a.pgm` with FAST keypoints (green crosses) and Harris keypoints (blue circles) overlaid. |
| `keypoints_B.ppm` | `scene_b.pgm` with FAST keypoints (green crosses) overlaid. |
| `matches.ppm` | Side-by-side A\|B canvas with a line per accepted match — **green** if it lands within the ground-truth-transform gate's pixel tolerance, **red** if not (a direct visualization of the `ground_truth_transform` gate). |
| `matches.csv` | One row per accepted match: keypoint coordinates, Hamming distances, and the per-match ground-truth-inlier verdict. |
| `gates_metrics.csv` | One row per measured gate/metric, machine-readable (mirrors the `[info]`/`GATE` lines below). |

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, and every MEASURED number (keypoint counts, verify diffs, gate statistics) — varies by machine/GPU architecture. | No. |
| `PROBLEM:`  | The exact problem instance (image size, transform, algorithm parameters). | Yes — stable (demo runs with no args). |
| `DATA:`     | Confirms the synthetic sample loaded. | Yes — stable. |
| `VERIFY(...): ` | PASS/FAIL verdict that a GPU stage matches its independent CPU reference — never embeds a number (CLAUDE.md's cross-GPU-portability rule). | Yes — stable. |
| `GATE <name>:` | PASS/FAIL verdict of one of the four independent, ground-truth-based gates. | Yes — stable. |
| `[time]`    | GPU kernel time and CPU reference time — a **teaching artifact, never a benchmark claim** (single-shot; see `../src/util/timer.cuh`). | No. |
| `ARTIFACT:` | Confirms the `out/` files were written. | Yes — stable. |
| `RESULT:`   | The overall PASS/FAIL verdict (every VERIFY and every GATE must pass). The program exits nonzero on `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info, per-stage `[info]` detail) are allowed. `#`-prefixed lines in that file are
comments.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list).
- **A `VERIFY(...)` line FAILs:** the GPU result disagreed with the CPU oracle beyond its documented
  tolerance — a real bug. Start in `../src/kernels.cu` and compare against `../src/reference_cpu.cpp`
  for the stage named in the failing line (`fast`/`harris`/`describe`/`hamming`).
- **A `GATE` line FAILs but every `VERIFY` passes:** the GPU and CPU agree with each other, but the
  RESULT doesn't match physical reality — look at `matches.ppm`'s red lines and `../src/main.cu`'s
  gate implementation; this is the class of bug a twin comparison structurally cannot catch (see
  `THEORY.md` "How we verify correctness").
- **Expected-line mismatch only:** the program passed its own checks but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The two
  are a contract; fix them together.
