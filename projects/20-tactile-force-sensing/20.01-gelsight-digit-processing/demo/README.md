# Demo — 20.01 GelSight/DIGIT processing: contact patch, shear field via optical flow, slip detection in real time

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

**One synthetic 100-frame gel-sensor sequence, three physics gates.** The demo renders a deterministic
sequence — a sphere presses into a marker-printed gel (0-24), then shears sideways while fully stuck
(24-60), then partially slips outward-in from the contact edge (60-100) — and runs the full GPU tactile
pipeline (contact mask → morphological open → patch stats → marker detect → marker track) on every
frame, cross-checking the GPU against a plain-C++ CPU oracle **exactly, frame by frame** (this project's
whole pipeline is integer/threshold arithmetic — no floating-point tolerance needed anywhere in that
check). It then compares the algorithm's own measurements against the **physics that generated the
scene** — the Hertzian contact footprint, the commanded shear translation, and the Cattaneo-Mindlin
slip-onset frame — not against itself:

```
CONTACT: patch area mean rel err 1.3% (max 1.3%) vs Hertzian footprint, centroid max err 0.13px, over 8 press-hold frames
SHEAR: mean tracked displacement max err 0.00px vs commanded 5.0px, over 12 shear-hold frames
SLIP: detected onset at frame 85 (modeled 86, |err|=1, tol +/-2); false slip during stick phase (frames 0-59): no
```

**This demo writes three artifacts** into `out/` (git-ignored, regenerated each run):

| File | What it shows |
|------|----------------|
| `contact_mask.pgm` | The final (post-morphological-open) binary contact mask at a representative PRESS-hold frame — open it and see a clean disk with speckle already removed by erosion+dilation. |
| `shear_field.csv` | Every marker's rest position and measured displacement at a representative SHEAR-hold frame — plot it as a quiver/vector field: markers inside the contact patch (`in_contact=1`) should all point the same commanded direction by the same magnitude; markers outside should show ~zero displacement. |
| `slip_timeline.csv` | Per-frame slip score and the phase it falls in across the WHOLE 100-frame run — **the teaching plot**. Plot `slip_score` vs `frame`: flat near zero through BASELINE/PRESS/SHEAR, then climbing through SLIP as the Cattaneo-Mindlin annulus grows, crossing the declare threshold (0.5) right around the modeled onset frame. |

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, the resolved scenario file path, per-frame mismatch count. | No. |
| `PROBLEM:`  | The problem instance (image size, frame counts per phase, marker count). | Yes — stable. |
| `SCENARIO:` | The loaded scenario (indenter shape, contact radius at max depth, commanded shear). | Yes — stable. |
| `[time]`    | Total/average GPU kernel time across the 100-frame run (5 kernels/frame) — a **teaching artifact, never a benchmark claim** (single-shot, one machine). | No. |
| `VERIFY:`   | The GPU-vs-CPU EXACT-equality gate, checked on EVERY one of the 100 frames (mask+morphology, patch stats, marker detect+track) — this project's pipeline is all integer/threshold arithmetic, so this is bit-for-bit equality, not a tolerance. Exits nonzero on any mismatch before the ground-truth gates even run. | Yes — stable. |
| `CONTACT:` / `SHEAR:` / `SLIP:` | The three ground-truth gates that decide `RESULT:` — measured algorithm output vs. the physics that generated the scene, with the tolerances (and their measured headroom) documented at the top of `../src/main.cu`. | Yes — stable. |
| `ARTIFACT:` | Confirms the three files above were written. | Yes — stable. |
| `RESULT:`   | `PASS`/`FAIL` verdict of all three ground-truth gates. The program exits nonzero on `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info) are allowed. `#`-prefixed lines in that file are comments.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list).
- **`VERIFY: FAIL`:** the GPU result disagreed with the CPU oracle on at least one frame — a real bug,
  and (because every pipeline operation here is integer/threshold arithmetic) never a rounding
  artifact. Start in `../src/kernels.cu` and diff against `../src/reference_cpu.cpp` function by
  function; the `[info] verify:` line gives the total mismatch count across all 100 frames and every
  checked stage.
- **`RESULT: FAIL` with `VERIFY: PASS`:** the pipeline is internally consistent but a ground-truth gate
  was not met — look at the `CONTACT:`/`SHEAR:`/`SLIP:` lines against the thresholds documented at the
  top of `../src/main.cu`, and at `slip_timeline.csv`/`shear_field.csv` to see WHERE it diverged from
  the modeled physics.
- **Expected-line mismatch only:** the program passed its own checks but printed different stable
  lines — someone changed the output (or the scenario, or a tuning constant in `kernels.cuh`) without
  updating `expected_output.txt` (or vice versa). The two are a contract; fix them together.
