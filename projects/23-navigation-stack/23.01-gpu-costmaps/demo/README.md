# Demo — 23.01 GPU costmaps: inflation, raytrace clearing, multi-layer fusion

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

A simulated warehouse AMR drives from `(1.0, 1.0)` to `(11.0, 11.0)` m across a 256x256 (12.8 m x
12.8 m) synthetic room with four pillar obstacles, replanning at 10 Hz. Every tick: a 360-beam LiDAR
scan is simulated against the true map (host), a three-kernel GPU pipeline turns that scan into a
fused master costmap (raytrace mark/clear -> bounded-radius inflation -> per-cell max fusion), a
fourth GPU kernel scores 4096 candidate `(v,w)` arcs against that costmap (the Dynamic Window
Approach), and the best admissible one drives the simulated differential-drive plant one tick. Before
any of that closed-loop driving starts, the demo runs its own correctness gate: one full costmap
update is checked **byte-exact** against a plain-C++ CPU oracle, and one DWA scoring pass is checked
against a CPU oracle within a documented floating-point tolerance.

The demo writes two artifacts into `out/` (git-ignored, regenerated every run):

| File | What it is |
|---|---|
| `out/costmap.pgm` | The MASTER costmap at the final tick — a viewable grayscale image where the byte value IS the cost (0 = free, ~1-253 = inflation gradient, 254 = lethal). Open it in any image viewer that reads PGM (GIMP, IrfanView, VS Code's own preview, or `python -c "from PIL import Image; Image.open('out/costmap.pgm').show()"`). |
| `out/path.csv` | The full driven trajectory: `t_s,x_m,y_m,theta_rad,v_ms,w_rads`, one row per control tick. Plot `x_m` vs `y_m` to see the robot weave around the four pillars; plot `v_ms`/`w_rads` vs `t_s` to see the dynamic window in action (both change smoothly, never jump, because every sample is acceleration-limited). |

**What to notice:** the pillars are clearly visible as bright (lethal) squares in `costmap.pgm`, each
surrounded by a soft gray halo (the inflation gradient) that the path in `path.csv` never crosses
into full white.

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, file paths, and measured diagnostic numbers — vary by machine/run. | No. |
| `PROBLEM:`  | The exact problem instance (grid size, LiDAR, DWA window). | Yes — stable. |
| `MAP:`, `SCENARIO:` | The loaded world and task, with numbers measured from the committed sample. | Yes — stable. |
| `[time]`    | CPU vs GPU timings — a **teaching artifact, never a benchmark claim** (single-shot, one machine). | No. |
| `VERIFY COSTMAP:`, `VERIFY DWA:` | The two independent GPU-vs-CPU correctness gates (§5). | Yes — stable. |
| `ARTIFACT:` | Confirms `out/costmap.pgm` and `out/path.csv` were written. | Yes — stable. |
| `RESULT:`   | `PASS`/`FAIL` verdict: both verify gates passed AND the goal was reached AND zero lethal-cell entries occurred along the driven path. The program exits nonzero on `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info) are allowed. `#`-prefixed lines in that file are comments.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list).
- **`VERIFY COSTMAP: FAIL` or `VERIFY DWA: FAIL`:** the GPU result disagreed with the CPU oracle — a
  real bug. Start in `../src/kernels.cu` and compare against `../src/reference_cpu.cpp` kernel by
  kernel (the file headers in both are written to be read side by side).
- **`RESULT: FAIL` with both verify gates passing:** the planner itself misbehaved — either it never
  reached the goal within the 500-step cap, or (far more seriously) the driven path touched a lethal
  cell despite passing verification. The `[info] final pose` and `emergency brakes` lines are the
  first place to look; `out/path.csv` plotted against `out/costmap.pgm` usually shows exactly where.
- **Expected-line mismatch only:** the program passed its own check but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The two
  are a contract; fix them together.
