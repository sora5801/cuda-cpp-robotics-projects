# Demo — 19.01 Parallel grasp-candidate scoring: antipodal sampling over point clouds

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

The demo runs the full grasp-candidate pipeline (PCA normals → candidate generation → scoring →
ranking) on three synthetic, analytically-known objects — a box, a cylinder, and a sphere — and
checks the result three different ways:

1. **GPU-vs-CPU agreement** (the §5 gate) on the box object: normals within an angle tolerance,
   candidate generation **exactly**, scoring within a relative tolerance.
2. **Analytic gates**, per object: does the top-10 ranked grasp list actually contain the
   geometrically-correct antipodal pairs (opposite box faces at the right width and axis; cylinder
   diameters perpendicular to its axis; sphere diameters)?
3. **A negative control**: 12 hand-picked box candidates on *adjacent* (non-opposite) faces — never
   antipodal, never force-closure — must ALL be rejected by the friction-cone gate; separately, the
   box's geometrically-antipodal but too-wide 100 mm axis must be rejected by the gripper-width gate,
   not silently accepted.

Two artifacts land in `demo/out/`:

- **`grasps.csv`** — the top-10 ranked grasps for each object: both contact points, the grasp axis,
  width, antipodal-quality score, friction-cone angles, and every gate's pass/fail flag.
- **`grasp_cloud.csv`** — a subsampled copy of each object's point cloud (`kind=cloud`) plus the two
  contact points of each object's top-5 grasps (`kind=grasp`, grouped by an `id` column) — enough to
  plot the cloud and draw a line segment through each grasp axis. Any plotting tool that reads CSV
  works; a minimal recipe: load the file, `scatter3d` the `kind=cloud` rows colored by `object`, then
  for each `object,id` pair with `kind=grasp` draw a line through its two rows.

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, measured timings, and measured gate statistics (candidate counts, worst GPU-vs-CPU deviations) — varies by machine and run. | No. |
| `PROBLEM:`  | The problem instance (candidate count, PCA parameters, object count). | Yes — stable. |
| `SCENARIO:` | Each object's name, point count, and ground-truth dimensions `[synthetic]`. | Yes — stable. |
| `VERIFY:`   | The three GPU-vs-CPU agreement checks (normals, candidate generation, scoring) on the box object. | Yes — stable. |
| `CHECK:`    | The analytic gates (per-object top-10 validity, box width gate, box adversarial negative control). | Yes — stable. |
| `ARTIFACT:` | Confirms `grasps.csv` / `grasp_cloud.csv` were written. | Yes — stable. |
| `RESULT:`   | `PASS`/`FAIL` verdict. The program exits nonzero on `FAIL`. | Yes — stable. |

`CHECK:` lines are deliberately **textual**, never embedding a specific measured number (width,
angle, count) — those numbers live only on the `[info]` line immediately above each check. FP32
arithmetic is not strictly associative, so which candidate wins a near-tie search can differ by a
GPU architecture (sm_75 vs. sm_86 vs. sm_89) even though every gate still passes with wide margin
(`THEORY.md` "Numerical considerations"); keeping measured numbers off the diffed lines is what
makes this demo portable across GPUs without a flaky expected-output file.

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info) are allowed. `#`-prefixed lines in that file are comments.

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name and compute capability — varies by machine. | No. |
| `PROBLEM:`  | The exact problem instance (sizes, parameters). | Yes — stable (demo runs with no args). |
| `[time]`    | CPU reference ms, GPU kernel ms, and a speed-up figure — a **teaching artifact, never a benchmark claim** (single-shot, kernel-only vs. one CPU core; first launches pay one-time init costs). | No. |
| `RESULT:`   | `PASS`/`FAIL` verdict of the GPU-vs-CPU check (tolerance documented in `../src/main.cu` and `THEORY.md`). The program exits nonzero on `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info) are allowed. `#`-prefixed lines in that file are comments.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list).
- **`RESULT: FAIL`:** the GPU result disagreed with the CPU oracle — a real bug. Start in
  `../src/kernels.cu` and compare against `../src/reference_cpu.cpp`.
- **Expected-line mismatch only:** the program passed its own check but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The two
  are a contract; fix them together.
