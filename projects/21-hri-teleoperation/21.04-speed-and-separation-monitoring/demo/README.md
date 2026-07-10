# Demo — 21.04 Speed-and-separation monitoring: depth streams → minimum-distance fields at frame rate (ISO/TS 15066 helper)

> Didactic implementation — **NOT a certified safety function**. Every run prints a `NOTICE:` line
> saying exactly this. See [`../src/kernels.cuh`](../src/kernels.cuh) for the full caveat.

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

A synthetic overhead depth camera watches a small collaborative work cell for 8 seconds (240 frames
at 30 Hz): a SCARA-style robot arm performs its own reach cycle while an anonymous person (a
torso+arm capsule pair — no identity, no pose beyond that) walks in from a lane beside the cell,
passes close to the robot's tool, and walks back out. Every frame, the GPU pipeline renders a
top-down depth image, classifies pixels BACKGROUND/ROBOT/HUMAN (background subtraction + the
robot's own known-pose self-filter), finds the minimum distance from any HUMAN pixel to the nearest
robot capsule, and drives a NORMAL → REDUCED → PROTECTIVE_STOP → REDUCED → NORMAL state machine by
comparing that distance against two ISO/TS-15066-*style* protective separation distances.

Two artifacts land in `demo/out/` (git-ignored scratch, regenerated every run):

- **`distance_field.pgm`** — the dense per-pixel clearance field (distance to the nearest robot
  capsule from every point the camera can see), rendered at the frame the run itself measured as
  the closest approach. Dark = close (dangerous), bright = far (safe) — the same convention as
  project 07.09's `distance.pgm`. View with any image tool that reads PGM (GIMP, IrfanView, or
  `matplotlib.pyplot.imread`).
- **`ssm_timeline.csv`** — one row per frame: `d_min`, both S_p thresholds, the SSM state, and the
  closest robot capsule's name. Plot `d_min` against the two threshold columns to see the whole
  approach/retreat cycle and exactly where the state machine escalates and (after its hysteresis
  hold) recovers.

The program does **not** just print PASS/FAIL: it computes, independently of the pixel pipeline, the
*true* closed-form minimum distance between the human's and robot's capsules from the scenario's own
parametric geometry, and checks the pixel pipeline against it on **every frame** — not just that the
kernels agree with their own CPU oracle (the usual §5 gate), but that the whole pipeline is measuring
the right thing, within a bound it derives and prints (see the `GATE ...:` lines below).

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, and every MEASURED number this run produced (thresholds, margins, transition frames, worst-case bounds). | No — these are honest per-run measurements, not fixed constants. |
| `NOTICE:`   | The safety caveat, every run. | Yes — stable. |
| `PROBLEM:`  | The exact problem instance (image size, cell size, frame count). | Yes — stable (demo runs with no args). |
| `SCENARIO:` | The loaded scenario's human path. | Yes — stable. |
| `VERIFY:`   | GPU-vs-CPU-oracle agreement on the three kernels (§5 gate). | Yes — stable text; the measured numbers behind it are on the `[info] verify:` line. |
| `ARTIFACT:` | Confirms both files above were written. | Yes — stable. |
| `[time]`    | GPU kernel timing — a **teaching artifact, never a benchmark claim** (single-shot, one machine). | No. |
| `GATE ...:` | One of the four closed-form verification gates (NO-FALSE-STOP, NO-MISSED-STOP, TRANSITIONS, D_MIN BOUND) — see `../THEORY.md` "How we verify correctness" for what each proves. | Yes — stable text; measured margins are on the preceding `[info]` line. |
| `RESULT:`   | Overall PASS/FAIL verdict. The program exits nonzero on `FAIL`. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, device info, measured margins) are allowed. `#`-prefixed lines in that file are comments.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list).
- **`VERIFY: FAIL`:** the GPU kernels disagree with their CPU oracle — a real pipeline bug. Start in
  `../src/kernels.cu` and compare against `../src/reference_cpu.cpp`.
- **A `GATE ...: FAIL`:** the pixel pipeline disagrees with the closed-form ground truth beyond its
  documented bound, or the SSM state machine didn't transition where the geometry says it should.
  Start in `../src/main.cu`'s `analytic::` namespace (the ground truth) and the `HysteresisFsm`
  (the state machine) — and read `../THEORY.md` "Numerical considerations" for the two error
  sources (pixel quantization and top-down silhouette visibility) the D_MIN BOUND gate accounts for.
- **Expected-line mismatch only:** the program passed its own checks but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The two
  are a contract; fix them together.
