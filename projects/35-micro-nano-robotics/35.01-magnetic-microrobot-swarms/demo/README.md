# Demo — 35.01 Magnetic microrobot swarms: Biot-Savart field computation + swarm dynamics

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

Two things, in sequence:

1. **A GPU Biot-Savart field solver.** The 4-coil arrangement's per-unit-current field maps are
   computed on the GPU (`biot_savart_basis_kernel`, one thread per grid cell, 720 segments summed per
   thread), then checked FIVE independent ways: a GPU-vs-CPU agreement check (`VERIFY_FIELD`), and
   three ANALYTIC PHYSICS gates against textbook closed forms / symmetry arguments that never touch the
   grid at all — a single loop's on-axis field (`GATE_ONAXIS`), a Helmholtz pair's central flatness
   (`GATE_HELMHOLTZ`), and a full-3D divergence sanity check (`GATE_DIVERGENCE`). `demo/out/
   field_magnitude.pgm` visualizes one illustrative combined-current field map.
2. **A low-Reynolds-number microrobot swarm steered by that field.** 1000 simulated superparamagnetic
   microrobots are pulled through an OPEN-LOOP, offline-designed 3-waypoint schedule (energize North,
   then East, then South) by the field's gradient. `GATE_ATTRACT` confirms single-coil energization
   pulls the swarm toward that coil (all 4 coils individually); `GATE_WAYPOINTS` confirms the real,
   dispersed 1000-robot swarm's centroid tracks the schedule's single-particle offline plan; `GATE_BOUNDS`
   confirms the swarm never left the mapped workspace or produced a non-finite position.
   `demo/out/swarm_trajectory.csv` (centroid + 5 sample robots over time) and `demo/out/swarm_final.pgm`
   (a density snapshot of where the swarm ended up) are the artifacts to inspect.

This is an **[R&D] catalog bullet's reduced-scope teaching version** — see `../README.md`
"Limitations & honesty" for exactly what is scoped in vs. documented-only.

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name; measured tolerances/distances/Reynolds number/planned waypoints/GATE_ATTRACT displacements — varies by machine and build config. | No. |
| `PROBLEM:` / `SCENARIO:` | The exact coil/field and swarm/schedule problem instance. | Yes — stable (demo runs with no args, reading `data/sample/microswarm_scenario.csv`). |
| `VERIFY_FIELD:` / `VERIFY_DYNAMICS:` | GPU-vs-independent-CPU-reference agreement verdicts. | Yes — stable (PASS/FAIL only; the achieved tolerance is on the paired `[info]` line). |
| `GATE_ONAXIS:` / `GATE_HELMHOLTZ:` / `GATE_DIVERGENCE:` | The three analytic physics gates (§ above). | Yes — stable. |
| `GATE_ATTRACT:` / `GATE_WAYPOINTS:` / `GATE_BOUNDS:` | The three swarm-dynamics physics gates. | Yes — stable. |
| `ARTIFACT:`  | Confirms a file was written under `demo/out/`. | Yes — stable. |
| `[time]`    | Kernel/CPU timings — a **teaching artifact, never a benchmark claim** (single-shot, one machine; first launches pay one-time init costs). | No. |
| `RESULT:`   | Overall `PASS`/`FAIL` verdict (the AND of every gate above). The program exits nonzero on `FAIL`. | Yes — stable. |

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
