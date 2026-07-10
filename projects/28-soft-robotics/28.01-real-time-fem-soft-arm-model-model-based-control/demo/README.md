# Demo — 28.01 Real-time FEM soft-arm model + model-based control (GPU SOFA-style)

One command builds (if needed), runs on the committed sample, and verifies the output:

```powershell
.\run_demo.ps1          # Windows (the required path: VS 2026 + CUDA 13.3)
```

```bash
./run_demo.sh           # Linux/macOS best-effort bonus (CMake path)
```

## What the demo demonstrates

One staged run (~7 s on the reference RTX 2080 SUPER) that takes a GPU soft-body simulator from
"does it match a CPU?" all the way to "can a controller use it?":

1. **VERIFY** — 500 steps of full corotational dynamics through the GPU's scatter+atomics assembly
   AND the CPU's deterministic sequential twin; every position/velocity must agree within a
   documented reassociation-aware tolerance.
2. **GATE static-deflection** — settle under a small tip load; the tip sag must match the analytic
   Euler-Bernoulli formula (measured ~0.6% error vs a 30% allowance).
3. **GATE first-mode-frequency** — release and ring; the tip trace's zero-crossing frequency must
   match the analytic cantilever first mode (measured ~1.4% error vs 20%).
4. **GATE energy-conservation** — the same ring's total energy (KE + elastic PE) must stay within
   8% of its initial value (measured ~4.3%) — the symplectic-Euler bounded-drift story, with the
   measured drift budget documented in `../src/main.cu`.
5. **IDENTIFY** — the controller measures its own model: a small tendon-tension differential is
   applied, the FEM settles, and the tip-deflection-per-tension Jacobian is read off (~0.012 m/N).
6. **SETPOINT 0–3** — closed-loop PI tracking of a step+hold tip-setpoint sequence through the
   antagonistic tendon pair, with per-setpoint rise time / overshoot / steady-state error printed
   and gated.
7. **REALTIME** — the measured real-time factor across every stepping phase (~1.7x on the reference
   GPU): the model simulates faster than the arm it represents, which is the entire premise of
   model-based soft-arm control.

**Artifacts written to `demo/out/`** (git-ignored; regenerated every run):

| File | Contents | Plot it |
|------|----------|---------|
| `tip_trajectory.csv` | `t_s, setpoint_y_m, tip_y_m, T_top_N, T_bottom_N` per control tick — the closed-loop story | tip + setpoint vs time: the step responses; tensions vs time: the antagonistic differential at work |
| `arm_snapshots.csv` | every node's `(x, y)` at three labeled instants (statically loaded / post-probe / end of control) | scatter one snapshot to SEE the bent arm's shape |
| `arm_deformed.pgm` | 480x140 ASCII-PGM rasterization of the final deformed mesh (openable in most viewers/editors) | just open it |

## How to read the output

| Line prefix | Meaning | Checked against `expected_output.txt`? |
|-------------|---------|----------------------------------------|
| `[demo]`    | Which project/demo this is. | Yes — stable. |
| `[info]`    | GPU name, measured errors/gains/metrics — every gate's actual numbers. | No (they vary by machine/run — the VERDICT lines are what is checked). |
| `PROBLEM:` / `MESH:` / `SCENARIO:` | The exact problem instance (mesh, material scale, dt, setpoint plan). | Yes — stable. |
| `[time]`    | Wall/GPU timings — **teaching artifacts, never benchmark claims** (single-shot, one machine). | No. |
| `VERIFY:` / `GATE ...:` / `IDENTIFY:` / `SETPOINT n:` / `REALTIME:` | PASS/FAIL verdicts with their fixed thresholds. | Yes — stable. |
| `ARTIFACT:` | What was written to `demo/out/`. | Yes — stable. |
| `RESULT:`   | The overall verdict; the program exits nonzero on FAIL. | Yes — stable. |

The runner scripts do a **subset diff**: every non-comment line of
[`expected_output.txt`](expected_output.txt) must appear verbatim in the output; extra lines
(timings, measurements) are allowed. `#`-prefixed lines in that file are comments.

## If it fails

- **Build fails:** see [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md) (toolchain
  install, CUDA/VS integration, GPU architecture list).
- **`SCENARIO: MALFORMED`:** `data/sample/arm_scenario.csv` disagrees with the compiled constants —
  regenerate it (`python ../scripts/make_synthetic.py`) or reconcile `../src/kernels.cuh` (the
  cross-check exists precisely to catch half-done model edits).
- **`VERIFY: FAIL`:** the GPU scatter assembly disagreed with the CPU oracle — a real bug. Start in
  `../src/kernels.cu` beside `../src/reference_cpu.cpp` (they are line-by-line twins).
- **`GATE ...: FAIL`:** the physics drifted from its analytic anchors — check the `[info]` line's
  measured number against the gate's documented allowance before touching anything.
- **Expected-line mismatch only:** the program passed its own checks but printed different stable
  lines — someone changed the output without updating `expected_output.txt` (or vice versa). The two
  are a contract; fix them together.
