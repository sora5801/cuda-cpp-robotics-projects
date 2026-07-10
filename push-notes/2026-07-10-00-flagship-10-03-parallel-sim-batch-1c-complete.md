# Push note — 2026-07-10-00: flagship 10.03 parallel sim batch 1c complete

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Flagship 10.03 — the **massively parallel robot sim** (Isaac-Gym pattern) — is done, and with it
**batch 1c is complete** (01.02, 02.06, 03.01, 10.03; **16/505 overall, 16 of 36 flagships**).
Ten thousand domain-randomized cart-pole environments step in lockstep on the GPU — the whole
1,000-step farm fused into one kernel launch — at a measured **8.5 billion env-steps/second**.
The three ingredients of every GPU RL training farm are taught explicitly: lockstep stepping
(with the SoA-vs-AoS layout argument as the central GPU lesson), domain randomization (two kinds
of randomness, drawn at different times, for different reasons), and in-kernel episode reset. The
physics gate is the elegant one: an undriven cart-pole must conserve energy, and the measured
1.0e-05 relative drift over 1,000 RK4 steps matches the truncation-error scaling THEORY.md
derives — the integrator's error made visible and predicted.

## What changed

- **[projects/10-physics-simulation/10.03-massively-parallel-robot-sim/](../projects/10-physics-simulation/10.03-massively-parallel-robot-sim/)** —
  complete: fused farm-step kernel (SoA state, per-env parameters, inline reset, per-env RNG
  streams), pole-placement balance policy stress-tested across the randomized fleet, CPU lockstep
  twin, energy-conservation experiment, metrics + energy-trace artifacts, full README / THEORY /
  PRACTICE.
- **[docs/STATUS.md](../docs/STATUS.md)** — 10.03 → `done` (**16/505**).

## New projects (didactic blurbs)

**10.03 — Massively parallel robot sim** (★ beginner, domain 10, flagship). The pattern behind
Isaac Gym / MJX / Brax, on a plant small enough to fully understand (08.01's cart-pole,
deliberately — the two projects cross-reference). Why SoA beats AoS for persistent lockstep
state, why the whole horizon fuses into one launch (no per-tick host round-trips — contrast with
08.01's control loop, which *needs* the host in the loop), and why one fixed policy across
10,000 randomized plants is exactly what sim-to-real training demands. The single most
interesting thing to look at: `demo/out/energy_drift.csv` plotted — RK4's O(dt⁵) truncation error
as a visible, predicted curve.

## How to build & run

```powershell
projects\10-physics-simulation\10.03-massively-parallel-robot-sim\demo\run_demo.ps1
# then plot demo\out\energy_drift.csv and skim demo\out\env_metrics.csv
```

## What to study here

Batch 1c as a set is the perception-and-simulation column: stereo depth (01.02), scan
registration (02.06), radar detection (03.01), and the sim farm (10.03) that trains the policies
consuming them. Within 10.03: `THEORY.md` §The GPU mapping (SoA vs AoS) and §Numerical
considerations (the energy-drift derivation) → `src/kernels.cuh` (the shared host/device design
and its documented departure from the twin rule) → `src/kernels.cu`. First exercise: break
lockstep by giving each env a random horizon and measure the divergence cost.

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-10), re-run independently by the lead after the builder's self-gate:

- `Release|x64` **and** `Debug|x64` build with **zero errors and zero new warnings**.
- `demo/run_demo.ps1` passes end to end: all 9 stable lines matched, exit 0.
- **Lockstep gate (256 envs × 220 steps):** worst state deviation 4.8e-07 (tol 1e-3);
  reset counts exactly matched 256/256; steps-balanced diff 0.
- **Farm gate (10,000 envs × 1,000 steps):** all finite; reset_count exactly 5 for every env
  (the provable minimum given the episode cap — derived, then observed); balanced fraction
  0.978–1.000.
- **Energy gate:** max relative drift 1.045e-05 over 1,000 undriven steps (bound 1e-3),
  consistent with the derived O(N·dt⁵) estimate.
- Throughput (teaching artifact): ≈ 8.5B env-steps/s single-shot (run-to-run variance noted).
- `tools/verify_project.py`: **all structural gates PASS**; no changes outside the project folder.

## Known limitations / TODOs

- Single rigid plant, no contacts/articulations (what Isaac Gym's PhysX layer adds — documented),
  fixed policy rather than learning (12.06's job, which consumes this pattern).
- Retrospective queue (for the §11 standards pass after all 36 flagships): the template-wide
  Debug LNK4099 PDB race (three sightings; 01.02's documented `/ignore:4099` is the candidate
  fix), and a template-level ruling on shared `__host__ __device__` helpers vs hand-duplicated
  CPU twins (10.03's documented departure).

## Next push preview

Batch 1d: 11.01 GPU LiDAR simulator, 12.01 TensorRT deploy (with its §5 fallback path), 13.03
foothold scoring, 14.02 traversability costmaps.
