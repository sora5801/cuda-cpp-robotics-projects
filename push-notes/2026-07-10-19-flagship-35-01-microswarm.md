# Push note — 2026-07-10-19: flagship 35.01 microswarm

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Flagship 35.01 — **magnetic microrobot swarms** — is done (**35/505; 35 of 36**): an [R&D]
reduced-scope teaching version pairing brute-force Biot–Savart coil-field computation (720 current
segments × 65,536 grid points, ~0.9 ms GPU) with low-Reynolds swarm dynamics for 1,000
superparamagnetic beads. The physics gates are electromagnetics classics: the single-loop on-axis
closed form (2.5e-4 relative), the Helmholtz-pair uniformity property (0.18% flatness over the
workspace), and a ∇·B ≈ 0 check whose finite-difference step had to be re-derived after a
*measured* FP32 catastrophic-cancellation failure at 1 µm (documented as the trade it is). The
micro-world is taught scale-first: Re ≈ 1e-4 justifies first-order Stokes dynamics, the computed
Brownian-vs-drift ratio (0.2 vs 11 µm/step) justifies the deterministic default, and Purcell's
scallop theorem is framed honestly — these beads are gradient-pulled, not swimmers. An open-loop
3-waypoint pull schedule, planned offline through the same field model, lands the 1,000-robot
centroid within **13 µm** of every waypoint.

## What changed

- **[projects/35-micro-nano-robotics/35.01-magnetic-microrobot-swarms/](../projects/35-micro-nano-robotics/35.01-magnetic-microrobot-swarms/)** —
  complete: Biot–Savart basis-map kernel (linearity of Maxwell exploited: 4 per-unit-current maps
  combine into any coil configuration), field-combine + gradient kernels, agent-farm swarm kernel,
  CPU twins, seven gates, field PGM + trajectory CSV + density PGM artifacts, full README /
  THEORY / PRACTICE.
- **[docs/STATUS.md](../docs/STATUS.md)** — 35.01 → `done` (**35/505**).

## New projects (didactic blurbs)

**35.01 — Magnetic microswarms** ([R&D], domain 35, flagship). Scale analysis as the entry point
to microrobotics: why inertia vanishes, why Brownian motion is a design consideration you
*compute*, why external fields are the actuator of choice when robots are too small to carry
motors, and how Maxwell's linearity turns four field maps into a steering basis. The research
frontier (closed-loop imaging feedback, in-vivo, heterogeneous swarms) is documented per the
[R&D] contract. The single most interesting thing to look at: `demo/out/swarm_trajectory.csv` —
a thousand-robot centroid tracing the planned path through three waypoints, open-loop.

## How to build & run

```powershell
projects\35-micro-nano-robotics\35.01-magnetic-microrobot-swarms\demo\run_demo.ps1
# then open demo\out\field_magnitude.pgm and plot demo\out\swarm_trajectory.csv
```

## What to study here

`THEORY.md` §The problem (the scale-analysis derivations — Re, Brownian, why fields) → the
superparamagnetic force derivation → `src/kernels.cu` (basis maps + the farm). First exercise:
turn the documented Brownian term on and watch the waypoint tolerance earn its size.

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-10), re-run independently by the lead after the builder's self-gate:

- `Release|x64` **and** `Debug|x64` build with **zero errors, zero warnings**.
- `demo/run_demo.ps1` passes end to end: all 15 stable lines matched, exit 0.
- **Twin gates:** field map 1.1e-11 T; 300-step chained dynamics 1.5e-08 m.
- **Physics gates:** on-axis closed form 2.5e-4 rel (tol 1%); Helmholtz flatness 1.8e-3 (tol 2%);
  normalized |∇·B| 1.0e-4 (tol 1e-3, with the FD-step cancellation story documented); all four
  coils attract correctly (317–360 µm centroid pulls); waypoints hit at 12.2/12.6/13.3 µm
  (tol 300 µm); swarm bounded and finite throughout.
- Timing (teaching artifacts): 4 basis maps ≈ 0.9 ms GPU (vs 76 ms CPU for one); 900-step
  1,000-robot run ≈ 93 ms.
- The builder also fixed two real portability bugs (host-compiled float4; a header include-order
  accident) — noted in its report.
- `tools/verify_project.py`: **all structural gates PASS**; no changes outside the project folder.

## Known limitations / TODOs

- [R&D] reduced scope stated in README §13: 2-D workspace, open-loop feedforward (closed-loop
  imaging feedback is the documented research step), superparamagnetic point-bead model, no
  inter-bead magnetic interactions (documented).

## Next push preview

36.03 lattice-robot kinematics — the 36th and final flagship — then the §11 standards
retrospective push closes Phase 1.
