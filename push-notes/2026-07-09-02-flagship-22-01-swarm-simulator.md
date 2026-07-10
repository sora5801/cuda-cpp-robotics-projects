# Push note — 2026-07-09-02: flagship 22.01 swarm simulator

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Flagship 22.01 — the **100k-agent swarm simulator** — is done: 100,000 boids flock in real time
(≈1 ms per step all-in on the reference GPU) using a counting-sort uniform-grid neighbor
structure, with a pheromone grid layer (deposit → diffuse → decay → gradient-following) as the
stigmergy milestone. Verification is a lockstep march against a brute-force O(N²) CPU oracle at
N=4,096 for 100 steps — worst position deviation 1.5e-05 m — and the emergent behavior is gated
quantitatively: mean local alignment 0.974 against a 0.5 threshold (random baseline ≈ 0). The
catalog bullet bundles flocking · pheromone grids · stigmergy; per the §2 bundle rule the README
documents which milestones are implemented (all three, as one system) and what the variants would
add.

## What changed

- **[projects/22-multi-robot-swarms/22.01-100k-agent-swarm-simulator/](../projects/22-multi-robot-swarms/22.01-100k-agent-swarm-simulator/)** —
  complete: bin/scatter/flock/pheromone kernels, brute-force CPU oracle, lockstep verification,
  emergence metrics, density/pheromone PGM + positions CSV artifacts, full README / THEORY /
  PRACTICE.
- **[docs/STATUS.md](../docs/STATUS.md)** — 22.01 → `done` (**7/505**).

## New projects (didactic blurbs)

**22.01 — 100k-agent swarm simulator** (★ beginner, domain 22, flagship). Teaches the two ideas
every large-scale multi-agent GPU code rests on: spatial binning (counting sort per step — why
O(N²) is impossible at 100k, and how a warp-friendly 3×3-cell gather replaces it) and
**atomic-reordering honesty** — atomicAdd makes float sums order-nondeterministic, which is THE
numerics teaching point of the project and drives the verification design (lockstep at small N
with a justified tolerance). The pheromone layer reuses 07.09's stencil pattern for
diffusion-decay. The single most interesting thing to look at: `demo/out/density.pgm` after 15
simulated seconds — the flocks are visible as filaments.

## How to build & run

```powershell
projects\22-multi-robot-swarms\22.01-100k-agent-swarm-simulator\demo\run_demo.ps1
# then open demo\out\density.pgm and demo\out\pheromone.pgm
```

## What to study here

Project `README.md` → `THEORY.md` (boids rules → binning complexity → the atomic-associativity
section) → `src/kernels.cu` (bin → scatter → gather-flock → pheromone stencil). First exercise:
crank the pheromone gain and watch stigmergic trails dominate the flocking.

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-09), re-run independently by the lead after the finisher's self-gate:

- `Release|x64` **and** `Debug|x64` build with **zero errors, zero warnings**.
- `demo/run_demo.ps1` passes end to end: all 6 stable lines matched, exit 0; artifacts valid
  (two 256×256 P5 PGMs + 1000-row CSV).
- **Lockstep oracle gate (N=4,096, 100 steps):** worst deviations pos 1.5e-05 m, vel 1.2e-07 m/s,
  pheromone 1.2e-07 (tol 1e-3 each); CPU O(N²) ≈ 1.6 s vs GPU ≈ 0.07 s (teaching artifact).
- **Emergence gate (N=100,000, 300 steps):** mean local alignment 0.974 (threshold ≥ 0.5; random
  ≈ 0), polarization 0.044, 0 escaped/NaN agents. Per-step timings: bin ≈ 0.45 ms (incl. host
  scan round-trip), flock kernel ≈ 0.51 ms, pheromone ≈ 0.014 ms.
- `tools/verify_project.py`: **all structural gates PASS**; no changes outside the project folder.

## Known limitations / TODOs

- 2-D, fixed arena, host-side scan in the binning pipeline (GPU scan is a README exercise),
  statistical rather than bitwise determinism at full N (documented atomic-reordering story) —
  all in README §Limitations.
- 31.01 finisher is running (last of interrupted batch 1a); its partial `src/` remains in-tree
  marked `in-progress`.

## Next push preview

31.01 Hamilton–Jacobi reachability — the safety-layer flagship, verified against the double
integrator's closed-form bang-bang solution. Then batch 1b: 06.05 STOMP, 15.01 min-snap, 17.01
Lambert/porkchop, 23.01 costmaps+DWA.
