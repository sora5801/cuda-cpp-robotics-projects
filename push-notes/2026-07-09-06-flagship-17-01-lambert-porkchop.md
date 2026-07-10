# Push note — 2026-07-09-06: flagship 17.01 lambert porkchop

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Flagship 17.01 — **batched Lambert solvers + porkchop plot generation** — is done: a 512×512 grid
of departure/arrival epochs between two synthetic circular heliocentric orbits, one GPU thread
solving one Lambert problem (universal variables, Stumpff functions, fixed-schedule bisection),
producing the classic porkchop picture in under a millisecond of kernel time. The analytic gate is
the highlight: for coplanar circular orbits the true optimum is the **Hohmann transfer**, derived
in closed form from vis-viva — and the grid minimum lands 0.14% above it, exactly the
grid-resolution gap the docs predict. A genuinely teachable coincidence surfaced on the way: the
Lambert Δθ=π singularity sits *on* the Hohmann optimum, which is why the solver's near-singular
NaN band brackets the porkchop's sweet spot — documented, not hidden.

## What changed

- **[projects/17-locomotion-space/17.01-batched-lambert-solvers-porkchop-plot-generation/](../projects/17-locomotion-space/17.01-batched-lambert-solvers-porkchop-plot-generation/)** —
  complete: Lambert kernel (stable half-angle formulation, five-way cell-status policy), CPU twin
  with a bit-identical iteration schedule, Hohmann closed-form oracle, porkchop PGM + minimum CSV
  artifacts, full README / THEORY / PRACTICE (canonical-units contract with SI conversion table).
- **[docs/STATUS.md](../docs/STATUS.md)** — 17.01 → `done` (**11/505**).

## New projects (didactic blurbs)

**17.01 — Lambert + porkchop** (★ beginner, domain 17, flagship). Teaches the two-body problem as
mission designers actually use it: what a Lambert problem is (boundary-value orbit determination),
why porkchop plots have their bowtie shape (synodic geometry), and how universal variables +
Stumpff functions make one solver work across elliptic/parabolic/hyperbolic transfers. GPU lesson:
an embarrassingly parallel grid where *iteration-count divergence* across a warp is the honest
cost model. The single most interesting thing to look at: `demo/out/porkchop.pgm` — bright
valleys, dark exclusion bands, and the near-singular seam running through the optimum.

## How to build & run

```powershell
projects\17-locomotion-space\17.01-batched-lambert-solvers-porkchop-plot-generation\demo\run_demo.ps1
# then open demo\out\porkchop.pgm
```

## What to study here

`THEORY.md` §The math (vis-viva → the Hohmann derivation — the ground truth) and §Numerical
considerations (Stumpff series switchover near z=0) → `src/kernels.cu`. First exercise: add the
long-way (Type II) branch and watch the porkchop's second lobe appear.

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-09), re-run independently by the lead after the builder's self-gate:

- `Release|x64` **and** `Debug|x64` build with **zero errors, zero warnings**.
- `demo/run_demo.ps1` passes end to end: all 8 stable lines matched, exit 0.
- **GPU-vs-CPU gate:** 0/262,144 cell-status mismatches; worst relative Δv deviation 8.4e-05
  (bound 1e-3).
- **NaN-policy gate:** near-singular + non-converged = 2.19% of attempted cells (bound 6%);
  0 non-converged.
- **Analytic gate:** grid minimum Δv = 0.188146 LU/TU vs closed-form Hohmann 0.187883 (+0.14%),
  TOF 4.375 vs 4.454 TU (−1.8%) — within the documented grid-resolution windows; cross-checked
  against an independent double-precision prototype.
- Timing (teaching artifact): 512² grid ≈ 0.8 ms GPU vs ≈ 102 ms CPU.
- `tools/verify_project.py`: **all structural gates PASS**; no changes outside the project folder.

## Known limitations / TODOs

- Coplanar circular synthetic orbits (no ephemerides — JPL SPICE/Horizons documented as the
  public-data production path), short-way single-revolution transfers only; long-way and
  multi-rev are documented exercises. Export-control orientation (ITAR/EAR) noted didactically in
  PRACTICE §4 per the SYSTEM_DESIGN item-6 map.

## Next push preview

23.01 GPU costmaps + DWA closes worker batch 1b; then batch 1c begins on the remaining 24
flagships (01.02 stereo SGM, 02.06 ICP, 03.01 FMCW+CFAR, …).
