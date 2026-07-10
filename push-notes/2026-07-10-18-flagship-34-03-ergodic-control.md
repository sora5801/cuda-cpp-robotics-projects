# Push note — 2026-07-10-18: flagship 34.03 ergodic control

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Flagship 34.03 — **ergodic control via spectral multiscale coverage** — is done (**34/505; 34 of
36**): the repository's first **[R&D] flagship**, shipped as the §2/§13 reduced-scope teaching
version that still clears every §9 gate. A first-order agent covers a bimodal information map so
that its *time-averaged* trajectory statistics match the target distribution — the ergodic
definition made literal by the gates: hotspot basin time-fractions land within 0.003–0.007 of the
numerically-integrated target masses, the spectral metric decreases 116×, and a lawnmower sweep
of identical length finishes **4.7× worse** (the negative control that proves the controller
earns its keep). The FFT hook the catalog names is real: φ_k comes from a DCT-I-via-cuFFT
even-extension identity derived from scratch and cross-checked against an independent no-FFT
cosine projection at 1.4e-11.

## What changed

- **[projects/34-theory-frontier/34.03-ergodic-control/](../projects/34-theory-frontier/34.03-ergodic-control/)** —
  complete: DCT-via-cuFFT φ_k (double precision, every call explained per rule 6), per-mode SMC
  update kernel, closed 6,000-step loop, CPU twin, five gates (transform / twin / ergodicity /
  coverage / negative-control), trajectory + metric + two-PGM artifacts, full README / THEORY /
  PRACTICE.
- **[docs/STATUS.md](../docs/STATUS.md)** — 34.03 → `done` (**34/505**).

## New projects (didactic blurbs)

**34.03 — Ergodic control** ([R&D], domain 34, flagship). Why raster coverage fails on nonuniform
information (the time-allocation argument), what ergodicity actually means in dynamical systems,
why the metric lives in a Sobolev-weighted spectral space, and how Mathew–Mezić's SMC law steers
with nothing but Fourier-coefficient mismatch. The full research frontier (multi-agent,
second-order dynamics, obstacles) is documented, not implemented — the honest [R&D] shape. The
single most interesting thing to look at: `demo/out/trajectory.csv` plotted over
`target_phi.pgm` — the path weaves densely exactly where the information is.

## How to build & run

```powershell
projects\34-theory-frontier\34.03-ergodic-control\demo\run_demo.ps1
# then plot demo\out\trajectory.csv over demo\out\target_phi.pgm; compare empirical_coverage.pgm
```

## What to study here

`THEORY.md` §The problem (coverage as time allocation) → the SMC control-law derivation → the
DCT-via-FFT identity in `src/kernels.cu`. First exercise: move the hotspots and watch the
trajectory re-allocate its time budget — no replanning code required, the metric does it.

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-10), re-run independently by the lead after the builder's self-gate:

- `Release|x64` **and** `Debug|x64` build with **zero errors, zero warnings**.
- `demo/run_demo.ps1` passes end to end: all 10 stable lines matched, exit 0.
- **Transform gate:** DCT-via-cuFFT vs independent direct cosine projection: 1.4e-11 rel.
- **Twin gate:** 8.3e-12 rel over 50 steps × 1,024 modes (FP64 throughout — documented as free at
  this size).
- **Math gates:** metric window-mean ↓116.5× with one +16.9% transient uptick (allowance
  documented); hotspot coverage Δ0.0028/Δ0.0074 vs numerically-integrated masses (tol 0.05);
  lawnmower negative control 4.7× worse (gate ≥3×).
- Fully deterministic — no RNG anywhere in the project.
- `tools/verify_project.py`: **all structural gates PASS**; no changes outside the project folder.

## Known limitations / TODOs

- [R&D] reduced scope stated in README §13: single agent, first-order dynamics, fixed target, no
  obstacles — the research versions documented in THEORY §real-world. cuFFT first-call warm-up
  variance noted honestly.

## Next push preview

35.01 magnetic microswarm fields and 36.03 lattice-robot kinematics — the final two flagships —
then the §11 standards retrospective push.
