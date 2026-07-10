# Data — 24.01 2D magnetostatic FEA solver on GPU → motor torque-ripple/cogging parameter sweeps

Provenance, licensing, and field documentation for everything under `data/` (CLAUDE.md §8).

## The rules (repo-wide, CLAUDE.md §8)

- **Synthetic-first.** Robotics data can almost always be synthesized with full ground truth;
  `../scripts/make_synthetic.py` is the default source, and synthetic data is **labeled synthetic
  everywhere it appears**.
- **Tiny committed sample.** `sample/` holds just enough committed data that the demo runs offline
  with zero downloads. Large/raw/downloaded data lives in `downloaded/` (git-ignored).
- **Public datasets** (where one genuinely teaches more) are fetched by `../scripts/download_data.ps1`
  / `.sh` — idempotent, with source URL, expected size, and checksum documented below. **Respect every
  license**; registration-gated or no-redistribution datasets (KITTI, nuScenes) are pointed at, never
  mirrored.
- **Never fabricate.** No made-up measurements, no invented ground truth passed off as real.

## This project's data

There is no "recording" to synthesize here in the usual sensor sense: this project's data IS a motor
**design** — a cross-section geometry, materials, and a parameter-sweep plan — chosen by the project,
not measured or downloaded. Writing those constants to a committed, checksummed CSV is the synthetic-
data story for a design-tool project (CLAUDE.md §8 still applies: labeled synthetic, reproducible from
a script, tiny).

| Property | Value |
|----------|-------|
| Kind | Synthetic — a motor-design scenario (geometry + materials + sweep plan), not measured data |
| Generator / source | `../scripts/make_synthetic.py` (no RNG — every value is a fixed, documented design constant) |
| License | Synthetic — repo MIT license applies |
| Size (committed) | 405 bytes |
| Checksum | SHA-256 `b91c281b0228e558d3575364e90da6169e8301db31aaf1bfb92af1108fd19ce0` |
| Regenerate with | `python ../scripts/make_synthetic.py` (deterministic — no seed needed, every field is a fixed constant) |

### Fields / format — `sample/motor_scenario.csv`

Row-labeled, comma-separated, order-free (the exact grammar `src/main.cu`'s `load_scenario()`
parses). All lengths are **meters**; all angles the solver itself ever sees are computed from these
in radians (the file stores only dimensionless fractions and counts for angle-like quantities). No
frames/transforms apply (a 2D cross-section in a single fixed plane) beyond the standard convention
that angles are measured counter-clockwise from the +x axis, consistent with `std::atan2(y,x)` in
`src/main.cu`.

| Row | Fields | Units | Meaning |
|-----|--------|-------|---------|
| `GRID` | nx, ny | cells | Grid resolution; must equal `kGridN` (256) in `src/kernels.cuh` — checked at load time so the committed sample and the compiled solver can never silently disagree. |
| `DOMAIN` | half_w | m | Domain half-width; the solve grid spans `[-half_w, +half_w]^2`. |
| `ROTOR` | r_rotor_core, mag_thk, air_gap | m | Rotor iron core outer radius; magnet radial thickness; mechanical air gap. Magnet outer radius and stator bore radius are derived (`r_rotor_core+mag_thk`, `+air_gap`). |
| `STATOR` | r_back_iron_in, r_stator_out | m | Boundary between the slotted tooth ring and the solid back iron; stator outer radius. |
| `POLES_SLOTS` | P, S | count | Rotor magnet pole count (even) and stator slot count. |
| `MATERIALS` | mu_r_iron, mu_r_magnet, Br | unitless, unitless, tesla | Relative permeability of iron (linear model) and magnets; magnet remanence. |
| `SLOT_OPEN` | slot_open_frac | fraction (0,1) | Fraction of one slot pitch left open (air) at the stator bore. |
| `SOLVER` | omega, n_sweeps | unitless, count | SOR relaxation factor; fixed red+black sweep-PAIR budget per solve. |
| `SWEEP_ARCS` | a_1 .. a_k | fraction (0,1] | The swept magnet pole-arc fractions (the design sweep's x-axis). |
| `SWEEP_ANGLES` | n_angles | count | Rotor-angle samples per magnet pole pitch; MUST be even (the physics-sanity mean-zero check in `src/main.cu` relies on the sample set being symmetric about the half-pole-pitch point — THEORY.md derives why). |

Every numeric choice above was validated in a standalone prototype before being committed (SOR
convergence measured, both analytic gates matched their closed forms, the arc-fraction sweep showed
a genuine non-monotonic minimum) — see `THEORY.md` "How we verify correctness" and "Numerical
considerations" for the measured numbers this scenario reproduces at every run.
