# Data — 20.01 GelSight/DIGIT processing: contact patch, shear field via optical flow, slip detection in real time

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

| Property | Value |
|----------|-------|
| Kind | **Synthetic** — a SCENARIO (task definition), not recorded/labeled images |
| Generator | `python ../scripts/make_synthetic.py` (defaults: `--indenter sphere --seed 42`) |
| License | Synthetic — the repository's MIT license applies |
| Size (committed) | 581 bytes (`tactile_scenario.csv`) |
| Regenerate with | `python ../scripts/make_synthetic.py` — byte-identical (no RNG state; a fixed integer hash — see the script) |

### Why a scenario file, not committed images (the DECISION)

Every project in this repo that verifies a *sequence* against physics rather than against a single
labeled frame has a choice: commit the rendered frames, or commit the scenario and render at run time.
This project follows **08.01's precedent** (the MPPI controller's `cartpole_scenario.csv`), not
01.02's (the stereo pair's committed PGMs), because the analogy is exact: a tactile sensor's "dataset"
is a **task** — which indenter shape, at what depth, sheared how far, over how many frames — and every
frame of the resulting 320x240x100-frame sequence (7.3 MB uncompressed) is a *deterministic, closed-form
consequence* of that task plus the fixed physical model in [`../src/kernels.cuh`](../src/kernels.cuh)
(Hertzian contact radius, the paraboloid shading profile, the Cattaneo-Mindlin stick/slip law). Committing
rendered frames would (a) bloat the repo for zero benefit — the frames regenerate byte-identically from
581 bytes — and (b) hide the fact that **every ground-truth number this project checks against (contact
footprint, marker displacement, slip-onset frame) is computed analytically from the SAME formulas that
generate the images**, not from a separately-authored label file that could drift out of sync with the
renderer. Keeping the scenario as the only committed artifact makes that "one source of truth" property
impossible to violate by accident.

### Fields / format

`tactile_scenario.csv` — two label,value rows (comment lines start with `#`):

| Field | Type | Meaning |
|-------|------|---------|
| `INDENTER` | `sphere` \| `edge` | Which indenter shape presses into the gel. The committed sample uses `sphere`; the demo's ground-truth gate thresholds in [`../src/main.cu`](../src/main.cu) are calibrated for it only — see the project README "Limitations & honesty" for the `edge` mode's documented, reduced scope. |
| `SEED` | non-negative integer | Seeds the gel's FIXED micro-texture noise (a deterministic hash, not per-frame RNG — see `../src/kernels.cuh` "SHADING MODEL" and `../src/main.cu`'s `hash01`/`texture_noise_gray`). The same seed reproduces the identical byte sequence on any machine. |

Every OTHER number the renderer and the ground-truth gates use — image size (320x240), the marker grid
(18px spacing, 221 markers), the Hertzian sphere radius and max indentation depth, the shading gain, the
Cattaneo-Mindlin stick-residual fraction, and the exact frame count of each phase (6 baseline + 24 press
+ 30 shear + 40 slip = 100 frames) — is **single-sourced** as `constexpr` values in
[`../src/kernels.cuh`](../src/kernels.cuh) (CLAUDE.md §12: "every float* state documents its layout in
one place"; the same rule applies to every algorithmic constant here). This scenario file only selects
the two knobs that are genuinely meant to vary between runs.

| File | SHA-256 |
|------|---------|
| `sample/tactile_scenario.csv` | `a01a2f584300319e82cb879c9a509b851c6ba57303a2fc2ede5b2f01c1314de0` |

**Ground truth**: unlike a project that ships a separate ground-truth file, every ground-truth quantity
this project's demo checks against (Hertzian contact footprint, commanded shear, the Cattaneo-Mindlin
slip-onset frame) is *computed at run time* in `../src/main.cu` from the scenario + `kernels.cuh`
constants — printed in the `CONTACT:`/`SHEAR:`/`SLIP:` lines alongside the measured values, never a
separately-authored label a renderer bug could silently drift away from.

The loader (`load_scenario` in `../src/main.cu`) is strict: a missing file, an unknown `INDENTER` value,
or a missing `SEED`/`INDENTER` row all abort the demo rather than silently falling back to a default.

> `sample/` also carries its own [README](sample/README.md) stating the folder-wide rules.
