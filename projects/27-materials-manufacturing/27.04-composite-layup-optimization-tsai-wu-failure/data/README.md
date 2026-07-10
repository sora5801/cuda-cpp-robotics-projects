# Data — 27.04 Composite layup optimization + Tsai-Wu failure envelope sweeps

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

There is no recorded dataset here at all — a laminate's material properties, strengths, stacking
alphabet, and the load cases it is scored against are **design parameters**, not measurements of
anything. `scripts/make_synthetic.py` writes all of them out as one small CSV.

| Property | Value |
|----------|-------|
| Kind | Synthetic (default, and only, source — CLAUDE.md §8) |
| Generator / source | `../scripts/make_synthetic.py` (no `--seed`: every computation in this project is a pure function of its inputs, so there is no randomness to seed — CLAUDE.md §8's "prefer deterministic no-noise") |
| License | Synthetic — repo MIT license applies |
| Size (committed) | `laminate_scenario.csv`, ~2.1 KiB |
| Checksum (SHA-256) | `2b545c6399040eeadff9721c6d519b28bab775441dcbea923c21ce027c5da1bf` |
| Regenerate with | `python make_synthetic.py` (writes the exact committed file byte-for-byte) |

### Fields / format

`data/sample/laminate_scenario.csv` — one `LABEL,value[,value,...]` row per field, parsed by the
strict loader in `../src/main.cu`'s `load_scenario()`. Every field is required.

| Label | Meaning | Units |
|-------|---------|-------|
| `E1_GPA` | Lamina fiber-direction Young's modulus | GPa |
| `E2_GPA` | Lamina transverse Young's modulus | GPa |
| `G12_GPA` | Lamina in-plane shear modulus | GPa |
| `NU12` | Lamina major Poisson ratio | unitless |
| `T_PLY_MM` | Single-ply cured thickness | mm |
| `XT_MPA` / `XC_MPA` | Longitudinal tensile / compressive strength (magnitude) | MPa |
| `YT_MPA` / `YC_MPA` | Transverse tensile / compressive strength (magnitude) | MPa |
| `S12_MPA` | In-plane shear strength | MPa |
| `N_ENV_MAX_NM` | Failure-envelope grid half-span (Nx and Ny each span `[-N_ENV_MAX_NM, +N_ENV_MAX_NM]`) | N/m |
| `ANGLE_ALPHABET_DEG` | The 4 candidate ply angles `{0, 45, -45, 90}` — the catalog's documented alphabet | degrees |
| `N_MIXED_CASES` | Row count of the `LOAD_MIXED` rows that follow (16 in the committed sample) | count |
| `LOAD_MIXED` (repeated) | One combined `Nx,Ny,Nxy` load case — see the generator's docstring for the 16-direction grid (12 biaxial directions at 30-deg steps + 2 pure-shear + 2 combined) | N/m each |
| `N_ALIGNED_CASES` | Row count of the `LOAD_ALIGNED` rows that follow (2 in the committed sample) | count |
| `LOAD_ALIGNED` (repeated) | One pure-Nx load case (tension, then compression) | N/m each |

**Material honesty:** `E1/E2/G12/nu12` and the five Tsai-Wu strengths are a **SYNTHETIC teaching
lamina** — order-of-magnitude consistent with a real aerospace-grade unidirectional carbon/epoxy
tape (comparable to published T300/5208-class values), but these exact numbers are invented for
this project and are **not** sourced from any manufacturer datasheet. `src/kernels.cuh`'s `Lamina`
struct documents every field's units; `THEORY.md` derives the physics each one feeds.

The two load-case SETS (`MIXED`, `ALIGNED`) are this project's actual "experiment": the ranking
sweep scores every one of the 256 candidate stacking sequences against both sets, and the classic
result (quasi-isotropic-like stacks win MIXED, 0-heavy stacks win ALIGNED) is the demo's headline
finding — see `README.md` and `THEORY.md` §the-problem.
