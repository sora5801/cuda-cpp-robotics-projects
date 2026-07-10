# Data — 26.01 Topology optimization (SIMP) on GPU for lightweight links and brackets

Provenance, licensing, and field documentation for everything under `data/` (CLAUDE.md §8).

## The rules (repo-wide, CLAUDE.md §8)

- **Synthetic-first.** Robotics data can almost always be synthesized with full ground truth;
  `../scripts/make_synthetic.py` is the default source, and synthetic data is **labeled synthetic
  everywhere it appears**.
- **Tiny committed sample.** `sample/` holds just enough committed data that the demo runs offline
  with zero downloads. Large/raw/downloaded data lives in `downloaded/` (git-ignored).
- **Never fabricate.** No made-up measurements, no invented ground truth passed off as real.

## This project's data — a PROBLEM DEFINITION, not a recording

This project's "data" is unusual for the repo: there is no sensor recording to synthesize. What
`data/sample/` holds is a **problem definition** — a mesh size, a material, a target volume
fraction, and a set of boundary conditions (which nodes are clamped, where the load is applied,
which elements are forced void). Every number in both files is a **deliberately chosen constant**,
not a random draw — `scripts/make_synthetic.py` writes them out exactly, no RNG involved, so
"regenerating" the sample means re-running the same script and getting byte-identical output.

| Property | Value |
|----------|-------|
| Kind | Synthetic (a problem definition, not measurements) |
| Generator | `../scripts/make_synthetic.py` (no RNG — deterministic constants) |
| License | Synthetic — repo MIT license applies |
| Size (committed) | `mbb_scenario.csv` 552 bytes; `bracket_scenario.csv` 557 bytes |
| SHA-256 (as generated 2026-07-10, default parameters) | `mbb_scenario.csv`: `89f13e7f31f068394b6ba56a8c6e31b128c9ac1dcd5813f06fcf97f478fb0037`<br>`bracket_scenario.csv`: `583eb6c101f0ed9cead4a2e1c98b28647ac5bde70ac9e08aa9cad20b2cd64e60` |
| Regenerate with | `python ../scripts/make_synthetic.py` (both files; `--mbb-only` / `--bracket-only` for one) |

### The two scenarios

- **`mbb_scenario.csv`** — the classic half-MBB beam (symmetry-reduced), the textbook validation
  case for topology optimization: a 120x40-element domain, a symmetry-plane roller support along
  the left edge, a point roller support at the bottom-right corner, and a downward point load at
  the top-left corner (the beam's midspan, after mirroring). Its expected result is famous enough
  to recognize on sight — a small number of diagonal struts connecting load to supports.
- **`bracket_scenario.csv`** — a robot L-bracket: an 80x80-element square domain with its top-right
  quadrant removed (a `PASSIVE_RECT` void region — the "L" notch), bolted to a robot's frame along
  the remaining top edge, and loaded at the bottom-right foot by a motor-flange point load. This is
  this flagship's design-for-robotics case (README "System context").

Both scenarios share the same material: **Aluminum 6061-T6** (`E0_PA = 6.89e10`, illustrative,
dated 2026-07-10 — verify current before relying on it for anything beyond this teaching demo),
`EMIN_RATIO = 1e-3` (the SIMP void-stiffness floor — THEORY.md "Numerical considerations" explains
why this project uses 1e-3 rather than the more common 1e-9), `VOLFRAC = 0.4` (40% of the *active*
design domain — passive/void elements are excluded from both the numerator and denominator of that
fraction), and `MAXOUTER = 80` outer SIMP iterations.

### Fields / format

Both files are read by `../src/main.cu`'s `load_topo_scenario()`. Rows (order-independent; `#`
lines are comments):

| Row | Fields | Meaning |
|---|---|---|
| `NELX,<int>` / `NELY,<int>` | element counts | mesh size along x (fast axis) / y (slow axis) |
| `E0_PA,<float>` | Pa | material Young's modulus |
| `EMIN_RATIO,<float>` | dimensionless | SIMP floor: `Emin = ratio * E0` |
| `VOLFRAC,<float>` | dimensionless in (0,1] | target volume fraction of ACTIVE elements |
| `MAXOUTER,<int>` | count | outer SIMP iteration cap |
| `FIX_RECT,i0,j0,i1,j1,xflag,yflag` (repeatable) | node indices + 0/1 flags | fixes `ux`/`uy` to 0 for every node in `[i0,i1]x[j0,j1]` |
| `LOAD,i,j,fx_N,fy_N` (repeatable) | node index + Newtons | a point load applied at node `(i,j)` |
| `PASSIVE_RECT,ex0,ey0,ex1,ey1` (repeatable, optional) | element indices | elements forced permanently void (`rho=0`), excluded from the design |

**Frame/units convention** (CLAUDE.md §12): node `(0,0)` is the domain's TOP-LEFT corner; `j`
increases DOWNWARD (matching the PGM artifact's row-major, top-row-first image layout — `../src/`
kernels.cuh's "MESH & DOF LAYOUT" comment is the canonical statement). Element size is
**nondimensional** ("1 element unit"): kernels.cuh proves that for a uniform SQUARE Q4 mesh the
element stiffness is exactly `E(rho) * KE_hat` regardless of the physical element size `h` — so no
physical length unit is needed by the solver itself. README/PRACTICE.md interpret the grid as
roughly 1 mm per element for BOM/labeling purposes (a ~120mm x 40mm MBB coupon, an ~80mm x 80mm
bracket — plausible small-robot-bracket scales), a **label, not a solver input**.
