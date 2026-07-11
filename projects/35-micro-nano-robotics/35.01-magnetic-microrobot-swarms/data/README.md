# Data — 35.01 Magnetic microrobot swarms: Biot-Savart field computation + swarm dynamics

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

Like 24.01's motor cross-section, this project's sample is a **DESIGN, not a recording**: there is no
public dataset for "a from-scratch 4-coil electromagnet arrangement plus a superparamagnetic microrobot
swarm" — every field below is a fixed engineering constant, chosen and verified during this project's
design (THEORY.md "The problem" derives the physics behind each choice; the demo's own `GATE_*` lines
verify the consequences at run time).

| Property | Value |
|----------|-------|
| Kind | Synthetic (100% — repo default, CLAUDE.md §8). No measurement, no public dataset applies. |
| Generator | `../scripts/make_synthetic.py` (no `--seed`/`--n` flags: every row is a fixed design constant, not sampled) |
| License | Synthetic — repo MIT license applies |
| Size (committed) | `data/sample/microswarm_scenario.csv` — 470 bytes |
| Checksum (SHA-256) | `6b6dc39f332b8dd34228fe8b956e8931ef03b57291041d7b49625a3219fd2dbd` |
| Regenerate with | `python make_synthetic.py` (deterministic — the scenario itself has no randomness; the ONE random draw in the whole demo, the swarm's initial cluster, is seeded inside `../src/main.cu` from the `SWARM` row's seed field below) |

### Fields / format

`data/sample/microswarm_scenario.csv` is a row-labeled CSV (`../src/main.cu`'s `load_scenario` parses
it; `#`-prefixed lines are provenance comments). Every length is **meters**, every current is
**ampere-turns** (current × wire-turn count — PRACTICE.md §2 explains why coil drive is quoted this
way), viscosity is **Pa·s**, angles do not appear (the coil geometry is generated procedurally from
radius + offset, not tabulated). All eight rows are required; `load_scenario` refuses to run on a
missing or unrecognized row rather than guessing (24.01/31.01's discipline for anything
physics-adjacent).

| Row | Fields | Meaning |
|-----|--------|---------|
| `GRID,<grid_n>` | `grid_n` (int) | Field-map resolution per axis — `256` → a 256×256 map, 65536 cells. |
| `COIL,<radius_m>,<offset_m>,<segs_per_coil>` | radius (m), offset (m), segment count (int) | Every one of the 4 coils shares this radius and this distance from the workspace origin to its own center, along its own axis. `offset_m = radius_m/2` is the **Helmholtz condition** — GATE_HELMHOLTZ in the demo depends on this exact ratio. `segs_per_coil=180` is the polygon discretization (720 straight segments total — the catalog bullet's named figure). |
| `WORKSPACE,<half_width_m>` | half-width (m) | The square field map / swarm arena spans `[-half, +half]` on both axes. `0.004` → an 8×8 mm workspace. |
| `FLUID,<viscosity_pa_s>` | dynamic viscosity (Pa·s) | The carrier fluid's viscosity — `0.001` = water at room temperature. |
| `BEAD,<radius_m>,<chi_eff>` | bead radius (m), dimensionless | The microrobot's physical size and its effective volume-susceptibility contrast against the fluid (THEORY.md derives the force law this feeds). |
| `CURRENT,<I0_ampere_turns>` | ampere-turns | The single per-coil drive magnitude every phase of the open-loop schedule uses. |
| `DYNAMICS,<dt_s>,<steps_per_phase>` | seconds, int | The explicit-Euler integration step and how many such steps make up one waypoint-schedule phase. |
| `SWARM,<n_robots>,<init_spread_m>,<seed>` | int, meters, uint32 | Swarm size, the standard deviation of the initial Gaussian cluster around the workspace origin, and the deterministic RNG seed for that one random draw. |

### Why these particular numbers (one line each; full derivations in THEORY.md)

- **Coil radius 20 mm, offset 10 mm:** centimeter-scale, desktop-demo-sized, and `offset = radius/2`
  satisfies the textbook Helmholtz-pair condition exactly.
- **180 segments/coil:** the polygon-vs-circle discretization error at this count is ~0.03% (measured
  by GATE_ONAXIS) — small enough that the analytic gates certify PHYSICS, not discretization artifacts.
- **8×8 mm workspace:** comfortably inside the region where the Helmholtz pair stays flat (measured
  ~0.18% variation) and where single-coil energization still produces a strong, unambiguous gradient.
- **5 µm bead radius, χ_eff = 0.4:** an illustrative superparamagnetic microparticle/cluster scale
  (commercial beads like Dynabeads range roughly 1–15 µm; PRACTICE.md §2 dates and caveats this).
- **500 A-turns, water, dt=0.5 s, 300 steps/phase:** chosen so the resulting drift speed (tens of µm/s)
  covers millimeters within a demo-friendly runtime while keeping the peak field (measured ~14–25 mT at
  these settings) below typical superparamagnetic saturation — see THEORY.md "Numerical considerations".
