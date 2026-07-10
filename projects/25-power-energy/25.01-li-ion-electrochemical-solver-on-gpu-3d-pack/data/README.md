# Data — 25.01 Li-ion electrochemical (SPM) solver on GPU + 3D pack thermal simulation + cooling-design sweeps

Provenance, licensing, and field documentation for everything under `data/` (CLAUDE.md §8).

## The rules (repo-wide, CLAUDE.md §8)

- **Synthetic-first.** Robotics data can almost always be synthesized with full ground truth;
  `../scripts/make_synthetic.py` is the default source, and synthetic data is **labeled synthetic
  everywhere it appears**.
- **Tiny committed sample.** `sample/` holds just enough committed data that the demo runs offline
  with zero downloads.
- **Public datasets** are fetched by `../scripts/download_data.ps1` / `.sh` where one genuinely teaches
  more. **Respect every license.**
- **Never fabricate.** No made-up measurements passed off as real.

## This project's data

There is no real, redistributable "24-cell robot battery pack" dataset — and even if one existed,
licensing real cell electrochemistry/pack-thermal data is exactly the risk CLAUDE.md §8 asks projects
to avoid. Every number below is a **hand-chosen synthetic teaching parameter**: shaped to be physically
plausible (real orders of magnitude for solid-state diffusivities, activation energies, thermal
conductivities, cooling coefficients) but **never claimed to match any real cell, pack, or published
dataset**. THEORY.md "The math" derives the resulting capacity/SOC-swing/thermal-rise arithmetic in
full so the specific numbers below are traceable, not arbitrary.

| Property | Value |
|----------|-------|
| Kind | Synthetic (no public dataset exists or applies here) |
| Generator | `../scripts/make_synthetic.py` — **no RNG**: every value is a literal constant, so the file is byte-identical on every machine, every run |
| License | Synthetic — the repository's MIT license applies to the generator and the generated file |
| File | `sample/pack_scenario.csv` (~1.6 KB, 39 lines) |
| SHA-256 | `59adccb8535ea97426aa5a32b4fe4a5e7b5ab7a8b5f22019c9e6c604c07d2711` |
| Regenerate with | `python scripts/make_synthetic.py` (writes `data/sample/pack_scenario.csv`) |

## Fields / format

Row-labeled CSV, comments (`#`) and blank lines ignored, loaded by `src/main.cu`'s `load_scenario()`
(strict: an unrecognized label or an out-of-range value aborts rather than silently running the wrong
pack). All lengths are meters, angles are not used, temperatures are always **kelvin** (never Celsius)
in the file and in every downstream computation (CLAUDE.md §12).

| Row | Fields | Units | Meaning |
|-----|--------|-------|---------|
| `THERMAL` | `rho_cp,kx,ky,kz,T_init,T_coolant` | J/(m³K), W/(mK)×3, K, K | Pack medium's volumetric heat capacity; **anisotropic** conductivity (`kx`,`ky` in-plane, `kz` through-plane — deliberately lower, THEORY.md "The problem"); initial and coolant temperature. |
| `CELL_DIMS` | `Lx,Ly,Lz` | m | One cell's footprint. Tiles exactly into the 4×3×2 pack → 32×24×16 thermal grid (8 voxels/cell/axis, compile-time checked in `kernels.cuh`). |
| `ANODE` / `CATHODE` | `R_p,D25,Ea_D,c_max,c0_frac,i0_ref,Ea_k,A_surf` | m, m²/s, J/mol, mol/m³, –, A/m², J/mol, m² | Representative spherical particle: radius; diffusivity + activation energy at the Arrhenius reference temperature (298.15 K); max lithium concentration; **initial** stoichiometry fraction; exchange-current-density prefactor + its activation energy (Butler-Volmer, `kernels.cuh` `ElectrodeGeom`); total active particle surface area **per cell** (sets the current→flux mapping). Anode is graphite-like, cathode is NMC-like — see THEORY.md for exactly what that means and does not mean here. |
| `ELEC` | `R_ohm` | Ω | Per-cell lumped internal ohmic resistance (current-collector + electrolyte + contact resistance, all folded into one number — the SPM's standard simplification). |
| `MISSION` | `dt_thermal,n_sub,duration_s` | s, –, s | Thermal timestep; electrochemistry substeps per thermal step (`dt_e = dt_thermal/n_sub`); total mission length. |
| `SEG` (×4, in order) | `kind,duration_s,current_A` | –, s, A | One repeating AMR duty-cycle segment: `kind` is a human label only (not parsed for logic); `current_A` is the **cell** current, positive = discharge. The four rows are one 120 s cycle (accelerate/cruise/idle/regen-charge), repeated to fill `MISSION`'s `duration_s`. |
| `SWEEP_H` | 6 values | W/(m²K) | The swept convective coefficients — crossed with `{bottom, side}` in `main.cu` for the 12-design sweep (`kernels.cuh` `kNDesigns`). |

OCV curves are **not** stored in this file — they are two closed-form synthetic polynomials
(`ocv_anode`, `ocv_cathode`) hard-coded in `src/main.cu`, documented there and in THEORY.md, because
they are *functions* rather than scenario data.

### Where the specific numbers came from

Every constant is chosen, not measured — but not arbitrary either. THEORY.md "The math" walks through
the full arithmetic; in brief:

- Diffusivities (`3.0e-14`, `1.0e-14` m²/s) and activation energies (25–35 kJ/mol) sit at realistic
  orders of magnitude for Li-ion solid-state diffusion and reaction kinetics.
- `A_surf` values (2.5, 3.0 m²) were chosen so a representative cell current (tens of amps) maps to a
  molar flux in the realistic 1e-5–1e-4 mol/(m²s) range.
- The mission's four currents were chosen so 10 duty cycles (1200 s) swing the pack's SOC a realistic
  **partial** ~15–40% (an AMR on shift does not deep-cycle every 20 minutes) while still generating
  enough ohmic + activation heat (`R_ohm=0.10 Ω`, chosen for visibility, not measured from any real
  cell) to make the cooling-design comparison show a real, non-trivial effect.
- `SWEEP_H` spans natural convection (~10 W/(m²K)) to an aggressive liquid cold plate (~500 W/(m²K)).

### Placeholder status

None — this sample is real (i.e., final) scenario data for this project's actual algorithm, not a
toolchain-validation placeholder.
