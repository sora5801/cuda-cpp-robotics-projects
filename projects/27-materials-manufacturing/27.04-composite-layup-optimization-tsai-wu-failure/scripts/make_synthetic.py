#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample-data generator for 27.04
(Composite layup optimization + Tsai-Wu failure envelope sweeps).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
---------------------------------------------------------------
This project's "data" is not a recording of anything — it is a TASK
DEFINITION: a plausible carbon/epoxy-class unidirectional lamina's elastic
constants and Tsai-Wu strengths (SYNTHETIC teaching parameters, in the
right ballpark for a real aerospace-grade tape but NOT sourced from any
particular datasheet — labeled synthetic everywhere they appear, per
CLAUDE.md paragraph 8), the 4-angle stacking alphabet, the envelope grid's
half-span, and the two load-case sets (MIXED, ALIGNED) the sweep scores
every candidate layup against. All of it is a handful of numbers with a
documented meaning, written out as a small, deterministic, human-readable
CSV. There is no randomness anywhere in this project's physics (every
computation in src/kernels.cuh is a pure function of its inputs), so like
18.01, this generator takes NO --seed: the same command always produces the
same bytes (CLAUDE.md paragraph 8: "seed only if noise is used; prefer
deterministic no-noise").

What the script writes
-----------------------
../data/sample/laminate_scenario.csv — one "LABEL,value[,value...]" row per
required scenario field (src/main.cu's load_scenario() parses this exact
format; see ../data/README.md for the full field-by-field documentation).

The two load-case sets are GENERATED here from a documented formula (not
hand-typed), then written out as concrete (Nx,Ny,Nxy) numbers — main.cu's
loader reads the numbers directly and never re-derives the generating
formula, so the CSV is the single source of truth for exactly which loads
were swept (CLAUDE.md paragraph 12).

  MIXED (16 cases, all magnitude N_REF_NM):
    directions 0..11 : 12 biaxial (Nx,Ny) directions at 30-degree steps
                        around the full circle (Nxy=0) — this is what
                        "mixed/combined loading" means for this project:
                        the load direction is not known in advance, so the
                        layup must resist EVERY direction.
    directions 12-13  : pure +Nxy and -Nxy (pure in-plane shear)
    directions 14-15  : combined Nx+Nxy and Ny+Nxy at 45 degrees between
                        the two components (each normalized back to
                        magnitude N_REF_NM)

  ALIGNED (2 cases, magnitude N_REF_NM):
    pure +Nx (tension) and pure -Nx (compression) — a single, KNOWN load
    direction, the case set that should make 0-degree-heavy stacks win.

Usage
-----
    python make_synthetic.py                     # writes the committed defaults
    python make_synthetic.py --n-ref-nm 150000    # a different reference load magnitude
    python make_synthetic.py --out ../data/sample/laminate_scenario.csv
"""

import argparse
import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Material defaults — a SYNTHETIC teaching carbon/epoxy-class unidirectional
# lamina (order-of-magnitude consistent with real aerospace tape systems
# such as T300/5208-class material, but these exact numbers are invented
# for this project, not copied from any datasheet — labeled synthetic
# everywhere, CLAUDE.md paragraph 8).
# ---------------------------------------------------------------------------
DEFAULTS = {
    "E1_GPA": 135.0,     # fiber-direction modulus, GPa
    "E2_GPA": 10.0,      # transverse modulus, GPa
    "G12_GPA": 5.0,      # in-plane shear modulus, GPa
    "NU12": 0.28,        # major Poisson ratio
    "T_PLY_MM": 0.125,   # single-ply cured thickness, mm (typical prepreg tape)

    "XT_MPA": 1500.0,    # longitudinal tensile strength, MPa
    "XC_MPA": 1200.0,    # longitudinal compressive strength (magnitude), MPa
    "YT_MPA": 50.0,      # transverse tensile strength, MPa
    "YC_MPA": 200.0,     # transverse compressive strength (magnitude), MPa
    "S12_MPA": 70.0,     # in-plane shear strength, MPa

    "N_ENV_MAX_NM": 3.0e6,   # envelope grid half-span, N/m — wide enough that the Tsai-Wu
                             # envelope boundary closes fully inside the grid in every
                             # quadrant, including the strong compression-compression corner
                             # (measured: with a 1e6 half-span the boundary was still open at
                             # the grid edge in that corner — not a bug, just a window too
                             # small for this laminate's compressive strength)
    "ANGLE_ALPHABET_DEG": (0.0, 45.0, -45.0, 90.0),  # the catalog's documented stacking alphabet

    "N_REF_NM": 1.2e5,   # reference load magnitude for every MIXED/ALIGNED case, N/m (120 N/mm) —
                         # tuned so the MIXED-set winner clears its worst-case load with margin
                         # (measured: worst-case factor ~1.46 — see README/THEORY for the
                         # measured ranking) while the ALIGNED-set winner clears it by ~10x,
                         # the numeric signature of "aligned stacks dominate aligned loading"
}


def make_mixed_cases(n_ref_nm: float) -> list[tuple[float, float, float]]:
    """16 combined Nx/Ny/Nxy load vectors, every one magnitude n_ref_nm.

    See the module docstring for the documented grid: 12 biaxial directions
    at 30-degree steps, 2 pure-shear cases, 2 combined Nx+shear/Ny+shear
    cases. Returns a list of (Nx, Ny, Nxy) tuples, N/m.
    """
    cases = []
    for k in range(12):
        phi = math.radians(k * 30.0)
        cases.append((n_ref_nm * math.cos(phi), n_ref_nm * math.sin(phi), 0.0))
    cases.append((0.0, 0.0, n_ref_nm))            # pure +shear
    cases.append((0.0, 0.0, -n_ref_nm))           # pure -shear
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    cases.append((n_ref_nm * inv_sqrt2, 0.0, n_ref_nm * inv_sqrt2))   # Nx + shear, unit magnitude
    cases.append((0.0, n_ref_nm * inv_sqrt2, n_ref_nm * inv_sqrt2))   # Ny + shear, unit magnitude
    assert len(cases) == 16
    return cases


def make_aligned_cases(n_ref_nm: float) -> list[tuple[float, float, float]]:
    """2 pure-Nx load vectors (tension, compression) — the ALIGNED set."""
    return [(n_ref_nm, 0.0, 0.0), (-n_ref_nm, 0.0, 0.0)]


def write_scenario(values: dict, out_path: Path) -> None:
    """Write one scenario CSV. `values` must contain every DEFAULTS key.

    Parameters
    ----------
    values   : dict of LABEL -> numeric value (or, for ANGLE_ALPHABET_DEG,
               a 4-tuple), fixed precision so the file is byte-stable
               across regenerations on any platform.
    out_path : destination CSV. Parent directories are created if missing.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mixed = make_mixed_cases(values["N_REF_NM"])
    aligned = make_aligned_cases(values["N_REF_NM"])
    alpha = values["ANGLE_ALPHABET_DEG"]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        f.write("# SYNTHETIC data — generated by scripts/make_synthetic.py for project 27.04\n")
        f.write("# This is a TASK DEFINITION (lamina material + Tsai-Wu strengths, stacking\n")
        f.write("# alphabet, load-case sets), not a recording — every field is a documented,\n")
        f.write("# labeled-synthetic design choice, in the right ballpark for a real\n")
        f.write("# aerospace-grade carbon/epoxy tape but not sourced from any datasheet.\n")
        f.write(f"# regenerate: python make_synthetic.py --out {out_path.as_posix()}\n")
        f.write("# columns: LABEL,value[,value,...] — see ../README.md and ../data/README.md\n")
        f.write("# for units/meaning of every field. Loads (LOAD_MIXED/LOAD_ALIGNED) are\n")
        f.write(f"# Nx_Npm,Ny_Npm,Nxy_Npm triples, each of magnitude N_REF_NM={values['N_REF_NM']:.1f} N/m,\n")
        f.write("# generated by this script's make_mixed_cases()/make_aligned_cases() (see the\n")
        f.write("# module docstring for the documented grid) and written out as concrete numbers.\n")

        f.write(f"E1_GPA,{values['E1_GPA']:.6f}\n")
        f.write(f"E2_GPA,{values['E2_GPA']:.6f}\n")
        f.write(f"G12_GPA,{values['G12_GPA']:.6f}\n")
        f.write(f"NU12,{values['NU12']:.6f}\n")
        f.write(f"T_PLY_MM,{values['T_PLY_MM']:.6f}\n")
        f.write(f"XT_MPA,{values['XT_MPA']:.6f}\n")
        f.write(f"XC_MPA,{values['XC_MPA']:.6f}\n")
        f.write(f"YT_MPA,{values['YT_MPA']:.6f}\n")
        f.write(f"YC_MPA,{values['YC_MPA']:.6f}\n")
        f.write(f"S12_MPA,{values['S12_MPA']:.6f}\n")
        f.write(f"N_ENV_MAX_NM,{values['N_ENV_MAX_NM']:.6f}\n")
        f.write(f"ANGLE_ALPHABET_DEG,{alpha[0]:.6f},{alpha[1]:.6f},{alpha[2]:.6f},{alpha[3]:.6f}\n")

        f.write(f"N_MIXED_CASES,{len(mixed)}\n")
        for (nx, ny, nxy) in mixed:
            f.write(f"LOAD_MIXED,{nx:.6f},{ny:.6f},{nxy:.6f}\n")
        f.write(f"N_ALIGNED_CASES,{len(aligned)}\n")
        for (nx, ny, nxy) in aligned:
            f.write(f"LOAD_ALIGNED,{nx:.6f},{ny:.6f},{nxy:.6f}\n")

    print(f"[make_synthetic] wrote scenario to {out_path} "
          f"(256 layups x [{len(mixed)} MIXED + {len(aligned)} ALIGNED] cases, labeled SYNTHETIC)")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_out = script_dir.parent / "data" / "sample" / "laminate_scenario.csv"

    parser = argparse.ArgumentParser(
        description="Generate the committed scenario (lamina material, Tsai-Wu strengths, "
                    "stacking alphabet, load-case sets) for project 27.04 "
                    "(Composite layup optimization + Tsai-Wu failure envelope sweeps).")
    parser.add_argument("--e1-gpa", type=float, default=DEFAULTS["E1_GPA"])
    parser.add_argument("--e2-gpa", type=float, default=DEFAULTS["E2_GPA"])
    parser.add_argument("--g12-gpa", type=float, default=DEFAULTS["G12_GPA"])
    parser.add_argument("--nu12", type=float, default=DEFAULTS["NU12"])
    parser.add_argument("--t-ply-mm", type=float, default=DEFAULTS["T_PLY_MM"])
    parser.add_argument("--xt-mpa", type=float, default=DEFAULTS["XT_MPA"])
    parser.add_argument("--xc-mpa", type=float, default=DEFAULTS["XC_MPA"])
    parser.add_argument("--yt-mpa", type=float, default=DEFAULTS["YT_MPA"])
    parser.add_argument("--yc-mpa", type=float, default=DEFAULTS["YC_MPA"])
    parser.add_argument("--s12-mpa", type=float, default=DEFAULTS["S12_MPA"])
    parser.add_argument("--n-env-max-nm", type=float, default=DEFAULTS["N_ENV_MAX_NM"])
    parser.add_argument("--n-ref-nm", type=float, default=DEFAULTS["N_REF_NM"],
                        help="magnitude of every generated MIXED/ALIGNED load case, N/m")
    parser.add_argument("--out", type=Path, default=default_out, help="output CSV path")
    args = parser.parse_args()

    if args.t_ply_mm <= 0.0 or args.n_env_max_nm <= 0.0 or args.n_ref_nm <= 0.0:
        parser.error("--t-ply-mm, --n-env-max-nm, --n-ref-nm must be > 0")

    values = {
        "E1_GPA": args.e1_gpa, "E2_GPA": args.e2_gpa, "G12_GPA": args.g12_gpa, "NU12": args.nu12,
        "T_PLY_MM": args.t_ply_mm,
        "XT_MPA": args.xt_mpa, "XC_MPA": args.xc_mpa, "YT_MPA": args.yt_mpa, "YC_MPA": args.yc_mpa,
        "S12_MPA": args.s12_mpa,
        "N_ENV_MAX_NM": args.n_env_max_nm,
        "ANGLE_ALPHABET_DEG": DEFAULTS["ANGLE_ALPHABET_DEG"],
        "N_REF_NM": args.n_ref_nm,
    }
    write_scenario(values, args.out)


if __name__ == "__main__":
    main()
