#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample-data generator for 24.01
(2D magnetostatic FEA solver on GPU -> motor torque-ripple/cogging parameter sweeps).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
-----------------------------------------------------------------
There is no "recording" to synthesize here in the usual sensor-data sense —
this project's "data" IS a motor design: a cross-section geometry, material
properties, and the parameter-sweep plan. All of it is CONSTANTS, chosen by
the project (not measured, not downloaded), so writing them to a committed
CSV is the synthetic-data story for a design-tool project: the file is the
single, versioned, checksummed source of truth main.cu loads at every run,
exactly like every other flagship's data/sample/*.csv scenario file.

What this script writes
------------------------
data/sample/motor_scenario.csv — one row per logical group of parameters
(grid resolution, domain size, rotor geometry, stator geometry, pole/slot
counts, material properties, the slot-opening fraction, the solver's SOR
factor and sweep-pair budget, and the cogging sweep's arc-fraction list and
rotor-angle sample count). main.cu's load_scenario() is the authoritative
parser (see its header comment for the exact row grammar); THIS script must
stay byte-for-byte in sync with that grammar, or the demo will refuse to
parse the file it just generated (an intentional early-failure to keep the
two from silently drifting apart — CLAUDE.md's "never quietly solve the
wrong problem" principle).

The chosen values are NOT arbitrary: every one of them was validated in a
standalone Python/NumPy prototype (SOR convergence measured, the annulus
analytic gate matched Ampere's law to ~0.2%, the flux-continuity gate
matched to ~1e-5, and the arc-fraction sweep showed a genuine, measured,
non-monotonic minimum in peak cogging torque) before being written here —
see THEORY.md "How we verify correctness" and "Numerical considerations".

Usage
-----
    python make_synthetic.py                         # writes the committed defaults
    python make_synthetic.py --out custom_path.csv    # write elsewhere (for experiments)
"""

import argparse
import hashlib
from pathlib import Path

# ---------------------------------------------------------------------------
# The scenario's numeric content — kept as named constants (not buried in
# f-strings) so a reader can see every design decision in one place, and so
# README.md / data/README.md can quote the SAME numbers without retyping
# them from a rendered CSV.
# ---------------------------------------------------------------------------
GRID_NX, GRID_NY = 256, 256          # matches kGridN in src/kernels.cuh exactly (checked at load time)
DOMAIN_HALF_W_M = 0.026              # domain spans [-26mm, +26mm]^2; just outside the stator OD

ROTOR_CORE_R_M = 0.010               # solid rotor iron core outer radius
MAGNET_THK_M = 0.003                 # magnet radial thickness -> magnet outer radius = 0.013 m
AIR_GAP_M = 0.001                    # mechanical air gap -> stator bore radius = 0.014 m

STATOR_BACK_IRON_IN_R_M = 0.019      # tooth-ring / back-iron boundary radius
STATOR_OUTER_R_M = 0.022             # stator outer radius (back iron closes the flux path here)

POLES = 4                            # rotor magnet pole count (even, alternating polarity)
SLOTS = 6                            # stator slot count

MU_R_IRON = 2000.0                   # linear relative permeability, rotor/stator iron (no saturation model)
MU_R_MAGNET = 1.05                   # linear relative permeability, NdFeB-class permanent magnets
BR_TESLA = 1.2                       # magnet remanence (typical N42-grade NdFeB, illustrative)

SLOT_OPEN_FRAC = 0.35                # fraction of one slot pitch left open (air) at the bore

SOR_OMEGA = 1.97                     # SOR relaxation factor (measured convergence — THEORY.md)
N_SWEEPS = 1500                      # fixed red+black sweep-PAIR budget per solve (measured: residual
                                      # ratio ~1e-6..1e-11 by this point on every fixture tested)

SWEEP_ARC_FRACS = [0.60, 0.70, 0.80, 0.90, 1.00]   # the design sweep's 5 magnet pole-arc fractions
SWEEP_N_ANGLES = 24                                 # rotor-angle samples per pole pitch (MUST be even)


def write_scenario(out_path: Path) -> str:
    """Write the motor_scenario.csv described above; return its SHA-256 hex digest.

    Parameters
    ----------
    out_path : destination CSV path. Parent directories are created if missing.

    The row grammar (label, then comma-separated numeric fields) is fixed
    and documented in ../src/main.cu's load_scenario(): GRID, DOMAIN, ROTOR,
    STATOR, POLES_SLOTS, MATERIALS, SLOT_OPEN, SOLVER, SWEEP_ARCS,
    SWEEP_ANGLES — every one required, order-free, unknown rows rejected.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# SYNTHETIC scenario for project 24.01 — a motor cross-section DESIGN, not a recording.",
        "# regenerate: python make_synthetic.py",
        "# All lengths are METERS. Row grammar is documented in ../src/main.cu (load_scenario).",
        f"GRID,{GRID_NX},{GRID_NY}",
        f"DOMAIN,{DOMAIN_HALF_W_M}",
        f"ROTOR,{ROTOR_CORE_R_M},{MAGNET_THK_M},{AIR_GAP_M}",
        f"STATOR,{STATOR_BACK_IRON_IN_R_M},{STATOR_OUTER_R_M}",
        f"POLES_SLOTS,{POLES},{SLOTS}",
        f"MATERIALS,{MU_R_IRON},{MU_R_MAGNET},{BR_TESLA}",
        f"SLOT_OPEN,{SLOT_OPEN_FRAC}",
        f"SOLVER,{SOR_OMEGA},{N_SWEEPS}",
        "SWEEP_ARCS," + ",".join(str(a) for a in SWEEP_ARC_FRACS),
        f"SWEEP_ANGLES,{SWEEP_N_ANGLES}",
        "",   # trailing newline
    ]
    text = "\n".join(lines)
    # newline="" + explicit "\n" join (not csv.writer) keeps the file
    # byte-identical across platforms — no csv-module dialect surprises,
    # and easy for a learner to read/regenerate by hand if needed.
    out_path.write_text(text, encoding="utf-8", newline="\n")

    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(f"[make_synthetic] wrote {out_path} ({out_path.stat().st_size} bytes, labeled SYNTHETIC)")
    print(f"[make_synthetic] sha256: {digest}")
    return digest


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_out = script_dir.parent / "data" / "sample" / "motor_scenario.csv"

    parser = argparse.ArgumentParser(
        description="Generate the committed motor-design scenario for project 24.01.")
    parser.add_argument("--out", type=Path, default=default_out,
                        help="output CSV path (default: ../data/sample/motor_scenario.csv)")
    args = parser.parse_args()

    write_scenario(args.out)


if __name__ == "__main__":
    main()
