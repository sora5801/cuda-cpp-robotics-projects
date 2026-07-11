#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample-data generator for 35.01
(Magnetic microrobot swarms: Biot-Savart field computation + swarm dynamics).

Why this script exists (CLAUDE.md §8: synthetic-first)
--------------------------------------------------------
There is no public dataset for "a from-scratch 4-coil electromagnet
arrangement and a superparamagnetic microrobot swarm" — this project's
sample is, like 24.01's motor cross-section, a DESIGN, not a recording:
coil geometry, fluid/bead material constants, and an open-loop current
schedule, all fixed engineering numbers rather than measurements. Every
field is written out with its unit and provenance in
../data/README.md; this script's only job is to emit that same content as
a deterministic, tiny, offline-runnable CSV that ../src/main.cu loads at
startup (`load_scenario`).

Determinism: there is no randomness in the SCENARIO itself (every row is a
fixed design constant) — running this script twice produces byte-identical
output. The ONE random draw in the whole demo (the swarm's initial cluster
positions) is seeded INSIDE main.cu (Xorshift32, seed taken from the
SWARM row below), not here.

Usage
-----
    python make_synthetic.py                          # writes the default scenario
    python make_synthetic.py --out <path>              # write elsewhere (testing only)

The parameters below were chosen and verified during this project's design
(see THEORY.md "The problem" and "How we verify correctness"):
  * coil radius/offset satisfy the Helmholtz condition (offset = radius/2,
    separation between opposing coils = radius) so GATE_HELMHOLTZ in the
    demo has a genuine textbook configuration to check;
  * the current (500 A-turns), bead radius (5 um), and fluid viscosity
    (water) combination keeps the field below typical superparamagnetic
    saturation (~tens of mT) while producing a swarm drift speed
    (tens of um/s) that covers millimeters within a few hundred simulated
    seconds — fast enough for a demo, slow enough to stay honestly in the
    low-Reynolds-number, quasi-static regime THEORY.md derives.
"""

import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# The scenario itself — see kernels.cuh's SwarmScenario struct for the unit
# of every field, and data/README.md for the full provenance/derivation.
# ---------------------------------------------------------------------------
GRID_N = 256                 # field-map resolution per axis
COIL_RADIUS_M = 0.020        # R: 20 mm coils
COIL_OFFSET_M = 0.010        # = R/2, the Helmholtz separation condition
SEGS_PER_COIL = 180          # straight segments approximating each coil's circle (720 total)
WORKSPACE_HALF_M = 0.004     # 8x8 mm workspace (half-width 4 mm)
MU_FLUID_PA_S = 1.0e-3       # water at room temperature
BEAD_RADIUS_M = 5.0e-6       # 5 um radius (10 um diameter) superparamagnetic bead/cluster
CHI_EFF = 0.4                # illustrative effective volume-susceptibility contrast
I0_AMPERE_TURNS = 500.0      # per-coil drive magnitude used by the waypoint schedule
DT_S = 0.5                   # explicit-Euler integration step
STEPS_PER_PHASE = 300        # Euler steps per waypoint-schedule phase (3 phases -> 900 steps total)
N_ROBOTS = 1000               # swarm size (the catalog bullet's "N=1000 magnetic microrobots")
INIT_SPREAD_M = 3.0e-4       # 0.3 mm std-dev of the initial Gaussian cluster
SEED = 1234                  # host RNG seed for the initial cluster (determinism, CLAUDE.md §12)


def make_scenario_csv(out_path: Path) -> None:
    """Write the fixed scenario as a row-labeled CSV. Every row is a design
    constant, not a measurement — see the module docstring and
    ../data/README.md for what each one means and why it was chosen.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        f.write("# SYNTHETIC scenario for project 35.01 — coil geometry + fluid/bead parameters\n")
        f.write("# + the open-loop current schedule's drive magnitude, not a recording.\n")
        f.write("# regenerate: python make_synthetic.py\n")
        f.write("# All lengths are METERS, currents are AMPERE-TURNS, viscosity is Pa*s.\n")
        f.write("# Row grammar documented in ../src/main.cu (load_scenario) and ../data/README.md.\n")
        f.write(f"GRID,{GRID_N}\n")
        f.write(f"COIL,{COIL_RADIUS_M},{COIL_OFFSET_M},{SEGS_PER_COIL}\n")
        f.write(f"WORKSPACE,{WORKSPACE_HALF_M}\n")
        f.write(f"FLUID,{MU_FLUID_PA_S}\n")
        f.write(f"BEAD,{BEAD_RADIUS_M},{CHI_EFF}\n")
        f.write(f"CURRENT,{I0_AMPERE_TURNS}\n")
        f.write(f"DYNAMICS,{DT_S},{STEPS_PER_PHASE}\n")
        f.write(f"SWARM,{N_ROBOTS},{INIT_SPREAD_M},{SEED}\n")

    print(f"[make_synthetic] wrote scenario to {out_path} (labeled SYNTHETIC, deterministic)")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_out = script_dir.parent / "data" / "sample" / "microswarm_scenario.csv"

    parser = argparse.ArgumentParser(
        description="Generate the tiny synthetic scenario for project 35.01 "
                    "(Magnetic microrobot swarms: Biot-Savart field computation + swarm dynamics).")
    parser.add_argument("--out", type=Path, default=default_out,
                        help="output CSV path (default: ../data/sample/microswarm_scenario.csv)")
    args = parser.parse_args()

    make_scenario_csv(args.out)


if __name__ == "__main__":
    main()
