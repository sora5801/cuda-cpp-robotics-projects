#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample-data generator for project 28.01
(Real-time FEM soft-arm model + model-based control (GPU SOFA-style)).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
---------------------------------------------------------------
This project's "data" is not recordings — it is the SCENARIO DEFINITION of a
synthetic soft arm: material constants, mesh geometry, tendon-actuation
parameters, controller tuning, and the closed-loop setpoint sequence. All of
it is synthetic by construction (there is no physical arm; the material is an
"elastomer-class" teaching material, NOT a datasheet silicone), so this
generator writes CONSTANTS, deterministically, with zero RNG — the same bytes
on every machine, every run. Provenance and field docs live in
../data/README.md; SHA-256 checksums of the committed file are recorded there.

The contract with the C++ side
------------------------------
../src/main.cu loads ../data/sample/arm_scenario.csv at startup and splits its
rows into two classes:

  * MODEL rows (mesh + material) — these must MATCH the compiled constants in
    ../src/kernels.cuh exactly, because the CFL timestep and the analytic-gate
    formulas are derived from them at compile time (kernels.cuh documents the
    derivations). main.cu CROSS-CHECKS every model row against the compiled
    value and refuses to run on a mismatch — the file cannot silently lie
    about what was simulated. To change the model, change kernels.cuh AND
    regenerate this file (edit the constants below to match).

  * SCENARIO rows (actuation, controller tuning, setpoint sequence) — these
    are genuinely runtime inputs: main.cu reads and uses them directly, so a
    learner can retune the controller or reshape the setpoint sequence by
    editing the CSV (or this generator) without touching C++ (README
    "Exercises" builds on exactly that).

Keep the constants below in lockstep with ../src/kernels.cuh (the C++ names
are given beside each value). If you change anything here, rerun this script
AND update the SHA-256 + expected output where they are documented.

Usage
-----
    python make_synthetic.py                          # writes ../data/sample/arm_scenario.csv
    python make_synthetic.py --out other/path.csv     # anywhere else (e.g. for experiments)
"""

import argparse
import hashlib
from pathlib import Path

# ---------------------------------------------------------------------------
# MODEL rows — must mirror ../src/kernels.cuh (constant names in comments).
# main.cu verifies each against the compiled constant (relative tolerance
# 1e-6 after float parsing) and aborts on mismatch.
# ---------------------------------------------------------------------------
MODEL = [
    # (key, value, meaning) — value strings parse exactly to the kernels.cuh constants.
    ("NELX",           "120",     "elements along the arm length (x) [kNelx]"),
    ("NELY",           "12",      "elements through the arm height (y) [kNely]"),
    ("ELEM_SIZE_M",    "0.002",   "square element side h (m) [kElemSize_m]"),
    ("YOUNGS_E_PA",    "1000000", "Young's modulus (Pa), synthetic elastomer-class [kYoungsE_Pa]"),
    ("POISSON_NU",     "0.4",     "Poisson's ratio (0.40 not 0.49 - Q4 locking, THEORY.md) [kPoissonNu]"),
    ("THICKNESS_M",    "0.02",    "out-of-plane depth (m) [kThickness_m]"),
    ("DENSITY_KGM3",   "1100",    "mass density (kg/m^3), synthetic silicone-like [kDensity_kgm3]"),
    ("DT_S",           "3e-05",   "integration timestep (s), CFL-derived in kernels.cuh [kDt_s]"),
    ("RAYLEIGH_ALPHA", "3.8",     "mass-proportional damping (1/s), zeta_1 ~ 0.149 [kRayleighAlphaOn]"),
    ("RAYLEIGH_BETA",  "2e-05",   "stiffness-proportional damping (s), high-mode damper [kRayleighBetaOn]"),
    ("TENDON_BIAS_N",  "0.25",    "co-contraction pretension per tendon (N); 2*bias ~ 51% of the buckling load [kTendonBiasN]"),
]

# ---------------------------------------------------------------------------
# SCENARIO rows — runtime inputs main.cu actually consumes. The tuning story
# for each value is told in ../src/main.cu beside its default constant and in
# THEORY.md "The model-based-control story".
# ---------------------------------------------------------------------------
SCENARIO = [
    ("PROBE_DELTA_T_N",    "0.18",  "tension differential (N) used to identify the tip Jacobian"),
    ("CONTROL_SUBSTEPS",   "100",   "dynamics steps per control tick (100 x 3e-5 s = 3 ms -> ~333 Hz)"),
    ("HOLD_STEPS",         "66000", "dynamics steps held per setpoint (~2.0 s ~ 4 first-mode periods)"),
    ("PI_MARGIN_ALPHA",    "0.3",   "Kp = alpha/|J|; resonant loop gain ~ alpha/(2*zeta) ~ 0.7 < 1 (no sustained ring)"),
    ("PI_INTEGRAL_TIME_S", "0.15",  "Ki = Kp/Ti (s); crossover ~ alpha/Ti = 2 rad/s, well below the resonance"),
    ("DELTA_T_CLAMP_FRAC", "1.8",   "|deltaT| <= frac * TENDON_BIAS_N (keeps both tendons taut: bias-clamp/2 > 0)"),
    ("SETPOINT_SAFE_FRAC", "0.65",  "setpoint scale = J * frac * clamp (stays inside the identified range)"),
    # The step+hold setpoint sequence, as fractions of the safe scale above:
    # up, down past center, half up, back to center — exercises both tendons
    # and a return-to-zero (integrator unwinding) in one run.
    ("SETPOINT_FRACS",     "0.6;-0.6;0.3;0", "setpoint sequence (fractions of the safe scale, ';'-separated)"),
]


def write_scenario(out_path: Path) -> None:
    """Write the scenario CSV: '#'-comment header (label + provenance +
    regeneration command), then 'key,value' rows. No RNG anywhere — the file
    is constants, byte-identical on every platform (LF line endings written
    explicitly so the SHA-256 in ../data/README.md is stable across OSes)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# SYNTHETIC data - generated by scripts/make_synthetic.py for project 28.01")
    lines.append("# Scenario definition for the corotational-FEM soft arm: model constants")
    lines.append("# (cross-checked against src/kernels.cuh at startup) + runtime controller")
    lines.append("# scenario. No RNG; the material is a synthetic elastomer-class teaching")
    lines.append("# material, not a datasheet substance. regenerate: python make_synthetic.py")
    lines.append("# format: key,value   (comments and blank lines ignored by the loader)")
    lines.append("")
    lines.append("# --- MODEL (must match src/kernels.cuh; loader aborts on mismatch) ---")
    for key, value, meaning in MODEL:
        lines.append(f"# {key}: {meaning}")
        lines.append(f"{key},{value}")
    lines.append("")
    lines.append("# --- SCENARIO (runtime inputs: actuation, controller tuning, setpoints) ---")
    for key, value, meaning in SCENARIO:
        lines.append(f"# {key}: {meaning}")
        lines.append(f"{key},{value}")

    data = ("\n".join(lines) + "\n").encode("utf-8")
    out_path.write_bytes(data)   # write_bytes: no platform newline translation

    sha = hashlib.sha256(data).hexdigest()
    print(f"[make_synthetic] wrote {out_path} ({len(data)} bytes, labeled SYNTHETIC)")
    print(f"[make_synthetic] SHA-256: {sha}")
    print("[make_synthetic] record this hash in ../data/README.md if the file changed")


def main() -> None:
    """Parse arguments and run the generator (kept separate so the writer is
    importable without argparse in the way — template convention)."""
    script_dir = Path(__file__).resolve().parent
    default_out = script_dir.parent / "data" / "sample" / "arm_scenario.csv"

    parser = argparse.ArgumentParser(
        description="Generate the synthetic scenario sample for project 28.01 "
                    "(soft-arm FEM + model-based control). Deterministic constants; no RNG.")
    parser.add_argument("--out", type=Path, default=default_out,
                        help="output CSV path (default: ../data/sample/arm_scenario.csv)")
    args = parser.parse_args()

    write_scenario(args.out)


if __name__ == "__main__":
    main()
