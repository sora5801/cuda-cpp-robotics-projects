#!/usr/bin/env python3
"""make_synthetic.py — synthetic phantom generator for 29.05 (Ultrasound: GPU beamforming).

EDUCATIONAL / SYNTHETIC ONLY. This project teaches ultrasound SIGNAL PROCESSING, not medicine:
every scatterer below is a simulated point in a numerical phantom (never patient data), and
nothing this project produces is a diagnostic or therapeutic claim (CLAUDE.md paragraph 1, 8).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
-----------------------------------------------------------------
A real ultrasound "phantom" is a physical block of tissue-mimicking gel with known wire targets
and inclusions, used to calibrate and QA real machines. We synthesize the DIGITAL equivalent: a
list of point scatterers with known positions and reflectivity, generated deterministically from
a fixed seed so the committed sample is tiny, license-clean, and reproducible bit-for-bit. The
C++ program (src/main.cu) reads this phantom and SIMULATES the channel data (the raw per-element
RF traces) itself, from the documented pulse-echo physics — this script only places the
scatterers, it does not synthesize any acoustic signal.

The phantom has three parts (see ../THEORY.md "The problem" for why each teaches something):
  * 9 WIRE TARGETS in a "+" cross pattern (a horizontal row + a vertical column sharing a center
    point at x=0, z=20 mm) — the classic QA-phantom pattern used to check that both LATERAL and
    AXIAL localization/resolution are correct, read straight off a recognizable shape.
  * ~700 scatterers packed at higher density and amplitude inside a small disk — a high-
    scattering INCLUSION region (never called a "lesion": this is signal processing, not
    diagnosis) that exercises the CONTRAST verification gate.
  * ~9000 background scatterers of IDENTICAL amplitude at uniformly random positions — this is
    not decorative noise: identical-amplitude, randomly-positioned coherent reflectors are
    EXACTLY the classical "random phasor sum" setup that produces Rayleigh-distributed SPECKLE
    (THEORY.md "The problem" derives this) — the phantom is built so speckle emerges from the
    same wave-interference physics a real scan would show, not from an added noise term.

IMPORTANT — single source of truth for array/pulse/phantom-region constants: the values below
MUST match src/kernels.cuh's constexpr constants verbatim (CLAUDE.md paragraph 12's "deliberate,
documented duplication" across languages). src/main.cu CROSS-CHECKS every field of
data/sample/array_params.csv against kernels.cuh at load time (the same pattern project 03.01
uses for radar_params.csv) — a drift between this script and kernels.cuh is a loud SCENARIO:
MISMATCH failure, not a silently wrong demo.

Usage
-----
    python make_synthetic.py                  # defaults: writes both committed sample files
    python make_synthetic.py --seed 7 --out-dir /tmp/alt
"""

import argparse
import csv
import math
import random
from pathlib import Path

# ===========================================================================
# Array / pulse / phantom-region constants — MUST MATCH src/kernels.cuh.
# ===========================================================================
NUM_ELEMENTS     = 64          # kNumElements
ELEMENT_PITCH_M  = 0.30e-3     # kElementPitchM
CENTER_FREQ_HZ   = 5.0e6       # kCenterFreqHz
SAMPLING_FREQ_HZ = 40.0e6      # kSamplingFreqHz
SOUND_SPEED_MPS  = 1540.0      # kSoundSpeedMps
F_NUMBER         = 1.5         # kFNumber
INCLUSION_X_M    = -6.0e-3     # kInclusionCenterXM
INCLUSION_Z_M    = 15.0e-3     # kInclusionCenterZM
INCLUSION_RADIUS_M = 2.5e-3    # kInclusionRadiusM

# Imaging field of view (kernels.cuh kImageXMinM/XMaxM/ZMinM/ZMaxM) — the phantom is generated
# slightly INSET from these bounds so every scatterer's pulse footprint has room to be simulated
# without truncation right at the image edge (see src/main.cu's channel-data synthesis).
FOV_X_MIN_M = -9.4e-3
FOV_X_MAX_M =  9.4e-3
FOV_Z_MIN_M = 10.3e-3
FOV_Z_MAX_M = 29.7e-3

DEFAULT_SEED = 42   # fixed seed: CLAUDE.md paragraph 12 determinism

# ---------------------------------------------------------------------------
# The 9-wire "+" cross pattern: a horizontal row at z=20mm (tests LATERAL
# localization/spacing across the field of view) and a vertical column at
# x=0 (tests AXIAL localization/spacing), sharing the center point
# (0, 20mm) — which src/main.cu also uses as "the isolated wire" for the
# resolution-measurement (PSF) gate, since its nearest neighbors are 4mm
# away in every direction, far beyond the ~0.3-0.5mm resolution the array
# achieves (see ../THEORY.md "The math").
# ---------------------------------------------------------------------------
WIRE_TARGETS_M = [
    (-8.0e-3, 20.0e-3), (-4.0e-3, 20.0e-3), (0.0, 20.0e-3), (4.0e-3, 20.0e-3), (8.0e-3, 20.0e-3),
    (0.0, 12.0e-3), (0.0, 16.0e-3), (0.0, 24.0e-3), (0.0, 28.0e-3),
]
WIRE_AMP = 15.0         # strong point reflectors — the brightest thing in the image, by design;
                        # deliberately far above the background/inclusion amplitudes below so each
                        # wire's mainlobe unambiguously dominates local speckle interference (a real
                        # QA-phantom wire is a highly reflective monofilament for the same reason —
                        # see PRACTICE.md §2)

INCLUSION_COUNT = 1400  # scatterers inside the disk (~2x the background density below)
INCLUSION_AMP = 1.4     # 1.4x background amplitude
BACKGROUND_COUNT = 18000 # scatterers over the rest of the field of view — enough for a fine-
                        # grained (not blobby) speckle texture at this array's resolution scale
BACKGROUND_AMP = 1.0    # IDENTICAL amplitude (the Rayleigh-speckle setup — see file header)


def in_inclusion(x_m: float, z_m: float) -> bool:
    """True if (x_m, z_m) falls inside the inclusion disk (kernels.cuh geometry)."""
    dx = x_m - INCLUSION_X_M
    dz = z_m - INCLUSION_Z_M
    return (dx * dx + dz * dz) <= (INCLUSION_RADIUS_M * INCLUSION_RADIUS_M)


def make_phantom(seed: int) -> list[tuple[str, float, float, float]]:
    """Build the full scatterer list deterministically. Returns rows of
    (kind, x_m, z_m, amp_rel) in a FIXED order: wires, then inclusion,
    then background — so the same seed always produces byte-identical
    output (CLAUDE.md paragraph 12).
    """
    rng = random.Random(seed)   # local RNG: never touch the global seed
    rows: list[tuple[str, float, float, float]] = []

    # -- Wires: fixed positions, no randomness -------------------------------
    for x_m, z_m in WIRE_TARGETS_M:
        rows.append(("wire", x_m, z_m, WIRE_AMP))

    # -- Inclusion: uniform-random points inside the disk (rejection
    #    sampling on a bounding square — the disk covers pi/4 ~ 78.5% of the
    #    square, so this converges in a handful of draws per accepted point).
    n = 0
    while n < INCLUSION_COUNT:
        x_m = rng.uniform(INCLUSION_X_M - INCLUSION_RADIUS_M, INCLUSION_X_M + INCLUSION_RADIUS_M)
        z_m = rng.uniform(INCLUSION_Z_M - INCLUSION_RADIUS_M, INCLUSION_Z_M + INCLUSION_RADIUS_M)
        if in_inclusion(x_m, z_m):
            rows.append(("inclusion", x_m, z_m, INCLUSION_AMP))
            n += 1

    # -- Background: uniform-random over the field of view, EXCLUDING the
    #    inclusion disk (the inclusion REPLACES background tissue, it does
    #    not sit on top of it — a real hyperechoic region displaces the
    #    surrounding tissue-mimicking material the same way).
    n = 0
    while n < BACKGROUND_COUNT:
        x_m = rng.uniform(FOV_X_MIN_M, FOV_X_MAX_M)
        z_m = rng.uniform(FOV_Z_MIN_M, FOV_Z_MAX_M)
        if not in_inclusion(x_m, z_m):
            rows.append(("speckle", x_m, z_m, BACKGROUND_AMP))
            n += 1

    return rows


def write_array_params(out_path: Path) -> None:
    """Write the single-row array/pulse/phantom-region parameter file that
    src/main.cu cross-checks against src/kernels.cuh's constexpr constants
    (the same pattern project 03.01 uses for radar_params.csv). One data
    row, 9 fields, in the exact order src/main.cu's loader expects.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        f.write("# SYNTHETIC data — 29.05 ultrasound array/pulse/phantom-region parameters\n")
        f.write("# EDUCATIONAL/SYNTHETIC ONLY — no patient data, no diagnostic or therapeutic claim.\n")
        f.write(f"# regenerate: python make_synthetic.py --seed {DEFAULT_SEED}\n")
        f.write("# columns (no header row, exactly one data row):\n")
        f.write("#   num_elements,pitch_m,fc_hz,fs_hz,c_mps,fnumber,incl_x_m,incl_z_m,incl_radius_m\n")
        writer = csv.writer(f)
        writer.writerow([
            NUM_ELEMENTS, f"{ELEMENT_PITCH_M:.8f}", f"{CENTER_FREQ_HZ:.1f}", f"{SAMPLING_FREQ_HZ:.1f}",
            f"{SOUND_SPEED_MPS:.1f}", f"{F_NUMBER:.3f}",
            f"{INCLUSION_X_M:.8f}", f"{INCLUSION_Z_M:.8f}", f"{INCLUSION_RADIUS_M:.8f}",
        ])
    print(f"[make_synthetic] wrote array_params to {out_path}")


def write_phantom(out_path: Path, rows: list[tuple[str, float, float, float]], seed: int) -> None:
    """Write the full scatterer list. columns: kind,x_m,z_m,amp_rel."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_wire = sum(1 for r in rows if r[0] == "wire")
    n_incl = sum(1 for r in rows if r[0] == "inclusion")
    n_spk  = sum(1 for r in rows if r[0] == "speckle")
    with out_path.open("w", newline="", encoding="utf-8") as f:
        f.write("# SYNTHETIC data — 29.05 ultrasound point-scatterer phantom\n")
        f.write("# EDUCATIONAL/SYNTHETIC ONLY — no patient data, no diagnostic or therapeutic claim.\n")
        f.write(f"# generated by scripts/make_synthetic.py, seed={seed}\n")
        f.write(f"# regenerate: python make_synthetic.py --seed {seed}\n")
        f.write(f"# counts: {n_wire} wire, {n_incl} inclusion, {n_spk} speckle (total {len(rows)})\n")
        f.write("# columns: kind,x_m,z_m,amp_rel (kind: wire | inclusion | speckle; x_m/z_m in meters,\n")
        f.write("#          array/image frame per kernels.cuh; amp_rel: unitless relative reflectivity)\n")
        writer = csv.writer(f)
        writer.writerow(["kind", "x_m", "z_m", "amp_rel"])
        for kind, x_m, z_m, amp in rows:
            writer.writerow([kind, f"{x_m:.8f}", f"{z_m:.8f}", f"{amp:.4f}"])
    print(f"[make_synthetic] wrote {len(rows)} scatterers ({n_wire} wire, {n_incl} inclusion, "
          f"{n_spk} speckle) to {out_path} (seed={seed}, labeled SYNTHETIC)")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_out_dir = script_dir.parent / "data" / "sample"

    parser = argparse.ArgumentParser(
        description="Generate the synthetic phantom for project 29.05 (Ultrasound: GPU beamforming). "
                    "Educational/synthetic only — no patient data.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"RNG seed for byte-identical reproducibility (default {DEFAULT_SEED})")
    parser.add_argument("--out-dir", type=Path, default=default_out_dir,
                        help="output directory (default: ../data/sample/)")
    args = parser.parse_args()

    rows = make_phantom(args.seed)
    write_array_params(args.out_dir / "array_params.csv")
    write_phantom(args.out_dir / "phantom.csv", rows, args.seed)


if __name__ == "__main__":
    main()
