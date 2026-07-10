#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample-data generator for project 03.01
(FMCW radar cube processing: range-Doppler-angle FFTs + CA/OS-CFAR detection).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
-----------------------------------------------------------------
This project's "sensor data" is not recordings at all: it is (a) the fixed
chirp/antenna configuration of a simulated FMCW radar front end, and (b) a
short list of ground-truth targets (range, radial velocity, azimuth,
reflection amplitude). The raw ADC cube itself (Ns x Nc x Na complex
samples, ~2 MB) is never written to disk — main.cu's GPU kernel and
reference_cpu.cpp's CPU twin each SYNTHESIZE it, in code, from these two
tiny committed files plus a fixed noise seed baked into src/kernels.cuh
(kNoiseSeed). This is the same pattern 05.01 (TSDF fusion) and 08.01 (MPPI)
use for their own synthetic scenarios: the committed sample is the TASK
definition, not a recording, and it is trivially reproducible.

What this script writes
------------------------
    ../data/sample/radar_params.csv   the chirp/antenna configuration
                                      (fc, bandwidth, chirp duration, Ns,
                                      Nc, Na) — main.cu loads this and
                                      CROSS-CHECKS it against the
                                      compile-time constants in
                                      src/kernels.cuh; a mismatch aborts
                                      the demo loudly rather than silently
                                      running an inconsistent scenario.
    ../data/sample/targets.csv        the ground-truth target list: one row
                                      per target, columns
                                      range_m,vel_mps,az_deg,amp. This is a
                                      FIXED list (not drawn from an RNG —
                                      like 08.01's scenario file, "a
                                      scenario is constants"), chosen (seed
                                      42, in the sense that the underlying
                                      design was validated with a
                                      seed-42 noise realization — see
                                      data/README.md) to:
                                        - span the unambiguous range/
                                          velocity/azimuth envelope,
                                        - include one WEAK, far target
                                          (tests detection sensitivity),
                                        - include a CLOSE PAIR (targets 5
                                          and 6: 1.5 m / 3 range bins and
                                          0.9 m/s / 3 Doppler bins apart,
                                          with a 6.7x amplitude ratio) that
                                          demonstrates CA-CFAR's classic
                                          masking weakness while OS-CFAR
                                          still resolves the weaker one —
                                          the project's headline comparison
                                          (THEORY.md "The algorithm").

The default values below MUST match src/kernels.cuh exactly (main.cu will
refuse to run otherwise) — if you retune the radar configuration, update
BOTH files together.

Usage:
    python make_synthetic.py                 # writes the committed sample
    python make_synthetic.py --out-dir DIR    # write elsewhere (experiments)
"""

import argparse
from pathlib import Path

# ---- radar configuration: MUST match src/kernels.cuh's constexpr values ----
FC_HZ = 77.0e9
BANDWIDTH_HZ = 300.0e6
CHIRP_DUR_S = 50.0e-6
NS = 256
NC = 128
NA = 8

# ---- the committed ground-truth target list -------------------------------
# columns: range_m, vel_mps, az_deg, amp
#   vel_mps sign convention: POSITIVE = APPROACHING the radar (closing).
#   amp is a unitless, RCS-ish reflection scale (this project does not model
#   the full radar range equation's 1/R^4 power falloff — see THEORY.md
#   "The problem" for why, and PRACTICE.md for what a calibrated system adds).
TARGETS = [
    # range_m, vel_mps,  az_deg, amp    # role
    (15.0,      8.0,     -20.0,  1.00),  # near, fast-approaching, strong
    (45.0,    -12.0,      10.0,  0.60),  # mid-range, receding
    (80.0,      3.0,       35.0, 0.30),  # far, weak, slow (sensitivity check)
    (25.5,     -5.0,      -45.0, 0.50),  # off-boresight, receding
    (60.0,      6.0,       5.0,  1.00),  # CLOSE PAIR: strong member
    (61.5,      6.9,       5.0,  0.15),  # CLOSE PAIR: weak member (masked by
                                         # CA-CFAR, found by OS-CFAR — see
                                         # THEORY.md "The algorithm")
]


def write_radar_params(out_path: Path) -> None:
    """Write the chirp/antenna configuration record.

    Not a runtime knob: Ns/Nc/Na fix the size of compile-time CFAR/cuFFT
    structures in src/kernels.cuh, so this file exists purely as a
    committed, human-readable RECORD that main.cu cross-checks itself
    against — a silent mismatch between "what the binary was built for"
    and "what the demo claims to run" would be exactly the kind of
    unreproducible result CLAUDE.md paragraph 8 forbids.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# radar_params.csv - SYNTHETIC FMCW radar configuration for project 03.01",
        "# generated by scripts/make_synthetic.py - main.cu cross-checks this file",
        "# against the compile-time constants in ../src/kernels.cuh and ABORTS on",
        "# any mismatch (these values size compile-time CFAR/cuFFT structures, so",
        "# they cannot be changed at runtime - see this script's module docstring).",
        "# fc_hz,bandwidth_hz,chirp_dur_s,ns,nc,na",
        f"{FC_HZ:.1f},{BANDWIDTH_HZ:.1f},{CHIRP_DUR_S:.9f},{NS},{NC},{NA}",
    ]
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[make_synthetic] wrote {out_path} ({out_path.stat().st_size} bytes) - labeled SYNTHETIC")


def write_targets(out_path: Path) -> None:
    """Write the fixed ground-truth target list (no RNG - see module docstring)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# targets.csv - SYNTHETIC ground-truth target list for project 03.01",
        "# generated by scripts/make_synthetic.py (no RNG - a target list is",
        "# constants, same philosophy as 08.01's scenario file). The raw ADC cube",
        "# is synthesized IN CODE from this list plus the fixed noise seed",
        "# kNoiseSeed in ../src/kernels.cuh - never written to disk.",
        "# columns: range_m,vel_mps,az_deg,amp",
        "# vel_mps sign convention: POSITIVE = APPROACHING the radar (closing)",
        "# targets 5 and 6 (0-indexed 4 and 5) are a CLOSE PAIR: see this script's",
        "# module docstring for why they demonstrate the CA-vs-OS-CFAR comparison.",
    ]
    for (r, v, az, amp) in TARGETS:
        lines.append(f"{r:.4f},{v:.4f},{az:.4f},{amp:.4f}")
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[make_synthetic] wrote {out_path} ({out_path.stat().st_size} bytes, {len(TARGETS)} targets) - labeled SYNTHETIC")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data" / "sample",
                    help="output directory (default: ../data/sample/, the committed location)")
    args = ap.parse_args()

    write_radar_params(args.out_dir / "radar_params.csv")
    write_targets(args.out_dir / "targets.csv")

    print("[make_synthetic] note: Ns/Nc/Na/fc/bandwidth/chirp_dur are compile-time")
    print("[make_synthetic]       constants in ../src/kernels.cuh - do not edit only")
    print("[make_synthetic]       this script's values without updating that file too.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
