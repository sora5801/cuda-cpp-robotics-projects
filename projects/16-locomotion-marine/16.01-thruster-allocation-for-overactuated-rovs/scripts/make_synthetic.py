#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample-data generator for 16.01
(Thruster allocation for overactuated ROVs, batched QP).

Why this script exists (CLAUDE.md §8: synthetic-first)
--------------------------------------------------------
Robotics data can almost always be synthesized with full ground truth, so
synthetic generation is this repository's DEFAULT data source. This script
writes two files into ../data/sample/:

  1. rov_geometry.csv   — a HUMAN-READABLE documentation copy of the 8
     thruster positions/directions/limits. The program itself does NOT read
     this file at runtime — the geometry is compiled in as constexpr tables
     in ../src/kernels.cuh (CLAUDE.md §12: state is single-sourced; the
     "source of truth" for the running program is the C++ header, not a
     CSV). This file exists so a learner can inspect the vehicle's geometry
     without reading C++, and so the numbers are traceable/regenerable.
     IF YOU CHANGE THE GEOMETRY: update kernels.cuh AND re-run this script
     so the two stay in agreement (there is no automated check for this;
     that is an honest, documented limitation — README "Limitations").

  2. wrench_batch.csv   — the file the program DOES read at runtime
     (data/sample/wrench_batch.csv, loaded by ../src/main.cu). A synthetic
     50-second, 10 Hz commanded-wrench trajectory representing a ROV holding
     station against a fluctuating current and executing two aggressive
     "docking correction" bursts (the kind of command a station-keeping
     controller or 16.09's docking-under-current MPPI would issue). The
     bursts are sized to SATURATE several thrusters on purpose — the demo
     needs both unsaturated and saturated rows to exercise both optimality
     gates (README "Expected output").

Determinism: everything here is closed-form sinusoids plus a SEEDED
random.Random(42) noise term (Python's Mersenne Twister — cross-platform
deterministic given a fixed seed), so the same command produces the exact
same bytes on every machine, every run (CLAUDE.md §12).

Usage
-----
    python make_synthetic.py                  # regenerate both sample files
    python make_synthetic.py --seed 7 --n 500  # experiment with a new seed
"""

import argparse
import csv
import math
import random
from pathlib import Path

DEFAULT_SEED = 42          # fixed seed -> byte-identical regeneration (CLAUDE.md §12)
DEFAULT_N = 500             # samples in the wrench trajectory (500 @ 10 Hz = 50 s)
DEFAULT_DT = 0.1            # sample period (s) -> 10 Hz, a realistic DP/allocation tick

# --- Vehicle geometry — MUST mirror ../src/kernels.cuh verbatim -------------
# (see the module docstring: this is documentation, not the runtime source).
K_HX, K_HY = 0.20, 0.15      # horizontal-thruster fore/aft, port/starboard offsets (m)
K_VX, K_VY = 0.15, 0.20      # vertical-thruster fore/aft, port/starboard offsets (m)
K_UMAX = 40.0                # per-thruster saturation limit (N), symmetric +-
C45 = math.sqrt(2.0) / 2.0   # cos(45 deg) = sin(45 deg)

# (name, x, y, z, dx, dy, dz) — body frame x-forward/y-starboard/z-down
# (Fossen/SNAME marine convention — see kernels.cuh header for why this
# project deviates from the repo's default x-forward/y-left/z-up).
THRUSTERS = [
    ("H1", +K_HX, +K_HY, 0.0, C45, -C45, 0.0),   # horizontal, fore-starboard
    ("H2", +K_HX, -K_HY, 0.0, C45, +C45, 0.0),   # horizontal, fore-port
    ("H3", -K_HX, +K_HY, 0.0, C45, +C45, 0.0),   # horizontal, aft-starboard
    ("H4", -K_HX, -K_HY, 0.0, C45, -C45, 0.0),   # horizontal, aft-port
    ("V1", +K_VX, +K_VY, 0.0, 0.0, 0.0, -1.0),   # vertical, fore-starboard corner
    ("V2", +K_VX, -K_VY, 0.0, 0.0, 0.0, -1.0),   # vertical, fore-port corner
    ("V3", -K_VX, +K_VY, 0.0, 0.0, 0.0, -1.0),   # vertical, aft-starboard corner
    ("V4", -K_VX, -K_VY, 0.0, 0.0, 0.0, -1.0),   # vertical, aft-port corner
]


def write_geometry_csv(out_path: Path) -> None:
    """Write the human-readable thruster-geometry documentation file.

    Columns: thruster id, mount position (m, body frame), unit thrust
    direction (dimensionless), and the saturation limit (N). See the module
    docstring for why this file is NOT read by the running program.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        f.write("# SYNTHETIC geometry documentation for project 16.01 — mirrors src/kernels.cuh\n")
        f.write("# NOT read at runtime (the program's source of truth is kernels.cuh); see make_synthetic.py docstring\n")
        f.write("# body frame: x-forward, y-starboard, z-down (Fossen/SNAME marine convention)\n")
        f.write("# regenerate: python make_synthetic.py\n")
        w = csv.writer(f)
        w.writerow(["thruster", "x_m", "y_m", "z_m", "dx", "dy", "dz", "u_max_N"])
        for name, x, y, z, dx, dy, dz in THRUSTERS:
            w.writerow([name, f"{x:.4f}", f"{y:.4f}", f"{z:.4f}",
                        f"{dx:.6f}", f"{dy:.6f}", f"{dz:.6f}", f"{K_UMAX:.2f}"])
    print(f"[make_synthetic] wrote {len(THRUSTERS)} thruster rows to {out_path} (documentation only)")


def raised_cosine(t: float, t0: float, duration: float) -> float:
    """A smooth 0->1->0 bump over [t0, t0+duration], peaking at the midpoint.

    Used to ramp the "docking correction burst" wrenches in and out smoothly
    (a real controller does not command a force step discontinuity) instead
    of a hard on/off pulse. Zero outside the window; continuous everywhere.
    """
    if t < t0 or t > t0 + duration:
        return 0.0
    return 0.5 * (1.0 - math.cos(2.0 * math.pi * (t - t0) / duration))


def make_wrench_row(t: float, rng: random.Random) -> tuple:
    """Synthesize one commanded wrench (Fx,Fy,Fz,Mx,My,Mz) at time t (s).

    Two layers, added together:
      1) BASE DISTURBANCE — small, always-on sinusoids (a few Newtons/
         Newton-meters) standing in for a station-keeping controller's
         continuous correction against a fluctuating current, PLUS a small
         seeded-random jitter per channel (turbulent-flow-like noise). This
         layer alone never saturates any thruster (README "Expected output"
         confirms this by construction: the batch's unsaturated subset comes
         from here).
      2) DOCKING-CORRECTION BURSTS — two raised-cosine-windowed pulses,
         deliberately large enough to saturate multiple thrusters at once:
         burst 1 (peak at t=15s) is a combined surge+yaw correction (like
         powering toward a dock while countering a yaw disturbance); burst 2
         (peak at t=33s) is a combined sway+roll+heave correction (like
         crabbing sideways against cross-current while trimming depth). This
         is exactly the kind of aggressive short-duration command a
         station-keeping/docking controller (16.09's docking-under-current
         MPPI, by name) would issue — and exactly what exercises this
         project's box constraints.

    Returns the 6-tuple (Fx_N, Fy_N, Fz_N, Mx_Nm, My_Nm, Mz_Nm).
    """
    two_pi = 2.0 * math.pi
    Fx = 4.0 * math.sin(two_pi * 0.05 * t) + 1.5 * math.sin(two_pi * 0.13 * t + 0.7)
    Fy = 3.0 * math.cos(two_pi * 0.04 * t + 0.3)
    Fz = 2.0 * math.sin(two_pi * 0.03 * t + 1.1)
    Mx = 0.5 * math.sin(two_pi * 0.06 * t)
    My = 0.4 * math.cos(two_pi * 0.05 * t + 0.2)
    Mz = 2.0 * math.sin(two_pi * 0.045 * t + 0.5)

    # Turbulent-current jitter: independent uniform noise per channel, small
    # relative to the base signal (never enough, by itself, to saturate).
    Fx += rng.uniform(-0.5, 0.5)
    Fy += rng.uniform(-0.5, 0.5)
    Fz += rng.uniform(-0.5, 0.5)
    Mx += rng.uniform(-0.5, 0.5)
    My += rng.uniform(-0.5, 0.5)
    Mz += rng.uniform(-0.5, 0.5)

    # Burst 1: forward+turn docking correction (saturates the HORIZONTAL group).
    w1 = raised_cosine(t, 12.5, 5.0)
    Fx += 130.0 * w1
    Mz += -30.0 * w1

    # Burst 2: sideways+roll+heave crabbing correction (saturates the mixed group).
    w2 = raised_cosine(t, 30.0, 6.0)
    Fy += 120.0 * w2
    Mx += 28.0 * w2
    Fz += -40.0 * w2

    return (Fx, Fy, Fz, Mx, My, Mz)


def write_wrench_batch_csv(out_path: Path, n: int, dt: float, seed: int) -> None:
    """Write the n-row synthetic wrench trajectory main.cu loads at runtime."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)   # LOCAL rng: never touch the global seed (CLAUDE.md pattern)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        f.write("# SYNTHETIC data — generated by scripts/make_synthetic.py for project 16.01\n")
        f.write(f"# regenerate: python make_synthetic.py --n {n} --seed {seed}\n")
        f.write("# a synthetic docking-under-current commanded-wrench trajectory, 10 Hz\n")
        f.write("# columns: t_s (s), Fx_N,Fy_N,Fz_N (N, body frame), Mx_Nm,My_Nm,Mz_Nm (N*m, body frame)\n")
        f.write("# body frame: x-forward, y-starboard, z-down (Fossen/SNAME marine convention)\n")
        w = csv.writer(f)
        w.writerow(["t_s", "Fx_N", "Fy_N", "Fz_N", "Mx_Nm", "My_Nm", "Mz_Nm"])
        for i in range(n):
            t = i * dt
            Fx, Fy, Fz, Mx, My, Mz = make_wrench_row(t, rng)
            w.writerow([f"{t:.2f}", f"{Fx:.6f}", f"{Fy:.6f}", f"{Fz:.6f}",
                        f"{Mx:.6f}", f"{My:.6f}", f"{Mz:.6f}"])
    print(f"[make_synthetic] wrote {n} rows ({n*dt:.1f} s @ {1.0/dt:.0f} Hz) to {out_path} "
          f"(seed={seed}, labeled SYNTHETIC)")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    sample_dir = script_dir.parent / "data" / "sample"

    parser = argparse.ArgumentParser(
        description="Generate the synthetic sample data for project 16.01 "
                    "(thruster allocation for overactuated ROVs).")
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help=f"number of wrench-trajectory samples (default {DEFAULT_N})")
    parser.add_argument("--dt", type=float, default=DEFAULT_DT,
                        help=f"sample period in seconds (default {DEFAULT_DT})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"RNG seed for byte-identical reproducibility (default {DEFAULT_SEED})")
    parser.add_argument("--out-dir", type=Path, default=sample_dir,
                        help="output directory (default ../data/sample/)")
    args = parser.parse_args()

    if args.n <= 0:
        parser.error("--n must be > 0")
    if args.dt <= 0:
        parser.error("--dt must be > 0")

    write_geometry_csv(args.out_dir / "rov_geometry.csv")
    write_wrench_batch_csv(args.out_dir / "wrench_batch.csv", args.n, args.dt, args.seed)


if __name__ == "__main__":
    main()
