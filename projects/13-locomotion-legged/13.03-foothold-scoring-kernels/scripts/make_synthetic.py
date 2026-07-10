#!/usr/bin/env python3
"""make_synthetic.py — synthetic terrain-scenario generator for project 13.03
(Foothold scoring kernels: slope, roughness, edge distance from elevation maps).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
---------------------------------------------------------------
Robotics elevation maps almost always have exact synthetic ground truth
available (a recorded depth camera over real terrain gives no ground-truth
slope to check a kernel against). This project's "dataset" is therefore not
a recording: it is a small, human-readable RECIPE (the same shape as project
08.01's cartpole_scenario.csv — "a scenario, not recordings") that fully
determines a 256x256 elevation map — flat ground, a ramp of a KNOWN angle, a
step of a KNOWN height, a field of scattered rocks, and a rectangular sensor
dropout (NaN hole) — plus a 5-segment foothold-query path that walks a
virtual quadruped's feet across every one of those features.

Why a recipe and not the full 65536-cell grid?
  1. `data/sample/` should stay tiny (CLAUDE.md paragraph 8: "kilobytes
     preferred") — a recipe is ~2 KiB; a full float32 grid would be ~256 KiB
     for this fairly small demo map and grows quadratically for anything
     bigger.
  2. The recipe is fully DETERMINISTIC closed-form geometry (ramp/step
     planes, rock domes, a rectangular hole) — `src/main.cu`'s
     `build_terrain()` regenerates the identical 256x256 grid from the same
     recipe with zero randomness of its own. The one piece of true
     randomness in the whole pipeline — WHERE the 16 rocks sit — is resolved
     HERE, once, with a seeded RNG, and the resulting rock centers/heights/
     radii are written into the recipe as literal numbers. `main.cu` never
     runs an RNG at all: it only ever parses numbers, which is what keeps
     the C++/GPU side of this project free of any cross-language
     determinism concerns.

What it writes: ../data/sample/terrain_scenario.csv

    RIPPLE,amplitude_m,wavelength_m         background sensor-noise-like
                                             ripple applied EVERYWHERE (a
                                             smooth deterministic sin*cos
                                             field, not real noise — its
                                             worst-case slope has a clean
                                             closed form; see THEORY.md)
    RAMP,x0,x1,y0,y1,angle_deg              a linear ramp confined to one
                                             rectangular footprint (m)
    STEP,x0,x1,y0,y1,edge_x,height_m        a vertical step confined to one
                                             rectangular footprint (m)
    ROCK,cx,cy,h,r                          one smooth dome per rock (m);
                                             16 of these rows
    HOLE,x0,x1,y0,y1                        a rectangular NaN block — a
                                             sensor dropout / real gap (m)
    PATH,x0,y0,x1,y1,n                      one straight query segment: n
                                             foothold queries linearly
                                             interpolated from (x0,y0) to
                                             (x1,y1); 5 of these rows, one
                                             per terrain feature, 200 points
                                             each = 1000 queries total

Grid constants (256x256 cells @ 0.02 m/cell = 5.12x5.12 m), the plane-fit
window radius, the friction/roughness/edge/weight/threshold constants, and
the query search radius are NOT part of this file — they are the "tuned,
taught setup" (the same distinction 08.01 draws for its MPPI weights) and
live as documented constants in ../src/kernels.cuh.

Usage:
    python make_synthetic.py                 # the committed scenario (seed 42)
    python make_synthetic.py --seed 7         # experiments; do not commit
"""

import argparse
import math
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Terrain layout: FIVE non-overlapping horizontal Y-BANDS across the
# 5.12x5.12 m map, each owned by exactly one feature (map-frame y = row *
# kCellM). Bands are separated by >=0.20 m gaps of plain background ripple
# — wide enough that the plane-fit window (0.1 m) and the edge-distance
# search (0.2 m) never blend two DIFFERENT features together, so every
# analytic gate in main.cu can assume a clean, isolated region. Both the
# RAMP's plateau and the STEP's upper shelf are defined to PERSIST for all
# x beyond their transition (never revert to baseline) — the alternative
# (stopping at a fixed x1) would silently create a second, undocumented
# cliff at that boundary, which this recipe deliberately avoids.
#
#   Band A  y in [0.20, 0.80]  FLAT CONTROL  (baseline ripple only)
#   Band B  y in [1.00, 1.60]  RAMP          (15 deg, x in [1.00,2.00])
#   Band C  y in [1.90, 2.50]  STEP          (0.12 m, edge at x=3.45)
#   Band D  y in [2.90, 4.00]  ROCK FIELD    (16 domes, x in [0.30,4.80])
#   Band E  y in [4.30, 4.70]  HOLE VICINITY (NaN block, x in [4.00,4.40])
# ---------------------------------------------------------------------------
ROCKS_X0, ROCKS_X1 = 0.30, 4.80
ROCKS_Y0, ROCKS_Y1 = 2.90, 4.00
N_ROCKS = 16
ROCK_H_RANGE = (0.02, 0.05)   # dome peak height (m) — gentle enough that a
                              # foothold search disc can usually find safe
                              # ground nearby (see README "Limitations")
ROCK_R_RANGE = (0.05, 0.09)   # dome footprint radius (m)
ROCK_MIN_GAP = 0.03           # minimum edge-to-edge clearance between rocks (m)


def place_rocks(rng: random.Random) -> list[tuple[float, float, float, float]]:
    """Rejection-sample N_ROCKS non-overlapping dome centers.

    Returns a list of (cx_m, cy_m, h_m, r_m). Rejection sampling (try a
    random point, keep it only if it clears every previously placed rock by
    ROCK_MIN_GAP) is the simplest correct way to scatter non-overlapping
    disks — the same technique 07.09's seed generator uses for grid cells,
    here in continuous space instead of on a lattice.
    """
    rocks: list[tuple[float, float, float, float]] = []
    attempts = 0
    while len(rocks) < N_ROCKS and attempts < 20000:
        attempts += 1
        r = rng.uniform(*ROCK_R_RANGE)
        cx = rng.uniform(ROCKS_X0 + r, ROCKS_X1 - r)
        cy = rng.uniform(ROCKS_Y0 + r, ROCKS_Y1 - r)
        ok = True
        for (ocx, ocy, _oh, orr) in rocks:
            if math.hypot(cx - ocx, cy - ocy) < (r + orr + ROCK_MIN_GAP):
                ok = False
                break
        if not ok:
            continue
        h = rng.uniform(*ROCK_H_RANGE)
        rocks.append((cx, cy, h, r))
    if len(rocks) < N_ROCKS:
        print(f"warning: only placed {len(rocks)}/{N_ROCKS} rocks after {attempts} attempts",
              file=sys.stderr)
    return rocks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for rock placement (default 42)")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data" / "sample" / "terrain_scenario.csv")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rocks = place_rocks(rng)

    lines = [
        "# terrain_scenario.csv - SYNTHETIC composed elevation-map recipe for project 13.03",
        "# generated by scripts/make_synthetic.py (rock placement RNG seed "
        f"{args.seed}; everything else is exact deterministic geometry)",
        "# Grid constants (W,H,cell_m), plane-fit radius, friction/roughness/edge/weight/",
        "# threshold constants and the query search radius are NOT here - they are the",
        "# tuned algorithm configuration and live in ../src/kernels.cuh (CLAUDE.md doc-8).",
        "#",
        "# RIPPLE,amplitude_m,wavelength_m        background ripple, applied everywhere",
        "# RAMP,x0,x1,y0,y1,angle_deg              linear ramp, one rectangular footprint (m)",
        "# STEP,x0,x1,y0,y1,edge_x,height_m        vertical step, one rectangular footprint (m)",
        "# ROCK,cx,cy,h,r                          one smooth dome (m); 16 rows",
        "# HOLE,x0,x1,y0,y1                        rectangular NaN block - sensor dropout (m)",
        "# PATH,x0,y0,x1,y1,n                      one straight foothold-query segment (m, count)",
        "# license: same as the repository (MIT) - fully synthetic, no external source",
        "RIPPLE,0.002,0.5",
        "RAMP,1.00,2.00,1.00,1.60,15.0",
        "STEP,3.00,3.90,1.90,2.50,3.45,0.12",
    ]
    for (cx, cy, h, r) in rocks:
        lines.append(f"ROCK,{cx:.4f},{cy:.4f},{h:.4f},{r:.4f}")
    lines += [
        "HOLE,4.00,4.40,4.40,4.60",
        "# 5 query segments x 200 points = 1000 foothold queries, one segment per feature:",
        "PATH,0.30,0.30,0.70,0.70,200",   # seg0: FLAT control region (Band A sanity baseline)
        "PATH,1.00,1.30,2.00,1.30,200",   # seg1: across the RAMP (Band B)
        "PATH,3.00,2.20,3.90,2.20,200",   # seg2: across the STEP (Band C)
        "PATH,0.50,3.40,4.50,3.40,200",   # seg3: through the ROCK field (Band D)
        "PATH,3.80,4.38,4.60,4.38,200",   # seg4: skirting the HOLE's edge (Band E)
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:   # LF pinned
        f.write("\n".join(lines) + "\n")

    print(f"wrote {args.out} ({args.out.stat().st_size} bytes: "
          f"{len(rocks)} rocks, 5 path segments, seed={args.seed}) - labeled SYNTHETIC")
    if args.seed != 42:
        print("note: non-default seed - fine for experiments, do NOT commit this file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
