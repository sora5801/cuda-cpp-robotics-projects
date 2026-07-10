#!/usr/bin/env python3
"""make_synthetic.py — synthetic scenario generator for project 14.02
(Traversability costmaps fusing semantics + geometry).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
---------------------------------------------------------------
An off-road elevation+semantics map has no real-world ground truth to check
a kernel against (a recorded LiDAR scan gives no "correct" answer for what
the fused traversability cost of a particular cell SHOULD be). This
project's "dataset" is therefore not a recording: it is a small, human-
readable RECIPE (the same shape as project 13.03's terrain_scenario.csv)
that fully determines BOTH a 256x256 elevation map and a co-registered
256x256 six-class semantic map with per-cell confidence — deterministically,
with the recipe's numbers as the only source of truth.

Why a recipe and not the two full 65536-cell grids
  1. `data/sample/` stays tiny (CLAUDE.md paragraph 8): this recipe is a few
     KiB; two full float32/uint8 256x256 grids would be ~300 KiB.
  2. The recipe is closed-form deterministic geometry (a ramp-like berm, a
     V-shaped ditch, dome-shaped rocks, sine ripples) plus piecewise-constant
     semantic regions — `src/main.cu`'s build_elevation()/build_semantics()
     regenerate the identical grids from the same recipe with ZERO run-time
     randomness. The one genuine random choice in the whole pipeline — WHERE
     the rock-patch domes sit — is resolved HERE, once, with a seeded RNG
     (seed 42), and the resulting rock centers/heights/radii are written into
     the recipe as literal numbers, exactly 13.03's pattern.

What it writes: ../data/sample/traversability_scenario.csv

    RIPPLE,amplitude_m,wavelength_m           background rolling-terrain
                                               ripple applied EVERYWHERE
                                               (z += A*sin(2*pi*(x+y)/L) — a
                                               deterministic traveling-wave
                                               ripple, not real sensor noise,
                                               chosen for its clean closed-
                                               form worst-case gradient;
                                               THEORY.md derives it)
    BERM,x0,x1,y0,y1,angle_deg                a linear ramp-then-plateau
                                               ridge confined to one
                                               rectangular footprint (m)
    DITCH,xs,xb0,xb1,xe,y0,y1,depth_m         a trapezoidal V-shaped
                                               depression: linear descent
                                               [xs,xb0], flat bottom
                                               [xb0,xb1], linear ascent
                                               [xb1,xe] (m)
    ROCK,cx,cy,h,r                            one smooth dome per rock (m);
                                               N of these rows, seed 42
    VEGBUMP,x0,x1,y0,y1,amplitude_m,wavelength_m
                                               a HIGH-frequency ripple ADDED
                                               on top of RIPPLE, confined to
                                               one rectangle — stands in for
                                               noisy LiDAR returns off a
                                               vegetation canopy (THEORY.md)
    SEMREGION,class_name,confidence,x0,x1,y0,y1
                                               paints one rectangle's semantic
                                               class + BASE confidence; rows
                                               are applied IN ORDER, later
                                               rows override earlier ones
                                               within any overlap (the first
                                               row covers the whole map as
                                               the default background)
    CONFNOISE,amplitude,wavelength            smooth deterministic per-cell
                                               confidence jitter added
                                               everywhere after painting,
                                               then clamped to [0.05,0.99] —
                                               stands in for a real softmax's
                                               cell-to-cell confidence
                                               variation without needing a
                                               full per-cell array in this
                                               tiny recipe file
    WAYPOINT,label,x,y                        one point on the teaching
                                               TRANSECT polyline that
                                               demo/out/layers.csv samples;
                                               consecutive waypoints are
                                               connected by straight legs

Grid constants (256x256 cells @ 0.10 m/cell = 25.6x25.6 m), the fit-window
radii, the six-class palette and its prior costs, the wheeled-vehicle
friction/rollover/wheel-radius constants the two hard-veto limits are
derived from, the fusion weights, and the speed-limit constants are NOT part
of this file — they are the "tuned, taught setup" (CLAUDE.md paragraph 8's
data-vs-algorithm distinction) and live as documented constants in
../src/kernels.cuh.

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
# Layout: SIX non-overlapping horizontal Y-BANDS across the 25.6x25.6 m map
# (map-frame y = row * kCellM), each owned by exactly one named feature.
# Bands are separated by >=0.3 m gaps of plain background ripple — wider than
# the geometric kernel's widest window (kFitRadiusCells=3 cells = 0.3 m half-
# width), so no analytic gate in main.cu ever sees two DIFFERENT features
# blended together (13.03's non-overlapping-bands discipline, reused).
#
#   Band 1  y in [ 0.5, 4.3)  CONTROL      background ripple only
#   Band 2  y in [ 4.6, 8.4)  BERM         18 deg ridge, x in [9.0,10.0]
#   Band 3  y in [ 8.7,12.5)  DITCH        0.5 m deep, 45 deg walls
#   Band 4  y in [12.8,16.6)  ROCK PATCH   10 domes (seeded), x in [1,24]
#   Band 5  y in [16.9,20.7)  WATER        pool x in [18,20], grass elsewhere
#   Band 6  y in [21.0,24.8)  VEGETATION   high-freq canopy bump, x in [10,16]
# ---------------------------------------------------------------------------
ROCKS_X0, ROCKS_X1 = 1.5, 23.5
ROCKS_Y0, ROCKS_Y1 = 13.1, 16.3
N_ROCKS = 10
ROCK_H_RANGE = (0.05, 0.15)   # dome peak height (m) — a real rock/boulder scale
ROCK_R_RANGE = (0.30, 0.80)   # dome footprint radius (m)
ROCK_MIN_GAP = 0.20           # minimum edge-to-edge clearance between rocks (m)


def place_rocks(rng: random.Random) -> list[tuple[float, float, float, float]]:
    """Rejection-sample N_ROCKS non-overlapping dome centers.

    Identical technique to 13.03's place_rocks(): try a random disk, keep it
    only if it clears every previously placed rock by ROCK_MIN_GAP. Returns
    a list of (cx_m, cy_m, h_m, r_m).
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
                    default=Path(__file__).resolve().parent.parent / "data" / "sample" / "traversability_scenario.csv")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rocks = place_rocks(rng)

    lines = [
        "# traversability_scenario.csv - SYNTHETIC composed elevation+semantics recipe for project 14.02",
        "# generated by scripts/make_synthetic.py (rock placement RNG seed "
        f"{args.seed}; everything else is exact deterministic geometry)",
        "# Grid constants (W,H,cell_m), fit-window radii, the six-class palette + priors, the",
        "# wheeled-vehicle friction/rollover/wheel-radius constants, fusion weights, and speed-",
        "# limit constants are NOT here - they are the tuned algorithm configuration and live in",
        "# ../src/kernels.cuh (CLAUDE.md doc-8).",
        "#",
        "# RIPPLE,amplitude_m,wavelength_m                  background rolling terrain, everywhere",
        "# BERM,x0,x1,y0,y1,angle_deg                       ramp-then-plateau ridge (m, deg)",
        "# DITCH,xs,xb0,xb1,xe,y0,y1,depth_m                trapezoidal V-shaped depression (m)",
        "# ROCK,cx,cy,h,r                                   one smooth dome (m); 10 rows",
        "# VEGBUMP,x0,x1,y0,y1,amplitude_m,wavelength_m     high-freq canopy-noise bump (m)",
        "# SEMREGION,class,confidence,x0,x1,y0,y1           semantic class + base confidence (m);",
        "#                                                  applied in order, later overrides earlier",
        "# CONFNOISE,amplitude,wavelength                   smooth per-cell confidence jitter",
        "# WAYPOINT,label,x,y                               one teaching-transect polyline point (m)",
        "# license: same as the repository (MIT) - fully synthetic, no external source",
        "RIPPLE,0.02,7.0",
        "BERM,9.0,10.0,4.6,8.4,18.0",
        "DITCH,12.5,13.0,13.4,13.9,8.7,12.5,0.5",
    ]
    for (cx, cy, h, r) in rocks:
        lines.append(f"ROCK,{cx:.4f},{cy:.4f},{h:.4f},{r:.4f}")
    lines += [
        "VEGBUMP,10.0,16.0,21.0,24.8,0.013,1.0",
        "# Semantic regions, painted in order (first row = whole-map default background):",
        "SEMREGION,DIRT,0.90,0.0,25.6,0.0,25.6",
        "SEMREGION,GRAVEL,0.85,8.5,14.5,4.6,12.5",
        "SEMREGION,GRAVEL,0.85,1.0,24.0,12.8,16.6",
        "SEMREGION,UNKNOWN,0.25,3.0,4.0,1.0,2.0",
        "SEMREGION,GRASS,0.90,1.0,7.0,17.0,20.5",
        "SEMREGION,WATER,0.85,18.0,20.0,17.9,19.7",
        "SEMREGION,VEGETATION,0.88,10.0,16.0,21.0,24.8",
        "CONFNOISE,0.06,3.3",
        "# Teaching-transect waypoints (demo/out/layers.csv): control -> berm -> ditch -> rocks",
        "# -> water pool -> vegetation patch, straight legs sampled at N points each:",
        "WAYPOINT,CONTROL,2.0,2.4",
        "WAYPOINT,BERM,9.5,6.5",
        "WAYPOINT,DITCH,13.2,10.6",
        "WAYPOINT,ROCKPATCH,12.0,14.7",
        "WAYPOINT,WATER,19.0,18.8",
        "WAYPOINT,VEGETATION,13.0,22.9",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:   # LF pinned
        f.write("\n".join(lines) + "\n")

    print(f"wrote {args.out} ({args.out.stat().st_size} bytes: "
          f"{len(rocks)} rocks, 6 semantic regions, 6 waypoints, seed={args.seed}) - labeled SYNTHETIC")
    if args.seed != 42:
        print("note: non-default seed - fine for experiments, do NOT commit this file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
