#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample generator for project 04.01
(Massive particle filter localization — 2-D range-beam MCL teaching core).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
---------------------------------------------------------------
A localization filter needs three things: a MAP, a stream of ODOMETRY, and a
stream of RANGE SCANS. All three can be synthesized with FULL ground truth —
which is exactly the robotics advantage this repository leans on: because we
also write down the true pose the scans were taken from, the demo can measure
its own closed-loop error (RMSE vs ground truth) with zero external data.

What it writes (both into ../data/sample/, both labeled SYNTHETIC):

  grid_map.txt          64x64 occupancy grid, 0.25 m/cell (16 m x 16 m world):
                        border walls + five rectangular obstacles, '.' free /
                        '#' occupied. Fixed geometry — no RNG in the map.
  trajectory_scans.csv  one INIT row (true start pose) + 120 STEP rows, each:
                        ground-truth pose AFTER the step's twist, the NOISY
                        odometry measurement of that twist, and 16 NOISY
                        ranges ray-cast from the post-step true pose.

Determinism: the single fixed seed (42, Python's specified Mersenne Twister —
cross-platform deterministic draws) drives all noise. Floats are written with
fixed precision, LF line endings pinned, so regeneration is byte-stable in
practice; the committed copy is canonical (see ../data/README.md for the
honest note about last-ulp libm differences across platforms).

Conventions (must mirror ../src/kernels.cuh — the C++ contract):
  * World frame: origin at the map's lower-left corner, x right, y up,
    right-handed; theta measured CCW from +x. SI units (m, s, rad).
  * Cell (ix, iy) covers [ix*res,(ix+1)*res) x [iy*res,(iy+1)*res).
  * Motion model: unicycle, EULER-integrated with the exact update order the
    GPU predict kernel uses (position with the OLD heading, then heading):
        x += v*cos(th)*dt;  y += v*sin(th)*dt;  th += w*dt
  * Sensor model: 16 beams at angles th + (-pi + b*pi/8), b = 0..15; ranges
    found by FIXED-STEP ray marching (0.125 m steps, max 8.0 m) — the same
    march the weight kernel performs, so the measurement model and the
    filter's expected-range model are consistent by construction.

Usage:
    python make_synthetic.py              # the committed sample (seed 42)
    python make_synthetic.py --seed 7     # experiments; do NOT commit
"""

import argparse
import math
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants mirrored from ../src/kernels.cuh (the single C++ source of truth).
# If you change one side, change the other — the data README documents both.
# ---------------------------------------------------------------------------
MAP_W = 64            # grid width  (cells)
MAP_H = 64            # grid height (cells)
RES_M = 0.25          # cell size (m) -> 16 m x 16 m world
NUM_BEAMS = 16        # range beams per scan (full 360 degrees)
R_MAX_M = 8.0         # maximum sensor range (m)
RAY_STEP_M = 0.125    # ray-march step (m) = RES_M / 2
DT_S = 0.1            # control/scan period (s) -> 10 Hz

# Measurement-noise magnitudes (generation side; the filter's kernels.cuh
# assumes slightly LARGER sigmas — deliberate headroom, see THEORY.md):
SIGMA_ODO_V = 0.05    # odometry linear-velocity noise std-dev (m/s)
SIGMA_ODO_W = 0.05    # odometry angular-velocity noise std-dev (rad/s)
SIGMA_SCAN = 0.10     # range-measurement noise std-dev (m)

DEFAULT_SEED = 42     # repo-law fixed seed (CLAUDE.md paragraph 12)

# Ground-truth course: a rounded-square loop, driven as commanded twists.
# 4 sides x (20 straight steps + 10 quarter-turn steps) = 120 steps = 12 s.
START_POSE = (6.0, 6.0, 0.0)          # (x m, y m, theta rad) — lower-left of the loop
V_CMD = 1.0                            # cruise speed (m/s)
W_TURN = math.pi / 2.0                 # quarter turn over 10 steps x 0.1 s = 1 s (rad/s)
LEG_STEPS = 20                         # straight-leg steps per side
TURN_STEPS = 10                        # arc steps per corner

# Obstacles as inclusive cell-index rectangles (ix0, ix1, iy0, iy1). The loop
# (bounding box roughly [5.4,8.7] x [6.0,9.3] m = cells [21,34]x[24,37])
# orbits the small central pillar E; A-D give the outer beams structure so
# the pose is observable in every direction. Chosen by hand, no RNG.
OBSTACLES = [
    (10, 18, 44, 52),   # A: block, upper-left quadrant
    (44, 52, 10, 16),   # B: block, lower-right quadrant
    (46, 52, 44, 50),   # C: block, upper-right quadrant
    (8, 14, 8, 12),     # D: block, lower-left quadrant
    (30, 31, 30, 31),   # E: 0.5 m pillar at (7.5..8.0, 7.5..8.0) m — inside the loop
]

MIN_CLEARANCE_M = 0.45  # generator asserts the true path keeps this margin


def build_map():
    """Return the occupancy grid as a list of MAP_H rows (iy = 0 first),
    each a list of MAP_W ints (0 free, 1 occupied).

    Geometry is fixed constants: a 1-cell border wall plus OBSTACLES. No RNG —
    a map is world structure, not a measurement.
    """
    occ = [[0] * MAP_W for _ in range(MAP_H)]
    for iy in range(MAP_H):
        for ix in range(MAP_W):
            if ix == 0 or iy == 0 or ix == MAP_W - 1 or iy == MAP_H - 1:
                occ[iy][ix] = 1  # border wall: the world is closed
    for (ix0, ix1, iy0, iy1) in OBSTACLES:
        for iy in range(iy0, iy1 + 1):
            for ix in range(ix0, ix1 + 1):
                occ[iy][ix] = 1
    return occ


def raycast(occ, x, y, theta):
    """Expected range (m) along heading `theta` from (x, y): fixed-step march.

    EXACTLY the algorithm the GPU weight kernel and the CPU oracle use
    (kernels.cu / reference_cpu.cpp): step RAY_STEP_M at a time, first sample
    at one step out, stop at the first occupied or off-map cell, else R_MAX_M.
    Python floats are doubles, the C++ side marches in float32 — the tiny
    discrepancy just looks like extra sensor noise (see ../data/README.md).
    """
    dx = math.cos(theta) * RAY_STEP_M
    dy = math.sin(theta) * RAY_STEP_M
    px, py, r = x, y, 0.0
    for _ in range(int(R_MAX_M / RAY_STEP_M)):     # 64 marching steps max
        px += dx
        py += dy
        r += RAY_STEP_M
        ix = math.floor(px / RES_M)
        iy = math.floor(py / RES_M)
        if ix < 0 or iy < 0 or ix >= MAP_W or iy >= MAP_H:
            return r                                # off the map = hit (border walls make this rare)
        if occ[iy][ix]:
            return r
    return R_MAX_M


def clearance_to_obstacles(occ, x, y):
    """Distance (m) from (x, y) to the nearest occupied cell RECTANGLE.

    Brute force over all occupied cells — fine at this scale (~1,300 cells x
    121 poses). Used only for the generator's self-check assertion below.
    """
    best = float("inf")
    for iy in range(MAP_H):
        for ix in range(MAP_W):
            if not occ[iy][ix]:
                continue
            # Closest point on the cell's axis-aligned rectangle to (x, y).
            cx = min(max(x, ix * RES_M), (ix + 1) * RES_M)
            cy = min(max(y, iy * RES_M), (iy + 1) * RES_M)
            best = min(best, math.hypot(x - cx, y - cy))
    return best


def command_sequence():
    """The scripted (v, w) twist commands: 4 x (straight leg + left quarter turn)."""
    cmds = []
    for _ in range(4):
        cmds += [(V_CMD, 0.0)] * LEG_STEPS
        cmds += [(V_CMD, W_TURN)] * TURN_STEPS
    return cmds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"noise RNG seed (default {DEFAULT_SEED}; the committed sample uses it)")
    ap.add_argument("--outdir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data" / "sample",
                    help="output directory (default ../data/sample)")
    args = ap.parse_args()

    rng = random.Random(args.seed)  # local RNG: cross-platform deterministic draws
    occ = build_map()
    cmds = command_sequence()

    # ---- integrate ground truth + synthesize measurements -------------------
    # The robot executes the commanded twists PERFECTLY (GT integrates the
    # commands); odometry is the noisy MEASUREMENT of each executed twist.
    # This puts all the estimation difficulty in the measurements — the
    # honest, simplest teaching setup (README "Limitations" owns it).
    x, y, th = START_POSE
    assert clearance_to_obstacles(occ, x, y) > MIN_CLEARANCE_M, "start pose too close to a wall"
    rows = []
    for step, (v, w) in enumerate(cmds):
        # Euler update in the EXACT order the GPU predict kernel uses:
        # position advances with the OLD heading, then the heading turns.
        x += v * math.cos(th) * DT_S
        y += v * math.sin(th) * DT_S
        th += w * DT_S                      # theta kept UNWRAPPED (grows to 2*pi over the loop)
        c = clearance_to_obstacles(occ, x, y)
        assert c > MIN_CLEARANCE_M, f"step {step}: path clearance {c:.3f} m too small"

        odo_v = v + rng.gauss(0.0, SIGMA_ODO_V)   # what the wheel encoders would report
        odo_w = w + rng.gauss(0.0, SIGMA_ODO_W)

        scan = []
        for b in range(NUM_BEAMS):
            ang = th + (-math.pi + b * (2.0 * math.pi / NUM_BEAMS))  # beam b, body-relative fan
            z = raycast(occ, x, y, ang) + rng.gauss(0.0, SIGMA_SCAN)
            scan.append(min(max(z, RAY_STEP_M), R_MAX_M))            # clamp to the sensor's physical span

        rows.append((step, (step + 1) * DT_S, x, y, th, odo_v, odo_w, scan))

    args.outdir.mkdir(parents=True, exist_ok=True)

    # ---- grid_map.txt --------------------------------------------------------
    # Written TOP ROW FIRST (iy = MAP_H-1) so the text file reads like a map
    # with +y up; the C++ loader flips (file line j -> iy = MAP_H-1-j).
    map_path = args.outdir / "grid_map.txt"
    lines = [
        "# grid_map.txt - SYNTHETIC occupancy grid map for project 04.01",
        "# generated by scripts/make_synthetic.py (map geometry is fixed constants - no RNG)",
        "# format: WIDTH,w / HEIGHT,h / RESOLUTION,meters-per-cell / MAP marker, then",
        "#         HEIGHT rows of WIDTH chars: '.' = free, '#' = occupied.",
        "# row order: TOP row first (iy = HEIGHT-1), so this file reads like a map with +y up;",
        "#            the loader flips (file line j -> iy = HEIGHT-1-j).",
        "# frame: world origin at the map's lower-left corner, x right, y up, right-handed, SI meters;",
        "#        cell (ix,iy) covers [ix*res,(ix+1)*res) x [iy*res,(iy+1)*res).",
        "# license: same as the repository (MIT) - fully synthetic, no external source",
        f"WIDTH,{MAP_W}",
        f"HEIGHT,{MAP_H}",
        f"RESOLUTION,{RES_M}",
        "MAP",
    ]
    for iy in range(MAP_H - 1, -1, -1):
        lines.append("".join("#" if occ[iy][ix] else "." for ix in range(MAP_W)))
    with open(map_path, "w", encoding="utf-8", newline="\n") as f:   # LF pinned
        f.write("\n".join(lines) + "\n")

    # ---- trajectory_scans.csv ------------------------------------------------
    log_path = args.outdir / "trajectory_scans.csv"
    zcols = ",".join(f"z{b:02d}_m" for b in range(NUM_BEAMS))
    lines = [
        "# trajectory_scans.csv - SYNTHETIC trajectory + noisy odometry + noisy range scans (project 04.01)",
        f"# generated by scripts/make_synthetic.py, seed {args.seed} (Python random.Random - deterministic draws)",
        "# INIT,x_m,y_m,theta_rad : the TRUE start pose (the filter initializes its particle cloud around it)",
        "# each STEP row: ground-truth pose AFTER applying that step's true twist; the NOISY odometry",
        "#   measurement of the twist (v + N(0,0.05) m/s, w + N(0,0.05) rad/s); and 16 NOISY ranges",
        "#   (true raycast + N(0,0.10) m, clamped to [0.125, 8.0]) measured at the post-step true pose.",
        "# beam b points at world angle gt_theta_rad + (-pi + b*pi/8): z00 rear, z04 right, z08 forward, z12 left.",
        "# units: m, m/s, rad, rad/s, s; frame: world (origin lower-left map corner, x right, y up); theta UNWRAPPED.",
        "# license: same as the repository (MIT) - fully synthetic, no external source",
        f"INIT,{START_POSE[0]:.6f},{START_POSE[1]:.6f},{START_POSE[2]:.6f}",
        f"STEP,t_s,gt_x_m,gt_y_m,gt_theta_rad,odo_v_ms,odo_w_rads,{zcols}",
    ]
    for (step, t, gx, gy, gth, ov, ow, scan) in rows:
        z = ",".join(f"{v:.4f}" for v in scan)
        lines.append(f"{step},{t:.1f},{gx:.6f},{gy:.6f},{gth:.6f},{ov:.6f},{ow:.6f},{z}")
    with open(log_path, "w", encoding="utf-8", newline="\n") as f:   # LF pinned
        f.write("\n".join(lines) + "\n")

    n_occ = sum(sum(row) for row in occ)
    print(f"[make_synthetic] wrote {map_path} ({map_path.stat().st_size} bytes: "
          f"{MAP_W}x{MAP_H} cells @ {RES_M} m, {n_occ} occupied) - labeled SYNTHETIC")
    print(f"[make_synthetic] wrote {log_path} ({log_path.stat().st_size} bytes: "
          f"{len(rows)} steps @ {1.0/DT_S:.0f} Hz, {NUM_BEAMS} beams/scan, seed {args.seed}) - labeled SYNTHETIC")
    if args.seed != DEFAULT_SEED:
        print("note: non-default seed - fine for experiments, do NOT commit these files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
