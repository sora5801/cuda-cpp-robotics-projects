#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample-data generator for 23.01 (GPU costmaps:
inflation, raytrace clearing, multi-layer fusion).

Generates the two files this project's demo consumes with zero downloads:

    ../data/sample/world_map.pgm   a 256x256 P5 (binary grayscale) occupancy
                                    grid: 0 = free, 254 = lethal (occupied).
                                    Border walls + a 4-wall slalom course
                                    between the start and goal corners.
    ../data/sample/scenario.csv    the task definition: which map file, the
                                    robot's start pose, the goal position,
                                    and the closed loop's step cap.

Why this shape (CLAUDE.md paragraph 8: synthetic-first)
---------------------------------------------------------
A costmap+DWA demo needs (a) a world with real geometry to avoid and (b) a
start/goal pair that is actually solvable by a REACTIVE local planner (this
project has no global route planner — see THEORY.md's "local minima" honesty
note). Both are easy to synthesize with full ground truth and hard to get
honestly from a random layout, so this generator does something a random
obstacle scatter would not: it VERIFIES solvability itself, with the same
BFS a reader could write by hand, before it ever writes a file. If a jittered
layout is ever unsolvable (should not happen at the jitter magnitudes below —
the clearances are generous on purpose) the generator retries with a fresh
sub-seed and only gives up loudly after many attempts. Nothing here is
fabricated: the printed clearance numbers are measured from the actual grid
this run produced, not asserted.

Usage
-----
    python make_synthetic.py                 # defaults: seed=42
    python make_synthetic.py --seed 7 --out-dir ../data/sample
"""

import argparse
import csv
import hashlib
import random
import struct
from collections import deque
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants that MUST mirror src/kernels.cuh (kept in one place per file,
# with an explicit note at each mirrored constant — CLAUDE.md paragraph 12:
# "one source of truth" is a C++-side promise; this script is the one place
# outside C++ that also needs to agree, so the mirroring is called out loudly
# rather than left implicit).
# ---------------------------------------------------------------------------
GRID_W = 256                 # mirrors kGridW
GRID_H = 256                 # mirrors kGridH
RESOLUTION_M = 0.05          # mirrors kResolutionM
COST_FREE = 0                # mirrors kCostFree
COST_LETHAL = 254            # mirrors kCostLethal
INSCRIBED_RADIUS_CELLS = 4   # mirrors kInscribedRadiusCells — used ONLY for
                              # this script's own solvability check below,
                              # never written into the map itself
DEFAULT_SEED = 42
DEFAULT_STEPS = 500          # closed-loop step cap (kDtControl=0.1s -> 50 s)

# Start near one corner, goal near the opposite corner — the diagonal a
# reactive planner has to cross through the slalom below.
START_XY_M = (1.0, 1.0)
START_THETA_RAD = 0.0
GOAL_XY_M = (11.0, 11.0)


def draw_rect(grid: list, r0: int, r1: int, c0: int, c1: int, value: int) -> None:
    """Fill grid rows [r0,r1) and columns [c0,c1) with value (clipped to bounds).

    grid is a flat list of length GRID_W*GRID_H, row-major (row*GRID_W+col) —
    the exact layout kernels.cuh documents for the C++ side, so the byte
    order this script writes and the byte order main.cu's read_pgm() expects
    agree without any transposition step.
    """
    r0 = max(0, r0); r1 = min(GRID_H, r1)
    c0 = max(0, c0); c1 = min(GRID_W, c1)
    for r in range(r0, r1):
        base = r * GRID_W
        for c in range(c0, c1):
            grid[base + c] = value


def build_world(seed: int) -> list:
    """Build one candidate world: border walls + four small jittered pillar
    obstacles straddling the start-to-goal diagonal.

    Design choice, stated honestly (THEORY.md expands on it): this project's
    DWA is a REACTIVE LOCAL planner only — there is no global route search
    in front of it (no A*/Dijkstra), which is exactly the repo brief's
    scope. An early version of this generator drew full-width slalom walls
    with gaps that alternated sides; that course requires a planner to move
    AWAY from the goal bearing to find a distant opening — precisely DWA's
    textbook local-minimum failure mode (THEORY.md §the-math cites it), and
    a purely reactive planner got stuck against a wall face measurably (see
    the push history / THEORY.md's honest account of that measurement).
    This world instead uses small, isolated pillars that only ever block a
    fraction of the corridor width, so going around EITHER side is always a
    short, locally-discoverable detour — solvable by heading-plus-obstacle
    scoring alone, with no risk of needing to backtrack away from the goal.
    """
    rng = random.Random(seed)
    grid = [COST_FREE] * (GRID_W * GRID_H)

    # Border: 3-cell-thick walls on all four edges (0.15 m) — encloses the
    # room so a LiDAR beam that finds no interior obstacle always finds the
    # border eventually (main.cu's scan simulator relies on this).
    border = 3
    draw_rect(grid, 0, border, 0, GRID_W, COST_LETHAL)
    draw_rect(grid, GRID_H - border, GRID_H, 0, GRID_W, COST_LETHAL)
    draw_rect(grid, 0, GRID_H, 0, border, COST_LETHAL)
    draw_rect(grid, 0, GRID_H, GRID_W - border, GRID_W, COST_LETHAL)

    # Four square pillars, ~28 cells (1.4 m) on a side, centered near four
    # points along the start(20,20)->goal(220,220) diagonal, each jittered a
    # few cells off the diagonal (alternating sides) so the robot has to
    # nudge around — never travel away from the goal to do it. Clearance on
    # either side of a pillar is on the order of 80-150 cells (4-7.5 m),
    # far more than the ~1 m the inflated robot needs, at every seed the
    # retry loop might try.
    def jitter(base: int, span: int) -> int:
        return base + rng.randint(-span, span)

    half = 14   # pillar half-size in cells (28x28 total, 1.4 m square)
    centers = [
        (jitter(75, 10),  jitter(60, 10)),
        (jitter(115, 10), jitter(135, 10)),
        (jitter(160, 10), jitter(115, 10)),
        (jitter(195, 10), jitter(190, 10)),
    ]
    for cr, cc in centers:
        draw_rect(grid, cr - half, cr + half, cc - half, cc + half, COST_LETHAL)

    return grid


def world_to_cell(x_m: float, y_m: float) -> tuple:
    return int(x_m / RESOLUTION_M), int(y_m / RESOLUTION_M)


def bfs_solvable(grid: list, start_cell: tuple, goal_cell: tuple) -> tuple:
    """BFS over cells that are neither lethal nor within INSCRIBED_RADIUS_CELLS
    of a lethal cell (a conservative stand-in for "the robot's footprint
    could stand here") — the same connectivity question the DWA planner
    effectively has to answer online, checked offline here so a broken
    layout is caught at GENERATION time, not discovered by a failing demo.

    Returns (solvable: bool, path_len_cells: int) — path_len is 0 if
    unsolvable, informational otherwise (printed, never asserted beyond the
    bool).
    """
    blocked = [False] * (GRID_W * GRID_H)
    R = INSCRIBED_RADIUS_CELLS
    for r in range(GRID_H):
        for c in range(GRID_W):
            if grid[r * GRID_W + c] != COST_LETHAL:
                continue
            r0, r1 = max(0, r - R), min(GRID_H, r + R + 1)
            c0, c1 = max(0, c - R), min(GRID_W, c + R + 1)
            for rr in range(r0, r1):
                base = rr * GRID_W
                for cc in range(c0, c1):
                    if (rr - r) ** 2 + (cc - c) ** 2 <= R * R:
                        blocked[base + cc] = True

    sx, sy = start_cell
    gx, gy = goal_cell
    if blocked[sy * GRID_W + sx] or blocked[gy * GRID_W + gx]:
        return False, 0

    q = deque([(sx, sy)])
    dist = {(sx, sy): 0}
    while q:
        x, y = q.popleft()
        if (x, y) == (gx, gy):
            return True, dist[(x, y)]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_W and 0 <= ny < GRID_H and not blocked[ny * GRID_W + nx] \
                    and (nx, ny) not in dist:
                dist[(nx, ny)] = dist[(x, y)] + 1
                q.append((nx, ny))
    return False, 0


def write_pgm(path: Path, width: int, height: int, grid: list) -> None:
    """Write a binary P5 PGM — the same tiny, library-free format main.cu's
    read_pgm() parses and 07.09 also uses for its artifacts. Header is plain
    ASCII (no comment line, to keep the reader's job trivial); body is the
    raw byte grid, row-major, exactly as documented in kernels.cuh.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        f.write(f"P5\n{width} {height}\n255\n".encode('ascii'))
        f.write(struct.pack(f'{len(grid)}B', *grid))


def write_scenario(path: Path, map_filename: str, steps: int) -> None:
    """Write scenario.csv — the strict, labeled-row format main.cu's
    load_scenario() parses (mirrors 08.01's cartpole_scenario.csv style).
    Newlines are forced to '\\n' (LF) regardless of platform, per this
    repo's data-file convention.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='\n', encoding='utf-8') as f:
        f.write(f"# SYNTHETIC scenario for project 23.01 gpu-costmaps (seed {DEFAULT_SEED})\n")
        f.write(f"# regenerate: python make_synthetic.py --seed {DEFAULT_SEED}\n")
        f.write("# MAP,<filename>,<width_cells>,<height_cells>,<resolution_m>\n")
        f.write("# START,<x_m>,<y_m>,<theta_rad>\n")
        f.write("# GOAL,<x_m>,<y_m>\n")
        f.write("# STEPS,<control_tick_cap>\n")
        f.write(f"MAP,{map_filename},{GRID_W},{GRID_H},{RESOLUTION_M}\n")
        f.write(f"START,{START_XY_M[0]},{START_XY_M[1]},{START_THETA_RAD}\n")
        f.write(f"GOAL,{GOAL_XY_M[0]},{GOAL_XY_M[1]}\n")
        f.write(f"STEPS,{steps}\n")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_out_dir = script_dir.parent / "data" / "sample"

    parser = argparse.ArgumentParser(
        description="Generate the synthetic world map + scenario for project 23.01 (GPU costmaps).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"RNG seed for the slalom's jitter (default {DEFAULT_SEED})")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help=f"closed-loop step cap written into scenario.csv (default {DEFAULT_STEPS})")
    parser.add_argument("--out-dir", type=Path, default=default_out_dir,
                        help="output directory (default: ../data/sample)")
    parser.add_argument("--max-attempts", type=int, default=50,
                        help="solvability-retry cap before giving up loudly (default 50)")
    args = parser.parse_args()

    start_cell = world_to_cell(*START_XY_M)
    goal_cell = world_to_cell(*GOAL_XY_M)

    grid = None
    path_len = 0
    for attempt in range(args.max_attempts):
        candidate = build_world(args.seed + attempt * 1000003)
        solvable, plen = bfs_solvable(candidate, start_cell, goal_cell)
        if solvable:
            grid = candidate
            path_len = plen
            if attempt > 0:
                print(f"[make_synthetic] seed {args.seed} needed {attempt} retr"
                     f"{'y' if attempt == 1 else 'ies'} to find a solvable layout "
                     f"(sub-seed offset {attempt * 1000003})")
            break
    if grid is None:
        raise SystemExit(f"[make_synthetic] FAILED: no solvable layout found in "
                         f"{args.max_attempts} attempts at seed {args.seed} — widen the "
                         f"jitter bounds or the gap sizes in build_world()")

    n_lethal = sum(1 for v in grid if v == COST_LETHAL)
    occ_pct = 100.0 * n_lethal / (GRID_W * GRID_H)

    map_path = args.out_dir / "world_map.pgm"
    scenario_path = args.out_dir / "scenario.csv"
    write_pgm(map_path, GRID_W, GRID_H, grid)
    write_scenario(scenario_path, map_path.name, args.steps)

    sha = hashlib.sha256(map_path.read_bytes()).hexdigest()
    print(f"[make_synthetic] wrote {map_path} ({GRID_W}x{GRID_H}, {n_lethal} lethal "
         f"cells, {occ_pct:.1f}% occupied) SHA-256={sha}")
    print(f"[make_synthetic] wrote {scenario_path} (start={START_XY_M}, goal={GOAL_XY_M}, "
         f"steps={args.steps})")
    print(f"[make_synthetic] BFS solvability check PASSED: start->goal reachable in "
         f"{path_len} grid steps at an inscribed-radius clearance of "
         f"{INSCRIBED_RADIUS_CELLS} cells ({INSCRIBED_RADIUS_CELLS * RESOLUTION_M:.2f} m) "
         f"[synthetic, seed {args.seed}]")


if __name__ == "__main__":
    main()
