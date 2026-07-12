#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample-data generator for 02.19
(PointPillars/CenterPoint voxelization + scatter kernels feeding TensorRT).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
-----------------------------------------------------------------
This project needs a small BEV scene with KNOWN object positions (so the
detection_closure gate in main.cu has ground truth to check against) plus a
DELIBERATE stress case that overflows one pillar's point cap (so the
cap_truncation determinism gate has something to truncate). Both are exact
by construction when synthesized, and neither exists in any off-the-shelf
dataset in this shape — so, per repo policy, everything here is synthetic,
labeled synthetic everywhere it appears, and reproducible bit-for-bit from a
fixed seed.

The point-stream layout (a data-layout CONTRACT with main.cu — recorded in
scene_meta.csv, never re-derived by guesswork)
-----------------------------------------------------------------------------
Points are written in exactly this order, back to back:
    [0, n_ground)                                 -- ground-plane returns
    [n_ground, n_ground + n_cars_total)            -- K cars, back to back,
                                                       points_per_car each
    [n_ground+n_cars_total, .. + n_clutter)        -- low, isolated clutter
    [.. , .. + n_capstress)                        -- the cap-stress pillar
n_capstress = 60 > kMaxPointsPerPillar (32, kernels.cuh) is DELIBERATE: every
one of those 60 points shares the SAME pillar key by construction, so
whichever binning method main.cu runs, some points must be dropped — the
cap_truncation gate studies exactly WHICH ones (see kernels.cuh's file
header "TWO BINNING METHODS" section).

File format: points.bin is the plain KITTI/PointPillars "raw velodyne.bin"
layout used by the real reference implementations this project teaches
toward (THEORY.md "Where this sits in the real world" names them) — no
header at all, just N*4 float32 (x,y,z,intensity) back to back; N is
recovered from the file size (N = bytes/16). Using the SAME on-disk shape
as the real tool this project is a teaching stand-in for is a deliberate,
free bit of authenticity.

Usage
-----
    python make_synthetic.py                 # writes the committed sample
    python make_synthetic.py --out DIR        # experiments; do not commit
"""

import argparse
import struct
import sys
from pathlib import Path

# ===========================================================================
# Deterministic RNG: xorshift32 (stdlib-only — repo convention, CLAUDE.md
# paragraph 12). Copied from 02.01's generator (cited: ../../02.01-voxel-
# grid-downsampling-with-gpu-spatial-hashing/scripts/make_synthetic.py) —
# the SAME three-shift/three-XOR core this repo's CUDA device code uses.
# ===========================================================================
class Xorshift32:
    def __init__(self, seed: int):
        s = seed & 0xFFFFFFFF
        if s == 0:
            s = 1  # xorshift32 is degenerate (stays 0 forever) at seed 0
        self.state = s

    def next_u32(self) -> int:
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17)
        x ^= (x << 5) & 0xFFFFFFFF
        x &= 0xFFFFFFFF
        self.state = x
        return x

    def uniform01(self) -> float:
        """(0,1], never exactly 0 — mirrors kernels.cu's uniform01()."""
        return (self.next_u32() >> 8) * (1.0 / 16777216.0) + (0.5 / 16777216.0)

    def uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.uniform01()


DEFAULT_SEED = 42  # repo convention (CLAUDE.md paragraph 12)

# ===========================================================================
# BEV grid geometry — MUST match kernels.cuh exactly (main.cu asserts a few
# of these at load time). Duplicated here (Python cannot #include a .cuh)
# rather than parsed from it — a documented, single point of truth risk the
# repo accepts elsewhere too (see 02.01's LEAF_M comment for the same trade).
# ===========================================================================
PILLAR_SIZE_M = 0.4
GRID_NX = 200
GRID_NY = 200
X_MIN = -40.0
Y_MIN = -40.0
Z_MIN = -3.0
Z_MAX = 5.0
MAX_POINTS_PER_PILLAR = 32

# PFN-lite shape (kernels.cuh): D input features, kPfnLinOut output channels.
NUM_POINT_FEATURES = 9
PFN_LIN_OUT = 4

# ---------------------------------------------------------------------------
# Six illustrative "parking-lot" car positions (BEV center, meters). Hand-
# placed (not randomly sampled) so separation is guaranteed by construction:
# every pair is >= 24 m apart, far beyond the car footprint (4.2 x 1.8 m)
# plus the NMS suppression radius (kNmsRadiusPillars=3 pillars = 1.2 m,
# kernels.cuh) — no two objects can ever merge into one detection, and the
# grid's 80x80 m window comfortably contains all six with margin to spare.
# All cars share ONE orientation (length along +x) — this project teaches
# pillarization/scatter, not box-orientation regression (README "Limitations").
# ---------------------------------------------------------------------------
CAR_CENTERS = [(20.0, 20.0), (-20.0, 20.0), (20.0, -20.0),
               (-20.0, -20.0), (0.0, 28.0), (0.0, -28.0)]
CAR_LENGTH_M = 4.2   # along x
CAR_WIDTH_M = 1.8    # along y
CAR_HEIGHT_M = 1.5   # roof height above ground (z=0..CAR_HEIGHT_M)
POINTS_PER_CAR = 220

GROUND_N = 6000
GROUND_Z_NOISE_M = 0.03

CLUTTER_N = 150
CLUTTER_Z_MAX_M = 0.35          # low: never tall enough to look like a car
CLUTTER_EXCLUDE_CAR_RADIUS_M = 6.0
CLUTTER_EXCLUDE_CAPSTRESS_RADIUS_M = 3.0

# The deliberate cap-truncation stress pillar: 60 points (> the 32 cap) all
# landing in ONE pillar, isolated (its neighbors are empty) so the head's
# spatial-smoothing conv (kernels.cuh/THEORY.md) does NOT mistake a single
# dense pillar for a spatially-clustered object — see detection_closure's
# comment in main.cu for why an isolated spike must NOT become a false peak.
CAPSTRESS_CENTER = (-30.2, -30.2)   # deliberately mid-cell (not on a pillar boundary, unlike -30.0
                                    # which sits exactly on a grid line at this pillar size/origin)
CAPSTRESS_N = 60
CAPSTRESS_Z_RANGE_M = (0.2, 0.5)   # low & narrow: not "tall" like a car


def pillar_key_of(x: float, y: float) -> int:
    """Python twin of kernels.cuh's pillar_key_of — used only to VALIDATE
    the generator's own placements (e.g. that CAPSTRESS_CENTER's 60 points
    truly land in one pillar), never trusted as ground truth on its own."""
    ix = int((x - X_MIN) // PILLAR_SIZE_M)
    iy = int((y - Y_MIN) // PILLAR_SIZE_M)
    if not (0 <= ix < GRID_NX and 0 <= iy < GRID_NY):
        return -1
    return iy * GRID_NX + ix


def in_any_car_footprint(x: float, y: float) -> bool:
    for (cx, cy) in CAR_CENTERS:
        if abs(x - cx) <= CAR_LENGTH_M / 2.0 and abs(y - cy) <= CAR_WIDTH_M / 2.0:
            return True
    return False


def build_ground(rng: Xorshift32):
    """Sparse ground-plane returns across the BEV window, EXCLUDING points
    that would fall under a car's footprint (a real LiDAR cannot see through
    a car — this is a physically-honest occlusion, not arbitrary thinning;
    see 02.01's make_synthetic.py for the same "no ceiling -> honest
    dropout" philosophy applied here to "no under-car ground")."""
    pts = []
    half_x = GRID_NX * PILLAR_SIZE_M / 2.0 - 0.5   # stay inside the grid with margin
    half_y = GRID_NY * PILLAR_SIZE_M / 2.0 - 0.5
    tries = 0
    while len(pts) < GROUND_N and tries < GROUND_N * 4:
        tries += 1
        x = rng.uniform(-half_x, half_x)
        y = rng.uniform(-half_y, half_y)
        if in_any_car_footprint(x, y):
            continue
        z = rng.uniform(-GROUND_Z_NOISE_M, GROUND_Z_NOISE_M)
        intensity = rng.uniform(0.05, 0.25)   # ground: low, fairly uniform reflectance
        pts.append((x, y, z, intensity))
    return pts


def build_car(rng: Xorshift32, cx: float, cy: float):
    """Sample POINTS_PER_CAR points on the 5 exposed faces of an axis-
    aligned box (roof + 4 sides; no bottom face — a real LiDAR never sees a
    car's underside from a ground-mounted sensor). ~40% roof, ~60% split
    across the 4 side walls — the roof is the single largest flat return,
    matching real car LiDAR returns' roof-dominant point density."""
    pts = []
    hl, hw, h = CAR_LENGTH_M / 2.0, CAR_WIDTH_M / 2.0, CAR_HEIGHT_M
    n_roof = int(POINTS_PER_CAR * 0.4)
    n_side_each = (POINTS_PER_CAR - n_roof) // 4
    for _ in range(n_roof):
        x = cx + rng.uniform(-hl, hl)
        y = cy + rng.uniform(-hw, hw)
        z = h + rng.uniform(-0.02, 0.02)
        intensity = rng.uniform(0.4, 0.9)   # painted metal: brighter than ground
        pts.append((x, y, z, intensity))
    # Four side walls: fix one horizontal axis at the car's edge, vary the
    # other horizontal axis along the wall and z from 0 (ground contact,
    # excluded — z>0.05 keeps a visible sliver above the ground plane) to h.
    walls = [
        (lambda t: (cx - hl, cy + t)),   # -x face
        (lambda t: (cx + hl, cy + t)),   # +x face
        (lambda t: (cx + t,  cy - hw)),  # -y face
        (lambda t: (cx + t,  cy + hw)),  # +y face
    ]
    span = [hw, hw, hl, hl]
    for wall_fn, s in zip(walls, span):
        for _ in range(n_side_each):
            t = rng.uniform(-s, s)
            x, y = wall_fn(t)
            z = rng.uniform(0.05, h)
            intensity = rng.uniform(0.3, 0.7)
            pts.append((x, y, z, intensity))
    return pts


def build_clutter(rng: Xorshift32):
    """Low, spatially-isolated single-return clutter — must NOT trigger the
    head (detection_closure's "zero false peaks on clutter" requirement).
    Rejection-sampled away from every car and the cap-stress pillar so no
    clutter point can accidentally pad a real object's or the stress
    pillar's point count."""
    pts = []
    half_x = GRID_NX * PILLAR_SIZE_M / 2.0 - 0.5
    half_y = GRID_NY * PILLAR_SIZE_M / 2.0 - 0.5
    tries = 0
    while len(pts) < CLUTTER_N and tries < CLUTTER_N * 40:
        tries += 1
        x = rng.uniform(-half_x, half_x)
        y = rng.uniform(-half_y, half_y)
        if any((x - cx) ** 2 + (y - cy) ** 2 < CLUTTER_EXCLUDE_CAR_RADIUS_M ** 2 for (cx, cy) in CAR_CENTERS):
            continue
        sx, sy = CAPSTRESS_CENTER
        if (x - sx) ** 2 + (y - sy) ** 2 < CLUTTER_EXCLUDE_CAPSTRESS_RADIUS_M ** 2:
            continue
        z = rng.uniform(0.02, CLUTTER_Z_MAX_M)
        intensity = rng.uniform(0.1, 0.5)
        pts.append((x, y, z, intensity))
    return pts


def build_capstress(rng: Xorshift32):
    """CAPSTRESS_N points jittered inside ONE pillar cell — the deliberate
    cap-overflow stress case (module docstring)."""
    cx, cy = CAPSTRESS_CENTER
    half = PILLAR_SIZE_M * 0.45   # stay safely inside the single target cell
    pts = []
    for _ in range(CAPSTRESS_N):
        x = cx + rng.uniform(-half, half)
        y = cy + rng.uniform(-half, half)
        z = rng.uniform(*CAPSTRESS_Z_RANGE_M)
        intensity = rng.uniform(0.2, 0.6)
        pts.append((x, y, z, intensity))
    return pts


def build_pfn_weights(rng: Xorshift32):
    """PFN-lite fixed weights (kernels.cuh: linear D=9 -> PFN_LIN_OUT=4,
    ReLU, then max-pooled over a pillar's points). NOT trained — a
    deterministic, small-magnitude draw from the same RNG stream (continued
    sequentially after the scene, so one seed reproduces EVERYTHING in this
    file bit-for-bit).

    Per-dimension weight RANGE is deliberately uneven (documented, a
    generator design choice, not a claim about real trained weights): raw
    x,y (feature dims 0,1) span up to +-40 m, so giving them the same weight
    range as the other 7 (mostly sub-2-m) dims would let two dims dominate
    every channel's activation and wash out everything else — the same
    "absolute position swamps local shape" problem the offset features
    (xc,yc,zc,xp,yp) exist to counteract (THEORY.md "The math"). Small
    weights on dims 0,1 keep this teaching demo's PFN output well-
    conditioned; a REAL trained PFN would learn whatever weighting the data
    rewards, position included.
    """
    dim_ranges = [0.01, 0.01, 0.20, 0.20, 0.30, 0.30, 0.30, 0.30, 0.30]  # per input dim, +-range
    assert len(dim_ranges) == NUM_POINT_FEATURES
    w = []
    for _ch in range(PFN_LIN_OUT):
        row = [rng.uniform(-r, r) for r in dim_ranges]
        w.append(row)
    b = [rng.uniform(-0.1, 0.1) for _ in range(PFN_LIN_OUT)]
    return w, b


def write_points_bin(out_path: Path, all_points):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    flat = []
    for (x, y, z, i) in all_points:
        flat.extend((x, y, z, i))
    with out_path.open('wb') as f:
        f.write(struct.pack(f'<{len(flat)}f', *flat))


def write_truths_csv(out_path: Path):
    with out_path.open('w') as f:
        f.write("# object_truths.csv -- ground-truth BEV object centers, project 02.19\n")
        f.write("# id,cx_m,cy_m,length_m,width_m,height_m,category\n")
        for i, (cx, cy) in enumerate(CAR_CENTERS):
            f.write(f"{i},{cx:.3f},{cy:.3f},{CAR_LENGTH_M:.2f},{CAR_WIDTH_M:.2f},{CAR_HEIGHT_M:.2f},car\n")


def write_scene_meta_csv(out_path: Path, n_ground, n_cars_total, n_clutter, n_capstress,
                         capstress_start_index, capstress_pillar_key):
    n_total = n_ground + n_cars_total + n_clutter + n_capstress
    with out_path.open('w') as f:
        f.write("# scene_meta.csv -- the point-stream layout CONTRACT main.cu reads verbatim, project 02.19\n")
        f.write("# key,value\n")
        f.write(f"n_total,{n_total}\n")
        f.write(f"n_ground,{n_ground}\n")
        f.write(f"n_cars_total,{n_cars_total}\n")
        f.write(f"n_cars,{len(CAR_CENTERS)}\n")
        f.write(f"points_per_car,{POINTS_PER_CAR}\n")
        f.write(f"n_clutter,{n_clutter}\n")
        f.write(f"n_capstress,{n_capstress}\n")
        f.write(f"capstress_start_index,{capstress_start_index}\n")
        f.write(f"capstress_pillar_key,{capstress_pillar_key}\n")
        f.write(f"capstress_center_x_m,{CAPSTRESS_CENTER[0]:.3f}\n")
        f.write(f"capstress_center_y_m,{CAPSTRESS_CENTER[1]:.3f}\n")


def write_pfn_weights_csv(out_path: Path, w, b):
    with out_path.open('w') as f:
        f.write("# pfn_lite_weights.csv -- FIXED (not trained), seed 42, project 02.19\n")
        f.write("# row = one output channel: w0,w1,...,w8,bias  (9 input-feature weights + 1 bias)\n")
        for row, bias in zip(w, b):
            f.write(",".join(f"{v:.8f}" for v in row) + f",{bias:.8f}\n")


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    sample_dir = script_dir.parent / 'data' / 'sample'

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help=f'xorshift32 seed for byte-identical reproducibility (default {DEFAULT_SEED})')
    parser.add_argument('--out', type=Path, default=sample_dir,
                        help='output directory (default: ../data/sample/)')
    args = parser.parse_args()

    rng = Xorshift32(args.seed)

    ground = build_ground(rng)
    cars = []
    for (cx, cy) in CAR_CENTERS:
        cars.extend(build_car(rng, cx, cy))
    clutter = build_clutter(rng)
    capstress = build_capstress(rng)
    lin_w, lin_b = build_pfn_weights(rng)

    # Validate the cap-stress placement lands in exactly ONE pillar (a
    # generator self-check, not part of the demo's own verification gates).
    keys = {pillar_key_of(x, y) for (x, y, _z, _i) in capstress}
    assert len(keys) == 1 and next(iter(keys)) >= 0, \
        "cap-stress points must land in exactly one in-window pillar"
    capstress_key = next(iter(keys))

    all_points = ground + cars + clutter + capstress
    n_ground, n_cars_total, n_clutter, n_capstress = len(ground), len(cars), len(clutter), len(capstress)
    capstress_start_index = n_ground + n_cars_total + n_clutter

    args.out.mkdir(parents=True, exist_ok=True)
    write_points_bin(args.out / 'points.bin', all_points)
    write_truths_csv(args.out / 'object_truths.csv')
    write_scene_meta_csv(args.out / 'scene_meta.csv', n_ground, n_cars_total, n_clutter, n_capstress,
                         capstress_start_index, capstress_key)
    write_pfn_weights_csv(args.out / 'pfn_lite_weights.csv', lin_w, lin_b)

    n_total = len(all_points)
    print(f"[make_synthetic] SYNTHETIC BEV scene (seed={args.seed}): "
          f"{n_ground} ground + {n_cars_total} car ({len(CAR_CENTERS)} x {POINTS_PER_CAR}) "
          f"+ {n_clutter} clutter + {n_capstress} cap-stress = {n_total} points total")
    print(f"[make_synthetic] cap-stress pillar key={capstress_key} at BEV {CAPSTRESS_CENTER} "
          f"({n_capstress} points > cap {MAX_POINTS_PER_PILLAR}, point indices "
          f"[{capstress_start_index}, {capstress_start_index + n_capstress}))")
    print(f"[make_synthetic] wrote {args.out / 'points.bin'} ({n_total * 16} bytes, "
          f"KITTI-style raw float32 x,y,z,intensity, labeled SYNTHETIC)")
    print(f"[make_synthetic] wrote {args.out / 'object_truths.csv'} ({len(CAR_CENTERS)} objects)")
    print(f"[make_synthetic] wrote {args.out / 'pfn_lite_weights.csv'} "
          f"({PFN_LIN_OUT} x {NUM_POINT_FEATURES} + {PFN_LIN_OUT} bias, FIXED not trained)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
