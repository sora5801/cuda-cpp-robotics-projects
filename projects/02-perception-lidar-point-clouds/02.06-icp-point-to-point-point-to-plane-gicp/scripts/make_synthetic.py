#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample-data generator for project 02.06
   (ICP: point-to-point -> point-to-plane -> GICP, all batched).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
-----------------------------------------------------------------
ICP needs two clouds AND a known ground-truth transform between them to be
verifiable at all -- a real LiDAR recording never comes with an exact answer
key. So this project's data is synthetic BY NECESSITY, not just by repo
default: we build a small structured "room" (two walls meeting in a corner,
a floor, and a box -- deliberately WALL-DOMINATED so point-to-plane ICP's
faster convergence on planar scenes, the thing this project teaches, is
actually exercised), sample two independent point clouds off its surfaces
(mimicking two different LiDAR scans of the same static scene), apply a
KNOWN rigid transform plus independent sensor noise to the second cloud,
and hand ICP the job of recovering that transform from the points alone.

What gets written (see ../data/README.md for the byte-exact format spec)
--------------------------------------------------------------------------
  pair0_source.bin, pair0_target.bin   the MAIN pair,  ~30000 points/cloud
  pair1_source.bin, pair1_target.bin   a SMALL 2nd pair, ~5000 points/cloud
  pairs_meta.csv                       ground-truth quaternion + translation
                                        + noise sigma + point counts, one
                                        row per pair (main.cu's loader)

Every draw comes from Python's std-lib `random.Random`, seeded, so the
committed files are byte-for-byte reproducible from this script alone --
no numpy, no external data, per this repo's Python-stdlib-only convention.

Usage
-----
    python make_synthetic.py                      # defaults: seed=42, writes ../data/sample/
    python make_synthetic.py --seed 7 --out DIR
"""

import argparse
import csv
import math
import random
import struct
from pathlib import Path

# ---------------------------------------------------------------------------
# Scene geometry (meters, SI, right-handed, matching CLAUDE.md paragraph 12).
# A room corner: a floor (z=0) and two walls meeting at x=-3, y=-3, plus a
# small box sitting on the floor away from the corner. All primitives are
# axis-aligned rectangles/boxes so uniform-on-surface sampling is a plain
# uniform draw over two parameters -- no rejection sampling needed.
# ---------------------------------------------------------------------------
ROOM_HALF = 3.0          # floor/walls span [-3, 3] in their free axes (6 m x 6 m room)
WALL_HEIGHT = 2.5         # m
BOX_CENTER = (1.0, 1.0, 0.4)     # m, sits on the floor away from the corner
BOX_HALF = (0.3, 0.3, 0.4)       # m, half-extents (box is 0.6 x 0.6 x 0.8 m)

# Per-point isotropic Gaussian position noise (m). CLAUDE.md-mandated
# scoping choice, documented honestly in THEORY.md "Numerical
# considerations": a real LiDAR's range noise is 1-D, along the beam: we
# simplify to isotropic 3-D jitter because this script builds a structured
# scene directly (no per-beam raycasting -- that is project 11.01's job,
# and its tiny outputs are what a real Chain-A pipeline would feed here).
NOISE_SIGMA_M = 0.005

# Surface areas (m^2) of each primitive -- used ONLY to allocate the point
# budget proportionally, so point DENSITY is roughly uniform across the
# whole scene (a learner plotting the cloud sees an even scatter, not a
# denser floor and a sparse box).
AREA_FLOOR = (2.0 * ROOM_HALF) ** 2                        # 36.0
AREA_WALL = (2.0 * ROOM_HALF) * WALL_HEIGHT                # 15.0 each
AREA_BOX_SIDE = (2.0 * BOX_HALF[1]) * (2.0 * BOX_HALF[2])  # +x/-x faces: 0.48 each
AREA_BOX_FRONT = (2.0 * BOX_HALF[0]) * (2.0 * BOX_HALF[2]) # +y/-y faces: 0.48 each
AREA_BOX_TOP = (2.0 * BOX_HALF[0]) * (2.0 * BOX_HALF[1])   # +z face:    0.36
AREA_BOX_TOTAL = 2 * AREA_BOX_SIDE + 2 * AREA_BOX_FRONT + AREA_BOX_TOP  # 2.28 (bottom omitted: it rests on the floor, unscannable)
AREA_TOTAL = AREA_FLOOR + 2 * AREA_WALL + AREA_BOX_TOTAL                # 68.28

# The two demo pairs. Rotation 5-10 deg + translation 0.2-0.4 m per the
# project brief -- these are GROUND TRUTH values baked into pairs_meta.csv;
# ICP's job is to recover them from the points alone.
#   pair0: the MAIN pair (~30000 pts/cloud), simple yaw about +z.
#   pair1: a SMALL second pair (~5000 pts/cloud), a tilted rotation axis --
#          proves the pipeline is not accidentally specialized to pure yaw.
PAIRS = [
    {
        "name": "pair0", "n_total": 30000, "seed_offset": 0,
        "rot_axis": (0.0, 0.0, 1.0), "rot_deg": 7.0,
        "t_m": (0.25, -0.18, 0.06),   # |t| = 0.314 m
    },
    {
        "name": "pair1", "n_total": 5000, "seed_offset": 100,
        "rot_axis": (0.2, 0.1, 1.0), "rot_deg": 9.0,
        "t_m": (-0.20, 0.32, -0.05),  # |t| = 0.381 m
    },
]


# ---------------------------------------------------------------------------
# Per-primitive uniform-on-surface samplers. Each returns a list of (x,y,z)
# tuples. Uniform draws over the two free parameters of an axis-aligned
# rectangle ARE uniform-on-area for a flat rectangle (no distortion to
# correct for, unlike e.g. a sphere) -- the simplest correct sampler.
# ---------------------------------------------------------------------------
def sample_floor(rng: random.Random, n: int) -> list:
    return [(rng.uniform(-ROOM_HALF, ROOM_HALF), rng.uniform(-ROOM_HALF, ROOM_HALF), 0.0)
            for _ in range(n)]


def sample_wall_a(rng: random.Random, n: int) -> list:
    # Plane x = -ROOM_HALF, spanning y and z -- the wall whose OUTWARD
    # (into-room) normal is +x.
    return [(-ROOM_HALF, rng.uniform(-ROOM_HALF, ROOM_HALF), rng.uniform(0.0, WALL_HEIGHT))
            for _ in range(n)]


def sample_wall_b(rng: random.Random, n: int) -> list:
    # Plane y = -ROOM_HALF, spanning x and z -- normal +y. Together with
    # wall A and the floor, three MUTUALLY ORTHOGONAL planes meet at the
    # corner (-3,-3,0) -- the geometric reason this scene fully constrains
    # all 6 DOF of a rigid transform (THEORY.md "The algorithm" explains
    # why a single infinite plane cannot: it leaves 3 DOF unconstrained).
    return [(rng.uniform(-ROOM_HALF, ROOM_HALF), -ROOM_HALF, rng.uniform(0.0, WALL_HEIGHT))
            for _ in range(n)]


def sample_box(rng: random.Random, n: int) -> list:
    # Five faces (bottom omitted -- it rests on the floor, unscannable),
    # chosen with probability proportional to their area so point density
    # matches the rest of the scene.
    cx, cy, cz = BOX_CENTER
    hx, hy, hz = BOX_HALF
    faces = [
        ("+x", AREA_BOX_SIDE), ("-x", AREA_BOX_SIDE),
        ("+y", AREA_BOX_FRONT), ("-y", AREA_BOX_FRONT),
        ("+z", AREA_BOX_TOP),
    ]
    pts = []
    for _ in range(n):
        r = rng.uniform(0.0, AREA_BOX_TOTAL)
        acc = 0.0
        face = faces[-1][0]
        for name, area in faces:
            acc += area
            if r <= acc:
                face = name
                break
        if face == "+x":
            pts.append((cx + hx, rng.uniform(cy - hy, cy + hy), rng.uniform(cz - hz, cz + hz)))
        elif face == "-x":
            pts.append((cx - hx, rng.uniform(cy - hy, cy + hy), rng.uniform(cz - hz, cz + hz)))
        elif face == "+y":
            pts.append((rng.uniform(cx - hx, cx + hx), cy + hy, rng.uniform(cz - hz, cz + hz)))
        elif face == "-y":
            pts.append((rng.uniform(cx - hx, cx + hx), cy - hy, rng.uniform(cz - hz, cz + hz)))
        else:  # "+z" -- top face
            pts.append((rng.uniform(cx - hx, cx + hx), rng.uniform(cy - hy, cy + hy), cz + hz))
    return pts


def sample_scene(rng: random.Random, n_total: int) -> list:
    """One full structured scan: floor + 2 walls + box, ~n_total points,
    density-balanced by surface area (see the AREA_* constants above).
    The LAST category (box) absorbs the rounding remainder so the returned
    list has EXACTLY n_total points."""
    n_floor = round(n_total * AREA_FLOOR / AREA_TOTAL)
    n_wall_a = round(n_total * AREA_WALL / AREA_TOTAL)
    n_wall_b = round(n_total * AREA_WALL / AREA_TOTAL)
    n_box = n_total - n_floor - n_wall_a - n_wall_b   # remainder -> exact total

    pts = []
    pts += sample_floor(rng, n_floor)
    pts += sample_wall_a(rng, n_wall_a)
    pts += sample_wall_b(rng, n_wall_b)
    pts += sample_box(rng, n_box)
    return pts


def add_noise(rng: random.Random, pts: list, sigma_m: float) -> list:
    """Independent isotropic Gaussian jitter per point, per axis -- the
    NOISE_SIGMA_M sensor-noise model documented at the top of this file."""
    return [(x + rng.gauss(0.0, sigma_m), y + rng.gauss(0.0, sigma_m), z + rng.gauss(0.0, sigma_m))
            for x, y, z in pts]


# ---------------------------------------------------------------------------
# Ground-truth SE(3): axis-angle -> unit quaternion (w,x,y,z), then the
# quaternion's equivalent rotation MATRIX applied point-by-point. Repo
# convention: quaternions are scalar-first, kept normalized at every
# boundary (CLAUDE.md paragraph 12) -- axis is normalized here so a caller
# can pass any nonzero vector, not just a unit one.
# ---------------------------------------------------------------------------
def axis_angle_to_quat(axis, angle_rad: float):
    ax, ay, az = axis
    n = math.sqrt(ax * ax + ay * ay + az * az)
    ax, ay, az = ax / n, ay / n, az / n
    half = angle_rad / 2.0
    s = math.sin(half)
    return (math.cos(half), ax * s, ay * s, az * s)   # (w, x, y, z)


def quat_to_matrix(q):
    w, x, y, z = q
    return (
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
        2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y),
    )


def transform_points(q, t, pts: list) -> list:
    r00, r01, r02, r10, r11, r12, r20, r21, r22 = quat_to_matrix(q)
    tx, ty, tz = t
    out = []
    for px, py, pz in pts:
        out.append((
            r00 * px + r01 * py + r02 * pz + tx,
            r10 * px + r11 * py + r12 * pz + ty,
            r20 * px + r21 * py + r22 * pz + tz,
        ))
    return out


# ---------------------------------------------------------------------------
# Binary cloud format -- documented byte-exactly in ../data/README.md too
# (the two documents must agree; this script is the format's source of
# truth). Chosen over CSV (as CLAUDE.md paragraph 8's "CSV/binary" allows)
# to demonstrate binary point-cloud I/O, the format real LiDAR drivers use.
#
#   offset 0, 4 bytes : ASCII magic "PC01"
#   offset 4, 4 bytes : uint32 little-endian point count N
#   offset 8, N*12 bytes : N * (float32 x, float32 y, float32 z), little-endian,
#                          meters, target-cloud's own frame (source clouds
#                          are in the CANONICAL/un-transformed frame; target
#                          clouds are already in the "scanned" frame -- see
#                          pairs_meta.csv for the transform between them)
# ---------------------------------------------------------------------------
def write_cloud_bin(path: Path, pts: list) -> None:
    flat = []
    for p in pts:
        flat.extend(p)
    with path.open("wb") as f:
        f.write(b"PC01")
        f.write(struct.pack("<I", len(pts)))
        f.write(struct.pack("<%df" % len(flat), *flat))


def generate_pair(spec: dict, base_seed: int):
    """Build one (source, target, q_gt, t_gt) tuple for one demo pair.

    Four INDEPENDENT RNG streams, in this fixed order (reproducibility
    requires the draw order to be pinned, not just the seeds):
      1. source-cloud surface sampling  (seed + offset + 0)
      2. source-cloud noise             (seed + offset + 1)
      3. target-cloud surface sampling  (seed + offset + 2) -- an
         INDEPENDENT resampling of the SAME primitives, mimicking a second,
         different LiDAR scan of the same static scene (not a copy of the
         source points -- see the file header "why this script exists")
      4. target-cloud noise             (seed + offset + 3)
    """
    s = base_seed + spec["seed_offset"]
    n = spec["n_total"]

    src_pts = sample_scene(random.Random(s + 0), n)
    src_pts = add_noise(random.Random(s + 1), src_pts, NOISE_SIGMA_M)

    tgt_canonical = sample_scene(random.Random(s + 2), n)
    q_gt = axis_angle_to_quat(spec["rot_axis"], math.radians(spec["rot_deg"]))
    t_gt = spec["t_m"]
    tgt_pts = transform_points(q_gt, t_gt, tgt_canonical)
    tgt_pts = add_noise(random.Random(s + 3), tgt_pts, NOISE_SIGMA_M)

    return src_pts, tgt_pts, q_gt, t_gt


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_out = script_dir.parent / "data" / "sample"

    parser = argparse.ArgumentParser(
        description="Generate the synthetic ICP demo pairs for project 02.06.")
    parser.add_argument("--seed", type=int, default=42,
                        help="base RNG seed (default 42; each pair offsets from it -- see PAIRS)")
    parser.add_argument("--out", type=Path, default=default_out,
                        help="output directory (default: ../data/sample/)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    meta_rows = []
    for spec in PAIRS:
        src_pts, tgt_pts, q_gt, t_gt = generate_pair(spec, args.seed)

        src_path = args.out / f"{spec['name']}_source.bin"
        tgt_path = args.out / f"{spec['name']}_target.bin"
        write_cloud_bin(src_path, src_pts)
        write_cloud_bin(tgt_path, tgt_pts)

        meta_rows.append([
            "PAIR", spec["name"], src_path.name, tgt_path.name,
            len(src_pts), len(tgt_pts),
            f"{q_gt[0]:.10f}", f"{q_gt[1]:.10f}", f"{q_gt[2]:.10f}", f"{q_gt[3]:.10f}",
            f"{t_gt[0]:.10f}", f"{t_gt[1]:.10f}", f"{t_gt[2]:.10f}",
            f"{NOISE_SIGMA_M:.10f}",
        ])
        print(f"[make_synthetic] {spec['name']}: wrote {len(src_pts)} source / {len(tgt_pts)} target points "
              f"(rot {spec['rot_deg']:.1f} deg about {spec['rot_axis']}, |t|="
              f"{math.sqrt(sum(c * c for c in t_gt)):.3f} m) [SYNTHETIC]")

    meta_path = args.out / "pairs_meta.csv"
    with meta_path.open("w", newline="", encoding="utf-8") as f:
        f.write("# SYNTHETIC ground-truth metadata for project 02.06 ICP demo pairs.\n")
        f.write(f"# regenerate: python make_synthetic.py --seed {args.seed}\n")
        f.write("# columns: PAIR,name,source_file,target_file,n_source,n_target,"
                "qw,qx,qy,qz,tx_m,ty_m,tz_m,noise_sigma_m\n")
        f.write("# q is the unit quaternion (w,x,y,z) and t the translation (m) of "
                "T_target_source: apply it to the SOURCE cloud to land it in the TARGET frame.\n")
        writer = csv.writer(f)
        for row in meta_rows:
            writer.writerow(row)

    print(f"[make_synthetic] wrote metadata to {meta_path}")


if __name__ == "__main__":
    main()
