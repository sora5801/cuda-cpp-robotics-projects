#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample generator for project 05.01
(TSDF fusion + marching-cubes mesh extraction).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
---------------------------------------------------------------
This project's committed sample is a CAMERA PATH, not recordings: pinhole
intrinsics plus a ring of poses orbiting the analytic scene (a sphere
floating above a ground plane, defined once in ../src/kernels.cuh). The
depth images themselves are rendered INSIDE the demo at run time, by
closed-form ray casting of that scene — so the committed data stays tiny
(~2.5 KiB), the demo needs zero downloads, and ground truth is the scene's
exact signed distance function rather than a stored fixture. This file
documents and reproduces the path; ../data/README.md documents the format.

What it writes: ../data/sample/camera_path.csv

    CAM,width,height,fx,fy,cx,cy      pinhole intrinsics (pixels)
    POSE,i,tx,ty,tz,qw,qx,qy,qz       T_world_cam per frame: camera position
                                      (m, world frame, z up) + orientation
                                      quaternion (w,x,y,z — repo order),
                                      camera optical convention x-right /
                                      y-down / z-forward

The default path is a 24-pose circle of radius 2.0 m at height 1.2 m, every
camera aimed at a point between the plane and the sphere — chosen so every
part of the sphere and the central plane is seen from many directions
(fusion averages best where views multiply). No RNG is involved (a camera
path is constants — closed-form trigonometry); the file is byte-reproducible
by construction, so there is no seed to fix (the repo's fixed-seed rule,
CLAUDE.md paragraph 12, is satisfied vacuously and honestly).

Usage:
    python make_synthetic.py                      # the committed path
    python make_synthetic.py --frames 48 --radius 2.5   # experiments; do not commit
"""

import argparse
import math
import sys
from pathlib import Path

# Where the cameras look: between the ground plane (z=0) and the sphere
# center (z=0.75) — must match the scene in ../src/kernels.cuh if edited.
TARGET = (0.0, 0.0, 0.55)


def look_at_quaternion(eye, target):
    """Camera orientation (w,x,y,z) for an OPTICAL-frame camera at `eye`
    looking at `target`, with the image's "up" aligned against world +z.

    Optical convention (x-right, y-down, z-forward — the domain standard,
    stated per CLAUDE.md paragraph 12): the rotation's COLUMNS are the
    camera axes expressed in world coordinates:
        z_cam = normalize(target - eye)            (forward = viewing ray)
        x_cam = normalize(z_cam x world_up)        (right, horizontal)
        y_cam = z_cam x x_cam                      (down — completes the
                                                    right-handed triad; its
                                                    world-z component is
                                                    negative when the camera
                                                    is not upside down)
    Degenerate case (looking straight up/down) cannot occur on this path
    (the cameras always look inward and downward at a slant).
    """
    fx, fy, fz = (target[0] - eye[0], target[1] - eye[1], target[2] - eye[2])
    n = math.sqrt(fx * fx + fy * fy + fz * fz)
    fx, fy, fz = fx / n, fy / n, fz / n                    # z_cam
    # x_cam = f x up, up = (0,0,1)  →  (fy, -fx, 0), normalized.
    n = math.sqrt(fx * fx + fy * fy)
    xx, xy, xz = fy / n, -fx / n, 0.0                      # x_cam
    # y_cam = z_cam x x_cam (right-handed; points "down" in world).
    yx = fy * xz - fz * xy
    yy = fz * xx - fx * xz
    yz = fx * xy - fy * xx

    # Rotation matrix with columns (x_cam, y_cam, z_cam) → quaternion by the
    # standard Shepperd branch selection (numerically safest pivot first).
    m = [[xx, yx, fx],
         [xy, yy, fy],
         [xz, yz, fz]]
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2][1] - m[1][2]) / s
        y = (m[0][2] - m[2][0]) / s
        z = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        w = (m[2][1] - m[1][2]) / s
        x = 0.25 * s
        y = (m[0][1] + m[1][0]) / s
        z = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        w = (m[0][2] - m[2][0]) / s
        x = (m[0][1] + m[1][0]) / s
        y = 0.25 * s
        z = (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        w = (m[1][0] - m[0][1]) / s
        x = (m[0][2] + m[2][0]) / s
        y = (m[1][2] + m[2][1]) / s
        z = 0.25 * s
    # Canonical sign (w >= 0) so regeneration is byte-stable regardless of branch.
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    return w, x, y, z


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=24,
                    help="number of poses on the circle (default 24)")
    ap.add_argument("--radius", type=float, default=2.0,
                    help="circle radius in m (default 2.0 — outside the 2.56 m volume)")
    ap.add_argument("--height", type=float, default=1.2,
                    help="camera height in m (default 1.2 — looking down at the scene)")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data" / "sample" / "camera_path.csv")
    args = ap.parse_args()
    if args.frames < 1:
        ap.error("--frames must be >= 1")

    lines = [
        "# camera_path.csv - SYNTHETIC camera trajectory for project 05.01",
        "# generated by scripts/make_synthetic.py (closed-form circle + look-at; no RNG - a camera path is constants)",
        "# CAM,width,height,fx,fy,cx,cy : pinhole intrinsics (px); optical frame x-right/y-down/z-forward",
        "# POSE,i,tx,ty,tz,qw,qx,qy,qz  : T_world_cam - camera position (m, world, z up) + quaternion (w,x,y,z)",
        "# license: same as the repository (MIT) - fully synthetic, no external source",
        # 160x120 with fx=fy=120 px → ~67°x53° FOV: the whole scene fits in
        # frame from 2 m out, and one pixel spans ~1 voxel at scene range —
        # matched sampling densities (THEORY.md §numerics).
        "CAM,160,120,120,120,79.5,59.5",
    ]
    for i in range(args.frames):
        phi = 2.0 * math.pi * i / args.frames            # evenly spaced around the ring
        eye = (args.radius * math.cos(phi), args.radius * math.sin(phi), args.height)
        w, x, y, z = look_at_quaternion(eye, TARGET)
        # %.9g: enough digits that float32 round-trips exactly; short enough
        # to keep the file readable and byte-stable across platforms.
        lines.append("POSE,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g"
                     % (i, eye[0], eye[1], eye[2], w, x, y, z))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:   # LF pinned
        f.write("\n".join(lines) + "\n")

    print(f"wrote {args.out} ({args.out.stat().st_size} bytes: {args.frames} poses, "
          f"radius {args.radius:g} m, height {args.height:g} m) - labeled SYNTHETIC")
    if args.frames != 24 or args.radius != 2.0 or args.height != 1.2:
        print("note: non-default path - fine for experiments, do NOT commit this file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
