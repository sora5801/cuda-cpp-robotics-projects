#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample-data generator for 19.01
(Parallel grasp-candidate scoring: antipodal sampling over point clouds).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
---------------------------------------------------------------
Robotics data can almost always be synthesized with full ground truth. This
project needs that ground truth precisely: the whole verification strategy
(README "Expected output", THEORY.md "How we verify correctness") depends on
objects whose GOOD GRASPS ARE KNOWN GEOMETRICALLY in advance, so the demo can
assert "the top-ranked grasps really are the box's opposite-face pairs" and
not just "the code ran". A real depth-camera scan of a real box has no such
ground truth to check against.

What this script generates
---------------------------
Three convex analytic objects, each a noisy point cloud sampled on its OUTER
surface (a depth camera never sees the inside of anything):

  * a BOX          — dims 60 x 40 x 100 mm. Antipodal grasps exist across
                      three axis pairs (opposite faces); the widths are
                      60 mm, 40 mm, and 100 mm. The gripper modeled here has
                      a 10-90 mm stroke (objects_meta.csv), so the 100 mm
                      axis is GEOMETRICALLY antipodal but GRIPPER-INFEASIBLE
                      — a deliberate, honest teaching case (README/THEORY).
  * a CYLINDER      — radius 25 mm, height 120 mm, LATERAL surface only (no
                      end caps — see data/README.md). Antipodal grasps exist
                      across any diameter (50 mm), at any height and any
                      angle around the axis.
  * a SPHERE        — radius 30 mm. Every diametral pair (60 mm) is a valid
                      antipodal grasp; there is no "wrong axis" on a sphere.

Every point gets light, physically-motivated noise: a small AXIAL offset
along the true surface normal (depth-camera range noise) plus a smaller
TANGENTIAL offset in the surface plane (lateral pixel jitter). Both are
zero-mean Gaussian, both far smaller than the point spacing, so PCA normal
recovery (src/kernels.cu's estimate_normals_kernel) stays accurate — see
data/README.md for the exact sigmas and the point-spacing arithmetic that
justifies them.

Determinism (CLAUDE.md §12): ONE seeded RNG stream, advanced in the fixed
order box -> cylinder -> sphere, each object drawing a fixed number of
random values per point (documented in each sample_* function) — the same
seed always reproduces the same bytes on any machine.

Usage
-----
    python make_synthetic.py                  # defaults: seed=42, writes to ../data/sample/
    python make_synthetic.py --seed 7 --out-dir /tmp/mygrasp
"""

import argparse
import csv
import math
import random
import struct
from pathlib import Path

# The fixed default seed (CLAUDE.md paragraph 12: determinism is repo law).
DEFAULT_SEED = 42

# ---------------------------------------------------------------------------
# Object geometry (meters). These are the GROUND TRUTH main.cu's analytic
# gates check the ranked grasps against — see objects_meta.csv, which carries
# these same numbers so main.cu never re-hardcodes them independently
# (CLAUDE.md §12: single-source shared constants).
# ---------------------------------------------------------------------------
BOX_DX_M, BOX_DY_M, BOX_DZ_M = 0.06, 0.04, 0.10   # full extents (x,y,z)
BOX_N = 6000

CYL_R_M, CYL_H_M = 0.025, 0.12                    # radius, height (lateral surface only)
CYL_N = 9000

SPH_R_M = 0.03
SPH_N = 7000

# Gripper + friction model shared by every object in this demo (illustrative
# parallel-jaw gripper, PRACTICE.md §2 dates and caveats the numbers).
GRIPPER_W_MIN_M = 0.01
GRIPPER_W_MAX_M = 0.09
FRICTION_MU = 0.5

# Noise model (meters): axial (along the true surface normal, i.e. simulated
# depth-camera range noise) and tangential (in-plane, simulated lateral
# pixel jitter). data/README.md justifies both against each object's point
# spacing.
NOISE_AXIAL_SIGMA_M = 0.0003        # 0.3 mm
NOISE_TANGENT_SIGMA_M = 0.00015     # 0.15 mm


def sample_box(rng: random.Random, n: int, dx: float, dy: float, dz: float):
    """Sample n points on the surface of an axis-aligned box centered at the
    origin, full extents (dx, dy, dz), area-weighted across its 6 faces.

    Draws per point: 1 (face choice) + 2 (in-plane position) + 1 (axial
    noise) + 2 (tangential noise) = 6 random values, in that fixed order —
    the exact order is what makes the output byte-reproducible for a given
    seed (CLAUDE.md §12).

    Returns a list of n (x, y, z) tuples, meters.
    """
    hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0
    # Six faces: (center, in-plane axis 1, in-plane axis 2, outward normal,
    # half-extent along axis 1, half-extent along axis 2, face area).
    faces = [
        ((hx, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0), hy, hz, dy * dz),   # +x
        ((-hx, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), hy, hz, dy * dz),  # -x
        ((0, hy, 0), (1, 0, 0), (0, 0, 1), (0, 1, 0), hx, hz, dx * dz),   # +y
        ((0, -hy, 0), (1, 0, 0), (0, 0, 1), (0, -1, 0), hx, hz, dx * dz),  # -y
        ((0, 0, hz), (1, 0, 0), (0, 1, 0), (0, 0, 1), hx, hy, dx * dy),   # +z
        ((0, 0, -hz), (1, 0, 0), (0, 1, 0), (0, 0, -1), hx, hy, dx * dy),  # -z
    ]
    total_area = sum(f[6] for f in faces)
    cum = []
    running = 0.0
    for f in faces:
        running += f[6] / total_area
        cum.append(running)

    pts = []
    for _ in range(n):
        r = rng.random()                          # draw 1: face choice
        face_idx = 0
        for i, c in enumerate(cum):
            if r <= c:
                face_idx = i
                break
        center, ax1, ax2, nrm, half1, half2, _area = faces[face_idx]

        u = rng.uniform(-half1, half1)             # draw 2: in-plane coord 1
        v = rng.uniform(-half2, half2)              # draw 3: in-plane coord 2
        axial = rng.gauss(0.0, NOISE_AXIAL_SIGMA_M)  # draw 4: axial (range) noise
        t1 = rng.gauss(0.0, NOISE_TANGENT_SIGMA_M)   # draw 5: tangential noise 1
        t2 = rng.gauss(0.0, NOISE_TANGENT_SIGMA_M)   # draw 6: tangential noise 2

        px = center[0] + u * ax1[0] + v * ax2[0] + axial * nrm[0] + t1 * ax1[0] + t2 * ax2[0]
        py = center[1] + u * ax1[1] + v * ax2[1] + axial * nrm[1] + t1 * ax1[1] + t2 * ax2[1]
        pz = center[2] + u * ax1[2] + v * ax2[2] + axial * nrm[2] + t1 * ax1[2] + t2 * ax2[2]
        pts.append((px, py, pz))
    return pts


def sample_cylinder(rng: random.Random, n: int, r: float, h: float):
    """Sample n points on the LATERAL surface only of a cylinder centered at
    the origin, axis along z, radius r, height h (no end caps — a real
    depth camera looking at a can from the side sees exactly this: the
    lateral surface, not the caps, and this project's grasp geometry only
    needs the lateral surface's diametral pairs — data/README.md).

    Draws per point: 1 (theta) + 1 (z) + 1 (radial/axial noise) + 1
    (tangential z noise) = 4 random values, fixed order.
    """
    pts = []
    for _ in range(n):
        theta = rng.uniform(0.0, 2.0 * math.pi)     # draw 1
        z = rng.uniform(-h / 2.0, h / 2.0)            # draw 2
        r_noise = rng.gauss(0.0, NOISE_AXIAL_SIGMA_M)  # draw 3: radial (range) noise
        z_noise = rng.gauss(0.0, NOISE_TANGENT_SIGMA_M)  # draw 4: tangential (along-axis) noise

        rr = r + r_noise
        px = rr * math.cos(theta)
        py = rr * math.sin(theta)
        pz = z + z_noise
        pts.append((px, py, pz))
    return pts


def sample_sphere(rng: random.Random, n: int, r: float):
    """Sample n points uniformly on a sphere of radius r centered at the
    origin, via the standard "normalize three independent Gaussians" trick
    (the only elementary construction that is exactly uniform on the sphere;
    e.g. picking uniform theta/phi over-samples the poles).

    Draws per point: 3 (direction) + 1 (radial noise) = 4 random values,
    fixed order.
    """
    pts = []
    for _ in range(n):
        gx = rng.gauss(0.0, 1.0)   # draw 1
        gy = rng.gauss(0.0, 1.0)   # draw 2
        gz = rng.gauss(0.0, 1.0)   # draw 3
        norm = math.sqrt(gx * gx + gy * gy + gz * gz)
        if norm < 1e-12:
            norm = 1.0  # astronomically unlikely with a seeded stream; guards a division by zero honestly
        dx, dy, dz = gx / norm, gy / norm, gz / norm

        r_noise = rng.gauss(0.0, NOISE_AXIAL_SIGMA_M)  # draw 4: radial (range) noise
        rr = r + r_noise
        pts.append((rr * dx, rr * dy, rr * dz))
    return pts


def write_cloud_bin(path: Path, points) -> None:
    """Write points (list of (x,y,z) float tuples, meters) in this project's
    binary cloud format: 4-byte magic 'GC01', little-endian uint32 count,
    then count*3 little-endian float32 values, interleaved x,y,z — the same
    interleaved-xyz convention every point-cloud project in this repo uses
    (docs/SYSTEM_DESIGN.md §3.6 PointCloud), in a project-local binary
    layout documented in data/README.md (distinct magic from project
    02.06's 'PC01' — self-containment rule, CLAUDE.md §4: no cross-project
    format coupling, even though the layout happens to be the same shape).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(b"GC01")
        f.write(struct.pack("<I", len(points)))
        for (x, y, z) in points:
            f.write(struct.pack("<fff", x, y, z))


def write_objects_meta(path: Path, rows) -> None:
    """Write objects_meta.csv — one row per object, format documented
    byte-exactly in data/README.md. `rows` is a list of dicts with the
    columns listed in the header comment written below.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        f.write("# SYNTHETIC data - generated by scripts/make_synthetic.py for project 19.01\n")
        f.write(f"# regenerate: python make_synthetic.py --seed {rows[0]['_seed']}\n")
        f.write("# columns: name,file,n_points,shape,param_a_m,param_b_m,param_c_m,"
                "gripper_w_min_m,gripper_w_max_m,friction_mu\n")
        f.write("# shape='box': param_a/b/c = full extents dim_x_m,dim_y_m,dim_z_m\n")
        f.write("# shape='cylinder': param_a=radius_m, param_b=height_m, param_c unused (0)\n")
        f.write("# shape='sphere': param_a=radius_m, param_b/c unused (0)\n")
        writer = csv.writer(f)
        writer.writerow(["name", "file", "n_points", "shape",
                         "param_a_m", "param_b_m", "param_c_m",
                         "gripper_w_min_m", "gripper_w_max_m", "friction_mu"])
        for row in rows:
            writer.writerow([row["name"], row["file"], row["n_points"], row["shape"],
                             f"{row['param_a_m']:.6f}", f"{row['param_b_m']:.6f}", f"{row['param_c_m']:.6f}",
                             f"{row['gripper_w_min_m']:.6f}", f"{row['gripper_w_max_m']:.6f}",
                             f"{row['friction_mu']:.6f}"])


def generate_all(seed: int, out_dir: Path) -> None:
    """Generate all three object clouds plus objects_meta.csv, in the FIXED
    order box -> cylinder -> sphere, from ONE seeded RNG stream shared
    across all three (CLAUDE.md §12: one seed, one deterministic byte
    stream, regardless of how many objects consume it).
    """
    rng = random.Random(seed)

    box_pts = sample_box(rng, BOX_N, BOX_DX_M, BOX_DY_M, BOX_DZ_M)
    cyl_pts = sample_cylinder(rng, CYL_N, CYL_R_M, CYL_H_M)
    sph_pts = sample_sphere(rng, SPH_N, SPH_R_M)

    write_cloud_bin(out_dir / "box_cloud.bin", box_pts)
    write_cloud_bin(out_dir / "cylinder_cloud.bin", cyl_pts)
    write_cloud_bin(out_dir / "sphere_cloud.bin", sph_pts)

    rows = [
        {"_seed": seed, "name": "box", "file": "box_cloud.bin", "n_points": BOX_N, "shape": "box",
         "param_a_m": BOX_DX_M, "param_b_m": BOX_DY_M, "param_c_m": BOX_DZ_M,
         "gripper_w_min_m": GRIPPER_W_MIN_M, "gripper_w_max_m": GRIPPER_W_MAX_M, "friction_mu": FRICTION_MU},
        {"_seed": seed, "name": "cylinder", "file": "cylinder_cloud.bin", "n_points": CYL_N, "shape": "cylinder",
         "param_a_m": CYL_R_M, "param_b_m": CYL_H_M, "param_c_m": 0.0,
         "gripper_w_min_m": GRIPPER_W_MIN_M, "gripper_w_max_m": GRIPPER_W_MAX_M, "friction_mu": FRICTION_MU},
        {"_seed": seed, "name": "sphere", "file": "sphere_cloud.bin", "n_points": SPH_N, "shape": "sphere",
         "param_a_m": SPH_R_M, "param_b_m": 0.0, "param_c_m": 0.0,
         "gripper_w_min_m": GRIPPER_W_MIN_M, "gripper_w_max_m": GRIPPER_W_MAX_M, "friction_mu": FRICTION_MU},
    ]
    write_objects_meta(out_dir / "objects_meta.csv", rows)

    print(f"[make_synthetic] wrote box ({BOX_N} pts), cylinder ({CYL_N} pts), "
          f"sphere ({SPH_N} pts) + objects_meta.csv to {out_dir} (seed={seed}, labeled SYNTHETIC)")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_out = script_dir.parent / "data" / "sample"

    parser = argparse.ArgumentParser(
        description="Generate the synthetic box/cylinder/sphere grasp-target point clouds "
                    "for project 19.01 (Parallel grasp-candidate scoring).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"RNG seed for byte-identical reproducibility (default {DEFAULT_SEED})")
    parser.add_argument("--out-dir", type=Path, default=default_out,
                        help="output directory (default: ../data/sample/)")
    args = parser.parse_args()

    generate_all(args.seed, args.out_dir)


if __name__ == "__main__":
    main()
