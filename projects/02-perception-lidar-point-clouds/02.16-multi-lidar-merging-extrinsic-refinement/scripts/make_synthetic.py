#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample-data generator for project 02.16
   (Multi-LiDAR merging + extrinsic refinement).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
-----------------------------------------------------------------
Extrinsic-refinement needs a KNOWN ground-truth mounting error to be
verifiable at all -- a real fleet vehicle never ships with an answer key
for "how far has this LiDAR actually drifted from its factory pose". So,
exactly like 01.17's camera-calibration data and 02.06's ICP pairs, this
project's data is synthetic BY NECESSITY: we build a small structured
"yard" (a ground plane plus four walls at three mutually orthogonal
orientations, plus a few decorative poles), scan it from THREE LiDARs
mounted on one vehicle rig, and give two of the three sensors a KNOWN,
tiny, deliberate mounting drift relative to their nominal (as-designed)
pose. main.cu's job is to detect that drift from overlapping geometry
alone, refine it away, and recover the known answer.

THE RIG (single source of truth: mirrors kernels.cuh's constants exactly
-- see that file's header for the ASCII rig diagram and the full citation
of 01.17's "mounts creep" drift story). This script duplicates the
numbers, in Python, because kernels.cuh's are C++ and cannot be included
here -- but both files derive them from the same documented mount
positions / mount yaws / drift vectors, so "true" as computed here and
"true" as re-derived by main.cu for grading describe the IDENTICAL rigid
transform up to float64-vs-float32 rounding, utterly negligible next to
this project's millimeter/tenth-of-a-degree gates:

    MAIN  (roof, 360 deg, no drift -- the trusted reference sensor)
    LEFT  (front-left corner, ~55 deg outward yaw, 0.8 deg / 3 cm drift)
    RIGHT (front-right corner, ~55 deg outward yaw, 0.8 deg / 3 cm drift,
           a DIFFERENT drift vector than LEFT so the two recoveries are
           not suspiciously identical)

FIELD-OF-VIEW MODEL (a documented simplification -- see kernels.cuh):
coverage is decided by the BASE-FRAME azimuth from each sensor's NOMINAL
mount origin (decoupled from the sensor's own small mounting tilt, which
affects the COORDINATES a sensor reports but not, to first order, which
wide FOV wedge a corner unit is built to cover). This keeps "which sensor
sees which surface" identical between the aligned and drifted cohorts
(only coordinates change with drift, never visibility), which is what
lets one shared world-point grid serve both cohorts (below).

TWO COHORTS, ONE WORLD GRID
----------------------------
We sample ONE deterministic world-point grid per surface, decide once
(from FOV geometry, nominal mount origins only) which sensors capture
each world point, and then -- for EACH cohort -- express every captured
point in its owning sensor's OWN frame under that cohort's TRUE extrinsic
(aligned cohort: true == nominal for every sensor; drifted cohort: true
== nominal + drift for LEFT/RIGHT). Because both cohorts start from the
IDENTICAL visibility decision, any difference a learner sees between the
two cohorts' merged clouds is caused ONLY by the drift itself -- not by
an accidental change in which points got sampled. Independent Gaussian
range noise (see NOISE_SIGMA_M) is added fresh per cohort per point.

What gets written (see ../data/README.md for the byte-exact format spec)
--------------------------------------------------------------------------
  aligned.csv   one row per (sensor, point) for the CONTROL rig (no drift)
  drifted.csv   the same points, same sensor assignment, for the PATIENT
                rig (LEFT/RIGHT drifted) -- the file main.cu spends most
                of its time on

Every draw comes from Python's std-lib `random.Random`, seeded 42 (the
same choice 02.06/08.01 make for their own synthetic generators) -- no
numpy, no external data, per this repo's Python-stdlib-only convention.

Usage
-----
    python make_synthetic.py                      # defaults: seed=42, writes ../data/sample/
    python make_synthetic.py --seed 7 --out DIR
"""

import argparse
import math
import random
from pathlib import Path

# ===========================================================================
# THE RIG — mirrors kernels.cuh's constants exactly (see this project's
# kernels.cuh header for the ASCII rig diagram and the full citation of
# 01.17's "mounts creep" drift story). Distances in meters, angles in
# degrees in comments / radians in code (repo convention, CLAUDE.md §12).
# ===========================================================================
DEG2RAD = math.pi / 180.0

# Sensor ids (must match kernels.cuh's kSensorMain/Left/Right).
SENSOR_MAIN, SENSOR_LEFT, SENSOR_RIGHT = 0, 1, 2
SENSOR_NAMES = {SENSOR_MAIN: "main", SENSOR_LEFT: "left", SENSOR_RIGHT: "right"}

# Surface ids (must match kernels.cuh's kSurface* constants).
SURF_GROUND, SURF_WALL_FRONT, SURF_WALL_LEFT, SURF_WALL_RIGHT, SURF_WALL_REAR, SURF_POLE = range(6)

# Nominal mount pose: position (m, base frame) + a single YAW about +z
# (radians) -- corner sensors are angled outward, the roof sensor is not
# (it spins 360 deg so its own boresight direction is irrelevant to
# coverage). See kernels.cuh for why FOV coverage is decided independently
# of this yaw (a documented simplification): the yaw still matters here
# because it rotates the COORDINATES each sensor reports.
MOUNT = {
    SENSOR_MAIN:  {"pos": (0.0, 0.0, 1.8), "yaw_deg": 0.0},
    SENSOR_LEFT:  {"pos": (1.8, 0.9, 0.5), "yaw_deg": 55.0},
    SENSOR_RIGHT: {"pos": (1.8, -0.9, 0.5), "yaw_deg": -55.0},
}

# FOV wedge (deg, contiguous, no wraparound) and max range (m), evaluated
# as the base-frame azimuth from the sensor's NOMINAL mount (x,y) origin.
# MAIN's wedge is unused (full 360); LEFT/RIGHT's 180 deg-wide wedges
# overlap only in the +/-15 deg forward cone -- see kernels.cuh for the
# overlap-zone table this geometry produces.
FOV = {
    SENSOR_MAIN:  {"az_min": None, "az_max": None, "max_range": 22.0},
    SENSOR_LEFT:  {"az_min": -15.0, "az_max": 165.0, "max_range": 16.0},
    SENSOR_RIGHT: {"az_min": -165.0, "az_max": 15.0, "max_range": 16.0},
}

# Drift: TRUE = Exp(drift_omega) * NOMINAL_R  (world/left-perturbation,
# 01.17's retraction convention, cited), t_true = t_nominal + drift_t.
# Magnitudes ~0.8 deg / 3 cm per the catalog brief's "mounts creep"
# example; DIFFERENT axes for LEFT vs RIGHT so the two recoveries are
# independent evidence, not a mirrored coincidence. MAIN never drifts —
# it is this project's trusted reference/anchor sensor.
DRIFT = {
    SENSOR_MAIN:  {"omega_deg": (0.0, 0.0, 0.0), "t_m": (0.0, 0.0, 0.0)},
    SENSOR_LEFT:  {"omega_deg": (0.5333, -0.5333, 0.2667), "t_m": (0.02592, 0.01296, -0.00778)},
    SENSOR_RIGHT: {"omega_deg": (-0.5444, 0.5444, 0.2177), "t_m": (-0.01418, 0.02364, 0.01182)},
}

# Per-point LiDAR range noise (m, 1-sigma), applied along the sensor-to-
# point direction (physically a RANGE noise, not isotropic Cartesian jitter
# -- see THEORY.md "Numerical considerations" for the honest simplification
# this still makes: a real beam also has angular/timing noise we omit).
NOISE_SIGMA_M = 0.006

# World-surface sampling grids. Ground is deliberately coarse (a LOT of
# candidate cells) with a WIDE extent so every sensor's FOV wedge is fully
# exercised; walls are finer (smaller extent, need good plane-fit density).
GROUND_EXTENT = 11.0     # m, half-width; grid spans [-11, 11] x [-11, 11]
GROUND_SPACING = 0.55    # m
WALL_Z_RANGE = (0.2, 3.0)   # m (start above 0 so no ground/wall coincident points)
WALL_SPACING = 0.30      # m
POLE_RADIUS = 0.15       # m
POLE_HEIGHT = 3.0        # m
POLE_CENTERS = [(4.0, 3.0), (4.0, -3.0), (-4.0, 3.0), (-4.0, -3.0)]  # m, (x, y)
POLE_RING_POINTS = 10    # points per height ring
POLE_RINGS = 10          # rings from base to top

WALL_FRONT_X = 10.0
WALL_LEFT_Y = 8.0
WALL_RIGHT_Y = -8.0
WALL_REAR_X = -10.0
WALL_Y_RANGE = (-6.0, 6.0)   # front/rear walls span this y range
WALL_X_RANGE = (-8.0, 12.0)  # left/right walls span this x range


# ===========================================================================
# Small SO(3)/rigid-transform helpers — a Python re-derivation of
# kernels.cuh's so3_exp/mat3_vec/mat3_mul (cited, reimplemented per this
# repo's self-containment rule: no cross-language include is possible, and
# a hand port is exactly the kind of "pure transcription" the twin-
# independence ruling in reference_cpu.cpp's header calls out as fine to
# duplicate for a closed-form primitive). Rotations are 3x3 row-major
# lists-of-3-lists, matching kernels.cuh's Rigid3::R layout.
# ===========================================================================
def so3_exp(omega):
    """Rodrigues' formula: exact SO(3) exponential of an axis-angle vector
    (radians). Mirrors kernels.cuh's so3_exp in ALGORITHM (not bit-for-bit
    in FLOATING POINT -- Python computes in float64, C++ in float32; see
    this file's header for why that is harmless here)."""
    x, y, z = omega
    theta = math.sqrt(x * x + y * y + z * z)
    sx = [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]]  # skew3(omega)
    if theta < 1e-8:
        return [[1.0 + sx[0][0], sx[0][1], sx[0][2]],
                [sx[1][0], 1.0 + sx[1][1], sx[1][2]],
                [sx[2][0], sx[2][1], 1.0 + sx[2][2]]]
    a = math.sin(theta) / theta
    b = (1.0 - math.cos(theta)) / (theta * theta)
    s2 = mat3_mul(sx, sx)
    R = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            ident = 1.0 if i == j else 0.0
            R[i][j] = ident + a * sx[i][j] + b * s2[i][j]
    return R


def mat3_mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def mat3_vec(R, p):
    return [sum(R[i][k] * p[k] for k in range(3)) for i in range(3)]


def mat3_transpose_vec(R, p):
    """R^T * p — used to go FROM base frame INTO a sensor's own frame:
    p_sensor = R_true^T * (p_base - t_true), the inverse of the rigid
    transform x_base = R_true * p_sensor + t_true."""
    Rt = [[R[j][i] for j in range(3)] for i in range(3)]
    return mat3_vec(Rt, p)


def sensor_extrinsic(sensor_id, drifted):
    """Return (R, t): the TRUE base<-sensor extrinsic for this cohort.
    x_base = R * p_sensor + t. drifted=False => aligned cohort (every
    sensor exactly at its nominal pose); drifted=True => LEFT/RIGHT carry
    their documented drift on top of the nominal mount."""
    mount = MOUNT[sensor_id]
    yaw = mount["yaw_deg"] * DEG2RAD
    R_nom = so3_exp((0.0, 0.0, yaw))
    t_nom = list(mount["pos"])
    if not drifted:
        return R_nom, t_nom
    d = DRIFT[sensor_id]
    omega = tuple(v * DEG2RAD for v in d["omega_deg"])
    R_true = mat3_mul(so3_exp(omega), R_nom)   # left-perturbation, 01.17 convention
    t_true = [t_nom[i] + d["t_m"][i] for i in range(3)]
    return R_true, t_true


# ---------------------------------------------------------------------------
# Surface samplers — each returns a list of (x, y, z, surface_id) WORLD
# (base-frame) points on a deterministic grid (not a random scatter): a
# shared grid is what lets the SAME physical world point be captured by
# multiple sensors (the overlap this whole project is about), rather than
# each sensor independently re-sampling the surface and never truly
# agreeing on a point (CLAUDE.md §1: teach the real mechanism, not a
# statistical stand-in for it).
# ---------------------------------------------------------------------------
def sample_ground():
    pts = []
    n = int(round(2 * GROUND_EXTENT / GROUND_SPACING))
    for i in range(n + 1):
        x = -GROUND_EXTENT + i * GROUND_SPACING
        for j in range(n + 1):
            y = -GROUND_EXTENT + j * GROUND_SPACING
            pts.append((x, y, 0.0, SURF_GROUND))
    return pts


def sample_wall(fixed_axis, fixed_value, free_range, z_range, spacing, surface_id):
    """fixed_axis: 'x' or 'y' -- the wall's constant coordinate."""
    pts = []
    nfree = int(round((free_range[1] - free_range[0]) / spacing))
    nz = int(round((z_range[1] - z_range[0]) / spacing))
    for i in range(nfree + 1):
        free = free_range[0] + i * spacing
        for k in range(nz + 1):
            z = z_range[0] + k * spacing
            if fixed_axis == "x":
                pts.append((fixed_value, free, z, surface_id))
            else:
                pts.append((free, fixed_value, z, surface_id))
    return pts


def sample_poles():
    pts = []
    for (cx, cy) in POLE_CENTERS:
        for r in range(POLE_RINGS):
            z = POLE_HEIGHT * (r + 0.5) / POLE_RINGS
            for a in range(POLE_RING_POINTS):
                theta = 2.0 * math.pi * a / POLE_RING_POINTS
                x = cx + POLE_RADIUS * math.cos(theta)
                y = cy + POLE_RADIUS * math.sin(theta)
                pts.append((x, y, z, SURF_POLE))
    return pts


def build_world_points():
    pts = []
    pts += sample_ground()
    pts += sample_wall("x", WALL_FRONT_X, WALL_Y_RANGE, WALL_Z_RANGE, WALL_SPACING, SURF_WALL_FRONT)
    pts += sample_wall("y", WALL_LEFT_Y, WALL_X_RANGE, WALL_Z_RANGE, WALL_SPACING, SURF_WALL_LEFT)
    pts += sample_wall("y", WALL_RIGHT_Y, WALL_X_RANGE, WALL_Z_RANGE, WALL_SPACING, SURF_WALL_RIGHT)
    pts += sample_wall("x", WALL_REAR_X, WALL_Y_RANGE, WALL_Z_RANGE, WALL_SPACING, SURF_WALL_REAR)
    pts += sample_poles()
    return pts


# ---------------------------------------------------------------------------
# visible_sensors — FOV/range membership test (kernels.cuh's documented
# simplification: base-frame azimuth from the sensor's NOMINAL mount
# origin, independent of cohort/drift -- see the file header). Returns the
# list of sensor ids that capture this world point.
# ---------------------------------------------------------------------------
def azimuth_deg(dx, dy):
    return math.degrees(math.atan2(dy, dx))


def visible_sensors(x, y):
    out = []
    for sid in (SENSOR_MAIN, SENSOR_LEFT, SENSOR_RIGHT):
        mx, my, _ = MOUNT[sid]["pos"]
        rng = math.hypot(x - mx, y - my)
        fov = FOV[sid]
        if rng > fov["max_range"]:
            continue
        if fov["az_min"] is None:   # MAIN: unrestricted azimuth
            out.append(sid)
            continue
        az = azimuth_deg(x - mx, y - my)
        if fov["az_min"] <= az <= fov["az_max"]:
            out.append(sid)
    return out


# ---------------------------------------------------------------------------
# write_cohort — express every (world point, capturing sensor) pair in the
# capturing sensor's OWN frame under this cohort's TRUE extrinsic, add
# range noise, and write one CSV row per (sensor, point).
# ---------------------------------------------------------------------------
def write_cohort(path: Path, world_points, assignment, drifted: bool, rng: random.Random):
    extr = {sid: sensor_extrinsic(sid, drifted) for sid in (SENSOR_MAIN, SENSOR_LEFT, SENSOR_RIGHT)}
    rows = []
    for (x, y, z, surface_id), sensors in zip(world_points, assignment):
        for sid in sensors:
            R, t = extr[sid]
            p_base = [x, y, z]
            p_sensor = mat3_transpose_vec(R, [p_base[i] - t[i] for i in range(3)])
            rng_m = math.sqrt(sum(c * c for c in p_sensor))
            if rng_m < 1e-6:
                continue  # degenerate: a point exactly at the sensor origin (never happens here)
            noise = rng.gauss(0.0, NOISE_SIGMA_M)   # 1-D range noise (see file header)
            scale = (rng_m + noise) / rng_m
            p_noisy = [c * scale for c in p_sensor]
            rows.append((sid, surface_id, p_noisy[0], p_noisy[1], p_noisy[2]))

    with open(path, "w", newline="") as f:
        f.write("# 02.16 multi-LiDAR sample -- columns: sensor_id,surface_id,x,y,z (meters, sensor-own frame)\n")
        f.write("# sensor_id: 0=main 1=left 2=right | surface_id: 0=ground 1=wall_front 2=wall_left 3=wall_right 4=wall_rear 5=pole\n")
        f.write(f"# cohort={'drifted' if drifted else 'aligned'} points={len(rows)} noise_sigma_m={NOISE_SIGMA_M}\n")
        for sid, surf, x, y, z in rows:
            f.write(f"{sid},{surf},{x:.6f},{y:.6f},{z:.6f}\n")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Generate the 02.16 multi-LiDAR aligned/drifted sample.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=str(Path(__file__).resolve().parent.parent / "data" / "sample"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    world_points = build_world_points()
    assignment = [visible_sensors(x, y) for (x, y, z, surf) in world_points]

    rng_aligned = random.Random(args.seed)
    rng_drifted = random.Random(args.seed + 1)   # independent noise stream per cohort (still deterministic)

    rows_aligned = write_cohort(out_dir / "aligned.csv", world_points, assignment, drifted=False, rng=rng_aligned)
    rows_drifted = write_cohort(out_dir / "drifted.csv", world_points, assignment, drifted=True, rng=rng_drifted)

    # ---- diagnostics (stdout only -- not part of any committed file) ------
    print(f"world points (pre-FOV):  {len(world_points)}")
    print(f"aligned.csv rows:        {len(rows_aligned)}")
    print(f"drifted.csv rows:        {len(rows_drifted)}")
    for sid in (SENSOR_MAIN, SENSOR_LEFT, SENSOR_RIGHT):
        for surf, name in [(SURF_GROUND, "ground"), (SURF_WALL_FRONT, "wall_front"),
                           (SURF_WALL_LEFT, "wall_left"), (SURF_WALL_RIGHT, "wall_right"),
                           (SURF_WALL_REAR, "wall_rear"), (SURF_POLE, "pole")]:
            n = sum(1 for r in rows_aligned if r[0] == sid and r[1] == surf)
            if n:
                print(f"  sensor={SENSOR_NAMES[sid]:5s} surface={name:10s} n={n}")


if __name__ == "__main__":
    main()
