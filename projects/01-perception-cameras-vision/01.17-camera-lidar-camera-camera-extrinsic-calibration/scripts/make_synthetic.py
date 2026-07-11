#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample-data generator for 01.17
(Camera-LiDAR / camera-camera extrinsic calibration, batched
reprojection-error minimization).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
-----------------------------------------------------------------
Extrinsic calibration is exactly the kind of problem robotics can synthesize
with FULL ground truth: we pick a true T_camera_lidar (or T_camera2_camera1),
pick a set of target poses, and compute EXACT correspondences by forward
projection. main.cu never sees this script's numbers directly — it adds its
own sensor noise on top of the TRUE correspondences this file writes (see
kernels.cuh's "CORRESPONDENCE LAYOUT" note) so that one committed file backs
the zero-noise sanity gate (no noise added) and the noise-scaling gate
(three documented sigmas) alike.

This file is also, deliberately, a SECOND, INDEPENDENT implementation of the
pinhole projection formula src/kernels.cuh's residual_and_jacobian() shares
between the GPU and CPU C++ paths (different language, no code in common) —
reference_cpu.cpp's file header explains exactly which gate this independence
buys the project (the zero-noise sanity gate is a negative control against a
bug hiding in the shared C++ camera model).

What this script generates (all pure Python 3.12 stdlib — no numpy, per the
project's dependency policy: CUDA toolkit + C++17 stdlib only, and this
script mirrors that discipline on the Python side)
------------------------------------------------------------------------
Three correspondence files, each a planar 4-fiducial target observed across
V=12 target poses (48 correspondences), written as TRUE (noise-free) values:

  cam_lidar_diverse.csv   — camera-LiDAR pair, 12 poses spanning a wide range
                            of depth (1.5-4.0 m) AND orientation (up to ~50
                            degrees off boresight) relative to the LiDAR —
                            the well-conditioned cohort.
  cam_lidar_coplanar.csv  — camera-LiDAR pair, SAME ground-truth extrinsic,
                            but 12 poses held at nearly the SAME depth and
                            orientation (only small lateral/jitter motion) —
                            the DEGENERATE cohort this project's degeneracy
                            gate uses to show pose diversity, not view COUNT,
                            is what conditions the solve (echoing 01.16's
                            Zhang-calibration finding, cited in THEORY.md).
  cam_cam_diverse.csv     — camera-camera pair, 12 diverse poses relative to
                            camera 1 (see the file's own docstring below for
                            why this scenario has no source-point noise).

Plus one tiny synthetic grayscale background, cam_background.pgm (a plain
P5 PGM, 160x120 — matching kIntrinsics' image size in kernels.cuh exactly, so
projected pixel coordinates land in-frame), used by the demo's overlay
artifact (demo/out/overlay.ppm, the "money shot": LiDAR points reprojected
onto this background before vs. after calibration).

Usage
-----
    python make_synthetic.py                  # writes all four files, seed 42
    python make_synthetic.py --seed 7 --out-dir ../data/sample
"""

import argparse
import csv
import math
import random
from pathlib import Path

DEFAULT_SEED = 42  # CLAUDE.md paragraph 12: fixed seed, byte-identical output every run/machine

# ---------------------------------------------------------------------------
# Camera model — MUST match kernels.cuh's kImageWidth/Height, kFx/kFy/kCx/kCy
# exactly (this is the one piece of data this script and the C++ side must
# manually keep in lockstep, since Python cannot #include a .cuh header;
# data/README.md repeats this warning for future editors).
# ---------------------------------------------------------------------------
IMAGE_W, IMAGE_H = 160, 120
FX, FY, CX, CY = 154.0, 152.0, 80.0, 60.0

# Ground-truth extrinsics (see kernels.cuh "PARAMETERIZATION": omega is an
# so(3) axis-angle log-rotation, radians; R_gt = Exp(omega_gt)).
OMEGA_GT_CAM_LIDAR = (0.030, -0.050, 0.100)   # rad (~6.7 deg total rotation)
T_GT_CAM_LIDAR     = (0.060, -0.040, 0.030)   # m   (~8.1 cm baseline — close rig mounting)

OMEGA_GT_CAM_CAM = (0.010, 0.020, -0.010)     # rad (~2.5 deg — small stereo vergence)
T_GT_CAM_CAM     = (0.120, 0.001, -0.002)     # m   (~12 cm stereo baseline, mostly along X)

# Planar target: 4 retroreflector-style fiducials at the corners of a 30cm x
# 20cm rectangle, board frame origin at the board's own center, board plane
# is board-local Z=0 (so the board's own surface normal is its local Z axis).
BOARD_POINTS = [
    (-0.15, -0.10, 0.0),
    ( 0.15, -0.10, 0.0),
    ( 0.15,  0.10, 0.0),
    (-0.15,  0.10, 0.0),
]

NUM_VIEWS = 12


# ===========================================================================
# Minimal pure-Python 3-vector / 3x3-matrix / Rodrigues math — an
# INDEPENDENT reimplementation of the same formulas src/kernels.cuh's
# so3_exp()/mat3_vec()/mat3_mul() define for the C++ side (see this file's
# module docstring for why the independence matters). Row-major 3x3
# matrices as 9-tuples, matching the C++ layout exactly (so a learner
# comparing the two sees the same indexing).
# ===========================================================================

def mat3_vec(R, p):
    """out = R @ p, R row-major 9-tuple, p a 3-tuple."""
    return (
        R[0] * p[0] + R[1] * p[1] + R[2] * p[2],
        R[3] * p[0] + R[4] * p[1] + R[5] * p[2],
        R[6] * p[0] + R[7] * p[1] + R[8] * p[2],
    )


def mat3_mul(A, B):
    """out = A @ B, both row-major 9-tuples."""
    out = [0.0] * 9
    for r in range(3):
        for c in range(3):
            out[r * 3 + c] = sum(A[r * 3 + k] * B[k * 3 + c] for k in range(3))
    return tuple(out)


def so3_exp(omega):
    """Rodrigues' formula: R = Exp([omega]_x), the exact SO(3) exponential —
    the same closed form as kernels.cuh's so3_exp (reimplemented here, in
    Python, independently — see the module docstring)."""
    wx, wy, wz = omega
    theta = math.sqrt(wx * wx + wy * wy + wz * wz)
    S = (0.0, -wz, wy, wz, 0.0, -wx, -wy, wx, 0.0)   # skew([omega])
    if theta < 1e-8:
        return (1 + S[0], S[1], S[2], S[3], 1 + S[4], S[5], S[6], S[7], 1 + S[8])
    a = math.sin(theta) / theta
    b = (1 - math.cos(theta)) / (theta * theta)
    S2 = mat3_mul(S, S)
    ident = (1, 0, 0, 0, 1, 0, 0, 0, 1)
    return tuple(ident[i] + a * S[i] + b * S2[i] for i in range(9))


def pinhole_project(p_cam):
    """(u, v) = project p_cam (already in the DEST camera's optical frame:
    z-forward, x-right, y-down — kernels.cuh's documented convention)."""
    u = FX * p_cam[0] / p_cam[2] + CX
    v = FY * p_cam[1] / p_cam[2] + CY
    return (u, v)


def compose(R_a, t_a, R_b, t_b):
    """T_a * T_b as a single (R, t): applying the result to a point p gives
    the same answer as applying T_b then T_a. R_a,R_b row-major 9-tuples."""
    R = mat3_mul(R_a, R_b)
    t = mat3_vec(R_a, t_b)
    t = (t[0] + t_a[0], t[1] + t_a[1], t[2] + t_a[2])
    return R, t


# ===========================================================================
# Target-pose cohorts. Each returns a list of NUM_VIEWS (R_pose, t_pose)
# pairs: the board's pose EXPRESSED IN THE SOURCE SENSOR'S FRAME (lidar
# frame for the two camera-LiDAR cohorts; camera-1 frame for camera-camera).
# ===========================================================================

def sample_diverse_poses(rng, depth_range, lateral_xy, max_tilt_rad):
    """Wide depth + wide orientation spread — the well-conditioned cohort.
    depth_range: (min, max) meters along the source frame's +Z (forward).
    lateral_xy: (max_x, max_y) meters of lateral spread at each view.
    max_tilt_rad: each view's rotation is a random axis, angle uniform in
    [0, max_tilt_rad] — i.e. the board's own surface normal points in a
    substantially different direction from view to view, which is exactly
    the "pose diversity" 01.16's Zhang-calibration finding (cited in
    THEORY.md) says a well-conditioned calibration needs.
    """
    poses = []
    for _ in range(NUM_VIEWS):
        depth = rng.uniform(*depth_range)
        x = rng.uniform(-lateral_xy[0], lateral_xy[0])
        y = rng.uniform(-lateral_xy[1], lateral_xy[1])
        t = (x, y, depth)

        axis = [rng.gauss(0.0, 1.0) for _ in range(3)]
        norm = math.sqrt(sum(a * a for a in axis)) or 1.0
        angle = rng.uniform(0.0, max_tilt_rad)
        omega = tuple(a / norm * angle for a in axis)
        R = so3_exp(omega)
        poses.append((R, t))
    return poses


def sample_coplanar_poses(rng, depth, depth_jitter, lateral_xy, max_tilt_rad):
    """Near-constant depth AND near-constant orientation — the DEGENERATE
    cohort: across all 12 views, the 48 target points span almost no depth
    variation and almost no orientation variation, so the reprojection-error
    solve has little information to disambiguate certain rotation/
    translation components (THEORY.md "The math" derives the observability
    argument this cohort is built to demonstrate).
    """
    poses = []
    for _ in range(NUM_VIEWS):
        z = depth + rng.uniform(-depth_jitter, depth_jitter)
        x = rng.uniform(-lateral_xy[0], lateral_xy[0])
        y = rng.uniform(-lateral_xy[1], lateral_xy[1])
        t = (x, y, z)

        axis = [rng.gauss(0.0, 1.0) for _ in range(3)]
        norm = math.sqrt(sum(a * a for a in axis)) or 1.0
        angle = rng.uniform(0.0, max_tilt_rad)   # SMALL — near-identical orientation every view
        omega = tuple(a / norm * angle for a in axis)
        R = so3_exp(omega)
        poses.append((R, t))
    return poses


def build_correspondences(poses, R_gt, t_gt):
    """For each (view, board pose), transform the 4 board points into the
    source frame (via the pose), then into the dest camera frame (via the
    ground-truth extrinsic) and project — returns a flat list of
    (view, point_idx, p_src_xyz, u_true, v_true) rows, TRUE/noise-free."""
    rows = []
    for view, (R_pose, t_pose) in enumerate(poses):
        for pt_idx, p_board in enumerate(BOARD_POINTS):
            p_src = mat3_vec(R_pose, p_board)
            p_src = (p_src[0] + t_pose[0], p_src[1] + t_pose[1], p_src[2] + t_pose[2])

            p_dst = mat3_vec(R_gt, p_src)
            p_dst = (p_dst[0] + t_gt[0], p_dst[1] + t_gt[1], p_dst[2] + t_gt[2])
            u, v = pinhole_project(p_dst)

            rows.append((view, pt_idx, p_src, (u, v)))
    return rows


def write_correspondence_csv(path: Path, scenario: str, omega_gt, t_gt, rows, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        f.write(f"# SYNTHETIC data — generated by scripts/make_synthetic.py for project 01.17\n")
        f.write(f"# scenario: {scenario} ({NUM_VIEWS} views x {len(BOARD_POINTS)} fiducials = {len(rows)} correspondences)\n")
        f.write(f"# regenerate: python make_synthetic.py --seed {seed}\n")
        f.write("# OMEGA_GT/T_GT: ground-truth extrinsic, so3 log-rotation (rad) + translation (m) —\n")
        f.write("#   R_gt = Exp(OMEGA_GT) (kernels.cuh so3_exp); p_dest = R_gt * p_src + T_GT\n")
        f.write("# INTRINSICS: fx,fy,cx,cy (px) — must match kernels.cuh's kFx/kFy/kCx/kCy exactly\n")
        f.write("# columns: view,point,px_src_m,py_src_m,pz_src_m,u_true_px,v_true_px (all EXACT, noise-free —\n")
        f.write("#   main.cu adds sensor noise itself; see kernels.cuh 'CORRESPONDENCE LAYOUT')\n")
        w = csv.writer(f)
        w.writerow(["OMEGA_GT", f"{omega_gt[0]:.8f}", f"{omega_gt[1]:.8f}", f"{omega_gt[2]:.8f}"])
        w.writerow(["T_GT", f"{t_gt[0]:.8f}", f"{t_gt[1]:.8f}", f"{t_gt[2]:.8f}"])
        w.writerow(["INTRINSICS", f"{FX:.4f}", f"{FY:.4f}", f"{CX:.4f}", f"{CY:.4f}"])
        for view, pt_idx, p_src, uv in rows:
            w.writerow(["CORR", view, pt_idx,
                       f"{p_src[0]:.8f}", f"{p_src[1]:.8f}", f"{p_src[2]:.8f}",
                       f"{uv[0]:.4f}", f"{uv[1]:.4f}"])
    in_frame = sum(1 for _, _, _, uv in rows if 0.0 <= uv[0] <= IMAGE_W and 0.0 <= uv[1] <= IMAGE_H)
    print(f"[make_synthetic] wrote {len(rows)} correspondences to {path} "
         f"({in_frame}/{len(rows)} project inside the {IMAGE_W}x{IMAGE_H} frame)")


def make_background_pgm(path: Path, seed: int) -> None:
    """A tiny (160x120) synthetic grayscale 'camera image' — a soft vertical
    gradient (floor/wall backdrop) plus a faint grid, used only as the
    overlay artifact's background so the demo's 'money shot' (reprojected
    LiDAR points before/after calibration) has a plausible scene to sit on.
    Deterministic (seeded speckle) and P5 (binary) PGM to keep the committed
    file tiny — CLAUDE.md paragraph 8: kilobytes preferred."""
    rng = random.Random(seed ^ 0xC0FFEE)
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = bytearray(IMAGE_W * IMAGE_H)
    for y in range(IMAGE_H):
        # Vertical gradient: darker top (60), lighter bottom (190) — a
        # "floor recedes into a wall" cue, entirely synthetic.
        base = 60 + int(130 * (y / (IMAGE_H - 1)))
        for x in range(IMAGE_W):
            val = base
            if x % 20 == 0 or y % 20 == 0:      # faint grid lines every 20 px
                val = max(0, val - 25)
            val += rng.randint(-4, 4)            # a little speckle so it doesn't look flat-shaded
            pixels[y * IMAGE_W + x] = max(0, min(255, val))
    with path.open("wb") as f:
        f.write(f"P5\n{IMAGE_W} {IMAGE_H}\n255\n".encode("ascii"))
        f.write(bytes(pixels))
    print(f"[make_synthetic] wrote {IMAGE_W}x{IMAGE_H} background to {path} ({path.stat().st_size} bytes)")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_out_dir = script_dir.parent / "data" / "sample"

    parser = argparse.ArgumentParser(
        description="Generate the synthetic correspondence sets + background for project 01.17.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"RNG seed for byte-identical reproducibility (default {DEFAULT_SEED})")
    parser.add_argument("--out-dir", type=Path, default=default_out_dir,
                        help="output directory (default: ../data/sample)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    R_gt_cl = so3_exp(OMEGA_GT_CAM_LIDAR)
    R_gt_cc = so3_exp(OMEGA_GT_CAM_CAM)

    # --- camera-LiDAR, diverse cohort: depth 1.5-4.0 m, tilt up to 50 deg ---
    diverse_poses = sample_diverse_poses(rng, depth_range=(1.5, 4.0),
                                         lateral_xy=(0.5, 0.35),
                                         max_tilt_rad=math.radians(50.0))
    rows = build_correspondences(diverse_poses, R_gt_cl, T_GT_CAM_LIDAR)
    write_correspondence_csv(args.out_dir / "cam_lidar_diverse.csv", "camera-LiDAR diverse-pose cohort",
                             OMEGA_GT_CAM_LIDAR, T_GT_CAM_LIDAR, rows, args.seed)

    # --- camera-LiDAR, coplanar (degenerate) cohort: depth ~2.5 m +-5 cm,
    #     tilt up to 3 deg — same ground truth, deliberately ill-posed poses.
    coplanar_poses = sample_coplanar_poses(rng, depth=2.5, depth_jitter=0.05,
                                           lateral_xy=(0.4, 0.3),
                                           max_tilt_rad=math.radians(3.0))
    rows = build_correspondences(coplanar_poses, R_gt_cl, T_GT_CAM_LIDAR)
    write_correspondence_csv(args.out_dir / "cam_lidar_coplanar.csv", "camera-LiDAR coplanar (degenerate) cohort",
                             OMEGA_GT_CAM_LIDAR, T_GT_CAM_LIDAR, rows, args.seed)

    # --- camera-camera, diverse cohort: depth 1.0-3.0 m, tilt up to 45 deg ---
    cc_poses = sample_diverse_poses(rng, depth_range=(1.0, 3.0),
                                    lateral_xy=(0.4, 0.3),
                                    max_tilt_rad=math.radians(45.0))
    rows = build_correspondences(cc_poses, R_gt_cc, T_GT_CAM_CAM)
    write_correspondence_csv(args.out_dir / "cam_cam_diverse.csv", "camera-camera diverse-pose cohort",
                             OMEGA_GT_CAM_CAM, T_GT_CAM_CAM, rows, args.seed)

    # --- background for the overlay artifact ---
    make_background_pgm(args.out_dir / "cam_background.pgm", args.seed)


if __name__ == "__main__":
    main()
