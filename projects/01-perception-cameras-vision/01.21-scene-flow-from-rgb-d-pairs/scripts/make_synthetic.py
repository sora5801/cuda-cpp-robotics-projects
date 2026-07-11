#!/usr/bin/env python3
"""make_synthetic.py -- synthetic RGB-D pair generator for 01.21 (Scene flow from RGB-D pairs).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
-----------------------------------------------------------------
Scene flow needs an answer key no real sensor can hand you: the EXACT 3-D displacement of
every scene point, split into "caused by the camera moving" (ego-motion) vs. "caused by the
object itself moving" (independent motion). Only a renderer that KNOWS both motions before
drawing a single pixel can produce that. This script ray-casts a small scene (technique
follows project 01.07's ray-cast renderer and 01.18's camera/ground/box primitives, both
cited, reimplemented independently for this project's own geometry and motion model) from
TWO camera poses related by a KNOWN rigid ego-motion, with ONE box that additionally moves by
its own KNOWN independent rigid motion between the two exposures. It writes:

  frame0_rgb.ppm / frame1_rgb.ppm            camera images at t0 / t1 (2x2 supersampled)
  frame0_depth.bin / frame1_depth.bin        EXACT dense depth (Pcam.z), float32, sensor-noised
  static_frame1_rgb.ppm / static_frame1_depth.bin   NEGATIVE-CONTROL t1: same camera ego-motion,
                                                      the box does NOT move (reuses frame0_* as t0)
  truth_flow.bin        dense 2-D optical flow (u,v), float32, frame0 pixel grid
  truth_scene_flow.bin  dense 3-D "raw" scene flow (P2_cam1 - P1_cam0), float32 (see below)
  truth_mask.pgm        255 = pixel's frame0 ray hit the MOVING object, 0 = everything else

Frame convention (load-bearing -- re-derived in THEORY.md "The math")
-----------------------------------------------------------------------
Camera0 sits at CAM_POS0 with IDENTITY orientation (its optical axis defines "forward" for
the whole scene). Camera1 sits at CAM_POS0 + T_EGO with orientation R_EGO (a small yaw) --
this pair (R_EGO, T_EGO) is "how the camera itself moved" between exposures, in WORLD/BODY
coordinates (x-forward, y-left, z-up). Camera-frame (OPTICAL: z-forward, x-right, y-down)
coordinates of a world point X are "X minus the camera's own position, un-rotated into the
camera's own BODY axes, THEN permuted into OPTICAL axes by M = body_to_cam" -- so for a point
FIXED in the world, v_cam0 = M @ (X - CAM_POS0) and v_cam1 = M @ R_EGO^T @ (X - CAM_POS1).
Substituting X = CAM_POS0 + M^T @ v_cam0 (M is a pure permutation, so M^-1 = M^T) gives
    v_cam1 = (M @ R_EGO^T @ M^T) @ v_cam0 + M @ (-R_EGO^T @ T_EGO)
(CAM_POS0 cancels algebraically; see THEORY.md for the full step-by-step). So the rigid
transform
    T_gt = (R_gt, t_gt) = (M @ R_EGO^T @ M^T,  M @ (-R_EGO^T @ T_EGO))
is EXACTLY the transform the C++ pipeline's robust ego-motion fit is trying to recover from the
raw (P1 -> P2) correspondence field. The M-CONJUGATION here is not optional bookkeeping: R_EGO
is a rotation matrix expressed in BODY axes, and a rotation matrix's numbers are basis-
dependent -- applying R_EGO^T directly to OPTICAL-frame points (skipping the M @ (...) @ M^T
conjugation) silently rotates about the WRONG axis (this was root-caused during this project's
own build: a rotation about the body's "up" axis is a rotation about the camera's OWN "-y"
(down) axis once you permute into optical coordinates, i.e. a rotation mixing x_cam/z_cam, not
x_cam/y_cam -- verified against an independent Horn fit on the exact (noise-free) ground truth
before trusting the constants below).

The MOVING object additionally translates by T_OBJ (world frame, no rotation -- keeps the
"object motion" ground truth a single 3-vector, not a second rotation-estimation problem).
Composing through the same algebra (THEORY.md works it in full) gives: for an object point,
    P2 = T_gt(P1) + c_gt,      c_gt = M @ (R_EGO^T @ T_OBJ)
a CONSTANT offset added on top of the background's own T_gt -- this is exactly why residual
segmentation works (every object pixel's residual after applying T_gt is ~c_gt, a fixed
nonzero vector; every static pixel's residual is ~sensor noise only) and exactly what the
object_motion [info] gate compares its own recovered offset against.

R_gt/t_gt/c_gt are computed once below and PRINTED so kernels.cuh's constexpr mirror (manually
synchronized, same convention as 01.18's kTCameraLidar -- change one, change both) can be
copy-pasted from this script's own console output rather than hand-derived a second time.

Determinism: xorshift32 only (never Python's `random`), matching 01.18's convention.

Usage
-----
    python make_synthetic.py                      # defaults: seed=42, writes into ../data/sample/
"""

import argparse
import math
import struct
from pathlib import Path

# ===========================================================================
# Camera / image constants -- MUST match src/kernels.cuh exactly (manually
# synchronized across the Python/CUDA-C++ boundary, see module docstring).
# ===========================================================================
IMG_W, IMG_H = 128, 96
FX, FY, CX, CY = 118.0, 116.0, 64.0, 48.0
MAX_DEPTH_M = 16.0
INVALID_DEPTH = -1.0

CAM_POS0 = (0.0, 0.0, 1.4)          # world meters; camera0 optical axis = world +x, no tilt

# Ego-motion: camera1 = camera0 rotated by THETA_EGO_DEG (yaw, about world +z / "up") and
# translated by T_EGO (world meters) -- an illustrative "robot drives forward while turning
# slightly" step. Sized to keep frame-to-frame optical flow comfortably above the noise floor
# for a teaching demo (README's rate/latency honesty note: a real 30 Hz step would be smaller
# and would need correspondingly finer-grained validation, not a different method).
THETA_EGO_DEG = 3.0
T_EGO = (0.09, 0.0, 0.0)

# The one independently moving object: a textured box that translates by T_OBJ (world meters,
# ROTATION-FREE on purpose -- keeps the "recovered vs. truth" object-motion gate a single,
# unambiguous 3-vector comparison instead of a second rotation-estimation problem).
OBJ_CENTER0 = (5.5, -0.8, 0.35)
OBJ_HALF = (0.35, 0.35, 0.35)
T_OBJ = (0.0, 0.30, 0.0)

DEPTH_NOISE_A_M = 0.0015     # sensor floor noise, meters (electronics/quantization)
DEPTH_NOISE_B = 0.00015      # 1/m -- range^2 term (disparity-quantization model, THEORY.md derives it;
                              # sized so the noise stays modest even at this scene's ~9 m max depth --
                              # see kernels.cuh's kSegThresholdKSigma comment for the budget this was tuned against)

DEFAULT_SEED = 42

# ===========================================================================
# xorshift32 -- portable deterministic PRNG (01.18's convention, cited). Every
# random choice in this script (surface texture, depth noise) routes through
# this so the committed sample is bit-identical across machines.
# ===========================================================================
MASK32 = 0xFFFFFFFF


def xorshift32(x: int) -> int:
    x &= MASK32
    x ^= (x << 13) & MASK32
    x ^= (x >> 17)
    x ^= (x << 5) & MASK32
    return x & MASK32


def cell_hash01(ix: int, iy: int, iz: int, seed: int) -> float:
    """Deterministic hash of an integer 3-D lattice cell -> float in [0,1). Two xorshift32
    rounds mixing (ix,iy,iz,seed) -- extends 01.18's cell_hash01 (2-D) to 3 inputs so ONE
    texture function can paint the ground (a z=0 slice), the wall (an x=const slice) and every
    box face (whichever axis is const on that face) with the same lattice, cited throughout."""
    x = (ix * 1103515245 + iy * 12345 + iz * 2246822519 + seed * 2654435761) & MASK32
    x = xorshift32(x)
    x = xorshift32(x)
    return (x & 0xFFFFFF) / float(0x1000000)


def uniform01(state: int):
    state = xorshift32(state)
    v = ((state >> 8) & 0xFFFFFF) / float(0x1000000) + (0.5 / 0x1000000)
    return state, v


def gaussian(state: int, sigma: float):
    """One N(0,sigma^2) draw via Box-Muller (01.18's construction, cited)."""
    state, u1 = uniform01(state)
    state, u2 = uniform01(state)
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return state, sigma * z


def value_noise3(x: float, y: float, z: float, cell: float, seed: int) -> float:
    """Trilinear-blended 3-D value noise: hash the 8 corners of the lattice cell containing
    (x,y,z) and smoothstep-blend them. Sampled AT a surface hit point, one of the three
    lattice axes is (locally) constant on a flat face, so this single function gives every
    axis-aligned surface in the scene (ground, wall, box faces) a natural mottled texture
    with no per-surface special-casing -- the fine gradient this project's dense optical flow
    needs almost everywhere (a flat, untextured face would starve Lucas-Kanade of the spatial
    gradient its normal-equations solve divides by, THEORY.md's aperture-problem section)."""
    gx, gy, gz = x / cell, y / cell, z / cell
    ix0, iy0, iz0 = math.floor(gx), math.floor(gy), math.floor(gz)
    tx, ty, tz = gx - ix0, gy - iy0, gz - iz0
    sx, sy, sz = tx * tx * (3 - 2 * tx), ty * ty * (3 - 2 * ty), tz * tz * (3 - 2 * tz)

    def h(dx, dy, dz):
        return cell_hash01(ix0 + dx, iy0 + dy, iz0 + dz, seed)

    c00 = h(0, 0, 0) * (1 - sx) + h(1, 0, 0) * sx
    c10 = h(0, 1, 0) * (1 - sx) + h(1, 1, 0) * sx
    c01 = h(0, 0, 1) * (1 - sx) + h(1, 0, 1) * sx
    c11 = h(0, 1, 1) * (1 - sx) + h(1, 1, 1) * sx
    c0 = c00 * (1 - sy) + c10 * sy
    c1 = c01 * (1 - sy) + c11 * sy
    return c0 * (1 - sz) + c1 * sz


# body_to_cam / cam_to_body: world/body axes (x-forward,y-left,z-up) <-> camera OPTICAL axes
# (z-forward,x-right,y-down) -- identical permutation to 01.18's kTCameraLidar derivation, cited.
def body_to_cam(v):
    x, y, z = v
    return (-y, -z, x)


def cam_to_body(v):
    xc, yc, zc = v
    return (zc, -xc, -yc)


def yaw_matrix(deg: float):
    """3x3 rotation about world +z ("up") by `deg` degrees -- row-major flat 9-tuple,
    R[r*3+c]. This is R_EGO: camera1's orientation expressed in world axes."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return (c, -s, 0.0,
            s, c, 0.0,
            0.0, 0.0, 1.0)


def mat_T_vec(R, v):
    """R^T @ v for a row-major 3x3 R -- un-rotating a world vector into the frame R describes."""
    return (R[0] * v[0] + R[3] * v[1] + R[6] * v[2],
            R[1] * v[0] + R[4] * v[1] + R[7] * v[2],
            R[2] * v[0] + R[5] * v[1] + R[8] * v[2])


def mat_vec(R, v):
    return (R[0] * v[0] + R[1] * v[1] + R[2] * v[2],
            R[3] * v[0] + R[4] * v[1] + R[5] * v[2],
            R[6] * v[0] + R[7] * v[1] + R[8] * v[2])


def vsub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vadd(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


R_EGO = yaw_matrix(THETA_EGO_DEG)
CAM_POS1 = vadd(CAM_POS0, T_EGO)

# ---------------------------------------------------------------------------
# Ground-truth rigid transforms (THEORY.md "The math" derives all three; the
# module docstring above derives the M-conjugation this section applies).
# Printed at the end of main() so main.cu's constexpr mirror can be
# copy-pasted rather than hand-derived a second time (manually-synchronized
# contract, see module docstring).
#
# M -- the body_to_cam() permutation, as a matrix: x_cam=-y_body, y_cam=
# -z_body, z_cam=x_body (row-major, M[r*3+c]). Applying body_to_cam() to the
# 3 standard basis vectors and reading off the result IS this matrix -- a
# cheap, hard-to-get-wrong way to turn the existing (tested) function into
# its matrix form without retyping the permutation a second time by hand.
# ---------------------------------------------------------------------------
def _matrix_of(fn):
    cols = [fn((1.0, 0.0, 0.0)), fn((0.0, 1.0, 0.0)), fn((0.0, 0.0, 1.0))]
    return tuple(cols[c][r] for r in range(3) for c in range(3))


M_BODY_TO_CAM = _matrix_of(body_to_cam)


def mat_mat(A, B):
    """A @ B for row-major flat 3x3 matrices."""
    return tuple(sum(A[r * 3 + k] * B[k * 3 + c] for k in range(3)) for r in range(3) for c in range(3))


def mat_transpose(A):
    return tuple(A[c * 3 + r] for r in range(3) for c in range(3))


R_GT_BODY = mat_transpose(R_EGO)                                     # R_EGO^T, BODY axes
T_GT_BODY = tuple(-x for x in mat_T_vec(R_EGO, T_EGO))                # -R_EGO^T @ T_EGO, BODY axes
C_GT_BODY = mat_vec(R_GT_BODY, T_OBJ)                                 # R_gt_body @ T_OBJ, BODY axes

# Conjugate into camera OPTICAL-frame coordinates (see module docstring for
# why this conjugation is load-bearing, not cosmetic): R_gt = M R_gt_body M^T.
R_GT = mat_mat(mat_mat(M_BODY_TO_CAM, R_GT_BODY), mat_transpose(M_BODY_TO_CAM))
T_GT = mat_vec(M_BODY_TO_CAM, T_GT_BODY)
C_GT = mat_vec(M_BODY_TO_CAM, C_GT_BODY)


# ===========================================================================
# Scene -- ground plane + a back wall + two static boxes + ONE moving box.
# Every surface uses value_noise3 for texture (see that function's docstring
# for why: dense optical flow needs gradient almost everywhere).
# ===========================================================================
GROUND_BASE = (95.0, 90.0, 82.0)
WALL_BASE = (120.0, 130.0, 140.0)
WALL_X = 9.0
WALL_HALF_Y = 8.0
# Tall enough that the wall fills the ENTIRE vertical field of view at
# WALL_X (1.4 camera height + 9*tan(22.5 deg half-VFOV) ~= 5.1 m, +margin) --
# no wall/sky boundary is visible anywhere in frame. An early version of
# this scene had a visible wall-top/sky seam a few rows from the image top;
# # Lucas-Kanade's brightness-constancy assumption genuinely breaks at a
# seam like that (sky and wall do not correspond frame-to-frame the way a
# continuous textured surface does), and it was a measurable source of bad
# flow. Removing the seam from view is the honest fix -- not a threshold
# tuned to paper over it (THEORY.md "Numerical considerations" discusses
# occlusion/appearance-change boundaries as LK's genuine, expected limit).
WALL_Z_TOP = 6.5

STATIC_BOXES = [
    {"name": "near_static_box", "center": (3.2, 1.6, 0.5), "half": (0.45, 0.45, 0.5), "base": (150.0, 70.0, 60.0)},
    {"name": "far_static_box", "center": (9.5, 2.4, 0.7), "half": (0.55, 0.55, 0.7), "base": (60.0, 130.0, 90.0)},
]
MOVING_BOX_BASE = (210.0, 150.0, 40.0)   # a saturated, visually distinct "mover"

LIGHT_DIR = (0.35, 0.30, 0.887)
_ll = math.sqrt(sum(c * c for c in LIGHT_DIR))
LIGHT_DIR = tuple(c / _ll for c in LIGHT_DIR)
AMBIENT, DIFFUSE = 0.42, 0.58


def shade(color, normal):
    ndotl = max(0.0, normal[0] * LIGHT_DIR[0] + normal[1] * LIGHT_DIR[1] + normal[2] * LIGHT_DIR[2])
    factor = AMBIENT + DIFFUSE * ndotl
    return tuple(max(0.0, min(255.0, c * factor)) for c in color)


def textured_color(base, hit, cell, amplitude, seed):
    n = value_noise3(hit[0], hit[1], hit[2], cell, seed)
    scale = 1.0 + (n * 2.0 - 1.0) * amplitude
    return tuple(max(0.0, min(255.0, c * scale)) for c in base)


# ---------------------------------------------------------------------------
# Ray/object intersection -- plane + AABB slab test, the same technique
# family 01.07/01.18 use (cited), reimplemented independently for this
# scene. Every intersect_* returns (s, normal, base_color, obj_id) or None;
# obj_id identifies WHICH object was hit (None = ground/wall) so the
# ground-truth pass below knows whether to apply T_OBJ before reprojecting.
# ---------------------------------------------------------------------------
MOVING_OBJ_ID = "moving_box"


def intersect_ground(O, D):
    if D[2] >= -1e-9:
        return None
    s = -O[2] / D[2]
    if s <= 1e-6:
        return None
    hit = (O[0] + s * D[0], O[1] + s * D[1], 0.0)
    # Cell size chosen so the texture stays well ABOVE the Nyquist limit
    # after the flow pyramid's 2x area-average downsample, even at this
    # surface's typical depth (THEORY.md "Numerical considerations" derives
    # the px-per-cycle budget this and the other textured surfaces below
    # are sized against — a too-fine cell here was an early bug that made
    # Lucas-Kanade alias and diverge on far, fine-textured surfaces).
    color = textured_color(GROUND_BASE, hit, 0.55, 0.28, 1001)
    return s, (0.0, 0.0, 1.0), color, None


def intersect_wall(O, D):
    if D[0] <= 1e-9:
        return None   # wall sits AHEAD of the camera at x=WALL_X > 0: only forward-pointing (+x) rays can reach it
    s = (WALL_X - O[0]) / D[0]
    if s <= 1e-6:
        return None
    hit = (WALL_X, O[1] + s * D[1], O[2] + s * D[2])
    if abs(hit[1]) > WALL_HALF_Y or hit[2] < 0.0 or hit[2] > WALL_Z_TOP:
        return None
    color = textured_color(WALL_BASE, hit, 0.90, 0.22, 2002)   # see intersect_ground's Nyquist-budget note
    return s, (-1.0, 0.0, 0.0), color, None


def intersect_box(O, D, center, half, base, seed, obj_id):
    lo = tuple(center[i] - half[i] for i in range(3))
    hi = tuple(center[i] + half[i] for i in range(3))
    tmin, tmax, hit_axis, hit_sign = -1e18, 1e18, -1, 1.0
    for ax in range(3):
        o, d = O[ax], D[ax]
        if abs(d) < 1e-12:
            if o < lo[ax] or o > hi[ax]:
                return None
            continue
        t0, t1 = (lo[ax] - o) / d, (hi[ax] - o) / d
        sign = 1.0
        if t0 > t1:
            t0, t1, sign = t1, t0, -1.0
        if t0 > tmin:
            tmin, hit_axis, hit_sign = t0, ax, sign
        if t1 < tmax:
            tmax = t1
    if tmin > tmax or tmax < 1e-6:
        return None
    s = tmin if tmin > 1e-6 else tmax
    normal = [0.0, 0.0, 0.0]
    normal[hit_axis] = -1.0 if D[hit_axis] > 0 else 1.0
    hit = (O[0] + s * D[0], O[1] + s * D[1], O[2] + s * D[2])
    color = textured_color(base, hit, 0.28, 0.35, seed)   # see intersect_ground's Nyquist-budget note
    return s, tuple(normal), color, obj_id


def trace_ray(O, D, obj_center):
    """Cast one ray against the WHOLE scene (obj_center = the moving box's CURRENT center for
    this render -- t0 and t1-dynamic pass different values, t1-static passes the t0 value).
    Returns (color, depth_or_None, obj_id_or_None)."""
    best = None   # (s, normal, color, obj_id)
    for hit in (intersect_ground(O, D), intersect_wall(O, D)):
        if hit is not None and (best is None or hit[0] < best[0]):
            best = hit
    for i, b in enumerate(STATIC_BOXES):
        hit = intersect_box(O, D, b["center"], b["half"], b["base"], 3100 + i, b["name"])
        if hit is not None and (best is None or hit[0] < best[0]):
            best = hit
    hit = intersect_box(O, D, obj_center, OBJ_HALF, MOVING_BOX_BASE, 4200, MOVING_OBJ_ID)
    if hit is not None and (best is None or hit[0] < best[0]):
        best = hit

    if best is None:
        up = max(0.0, min(1.0, D[2] / (abs(D[2]) + 1.0) + 0.5))
        return (150 + 60 * up, 175 + 55 * up, 225), None, None
    s, normal, color, obj_id = best
    return shade(color, normal), s, obj_id


def camera_ray_cam_frame(u, v):
    """Unnormalized camera-OPTICAL-frame ray direction for pixel (u,v): z-component == 1, so
    the ray parameter `s` returned by trace_ray IS the pixel's depth (Pcam.z) directly."""
    return ((u - CX) / FX, (v - CY) / FY, 1.0)


def render(cam_pos, R_cam, obj_center, seed, state):
    """Ray-cast one full frame from `cam_pos` with world orientation `R_cam` (row-major 3x3;
    identity for camera0, R_EGO for camera1), the moving box at `obj_center`. Returns
    (rgb bytearray, depth list, obj_id-per-pixel list, new PRNG state)."""
    rgb = bytearray(IMG_W * IMG_H * 3)
    depth = [INVALID_DEPTH] * (IMG_W * IMG_H)
    obj_ids = [None] * (IMG_W * IMG_H)

    for v in range(IMG_H):
        for u in range(IMG_W):
            d_cam = camera_ray_cam_frame(u + 0.5, v + 0.5)   # pixel-CENTER convention (matches kernels.cuh backprojection)
            d_body = cam_to_body(d_cam)
            d_world = mat_vec(R_cam, d_body)
            _, s, obj_id = trace_ray(cam_pos, d_world, obj_center)
            idx = v * IMG_W + u
            if s is not None and 0.0 < s <= MAX_DEPTH_M:
                state, noise = gaussian(state, DEPTH_NOISE_A_M + DEPTH_NOISE_B * s * s)
                depth[idx] = max(0.02, s + noise)
                obj_ids[idx] = obj_id

            r_acc = g_acc = b_acc = 0.0
            for sy in (0.25, 0.75):
                for sx in (0.25, 0.75):
                    d_cam2 = camera_ray_cam_frame(u + sx, v + sy)
                    d_world2 = mat_vec(R_cam, cam_to_body(d_cam2))
                    color, _, _ = trace_ray(cam_pos, d_world2, obj_center)
                    r_acc += color[0]; g_acc += color[1]; b_acc += color[2]
            rgb[idx * 3 + 0] = int(max(0, min(255, r_acc / 4.0)))
            rgb[idx * 3 + 1] = int(max(0, min(255, g_acc / 4.0)))
            rgb[idx * 3 + 2] = int(max(0, min(255, b_acc / 4.0)))

    return rgb, depth, obj_ids, state


def compute_truth(depth0, obj_ids0):
    """Exact dense ground truth for the DYNAMIC pair, computed directly from frame0's own ray
    geometry -- never from frame1's rendered/possibly-occluded depth (see module docstring:
    this is the honest, non-circular way to grade the algorithm). Returns (flow_uv list of
    (u,v) or None, scene_flow_xyz list of (x,y,z) or None, mask list of 0/1)."""
    flow = [None] * (IMG_W * IMG_H)
    sflow = [None] * (IMG_W * IMG_H)
    mask = [0] * (IMG_W * IMG_H)

    for v in range(IMG_H):
        for u in range(IMG_W):
            idx = v * IMG_W + u
            d0 = depth0[idx]
            if d0 == INVALID_DEPTH:
                continue   # sky: no truth
            # Re-trace the EXACT (noise-free) camera0-frame point from the pixel-center ray
            # geometry (depth0 already carries sensor noise; using the noisy depth for the
            # TRUTH would conflate "truth" with "one noisy sample").
            d_cam = camera_ray_cam_frame(u + 0.5, v + 0.5)
            d_body = cam_to_body(d_cam)
            is_mover = (obj_ids0[idx] == MOVING_OBJ_ID)
            _, s_exact, _ = trace_ray(CAM_POS0, d_body, OBJ_CENTER0)
            if s_exact is None:
                continue
            P1 = (s_exact * d_cam[0], s_exact * d_cam[1], s_exact * d_cam[2])   # exact camera0-frame point
            world_hit0 = vadd(CAM_POS0, (s_exact * d_body[0], s_exact * d_body[1], s_exact * d_body[2]))
            world_hit1 = vadd(world_hit0, T_OBJ) if is_mover else world_hit0

            rel = vsub(world_hit1, CAM_POS1)
            d_body1 = mat_T_vec(R_EGO, rel)          # un-rotate into camera1's own body axes
            d_cam1 = body_to_cam(d_body1)             # camera1 OPTICAL-frame coordinates
            if d_cam1[2] <= 1e-6:
                continue   # behind camera1: no correspondence exists there
            P2 = d_cam1
            u1 = FX * d_cam1[0] / d_cam1[2] + CX
            v1 = FY * d_cam1[1] / d_cam1[2] + CY

            flow[idx] = (u1 - (u + 0.5), v1 - (v + 0.5))
            sflow[idx] = vsub(P2, P1)
            mask[idx] = 1 if is_mover else 0

    return flow, sflow, mask


def write_ppm(path: Path, rgb: bytearray) -> None:
    with open(path, "wb") as f:
        f.write(f"P6\n{IMG_W} {IMG_H}\n255\n".encode("ascii"))
        f.write(bytes(rgb))


def write_f32(path: Path, values) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}f", *values))


def write_pgm(path: Path, gray: bytearray) -> None:
    with open(path, "wb") as f:
        f.write(f"P5\n{IMG_W} {IMG_H}\n255\n".encode("ascii"))
        f.write(bytes(gray))


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_out = script_dir.parent / "data" / "sample"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", type=Path, default=default_out)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    state = args.seed

    print("[make_synthetic] rendering frame0 (t0, shared by both the dynamic and negative-control pairs)...")
    rgb0, depth0, obj_ids0, state = render(CAM_POS0, (1, 0, 0, 0, 1, 0, 0, 0, 1), OBJ_CENTER0, args.seed, state)
    write_ppm(args.out / "frame0_rgb.ppm", rgb0)
    write_f32(args.out / "frame0_depth.bin", depth0)

    print("[make_synthetic] rendering frame1 (t1, DYNAMIC pair: camera moves AND the box moves)...")
    obj_center1_dynamic = vadd(OBJ_CENTER0, T_OBJ)
    rgb1, depth1, obj_ids1, state = render(CAM_POS1, R_EGO, obj_center1_dynamic, args.seed + 1, state)
    write_ppm(args.out / "frame1_rgb.ppm", rgb1)
    write_f32(args.out / "frame1_depth.bin", depth1)

    print("[make_synthetic] rendering static_frame1 (NEGATIVE CONTROL: camera moves, box stays put)...")
    rgb1s, depth1s, _obj_ids1s, state = render(CAM_POS1, R_EGO, OBJ_CENTER0, args.seed + 2, state)
    write_ppm(args.out / "static_frame1_rgb.ppm", rgb1s)
    write_f32(args.out / "static_frame1_depth.bin", depth1s)
    # frame0 is IDENTICAL for both pairs (the object has not moved yet at t0 either way) --
    # the negative-control pair reuses frame0_rgb.ppm/frame0_depth.bin directly (main.cu loads
    # the same two files for both runs), so no static_frame0_* files are written (avoids a
    # byte-for-byte duplicate committed asset, CLAUDE.md paragraph 8's "tiny sample" spirit).

    print("[make_synthetic] computing exact dense ground truth (2-D flow, 3-D scene flow, moving mask)...")
    flow, sflow, mask = compute_truth(depth0, obj_ids0)
    flow_flat = []
    sflow_flat = []
    n_flow_valid = 0
    for i in range(IMG_W * IMG_H):
        if flow[i] is None:
            flow_flat += [0.0, 0.0]
            sflow_flat += [0.0, 0.0, 0.0]
        else:
            flow_flat += [flow[i][0], flow[i][1]]
            sflow_flat += list(sflow[i])
            n_flow_valid += 1
    write_f32(args.out / "truth_flow.bin", flow_flat)
    write_f32(args.out / "truth_scene_flow.bin", sflow_flat)
    # truth validity is IMPLICIT: a pixel has truth iff frame0_depth.bin is not INVALID_DEPTH
    # there (checked directly, see module docstring) -- no separate validity file needed.
    mask_pgm = bytearray(255 if m else 0 for m in mask)
    write_pgm(args.out / "truth_mask.pgm", mask_pgm)

    n_obj = sum(mask)
    n_sky0 = sum(1 for d in depth0 if d == INVALID_DEPTH)
    print(f"[make_synthetic] frame0: {IMG_W * IMG_H - n_sky0}/{IMG_W * IMG_H} pixels hit the scene "
          f"({100.0 * (IMG_W * IMG_H - n_sky0) / (IMG_W * IMG_H):.1f}%), {n_obj} pixels "
          f"({100.0 * n_obj / (IMG_W * IMG_H):.1f}%) are the moving object")
    print(f"[make_synthetic] dense truth: {n_flow_valid}/{IMG_W * IMG_H} pixels have a valid "
          f"flow/scene-flow correspondence (rest are sky or ran off-frame in camera1)")

    print(f"[make_synthetic] ground truth transforms (mirror these into kernels.cuh as constexpr, "
          f"see that file's header):")
    print(f"[make_synthetic]   R_gt (row-major) = {tuple(round(x, 8) for x in R_GT)}")
    print(f"[make_synthetic]   t_gt             = {tuple(round(x, 8) for x in T_GT)}")
    print(f"[make_synthetic]   c_gt (obj offset)= {tuple(round(x, 8) for x in C_GT)}")
    print(f"[make_synthetic]   |t_gt| = {math.sqrt(sum(x * x for x in T_GT)):.6f} m, "
          f"ego rotation = {THETA_EGO_DEG:.3f} deg, |T_OBJ| = {math.sqrt(sum(x * x for x in T_OBJ)):.6f} m")

    print("[make_synthetic] done. All arrays labeled SYNTHETIC; regenerate with "
          f"`python make_synthetic.py --seed {args.seed}`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
