#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample-data generator for 02.10
(FPFH descriptors + RANSAC global registration).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
-----------------------------------------------------------------
Global registration can only be graded honestly against a scene whose TRUE
relative pose between two scans is known exactly — not eyeballed. This
script builds ONE structured world (a room: floor + 4 walls + a box crate +
a cylindrical pillar — 02.01-style analytic-surface machinery, cited, reused
here for a multi-primitive ROOM instead of a single ground plane), samples it
densely in the WORLD frame (grid + jitter, 02.09's identical sampler
philosophy), and then "scans" it from TWO sensor poses related by a LARGE,
known rigid transform (140 deg yaw + 8 m translation — far outside any local
method's convergence basin; see THEORY.md "Where this sits in the real
world" and the icp_negative_control gate in main.cu, which measures this
honestly rather than asserting it).

Because both scans are SUBSETS of the same world point set, every physically
identical point that survives into both scans is known EXACTLY — no nearest-
neighbor guessing required. This ground-truth correspondence table is what
main.cu's descriptor_invariance gate uses (FPFH of the SAME physical point
computed independently in each scan's own local frame; THEORY.md derives WHY
that should be nearly identical) and what the registration_recovery /
icp_negative_control gates check the recovered pose against.

Visibility model (an honest, documented simplification — README/THEORY.md
"Limitations & honesty"): a world point is "seen" by a sensor pose if (a) it
is within MAX_RANGE_M of the sensor, and (b) its own analytic outward normal
faces generally toward the sensor (dot(normal, direction-to-sensor) > 0).
This reproduces the two real effects that create PARTIAL OVERLAP between two
LiDAR scans of the same scene — range limits and back-face rejection — but
it is NOT a full ray-caster: it does not model one object occluding another
behind it (e.g. the crate hiding a patch of wall). For this room's geometry
that gap rarely matters (the crate and pillar are small compared to the
walls they might shadow) and is the honest, stated cost of a script this
simple; a true ray-caster is project 11.01's job (GPU LiDAR simulator).

Two "pairs" are written, all from the SAME extreme 140deg/8m relative pose
(the RATIFIED SCOPE's negative-control transform), differing only in sensor
placement/range so their MEASURED overlap differs:
    pair0 — moderate overlap cohort ("the main case"), noise=0 (clean)
    pair1 — SAME geometry as pair0, moderate sensor noise added (the
            realistic case: this is what main.cu's headline
            registration_recovery / icp_negative_control gates run on)
    pair2 — a LOW-OVERLAP stress cohort (tighter range / more extreme
            placement), same noise level as pair1 — main.cu MEASURES and
            REPORTS this honestly; it is not required to succeed (see
            README "Limitations & honesty" and the low_overlap gate).
Overlap fractions are not hand-picked numbers pasted into a docstring — they
are MEASURED by this script from the actual visibility masks and printed
(and recorded in data/README.md from a real run), per CLAUDE.md's
never-fabricate rule.

Noise model: Gaussian, ALONG THE TRUE NORMAL only (identical justification
to 02.09's make_synthetic.py: dominant real LiDAR error is range/line-of-
sight noise, and for a surface seen close to head-on the line of sight is
close to the normal), sigma = 0 (pair0) or 10 mm (pair1, pair2) — drawn
INDEPENDENTLY per scan (a real sensor's noise on the same physical point
differs shot to shot; this script honors that instead of reusing one draw).

Usage
-----
    python make_synthetic.py                    # writes the committed sample
    python make_synthetic.py --out-dir DIR       # experiments; do not commit
"""

import argparse
import math
import struct
import sys
from pathlib import Path

# ===========================================================================
# Deterministic RNG: xorshift32 (stdlib-only, no `random` module — the repo's
# fixed-seed convention; identical algorithm to 02.01/02.05/02.09/08.01's
# device-side generators and their own make_synthetic.py scripts, cited not
# reinvented — see 02.09's module docstring for the full rationale).
# ===========================================================================
class Xorshift32:
    """32-bit xorshift PRNG (Marsaglia 2003)."""

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
        return (self.next_u32() >> 8) * (1.0 / 16777216.0) + (0.5 / 16777216.0)

    def uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.uniform01()

    def gauss(self) -> float:
        """Standard normal via Box-Muller (stdlib math only — see 02.09's
        make_synthetic.py for the identical, fully-documented approach)."""
        u1 = max(self.uniform01(), 1.0e-12)   # guard log(0)
        u2 = self.uniform01()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


DEFAULT_SEED = 42  # repo convention

# ===========================================================================
# Plain 3-vector / quaternion helpers (tuples of floats; no numpy).
# ===========================================================================
def v_add(a, b): return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
def v_sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def v_scale(a, s): return (a[0] * s, a[1] * s, a[2] * s)
def v_dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def v_norm(a): return math.sqrt(v_dot(a, a))
def v_normalize(a):
    n = v_norm(a)
    if n < 1.0e-12:
        return (0.0, 0.0, 1.0)
    return (a[0] / n, a[1] / n, a[2] / n)


def mat_apply(R, v):
    return (R[0] * v[0] + R[1] * v[1] + R[2] * v[2],
            R[3] * v[0] + R[4] * v[1] + R[5] * v[2],
            R[6] * v[0] + R[7] * v[1] + R[8] * v[2])


def mat_transpose(R):
    return (R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8])


def v_normalize4(q):
    n = math.sqrt(sum(c * c for c in q))
    return tuple(c / n for c in q) if n > 1e-12 else (1.0, 0.0, 0.0, 0.0)


# ===========================================================================
# THE SCENE — a room: floor + 4 walls + one box crate + one cylindrical
# pillar. Each surface primitive is sampled on a jittered grid (02.09's
# sampler philosophy) and every sample point carries its OWN analytic
# outward unit normal — used both for realistic along-normal sensor noise
# AND for this script's visibility test (see module docstring).
#
# Room extent: x,y in [-ROOM_HALF, ROOM_HALF], floor z=0, walls z in
# [0, WALL_HEIGHT_M]. The crate and pillar sit inside, off-center, so their
# sharp corners / constant-curvature surface give FPFH genuinely distinctive
# local geometry to lock onto — flat floor/wall patches are, by contrast,
# locally SELF-SIMILAR almost everywhere (the exact "aliasing" hazard
# 01.04/01.05's THEORY.md sections name for 2-D features, arriving in 3-D;
# README/THEORY.md cite this explicitly as the reason the ratio test can
# still admit occasional wrong correspondences on the walls/floor even
# though it screens most of them, and why RANSAC — not descriptor matching
# alone — is the step that actually earns robustness).
# ===========================================================================
ROOM_HALF_M    = 10.0
WALL_HEIGHT_M  = 3.0
GRID_SPACING_M = 0.42          # target world-sample spacing (tuned for ~1500-2500 pts/scan after visibility)
JITTER_FRAC    = 0.35          # +-fraction of GRID_SPACING_M, grid+jitter (near-uniform density, no hard aliasing)

CRATE_CENTER   = (3.0, -3.0, 0.0)
CRATE_HALF     = (0.9, 0.9, 0.6)   # half-extents (m); sits on the floor, top at z=1.2

PILLAR_CENTER_XY = (-4.0, 4.0)
PILLAR_RADIUS_M  = 0.45
PILLAR_HEIGHT_M  = WALL_HEIGHT_M

NOISE_SIGMA_CLEAN_M = 0.0
NOISE_SIGMA_MOD_M   = 0.010    # 10 mm — a realistic mid-range LiDAR range-noise figure


def sample_quad(rng, center, u_axis, v_axis, half_u, half_v, normal, spacing):
    """Grid+jitter sample one flat rectangular patch (center +- half_u along
    u_axis, +- half_v along v_axis), returning [(pos, normal), ...]. Shared
    by every planar surface below (floor, 4 walls, 5 crate faces) — one
    sampler, many call sites, per the repo's usual "single formula, many
    uses" style (e.g. 02.09's direction()/make_perp_basis())."""
    nu = max(1, int(round(2.0 * half_u / spacing)))
    nv = max(1, int(round(2.0 * half_v / spacing)))
    pts = []
    for i in range(nu):
        for j in range(nv):
            jx = rng.uniform(-JITTER_FRAC, JITTER_FRAC) * spacing
            jy = rng.uniform(-JITTER_FRAC, JITTER_FRAC) * spacing
            uc = -half_u + (i + 0.5) * (2.0 * half_u / nu) + jx
            vc = -half_v + (j + 0.5) * (2.0 * half_v / nv) + jy
            p = v_add(center, v_add(v_scale(u_axis, uc), v_scale(v_axis, vc)))
            pts.append((p, normal))
    return pts


def sample_cylinder_lateral(rng, center_xy, radius, height, spacing):
    """Grid+jitter sample the LATERAL surface of a vertical cylinder (caps
    excluded — the pillar's top/bottom are never LiDAR-visible from a
    sensor at typical scan height, so modeling them would only waste
    points). Circumference / height determine the (azimuth, z) grid size;
    each sample's normal is the outward RADIAL direction."""
    circumference = 2.0 * math.pi * radius
    n_az = max(8, int(round(circumference / spacing)))
    n_z = max(1, int(round(height / spacing)))
    pts = []
    for i in range(n_az):
        for j in range(n_z):
            jaz = rng.uniform(-JITTER_FRAC, JITTER_FRAC) * (2.0 * math.pi / n_az)
            jz = rng.uniform(-JITTER_FRAC, JITTER_FRAC) * spacing
            az = (i + 0.5) * (2.0 * math.pi / n_az) + jaz
            z = (j + 0.5) * (height / n_z) + jz
            nrm = (math.cos(az), math.sin(az), 0.0)
            p = (center_xy[0] + radius * math.cos(az), center_xy[1] + radius * math.sin(az), z)
            pts.append((p, nrm))
    return pts


def build_world(rng):
    """Assemble the whole room's WORLD-frame point set: floor, 4 walls, the
    5 visible crate faces (all but the bottom, which sits on the floor and
    is never seen), and the pillar's lateral surface. Returns a flat list
    of (pos_xyz, normal_xyz)."""
    pts = []

    # Floor: z=0, normal +z, spanning the room.
    pts += sample_quad(rng, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                       ROOM_HALF_M, ROOM_HALF_M, (0.0, 0.0, 1.0), GRID_SPACING_M)

    # 4 walls: inward-facing normals (the room is seen from the INSIDE).
    h = WALL_HEIGHT_M
    pts += sample_quad(rng, (-ROOM_HALF_M, 0.0, h / 2), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
                       ROOM_HALF_M, h / 2, (1.0, 0.0, 0.0), GRID_SPACING_M)     # x=-R wall, normal +x
    pts += sample_quad(rng, (ROOM_HALF_M, 0.0, h / 2), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
                       ROOM_HALF_M, h / 2, (-1.0, 0.0, 0.0), GRID_SPACING_M)    # x=+R wall, normal -x
    pts += sample_quad(rng, (0.0, -ROOM_HALF_M, h / 2), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0),
                       ROOM_HALF_M, h / 2, (0.0, 1.0, 0.0), GRID_SPACING_M)     # y=-R wall, normal +y
    pts += sample_quad(rng, (0.0, ROOM_HALF_M, h / 2), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0),
                       ROOM_HALF_M, h / 2, (0.0, -1.0, 0.0), GRID_SPACING_M)    # y=+R wall, normal -y

    # Crate: axis-aligned box, 5 outward faces (top + 4 sides); bottom omitted.
    cx, cy, cz0 = CRATE_CENTER
    hx, hy, hz = CRATE_HALF
    zc = cz0 + hz
    pts += sample_quad(rng, (cx, cy, cz0 + 2 * hz), (1, 0, 0), (0, 1, 0), hx, hy, (0, 0, 1), GRID_SPACING_M)   # top
    pts += sample_quad(rng, (cx - hx, cy, zc), (0, 1, 0), (0, 0, 1), hy, hz, (-1, 0, 0), GRID_SPACING_M)       # -x face
    pts += sample_quad(rng, (cx + hx, cy, zc), (0, 1, 0), (0, 0, 1), hy, hz, (1, 0, 0), GRID_SPACING_M)        # +x face
    pts += sample_quad(rng, (cx, cy - hy, zc), (1, 0, 0), (0, 0, 1), hx, hz, (0, -1, 0), GRID_SPACING_M)       # -y face
    pts += sample_quad(rng, (cx, cy + hy, zc), (1, 0, 0), (0, 0, 1), hx, hz, (0, 1, 0), GRID_SPACING_M)        # +y face

    # Pillar: lateral cylinder surface.
    pts += sample_cylinder_lateral(rng, PILLAR_CENTER_XY, PILLAR_RADIUS_M, PILLAR_HEIGHT_M, GRID_SPACING_M)

    return pts


# ===========================================================================
# Sensor placement — the RATIFIED negative-control relative pose: 140 deg
# yaw + 8 m translation between pose A and pose B. Both poses are fixed
# WORLD positions/orientations; a point's LOCAL (sensor-frame) coordinates
# are world coordinates expressed in that sensor's own frame: p_local =
# R_world_sensor^T * (p_world - sensor_pos) — i.e. T_sensor_world applied,
# the inverse of "where the sensor sits in the world" (CLAUDE.md paragraph
# 12's T_parent_child convention: R_world_sensor is "sensor expressed in
# world"; its transpose/inverse maps world points INTO the sensor frame).
# ===========================================================================
RELATIVE_YAW_DEG = 140.0
RELATIVE_TRANS_M = 8.0
RELATIVE_TRANS_HEADING_DEG = 35.0   # azimuth (world XY) of the 8 m translation

POSE_A_POS   = (-2.0, -2.0, 1.3)
POSE_A_YAW_DEG = 0.0

_dx = RELATIVE_TRANS_M * math.cos(math.radians(RELATIVE_TRANS_HEADING_DEG))
_dy = RELATIVE_TRANS_M * math.sin(math.radians(RELATIVE_TRANS_HEADING_DEG))
POSE_B_POS   = (POSE_A_POS[0] + _dx, POSE_A_POS[1] + _dy, POSE_A_POS[2])
POSE_B_YAW_DEG = POSE_A_YAW_DEG + RELATIVE_YAW_DEG

# Per-pair sensor range (pair0/pair1 share geometry; pair2 is the low-
# overlap stress cohort — a TIGHTER range on pose B is a physically honest
# way to shrink overlap: a shorter-range sensor variant, or a scan taken
# with more of the room already out of view).
MAX_RANGE_MAIN_M   = 13.0     # pair0 / pair1
MAX_RANGE_STRESS_M = 8.5      # pair2 (pose B only) — see build_pair()


def pose_rotation_matrix(yaw_deg):
    """World-frame rotation for a sensor pose: yaw about +z only (the
    sensor's mounting is level — pitch/roll = 0, the common ground-vehicle
    LiDAR mount assumption also used by 02.01/02.03)."""
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    return (c, -s, 0.0,
            s,  c, 0.0,
            0.0, 0.0, 1.0)


def world_to_sensor(p_world, pos, R_world_sensor):
    """p_local = R^T * (p_world - pos) — world point expressed in the
    sensor's own frame (+x forward, +y left, +z up at yaw=0, CLAUDE.md
    paragraph 12's body convention)."""
    d = v_sub(p_world, pos)
    Rt = mat_transpose(R_world_sensor)
    return mat_apply(Rt, d)


def visible_mask(world_pts, pos, R_world_sensor, max_range):
    """Boolean mask: which world points this sensor pose SEES (module
    docstring's range + back-face visibility model)."""
    mask = []
    for (p, n) in world_pts:
        d = v_sub(pos, p)                 # vector FROM the point TO the sensor
        r = v_norm(d)
        if r > max_range or r < 1e-6:
            mask.append(False)
            continue
        facing = v_dot(n, v_scale(d, 1.0 / r))
        mask.append(facing > 0.02)        # small positive margin: reject near-grazing self-shadowing
    return mask


def build_pair(rng, world_pts, max_range_b, noise_sigma_m, label):
    """Build one (source, target) scan pair from the shared world point
    set: source = pose A's visible subset, target = pose B's visible
    subset, EACH in its OWN sensor-local frame, with independent along-
    normal Gaussian noise. Returns (source_xyz, target_xyz, src_world_idx,
    tgt_world_idx, overlap_fraction) — overlap_fraction is MEASURED, not
    assumed."""
    R_a = pose_rotation_matrix(POSE_A_YAW_DEG)
    R_b = pose_rotation_matrix(POSE_B_YAW_DEG)

    mask_a = visible_mask(world_pts, POSE_A_POS, R_a, MAX_RANGE_MAIN_M)
    mask_b = visible_mask(world_pts, POSE_B_POS, R_b, max_range_b)

    n_a = sum(mask_a)
    n_b = sum(mask_b)
    n_both = sum(1 for a, b in zip(mask_a, mask_b) if a and b)
    n_union = sum(1 for a, b in zip(mask_a, mask_b) if a or b)
    overlap_fraction = (n_both / n_union) if n_union > 0 else 0.0

    src_xyz, src_world_idx = [], []
    tgt_xyz, tgt_world_idx = [], []
    for wi, ((p, n), va, vb) in enumerate(zip(world_pts, mask_a, mask_b)):
        if va:
            local = world_to_sensor(p, POSE_A_POS, R_a)
            n_local = mat_apply(mat_transpose(R_a), n)
            noisy = v_add(local, v_scale(n_local, noise_sigma_m * rng.gauss()))
            src_xyz.append(noisy); src_world_idx.append(wi)
        if vb:
            local = world_to_sensor(p, POSE_B_POS, R_b)
            n_local = mat_apply(mat_transpose(R_b), n)
            noisy = v_add(local, v_scale(n_local, noise_sigma_m * rng.gauss()))
            tgt_xyz.append(noisy); tgt_world_idx.append(wi)

    print(f"[make_synthetic] {label}: pose A sees {n_a} pts, pose B sees {n_b} pts, "
          f"overlap(|A n B|/|A u B|) = {overlap_fraction * 100.0:.1f}% (measured)")

    return src_xyz, tgt_xyz, src_world_idx, tgt_world_idx, overlap_fraction


def ground_truth_transform():
    """The TRUE T_target_source (02.06's naming convention, cited): apply
    this to a point in the SOURCE (pose A) frame to land it in the TARGET
    (pose B) frame. Derivation: p_world = R_a*p_src + pos_a = R_b*p_tgt +
    pos_b  =>  p_tgt = R_b^T*R_a*p_src + R_b^T*(pos_a - pos_b)."""
    R_a = pose_rotation_matrix(POSE_A_YAW_DEG)
    R_b = pose_rotation_matrix(POSE_B_YAW_DEG)
    Rb_t = mat_transpose(R_b)
    R = tuple(sum(Rb_t[i * 3 + k] * R_a[k * 3 + j] for k in range(3)) for i in range(3) for j in range(3))
    t = mat_apply(Rb_t, v_sub(POSE_A_POS, POSE_B_POS))
    return R, t


def matrix_to_quat(R):
    """Shepperd's method (robust matrix->quaternion) — used ONLY to write
    the ground-truth quaternion into pairs_meta.csv in a compact form;
    main.cu re-derives R from (t,q) the usual way (02.06's quat_to_matrix,
    cited)."""
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = R
    tr = m00 + m11 + m22
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif m00 > m11 and m00 > m22:
        S = math.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = math.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = math.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S
    return v_normalize4((w, x, y, z))


# ===========================================================================
# Binary sample format (see data/README.md for the field-by-field spec this
# mirrors): one file per cloud, "FPFHPAIR1" magic, then a point count, then
# flat xyz, then flat WORLD-INDEX (int32) — the world index is the ground-
# truth correspondence key (identical world index in the sibling file = the
# SAME physical point).
# ===========================================================================
def write_cloud_bin(path: Path, xyz, world_idx):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(xyz)
    with path.open('wb') as f:
        f.write(b'FPFHPAIR1')
        f.write(struct.pack('<i', n))
        flat = []
        for (x, y, z) in xyz:
            flat.extend((x, y, z))
        f.write(struct.pack(f'<{len(flat)}f', *flat))
        f.write(struct.pack(f'<{n}i', *world_idx))


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_out_dir = script_dir.parent / 'data' / 'sample'

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help=f'xorshift32 seed for byte-identical reproducibility (default {DEFAULT_SEED})')
    parser.add_argument('--out-dir', type=Path, default=default_out_dir,
                        help='output directory (default: ../data/sample/)')
    args = parser.parse_args()

    rng = Xorshift32(args.seed)
    world_pts = build_world(rng)
    print(f"[make_synthetic] world scene: {len(world_pts)} surface points "
          f"(floor+walls+crate+pillar), seed={args.seed}")

    R_true, t_true = ground_truth_transform()
    q_true = matrix_to_quat(R_true)
    print(f"[make_synthetic] TRUE T_target_source: t=({t_true[0]:.3f},{t_true[1]:.3f},{t_true[2]:.3f}) m, "
          f"yaw={RELATIVE_YAW_DEG:.1f} deg, |t|={v_norm(t_true):.3f} m")

    pairs = []
    s0, t0, si0, ti0, ov0 = build_pair(rng, world_pts, MAX_RANGE_MAIN_M, NOISE_SIGMA_CLEAN_M, 'pair0 (clean, main)')
    pairs.append(('pair0', s0, t0, si0, ti0, ov0, NOISE_SIGMA_CLEAN_M))
    s1, t1, si1, ti1, ov1 = build_pair(rng, world_pts, MAX_RANGE_MAIN_M, NOISE_SIGMA_MOD_M, 'pair1 (noisy, main)')
    pairs.append(('pair1', s1, t1, si1, ti1, ov1, NOISE_SIGMA_MOD_M))
    s2, t2, si2, ti2, ov2 = build_pair(rng, world_pts, MAX_RANGE_STRESS_M, NOISE_SIGMA_MOD_M, 'pair2 (noisy, low-overlap)')
    pairs.append(('pair2', s2, t2, si2, ti2, ov2, NOISE_SIGMA_MOD_M))

    meta_path = args.out_dir / 'pairs_meta.csv'
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with meta_path.open('w', newline='') as f:
        f.write("pair,n_source,n_target,overlap_fraction,noise_sigma_m,"
                "tx,ty,tz,qw,qx,qy,qz,relative_yaw_deg,relative_trans_m\n")
        for (name, s, t, si, ti, ov, sigma) in pairs:
            write_cloud_bin(args.out_dir / f'{name}_source.bin', s, si)
            write_cloud_bin(args.out_dir / f'{name}_target.bin', t, ti)
            f.write(f"{name},{len(s)},{len(t)},{ov:.6f},{sigma:.4f},"
                    f"{t_true[0]:.6f},{t_true[1]:.6f},{t_true[2]:.6f},"
                    f"{q_true[0]:.6f},{q_true[1]:.6f},{q_true[2]:.6f},{q_true[3]:.6f},"
                    f"{RELATIVE_YAW_DEG:.3f},{RELATIVE_TRANS_M:.3f}\n")

    print(f"[make_synthetic] wrote {meta_path} and {len(pairs)} pair(s) x (source,target) .bin to {args.out_dir}")
    print("[make_synthetic] all data SYNTHETIC (CLAUDE.md paragraph 8)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
