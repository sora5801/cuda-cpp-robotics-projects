#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample-data generator for project 02.07
   (NDT scan matching (Autoware-style map localizer)).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
-----------------------------------------------------------------
NDT (like 02.06's ICP) needs a MAP, a SCAN, and a KNOWN ground-truth pose
between them to be verifiable at all -- a real LiDAR recording never comes
with an exact answer key. This script builds a small structured "building"
(an L-shaped corridor opening into a room, deliberately corridor-shaped so
the classic "sliding along a hallway" degeneracy the project's THEORY.md
discusses is actually present in the data, not just asserted), simulates a
16-beam LiDAR scan of it from a KNOWN sensor pose, and hands NDT and ICP the
job of recovering that pose from a family of perturbed initial guesses.

Determinism (paragraph 12): every draw comes from a local reimplementation
of the repo's xorshift32 + Box-Muller generator (08.01/01.17's exact
construction, cited in the code below) so the committed files are
byte-for-byte reproducible from this script alone, AND use the identical
named algorithm the C++ side (main.cu) would use if it needed its own RNG
(it does not: this script precomputes and commits the perturbation cohort,
so main.cu never needs randomness at all -- one less place determinism
could silently break).

What gets written (see ../data/README.md for the byte-exact format spec)
--------------------------------------------------------------------------
  map.bin              the MAP point cloud (dense survey of the building)
  scan_main.bin         one FULL-resolution scan from the true pose, WITH
                        outliers -- used for the single-registration
                        verification run and the before/after artifact
  scan_cohort.bin        a REDUCED-resolution scan (same true pose), WITH
                        outliers -- used for the many-trial basin/
                        convergence/accuracy/outlier-robustness gates
  scan_cohort_clean.bin  the SAME cohort scan, outlier fraction forced to 0,
                        SAME beam directions and range noise draws --
                        isolates the outlier's effect for the
                        outlier_robustness gate (paired comparison)
  icp_target.bin         a voxel-averaged DOWNSAMPLE of the map, used as
                        the compact ICP contrast baseline's target cloud
  cohort.csv             the committed table of perturbed initial poses
                        (magnitude bin + full Rigid3) every registration
                        trial main.cu runs starts from
  meta.csv               ground-truth pose + generation parameters

Usage
-----
    python make_synthetic.py                      # defaults: seed=42, writes ../data/sample/
    python make_synthetic.py --seed 7 --out DIR
"""

import argparse
import csv
import math
import struct
from pathlib import Path

# ===========================================================================
# xorshift32 + Box-Muller — 08.01/01.17's EXACT construction (cited),
# reimplemented here in Python so this script uses the SAME NAMED algorithm
# the C++ side of this repository standardizes on, per this project's data
# generation convention (stdlib-only, no numpy).
# ===========================================================================
class Xorshift32:
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF
        if self.state == 0:
            self.state = 1   # xorshift32 is degenerate at state 0 -- never seed with it

    def next_u32(self) -> int:
        x = self.state
        x = (x ^ (x << 13)) & 0xFFFFFFFF
        x = (x ^ (x >> 17)) & 0xFFFFFFFF
        x = (x ^ (x << 5)) & 0xFFFFFFFF
        self.state = x
        return x

    def uniform01(self) -> float:
        """(0,1] -- never exactly 0, safe for log() inside gauss()."""
        return (self.next_u32() >> 8) * (1.0 / 16777216.0) + (0.5 / 16777216.0)

    def uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.uniform01()

    def gauss(self, sigma: float) -> float:
        """One N(0, sigma^2) draw, Box-Muller (08.01's exact formula)."""
        u1 = self.uniform01()
        u2 = self.uniform01()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return sigma * z


# ===========================================================================
# Scene geometry (meters, SI, right-handed, CLAUDE.md paragraph 12) — MUST
# match kernels.cuh's kMapOriginX/Y/Z and kMapSizeX/Y/Z (the grid bounding
# box is sized to contain this scene with margin; see that header's comment).
#
# An L-shaped building: a narrow CORRIDOR (the project's deliberate
# degeneracy axis -- THEORY.md "physics-first" explains why sliding along a
# corridor is nearly unobservable from two parallel walls alone) opening
# into a wider ROOM with a PILLAR (the feature that fully constrains the
# corridor's sliding direction once a scan reaches far enough to see it).
# All primitives are axis-aligned rectangles, so ray-plane intersection and
# uniform-on-area sampling are both closed-form (02.06's exact "why
# axis-aligned rectangles" reasoning, cited).
# ===========================================================================
CORRIDOR_X = (0.0, 10.0)
CORRIDOR_Y = (-1.5, 1.5)
ROOM_X = (10.0, 16.0)
ROOM_Y = (-4.0, 4.0)
WALL_Z = (0.0, 2.5)
PILLAR_CENTER = (13.0, 0.0, 0.5)
PILLAR_HALF = (0.3, 0.3, 0.5)

# Each rectangle: (axis, coord, ax1, (min1,max1), ax2, (min2,max2)).
# axis/ax1/ax2 are 0=x, 1=y, 2=z. A point on this rectangle has
# p[axis]=coord, p[ax1] in (min1,max1), p[ax2] in (min2,max2).
PX0, PX1 = PILLAR_CENTER[0] - PILLAR_HALF[0], PILLAR_CENTER[0] + PILLAR_HALF[0]
PY0, PY1 = PILLAR_CENTER[1] - PILLAR_HALF[1], PILLAR_CENTER[1] + PILLAR_HALF[1]
PZ0, PZ1 = 0.0, PILLAR_CENTER[2] + PILLAR_HALF[2]

RECTANGLES = [
    # floor: corridor, then room
    (2, 0.0, 0, CORRIDOR_X, 1, CORRIDOR_Y),
    (2, 0.0, 0, ROOM_X, 1, ROOM_Y),
    # corridor side walls
    (1, CORRIDOR_Y[0], 0, CORRIDOR_X, 2, WALL_Z),
    (1, CORRIDOR_Y[1], 0, CORRIDOR_X, 2, WALL_Z),
    # room side walls + far wall
    (1, ROOM_Y[0], 0, ROOM_X, 2, WALL_Z),
    (1, ROOM_Y[1], 0, ROOM_X, 2, WALL_Z),
    (0, ROOM_X[1], 1, ROOM_Y, 2, WALL_Z),
    # transition stub walls where the corridor (narrow) meets the room (wide)
    (0, ROOM_X[0], 1, (ROOM_Y[0], CORRIDOR_Y[0]), 2, WALL_Z),
    (0, ROOM_X[0], 1, (CORRIDOR_Y[1], ROOM_Y[1]), 2, WALL_Z),
    # pillar, 5 faces (bottom omitted -- rests on the floor, unscannable)
    (0, PX1, 1, (PY0, PY1), 2, (PZ0, PZ1)),
    (0, PX0, 1, (PY0, PY1), 2, (PZ0, PZ1)),
    (1, PY1, 0, (PX0, PX1), 2, (PZ0, PZ1)),
    (1, PY0, 0, (PX0, PX1), 2, (PZ0, PZ1)),
    (2, PZ1, 0, (PX0, PX1), 1, (PY0, PY1)),
]

# Ground-truth sensor pose: standing in the corridor, level, facing +x down
# the hallway (yaw=0 => R = identity) -- a realistic robot-mounted-LiDAR
# height and heading. R=I is a SIMPLIFICATION (README "Limitations"): it
# keeps the scene generator's raycasting closed-form (world frame == sensor
# frame for direction purposes) while the OPTIMIZER is never told this --
# every cohort trial perturbs full 6-DOF (translation AND yaw) away from it,
# so recovering R=I is a genuine, unaided 6-DOF estimation, not a freebie.
TRUE_T = (5.0, 0.0, 1.2)
TRUE_R = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)   # row-major identity

# LiDAR model: 16 channels spanning +-15 deg elevation (a VLP-16-like
# geometry -- the catalog bullet's "16-beam" number), range gated, isotropic
# range noise, and a documented TRUE outlier fraction (dynamic-object
# stand-ins -- the reason NDT's d1/d2 mixture assumes a uniform outlier
# density at all, kernels.cuh's file header names this explicitly).
CHANNELS = 16
ELEV_MIN_DEG, ELEV_MAX_DEG = -15.0, 15.0
RANGE_MIN_M, RANGE_MAX_M = 0.3, 16.0
RANGE_NOISE_SIGMA_M = 0.02
TRUE_OUTLIER_FRACTION = 0.05

AZIMUTH_STEPS_MAIN = 360     # scan_main.bin -- the verification/artifact scan
AZIMUTH_STEPS_COHORT = 90    # scan_cohort*.bin -- the many-trial gates' scan

MAP_N_POINTS = 40000
ICP_TARGET_LEAF_M = 0.5      # voxel-average downsample leaf for the ICP contrast's target cloud

# Multi-resolution voxel leaves (MUST match kernels.cuh kLeafCoarse/kLeafFine).
LEAF_COARSE_M = 2.0
LEAF_FINE_M = 1.0

# Perturbation cohort: translation-magnitude bins (m) with a 1:1 paired yaw
# magnitude (deg) -- 01.17's exact "deliberately large enough to actually
# find a basin boundary" reasoning (cited): if every bin converged 100% of
# the time the sweep would teach nothing about basin SIZE.
#
# COHORT_TRIALS_PER_BIN was raised from 15 to 40 during this project's
# finisher pass (THEORY.md "numerical considerations" tells the story in
# full): a lead review found the smallest-perturbation bin's 15-trial
# sample too small to trust -- individual trials landing near this scene's
# corridor-degenerate axis (the same axis STAGE I's degenerate_axis report
# names) could swing a 15-trial bin's measured rate by 1-2 trials = 6.7-
# 13.3 percentage points on their own. Re-measuring at 40 trials/bin (and,
# during tuning, spot-checked at 60) gave a materially more STABLE number
# (65-68% across sample sizes, vs. anywhere from 60-67% at n=15) -- the
# achievable rate did not fundamentally change, but the MEASUREMENT became
# trustworthy rather than a single small sample's luck.
COHORT_TRANS_BINS_M = [0.2, 0.5, 0.8, 1.2, 1.6, 2.0]
COHORT_YAW_BINS_DEG = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
COHORT_TRIALS_PER_BIN = 40


# ---------------------------------------------------------------------------
# Ray-rectangle intersection: closed-form (axis-aligned plane + 2-D bounds
# check), the same style 02.06's scene sampler uses for uniform-on-area
# sampling, applied here to raycasting instead. Returns the hit distance t
# (range_min <= t <= range_max) or None.
# ---------------------------------------------------------------------------
def intersect_ray_rect(origin, direction, rect, range_min, range_max):
    axis, coord, ax1, (lo1, hi1), ax2, (lo2, hi2) = rect
    d_axis = direction[axis]
    if abs(d_axis) < 1.0e-9:
        return None
    t = (coord - origin[axis]) / d_axis
    if t < range_min or t > range_max:
        return None
    p1 = origin[ax1] + t * direction[ax1]
    if not (lo1 <= p1 <= hi1):
        return None
    p2 = origin[ax2] + t * direction[ax2]
    if not (lo2 <= p2 <= hi2):
        return None
    return t


def raycast_scene(origin, direction) -> float:
    """Nearest valid hit range among ALL scene rectangles, or None. Taking
    the MINIMUM valid t across every rectangle (including the pillar's own
    faces) is what makes the pillar correctly OCCLUDE the far wall behind
    it -- no separate visibility/occlusion pass is needed for convex box
    obstacles union'd with open planes (THEORY.md does not belabor this; it
    is a standard raycasting fact, noted here for the curious reader)."""
    best = None
    for rect in RECTANGLES:
        t = intersect_ray_rect(origin, direction, rect, RANGE_MIN_M, RANGE_MAX_M)
        if t is not None and (best is None or t < best):
            best = t
    return best


def rect_area(rect) -> float:
    _, _, _, (lo1, hi1), _, (lo2, hi2) = rect
    return (hi1 - lo1) * (hi2 - lo2)


def sample_point_on_rect(rng: Xorshift32, rect):
    axis, coord, ax1, (lo1, hi1), ax2, (lo2, hi2) = rect
    p = [0.0, 0.0, 0.0]
    p[axis] = coord
    p[ax1] = rng.uniform(lo1, hi1)
    p[ax2] = rng.uniform(lo2, hi2)
    return tuple(p)


def sample_map(rng: Xorshift32, n_total: int) -> list:
    """Uniform-on-area sampling across every RECTANGLES entry, point budget
    proportional to area (02.06's exact density-balancing approach, cited)
    so a learner plotting the cloud sees an even scatter, not a denser
    floor and a sparse pillar. The LAST rectangle absorbs the rounding
    remainder so the total is exact."""
    areas = [rect_area(r) for r in RECTANGLES]
    total_area = sum(areas)
    counts = [round(n_total * a / total_area) for a in areas]
    counts[-1] += n_total - sum(counts)   # exact total

    pts = []
    for rect, n in zip(RECTANGLES, counts):
        for _ in range(max(0, n)):
            pts.append(sample_point_on_rect(rng, rect))
    return pts


# ---------------------------------------------------------------------------
# 16-beam LiDAR simulation — the catalog bullet's "16-beam machinery",
# reduced-scope relative to 11.01's full GPU BVH mesh raycaster (cited: that
# project's job is raycasting an arbitrary triangle mesh on the GPU; this
# scene is a handful of axis-aligned rectangles, so a closed-form analytic
# intersection replaces the BVH entirely -- correct, and far simpler, at
# this scene's scale). Beam layout follows 11.01's channel-major convention
# (cited): beam = channel * azimuth_steps + azimuth_idx.
#
# Per-beam draw order (fixed, so scan_cohort.bin and scan_cohort_clean.bin
# can share the SAME true-surface geometry and noise while differing ONLY
# in the outlier decision -- see generate_scan()'s docstring):
#   1. is_outlier decision (uniform01 < outlier_fraction)
#   2. range noise (gaussian)
#   3. IF outlier: an extra uniform-depth draw (the "wrong-depth return"
#      outlier model -- a beam that returned a plausible range, just not
#      the true surface's -- the simplest honest stand-in for a dynamic
#      object or spurious multipath return crossing the beam)
# ---------------------------------------------------------------------------
def beam_direction(channel: int, azimuth_idx: int, azimuth_steps: int):
    elev_deg = ELEV_MIN_DEG + channel * (ELEV_MAX_DEG - ELEV_MIN_DEG) / (CHANNELS - 1)
    az_deg = azimuth_idx * 360.0 / azimuth_steps
    elev = math.radians(elev_deg)
    az = math.radians(az_deg)
    ce = math.cos(elev)
    return (ce * math.cos(az), ce * math.sin(az), math.sin(elev))


def generate_scan(azimuth_steps: int, outlier_fraction: float,
                   rng_outlier: Xorshift32, rng_noise: Xorshift32, rng_outlier_depth: Xorshift32):
    """One full sweep, CHANNELS x azimuth_steps beams, channel-major order.
    Returns a list of (x,y,z) points in the SENSOR-LOCAL frame (origin at
    the sensor, NOT offset by TRUE_T -- applying T_map_scan=(TRUE_R,TRUE_T)
    to these points reproduces their true MAP-frame position; this is
    exactly the data layout the registration algorithms are handed).

    Called with outlier_fraction=0.0 and the SAME rng_outlier/rng_noise/
    rng_outlier_depth starting states as the WITH-outliers call to build
    the "clean" paired scan for the outlier_robustness gate.

    A real bug this project's own outlier_robustness gate exposed during
    the finisher pass (THEORY.md "numerical considerations" tells the
    story): an EARLIER version drew the outlier's "wrong-depth" replacement
    from rng_noise -- the SAME stream every inlier's Gaussian noise also
    reads from. Even though the clean pass (outlier_fraction=0.0) never
    itself takes the is_outlier branch, the WITH-outliers pass consumed an
    EXTRA rng_noise draw every time it did -- desynchronizing rng_noise's
    internal xorshift32 STATE between the two passes from the first
    outlier onward, so every inlier noise draw AFTER that point differed
    between "clean" and "with outliers" even though both are nominally
    scoring the SAME true surface. The two scans were never actually a
    paired, single-variable-toggled comparison past beam ~20 (1/0.05) --
    they were two SEPARATE noisy scans that happened to share a few dozen
    early draws, which is exactly the kind of confound that can flip an
    A/B measurement's sign by chance. The fix: give the outlier depth draw
    its OWN independent stream (rng_outlier_depth) so consuming it can
    never perturb rng_noise's state -- now EVERY inlier beam's noise draw
    is bit-for-bit identical between the clean and with-outliers scans,
    and the only thing that can differ, beam for beam, is whether that
    beam's return was overridden by an outlier."""
    pts = []
    for c in range(CHANNELS):
        for a in range(azimuth_steps):
            direction = beam_direction(c, a, azimuth_steps)
            true_range = raycast_scene(TRUE_T, direction)

            is_outlier = rng_outlier.uniform01() < outlier_fraction   # draw ALWAYS (stream alignment)
            noise = rng_noise.gauss(RANGE_NOISE_SIGMA_M)              # draw ALWAYS (stream alignment) -- NEVER touched by the outlier branch below

            if is_outlier:
                # Wrong-depth return: a plausible range along the SAME beam
                # direction, uncorrelated with the true surface. Drawn from
                # its OWN stream (rng_outlier_depth), not rng_noise -- see
                # this function's docstring for the desync bug this fixes.
                depth = rng_outlier_depth.uniform(RANGE_MIN_M, RANGE_MAX_M)
                pts.append((direction[0] * depth, direction[1] * depth, direction[2] * depth))
                continue

            if true_range is None:
                continue   # no surface in range along this beam -- no return, honestly dropped

            r = true_range + noise
            if r < RANGE_MIN_M:
                r = RANGE_MIN_M   # clamp: noise should not produce a negative/behind-sensor range
            pts.append((direction[0] * r, direction[1] * r, direction[2] * r))
    return pts


# ---------------------------------------------------------------------------
# Binary cloud format — 02.06's exact "PC01" format (cited): magic, uint32
# little-endian count, then N*(float32 x,y,z) little-endian, meters.
# ---------------------------------------------------------------------------
def write_cloud_bin(path: Path, pts: list) -> None:
    flat = []
    for p in pts:
        flat.extend(p)
    with path.open("wb") as f:
        f.write(b"PC01")
        f.write(struct.pack("<I", len(pts)))
        f.write(struct.pack("<%df" % len(flat), *flat))


def downsample_grid_average(pts: list, leaf: float) -> list:
    """Simple dict-keyed voxel-average downsample (Python-side ONLY -- an
    independent, throwaway implementation used purely to build the ICP
    contrast's target cloud; it shares no code with kernels.cuh's
    voxel_index()/NDT voxel builder, deliberately, since this file's output
    is DATA, not part of the algorithm under test)."""
    buckets = {}
    for x, y, z in pts:
        key = (math.floor(x / leaf), math.floor(y / leaf), math.floor(z / leaf))
        b = buckets.get(key)
        if b is None:
            buckets[key] = [x, y, z, 1]
        else:
            b[0] += x; b[1] += y; b[2] += z; b[3] += 1
    out = []
    for key in sorted(buckets.keys()):
        sx, sy, sz, n = buckets[key]
        out.append((sx / n, sy / n, sz / n))
    return out


# ---------------------------------------------------------------------------
# Perturbation cohort: for each (trans_bin, yaw_bin) pair (paired 1:1 by
# index, COHORT_TRANS_BINS_M[i] <-> COHORT_YAW_BINS_DEG[i]), draw
# COHORT_TRIALS_PER_BIN initial guesses: a random XY direction at the bin's
# translation magnitude (z held fixed -- README "Limitations": roll/pitch/
# z are not perturbed, a deliberate scope reduction since the sensor's true
# height/level orientation would realistically come from a wheel/IMU-based
# prior, not the LiDAR match itself) composed with a random-sign yaw offset
# at the bin's magnitude -- a LEFT/world-frame perturbation of TRUE_R,
# matching kernels.cuh's retract() convention (R_init = Exp(omega)*R_true).
# ---------------------------------------------------------------------------
def yaw_matrix(yaw_rad: float):
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return (c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)


def mat3_mul(a, b):
    out = [0.0] * 9
    for r in range(3):
        for c in range(3):
            out[r * 3 + c] = sum(a[r * 3 + k] * b[k * 3 + c] for k in range(3))
    return tuple(out)


def build_cohort(rng: Xorshift32):
    rows = []
    trial_id = 0
    for bin_idx, (trans_mag, yaw_mag_deg) in enumerate(zip(COHORT_TRANS_BINS_M, COHORT_YAW_BINS_DEG)):
        for _ in range(COHORT_TRIALS_PER_BIN):
            phi = rng.uniform(0.0, 2.0 * math.pi)
            dx = trans_mag * math.cos(phi)
            dy = trans_mag * math.sin(phi)
            yaw_sign = 1.0 if rng.uniform01() < 0.5 else -1.0
            dyaw = math.radians(yaw_mag_deg) * yaw_sign

            R_init = mat3_mul(yaw_matrix(dyaw), TRUE_R)   # LEFT perturbation, retract()'s convention
            t_init = (TRUE_T[0] + dx, TRUE_T[1] + dy, TRUE_T[2])

            rows.append([trial_id, bin_idx, f"{trans_mag:.4f}", f"{yaw_mag_deg:.4f}",
                        *[f"{v:.10f}" for v in R_init], *[f"{v:.10f}" for v in t_init]])
            trial_id += 1
    return rows


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_out = script_dir.parent / "data" / "sample"

    parser = argparse.ArgumentParser(
        description="Generate the synthetic NDT map/scan/cohort data for project 02.07.")
    parser.add_argument("--seed", type=int, default=42, help="base RNG seed (default 42)")
    parser.add_argument("--out", type=Path, default=default_out, help="output directory")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Independent RNG streams, fixed order (02.06's discipline, cited):
    # each gets its own seed offset so re-ordering one draw sequence never
    # perturbs another.
    rng_map = Xorshift32(args.seed + 0)
    rng_outlier_main = Xorshift32(args.seed + 1)
    rng_noise_main = Xorshift32(args.seed + 2)
    rng_outlier_cohort = Xorshift32(args.seed + 3)
    rng_noise_cohort = Xorshift32(args.seed + 4)
    rng_outlier_cohort_clean = Xorshift32(args.seed + 3)   # SAME seed/state as rng_outlier_cohort (stream alignment)
    rng_noise_cohort_clean = Xorshift32(args.seed + 4)     # SAME seed/state as rng_noise_cohort (stream alignment)
    rng_cohort_perturb = Xorshift32(args.seed + 5)
    # Outlier "wrong-depth" replacement draws get their OWN streams (seed+6
    # for main, seed+7 for cohort), separate from rng_noise_* -- fixes a
    # real stream-desync bug generate_scan()'s docstring documents in full:
    # consuming an extra rng_noise draw only on the outlier branch used to
    # silently desynchronize the clean/with-outliers scan pair from the
    # first outlier onward, corrupting the outlier_robustness gate's A/B
    # comparison. rng_outlier_depth_cohort_clean shares rng_outlier_depth_
    # cohort's seed for the same "independent but aligned" reason as the
    # other _clean streams above, even though the clean pass never actually
    # draws from it (outlier_fraction=0.0 never takes that branch).
    rng_outlier_depth_main = Xorshift32(args.seed + 6)
    rng_outlier_depth_cohort = Xorshift32(args.seed + 7)
    rng_outlier_depth_cohort_clean = Xorshift32(args.seed + 7)

    print("[make_synthetic] sampling map...")
    map_pts = sample_map(rng_map, MAP_N_POINTS)
    write_cloud_bin(args.out / "map.bin", map_pts)
    print(f"[make_synthetic] map.bin: {len(map_pts)} points [SYNTHETIC]")

    print("[make_synthetic] raycasting scan_main (full resolution, with outliers)...")
    scan_main = generate_scan(AZIMUTH_STEPS_MAIN, TRUE_OUTLIER_FRACTION, rng_outlier_main, rng_noise_main, rng_outlier_depth_main)
    write_cloud_bin(args.out / "scan_main.bin", scan_main)
    print(f"[make_synthetic] scan_main.bin: {len(scan_main)} points "
          f"(of {CHANNELS * AZIMUTH_STEPS_MAIN} beams cast) [SYNTHETIC]")

    print("[make_synthetic] raycasting scan_cohort (reduced resolution, with outliers)...")
    scan_cohort = generate_scan(AZIMUTH_STEPS_COHORT, TRUE_OUTLIER_FRACTION, rng_outlier_cohort, rng_noise_cohort, rng_outlier_depth_cohort)
    write_cloud_bin(args.out / "scan_cohort.bin", scan_cohort)
    print(f"[make_synthetic] scan_cohort.bin: {len(scan_cohort)} points "
          f"(of {CHANNELS * AZIMUTH_STEPS_COHORT} beams cast) [SYNTHETIC]")

    print("[make_synthetic] raycasting scan_cohort_clean (reduced resolution, outlier-free, paired)...")
    scan_cohort_clean = generate_scan(AZIMUTH_STEPS_COHORT, 0.0, rng_outlier_cohort_clean, rng_noise_cohort_clean, rng_outlier_depth_cohort_clean)
    write_cloud_bin(args.out / "scan_cohort_clean.bin", scan_cohort_clean)
    print(f"[make_synthetic] scan_cohort_clean.bin: {len(scan_cohort_clean)} points [SYNTHETIC]")

    print("[make_synthetic] downsampling map for the ICP contrast target...")
    icp_target = downsample_grid_average(map_pts, ICP_TARGET_LEAF_M)
    write_cloud_bin(args.out / "icp_target.bin", icp_target)
    print(f"[make_synthetic] icp_target.bin: {len(icp_target)} points (leaf={ICP_TARGET_LEAF_M} m) [SYNTHETIC]")

    print("[make_synthetic] building perturbation cohort...")
    cohort_rows = build_cohort(rng_cohort_perturb)
    cohort_path = args.out / "cohort.csv"
    with cohort_path.open("w", newline="", encoding="utf-8") as f:
        f.write("# SYNTHETIC perturbed-initial-pose cohort for project 02.07's basin/convergence/accuracy gates.\n")
        f.write(f"# regenerate: python make_synthetic.py --seed {args.seed}\n")
        f.write("# columns: trial_id,bin_index,magnitude_trans_m,magnitude_yaw_deg,"
                "r00,r01,r02,r10,r11,r12,r20,r21,r22,tx,ty,tz\n")
        f.write("# R,t is the INITIAL GUESS T_map_scan (row-major R, t in meters) every registration trial starts from.\n")
        writer = csv.writer(f)
        for row in cohort_rows:
            writer.writerow(row)
    print(f"[make_synthetic] cohort.csv: {len(cohort_rows)} trials "
          f"({len(COHORT_TRANS_BINS_M)} bins x {COHORT_TRIALS_PER_BIN} trials)")

    meta_path = args.out / "meta.csv"
    with meta_path.open("w", newline="", encoding="utf-8") as f:
        f.write("# SYNTHETIC ground-truth metadata for project 02.07's NDT/ICP demo.\n")
        f.write(f"# regenerate: python make_synthetic.py --seed {args.seed}\n")
        f.write("# GT_POSE row: r00,r01,r02,r10,r11,r12,r20,r21,r22,tx,ty,tz -- T_map_scan (row-major R, t in meters).\n")
        f.write("# COUNTS row: n_map,n_scan_main,n_scan_cohort,n_scan_cohort_clean,n_icp_target.\n")
        f.write("# PARAMS row: range_noise_sigma_m,true_outlier_fraction,leaf_coarse_m,leaf_fine_m,icp_target_leaf_m.\n")
        writer = csv.writer(f)
        writer.writerow(["GT_POSE", *[f"{v:.10f}" for v in TRUE_R], *[f"{v:.10f}" for v in TRUE_T]])
        writer.writerow(["COUNTS", len(map_pts), len(scan_main), len(scan_cohort), len(scan_cohort_clean), len(icp_target)])
        writer.writerow(["PARAMS", f"{RANGE_NOISE_SIGMA_M:.6f}", f"{TRUE_OUTLIER_FRACTION:.6f}",
                         f"{LEAF_COARSE_M:.4f}", f"{LEAF_FINE_M:.4f}", f"{ICP_TARGET_LEAF_M:.4f}"])
    print(f"[make_synthetic] wrote metadata to {meta_path}")


if __name__ == "__main__":
    main()
