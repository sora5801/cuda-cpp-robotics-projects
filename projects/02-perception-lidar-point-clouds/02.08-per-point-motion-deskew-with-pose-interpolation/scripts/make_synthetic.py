#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample-data generator for 02.08
(Per-point motion deskew with pose interpolation).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
---------------------------------------------------------------
Motion deskew needs something no public LiDAR dataset hands you cleanly:
the EXACT continuous ego-trajectory during a single sweep, AND the "perfect"
undistorted scan the same scene would have produced from a single instant —
so the restoration error can be MEASURED against ground truth, not guessed.
That is only available synthetically (full ground truth is the whole point
of CLAUDE.md paragraph 8's synthetic-first default). This script is the sole
source of the committed sample under ../data/sample/ — no download, no
license question, bit-for-bit reproducible from a fixed seed.

The physical picture (THEORY.md "The problem" derives this in full)
---------------------------------------------------------------------
A spinning 16-beam LiDAR completes one sweep (0..360 deg of azimuth) in
SWEEP_DURATION_S seconds. It does NOT capture the scene instantaneously —
beam k fires at azimuth step a at time t(a) = a/AZIMUTH_STEPS * SWEEP_DURATION_S,
in the SENSOR's OWN, instantaneous body frame at that exact moment. If the
platform carrying the sensor is moving, "the scan" is actually hundreds of
tiny snapshots, each correct in a DIFFERENT frame, naively stacked together
as if they shared one — that stacking IS the distortion this project undoes.

Four motion cohorts (the project's teaching ladder), one static scene
------------------------------------------------------------------------
Every cohort ray-casts the SAME room (see build_scene_hit below) but moves
the platform along a different GROUND-TRUTH trajectory during the sweep:

  0 STRAIGHT   — constant velocity, constant heading. Position LERP between
                 two trajectory samples is EXACT for constant velocity (no
                 acceleration to miss) — the textbook "why LERP is enough
                 for translation" case, and this project's own consistency
                 check: 2-sample and 21-sample interpolation must agree here.
  1 ARC        — constant yaw RATE (a "unicycle" turn), closed-form position
                 via the standard unicycle arc integral. SLERP of the two
                 endpoint orientations reproduces the TRUE heading at every
                 instant exactly (constant angular velocity IS a geodesic at
                 constant speed) — but 2-sample POSITION interpolation is a
                 CHORD across a curved arc: the "constant-velocity assumption
                 biased" case the catalog bullet asks to measure.
  2 WIGGLE     — constant translation, SINUSOIDALLY OSCILLATING yaw (several
                 full cycles within one 100 ms sweep). No 2-point (start/end)
                 interpolation can see oscillations that happen entirely
                 BETWEEN its two samples — this cohort is designed to make
                 the dense-vs-sparse sampling gap as large as honestly
                 possible without becoming absurd (see WIGGLE_* below).
  3 STATIONARY — zero velocity, zero yaw rate: the "identity control". A
                 deskew pipeline that is not literally a no-op on a
                 non-moving platform has a bug, full stop.

Ground truth per ray (the correctness oracle's raw material — READ THIS
CAREFULLY, it is the one place a plausible-looking alternative is WRONG)
--------------------------------------------------------------------------
For every candidate (azimuth step, beam) pair, this script ray-casts ONCE,
from the TRUE moving pose at the ray's own firing time t(a), against the
static scene. That hit gives TWO things for the SAME physical 3-D point:

  (a) the "skewed" / raw point: the local direction times the measured
      range, r*d_local — exactly what a real LiDAR driver emits (no motion
      compensation, THEORY.md "The problem"); and
  (b) the "truth" point: that SAME physical world point, re-expressed in the
      reference frame using the platform's TRUE (continuous, un-discretized)
      pose at BOTH the firing time and the fixed reference instant
      t_ref = SWEEP_DURATION_S (sweep end) — i.e. the coordinates a
      PERFECT deskew (one with exact, noise-free knowledge of the whole
      trajectory) would produce.

A tempting-but-WRONG alternative is to fire a SECOND, independent ray in the
same local direction d_local from the reference pose and call that "truth":
it usually hits a DIFFERENT surface point (a different point on the wall,
or even a different wall) because the world direction and origin both
change with the platform's pose — so the "error" you would measure is
dominated by which surface each ray happened to hit, not by how well the
deskew pipeline reconstructs a KNOWN point. Transforming the SAME hit point
analytically (as done here) isolates exactly what this project measures:
the gap between a pipeline that only has DISCRETE trajectory SAMPLES to
interpolate from, and the continuous ground truth this generator alone
knows exactly.

What main.cu/kernels.cu never see: the CONTINUOUS trajectory function above.
They only see DENSE_SAMPLES (a 200 Hz-equivalent discretization of it) and,
derived from that, a 2-sample SPARSE regime (first/last dense sample) — the
two "regimes" the catalog bullet asks to compare. The gap between what the
deskew pipeline RECOVERS from those samples and the analytic truth this
script knows exactly is the whole measurement.

Usage
-----
    python make_synthetic.py                 # writes the committed sample
    python make_synthetic.py --out DIR/FILE   # experiments; do not commit
"""

import argparse
import math
import struct
import sys
from pathlib import Path

# ===========================================================================
# Deterministic RNG: xorshift32 (stdlib-only, repo convention, CLAUDE.md
# paragraph 12) — used ONLY for the small per-return range noise below, never
# for anything that would make the ground-truth trajectories non-reproducible.
# ===========================================================================
class Xorshift32:
    def __init__(self, seed: int):
        s = seed & 0xFFFFFFFF
        if s == 0:
            s = 1  # degenerate at seed 0 (stays 0 forever) — same guard used repo-wide
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


DEFAULT_SEED = 42  # repo convention (CLAUDE.md paragraph 12)

# ===========================================================================
# Beam model + sweep timing — MUST MATCH ../src/kernels.cuh's kBeamElevDeg /
# kAzimuthSteps / kSweepDurationS exactly (main.cu asserts n_beams and
# azimuth_steps read from the file header against the compiled-in constants
# at load time — a data/code consistency check, not a coincidence).
#
# The beam table is the SAME -15..+15 deg in 2 deg steps 16-channel model
# 02.01 cites from 01.18's THEORY.md; reused here verbatim (see that
# project's data/README.md for the original derivation).
# ===========================================================================
BEAM_ELEV_DEG = list(range(-15, 16, 2))
NUM_BEAMS = len(BEAM_ELEV_DEG)
assert NUM_BEAMS == 16, "beam table must match the repo's cited 16-beam model exactly"

AZIMUTH_STEPS = 360          # 1 deg/step — a realistic single-sweep resolution, kept modest so the sample stays small
SWEEP_DURATION_S = 0.100     # 100 ms per revolution (10 Hz spin rate) — a realistic mechanical-LiDAR sweep period
RANGE_NOISE_M = 0.01         # +-1 cm per-return range noise (order of magnitude: 02.01's RANGE_NOISE_M)
MAX_RANGE_M = 30.0

# ===========================================================================
# Scene geometry — a simplified version of 02.01's room (cited): floor +
# four walls, NO boxes (this project's teaching payload is the TRAJECTORY,
# not scene complexity; 02.01 is the place to see adversarial box clutter).
# All coordinates are WORLD-frame meters; the platform starts near the
# origin and moves only 1-1.5 m over one 100 ms sweep, so it stays deep
# inside the room for every cohort.
# ===========================================================================
SENSOR_HEIGHT_M = 1.5        # sensor mount height above the floor
ROOM_HALF_M = 8.0            # walls at x,y = +-ROOM_HALF_M -> a 16x16 m room
WALL_TOP_M = 1.5             # walls span z in [-SENSOR_HEIGHT_M, WALL_TOP_M] (a 3 m physical wall)


def ray_intersect_scene(ox, oy, oz, dx, dy, dz):
    """Ray-cast one ray (world origin o, world UNIT direction d) against the
    floor + 4 walls; return the smallest positive hit range t, or None.

    Same slab/plane-intersection reasoning as 02.01's ray_intersect_scene
    (cited) — reproduced here (not imported) per the repo's "every project
    self-contained" rule (CLAUDE.md section 4).
    """
    best_t = None

    # Floor: z = -SENSOR_HEIGHT_M (world), only where |x|,|y| <= ROOM_HALF_M.
    if abs(dz) > 1e-9:
        t = (-SENSOR_HEIGHT_M - oz) / dz
        if t > 1e-6:
            hx, hy = ox + dx * t, oy + dy * t
            if abs(hx) <= ROOM_HALF_M and abs(hy) <= ROOM_HALF_M:
                best_t = t if best_t is None else min(best_t, t)

    # Four walls (axis-aligned planes), each clipped to its z-span and to the
    # room footprint along the OTHER horizontal axis.
    for axis_is_x in (True, False):
        for sign in (+1.0, -1.0):
            plane = sign * ROOM_HALF_M
            d_axis = dx if axis_is_x else dy
            o_axis = ox if axis_is_x else oy
            if abs(d_axis) < 1e-9:
                continue
            t = (plane - o_axis) / d_axis
            if t <= 1e-6:
                continue
            hz = oz + dz * t
            other = (oy + dy * t) if axis_is_x else (ox + dx * t)
            if -SENSOR_HEIGHT_M <= hz <= WALL_TOP_M and abs(other) <= ROOM_HALF_M:
                best_t = t if best_t is None else min(best_t, t)

    if best_t is not None and best_t <= MAX_RANGE_M:
        return best_t
    return None


# ---------------------------------------------------------------------------
# Quaternion helpers (repo order w,x,y,z — CLAUDE.md paragraph 12, cited from
# 09.01's kernels.cuh convention). Minimal, plain-Python mirrors of the
# HD C++ helpers in ../src/kernels.cuh — kept independent (this script is the
# GROUND-TRUTH generator; it must not share code with the pipeline it is
# testing, or a shared bug would generate its own "confirmation").
# ---------------------------------------------------------------------------
def quat_from_yaw(yaw_rad):
    """Rotation about world +z (yaw) by yaw_rad, as (w,x,y,z)."""
    h = 0.5 * yaw_rad
    return (math.cos(h), 0.0, 0.0, math.sin(h))


def quat_rotate(q, v):
    """Rotate 3-vector v by unit quaternion q (w,x,y,z), sandwich-product
    formula (same identity ../src/kernels.cuh's quat_rotate implements)."""
    w, x, y, z = q
    vx, vy, vz = v
    # t = 2 * cross((x,y,z), v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    # v' = v + w*t + cross((x,y,z), t)
    rx = vx + w * tx + (y * tz - z * ty)
    ry = vy + w * ty + (z * tx - x * tz)
    rz = vz + w * tz + (x * ty - y * tx)
    return (rx, ry, rz)


# ===========================================================================
# The four ground-truth trajectories: each is a function true_pose(t) -> ((px,
# py,pz), (qw,qx,qy,qz)) in the WORLD frame, exact (no discretization) — the
# thing NEITHER the GPU kernel nor the CPU reference ever sees directly; they
# only see DENSE_SAMPLES of it (below).
# ===========================================================================

# Cohort 0 — STRAIGHT constant velocity, constant heading.
STRAIGHT_V = (15.0, 0.0, 0.0)   # m/s -- 15 m/s * 0.1 s sweep = 1.5 m travelled (the THEORY.md distortion-table number)

def true_pose_straight(t):
    p = (STRAIGHT_V[0] * t, STRAIGHT_V[1] * t, STRAIGHT_V[2] * t)
    return p, (1.0, 0.0, 0.0, 0.0)   # identity orientation throughout


# Cohort 1 — ARC: constant forward speed + constant yaw rate (a "unicycle").
ARC_SPEED_MS = 15.0     # m/s forward body-frame speed
ARC_YAW_RATE = 6.0      # rad/s -- 0.6 rad (~34 deg) total yaw change over the 100 ms sweep

def true_pose_arc(t):
    yaw = ARC_YAW_RATE * t
    # Closed-form unicycle position integral from yaw0=0, p0=(0,0):
    #   x(t) = (v/omega) * sin(omega t),  y(t) = (v/omega) * (1 - cos(omega t))
    r = ARC_SPEED_MS / ARC_YAW_RATE
    p = (r * math.sin(yaw), r * (1.0 - math.cos(yaw)), 0.0)
    return p, quat_from_yaw(yaw)


# Cohort 2 — WIGGLE: modest constant translation + an AGGRESSIVE sinusoidal
# yaw oscillation with several full periods inside one sweep — the cohort
# designed to make the sparse (2-sample) regime miss the motion entirely
# while the dense (200 Hz-equivalent) regime still resolves it.
WIGGLE_V = (10.0, 0.0, 0.0)     # m/s -- modest translation so yaw error dominates the picture
WIGGLE_AMPLITUDE_RAD = 0.35     # ~20 deg peak yaw deviation
WIGGLE_FREQ_HZ = 25.0           # 2.5 full oscillation cycles inside the 100 ms sweep

def true_pose_wiggle(t):
    yaw = WIGGLE_AMPLITUDE_RAD * math.sin(2.0 * math.pi * WIGGLE_FREQ_HZ * t)
    p = (WIGGLE_V[0] * t, WIGGLE_V[1] * t, WIGGLE_V[2] * t)
    return p, quat_from_yaw(yaw)


# Cohort 3 — STATIONARY: the identity control.
def true_pose_stationary(t):
    return (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)


COHORTS = [
    ("straight", true_pose_straight),
    ("arc", true_pose_arc),
    ("wiggle", true_pose_wiggle),
    ("stationary", true_pose_stationary),
]

# Dense trajectory sampling: a 200 Hz-EQUIVALENT discretization of the true
# trajectory over the sweep (matches ../src/kernels.cuh's kDenseSamples —
# main.cu asserts the file's per-cohort sample count against it). 21 samples
# at 5 ms spacing span exactly [0, 100] ms inclusive of both ends.
DENSE_SAMPLES = 21
assert abs((DENSE_SAMPLES - 1) * (SWEEP_DURATION_S / (DENSE_SAMPLES - 1)) - SWEEP_DURATION_S) < 1e-9


def build_dense_trajectory(true_pose_fn):
    """Sample true_pose_fn at DENSE_SAMPLES evenly-spaced instants covering
    the whole sweep [0, SWEEP_DURATION_S] inclusive. Returns a flat list of
    DENSE_SAMPLES * 8 floats: (t_s, px,py,pz, qw,qx,qy,qz) per sample — the
    exact layout ../src/kernels.cuh's trajectory contract documents.
    The SPARSE regime is never stored separately: main.cu derives it by
    taking just this array's FIRST and LAST samples (one honest source of
    ground truth, two ways of subsampling it)."""
    flat = []
    dt = SWEEP_DURATION_S / (DENSE_SAMPLES - 1)
    for i in range(DENSE_SAMPLES):
        t = i * dt
        (px, py, pz), (qw, qx, qy, qz) = true_pose_fn(t)
        flat.extend((t, px, py, pz, qw, qx, qy, qz))
    return flat


def build_cohort_points(true_pose_fn, rng: Xorshift32):
    """Ray-cast one cohort: for every (azimuth step, beam) pair, cast ONE ray
    from the TRUE moving pose at that ray's own firing time. The hit point
    yields both the 'skewed' record (local coordinates at firing time — the
    raw driver output) and the 'truth' record (the SAME physical point,
    analytically re-expressed in the reference frame via the TRUE
    trajectory — the module docstring explains why this, and not a second
    independent ray-cast, is the correct ground truth).

    Returns (t_list, beam_id_list, skewed_xyz, truth_xyz, n_rays_cast).
    """
    t_ref = SWEEP_DURATION_S
    p_ref, q_ref = true_pose_fn(t_ref)
    # Conjugate of q_ref: rotates a WORLD-frame vector into the reference
    # sensor's own frame (R(conj(q))= R(q)^-1 for a unit quaternion — the
    # same fact 01.10's quat_conj comment cites).
    qrw, qrx, qry, qrz = q_ref
    q_ref_conj = (qrw, -qrx, -qry, -qrz)

    t_list, beam_list, skewed, truth = [], [], [], []
    n_rays_cast = 0

    dt = SWEEP_DURATION_S / AZIMUTH_STEPS
    for az_step in range(AZIMUTH_STEPS):
        t_i = az_step * dt                       # this shot's firing time (< SWEEP_DURATION_S)
        az = 2.0 * math.pi * az_step / AZIMUTH_STEPS
        p_i, q_i = true_pose_fn(t_i)

        for beam_id, el_deg in enumerate(BEAM_ELEV_DEG):
            el = math.radians(el_deg)
            cel, sel = math.cos(el), math.sin(el)
            # Direction in the SENSOR's own body frame at this instant (the
            # beam model 02.01 cites from 01.18, reused verbatim).
            dx_local = cel * math.cos(az)
            dy_local = cel * math.sin(az)
            dz_local = sel
            n_rays_cast += 1

            # Cast from the TRUE pose at firing time t_i (world direction =
            # q_i applied to the local direction — this IS the physical ray
            # a real spinning LiDAR emits at this instant).
            wdx, wdy, wdz = quat_rotate(q_i, (dx_local, dy_local, dz_local))
            t_hit = ray_intersect_scene(p_i[0], p_i[1], p_i[2], wdx, wdy, wdz)
            if t_hit is None:
                continue  # honest dropout: this ray never reaches the scene from here

            r = t_hit + rng.uniform(-RANGE_NOISE_M, RANGE_NOISE_M)
            if r <= 1e-6:
                r = t_hit

            # (a) SKEWED record: local direction * measured range, in the
            # sensor's OWN frame at t_i — exactly the raw driver output.
            local_pt = (dx_local * r, dy_local * r, dz_local * r)

            # World-frame hit point (using the TRUE, continuous p_i/q_i —
            # this generator's privileged knowledge; the pipeline under test
            # never sees this array, only the discretized trajectory below).
            world_pt = (p_i[0] + wdx * r, p_i[1] + wdy * r, p_i[2] + wdz * r)

            # (b) TRUTH record: the SAME world point, re-expressed in the
            # reference sensor frame via the TRUE p_ref/q_ref — the exact
            # rigid transform ../src/kernels.cuh's deskew_one_point performs
            # (see that header for the derivation), evaluated here with
            # PERFECT (non-interpolated) poses. This is what the pipeline
            # would produce if its trajectory samples were infinitely dense.
            dpx, dpy, dpz = world_pt[0] - p_ref[0], world_pt[1] - p_ref[1], world_pt[2] - p_ref[2]
            truth_pt = quat_rotate(q_ref_conj, (dpx, dpy, dpz))

            t_list.append(t_i)
            beam_list.append(beam_id)
            skewed.append(local_pt)
            truth.append(truth_pt)

    return t_list, beam_list, skewed, truth, n_rays_cast


def write_binary_sample(out_path: Path, cohort_results):
    """Write the committed sample as a small fixed binary format:

        bytes  0.. 7  magic          b'DESKEW01' (8 bytes, no null terminator)
        bytes  8..11  int32          num_cohorts (4)
        bytes 12..15  int32          num_beams (16)
        bytes 16..19  int32          azimuth_steps
        bytes 20..23  int32          n_dense_samples (21)
        bytes 24..27  float32        sweep_duration_s
        bytes 28..31  float32        room_half_m (documentation only)
        then, for each of the 4 cohorts IN FIXED ORDER (straight, arc, wiggle,
        stationary — ../src/kernels.cuh's kCohortNames documents this order):
          int32           cohort_id (0..3, redundant with position -- a
                           self-check main.cu's loader verifies)
          int32           n_points
          float32 x (n_dense_samples*8)   dense trajectory samples, layout
                           (t_s, px,py,pz, qw,qx,qy,qz) per sample
          per point i in [0, n_points):
            float32       t_s        firing time, seconds, relative to sweep start
            int32         beam_id    0..15, indexes kBeamElevDeg
            float32 x 3   xyz        SKEWED point, meters, sensor-local frame AT t_s
          per point i in [0, n_points):
            float32 x 3   xyz        TRUTH point, meters, sensor-local frame AT t_ref

    Every field is written with an EXPLICIT little-endian struct format
    (never a raw fwrite of a C struct), matching the rest of this domain
    (02.01's write_binary_sample, cited) so the layout never depends on any
    compiler's padding/alignment rules.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_points = 0
    with out_path.open('wb') as f:
        f.write(b'DESKEW01')
        f.write(struct.pack('<iiii', len(cohort_results), NUM_BEAMS, AZIMUTH_STEPS, DENSE_SAMPLES))
        f.write(struct.pack('<ff', SWEEP_DURATION_S, ROOM_HALF_M))

        for cohort_id, (name, dense_traj, t_list, beam_list, skewed, truth) in enumerate(cohort_results):
            n_points = len(t_list)
            total_points += n_points
            f.write(struct.pack('<ii', cohort_id, n_points))
            f.write(struct.pack(f'<{len(dense_traj)}f', *dense_traj))

            pt_flat = []
            for i in range(n_points):
                pt_flat.extend((t_list[i], skewed[i][0], skewed[i][1], skewed[i][2]))
            # t and xyz interleave as floats, but beam_id is an int32 sitting
            # BETWEEN t and xyz per the format table above -- pack row-by-row
            # (not as one big block) so the int32/float32 boundary is exact.
            for i in range(n_points):
                f.write(struct.pack('<f', t_list[i]))
                f.write(struct.pack('<i', beam_list[i]))
                f.write(struct.pack('<fff', *skewed[i]))

            truth_flat = []
            for i in range(n_points):
                truth_flat.extend(truth[i])
            f.write(struct.pack(f'<{len(truth_flat)}f', *truth_flat))

    return total_points


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_out = script_dir.parent / 'data' / 'sample' / 'deskew_scan.bin'

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help=f'xorshift32 seed for the range-noise stream (default {DEFAULT_SEED})')
    parser.add_argument('--out', type=Path, default=default_out,
                        help='output binary path (default: ../data/sample/deskew_scan.bin)')
    args = parser.parse_args()

    rng = Xorshift32(args.seed)

    cohort_results = []
    for name, true_pose_fn in COHORTS:
        dense_traj = build_dense_trajectory(true_pose_fn)
        t_list, beam_list, skewed, truth, n_rays_cast = build_cohort_points(true_pose_fn, rng)
        cohort_results.append((name, dense_traj, t_list, beam_list, skewed, truth))
        hit_rate = 100.0 * len(t_list) / n_rays_cast if n_rays_cast else 0.0
        print(f"[make_synthetic] cohort '{name}': {n_rays_cast} candidate rays "
              f"({NUM_BEAMS} beams x {AZIMUTH_STEPS} azimuth steps) -> {len(t_list)} paired "
              f"skewed/truth points ({hit_rate:.1f}% hit rate)")

    total_points = write_binary_sample(args.out, cohort_results)
    size_bytes = args.out.stat().st_size
    print(f"[make_synthetic] wrote {args.out} ({total_points} total points across "
          f"{len(COHORTS)} cohorts, {size_bytes} bytes, sweep={SWEEP_DURATION_S*1000:.0f} ms, "
          f"labeled SYNTHETIC)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
