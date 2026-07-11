#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample generator for project 01.10
(Rolling-shutter correction using IMU rates).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
-----------------------------------------------------------------
Rolling-shutter correction needs a scene with FULL ground truth for "what a
global-shutter camera would have captured" — a real RS camera cannot give
you that for free (there is no way to un-shutter a real sensor after the
fact). Synthetic authorship gives it for free and EXACTLY: this script knows
the TRUE, continuous camera-rotation trajectory (no gyro noise at all), so
it can render both the distorted rolling-shutter capture AND the undistorted
ground-truth reference from the SAME analytic model, then hand the C++
pipeline only a noisy, sparse, 200 Hz gyro trace to reconstruct one from the
other — exactly the real-world problem, minus the impossibility of ever
knowing ground truth for a real capture.

The scene: a "distant panorama", built as a reference-plane texture
--------------------------------------------------------------------
The catalog bullet's scope is PURE ROTATION (no translation), and with zero
translation, scene DEPTH never matters: every camera ray always samples the
same world point regardless of how far away it is. That means "an infinite
panorama" and "a textured plane fixed at the reference camera's own image
plane, warped into every other camera orientation by a pure rotation" are
mathematically equivalent for this project (both are gnomonic/perspective
projections of the same world content) — and the second framing is exactly
01.01's "author the scene directly in the ideal/rectified frame, then warp
it through the camera model to build the raw capture" pattern, generalized
from ONE fixed rectifying rotation to a TIME-VARYING one. This script uses
that framing: the texture (see texture_intensity() below) is a function of
NORMALIZED REFERENCE-CAMERA coordinates (u, v); rendering the ground-truth
GS frame is a DIRECT raster of it (no warp — R_rel(t_ref) is identity by
construction); rendering the RS frame applies, PER ROW, the same K*R*K^-1
row-homography construction ../src/kernels.cuh derives (that file's "THE ROW
HOMOGRAPHY" section) — this script computes the FORWARD direction (world/
reference ray -> that row's raw pixel), the exact inverse of what the C++
pipeline solves for.

Heeding project 01.04's self-similarity lesson: a pure checkerboard's
repeating cells are locally self-similar (a detector cannot tell one cell
boundary from its neighbor by local appearance alone), which is a real
hazard for a MOVING-window feature detector, not for a fixed threshold-
crossing line search — but a busy, non-periodic background is still used
here (see hashed_cell() below: a two-octave, per-cell XOR-shift hash, never
a repeating tile) so the marker-line detector in ../src/main.cu is never
tempted to lock onto a look-alike background edge, AND the image looks like
real, textured content rather than a sparse test pattern.

What it writes (into ../data/sample/, well under the CLAUDE.md paragraph 8
budget — see the printed byte count at the end):

    rs_input.pgm         kImgW x kImgH, 8-bit grayscale (PGM P5) — the
                          captured ROLLING-SHUTTER frame (row v exposed at
                          the TRUE trajectory's orientation at time t(v)).
    ground_truth_gs.pgm  kImgW x kImgH, 8-bit grayscale (PGM P5) — what an
                          ideal GLOBAL-SHUTTER camera at the SAME orientation
                          as the RS frame's reference row would have seen —
                          the restoration gate's ground truth.
    gyro_clean.csv        ~10 rows of (t_s, wx, wy, wz) — the TRUE angular
                          velocity, sampled at 200 Hz, no noise.
    gyro_degraded.csv     the SAME sample times, with a constant bias and
                          random noise added per axis (see DEGRADE_* below) —
                          a simple, illustrative sensor-error model, not a
                          calibrated IMU noise spec.
    params.csv            every constant this script used, for provenance
                          (../data/README.md documents each field).

Usage:
    python make_synthetic.py                  # the committed 384x288 scene
    python make_synthetic.py --seed 7          # a different seed; do not commit
"""

import argparse
import csv
import math
import sys
from pathlib import Path

# ===========================================================================
# XorShift32 — the repo-standard tiny deterministic PRNG (CLAUDE.md's
# "no std::uniform_real_distribution / no numpy — xorshift32, seed 42" rule,
# same generator project 01.04's synthesizer uses). ONE dead state (seed=0)
# is avoided by construction wherever a seed is derived below.
# ===========================================================================
class XorShift32:
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF
        if self.state == 0:
            self.state = 0x9E3779B9   # never let the dead state escape a derived seed of 0

    def next_u32(self) -> int:
        x = self.state
        x = (x ^ (x << 13)) & 0xFFFFFFFF
        x = (x ^ (x >> 17)) & 0xFFFFFFFF
        x = (x ^ (x << 5)) & 0xFFFFFFFF
        self.state = x
        return x

    def uniform(self, lo: float, hi: float) -> float:
        u = self.next_u32() / 4294967296.0   # 2^32 -> [0, 1)
        return lo + u * (hi - lo)


# ===========================================================================
# Geometry / timing constants — MUST MATCH ../src/kernels.cuh's constexpr
# block of the same names exactly (that file is the C++ single source of
# truth; this comment is the CLAUDE.md paragraph 12 cross-reference).
# ===========================================================================
IMG_W = 384                          # must match kImgW
IMG_H = 288                          # must match kImgH
FX = 380.0                           # must match kFx
FY = 380.0                           # must match kFy
CX = (IMG_W - 1) * 0.5               # must match kCx = 191.5
CY = (IMG_H - 1) * 0.5               # must match kCy = 143.5

READOUT_TIME_S = 0.025               # must match kReadoutTimeS (25 ms)
LINE_TIME_S = READOUT_TIME_S / IMG_H  # must match kLineTimeS
FRAME_T0_S = 0.0                     # must match kFrameT0S
# must match kFrameTRefS: the MIDDLE row's own exposure time.
FRAME_TREF_S = FRAME_T0_S + 0.5 * (IMG_H - 1) * LINE_TIME_S

GYRO_RATE_HZ = 200.0                 # must match kGyroRateHz
GYRO_DT_S = 1.0 / GYRO_RATE_HZ        # 5 ms nominal spacing

# The gyro TRACE's time window: wider than [FRAME_T0_S, FRAME_T0_S+READOUT_TIME_S]
# by a small margin on each side, so ../src/main.cu's fine-integration
# interpolation always has real bracketing samples for every row time AND
# for FRAME_TREF_S itself (no extrapolation ever needed). Generator-only —
# the C++ side reads whatever timestamps are actually in the CSV, so this
# does not need a kernels.cuh mirror, only internal self-consistency.
GYRO_WINDOW_T0_S = -0.010
GYRO_WINDOW_T1_S = 0.035

# Fine-integration step for THIS script's OWN ground-truth trajectory
# (independent of, and much finer than, ../src/main.cu's gyro-sample-driven
# integration — this is the "what actually happened" reference the C++
# pipeline is reconstructing from noisy/sparse samples, so it uses the
# TRUE continuous omega(t) below, not the discretized CSV).
FINE_DT_S = 5.0e-5                   # 0.05 ms sub-steps -> 900 steps over the window

# ===========================================================================
# The TRUE rotation profile — body-frame (camera-axes) angular velocity,
# rad/s, as a sum of one sinusoid per axis (pitch about camera X, yaw about
# camera Y, roll about camera Z — the camera-optical frame exception stated
# once at this API boundary, matching ../src/kernels.cuh's file header).
# Amplitudes/frequencies are chosen to produce a VISIBLE several-to-many-
# pixel row shear over one frame's 25 ms readout (see the measured skew
# reported by ../src/main.cu's straightness_negative_control gate) — a
# "jello" handheld/drone-class jitter, not a slow pan.
# ===========================================================================
A_PITCH_DEG, F_PITCH_HZ, PHI_PITCH = 1.8, 3.2, 1.1     # about camera X
A_YAW_DEG,   F_YAW_HZ,   PHI_YAW   = 2.5, 5.0, 0.3     # about camera Y
A_ROLL_DEG,  F_ROLL_HZ,  PHI_ROLL  = 1.2, 7.0, 2.0     # about camera Z

A_PITCH_RAD = math.radians(A_PITCH_DEG)
A_YAW_RAD = math.radians(A_YAW_DEG)
A_ROLL_RAD = math.radians(A_ROLL_DEG)


def true_omega(t: float):
    """The TRUE body-frame angular velocity (wx, wy, wz), rad/s, at time t.

    Each component is the exact time-derivative of an amplitude*sin(...)
    Euler-like single-axis profile, TREATED as a body rate directly (not
    derived from an Euler-angle composition — a real gyro measures body
    rates, and integrating them via quaternion kinematics, as both this
    script's fine trajectory below and ../src/main.cu do independently, is
    the physically honest path — see ../THEORY.md "The math").
    """
    wx = 2.0 * math.pi * F_PITCH_HZ * A_PITCH_RAD * math.cos(2.0 * math.pi * F_PITCH_HZ * t + PHI_PITCH)
    wy = 2.0 * math.pi * F_YAW_HZ * A_YAW_RAD * math.cos(2.0 * math.pi * F_YAW_HZ * t + PHI_YAW)
    wz = 2.0 * math.pi * F_ROLL_HZ * A_ROLL_RAD * math.cos(2.0 * math.pi * F_ROLL_HZ * t + PHI_ROLL)
    return wx, wy, wz


# ===========================================================================
# Degraded-gyro error model (generator-only constants — no kernels.cuh
# mirror needed, see that file's comment on set_row_lut/GyroSample). An
# ILLUSTRATIVE, not calibrated, sensor-error model: a constant per-axis
# bias (the dominant real MEMS-gyro error term) plus uniform (not Gaussian —
# stated honestly, a simplification) per-sample noise.
# ===========================================================================
# 3.0 deg/s is a realistic UNCALIBRATED consumer MEMS-gyro bias (bias
# instability before any online estimation) — measured (see README/THEORY
# "Expected output") to move the corrected restoration error by a real,
# visible factor while the correction still comfortably beats doing
# nothing at all (GATE gyro_degradation). An earlier, smaller 0.5 deg/s
# choice integrated to well under one pixel over this project's short 45 ms
# gyro window and was barely distinguishable from the clean run — too small
# to teach the lesson honestly, so it was widened.
GYRO_BIAS_DPS = 8.0                                    # constant bias, deg/s, same sign every axis
GYRO_BIAS_RAD_S = math.radians(GYRO_BIAS_DPS)
GYRO_NOISE_HALF_WIDTH_RAD_S = 0.18                      # uniform noise half-width, rad/s (~10.3 deg/s peak)

# ===========================================================================
# Marker-line + background texture constants — MUST MATCH ../src/main.cu's
# kLineThreshold (195.0) and its background-range assumption ([30,140], line
# at 255) in its find_line_center_x()/kLineThreshold comment.
# ===========================================================================
LINE_HALF_WIDTH_U = 2.0 / FX          # marker line half-width in normalized-u units (~2 px at FX)
LINE_INTENSITY = 255
BG_LO, BG_HI = 30.0, 140.0            # background intensity range (comfortably below the 195 threshold)


# ---------------------------------------------------------------------------
# Quaternion algebra — Python's OWN independent re-implementation (never
# imported from the C++ side) of the same (w, x, y, z) repo-order math
# ../src/kernels.cuh defines, in double precision. Three independent
# languages/files now compute this project's rotation kinematics (this
# script, kernels.cu's GPU kernel, reference_cpu.cpp's CPU twin) — agreement
# across all three, exercised end-to-end by main.cu's restoration gate, is
# strong evidence the PHYSICS is right, not just one file's arithmetic
# (same cross-language-agreement argument 01.01's make_synthetic.py makes).
# ---------------------------------------------------------------------------
def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_conj(q):
    w, x, y, z = q
    return (w, -x, -y, -z)


def quat_normalize(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-20:
        return (1.0, 0.0, 0.0, 0.0)
    return (w / n, x / n, y / n, z / n)


def quat_integrate_step(q, wx, wy, wz, dt):
    """Exponential-map integration step — see ../src/kernels.cuh's
    quat_integrate_step() for the full derivation this mirrors independently
    (constant-omega-over-dt exact update, right-multiplied body-frame
    delta rotation)."""
    wmag = math.sqrt(wx * wx + wy * wy + wz * wz)
    if wmag < 1e-9:
        dq = quat_normalize((1.0, 0.5 * wx * dt, 0.5 * wy * dt, 0.5 * wz * dt))
    else:
        half_angle = 0.5 * wmag * dt
        s = math.sin(half_angle) / wmag
        dq = (math.cos(half_angle), wx * s, wy * s, wz * s)
    return quat_normalize(quat_mul(q, dq))


def quat_to_mat3(q):
    """Unit quaternion -> row-major 3x3 rotation matrix (v' = R*v), the same
    formula as ../src/kernels.cuh's quat_to_mat3 (independently retyped)."""
    w, x, y, z = q
    return (
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w),
        2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
        2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y),
    )


def mat3_apply(R, v):
    r0, r1, r2, r3, r4, r5, r6, r7, r8 = R
    vx, vy, vz = v
    return (r0 * vx + r1 * vy + r2 * vz,
            r3 * vx + r4 * vy + r5 * vz,
            r6 * vx + r7 * vy + r8 * vz)


# ---------------------------------------------------------------------------
# build_fine_trajectory — integrate the TRUE, continuous omega(t) from
# GYRO_WINDOW_T0_S to GYRO_WINDOW_T1_S at FINE_DT_S resolution. This is the
# "what actually happened" ground truth this script alone has access to
# (the C++ pipeline only ever sees the sparse, noisy 200 Hz CSV — see the
# file header). Returns a list of (t, quat) pairs, evenly spaced by
# FINE_DT_S, anchored q=identity at GYRO_WINDOW_T0_S (an arbitrary but self-
# consistent choice — only relative rotations are ever observed downstream).
# ---------------------------------------------------------------------------
def build_fine_trajectory():
    n_steps = int(round((GYRO_WINDOW_T1_S - GYRO_WINDOW_T0_S) / FINE_DT_S))
    traj = [(GYRO_WINDOW_T0_S, (1.0, 0.0, 0.0, 0.0))]
    q = (1.0, 0.0, 0.0, 0.0)
    t = GYRO_WINDOW_T0_S
    for i in range(n_steps):
        # Midpoint-in-time omega sample for this sub-step (same accuracy
        # trick ../src/main.cu's integrate_gyro_to_fine_trajectory uses,
        # here applied to the TRUE continuous omega(t) rather than a
        # linear interpolation between sparse samples — the generator can
        # afford to just call true_omega() at the exact midpoint).
        t_mid = t + 0.5 * FINE_DT_S
        wx, wy, wz = true_omega(t_mid)
        q = quat_integrate_step(q, wx, wy, wz, FINE_DT_S)
        t += FINE_DT_S
        traj.append((t, q))
    return traj


def interpolate_fine(traj, t: float):
    """Linear-interpolate the fine trajectory at time t (binary search;
    traj is time-ordered and evenly spaced, so a direct index also works,
    but a search mirrors ../src/main.cu's interpolate_fine() shape without
    depending on FINE_DT_S's exact value staying in sync)."""
    if t <= traj[0][0]:
        return traj[0][1]
    if t >= traj[-1][0]:
        return traj[-1][1]
    lo, hi = 0, len(traj) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if traj[mid][0] <= t:
            lo = mid
        else:
            hi = mid
    t0, q0 = traj[lo]
    t1, q1 = traj[hi]
    frac = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
    w = q0[0] + (q1[0] - q0[0]) * frac
    x = q0[1] + (q1[1] - q0[1]) * frac
    y = q0[2] + (q1[2] - q0[2]) * frac
    z = q0[3] + (q1[3] - q0[3]) * frac
    return quat_normalize((w, x, y, z))


# ---------------------------------------------------------------------------
# hashed_cell / background_intensity — the multi-scale, non-periodic
# background texture (file header: heeding 01.04's self-similarity lesson).
# Each octave partitions (u, v) into CELL-sized bins and hashes the bin's
# integer coordinates (plus a salt distinguishing the octave AND this run's
# --seed) through a freshly-seeded XorShift32 stream to a deterministic
# value in [0, 1) — the same "hash the cell index, don't tile a pattern"
# idea 01.04's cell_color() uses, adapted to a continuous (u, v) domain
# (this project's rays are not laid out on any fixed pixel grid once
# rotated, so the hash must accept CONTINUOUS coordinates, unlike 01.04's
# integer patch/cell indices).
# ---------------------------------------------------------------------------
def hashed_cell(u: float, v: float, cell: float, salt: int, seed: int) -> float:
    cu = math.floor(u / cell)
    cv = math.floor(v / cell)
    # Fold the two cell indices, the octave salt, and the run seed into one
    # 32-bit integer, then draw ONE value from a stream seeded on exactly
    # that combination — deterministic (same inputs -> same texture, every
    # run) and non-periodic (no two nearby cells share a period).
    h = (int(cu) * 92821 + int(cv) * 68917 + salt * 15485863 + seed * 2654435761) & 0xFFFFFFFF
    return XorShift32(h).uniform(0.0, 1.0)


def background_intensity(u: float, v: float, seed: int) -> float:
    octave_a = hashed_cell(u, v, 0.14, 1, seed)     # coarse octave
    octave_b = hashed_cell(u, v, 0.045, 2, seed)    # fine octave
    mix = 0.55 * octave_a + 0.45 * octave_b          # in [0, 1)
    return BG_LO + mix * (BG_HI - BG_LO)


def texture_intensity(u: float, v: float, seed: int) -> float:
    """The scene authored in the REFERENCE camera's own normalized
    coordinates (u, v) = ((x-CX)/FX, (y-CY)/FY) — see the file header for
    why this is the didactically-honest equivalent of a true panorama under
    pure rotation. A single bright vertical marker line at u=0 (the
    straightness gate's known-straight feature) painted over the hashed
    background."""
    if abs(u) < LINE_HALF_WIDTH_U:
        return float(LINE_INTENSITY)
    return background_intensity(u, v, seed)


# ---------------------------------------------------------------------------
# render_ground_truth — direct raster of texture_intensity over the nominal
# pixel grid (no rotation applied: the GS reference view IS this script's
# authoring frame, by construction — see the file header).
# ---------------------------------------------------------------------------
def render_ground_truth(seed: int) -> bytearray:
    out = bytearray(IMG_W * IMG_H)
    for y in range(IMG_H):
        row = y * IMG_W
        v = (y - CY) / FY
        for x in range(IMG_W):
            u = (x - CX) / FX
            val = texture_intensity(u, v, seed)
            out[row + x] = int(round(min(max(val, 0.0), 255.0)))
    return out


# ---------------------------------------------------------------------------
# render_rs_frame — the captured ROLLING-SHUTTER frame: row y is rendered
# using the TRUE orientation at t(y) = FRAME_T0_S + y*LINE_TIME_S, one row-
# homography matrix at a time (../src/kernels.cuh's "THE ROW HOMOGRAPHY"
# section derives the formula this implements in the FORWARD direction).
#
#   q_ref = true orientation at FRAME_TREF_S (the GS reference instant).
#   q_row = true orientation at this row's own exposure time t(y).
#   q_M   = conj(q_ref) (x) q_row   -- rotates a RAW/row-camera-frame ray
#           INTO the reference camera's own local coordinates; this is the
#           EXACT TRANSPOSE of what ../src/main.cu's row LUT stores
#           (q_rel = conj(q_row) (x) q_ref), matching the forward/inverse
#           pairing 01.01's R_rect_raw / R_rect_raw^T uses (kernels.cuh's
#           file header states this explicitly).
# ---------------------------------------------------------------------------
def render_rs_frame(fine_traj, seed: int) -> bytearray:
    out = bytearray(IMG_W * IMG_H)
    q_ref = interpolate_fine(fine_traj, FRAME_TREF_S)
    q_ref_conj = quat_conj(q_ref)
    for y in range(IMG_H):
        t_row = FRAME_T0_S + y * LINE_TIME_S
        q_row = interpolate_fine(fine_traj, t_row)
        q_m = quat_normalize(quat_mul(q_ref_conj, q_row))
        r_m = quat_to_mat3(q_m)
        row = y * IMG_W
        ry = (y - CY) / FY
        for x in range(IMG_W):
            rx = (x - CX) / FX
            wx, wy, wz = mat3_apply(r_m, (rx, ry, 1.0))
            if wz <= 1e-6:
                # Should not occur at this project's small jitter amplitudes
                # (a few degrees) and moderate FOV — guarded defensively
                # rather than silently producing a garbage sample.
                out[row + x] = int(BG_LO)
                continue
            u = wx / wz
            v = wy / wz
            val = texture_intensity(u, v, seed)
            out[row + x] = int(round(min(max(val, 0.0), 255.0)))
    return out


# ---------------------------------------------------------------------------
# write_gyro_csv — one 200 Hz gyro trace over the window, either the TRUE
# rates (clean=True) or with a constant bias + uniform noise added
# (clean=False — the file header's illustrative sensor-error model).
# ---------------------------------------------------------------------------
def write_gyro_csv(path: Path, clean: bool, seed: int) -> int:
    n_samples = int(round((GYRO_WINDOW_T1_S - GYRO_WINDOW_T0_S) / GYRO_DT_S)) + 1
    # A DIFFERENT deterministic stream from the texture hashing (distinct
    # salt) so the noise draws are reproducible independent of image size/
    # texture parameter choices.
    noise_rng = XorShift32((seed * 40503 + 7) & 0xFFFFFFFF)

    with path.open("w", newline="", encoding="utf-8") as f:
        f.write(f"# SYNTHETIC gyro trace ({'clean' if clean else 'degraded'}) for project 01.10, seed={seed}\n")
        f.write(f"# regenerate: python make_synthetic.py --seed {seed}\n")
        if clean:
            f.write("# columns: t_s (frame-relative seconds), wx_rad_s, wy_rad_s, wz_rad_s -- TRUE angular rate, no noise\n")
        else:
            f.write(f"# columns: t_s, wx_rad_s, wy_rad_s, wz_rad_s -- TRUE rate + constant bias "
                    f"({GYRO_BIAS_DPS} deg/s per axis) + uniform noise (+-{GYRO_NOISE_HALF_WIDTH_RAD_S} rad/s), "
                    f"illustrative only, not a calibrated IMU noise model\n")
        writer = csv.writer(f)
        writer.writerow(["t_s", "wx_rad_s", "wy_rad_s", "wz_rad_s"])
        for i in range(n_samples):
            t = GYRO_WINDOW_T0_S + i * GYRO_DT_S
            wx, wy, wz = true_omega(t)
            if not clean:
                wx += GYRO_BIAS_RAD_S + noise_rng.uniform(-GYRO_NOISE_HALF_WIDTH_RAD_S, GYRO_NOISE_HALF_WIDTH_RAD_S)
                wy += GYRO_BIAS_RAD_S + noise_rng.uniform(-GYRO_NOISE_HALF_WIDTH_RAD_S, GYRO_NOISE_HALF_WIDTH_RAD_S)
                wz += GYRO_BIAS_RAD_S + noise_rng.uniform(-GYRO_NOISE_HALF_WIDTH_RAD_S, GYRO_NOISE_HALF_WIDTH_RAD_S)
            writer.writerow([f"{t:.6f}", f"{wx:.8f}", f"{wy:.8f}", f"{wz:.8f}"])
    return n_samples


def write_params_csv(path: Path, seed: int) -> None:
    rows = [
        ("img_w", IMG_W), ("img_h", IMG_H),
        ("fx", FX), ("fy", FY), ("cx", CX), ("cy", CY),
        ("readout_time_s", READOUT_TIME_S), ("line_time_s", LINE_TIME_S),
        ("frame_t0_s", FRAME_T0_S), ("frame_tref_s", FRAME_TREF_S),
        ("gyro_rate_hz", GYRO_RATE_HZ),
        ("gyro_window_t0_s", GYRO_WINDOW_T0_S), ("gyro_window_t1_s", GYRO_WINDOW_T1_S),
        ("rotation_pitch_amp_deg", A_PITCH_DEG), ("rotation_pitch_freq_hz", F_PITCH_HZ), ("rotation_pitch_phase_rad", PHI_PITCH),
        ("rotation_yaw_amp_deg", A_YAW_DEG), ("rotation_yaw_freq_hz", F_YAW_HZ), ("rotation_yaw_phase_rad", PHI_YAW),
        ("rotation_roll_amp_deg", A_ROLL_DEG), ("rotation_roll_freq_hz", F_ROLL_HZ), ("rotation_roll_phase_rad", PHI_ROLL),
        ("gyro_bias_dps", GYRO_BIAS_DPS), ("gyro_noise_half_width_rad_s", GYRO_NOISE_HALF_WIDTH_RAD_S),
        ("seed", seed),
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        f.write("# SYNTHETIC scene parameters for project 01.10 -- provenance record, not read by ../src/main.cu\n")
        f.write("# (the C++ program uses its own baked kernels.cuh constants; this file documents that they MATCH)\n")
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for k, v in rows:
            writer.writerow([k, v])


def write_pgm(path: Path, width: int, height: int, data: bytearray) -> None:
    """8-bit binary PGM (P5) — one grayscale byte per pixel, same format
    used throughout this repo (e.g. 01.01's write_pgm)."""
    with open(path, "wb") as f:
        f.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        f.write(data)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for the background texture + degraded-gyro noise (default 42; the "
                         "geometry/timing/rotation-profile constants above are NOT seed-dependent)")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data" / "sample",
                    help="output directory (default ../data/sample)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[make_synthetic] integrating the TRUE fine rotation trajectory ...")
    fine_traj = build_fine_trajectory()

    print("[make_synthetic] rendering ground_truth_gs.pgm ...")
    ground_truth = render_ground_truth(args.seed)
    write_pgm(args.out_dir / "ground_truth_gs.pgm", IMG_W, IMG_H, ground_truth)

    print("[make_synthetic] rendering rs_input.pgm (per-row row-homography warp) ...")
    rs_frame = render_rs_frame(fine_traj, args.seed)
    write_pgm(args.out_dir / "rs_input.pgm", IMG_W, IMG_H, rs_frame)

    n_clean = write_gyro_csv(args.out_dir / "gyro_clean.csv", clean=True, seed=args.seed)
    n_degraded = write_gyro_csv(args.out_dir / "gyro_degraded.csv", clean=False, seed=args.seed)

    write_params_csv(args.out_dir / "params.csv", args.seed)

    total_bytes = sum((args.out_dir / n).stat().st_size
                      for n in ("rs_input.pgm", "ground_truth_gs.pgm", "gyro_clean.csv", "gyro_degraded.csv", "params.csv"))
    print(f"[make_synthetic] wrote {args.out_dir}: {IMG_W}x{IMG_H} RS + GS frames, "
          f"{n_clean} clean / {n_degraded} degraded gyro samples ({total_bytes} bytes total) - labeled SYNTHETIC")
    print(f"[make_synthetic] readout={READOUT_TIME_S * 1000:.1f} ms, t_line={LINE_TIME_S * 1e6:.2f} us, "
          f"t_ref={FRAME_TREF_S * 1000:.3f} ms, seed={args.seed}")
    if args.out_dir != (Path(__file__).resolve().parent.parent / "data" / "sample"):
        print("[make_synthetic] note: non-default --out-dir - fine for experiments, do NOT commit these files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
