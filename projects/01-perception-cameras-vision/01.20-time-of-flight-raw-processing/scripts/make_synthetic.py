#!/usr/bin/env python3
"""make_synthetic.py — synthetic continuous-wave indirect-ToF (iToF) raw tap
generator for project 01.20 (Time-of-flight raw processing: phase
unwrapping, flying-pixel removal).

Why this script exists (CLAUDE.md §8: synthetic-first)
--------------------------------------------------------
A real iToF camera needs a modulated illuminator, a lock-in-style sensor,
and a physical scene. This script instead RENDERS one, analytically, from
first principles, so the demo runs offline, the ground truth (true metric
depth, which analytic surface produced each pixel, and — uniquely to this
project — whether a pixel's return was a genuine PHASOR MIXTURE of two
surfaces) is EXACT, and the whole 8-frame tap stack regenerates byte-for-byte
from one fixed seed.

What it renders — THE SENSOR MODEL (mirrors ../src/kernels.cuh EXACTLY; if
you change one, change the other — main.cu asserts params.csv against the
compiled constants and fails loudly on drift)
--------------------------------------------------------------------------
  * A pinhole CAMERA (kCamW x kCamH) looking at a SCENE of three analytic
    surfaces sharing one depth buffer: a tilted BACKGROUND plane (far), a
    SPHERE, and a fronto-parallel BOX top face (both near) — the same
    surface trio 01.19 uses, re-used here specifically so the two projects'
    reconstruction gates (plane RMS / sphere radius / step height) are
    directly comparable "same test, different sensor" (README cites this).
  * Depths are chosen ROOM-SCALE (1.5-5.6 m) and DELIBERATELY so the
    background's high-frequency (60 MHz) wrapped phase lands far from the
    foreground objects' wrapped phase (see "Depth budget" below) — this is
    what makes flying pixels at the object silhouettes a DRAMATIC, clearly
    visible phasor mixture rather than a near-invisible near-alias.
  * Each camera pixel is a small AREA, not a ray: this script supersamples
    every pixel on a 4x4 sub-pixel grid (16 rays), buckets the sub-rays by
    which analytic surface they hit, and — for a pixel whose sub-rays hit
    MORE THAN ONE surface (a silhouette edge) — sums the surfaces'
    modulation PHASORS with AREA WEIGHTS. Because a real ToF pixel's 4
    correlation taps are LINEAR in incident correlated power, this phasor
    sum is exactly what a real sensor's charge wells would integrate over
    that same mixed footprint — see "Pattern rendering" below for the
    closed-form derivation this script implements.
  * Ground truth per camera pixel: the CENTER-RAY true depth (m) and
    surface id (for the reconstruction gates — computed from a single ray
    at the pixel center, NOT the supersampled mixture, exactly mirroring
    01.19's `cast_pixel`), PLUS a genuinely independent
    `is_flying_truth` flag: true iff the SUPERSAMPLED analysis found at
    least two surfaces each claiming a non-trivial share of the pixel's
    area. This label is computed from generator-internal knowledge (sub-ray
    surface membership) that the C++ detection pipeline never sees — the
    same "grading label is not a decoding input" independence 01.19's
    truth_surface enjoys, and precisely what lets the flying_pixel gate in
    main.cu score precision/recall without circularity.

Determinism (CLAUDE.md §12; project MACHINE FACTS: stdlib-only, xorshift32,
seed 42) — no third-party packages (no numpy); the same xorshift32 +
Box-Muller generator used throughout this repo's C++ demos (e.g. 08.01,
01.19) is re-implemented here in pure Python so noise draws come from the
exact bit-manipulation algorithm a learner also meets in every .cu file.

Usage
-----
    python make_synthetic.py                  # defaults: writes ../data/sample/
    python make_synthetic.py --seed 42 --out ../data/sample
"""

import argparse
import math
import struct
from pathlib import Path

# =============================================================================
# Camera + ToF constants — MUST MATCH ../src/kernels.cuh (hand-synced; see
# file header). main.cu reads params.csv and asserts these against the
# compiled kernels.cuh constants before doing anything else with the data.
# =============================================================================
CAM_W, CAM_H = 160, 120                  # camera resolution (px) == kCamW, kCamH
CAM_FX, CAM_FY = 150.0, 150.0            # camera focal length (px) == kCamFx/Fy
CAM_CX, CAM_CY = 80.0, 60.0              # camera principal point (px) == kCamCx/Cy

SPEED_OF_LIGHT_MPS = 299792458.0         # exact, m/s == kSpeedOfLightMps
FREQ1_HZ = 60.0e6                        # fine channel == kFreq1Hz
FREQ2_HZ = 20.0e6                        # coarse channel == kFreq2Hz (FREQ1_HZ/FREQ2_HZ == 3 exactly)
AMBIG1_M = SPEED_OF_LIGHT_MPS / (2.0 * FREQ1_HZ)   # == kAmbig1M, ~2.498 m
AMBIG2_M = SPEED_OF_LIGHT_MPS / (2.0 * FREQ2_HZ)   # == kAmbig2M, ~7.495 m
NUM_TAPS = 4                             # == kNumTaps (0/90/180/270 deg demodulation)

MAX_SCENE_DEPTH_M = 6.0                  # == kMaxSceneDepthM (asserted below: every rendered depth < this)

# =============================================================================
# Scene — three analytic surfaces sharing one depth buffer (camera-frame ==
# world-frame, meters, exactly 01.19's convention). Depths were picked (see
# the diagnostic printout this script emits) so that:
#   (a) the background sits at 5.0-5.6 m — well past kAmbig1M (~2.5 m), so
#       the SINGLE-frequency (60 MHz) depth genuinely wraps there (TWICE:
#       floor(5.2/2.498) == 2) — the designed aliasing_demo gate exploits
#       exactly this;
#   (b) the box and sphere sit at 1.5 m and ~2.05 m respectively — both
#       WELL UNDER kAmbig1M, so single-frequency ranging is already correct
#       there (no wrap) — the aliasing problem is specifically a FAR-SCENE
#       problem, exactly as it is on real single-frequency CW ToF cameras;
#   (c) the background's WRAPPED (mod kAmbig1M) phase sits far from the
#       foreground objects' phase at most silhouette pixels, so the
#       phasor-mixing math at flying-pixel edges produces a genuinely
#       different (not near-identical) mixed phase — a dramatic, legible
#       demo rather than a coincidental near-miss.
# =============================================================================
SURF_NONE, SURF_BACKGROUND, SURF_SPHERE, SURF_BOX = 0, 1, 2, 3

# Background: height-field plane Z = BG_Z0 + BG_AX*X (01.19's convention:
# algebraically simpler than a general plane-normal intersection, and exactly
# equivalent for a plane that is a function of camera rays).
BG_Z0 = 5.246              # depth at world X=0 (m)
BG_AX = 0.1051042353        # dZ/dX slope == tan(6 deg): a visible-but-gentle tilt
BG_ALBEDO = 0.70            # diffuse reflectance, unitless in [0,1]

# Sphere: ordinary ray-sphere intersection.
SPHERE_C = (0.35, -0.15, 2.05)    # center (m), camera frame
SPHERE_R = 0.30                    # radius (m)
SPHERE_ALBEDO = 0.60

# Box: fronto-parallel top face only (01.19's didactic simplification — a
# clean depth STEP without modeling a full 3-D box's side walls).
BOX_Z = 1.5
BOX_X_RANGE = (-0.70, -0.15)
BOX_Y_RANGE = (-0.30, 0.25)
BOX_ALBEDO = 0.50

# Low-reflectivity ("dark") cohort — the amplitude-mask honesty test
# (README/THEORY "dark_cohort"). A rectangular patch of the BACKGROUND in
# camera-pixel space, chosen to avoid the sphere/box silhouettes entirely
# (diagnostic printout below confirms zero overlap on the committed seed).
DARK_PATCH_ROWS = (90, 110)
DARK_PATCH_COLS = (90, 150)
DARK_ALBEDO = 0.04           # near-black — the whole point of the cohort

# Radiometry: captured_tap_k = AMBIENT + albedo*GAIN*cos(phase + k*pi/2) + noise.
# Flat gain model (no 1/z^2 falloff) — the SAME simplification 01.19 makes
# for its own radiometric model; documented honestly in README "Limitations".
AMBIENT = 100.0              # constant ambient/DC offset (intensity counts, 0-255 scale)
GAIN = 90.0                  # albedo-to-amplitude gain (counts); chosen so AMBIENT +/- GAIN*albedo_max stays in [0,255]
NOISE_SIGMA = 3.0            # per-pixel, per-tap Gaussian sensor noise (counts): photon shot + read noise, combined

# Supersampling: SS x SS sub-rays per pixel, used ONLY to compute each
# pixel's surface-membership AREA WEIGHTS for the phasor-mixing forward
# model (never used as the "ground truth" depth — see file header).
SS = 4                        # 16 sub-rays/pixel; ~307k ray casts total, seconds in pure Python

# Ground-truth flying-pixel definition (generator-internal; the C++ pipeline
# never sees sub-ray membership): a pixel is a TRUE flying pixel iff at
# least 2 surfaces appear among its SS*SS sub-rays AND the second-largest
# surface's weight is >= this floor (rejects single-stray-subsample noise
# at a corner as "not really mixed").
TRUTH_MIN_SECOND_WEIGHT = 2.0 / (SS * SS)   # 2 of 16 sub-rays, i.e. >= 12.5% area

DEFAULT_SEED = 42


# =============================================================================
# xorshift32 + Box-Muller — the repo's portable deterministic RNG (see file
# header). Bit-for-bit the same algorithm as 01.19's / 08.01's generators.
# =============================================================================
class Xorshift32:
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF
        if self.state == 0:
            self.state = 0x9E3779B9

    def next_u32(self) -> int:
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17)
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x & 0xFFFFFFFF
        return self.state

    def uniform01(self) -> float:
        return (self.next_u32() >> 8) * (1.0 / 16777216.0) + (0.5 / 16777216.0)

    def gaussian(self, sigma: float) -> float:
        u1 = self.uniform01()
        u2 = self.uniform01()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return sigma * z


# =============================================================================
# Ray casting — one (sub-)ray -> the winning surface's (depth, surface_id).
# Identical primitive math to 01.19 (background height-field, box constant-
# depth footprint, sphere quadratic), re-typed here independently since this
# script is its own codebase (the same "independent verification tier"
# 01.19's reference_cpu.cpp header argues for — see main.cu's gates, which
# check the C++ pipeline against THIS Python ground truth, not just against
# itself).
# =============================================================================
def ray_dir(px: float, py: float):
    """Pixel-space point (sub-pixel-accurate) -> camera ray direction
    (dx, dy, implicit dz=1). MUST match kernels.cuh's pixel-center convention
    when px=col+0.5, py=row+0.5 (the CENTER ray used for ground truth)."""
    dx = (px - CAM_CX) / CAM_FX
    dy = (py - CAM_CY) / CAM_FY
    return dx, dy


def intersect_background(dx: float, dy: float):
    denom = 1.0 - BG_AX * dx
    if denom <= 1e-6:
        return None
    t = BG_Z0 / denom
    return t if t > 0.0 else None


def intersect_box(dx: float, dy: float):
    t = BOX_Z
    x, y = t * dx, t * dy
    if BOX_X_RANGE[0] <= x <= BOX_X_RANGE[1] and BOX_Y_RANGE[0] <= y <= BOX_Y_RANGE[1]:
        return t
    return None


def intersect_sphere(dx: float, dy: float):
    dz = 1.0
    cx, cy, cz = SPHERE_C
    lx, ly, lz = -cx, -cy, -cz
    a = dx * dx + dy * dy + dz * dz
    b = 2.0 * (dx * lx + dy * ly + dz * lz)
    c = lx * lx + ly * ly + lz * lz - SPHERE_R * SPHERE_R
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    lo, hi = (t1, t2) if t1 < t2 else (t2, t1)
    if lo > 1e-6:
        return lo
    if hi > 1e-6:
        return hi
    return None


def cast_ray(dx: float, dy: float):
    """Depth-buffer resolve for ONE ray: smallest positive t wins (closest to
    camera). Returns (t, surface_id); the background is an (approximately)
    infinite height field within this camera's FOV, so every ray resolves to
    SOME surface (no SURF_NONE in practice on this scene, verified below)."""
    candidates = []
    t_bg = intersect_background(dx, dy)
    if t_bg is not None:
        candidates.append((t_bg, SURF_BACKGROUND))
    t_box = intersect_box(dx, dy)
    if t_box is not None:
        candidates.append((t_box, SURF_BOX))
    t_sph = intersect_sphere(dx, dy)
    if t_sph is not None:
        candidates.append((t_sph, SURF_SPHERE))
    if not candidates:
        return None
    return min(candidates, key=lambda p: p[0])


def albedo_of(surf: int, row: int, col: int) -> float:
    """Per-surface albedo, including the dark-cohort override on the
    background (camera-pixel-space footprint, same convention as 01.19's
    dark stripe)."""
    if surf == SURF_BACKGROUND:
        in_patch = (DARK_PATCH_ROWS[0] <= row < DARK_PATCH_ROWS[1] and
                    DARK_PATCH_COLS[0] <= col < DARK_PATCH_COLS[1])
        return DARK_ALBEDO if in_patch else BG_ALBEDO
    if surf == SURF_SPHERE:
        return SPHERE_ALBEDO
    if surf == SURF_BOX:
        return BOX_ALBEDO
    return 0.0


# =============================================================================
# Phasor-mixing forward model — THE core of this project's synthetic ToF
# renderer (kernels.cuh file header derives the physics; this is its direct
# numeric implementation).
#
# A pixel's SS*SS sub-rays are bucketed by surface id into up to 3 groups.
# Group g contributes weight w_g = count_g/(SS*SS), mean depth z_g (the
# average TRUE depth of its sub-rays -- the plane/sphere both vary smoothly
# enough within one pixel that this is an accurate area-weighted centroid),
# and a fixed albedo a_g. For frequency `freq` (ambiguity range
# `ambig = c/(2*freq)`), group g's WRAPPED phase is
#
#     phase_g = 2*pi * frac(z_g / ambig)          frac(x) = x - floor(x)
#
# (the SAME phase/depth relation `kernels.cuh` fixes: distance = c*phase /
# (4*pi*f) inverted for phase). The pixel's tap k (k=0..3, offsets
# 0/90/180/270 deg -- kernels.cuh's C_k(phi) = A + B*cos(phi + k*pi/2)
# convention) is then the AREA-WEIGHTED SUM over groups:
#
#     tap_k = AMBIENT + sum_g [ w_g * a_g * GAIN * cos(phase_g + k*pi/2) ]
#
# This is a literal phasor addition: each group's (cos, sin) pair is a 2-D
# vector of length (a_g*GAIN) at angle phase_g, and summing tap
# CONTRIBUTIONS linearly (as a real correlator would, since it integrates
# incident optical power linearly) is EXACTLY summing those vectors with
# real-valued weights w_g. For a clean (single-group) pixel this collapses
# to the ordinary single-surface formula; for a MIXED pixel, the resulting
# phase/amplitude recovered by atan2/sqrt downstream is, in general, NEITHER
# surface's true phase NOR their depth average -- see THEORY.md "The
# problem" for the full worked numeric example.
# =============================================================================
def render_taps_for_pixel(groups, freq: float):
    """groups: list of (weight, albedo, mean_depth). Returns 4 ideal
    (noise-free, unquantized) tap values for the given modulation frequency."""
    ambig = SPEED_OF_LIGHT_MPS / (2.0 * freq)
    taps_ideal = [AMBIENT, AMBIENT, AMBIENT, AMBIENT]
    for (w, a, z) in groups:
        frac = (z / ambig) - math.floor(z / ambig)     # z mod ambig, normalized to [0,1)
        phase_g = 2.0 * math.pi * frac
        amp_g = w * a * GAIN
        for k in range(NUM_TAPS):
            taps_ideal[k] += amp_g * math.cos(phase_g + k * (math.pi / 2.0))
    return taps_ideal


def write_pgm(path: Path, img_u8, w: int, h: int):
    with path.open("wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        f.write(bytes(img_u8))


def generate(seed: int, out_dir: Path, verbose: bool = True):
    out_dir.mkdir(parents=True, exist_ok=True)
    w, h, n = CAM_W, CAM_H, CAM_W * CAM_H
    rng = Xorshift32(seed)

    # ---- 1) Ground truth: ONE center ray per pixel (never supersampled) ---
    truth_depth = [0.0] * n
    truth_surface = [SURF_NONE] * n
    surf_counts = {SURF_NONE: 0, SURF_BACKGROUND: 0, SURF_SPHERE: 0, SURF_BOX: 0}
    max_truth_depth = 0.0
    for row in range(h):
        for col in range(w):
            i = row * w + col
            dx, dy = ray_dir(col + 0.5, row + 0.5)
            hit = cast_ray(dx, dy)
            if hit is None:
                surf_counts[SURF_NONE] += 1
                continue
            t, surf = hit
            truth_depth[i] = t
            truth_surface[i] = surf
            surf_counts[surf] += 1
            max_truth_depth = max(max_truth_depth, t)

    if max_truth_depth >= MAX_SCENE_DEPTH_M:
        raise SystemExit(f"[make_synthetic] FATAL: max truth depth {max_truth_depth:.3f} m "
                          f">= MAX_SCENE_DEPTH_M {MAX_SCENE_DEPTH_M} m -- kMaxWraps1/2 bounds in "
                          f"kernels.cuh would be violated. Adjust the scene or the budget.")

    # ---- 2) Per-pixel supersampled surface-membership groups (for both the
    #         phasor-mixing render AND the independent truth_flying label) --
    is_flying_truth = [0] * n
    group_lists = [None] * n   # cache: list[(weight, albedo, mean_depth)] per pixel
    flying_edge_count = 0
    for row in range(h):
        for col in range(w):
            i = row * w + col
            buckets = {}   # surface_id -> [count, depth_sum]
            for sj in range(SS):
                for si in range(SS):
                    px = col + (si + 0.5) / SS
                    py = row + (sj + 0.5) / SS
                    dx, dy = ray_dir(px, py)
                    hit = cast_ray(dx, dy)
                    if hit is None:
                        continue
                    t, surf = hit
                    b = buckets.setdefault(surf, [0, 0.0])
                    b[0] += 1
                    b[1] += t
            total = float(SS * SS)
            groups = []
            for surf, (cnt, depth_sum) in buckets.items():
                weight = cnt / total
                mean_depth = depth_sum / cnt
                groups.append((weight, albedo_of(surf, row, col), mean_depth))
            group_lists[i] = groups
            if len(groups) >= 2:
                weights_sorted = sorted((g[0] for g in groups), reverse=True)
                if weights_sorted[1] >= TRUTH_MIN_SECOND_WEIGHT:
                    is_flying_truth[i] = 1
                    flying_edge_count += 1

    if verbose:
        print(f"[make_synthetic] surface pixel counts (center-ray truth): "
              f"background={surf_counts[SURF_BACKGROUND]} sphere={surf_counts[SURF_SPHERE]} "
              f"box={surf_counts[SURF_BOX]} none={surf_counts[SURF_NONE]}")
        print(f"[make_synthetic] max truth depth: {max_truth_depth:.3f} m "
              f"(budget kMaxSceneDepthM={MAX_SCENE_DEPTH_M} m)")
        print(f"[make_synthetic] true flying (mixed-return) pixels: {flying_edge_count} / {n} "
              f"({100.0 * flying_edge_count / n:.2f}%)")

    # ---- 3) Render the 8-frame tap stack: 2 frequencies x 4 taps ----------
    manifest = []
    for freq_idx, freq in enumerate((FREQ1_HZ, FREQ2_HZ), start=1):
        tap_imgs = [bytearray(n) for _ in range(NUM_TAPS)]
        for i in range(n):
            ideal = render_taps_for_pixel(group_lists[i], freq)
            for k in range(NUM_TAPS):
                v = ideal[k] + rng.gaussian(NOISE_SIGMA)
                iv = int(round(v))
                iv = 0 if iv < 0 else (255 if iv > 255 else iv)
                tap_imgs[k][i] = iv
        for k in range(NUM_TAPS):
            fname = f"tof_f{freq_idx}_tap{k}.pgm"
            write_pgm(out_dir / fname, tap_imgs[k], w, h)
            manifest.append((fname, f"frequency {freq_idx} ({freq/1e6:.0f} MHz), tap {k} ({k*90} deg)"))

    # ---- 4) Ground-truth binaries: depth (f32), surface (u8), flying (u8) -
    with (out_dir / "truth_depth.bin").open("wb") as f:
        f.write(struct.pack(f"<{n}f", *truth_depth))
    with (out_dir / "truth_surface.bin").open("wb") as f:
        f.write(struct.pack(f"<{n}B", *truth_surface))
    with (out_dir / "truth_flying.bin").open("wb") as f:
        f.write(struct.pack(f"<{n}B", *is_flying_truth))

    # ---- 5) params.csv -- sensor/scene constants, scene truth, provenance -
    xc_box = 0.5 * (BOX_X_RANGE[0] + BOX_X_RANGE[1])
    bg_z_at_box = BG_Z0 + BG_AX * xc_box
    step_height_truth = bg_z_at_box - BOX_Z

    params_path = out_dir / "params.csv"
    with params_path.open("w", newline="", encoding="utf-8") as f:
        f.write("# SYNTHETIC data - generated by scripts/make_synthetic.py for project 01.20\n")
        f.write(f"# regenerate: python make_synthetic.py --seed {seed}\n")
        f.write("# key,value  (units in the key name; all geometry is camera-frame == world-frame)\n")
        rows = [
            ("cam_w_px", CAM_W), ("cam_h_px", CAM_H),
            ("cam_fx_px", CAM_FX), ("cam_fy_px", CAM_FY),
            ("cam_cx_px", CAM_CX), ("cam_cy_px", CAM_CY),
            ("freq1_hz", FREQ1_HZ), ("freq2_hz", FREQ2_HZ), ("num_taps", NUM_TAPS),
            ("max_scene_depth_m", MAX_SCENE_DEPTH_M),
            ("seed", seed),
            ("bg_z0_m", BG_Z0), ("bg_ax_slope", BG_AX), ("bg_albedo", BG_ALBEDO),
            ("sphere_cx_m", SPHERE_C[0]), ("sphere_cy_m", SPHERE_C[1]), ("sphere_cz_m", SPHERE_C[2]),
            ("sphere_radius_m_truth", SPHERE_R), ("sphere_albedo", SPHERE_ALBEDO),
            ("box_z_m", BOX_Z),
            ("box_x_min_m", BOX_X_RANGE[0]), ("box_x_max_m", BOX_X_RANGE[1]),
            ("box_y_min_m", BOX_Y_RANGE[0]), ("box_y_max_m", BOX_Y_RANGE[1]),
            ("box_albedo", BOX_ALBEDO), ("step_height_m_truth", step_height_truth),
            ("dark_patch_row_min", DARK_PATCH_ROWS[0]), ("dark_patch_row_max", DARK_PATCH_ROWS[1]),
            ("dark_patch_col_min", DARK_PATCH_COLS[0]), ("dark_patch_col_max", DARK_PATCH_COLS[1]),
            ("dark_albedo", DARK_ALBEDO),
            ("ambient_counts", AMBIENT), ("gain_counts", GAIN), ("noise_sigma_counts", NOISE_SIGMA),
            ("supersample_factor", SS),
            ("surf_background_pixels", surf_counts[SURF_BACKGROUND]),
            ("surf_sphere_pixels", surf_counts[SURF_SPHERE]),
            ("surf_box_pixels", surf_counts[SURF_BOX]),
            ("surf_none_pixels", surf_counts[SURF_NONE]),
            ("true_flying_pixels", flying_edge_count),
        ]
        for k, v in rows:
            f.write(f"{k},{v}\n")

    if verbose:
        print(f"[make_synthetic] step height truth (box vs local background): {step_height_truth * 1000.0:.1f} mm")
        print(f"[make_synthetic] wrote {len(manifest) + 4} files to {out_dir} (seed={seed}, labeled SYNTHETIC)")

    return {
        "manifest": manifest,
        "surf_counts": surf_counts,
        "flying_edge_count": flying_edge_count,
        "step_height_truth": step_height_truth,
    }


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_out = script_dir.parent / "data" / "sample"

    parser = argparse.ArgumentParser(
        description="Generate the synthetic iToF tap stack for project 01.20.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"RNG seed for byte-identical reproducibility (default {DEFAULT_SEED})")
    parser.add_argument("--out", type=Path, default=default_out,
                        help="output directory (default: ../data/sample)")
    args = parser.parse_args()

    generate(args.seed, args.out)


if __name__ == "__main__":
    main()
