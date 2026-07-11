#!/usr/bin/env python3
# ===========================================================================
# make_synthetic.py — synthetic photometric-calibration rig generator for
#                      project 01.09 (Photometric/vignetting calibration
#                      kernels)
#
# Role in the project
# --------------------
# This is the ONLY place the ground truth exists in closed form. Everything
# downstream (the CUDA program's dark/flat-stack calibration, its parametric
# radial fit, and every verification gate in src/main.cu) is graded against
# what THIS script wrote to data/sample/ — every constant here that the C++
# side also needs is called out with a "MUST MATCH src/main.cu's k...`
# comment, mirroring the discipline in 01.08's make_synthetic.py.
#
# The model (single source of truth; see src/kernels.cuh SECTION 1 for the
# C++ restatement and THEORY.md "The math" for the full derivation):
#
#     I(x, y) = g(x, y) * L(x, y) + o(x, y) + noise
#     g(x, y) = V(x, y) * PRNU(x, y)          (multiplicative field)
#     o(x, y) = BLACK_LEVEL + DSNU(x, y)      (additive field)
#
#   V     — cos^4-law optical vignette, focal ratio FOCAL_EFF_PX, with a
#           small deliberate DECENTERING (CENTER_OFFSET_X/Y_PX) from the
#           geometric image center — a real lens's optical axis is never
#           pixel-perfectly aligned with the sensor's mechanical center.
#   PRNU  — photo-response non-uniformity: a smooth low-frequency term (lens/
#           coating gradient) PLUS a hashed per-pixel term (silicon-level
#           quantum-efficiency variation), amplitude ~+-2% total.
#   DSNU  — dark-signal non-uniformity: a hashed per-pixel fixed-pattern
#           offset (dark-current variation) riding on BLACK_LEVEL, the bulk
#           pedestal every real sensor adds so negative read noise never
#           clips at an unsigned ADC's zero (see THEORY.md "Numerical
#           considerations").
#
# What this generates (stdlib only, CLAUDE.md paragraph 5's "vendored
# header" rule is about C++; a bare `python3` runs this anywhere):
#
#   1) N_DARK=16 dark frames (aperture closed, L=0 everywhere) — write to
#      dark_00.pgm .. dark_15.pgm.
#   2) N_FLAT=16 flat frames (uniform illumination L=L_FLAT) — write to
#      flat_00.pgm .. flat_15.pgm.
#   3) ONE natural test scene (textured background + five flat "gray card"
#      swatches of IDENTICAL true radiance, one centered, four near the
#      corners) — write to scene.pgm.
#   4) The exact ground-truth additive field o(x,y) — dsnu_true.bin — and
#      the exact ground-truth multiplicative field g(x,y) = V*PRNU —
#      gain_true.bin — both raw float32 binary dumps (the 01.08
#      ground_truth_radiance.bin convention: PRNU is a genuinely spatial
#      hashed pattern, not a 1-D closed form, so it is committed as DATA
#      rather than re-derived in C++; the vignette's own cos^4 SHAPE, being
#      a true 1-D closed form, IS independently re-derived in main.cu — see
#      its "kFocalEffPxTrue" constants and the crf_true_g precedent in
#      01.08's main.cu for why that split is the right one).
#   5) params.csv — every constant below, machine-readable, so main.cu (and
#      a human) can cross-check without re-deriving.
#
# Determinism: everything here is seeded (SEED = 42, CLAUDE.md paragraph 8);
# rerunning this script bit-for-bit reproduces every committed sample file.
#
# Read this after: THEORY.md "The problem" and "The math" (the vignette,
# PRNU, and DSNU sections use these exact symbols: V, PRNU, DSNU, g, o).
# ===========================================================================

import csv
import math
import os
import struct

# ---------------------------------------------------------------------------
# XorShift32 — the repo's standard tiny deterministic PRNG (CLAUDE.md
# "MACHINE FACTS": no std::uniform_real_distribution anywhere, C++ or Python,
# so the two languages can never disagree about what "random" means). One
# 32-bit state word, three shifts, full period 2^32-1 over its reachable
# states. State 0 is absorbing (0 xorshifts to 0 forever) so the constructor
# guards against it — same convention as 01.04/01.08's copy of this class.
# ---------------------------------------------------------------------------
class XorShift32:
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF
        if self.state == 0:
            self.state = 0x9E3779B9  # any nonzero value; golden-ratio constant by convention

    def next_u32(self) -> int:
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17)
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x & 0xFFFFFFFF
        return self.state

    def next_unit(self) -> float:
        """Uniform float in [0, 1) — next_u32() scaled by 2^-32."""
        return self.next_u32() / 4294967296.0

    def next_signed(self) -> float:
        """Uniform float in [-1, 1) — for noise fields, zero-mean-ish."""
        return self.next_unit() * 2.0 - 1.0

    def next_gaussian(self) -> float:
        """One N(0,1) draw via Box-Muller (the 08.01 MPPI precedent for a
        cheap, dependency-free Gaussian from this same PRNG family)."""
        u1 = self.next_unit()
        if u1 < 1e-12:
            u1 = 1e-12   # guard log(0) — astronomically unlikely, defensive only
        u2 = self.next_unit()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


# ===========================================================================
# Problem geometry — MUST MATCH src/kernels.cuh's kW/kH/kN and
# src/main.cu's kNumDarkFrames/kNumFlatFrames.
# ===========================================================================
W, H = 160, 120                 # pixels; matches 01.03/01.08's pyramid-friendly precedent
N = W * H                       # 19,200 pixels per frame
SEED = 42                       # repo-wide convention (CLAUDE.md "MACHINE FACTS")

N_DARK = 16                     # dark frames in the dark stack (task brief: "average N=16 dark frames")
N_FLAT = 16                     # flat frames in the flat stack (task brief: "average N=16 flat-field frames")

# ---------------------------------------------------------------------------
# The vignette V(x,y) — cos^4 falloff (THEORY.md derives this from solid
# angle + projected-aperture first principles). FOCAL_EFF_PX is an
# "effective focal length" in PIXEL units (not millimeters — this project
# never calibrates a physical pixel pitch, see README "Limitations &
# honesty"): the ratio r/FOCAL_EFF_PX IS the tangent of the off-axis angle.
# CENTER_OFFSET_*_PX is a small DECENTERING of the true optical axis from
# the sensor's geometric center — realistic (no lens is perfectly aligned to
# the die) and deliberately small enough that the parametric fit's
# assume-geometric-center simplification (main.cu; THEORY.md "The algorithm")
# stays honestly defensible. MUST MATCH src/main.cu's kFocalEffPxTrue /
# kCenterOffsetXTrue / kCenterOffsetYTrue (the independent ground-truth
# re-derivation used ONLY by the radial_fit gate).
# ---------------------------------------------------------------------------
FOCAL_EFF_PX = 200.0
CENTER_OFFSET_X_PX = 3.0
CENTER_OFFSET_Y_PX = -2.0
CX = W / 2.0 + CENTER_OFFSET_X_PX   # true optical center, x (83.0)
CY = H / 2.0 + CENTER_OFFSET_Y_PX   # true optical center, y (58.0)

# ---------------------------------------------------------------------------
# PRNU(x,y) — smooth-plus-hashed multiplicative gain nonuniformity, total
# amplitude ~+-2% (task brief). The smooth term models a lens-coating or
# microlens-array gradient (low spatial frequency); the hashed term models
# silicon-level per-pixel quantum-efficiency variation (THEORY.md "The
# problem" derives both physical sources). Mean ~= 1.0 over the whole image
# by construction (the smooth term integrates to ~0, the hashed term is
# drawn symmetric around 0) — this is what makes "normalize by center
# region" (main.cu) a meaningful way to pin the nonparametric gain map's
# scale (see main.cu's center-ROI comment for the honest caveat that the
# ROI sits near, not exactly at, the true optical center).
# ---------------------------------------------------------------------------
PRNU_SMOOTH_AMPL = 0.010   # +-1% smooth term
PRNU_HASH_AMPL = 0.010     # +-1% hashed term (total PRNU excursion ~+-2%)

# ---------------------------------------------------------------------------
# DSNU(x,y) — hashed dark-signal non-uniformity, amplitude ~+-2 code-value
# units (LSB) per the task brief, riding on a BLACK_LEVEL pedestal. Real
# sensors add exactly this kind of pedestal so that read-noise's negative
# excursions never clip against an unsigned ADC's zero floor — see
# THEORY.md "Numerical considerations" for the honest accounting of why this
# project folds the pedestal INTO "DSNU" (the single additive field o(x,y)
# the dark-stack calibration recovers) rather than modeling it as a separate
# camera register.
# ---------------------------------------------------------------------------
BLACK_LEVEL = 8.0
DSNU_HASH_AMPL = 2.0

# ---------------------------------------------------------------------------
# Sensor noise — read noise (signal-independent Gaussian floor) + shot noise
# (signal-dependent, sigma proportional to sqrt of the LIGHT-GENERATED
# signal only — dark current's own shot noise is folded into the fixed
# READ_NOISE_SIGMA floor here, a documented simplification; THEORY.md
# "Numerical considerations"). This is what the noise_averaging gate's
# 1/sqrt(N) law is measured against.
# ---------------------------------------------------------------------------
READ_NOISE_SIGMA = 1.2
SHOT_NOISE_K = 0.05

# ---------------------------------------------------------------------------
# Illumination levels. L_FLAT is the flat-field target's UNIFORM radiance
# (an integrating sphere or diffuser plate in the real practice this
# teaches toward — see PRACTICE.md). L_SCENE_BASE/AMPL describe the natural
# test scene's textured background; L_SWATCH is the identical true radiance
# baked into all five gray-card swatches (the correction_efficacy gate's
# "identical-albedo patches" — see main.cu). MUST MATCH src/main.cu's
# kLFlatTrue (used only by the noise_averaging gate's independent expected-
# value reconstruction).
# ---------------------------------------------------------------------------
L_FLAT = 180.0
L_SCENE_BASE = 170.0
SCENE_TEXTURE_AMPL = 0.15
SCENE_NOISE_CELL_PX = 20.0
L_SWATCH = 200.0

# Swatch rectangles (x0, x1, y0, y1), half-open — MUST MATCH src/main.cu's
# k*Swatch constants (the correction_efficacy gate reads these same regions
# out of the corrected/uncorrected scene). One centered (near the optical
# axis, where the vignette barely attenuates) and four near the true image
# corners (where the vignette attenuates most) — chosen close enough to the
# corners that the uncorrected disparity lands in the task brief's
# documented ~30-40% range (measured, not assumed; see main.cu's [info] line).
SWATCH_CENTER = (72, 88, 52, 68)
SWATCH_TL = (4, 20, 4, 20)
SWATCH_TR = (140, 156, 4, 20)
SWATCH_BL = (4, 20, 100, 116)
SWATCH_BR = (140, 156, 100, 116)
SWATCHES = {
    "CENTER": SWATCH_CENTER, "TL": SWATCH_TL, "TR": SWATCH_TR,
    "BL": SWATCH_BL, "BR": SWATCH_BR,
}


def hashed_unit(x: int, y: int, salt: int) -> float:
    """Deterministic, per-INTEGER-pixel pseudo-random value in [-1, 1].

    Unlike value_noise() below (which interpolates for a SMOOTH field),
    PRNU's hashed term and DSNU are meant to be uncorrelated pixel-to-pixel
    (silicon-level variation has no reason to be spatially smooth), so this
    draws ONE fresh xorshift32 stream per integer (x, y, salt) triple and
    takes its first value — the same "combine coordinates + salt into one
    seed" idiom 01.04/01.08 use, reused here for a discrete rather than
    continuous field.
    """
    seed = (x * 374761393 + y * 668265263 + salt * 2246822519) & 0xFFFFFFFF
    return XorShift32(seed).next_signed()


def value_noise(x: float, y: float, salt: int) -> float:
    """Smooth, deterministic hashed lattice noise in [-1, 1] at continuous
    (x, y), via bilinear interpolation between four hashed cell-corner
    values — the exact technique 01.08's make_synthetic.py uses for its
    outdoor scene texture, reused here for the test scene's background.
    """
    cell = SCENE_NOISE_CELL_PX
    cxf, cyf = x / cell, y / cell
    x0, y0 = math.floor(cxf), math.floor(cyf)
    fx, fy = cxf - x0, cyf - y0

    def corner(ix: int, iy: int) -> float:
        return hashed_unit(int(ix), int(iy), salt)

    h00, h10 = corner(x0, y0), corner(x0 + 1, y0)
    h01, h11 = corner(x0, y0 + 1), corner(x0 + 1, y0 + 1)
    h0 = h00 * (1.0 - fx) + h10 * fx
    h1 = h01 * (1.0 - fx) + h11 * fx
    return h0 * (1.0 - fy) + h1 * fy


def vignette_v(x: float, y: float) -> float:
    """True optical vignette V(x,y) = cos^4(theta), theta = angle off the
    TRUE optical axis (CX, CY) — the cos^4 law derived in THEORY.md "The
    problem" from solid-angle foreshortening + the projected-aperture
    argument. (x, y) are continuous PIXEL-CENTER coordinates (integer pixel
    p sampled at p+0.5 — see radiance/gain evaluation below)."""
    dx, dy = x - CX, y - CY
    r = math.hypot(dx, dy)
    theta = math.atan2(r, FOCAL_EFF_PX)
    c = math.cos(theta)
    return c * c * c * c


def prnu(x: int, y: int) -> float:
    """True photo-response non-uniformity PRNU(x,y) — smooth term (a
    low-frequency sinusoidal stand-in for a coating/microlens gradient) plus
    a hashed per-pixel term (quantum-efficiency variation). See the
    PRNU_SMOOTH_AMPL/PRNU_HASH_AMPL module comment."""
    xf, yf = x + 0.5, y + 0.5   # pixel-CENTER sampling, matching vignette_v
    smooth = PRNU_SMOOTH_AMPL * math.sin(2.0 * math.pi * 3.0 * xf / W) \
                               * math.cos(2.0 * math.pi * 2.0 * yf / H)
    hashed = PRNU_HASH_AMPL * hashed_unit(x, y, salt=101)
    return 1.0 + smooth + hashed


def dsnu(x: int, y: int) -> float:
    """True additive dark-signal field o(x,y) = BLACK_LEVEL + hashed DSNU
    term. See the BLACK_LEVEL/DSNU_HASH_AMPL module comment."""
    return BLACK_LEVEL + DSNU_HASH_AMPL * hashed_unit(x, y, salt=202)


def gain_true(x: int, y: int) -> float:
    """True multiplicative field g(x,y) = V(x,y) * PRNU(x,y) — exactly what
    src/main.cu's gain_recovery gate grades the nonparametric flat-field map
    against (loaded directly from gain_true.bin, see the file header)."""
    return vignette_v(x + 0.5, y + 0.5) * prnu(x, y)


def scene_radiance(x: int, y: int) -> float:
    """True scene radiance L(x,y) for the ONE test frame: textured
    background, with the five SWATCHES overridden to a flat, texture-free
    L_SWATCH (a real reflectance/gray-card standard reads far more uniform
    than a general scene — see 01.08's painted-line precedent for the same
    "flat calibration feature carved out of a textured scene" idiom)."""
    for (x0, x1, y0, y1) in SWATCHES.values():
        if x0 <= x < x1 and y0 <= y < y1:
            return L_SWATCH
    xf, yf = x + 0.5, y + 0.5
    return L_SCENE_BASE * (1.0 + SCENE_TEXTURE_AMPL * value_noise(xf, yf, salt=303))


def render_frame(light_signal_field, noise_seed: int, dsnu_field, out: bytearray) -> None:
    """Render ONE noisy 8-bit frame in place into `out` (row-major W*H bytes).

    Parameters:
      light_signal_field — callable(x,y) -> gain_true(x,y)*L(x,y), the
                            PHOTO-GENERATED signal (drives shot noise).
      noise_seed          — this frame's independent PRNG seed (derived from
                             SEED + a per-frame offset — see callers).
      dsnu_field           — callable(x,y) -> o(x,y), the additive offset.
      out                  — pre-sized bytearray(W*H), overwritten.

    Noise model (see READ_NOISE_SIGMA/SHOT_NOISE_K's module comment):
        sigma_total = sqrt(READ_NOISE_SIGMA^2 + (SHOT_NOISE_K^2)*light_signal)
        pixel = round(clamp(light_signal + o(x,y) + N(0, sigma_total), 0, 255))
    """
    rng = XorShift32(noise_seed)
    for y in range(H):
        for x in range(W):
            light = light_signal_field(x, y)
            sigma_shot = SHOT_NOISE_K * math.sqrt(light) if light > 0.0 else 0.0
            sigma_total = math.sqrt(READ_NOISE_SIGMA * READ_NOISE_SIGMA + sigma_shot * sigma_shot)
            noisy = light + dsnu_field(x, y) + rng.next_gaussian() * sigma_total
            clipped = 0 if noisy < 0.0 else (255 if noisy > 255.0 else int(noisy + 0.5))
            out[y * W + x] = clipped


def write_pgm(path: str, width: int, height: int, gray: bytearray) -> None:
    """Minimal binary PGM (P5) writer — same on-disk convention as
    01.01/01.03/01.08's write_pgm, reimplemented independently here (Python
    side, never shared with the C++ reader — CLAUDE.md paragraph 8)."""
    with open(path, "wb") as f:
        f.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        f.write(bytes(gray))


def write_float_binary(path: str, values) -> None:
    """Raw float32 binary dump, row-major (y*W+x), little-endian IEEE-754,
    NO header (documented once in data/README.md — the C++ reader is then a
    single fread-equivalent call, the same convention 01.08's
    ground_truth_radiance.bin uses)."""
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}f", *values))


def main() -> None:
    sample_dir = os.path.join(os.path.dirname(__file__), "..", "data", "sample")
    os.makedirs(sample_dir, exist_ok=True)

    stale = os.path.join(sample_dir, "saxpy_sample.csv")
    if os.path.exists(stale):
        os.remove(stale)   # this project reads no such file — remove the scaffold-era placeholder

    # ---- precompute the two ground-truth fields ONCE (every frame reuses
    #      them; only the per-frame NOISE differs) -------------------------
    gain_field = [[gain_true(x, y) for x in range(W)] for y in range(H)]
    dsnu_field = [[dsnu(x, y) for x in range(W)] for y in range(H)]
    scene_field = [[scene_radiance(x, y) for x in range(W)] for y in range(H)]

    gain_flat = [gain_field[y][x] for y in range(H) for x in range(W)]
    dsnu_flat = [dsnu_field[y][x] for y in range(H) for x in range(W)]
    write_float_binary(os.path.join(sample_dir, "gain_true.bin"), gain_flat)
    write_float_binary(os.path.join(sample_dir, "dsnu_true.bin"), dsnu_flat)
    print(f"[make_synthetic] gain_true range: {min(gain_flat):.4f} .. {max(gain_flat):.4f} "
          f"(vignette+PRNU combined multiplicative field)")
    print(f"[make_synthetic] dsnu_true range: {min(dsnu_flat):.4f} .. {max(dsnu_flat):.4f} "
          f"(black level {BLACK_LEVEL} + hashed dark-signal pattern)")

    def light_dark(x, y):
        return 0.0   # aperture closed: no photo-generated signal at all

    def light_flat(x, y):
        return gain_field[y][x] * L_FLAT

    def light_scene(x, y):
        return gain_field[y][x] * scene_field[y][x]

    def dsnu_lookup(x, y):
        return dsnu_field[y][x]

    # ---- dark stack ---------------------------------------------------------
    for i in range(N_DARK):
        buf = bytearray(N)
        render_frame(light_dark, SEED + 1000 * i, dsnu_lookup, buf)
        write_pgm(os.path.join(sample_dir, f"dark_{i:02d}.pgm"), W, H, buf)
    print(f"[make_synthetic] wrote {N_DARK} dark_*.pgm frames (aperture closed, L=0, "
          f"expected value = o(x,y) only)")

    # ---- flat stack -----------------------------------------------------
    for i in range(N_FLAT):
        buf = bytearray(N)
        render_frame(light_flat, SEED + 2000 * i, dsnu_lookup, buf)
        write_pgm(os.path.join(sample_dir, f"flat_{i:02d}.pgm"), W, H, buf)
    print(f"[make_synthetic] wrote {N_FLAT} flat_*.pgm frames (uniform L_FLAT={L_FLAT}, "
          f"expected value = g(x,y)*L_FLAT + o(x,y))")

    # ---- the one natural test scene --------------------------------------
    buf = bytearray(N)
    render_frame(light_scene, SEED + 9999, dsnu_lookup, buf)
    write_pgm(os.path.join(sample_dir, "scene.pgm"), W, H, buf)
    print(f"[make_synthetic] wrote scene.pgm (textured background + 5 identical-radiance "
          f"L_SWATCH={L_SWATCH} gray-card swatches: CENTER + 4 corners)")

    # ---- params.csv: every constant a downstream reader might need -------
    params_path = os.path.join(sample_dir, "params.csv")
    with open(params_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        w.writerow(["W", W]); w.writerow(["H", H])
        w.writerow(["N_DARK", N_DARK]); w.writerow(["N_FLAT", N_FLAT])
        w.writerow(["SEED", SEED])
        w.writerow(["FOCAL_EFF_PX", FOCAL_EFF_PX])
        w.writerow(["CENTER_OFFSET_X_PX", CENTER_OFFSET_X_PX])
        w.writerow(["CENTER_OFFSET_Y_PX", CENTER_OFFSET_Y_PX])
        w.writerow(["PRNU_SMOOTH_AMPL", PRNU_SMOOTH_AMPL]); w.writerow(["PRNU_HASH_AMPL", PRNU_HASH_AMPL])
        w.writerow(["BLACK_LEVEL", BLACK_LEVEL]); w.writerow(["DSNU_HASH_AMPL", DSNU_HASH_AMPL])
        w.writerow(["READ_NOISE_SIGMA", READ_NOISE_SIGMA]); w.writerow(["SHOT_NOISE_K", SHOT_NOISE_K])
        w.writerow(["L_FLAT", L_FLAT]); w.writerow(["L_SCENE_BASE", L_SCENE_BASE])
        w.writerow(["SCENE_TEXTURE_AMPL", SCENE_TEXTURE_AMPL]); w.writerow(["L_SWATCH", L_SWATCH])
        for name, rect in SWATCHES.items():
            w.writerow([f"SWATCH_{name}", f"{rect}"])

    print(f"[make_synthetic] wrote params.csv, {N_DARK}+{N_FLAT} calibration frames, scene.pgm, "
          f"gain_true.bin, dsnu_true.bin into {os.path.abspath(sample_dir)}")


if __name__ == "__main__":
    main()
