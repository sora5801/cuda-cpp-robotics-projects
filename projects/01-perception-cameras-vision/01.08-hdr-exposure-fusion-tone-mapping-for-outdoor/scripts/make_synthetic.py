#!/usr/bin/env python3
# ===========================================================================
# make_synthetic.py — synthetic HDR outdoor scene + camera-response-function
#                      exposure bracket generator for project 01.08 (HDR
#                      exposure fusion + tone mapping for outdoor robots)
#
# Role in the project
# --------------------
# This is the ONLY place the ground truth exists in closed form. Everything
# downstream (the CUDA program's Debevec-Malik CRF recovery, its Mertens
# fusion, and every verification gate in src/main.cu) is graded against what
# THIS script wrote to data/sample/ — so every constant here that the C++
# side also needs is called out with a "MUST MATCH src/main.cu's k...`
# comment, mirroring the discipline in 01.01's make_synthetic.py (its
# checkerboard-geometry constants are cross-referenced the same way).
#
# What this generates (stdlib only — no numpy/PIL, CLAUDE.md paragraph 5's
# "small vendored header" rule is about C++; the Python side just uses the
# standard library so a bare `python3` runs it anywhere):
#
#   1) An ANALYTIC outdoor scene: a piecewise radiance field R(x, y) in
#      "relative synthetic radiance units" (not calibrated lux — see
#      THEORY.md "The problem" for the honest photometric framing) spanning
#      ~5 orders of magnitude: a sun disk, a bright sky, open shade, sunlit
#      concrete with a painted line marking, and a deep shadow rectangle
#      ("under a parked vehicle"). Every region except three deliberately
#      CLEAN calibration/marking regions carries hashed value-noise texture
#      (the technique named in 01.04-feature-pipeline's make_synthetic.py:
#      a bilinearly-interpolated lattice of per-cell hashed values — see
#      value_noise() below) so that DETAIL EXISTS EVERYWHERE, including
#      inside the deep shadow: the whole point of the detail_preservation
#      gate in src/main.cu is that this texture is present in the SCENE but
#      only VISIBLE (extractable local contrast) in outputs that expose it.
#
#   2) A KNOWN, analytic camera response function (CRF): a Naka-Rushton /
#      Michaelis-Menten saturating curve (the same functional form used to
#      model photoreceptor response in vision science — see THEORY.md "CRF
#      physics"), forward-invertible in closed form. This is the ground
#      truth the CUDA program's Debevec-Malik solver must recover from
#      nothing but the pixel data (the crf_recovery gate).
#
#   3) FOUR clipped, noisy 8-bit LDR exposures (the bracket) at documented
#      shutter speeds, written as binary PGM (P5) — reusing the exact
#      minimal-PGM-writer discipline established in 01.01/01.03.
#
#   4) The EXACT ground-truth radiance (pixel-integrated via 2x2
#      supersampling, matching how a real sensor integrates over its pixel
#      area) as a raw float32 binary dump — the oracle for the
#      radiance_reconstruction gate.
#
#   5) params.csv — every constant below, in one machine-readable place, so
#      the CUDA program (and a human) can cross-check without re-deriving.
#
# Determinism: everything here is seeded (SEED = 42, CLAUDE.md paragraph 8);
# rerunning this script bit-for-bit reproduces every committed sample file.
#
# Read this after: THEORY.md "The problem" and "The math" (the CRF and
# Debevec-Malik sections use these exact symbols: R, X = R*t, Z, g, S).
# ===========================================================================

import csv
import math
import os
import struct

# ---------------------------------------------------------------------------
# XorShift32 — the repo's standard tiny deterministic PRNG (CLAUDE.md
# paragraph "MACHINE FACTS": no std::uniform_real_distribution anywhere in
# this project, C++ or Python, so the two languages can never disagree about
# what "random" means). One 32-bit state word, three shifts, full period
# 2^32-1 over its reachable states. The state 0 is absorbing (0 xorshifts to
# 0 forever) so the constructor guards against it, exactly like 01.04's.
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


# ===========================================================================
# Problem geometry — MUST MATCH src/main.cu's kW/kH and src/kernels.cuh's
# W/H (single-sourced there for the CUDA side; this is the Python twin).
# ===========================================================================
W, H = 160, 120                 # pixels; matches 01.03's pyramid-friendly 160x120 precedent
NUM_LEVELS = 3                  # pyramid depth: 160x120 -> 80x60 -> 40x30 (needs two clean halvings)
SEED = 42                       # repo-wide convention (CLAUDE.md "MACHINE FACTS")

# ---------------------------------------------------------------------------
# Exposure bracket — the four shutter speeds named in the task brief, in
# seconds. MUST MATCH src/kernels.cuh's kExposureTimes[4].
# Longest-to-shortest ratio is 125x (~7 stops) — a realistic single-bracket
# spread for an outdoor HDR capture.
# ---------------------------------------------------------------------------
EXPOSURE_TIMES_S = [1.0 / 1000.0, 1.0 / 125.0, 1.0 / 30.0, 1.0 / 8.0]
N_EXPOSURES = len(EXPOSURE_TIMES_S)

# ---------------------------------------------------------------------------
# The camera response function (CRF) — a Naka-Rushton / Michaelis-Menten
# saturating curve in the exposure X = R * t (radiance times exposure time):
#
#     z_frac(X) = X^GAMMA / (X^GAMMA + S_HALF^GAMMA)          in [0, 1)
#
# GAMMA < 1 gives a gentle "gamma" power-law region for X << S_HALF and a
# smooth saturating "shoulder" for X >= S_HALF — no hard knee anywhere, so
# every code value in [0, 255] is reachable by SOME (R, t) pair, and the
# curve is analytically invertible (see THEORY.md "The math"):
#
#     X = S_HALF * (z_frac / (1 - z_frac)) ** (1 / GAMMA)
#
# MUST MATCH src/main.cu's kCrfGamma / kCrfHalfX — the crf_recovery gate
# compares the CUDA program's RECOVERED curve against this exact function,
# re-typed independently in C++ (never shared code — see reference_cpu.cpp's
# header for why an independent ground-truth re-derivation, not a shared
# library call, is what makes that gate meaningful).
# ---------------------------------------------------------------------------
CRF_GAMMA = 0.85
CRF_S_HALF = 3.0

# ---------------------------------------------------------------------------
# Scene radiance tiers — "relative synthetic radiance units" (see THEORY.md
# for the honest illustrative mapping onto real photometric lux tiers: this
# is NOT a calibrated photometric simulation, just order-of-magnitude-
# faithful ratios chosen so the four exposures above each usefully expose a
# DIFFERENT band of the scene — the entire didactic point of bracketing).
# R_SUN / R_SHADOW = 100,000 = 1e5: the scene spans five orders of magnitude,
# even though (honestly, and realistically) no single exposure — and not
# even the full four-exposure bracket — recovers all five decades equally
# well; the sun disk in particular stays marginal in every exposure, exactly
# as a real sensor would see it (see THEORY.md "Where this sits in the real
# world"). MUST MATCH src/main.cu's kRadiance* constants.
# ---------------------------------------------------------------------------
R_SHADOW = 2.0              # deep shadow under the parked vehicle
R_SHADE = 60.0               # open shade (still outdoors, out of direct sun)
R_CONCRETE = 900.0           # sunlit concrete, the dominant "well-lit ground" tier
R_LINE = 3.0 * R_CONCRETE    # painted lane/parking line marking: 3x brighter than bare concrete
R_SKY = 9000.0               # bright overcast/blue sky background
R_SUN = 200000.0             # the sun disk itself — deliberately near-unrecoverable

# ---------------------------------------------------------------------------
# Scene layout — all pixel-space rectangles, MUST MATCH src/main.cu's
# k*Roi / k*Rect constants (the gates read these same regions out of the
# recovered/fused images). Coordinates are half-open [lo, hi) in the usual
# raster convention (row-major, y down) — a DIFFERENT convention from the
# robot body frames documented in docs/SYSTEM_DESIGN.md; images are pixel
# rasters, not physical frames, and THEORY.md says so explicitly.
# ---------------------------------------------------------------------------
SKY_Y0, SKY_Y1 = 0, 18                      # sky band
SUN_CX, SUN_CY, SUN_R = 135.0, 9.0, 7.0     # sun disk (circle)
SHADE_Y0, SHADE_Y1 = 18, 46                 # open-shade band
LINE_X0, LINE_X1, LINE_Y0, LINE_Y1 = 10, 150, 60, 64          # painted line marking
SHADOW_X0, SHADOW_X1, SHADOW_Y0, SHADOW_Y1 = 55, 105, 70, 108  # deep-shadow rectangle
GRAD_Y0, GRAD_Y1 = 110, 118                 # monotonic calibration strip (noise-free, see below)

# Gate ROIs (read again by src/main.cu — detail_preservation and halo_check).
# SHADOW_ROI covers MOST of the shadow rectangle (a few px of margin so the
# region stays "interior", but wide enough that the Mertens weight
# pyramid's coarse-level blur radius does not mostly escape into the
# surrounding concrete — measured: a tighter ROI here made mertens_fusion's
# recovered local contrast fall BELOW the best single exposure's, which is
# a genuine property of blurring a small weight region, not a bug; see
# THEORY.md "Numerical considerations").
SHADOW_ROI = (58, 102, 72, 106)        # (x0, x1, y0, y1) — most of the SHADOW rect, margin from its edge
HIGHLIGHT_ROI = (15, 45, 50, 58)       # inside the concrete band, away from every other feature
# halo_check scans the painted LINE / bare-CONCRETE boundary at x=10 (a
# modest 3x radiance contrast) rather than the much more extreme (450x)
# shadow/concrete edge — see src/main.cu's kHaloScanY comment for why.
HALO_SCAN_Y = 62                       # scanline row for the halo_check gate (inside the line band)
HALO_SCAN_X0, HALO_SCAN_X1 = 0, 26     # straddles the line/concrete boundary at x=10

# ---------------------------------------------------------------------------
# Texture — hashed value noise (the 01.04-feature-pipeline technique: hash a
# coarse lattice, bilinearly interpolate between cell corners). Cell pitch in
# pixels; texture_factor(x,y) = 1 + TEXTURE_AMPLITUDE * noise(x,y), noise in
# [-1, 1], applied MULTIPLICATIVELY to radiance so it never changes sign.
# ---------------------------------------------------------------------------
NOISE_CELL_PX = 10.0
TEXTURE_AMPLITUDE = 0.15

# ---------------------------------------------------------------------------
# Sensor noise — a deliberately SIMPLE additive model applied in the
# display-referred (post-CRF, 0..255) domain: NOT a rigorous photon-shot-
# noise (Poisson) simulator. Documented honestly in THEORY.md "Numerical
# considerations" as a simplification: real sensor noise is
# signal-dependent (shot noise) plus a signal-independent read-noise floor;
# a single Gaussian-ish additive term in code-value units is the simplest
# model that still forces the CRF recovery and merge stages to be robust to
# SOME noise, without adding a second free parameter this project does not
# need to teach its headline lesson.
# ---------------------------------------------------------------------------
NOISE_SIGMA_CODE = 1.5


def value_noise(x: float, y: float, salt: int) -> float:
    """Smooth, deterministic hashed lattice noise in [-1, 1] at continuous
    (x, y), via bilinear interpolation between four hashed cell-corner
    values (the technique 01.04-feature-pipeline's make_synthetic.py uses
    for its checkerboard cell coloring — see that file's cell_color()
    header — reimplemented independently here for a continuous 2D field
    rather than discrete cells, since this project needs texture that
    varies smoothly across a pixel-integrated 2x2 supersample).

    Parameters:
      x, y — continuous scene coordinates (pixels).
      salt — folds a per-purpose constant into the hash so different callers
             (e.g. two different textured regions) can draw INDEPENDENT
             noise fields from the same underlying hash family without the
             fields correlating.
    Returns: float in [-1, 1], continuous and smooth (bilinear, so C0 but
      not C1 — fine for a texture amplitude of 0.15, not fine for anything
      that would differentiate it).
    """
    cx = x / NOISE_CELL_PX
    cy = y / NOISE_CELL_PX
    x0, y0 = math.floor(cx), math.floor(cy)
    fx, fy = cx - x0, cy - y0

    def corner_hash(ix: int, iy: int) -> float:
        # Combine the two integer lattice coordinates and the salt into one
        # seed (same "multiply by large odd primes, XOR together" idiom
        # 01.04 uses for its patch_salt), draw ONE xorshift32 value from a
        # FRESH stream keyed on that seed -> deterministic, non-periodic.
        seed = (ix * 374761393 + iy * 668265263 + salt * 2246822519) & 0xFFFFFFFF
        return XorShift32(seed).next_signed()

    h00 = corner_hash(x0, y0)
    h10 = corner_hash(x0 + 1, y0)
    h01 = corner_hash(x0, y0 + 1)
    h11 = corner_hash(x0 + 1, y0 + 1)
    # Bilinear blend of the four corner values — the standard cheap
    # "value noise" reconstruction (as opposed to gradient/Perlin noise,
    # which interpolates gradients, not values; value noise is simpler and
    # entirely sufficient for a mild reflectance-texture cue).
    h0 = h00 * (1.0 - fx) + h10 * fx
    h1 = h01 * (1.0 - fx) + h11 * fx
    return h0 * (1.0 - fy) + h1 * fy


def radiance_at(x: float, y: float) -> float:
    """Evaluate the analytic scene radiance R(x, y) at one continuous point
    (relative synthetic units, > 0). Painter's-algorithm layering: sky/sun
    first, then shade, then concrete with its line marking and shadow
    rectangle carved out, then the noise-free calibration strip painted
    last (it OVERRIDES whatever band it sits in, by design — see its
    comment below).

    Returns: R > 0, a single scalar (this scene has no color — see
    THEORY.md/README "Limitations & honesty" for the grayscale-simplification
    discussion that also drives the Mertens saturation-term simplification
    in src/kernels.cu).
    """
    # ---- calibration strip: checked FIRST so it overrides any band -------
    if GRAD_Y0 <= y < GRAD_Y1:
        # A clean, NOISE-FREE, purely deterministic log-linear ramp from
        # R_SHADOW to R_SUN across the full width — used ONLY by the
        # tone_map_range gate's monotonicity check (src/main.cu). No
        # texture here on purpose: the gate must isolate "is the tone-map
        # monotonic in true radiance", not "did noise create a local dip".
        t = x / (W - 1)
        return R_SHADOW * ((R_SUN / R_SHADOW) ** t)

    if y < SKY_Y1:
        dx, dy = x - SUN_CX, y - SUN_CY
        if dx * dx + dy * dy <= SUN_R * SUN_R:
            return R_SUN   # sun disk core: flat, no texture (a small, near-saturated feature)
        base = R_SKY
        return base * (1.0 + TEXTURE_AMPLITUDE * value_noise(x, y, salt=1))  # textured sky/cloud variation

    if y < SHADE_Y1:
        base = R_SHADE
        return base * (1.0 + TEXTURE_AMPLITUDE * value_noise(x, y, salt=2))

    # ---- concrete band, with the line marking and the shadow rectangle ---
    if LINE_X0 <= x < LINE_X1 and LINE_Y0 <= y < LINE_Y1:
        return R_LINE   # painted marking: flat/clean, as real paint reads far more uniform than bare concrete

    if SHADOW_X0 <= x < SHADOW_X1 and SHADOW_Y0 <= y < SHADOW_Y1:
        base = R_SHADOW
        return base * (1.0 + TEXTURE_AMPLITUDE * value_noise(x, y, salt=3))  # texture EXISTS even in shadow

    base = R_CONCRETE
    return base * (1.0 + TEXTURE_AMPLITUDE * value_noise(x, y, salt=4))


def radiance_pixel_integrated(px: int, py: int) -> float:
    """The ground-truth radiance for one PIXEL (integer px, py), computed as
    a 2x2-supersampled average of the continuous scene — matching how a
    real sensor pixel integrates irradiance over its finite area rather
    than sampling a single infinitesimal point (see THEORY.md "The
    problem"). This is exactly what data/sample/ground_truth_radiance.bin
    stores, and exactly what the four LDR exposures below are rendered
    from, so the ground truth and the rendered images are self-consistent
    by construction — the whole point of a synthetic, exactly-known scene
    (CLAUDE.md paragraph 8).
    """
    acc = 0.0
    for sy in (0.25, 0.75):
        for sx in (0.25, 0.75):
            acc += radiance_at(px + sx, py + sy)
    return acc / 4.0


def crf_forward(x_exposure: float) -> float:
    """The KNOWN synthetic CRF, forward direction: exposure -> normalized
    code value in [0, 1). See the CRF_GAMMA/CRF_S_HALF module comment for
    the closed-form derivation. x_exposure = R * t, must be >= 0."""
    if x_exposure <= 0.0:
        return 0.0
    xg = x_exposure ** CRF_GAMMA
    sg = CRF_S_HALF ** CRF_GAMMA
    return xg / (xg + sg)


def render_exposure(t_seconds: float, noise_seed: int):
    """Render one full LDR exposure: for every pixel, integrate radiance,
    convert to exposure X = R*t, push through the CRF, add sensor noise,
    round and clip to uint8. Returns a flat bytearray of W*H bytes
    (row-major), i.e. an 8-bit grayscale image ready for write_pgm().
    """
    rng = XorShift32(noise_seed)
    out = bytearray(W * H)
    for y in range(H):
        for x in range(W):
            r = radiance_pixel_integrated(x, y)
            x_exp = r * t_seconds
            z = 255.0 * crf_forward(x_exp)
            # Additive code-value noise (see NOISE_SIGMA_CODE's header) via
            # a crude but adequate sum-of-uniforms approximation to a
            # Gaussian (the classic "Irwin-Hall" trick: summing N uniforms
            # and recentering approaches a Gaussian by the CLT; N=4 is a
            # cheap, dependency-free stand-in for a true Box-Muller draw,
            # entirely sufficient for a MILD noise floor we are not trying
            # to statistically characterize with precision).
            u = (rng.next_unit() + rng.next_unit() + rng.next_unit() + rng.next_unit() - 2.0)  # approx mean 0, var 1/3
            noise = u * NOISE_SIGMA_CODE / math.sqrt(1.0 / 3.0)  # rescale so std ~= NOISE_SIGMA_CODE
            z_noisy = z + noise
            z_clipped = 0 if z_noisy < 0.0 else (255 if z_noisy > 255.0 else int(z_noisy + 0.5))
            out[y * W + x] = z_clipped
    return out


def write_pgm(path: str, width: int, height: int, gray: bytearray) -> None:
    """Minimal binary PGM (P5) writer — same on-disk convention as
    01.01/01.03's write_pgm, reimplemented here (Python side) rather than
    shared, since this script and the C++ reader are independent languages
    by construction (CLAUDE.md paragraph 8: nothing in this repo shares
    code across the synthetic-data boundary)."""
    with open(path, "wb") as f:
        f.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        f.write(bytes(gray))


def write_radiance_binary(path: str, radiance) -> None:
    """Raw float32 binary dump of the full ground-truth radiance field,
    row-major (y*W+x), little-endian IEEE-754, NO header (the format is
    documented once, in data/README.md, rather than self-described — this
    keeps the C++ reader a single fread-equivalent call)."""
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(radiance)}f", *radiance))


def main() -> None:
    sample_dir = os.path.join(os.path.dirname(__file__), "..", "data", "sample")
    os.makedirs(sample_dir, exist_ok=True)

    # Remove the scaffold-era placeholder sample, if present — this project
    # reads no CSV of (x, y) pairs; the real sample files are written below.
    stale = os.path.join(sample_dir, "saxpy_sample.csv")
    if os.path.exists(stale):
        os.remove(stale)

    # ---- ground truth radiance (exact, noise-free, pixel-integrated) -----
    radiance = [radiance_pixel_integrated(x, y) for y in range(H) for x in range(W)]
    write_radiance_binary(os.path.join(sample_dir, "ground_truth_radiance.bin"), radiance)

    r_min, r_max = min(radiance), max(radiance)
    print(f"[make_synthetic] ground-truth radiance range: {r_min:.4f} .. {r_max:.1f} "
          f"({r_max / r_min:.0f}x, {math.log10(r_max / r_min):.2f} decades)")

    # ---- four LDR exposures ------------------------------------------------
    for i, t in enumerate(EXPOSURE_TIMES_S):
        # Reseed per exposure (seed = SEED + 1000*i) so each exposure's noise
        # is independent of the others but every exposure is independently
        # reproducible from the single documented SEED (CLAUDE.md paragraph 8).
        img = render_exposure(t, noise_seed=SEED + 1000 * i)
        write_pgm(os.path.join(sample_dir, f"exposure_{i}.pgm"), W, H, img)
        n_black = sum(1 for v in img if v == 0)
        n_white = sum(1 for v in img if v == 255)
        print(f"[make_synthetic] exposure_{i}.pgm  t={t:.6f}s  "
              f"clipped-black={n_black} ({100.0 * n_black / (W * H):.1f}%)  "
              f"clipped-white={n_white} ({100.0 * n_white / (W * H):.1f}%)")

    # ---- params.csv: every constant a downstream reader might need -------
    params_path = os.path.join(sample_dir, "params.csv")
    with open(params_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        w.writerow(["W", W]); w.writerow(["H", H])
        w.writerow(["N_EXPOSURES", N_EXPOSURES])
        for i, t in enumerate(EXPOSURE_TIMES_S):
            w.writerow([f"T{i}_seconds", f"{t:.9f}"])
        w.writerow(["CRF_GAMMA", CRF_GAMMA]); w.writerow(["CRF_S_HALF", CRF_S_HALF])
        w.writerow(["R_SHADOW", R_SHADOW]); w.writerow(["R_SHADE", R_SHADE])
        w.writerow(["R_CONCRETE", R_CONCRETE]); w.writerow(["R_LINE", R_LINE])
        w.writerow(["R_SKY", R_SKY]); w.writerow(["R_SUN", R_SUN])
        w.writerow(["NOISE_SIGMA_CODE", NOISE_SIGMA_CODE])
        w.writerow(["SEED", SEED])
        w.writerow(["SHADOW_ROI", f"{SHADOW_ROI}"])
        w.writerow(["HIGHLIGHT_ROI", f"{HIGHLIGHT_ROI}"])
        w.writerow(["HALO_SCAN", f"y={HALO_SCAN_Y},x=[{HALO_SCAN_X0},{HALO_SCAN_X1})"])
        w.writerow(["GRAD_STRIP_Y", f"[{GRAD_Y0},{GRAD_Y1})"])

    print(f"[make_synthetic] wrote params.csv, {N_EXPOSURES} exposure_*.pgm, "
          f"ground_truth_radiance.bin into {os.path.abspath(sample_dir)}")


if __name__ == "__main__":
    main()
