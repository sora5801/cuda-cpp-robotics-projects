#!/usr/bin/env python3
"""make_synthetic.py — synthetic sensor + scene generator for project 01.23
(Full RAW->RGB ISP: black level -> lens shading -> defect correction ->
white balance -> demosaic (MHC + bilinear) -> CCM -> gamma).

Why this script exists (CLAUDE.md section 8: synthetic-first)
-----------------------------------------------------------------
A real RAW sensor never comes with ground truth: you cannot ask a physical
camera "what SHOULD this raw frame have been, with zero shading, zero
defects, zero noise, and a known illuminant?" A synthetic sensor model can
answer that question EXACTLY, for every stage of the pipeline, because this
script IS the forward model kernels.cu/reference_cpu.cpp invert. Every
number the ISP needs to correct — the black level, the lens-shading
polynomial, the spectral crosstalk matrix, the illuminant gains — is defined
ONCE, here, and cross-referenced ("MUST MATCH ../src/kernels.cuh") in that
header so the two files can never silently disagree about what "correct"
means (the same discipline sibling flagship 01.01 uses for its camera model).

The generation direction (the physically honest one, and the exact inverse
of what the ISP undoes):

    scene reflectance (linear sRGB, authored directly below)
      -> illuminant gain (per-channel signal scaling: D65 is neutral,
         tungsten is red-heavy/blue-poor)
      -> spectral crosstalk matrix M (each color filter leaks a little
         signal from its neighbors)
      -> per-pixel Bayer sampling (mosaic: keep only the native channel)
      -> lens shading (radial multiplicative light falloff)
      -> black level (additive sensor offset)
      -> shot + read noise (signal-dependent Gaussian)
      -> quantize to a 10-bit RAW10-in-uint16 code
      -> inject the committed defect list (stuck pixels, applied LAST —
         physically, a broken photosite reads a fixed pattern regardless of
         what light actually arrived)

kernels.cu's ISP undoes stages, in reverse order, using the SAME documented
constants: black level -> shading -> defects -> white balance -> demosaic ->
CCM -> gamma. Because generator and pipeline share every numeric constant
(cross-checked, never imported — this script reimplements the physics
independently in Python, the same "three independent parties agree"
discipline 01.01 uses for its camera model), the pipeline's output should
closely reproduce the scene's own appearance; main.cu's gates measure
exactly how closely.

RNG discipline (CLAUDE.md machine facts: no numpy, xorshift32, seed 42)
-----------------------------------------------------------------------
ONE xorshift32 generator, seeded 42, is used for the WHOLE script, drawn in
this FIXED order (reproducing the committed sample bit-for-bit requires this
exact sequence):
  1. hashed-texture block-color selection (raster order over the texture region)
  2. defect-pixel candidate placement (rejection sampling until 16 accepted)
  3. D65 scene noise (raster order, 2 uniform draws per pixel via Box-Muller)
  4. tungsten scene noise (same, continuing the same stream)
Gaussian noise is drawn via the Box-Muller transform (two uniforms -> one
normal deviate) — the standard stdlib-only way to get Gaussian noise without
numpy/random.gauss (CLAUDE.md machine facts: stdlib only).

What this writes (into ../data/sample/, all tiny — see data/README.md for
exact byte counts and SHA-256 checksums):

    raw_mosaic_d65.bin        kRawW*kRawH uint16 (little-endian), RGGB, the
                               D65-illuminated RAW10-in-uint16 sensor frame —
                               one of the two ISP inputs.
    raw_mosaic_tungsten.bin   same, under the tungsten illuminant.
    true_sensor_rgb_d65.bin   kRawW*kRawH*3 float32 (little-endian): the
                               NOISELESS, PRE-shading sensor-domain RGB under
                               D65 (illuminant+crosstalk only) — ground truth
                               for the demosaic-quality gate (isolated from
                               AWB error: under D65, white balance is a
                               near-no-op by construction, see kernels.cuh's
                               kTrueAwbGainD65).
    true_scene_srgb.ppm       kRawW*kRawH*3 uint8 (PPM P6): gamma_encode of
                               the ORIGINAL scene reflectance — the "what a
                               perfect ISP would render" ground truth for the
                               end-to-end PSNR gate, illuminant-independent.
    defect_list.csv           x,y,kind — the committed factory defect map
                               (loaded at RUNTIME by main.cu, unlike every
                               other constant here, which is compiled into
                               kernels.cuh — see that file's section 4).
    params.txt                human-readable dump of every generation
                               constant, for a reader auditing data/sample/
                               without opening kernels.cuh side by side.

Usage:
    python make_synthetic.py               # regenerates the exact committed sample
"""

import math
import struct
import sys
from pathlib import Path

# ===========================================================================
# MUST MATCH ../src/kernels.cuh — every constant in this block is this
# script's half of the "generator and pipeline agree" contract. A change to
# either side without the other is a project-breaking bug, not a style slip.
# ===========================================================================
RAW_W, RAW_H = 160, 120
BLACK_LEVEL, WHITE_LEVEL = 64, 1023
SAT_RANGE = WHITE_LEVEL - BLACK_LEVEL   # 959

CHART_COLS, CHART_ROWS = 6, 4
PATCH_SIZE, PATCH_GAP = 20, 1
CHART_X0, CHART_Y0 = 4, 4
CHART_W = CHART_COLS * PATCH_SIZE + (CHART_COLS - 1) * PATCH_GAP   # 125
CHART_H = CHART_ROWS * PATCH_SIZE + (CHART_ROWS - 1) * PATCH_GAP   # 83

CARD_X0, CARD_Y0, CARD_W, CARD_H = 4, 91, 86, 20
CARD_SRGB8 = (230, 230, 230)

TEX_X0, TEX_Y0, TEX_W, TEX_H = 133, 4, 23, 107
TEX_BLOCK = 4
TEX_PALETTE = [
    (220, 50, 50), (50, 220, 50), (50, 50, 220), (220, 220, 50),
    (220, 50, 220), (50, 220, 220), (230, 230, 230), (30, 30, 30),
]

BACKGROUND_SRGB8 = (30, 30, 30)

# 24-patch chart, row-major (row 0 = skin/earth, row 1 = warm [cols 0-2 =
# the red-heavy AWB-failure crop], row 2 = cool/primary/secondary, row 3 =
# the grayscale ramp). See kernels.cuh section 1 for the full documentation
# of why these are ILLUSTRATIVE values, not certified colorimetric data.
CHART_REF_SRGB8 = [
    (115, 82, 68), (194, 150, 130), (98, 122, 157), (87, 108, 67), (133, 128, 177), (103, 189, 170),
    (214, 126, 44), (193, 60, 56), (222, 158, 46), (94, 60, 108), (157, 188, 64), (56, 61, 150),
    (56, 80, 152), (70, 148, 73), (60, 150, 175), (188, 84, 150), (231, 199, 31), (52, 126, 145),
    (243, 243, 242), (200, 200, 200), (160, 160, 160), (120, 120, 120), (85, 85, 85), (52, 52, 52),
]

SHADE_CX, SHADE_CY = (RAW_W - 1) * 0.5, (RAW_H - 1) * 0.5
SHADE_RNORM = math.hypot(SHADE_CX, SHADE_CY)   # exact double precision; kernels.cuh hand-rounds to float32
SHADE_A2, SHADE_A4 = -0.35, 0.10
SHADE_GAIN_FLOOR = 0.35

# Spectral crosstalk matrix M (row = filter R/G/B, col = R/G/B light it
# responds to). Rows sum to 1.0 -> a neutral input reproduces itself at the
# sensor (see kernels.cuh's kM00.. for the full physical reasoning).
M = [
    [0.72, 0.22, 0.06],
    [0.10, 0.78, 0.12],
    [0.06, 0.20, 0.74],
]

ILLUMINANT_D65 = (1.00, 1.00, 1.00)
ILLUMINANT_TUNGSTEN = (1.42, 1.00, 0.53)

READ_NOISE_DN = 2.0
SHOT_NOISE_K = 0.02

NUM_DEFECTS = 16
DEFECT_KINDS = ["stuck_high", "stuck_low", "stuck_mid"]
DEFECT_VALUES = {"stuck_high": WHITE_LEVEL, "stuck_low": 0, "stuck_mid": 512}

SEED = 42

# ---------------------------------------------------------------------------
# xorshift32 PRNG — stdlib-only, deterministic, the repo-wide substitute for
# std::uniform_real_distribution/numpy. 32-bit state, the textbook
# Marsaglia xorshift recurrence (13,17,5).
# ---------------------------------------------------------------------------
class Xorshift32:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF
        if self.state == 0:
            self.state = 0x9E3779B9   # xorshift's fixed point is 0; never seed with it

    def next_u32(self):
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17)
        x ^= (x << 5) & 0xFFFFFFFF
        x &= 0xFFFFFFFF
        self.state = x
        return x

    def uniform01(self):
        # 24 bits of resolution in [0, 1) — plenty for noise/texture/defect
        # sampling at this project's scale; avoids the low-order-bit
        # weakness xorshift32 is known to have if only the bottom few bits
        # were used directly.
        return (self.next_u32() >> 8) / float(1 << 24)

    def gauss(self):
        # Box-Muller: two independent uniforms -> one standard-normal
        # deviate. u1 is floored away from 0 so log() never sees 0.
        u1 = max(self.uniform01(), 1e-12)
        u2 = self.uniform01()
        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        return r * math.cos(theta)

    def randint(self, lo, hi_inclusive):
        span = hi_inclusive - lo + 1
        return lo + (self.next_u32() % span)


# ---------------------------------------------------------------------------
# sRGB transfer function — independent Python twin of kernels.cuh's
# srgb_encode/srgb_decode (three independent implementations of the SAME
# published formula: this script, kernels.cu's device code, reference_cpu.cpp's
# host code — agreement across all three is strong evidence the physics is
# right, not just one file's arithmetic).
# ---------------------------------------------------------------------------
def srgb_decode(s):
    s = min(max(s, 0.0), 1.0)
    if s <= 0.04045:
        return s / 12.92
    return ((s + 0.055) / 1.055) ** 2.4


def srgb_encode(lin):
    lin = min(max(lin, 0.0), 1.0)
    if lin <= 0.0031308:
        return 12.92 * lin
    return 1.055 * (lin ** (1.0 / 2.4)) - 0.055


def bayer_phase_at(x, y):
    """0=R, 1=Gr (green, red row), 2=Gb (green, blue row), 3=B — MUST MATCH
    kernels.cuh's bayer_phase_at()."""
    even_row = (y % 2) == 0
    even_col = (x % 2) == 0
    if even_row and even_col:
        return 0
    if even_row and not even_col:
        return 1
    if not even_row and even_col:
        return 2
    return 3


def phase_channel(phase):
    """Collapse a 4-way Bayer phase to the 3-way R/G/B channel index used by
    the crosstalk matrix M and the illuminant gain vectors."""
    if phase == 0:
        return 0
    if phase == 3:
        return 2
    return 1


def shading_gain_at(x, y):
    dx = x - SHADE_CX
    dy = y - SHADE_CY
    r = math.hypot(dx, dy) / SHADE_RNORM
    r2 = r * r
    return 1.0 + SHADE_A2 * r2 + SHADE_A4 * r2 * r2


def apply_matrix3(v, mat):
    return [
        mat[0][0] * v[0] + mat[0][1] * v[1] + mat[0][2] * v[2],
        mat[1][0] * v[0] + mat[1][1] * v[1] + mat[1][2] * v[2],
        mat[2][0] * v[0] + mat[2][1] * v[1] + mat[2][2] * v[2],
    ]


# ===========================================================================
# Scene authoring — build scene_linear[y][x] = [R,G,B] linear-sRGB
# reflectance, kRawW x kRawH, per the layout diagram in kernels.cuh section 1.
# ===========================================================================
def build_scene(rng):
    bg_lin = [srgb_decode(c / 255.0) for c in BACKGROUND_SRGB8]
    scene = [[list(bg_lin) for _ in range(RAW_W)] for _ in range(RAW_H)]

    # 24-patch chart.
    for r in range(CHART_ROWS):
        for c in range(CHART_COLS):
            idx = r * CHART_COLS + c
            lin = [srgb_decode(v / 255.0) for v in CHART_REF_SRGB8[idx]]
            x0 = CHART_X0 + c * (PATCH_SIZE + PATCH_GAP)
            y0 = CHART_Y0 + r * (PATCH_SIZE + PATCH_GAP)
            for yy in range(y0, y0 + PATCH_SIZE):
                for xx in range(x0, x0 + PATCH_SIZE):
                    scene[yy][xx] = list(lin)

    # AWB reference card.
    card_lin = [srgb_decode(c / 255.0) for c in CARD_SRGB8]
    for yy in range(CARD_Y0, CARD_Y0 + CARD_H):
        for xx in range(CARD_X0, CARD_X0 + CARD_W):
            scene[yy][xx] = list(card_lin)

    # Hashed texture: TEX_BLOCK x TEX_BLOCK blocks, each one palette color
    # chosen by the shared RNG stream (draw order: raster over blocks).
    palette_lin = [[srgb_decode(v / 255.0) for v in col] for col in TEX_PALETTE]
    for by in range(TEX_Y0, TEX_Y0 + TEX_H, TEX_BLOCK):
        for bx in range(TEX_X0, TEX_X0 + TEX_W, TEX_BLOCK):
            pal_idx = rng.randint(0, len(TEX_PALETTE) - 1)
            col = palette_lin[pal_idx]
            for yy in range(by, min(by + TEX_BLOCK, TEX_Y0 + TEX_H)):
                for xx in range(bx, min(bx + TEX_BLOCK, TEX_X0 + TEX_W)):
                    scene[yy][xx] = list(col)

    return scene


def blur_scene(scene):
    """Apply a mild separable [1,2,1]/4 optical blur to the authored scene,
    TWICE (equivalent to a small Gaussian), before it ever reaches the
    crosstalk/mosaic/shading model.

    Why this exists (a real physical effect, not a numerical convenience):
    every real lens has a point-spread function, and every real sensor sits
    behind an optical low-pass filter (OLPF) specifically so a Bayer sensor
    never sees a genuine one-pixel step edge — TRUE step edges alias badly
    into the mosaic and make ANY demosaicer (bilinear or MHC) ring badly at
    the boundary (THEORY.md derives why: a gradient-corrected demosaicer's
    "correction" term amplifies exactly the high-frequency content a step
    edge is made of). This project's authored scene (a color chart against a
    flat background, a hashed texture of solid-color blocks) is UNUSUALLY
    edge-rich for a synthetic test image — deliberately, so demosaic quality
    is measurable — but without any blur those edges are harsher than any
    real lens would ever deliver, and the resulting ringing artifacts would
    swamp the end-to-end PSNR gates with a demosaic-only effect that has
    nothing to do with the ISP stages actually under test (black level,
    shading, defects, AWB, CCM). A mild, explicitly documented blur restores
    the "a real sensor never sees a literal step edge" assumption every
    demosaic algorithm (including production ones) is built on.
    """
    def blur_pass(src):
        # Horizontal pass: out[y][x] = 0.25*src[x-1] + 0.5*src[x] + 0.25*src[x+1], edge-clamped.
        tmp = [[None] * RAW_W for _ in range(RAW_H)]
        for y in range(RAW_H):
            for x in range(RAW_W):
                xm = max(x - 1, 0)
                xp = min(x + 1, RAW_W - 1)
                a, c_, b = src[y][xm], src[y][x], src[y][xp]
                tmp[y][x] = [0.25 * a[k] + 0.5 * c_[k] + 0.25 * b[k] for k in range(3)]
        # Vertical pass: same [1,2,1]/4 kernel down columns.
        out = [[None] * RAW_W for _ in range(RAW_H)]
        for y in range(RAW_H):
            ym = max(y - 1, 0)
            yp = min(y + 1, RAW_H - 1)
            for x in range(RAW_W):
                a, c_, b = tmp[ym][x], tmp[y][x], tmp[yp][x]
                out[y][x] = [0.25 * a[k] + 0.5 * c_[k] + 0.25 * b[k] for k in range(3)]
        return out

    blurred = blur_pass(scene)   # one pass ~ sigma ~0.6px: enough to kill literal step edges
    return blurred                # while leaving real per-pixel detail for demosaic to resolve


# ===========================================================================
# Defect-pixel placement — 16 locations, confined to the flat BACKGROUND
# region (never inside the chart/card/texture, with a 3px safety margin, and
# never within 6px of the canvas border) so median-of-same-phase-neighbor
# recovery always has a locally uniform, exactly-known truth to converge to
# (main.cu's defect_recovery gate scores against that truth).
# ===========================================================================
def rect_hit_with_margin(x, y, rx0, ry0, rw, rh, margin):
    return (rx0 - margin <= x < rx0 + rw + margin) and (ry0 - margin <= y < ry0 + rh + margin)


def is_background_safe(x, y):
    if not (6 <= x < RAW_W - 6 and 6 <= y < RAW_H - 6):
        return False
    for (rx0, ry0, rw, rh) in [
        (CHART_X0, CHART_Y0, CHART_W, CHART_H),
        (CARD_X0, CARD_Y0, CARD_W, CARD_H),
        (TEX_X0, TEX_Y0, TEX_W, TEX_H),
    ]:
        if rect_hit_with_margin(x, y, rx0, ry0, rw, rh, 3):
            return False
    return True


def place_defects(rng):
    defects = []
    seen = set()
    while len(defects) < NUM_DEFECTS:
        x = rng.randint(6, RAW_W - 7)
        y = rng.randint(6, RAW_H - 7)
        if (x, y) in seen or not is_background_safe(x, y):
            continue
        seen.add((x, y))
        kind = DEFECT_KINDS[len(defects) % len(DEFECT_KINDS)]
        defects.append((x, y, kind))
    return defects


# ===========================================================================
# Forward sensor model — scene -> RAW mosaic, one illuminant at a time.
# ===========================================================================
def compute_true_sensor_rgb(scene, illum_gain):
    """Noiseless, pre-shading sensor-domain RGB (illuminant + crosstalk
    only) at every pixel — the demosaic-quality ground truth."""
    out = [[None] * RAW_W for _ in range(RAW_H)]
    for y in range(RAW_H):
        for x in range(RAW_W):
            refl = scene[y][x]
            lit = [refl[0] * illum_gain[0], refl[1] * illum_gain[1], refl[2] * illum_gain[2]]
            out[y][x] = apply_matrix3(lit, M)
    return out


def render_raw_mosaic(true_sensor_rgb, rng, defects):
    """true_sensor_rgb -> RAW10-in-uint16 mosaic: mosaic sampling, lens
    shading, black level, shot+read noise, quantization, then defect
    injection (defects overwrite whatever the physics would have produced —
    a broken photosite ignores the scene entirely)."""
    raw = [[0] * RAW_W for _ in range(RAW_H)]
    for y in range(RAW_H):
        for x in range(RAW_W):
            phase = bayer_phase_at(x, y)
            ch = phase_channel(phase)
            ideal = true_sensor_rgb[y][x][ch]
            shaded = ideal * shading_gain_at(x, y)
            shaded = min(max(shaded, 0.0), 1.0)
            raw_dn_ideal = BLACK_LEVEL + SAT_RANGE * shaded
            signal_above_black = max(raw_dn_ideal - BLACK_LEVEL, 0.0)
            sigma = math.sqrt(READ_NOISE_DN ** 2 + SHOT_NOISE_K * signal_above_black)
            noisy = raw_dn_ideal + sigma * rng.gauss()
            dn = int(round(noisy))
            dn = min(max(dn, 0), WHITE_LEVEL)
            raw[y][x] = dn

    for (dx, dy, kind) in defects:
        raw[dy][dx] = DEFECT_VALUES[kind]

    return raw


# ===========================================================================
# File writers.
# ===========================================================================
def write_raw_u16(path, raw):
    with open(path, "wb") as f:
        for y in range(RAW_H):
            f.write(struct.pack("<%dH" % RAW_W, *raw[y]))


def write_rgb_f32(path, rgb):
    with open(path, "wb") as f:
        for y in range(RAW_H):
            row = []
            for x in range(RAW_W):
                row.extend(rgb[y][x])
            f.write(struct.pack("<%df" % len(row), *row))


def write_scene_ppm(path, scene):
    with open(path, "wb") as f:
        f.write(b"P6\n%d %d\n255\n" % (RAW_W, RAW_H))
        buf = bytearray(RAW_W * RAW_H * 3)
        i = 0
        for y in range(RAW_H):
            for x in range(RAW_W):
                lin = scene[y][x]
                for c in range(3):
                    v = int(round(srgb_encode(lin[c]) * 255.0))
                    buf[i] = min(max(v, 0), 255)
                    i += 1
        f.write(bytes(buf))


def write_defects_csv(path, defects):
    with open(path, "w") as f:
        f.write("x,y,kind\n")
        for (x, y, kind) in defects:
            f.write("%d,%d,%s\n" % (x, y, kind))


def write_params_txt(path, defects):
    lines = [
        "# 01.23 synthetic sensor/scene parameters -- human-readable mirror of",
        "# ../src/kernels.cuh (source of truth) and this script's constants block.",
        "raw_w=%d raw_h=%d" % (RAW_W, RAW_H),
        "black_level=%d white_level=%d sat_range=%d" % (BLACK_LEVEL, WHITE_LEVEL, SAT_RANGE),
        "shading: cx=%.4f cy=%.4f rnorm=%.4f a2=%.3f a4=%.3f gain_floor=%.3f"
        % (SHADE_CX, SHADE_CY, SHADE_RNORM, SHADE_A2, SHADE_A4, SHADE_GAIN_FLOOR),
        "crosstalk M = %s" % M,
        "illuminant D65 gain = %s" % (ILLUMINANT_D65,),
        "illuminant tungsten gain = %s" % (ILLUMINANT_TUNGSTEN,),
        "noise: read_sigma_dn=%.2f shot_k=%.3f" % (READ_NOISE_DN, SHOT_NOISE_K),
        "num_defects=%d" % NUM_DEFECTS,
        "seed=%d (xorshift32, draw order: texture -> defects -> D65 noise -> tungsten noise)" % SEED,
        "",
        "defects:",
    ]
    for (x, y, kind) in defects:
        lines.append("  (%d,%d) %s" % (x, y, kind))
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    out_dir = Path(__file__).resolve().parent.parent / "data" / "sample"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = Xorshift32(SEED)

    scene = build_scene(rng)                 # draws: texture block colors
    scene = blur_scene(scene)                 # mild optical blur -- see blur_scene()'s docstring
    defects = place_defects(rng)              # draws: defect candidate rejection sampling

    true_sensor_rgb_d65 = compute_true_sensor_rgb(scene, ILLUMINANT_D65)
    true_sensor_rgb_tungsten = compute_true_sensor_rgb(scene, ILLUMINANT_TUNGSTEN)

    raw_d65 = render_raw_mosaic(true_sensor_rgb_d65, rng, defects)             # draws: D65 noise
    raw_tungsten = render_raw_mosaic(true_sensor_rgb_tungsten, rng, defects)   # draws: tungsten noise

    write_raw_u16(out_dir / "raw_mosaic_d65.bin", raw_d65)
    write_raw_u16(out_dir / "raw_mosaic_tungsten.bin", raw_tungsten)
    write_rgb_f32(out_dir / "true_sensor_rgb_d65.bin", true_sensor_rgb_d65)
    write_scene_ppm(out_dir / "true_scene_srgb.ppm", scene)
    write_defects_csv(out_dir / "defect_list.csv", defects)
    write_params_txt(out_dir / "params.txt", defects)

    print("[make_synthetic] wrote %dx%d RAW10-in-uint16 mosaics (D65, tungsten), "
          "true_sensor_rgb_d65.bin, true_scene_srgb.ppm, defect_list.csv (%d defects), "
          "params.txt -> %s" % (RAW_W, RAW_H, NUM_DEFECTS, out_dir))


if __name__ == "__main__":
    sys.exit(main())
