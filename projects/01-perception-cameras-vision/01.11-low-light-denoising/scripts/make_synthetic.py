#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample-data generator for 01.11 (Low-light
denoising: bilateral, non-local means, BM3D-lite).

Writes TWO 200x150 8-bit PGM frames into ../data/sample/:
    clean.pgm  — the noise-free ground-truth scene (never seen by any
                 denoiser; used ONLY by main.cu's gates, exactly the way
                 01.09's *_true.bin ground-truth dumps are used).
    noisy.pgm  — the SAME scene through this project's low-light sensor
                 model: exact Poisson shot noise + additive Gaussian read
                 noise + 8-bit quantization. This is the ONLY input every
                 denoiser (bilateral/NLM/BM3D-lite/Gaussian baseline)
                 actually sees.

Why exact Poisson, not the Gaussian shot-noise approximation (task brief:
"pick, justify" — see also kernels.cuh Section 2's header)
--------------------------------------------------------------------------
This project's deliberately extreme operating point (peak signal
kPeakElectrons=40 electrons; the darkest committed flat patch sits at only
~4.4 expected electrons) is exactly where a Gaussian approximation to
Poisson noise breaks down most visibly: Poisson is DISCRETE, RIGHT-SKEWED,
and can never go negative, while a Gaussian with matching variance can (and,
at a mean of 4.4, regularly would). Since a single offline Python script
pays this cost ONCE (not per frame, per tick, the way a real-time GPU noise
generator would), we sample EXACTLY via Knuth's classic multiplicative
inversion algorithm — O(lambda) uniform draws per pixel, trivial at
lambda <= 40 (poisson_knuth() below), using this repo's standard XorShift32
stream so nothing here depends on a library RNG (CLAUDE.md "MACHINE FACTS":
no std::uniform_real_distribution anywhere, C++ or Python).

Geometry, noise-model, and scene-layout constants below MUST MATCH
../src/kernels.cuh's Sections 1-3 EXACTLY — that header is annotated with
the identical values and the reasoning behind each one; this file re-states
them independently (never imports the C++ header) per the repo's synthetic-
data discipline (01.09's make_synthetic.py is the precedent this file
follows closely: same XorShift32 class, same hashed_unit/value_noise
bilinear-lattice-noise technique for the scene's background texture).

Read this after: kernels.cuh Sections 1-3, THEORY.md "The problem"/"The math".
"""

import csv
import math
import os
import struct

# ---------------------------------------------------------------------------
# XorShift32 — the repo's standard tiny deterministic PRNG (CLAUDE.md
# "MACHINE FACTS": no std::uniform_real_distribution anywhere, so C++ and
# Python can never disagree about what "random" means). Copied from the
# 01.09/01.04/01.08 precedent verbatim (one 32-bit state word, three shifts,
# full period over its reachable states; state 0 is absorbing so the
# constructor guards against it).
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
        """Uniform float in [-1, 1)."""
        return self.next_unit() * 2.0 - 1.0

    def next_gaussian(self) -> float:
        """One N(0,1) draw via Box-Muller (the 08.01 MPPI precedent)."""
        u1 = self.next_unit()
        if u1 < 1e-12:
            u1 = 1e-12   # guard log(0) — astronomically unlikely, defensive only
        u2 = self.next_unit()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def poisson_knuth(rng: XorShift32, lam: float) -> int:
    """Exact Poisson(lam) via Knuth's multiplicative-inversion algorithm (see
    this file's module docstring for why EXACT sampling was chosen over the
    Gaussian shot-noise approximation at this project's low-light operating
    point). Draws uniform variates from `rng` and multiplies them together
    until the running product drops below exp(-lam); the number of draws
    taken, minus one, is the Poisson sample. Cost: O(lam) uniform draws —
    trivial for this project's lam <= kPeakElectrons = 40.
    """
    if lam <= 0.0:
        return 0
    big_l = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.next_unit()
        if p <= big_l:
            return k - 1


# ===========================================================================
# SECTION 1/2 — geometry + noise model. MUST MATCH src/kernels.cuh Sections
# 1-2 (kW/kH/kN, kPeakElectrons/kReadNoiseE/kDnPerElectron) EXACTLY. See
# kernels.cuh's kW comment for why the canvas is 200x150, not 01.09's
# 160x120 precedent: this project's methods have real spatial reach
# (BM3D-lite's block-match search extends ~14 px from a reference anchor),
# and 160x120 with 24x24 flat patches measurably let neighboring texture
# leak into every filter's output near a patch border.
# ===========================================================================
W, H = 200, 150
N = W * H
SEED = 42

PEAK_ELECTRONS = 40.0                 # e-, expected signal at code value 255 (kPeakElectrons)
READ_NOISE_E = 2.0                    # e- rms, signal-independent (kReadNoiseE)
DN_PER_ELECTRON = 255.0 / PEAK_ELECTRONS   # = 6.375 DN/e- (kDnPerElectron)

# ===========================================================================
# SECTION 3 — scene layout. MUST MATCH src/kernels.cuh Section 3's Rect
# constants EXACTLY (same rectangles, same values) — every gate in main.cu
# reads pixels out of these same regions in BOTH the ground truth this
# script writes and every denoiser's output. 48x48 flat patches (not
# 24x24): main.cu measures only their INNER 16x16 (a 16-px erosion margin),
# keeping every method's spatial reach away from the patch border — see
# kernels.cuh's kFlatDark comment for why.
#
# Bright levels sit a comfortable margin below 255 so the noise model's
# upper tail is not clipped (kFlatBright's comment in kernels.cuh measures
# the ~17% measured-std bias code value 224 produced before this margin
# was added — biasing noise_model_sanity's measured-vs-predicted ratio,
# not the noise model itself).
FLAT_DARK = (8, 56, 8, 56)            # (x0, x1, y0, y1), half-open
FLAT_MID = (144, 192, 8, 56)
FLAT_BRIGHT = (8, 56, 94, 142)
FLAT_DARK_DN = 28.0
FLAT_MID_DN = 128.0
FLAT_BRIGHT_DN = 175.0

EDGE_REGION = (70, 136, 64, 90)
EDGE_STEP_X = 103
EDGE_LO_DN = 24.0
EDGE_HI_DN = 200.0

FINE_DETAIL = (144, 192, 94, 142)
FINE_STRIPE_PERIOD = 4
FINE_LO_DN = 50.0
FINE_HI_DN = 200.0

TEXTURE_ROI = (64, 136, 8, 56)        # documented here for cross-reference; not used during generation

# ---------------------------------------------------------------------------
# Background texture: a THREE-OCTAVE hashed value-noise field (base mean
# TEX_BASE, three (cell_px, amplitude, salt) octaves summed). Deliberately
# HASHED rather than a strict repeating tile: a perfectly periodic texture
# is locally IDENTICAL everywhere it repeats — the exact self-similarity
# trap 01.04's make_synthetic.py documents for checkerboard corners (a
# strict two-tone checkerboard looks the same at every interior corner,
# rotated 90 degrees) and fixes there with a 5-value hashed cell palette.
# Here the analogous risk is NLM/BM3D-lite block-matching against
# EXACT-duplicate patches everywhere, which would flatter both methods with
# an unrealistically easy search problem; three independently-hashed noise
# octaves give the texture genuine, imperfect self-similarity (some patches
# resemble others; none are exact tile copies) — a fair, honest test of
# patch-similarity search.
# ---------------------------------------------------------------------------
TEX_BASE = 120.0
TEX_OCTAVES = [   # (cell_px, amplitude_dn, hash_salt)
    (40.0, 26.0, 401),
    (16.0, 14.0, 402),
    (7.0, 8.0, 403),
]


def hashed_unit(x: int, y: int, salt: int) -> float:
    """Deterministic, per-INTEGER-pixel pseudo-random value in [-1, 1] (the
    01.09 precedent — one fresh xorshift32 stream per (x, y, salt) triple,
    used as the four LATTICE CORNERS value_noise() interpolates between)."""
    seed = (x * 374761393 + y * 668265263 + salt * 2246822519) & 0xFFFFFFFF
    return XorShift32(seed).next_signed()


def value_noise(x: float, y: float, cell_px: float, salt: int) -> float:
    """Smooth, deterministic hashed lattice noise in [-1, 1] at continuous
    (x, y): bilinear interpolation between four hashed cell-corner values
    (the 01.08/01.09 technique, parameterized here by cell_px so
    scene_texture() below can call it at three different spatial scales)."""
    cxf, cyf = x / cell_px, y / cell_px
    x0, y0 = math.floor(cxf), math.floor(cyf)
    fx, fy = cxf - x0, cyf - y0

    def corner(ix: float, iy: float) -> float:
        return hashed_unit(int(ix), int(iy), salt)

    h00, h10 = corner(x0, y0), corner(x0 + 1, y0)
    h01, h11 = corner(x0, y0 + 1), corner(x0 + 1, y0 + 1)
    h0 = h00 * (1.0 - fx) + h10 * fx
    h1 = h01 * (1.0 - fx) + h11 * fx
    return h0 * (1.0 - fy) + h1 * fy


def scene_texture(x: int, y: int) -> float:
    """The multi-scale hashed background, in DN units. Pixel-CENTER sampled
    (x+0.5, y+0.5), matching 01.09's convention."""
    xf, yf = x + 0.5, y + 0.5
    v = TEX_BASE
    for cell_px, ampl, salt in TEX_OCTAVES:
        v += ampl * value_noise(xf, yf, cell_px, salt)
    return v


def _in_rect(x: int, y: int, rect) -> bool:
    x0, x1, y0, y1 = rect
    return x0 <= x < x1 and y0 <= y < y1


def clean_dn_at(x: int, y: int) -> float:
    """The CLEAN (noise-free) scene value at pixel (x, y), in DN units — the
    composition order is: hashed texture background, then the flat patches/
    edge/fine-detail rectangles PAINTED OVER it (01.09's swatch-over-vignette
    idiom, applied to a scene instead of a calibration target). The
    rectangles are mutually disjoint by construction, so check order does
    not matter for correctness."""
    if _in_rect(x, y, FLAT_DARK):
        return FLAT_DARK_DN
    if _in_rect(x, y, FLAT_MID):
        return FLAT_MID_DN
    if _in_rect(x, y, FLAT_BRIGHT):
        return FLAT_BRIGHT_DN
    if _in_rect(x, y, EDGE_REGION):
        return EDGE_LO_DN if x < EDGE_STEP_X else EDGE_HI_DN
    if _in_rect(x, y, FINE_DETAIL):
        period_pos = (x - FINE_DETAIL[0]) % FINE_STRIPE_PERIOD
        return FINE_LO_DN if period_pos < (FINE_STRIPE_PERIOD // 2) else FINE_HI_DN
    return scene_texture(x, y)


def render_noisy(clean_flat, seed: int) -> bytearray:
    """Draw ONE noisy 8-bit frame from the clean ground truth: EXACT Poisson
    shot noise (poisson_knuth) at each pixel's expected electron count, plus
    additive Gaussian read noise, converted back to DN and quantized. ONE
    continuous XorShift32 stream drives every draw in raster order — see
    kernels.cuh Section 2 / this file's module docstring for the formula's
    derivation."""
    rng = XorShift32(seed)
    out = bytearray(N)
    for i in range(N):
        clean_dn = clean_flat[i]
        signal_e = clean_dn / DN_PER_ELECTRON
        shot_e = float(poisson_knuth(rng, signal_e))            # EXACT Poisson draw, in electrons
        noisy_e = shot_e + rng.next_gaussian() * READ_NOISE_E   # + independent Gaussian read noise
        noisy_dn = noisy_e * DN_PER_ELECTRON                     # back to DN via the fixed linear gain
        clipped = 0 if noisy_dn < 0.0 else (255 if noisy_dn > 255.0 else int(noisy_dn + 0.5))
        out[i] = clipped
    return out


def write_pgm(path: str, width: int, height: int, gray) -> None:
    """Minimal binary PGM (P5) writer — the 01.01/01.03/01.08/01.09
    convention, reimplemented independently here (never shared code with
    the C++ reader in main.cu, CLAUDE.md §8)."""
    with open(path, "wb") as f:
        f.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        f.write(bytes(gray))


def main() -> None:
    sample_dir = os.path.join(os.path.dirname(__file__), "..", "data", "sample")
    os.makedirs(sample_dir, exist_ok=True)

    stale = os.path.join(sample_dir, "saxpy_sample.csv")
    if os.path.exists(stale):
        os.remove(stale)   # this project reads no such file — remove the scaffold-era placeholder

    # ---- the clean ground-truth scene (computed once; every gate in
    # main.cu reads it directly — never regenerated by the C++/CUDA side) --
    clean_flat_f = [clean_dn_at(x, y) for y in range(H) for x in range(W)]
    clean_u8 = bytearray(
        0 if v < 0.0 else (255 if v > 255.0 else int(v + 0.5)) for v in clean_flat_f
    )
    write_pgm(os.path.join(sample_dir, "clean.pgm"), W, H, clean_u8)
    print(f"[make_synthetic] wrote clean.pgm: range [{min(clean_flat_f):.1f}, "
          f"{max(clean_flat_f):.1f}] DN (texture base {TEX_BASE}, 3 hashed octaves, "
          f"+ 3 flat patches + 1 step edge + 1 fine-detail ruling)")

    # ---- the noisy frame every denoiser actually sees ----------------------
    noisy_u8 = render_noisy(clean_flat_f, SEED)
    write_pgm(os.path.join(sample_dir, "noisy.pgm"), W, H, noisy_u8)
    mean_noisy = sum(noisy_u8) / float(N)
    print(f"[make_synthetic] wrote noisy.pgm: mean {mean_noisy:.2f} DN (seed={SEED}, "
          f"exact Poisson shot noise + N(0,{READ_NOISE_E}^2 e-^2) read noise + 8-bit "
          f"quantization; peak signal {PEAK_ELECTRONS:.0f} e- at code value 255) [synthetic]")

    # ---- a quick, honest per-flat-patch noise summary (a developer sanity
    # check; the AUTHORITATIVE version of this same comparison is main.cu's
    # noise_model_sanity gate, computed independently in C++) --------------
    def patch_stats(rect, clean_dn):
        x0, x1, y0, y1 = rect
        vals = [float(noisy_u8[y * W + x]) - clean_dn for y in range(y0, y1) for x in range(x0, x1)]
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        return math.sqrt(var)

    for name, rect, dn in (("dark", FLAT_DARK, FLAT_DARK_DN),
                           ("mid", FLAT_MID, FLAT_MID_DN),
                           ("bright", FLAT_BRIGHT, FLAT_BRIGHT_DN)):
        std = patch_stats(rect, dn)
        signal_e = dn / DN_PER_ELECTRON
        predicted = math.sqrt(signal_e + READ_NOISE_E * READ_NOISE_E) * DN_PER_ELECTRON
        print(f"[make_synthetic]   flat_{name} (clean={dn:.0f} DN, {signal_e:.2f} e-): "
              f"measured noisy std = {std:.2f} DN, analytic prediction = {predicted:.2f} DN")

    # ---- params.csv: every generation constant a downstream reader might
    # need (the 01.09 precedent) -------------------------------------------
    params_path = os.path.join(sample_dir, "params.csv")
    with open(params_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        w.writerow(["W", W]); w.writerow(["H", H]); w.writerow(["SEED", SEED])
        w.writerow(["PEAK_ELECTRONS", PEAK_ELECTRONS])
        w.writerow(["READ_NOISE_E", READ_NOISE_E])
        w.writerow(["DN_PER_ELECTRON", DN_PER_ELECTRON])
        w.writerow(["TEX_BASE", TEX_BASE])
        for cell_px, ampl, salt in TEX_OCTAVES:
            w.writerow([f"TEX_OCTAVE_cell{int(cell_px)}", f"ampl={ampl},salt={salt}"])
        for label, rect in (("FLAT_DARK", FLAT_DARK), ("FLAT_MID", FLAT_MID),
                           ("FLAT_BRIGHT", FLAT_BRIGHT), ("EDGE_REGION", EDGE_REGION),
                           ("FINE_DETAIL", FINE_DETAIL), ("TEXTURE_ROI", TEXTURE_ROI)):
            w.writerow([label, f"{rect}"])
        w.writerow(["EDGE_STEP_X", EDGE_STEP_X])
        w.writerow(["EDGE_LO_DN", EDGE_LO_DN]); w.writerow(["EDGE_HI_DN", EDGE_HI_DN])

    print(f"[make_synthetic] wrote params.csv, clean.pgm, noisy.pgm into {os.path.abspath(sample_dir)}")


if __name__ == "__main__":
    main()
