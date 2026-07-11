#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample-data generator for 01.15
(Background subtraction for fixed-workspace cells).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
-----------------------------------------------------------------
A real camera recording of a work cell can never hand us EXACT ground truth
for "which pixels are foreground, right now" — we would be guessing at our
own oracle. An analytic, hand-specified scene function gives exact ground
truth for free: every event below (an arm sweeping through, a box placed
and left, a status lamp blinking, a slow illumination ramp) is a closed-form
function of frame index and pixel position, so main.cu can recompute the
SAME rectangles as ground truth without ever having to "detect" anything.

What this script writes (into ../data/sample/, all committed, all tiny)
-------------------------------------------------------------------------
  frames/frame_000.pgm .. frame_159.pgm   — 128x96 grayscale, P5 binary,
      the full 160-frame designed sequence (see "Size decision" below).

Size decision (documented per the project brief; see ../data/README.md for
the byte math in full): the catalog's illustrative 240x180 frame size would
commit ~6.8 MiB for all 160 frames — far larger than any other sample in
this repository (most are under a few hundred KiB; the repo's biggest
neighbor, 30.01's bundled multi-milestone sample, is 1.5 MiB). Reduced to
128x96, the full offline-runnable sequence is ~1.9 MiB — the demo MUST run
with zero downloads and zero Python at run time (CLAUDE.md paragraph 4), so
"commit fewer frames" was rejected (it would silently shrink the designed
event schedule); "commit smaller frames" was not.

Determinism (CLAUDE.md paragraph 12 / this project's brief): TWO
independent deterministic sources, kept conceptually separate on purpose:
  * A hand-rolled xorshift32 PRNG (Marsaglia) — the SAME 4-line recurrence
    project 01.04's make_synthetic.py uses for its own reasons — drives the
    STATEFUL per-pixel sensor-noise stream (one Box-Muller draw per pixel
    per frame, consumed in raster-then-frame order, seed 42). This is a
    TIME-VARYING signal: replaying it never gives the same pixel twice.
  * A stateless integer mixing hash (pixel_hash_u32, NOT xorshift32) drives
    the FIXED spatial texture pattern (the bench's "wood grain," the wall's
    subtle mottling). This is deliberately NOT a PRNG stream: it is a pure
    function of (x, y, region_salt), so the SAME pixel gets the SAME
    texture offset on every frame — a real material's surface pattern does
    not re-randomize itself 30 times a second. Conflating the two would
    turn "spatial texture" into "flicker," a different physical phenomenon
    (and would break the deterministic background regions this project's
    models are meant to learn).
No numpy, no Python `random` module — CLAUDE.md's stdlib-only rule for this
project.

Geometry/timing constants below are the CANONICAL numbers this project's
five designed events use. THE SAME NUMBERS are hardcoded a second time, with
a cross-referencing comment, in ../src/kernels.cuh (that header explains why
a Python script cannot #include a .cuh file, and why duplicating DATA this
way is not the twin-independence concern CLAUDE.md's reference_cpu.cpp
ruling addresses — see that file's header). If you change an event's
geometry here, change kernels.cuh's copy in the same edit.

Usage
-----
    python make_synthetic.py                     # writes the committed default sample
    python make_synthetic.py --out-dir /tmp/x     # regenerate elsewhere for inspection
"""

import argparse
import math
import os
import sys

# ===========================================================================
# Sequence geometry — MUST MATCH ../src/kernels.cuh SECTION 1/2 EXACTLY.
# ===========================================================================
IMG_W = 128
IMG_H = 96
SEQ_T = 160
SEED = 42

# ---- static backdrop rectangles (half-open [x0,x0+w) x [y0,y0+h)) --------
BENCH_X, BENCH_Y, BENCH_W, BENCH_H = 5, 55, 118, 35
FIX1_X, FIX1_Y, FIX1_W, FIX1_H = 15, 60, 20, 8
FIX2_X, FIX2_Y, FIX2_W, FIX2_H = 95, 62, 15, 10

WALL_MEAN, WALL_TEX_HALF = 90.0, 6
BENCH_MEAN, BENCH_TEX_HALF = 140.0, 10
FIXTURE_MEAN, FIXTURE_TEX_HALF = 55.0, 5

# ---- E1 / E5: the two-link arm sweep (kernels.cuh SECTION 2) -------------
E1_FRAME_START, E1_FRAME_END = 20, 50
E5_FRAME_START, E5_FRAME_END = 130, 150
ARM_A_W, ARM_A_H = 14, 10
ARM_B_W, ARM_B_H = 10, 8
ARM_B_DX, ARM_B_DY = 16, 6
E1_ARM_Y, E1_X_START, E1_X_END = 40, 10, 100
E5_ARM_Y, E5_X_START, E5_X_END = 15, 30, 80   # capped at 80 -- see kernels.cuh's comment on why not 100
ARM_MEAN, ARM_TEX_HALF = 175.0, 8

# ---- E2: the absorption-test box (placed at frame 60, stays forever) ----
E2_FRAME_PLACED = 60
BOX_X, BOX_Y, BOX_W, BOX_H = 70, 70, 18, 14
BOX_MEAN, BOX_TEX_HALF = 205.0, 5

# ---- E4: the blinking status lamp (the bimodal-background lesson) --------
LAMP_X, LAMP_Y, LAMP_W, LAMP_H = 100, 10, 10, 8
LAMP_PERIOD_FRAMES = 8
LAMP_HIGH, LAMP_LOW, LAMP_TEX_HALF = 200.0, 60.0, 3

# ---- E3: the uniform illumination ramp -----------------------------------
ILLUM_RAMP_FRAC = 0.15

# ---- sensor noise (Python-only: main.cu never re-derives noise, it just
# reads the pixel values this script already baked in) --------------------
NOISE_SIGMA = 3.0   # intensity units -- matches ../src/kernels.cuh's SG_VAR_FLOOR = 9.0 = NOISE_SIGMA**2


# ===========================================================================
# xorshift32 -- Marsaglia's 32-bit xorshift PRNG, one step. Byte-for-byte
# the same recurrence as project 01.04's make_synthetic.py -- see that
# file for the "why a hand-rolled PRNG, not Python's Mersenne Twister"
# rationale (CLAUDE.md determinism rule, applied identically here).
# ===========================================================================
class XorShift32:
    def __init__(self, seed: int):
        assert seed != 0, "xorshift32 has one dead state: seed must be non-zero"
        self.state = seed & 0xFFFFFFFF

    def next_u32(self) -> int:
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17)
        x ^= (x << 5) & 0xFFFFFFFF
        x &= 0xFFFFFFFF
        self.state = x
        return x

    def next_unit(self) -> float:
        """Deterministic float in (0, 1], never exactly 0 (Box-Muller needs log(u) > -inf)."""
        u = self.next_u32()
        return ((u >> 8) + 1) / float(1 << 24)   # 24-bit mantissa -> (0,1], +1 excludes exact 0


def gaussian_pair(rng: XorShift32) -> float:
    """One Box-Muller transform: TWO independent uniform draws -> TWO
    independent standard-normal (mean 0, sigma 1) samples. We only ever
    consume z0 below (z1 is discarded) -- this halves throughput but keeps
    the "one noise draw per pixel per frame" bookkeeping trivially simple
    for a teaching script; README Exercises suggests using both.
    """
    u1 = rng.next_unit()
    u2 = rng.next_unit()
    r = math.sqrt(-2.0 * math.log(u1))
    z0 = r * math.cos(2.0 * math.pi * u2)
    return z0


# ===========================================================================
# pixel_hash_u32 -- a STATELESS integer mixing hash (Murmur3-finalizer
# style: multiply, xor-shift, multiply, xor-shift). Deliberately NOT
# xorshift32 -- see the module docstring's "two independent deterministic
# sources" note. Same (x, y, salt) always produces the same output, on
# every call, forever -- that is what makes a FIXED spatial texture fixed.
# ===========================================================================
def pixel_hash_u32(x: int, y: int, salt: int) -> int:
    h = (x * 0x27220A5F) ^ (y * 0x9E3779B1) ^ (salt * 0x85EBCA6B)
    h &= 0xFFFFFFFF
    h ^= (h >> 15)
    h = (h * 0x2C1B3C6D) & 0xFFFFFFFF
    h ^= (h >> 12)
    h = (h * 0x297A2D39) & 0xFFFFFFFF
    h ^= (h >> 15)
    return h & 0xFFFFFFFF


def texture_offset(x: int, y: int, salt: int, half_range: int) -> int:
    """A deterministic per-pixel texture wobble in [-half_range, +half_range]."""
    if half_range <= 0:
        return 0
    span = 2 * half_range + 1
    return (pixel_hash_u32(x, y, salt) % span) - half_range


# ===========================================================================
# round_half_up -- floor(v + 0.5). Used for the arm's linearly-interpolated
# x position, which is not always an integer (E5's 50px/20-frame sweep hits
# half-integers). ../src/main.cu uses std::floor(v + 0.5) for the SAME
# formula when it recomputes ground truth -- the two MUST agree bit-for-bit
# on every frame's rectangle, or the intrusion_detection gate would compare
# a rendered arm against a truth rectangle one pixel off from where it was
# actually drawn. floor(v+0.5) (round-half-up) is used instead of either
# language's native round() specifically because Python's round() uses
# banker's rounding (round-half-to-even) while C++'s std::lround rounds
# half away from zero -- these agree with EACH OTHER almost always but not
# by written guarantee, whereas floor(v+0.5) is the exact same three
# floating-point operations in both languages.
# ===========================================================================
def round_half_up(v: float) -> int:
    return int(math.floor(v + 0.5))


def arm_link_a_x(t: int, frame_start: int, frame_end: int, x_start: int, x_end: int) -> int:
    span = float(frame_end - frame_start)
    frac = (t - frame_start) / span if span > 0.0 else 0.0
    return round_half_up(x_start + frac * (x_end - x_start))


def in_rect(x: int, y: int, rx: int, ry: int, rw: int, rh: int) -> bool:
    return rx <= x < rx + rw and ry <= y < ry + rh


# ===========================================================================
# render_pixel_base -- the CLEAN (pre-noise, pre-illumination) intensity of
# pixel (x, y) at frame t, picking the first matching region in PRIORITY
# order (an event overlay always wins over the static backdrop beneath it).
# Returns (mean, texture_half_range, texture_salt) so the caller adds the
# SAME kind of hashed wobble to every region, just with region-specific
# parameters -- one shared "add texture" step instead of one per branch.
# ===========================================================================
def render_pixel_base(x: int, y: int, t: int):
    # ---- E1 / E5: arm intrusion (two co-moving rectangles) --------------
    for (fs, fe, ay, xs, xe) in ((E1_FRAME_START, E1_FRAME_END, E1_ARM_Y, E1_X_START, E1_X_END),
                                  (E5_FRAME_START, E5_FRAME_END, E5_ARM_Y, E5_X_START, E5_X_END)):
        if fs <= t <= fe:
            ax = arm_link_a_x(t, fs, fe, xs, xe)
            if in_rect(x, y, ax, ay, ARM_A_W, ARM_A_H):
                return ARM_MEAN, ARM_TEX_HALF, 101
            bx, by = ax + ARM_B_DX, ay + ARM_B_DY
            if in_rect(x, y, bx, by, ARM_B_W, ARM_B_H):
                return ARM_MEAN, ARM_TEX_HALF, 102

    # ---- E2: the absorption-test box, once placed ------------------------
    if t >= E2_FRAME_PLACED and in_rect(x, y, BOX_X, BOX_Y, BOX_W, BOX_H):
        return BOX_MEAN, BOX_TEX_HALF, 103

    # ---- E4: the blinking status lamp (always present, from frame 0) -----
    if in_rect(x, y, LAMP_X, LAMP_Y, LAMP_W, LAMP_H):
        state_high = ((t // LAMP_PERIOD_FRAMES) % 2) == 0
        return (LAMP_HIGH if state_high else LAMP_LOW), LAMP_TEX_HALF, 104

    # ---- static backdrop: fixtures, then bench, then wall -----------------
    if in_rect(x, y, FIX1_X, FIX1_Y, FIX1_W, FIX1_H) or in_rect(x, y, FIX2_X, FIX2_Y, FIX2_W, FIX2_H):
        return FIXTURE_MEAN, FIXTURE_TEX_HALF, 105
    if in_rect(x, y, BENCH_X, BENCH_Y, BENCH_W, BENCH_H):
        return BENCH_MEAN, BENCH_TEX_HALF, 106
    return WALL_MEAN, WALL_TEX_HALF, 107


def illumination_scale(t: int) -> float:
    """L(t) = 1 + ILLUM_RAMP_FRAC * t/(SEQ_T-1) -- see kernels.cuh SECTION 2."""
    return 1.0 + ILLUM_RAMP_FRAC * (t / float(SEQ_T - 1))


# ===========================================================================
# PGM (P5) writer -- minimal, matching the strict-reader convention this
# repository's other projects use (e.g. 01.04's read_pgm): we only ever
# need to WRITE files our own reader (main.cu) will read back, so the
# format is the plainest legal P5: one-line-per-header-field, ASCII, then
# raw 8-bit binary payload -- no comments, no surprises.
# ===========================================================================
def write_pgm(path: str, w: int, h: int, pixels: bytes) -> None:
    with open(path, "wb") as f:
        f.write(b"P5\n")
        f.write(f"{w} {h}\n".encode("ascii"))
        f.write(b"255\n")
        f.write(pixels)


def render_sequence(out_dir: str) -> None:
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # ONE continuous noise stream across the whole sequence, consumed in
    # (t, y, x) order -- see the module docstring's "two independent
    # deterministic sources" note for why this is a SEPARATE stream from
    # the stateless texture hash above.
    noise_rng = XorShift32(SEED)

    total_bytes = 0
    for t in range(SEQ_T):
        L = illumination_scale(t)
        row_bytes = bytearray(IMG_W * IMG_H)
        for y in range(IMG_H):
            base_row = y * IMG_W
            for x in range(IMG_W):
                mean, tex_half, salt = render_pixel_base(x, y, t)
                clean = (mean + texture_offset(x, y, salt, tex_half)) * L
                noisy = clean + NOISE_SIGMA * gaussian_pair(noise_rng)
                # Clip to the legal uint8 range and round to the nearest
                # integer intensity -- the same "clip then quantize" order
                # a real 8-bit sensor's ADC effectively performs.
                v = round_half_up(max(0.0, min(255.0, noisy)))
                row_bytes[base_row + x] = v
        path = os.path.join(frames_dir, f"frame_{t:03d}.pgm")
        write_pgm(path, IMG_W, IMG_H, bytes(row_bytes))
        total_bytes += os.path.getsize(path)

    print(f"wrote {SEQ_T} frames ({IMG_W}x{IMG_H}) to {frames_dir}")
    print(f"total on-disk size: {total_bytes} bytes ({total_bytes / (1024 * 1024):.2f} MiB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=None,
                     help="output directory (default: ../data/sample relative to this script)")
    args = ap.parse_args()

    out_dir = args.out_dir
    if out_dir is None:
        here = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(here, "..", "data", "sample")
    render_sequence(out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
