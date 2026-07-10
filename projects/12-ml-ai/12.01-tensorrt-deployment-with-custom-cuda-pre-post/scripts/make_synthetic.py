#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample generator for project 12.01
(TensorRT deployment with custom CUDA pre/post kernels: NMS, argmax decode,
keypoint extraction).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
---------------------------------------------------------------
This project's "network" is not trained — it is a small, DETERMINISTIC
tensor program whose weights are hand-designed so the whole pipeline is
checkable by hand (THEORY.md works the arithmetic). This script is the
SINGLE WRITER of that design: it emits

    ../data/sample/weights.bin        the 6-layer weight blob (460 bytes)
    ../data/sample/test_scene.ppm     an 80x80 RGB synthetic test image
    ../data/sample/ground_truth.csv   the KNOWN objects placed in that image

byte-identically every run (the only randomness — a mild background dither,
so the scene is not a flat, unrealistically clean gray — is seeded and
documented). ../src/kernels.cuh SECTION 1 declares the same constants this
script uses; the two files must be changed together (that header says so).

THE SCENE, briefly (THEORY.md has the full derivation): a mid-gray
background with a few solid rectangles, "red" (240,50,50) or "blue"
(50,50,240). The hand-designed weights compute, per output channel,
avg(R_norm) - avg(B_norm) (channel 0, "redness") and its negation
(channel 1, "blueness") — a linear combination that is exactly 0 on the
gray background and strongly positive inside an object of the matching
color, entirely independent of any learned parameter. Object size (15x15
source px) and the fixed anchor box (12x12 network-input px, i.e. exactly
15*0.8) are chosen so a single anchor, undecorated by any learned box
regression, already lands within the documented tolerance of every
object's true box.

Usage:
    python make_synthetic.py                 # the committed sample (seed 42)
    python make_synthetic.py --seed 7         # experiment; do NOT commit
"""

import argparse
import struct
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Scene / architecture constants — MUST match src/kernels.cuh SECTION 1
# exactly (this is the one place outside that header where they are
# repeated; a mismatch here would silently desync the committed sample from
# the compiled program's expectations — the strict loaders in main.cu catch
# a SIZE mismatch immediately, but not a semantic drift, so keep the two in
# lockstep by hand when either changes).
# ---------------------------------------------------------------------------
SRC_W = SRC_H = 80
BG = 128                     # background gray level, all channels
DITHER_AMPL = 3              # +/- per-channel background dither amplitude
RED = (240, 50, 50)
BLUE = (50, 50, 240)

# (x0, y0, w, h, class_id) in SOURCE-image pixels; class 0 = red, 1 = blue.
# Sizes are 15x15 so the network-input resize (scale 64/80 = 0.8) lands
# them at EXACTLY 12x12 — the fixed anchor size below.
OBJECTS = [
    (13, 13, 15, 15, 0),
    (52, 15, 15, 15, 1),
    (33, 51, 15, 15, 0),
]

CONV1_IN, CONV1_OUT, CONV1_K = 3, 2, 3
CONV2_IN, CONV2_OUT, CONV2_K = 2, 2, 3
HEAD_IN, HEAD_OUT, HEAD_K = 2, 6, 1
BIAS_SCORE = 1.0              # head_b[0]=head_b[1] = -BIAS_SCORE (see THEORY.md)

WEIGHTS_MAGIC = b"RCWTPK01"
WEIGHTS_VERSION = 1


def make_weights_bytes() -> bytes:
    """Build the 460-byte weight blob described in kernels.cuh SECTION 3.

    Every weight below is a small, hand-chosen constant — see the module
    docstring and THEORY.md "The math" for the derivation of WHY these
    particular numbers make the scene's objects separable. Returns the
    exact byte sequence main.cu's load_weight_blob() expects: an 8-byte
    magic, a uint32 format version, then float32 arrays back-to-back in
    the documented order — no struct padding, no ambiguity.
    """
    def zeros(*dims):
        n = 1
        for d in dims:
            n *= d
        return [0.0] * n

    # conv1: out0 = "redness" = +1/9 * avg(R) - 1/9 * avg(B), out1 = its
    # negation ("blueness"). Channel G is unused (weight 0) — a deliberate
    # scoping simplification (README "Limitations"): this synthetic task
    # only needs 2 of 3 camera channels to separate red from blue from gray.
    conv1_w = zeros(CONV1_OUT, CONV1_IN, CONV1_K, CONV1_K)
    def idx4(shape, o, i, y, x):
        _, I, K, _ = shape
        return ((o * I + i) * K + y) * K + x
    shape1 = (CONV1_OUT, CONV1_IN, CONV1_K, CONV1_K)
    for y in range(CONV1_K):
        for x in range(CONV1_K):
            conv1_w[idx4(shape1, 0, 0, y, x)] = 1.0 / 9.0    # out0, R
            conv1_w[idx4(shape1, 0, 2, y, x)] = -1.0 / 9.0   # out0, B
            conv1_w[idx4(shape1, 1, 0, y, x)] = -1.0 / 9.0   # out1, R
            conv1_w[idx4(shape1, 1, 2, y, x)] = 1.0 / 9.0    # out1, B
    conv1_b = [0.0, 0.0]

    # conv2: a further 3x3/stride-2 average-pool of EACH channel
    # independently (out_c reads only in_c) — downsamples 32x32 -> 16x16
    # without mixing the redness/blueness signals.
    shape2 = (CONV2_OUT, CONV2_IN, CONV2_K, CONV2_K)
    conv2_w = zeros(CONV2_OUT, CONV2_IN, CONV2_K, CONV2_K)
    for c in range(CONV2_OUT):
        for y in range(CONV2_K):
            for x in range(CONV2_K):
                conv2_w[idx4(shape2, c, c, y, x)] = 1.0 / 9.0
    conv2_b = [0.0, 0.0]

    # head (1x1 conv): class scores read the matching feature 1:1 minus a
    # fixed bias (the score threshold's complement — see THEORY.md); the 4
    # box-regression channels are all-zero weight/bias, so every decoded
    # box resolves to the bare anchor (README "Limitations" states this
    # honestly: this teaches the CORRECT anchor-decode arithmetic with a
    # placeholder regression, not a trained refinement).
    shapeH = (HEAD_OUT, HEAD_IN, HEAD_K, HEAD_K)
    head_w = zeros(HEAD_OUT, HEAD_IN, HEAD_K, HEAD_K)
    head_w[idx4(shapeH, 0, 0, 0, 0)] = 1.0   # class0 (red) score <- redness feature
    head_w[idx4(shapeH, 1, 1, 0, 0)] = 1.0   # class1 (blue) score <- blueness feature
    head_b = [-BIAS_SCORE, -BIAS_SCORE, 0.0, 0.0, 0.0, 0.0]

    payload = struct.pack(
        f"<{len(conv1_w)}f{len(conv1_b)}f{len(conv2_w)}f{len(conv2_b)}f{len(head_w)}f{len(head_b)}f",
        *conv1_w, *conv1_b, *conv2_w, *conv2_b, *head_w, *head_b,
    )
    return WEIGHTS_MAGIC + struct.pack("<I", WEIGHTS_VERSION) + payload


def make_scene_pixels(seed: int):
    """Return an SRC_H x SRC_W list-of-rows of (r,g,b) tuples: the mildly
    dithered gray background with OBJECTS painted on top, exactly as
    THEORY.md describes. The dither uses a LOCAL, seeded RNG (never the
    global one — CLAUDE.md paragraph 12 determinism discipline) so the
    file is byte-identical for a given seed on every machine.
    """
    rng = random.Random(seed)
    # Precompute the dither for every pixel/channel up front, in raster
    # order, so the byte stream is a pure function of (seed, position) —
    # not of draw ORDER relative to object painting below.
    dither = [[[rng.randint(-DITHER_AMPL, DITHER_AMPL) for _ in range(3)]
              for _ in range(SRC_W)] for _ in range(SRC_H)]

    rows = []
    for y in range(SRC_H):
        row = []
        for x in range(SRC_W):
            dr, dg, db = dither[y][x]
            r, g, b = BG + dr, BG + dg, BG + db
            for (ox, oy, ow, oh, cls) in OBJECTS:
                if ox <= x < ox + ow and oy <= y < oy + oh:
                    r, g, b = RED if cls == 0 else BLUE
                    break
            row.append((r, g, b))
        rows.append(row)
    return rows


def make_ppm_bytes(seed: int) -> bytes:
    """Encode make_scene_pixels() as a binary PPM (P6): the exact header
    format ../src/main.cu's load_ppm() parses ("P6\\n80 80\\n255\\n" then
    raw bytes) — see kernels.cuh for why this project uses this tiny,
    library-free image format instead of PNG/JPEG (CLAUDE.md paragraph 5:
    default dependency budget is the CUDA toolkit + C++17 stdlib only, and
    Python's stdlib on this machine has no PIL — see README "Data").
    """
    rows = make_scene_pixels(seed)
    header = f"P6\n{SRC_W} {SRC_H}\n255\n".encode("ascii")
    body = bytearray()
    for row in rows:
        for (r, g, b) in row:
            body += bytes((r, g, b))
    return header + bytes(body)


def make_ground_truth_csv() -> str:
    lines = [
        "# ground_truth.csv - SYNTHETIC ground-truth objects for project 12.01's test_scene.ppm",
        "# generated by scripts/make_synthetic.py - paired 1:1 with test_scene.ppm, do not hand-edit",
        "# OBJ,class_id,x0,y0,w,h : SOURCE-image pixel coords (int), class 0=red 1=blue",
        "# license: same as the repository (MIT) - fully synthetic, no external source",
    ]
    for (x0, y0, w, h, cls) in OBJECTS:
        lines.append(f"OBJ,{cls},{x0},{y0},{w},{h}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for the background dither (default 42; the committed sample)")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data" / "sample",
                    help="output directory (default ../data/sample)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    weights_bytes = make_weights_bytes()
    weights_path = args.out_dir / "weights.bin"
    with open(weights_path, "wb") as f:
        f.write(weights_bytes)

    ppm_bytes = make_ppm_bytes(args.seed)
    ppm_path = args.out_dir / "test_scene.ppm"
    with open(ppm_path, "wb") as f:
        f.write(ppm_bytes)

    gt_csv = make_ground_truth_csv()
    gt_path = args.out_dir / "ground_truth.csv"
    with open(gt_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(gt_csv)

    print(f"wrote {weights_path} ({len(weights_bytes)} bytes) - labeled SYNTHETIC, fixed (not trained)")
    print(f"wrote {ppm_path} ({len(ppm_bytes)} bytes: {SRC_W}x{SRC_H} RGB, dither seed={args.seed}) - labeled SYNTHETIC")
    print(f"wrote {gt_path} ({len(OBJECTS)} objects) - labeled SYNTHETIC")
    if args.seed != 42:
        print("note: non-default seed - fine for experiments, do NOT commit these files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
