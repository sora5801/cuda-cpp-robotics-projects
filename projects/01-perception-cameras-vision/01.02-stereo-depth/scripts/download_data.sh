#!/usr/bin/env bash
# ===========================================================================
# download_data.sh — public-dataset fetcher for 01.02 (Stereo depth: block matching, then Semi-Global Matching (SGM) kernels)
# (Linux/macOS twin of download_data.ps1 — keep the two in sync.)
#
# 01.02 DECISION: no public dataset is fetched here — v1 ships fully
# synthetic (see ../data/README.md "Why synthetic, not Middlebury/KITTI").
# In short:
#   * Middlebury Stereo (2001-2021 sets) — free for research use, but no
#     blanket redistribution right into a public MIT-licensed repo; the fair
#     path is linking learners to the official site, not re-hosting frames.
#   * KITTI Stereo — explicitly non-commercial / no-redistribution (CLAUDE.md
#     paragraph 8 names KITTI as the standing example of exactly this case).
#   * Both give SPARSE ground truth, which would force this project's
#     ground-truth gate to work around missing values before it could even
#     compare block matching against SGM — muddying the point of the demo.
# ../scripts/make_synthetic.py instead authors the scene directly in
# disparity space and derives the right image + an EXACT occlusion mask via
# physically-correct z-buffer forward warping — denser and more honest than
# either public set for THIS project's teaching goal, with zero license risk.
#
# Want to try a real photograph next? README.md "Exercises" points at both
# datasets' official pages — study them, do not mirror them here.
#
# Repo policy (CLAUDE.md paragraph 8) for any future real fetch here:
#   * Idempotent — safe to re-run; skip files that already exist with the
#     right checksum.
#   * Documented — source URL, expected size, SHA-256 checksum, and LICENSE
#     stated in comments here AND in ../data/README.md.
#   * License-respecting — non-commercial / no-redistribution datasets get
#     pointed at their official source; registration walls never bypassed;
#     the committed sample stays synthetic in those cases.
#   * Downloads land in ../data/downloaded/ (git-ignored), NEVER in
#     ../data/sample/.
#
# Illustrative shape of a real fetch (kept as a comment on purpose):
#   URL='https://vision.middlebury.edu/stereo/data/...'   # official source
#   SHA='SHA256-HEX-HERE'                                 # pin the bytes
#   DEST="$(dirname "$0")/../data/downloaded/middlebury_2014_subset.zip"
#   mkdir -p "$(dirname "$DEST")"
#   if [ ! -f "$DEST" ] || ! echo "$SHA  $DEST" | sha256sum -c - >/dev/null 2>&1; then
#       curl -L -o "$DEST" "$URL"
#       echo "$SHA  $DEST" | sha256sum -c -   # hard-fail on checksum mismatch
#   fi
# ===========================================================================
set -euo pipefail

# Default body: this project uses SYNTHETIC data (the repository default —
# CLAUDE.md paragraph 8), so there is nothing to download. Exit 0 so callers
# can invoke every project's script uniformly.
echo "[download_data] Project 01.02 uses synthetic sample data (repo default; CLAUDE.md paragraph 8)."
echo "[download_data] DECISION: Middlebury/KITTI exist but license/redistribution terms and sparse"
echo "[download_data] ground truth favor synthetic for v1 - see ../data/README.md for the full reasoning."
echo "[download_data] Nothing to download. Generate/regenerate the sample with:"
echo "[download_data]     python $(dirname "$0")/make_synthetic.py"
exit 0
