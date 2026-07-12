#!/usr/bin/env bash
# ===========================================================================
# download_data.sh — public-dataset fetcher for 02.19 (PointPillars/CenterPoint voxelization + scatter kernels feeding TensorRT)
# (Linux/macOS twin of download_data.ps1 — keep the two in sync.)
#
# DECISION (not a placeholder): this project stays synthetic-only. A real
# BEV detection dataset (KITTI, nuScenes) would teach realistic point
# statistics, but not the two things this project's gates actually need --
# EXACT ground-truth object centers (for detection_closure) and a
# DELIBERATE, precisely-placed cap-overflow pillar (for cap_truncation,
# kernels.cuh's file header). Both require synthesis with known-by-
# construction truth; a real scan cannot supply either honestly. This
# script therefore stays the intentional no-op below (CLAUDE.md paragraph 8:
# synthetic-first is the default, and here it is also the *better* choice,
# not merely the fallback).
#
# Repo policy (CLAUDE.md paragraph 8) for any FUTURE real fetch here:
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
#   URL='https://example.org/dataset/tiny_subset.zip'    # official source
#   SHA='SHA256-HEX-HERE'                                # pin the bytes (~123 MB)
#   DEST="$(dirname "$0")/../data/downloaded/tiny_subset.zip"
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
echo "[download_data] Project 02.19 uses synthetic sample data (repo default; CLAUDE.md paragraph 8)."
echo "[download_data] Nothing to download. Generate/regenerate the sample with:"
echo "[download_data]     python $(dirname "$0")/make_synthetic.py"
exit 0
