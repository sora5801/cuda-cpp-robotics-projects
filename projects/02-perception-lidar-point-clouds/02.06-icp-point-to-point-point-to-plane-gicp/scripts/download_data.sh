#!/usr/bin/env bash
# ===========================================================================
# download_data.sh — public-dataset fetcher for 02.06 (ICP: point-to-point → point-to-plane → GICP, all batched)
# (Linux/macOS twin of download_data.ps1 — keep the two in sync.)
#
# DECISION (CLAUDE.md paragraph 8): this project stays synthetic-only. ICP
# needs a KNOWN ground-truth transform to be verifiable at all, and public
# LiDAR datasets (KITTI, nuScenes, ...) do not ship exact scan-to-scan
# ground truth of the precision this project's pose-error gate checks —
# and both carry non-commercial/no-redistribution licenses that would
# forbid committing even a tiny derived sample here (CLAUDE.md paragraph 8
# names exactly this KITTI/nuScenes case). A learner who wants to try real
# LiDAR scans should point 02.08 (deskewing) or 05.09 (LIO) at KITTI's
# official download portal directly — this project's synthetic room (see
# ../data/README.md) is deliberately the more teachable input for THIS
# lesson (a controllable ground truth beats an uncontrollable real one).
# This script therefore remains an honest no-op below.
#
# Repo policy (CLAUDE.md paragraph 8) for any real implementation here:
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
echo "[download_data] Project 02.06 uses synthetic sample data (repo default; CLAUDE.md paragraph 8)."
echo "[download_data] Nothing to download. Generate/regenerate the sample with:"
echo "[download_data]     python $(dirname "$0")/make_synthetic.py"
exit 0
