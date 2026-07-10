#!/usr/bin/env bash
# ===========================================================================
# download_data.sh — public-dataset fetcher for 31.01 (Hamilton-Jacobi reachability: level-set grid solvers (stencil ops — GPU-perfect))
# (Linux/macOS twin of download_data.ps1 — keep the two in sync.)
#
# 31.01 DECISION: no public dataset applies — this project's "data" is a
# problem definition (grid, dynamics bound, target level, horizon), not
# recordings. The initial level function, every PDE sweep, and the analytic
# minimum-time oracle are all computed in-demo from that scenario plus
# closed-form mathematics (src/reference_cpu.cpp's min_time_to_origin).
# This script stays the honest no-op below; ../scripts/make_synthetic.py
# writes the scenario (see ../data/README.md).
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
echo "[download_data] Project 31.01 uses a synthetic scenario, not recordings (repo default; CLAUDE.md paragraph 8)."
echo "[download_data] Nothing to download. Generate/regenerate the sample with:"
echo "[download_data]     python $(dirname "$0")/make_synthetic.py"
exit 0
