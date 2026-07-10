#!/usr/bin/env bash
# ===========================================================================
# download_data.sh — public-dataset fetcher for 03.01 (FMCW radar cube processing: range-Doppler-angle FFTs + CA/OS-CFAR detection)
# (Linux/macOS twin of download_data.ps1 — keep the two in sync.)
#
# DECISION (this project): synthetic-only. Real automotive radar cube /
# ADC-level recordings are effectively never published raw (radar OEMs
# treat raw cubes as proprietary; public datasets like nuScenes/RADIal ship
# already-processed detections or spectrograms, not the Ns x Nc x Na
# complex baseband this project teaches from). Nothing here would teach
# the range/Doppler/angle-FFT pipeline better than a synthetic cube with
# EXACT, documented ground truth (CLAUDE.md paragraph 8's synthetic-first
# default) — so this script is an honest, permanent no-op.
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
echo "[download_data] Project 03.01 uses synthetic sample data (repo default; CLAUDE.md paragraph 8)."
echo "[download_data] Nothing to download. Generate/regenerate the sample with:"
echo "[download_data]     python $(dirname "$0")/make_synthetic.py"
exit 0
