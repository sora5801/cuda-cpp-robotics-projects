#!/usr/bin/env bash
# ===========================================================================
# download_data.sh — public-dataset fetcher for 29.05 (Ultrasound: GPU beamforming, elastography, image-based servoing)
# (Linux/macOS twin of download_data.ps1 — keep the two in sync.)
#
# DECIDED: this project stays synthetic-only, permanently — no public dataset
# fetch is implemented. Reason: there is no license that permits
# redistributing real patient ultrasound RF/channel data (the raw signal
# this project's beamformer consumes), and PICMUS-style public plane-wave
# datasets (README "Prior art") are simulated/phantom acquisitions best
# fetched and explored directly from their own source, not mirrored here.
# The committed phantom (data/sample/) is therefore the permanent source of
# truth; this script is an honest no-op, kept only so every project in the
# repo exposes the same callable script shape (CLAUDE.md paragraph 8).
# ===========================================================================
set -euo pipefail

# Default body: this project uses SYNTHETIC data (the repository default —
# CLAUDE.md paragraph 8), so there is nothing to download. Exit 0 so callers
# can invoke every project's script uniformly.
echo "[download_data] Project 29.05 uses synthetic sample data (repo default; CLAUDE.md paragraph 8)."
echo "[download_data] Nothing to download. Generate/regenerate the sample with:"
echo "[download_data]     python $(dirname "$0")/make_synthetic.py"
exit 0
