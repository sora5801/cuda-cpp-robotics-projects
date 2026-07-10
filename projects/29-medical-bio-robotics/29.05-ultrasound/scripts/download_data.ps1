# ===========================================================================
# download_data.ps1 — public-dataset fetcher for 29.05 (Ultrasound: GPU beamforming, elastography, image-based servoing)
#
# DECIDED: this project stays synthetic-only, permanently — no public dataset
# fetch is implemented. Reason: there is no license that permits
# redistributing real patient ultrasound RF/channel data (the raw signal
# this project's beamformer consumes), and PICMUS-style public plane-wave
# datasets (README "Prior art") are simulated/phantom acquisitions best
# fetched and explored directly from their own source, not mirrored here.
# The committed phantom (data/sample/) is therefore the permanent source of
# truth; this script is an honest no-op, kept only so every project in the
# repo exposes the same callable script shape.
#
# Repo policy (CLAUDE.md paragraph 8) for any dataset fetch, for reference:
#   * Idempotent — safe to re-run; skip files that already exist with the
#     right checksum.
#   * Documented — source URL, expected size, SHA-256 checksum, and LICENSE
#     stated in comments here AND in ../data/README.md.
#   * License-respecting — non-commercial / no-redistribution datasets
#     (KITTI, nuScenes) get pointed at their official source with
#     instructions; registration walls are never bypassed; the committed
#     sample under ../data/sample/ stays synthetic in those cases.
#   * Downloads land in ../data/downloaded/ (git-ignored), NEVER in
#     ../data/sample/ (which holds only the tiny committed sample).
#
# Illustrative shape of a real fetch (kept as a comment on purpose):
#   $url      = 'https://example.org/dataset/tiny_subset.zip'   # official source
#   $expected = 'SHA256-HEX-HERE'                               # pin the bytes
#   $dest     = Join-Path $PSScriptRoot '..\data\downloaded\tiny_subset.zip'
#   if (-not (Test-Path $dest) -or
#       (Get-FileHash $dest -Algorithm SHA256).Hash -ne $expected) {
#       Invoke-WebRequest -Uri $url -OutFile $dest               # ~123 MB
#       if ((Get-FileHash $dest -Algorithm SHA256).Hash -ne $expected) {
#           Write-Error 'checksum mismatch — refusing to use the file'
#       }
#   }
# ===========================================================================

# Default body: this project uses SYNTHETIC data (the repository default —
# CLAUDE.md paragraph 8), so there is nothing to download. Exit 0 so callers
# and CI can invoke every project's script uniformly without special cases.
Write-Host "[download_data] Project 29.05 uses synthetic sample data (repo default; CLAUDE.md paragraph 8)."
Write-Host "[download_data] Nothing to download. Generate/regenerate the sample with:"
Write-Host "[download_data]     python $PSScriptRoot\make_synthetic.py"
exit 0
