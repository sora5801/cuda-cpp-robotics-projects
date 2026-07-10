# ===========================================================================
# download_data.ps1 — public-dataset fetcher for 23.01 (GPU costmaps: inflation, raytrace clearing, multi-layer fusion)
#
# DECISION (resolved, not a placeholder): this project uses SYNTHETIC data
# only (data/README.md) — a 2-D occupancy grid and a start/goal scenario are
# exactly the kind of thing this repo can synthesize with full ground truth,
# and a solvability-verified synthetic map teaches the costmap+DWA pipeline
# at least as well as any public dataset would. No public dataset was found
# to genuinely teach more, so this script is an intentional no-op, kept only
# for interface consistency with every other project's scripts/ folder.
#
# Repo policy (CLAUDE.md paragraph 8) for any future real implementation here:
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
Write-Host "[download_data] Project 23.01 uses synthetic sample data (repo default; CLAUDE.md paragraph 8)."
Write-Host "[download_data] Nothing to download. Generate/regenerate the sample with:"
Write-Host "[download_data]     python $PSScriptRoot\make_synthetic.py"
exit 0
