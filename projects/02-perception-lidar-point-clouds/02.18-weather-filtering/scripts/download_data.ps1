# ===========================================================================
# download_data.ps1 — public-dataset fetcher for 02.18 (Weather filtering: snow/rain/dust outlier removal (DROR/LIOR))
#
# DECISION (not a placeholder): this project stays synthetic-only. The real
# public weather-LiDAR datasets that would apply here — CADC (Canadian
# Adverse Driving Conditions) and WADS (Winter Adverse Driving dataSet, both
# named in THEORY.md "Where this sits in the real world") — ship raw driving
# logs, not a per-point real/scatterer ground-truth label; recovering that
# label from either would need hand-annotation this repo cannot do
# automatically, and CADC in particular carries a non-commercial research
# license (README "Data" states this explicitly, CLAUDE.md paragraph 8: such
# datasets are pointed at, never mirrored). Synthesizing the physics instead
# (scripts/make_synthetic.py) gives EXACT per-point ground truth for free —
# the deciding factor, not a shortcut.
#
# Repo policy (CLAUDE.md paragraph 8) for any future real fetch added here:
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
Write-Host "[download_data] Project 02.18 uses synthetic sample data (repo default; CLAUDE.md paragraph 8)."
Write-Host "[download_data] Nothing to download. Generate/regenerate the sample with:"
Write-Host "[download_data]     python $PSScriptRoot\make_synthetic.py"
exit 0
