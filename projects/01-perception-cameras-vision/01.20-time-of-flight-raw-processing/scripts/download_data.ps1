# ===========================================================================
# download_data.ps1 — public-dataset fetcher for 01.20 (Time-of-flight raw processing: phase unwrapping, flying-pixel removal)
#
# STATUS: intentional no-op. This project is 100% synthetic (scripts/
# make_synthetic.py) — no public iToF raw-tap dataset with the exact 4-tap,
# dual-frequency, ground-truth-labeled-flying-pixel shape this project needs
# is known to exist with a redistributable license, so there is nothing to
# fetch here (README "Limitations" states this honestly).
#
# Repo policy (CLAUDE.md paragraph 8) were a future maintainer to add a real
# fetch here:
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
Write-Host "[download_data] Project 01.20 uses synthetic sample data (repo default; CLAUDE.md paragraph 8)."
Write-Host "[download_data] Nothing to download. Generate/regenerate the sample with:"
Write-Host "[download_data]     python $PSScriptRoot\make_synthetic.py"
exit 0
