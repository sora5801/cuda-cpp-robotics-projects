# ===========================================================================
# download_data.ps1 — public-dataset fetcher for 02.06 (ICP: point-to-point → point-to-plane → GICP, all batched)
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
Write-Host "[download_data] Project 02.06 uses synthetic sample data (repo default; CLAUDE.md paragraph 8)."
Write-Host "[download_data] Nothing to download. Generate/regenerate the sample with:"
Write-Host "[download_data]     python $PSScriptRoot\make_synthetic.py"
exit 0
