# ===========================================================================
# download_data.ps1 — public-dataset fetcher for 01.02 (Stereo depth: block matching, then Semi-Global Matching (SGM) kernels)
#
# 01.02 DECISION: no public dataset is fetched here — v1 ships fully
# synthetic (see ../data/README.md "Why synthetic, not Middlebury/KITTI").
# In short:
#   * Middlebury Stereo (2001-2021 sets) — free for research use, but no
#     blanket redistribution right into a public MIT-licensed repo; the fair
#     path is linking learners to the official site, not re-hosting frames.
#   * KITTI Stereo — explicitly non-commercial / no-redistribution (CLAUDE.md
#     paragraph 8 names KITTI as the standing example of exactly this case).
#   * Both give SPARSE ground truth, which would force this project's
#     ground-truth gate to work around missing values before it could even
#     compare block matching against SGM — muddying the point of the demo.
# ../scripts/make_synthetic.py instead authors the scene directly in
# disparity space and derives the right image + an EXACT occlusion mask via
# physically-correct z-buffer forward warping — denser and more honest than
# either public set for THIS project's teaching goal, with zero license risk.
#
# Want to try a real photograph next? README.md "Exercises" points at both
# datasets' official pages — study them, do not mirror them here.
#
# Repo policy (CLAUDE.md paragraph 8) for any future real fetch here:
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
#   $url      = 'https://vision.middlebury.edu/stereo/data/...'      # official source
#   $expected = 'SHA256-HEX-HERE'                               # pin the bytes
#   $dest     = Join-Path $PSScriptRoot '..\data\downloaded\middlebury_2014_subset.zip'
#   if (-not (Test-Path $dest) -or
#       (Get-FileHash $dest -Algorithm SHA256).Hash -ne $expected) {
#       Invoke-WebRequest -Uri $url -OutFile $dest
#       if ((Get-FileHash $dest -Algorithm SHA256).Hash -ne $expected) {
#           Write-Error 'checksum mismatch — refusing to use the file'
#       }
#   }
# ===========================================================================

# Default body: this project uses SYNTHETIC data (the repository default —
# CLAUDE.md paragraph 8), so there is nothing to download. Exit 0 so callers
# and CI can invoke every project's script uniformly without special cases.
Write-Host "[download_data] Project 01.02 uses synthetic sample data (repo default; CLAUDE.md paragraph 8)."
Write-Host "[download_data] DECISION: Middlebury/KITTI exist but license/redistribution terms and sparse"
Write-Host "[download_data] ground truth favor synthetic for v1 - see ../data/README.md for the full reasoning."
Write-Host "[download_data] Nothing to download. Generate/regenerate the sample with:"
Write-Host "[download_data]     python $PSScriptRoot\make_synthetic.py"
exit 0
