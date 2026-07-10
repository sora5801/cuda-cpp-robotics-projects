# ===========================================================================
# download_data.ps1 — public-dataset fetcher for 17.01 (Batched Lambert solvers + porkchop plot generation)
#
# 17.01 DECISION: no public dataset is fetched here. This project's two
# bodies are SYNTHETIC coplanar circular orbits (an Earth-like body at
# 1 AU, a Mars-like body at 1.524 AU) — the whole scenario is six numbers,
# not recordings (see ../data/README.md and ../scripts/make_synthetic.py).
#
# Real planetary ephemerides ARE public and free — NASA JPL publishes SPICE
# kernels (https://naif.jpl.nasa.gov/naif/data.html) and the Horizons system
# (https://ssd.jpl.nasa.gov/horizons/) serves position/velocity vectors for
# every major body, no registration wall, permissive reuse terms. Swapping
# them in is a real, honest next step for this project (README §11 "Prior
# art", THEORY.md §where-this-sits-in-the-real-world) — it changes only
# body_state()'s inputs (real r(t), v(t) instead of the closed-form circular
# orbit) in kernels.cu/reference_cpu.cpp; the Lambert solver itself does not
# care where its position/velocity vectors came from. It is explicitly OUT
# OF SCOPE for this v1: a real ephemeris turns "download nothing, build
# offline" into "fetch and parse a SPICE kernel", a genuinely different
# project shape this repo's Phase 1 flagships deliberately avoid taking on.
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
Write-Host "[download_data] Project 17.01 uses synthetic sample data (repo default; CLAUDE.md paragraph 8)."
Write-Host "[download_data] Nothing to download. Generate/regenerate the sample with:"
Write-Host "[download_data]     python $PSScriptRoot\make_synthetic.py"
Write-Host "[download_data] Real ephemerides ARE public (NASA JPL SPICE/Horizons) but are out of scope for v1 - see the DECISION note above."
exit 0
