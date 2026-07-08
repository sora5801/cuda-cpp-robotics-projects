# ===========================================================================
# run_demo.ps1 — one-command demo for 17.11 (Polyhedral asteroid gravity-field batch evaluation for hovering control)
#
# What this does (CLAUDE.md paragraph 6.3): builds the project if needed,
# runs it on the sample data, prints the output, and checks the STABLE lines
# of that output against expected_output.txt. Exit code 0 = demo passed.
#
# WORKS AS SCAFFOLDED: the template's SAXPY placeholder builds, runs, and
# matches its committed expected_output.txt out of the box — so running this
# script on a freshly scaffolded project is a full toolchain smoke test.
# TODO(scaffold): once the real implementation lands, this script usually
# needs NO changes — just keep expected_output.txt in sync with the stable
# lines main.cu prints (see the "output contract" comment in ../src/main.cu).
#
# Diffing rules (shared contract with expected_output.txt and main.cu):
#   * Lines in expected_output.txt starting with '#' are COMMENTS — ignored.
#   * Blank lines are ignored.
#   * Every remaining expected line must appear VERBATIM (after trailing-
#     whitespace trim) somewhere in the program's stdout. This "subset"
#     check lets the program also print non-deterministic lines (timings
#     "[time]", device info "[info]") that are deliberately not checked.
#
# Usage:  .\run_demo.ps1          (from anywhere; paths resolve relative to
#                                  this script, not the current directory)
# ===========================================================================

$ErrorActionPreference = 'Stop'   # any unexpected failure stops the script loudly

# ---- Paths (all relative to this script's location) -----------------------
$DemoDir    = $PSScriptRoot                          # .../demo
$ProjectDir = Split-Path -Parent $DemoDir            # the project root
$Slug       = 'polyhedral-asteroid-gravity-field-batch'                     # substituted by scaffold.py
$Sln        = Join-Path $ProjectDir "build\$Slug.sln"
# The .vcxproj pins OutDir to $(ProjectDir)x64\$(Configuration)\ — with the
# project file living in build\, the Release exe lands here deterministically:
$Exe        = Join-Path $ProjectDir "build\x64\Release\$Slug.exe"
$Expected   = Join-Path $DemoDir 'expected_output.txt'

# ---- 1) Build if the exe is missing ----------------------------------------
if (-not (Test-Path $Exe)) {
    Write-Host "[run_demo] $Slug.exe not found - building Release|x64 ..."

    # Locate MSBuild. Preferred: vswhere, the official VS locator that ships
    # at a FIXED path with every Visual Studio install - it answers "where is
    # the newest MSBuild?" without guessing at version-numbered directories.
    $msbuild = $null
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path $vswhere) {
        # -products * : include Community/Professional/Enterprise/BuildTools
        # -latest     : newest installed VS
        # -find       : glob under the install root for MSBuild.exe itself
        $msbuild = & $vswhere -latest -products * -requires Microsoft.Component.MSBuild `
                              -find 'MSBuild\**\Bin\MSBuild.exe' | Select-Object -First 1
    }
    # Fallback: the known VS 2026 Community path on the reference machine
    # (documented in docs/BUILD_GUIDE.md). Used only if vswhere is absent.
    if (-not $msbuild -or -not (Test-Path $msbuild)) {
        $fallback = 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe'
        if (Test-Path $fallback) { $msbuild = $fallback }
    }
    if (-not $msbuild) {
        Write-Error ("MSBuild not found. Install Visual Studio 2026 with 'Desktop development with C++' " +
                     "and CUDA Toolkit 13.3 - see docs/BUILD_GUIDE.md at the repo root.")
    }

    Write-Host "[run_demo] using MSBuild: $msbuild"
    # /m parallel build; /v:minimal keeps the log readable; the CUDA 13.3
    # MSBuild integration (imported by the .vcxproj) drives nvcc for the .cu files.
    & $msbuild $Sln /p:Configuration=Release /p:Platform=x64 /m /v:minimal /nologo
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[run_demo] BUILD FAILED (MSBuild exit code $LASTEXITCODE)" -ForegroundColor Red
        exit 1
    }
    if (-not (Test-Path $Exe)) {
        Write-Host "[run_demo] build reported success but $Exe is missing - check OutDir in the .vcxproj" -ForegroundColor Red
        exit 1
    }
}

# ---- 2) Run the demo executable (no arguments: the checked output depends
#         on the default problem size baked into expected_output.txt) --------
Write-Host "[run_demo] running: $Exe"
Write-Host '--------------------------------------------------------------------'
# Capture stdout+stderr line by line (as strings) AND echo them for the user.
$output = & $Exe 2>&1 | ForEach-Object { "$_" }
$exeExit = $LASTEXITCODE
$output | ForEach-Object { Write-Host $_ }
Write-Host '--------------------------------------------------------------------'

# The program itself gates on its GPU-vs-CPU verification (nonzero exit on
# mismatch) - respect that before even looking at the text.
if ($exeExit -ne 0) {
    Write-Host "[run_demo] DEMO FAILED: executable exited with code $exeExit" -ForegroundColor Red
    exit $exeExit
}

# ---- 3) Diff the stable lines against expected_output.txt ------------------
if (-not (Test-Path $Expected)) {
    Write-Host "[run_demo] WARNING: expected_output.txt missing - ran OK but nothing was verified" -ForegroundColor Yellow
    exit 0
}

# Keep only checkable expected lines: drop blanks and '#' comment lines
# (the TODO(scaffold) marker line in the template is such a comment).
$expectedLines = @(Get-Content $Expected |
    Where-Object { $_.Trim() -ne '' -and -not $_.TrimStart().StartsWith('#') } |
    ForEach-Object { $_.TrimEnd() })

if ($expectedLines.Count -eq 0) {
    Write-Host "[run_demo] WARNING: expected_output.txt has no checkable lines - ran OK but nothing was verified" -ForegroundColor Yellow
    exit 0
}

# Normalize actual output the same way (trailing whitespace / CR trimmed).
$actualLines = @($output | ForEach-Object { $_.TrimEnd() })

$missing = @($expectedLines | Where-Object { $actualLines -notcontains $_ })

if ($missing.Count -gt 0) {
    Write-Host "[run_demo] DEMO FAILED: expected line(s) not found in output:" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "  expected: $_" -ForegroundColor Red }
    Write-Host "[run_demo] If you changed the program's stable output on purpose, update demo/expected_output.txt in the same change."
    exit 1
}

Write-Host "[run_demo] DEMO PASSED: all $($expectedLines.Count) expected line(s) found (exit 0)" -ForegroundColor Green
exit 0
