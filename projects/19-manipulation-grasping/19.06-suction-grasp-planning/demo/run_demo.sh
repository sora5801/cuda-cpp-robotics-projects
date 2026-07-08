#!/usr/bin/env bash
# ===========================================================================
# run_demo.sh — one-command demo for 19.06 (Suction grasp planning: normals + seal-quality evaluation)
# Linux/macOS twin of run_demo.ps1, using the OPTIONAL CMake build path.
#
# Status: BEST-EFFORT bonus for Linux learners (CLAUDE.md paragraph 5) — the
# Visual Studio solution + run_demo.ps1 is the required, verified deliverable.
# Requires: cmake >= 3.24, a CUDA 13.x toolkit, and a host C++ compiler.
#
# Same diffing contract as run_demo.ps1: '#'-prefixed and blank lines in
# expected_output.txt are ignored; every remaining line must appear verbatim
# in the program's output (subset check - timing/info lines are not checked).
# TODO(scaffold): keep expected_output.txt in sync with the real project's
# stable output lines (see the "output contract" comment in ../src/main.cu).
# ===========================================================================
set -euo pipefail

# ---- Paths (relative to this script, not the caller's CWD) ----------------
DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$DEMO_DIR")"
SLUG='suction-grasp-planning'
BUILD_DIR="$PROJECT_DIR/build-cmake"      # kept apart from build/ (the VS tree); git-ignored
EXPECTED="$DEMO_DIR/expected_output.txt"

# ---- 1) Configure + build via CMake (idempotent; cheap when up to date) ----
echo "[run_demo] configuring + building with CMake into $BUILD_DIR ..."
cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --config Release -j

# Single-config generators (Makefiles/Ninja) put the exe at the build root;
# multi-config generators put it under Release/. Try both.
EXE="$BUILD_DIR/$SLUG"
[ -x "$EXE" ] || EXE="$BUILD_DIR/Release/$SLUG"
if [ ! -x "$EXE" ]; then
    echo "[run_demo] DEMO FAILED: built executable not found under $BUILD_DIR" >&2
    exit 1
fi

# ---- 2) Run (no arguments: expected_output.txt assumes the default size) ---
echo "[run_demo] running: $EXE"
echo '--------------------------------------------------------------------'
# Capture output while still showing it; preserve the exe's exit code
# (the program exits nonzero when its GPU-vs-CPU verification fails).
set +e
OUTPUT="$("$EXE" 2>&1)"
EXE_EXIT=$?
set -e
printf '%s\n' "$OUTPUT"
echo '--------------------------------------------------------------------'
if [ "$EXE_EXIT" -ne 0 ]; then
    echo "[run_demo] DEMO FAILED: executable exited with code $EXE_EXIT" >&2
    exit "$EXE_EXIT"
fi

# ---- 3) Diff stable lines against expected_output.txt ----------------------
if [ ! -f "$EXPECTED" ]; then
    echo "[run_demo] WARNING: expected_output.txt missing - ran OK but nothing was verified"
    exit 0
fi

CHECKED=0
MISSING=0
# Read expected lines; skip blanks and '#' comments; require each remaining
# line to appear verbatim in the output (grep -F fixed-string, -x whole-line).
while IFS= read -r line || [ -n "$line" ]; do
    trimmed="$(printf '%s' "$line" | sed -e 's/[[:space:]]*$//')"
    case "$trimmed" in
        ''|'#'*) continue ;;   # blank or comment: not checked
    esac
    CHECKED=$((CHECKED + 1))
    if ! printf '%s\n' "$OUTPUT" | sed -e 's/[[:space:]]*$//' | grep -Fxq -- "$trimmed"; then
        echo "[run_demo] expected line not found: $trimmed" >&2
        MISSING=$((MISSING + 1))
    fi
done < "$EXPECTED"

if [ "$CHECKED" -eq 0 ]; then
    echo "[run_demo] WARNING: expected_output.txt has no checkable lines - ran OK but nothing was verified"
    exit 0
fi
if [ "$MISSING" -gt 0 ]; then
    echo "[run_demo] DEMO FAILED: $MISSING expected line(s) missing (update expected_output.txt if the change was intentional)" >&2
    exit 1
fi
echo "[run_demo] DEMO PASSED: all $CHECKED expected line(s) found (exit 0)"
exit 0
