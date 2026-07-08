#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# tools/verify_project.py — check projects against the Definition of Done
#
# Role in the repo (see CLAUDE.md §9):
#   A project may only be marked "done" when every gate in §9 passes. Some
#   gates need a human or a compiler (VS build, demo run, spot-reading the
#   comments for actual pedagogy); THIS script automates the STRUCTURAL
#   gates — files present, canonical headings present, no scaffold TODOs
#   left, comment-density floor, expected output non-trivial. Workers run
#   it before handing a branch to the lead; the lead runs it again (often
#   with --all) before merging and pushing (CLAUDE.md §10).
#
#   Passing this script is NECESSARY, not SUFFICIENT: it cannot tell a
#   brilliant explanation from filler. It keeps the floor honest; humans
#   keep the ceiling.
#
# Usage:
#   python tools/verify_project.py projects/33-foundational-libraries/33.01-batched-small-matrix-linalg
#   python tools/verify_project.py <path> <path> ...
#   python tools/verify_project.py --all       # every projects/*/* folder
#
# Exit code: 0 only if every checked project passes every gate.
# ---------------------------------------------------------------------------

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Canonical headings — these EXACT strings are the contract shared by
# docs/PROJECT_TEMPLATE/, tools/scaffold.py, and this checker. If one is
# ever changed, all three must change in lockstep (CLAUDE.md §4).
# ---------------------------------------------------------------------------

# README.md: the title line is section 1 of 13 — matched by pattern below,
# not by exact string, because it embeds the project's own ID and name.
README_TITLE_RE = re.compile(r"^# \d\d\.\d\d — ")
README_HEADINGS = [
    "## Overview",
    "## What this computes & why the GPU helps",
    "## System context — where this sits in a robot",
    "## The algorithm in brief",
    "## Build",
    "## Run the demo",
    "## Data",
    "## Expected output",
    "## Code tour",
    "## Prior art & further reading",
    "## Exercises",
    "## Limitations & honesty",
]

THEORY_HEADINGS = [
    "## The problem — physics & engineering first",
    "## The math",
    "## The algorithm",
    "## The GPU mapping",
    "## Numerical considerations",
    "## How we verify correctness",
    "## Where this sits in the real world",
]

PRACTICE_HEADINGS = [
    "## 1. Building it — construction of the robot/part",
    "## 2. Real hardware — chips, parts, illustrative BOM",
    "## 3. Installation & integration — putting it on a real robot",
    "## 4. Business & regulatory context",
]

# The unfinished-work marker stamped by the template. ANY survivor in these
# learner-facing files means the project is not done (worker brief / §9).
SCAFFOLD_MARKER = "TODO(scaffold):"
MARKER_FILES = [
    "README.md", "THEORY.md", "PRACTICE.md",
    "data/README.md", "demo/README.md", "demo/expected_output.txt",
]

# Files and directories every project must have (CLAUDE.md §4 layout).
# ("dir_nonempty" = the directory must exist AND contain at least one file,
# recursively — an empty src/util/ or data/sample/ teaches nothing.)
REQUIRED_FILES = [
    "README.md", "THEORY.md", "PRACTICE.md", ".gitignore",
    "src/main.cu", "src/kernels.cu", "src/kernels.cuh",
    "src/reference_cpu.cpp",
    "data/README.md",
    "scripts/make_synthetic.py",
    "demo/run_demo.ps1", "demo/expected_output.txt", "demo/README.md",
]
REQUIRED_NONEMPTY_DIRS = ["src/util", "data/sample"]
REQUIRED_GLOBS = ["build/*.sln", "build/*.vcxproj"]

# Comment-density floor over src/ (CLAUDE.md §6.2): ~0.4 non-trivial
# comment lines per code line. A SAFETY NET, not the goal — the goal is a
# ratio a stranger could learn from, often >= 1:1 in kernel files.
DENSITY_FLOOR = 0.4
SRC_EXTENSIONS = {".cu", ".cuh", ".cpp", ".h"}


def read_text(path: Path) -> str:
    """Read a UTF-8 text file tolerantly (BOM ok, stray bytes replaced).

    Verification must never crash on a worker's odd editor settings — a
    mojibake character should at worst fail a heading match, visibly.
    """
    return path.read_text(encoding="utf-8-sig", errors="replace")


def missing_headings(text: str, headings: list[str]) -> list[str]:
    """Return the canonical headings NOT present as whole lines in `text`.

    We match whole stripped lines, not substrings, so a heading quoted in
    prose ("see the ## Build section") can never satisfy the gate.
    """
    lines = {line.strip() for line in text.splitlines()}
    return [h for h in headings if h not in lines]


def comment_density(src_dir: Path) -> tuple[int, int, float]:
    """Measure (comment_lines, code_lines, ratio) over src/ recursively.

    The heuristic — honest about what it is:
      * A line counts as a COMMENT line if its stripped form starts with
        '//' or it sits inside a '/* ... */' block. Mixed lines ("int x;
        // meters") count as CODE — this UNDERCOUNTS comments, which is the
        safe direction for a floor.
      * A comment line only COUNTS if it carries >= 3 alphanumeric chars,
        filtering pure decoration like '// ----------' that teaches nothing.
      * A CODE line is any non-blank line that is not a comment line.
      * Block-comment tracking is line-based: '/*' opening mid-line marks
        the FOLLOWING lines as comments until a line containing '*/'; the
        opening line itself stays code if code precedes the '/*'. String
        literals containing '/*' can fool it — acceptable for a heuristic
        whose only job is to catch grossly under-commented files.

    This is deliberately a FLOOR, not the goal (§6.2): passing it proves
    nothing about teaching quality; failing it proves the file is bare.
    """
    comment_lines = 0
    code_lines = 0
    for path in sorted(src_dir.rglob("*")):
        if not (path.is_file() and path.suffix in SRC_EXTENSIONS):
            continue
        in_block = False  # are we inside a /* ... */ block right now?
        for line in read_text(path).splitlines():
            stripped = line.strip()
            if not stripped:
                continue  # blank lines count as neither
            if in_block:
                # Everything until '*/' is comment; note where the block ends.
                is_comment = True
                if "*/" in stripped:
                    in_block = False
            elif stripped.startswith("//"):
                is_comment = True
            elif stripped.startswith("/*"):
                # A line that IS the start of a block comment.
                is_comment = True
                if "*/" not in stripped:
                    in_block = True
            else:
                # Code line — but it may OPEN a block comment mid-line
                # ("int x; /* layout: ..." ). The line itself stays code.
                is_comment = False
                if "/*" in stripped and "*/" not in stripped.split("/*", 1)[1]:
                    in_block = True
            if is_comment:
                # Non-trivial only: demand some actual words, not dashes.
                if sum(c.isalnum() for c in stripped) >= 3:
                    comment_lines += 1
            else:
                code_lines += 1
    ratio = (comment_lines / code_lines) if code_lines else 0.0
    return comment_lines, code_lines, ratio


def check_project(proj: Path) -> tuple[bool, list[tuple[str, bool, list[str]]]]:
    """Run every structural gate on one project folder.

    Returns (all_passed, gates) where each gate is
    (label, passed, detail_lines) — detail lines explain a failure well
    enough that the worker knows exactly what to fix.
    """
    gates: list[tuple[str, bool, list[str]]] = []

    # -- Gate a: required files, non-empty dirs, and build globs -----------
    detail: list[str] = []
    for rel in REQUIRED_FILES:
        if not (proj / rel).is_file():
            detail.append(f"missing file: {rel}")
    for rel in REQUIRED_NONEMPTY_DIRS:
        d = proj / rel
        if not d.is_dir():
            detail.append(f"missing directory: {rel}/")
        elif not any(p.is_file() for p in d.rglob("*")):
            detail.append(f"directory is empty: {rel}/")
    for pattern in REQUIRED_GLOBS:
        if not list(proj.glob(pattern)):
            detail.append(f"no file matches: {pattern}")
    gates.append(("required files & dirs (§4 layout)", not detail, detail))

    # -- Gate b: README canonical sections (all 13) -------------------------
    detail = []
    readme = proj / "README.md"
    if readme.is_file():
        text = read_text(readme)
        if not any(README_TITLE_RE.match(line.strip())
                   for line in text.splitlines()):
            detail.append('missing title line: "# SS.NN — <name>"')
        detail += [f"missing heading: {h!r}"
                   for h in missing_headings(text, README_HEADINGS)]
    else:
        detail.append("README.md not found")
    gates.append(("README.md — 13 canonical sections (§4.1)",
                  not detail, detail))

    # -- Gate c: THEORY (7 headings) and PRACTICE (4 headings) --------------
    for fname, headings, label in [
            ("THEORY.md", THEORY_HEADINGS, "THEORY.md — 7 sections (§4.2)"),
            ("PRACTICE.md", PRACTICE_HEADINGS, "PRACTICE.md — 4 sections (§4.3)")]:
        detail = []
        f = proj / fname
        if f.is_file():
            detail = [f"missing heading: {h!r}"
                      for h in missing_headings(read_text(f), headings)]
        else:
            detail = [f"{fname} not found"]
        gates.append((label, not detail, detail))

    # -- Gate d: no scaffold markers left in learner-facing files -----------
    detail = []
    for rel in MARKER_FILES:
        f = proj / rel
        if f.is_file():
            n = read_text(f).count(SCAFFOLD_MARKER)
            if n:
                detail.append(f"{rel}: {n} '{SCAFFOLD_MARKER}' marker(s) remain")
        # A missing file is gate a's failure, not repeated here.
    gates.append((f"no {SCAFFOLD_MARKER!r} markers remain", not detail, detail))

    # -- Gate e: comment density over src/ -----------------------------------
    detail = []
    src = proj / "src"
    if src.is_dir():
        comments, code, ratio = comment_density(src)
        if code == 0:
            detail.append("src/ contains no code lines at all")
        elif ratio < DENSITY_FLOOR:
            detail.append(
                f"density {ratio:.2f} ({comments} comment / {code} code "
                f"lines) is below the {DENSITY_FLOOR} floor — and the floor "
                f"is a safety net, not the goal (§6.2)")
    else:
        detail.append("src/ not found")
    gates.append((f"comment density >= {DENSITY_FLOOR} in src/ (§6.2 floor)",
                  not detail, detail))

    # -- Gate f: expected_output.txt has real content ------------------------
    detail = []
    exp = proj / "demo" / "expected_output.txt"
    if exp.is_file():
        # Lines starting with '#' are annotations about the output, not the
        # output itself — an all-comment file promises the learner nothing.
        payload = [ln for ln in read_text(exp).splitlines()
                   if ln.strip() and not ln.lstrip().startswith("#")]
        if not payload:
            detail.append("expected_output.txt has no non-comment content")
    else:
        detail.append("demo/expected_output.txt not found")
    gates.append(("expected_output.txt non-empty (ignoring # lines)",
                  not detail, detail))

    return all(ok for _, ok, _ in gates), gates


def print_report(proj: Path, passed: bool,
                 gates: list[tuple[str, bool, list[str]]]) -> None:
    """Friendly aligned checklist for one project."""
    try:
        shown = proj.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        shown = str(proj)  # project outside the repo — show as given
    print(f"\n== {shown} ==")
    width = max(len(label) for label, _, _ in gates)
    for label, ok, detail in gates:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:<{width}}")
        for line in detail:
            print(f"         - {line}")
    n_fail = sum(1 for _, ok, _ in gates if not ok)
    print(f"  RESULT: {'PASS' if passed else f'FAIL ({n_fail} gate(s) failed)'}")


def main() -> int:
    # Same Windows-console UTF-8 fix as catalog.py — headings contain '—'
    # and '★' which legacy codepages cannot print.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(
        description="Check project folders against the structural gates of "
                    "the Definition of Done (CLAUDE.md §9). Necessary, not "
                    "sufficient: builds, demo runs, and comment QUALITY "
                    "remain human/compiler checks.")
    ap.add_argument("paths", nargs="*", metavar="PROJECT_DIR",
                    help="project folder(s) to check")
    ap.add_argument("--all", action="store_true",
                    help="check every projects/*/* folder in the repo")
    args = ap.parse_args()

    if args.all:
        projects_root = REPO_ROOT / "projects"
        # projects/<domain-slug>/<SS.NN-slug>/ — two levels down, dirs only.
        targets = sorted(p for p in projects_root.glob("*/*")
                         if p.is_dir()) if projects_root.is_dir() else []
    else:
        targets = [Path(p) for p in args.paths]

    if not targets:
        print("No project folders to check. Give paths, or --all "
              "(after tools/scaffold.py has stamped projects/).",
              file=sys.stderr)
        return 1

    n_pass = 0
    for proj in targets:
        if not proj.is_dir():
            print(f"\n== {proj} ==\n  ERROR: not a directory")
            continue
        passed, gates = check_project(proj)
        print_report(proj, passed, gates)
        n_pass += passed

    print(f"\nverify: {n_pass}/{len(targets)} project(s) pass all "
          f"structural gates")
    return 0 if n_pass == len(targets) else 1


if __name__ == "__main__":
    sys.exit(main())
