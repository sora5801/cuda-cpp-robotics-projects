#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# tools/new_pushnote.py — generate a dated push-note stub
#
# Role in the repo (see CLAUDE.md §7.1, mandatory and load-bearing):
#   EVERY push to origin/main ships with a didactic note in push-notes/
#   explaining what was added and how to study it — written BEFORE the push
#   and included IN it, so the repo always explains its own latest state.
#   This script creates the correctly-named stub with the eight required
#   sections so no push-note ever forgets one.
#
#   Filename convention: push-notes/YYYY-MM-DD-NN-<short-title>.md where
#   NN is that day's zero-padded push counter — the FIRST note of a day is
#   00, so NN is simply "how many notes dated today already exist".
#
# Usage:
#   python tools/new_pushnote.py phase 1 flagships batch 1
#     -> push-notes/2026-07-08-00-phase-1-flagships-batch-1.md
# ---------------------------------------------------------------------------

import argparse
import re
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PUSH_NOTES_DIR = REPO_ROOT / "push-notes"

# The eight required sections of CLAUDE.md §7.1, in contract order, each
# with a one-line reminder of what belongs under it. The reminders carry a
# "TODO(push-note):" marker (never a bare TODO — repo convention) so an
# unfinished note is grep-ably obvious before it gets committed.
SECTIONS = [
    ("Summary",
     "One paragraph: what this push adds and why it matters to the learner."),
    ("What changed",
     "New/edited projects and files, grouped and linked (relative paths)."),
    ("New projects (didactic blurbs)",
     "Per new project, 3-5 sentences: the concept taught, the CUDA pattern, "
     "where it sits in the robot (one line), and the single most "
     "interesting thing to look at."),
    ("How to build & run",
     "Exact commands to build and run the new material (VS solution + "
     "demo/run_demo)."),
    ("What to study here",
     "A suggested reading path through the new material, plus 1-2 exercises."),
    ("Verification",
     "What was checked: build passed? demo matched expected_output.txt? on "
     "what GPU/arch? (never claim runs that did not happen)."),
    ("Known limitations / TODOs",
     "Honest notes: what is simplified, deferred, or still red."),
    ("Next push preview",
     "One or two lines on what is planned next."),
]


def kebab(words: list[str]) -> str:
    """Join the title words into a filesystem-safe kebab short-title.

    Lowercase ASCII alphanumerics and hyphens only — the title becomes part
    of a filename that must sort and link cleanly on every platform.
    """
    joined = " ".join(words).lower()
    parts = [p for p in re.split(r"[^a-z0-9]+", joined) if p]
    return "-".join(parts)


def main() -> int:
    # Windows-console UTF-8 fix (same as the sibling tools) — paths and
    # reminders may contain characters legacy codepages cannot print.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(
        description="Create a dated push-note stub with the eight required "
                    "CLAUDE.md §7.1 sections. Write the note BEFORE pushing "
                    "and include it IN the push.")
    ap.add_argument("title", nargs="+",
                    help="short title words (joined to a kebab short-title)")
    args = ap.parse_args()

    short_title = kebab(args.title)
    if not short_title:
        print("ERROR: title reduced to nothing after cleaning — give at "
              "least one alphanumeric word.", file=sys.stderr)
        return 1

    today = date.today().isoformat()
    PUSH_NOTES_DIR.mkdir(parents=True, exist_ok=True)

    # NN = zero-padded count of push-notes ALREADY dated today: the first
    # note of the day is 00, the second 01, and so on. Counting existing
    # files (rather than keeping a counter elsewhere) makes the scheme
    # self-healing and needs no state beyond the folder itself.
    nn = len(list(PUSH_NOTES_DIR.glob(f"{today}-*.md")))
    path = PUSH_NOTES_DIR / f"{today}-{nn:02d}-{short_title}.md"
    if path.exists():
        # Only reachable if today's numbering was disturbed by hand (e.g. a
        # note was deleted, shifting the count onto an existing NN).
        print(f"ERROR: {path} already exists — resolve the numbering by "
              f"hand (notes must never be overwritten).", file=sys.stderr)
        return 1

    title_text = " ".join(args.title)
    body = [
        f"# Push note — {today}-{nn:02d}: {title_text}",
        "",
        "> Push-note per CLAUDE.md §7.1 — written **before** the push and "
        "included **in** it, so the",
        "> repository always explains its own latest state. After filling "
        "this in, prepend a one-line",
        "> entry linking this file to the root `CHANGELOG.md`, then push.",
        "",
    ]
    for heading, reminder in SECTIONS:
        body += [
            f"## {heading}",
            "",
            f"TODO(push-note): {reminder}",
            "",
        ]
    path.write_text("\n".join(body).rstrip() + "\n", encoding="utf-8")

    print(f"created {path.relative_to(REPO_ROOT).as_posix()}")
    print("Reminder: prepend a one-line entry linking this push-note to "
          "the root CHANGELOG.md (newest first), and fill in every "
          "'TODO(push-note):' before pushing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
