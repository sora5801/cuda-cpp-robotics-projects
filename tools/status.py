#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# tools/status.py — (re)generate docs/STATUS.md, the work-queue dashboard
#
# Role in the repo (see CLAUDE.md §10 "Claiming protocol"):
#   docs/STATUS.md is the shared work queue. Workers claim a project by
#   editing ONLY their own row (Status → in-progress, Owner → their name)
#   on their branch; the lead flips rows to done at merge time and resolves
#   the rare claim race. This script rebuilds the dashboard's STRUCTURE
#   from catalog.json (so new projects appear, counts stay honest) while
#   CARRYING FORWARD the human-edited state.
#
#   THE LOAD-BEARING RULE: regeneration must NEVER lose a claim. If
#   docs/STATUS.md already exists we parse its table rows by project ID and
#   preserve the Status / Owner / Notes cells verbatim; only projects with
#   no existing row default to `todo`. Everything else in the file
#   (headers, summaries, per-domain counts) is regenerated and should never
#   be hand-edited.
#
# Usage:
#   python tools/status.py           # rewrite docs/STATUS.md
#   python tools/status.py --check   # print the summary; write nothing
# ---------------------------------------------------------------------------

import argparse
import json
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_JSON = REPO_ROOT / "catalog.json"
STATUS_MD = REPO_ROOT / "docs" / "STATUS.md"

# The four ratified statuses (CLAUDE.md §10). Anything else found in an
# existing row is preserved verbatim (never destroy a worker's edit) but
# reported so the lead can fix the typo.
KNOWN_STATUSES = ("todo", "in-progress", "done", "blocked")

# ---------------------------------------------------------------------------
# Phase 1 flagships — hardcoded from the CLAUDE.md §11 flagship table (one
# polished project per domain, built first so every domain gets a
# best-in-class exemplar early). §11 says these IDs are EXPECTED positions:
# if a flagship proves intractable, the lead may swap it for a sibling in
# the same domain — update this set AND CLAUDE.md §11 together.
# ---------------------------------------------------------------------------
FLAGSHIP_IDS = {
    "01.02", "02.06", "03.01", "04.01", "05.01", "06.05", "07.09", "08.01",
    "09.01", "10.03", "11.01", "12.01", "13.03", "14.02", "15.01", "16.01",
    "17.01", "18.01", "19.01", "20.01", "21.04", "22.01", "23.01", "24.01",
    "25.01", "26.01", "27.04", "28.01", "29.05", "30.01", "31.01", "32.02",
    "33.01", "34.03", "35.01", "36.03",
}


def badge(tags: dict) -> str:
    """Difficulty cell text — mirrors scaffold.py's badge rendering."""
    star, rnd = tags.get("star", False), tags.get("rnd", False)
    if star and rnd:
        return "★ beginner · [R&D]"
    if star:
        return "★ beginner"
    if rnd:
        return "[R&D] research"
    return "intermediate"


def md_cell(text: str) -> str:
    """Escape '|' so free-text cells (project names, notes) can never break
    the markdown table they live in — the parser below splits on '|'."""
    return text.replace("|", "\\|")


def parse_existing(path: Path) -> dict[str, dict[str, str]]:
    """Extract {project_id: {status, owner, notes}} from an existing
    STATUS.md — the carry-forward half of the load-bearing rule.

    We recognize a project row purely by shape: a markdown table row whose
    first cell looks like an SS.NN ID. Column POSITIONS are the contract
    (| ID | Project | Difficulty | Flagship | Status | Owner | Notes |);
    we index from the ends so a Notes cell containing an escaped pipe is
    still recovered whole.
    """
    state: dict[str, dict[str, str]] = {}
    if not path.is_file():
        return state  # first run: everything defaults to todo
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        # Split on UNESCAPED pipes only, then un-escape inside each cell.
        cells = [c.strip().replace("\\|", "|")
                 for c in line.strip("|").replace("\\|", "\x00").split("|")]
        cells = [c.replace("\x00", "|") for c in cells]
        if len(cells) < 7:
            continue  # header, separator, or some other table
        pid = cells[0]
        if not (len(pid) == 5 and pid[:2].isdigit()
                and pid[2] == "." and pid[3:].isdigit()):
            continue  # not a project row (e.g. the per-domain count table)
        state[pid] = {
            "status": cells[4] or "todo",
            "owner": cells[5],
            "notes": "|".join(cells[6:]),  # notes is the LAST column
        }
    return state


def build_status_md(catalog: dict,
                    prior: dict[str, dict[str, str]]) -> tuple[str, dict]:
    """Render the full STATUS.md text. Returns (markdown, stats) where
    stats feeds the console summary (--check prints it without writing)."""
    # ---- gather per-project rows, merging prior state --------------------
    counts: dict[str, int] = {}          # status value → count
    unknown_statuses: list[str] = []     # typos to surface to the lead
    domain_done: dict[int, tuple[int, int]] = {}  # number → (done, total)
    lines: list[str] = []

    # ---- header: what this file is and how to use it ---------------------
    lines += [
        "# STATUS — work queue & progress dashboard",
        "",
        f"> Generated by `tools/status.py` on {date.today().isoformat()} "
        f"from `catalog.json`.",
        "> **Hand-edit ONLY the Status / Owner / Notes cells** of your own "
        "project row — regeneration",
        "> rebuilds everything else but carries those three cells forward "
        "by project ID.",
        "",
        "## Claiming protocol (CLAUDE.md §10)",
        "",
        "- **Workers:** claim the highest-priority `todo` project by "
        "setting its Status to `in-progress`",
        "  and putting your agent name in Owner — editing **only your own "
        "row**, on your own",
        "  `proj/<SS.NN>-<slug>` branch. Build to the Definition of Done "
        "(CLAUDE.md §9), run",
        "  `tools/verify_project.py`, hand back to the lead. Never push to "
        "`main` directly.",
        "- **Lead:** merges green branches, flips rows to `done`, resolves "
        "claim races, regenerates",
        "  this file, writes the push-note.",
        "- **Statuses:** `todo` | `in-progress` | `done` | `blocked` "
        "(blocked rows explain why in Notes).",
        "- **Priority (CLAUDE.md §11):** during Phase 1, flagships (⭐ "
        "column) come first — one polished",
        "  project per domain. Within a domain thereafter: ★ beginner "
        "first, then untagged intermediate,",
        "  then [R&D] last; ties break by ID.",
        "",
    ]

    # ---- per-domain sections (built first so counts exist for summary) ----
    domain_sections: list[str] = []
    for dom in catalog["domains"]:
        done_in_dom = 0
        rows = []
        for p in dom["projects"]:
            prev = prior.get(p["id"], {})
            status = prev.get("status", "todo")
            owner = prev.get("owner", "")
            notes = prev.get("notes", "")
            if status not in KNOWN_STATUSES:
                unknown_statuses.append(f"{p['id']}: '{status}'")
            counts[status] = counts.get(status, 0) + 1
            done_in_dom += (status == "done")
            rows.append(
                f"| {p['id']} | {md_cell(p['name'])} | {badge(p['tags'])} "
                f"| {'⭐' if p['id'] in FLAGSHIP_IDS else ''} "
                f"| {status} | {md_cell(owner)} | {md_cell(notes)} |")
        domain_done[dom["number"]] = (done_in_dom, len(dom["projects"]))
        domain_sections += [
            f"## {dom['number']}. {dom['title']} (`{dom['slug']}`) — "
            f"{done_in_dom}/{len(dom['projects'])} done",
            "",
            "| ID | Project | Difficulty | Flagship | Status | Owner | Notes |",
            "|----|---------|------------|----------|--------|-------|-------|",
            *rows,
            "",
        ]

    total = catalog["project_count"]
    n_done = counts.get("done", 0)

    # ---- summary block -----------------------------------------------------
    lines += [
        "## Summary",
        "",
        f"**Overall: {n_done}/{total} done** "
        f"({n_done / total:.1%})" if total else "**Overall: 0/0**",
        "",
        "| Status | Count |",
        "|--------|-------|",
    ]
    # Ratified statuses first (stable order), then any stragglers found.
    for s in KNOWN_STATUSES:
        lines.append(f"| {s} | {counts.get(s, 0)} |")
    for s in sorted(set(counts) - set(KNOWN_STATUSES)):
        lines.append(f"| {md_cell(s)} (unknown) | {counts[s]} |")
    lines += [
        "",
        "| Domain | Done / Total |",
        "|--------|--------------|",
    ]
    for dom in catalog["domains"]:
        d, t = domain_done[dom["number"]]
        lines.append(f"| {dom['number']:02d} {md_cell(dom['title'])} | {d}/{t} |")
    lines += [""]

    lines += domain_sections
    stats = {"counts": counts, "done": n_done, "total": total,
             "domain_done": domain_done,
             "unknown_statuses": unknown_statuses,
             "carried": len(prior)}
    return "\n".join(lines).rstrip() + "\n", stats


def main() -> int:
    # Windows-console UTF-8 fix (same as catalog.py) — this report prints
    # '★' and '⭐' which legacy codepages cannot encode.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(
        description="(Re)generate docs/STATUS.md from catalog.json, "
                    "carrying forward the human-edited Status/Owner/Notes "
                    "cells so regeneration never loses a claim.")
    ap.add_argument("--check", action="store_true",
                    help="print the summary; do not write docs/STATUS.md")
    args = ap.parse_args()

    if not CATALOG_JSON.exists():
        print(f"ERROR: {CATALOG_JSON} not found — run tools/catalog.py first.",
              file=sys.stderr)
        return 1
    catalog = json.loads(CATALOG_JSON.read_text(encoding="utf-8"))

    prior = parse_existing(STATUS_MD)
    text, stats = build_status_md(catalog, prior)

    # Console summary — identical in --check and write modes, so a worker
    # can sanity-check the queue without touching the file.
    print(f"status: {stats['done']}/{stats['total']} done")
    for s in KNOWN_STATUSES:
        print(f"  {s:<12} {stats['counts'].get(s, 0)}")
    print(f"  carried forward {stats['carried']} existing row(s) "
          f"from docs/STATUS.md" if stats["carried"]
          else "  no existing docs/STATUS.md rows — all projects start 'todo'")
    if stats["unknown_statuses"]:
        # Preserved verbatim in the file (never destroy an edit), but the
        # lead should normalize these to one of the four ratified values.
        print("  WARNING: unknown status values preserved verbatim:")
        for u in stats["unknown_statuses"]:
            print(f"    - {u}")

    if args.check:
        print("  (--check: nothing written)")
        return 0

    STATUS_MD.parent.mkdir(parents=True, exist_ok=True)
    STATUS_MD.write_text(text, encoding="utf-8")
    print(f"  wrote {STATUS_MD.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
