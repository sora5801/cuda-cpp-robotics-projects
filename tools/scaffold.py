#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# tools/scaffold.py — stamp out project skeletons from the catalog
#
# Role in the repo (see CLAUDE.md §2, §3, §4, §11 Phase 0):
#   catalog.json (produced by tools/catalog.py — read THAT first) lists every
#   project with its deterministic ID, slug, and destination folder. This
#   script copies the canonical empty project, docs/PROJECT_TEMPLATE/, into
#   each destination, substituting the {{TOKEN}} placeholders so every stub
#   README/THEORY/PRACTICE/.vcxproj arrives pre-filled with the project's
#   identity. Workers then take a scaffolded folder to the Definition of
#   Done (CLAUDE.md §9); tools/verify_project.py checks their work.
#
#   The three tools form a pipeline:
#       catalog.py  →  catalog.json  →  scaffold.py  →  projects/…  (this)
#                                    →  status.py    →  docs/STATUS.md
#
# The cardinal safety rule (CLAUDE.md §10: one agent owns one folder):
#   scaffolding is IDEMPOTENT and NON-DESTRUCTIVE. If a destination folder
#   exists AT ALL — even half-built, even empty — we skip it entirely.
#   A worker's in-progress project must never be overwritten by a re-run.
#
# Usage:
#   python tools/scaffold.py                    # stamp every missing project
#   python tools/scaffold.py --only 33.01       # just one (or a comma list)
#   python tools/scaffold.py --dry-run          # report, write nothing
# ---------------------------------------------------------------------------

import argparse
import json
import sys
import uuid
from datetime import date
from pathlib import Path

# Resolve the repo root from this file's own location (tools/ is one level
# below the root) so the script works from any current directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_JSON = REPO_ROOT / "catalog.json"
TEMPLATE_DIR = REPO_ROOT / "docs" / "PROJECT_TEMPLATE"

# ---------------------------------------------------------------------------
# {{BUNDLED_NOTE}} — inserted only for "bundled" catalog bullets.
#
# CLAUDE.md §2: some catalog bullets name several related ideas separated by
# '·' or ';'. They stay ONE project; the named components become milestones
# or sub-demos inside it. The scaffolded README must say so up front, so the
# worker who claims the project scopes it correctly from the start.
# ---------------------------------------------------------------------------
BUNDLED_NOTE = (
    "> **Bundled project.** This catalog bullet names several related "
    "components; per the repository contract (CLAUDE.md §2) they are ONE "
    "project, and the named components become milestones / sub-demos inside "
    "it. This README must state which components are implemented and which "
    "are documented-only, and THEORY.md must cover the ideas shared across "
    "the bundle.\n"
)


def difficulty_badge(tags: dict) -> str:
    """Render the human-facing difficulty badge from the catalog tags.

    The badge is what the reader sees at the top of the README; the machine
    field ({{DIFFICULTY}}) stays the single word from catalog.json. A bullet
    can carry BOTH ★ and [R&D] (catalog.py surfaces those as anomalies);
    the badge then shows both so nothing about the bullet is hidden.
    """
    star = tags.get("star", False)
    rnd = tags.get("rnd", False)
    if star and rnd:
        return "★ beginner · [R&D]"
    if star:
        return "★ beginner"
    if rnd:
        return "[R&D] research"
    return "intermediate"


def project_guid(project_id: str) -> str:
    """Deterministic uppercase braced GUID for the Visual Studio project.

    Why uuid5 (name-based, SHA-1) and not uuid4 (random): re-running the
    scaffold — or regenerating a lost .vcxproj by hand — must always produce
    the SAME GUID for the same project ID, so solutions and project
    references never silently diverge between runs or machines. The
    namespace string is the repo name so no other repo's IDs collide.
    """
    u = uuid.uuid5(uuid.NAMESPACE_URL, "cuda-cpp-robotics-projects/" + project_id)
    return "{" + str(u).upper() + "}"


def build_tokens(project: dict, domain: dict) -> dict[str, str]:
    """Assemble the full {{TOKEN}} → value table for one project.

    This is the single place the token contract (see the PROJECT_TEMPLATE
    stubs and CLAUDE.md §4) is implemented. Every text file copied out of
    the template gets ALL of these substituted — a token that survives into
    a scaffolded file is a bug here or a typo in the template.
    """
    folder = Path(project["folder"]).name  # e.g. "08.01-mppi-controller-…"
    return {
        "{{PROJECT_ID}}": project["id"],
        "{{PROJECT_NAME}}": project["name"],
        "{{PROJECT_SLUG}}": project["slug"],
        "{{PROJECT_FOLDER}}": folder,
        "{{DOMAIN_SLUG}}": domain["slug"],
        # Human-facing number, NOT zero-padded ("8", not "08") — it appears
        # in prose like "Domain: 8. Control Systems".
        "{{DOMAIN_NUMBER}}": str(domain["number"]),
        "{{DOMAIN_TITLE}}": domain["title"],
        "{{DIFFICULTY}}": project["difficulty"],
        "{{DIFFICULTY_BADGE}}": difficulty_badge(project["tags"]),
        # The one-line summary IS the cleaned bullet name — catalog bullets
        # are already summary-shaped; workers expand it in the README.
        "{{SUMMARY}}": project["name"],
        "{{RAW_BULLET}}": project["raw"],
        "{{GUID}}": project_guid(project["id"]),
        "{{BUNDLED_NOTE}}": BUNDLED_NOTE if project.get("bundled") else "",
        # Dates the illustrative hardware/BOM content in PRACTICE.md stubs
        # (CLAUDE.md §4.3: part numbers and prices go stale — date them).
        "{{SCAFFOLD_DATE}}": date.today().isoformat(),
    }


def substitute(text: str, tokens: dict[str, str]) -> str:
    """Replace every {{TOKEN}} occurrence in `text`.

    Plain sequential str.replace is enough: token names never overlap and
    none of the VALUES contains another token, so order cannot matter.
    """
    for token, value in tokens.items():
        text = text.replace(token, value)
    return text


def stamp_project(project: dict, domain: dict, dry_run: bool) -> str:
    """Create one project skeleton. Returns 'created' | 'skipped'.

    NON-DESTRUCTIVE BY CONTRACT: if the destination exists at all we do not
    look inside it, we do not merge, we do not repair — we skip. The folder
    belongs to whichever worker claimed it (CLAUDE.md §10).
    """
    dest = REPO_ROOT / project["folder"]  # catalog "folder" used verbatim
    if dest.exists():
        return "skipped"
    if dry_run:
        return "created"  # counted as would-create; nothing is written

    tokens = build_tokens(project, domain)

    # Walk the template depth-first. sorted() gives parents before children
    # (a path always sorts before the paths inside it), so directories are
    # created before the files they contain.
    for src in sorted(TEMPLATE_DIR.rglob("*")):
        rel = src.relative_to(TEMPLATE_DIR)
        # File-rename rule (token contract): any path component whose NAME
        # contains "TEMPLATE" is renamed to the project slug — this is how
        # TEMPLATE.sln / TEMPLATE.vcxproj / TEMPLATE.vcxproj.filters become
        # <slug>.sln / <slug>.vcxproj / <slug>.vcxproj.filters.
        out = dest.joinpath(
            *(part.replace("TEMPLATE", project["slug"]) for part in rel.parts))
        if src.is_dir():
            # Recreate directories even when empty (e.g. data/sample/ may
            # hold only a .gitkeep, or nothing at all in the template).
            out.mkdir(parents=True, exist_ok=True)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            # All template files are text (contract). We read/write BYTES and
            # substitute on the decoded string so the template's own line
            # endings survive byte-for-byte — .sh stubs must keep LF on
            # Windows, and we refuse to let text-mode newline translation
            # decide that for us.
            content = src.read_bytes().decode("utf-8")
            out.write_bytes(substitute(content, tokens).encode("utf-8"))
    return "created"


def main() -> int:
    # Windows consoles default to a legacy codepage that cannot print '★';
    # force UTF-8 (same fix as catalog.py) so reports render everywhere.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(
        description="Stamp project skeletons from catalog.json + "
                    "docs/PROJECT_TEMPLATE/. Idempotent: existing project "
                    "folders are always skipped, never touched.")
    ap.add_argument("--only", metavar="ID[,ID,...]",
                    help="scaffold only these project IDs (e.g. 33.01,08.01)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what WOULD be created; write nothing")
    args = ap.parse_args()

    if not CATALOG_JSON.exists():
        print(f"ERROR: {CATALOG_JSON} not found — run tools/catalog.py first.",
              file=sys.stderr)
        return 1
    catalog = json.loads(CATALOG_JSON.read_text(encoding="utf-8"))

    # The template is only READ on a real run; a dry run can still report
    # the plan while the template is being authored — but say so loudly.
    if not TEMPLATE_DIR.exists():
        if args.dry_run:
            print(f"WARNING: template missing ({TEMPLATE_DIR}) — dry-run "
                  f"plan shown anyway; a real run would fail.")
        else:
            print(f"ERROR: template not found: {TEMPLATE_DIR}",
                  file=sys.stderr)
            return 1

    # Flatten (project, domain) pairs — tokens need domain fields too.
    pairs = [(p, d) for d in catalog["domains"] for p in d["projects"]]

    # --only filter: validate every requested ID so a typo ("33.1") fails
    # loudly instead of silently scaffolding nothing.
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        known = {p["id"] for p, _ in pairs}
        unknown = sorted(wanted - known)
        if unknown:
            print(f"ERROR: unknown project ID(s): {', '.join(unknown)}",
                  file=sys.stderr)
            return 1
        pairs = [(p, d) for p, d in pairs if p["id"] in wanted]

    created = skipped = 0
    for project, domain in pairs:
        outcome = stamp_project(project, domain, args.dry_run)
        if outcome == "created":
            created += 1
            verb = "WOULD create" if args.dry_run else "created"
            print(f"  {verb}  {project['folder']}")
        else:
            skipped += 1

    mode = " (dry run — nothing written)" if args.dry_run else ""
    print(f"scaffold: {created} created, {skipped} skipped "
          f"(already exist), {len(pairs)} considered{mode}")
    if created and not args.dry_run:
        # STATUS.md is generated from catalog.json + its own previous state;
        # newly stamped projects should appear on the work queue promptly.
        print("Reminder: run `python tools/status.py` to refresh "
              "docs/STATUS.md with the new projects.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
