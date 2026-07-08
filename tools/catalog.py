#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# tools/catalog.py — parse the project catalog into machine-readable form
#
# Role in the repo (see CLAUDE.md §2):
#   `cuda-cpp-robotics-projects.md` is the READ-ONLY source of truth listing
#   every project as a markdown bullet inside 36 numbered sections. This
#   script is the ONE place where that markdown is parsed; every other tool
#   (scaffold.py, status.py, verify sweeps) consumes the generated
#   `catalog.json` and never re-parses the markdown. Centralizing the parsing
#   here means the ID assignment (`SS.NN`, deterministic by position) can
#   never drift between tools.
#
# Read this file BEFORE scaffold.py — scaffold consumes what this produces.
#
# Usage:
#   python tools/catalog.py            # writes catalog.json at the repo root
#   python tools/catalog.py --check    # parse + report only, write nothing
#
# Output: catalog.json — schema documented in `build_catalog()` below.
# ---------------------------------------------------------------------------

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

# The repo root is one level above tools/. We resolve it from this file's own
# location so the script works no matter what the current directory is.
REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_MD = REPO_ROOT / "cuda-cpp-robotics-projects.md"
CATALOG_JSON = REPO_ROOT / "catalog.json"

# ---------------------------------------------------------------------------
# Domain slugs — RATIFIED table from CLAUDE.md §3.
#
# We deliberately hard-code this mapping instead of deriving slugs from the
# section titles algorithmically: CLAUDE.md fixed these 36 names as part of
# the repository contract, and folder names must never change once projects
# exist under them. Parsing validates that the catalog's numbered sections
# are exactly 1..36 so this table can never silently go stale.
# ---------------------------------------------------------------------------
DOMAIN_SLUGS = {
    1:  "01-perception-cameras-vision",
    2:  "02-perception-lidar-point-clouds",
    3:  "03-perception-radar-sonar-event",
    4:  "04-sensor-fusion-state-estimation",
    5:  "05-slam-mapping-localization",
    6:  "06-motion-planning",
    7:  "07-collision-geometry",
    8:  "08-control-systems",
    9:  "09-dynamics-kinematics",
    10: "10-physics-simulation",
    11: "11-sensor-sim-digital-twins",
    12: "12-ml-ai",
    13: "13-locomotion-legged",
    14: "14-locomotion-wheeled",
    15: "15-locomotion-aerial",
    16: "16-locomotion-marine",
    17: "17-locomotion-space",
    18: "18-locomotion-other",
    19: "19-manipulation-grasping",
    20: "20-tactile-force-sensing",
    21: "21-hri-teleoperation",
    22: "22-multi-robot-swarms",
    23: "23-navigation-stack",
    24: "24-actuators-motors",
    25: "25-power-energy",
    26: "26-mechanical-design-structures",
    27: "27-materials-manufacturing",
    28: "28-soft-robotics",
    29: "29-medical-bio-robotics",
    30: "30-field-robotics",
    31: "31-safety-verification",
    32: "32-embedded-systems-infra",
    33: "33-foundational-libraries",
    34: "34-theory-frontier",
    35: "35-micro-nano-robotics",
    36: "36-modular-reconfigurable",
}

# A numbered section header looks like "## 12. Machine Learning & AI for
# Robots". Only these count as domains — "## Where to Start" has no number
# and is therefore prose, exactly as CLAUDE.md §2 requires.
SECTION_RE = re.compile(r"^##\s+(\d+)\.\s+(.+?)\s*$")

# A project is a TOP-LEVEL bullet: "- " at column 0. The catalog has no
# nested bullets today; if one ever appears (leading whitespace before "-"),
# it would NOT match this regex and would be reported as an anomaly below
# rather than silently swallowed.
BULLET_RE = re.compile(r"^- (.+?)\s*$")

# Connector words we refuse to END a slug with — "…-gicp-all" reads better
# than "…-gicp-all-batched" truncated to "…-all". Purely cosmetic; the
# SS.NN prefix already guarantees folder-name uniqueness.
SLUG_STOP_TAIL = {
    "to", "of", "and", "the", "for", "a", "an", "in", "on", "via",
    "with", "from", "all", "at", "over", "per", "vs", "or", "plus",
}


def slugify(name: str, max_chars: int = 48) -> str:
    """Turn a cleaned project name into a deterministic folder slug.

    Rules (CLAUDE.md §2): lowercase, ASCII, spaces and '/' become '-', drop
    punctuation, trim to something readable. The exact recipe:

      1. Drop parenthetical asides — "(Isaac-Gym-style: …)" is commentary,
         not a name. Done FIRST so a colon hiding inside parens can't
         confuse step 2.
      2. Prefer the pre-colon headline ("Stereo depth: block matching…" →
         "Stereo depth") — catalog bullets usually front-load the name.
         If the headline is a single generic word ("Agriculture:"), fall
         back to the whole cleaned name so slugs stay meaningful.
      3. NFKD-normalize then strip to ASCII (drops ★ →, ↔, ⁵, ∞ …).
      4. Everything non-alphanumeric becomes '-', runs collapse.
      5. Greedily keep whole words up to `max_chars`, then trim trailing
         connector words ("…-gicp-all" → "…-gicp").

    Determinism matters more than beauty: the same bullet must always yield
    the same folder name, forever (IDs and folders are never renumbered).
    """
    # (1) remove parenthetical asides
    base = re.sub(r"\([^)]*\)", " ", name)
    # (2) prefer the pre-colon headline when it is descriptive enough
    if ":" in base:
        head = base.split(":", 1)[0].strip()
        # "descriptive enough" = at least two words or a reasonably long one
        if len(head.split()) >= 2 or len(head) >= 10:
            base = head
    # (3) unicode → ascii (NFKD turns e.g. superscript ⁵ into 5, then the
    # ascii encode drops anything that has no ascii equivalent, like ★ or ∞)
    base = unicodedata.normalize("NFKD", base)
    base = base.encode("ascii", "ignore").decode("ascii").lower()
    # (4) collapse everything non-alphanumeric into single hyphens
    words = [w for w in re.split(r"[^a-z0-9]+", base) if w]
    # (5) greedy word-wise truncation to max_chars
    kept = []
    length = 0
    for w in words:
        add = len(w) + (1 if kept else 0)   # +1 for the joining hyphen
        if length + add > max_chars:
            break
        kept.append(w)
        length += add
    # cosmetic: never end on a connector word
    while len(kept) > 1 and kept[-1] in SLUG_STOP_TAIL:
        kept.pop()
    return "-".join(kept) if kept else "project"


def clean_name(raw: str) -> tuple[str, bool, bool]:
    """Strip difficulty tags from a raw bullet, per CLAUDE.md §2.

    Returns (clean_name, has_star, has_rnd).

    - A leading '★ ' marks a beginner entry point.
    - '[R&D]' may appear trailing OR inline (some bundled bullets carry
      several, e.g. §24's artificial-muscle bundle) — we strip every
      occurrence but record that at least one was present.
    """
    has_star = raw.startswith("★")
    name = raw.lstrip("★").strip()
    has_rnd = "[R&D]" in name
    name = name.replace("[R&D]", "")
    # Collapse whitespace runs left behind by tag removal, and tidy any
    # separator that ended up dangling at the edges ("… · " → "…").
    name = re.sub(r"\s{2,}", " ", name).strip()
    name = re.sub(r"\s*([·;,])\s*$", "", name).strip()
    # Also tidy "a · · b" artifacts if a tag sat between separators.
    name = re.sub(r"(\s[·;])\s*[·;]\s", r"\1 ", name)
    return name, has_star, has_rnd


def parse_catalog(text: str) -> tuple[list[dict], list[str]]:
    """Parse the catalog markdown into a list of domain dicts.

    Returns (domains, anomalies). Anomalies are human-readable strings for
    anything surprising — the contract (§2 / §13) says surprises get
    SURFACED (in the push-note), never silently patched.
    """
    domains: list[dict] = []
    anomalies: list[str] = []
    current: dict | None = None   # the domain we are currently filling

    for lineno, line in enumerate(text.splitlines(), start=1):
        sec = SECTION_RE.match(line)
        if sec:
            number = int(sec.group(1))
            title = sec.group(2)
            if number not in DOMAIN_SLUGS:
                anomalies.append(
                    f"line {lineno}: numbered section {number} has no "
                    f"ratified slug in CLAUDE.md §3 — skipped"
                )
                current = None
                continue
            current = {
                "number": number,
                "title": title,
                "slug": DOMAIN_SLUGS[number],
                "projects": [],
            }
            domains.append(current)
            continue

        bullet = BULLET_RE.match(line)
        if bullet and current is not None:
            raw = bullet.group(1)
            name, has_star, has_rnd = clean_name(raw)

            # Difficulty per §2: ★ → beginner, [R&D] → research, else
            # intermediate. §2 lists the ★ rule first, so ★ wins when a
            # bullet carries BOTH tags — but that combination is unusual
            # enough to surface as an anomaly for the push-note.
            if has_star and has_rnd:
                anomalies.append(
                    f"line {lineno}: bullet carries BOTH ★ and [R&D] "
                    f"('{name[:60]}…') — difficulty set to 'beginner' per "
                    f"§2 rule order; flags preserved in tags"
                )
            difficulty = (
                "beginner" if has_star
                else "research" if has_rnd
                else "intermediate"
            )

            # IDs are positional and zero-padded: 3rd bullet of section 8
            # → "08.03". NEVER renumber, NEVER sort (contract §2).
            ordinal = len(current["projects"]) + 1
            pid = f"{current['number']:02d}.{ordinal:02d}"
            slug = slugify(name)

            # Bundled bullets (several ideas joined by '·' or ';') stay ONE
            # project whose components become milestones — we just mark
            # them so the scaffolded README stub can say so.
            bundled = ("·" in name) or (";" in name)

            current["projects"].append({
                "id": pid,
                "name": name,
                "raw": raw,
                "slug": slug,
                "folder": f"projects/{current['slug']}/{pid}-{slug}",
                "difficulty": difficulty,
                "tags": {"star": has_star, "rnd": has_rnd},
                "bundled": bundled,
            })
            continue

        # A line that LOOKS like a nested/indented bullet inside a numbered
        # section would be a new catalog shape we have no rule for — flag it.
        if current is not None and re.match(r"^\s+-\s", line):
            anomalies.append(
                f"line {lineno}: indented bullet inside section "
                f"{current['number']} — catalog has no nesting rule; ignored"
            )

    # Structural sanity: the ratified table promises exactly sections 1..36.
    seen = [d["number"] for d in domains]
    missing = sorted(set(DOMAIN_SLUGS) - set(seen))
    if missing:
        anomalies.append(f"sections missing from catalog: {missing}")
    dupes = sorted({n for n in seen if seen.count(n) > 1})
    if dupes:
        anomalies.append(f"duplicate section numbers: {dupes}")
    for d in domains:
        if not d["projects"]:
            anomalies.append(f"section {d['number']} ('{d['title']}') has no bullets")

    return domains, anomalies


def build_catalog() -> dict:
    """Assemble the full catalog.json document.

    Schema (consumed by scaffold.py / status.py — change it only in
    lockstep with them):
      {
        "schema_version": 1,
        "source": "cuda-cpp-robotics-projects.md",
        "generator": "tools/catalog.py",
        "project_count": <int>,
        "counts_by_difficulty": {"beginner": n, "intermediate": n, "research": n},
        "anomalies": [ "<string>", ... ],
        "domains": [
          { "number", "title", "slug",
            "projects": [ { "id", "name", "raw", "slug", "folder",
                            "difficulty", "tags": {"star","rnd"},
                            "bundled" } ] }
        ]
      }
    """
    text = CATALOG_MD.read_text(encoding="utf-8")
    domains, anomalies = parse_catalog(text)
    projects = [p for d in domains for p in d["projects"]]
    by_diff = {"beginner": 0, "intermediate": 0, "research": 0}
    for p in projects:
        by_diff[p["difficulty"]] += 1
    return {
        "schema_version": 1,
        "source": CATALOG_MD.name,
        "generator": "tools/catalog.py",
        "project_count": len(projects),
        "counts_by_difficulty": by_diff,
        "anomalies": anomalies,
        "domains": domains,
    }


def main() -> int:
    # Windows consoles default to a legacy codepage (cp1252) that cannot
    # print '★'. Force UTF-8 so the report renders the catalog's own
    # symbols; errors="replace" keeps us alive even on odd terminals.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="parse and report, but do not write catalog.json")
    args = ap.parse_args()

    if not CATALOG_MD.exists():
        print(f"ERROR: catalog not found: {CATALOG_MD}", file=sys.stderr)
        return 1

    catalog = build_catalog()

    # Human-readable report — this is what the bootstrap push-note quotes.
    print(f"Parsed {CATALOG_MD.name}")
    print(f"  domains : {len(catalog['domains'])}")
    print(f"  projects: {catalog['project_count']} "
          f"(★ {catalog['counts_by_difficulty']['beginner']}, "
          f"intermediate {catalog['counts_by_difficulty']['intermediate']}, "
          f"[R&D] {catalog['counts_by_difficulty']['research']})")
    print("  per-domain: " + ", ".join(
        f"{d['number']:02d}:{len(d['projects'])}" for d in catalog["domains"]))
    if catalog["anomalies"]:
        print("  anomalies:")
        for a in catalog["anomalies"]:
            print(f"    - {a}")
    else:
        print("  anomalies: none")

    if not args.check:
        CATALOG_JSON.write_text(
            json.dumps(catalog, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        print(f"  wrote {CATALOG_JSON.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
