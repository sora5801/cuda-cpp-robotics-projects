# Data — 01.14 Template matching (NCC) at scale for pick verification

Provenance, licensing, and field documentation for everything under `data/` (CLAUDE.md §8).

## The rules (repo-wide, CLAUDE.md §8)

- **Synthetic-first.** Robotics data can almost always be synthesized with full ground truth;
  `../scripts/make_synthetic.py` is the default source, and synthetic data is **labeled synthetic
  everywhere it appears**.
- **Tiny committed sample.** `sample/` holds just enough committed data that the demo runs offline
  with zero downloads. Large/raw/downloaded data lives in `downloaded/` (git-ignored).
- **Public datasets** (where one genuinely teaches more) are fetched by `../scripts/download_data.ps1`
  / `.sh` — idempotent, with source URL, expected size, and checksum documented below. **Respect every
  license**; registration-gated or no-redistribution datasets (KITTI, nuScenes) are pointed at, never
  mirrored.
- **Never fabricate.** No made-up measurements, no invented ground truth passed off as real.

## This project's data

Why synthetic is not a compromise here (CLAUDE.md §8's default, and the honest right call): pick
verification needs EXACT ground truth for what part is *actually* in each tray slot, at what pose, and
what the correct verdict is — a real photograph of a real pick-and-place cell cannot give you that
without hand-labeling every slot after the fact, while a renderer that places the parts itself gives
it for free, by construction. `scripts/download_data.ps1` / `.sh` are therefore honest no-ops for this
project — no public "pick verification tray" dataset with this kind of per-slot ground truth exists
anyway.

| Property | Value |
|----------|-------|
| Kind | Synthetic (repo default) |
| Generator | `../scripts/make_synthetic.py --seed 42` |
| License | Synthetic — repo MIT license applies (no external data, no restrictions) |
| Size (committed) | ~78 KB total (70 KB tray PGM + 8.5 KB templates PGM + 1 KB truth CSV) |
| Checksum (SHA-256) | `tray.pgm`: `77763fd22a91d510bc275083374170ecb8eded1d950143a3e6cfa3953838c503` |
| | `templates.pgm`: `a34f6137da9b35a963dfc6ee21615c034ba59b8b852f36d8f15bc988ace9927d` |
| | `truth.csv`: `e1d0321ab3a0569d4cf09f0dfb50c5c9ff979667bbe593069a8614a2b219a1c7` |
| Regenerate with | `python ../scripts/make_synthetic.py` (defaults reproduce the committed files exactly) |

Recompute any checksum with `python -c "import hashlib;print(hashlib.sha256(open('FILE','rb').read()).hexdigest())"`.

### What the scene depicts

A 324x220, 8-bit grayscale image of a 6x4 (24-slot) parts tray photographed from directly above after
a robot pick-and-place cycle, plus a stacked set of 15 "golden reference" templates (3 machined part
silhouettes — a corner **bracket**, a toothed **gear disk**, and a two-hole **connector block** — each
pre-rotated to 5 angles: -6, -3, 0, +3, +6 degrees). Every slot has a designed, deterministic outcome
(the ground truth): 22 slots hold the correct part, correctly placed (one of them, `offset`, shifted
5,-6 px within the +-8 px search range; one, `rotated`, placed at 24 degrees — see
`../THEORY.md`/`../scripts/make_synthetic.py` for why that specific angle was *measured*, not guessed,
to cleanly separate single-template from rotation-set recovery; one, `shadow`, rendered under a
partial illumination gradient); one slot (`wrong_part`) holds the wrong part entirely; one slot
(`empty`) holds nothing. `../src/main.cu`'s job is to recover every one of these outcomes from pixels
alone.

### Fields / format

**`tray.pgm`** — binary PGM (P5): a 3-line ASCII header (`P5\n324 220\n255\n`) followed by 71,280 raw
uint8 grayscale samples, row-major, `idx = y*324+x`. Pixel coordinates only (`x` right, `y` down) — see
`../src/kernels.cuh` SECTION 1 for the exact tray/slot/window geometry this indexes into.

**`templates.pgm`** — binary PGM (P5), `P5\n24 360\n255\n` followed by 8,640 raw uint8 samples: 15
templates of 24x24 pixels each, stacked **vertically** in `template_id = type*5 + rotation_index` order
(bracket -6..+6, gear disk -6..+6, connector block -6..+6) — the raw bytes ARE exactly the flat
`[NUM_TEMPLATES][TEMPLATE_SIZE][TEMPLATE_SIZE]` array `../src/kernels.cuh` expects, no reshuffling.

**`truth.csv`** — plain CSV, `#`-prefixed header comments, one row per slot:

| Column | Meaning |
|--------|---------|
| `slot` | Row-major slot index, `row*6 + col`, `[0, 24)`. |
| `row`, `col` | Tray grid position, `row in [0,4)`, `col in [0,6)`. |
| `cohort` | `plain` / `offset` / `rotated` / `wrong_part` / `empty` / `shadow` — which designed case this slot is (README "The algorithm in brief" / `THEORY.md` name each). |
| `expected_type` | 0=bracket, 1=gear_disk, 2=connector_block — the part the tray's "recipe" (`slot % 3`) says should be here. |
| `actual_type` | The part actually rendered here; `-1` for the `empty` cohort. |
| `rotation_deg` | The applied rotation of the actual part, degrees (0 unless `cohort=rotated`). |
| `offset_dx_px`, `offset_dy_px` | The applied placement offset from the slot's nominal center, pixels (0 unless `cohort=offset`). |
| `shadow` | `1` if the illumination gradient was applied to this slot (only `cohort=shadow`), else `0`. |
| `verdict` | The CORRECT classification: `OK` / `WRONG_PART` / `EMPTY` — what `../src/main.cu`'s `classification` gate checks its own output against. |

All positions are in pixels, tray-image frame (`../src/kernels.cuh` SECTION 1); this teaching project
does not model a camera's mm-per-pixel intrinsic calibration (see `../README.md` "Limitations & honesty"
and `../PRACTICE.md` §3 for how a real station gets millimeters out of a pipeline like this one).
