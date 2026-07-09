# Data — 07.09 Jump-flooding Voronoi/distance transforms (easy, visual, useful)

Provenance, licensing, and field documentation for everything under `data/` (CLAUDE.md §8).

## The rules (repo-wide, CLAUDE.md §8)

- **Synthetic-first.** Robotics data can almost always be synthesized with full ground truth;
  `../scripts/make_synthetic.py` is the default source, and synthetic data is **labeled synthetic
  everywhere it appears**.
- **Tiny committed sample.** `sample/` holds just enough committed data that the demo runs offline
  with zero downloads. Large/raw/downloaded data lives in `downloaded/` (git-ignored).
- **Public datasets** are fetched by `../scripts/download_data.ps1`/`.sh` where one genuinely
  teaches more — idempotent, documented, license-respecting. **This project needs none** (the input
  is seed cells on a grid; the exact CPU oracle supplies ground truth at run time).
- **Never fabricate.** No made-up measurements, no invented ground truth passed off as real.

## This project's data

| Property | Value |
|----------|-------|
| Kind | **Synthetic** (100% generated) — Voronoi seeds ≙ obstacle cells on a costmap |
| File | `sample/jfa_seeds.csv` |
| Generator / source | `python ../scripts/make_synthetic.py` (defaults: seed 42, 64 seeds, 512×512 grid) |
| License | Synthetic — the repository's MIT license applies |
| Size (committed) | ~1.2 KiB |
| Checksum (SHA-256) | `cdad3167729b6a91474f2c912c99b6dd30e079afefdb2d0e08ae48095faebc4f` |
| Regenerate with | `python ../scripts/make_synthetic.py` — byte-identical for the default seed 42 |

### Fields / format

Plain-text CSV; `#` lines are comments. One row per seed:

```
S,<id>,x,y
```

| Field | Type / range | Meaning |
|-------|--------------|---------|
| `S` | literal | Row label (the only one in this file) |
| `id` | int, sequential from 0 | The seed's Voronoi label — what the output regions are colored by |
| `x`, `y` | int, `[0,512)` each | The seed's cell on the fixed 512×512 sample grid (row-major, x rightward, y downward — image convention; see `../src/kernels.cuh`) |

Seed cells are guaranteed **distinct** (the GPU scatter kernel relies on it; both the generator and
the loader in `../src/main.cu` enforce it). Units: cells — the demo grid is unitless; a robotics
consumer scales by its map resolution (e.g. 0.05 m/cell on a navigation costmap), which turns the
output distance field directly into meters of clearance.

The loader is strict: unknown labels, out-of-range coordinates, non-sequential ids, or duplicate
cells abort the demo — corrupt data can never quietly pass.

> `sample/` also carries its own [README](sample/README.md) stating the folder-wide rules.
