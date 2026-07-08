# Data — 02.02 ROI crop, passthrough, organized↔unorganized conversion kernels

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

TODO(scaffold): fill in the table and field documentation for the real sample data.

| Property | Value |
|----------|-------|
| Kind | TODO(scaffold): synthetic (default) or public dataset (name it) |
| Generator / source | TODO(scaffold): `../scripts/make_synthetic.py` invocation, or source URL |
| License | TODO(scaffold): e.g. "synthetic — repo MIT license applies" or the dataset's license |
| Size (committed) | TODO(scaffold): keep it tiny (well under 50 MB; prefer KB) |
| Checksum | TODO(scaffold): SHA-256 of each committed sample file |
| Regenerate with | TODO(scaffold): exact command, including the fixed seed |

### Fields / format

TODO(scaffold): document every column/field of every sample file — name, type, **units, frame**
(SI, right-handed, `T_parent_child` conventions per CLAUDE.md §12), and valid range.

> **Placeholder status:** as scaffolded, the SAXPY placeholder demo generates its input **in memory**
> (deterministically, no seed needed — see `make_input()` in `../src/main.cu`) and needs no files.
> `../scripts/make_synthetic.py` writes a small demonstration CSV into `sample/` so the synthetic-data
> pattern is visible from day one.
