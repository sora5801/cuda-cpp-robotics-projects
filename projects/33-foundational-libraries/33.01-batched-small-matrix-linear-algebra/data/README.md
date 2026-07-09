# Data — 33.01 Batched small-matrix linear algebra (3×3, 4×4, 6×6 — the robotics sizes)

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

| Property | Value |
|----------|-------|
| Kind | **Synthetic** (100% generated; no external dataset — for a linear-algebra library, the "sensor data" simply *is* matrices) |
| File | `sample/smallmat_sample.csv` |
| Generator / source | `python ../scripts/make_synthetic.py` (defaults: seed 42, output path pre-wired) |
| License | Synthetic — the repository's MIT license applies; no third-party terms involved |
| Size (committed) | ~59 KiB |
| Checksum (SHA-256) | `b075cdff6d49bd2add867ff4753fc722e3b919ca2869821a740bcfbf906f6ffe` |
| Regenerate with | `python ../scripts/make_synthetic.py` — byte-identical output for the default seed 42 (Python's `random.Random` is specified by the language reference and stable across platforms) |

There is deliberately **no ground-truth column** in the file: the demo computes every answer twice —
GPU kernels vs. the plain-C++ CPU oracle in `../src/reference_cpu.cpp` — and compares within the
tolerances documented in `../src/main.cu`. The oracle *is* the ground truth; the file only supplies
reproducible, well-conditioned inputs.

### Fields / format

Plain-text CSV. Lines starting with `#` are comments (provenance and the SYNTHETIC label). Every
data row is one problem:

```
label,index,v0,v1,...,v{m-1}
```

| Column | Type | Meaning |
|--------|------|---------|
| `label` | string | Which array this row belongs to (table below) |
| `index` | int | Row number within its label, 0-based — informational; the loader keys on labels and preserves file order |
| `v0..v{m-1}` | float (`%.9g`, parses exactly at FP32) | The matrix/vector values, **row-major** (element *(i,j)* of an *n×n* matrix is `v[i*n + j]`) |

| Label | Values per row | Rows | Meaning |
|-------|----------------|------|---------|
| `A3`,`B3` | 9 | 64 each | Matmul input pairs, *n*=3 (`A3` row *k* pairs with `B3` row *k*) |
| `A4`,`B4` | 16 | 32 each | Matmul input pairs, *n*=4 |
| `A6`,`B6` | 36 | 16 each | Matmul input pairs, *n*=6 |
| `S6` | 36 | 32 | Symmetric positive-definite 6×6 matrices, built as `A = G·Gᵀ + 6·I` with `G` uniform in [−1,1) — strictly SPD with single-digit condition number (the construction is explained in `../scripts/make_synthetic.py` and mirrored by `make_spd_batch` in `../src/main.cu`) |
| `b6` | 6 | 32 | Right-hand-side vectors for the `S6` systems (`S6` row *k* pairs with `b6` row *k*) |

**Units and frames:** none — these are dimensionless unit-scale test matrices, not physical
quantities. (Real consumers of this library attach units at the call site: inertia matrices in
kg·m², Jacobians mapping rad/s → m/s; see README.md §System context.) Value range: matmul inputs and
`b6` entries lie in [−1,1); `S6` diagonals lie in [6−ε, 6+‖G‖²].

The loader (`load_sample()` in `../src/main.cu`) is strict: unknown labels, wrong value counts, or
mismatched pair counts abort the load and fail the demo — corrupt data can never quietly pass.

> `sample/` also carries its own [README](sample/README.md) stating the folder-wide rules.
