# Data — 09.01 Batched forward kinematics (10⁵ configurations — the foundation for everything above)

Provenance, licensing, and field documentation for everything under `data/` (CLAUDE.md §8).

## The rules (repo-wide, CLAUDE.md §8)

- **Synthetic-first.** Robotics data can almost always be synthesized with full ground truth;
  `../scripts/make_synthetic.py` is the default source, and synthetic data is **labeled synthetic
  everywhere it appears**.
- **Tiny committed sample.** `sample/` holds just enough committed data that the demo runs offline
  with zero downloads. Large/raw/downloaded data lives in `downloaded/` (git-ignored).
- **Public datasets** are fetched by `../scripts/download_data.ps1`/`.sh` where one genuinely
  teaches more — idempotent, documented, license-respecting. **This project needs none** (see below).
- **Never fabricate.** No made-up measurements, no invented ground truth passed off as real.

## This project's data

| Property | Value |
|----------|-------|
| Kind | **Synthetic** (100% generated) — a robot model is a list of numbers and configurations are angles; ground truth comes from computing FK twice (GPU vs the CPU oracle), so the file carries only inputs |
| File | `sample/fk_sample.csv` |
| Generator / source | `python ../scripts/make_synthetic.py` (defaults: seed 42, 64 configurations) |
| License | Synthetic — the repository's MIT license applies; the arm is a generic archetype, **no vendor's product** |
| Size (committed) | ~5.3 KiB |
| Checksum (SHA-256) | `6a5bdf9d7fc373d99241e56b051cdedfdc8e89babb3d6eb121740a228bba0b7d` |
| Regenerate with | `python ../scripts/make_synthetic.py` — byte-identical for the default seed 42 |

### Fields / format

Plain-text CSV; `#` lines are comments. Two row types (loader: `load_sample()` in
[`../src/main.cu`](../src/main.cu); layout authority: [`../src/kernels.cuh`](../src/kernels.cuh)):

**`J,<idx>,tx,ty,tz,qw,qx,qy,qz,ax,ay,az`** — one row per joint, **in chain order** (base first).

| Field | Units / frame | Meaning |
|-------|---------------|---------|
| `tx ty tz` | meters, previous link's frame | Fixed translation of `T_fix(j)`: where joint *j* sits on the previous link (what a URDF `<origin xyz>` encodes) |
| `qw qx qy qz` | unitless, **(w,x,y,z) repo order**, normalized | Fixed rotation of `T_fix(j)` (URDF `<origin rpy>` equivalent); identity `1,0,0,0` for every joint of the sample arm — its geometry lives in the axes instead |
| `ax ay az` | unit vector, joint *j*'s frame | Revolute joint axis; rotation by angle `q_j` about it (Rodrigues) |

**`Q,<idx>,q0,...,q5`** — one joint configuration: angles in **radians**, wrapped to **(−π, π]**
(the repo's canonical interval), uniform random, seed 42.

The sample robot: a synthetic generic 6-DoF anthropomorphic arm — base yaw (Z), shoulder pitch (Y),
elbow pitch (Y) after a 0.35 m upper arm, forearm roll (X) after 0.30 m, wrist pitch (Y) and roll
(X) on short 0.08/0.07 m links; ~0.90 m reach. Frames right-handed, x-forward/y-left/z-up. Every
number is invented for teaching (documented in `../scripts/make_synthetic.py`).

The loader is strict: unknown labels, wrong value counts, non-normalized quaternions/axes (beyond
1e-3 file rounding, re-normalized at load), or Q rows disagreeing with the J-row count abort the
demo — corrupt data can never quietly pass.

> `sample/` also carries its own [README](sample/README.md) stating the folder-wide rules.
