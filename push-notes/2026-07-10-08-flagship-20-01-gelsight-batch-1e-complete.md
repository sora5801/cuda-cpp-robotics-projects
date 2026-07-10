# Push note — 2026-07-10-08: flagship 20.01 gelsight batch 1e complete

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Flagship 20.01 — **GelSight/DIGIT tactile processing** — is done, closing **batch 1e** (16.01,
18.01, 19.01, 20.01; **24/505 overall, 24 of 36 flagships**). A synthetic vision-based tactile
sensor sequence (100 frames: press → shear → slip, physics from Johnson's *Contact Mechanics*)
flows through the catalog's three named stages — contact patch, marker-tracked shear field, slip
detection — with every ground truth known by construction: the contact footprint follows the
Hertzian a = √(Rδ) law (measured patch area within 1.3%), the shear field recovers the commanded
translation *exactly* (proved as an integer-rounding identity, not luck), and slip onset is
detected **one frame** from the Cattaneo–Mindlin model's prediction with zero false alarms across
the 60-frame stick phase. The whole per-frame pipeline is integer/threshold arithmetic, so
GPU-vs-CPU verification is bit-exact on every one of the 100 frames.

## What changed

- **[projects/20-tactile-force-sensing/20.01-gelsight-digit-processing/](../projects/20-tactile-force-sensing/20.01-gelsight-digit-processing/)** —
  complete: contact-mask + morphology kernels, per-marker detect/track kernels, host Procrustes
  rigid fit + slip scoring, in-code deterministic frame renderer (581-byte committed scenario),
  CPU twin (bit-exact), contact/shear/slip gates, three artifacts, full README / THEORY /
  PRACTICE.
- **[docs/STATUS.md](../docs/STATUS.md)** — 20.01 → `done` (**24/505**).

## New projects (didactic blurbs)

**20.01 — GelSight processing** (★ beginner, domain 20, flagship). How a camera becomes a
fingertip: elastomer + internal illumination turn contact geometry into photometry, printed
markers turn membrane motion into a trackable displacement field, and contact mechanics turns
that field into physics — the stick-slip annulus (periphery slips first, c = a(1−s)^⅓) is why
partial slip is *detectable before the object is lost*, which is the entire value of tactile
sensing in grasping. The single most interesting thing to look at: `demo/out/slip_timeline.csv`
plotted — the slip score sitting at zero through stick, then climbing exactly where the model
says it must.

## How to build & run

```powershell
projects\20-tactile-force-sensing\20.01-gelsight-digit-processing\demo\run_demo.ps1
# then plot demo\out\slip_timeline.csv and open demo\out\contact_mask.pgm
```

## What to study here

Batch 1e as a set is the manipulation-adjacent column: allocate thrust (16.01), locomote (18.01),
choose a grasp (19.01), feel it slipping (20.01). Within 20.01: `THEORY.md` §The problem (the
sensor's photometric physics and the honest intensity-proxy scoping vs real photometric stereo) →
the Cattaneo–Mindlin derivation → `src/kernels.cu`. First exercise: run the edge indenter mode
(`make_synthetic.py --indenter edge`) and watch the patch/shear behavior change shape.

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-10), re-run independently by the lead after the builder's self-gate:

- `Release|x64` **and** `Debug|x64` build with **zero errors, zero new warnings**.
- `demo/run_demo.ps1` passes end to end: all 9 stable lines matched, exit 0.
- **GPU-vs-CPU gate: bit-exact** — 0 mismatches over 100 frames × 5 kernel stages.
- **Ground-truth gates:** contact-patch area error 1.3% vs the Hertzian footprint (gate 5%),
  centroid 0.13 px (gate 1.0); shear displacement error 0.00 px exact vs the commanded 5.0 px;
  slip onset frame 85 vs modeled 86 (gate ±2), zero false slips in frames 0–59.
- The slip threshold was calibrated against the model (1.5 → 1.1 px) with the tuning documented —
  measured alignment, not forced passing.
- Timing (teaching artifact): ≈ 0.17 ms/frame for the 5-kernel pipeline.
- `tools/verify_project.py`: **all structural gates PASS**; no changes outside the project folder.

## Known limitations / TODOs

- Intensity-proxy indentation (real GelSight reconstructs depth via 3-color photometric stereo —
  the documented milestone), synthetic gel model, sparse marker flow rather than dense flow
  (trade documented). Edge indenter implemented but not gated.

## Next push preview

Batch 1f: 21.04 speed-and-separation monitoring (the ISO/TS 15066-adjacent safety flagship —
didactic-not-certified per §8), 24.01 magnetostatic FEA sweeps, 25.01 battery electro-thermal,
26.01 topology optimization.
