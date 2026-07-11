# Push note — 2026-07-10-22: Phase 2 opens — batch 2a, domain 01's ★ projects

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

**Phase 2 begins.** Per §11 the build-out proceeds domain by domain, easiest-first, and pushes
per batch: this batch completes domain 01's three remaining ★ beginner projects (**39/505**) —
the classical camera front-end trilogy every perception stack stands on. **01.01** teaches the
five-stage ISP (debayer → undistort+rectify → resize → normalize) twice — staged and *fused* —
making kernel fusion a measured memory-traffic argument (32% derived saving), with a distortion
negative control proving the correction is real. **01.04** teaches the sparse-feature front-end
of visual odometry (FAST-9 and Harris side by side, oriented rBRIEF descriptors, `__popc`
Hamming matching) on a ground-truth-known image pair — descriptors bit-exact by construction,
92.3% of accepted matches landing on the known transform, and an unrelated-scene negative
control at 0/17. **01.06** builds a complete square-fiducial detector-decoder whose 32-code
dictionary is *generated, not copied* — seeded greedy search to a measured minimum Hamming
distance of 5 — so the coding-theory contract (correction capacity 2) is taught and then **gated
in both directions**: at-capacity bit flips must decode, beyond-capacity flips must be rejected.
These are also the first projects to inherit the standards retrospective: fresh `util/` with
`paths.h`, the twin-vs-shared ruling, and the pre-patched LNK4099 suppression.

## What changed

- **[projects/01-perception-cameras-vision/01.01-full-gpu-image-pipeline/](../projects/01-perception-cameras-vision/01.01-full-gpu-image-pipeline/)** —
  complete: RGGB bilinear debayer, GPU-built inverse remap LUT, 2× area-average resize,
  deterministic tree-reduction normalize; staged **and** fused pipelines with a derived
  memory-traffic account; 7 physics gates; per-stage PPM/PGM artifacts.
- **[projects/01-perception-cameras-vision/01.04-feature-pipeline/](../projects/01-perception-cameras-vision/01.04-feature-pipeline/)** —
  complete: FAST-9 (bit-exact end to end) + Harris (relative-tolerance twin, argued from the
  response's 13-orders-of-magnitude range), intensity-centroid ORB with 30-bin orientation
  quantization (bit-exact descriptors), brute-force Hamming with ratio + mutual checks; 4 gates.
- **[projects/01-perception-cameras-vision/01.06-apriltag-aruco-gpu-detector-decoder-for-high/](../projects/01-perception-cameras-vision/01.06-apriltag-aruco-gpu-detector-decoder-for-high/)** —
  complete: adaptive threshold → CCL (30.01's pattern, cited) → quad extraction + sub-pixel
  refinement → per-candidate DLT homography (33.01 cited) → 4-rotation grid decode → pose
  from homography; 5 gates including decode robustness both ways and a tag-free
  false-positive scene.
- **[docs/STATUS.md](../docs/STATUS.md)** — 01.01, 01.04, 01.06 → `done` (**39/505**).

## New projects (didactic blurbs)

**01.01 — Full GPU image pipeline** (★). Where every pixel a robot ever sees comes from: Bayer
mosaic physics, Brown–Conrady distortion, inverse mapping (why forward mapping leaves holes),
anti-aliasing as area averaging, and the fusion lesson — one gather kernel replacing three
round-trips. Most interesting artifact: `demo/out/rectified.ppm` next to `bayer_input.pgm` —
the visibly curved checkerboard squared up.

**01.04 — Feature pipeline** (★). What "trackable" means (aperture problem, structure-tensor
eigenvalues), why binary descriptors + popcount replaced float L2 for real-time matching, and
an honest failure study: the builder's first strictly-alternating checkerboard scene was
locally self-similar and tanked matching (~9% repeatability); hashed cell colors + antialiasing
took it to 63.7% — both documented in THEORY.md. Most interesting artifact:
`demo/out/matches.ppm`.

**01.06 — Fiducial detector-decoder** (★). Tags as a *designed* signal: the dictionary is a
coding-theory object (minimum distance over rotation orbits → correction capacity), and the
demo proves the contract experimentally in both directions. Also carries a worked bug case
study: a multiplicative corner-search margin that locked onto unrelated image content, caught
by the corner gate at 31 px error and fixed to an additive margin. Most interesting artifact:
`demo/out/detections_overlay.ppm`.

## How to build & run

```powershell
projects\01-perception-cameras-vision\01.01-full-gpu-image-pipeline\demo\run_demo.ps1
projects\01-perception-cameras-vision\01.04-feature-pipeline\demo\run_demo.ps1
projects\01-perception-cameras-vision\01.06-apriltag-aruco-gpu-detector-decoder-for-high\demo\run_demo.ps1
```

## What to study here

Read the three in order — they are one story: 01.01 makes geometrically-correct pixels, 01.04
turns them into trackable structure, 01.06 closes the loop with a designed target and a metric
pose. Then read each THEORY.md's failure/bug case study (scene self-similarity in 01.04, the
corner-margin bug in 01.06) — Phase 2's didactic habit of keeping honest mistakes as material.
Exercise: chain them — feed 01.01's rectified output into 01.04's detector and watch
repeatability against the unrectified image.

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-10), each project re-verified independently by the lead after the builder's
self-gate — all three: `Release|x64` **and** `Debug|x64` clean rebuilds, zero errors zero
warnings; demos exit 0 with all stable lines matched (13/13, 13/13, 11/11);
`tools/verify_project.py` all structural gates PASS; no changes outside each project's folder.

- **01.01:** debayer twin exact; remap/resize/fused twins ≤1 LSB; roundtrip 0.00000 px;
  straightness 0.74 px rectified vs 1.32 px raw (negative control); fused-vs-staged 0.0187;
  staged ≈0.44 ms vs fused ≈0.29 ms.
- **01.04:** FAST + descriptors + Hamming bit-exact; transform gate 60/65 (92.3% ≥ 90%);
  rotation recovered 11.51° of a true 12.0°; repeatability 63.7%; negative control 0/17.
  Builder disclosed reconstructing accidentally-deleted build files; lead verified the
  reconstruction (uuid5 GUID exact, all conventions intact) before accepting.
- **01.06:** all-integer stages exact (mask/CCL 0/172,800 across 3 scenes); 6/6 detections,
  corners ≤2.63 px (tol 3.5); pose ≤9.6° / 12.5% of tag size (honest homography-pose bounds,
  IPPE documented as the production refinement); robustness 4/4; false positives 0.

## Known limitations / TODOs

- 01.01: debayer excluded from the fused kernel (documented as the exercise); bilinear demosaic
  (Malvar–He–Cutler documented, not implemented).
- 01.04: didactic seeded BRIEF pattern (differs from OpenCV's learned pattern, argued); no
  scale pyramid (single-octave ORB — stated in README §13).
- 01.06: extreme-corner quad extraction is honestly weaker than production gradient-clustering
  (border-decode tolerance measured and documented accordingly); pose bounds wide without IPPE.

## Next push preview

Batch 2b: domain 01 intermediates begin — 01.03 optical flow (pyramidal Lucas–Kanade +
census-transform flow), then onward through the domain in ID order.
