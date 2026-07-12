# Push note — 2026-07-11-06: batch 2h — search, localization, deskew, geometry

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Batch 2h (**67/505**; domain 02 at 9/20) delivers the LBVH neighbor engine (02.05), NDT map
localization (02.07), motion deskew (02.08), and per-point normals (02.09). Two verification
stories headline. First, **the third verification tier earned its keep twice**: 02.05's missing
`__threadfence()` in the AABB propagation sailed through the twin comparisons and fell only to
the O(N·Q) brute-force anchor, and 02.09's brute-force gate exposed that the naive
"stop-once-K-found" KNN ring termination is not provably correct — both fixed with derived
rules, both kept as first-class THEORY material. Second, **02.07 is the batch's held-and-fixed
project**: the lead held the first delivery (16.7% convergence, a backwards outlier study), and
the finisher round *instrumented* the truth rather than tuning past it — convergence is
governed by perturbation **direction** (0/9 along the corridor axis vs 84% off-axis, tying the
basin gate directly to the measured Hessian degeneracy), the backwards outlier result was a
genuine RNG-stream desync in the generator, and an exhaustive parameter sweep is documented
with its negative results. The honest verdict is taught: ICP wins this small dense scene;
NDT's O(1) economics pay at map scale; and the corridor axis is why production fuses odometry.

## What changed

- **[projects/02-perception-lidar-point-clouds/02.05-kd-tree-or-lbvh-construction-knn-radius-search/](../projects/02-perception-lidar-point-clouds/02.05-kd-tree-or-lbvh-construction-knn-radius-search/)** —
  Karras radix tree from 64-bit augmented Morton keys (duplicate case eliminated by design,
  depth ≤62 proved), atomic AABB propagation, stack-based radius + register-heap KNN, honest
  hash-vs-BVH study; topology bit-exact over 398,807 nodes.
- **[projects/02-perception-lidar-point-clouds/02.07-ndt-scan-matching/](../projects/02-perception-lidar-point-clouds/02.07-ndt-scan-matching/)** —
  NDT voxel likelihood with derived gradient/Hessian (jacobian-gated), multi-resolution Newton,
  ICP contrast, failure-diagnosis instrumentation in the demo output; worker + finisher.
- **[projects/02-perception-lidar-point-clouds/02.08-per-point-motion-deskew-with-pose-interpolation/](../projects/02-perception-lidar-point-clouds/02.08-per-point-motion-deskew-with-pose-interpolation/)** —
  LERP+SLERP pose interpolation (double-cover and small-angle gates), four motion cohorts with
  an exact instantaneous-truth oracle, dense-vs-sparse sampling lesson, wall-fit payoff.
- **[projects/02-perception-lidar-point-clouds/02.09-normal-curvature-estimation-at-millions/](../projects/02-perception-lidar-point-clouds/02.09-normal-curvature-estimation-at-millions/)** —
  voxel-hash KNN + Jacobi eigensolve + oriented normals + surface-variation curvature +
  degeneracy flags; analytic plane/sphere/cylinder/edge truth; throughput gated at 19.9 Mpts/s.
- **[docs/STATUS.md](../docs/STATUS.md)** — 02.05, 02.07–02.09 → `done` (**67/505**).

## New projects (didactic blurbs)

**02.05 — LBVH + KNN.** The GPU-native answer to "who's near me": Morton codes as approximate
space-filling order (locality quantified at 1.6×), every radix-tree node built independently in
parallel, and a depth bound *proved* rather than hoped. The threadfence story is the chapter's
moral: memory-ordering bugs hide from twins that share your assumptions.

**02.07 — NDT scan matching.** Distribution matching taught against ICP, ending in the domain's
best observability lesson: a corridor cannot localize along itself — measured as a 0%-vs-84%
convergence split by perturbation direction, matched to Hessian conditioning, resolved in
production by odometry fusion. The parameter sweep's negative results are part of the material.

**02.08 — Motion deskew.** At 15 m/s a spinning LiDAR smears the world by 1.5 m per sweep;
SLERP along the trajectory puts it back — walls tighten 42× (24 cm → 5.6 mm, matching truth).
The stationary control is exact, and constant-velocity interpolation is *proven* exact on the
straight cohort while failing 19× on the wiggle — the sampling lesson with its own consistency
check.

**02.09 — Normals + curvature.** The feature layer point-to-plane ICP stands on: PCA normals
with the sign ambiguity taught, curvature ordering gated across four analytic surfaces, and the
aggregate beauty of 700 estimated normals recovering a cylinder's axis to 0.07°. Ships the
catalog's throughput promise honestly: 19.9 Mpts/s, methodology labeled.

## How to build & run

```powershell
projects\02-perception-lidar-point-clouds\02.05-kd-tree-or-lbvh-construction-knn-radius-search\demo\run_demo.ps1
projects\02-perception-lidar-point-clouds\02.07-ndt-scan-matching\demo\run_demo.ps1
projects\02-perception-lidar-point-clouds\02.08-per-point-motion-deskew-with-pose-interpolation\demo\run_demo.ps1
projects\02-perception-lidar-point-clouds\02.09-normal-curvature-estimation-at-millions\demo\run_demo.ps1
```

## What to study here

The verification arc is the batch's course: 02.05's threadfence and 02.09's ring-termination
bugs were both invisible to twin comparison and caught only by oracles that share *nothing*
with the implementation — read those two THEORY sections together, then 02.07's
failure-diagnosis instrumentation as the same philosophy applied to convergence claims.
Exercise: take 02.08's deskewed wiggle scan and feed it to 02.07's NDT — predict, then measure,
what the residual 9.6 cm deskew error does to the basin.

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-11), each project re-verified independently by the lead after the builder's
self-gate — all four: `Release|x64` **and** `Debug|x64` clean rebuilds, zero errors zero
warnings; demos exit 0 with all stable lines matched (16/16, 14/14, 16/16, 18/18);
`tools/verify_project.py` all structural gates PASS.

- **02.05:** tree topology bit-exact (398,807 nodes); brute-force anchor exact both query
  types; leaf-coverage and AABB-containment invariants exact; stack high-water 21 vs proved 62.
- **02.07:** worker + finisher (lead held the first delivery); twins/jacobian/score-sanity
  unregressed; the direction-split, paired outlier study, and converging twin example all
  verified in the lead's own demo run.
- **02.08:** twins to 2.4e-6 m; identity control exact 0.0; restoration sub-mm (straight/arc)
  vs 0.75–2.3 m undeskewed; SLERP gates 5.6e-7 rad; 18,426 points in 0.125 ms.
- **02.09:** KNN exact vs two independent searches + brute-force anchor (the termination-rule
  save); plane anchor 0.0024°; orientation 8,073/8,073; throughput 19.9 Mpts/s (floor 8);
  Debug passes correctness but not the throughput floor (`-G`, documented honestly).

## Known limitations / TODOs

- 02.05: rebuild-per-scan (refit documented). 02.07: small-scene ICP-vs-NDT verdict is
  scene-scale-specific (argued); z observability limited by scene design (measured, reported).
  02.08: poses assumed given (tightly-coupled undistortion named as the frontier). 02.09:
  surface variation ≠ true curvature (relationship documented); throughput methodology is
  replicated-sample, labeled.

## Next push preview

Batch 2i: 02.10 FPFH + RANSAC global registration, 02.11 Scan Context loop closure, 02.12
range-image conversion + depth clustering, 02.13 dynamic point removal.
