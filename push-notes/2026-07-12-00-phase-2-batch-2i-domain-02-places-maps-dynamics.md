# Push note — 2026-07-12-00: batch 2i — global registration, places, and living maps

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Batch 2i (**71/505**; domain 02 at 13/20) is the "getting un-lost and keeping maps honest" arc:
FPFH global registration (02.10), Scan Context place recognition (02.11), range-image depth
clustering (02.12), and dynamic point removal (02.13). The through-line is *properties gated by
measurement*: 02.10 doesn't assert FPFH's pose invariance, it measures it (mean L1 0.057 between
descriptors of the same physical points seen from frames 140°/8 m apart) and then proves the
method earns its keep with an ICP-from-identity negative control stranded at 141.7°/7.2 m;
02.11 gates rotation invariance *with its free by-product* (the column-shift argmin recovering
the true 180° yaw exactly) and the safety-critical direction hard (0/8 false loop closures, the
map-corruption stake stated); 02.12's designed scene lands the β-criterion's advantage (a person
0.19 m from a wall: 0 shared depth clusters vs 1 for fixed-distance Euclidean) *and* its grazing
weakness (13 fragments, asserted); and 02.13 turns free-space physics into a filter with
bit-exact DDA and the field's honest failure mode on display — the thin pole 100% eroded, the
reason Removert-class methods exist. Every quantitative claim in the batch traces to a run.

## What changed

- **[projects/02-perception-lidar-point-clouds/02.10-fpfh-descriptors-ransac-global-registration/](../projects/02-perception-lidar-point-clouds/02.10-fpfh-descriptors-ransac-global-registration/)** —
  Darboux-frame SPFH/FPFH derived, correspondence RANSAC with edge-length prescreen (99.2%
  rejected pre-fit) and the 3-sample formula gated (budget honestly raised 4,000→8,192 when
  w=0.105 demanded 6,012); cold-start recovery to 0.000°/0.011 m; 33.5%-overlap failure
  reported, not gated.
- **[projects/02-perception-lidar-point-clouds/02.11-scan-context-ring-descriptor-loop-closure-search/](../projects/02-perception-lidar-point-clouds/02.11-scan-context-ring-descriptor-loop-closure-search/)** —
  ring×sector max-z descriptor, ring-key prefilter (recall 0.862 at budget 12, measured),
  shift-minimizing cosine distance; precision 1.000 / recall 0.769 with all misses in the
  lateral-offset cohort (the known limit, measured per offset); yaw handoff to ICP demonstrated.
- **[projects/02-perception-lidar-point-clouds/02.12-range-image-conversion-depth-clustering/](../projects/02-perception-lidar-point-clouds/02.12-range-image-conversion-depth-clustering/)** —
  Bogoslavskyi–Stachniss β criterion derived from the line-of-sight triangle, union-find on
  image-grid edges (02.04's kernels reused verbatim), angle-walk ground removal (P 0.979 /
  R 1.000), injected phantom points verifying the atomicMin race; image path 1.5× faster.
- **[projects/02-perception-lidar-point-clouds/02.13-dynamic-point-removal/](../projects/02-perception-lidar-point-clouds/02.13-dynamic-point-removal/)** —
  Amanatides–Woo DDA (bit-exact via integer stopping + fmaf discipline), three-way hit/pass
  ledger with exact accounting, ghost trail 97.7% removed, the late-leaver temporal lesson
  (3.8% → 94.2%), max-range carving proven by ledger decomposition.
- **[docs/STATUS.md](../docs/STATUS.md)** — 02.10–02.13 → `done` (**71/505**).

## New projects (didactic blurbs)

**02.10 — FPFH + RANSAC.** Why local geometry can be described pose-invariantly (the Darboux
frame), and the global-then-local doctrine made concrete: RANSAC finds the 140°/8 m transform no
local method could, then ICP polishes. The prescreen is the unsung hero — 99.2% of hypotheses
die before the expensive fit.

**02.11 — Scan Context.** Place recognition from vertical structure: rotation becomes a column
shift, the shift is a free yaw estimate, and a false loop closure is the one error a SLAM system
cannot afford — gated at zero, threshold chosen mid-gap rather than curve-hugging. Two found
bugs (a sentinel eating ground cells; anchors 6 m apart from an off-by-half) kept as material.

**02.12 — Depth clustering.** The range image is the LiDAR's native geometry; adjacency by
index replaces neighbor search entirely. The β criterion separates by *depth gap* — range-free
where Euclidean d is range-bound — and fragments at grazing incidence, both demonstrated on
designed cohorts. Occlusion shadows and ring-step bridging: two scene-design lessons measured
into the docs.

**02.13 — Dynamic removal.** LiDAR's free-space information is as valuable as its hits: rays
that pass through yesterday's points expose yesterday's cars. The pedestrian cohort teaches
temporal evidence (nearly invisible while present, 94% removed after leaving), and the thin
pole teaches the cost of voxel discretization — eroded entirely, honestly, with the
visibility-method literature named as the fix.

## How to build & run

```powershell
projects\02-perception-lidar-point-clouds\02.10-fpfh-descriptors-ransac-global-registration\demo\run_demo.ps1
projects\02-perception-lidar-point-clouds\02.11-scan-context-ring-descriptor-loop-closure-search\demo\run_demo.ps1
projects\02-perception-lidar-point-clouds\02.12-range-image-conversion-depth-clustering\demo\run_demo.ps1
projects\02-perception-lidar-point-clouds\02.13-dynamic-point-removal\demo\run_demo.ps1
```

## What to study here

The batch composes into SLAM's outer loop: 02.11 says "you've been here," 02.10 proves it
geometrically, 02.13 keeps the resulting map honest, and 02.12 shows the low-latency lane for
the objects that move through it. Study the three negative controls side by side —
ICP-from-identity (02.10), the never-revisited places (02.11), the phantom collision points
(02.12) — as three shapes of the same discipline. Exercise: wire 02.11's detected pair into
02.10's registration and check the recovered transform against the trajectory truth.

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, driver 591.86, CUDA 13.3, VS 2026
v145, 2026-07-11/12), each project re-verified independently by the lead after the builder's
self-gate — all four: `Release|x64` **and** `Debug|x64` clean rebuilds, zero errors zero
warnings; demos exit 0 with all stable lines matched (20/20, 16/16, 19/19, 14/14);
`tools/verify_project.py` all structural gates PASS; no changes outside each project's folder.

- **02.10:** eight verify stages 100% agreement; invariance 0.057 L1; recovery 0.000°/0.011 m;
  negative control 141.7°/7.2 m; formula check k=6,012 < 8,192.
- **02.11:** SC cells 99.74% exact (boundary ties documented); P 1.000 / R 0.769; rotated
  recall 1.000 with 0.0° yaw error; 0/8 false closures; prefilter recall 0.862.
- **02.12:** all six verify stages exact; clean-cohort IoUs 1.000; showcase 0-vs-1 shared
  clusters at a 0.19 m gap; grazing 13 fragments asserted; ground P 0.979 / R 1.000.
- **02.13:** DDA/ledger/classification bit-exact (0 mismatches over 2.3M counters); ghosts
  97.7%; late-leaver 3.8%→94.2%; statics 5.4% ≤ 15% with the pole/edge cohorts separated;
  accounting exact (9,051 = 9,051).

## Known limitations / TODOs

- 02.10: brute-force matching at teaching scale (indexing named for production); low overlap
  fails honestly. 02.11: translation sensitivity measured, not solved (SC++ named). 02.12:
  β threshold is scene-tuned (the theta trade documented). 02.13: batch post-session carving
  (incremental/visibility methods named); thin structures erode by design limitation.

## Next push preview

Batch 2j: 02.14 moving-object segmentation, 02.15 point-cloud compression, 02.16 multi-LiDAR
merging, 02.17 LiDAR-camera fusion kernels — domain 02's final stretch begins.
