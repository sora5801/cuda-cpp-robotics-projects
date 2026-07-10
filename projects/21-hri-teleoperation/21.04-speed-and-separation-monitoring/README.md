# 21.04 — Speed-and-separation monitoring: depth streams → minimum-distance fields at frame rate (ISO/TS 15066 helper)

**Difficulty:** ★ beginner · **Domain:** 21. Human-Robot Interaction & Teleoperation

> Catalog bullet (source of truth, verbatim): `★ Speed-and-separation monitoring: depth streams → minimum-distance fields at frame rate (ISO/TS 15066 helper)`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).
>
> **>>> DIDACTIC IMPLEMENTATION — NOT A CERTIFIED SAFETY FUNCTION. <<<** This project computes
> metrics *adjacent to* ISO/TS 15066's speed-and-separation-monitoring concept, for teaching. It is
> **not** a certified implementation of any standard, is **not** safety-rated, and must never guard a
> real robot. The demo itself prints a stable `NOTICE:` line saying exactly this, every run. The
> synthetic "human" in this project is an **anonymous cylinder/capsule pair** — there is no
> identity, tracking, or recognition anywhere in this code; the framing is collaborative safety
> (protecting a generic person near a robot), never surveillance of individuals.

## Overview

A synthetic overhead depth camera watches a small collaborative work cell for 8 seconds: a
SCARA-style robot arm (8 teaching capsules, from real forward kinematics) performs its own reach
cycle while an anonymous person — a torso+arm capsule pair, nothing more — walks in from the side,
passes close to the robot's tool, and walks back out. Every frame (30 Hz, matching a real depth
camera's rate), the GPU pipeline renders the depth image, separates the person's silhouette from the
robot's own known silhouette, finds the minimum distance from any point on the person to the nearest
robot part, and compares that distance against two ISO/TS-15066-*style* protective separation
distances to drive a NORMAL → REDUCED → PROTECTIVE_STOP state machine with documented hysteresis. All
three named catalog components are implemented: the minimum-distance *field* at frame rate (a
map+reduce kernel over human pixels), the dense per-pixel clearance-*field* artifact the bullet also
names (the same distance function run everywhere, not just over human pixels), and the ISO/TS-15066
*helper* decision itself. The demo verifies its own output four independent ways against a
closed-form ground truth it computes directly from the scenario's geometry — not just "the GPU
matches the CPU," but "the whole pipeline is measuring the right thing" (see
[Expected output](#expected-output)).

## What this computes & why the GPU helps

Per frame: a 200×200-pixel top-down depth image (40,000 pixels) is rendered and classified against
10 capsules (a **map**), then every HUMAN-labeled pixel's distance to the nearest of 8 robot capsules
is found and collapsed to one number (a **map + reduce**), and — once per demo, for the artifact —
the same distance is computed at *every* pixel (a **map** again, reusing the same device function).

- **Pattern:** map (render/classify, dense field) and map+reduce (the minimum-distance field) — one
  thread per pixel throughout; the reduction is a canonical shared-memory tree reduction, notable
  because MIN is exactly commutative/associative in IEEE-754, unlike a summed reduction (THEORY.md
  "Numerical considerations").
- **Why the GPU helps:** at 30 Hz, the whole per-frame budget is ~33 ms; this project's two per-frame
  kernels (render+classify, then the min-distance map+reduce) together average well under 1 ms on an
  RTX 2080 SUPER (measured, single-shot — see the demo's `[time]` line) — the same "the GPU is
  negligible against the real-time budget" story SYSTEM_DESIGN.md tells for perception/planning
  layers generally.
- **Measured reality:** the demo's own VERIFY stage found 0 label mismatches between the GPU
  classification and its independent CPU oracle across 40,000 pixels × 2 frames, and the min-distance
  reduction agreed with the CPU's sequential scan to within 1.2e-7 m — see [Expected output](#expected-output).

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** a **cross-cutting safety monitor**, sitting beside (not inside) the
  perception→planning→control chain — SYSTEM_DESIGN.md §2.2's manipulator work-cell diagram places
  "HUMAN SAFETY [21 →] speed-and-separation monitoring" as its own block, reading the cell's camera
  independently and issuing "slow/stop overrides" into joint control. This project computes exactly
  that block's core metric.
- **Upstream inputs:** a depth stream (message shape: `sensor_msgs/Image`-like, SYSTEM_DESIGN.md
  §3.6's `Image` struct — here a synthetic top-down depth image) and the robot's own state (message
  shape: `JointState`-like — here the SCARA joint angles that drive `build_scene()`'s forward
  kinematics, standing in for what a real system would read off the robot controller).
- **Downstream consumers:** the robot's **actuation/control layer** — but only as an *advisory*
  input. This project's state machine output (NORMAL/REDUCED/PROTECTIVE_STOP) is exactly the kind of
  signal SYSTEM_DESIGN.md §2.2 shows overriding joint control — the honesty that matters (and this
  project's central caveat) is that **this module advises; a certified safety-rated layer must act**
  (§6.1's hardwired safety chain — E-stop relays, STO inputs on the drive — is what actually cuts
  power, never a GPU program). Catalog project **31.01** (Hamilton-Jacobi reachability) makes the
  identical point for its own safety-adjacent metric; both projects are natural siblings of the
  certified-layer discussion in `PRACTICE.md` §3/§4 here and there.
- **Rate/latency budget:** SYSTEM_DESIGN.md §1.1 puts camera→perception at 30–60 Hz with a
  sub-frame latency budget; this demo runs the full render+classify+reduce pipeline at a synthetic
  30 Hz with well under 1 ms of measured GPU kernel time per frame (see the `[time]` line) — the
  reaction-time budget argument matters here specifically: this project's own S_p formula
  (kernels.cuh SECTION 6) treats the camera's frame period as part of the system's total reaction
  time T_r, so running the monitor slower than the camera directly enlarges the protective distance
  a real system would need.
- **Reference robot(s):** the **6-DoF manipulator work cell** (SYSTEM_DESIGN.md §2.2) — the SCARA
  arm modeled here is a real, named sub-type of that reference robot's actuation, and the human-safety
  block in that section's diagram is precisely this project's slot.
- **In production:** certified area scanners and safety-rated depth/lidar systems (SICK, Pilz, and
  similar functional-safety vendors) occupy this slot for real, with independent, redundant,
  certified sensing and a hardwired stop path — see [Prior art](#prior-art--further-reading) and
  `PRACTICE.md` §2/§4.
- **Owning team:** functional safety, working with the HRI/perception team that owns the depth
  pipeline (SYSTEM_DESIGN.md §5.1) — QA & functional safety literally owns catalog domain 31 and HRI
  safety in that org map; this project's advisory metric is exactly the kind of thing that team
  evaluates (and would never accept as the certified layer itself).

## The algorithm in brief

- **Top-down depth rendering + robot self-filter** — exact closed-form top height for
  horizontal/vertical capsules (never approximate — a deliberate scoping constraint), classified
  BACKGROUND / ROBOT / HUMAN by comparing against the robot's own known pose. → [THEORY.md](THEORY.md)
  §The math, §The GPU mapping.
- **Point-capsule distance** (derived from first principles: minimize a convex quadratic in the
  segment parameter, clamp to [0,1]) — the per-human-pixel candidate distance. → THEORY.md §The math.
- **Minimum-distance field**: map (per-pixel candidate) + canonical shared-memory tree **reduce** to
  the frame's d_min, and its closest robot capsule. → THEORY.md §The GPU mapping.
- **Dense clearance field**: the same distance function, applied to every pixel, for the visual
  artifact the catalog bullet names. → THEORY.md §The algorithm.
- **ISO/TS-15066-*style* protective separation distance S_p**, evaluated at two robot speeds to get
  a two-boundary (three-zone) state machine, with escalate-immediately / de-escalate-after-hysteresis
  asymmetry. → THEORY.md §The math, §The algorithm (the false-stop/missed-stop argument).
- **Segment-segment (capsule-capsule) distance** (Ericson's / Lumelsky's classic closed-form
  algorithm, reimplemented didactically) — the independent, pixel-free analytic ground truth every
  verification gate is checked against. → THEORY.md §The math, §How we verify correctness.

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/speed-and-separation-monitoring.sln`](build/speed-and-separation-monitoring.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/speed-and-separation-monitoring.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none** — CUDA toolkit libraries + C++17 standard library
only. No cuBLAS/cuFFT/cuRAND/Thrust: every kernel here is a hand-rolled map or map+reduce over a
tiny, fixed capsule list — exactly the kind of computation this repo prefers to hand-roll rather than
reach for a library (CLAUDE.md §5).

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU
*and* the closed-form ground truth):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) — including how
to read the `distance_field.pgm` and `ssm_timeline.csv` artifacts.

## Data

The committed sample is a **scenario**, not a recording: `data/sample/ssm_scenario.csv` (613 bytes,
synthetic, no RNG anywhere in this project) — 240 frames at 30 Hz, and the human's walk start/
turnaround points. Every depth pixel, robot pose, and human capsule position is synthesized in
closed form at run time from that tiny scenario plus the compile-time model in
[`src/kernels.cuh`](src/kernels.cuh) (the same synthetic-first, data-vs-model split project 08.01
uses). No public dataset applies to a synthetic SSM teaching pipeline; `scripts/download_data.ps1` is
an honest no-op. Details, checksum, and field documentation: [`data/README.md`](data/README.md).

## Expected output

Eleven stable lines — banner, `NOTICE:`, `PROBLEM:`, `SCENARIO:`, `VERIFY:`, `ARTIFACT:`, four
`GATE ...:` lines, and `RESULT:` — checked as a subset diff by
[`demo/expected_output.txt`](demo/expected_output.txt). **Five** independent verifications happen
before `RESULT:` is printed:

1. **The §5 GPU-vs-CPU gate (`VERIFY:`)**: at two frames (start and the scenario's designed
   midpoint), all three kernels' outputs are compared against `src/reference_cpu.cpp`'s independent
   oracle — labels must match **exactly** (measured: 0 mismatches over 40,000 pixels × 2 frames);
   depth, d_min, and the dense field agree within tolerance 1e-4 m (measured worst: 1.55e-6 m depth,
   1.19e-7 m d_min, 3.58e-7 m dense field — FP32 rounding only).
2. **`GATE NO-FALSE-STOP`**: on every frame where the closed-form (pixel-free) distance exceeds
   S_p_full by more than a documented margin, the pipeline must show NORMAL — measured: 139/240
   qualifying frames, 0 violations, tightest margin actually tested 0.054 m.
3. **`GATE NO-MISSED-STOP`**: on every frame where the closed-form distance is below S_p_reduced by
   more than a documented margin, the pipeline must show PROTECTIVE_STOP — measured: 65/240
   qualifying frames, 0 violations, tightest gap actually tested 0.002 m.
4. **`GATE TRANSITIONS`**: all 4 state transitions (NORMAL→REDUCED→PROTECTIVE_STOP on approach,
   PROTECTIVE_STOP→REDUCED→NORMAL on retreat, the second pair delayed by the documented hysteresis
   hold) must land within ±1 frame of the closed-form S_p crossing — measured: worst observed offset
   0 frames (exact, every transition), at frames 74, 82, 163, 171 of 240.
5. **`GATE D_MIN BOUND`**: the pixel pipeline's d_min must sandwich the closed-form distance from
   above by no more than a *derived* bound (pixel quantization + a second, larger "top-down
   silhouette visibility" term this project's own development surfaced — THEORY.md "Numerical
   considerations" tells that story) — measured worst-case overestimate: 0.029 m, comfortably inside
   the derived 0.070 m bound.

`RESULT: PASS` only when all five hold. Success thresholds carry real headroom (2–1000×, depending on
the check) so ordinary FP32/platform differences cannot flip the verdict; every measured number above
is printed on an `[info]` line every run, not asserted silently.

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — the whole project's contract in one file: the capsule
   representation and its horizontal/vertical scoping constraint (SECTION 1), cell/camera geometry
   (SECTION 2), the SCARA robot model (SECTION 3), the human model and why its torso radius is what
   it is (SECTION 4), the state machine (SECTION 5), the S_p formula (SECTION 6), and the
   verification gate parameters with their derivations (SECTION 7). Read this first.
2. [`src/main.cu`](src/main.cu) — orchestration: scenario loading, `build_scene()` (the shared FK +
   human-path generator), the `analytic::` namespace (the independent closed-form ground truth,
   including the segment-segment distance derivation), the SSM hysteresis state machine, the VERIFY
   stage, the SEQUENCE stage, and the four gates. The single most interesting thing to look at: the
   `analytic::` namespace runs in *double* precision, deliberately more precise than the FP32
   pipeline it checks.
3. [`src/kernels.cu`](src/kernels.cu) — the three GPU kernels: `render_classify_kernel` (map),
   `human_min_distance_kernel` (map + canonical shared-memory tree reduce), and
   `dense_distance_field_kernel` (map, reusing the same device distance function).
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the independent CPU oracle twin of the three
   kernels above; diff it against `kernels.cu` to see exactly what parallelization changed (spoiler:
   almost nothing — the geometry math is identical, only the loop became threads).
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers (copied, not shared — §4 rule).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **ISO/TS 15066** (and ISO 10218's collaborative-operation annex) — the actual standard this project's
  S_p formula is structurally *inspired by*; this project is not a reproduction of it and is not a
  substitute for reading it. See `PRACTICE.md` §4 for the orientation, not compliance, framing.
- **SICK, Pilz, and other certified safety-scanner/area-monitoring vendors** — what actually occupies
  this project's slot on a real collaborative cell: certified sensors, redundant channels, a
  safety-rated controller. `PRACTICE.md` §2 compares their category to this project's synthetic
  camera honestly.
- **Ericson, "Real-Time Collision Detection"** (§5.1.9, closest point between two segments) — the
  segment-segment distance algorithm `main.cu`'s `analytic::closest_seg_seg_distance` reimplements
  didactically; also the standard reference for capsule/swept-sphere collision primitives generally
  (the same primitive PCL, MoveIt/FCL, Drake, and cuRobo all use for link/obstacle clearance).
- **nvblox / cuRobo** — production GPU-accelerated distance/clearance fields for manipulation; this
  project's dense clearance-field artifact is a tiny, brute-force, fully-transparent teaching version
  of the same idea (SDF/ESDF-style fields), computed by direct point-capsule evaluation instead of a
  voxel grid or GJK/EPA-based solver.
- **Project 31.01** (Hamilton-Jacobi reachability) and **31.04** (CBF safety filters) — this repo's
  other safety-adjacent didactic metrics; all three share the identical caveat (sim-validated,
  didactic, not certified) and the identical "advises, does not act" relationship to a real safety
  chain (SYSTEM_DESIGN.md §6.1).

## Exercises

1. **Plot the artifact:** `demo/out/ssm_timeline.csv` → `d_min` vs `t_s`, with `sp_full_m` and
   `sp_reduced_m` as horizontal reference lines. Identify the four transition frames by eye and
   compare them to the `[info] transition N: ...` lines the demo prints.
2. **Break the hysteresis:** set `kHysteresisHoldFrames` (kernels.cuh) to 1 and rebuild. Explain from
   `HysteresisFsm::step`'s logic why the `GATE TRANSITIONS` predicted-frame formula still has to
   change to match (hint: it is not just "subtract 4").
3. **Make the torso win again:** raise `kHumanTorsoRadius` back toward 0.22 m and rerun. Watch
   `GATE D_MIN BOUND` fail by tens of centimeters, and read the `[info] d_min bound:` line's "worst
   observed offset" — then read the comment above `kHumanTorsoRadius` in `kernels.cuh` for why.
4. **Derive the general silhouette-sag bound:** `kSilhouetteSagBound` (kernels.cuh SECTION 7) is
   calibrated to *this* scenario's known height gaps (≤ 0.05 m). Derive a bound that depends on the
   actual height difference between the human capsule and whichever robot capsule turns out to be
   closest, rather than a fixed constant — THEORY.md "Numerical considerations" sketches where to
   start.
5. **Climb toward a real sensor:** replace the synthetic orthographic camera with a perspective model
   (a pinhole camera looking down at an angle, not straight down) and see which of this project's
   "exact" rendering claims (kernels.cuh SECTION 1) stop holding, and why THEORY.md flags perspective
   as future/exercise territory rather than baseline scope.

## Limitations & honesty

- **Didactic ISO/TS-15066-*style* metric, not a certified implementation, ever.** The S_p formula
  matches the standard's published *term structure* with illustrative, rounded, scenario-fit
  constants — never the standard's actual defaults, never a substitute for reading (and, for real
  work, licensing) the standard itself. See `PRACTICE.md` §4.
- **Single overhead camera, orthographic, noiseless.** A real depth camera is perspective, noisy, and
  ideally not alone (multiple/side viewpoints fuse to see what one overhead view cannot). This
  project's own numbers make the cost of that simplification concrete: the silhouette-visibility
  effect in `GATE D_MIN BOUND` is a *direct, measured consequence* of "a top-down-only camera cannot
  see a standing capsule's side" — a real limitation, not hidden, turned into the project's most
  interesting numerical finding (THEORY.md "Numerical considerations", Exercise 3).
- **Capsules are horizontal or vertical, by construction, never tilted.** This makes the top-down
  renderer exact in closed form instead of needing an iterative solve; a tilted-link renderer is
  real, harder, and left as Exercise 5 / future work.
- **The robot's "known pose" is exactly true.** Self-filtering against a real robot's calibration and
  encoder uncertainty needs a nonzero, tuned tolerance band; here the "known" pose IS the true pose,
  so `kSelfFilterEps` is a placeholder magnitude, not a calibrated one. `PRACTICE.md` §1 discusses the
  real source of that uncertainty.
- **Advisory only, never a stop authority.** This project's output is a *state*
  (NORMAL/REDUCED/PROTECTIVE_STOP), never a command that reaches real hardware; nothing here is
  wired to actuate anything, and it never should be without an independent, certified safety chain
  (SYSTEM_DESIGN.md §6.1) between this code and any motor. **Sim-validated only, not
  safety-certified — CLAUDE.md §1, §8, at full strength**, because this is precisely the class of
  project whose whole subject matter is human safety near a moving machine.
