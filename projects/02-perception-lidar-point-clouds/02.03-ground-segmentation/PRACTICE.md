# 02.03 — Ground segmentation: RANSAC plane fit; Patchwork++-style GPU port: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

This project is pure perception software — there is no physical part *of this algorithm* to build. What
does need physical construction is the thing this algorithm's assumption is *about*: the ground itself,
and the sensor mount that looks at it. Both matter enough to a real deployment that they belong here.

**Where the "ground is one (or a few) planes" assumption breaks, physically:**

- **Curbs and steps.** A curb is a vertical discontinuity a few centimeters to ~15 cm tall — shorter
  than most obstacles this project's scene models, and geometrically almost indistinguishable from the
  "obstacle base" ambiguity this project's `obstacle_rejection` gate already measures (8.96% CZM
  false-positive rate on standing obstacles' base rims). A curb *below* a robot's ground-clearance
  threshold should be traversable; one above it should not — but a pure height-threshold classifier like
  both of this project's milestones cannot make that distinction on its own. Real systems add a
  *dedicated* curb/step detector (a height-jump filter along range, not just a plane-fit residual) as a
  second-stage classifier layered on top of ground segmentation.
- **Grass and vegetation.** Grass is not a rigid surface — individual blades scatter LiDAR returns across
  a height BAND (a few cm to tens of cm) rather than a clean plane, so the "flatness" test
  (`kCzmFlatnessMaxRmsM`) that correctly rejects an obstacle also rejects a genuinely-driveable grassy
  patch on outdoor/field robots. Field robotics systems (see [30.01](../../30-field-robotics/README.md))
  typically widen the flatness tolerance for known-outdoor operating regions or fuse in a camera-based
  vegetation classifier.
- **Rain-wet and reflective surfaces.** A wet road or a polished warehouse floor can specularly reflect a
  LiDAR beam, producing a "virtual" return from *below* the true ground plane (a mirror-image reflection
  off whatever is beyond the wet patch) — physically a multi-path phenomenon, not sensor noise. Patchwork
  ++'s Reflected Noise Removal (RNR) stage exists specifically to detect and discard these; this
  project's teaching version has no such stage (THEORY.md "Where this sits in the real world").
- **Negative obstacles (the pothole problem).** A pothole, an open manhole, a downward stairwell, or a
  loading-dock edge is a *hole* in the ground plane — the opposite failure mode from an obstacle: instead
  of an unexpected RETURN above the plane, there is an unexpected ABSENCE of return (or a return from
  much farther away, at the bottom of the hole or beyond it) at a range/azimuth where the ground model
  predicts a normal return. Neither of this project's milestones models this: both classify strictly from
  the returns that exist, never reasoning about a *missing* return. A real system pairs ground
  segmentation with an explicit negative-obstacle detector (commonly: compare the observed range at each
  azimuth against the range the fitted local ground plane predicts, and flag a large positive range
  discrepancy — "the ground should be HERE, but the beam went further").

**Sensor mounting height and angle — the trade this project's scene makes concrete.** This project's
scene mounts its (simulated) sensor at 1.5 m and finds that only 5 of 16 beam elevations actually reach
the floor within a useful range (THEORY.md "The problem" derives `r(θ) = H/sin|θ|`). Mounting *higher*
increases every ground-ring's range (worse near-field coverage, the exact gap this project's canopy
placement had to route around — see `scripts/make_synthetic.py`'s module docstring); mounting *lower*
brings ground rings closer in but raises the risk of the sensor itself being below obstacle height (an
AMR's own body can occlude downward beams). Real platforms split the difference with **multiple**
LiDAR units at different heights/angles specifically to densify near-field ground coverage — exactly the
gap a single-sensor version of this scene exposed.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-11. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

Ground segmentation runs on whatever compute tier already hosts the rest of a robot's perception stack
— it is a *cheap* stage (this project measures ~6 ms for RANSAC full-scene, ~1 ms for the CZM, on a
desktop RTX 2080 SUPER) relative to the sensor and compute that feed it, so the interesting hardware
choices are upstream (the LiDAR) and downstream (where this code physically runs).

| Tier | Compute (where this code runs) | LiDAR (what feeds it) | Rough cost tier |
|------|-------------------------------|------------------------|------------------|
| Hobby/research | Jetson Orin Nano / NX (8–16 TOPS-class, ~15–25 W) | 16-beam mechanical spinning LiDAR (e.g., a Velodyne VLP-16-class or RoboSense RS-16-class unit — this project's own beam table is modeled on exactly this beam count) | Compute ~$200–500; LiDAR ~$2,000–4,000 |
| Prof./mid-volume AMR | Jetson Orin AGX or an embedded x86 + entry discrete GPU | 32- to 64-beam mechanical spinning LiDAR, or a solid-state/MEMS unit with a denser near-field pattern | Compute ~$1,000–2,000; LiDAR ~$4,000–10,000 |
| AV-grade / industrial | x86 server-class + data-center-class discrete GPU (redundant pair) | Multiple 128-beam-class spinning units + solid-state units for near-field blind-spot coverage | Compute several $1,000s; LiDAR $10,000s per unit, multiple units per vehicle |

**Why LiDAR beam count matters more here than compute tier:** THEORY.md's beam-ring analysis shows that
*this project's own 16-beam table leaves large near-field ground gaps* — a real 16-beam sensor has
exactly the same limitation, which is precisely why production ground-truth-critical systems (AV-grade
especially) use 32-, 64-, or 128-beam sensors, or add solid-state/MEMS units specifically for dense
near-field coverage, rather than trying to compute their way out of a sparse-beam sensor's fundamental
angular resolution limit.

**No dedicated silicon.** Unlike the actuation-chain projects in this repository, ground segmentation has
no motor-control MCU, gate driver, or current-sense amplifier to speak of — it consumes a `PointCloud`
message and produces a `ground` mask, entirely in software on the perception compute tier above.

## 3. Installation & integration — putting it on a real robot

**Where this code would physically run.** The same compute unit that runs the rest of the point-cloud
perception pipeline (LiDAR driver → downsampling → ground segmentation → clustering/costmap) — typically
the robot's main perception computer (Jetson-class or x86+dGPU, per §2), not a separate microcontroller,
since it needs the CUDA-capable GPU the rest of that pipeline already uses.

**OS and real-time constraints.** Ubuntu + ROS 2 is the overwhelmingly common choice for this layer of
the stack (unlike a motor-control loop, ground segmentation does not need a hard real-time OS — a
10–20 Hz soft deadline, generously met by this project's measured sub-10ms runtime, is well within a
standard Linux + ROS 2 executor's jitter budget).

**The ROS 2 node/topic shape.** A ground-segmentation node sits between the LiDAR driver (or a
downsampling node) and every consumer named in README "System context":

```
/lidar/points_downsampled  (sensor_msgs/PointCloud2)
        │
        ▼
  [ ground_segmentation_node ]   <- this project's algorithm, wrapped as a ROS 2 node
        │
        ├──▶ /ground/points          (sensor_msgs/PointCloud2 — ground-only, for costmap/traversability)
        └──▶ /nonground/points       (sensor_msgs/PointCloud2 — for 02.04-style clustering)
```

A production node typically publishes the classification as a `pcl::PointIndices`-style index list or an
extra `PointField` (an "is_ground" channel appended to the cloud) rather than two full copies of the
cloud, to avoid doubling bandwidth on a topic that can carry hundreds of thousands of points per second.

**Parameter tuning per platform — the curb-height decision.** The single most consequential tuning knob
this project exposes (`kCzmClassifyDistM`, `kCzmFlatnessMaxRmsM`, `kCzmUprightMaxDeg` in `kernels.cuh`)
is exactly the knob real deployments retune per PLATFORM, because different platforms have different
ground-clearance and curb-crossing capability:

- A **sidewalk delivery robot or power wheelchair** has a low ground clearance and cannot cross even a
  modest curb — it *wants* a tight classify threshold (obstacles as short as a few centimeters must
  register as non-ground) and treats any curb as a hard obstacle, accepting the loss of some driveable
  area at curb transitions.
- A **warehouse forklift-class AMR or an off-road utility vehicle** has real ground clearance and
  suspension travel — it can safely cross a curb-height discontinuity that would stop the wheelchair, so
  its ground-segmentation parameters (or a downstream traversability-costmap stage that CONSUMES this
  project's output — see [14.02](../../14-locomotion-wheeled/14.02-traversability-costmaps-fusing-semantics/README.md))
  are tuned to admit a wider height band as "still traversable."

This is a genuinely PLATFORM-SPECIFIC decision, not a universal "more accurate" vs. "less accurate"
tuning axis — the same physical curb is correctly obstacle for one robot and correctly traversable for
another.

**Sensor calibration and bring-up.** Ground segmentation is directly sensitive to LiDAR **extrinsic**
calibration (the sensor's mounting height and tilt relative to the robot's base frame) — a few degrees
of unmodeled tilt shifts the "true vertical" this project's uprightness test (`kCzmUprightMaxDeg`)
compares against, and a wrong mounting-height offset shifts every fitted plane's `d` (offset) parameter.
Bring-up procedure: (1) mount the sensor and measure its extrinsic transform (height, tilt) by hand or
with a calibration target; (2) run ground segmentation on a KNOWN-flat calibration surface and check the
fitted plane's normal is within a small tolerance of true vertical and its offset matches the measured
mounting height (exactly the `ransac_flat` gate's check, on real hardware instead of a synthetic scene);
(3) iterate the extrinsic calibration until it does.

**The safe hardware-testing ladder (CLAUDE.md §1 caveat applies in full).** Everything in this repository
is sim-validated only, never safety-certified. Before any real-hardware use of a ground-segmentation
output to command motion: (1) **simulation** — exactly this project's synthetic-scene validation; (2)
**HIL (hardware-in-the-loop)** — real sensor, recorded or live data, output compared against hand-labeled
ground truth, no motion commanded; (3) **bench jig / tethered / current-limited** — the robot is
physically restrained or power-limited while the FULL perception→planning→control loop runs closed-loop
for the first time; (4) **free running**, only after (1)–(3) pass with margin, with a hardware E-stop and
software safety limits active at every rung. Never skip a rung.

## 4. Business & regulatory context

*Didactic orientation only — see the disclaimer at the end of this file.*

**Who needs this, and where.** Ground segmentation is a near-universal perception primitive: every
LiDAR-equipped mobile robot needs *some* answer to "what is drivable," from a $500 hobbyist rover to a
Class-8 autonomous truck. It sits squarely inside the **Perception** team's ownership
(SYSTEM_DESIGN.md §5.1's org map) — usually the same engineers who own point-cloud preprocessing and
hand results to Planning/Controls for costmap consumption, with close collaboration with the Simulation
& Tools team (who own the synthetic-scene test infrastructure this project's own demo is a miniature
version of).

**Main players.** Open-source: PCL's `SACSegmentation`, Autoware's ground-filter packages, and the
Patchwork/Patchwork++ research releases this project's Milestone 2 is modeled on (all cited in README
"Prior art"). Commercial: every AV stack (Waymo, Cruise-successors, Zoox, and Chinese AV players) and
every warehouse-AMR vendor (Locus Robotics, 6 River Systems, Fetch/Zebra-class platforms) implements some
version of this stage internally — it is rarely a stand-alone product, more a foundational module inside
a larger perception stack.

**What getting it wrong costs.** For a low-speed indoor AMR, a ground-segmentation error typically costs
a stalled mission (the robot refuses to cross a false-positive "obstacle," or nudges an actual small
obstacle it should have avoided) — a downtime and productivity cost, not usually a safety incident,
BECAUSE these platforms move slowly and are usually further protected by independent safety-rated
laser scanners for the actual stop/slow decision (see [21.04](../../21-hri-teleoperation/21.04-speed-and-separation-monitor-ur-cobot-style/README.md)-style
speed-and-separation monitoring). For an outdoor AV, the stakes are categorically higher: misclassifying
a curb, a pothole, or a road edge as driveable (or vice versa) is a genuine safety hazard at highway
speed, which is why ground/road-surface segmentation is treated as safety-relevant perception in AV
functional-safety cases, not a convenience feature.

**Regulatory path (orientation, not compliance guidance — cite SYSTEM_DESIGN.md §6.2's regulatory map).**
For an indoor/warehouse AMR, ISO 13482 (personal-care/service robot safety) and the broader industrial
mobile-robot safety literature apply to the ROBOT's overall safety case, of which ground segmentation is
one contributing input, never the sole safety mechanism (independent range-sensor-based safety fields
are the actual certified stop/slow mechanism on most commercial AMRs today). For an autonomous vehicle,
ISO 26262 (functional safety) and UL 4600 (safety case methodology for fully autonomous products) govern
how a perception module's failure modes must be analyzed, tested, and argued safe — ground/road-surface
segmentation accuracy is exactly the kind of measured-and-margined evidence (this project's own gate
methodology, at toy scale) a real UL 4600 safety case would need to document, with orders of magnitude
more scenario coverage and real-world validation than one synthetic scene.

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
