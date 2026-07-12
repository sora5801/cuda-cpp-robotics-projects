# 02.14 — Moving-object segmentation from sequential scans: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

This project is pure software running on data a real spinning LiDAR already produces — there is no
new mechanical part to build. The physical carrier worth describing is the LiDAR unit itself and its
mount, since this project's whole input (posed, organized scans at native `(ring, azimuth)` resolution
— README "System context") is exactly what that unit's driver reports.

A mechanical spinning LiDAR (the sensor model this project's 16-beam geometry is patterned on, e.g. a
Velodyne/Ouster-class VLP-16 or OS1-16 equivalent) is a sealed rotating assembly: a laser/receiver
stack spins on a bearing at a fixed RPM (10 Hz = 600 RPM is the common default, matching this
project's implicit assumption that "one scan" = one full 360-degree sweep), with power and data
crossing the rotating joint via a slip ring or, in solid-state/MEMS designs, no moving parts at all
(a fixed array steered electronically). Mounting matters directly to this project's algorithm: the
mount must be RIGID (any flex between the sensor and the vehicle chassis it is "posed" relative to
directly corrupts the `T_world_sensor` pose this project's reprojection math assumes is exact —
THEORY.md "Numerical considerations" derives the pose-error-to-residual coupling), vibration-isolated
(a spinning mechanical assembly is itself a vibration source and a vibration receiver), and positioned
with a clear, unobstructed field of view matching the elevation range this project models
(-15..+15 degrees) — a sensor mounted too low or shadowed by the vehicle's own body would report the
robot's own chassis as a permanent "static" return in every scan, an artifact this project's synthetic
scene does not model but a real integration must handle (see §3).

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-12. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

**Compute this module would run on.** Online MOS at 10-20 Hz on a real sensor's full point count
(30k-500k points/scan, README "Limitations") is squarely GPU perception territory
(SYSTEM_DESIGN.md §1.2): the same five kernels this project ships, unchanged in shape, would run on:

| Tier | Example compute | Notes |
|------|------------------|-------|
| Embedded / production robot | NVIDIA Jetson Orin NX/AGX (Jetson-class SoC, integrated GPU) | The default for a warehouse AMR or a compact AV prototype (SYSTEM_DESIGN.md §2.1's "one embedded GPU SoC") — this project's kernels' tiny per-scan working set (kilobytes to low megabytes of range images) fits comfortably in an Orin's shared memory. |
| Desktop / dev workstation | x86 + a discrete RTX-class GPU (e.g. RTX 2080/4070-class, this repo's reference machine) | Where this project's own demo runs; also a common AV prototype/dev-vehicle compute tier before embedding. |
| Research / this repo's floor | Any sm_75+ (Turing or newer) CUDA GPU | CLAUDE.md §5's ratified floor; this project's `CodeGeneration` list targets sm_75/86/89 plus PTX fallback. |

**The sensor itself.** A 16-32-beam mechanical spinning LiDAR (illustrative examples: Velodyne
VLP-16/Puck-class, Ouster OS1-32-class, Hesai XT-class) connects over Gigabit Ethernet (UDP packets,
one per firing group) or, on older/lower-cost units, a proprietary serial link; a solid-state/MEMS
unit (e.g. Livox-class, Hesai AT-class) trades the mechanical spin for a smaller, sometimes irregular
(non-repeating) scan pattern that would need this project's `(ring, az_bin)` binning replaced with a
direct spherical-coordinate binning (README Exercise territory, not implemented here). Illustrative
cost tiers: hobby/research 16-beam units in the low thousands of USD; automotive-grade 32-128-beam
units used in AV stacks range from several thousand to tens of thousands of USD depending on range,
resolution, and automotive qualification.

**Nothing else is specific to this module.** Unlike an actuator or power-electronics project, MOS has
no motor-control silicon, gate drivers, or current-sense chain of its own — its only hardware
dependency is the LiDAR (above) and the compute tier that runs its perception stack (above); it
consumes whatever pose stream the localization stack already produces (§3).

## 3. Installation & integration — putting it on a real robot

**Where this runs.** On the SAME compute that runs the rest of the point-cloud perception pipeline
(README "System context": right after ground-plane removal / organization, domain 02, and before
tracking/mapping, domains 04/05) — typically a Linux process (or ROS 2 node) on the robot's main
perception compute, NOT a real-time MCU: this module's 10-20 Hz budget (README) is well within Linux's
soft-real-time capability with a reasonably isolated process, unlike the kHz control loops that
require a dedicated real-time OS or MCU (SYSTEM_DESIGN.md §1.1).

**ROS 2 node shape (illustrative).** A `moving_object_segmentation` node subscribing to a
`sensor_msgs/PointCloud2`-shaped organized scan (this project's `PointCloud` message-shape convention,
SYSTEM_DESIGN.md §3.6) PLUS the robot's current pose estimate (a `nav_msgs/Odometry`-shaped
`T_world_sensor`, README "System context"'s upstream input), internally buffering the last
`kMaxWindowM` scans, and publishing TWO outputs matching this project's two downstream lanes: a
`PointCloud2`-shaped "movers" topic (feeding a tracker, domain 04.xx) and a `PointCloud2`-shaped
"statics" topic (feeding a mapper, domain 05.xx) — exactly the fork README "System context" names.

**Bus / transport.** The LiDAR itself typically talks Ethernet/UDP (not CAN-FD or EtherCAT — those are
the actuation-side buses this project never touches); this module's OWN input/output, once inside the
robot's compute, is intra-process or DDS (ROS 2's default middleware) traffic, not a fieldbus.

**Calibration and bring-up dependency — the honest coupling.** This module's accuracy is BOUNDED by
two upstream calibrations it never re-derives: (a) the LiDAR-to-body extrinsic calibration (so that
"the sensor's pose" and "the robot's pose" are related by a known, fixed transform), and (b) the
localization/odometry stack's own accuracy (THEORY.md "Numerical considerations" derives the direct
pose-error-to-false-positive coupling — this is the same feedback-loop honesty **02.07**'s ground-plane
removal project states for its own upstream dependency: a perception module this far down the stack
inherits every upstream error). Bring-up practice: verify LiDAR-to-body extrinsics with a static
calibration target BEFORE trusting this module's output; if `static_precision`-equivalent false
positives appear on a real robot's genuinely static structure, the FIRST suspect is localization
drift, not this module's threshold.

**The safe testing ladder (CLAUDE.md §1 applies in full — nothing here is safety-certified).**

1. **Simulation** — exactly this project's demo: synthetic scans with known ground truth, gating
   correctness before any hardware is involved.
2. **Recorded-data replay (a light HIL step)** — feed this module logged scans from a real LiDAR on a
   real robot (stationary or teleoperated, no autonomous motion), comparing its mover/static split
   against a human-labeled or a slower offline reference (e.g. this repo's own **02.13**, which has the
   luxury of the FULL recording to work from) — the first point real sensor noise, real disocclusion
   geometry, and real localization drift enter the picture.
3. **Bench / tethered, output NOT wired to control** — run this module live on the robot's own compute
   while the robot is stationary or manually driven, logging its output for offline grading, with the
   output explicitly NOT connected to any planner or map yet.
4. **Closed-loop, low-speed, geofenced, E-stop in reach** — only once (2) and (3) show acceptable
   mover/static separation does this module's output get wired into a real tracker/mapper, and only
   with an independent safety monitor (SYSTEM_DESIGN.md's cross-cutting "Safety Monitor" layer, domain
   31) watching for the failure modes named in §4 below.

## 4. Business & regulatory context

**Who needs this, and why it matters commercially.** Any robot operating **among people or other
moving agents** needs SOME answer to "what around me is moving right now" before it can plan safely —
this is the shared need behind SYSTEM_DESIGN.md's warehouse-AMR and autonomous-vehicle reference
robots (README "System context"), and it is a load-bearing PERCEPTION capability, not a nice-to-have:
a planner that treats every obstacle as static will either freeze constantly (over-cautious) or,
worse, fail to anticipate a mover's future position (under-cautious) — SYSTEM_DESIGN.md's Prediction
layer literally cannot do its job without a mover/static split feeding it.

**Getting it wrong costs something different in each direction — the central business/safety
tension.** This project's own gates measure both failure modes explicitly, and the reason they are
gated SEPARATELY (`mover_detection` recall vs. `static_precision` false-positive rate), not combined
into one score, is that they have DIFFERENT, ASYMMETRIC costs downstream:

- **A static point misclassified as moving** vanishes from the map (README "System context"'s
  "statics -> mapping" lane never receives it) — a permanent wall or shelf can be silently dropped
  from the robot's world model, and enough of this failure mode degrades localization and collision
  avoidance against structure the robot should have known about.
- **A moving point misclassified as static** becomes a PHANTOM STATIC OBSTACLE at best (the map now
  contains a "wall" where a car briefly stood, until the offline cleanup this repo's **02.13** would
  eventually perform) — or, more seriously, the tracker/prediction layer never receives it as a mover
  at all, so the planner has NO forecast for where that object will be next. In an AMR warehouse or an
  AV stack operating among people, this is the more dangerous direction: a genuinely moving
  pedestrian or vehicle that this module fails to flag is invisible to every downstream safety
  mechanism that depends on motion classification.

**Regulatory path (orientation only, SYSTEM_DESIGN.md §6.2 — never compliance guidance).** This
module's two reference robots sit in different regulatory rows of that table: a warehouse AMR falls
under **ISO 13482** (service robots sharing space with untrained people); an autonomous-vehicle stack
falls under **ISO 26262** (functional safety of E/E systems) and, for higher autonomy levels, **UL
4600**'s safety-case framework. Neither standard certifies a perception ALGORITHM in isolation — they
certify the SYSTEM's argued, evidenced safety case, of which a module like this one's measured
recall/precision (README "Expected output") would be ONE input among many (sensor redundancy, safety
monitors, fallback behaviors). Nothing in this repository constitutes such a certified system.

**Where this work lives inside a robotics company.** Perception (SYSTEM_DESIGN.md §5.1: domains
01/02/03/20) owns and tunes this module; typical role titles are "Perception Engineer" or "LiDAR
Perception Engineer." Its two direct downstream consumers — Controls & Autonomy's tracking/prediction
work and mapping work — are the team's closest collaborators (the same team, per SYSTEM_DESIGN.md
§5.1, that owns domains 04-09/23); QA & Functional Safety (domain 31, §5.1) is the adjacent team that
would define and evaluate the acceptance thresholds a shipped version of this module's gates would
need to clear.

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
