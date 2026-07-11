# 01.17 — Camera-LiDAR / camera-camera extrinsic calibration (batched reprojection-error optimization): Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

*Section dated 2026-07-11.*

The physical carrier this code serves is a **calibration cell**: a fixture that presents a known target
to a sensor rig from many controlled poses. Two target constructions matter here, because this project's
two scenarios need different physical targets:

- **The camera-camera / camera-only target** — a flat board (aluminum composite panel or rigid
  fiberglass, chosen for flatness under temperature and handling rather than raw stiffness) printed or
  laser-etched with a **matte, high-contrast pattern** (checkerboard or ChArUco — 01.16's target). Matte
  finish matters: a glossy print produces specular highlights that move as the camera or lights move,
  corrupting sub-pixel corner detection; calibration targets are printed on matte photo stock or etched
  (not printed) into anodized aluminum for a durable, glare-free surface.
- **The camera-LiDAR target** — this project's 4-fiducial board additionally needs the fiducials to be
  **LiDAR-visible**, which a flat printed dot is NOT: a matte-black printed circle reflects almost no
  laser return, indistinguishable from the board's edge in a point cloud. Real camera-LiDAR targets use
  **retroreflective material** (3M Scotchlite-class sheeting, the same material as road signs) cut into
  circles or squares at the fiducial locations, mounted flush on the matte board — the retroreflector
  returns a strong, easily-thresholded LiDAR intensity spike exactly where the camera also sees a
  high-contrast marker, giving BOTH sensors a confident detection of the SAME physical point. (Some
  designs instead cut holes through the board at the fiducial locations, giving the LiDAR a depth
  discontinuity to detect instead of an intensity spike — a materials/detection-algorithm tradeoff a
  real calibration-cell designer chooses based on their LiDAR's return-intensity fidelity.)
- **Rigid fixturing.** The target is mounted on a stand (a camera tripod for a bench setup, or a
  purpose-built calibration frame for a factory cell) that must not flex or vibrate between the moment a
  pose is "locked in" and the moment both sensors capture it — any relative motion between the LiDAR
  scan and the camera frame's exposure directly corrupts a correspondence, and a nonrigid stand is a
  common source of calibration-cell teething problems.
- **Presenting the poses.** V=12 (this project's number) or more distinct poses need to be produced.
  Three tiers, cheapest to most repeatable: (a) an operator manually moving the board by hand through a
  rough script of positions/tilts — cheap, but pose diversity and repeatability depend entirely on
  operator care (THEORY.md's degeneracy lesson makes clear why sloppy presentation directly costs
  accuracy); (b) a motorized **turntable** plus a manually-adjustable tilt bracket — repeatable rotation,
  manual tilt diversity; (c) a **robot arm presenting the board** (the same idea SYSTEM_DESIGN.md §2.2's
  manipulator work cell embodies) — fully programmable, repeatable pose sequences, and the only tier that
  can guarantee the kind of DELIBERATE pose diversity (wide depth AND orientation spread) this project's
  synthetic "diverse" cohort models; a factory calibration cell for a production AV line almost always
  uses this tier.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-11. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

This project's OWN compute footprint is trivial (a fixed 6x6 solve, a 48-correspondence reduction) — the
"real hardware" question here is really about the CALIBRATION CELL and the sensors being calibrated:

| Tier | Compute | Target/fixture | Notes |
|------|---------|-----------------|-------|
| Hobby/bench | A laptop with any CUDA GPU (this project's own reference RTX 2080 SUPER is well past what is needed — even a GTX 1650-class part suffices) | A printed checkerboard on foamcore, hand-held or on a photo tripod (~$30-80) | Fine for learning and single-unit robot calibration; no retroreflective fiducials needed if only camera-camera is being calibrated |
| Research/small-fleet | A desktop workstation (RTX 4000-class) driving a capture rig with a motorized pan-tilt or turntable (~$200-600 for a hobby turntable, ~$1000+ for a repeatable research-grade one) | Laser-cut acrylic or aluminum-composite board with retroreflective vinyl fiducials (3M Scotchlite-class sheeting, roughly $10-30/sheet, illustrative) applied at measured positions | Good enough for a small robotics lab calibrating a handful of rigs per week |
| Industrial/factory | A fixed calibration-cell PC (often an industrial x86 box, sometimes with a Jetson Orin-class module if the cell also runs live vision QA) driving a 6-DoF robot arm (a small industrial arm, e.g. UR5e-class, ~$25-35k illustrative) that presents a precision-machined aluminum target board through a programmed pose sequence | CNC-machined aluminum board, positions of retroreflective/optical fiducials measured by a coordinate-measuring machine (CMM) to sub-0.1mm accuracy — the target's OWN geometry must be known far more accurately than the calibration this cell produces, or the target's uncertainty becomes the calibration's error floor | This is the tier a real AV or industrial-robot manufacturer runs at end-of-line; the GPU-accelerated solve this project teaches is the kind of software running on the cell PC |

The sensors BEING calibrated (not this project's own compute) span the usual perception-domain range:
automotive-grade cameras (GMSL/FPD-Link serialized, ~$50-300/unit illustrative) and mechanical or
solid-state LiDARs (from ~$100 hobby units to $1000s-$10,000s+ for automotive-grade time-of-flight
units) — see 01.xx/02.xx projects' own `PRACTICE.md` for the sensor-specific hardware detail this
project deliberately does not repeat.

## 3. Installation & integration — putting it on a real robot

*Section dated 2026-07-11.*

- **Where this runs.** Calibration is an OFFLINE (or periodic, out-of-loop) computation — it does not
  run on the robot's real-time perception computer during normal operation. In practice it runs either
  (a) on the calibration-cell PC at end-of-line (factory setting, one-time per unit), or (b) as a
  maintenance utility invoked ON the robot's own compute (SSH'd into, or run from a service laptop
  plugged into the robot's network) during field service — same software, different trigger.
- **Output: a static transform, not a topic.** The recovered `T_camera_lidar` (or `T_camera2_camera1`)
  is a CONSTANT for the robot's service life (until the next recalibration event) — in ROS 2 terms, it
  is published as a **static transform** via `tf2_ros::StaticTransformBroadcaster`, sourced from a
  calibration YAML file checked into the robot's configuration (not computed at every boot from scratch).
  The calibration file conventionally records the SE(3) transform as translation + quaternion
  (SYSTEM_DESIGN.md §3.4's `(w,x,y,z)` order, restated at this boundary too) plus metadata: calibration
  date, target used, reprojection RMS achieved, and the tool/version that produced it — exactly the
  audit trail a manufacturing QA process (§4 below) needs.
- **Bring-up / calibration procedure on a real cell.** (1) mount sensors rigidly and torque fasteners to
  spec (loose fasteners are the vibration-drift failure mode THEORY.md names); (2) capture the target
  across the programmed pose sequence, checking LIVE that each view's detections succeed before moving
  on (a failed detection silently dropped, rather than retried, quietly degrades the "12 views" this
  project's `kNumViews` assumes into fewer effective, possibly less-diverse ones); (3) run the solve;
  (4) **validate**, not just trust the solver's own reported residual — reproject a HELD-OUT target pose
  (one the solver never saw) and check its residual independently, the calibration-world equivalent of a
  train/test split; (5) write the calibration file and log the achieved accuracy.
- **Field validation, without a calibration cell.** A deployed robot cannot always drive back to a
  factory cell. A lightweight field check: place ONE known target (even a single retroreflective marker
  at a measured position) in the sensors' shared field of view, project it through the CURRENT stored
  extrinsic, and compare the reprojection error against a threshold — a degraded (but non-zero-effort)
  version of this project's own `RECOVERY_CAM_LIDAR`/`RECOVERY_CAM_CAM` gates, run against the real robot
  instead of synthetic data.
- **Recalibration triggers and monitoring.** THEORY.md quantified three drift causes (thermal, vibration,
  crash/service); the corresponding triggers are: (a) **scheduled** — periodic recalibration on a
  maintenance interval sized to the thermal/vibration drift RATE for the specific mount (a rigid,
  well-torqued industrial arm mount drifts far slower than a lightweight drone frame); (b)
  **event-based** — unconditional recalibration after ANY service touching a sensor mount, or after a
  detected collision/hard-stop; (c) **continuous** — a **reprojection-residual watchdog**: fuse LiDAR and
  camera data continuously using the stored extrinsic and monitor the RUNNING reprojection residual on
  naturally-occurring scene features (not a dedicated target); a residual that creeps upward over days
  is exactly the drift signal that should trigger an out-of-cycle recalibration, without waiting for the
  scheduled interval. This is the production-grade generalization of this project's own
  `NOISE_SCALING`/`DEGENERACY` gates: both show that reprojection residual is a sensitive, quantitative
  proxy for calibration health.
- **Safety ladder.** This project computes calibration parameters only — it never commands motion, so
  the usual sim -> HIL -> bench -> free-running ladder (CLAUDE.md §1) applies at one remove: a WRONG
  calibration does not move the robot itself, but silently corrupts every downstream perception/fusion
  result that DOES command motion. Treat a newly-computed calibration as untrusted until the held-out
  validation step above passes, exactly as you would treat a new planner or controller before letting it
  drive real actuators.

## 4. Business & regulatory context

*Section dated 2026-07-11. Didactic orientation only — not procurement, legal, or compliance advice.*

**Who needs this, and where the market sits.** Every company shipping a multi-sensor autonomous system —
AV manufacturers, warehouse/logistics robotics, agricultural and construction autonomy, and any
service/delivery robot with more than one exteroceptive sensor — needs extrinsic calibration as
standard manufacturing infrastructure (SYSTEM_DESIGN.md §5.2: this sits squarely in the EVT/DVT/PVT
hardware-validation stage-gate, alongside 01.16's intrinsic calibration, and recurs in fleet
operations as a service/recalibration task). Commercial and open-source players span the full stack:
**Kalibr** (open-source, the research-community default), vendor-specific factory tools built by AV
OEMs and Tier 1 suppliers (proprietary, tightly coupled to their own sensor suites), and general
machine-vision calibration suites (e.g., MVTec HALCON, Cognex VisionPro) that some manufacturers adapt
for camera-camera work.

**What getting it wrong costs.** A wrong extrinsic does not fail loudly — it fails as silently-degraded
fusion accuracy, exactly the "distance-accurate sensor overlaid in the wrong place" failure THEORY.md's
range-dependent error analysis quantifies. Downstream costs compound: false-negative obstacle detection
(a LiDAR return "painted" with the wrong camera pixel, misclassified), false-positive braking events (a
fused detection that does not actually exist), and — at the extreme — a safety-relevant miss in exactly
the collision-avoidance path SYSTEM_DESIGN.md §6.2's regulatory table cares about. This is why
manufacturing QA (below) treats calibration as a PASS/FAIL production gate, not an advisory step.

**Manufacturing QA framing.** At end-of-line, extrinsic calibration is a per-unit production TEST, not
just a setup step: every unit's achieved reprojection residual is logged against a pass threshold (this
project's own gate-with-measured-margin discipline — see `THEORY.md`'s results table — is a small,
concrete illustration of exactly this kind of QA acceptance criterion), and units that fail get
re-fixtured and re-calibrated before shipping, the same "test, don't just assume" discipline
SYSTEM_DESIGN.md §5.1's "QA & functional safety" team owns for the whole vehicle/robot.

**Service economics.** Recalibration is a recurring FLEET-OPERATIONS cost (SYSTEM_DESIGN.md §5.2's
longest-lived lifecycle phase): scheduled recalibration needs a technician's time (or, increasingly, a
field-deployable lightweight check like this file's §3 procedure, designed specifically to reduce that
technician time); unscheduled recalibration after a collision or service event is an unplanned
maintenance cost that fleet-reliability engineering (SYSTEM_DESIGN.md §5.4's unit-economics paragraph)
tries to minimize by designing mounts that resist drift in the first place — a mechanical-engineering
investment (`PRACTICE.md` §1's fixturing discussion, in miniature) that trades upfront BOM cost against
downstream service cost, the classic robotics-as-a-service reliability tradeoff.

**Regulatory context.** Extrinsic calibration is not itself named in any standard, but it is
foundational EVIDENCE inside the regulatory paths SYSTEM_DESIGN.md §6.2 maps for the robot types that
depend on it — most directly **ISO 26262** (functional safety of E/E systems) and **UL 4600** (the
safety-case standard for full autonomy) for autonomous vehicles, where a documented, repeatable, and
periodically-verified calibration procedure is part of the argued-and-evidenced safety case those
standards require; and **ISO 10218**/**ISO/TS 15066** for industrial arms with vision-guided or
collision-avoidance behavior. Label this orientation, not compliance guidance — the specific evidentiary
requirements are program- and jurisdiction-specific.

**Where this work lives in a company.** Owning team: **perception / calibration engineering**
(SYSTEM_DESIGN.md §5.1's "Perception" row), typically a small specialized group within the larger
perception organization (titles like "Calibration Engineer" or "Perception Systems Engineer" are common
at AV and robotics companies). Adjacent teams: mechanical engineering (fixture and mount design, this
file §1), manufacturing/test engineering (the factory-line QA gate, above), and fleet operations (field
recalibration triggers and monitoring, §3) — calibration is one of the few topics that genuinely touches
all three phases of SYSTEM_DESIGN.md §5.2's product lifecycle from EVT through fleet operations.

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
