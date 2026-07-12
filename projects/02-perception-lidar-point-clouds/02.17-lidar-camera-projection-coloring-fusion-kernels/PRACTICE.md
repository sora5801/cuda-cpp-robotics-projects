# 02.17 — LiDAR-camera projection/coloring fusion kernels: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Section dated 2026-07-12.

## 1. Building it — construction of the robot/part

This project's "part" is the **camera-LiDAR fusion rig** — two sensors sharing one rigid mount, one
clock, and (in production) one calibration record. Three physical realities the fusion software inherits
directly, in the order they bite:

- **Synchronization (FSYNC/PTP).** A camera integrates light over an exposure window; a spinning or
  scanning LiDAR sweeps continuously. Fusing "this camera frame" with "these LiDAR returns" correctly
  requires both sensors to agree on WHEN "now" is — either the camera's shutter is hardware-triggered off
  the LiDAR's own sweep-start pulse (FSYNC, the tighter option) or both devices discipline their clocks
  to a shared PTP (IEEE 1588) grandmaster and every message carries a hardware timestamp a fusion node
  aligns after the fact. [`01.20`](../../01-perception-cameras-vision/01.20-time-of-flight-raw-processing)'s
  own time-of-flight timing-budget discussion and [`02.08`](../02.08-per-point-motion-deskew-with-pose-interpolation)'s
  deskew both depend on this same hardware timing chain being right BEFORE either algorithm runs — this
  project's single static synthetic frame (README "Limitations") is exactly the case where sync trivially
  holds because there is only one instant to begin with; a real rig never gets that for free.
- **Rolling shutter.** Most commodity/automotive cameras read out row-by-row, not all at once — a fast-
  moving scene point can shift several pixels between the first and last row's exposure instant. This
  project's ideal camera model assumes a single, simultaneous capture; a real fusion pipeline needs the
  same per-row timing correction [`01.10`](../../01-perception-cameras-vision/01.10-rolling-shutter-correction-using-imu-rates)
  implements (named here, not reimplemented) BEFORE projecting LiDAR points into a rolling-shutter image,
  or points near the top and bottom of the frame land on subtly wrong pixels even with a perfect
  extrinsic and perfect sync.
- **Mounting and the calibration-maintenance loop.** The rig's rigid mount is exactly what
  [`01.17`](../../01-perception-cameras-vision/01.17-camera-lidar-camera-camera-extrinsic-calibration)
  measures once (or periodically) and this project consumes as a fixed constant; a multi-LiDAR rig
  additionally runs [`02.16`](../02.16-multi-lidar-merging-extrinsic-refinement)'s continuous refinement
  to keep several LiDARs' extrinsics mutually consistent. **This project is that maintenance loop's
  visual smoke test**: 01.17 reports a recovered rotation/translation error in degrees and millimeters,
  numbers no one can eyeball; this project's `cloud_topview.ppm`/`cloud_sideview.ppm` artifacts and its
  calibration-error sensitivity sweep turn the SAME error into a visibly wrong-colored point cloud a
  technician can look at and immediately distrust — the sensitivity curve (THEORY.md "The math") is the
  quantified version of "how bad does the calibration have to drift before someone notices by eye."

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-12. All parts named below are **illustrative examples, never endorsements**;
part numbers and prices go stale — verify current before relying on any of them.*

| Piece | Illustrative choices (2026) | Role in this project's pipeline |
|---|---|---|
| Compute | Jetson Orin class (embedded AV/AMR) / x86 + discrete RTX (reference machine: RTX 2080 SUPER) | The four fusion kernels; well under 1 ms per frame at this project's point counts |
| Camera | Hobby: USB3 global-shutter module (~US$100s); automotive: GMSL2 imager with hardware FSYNC (~US$100s–1,000s); industrial: machine-vision camera with PTP support (~US$1,000s+) | The dense RGB source `sample_bilinear_kernel` reads |
| LiDAR | Hobby/research: solid-state or mechanical spinning unit, 16–32 channel (~US$1,000s); automotive: 64–128+ channel, hardware-timestamped (~US$1,000s–10,000s+) | The sparse geometry source; channel count and scan rate directly set this project's "how sparse is sparse" story (THEORY.md "The problem") |
| Sync hardware | FSYNC trigger cable / GPS-disciplined PTP grandmaster switch | Closes the timing gap §1 names as the first thing that must be right |
| Rig / mount | Machined aluminum bracket, precision-located mounting bosses, thread-locking hardware | The physical baseline `b` THEORY.md's occlusion formula is a function of — a taller/stiffer mount changes the cohort math, not just the aesthetics |
| Rig calibration | A physical checkerboard/ChArUco target + fixture (01.17's own bring-up hardware) | Produces the `T_camera_lidar` this project consumes as a fixed constant |

## 3. Installation & integration — putting it on a real robot

- **Process shape.** On a ROS 2 robot this project's four kernels would live inside a single **fusion
  node** subscribing `sensor_msgs/PointCloud2` (LiDAR) and `sensor_msgs/Image` (camera), publishing a
  colored `PointCloud2` (Direction A) and/or a `sensor_msgs/Image` depth product (Direction B,
  `depth_image_proc`-shaped, THEORY.md "Where this sits in the real world"). The demo's structure —
  persistent device buffers, one launch sequence per frame, host-side gating — maps directly onto that
  node's per-callback work.
- **Message synchronization, honestly.** A real node cannot assume a matched camera frame and LiDAR sweep
  arrive together; ROS 2's `message_filters::TimeSynchronizer`/`ApproximateTimeSynchronizer` buffers and
  pairs messages within a tolerance window — itself only as good as §1's hardware sync. Where sync is
  loose (no FSYNC, PTP-only), a fielded system typically widens this project's occlusion band
  (`kOcclusionBandM`) or falls back to timestamp-nearest pairing with an explicit staleness bound, rather
  than silently fusing a mismatched pair.
- **Latency budget.** README "System context" states the measured kernel-time budget (well under a
  millisecond at this project's point counts); the REAL per-frame latency a fielded system cares about
  also includes the `PointCloud2`/`Image` message deserialization, the H2D copy, and — if the fusion
  output feeds a planner — the round trip back to host memory. At camera rate (10–30 Hz,
  SYSTEM_DESIGN.md item 1), this project's kernel cost is a rounding error next to those I/O stages; a
  production implementation would profile the WHOLE callback, not just the kernels.
- **Bring-up = the testing ladder, rung by rung (CLAUDE.md §1).**
  1. *Simulation* — this demo, plus a wider calibration-error sweep than README's three levels
     (Exercise 1) to characterize the fielded rig's actual error budget before trusting any output.
  2. *Bench, static scene* — a real camera+LiDAR pair pointed at a known, physically-measured target
     (a checkerboard at a taped-off distance) so `coloring_accuracy`-style gates have a REAL ground
     truth to compare against, not synthetic.
  3. *Bench, controlled motion* — a turntable or slow rail move to exercise §1's sync/rolling-shutter/
     deskew chain before trusting the pipeline on a free-moving platform.
  4. *On-vehicle, logged only* — record fused output for offline review before any downstream consumer
     (planner, HMI overlay, mapping) is allowed to ACT on it.
- **N/A here:** no ROS 2 node, no message_filters pairing, and no CAN-FD/EtherCAT bus integration are
  implemented in this project — the demo's "sensors" are two files loaded once (`data/sample/`), stated
  per contract. `PRACTICE.md` §3's own testing ladder is the honest map from this teaching core to that
  real integration, not a claim that the integration already exists here.

## 4. Business & regulatory context

- **Who needs this capability.** Camera-LiDAR fusion is close to a NORM in autonomous-vehicle perception
  stacks (every major AV program runs some version of point coloring and/or depth-image fusion) and
  increasingly common in warehouse AMRs and delivery robots that combine a cheap camera's semantic
  richness with a LiDAR's reliable geometry. It is the concrete, per-frame instance of the "why calibrate
  multiple sensors together at all" question SYSTEM_DESIGN.md item 5's perception team owns.
- **The players.** Every AV/AMR company's internal perception team owns a version of this pipeline
  (build, not buy — the extrinsic, the occlusion policy, and the accuracy bar are close enough to the
  product that they are rarely outsourced); the open ecosystem's closest equivalents are ROS 2's
  `image_geometry`/`depth_image_proc` (the plumbing, not the policy) and academic PointPainting-lineage
  code releases (the semantic-painting extension, README "Prior art").
- **Cost of getting it wrong.** A silently miscalibrated or unsynced fusion node does not crash — it
  produces PLAUSIBLE-LOOKING, WRONG output (a colored point cloud that "looks right" at a glance but
  paints occluded points the wrong color, or a depth image subtly offset from the RGB it accompanies).
  That is a worse failure mode than a crash for anything downstream that trusts the fusion output without
  its own independent check — exactly why this project's evaluation gates exist as *quantified*, not
  eyeballed, checks, and why §1's "visual smoke test" framing matters: a human glancing at a colored
  cloud can catch what a silent numeric drift would not.
- **Regulatory.** No dedicated standard governs camera-LiDAR fusion specifically; it is one input into
  the broader perception system regulators DO care about (AV: ISO 26262 functional safety / UL 4600
  safety-case argumentation; industrial and service robots: ISO 10218/ISO 13482 — SYSTEM_DESIGN.md item
  6's orientation map). Where fusion output feeds any safety-relevant decision, the calibration and
  synchronization chain this file names (§1, §3) becomes part of that system's safety case, not an
  implementation detail — an orientation, not a compliance claim.
- **Owning team.** Perception / sensor-fusion (SYSTEM_DESIGN.md item 5's "Perception" row); adjacent
  teams: calibration engineering (owns 01.17/02.16's periodic recalibration), embedded/electrical (owns
  §1's FSYNC/PTP hardware chain), and mapping/planning (the primary consumer of this project's colored
  and depth-fused output).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
