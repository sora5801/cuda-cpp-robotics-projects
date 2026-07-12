# 02.09 — Normal + curvature estimation at millions of points/sec: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

This project is a pure software/geometric-compute stage — it has no mechanical part of its own to build.
What it DOES need physically is the sensor that produces its input point cloud, so this section describes
that carrier instead, at the level of construction detail relevant to why its OUTPUT (points, with their
spacing and noise characteristics) looks the way it does — the same reasoning `THEORY.md`'s "The problem"
section leans on.

A spinning mechanical LiDAR (the sensor this project's synthetic-data noise model is tuned against) is
built around a rotating optical bench: a stack of laser emitter/receiver pairs (one per beam/ring),
mounted on a motor-driven turret, spinning at a fixed rate (600-1200 RPM, i.e. 10-20 Hz sweep rate is
typical for automotive units). Construction tolerances that directly determine the point-cloud
characteristics this project's normal estimator has to cope with: **beam-to-beam elevation-angle
calibration** (each laser's exact mounting angle is individually factory-calibrated and stored in a
per-unit calibration file — a few hundredths of a degree of uncalibrated error directly widens the
effective K-neighborhood's spread across rings, inflating measured curvature/surface-variation even on a
truly flat wall); **rotor bearing wear and vibration** (a worn bearing introduces angular jitter beyond
the sensor's own specified range-noise budget — this shows up as EXTRA lateral scatter this project's
`sigma_m` noise model does not distinguish from intentional range noise); **the optical window's
cleanliness and scratches** (dust, rain, or a scratched dome scatters/attenuates the return beam,
producing dropped returns and range outliers that look, to a downstream normal estimator, like the
"isolated" or "high-noise" failure modes this project's `kDegenIsolated` flag and noise cohorts model,
without themselves being neatly Gaussian). A solid-state or flash LiDAR (no moving parts, a fixed array
of emitters) sidesteps the rotor-wear failure mode entirely but has its own construction constraint: FOV
is fixed by the array's optical design, and per-pixel range precision trades off against array density
on the same silicon die.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-11. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

**Compute this stage runs on.** Normal estimation sits early in the perception pipeline and typically
shares a GPU with the rest of perception (detection, segmentation, mapping) rather than owning a
dedicated chip:

- **Research / prototyping tier**: any desktop NVIDIA GPU, Turing (sm_75, this project's floor) or
  newer — this project's own reference measurements were taken on an RTX 2080 SUPER. ~$400-1500.
- **Embedded/production tier (mobile robot, AV)**: NVIDIA Jetson AGX Orin (Ampere-class SM, up to 275
  TOPS INT8 / substantial FP32 throughput, 15-60W configurable power envelope) or Jetson Orin NX for
  smaller platforms — the SWaP (size, weight, and power) budget every real mobile robot's compute
  actually has to fit inside, unlike a desktop workstation. ~$400-2000 module cost depending on variant
  (2025-era pricing; verify current).
- **Industrial-grade / functional-safety-relevant tier**: an x86 + discrete RTX-class embedded GPU
  (e.g., an industrial PC with an RTX A-series or L-series card) where ISO 26262 ASIL-rated companion
  MCUs handle the safety-critical envelope and the GPU handles perception as a best-effort, monitored
  subsystem — the architecture pattern `docs/SYSTEM_DESIGN.md` item 6 names generically for AV-class
  compute.

**The sensor** (not this project's own hardware, but its input's source, illustrative tiers):
- Hobby/research: Livox Mid-360 or similar solid-state LiDAR (~$500-1000), or a depth camera
  (Intel RealSense D435/D455, ~$200-400) for indoor/short-range work.
- Research-grade spinning mechanical: Velodyne/Ouster 16-32-beam units (~$4,000-12,000).
- Automotive-grade: Hesai/RoboSense/Ouster 128-beam or solid-state automotive LiDAR (~$500-2,000 at
  automotive volume pricing, considerably more at low volume) — the class of sensor this project's noise
  model (15 mm high-noise cohort) is calibrated against, citing the same figure `02.01`/`02.05` use.

**No custom silicon is implied by this project** — it is a software algorithm running on a general-
purpose GPU, not an ASIC/FPGA design; N/A beyond the compute tiers above.

## 3. Installation & integration — putting it on a real robot

**Where this code would run.** The same GPU-bearing compute module running the rest of the perception
stack (Jetson-class SoC on a mobile robot; an embedded x86+GPU box on a larger platform) — this stage is
never given its own dedicated processor in a real deployment; it is one kernel launch (or a fused
sequence of launches) inside a larger perception graph.

**OS and real-time constraints.** Typically Linux (Ubuntu LTS + the vendor's L4T/JetPack BSP on Jetson);
this stage itself is NOT hard-real-time (a late or dropped normal-estimation frame degrades downstream
registration/planning quality rather than causing an immediate safety fault), but it sits inside a
pipeline whose END-to-end latency budget (sensor to actuator) often IS safety-relevant, so its throughput
(`THEORY.md`'s Mpts/s story) is a real scheduling input even without hard-real-time guarantees on this
stage alone.

**ROS 2 node/topic shape.** A perception node subscribing to `sensor_msgs/msg/PointCloud2` (the
deskewed, possibly-downsampled cloud from upstream `02.08`/`02.01`-equivalent nodes) and publishing an
AUGMENTED point cloud with per-point normal (nx,ny,nz) and curvature fields appended — `PointCloud2`
natively supports arbitrary named float fields, so a real ROS 2 implementation would add `normal_x`,
`normal_y`, `normal_z`, `curvature` fields to the SAME message rather than publishing a separate topic
(PCL's own `pcl::PointNormal` point type is exactly this convention, and PCL <-> ROS 2 message conversion
is a solved, standard bridge). This project's own `PointCloud`/message-shaped-struct convention
(`docs/SYSTEM_DESIGN.md` item 3) mirrors that field layout deliberately.

**Bus/interface**: none directly — this stage consumes and produces in-GPU-memory or ROS 2 message data;
it does not itself talk to a CAN-FD/EtherCAT/sensor bus (that happens upstream, at the LiDAR driver).

**Calibration & bring-up**: the sensor's own beam-elevation and range calibration (factory-supplied,
occasionally re-verified against a known flat target) is the only calibration this stage's CORRECTNESS
depends on — a miscalibrated LiDAR produces a point cloud whose "flat wall" is not actually flat in the
sensor frame, which this stage would then (correctly, from its own point of view) report as having
nonzero curvature. There is no separate calibration procedure for the normal-estimation stage itself.

**The safe hardware-testing ladder** (CLAUDE.md §1): this stage computes geometry and commands no
actuator directly, so it sits at the SAFEST end of the ladder among this repo's projects — simulation
(this project's own analytic-surface demo) is sufficient to validate its correctness; the ladder becomes
relevant only for whatever DOWNSTREAM consumer (ICP-driven localization, grasp execution) eventually
commands motion, and that consumer's own project documents its own ladder. Feeding this stage's output
into a real robot's control loop without validating the WHOLE downstream chain in simulation first would
be the actual safety-relevant mistake, not this stage in isolation.

## 4. Business & regulatory context

**Who needs this capability.** Every company building a LiDAR- or depth-camera-based autonomy stack:
AV/robotaxi companies (Waymo, Cruise-successors, Chinese AV players), warehouse/logistics AMR vendors
(Locus Robotics, 6 River Systems-class platforms), industrial bin-picking and manipulation vendors
(anywhere antipodal grasp scoring is used), and every SLAM/mapping tool vendor (surveying, construction
progress capture) — normal estimation is upstream INFRASTRUCTURE for all of them, rarely a standalone
product feature a customer asks for by name.

**Commercial and open-source players.** Open-source: PCL, Open3D (both ubiquitous, BSD/MIT-licensed,
the de facto standard implementations named throughout `THEORY.md`). Commercial: normal estimation is
typically bundled inside a vendor's larger perception SDK (NVIDIA Isaac Perceptor, vendor-specific SLAM
stacks) rather than sold as a standalone component — it is table-stakes infrastructure, not a
differentiator, which is exactly why open-source implementations dominate this specific stage even in
commercial stacks.

**What getting it wrong costs.** A systematically biased normal estimator degrades EVERY downstream
consumer simultaneously: worse ICP convergence (`02.06`) means worse localization means worse mapping;
worse grasp-normal estimates mean higher grasp failure/drop rates on a manipulation line (a direct
throughput/cost impact in a warehouse); worse ground-plane/traversability estimates on a mobile robot
mean either false-positive obstacle stops (throughput cost) or false-negative missed obstacles (a real
safety risk). Because this stage is silent, cheap infrastructure that "just works" until it does not,
a regression here is a classic hard-to-diagnose root cause for downstream failures — the whole reason
this project's verification section is as layered as it is (three independent oracles PLUS closed-form
analytic gates) is that this stage's correctness is load-bearing for everything above it.

**Regulatory path.** This stage is a perception COMPONENT, not a certified system in its own right — it
inherits whatever regulatory path its host system follows (cite `docs/SYSTEM_DESIGN.md` item 6's map):
AV-class deployments fall under ISO 26262 (functional safety) and, in the US, an evolving UL 4600 /
NHTSA framework; industrial manipulation falls under ISO 10218 / ISO/TS 15066 (collaborative robotics);
none of these standards certify a "normal estimation algorithm" in isolation — they certify the
INTEGRATED perception-to-actuation chain's safety case, of which an algorithm like this one is one
verified/validated ingredient among many. This project's own gates (closed-form analytic ground truth,
independent oracles) are the KIND of evidence a real functional-safety case would want for this
component, though at nowhere near the rigor (statistical coverage, adversarial/edge-case testing,
formal traceability) an actual ISO 26262 software unit would require.

**Where this work lives inside a robotics company.** The Perception team (README "System context");
peer/adjacent teams: Localization & Mapping (the direct consumer via `02.06`), Manipulation/Grasping
(the direct consumer via `19.01`), and Simulation & Tools (owns the synthetic-data generation and
verification harness pattern this project itself is an instance of). Typical role titles: Perception
Engineer, Computer Vision Engineer, Robotics Software Engineer (Perception).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
