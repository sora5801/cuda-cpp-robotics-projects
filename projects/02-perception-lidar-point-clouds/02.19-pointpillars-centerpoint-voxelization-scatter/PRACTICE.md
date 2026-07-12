# 02.19 — PointPillars/CenterPoint voxelization + scatter kernels feeding TensorRT: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

This project is software-only — there is no physical part *of this kernel* to assemble. The physical
carrier is the **LiDAR sensor + compute mount** that this preprocessing stage would run downstream of on
a real vehicle or AMR (SYSTEM_DESIGN.md §2.1/§2.5). What that construction actually looks like, briefly:

- **The LiDAR itself** (mechanical spinning or solid-state/MEMS) is mounted rigidly to the vehicle's
  structure — typically roof-mounted for an AV (maximum unobstructed field of view), or a fixed mast on
  an AMR — with a machined or 3-D-printed bracket holding it to a precisely surveyed pose (its
  extrinsic calibration, `T_vehicle_lidar`, is a physical fact that must be measured after mounting, not
  assumed). Vibration isolation (rubber/silicone mounts) matters: a spinning LiDAR's return-timing
  accuracy degrades under high-frequency vibration, directly corrupting the `(x,y,z)` this project bins.
- **The compute** this pipeline would run on is NOT co-located with the sensor in most designs — the
  sensor connects via Ethernet (most modern automotive/industrial LiDARs) or a proprietary serial/USB
  link to a central compute unit elsewhere in the vehicle (§3 below). The sensor's own housing is
  environmentally sealed (IP67/IP69K typical for automotive-grade units) against dust, rain, and
  pressure-washing; the compute unit is typically in its own sealed, actively- or passively-cooled
  enclosure, connected by shielded, often ruggedized (locking, vibration-rated) Ethernet or CAN cabling.
- **What breaks in the field:** connector fatigue and cable chafing at mounting points (the single most
  common LiDAR field failure in mobile robotics, ahead of the sensor itself); condensation inside a
  poorly-sealed enclosure shorting the sensor's rotating-assembly electronics; and — specific to this
  project's downstream consequence — a loosened or shifted mount silently invalidating the extrinsic
  calibration, which shows up not as a hard failure but as subtly wrong `(x,y,z)` feeding every stage
  this project implements, with no error message anywhere in this pipeline.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-12. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

**Where this pipeline runs, relative to the engine it feeds** (the load-bearing question for this
project specifically): the pillarization/scatter kernels here and the downstream TensorRT-deployed
network (project 12.01) are two stages of ONE forward pass on the SAME GPU — not separate machines.
The relevant hardware choice is therefore "which GPU-bearing compute box is on the vehicle," not a
sensor-vs-compute split:

| Tier | Illustrative compute | Notes |
|------|----------------------|-------|
| Hobby / research | A desktop RTX-class GPU (e.g. RTX 4070-4090 class) in a robot-mounted mini-ITX box, or a Jetson AGX Orin (64 GB) dev kit | What most research AV/AMR platforms actually run; ample headroom to develop and profile this exact pipeline; not automotive-qualified. |
| Industrial / commercial AMR | Jetson Orin NX/Orin Nano modules on a carrier board, or a ruggedized industrial PC with an embedded RTX/Ada-generation GPU | Fanless or actively-cooled sealed enclosures, extended temperature range, conformal-coated boards; this is the realistic target for a warehouse AMR's LiDAR-detection upgrade path (README "System context"). |
| Automotive-grade AV | NVIDIA DRIVE Orin/Thor-class SoCs, or equivalent automotive-qualified compute (Qualcomm Ride, Mobileye EyeQ) | ASIL-rated compute islands, lockstep safety cores alongside the GPU compute, AEC-Q100-qualified components throughout; PRACTICE.md §4's regulatory burden is what drives this tier's cost and validation timeline. |

**The LiDAR sensor itself** (feeding `points.bin`'s real-world equivalent): hobby/research — Velodyne
VLP-16/Ouster OS1 class (mechanical spinning, ~16-128 channels, Ethernet UDP output); industrial —
Hesai/RoboSense automotive-grade units or solid-state (Livox, Ouster digital lidar) for no-moving-parts
reliability; automotive AV — automotive-qualified solid-state or MEMS units from Hesai, Innoviz, Cepton,
or similar, meeting automotive vibration/EMC/temperature specs. **Comms:** Gigabit Ethernet (the
near-universal modern LiDAR interface — the sensor streams UDP packets of raw returns, which a driver
node reassembles into the `(x,y,z,intensity)` array this project's `points.bin` layout stands in for) or
CAN-FD for some industrial units.

**Compute chips relevant to THIS project's kernels specifically:** the pillarization/scatter/conv
kernels are ordinary CUDA — any CUDA-capable GPU from `sm_75` (Turing, this repo's floor) up runs them;
there is no specialized silicon for the binning/scatter step itself. The DOWNSTREAM TensorRT-deployed
network (12.01) is where Tensor Cores (Volta+) and, for INT8 deployment, DLA (Deep Learning Accelerator,
present on Jetson Orin) become relevant — this project's own kernels do not use either.

## 3. Installation & integration — putting it on a real robot

**Where it runs:** on the same GPU-bearing compute unit as the rest of the perception stack, as one
stage of a larger perception process — not a standalone node with its own hardware. In a ROS 2-shaped
deployment, this pipeline would live inside (or immediately upstream of) a `lidar_detector_node` that
subscribes to a `sensor_msgs/PointCloud2` topic (the real-world analog of `points.bin`, matching
`docs/SYSTEM_DESIGN.md` §3.6's `PointCloud` message convention this project's data layout already
follows) and publishes `vision_msgs/Detection3DArray` (or a project-specific detections message) — the
tensor hand-off this project's `[info] trt_handoff` line documents sits entirely INSIDE that one node,
between its preprocessing and inference sub-stages, never crossing a ROS topic boundary itself (crossing
a topic boundary there would add serialization latency this project's whole design exists to avoid).

**OS and real-time constraints:** Linux (Ubuntu is the near-universal robotics choice) with the NVIDIA
driver + CUDA + TensorRT stack; this is a SOFT real-time stage (SYSTEM_DESIGN.md §1.1's "LiDAR →
perception" boundary: 10-20 Hz, <50-100 ms budget) — a missed frame is a dropped detection cycle, not a
hardware-damaging fault the way a missed motor current-loop tick would be, so it does not require a
hard-real-time kernel or RTOS.

**Which bus it consumes/commands:** consumes the LiDAR's Ethernet/UDP stream (via the sensor vendor's
ROS 2 driver, or a custom UDP parser); this project's own output (a detection list) would typically be
consumed internally by a tracking/prediction node over shared memory or a ROS 2 topic, never over a
field bus (CAN/EtherCAT) directly — this stage does not command actuators.

**Calibration and bring-up:** the LiDAR's extrinsic calibration (`T_vehicle_lidar`) must be measured
(checkerboard/target-based or SLAM-based extrinsic calibration tools) and kept current — §1's mounting
discussion; this project's `kXMin/kYMin` BEV-window origin is defined RELATIVE TO the sensor/ego frame
that calibration establishes, so a wrong extrinsic silently shifts every pillar's world position.
Intrinsic calibration (per-beam range/intensity correction) is the LiDAR vendor's factory calibration,
consumed upstream of this project (domain 02's earlier projects, e.g.
[`02.20`](../02.20-lidar-intensity-calibration-across-channels/README.md), own that).

**The safe hardware-testing ladder** (CLAUDE.md §1's mandatory caveat: this project's demo output could,
in a real deployment, ultimately influence a robot's motion — every rung below applies):
simulation (this project's entire demo, and the synthetic scenes any real team would test a preprocessing
change against first) → hardware-in-the-loop (replay real, recorded LiDAR logs through the SAME code
running on the target compute, comparing against previously-validated detections — the standard
regression-test rung for a perception change) → bench/tethered testing (the sensor mounted on a static
or slow-moving test rig, detections logged and reviewed but NOT acted on by any controller) → limited
free-running operation (geofenced, speed-limited, always with a safety driver/operator and a working
E-stop) → full operation. This project's own demo never leaves the simulation rung — nothing here has
been validated on real hardware, and README/THEORY say so explicitly.

**Tensor-contract versioning discipline** (the load-bearing integration detail for THIS project
specifically, echoing [`01.14`](../../01-perception-cameras-vision/01.14-optical-flow-farneback-or-deep-learned-inference/README.md)'s
continuity theme): the exact shapes this project produces — `[P_occ, 32, 9]` pillar features,
`[P_occ, 2]` coords, `[1,6,200,200]` canvas (`[info] trt_handoff`) — are a REAL INTERFACE CONTRACT
between whichever team owns preprocessing and whichever team owns the trained network/TensorRT engine
(§4 below). Changing `kMaxPointsPerPillar`, `kNumPointFeatures`, or the grid dimensions on one side
without a coordinated, VERSIONED change on the other silently breaks the engine's expected input shape —
exactly the kind of change that must be caught by an integration test (this project's `VERIFY`/`GATE`
suite is the template for what that test should check on the preprocessing side) rather than discovered
in the field. The chain [`02.18`](../02.18-weather-filtering/README.md) → **02.19** →
[`12.01`](../../12-ml-ai/12.01-tensorrt-deployment-with-custom-cuda-pre-post/README.md) is a real,
ordered pipeline of exactly this kind: each stage's output shape is the next stage's input contract.

## 4. Business & regulatory context

**Who needs this, and in which products:** any company shipping LiDAR-based 3-D object detection —
autonomous-vehicle developers (Waymo, Cruise-successors, Zoox, and the broader AV supply chain), AMR/AGV
makers adding 3-D obstacle classification beyond simple 2-D-LiDAR safety fields (SYSTEM_DESIGN.md §2.1),
and any company building a perception SDK for others to integrate (NVIDIA's own DeepStream/Isaac
ecosystem is the most visible example, and the direct commercial descendant of the CUDA-PointPillars
reference implementation this project's THEORY.md cites).

**Main players:** commercially, this exact capability (real-time LiDAR 3-D detection preprocessing +
inference) is table stakes inside NVIDIA's DRIVE/Isaac/DeepStream stacks, Waabi/Waymo/Cruise-class AV
software stacks (largely proprietary, in-house), and open perception frameworks (Autoware, Apollo) that
smaller AV and robotics companies build on rather than re-deriving. Open-source implementations
(mmdetection3d, OpenPCDet) are the dominant RESEARCH/prototyping path; production deployment nearly
always re-implements the preprocessing kernels for the target embedded GPU, which is precisely the gap
this project is a teaching-scale version of.

**What getting it wrong costs:** for an AV, a missed or badly-localized detection from a broken
preprocessing stage is a safety-critical failure with potential for injury, recall, and liability — the
entire reason ISO 26262/UL 4600 (below) demand rigorous process, not just a working demo. For an AMR, a
missed detection risks a collision with people or equipment (lower kinetic energy than a road vehicle,
but still a real injury/liability/downtime risk); a systematically WRONG (not just missing) detection —
exactly the class of bug this project's `cap_truncation` gate is built to catch, since a silently
nondeterministic preprocessing stage can pass validation on one run and fail on the next with IDENTICAL
sensor data — is arguably worse, because it erodes confidence in validation itself.

**Applicable standards / regulatory path** (SYSTEM_DESIGN.md §6.2's regulatory map, this project's row):
for an AV consumer, **ISO 26262** (functional safety of electrical/electronic systems — process rigor,
ASIL-level requirements) and **UL 4600** (a full safety CASE for autonomous operation, which a
perception pipeline's validation evidence feeds directly) are the relevant frameworks; for an AMR
consumer, **ISO 13482** (safety for personal-care/service robots sharing space with untrained people) is
the closer fit. Both are ORIENTATION here, not compliance guidance — a real deployment's safety case
would need to argue (with evidence: test coverage, failure-mode analysis, validation datasets) that a
preprocessing bug of the kind this project demonstrates (silent, input-order-dependent nondeterminism)
cannot cause an unsafe detection miss — a genuinely hard argument, and part of why AV perception
validation costs so much more than the modeling work itself suggests it should.

**Deterministic preprocessing as an auditability asset** (the concrete business angle this project's
`cap_truncation` gate surfaces): a safety case built on Method A's atomic binning would need to argue
that its (real, measured) input-order-dependent nondeterminism cannot flip a detection outcome near a
decision boundary — a hard, probabilistic argument. A safety case built on Method B's deterministic
sorted binning can instead argue "the same sensor input always produces the same detection," a much
STRONGER and cheaper-to-audit property, at the cost of the sort's `O(N log N)` overhead vs. Method A's
purely `O(N)` atomic path. This is a genuine, quantifiable engineering trade a real perception team
weighs — reproducibility for validation/debugging vs. raw throughput — and this project's two
implementations are a working demonstration of exactly that trade, not just a teaching exercise about
GPU atomics.

**Where the work lives inside a robotics company** (SYSTEM_DESIGN.md §5.1): this project sits at the
seam between **Perception** (owns the point-cloud input contract and the classical preprocessing
upstream of this project — domains 01-03) and **ML/data** (owns the learned network, training data, and
TensorRT deployment — domain 12); the specific role that owns THIS boundary is often titled "Perception
ML Infrastructure Engineer" or "Applied ML Engineer, Perception" — someone comfortable in both CUDA and
the training framework, since correctness here (this project's whole subject) is invisible to a model
accuracy metric but directly determines whether the model's accuracy claims transfer to deployed
hardware. Adjacent teams: **Simulation & tools** (owns the synthetic scene generation this project's own
`make_synthetic.py` is a small-scale stand-in for) and **QA & functional safety** (consumes this
project's VERIFY/GATE suite's spirit as the template for a real validation test plan).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice.*
