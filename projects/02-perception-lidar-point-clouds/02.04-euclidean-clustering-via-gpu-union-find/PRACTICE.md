# 02.04 — Euclidean clustering via GPU union-find / connected components: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.
>
> *Sections dated 2026-07-11.*

## 1. Building it — construction of the robot/part

This project is pure software — it consumes a point cloud and produces cluster labels, with no
mechanical or electrical construction of its own. What it is honest to describe instead is what a
clustering FAILURE costs operationally, since that is the "physical" consequence this algorithm's
correctness actually has once it is wired into a real perception stack:

- **A merged pedestrian + street furniture cluster** (this project's chaining-test scenario, made real):
  if a person walking near a pole or a wall gets clustered together with it, a downstream tracker
  (domain 04.xx) either drops the pair as one static-looking blob (a pole does not move; if the merged
  cluster's velocity estimate averages "mostly stationary," a moving pedestrian can be MISSED entirely
  by a motion-based filter) or, if it does track the merged blob, hands the planner a single object with
  a confusing, inconsistent shape and motion signature frame to frame.
- **A split truck** (the over-segmentation / occlusion dual of the chaining failure, THEORY.md "The
  problem"): if one truck's returns split into a cab cluster and a trailer cluster because of a
  self-occlusion gap, a tracker can spawn TWO track ids for one vehicle, each with half the truck's true
  extent — a classic source of "track fragmentation" or "identity churn," where the SAME physical object
  is repeatedly assigned new ids as clustering flickers between one blob and two, frame to frame. This
  is a known, named failure mode in production AV perception stacks, not a hypothetical.

Both failures are downstream INTEGRITY costs of a clustering stage, not construction/manufacturing costs
— the closest this project comes to "what breaks in the field" is "what a bad cluster does to everything
built on top of it."

## 2. Real hardware — chips, parts, illustrative BOM

*All parts named below are **illustrative examples, never endorsements**; part numbers and prices go
stale — verify current before relying on any of them.*

This is a perception COMPUTE stage, not a sensor or actuator — there is no clustering-specific silicon.
What matters is the compute tier it runs on and the tolerance-vs-platform trade a real deployment makes:

| Tier | Example compute | Where clustering runs | Notes |
|------|------------------|------------------------|-------|
| Research / prototyping | Desktop x86 + discrete RTX-class GPU (this project's own reference machine) | Same box as SLAM/mapping development | Ample headroom; the 1,469-point committed scene runs in well under 1 ms |
| Embedded / production robot | Jetson Orin-class SoC (e.g. Orin NX/AGX) | The robot's main perception compute module | Real scenes are 10–100x this project's point count; `kMaxEdgesPerPoint` and edge-buffer sizing (THEORY.md "The GPU mapping") would need re-deriving at that scale and density |
| MCU-class | N/A | Never — this workload (spatial hashing, Thrust sort/scan, thousands of atomics) needs a real GPU or at minimum a multi-core CPU; it does not fit a microcontroller's memory or compute budget | |

**Tolerance-vs-platform honesty:** `kClusterToleranceM` (this project's `d`) is not a hardware constant
— it is TUNED per sensor and per deployment. A short-range solid-state LiDAR with tight angular
resolution can afford a smaller `d` (finer object separation, more false splits if too small); a sparse
long-range mechanical LiDAR needs a larger `d` to keep a single surface connected at range, at the cost
of a higher chaining risk (THEORY.md "The problem"). There is no universal correct value; production
systems tune it against a labeled validation set per sensor/platform combination, and often make it
RANGE-ADAPTIVE (a documented limitation, README "Limitations & honesty").

## 3. Installation & integration — putting it on a real robot

**Where it runs:** the SAME perception compute node that runs 02.01 (downsampling) and 02.03 (ground
segmentation) — this stage's input IS their output, so keeping the whole 02.01 → 02.03 → 02.04 chain on
one GPU avoids a host-device round-trip between every stage (each stage's output could stay resident in
device memory rather than copying back to host and out again — an optimization this project's own
demo does NOT implement, since each project in this repo is required to stay independently runnable and
self-contained, CLAUDE.md §4; a real fused pipeline would keep everything on-device end to end).

**OS and real-time constraints:** Linux (typically Ubuntu, JetPack on Jetson hardware) with the CUDA
driver stack; this stage's few-hundred-microsecond-to-low-millisecond runtime (README "System context")
sits comfortably inside a 10–20 Hz scan-rate budget even without a hard real-time OS, but a production
integration would still run it inside a bounded-latency ROS 2 executor callback so a slow scan cannot
silently stall the planning loop behind it.

**ROS 2 node/topic shape:** a `PointCloudCluster` node subscribing to 02.03's non-ground `PointCloud2`
topic (the ground-removed cloud), publishing a custom `ClusterArray`-shaped message (per-cluster
centroid `geometry_msgs/Point`, count, and an axis-aligned bounding box, closely resembling
`vision_msgs/Detection3DArray` in a real ROS 2 stack) that 23.01's costmap node and a 04.xx tracker node
both subscribe to — the exact `docs/SYSTEM_DESIGN.md` §3 message-shape convention this repo's projects
follow so the conceptual ROS 2 mapping stays obvious even though no project links against real ROS 2.

**Parameter co-tuning discipline (the integration trap this project's own design highlights):** `leaf`
(the voxel size), `d` (the cluster tolerance), and `min_cluster_size` are NOT independent knobs — this
project sets `leaf == d` deliberately (THEORY.md "The math" proves why that specific relationship makes
the 27-cell stencil exact), so changing one without the other breaks a correctness invariant, not just a
performance tuning. A real integration should treat `(leaf, d)` as ONE co-tuned parameter pair, and
`min_cluster_size` as a SEPARATE, sensor-density-dependent knob (too small and every echo lobe is a
"cluster"; too large and small-but-real objects like traffic cones vanish).

**Safe testing ladder:** this project's own demo IS entirely simulation (a hand-built synthetic scene) —
there is no hardware-testing ladder to climb for the algorithm itself, since it never touches an
actuator. The ladder belongs to whatever CONSUMES its output: a costmap or planner that uses cluster
output to command motion follows the standard sim → HIL → bench/tethered → free-running ladder with
E-stop and limits at every rung (CLAUDE.md §1) — this project's job is to hand that consumer clean,
verified data, nothing more.

## 4. Business & regulatory context

**Who needs this:** every mobile robot that must avoid discrete obstacles — warehouse AMRs, autonomous
vehicles, agricultural robots, delivery robots — needs SOME form of "turn points into objects," whether
via geometric clustering (this project), a learned instance head, or both in combination as a
cross-check.

**Commercial and open-source players:** PCL (open source, the reference CPU implementation this
project's algorithm teaches toward); NVIDIA cuML/cuGraph (open-source GPU graph libraries used in
production perception stacks); every major AV company (Waymo, Cruise, Zoox and peers) and every major
industrial AMR vendor (Locus Robotics, 6 River Systems, Fetch/Zebra and peers) ships SOME clustering or
instance-segmentation stage in its perception pipeline, almost always proprietary and tuned to its own
sensor suite.

**What getting it wrong costs:** as Section 1 above describes concretely — a missed pedestrian (merged
into a static object, invisible to a motion-gated safety system) is a safety-critical failure with
potential injury/fatality and liability exposure; a fragmented track (one truck reported as two moving
objects) can cause an unnecessarily conservative (costly downtime, false-positive braking) or, worse,
an inconsistent planner response. Neither is this project's problem to solve alone — but it is
this project's job to hand the rest of the stack an HONEST result, including the honest chaining
failure this project deliberately surfaces rather than hides.

**Regulatory path (orientation, not compliance guidance — `docs/SYSTEM_DESIGN.md` item 6):**
object-detection/perception-object INTEGRITY is a named concern in AV safety cases under frameworks like
UL 4600 (which asks explicitly how a system justifies its perception pipeline's completeness and
correctness) and ISO 26262 (functional safety of the electrical/electronic systems consuming perception
output); for service/industrial robots, ISO 13482 / ISO 10218 + ISO/TS 15066 govern the SAFETY FUNCTIONS
downstream of perception (speed-and-separation monitoring, e-stop behavior) rather than the perception
algorithm itself. This project computes none of those certified functions and makes no compliance claim
— it is an orientation map, not guidance.

**Where this work lives in a robotics company:** the perception team, specifically an "obstacle
detection" or "object proposal" sub-team (`docs/SYSTEM_DESIGN.md` item 5) — adjacent to the mapping team
that owns 02.01/02.03's upstream stages, the tracking/fusion team that owns 04.xx's downstream
consumption, and the planning team that owns 23.01's costmap consumption of this stage's output. Typical
role titles: perception engineer, robotics software engineer (perception), or (at companies with a
dedicated GPU-systems group) CUDA/GPU systems engineer working embedded within the perception team.

---

*Didactic orientation only — **not** procurement, legal, or compliance advice.*
