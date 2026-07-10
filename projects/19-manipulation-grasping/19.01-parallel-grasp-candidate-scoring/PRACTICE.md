# 19.01 — Parallel grasp-candidate scoring: antipodal sampling over point clouds: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

This project is computational (a scoring algorithm over a point cloud) — it has no part of its own to
manufacture. Its physical carrier is the **two-finger parallel-jaw gripper** whose stroke and
friction model this project's gates are built around (`data/sample/objects_meta.csv`'s
`gripper_w_min_m`/`gripper_w_max_m`/`friction_mu`).

**Construction of a parallel-jaw gripper, briefly.** A rigid body (housing) carries a linear actuator
— most commonly a single lead-screw or rack-and-pinion driven by one small BLDC/DC motor, mechanically
linked so BOTH jaws move symmetrically toward or away from the centerline (a "one motor, two jaws"
design is standard; it guarantees the grasp stays centered without extra sensing). Each jaw ends in a
**fingertip** — a replaceable pad, usually rubber, silicone, or a 3D-printed TPU shape, sometimes with
a compliant/underactuated linkage for slight shape conformance. The fingertip material is what
actually sets the real-world `friction_mu` this project's friction-cone gate consumes — a bare
aluminum fingertip on a painted steel part might see `mu` ≈ 0.15–0.25; a silicone pad on the same part
can reach `mu` ≈ 0.6–0.9 (illustrative ranges; PRACTICE.md's dated-parts caveat applies to friction
coefficients too — they are measured empirically per material pair, never looked up from a single
universal table). Wiring runs a small number of conductors (motor power + encoder/Hall feedback,
sometimes a force sensor) down the wrist to the arm's own cabling; on a collaborative or bin-picking
cell the gripper is usually the single most field-replaceable part on the arm (fingertips wear or
break on a misjudged grasp far more often than the arm itself fails).

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-10. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

**Where this CODE runs.** Grasp-candidate scoring is a perception/planning workload
(SYSTEM_DESIGN.md §6.1's compute tier), not a real-time control loop — it runs on the cell's main
compute, not on any motor-control MCU:

| Tier | Illustrative example | Notes |
|------|----------------------|-------|
| Hobby / research desktop | any CUDA-capable desktop GPU (this project's own dev machine: RTX 2080 SUPER, sm_75) | what this repo's demo actually runs on |
| Industrial PC + discrete GPU | an IPC (e.g. a fanless industrial box) + a workstation-class GPU (e.g. an RTX A-series or equivalent) | SYSTEM_DESIGN.md §2.2's modeled work-cell compute tier — "industrial PC + discrete GPU beside the vendor's arm controller" |
| Embedded / mobile manipulator | Jetson Orin-class SoC | when the grasp-scoring stage rides on the robot itself (a mobile manipulator) rather than a fixed cell PC |

**The gripper's own silicon** (illustrative, dated 2026-07-10 — verify current):

| Function | Illustrative part class | Notes |
|----------|--------------------------|-------|
| Motor drive | a small BLDC/DC gate-driver + H-bridge IC (e.g. TI DRV8xxx family, or an integrated smart-driver module) | closes the jaw; usually PWM-commanded over the gripper's own local bus |
| Position/force sensing | a Hall-effect or optical encoder on the actuator shaft; optionally a load cell or strain gauge for force feedback | position tells the controller how far the jaws closed; force feedback (where present) is the real-world signal this project's `width_m`/geometric estimate stands in for |
| Comms to the arm/cell PC | RS-485, CAN, EtherCAT, or a vendor-proprietary tool-flange protocol | carries open/close commands and feedback; a mainstream commercial parallel gripper (illustrative stroke class matching this project's modeled 10–90 mm) typically exposes one of these |
| Sensors upstream of this code | a wrist- or frame-mounted stereo or structured-light depth camera (e.g. an industrial RGB-D unit) | produces the segmented `PointCloud` this project's candidates are sampled from — see README "System context" |

## 3. Installation & integration — putting it on a real robot

**Where this code would run.** On the manipulator work cell's main compute (SYSTEM_DESIGN.md §2.2),
alongside — not instead of — the vendor's own arm controller, which continues to own the low-level
joint servo loops (0.5–1 kHz, SYSTEM_DESIGN.md §1.1). This project's output (a ranked grasp-pose
list) crosses a process/node boundary to reach the IK and motion-planning stages, never a motor
directly.

**OS and real-time constraints.** Linux (or Windows) with a CUDA driver; NOT a hard-real-time
requirement — grasp scoring runs once per view inside a soft-deadline planning budget
(SYSTEM_DESIGN.md §1.1's "local planner" row, 20–100 ms), tolerant of the OS scheduling jitter a
general-purpose kernel introduces. Contrast the arm's own joint-control loop, which DOES need a
real-time OS or a dedicated real-time core.

**The ROS 2 node/topic shape it would take** (SYSTEM_DESIGN.md §3.6 message-shape convention):

```
Node: grasp_candidate_scorer
  subscribes:  /segmented_object_cloud   (sensor_msgs/PointCloud2  <-> this project's PointCloud)
  publishes:   /ranked_grasp_candidates  (a custom GraspCandidateArray msg: one Pose + width + score per grasp,
                                          shaped like this project's GraspScore/contact-point pair — see kernels.cuh)
```

Downstream, `/ranked_grasp_candidates` would be consumed by the 19.08-style reachability-ranking node
(README "System context"), which republishes the reachable subset for the motion planner (06.07).

**Calibration and bring-up.** Two calibrations matter before any grasp this project scores can be
trusted: (1) the depth camera's extrinsic calibration to the arm's base frame (a wrong `T_base_camera`
silently shifts every contact point and normal — the same `T_parent_child` discipline
SYSTEM_DESIGN.md §3.3 states repo-wide), and (2) the gripper's OWN stroke/force calibration (does
"closed" actually mean the modeled `gripper_w_max_m`? is the reported jaw position accurate?). Neither
calibration is performed by this project's code — it CONSUMES calibrated inputs and PRODUCES grasp
poses in whatever frame the input cloud arrived in.

**The safe hardware-testing ladder** (CLAUDE.md §1's caveat applies in full: everything in this
project is validated only in simulation, on synthetic point clouds):

1. **Simulation** — exactly this project's demo: synthetic clouds, no hardware, full analytic
   ground truth to check against.
2. **HIL / bench** — feed REAL depth-camera scans of REAL objects (still no arm motion) into this
   project's pipeline; compare ranked grasps against a human's judgment before trusting any of them.
3. **Tethered / current-limited execution** — command the gripper (not the arm) to close on a
   scored grasp with the arm STATIONARY and an E-stop within reach, current-limited so a bad grasp
   cannot damage the fingertip or object.
4. **Free running, fenced or collaborative** — only after stages 1–3 pass repeatedly, with the
   cell's own safety monitor (SYSTEM_DESIGN.md §2.2's "HUMAN SAFETY [21 →]" block; 21.04's
   speed-and-separation monitoring) active throughout.

No project in this repository is safety-certified; running any of this on real hardware is the
owner's decision and responsibility, at every rung of that ladder.

## 4. Business & regulatory context

**Who needs this.** Any operation that picks unstructured or loosely-structured parts with a robot
arm: bin picking (parts arriving jumbled in a tote), kitting, machine tending, and general
pick-and-place — the core workload of the manipulator work cell (SYSTEM_DESIGN.md §2.2). This is one
of the largest applied-robotics markets by dollar volume outside mobile-robot fleets, spanning
electronics assembly, warehousing/logistics, food handling, and general manufacturing.

**Main players.** Commercial and open-source grasp-planning software: **GPD** and **Dex-Net /
GQ-CNN** (research-grade, widely cited, partly open-source — README "Prior art"); **GraspIt!**
(academic, open-source, the classical force-closure simulator); and multiple commercial bin-picking
vision+grasp software stacks sold by major machine-vision and robotics-automation vendors, usually
bundled with a specific depth-camera/gripper combination as a turnkey cell. Gripper hardware itself
is a distinct, competitive commercial market (parallel-jaw, adaptive/underactuated, and vacuum/suction
grippers each have established vendors — PRACTICE.md's illustrative-parts caveat applies to naming
any of them specifically).

**What getting it wrong costs.** A grasp that LOOKS good geometrically but fails in practice costs a
dropped part (line stoppage, at minimum), a damaged part (scrap cost), or — worst case — a collision
between the gripper/arm and the bin or fixture (potential equipment damage and, if a person is in the
cell, a genuine safety event). This is why production stacks never rely on a single scoring pass: a
ranked TOP-M list (this project's actual output shape) lets the reachability and motion-planning
stages fall back to the next candidate when the first choice fails a downstream check, and why force
and slip sensing at the gripper (real hardware this project only approximates geometrically) is the
standard mitigation for "the grasp looked fine but the part is sliding out."

**Regulatory path** (SYSTEM_DESIGN.md §6.2's orientation table — didactic orientation, not compliance
guidance): industrial arms fall under **ISO 10218** (robot and system-integration safety) and, for
any cell where a person may be near the arm while it operates, **ISO/TS 15066** (collaborative
operation — power/force limits or speed-and-separation monitoring, the 21.04-style metrics
SYSTEM_DESIGN.md §2.2 shows watching this very work cell). Grasp-candidate scoring itself produces no
motion and carries no direct certification burden, but it sits inside a chain (scoring → reachability
→ motion planning → arm servo loops) whose FINAL output is exactly what those standards govern —
review and validation of the whole chain, not any one stage in isolation, is what a real safety case
requires.

**Where this work lives in a company** (SYSTEM_DESIGN.md §5.1): the **manipulation** sub-team within
controls/autonomy owns this code, working closely with **perception** (who provide the segmented
cloud this project consumes) and with whichever team owns kinematics/motion-planning (who consume
this project's ranked output). Adjacent teams: **ML/data** (if a learned scorer is layered on top,
README "Where this sits in the real world"), **QA/functional safety** (who validate the whole
perceive→grasp→move chain against ISO 10218/15066), and **mechanical/electrical engineering** (who
own the gripper hardware itself, PRACTICE.md §1–§2).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
