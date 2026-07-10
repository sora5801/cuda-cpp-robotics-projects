# 06.05 — STOMP: parallel noisy-rollout trajectory optimization (born for GPU): Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> STOMP is a planning *abstraction* — its physical carrier is the robot whose motion the trajectory
> commands and the compute module the planner runs on; sections 1–2 teach those honestly.

## 1. Building it — construction of the robot/part

STOMP produces no hardware — it produces a *trajectory*. The honest physical subject of this section is
therefore the machine that will execute that trajectory, because how it is built determines what a
"good" trajectory even means (the smoothness term and the collision margin in THEORY.md exist because
of the realities below). Two carriers, both from the README's reference robots:

- **Warehouse AMR (mobile).** A steel/aluminium chassis on differential or omni wheels, LiDAR + wheel
  encoders + IMU for state, a compute box bolted inside. Construction realities the planner inherits:
  wheel slip and caster shimmy (the executed path deviates from the planned one — hence *margin*),
  chassis flex and payload shift (the centre of mass the smoothness term implicitly protects moves),
  and mounting tolerances between the LiDAR frame and the drive frame (a miscalibrated `T_base_lidar`
  puts the cost map in the wrong place, so a "collision-free" plan clips a real shelf). What breaks in
  the field: encoders drift, LiDAR windows fog/scratch, and bumpers/E-stop wiring on a moving base
  fatigue.
- **6-DoF manipulator work cell (fixed).** Cast or extruded links, harmonic-drive or cycloidal gearboxes
  at each joint, absolute encoders, a controller cabinet. Realities the planner inherits: gearbox
  backlash and joint compliance (a kinked, high-jerk plan excites structural oscillation the smoothness
  term is there to avoid), and payload/tool changes that move the collision geometry. What breaks:
  cable-carrier flex fatigue, encoder-zero drift after a crash, gearbox wear changing the effective
  dynamics.

The GPU compute enclosure that runs the kernel — thermals, mounting, vibration isolation, connectorized
power — is shared with every GPU project in this repo (33.01 PRACTICE §1 covers it); nothing here is
built specially for STOMP.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-09. All parts named below are **illustrative examples, never endorsements**;
part numbers and prices go stale — verify current before relying on any of them.*

Where a real version of this planner runs and what feeds it:

| Piece | Illustrative choices (2026) | Role in this project's loop |
|---|---|---|
| Compute for the rollouts | Jetson Orin / Thor class on a mobile robot; x86 + discrete RTX in a work-cell controller or an offboard planning server (reference machine: RTX 2080 SUPER) | The K-rollout scoring kernel |
| Planning host CPU | The same box's ARM/x86 cores (Linux, often PREEMPT_RT) | The per-waypoint softmin update, the `M` matvecs, the plan/replan loop |
| Cost-map source | LiDAR (e.g. spinning or solid-state) + depth cameras → an occupancy/ESDF grid; or, for an arm, a depth camera → octomap → collision spheres | Produces the obstacle-cost FIELD this project consumes (07.09's SDF, or a `costmap_2d`/`nvblox` layer) |
| State estimation | Wheel encoders + IMU + LiDAR odometry (AMR); joint encoders (arm) | Where the planned trajectory starts (the fixed `start`) |
| Downstream drive silicon | Servo/BLDC drives: control MCU + gate driver + power stage + current-sense; encoders | Tracks the emitted trajectory (the consumer, not part of the planner) |
| E-stop chain | Certified dual-channel safety relay / safety controller | Must exist BEFORE any motion test (see §3) |

Cost tiers are dominated by the compute and the sensor suite, not by STOMP itself (it is software): a
hobby/research setup (Jetson + a low-cost LiDAR) runs in the low-thousands of dollars; an industrial
work-cell controller with certified servo drives runs to tens of thousands per cell. The planner's shape
is invariant across them — only the cost-map source and the dimension change.

## 3. Installation & integration — putting it on a real robot

**This project's output is a trajectory — a motion command in the making. The §1 caveat applies at full
strength: everything here is sim-validated on a synthetic map only; nothing below is a license to
actuate.**

- **Process/plugin shape.** On a ROS 2 robot STOMP runs as a **planning plugin**, not a bare node.
  MoveIt exposes exactly this: a `planning_interface::PlannerManager` / `PlanningContext` that receives
  a `MotionPlanRequest` (start state, goal constraints, the planning scene / collision world) and
  returns a `RobotTrajectory` (`trajectory_msgs/JointTrajectory`). The demo's structure maps one-to-one:
  build the cost field from the scene → precompute `M` → iterate GPU scoring + per-waypoint update →
  return the smoothed path. For an AMR the equivalent is a `nav2` controller/planner plugin consuming a
  `costmap_2d` and publishing a `nav_msgs/Path`.
- **Where it runs and its real-time character.** Planning is a **soft-real-time** activity (unlike the
  kHz control loop that tracks the plan): it runs on the application CPU/GPU, replanning at 10–50 Hz
  (SYSTEM_DESIGN §1.1/§4.1). A fielded planner warm-starts from the previous plan (so a few iterations
  suffice), monitors its own deadline, and holds the last valid plan on a miss — a stochastic optimizer
  degrades gracefully because a slightly-stale smooth plan is still a plan. The GPU call sits inside the
  planning loop; it never sits in the control loop.
- **Buses it touches.** The planner consumes a cost map over the compute fabric (Ethernet/shared memory)
  and emits a trajectory to the controller, which is what actually commands the drives over **EtherCAT /
  CAN-FD** (SYSTEM_DESIGN item 6). STOMP itself commands no bus directly — it hands a path to the layer
  below. (N/A: no fieldbus is implemented in this teaching project; the "robot" is a point in a synthetic map.)
- **Calibration / bring-up.** The plan is only as good as the cost map's frame: calibrate `T_base_sensor`
  (a wrong extrinsic silently offsets every obstacle), verify the inflation radius against the robot's
  real footprint plus a safety margin, and confirm the start state matches the estimator.
- **The safe testing ladder (CLAUDE.md §1), rung by rung, with E-stop at every rung:**
  1. *Simulation* — this demo, plus perturbed maps and a real collision model (Isaac/Gazebo/MoveIt scene).
  2. *HIL* — the planner feeding a real controller driving a simulated plant on the target compute, with
     deadline monitoring on.
  3. *Bench / tethered / speed-limited* — the real robot at a fraction of speed, workspace clear, E-stop
     tested first (press it; watch it work) before granting any motion authority.
  4. *Free running* — only inside a physical envelope (workspace limits, drive-level torque/velocity
     caps, a monitored safe set 31.01) that does not trust this software to behave.

## 4. Business & regulatory context

- **Who needs this capability.** Anyone shipping a robot that moves through a cluttered world:
  industrial-arm integrators (bin picking, welding, assembly around fixtures), warehouse/AMR fleet
  operators (navigation among racks and people), surgical and inspection robots, and increasingly
  humanoid and mobile-manipulation companies. Motion planning is the layer that turns "the robot knows
  where things are" into "the robot moves without hitting them."
- **The players.** Open source: **MoveIt** (and PickNik, its commercial steward) ships STOMP/CHOMP/OMPL
  plugins; **Nav2** owns mobile planning; **OMPL** is the sampling-planner backbone. Commercial/GPU:
  **NVIDIA cuRobo / Isaac** push GPU trajectory optimization; every major arm vendor (FANUC, KUKA, ABB,
  UR) ships a proprietary planner tuned to its controller. Build-vs-buy: most robotics companies *use*
  MoveIt/Nav2 or a vendor planner and *tune* it; a GPU-accelerated custom optimizer is a build decision
  only when cycle time or clutter defeats the stock planner (SYSTEM_DESIGN §5.3).
- **What getting it wrong costs.** A planner that emits a colliding or jerky path costs collisions
  (damaged robot, payload, fixtures — or, near people, injury), scrapped cycles, and downtime; in a
  fleet, a systematic planning bug is a recall-class event. The mitigations are architectural, not
  aspirational: a conservative collision model with margin, an independent runtime safety monitor
  (31.01 reachability / 31.04 CBF filters — this project's natural safety siblings), certified drive-level
  limits, and staged bring-up (§3).
- **Regulatory landscape** (orientation only; cite SYSTEM_DESIGN item 6). For **industrial arms**, the
  motion the planner produces is executed under **ISO 10218-1/-2** (robot + cell safety) and, for
  collaborative operation, **ISO/TS 15066** (speed-and-separation, power-and-force limiting — the
  planner must respect the separation the safety function enforces). For **service/mobile robots**,
  **ISO 13482**. For **AVs**, **ISO 26262 / UL 4600**. A GPU-hosted stochastic planner is not itself a
  certified safety function — fielded architectures keep the *safety* case in a deterministic monitor and
  certified drive limits *around* the planner, exactly as §3 describes.
- **Where the work lives.** The **controls & autonomy** team owns this (SYSTEM_DESIGN §5.1 org map:
  domains 04–09/23), with titles like motion-planning engineer / robotics software engineer (autonomy).
  Adjacent teams: **perception** (supplies the cost map this planner consumes), **controls** (owns the
  tracking controller downstream), and **QA & functional safety** (owns the envelope and has veto power).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
