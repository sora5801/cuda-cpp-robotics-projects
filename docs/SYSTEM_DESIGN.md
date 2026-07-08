# SYSTEM_DESIGN.md — The Robot Every Project Plugs Into

> **What this document is.** Every one of the ~505 projects in this repository implements one small,
> GPU-accelerated piece of a robot. This document is the shared architecture those pieces plug into.
> Every project `README.md` cites it from its **System context** section (CLAUDE.md §4.1 item 4), and
> every project `PRACTICE.md` grounds its physical/commercial claims in items 5–6 here (CLAUDE.md §4.3).
> Read this document once, early — then every kernel you study will have an address inside a machine.
>
> **What this document is not.** It is a *teaching map*, not a production architecture, a business
> plan, or compliance guidance. Sections 5 and 6 in particular are didactic orientation — see the
> labels there.
>
> Contract: [`../CLAUDE.md`](../CLAUDE.md) §3.1. Catalog: [`../catalog.json`](../catalog.json)
> (505 projects, 36 domains). Build instructions: [`BUILD_GUIDE.md`](BUILD_GUIDE.md).

---

## Contents

1. [The canonical autonomy stack](#1-the-canonical-autonomy-stack)
2. [Five reference robots](#2-five-reference-robots)
3. [Interface conventions shared by all projects](#3-interface-conventions-shared-by-all-projects)
4. [The composition map — chains through the repo](#4-the-composition-map--chains-through-the-repo)
5. [The whole: anatomy of a robotics company](#5-the-whole-anatomy-of-a-robotics-company)
6. [Robot internals & the regulatory map](#6-robot-internals--the-regulatory-map)
7. [What your README "System context" section must quote](#7-what-your-readme-system-context-section-must-quote)

---

## 1. The canonical autonomy stack

Almost every autonomous robot — warehouse cart, surgical arm, quadruped, drone, self-driving car —
is organized as the same pipeline: sense the world, estimate what is true, predict what happens next,
plan what to do, control the actuators that do it. Around that pipeline sit four cross-cutting
concerns that touch every layer. This is the reference frame for the whole repository: **every project
README states which box (or boundary between boxes) it lives in.**

```
                 ┌─────────────────────────────────────────────────────────────────┐
                 │                        THE AUTONOMY STACK                        │
                 └─────────────────────────────────────────────────────────────────┘

   physical world
        │  photons, pressure, fields, inertia
        ▼
  ┌───────────┐   ┌────────────┐   ┌──────────────────────┐   ┌────────────┐
  │  SENSORS  │──▶│ PERCEPTION │──▶│  STATE ESTIMATION /  │──▶│ PREDICTION │
  │ camera,   │   │ detect,    │   │  WORLD MODEL         │   │ where will │
  │ LiDAR,    │   │ segment,   │   │ where am I? what is  │   │ things be  │
  │ radar,    │   │ depth,     │   │ around me? (filters, │   │ in 1..10 s?│
  │ IMU, F/T, │   │ flow,      │   │ SLAM, maps, fusion)  │   │            │
  │ encoders  │   │ features   │   │                      │   │            │
  └───────────┘   └────────────┘   └──────────────────────┘   └─────┬──────┘
   domains 01,02,   domains 01,02,    domains 04, 05, 20            │
   03, 20 (+11 sim)  03, 12           (+23 costmaps)                ▼
                                                     ┌───────────────────────────┐
                                                     │  PLANNING                 │
                                                     │  global route  →  local   │
                                                     │  trajectory (obstacle-    │
                                                     │  aware, dynamic-feasible) │
                                                     └────────────┬──────────────┘
                                                       domains 06, 23, 13, 15…
                                                                  ▼
                    ┌───────────┐                    ┌───────────────────────────┐
   physical world ◀─│ ACTUATION │◀───────────────────│  CONTROL                  │
   torque, thrust,  │ motors,   │  torque/velocity/  │  track the trajectory:    │
   force, motion    │ drivers,  │  current commands  │  MPC, MPPI, LQR, WBC,     │
                    │ gearboxes │                    │  servo loops              │
                    └───────────┘                    └───────────────────────────┘
                     domains 24, 25                    domains 08, 09 (models)

  ─────────────────────────── CROSS-CUTTING (touch every layer) ───────────────────────────
  ┌────────────────────────┐ ┌──────────────┐ ┌────────────────┐ ┌────────────────────────┐
  │ SIMULATION & DIGITAL   │ │ LEARNING     │ │ SAFETY MONITOR │ │ INFRASTRUCTURE         │
  │ TWIN — sensor sim,     │ │ — training,  │ │ — reachability,│ │ — compute, comms,      │
  │ physics, HIL           │ │ inference,   │ │ CBFs, runtime  │ │ power, mechanical      │
  │ domains 10, 11         │ │ RL           │ │ verification   │ │ structure              │
  │                        │ │ domain 12    │ │ domain 31      │ │ domains 25,26,27,32,33 │
  └────────────────────────┘ └──────────────┘ └────────────────┘ └────────────────────────┘
```

Domains not named above slot in by robot type rather than by layer: locomotion (13–18) and
manipulation (19) each *span* perception→control for their morphology; HRI (21), swarms (22), and
navigation (23) are stack slices; soft/medical/field/micro/modular robotics (28–30, 35, 36) are the
same stack under unusual physics; 34 is the theory frontier that feeds all of it.

### 1.1 Rates and latency budgets at each boundary

Robotics is a *real-time* systems discipline: every arrow in the diagram above carries data at a
characteristic rate, and every box has a latency budget it must meet or the robot oscillates, drifts,
or crashes. These numbers (from CLAUDE.md §3.1) are the ones every project README quotes:

| Boundary / loop                          | Typical rate      | Latency budget (rule of thumb)       | What breaks if you miss it                          |
|------------------------------------------|-------------------|--------------------------------------|-----------------------------------------------------|
| Camera → perception                      | **30–60 Hz**      | < 1 frame (16–33 ms) end-to-end      | Stale detections; motion blur relative to control   |
| LiDAR → perception/mapping               | **10–20 Hz**      | < 1 scan period (50–100 ms)          | Map lag; obstacles appear "late"                    |
| IMU → state estimator                    | 200–1000 Hz       | sub-millisecond timestamping         | Attitude drift, bad preintegration                  |
| State estimator output                   | **100–400 Hz**    | few ms; jitter matters as much       | Controllers act on stale/jittery state              |
| Global planner (route)                   | 0.1–1 Hz          | 100 ms – seconds (event-driven)      | Suboptimal routes only — usually survivable         |
| Local planner / trajectory replan        | **10–50 Hz**      | 20–100 ms per replan                 | Cannot dodge dynamic obstacles                      |
| Whole-body / trajectory-tracking control | **0.5–1 kHz**     | 1–2 ms, *hard* deadline              | Instability: a legged robot literally falls over    |
| Motor current (torque) loops             | **10–20 kHz**     | 50–100 µs, in silicon/MCU firmware   | Motor cogging, overheating, blown FETs              |

Two lessons the learner should internalize:

- **Rates decrease as you go up the stack, and stakes per tick decrease with them.** A missed
  current-loop tick can destroy hardware in milliseconds; a missed global-replan tick costs you a
  slightly longer route. This is why the bottom of the stack lives in MCUs and FPGAs with hard
  real-time guarantees, and the top lives on Linux + GPU with soft deadlines.
- **Latency and rate are different budgets.** A perception pipeline at 30 Hz with 200 ms latency is
  *worse* for control than one at 15 Hz with 40 ms latency. Projects report both where they measure.

### 1.2 Where the GPU sits — classically, and at the frontier

**Classical GPU territory (most of this repo):** perception, mapping, planning, and simulation. These
layers are *data-parallel by nature* — millions of pixels/points (domains 01–03), thousands of
voxels/rays (05, 07, 11), thousands of candidate trajectories or IK seeds (06, 08, 09), thousands of
simulated environments (10, 12, 13). They also run at 10–60 Hz, where the ~10–100 µs cost of a kernel
launch and a PCIe copy is negligible against a 20–100 ms budget. This is why production stacks
(Isaac, Autoware, nvblox, cuRobo) put the GPU exactly here.

**The research frontier (the repo's `[R&D]`-flavored edge):** pushing the GPU *down* the stack into
the 0.5–1 kHz control layer and into safety monitors. At 1 kHz you have a 1 ms budget, and launch
overhead, scheduling jitter, and host↔device copies become first-order problems — the classic answers
are CUDA Graphs, persistent kernels, pinned zero-copy memory, and keeping the whole loop resident on
the device (domain 32, e.g. the CUDA-Graphs control loop; domain 08 sampling controllers like MPPI
that *earn* their place at kHz by evaluating 10⁴ rollouts per tick; domain 31 safety filters like CBF
batches evaluated at kHz). The motor current loop (10–20 kHz) remains MCU/silicon territory
essentially everywhere — projects that touch it (domain 24) simulate it on the GPU rather than
pretending to close it there.

---

## 2. Five reference robots

Abstract stacks are hard to love. Here are five concrete machines, each a *suggested system design*
the learner can grow toward by completing the mapped domains. Every project README names which of
these five it belongs to (often more than one). Block diagrams show `[domain →]` ownership; rates are
the §1.1 budgets specialized to the machine.

### 2.1 Warehouse AMR (autonomous mobile robot)

**What it does.** A ~50–150 kg differential-drive or omnidirectional cart that ferries totes/pallets
around a warehouse or factory floor, sharing space with people and forklifts. The commercially most
common autonomous robot on Earth — think fleet of hundreds per site, 20-hour duty cycles.

**Suite.** *Sensors:* 2× safety-rated 2D LiDAR (front/rear), 1× 3D LiDAR or 2–4 depth cameras, wheel
encoders, IMU, bumpers. *Compute:* one embedded GPU SoC (Jetson-class) or small x86 + entry dGPU,
plus a safety PLC/controller. *Actuation:* 2–4 BLDC hub or geared motors with off-the-shelf drives,
lift/latch actuator.

```
              WAREHOUSE AMR                                     rate budget
 ┌──────────────────────────────────────────────┐
 │ 2D/3D LiDAR, depth cams, encoders, IMU       │              LiDAR 10–20 Hz
 └───────────────┬──────────────────────────────┘              depth 30 Hz, IMU 200 Hz
                 ▼
 ┌──────────────────────────────────────────────┐
 │ POINT-CLOUD PERCEPTION      [02 →]           │              per-scan, < 50 ms
 │ downsample, ground seg, clustering, deskew   │
 └───────────────┬──────────────────────────────┘
                 ▼
 ┌──────────────────────────────────────────────┐
 │ LOCALIZATION & MAPPING      [04, 05 →]       │              estimator 100–400 Hz
 │ particle filter / scan matching vs. site map │              map update 10–20 Hz
 └───────────────┬──────────────────────────────┘
                 ▼
 ┌──────────────────────────────────────────────┐
 │ NAVIGATION STACK            [23, 06 →]       │              costmap 10–20 Hz
 │ layered costmaps → global route → DWA/MPPI   │              local plan 10–50 Hz
 │ local planner                                │
 └───────────────┬──────────────────────────────┘
                 ▼
 ┌──────────────────────────────────────────────┐   ┌────────────────────────────┐
 │ MOTION CONTROL              [08 →]           │   │ SAFETY MONITOR   [31 →]    │
 │ velocity/trajectory tracking → wheel cmds    │◀──│ field violations, reach-   │
 └───────────────┬──────────────────────────────┘   │ ability, watchdogs         │
                 ▼                                  └────────────────────────────┘
 ┌──────────────────────────────────────────────┐   ┌────────────────────────────┐
 │ WHEEL MOTORS + DRIVES                        │   │ POWER: Li-ion pack, BMS,   │
 │ (velocity loops in the drives, 10–20 kHz)    │   │ charge dock       [25 →]   │
 └──────────────────────────────────────────────┘   └────────────────────────────┘
        Embedded compute, comms, fleet uplink                        [32 →]
```

Domains: **02** LiDAR perception · **04** localization filters · **05** mapping · **23** costmaps &
navigation · **06** planning · **08** control · **25** battery/energy · **31** safety · **32**
embedded infrastructure.

### 2.2 6-DoF manipulator work cell

**What it does.** An industrial arm (payload 3–20 kg) in a fenced or collaborative cell doing
pick-and-place, bin picking, machine tending, or assembly. Cycle time is money: the whole
perceive→plan→move loop should fit in a few hundred milliseconds.

**Suite.** *Sensors:* wrist- or frame-mounted stereo/structured-light 3D camera, joint encoders (in
the arm), optional F/T sensor at the wrist, cell-guarding sensors (light curtain / safety scanner).
*Compute:* industrial PC + discrete GPU beside the vendor's arm controller. *Actuation:* 6 servo
joints (the vendor closes those loops); gripper or suction end-effector.

```
              MANIPULATOR WORK CELL                             rate budget
 ┌──────────────────────────────────────────────┐
 │ 3D CAMERA over bin / workspace               │               camera 30–60 Hz
 └───────────────┬──────────────────────────────┘
                 ▼
 ┌──────────────────────────────────────────────┐
 │ VISION: stereo depth, segmentation,          │               < 100 ms per view
 │ object pose               [01 →]             │
 └───────────────┬──────────────────────────────┘
                 ▼
 ┌──────────────────────────────────────────────┐   ┌────────────────────────────┐
 │ GRASP PLANNING            [19 →]             │   │ HUMAN SAFETY     [21 →]    │
 │ candidate sampling + scoring                 │   │ speed-and-separation       │
 └───────────────┬──────────────────────────────┘   │ monitoring of the cell     │
                 ▼                                  └──────────────┬─────────────┘
 ┌──────────────────────────────────────────────┐                  │ slow/stop
 │ KINEMATICS: batched IK, reachability,        │                  │ overrides
 │ Jacobians                 [09 →]             │                  ▼
 └───────────────┬──────────────────────────────┘   ┌────────────────────────────┐
                 ▼                                  │ JOINT CONTROL    [08 →]    │
 ┌──────────────────────────────────────────────┐   │ trajectory tracking        │
 │ ARM MOTION PLANNING       [06, 07 →]         │──▶│ 0.5–1 kHz (vendor          │
 │ collision-aware trajectory opt (cuRobo-style)│   │ controller interpolates)   │
 │ plan in 10–100 ms for cycle time             │   └──────────────┬─────────────┘
 └──────────────────────────────────────────────┘                  ▼
                                                    ┌────────────────────────────┐
                                                    │ SERVO JOINTS + GRIPPER     │
                                                    │            [24 →]          │
                                                    │ current loops 10–20 kHz    │
                                                    └────────────────────────────┘
```

Domains: **01** camera perception · **19** grasping · **09** kinematics/dynamics · **06** planning ·
**07** collision geometry · **08** control · **21** HRI/safety monitoring · **24** actuators.

### 2.3 Quadruped

**What it does.** A 10–50 kg four-legged robot for inspection, security patrol, and research —
terrain a wheeled robot cannot cross. The defining challenge: it is *dynamically unstable*; balance
is re-earned 500–1000 times per second.

**Suite.** *Sensors:* depth cameras and/or LiDAR for terrain, high-rate IMU (400–1000 Hz), joint
encoders and (often) joint-torque sensing, foot contact sensors. *Compute:* Jetson-class GPU SoC for
perception/planning + a real-time MCU or RT-Linux core for whole-body control. *Actuation:* 12
quasi-direct-drive or geared BLDC joints.

```
              QUADRUPED                                          rate budget
 ┌──────────────────────────────────────────────┐
 │ depth/LiDAR   IMU + joint encoders + feet    │               depth 30 Hz
 └──────┬────────────────┬─────────────────────┘                IMU 400–1000 Hz
        ▼                ▼
 ┌───────────────┐  ┌──────────────────────────┐
 │ ELEVATION /   │  │ LEGGED STATE ESTIMATION  │                estimator 100–400 Hz
 │ TERRAIN MAP   │  │ [04 →] invariant EKF,    │
 │ [05 →]        │  │ contact estimation       │
 └──────┬────────┘  └──────────┬───────────────┘
        ▼                      │
 ┌──────────────────────────────────────────────┐
 │ LOCOMOTION PLANNING       [13 →]             │                footholds 10–50 Hz
 │ foothold scoring, gait timing,               │
 │ centroidal MPC                               │
 └───────────────┬──────────────────────────────┘
                 ▼
 ┌──────────────────────────────────────────────┐   ┌────────────────────────────┐
 │ WHOLE-BODY CONTROL / RL POLICY               │   │ TRAINED IN SIMULATION      │
 │ [13, 12 →]  0.5–1 kHz — HARD deadline        │◀══│ [10 →] 10k parallel envs,  │
 │ (miss it and the robot falls)                │   │ [12 →] RL policy training  │
 └───────────────┬──────────────────────────────┘   └────────────────────────────┘
                 ▼
 ┌──────────────────────────────────────────────┐   ┌────────────────────────────┐
 │ 12 JOINT ACTUATORS        [24 →]             │   │ BATTERY & THERMALS [25 →]  │
 │ torque control, current loops 10–20 kHz      │   │ 30–90 min runtime reality  │
 └──────────────────────────────────────────────┘   └────────────────────────────┘
```

Domains: **13** legged locomotion · **04** state estimation · **05** elevation mapping · **10**
parallel physics sim · **12** RL/learning · **24** actuators · **25** power. Note the sim→real
pattern: the GPU's biggest quadruped contribution is *offline* — thousands of simulated robots
training the policy that then runs in milliseconds onboard.

### 2.4 Quadrotor

**What it does.** A 0.5–5 kg multirotor for inspection, mapping, cinematography, or racing.
Aggressive dynamics (attitude time constants of tens of milliseconds), severe SWaP limits (every
gram of compute costs flight time), and no "stop and think" — it must keep flying while it plans.

**Suite.** *Sensors:* IMU (1 kHz), cameras (VIO + task), barometer, GPS/GNSS outdoors, optionally a
small LiDAR. *Compute:* flight-controller MCU (PX4-class) + companion computer (Jetson-class) for
vision/planning. *Actuation:* 4 ESC-driven BLDC rotors.

```
              QUADROTOR                                          rate budget
 ┌──────────────────────────────────────────────┐
 │ IMU 1 kHz · cameras 30–60 Hz · GNSS 10 Hz    │
 └───────────────┬──────────────────────────────┘
                 ▼
 ┌──────────────────────────────────────────────┐   ┌────────────────────────────┐
 │ STATE ESTIMATION          [04 →]             │   │ SENSOR/VEHICLE SIMULATION  │
 │ VIO / EKF fusion, 100–400 Hz output          │   │ [11 →] IMU error models,   │
 └───────────────┬──────────────────────────────┘   │ camera sim, wind fields —  │
                 ▼                                  │ test before you fly        │
 ┌──────────────────────────────────────────────┐   └────────────────────────────┘
 │ TRAJECTORY PLANNING       [15 →]             │
 │ minimum-snap / time-optimal, replan 10–50 Hz │   ┌────────────────────────────┐
 └───────────────┬──────────────────────────────┘   │ MULTI-DRONE COORDINATION   │
                 ▼                                  │ [22 →] swarm planning,     │
 ┌──────────────────────────────────────────────┐   │ collision avoidance        │
 │ FLIGHT CONTROL            [08, 15 →]         │◀──│ (when flying as a fleet)   │
 │ position/attitude MPC or MPPI,               │   └────────────────────────────┘
 │ 0.5–1 kHz attitude loop on the FC MCU        │
 └───────────────┬──────────────────────────────┘
                 ▼
 ┌──────────────────────────────────────────────┐
 │ CONTROL ALLOCATION → 4 ESCs → rotors         │
 │ ESC current loops 10–20 kHz                  │
 └──────────────────────────────────────────────┘
```

Domains: **15** aerial locomotion · **04** state estimation · **08** control · **11** sensor
simulation · **22** swarms.

### 2.5 Autonomous-vehicle stack

**What it does.** A passenger car or shuttle driving public roads. The maximal version of the stack:
every sensor modality, redundant everything, and the heaviest regulatory burden in robotics. Even as
a study target it is the best "final exam" because it exercises nearly every domain at production
scale.

**Suite.** *Sensors:* 6–12 cameras, 1–5 LiDARs, 3–6 radars, ultrasonics, GNSS+RTK, high-grade IMU,
wheel odometry. *Compute:* automotive-grade GPU compute (100s of TOPS) + safety-rated MCUs with
locked-step cores. *Actuation:* drive-by-wire steering, throttle, brake — each with its own safety
path.

```
              AUTONOMOUS VEHICLE                                 rate budget
 ┌──────────────────────────────────────────────────────────────┐
 │ cameras ×8   LiDAR ×3    radar ×5     GNSS/IMU/odometry      │   30–60 / 10–20 /
 └──────┬──────────┬───────────┬───────────────┬────────────────┘   15–25 Hz / 100 Hz
        ▼          ▼           ▼               │
 ┌───────────┐┌───────────┐┌────────────┐      │
 │ VISION    ││ LIDAR     ││ RADAR      │      │                  per-frame/scan
 │ [01 →]    ││ [02 →]    ││ [03 →]     │      │                  budgets, < 100 ms
 └─────┬─────┘└─────┬─────┘└─────┬──────┘      │                  sensor→objects
       └─────┬──────┴────────────┘             │
             ▼                                 ▼
 ┌──────────────────────────────┐  ┌───────────────────────────┐
 │ FUSION & TRACKING [04 →]     │  │ LOCALIZATION vs. HD MAP   │  estimator
 │ multi-target, multi-sensor   │  │ [05 →]                    │  100–400 Hz
 └─────────────┬────────────────┘  └────────────┬──────────────┘
               └────────────┬───────────────────┘
                            ▼
 ┌──────────────────────────────────────────────┐  ┌───────────────────────────┐
 │ PREDICTION + BEHAVIOR + MOTION PLANNING      │  │ SAFETY CASE     [31 →]    │
 │ [06 →] route → lattice/hybrid-A* → local     │  │ reachability, runtime     │
 │ trajectory, replan 10–50 Hz                  │  │ monitors, scenario farms, │
 └─────────────┬────────────────────────────────┘  │ redundant channel         │
               ▼                                   └────────────┬──────────────┘
 ┌──────────────────────────────────────────────┐               │ independent
 │ VEHICLE CONTROL           [14, 08 →]         │◀──────────────┘ stop path
 │ lateral+longitudinal tracking, 100 Hz–1 kHz  │
 └─────────────┬────────────────────────────────┘
               ▼
 ┌──────────────────────────────────────────────┐
 │ DRIVE-BY-WIRE: steer/brake/throttle ECUs     │   safety-rated MCUs,
 │ + EMBEDDED PLATFORM        [32 →]            │   locked-step, watchdogs
 └──────────────────────────────────────────────┘
```

Domains: **01** cameras · **02** LiDAR · **03** radar · **04** fusion/tracking · **05**
localization/HD maps · **06** planning · **14** wheeled vehicle dynamics/control · **31** safety &
verification · **32** embedded systems.

---

## 3. Interface conventions shared by all projects

Projects in this repo never link against each other (§4 self-containment) — but they *speak the same
language*, so a learner can see exactly how output of one becomes input of another, and how both map
onto a real middleware like ROS 2. These conventions are binding on all `src/` code (CLAUDE.md §12).

### 3.1 Units: SI, always, named

Meters, seconds, radians, kilograms, newtons, N·m, volts, amps. **No degrees, no millimeters, no
feet** inside computation — convert at I/O edges only, loudly. Variable names carry units where
ambiguity is possible: `dt_s`, `range_m`, `omega_rad_s`, `torque_nm`, `voltage_v`. A comment that
omits units on a physical quantity is a defect.

### 3.2 Frames: right-handed, named, x-forward/y-left/z-up

- All frames are **right-handed**. The default body convention is **x-forward, y-left, z-up**
  (the ROS REP-103 convention). Domains with a strong contrary standard (e.g., camera optics uses
  z-forward/x-right/y-down; aerospace often uses NED) may use it **but must state so at every API
  boundary**.
- Common frame names, used consistently: `world` (fixed, gravity-aligned), `map` (SLAM origin),
  `odom` (drifting continuous odometry origin), `base` (robot body), `camera`, `lidar`, `imu`,
  `ee` (end-effector), `joint<i>`.

### 3.3 Transforms: `T_parent_child` — "child expressed in parent"

A rigid transform named `T_parent_child` takes a point expressed in the *child* frame and re-expresses
it in the *parent* frame:

```
p_world = T_world_base * p_base          // a point on the robot, in world coordinates
```

The naming makes composition self-checking — inner frame names must match and cancel:

```
T_world_camera = T_world_base * T_base_camera      //  base cancels: world←base←camera. OK.
T_world_camera = T_base_world * T_base_camera      //  world/base mismatch — WRONG, and visibly so.
```

Inverses swap the names: `T_child_parent = inverse(T_parent_child)`. Every function that takes or
returns a transform documents it in this notation.

### 3.4 Quaternions: `(w, x, y, z)`, unit-norm, documented at every boundary

Quaternions are stored **scalar-first `(w, x, y, z)`** and kept normalized (renormalize after chains
of multiplications — see any project's THEORY.md on normalization drift). Because half the world
(Eigen ctor, ROS msg fields) is `(x, y, z, w)` scalar-last, **every API boundary that passes a
quaternion re-states the order in its comment**. No exceptions; this is the single most common
silent-corruption bug in robotics code.

### 3.5 Time: monotonic seconds, `double`

Timestamps are seconds as `double` from a **monotonic** clock (never wall clock — NTP steps and DST
have crashed real robots). A `double` holds ~microsecond resolution over years of uptime, which is
sufficient at our layers; where sub-microsecond matters (IMU hardware timestamping), projects say so.
Durations are also seconds: `dt_s`.

### 3.6 Message-shaped structs — the ROS 2 lookalikes

Projects exchange data (conceptually, and via files in `data/sample/`) using plain C++ structs that
deliberately mirror ROS 2 message types, so the day the learner meets real middleware, every field is
already familiar. Canonical sketches (each project copies what it needs into its own `src/`, per the
self-containment rule, and comments every field with units and frame):

```cpp
// A timestamp + frame pair: prefix of every message, mirroring std_msgs/Header.
struct Header {
    double      stamp_s;    // monotonic time of measurement/validity, seconds (§3.5)
    const char* frame_id;   // frame the data is expressed in, e.g. "lidar" (§3.2)
};

// mirrors sensor_msgs/PointCloud2 (flattened & simplified for teaching)
struct PointCloud {
    Header header;
    int    num_points;      // number of valid points
    float* xyz;             // [num_points*3] interleaved x,y,z in meters, header.frame_id frame
    float* intensity;       // [num_points] sensor-specific return strength, may be nullptr
};

// mirrors sensor_msgs/Image
struct Image {
    Header   header;
    int      width, height; // pixels
    int      channels;      // 1 = gray/depth, 3 = RGB
    float*   data;          // [height*width*channels] row-major; depth images are meters
};

// mirrors sensor_msgs/Imu — the (w,x,y,z) order is restated per §3.4
struct Imu {
    Header header;
    float  orientation_wxyz[4];   // unit quaternion, SCALAR-FIRST (w,x,y,z), body←world
    float  angular_velocity[3];   // rad/s, body frame
    float  linear_acceleration[3];// m/s^2, body frame, INCLUDES gravity (sensor convention)
};

// mirrors geometry_msgs/Twist — the universal mobile-robot velocity command
struct Twist {
    float linear[3];   // m/s   (vx, vy, vz) in the body frame (x-forward, y-left, z-up)
    float angular[3];  // rad/s (wx, wy, wz) body frame; wz is the "turn rate" for planar robots
};

// mirrors sensor_msgs/JointState — arms, legs, anything articulated
struct JointState {
    Header  header;
    int     num_joints;
    float*  position;   // [num_joints] rad (revolute) or m (prismatic) — document per robot
    float*  velocity;   // [num_joints] rad/s or m/s
    float*  effort;     // [num_joints] N*m or N
};

// mirrors nav_msgs/OccupancyGrid — the planner's world
struct OccupancyGrid {
    Header  header;
    int     width, height;   // cells
    float   resolution_m;    // meters per cell edge
    float   origin_xy[2];    // world coords of cell (0,0) corner, meters
    int8_t* data;            // [height*width] row-major; -1 unknown, 0 free ... 100 occupied
};
```

The mapping to real middleware is then one-to-one: `PointCloud` ↔ `sensor_msgs/msg/PointCloud2`,
`Twist` ↔ `geometry_msgs/msg/Twist`, and a project's "consumes X / produces Y" line reads exactly
like a ROS 2 node's subscribed/published topics (each project's `PRACTICE.md` §3 spells out that node
shape).

### 3.7 Angle wrapping: `(-π, π]`, at defined points only

Angles wrap to the half-open interval **(-π, π]** — but *only at documented normalization points*
(after integration steps, before shortest-arc differences, at message boundaries). Wrapping
opportunistically inside math breaks derivatives and interpolation; every project that touches
orientation states *where* it wraps. Angle *differences* always go through a `wrap_to_pi(a - b)`
helper, never raw subtraction.

---

## 4. The composition map — chains through the repo

How do ~505 study projects become a robot? By chaining: the *output data shape* of one project is the
*input data shape* of another (§3.6 structs), so the learner can trace a full sensor→actuator path
through the repo. Four worked chains follow — one per reference robot class.

> **Self-containment reminder (CLAUDE.md §4).** These chains are **conceptual**: data-shape
> compatible, rate-compatible, teachable end-to-end. Projects are **never** coupled at build or run
> time — every project builds and demos alone, on committed sample data. Where a chain is exercised
> concretely, an upstream project's tiny output is *copied into* the downstream project's
> `data/sample/` and labeled with its provenance. No project references another project's folder.

### 4.1 Chain A — mobile robot mapping & planning (the AMR spine)

The canonical chain, quoted in CLAUDE.md §3.1: from photons (simulated) to torques, with a safety
monitor watching.

```
 [11.01 GPU LiDAR simulator] ── PointCloud (10–20 Hz) ──▶ [02.06 ICP registration]
        sensor sim: BVH raycasting,                            scan-to-scan/scan-to-map pose,
        beam divergence, dropout noise                         feeds the estimator at scan rate
                                                                       │ T_map_base (10–20 Hz,
                                                                       ▼  smoothed to 100–400 Hz)
 [31.01 HJ reachability]                                  [05.01 TSDF fusion]
        offline: compute the safe set;   ◀── watches ──        integrate depth into a voxel
        online: is the state leaving it?      state            volume; extract surface mesh
        (safety monitor, cross-cutting)                                │ voxel grid (10–20 Hz)
                                                                       ▼
 [08.01 MPPI controller] ◀── trajectory (10–50 Hz) ── [06.05 STOMP] ◀── [07.09 jump-flooding
        10⁴ GPU rollouts per tick,             noisy-rollout                distance field]
        control output 0.5–1 kHz               trajectory optimization      obstacle costs +
        → Twist / wheel commands               over the distance field      gradients from the map
```

Rates it must meet (§1.1): LiDAR-rate front end (10–20 Hz, < 100 ms), estimator-rate pose output
(100–400 Hz), planner replans at 10–50 Hz, MPPI control at 0.5–1 kHz. Reference robot: **warehouse
AMR** (§2.1); much of it transfers directly to the **AV** (§2.5).

### 4.2 Chain B — manipulator pick-and-place (the work-cell spine)

From a stereo pair to a collision-free grasp motion. Reference robot: **manipulator work cell**
(§2.2). Cycle time is the budget: the whole chain should complete in a few hundred ms per pick.

```
 [01.02 Stereo depth (SGM)] ── depth Image (30–60 Hz) ──▶ [19.01 Antipodal grasp scoring]
        block matching → SGM kernels                            sample + score thousands of
        on the wrist camera pair                                grasp candidates on the cloud
                                                                       │ ranked grasp poses
                                                                       │ (per view, < 100 ms)
                                                                       ▼
 [09.05 Batched numerical IK] ◀── grasp poses ── damped-least-squares IK with random
        thousands of seeds in parallel:            restarts: WHICH grasps can the arm
        feasible joint configs per grasp           actually reach, and how?
                │ joint-space goals
                ▼
 [06.07 cuRobo-style arm planner] ── JointState trajectory ──▶ [08.03 iLQR/DDP tracking control]
        massively parallel seeded IK +                              batched line search; tracks the
        trajectory opt + collision checks                           plan at 0.5–1 kHz (vendor servo
        (uses 07.x collision kernels);                              loops close current at 10–20 kHz)
        plan in 10–100 ms for cycle time
```

With **[21.04 speed-and-separation monitoring]** watching the cell as the cross-cutting safety layer
(slows or stops the arm when a person approaches — didactic ISO/TS 15066-style metrics, not a
certified implementation).

### 4.3 Chain C — quadruped locomotion (the legged spine)

From simulated depth to foot torques — plus the offline sim→real loop that makes modern quadrupeds
work. Reference robot: **quadruped** (§2.3).

```
 OFFLINE (train before you walk):
 [10.03 10k-env parallel robot sim] ══ experience ══▶ [12.05/12.06 RL training kernels]
        thousands of simulated quadrupeds                PPO/GAE on GPU → trained policy
        (dynamics via 09.03-style ABA/RNEA)              weights, deployed below

 ONLINE (on the robot):
 [11.02 Depth-camera simulator] ── depth Image (30 Hz) ──▶ [05.05 Elevation (2.5D) mapping]
        stands in for the real sensor                          height + uncertainty grid
                                                               around the robot (10–30 Hz)
 [04.08 Invariant EKF (legged)]                                        │
   + [04.09 contact estimation]                                        ▼
        IMU + kinematics + feet →                          [13.03 Foothold scoring kernels]
        base pose/velocity at 100–400 Hz                       slope, roughness, edge distance
                │ state                                        per candidate foothold (10–50 Hz)
                ▼                                                      │ footholds
 [13.02 Centroidal MPC] ◀──────────────────────────────────────────────┘
        batched QP + gait-timing search; body trajectory + contact forces (50–100 Hz)
                │
                ▼
 whole-body control / RL policy at 0.5–1 kHz  ──▶  12 joint torque commands
        (dynamics terms from [09.03 GPU Featherstone ABA/RNEA])   current loops 10–20 kHz
```

The hard deadline lives at the bottom: miss the 1 kHz whole-body tick and the robot falls. That is
precisely the §1.2 frontier — most of this chain's GPU work runs at 10–100 Hz or offline.

### 4.4 Chain D — quadrotor flight (the aerial spine)

From simulated IMU noise to rotor commands. Reference robot: **quadrotor** (§2.4).

```
 [11.06 IMU error simulation] ── Imu stream (200–1000 Hz) ──▶ [04.07 Sliding-window VIO backend]
        bias random walks, noise                                   fuses IMU + camera features;
        across Monte Carlo runs                                    GPU marginalization; state out
        (test the estimator before flying)                         at 100–400 Hz
                                                                       │ T_world_base, velocity
                                                                       ▼
 [31.04 CBF safety filter]                              [15.01 Minimum-snap trajectories]
        evaluated at kHz over the      ◀── watches ──        batched over waypoint sets;
        commanded control set;             commands          replan 10–50 Hz
        minimally modifies unsafe                                      │ trajectory
        commands                                                       ▼
                └────── filtered cmds ────── [15.03 MPPI/NMPC quadrotor control]
                                                   position/attitude loops 0.5–1 kHz
                                                        │ body wrench
                                                        ▼
                                             [08.12 Control allocation (rotors)]
                                                   batched QP: wrench → 4 rotor thrusts
                                                   → ESCs (current loops 10–20 kHz)
```

### 4.5 Reading the map

Each chain is also a **suggested study order**: build the upstream project first, look at its output
artifact in `demo/`, then open the downstream project and find the struct that consumes exactly that
shape. When you have walked one chain end-to-end you understand a robot; when you can *design* a new
chain (the catalog supports hundreds), you understand robotics. Project READMEs name the chain(s)
they belong to in their System context section (§7 below).

---

## 5. The whole: anatomy of a robotics company

> **Label: didactic sketch — not business, hiring, or investment advice.** Real companies vary
> wildly; this is the *median* shape, drawn so the learner can place each repo domain on an org chart
> and understand who would own their code. `PRACTICE.md` §4 in every project cites this section.

### 5.1 The org map — and which domains each team owns

A robotics company is mostly *not* autonomy engineers. A typical mid-size robot maker (100–1000
people) looks like this; the right column shows the repo domains each team would own or use daily:

| Team | What they do | Repo domains they own/use |
|------|--------------|---------------------------|
| **Mechanical engineering** | Structures, linkages, enclosures, thermal, DFM | 26 (design/structures), 27 (materials/mfg), 28 (soft), 36 (modular) |
| **Electrical engineering** | PCBs, power electronics, motor drives, harnessing, EMI | 24 (actuators), 25 (power) |
| **Embedded / firmware** | MCU code, RTOS, drivers, buses (CAN/EtherCAT), bootloaders, OTA | 32 (embedded infra), 24 (motor control fw) |
| **Perception** | Cameras, LiDAR, radar pipelines; calibration; detection | 01, 02, 03, 20 (tactile) |
| **Controls & autonomy** | State estimation, SLAM, planning, control | 04, 05, 06, 07, 08, 09, 23, + 13–18 per morphology |
| **ML / data** | Model training, datasets, labeling, deployment (TensorRT) | 12, plus data tooling everywhere |
| **Simulation & tools** | Sim environments, digital twins, CI farms, internal libs | 10, 11, 33 (foundational GPU libs), 34 |
| **Manufacturing & supply chain** | BOM, vendors, factory line, test fixtures | consumes 26/27; drives cost reality in §5.4 |
| **QA & functional safety** | Test plans, HIL rigs, safety cases, standards evidence | 31 (safety/verification), 21 (HRI safety) |
| **Fleet operations** | Deployed-robot monitoring, remote ops, incident response | 31.07 (telemetry anomaly), 32, 22 (fleet coordination) |
| **Product management** | What to build, for whom, at what price | consumes everything; owns nothing in-repo |
| **Regulatory / compliance** | Certifications, audits, documentation trail | consumes 31; the §6.2 map is their world |
| **Sales, support, field service** | Selling, installing, fixing robots on site | consume PRACTICE.md-style knowledge daily |

Takeaway for the learner: this repo's code maps mostly onto four teams (perception,
controls/autonomy, ML/data, simulation/tools) — but PRACTICE.md exists precisely because shipping a
robot requires understanding the other nine.

### 5.2 Product lifecycle — where each kind of project matters

```
 CONCEPT ──▶ PROTOTYPE ──▶ EVT ──▶ DVT ──▶ PVT ──▶ PRODUCTION ──▶ FLEET OPERATIONS
 (paper +     (one works,   (engineering  (design    (production   (hundreds/    (thousands in
  sim only)    barely)       validation:   validation: validation:  thousands     the field for
                             does the      is the      can the       built)        years)
                             design work?) design      factory
                                           right?)     build it?)
```

- **Concept/prototype:** simulation (10, 11) and rapid algorithm work (06, 08, 12) dominate — the
  GPU lets you evaluate designs before metal exists. Most of this repo lives here didactically.
- **EVT/DVT/PVT** (engineering/design/production validation — hardware-industry stage gates):
  calibration pipelines (01, 02), HIL rigs (11.11), fault-injection and validation farms (31.05,
  31.06), thermal/battery characterization (25). Algorithms freeze; evidence accumulates.
- **Production:** manufacturing test software, per-unit calibration, cost-down engineering (26, 27).
- **Fleet operations:** telemetry anomaly detection (31.07), map/data pipelines at fleet scale
  (02.15, 05.18), OTA and monitoring (32). This phase lasts 5–10× longer than all the others and
  usually decides profitability.

### 5.3 Build vs. buy — didactic criteria

Should a team hand-roll a kernel (as this repo does everywhere) or adopt nvblox / cuRobo / Isaac /
OpenCV-CUDA / a vendor SDK? Honest criteria, in the order real teams weigh them:

1. **Is it your differentiator?** If customers choose you *because* of your planner, own it. If
   mapping is table stakes for you, buy/adopt it (nvblox) and spend your engineers elsewhere.
2. **Does an adopted library actually fit?** Libraries encode assumptions (sensor models, memory
   budgets, robot morphology). Measure the fit *on your robot* before committing — the cost of
   fighting a framework often exceeds the cost of writing the 2,000 lines you actually need.
3. **Can you debug it at 3 a.m.?** A black box you cannot profile or fix is a liability in fleet
   operations. This is the strongest argument for this repo's "no black boxes" rule (CLAUDE.md §1):
   even when you buy, you must *understand* what you bought — enough to have built a toy version.
4. **License, support horizon, supply chain.** SDK license terms, vendor lock-in, and whether the
   dependency will exist in five years are engineering inputs, not legal trivia.

The repo's position: *build to learn, so you can buy with judgment.* Every project README §11 names
the production library it teaches toward.

### 5.4 Unit economics in one didactic paragraph

A robot's **BOM** (bill of materials — the sum of every part's cost) sets the floor on price;
hardware margins in robotics are thin (often 20–50% gross, versus ~80%+ for software), which is why
so many companies sell **RaaS** (robots-as-a-service: monthly fee, company keeps ownership and
maintenance) instead of **capital sale** (customer buys the robot outright). RaaS smooths customer
adoption and creates recurring revenue but concentrates maintenance/fleet cost on the maker — making
reliability engineering (domain 31) and cheap remote diagnostics (31.07, 32) directly
profit-relevant. Software's role in the economics: perception/autonomy quality drives the *labor
replaced or augmented* per robot, which is the number the customer actually pays for. Illustrative
orders of magnitude only; every real case differs.

---

## 6. Robot internals & the regulatory map

> **Label: orientation map, not compliance guidance.** The diagram shows what is physically inside a
> generic modern robot; the table lists which rulebooks *exist* per robot type. Neither is advice on
> how to certify anything (CLAUDE.md §1, §8). Every project `PRACTICE.md` §2–§4 cites this section.

### 6.1 Inside a real robot — generic hardware architecture

```
 ┌────────────────────────────────────────────────────────────────────────────────────┐
 │                              INSIDE A REAL ROBOT                                   │
 │                                                                                    │
 │  COMPUTE TIER                                  SENSOR SUITE                        │
 │  ┌─────────────────────────────┐               ┌────────────────────────────────┐  │
 │  │ GPU SoC (Jetson-class)      │◀── MIPI/USB/──│ cameras, LiDAR, radar, IMU,    │  │
 │  │  or x86 + discrete GPU      │    GigE ──────│ GNSS, F/T, tactile, encoders   │  │
 │  │  [01-12,23: perception/     │               │ [01,02,03,20 + 11 to simulate] │  │
 │  │   mapping/planning run here]│               └────────────────────────────────┘  │
 │  ├─────────────────────────────┤                                                   │
 │  │ real-time MCUs (Cortex-M/R) │   COMMS BUSES                                     │
 │  │  [08 control @ 1 kHz,       │   ┌────────────────────────────────────────────┐  │
 │  │   24 motor fw @ 10-20 kHz,  │◀─▶│ CAN-FD (motors, BMS, ~5 Mbps, robust)      │  │
 │  │   32 embedded infra]        │   │ EtherCAT (synced servo chains, sub-ms)     │  │
 │  ├─────────────────────────────┤   │ Ethernet/TSN (sensors, compute↔compute)    │  │
 │  │ SAFETY CONTROLLER           │   │            [32: drivers, timing, DMA]      │  │
 │  │  independent, dual-channel  │   └────────────────────────────────────────────┘  │
 │  │  [31: what it monitors]     │                                                   │
 │  └─────────────────────────────┘                                                   │
 │                                                                                    │
 │  ACTUATION CHAIN (one per joint/wheel/rotor)                     [24, 9, 26]       │
 │  ┌─────┐   ┌────────────┐   ┌─────────────┐   ┌───────┐   ┌─────────┐   ┌────────┐ │
 │  │ MCU │──▶│ gate driver│──▶│ power stage │──▶│ motor │──▶│ gearbox │──▶│ load   │ │
 │  │(PWM)│   │ (level-    │   │ (MOSFET/GaN │   │ (BLDC/│   │ (or     │   │ (joint,│ │
 │  └──▲──┘   │  shift +   │   │  half-      │   │ PMSM) │   │ direct  │   │ wheel, │ │
 │     │      │  protect)  │   │  bridges)   │   └───┬───┘   │ drive)  │   │ rotor) │ │
 │     │      └────────────┘   └─────────────┘       │       └─────────┘   └────────┘ │
 │     │                                         ┌───▼────┐                           │
 │     └──── current sense + ────────────────────│ encoder│  position/velocity        │
 │           encoder feedback (10-20 kHz loop)   └────────┘  feedback                  │
 │                                                                                    │
 │  POWER TREE                                                          [25]          │
 │  ┌─────────┐   ┌─────┐   ┌──────────────────────────────────────────┐              │
 │  │ battery │──▶│ BMS │──▶│ DC/DC rails: 48V bus (motors), 12V       │              │
 │  │ pack    │   │     │   │ (sensors/compute), 5V/3.3V (logic)       │              │
 │  └─────────┘   └─────┘   └──────────────────────────────────────────┘              │
 │                                                                                    │
 │  SAFETY CHAIN (hardwired, independent of the compute tier)          [31, 21]       │
 │  E-stop buttons ──▶ safety relay ──▶ motor power cutoff (STO)                      │
 │  watchdogs (MCU + host)  ·  redundant monitors  ·  safety-rated sensor zones       │
 └────────────────────────────────────────────────────────────────────────────────────┘
```

Reading notes for the learner:

- **The compute tier is layered by deadline** (§1.1): GPU SoC / x86+dGPU for the 10–60 Hz soft
  real-time layers, MCUs for the kHz hard real-time layers, and a *separate, simpler, certifiable*
  safety controller whose job is to distrust everything above it.
- **The actuation chain is where code meets physics**: a PWM pattern from an MCU becomes gate
  charge, becomes phase current, becomes torque through the motor constant, gets multiplied (and
  made backdrivable or not) by the gearbox, and comes back as encoder counts. Domain 24 projects
  simulate pieces of this chain; domain 9 models what it does to the robot's dynamics.
- **The safety chain is deliberately boring**: relays, dual channels, hardwired E-stops, STO
  (safe-torque-off) inputs on motor drives. No GPU code in this repo belongs on it — which is
  exactly why domain 31 projects say "sim-validated only, not safety-certified."

### 6.2 The regulatory landscape by robot type

Which rulebooks govern which machines — an orientation table only (standards evolve; numbers here
identify documents, not summarize them):

| Robot type | Main standards / regulatory path | Flavor of what they demand |
|------------|----------------------------------|-----------------------------|
| Industrial arms | **ISO 10218** (robot + integration safety), **ISO/TS 15066** (collaborative operation) | Risk assessment, guarding or power/force limits, speed-and-separation distances (see 21.04 — didactic metrics only) |
| Service / personal-care robots | **ISO 13482** | Hazard analysis for robots sharing space with untrained people |
| Autonomous vehicles | **ISO 26262** (functional safety of E/E systems), **UL 4600** (safety case for full autonomy), regional type-approval | Rigorous development process (ASIL levels), argued+evidenced safety case, scenario validation (31.05's world) |
| Medical robots | **IEC 60601** family (electrical safety/EMC), **FDA pathways** (510(k)/De Novo/PMA in the US), EU MDR | Clinical evidence, quality systems, post-market surveillance — repo §29 projects are educational only, no clinical claims |
| Drones / UAS | **FAA Part 107** (US commercial ops), **EASA** open/specific/certified categories (EU) | Operational limits, pilot/operator requirements, airworthiness at the certified end |
| Marine / AUV-ASV | **COLREGs** (collision rules at sea), class-society rules for larger vessels | Right-of-way behavior any autonomous surface vessel must encode |
| Space & defense-adjacent | **Export controls** (ITAR/EAR in the US and analogues elsewhere) | Restrictions on sharing hardware, software, and even technical data across borders — reaches software engineers directly |

Cross-cutting all rows: machinery directives (EU), EMC and radio certification for anything with
motors and radios, and workplace-safety law wherever robots share space with workers. If a project's
domain appears in this table, its `PRACTICE.md` §4 names the relevant row and repeats the label:
**orientation, not compliance guidance.**

---

## 7. What your README "System context" section must quote

Every project README contains a **System context — where this sits in a robot** section (CLAUDE.md
§4.1 item 4; checked by `tools/verify_project.py`). It must state, in this order, quoting this
document:

1. **Position in the stack** — which §1 box (or boundary) the project lives in, in one sentence.
2. **Upstream inputs and downstream consumers** — named as §3.6 message-shaped structs, e.g.
   "consumes `PointCloud` (10–20 Hz, `lidar` frame); produces `T_map_base` pose updates."
3. **Rate/latency budget** — the row(s) of the §1.1 table this project would face on a real robot,
   with any project-specific tightening ("must fit in the 10–50 Hz local-planner tick, ~20 ms").
4. **Reference robot(s)** — which of the five §2 machines use this block, and which §4 chain(s) the
   project belongs to, if any.
5. **What replaces/surrounds it in production** — one line naming the production-grade counterpart
   (also expanded in README §11 Prior art).
6. **Owning team** — one line placing the work in the §5.1 org map ("this lives with the
   controls & autonomy team, adjacent to embedded/firmware").

Link this document as `../../../docs/SYSTEM_DESIGN.md` (from a project folder) and your project's
`PRACTICE.md` for the physical/commercial grounding of items 5–6. If your project genuinely sits
outside the stack (e.g., a pure GPU-library foundation from domain 33), say so honestly and state
which layers *call* it instead.

*Every kernel has an address inside a machine. This document is the street map.*
