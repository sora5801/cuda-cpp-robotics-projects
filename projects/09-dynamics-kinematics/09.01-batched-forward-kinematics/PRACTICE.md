# 09.01 — Batched forward kinematics (10⁵ configurations — the foundation for everything above): Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> This project is software, but unlike a generic math library it models a *specific physical
> object* — a robot arm — so sections 1–2 teach the arm whose numbers FK consumes.

## 1. Building it — construction of the robot/part

The subsystem FK models is the **serial arm itself** — and its construction is exactly where the
model's numbers come from:

- **Links** are castings (aluminum, industrial arms), machined billet, or carbon tube (lightweight
  cobots). The CAD geometry defines the *nominal* `t_j` translations in the model; casting +
  machining tolerances (±0.02–0.1 mm per feature) and assembly stack-up are why nominal FK is only
  ~0.5–1 mm accurate at the tool, while the *repeatability* of a good arm is ~0.02–0.1 mm — the gap
  between the two is closed by **kinematic calibration** (measure poses with a laser tracker or CMM,
  fit corrected parameters; the corrected numbers replace this project's model rows one-for-one).
- **Joints** are the construction crux: a motor (frameless torque motor or servo + gearbox), a
  **gearbox** (harmonic drive for wrists — zero backlash, some compliance; planetary or cycloidal
  for high-torque axes), crossed-roller bearings carrying the structural loads, an **encoder**
  (see §2), brakes on gravity-loaded axes, and seals. What breaks in the field: gearbox wear
  (harmonic-drive flexspline fatigue), encoder contamination, cable-harness fatigue at moving
  joints (the harness through a hollow wrist is a wear part), and crash-induced geometry shifts —
  the last one silently invalidates calibration, which is why cells re-verify after collisions.
- **Assembly** sets the axes: joint axis vectors in the model are nominal `Z`/`Y`/`X` directions;
  real axes miss by arc-minutes (bearing seat tolerances), another calibration-absorbed effect.

Wiring/mounting reality for the compute that runs FK itself: covered in 33.01's PRACTICE §1 (same
carrier hardware; not repeated here — deliberate cross-reference, not padding).

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-08. All parts named below are **illustrative examples, never endorsements**;
part numbers and prices go stale — verify current before relying on any of them.*

The hardware FK *talks to* is the sensing chain that produces `q`, plus the compute that runs the
batches:

| Piece | Illustrative choices (2026) | Why it matters to FK |
|---|---|---|
| Joint encoders | Absolute magnetic/capacitive encoder ICs and kit encoders of the 17–23-bit class on the joint output; incremental + motor-side absolute as the budget option | Sets the quantization of FK's *input*: 19 bits ≈ 1.2e-5 rad ≈ 12 µm at a 1 m reach — conveniently just below our 1e-4 comparison tolerance |
| Encoder/motor comms | EtherCAT or CAN-FD to the joint drives; 1–8 kHz servo cycle | Determines how fresh the `q` snapshot is; FK on stale angles is the classic "why is the pose lagging" bug |
| Compute — hobby | Jetson Orin Nano class (~US$250) | Runs this demo's 200k-config batch with ease; unified memory skips PCIe copies |
| Compute — research | x86 + RTX-class dGPU (the reference machine here: RTX 2080 SUPER) or AGX Orin class (~US$1–3k) | The regime the measured numbers come from |
| Compute — industrial | Control-cabinet IPC + embedded GPU, or the arm vendor's controller + a vision/planning PC beside it | In work cells, FK-heavy planning usually runs *beside* the vendor controller, which keeps its own certified kinematics |
| The arm archetype | Educational 6-DoF class (few-hundred-US$ hobby arms) → cobot class (~US$20–50k) → industrial 6-axis (~US$25–100k+) | The same 10-numbers-per-joint model describes all of them; only the tolerances and certifications change |

## 3. Installation & integration — putting it on a real robot

- **Where it runs:** as a library inside planning/control processes (see 33.01 PRACTICE §3 for the
  library-shipping story — identical here). In a ROS 2 cell: a `MoveIt`-adjacent planning node or a
  custom IK/reachability service subscribes to `sensor_msgs/JointState` (the measured `q`), loads
  the robot description from the `robot_description` parameter (URDF), and calls batched FK
  internally; outputs flow as `geometry_msgs/PoseArray` or stay in-process as planner cost terms.
- **The model pipeline is the real integration work:** URDF → this project's 10-float rows is a
  mechanical translation (Exercise 1 territory), but *which* URDF matters — nominal CAD numbers or
  the calibrated per-robot parameters. Fielded cells version the calibrated model per serial
  number and re-calibrate after crashes or gearbox swaps; loading the wrong robot's calibration is
  a real (and embarrassing) field bug. This project's `set_robot_model()`-once pattern mirrors that
  load-at-startup discipline.
- **Real-time constraints:** FK for *planning* runs at planner rates (10–50 Hz) on the GPU tier —
  uncritical. FK for *servoing* (the 1 kHz loop's own pose feedback) stays on the CPU/MCU tier
  today (GPU submission jitter; see 33.01 PRACTICE §3 and projects 32.02/32.03). The division of
  labor — batched hypothesis FK on GPU, single measured-state FK on CPU — is the standard shape.
- **Calibration & bring-up:** encoder zeroing (drive each joint to its index/home), then kinematic
  calibration as in §1. The safe-testing ladder applies to the *consumers* that move the arm; FK
  itself computes on hypothetical angles and moves nothing — but its correctness gates everything
  above it, which is why bring-up procedures verify FK against a measured tool position before any
  autonomous motion (touch-point tests: jog to a fixture, compare).
- **N/A here:** no bus is commanded and no E-stop is wired *by this code* — FK is upstream of the
  motion stack that owns those (stated per contract, not padded).

## 4. Business & regulatory context

- **Who needs it:** every manipulation company (industrial, cobot, humanoid, surgical), plus
  simulation and offline-programming vendors. Batched FK specifically is the price of admission to
  modern sampling-based planning/IK — the difference between evaluating 100 IK seeds and 100,000
  per tick is a product capability (cycle-time optimization, cluttered-scene grasping).
- **The players:** arm vendors ship certified controllers with their own kinematics (their
  calibration is part of the product); NVIDIA's cuRobo/Isaac push the batched-GPU planning story;
  Pinocchio/MoveIt anchor open source; offline-programming vendors live entirely on accurate FK +
  calibration.
- **Cost of getting it wrong:** a kinematics bug is a *systematic* pose error — scrapped parts,
  crashed tooling, line downtime; in surgical/precision contexts, recalls. The quiet failure mode
  is stale/wrong calibration rather than wrong math, which is why traceability of model parameters
  (who calibrated which serial number when) is a real quality-system artifact.
- **Regulatory:** the arm's safety functions (speed/force limits for cobots under ISO 10218 /
  ISO/TS 15066) live in the certified controller, not in planning FK; planning-side code like this
  is uncertified supporting software (the SOUP framing — see 33.01 PRACTICE §4 and the
  SYSTEM_DESIGN item-6 orientation map). Calibration and accuracy claims, however, feed
  *performance* specs (ISO 9283 pose-accuracy test methods) that customers contractually rely on.
- **Owning team:** motion planning / controls (titles: robotics software engineer — motion
  planning; kinematics & calibration engineer), adjacent to mechanical (produces the geometry),
  manufacturing/quality (owns calibration stations), and the safety team (owns what may move).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
