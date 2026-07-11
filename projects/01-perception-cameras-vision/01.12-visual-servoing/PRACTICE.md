# 01.12 — Visual servoing: image-Jacobian control loop entirely on GPU: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

**The physical carrier this code would serve: an eye-in-hand camera rig on a robot arm's wrist.**
IBVS's defining hardware choice is *where the camera lives* — this project assumes **eye-in-hand**
(camera rigidly mounted on the end effector, moving with it), the more common of the two IBVS
mounting conventions and the one that makes the interaction matrix in `THEORY.md` directly the
camera's *own* twist (the alternative, **eye-to-hand**, mounts the camera in the workspace watching
the end effector — the same math applies, but the twist being controlled is the *target's apparent*
motion relative to a stationary camera, and the sign conventions and Jacobian composition differ; not
implemented here, named for completeness).

- **Mounting.** The camera bracket bolts to the wrist flange (or just proximal to the gripper), sized
  to keep the target inside the field of view across the full servo range this project's nominal
  cohort exercises (±0.15 m lateral, up to 15° of rotation error — README "Limitations" states these
  are the TUNED ranges this specific controller was measured against, not a universal IBVS claim). A
  stiff, vibration-damped bracket matters more here than in most perception mounting: any bracket
  flex during a fast servo move directly aliases into feature-position noise, feeding straight into
  the interaction-matrix rows this project derives — a wobbly mount does not just blur an image, it
  injects spurious velocity into the control law every single step.
- **Hand-eye calibration — the `AX = XB` problem.** Before ANY of this project's math is meaningful on
  real hardware, the FIXED transform between the camera frame and the end-effector (or wrist joint)
  frame must be known — the classical **hand-eye calibration** problem, formulated as `AX = XB` (A:
  consecutive robot poses from forward kinematics; B: the corresponding consecutive camera poses from
  a calibration target; X: the unknown camera-to-flange transform), solved by moving the arm through a
  calibration sequence and least-squares-fitting `X` (Tsai-Lenz, Park-Martin, and dual-quaternion
  methods are the standard solvers). A wrong `X` does not blow up this project's *feature-space*
  control law directly (IBVS's whole appeal is exactly this robustness — see `THEORY.md`), but it DOES
  corrupt the mapping from the computed camera twist to the ARM's joint commands (README "System
  context": the twist hand-off to project 09.02's robot Jacobian) — a poorly calibrated `X` shows up
  as the end effector moving on a curved, not straight, path toward the correct final pose.
- **Lighting for target visibility.** A fiducial-based system (this project's assumed upstream, 01.06)
  needs consistent, glare-free illumination on the target across the whole approach: diffuse ring
  lighting co-mounted with the camera is the common industrial answer (keeps illumination angle fixed
  relative to the target regardless of arm pose, unlike ambient room lighting); a matte (non-
  retroreflective) fiducial print avoids the corner-blooming that degrades sub-pixel corner
  localization precisely when the target is close, i.e. right when this project's convergence
  threshold (`kConvergeEps`) is being approached and precision matters most.
- **What breaks in the field.** Bracket loosening (vibration, thermal cycling) silently degrades hand-
  eye calibration over weeks — a periodic recalibration check (or an in-line residual-error monitor)
  is standard practice; a scuffed/dirty fiducial degrades corner localization noise, which this
  project's damping term (`kDampingMu`, `THEORY.md` numerics) is the honest simulation-side analog of
  guarding against, though it is tuned for numerical robustness here, not measured against real
  detector noise statistics.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-11. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

| Tier | Camera | Compute | Illustrative cost (verify current) |
|------|--------|---------|--------------------------------------|
| Hobby/research | USB global-shutter camera (e.g. a Basler dart or an OpenMV-class board), fixed-focus lens sized for the working distance | A desktop PC with any CUDA GPU (this project's demo runs on any sm_75+ card, e.g. an RTX 2080-class); no real-time OS needed at IBVS's 30–60 Hz camera rate | Camera $150–500; compute is whatever workstation is on hand |
| Research/pilot-line | Industrial GigE/USB3 global-shutter camera with a locked-down lens (fixed focus, fixed aperture — auto-anything defeats a calibrated projection model), IP-rated housing near a wet/dusty cell | Small industrial PC or Jetson Orin-class module bolted near the arm base, running the feature detector (01.06) and this project's control loop | Camera $500–2,000; Jetson Orin-class module $500–2,000 |
| Industrial/production | Vendor-integrated smart camera (onboard fiducial decoding, e.g. Cognex/Keyence-class) or a machine-vision camera on a certified industrial PC, ring-light illuminator, connectorized/sealed cabling matching the cell's IP rating | Industrial PC co-located with (or replacing) the arm vendor's own vision add-on; real-time constraints usually met by keeping the FAST inner joint loop (0.5–1 kHz, SYSTEM_DESIGN.md §1.1) on the vendor's own controller and running only the 30–60 Hz IBVS twist computation on the external PC | Full vision-guided-robot package: often $5,000–$20,000+ including integration |

**Compute tier for THIS project's actual computation**: modest. The IBVS control law itself (one
`6×6` solve, `THEORY.md` "The algorithm") is trivial even on an MCU; this project's GPU angle is the
*offline/design-time* convergence-basin STUDY (4096 loops in parallel), not an online hardware
requirement — a real deployed IBVS loop runs the SAME control law on ordinary embedded compute at
camera rate, no GPU required at all. The GPU is a *development and validation* tool here (tune `λ`,
`μ`, verify the basin, before ever touching hardware), not a runtime dependency — worth stating
explicitly since it is easy to over-read "entirely on GPU" (the catalog bullet) as an online-compute
claim.

## 3. Installation & integration — putting it on a real robot

- **Where this code would run.** The 30–60 Hz IBVS loop (feature detection + this project's control
  law) runs on the arm-side industrial PC or companion computer — NOT on the arm vendor's own
  real-time joint controller, which instead consumes the resulting Cartesian/joint-velocity command at
  its own 0.5–1 kHz rate and interpolates (README "System context" states this multirate split
  explicitly). No hard real-time OS is required for the 30–60 Hz layer; the arm vendor's controller
  handles the hard real-time layer as it already does for any Cartesian-velocity teleoperation input.
- **ROS 2 shape.** The direct ROS 2 analog is **`moveit_servo`** — a real package that consumes a
  `geometry_msgs/msg/TwistStamped` (this project's `v_c`, SYSTEM_DESIGN.md §3.6's `Twist` struct) and
  streams it through the robot's Jacobian into joint velocity commands, with built-in singularity and
  joint-limit avoidance this project's teaching-scope damping (`THEORY.md` numerics) only partially
  covers. This project's feature-error computation (`ibvs_compute_step`) would sit UPSTREAM of
  `moveit_servo`, publishing the twist it currently returns from `main.cu`'s device code as a ROS 2
  topic instead.
- **Bus/interface.** Camera: USB3/GigE Vision (industrial) or MIPI-CSI (embedded/Jetson-integrated).
  Twist-to-joint-velocity command: whatever the arm vendor exposes — often EtherCAT (synced servo
  chains, SYSTEM_DESIGN.md §6.1) for industrial arms, or a vendor Ethernet/ROS 2 driver for
  collaborative arms.
- **Calibration & bring-up.** (1) intrinsic camera calibration (standard checkerboard/ChArUco — repo
  domain 01.16); (2) hand-eye calibration (`AX=XB`, §1 above); (3) verify the target's known geometry
  matches `kTargetHalfSize` (or, on real hardware, whatever the actual fiducial's physical size is —
  this is a common, embarrassing real-world bug: a wrong physical marker size scales EVERY depth
  estimate uniformly wrong); (4) run this project's demo-equivalent OFFLINE first (log real camera
  poses through the actual detector, replay through the SAME control law in simulation) before ever
  commanding the physical arm.
- **The safe hardware-testing ladder (CLAUDE.md §1, §8 — mandatory for ANY project whose output
  commands real motion):**
  1. **Simulation** — exactly this project: closed-loop IBVS entirely in software, batch-verified
     across a convergence basin, before any hardware exists in the loop.
  2. **HIL (hardware-in-the-loop)** — the real camera and detector feeding real image data of a
     STATIONARY or manually-posed target into the SAME control law, with the computed twist logged but
     NOT sent to the arm — verifies the perception-to-twist path end to end without risking motion.
  3. **Bench jig / tethered / current-limited** — the arm executing the computed twist at reduced
     speed/torque limits, on a bench or in a fixture with generous clearance, physical E-stop within
     reach, and a human hand on the enable switch the entire time.
  4. **Free running** — full-speed, full-workspace operation, only after (1)-(3) pass repeatedly and a
     documented risk assessment (ISO 10218/TS 15066 orientation, §4 below) is in place.
  Workspace limits (software AND hardware end stops) and an E-stop that cuts motor power independent
  of the compute stack (SYSTEM_DESIGN.md §6.1's hardwired safety chain — "no GPU code in this repo
  belongs on it") apply at every rung from (3) onward.

## 4. Business & regulatory context

**Who needs this.** Precision assembly and insertion tasks where camera-relative correction beats an
open-loop planned approach: connector/PCB insertion, peg-in-hole assembly, small-parts kitting with
loose part tolerances, and (the extension named in README's System context) vision-guided drone
landing on a marked pad. This is a narrower, higher-precision niche than general pick-and-place
(usually served by PBVS or learned grasping, per `THEORY.md`'s Prior art) — the market is
electronics/precision manufacturing integrators and robot-arm OEMs' vision add-on products, plus the
research/hobby robotics community building on open platforms.

**Commercial and open-source players.** ViSP (Inria, open-source, the direct academic/production
reference this project's `THEORY.md` cites); `moveit_servo` (open-source, ROS 2, the integration point
named in §3); commercial machine-vision vendors (Cognex, Keyence, and others) selling smart cameras
with onboard fiducial decoding and, increasingly, onboard visual-servo primitives; robot-arm OEMs
(Universal Robots, Fanuc, ABB, and others) offering vision-guided-motion options as part of their own
controller software.

**What getting it wrong costs.** In a collaborative or shared-space cell, a mistuned servo gain
(`THEORY.md`'s `λ`) or an unhandled retreat-pathology-style instability (`THEORY.md` "The problem")
is a **collision risk with people or fixtures**, not just a failed assembly — exactly why the safe
hardware-testing ladder in §3 is mandatory, not optional, before any real deployment. In pure
production-line contexts, an under-converged servo (this project's `convergence_basin` gate made
concrete) costs cycle time and yield (rejected/misassembled parts) rather than safety, but is still
measured in real money at production volumes.

**Regulatory orientation** (SYSTEM_DESIGN.md §6.2 — didactic orientation, **not** compliance advice):
a robot arm running an IBVS loop in a shared or collaborative workspace falls under **ISO 10218**
(industrial robot + system integration safety) and, if operating in close proximity to people without
full guarding, **ISO/TS 15066** (collaborative operation — the speed-and-separation and power/force
limiting requirements that repo project 21.04 computes didactically). None of this project's code is
a certified implementation of either standard — CLAUDE.md §1's caveat applies in full: sim-validated
only, no safety certification claimed or implied.

**Where this work lives inside a robotics company** (SYSTEM_DESIGN.md §5.1): **controls &
manipulation**, the same team that would own repo domains 06 (motion planning), 08 (control), and 09
(kinematics/dynamics) — adjacent to **perception** (who would own the upstream fiducial detector, 01.x)
and, for a shipped product, **QA & functional safety** (who would own the ISO 10218/TS 15066 risk
assessment and evidence trail before this code ever drives a real arm).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
