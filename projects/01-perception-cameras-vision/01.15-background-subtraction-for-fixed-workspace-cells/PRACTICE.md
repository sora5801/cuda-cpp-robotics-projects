# 01.15 — Background subtraction for fixed-workspace cells: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5-6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

This project's "hardware" is a **fixed camera installation**, not a moving mechanism — construction
here means MOUNTING, not manufacturing a part. Three physical requirements follow directly from
THEORY.md's "The problem" (every one of them exists specifically to keep the scene's statistics
close to this project's modeling assumptions):

- **Rigid mounting — vibration is the enemy.** THEORY.md's whole "why a fixed camera turns
  background into a statistical object" argument assumes pixel `(x, y)` maps to the SAME physical
  direction, frame after frame. A camera on a vibrating structural member (a press, a conveyor
  frame, anything near rotating machinery) violates that assumption at the sub-pixel level
  continuously — every edge in the scene "breathes" a fraction of a pixel, which a per-pixel model
  reads as noise it did not design for. Real installations use a stiff, independently-braced mount
  (not clamped to the same structure the equipment vibrates), isolated from the work cell's own
  mechanical noise floor where possible, and torque-checked/re-verified on a maintenance schedule —
  a loosened mount is a silent failure mode (the model slowly "learns" the new, wrong alignment via
  its own adaptation, exactly the mechanism THEORY.md's absorption story describes, now working
  against the installer instead of for the intrusion lesson).
- **Lighting stabilization.** THEORY.md names mains flicker, HVAC-driven dimming, and sunlight as
  illumination-drift sources; a real installation fights all three with FIXED-INTENSITY,
  DC-driven (not phase-cut-dimmed AC) machine-vision lighting, physically shrouded from ambient
  light changes (a hood or enclosure over the work cell, blackout of any window/dock-door
  sightline), and ideally on its own regulated power feed independent of the cell's motors (which
  can sag a shared supply's voltage — and hence a lamp's brightness — during a motion cycle).
- **Lens/aperture lockdown, and why "auto-anything is the enemy."** THEORY.md is explicit: this
  project assumes a LOCKED-exposure camera. In practice that means physically taping or Loctite-ing
  the aperture ring (if the lens has a manual iris) after final focus/exposure setup, disabling
  auto-exposure, auto-gain, auto-white-balance, and any "scene mode" in the camera's firmware, and
  documenting the locked exposure/gain/white-balance values in the cell's commissioning record so a
  technician who power-cycles the camera restores the SAME settings rather than whatever the
  firmware defaults to. A single accidental auto-exposure re-enable after a camera swap would
  silently break every gate this project's demo checks, in a way that looks like "the algorithm
  stopped working" rather than "the installation regressed."
- **What breaks in the field:** condensation or dust on the lens (reads as a slow, spatially
  uneven "illumination drift" no ramp constant models); a technician bumping the camera during
  unrelated maintenance (an instant full-frame re-registration, which every model in this project
  will initially flag as 100% foreground until re-converged); cable strain at the connector from
  repeated cell access (intermittent frame drops or corruption, not directly modeled here but a
  real operational hazard); LED fixture lumen depreciation over its service life (a genuine,
  slow, MONTHS-long drift this project's single-frame-timescale +15% ramp does not represent but
  which the same adaptive-vs-static lesson still applies to).

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-11. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

| Tier | Compute | Camera | Notes |
|------|---------|--------|-------|
| **Hobby/prototype** | Raspberry Pi 5 (CPU-only OpenCV `BackgroundSubtractorMOG2`, no GPU needed at this resolution) or a Jetson Orin Nano dev kit if GPU headroom is wanted for other perception tasks sharing the same board | A fixed-focus USB machine-vision camera (e.g. an industrial-grade UVC camera with exposure-lock support in its driver, ~$50-150) | This project's whole workload (128x96, 3 models, 160 frames) is trivially CPU-fast at real camera resolutions too — a GPU is not REQUIRED for background subtraction alone at this scale; it earns its keep when this perception task shares a board with something heavier (a detector, a SLAM front end). |
| **Research/pilot cell** | NVIDIA Jetson Orin NX/AGX — the class of module SYSTEM_DESIGN.md's "robot internals" diagram puts at the perception tier of a manipulator work cell or AMR | A GigE Vision or USB3 Vision industrial camera with a documented, lockable exposure/gain API (e.g. Basler ace, FLIR Blackfly S) and a fixed-focal-length lens with a manual (or locked auto) iris | GigE Vision's precise trigger/timestamp support matters more here than raw resolution — this project's models assume a clean, jitter-free frame CADENCE (THEORY.md's rate/latency budget), which a consumer webcam's variable-frame-time USB stack does not reliably give. |
| **Production line** | An industrial PC with a discrete RTX-class GPU (shared across multiple cameras/cells, running many instances of this kind of model) or a per-cell Jetson Orin, depending on the plant's compute-centralization strategy | A machine-vision camera on a permanent, vibration-isolated mount (see §1) with a lockable C-mount lens, IP-rated housing if the environment has coolant mist/dust, and a hardware trigger line synced to the cell's PLC cycle (so "frame N" has a known, auditable relationship to "cycle N") | At this tier, the camera's own manufacturer typically ships (or the integrator writes) the exposure/gain lockdown and drift-monitoring logic as a supported feature, not a bespoke script — this project's kernels teach the ALGORITHM a vendor SDK's "background learning" checkbox is actually running. |

Actuation-chain silicon (motor-control MCUs, gate drivers, encoder ICs) is **N/A** — this project has
no actuator anywhere in its own scope; it is a pure sensing/perception component that would sit
alongside, never inside, a cell's motion-control hardware.

## 3. Installation & integration — putting it on a real robot

- **Where this runs:** the perception-tier compute named in §2 — a Jetson module or an industrial
  PC's GPU, physically local to (or on the same LAN segment as) the fixed camera, not a cloud
  service (the 30 Hz budget in README "System context" has no round-trip-latency room for one).
- **OS / real-time constraints:** a standard Linux (Ubuntu/JetPack) userspace process is adequate —
  this workload has no hard real-time deadline in the RTOS sense; a missed or late frame simply
  means one delayed classification, not a control-loop fault. The 30 Hz SOFT budget (README) is
  generously met (this project's measured full-pipeline time is a small fraction of one frame
  period) with wide margin for scheduling jitter on a normal Linux kernel.
- **ROS 2 node shape:** a single node subscribing to the fixed camera's `sensor_msgs/msg/Image`
  topic (SYSTEM_DESIGN.md §3.6's `Image` message shape, matching this project's own upstream-input
  framing), maintaining the per-pixel model state as the node's own internal buffers (not published
  — it is large and high-rate), and publishing a small, LOW-rate output: a foreground mask
  (`sensor_msgs/msg/Image`, mono8) for visualization/debugging, and a lightweight EVENT message
  (custom, or `diagnostic_msgs/msg/DiagnosticStatus`-shaped) carrying "intrusion active: yes/no",
  "region ID", and a timestamp — the shape README "System context" describes as this project's
  downstream interface.
- **Bus / interface:** camera-to-compute is a vision interface (GigE Vision, USB3 Vision, or
  MIPI CSI for an embedded module) — not a control bus; this project never talks CAN-FD or EtherCAT
  directly. Its OUTPUT event, once past the ROS 2 layer, would typically reach a PLC or cell
  controller over whatever integration bus that controller already uses (often EtherCAT or a
  vendor-specific fieldbus in an industrial cell) — a translation this project's own scope stops
  short of.
- **Calibration / bring-up procedure:** (1) mount and lock the camera per §1; (2) capture and lock
  exposure/gain/white-balance with the cell in its NORMAL idle state (no product, no operator) —
  this becomes the effective "frame 0" every model in this project initializes from; (3) run a
  BURN-IN period (this project's own `SG_ALPHA`/`MOG_LR_*` time constants suggest tens of frames —
  seconds at 30 Hz — is enough for the adaptive models to reach steady state; a real installation
  would burn in for minutes to be conservative) before trusting any intrusion alarm; (4) re-run
  bring-up after ANY camera/lens/lighting change, and on a scheduled interval to catch slow drift
  the models themselves cannot distinguish from a legitimate change (see the recalibration triggers
  below).
- **Recalibration triggers:** seasonal daylight changes (if any stray natural light reaches the
  cell despite shrouding), fixture relamping or LED replacement (a step-change in brightness a
  slow-adapting model will treat as an event, not instantly), any physical change to the cell's
  fixed background (a new permanent fixture, a repainted surface), and camera/lens replacement
  after a fault — each of these should re-trigger the bring-up burn-in above, not be silently
  absorbed by the running models over the following minutes.
- **Region-of-interest (ROI) configuration discipline.** A real cell rarely wants EVERY pixel
  treated identically: this project's own designed scene distinguishes a genuinely bimodal region
  (the lamp) from ordinary background specifically so a learner sees why "treat the whole frame
  uniformly" already needs per-region tuning even in a toy scene. A production deployment goes
  further: explicit ROI masks marking zones as "always ignore" (a status indicator known to blink,
  exactly this project's lamp lesson generalized), "intrusion-critical" (weighted more heavily,
  faster alarm), or "informational only" (logged, not alarmed) — configuration data that belongs in
  the cell's commissioning record, version-controlled alongside the camera's locked exposure
  settings, not hand-tuned ad hoc after installation.
- **The safe hardware-testing ladder (CLAUDE.md §1 applies in full):** simulation (this project's
  own synthetic sequence, and any additional synthetic scenes a learner builds per README Exercise
  5) → hardware-in-the-loop with a real camera on a BENCH mock-up of the cell's geometry, no live
  equipment nearby → a tethered/observation-only install on the real cell with its alarm output
  wired to nothing but a log file (verify false-positive/false-negative rates on real footage before
  anything downstream reacts to this project's output) → only then, an alarm output wired to
  something consequential, with an independent E-stop and hardware watchout at every rung. This
  project's own code is **sim-validated only** (CLAUDE.md §1) — none of the numbers in
  `demo/expected_output.txt` constitute validation on real hardware, real lighting, or a real work
  cell.

## 4. Business & regulatory context

*Didactic orientation only — see the disclaimer at the foot of this file.*

**Who needs this, and in what products.** Fixed-camera workspace monitoring is a widely deployed
capability across manufacturing (cell intrusion/tamper alerts, unattended-machine monitoring),
warehousing/logistics (staging-area occupancy, dock-door activity, tote/kit-completeness triggers —
the pick-verify framing README "System context" names, feeding project **01.14**'s template
matching for the actual identity check), and general facility monitoring (equipment-area access
logging). It is a mature, commodity capability — the interesting engineering is almost always in
tuning it to a SPECIFIC scene (this project's whole point: naive vs. adaptive vs. multi-modal is a
real, recurring integration decision, not an academic exercise) rather than in inventing a new
algorithm.

**Commercial and open-source players.** OpenCV (BSD-licensed, the `BackgroundSubtractorMOG2`/`KNN`
this project's models simplify) underlies a huge fraction of both open-source and commercial
machine-vision tooling; camera/vision-system vendors (Cognex, Keyence, Basler, FLIR/Teledyne) ship
background-subtraction or "scene change" features as part of their integrated smart-camera or
vision-software product lines; video-analytics platforms (both on-prem industrial and cloud-based
security/retail-analytics products) build on the same classical or, increasingly, learned
(THEORY.md's "where this sits in the real world") change-detection core.

**What getting it wrong costs.** A background subtractor tuned too SENSITIVE (this project's
`illumination_drift` gate names the exact failure mode) generates alarm fatigue — operators learn
to ignore a system that cries wolf at every lighting change, which is itself a safety-adjacent cost
even for a purely informational monitor. Tuned too INSENSITIVE (this project's `bimodal_lesson`
gate names the complementary failure — a model that cannot represent a legitimate two-state pixel
either constantly false-alarms there or, if naively "fixed" by raising every threshold, misses real
events too), a genuine intrusion or process fault goes unnoticed — the cost ranges from a missed
pick-verification catch (a defective shipment) to, in the worst case people conflate this
technology with, a missed genuine safety event (which is exactly why §3's ladder and the callout
below exist).

**Regulatory path — and the one line that matters most.** Workspace-monitoring-for-INFORMATION
(occupancy, tamper alerts, pick-verify triggers) has no dedicated functional-safety certification
requirement — it is engineering practice, not a regulated safety function. Workspace-monitoring-
FOR-HUMAN-SAFETY is an entirely different regulatory category, and this project's output must
never be mistaken for it: project **21.04** (speed-and-separation monitoring) is this repository's
dedicated treatment of that boundary, and this project inherits its caveat verbatim — a certified
protective function watching for human intrusion is evaluated against ISO/TS 15066 (collaborative
robot applications) and built from hardware certified to a defined Performance Level under
ISO 13849 (safety-rated laser scanners, light curtains, pressure-sensitive floor mats — devices
engineered from the ground up for redundant, fail-safe detection, not a single RGB camera and a
software heuristic). See `docs/SYSTEM_DESIGN.md` item 6's regulatory map for the fuller orientation.

**Where this work lives in a robotics company.** Perception engineering (the camera pipeline and
model itself) and cell-controls/manufacturing engineering (the installation, ROI configuration, and
integration into the cell's supervisory logic) jointly own this capability — continuing projects
01.13/01.14's framing for this repository's industrial-machine-vision cluster (SYSTEM_DESIGN.md item
5's org map). Adjacent teams: functional-safety engineering (who owns anything that DOES need
ISO 13849/ISO/TS 15066 certification, and who must be the ones to say "no" if anyone proposes using
this project's output that way) and fleet/plant operations (who own the ongoing recalibration and
false-alarm-rate monitoring §3 describes as a running operational cost, not a one-time install
task).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never
fabricated.*
