# 20.01 — GelSight/DIGIT processing: contact patch, shear field via optical flow, slip detection in real time: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

A GelSight/DIGIT-style sensor is a small, purpose-built optical assembly, usually sized to fit a
fingertip (DIGIT's housing is roughly 20 x 27 x 18 mm):

- **The gel membrane.** A layer of clear, soft **silicone elastomer** (a few millimeters thick),
  typically cast in a mold against the desired outer contact shape (flat, domed, or fingertip-curved),
  then coated on its OUTER face with a thin **reflective or opaque skin** (production sensors use a
  specular aluminum-flake or paint coating so the camera sees the gel's deformed geometry via
  reflected light, not whatever the touched object looks like) and printed on its INNER face with the
  **marker dot grid** (this project's `kMarkerSpacingPx`/`kMarkerDarkGray` — silk-screened or
  inkjet-printed ink dots on production sensors). The gel is bonded to a rigid, optically clear acrylic
  or glass backing plate that the camera looks through.
- **Illumination.** Several LEDs (production sensors commonly use 3+ different colors, one per
  direction) are mounted around the gel's edge, INSIDE the housing, angled to graze the gel's inner
  surface — the directional-color scheme is what lets production photometric-stereo reconstruct actual
  depth (THEORY.md "Where this sits in the real world"); this project's single-scalar shading model
  needs only one effective illumination direction.
- **The camera.** A small board-level camera module (a wide-angle lens is common, since the sensing
  area sits very close to the lens) faces the gel from inside the rigid housing, framed so its field of
  view exactly covers the gel's contact area.
- **Mounting, wiring, sealing.** The whole assembly is potted/sealed into a fingertip housing that
  bolts or clips onto a gripper jaw; the camera's cable (USB or a flex-PCB MIPI ribbon) routes through
  or alongside the gripper's other wiring to the hand/wrist, sharing the same strain-relief and
  connector concerns as any other wrist-mounted sensor cable that must survive thousands of open/close
  cycles.
- **What breaks in the field, and why it matters for THIS project's assumptions.** The gel **wears**:
  repeated contact with sharp or abrasive objects scratches or tears the reflective coating and
  eventually the gel itself, degrading both the marker pattern's visibility (the shear-field kernel's
  whole input) and the shading response (the contact-mask kernel's threshold assumption) — production
  GelSight/DIGIT gels are a **field-replaceable consumable**, not a permanent part. This project's
  fixed-baseline-frame design (README "Data": one no-contact reference frame is captured once and
  reused for the whole run) is a direct simplification of a real bring-up step: a real sensor needs a
  **fresh baseline capture after every gel replacement**, since a new gel's exact texture and coating
  differ from the old one's — stated here honestly as a limitation of this teaching demo's single-shot
  calibration, not something the pipeline itself would need to change to fix.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-10. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

| Piece | Illustrative choices (2026) | Role in this project's pipeline |
|---|---|---|
| Gel-facing camera | Small board camera module, global or rolling shutter, USB-UVC or MIPI-CSI, e.g. DIGIT-class ~640x480 sensors, or research rigs using off-the-shelf USB endoscope-style modules | Produces the `Image` (320x240 in this project's synthetic scale) the pipeline consumes every frame |
| Illumination | 3+ color LEDs (red/green/blue or similar), driven by a small constant-current LED driver IC | Produces the shading signal `shading_darkening()` models as a scalar; production reconstructs full depth from it |
| Local compute (in-hand or wrist) | Research: none — camera streams raw over USB to a host PC/laptop; compact fielded designs: a small MCU or SoC (Raspberry Pi-class / Jetson Nano-class) local to the hand for pre-processing before uplink | Where the FIVE kernels in this project would actually execute on a real robot (see §3) |
| Host compute (where this project's kernels realistically run) | x86 + discrete GPU (this project's reference machine: RTX 2080 SUPER) or Jetson Orin-class embedded GPU SoC | Runs the CUDA pipeline; at 320x240 x 221 markers, even a small embedded GPU has enormous headroom versus this project's measured ~0.2 ms/frame |
| Cabling / connector | USB 2.0/3.0 micro or flex-PCB ribbon through the wrist, small-gauge LED power leads | Camera + illumination power/data, routed alongside the gripper's other wiring |
| Gel material | RTV silicone (e.g., Smooth-On-class casting silicones used by research groups), reflective coating (aluminum-flake paint or vapor-deposited layer) | The membrane itself; consumable, field-replaceable (§1) |

**Sensor-family cost/complexity comparison** (all tiers illustrative, dated 2026-07-10):

| Sensor family | Transduction | Spatial resolution | Rough cost tier | This project's relationship to it |
|---|---|---|---|---|
| GelSight Inc. sensors | Vision + photometric-stereo gel | Very high (camera-pixel-scale) | Research/industrial, $$$-$$$$ | This project's namesake; full photometric depth is out of scope (THEORY.md) |
| DIGIT (Meta/GelSight/Wonik) | Vision + marker tracking, same gel principle, open hardware | High | Low-cost research, $ | Closest in scope to this project's marker-tracking shear-field kernel |
| TacTip (Bristol) | Vision + pin-tip marker tracking on a soft dome | Medium (pin-count limited) | Research, $-$$ | Same marker-tracking ALGORITHM family, different mechanical transducer (THEORY.md) |
| uSkin (XELA) | Magnetic (3-axis Hall-effect taxels) | Lower (taxel-count limited) | Research/industrial, $$-$$$ | No camera, no image pipeline — the contrasting non-vision approach |

## 3. Installation & integration — putting it on a real robot

- **Where this code would physically run.** The five GPU kernels in this project are lightweight enough
  (measured ~0.2 ms/frame at 320x240, 221 markers) to run on the SAME compute that already hosts the
  arm's other perception/planning code (SYSTEM_DESIGN.md §2.2's manipulator work cell: "industrial PC +
  discrete GPU beside the vendor's arm controller") — a dedicated tactile-only compute tier is not
  required at this scale, though a very compact hand might still pre-process locally (a Jetson-class
  SoC in the wrist) purely for cabling/bandwidth reasons, then send only the compact `slip_score`/
  contact-patch summary upstream rather than raw video.
- **OS / real-time constraints.** Ordinary Linux (no hard real-time OS needed) is sufficient for the
  30-60 Hz camera-rate PROCESSING this project implements; the tight constraint (README "System
  context": a ~10 ms slip-to-force-adjustment budget) lives in the DOWNSTREAM grasp-force controller's
  loop, not in this pipeline's own tick — this project's measured 0.2 ms/frame leaves that downstream
  budget almost entirely intact.
- **ROS 2 node/topic shape.** This pipeline would run as a perception node: subscribes an `Image`
  (SYSTEM_DESIGN.md §3.6) from the gel camera driver (a standard `usb_cam`/`v4l2`-class node, or a
  vendor DIGIT/GelSight ROS 2 driver), publishes a small custom message carrying contact-patch area/
  centroid, the shear-field marker displacements (or just their summary), and the slip score/flag — the
  natural consumer being a grasp-force controller node (this project's README names
  [`19.01`](../../19-manipulation-grasping/19.01-parallel-grasp-candidate-scoring/README.md)'s executor
  and [`20.04`](../20.04-learned-slip-prediction-fused-into-the-grasp/README.md) as the downstream
  slot).
- **Bus / interface.** The camera itself is USB (UVC) or MIPI-CSI directly into the host's camera
  interface — NOT a CAN-FD/EtherCAT fieldbus (those carry the gripper's own motor commands/feedback,
  a separate, lower-rate channel this pipeline's OUTPUT would feed INTO, not travel over itself).
- **Calibration / bring-up procedure.** (1) Mount the sensor with no object present; (2) capture the
  no-contact baseline frame this project's pipeline subtracts against every subsequent frame (exactly
  `main.cu`'s `h_baseline`, captured once at startup) — and REPEAT this step after every gel
  replacement (§1); (3) verify the marker grid is visible and roughly at its expected rest-lattice
  positions (a real system's ONE-TIME blob-detection calibration pass — THEORY.md "Where this sits in
  the real world" — that this project's synthetic ground truth sidesteps); (4) only then trust the
  contact/shear/slip outputs.
- **The safe hardware-testing ladder (CLAUDE.md §1 applies at full strength once this feeds a force
  command):** (1) *Simulation* — this project's synthetic scenario, exactly as shipped. (2) *HIL* — the
  same pipeline against a real captured tactile video with independently-verified ground truth (a
  logged press/shear/slip sequence, hand-labeled or instrumented against a force/torque reference).
  (3) *Bench, tethered, current-limited* — the physical sensor mounted on a fixed jig, gripper motor
  current capped, verifying the slip signal against manually-induced slip before any closed-loop
  authority. (4) *Free running* — only once the downstream grasp controller has its own independent
  force/current limits and an E-stop path that does not depend on this pipeline's output being correct.
- **N/A here:** no fieldbus is implemented in this project (the demo's "sensor" is the in-code
  renderer); a real deployment's camera driver and the downstream grasp-controller message bus are both
  outside this project's scope, stated per contract.

## 4. Business & regulatory context

- **Who needs this capability.** Any manipulation task where a fixed grip force is either wasteful
  (crushing delicate objects) or insufficient (heavy/slippery objects): warehouse/logistics
  picking, food handling, electronics assembly, and increasingly humanoid/dexterous-hand research —
  anywhere a gripper must adapt force to an object it cannot fully characterize in advance. It is the
  manipulation-layer capability that turns "the arm grasped SOMETHING" into "the arm is holding it
  correctly and will notice before it drops."
- **The players.** GelSight Inc. (commercial vision-based tactile sensors, the sensor family this
  project is named after), Meta/FAIR's open-hardware DIGIT (co-developed with GelSight Inc. and Wonik
  Robotics — the low-cost research-fleet enabler), Bristol Robotics Laboratory's TacTip (academic,
  biomimetic), and XELA Robotics' uSkin (magnetic, non-vision) represent the sensor-hardware side;
  every humanoid/dexterous-manipulation company building in-house tactile fingertips (an increasingly
  common build decision as hands get more dexterous) is a downstream integrator and often a
  algorithm-side competitor to this project's pipeline. Build-vs-buy: the SENSOR hardware is usually
  bought (or licensed) from one of the named families; the PROCESSING pipeline (this project's
  territory) is commonly built in-house once a company's grasp strategy depends on tactile feedback,
  because the exact slip-detection tuning is close to the product's differentiator (SYSTEM_DESIGN.md
  §5.3's build-vs-buy criteria).
- **Cost of getting it wrong.** A missed or late slip detection drops the object — at best a re-pick
  cycle-time cost, at worst (heavy, fragile, or hazardous payloads) a safety or product-damage
  incident. A FALSE slip detection (over-tightening grip unnecessarily) can crush delicate items or
  waste actuator torque/thermal budget on every cycle — the CONTACT/SHEAR/SLIP gates this project
  checks against real physics exist precisely because both failure directions are costly, not just one.
- **Regulatory.** This project's output is a signal into a manipulator's force-control loop — the same
  regulatory landscape as any industrial-arm gripper action: **ISO 10218** (robot + system integration
  safety) and **ISO/TS 15066** (collaborative-operation force/speed limits) for industrial arms sharing
  space with people (SYSTEM_DESIGN.md §6.2's regulatory table). Nothing in this project is a certified
  implementation of any force/pressure limit those standards define — it is a didactic measurement
  pipeline, and any real deployment's safety case would need independent, certified force/current
  limits at the drive level that do NOT trust this software (the same architectural point 08.01's
  PRACTICE.md makes for its own force-command output).
- **Owning team.** Manipulation / tactile sensing, inside controls & autonomy (SYSTEM_DESIGN.md §5.1);
  titles commonly seen: manipulation engineer, robotics perception engineer (tactile). Adjacent teams:
  mechanical engineering (owns the gel/housing construction, §1), embedded/firmware (owns any in-hand
  pre-processing compute, §2-§3), and functional safety (owns the force/current limiting envelope
  around whatever grasp-force decision this pipeline's slip signal feeds, §4 above).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
