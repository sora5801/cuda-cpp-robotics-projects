# 01.09 — Photometric/vignetting calibration kernels: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5-6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

This project's physical carrier is TWO things: the **camera module** being calibrated, and the
**calibration rig** that captures the dark/flat stacks this code processes.

- **The camera module.** Same construction as 01.08's camera module (lens/sensor/housing stack, epoxy- or
  set-screw-locked focus, IP-rated sealing for an outdoor unit — see that project's PRACTICE.md §1 for the
  full mechanical treatment). Two details matter SPECIFICALLY for this project: (a) **lens-to-sensor
  decentering** — the small (sub-millimeter) lateral misalignment between the lens's optical axis and the
  sensor die's geometric center that this project's synthetic `CENTER_OFFSET_X/Y_PX` models is a genuine
  ASSEMBLY-tolerance effect, not an invented complication — every real lens mount has some decentering
  budget (typically tens of microns to a few hundred microns, translating to a handful of pixels at
  typical focal lengths); (b) **sensor die placement and wire-bond/flip-chip tolerance** contribute
  directly to PRNU/DSNU (a die placed slightly off-center or with local doping variation from wafer edge
  effects reads as exactly the per-pixel gain/offset variation this project's dark/flat stacks recover).
- **The calibration rig.** An INTEGRATING SPHERE (a hollow sphere with a diffusely-reflective interior
  coating, an entrance port for a light source, and an exit port the camera looks into) or, more cheaply, a
  large diffuser plate (opal glass or PTFE) evenly backlit by several LED panels, is the standard way to
  produce the near-perfectly-UNIFORM illumination a flat-frame capture needs — any residual non-uniformity
  in the illumination itself directly contaminates the recovered gain map, so rig uniformity (typically
  specified to within 1-2% across the field of view) matters as much as sensor quality.
- **Dark-frame capture** needs only a light-tight enclosure or a body cap over the lens — no special
  fixture, but the SAME thermal environment as the flat capture (dark current is temperature-dependent,
  THEORY.md), so dark and flat stacks are captured back-to-back at a stable ambient temperature, not on
  separate days.
- **What breaks in the field:** rig-side — integrating sphere coating degrades and yellows with UV/age,
  slowly biasing the "known-uniform" assumption the whole calibration rests on, and must itself be
  periodically re-verified against a calibrated reference; camera-side — a bumped lens mount shifts the
  decentering this project models, invalidating the RADIAL fit's implicit center assumption
  (THEORY.md "The algorithm") until recalibrated.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-11. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

| Tier | Compute | Calibration rig | Notes |
|---|---|---|---|
| Hobby / prototype | A laptop or Raspberry Pi 5 doing the offline calibration compute (this pipeline finishes in well under a millisecond of GPU time on a desktop RTX-class card, per `demo/expected_output.txt` — even a CPU-only run is fast enough for a one-time bring-up step) | A sheet of translucent white acrylic or a foam-core-and-tracing-paper diffuser, backlit by a ring light or a few LED panels; a lens cap for dark frames | Adequate for a hobby-scale calibration; illumination uniformity is the main quality risk at this tier |
| Research / mid-tier robotics | NVIDIA Jetson Orin NX/Nano (the GPU tier this repo targets, CLAUDE.md §5) for the LIVE per-frame `correction_kernel`; the calibration COMPUTE itself can run on any machine, offline | A small integrating sphere (a few inches to ~30 cm diameter, ~$500-3,000) with a calibrated LED or halogen source | The correction (not the calibration) is what needs to run on the robot's own compute, at the camera's live frame rate |
| Industrial / production | A dedicated manufacturing-test PC per test station (not the robot's own onboard compute — calibration happens on the LINE, not in the field) | A large (12"+ / 30 cm+) precision integrating sphere with a spectrally-calibrated, stabilized light source (~$5,000-30,000+), often paired with commercial flat-fielding software (e.g., from a machine-vision camera vendor's own calibration toolkit) | This is where EMVA 1288-style formal sensor characterization is actually performed, with traceable calibration certificates |

- **Compute:** the demo here runs comfortably on a desktop RTX-class GPU (the repo's reference machine,
  CLAUDE.md); a fielded robot's LIVE correction path needs only Jetson Orin-class (or smaller) compute —
  `correction_kernel` is a trivial per-pixel map, cheap even at production resolutions. The CALIBRATION
  half is an offline, one-time (or infrequent) computation and has essentially no hardware constraint.
- **Sensor interface:** MIPI CSI-2 or GigE/USB3 Vision, same as 01.08 — the calibration rig captures
  through the SAME interface the camera uses in operation, so the recovered fields are valid for the
  actual production data path, not a bench-only configuration.
- **Illumination source stability:** for EMVA 1288-grade characterization specifically, the light source's
  own spectral and intensity STABILITY over the capture sequence matters — a flickering or drifting source
  contaminates the flat-stack average in a way frame-averaging cannot remove (it is not random noise, it
  is a slow systematic drift), which is why industrial rigs use stabilized, often feedback-controlled
  sources rather than off-the-shelf LED panels.

## 3. Installation & integration — putting it on a real robot

- **Where the two halves run.** The CALIBRATION half (dark/flat-stack capture + this project's GPU
  pipeline) runs OFFLINE, either on a dedicated manufacturing-test station (production) or on any
  developer machine with the camera tethered (bring-up/R&D) — never on the robot's own embedded compute in
  the field. The CORRECTION half (`correction_kernel`) runs ON the robot, on the SAME perception-compute
  tier that runs the rest of the vision stack (01.01's debayer pipeline, 01.04's feature detection) — see
  README "System context" for the two separate rate budgets this split implies.
- **OS / real-time constraints.** Standard Linux (Ubuntu/JetPack) for both halves; the live correction is
  NOT a hard-real-time control loop (`docs/SYSTEM_DESIGN.md`'s kHz control tier) — it is a perception
  pre-processing stage that must simply finish well inside the camera's own frame period.
- **ROS 2 shape.** The recovered calibration (this project's `gain_recovered`/`dsnu_recovered` maps, or
  the compact `a2,a4,a6` parametric form) is naturally published/stored the same way ROS 2 stores intrinsic
  calibration today: a `sensor_msgs/CameraInfo`-ADJACENT YAML/binary calibration file loaded once at node
  startup (the same pattern `camera_calibration`/`image_pipeline` uses for lens distortion coefficients —
  this project's gain/offset fields are a natural sibling entry in that same file, not a separate topic).
  The LIVE correction itself is a simple filter node: subscribes to the raw `sensor_msgs/Image`, applies
  `correction_kernel`, republishes a corrected `sensor_msgs/Image` — a single-responsibility node upstream
  of everything in `docs/SYSTEM_DESIGN.md`'s perception stack.
- **Calibration & bring-up procedure.** (1) Mount the camera in front of the calibration rig; (2) capture
  `N_dark=16` dark frames (lens capped, same thermal state as step 3); (3) capture `N_flat=16` flat frames
  through the integrating sphere/diffuser at a documented illumination level; (4) run this project's GPU
  pipeline to produce the calibration file; (5) store it alongside the camera's serial number / mount
  position, since NUC tables are per-UNIT, not per-model (silicon variation is unit-specific by
  definition). **Recalibration triggers:** any lens swap or remount (invalidates the vignette/decentering
  fit), a significant sustained temperature shift (invalidates DSNU specifically — dark current
  roughly doubles every 6-8 degC, THEORY.md), and routine periodic re-verification on a fixed schedule for
  safety- or measurement-critical deployments.
- **The safe hardware-testing ladder.** This project's output is a calibration file / corrected image,
  never a motion command — the usual actuator-testing ladder (simulation -> HIL -> bench-jig/tethered ->
  free-running, with E-stop and limits at every rung) is **N/A** in the strict sense: nothing in this
  pipeline drives an actuator. The practical caution, same as 01.08's: any downstream consumer that gates
  a motion decision on corrected imagery (a vision-based inspection reject, an obstacle detector) inherits
  this pipeline's failure modes (a stale/wrong calibration reading as a phantom intensity gradient) as an
  input-quality risk its OWN testing ladder must account for. Per CLAUDE.md §1: everything in this
  repository is sim-validated only, not safety-certified.

## 4. Business & regulatory context

*Didactic orientation, dated 2026-07-11 — not procurement, legal, or compliance advice.*

- **Who needs this.** Any robot or system whose camera output feeds a task sensitive to ABSOLUTE or
  RELATIVE pixel intensity, not just edges/shapes: machine-vision inspection (bin-picking vision systems,
  PCB/produce inspection, color/reflectance measurement — `docs/SYSTEM_DESIGN.md`'s manipulator work cell,
  domain 01), photometric/direct visual SLAM systems (DSO-class, README "System context"), multi-camera
  surround-view stitching where seam mismatches must be invisible (01.07, by name), and any camera
  characterized to a formal metrology standard for a regulated or contractual reason (medical/scientific
  imaging, quality-controlled manufacturing lines).
- **Commercial and open-source players.** Machine-vision camera vendors (Basler, FLIR/Teledyne, Allied
  Vision, Sony's industrial sensor line, among others) ship factory-calibrated NUC tables and often
  in-camera flat-field-correction firmware as a standard feature; EMVA (the European Machine Vision
  Association) maintains the EMVA 1288 standard itself; OpenCV and vendor SDKs both ship flat-field
  correction utilities; the direct-VO/SLAM research community (DSO and its descendants) maintains
  open-source photometric-calibration tooling.
- **What getting it wrong costs.** For inspection/metrology use: an uncorrected vignette reads as a
  spatially-varying MEASUREMENT BIAS — a part near the frame edge measures systematically different from
  an identical part near the center, a false-reject/false-accept risk with direct cost (scrapped good
  parts, or worse, accepted bad ones). For photometric SLAM: an uncalibrated vignette directly corrupts the
  direct-intensity residuals the whole estimator is built on, degrading pose accuracy in a way that can be
  hard to diagnose (it looks like "worse tracking," not an obvious hard failure). For multi-camera
  stitching: visible seams are a product-quality/customer-perception cost, not usually a safety one.
- **Standards / regulatory path.** **EMVA 1288** ("Standard for Characterization of Image Sensors and
  Cameras") is the directly-relevant standard here: it formalizes exactly the PRNU/DSNU/dark-current/noise
  measurement procedures THEORY.md's "Where this sits in the real world" section names — cited here as
  orientation (this project computes illustrative, teaching-scale versions of related quantities; it does
  not claim EMVA 1288 compliance or certification). More broadly, this work sits under whatever
  functional-safety or quality framework governs the platform/product it ships in — see
  `docs/SYSTEM_DESIGN.md` item 6 for the full regulatory-landscape map by robot type (industrial arms:
  ISO 10218/ISO-TS 15066; service robots: ISO 13482; AVs: ISO 26262/UL 4600; medical: IEC 60601) — this
  project's own output is a calibration artifact several layers removed from any of those certifications,
  not itself a certified measurement.
- **Where this work lives inside a robotics company.** Perception / camera-systems engineering owns the
  algorithm and the calibration file format (`docs/SYSTEM_DESIGN.md` item 5); **manufacturing test
  engineering** owns running it on every unit at EVT/DVT/PVT bring-up (item 5.2 lists "calibration
  pipelines (01, 02)" explicitly as an EVT/DVT/PVT-stage deliverable); adjacent teams are optics/mechanical
  (lens mount tolerances, Section 1) and, for an inspection-grade or metrology product, a dedicated
  image-quality/calibration-metrology function responsible for the rig itself and its own traceability.
