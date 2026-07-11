# 01.23 — Full RAW→RGB ISP on Jetson (Argus + custom CUDA stages): Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md section 4.3). It grounds the README "System context" section in the physical and
> commercial whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.
>
> *Sections dated 2026-07-11. All parts, prices, and vendor names below are **illustrative examples,
> never endorsements** — verify current before relying on any of them.*

## 1. Building it — construction of the robot/part

This project's "part" is the **camera module**: the physical carrier every stage in this ISP exists
to process the output of.

**Construction.** A robot-grade camera module is a small PCB stack: an image sensor die (the Bayer
CFA is literally deposited on top of the silicon photosite array during fabrication — the physical
origin of every raw pixel this project's `bayer_phase_at()` reasons about), a lens barrel (glass or
molded-plastic elements, sometimes with an IR-cut filter bonded on top), and a flex-PCB or rigid
connector carrying the MIPI CSI-2 differential pair lanes plus I2C for register control (exposure,
gain, sensor mode) to the host board. Mounting tolerance matters more than most learners expect: the
lens-to-sensor distance (back-focal distance) is set at manufacturing time to microns of precision —
this project's lens-shading model (`THEORY.md`'s cosine-fourth-law falloff) is directly sensitive to
this alignment, and a module dropped or thermally cycled beyond spec can shift its shading pattern
enough to need re-calibration. **What breaks in the field:** connector fatigue at the flex-PCB bend
(vibration-heavy field robots see this first), lens-barrel loosening (a focus/sharpness failure, not
this project's domain but a common field complaint that gets misattributed to "the ISP"), and CFA/
microlens degradation under sustained UV or thermal stress (a slow, permanent shading/color-cast
drift this project's shading and CCM stages assume is FIXED and known, which is why a real product
re-calibrates periodically).

## 2. Real hardware — chips, parts, illustrative BOM

**Compute tier.** This project's stages, running as custom CUDA, target one of three tiers depending
on the robot's compute budget: (a) **Jetson Orin-class SoC** (e.g., Orin Nano/NX/AGX) — the realistic
target for the catalog bullet's premise, with an integrated GPU alongside the CSI-2 receiver and the
fixed-function hardware ISP `THEORY.md`/README's "Jetson story" describes; (b) **x86 + discrete GPU**
(this project's actual desktop-teaching target, an RTX 2080 SUPER) — common in a larger ground
vehicle or a development/bring-up rig, no hardware ISP available at all, so custom CUDA stages like
this project's are the ONLY software path; (c) **MCU-class** (no GPU) — some low-cost robots run a
much-reduced ISP (often just black level + a cheap demosaic) on a microcontroller's DSP/NPU
accelerator; this project's full pipeline would not fit that tier without significant reduction.

**Sensor.** Illustrative examples across cost tiers (verify current pricing/availability):
hobby/research (Raspberry Pi Camera Module 3, Sony IMX708, ~$25–35; e-con Systems / Leopard Imaging
Jetson-compatible modules, ~$50–150), industrial (Sony IMX-series global-shutter modules via Basler/
FLIR/Allied Vision, ~$200–600, adds features this project's synthetic model does not — global vs.
rolling shutter, higher dynamic range, external trigger for multi-sensor sync).

**Compute silicon around the sensor.** A real bring-up also touches: the CSI-2 receiver PHY (built
into the Jetson SoC or a bridge chip like a Toshiba/Lontium CSI-to-something converter for non-native
sensors), an I2C GPIO expander if the sensor needs more control lines than the carrier board exposes
natively, and — if the defect map or lens-shading calibration is stored per-module rather than
per-model — a small EEPROM on the camera module flex-PCB itself (PRACTICE.md's "Installation" section
below expands on this calibration-storage pattern).

## 3. Installation & integration — putting it on a real robot

**Where this code would run.** On a Jetson-class robot, this project's stages (if used instead of, or
alongside, the hardware ISP) would run on the **GPU tile of the Jetson SoC itself** — not a separate
computer — sharing memory with the CPU cores via Jetson's unified memory architecture (a real
advantage over this project's desktop teaching setup, which pays an explicit PCIe H2D/D2H copy the
Jetson target would not need — **zero-copy NVMM buffers**, one paragraph: Jetson's multimedia API
allocates camera frames in NVMM (NVIDIA Multimedia memory), a physically contiguous buffer type both
the CSI-2 capture hardware AND the GPU can address directly — a CUDA kernel can operate on a freshly
captured frame with ZERO memory copy, which is exactly the "the GPU sees the same bytes the sensor
wrote" property that makes GPU ISP stages on Jetson viable at frame rate; this project's desktop
`cudaMemcpy` calls stand in for that unified-memory path honestly, not deceptively — see
`THEORY.md`'s note that this project measures desktop numbers only).

**OS & real-time constraints.** Jetson runs L4T (a customized Ubuntu), not a hard-real-time OS; camera
capture and ISP processing run as regular (if high-priority) Linux processes/threads, not RTOS tasks
— acceptable because a dropped or late FRAME is usually recoverable (the next frame arrives in
~16–33 ms), unlike a dropped CONTROL loop tick.

**ROS 2 shape.** A production integration would wrap this pipeline's output as a `sensor_msgs/Image`
(or `sensor_msgs/CompressedImage`) published from a `camera_driver` node — the RAW input itself would
never cross a ROS 2 topic in a well-designed system (RAW10 is sensor-specific and un-interpretable
without this project's exact calibration constants); only the corrected, demosaiced RGB (or the
still-Bayer-but-radiometrically-corrected frame, for a downstream node that wants to demosaic itself)
would be published, matching `docs/SYSTEM_DESIGN.md`'s message-shape convention.

**Bus.** The sensor-to-SoC link is MIPI CSI-2 (not a general-purpose bus like CAN/EtherCAT — CSI-2 is
a dedicated high-bandwidth video link, point-to-point, not a shared/addressable bus); downstream of
the ISP, the corrected image would typically stay on-chip (GPU→CPU shared memory) rather than
crossing any external bus at all, unless the frame is being streamed to another compute node over
Ethernet for a distributed perception system.

**Calibration & bring-up procedure.** (1) Capture a black frame (lens capped) at every supported gain/
exposure setting to measure the real black level and per-pixel offset (this project's `kBlackLevel`
stand-in). (2) Capture a uniformly lit flat-field target to measure the real lens-shading gain map
(sibling **01.09** is this repo's dedicated project for fitting exactly this polynomial from real
data, rather than assuming it known as this project does). (3) Run a defect-scan routine (multiple
dark and saturated captures, statistically flag pixels that are stuck or excessively noisy) to build
the real defect map — usually written to a small EEPROM on the camera module itself (not the host
SoC), so the calibration travels with the physical sensor even if it is later paired with a different
compute board. (4) Shoot a color chart (a REAL Macbeth/X-Rite ColorChecker, unlike this project's
illustrative synthetic one) under multiple known illuminants to fit the real CCM and AWB presets.

**The safe hardware-testing ladder** (CLAUDE.md section 1 caveat: this project's output is an image,
not a motion command, but the ladder pattern is stated for completeness and consistency with the
rest of the repo): simulation (this project, entirely) → HIL (replay a recorded RAW stream through
the real Jetson hardware/software path, no live sensor) → bench jig (a live sensor on a bench,
tethered, verifying capture and calibration bring-up before any robot integration) → free running (the
module mounted on the actual robot). Nothing in this project has been run past the first rung.

## 4. Business & regulatory context

**Who needs this, and why it's a real specialization.** Every camera-bearing product — robots,
phones, automotive ADAS/AV, drones, medical imaging, machine vision/inspection — needs *some* form of
this pipeline, and the quality of the tuning (not just the presence of the stages) is a genuine
product differentiator: two cameras with IDENTICAL silicon can look dramatically different depending
on ISP tuning quality, which is why **camera/ISP tuning engineer** is a distinct, valued role
(adjacent teams: sensor/optics hardware, embedded/driver, perception/CV, and — for consumer products —
imaging science/color science). **Commercial players:** ISP IP and tuning tool vendors (ARM Mali-C
ISP IP, Qualcomm Spectra, on the silicon side; NVIDIA's own libargus/Jetson ISP tuning tools), camera
module integrators (e-con Systems, Leopard Imaging, Framos), and a cottage industry of independent
camera-tuning consultancies that robotics/automotive companies contract for exactly the bring-up
procedure in section 3.

**What getting it wrong costs.** A badly tuned or buggy ISP stage is rarely a *safety* failure by
itself (contrast with a control-loop bug) but is a real **product-quality and downstream-perception**
failure: a bad AWB preset (this project's `tungsten_wrong_awb_negative_control` gate is a miniature
of exactly this) systematically shifts colors a downstream classifier or human operator relies on; a
buggy defect-correction stage can leave hot pixels that a detection network learns to (wrongly) treat
as signal; an uncorrected lens-shading falloff biases exposure/AWB statistics computed from
corner-heavy scene content. In automotive/robotics specifically, perception failures downstream of a
bad ISP are a real, if usually not headline, cause of costly re-validation cycles.

**Regulatory path.** Camera image quality itself is rarely directly regulated, but downstream
requirements constrain it: automotive perception cameras increasingly cite **IEEE P2020** (Automotive
Image Quality) — a standards effort covering exactly the metrics this project's gates are a teaching
miniature of (SNR, dynamic range, color accuracy, geometric distortion) — continuing sibling
**01.08**'s IEEE P2020 citation for the tone-mapping/HDR side of the same standard. For a
safety-relevant application, the ISP sits under whatever the DOWNSTREAM system's certification path
requires (ISO 26262 for an AV perception stack, ISO 10218/ISO/TS 15066 for an industrial-arm vision
system) — see `docs/SYSTEM_DESIGN.md` item 6's regulatory map. **This is didactic orientation, not
compliance guidance** — no claim of certification, fitness, or regulatory adequacy is made anywhere
in this project.

---

*Didactic orientation only — **not** procurement, legal, or compliance advice.*
