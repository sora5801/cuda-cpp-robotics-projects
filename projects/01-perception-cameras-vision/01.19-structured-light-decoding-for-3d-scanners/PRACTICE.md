# 01.19 — Structured-light decoding (Gray code, phase shift) for 3D scanners: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.
>
> *Sections dated 2026-07-11.*

## 1. Building it — construction of the robot/part

**The scanner head.** A structured-light scanner head is a rigid enclosure holding a **projector**
and a **camera** (sometimes two cameras, for a stereo-assisted variant) at a fixed, precisely known
offset — the `kBaselineM` this project's `kernels.cuh` treats as a compile-time constant. Physically
building one means:

- **The projector engine.** A DLP (Digital Light Processing) projector's core is a **DMD** (Digital
  Micromirror Device): a chip carrying one microscopic, individually-tiltable aluminum mirror per
  pixel (up to a few million), each flipping between two ~10-12 degree positions tens of thousands of
  times per second under electrostatic control, steering a fixed light source's beam either INTO or
  AWAY FROM the projection lens — a binary (on/off) light valve at the hardware level, which is
  exactly why DMD-based projectors render Gray code's binary bit-planes natively and synthesize
  grayscale/sinusoidal patterns (this project's phase-shift stripes) via fast time-domain dithering of
  that binary switching. 3LCD and LCoS projectors instead use liquid-crystal panels that continuously
  vary transmittance/reflectance per pixel — natively analog, and often the choice for the smoothest
  sinusoidal fringes.
- **The rigid frame and thermal drift (honesty).** The camera and projector are bolted to a common
  metal (usually aluminum or a low-thermal-expansion alloy for precision units) chassis, calibrated
  ONCE (a checkerboard/ChArUco procedure — project 01.16 — plus the camera-projector extrinsic
  solve — project 01.17). That calibration is only as good as the frame's rigidity: differential
  thermal expansion between the camera and projector mounting points (the projector's light source is
  a real heat source; aluminum's thermal expansion is ~23 micrometers per meter per degree C) shifts
  the effective baseline by tens of micrometers per degree of temperature swing — small compared to
  a `120 mm` baseline, but NOT negligible against this project's own measured ~mm-scale reconstruction
  precision (THEORY.md's worked depth-precision example). Industrial units address this with
  temperature-monitored recalibration schedules or athermal (invar/carbon-fiber) frames; hobby builds
  usually just accept the drift and recalibrate often.
- **Projection-frequency / exposure synchronization wiring.** The camera's exposure window must be
  synchronized to each projector pattern's display window — a hardware trigger line (projector "frame
  sync" or "vsync" output into the camera's external-trigger input, or vice versa) is standard in
  purpose-built scanners; DIY rigs often fake it with a fixed delay tuned to the projector's known
  refresh rate, which is fragile against the SAME per-pattern motion sensitivity this project's README
  "Limitations" already discusses. Camera exposure must also be short enough to avoid capturing a
  PARTIAL pattern transition (motion blur of the pattern itself, distinct from scene motion).

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-11. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

| Tier | Projector | Camera | Compute | Rough head cost (verify current) |
|------|-----------|--------|---------|-------------------------------------|
| Hobby / maker | A commodity DLP pico-projector (e.g. TI DLP-Lightcrafter-class eval boards, or a repurposed consumer pico-projector) | A USB3 machine-vision camera (e.g. an entry FLIR/Basler global-shutter module) or even a webcam for a first prototype | A desktop PC with a discrete GPU (this repo's own target) | Roughly USD 300-1,500 for projector + camera, excluding the GPU host |
| Research / prosumer | A dedicated structured-light engine board (TI DLP-based reference designs; higher pattern rate, external-trigger I/O) | A synchronized global-shutter industrial camera with hardware trigger input | Jetson Orin-class SoC (edge) or an industrial PC + RTX-class dGPU | Roughly USD 2,000-8,000 |
| Industrial (illustrative — Zivid/Photoneo/Ensenso/Gocator-class) | A purpose-built **blue-LED** structured-light engine (blue wavelengths reduce ambient/IR interference and improve contrast on many materials; often multi-frequency phase capable) in a sealed, athermal housing | Two synchronized global-shutter sensors (stereo-assisted structured light) or one high-resolution sensor | Onboard embedded compute (often an FPGA/SoC doing pattern generation AND initial decode) plus a host PC | Roughly USD 8,000-40,000+ per head |

**Actuation chain silicon:** N/A for this project — a structured-light SCANNER has no actuation
chain of its own; the reference-robot integration in §3 below drives a SEPARATE actuation chain
(the manipulator arm) that CONSUMES this project's output.

**Comms/power:** USB3 or GigE Vision (camera), HDMI/DisplayPort or a proprietary digital interface
(projector pattern feed in industrial units — patterns are often stored on-device, not streamed
live), 12-24 V DC power for the projector's light source and cooling fan.

## 3. Installation & integration — putting it on a real robot

**Where this runs.** In the reference **6-DoF manipulator work cell** (SYSTEM_DESIGN.md §2.2), the
scanner head mounts wrist- or frame-fixed over the workspace/bin; the decode pipeline this project
implements runs on the cell's industrial PC (the same machine, or one adjacent to, the arm
controller) — the pattern acquisition + decode + triangulation cycle (README "System context": ~200
ms - 2 s per full scan) sits comfortably outside any hard real-time deadline; the arm's own joint
control loop (SYSTEM_DESIGN.md §1.1: 0.5-1 kHz) is a completely separate, much faster, downstream
consumer of this project's OUTPUT (a static point cloud), not of its internals.

**Bin-picking cell workflow (SYSTEM_DESIGN.md Chain B, §4.2, this project substituting for `01.02`):**

```
[project a full 20-pattern stack over the bin] -> [this project's 5-stage decode] -> PointCloud
    -> [19.01 antipodal grasp scoring: sample + score grasp candidates on the cloud]
    -> [09.05 batched IK / 06.07 motion planning -> 08.03 tracking control: execute the grasp]
    -> [01.14 template matching at scale: VERIFY the part was actually picked/placed correctly]
```

Structured light is the natural front end for this chain specifically when bin contents are
textureless or reflective (raw metal parts, injection-molded plastic) — exactly where `01.02`'s
passive stereo block matching runs out of texture to match (README "Prior art").

**ROS 2 node/topic shape.** This project's output maps directly onto a `sensor_msgs/msg/
PointCloud2` publication from a `structured_light_scanner_node`:

```
/scanner/points        sensor_msgs/msg/PointCloud2   (camera optical frame; one message per completed scan)
/scanner/pattern_stack sensor_msgs/msg/Image[]        (optional: the raw captures, for offline reprocessing/debug)
/scanner/scan_trigger  std_srvs/srv/Trigger            (a SERVICE, not a topic — "acquire and decode one scan now",
                                                          reflecting that this is a request-response, not a streaming, sensor)
```

modeled on this repo's `PointCloud` sketch (SYSTEM_DESIGN.md §3.6) — the `frame_id` would be the
scanner's calibrated optical frame, `T_base_scanner` published once (a static transform) from the
same 01.16/01.17-style calibration this project's `kernels.cuh` constants stand in for.

**Calibration maintenance.** Camera intrinsics (01.16) and camera-projector extrinsics (01.17) are
NOT one-and-done: §1's thermal-drift discussion means a real cell schedules periodic
re-calibration (daily/weekly depending on tolerance budget and environment) using a physical
checkerboard/ChArUco target placed in the scan volume, with the reprojection-error trend logged as a
maintenance signal (a sudden jump flags a bump, a loose mount, or a failed mirror/panel element).

**Safe hardware-testing ladder (CLAUDE.md §1 caveat applies in full — this repo is sim-validated
only):** simulation (this project, exactly as shipped) -> hardware-in-the-loop with a static test
target and NO robot motion commanded from the scan result -> a bench jig where the scanner's output
drives a LOGGED-ONLY grasp plan (no physical execution) for a human to review -> tethered/current-
limited execution on a real arm with an E-stop and reduced speed/force limits -> free running only
after the above stages demonstrate consistent, bounded reconstruction error on KNOWN test geometry
(a calibration sphere or gauge block is the standard metrology check, mirroring this project's own
`reconstruction_sphere`/`reconstruction_step` gates).

## 4. Business & regulatory context

**Who needs this, and the market.** Metrology-grade 3-D scanning is a distinct, mature commercial
segment (bin-picking vision for logistics/manufacturing, dimensional inspection/quality control,
reverse engineering, and — outside robotics — dentistry, cultural-heritage digitization, and VFX)
worth billions of dollars globally; the main commercial players building camera-projector heads
range from industrial-vision specialists (Zivid, Photoneo, IDS/Ensenso, LMI/Gocator, Cognex) to
metrology-focused vendors (GOM/Zeiss, Artec3D), alongside open-source/hobby tooling (OpenCV's
`structured_light` module, this project's own teaching implementation) — SYSTEM_DESIGN.md §5.1
places this work with the **Perception** org (domains 01/02/03/20), directly adjacent to the
**calibration** team (01.16/01.17) and the **manipulation/grasp-planning** team (19.x) that consumes
this project's point cloud; typical role titles include perception engineer, 3-D vision engineer,
and (at scanner VENDORS specifically) optical/systems engineer.

**What getting it wrong costs.** A miscalibrated or drifted scanner silently degrades grasp success
rate (missed picks, dropped parts) or — for inspection use — passes out-of-tolerance parts, both of
which are quality/liability costs the METROLOGY discipline exists to bound (traceable calibration
standards, gauge R&R studies) — this project's `reconstruction_sphere`/`reconstruction_step` gates are
a toy, didactic version of exactly that discipline: verify a known geometric truth, not just internal
self-consistency.

**Photobiological safety of projectors (orientation, not compliance guidance).** A structured-light
projector is, physically, a bright light source aimed at a scene that may include people (an operator
reaching into a bin-picking cell, or standing nearby). **IEC 62471** ("Photobiological safety of
lamps and lamp systems") is the relevant standard family governing exposure limits for this class of
device — the SAME "bright light near people" concern project 01.08 (HDR/exposure fusion for outdoor
robots) touches from the CAPTURE side; here it applies to the EMITTING side. This is cited as an
orientation pointer only — see SYSTEM_DESIGN.md §6.2's regulatory table, which does not carry a
dedicated row for photobiological/lighting standards (it is cross-cutting rather than robot-type-
specific) and instead lists the closest-fit collaborative-safety standards (ISO/TS 15066) that would
govern the CELL this scanner sits in, cited in that same table.

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
