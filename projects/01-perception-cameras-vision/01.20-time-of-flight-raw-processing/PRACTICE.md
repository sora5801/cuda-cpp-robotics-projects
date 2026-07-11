# 01.20 — Time-of-flight raw processing: phase unwrapping, flying-pixel removal: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.
>
> *Sections dated 2026-07-11.*

## 1. Building it — construction of the robot/part

**The iToF camera module.** Unlike 01.19's two-body (camera + projector) rig, an iToF camera is
typically a SINGLE compact module: an illuminator, a lens, and a sensor sharing one small PCB and
housing. Physically building one means:

- **The illuminator.** A ring or cluster of IR **VCSELs** (vertical-cavity surface-emitting lasers,
  850 or 940 nm — outside the visible band; 940 nm sits in a solar-irradiance dip, helping daylight
  performance) or IR **LEDs**, driven by a dedicated **laser/LED driver IC** that modulates drive
  current at the sensor's chosen frequency (tens of MHz) with tight phase control — the driver's own
  output-phase stability directly sets a floor on this project's `phi` measurement's absolute accuracy
  (any drift here shifts every pixel's phase identically, a DC offset THEORY.md's ambient-cancellation
  algebra does NOT protect against, since it is a shift in the reference signal, not the ambient light).
  A diffuser or micro-lens array spreads the VCSEL array's naturally narrow beams into a wider,
  eye-safety-compliant illumination pattern covering the sensor's field of view.
- **The sensor + lens co-design.** The demodulating pixel array (the "lock-in" or "current-assisted
  photonic demodulator" — CAPD — sensor, this project's `extract_phase_amplitude_kernel`'s physical
  carrier) sits behind a lens whose f-number and IR bandpass filter are matched to the illuminator's
  wavelength — the bandpass filter is what keeps ordinary visible ambient light from swamping the weak
  modulated IR return (the `A` ambient term THEORY.md's tap algebra cancels ALGEBRAICALLY still has a
  physical DYNAMIC-RANGE cost: too much unfiltered ambient light saturates the sensor's charge wells
  before the small modulated signal `B` can be resolved at all).
- **Calibration (honesty).** Every real iToF module needs at least two factory calibrations this
  project's fixed `kernels.cuh` constants stand in for: **fixed-pattern-noise (FPN) / per-pixel offset
  calibration** (manufacturing variation in each pixel's demodulator gives it a slightly different `A`
  and `B` even under identical illumination) and **"wiggling error" calibration** (real CW demodulation
  is never a perfect sinusoid — a real square-wave-driven illuminator and imperfect reference-signal
  generation introduce a small, systematic, DEPTH-DEPENDENT phase error that "wiggles" periodically
  with true depth; vendors correct it with a per-pixel lookup table built from a calibration rig at
  known distances). This project's THEORY.md "The math" derives the IDEAL sinusoidal correlation; real
  hardware's wiggling error is named here, honestly, as a real, uncorrected gap between this project's
  model and a production sensor.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-11. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

| Tier | iToF sensor/module | Illuminator | Compute | Rough module cost (verify current) |
|------|---------------------|-------------|---------|--------------------------------------|
| Hobby / maker | A breakout board built on a consumer iToF chip (e.g. ST VL53L-series-class or a repurposed Kinect-Azure-class module) | Often integrated on the same module | A desktop PC / SBC reading the module over I2C or USB | Roughly USD 20-150 |
| Research / prosumer | A dedicated iToF reference-design module (e.g. PMD/Infineon REAL3-class or Melexis MLX75027-class evaluation kits) with raw-tap access | A driver-controlled VCSEL array, tunable modulation frequency | Jetson Orin-class SoC (edge) or an industrial PC + this repo's own RTX-class dGPU target | Roughly USD 300-2,000 |
| Industrial / automotive (illustrative — Azure-Kinect-class / automotive-cabin-sensing-class) | A qualified automotive- or industrial-grade iToF sensor with on-chip ISP, temperature-compensated calibration tables, and functional-safety-relevant diagnostics | A qualified, eye-safety-certified VCSEL driver assembly | Dedicated SoC or automotive-qualified compute module | Roughly USD 2,000-15,000+ per qualified module/program (NRE-heavy; per-unit cost is much lower at automotive volume) |

**Actuation chain silicon:** N/A for this project — an iToF camera has no actuation chain of its own;
the reference-robot integration in §3 below drives a SEPARATE actuation chain (the AMR's drivetrain or
the manipulator arm) that CONSUMES this project's output.

**Comms/power:** I2C or MIPI CSI-2 (sensor-to-host raw frame transfer, depending on module class), USB
or a dedicated serializer link for longer cable runs; a few watts for the illuminator's drive
electronics (the dominant power draw of the whole module — see the eye-safety discussion in §4) plus
milliwatts for the sensor/ISP itself.

## 3. Installation & integration — putting it on a real robot

**Where this runs.** In the reference **warehouse AMR** (SYSTEM_DESIGN.md §2.1), a forward- or
downward-facing iToF module mounts on the chassis for near-field obstacle/cliff sensing, feeding the
same industrial or embedded PC that runs the rest of the perception stack; in the **6-DoF manipulator
work cell** (SYSTEM_DESIGN.md §2.2), a wrist-mounted iToF module feeds the arm controller's adjacent
perception PC for close-range grasp-approach sensing. Either way, this project's decode pipeline
(README "System context": comfortably sub-millisecond compute at this project's problem size) sits
well inside the sensor's own **15-30 Hz** frame budget — the bottleneck is the SENSOR's multi-tap
capture and readout, not this decode math.

**Interference between multiple ToF cameras (the multi-robot problem).** Every iToF camera in a shared
space EMITS modulated light at its own reference frequency, and every OTHER iToF camera nearby (or the
same one seeing its own light reflected back via a second path) can pick up that light too — if two
units happen to share a modulation frequency, their signals genuinely interfere, corrupting BOTH
cameras' phase estimates in a way that looks like extra noise or a systematic bias, not a clean
failure. Real multi-camera deployments (a warehouse fleet of several AMRs, each with a forward iToF
sensor) mitigate this with: **frequency diversity** (assign each unit, or each of its channels, a
distinct modulation frequency by configuration, the same knob this project's `kFreq1Hz`/`kFreq2Hz`
constants represent); **time-division multiplexing** (synchronize nearby units so only one illuminates
at a time, at the cost of reduced per-unit frame rate); and **pseudo-random phase/frequency hopping**
(spread-spectrum-style, making sustained interference from any one other unit statistically unlikely).
This project's fixed two-frequency scene does not exercise multi-camera interference (a single,
isolated sensor); it is named here as the real-deployment problem the "two fixed CW frequencies"
design choice would need to grow beyond.

**ROS 2 node/topic shape.** This project's output maps onto a `sensor_msgs/msg/PointCloud2`
publication (and, upstream, the raw taps as a debug topic) from a `tof_camera_node`:

```
/tof/points          sensor_msgs/msg/PointCloud2   (camera optical frame; one message per depth frame, ~15-30 Hz)
/tof/depth           sensor_msgs/msg/Image          (32-bit float depth image, same rate — the common alternative
                                                       consumption shape for costmap/obstacle-avoidance nodes)
/tof/raw_taps        sensor_msgs/msg/Image[]        (optional: the 4-8 raw tap frames, for offline reprocessing/debug)
```

modeled on this repo's `PointCloud`/`Image` sketches (SYSTEM_DESIGN.md §3.6) — the `frame_id` would be
the sensor's calibrated optical frame, published as a static transform from the same 01.16-style
intrinsic calibration this project's `kernels.cuh` constants stand in for.

**Exposure/frequency configuration discipline.** Unlike a plain camera, an iToF sensor's "exposure"
setting trades off against BOTH saturation (too long an integration window on a bright/close scene
clips the charge wells, corrupting `A`/`B`) and depth noise (too short starves `B` of signal, per
THEORY.md's derived `sigma_Z ~ 1/B` law) — real drivers expose an auto-exposure mode that targets a
per-frame mean amplitude, analogous to ordinary camera auto-exposure but tuned against THIS project's
confidence signal rather than pixel brightness. Frequency selection is a SEPARATE, coarser
configuration choice (this project's `kFreq1Hz`/`kFreq2Hz`): switching frequencies trades ambiguity
range against precision (THEORY.md "The math") and should be set once per deployment's expected working
distance, not per frame.

**Safe hardware-testing ladder (CLAUDE.md §1 caveat applies in full — this repo is sim-validated
only):** simulation (this project, exactly as shipped) → hardware-in-the-loop with a static test
target (a matte-and-a-glossy patch, at a known distance, to exercise both the confidence mask and a
real multipath case this project does not model) and NO robot motion commanded from the sensor's
output → a bench jig where the sensor's output drives a LOGGED-ONLY obstacle-avoidance or grasp plan
(no physical execution) for a human to review → tethered/current-limited execution on a real AMR or
arm with an E-stop and reduced speed/force limits → free running only after the above stages
demonstrate consistent, bounded reconstruction error on KNOWN test geometry (a calibration sphere or
gauge block at measured distances — mirroring this project's own `reconstruction_sphere`/
`reconstruction_step` gates).

## 4. Business & regulatory context

**Who needs this, and the market.** Compact, low-cost 3-D ranging is a large, fast-growing commercial
segment spanning consumer electronics (phone face-unlock and AR, gaming-console body tracking),
automotive in-cabin sensing (occupant monitoring, gesture control — a market growing alongside ADAS
regulation), and robotics (compact obstacle/proximity sensing on AMRs and near-field grasp sensing on
manipulators). Major commercial players building iToF silicon and modules include Sony (depth-sensing
CMOS), Infineon/PMD Technologies (REAL3), Melexis, STMicroelectronics (the FlightSense / VL53L family,
mostly single-zone/low-resolution dToF and iToF variants), and Microsoft/Analog-Devices-class
components historically behind the Kinect/Azure Kinect line; SYSTEM_DESIGN.md §5.1 places this work
with the **Perception** org (domains 01/02/03/20), directly adjacent to the calibration team (01.16)
and the navigation/manipulation teams (23.x/19.x) that consume this project's point cloud; typical
role titles include perception engineer, optical/systems engineer (at sensor VENDORS specifically),
and depth-sensing / computer-vision engineer.

**What getting it wrong costs.** An iToF sensor with an uncorrected wrap-decision failure rate, an
untuned confidence floor, or an unfiltered flying-pixel population feeding an obstacle-avoidance
costmap can report a phantom obstacle (a flying pixel hanging in open space — a false stop / wasted
detour) or, worse, MISS a real close-range obstacle whose return got masked or wrongly unwrapped — this
project's `aliasing_demo`, `unwrap_recovery`, and `flying_pixel` gates are a toy, didactic version of
the validation discipline a real sensor-integration team runs before trusting a module's output near
people or expensive equipment.

**Eye safety of IR illuminators (orientation, not compliance guidance).** An iToF illuminator is,
physically, an infrared laser or LED source aimed at a scene that may include people — the SAME
concern project 01.18's structured-light-scanner sibling (01.19 PRACTICE.md §4) raises for its visible-
light projector, here for an INFRARED source specifically, which carries the added hazard that the
human eye's blink/aversion reflex does NOT protect against invisible wavelengths. **IEC 60825**
(laser product safety, relevant if the illuminator is VCSEL/laser-based) and **IEC 62471**
(photobiological safety of lamps and lamp systems, relevant for LED-based illuminators) are the
standard families governing exposure limits for this class of device — cited here as an orientation
pointer only, the same cross-reference this repo's LiDAR projects (domain 02, e.g. 02.06's citation of
laser safety) make for their own emitters; see SYSTEM_DESIGN.md §6.2's regulatory table for the
closest-fit collaborative-safety standards (ISO/TS 15066) that would govern the CELL a manipulator-
mounted module sits in.

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
