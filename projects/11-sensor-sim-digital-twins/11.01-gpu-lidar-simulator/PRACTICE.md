# 11.01 — GPU LiDAR simulator: BVH raycasting + beam divergence, intensity, dropout noise: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> This project's own output stands in for a physical sensor's data — so this file's §1–2 teach the
> REAL sensor's construction and silicon, the way a control project's §1–2 would teach its actuation
> chain (08.01 PRACTICE.md's own framing, applied here to sensing instead of actuation).

## 1. Building it — construction of the robot/part

The physical carrier this project's code stands in for is a **mechanical spinning LiDAR head** (the
sensor family this project's scan-pattern model — fixed elevation channels, rotating azimuth sweep —
directly represents):

- **The rotating assembly.** `channels` laser-diode/photodetector pairs are mounted at fixed elevation
  angles on a core that spins about a vertical axis, driven by a small BLDC motor. Power and data must
  cross the rotating joint: older/cheaper designs use a physical **slip ring** (a wearing mechanical
  contact — the classic long-service-life failure point of a spinning LiDAR); newer designs use a
  non-contact rotary transformer for power plus an optical or capacitive coupling for data, trading a
  wear item for more complex electronics. Rotational **balance** matters: an unbalanced spinning head
  vibrates itself and its bearings apart over months of continuous operation — units are dynamically
  balanced at manufacture.
- **The window/dome.** The spinning core (or a static housing around it) is enclosed by a dome or
  cylindrical window transparent to the laser wavelength (905 nm or 1550 nm — see §2), sealed to an
  IP rating against dust and rain, often anti-reflective coated. **Window fouling — dust, mud, water
  film, condensation — is a real, common field failure mode that degrades or eliminates returns exactly
  the way this project's statistical dropout model represents it**: not a hardware fault, just less
  light coming back. Scratches and pitting (stone chips on a vehicle-mounted unit) do the same
  permanently.
- **Solid-state/MEMS alternatives skip the spinning assembly entirely** — a MEMS micro-mirror or an
  optical phased array steers the beam electronically, trading mechanical wear for a smaller field of
  view or a non-repeating scan pattern (both real trade-offs against this project's simple "fixed grid
  of channels x azimuth steps" model, which best describes classic mechanical spinning units).
- **Per-channel calibration.** Real laser alignment is never perfectly even; every unit ships
  factory-calibrated with per-channel elevation-angle and timing corrections baked into its driver
  firmware — this project's evenly-spaced elevation model (README/THEORY) is the teaching
  simplification a real driver's calibration table replaces.
- **What breaks in the field:** window fouling/scratches (above), slip-ring or bearing wear on the
  spinning joint, connector fatigue at the rotating interface, and seal degradation letting moisture in
  — the mechanical-wear story of any spinning sensor on a vibrating vehicle.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-10. All parts named below are **illustrative examples, never endorsements**;
part numbers and prices go stale — verify current before relying on any of them.*

**Running this simulator** (compute tier — the interesting column is where the ~1 ms/frame raycast
executes): Jetson Orin class for an on-robot/edge deployment of a similar tool, or x86 + discrete RTX
for a workstation/CI farm (reference machine: RTX 2080 SUPER) — the same GPU that would ALSO run the
real-time perception stack downstream of this simulator's output, or a much larger cloud GPU farm for
bulk synthetic-dataset generation (thousands of scenes in parallel, an embarrassingly parallel batch
job at the fleet level, not a single simulator instance).

**The real sensor this project models**, tiered:

| Tier | Illustrative examples (2026) | What's different from this project's model |
|---|---|---|
| Hobby 2D | RPLidar-class rotating single-plane units (~US$100s) | One channel, not a multi-channel 3-D scan; a real classroom starting point |
| Research/robotics multi-channel spinning | Velodyne/Hesai/Ouster-class 16–128 channel (~US$1k–10k+ depending on channel count and range) | Per-channel calibration, multi-return, real noise characterization this project approximates |
| Solid-state / MEMS | Livox, Hesai AT-series, Ouster solid-state (~US$500–2k at volume) | No spinning assembly; often a non-repeating or limited field-of-view scan pattern, not this project's fixed grid |
| FMCW (coherent) | Aeva-class, and other frequency-modulated coherent LiDAR products | Measures per-point RADIAL VELOCITY as well as range (via Doppler beat frequency) — a genuinely different physical principle from this project's pulsed time-of-flight model (THEORY.md "Where this sits in the real world") |
| Automotive long-range | 905 nm or 1550 nm units built for ASIL-rated integration | 1550 nm allows higher transmit power at the same eye-safety class than 905 nm (a real wavelength-choice trade-off — 1550 nm is absorbed by the eye's cornea before reaching the retina) |

**The silicon inside a real unit:** laser driver ICs (nanosecond-scale pulse generation), an APD or
SPAD receiver front-end with a transimpedance amplifier (TIA), a high-speed ADC or time-to-digital
converter (TDC) for the leading-edge timing this project's `range_noise_base_m`/`per_m` stand in for,
and an FPGA/ASIC/SoC that assembles per-channel timing + reflectivity into the point stream, typically
output over Ethernet/UDP (the common interface for robotics/automotive LiDAR — a real driver node
parses that UDP packet stream into `sensor_msgs/PointCloud2`, the message shape this project's own
output deliberately mirrors).

## 3. Installation & integration — putting it on a real robot

- **Where THIS CODE runs:** nowhere on a real robot, by design — it is an **offline / CI / HIL
  synthetic-data tool**, run on a simulation-and-tools team's workstation or cloud GPU farm whenever a
  scene or scenario needs regenerating, with no real-time constraint of its own (contrast: the real
  sensor's own scan rate is fixed in hardware at 10–20 Hz, and its DRIVER node has a soft real-time
  budget just to keep up with the incoming packet stream without drops).
- **What a real sensor's ROS 2 integration looks like** (for context, since this project's output is
  shaped to match it): a vendor driver node subscribes to nothing and publishes `sensor_msgs/msg/PointCloud2`
  (matching this project's own `PointCloud` struct — SYSTEM_DESIGN.md §3.6) at the sensor's native scan
  rate, typically over UDP from the sensor's own onboard processing; `tf2` carries the sensor's
  extrinsic pose (`T_base_lidar`, exactly what `data/sample/sensor_poses.csv` supplies to this project),
  established once during **extrinsic calibration** (a checkerboard-in-3D or structure-based routine —
  the real-world counterpart of this project's committed pose file, minus the calibration procedure
  that would normally produce it).
- **The testing ladder for a SIMULATOR is shaped differently from a controller's** (CLAUDE.md §1's
  ladder assumes something is about to move; nothing here does):
  1. *Synthetic self-consistency* — this project's own analytic gates (THEORY.md "How we verify
     correctness"): does the simulator's OWN physics model behave the way its own equations predict?
  2. *Synthetic-vs-real comparison* — run the SAME downstream perception algorithm against this
     simulator's output and against a real sensor's recording of a similar scene; compare detection/
     registration quality. This is the actual "sim-to-real gap" measurement every synthetic-data tool
     eventually needs, and this project's README §Limitations names every place that gap is likely
     widest (multi-return, atmospheric effects, per-beam origin offset).
  3. *Integrated CI* — this simulator's output feeding a nightly regression test of the full
     perception/mapping stack (a real, common use of exactly this kind of tool at fleet-scale companies).
  4. *Only once a CONTROLLER acts on real sensor data* does this repo's usual hardware-testing ladder
     (simulation -> HIL -> bench -> free running, with E-stops throughout) apply — and it applies to
     THAT controller/perception stack, not to this simulator, which never commands anything.
- **N/A here:** no fieldbus, no onboard deployment, no calibration procedure — this project generates
  data offline; it does not run on, or talk to, any physical robot. Stated per contract.

## 4. Business & regulatory context

- **Who needs this capability:** every company doing serious perception/mapping/planning development
  for LiDAR-equipped robots or vehicles needs large volumes of diverse, PRECISELY GROUND-TRUTHED sensor
  data — synthetic generation is now standard industry practice specifically because real data
  collection is slow, expensive, cannot be safely driven into an edge case on purpose, and never ships
  perfect ground truth the way a simulator does by construction. This capability sits squarely inside
  the digital-twin/simulation-and-tools function every serious robotics or AV program builds.
- **The players:** NVIDIA Isaac Sim/Omniverse (this project's most direct production analogue), CARLA
  (open-source, heavily used in AV research), and a substantial commercial AV-simulation industry —
  Applied Intuition, Waabi, dSPACE, ANSYS AVxcelerate, rFpro, Cognata, Foretellix, among others — each
  competing on physical fidelity, scenario-authoring tooling, and how convincingly their synthetic
  sensor data closes the sim-to-real gap. Build-vs-buy (SYSTEM_DESIGN item 5.3): most companies BUY a
  simulation platform and build scenario/scene content on top of it; a from-scratch sensor simulator
  like this teaching project is rare in production, exactly because the physical fidelity bar (§THEORY
  "Where this sits in the real world") is high and the commercial platforms have already climbed it.
- **Cost of getting it wrong:** a simulator whose synthetic data does not match real sensor behavior
  creates FALSE CONFIDENCE — a perception stack validated against an optimistic simulator can fail on a
  real sensor's actual noise/dropout/multi-return characteristics in the field, a genuine safety and
  liability risk. This is precisely why this project documents every simplification honestly in README
  §Limitations rather than letting a learner believe the teaching core is production-fidelity: an
  acknowledged sim-to-real gap is manageable; a hidden one is a business and safety problem.
- **Regulatory:** simulation itself is not directly regulated the way a deployed sensor or controller
  is, but simulation-based evidence is INCREASINGLY part of the regulatory pathway itself — UL 4600's
  safety-case framework explicitly accepts simulation alongside real-world testing as safety evidence,
  and AV regulatory frameworks are developing simulation-mileage credit programs (SYSTEM_DESIGN.md
  §6.2's AV row: ISO 26262, UL 4600). Consult the SYSTEM_DESIGN item-6 orientation map for the broader
  regulatory landscape; this section is orientation, not compliance guidance.
- **Owning team:** simulation & tools (SYSTEM_DESIGN item 5's org map) — producing scenes and sensor
  models that the perception team consumes for algorithm development and the QA/functional-safety team
  consumes for regression and edge-case test coverage; adjacent to ML/data (which needs synthetic
  training data at scale) and to the perception team that owns the sensor drivers this project's output
  format deliberately mirrors.

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
