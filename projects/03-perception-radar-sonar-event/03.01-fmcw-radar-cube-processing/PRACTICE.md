# 03.01 — FMCW radar cube processing: range-Doppler-angle FFTs + CA/OS-CFAR detection: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

This project's physical carrier is a **radar module**: a small RF/antenna PCB plus a compute die,
usually potted or enclosed behind a radar-transparent cover (a "radome"), mounted somewhere on the
robot with a clear view of the environment it needs to sense.

- **The antenna board.** The transmit and receive antennas (this project's `Na = 8` receive elements)
  are typically etched directly into the PCB as patch antennas — no separate antenna hardware, just
  copper traces shaped to resonate at 76-81 GHz. Element spacing (this project's `d = lambda/2 ~ 1.95
  mm` at 77 GHz) is a PCB LAYOUT tolerance problem at these dimensions: a fabrication error of a few
  tens of microns measurably shifts the effective spacing and therefore the angle-estimation math this
  project's `kAntennaSpacingM` constant assumes exactly. High-frequency PCB substrates (e.g. PTFE-based
  laminates, not ordinary FR4, which absorbs too much energy at 77 GHz) and controlled trace impedance
  are load-bearing manufacturing details, not incidental ones.
- **The RF/mixed-signal die(s).** A monolithic microwave IC (MMIC) generates the chirp, drives the
  transmit antennas, and downconverts the receive antennas' echoes to baseband — the physical
  implementation of this project's `synthesize_cube_kernel` formula, done in silicon and analog RF
  circuitry instead of software.
- **The radome and mounting.** A radar-transparent cover protects the antenna board from weather and
  road debris while (ideally) not distorting the beam pattern — radome material and thickness are
  tuned to the operating wavelength; a radome designed for a different frequency band measurably
  degrades range and angle accuracy. Mounting angle/height calibration (where "azimuth = 0" actually
  points in the vehicle/robot frame) is a bring-up step every installation repeats — get it wrong and
  every downstream consumer inherits a constant angular bias.
- **What breaks in the field:** radome contamination (mud, ice, road salt) attenuates or scatters the
  signal; connector/cable degradation on a vibrating platform; and — the failure mode specific to
  radar — RF interference from OTHER radars (a busy road full of similar-band automotive radars, or
  a robot fleet of the same sensor) that this project's clean synthetic noise floor does not model at
  all (see THEORY.md "Limitations" for the honest scope).

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-09. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

Automotive/robotics FMCW radar splits cleanly into three frequency tiers, each with different range,
resolution, and cost trade-offs:

| Tier | Illustrative parts (2026) | Typical use | Cost tier |
|---|---|---|---|
| 24 GHz (legacy, narrowband) | Older blind-spot/corner radar chipsets | Short/mid-range, being phased out for 77 GHz | Low (~US$10s per module) |
| 60 GHz (ISM-band, short range) | Infineon BGT60-series, Google Soli-class | In-cabin sensing, gesture, short-range robotics | Low-mid (~US$10-30 per module, hobby-accessible dev kits) |
| **76-81 GHz (this project's band)** | TI AWR1xxx/AWR2xxx ("IWR" = industrial variant), NXP S32R-series, Infineon RXS8160-class | Long-range automotive, industrial/robotics perception | Mid (~US$50-200 research module) to higher for automotive-qualified parts |

Where this project's compute maps onto real hardware:

| Piece | Illustrative choices (2026) | Role in this project's pipeline |
|---|---|---|
| RF front end + on-chip DSP/accelerator | TI AWR2944-class (integrated MCU + hardware FFT accelerator), or a separate RF frontend feeding an FPGA/DSP | Runs THIS PROJECT'S ENTIRE PIPELINE (synthesis is the RF hardware itself; range/Doppler/CFAR/angle run on the chip's DSP or dedicated accelerator) — notably, this pipeline does NOT typically need a discrete GPU in production; the reference machine's RTX 2080 SUPER stands in for a purpose-built accelerator here purely for teaching |
| Host/fusion compute | Jetson-class GPU SoC or x86 + dGPU (the "downstream" box) | Consumes this project's OUTPUT (a detection list), runs sensor fusion/tracking (domain 04) — this is where a GPU genuinely earns its place in a radar-equipped robot |
| Power | Automotive 12V rail (vehicles) or the robot's DC/DC tree (SYSTEM_DESIGN §6.1) | Powers the RF front end and its digital back end |

The reference machine this project actually built and measured on (RTX 2080 SUPER, sm_75) is a
**desktop teaching stand-in**: real radar SoCs run this pipeline on far smaller, far lower-power
silicon than a discrete GPU, at a fraction of the cost — the GPU's role here is pedagogical (a
CUDA-programmable platform to teach the SAME algorithms), not a claim that production radar needs one.

## 3. Installation & integration — putting it on a real robot

- **Where this code would physically run:** on the radar module's OWN compute (the DSP/accelerator
  named above), not on the robot's general-purpose compute — a real radar module ships a finished
  detection list, not a raw cube, to the rest of the robot. This project's GPU implementation is a
  TEACHING stand-in for that on-module DSP/accelerator (§2's honest caveat), useful for learning the
  algorithms even though it is not the deployment target.
- **The interface a real radar module exposes:** almost universally **CAN-FD or Automotive Ethernet**
  carrying a **detection list** (range, velocity, azimuth, RCS/power per detection — exactly this
  project's `Detection` struct, extended with a calibrated RCS in production) at the 10-20 Hz frame
  rate; the raw ADC cube this project processes NEVER leaves the module in a production system — it is
  proprietary, high-bandwidth (this project's single frame is ~2 MB; a continuous stream at 10-20 Hz
  would be tens of MB/s), and simply not exposed on any external bus. Some development/research
  modules (e.g. TI's mmWave SDK boards) DO expose raw ADC data over USB/Ethernet for exactly the kind
  of algorithm development this project teaches — that development-mode interface is this project's
  closest real-world analogue.
- **ROS 2 shape:** a radar driver node publishes a detection-list message (there is no exact
  standard message across the ROS ecosystem the way `sensor_msgs/PointCloud2` is standard for LiDAR;
  common practice is a custom or `radar_msgs`-family message carrying per-detection range/velocity/
  angle/RCS) at the sensor's native frame rate — the downstream tracking node (domain 04) subscribes
  to it exactly as `PointCloud`/`Image` consumers do elsewhere in this repo's convention
  (SYSTEM_DESIGN §3.6).
- **Calibration and bring-up:** boresight alignment (mounting angle vs. the vehicle/robot's forward
  axis — a constant azimuth bias if wrong), and for velocity, verifying the sign convention this
  project fixes throughout (`kernels.cuh`: positive = approaching) matches whatever convention the
  actual module/driver uses — a silent sign flip here would make an approaching obstacle look like it
  is receding, a genuinely dangerous class of bug were this ever connected to a real safety-relevant
  system (it is not, in this repository — CLAUDE.md §1).
- **Testing ladder:** simulation (this project's synthetic scenes, with edited target lists per README
  Exercise 3) -> a radar module on a bench with a rotating corner reflector at known range/angle (the
  standard radar bring-up rig) -> mounted on the stationary robot observing a moving target of known
  trajectory -> mounted on the moving robot. Nothing in this project commands motion; it only produces
  a detection list, so the strict E-stop/current-limited rungs 08.01's PRACTICE.md walks for a
  CONTROLLER do not directly apply here — but any system built ON TOP of this detection list that DOES
  command motion inherits that full ladder.

## 4. Business & regulatory context

- **Who needs this capability:** automotive OEMs and Tier-1 suppliers (ADAS and autonomous-driving
  stacks — radar is one of the few sensors that keeps working in rain, fog, and glare, which is why
  every serious AV sensor suite includes it alongside cameras and LiDAR rather than treating it as
  redundant), industrial/warehouse robotics (all-weather obstacle detection), and increasingly
  consumer/robotics gesture and presence sensing at 60 GHz.
- **The players:** automotive radar silicon is dominated by a small number of vendors (TI, NXP,
  Infineon, Bosch, Continental, and others building on that silicon); Tier-1 suppliers integrate the
  silicon into shipped modules; OEMs and AV companies build the fusion/tracking/planning stack this
  project's detection list feeds into. Build-vs-buy is rarely a question for the RF front end itself
  (silicon-level radar design is a deep specialty almost nobody builds in-house) — the build decision
  is usually about the FUSION/TRACKING software layer above it (SYSTEM_DESIGN §5.3).
- **What getting it wrong costs:** a radar that misses a target (a false-negative, the CA-CFAR masking
  failure this project measures) in a safety-relevant stack is a missed-detection safety incident; a
  radar that reports too many false alarms erodes trust in the whole sensor suite and can trigger
  false braking/avoidance events — both failure directions are commercially and safety expensive,
  which is exactly why CFAR's whole design point is CONTROLLING the false-alarm rate rather than just
  maximizing detection probability.
- **Regulatory: radio spectrum, not (directly) functional safety.** Unlike most of this repository's
  PRACTICE.md sections, radar's FIRST regulatory hurdle is **spectrum allocation**, not a safety
  standard: the 76-81 GHz automotive radar band is a specifically allocated, regulated RF spectrum
  band — in the US, FCC Part 95/15 rules (and dedicated automotive-radar allocations); in Europe,
  ETSI harmonized standards (e.g. the EN 303 396 / EN 302 264 family for automotive short-range radar)
  govern transmit power, bandwidth, and out-of-band emissions. A radar product cannot ship without
  clearing these RF emissions rules, entirely independent of any functional-safety case. FUNCTIONAL
  safety then follows the same paths named in
  [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) item 6 as any other AV/robotics
  perception sensor — ISO 26262 for automotive E/E systems, UL 4600 for a full autonomy safety case —
  this is orientation, not compliance guidance, and this repository's radar work is educational only.
- **Owning team:** perception (SYSTEM_DESIGN §5.1), often with a dedicated RF/radar-signal-processing
  sub-team distinct from the camera/LiDAR perception team (the underlying physics and hardware are
  different enough to warrant separate expertise); adjacent teams: electrical engineering (owns the RF
  front-end hardware this project's cube stands in for), sensor fusion/tracking (the direct downstream
  consumer), and regulatory/compliance (owns the spectrum-allocation clearance named above, a step
  most other repo domains do not face).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
