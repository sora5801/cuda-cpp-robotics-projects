# 02.20 — LiDAR intensity calibration across channels: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

This project is a calibration ALGORITHM, not a physical part — but it exists entirely to characterize
one: the receive chain of a real spinning LiDAR head (THEORY.md "The problem"). Concretely, what is
being built and calibrated:

- **The rotating head.** Sixteen laser-diode/APD pairs are mounted on a common rotor, each firing
  down its own fixed elevation angle, spinning at a controlled rate (typically 5–20 Hz) to sweep
  azimuth. Assembly tolerances here — how precisely each transmit/receive pair is aligned to its
  nominal elevation and to each other — are the PHYSICAL root cause of the per-channel gain
  differences this project recovers: sub-millimeter misalignment of a receive lens relative to its
  APD's active area attenuates that one channel, permanently, from the day it left the factory.
- **The APD array and its bias network.** Each APD needs its own reverse-bias voltage, generated and
  regulated on a shared PCB; small resistor-tolerance and thermal-path differences across sixteen
  channels on one board are a second, independent source of per-channel gain spread (THEORY.md).
- **Sealing and thermal path.** The rotating head is typically sealed (IP67-class in outdoor-rated
  units) with a thermally conductive path from the APD array to the housing — APD gain is
  temperature-sensitive, so the housing's thermal design directly affects how much a calibration
  performed at one temperature drifts by the time the robot operates at another (§3 below).
- **What breaks in the field:** connector/seal degradation letting moisture reach the receive
  electronics (a sudden, large gain change on one or more channels — this project's tools would flag
  such a channel as an outlier against its own calibration history, not just recover A number for
  it); laser diode end-of-life power droop (a slow, monotonic single-channel gain decline over
  months); and physical shock (a dropped or collided unit) shifting receive-lens alignment abruptly.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-12. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

**Compute this calibration would run on:** it is a batch/offline computation over one or more
accumulated scans, not a hard-real-time perception kernel — it comfortably fits on the SAME compute
tier that runs the rest of perception (`SYSTEM_DESIGN.md` §6.1's "GPU SoC / x86+dGPU" tier: e.g. a
Jetson Orin-class module on a fielded robot, or an engineer's workstation GPU during bring-up/field
verification). Illustrative cost tiers:

| Tier | Example compute | Notes |
|------|------------------|-------|
| Hobby/research | Any desktop with a CUDA GPU (this project's own dev target: RTX 2080 SUPER class), or even CPU-only for a single calibration run (this project's whole pipeline finishes in well under a millisecond on GPU; a CPU-only run would still be milliseconds) | The workload is tiny; almost anything runs it |
| Fielded robot | Jetson Orin NX/AGX-class module already running perception | Calibration piggybacks on existing compute — no dedicated hardware needed |
| Fleet-scale | A cloud/on-prem batch job ingesting logged scans from many robots | Runs the SAME algorithm per-robot; the "hardware" here is fleet log storage/compute, not anything sensor-adjacent |

**The sensor itself** (illustrative, dated): a 16-channel mechanical spinning LiDAR in the
Velodyne VLP-16 / Ouster OS1-16 / Hesai XT16 class — research-grade units in this class are
commonly USD 1,000–4,000 (2026 street pricing, verify current); industrial/automotive-grade
higher-channel-count units with tighter factory intensity calibration cost substantially more. The
receive-chain silicon inside such a head (illustrative, not sourced from any specific vendor's BOM):
an APD array or discrete APDs per channel, a transimpedance amplifier (TIA) per channel, and an
ADC/time-to-digital converter shared or per-channel depending on architecture — none of this is
something an integrator touches directly; this project's calibration operates entirely on the
DIGITIZED intensity the driver already outputs, deliberately hardware-agnostic above the point-cloud
interface.

## 3. Installation & integration — putting it on a real robot

**Where this runs.** The SAME perception-tier compute already consuming the LiDAR's point-cloud
stream (`SYSTEM_DESIGN.md` §6.1) — no dedicated hardware, no real-time constraint. As a ROS 2 node
shape (`SYSTEM_DESIGN.md` §3.6): a `lidar_intensity_calibration` node subscribes to accumulated
`sensor_msgs/msg/PointCloud2` scans (ideally several sweeps' worth, or a short recorded drive, for
enough channel-overlap voxels — README "Limitations"), and publishes (or writes to a parameter/config
file) a 16-element `float64[]` gain table, consumed by a downstream `intensity_normalizer` node or
compositional filter that republishes a corrected `PointCloud2` for everything else (02.18's LIOR,
place-recognition, lane-detection nodes) to consume.

**Bus/interface it touches:** none directly — it is a software post-processing stage over whatever
transport already carries `PointCloud2` (typically Ethernet/DDS in a ROS 2 system); it never speaks to
the LiDAR's own configuration interface (usually a small Ethernet/UDP control-and-telemetry channel
the vendor SDK owns).

**Calibration/bring-up cadence — factory + periodic, not per-scan.** Unlike most perception nodes in
this repo (running at sensor rate, `SYSTEM_DESIGN.md` §1.1), this is explicitly NOT a per-frame
computation:

1. **Factory / integration-time baseline.** Run once when a sensor is first mounted and commissioned
   on a robot (or accept the vendor's shipped calibration table as the baseline and use this
   project's technique only to VERIFY it, not replace it — the more common real workflow).
2. **Periodic refresh.** Re-run on a cadence driven by the physical drift mechanisms in §1 (weeks to
   months, not hours) — triggered by calendar time, by accumulated operating hours, or by a
   monitoring signal (e.g. this project's own `consistency_improvement`-style metric trending upward
   on live fleet data, a genuine drift-detection use of the SAME math this project's gate performs for
   grading).
3. **Triggered re-check** after any event likely to shift channel gains: a physical shock/collision,
   a housing reseal after service, or a firmware/driver update that changes how intensity is reported.

**The 02.18 co-update discipline.** Because project 02.18's LIOR filter (and anything else
threshold-based on intensity) is directly sensitive to this calibration (README "System context"
quotes the measured -5.9pp recall cost of NOT having it), any real deployment must treat the intensity
gain table and LIOR's own threshold parameters as a linked pair: refreshing one without checking the
other reintroduces exactly the failure mode this project's `consistency_improvement` gate demonstrates
closing. A fleet operator's calibration-refresh procedure should re-validate (not necessarily re-tune)
downstream intensity-threshold parameters in the same maintenance window.

**The safe hardware-testing ladder** (CLAUDE.md §1's caveat applies in full — nothing here commands
motion, but the LADDER discipline still matters because a WRONG calibration silently degrades
downstream safety-relevant filters like LIOR): simulation (this project's synthetic scene) → replay
against LOGGED real scans from the target sensor (compare recovered gains run-to-run for stability
before trusting a change) → a bench/tethered vehicle with the corrected intensity feeding a
NON-safety-critical consumer first (visualization, logging) → only then feed a live intensity-
dependent filter that affects planning, with human oversight during the transition.

## 4. Business & regulatory context

**Who needs this.** Any fleet operator or integrator running a LiDAR-equipped robot in varied weather
or lighting (autonomous vehicles, outdoor AMRs, agricultural/field robots — `SYSTEM_DESIGN.md` §2.1,
§2.5) that relies on intensity for ANY downstream decision: weather filtering (02.18), lane/marking
detection, retroreflective-target detection, or intensity-augmented place recognition. Sensor
integrators reselling or private-labeling LiDAR units also need a field-verification technique
independent of the vendor's factory claims.

**Commercial and open-source players.** LiDAR vendors (Velodyne/Ouster — now merged, Hesai, Livox,
Cepton, Innoviz, and others) each ship their own factory intensity-calibration process and firmware
correction tables as part of the product; this is normally NOT something an integrator re-derives
unless verifying or working around a stale/missing table. On the open-source side, Autoware.Universe
and other production autonomy stacks assume calibrated intensity is available upstream (from the
driver) rather than performing intensity self-calibration themselves — this project's algorithm is
closer to research/field-diagnostic tooling than to a shipped, widely-adopted open-source component
today.

**Cost of getting it wrong.** A silently miscalibrated channel does not throw an error — it degrades
whatever trusts intensity, quietly. In the concrete case this project measures (README "System
context"), a real weather filter's recall drops by ~6 percentage points, meaning MORE unfiltered snow/
rain speckle reaches downstream planning — a real-weather-conditions ODD (Operational Design Domain)
regression, not a crash, but exactly the kind of slow degradation that erodes trust in an "all-weather"
capability claim and can trigger costly re-validation or a fleet-wide sensor audit once discovered.

**Regulatory path.** Intensity calibration itself is not directly named in any of `SYSTEM_DESIGN.md`
§6.2's standards, but it is a real input to systems that ARE: an autonomous-vehicle safety case under
**ISO 26262**/**UL 4600** that relies on LiDAR-intensity-based lane or retroreflective-target detection
must be able to show that detection performance is characterized ACROSS the sensor's service life,
including gain drift — this project's periodic-refresh discipline (§3) is the kind of evidence such a
case would cite. This section is **didactic orientation only — not procurement, legal, or compliance
advice** (`SYSTEM_DESIGN.md` §6.2's label applies in full).

**Where this lives inside a robotics company.** Owning team: **perception/calibration**
(`SYSTEM_DESIGN.md` §5.1) — commonly a small team or a rotating responsibility inside the broader
perception group, since sensor calibration (intensity here; extrinsics in project 02.16; camera
photometric calibration in project 01.09) recurs across every sensor a robot carries. Typical role
titles: perception engineer, sensor calibration engineer, or (at fleet scale) a fleet reliability/
perception-quality engineer who monitors calibration drift across the deployed fleet. Adjacent teams:
**fleet operations** (who would trigger and consume a periodic recalibration signal in production,
`SYSTEM_DESIGN.md` §5.1), **QA & functional safety** (who would set the acceptance criteria and
re-validation cadence for anything safety-relevant that consumes calibrated intensity), and
**embedded/firmware** (who own the sensor driver that ultimately applies whichever gain table is
current).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
