# 04.01 — Massive particle filter localization (10⁵–10⁶ particles, GPU likelihoods + resampling): Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

This project's physical carrier is the **localization sensor suite** on a mobile-robot chassis: the
range sensor (planar LiDAR, or a ring of sonar/ToF units — this project's 16-beam abstraction could
represent either) and the drivetrain that produces odometry.

- **The range sensor.** A scanning planar LiDAR (SICK, Hokuyo, RPLiDAR-class units) houses a laser
  diode and a photodetector behind a rotating mirror or a spinning head, timed by dedicated ranging
  electronics, all inside a sealed housing with an optical window. Construction realities that
  directly corrupt the sensor model this project implements: the optical window must stay clean —
  dust, condensation, or scratches attenuate or scatter returns and inflate range noise beyond the
  Gaussian σ this filter assumes; the mounting bracket must be **rigid** and its offset to the
  robot's body frame (`base_link`) precisely known, because every beam angle in
  [`src/kernels.cuh`](src/kernels.cuh)'s sensor model is implicitly "from the sensor's calibrated
  pose" — a loose or miscalibrated mount silently rotates or offsets every expected range, and the
  filter will confidently converge to the *wrong* pose rather than fail loudly.
- **The odometry source.** Incremental or absolute encoders on each drive wheel's motor shaft (or
  integrated into the servo drive), feeding a wheel odometry computation (wheel radius × angle,
  differenced with the wheelbase geometry for a differential-drive robot). Construction realities:
  encoder cable runs on a moving chassis fatigue at the strain-relief point over thousands of duty
  cycles; connector contacts corrode or loosen under vibration, both of which show up as odometry
  *dropouts* far outside the smooth Gaussian noise this filter's motion model assumes — a real
  deployment needs input sanity-checking this teaching core does not implement.
- **What breaks in the field.** LiDAR window fouling (gradual — degrades range accuracy before it
  fails outright), encoder connector fatigue (sudden — a step change in odometry quality), wheel
  wear changing the effective radius (slow drift the filter's motion-noise sigma is meant to
  absorb, up to a point), and — the construction-level failure every safety section in this repo
  eventually reaches — E-stop wiring bypassed "temporarily" during bring-up and never restored.

The GPU compute enclosure story (where the kernels in this project physically execute) is
33.01 PRACTICE §1's, unchanged — this project adds no new compute-hardware construction concerns
beyond what any GPU-hosted robot software does.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-09. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

| Piece | Illustrative choices (2026) | Role in this project's filter |
|---|---|---|
| Compute for predict/weight kernels | Jetson Orin-class SoC (embedded AMR) or x86 + entry/mid discrete GPU (reference machine here: RTX 2080 SUPER, sm_75) | Runs `pf_predict_kernel`/`pf_weight_kernel` — ~1–7 ms/scan at K=10⁵–10⁶ on the reference GPU |
| Range sensor | Hobby: RPLiDAR A1/A2-class 2D spinning LiDAR (~US$100–300, ~5.5–8 m range); research: Hokuyo URG-series (~US$1,000–2,000, better accuracy/rate); industrial/safety-rated: SICK microScan3/nanoScan3-class (~US$3,000–5,000+, safety-rated field zones, the class an AMR needs for the person-safety story, not just localization) | Produces the range fan `pf_weight_kernel` scores particles against |
| Wheel encoders | Incremental quadrature encoders on motor shafts (~US$10–50 each, hobby/research) or integrated into a servo drive's feedback (industrial, bundled cost) | Source of the noisy odometry twist `pf_predict_kernel` propagates |
| Optional IMU (fusion, not implemented here) | Hobby: MPU6050/BMI-class (~US$5–20); industrial: Xsens/VectorNav-class (~US$500–3,000+) | Would tighten the motion model between scans in a production stack (README §Limitations); this project uses odometry alone |
| Comms to the range sensor | USB or Ethernet (most LiDAR units); industrial units may offer a safety-rated fieldbus output | Delivers the raw scan to the compute tier |
| E-stop / safety chain | Certified relay or safety controller, dual-channel, independent of the compute tier | Must exist before ANY hardware experiment involving this filter's output — see §3 |

The compute cost dominates for K beyond ~10⁵ particles at this project's beam count; the sensor and
encoder line items are essentially fixed costs of "having a mobile robot at all," not specific to
running a particle filter versus any other localization method.

## 3. Installation & integration — putting it on a real robot

- **Process shape.** On a ROS 2 robot this filter runs as a node analogous to `nav2_amcl` inside
  the Nav2 lifecycle-node stack: subscribes a `sensor_msgs/LaserScan` (or a reduced range-fan
  derived from one) and an odometry `Twist`, holds a latched `nav_msgs/OccupancyGrid` (the map,
  loaded once — matching this project's `load_map`), and publishes a pose estimate
  (`geometry_msgs/PoseWithCovarianceStamped`) plus the `map → odom` transform that lets every other
  node keep consuming `odom`-frame data without caring that localization corrects it underneath.
  The demo's structure — persistent device buffers allocated once, one predict+weight kernel pair
  per scan, host-side normalize/estimate/resample — maps onto that node's per-scan callback almost
  line for line.
- **Real-time constraints, honestly.** At 10 Hz with a ~1–7 ms GPU call (this project's measured
  range), the filter has enormous headroom inside a 100 ms scan period — comfortably schedulable
  even as a **soft** real-time task on a general-purpose Linux scheduler; nothing in this filter
  needs PREEMPT_RT-grade guarantees the way a kHz whole-body controller does (SYSTEM_DESIGN §1.1's
  division of labor). What *does* need attention: a missed or late scan should not silently corrupt
  the pose estimate — a fielded node checks scan/odometry timestamps for staleness and, on a
  prolonged gap, degrades to "localization uncertain" rather than confidently reporting a pose it
  has not actually updated.
- **Bring-up = the testing ladder, rung by rung (CLAUDE.md §1).** This project's output is a pose
  *belief*, not a motion command — but a wrong belief is exactly what makes a downstream navigation
  stack drive somewhere wrong, so the ladder still applies in full:
  1. *Simulation* — this demo, plus stress cases: a much noisier scan/odometry sigma than the
     sample, a deliberately bad initial cloud (Exercise 4's global-localization variant), and a map
     with a symmetric corridor (the classic ambiguous-localization stress test).
  2. *HIL / recorded-data replay* — run the filter against a real robot's recorded scan/odometry
     bag (or a live feed) with its motion commands disconnected, so a bad estimate cannot yet steer
     anything; compare the estimate against a trusted reference (e.g., a motion-capture pose or a
     hand-surveyed checkpoint).
  3. *Bench / tethered* — wheels off the ground or the robot tethered at low speed in a small,
     cleared, already-mapped area; verify the LiDAR→base_link extrinsic calibration and the
     wheel-odometry scale/track-width calibration *first* — both are silent-failure modes this
     filter cannot self-diagnose (a miscalibrated extrinsic looks, to the filter, like extra sensor
     noise, not like an error).
  4. *Supervised low-speed operation* — the estimate now permitted to inform navigation, with an
     operator and E-stop present, in the mapped area only.
  5. *Free running* — only after the above, and only within whatever physical/software safety
     envelope (domain 31) does not depend on this filter's output being correct on every single
     scan.
- **Calibration this filter's correctness depends on:** the LiDAR-to-`base_link` extrinsic
  transform (a rigid offset — get it wrong and every ray-cast in `pf_weight_kernel`'s mental model
  is silently rotated/translated relative to reality), wheel radius and track width (the odometry
  scale factors that turn encoder ticks into the `(v, ω)` twist `pf_predict_kernel` consumes), and
  the map itself (built once via SLAM — domain 05 — and kept current; a stale map with moved
  shelving or walls corrupts the measurement model exactly as much as a sensor fault would).

## 4. Business & regulatory context

- **Who needs this.** Any indoor mobile robot that must know its pose against a known map without
  external infrastructure (no GPS indoors): warehouse/logistics AMRs (the dominant commercial
  volume — fleets of hundreds per site is routine), hospital and hotel service robots, cleaning
  robots, and mapped-facility inspection robots. It is one of the most commercially deployed
  capabilities this entire repository touches.
- **The players.** ROS 2 Nav2 (`nav2_amcl`) is the dominant open-source production implementation;
  commercial AMR makers (e.g., the warehouse-automation space broadly — Locus Robotics, 6 River
  Systems, MiR, Fetch/Zebra, Omron, and many others) run localization stacks descended from the same
  MCL family, often with proprietary tuning, sensor fusion, and fleet-scale map management layered
  on top; fleet-management platforms (their own commercial category) consume the pose output at
  scale across many robots and a site.
- **Cost of getting it wrong.** A localization failure does not usually look like a crash in
  software — it looks like a robot that confidently drives into a shelf, a wall, or a person because
  it *believes* it is somewhere it is not. Consequences scale from a damaged robot and a knocked-over
  shelf (a common, survivable failure) to line-stoppage downtime in a fulfillment center (real
  revenue cost) to a genuine safety incident if the mislocalized robot enters a zone it should have
  avoided. Because this filter's output silently underlies every downstream navigation decision,
  its failure modes are the classic "wrong but confident" kind that safety engineering specifically
  designs around (independent, localization-agnostic safety layers — bumpers, safety-rated LiDAR
  zones, geofencing — SYSTEM_DESIGN §6.1's safety chain).
- **Regulatory.** Per the SYSTEM_DESIGN §6.2 orientation table: warehouse/industrial AMRs sit under
  the same broad umbrella as other industrial mobile machinery (**ISO 3691-4** governs driverless
  industrial trucks specifically, alongside the more general **ISO 13482** for service robots
  sharing space with untrained people — both outside this repo's four-row didactic table but
  directly relevant, so named here honestly); none of this project's code is safety-certified, and
  a certified safety case for a real AMR treats localization as an *input* to independently-verified
  safety functions (E-stop chains, safety-rated LiDAR field zones), never as the safety mechanism
  itself. This is an orientation map, not compliance guidance — consult the SYSTEM_DESIGN §6.2 table
  and a qualified functional-safety engineer for anything approaching a real deployment.
- **Owning team.** Controls & autonomy, state-estimation sub-team (SYSTEM_DESIGN §5.1) — typical
  titles: robotics software engineer (localization/state estimation), autonomy engineer; adjacent
  teams: perception (owns the LiDAR driver, calibration, and any learned front-end), simulation
  (owns the map/sensor models this filter is tuned against and validated in), and QA/functional
  safety (owns the independent safety layers that do not trust this filter's output).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
