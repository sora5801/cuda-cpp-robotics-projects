# 02.06 — ICP: point-to-point → point-to-plane → GICP, all batched: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

This project's physical carrier is the **LiDAR sensor mount and its data path**, not an actuator — ICP
consumes what the sensor produces and hands a pose to software, so "construction" here means how the
sensor itself is built and attached, and what breaks between the photons and the point cloud this
project's kernels see.

- **Spinning mechanical LiDAR** (Velodyne/Ouster-class, and most research AMR/AV rigs historically): a
  rotating head carrying a vertical stack of laser/detector pairs, spun by a small brushless motor at
  5–20 Hz, with an optical/inductive slip ring or wireless link carrying data and power across the
  rotating joint (no wires can simply cross it — this is a genuine mechanical-design constraint, not an
  afterthought). Housed in a sealed dome (IP67-class is typical) to keep dust and moisture off the
  optics; the dome's own scratches, condensation, and dust ACCUMULATION are a real field-reliability
  failure mode (a dirty or scratched dome degrades return intensity and adds spurious short-range
  returns — exactly the kind of noise this project's synthetic `NOISE_SIGMA_M` stands in for, idealized).
- **Solid-state / MEMS / flash LiDAR** (increasingly common on production AMRs and automotive-grade
  units): no spinning mass — beam steering via a MEMS mirror, an optical phased array, or a fixed flash
  illuminator with a detector array. Mechanically simpler (no slip ring, no rotating-mass vibration,
  often lower profile) at the cost of a smaller field of view per unit (production designs tile several
  units to cover 360°) and a different noise/range-vs-angle characteristic than a spinning unit.
- **Mounting.** Rigid, vibration-isolated mounting matters MORE than it looks: any flex between the
  LiDAR and the robot's body frame between calibration time and run time is an uncorrected extrinsic-
  calibration error that ICP cannot distinguish from real motion — every point this project's kernels
  ingest is only as good as the assumption that the sensor-to-body transform is fixed and known.
- **What breaks in the field:** slip-ring wear and connector fatigue on spinning units (a rotating joint
  is a wear part by definition); dome scratches/fogging degrading intensity and range accuracy; loose
  mounts introducing timing-correlated vibration that looks like structured noise; cable strain at the
  sensor's connector on a moving robot (the classic "works on the bench, fails after a week of driving"
  failure). The GPU compute enclosure itself (where this project's kernels actually execute) follows the
  same construction story as every other domain-33-adjacent project — see 33.01 PRACTICE §1.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-09. All parts named below are **illustrative examples, never endorsements**;
part numbers and prices go stale — verify current before relying on any of them.*

| Piece | Illustrative choices (2026) | Role in this project's loop |
|---|---|---|
| Spinning mechanical LiDAR | Hobby/research: Velodyne Puck-class, ~US$4k–8k (legacy, largely superseded); mid-tier: Ouster OS0/OS1-class, ~US$4k–12k depending on channel count | The "target cloud" scan (or, on the next scan, the "source cloud") — 10–20 Hz `PointCloud` output |
| Solid-state / MEMS LiDAR | Automotive-grade: Hesai/RoboSense/Livox-class units, ~US$500–3k at automotive volumes; research MEMS units, ~US$1k–5k | Same role, smaller FoV per unit, often lower per-unit cost at volume |
| 2D safety-rated LiDAR | SICK/Hokuyo/Keyence-class, ~US$1k–3k | Common on AMRs for the CERTIFIED safety-stop function — a SEPARATE, safety-rated path, never the pose-estimation LiDAR this project models |
| Compute for the ICP loop | Jetson Orin class (embedded AMR) / x86 + discrete RTX (reference machine here: RTX 2080 SUPER) | The correspondence-search + reduction kernels, per scan |
| Real-time host | The same box's CPU (Linux, often PREEMPT_RT) | Loads scans, runs the closed loop, hands the pose to the state estimator |
| IMU (for real deployments, not this project) | MEMS IMU (research: ~US$50–500) to industrial/tactical grade (~US$1k–10k+) | Provides motion PREDICTION between scans and the correction 02.08's deskewing needs — this project's synthetic clouds have no distortion to correct, so no IMU model appears here |

The LiDAR-to-compute interface is typically **Ethernet** (UDP point-cloud packets, the near-universal
choice for spinning mechanical units) or a proprietary/automotive interface (e.g. Ethernet/AVB, or a
FPD-Link/GMSL-style serdes link for automotive-grade solid-state units) — SYSTEM_DESIGN item 6's comms-
bus map places this alongside the other sensor buses a robot's compute tier terminates.

## 3. Installation & integration — putting it on a real robot

- **Where this runs.** On a real AMR or AV, this loop runs as a **perception/localization node** on the
  main compute tier (the same box or a peer box to the GPU that would run 01/02/03's other perception
  kernels) — NOT on a safety-rated MCU (SYSTEM_DESIGN item 6's compute-tier split: pose estimation is a
  soft-real-time, GPU-suited job; the certified E-stop/safety-LiDAR path is separate hardware entirely).
- **ROS 2 node shape.** A registration node would SUBSCRIBE a `sensor_msgs/msg/PointCloud2` (this
  project's `PointCloud` struct is deliberately message-shaped to match — kernels.cuh cites
  `docs/SYSTEM_DESIGN.md` §3.6) at the LiDAR's native rate (10–20 Hz) and PUBLISH a relative pose —
  `geometry_msgs/msg/TransformStamped` (scan-to-scan) or feed a pose-graph/factor-graph backend directly
  (scan-to-map, 05.09's territory) — plus, ideally, a covariance/quality estimate the downstream filter
  can weight (this project's RMS-and-iteration-count diagnostics are the DIDACTIC stand-in for that; a
  real node would compute the Hessian-based covariance approximation `THEORY.md`'s `H` matrix already
  gives for free, at essentially zero extra cost — a natural next exercise beyond this project's scope).
- **Motion-distortion reality — the gap this project's synthetic data does not model.** A spinning
  LiDAR's single 360° "frame" is collected over ~50–100 ms while the robot keeps moving; every point in
  a raw sweep is technically valid at a SLIGHTLY different pose than every other point in the same
  sweep. Real ICP pipelines correct this (motion-distortion "deskewing," 02.08's job — usually via IMU-
  or constant-velocity-based per-point pose interpolation) BEFORE handing a cloud to registration, or the
  systematic distortion masquerades as noise ICP cannot distinguish from the real misalignment it is
  trying to solve for. This project's synthetic clouds are, by construction, distortion-free (every point
  in a cloud is generated from the SAME pose) — an idealization stated plainly here and in
  [`README.md`](README.md)'s Limitations, not hidden.
- **Calibration and bring-up.** Extrinsic calibration (the fixed LiDAR-to-robot-body transform — see §1)
  must be measured and loaded before ANY registration result means anything in the robot's own frame;
  intrinsic calibration (per-beam range/angle correction) is usually factory-provided but should be
  spot-checked. Bring-up follows the repo-wide testing ladder (CLAUDE.md §1): **simulation** (this
  project's synthetic pairs, or a full sensor sim like 11.01) → **recorded-data replay** (bag files of
  real scans, run through the SAME registration code, offline, no robot motion involved) → **HIL / bench,
  stationary** (a real sensor on a bench or a tripod, robot drive disabled, comparing registration output
  against a known/measured motion) → **tethered / low-speed** → **free running**. Because this project's
  OUTPUT is a pose (not a motor command), it does not itself command actuation — but a downstream
  controller acting on a WRONG pose absolutely can, so the full ladder still applies to the pipeline as a
  whole, and the CLAUDE.md §1 sim-validated-only caveat is repeated here for that reason.
- **N/A here:** no fieldbus is implemented in this project — a real deployment would receive the LiDAR's
  native packets over Ethernet/UDP and publish its result over ROS 2/DDS; this demo's "sensor" is two
  files on disk. Stated per contract.

## 4. Business & regulatory context

- **Who needs LiDAR registration:** every company shipping a LiDAR-equipped mobile robot needs SOME
  form of this capability — warehouse/logistics AMRs (localization against a site map), autonomous
  vehicles (localization against an HD map, and/or LiDAR-based odometry as one leg of a redundant state
  estimate), and increasingly legged/field robots carrying a spinning or solid-state LiDAR for the same
  reason. It is squarely a **perception** deliverable handed to **controls & autonomy**
  (`docs/SYSTEM_DESIGN.md` §5.1's org map — this project's own README System-context section names both
  teams explicitly).
- **The players:** PCL and Open3D as the open-source incumbents nearly every team starts from or
  benchmarks against; KISS-ICP and FAST-GICP as the more recent, research-grade-but-widely-adopted
  LiDAR-odometry-focused implementations; and every AMR/AV company's internal perception/SLAM team as
  the ultimate owner of a production registration stack (build-vs-buy, `docs/SYSTEM_DESIGN.md` §5.3: the
  correspondence-search/reduction CORE is commodity enough to adopt from PCL/Open3D; the tuning, robust-
  kernel weighting, and integration into a company's specific state estimator is usually where the real
  engineering — and the differentiation — happens).
- **Cost of getting it wrong:** a registration bug is a SILENT failure mode — ICP can converge cleanly
  to a WRONG local optimum (a classic risk on repetitive/symmetric geometry, e.g. a long straight
  corridor with few cross-features) and hand a confident, plausible-looking, incorrect pose to
  everything downstream. Depending on what trusts that pose, the consequence ranges from a lost/confused
  robot (an AMR that stops and asks for help) to, in a more automated deployment, a collision — which is
  exactly why production stacks pair registration with independent sanity checks (residual/fit-quality
  thresholds, cross-checking against wheel odometry or an IMU-propagated prior, and a safety monitor
  watching the STATE the pose feeds — 31.x's territory) rather than trusting a single registration result
  outright.
- **Regulatory:** LiDAR-based localization is a COMPONENT inside a larger autonomy stack's safety case,
  not itself directly named by a standard. For a warehouse AMR, the relevant path runs through
  **ISO 13482** (service-robot hazard analysis, `docs/SYSTEM_DESIGN.md` §6.2); for an autonomous vehicle,
  through **ISO 26262** / **UL 4600** (functional safety and the full autonomy safety case, same table).
  In both cases, a mislocalized robot is exactly the kind of failure mode a §6.2-style safety case must
  enumerate and mitigate — this project computes the perception/localization COMPONENT the safety case
  would reason about, and makes no certification claim of its own (CLAUDE.md §1).
- **Owning team:** perception (owns the correspondence search, normal estimation, and the registration
  code itself) handing its output to controls/autonomy / SLAM (owns the pose graph or filter that
  CONSUMES the pose) — `docs/SYSTEM_DESIGN.md` §5.1's org map; adjacent teams include simulation (owns
  11.01-style sensor sims used to generate more realistic test data than this project's structured room)
  and QA/functional safety (owns the residual/fit-quality monitoring that catches a silently-wrong
  registration before it reaches a controller).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
