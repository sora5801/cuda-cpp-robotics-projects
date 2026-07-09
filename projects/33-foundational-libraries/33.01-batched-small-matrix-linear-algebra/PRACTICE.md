# 33.01 — Batched small-matrix linear algebra (3×3, 4×4, 6×6 — the robotics sizes): Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — this is a software-only foundational library, so sections 1–2 teach
> the *physical carrier* the code runs on, and section 3 says honestly which integration topics do
> not apply.

## 1. Building it — construction of the robot/part

No robot part is constructed for a linear-algebra library — the honest framing here is the
**physical carrier**: the compute module this code executes on and how *that* is built into a robot,
because those construction realities decide whether GPU code is even deployable on a machine.

- **Where the computer lives.** On an AMR or quadruped, the compute sits in a sealed electronics bay
  near the robot's center of mass; on a manipulator work cell, usually in the control cabinet beside
  the servo drives. Two very different worlds: the cabinet has mains power, forced air, and space
  for a full x86 tower with a discrete GPU; the mobile robot has a battery budget, a vibration
  spectrum, and maybe 2–4 liters for all electronics — which is why embedded GPU modules
  (Jetson-class system-on-modules on a carrier board) dominate mobile platforms.
- **Thermal construction.** A discrete RTX-class GPU dissipates 100–300 W; a Jetson-class module
  15–60 W. Mobile robots almost always cool passively or with modest fans: the module is clamped to
  a machined aluminum cold plate through a thermal-interface pad, and the chassis itself becomes the
  radiator. Under-provisioned thermal paths show up in the field as *silent clock throttling* — the
  batch that met its 2.5 ms estimator budget on the bench misses it at 45 °C ambient. (This is why
  BUILD_GUIDE-style timing lines always name the machine; clocks are physics, not constants.)
- **Vibration and mounting.** Compute boards on legged/off-road platforms are mounted on elastomer
  isolators; connectors are the failure point, so fielded robots use locking connectors (screw-lock
  coax, latching Molex/JST, M8/M12 circular connectors for anything leaving the enclosure) — a
  loose SO-DIMM or fan header from vibration is a classic field failure.
- **EMI proximity.** Motor drives switch tens of amps at tens of kHz a hand-span away from the
  compute bay. Construction countermeasures: grounded enclosure walls between power and logic
  sections, shielded/twisted power runs, and keeping high-speed signal cables (camera MIPI/GMSL,
  PCIe risers) short and away from the power stage. GPU compute itself is unaffected by clean
  design, but corrupted sensor links upstream of this library's inputs are a real failure mode.

What breaks in the field, ranked by frequency in practice: connectors (vibration), fans (dust,
bearing wear), thermal pads (dry-out → throttling), storage (write-heavy logging on eMMC/SD).

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-08. All parts named below are **illustrative examples, never endorsements**;
part numbers and prices go stale — verify current before relying on any of them.*

There are no sensors or actuators in this project's bill of materials — its "hardware interface" is
a memory bus. The relevant hardware question is **which compute tier runs batched-linalg robotics
code**, because the tier sets the batch sizes and rates you can promise (this demo needs ~90 MB of
device memory at default sizes — three 200,000×36-float buffers for the n=6 matmul stage plus solve
buffers — trivial for every tier below):

| Tier | Illustrative compute (2026) | Rough cost class | What it means for this library |
|---|---|---|---|
| Hobby / student | Jetson Orin Nano class module (~1024-core Ampere-class iGPU, 8 GB shared) | ~US$250 board | Batches of 10⁴–10⁵ small problems fit easily; shared CPU/GPU memory removes the H2D copy cost that dominates small batches |
| Research platform | Jetson AGX Orin class (~2048-core iGPU, 32–64 GB) or x86 + RTX 4000/5000-class dGPU | ~US$1–3k | The reference regime for this repo; the demo's numbers (RTX 2080 SUPER, sm_75) sit here |
| Industrial / fielded | Rugged fanless IPC + embedded RTX module, or industrial Jetson SKUs (extended temperature, locked firmware, long-availability guarantees) | ~US$3–10k+ | Same silicon families; you pay for temperature range, vibration rating, supply longevity, and certification support — not FLOPs |

Notes that matter to this project specifically: the CUDA floor in this repo is `sm_75`; every tier
above clears it. Discrete-GPU tiers pay PCIe transfer latency on every batch (visible in
`main.cu`'s H2D/D2H structure); integrated tiers (Jetson) can use zero-copy/unified memory instead —
which is exactly the concern of projects 32.01/32.06, not this one.

## 3. Installation & integration — putting it on a real robot

**Where it runs.** As a *library*, this code ships inside someone else's process: statically linked
(the honest default for robotics — no DLL-versioning surprises at 3 a.m.) or as a shared library in
an internal "robotics math" package. In a ROS 2 stack it would live as a plain C++ library target
that `rclcpp` component nodes link against — there is no "batched linalg node"; the estimator node,
the planner node, and the controller node each call it in-process. Its API surface is exactly the
two launchers in [`src/kernels.cuh`](src/kernels.cuh) writ large.

**OS and real-time constraints — the honest story.** Linux with `PREEMPT_RT` gives the *CPU* side
deterministic scheduling, but CUDA work submission and completion are **not hard-real-time**:
driver-level queuing, clock changes, and other contexts introduce jitter. Practice today: keep the
GPU *off* the safety-critical path (the ≥1 kHz loops stay on CPU/MCU), let the GPU serve the
100–400 Hz estimation and 10–50 Hz planning tiers with deadline monitoring, and bound worst-case
latency empirically. The frontier work to change this — CUDA Graphs for jitter-free fixed-rate
loops and persistent kernels — is exactly projects 32.02 and 32.03; read them as this section's
continuation.

**What honestly does not apply** (stated per contract rather than padded): there is no fieldbus to
command — N/A because the library touches no CAN-FD/EtherCAT devices, only memory. There is no
calibration or bring-up procedure — N/A because there is no sensor or actuator to calibrate. There
is no E-stop integration — N/A because nothing here can move; the safe-testing ladder (sim → HIL →
bench → tethered → free) applies to *consumers* of this library (08.01's cart-pole/quadrotor
controllers carry that caveat themselves). What *does* apply to a library: unit tests against an
oracle (this project's whole demo design), version pinning of the CUDA toolkit in the robot image,
and regression-testing numerical outputs when the toolkit or GPU changes — a driver update that
changes FMA codegen is a real integration event for fielded fleets.

## 4. Business & regulatory context

**Who needs this.** Everyone shipping GPU autonomy, whether they know it or not: manipulator and
humanoid companies (batched IK, whole-body control), quadruped/AMR vendors (sampling controllers,
estimator banks), simulation companies (10⁴ environments × small-matrix physics), surgical-robot
planners. Batched small-matrix work is the substrate under cuRobo-style motion planning and
Isaac-Gym-style training — the capabilities investors actually see demos of.

**The players.** NVIDIA's stack (cuBLAS/cuSOLVER batched, CUTLASS, Warp, cuRobo, Isaac) is the
commercial center of gravity; MAGMA is the academic reference; Eigen owns the CPU side; JAX/PyTorch
serve the research loop. The build-vs-buy call (SYSTEM_DESIGN item 5): use vendor libraries for
standard-shaped standalone batches; hand-roll (this project's skill) when the operation must be
*fused* inside a bigger kernel or when the vendor path doesn't exist on your target.

**Cost of getting it wrong.** A wrong solve inside a controller is a wrong torque command; inside an
estimator, a confidently wrong state. The failure is silent — which is why this project's NaN policy,
oracle testing, and tolerance discipline are not pedantry but the miniature version of what a
professional team runs in CI on every toolchain bump. Fleet-scale cost: a numerical regression from
a driver update, undetected, becomes a fleet-wide behavioral change overnight.

**Regulatory reality.** Libraries are not certified; **systems** are. Under the machinery/robot
standards (ISO 10218 for industrial arms, ISO 13482 for service robots — see the SYSTEM_DESIGN
item-6 orientation map), certification attaches to the integrated safety functions, and generic
software components enter as supporting items subject to the manufacturer's software lifecycle
duties; in medical contexts (IEC 62304) third-party code like this is treated as SOUP — "software of
unknown provenance" — requiring documented requirements, verification, and change tracking. The
practical consequence for a team: keep the math library's tests, versions, and known-anomaly list
audit-ready, and never let uncertified GPU code *be* the safety function (the safety chain stays on
certified controllers — SYSTEM_DESIGN item 6).

**Where the work lives.** Owning team: the autonomy-platform / robotics-math / simulation-and-tools
group (titles: robotics software engineer — controls; GPU/HPC engineer; simulation engineer).
Adjacent: controls (the consumer), QA/functional safety (the auditor), DevOps/fleet (the toolkit
pinning). It is classic *leverage* work: a three-person library team multiplies every downstream
team's throughput.

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
