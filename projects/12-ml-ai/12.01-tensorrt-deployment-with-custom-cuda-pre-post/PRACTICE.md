# 12.01 — TensorRT deployment with custom CUDA pre/post kernels: NMS, argmax decode, keypoint extraction: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

This project's physical carrier is the **camera + inference-compute pairing** that sits at the front
of a robot's perception chain — a construction problem in two halves:

- **The camera module.** A rigid PCB with the image sensor die, a lens holder (M12/S-mount for
  industrial cameras, fixed-focus molded optics for consumer/embedded modules), and either a MIPI
  CSI-2 ribbon connector (short-run, board-to-board, the common choice inside a robot chassis where
  the camera sits centimeters from the compute board) or a GMSL/FPD-Link SerDes chip pair (for
  runs of a meter or more — automotive-grade, EMI-hardened, common on AMRs and AVs where the camera
  is on an arm or bumper far from the compute enclosure). Mounting matters mechanically: vibration
  on a legged or wheeled robot can blur frames or, worse, shift the lens's optical axis relative to
  its calibrated intrinsics over time — a camera bracket's stiffness is a perception-accuracy
  decision, not just a mechanical one.
- **The inference-compute enclosure.** The GPU/SoC that runs this project's kernels sits inside a
  sealed or vented enclosure sized for its THERMAL DESIGN POWER: a Jetson Orin module dissipates
  15–60 W depending on power mode, almost all of it through a heatsink (passive on small/quiet
  builds, a fan on higher power modes) — undersized cooling silently throttles the clock (and this
  project's kernel timings) long before it causes a hard failure, which is why thermal validation is
  its own EVT/DVT step (SYSTEM_DESIGN §5.2), not an afterthought. Connectors (power, the camera
  bus, Ethernet/CAN for the rest of the robot) need strain relief and, outdoors or on a field robot,
  IP-rated sealing; the enclosure itself needs EMI shielding if it sits near motor drives (24) or
  high-current power electronics (25) — a GPU's switching power supply is itself a modest EMI
  source. The GPU compute enclosure's general construction story is 33.01 PRACTICE §1's, unchanged;
  what THIS project adds is the camera side of the pairing.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-10. All parts named below are **illustrative examples, never
endorsements**; part numbers and prices go stale — verify current before relying on any of them.*

**Compute tiers for this exact workload** (a small CNN + custom NMS/decode kernels — this project's
shape, scaled up):

| Tier | Illustrative choice (2026) | Where TensorRT fits | Rough cost (module only) |
|---|---|---|---|
| Embedded/edge SoC | Jetson Orin Nano / Orin NX (GPU + DLA on one die) | TensorRT is NVIDIA's native path here; the DLA can run supported conv layers at a fraction of the GPU's power | ~US$200–600 |
| x86 + discrete GPU | Small-form-factor PC + RTX-class dGPU (reference machine: RTX 2080 SUPER) | Same TensorRT toolchain as this project's optional path; more power/cooling budget, more raw throughput | ~US$1–3k system |
| MCU-class (no GPU) | Cortex-M/R + a tiny NN accelerator (e.g., Arm Ethos-class) | TensorRT does not target this tier; frameworks like TFLite Micro or vendor SDKs do — OUT of this project's scope, named for completeness | ~US$5–50 |

**Camera hardware:** a global-shutter or rolling-shutter CMOS sensor module (industrial: Sony
IMX-class sensors in an MIPI or GMSL camera; hobby/research: a Raspberry Pi HQ Camera-class module
or a USB3 UVC camera for a quick bench setup); a lens matched to the required field of view and
working distance; for outdoor/field robots, an IP67-rated housing.

**Silicon this project's kernels are the software analogue of:** on a DLA-equipped SoC, the
conv/ReLU layers this project hand-writes could instead run on the fixed-function DLA block (via
TensorRT's DLA backend) while the post-processing kernels (NMS, decode, keypoint — genuinely custom
logic, not a standard conv/pool graph) still run on the CUDA cores — the SAME split this project's
software already makes between "the inference core" and "the custom kernels", now mapped onto two
physically distinct pieces of silicon on the same chip.

## 3. Installation & integration — putting it on a real robot

- **Where this code runs:** the SAME onboard GPU-class compute that runs the rest of the perception
  stack (SYSTEM_DESIGN §6.1's compute tier) — never a cloud round-trip for anything in the 30–60 Hz
  camera loop (the network latency alone would blow the 16–33 ms budget). This project's process
  would live as a perception NODE: subscribing a camera `Image` topic, publishing a detection-list
  message (a `Detection2DArray`-shaped topic in ROS 2 terms, mirroring this project's own
  `Detection` struct field-for-field) for the tracker/planner downstream.
- **OS / real-time constraints:** perception nodes at this rate are typically SOFT real-time on a
  general-purpose Linux (not the hard-deadline RT kernel the control loop needs — SYSTEM_DESIGN
  §1.2's division of labor): an occasional missed frame degrades tracking smoothness but does not,
  by itself, destabilize the robot the way a missed control tick does (08.01 PRACTICE §3 discusses
  the control-loop end of that same division).
- **ENGINE VERSIONING AND REBUILD REALITY — the operational headache this project's fallback-path
  design rule sidesteps, and any real TensorRT deployment must plan for.** A serialized TensorRT
  engine (THEORY.md "Where this sits in the real world") is NOT portable: it is tied to the specific
  TensorRT version, CUDA version, and GPU architecture it was built on (the builder's tactic
  selection is literally a per-GPU benchmark result baked into the file). Practically, this means:
  - Every fleet hardware revision (a new Jetson SKU, a driver/CUDA upgrade) needs its engines
    REBUILT, not just redeployed — a real CI/CD step, not a one-time cost.
  - A model update (new weights from the ML/data team) similarly requires a rebuild before it can
    ship, adding minutes-to-tens-of-minutes of build latency (the tactic search itself) to every
    release cycle — teams typically build engines in CI on representative target hardware (or an
    identical GPU in the cloud) rather than on the robot itself.
  - This project's fallback path — plain CUDA kernels, portable across any `sm_75+` GPU with no
    rebuild-per-target step — is the honest reason a from-scratch teaching project defaults to it:
    it is not just "simpler", it sidesteps a real fleet-operations problem that TensorRT deployments
    must solve deliberately (cache built engines per hardware/software revision, verify the cache
    key includes every input that can change the serialized bytes).
- **Bring-up = the testing ladder, rung by rung (CLAUDE.md §1):**
  1. *Offline/simulation* — this demo, plus running the pipeline against a broader synthetic or
     recorded image set before trusting any single scene.
  2. *HIL* — the compiled pipeline on the TARGET compute (Jetson-class or the fielded x86+dGPU),
     fed recorded camera data, checked against the same ground-truth-gate discipline this demo uses.
  3. *Bench, camera live* — the real camera streaming into the pipeline on a bench, output logged
     and reviewed, no downstream consumer acting on it yet.
  4. *On-robot, shadow mode* — the detector runs on the moving robot but its output only LOGS
     (never feeds the planner/control stack) until enough on-robot data has been reviewed.
  5. *On-robot, live* — only after shadow-mode review, and only feeding the tracker/planner with the
     safety monitor (31.x) watching, per this repo's standing sim-validated-only caveat.
- **Calibration:** camera intrinsics/extrinsics (needed to turn an image-space detection into a
  metric robot-frame position — out of THIS project's scope, which stops at image-pixel detections,
  but the very next step any real integration takes).
- **N/A here:** no fieldbus is implemented in this project (a real deployment's detection output
  would publish over the robot's internal network/DDS, not a fieldbus like CAN/EtherCAT — those
  carry actuator commands, not perception messages, per SYSTEM_DESIGN §6.1's bus table). Stated per
  contract.

## 4. Business & regulatory context

- **Who needs this capability:** any robotics company shipping a camera-based perception feature —
  warehouse AMRs (tote/obstacle/person detection), AV stacks (the maximal case), manipulation cells
  (bin-picking object detection), and increasingly agriculture/field robotics (30.x). The specific
  SKILL this project teaches — gluing custom logic to a vendor inference engine — is needed anywhere
  a team adopts TensorRT (or an equivalent) but the vendor's built-in post-processing does not match
  their exact model output shape, which is close to universal for anything beyond a stock,
  off-the-shelf detector.
- **The players:** NVIDIA (TensorRT, Triton, DeepStream — the ecosystem this project's optional path
  targets) is the dominant edge-inference vendor in robotics; competing/complementary paths include
  ONNX Runtime (framework-agnostic, multiple execution providers including TensorRT), OpenVINO
  (Intel-silicon-targeted), and vendor-specific NPUs' own SDKs (Qualcomm, Google Edge TPU) for
  non-NVIDIA silicon. Build-vs-buy (SYSTEM_DESIGN §5.3): the pre/post KERNELS are almost always
  built in-house (they encode the team's own model's output shape); the inference ENGINE itself is
  almost always bought/adopted (TensorRT, not a hand-rolled GEMM library) — this project's own
  architecture is a worked example of exactly that split.
- **MODEL GOVERNANCE — the orientation this project's "engine versioning" note (§3) leads into.**
  A deployed model is not a static artifact: it has a VERSION (which training run, which weights
  file — this project's `data/README.md` checksum table is the teaching-scale version of a real
  model registry entry), a BUILD (which engine, for which hardware/software revision — §3's rebuild
  reality), and a DEPLOYMENT record (which robots are running which version, and since when — needed
  to reproduce a field issue, and to roll back a bad model update safely). Fleet-scale robotics
  companies typically own this as an internal "model registry + fleet OTA" system, adjacent to (and
  informed by) the safety case for whatever the model's output feeds — SYSTEM_DESIGN §5.2's fleet-
  operations phase, where this kind of governance decides profitability as much as any algorithm
  does.
- **Cost of getting it wrong:** a perception regression that ships silently (a rebuilt engine that
  quietly performs worse on some object class, an INT8 calibration drift) degrades detection quality
  fleet-wide before anyone notices — the mitigations are process, not code: staged rollouts (a new
  model/engine version ships to a small fraction of the fleet first), shadow-mode comparison against
  the previous version (§3's bring-up ladder, applied to UPDATES not just first deployment), and
  automated regression gates resembling this project's own ground-truth gate, run against a held-out
  scene set before any release.
- **Regulatory:** wherever this project's output eventually feeds a safety-relevant decision (an AMR
  stopping for an obstacle, an AV's object list), the APPLICABLE standard is the one for the ROBOT
  TYPE, not for "the model" in isolation — SYSTEM_DESIGN §6.2's table: ISO 10218/ISO TS 15066 for
  industrial arms, ISO 13482 for service robots, ISO 26262/UL 4600 for AVs. All of them, in
  practice, expect a documented model governance and validation process resembling the paragraph
  above; none of them certify a neural network by itself — orientation only, not compliance advice.
- **Owning team:** **ML/perception deployment** (titles: deployment engineer, MLOps engineer,
  perception software engineer) — the team that takes a model from the ML/data team (SYSTEM_DESIGN
  §5.1) and makes it run in the robot's real-time and hardware budget; adjacent teams: perception
  (owns model accuracy and training data), embedded (owns the target compute and its OS image), and
  fleet operations (owns the rollout process and the model registry this section describes).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
