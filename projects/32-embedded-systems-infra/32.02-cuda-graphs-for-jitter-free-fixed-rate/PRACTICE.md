# 32.02 — CUDA Graphs for jitter-free fixed-rate perception-control loops: Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> This is a systems/infrastructure project, not a sensor or actuator one — sections 1–2 teach the
> *physical carrier* (the real-time compute stack) this code would run inside, section 3 is where
> the project's actual subject matter (integration into a fixed-rate robot loop) lives, and section
> 4 covers who cares and why.

## 1. Building it — construction of the robot/part

No sensor, actuator, or mechanical part is constructed by this project — the honest physical carrier
is the **compute enclosure and its real-time host** the tick pipeline would run inside on a real
robot, which is 33.01 PRACTICE §1's story, unchanged: where the compute module sits in the robot
(sealed electronics bay near the center of mass on a mobile platform; control cabinet on a
manipulator work cell), how it is thermally managed (a discrete RTX-class GPU dissipates 100–300 W
and needs forced air or a substantial heatsink; a Jetson-class module at 15–60 W is more forgiving
but still throttles silently under poor thermal design — 33.01 PRACTICE §1's warning that "the batch
that met its budget on the bench misses it at 45 °C ambient" applies to this project's tick-timing
claims exactly as written), and how it is mounted/connected (vibration-isolated, locked connectors —
a marginal SO-DIMM or fan-header connection is a classic field failure that would show up first as
*exactly the kind of tail-latency spike this project's `[info]` lines report*, before it showed up as
anything more dramatic).

**What is specific to THIS project, beyond 33.01's story:** a fixed-rate loop additionally cares
about the **host CPU's** real-time behavior, not just the GPU's. On the reference machine this is a
general-purpose Windows desktop CPU with no real-time guarantees at all — the pacing clock's
`timeBeginPeriod(1)` request and hybrid sleep+spin loop (THEORY.md "Measurement methodology") are
software workarounds for hardware/OS that was never built to promise a 4 ms deadline. A real
fixed-rate robot loop's physical construction answer to this is either (a) an isolated CPU core with
IRQs and other processes steered away from it (`isolcpus`/`nohz_full` on Linux, or a dedicated
real-time coprocessor), or (b) accepting the soft-real-time budget this project measures and
building the *robot's* safety margins around it, never assuming zero jitter.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-10. All parts named below are **illustrative examples, never endorsements**;
part numbers and prices go stale — verify current before relying on any of them.*

The question this project's hardware section answers is narrower than most: **which compute tier
can actually run a graph-launched tick loop with the timing properties measured here, and what
changes the timing story per tier:**

| Tier | Illustrative compute (2026) | What changes for THIS project |
|---|---|---|
| Hobby / student | Jetson Orin Nano class SoM (~1024-core Ampere-class iGPU, 8 GB shared) | Unified memory removes the H2D/D2H copy pair from every tick (2 of mode A's 12 calls gone); no WDDM — Linux driver stack instead, different (typically lower) per-launch overhead floor; Nano-class Orin has no PREEMPT_RT out of the box, so host jitter is still a real concern |
| Research platform | Jetson AGX Orin class, or x86 + RTX 4000/5000-class dGPU (**reference machine: RTX 2080 SUPER, sm_75, Windows WDDM**) | The regime this project's numbers were measured in; an x86+dGPU workstation running Linux instead of Windows removes WDDM's scheduling layer even without a real-time kernel — THEORY.md's "Where this sits in the real world" names this as the most likely single change to the measured latency-regression finding |
| Industrial / fielded | Rugged fanless IPC + embedded RTX module or industrial Jetson SKU running Linux + PREEMPT_RT, or a workstation/datacenter GPU in **TCC mode** | TCC removes WDDM outright (only on GPUs that support it — not this project's consumer card); PREEMPT_RT gives the HOST side of `PacingClock` genuine hard-real-time scheduling, which is what would let the tail-latency numbers this project reports as `[info]`-only become a number worth *gating* on |

Notes specific to this project: no sensors, actuators, or comms transceivers appear in its BOM at
all — its only "hardware interface" is the CPU-GPU link (PCIe on a discrete-GPU tier, an internal
fabric on Jetson's unified-memory tier) and the host CPU's timer/scheduler hardware
(`QueryPerformanceCounter`'s backing hardware timer on Windows; the equivalent monotonic clock
source on Linux). The CUDA floor in this repo is `sm_75`; every tier above clears it.

## 3. Installation & integration — putting it on a real robot

**This is the section this project is actually about** — everything above exists to ground it.

- **Where this code would run.** The same GPU-hosting computer that runs the robot's
  perception/planning stack (SYSTEM_DESIGN §6.1's "GPU SoC / x86 + dGPU" compute tier) — never the
  kHz-class motor-control MCUs, which stay firmware-simple and deterministic by construction
  (SYSTEM_DESIGN §1.2: "the motor current loop remains MCU/silicon territory essentially
  everywhere"). This project's tick pipeline is exactly the shape of code that would live inside a
  planning/control ROS 2 node on that GPU host.
- **The ROS 2 node/topic shape.** A `ros2_control` hardware-interface-style controller plugin (the
  same shape 08.01 PRACTICE §3 describes) whose `update()` callback IS this project's `run_mode`
  loop body: read the latest state/sensor topics into the pinned staging buffers, `cudaGraphLaunch`
  (or the naive sequence, per this project's finding, probably not preferred), publish the command
  topic from `d_u_published`. The controller manager's own timer replaces this demo's `PacingClock`
  in a real deployment — but the SAME question this project answers ("how much host time does one
  tick's GPU submission cost, and does that reduction actually lower end-to-end latency") is exactly
  what an engineer integrating GPU work into a `ros2_control` loop needs measured, on THEIR
  hardware, before trusting it at THEIR chosen rate.
- **Real-time constraints, honestly extended from 08.01/33.01's story.** 33.01 PRACTICE §3 already
  states the baseline: "CUDA work submission and completion are not hard-real-time... keep the GPU
  off the safety-critical path... bound worst-case latency empirically... [this] is exactly projects
  32.02 and 32.03." This project is the empirical bounding 33.01 promised, and its own honest
  finding sharpens the warning: **do not assume CUDA Graphs make a loop hard-real-time just because
  they reduce mean submit time** — this project's own p99 latency numbers went the WRONG way for
  the graph modes on this platform. A real integration decision needs THIS project's methodology
  (measure submit AND latency AND tail, on the target hardware, before and after adopting graphs),
  not the marketing headline.
- **Bring-up = the testing ladder, rung by rung, specialized to a TIMING claim (CLAUDE.md §1):**
  1. *Simulation* — this demo, on the target compute hardware (not necessarily the target ROBOT),
     measuring the same three metrics this project measures, before any hardware is involved.
  2. *HIL* — the SAME tick pipeline (or the real controller it stands in for) running against a
     real-time-simulated plant on the actual target compute, with the actual target OS/RT
     configuration, so WDDM-vs-Linux-vs-TCC differences THEORY.md discusses are no longer
     theoretical for your specific deployment.
  3. *Bench, current-limited* — only once the timing budget is validated on real hardware does it
     make sense to let this loop's OUTPUT reach anything that can move — at that point 08.01
     PRACTICE §3's bench-then-tethered-then-free-running ladder applies to whatever controller this
     infrastructure is serving, with an independent deadline-miss monitor watching from the start.
  4. *Free running* — gated on the SAME envelope 08.01 PRACTICE §3 describes; this project's own
     scope never reaches here (README "Limitations": no plant is actually driven).
- **N/A here:** no calibration procedure — N/A because nothing sensed or actuated is calibrated by
  this code. No fieldbus integration — N/A because this project's "actuation" is a function call
  into an in-memory buffer, not a CAN-FD/EtherCAT transaction (SYSTEM_DESIGN item 6's territory,
  owned by whatever real controller sits downstream of this infrastructure).

## 4. Business & regulatory context

- **Who needs this capability.** Any robotics company pushing GPU compute toward tighter control
  loops than the classical 10–60 Hz perception/planning band: legged-robot companies (whole-body
  control at 0.5–1 kHz), high-speed manipulation and agile-drone companies, and any team whose
  perception/planning stack has grown expensive enough on CPU that "just add a GPU" is being
  proposed for a loop that was previously firmware-simple. The question this project trains an
  engineer to ask before that proposal ships: *does the GPU orchestration technique you plan to use
  actually deliver the latency property you need, on the hardware you'll actually field, measured —
  not assumed from a blog post?*
- **The players.** NVIDIA's own CUDA Graphs documentation and TensorRT/Triton (README "Prior art")
  are the production precedent for the SUBMIT-time claim; there is comparatively little public,
  vendor-neutral discussion of the LATENCY/tail-jitter caveat this project measured — which is
  itself a market gap a systems-minded robotics engineer can fill with exactly this kind of
  in-house measurement discipline. Real-time OS vendors (Wind River, QNX) and the PREEMPT_RT Linux
  community are the counterpart expertise for the HOST-side half of the story.
- **Cost of getting it wrong.** Believing "CUDA Graphs made my loop real-time" without measuring
  latency and tail (not just mean submit time) is exactly the kind of confidently-wrong engineering
  decision that costs a robotics company field incidents: a control loop that silently degrades
  under load because its GPU orchestration was optimized for the wrong metric. The mitigation is
  architectural, matching 08.01 PRACTICE §4's pattern: certified/deterministic layers at the
  actuation seat that do not trust this project's timing claims blindly, independent deadline-miss
  monitoring (31.x), and — the discipline this project itself models — measuring submit time,
  end-to-end latency, AND tail jitter as three separate numbers, on the target hardware, before a
  design decision ships.
- **Regulatory.** The same reality 33.01 PRACTICE §4 and 08.01 PRACTICE §4 state: GPU-hosted,
  WDDM/driver-scheduled compute is not what today's functional-safety certification regimes (ISO
  10218/13482 for robots, ISO 26262/UL 4600 for vehicles — SYSTEM_DESIGN item-6 orientation map)
  know how to bless as a timing-deterministic component. Where a real-time GUARANTEE (not just a
  good measured average) is required for certification, the actuation seat stays on a certified
  deterministic controller (an MCU running fixed-cycle firmware), with GPU-accelerated planning —
  captured graphs or not — feeding it from above, inside a monitored envelope, never holding the
  hard deadline itself. This project's own honest negative result (graphs did not reduce THIS
  platform's tail latency) is a concrete illustration of exactly why that architectural boundary
  exists, not an argument against ever using GPU compute in a tight loop.
- **Owning team.** Embedded/platform engineering (titles: embedded systems engineer, platform/
  real-time systems engineer, GPU systems engineer) owns this project's subject matter directly;
  adjacent teams: controls/autonomy (owns 08.01, the workload this infrastructure wraps, and would
  be the first to ask for these latency numbers before adopting graphs in their loop), and
  functional safety (owns the envelope around anything this infrastructure ultimately feeds,
  per SYSTEM_DESIGN item 5's org map).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
