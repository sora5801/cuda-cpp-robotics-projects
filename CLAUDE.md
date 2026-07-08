# CLAUDE.md — Repository Contract & Working Agreement

> This file is auto-loaded by Claude Code at the start of every session. It is the **single source of
> truth** for how work happens in this repository. Read it fully before doing anything. When in doubt,
> follow this document over any habit or default. If a rule here ever blocks progress, stop and surface
> the conflict in a `push-note` rather than silently working around it.
>
> This repository is the **sibling of `cuda-cpp-healthcare-projects`** and deliberately mirrors its
> conventions, tooling, and workflow. Where this document is silent, that repo's precedent applies.

---

## 1. Mission & philosophy

This repository, **`cuda-cpp-robotics-projects`**, is a **didactic study collection**: ~420 self-contained
CUDA C++ projects spanning all of robotics — perception (camera/LiDAR/radar/sonar/event/tactile), sensor
fusion & state estimation, SLAM & mapping, motion planning, collision & geometry, control systems,
dynamics & kinematics, physics & sensor simulation, machine learning for robots, every locomotion mode
(legged, wheeled, aerial, marine, space, and exotic), manipulation & grasping, human-robot interaction,
swarms, navigation, actuators & motors, power & energy, mechanical design & structures, materials &
manufacturing, soft robotics, medical robotics, field robotics, safety & verification, embedded/Jetson
systems, foundational GPU libraries, research frontiers, micro/nano robotics, and modular robots.

The reader (the repository owner) is using this as **study material** — not production code, but a
starting point on the path toward real systems. That single fact drives every decision:

- **Teaching beats cleverness.** A slower kernel that a learner can follow is better than a fast one they
  cannot. When forced to choose, choose the version that teaches, and *explain the faster version in
  comments*.
- **Every artifact must explain itself.** Code, build files, data scripts, and demos are all teaching
  surfaces. A file with sparse comments is considered **unfinished**, no matter how well it runs.
- **Nothing is a black box.** If a project uses a library kernel (cuBLAS, cuFFT, Thrust…), the surrounding
  comments must explain *what that call computes, why it is used here, and what it would take to write by
  hand*.
- **Every project knows its place in a robot — and in the whole.** Robotics is a systems discipline. Each
  project must explain where it sits in a full robot architecture — what feeds it, what consumes it, at
  what rate it must run — and where it sits in the wider whole: the physical machine, the hardware it
  runs on, and the robotics company that would ship it (see §3.1, §4.3, README section 4). A kernel
  without system context is half-taught.
- **Teach the whole robot, not just the kernel.** The didactic surface extends beyond code and math into
  the physics and engineering of the machine, how the relevant part is physically constructed, the actual
  hardware (chips, sensors, drivers) the code would run on and talk to, how it would be installed on and
  work with a real robot, and the business and regulatory reality around it. Each project carries a
  `PRACTICE.md` for exactly this (§4.3).
- **Reproducibility is sacred.** Anyone should be able to clone, open the Visual Studio solution, build,
  run the demo, and see the documented result — on a normal Windows + NVIDIA machine, with no robot
  hardware attached.

> **Not for deployment.** Everything here is educational. Nothing is safety-certified (no ISO 10218,
> ISO 13482, ISO/TS 15066, or ISO 26262 compliance is claimed — even projects that *compute* metrics from
> those standards do so didactically). Control code is validated in simulation only. Running any of it on
> a physical robot is the owner's decision and responsibility, and READMEs must say so wherever motion of
> real hardware is conceivable. No project may be framed for weaponization or for surveillance of
> individuals; person-perception projects are framed around collaborative safety.

---

## 2. Source-of-truth inputs

One file at the repo root defines *what* to build. It was generated before this repo existed and must be
treated as a read-only reference (do not rewrite it; if something in it is wrong, note it in a push-note):

| File | Role |
|------|------|
| `cuda-cpp-robotics-projects.md` | The **catalog**. 36 numbered sections (`## 1.` … `## 36.`), each a domain. Every top-level `- ` bullet inside a numbered section is **one project**. `★` marks beginner-friendly entry points; `[R&D]` marks research-stage topics. The intro paragraph, the Legend, and the trailing `## Where to Start` section are prose, **not** projects. |

**Catalog parsing rules (implemented once in `tools/catalog.py`, output `catalog.json`):**

- Sections: `## <n>. <Title>` → domain number + name → domain slug (see §3). Only numbered sections count.
- Projects: each top-level bullet in a numbered section, in order → ID `SS.NN` (section and ordinal both
  zero-padded to two digits: 3rd bullet of section 8 → `08.03`). IDs are **deterministic by position** —
  never renumber, never sort.
- Tags: leading `★` → difficulty `beginner`; trailing/inline `[R&D]` → difficulty `research`; otherwise
  `intermediate`. Strip tags from the project name.
- **Bundled bullets stay one project.** Some bullets list several related ideas (separated by `·`, `;`, or
  commas — e.g., the agriculture bundle in §30, the event-camera bullet in §3, the artificial-muscle
  bullet in §24). Do **not** split them: the bullet is one project whose named components become
  **milestones / sub-demos** inside it. The project README must state which components are implemented
  and which are documented-only, and `THEORY.md` must cover the ideas shared across the bundle.
- Names → folder slugs: lowercase, ASCII, spaces and `/`→`-`, drop punctuation, trim to something readable
  (`scaffold.py` does this deterministically — always use it rather than inventing names).

**How each catalog bullet maps into its project:**

- Bullet text + ID → folder name, README title, and one-paragraph summary.
- The methods named in the bullet (e.g., "SGM", "MPPI", "jump flooding", "Featherstone ABA") → the
  algorithms `THEORY.md` must explain and the code must implement.
- The tag → difficulty badge and build priority (§11). `[R&D]` projects may ship as **reduced-scope
  teaching versions** with the full research version described in `THEORY.md` under "Where this sits in
  the real world".
- The section → the domain folder, and the default placement in the autonomy stack (§3.1).

Treat `catalog.json` as the machine-readable catalog; agents never re-parse the markdown by hand.

---

## 3. Repository layout

```
cuda-cpp-robotics-projects/
├── CLAUDE.md                        # this file (the contract)
├── README.md                        # front door: what the repo is, how to use it, domain index
├── LICENSE                          # MIT (code) — see §8 for data licensing
├── .gitignore                       # ignores build artifacts, large data, secrets
├── CHANGELOG.md                     # concise index of every push, links into /push-notes
├── cuda-cpp-robotics-projects.md    # the catalog (source of truth, read-only)
├── catalog.json                     # generated machine-readable catalog
├── docs/
│   ├── COMMENTING_STANDARD.md       # the full commenting rubric (canonical copy)
│   ├── BUILD_GUIDE.md               # installing CUDA + VS, building any project
│   ├── SYSTEM_DESIGN.md             # the robot-architecture reference every project plugs into (§3.1)
│   ├── PROJECT_TEMPLATE/            # the canonical empty project (copied to start each one)
│   └── STATUS.md                    # generated dashboard: per-project todo/in-progress/done
├── push-notes/                      # one didactic note per push (§7.1)
├── tools/
│   ├── catalog.py                   # catalog .md -> catalog.json (parsing rules in §2)
│   ├── scaffold.py                  # catalog.json -> all project skeletons
│   ├── verify_project.py            # checks a project meets the Definition of Done
│   ├── status.py                    # (re)generates docs/STATUS.md work-queue dashboard
│   └── new_pushnote.py              # generates a dated push-note stub
├── showcase/                        # top-level demo that ties everything together (§6.3)
│   ├── showcase.sln
│   └── ...
└── projects/
    ├── 01-perception-cameras-vision/
    │   ├── 01.01-gpu-image-pipeline/
    │   ├── 01.02-stereo-depth-sgm/
    │   └── ...
    ├── 02-perception-lidar-point-clouds/
    └── ... (all 36 domain folders below)
```

### Domain folder slugs (section number → slug)

```
01-perception-cameras-vision      13-locomotion-legged            25-power-energy
02-perception-lidar-point-clouds  14-locomotion-wheeled           26-mechanical-design-structures
03-perception-radar-sonar-event   15-locomotion-aerial            27-materials-manufacturing
04-sensor-fusion-state-estimation 16-locomotion-marine            28-soft-robotics
05-slam-mapping-localization      17-locomotion-space             29-medical-bio-robotics
06-motion-planning                18-locomotion-other             30-field-robotics
07-collision-geometry             19-manipulation-grasping        31-safety-verification
08-control-systems                20-tactile-force-sensing        32-embedded-systems-infra
09-dynamics-kinematics            21-hri-teleoperation            33-foundational-libraries
10-physics-simulation             22-multi-robot-swarms           34-theory-frontier
11-sensor-sim-digital-twins       23-navigation-stack             35-micro-nano-robotics
12-ml-ai                          24-actuators-motors             36-modular-reconfigurable
```

Project folders: `projects/<domain-slug>/<SS.NN>-<project-slug>/`, generated by `tools/scaffold.py`.

### 3.1 `docs/SYSTEM_DESIGN.md` — the robot every project plugs into (load-bearing)

The owner asked for system design: how each project fits into a larger robotic system. This document is
written during Phase 0 and is the **shared architecture reference** that every project README cites. It
must contain:

1. **The canonical autonomy stack** — a Mermaid/ASCII layer diagram plus prose:

```
   [Sensors] → [Perception] → [State estimation / World model] → [Prediction]
        → [Planning: global → local] → [Control] → [Actuation]
   Cross-cutting: [Simulation & digital twin] [Learning] [Safety monitor]
                  [Infrastructure: compute, comms, power, mechanical structure]
```

   with typical data rates and latency budgets per boundary (e.g., camera 30–60 Hz, LiDAR 10–20 Hz,
   state estimator 100–400 Hz, local planner 10–50 Hz, whole-body control 0.5–1 kHz, motor current loops
   10–20 kHz) and where the GPU classically sits (perception/mapping/planning/sim) vs. where it is the
   research frontier (kHz control, safety monitors).

2. **Five reference robots**, each with a block diagram mapping repo domains onto its blocks:
   a warehouse AMR (02/04/05/23/06/08/25/31/32), a 6-DoF manipulator work cell (01/19/09/06/07/08/21/24),
   a quadruped (13/04/05/10/12/24/25), a quadrotor (15/04/08/11/22), and an autonomous-vehicle stack
   (01/02/03/04/05/06/14/31/32). These are *suggested system designs* the learner can grow toward.

3. **Interface conventions** shared by all projects so they compose conceptually: SI units; right-handed
   frames with named conventions (`T_parent_child` transform notation, quaternion order documented);
   monotonic timestamps in seconds (double); message-shaped structs that deliberately resemble ROS 2
   types (`PointCloud`, `Image`, `Twist`, `JointState`) so the mapping to a real middleware is obvious.

4. **A composition map**: worked examples of chains through the repo (e.g., 11.01 LiDAR sim → 02.06 ICP →
   05.01 TSDF → 07.09 distance field → 06.05 STOMP → 08.01 MPPI, with 31.01 reachability watching), so a
   learner sees how study projects become a robot.

5. **The whole: anatomy of a robotics company** — where work like this lives commercially: a typical org
   map (mechanical, electrical, embedded, perception, controls/autonomy, ML/data, simulation & tools,
   manufacturing & supply chain, QA & functional safety, fleet operations, product, regulatory/compliance,
   sales & support) with the repo domains each team would own; the product lifecycle (concept → prototype
   → EVT/DVT/PVT → production → fleet operations) and where each kind of project matters in it;
   build-vs-buy judgment calls; and unit-economics basics (BOM cost, margin, robots-as-a-service vs.
   capital sale). Didactic sketches, not business advice.

6. **Robot internals & the regulatory map** — a generic "inside a real robot" hardware architecture
   diagram: compute tier (GPU SoC / x86 + dGPU / MCUs / safety controller), sensor suite, actuation chain
   (MCU → gate driver → power stage → motor → gearbox → encoder), power tree (battery → BMS → DC/DC
   rails), comms buses (CAN-FD, EtherCAT, Ethernet/TSN), and the safety chain (E-stop, watchdogs,
   redundant monitors) — with repo domains mapped onto it; plus a regulatory landscape overview by robot
   type (industrial arms: ISO 10218 / ISO/TS 15066; service robots: ISO 13482; AVs: ISO 26262 / UL 4600;
   medical: IEC 60601 / FDA pathways; drones: FAA Part 107 / EASA; marine: COLREGs; space and anything
   defense-adjacent: export controls). Clearly labeled an orientation map, not compliance guidance.

Every project README **must** contain a "System context" section (§4.1 item 4) that quotes its position
in this stack: upstream inputs, downstream consumers, the rate/latency budget it would face on a real
robot, and which reference robot(s) it belongs to. `verify_project.py` checks the section exists. Each
project's `PRACTICE.md` (§4.3) then grounds that position in the physical and commercial whole —
construction, hardware, installation, business, regulation — citing items 5–6 here.

---

## 4. The standard project layout (Definition of "a project exists")

Every project folder MUST contain exactly this structure. `docs/PROJECT_TEMPLATE/` is the canonical copy;
`scaffold.py` stamps it out with catalog fields pre-filled.

```
<SS.NN>-<slug>/
├── README.md            # the learner's entry point (see §4.1)
├── THEORY.md            # the deep didactic explanation (the "why") (see §4.2)
├── PRACTICE.md          # the physical/commercial companion (the "with what") (see §4.3)
├── src/                 # implementation: .cu / .cpp / .cuh / .h — maximally commented
│   ├── main.cu          # entry point: parses args, loads data, runs, prints/saves result
│   ├── kernels.cu       # the GPU kernels (one teaching-focused kernel per concept)
│   ├── kernels.cuh      # kernel declarations + extensive header comments
│   ├── reference_cpu.cpp# a plain-C++ reference implementation used to VERIFY the GPU result
│   └── util/            # timing, CUDA_CHECK error macros, I/O helpers (copied from template)
├── data/
│   ├── sample/          # a TINY committed sample so the demo runs with zero downloads
│   └── README.md        # provenance, license, size, checksum, what each field means
├── scripts/
│   ├── make_synthetic.py         # generate synthetic data — the DEFAULT for robotics (§8)
│   └── download_data.ps1 / .sh   # fetch a public dataset where one exists and its license allows
├── demo/
│   ├── run_demo.ps1 / .sh        # one command: build (if needed) + run on sample + show result
│   ├── expected_output.txt       # what the learner should see (used by verify)
│   └── README.md                 # what the demo demonstrates, annotated
├── build/
│   ├── <slug>.sln                # Visual Studio 2026 solution (v145 toolset)
│   ├── <slug>.vcxproj            # CUDA project (see §5)
│   └── <slug>.vcxproj.filters
├── CMakeLists.txt       # OPTIONAL cross-platform build (nice-to-have; VS is the required one)
└── .gitignore           # ignores x64/, *.obj, *.exe, downloaded data, etc.
```

**Self-containment rule (robotics-specific).** Projects may *conceptually* chain (§3.1), and a perception
project may use data produced by a sensor-sim project — but every project stays individually buildable and
runnable: copy the tiny generated sample **into** `data/sample/`, never reference another project's folder
at build or run time. Shared `src/util/` code is copied from the template, not symlinked — deliberate,
documented duplication for didactic independence.

> A project is **not done** until *all* of the above exist, the VS build succeeds, the demo runs and
> matches `expected_output.txt` (within documented tolerance), and the comment density passes
> `verify_project.py`. See §9.

### 4.1 `README.md` (per project) — required sections

1. **Title** — `# <SS.NN> — <Project name>` and the difficulty badge (★ beginner / intermediate / [R&D]).
2. **One-paragraph summary** — what it does, in plain language.
3. **What this computes & why the GPU helps** — name the bottleneck that is parallelized and the
   parallelization pattern (map / reduce / stencil / scan / batched-solve / sampling).
4. **System context — where this sits in a robot** — position in the §3.1 stack; upstream inputs and
   downstream consumers (named as message-shaped interfaces); realistic rate/latency budget; which
   reference robot(s) use it; what would replace or surround it in production; one line on where this
   work lives in a robotics company (owning team). Link `docs/SYSTEM_DESIGN.md` and `PRACTICE.md`.
5. **The algorithm in brief** — bullet list of the key algorithms (link to `THEORY.md` for depth).
6. **Build** — exact steps (open `build/<slug>.sln` in VS 2026, select `Release|x64`, Build). Link
   `docs/BUILD_GUIDE.md`. List any optional dependency and its fallback (§5).
7. **Run the demo** — the single command in `demo/`.
8. **Data** — what the sample is (synthetic vs. public), how to regenerate or download, licensing.
9. **Expected output** — what success looks like; how the GPU result is checked against the CPU reference.
10. **Code tour** — a short guided reading order through `src/` ("start in `main.cu`, then `kernels.cu`…").
11. **Prior art & further reading** — the real tools/papers this teaches toward (e.g., PCL, OpenCV CUDA,
    nvblox, cuRobo, Isaac Gym/Lab, GTSAM, OMPL, Drake, MuJoCo, PX4, Nav2, MoveIt, SOFA), one line each on
    what to learn from them. Study these; do **not** copy code wholesale — reimplement didactically and credit.
12. **Exercises** — 3–5 "try this next" extensions for the learner.
13. **Limitations & honesty** — what is simplified, what is synthetic, what would differ in production,
    and (where relevant) the real-hardware safety caveat from §1.

### 4.2 `THEORY.md` (per project) — the deep dive

This is where the teaching lives. Expected contents:

- **The problem — physics & engineering first**: the physical phenomenon and robotics task being solved,
  taught from first principles wherever possible (mechanics, dynamics, electromagnetism, optics,
  acoustics, thermodynamics, materials — whatever governs this project), plus the engineering constraints
  a real robot imposes: noise floors, tolerances, bandwidth, latency, thermal limits, vibration, EMI,
  wear. Go as deep into the physics as the project allows; where a project is purely computational, say
  so and teach the physics of its nearest physical carrier.
- **The math**: governing equations / formal problem statement, with notation defined (units, frames).
- **The algorithm**: step-by-step, with complexity analysis (serial vs. parallel).
- **The GPU mapping**: how the algorithm becomes threads/blocks/grids; memory hierarchy used (global /
  shared / registers / constant / texture) and *why*; occupancy and bandwidth considerations; which CUDA
  library does what.
- **Numerical considerations**: precision (FP32/FP64), stability, race conditions, atomics, determinism —
  and robotics-specific ones: angle wrapping, quaternion normalization drift, stiff ODEs, ill-conditioned
  Jacobians near singularities.
- **How we verify correctness**: the CPU reference, the tolerance, edge cases.
- **Where this sits in the real world**: how production robotics stacks (named in README §11) do it
  differently; for `[R&D]` projects, what the open problems are and what the full version would need.

Write `THEORY.md` as if explaining to a sharp student who knows C++ but is new to CUDA and new to
robotics. Diagrams in Mermaid/ASCII are welcome.

### 4.3 `PRACTICE.md` (per project) — from code to physical robot

`THEORY.md` teaches the math and the GPU; `PRACTICE.md` teaches the machine and the world around it.
Four required sections. Where a topic truly cannot apply, write "N/A because …" honestly — never pad,
never fabricate:

1. **Building it — construction of the robot/part.** The mechanical and electrical construction of the
   subsystem this project belongs to (if possible): how it is physically built, assembled, and
   manufactured; materials and tolerances; mounting, wiring, connectors, sealing, shielding; what breaks
   in the field and why. For abstract/software-only projects, describe the physical carrier the code
   would serve and its construction instead.
2. **Real hardware — chips, parts, illustrative BOM.** The actual hardware this would run on and talk to
   (if possible): compute (e.g., Jetson Orin class vs. x86 + discrete RTX vs. MCU-class), the sensors and
   their interfaces, the actuation chain's silicon (motor-control MCUs, gate drivers, current-sense amps,
   encoder ICs), comms transceivers, power parts (BMS, DC/DC). Offer hobby / research / industrial-grade
   alternatives and rough cost tiers. All parts are **illustrative examples, never endorsements**; part
   numbers and prices go stale — date the section and say "verify current".
3. **Installation & integration — putting it on a real robot.** Where this code would physically run
   (which computer on which robot); OS and real-time constraints; the ROS 2 node/topic shape it would
   take; which bus it consumes or commands (CAN-FD, EtherCAT, Ethernet); sensor/actuator calibration and
   bring-up procedure; and the safe hardware-testing ladder — simulation → HIL → bench jig / tethered /
   current-limited → free running — with E-stop and limits at every rung (§1 caveat applies).
4. **Business & regulatory context.** Who needs this capability, in which products and markets; the main
   commercial and open-source players; what getting it wrong costs (downtime, recalls, liability); the
   applicable standards/regulatory path for this domain (cite the §3.1 item-6 map); and where the work
   lives inside a robotics company — owning team, typical role titles, adjacent teams (§3.1 item 5).
   Label it didactic orientation — **not** procurement, legal, or compliance advice.

`docs/PROJECT_TEMPLATE/` carries the stub; `verify_project.py` checks the file and its four sections
exist. Depth scales with relevance — an actuator project goes deep on 1–2, a planner on 3–4 — but every
section is genuinely written or honestly N/A'd.

---

## 5. Build standard (CUDA + Visual Studio)

> **Toolchain (inherited, already ratified).** The owner's machine runs **CUDA Toolkit 13.3 + Visual
> Studio 2026 (Community, v145 toolset)**, ratified and verified working in the sibling
> `cuda-cpp-healthcare-projects` repo (`nvcc` compiles for the local GPU's `sm_75`; the `CUDA 13.3`
> MSBuild integration is installed under `v180\BuildCustomizations`). This repo adopts the same standard.
> CUDA 13 dropped Maxwell/Pascal/Volta; `sm_75` (Turing) is the floor.

**Target toolchain (decided for this repo):**

- **Visual Studio 2026** (Community; `v145` platform toolset) with *Desktop development with C++*.
- **CUDA Toolkit 13.3** with the VS integration (`CUDA 13.3.props/.targets`). Each `.vcxproj` imports
  them; to retarget another CUDA version, change those two filenames.
- **Multi-architecture builds:** `code generation = compute_75,sm_75;compute_86,sm_86;compute_89,sm_89`
  plus `compute_89,compute_89` (PTX) last for JIT on newer cards. `BUILD_GUIDE.md` documents detecting the
  local GPU (`nvidia-smi`) and narrowing the list for faster local builds.
- **Configurations:** ship both `Debug|x64` and `Release|x64`. Release uses host `-O2/O3`;
  `--use_fast_math` **only** where the project explicitly tolerates it (document it — many robotics
  kernels care about reproducible floats). Debug enables `-G` device debug and `-lineinfo` for Nsight.

**Every project's VS solution must:**

1. Build out-of-the-box on a clean machine that has VS 2026 + CUDA 13.3, with **no manual path edits**.
   Use `$(CUDA_PATH)` and the props/targets integration; never hardcode absolute paths.
2. Link only what it uses; if it uses cuBLAS/cuFFT/cuRAND/cuSOLVER/cuSPARSE/Thrust/CUB, add the library
   and **comment in the `.vcxproj`** (MSBuild XML supports `<!-- -->`) why each is linked.
3. Produce a single runnable `.exe` whose output matches `demo/expected_output.txt`.

**Dependency policy (robotics-specific, load-bearing):**

- **Default allowed:** the CUDA toolkit libraries above + C++17 standard library. Nothing else.
- **Small vendored single-header libs** (e.g., `stb_image`, `tinyobjloader`) may be copied into
  `src/thirdparty/` with their license header intact and a comment explaining what they do — only when
  hand-rolling would teach nothing (e.g., PNG decoding).
- **Heavy SDKs (TensorRT, OptiX, OpenCV, ROS 2, Isaac, cuDNN)** are allowed **only** in projects that are
  explicitly *about* them (e.g., 12.01 TensorRT deployment, OptiX raycasting variants). Such projects
  must (a) document the exact extra install in README §6, and (b) still provide a **fallback demo path**
  (plain-CUDA or CPU) so the Definition of Done — demo runs on a clean VS+CUDA machine — still holds.
  Prefer hand-rolled teaching versions everywhere else (e.g., build your own BVH before touching OptiX).
- **Hardware-dependent projects** (Jetson-specific pipelines, NVENC/NVDEC, GPUDirect RDMA, multi-GPU):
  implement the desktop-runnable teaching core, make the demo run on a single desktop GPU (reduced scope
  is fine and labeled), and document the real-hardware path in THEORY.md. Never mark a project done whose
  demo cannot run on the owner's machine.

**CPU reference path:** every project includes `reference_cpu.cpp` — a small, plain, heavily-commented CPU
implementation of the same computation. It is (a) the teaching baseline that makes the GPU speed-up
legible, and (b) the demo's correctness oracle: run both, assert agreement within documented tolerance.
Where a computation is stochastic (particle filters, MPPI, RL), fix seeds and compare statistics or use a
deterministic mode — document the choice.

**Optional `CMakeLists.txt`:** provide where low-cost (helps Linux learners and CI). The **VS solution is
the required deliverable**; CMake is a bonus, never a substitute.

`docs/BUILD_GUIDE.md` is the canonical, copy-paste-friendly guide for installing the toolchain and
building any project. If a build step changes, update it in the same push.

---

## 6. Commenting standard — the heart of this repository

> The owner asked for **"as much comment as possible, explaining what each function does, what each
> variable is for, what the logic and thought process is, how everything ties together."** Take this
> literally. Over-comment on purpose. The canonical, full rubric lives in `docs/COMMENTING_STANDARD.md`;
> this section is the binding summary.

### 6.1 Rules

1. **File header block** at the top of every source file: what this file is, its role in the project, the
   key idea, inputs/outputs, and a "read this after / before" pointer to other files. Reference the
   catalog ID so the file is traceable.
2. **Every function** gets a doc-comment block: purpose, each parameter (units! frames! ranges!
   ownership!), return value, side effects, complexity, and *why it exists*. For kernels, additionally
   document the **launch configuration** (grid/block dims and the reasoning), memory spaces touched,
   atomics/shared-memory use, and the thread-to-data mapping (e.g., "thread `(bx,tx)` owns rollout
   `k = bx*blockDim.x + tx`").
3. **Every non-trivial variable** gets an inline note on first use: what it represents, its units and
   frame, and why it has the type/size it does. Especially flag indices, strides, padded sizes, state
   vector layouts, and anything in device memory.
4. **Narrate the thought process.** Before a block of logic, explain the *intent* and the alternative you
   rejected ("We tile into shared memory because the naive version re-reads global memory N times; see
   THEORY.md §GPU-mapping"). Comments answer **why**, not just restate the code.
5. **Tie it together.** Where a function hands off to another, say so ("costs feed the softmin weight
   reduction in `update_control()`"). Cross-reference README/THEORY sections and the §3.1 system context
   by name.
6. **Explain library calls.** Any cuBLAS/cuFFT/Thrust/etc. call gets 2–4 lines: what it computes
   mathematically, why we use it instead of hand-rolling, and the shape/layout of inputs/outputs.
7. **CUDA error checking is always visible and explained.** Wrap API calls in a `CUDA_CHECK(...)` macro
   (defined and commented once in `src/util/`), plus `cudaGetLastError()` after launches; comment what
   class of failure each guarded call can hit.
8. **No commented-out dead code.** Comments teach; they do not store graveyards.

### 6.2 Density target & illustrative example

Aim for a **comment-to-code ratio a stranger could learn from** — in practice often **≥ 1:1** by line in
kernel files. `verify_project.py` enforces a floor (default ~0.4 non-trivial comment lines per code line
in `src/`) — a safety net, not the goal. The goal is comprehension. A taste (abbreviated), in the style
expected here:

```cpp
// ---------------------------------------------------------------------------
// kernels.cu — MPPI rollout kernel (Project 08.01; see ../THEORY.md)
//
// Big idea: MPPI steers by SAMPLING. Each GPU thread simulates ONE candidate
// control sequence ("rollout") of the system dynamics over the horizon, adds
// up its cost, and the host then blends all sequences with softmin weights.
// K rollouts are fully independent -> one thread per rollout is the natural
// GPU mapping (K ~ 10,000+; a CPU manages dozens).
// ---------------------------------------------------------------------------

// One thread = one rollout. grid: ceil(K/256) blocks, block: 256 threads
// (good occupancy default on sm_75..sm_89; see THEORY.md §occupancy).
__global__ void mppi_rollouts(
    const float* __restrict__ x0,     // [NX] initial state, shared by all rollouts (units: SI, body frame)
    const float* __restrict__ u_nom,  // [T*NU] nominal control sequence from last iteration
    const float* __restrict__ eps,    // [K*T*NU] pre-generated cuRAND noise (per-rollout perturbations)
    float*       __restrict__ cost)   // [K] OUT: total trajectory cost per rollout
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's rollout index
    if (k >= K) return;                             // guard the ragged last block

    // Local state lives in REGISTERS: it is private per thread, updated T times,
    // and never shared — registers are the fastest memory we have. NX is small
    // (cart-pole: 4), so no spilling; we re-check this claim in Nsight (THEORY §profiling).
    float x[NX];
    for (int i = 0; i < NX; ++i) x[i] = x0[i];      // every rollout starts from the SAME state

    float c = 0.0f;                                 // running cost accumulator (unitless, weighted)
    for (int t = 0; t < T; ++t) {                   // march the horizon: x_{t+1} = f(x_t, u_t + eps)
        // ... step dynamics (RK4, inlined & commented in dynamics.cuh), accumulate stage cost ...
    }
    cost[k] = c;   // one coalesced write; the softmin reduction consumes this next (see update kernel)
}
```

(The real file expands every elision and explains the noise scaling, cost shaping, and the softmin.)

### 6.3 Demos

- **Per project:** `demo/run_demo.*` is one command that builds if needed, runs on `data/sample/`, and
  prints a clearly-labeled result plus the GPU-vs-CPU agreement check and a timing line. Where the result
  is inherently visual (depth maps, trajectories, fields), also write a PNG/CSV/OBJ artifact into `demo/`
  and reference it from `expected_output.txt`. `demo/README.md` annotates what the learner is seeing.
- **Top-level showcase:** `showcase/` is a project of its own that ties the collection together — at
  minimum a menu/CLI that lists the 36 domains, shows each project's one-liner + system-context blurb,
  and launches any project's demo; plus a `SHOWCASE.md` tour with a suggested "build a robot" reading
  path through the domains (§3.1 composition map). It is the "demo file showcasing everything" the owner
  asked for. Build it as a VS solution like any project.

---

## 7. Git & GitHub workflow

**Remote:** a **public** GitHub repository named **`cuda-cpp-robotics-projects`** under the owner's
account, created once during bootstrap with the GitHub CLI (`gh repo create cuda-cpp-robotics-projects
--public --source=. --remote=origin`). The owner is authenticated in their own environment; **never handle
their credentials or tokens** — if `gh` is not authenticated, stop and ask the owner to run `gh auth login`.

**Commit conventions:**

- Conventional-commit style with catalog IDs as scope: `feat(08.01): mppi rollout + softmin kernels`,
  `docs(system-design): reference architectures`, `chore(scaffold): stamp 36 domains`, `fix(05.01): tsdf
  truncation off-by-one`.
- Small, coherent commits; one project milestone per commit where practical.
- Never commit secrets, tokens, or large/raw datasets (see `.gitignore` and §8).

**Branching:** trunk-based on `main`. Under Ultracode (§10), each worker uses a short-lived branch
`proj/<SS.NN>-<slug>` touching only its own project folder; the lead merges after `verify_project.py`
passes.

### 7.1 Push-notes (REQUIRED on every push)

> The owner asked: **"every time something new gets pushed onto GitHub, create a .md explaining what was
> added."** This is mandatory and load-bearing.

For **every push to `origin/main`**, add `push-notes/YYYY-MM-DD-NN-short-title.md` (NN = that day's push
counter, zero-padded), stubbed by `tools/new_pushnote.py`. Each push-note must contain:

1. **Summary** — one paragraph: what this push adds and why it matters to the learner.
2. **What changed** — new/edited projects and files, grouped and linked (relative paths).
3. **For each new project** — a 3–5 sentence didactic blurb: the concept it teaches, the CUDA pattern,
   where it sits in the robot (one line), and the single most interesting thing to look at.
4. **How to build & run the new material** — exact commands.
5. **What to study here** — a suggested reading path and 1–2 exercises.
6. **Verification** — what was checked (build passed? demo matched expected? on what GPU/arch?).
7. **Known limitations / TODOs** — honest notes.
8. **Next push preview** — what is planned next.

Then prepend a one-line entry to root `CHANGELOG.md` linking the new push-note. The push-note is written
**before** the push and included **in** that push (so the repo always explains its own latest state).

---

## 8. Datasets, licensing & safety

**Handling (decided for this repo): synthetic-first, download-scripts second, tiny committed samples.**

- Robotics has an advantage over medical data: almost everything can be **synthesized with full ground
  truth** (poses, depth, flow, contacts). `scripts/make_synthetic.py` is therefore the **default** data
  source; several projects (§11 sensor sim) *are* the synthesizers, and their tiny outputs may seed
  sibling projects' `data/sample/` (copied in, per the §4 self-containment rule). Synthetic data must be
  labeled synthetic everywhere it appears.
- Where a public dataset genuinely teaches more, `scripts/download_data.*` fetches it: idempotent,
  documented, with source URL, expected size, and checksum. **Respect every license.** Notable cases:
  KITTI and nuScenes are non-commercial/research licenses that forbid redistribution — the script points
  at the official source and the committed sample stays synthetic; TUM RGB-D and EuRoC MAV are
  attribution-friendly; YCB object meshes vary per object; anything registration-gated gets instructions,
  never a bypass.
- Commit only a **tiny** sample under `data/sample/` so demos run offline. `.gitignore` excludes
  downloaded/large data and build artifacts. If a genuinely necessary committed asset exceeds ~50 MB, use
  Git LFS and note it; prefer to avoid it.
- **Never fabricate results.** Timing numbers in READMEs/push-notes come from actual local runs (state
  the GPU). Speed-ups are teaching artifacts, never benchmark claims.

**Safety guardrails baked into the work (§1 restated as rules):**

- Educational framing in every README; explicit "sim-validated only, not safety-certified" caveat in any
  project whose output could command motion of real hardware (control, planning, teleop, HRI).
- Standards-adjacent projects (e.g., 21.04 speed-and-separation monitoring) compute the published metrics
  didactically and must state they are **not** a certified implementation.
- No weaponization framing; person-perception projects are framed for collaborative safety, not for
  identifying or tracking individuals.
- Medical-robotics projects (§29) carry the sibling repo's rule: educational only, no diagnostic or
  therapeutic claims, and patient-derived data only under licenses that allow redistribution (else synthetic).

---

## 9. Definition of Done & verification gates

A project may be marked **done** (and pushed) only when **all** of these pass:

- [ ] Folder matches the §4 standard layout exactly (run `tools/verify_project.py <path>`).
- [ ] `README.md` has all 13 required sections (§4.1) — including **System context**; `THEORY.md` covers
      physics/engineering-first problem → math → algorithm → GPU mapping → numerics → verification →
      real world.
- [ ] `PRACTICE.md` exists with its four sections (§4.3): construction, real hardware/BOM, installation &
      integration on a real robot, business & regulatory — each genuinely written or honestly N/A'd.
- [ ] `src/` compiles via the VS solution in `Release|x64` **and** `Debug|x64` with zero errors and zero
      new warnings (treat warnings as defects to explain or fix).
- [ ] `reference_cpu.cpp` exists; the demo runs GPU + CPU and asserts agreement within documented
      tolerance (or documented statistical/seeded-deterministic check for stochastic projects).
- [ ] `demo/run_demo.*` runs on `data/sample/` with **no downloads and no extra SDKs** and matches
      `demo/expected_output.txt`.
- [ ] Commenting passes the density floor and, more importantly, a human could learn from it (spot-read).
- [ ] `data/` sample present and labeled (synthetic/public); scripts documented; licenses respected.
- [ ] `docs/STATUS.md` updated (project → done); a push-note written; committed and pushed.

**`tools/verify_project.py`** automates the structural checks (files present — including `PRACTICE.md`
and its four sections, README sections present — including System context, comment-density heuristic,
expected_output present). It prints a checklist with
pass/fail. Do not mark a task complete while any gate fails — keep it in_progress and write a push-note
explaining the blocker.

**CI (optional but recommended):** a GitHub Actions workflow that runs `verify_project.py` across all
projects and **compiles** changed CUDA projects. GitHub's hosted runners have **no NVIDIA GPU**, so CI can
*compile* but cannot *run* kernels — running/demoing is a **local** step. Document this clearly; never let
a green build badge imply kernels were executed in CI.

---

## 10. Multi-agent orchestration (Ultracode)

This repo is built by many agents working in parallel. The cardinal rule that makes parallelism safe:

> **One agent owns one project folder at a time. Agents never edit files outside their own
> `projects/<…>/` folder** (except via the lead). No two agents touch the same file, so concurrent work
> never conflicts.

**Roles:**

- **Lead/integrator (one):** owns all shared/root files — `CLAUDE.md`, `README.md`, `CHANGELOG.md`,
  `docs/` (including `SYSTEM_DESIGN.md`), `tools/`, `catalog.json`, `docs/STATUS.md`, `.gitignore`, CI,
  `showcase/`, and the GitHub remote. The lead runs bootstrap (§11 Phase 0), assigns projects, merges
  branches, writes the push-note for each push, and pushes. Workers do **not** push to `main` directly.
- **Workers (many):** each claims one project from `docs/STATUS.md` (set it `in-progress` with the
  agent's name), builds it to the Definition of Done on a `proj/<SS.NN>-<slug>` branch, runs
  `verify_project.py`, then hands back to the lead for merge. Then claims the next unclaimed project.

**Claiming protocol:** `docs/STATUS.md` (generated from `catalog.json`) is the work queue. A worker claims
the highest-priority `todo` item by editing only its own status row on its branch; the lead resolves rare
claim races at merge time. Keep batches modest (8–16 workers in flight) so review and merges stay
tractable.

**Integration checkpoints:** after each batch, the lead (a) merges all green branches, (b) runs the full
`verify_project.py` sweep + a build of changed projects, (c) writes one push-note covering the batch,
(d) updates `STATUS.md` and `CHANGELOG.md`, (e) pushes. A red project stays on its branch with a TODO
note; it is not merged until green.

**Consistency:** every worker follows this file and `docs/COMMENTING_STANDARD.md` verbatim, starts from
`docs/PROJECT_TEMPLATE/`, and copies (never symlinks) `src/util/` from the template.

---

## 11. Rollout plan (phased)

**Phase 0 — Bootstrap (lead, once).** Write root files (`README.md`, `LICENSE` MIT, `.gitignore`,
`CHANGELOG.md`), `docs/` (`COMMENTING_STANDARD.md`, `BUILD_GUIDE.md`, `SYSTEM_DESIGN.md` per §3.1,
`PROJECT_TEMPLATE/`), `tools/` (all five scripts), run `catalog.py` → `catalog.json` (report the exact
project count), run `scaffold.py` to stamp **all** project skeletons (catalog-prefilled README stubs plus
THEORY.md/PRACTICE.md stubs with their required section headers + TODO markers), generate `docs/STATUS.md`, init git, create the **public** GitHub repo, first commit +
push, and write `push-notes/<date>-00-bootstrap.md`.

**Phase 1 — Flagships (one polished project per domain, 36 total).** Build these *completely* first so
the owner has best-in-class study material in every domain quickly and the standards get battle-tested
before scaling. Push in small batches (~6 flagships per push-note). Suggested flagships (IDs are expected
positions — confirm against `catalog.json` after scaffold; swap for a tractable sibling in the same
domain if needed, preferring ★ entries with clean demos):

| Domain | Suggested flagship | Domain | Suggested flagship |
|--------|--------------------|--------|--------------------|
| 01 Cameras & vision | `01.02` Stereo SGM depth | 19 Manipulation | `19.01` Antipodal grasp scoring |
| 02 LiDAR & point clouds | `02.06` GPU ICP (pt-to-plane) | 20 Tactile | `20.01` GelSight contact/shear/slip |
| 03 Radar/sonar/event | `03.01` FMCW cube + CFAR | 21 HRI | `21.04` Speed-and-separation monitor |
| 04 Fusion & estimation | `04.01` Massive particle filter | 22 Swarms | `22.01` 100k-agent swarm sim |
| 05 SLAM & mapping | `05.01` TSDF fusion (KinectFusion) | 23 Navigation | `23.01` GPU costmaps + DWA |
| 06 Motion planning | `06.05` STOMP | 24 Actuators | `24.01` Magnetostatic FEA sweeps |
| 07 Collision & geometry | `07.09` Jump-flooding SDF/Voronoi | 25 Power | `25.01` Battery electro-thermal |
| 08 Control | `08.01` MPPI (cart-pole → quadrotor) | 26 Mech design | `26.01` Topology optimization |
| 09 Dynamics & kinematics | `09.01` Batched FK (+Jacobians) | 27 Materials | `27.04` Composite layup sweeps |
| 10 Physics sim | `10.03` 10k-env parallel robot sim | 28 Soft robotics | `28.01` Real-time FEM soft arm |
| 11 Sensor sim | `11.01` GPU LiDAR simulator | 29 Medical | `29.05` Ultrasound beamforming |
| 12 ML & AI | `12.01` TensorRT deploy + custom kernels | 30 Field | `30.01` Fruit detect+localize (milestone 1) |
| 13 Legged | `13.03` Foothold scoring kernels | 31 Safety | `31.01` HJ reachability level sets |
| 14 Wheeled | `14.02` Traversability costmaps | 32 Embedded | `32.02` CUDA Graphs control loop |
| 15 Aerial | `15.01` Minimum-snap batches | 33 Foundations | `33.01` Batched small-matrix linalg |
| 16 Marine | `16.01` Thruster allocation QP | 34 Theory | `34.03` Ergodic control (FFT) |
| 17 Space | `17.01` Lambert + porkchop plots | 35 Micro/nano | `35.01` Magnetic microswarm fields |
| 18 Other locomotion | `18.01` Snake serpenoid sweeps | 36 Modular | `36.03` Lattice-robot kinematics |

Build order within Phase 1: start with `33.01`, `09.01`, `07.09`, `08.01` — they are foundations other
flagships reuse patterns from. After Phase 1, reassess the standards and update the template/docs if the
flagships surfaced improvements.

**Phase 2 — Batched build-out (remaining ~380).** Work domain by domain, **easiest-first within a
domain** (★ → untagged → [R&D]), many workers in parallel per §10. Each project to full Definition of
Done. Push per batch with a push-note. Keep `docs/STATUS.md` and `CHANGELOG.md` current. `[R&D]` projects
may ship as reduced-scope teaching versions (full version documented in THEORY.md) — but the gates in §9
still apply in full.

**Phase 3 — Showcase & polish.** Build `showcase/` (the everything-demo, §6.3), the top-level `README.md`
index with a domain map and the §3.1 composition diagram, optional CI, and a final pass for cross-links
(especially README System-context ↔ `SYSTEM_DESIGN.md` ↔ `PRACTICE.md`) and consistency. Final push + summary push-note.

**Priority signal:** within a domain, rank `todo` by difficulty (★ first, `[R&D]` last), then by ID.
This front-loads quick didactic wins and defers frontier projects.

---

## 12. Conventions quick-reference

- **Language/standard:** CUDA C++ targeting C++17 host code; `.cu`/`.cuh` device, `.cpp`/`.h` host.
- **Style:** clear names over short ones; `snake_case` functions/variables, `PascalCase` types,
  `UPPER_CASE` macros/constants; `d_`/`h_` prefixes for device/host pointers.
- **Units & frames (robotics-critical):** SI everywhere — meters, seconds, radians, N·m; document units
  in names or comments (`dt_s`, `torque_nm`, `omega_rad_s`). Right-handed frames; body convention
  x-forward/y-left/z-up unless a domain standard says otherwise (state it). Transforms named
  `T_parent_child` ("child expressed in parent"); quaternions stored `(w,x,y,z)`, kept normalized, order
  documented at every API boundary. Angles wrapped to `(-π, π]` at defined points only.
- **State vectors:** every `float* state` documents its layout in one place and cross-references it.
- **Errors:** every CUDA API/kernel launch checked via the commented `CUDA_CHECK` macro;
  `cudaGetLastError()` after launches.
- **Timing:** CUDA events for kernel timing; print a clearly-labeled ms figure and (where meaningful) a
  GPU-vs-CPU speed-up — *a teaching artifact, never a benchmark claim*.
- **Determinism:** fix RNG seeds in demos (`curand` seeds documented); prefer deterministic reductions in
  teaching code; where atomics reorder float sums, say so and explain the caveat.
- **Markdown:** every project README/THEORY/PRACTICE uses the same heading order; keep relative links
  working.
- **Hardware & business content (§4.3):** illustrative and dated, never prescriptive — no endorsements,
  no procurement/legal/compliance advice; regulatory references are orientation, not guidance.
- **No black boxes, no fabricated data or timings, no deployment/safety claims, no committed secrets.**

---

## 13. When you (Claude Code) are unsure

- If a catalog bullet is ambiguous or bundles too much, implement the **simplest correct teaching
  version** of the primary named method, document the rest in `THEORY.md` under "Where this sits in the
  real world", and state the scoping in README §13.
- If a build/tooling assumption fails on the owner's machine, **stop and ask** rather than guessing — and
  capture the fix in `docs/BUILD_GUIDE.md`.
- If a project seems to need hardware the owner lacks (Jetson, LiDAR, multi-GPU), build the desktop
  teaching core per §5 and document the hardware path — do not stall, do not fake results.
- If a dataset cannot be obtained legally, **switch to synthetic** and label it.
- Never silently skip a Definition-of-Done gate. Surface it.

*End of contract. Build to teach.*


