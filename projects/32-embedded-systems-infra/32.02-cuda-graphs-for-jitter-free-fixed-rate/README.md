# 32.02 — CUDA Graphs for jitter-free fixed-rate perception-control loops

**Difficulty:** intermediate · **Domain:** 32. Embedded, Jetson & Systems Infrastructure

> Catalog bullet (source of truth, verbatim): `CUDA Graphs for jitter-free fixed-rate perception-control loops`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

This project is a **latency/jitter measurement study**, not a new controller. It takes one
realistic "tick" of a fixed-rate robot perception-control loop — eight small CUDA kernels plus
two device-to-device "publish" copies, wired together exactly like a real pipeline (perception →
estimation → MPPI-style planning → control) — and runs it **2000 times at a fixed 250 Hz** three
different ways: (A) twelve individual CUDA API calls issued fresh every tick, the way every
learner starts; (B) the identical twelve calls captured **once** into a CUDA Graph and replayed
with a single `cudaGraphLaunch()`; (C) the same graph, but with one input double-buffered and
repointed every tick via `cudaGraphExecKernelNodeSetParams` — the technique real systems use when
a captured node's *argument*, not just the memory behind it, must legitimately change. For every
mode it measures host **submit time**, end-to-end **latency**, and the achieved **pacing period**,
then reports the honest result — including where CUDA Graphs *do not* help on this machine. The
rollout kernel at the pipeline's core is project **08.01**'s MPPI cart-pole dynamics, reused **by
name** at a much smaller K/T; everything else (the perception/estimation/control stages, the
graph-capture and explicit-construction machinery, the pacing clock, the measurement harness) is
new. All three modes, the CPU correctness oracle, and the full measurement/artifact pipeline are
implemented — there is no documented-only component to this bullet.

## What this computes & why the GPU helps

Per tick: a small perception→estimation→planning→control DAG (map → stencil → predict → reduce+fuse
→ K=512×T=16 MPPI rollout batch → min-reduce → softmin-weight reduce → T weighted blends), sized so
the whole tick's **device** work is ~0.21 ms — deliberately tiny (see "The sizing argument" below).

- **Pattern:** this project is not about ONE parallel pattern — the tick strings together a *map*
  (stage 1), a *stencil* (stage 2), tiny *pointwise* kernels (stages 3), and three *reductions*
  (stages 4, 6, 7) around one *batched-sampling* kernel (stage 5, 08.01's pattern) and a further
  *reduction* (stage 8). The GPU-acceleration story here is not "this is embarrassingly parallel" —
  it is "this DAG of small parallel kernels must be *submitted* fast enough, every 4 ms, forever."
- **The sizing argument (load-bearing):** launch/driver overhead on Windows WDDM runs a few to a
  few hundred microseconds per `cudaLaunchKernel`/`cudaMemcpyAsync` call (THEORY.md derives the
  path). That overhead is *invisible* next to a 5 ms perception kernel — and *dominant* next to a
  20 µs one. This project's eight kernels are kept tiny on purpose so the measured effect (launch
  overhead as a fraction of the tick) is the thing being studied, not noise beneath a bigger
  workload. Measured on this machine: ~0.21 ms of device time hides behind ~135 µs (mode A) to
  ~75 µs (mode B) of host submission time per tick — the same order of magnitude, exactly the
  regime where orchestration technique matters.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** cross-cutting **infrastructure** (SYSTEM_DESIGN §1 diagram's "Infrastructure"
  box, domain 32) wrapping the **planning→control boundary** — the tick pipeline this project
  studies spans local planning and control, the same boundary project 08.01 occupies. This project
  IS the concrete instance of SYSTEM_DESIGN §1.2's stated research frontier: *"pushing the GPU down
  the stack into the 0.5–1 kHz control layer... the classic answers are CUDA Graphs, persistent
  kernels, pinned zero-copy memory... domain 32, e.g. the CUDA-Graphs control loop."* That sentence
  names this exact project.
- **Upstream inputs:** any fixed-rate GPU pipeline that must run on a schedule — named directly,
  **project 08.01's MPPI controller** (whose rollout kernel this project's stage 5 reuses), and more
  generally anything in SYSTEM_DESIGN's planning/control layer that currently pays a naive
  per-kernel launch tax every tick.
- **Downstream consumers:** the real robot's control process reading the tick's published outputs
  (`Twist`/`JointState`-shaped command, §3.6) at a hard deadline — this project's two D2D "publish"
  copies are exactly the hand-off a `ros2_control` update() would perform (PRACTICE §3).
- **Rate/latency budget:** this project runs its study at **250 Hz** (4 ms period) — squarely inside
  SYSTEM_DESIGN §1.1's *local planner / trajectory replan* band (10–50 Hz) pushed toward the
  *whole-body / trajectory-tracking control* band's low end (0.5–1 kHz, 1–2 ms hard deadline). The
  250 Hz choice is deliberate: it is fast enough that a ~100 µs difference in submit time is a
  meaningful fraction of the budget (the whole point), while staying inside what a *software-paced*
  Win32 host thread can hit reliably (measured p50 accuracy: within ~0.3 µs of the 4000 µs target —
  see "Expected output"). Reaching the 1 kHz frontier for real needs the techniques THEORY.md's
  "Where this sits in the real world" section names (persistent kernels — **project 32.03**, TCC/
  Linux + PREEMPT_RT, or a dedicated real-time core) — this project measures the *first* rung of
  that ladder honestly, not the last one.
- **Reference robot(s):** the **quadruped** (SYSTEM_DESIGN §2.3's 0.5–1 kHz whole-body control, "the
  hard deadline lives at the bottom" per §4.3) and the **6-DoF manipulator work cell** (§2.2's
  vendor-controller trajectory-tracking loop) most directly — both are exactly the kind of tight,
  fixed-rate loop this project's measurement methodology would be applied to before trusting a GPU
  there. The **warehouse AMR**'s MPPI local planner (§2.1, §4.1 Chain A) is 08.01's home and this
  project's nearest upstream sibling.
- **In production:** CUDA Graphs are production technology today in TensorRT (graph-capturing whole
  inference engines) and Triton Inference Server (reducing per-request launch overhead at scale);
  NVIDIA's Isaac/cuMotion stacks use graphs for planning pipelines. **Project 32.03 (persistent
  kernels for microsecond latency)** is the next rung on the frontier ladder this project starts —
  where even `cudaGraphLaunch`'s residual host-driver round trip is removed by keeping a kernel
  resident on the GPU and signaling it through memory instead of relaunching it.
- **Owning team:** embedded/platform engineering (SYSTEM_DESIGN §5.1's "Embedded / firmware" row,
  domain 32) — the team that owns the *real-time budget* every other team's GPU code must fit
  inside; adjacent to controls/autonomy (08.01's owning team, whose workload this project wraps) and
  simulation/tools (whose profiling conventions this project's measurement methodology follows).

## The algorithm in brief

- **The tick pipeline (8 kernels + 2 D2D copies)** — perception (map + stencil) → estimation
  (predict + reduce/fuse) → MPPI rollout batch (08.01's dynamics, K=512×T=16) → softmin reduction
  (min + weighted-exp-sum) → control blend (T weighted reductions) → publish. →
  [`src/kernels.cuh`](src/kernels.cuh)/[`src/kernels.cu`](src/kernels.cu); THEORY.md §The algorithm.
- **CUDA Graph capture (mode B)** — `cudaStreamBeginCapture`/`cudaStreamEndCapture` recording the
  exact same function call mode A uses, `cudaGraphInstantiate` once, `cudaGraphLaunch` every tick. →
  THEORY.md §What CUDA Graphs actually are.
  **update-vs-recapture, not update-vs-nothing.**
- **Explicit graph construction + parameter update (mode C)** — `cudaGraphAddKernelNode`/
  `cudaGraphAddMemcpyNode1D` building the DAG by hand so one node's pointer argument can be
  repointed every tick via `cudaGraphExecKernelNodeSetParams` (double-buffered sensor input). →
  THEORY.md §Update vs recapture.
- **Hybrid sleep+spin pacing** — `timeBeginPeriod(1)` + `Sleep()` for the bulk of each 4 ms period,
  busy-spin on `QueryPerformanceCounter` for the final ~1.2 ms, because Windows' default scheduler
  granularity cannot hit a 4 ms deadline by itself (measured on this machine: `Sleep(1)` actually
  takes ~1.6–1.9 ms — see the demo's `[info]` calibration line). →
  THEORY.md §Measurement methodology.
- **Bit-identical cross-mode verification** — all three modes run the identical deterministic
  kernels on identical per-tick inputs with identical launch geometry, so their outputs are checked
  for exact equality, not tolerance. → THEORY.md §Numerical considerations.

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/cuda-graphs-for-jitter-free-fixed-rate.sln`](build/cuda-graphs-for-jitter-free-fixed-rate.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/cuda-graphs-for-jitter-free-fixed-rate.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **`winmm.lib`** (Windows only) — `main.cu`'s pacing
clock calls `timeBeginPeriod`/`timeEndPeriod` to get 1 ms `Sleep()` resolution; without it the
hybrid sleep+spin scheme still runs but would need a larger, less efficient spin margin. The
CMake/Linux path has no equivalent call (see THEORY.md "Windows WDDM vs Linux/TCC vs Jetson") and
links no extra library. No other dependency beyond the CUDA runtime + C++17 standard library.

## Run the demo

One command, from this folder (builds first if needed, runs the full three-mode study, checks
GPU-vs-CPU correctness and cross-mode bit-identity):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh` (pacing is less precise there — see THEORY.md). See
[`demo/README.md`](demo/README.md) — including the **two CSV artifacts to plot**. Total runtime is
~24 s (3 modes × 2000 ticks × 4 ms pacing, plus warmup and setup) — comfortably under a minute.

## Data

The committed sample is a **scenario**, not recordings:
`data/sample/tick_scenario.csv` (~0.6 KiB, synthetic, no RNG in the file itself) — the starting
state estimate, the two RNG seeds, and the tick-count/pacing parameters the study runs with. The
per-tick synthetic sensor readings and MPPI exploration noise are generated **in-demo**, in memory,
from these seeds every run (`src/main.cu`'s `tick_inputs()`) — deterministic, so every mode sees
bit-identical inputs at the same tick index. No public dataset applies (this is a synthetic
measurement workload, not sensor data); `scripts/download_data.ps1` is an honest no-op. Details,
including the SHA-256 of the committed file: [`data/README.md`](data/README.md).

## Expected output

Eleven stable lines — banner, `PROBLEM:`, `SCENARIO:`, `VERIFY:`, `CROSSMODE:`, three `GATE ...:`
lines, two `ARTIFACT:` lines, and `RESULT: PASS` — checked as a subset diff by
[`demo/expected_output.txt`](demo/expected_output.txt). **Two independent correctness checks, both
required to pass** (the doubled-up §5 gate this project's brief calls for):

1. **VERIFY** — tick 0's GPU path (mode A) vs. [`src/reference_cpu.cpp`](src/reference_cpu.cpp)'s
   plain-C++ twin of the *whole* tick, within documented tolerance (rel 1e-3 on rollout costs, abs
   1e-3 on the blended plan and corrected state estimate — 08.01's precedent). Measured worst case
   on this machine: rollout-cost rel deviation `4.130e-07`, u_nom abs `5.066e-07`, state abs
   `0.0` — five orders of magnitude inside the gate.
2. **CROSSMODE** — modes B and C's full 2000-tick output trajectories (the published control plan
   and state estimate, every tick) compared **bit-for-bit** against mode A's. Measured: **PASS**,
   zero mismatches over 2000 × (16 + 4) = 40,000 floats per graph mode. This is the check that
   proves the orchestration technique never changes the answer.

Three measurement **gates**, all satisfied on this machine and printed with the policy (not the
raw numbers, which vary run to run and live in the `[info]` lines and the CSV artifacts):
`submit-reduction` (both graph modes' mean host submit time ≤ 75% of naive), `gpu-work-consistency`
(mean device-timeline time per tick agrees across all three modes within 2×), `pacing-accuracy`
(every mode's achieved p50 period within ±600 µs of the 4000 µs target). **Tail latency (p95/p99/
max) is never gated** — it is measured and reported honestly as `[info]`/CSV data, because on
Windows WDDM the tail is dominated by OS/driver scheduling this project does not control (see
"The honest result" below and THEORY.md).

### The honest result (measured on this machine, RTX 2080 SUPER, driver-current as of the last run)

| Metric (mean) | A: stream (naive) | B: graph (static) | C: graph (SetParams) |
|---|---|---|---|
| Host submit time | ~135–145 µs | **~75–78 µs (≈45% lower)** | ~85–93 µs (≈35% lower) |
| End-to-end latency | ~347–355 µs | ~385–399 µs (**higher**) | ~392–404 µs (**higher**) |
| Device-timeline exec | ~0.210 ms | ~0.211–0.222 ms | ~0.222–0.228 ms |
| Achieved p50 period | 4000.0–4000.4 µs | 4000.0–4000.4 µs | 3999.6–4000.4 µs |

**Launch-call reduction:** mode A issues 12 host CUDA API calls per tick (2×`cudaMemcpyAsync` H2D +
8 kernel launches + 2×`cudaMemcpyAsync` D2D); mode B issues **1** (`cudaGraphLaunch`); mode C issues
**3** (one memcpy into the alternating buffer, one `cudaGraphExecKernelNodeSetParams`, one
`cudaGraphLaunch`). The submit-time numbers above are that reduction paying off.

**The unmarketed part, exactly as measured:** CUDA Graphs cut this machine's *submission* overhead
substantially — but they did **not** reduce end-to-end tick latency or tail jitter; on every run,
mean and p99 latency were *higher* for both graph modes than for naive per-kernel launches (p99
latency: A ≈ 962–1041 µs vs. B/C ≈ 1230–1240 µs — see `demo/out/jitter_summary.csv` for the exact
numbers from your run). THEORY.md "Where this sits in the real world" discusses the likely cause
(inter-node synchronization overhead inside a many-tiny-kernel graph replay, plus WDDM's own
batching being largely amortized already for a single-stream naive sequence) — the honest teaching
point is that **submission overhead and completion latency are different budgets**, graphs address
the first reliably and the second not at all on this platform, and a real system deciding whether
to adopt graphs needs to measure *both*, not assume the marketing headline transfers.

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — the tick's contract: sizes, the 08.01-borrowed plant, the
   measurement-record shapes every other file agrees on.
2. [`src/kernels.cu`](src/kernels.cu) — the eight kernels, stage by stage, each with its own
   thread-mapping and memory-space commentary.
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the plain-C++ twin of the whole tick (the
   VERIFY-stage oracle).
4. [`src/main.cu`](src/main.cu) — the heart of the project: `submit_naive` (mode A's body, and
   exactly what gets captured for mode B), `capture_graph_stream` (mode B's setup),
   `build_graph_setparams`/`GraphCState`/`submit_setparams` (mode C's explicit construction and
   per-tick update — the single most interesting thing to look at), `PacingClock` (the hybrid
   sleep+spin scheduler), and `run_mode` (the measurement loop all three modes share).
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers (copied, not shared — §4 rule).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **NVIDIA's "Getting Started with CUDA Graphs" developer blog + the CUDA Programming Guide's Graphs
  chapter** — the canonical API reference for everything `main.cu` uses
  (`cudaStreamBeginCapture`/`cudaGraphAddKernelNode`/`cudaGraphExecKernelNodeSetParams`); study the
  full API surface this project only samples.
- **TensorRT** — captures whole inference engines as CUDA Graphs in production; the reason "graphs
  reduce launch overhead" is not a toy claim, just one this project shows has *limits*.
- **Triton Inference Server** — uses CUDA Graphs specifically to cut per-request submission latency
  at high request rates, the same submit-time story this project measures at tick granularity.
- **08.01 (MPPI controller)** — the exact rollout kernel this project's stage 5 wraps; read it for
  the physics and the softmin derivation this project deliberately does not repeat.
- **33.01 (Batched small-matrix linear algebra) PRACTICE.md §3** — names this project ("32.02") as
  the frontier that tightens the "GPU is not hard-real-time on WDDM" story it establishes; read it
  first for the honest baseline this project's numbers extend.
- **Project 32.03 (persistent kernels for microsecond latency)** — the next rung: removing even
  `cudaGraphLaunch`'s host-driver round trip by keeping a kernel resident and signaling it through
  memory. This project's measured latency floor is roughly where 32.03's motivation begins.

## Exercises

1. **Plot the artifacts:** `demo/out/latency_histogram.csv` → a histogram of `latency_us` per mode
   (the "jitter" the catalog bullet names). Overlay all three modes and see the tail visually, not
   just as p99 numbers.
2. **Multi-stream capture:** mode C's explicit graph is deliberately forced into the same *linear*
   dependency chain as mode B's stream capture (see the comment in `build_graph_setparams`), even
   though stages like `state_predict` and `sensor_scale_bias` have no real data dependency. Rebuild
   it as a true fork-join DAG (or capture mode B on two streams) and measure whether the GPU
   actually overlaps the independent branches — and whether that changes the gpu-exec numbers.
3. **Recapture, measured:** add a fourth mode that destroys and rebuilds the whole graph every tick
   (`cudaGraphDestroy` + a fresh `cudaStreamBeginCapture`/`EndCapture`/`cudaGraphInstantiate`
   instead of `cudaGraphExecKernelNodeSetParams`) and measure its submit time against mode C's.
   THEORY.md predicts recapture will be far more expensive; find out by how much on your machine.
4. **Chase the latency gap:** modes B/C show *higher* mean and p99 latency than mode A despite lower
   submit time. Use Nsight Systems to look inside one tick's graph replay vs. one tick's naive
   sequence and find where the extra time goes (candidate: inter-node synchronization primitives).
5. **Raise the stakes:** shrink `kDt`/lower `kHorizon` further, or raise `kPacingHz` toward 1000 in
   `data/sample/tick_scenario.csv`, and watch the submit-time fraction of the budget grow — at what
   point does mode A start missing deadlines that mode B/C still hit?

## Limitations & honesty

- **Not a controller demo.** The tick pipeline does not close the loop against an evolving simulated
  plant (08.01 already teaches that); the state estimate evolves via its own internal predict/
  correct recursion, decoupled from the blended control plan. This project's job is measuring
  orchestration overhead on a *representative* DAG, not demonstrating control performance —
  conflating the two would have made the 2000-tick determinism claim far harder to keep honest
  (chaotic cart-pole dynamics feeding back through 2000 ticks would amplify any tiny cross-mode
  floating-point divergence into a real difference; this project's scope avoids that risk on
  purpose rather than papering over it).
- **The sensor-to-measurement model is deliberately fictional.** Stage 4's "range scan → position
  proxy" relationship exists to give the pipeline a realistic multi-kernel *shape* (map → stencil →
  reduce → fuse), not to teach sensor modeling (domains 01–03's job) or estimation theory (domain
  04's job — a real EKF/UKF computes a covariance-derived Kalman gain; this project uses a fixed
  gain, stated as a simplification in `kernels.cuh`).
- **Mode C's dependency chain is deliberately serialized**, not the true DAG its explicit
  construction could express, so that its device-timeline execution time is comparable to mode B's
  necessarily-linear stream capture (see Exercise 2 for the alternative).
- **Windows-first.** The pacing clock's precision techniques (`timeBeginPeriod`, the measured
  Sleep(1) calibration) are Windows-specific; the CMake/Linux path runs but with a coarser fallback
  (THEORY.md documents what changes on Linux/PREEMPT_RT and Jetson).
- **Timings are teaching artifacts** — single-shot per run, one machine (RTX 2080 SUPER, sm_75,
  WDDM), never a benchmark claim. Re-run `demo/run_demo.ps1` and read your own `[info]`/`[time]`
  lines and CSVs; do not trust the numbers quoted here on a different machine or driver.
- **No hardware motion.** This project's output never commands anything (it publishes to in-memory
  buffers a real system would forward to actuation); the repo-wide sim-validated-only caveat applies
  at minimal strength here, but is stated for consistency with CLAUDE.md §1.
