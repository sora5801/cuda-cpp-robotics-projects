# 32.02 — CUDA Graphs for jitter-free fixed-rate perception-control loops: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). This is a
> **systems** project, not a physics one; the physics-first tradition below is honestly adapted —
> the "physical phenomenon" here is a *control loop's timing*, and timing is a physical quantity
> with its own real consequences, derived first before anything about CUDA appears.

## The problem — physics & engineering first

**Why computation jitter is a physical disturbance, not a software inconvenience.** A digital
controller does not apply a continuous force; it samples the world, computes, and holds a command
constant until the next tick (zero-order hold — 08.01's THEORY.md uses the same model). If tick `k`'s
command is computed and applied at time `t_k`, the controller's *intended* behavior assumes
`t_{k+1} - t_k = dt` exactly, every time. **Jitter is the difference between that assumption and
reality.** Consider a simple linear plant under proportional feedback, `u(t) = -K x(t)`, discretized
with a zero-order hold over an *actual* (jittered) interval `dt + δ`:

```
x_{k+1} = A(dt+δ) x_k + B(dt+δ) u_k
```

`A(dt+δ)` and `B(dt+δ)` are the plant's exact discretization at the *actual* interval, not the
*nominal* one the gains `K` were tuned for. A late tick (`δ > 0`) means the plant evolves under the
OLD command for longer than the control law assumed, injecting a small position/velocity error
every single tick — exactly like adding a random disturbance whose magnitude scales with `δ`. This
is not a hand-wave: in frequency-domain terms, a control loop closed at nominal rate `f = 1/dt` has
a **phase margin** — the extra phase lag the loop can tolerate before instability — and a zero-order
hold consumes phase margin proportional to `ω·dt/2` at frequency `ω` (the ZOH's phase lag is
`-ω·dt/2` radians). **Random variation in `dt` is randomly-varying phase lag** — it does not shift
the nominal margin so much as add *noise* to it, which is why practitioners describe jittery control
loops as behaving like they have added, unmodeled high-frequency disturbance: the tighter the loop
(the higher `1/dt` relative to the plant's fastest mode), the more margin a given `δ` eats, which is
exactly why SYSTEM_DESIGN.md §1.1 marks the 0.5–1 kHz whole-body control band "HARD deadline...
miss it and a legged robot literally falls over" while the 10–50 Hz local-planner band tolerates
misses gracefully (08.01's PRACTICE.md §3: "MPPI degrades gracefully — yesterday's shifted plan is
still a plan"). **A late torque command is a disturbance the control law never accounted for.**

**The engineering constraint this project studies.** Given that jitter has a real cost, the
engineering question is mechanical: *what actually makes a tick late on a GPU-accelerated loop?*
Three budgets stack on top of each other every tick: (1) the GPU's own execution time for the
tick's kernels, (2) the **host-side cost of telling the GPU to do that work** (this project's
subject), and (3) OS/driver scheduling variance around both. Perception and planning kernels
(domains 01–08) are usually *treated* as free of (2) because their tick budget is tens of
milliseconds and a dozen kernel launches cost microseconds — two orders of magnitude of headroom.
Push the same style of workload toward a 4 ms or 1 ms budget (SYSTEM_DESIGN §1.2's stated research
frontier) and (2) stops being free. This project isolates and measures (2) and (3) in controlled
conditions: the *same* GPU work, submitted three different ways, so any difference in the measured
numbers is attributable to *how the work reached the GPU*, not to what the work was.

## The math

**The tick's DAG, formally.** Let a tick be a directed acyclic graph of eight kernels
`K_1..K_8` plus two copy operations `C_1, C_2` (kernels.cuh/kernels.cu name every stage). Define
the tick's per-mode SUBMIT operator `S_m` (host wall time spent issuing the DAG's operations for
mode `m ∈ {A, B, C}`, measured with no waiting) and LATENCY operator `L_m` (host wall time from
submission start to GPU-completion-confirmed). The claim this project tests:

```
S_A  >  S_B , S_C          (submission overhead is orchestration-dependent)
L_A  ≈? L_B ≈? L_C          (is end-to-end latency orchestration-dependent too? MEASURE, do not assume)
f(K_1..K_8, C_1, C_2)  identical  ∀ m         (the DAG computes the same function regardless of m)
```

The third line is the correctness invariant `main.cu`'s CROSSMODE check verifies; the first two are
what the GATE checks and the `[info]`/CSV numbers report — and, measured honestly on this project's
reference machine, the SECOND line does **not** hold (`L_B, L_C > L_A` — see README "The honest
result"). That negative result is itself part of what this project teaches.

**Pacing, formally.** A fixed-rate loop targets tick times `t_k = t_0 + k·dt`. The ACHIEVED period
`p_k = t_k^{actual} - t_{k-1}^{actual}` is what `main.cu` logs as `period_us`; the study's pacing
gate checks `median(p_k) ≈ dt` (with a documented tolerance), never `max(p_k)` (the tail is
reported, never gated — see "How we verify correctness").

## The algorithm

**Step-by-step, per tick** (kernels.cu's stage numbering; complexity is per-tick, not asymptotic —
every stage here operates on a FIXED, tiny size, by the sizing argument in README):

1. **sensor_scale_bias** (map, `O(N)`, `N = kSensorN = 512`) — affine transform, one thread/element.
2. **sensor_smooth** (3-tap stencil, `O(N)`) — denoise, one thread/element, boundary-clamped.
3. **state_predict** (`O(1)`, 4 threads) — constant-velocity kinematic predict.
4. **state_correct** (reduction + fuse, `O(N)` work / `O(log N)` depth) — mean of the smoothed
   array via a shared-memory tree reduction (9 rounds for `N=512=2^9`), then a fixed-gain
   complementary fusion into the predicted position.
5. **mppi_rollout** (batched sampling, `O(K·T)` work / `O(T)` depth per thread, `K=512, T=16`) —
   08.01's cart-pole RK4 dynamics + cost, one thread per rollout, entirely parallel across `K`.
6. **cost_min** (reduction, `O(K)` work / `O(log K)` depth) — the softmin numerical-safety minimum.
7. **softmin_weight** (elementwise + reduction, `O(K)` / `O(log K)`) — importance weights and their
   sum, in one pass.
8. **control_blend** (`T` independent reductions, `O(K)` work / `O(log K)` depth **each**, `T=16` of
   them run as `T` CUDA blocks) — fold the weighted noise back into the plan, clamp.
9. **publish ×2** (`O(1)`/`O(N)` copies) — hand the tick's two outputs to stable buffers.

**Serial-vs-parallel cost.** On a single CPU core, the whole tick is `O(N + K·T + K log K)` — a few
hundred thousand scalar operations, sub-millisecond even sequentially (this project's rollout size
is small — see 08.01's THEORY.md for where the CPU/GPU gap becomes dramatic at production K/T).
**The point of this project is NOT that the parallel algorithm beats the serial one on FLOPs** —
at K=512×T=16 it barely does. The point is what stands between "the algorithm is parallel" and
"the GPU is doing useful work every 4 ms without fail": nine to twelve **separate host-to-device
hand-offs**, each with its own fixed cost, repeated 2000 times per mode. That fixed cost, not the
arithmetic, is this project's subject — which is why "the algorithm" section here is unusually
short relative to its GPU-mapping section below.

## The GPU mapping

**Thread-to-data mapping per stage** (each kernel's own header comment in `kernels.cu` gives the
full reasoning; summarized here):

```
stage 1-2  sensor_scale_bias / sensor_smooth : 1 block x kSensorN(512) threads, 1 thread = 1 sample
stage 3    state_predict                     : 1 block x kNX(4) threads, 1 thread = 1 state channel
stage 4    state_correct                     : 1 block x kSensorN(512) threads, shared-mem tree reduce
stage 5    mppi_rollout                      : 1 block x kRollouts(512) threads, 1 thread = 1 rollout
stage 6-7  cost_min / softmin_weight         : 1 block x kRollouts(512) threads, shared-mem tree reduce
stage 8    control_blend                     : kHorizon(16) blocks x kRollouts(512) threads,
                                                blockIdx.x = horizon step t, one independent
                                                reduction per block
```

Every kernel fits in ONE block (`kSensorN`, `kRollouts` ≤ 1024, the sm_75+ per-block thread limit) —
a deliberate simplification (README "the sizing argument") that keeps every reduction a single
shared-memory tree with no cross-block combine step, the simplest form of the pattern 33.01/08.01
teach at larger scale. **Memory hierarchy:** registers hold each rollout's state in stage 5
(08.01's reasoning, reused verbatim); shared memory holds every reduction's working set (stages 4,
6, 7, 8 — 2 KiB per 512-float block, comfortably inside the 48+ KiB/block budget on sm_75); global
memory carries data between kernels (no persistent-kernel residency here — that is project 32.03's
territory, see "Where this sits in the real world"). No `__constant__` memory, no textures — the
tick's data is all small, per-tick-variable buffers, not the kind of broadcast-read-only data that
pattern serves (09.01 uses `__constant__` for exactly that reason; this project does not qualify).

**What a kernel launch actually costs — the path this project's numbers are measuring.** Calling
`my_kernel<<<grid, block>>>(args)` (or `cudaLaunchKernel`) is not "the GPU starts running." The
call: (1) validates arguments and packs them into a command-buffer entry in the **CUDA user-mode
driver**; (2) on Windows, hands that entry to the **WDDM (Windows Display Driver Model) kernel-mode
scheduler**, which batches submissions from all processes sharing the GPU into hardware command
queues — this is the layer with the least visibility from user code, and the layer most often
responsible for the SUBMIT-vs-LATENCY gap this project measures (`t1-t0` vs. `t2-t0` in `main.cu`'s
`run_mode`); (3) the GPU's own command processor dequeues and executes the work when its turn comes.
`cudaMemcpyAsync` walks the same path. **Every one of mode A's twelve calls pays step (1) fully and
usually contributes to a fresh WDDM batch decision**; `cudaGraphLaunch` in modes B/C pays step (1)
**once** for the whole DAG (the driver already validated the topology at `cudaGraphInstantiate`
time) and hands WDDM a *single* pre-built command sequence — which is exactly why submit time drops
(README's measured ~45%) even though the same eleven-to-twelve GPU-side operations still execute.

**What CUDA Graphs actually are.** A `cudaGraph_t` is a **DAG description**: nodes (kernel launches,
memcpys, memsets, host callbacks, even child graphs) and the dependency edges between them,
entirely inert data until instantiated. Two ways to build one, both used in this project:

- **Stream capture** (`cudaStreamBeginCapture` → normal API calls on that stream, now *recorded*
  instead of executed → `cudaStreamEndCapture` returns the `cudaGraph_t`) — the convenient path,
  used for mode B via `capture_graph_stream()`. Stream-*issue* order becomes graph-*dependency*
  order: because `submit_naive()`'s twelve calls all target one stream with no forking, the
  captured graph is a straight line, node depending only on the one before it.
- **Explicit construction** (`cudaGraphCreate` + `cudaGraphAddKernelNode`/`cudaGraphAddMemcpyNode1D`
  + manually-specified dependency arrays) — the path `build_graph_setparams()` uses for mode C,
  because it needs a `cudaKernelNodeParams` handle it can hold onto and mutate later (stream capture
  hands you a finished graph, not the individual node-construction structs).

`cudaGraphInstantiate(&graphExec, graph, 0)` is the step that turns the DAG *description* into a
`cudaGraphExec_t` — an **executable** the driver has validated (no cycles, compatible node types,
resources resolved) and uploaded in a form it can replay cheaply. This is the "topology freeze": a
`cudaGraphExec_t`'s *shape* (which kernels, how many nodes, what depends on what) is fixed at
instantiation; only node *parameters* (pointers, scalars, launch dimensions) can change afterward,
via the `cudaGraphExecXxxNodeSetParams` family — never the topology itself. `cudaGraphLaunch`
enqueues the WHOLE pre-validated sequence with one driver call.

**Update vs. recapture.** If a captured node's data lives behind a fixed pointer (this project's
`d_eps`, refreshed by overwriting the *host* source buffer before each launch — both modes B and C
do this for the exploration noise), **no update API is needed at all**: the graph replays the same
memcpy, which faithfully re-reads whatever bytes are currently at the source address. An update API
is only needed when the *node's own parameters* must change — a different device pointer (this
project's mode C: the sensor node's input alternates between two allocations), a different launch
geometry, or (not exercised here) a different kernel function entirely. Two ways to handle that:

- **`cudaGraphExecKernelNodeSetParams`** (mode C's choice): patch ONE node's parameters in the
  already-instantiated `cudaGraphExec_t`, in place. Cost scales with what changed, not with graph
  size — the driver validates and updates only the touched node.
- **Recapture** (destroy the graph, capture/construct a new one, instantiate again): correct, but
  pays the FULL instantiation cost every time — the same topology validation and driver-side upload
  `cudaGraphInstantiate` does once at setup, now repeated every tick. For a DAG this project's size
  (11–12 nodes) that cost is measured in the tens-of-microseconds-to-low-milliseconds range on
  typical hardware (README's Exercise 3 asks you to measure it on yours) — potentially **larger
  than the entire submit-time saving graphs were adopted for**. The lesson: *SetParams is how you
  change a captured graph's DATA every tick; recapture is how you change its SHAPE, and should
  happen rarely, if ever, inside a hot loop.*

**Why mode C's chain is forced linear.** `build_graph_setparams()`'s explicit construction *could*
express the tick's true dependencies (e.g., `state_predict` and `sensor_scale_bias` share no data
and could run concurrently) — CUDA Graphs are one of the few places the GPU is explicitly allowed to
schedule independent nodes concurrently rather than in stream-issue order. This project deliberately
does **not** exploit that: mode B's stream capture, by construction, cannot express any concurrency
a single stream doesn't already have (stream-order IS the dependency order), so letting mode C's
explicit graph run genuinely parallel branches would make its `gpu_exec_ms` incomparable to mode
B's for reasons that have nothing to do with the orchestration technique this project studies — a
confound, not a finding. README's Exercise 2 is exactly this experiment, done honestly, on purpose,
*outside* the controlled comparison.

## Numerical considerations

**Precision.** FP32 throughout, matching every other project in this repo (CLAUDE.md §12); the
rollout kernel's `sinf/cosf` are the precise (not `__sinf/__cosf`) intrinsics, for the same reason
08.01 gives — rollouts integrate angles that pass through a full range, where the fast intrinsics'
error grows. Reductions (stages 4, 6, 7, 8) use `float` shared-memory accumulators with a
POWER-OF-TWO thread count (`kSensorN = kRollouts = 512 = 2^9`), so every tree reduction terminates
in exactly 9 rounds with **no ragged tail to guard** — one motivation for choosing these sizes.

**Determinism and race conditions.** No kernel in this pipeline uses an atomic operation — every
reduction is a synchronized shared-memory tree (`__syncthreads()` between rounds), which is
DETERMINISTIC for a FIXED thread count and FIXED launch configuration: the same floating-point
operations happen in the same order every time the kernel runs, regardless of which host API
sequence triggered the launch. This is the numerical fact the CROSSMODE claim rests on: **modes A,
B, and C launch the identical kernels with the identical grid/block dimensions on the identical
device data**, so their outputs are not merely close, they are the same IEEE-754 bit patterns,
tick after tick, for the whole 2000-tick run — checked in `main.cu`, not asserted. (Contrast with
08.01's host-side softmin blend, which is *also* deterministic but lives on the CPU; this project
moved the equivalent stages — 6, 7, 8 — onto the GPU specifically because a captured graph cannot
pause mid-replay for a host round-trip, per "The GPU mapping" above — an example of orchestration
constraints shaping an algorithm choice, not just wrapping one.)

**Why bit-identical is achievable here, precisely.** Bit-identical GPU-vs-GPU determinism requires
three things this project's design guarantees: (1) no atomics (ordering-dependent), (2) no
data-dependent control flow that could vary block/warp scheduling's effect on results (every branch
here is on a FIXED loop bound or a FIXED thread index, never on kernel *input values*), and (3)
identical launch geometry every time (grid/block dims are compile-time constants from
`kernels.cuh`, never computed from runtime data). Modes A/B/C differ ONLY in orchestration — none of
(1)–(3) — so the bit-identical claim is not an accident of "usually the same," it is a *designed*
invariant, and `main.cu`'s CROSSMODE check is a proof, not a hope.

**Why GPU-vs-CPU still needs a tolerance.** `reference_cpu.cpp`'s tick twin sums `kSensorN=512`
floats SEQUENTIALLY (`state_correct_kernel`'s CPU counterpart), while the GPU's shared-memory tree
reduction sums the same 512 values in a DIFFERENT ORDER (pairwise, tree-shaped) — floating-point
addition is not associative, so these two correct programs can legitimately disagree in the last few
bits. `mppi_rollout`'s 16 chained RK4 steps compound this with `sinf/cosf`-vs-`std::sin/std::cos`
implementation differences (08.01's THEORY.md measures ~1e-7 relative divergence over 50 such steps;
this project's shorter 16-step horizon gives even more headroom). Measured worst case on this
machine: rollout-cost relative deviation `4.130e-07` — four orders of magnitude inside the `1e-3`
gate (see "How we verify correctness").

**Angle wrapping.** The rollout kernel integrates `theta` UNWRAPPED across its 16-step horizon
(08.01's discipline: the cost function uses `cos(theta)`, which never cares about wrapping); this
project's tick pipeline has no OTHER angle-bearing quantity and no plant-stepping stage (see README
"Limitations"), so it has no additional wrap point to define — the state estimate's `theta` channel
simply accumulates via `state_predict`'s constant-velocity integration and is never wrapped,
honestly noted rather than silently glossed over.

## How we verify correctness

Two INDEPENDENT checks, because this project makes two independent claims (the doubled §5 gate its
brief calls for):

1. **VERIFY (the standard §5 gate)** — tick 0's inputs through mode A's GPU path and
   `reference_cpu.cpp`'s plain-C++ twin of the WHOLE tick (not one kernel — the eight-stage
   pipeline end to end), compared on three quantities: the 512 rollout costs (relative tolerance
   `1e-3`, floor `max(1, |cost|)` — 08.01's precedent, justified above), the blended 16-element
   control plan (absolute tolerance `1e-3` N), and the corrected 4-element state estimate (absolute
   tolerance `1e-3`). Measured worst case: `4.130e-07` / `5.066e-07` / `0.0` respectively — passes
   with wide margin, catching indexing/layout/clamp/integrator bugs (which shift results at order
   1, not order `1e-7`) instantly.
2. **CROSSMODE (this project's own claim)** — every one of 2000 measured ticks' published outputs
   (16 control floats + 4 state floats = 40,000 floats total per graph mode) from modes B and C
   compared **bit-for-bit** (`==`, not tolerance) against mode A's. This is the check that makes
   "orchestration never changes the answer" a *verified fact* about this specific run, not an
   assumption baked into the measurement's interpretation. It also indirectly re-verifies mode A
   4000 times over (once per mode-A tick, since mode A is the reference every other mode is
   compared to) — a form of regression testing the study gets for free.

Both checks are FIXED-SEED/deterministic (no statistical comparison needed — this pipeline has no
inherent randomness beyond its RNG-seeded inputs, which are identical across modes by construction),
so there are no edge cases to enumerate beyond "does every tick agree" — which the check answers
exhaustively over the full measured window, not by sampling.

## Where this sits in the real world

- **TensorRT** captures entire inference engines as CUDA Graphs specifically to remove per-layer (or
  per-kernel-fusion-group) launch overhead at inference time — the same submit-time story this
  project measures, at a coarser granularity and with much larger, more numerous kernels than this
  project's deliberately tiny eight.
- **Triton Inference Server** uses CUDA Graphs to cut host-side dispatch latency under high request
  rates, where (like this project) the *submission* cost, not the compute, limits throughput at
  small batch/model sizes.
- **Jetson / embedded deployments** change two things this project's Windows-WDDM numbers do not
  capture: (1) Jetson's unified memory removes the H2D/D2H copy cost entirely for CPU-GPU-shared
  buffers (33.01 PRACTICE §2 names this explicitly), which would remove two of this project's twelve
  naive-mode calls outright; (2) Jetson can run Linux with a real-time-patched kernel and
  `SCHED_FIFO` threads, giving the HOST side of pacing genuine hard-real-time guarantees this
  project's stock-Windows `PacingClock` cannot claim (measured p50 accuracy here is excellent —
  within a few hundred nanoseconds — but the MAX/tail is not bounded, exactly the WDDM-scheduling
  honesty this project's brief asks for).
- **Windows WDDM vs. Linux vs. TCC, concretely.** WDDM (this project's platform) time-slices the GPU
  across processes and the desktop compositor, and its command-submission path includes kernel-mode
  scheduling decisions this project cannot see or control — the most likely explanation for the
  measured LATENCY regression in graph modes (README "The honest result"): a captured graph's
  multiple internal command-buffer entries may cross WDDM scheduling-quantum boundaries in ways a
  naive sequence, submitted incrementally with the driver making per-call batching choices, happens
  not to on this workload size. **TCC (Tesla Compute Cluster) mode**, available on workstation/data-
  center GPUs (not this project's consumer RTX 2080 SUPER), removes the WDDM layer entirely — CUDA
  Graphs' submit-time advantage is well-documented to translate more directly into latency
  improvements there, precisely because the confounding scheduler this project's numbers are
  fighting is absent. **Linux (non-Jetson, non-RT)** sits between: no WDDM, but still a general-
  purpose OS scheduler for the host thread; **Linux + PREEMPT_RT** (or an isolated `SCHED_FIFO`
  core) is what removes the HOST-side jitter this project's `Sleep()`-based pacing cannot. The
  honest summary: **this project's numbers are WDDM numbers**; re-running the identical code on
  TCC/Linux/Jetson would very plausibly change which claims hold, and that platform-dependence is
  itself the lesson SYSTEM_DESIGN §1.2 gestures at when it calls kHz-class GPU control "the research
  frontier."
- **Project 32.03 (persistent kernels for microsecond latency)** is the next rung on this exact
  ladder: instead of relaunching (even via a graph) every tick, keep a kernel resident on the GPU in
  a spin-loop, and signal new work through a memory flag the host writes directly — removing the
  launch/graph-launch round trip (and, on TCC/Jetson especially, most of what remains of this
  project's measured submit-time cost) at the price of dedicating an SM continuously and a much
  harder correctness/synchronization story. This project's measured floor (~75–90 µs mean submit,
  ~350–400 µs mean end-to-end latency) is the number 32.03 exists to beat.
- **What the full production version adds beyond this teaching core:** multi-stream graphs
  exploiting real intra-tick parallelism (Exercise 2), `cudaGraphExecUpdate` for larger structural
  changes than `SetParams` handles, CUDA's "graph node priorities" and MPS-aware scheduling for
  sharing a GPU with other processes (32.11's territory), and — for the safety case this loop would
  need before touching real actuators — an independent watchdog that treats a missed deadline as a
  fault, not a statistic (31.x; PRACTICE §3–4).
