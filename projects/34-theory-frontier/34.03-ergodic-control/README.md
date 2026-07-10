# 34.03 — Ergodic control: spectral multiscale coverage (FFT-based — very GPU-friendly)

**Difficulty:** [R&D] research · **Domain:** 34. Theoretical & Research Frontier

> Catalog bullet (source of truth, verbatim): `Ergodic control: spectral multiscale coverage (FFT-based — very GPU-friendly) [R&D]`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

Most "coverage" robots sweep a workspace uniformly — a lawnmower, a raster scan — because it is
simple to implement and easy to verify. But information is rarely uniform: two survey sites matter
more than the empty ground between them, one patch of crop is stressed and the rest is fine, a
search area has a highest-probability zone. **Ergodic control** steers a robot so its **time-averaged
occupancy statistics** match a target information density `phi(x)`, spending time in proportion to
where the information is — without ever planning a path. This project implements **Spectral
Multiscale Coverage (SMC)**, the classic ergodic-control algorithm (Mathew & Mezic, 2011), for a
single 2-D, first-order agent exploring `[0,1]^2` against a synthetic two-hotspot target density. The
target's Fourier (cosine) coefficients are computed **once**, on the GPU, via a discrete cosine
transform built from a single `cufftPlan2d`/`cufftExecZ2Z` call — the catalog bullet's named
"FFT-based, very GPU-friendly" hook. Every control step then updates all 1,024 spectral modes **in
parallel, one GPU thread per mode**, and steers by the negative gradient of the resulting
coverage-mismatch metric. The demo runs a full closed loop (6,000 steps, 60 s of simulated time),
checks four independent correctness/behavior gates (transform correctness, GPU-vs-CPU numerical
agreement, ergodic-metric decrease, and basin-mass coverage), and proves the controller earns its
keep against a **negative control**: an identical-length lawnmower sweep that ignores `phi`
entirely and ends up with a measurably worse coverage metric.

This is a **reduced-scope teaching version** of an `[R&D]` catalog bullet (CLAUDE.md §2/§13): one
agent, first-order dynamics, a fixed bimodal target, K=32×32 modes, a single 60 s run. The full
research picture — multi-agent SMC, second-order dynamics, obstacles, adaptive mode truncation — is
documented, not implemented; see [Limitations & honesty](#limitations--honesty) and
[THEORY.md §Where this sits in the real world](THEORY.md#where-this-sits-in-the-real-world).

## What this computes & why the GPU helps

Two GPU computations, at very different rates:

- **Once, at startup:** the target density's 1,024 Fourier coefficients `phi_k`, via a **DCT-via-FFT**
  pipeline — mirror the density grid into an even-symmetric extension, run one `cufftPlan2d` /
  `cufftExecZ2Z` (`CUFFT_Z2Z`, double precision) 2-D transform, extract and orthonormalize. Pattern:
  **library call (FFT)**, wrapped by two small **map** kernels (the mirror and the extraction).
- **Every control step (×6,000):** update all 1,024 modes' running time-average `c_k` and their
  contribution to the ergodic-descent direction. Pattern: **map** — one GPU thread per mode, each mode
  fully independent (no shared memory, no atomics). The reduction to a single 2-vector control `u` is
  kept on the **host**, deliberately, the same transparency trade 08.01's MPPI controller makes for its
  softmin blend.

**Honesty about scale:** K=1,024 threads is *small* parallelism next to a perception kernel's millions
of threads — this project's teaching point is the **mapping** (mode-parallel, not point-parallel), not
a large speed-up at this K. THEORY.md §GPU mapping is explicit that the pattern is exactly what a
multi-agent or finer-K extension needs to actually saturate a GPU.

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** the **planning/guidance layer**, specifically the *coverage/exploration*
  sub-problem — it sits between "where is the information?" (perception/mapping) and "what force do the
  motors apply?" (control). Like 08.01's MPPI, it spans planner and controller roles: it never computes
  a discrete path, only a continuous per-step heading.
- **Upstream inputs:** a target information density `phi(x)` over the workspace — in a real system this
  comes from an **exploration/information-gain** front end: **05.15** (frontier detection +
  information-gain scoring for exploration) or **23.09** (exploration: per-viewpoint information gain in
  parallel) would produce exactly the kind of scalar field this project consumes as `phi`. Also needed:
  a state estimate (position `x`), matching `nav_msgs/Odometry`-shaped input elsewhere in this repo.
- **Downstream consumers:** a velocity/heading command to a **local planner or velocity controller**
  (e.g., 23.01's costmap-based local planner, or a direct velocity controller on a holonomic base) —
  ergodic control emits a *desired direction and speed budget*, not motor currents.
- **Rate / latency budget:** this is a **guidance-layer** capability: 10–100 Hz is the realistic band
  (SYSTEM_DESIGN item 1) — far below perception's 30–60 Hz and control's 0.5–1 kHz, since the quantity
  being tracked (a spatial coverage statistic) changes on the timescale of the whole mission, not a
  single control tick. This demo runs its GPU SMC-step kernel in ~0.04–0.1 ms/step, orders of magnitude
  of headroom inside a 100 Hz (10 ms) budget.
- **Reference robot(s):** the **quadrotor** (SYSTEM_DESIGN §2.4) is the natural fit — inspection,
  search-and-rescue, and environmental-monitoring missions are explicitly coverage problems with
  non-uniform information value, and a quadrotor's SWaP-limited compute is exactly where a
  small-but-real GPU workload like this one (versus a CPU-bound alternative) matters.
- **In production:** research/prototype ergodic-exploration stacks (see Prior art below) hold this
  seat; most fielded coverage missions today instead use frontier-based exploration (maximize new-area
  discovery) or hand-tuned waypoint grids — ergodic control's adoption is still largely academic/
  research-stage, which is exactly why this catalog bullet is tagged `[R&D]`.
- **Owning team:** research/autonomy (SYSTEM_DESIGN item 5) — an exploratory capability, not (yet) a
  shipped planner; would sit beside the team owning 05.15/23.09-style information-gain scoring.

## The algorithm in brief

- **Ergodic metric** — compare the trajectory's time-averaged Fourier coefficients `c_k(t)` against the
  target's `phi_k`, weighted by a Sobolev weight `Lambda_k = (1+||k||^2)^-1.5` that prioritizes getting
  large-scale structure right first. → [THEORY.md §The math](THEORY.md#the-math).
- **Cosine (Neumann) basis** — `f_k(x) = cos(k1*pi*x1)*cos(k2*pi*x2)/h_k`, chosen because it has zero
  derivative at the domain walls (no-flux boundary), matching a workspace the agent cannot leave.
  → [THEORY.md §The problem](THEORY.md#the-problem--physics--engineering-first).
- **SMC control law** — `u = -v_max * B/||B||`, `B = sum_k Lambda_k*(c_k-phi_k)*grad f_k(x)`: a
  bang-bang (constant-speed) steepest-descent direction on the ergodic metric.
  → [THEORY.md §The math](THEORY.md#the-math).
- **DCT-via-FFT** — the target's `phi_k` computed once via a mirrored-grid, single 2-D `cufftPlan2d`
  transform, verified against an independent direct cosine-sum CPU oracle.
  → [THEORY.md §The GPU mapping](THEORY.md#the-gpu-mapping).

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/ergodic-control.sln`](build/ergodic-control.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/ergodic-control.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **cuFFT** (`cufft.lib`) — a CUDA Toolkit library computing
the one-shot DCT-via-FFT transform; no fallback needed (it ships with every CUDA Toolkit install this
repo already requires). No other optional dependency.

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at — including the **four artifacts** the demo writes.

## Data

The committed sample is a **scenario**, not recordings: `data/sample/ergodic_scenario.csv` (~0.6 KiB,
synthetic, no RNG) — the agent's start position and the run's step count, plus documentary comments
recording the target-distribution and controller constants (the values actually compiled into the
program live in `src/kernels.cuh`, the single source of truth, CLAUDE.md §12). No public dataset
applies — there is no real-world "correct" information density to download; the target is
constructed, and labeled synthetic everywhere it appears. `scripts/download_data.ps1`/`.sh` are
honest no-ops. Details and the committed file's SHA-256: [`data/README.md`](data/README.md).

## Expected output

Ten stable lines — banner, `PROBLEM:`, `SCENARIO:`, five gate verdicts, `ARTIFACT:`, `RESULT:` —
checked as a subset diff by [`demo/expected_output.txt`](demo/expected_output.txt). Five independent
checks, all measured on the reference machine (RTX 2080 SUPER, sm_75):

1. **TRANSFORM** (the transform-correctness gate) — the GPU DCT-via-cuFFT `phi_k` vs. an independent,
   no-FFT CPU direct cosine-projection oracle: rel tol `1e-6`, **measured worst deviation 1.4e-11**.
2. **VERIFY** (the §5 GPU-vs-CPU gate) — the GPU per-mode SMC update vs. its CPU twin, over a 50-step
   window, every mode's `c_k`/`Bx`/`By`: rel tol `1e-6`, **measured worst deviation 8.3e-12**.
3. **ERGODICITY** — the ergodic metric's window-averaged value must fall ≥5× from the first sixth of
   the run to the last, with at most one window-to-window uptick capped at 25%: **measured 116.5×
   decrease** (2.42e-01 → 2.07e-03), **1 of 5 transitions upticked, +16.9%** (SMC's bang-bang law is not
   a smooth descent — THEORY.md §numerics explains why a single transient uptick is expected, not a bug).
4. **COVERAGE** — the fraction of run-time spent in each hotspot's basin (radius 0.15) must land within
   0.05 (absolute) of that basin's numerically-integrated target probability mass: **measured**
   hotspot 1 `0.1403` vs. target `0.1431` (Δ0.0028), hotspot 2 `0.1705` vs. target `0.1779` (Δ0.0074) —
   comfortably inside the bound.
5. **NEGATIVE-CONTROL** — an identical-length lawnmower (boustrophedon) sweep, ignoring `phi` entirely,
   must reach a final ergodic metric ≥3× worse than SMC's: **measured 4.7×** (lawnmower `1.68e-02` vs.
   SMC `3.57e-03`) — proof the controller is doing something a dense-but-blind path would not.

`RESULT: PASS` requires all five. Timings ("[time]" lines) are teaching artifacts (single-shot,
kernel-only where labeled; the first cuFFT plan pays a one-time module-load cost, documented in
THEORY.md §GPU mapping) — never a benchmark claim.

## Code tour

A guided reading order through `src/`:

1. [`src/main.cu`](src/main.cu) — the whole demo: build + normalize `phi`, the five gates, the closed
   loop, the negative control, and the four artifact writers.
2. [`src/kernels.cuh`](src/kernels.cuh) — the domain, the basis, the target density, the agent, and
   every constant the GPU/CPU paths share — read this before anything else in `src/`.
3. [`src/kernels.cu`](src/kernels.cu) — the GPU pipeline: the DCT-via-FFT (with the full mirror/FFT/
   extract derivation in the file header) and the per-mode SMC-step kernel. The single most interesting
   thing to look at: `smc_step_kernel` — the entire control law's *inner loop* fits in about 20 lines.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the two CPU oracles *and* `integrate_agent_cpu`,
   the plant (the project's single defined domain-boundary reflection point).
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers (copied, not shared — §4 rule).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **Mathew & Mezic (2011), "Metrics for ergodicity and design of ergodic dynamics for multi-agent
  systems"** — the SMC paper this project directly implements; the ergodic metric, the Sobolev
  weighting, and the steepest-descent control law all trace to this work.
- **Miller & Murphey (2013+), trajectory-optimization ergodic control** — reformulates ergodic coverage
  as a receding-horizon trajectory-optimization problem (closer in spirit to 08.01's MPPI than to this
  project's greedy bang-bang law) — the natural "what would a smoother controller look like" next step.
- **The broader ergodic-exploration research ecosystem** — active academic robotics research (multi-
  robot ergodic coverage, ergodic exploration under uncertainty, learned information maps feeding
  ergodic controllers) — this bullet is `[R&D]` precisely because none of this has a dominant, settled
  production implementation the way MPPI (08.01) or A* (06.x) do.
- **05.15 / 23.09 (this repo)** — the information-gain scoring this project's `phi` stands in for; study
  those projects for how a real information density would actually be built from sensor data.
- **cuFFT documentation (NVIDIA)** — the library this project's DCT-via-FFT pipeline is built on; study
  its planning API (`cufftPlanMany`, batching) beyond this project's simplest-possible single-transform
  usage — 03.01's FMCW radar project is this repo's deep-dive into cuFFT batching.

## Exercises

1. **Plot the artifacts:** `demo/out/ergodic_metric.csv` → epsilon vs. t (log scale reveals the
   window-to-window decrease clearly); `demo/out/trajectory.csv` → x1,x2 (does the path visibly
   thicken over the two hotspots?).
2. **Move the hotspots:** edit `kMu1X/kMu1Y/kMu2X/kMu2Y` in `src/kernels.cuh`, rebuild, and confirm the
   COVERAGE gate's printed target masses (and the achieved fractions) move with them.
3. **Break the Sobolev weight:** set `kSobolevS` to 0.1 (barely favors low modes) and to 5.0 (heavily
   favors low modes) and explain the coverage-quality difference from the printed ergodic-metric curve.
4. **Fuse the reduction:** move the host-side `Bx`/`By` sum in `main.cu` onto the GPU (a small
   parallel-reduction kernel over `d_Bx`/`d_By`) and measure whether the per-step download this removes
   is actually a bottleneck at K=1,024 (README's honesty note says it should not be — verify it).
5. **Climb toward the full research version:** add a second agent (two independent `x`, two independent
   `S_k` buffers, one shared `phi_k`) and explore how you would extend the ergodic metric to
   multi-agent coverage (THEORY.md §Where this sits in the real world names the open problem).

## Limitations & honesty

- **Reduced-scope `[R&D]` teaching version (CLAUDE.md §2/§13):** this project implements ONE agent,
  first-order (single-integrator) dynamics, a FIXED bimodal target density, K=32×32=1,024 modes, and a
  single deterministic 60 s run. The full research version — multi-agent SMC, second-order dynamics,
  obstacle-aware coverage, adaptive/streaming mode truncation, learned information maps — is
  documented in [THEORY.md §Where this sits in the real world](THEORY.md#where-this-sits-in-the-real-world),
  not implemented.
- **Bang-bang control is deliberately simple.** The SMC law used here (Mathew & Mezic's original
  constant-speed, normalized-gradient law) chatters near-ergodic states rather than slowing down
  smoothly; THEORY.md §numerics discusses this and names Miller & Murphey's trajectory-optimization
  reformulation as the smoother alternative.
- **The domain is a normalized, obstacle-free unit square** — no obstacles, no sensor model, no
  dynamics beyond `xdot=u`. [`PRACTICE.md`](PRACTICE.md) discusses what changes on a real platform.
- **Small-K GPU honesty:** at K=1,024 modes and a single agent, the GPU is not the bottleneck for
  anything in this demo — the teaching point is the *mapping*, not a measured speed-up (README
  §What this computes is explicit about this).
- **Timings are teaching artifacts** — single-shot, one machine, kernel-only where labeled; the first
  cuFFT plan call pays a one-time module-load cost (visible in repeated runs' `[time]` lines).
- **Sim-only; the [R&D] frontier caveat (CLAUDE.md §1):** this project's output (`u`, a velocity
  command) is the archetype of a signal that could command real hardware. Everything here ran only in
  simulation against a synthetic target; nothing is safety-certified, and any real-robot use would need
  the full testing ladder in [`PRACTICE.md §3`](PRACTICE.md) plus an independent safety envelope.
