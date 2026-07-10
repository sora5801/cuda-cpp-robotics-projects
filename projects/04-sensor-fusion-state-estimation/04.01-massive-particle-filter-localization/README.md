# 04.01 — Massive particle filter localization (10⁵–10⁶ particles, GPU likelihoods + resampling)

**Difficulty:** ★ beginner · **Domain:** 4. Sensor Fusion & State Estimation

> Catalog bullet (source of truth, verbatim): `★ Massive particle filter localization (10⁵–10⁶ particles, GPU likelihoods + resampling)`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

**A particle filter answers "where am I?" by keeping thousands of guesses alive and letting the
sensor vote.** Given a known occupancy-grid map, noisy wheel odometry, and a noisy 16-beam range
scan every 100 ms, this project runs a bootstrap Monte Carlo Localization (MCL) filter: a cloud of
K pose hypotheses ("particles") is pushed through the motion model, scored against the scan by
ray-casting what each hypothesis *would* see, and resampled toward the hypotheses that scored
well. The catalog's 10⁵–10⁶ particle range is not a stretch goal — it is the point: a CPU manages
thousands of particles per scan, a GPU manages a million, and that headroom is what turns MCL from
"usually works" into "recovers from a bad initialization or a symmetric corridor almost every
time." The demo closes the loop for 120 steps around a rounded-square path and writes
`demo/out/trajectory_est.csv` — plot the estimate against ground truth and watch a 0.3 m initial
cloud collapse onto the true path within the first few scans. Every component in the catalog bullet
is implemented: GPU predict, GPU (ray-cast) likelihoods, and resampling — the last one runs on the
**host** in this teaching version (O(K), 40 lines of plain C++ worth reading before you optimize
it away; the GPU prefix-sum resampler is README Exercise 5).

## What this computes & why the GPU helps

Per scan: **predict** perturbs K particles through the odometry twist (cheap, ~10 flops/particle),
then **weight** ray-casts kNumBeams=16 beams up to kMaxRaySteps=64 map cells deep from *every*
particle — up to K×16×64 ≈ 10⁸ occupancy-grid lookups per scan at K=100,000.

- **Pattern:** batched independent evaluation — a **map** over particles (one GPU thread = one
  full pose hypothesis: its motion update *and* its 16-beam ray-cast), with zero interaction
  between particles within a step, by construction. The likelihood evaluation itself is a form of
  **sampling**: each particle is a Monte Carlo sample of the posterior over poses, and the GPU's
  job is to price every sample against the sensor in parallel.
- **Measured reality (RTX 2080 SUPER, sm_75 — a teaching artifact, not a benchmark claim):** the
  isolated weight kernel takes ~0.58 ms at K=100,000 where one CPU core needs ~350 ms (~600×); at
  K=1,000,000 it is ~7.3 ms vs ~3.7 s (~500×). The predict+weight pair together average ~1.0 ms per
  scan over the closed loop at K=100,000 and ~6.7 ms at K=1,000,000 — both comfortably inside the
  100 ms (10 Hz) scan budget, which is exactly why the catalog can ask for 10⁵–10⁶ particles at all.
- **SoA layout, cache-friendly map:** particle poses live as three separate float arrays (not an
  array of `{x,y,th}` structs) so every kernel access is a coalesced 32-float warp read
  (documented once in [`src/kernels.cuh`](src/kernels.cuh)); the sample's 64×64 occupancy grid is
  only 4 KiB, so after the first touch every ray-march lookup is an L1/L2 hit — the deliberate
  opposite of a cache-hostile access pattern (THEORY.md §GPU-mapping discusses what changes with a
  building-scale map that does *not* fit cache).

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** the **state estimation / world model** layer (SYSTEM_DESIGN §1) —
  specifically **localization against a known map**: it consumes raw sensor data and produces the
  "where am I?" belief that prediction, planning, and control all build on. It sits directly
  downstream of perception (the range scan) and upstream of everything else.
- **Upstream inputs, as message-shaped interfaces (SYSTEM_DESIGN §3.6):** an `OccupancyGrid` (the
  known map — loaded once, static for this project; domain 05 SLAM produces it live), a `Twist`
  (the noisy odometry measurement, one per scan), and a reduced `sensor_msgs/LaserScan` analogue
  (16 beams instead of hundreds — README §Limitations owns the reduction) at the scan rate.
- **Downstream consumers:** a pose estimate shaped like a `T_map_base` transform (plus an implicit
  covariance carried by the particle cloud's spread) at scan rate, consumed by planning (global
  route + local trajectory) and, after odometry-rate interpolation, by control.
- **Rate / latency budget:** this filter runs at the SYSTEM_DESIGN §1.1 "LiDAR → perception/mapping"
  row (10–20 Hz, <100 ms; here 10 Hz, kDt=0.1 s, matching the sample). Production localization
  stacks publish at that same scan rate but blend with wheel odometry between scans to hand control
  a smoothed, higher-rate (100–400 Hz) pose — our GPU kernels use only ~1–7 ms of that 100 ms
  budget even at the catalog's largest K, so particle count is not what limits this filter's rate;
  the host-side download/resample round-trip is (README Exercise 5 removes it).
- **Reference robot(s):** the **warehouse AMR** (SYSTEM_DESIGN §2.1) most directly — its
  LOCALIZATION & MAPPING block is named "particle filter / scan matching vs. site map" verbatim,
  and this project *is* Chain A's (§4.1) localization stage. The **autonomous-vehicle stack**
  (§2.5) also localizes against a known (HD) map, though at that scale scan-matching against a
  point-cloud map (project 02.06) usually does the heavy lifting, with MCL-style filters as a
  fallback/relocalization method.
- **In production:** ROS 2 Nav2's `nav2_amcl` (CPU, adaptive KLD-sampling, typically hundreds to a
  few thousand particles) is the direct production descendant most AMRs run today; GPU-MCL research
  variants exist for exactly the reason this project demonstrates — more particles, cheaply.
  Modern high-resolution LiDAR often favors scan-matching (ICP/NDT, project 02.06) for its
  sample efficiency, with particle filters kept for global/kidnapped-robot relocalization.
- **Owning team:** controls & autonomy — state estimation sub-team (SYSTEM_DESIGN §5.1); adjacent
  to perception (owns the LiDAR driver and its calibration) and simulation (owns the map/sensor
  models this filter trusts).

## The algorithm in brief

Bullet list of the key algorithms this project implements; link to [`THEORY.md`](THEORY.md) for depth.

- **Bootstrap particle filter (Sampling Importance Resampling / SIR)** — the classic recursive
  Bayes filter approximated by K weighted samples. → [THEORY.md](THEORY.md) §The math.
- **Predict: unicycle motion model + counter-based noise** — each particle's odometry twist is
  perturbed by an in-kernel xorshift32/Box–Muller draw seeded purely by `(particle id, step)` — no
  cuRAND, bit-reproducible. → THEORY §The algorithm, §Numerical considerations.
- **Weight: fixed-step ray-cast range sensor model** — 16 beams per particle, Gaussian likelihood
  of measured vs. expected range, accumulated in log space to avoid underflow. → THEORY §The math,
  §The GPU mapping.
- **Systematic (low-variance) resampling**, on the host — one uniform draw, K evenly-spaced CDF
  probes; O(K), no particle's weight above 1/K can be dropped entirely. → THEORY §The algorithm.
- **Circular-mean pose estimate** — weighted mean of (x, y); heading via `atan2` of the weighted
  mean of (sin θ, cos θ), the correct mean for a wrapped angle. → THEORY §Numerical considerations.

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/massive-particle-filter-localization.sln`](build/massive-particle-filter-localization.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/massive-particle-filter-localization.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none** — CUDA runtime + C++17 standard library only
(the RNG is a hand-rolled in-kernel xorshift32, deliberately not cuRAND; README Exercise 3 swaps
one in to compare).

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) for what you are
looking at.

## Data

What the committed sample is (synthetic by default, per CLAUDE.md §8), how to regenerate or download it,
and its licensing. Details and provenance in [`data/README.md`](data/README.md).

The committed sample is fully **synthetic**: `data/sample/grid_map.txt` (a 64×64, 0.25 m/cell
occupancy grid with border walls and five obstacles) and `data/sample/trajectory_scans.csv` (a
120-step rounded-square drive: ground-truth pose, noisy odometry, and noisy 16-beam scans at every
step), both written by `python scripts/make_synthetic.py` (fixed seed 42). No public dataset
applies — the classic candidates (TUM RGB-D, EuRoC, KITTI) carry the wrong sensor for a planar
range-fan MCL teaching core, and the closed-loop RMSE gate needs *exact* ground truth, which only
synthesis provides for free; `scripts/download_data.ps1`/`.sh` are honest documented no-ops.

## Expected output

What success looks like, and how the GPU result is checked against the CPU reference
(`src/reference_cpu.cpp`) within a documented tolerance. The canonical lines live in
[`demo/expected_output.txt`](demo/expected_output.txt).

Six stable lines — banner, `PROBLEM:`, `SAMPLE:`, `VERIFY: PASS`, `ARTIFACT:`, `RESULT: PASS` —
checked as a subset diff. Two distinct verifications gate the run: **(1)** the §5 GPU-vs-CPU gate —
step 0's predict poses must agree with `reference_cpu.cpp` within absolute 1e-4 (m/rad; measured
worst case ~4.8e-7), and weight log-likelihoods on identical poses within relative 1e-3 (measured
worst case ~2.4e-7); **(2)** the closed-loop estimation check — position RMSE of the estimate vs.
the synthetic (fully-known) ground truth over all 120 steps must beat 0.15 m (measured ~0.019 m at
the default K=100,000). The demo also writes `demo/out/trajectory_est.csv` (git-ignored,
regenerated each run) with ground-truth and estimated pose plus per-step position error — plot it
to *see* the cloud collapse onto the true path.

## Code tour

A guided reading order through `src/`:

1. [`src/main.cu`](src/main.cu) — entry point: arguments, data, CPU reference, GPU path, verification, timing.
2. [`src/kernels.cuh`](src/kernels.cuh) — the kernel interface and why it is shaped that way.
3. [`src/kernels.cu`](src/kernels.cu) — the GPU kernels themselves (the heart of the project).
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the plain-C++ correctness oracle.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, and why they are copied, not shared.

Read `kernels.cuh` first — it is the one-page contract (frames, particle layout, determinism rules)
that `kernels.cu` and `reference_cpu.cpp` both implement line-by-line. The single most interesting
thing in the project is `raycast_range_dev`/`raycast_range_host` in `kernels.cu`/`reference_cpu.cpp`
— the contraction-safe ray-march whose comments explain why a *discontinuous* function (one flipped
map cell can move a range by meters) needs stricter numerical discipline than the smooth motion
update next to it. Then `main.cu`'s closed loop shows how the two kernels plus ~40 lines of host
code (normalize → estimate → resample) become a working filter.

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **Dellaert, Fox, Burgard & Thrun (1999), "Monte Carlo Localization for Mobile Robots"** — the
  paper that introduced particle-filter localization; this project implements its core loop almost
  verbatim, minus the adaptive particle count.
- **Thrun, Burgard & Fox, *Probabilistic Robotics*** — the standard textbook treatment of Bayes
  filters, the beam and likelihood-field sensor models, and resampling strategies; THEORY.md's math
  section follows its notation closely.
- **ROS 2 Nav2's `nav2_amcl`** — the production CPU descendant most warehouse AMRs run today
  (adaptive KLD-sampling, typically hundreds to a few thousand particles); compare its particle
  budget with what a GPU affords here.
- **Project 02.06 (GPU ICP)** — the scan-matching alternative/complement to beam-based MCL; modern
  high-resolution LiDAR often favors it for sample efficiency.
- **Project 07.09 (jump-flooding SDF/Voronoi)** — the distance-field data structure behind
  *likelihood-field* sensor models, the O(1)-per-beam production remedy for this project's O(64)
  ray-march (THEORY.md §Where this sits in the real world).
- **GTSAM** — factor-graph estimation; where localization grows into full SLAM (unknown map) or
  fuses many sensors beyond a single filter step.

## Exercises

3–5 "try this next" extensions for the learner, ordered easiest first.

1. **Plot the artifact.** `demo/out/trajectory_est.csv` → plot `est_x/est_y` over `gt_x/gt_y` and
   watch the cloud hug the rounded-square loop; plot `err_pos_m` vs. `t_s` and find the step where
   the initial 0.3 m spread has visibly collapsed.
2. **Watch weight degeneracy.** Print (or log) the per-step effective sample size (`main.cu`
   already computes it) at a few different `kSigmaZ` values; shrink it until ESS collapses toward 1
   most steps, and connect what you see to THEORY.md §Numerical considerations.
3. **Swap the RNG.** Replace the in-kernel xorshift32/Box–Muller with cuRAND (Philox, one stream
   per particle) in `pf_predict_kernel`; compare statistical quality and kernel time, and document
   the determinism you traded away (kernels.cu references this as the on-device RNG alternative).
4. **Global localization.** Replace the Gaussian initial cloud (pose *tracking*, seeded near the
   known start pose) with a uniform spread over every free cell of the map — the classic
   "kidnapped robot" setup — and measure how many steps convergence takes vs. whether it converges
   at all with K=100,000 (kernels.cuh and main.cu both flag this as the tracking/global-localization
   scoping choice made here).
5. **Move resampling to the GPU.** Implement a parallel prefix-sum (or Metropolis-based) resampler
   as a kernel, eliminating the per-step host round-trip (`main.cu`'s ~1.6 MB/step download at
   K=100,000) that this teaching version accepts for readability (kernels.cuh and main.cu both
   point here as the natural next optimization).

## Limitations & honesty

What is simplified, what is synthetic, and what would differ in production.

- **16 beams, not 360–1080.** Real planar LiDARs return hundreds to over a thousand beams per scan;
  16 keeps the ray-march cost and the printed math tractable for a first read, at the price of
  angular resolution a real filter would have.
- **Localization, not SLAM.** The map is known and static — building it live is domain 05's job.
- **Pose *tracking*, not global localization.** The particle cloud starts Gaussian-distributed
  around the true start pose; the harder "kidnapped robot" problem (uniform initial spread, or
  re-localizing after a big odometry failure) is scoped out and left as Exercise 4.
- **Resampling stays on the host.** Simple and readable at the cost of a per-step device↔host
  round-trip; Exercise 5 removes it, and production GPU-MCL implementations keep the whole filter
  resident on the device.
- **`xorshift32` is a teaching RNG**, not a statistically rigorous one — fine for exploration noise
  at this scale, not appropriate anywhere randomness quality matters beyond "good enough to explore
  pose space and stay reproducible."
- **Fixed-step ray-marching**, not a precomputed likelihood field — O(64) global-memory-adjacent
  lookups per beam here vs. the O(1) table lookup production stacks use (THEORY.md §Where this sits
  in the real world; project 07.09 builds the distance field that lookup needs).
- **Synthetic sensor and motion noise** — zero-mean Gaussian only; no multipath, specular dropout,
  dynamic obstacles/people in the scan, or non-Gaussian wheel-slip odometry failures a real robot
  will see.
- **Sim-validated only, not safety-certified (CLAUDE.md §1, §8).** This project's output is a pose
  *belief*, not a motion command — but a mislocalized robot is exactly what makes downstream
  navigation and control move to the wrong place. Everything here ran only against synthetic data;
  no real-sensor or real-robot claim is made, and any hardware use would demand the full testing
  ladder in [`PRACTICE.md`](PRACTICE.md) §3.
