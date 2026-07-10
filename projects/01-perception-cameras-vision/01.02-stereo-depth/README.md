# 01.02 — Stereo depth: block matching, then Semi-Global Matching (SGM) kernels

**Difficulty:** ★ beginner · **Domain:** 1. Perception — Cameras & Vision

> Catalog bullet (source of truth, verbatim): `★ Stereo depth: block matching, then Semi-Global Matching (SGM) kernels`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

**Two cameras, one baseline, and a question asked 384×288×64 times a frame: "which column of the
right image shows the same point the left image sees at (x, y)?"** This project builds that answer
as a GPU pipeline, taught as a progression exactly as the catalog bullet names it: a **census
transform** turns each pixel into a robust local signature; a **Hamming-distance cost volume** scores
every candidate disparity for every pixel; **block matching** (BM) takes the cheapest answer per
pixel with no context at all; **Semi-Global Matching** (SGM) aggregates that same cost along four 1-D
paths crossing each pixel before choosing — a smoothness prior a real GPU can afford. Both methods run
on a synthetic rectified stereo pair with **exact, dense ground truth** (scripts/make_synthetic.py
authors the scene directly in disparity space and derives the right image by physically-correct
occlusion-aware forward warping), so the demo does not just claim SGM is better — it **measures** it:
block matching gets 63.35% of unoccluded pixels right within one disparity level; SGM gets 97.52%; the
34-point gap is the whole point of the project, visible in the disparity PGMs the demo writes. Both
components named in the catalog bullet are fully implemented (not one reduced-scope stand-in for the
other) — this is the complete bundle.

> **Template placeholder notice — resolved.** The scaffolded SAXPY placeholder has been fully
> replaced. `src/` now implements the census/cost-volume/BM/SGM pipeline described above; the SAXPY
> smoke test no longer exists in this project.

## What this computes & why the GPU helps

Per frame: a 7×7 stencil over ~110K pixels (census), then D=64 independent Hamming distances **per
pixel** for the cost volume (~7.1M comparisons), then four sequential-but-parallel-across-scanlines
passes over that volume for SGM, then several more independent per-pixel decisions (winner-take-all,
left-right check, median filter).

- **Patterns, one per stage** (THEORY.md "The algorithm" walks each): **stencil** (census: every
  thread reads a 7×7 neighborhood), **map** (the cost volume and every winner-take-all/check/filter
  kernel: one thread, one independent answer), and **scan** (SGM aggregation: one thread per
  scanline, marching sequentially because each step needs the previous step's full result — the one
  place in this project that is *not* embarrassingly parallel, and worth understanding precisely
  because of that).
- **Measured reality:** on the reference GPU, census + the full 7.1M-entry cost volume take
  ~0.6–1.5 ms; the 4-direction SGM aggregation, which parallelizes over only ~300–400 scanlines
  instead of ~110K pixels, takes ~60–80 ms — a two-order-of-magnitude gap that is itself a teaching
  result (THEORY.md "The GPU mapping" names it "the parallelism tension SGM is famous for").
- **Layout lesson applied:** the cost volume is stored **D-major** (`cost[d*H*W + y*W + x]`, not the
  seemingly-more-natural pixel-major `cost[(y*W+x)*D + d]`) so the single hottest kernel (cost-volume
  construction) writes coalesced — and the SGM vertical paths get coalescing across threads as a free
  side effect, while the horizontal paths do not; both the win and the honest cost are explained where
  the decision is made, in [`src/kernels.cuh`](src/kernels.cuh).

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** **perception** — the first box after the sensor: it turns two rectified images
  into a per-pixel depth estimate, upstream of everything that needs 3-D structure.
- **Upstream inputs:** a pair of already-**rectified** camera frames (message shape: `Image` ×2, same
  timestamp, same intrinsics, epipolar-aligned rows — SYSTEM_DESIGN's interface conventions). This
  project's self-containment rule (CLAUDE.md §4) means it does not perform rectification itself —
  that is project 01.01/01.07's job in a real pipeline; here the synthetic generator authors an
  already-rectified pair directly, and the loader documents that assumption (`data/README.md`).
- **Downstream consumers:** anything that needs 3-D structure from the disparity map (converted to
  depth/point cloud by the standard `Z = f·B/d` relation — THEORY.md derives it): TSDF fusion (05.01,
  `PointCloud`/depth `Image` in), obstacle/costmap layers (23.01), and object pose estimation feeding
  grasp planning (19.01) on a manipulator work cell.
- **Rate / latency budget:** SYSTEM_DESIGN item 1 puts camera→perception at **30–60 Hz**, budget
  **< 1 frame (16–33 ms) end-to-end**; the manipulator-cell reference design (SYSTEM_DESIGN §2.2)
  further budgets **< 100 ms per view** for the whole vision stage this project's output feeds. This
  demo's own SGM pass (~60–80 ms, one machine, unoptimized scanline parallelism) is a HONEST MISS of
  that budget as shipped — see Limitations and THEORY.md "Where this sits in the real world" for
  exactly what production SGM does differently to hit it.
- **Reference robot(s):** the **6-DoF manipulator work cell** (SYSTEM_DESIGN §2.2: `01.02 Stereo depth
  (SGM) → 19.01 Antipodal grasp scoring`, explicitly named in the composition map) and the
  **autonomous-vehicle stack** (SYSTEM_DESIGN §2.5: domain 01 cameras feed multi-sensor fusion
  alongside LiDAR/radar at 30–60 Hz).
- **In production:** a dedicated ASIC/FPGA block or a heavily-tuned CUDA library (libSGM, OpenCV CUDA
  StereoSGM, NVIDIA VPI) running 8-direction SGM at real-time rates — or, increasingly, an **active**
  depth sensor (structured light / ToF) that skips matching entirely at the cost of range and
  outdoor robustness (THEORY.md "Where this sits in the real world" compares all of these).
- **Owning team:** perception (SYSTEM_DESIGN item 5) — the team that also owns camera calibration,
  rectification, and the depth/point-cloud interface every downstream planning team consumes.

## The algorithm in brief

- **Census transform** — replace each pixel's raw intensity with a 48-bit signature recording
  "am I brighter/darker than each of my 7×7 neighbors?" — robust to the left/right brightness
  mismatch every real stereo rig has. → [THEORY.md](THEORY.md) §The problem, §The algorithm.
- **Hamming-distance cost volume** — for every pixel and D=64 candidate disparities, the popcount of
  the XOR between left and right census signatures. → THEORY §The math, §The GPU mapping (the D-major
  layout argument lives in [`src/kernels.cuh`](src/kernels.cuh)).
- **Block matching (winner-take-all)** — per pixel, the disparity with the smallest raw cost, plus a
  left-right consistency check. The teaching baseline. → THEORY §The algorithm.
- **Semi-Global Matching** — aggregate the cost volume along 4 one-dimensional paths (L→R, R→L, T→B,
  B→T) with a small penalty P1 for a 1-level disparity jump and a larger penalty P2 for anything
  bigger, before winner-take-all + left-right check + a 3×3 median. → THEORY §The math (the energy
  model), §The GPU mapping (why 4 paths, not the production 8).

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/stereo-depth.sln`](build/stereo-depth.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/stereo-depth.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: **none** — CUDA runtime + C++17 standard library only. The
one CUDA "library call" this project makes is `__popcll` (a single hardware population-count
instruction, not a library in the cuBLAS/cuFFT sense); its CPU-side hand-written twin sits in
`src/reference_cpu.cpp` for exactly the reason CLAUDE.md §1 asks for — "what would it take to write by
hand" is answered in code, not just prose.

## Run the demo

One command, from this folder (builds first if needed, runs on `data/sample/`, checks GPU vs CPU):

```powershell
.\demo\run_demo.ps1
```

Linux/CMake equivalent: `./demo/run_demo.sh`. See [`demo/README.md`](demo/README.md) — including
**which artifact PGMs to open** to see BM's streaks and SGM's fix with your own eyes.

## Data

The committed sample is a **synthetic**, fully **dense**, exactly-known rectified stereo pair:
`data/sample/{left,right,gt_disparity,gt_valid}.pgm` (~432 KiB total, generated by
`scripts/make_synthetic.py` from a single documented seed, 42). The scene — a coarse-textured ground
plane at varying depth plus three overlapping fronto-parallel rectangles — and the exact z-buffer
occlusion handling are fully specified in [`data/README.md`](data/README.md), which also documents WHY
this project ships synthetic data rather than Middlebury or KITTI (short version: licensing/
redistribution terms and sparse ground truth; `scripts/download_data.ps1`/`.sh` record the same
decision and point at both datasets' official pages for learners who want to try real photographs
next — README "Exercises").

## Expected output

Eight stable lines — banner, `PROBLEM:`, `DATA:`, `VERIFY:`, `BM:`, `SGM:`, `ARTIFACT:`, `RESULT:` —
checked as a subset diff by [`demo/expected_output.txt`](demo/expected_output.txt). Two genuinely
different kinds of verification, because this project makes two genuinely different claims:

1. **The §5 GPU-vs-CPU gate (VERIFY, `EXACT` equality)** — census, the cost volume, one SGM
   aggregation path, and both final disparity maps (BM and SGM, each run end to end) must match
   [`src/reference_cpu.cpp`](src/reference_cpu.cpp) **bit-for-bit**, not within a tolerance. This is
   possible (and required — see THEORY.md "Numerical considerations") because every operation in this
   pipeline is integer arithmetic: population count and integer min/add, nowhere a float. Measured:
   **0 mismatches** on every one of the five checkpoints.
2. **The ground-truth gate (`BM:`/`SGM:`/`RESULT:`)** — the "good-pixel rate" (fraction of the 95,448
   GT-valid, unoccluded pixels with `|disp - gt| <= 1`) must clear a documented floor for BOTH methods,
   **and** SGM's rate must exceed BM's by a documented margin. Measured on the reference machine (RTX
   2080 SUPER, sm_75): **BM 63.35%**, **SGM 97.52%**, margin **34.17 points** — thresholds in
   [`src/main.cu`](src/main.cu) carry 13–19 points of headroom below every measured value so the gate
   stays robust to legitimate platform differences without ever being able to hide a real regression.

## Code tour

A guided reading order through `src/`:

1. [`src/kernels.cuh`](src/kernels.cuh) — the project's one-place contract: image layout, the
   D-major cost-volume layout argument (read this before anything else — it explains a decision every
   other file assumes), disparity conventions, and every invalid-value sentinel.
2. [`src/main.cu`](src/main.cu) — the whole pipeline in plain sight: load → census+cost (shared) →
   BM → SGM → VERIFY (GPU vs CPU, exact) → ground-truth gate → artifacts.
3. [`src/kernels.cu`](src/kernels.cu) — the heart: six kernel families, each with a one-paragraph
   "big idea" header. The single most interesting kernel: `sgm_path_kernel` — the one place in this
   project where a thread's steps are NOT independent, and the comment explains exactly why and what
   that costs.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the line-by-line CPU oracle, including the
   hand-written SWAR popcount (compare it to `kernels.cu`'s single `__popcll` intrinsic).
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers (copied, not shared — §4 rule).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- **Zabih & Woodfill (1994), "Non-parametric local transforms for computing visual correspondence"** —
  the census transform's origin paper; this project's `census_kernel` is a direct, small-scale
  implementation of its central idea.
- **Hirschmüller (2008), "Stereo Processing by Semi-Global Matching and Mutual Information"** — the
  SGM paper; the path-aggregation recurrence in `sgm_path_kernel` is a direct, 4-direction
  implementation of its energy-minimization scheme (the paper's full version uses 8 and Mutual
  Information as the data term — THEORY.md "Where this sits in the real world" names both gaps).
- **libSGM (fixstars)** and **OpenCV `cuda::StereoSGM`** — production, real-time, 8-direction CUDA SGM
  implementations; compare their kernel structure and memory layout choices against this project's
  once you have read `kernels.cuh`'s layout argument.
- **NVIDIA VPI (Vision Programming Interface)** — ships a hardware-accelerated (PVA/GPU) stereo
  disparity estimator on Jetson platforms; the production answer to "how does this run on a robot's
  actual compute" (PRACTICE.md §2/§3).
- **Middlebury Stereo Evaluation** and **KITTI Stereo** — the standard public benchmarks this
  project's synthetic ground truth deliberately avoids redistributing (data/README.md explains why);
  study their evaluation protocol (good-pixel rate at various tolerances) — this project's own
  ground-truth gate is a small, self-contained version of exactly that protocol.

## Exercises

1. **Open the artifacts.** `demo/out/disparity_bm.pgm`, `disparity_sgm.pgm`, `error_map.pgm` — find a
   streak in the BM map, then find the same region in the SGM map and the error map. Explain in your
   own words WHY the smoothness penalty fixed that specific spot (hint: look at the local texture in
   `data/sample/left.pgm` there).
2. **Break the penalties.** Set `kP2` in `kernels.cuh` to something absurdly small (e.g. 2, close to
   `kP1`) and rebuild — SGM should degrade toward BM's behavior (the smoothness term stops
   distinguishing "small jump" from "big jump"). Then set `kP1`/`kP2` both very large and watch the
   disparity map go flat (over-smoothed, real depth edges erased). Document what you observe against
   the good-pixel rate.
3. **Add the 4 missing directions.** Extend to full 8-direction SGM (the 4 diagonals) — `kernels.cuh`
   and `kernels.cu`'s `sgm_path_kernel` comments name exactly what memory-access pattern a diagonal
   path has in the D-major layout (neither row- nor column-adjacent) and why it needs care. Measure
   the good-pixel-rate improvement (production SGM's headline reason for using 8).
4. **Shared-memory tile the census kernel.** `kernels.cu`'s `census_kernel` comment names the
   redundant global-memory traffic of the current (untiled) version; implement a `__shared__`-memory
   tiled version and measure the kernel-time improvement with `nvidia-nsight-compute` or the existing
   `[time]` lines.
5. **Try a real photograph.** Point `--data` at a real rectified stereo pair (Middlebury's site has
   several with permissive research licenses) and see how the good-pixel rate — and the BM-vs-SGM gap
   — changes on real sensor noise instead of synthetic texture.

## Limitations & honesty

- **Synthetic scene, not a photograph.** The texture is procedural value noise, not a real camera's
  radiometric response — real cameras add sensor noise, JPEG-like compression artifacts, specular
  highlights, and genuine repetitive structure (brick walls, tiled floors) that this scene only
  approximates via its coarse-frequency-dominated noise (`data/README.md` explains the choice).
  Exercise 5 is the natural next step.
- **Rectification is out of scope by design.** This project starts from an already-rectified pair
  (the self-containment rule, CLAUDE.md §4) — a real pipeline's rectification stage (01.01/01.07) is
  documented, not implemented, here.
- **4 SGM paths, not the production 8.** A deliberate, named scoping choice (README "The algorithm in
  brief", THEORY.md "The GPU mapping") — the measured 34-point BM-to-SGM gap already demonstrates the
  headline lesson; Exercise 3 is the full version.
- **The SGM kernel is not latency-tuned.** ~60–80 ms on the reference GPU for a 384×288 frame is far
  outside a real 30–60 Hz camera loop's budget (System context above) — the "one thread per scanline"
  mapping is a deliberately simple, readable teaching structure, and THEORY.md/Exercise 4 name exactly
  what production libraries (libSGM, OpenCV CUDA StereoSGM) do differently to close that gap.
- **No shared-memory tiling anywhere** — every kernel re-reads overlapping neighborhoods from global
  memory independently (named honestly in `kernels.cu`'s comments) in exchange for a kernel body a
  reader can verify by eye against `reference_cpu.cpp` line by line. Exercise 4.
- **Timings are teaching artifacts** — single-shot, one machine, kernel-only where labeled; never a
  benchmark claim (CLAUDE.md §12).
- **Sim-validated only (CLAUDE.md §1):** this project's output is a *depth estimate*, not a motion
  command, so the strongest form of the repo's real-hardware caveat does not apply directly — but any
  downstream consumer that turns this depth into motion (grasp planning, obstacle avoidance) inherits
  the full caveat: nothing here is safety-certified, and camera-only depth in production robotics is
  routinely fused with other modalities (LiDAR, active depth) specifically because monocular/stereo
  vision alone is not treated as sufficient for safety-critical decisions.
