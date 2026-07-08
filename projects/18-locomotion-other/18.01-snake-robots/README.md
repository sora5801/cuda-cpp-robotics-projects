# 18.01 — Snake robots: serpenoid gait sweeps coupled to granular sim

**Difficulty:** intermediate · **Domain:** 18. Locomotion — Everything Else

> Catalog bullet (source of truth, verbatim): `Snake robots: serpenoid gait sweeps coupled to granular sim`
>
> Educational project — study material, not production code. Nothing here is safety-certified.
> If this project's output could ever command motion of real hardware, it is **sim-validated only**;
> see [Limitations & honesty](#limitations--honesty).

## Overview

Snake robots: serpenoid gait sweeps coupled to granular sim



TODO(scaffold): Expand this into one honest plain-language paragraph: what this project does, what a
learner will understand after studying it, and what artifact the demo produces. If the catalog bullet
bundles several components, list here which are implemented and which are documented-only (CLAUDE.md §2).

> **Template placeholder notice.** As scaffolded, `src/` contains a tiny fully-working SAXPY
> (`y = a*x + y`) placeholder that compiles, runs, and passes its own GPU-vs-CPU check. It exists to
> validate your toolchain and to demonstrate the repo's coding/commenting standards. It is **not** this
> project's real implementation — every file marks its replacement points with `TODO(scaffold):`.

## What this computes & why the GPU helps

Name the computation, the bottleneck being parallelized, and the parallelization pattern
(map / reduce / stencil / scan / batched-solve / sampling). One short paragraph plus a bullet or two.

TODO(scaffold): Describe the real computation and its GPU pattern. (The placeholder computes SAXPY —
a pure *map*: every output element is independent, so one thread per element saturates memory bandwidth.)

## System context — where this sits in a robot

Where this project lives in the canonical autonomy stack (see
[`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md)) and in the physical/commercial whole
(see [`PRACTICE.md`](PRACTICE.md)).

- **Stack position:** TODO(scaffold): which layer (sensors / perception / estimation / prediction / planning / control / actuation / cross-cutting)?
- **Upstream inputs:** TODO(scaffold): what feeds it, named as message-shaped interfaces (e.g., `PointCloud`, `JointState`)?
- **Downstream consumers:** TODO(scaffold): who consumes the output, and in what message shape?
- **Rate / latency budget:** TODO(scaffold): realistic Hz and per-cycle latency on a real robot (cite SYSTEM_DESIGN.md item 1).
- **Reference robot(s):** TODO(scaffold): which of the five reference robots use this (AMR / manipulator cell / quadruped / quadrotor / AV stack)?
- **In production:** TODO(scaffold): what would replace or surround this component in a shipping stack?
- **Owning team:** TODO(scaffold): one line — where this work lives in a robotics company (SYSTEM_DESIGN.md item 5).

## The algorithm in brief

Bullet list of the key algorithms this project implements; link to [`THEORY.md`](THEORY.md) for depth.

- TODO(scaffold): list the algorithms named in the catalog bullet, one bullet each, with a THEORY.md anchor.

## Build

Requires Visual Studio 2026 (v145 toolset) + CUDA Toolkit 13.3 — full install and troubleshooting steps
live in [`../../../docs/BUILD_GUIDE.md`](../../../docs/BUILD_GUIDE.md).

1. Open [`build/snake-robots.sln`](build/snake-robots.sln) in Visual Studio 2026.
2. Select the `Release|x64` configuration.
3. Build (Ctrl+Shift+B). The executable lands at `build/x64/Release/snake-robots.exe`.

Optional cross-platform path: `CMakeLists.txt` at the project root (a bonus for Linux learners; the VS
solution is the required deliverable, CLAUDE.md §5).

Optional dependencies and their fallbacks: TODO(scaffold): list any (default: none — CUDA toolkit
libraries + C++17 standard library only).

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

TODO(scaffold): describe this project's sample data, how `scripts/make_synthetic.py` generates it, and
(if a public dataset applies) what `scripts/download_data.ps1` fetches and under what license.

## Expected output

What success looks like, and how the GPU result is checked against the CPU reference
(`src/reference_cpu.cpp`) within a documented tolerance. The canonical lines live in
[`demo/expected_output.txt`](demo/expected_output.txt).

TODO(scaffold): describe the real output, the verification tolerance, and any artifact (PNG/CSV/OBJ)
the demo writes. (The placeholder prints a `PROBLEM:` line and a `RESULT: PASS` line; timings vary by
GPU and are deliberately not diffed.)

## Code tour

A guided reading order through `src/`:

1. [`src/main.cu`](src/main.cu) — entry point: arguments, data, CPU reference, GPU path, verification, timing.
2. [`src/kernels.cuh`](src/kernels.cuh) — the kernel interface and why it is shaped that way.
3. [`src/kernels.cu`](src/kernels.cu) — the GPU kernels themselves (the heart of the project).
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the plain-C++ correctness oracle.
5. [`src/util/`](src/util/README.md) — `CUDA_CHECK`, timers, and why they are copied, not shared.

TODO(scaffold): update this tour for the real implementation (add files, name the most interesting kernel).

## Prior art & further reading

The real tools and papers this project teaches toward — study them, do not copy them (CLAUDE.md §4.1).

- TODO(scaffold): 3–6 entries (e.g., PCL, OpenCV CUDA, nvblox, cuRobo, GTSAM, OMPL, Drake, MuJoCo,
  PX4, Nav2, MoveIt), one line each on what to learn from it.

## Exercises

3–5 "try this next" extensions for the learner, ordered easiest first.

1. TODO(scaffold): exercise 1.
2. TODO(scaffold): exercise 2.
3. TODO(scaffold): exercise 3.

## Limitations & honesty

What is simplified, what is synthetic, and what would differ in production.

- TODO(scaffold): list the real limitations and scoping decisions (including any reduced-scope choice
  for `[R&D]` bullets, and the sim-validated-only / not-safety-certified caveat where motion of real
  hardware is conceivable — CLAUDE.md §1, §8).
- As scaffolded, `src/` is the SAXPY toolchain-validation placeholder, not this project's algorithm.
