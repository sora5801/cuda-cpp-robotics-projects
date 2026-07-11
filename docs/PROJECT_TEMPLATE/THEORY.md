# {{PROJECT_ID}} — {{PROJECT_NAME}}: Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2). Mermaid/ASCII
> diagrams are welcome throughout. Define every symbol, unit, and frame on first use.

## The problem — physics & engineering first

Teach the physical phenomenon and the robotics task from first principles: whatever governs this
project (mechanics, dynamics, electromagnetism, optics, acoustics, thermodynamics, materials), plus
the engineering constraints a real robot imposes — noise floors, tolerances, bandwidth, latency,
thermal limits, vibration, EMI, wear. If this project is purely computational, say so honestly and
teach the physics of its nearest physical carrier.

TODO(scaffold): write the physics-and-engineering-first problem statement.

## The math

The governing equations / formal problem statement, with all notation defined: units (SI), frames
(right-handed, `T_parent_child` transform notation), state-vector layouts, and assumptions.

TODO(scaffold): write the formal problem statement and governing equations.

## The algorithm

Step-by-step description of the method(s) named in the catalog bullet, with complexity analysis —
serial cost vs. parallel cost — and the key data structures.

TODO(scaffold): write the algorithm walk-through and complexity analysis.

## The GPU mapping

How the algorithm becomes threads/blocks/grids: the thread-to-data mapping, which levels of the
memory hierarchy are used (global / shared / registers / constant / texture) and *why*, occupancy and
bandwidth considerations, and what any CUDA library call (cuBLAS/cuFFT/Thrust/CUB/…) computes and
what it would take to write by hand (no black boxes, CLAUDE.md §1).

TODO(scaffold): write the GPU mapping. (The scaffolded SAXPY placeholder demonstrates the simplest
possible mapping — a grid-stride map over independent elements; see `src/kernels.cu` — replace this
note with the real project's mapping.)

## Numerical considerations

Precision (FP32/FP64), stability, race conditions, atomics, determinism — plus the robotics-specific
hazards: angle wrapping, quaternion normalization drift, stiff ODEs, ill-conditioned Jacobians near
singularities. State which apply here and how the code handles them.

TODO(scaffold): write the numerical-considerations section.

## How we verify correctness

Two tiers, both required (Phase-1 standards retrospective ruling — see the header of
`src/reference_cpu.cpp` for the full statement):

1. **The CPU twin** (`src/reference_cpu.cpp`): element-wise agreement within a documented tolerance,
   and why that tolerance — for stochastic algorithms, the fixed-seed / statistical-comparison
   strategy (CLAUDE.md §5). The twin catches *parallelization* bugs: indexing, races, tails, stale
   memory. It is blind to any bug the two paths share.
2. **Independent gates that do not route through shared code**: a closed-form/analytic known answer,
   a physical invariant (conservation, symmetry, a limit case), and/or a negative control that must
   *fail*. These catch *algorithmic* bugs the twin cannot see — proven necessary in flagship 13.03,
   where an identical bug in both twins passed the twin comparison and was caught only by the
   analytic gate.

Also list the edge cases exercised.

TODO(scaffold): document both tiers — the twin tolerance and the independent gates.

## Where this sits in the real world

How production robotics stacks (the tools named in README "Prior art") do this differently — and for
`[R&D]` topics, the open problems and what the full research version would need beyond this teaching
version.

TODO(scaffold): write the real-world comparison.
