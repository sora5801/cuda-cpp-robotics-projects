# 09.01 — Batched forward kinematics (10⁵ configurations — the foundation for everything above): Theory

> The deep didactic explanation — the "why" behind [`README.md`](README.md). Written for a sharp
> student who knows C++ but is new to CUDA and new to robotics (CLAUDE.md §4.2).

## The problem — physics & engineering first

A serial manipulator is a chain of **rigid links** connected by **revolute joints** — motors that
rotate one link relative to the previous one about a fixed axis. "Rigid" is the physics that makes
kinematics tractable: a machined aluminum or carbon link deflects microns under load, so its
geometry is a *constant* — captured once, at design time, as a fixed transform from the previous
joint's frame to the next joint's frame. FK is then pure geometry: compose those constants with
the current joint angles to locate the end-effector.

Where the constants come from, physically: the CAD model fixes nominal link geometry
(`t_j`, the fixed rotations); machining and assembly tolerances perturb it (±0.02–0.1 mm per link,
which is why *calibrated* kinematic parameters — measured on the real arm with a laser tracker —
replace nominal ones on precision cells); joint encoders measure `q_j` (typically 17–23-bit
absolute encoders at the joint output or motor side + gear ratio, so the *input* to FK carries
its own quantization noise ~10⁻⁵ rad). Engineering constraints that shape the computation: FK sits
inside planner ticks (10–50 Hz) and IK inner loops, and modern *sampling* methods evaluate FK for
**populations** of hypothetical configurations — 10⁵ IK seeds, rollout states, or grasp candidates
per cycle (SYSTEM_DESIGN item 1's budgets). None of those hypothetical configurations ever touch
the physical arm; FK is the *model* half of model-based robotics. What breaks physically if FK is
wrong: everything downstream — a 1 mm FK bias becomes a 1 mm grasp miss (or a collision), which is
why the CPU-oracle discipline here mirrors how real stacks validate kinematics changes.

Frames and units (SYSTEM_DESIGN conventions): SI meters/radians, right-handed frames,
x-forward/y-left/z-up, transforms named `T_parent_child` — "child expressed in parent".

## The math

**Rigid transforms.** A pose is `T = (R, p)`: `R ∈ SO(3)` (3×3 rotation: orthonormal columns,
`det R = +1`) and `p ∈ ℝ³` (meters). Acting on a point: `x_parent = R·x_child + p`. Composition
(the whole algorithm, really):

```
T_a_c = T_a_b · T_b_c   ⇔   R_a_c = R_a_b·R_b_c ,   p_a_c = p_a_b + R_a_b·p_b_c
```

**The chain.** Joint j contributes `T_link(j−1)_link(j)(q_j) = T_fix(j) · Rot(axis_j, q_j)`, where
`T_fix(j) = (R_fix_j, t_j)` is the constant link geometry (a URDF `<origin>`) and
`Rot(a, θ)` is the rotation by `θ` about unit axis `a` — **Rodrigues' formula**:

```
Rot(a, θ) = I + sinθ·[a]ₓ + (1−cosθ)·[a]ₓ² ,   [a]ₓ = skew(a)
```

(from integrating the rotation generator; every revolute joint in robotics is one Rodrigues
evaluation). The end-effector pose is the ordered product

```
T_base_ee(q) = Π_{j=0..NJ−1} T_fix(j) · Rot(axis_j, q_j)
```

**Quaternions.** The output orientation ships as a unit quaternion `q = (w, x, y, z)` (repo order,
documented at every boundary): 4 numbers instead of 9, no orthonormality drift to repair, and the
representation downstream consumers (ROS `geometry_msgs/Pose`) expect. The **double cover** is the
one subtlety: `q` and `−q` are the same rotation, so pose *comparison* must align hemispheres
first — the comparator in `main.cu` flips one side when `⟨q₁,q₂⟩ < 0`. Matrix→quaternion uses
**Shepperd's method**: of the four algebraically-equivalent extraction formulas (each dividing by
one of `4w, 4x, 4y, 4z`), pick the one whose divisor is largest (via trace comparison) so no
near-zero division ever happens — the naive `w = √(1+tr)/2`-only version explodes for rotations
near 180°.

**Cost.** Per joint: one Rodrigues (~30 flops), two 3×3 multiplies (54 FMAs), one rotate-add
(9 FMAs) ≈ 150 flops; ~900 flops for the 6-joint chain plus ~40 for conversion. Batch of K:
`O(K·NJ)` — embarrassingly parallel across K, strictly sequential in j.

## The algorithm

Per configuration k (identical on GPU thread k and in the CPU loop's iteration k — the files are
diffable twins):

1. Initialize `(R, p) ← (I, 0)` — link 0's parent *is* the base frame.
2. For each joint j: `p ← p + R·t_j` (walk out along the link — `t_j` lives in the previous link's
   frame, so rotate it into base first); `R ← R·R_fix_j`; `R ← R·Rot(axis_j, q_j)`.
3. Convert `R` → quaternion (Shepperd), renormalize once, emit the 7-float pose.

Serial complexity `O(K·NJ)`; parallel span `O(NJ)` (one chain). Data structures: three flat arrays
(model `NJ×10`, configurations `K×NJ`, poses `K×7`) with layouts defined once in
[`src/kernels.cuh`](src/kernels.cuh).

## The GPU mapping

```
thread k ──owns── configuration k :  q[k*NJ .. k*NJ+NJ)  →  pose[k*7 .. k*7+7)
grid = ceil(K/256) blocks × 256 threads      (ragged tail guarded)

memory:   registers : (R, p) chain state + temporaries (~40 regs/thread)
          constant  : the robot model  — every thread, same address, same j
          global    : q (read once per joint), pose (written once)
```

**The new concept: `__constant__` memory.** The model is *uniform* — at loop iteration j, all 32
lanes of a warp read the same `c_model[j*10 + i]`. Constant memory is built for exactly this: a
small (64 KB) space with a per-SM cache whose read path **broadcasts** a uniform address to the
whole warp in one transaction — after first touch it costs about as much as a register read. Two
fine-print facts worth owning: (1) if lanes read *different* constant addresses the accesses
serialize (constant cache serves one address per cycle) — uniformity is the whole deal; (2) the
64 KB budget is why `kMaxJoints` is a compile-time cap. Alternatives and why not: plain global
memory works but costs cache hits per load with no broadcast guarantee (Exercise 5 measures the
difference); kernel arguments could carry small models but re-upload per launch and teach nothing.

**Registers, again.** The `(R, p)` state and all temporaries are fixed-size arrays indexed by
literals (the helpers unroll internally), so they live in registers — same reasoning as project
33.01, which is the prerequisite read. The **joint loop does not unroll** (runtime `nj`), and does
not need to: what must stay literal is the *array indexing inside* each iteration, and it does.

**Divergence.** Shepperd's 4-way branch can split a warp (different configurations land in
different cases). Each path is ~10 flops, so the cost is noise — and divergence affects *timing
only*, never values. The FK loop itself is uniform (every thread runs the same j sequence), which
is the well-behaved default this pattern enjoys.

**Coalescing (honest, as always).** Thread k reads `q[k*6+j]`: at fixed j, a warp's addresses are
24 bytes apart — imperfect (each 128-byte segment serves ~5 threads). Pose writes: 28-byte stride.
Both are small next to ~900 flops of arithmetic per thread; the SoA fix is the same story as 33.01
and deliberately not applied here.

**No shared memory** — threads share only the model, and constant memory already handles uniform
sharing better than a shared-memory copy would (no staging, no sync).

## Numerical considerations

- **FP32 across a 6-transform chain.** Orthonormality of `R` erodes by ~1e-6 over six composed
  rotations (each multiply mixes ~1-ulp errors). At ~1 m arm scale that is sub-micron position
  noise — far below real-world calibration error (~0.05 mm). The defensive quaternion
  renormalization at the end restores the unit-norm *output* contract exactly; the norm-invariant
  check in `main.cu` (‖q‖ = 1 ± 1e-4) would catch a conversion bug loudly.
- **Trig implementations differ.** The kernel uses CUDA's `sincosf`, the oracle `std::sin/cos` —
  both ~1-ulp correct, not bit-identical (and the *fast* `__sincosf` intrinsic was deliberately
  rejected: its error grows near |θ| = π, exactly where wrapped angles live). This is one of two
  reasons comparisons are tolerance-based; the other is FMA contraction (explicit `fmaf` on GPU,
  compiler's choice on CPU). Measured combined effect: ~9e-08 m / ~1.8e-07 — the tolerances
  (1e-4/1e-4) sit two-plus orders above it and six-plus orders below any real indexing bug.
- **Angle wrapping:** inputs are wrapped to **(−π, π]** *by the producer* (generator/sample); FK
  itself is 2π-periodic and needs no wrapping internally — stated so the wrap-point rule
  (CLAUDE.md §12: wrap at defined points only) is explicit.
- **The double cover** (see §The math) is handled at *comparison* time, not by canonicalizing the
  output — production consumers (interpolation, filters) handle hemispheres themselves, and
  silently flipping signs inside FK would surprise them.
- **Determinism:** batch inputs come from the same portable xorshift32 as 33.01 (std distributions
  are not bit-portable across standard libraries); the kernel itself has no atomics or reductions,
  so results are bit-stable run to run on a given machine.
- **Not applicable here:** stiff ODEs, ill-conditioned Jacobians (FK has no solve — those hazards
  begin in 09.02/09.05), quaternion *integration* drift (no integration happens).

## How we verify correctness

Two stages, GPU vs the sequential twin ([`src/reference_cpu.cpp`](src/reference_cpu.cpp)) on
identical inputs:

1. **Sample stage** — the committed 6-joint model + 64 configurations (strict loader: unknown
   labels, wrong counts, or non-normalized quaternions/axes abort; the model is validated and
   re-normalized ONCE at load so kernels can trust it).
2. **Batch stage** — 200,000 configurations regenerated deterministically (seed 42) each run;
   exercises many blocks + the ragged tail (the 64-config sample covers the sub-block extreme).

Comparator: position `|Δp| ≤ 1e-4 m` per component; quaternion `|Δq| ≤ 1e-4` per component **after
hemisphere alignment** (flip via dot-product sign — comparing rotations, not sign conventions);
plus the ‖q‖ = 1 ± 1e-4 invariant on **both** paths (catches a broken conversion even if both
paths broke identically-shaped ways). Failure modes by check: chain-composition/indexing bugs →
centimeter-to-meter position errors (caught by the loosest gate at 100× margin); Shepperd branch
bugs → hemisphere-aligned component errors near 180° rotations; conversion normalization bugs →
norm invariant. The demo exits nonzero on any failure; `demo/expected_output.txt` pins the stable
lines.

## Where this sits in the real world

- **cuRobo** (NVIDIA) is this project at industrial strength: batched FK/IK/collision fused into
  planning kernels, SoA layouts, multi-arm batches — its kinematics core is the direct descendant
  of the pattern taught here.
- **Pinocchio** is the CPU gold standard (spatial algebra, one robot at a time, heavily optimized);
  **KDL/MoveIt** the classic ROS path. The batched-GPU rethink exists precisely because their
  one-at-a-time model caps sampling methods.
- **GPU physics engines** (PhysX articulations, Isaac Lab, MuJoCo-MJX) embed FK inside their
  stepping kernels — the fusion argument again: you hand-roll FK so you can fuse it into dynamics
  (09.03), rollout costs (08.01), or reachability scoring (19.08), where library-call granularity
  would kill performance.
- **Real stacks load URDFs** and calibrate parameters per physical robot; the 10-float model row
  here is a teaching-sized URDF `<joint>` — the mapping is one-to-one and worth doing by hand once
  (Exercise 1 starts it).
- **What the full version adds:** kinematic trees (branching, floating base), prismatic/continuous
  joints, mimic joints, per-joint limits, tool/sensor frames, and calibrated parameters — scope
  documented as future siblings in domain 09, none of it changing the core mapping taught here.
