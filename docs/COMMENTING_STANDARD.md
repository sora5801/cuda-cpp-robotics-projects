# COMMENTING_STANDARD.md — The Canonical Commenting Rubric

> This is the full, binding rubric that `CLAUDE.md` §6 summarizes. Every worker follows it **verbatim**.
> `tools/verify_project.py` enforces a mechanical floor; a human spot-read enforces the real standard.
> When this document and a habit disagree, this document wins. When this document and `CLAUDE.md`
> disagree, `CLAUDE.md` wins — and the conflict goes in a push-note.

---

## 0. Purpose: why we over-comment on purpose

The owner of this repository asked for **"as much comment as possible, explaining what each function
does, what each variable is for, what the logic and thought process is, how everything ties together."**
We take that literally.

This repo is **study material**. The code is not the product — the *understanding* is. Every source
file is a teaching surface, and the test of any file is:

> **Could a sharp stranger — someone who knows C++ but is new to CUDA and new to robotics — open this
> file cold and learn both the CUDA pattern and the robotics concept from it?**

If the answer is no, the file is **unfinished**, no matter how well it compiles or how fast it runs.

Consequences of that goal:

- **Over-comment on purpose.** In ordinary production code, heavy commenting is a smell ("the code
  should speak for itself"). Here it is the point. We are writing an annotated textbook whose exercises
  happen to compile.
- **Comments answer *why*, not *what*.** The compiler already knows *what* the code does. The learner
  needs to know why this approach, why this memory space, why this block size, why this tolerance, and
  what was rejected.
- **Comments are load-bearing.** A wrong comment is a bug of the worst kind: it compiles, it runs, and
  it teaches a falsehood. Treat comment accuracy with the same seriousness as code correctness. When
  you change code, you change its comments **in the same edit**.

### The density floor vs. the density goal

`tools/verify_project.py` enforces a floor of roughly **0.4 non-trivial comment lines per code line**
across `src/`. Understand what that number is:

- The floor is a **safety net** that catches abandoned or rushed files. It is *not* the target.
- Kernel files (`kernels.cu`, `dynamics.cuh`, anything with `__global__`/`__device__` code) typically
  land at **1:1 or beyond** — every design decision in a kernel (thread mapping, memory space,
  synchronization, numeric precision) is a teaching moment.
- "Non-trivial" excludes decoration: a line containing only `//` or `// ---` or a brace does not count.
  Neither does a comment that merely restates the code (see §12). Do not game the ratio; the human
  spot-read (§11) is the gate that matters.
- Host glue (argument parsing, file I/O) may legitimately sit nearer the floor — but even there, the
  *file header* and *function doc-comments* are never optional.

---

## 1. Rule 1 — Every source file opens with a file header block

**The rule.** Every `.cu`, `.cuh`, `.cpp`, `.h`, and every nontrivial script (`.py`, `.ps1`, `.sh`)
begins with a header block stating: what this file is, its role in the project, the key idea, its
inputs and outputs, and reading-order pointers to sibling files. It references the catalog ID so the
file is traceable back to `catalog.json` even if copied out of the repo.

**The exact template** (adapt comment syntax to the language; keep every field):

```cpp
// ============================================================================
// <filename> — <one-line description>            (Project <SS.NN> — <name>)
// ============================================================================
//
// WHAT THIS FILE IS
//   <1–3 sentences: what lives in this file and what it is responsible for.>
//
// ROLE IN THE PROJECT
//   <Where this file sits in the project's data flow: what calls into it,
//    what it calls, what artifact/result it contributes to the demo.>
//
// KEY IDEA
//   <The single most important insight a learner should take from this file.
//    For a kernel file: the parallelization pattern and thread-to-data
//    mapping. For a host file: the orchestration story.>
//
// INPUTS / OUTPUTS
//   In : <data consumed — formats, shapes, units, frames, memory space>
//   Out: <data produced — formats, shapes, units, frames, memory space>
//
// READING ORDER
//   Read after : <file(s) the learner should already have read>
//   Read before: <file(s) that build on this one>
//   Deep dive  : ../THEORY.md §<section> ; system context: ../README.md §4
// ============================================================================
```

**GOOD** (from a stereo-depth project):

```cpp
// ============================================================================
// kernels.cu — SGM cost aggregation kernels        (Project 01.02 — Stereo SGM)
// ============================================================================
//
// WHAT THIS FILE IS
//   The GPU implementation of semi-global matching's path-wise cost
//   aggregation — the part of SGM that turns a noisy per-pixel matching-cost
//   volume into a smooth one by penalizing disparity jumps along scanlines.
//
// ROLE IN THE PROJECT
//   main.cu builds the raw cost volume (census transform + Hamming distance,
//   see cost_volume.cu), then calls aggregate_paths() here once per path
//   direction. The aggregated volume feeds the winner-take-all kernel in
//   wta.cu, which produces the final disparity map the demo writes as PNG.
//
// KEY IDEA
//   Aggregation along a scanline is a RECURRENCE (each pixel depends on its
//   predecessor along the path), so we cannot parallelize along the path.
//   We parallelize ACROSS paths instead: one thread block per scanline, one
//   thread per disparity candidate. That inversion is the whole trick.
//
// INPUTS / OUTPUTS
//   In : d_cost   [H*W*D] uint16, raw matching cost, device global memory
//   Out: d_aggr   [H*W*D] uint16, path-aggregated cost (accumulated in place
//        across the 4 path directions via atomicAdd — see §Rule-7 note below)
//
// READING ORDER
//   Read after : cost_volume.cu (where d_cost comes from)
//   Read before: wta.cu (which consumes d_aggr)
//   Deep dive  : ../THEORY.md §"The algorithm" and §"GPU mapping"
// ============================================================================
```

**BAD** (all-too-common counter-example — fails on every field):

```cpp
// kernels.cu
// CUDA kernels for the project.
// Author: worker-7
// Date: 2026-07-08
```

Why it fails: says nothing a directory listing doesn't; no role, no idea, no I/O, no reading order,
no catalog ID; and author/date lines are noise (git already tracks those).

---

## 2. Rule 2 — Every function gets a doc-comment block

**The rule.** Every function — host or device — is preceded by a block documenting: purpose, each
parameter (with **units, frames, ranges, and ownership/memory space**), return value, side effects,
complexity, and *why the function exists at all*. Kernels document more (second template below).

### 2.1 Host function template

```cpp
// ----------------------------------------------------------------------------
// <function_name> — <one-line purpose>
//
// WHY THIS EXISTS
//   <The role this plays in the pipeline; what would break without it.>
//
// PARAMETERS
//   <name>  : <meaning; UNITS; FRAME if spatial; valid RANGE; OWNERSHIP —
//              who allocates, who frees, host or device pointer>
//   ...
//
// RETURNS
//   <meaning, units, error convention (or `void` + which out-params are set)>
//
// SIDE EFFECTS
//   <device allocations, stream sync, file writes, global state, RNG advance>
//
// COMPLEXITY
//   <O(...) in problem size; memory traffic if it dominates>
// ----------------------------------------------------------------------------
```

### 2.2 Kernel template (everything above **plus** four kernel-specific fields)

```cpp
// ----------------------------------------------------------------------------
// <kernel_name> — <one-line purpose>
//
// WHY THIS EXISTS / PARAMETERS / SIDE EFFECTS  — as in the host template.
//
// LAUNCH CONFIGURATION
//   grid : <dims and the formula, e.g. ceil(K/256) blocks — and WHY>
//   block: <dims — and WHY this size: occupancy, shared-mem budget, warp math>
//
// THREAD-TO-DATA MAPPING
//   <Exactly which datum each thread owns, as a formula, e.g.
//    "thread (bx,tx) owns rollout k = bx*blockDim.x + tx". This single line
//    is the most important comment in any kernel.>
//
// MEMORY SPACES TOUCHED
//   <global reads/writes and whether coalesced; shared memory (size, purpose);
//    registers (per-thread arrays and whether they risk spilling); constant/
//    texture if used — and WHY each space was chosen.>
//
// ATOMICS / SYNCHRONIZATION
//   <every atomic and every __syncthreads(), what race it prevents, and the
//    determinism consequence (float atomics reorder sums — say so).>
// ----------------------------------------------------------------------------
```

**GOOD** (host-side, abbreviated):

```cpp
// ----------------------------------------------------------------------------
// upload_pointcloud — copy a host point cloud into device memory
//
// WHY THIS EXISTS
//   Isolates the host→device transfer so main.cu reads as a pipeline and so
//   the timing harness can measure transfer cost separately from kernel cost
//   (the README's speed-up table needs both numbers).
//
// PARAMETERS
//   h_points : host pointer, [n*3] floats, XYZ interleaved, meters, sensor
//              frame (x-forward/y-left/z-up). Caller owns; read-only here.
//   n        : point count, > 0, <= MAX_POINTS (guarded below).
//   d_points : OUT device pointer, allocated HERE (cudaMalloc, n*3 floats);
//              caller must cudaFree. We allocate inside so callers cannot
//              get the size arithmetic wrong.
//
// RETURNS
//   void — failure is fatal via CUDA_CHECK (teaching code: fail loudly).
//
// SIDE EFFECTS
//   One cudaMalloc + one cudaMemcpy (synchronous, default stream).
//
// COMPLEXITY
//   O(n) bytes over PCIe — for our 60k-point sample this is ~0.7 MB, i.e.
//   negligible next to the ICP iterations that follow.
// ----------------------------------------------------------------------------
```

**BAD:**

```cpp
// Uploads the point cloud.
void upload_pointcloud(const float* h_points, int n, float** d_points);
```

Why it fails: restates the name; no units, no frame, no ownership (who frees `*d_points`?), no side
effects, no reason the function exists. A learner cannot call this safely, let alone learn from it.

---

## 3. Rule 3 — Every non-trivial variable is annotated on first use

**The rule.** On first use, every non-trivial variable gets an inline note: what it represents, its
**units and frame**, and why it has the type/size it does. The mandatory-annotation list — things a
learner *cannot* guess and *will* get wrong without help:

- **Indices** (what space do they index? row-major or column-major? zero-based?)
- **Strides and pitches** (elements or bytes? why padded?)
- **Padded / rounded-up sizes** (padded to what, and why — warp multiple? bank conflicts? alignment?)
- **State-vector layouts** (which component lives at which offset — documented ONCE, referenced everywhere; see §9)
- **Anything in device memory** (the `d_` prefix says *where* it lives; the comment says *what* lives there and its layout)
- **Magic constants** (tolerances, gains, iteration caps — where the number comes from)

**GOOD:**

```cpp
int   n_pad   = (n + 31) & ~31;   // n rounded UP to a multiple of 32 so every
                                  // warp is full; the tail threads are masked
                                  // by the `if (i < n)` guard in the kernel.
float dt_s    = 0.002f;           // integration step [s] = 500 Hz, chosen to
                                  // keep the explicit RK4 stable for this
                                  // system's fastest mode (~40 Hz; THEORY.md
                                  // §numerics derives the stability bound).
float* d_cost = nullptr;          // DEVICE [K] one scalar per rollout: total
                                  // trajectory cost, written once per thread
                                  // (coalesced), consumed by the softmin
                                  // reduction in update_control().
```

**BAD:**

```cpp
int n_pad = (n + 31) & ~31;  // pad n
float dt = 0.002f;           // time step
float* d_cost;               // cost array
```

Why it fails: "pad n" restates the code without the *why*; `dt` has no units and no justification for
the value; `d_cost` gives no shape, no producer/consumer, no layout. Each of these is exactly the kind
of line a learner stares at for ten minutes.

---

## 4. Rule 4 — Narrate the thought process (including rejected alternatives)

**The rule.** Before any non-obvious block of logic, write the *intent* and, where a real alternative
existed, name it and say why it lost. This is the difference between documentation and teaching: the
learner needs to see the decision, not just the outcome. Cross-reference `THEORY.md` when the full
argument lives there.

**GOOD:**

```cpp
// We tile the transform matrix into SHARED memory even though it is only 16
// floats. Alternative considered: read it from global memory in every thread
// (simple, and L2 would likely cache it) — or use __constant__ memory (ideal
// for values uniform across the grid). We chose shared here because this
// project's THEORY.md teaches the shared-memory pattern explicitly, and the
// constant-memory variant is left as README exercise #2. The performance
// difference at this size is negligible; the *pattern* is the payload.
__shared__ float T[16];
if (threadIdx.x < 16) T[threadIdx.x] = d_T[threadIdx.x];
__syncthreads();   // every thread reads T below — nobody may proceed until
                   // all 16 loads have landed (classic load/sync/use shape).
```

**BAD:**

```cpp
// Load the matrix into shared memory.
__shared__ float T[16];
if (threadIdx.x < 16) T[threadIdx.x] = d_T[threadIdx.x];
__syncthreads();
```

Why it fails: says *what* (visible from the code) and hides *why* (the actual lesson). The learner is
left believing shared memory is always the right answer — a falsehood.

Narration duty applies doubly to **anything counter-intuitive**: a serial loop inside a kernel, a
seemingly redundant copy, a `float` where `double` looked safer, an early exit. If a smart reader
would ask "wait, why?", the comment answers before they ask.

---

## 5. Rule 5 — Tie it together: cross-reference relentlessly

**The rule.** Code does not exist in a vacuum. Where a function hands its output to another, **say
so by name**. Where a design decision is explained in `THEORY.md`, cite the section. Where the data
rate or latency budget comes from the robot architecture, cite `README.md` §"System context" and
`docs/SYSTEM_DESIGN.md`. A learner should be able to navigate the whole project by following comment
cross-references alone.

**GOOD:**

```cpp
cost[k] = c;   // one coalesced write per thread. This array is consumed by
               // softmin_weights() in update.cu, which turns costs into the
               // exponential weights of THEORY.md eq. (7). On a real robot
               // this whole rollout+update cycle must fit the 20 ms local-
               // planner budget (README §"System context").
```

**BAD:**

```cpp
cost[k] = c;   // store the cost
```

Cross-reference targets, in decreasing frequency: sibling functions ("feeds X", "consumes Y"),
`THEORY.md` sections (math and GPU-mapping arguments), `README.md` §System-context (rates, budgets,
robot placement), `docs/SYSTEM_DESIGN.md` (the stack), `PRACTICE.md` (the hardware this would run on).
Use stable section names, not line numbers.

---

## 6. Rule 6 — Explain every library call

**The rule.** Nothing is a black box. Any call into cuBLAS, cuFFT, cuRAND, cuSOLVER, cuSPARSE,
Thrust, CUB — or any vendored third-party helper — gets **2–4 lines** covering:

1. **What it computes, mathematically** (the actual equation or operation, with shapes/layouts —
   cuBLAS is column-major and that *will* bite the learner; say it every time it matters).
2. **Why we call the library instead of hand-rolling** (and what hand-rolling would take — often
   "we hand-roll exactly this in project SS.NN" is the best possible pointer).
3. **The shape/layout of inputs and outputs** as this call site uses them.

**GOOD:**

```cpp
// cublasSgemm computes C = alpha*op(A)*op(B) + beta*C in COLUMN-major order.
// Here: C[6xK] = J[6xN] * dq[NxK] — the batched Jacobian-times-velocity
// product of THEORY.md eq. (4). Our arrays are row-major, so we exploit the
// identity (A*B)^T = B^T * A^T and swap the operand order instead of
// transposing data (zero-copy trick, explained in THEORY.md §GPU-mapping).
// Hand-rolling a 6xN GEMM is project 33.01's whole point — read it to see
// what cuBLAS is doing under the hood (tiling, tensor-core dispatch).
CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                         K, 6, N, &one, d_dq, K, d_J, N, &zero, d_v, K));
```

**BAD:**

```cpp
// Multiply the matrices.
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, 6, N, &one, d_dq, K, d_J, N, &zero, d_v, K);
```

Why it fails twice over: no math, no layout warning, no why-a-library — *and* the call is unchecked
(see Rule 7). Same duty applies to Thrust (`thrust::reduce` → "parallel sum via a tree reduction;
we hand-roll the same reduction in kernels.cu so you can compare"), CUB, and cuRAND (document the
generator, the seed, and the reproducibility contract — `CLAUDE.md` §12 determinism).

---

## 7. Rule 7 — CUDA error checking is always visible and always explained

**The rule.** Every CUDA runtime/API call is wrapped in the `CUDA_CHECK(...)` macro (defined and
extensively commented **once** in `src/util/cuda_check.h`, copied from the template — never redefined
per file). Every kernel launch is followed by `CUDA_CHECK(cudaGetLastError())` (launch-config errors)
and, where the result is consumed immediately, `CUDA_CHECK(cudaDeviceSynchronize())` (async execution
errors). Each *guarded call site* carries a short note on **what class of failure it can hit** — that
note is what turns boilerplate into teaching.

**GOOD:**

```cpp
// cudaMalloc fails with cudaErrorMemoryAllocation when the device is out of
// memory — at K=100k rollouts this buffer is 400 KB, tiny for an 8 GB card,
// but the guard documents the failure mode and keeps the habit visible.
CUDA_CHECK(cudaMalloc(&d_cost, K * sizeof(float)));

mppi_rollouts<<<grid, block>>>(d_x0, d_u, d_eps, d_cost);
// A kernel launch is ASYNCHRONOUS and returns no status. Two checks needed:
//   1) cudaGetLastError()      — catches LAUNCH errors (bad grid dims,
//      too much shared memory requested, invalid device function).
//   2) cudaDeviceSynchronize() — blocks until the kernel finishes and
//      catches EXECUTION errors (out-of-bounds access, illegal address).
// Teaching code synchronizes eagerly for clear error attribution; production
// code overlaps work and checks later — the trade-off is THEORY.md §streams.
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
```

**BAD:**

```cpp
cudaMalloc(&d_cost, K * sizeof(float));           // unchecked: silent nullptr on OOM
mppi_rollouts<<<grid, block>>>(d_x0, d_u, d_eps, d_cost);
                                                  // unchecked: an illegal access here
                                                  // surfaces 200 lines later at the
                                                  // next sync, blamed on innocent code
cudaMemcpy(h_cost, d_cost, K * sizeof(float), cudaMemcpyDeviceToHost);
```

Why it fails: unchecked CUDA errors are *deferred*, so the eventual crash misattributes the bug —
the single most time-wasting failure mode a CUDA learner hits. We never model it.

---

## 8. Rule 8 — No commented-out dead code

**The rule.** Comments teach; they do not store graveyards. Commented-out code is banned because it
rots silently, confuses the reading order, and teaches nothing.

- Rejected alternative worth remembering? **Describe it in prose** (Rule 4) — "we rejected X because
  Y" — do not paste its corpse.
- Old version you might want back? That is what **git history** is for.
- The *one* exception: a short (≤ ~5 line) illustrative snippet inside a prose comment that shows
  what the naive/alternative version *would look like*, clearly framed as illustration:

```cpp
// The naive version would re-read d_T from global memory in every thread:
//     float y = d_T[0]*x + d_T[1]*p.y + ...;   // 16 global reads * N threads
// We instead stage T in shared memory once per block (below), cutting global
// traffic by ~blockDim.x. See THEORY.md §GPU-mapping for the arithmetic.
```

That is teaching. This is a graveyard, and it is banned:

```cpp
// kernel<<<grid2, block2>>>(d_a, d_b);   // old launch, keep just in case
// // float tol = 1e-5f;
// // if (err > tol) printf("bad\n");
```

---

## 9. Robotics-specific commenting duties

Robotics code has failure modes that generic CUDA code does not. These duties come from `CLAUDE.md`
§12 and are **binding** wherever the concept appears:

1. **Units in names *and* comments.** SI everywhere. Encode units in identifiers where cheap
   (`dt_s`, `torque_nm`, `omega_rad_s`, `dist_m`) and state them in the annotation regardless. A
   bare `float velocity` is a defect.
2. **Frames are always named.** Every spatial quantity states its frame on first use: `p_world`,
   `v_body` — right-handed, x-forward/y-left/z-up unless a domain standard overrides (then say which
   standard, e.g. "camera frame: z-forward/x-right, per OpenCV convention").
3. **Transforms use `T_parent_child` notation** and the comment spells the reading: `T_world_base`
   is "base expressed in world"; composing `p_world = T_world_base * p_base` reads left-to-right by
   cancelling the inner frame. Say this at least once per file that composes transforms.
4. **Quaternion order `(w, x, y, z)` is restated at every API boundary.** Every function that accepts
   or returns a quaternion says the storage order and the normalization contract in its doc-comment.
   (Half the ecosystem uses `(x,y,z,w)` — Eigen vs. ROS — so this is a live wire; never assume.)
   Where quaternions are integrated, comment where re-normalization happens and why drift accumulates.
5. **Angle wrapping happens at defined points only — and those points are commented.** Every wrap to
   `(-π, π]` is marked ("wrap HERE, once, after integration — wrapping inside the error computation
   would corrupt the shortest-path difference; THEORY.md §numerics"). Unmarked wrapping (or its
   absence) is where robotics math quietly breaks.
6. **State-vector layout is documented exactly once and cross-referenced everywhere.** Every
   `float* state` names its layout in a single authoritative comment (usually in the project's main
   header, e.g. "STATE LAYOUT [NX=4]: x[0]=cart position m, x[1]=cart velocity m/s, x[2]=pole angle
   rad from vertical, x[3]=pole angular velocity rad/s") and every other use points at it ("layout:
   see dynamics.cuh §STATE LAYOUT"). Duplicate layout comments drift; one source of truth does not.
7. **Rates and budgets are cited, not invented.** Where a comment claims "this must run at 1 kHz" or
   "within the 20 ms planning budget", cite README §System-context / `docs/SYSTEM_DESIGN.md` so the
   learner can trace the number to the architecture.

---

## 10. Worked example — the standard, end to end

The following pair — one kernel, one host function — is the density and voice to aim for. (Kernel
adapted from the MPPI flagship 08.01; elisions are marked and would be fully expanded in the real file.)

### 10.1 A fully-commented kernel

```cpp
// ----------------------------------------------------------------------------
// mppi_rollouts — simulate K perturbed control sequences and score each one
//
// WHY THIS EXISTS
//   MPPI steers by SAMPLING: it perturbs the nominal control sequence K ways,
//   simulates each candidate forward through the dynamics, and blends them by
//   cost. This kernel is the expensive middle step — K independent forward
//   simulations — and the reason the algorithm wants a GPU at all: K ~ 10,000+
//   rollouts per control cycle, where a CPU manages dozens.
//
// PARAMETERS
//   x0    : DEVICE [NX] initial state, SI units, layout: dynamics.cuh
//           §STATE LAYOUT. Read-only, identical for all rollouts.
//   u_nom : DEVICE [T*NU] nominal controls from last iteration, row t = time
//           step t. Units: N (cart force). Read-only.
//   eps   : DEVICE [K*T*NU] zero-mean Gaussian perturbations, pre-generated
//           by cuRAND on the host side (seed fixed = 42 for the demo's
//           determinism contract — CLAUDE.md §12). Layout: rollout-major so
//           that consecutive THREADS read consecutive ADDRESSES at each t
//           (coalescing; THEORY.md §GPU-mapping works out the stride math).
//   cost  : DEVICE [K] OUT — total trajectory cost per rollout. Written
//           exactly once per thread; consumed by softmin_weights() next.
//
// LAUNCH CONFIGURATION
//   grid : ceil(K/256) blocks   — enough blocks to cover K with a guard.
//   block: 256 threads          — a solid occupancy default on sm_75..sm_89:
//          multiple of the 32-wide warp, small enough to keep register
//          pressure from capping resident blocks. We measured 128/256/512 in
//          Nsight; 256 won by a hair (numbers in THEORY.md §occupancy).
//
// THREAD-TO-DATA MAPPING
//   Thread (bx, tx) owns rollout k = bx*blockDim.x + tx — one thread per
//   rollout, because rollouts are FULLY independent. No inter-thread
//   communication is needed until the softmin reduction (separate kernel).
//
// MEMORY SPACES TOUCHED
//   Registers: the state x[NX] lives per-thread in registers — private,
//              updated T times, never shared: the fastest memory we have.
//              NX=4 (cart-pole) so no spill risk; re-verified in Nsight.
//   Global   : reads of u_nom/eps (coalesced by the eps layout above), one
//              coalesced write of cost[k]. x0 is read by every thread and
//              served by L2/L1 after the first warp touches it.
//   Shared   : none — nothing here is reused across threads within a block.
//
// ATOMICS / SYNCHRONIZATION
//   None. Independence is the whole design; if you find yourself wanting an
//   atomic in this kernel, the thread mapping has gone wrong.
//
// COMPLEXITY
//   O(T) work per thread, O(K*T) total — perfectly parallel in K.
// ----------------------------------------------------------------------------
__global__ void mppi_rollouts(
    const float* __restrict__ x0,     // [NX] shared start state (see doc block)
    const float* __restrict__ u_nom,  // [T*NU] nominal control sequence
    const float* __restrict__ eps,    // [K*T*NU] per-rollout noise
    float*       __restrict__ cost)   // [K] OUT: per-rollout total cost
{
    // Global rollout index for THIS thread — the mapping promised above.
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;   // guard the ragged last block: K is rarely a
                          // multiple of 256, and the extra threads must not
                          // write out of bounds.

    // Private copy of the state in REGISTERS (see MEMORY SPACES above).
    // Every rollout starts from the SAME measured state x0 — MPPI explores
    // control space, not state space.
    float x[NX];
    for (int i = 0; i < NX; ++i) x[i] = x0[i];

    float c = 0.0f;   // running cost accumulator (unitless — a weighted sum
                      // of squared tracking errors; weights in cost.cuh).

    // March the horizon: x_{t+1} = f(x_t, u_t + eps_t), accumulating stage
    // cost. This loop is SERIAL on purpose — each step depends on the last
    // (it is a simulation) — and that is fine: parallelism lives across k,
    // not across t. (Rule-4 note: parallel-in-time methods exist; THEORY.md
    // §"real world" points at them. Overkill at T=64.)
    for (int t = 0; t < T; ++t) {
        // Perturbed control for this rollout at this step. Index arithmetic:
        // eps is rollout-major [k][t][u] flattened, so consecutive k (i.e.
        // consecutive THREADS in a warp) hit consecutive addresses — the
        // coalescing promised in the doc block.
        float u = u_nom[t] + eps[(k * T + t) * NU];   // [N] cart force

        // One RK4 step of the cart-pole ODE. step_rk4 is __device__ inline
        // in dynamics.cuh, where the equations of motion, the state layout,
        // and the dt_s stability argument are all commented in full.
        step_rk4(x, u, DT_S);

        // Stage cost: penalize deviation from upright + control effort.
        // Formula and weights: THEORY.md eq. (5), tuned in cost.cuh.
        c += stage_cost(x, u);
    }

    // Single coalesced write. softmin_weights() in update.cu reads this next
    // and turns costs into blending weights (THEORY.md eq. (7)).
    cost[k] = c;
}
```

### 10.2 A fully-commented host function

```cpp
// ----------------------------------------------------------------------------
// run_mppi_iteration — one full plan-refine cycle on the GPU
//
// WHY THIS EXISTS
//   Wraps the sample→simulate→blend cycle so that main.cu's control loop is
//   readable as pseudocode. On a real robot this function would be the body
//   of the local-planner node, called at 10–50 Hz (README §System context);
//   in the demo it is called in a loop until the cart-pole swings up.
//
// PARAMETERS
//   ws     : the MppiWorkspace holding ALL device buffers (created once by
//            workspace_create(), freed by workspace_destroy() — we allocate
//            NOTHING per iteration, a real-time habit worth teaching: malloc
//            in a control loop causes latency spikes; PRACTICE.md §3).
//   h_x0   : host [NX] current measured state; layout: dynamics.cuh
//            §STATE LAYOUT. In the demo it comes from the CPU-side simulator;
//            on a robot it would come from the state estimator at 100–400 Hz
//            (docs/SYSTEM_DESIGN.md, estimation→planning boundary).
//   h_u_nom: host [T*NU] IN/OUT — nominal control sequence; refined in place.
//
// RETURNS
//   Best (weighted) cost this iteration — the demo prints it so the learner
//   watches the planner converge; expected_output.txt pins its trajectory.
//
// SIDE EFFECTS
//   Advances ws.rng (cuRAND) — the demo's fixed seed makes the *sequence*
//   reproducible run-to-run, which is what expected_output.txt relies on.
//   Synchronizes the default stream (eager sync = clear error attribution;
//   the streams exercise in README §12 removes it).
//
// COMPLEXITY
//   O(K*T) device work + O(K) reduction + O(T*NU) host↔device traffic.
// ----------------------------------------------------------------------------
float run_mppi_iteration(MppiWorkspace& ws, const float* h_x0, float* h_u_nom)
{
    // --- 1. Upload the two small host inputs. -------------------------------
    // These are tiny (NX=4 and T*NU=64 floats) — transfer cost is noise here,
    // but we still time it in the demo so the learner sees the PCIe cost
    // CLASS exists (it dominates in data-heavy projects like 02.06 ICP).
    CUDA_CHECK(cudaMemcpy(ws.d_x0,    h_x0,    NX * sizeof(float),
                          cudaMemcpyHostToDevice));   // fails only on bad ptr/size
    CUDA_CHECK(cudaMemcpy(ws.d_u_nom, h_u_nom, T * NU * sizeof(float),
                          cudaMemcpyHostToDevice));

    // --- 2. Draw K*T*NU Gaussian perturbations on-device. -------------------
    // curandGenerateNormal fills ws.d_eps with N(0, SIGMA_U) samples using
    // the Philox generator seeded ONCE at workspace creation (seed 42 — the
    // determinism contract, CLAUDE.md §12). Why a library: bringing up a
    // correct, parallel RNG is its own project; here it would only distract
    // from the MPPI lesson. What it would take by hand: THEORY.md §"real
    // world" sketches counter-based RNGs (Philox) in two paragraphs.
    CURAND_CHECK(curandGenerateNormal(ws.rng, ws.d_eps,
                                      (size_t)K * T * NU, 0.0f, SIGMA_U));

    // --- 3. The main event: K rollouts in parallel. --------------------------
    // Launch shape mirrors the kernel doc block: one thread per rollout,
    // 256-thread blocks, guarded ragged tail.
    dim3 block(256);
    dim3 grid((K + block.x - 1) / block.x);   // ceil-divide: cover all K
    mppi_rollouts<<<grid, block>>>(ws.d_x0, ws.d_u_nom, ws.d_eps, ws.d_cost);
    CUDA_CHECK(cudaGetLastError());        // launch errors (bad config)
    CUDA_CHECK(cudaDeviceSynchronize());   // execution errors (bad access);
                                           // eager sync per the doc block.

    // --- 4. Blend: softmin weights + weighted noise average. ----------------
    // Two small kernels in update.cu turn d_cost into weights (THEORY.md
    // eq. (7)) and fold the weighted noise back into d_u_nom. Hand-rolled
    // shared-memory reductions — deliberately, so the learner meets the
    // reduction pattern here before seeing CUB do it in project 33.01.
    float best = softmin_update(ws);   // also returns the blended cost

    // --- 5. Download the refined sequence for the next cycle / the demo log.
    CUDA_CHECK(cudaMemcpy(h_u_nom, ws.d_u_nom, T * NU * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return best;   // main.cu prints this; expected_output.txt pins the trend
}
```

Note what the pair demonstrates: every rule above appears at least once — header-style doc blocks
(1, 2), annotated variables with units and layouts (3), narrated decisions with rejected alternatives
(4), named handoffs and doc cross-references (5), an explained library call (6), visible and
explained error checking (7), an *illustrative* alternative in prose rather than dead code (8), and
the robotics duties: units, seeds, state-layout single-sourcing, rate citations (§9).

---

## 11. Reviewer checklist — run this before handing a project back

A worker self-audits against this list before setting a project to review; the lead spot-checks the
same list at merge. Every box must be honestly checkable.

- [ ] **Stranger test (the only one that really matters):** open `kernels.cu` cold at a random spot —
      can you tell, from comments alone, what each thread owns and why the code is shaped this way?
- [ ] Every source file (including `.cuh`, `.h`, `reference_cpu.cpp`, and scripts) opens with the full
      §1 header block, catalog ID included, reading-order pointers valid.
- [ ] Every function has the §2 doc block; every **kernel** additionally states launch config +
      reasoning, thread-to-data mapping, memory spaces + why, and atomics/sync (or explicitly "none").
- [ ] Every parameter and non-trivial variable states units, frame (if spatial), range, and
      ownership/memory space; all indices, strides, and padded sizes explain their arithmetic.
- [ ] Every state-vector layout is documented in exactly ONE place and cross-referenced everywhere else.
- [ ] Quaternion order `(w,x,y,z)` restated at every API boundary that touches one; every angle-wrap
      point is marked; every transform uses `T_parent_child` with the reading spelled out somewhere in the file.
- [ ] Every non-obvious design decision is narrated, with the rejected alternative named where one existed.
- [ ] Every library call (cuBLAS/cuFFT/cuRAND/Thrust/CUB/vendored) has its 2–4 lines: math, why-not-hand-rolled, shapes/layouts.
- [ ] Every CUDA API call goes through `CUDA_CHECK`; every launch is followed by
      `cudaGetLastError()` (+ sync where taught); guarded sites note their failure class.
- [ ] Zero commented-out dead code; any illustrative snippet inside prose is short and clearly framed.
- [ ] All cross-references resolve: named functions exist, THEORY/README/SYSTEM_DESIGN sections exist,
      relative links work on Windows.
- [ ] Comments match the code *as it is now* — re-read every comment touched by your last round of code
      changes; stale comments are defects.
- [ ] All timing/perf claims in comments come from actual local runs (state the GPU) or cite THEORY.md — never invented.
- [ ] `tools/verify_project.py` passes the density floor — and the kernel files are visibly denser
      than the floor (≈1:1), because the floor is not the goal.
- [ ] Voice check: comments talk to the **learner**, teach in full sentences, and never assume the
      reader was present for this conversation.

---

## 12. What NOT to do — the anti-pattern gallery

Density without substance is worse than silence: it trains the learner to skip comments. Banned:

**1. Comments that restate the code.**

```cpp
i++;                 // increment i
return cost;         // return the cost
```

If deleting the comment loses nothing, delete the comment — or replace it with the *why* that was
missing.

**2. Decoration noise.** Banner ASCII-art beyond the standard header/section separators, comment
lines with no content (`// ----- stuff -----` around a single line), emphasis walls
(`//!!!! IMPORTANT !!!!`). Structure comes from the templates in §1–2, not from ornamentation.
(These lines don't count toward density anyway — see §0.)

**3. Stale comments.** A comment describing last week's version of the code is a **defect**, tracked
and fixed like any bug. The rule that prevents it: *code edits and comment edits travel in the same
change, always.* If you touch a function body, you re-read its doc block before moving on.

**4. Dead-code graveyards.** Covered in §8. Git remembers; comments teach.

**5. Comments addressed to the reviewer instead of the learner.** The audience is always the sharp
stranger of §0 — never the lead, never a fellow worker, never a future version of yourself who
remembers the context.

```cpp
// TODO(worker-3): ask lead if this is ok
// per the discussion, keeping this as-is for now
// hack to make verify pass, revisit later
```

None of these survive to a done project. Genuine open limitations belong in README §13
("Limitations & honesty") or THEORY.md — stated *for the learner*, with the reason: "This uses a
fixed block size; a production version would query occupancy at runtime (see THEORY.md §occupancy)."

**6. Hedging filler.** "This should probably work", "not sure why but this fixes it". If you don't
understand a line, you are not done with it — understand it, then teach it.

**7. Gaming the density floor.** Splitting one comment across five lines, restating code (see #1),
or padding headers to hit 0.4. The floor is a tripwire for the careless, not a target for the
compliant; the spot-read catches gaming, and a gamed file is returned as unfinished.

---

*End of rubric. When in doubt, add the explanation — then make sure it is true.*
