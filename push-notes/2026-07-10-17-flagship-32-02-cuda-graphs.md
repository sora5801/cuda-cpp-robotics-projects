# Push note — 2026-07-10-17: flagship 32.02 cuda graphs

> Push-note per CLAUDE.md §7.1 — written **before** the push and included **in** it, so the
> repository always explains its own latest state.

## Summary

Flagship 32.02 — **CUDA Graphs for fixed-rate perception-control loops** — opens the final
flagship batch (**33/505; 33 of 36**). A realistic 12-call robot tick (sensor preprocessing →
fusion → a miniature 08.01-style MPPI rollout → reductions → control blend) runs 2,000 times at a
paced 250 Hz in three modes: naive per-kernel stream launches, a stream-captured CUDA Graph, and
an explicitly-constructed graph with per-tick `cudaGraphExecKernelNodeSetParams` updates. The
outputs are **bit-identical across all three modes** (determinism by design), and the measurements
tell the honest story the project exists to teach: graphs cut host **submit** time ~47% (150 →
80 µs mean; p99 501 → 229 µs) — the reliable claim — but **end-to-end latency and tail jitter got
slightly worse, not better, on this Windows/WDDM consumer-GPU setup**, reported exactly as
measured with the WDDM queueing explanation and the TCC/Linux/Jetson contrast documented. The
methodology (hybrid sleep+spin pacing with `Sleep(1)`'s ~1.7 ms overshoot measured live, event
timelines vs host clocks) is the durable lesson.

## What changed

- **[projects/32-embedded-systems-infra/32.02-cuda-graphs-for-jitter-free-fixed-rate/](../projects/32-embedded-systems-infra/32.02-cuda-graphs-for-jitter-free-fixed-rate/)** —
  complete: the 8-kernel + 2-copy tick pipeline, three execution modes (every graph API call
  explained per §6.1 rule 6), CPU twin, cross-mode bit-identity gate, measurement study +
  histogram/summary artifacts, full README / THEORY / PRACTICE.
- **[docs/STATUS.md](../docs/STATUS.md)** — 32.02 → `done` (**33/505**).

## New projects (didactic blurbs)

**32.02 — CUDA Graphs control loop** (★ beginner, domain 32, flagship). What a kernel launch
actually costs (the user→driver→WDDM path), what graphs actually are (captured DAG → frozen
topology → one launch), update-vs-recapture, and why computation jitter physically matters to a
control loop (a late torque command *is* a disturbance — the phase-margin argument). The honest
negative result is the feature: on WDDM, submit overhead shrinks but the OS still owns the tail —
knowing *where* determinism comes from is the embedded-robotics lesson. The single most
interesting thing to look at: `demo/out/latency_histogram.csv` plotted per mode.

## How to build & run

```powershell
projects\32-embedded-systems-infra\32.02-cuda-graphs-for-jitter-free-fixed-rate\demo\run_demo.ps1
# ~25 s (3 modes x 2000 paced ticks); then plot demo\out\latency_histogram.csv
```

## What to study here

`THEORY.md` §The problem (jitter as a disturbance) and the WDDM/TCC/Jetson scheduling honesty →
`src/kernels.cu` (the graph construction path). First exercise: shrink the tick's kernels further
and watch the submit-overhead fraction — and the graph win — grow.

## Verification

On the owner's machine (NVIDIA GeForce RTX 2080 SUPER, sm_75, **Windows WDDM** — the load-bearing
context, driver 591.86, CUDA 13.3, VS 2026 v145, 2026-07-10), re-run independently by the lead:

- `Release|x64` **and** `Debug|x64` build with **zero errors, zero warnings**.
- `demo/run_demo.ps1` passes end to end: all 11 stable lines matched, exit 0 (~25 s).
- **Correctness gates:** CPU-twin deviation 4.1e-07; all three modes **bit-identical** over
  2,000 ticks × 20 floats each.
- **Measurement gates:** submit means A=150.3 / B=80.3 / C=90.2 µs (gate: graphs ≤ 75% of naive —
  passed with margin); device-exec consistency 1.05× across modes; pacing p50 within 0.4 µs of
  the 4,000 µs target on all modes.
- **Reported, not gated (the honesty):** end-to-end latency means A=358 / B=400 / C=401 µs and
  p99 tails favored the *naive* mode on this WDDM setup — documented with cause and platform
  contrast.
- The builder disclosed accidentally deleting and reconstructing the build files mid-task; the
  lead verified the reconstruction (deterministic GUID matches the scaffold formula exactly).
- `tools/verify_project.py`: **all structural gates PASS**; no changes outside the project folder.

## Known limitations / TODOs

- WDDM consumer-GPU context (the point, stated); 250 Hz software pacing (the 1 kHz frontier and
  persistent kernels are 32.03, named); single-stream linear DAG (the anti-confound choice,
  documented).

## Next push preview

The final three flagships, all [R&D] reduced-scope teaching versions per §2/§13: 34.03 ergodic
control (FFT), 35.01 magnetic microswarm fields, 36.03 lattice-robot kinematics. Then the §11
standards retrospective push.
