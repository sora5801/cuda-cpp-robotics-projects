# `src/util/` — shared teaching utilities (copied, never symlinked)

Small, heavily-commented helpers that every project in this repository uses:

| File | What it provides | Read it when… |
|------|------------------|----------------|
| [`cuda_check.cuh`](cuda_check.cuh) | `CUDA_CHECK(call)` and `CUDA_CHECK_LAST_ERROR(what)` — the mandatory, visible error checking around every CUDA API call and kernel launch (CLAUDE.md §6.1 rule 7) — plus `print_device_info()`. | …you meet your first `CUDA_CHECK` in `main.cu`. |
| [`timer.cuh`](timer.cuh) | `GpuTimer` (cudaEvent-based — the *correct* way to time asynchronous GPU work) and `CpuTimer` (std::chrono, for synchronous host code). | …you reach the timing lines in `main.cu` and wonder why host clocks lie about kernels. |

## Why is this folder *copied* into every project instead of shared?

Deliberate, documented duplication — the repo's **self-containment rule** (CLAUDE.md §4):

- **Every project is individually buildable and readable.** You can copy one project folder onto
  another machine, open its `.sln`, and everything is there. No include paths reaching into sibling
  folders, no symlinks (which Git on Windows handles poorly and zips lose entirely).
- **Projects never break each other.** A change to one project's util copy cannot ripple into 500
  others. In production code that trade-off usually goes the other way (one shared library, one fix
  site); here, didactic independence wins — and knowing *when* to make that trade is itself the lesson.
- **The canonical source** is `docs/PROJECT_TEMPLATE/src/util/`. If you improve a util file, improve
  the template copy too, so future scaffolds inherit the fix. Existing projects keep their own copies
  (drift is acceptable; breakage is not).

Everything here sticks to the CUDA toolkit + C++17 standard library — the repo's default dependency
budget (CLAUDE.md §5).
