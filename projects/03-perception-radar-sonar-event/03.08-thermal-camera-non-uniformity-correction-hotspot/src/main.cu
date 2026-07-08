// ===========================================================================
// main.cu — entry point for project 03.08 (Thermal camera non-uniformity correction + hotspot detection)
//
// TEMPLATE PLACEHOLDER — replace with this project's real implementation.
// TODO(scaffold): replace the SAXPY placeholder below with the real pipeline
// (keep the overall shape: parse args -> load/make data -> CPU reference ->
// GPU path -> verify -> report). Every replacement point is marked.
//
// Role in the project
// -------------------
// This file is the demo executable. It owns the *orchestration*: argument
// parsing, data creation, host<->device transfers, timing, and the
// GPU-vs-CPU verification that every project in this repo must perform
// (CLAUDE.md §5, §9). The GPU kernels themselves live in kernels.cu; the
// CPU correctness oracle lives in reference_cpu.cpp.
//
// The big idea of the placeholder
// -------------------------------
// SAXPY ("Single-precision A times X Plus Y") computes, element by element,
//
//     y[i] = a * x[i] + y[i]        for i = 0 .. n-1
//
// It is the "hello, world" of GPU computing: every output element is
// independent of every other, so the natural GPU mapping is one thread per
// element — the *map* pattern. SAXPY does almost no arithmetic per byte
// moved (2 FLOPs per 12 bytes), so it is MEMORY-BANDWIDTH BOUND: it measures
// how fast the GPU can stream memory, not how fast it can multiply. That
// makes it the perfect toolchain smoke test — if this builds, runs, and the
// GPU matches the CPU, your VS 2026 + CUDA 13.3 install is healthy.
//
// Read this after / before
// ------------------------
// Read this file FIRST, then kernels.cuh (the interface), then kernels.cu
// (the kernel), then reference_cpu.cpp (the oracle). util/ holds the
// CUDA_CHECK error macro and the timers — skim their headers as you meet
// each one in the code below.
//
// Output contract (load-bearing!)
// -------------------------------
// demo/run_demo.ps1 diffs the stable lines of this program's stdout against
// demo/expected_output.txt. Stable lines are the "[demo]", "PROBLEM:" and
// "RESULT:" lines — they contain NO timings and NO device names, so they are
// deterministic on any GPU. Timing/device lines are prefixed "[time]" /
// "[info]" and are deliberately NOT diffed (they vary run to run). If you
// change a stable line here you MUST update demo/expected_output.txt in the
// same change, and vice versa.
// ===========================================================================

#include <cstdio>    // printf/fprintf — we print a small, stable, greppable report
#include <cstdlib>   // EXIT_SUCCESS/EXIT_FAILURE, strtol for argument parsing
#include <cmath>     // std::fabs for the max-absolute-difference check
#include <vector>    // host-side buffers; RAII beats manual new[]/delete[]

#include "kernels.cuh"           // saxpy_kernel + launch_saxpy (GPU) and saxpy_cpu (oracle)
#include "util/cuda_check.cuh"   // CUDA_CHECK / CUDA_CHECK_LAST_ERROR + print_device_info
#include "util/timer.cuh"        // GpuTimer (cudaEvent-based) and CpuTimer (std::chrono)

// ---------------------------------------------------------------------------
// Problem constants.
//
// DEFAULT_N: 2^20 elements (= 1,048,576). Big enough that the GPU launch
//   overhead (~a few microseconds) is amortized and the timing line means
//   something; small enough (12 MiB of traffic) to run instantly on any GPU.
//   NOTE: the default value is baked into demo/expected_output.txt's
//   "PROBLEM:" line — the demo must run with no arguments to match.
// SAXPY_A: the scalar 'a'. 2.0 is exactly representable in FP32, which keeps
//   the arithmetic clean for teaching purposes.
// TOLERANCE: max allowed |gpu - cpu| per element. CPU and GPU may round the
//   multiply-add differently (the GPU compiler typically fuses a*x+y into a
//   single FMA with ONE rounding; the CPU may round twice) — that difference
//   is at most ~1 ULP here, i.e. ~1e-7 for values of magnitude ~2, so 1e-6
//   is a comfortable-but-honest bound. See THEORY.md "Numerical
//   considerations" for the general story.
// TODO(scaffold): replace with the real project's problem constants.
// ---------------------------------------------------------------------------
static const int   DEFAULT_N = 1 << 20;  // element count (unitless placeholder data)
static const float SAXPY_A   = 2.0f;     // the scalar 'a' in y = a*x + y
static const float TOLERANCE = 1e-6f;    // max |gpu-cpu| accepted as agreement

// ---------------------------------------------------------------------------
// make_input — fill the host input vectors DETERMINISTICALLY.
//
// Purpose:   produce the same bytes on every machine, every run, with no
//            files and no RNG — so the demo is reproducible offline
//            (CLAUDE.md §8: synthetic-first, deterministic demos).
// Params:    n  — element count (must be > 0)
//            x  — [n] OUT: input vector (read-only in the computation)
//            y  — [n] OUT: input/output vector (overwritten by SAXPY)
// Why index-derived values instead of a RNG: the values themselves do not
// matter for a smoke test; deriving them from the index keeps the whole
// program free of seed-management questions. The modulo keeps magnitudes
// small (~0..1) so FP32 rounding stays uniform across the vector.
// Side effects: none beyond writing x and y. Complexity: O(n).
// TODO(scaffold): replace with real data loading from ../data/sample/ (or a
// call into this project's synthetic generator).
// ---------------------------------------------------------------------------
static void make_input(int n, std::vector<float>& x, std::vector<float>& y)
{
    x.resize(static_cast<size_t>(n));
    y.resize(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        // Small, exactly-computable patterns; % keeps values bounded so the
        // element magnitude (and hence the rounding scale) is uniform.
        x[static_cast<size_t>(i)] = 0.001f * static_cast<float>(i % 1024); // in [0, 1.023]
        y[static_cast<size_t>(i)] = 1.0f + 0.0005f * static_cast<float>(i % 512); // in [1.0, 1.2555]
    }
}

// ---------------------------------------------------------------------------
// main — orchestrates the whole demo. Returns EXIT_SUCCESS only when the GPU
// result agrees with the CPU reference within TOLERANCE; a mismatch returns
// EXIT_FAILURE so demo scripts (and CI) can gate on the exit code.
//
// Usage: <exe> [n]
//   n — optional element count override (default 1,048,576). Overriding n
//       changes the "PROBLEM:" line, so the checked demo must run with NO
//       arguments; the override exists for learners to experiment with sizes.
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // ---- 0) Arguments -----------------------------------------------------
    // One optional positional argument: n. We use strtol (not atoi) so a
    // malformed argument is detectable instead of silently becoming 0.
    // TODO(scaffold): replace with this project's real CLI (sizes, input
    // paths, iteration counts, seeds — document every flag here).
    int n = DEFAULT_N;
    if (argc > 1) {
        char* end = nullptr;
        long v = std::strtol(argv[1], &end, 10);
        if (end == argv[1] || v <= 0) {
            std::fprintf(stderr, "usage: %s [n>0]   (default n = %d)\n", argv[0], DEFAULT_N);
            return EXIT_FAILURE;
        }
        n = static_cast<int>(v);
    }

    // Stable line #1 — identifies the demo (diffed by run_demo, see header).
    std::printf("[demo] template placeholder demo for project 03.08 (thermal-camera-non-uniformity-correction-hotspot)\n");

    // "[info]" lines are NOT diffed: device names differ across machines.
    // print_device_info also serves as our "is there a CUDA device at all?"
    // check — it fails loudly through CUDA_CHECK if no driver/GPU is present.
    print_device_info();

    // Stable line #2 — states the problem instance. %d and %.1f formatting is
    // deterministic for these values; keep the text in lockstep with
    // demo/expected_output.txt.
    std::printf("PROBLEM: SAXPY y = a*x + y, n = %d elements, a = %.1f\n", n, SAXPY_A);

    // ---- 1) Data ----------------------------------------------------------
    // Three host buffers:
    //   h_x     — the input vector x (never modified),
    //   h_y_cpu — starts as y, then holds the CPU reference result,
    //   h_y_gpu — a second copy of y; the GPU result is copied back into it.
    // We keep TWO independent y copies because SAXPY updates y in place — the
    // CPU and GPU must each start from identical, untouched inputs.
    std::vector<float> h_x, h_y_cpu;
    make_input(n, h_x, h_y_cpu);
    std::vector<float> h_y_gpu = h_y_cpu;   // deep copy: identical starting y

    // ---- 2) CPU reference (the correctness oracle) ------------------------
    // Runs FIRST so that if the GPU path is broken, we still saw the baseline
    // work. Timing uses a wall-clock CpuTimer (host code is synchronous — no
    // events needed; see util/timer.cuh for why the GPU is different).
    CpuTimer cpu_timer;
    cpu_timer.begin();
    saxpy_cpu(n, SAXPY_A, h_x.data(), h_y_cpu.data());   // defined in reference_cpu.cpp
    const double cpu_ms = cpu_timer.end_ms();

    // ---- 3) GPU path -------------------------------------------------------
    // The canonical 5 steps of every basic CUDA program — spelled out here on
    // purpose, because this sequence recurs in all 500+ projects:
    //   allocate device memory -> copy inputs H2D -> launch kernel(s)
    //   -> copy results D2H -> free device memory.
    // d_ prefix = device pointer, h_ prefix = host pointer (CLAUDE.md §12).
    float* d_x = nullptr;  // device copy of x, [n] floats, read-only in kernel
    float* d_y = nullptr;  // device copy of y, [n] floats, updated in place

    const size_t bytes = static_cast<size_t>(n) * sizeof(float); // size_t: avoids int overflow for large n

    // cudaMalloc can fail with cudaErrorMemoryAllocation if the GPU is out of
    // memory — CUDA_CHECK turns that into a readable message + hard exit.
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));

    // Host-to-device copies. These can fail on invalid pointers/sizes (a
    // programming bug) — again surfaced by CUDA_CHECK, never ignored.
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(),     bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y_gpu.data(), bytes, cudaMemcpyHostToDevice));

    // Time ONLY the kernel with cudaEvents (not the copies) — we want the
    // compute figure, and events measure on the GPU's own timeline (see
    // util/timer.cuh for why host clocks mis-measure async GPU work).
    // NOTE for learners: the very first kernel launch of a process can pay a
    // one-time module-load/JIT cost, so this single-shot number is a teaching
    // artifact, not a benchmark (CLAUDE.md §12). Real projects should warm up
    // and average — TODO(scaffold): do that in the real implementation.
    GpuTimer gpu_timer;
    gpu_timer.begin();
    launch_saxpy(n, SAXPY_A, d_x, d_y);      // wrapper in kernels.cu: grid math + launch + launch-error check
    const float gpu_ms = gpu_timer.end_ms(); // synchronizes on the stop event -> kernel has finished

    // Device-to-host copy of the result. cudaMemcpy is synchronizing here
    // anyway, but the timer's event-sync above already guaranteed completion.
    CUDA_CHECK(cudaMemcpy(h_y_gpu.data(), d_y, bytes, cudaMemcpyDeviceToHost));

    // Free device memory as soon as we are done with it. For a program this
    // small it is not strictly necessary (process exit frees everything), but
    // the habit matters in long-running robot processes where leaked device
    // memory kills you after hours, not seconds.
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    // ---- 4) Verify: GPU vs CPU, element by element -------------------------
    // max_abs_diff is the L-infinity norm of (gpu - cpu) — the strictest of
    // the simple norms, and the easiest to reason about: "no element is off
    // by more than TOLERANCE". double accumulator not needed for a max (no
    // summation, so no accumulation error), float is fine.
    float max_abs_diff = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float d = std::fabs(h_y_gpu[static_cast<size_t>(i)] - h_y_cpu[static_cast<size_t>(i)]);
        if (d > max_abs_diff) max_abs_diff = d;
    }
    const bool pass = (max_abs_diff <= TOLERANCE);

    // ---- 5) Report ----------------------------------------------------------
    // "[time]" lines vary run-to-run and machine-to-machine -> NOT diffed.
    // The speed-up figure is a TEACHING ARTIFACT, never a benchmark claim
    // (CLAUDE.md §12): single-shot, kernel-only vs. single-thread CPU.
    std::printf("[time] CPU reference: %.3f ms\n", cpu_ms);
    std::printf("[time] GPU kernel:    %.3f ms\n", static_cast<double>(gpu_ms));
    if (gpu_ms > 0.0f) {
        std::printf("[time] speed-up (teaching artifact, not a benchmark): %.1fx\n",
                    cpu_ms / static_cast<double>(gpu_ms));
    }

    // Stable line #3 — the verdict. The PASS line contains NO numbers that
    // could vary across GPUs (FMA rounding differs by architecture), so it is
    // byte-identical everywhere; the FAIL line DOES include the measured
    // difference, because when things break you want the number.
    if (pass) {
        std::printf("RESULT: PASS (GPU matches CPU reference, max |gpu-cpu| <= tol 1e-6)\n");
        return EXIT_SUCCESS;
    } else {
        std::printf("RESULT: FAIL (max |gpu-cpu| = %.6e > tol 1e-6)\n",
                    static_cast<double>(max_abs_diff));
        return EXIT_FAILURE;   // nonzero exit -> demo scripts and CI see the failure
    }
}
