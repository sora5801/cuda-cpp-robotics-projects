// ===========================================================================
// util/timer.cuh — GPU (cudaEvent) and CPU (std::chrono) timers
//                  (project 15.16; copied from docs/PROJECT_TEMPLATE
//                  — see util/README.md for the deliberate-duplication rule)
//
// Role in the project
// -------------------
// Every demo in this repository prints a clearly-labeled timing line
// (CLAUDE.md §12). This header provides the two timers those lines use:
//
//   GpuTimer — measures work on the GPU's own timeline via cudaEvents.
//   CpuTimer — measures synchronous host code via std::chrono.
//
// WHY cudaEvents and not the host clock for GPU work?
// ---------------------------------------------------
// Kernel launches are ASYNCHRONOUS: <<<...>>> returns to the CPU immediately,
// usually microseconds later, while the kernel is still running. Wrapping a
// launch in std::chrono therefore measures the LAUNCH OVERHEAD, not the
// kernel — a classic beginner trap that reports absurd 1000x speed-ups.
// cudaEvents are timestamps recorded BY THE GPU, in stream order: record a
// start event, launch, record a stop event, synchronize on the stop event,
// then ask the driver for the elapsed time between the two GPU timestamps.
// (Alternative fix: cudaDeviceSynchronize() before/after host timing — works,
// but events are finer-grained, per-stream, and idiomatic; learn them here.)
//
// Timing discipline (repo-wide): timings are TEACHING ARTIFACTS, never
// benchmark claims. First launches pay one-time init/JIT costs; serious
// measurement warms up and averages. Demos print single-shot numbers and say
// so honestly.
// ===========================================================================
#ifndef PROJECT_UTIL_TIMER_CUH
#define PROJECT_UTIL_TIMER_CUH

#include <cuda_runtime.h>       // cudaEvent_* API
#include <chrono>               // std::chrono::steady_clock for host timing
#include "cuda_check.cuh"       // every CUDA call below is checked, as always

// ---------------------------------------------------------------------------
// GpuTimer — measures elapsed GPU time between begin() and end_ms().
//
// Usage:
//     GpuTimer t;
//     t.begin();                     // records the start event in stream 0
//     my_kernel<<<g, b>>>(...);      // the work being measured
//     float ms = t.end_ms();         // records stop, WAITS for it, returns ms
//
// Notes:
//  * end_ms() SYNCHRONIZES on the stop event — after it returns, all work
//    recorded before the stop event has finished. Callers rely on this
//    (main.cu copies results back immediately after timing).
//  * Events are recorded into the default stream (0), matching the simple
//    single-stream structure of the demos. Multi-stream projects should
//    record into their own streams — TODO(scaffold): adapt if this project
//    uses streams.
//  * Resolution is roughly half a microsecond — plenty for kernel timing.
//  * RAII: the constructor creates the two events, the destructor destroys
//    them, so a GpuTimer cannot leak events even on early returns.
// ---------------------------------------------------------------------------
struct GpuTimer {
    cudaEvent_t start_event;  // GPU timestamp marking the start of the region
    cudaEvent_t stop_event;   // GPU timestamp marking the end of the region

    GpuTimer()
    {
        // Event creation can fail if the context is broken (e.g., after an
        // earlier kernel crash) — checked like everything else.
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
    }

    ~GpuTimer()
    {
        // Destructors must not exit() the program mid-unwind, so these two
        // calls are deliberately unchecked — the events were valid at
        // construction; destruction failing implies a torn-down context
        // where the process is already on its way out.
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    // Non-copyable: copying would double-destroy the events. (Teaching note:
    // any type owning a raw handle should delete or define copy semantics.)
    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    // Mark the start of the timed region (enqueued in stream order — it
    // "happens" on the GPU after all previously enqueued work).
    void begin()
    {
        CUDA_CHECK(cudaEventRecord(start_event, 0));
    }

    // Mark the end, wait for it, and return elapsed milliseconds.
    float end_ms()
    {
        CUDA_CHECK(cudaEventRecord(stop_event, 0));
        // Block the host until the stop event has actually occurred on the
        // GPU — i.e., until everything we are timing has finished.
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float ms = 0.0f;  // elapsed time in milliseconds (float: CUDA's native unit here)
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));
        return ms;
    }
};

// ---------------------------------------------------------------------------
// CpuTimer — wall-clock timer for SYNCHRONOUS host code (the CPU reference).
//
// steady_clock, not system_clock: steady_clock is monotonic — it never jumps
// backwards when NTP or the user adjusts the wall time, so intervals are
// always valid. (The same monotonicity requirement appears all over robotics:
// timestamps in this repo's message structs are monotonic seconds for exactly
// this reason — see docs/SYSTEM_DESIGN.md interface conventions.)
//
// This timer is ONLY correct for code that has finished when the call
// returns — host loops, file I/O. Never time a kernel launch with it (see
// the header comment above for why).
// ---------------------------------------------------------------------------
struct CpuTimer {
    std::chrono::steady_clock::time_point t0;  // start timestamp (monotonic)

    // Capture the start time. (Trivial, but symmetric with GpuTimer::begin
    // so demo code reads uniformly.)
    void begin()
    {
        t0 = std::chrono::steady_clock::now();
    }

    // Return elapsed wall-clock milliseconds since begin(), as double —
    // duration<double, milli> avoids integer truncation for sub-ms work.
    double end_ms() const
    {
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

#endif // PROJECT_UTIL_TIMER_CUH
