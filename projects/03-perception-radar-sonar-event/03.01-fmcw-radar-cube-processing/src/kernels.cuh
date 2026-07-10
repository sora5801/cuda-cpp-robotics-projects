// ===========================================================================
// kernels.cuh — interface & single-source contract for project 03.01
//               FMCW radar cube processing: range-Doppler-angle FFTs +
//               CA/OS-CFAR detection
//
// Role in the project
// -------------------
// The CONTRACT between main.cu (orchestration), kernels.cu (the GPU
// pipeline), and reference_cpu.cpp (the O(N^2)-DFT oracle). Everything all
// three must agree on — chirp/antenna parameters, the raw-cube MEMORY
// LAYOUT, the complex number representation, the CFAR window geometry, and
// the detection record shape — is defined HERE, once (CLAUDE.md §12).
//
// THE PHYSICAL SCENARIO IN ONE PARAGRAPH (THEORY.md derives all of this):
// An FMCW ("frequency-modulated continuous wave") radar transmits a
// repeated linear "chirp" — a tone that sweeps bandwidth B over Tc seconds
// — and mixes the echo with a copy of the outgoing chirp. For a stationary
// target the result ("beat" signal) is a single tone whose frequency is
// proportional to range; sampled at Ns points per chirp, an FFT along
// those Ns samples turns "frequency" into "range". A moving target's echo
// shifts phase slightly from chirp to chirp; sampling Nc chirps per frame
// and FFT-ing ACROSS chirps turns "phase drift" into "velocity" (Doppler).
// An antenna ARRAY sees the same echo at each of Na elements with a
// geometric phase step set by the arrival angle; FFT-ing across antennas
// turns "phase step" into "angle". Three FFTs, three physical quantities,
// all read off the SAME underlying idea (phase accumulation vs. a
// physical variable) at three different time/space scales — the
// unification THEORY.md teaches explicitly.
//
// RAW-CUBE MEMORY LAYOUT — float2-per-sample, [Ns][Nc][Na] row-major:
//     offset(n, c, a) = n * (kNc * kNa) + c * kNa + a
//   with n = fast-time sample index (0..Ns-1, the RANGE axis),
//        c = slow-time chirp index  (0..Nc-1, the DOPPLER axis),
//        a = antenna/channel index  (0..Na-1, the ANGLE axis, FASTEST).
// This ordering (antenna fastest) is a common real front-end layout: ADC
// channels for all RX antennas of one sample typically arrive interleaved.
// It also gives the RANGE FFT (transformed axis = n, the SLOWEST axis) a
// single clean cuFFT batched call: batch = Nc*Na, istride = Nc*Na,
// idist = 1 (kernels.cu explains why this is the "advanced data layout").
// The DOPPLER FFT (transformed axis = c, the MIDDLE axis) cannot be
// expressed as one cuFFT call in this layout — no single-loop "batch"
// index can linearly re-index a (n, a) pair through a 3-D cube — so it
// is issued as Ns batched calls (batch = Na, istride = Na, idist = 1),
// one per range bin, reusing ONE plan. kernels.cu's header comment walks
// this reasoning in full; it is the project's main "advanced layout"
// lesson (CLAUDE.md §5's cuFFT teaching requirement).
//
// COMPLEX NUMBER TYPE — ComplexF32 (below) rather than cuFFT's own
// cufftComplex: reference_cpu.cpp is deliberately host-only C++ that must
// build with NO CUDA/cuFFT headers (CLAUDE.md §5 — "the correctness
// oracle never depends on nvcc"). ComplexF32 is bit-for-bit layout
// compatible with cufftComplex (two consecutive floats, real then
// imaginary — exactly cufftComplex's {x, y}); kernels.cu documents the
// (safe, deliberate) reinterpret_cast where the two meet.
//
// Read this after: main.cu.  Read this before: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

// ---------------------------------------------------------------------------
// ComplexF32 — a plain, CUDA-header-free complex sample. See the file header
// above for why this exists instead of using cufftComplex directly here.
// ---------------------------------------------------------------------------
struct ComplexF32 {
    float re;   // in-phase (I) component
    float im;   // quadrature (Q) component
};

// ---------------------------------------------------------------------------
// RadarTarget — one injected ground-truth target (the committed scenario's
// per-row shape; loaded from data/sample/targets.csv by main.cu).
// ---------------------------------------------------------------------------
struct RadarTarget {
    float range_m;     // true range (m), 0 <= range_m < kRangeMaxM
    float vel_mps;      // true radial velocity (m/s); POSITIVE = APPROACHING
                        // the radar (closing, range decreasing) — the sign
                        // convention this project fixes throughout; see
                        // THEORY.md "The math" for why the choice is
                        // arbitrary but must be applied consistently.
    float az_deg;       // true azimuth (deg), 0 = boresight (broadside),
                        // positive = toward +x (the array's "antenna 1"
                        // side) per the ULA steering convention in
                        // THEORY.md "The math".
    float amp;          // target reflection amplitude (RCS-ish, unitless
                        // teaching scale — see THEORY.md "The problem" for
                        // why we do not model the full radar-range-equation
                        // power law here).
};

// ---------------------------------------------------------------------------
// Detection — one CFAR-flagged, clustered range-Doppler cell after angle
// estimation. Filled on the host (main.cu) from GPU-computed pieces.
// ---------------------------------------------------------------------------
struct Detection {
    int   kr;           // range-FFT bin index (0..kNs-1)
    int   kd;            // FFTSHIFTED Doppler-FFT bin index (0..kNc-1);
                        // bin kNc/2 is zero velocity (see bin_to_vel below)
    float range_m;       // kr * kRangeResM
    float vel_mps;        // (kd - kNc/2) * kVelResMps
    float az_deg;         // estimated azimuth from the zero-padded angle FFT
    float power;          // the RD-map power at (kr, kd) that triggered CFAR
};

// ===========================================================================
// RADAR / CHIRP PARAMETERS — the single source of truth for cube synthesis,
// both processing paths, and every resolution/ambiguity formula quoted in
// README.md and THEORY.md. Values are a plausible mid-range automotive FMCW
// configuration (comparable in spirit to TI AWR/IWR-class parameter sheets;
// see README "Prior art"), chosen for round, checkable derived numbers.
// ===========================================================================
constexpr float kC          = 299792458.0f;  // speed of light, m/s (exact, SI)
constexpr float kFc         = 77.0e9f;       // carrier frequency, Hz (77 GHz automotive band)
constexpr float kBandwidth  = 300.0e6f;      // chirp sweep bandwidth B, Hz
constexpr float kChirpDur   = 50.0e-6f;      // chirp duration Tc, s (the ACTIVE sweep time)

constexpr int kNs    = 256;   // ADC samples per chirp   (fast-time / RANGE axis length)
constexpr int kNc    = 128;   // chirps per frame        (slow-time / DOPPLER axis length)
constexpr int kNa    = 8;     // virtual RX antennas, half-wavelength ULA (ANGLE axis length)
constexpr int kNaFft = 64;    // zero-padded angle-FFT length (interpolates the Na=8 array;
                              // does NOT add true angular resolution — THEORY.md is explicit
                              // about that distinction).

// ---- Derived chirp/ADC quantities (constexpr: computed once, at compile
// time, from the four physical parameters above — one source of truth). ----
constexpr float kSlope   = kBandwidth / kChirpDur;      // S: chirp slope, Hz/s
constexpr float kFs      = static_cast<float>(kNs) / kChirpDur; // ADC sample rate, Hz
constexpr float kLambda  = kC / kFc;                    // carrier wavelength, m (~3.89 mm at 77 GHz)
constexpr float kAntennaSpacingM = kLambda / 2.0f;       // ULA element pitch d (half-wavelength:
                                                          // the Nyquist spatial-sampling choice that
                                                          // gives a grating-lobe-free +/-90 deg field
                                                          // of view — THEORY.md derives this).

// ---- Resolution & ambiguity limits (derived; see THEORY.md "The math" for
// every derivation). These are the numbers README quotes and the ones the
// ground-truth verification tolerances in main.cu are built from. ----------
constexpr float kRangeResM  = kC / (2.0f * kBandwidth);              // dR = c/(2B)
constexpr float kRangeMaxM  = static_cast<float>(kNs) * kRangeResM;   // Ns * dR (this project's
                                                                       // complex-baseband cube uses the
                                                                       // FULL Ns bins for unambiguous
                                                                       // range — see THEORY.md "Numerical
                                                                       // considerations" for why a real
                                                                       // ADC sampling only I is different)
constexpr float kVelResMps  = kLambda / (2.0f * static_cast<float>(kNc) * kChirpDur); // dv = lambda/(2*Nc*Tc)
constexpr float kVelMaxMps  = kLambda / (4.0f * kChirpDur);           // +/- one-sided unambiguous velocity

// ---------------------------------------------------------------------------
// Synthesis noise. Complex baseband thermal-noise-floor model: independent
// zero-mean Gaussian on I and Q, std dev kNoiseStd per component (unitless,
// same scale as target `amp`). See kernels.cu / reference_cpu.cpp for the
// per-sample deterministic xorshift32 + Box-Muller generator both paths
// share bit-for-bit (float arithmetic on both sides — CLAUDE.md §12).
// ---------------------------------------------------------------------------
constexpr float    kNoiseStd  = 0.05f;
constexpr unsigned kNoiseSeed = 42u;      // fixed seed: CLAUDE.md §12 determinism

constexpr int kMaxTargets = 16;   // committed scenario capacity (6 used; headroom for exercises)

// ===========================================================================
// 2-D CFAR geometry — shared by CA-CFAR and OS-CFAR so the comparison in
// README/THEORY isolates exactly one variable (the training STATISTIC),
// nothing else. See THEORY.md "The algorithm" for the guard/training-band
// picture and the calibration story behind the two alpha constants below.
// ===========================================================================
constexpr int kCfarGuard  = 2;   // guard cells on each side of the cell under test (CUT), per axis
constexpr int kCfarTrain  = 5;   // training cells beyond the guard band, per side, per axis
constexpr int kCfarHalf   = kCfarGuard + kCfarTrain;              // = 7
constexpr int kCfarWindow = 2 * kCfarHalf + 1;                    // = 15 (full window edge length)
constexpr int kCfarNTrain =
    kCfarWindow * kCfarWindow - (2 * kCfarGuard + 1) * (2 * kCfarGuard + 1);  // = 200 training cells

// Threshold-scale factors ("alpha" in P_fa = f(alpha, N_train)). Both were
// CALIBRATED empirically against a large noise-only synthetic cube (no
// targets) targeting a per-cell false-alarm probability P_fa = 1e-4, by
// taking the (1 - P_fa) quantile of the empirical ratio (cell power) /
// (training statistic) over ~30k independent noise-only cells — see
// scripts/make_synthetic.py's docstring and THEORY.md "How we verify
// correctness" for the calibration procedure and the MEASURED false-alarm
// counts these constants actually produce on the committed scene (both
// numbers are printed by the demo, not asserted blindly).
constexpr float kAlphaCA      = 3.193f;   // CA-CFAR: cell > alpha * mean(training cells)
constexpr float kAlphaOS      = 2.585f;   // OS-CFAR: cell > alpha * training[rank]  (sorted ascending)
constexpr float kOsRankFrac   = 0.75f;    // OS-CFAR order-statistic rank (75th percentile, "3rd quartile")
constexpr int   kOsRankIndex  = 149;      // round(kOsRankFrac * (kCfarNTrain - 1)) = round(0.75*199) = 149

constexpr int kMaxDetections = 512;   // per-detector cap on clustered detections (each detector
                                      // realistically yields single digits to tens on this scene)

// ---------------------------------------------------------------------------
// Ground-truth verification tolerances (main.cu's "GROUND-TRUTH gates").
// Each bound is the corresponding RESOLUTION CELL — i.e. "the estimate may
// be off by at most one bin", the honest, formula-derived bound a bin-
// quantized (non-interpolated) FFT peak read-out can promise. Azimuth gets
// a wider bound: THEORY.md derives the zero-padded-FFT quantization error
// (~1.3 deg worst case at the 45 deg targets in the committed scene) and
// this triples it for noise-driven jitter headroom — the same "wide
// margin so ULP-level platform differences cannot flip the verdict"
// philosophy used throughout this repo (see 08.01 for the precedent).
// ---------------------------------------------------------------------------
constexpr float kRangeTolM  = kRangeResM;      // 1 range bin (~0.50 m)
constexpr float kVelTolMps  = kVelResMps;      // 1 Doppler bin (~0.30 m/s)
constexpr float kAzTolDeg   = 3.0f;            // ~3x the worst-case quantization step
constexpr int   kMaxFalseAlarmsOS = 8;         // generous bound above the measured 0-2 (see README)

// ===========================================================================
// GPU launchers (defined in kernels.cu). Every one owns its own launch
// configuration and post-launch CUDA_CHECK_LAST_ERROR call (CLAUDE.md §6.1
// rule 7); main.cu calls only these, never a __global__ kernel directly.
// All pointer parameters are DEVICE pointers unless documented "HOST".
// ===========================================================================

// launch_synthesize_cube — build the raw ADC cube: d_cube[Ns*Nc*Na] = the
// sum of `num_targets` targets' beat/Doppler/angle phasors plus per-sample
// complex Gaussian noise. d_targets is a DEVICE array of `num_targets`
// RadarTarget. One thread per complex sample (see kernels.cu for the
// thread-to-data mapping and the noise generator).
void launch_synthesize_cube(ComplexF32* d_cube,
                            const RadarTarget* d_targets, int num_targets);

// launch_hann_window_range / _doppler — multiply every sample by a Hann
// taper along the named axis (range: length Ns before the range FFT;
// doppler: length Nc, applied to the RANGE-FFT OUTPUT before the Doppler
// FFT). Both windows are precomputed once and cached in device globals by
// the FIRST call (see kernels.cu); both are simple, fully elementwise maps.
void launch_hann_window_range(ComplexF32* d_cube);
void launch_hann_window_doppler(ComplexF32* d_cube);

// launch_range_fft / launch_doppler_fft — the two batched cuFFT C2C
// transforms that turn fast-time / slow-time phase into range / Doppler
// bins, IN PLACE on d_cube. See kernels.cu for what each cufftPlanMany
// call computes, why cuFFT (not a hand-rolled FFT) is used here, and the
// exact istride/idist "advanced data layout" each one relies on.
void launch_range_fft(ComplexF32* d_cube);
void launch_doppler_fft(ComplexF32* d_cube);

// launch_fftshift_doppler — remap the DOPPLER axis from cuFFT's natural
// bin order (0, +1, ..., kNc/2-1, -kNc/2, ..., -1) to a CENTERED order
// where index kNc/2 is zero velocity (matching numpy's fftshift and the
// CPU oracle's write-time remap in reference_cpu.cpp's process_rd_map_cpu).
// OUT OF PLACE: d_out must be a distinct kNs*kNc*kNa ComplexF32 buffer.
// Every downstream stage (noncoherent integration, both CFAR detectors,
// the angle-snapshot gather) consumes the SHIFTED buffer exclusively, so
// "Doppler bin index" means the same, non-wrapping, centered thing
// everywhere past this point in the pipeline (kernels.cuh's Detection::kd).
void launch_fftshift_doppler(const ComplexF32* d_in, ComplexF32* d_out);

// launch_noncoherent_integrate — collapse the antenna axis into a single
// range-Doppler POWER map: d_rd_power[n*kNc+c] = mean_a |d_cube[n,c,a]|^2.
// One thread per (n, c) output cell.
void launch_noncoherent_integrate(const ComplexF32* d_cube, float* d_rd_power);

// launch_cfar_ca / launch_cfar_os — the 2-D CFAR detectors. One thread per
// interior (n, c) cell (border cells narrower than kCfarHalf from any edge
// are never flagged — see kernels.cu). d_det[n*kNc+c] is written 1/0;
// d_thresh[n*kNc+c] is the alpha-scaled training statistic (kept for the
// detections.csv diagnostic dump and THEORY.md's worked margin numbers).
void launch_cfar_ca(const float* d_rd_power, unsigned char* d_det, float* d_thresh);
void launch_cfar_os(const float* d_rd_power, unsigned char* d_det, float* d_thresh);

// launch_angle_fft — per-detection azimuth estimation. d_snapshots is a
// DEVICE buffer of num_det * kNaFft ComplexF32 (already gathered +
// zero-padded by the caller, see main.cu); transformed IN PLACE by one
// batched, CONTIGUOUS cuFFT C2C call (batch = num_det, istride = 1,
// idist = kNaFft) — the project's third and simplest cuFFT usage,
// deliberately contrasted with the two "advanced layout" calls above.
void launch_angle_fft(ComplexF32* d_snapshots, int num_det);

// launch_gather_angle_snapshots — for each of num_det detections (kr[i],
// kd[i]), copy the kNa antenna samples d_cube[kr,kd,0..kNa-1] into
// d_snapshots[i*kNaFft .. i*kNaFft+kNa-1] and ZERO the remaining
// kNaFft-kNa slots (the zero-padding launch_angle_fft's FFT then
// interpolates). One thread per (detection, output-bin) pair.
void launch_gather_angle_snapshots(const ComplexF32* d_cube,
                                   const int* d_kr, const int* d_kd, int num_det,
                                   ComplexF32* d_snapshots);

// launch_find_angle_peaks — one thread per detection: scan its kNaFft
// spectrum for the peak-magnitude bin and convert to azimuth degrees
// (the ULA steering-angle formula from THEORY.md). Writes d_az_deg[num_det].
void launch_find_angle_peaks(const ComplexF32* d_snapshots, int num_det, float* d_az_deg);

// ---------------------------------------------------------------------------
// CPU references (reference_cpu.cpp) — the O(N^2)-DFT oracle twin of the
// ENTIRE GPU pipeline above, function-for-function. main.cu runs both on
// the identical committed target list and requires agreement within a
// documented tolerance (the project's §5 GPU-vs-CPU VERIFY gate).
// ---------------------------------------------------------------------------

// synthesize_cube_cpu — the exact host twin of launch_synthesize_cube:
// same per-sample formula, same per-sample xorshift32+Box-Muller noise
// (so the CPU and GPU cubes agree to near-ULP precision with NO data
// transfer between the two paths — see THEORY.md "Numerical considerations",
// and compare with 08.01's identical host/device noise-parity strategy).
void synthesize_cube_cpu(ComplexF32* cube, const RadarTarget* targets, int num_targets);

// process_rd_map_cpu — Hann-window + O(N^2) range DFT + O(N^2) Doppler DFT
// (both via precomputed twiddle tables, NOT a recursive FFT — see the file
// header comment in reference_cpu.cpp for why the naive DFT is affordable
// here) + noncoherent antenna integration. Fills rd_power[kNs*kNc].
void process_rd_map_cpu(const ComplexF32* cube, float* rd_power);

// cfar_ca_cpu / cfar_os_cpu — line-by-line twins of the GPU CFAR kernels
// (same geometry, same alpha constants). det[] and thresh[] as above.
void cfar_ca_cpu(const float* rd_power, unsigned char* det, float* thresh);
void cfar_os_cpu(const float* rd_power, unsigned char* det, float* thresh);

// angle_estimate_cpu — O(kNaFft * kNa) zero-padded DFT twin of the
// GPU angle-FFT path for ONE detection's antenna snapshot (cube[kr,kd,*]).
// Returns the estimated azimuth in degrees.
float angle_estimate_cpu(const ComplexF32* cube, int kr, int kd);

#endif // PROJECT_KERNELS_CUH
