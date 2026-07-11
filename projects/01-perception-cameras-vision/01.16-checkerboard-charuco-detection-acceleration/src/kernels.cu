// ===========================================================================
// kernels.cu — GPU kernels for project 01.16 (Checkerboard/ChArUco detection
//              acceleration for auto-calibration rigs)
//
// Four kernel families, in pipeline order (kernels.cuh's file header derives
// the math for each): saddle response (pixel-parallel, batched) -> NMS
// (pixel-parallel, batched) -> sub-pixel refinement (candidate-parallel) ->
// marker decode (marker-slot-parallel). Grid ordering / DLT / Zhang stay on
// the host (reference_cpu.cpp) — see kernels.cuh's independence note.
//
// Read this after: main.cu, kernels.cuh.  Read this before: reference_cpu.cpp.
// ===========================================================================

#include "kernels.cuh"
#include "util/cuda_check.cuh"

// ---------------------------------------------------------------------------
// bilinear_sample_device — clamp-to-edge bilinear intensity sample of ONE
// view's slice of the batched grayscale array at a FRACTIONAL pixel
// location (px, py). Used by both the sub-pixel refinement kernel (image
// gradients) and the marker-decode kernel (perspective cell sampling) — a
// pure numerical helper, not "the algorithm under test" for either stage
// (each stage's surrounding logic is still typed independently on the CPU
// side — see reference_cpu.cpp's OWN bilinear_sample_cpu, a deliberately
// separate implementation, per the twin-independence ruling).
//
// Parameters: gray — [kBatchPixels] uint8, the WHOLE batch; view selects
//             the W*H slice. px, py — fractional pixel coords (may be
//             slightly outside [0,W)x[0,H) — clamped, never a wild read).
// Returns: interpolated intensity, 0..255 range (as float).
// ---------------------------------------------------------------------------
__device__ __forceinline__ float bilinear_sample_device(const unsigned char* __restrict__ gray,
                                                         int view, float px, float py)
{
    // Clamp the CONTINUOUS coordinate first (not just the integer floor) so
    // a query a hair outside the image still reads the edge pixel's own
    // neighborhood instead of wrapping into a different row (a classic
    // off-by-one that corrupts the top/bottom row of every image).
    if (px < 0.0f) px = 0.0f; if (px > kImgW - 1.0f) px = kImgW - 1.0f;
    if (py < 0.0f) py = 0.0f; if (py > kImgH - 1.0f) py = kImgH - 1.0f;

    const int x0 = static_cast<int>(px);
    const int y0 = static_cast<int>(py);
    const int x1 = (x0 + 1 < kImgW) ? x0 + 1 : x0;
    const int y1 = (y0 + 1 < kImgH) ? y0 + 1 : y0;
    const float fx = px - static_cast<float>(x0);   // fractional part, in [0,1)
    const float fy = py - static_cast<float>(y0);

    const float i00 = static_cast<float>(gray[batch_pixel_index(view, x0, y0)]);
    const float i10 = static_cast<float>(gray[batch_pixel_index(view, x1, y0)]);
    const float i01 = static_cast<float>(gray[batch_pixel_index(view, x0, y1)]);
    const float i11 = static_cast<float>(gray[batch_pixel_index(view, x1, y1)]);

    // Standard bilinear blend: interpolate along x on both rows, then along
    // y between the two row results.
    const float top = i00 + (i10 - i00) * fx;
    const float bot = i01 + (i11 - i01) * fx;
    return top + (bot - top) * fy;
}

// ---------------------------------------------------------------------------
// apply_homography_device — board-plane meters (X,Y) -> pixel (u,v) through
// a 3x3 row-major Homography (double precision, matching the DLT/Zhang
// solve's own precision — see kernels.cuh). Returns false if the homogeneous
// weight w is degenerate (near zero -- a homography that maps this point to
// infinity, never expected for interior board points but guarded anyway).
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool apply_homography_device(const Homography& H, float X, float Y,
                                                         float& u, float& v)
{
    const double w = H.h[6] * X + H.h[7] * Y + H.h[8];
    if (fabs(w) < 1e-12) return false;
    u = static_cast<float>((H.h[0] * X + H.h[1] * Y + H.h[2]) / w);
    v = static_cast<float>((H.h[3] * X + H.h[4] * Y + H.h[5]) / w);
    return true;
}

// ===========================================================================
// STAGE 1 — saddle_response_kernel (PIXEL-parallel, BATCHED).
//
// Big idea: an X-corner (inner checkerboard vertex) is a SADDLE POINT of
// image intensity — walking along ONE diagonal through it, intensity rises;
// walking along the OTHER diagonal, it falls. That is exactly the geometric
// meaning of a NEGATIVE Hessian determinant: det(Hessian) = Ixx*Iyy - Ixy^2
// < 0 means the local quadratic approximation of I(x,y) has one positive and
// one negative principal curvature (a saddle), vs. > 0 for a smooth extremum
// (max/min — flat squares' interiors) and vs. Harris's structure tensor
// (01.04's harris_response_kernel), which instead looks at FIRST derivatives
// and flags a corner when BOTH eigenvalues of the gradient outer-product
// tensor are large and POSITIVE — a fundamentally different object
// (Harris finds an "L" where two straight edges meet at any angle; the
// Hessian test finds the "X" pattern specific to a 4-quadrant checkerboard
// vertex, and does NOT fire on a plain "L" corner, since an L-shaped step
// edge has a locally near-zero Hessian off the edge itself). THEORY.md "The
// math" derives this side by side with 01.04's Harris score.
//
// Thread-to-data mapping: ONE FLAT grid-stride loop over
// idx in [0, kBatchPixels) -- kernels.cuh's batch_pixel_index() decomposes
// idx into (view, x, y). This fuses "view-parallel" and "pixel-parallel"
// into a single 1-D launch: adjacent idx values are adjacent pixels within
// the SAME view (coalesced global memory access, the same reasoning as the
// SAXPY placeholder), and every view gets equal pixel-parallel throughput
// without a separate kernel per view or a 3-D grid (THEORY.md "The GPU
// mapping" argues this choice and measures the resulting occupancy).
//
// Parameters: gray — [kBatchPixels] uint8 IN. resp — [kBatchPixels] float
// OUT (>=0; 0 wherever the +-kSaddleStep stencil would read outside the
// OWNING VIEW's own W x H frame -- each view's border is its own border,
// never a neighboring view's pixels, because batch_pixel_index() keeps
// every view's slice self-contained).
// ---------------------------------------------------------------------------
__global__ void saddle_response_kernel(const unsigned char* __restrict__ gray, float* __restrict__ resp,
                                       int total_pixels)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_pixels; idx += gridDim.x * blockDim.x) {
        const int view  = idx / kViewPixels;             // which of the kNumViews images this pixel belongs to
        const int local = idx - view * kViewPixels;       // pixel offset WITHIN that view (0 .. kViewPixels-1)
        const int x = local % kImgW;
        const int y = local / kImgW;

        const int s = kSaddleStep;
        if (x < s || x >= kImgW - s || y < s || y >= kImgH - s) {
            resp[idx] = 0.0f;   // too close to THIS VIEW's own border for the +-s stencil
            continue;
        }

        // Six neighborhood reads (integer offsets -> exact array indices, no
        // interpolation, so this stage is fully reproducible in exact
        // integer-valued float arithmetic — see kernels.cuh's file header
        // "TWIN-INDEPENDENCE": every intensity here is an exact small
        // integer (0..255) and every product below stays under float32's
        // 2^24 exact-integer range, so GPU (FMA-permitted) and CPU (no
        // contraction) compute BIT-IDENTICAL results — no ULP amplification
        // story for this stage, unlike sub-pixel refinement (stage 3).
        const float Ic  = static_cast<float>(gray[batch_pixel_index(view, x,     y    )]);
        const float Ixp = static_cast<float>(gray[batch_pixel_index(view, x + s, y    )]);
        const float Ixm = static_cast<float>(gray[batch_pixel_index(view, x - s, y    )]);
        const float Iyp = static_cast<float>(gray[batch_pixel_index(view, x,     y + s)]);
        const float Iym = static_cast<float>(gray[batch_pixel_index(view, x,     y - s)]);
        const float Ipp = static_cast<float>(gray[batch_pixel_index(view, x + s, y + s)]);
        const float Ipm = static_cast<float>(gray[batch_pixel_index(view, x + s, y - s)]);
        const float Imp = static_cast<float>(gray[batch_pixel_index(view, x - s, y + s)]);
        const float Imm = static_cast<float>(gray[batch_pixel_index(view, x - s, y - s)]);

        // Second-derivative finite differences (step s): the standard
        // 3-point central second-difference formula, and the 4-point mixed
        // partial (a "cross" stencil) — THEORY.md "The math" derives both
        // from a Taylor expansion of I(x,y) about (x,y).
        const float Ixx = Ixp - 2.0f * Ic + Ixm;
        const float Iyy = Iyp - 2.0f * Ic + Iym;
        const float Ixy = (Ipp - Ipm - Imp + Imm) * 0.25f;

        const float det = Ixx * Iyy - Ixy * Ixy;
        // Both axes must show REAL curvature (kMinAxisCurvature's doc
        // comment) -- rejects "strong edge + noise" false saddle signals.
        // AND opposite quadrants must be near-equal in color (kMaxDiagonal
        // Asymmetry's doc comment) -- rejects three-region T-junctions
        // (e.g. the board's own outer silhouette) that satisfy the first
        // two tests but are not actually a two-color diagonal saddle.
        const bool axes_ok = (fabsf(Ixx) >= kMinAxisCurvature) && (fabsf(Iyy) >= kMinAxisCurvature);
        const bool diag_ok = (fabsf(Ipp - Imm) <= kMaxDiagonalAsymmetry) && (fabsf(Ipm - Imp) <= kMaxDiagonalAsymmetry);
        const bool is_saddle = (det < 0.0f) && axes_ok && diag_ok;
        resp[idx] = is_saddle ? -det : 0.0f;
    }
}

void launch_saddle_response(const unsigned char* d_gray, float* d_resp, int num_views)
{
    const int total_pixels = num_views * kViewPixels;
    const int block = 256;
    int grid = (total_pixels + block - 1) / block;
    if (grid > 8192) grid = 8192;   // grid-stride loop absorbs the remainder (see saxpy precedent)
    saddle_response_kernel<<<grid, block>>>(d_gray, d_resp, total_pixels);
    CUDA_CHECK_LAST_ERROR("saddle_response_kernel launch");
}

// ===========================================================================
// STAGE 2 — nms_candidates_kernel (PIXEL-parallel, BATCHED, atomic
// compaction).
//
// A pixel is a CANDIDATE corner iff (a) its response exceeds
// kSaddleRespThresh, and (b) it is a STRICT local maximum over its own
// (2*kNmsRadius+1)^2 window WITHIN THE SAME VIEW (a tie with any neighbor
// suppresses BOTH pixels — the same deterministic, twin-reproducible rule
// 01.04's FAST/Harris NMS uses, chosen for the identical reason: it makes
// the GPU and CPU candidate SETS compare exactly equal, not just
// element-count equal, because there is no ambiguity about which of two
// equal-score neighbors "wins").
//
// Accepted candidates are atomically appended to THIS VIEW's own slice of
// d_cand (view v owns d_cand[v*kMaxCandidatesPerView .. (v+1)*kMaxCandidatesPerView)),
// via a per-view atomicAdd counter — the same "pack via atomic compaction"
// pattern as 01.04/01.06, generalized here to B independent counters (one
// per view) instead of one global counter, so a very textured view (e.g.
// the negative control) cannot starve another view's candidate slots.
// ---------------------------------------------------------------------------
__global__ void nms_candidates_kernel(const float* __restrict__ resp,
                                      RawCandidate* __restrict__ cand,
                                      int* __restrict__ view_counts,
                                      int total_pixels)
{
    const int margin = kNmsRadius + kSaddleStep;   // response is 0 (never a candidate) inside this margin anyway
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_pixels; idx += gridDim.x * blockDim.x) {
        const int view  = idx / kViewPixels;
        const int local = idx - view * kViewPixels;
        const int x = local % kImgW;
        const int y = local / kImgW;
        if (x < margin || x >= kImgW - margin || y < margin || y >= kImgH - margin) continue;

        const float center = resp[idx];
        if (center <= kSaddleRespThresh) continue;

        bool is_max = true;
        for (int dy = -kNmsRadius; dy <= kNmsRadius && is_max; ++dy) {
            for (int dx = -kNmsRadius; dx <= kNmsRadius; ++dx) {
                if (dx == 0 && dy == 0) continue;
                const float other = resp[batch_pixel_index(view, x + dx, y + dy)];
                if (other >= center) { is_max = false; break; }   // >= : a TIE also suppresses (see header)
            }
        }
        if (!is_max) continue;

        // Reserve this view's next candidate slot. atomicAdd on a per-VIEW
        // counter (not one global counter) is what keeps one view's texture
        // from crowding out another's candidate budget.
        const int slot = atomicAdd(&view_counts[view], 1);
        if (slot < kMaxCandidatesPerView) {
            RawCandidate c;
            c.view = view; c.x = x; c.y = y; c.score = center;
            cand[view * kMaxCandidatesPerView + slot] = c;
        }
        // slot >= kMaxCandidatesPerView: silently dropped (capacity
        // exceeded); main.cu prints the raw view_counts[] so an overflow is
        // visible, never silently hidden (CLAUDE.md §13).
    }
}

void launch_nms_candidates(const float* d_resp, RawCandidate* d_cand, int* d_view_counts, int num_views)
{
    const int total_pixels = num_views * kViewPixels;
    const int block = 256;
    int grid = (total_pixels + block - 1) / block;
    if (grid > 8192) grid = 8192;
    nms_candidates_kernel<<<grid, block>>>(d_resp, d_cand, d_view_counts, total_pixels);
    CUDA_CHECK_LAST_ERROR("nms_candidates_kernel launch");
}

// ===========================================================================
// STAGE 3 — subpixel_refine_kernel (CANDIDATE-parallel: one thread per
// candidate).
//
// The cornerSubPix idea (THEORY.md "The math" derives the linear system in
// full): at a TRUE corner, the image gradient at every nearby pixel q is
// (in the noise-free limit) ORTHOGONAL to the vector from q to the corner
// c — because along any ray INTO the corner from a nearby point, intensity
// is locally constant near an edge but the corner itself is where multiple
// edges meet, so a small neighborhood's gradient field points radially
// AWAY from/INTO the corner along edges, never tangentially past it. This
// gives, for every sample q with gradient g(q): g(q) . (q - c) = 0, i.e.
// g(q).q = g(q).c. Stack this over a window of samples q_1..q_M and solve
// the LEAST-SQUARES 2x2 system  [sum g g^T] c = [sum g g^T q]  for c. We
// iterate kRefineIters times, RECENTERING the sample window at the latest
// estimate of c each time (a fixed-point iteration — THEORY.md "Numerical
// considerations" discusses convergence).
//
// Thread-to-data mapping: thread i owns candidate i (flattened across every
// view); its own view's slice of the batched image is read via cand[i].view.
// No shared memory: each thread's window (up to (2*kRefineWinRadius+1)^2-1
// samples) is entirely private, so registers/local memory are the natural
// home — a small per-thread working set, the same "batched small solve per
// thread" spirit as 33.01's Jacobians and 01.06's DLT (cited there).
// ---------------------------------------------------------------------------
__global__ void subpixel_refine_kernel(const unsigned char* __restrict__ gray,
                                       const RawCandidate* __restrict__ cand, int n,
                                       RefinedCorner* __restrict__ out)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const RawCandidate c = cand[i];
        float cx = static_cast<float>(c.x);
        float cy = static_cast<float>(c.y);
        bool valid = true;

        for (int iter = 0; iter < kRefineIters; ++iter) {
            double G00 = 0.0, G01 = 0.0, G11 = 0.0, bxv = 0.0, byv = 0.0;
            for (int dy = -kRefineWinRadius; dy <= kRefineWinRadius; ++dy) {
                for (int dx = -kRefineWinRadius; dx <= kRefineWinRadius; ++dx) {
                    if (dx == 0 && dy == 0) continue;   // gradient AT the corner itself is ~0, uninformative
                    const float qx = cx + static_cast<float>(dx);
                    const float qy = cy + static_cast<float>(dy);
                    // Central-difference gradient at q, via bilinear samples
                    // one pixel either side (a Sobel-free, minimal gradient
                    // estimator — sufficient here because the WINDOW SUM,
                    // not any single sample, drives the fit).
                    const float gx = (bilinear_sample_device(gray, c.view, qx + 1.0f, qy) -
                                     bilinear_sample_device(gray, c.view, qx - 1.0f, qy)) * 0.5f;
                    const float gy = (bilinear_sample_device(gray, c.view, qx, qy + 1.0f) -
                                     bilinear_sample_device(gray, c.view, qx, qy - 1.0f)) * 0.5f;
                    const double gxd = static_cast<double>(gx), gyd = static_cast<double>(gy);
                    G00 += gxd * gxd; G01 += gxd * gyd; G11 += gyd * gyd;
                    bxv += gxd * gxd * static_cast<double>(qx) + gxd * gyd * static_cast<double>(qy);
                    byv += gxd * gyd * static_cast<double>(qx) + gyd * gyd * static_cast<double>(qy);
                }
            }
            const double det = G00 * G11 - G01 * G01;
            if (fabs(det) < 1e-6) { valid = false; break; }   // degenerate (flat/low-texture) neighborhood
            const double new_cx = (bxv * G11 - G01 * byv) / det;
            const double new_cy = (G00 * byv - G01 * bxv) / det;
            cx = static_cast<float>(new_cx);
            cy = static_cast<float>(new_cy);
        }

        RefinedCorner r;
        r.view = c.view; r.x = cx; r.y = cy; r.valid = valid;
        out[i] = r;
    }
}

void launch_subpixel_refine(const unsigned char* d_gray, const RawCandidate* d_cand, int n, RefinedCorner* d_out)
{
    if (n <= 0) return;
    const int block = 128;
    int grid = (n + block - 1) / block;
    if (grid < 1) grid = 1;
    subpixel_refine_kernel<<<grid, block>>>(d_gray, d_cand, n, d_out);
    CUDA_CHECK_LAST_ERROR("subpixel_refine_kernel launch");
}

// ===========================================================================
// STAGE 4 — marker_decode_kernel (one thread per (view, marker_id) pair,
// kNumViews*kNumMarkerCodes threads total).
//
// This is where the ChArUco "anchoring" actually happens (kernels.cuh's
// file header walks the two-hypothesis idea): the homography H[view] was
// fit ASSUMING the grid-ordering stage's provisional (i,j) labeling is
// correct — but a plain checkerboard's own geometry cannot tell "correct"
// from "180-degree-rotated" apart (mirror_square()'s doc comment). So for
// EVERY marker slot we try BOTH: sample as if the provisional labeling is
// right (the IDENTITY hypothesis), and sample as if it is 180-degree
// flipped (the MIRRORED hypothesis — same homography, but evaluated at the
// square's mirrored board-plane position, and the sampled 5x5 grid read
// back with (r,c) -> (N-1-r,N-1-c), undoing the physical 180-degree flip a
// truly-rotated board would print onto the marker itself). Whichever
// hypothesis clears the border-ring + Hamming test is a VOTE main.cu tallies
// across every marker in the view to decide whether to relabel every corner
// (the ambiguity_lesson / grid_ordering gates exercise exactly this).
//
// No shared memory, no cross-thread communication: each thread's 2*25 = 50
// bilinear samples are entirely private (same "tiny independent solve per
// thread" spirit as stage 3).
// ---------------------------------------------------------------------------
__device__ void decode_one_hypothesis(const unsigned char* __restrict__ gray, int view,
                                      const Homography& H, int sample_bx, int sample_by,
                                      bool mirrored, uint16_t true_code, int correction_capacity,
                                      bool& border_ok, bool& accepted, int& hamming_out)
{
    float scx, scy;
    square_center_board_xy(sample_bx, sample_by, scx, scy);
    const float half = kMarkerFillFrac * kSquareSizeM * 0.5f;
    const float cell = (2.0f * half) / static_cast<float>(kMarkerGridN);

    // cell_black[r][c] indexed by LOGICAL (r,c) -- see this function's
    // caller and the file header for why the mirrored hypothesis writes
    // into (N-1-r, N-1-c) instead of (r,c).
    bool cell_black[kMarkerGridN][kMarkerGridN];

    for (int r = 0; r < kMarkerGridN; ++r) {
        for (int c = 0; c < kMarkerGridN; ++c) {
            const float lx = (scx - sample_bx * kSquareSizeM) - half + (static_cast<float>(c) + 0.5f) * cell;
            const float ly = (scy - sample_by * kSquareSizeM) - half + (static_cast<float>(r) + 0.5f) * cell;
            const float boardX = sample_bx * kSquareSizeM + lx;
            const float boardY = sample_by * kSquareSizeM + ly;
            float u, v;
            bool black = true;
            if (apply_homography_device(H, boardX, boardY, u, v)) {
                const float val = bilinear_sample_device(gray, view, u, v);
                black = (val < kInkMidThreshold);
            }
            const int lr = mirrored ? (kMarkerGridN - 1 - r) : r;
            const int lc = mirrored ? (kMarkerGridN - 1 - c) : c;
            cell_black[lr][lc] = black;
        }
    }

    int border_errors = 0;
    for (int r = 0; r < kMarkerGridN; ++r)
        for (int c = 0; c < kMarkerGridN; ++c)
            if (marker_is_border_cell(r, c) && !cell_black[r][c]) ++border_errors;
    border_ok = (border_errors <= kMaxMarkerBorderErrors);

    uint16_t code = 0;
    for (int r = 1; r < kMarkerGridN - 1; ++r)
        for (int c = 1; c < kMarkerGridN - 1; ++c)
            if (cell_black[r][c]) code |= static_cast<uint16_t>(1u << marker_payload_bit_index(r - 1, c - 1));

    hamming_out = popcount_u32(static_cast<unsigned int>(code ^ true_code));
    accepted = border_ok && (hamming_out <= correction_capacity);
}

__global__ void marker_decode_kernel(const unsigned char* __restrict__ gray,
                                     const Homography* __restrict__ homs,
                                     const uint16_t* __restrict__ true_codes, int correction_capacity,
                                     MarkerDecodeResult* __restrict__ results)
{
    const int total = kNumViews * kNumMarkerCodes;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int view = idx / kNumMarkerCodes;
    const int marker_id = idx % kNumMarkerCodes;

    MarkerDecodeResult r;
    r.view = view; r.marker_id = marker_id;
    r.border_ok_identity = false; r.border_ok_mirrored = false;
    r.accepted = false; r.hyp_mirrored = false; r.hamming_distance = kMarkerPayloadBits;

    const Homography H = homs[view];
    if (!H.valid) { results[idx] = r; return; }

    int bx, by; square_of_marker_id(marker_id, bx, by);
    const uint16_t true_code = true_codes[marker_id];

    bool border_id, acc_id; int ham_id;
    decode_one_hypothesis(gray, view, H, bx, by, /*mirrored=*/false, true_code, correction_capacity,
                          border_id, acc_id, ham_id);
    r.border_ok_identity = border_id;

    int mbx, mby; mirror_square(bx, by, mbx, mby);
    bool border_mir, acc_mir; int ham_mir;
    decode_one_hypothesis(gray, view, H, mbx, mby, /*mirrored=*/true, true_code, correction_capacity,
                          border_mir, acc_mir, ham_mir);
    r.border_ok_mirrored = border_mir;

    if (acc_id) {
        r.accepted = true; r.hyp_mirrored = false; r.hamming_distance = ham_id;
    } else if (acc_mir) {
        r.accepted = true; r.hyp_mirrored = true; r.hamming_distance = ham_mir;
    } else {
        // Neither hypothesis cleared the test -- report the closer one
        // (debug/artifact value only; `accepted` is false either way).
        if (ham_id <= ham_mir) { r.hyp_mirrored = false; r.hamming_distance = ham_id; }
        else                    { r.hyp_mirrored = true;  r.hamming_distance = ham_mir; }
    }
    results[idx] = r;
}

void launch_marker_decode(const unsigned char* d_gray, const Homography* d_homography,
                          const uint16_t* d_true_codes, int correction_capacity,
                          MarkerDecodeResult* d_results)
{
    const int total = kNumViews * kNumMarkerCodes;
    const int block = 64;
    const int grid = (total + block - 1) / block;
    marker_decode_kernel<<<grid, block>>>(d_gray, d_homography, d_true_codes, correction_capacity, d_results);
    CUDA_CHECK_LAST_ERROR("marker_decode_kernel launch");
}
