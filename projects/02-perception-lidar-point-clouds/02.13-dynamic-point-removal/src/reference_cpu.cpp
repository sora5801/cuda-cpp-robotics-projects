// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU reference for project 02.13
//                     (Dynamic point removal (raycast free-space carving))
//
// Independence ruling applied to THIS project (see docs/PROJECT_TEMPLATE's
// reference_cpu.cpp for the full ruling text; summarized here as it applies)
// --------------------------------------------------------------------------
// kernels.cuh single-sources the DATA-LAYOUT contract (voxel indexing: floor
// divide + pack three ints into a linear index) as shared HD functions — a
// case the ruling explicitly permits ("pure token-for-token transcription").
// What this file does NOT share with kernels.cu is the ALGORITHM: every
// function below — the DDA march (carve_one_beam_cpu / carve_cpu /
// carve_trace_one_beam_cpu) and the classification ratio (classify_cpu) — is
// typed fresh, independently, reading only kernels.cuh's shared constants
// and HD helpers, never kernels.cu's device code (which this host-compiled
// translation unit cannot even see — the __CUDACC__ fence hides it).
//
// Because sharing the indexing math alone could not catch a bug INSIDE that
// shared math, this project's independent verification does not stop at
// GPU-vs-CPU agreement: main.cu's ghost_removal / late_leaver /
// static_preservation gates (README "Expected output") compare the final
// classification against GROUND TRUTH loaded straight from
// data/sample/beams.csv's truth_dynamic column — data neither this file nor
// kernels.cu ever touches during carving or classification. That is the
// gate that "does not route through the shared code" the ruling requires.
//
// Two jobs in this project (all declared in kernels.cuh):
//   1) carve_cpu / carve_trace_one_beam_cpu — the ORACLE twins of
//      carve_kernel / carve_trace_kernel: same DDA algorithm, typed fresh,
//      sequential over beams (no atomics needed — a single thread can never
//      race itself, so plain += replaces atomicAdd throughout).
//   2) classify_cpu — the oracle twin of classify_kernel.
//
// Rules for this file: plain C++17, no CUDA headers, no cleverness. If the
// reference is clever, it can be wrong, and then the oracle lies.
//
// Read this after: kernels.cu — then compare the two DDA marches side by side.
// ===========================================================================

#include "kernels.cuh"   // shared constants, layouts, HD indexing helpers, signatures

#include <cmath>         // std::fmaf — the same explicit fused multiply-add kernels.cu's
                        // device code uses, for the GPU/CPU determinism contract
#include <cfloat>        // FLT_MAX

// ---------------------------------------------------------------------------
// carve_one_beam_cpu — the INDEPENDENT host twin of kernels.cu's
// carve_one_beam. Same algorithm (Amanatides & Woo march; see kernels.cu's
// extensive walkthrough — not repeated here, only the CPU-specific notes
// below), typed fresh: every step below was written by reading the
// ALGORITHM DESCRIPTION (kernels.cuh's file header / THEORY.md), not by
// copying kernels.cu's text, exactly what the independence ruling asks for.
//
// Differences from the GPU version that are NOT bugs:
//   * atomicAdd -> plain "+= 1u" (sequential, no race possible);
//   * fmaf() (a CUDA device intrinsic) -> std::fmaf() (its C++17 standard-
//     library equivalent) — both compute the SAME single-rounding fused
//     multiply-add on IEEE-754 floats, which is exactly why the two paths'
//     voxel SEQUENCES are expected to match bit-for-bit, not just nearly
//     (kernels.cu's file header "DETERMINISM" note).
// ---------------------------------------------------------------------------
static void carve_one_beam_cpu(
    float ox, float oy, float oz,
    float dx, float dy, float dz,
    float range_m, bool is_hit,
    unsigned int* hits, unsigned int* pass_hit, unsigned int* pass_maxrange,
    int* trace_out, int trace_cap, int* trace_len)
{
    int cx = voxel_coord_axis(ox, kGridOriginX, kVoxelSizeM);
    int cy = voxel_coord_axis(oy, kGridOriginY, kVoxelSizeM);
    int cz = voxel_coord_axis(oz, kGridOriginZ, kVoxelSizeM);

    const float ex = std::fmaf(dx, range_m, ox);
    const float ey = std::fmaf(dy, range_m, oy);
    const float ez = std::fmaf(dz, range_m, oz);
    int tx, ty, tz;
    world_to_voxel(ex, ey, ez, tx, ty, tz);

    int n_trace = 0;

    if (cx == tx && cy == ty && cz == tz) {
        if (is_hit && hits != nullptr && voxel_in_bounds(cx, cy, cz))
            hits[voxel_index(cx, cy, cz)] += 1u;
        if (trace_len != nullptr) *trace_len = 0;
        return;
    }

    // Per-axis step/tMax/tDelta — the identical closed form kernels.cu
    // derives (Amanatides & Woo), spelled out fresh here rather than shared,
    // per the independence ruling. No macro this time (a CPU file, no
    // pressure to avoid three copies of a GPU-launch-site block) — the three
    // axes are just written out by hand, x, then y, then z.
    int stepx, stepy, stepz;
    float tMaxX, tMaxY, tMaxZ, tDeltaX, tDeltaY, tDeltaZ;

    if (dx > 0.0f) {
        stepx = 1;
        const float boundary = std::fmaf(static_cast<float>(cx + 1), kVoxelSizeM, kGridOriginX);
        tMaxX = (boundary - ox) / dx;
        tDeltaX = kVoxelSizeM / dx;
    } else if (dx < 0.0f) {
        stepx = -1;
        const float boundary = std::fmaf(static_cast<float>(cx), kVoxelSizeM, kGridOriginX);
        tMaxX = (boundary - ox) / dx;
        tDeltaX = kVoxelSizeM / (-dx);
    } else {
        stepx = 0; tMaxX = FLT_MAX; tDeltaX = FLT_MAX;
    }

    if (dy > 0.0f) {
        stepy = 1;
        const float boundary = std::fmaf(static_cast<float>(cy + 1), kVoxelSizeM, kGridOriginY);
        tMaxY = (boundary - oy) / dy;
        tDeltaY = kVoxelSizeM / dy;
    } else if (dy < 0.0f) {
        stepy = -1;
        const float boundary = std::fmaf(static_cast<float>(cy), kVoxelSizeM, kGridOriginY);
        tMaxY = (boundary - oy) / dy;
        tDeltaY = kVoxelSizeM / (-dy);
    } else {
        stepy = 0; tMaxY = FLT_MAX; tDeltaY = FLT_MAX;
    }

    if (dz > 0.0f) {
        stepz = 1;
        const float boundary = std::fmaf(static_cast<float>(cz + 1), kVoxelSizeM, kGridOriginZ);
        tMaxZ = (boundary - oz) / dz;
        tDeltaZ = kVoxelSizeM / dz;
    } else if (dz < 0.0f) {
        stepz = -1;
        const float boundary = std::fmaf(static_cast<float>(cz), kVoxelSizeM, kGridOriginZ);
        tMaxZ = (boundary - oz) / dz;
        tDeltaZ = kVoxelSizeM / (-dz);
    } else {
        stepz = 0; tMaxZ = FLT_MAX; tDeltaZ = FLT_MAX;
    }

    for (int steps = 0; steps < kMaxDDASteps; ++steps) {
        if (tMaxX <= tMaxY && tMaxX <= tMaxZ) { cx += stepx; tMaxX += tDeltaX; }
        else if (tMaxY <= tMaxZ)              { cy += stepy; tMaxY += tDeltaY; }
        else                                   { cz += stepz; tMaxZ += tDeltaZ; }

        if (!voxel_in_bounds(cx, cy, cz))
            break;

        const bool reached = (cx == tx && cy == ty && cz == tz);
        const int v = voxel_index(cx, cy, cz);

        if (reached && is_hit) {
            if (hits != nullptr) hits[v] += 1u;
            if (trace_out != nullptr && n_trace < trace_cap) trace_out[n_trace++] = v;
            break;
        }

        if (is_hit) { if (pass_hit != nullptr) pass_hit[v] += 1u; }
        else        { if (pass_maxrange != nullptr) pass_maxrange[v] += 1u; }
        if (trace_out != nullptr && n_trace < trace_cap) trace_out[n_trace++] = v;

        if (reached) break;
    }

    if (trace_len != nullptr) *trace_len = n_trace;
}

// ---------------------------------------------------------------------------
// carve_cpu — sequential twin of carve_kernel: every beam, one after
// another, into the SAME three ledger arrays the GPU path fills.
// scan_origin_xyz is a plain host array (kernels.cuh's file header: no
// __constant__ memory on a CPU), interleaved [kNumScans*3].
// ---------------------------------------------------------------------------
void carve_cpu(int n_beams, const int* scan_id, const float* dir,
               const unsigned char* is_hit, const float* range,
               const float* scan_origin_xyz,
               unsigned int* hits, unsigned int* pass_from_hit,
               unsigned int* pass_from_maxrange)
{
    for (int i = 0; i < n_beams; ++i) {
        const int sid = scan_id[i];
        const float ox = scan_origin_xyz[sid * 3 + 0];
        const float oy = scan_origin_xyz[sid * 3 + 1];
        const float oz = scan_origin_xyz[sid * 3 + 2];
        carve_one_beam_cpu(ox, oy, oz,
                           dir[i * 3 + 0], dir[i * 3 + 1], dir[i * 3 + 2],
                           range[i], is_hit[i] != 0,
                           hits, pass_from_hit, pass_from_maxrange,
                           nullptr, 0, nullptr);
    }
}

// ---------------------------------------------------------------------------
// carve_trace_one_beam_cpu — the INDEPENDENT twin of carve_trace_kernel, for
// ONE beam (kernels.cuh's declaration comment: main.cu calls this once per
// beam in the documented verify subset). Returns the visited voxel count.
// ---------------------------------------------------------------------------
int carve_trace_one_beam_cpu(float ox, float oy, float oz,
                             float dx, float dy, float dz,
                             float range_m, bool is_hit,
                             int* trace_out)
{
    int len = 0;
    carve_one_beam_cpu(ox, oy, oz, dx, dy, dz, range_m, is_hit,
                       nullptr, nullptr, nullptr,
                       trace_out, kMaxDDASteps, &len);
    return len;
}

// ---------------------------------------------------------------------------
// classify_cpu — sequential twin of classify_kernel: same ratio formula,
// same sentinel convention for max-range beams (kernels.cuh's declaration
// comment for classify_kernel documents both).
// ---------------------------------------------------------------------------
void classify_cpu(int n_beams, const int* scan_id, const float* dir,
                  const unsigned char* is_hit, const float* range,
                  const float* scan_origin_xyz,
                  const unsigned int* hits, const unsigned int* pass_from_hit,
                  const unsigned int* pass_from_maxrange, float threshold,
                  float* score_out, int* label_out)
{
    for (int i = 0; i < n_beams; ++i) {
        if (is_hit[i] == 0) {
            score_out[i] = -1.0f;
            label_out[i] = -1;
            continue;
        }

        const int sid = scan_id[i];
        const float px = std::fmaf(dir[i * 3 + 0], range[i], scan_origin_xyz[sid * 3 + 0]);
        const float py = std::fmaf(dir[i * 3 + 1], range[i], scan_origin_xyz[sid * 3 + 1]);
        const float pz = std::fmaf(dir[i * 3 + 2], range[i], scan_origin_xyz[sid * 3 + 2]);

        int ix, iy, iz;
        world_to_voxel(px, py, pz, ix, iy, iz);

        float score = -1.0f;
        int label = -1;
        if (voxel_in_bounds(ix, iy, iz)) {
            const int v = voxel_index(ix, iy, iz);
            const unsigned int h = hits[v];
            const unsigned int p = pass_from_hit[v] + pass_from_maxrange[v];
            const unsigned int total = h + p;
            score = (total > 0u) ? (static_cast<float>(p) / static_cast<float>(total)) : 0.0f;
            label = (score >= threshold) ? 1 : 0;
        }
        score_out[i] = score;
        label_out[i] = label;
    }
}
