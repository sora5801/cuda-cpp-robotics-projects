// ===========================================================================
// reference_cpu.cpp — plain-C++ CPU references for project 02.01
//                     (Voxel-grid downsampling with GPU spatial hashing)
//
// WHY does a GPU repository ship a CPU implementation of everything?
// ------------------------------------------------------------------
// Two load-bearing reasons (CLAUDE.md §5): (1) the CORRECTNESS ORACLE — GPU
// code fails in ways CPU code cannot (wrong thread indexing, races, stale
// device memory); (2) the TEACHING BASELINE — reading this file first, then
// kernels.cu, shows exactly what parallelization changed.
//
// Independence ruling for THIS project (CLAUDE.md §5 / template ruling) —
// read this before assuming all three functions below play the same role:
// -----------------------------------------------------------------------
//   * compute_keys_cpu shares kernels.cuh's voxel_coord/pack_voxel_key
//     functions DIRECTLY — this is a DATA-LAYOUT CONTRACT (an indexing
//     formula), which the ruling requires to be single-sourced, not
//     independently reimplemented. Its GPU counterpart (kernels.cu's
//     device-side transcription) is a SEPARATE, hand-copied piece of code
//     for a different reason (nvcc cannot call a host-only function from
//     device code — see kernels.cuh's file header) — VERIFY(keys) in
//     main.cu is the independent gate that catches drift between those
//     two device/host copies.
//
//   * sort_based_downsample_cpu is Method B's BIT-EXACT twin — DELIBERATELY
//     NOT independent in algorithm or order. Method B's whole teaching
//     point is that FIXING the summation order (stable sort by key, then
//     sequential per-voxel accumulation) makes float reduction reproducible
//     across host and device; reproducing that exact order here is the
//     point, not a shortcut. The independent check for this path is
//     GATE centroid_containment (a geometric invariant that does not care
//     how the centroid was computed) plus GATE partition_invariant (exact
//     integer bookkeeping) — both computed from the GPU's OWN output,
//     never routed through this twin.
//
//   * hashmap_downsample_cpu is Method A's INDEPENDENT twin: a genuinely
//     different data structure (std::unordered_map's internal chaining, not
//     this project's open-addressing table), a genuinely different
//     accumulation order (sequential point index 0..n-1, not GPU
//     thread-scheduling order), and higher precision (double, not float) —
//     the same "give the oracle more precision than the thing under test"
//     choice project 02.06's build_normal_system_cpu makes. This is the
//     "write the algorithmic core twice, independently" default the ruling
//     asks for wherever duplication is not pure transcription.
//
// Rules for this file: plain C++17, no CUDA headers, no hand-vectorization,
// no OpenMP, no cleverness — a dead-simple sequential version a reader can
// verify by eye.
//
// Read this after: kernels.cu — then compare the two side by side.
// ===========================================================================

#include "kernels.cuh"
#include <algorithm>   // std::stable_sort — the CPU twin of thrust::stable_sort_by_key
#include <vector>

// ---------------------------------------------------------------------------
// compute_keys_cpu — sequential twin of compute_keys_kernel. Calls this
// header's OWN voxel_coord/pack_voxel_key (see the independence-ruling
// comment above: this is a shared LAYOUT formula, not a duplicated
// algorithm). O(n) time, O(1) extra space per point.
// ---------------------------------------------------------------------------
void compute_keys_cpu(int n, const float* xyz, float leaf, unsigned long long* keys_out)
{
    for (int i = 0; i < n; ++i) {
        const int32_t vx = voxel_coord(xyz[i * 3 + 0], leaf);
        const int32_t vy = voxel_coord(xyz[i * 3 + 1], leaf);
        const int32_t vz = voxel_coord(xyz[i * 3 + 2], leaf);
        keys_out[i] = pack_voxel_key(vx, vy, vz);
    }
}

// ---------------------------------------------------------------------------
// sort_based_downsample_cpu — Method B's bit-exact twin.
//
// Step 1: compute every point's key (the shared layout formula again).
// Step 2: build the identity index array 0..n-1 and STABLE-sort it by key.
//         std::stable_sort's contract is EXACTLY thrust::stable_sort_by_key's
//         contract used on the GPU side: equal keys keep their RELATIVE
//         INPUT ORDER. Starting from an already-ascending index array
//         (0,1,2,...,n-1) means that guarantee reduces to "within a voxel,
//         points appear in ascending original index order" — the identical
//         rule the GPU's stable sort enforces, so the two permutations are
//         IDENTICAL for identical input keys (and VERIFY(keys) already
//         established the keys themselves are identical).
// Step 3: walk the sorted index array once, left to right, splitting it
//         into runs of equal key (a run == one voxel's points, contiguous
//         because the array is sorted) and summing each run's x/y/z in
//         PLAIN FLOAT, IN THAT SORTED ORDER — the same accumulator type and
//         the same operand order segmented_reduce_kernel uses on the GPU.
//         Neither side uses a fused multiply-add (there is no multiply
//         here) or a fast-math flag, so IEEE-754 round-to-nearest-even
//         float addition is fully determined by the operand sequence —
//         which is now identical on both sides. That equality of sequence,
//         not any special numerical trick, is what makes the two outputs
//         bit-exact (THEORY.md "Numerical considerations" makes the case
//         from first principles).
//
// Returns: the number of occupied voxels, written to out_xyz/out_count/
// out_key[0 .. return value) in ASCENDING VOXEL-KEY order — matching the
// GPU path's output order so main.cu can compare the two POSITIONALLY.
// ---------------------------------------------------------------------------
int sort_based_downsample_cpu(int n, const float* xyz, float leaf,
                              float* out_xyz, unsigned int* out_count,
                              unsigned long long* out_key)
{
    std::vector<unsigned long long> keys(static_cast<size_t>(n));
    compute_keys_cpu(n, xyz, leaf, keys.data());

    // idx starts as the identity permutation 0,1,2,...,n-1 — ascending
    // original point index, the same starting point thrust::sequence gives
    // the GPU path.
    std::vector<int> idx(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) idx[static_cast<size_t>(i)] = i;

    // Comparator sorts by KEY ONLY (never by idx itself) — leaving ties to
    // std::stable_sort's stability guarantee, exactly mirroring
    // thrust::stable_sort_by_key(keys, keys+n, idx) on the GPU side.
    std::stable_sort(idx.begin(), idx.end(),
                     [&keys](int a, int b) { return keys[static_cast<size_t>(a)] < keys[static_cast<size_t>(b)]; });

    int num_voxels = 0;
    int i = 0;
    while (i < n) {
        // Find the end of this run: every position sharing idx[i]'s key.
        int j = i;
        const unsigned long long run_key = keys[static_cast<size_t>(idx[static_cast<size_t>(i)])];
        while (j < n && keys[static_cast<size_t>(idx[static_cast<size_t>(j)])] == run_key) ++j;

        // Sequential float sum, sorted-position order i..j-1 — the fixed
        // order segmented_reduce_kernel also uses.
        float sx = 0.0f, sy = 0.0f, sz = 0.0f;
        for (int k = i; k < j; ++k) {
            const int p = idx[static_cast<size_t>(k)];
            sx += xyz[p * 3 + 0];
            sy += xyz[p * 3 + 1];
            sz += xyz[p * 3 + 2];
        }
        const unsigned int cnt = static_cast<unsigned int>(j - i);
        out_xyz[num_voxels * 3 + 0] = sx / static_cast<float>(cnt);
        out_xyz[num_voxels * 3 + 1] = sy / static_cast<float>(cnt);
        out_xyz[num_voxels * 3 + 2] = sz / static_cast<float>(cnt);
        out_count[num_voxels] = cnt;
        out_key[num_voxels]   = run_key;
        ++num_voxels;

        i = j;
    }
    return num_voxels;
}

// ---------------------------------------------------------------------------
// hashmap_downsample_cpu — Method A's independent oracle.
//
// std::unordered_map<key, VoxelAccumD>: a hash table with entirely
// different internals from this project's hand-rolled open-addressing
// table (chaining via linked buckets, a different hash function — the
// standard library's std::hash<unsigned long long>, not our Teschner
// spatial_hash — and dynamic resizing/rehashing we never see or control).
// Iterating points 0..n-1 IN ORDER and accumulating with += into
// map[key].sx (etc.) is a THIRD distinct summation order from both GPU
// Method A (hardware thread-scheduling order) and Method B/CPU-twin
// (sorted-key order) — and it accumulates in DOUBLE, which is more
// precise than either GPU path. This combination — different structure,
// different order, more precision — is what makes it a meaningful
// INDEPENDENT check on Method A's GPU result rather than a restatement of
// the same computation (the independence ruling's bar for the default,
// non-shared case).
// ---------------------------------------------------------------------------
void hashmap_downsample_cpu(int n, const float* xyz, float leaf,
                            std::unordered_map<unsigned long long, VoxelAccumD>& out)
{
    out.clear();
    // A light reserve (not exact — occupied voxel count is unknown until
    // we are done) avoids most of unordered_map's incremental rehashing on
    // a quarter-million-point input; purely a performance courtesy, has no
    // effect on the result.
    out.reserve(static_cast<size_t>(n) / 4);

    for (int i = 0; i < n; ++i) {
        const int32_t vx = voxel_coord(xyz[i * 3 + 0], leaf);
        const int32_t vy = voxel_coord(xyz[i * 3 + 1], leaf);
        const int32_t vz = voxel_coord(xyz[i * 3 + 2], leaf);
        const uint64_t key = pack_voxel_key(vx, vy, vz);

        // operator[] default-constructs a fresh VoxelAccumD{0,0,0,0} the
        // first time a key is seen (its in-class initializers handle that —
        // see kernels.cuh), then every subsequent point for the same voxel
        // finds the existing entry and accumulates into it.
        VoxelAccumD& acc = out[key];
        acc.sx += static_cast<double>(xyz[i * 3 + 0]);
        acc.sy += static_cast<double>(xyz[i * 3 + 1]);
        acc.sz += static_cast<double>(xyz[i * 3 + 2]);
        acc.count += 1u;
    }
}
