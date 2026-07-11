# Data — 01.16 Checkerboard/ChArUco detection acceleration for auto-calibration rigs

Provenance, licensing, and field documentation for everything under `data/` (CLAUDE.md §8).

## The rules (repo-wide, CLAUDE.md §8)

- **Synthetic-first.** Robotics data can almost always be synthesized with full ground truth;
  `../scripts/make_synthetic.py` is the default source, and synthetic data is **labeled synthetic
  everywhere it appears**.
- **Tiny committed sample.** `sample/` holds just enough committed data that the demo runs offline
  with zero downloads.
- **Never fabricate.** No made-up measurements, no invented ground truth passed off as real.

## This project's data

| Property | Value |
|----------|-------|
| Kind | 100% synthetic — a rendered ChArUco calibration board, 8 camera views + 1 board-free negative control |
| Generator | `../scripts/make_synthetic.py`, seed 42, xorshift32 (no numpy, no `std::uniform_real_distribution`) |
| License | Synthetic — repo MIT license applies (no third-party imagery, no third-party dictionary bit tables) |
| Size (committed) | ~700 KB total (10 PGMs + 5 small CSVs + 2 dictionary files) |
| Regenerate with | `python scripts/make_synthetic.py` (deterministic; re-running reproduces every byte) |

## What's here

```
sample/
  marker_dictionary.bin        24 x 9-bit ArUco-style marker codes + header (binary, little-endian)
  marker_dictionary.csv        the same codes, human-readable
  view00.pgm .. view07.pgm     the 8 rig views (320x240, P5 grayscale)
  negative_control.pgm         board-free textured clutter (the false-positive test)
  corners_truth.csv            every one of the 8 views' 35 inner-corner ground-truth positions
  poses_truth.csv              R, t (and yaw/pitch/roll/depth) per view
  intrinsics_truth.csv         the GROUND-TRUTH camera intrinsics (fx, fy, cx, cy) -- read ONLY by
                                main.cu's mini_calibration GATE, never by the detection/ordering
                                pipeline itself (that is the entire point of a calibration exercise)
  occluder_truth.csv           view07's occluder rectangle, in pixels
```

## Field documentation

### `marker_dictionary.bin`

Binary, little-endian: 5x `int32` header `(num_codes, bits_per_code, grid_n, min_distance,
correction_capacity)` followed by `num_codes` x `uint16` codes. `num_codes=24`, `bits_per_code=9`
(a 3x3 payload), `grid_n=5` (the marker's own 5x5 sampled grid including its 1-cell border ring).
`min_distance`/`correction_capacity` are MEASURED (not assumed) minimum pairwise Hamming distance
and its implied single-error-correction radius `floor((d-1)/2)` — see `kernels.cuh`'s
`generate_marker_dictionary` citation in `scripts/make_synthetic.py`.

### `marker_dictionary.csv`

`marker_id,square_bx,square_by,code_hex,code_binary_9bit` — `marker_id` in `[0,24)` is the
row-major (by, then bx) enumeration of the board's WHITE squares (`kernels.cuh`'s
`build_marker_id_table`); `square_bx`/`square_by` are that marker's home square in board-square
coordinates `[0,8) x [0,6)`.

### `view*.pgm`, `negative_control.pgm`

PGM P5, 320x240, 8-bit grayscale. Synthetic renders: an inverse-mapped (pixel -> board-plane)
rasterizer with 2x2 supersampling, a mild illumination gradient, a 5-tap Gaussian blur, and additive
Gaussian sensor noise (all documented in `make_synthetic.py`, all synthetic).

### `corners_truth.csv`

`view_index,i,j,x_px,y_px,visible` — one row per (view, inner corner). `i` in `[0,7)` is the
board's X-direction (column) index, `j` in `[0,5)` the Y-direction (row) index, both 0-based,
matching `kernels.cuh`'s `corner_board_xy(i,j) = ((i+1)*SQUARE, (j+1)*SQUARE)` meters convention
(one square of margin from the board's own outer edge). `x_px`/`y_px` are the analytic (noise-free)
pixel projection of that corner under the view's true camera pose. `visible` is `0` only for
`view07`'s occluded corners (or any corner projecting outside the frame, which does not occur in
this project's committed poses).

### `poses_truth.csv`

`view_index,name,yaw_deg,pitch_deg,roll_deg,depth_m,R00..R22,t0..t2` — the Euler angles used to
GENERATE each view's rotation (an arbitrary composition order, `Rz(roll)*Rx(pitch)*Ry(yaw)`, never
assumed by the detection pipeline, which only ever sees the resulting `R`), the nominal depth, and
the resulting row-major 3x3 rotation `R` + translation `t` (meters) — the camera-frame pose of the
board's own local origin.

### `intrinsics_truth.csv`

`fx,fy,cx,cy` — the pinhole intrinsics ACTUALLY used to render every view (units: pixels; `fx != fy`
on purpose, to exercise Zhang's method recovering both independently). `cx = (320-1)/2 = 159.5`,
`cy = (240-1)/2 = 119.5` (pixel-center convention). Read once by `main.cu`, strictly to GATE the
recovered `mini_calibration` result — never fed into detection, ordering, or marker decode.

### `occluder_truth.csv`

`view_index,name,x0_px,y0_px,x1_px,y1_px` — the flat rectangular occluder painted over `view07`
(a documented stand-in for "a cable or mounting bracket", `PRACTICE.md` section 1), in image pixels.

## SHA-256 checksums (of the committed files, printed by `make_synthetic.py`'s own run)

```
6464c6466397de06a3827ddd04a8ec54463042a5cb1bd38adcbdfd5a337778c8  marker_dictionary.bin
745619df6fd9a9930ffdc9b86c6ef9026d3e40f858395dd44e9224f336b8815f  marker_dictionary.csv
faf59351bc89c83dce8e3ca9ba9e045bf3ab791417c8e89c346305378214a875  corners_truth.csv
72fb5eececc8266371f0c8a54d4ba73bbf0b3abe6f2f2a9cf6bb58dfc3b134d7  poses_truth.csv
c876536b55e1b978ed6a3b29980bcaf96933ada74cfc326b7b36e61a21a1c5ac  intrinsics_truth.csv
7c8be1ec79d46a8dcd6fcefb818948e79ade4f6d00853d7dbf12eb76701a1cd5  occluder_truth.csv
90860a1e143a9e54a5c1af30ff25d4d597c0618f75b11cb190bad88b6b10283a  negative_control.pgm
fa0ea3b709649a373eb65522aa21589cfb095fb72e04e38c37ed97fc74b86dc3  view00.pgm
1588467ad11f1820ba899ecb47ba4f9c43025668415ae0615956c93f40c99e09  view01.pgm
84fbec4d8e183f8349ab4a49d012a90078e026429271a6746e7684fef0bbecbf  view02.pgm
bb30d3bee957e682a37452eb507680def4350167e9cf335c685ba03101e2e71e  view03.pgm
7064089eff3e65ed91287eaeb50e48975a5e4c665c39fa3238b5f96f8346cf24  view04.pgm
47c19d0933dbfe041602401e6e6388031ccb273b973e126ce189bad315cd2fcd  view05.pgm
e7d861f72ae1870a4fdfceb9ed4d069ea5345d62fc669add4e9344019c1dc3b0  view06.pgm
cc1dd85a972d3de3cc094bc7791a60733831aa4607be3d884296b30fac58c9cd  view07.pgm
```
