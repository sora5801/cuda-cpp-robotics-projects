#!/usr/bin/env python3
"""make_synthetic.py — synthetic sample-data generator for 26.01
(Topology optimization (SIMP) on GPU for lightweight links and brackets).

Why this script exists (CLAUDE.md paragraph 8: synthetic-first)
---------------------------------------------------------------
This project's "data" is not sensor data at all -- it is a PROBLEM
DEFINITION: a mesh size, a material, a volume target, and a set of boundary
conditions (which nodes are clamped, where the load is applied, which
elements are forced void). All of that is CONSTANTS -- no randomness, no
ground truth to synthesize -- so "synthetic-first" here means "write the
exact, deterministic scenario the demo runs" rather than drawing from a
generator. Two committed scenarios ship, matching the two load cases the
catalog bullet asks for ("lightweight links and brackets"):

  mbb_scenario.csv     -- the classic half-MBB beam (symmetry-reduced), the
                           textbook validation case for topology optimization:
                           its expected result -- a diagonal strut lattice --
                           is well known, making it a sanity check anyone who
                           has seen a topology-optimization paper recognizes.
  bracket_scenario.csv -- an L-shaped robot joint bracket: a square domain
                           with its top-right quadrant carved out (void/
                           passive), clamped along the remaining top edge
                           (bolted to the robot's frame) and loaded at the
                           bottom-right foot (a motor-flange load pulling on
                           the bracket) -- the design-for-robotics story this
                           flagship exists to tell.

Scenario file format (parsed by ../src/main.cu's load_topo_scenario()):
    NELX,<int>                          elements along x (fast axis)
    NELY,<int>                          elements along y (slow axis)
    E0_PA,<float>                       material Young's modulus (Pa)
    EMIN_RATIO,<float>                  SIMP floor: Emin = ratio * E0
    VOLFRAC,<float>                     target volume fraction of ACTIVE elements
    MAXOUTER,<int>                      outer SIMP iteration cap
    FIX_RECT,i0,j0,i1,j1,xflag,yflag    fix nodes in [i0,i1]x[j0,j1] (repeatable)
    LOAD,i,j,fx_N,fy_N                  point load at node (i,j) (repeatable)
    PASSIVE_RECT,ex0,ey0,ex1,ey1        elements forced permanently void (repeatable)
Node (0,0) is the TOP-LEFT of the domain; j increases DOWNWARD (row-major,
top row first -- matches the PGM artifact's image layout, main.cu documents
the same convention in kernels.cuh).

Usage
-----
    python make_synthetic.py                  # writes both scenarios with the
                                                # exact parameters the demo expects
    python make_synthetic.py --mbb-only
    python make_synthetic.py --bracket-only
"""

import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Material: Aluminum 6061-T6, an illustrative, commonly-used structural
# aluminum for lightweight robot brackets/links (CLAUDE.md paragraph 8:
# label real-world numbers honestly; this one is dated in PRACTICE.md and is
# not an endorsement of any specific supplier or temper -- verify current).
# ---------------------------------------------------------------------------
E0_PA = 68.9e9        # Young's modulus, Pa
EMIN_RATIO = 1.0e-3    # SIMP floor Emin/E0 -- see THEORY.md "Numerical considerations"
                       # for why this project uses 1e-3 rather than the more
                       # common 1e-9: with only a Jacobi (diagonal) preconditioner
                       # -- chosen for its clean, matrix-free GPU mapping -- a
                       # 1e-9 stiffness contrast makes the CG system too
                       # ill-conditioned to solve in the demo's time budget;
                       # 1e-3 still suppresses void-region stiffness to 0.1% of
                       # solid while keeping the linear system tractable. A
                       # documented, honest trade -- not hidden.
VOLFRAC = 0.4          # target material fraction of the ACTIVE design domain
MAXOUTER = 80          # outer SIMP iteration cap (README/THEORY document the
                       # measured convergence behavior at this cap)
LOAD_N = 5000.0        # illustrative point-load magnitude (N) -- SIMP's optimal
                       # TOPOLOGY is invariant to load magnitude (a linear-
                       # elasticity property: scaling F only rescales U and
                       # compliance quadratically, never the optimizer's
                       # decisions -- THEORY.md derives this), so this number
                       # is chosen only to "look like" a real bracket load.


def write_mbb_csv(out_path: Path, nelx: int = 120, nely: int = 40) -> None:
    """Write the classic half-MBB beam scenario (symmetry-reduced).

    Boundary conditions (the textbook setup -- see THEORY.md for the full
    picture including the mirrored other half):
      - Left edge (i=0, all j): ux=0 -- the SYMMETRY PLANE of the full MBB
        beam (only the right half is modeled; mirroring this half about
        x=0 reconstructs the classic simply-supported beam).
      - Bottom-right corner (i=nelx, j=nely): uy=0 -- the beam's roller
        support.
      - Top-left corner (i=0, j=0): downward point load -- the beam's
        midspan load (after mirroring, this becomes the beam's center).
    Expected result (README "Expected output"): a small number of diagonal
    struts connecting the load to the support -- the most famous picture in
    the topology-optimization literature.
    """
    lines = [
        "# SYNTHETIC data (a PROBLEM DEFINITION, not measurements) -- generated by",
        "# scripts/make_synthetic.py for project 26.01. The classic half-MBB beam",
        "# (symmetry-reduced): the textbook topology-optimization validation case.",
        f"# regenerate: python make_synthetic.py --mbb-only  (nelx={nelx}, nely={nely})",
        "# columns: see the file header comment in this script / ../src/main.cu's load_topo_scenario()",
        f"NELX,{nelx}",
        f"NELY,{nely}",
        f"E0_PA,{E0_PA:.6e}",
        f"EMIN_RATIO,{EMIN_RATIO:.6e}",
        f"VOLFRAC,{VOLFRAC}",
        f"MAXOUTER,{MAXOUTER}",
        f"FIX_RECT,0,0,0,{nely},1,0",              # symmetry plane: ux=0 along the whole left edge
        f"FIX_RECT,{nelx},{nely},{nelx},{nely},0,1", # roller support: uy=0 at the bottom-right corner
        f"LOAD,0,0,0,-{LOAD_N:.1f}",                # midspan load at the top-left corner, downward
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote MBB scenario to {out_path} ({nelx}x{nely} elements, labeled SYNTHETIC)")


def write_bracket_csv(out_path: Path, nelx: int = 80, nely: int = 80) -> None:
    """Write the robot L-bracket scenario.

    Boundary conditions (the design-for-robotics story):
      - Top edge, left half (i in [0, nelx/2], j=0): BOTH dofs fixed -- the
        bracket is bolted to the robot's structural frame along this edge.
      - Top-right quadrant (ex in [nelx/2, nelx), ey in [0, nely/2)):
        PASSIVE (forced void) -- the material the "L" shape excludes.
      - Bottom-right corner (i=nelx, j=nely): downward point load -- a
        motor-flange load transmitted through the bracket's foot.
    Expected result (README "Expected output"): material concentrates along
    a diagonal strut from the fixed edge toward the load, routing AROUND the
    reentrant (concave) corner where the L-notch begins -- that corner is a
    stress singularity in continuum elasticity, and SIMP naturally avoids
    over-committing material there (THEORY.md tells this story).
    """
    half_x = nelx // 2
    lines = [
        "# SYNTHETIC data (a PROBLEM DEFINITION, not measurements) -- generated by",
        "# scripts/make_synthetic.py for project 26.01. A robot L-bracket: bolted",
        "# along the top-left edge, loaded at the bottom-right foot by a motor flange.",
        f"# regenerate: python make_synthetic.py --bracket-only  (nelx={nelx}, nely={nely})",
        "# columns: see the file header comment in this script / ../src/main.cu's load_topo_scenario()",
        f"NELX,{nelx}",
        f"NELY,{nely}",
        f"E0_PA,{E0_PA:.6e}",
        f"EMIN_RATIO,{EMIN_RATIO:.6e}",
        f"VOLFRAC,{VOLFRAC}",
        f"MAXOUTER,{MAXOUTER}",
        f"PASSIVE_RECT,{half_x},0,{nelx-1},{nely//2-1}",   # top-right quadrant: void (the "L" notch)
        f"FIX_RECT,0,0,{half_x},0,1,1",                    # bolted to the frame: top edge, left half
        f"LOAD,{nelx},{nely},0,-{LOAD_N:.1f}",             # motor-flange load at the bottom-right foot
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote bracket scenario to {out_path} ({nelx}x{nely} elements, labeled SYNTHETIC)")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    sample_dir = script_dir.parent / "data" / "sample"

    parser = argparse.ArgumentParser(
        description="Generate the committed SIMP topology-optimization scenarios for project 26.01.")
    parser.add_argument("--mbb-only", action="store_true", help="write only mbb_scenario.csv")
    parser.add_argument("--bracket-only", action="store_true", help="write only bracket_scenario.csv")
    parser.add_argument("--nelx-mbb", type=int, default=120)
    parser.add_argument("--nely-mbb", type=int, default=40)
    parser.add_argument("--nelx-bracket", type=int, default=80)
    parser.add_argument("--nely-bracket", type=int, default=80)
    args = parser.parse_args()

    do_mbb = not args.bracket_only
    do_bracket = not args.mbb_only

    if do_mbb:
        write_mbb_csv(sample_dir / "mbb_scenario.csv", args.nelx_mbb, args.nely_mbb)
    if do_bracket:
        write_bracket_csv(sample_dir / "bracket_scenario.csv", args.nelx_bracket, args.nely_bracket)


if __name__ == "__main__":
    main()
