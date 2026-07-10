# Changelog

This file is the concise index of every push to this repository. The convention (CLAUDE.md §7.1): every
push to `origin/main` gets exactly **one line** here, newest first, and each line links the push's
didactic note in [push-notes/](push-notes/) — where the full story lives: what was added and why it
matters to the learner, how to build and run it, what to study, what was verified, and known
limitations. The push-note is written *before* the push and included *in* it, so the repository always
explains its own latest state. Entry format:

```
- YYYY-MM-DD — short title — [push-note](push-notes/YYYY-MM-DD-NN-short-title.md)
```

## Pushes

<!-- Prepend new entries directly below this line (newest first). -->
- 2026-07-10 — flagship 28.01 real-time FEM soft arm done (30/505); EB 0.5%, f1 1.4%, real-time factor 1.69×, four setpoints tracked — [push-note](push-notes/2026-07-10-14-flagship-28-01-soft-arm-fem.md)
- 2026-07-10 — flagship 27.04 composite layup + Tsai–Wu done (29/505); four ~1e-7 analytic gates incl. the F66=3F11 isotropy proof — [push-note](push-notes/2026-07-10-13-flagship-27-04-composite-layup.md)
- 2026-07-10 — flagship 26.01 SIMP topology optimization done (28/505) — batch 1f complete; patch test 2.8e-6, textbook strut topologies — [push-note](push-notes/2026-07-10-12-flagship-26-01-topology-optimization-batch-1f-complete.md)
- 2026-07-10 — flagship 25.01 battery electro-thermal done (27/505); jR/5D gate convergence-calibrated; conduction-vs-boundary cooling finding — [push-note](push-notes/2026-07-10-11-flagship-25-01-battery-pack.md)
- 2026-07-10 — flagship 24.01 magnetostatic FEA + cogging sweeps done (26/505); Ampère gate 0.19%; cogging minimum found at arc 0.70 — [push-note](push-notes/2026-07-10-10-flagship-24-01-magnetostatic-fea.md)
- 2026-07-10 — flagship 21.04 speed-and-separation monitoring done (25/505); 0-frame transition offsets, zero false/missed stops; didactic-not-certified NOTICE in the output contract — [push-note](push-notes/2026-07-10-09-flagship-21-04-ssm.md)
- 2026-07-10 — flagship 20.01 GelSight tactile processing done (24/505) — batch 1e complete; slip onset within 1 frame of the Cattaneo–Mindlin model — [push-note](push-notes/2026-07-10-08-flagship-20-01-gelsight-batch-1e-complete.md)
- 2026-07-10 — flagship 19.01 antipodal grasp scoring done (23/505); analytic-object gates + 12/12 adversarial rejection — [push-note](push-notes/2026-07-10-07-flagship-19-01-grasp-scoring.md)
- 2026-07-10 — flagship 18.01 snake serpenoid sweeps done (22/505); anisotropy-necessity measured as a gate (6.3%) — [push-note](push-notes/2026-07-10-06-flagship-18-01-snake-serpenoid.md)
- 2026-07-10 — flagship 16.01 thruster allocation QP done (21/505); pseudoinverse-clip vs QP demonstrated (7.6% vs 70.5% surge retention) — [push-note](push-notes/2026-07-10-05-flagship-16-01-thruster-allocation.md)
- 2026-07-10 — flagship 14.02 traversability fusion done (20/505) — batch 1d complete; semantics vetoes flat water, rescues rough grass — [push-note](push-notes/2026-07-10-04-flagship-14-02-traversability-batch-1d-complete.md)
- 2026-07-10 — flagship 13.03 foothold scoring done (19/505); ramp gate 15.007° vs 15.00°; the twin-invisible shared-bug case study — [push-note](push-notes/2026-07-10-03-flagship-13-03-foothold-scoring.md)
- 2026-07-10 — flagship 12.01 TensorRT deploy done (18/505); first heavy-SDK project — §5 fallback path is the default build, TRT off-by-default — [push-note](push-notes/2026-07-10-02-flagship-12-01-tensorrt-deploy.md)
- 2026-07-10 — flagship 11.01 GPU LiDAR simulator done (17/505); hand-built BVH with proved depth bound; analytic radiometry gates exact — [push-note](push-notes/2026-07-10-01-flagship-11-01-lidar-simulator.md)
- 2026-07-10 — flagship 10.03 massively-parallel robot sim done (16/505) — batch 1c complete; 8.5B env-steps/s, energy-drift gate — [push-note](push-notes/2026-07-10-00-flagship-10-03-parallel-sim-batch-1c-complete.md)
- 2026-07-09 — flagship 03.01 FMCW radar cube + CFAR done (15/505); first cuFFT project; CA-vs-OS masking demonstrated — [push-note](push-notes/2026-07-09-10-flagship-03-01-fmcw-cfar.md)
- 2026-07-09 — flagship 02.06 GPU ICP done (14/505); point-to-plane 6 vs 48 iterations, sub-mm ground-truth errors — [push-note](push-notes/2026-07-09-09-flagship-02-06-gpu-icp.md)
- 2026-07-09 — flagship 01.02 stereo BM→SGM done (13/505); exact GPU/CPU equality over ~14M values; SGM beats BM by 34 points — [push-note](push-notes/2026-07-09-08-flagship-01-02-stereo-sgm.md)
- 2026-07-09 — flagship 23.01 GPU costmaps + DWA done (12/505) — batch 1b complete; byte-exact costmap verification — [push-note](push-notes/2026-07-09-07-flagship-23-01-costmaps-dwa-batch-1b-complete.md)
- 2026-07-09 — flagship 17.01 Lambert + porkchop done (11/505); grid minimum lands 0.14% above the closed-form Hohmann optimum — [push-note](push-notes/2026-07-09-06-flagship-17-01-lambert-porkchop.md)
- 2026-07-09 — flagship 15.01 minimum-snap batches done (10/505); constraint-definition audit over all 10k sets — [push-note](push-notes/2026-07-09-05-flagship-15-01-minimum-snap.md)
- 2026-07-09 — flagship 06.05 STOMP done (9/505); smooth-noise trajectory optimization, MPPI's planning cousin — [push-note](push-notes/2026-07-09-04-flagship-06-05-stomp.md)
- 2026-07-09 — flagship 31.01 HJ reachability done (8/505) — batch 1a complete; verified against the closed-form bang-bang solution — [push-note](push-notes/2026-07-09-03-flagship-31-01-hj-reachability-batch-1a-complete.md)
- 2026-07-09 — flagship 22.01 100k-agent swarm simulator done (7/505); lockstep oracle + emergence gates — [push-note](push-notes/2026-07-09-02-flagship-22-01-swarm-simulator.md)
- 2026-07-09 — flagship 05.01 TSDF fusion + marching cubes done (6/505); analytic ground truth; fabricated bounds caught and replaced with measured ones — [push-note](push-notes/2026-07-09-01-flagship-05-01-tsdf-fusion.md)
- 2026-07-09 — flagship 04.01 massive particle filter done (5/505); carries unverified in-progress src for 05.01/22.01/31.01 — [push-note](push-notes/2026-07-09-00-flagship-04-01-particle-filter.md)
- 2026-07-08 — flagship 08.01 MPPI cart-pole done (4/505) — the four Phase-1 foundations are complete — [push-note](push-notes/2026-07-08-04-flagship-08-01-mppi-foundations-complete.md)
- 2026-07-08 — flagship 07.09 jump-flooding Voronoi/distance transforms done (3/505); 1+JFA after the oracle caught plain JFA — [push-note](push-notes/2026-07-08-03-flagship-07-09-jump-flooding.md)
- 2026-07-08 — flagship 09.01 batched forward kinematics done (2/505) — [push-note](push-notes/2026-07-08-02-flagship-09-01-batched-fk.md)
- 2026-07-08 — flagship 33.01 batched small-matrix linalg done (1/505); template Debug -G fix — [push-note](push-notes/2026-07-08-01-flagship-33-01-small-matrix-linalg.md)
- 2026-07-08 — bootstrap: contract, docs, tools, catalog (505 projects / 36 domains), all project skeletons — [push-note](push-notes/2026-07-08-00-bootstrap.md)
