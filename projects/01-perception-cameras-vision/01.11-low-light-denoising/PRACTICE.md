# 01.11 — Low-light denoising (bilateral, non-local means, fast BM3D variant): Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.

## 1. Building it — construction of the robot/part

The physical carrier for this project's code is the **camera module** — specifically, the sensor
package and the lens that decide how many photons this project's denoisers have to work with before
any algorithm runs at all.

- **The sensor die and package.** A CMOS image sensor is a grid of photosites, each a photodiode plus
  in-pixel amplification circuitry, bonded to (or, for BSI parts, stacked beneath) color-filter and
  microlens layers, wire-bonded or flip-chip mounted to a PCB, and sealed behind a cover glass or IR-cut
  filter window. Pixel PITCH (the physical size of one photosite) is the single biggest lever on
  low-light performance this project's noise model has: a larger pixel collects more photons per unit
  exposure time (higher `lambda`, better `sqrt(lambda)` SNR) at the cost of resolution or sensor area —
  a direct, physical trade this project's `kPeakElectrons` constant stands in for.
- **Back-side illumination (BSI) vs front-side (FSI).** FSI sensors route wiring OVER the photodiode,
  blocking some incoming light; BSI sensors move the wiring behind the photodiode, letting more photons
  reach it per unit area — the single biggest low-light manufacturing advance of the last ~15 years,
  and the reason "BSI CMOS" appears in nearly every modern phone/robotics camera spec sheet.
- **The lens and its aperture.** A faster lens (lower f-number) collects more light per unit exposure
  time at the cost of a shallower depth of field and (usually) more expensive, more complex glass —
  the optical-side lever on the same `lambda` budget the sensor-side pixel-pitch lever addresses.
- **Active illumination (IR).** Many low-light robotics cameras pair a rolling-shutter or global-
  shutter sensor with a synchronized near-IR (NIR, ~850-940 nm) illuminator, letting the robot "make its
  own light" invisibly to human eyes nearby. **Eye-safety orientation (not a certification claim):**
  any IR illuminator strong enough to matter is a laser-safety-adjacent hazard even though it is
  invisible — the relevant orientation standard is **IEC 62471** (photobiological safety of lamps and
  lamp systems), which classifies emitters by exposure-limit risk group; a fielded NIR illuminator
  needs its emission power and beam geometry checked against IEC 62471 (or the applicable laser-safety
  standard if the source is a laser diode rather than an LED array) BEFORE deployment near people —
  this is orientation, not a substitute for an actual photobiological safety assessment by someone
  qualified to perform one.
- **What breaks in the field:** condensation/fogging on the cover glass in cold, damp night
  operation (a real low-light-specific failure mode — the same "warehouse AMR night shift" scenario
  README's System Context names); IR illuminator LEDs degrading with age/heat, silently raising the
  effective noise floor this project's `kPeakElectrons` constant models as fixed; and connector/cable
  flex fatigue on a moving platform's camera mount, same as any vibration-exposed sensor package.

## 2. Real hardware — chips, parts, illustrative BOM

*Section dated 2026-07-11. All parts named below are **illustrative examples, never endorsements**;
part numbers and prices go stale — verify current before relying on any of them.*

| Piece | Illustrative choices (2026) | Role in this project's story |
|---|---|---|
| Compute for the denoiser | Jetson Orin class (edge, on-robot) / x86 + discrete RTX (reference machine: RTX 2080 SUPER) | The GPU kernels in `src/kernels.cu` — measured ~8.2 ms/frame for BM3D-lite at this project's teaching resolution (README "Rate honesty" extrapolates to camera resolution) |
| Low-light sensor | Hobby: Sony IMX-series BSI CMOS action-cam/board modules (~US$20-100); research: global-shutter machine-vision BSI sensors (Sony Pregius-S class, ~US$200-800/module); industrial: certified/ruggedized low-light modules with active cooling (~US$1k+) | The photon-collection hardware whose `kPeakElectrons`-class photon budget this project's noise model stands in for |
| Fast lens | f/1.4-f/1.8 C/CS-mount glass (hobby ~US$30-150; research/industrial machine-vision lenses ~US$200-1,000+) | The optical-side lever on signal (see §1) |
| NIR illuminator | Hobby: 850nm LED ring/bar (~US$10-40); research/industrial: pulsed/synchronized NIR illuminator modules with drive electronics (~US$100-500+) | Active low-light augmentation — IEC 62471 orientation applies (§1) |
| ISP / NR hardware | Fixed-function ISP block inside the SoC (Jetson's on-die ISP, a phone SoC's ISP, or a dedicated ISP chip) | Where a PRODUCTION version of this project's noise-reduction step would actually run — not a discrete GPU kernel (README "System context"/"Rate honesty") |
| Ambient/environmental | IP-rated enclosure + heater/defogger for cold, damp night operation | Addresses the condensation failure mode named in §1 |

## 3. Installation & integration — putting it on a real robot

- **Where this runs, physically:** on a ROS 2 robot, the camera driver node publishes raw or debayered
  frames (`sensor_msgs/Image`); this project's denoising step would run as a downstream image-processing
  node (or, more realistically per README "System context", INSIDE the camera's own ISP before the
  frame ever reaches ROS at all) — the demo's structure (load a frame, run one GPU pass, produce a
  denoised frame) maps directly onto a `image_transport`-style processing node's per-frame callback.
- **Real-time constraints, honestly:** README "Rate honesty" already measured that NLM and BM3D-lite,
  as implemented here, do NOT fit a 30 Hz / 1920x1080 budget — a real deployment either (a) runs a much
  cheaper filter (bilateral is the closest of the three to camera-rate-feasible, and even it needs the
  shared-memory tiling this project teaches to have a chance), (b) accepts a lower frame rate for a
  night-only operating mode, or (c) replaces the whole classical pipeline with a single-pass learned
  denoiser (README "Prior art"). This is a genuine engineering trade-off a fielded system must resolve
  explicitly, not a detail to discover in production.
- **Tuning against downstream consumers, not human eyes.** A PSNR/edge-preservation-tuned denoiser
  (this project's own gates) optimizes for what a HUMAN would judge as good image quality; a
  perception pipeline consuming the output (feature detectors, object detectors, a learned policy) may
  have DIFFERENT preferences — e.g., a feature detector might prefer a filter that preserves corner
  sharpness even at the cost of more residual noise elsewhere, which is not what `edge_preservation`'s
  single-edge metric measures. A real integration tunes (or re-trains) against the DOWNSTREAM task's
  own success metric, not against a human-legibility proxy like PSNR — a general lesson about
  perception-pipeline tuning this project's own gate design deliberately does not solve.
- **Calibration/bring-up ladder (CLAUDE.md §1's testing-ladder discipline, applied to a perception
  filter rather than a controller — the stakes are lower here since nothing moves, but the discipline
  still applies before trusting a denoised frame in a decision loop):**
  1. *Simulation* — this demo, plus adversarial-scene stress tests (near-zero-signal regions, extreme
     contrast) beyond this project's committed sample.
  2. *Bench, real camera, controlled lighting* — sweep exposure/gain down to the target low-light
     operating point on a real sensor; verify the ACTUAL noise characteristic (dark-frame/flat-frame
     stack, 01.09's own procedure) resembles this project's assumed Poisson+read model before trusting
     any of its tuned parameters (`kBilateralSigmaRange`, `kNlmH`, `kBm3dAssumedSigmaDn`) on real data.
  3. *On-robot, logged only* — run the denoiser on the robot's actual camera feed, log both raw and
     denoised streams, verify downstream consumers (feature detectors, detectors) behave sanely before
     letting their output influence any decision.
  4. *On-robot, in the loop* — only after (1)-(3), and only for a role appropriate to a
     PERCEPTION-QUALITY component (never as a safety-critical signal on its own — CLAUDE.md §1).
- **N/A here:** no fieldbus, no actuation — this project's "installation" ladder is a perception-
  pipeline bring-up, not a hardware-testing ladder for something that moves; stated per contract.

## 4. Business & regulatory context

- **Who needs low-light robotic vision:** security/patrol robots (explicitly a night-shift-heavy
  market), warehouse AMRs running 24-hour fulfillment operations (README's reference robot), agriculture
  robots operating at dawn/dusk to avoid daytime heat/worker overlap, search-and-rescue and inspection
  platforms working in unlit or damaged environments, and any indoor robot whose facility's lighting is
  deliberately dimmed for energy cost at night.
- **The players:** every machine-vision camera vendor (Sony, OnSemi/onsemi, OmniVision as sensor
  suppliers; Basler, FLIR/Teledyne, Intel RealSense and similar as module integrators) competes partly
  on low-light sensor performance; OpenCV's classical implementations (README "Prior art") are the
  default open-source baseline nearly every robotics team starts from; the learned-denoiser research
  community (Noise2Noise-descended work) is where the state of the art has been moving.
  Build-vs-buy: most robotics companies BUY the sensor/ISP and, at most, TUNE or fine-tune the NR stage
  rather than building a bilateral/NLM/BM3D pipeline from scratch — this project's value is
  understanding what that purchased/tuned stage is actually doing, not shipping this exact code.
- **Cost of getting it wrong:** a perception pipeline that silently degrades at night (rather than
  failing loudly) is a genuine safety and reliability risk — a warehouse AMR that mis-detects an
  obstacle at 2 a.m. because its feature detector was tuned on daytime SNR is a real operational
  failure mode this project's README "System context" names explicitly (01.04's FAST/Harris corners
  breaking under heavy shot noise). The mitigation is architectural: characterize and test the ACTUAL
  low-light operating envelope (the bring-up ladder in §3), not assume daytime tuning generalizes.
- **Regulatory:** this project's own output (a denoised image) is not itself a regulated artifact, but
  the SENSOR SYSTEM it feeds into inherits whatever regulatory path applies to the robot it rides on
  (see `../../../docs/SYSTEM_DESIGN.md` item 6's regulatory map — service robots: ISO 13482; AVs:
  ISO 26262/UL 4600; medical: IEC 60601 — the low-light imaging chain is a perception INPUT into
  whichever safety case governs the platform, not a separately certified component). The IR-illuminator
  eye-safety orientation in §1 (IEC 62471) is the one hardware-specific regulatory hook that is
  genuinely THIS project's business, since it is a direct consequence of choosing active low-light
  illumination.
- **Owning team:** perception / ISP (image-quality and camera-pipeline engineers — the same team
  named in README "System context" as owning 01.01/01.09); adjacent teams: the ML team (owns any
  learned-denoiser replacement, README "Prior art"), the mechanical/electrical team (owns the physical
  camera module and illuminator, §1-2), and the safety team (owns the eye-safety and operational-
  envelope sign-off for any active illumination, §1/§4).

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
