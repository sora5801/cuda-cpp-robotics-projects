# 02.18 — Weather filtering: snow/rain/dust outlier removal (DROR/LIOR): Practice

> `THEORY.md` teaches the math and the GPU; this file teaches the machine and the world around it
> (CLAUDE.md §4.3). It grounds the README "System context" section in the physical and commercial
> whole, citing [`../../../docs/SYSTEM_DESIGN.md`](../../../docs/SYSTEM_DESIGN.md) items 5–6.
> Depth scales with relevance — but every section is genuinely written or honestly N/A'd.
>
> *Sections below dated 2026-07-12. All parts, prices, and vendor names are **illustrative examples,
> never endorsements**; they go stale — verify current before relying on any of them.*

## 1. Building it — construction of the robot/part

This project's algorithm runs on data from a physical LiDAR unit, and winter/dust operation is
fundamentally a HARDWARE problem before it is a software one — no filter compensates for a sensor that
cannot see out of its own housing.

**The sensor housing.** A spinning or solid-state LiDAR's optical window is a precision-machined
polycarbonate or glass dome/window, sealed to an IP67-or-better rating against the housing body with an
O-ring or gasket. That window is the single point of failure for winter operation: a layer of frost,
packed snow, or a film of road spray dramatically attenuates or fully blocks every beam that passes
through it — no amount of DROR/LIOR filtering downstream can recover a return the window itself never
let out. Real systems address this with (a) a resistive HEATING ELEMENT bonded to or embedded in the
window (a thin-film heater, similar in principle to a rear-windshield defroster grid, driven by the
vehicle's 12/24/48 V bus and thermostatically or PWM-controlled), (b) a HYDROPHOBIC / OLEOPHOBIC
coating on the outer window surface (the same family of coating used on smartphone screens and
automotive windshields) that encourages water and snow to bead and shed rather than sheet or freeze
in place, and (c) mounting GEOMETRY that keeps the window as close to vertical as the sensor's field of
view allows, since a vertical surface sheds precipitation far better than a horizontal or upward-facing
one.

**Spray from lead vehicles — a mounting and maintenance problem, not just a software one.** On a road
vehicle following another vehicle in wet conditions, tire spray thrown up from the vehicle ahead is a
major, TRANSIENT source of exactly the "airborne scatterer" signal this project's filters target — a
dense, briefly very high density cloud of water droplets directly in the sensor's forward field of
view. Mounting position matters: sensors mounted low and forward (bumper-height) see far more of this
than roof-mounted sensors; a production vehicle's sensor layout is a real engineering tradeoff between
the wider ground-level field of view a low mount gives and the spray/road-debris exposure it accepts in
exchange. Regular cleaning (automated washer/wiper systems on the sensor window itself, borrowed
directly from automotive headlight-washer engineering) is standard practice on production
robotaxi/AV sensor suites for exactly this reason.

**What breaks in the field.** Cracked or delaminated hydrophobic coatings after repeated freeze-thaw
cycles or automated car-wash brushes; heater element failure (an open circuit in the resistive grid,
usually from thermal-cycling fatigue at a solder joint) leaving a housing that fogs/frosts and never
clears; connector corrosion at the sensor's power/data pigtail from repeated exposure to road salt —
the single most common winter-specific field failure on outdoor robotics sensor suites in
salt-treated-road climates.

## 2. Real hardware — chips, parts, illustrative BOM

The actual hardware this project's data would come from and this filter would run on:

**Sensors.** Spinning mechanical LiDAR (illustrative examples: Velodyne/Ouster/Hesai-class units, the
family this project's synthetic 16-beam model is patterned after) or solid-state/MEMS LiDAR
(illustrative: Livox, Hesai AT-class, Innoviz-class units) — research-grade units run roughly
$4,000-$15,000+ per unit; automotive-qualified solid-state units aimed at mass production target
sub-$1,000 at volume (both figures are order-of-magnitude and change often — verify current). Most
research-grade units expose a raw INTENSITY channel per return, a prerequisite for this project's
LIOR filter; cheaper/consumer-grade units sometimes do not, which is itself a real system-design
constraint worth knowing before committing to an intensity-dependent filter.

**Compute tier.** This project's own workload (brute-force O(n^2) search over ~1,000-1,500 points) is
trivial even for embedded compute; a real deployment's compute choice is driven by the REST of the
perception stack, not this filter alone. Illustrative tiers: hobby/research (a desktop x86 + a
discrete RTX-class GPU, this project's own development target); embedded/production (an NVIDIA
Jetson Orin-class SoC, roughly $400-$2,000 depending on module and carrier board, the common choice for
an AMR or a research AV prototype); industrial/production-AV (automotive-qualified GPU compute
platforms rated for the −40°C to +85°C automotive temperature range and functional-safety
certification, an entirely different cost and qualification tier than either of the above — the
"industrial-grade" alternative CLAUDE.md's BOM guidance asks every project to name).

**Housing/heating hardware.** Resistive window heater elements and controllers, and hydrophobic/
oleophobic optical coatings (both discussed in §1), are typically sourced as part of the LiDAR vendor's
own housing design on integrated units, or as an aftermarket/custom add-on when integrating a "bare"
LiDAR module into a custom enclosure — a real BOM line item distinct from the sensor itself on a
from-scratch integration.

**What this project's own code needs.** Nothing beyond the CUDA toolkit + C++17 standard library
(README "Build") — the filters themselves have no special hardware dependency; the hardware discussion
above concerns the SENSOR this project's data represents, not the filtering compute.

## 3. Installation & integration — putting it on a real robot

**Where in the pipeline.** This filter belongs immediately after the raw point cloud comes off the
LiDAR driver and BEFORE every consumer that assumes the cloud describes real structure — ground
segmentation, clustering, mapping, and costmap generation all sit downstream of it (README "System
context" names the specific projects: 02.04 clustering, 02.13/05.01 mapping, 23.01 costmaps). Running
it AFTER any of those would defeat the point: a clustering step fed unfiltered snow speckle has already
wasted compute grouping thousands of one-point "objects" before this filter ever gets a chance to
remove them.

**ROS 2 node shape.** A `weather_filter_node` subscribing to `sensor_msgs/msg/PointCloud2` (the real
counterpart to this project's `PointCloud` struct, `SYSTEM_DESIGN.md` §3.6) at the sensor's native scan
rate (10-20 Hz), publishing a filtered `sensor_msgs/msg/PointCloud2` on a downstream topic, with the
three filter parameter sets exposed as ROS 2 parameters (dynamically reconfigurable, so an operator or
a higher-level weather-classification node can retune without a rebuild). Runs on whichever compute
node hosts the rest of the perception pipeline — no dedicated hardware needed beyond that (§2).

**Parameter policy per weather mode — the operations question.** This project's constants
(`kSorStdMult`, `kDrorBeta`/`kDrorKMin`, `kLiorIntensityThresh`/`kLiorRadius`) were tuned against ONE
synthetic scenario at ONE illustrative particle density per weather type (THEORY.md "Numerical
considerations" — the dust-plume-core measurement shows exactly how density-sensitive DROR/LIOR's
behavior is). A real deployment needs a POLICY for which parameter set is active when: options range
from (a) a single, conservative, always-on parameter set tuned for the worst conditions the ODD
includes (simple, but leaves recall on the table in clear weather), to (b) an operator- or
dispatch-selected "weather mode" flipped manually before a shift in known conditions (the AMR-fleet-
operations answer: a site supervisor flips "winter mode" fleet-wide when snow starts), to (c) an
automatic weather-CLASSIFIER (fed by a rain/humidity sensor, a camera-based weather classifier, or the
LiDAR's own return-rate statistics) that retunes parameters in real time — the most capable and the
most engineering effort. Who owns that decision in production is itself a real question this project's
`combined` `[info]` measurement (README) speaks to directly: a UNION(DROR, LIOR) policy trades some
precision for materially higher recall, a choice a safety/operations team, not just a perception
engineer, should sign off on.

**Degraded-mode behavior, honestly.** This project's own `dust_plume_honesty` measurement (THEORY.md)
is the operationally important lesson: in sufficiently dense weather, BOTH filters' effectiveness
degrades, and a real system needs an honest DEGRADED-MODE behavior for when filtering confidence itself
drops — options include falling back to a lower operating speed, requesting a safety driver / remote
operator take-over, or an outright stop, never silently trusting an increasingly unreliable filtered
cloud. This is exactly the kind of judgment call a `31.xx`-style safety monitor (`SYSTEM_DESIGN.md`'s
cross-cutting safety-monitor layer) is meant to make, not something this filter node decides alone.

**The safe hardware-testing ladder** (CLAUDE.md §1's sim-validated-only caveat applies in full): every
threshold in this project was tuned and measured entirely in simulation against synthetic data.
Before any weather-filter tuning informs a real robot's behavior, the standard ladder applies —
simulation (this project's own scope) → hardware-in-the-loop replay of RECORDED real weather LiDAR
data (CADC/WADS, THEORY.md "Where this sits in the real world") with the perception stack running live
→ bench/tethered testing with a real sensor in an artificial snow/spray rig → limited, supervised,
low-speed field testing with a human safety driver and clear E-stop authority → staged operational
deployment. Nothing in this repository substitutes for that ladder.

## 4. Business & regulatory context

**Who needs this, and why it is commercially load-bearing.** Every outdoor autonomous system — AVs,
outdoor AMRs, agricultural and construction robots, delivery robots — has an Operational Design Domain
(ODD, `SYSTEM_DESIGN.md` §6.2's AV row) that either includes or excludes adverse weather. Expanding
that ODD to cover snow, rain, and dust is a DIRECT commercial driver: a robotaxi fleet, a warehouse
AMR, or an agricultural robot that must park itself the moment it starts snowing loses revenue-days and
market credibility every winter it cannot operate through. Weather filtering of the kind this project
teaches is one necessary (not sufficient — see PRACTICE §1's hardware side, and §3's degraded-mode
honesty) piece of that ODD-expansion story, which is why companies competing on all-weather capability
(named generically here — commercial and open-source players in this space include automotive AV
developers, LiDAR OEMs marketing weather-robust sensing, and the open Autoware ecosystem cited in
README "Prior art") invest specifically in it.

**What getting it wrong costs.** A filter tuned too AGGRESSIVELY (removing too many real points, this
project's `real_point_preservation` gate) risks deleting a genuine obstacle from the map or costmap —
a safety-relevant failure with real liability exposure if it contributes to a collision. A filter
tuned too CONSERVATIVELY (this project's SOR baseline, deliberately included to show what "too
conservative in the wrong way" looks like — high false-removal on legitimate far structure while
still under-removing weather noise, README "Expected output") degrades map quality and can trigger
unnecessary safety stops, a reliability/uptime cost rather than a direct safety one, but still
commercially expensive at fleet scale (unnecessary stops mean missed deliveries, unhappy riders,
support-ticket volume).

**Regulatory path.** This capability sits inside the perception layer of whichever regulatory
framework governs the host robot (`SYSTEM_DESIGN.md` §6.2's regulatory map, cited in full there — this
is an orientation pointer, not compliance guidance): for an autonomous-vehicle stack, ISO 26262
functional-safety process and UL 4600's safety-case argumentation would need to cover perception
degradation in adverse weather as an explicit hazard, typically backed by the kind of scenario-based
validation project 31.05 discusses; for a service/outdoor AMR, ISO 13482's hazard-analysis framing
covers "robot behaves unsafely because its perception degraded" as a hazard class regardless of the
specific weather-filtering algorithm used underneath.

**Where this work lives inside a robotics company.** Owning team: **perception**
(`SYSTEM_DESIGN.md` §5.1), specifically whichever sub-team owns the LiDAR pipeline; the parameter-
policy question in PRACTICE §3 above is a direct dependency for **controls & autonomy** (who consume
the filtered cloud downstream) and **QA & functional safety** (who must validate degraded-mode
behavior before any ODD expansion ships) — a weather-filtering regression is a genuinely cross-team
incident, not a perception-team-only bug, echoing README "System context"'s owning-team note.

---

*Didactic orientation only — **not** procurement, legal, or compliance advice. Where a topic truly
cannot apply, a section may be honestly N/A'd as "N/A because …" — but never padded, never fabricated.*
