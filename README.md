# Golden-shiner schooling in Python + Mesa

A complete educational starting point for asking: **How do local fish interactions
produce collective movement?** Each arrow is one Mesa agent. The simulation includes
repulsion, attraction, optional alignment, limited vision, changing speed, tank walls,
and an optional light gradient. The school forms from these individual rules.
The interactive app starts with light enabled: **dark at the bottom, bright at
the top**, with a preferred light level of **25%**. The green band shows the
acceptable 15–35% range. Fish seek this band while continuing to swim together.
Navigation requires shared light readings from schoolmates. Fish lose health in
unsuitable light, recover and reproduce in suitable light, and can be disturbed
using the **Predator attack**, **Storm**, and **Clear disturbances** buttons.

## Start the interactive simulation

The local `.venv` environment has been prepared in this folder. From a terminal here:

```sh
source .venv/bin/activate
solara run app.py
```

Open **http://localhost:8765**. Press **Play** to run, pause to inspect, or use the
step control. Change the model parameters and press **Reset** to apply them to a new
run. The random seed makes runs repeatable. Download the measurements using the CSV
button. Stop the server with Ctrl+C in the terminal.

To install on another machine (Python 3.12+):

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
solara run app.py
```

On Windows, activate with `.venv\Scripts\activate` instead. Mesa is pinned to 3.5.1;
`requirements-lock.txt` records the full macOS/Python 3.13 environment used for
verification. On the same platform, install with `-r requirements-lock.txt` to
reproduce it; use `requirements.txt` on other platforms.

## Files

| File | Purpose |
| --- | --- |
| `model.py` | Fish agents, movement, environment, group measurements |
| `app.py` | Mesa/Solara interactive controls, tank view, plots, CSV download |
| `run.py` | Run one simulation without a browser; save CSV and JSON settings |
| `experiments.py` | Repeat runs over alignment strengths and summarise results |
| `tests/test_model.py` | Behaviour, numerical limits, reproducibility, metrics and export checks |
| `tests/test_app.py` | Actual Step/Reset controls, parameter changes and light-mode plotting |

## What happens at each time step?

1. Take a snapshot of every fish's position, heading and speed.
2. Each fish detects neighbours within its distance and viewing-angle limits.
3. Very close fish trigger repulsion, including in the rear blind region. When
   repulsion is active it takes priority over attraction and alignment.
4. Otherwise, the fish combines attraction toward visible neighbours and optional
   alignment with their headings. Previous heading provides movement persistence.
5. Add wall avoidance, collective light-seeking when enough neighbours share
   readings, and a small
   random turn. Limit turning and acceleration.
6. Calculate speed from the baseline (or distance from preferred brightness), then adjust it in
   response to visible neighbours ahead or behind.
7. Once **all fish have planned**, update all positions. Reflect at a wall if
   anticipatory avoidance was insufficient.
8. Apply predator captures, update health and safe residence time, remove dead
   fish, and let eligible adult pairs produce juveniles. Rebuild the population
   snapshot and record measurements. Stop if the population becomes extinct.

Schooling requires at least two visible neighbours by default. Smaller local
groups are labelled roaming. A fish near a predator switches to escaping.
These individual states are separate from group labels such as milling.

All distances use **body lengths (BL)**. One tick represents **0.1 seconds** by
default, and the tank is 60 × 40 BL. A fish's body length is the unit of distance;
arrows do not represent solid bodies. To express a length in centimetres, multiply
by the body length measured in your chosen dataset.

## Main parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `num_fish` | 60 | Number of agents |
| `interaction_radius` | 8 BL | Maximum distance for social detection |
| `repulsion_radius` | 1 BL | Distance below which avoidance has priority |
| `repulsion_weight` | 3 | Strength of short-range avoidance |
| `attraction_weight` | 0.6 | Strength of turning toward neighbours |
| `alignment_weight` | 1 | Strength of matching neighbours' headings; 0 disables it |
| `vision_angle` | 300° | Visible angular region centred on heading |
| `base_speed` | 2 BL/s | Target speed with light disabled |
| `min_speed`, `max_speed` | 0.5, 4 BL/s | Swimming-speed limits |
| `speed_regulation` | 1 | Strength of social speed adjustment, in BL/s |
| `max_acceleration` | 2 BL/s² | Limit on speed change |
| `max_turn` | 2.5 rad/s | Limit on ordinary turning |
| `noise` | 0.15 rad/√s | Random angular perturbation strength |
| `light_enabled` | False in Python; True in the app | Enable preferred-light steering and speed response |
| `preferred_light` | 0.25 | Desired brightness: 0 is fully dark; 1 is fully bright |
| `light_tolerance` | 0.10 | Acceptable deviation from the preferred level (10 percentage points) |
| `light_weight` | 3 | Strength of steering toward preferred light; 0 disables this steering |
| `school_min_neighbors` | 2 | Number of visible schoolmates required for light-direction estimation |
| `lifecycle_enabled` | True | Enable health loss, recovery and reproduction when light is enabled |
| `survival_time` | 60 s | Time for full health to drain just outside the safe band; extreme light drains it faster |
| `recovery_time` | 15 s | Time to recover from zero to full health inside the band |
| `reproduction_time` | 20 s | Uninterrupted time in the band required before breeding |
| `maturity_time` | 30 s | Age at which a juvenile becomes eligible to breed |
| `max_population` | 200 | Population cap; prevents unlimited growth and excessive rendering costs |
| `seed` | 42 | Seed for reproducible random choices |

Constructor parameters not exposed as sliders can be changed in Python. Examples:

```python
from model import GoldenShinerModel

model = GoldenShinerModel(num_fish=100, alignment_weight=0, light_enabled=True, seed=7)
for _ in range(1000):
    if not model.running:
        break
    model.step()
data = model.datacollector.get_model_vars_dataframe()
print(data.tail())
```

## Understanding the measurements

- **Polarization:** length of the average unit heading; 1 means all fish share a
  heading. Low polarization can mean disorder or organised circling.
- **Rotation:** absolute average signed cross product of each unit radial vector
  (from the group centre) with its unit heading; 1 means common tangential circling.
  Fish at the exact centre contribute zero.
- **Cohesion:** fraction of fish in the largest connected component. A connection
  exists when two fish are within the interaction radius, regardless of visibility.
- **Nearest neighbour:** average distance to each fish's closest neighbour. It is
  undefined (blank/NaN) for a single fish.
- **Group radius:** root-mean-square distance from the group's centre.
- **Density:** fish count divided by the axis-aligned bounding-box area of the
  whole population; each side is floored at 1 BL. This rough statistic includes
  empty gaps when groups split and is sensitive to school orientation.
- **Mean speed:** average swimming speed in BL/s.
- **Mean brightness:** average local light intensity; lower means darker.
- **Fraction in dark region:** fraction occupying the bottom, darkest quarter of the tank.
- **Mean light error:** average of each fish's absolute difference from preferred
  brightness. This avoids mistaking half the fish being too bright and half too
  dark for a successful school just because their mean is correct.
- **Fraction in preferred band:** fraction within `preferred_light ± light_tolerance`.
  All light measures are undefined when light is disabled.
- **Population, births, light deaths, predator deaths:** current fish count and
  cumulative demographic totals. Population = initial fish + births − deaths.
- **Mean health:** average surviving fish health, between zero and one.
- **Schooling fraction:** fraction with enough visible neighbours during their
  last sensing step; being close to the whole school is not sufficient.
- **Predator active, storm active:** 1 during an event and 0 otherwise.

An extinct population has zero counts and group-order measures; average light
and nearest-neighbour distance are undefined. The interface stops and offers Reset.

Group labels use illustrative thresholds: polarized means polarization > 0.65
and rotation < 0.35; milling means the reverse; swarm means both < 0.35. Other
combinations are transitions. Cohesion < 0.8 is labelled fragmented first. A
single fish is labelled solitary. These labels describe snapshots, not a validated
classification of stable biological states.

## Light experiment

Brightness varies **only along the y axis**: 0 at the bottom and 1 at the top.
Fish at the same height experience the same light, whatever their x coordinate.
The preferred 25% level lies at y = 10 BL in the default 40 BL-high tank. Its
acceptable 15–35% band spans y = 6–14 BL.

Each fish senses **only its local brightness**, not the true environmental
gradient. Finding a direction requires information from the local school:

1. Detect visible neighbours within the interaction radius.
2. With at least two visible neighbours, compare their brightness readings with
   the focal fish's reading. Fit a local light gradient from the differences in
   position and brightness using least squares.
3. Use that shared estimate to turn toward preferred brightness. A one-second
   look-ahead reduces overshoot. Social and wall responses still apply.
4. Without enough neighbours, or without spatial light contrast, there is **no
   directional light-seeking**. Solitary fish and pairs roam, although they can
   still change speed in response to the brightness they experience.

This capability depends on local observations, not the total population counter:
a lone fish cannot borrow knowledge from a distant school. A solitary fish can
still reach good light by chance and remain there longer by slowing down.

```text
light intensity = clip(y / tank height + storm offset, 0, 1)
mismatch = abs(light intensity − preferred light)
maximum possible mismatch = max(preferred light, 1 − preferred light)
target speed = minimum speed + (mismatch / maximum possible mismatch)
               × (maximum speed − minimum speed)
```

Social speed regulation then adjusts this target, and speed/acceleration limits
are applied. Fish keep moving inside the band; the model does not freeze them
or move them instantly to the target. Baseline speed controls their initial speed
in light mode. The environment determines the ongoing target speed.

**These are modelling assumptions:** fish share local brightness information,
and at least three locally connected fish are needed to estimate a direction.
The 25% value is the project's selected preference, not a measured biological
optimum for golden shiners. The information-sharing rule provides a concrete
collective mechanism; it is not claimed to be an experimentally fitted fish rule.

Use the preferred-light slider to move the band, acceptable deviation to change
its width, and light-seeking strength to adjust the environmental response. Press
Reset after changing parameters. Measure both occupancy of the band and cohesion:
fish can share a light preference while still splitting into separate groups.
Strong social weights, weak light response, high noise, extreme targets near walls,
or very narrow tolerances can reduce success. Compare settings over multiple seeds.

## Survival and reproduction

Each fish starts with health 1. Outside the preferred band, health decreases at
`(1 + excess mismatch / maximum mismatch) / survival_time` per second. A fish
dies when health reaches zero. Greater light mismatch therefore shortens survival;
time spent in the preferred band restores health at `1 / recovery_time` per second.

Two nearby adults can produce one juvenile when **both** have health at least 0.8
and have spent 20 uninterrupted seconds in the preferred band. Leaving the band
resets residence time; breeding resets both parents' timers. A fish can participate
in at most one pairing per tick, and escaping fish cannot breed. Newborns start
near their parents, join movement on the next tick, and must reach age 30 seconds
before breeding. Initial fish are mature adults. These are abstract parent pairs,
not a model of fish sex, eggs, incubation, or actual generation times.

Dead fish are removed from both the spatial environment and Mesa's agent registry.
Population is capped at 200 by default. Disable light survival and reproduction
for movement-only comparisons; predator captures remain active if you introduce
a predator. Light-dependent health and breeding are suspended when light is off.
The compressed lifecycle times are for experimentation, not biological predictions.

## Perturbation buttons

- **Predator attack:** introduce a red predator marker near the school for 12
  simulated seconds. It pursues the nearest fish at up to 5 BL/s. Fish within
  12 BL flee and accelerate; the predator can capture one fish within 0.9 BL at
  most once per second. Escaping temporarily takes priority over light-seeking.
- **Storm:** for 25 simulated seconds, reduce light intensity by 0.35 and apply a
  current of (1.0, 0.3) BL/s. The 25% optimum moves upward from y = 10 to y = 24 BL;
  the background and green band update immediately. The original environment
  returns automatically when the storm ends.
- **Clear disturbances:** immediately remove the predator and restore normal
  light and current. It does not restore dead fish or reset the population.

Buttons act on the current run without Reset. Predator and storm can coexist;
clicking an already active event cannot stack duplicates. Event durations advance
only as simulated time advances, so press Play or Step after introducing one.
Events are recorded with their simulation time and duration and can be downloaded
using **Download event log**. In Python, the same actions are available as
`model.trigger_predator()`, `model.trigger_storm()`, and `model.clear_disturbances()`.

## Run and compare experiments

One run, with a CSV time series and a JSON file containing all constructor settings:

```sh
python run.py --steps 1000 --fish 60 --seed 42 --output results/basic.csv
python run.py --steps 1000 --light --preferred-light 0.25 --output results/light.csv
python run.py --steps 1000 --light --no-lifecycle --output results/movement_only.csv
```

Repeat each alignment setting with five seeds, discard the first 300 steps,
and summarise the remaining measurements:

```sh
python experiments.py --alignments 0 0.5 1 2 --repeats 5 --steps 1000 --warmup 300
python experiments.py --light --preferred-light 0.25 --light-weight 3 --output results/light_sweep
```

The same seeds are used across settings to provide matching initial conditions.
`summary.csv` contains one time-averaged row per run; `aggregate.csv` contains
the mean and sample standard deviation across runs. Standard deviation is blank
when only one repeat is requested. Raw time series and JSON settings are also
saved. Existing files with the same names are overwritten; use `--output` to
choose a new results folder for another experiment.

An initial CSV row records time zero, so 1,000 completed steps produce 1,001 rows.
Runs stop early at extinction. JSON settings include completed steps, and sweep
summaries include an extinction flag. A run extinct before warm-up has no
post-warm-up averages; these remain blank rather than inventing values. Runtime
grows approximately with the square of fish count; start with the default 60 fish.

## Scientific scope

This is a working research scaffold, **not a calibrated golden-shiner model**.
The defaults are explicit assumptions, and no particular collective state or
improvement in navigation is guaranteed. A convincing animation alone does not
validate the individual rules.

The model uses additive social responses, explicit optional alignment and
preferred-light steering, a simple viewing-angle mask, and point-like agents.
It omits visual occlusion, experimentally
fitted interaction functions, three-body effects, and food. Predators and storms
are simplified perturbations. Repulsion
encourages separation but is not a hard collision constraint. Initial positions
can be close together. Wall reflection can change heading faster than the usual
turn limit, so inspect boundary effects when comparing with experiments.

Useful next steps are fitting interaction distances and speed responses to a
specific dataset, checking sensitivity to time step and tank size, comparing
models with/without explicit alignment, and comparing measured distributions
with real trajectories.

Scientific background:

- [Collective states in golden shiners](https://pmc.ncbi.nlm.nih.gov/articles/PMC3585391/)
- [Measured individual interactions](https://pmc.ncbi.nlm.nih.gov/articles/PMC3219116/)
- [Visual detection and obstruction](https://pmc.ncbi.nlm.nih.gov/articles/PMC8261228/)
- [Collective gradient sensing](https://www.nature.com/articles/s41598-018-26037-9)
- [Mesa visualization documentation](https://mesa.readthedocs.io/stable/apis/visualization.html)

## Check the implementation

```sh
python -m pytest -q
```

An optional browser regression checks that all 60 fish are actually drawn, and
that Step, Play/Pause and the light setting work in the webpage. With the server
running in another terminal:

```sh
python -m pip install -r requirements-browser.txt
python -m playwright install chromium
MAS_BROWSER_URL=http://127.0.0.1:8765 python -m pytest tests/test_browser.py -q
```

You can set `MAS_BROWSER_EXECUTABLE` to an installed Chrome executable instead
of downloading Chromium. The browser check is skipped during normal test runs.

If the controls appear but the fish and charts are blank, reinstall with
`python -m pip install -r requirements.txt`, restart Solara, and refresh the page.
The visualization dependencies are constrained to Vue 2 because the draggable
plot panel fails under Vue 3 with the browser error `GlobalVue.use is not a function`.
