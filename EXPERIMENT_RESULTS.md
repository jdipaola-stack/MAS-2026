# Initial experiment results

These are small **five-run pilots**, not a statistically conclusive demonstration.
Both use 20 fish, seeds 42–46, a 60-second horizon, a 25% preferred light level,
and 2 seconds of uninterrupted residence to qualify as an arrival. Starting
conditions are matched across treatments. Births, deaths and perturbations are off.

## Informed minority

| Informed | Runs where all fish arrived | Mean last arrival, successful runs | Mean occupancy during final 12 s |
| --- | --- | --- | --- |
| 0% | 5/5 | 17.74 s | 17.4% |
| 10% | 4/5 | 15.63 s | 83.7% |
| 25% | 5/5 | 15.02 s | 98.9% |
| 100% | 5/5 | 12.32 s | 100% |

Uninformed fish can wander into the band by chance, so first-arrival success alone
does not establish informed leadership. The clearest pilot difference is sustained
occupancy: adding informed individuals helps the group remain in suitable light.
The 10% treatment did **not** always bring every fish to safety: one run reached
only 45% within the horizon. Its restricted mean completion time is 24.50 s, which
includes that noncompletion; the 15.63 s successful-only mean must not hide it.

Data: [runs](results/minority_pilot/runs.csv),
[aggregate](results/minority_pilot/aggregate.csv),
[chart](results/minority_pilot/comparison.png).

## Recruitment

| Condition | Runs where all fish arrived | Restricted mean completion time | Mean occupancy during final 12 s |
| --- | --- | --- | --- |
| Following + signaling | 5/5 | 24.32 s | 99.0% |
| Following without signaling | 5/5 | 21.18 s | 41.0% |
| N independent searches | 1/5 | 59.04 s | 36.3% |

Both social conditions completed the whole starting cohort more reliably than
independent searches in this pilot. **Adding the signal did not accelerate the
last arrival relative to ordinary following**: it was slower in all five paired
runs. Its main benefit here was keeping fish in the safe band.

Only one independent run completed all arrivals within 60 seconds. The other four
are censored, so 59.04 s is a restricted mean, not their true mean completion time.
Do not call the ratio of that capped mean to 24.32 s a proven speed-up factor.
The per-agent files record lone-search arrival times; the group's last arrival is
compared with the maximum over N solo searches, not with one solo search or the
first successful individual.

Data: [runs](results/recruitment_pilot/runs.csv),
[aggregate](results/recruitment_pilot/aggregate.csv),
[paired comparisons](results/recruitment_pilot/paired_comparison.csv),
[chart](results/recruitment_pilot/comparison.png).

## Reproduce

```sh
python research_experiments.py minority --fractions 0 .1 .25 1 --fish 20 --repeats 5 --steps 600 --output results/minority_pilot
python research_experiments.py recruitment --fish 20 --repeats 5 --steps 600 --output results/recruitment_pilot
```

Use more seeds, several group sizes, and multiple signal strengths before drawing
strong conclusions. Report simultaneous safety and retention alongside first
arrivals; these measure different properties of collective navigation.
