"""Repeat simulations at several alignment strengths, summarising after warm-up."""

import argparse
from pathlib import Path

import pandas as pd

from run import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignments", type=float, nargs="+", default=[0, 0.5, 1, 2])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--fish", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--light", action="store_true")
    parser.add_argument("--preferred-light", type=float, default=0.25)
    parser.add_argument("--light-tolerance", type=float, default=0.10)
    parser.add_argument("--light-weight", type=float, default=3.0)
    parser.add_argument("--no-lifecycle", action="store_true")
    parser.add_argument("--output", default="results/alignment_sweep")
    args = parser.parse_args()
    if args.repeats < 1 or not 0 <= args.warmup < args.steps:
        parser.error("Require repeats >= 1 and 0 <= warmup < steps")
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for setting, alignment in enumerate(args.alignments):
        for repeat in range(args.repeats):
            seed = args.seed + repeat
            model = run_experiment(args.steps, folder / f"setting_{setting}_seed_{seed}.csv",
                                   num_fish=args.fish, seed=seed, alignment_weight=alignment,
                                   light_enabled=args.light, preferred_light=args.preferred_light,
                                   light_tolerance=args.light_tolerance, light_weight=args.light_weight,
                                   lifecycle_enabled=not args.no_lifecycle)
            # Row zero is the initial condition. Exclude it and all warm-up ticks.
            frame = model.datacollector.get_model_vars_dataframe().iloc[args.warmup + 1:]
            row = {"alignment": alignment, "seed": seed, "warmup_steps": args.warmup}
            row.update({"completed_steps": model.steps, "extinct": not model.running})
            row.update(frame.drop(columns=["Time (s)"]).mean(numeric_only=True).to_dict())
            rows.append(row)
            print(f"Alignment {alignment:g}, seed {seed}: mean polarization {row['Polarization']:.3f}", flush=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(folder / "summary.csv", index=False)
    # Average and variability across independent runs, not pooled time samples.
    summary.drop(columns=["seed", "warmup_steps"]).groupby("alignment").agg(["mean", "std"]).to_csv(folder / "aggregate.csv")
    print(f"Saved individual runs, summary.csv and aggregate.csv in {folder}")


if __name__ == "__main__":
    main()
