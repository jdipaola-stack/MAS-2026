"""Controlled minority-leadership and recruitment experiments.

Examples:
    python research_experiments.py minority --fractions 0 .05 .1 .25 1
    python research_experiments.py recruitment --repeats 10

No births, deaths, or perturbations. Treatments use matching starting positions,
headings, tank, light field, and time horizon. Non-arrivals remain censored.
"""

import argparse
import inspect
import json
from pathlib import Path

import mesa
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from model import GoldenShinerModel


def run_trial(parameters, steps, folder, name):
    model = GoldenShinerModel(**parameters)
    # Save starting conditions as well as seeds so comparisons can be audited.
    initial = pd.DataFrame({"agent_id": [f.unique_id for f in model.fish],
                            "x": model.positions[:, 0], "y": model.positions[:, 1],
                            "heading": model.headings, "informed": [f.informed for f in model.fish]})
    for _ in range(steps):
        model.step()
    horizon = steps * model.dt
    summary = model.arrival_summary()
    summary.update({"condition": name, "seed": parameters["seed"], "n": model.initial_population,
                    "informed_count": sum(model.cohort_roles.values()),
                    "actual_informed_fraction": sum(model.cohort_roles.values()) / model.initial_population,
                    "horizon_s": horizon, "final_occupancy": model.metrics["Fraction in preferred band"],
                    "late_occupancy": model.datacollector.get_model_vars_dataframe()["Fraction in preferred band"].tail(max(1, steps // 5)).mean(),
                    # min(T, horizon) is a restricted observation, not an invented arrival.
                    "restricted_completion_s": summary["time_all_s"] if summary["success"] else horizon})
    arrivals = pd.DataFrame([{"agent_id": identifier, "informed": model.cohort_roles[identifier],
                              "arrival_s": arrival, "observed": arrival is not None,
                              "censored_at_s": None if arrival is not None else horizon}
                             for identifier, arrival in model.arrival_times.items()])
    arrivals["arrival_s"] = pd.to_numeric(arrivals["arrival_s"], errors="raise")
    arrivals["restricted_arrival_s"] = arrivals["arrival_s"].fillna(horizon)
    summary["individual_success_rate"] = arrivals["observed"].mean()
    summary["mean_individual_arrival_successful_s"] = arrivals["arrival_s"].mean()
    summary["mean_individual_restricted_s"] = arrivals["restricted_arrival_s"].mean()
    destination = folder / f"{name}_seed_{parameters['seed']}"
    initial.to_csv(destination.with_name(destination.name + "_initial.csv"), index=False)
    arrivals.to_csv(destination.with_name(destination.name + "_arrivals.csv"), index=False)
    model.datacollector.get_model_vars_dataframe().to_csv(destination.with_name(destination.name + ".csv"), index=False)
    resolved = {key: value.default for key, value in inspect.signature(GoldenShinerModel).parameters.items()}
    resolved.update(parameters)
    destination.with_name(destination.name + ".json").write_text(json.dumps(
        {"parameters": resolved, "steps": steps, "mesa_version": mesa.__version__,
         "success_definition": "Every starting fish has spent safety_hold_time consecutive seconds in the band"}, indent=2) + "\n")
    return summary


def aggregate_runs(runs):
    """Keep successes and failures in rate/restricted-time denominators."""
    rows = []
    for condition, frame in runs.groupby("condition", sort=False):
        rows.append({"condition": condition, "runs": len(frame), "successes": int(frame["success"].sum()),
                     "actual_informed_fraction": frame["actual_informed_fraction"].mean(),
                     "success_rate": frame["success"].mean(),
                     "mean_last_arrival_successful_s": frame["time_all_s"].mean(),
                     "mean_restricted_completion_s": frame["restricted_completion_s"].mean(),
                     "mean_fraction_arrived": frame["fraction_arrived"].mean(),
                     "mean_uninformed_fraction_arrived": frame["uninformed_fraction_arrived"].mean(),
                     "mean_late_occupancy": frame["late_occupancy"].mean(),
                     "mean_individual_restricted_s": frame["mean_individual_restricted_s"].mean()})
    return pd.DataFrame(rows)


def paired_recruitment_comparison(runs):
    rows = []
    for seed, frame in runs.groupby("seed"):
        indexed = frame.set_index("condition")
        treatment = indexed.loc["recruitment"]
        for baseline in ("social_no_signal", "independent"):
            control = indexed.loc[baseline]
            both = bool(treatment["success"] and control["success"])
            rows.append({"seed": seed, "baseline": baseline,
                         "recruitment_success": treatment["success"], "baseline_success": control["success"],
                         "both_completed": both,
                         "completion_speedup_baseline_over_recruitment": control["time_all_s"] / treatment["time_all_s"] if both else None,
                         "restricted_seconds_saved": control["restricted_completion_s"] - treatment["restricted_completion_s"]})
    return pd.DataFrame(rows)


def save_plot(aggregate, folder):
    figure = Figure(figsize=(11, 4.5), layout="constrained")
    success, time = figure.subplots(1, 2)
    labels = aggregate["condition"]
    success.bar(labels, aggregate["success_rate"], color="#0f766e")
    success.set(ylim=(0, 1.05), ylabel="Fraction of runs completed", title="Did every starting fish reach safety?")
    time.bar(labels, aggregate["mean_restricted_completion_s"], color="#2563eb")
    time.set(ylabel="Seconds (capped at observation horizon)", title="Restricted completion time · lower is better")
    for axis in (success, time):
        axis.tick_params(axis="x", labelrotation=30)
        axis.grid(axis="y", alpha=0.2)
    figure.savefig(folder / "comparison.png", dpi=160)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=["minority", "recruitment"])
    parser.add_argument("--fractions", nargs="+", type=float, default=[0, 0.05, 0.1, 0.25, 1])
    parser.add_argument("--fish", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hold-time", type=float, default=2.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or args.steps < 1 or args.fish < 1 or not np.isfinite(args.hold_time) or args.hold_time <= 0:
        parser.error("fish, repeats, steps and hold-time must be positive")
    if any(not np.isfinite(f) or not 0 <= f <= 1 for f in args.fractions):
        parser.error("fractions must be between 0 and 1")
    if len(set(args.fractions)) != len(args.fractions):
        parser.error("fractions must be distinct")
    folder = args.output or Path("results") / args.experiment
    folder.mkdir(parents=True, exist_ok=True)
    conditions = ([(f"informed_{fraction:g}", {"informed_fraction": fraction}) for fraction in args.fractions]
                  if args.experiment == "minority" else
                  [("recruitment", {"social_enabled": True, "recruitment_enabled": True}),
                   ("social_no_signal", {"social_enabled": True, "recruitment_enabled": False}),
                   ("independent", {"social_enabled": False, "recruitment_enabled": False})])
    rows = []
    for repeat in range(args.repeats):
        for label, treatment in conditions:
            parameters = {"num_fish": args.fish, "seed": args.seed + repeat,
                          "max_population": max(200, args.fish), "light_enabled": True,
                          "lifecycle_enabled": False, "start_away": True,
                          "navigation_mode": "Informed minority" if args.experiment == "minority" else "Recruitment",
                          "safety_hold_time": args.hold_time, **treatment}
            row = run_trial(parameters, args.steps, folder, label)
            rows.append(row)
            print(f"{label}, seed {parameters['seed']}: {row['fraction_arrived']:.0%} arrived; "
                  f"last arrival = {row['time_all_s'] if row['success'] else 'not observed'}", flush=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(folder / "runs.csv", index=False)
    aggregate = aggregate_runs(runs)
    aggregate.to_csv(folder / "aggregate.csv", index=False)
    if args.experiment == "recruitment":
        paired_recruitment_comparison(runs).to_csv(folder / "paired_comparison.csv", index=False)
    save_plot(aggregate, folder)
    print(aggregate.to_string(index=False))
    print(f"Saved results and comparison.png in {folder}")


if __name__ == "__main__":
    main()
