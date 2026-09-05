"""Run one experiment without opening a browser and save its measurements."""

import argparse
import json
from pathlib import Path

from model import GoldenShinerModel


def run_experiment(steps=1000, output="results/run.csv", **parameters):
    if steps < 1:
        raise ValueError("steps must be positive")
    model = GoldenShinerModel(**parameters)
    for _ in range(steps):
        if not model.running:
            break
        model.step()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.datacollector.get_model_vars_dataframe().to_csv(destination, index=False)
    # Keep all constructor defaults alongside results so the run can be reproduced.
    import inspect
    import mesa
    settings = {name: parameters.get(name, parameter.default)
                for name, parameter in inspect.signature(GoldenShinerModel).parameters.items()}
    destination.with_suffix(".json").write_text(
        json.dumps({"steps": steps, "completed_steps": model.steps, "mesa_version": mesa.__version__,
                    "parameters": settings, "events": model.events}, indent=2) + "\n"
    )
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--fish", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alignment", type=float, default=1.0)
    parser.add_argument("--light", action="store_true")
    parser.add_argument("--preferred-light", type=float, default=0.25)
    parser.add_argument("--light-tolerance", type=float, default=0.10)
    parser.add_argument("--light-weight", type=float, default=3.0)
    parser.add_argument("--no-lifecycle", action="store_true")
    parser.add_argument("--survival-time", type=float, default=60.0)
    parser.add_argument("--reproduction-time", type=float, default=20.0)
    parser.add_argument("--output", default="results/run.csv")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    model = run_experiment(args.steps, args.output, num_fish=args.fish, seed=args.seed,
                           alignment_weight=args.alignment, light_enabled=args.light,
                           preferred_light=args.preferred_light, light_tolerance=args.light_tolerance,
                           light_weight=args.light_weight, lifecycle_enabled=not args.no_lifecycle,
                           survival_time=args.survival_time, reproduction_time=args.reproduction_time)
    print(f"Saved {args.output} and its JSON settings ({model.steps * model.dt:.1f} simulated seconds)")
    print(f"Final state: {model.collective_state}; polarization: {model.metrics['Polarization']:.3f}; "
          f"rotation: {model.metrics['Rotation']:.3f}")
    if args.light:
        print(f"Preferred light: {model.preferred_light:.0%}; "
              f"fish in preferred band: {model.metrics['Fraction in preferred band']:.0%}; "
              f"mean light error: {model.metrics['Mean light error']:.3f}")
    print(f"Population: {model.num_fish}; births: {model.births}; "
          f"light deaths: {model.light_deaths}; predator deaths: {model.predator_deaths}")


if __name__ == "__main__":
    main()
