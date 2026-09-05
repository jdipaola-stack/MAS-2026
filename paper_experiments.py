"""Sweep the 2018 model's environmental weight and school size with matched fields."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from model import GoldenShinerModel


def interpolate_grid(grid, positions, width, height):
    """Bilinear sampling of a y-by-x field, including the arena boundary."""
    scaled = np.asarray(positions) / [width, height] * [grid.shape[1] - 1, grid.shape[0] - 1]
    scaled = np.clip(scaled, [0, 0], [grid.shape[1] - 1, grid.shape[0] - 1])
    x, y = scaled[..., 0], scaled[..., 1]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, grid.shape[1] - 1), np.minimum(y0 + 1, grid.shape[0] - 1)
    fx, fy = x - x0, y - y0
    return ((1 - fx) * (1 - fy) * grid[y0, x0] + fx * (1 - fy) * grid[y0, x1]
            + (1 - fx) * fy * grid[y1, x0] + fx * fy * grid[y1, x1])


def tracking_performance(field, positions, times, grid_size=96):
    """Eq. 1: psi / psi_null, using the full observed temporal-average field.

    Null light is evaluated along the SAME fish trajectories, not at uniformly
    random tank positions. The grid is an explicit numerical approximation.
    """
    positions, times = np.asarray(positions), np.asarray(times)
    if positions.ndim != 3 or positions.shape[-1] != 2 or len(positions) != len(times) or not len(times):
        raise ValueError("Need matching, nonempty T × N × 2 positions and T times")
    if positions.shape[1] == 0 or grid_size < 2:
        raise ValueError("Need fish and at least two grid points per dimension")
    x, y = np.meshgrid(np.linspace(0, field.bounds[0], grid_size),
                       np.linspace(0, field.bounds[1], grid_size))
    points = np.stack((x, y), axis=-1)
    mean_field = np.zeros_like(x)
    raw = []
    for sample, time in zip(positions, times, strict=True):
        mean_field += field.values(points, time) / len(times)
        raw.append(np.mean(1 - field.values(sample, time)))
    psi = float(np.mean(raw))
    null = float(np.mean(1 - interpolate_grid(mean_field, positions, *field.bounds)))
    return {"psi_raw": psi, "psi_null": null, "Psi": psi / null if null > 1e-12 else None}


def run_paper_trial(*, fish, weight, seed, steps, sample_every, environmental_noise,
                    gradient_error=0.0, social=True, grid_size=96, output=None):
    if steps < 1 or sample_every < 1 or grid_size < 2:
        raise ValueError("steps/sample_every must be positive; grid_size must be >= 2")
    model = GoldenShinerModel(num_fish=fish, max_population=max(200, fish), seed=seed,
                              navigation_mode="Paper model", paper_species="Custom weight",
                              gradient_weight=weight, gradient_error=gradient_error,
                              environmental_noise=environmental_noise, light_enabled=True,
                              lifecycle_enabled=False, social_enabled=social, start_away=True)
    positions, times, nearest, cohesion = [], [], [], []
    for tick in range(1, steps + 1):
        model.step()
        if tick % sample_every == 0 or tick == steps:
            positions.append(model.positions.copy())
            times.append(model.steps * model.dt)
            nearest.append(model.metrics["Nearest neighbour (BL)"])
            cohesion.append(model.metrics["Cohesion"])
    metrics = tracking_performance(model.light_field, positions, times, grid_size)
    metrics.update({"fish": fish, "weight": weight, "seed": seed, "social": social,
                    "noise": environmental_noise, "gradient_error": gradient_error,
                    "mean_nearest_neighbor_bl": float(np.mean(nearest)),
                    "mean_cohesion": float(np.mean(cohesion)), "duration_s": steps * model.dt,
                    "samples": len(times), "null_grid_size": grid_size})
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        name = f"n{fish}_w{weight:g}_seed{seed}_social{int(social)}"
        np.savez_compressed(output / (name + "_trajectories.npz"), positions=positions, times=times)
        model.datacollector.get_model_vars_dataframe().to_csv(output / (name + ".csv"), index=False)
        (output / (name + ".json")).write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fish", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--weights", nargs="+", type=float, default=[0, 1, 31.6, 100])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--sample-every", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=96)
    parser.add_argument("--noise", type=float, default=0.25)
    parser.add_argument("--gradient-error", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--independent-control", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/paper"))
    args = parser.parse_args()
    if min(args.fish + [args.repeats, args.steps, args.sample_every]) < 1 or args.grid_size < 2:
        parser.error("Counts must be positive and grid-size at least 2")
    if (any(not np.isfinite(w) or w < 0 for w in args.weights) or not 0 <= args.noise <= 1
            or not np.isfinite(args.gradient_error) or args.gradient_error < 0):
        parser.error("Weights/error must be finite and nonnegative; noise must be between 0 and 1")
    if len(set(args.fish)) != len(args.fish) or len(set(args.weights)) != len(args.weights):
        parser.error("Group sizes and weights must be distinct")
    args.output.mkdir(parents=True, exist_ok=True)
    settings = vars(args).copy()
    settings["output"] = str(args.output)
    settings["model"] = "2018 equations; synthetic Gaussian patch + eight smooth noise modes; 5 cm/BL"
    import mesa
    settings["mesa_version"] = mesa.__version__
    settings["resolved_paper_constants"] = {"width_bl": 28, "height_bl": 28, "dt_s": 0.125,
        "repulsion_bl": 0.5, "orientation_bl": 3, "attraction_bl": 5.5,
        "vision_deg": 270, "turn_radians_per_update": 1.75, "social_error_rad": 0.01,
        "min_speed_bl_s": 0.2, "max_speed_bl_s": 5, "patch_speed_bl_s": 1.142,
        "patch_decay_bl": 7.7, "lifecycle_enabled": False, "start_away": True}
    (args.output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    rows = []
    for seed in range(args.seed, args.seed + args.repeats):
        for fish in args.fish:
            for weight in args.weights:
                for social in ([True, False] if args.independent_control else [True]):
                    row = run_paper_trial(fish=fish, weight=weight, seed=seed, steps=args.steps,
                        sample_every=args.sample_every, environmental_noise=args.noise,
                        gradient_error=args.gradient_error, social=social, grid_size=args.grid_size,
                        output=args.output)
                    rows.append(row)
                    print(f"N={fish}, w={weight:g}, seed={seed}, social={social}: Ψ={row['Psi']}", flush=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(args.output / "runs.csv", index=False)
    aggregate = runs.groupby(["fish", "weight", "social"], as_index=False).agg(
        runs=("Psi", "size"), mean_Psi=("Psi", "mean"), sem_Psi=("Psi", "sem"),
        mean_nearest_neighbor_bl=("mean_nearest_neighbor_bl", "mean"),
        mean_cohesion=("mean_cohesion", "mean"))
    aggregate.to_csv(args.output / "aggregate.csv", index=False)
    figure = Figure(figsize=(10, 4), layout="constrained")
    performance, distance = figure.subplots(1, 2)
    for (fish, social), group in aggregate.groupby(["fish", "social"]):
        label = f"N={fish}, {'school' if social else 'independent'}"
        performance.plot(group.weight, group.mean_Psi, "o-", label=label)
        distance.plot(group.weight, group.mean_nearest_neighbor_bl, "o-", label=label)
    performance.axhline(1, color="gray", linestyle=":")
    performance.set(ylabel="Normalized tracking Ψ")
    distance.set(ylabel="Mean nearest neighbour (BL)")
    for axis in (performance, distance):
        axis.set_xscale("symlog", linthresh=0.1)
        axis.set_xlabel("Environmental weight w")
        axis.legend(fontsize=7)
        axis.grid(alpha=0.2)
    figure.savefig(args.output / "comparison.png", dpi=160)
    print(f"Saved trajectories, settings, runs, aggregate and comparison.png in {args.output}")


if __name__ == "__main__":
    main()
