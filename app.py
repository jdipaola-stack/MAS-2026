"""Run with: .venv/bin/solara run app.py"""

import json

import numpy as np
import solara
from matplotlib.figure import Figure
from mesa.visualization import Slider, SolaraViz
from mesa.visualization.utils import force_update, update_counter

from model import GoldenShinerModel


def tank_figure(model):
    """Arrows indicate heading; their colour shows swimming speed."""
    figure = Figure(figsize=(8, 5.5), layout="constrained")
    ax = figure.subplots()
    ax.set_facecolor("#eaf4f6")
    if model.light_enabled:
        brightness = np.tile(np.clip(np.linspace(0, 1, 256) + model.light_offset, 0, 1)[:, None], (1, 2))
        ax.imshow(brightness, extent=(0, model.width, 0, model.height),
                  origin="lower", cmap="gray", vmin=0, vmax=1, alpha=0.75, aspect="auto")
        lower, upper = model.preferred_band
        ax.axhspan(lower, upper, color="#34d399", alpha=0.22)
        target_y = np.clip(model.preferred_light - model.light_offset, 0, 1) * model.height
        ax.axhline(target_y, color="#34d399", linestyle="--", linewidth=1.3)
        ax.text(0.02, target_y,
                f"Preferred light: {model.preferred_light:.0%}",
                transform=ax.get_yaxis_transform(), va="bottom", color="#065f46",
                bbox={"facecolor": "#ecfdf5", "alpha": 0.9, "edgecolor": "none", "pad": 2}, fontsize=8)
        ax.text(model.width * 0.02, model.height * 0.96, "BRIGHT", va="top", color="#334155")
        ax.text(model.width * 0.02, model.height * 0.04, "DARK", va="bottom", color="white")
        light_axis = ax.secondary_yaxis("right", functions=(
            lambda y: 100 * (y / model.height + model.light_offset),
            lambda light: (light / 100 - model.light_offset) * model.height))
        light_axis.set_ylabel("Light intensity (%)")
        light_axis.set_yticks([0, 25, 50, 75, 100])
    arrows = ax.quiver(model.positions[:, 0], model.positions[:, 1],
                       model.directions[:, 0], model.directions[:, 1], model.speeds,
                       cmap="viridis", clim=(model.min_speed, model.max_speed),
                       angles="xy", scale_units="xy", scale=0.6, pivot="mid", width=0.005,
                       edgecolors="white" if model.light_enabled else "none", linewidths=0.25)
    figure.colorbar(arrows, ax=ax, label="Swimming speed (BL/s)", shrink=0.7,
                   pad=0.16 if model.light_enabled else 0.04)
    unhealthy = np.array([f.health < 0.35 for f in model.fish], dtype=bool)
    if unhealthy.any():
        ax.scatter(model.positions[unhealthy, 0], model.positions[unhealthy, 1],
                   facecolors="none", edgecolors="#ef4444", s=40, linewidths=1.0)
    if model.predator_position is not None:
        ax.scatter(*model.predator_position, marker="X", c="#dc2626", s=130,
                   edgecolors="white", linewidths=1.0, zorder=5)
        ax.annotate("PREDATOR", model.predator_position, xytext=(5, 8),
                    textcoords="offset points", color="#dc2626", fontsize=8, weight="bold")
    if model.storm_until > model.steps * model.dt:
        ax.text(0.98, 0.97, "STORM · current →", transform=ax.transAxes,
                ha="right", va="top", color="#0284c7", weight="bold", fontsize=9)
    if not model.fish:
        ax.text(0.5, 0.5, "Population extinct\nPress Reset to start again", transform=ax.transAxes,
                ha="center", va="center", color="#dc2626", fontsize=12)
    ax.set(xlim=(0, model.width), ylim=(0, model.height), aspect="equal",
           xlabel="Position (body lengths)", ylabel="Position (body lengths)",
           title=f"{model.num_fish} fish · {model.steps * model.dt:.1f} s · {model.collective_state}")
    return figure


@solara.component
def PerturbationControls(model):
    update_counter.get()

    def predator():
        model.trigger_predator()
        force_update()

    def storm():
        model.trigger_storm()
        force_update()

    def clear():
        model.clear_disturbances()
        force_update()

    now = model.steps * model.dt
    with solara.Row():
        solara.Button("Predator attack", on_click=predator, color="error",
                      disabled=not model.fish or model.predator_position is not None)
        solara.Button("Storm", on_click=storm, color="primary",
                      disabled=not model.fish or model.storm_until > now)
        solara.Button("Clear disturbances", on_click=clear,
                      disabled=model.predator_position is None and model.storm_until <= now)
    active = []
    if model.predator_position is not None:
        active.append(f"Predator: {max(0, model.predator_until - now):.1f} s remaining")
    if model.storm_until > now:
        active.append(f"Storm: {model.storm_until - now:.1f} s remaining")
    solara.Markdown(" · ".join(active) if active else
                    "**Disturb the school:** a predator hunts fish; a storm shifts the light band and adds a current.")


@solara.component
def SchoolView(model):
    update_counter.get()
    PerturbationControls(model)
    solara.FigureMatplotlib(tank_figure(model))
    m = model.metrics
    solara.Markdown(
        f"**Population: {model.num_fish} / {model.max_population}** · **Births: {model.births}** · "
        f"**Light deaths: {model.light_deaths}** · **Predator deaths: {model.predator_deaths}**\n\n"
        f"Mean health: **{m['Mean health']:.0%}** · "
        f"Fish with enough schoolmates: **{m['Schooling fraction']:.0%}**. Red rings mark low health."
    )
    solara.Markdown(
        f"**Polarization:** {m['Polarization']:.2f} · **Rotation:** {m['Rotation']:.2f} · "
        f"**Cohesion:** {m['Cohesion']:.0%}\n\n"
        f"Nearest neighbour: **{m['Nearest neighbour (BL)']:.2f} BL** · "
        f"Mean speed: **{m['Mean speed (BL/s)']:.2f} BL/s**"
    )
    if model.light_enabled:
        solara.Markdown(
            f"**Sweet spot: {model.preferred_light:.0%} light** · "
            f"**Fish in preferred band: {m['Fraction in preferred band']:.0%}**\n\n"
            f"Mean light experienced: **{m['Mean brightness']:.0%}** · "
            f"Mean distance from preferred light: **{100 * m['Mean light error']:.1f} percentage points**"
        )


@solara.component
def HistoryView(model):
    update_counter.get()
    frame = model.datacollector.get_model_vars_dataframe()
    figure = Figure(figsize=(8, 7), layout="constrained")
    top, bottom, population = figure.subplots(3, 1, sharex=True)
    time = frame["Time (s)"]
    for name, color in (("Polarization", "#0f766e"), ("Rotation", "#d97706"), ("Cohesion", "#7c3aed")):
        top.plot(time, frame[name], label=name, color=color)
    top.set(ylim=(-0.03, 1.05), ylabel="Group measures", title="How the school changes")
    top.legend(loc="upper right", fontsize=8)
    if model.light_enabled:
        bottom.plot(time, frame["Mean brightness"], label="Mean brightness", color="#64748b")
        bottom.plot(time, frame["Fraction in preferred band"], label="Fish in preferred band", color="#0f766e")
        bottom.axhline(model.preferred_light, color="#d97706", linestyle="--", label="Preferred light")
        bottom.axhspan(max(0, model.preferred_light - model.light_tolerance),
                       min(1, model.preferred_light + model.light_tolerance), color="#34d399", alpha=0.12)
        bottom.set(ylim=(-0.03, 1.05), ylabel="Light response")
    else:
        bottom.plot(time, frame["Nearest neighbour (BL)"], label="Nearest neighbour", color="#2563eb")
        bottom.plot(time, frame["Group radius (BL)"], label="Group radius", color="#d97706")
        bottom.set(ylabel="Distance (BL)")
    bottom.legend(loc="upper right", fontsize=8)
    population.plot(time, frame["Population"], label="Alive", color="#2563eb")
    population.plot(time, frame["Births"], label="Births (total)", color="#16a34a")
    population.plot(time, frame["Light deaths"] + frame["Predator deaths"], label="Deaths (total)", color="#dc2626")
    population.set(ylabel="Fish", xlabel="Time (seconds)")
    population.legend(loc="upper left", fontsize=8)
    for ax in (top, bottom, population):
        ax.grid(alpha=0.2)
    solara.FigureMatplotlib(figure)
    solara.FileDownload(frame.to_csv(index=False), filename="golden_shiners.csv", label="Download measurements (CSV)")
    solara.FileDownload(json.dumps(model.events, indent=2), filename="disturbances.json", label="Download event log")
    solara.Markdown(
        "**Reading the plots:** polarization near 1 means a shared heading; rotation near 1 "
        "means circling in a common direction. Cohesion is the fraction in the largest "
        "connected group. State labels are approximate.\n\n"
        "Change parameters and press **Reset** to start a new experiment. "
        "The same seed repeats the same starting conditions. "
        "Light increases from bottom to top. Fish must share readings with at least "
        f"**{model.school_min_neighbors} nearby schoolmates** to estimate which way to turn. "
        "Isolated fish can only roam and change speed.\n\n"
        "Outside the green band, health falls; inside it, health recovers. "
        f"Two healthy adults can produce a juvenile after **{model.reproduction_time:g} uninterrupted seconds** "
        "in the preferred light. Juveniles must mature before breeding. "
        "Event timers advance only while the simulation runs."
    )


MODEL_PARAMS = {
    "num_fish": Slider("Number of fish", 60, 1, 200, 1),
    "alignment_weight": Slider("Alignment strength", 1.0, 0.0, 3.0, 0.1),
    "attraction_weight": Slider("Attraction strength", 0.6, 0.0, 2.0, 0.1),
    "repulsion_weight": Slider("Repulsion strength", 3.0, 0.0, 5.0, 0.1),
    "interaction_radius": Slider("Detection distance (BL)", 8.0, 2.0, 15.0, 0.5),
    "vision_angle": Slider("Viewing angle (degrees)", 300.0, 60.0, 360.0, 10.0),
    "base_speed": Slider("Baseline speed (BL/s)", 2.0, 0.5, 4.0, 0.1),
    "speed_regulation": Slider("Social speed response", 1.0, 0.0, 3.0, 0.1),
    "noise": Slider("Random turning", 0.15, 0.0, 1.5, 0.05),
    "light_enabled": {"type": "Checkbox", "label": "Enable light gradient", "value": True},
    "preferred_light": Slider("Preferred light (0 = dark, 1 = bright)", 0.25, 0.05, 0.95, 0.05),
    "light_tolerance": Slider("Acceptable light deviation", 0.10, 0.02, 0.30, 0.02),
    "light_weight": Slider("Light-seeking strength", 3.0, 0.0, 6.0, 0.25),
    "school_min_neighbors": Slider("Schoolmates needed to navigate", 2, 2, 8, 1),
    "lifecycle_enabled": {"type": "Checkbox", "label": "Enable light survival and reproduction", "value": True},
    "survival_time": Slider("Survival outside band (seconds, maximum)", 60.0, 10.0, 180.0, 5.0),
    "reproduction_time": Slider("Safe time before reproduction (seconds)", 20.0, 5.0, 60.0, 5.0),
    "max_population": Slider("Population limit", 200, 200, 400, 25),
    "seed": Slider("Random seed", 42, 0, 100, 1),
}


@solara.component
def Page():
    # Each browser session owns its model; rerenders preserve the initial instance.
    initial_model = solara.use_memo(lambda: GoldenShinerModel(light_enabled=True), dependencies=[])
    SolaraViz(initial_model, components=[SchoolView, HistoryView], model_params=MODEL_PARAMS,
              name="Golden shiners · Collective movement", play_interval=100, render_interval=2)
