"""Exercise the real Mesa controls and both plotting branches without a browser."""

import ipyvuetify as widgets
import solara
from mesa.visualization import SolaraViz

from app import HistoryView, MODEL_PARAMS, SchoolView, tank_figure, PerturbationControls
from model import GoldenShinerModel


def test_step_parameter_change_reset_and_light_render():
    model = solara.reactive(GoldenShinerModel())
    page = solara.AppLayout(children=[
        SolaraViz(model, components=[SchoolView, HistoryView],
                  model_params=MODEL_PARAMS, render_interval=2)
    ])
    _, context = solara.render(page, handle_error=False)
    try:
        context.find(widgets.Btn, children=["Step"]).widget.fire_event("click", {})
        assert model.value.steps == 2
        context.find(widgets.Slider, label="Number of fish").widget.v_model = 12
        context.find(widgets.Slider, label="Preferred light (0 = dark, 1 = bright)").widget.v_model = 0.75
        context.find(widgets.Checkbox, label="Enable light gradient").widget.v_model = True
        context.find(widgets.Btn, children=["Reset"]).widget.fire_event("click", {})
        assert model.value.steps == 0
        assert model.value.num_fish == 12
        assert model.value.light_enabled
        assert model.value.preferred_light == 0.75
        context.find(widgets.Select, label="Navigation experiment").widget.v_model = "Recruitment"
        context.find(widgets.Btn, children=["Reset"]).widget.fire_event("click", {})
        assert model.value.navigation_mode == "Recruitment"
        context.find(widgets.Btn, children=["Step"]).widget.fire_event("click", {})
        assert model.value.steps == 2
        assert len(model.value.datacollector.get_model_vars_dataframe()) == 3
    finally:
        context.close()


def test_background_and_target_band_match_the_light_field():
    import numpy as np

    model = GoldenShinerModel(light_enabled=True)
    figure = tank_figure(model)
    axis = figure.axes[0]
    background = np.asarray(axis.images[0].get_array())
    np.testing.assert_array_equal(background[:, 0], background[:, 1])
    assert background[0, 0] == model.light_at([0, 0]) == 0
    assert background[-1, 0] == model.light_at([0, model.height]) == 1
    np.testing.assert_allclose(axis.lines[0].get_ydata(), [10, 10])


def test_perturbation_buttons_update_the_model_without_advancing_time():
    model = GoldenShinerModel(light_enabled=True)
    _, context = solara.render(PerturbationControls(model), handle_error=False)
    try:
        context.find(widgets.Btn, children=["Predator attack"]).widget.fire_event("click", {})
        assert model.predator_position is not None and model.steps == 0
        context.find(widgets.Btn, children=["Storm"]).widget.fire_event("click", {})
        assert model.light_offset == -0.35 and model.steps == 0
        context.find(widgets.Btn, children=["Clear disturbances"]).widget.fire_event("click", {})
        assert model.predator_position is None and model.light_offset == 0
    finally:
        context.close()


def test_extinct_population_renders_without_errors():
    import io

    model = GoldenShinerModel(num_fish=1, light_enabled=True, survival_time=0.01)
    model.space.move_agent(model.fish[0], (30, 30))
    model.step()
    assert not model.running
    figure = tank_figure(model)
    image = io.BytesIO()
    figure.savefig(image, format="svg")
    assert "Population extinct" in image.getvalue().decode()
