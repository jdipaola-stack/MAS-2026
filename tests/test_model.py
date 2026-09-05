import math

import numpy as np
import pandas as pd
import pytest

from model import GoldenShinerModel, measure_school
from run import run_experiment


def arrange(model, positions, headings):
    for fish, position, heading in zip(model.fish, positions, headings, strict=True):
        model.space.move_agent(fish, tuple(position))
        fish.heading = heading
    model.refresh_snapshot()


def test_metrics_distinguish_parallel_and_circular_motion():
    angles = np.linspace(0, 2 * math.pi, 32, endpoint=False)
    positions = np.column_stack((np.cos(angles), np.sin(angles)))
    parallel = measure_school(positions, np.zeros(32), np.ones(32), 1)
    circular = measure_school(positions, angles + math.pi / 2, np.ones(32), 1)
    assert parallel["Polarization"] == pytest.approx(1)
    assert parallel["Rotation"] == pytest.approx(0, abs=1e-12)
    assert circular["Polarization"] == pytest.approx(0, abs=1e-12)
    assert circular["Rotation"] == pytest.approx(1)


def test_cohesion_measures_largest_connected_component():
    metrics = measure_school(np.array([[0., 0], [1, 0], [2, 0], [10, 0]]),
                             np.zeros(4), np.ones(4), 1.1)
    assert metrics["Cohesion"] == 0.75
    assert metrics["Nearest neighbour (BL)"] == pytest.approx(2.75)


def test_seed_repeats_complete_trajectory():
    a = GoldenShinerModel(num_fish=12, seed=7, light_enabled=True)
    b = GoldenShinerModel(num_fish=12, seed=7, light_enabled=True)
    c = GoldenShinerModel(num_fish=12, seed=8)
    assert not np.allclose(a.positions, c.positions)
    for _ in range(40):
        a.step()
        b.step()
    np.testing.assert_array_equal(a.positions, b.positions)
    pd.testing.assert_frame_equal(a.datacollector.get_model_vars_dataframe(),
                                  b.datacollector.get_model_vars_dataframe())


@pytest.mark.parametrize("light", [False, True])
def test_long_run_stays_finite_inside_tank_and_speed_limits(light):
    model = GoldenShinerModel(num_fish=20, width=16, height=12, light_enabled=light, seed=13)
    for _ in range(250):
        model.step()
        assert np.isfinite(model.positions).all()
        assert (model.positions >= 0).all()
        assert (model.positions < [model.width, model.height]).all()
        assert (model.speeds >= model.min_speed - 1e-12).all()
        assert (model.speeds <= model.max_speed + 1e-12).all()
        for key in ("Polarization", "Rotation", "Cohesion"):
            assert 0 <= model.metrics[key] <= 1
    data = model.datacollector.get_model_vars_dataframe()
    assert len(data) == 251
    assert data.iloc[-1]["Time (s)"] == pytest.approx(25)


def test_visual_field_excludes_distant_neighbour_behind():
    model = GoldenShinerModel(num_fish=2, vision_angle=90, noise=0)
    arrange(model, [[30, 20], [27, 20]], [0, 0])
    model.agents.do("plan")
    assert model.fish[0].next_state == "roaming"
    assert model.fish[0].school_neighbors == 0
    assert model.fish[1].school_neighbors == 1
    assert model.fish[1].next_state == "roaming"  # A pair cannot pool enough readings.


def test_attraction_turns_toward_neighbour():
    model = GoldenShinerModel(num_fish=2, vision_angle=360, noise=0, alignment_weight=0)
    arrange(model, [[30, 20], [30, 24]], [0, 0])
    model.agents.do("plan")
    assert model.fish[0].next_heading > 0
    assert model.fish[1].next_heading < 0


def test_close_fish_ahead_slows_focal_fish():
    model = GoldenShinerModel(num_fish=2, noise=0)
    arrange(model, [[30, 20], [30.5, 20]], [0, 0])
    model.fish[0].plan()
    assert model.fish[0].next_speed < model.fish[0].speed


def test_light_increases_with_y_and_is_independent_of_x():
    model = GoldenShinerModel(width=60, height=40)
    assert model.light_at([10, 0]) == 0
    assert model.light_at([10, 40]) == 1
    assert model.light_at([10, 10]) == model.light_at([50, 10]) == 0.25
    assert model.preferred_band == pytest.approx((6, 14))


def test_preferred_light_is_slow_point_not_the_darkest_water():
    model = GoldenShinerModel(preferred_light=0.25)
    assert model.speed_from_light(0.25) == model.min_speed
    assert model.speed_from_light(0) > model.speed_from_light(0.25)
    assert model.speed_from_light(0.5) > model.speed_from_light(0.25)
    assert model.speed_from_light(0) == model.speed_from_light(0.5)


@pytest.mark.parametrize("y,sign", [(4, 1), (30, -1)])
def test_fish_turn_toward_preferred_light_from_both_sides(y, sign):
    model = GoldenShinerModel(num_fish=3, light_enabled=True, noise=0,
                             attraction_weight=0, alignment_weight=0, vision_angle=360)
    arrange(model, [[30, y], [32, y + 1], [28, y + 1]], [0, 0, 0])
    model.fish[0].plan()
    assert sign * model.fish[0].next_heading > 0


def test_light_response_can_be_disabled():
    for parameters in ({"light_enabled": False}, {"light_enabled": True, "light_weight": 0}):
        model = GoldenShinerModel(num_fish=1, noise=0, **parameters)
        arrange(model, [[30, 30]], [0])
        model.fish[0].plan()
        assert model.fish[0].next_heading == 0


def test_light_metrics_measure_individual_errors_not_only_the_mean():
    model = GoldenShinerModel(num_fish=2, light_enabled=True, preferred_light=0.5)
    arrange(model, [[10, 10], [50, 30]], [0, 0])
    model.update_metrics()
    assert model.metrics["Mean brightness"] == 0.5
    assert model.metrics["Mean light error"] == 0.25
    assert model.metrics["Fraction in preferred band"] == 0
    assert model.metrics["Fraction in dark region"] == 0.5


@pytest.mark.parametrize("seed,target", [(7, 0.25), (42, 0.25), (81, 0.75)])
def test_school_finds_and_stays_in_preferred_band(seed, target):
    model = GoldenShinerModel(num_fish=30, light_enabled=True, preferred_light=target,
                             seed=seed, lifecycle_enabled=False)
    initial_error = model.metrics["Mean light error"]
    for _ in range(900):
        model.step()
    final_window = model.datacollector.get_model_vars_dataframe().tail(200)
    assert final_window["Mean light error"].mean() < initial_error * 0.25
    assert final_window["Fraction in preferred band"].mean() > 0.90
    assert final_window["Cohesion"].mean() > 0.80
    assert model.speeds.min() >= model.min_speed - 1e-12


def test_planning_does_not_move_or_change_current_headings():
    model = GoldenShinerModel(num_fish=8)
    positions, headings = model.positions.copy(), model.headings.copy()
    model.agents.do("plan")
    np.testing.assert_array_equal([f.pos for f in model.fish], positions)
    np.testing.assert_array_equal([f.heading for f in model.fish], headings)


def test_reflection_contains_fish_crossing_corner():
    model = GoldenShinerModel(num_fish=1)
    fish = model.fish[0]
    arrange(model, [[59.99, 39.99]], [math.pi / 4])
    fish.next_speed = 4
    fish.next_heading = math.pi / 4
    fish.next_state = "roaming"
    fish.advance()
    assert 0 < fish.pos[0] < 60
    assert 0 < fish.pos[1] < 40
    assert math.cos(fish.heading) < 0 and math.sin(fish.heading) < 0


def test_single_fish_and_exact_overlap_are_supported():
    single = GoldenShinerModel(num_fish=1)
    single.step()
    assert single.collective_state == "Solitary"
    assert math.isnan(single.metrics["Nearest neighbour (BL)"])
    pair = GoldenShinerModel(num_fish=2)
    arrange(pair, [[30, 20], [30, 20]], [0, 0])
    pair.step()
    assert np.isfinite(pair.positions).all()


@pytest.mark.parametrize("parameters", [
    {"num_fish": 0}, {"num_fish": 2.5}, {"dt": 0}, {"dt": 100},
    {"noise": -1}, {"vision_angle": 361}, {"base_speed": 10},
    {"interaction_radius": 0.5}, {"width": float("nan")},
    {"preferred_light": -0.1}, {"preferred_light": 1.1},
    {"light_tolerance": 0}, {"light_tolerance": 1.1}, {"light_weight": -1},
    {"survival_time": 0}, {"reproduction_time": 0}, {"maturity_time": 0},
    {"recovery_time": -1}, {"max_population": 10}, {"school_min_neighbors": 1},
])
def test_invalid_parameters_raise_clear_errors(parameters):
    with pytest.raises(ValueError):
        GoldenShinerModel(**parameters)


def test_cli_export_contains_measurements_and_settings(tmp_path):
    destination = tmp_path / "run.csv"
    run_experiment(steps=4, output=destination, num_fish=3, seed=6)
    frame = pd.read_csv(destination)
    assert len(frame) == 5
    assert frame["Time (s)"].iloc[-1] == pytest.approx(0.4)
    import json
    metadata = json.loads(destination.with_suffix(".json").read_text())
    assert metadata["parameters"]["seed"] == 6
    assert metadata["parameters"]["width"] == 60
