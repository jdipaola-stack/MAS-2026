import json

import numpy as np
import pandas as pd
import pytest

from model import Fish, GoldenShinerModel
from research_experiments import aggregate_runs, paired_recruitment_comparison, run_trial


def arrange(model, positions):
    for fish, position in zip(model.fish, positions, strict=True):
        model.space.move_agent(fish, tuple(position))
        fish.heading = 0.0
    model.refresh_snapshot()


def test_role_fractions_have_matching_starts_and_nested_informed_sets():
    models = [GoldenShinerModel(num_fish=20, navigation_mode="Informed minority",
                               informed_fraction=f, start_away=True, seed=9)
              for f in (0, .1, .25, 1)]
    assert [sum(f.informed for f in m.fish) for m in models] == [0, 2, 5, 20]
    for model in models[1:]:
        np.testing.assert_array_equal(model.positions, models[0].positions)
        np.testing.assert_array_equal(model.headings, models[0].headings)
        assert model.random.getstate() == models[0].random.getstate()
    assert {f.index for f in models[1].fish if f.informed} <= {f.index for f in models[2].fish if f.informed}


def test_uninformed_fish_cannot_read_light_even_to_change_speed(monkeypatch):
    model = GoldenShinerModel(num_fish=1, navigation_mode="Informed minority",
                             informed_fraction=0, light_enabled=True, noise=0)
    arrange(model, [[30, 30]])

    def forbidden(*args):
        pytest.fail("An uninformed fish tried to read the light field")

    monkeypatch.setattr(model, "light_at", forbidden)
    monkeypatch.setattr(model, "speed_from_light", forbidden)
    model.fish[0].plan()
    assert model.fish[0].next_heading == 0
    assert model.fish[0].next_speed == model.base_speed


def test_informed_fish_can_navigate_alone():
    model = GoldenShinerModel(num_fish=1, navigation_mode="Informed minority",
                             informed_fraction=1, light_enabled=True, noise=0)
    arrange(model, [[30, 30]])
    model.fish[0].plan()
    assert model.fish[0].next_heading < 0


def test_uninformed_fish_do_not_privilege_informed_neighbours():
    model = GoldenShinerModel(num_fish=2, navigation_mode="Informed minority",
                             informed_fraction=0, light_enabled=True, noise=0)
    arrange(model, [[30, 30], [32, 32]])
    model.fish[1].speed = 1.0
    model.refresh_snapshot()
    model.fish[0].plan()
    movement = model.fish[0].next_heading, model.fish[0].next_speed
    model.fish[1].informed = True
    model.fish[0].plan()
    assert (model.fish[0].next_heading, model.fish[0].next_speed) == movement
    assert model.fish[0].next_speed < model.base_speed


def test_recruitment_requires_a_visible_local_signal():
    model = GoldenShinerModel(num_fish=2, navigation_mode="Recruitment",
                             light_enabled=True, noise=0, attraction_weight=0, alignment_weight=0)
    arrange(model, [[30, 16], [32, 10]])
    assert model.signaling.tolist() == [False, True]
    model.fish[0].plan()
    assert model.fish[0].next_heading < 0
    model.recruitment_enabled = False
    model.refresh_snapshot()
    model.fish[0].plan()
    assert model.fish[0].next_heading == 0
    model.recruitment_enabled = True
    arrange(model, [[10, 16], [50, 10]])
    model.fish[0].plan()
    assert model.fish[0].next_heading == 0


def test_independent_control_has_no_neighbour_influence():
    model = GoldenShinerModel(num_fish=2, navigation_mode="Recruitment", social_enabled=False,
                             light_enabled=True, noise=0)
    arrange(model, [[30, 16], [32, 10]])
    model.fish[0].plan()
    movement = model.fish[0].next_heading, model.fish[0].next_speed
    arrange(model, [[30, 16], [30.1, 16]])
    model.fish[0].plan()
    assert (model.fish[0].next_heading, model.fish[0].next_speed) == movement


def test_last_first_arrival_is_distinct_from_simultaneous_safety():
    model = GoldenShinerModel(num_fish=2, light_enabled=True, safety_hold_time=.2)
    arrange(model, [[30, 10], [32, 30]])
    for step in (1, 2):
        model.steps = step
        model.record_arrivals()
    assert model.arrival_summary()["fraction_arrived"] == .5
    arrange(model, [[30, 30], [32, 10]])
    for step in (3, 4):
        model.steps = step
        model.record_arrivals()
    result = model.arrival_summary()
    assert result["time_all_s"] == .4
    assert result["time_all_together_s"] is None


def test_deaths_and_births_do_not_shrink_or_enlarge_the_starting_cohort():
    model = GoldenShinerModel(num_fish=2, light_enabled=True, safety_hold_time=.1)
    model.remove_fish(model.fish[0], "light")
    child = Fish(model, 1, [32, 10], 0, newborn=True)
    model.fish.append(child)
    arrange(model, [[30, 10], [32, 10]])
    model.steps = 1
    model.record_arrivals()
    assert model.arrival_summary()["fraction_arrived"] == .5
    assert not model.arrival_summary()["success"]
    assert child.unique_id not in model.arrival_times


def test_trial_preserves_fraction_in_filenames_and_censors_non_arrivals(tmp_path):
    parameters = dict(num_fish=2, navigation_mode="Informed minority", light_enabled=True,
                      lifecycle_enabled=False, start_away=True, seed=42)
    for label in ("informed_0.1", "informed_0.25"):
        result = run_trial(parameters, 1, tmp_path, label)
        assert not result["success"] and result["time_all_s"] is None
        assert result["restricted_completion_s"] == .1
        config = json.loads((tmp_path / f"{label}_seed_42.json").read_text())
        assert config["parameters"]["lifecycle_enabled"] is False
    assert len(list(tmp_path.glob("*.csv"))) == 6


def test_failures_remain_in_aggregate_denominators():
    rows = pd.DataFrame([
        dict(condition="a", success=True, time_all_s=10., restricted_completion_s=10.,
             fraction_arrived=1., uninformed_fraction_arrived=1., late_occupancy=1., mean_individual_restricted_s=8., actual_informed_fraction=.1),
        dict(condition="a", success=False, time_all_s=None, restricted_completion_s=60.,
             fraction_arrived=.5, uninformed_fraction_arrived=.5, late_occupancy=.5, mean_individual_restricted_s=35., actual_informed_fraction=.1),
    ])
    result = aggregate_runs(rows).iloc[0]
    assert result["success_rate"] == .5
    assert result["mean_restricted_completion_s"] == 35
    assert result["mean_last_arrival_successful_s"] == 10


def test_no_speedup_is_invented_for_censored_pairs():
    rows = pd.DataFrame([
        dict(seed=1, condition="recruitment", success=False, time_all_s=None, restricted_completion_s=60.),
        dict(seed=1, condition="social_no_signal", success=True, time_all_s=30., restricted_completion_s=30.),
        dict(seed=1, condition="independent", success=True, time_all_s=20., restricted_completion_s=20.),
    ])
    comparison = paired_recruitment_comparison(rows)
    assert not comparison["both_completed"].any()
    assert comparison["completion_speedup_baseline_over_recruitment"].isna().all()
    assert (comparison["restricted_seconds_saved"] < 0).all()
