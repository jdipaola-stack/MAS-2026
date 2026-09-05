import numpy as np
import pytest

from model import GoldenShinerModel


def arrange(model, positions):
    for fish, position in zip(model.fish, positions, strict=True):
        model.space.move_agent(fish, tuple(position))
        fish.heading = 0.0
    model.refresh_snapshot()


@pytest.mark.parametrize("count", [1, 2])
def test_isolated_fish_and_pairs_have_no_directional_light_information(count):
    model = GoldenShinerModel(num_fish=count, light_enabled=True, noise=0,
                             attraction_weight=0, alignment_weight=0)
    arrange(model, [[30, 30], [32, 31]][:count])
    model.agents.do("plan")
    assert all(f.next_heading == 0 for f in model.fish)


def test_distant_school_does_not_give_isolated_fish_navigation():
    model = GoldenShinerModel(num_fish=4, light_enabled=True, noise=0,
                             attraction_weight=0, alignment_weight=0, vision_angle=360)
    arrange(model, [[10, 30], [40, 30], [42, 31], [38, 31]])
    model.agents.do("plan")
    assert model.fish[0].next_heading == 0
    assert model.fish[1].next_heading < 0


def test_school_needs_spatial_light_contrast():
    model = GoldenShinerModel(num_fish=3, light_enabled=True, noise=0,
                             attraction_weight=0, alignment_weight=0)
    arrange(model, [[30, 30], [32, 30], [28, 30]])
    model.fish[0].plan()
    assert model.fish[0].next_heading == 0


def test_prolonged_bad_light_causes_extinction_and_cleans_mesa_space():
    model = GoldenShinerModel(num_fish=1, light_enabled=True, survival_time=0.2, noise=0)
    arrange(model, [[30, 30]])
    for _ in range(3):
        if model.running:
            model.step()
    assert model.collective_state == "Extinct"
    assert not model.running
    assert model.num_fish == len(model.fish) == len(model.agents) == 0
    assert model.positions.shape == (0, 2)
    assert len(model.space.agents) == 0
    assert model.light_deaths == 1
    assert model.metrics["Population"] == 0


def test_safe_light_restores_health_and_unsafe_light_resets_breeding_timer():
    model = GoldenShinerModel(num_fish=1, light_enabled=True)
    arrange(model, [[30, 10]])
    fish = model.fish[0]
    fish.health = 0.5
    model.step()
    assert fish.health > 0.5 and fish.safe_time > 0
    arrange(model, [[30, 30]])
    model.step()
    assert fish.safe_time == 0


def test_adult_pair_reproduces_after_residence_and_respects_population_cap():
    model = GoldenShinerModel(num_fish=2, light_enabled=True, reproduction_time=0.2,
                             maturity_time=1, max_population=3, noise=0)
    arrange(model, [[30, 10], [32, 10]])
    model.step()
    assert model.births == 0
    model.step()
    assert model.births == 1 and model.num_fish == 3
    child = model.fish[-1]
    assert child.age == 0 and child.safe_time == 0
    assert all(f.safe_time == 0 for f in model.fish[:2])
    for _ in range(5):
        model.step()
    assert model.num_fish == 3 and model.births == 1
    assert child.age < model.maturity_time


def test_juveniles_and_single_adults_cannot_reproduce():
    model = GoldenShinerModel(num_fish=2, light_enabled=True, reproduction_time=0.1,
                             maturity_time=30, noise=0)
    arrange(model, [[30, 10], [32, 10]])
    model.fish[1].age = 0
    for _ in range(5):
        model.step()
    assert model.births == 0


def test_lifecycle_toggle_disables_light_deaths_and_births():
    model = GoldenShinerModel(num_fish=2, light_enabled=True, lifecycle_enabled=False,
                             survival_time=0.1, reproduction_time=0.1)
    arrange(model, [[30, 30], [32, 30]])
    for _ in range(5):
        model.step()
    assert model.num_fish == 2 and model.light_deaths == model.births == 0


def test_predator_causes_fleeing_and_can_remove_fish():
    model = GoldenShinerModel(num_fish=2, noise=0)
    arrange(model, [[30, 20], [32, 20]])
    assert model.trigger_predator()
    model.predator_position = np.array([29.5, 20.0])
    model.fish[0].plan()
    assert model.fish[0].next_state == "escaping"
    assert model.fish[0].next_speed > model.fish[0].speed
    model.update_population()
    model.refresh_snapshot()
    assert model.predator_deaths == 1 and model.num_fish == 1
    assert len(model.agents) == 1


def test_storm_shifts_light_band_adds_current_and_expires():
    model = GoldenShinerModel(num_fish=1, light_enabled=True, noise=0)
    arrange(model, [[30, 10]])
    assert model.trigger_storm(duration=0.2)
    assert model.light_at([30, 10]) == 0
    assert model.preferred_band == pytest.approx((20, 28))
    assert np.linalg.norm(model.current) > 0
    model.step()
    assert model.metrics["Storm active"] == 1
    model.step()
    assert model.metrics["Storm active"] == 0
    assert model.light_offset == 0 and np.all(model.current == 0)
    assert model.preferred_band == pytest.approx((6, 14))


def test_disturbances_do_not_stack_and_clear_restores_environment():
    model = GoldenShinerModel(num_fish=3, light_enabled=True)
    assert model.trigger_predator() and model.trigger_storm()
    assert not model.trigger_predator() and not model.trigger_storm()
    assert len(model.events) == 2
    model.clear_disturbances()
    assert model.predator_position is None and model.light_offset == 0
    assert model.metrics["Predator active"] == model.metrics["Storm active"] == 0
    assert [event["event"] for event in model.events] == ["predator", "storm", "clear"]


def test_population_and_agent_indices_remain_consistent_during_events():
    model = GoldenShinerModel(num_fish=10, light_enabled=True, reproduction_time=1,
                             survival_time=5, maturity_time=2, max_population=20)
    for tick in range(120):
        if not model.running:
            break
        if tick == 10:
            model.trigger_predator(duration=3)
        if tick == 30:
            model.trigger_storm(duration=4)
        model.step()
        assert model.num_fish == len(model.fish) == len(model.agents)
        assert model.num_fish == model.initial_population + model.births - model.light_deaths - model.predator_deaths
        assert [f.index for f in model.fish] == list(range(model.num_fish))
        assert model.positions.shape == (model.num_fish, 2)


def test_predator_expires_without_remaining_flight_signal():
    model = GoldenShinerModel(num_fish=3)
    model.trigger_predator(duration=0.1)
    model.step()
    assert model.predator_position is None
    np.testing.assert_array_equal(model.predator_response([30, 20]), [0, 0])
