"""An educational golden-shiner model. Distances use body lengths (BL).

One tick represents dt seconds. All agents plan from the same snapshot before
any agent moves. The rules are hypotheses, not fitted biological parameters.
"""

from __future__ import annotations

import math

import mesa
import numpy as np
from mesa.datacollection import DataCollector
from mesa.space import ContinuousSpace


def unit(vector):
    """Return a unit vector, or a zero vector when no direction is defined."""
    length = np.linalg.norm(vector)
    return vector / length if length > 1e-12 else np.zeros(2)


def measure_school(positions, headings, speeds, interaction_radius):
    """Calculate group statistics from arrays without changing any fish."""
    n = len(positions)
    if n == 0:
        return {"Polarization": 0.0, "Rotation": 0.0, "Cohesion": 0.0,
                "Nearest neighbour (BL)": float("nan"), "Mean speed (BL/s)": 0.0,
                "Group radius (BL)": 0.0, "Density (fish/BL²)": 0.0}
    directions = np.column_stack((np.cos(headings), np.sin(headings)))
    polarization = float(np.linalg.norm(directions.mean(axis=0)))
    offsets = positions - positions.mean(axis=0)
    radii = np.linalg.norm(offsets, axis=1)
    radial = np.divide(offsets, radii[:, None], out=np.zeros_like(offsets),
                       where=radii[:, None] > 1e-12)
    rotation = float(abs(np.mean(radial[:, 0] * directions[:, 1]
                                - radial[:, 1] * directions[:, 0])))
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = float(distances.min(axis=1).mean()) if n > 1 else float("nan")

    # Cohesion = fraction belonging to the largest distance-connected group.
    adjacency = distances <= interaction_radius
    unseen = set(range(n))
    largest = 0
    while unseen:
        frontier = [unseen.pop()]
        size = 0
        while frontier:
            current = frontier.pop()
            size += 1
            linked = set(np.flatnonzero(adjacency[current])) & unseen
            unseen.difference_update(linked)
            frontier.extend(linked)
        largest = max(largest, size)

    # A descriptive estimate, not the density inside a biological school outline.
    box_area = float(np.prod(np.maximum(np.ptp(positions, axis=0), 1.0)))
    return {
        "Polarization": float(np.clip(polarization, 0, 1)),
        "Rotation": float(np.clip(rotation, 0, 1)),
        "Cohesion": largest / n,
        "Nearest neighbour (BL)": nearest,
        "Mean speed (BL/s)": float(np.mean(speeds)),
        "Group radius (BL)": float(np.sqrt(np.mean(radii**2))),
        "Density (fish/BL²)": n / box_area,
    }


class Fish(mesa.Agent):
    """A fish that senses locally, plans its movement, then advances."""

    def __init__(self, model, index, position, heading, newborn=False):
        super().__init__(model)
        self.index = index
        self.heading = heading
        self.speed = model.base_speed
        self.state = "roaming"
        self.health = 1.0
        self.age = 0.0 if newborn else model.maturity_time
        self.safe_time = 0.0
        self.school_neighbors = 0
        model.space.place_agent(self, tuple(position))

    def plan(self):
        m = self.model
        position = m.positions[self.index]
        forward = m.directions[self.index]
        offsets = m.positions - position
        distances = np.linalg.norm(offsets, axis=1)
        others = np.arange(m.num_fish) != self.index
        bearings = np.arctan2(offsets[:, 1], offsets[:, 0])
        angles = (bearings - self.heading + math.pi) % (2 * math.pi) - math.pi
        visible = others & (distances <= m.interaction_radius)
        visible &= np.abs(angles) <= math.radians(m.vision_angle) / 2
        # Very close neighbours trigger avoidance even in the rear blind region.
        close = others & (distances < m.repulsion_radius)
        self.school_neighbors = int(np.count_nonzero(visible))
        self.next_state = "schooling" if self.school_neighbors >= m.school_min_neighbors else "roaming"
        steering = np.zeros(2)
        if np.any(close) and m.repulsion_weight > 0:
            away = -offsets[close] / np.maximum(distances[close, None], 1e-9)
            steering = m.repulsion_weight * unit(away.sum(axis=0))
            # Symmetric crowding and exact overlaps still need an escape direction.
            if np.linalg.norm(steering) < 1e-12:
                angle = self.random.uniform(-math.pi, math.pi)
                steering = m.repulsion_weight * np.array([math.cos(angle), math.sin(angle)])
        elif np.any(visible):
            attraction = unit(offsets[visible].mean(axis=0))
            alignment = unit(m.directions[visible].mean(axis=0))
            steering = m.attraction_weight * attraction + m.alignment_weight * alignment

        # Anticipate the walls; reflection below is a final containment safeguard.
        margin = m.wall_margin
        wall = np.array([
            max(0, 1 - position[0] / margin) - max(0, 1 - (m.width - position[0]) / margin),
            max(0, 1 - position[1] / margin) - max(0, 1 - (m.height - position[1]) / margin),
        ])
        light_response = m.school_light_steering(self.index, visible)
        danger = m.predator_response(position)
        escaping = np.linalg.norm(danger) > 0
        if escaping:
            self.next_state = "escaping"
        desired = forward + steering + 5.0 * wall + (danger if escaping else light_response)
        target_heading = math.atan2(desired[1], desired[0]) if np.linalg.norm(desired) > 1e-12 else self.heading
        # Angular noise scales with sqrt(dt); turning is capped in radians/second.
        error = self.random.gauss(0, m.noise * math.sqrt(m.dt))
        turn = (target_heading - self.heading + error + math.pi) % (2 * math.pi) - math.pi
        self.next_heading = self.heading + float(np.clip(turn, -m.max_turn * m.dt, m.max_turn * m.dt))

        target_speed = m.speed_from_light(m.light_at(position)) if m.light_enabled else m.base_speed
        if np.any(visible) and m.speed_regulation > 0:
            # A neighbour ahead encourages catching up; a close fish ahead slows us.
            longitudinal = offsets @ forward
            social = np.clip(longitudinal[visible] / m.interaction_radius, -1, 1)
            too_close = distances[visible] < m.repulsion_radius
            social[too_close] = -np.sign(longitudinal[visible][too_close])
            target_speed += m.speed_regulation * float(social.mean())
        target_speed = float(np.clip(target_speed, m.min_speed, m.max_speed))
        if escaping:
            target_speed = m.max_speed
        self.next_speed = self.speed + float(np.clip(target_speed - self.speed,
                                                    -m.max_acceleration * m.dt,
                                                    m.max_acceleration * m.dt))

    def advance(self):
        self.heading = self.next_heading
        self.speed = self.next_speed
        self.state = self.next_state
        direction = np.array([math.cos(self.heading), math.sin(self.heading)])
        position = np.asarray(self.pos) + (self.speed * direction + self.model.current) * self.model.dt
        # Reflect positions robustly, including current-driven wall crossings.
        for axis, limit in enumerate((self.model.width, self.model.height)):
            folded = position[axis] % (2 * limit)
            if folded >= limit:
                position[axis] = 2 * limit - folded
                direction[axis] *= -1
            else:
                position[axis] = folded
            position[axis] = np.clip(position[axis], 0, np.nextafter(limit, 0))
        self.heading = math.atan2(direction[1], direction[0])
        self.model.space.move_agent(self, tuple(position))


class GoldenShinerModel(mesa.Model):
    """A tank with optional light increasing upward and a preferred light level."""

    def __init__(self, num_fish=60, width=60.0, height=40.0,
                 interaction_radius=8.0, repulsion_radius=1.0,
                 repulsion_weight=3.0, attraction_weight=0.6, alignment_weight=1.0,
                 vision_angle=300.0, base_speed=2.0, min_speed=0.5, max_speed=4.0,
                 speed_regulation=1.0, max_acceleration=2.0, max_turn=2.5,
                 noise=0.15, dt=0.1, light_enabled=False, preferred_light=0.25,
                 light_tolerance=0.10, light_weight=3.0, school_min_neighbors=2,
                 lifecycle_enabled=True, survival_time=60.0, recovery_time=15.0,
                 reproduction_time=20.0, maturity_time=30.0, max_population=200, seed=42):
        parameters = dict(width=width, height=height, interaction_radius=interaction_radius,
                          repulsion_radius=repulsion_radius, repulsion_weight=repulsion_weight,
                          attraction_weight=attraction_weight, alignment_weight=alignment_weight,
                          vision_angle=vision_angle, base_speed=base_speed, min_speed=min_speed,
                          max_speed=max_speed, speed_regulation=speed_regulation,
                          max_acceleration=max_acceleration, max_turn=max_turn, noise=noise, dt=dt,
                          preferred_light=preferred_light, light_tolerance=light_tolerance,
                          light_weight=light_weight, survival_time=survival_time,
                          recovery_time=recovery_time, reproduction_time=reproduction_time,
                          maturity_time=maturity_time)
        if isinstance(num_fish, bool) or not isinstance(num_fish, (int, np.integer)) or num_fish < 1:
            raise ValueError("num_fish must be a positive integer")
        for name, value in (("school_min_neighbors", school_min_neighbors), ("max_population", max_population)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if school_min_neighbors < 2:
            raise ValueError("school_min_neighbors must be at least 2 (three fish form a school)")
        if max_population < num_fish:
            raise ValueError("max_population cannot be smaller than the starting population")
        if any(not math.isfinite(v) for v in parameters.values()):
            raise ValueError("All numeric parameters must be finite")
        for name in ("width", "height", "interaction_radius", "repulsion_radius", "min_speed",
                     "max_speed", "max_acceleration", "max_turn", "dt", "survival_time",
                     "recovery_time", "reproduction_time", "maturity_time"):
            if parameters[name] <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("repulsion_weight", "attraction_weight", "alignment_weight", "speed_regulation", "noise", "light_weight"):
            if parameters[name] < 0:
                raise ValueError(f"{name} cannot be negative")
        if not min_speed <= base_speed <= max_speed:
            raise ValueError("Require min_speed <= base_speed <= max_speed")
        if not 0 < vision_angle <= 360:
            raise ValueError("vision_angle must be in (0, 360]")
        if not 0 <= preferred_light <= 1:
            raise ValueError("preferred_light must be in [0, 1]")
        if not 0 < light_tolerance <= 1:
            raise ValueError("light_tolerance must be in (0, 1]")
        if repulsion_radius >= interaction_radius:
            raise ValueError("repulsion_radius must be smaller than interaction_radius")
        if max_speed * dt >= min(width, height):
            raise ValueError("dt is too large: a fish could cross the tank in one tick")
        super().__init__(rng=seed)
        for name, value in parameters.items():
            setattr(self, name, float(value))
        self.num_fish = num_fish
        self.initial_population = num_fish
        self.school_min_neighbors = school_min_neighbors
        self.lifecycle_enabled = bool(lifecycle_enabled)
        self.max_population = max_population
        self.births = 0
        self.light_deaths = 0
        self.predator_deaths = 0
        self.events = []
        self.predator_position = None
        self.predator_until = 0.0
        self.next_predator_catch = 0.0
        self.storm_until = 0.0
        self.light_offset = 0.0
        self.current = np.zeros(2)
        self.light_enabled = bool(light_enabled)
        self.wall_margin = min(4.0, width / 4, height / 4)
        self.space = ContinuousSpace(width, height, torus=False)
        self.fish = []
        for index in range(num_fish):
            angle = self.random.uniform(-math.pi, math.pi)
            radius = math.sqrt(self.random.random()) * min(width, height) * 0.3
            position = np.array([width / 2 + radius * math.cos(angle),
                                 height / 2 + radius * math.sin(angle)])
            self.fish.append(Fish(self, index, position, self.random.uniform(-math.pi, math.pi)))
        self.refresh_snapshot()
        self.update_metrics()
        reporters = {name: (lambda model, key=name: model.metrics[key]) for name in self.metrics}
        reporters.update({"Time (s)": lambda m: m.steps * m.dt, "State": "collective_state"})
        self.datacollector = DataCollector(model_reporters=reporters)
        self.datacollector.collect(self)

    def light_at(self, position):
        """Local brightness depends only on y: dark bottom, bright top."""
        return float(np.clip(position[1] / self.height + self.light_offset, 0, 1))

    @property
    def preferred_band(self):
        """Vertical bounds of the acceptable light range, in body lengths."""
        # Include saturated dark/bright regions when they are acceptable.
        low, high = self.preferred_light - self.light_tolerance, self.preferred_light + self.light_tolerance
        bottom = 0.0 if low <= 0 else np.clip(low - self.light_offset, 0, 1) * self.height
        top = self.height if high >= 1 else np.clip(high - self.light_offset, 0, 1) * self.height
        return float(bottom), float(top)

    def school_light_steering(self, index, neighbors):
        """Infer light's direction ONLY from samples shared by nearby fish.

        No analytic environmental gradient is supplied to agents. A solitary
        fish, a pair, or neighbours without spatial light contrast cannot navigate.
        Relative positions and brightness differences give a least-squares estimate.
        """
        if not self.light_enabled or self.light_weight == 0 or np.count_nonzero(neighbors) < self.school_min_neighbors:
            return np.zeros(2)
        offsets = self.positions[neighbors] - self.positions[index]
        contrasts = self.brightness[neighbors] - self.brightness[index]
        gradient = np.linalg.lstsq(offsets, contrasts, rcond=None)[0]
        if np.linalg.norm(gradient) < 1e-9:
            return np.zeros(2)
        predicted_light = self.brightness[index] + float(gradient @ self.directions[index]) * self.speeds[index]
        response = np.clip((self.preferred_light - predicted_light) / self.light_tolerance, -1, 1)
        return self.light_weight * response * unit(gradient)

    def speed_from_light(self, intensity):
        """Swim slowly near the optimum; faster when either too bright or too dark."""
        mismatch = abs(float(np.clip(intensity, 0, 1)) - self.preferred_light)
        maximum_mismatch = max(self.preferred_light, 1 - self.preferred_light)
        return self.min_speed + mismatch / maximum_mismatch * (self.max_speed - self.min_speed)

    def refresh_snapshot(self):
        self.num_fish = len(self.fish)
        for index, fish in enumerate(self.fish):
            fish.index = index
        self.positions = np.array([fish.pos for fish in self.fish], dtype=float).reshape(-1, 2)
        self.headings = np.array([fish.heading for fish in self.fish])
        self.speeds = np.array([fish.speed for fish in self.fish])
        self.directions = np.column_stack((np.cos(self.headings), np.sin(self.headings)))
        self.brightness = np.clip(self.positions[:, 1] / self.height + self.light_offset, 0, 1)

    def update_metrics(self):
        self.metrics = measure_school(self.positions, self.headings, self.speeds, self.interaction_radius)
        brightness = self.brightness
        mismatch = np.abs(brightness - self.preferred_light)
        has_light = self.light_enabled and self.num_fish > 0
        self.metrics["Mean brightness"] = float(np.mean(brightness)) if has_light else float("nan")
        self.metrics["Fraction in dark region"] = float(np.mean(brightness <= 0.25)) if has_light else float("nan")
        self.metrics["Mean light error"] = float(np.mean(mismatch)) if has_light else float("nan")
        self.metrics["Fraction in preferred band"] = float(np.mean(mismatch <= self.light_tolerance + 1e-12)) if has_light else float("nan")
        self.metrics.update({"Population": self.num_fish, "Births": self.births,
                             "Light deaths": self.light_deaths, "Predator deaths": self.predator_deaths,
                             "Mean health": float(np.mean([f.health for f in self.fish])) if self.fish else 0.0,
                             "Schooling fraction": float(np.mean([f.school_neighbors >= self.school_min_neighbors for f in self.fish])) if self.fish else 0.0,
                             "Predator active": int(self.predator_position is not None),
                             "Storm active": int(self.storm_until > self.steps * self.dt)})
        p, r = self.metrics["Polarization"], self.metrics["Rotation"]
        if self.num_fish == 0:
            self.collective_state = "Extinct"
        elif self.num_fish == 1:
            self.collective_state = "Solitary"
        elif self.metrics["Cohesion"] < 0.8:
            self.collective_state = "Fragmented"
        elif p > 0.65 and r < 0.35:
            self.collective_state = "Polarized"
        elif r > 0.65 and p < 0.35:
            self.collective_state = "Milling"
        elif p < 0.35 and r < 0.35:
            self.collective_state = "Swarm"
        else:
            self.collective_state = "Transition"

    def step(self):
        if not self.running:
            return
        self.update_disturbances()
        self.refresh_snapshot()
        self.agents.do("plan")
        self.agents.do("advance")
        self.update_population()
        self.refresh_snapshot()
        self.update_metrics()
        self.datacollector.collect(self)
        if not self.fish:
            self.running = False

    def remove_fish(self, fish, cause):
        self.space.remove_agent(fish)
        self.fish.remove(fish)
        fish.remove()  # Deregister from Mesa's AgentSet as well.
        if cause == "light":
            self.light_deaths += 1
        else:
            self.predator_deaths += 1

    def update_population(self):
        """Apply deaths first, then pair eligible adults; newborns act next tick."""
        now = self.steps * self.dt
        if self.predator_position is not None and now >= self.next_predator_catch and self.fish:
            victim = min(self.fish, key=lambda f: np.linalg.norm(np.asarray(f.pos) - self.predator_position))
            if np.linalg.norm(np.asarray(victim.pos) - self.predator_position) <= 0.9:
                self.remove_fish(victim, "predator")
                self.next_predator_catch = now + 1.0
        if not self.lifecycle_enabled or not self.light_enabled:
            return
        for fish in list(self.fish):
            fish.age += self.dt
            mismatch = abs(self.light_at(fish.pos) - self.preferred_light)
            if mismatch <= self.light_tolerance + 1e-12:
                fish.health = min(1.0, fish.health + self.dt / self.recovery_time)
                fish.safe_time += self.dt
            else:
                severity = 1 + (mismatch - self.light_tolerance) / max(self.preferred_light, 1 - self.preferred_light)
                fish.health -= severity * self.dt / self.survival_time
                fish.safe_time = 0.0
            if fish.health <= 0:
                self.remove_fish(fish, "light")

        eligible = [f for f in self.fish if f.age >= self.maturity_time
                    and f.safe_time + 1e-9 >= self.reproduction_time and f.health >= 0.8
                    and f.state != "escaping"]
        used = set()
        for parent in eligible:
            if len(self.fish) >= self.max_population:
                break
            if parent in used:
                continue
            mates = [f for f in eligible if f is not parent and f not in used
                     and np.linalg.norm(np.asarray(f.pos) - parent.pos) <= self.interaction_radius]
            if not mates:
                continue
            mate = min(mates, key=lambda f: np.linalg.norm(np.asarray(f.pos) - parent.pos))
            center = (np.asarray(parent.pos) + mate.pos) / 2
            angle = self.random.uniform(-math.pi, math.pi)
            position = center + self.repulsion_radius * np.array([math.cos(angle), math.sin(angle)])
            position = np.clip(position, [0, 0], np.nextafter([self.width, self.height], [0, 0]))
            child = Fish(self, len(self.fish), position, parent.heading, newborn=True)
            self.fish.append(child)
            self.births += 1
            parent.safe_time = mate.safe_time = 0.0
            used.update((parent, mate))

    def trigger_predator(self, duration=12.0):
        """Introduce a predator near the school; return False if one is active."""
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be positive and finite")
        if not self.fish or self.predator_position is not None:
            return False
        center = np.mean([f.pos for f in self.fish], axis=0)
        self.predator_position = np.clip(center + [10, 0], [0, 0], [self.width, self.height])
        self.predator_until = self.steps * self.dt + duration
        self.next_predator_catch = self.steps * self.dt
        self.events.append({"time": self.steps * self.dt, "event": "predator", "duration": duration})
        self.update_metrics()
        return True

    def trigger_storm(self, duration=25.0):
        """Darken the tank and introduce a current, temporarily shifting the band."""
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be positive and finite")
        if not self.fish or self.storm_until > self.steps * self.dt:
            return False
        self.storm_until = self.steps * self.dt + duration
        self.light_offset = -0.35
        self.current = np.array([1.0, 0.3])
        self.events.append({"time": self.steps * self.dt, "event": "storm", "duration": duration})
        self.refresh_snapshot()
        self.update_metrics()
        return True

    def clear_disturbances(self):
        self.predator_position = None
        self.predator_until = self.storm_until = 0.0
        self.light_offset = 0.0
        self.current = np.zeros(2)
        self.events.append({"time": self.steps * self.dt, "event": "clear", "duration": 0.0})
        self.refresh_snapshot()
        self.update_metrics()

    def predator_response(self, position):
        if self.predator_position is None:
            return np.zeros(2)
        away = np.asarray(position) - self.predator_position
        distance = np.linalg.norm(away)
        if distance >= 12:
            return np.zeros(2)
        return 8.0 * unit(away) if distance > 1e-12 else np.array([8.0, 0.0])

    def update_disturbances(self):
        now = self.steps * self.dt
        if now >= self.storm_until:
            self.light_offset = 0.0
            self.current = np.zeros(2)
        if self.predator_position is not None:
            if now >= self.predator_until:
                self.predator_position = None
            elif self.fish:
                target = min(self.fish, key=lambda f: np.linalg.norm(np.asarray(f.pos) - self.predator_position))
                delta = np.asarray(target.pos) - self.predator_position
                self.predator_position += unit(delta) * min(np.linalg.norm(delta), 5.0 * self.dt)
