"""Puckett et al. (2018) movement equations and a reproducible light-field surrogate.

Equations 6–7 and supplement equations 4–10 are implemented in body lengths.
The light generator is our approximation, not the authors' original noise movie.
See PAPER_MODEL.md for the parameter mapping and replication limits.
"""
###
import math

import numpy as np


class MovingLightField:
    """Gaussian dark patch plus smooth noise, independent of fish and their RNG."""

    def __init__(self, width, height, seed=42, noise=0.25, speed=1.142, decay=7.7):
        self.bounds = np.array([width, height], dtype=float)
        self.noise, self.speed, self.decay = noise, speed, decay
        path_seed, noise_seed = np.random.SeedSequence(seed).spawn(2)
        self.path_rng = np.random.default_rng(path_seed)
        rng = np.random.default_rng(noise_seed)
        self.nodes = [self.bounds * [0.5, 0.3]]
        self.angles = []
        self.wavevectors = rng.normal(0, 0.6, (8, 2))
        self.frequencies = rng.uniform(-0.7, 0.7, 8)
        self.phases = rng.uniform(-math.pi, math.pi, 8)

    def center_at(self, time):
        # Choose a new random direction every five seconds, reflecting at walls.
        segment = int(time // 5)
        while len(self.angles) <= segment:
            angle = self.path_rng.uniform(-math.pi, math.pi)
            self.angles.append(angle)
            delta = self.speed * 5 * np.array([math.cos(angle), math.sin(angle)])
            self.nodes.append(self._reflect(self.nodes[-1] + delta))
        angle = self.angles[segment]
        delta = self.speed * (time - segment * 5) * np.array([math.cos(angle), math.sin(angle)])
        return self._reflect(self.nodes[segment] + delta)

    def _reflect(self, position):
        folded = np.mod(position, 2 * self.bounds)
        return np.minimum(folded, 2 * self.bounds - folded)

    def values(self, positions, time):
        positions = np.asarray(positions, dtype=float)
        radius_squared = np.sum((positions - self.center_at(time)) ** 2, axis=-1)
        base = 1 - np.exp(-radius_squared / (2 * self.decay ** 2))
        phase = positions @ self.wavevectors.T + self.frequencies * time + self.phases
        background = np.sin(phase).sum(axis=-1) / math.sqrt(4)
        return np.clip(base + self.noise * background, 0, 1)


def social_direction(model, index):
    """Disjoint repulsion/orientation/attraction zones (supplement Eqs. 4–7)."""
    from model import unit

    forward = model.directions[index]
    if not model.social_enabled:
        return forward.copy(), 0
    offsets = model.positions - model.positions[index]
    distances = np.linalg.norm(offsets, axis=1)
    others = np.arange(model.num_fish) != index
    bearings = np.arctan2(offsets[:, 1], offsets[:, 0])
    angles = (bearings - model.headings[index] + math.pi) % (2 * math.pi) - math.pi
    visible = others & (np.abs(angles) <= math.radians(model.vision_angle) / 2)
    # Emergency repulsion includes the rear blind sector, as in our bounded tank.
    close = others & (distances < model.repulsion_radius)
    neighbors = int(np.count_nonzero(visible & (distances <= model.interaction_radius)))
    if close.any():
        direction = -np.sum(offsets[close] / np.maximum(distances[close, None], 1e-12), axis=0)
        if np.linalg.norm(direction) < 1e-12:
            # Deterministic opposite directions also separate exact overlaps.
            angle = index * math.pi * (3 - math.sqrt(5))
            direction = np.array([math.cos(angle), math.sin(angle)])
    else:
        orient = visible & (distances >= model.repulsion_radius) & (distances < 3.0)
        attract = visible & (distances >= 3.0) & (distances <= model.interaction_radius)
        direction = model.directions[orient].sum(axis=0)
        direction += np.sum(offsets[attract] / distances[attract, None], axis=0)
    return (unit(direction) if np.linalg.norm(direction) > 1e-12 else forward.copy()), neighbors


def environmental_direction(model, index):
    """Shiners never sample a gradient; tetra/custom modes may probe locally."""
    if not model.light_enabled or model.effective_gradient_weight == 0:
        return np.zeros(2)
    position = model.positions[index]
    probe = 0.01  # BL; centered finite differences in the displayed light field.
    gradient = np.array([(model.light_at(position + axis * probe)
                          - model.light_at(position - axis * probe)) / (2 * probe)
                         for axis in np.eye(2)])
    magnitude = np.linalg.norm(gradient)
    # Table S1 threshold is per cm: multiply by 5 cm/BL when using BL.
    if magnitude < 0.0005:
        return np.zeros(2)
    angle = math.atan2(-gradient[1], -gradient[0])
    angle += model.sensing_rng.normal(0, model.gradient_error)
    return np.array([math.cos(angle), math.sin(angle)])


def plan_paper_fish(fish):
    """Normalize social/environmental cues separately, then combine with w."""
    model = fish.model
    social, fish.school_neighbors = social_direction(model, fish.index)
    # Draw social error even for w=0 so matched treatments consume the same RNG.
    error = model.random.gauss(0, model.noise)
    social = np.array([[math.cos(error), -math.sin(error)],
                       [math.sin(error), math.cos(error)]]) @ social
    environment = environmental_direction(model, fish.index)
    desired = social + model.effective_gradient_weight * environment
    fish.next_state = "schooling" if fish.school_neighbors >= model.school_min_neighbors else "roaming"
    danger = model.predator_response(model.positions[fish.index])
    if np.linalg.norm(danger) > 0:
        desired = danger
        fish.next_state = "escaping"
    # Reflection during advance supplies the bounded-arena wall rule.
    heading = math.atan2(desired[1], desired[0]) if np.linalg.norm(desired) > 1e-12 else fish.heading
    turn = (heading - fish.heading + math.pi) % (2 * math.pi) - math.pi
    fish.next_heading = fish.heading + float(np.clip(turn, -model.max_turn * model.dt,
                                                    model.max_turn * model.dt))
    # Eq. 7 uses the instantaneous light-modulated speed, with no social speed term.
    fish.next_speed = (model.speed_from_light(model.light_at(fish.pos))
                       if model.light_enabled else model.base_speed)
    if fish.next_state == "escaping":
        fish.next_speed = model.max_speed
