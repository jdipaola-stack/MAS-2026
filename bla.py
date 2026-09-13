import numpy as np
import mesa

from mesa.experimental.continuous_space import (
    ContinuousSpaceAgent,
    ContinuousSpace,
)

from mesa.visualization import SolaraViz, make_space_component
from matplotlib.markers import MarkerStyle


# ============================================================
# AGENT
# ============================================================

class GoldenShiners(ContinuousSpaceAgent):
    """Golden Shiner fish with simple Couzin-style repulsion."""

    def __init__(self, model, space, position=(0, 0), speed=1.0):
        super().__init__(space, model)

        self.position = np.array(position, dtype=float)
        self.speed = speed

        # Random initial direction
        angle = self.model.rng.uniform(0, 2 * np.pi)
        self.direction = np.array([
            np.cos(angle),
            np.sin(angle)
        ])

        # Variables used during simultaneous updating
        self.next_direction = self.direction.copy()
        self.next_position = self.position.copy()

    # --------------------------------------------------------
    # Find nearby fish
    # --------------------------------------------------------

    def get_neighbors(self):
        neighbors = []

        for other_fish in self.model.agents:

            if other_fish is self:
                continue

            displacement = other_fish.position - self.position
            distance = np.linalg.norm(displacement)

            if distance <= self.model.attraction_radius:
                neighbors.append(
                    (other_fish, displacement, distance)
                )

        return neighbors

    # --------------------------------------------------------
    # REPULSION
    # --------------------------------------------------------

    def repulsion(self, neighbors):

        repel = np.zeros(2)

        for _, displacement, distance in neighbors:

            # Only fish inside repulsion zone matter
            if distance < self.model.repulsion_radius:

                # Avoid division by zero
                if distance == 0:
                    continue

                # Unit vector pointing FROM us TO neighbor
                direction_neighbor = displacement / distance

                # Move in opposite direction
                repel -= direction_neighbor

        magnitude = np.linalg.norm(repel)

        if magnitude == 0:
            return None

        return repel / magnitude

    # --------------------------------------------------------
    # DECIDE
    # --------------------------------------------------------

    def decide(self):

        neighbors = self.get_neighbors()

        repel = self.repulsion(neighbors)

        # If there is a repulsion vector, use it.
        # Otherwise continue in current direction.
        if repel is not None:
            self.next_direction = repel
        else:
            self.next_direction = self.direction.copy()

        # Normalize just in case
        magnitude = np.linalg.norm(self.next_direction)

        if magnitude > 0:
            self.next_direction /= magnitude

        # Proposed movement
        movement = (
            self.next_direction
            * self.speed
            * self.model.dt
        )

        self.next_position = self.position + movement

        # Handle walls
        self.bounce(self.next_position)

    # --------------------------------------------------------
    # ADVANCE
    # --------------------------------------------------------

    def advance(self):

        # Everybody has already decided where to go.
        # Now everybody moves.

        self.direction = self.next_direction.copy()

        self.position = self.next_position.copy()

    # --------------------------------------------------------
    # WALL BOUNCING
    # --------------------------------------------------------

    def bounce(self, position):

        for axis, (minimum, maximum) in enumerate(self.model.bounds):

            if position[axis] < minimum:

                position[axis] = (
                    2 * minimum - position[axis]
                )

                self.next_direction[axis] *= -1

            elif position[axis] > maximum:

                position[axis] = (
                    2 * maximum - position[axis]
                )

                self.next_direction[axis] *= -1


# ============================================================
# MODEL
# ============================================================

class GoldenShinersModel(mesa.Model):

    """Phase 1D: Couzin repulsion."""

    def __init__(
        self,
        width=20,
        height=20,
        speed=1.0,
        n_fish=20,
        seed=None,
    ):

        super().__init__(seed=seed)

        # Space boundaries
        self.bounds = np.array([
            [0, width],
            [0, height],
        ])

        # Time step
        self.dt = 0.125

        # Couzin zones
        self.repulsion_radius = 0.5
        self.orientation_radius = 3.0
        self.attraction_radius = 5.5

        # Continuous space
        self.space = ContinuousSpace(
            self.bounds,
            torus=False,
            random=self.random,
            n_agents=n_fish,
        )

        # Create fish
        for _ in range(n_fish):

            position = (
                self.rng.random(2)
                * np.array([width, height])
            )

            GoldenShiners(
                self,
                self.space,
                position=position,
                speed=speed,
            )

    # --------------------------------------------------------
    # MODEL STEP
    # --------------------------------------------------------

    def step(self):

        # PHASE 1:
        # Everybody looks at the current state
        # and decides where they want to go.
        self.agents.do("decide")

        # PHASE 2:
        # Everybody moves.
        self.agents.do("advance")


# ============================================================
# VISUALIZATION
# ============================================================

def agent_draw(agent):
    return {
        "color": "blue",
        "size": 15,
        "marker": ">",
    }


# ============================================================
# SOLARA
# ============================================================

model = GoldenShinersModel(
    width=20,
    height=20,
    n_fish=20,
    speed=1.0,
    seed=1,
)

page = SolaraViz(
    model,
    components=[
        make_space_component(
            agent_portrayal=agent_draw
        )
    ],
    name="Phase 1D: Repulsion",
)

page