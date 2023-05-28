from dataclasses import dataclass
from typing import List

import numpy as np
from voxbloxpy.core import Grid

from rpbench.interface import SDFProtocol, WorldBase
from rpbench.utils import temp_seed


@dataclass
class CircleObstacle:
    center: np.ndarray
    radius: float

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        dists = np.sqrt(np.sum((points - self.center) ** 2, axis=1)) - self.radius
        return dists

    def is_colliding(self, pos: np.ndarray) -> bool:
        dist = self.signed_distance(np.expand_dims(pos, axis=0))[0]
        return dist < 0.0

    def as_vector(self) -> np.ndarray:
        return np.hstack([self.center, self.radius])


@dataclass
class BubblyWorld(WorldBase):
    obstacles: List[CircleObstacle]

    @classmethod
    def sample(cls, standard: bool = False) -> "BubblyWorld":
        n_obs = 12

        obstacles = []
        start_pos = np.ones(2) * 0.1
        standard_goal_pos = np.ones(2) * 0.9

        with temp_seed(1, standard):
            while True:
                center = np.random.rand(2)
                radius = 0.06 + np.random.rand() * 0.06
                obstacle = CircleObstacle(center, radius)

                if obstacle.is_colliding(start_pos):
                    continue
                if standard and obstacle.is_colliding(standard_goal_pos):
                    continue

                obstacles.append(obstacle)
                if len(obstacles) == n_obs:
                    return cls(obstacles)

    def export_intrinsic_description(self) -> np.ndarray:
        return np.hstack([obs.as_vector() for obs in self.obstacles])

    def get_grid(self) -> Grid:
        return Grid(np.zeros(2), np.ones(2), (56, 56, 56))

    def get_exact_sdf(self) -> SDFProtocol:
        def f(x: np.ndarray):
            dist_list = [obs.signed_distance(x) for obs in self.obstacles]
            return np.min(np.array(dist_list), axis=0)

        return f

    def visualize(self, fax) -> None:
        fig, ax = fax

        n_grid = 100
        xlin = np.linspace(0.0, 1.0, n_grid)
        ylin = np.linspace(0.0, 1.0, n_grid)
        meshes = np.meshgrid(xlin, ylin)
        meshes_flatten = [mesh.flatten() for mesh in meshes]
        pts = np.array([p for p in zip(*meshes_flatten)])

        sdf = self.get_exact_sdf()

        sdf_mesh = sdf(pts).reshape(n_grid, n_grid)
        ax.contourf(xlin, ylin, sdf_mesh, cmap="summer")
        ax.contour(xlin, ylin, sdf_mesh, cmap="gray", levels=[0.0])
