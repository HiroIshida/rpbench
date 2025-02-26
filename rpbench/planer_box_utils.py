import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class PlanerCoords:
    pos: np.ndarray
    angle: float

    @classmethod
    def create(cls, x, y, angle) -> "PlanerCoords":
        return PlanerCoords(np.array([x, y]), angle)

    @classmethod
    def standard(cls) -> "PlanerCoords":
        return PlanerCoords.create(0.0, 0.0, 0.0)


def rotation_matrix_2d(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    rotmat = np.array([[c, -s], [s, c]])
    return rotmat


class Primitive2d:
    ...


@dataclass
class Circle(Primitive2d):
    center: np.ndarray
    radius: float

    def visualize(self, fax, color="red") -> None:
        fig, ax = fax
        ax.add_patch(plt.Circle(self.center, self.radius, color=color, fill=False))


@dataclass
class Box2d(Primitive2d):
    extent: np.ndarray
    coords: PlanerCoords

    @property
    def verts(self) -> np.ndarray:
        half_extent = self.extent * 0.5
        dir1 = rotation_matrix_2d(self.coords.angle).dot(half_extent)

        half_extent_rev = half_extent
        half_extent_rev[0] *= -1.0

        dir2 = rotation_matrix_2d(self.coords.angle).dot(half_extent_rev)

        v1 = self.coords.pos + dir1
        v2 = self.coords.pos + dir2
        v3 = self.coords.pos - dir1
        v4 = self.coords.pos - dir2
        return np.array([v1, v2, v3, v4])

    @property
    def edges(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        v0, v1, v2, v3 = self.verts
        return [(v0, v1), (v1, v2), (v2, v3), (v3, v0)]

    def visualize(self, fax, color="red") -> None:
        fig, ax = fax
        verts = self.verts
        ax.scatter(verts[:, 0], verts[:, 1], c=color)

        idx_pairs = [[0, 1], [1, 2], [2, 3], [3, 0]]
        for i, j in idx_pairs:
            v1 = verts[i]
            v2 = verts[j]
            ax.plot([v1[0], v2[0]], [v1[1], v2[1]], color=color)

    def sd(self, points):
        n_pts, _ = points.shape
        half_extent = self.extent * 0.5

        # pts_from_center = points - self.coords.pos
        # consider yaw
        mat = rotation_matrix_2d(-self.coords.angle)
        pts_from_center = mat.dot((points - self.coords.pos).T).T
        sd_vals_each_axis = np.abs(pts_from_center) - half_extent[None, :]

        positive_dists_each_axis = np.maximum(sd_vals_each_axis, 0.0)
        positive_dists = np.sqrt(np.sum(positive_dists_each_axis**2, axis=1))

        negative_dists_each_axis = np.max(sd_vals_each_axis, axis=1)
        negative_dists = np.minimum(negative_dists_each_axis, 0.0)

        sd_vals = positive_dists + negative_dists
        return sd_vals

    def collides_with_box(self, other: "Box2d") -> bool:
        def is_separating(edge: Tuple[np.ndarray, np.ndarray]):
            v0, v1 = edge
            vec_self = v1 - v0
            for v_other in other.verts:
                vec_other = v_other - v0
                if np.linalg.det(np.vstack([vec_self, vec_other]).T) > 0:
                    return False
            return True

        for edge in self.edges:
            if is_separating(edge):
                return False
        return True

    def contains(self, other: Primitive2d) -> bool:
        if isinstance(other, Box2d):
            return bool(np.all(self.sd(other.verts) < 0.0))
        elif isinstance(other, Circle):
            sdist = self.sd(np.expand_dims(other.center, axis=0))[0]
            return sdist < -other.radius
        else:
            raise NotImplementedError

    def sample_point(self) -> np.ndarray:
        half_extent = self.extent * 0.5
        x = np.random.uniform(-half_extent[0], half_extent[0])
        y = np.random.uniform(-half_extent[1], half_extent[1])
        return self.coords.pos + rotation_matrix_2d(self.coords.angle).dot(np.array([x, y]))


def is_colliding(shape1: Primitive2d, shape2: Primitive2d) -> bool:

    if isinstance(shape1, Circle) and isinstance(shape2, Circle):
        dist = np.linalg.norm(shape1.center - shape2.center)
        return bool(dist < shape1.radius + shape2.radius)

    elif isinstance(shape1, Box2d) and isinstance(shape2, Box2d):
        # NOTE: This is just a heuristic. It is not a perfect collision detection.
        if np.any(shape1.sd(shape2.verts) < 0.0) or np.any(shape2.sd(shape1.verts) < 0.0):
            return True
        if shape2.sd(shape1.coords.pos[None, :])[0] < 0.0:
            return True
        if shape1.sd(shape2.coords.pos[None, :])[0] < 0.0:
            return True
        return False
    elif isinstance(shape1, Box2d) and isinstance(shape2, Circle):
        return shape1.sd(np.expand_dims(shape2.center, axis=0))[0] < shape2.radius
    elif isinstance(shape1, Circle) and isinstance(shape2, Box2d):
        return shape2.sd(np.expand_dims(shape1.center, axis=0))[0] < shape1.radius
    else:
        raise NotImplementedError


def sample_box(
    table_extent: np.ndarray, box_extent: np.ndarray, obstacles: List[Box2d], n_budget: int = 30
) -> Optional[Box2d]:
    table = Box2d(table_extent, PlanerCoords.standard())

    for _ in range(n_budget):
        box_pos_cand = -0.5 * table_extent + table_extent * np.random.rand(2)
        angle_cand = -0.5 * np.pi + np.random.rand() * np.pi
        box_cand = Box2d(box_extent, PlanerCoords(box_pos_cand, angle_cand))

        def is_valid(box_cand):
            is_inside = np.all(table.sd(box_cand.verts) < 0.0)
            if is_inside:
                for obs in obstacles:
                    if is_colliding(box_cand, obs):
                        return False
                return True
            return False

        if is_valid(box_cand):
            return box_cand
    return None
