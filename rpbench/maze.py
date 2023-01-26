from abc import abstractmethod
from dataclasses import dataclass
from typing import Tuple, Type, TypeVar

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from rpbench.interface import GridProtocol, SDFProtocol, WorldBase
from rpbench.utils import temp_seed

MazeWorldT = TypeVar("MazeWorldT", bound="MazeWorldBase")


@dataclass
class Grid:
    lb: np.ndarray
    ub: np.ndarray
    sizes: Tuple[int, ...]


@dataclass
class MazeParam:
    width: int
    complexity: float
    density: float


@dataclass
class MazeWorldBase(WorldBase):
    M: np.ndarray

    @classmethod
    @abstractmethod
    def get_param(cls) -> MazeParam:
        ...

    @classmethod
    def sample(cls: Type[MazeWorldT], standard: bool = False) -> MazeWorldT:
        param = cls.get_param()
        shape = (param.width, param.width)

        complexity = int(param.complexity * (5 * (shape[0] + shape[1])))
        density = int(param.density * ((shape[0] // 2) * (shape[1] // 2)))

        Z = np.zeros(shape, dtype=bool)
        Z[0, :] = Z[-1, :] = 1
        Z[:, 0] = Z[:, -1] = 1

        with temp_seed(0, standard):
            for i in range(density):
                x, y = (
                    np.random.randint(0, shape[1] // 2 + 1) * 2,
                    np.random.randint(0, shape[0] // 2 + 1) * 2,
                )
                Z[y, x] = 1
                for j in range(complexity):
                    neighbours = []
                    if x > 1:
                        neighbours.append((y, x - 2))
                    if x < shape[1] - 2:
                        neighbours.append((y, x + 2))
                    if y > 1:
                        neighbours.append((y - 2, x))
                    if y < shape[0] - 2:
                        neighbours.append((y + 2, x))
                    if len(neighbours):
                        y_, x_ = neighbours[np.random.randint(0, len(neighbours))]
                        if Z[y_, x_] == 0:
                            Z[y_, x_] = 1
                            Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                            x, y = x_, y_
        return cls(Z.astype(int))

    @staticmethod
    def compute_box_sdf(points: np.ndarray, origin: np.ndarray, width: np.ndarray):
        half_extent = width * 0.5
        pts_from_center = points - origin[None, :]
        sd_vals_each_axis = np.abs(pts_from_center) - half_extent[None, :]

        positive_dists_each_axis = np.maximum(sd_vals_each_axis, 0.0)
        positive_dists = np.sqrt(np.sum(positive_dists_each_axis**2, axis=1))

        negative_dists_each_axis = np.max(sd_vals_each_axis, axis=1)
        negative_dists = np.minimum(negative_dists_each_axis, 0.0)

        sd_vals = positive_dists + negative_dists
        return sd_vals

    def get_grid(self) -> GridProtocol:
        return Grid(np.zeros(2), np.ones(2), (100, 100))

    def get_exact_sdf(self) -> SDFProtocol:
        box_width = np.ones(2) / self.M.shape
        index_pair_list = [np.array(e) for e in zip(*np.where(self.M == 1))]
        np.ones((100, 100)) * np.inf

        N = 100
        b_min = np.zeros(2)
        b_max = np.ones(2)
        (b_max - b_min) / N
        xlin, ylin = [np.linspace(b_min[i], b_max[i], N) for i in range(2)]
        X, Y = np.meshgrid(xlin, ylin)
        pts = np.array(list(zip(X.flatten(), Y.flatten())))

        box_width = np.ones(2) / self.M.shape

        vals = np.ones(len(pts)) * np.inf
        for index_pair in index_pair_list:
            pos = box_width * np.array(index_pair) + box_width * 0.5
            vals_cand = MazeWorldBase.compute_box_sdf(pts, pos, box_width)
            vals = np.minimum(vals, vals_cand)

        itp = RegularGridInterpolator(pts, vals.T)
        return itp


@dataclass
class MazeWorld(MazeWorldBase):
    @classmethod
    def get_param(cls) -> MazeParam:
        return MazeParam(13, 0.1, 0.4)
