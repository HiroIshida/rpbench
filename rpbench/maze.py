from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Type, TypeVar

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from skmp.constraint import BoxConst, ConfigPointConst, PointCollFreeConst
from skmp.solver.interface import Problem, ResultProtocol
from skmp.solver.nlp_solver.sqp_based_solver import SQPBasedSolver, SQPBasedSolverConfig
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig

from rpbench.interface import DescriptionTable, GridSDFProtocol, TaskBase, WorldBase
from rpbench.utils import temp_seed

MazeWorldT = TypeVar("MazeWorldT", bound="MazeWorldBase")


@dataclass
class Grid2d:
    lb: np.ndarray
    ub: np.ndarray
    sizes: Tuple[int, int]


@dataclass
class Grid2dSDF:
    values: np.ndarray
    grid: Grid2d
    itp: RegularGridInterpolator

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.itp(x)


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

    def get_grid(self) -> Grid2d:
        return Grid2d(np.zeros(2), np.ones(2), (100, 100))

    def get_exact_sdf(self) -> Grid2dSDF:
        box_width = np.ones(2) / self.M.shape
        index_pair_list = [np.array(e) for e in zip(*np.where(self.M == 1))]

        grid = self.get_grid()

        xlin, ylin = [np.linspace(grid.lb[i], grid.ub[i], grid.sizes[i]) for i in range(2)]
        X, Y = np.meshgrid(xlin, ylin)
        pts = np.array(list(zip(X.flatten(), Y.flatten())))

        box_width = np.ones(2) / self.M.shape

        vals = np.ones(len(pts)) * np.inf
        for index_pair in index_pair_list:
            pos = box_width * np.array(index_pair) + box_width * 0.5
            vals_cand = MazeWorldBase.compute_box_sdf(pts, pos, box_width)
            vals = np.minimum(vals, vals_cand)

        itp = RegularGridInterpolator(
            (xlin, ylin), vals.reshape(grid.sizes), bounds_error=False, fill_value=1.0
        )
        return Grid2dSDF(vals, self.get_grid(), itp)

    def visualize(self, with_contour: bool = False) -> Tuple:
        fig, ax = plt.subplots()
        grid = self.get_grid()
        xlin, ylin = [np.linspace(grid.lb[i], grid.ub[i], 200) for i in range(2)]
        meshes = np.meshgrid(xlin, ylin)
        meshes_flatten = [mesh.flatten() for mesh in meshes]
        pts = np.array([p for p in zip(*meshes_flatten)])

        sdf = self.get_exact_sdf()
        sdf_mesh = sdf(pts).reshape(200, 200)
        if with_contour:
            ax.contourf(xlin, ylin, sdf_mesh, cmap="summer")
        ax.contour(xlin, ylin, sdf_mesh, cmap="gray", levels=[0.0])
        return fig, ax


@dataclass
class MazeWorld(MazeWorldBase):
    @classmethod
    def get_param(cls) -> MazeParam:
        return MazeParam(23, 0.9, 0.9)


StartAndGoal = Tuple[np.ndarray, np.ndarray]


class MazeSolvingTask(TaskBase[MazeWorld, StartAndGoal, None]):
    @staticmethod
    def get_world_type() -> Type[MazeWorld]:
        return MazeWorld

    @staticmethod
    def get_robot_model() -> None:
        return None

    @staticmethod
    def create_gridsdf(world: MazeWorld, robot_model: None) -> GridSDFProtocol:
        return world.get_exact_sdf()

    @classmethod
    def sample_descriptions(
        cls, world: MazeWorld, n_sample: int, standard: bool = False
    ) -> List[StartAndGoal]:

        maze_param = world.get_param()

        sdf = world.get_exact_sdf()

        if standard:
            assert n_sample == 1
            cell_width = 1.0 / maze_param.width
            start = np.array([cell_width * 1.5, cell_width * 1.5])
            goal = np.array([1.0 - cell_width * 1.5, 1.0 - cell_width * 1.5])
            return [(start, goal)]
        else:
            descs: List[StartAndGoal] = []
            while len(descs) < n_sample:
                start = np.random.rand(2)
                goal = np.random.rand(2)
                if np.all(sdf(np.stack([start, goal])) > 0):
                    descs.append((start, goal))
            return descs

    def export_table(self) -> DescriptionTable:
        assert self._gridsdf is not None
        wd = {}
        wd["world"] = self.world.M.flatten()  # vector description

        wcd_list = []
        for desc in self.descriptions:
            wcd = {}
            wcd["start"] = desc[0]
            wcd["goal"] = desc[1]
            wcd_list.append(wcd)
        return DescriptionTable(wd, wcd_list)

    def solve_default_each(self, problem: Problem) -> ResultProtocol:
        ompl_sovler = OMPLSolver.init(OMPLSolverConfig(n_max_call=50000, simplify=True))
        ompl_sovler.setup(problem)
        ompl_res = ompl_sovler.solve()

        nlp_solver = SQPBasedSolver.init(SQPBasedSolverConfig(n_wp=100))
        nlp_solver.setup(problem)
        res = nlp_solver.solve(ompl_res.traj)
        return res

    @classmethod
    def get_dof(cls) -> int:
        return 2

    def export_problems(self) -> List[Problem]:
        sdf = self.world.get_exact_sdf()
        probs = []
        for desc in self.descriptions:
            start, goal = desc

            box = BoxConst(np.zeros(self.get_dof()), np.ones(self.get_dof()))
            goal_const = ConfigPointConst(goal)
            prob = Problem(
                start, box, goal_const, PointCollFreeConst(sdf), None, motion_step_box_=0.02
            )
            probs.append(prob)
        return probs

    def visualize(self, debug: bool = False) -> Tuple:
        fig, ax = self.world.visualize(debug)

        for desc in self.descriptions:
            start, goal = desc
            ax.scatter(start[0], start[1], label="start")
            ax.scatter(goal[0], goal[1], label="goal")
        return fig, ax
