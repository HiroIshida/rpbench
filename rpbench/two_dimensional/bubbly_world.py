from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Type, TypeVar, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from skmp.constraint import BoxConst, ConfigPointConst, PointCollFreeConst
from skmp.solver.interface import Problem, ResultProtocol
from skmp.solver.nlp_solver.sqp_based_solver import SQPBasedSolver, SQPBasedSolverConfig
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig
from skmp.trajectory import Trajectory

from rpbench.interface import DescriptionTable, SDFProtocol, TaskBase, WorldBase
from rpbench.two_dimensional.utils import Grid2d, Grid2dSDF
from rpbench.utils import temp_seed


@dataclass
class CircleObstacle:
    center: np.ndarray
    radius: float

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        dists = np.sqrt(np.sum((points - self.center) ** 2, axis=1)) - self.radius
        return dists

    def is_colliding(self, pos: np.ndarray, margin: float = 0.0) -> bool:
        dist = self.signed_distance(np.expand_dims(pos, axis=0))[0] - margin
        return dist < 0.0

    def as_vector(self) -> np.ndarray:
        return np.hstack([self.center, self.radius])


BubblyWorldT = TypeVar("BubblyWorldT", bound="BubblyWorldBase")


@dataclass
class BubblyMetaParameter:
    n_obs: int
    circle_r_min: float
    circle_r_max: float


@dataclass
class BubblyWorldBase(WorldBase):
    obstacles: List[CircleObstacle]

    @classmethod
    @abstractmethod
    def get_meta_parameter(cls) -> BubblyMetaParameter:
        ...

    @classmethod
    def get_margin(cls) -> float:
        return 0.01

    @classmethod
    def sample(cls: Type[BubblyWorldT], standard: bool = False) -> BubblyWorldT:
        meta_param = cls.get_meta_parameter()
        n_obs = meta_param.n_obs
        r_min = meta_param.circle_r_min
        r_max = meta_param.circle_r_max

        obstacles = []
        start_pos = np.ones(2) * 0.1
        standard_goal_pos = np.ones(2) * 0.95

        with temp_seed(1, standard):
            while True:
                center = np.random.rand(2)
                radius = r_min + np.random.rand() * (r_max - r_min)
                obstacle = CircleObstacle(center, radius)

                if obstacle.is_colliding(start_pos, cls.get_margin()):
                    continue
                if standard and obstacle.is_colliding(standard_goal_pos, cls.get_margin()):
                    continue

                obstacles.append(obstacle)
                if len(obstacles) == n_obs:
                    return cls(obstacles)

    def export_intrinsic_description(self) -> np.ndarray:
        return np.hstack([obs.as_vector() for obs in self.obstacles])

    def get_grid(self) -> Grid2d:
        return Grid2d(np.zeros(2), np.ones(2), (112, 112))

    def get_exact_sdf(self, for_visualize: bool = False) -> SDFProtocol:
        margin = 0.0 if for_visualize else self.get_margin()

        def f(x: np.ndarray):
            dist_list = [obs.signed_distance(x) - margin for obs in self.obstacles]
            return np.min(np.array(dist_list), axis=0)

        return f

    def visualize(self, fax) -> None:
        fig, ax = fax

        n_grid = 200
        xlin = np.linspace(0.0, 1.0, n_grid)
        ylin = np.linspace(0.0, 1.0, n_grid)
        meshes = np.meshgrid(xlin, ylin)
        meshes_flatten = [mesh.flatten() for mesh in meshes]
        pts = np.array([p for p in zip(*meshes_flatten)])

        sdf = self.get_exact_sdf(for_visualize=True)

        sdf_mesh = sdf(pts).reshape(n_grid, n_grid)
        # ax.contourf(xlin, ylin, sdf_mesh, cmap="summer")
        ax.contourf(xlin, ylin, sdf_mesh, levels=[-100, 0.0, 100], colors=["gray", "white"])
        ax.contour(xlin, ylin, sdf_mesh, cmap="gray", levels=[0.0])


@dataclass
class BubblyWorldSimple(BubblyWorldBase):
    @classmethod
    def get_meta_parameter(cls) -> BubblyMetaParameter:
        return BubblyMetaParameter(10, 0.05, 0.2)


@dataclass
class BubblyWorldComplex(BubblyWorldBase):
    @classmethod
    def get_meta_parameter(cls) -> BubblyMetaParameter:
        return BubblyMetaParameter(40, 0.04, 0.08)


class BubblyPointConnectTaskBase(TaskBase[BubblyWorldT, Tuple[np.ndarray, ...], None]):
    @staticmethod
    def get_robot_model() -> None:
        return None

    @classmethod
    def sample_descriptions(
        cls, world: BubblyWorldT, n_sample: int, standard: bool = False
    ) -> List[Tuple[np.ndarray, ...]]:

        descriptions = []

        start = np.ones(2) * 0.1
        for _ in range(n_sample):
            if standard:
                goal = np.ones(2) * 0.95
            else:
                sdf = world.get_exact_sdf()

                while True:
                    goal = np.random.rand(2)
                    if np.linalg.norm(goal - start) > 0.4:
                        val = sdf(np.expand_dims(goal, axis=0))[0]
                        if val > 0.0:
                            break
            descriptions.append((start, goal))
        return descriptions  # type: ignore

    def export_table(self) -> DescriptionTable:
        wd = {}
        if self._gridsdf is None:
            wd["world"] = self.world.export_intrinsic_description()
        else:
            wd["world"] = self._gridsdf.values.reshape(self._gridsdf.grid.sizes)

        wcd_list = []
        for desc in self.descriptions:
            wcd = {}
            wcd["start"] = desc[0]
            wcd["goal"] = desc[1]
            wcd_list.append(wcd)
        return DescriptionTable(wd, wcd_list)

    def solve_default_each(self, problem: Problem) -> ResultProtocol:
        ompl_sovler = OMPLSolver.init(OMPLSolverConfig(n_max_call=10000, simplify=True))
        ompl_sovler.setup(problem)
        ompl_res = ompl_sovler.solve()
        if ompl_res.traj is None:
            return ompl_res

        nlp_solver = SQPBasedSolver.init(SQPBasedSolverConfig(n_wp=30))
        nlp_solver.setup(problem)
        res = nlp_solver.solve(ompl_res.traj)
        return res

    @classmethod
    def get_dof(cls) -> int:
        return 2

    def export_problems(self) -> List[Problem]:
        if self._gridsdf is None:
            sdf = self.world.get_exact_sdf()
        else:
            sdf = self._gridsdf

        probs = []
        for desc in self.descriptions:
            start, goal = desc

            box = BoxConst(np.zeros(self.get_dof()), np.ones(self.get_dof()))
            goal_const = ConfigPointConst(goal)
            prob = Problem(
                start, box, goal_const, PointCollFreeConst(sdf), None, motion_step_box_=0.03
            )
            probs.append(prob)
        return probs

    def export_intrinsic_descriptions(self) -> List[np.ndarray]:
        return [self.world.export_intrinsic_description()] * self.n_inner_task


class WithoutGridSDFMixin:
    @staticmethod
    def create_gridsdf(world: BubblyWorldBase, robot_model: None) -> None:
        return None


class WithGridSDFMixin:
    @staticmethod
    def create_gridsdf(world: BubblyWorldBase, robot_model: None) -> Grid2dSDF:
        grid = world.get_grid()

        xlin, ylin = [np.linspace(grid.lb[i], grid.ub[i], grid.sizes[i]) for i in range(2)]
        X, Y = np.meshgrid(xlin, ylin)
        pts = np.array(list(zip(X.flatten(), Y.flatten())))

        sdf = world.get_exact_sdf()
        vals = sdf.__call__(pts)

        itp = RegularGridInterpolator(
            (xlin, ylin), vals.reshape(grid.sizes).T, bounds_error=False, fill_value=10.0
        )
        return Grid2dSDF(vals, world.get_grid(), itp)


class SimpleWorldProviderMixin:
    @staticmethod
    def get_world_type() -> Type[BubblyWorldSimple]:
        return BubblyWorldSimple


class ComplexWorldProviderMixin:
    @staticmethod
    def get_world_type() -> Type[BubblyWorldComplex]:
        return BubblyWorldComplex


class BubblySimplePointConnectTask(
    SimpleWorldProviderMixin, WithoutGridSDFMixin, BubblyPointConnectTaskBase[BubblyWorldSimple]
):
    ...


class BubblySimpleMeshPointConnectTask(
    SimpleWorldProviderMixin, WithGridSDFMixin, BubblyPointConnectTaskBase[BubblyWorldSimple]
):
    ...


class BubblyComplexPointConnectTask(
    ComplexWorldProviderMixin, WithoutGridSDFMixin, BubblyPointConnectTaskBase[BubblyWorldComplex]
):
    ...


class BubblyComplexMeshPointConnectTask(
    ComplexWorldProviderMixin, WithGridSDFMixin, BubblyPointConnectTaskBase[BubblyWorldComplex]
):
    ...


class Taskvisualizer:
    fax: Tuple

    def __init__(self, task: BubblyPointConnectTaskBase):
        fig, ax = plt.subplots()
        task.world.visualize((fig, ax))
        for start, goal in task.descriptions:
            ax.scatter(start[0], start[1], c="k")
            ax.scatter(goal[0], goal[1], c="r")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        self.fax = (fig, ax)

    def visualize_trajectories(self, trajs: Union[List[Trajectory], Trajectory]) -> None:
        fig, ax = self.fax
        if isinstance(trajs, Trajectory):
            trajs = [trajs]

        for traj in trajs:
            arr = traj.numpy()
            ax.plot(arr[:, 0], arr[:, 1])
