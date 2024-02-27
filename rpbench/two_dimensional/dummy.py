from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Type, TypeVar

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import gaussian_kde
from skmp.constraint import BoxConst, ConfigPointConst, PointCollFreeConst
from skmp.solver.interface import AbstractScratchSolver, Problem
from skmp.trajectory import Trajectory

from rpbench.interface import SDFProtocol, TaskBase, TaskExpression, WorldBase
from rpbench.two_dimensional.utils import Grid2d, Grid2dSDF
from rpbench.utils import temp_seed

DummyWorldT = TypeVar("DummyWorldT", bound="DummyWorldBase")
DummyTaskT = TypeVar("DummyTaskT", bound="DummyTaskBase")


@dataclass
class DummyWorldBase(WorldBase):
    kde: gaussian_kde
    b_min: np.ndarray
    b_max: np.ndarray

    @abstractmethod
    def visualize(self, fax) -> None:
        ...


class DummyWorld(DummyWorldBase):
    @classmethod
    def sample(cls, standard: bool = False) -> "DummyWorld":
        with temp_seed(100, True):
            n_data = 10
            y = np.random.randn(n_data)
            x = np.random.randn(n_data) * 1.5 + y
            xy = np.vstack([x, y])
            kde = gaussian_kde(xy)
        return cls(kde, np.array([-2.5, -2.0]), np.array([3.0, 2.0]))

    def get_exact_sdf(self) -> Grid2dSDF:
        grid = Grid2d(self.b_min, self.b_max, (56, 56))
        xlin, ylin = [np.linspace(grid.lb[i], grid.ub[i], grid.sizes[i]) for i in range(2)]
        X, Y = np.meshgrid(xlin, ylin)
        pts = np.array(list(zip(X.flatten(), Y.flatten())))
        vals = self.kde(pts.T) - 0.06
        val_min = np.min(vals)
        assert val_min < 0
        value_scale = -2.0 / val_min
        vals = vals * value_scale

        itp = RegularGridInterpolator(
            (xlin, ylin), vals.reshape(grid.sizes), bounds_error=False, fill_value=1.0
        )
        return Grid2dSDF(vals, grid, itp)

    def visualize(self, fax) -> None:
        fig, ax = fax

        n_grid = 200
        xlin = np.linspace(self.b_min[0], self.b_max[0], n_grid)
        ylin = np.linspace(self.b_min[1], self.b_max[1], n_grid)
        meshes = np.meshgrid(xlin, ylin)
        meshes_flatten = [mesh.flatten() for mesh in meshes]
        pts = np.array([p for p in zip(*meshes_flatten)])

        sdf = self.get_exact_sdf()

        sdf_mesh = sdf(pts).reshape(n_grid, n_grid)
        ax.contour(xlin, ylin, sdf_mesh, cmap="gray", levels=[0.0])
        # cf = ax.contourf(xlin, ylin, sdf_mesh, cmap="summer")
        # plt.colorbar(cf)


class ProbDummyWorld(DummyWorldBase):
    @classmethod
    def sample(cls, standard: bool = False) -> "ProbDummyWorld":
        with temp_seed(2, True):
            width = np.array([0.6, 1.0])
            margin = 0.15
            points = np.random.rand(10, 2) * (width - margin * 2) + margin
            kde = gaussian_kde(points.T)
        return cls(kde, np.array([-0.2, 0.0]), width)

    def get_exact_sdf(self) -> SDFProtocol:
        def sdf(X: np.ndarray) -> np.ndarray:
            r = 0.25
            ret1 = (X[:, 0] / 1.0) ** 2 + ((X[:, 1] - 0.5) / 0.5) ** 2 - r**2
            ret2 = ((X[:, 0] - 0.6) / 1.0) ** 2 + ((X[:, 1] - 0.3) / 0.5) ** 2 - r**2
            return np.minimum(ret1, ret2)

        return sdf

    def visualize(self, fax):
        fig, ax = fax

        x = np.linspace(0, 0.6, 100)
        y = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(x, y)
        pts = np.vstack([X.ravel(), Y.ravel()]).T
        Z = self.kde.evaluate(pts.T).reshape(X.shape)
        ax.contourf(X, Y, Z, 20, cmap=cm.Reds)

        sdf = self.get_exact_sdf()
        Z_infeasible = sdf(pts).reshape(X.shape)

        ax.set_xlim([0, 0.6])
        ax.set_ylim([0, 1.0])
        ax.set_aspect("equal", adjustable="box")
        ax.contourf(X, Y, Z, 20, cmap=cm.Reds)

        contourf_kwargs = {
            "colors": ["gray"],
            "alpha": 0.0,
            "hatches": ["////"],
            "edgecolor": "r",
            "facecolor": "blue",
            "color": "red",
        }
        ax.contourf(X, Y, Z_infeasible, levels=[-np.inf, 0], **contourf_kwargs)
        ax.contour(X, Y, Z_infeasible, levels=[0], colors="black")


@dataclass
class DummyConfig:
    n_max_call: int
    random: bool = True
    random_scale: float = 2.0
    random_force_failure_rate: float = 0.2  # to intensionaly create false positive state
    timeout: Optional[float] = None


@dataclass
class DummyResult:
    traj: Optional[Trajectory]
    time_elapsed: Optional[float]
    n_call: int

    @classmethod
    def abnormal(cls) -> "DummyResult":
        return cls(None, None, -1)


@dataclass
class DummySolver(AbstractScratchSolver[DummyConfig, DummyResult]):
    config: DummyConfig

    @classmethod
    def init(cls, config: DummyConfig) -> "DummySolver":
        return cls(config)

    def get_result_type(self) -> Type[DummyResult]:
        return DummyResult

    def _setup(self, problem: Problem):
        pass

    def _solve(self, guiding_traj: Optional[Trajectory] = None) -> DummyResult:
        assert guiding_traj is not None
        q_end_init = guiding_traj.numpy()[-1]
        assert self.problem is not None
        assert isinstance(self.problem.goal_const, ConfigPointConst)

        if self.problem.global_ineq_const is not None:
            if not self.problem.global_ineq_const.is_valid(self.problem.goal_const.desired_angles):
                return DummyResult.abnormal()

        q_end_target = self.problem.goal_const.desired_angles
        dist = np.linalg.norm(q_end_target - q_end_init)
        n_call = int(dist * 1000) + 1
        if self.config.random:
            if np.random.rand() < self.config.random_force_failure_rate:
                n_call = self.config.n_max_call + 1  # meaning failure
            else:
                n_call = n_call + int(np.random.randint(n_call) * self.config.random_scale)
        traj: Optional[Trajectory]
        if n_call < self.config.n_max_call:
            traj = Trajectory.from_two_points(q_end_init, q_end_target, 2)
        else:
            traj = None
        return DummyResult(traj, None, n_call)


class DummyTaskBase(TaskBase[DummyWorldT, np.ndarray, None]):
    @classmethod
    def from_task_params(cls: Type[DummyTaskT], desc_vecs: np.ndarray) -> DummyTaskT:
        world = cls.get_world_type().sample(True)
        # split desc_vecs by dof of this task get_task_dof
        desc_vec_list = list(desc_vecs.reshape(-1, cls.get_task_dof()))
        return cls(world, desc_vec_list)

    @staticmethod
    def get_robot_model() -> None:
        return None

    def export_table(self, use_matrix: bool) -> TaskExpression:
        # don't depend on use_matrix
        return TaskExpression(None, None, self.descriptions)

    def solve_default_each(self, problem: Problem) -> DummyResult:
        x0 = problem.start
        assert isinstance(problem.goal_const, ConfigPointConst)
        x1 = problem.goal_const.desired_angles

        if problem.global_ineq_const is not None:
            if not problem.global_ineq_const.is_valid(x1):
                return DummyResult.abnormal()

        assert np.linalg.norm(x1 - x0) < 1e-3
        dummy_traj = Trajectory.from_two_points(x0, x1, 2)
        return DummyResult(dummy_traj, 1.0, 3000)  # whatever

    def export_problems(self) -> List[Problem]:
        box = BoxConst(self.world.b_min, self.world.b_max)
        sdf = self.world.get_exact_sdf()
        # create easy simple dummy problem
        probs = []
        for desc in self.descriptions:
            goal_const = ConfigPointConst(desc)
            prob = Problem(
                desc,
                box,
                goal_const,
                PointCollFreeConst(sdf),
                None,
                skip_init_feasibility_check=True,
            )
            probs.append(prob)
        return probs

    @classmethod
    def get_task_dof(cls) -> int:
        return 2

    def visualize(self) -> Tuple:
        fig, ax = plt.subplots()
        self.world.visualize((fig, ax))

        for goal in self.descriptions:
            ax.scatter(goal[0], goal[1], label="point", c="red")
        return fig, ax


class DummyTask(DummyTaskBase[DummyWorld]):
    @staticmethod
    def get_world_type() -> Type[DummyWorld]:
        return DummyWorld

    @classmethod
    def sample_descriptions(
        cls, world: DummyWorld, n_sample: int, standard: bool = False
    ) -> List[np.ndarray]:

        sdf = world.get_exact_sdf()

        if standard:
            assert n_sample == 1
            return [np.zeros(2)]
        else:
            descs: List[np.ndarray] = []
            while len(descs) < n_sample:
                p = np.random.rand(2) * (world.b_max - world.b_min) + world.b_min
                if sdf(np.expand_dims(p, axis=0))[0] > 0:
                    descs.append(p)
            return descs


class DummyMeshTask(DummyTask):
    def export_table(self, use_matrix: bool) -> TaskExpression:
        if use_matrix:
            image = np.zeros((56, 56))
        else:
            image = None
        return TaskExpression(None, image, self.descriptions)


class ProbDummyTask(DummyTaskBase[ProbDummyWorld]):
    @staticmethod
    def get_world_type() -> Type[ProbDummyWorld]:
        return ProbDummyWorld

    @classmethod
    def sample_descriptions(
        cls, world: ProbDummyWorld, n_sample: int, standard: bool = False
    ) -> List[np.ndarray]:

        world.get_exact_sdf()

        if standard:
            assert n_sample == 1
            return [np.zeros(2)]
        else:
            descs: List[np.ndarray] = []
            while len(descs) < n_sample:
                p = world.kde.resample(1).flatten()
                descs.append(p)
            return descs
