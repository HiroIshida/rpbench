import time
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Type, TypeVar, Union

import disbmp
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
from disbmp import BoundingBox, FastMarchingTree, State
from scipy.interpolate import RegularGridInterpolator
from skmp.solver.interface import (
    AbstractDataDrivenSolver,
    AbstractScratchSolver,
    Problem,
    ResultProtocol,
)
from skmp.solver.nlp_solver.osqp_sqp import Differentiable, OsqpSqpConfig, OsqpSqpSolver

import rpbench.two_dimensional.double_integrator_trajopt as diopt
from rpbench.interface import DescriptionTable, SDFProtocol, TaskBase, WorldBase
from rpbench.two_dimensional.double_integrator_trajopt import (
    TrajectoryBound,
    TrajectoryCostFunction,
    TrajectoryDifferentialConstraint,
    TrajectoryEndPointConstraint,
    TrajectoryObstacleAvoidanceConstraint,
)
from rpbench.two_dimensional.utils import Grid2d, Grid2dSDF
from rpbench.utils import temp_seed


@dataclass
class DoubleIntegratorPlanningProblem:
    start: np.ndarray
    goal: np.ndarray
    sdf: SDFProtocol
    tbound: TrajectoryBound
    dt: float

    def check_init_feasibility(self) -> bool:
        return True, "always_ok"


@dataclass
class DoubleIntegratorPlanningConfig:
    n_wp: int
    n_max_call: int
    timeout: Optional[int] = None


@dataclass
class DoubleIntegratorPlanningResult:
    traj: Optional[diopt.Trajectory]
    time_elapsed: Optional[float]
    n_call: int

    @classmethod
    def abnormal(cls) -> "DoubleIntegratorPlanningResult":
        return cls(None, None, -1)


@dataclass
class DoubleIntegratorOptimizationSolver(
    AbstractScratchSolver[DoubleIntegratorPlanningConfig, DoubleIntegratorPlanningResult]
):
    config: DoubleIntegratorPlanningConfig
    problem: Optional[DoubleIntegratorPlanningProblem]
    traj_conf: Optional[diopt.TrajectoryConfig]
    osqp_solver: Optional[OsqpSqpSolver]

    def get_result_type(self) -> DoubleIntegratorPlanningResult:
        return DoubleIntegratorPlanningResult

    @classmethod
    def init(cls, conf: DoubleIntegratorPlanningConfig) -> "DoubleIntegratorOptimizationSolver":
        return cls(conf, None, None, None)

    def _setup(self, problem: DoubleIntegratorPlanningProblem) -> None:
        # setup optimization problem by direct transcription
        traj_conf = diopt.TrajectoryConfig(2, self.config.n_wp, problem.dt)

        eq_consts: List[Differentiable] = []
        diff_const = TrajectoryDifferentialConstraint(traj_conf)
        eq_consts.append(diff_const)

        end_const = TrajectoryEndPointConstraint(traj_conf, problem.start, problem.goal)
        eq_consts.append(end_const)

        def eq_const(traj: np.ndarray) -> Tuple[np.ndarray, sparse.csc_matrix]:
            vals = []
            jacs = []
            for c in eq_consts:
                val, jac = c(traj)
                vals.append(val)
                jacs.append(jac)
            val = np.hstack(vals)
            jac = sparse.vstack(jacs)
            return val, jac

        ineq_const = TrajectoryObstacleAvoidanceConstraint(traj_conf, problem.sdf)

        cost_fun = TrajectoryCostFunction(traj_conf)
        lb = problem.tbound.lower_bound(traj_conf.n_steps)
        ub = problem.tbound.upper_bound(traj_conf.n_steps)

        osqp_solver = OsqpSqpSolver(cost_fun.cost_matrix, eq_const, ineq_const, lb, ub)
        self.traj_conf = traj_conf
        self.osqp_solver = osqp_solver

    def _solve(self, guiding_traj: Optional[disbmp.Trajectory]) -> DoubleIntegratorPlanningResult:
        assert self.problem is not None
        assert self.traj_conf is not None
        assert self.osqp_solver is not None

        if guiding_traj is None:
            traj_guess = diopt.Trajectory.from_two_points(
                self.problem.start, self.problem.goal, self.traj_conf
            )
        else:
            times = np.linspace(0, guiding_traj.get_duration(), self.config.n_wp)
            states = np.array([guiding_traj.interpolate(t) for t in times])
            traj_guess = diopt.Trajectory.from_X_and_V(states[:, :2], states[:, 2:], self.traj_conf)

        osqp_conf = OsqpSqpConfig(n_max_eval=self.config.n_max_call)
        ret = self.osqp_solver.solve(traj_guess.to_array(), osqp_conf)
        if ret.success:
            traj = diopt.Trajectory.from_array(ret.x, self.traj_conf)
            return DoubleIntegratorPlanningResult(traj, None, ret.nit)
        else:
            return DoubleIntegratorPlanningResult(None, None, ret.nit)


@dataclass
class DoubleIntegratorOptimizationDataDrivenSolver(
    AbstractDataDrivenSolver[DoubleIntegratorPlanningConfig, DoubleIntegratorPlanningResult]
):
    config: DoubleIntegratorPlanningConfig
    internal_solver: DoubleIntegratorOptimizationSolver
    vec_descs: np.ndarray
    trajectories: List[disbmp.Trajectory]

    @classmethod
    def init(
        cls,
        config: DoubleIntegratorPlanningConfig,
        dataset: List[Tuple[np.ndarray, disbmp.Trajectory]],
    ) -> "DoubleIntegratorOptimizationDataDrivenSolver":
        vec_descs = np.array([p[0] for p in dataset])
        trajectories = [p[1] for p in dataset]
        internal_solver = DoubleIntegratorOptimizationSolver.init(config)
        return cls(config, internal_solver, vec_descs, trajectories)

    def _solve(self, query_desc: Optional[np.ndarray] = None) -> DoubleIntegratorPlanningResult:
        if query_desc is not None:
            sqdists = np.sum((self.vec_descs - query_desc) ** 2, axis=1)
            idx_closest = np.argmin(sqdists)
            reuse_traj = self.trajectories[idx_closest]
        else:
            reuse_traj = None
        result = self.internal_solver._solve(reuse_traj)
        return result

    @classmethod
    def get_result_type(cls) -> Type[DoubleIntegratorPlanningResult]:
        return DoubleIntegratorPlanningResult

    def _setup(self, problem: Problem) -> None:
        self.internal_solver.setup(problem)


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

        if n_obs == 0:
            dummy_circle = CircleObstacle(np.ones(2) * 0.5, 0.0)
            return cls([dummy_circle])

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
        # return Grid2d(np.zeros(2), np.ones(2), (112, 112))
        return Grid2d(np.zeros(2), np.ones(2), (56, 56))

    def get_grid_map(self) -> np.ndarray:
        grid = self.get_grid()

        xlin, ylin = [np.linspace(grid.lb[i], grid.ub[i], grid.sizes[i]) for i in range(2)]
        X, Y = np.meshgrid(xlin, ylin)
        pts = np.array(list(zip(X.flatten(), Y.flatten())))

        sdf = self.get_exact_sdf()
        vals = sdf.__call__(pts)
        grid_map = vals.reshape(grid.sizes).T
        return grid_map

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
class BubblyWorldEmpty(BubblyWorldBase):
    @classmethod
    def get_meta_parameter(cls) -> BubblyMetaParameter:
        return BubblyMetaParameter(0, np.inf, np.inf)


@dataclass
class BubblyWorldSimple(BubblyWorldBase):
    @classmethod
    def get_meta_parameter(cls) -> BubblyMetaParameter:
        return BubblyMetaParameter(10, 0.05, 0.2)


@dataclass
class BubblyWorldModerate(BubblyWorldBase):
    @classmethod
    def get_meta_parameter(cls) -> BubblyMetaParameter:
        return BubblyMetaParameter(20, 0.04, 0.16)


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
        if self.cache is None:
            wd["world"] = self.world.export_intrinsic_description()
        else:
            wd["world"] = self.cache.values.reshape(self.cache.grid.sizes)

        wcd_list = []
        for desc in self.descriptions:
            wcd = {}
            wcd["start"] = desc[0]
            wcd["goal"] = desc[1]
            wcd_list.append(wcd)
        return DescriptionTable(wd, wcd_list)

    @dataclass
    class _FMTResult:
        traj: Optional[disbmp.Trajectory]
        time_elapsed: float
        n_call: int

    def solve_default_each(self, problem: DoubleIntegratorPlanningProblem) -> ResultProtocol:
        s_min = np.hstack([problem.tbound.x_min, problem.tbound.v_min])
        s_max = np.hstack([problem.tbound.x_max, problem.tbound.v_max])
        bbox = BoundingBox(s_min, s_max)
        s_start = State(np.hstack([problem.start, np.zeros(2)]))
        s_goal = State(np.hstack([problem.goal, np.zeros(2)]))

        def is_obstacle_free(state: State) -> bool:
            x = state.to_vector()[:2]
            return problem.sdf(np.expand_dims(x, axis=0))[0] > 0.0

        N = 3000
        ts = time.time()
        fmt = FastMarchingTree(s_start, s_goal, is_obstacle_free, bbox, problem.dt, 1.0, N)
        is_solved = fmt.solve(N)
        time_elapsed = time.time() - ts
        if is_solved:
            traj = fmt.get_solution()
            return self._FMTResult(traj, time_elapsed, 0)  # 0 is dummy
        else:
            return self._FMTResult(None, time_elapsed, 0)  # 0 is dummy

    @classmethod
    def get_dof(cls) -> int:
        return 2

    def export_problems(self) -> List[Problem]:
        if self.cache is None:
            self.cache = self.create_cache(self.world, None)
        sdf = self.cache

        tbound = TrajectoryBound(
            np.ones(2) * 0.0,
            np.ones(2) * 1.0,
            np.ones(2) * -0.3,
            np.ones(2) * 0.3,
            np.ones(2) * -0.1,
            np.ones(2) * 0.1,
        )

        probs = []
        for desc in self.descriptions:
            start, goal = desc
            problem = DoubleIntegratorPlanningProblem(start, goal, sdf, tbound, 0.2)
            probs.append(problem)
        return probs

    def export_intrinsic_descriptions(self) -> List[np.ndarray]:
        # return [self.world.export_intrinsic_description()] * self.n_inner_task
        return [np.hstack(desc) for desc in self.descriptions] * self.n_inner_task

    @staticmethod
    def create_cache(world: BubblyWorldBase, robot_model: None) -> Grid2dSDF:
        # TODO: redundant implementation with world.get_grid_map()
        grid = world.get_grid()
        xlin, ylin = [np.linspace(grid.lb[i], grid.ub[i], grid.sizes[i]) for i in range(2)]
        grid_map = world.get_grid_map()
        itp = RegularGridInterpolator(
            (xlin, ylin), grid_map, bounds_error=False, fill_value=10.0, method="cubic"
        )
        vals = grid_map.flatten()
        return Grid2dSDF(vals, world.get_grid(), itp)


class EmptyWorldProviderMixin:
    @staticmethod
    def get_world_type() -> Type[BubblyWorldEmpty]:
        return BubblyWorldEmpty


class SimpleWorldProviderMixin:
    @staticmethod
    def get_world_type() -> Type[BubblyWorldSimple]:
        return BubblyWorldSimple


class ModerateWorldProviderMixin:
    @staticmethod
    def get_world_type() -> Type[BubblyWorldModerate]:
        return BubblyWorldModerate


class ComplexWorldProviderMixin:
    @staticmethod
    def get_world_type() -> Type[BubblyWorldComplex]:
        return BubblyWorldComplex


class BubblyEmptyMeshPointConnectTask(
    EmptyWorldProviderMixin, BubblyPointConnectTaskBase[BubblyWorldEmpty]
):
    ...


class BubblySimpleMeshPointConnectTask(
    SimpleWorldProviderMixin, BubblyPointConnectTaskBase[BubblyWorldSimple]
):
    ...


class BubblyComplexMeshPointConnectTask(
    ComplexWorldProviderMixin, BubblyPointConnectTaskBase[BubblyWorldComplex]
):
    ...


class BubblyModerateMeshPointConnectTask(
    ModerateWorldProviderMixin, BubblyPointConnectTaskBase[BubblyWorldModerate]
):
    ...


class Taskvisualizer:
    fax: Tuple

    def __init__(self, task: BubblyPointConnectTaskBase):
        fig, ax = plt.subplots()
        task.world.visualize((fig, ax))
        for start, goal in task.descriptions:
            ax.plot(start[0], start[1], "mo", markersize=10, label="start")
            ax.plot(goal[0], goal[1], "m*", markersize=10, label="goal")
        ax.set_xlim([-0.1, 1.1])
        ax.set_ylim([-0.1, 1.1])
        ax.plot([0, 0, 1, 1, 0], [0, 1, 1, 0, 0], c="k")
        self.fax = (fig, ax)

    def visualize_trajectories(
        self, trajs: Union[List[diopt.Trajectory], diopt.Trajectory], **kwargs
    ) -> None:
        fig, ax = self.fax
        if isinstance(trajs, diopt.Trajectory):
            trajs = [trajs]
        for traj in trajs:
            ax.plot(traj.X[:, 0], traj.X[:, 1], **kwargs)
            ax.plot(traj.X[-1, 0], traj.X[-1, 1], "o", color=kwargs["color"], markersize=2)
