import time
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, ClassVar, List, Optional, Tuple, Type, TypeVar, Union

import disbmp
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
from disbmp import RRT, BoundingBox, FastMarchingTree, State
from skmp.solver.interface import AbstractScratchSolver, Problem, ResultProtocol
from skmp.solver.nlp_solver.osqp_sqp import Differentiable, OsqpSqpConfig, OsqpSqpSolver

import rpbench.two_dimensional.double_integrator_trajopt as diopt
from rpbench.interface import (
    SamplableWorldBase,
    SDFProtocol,
    TaskBase,
    TaskExpression,
    TaskWithWorldCondBase,
)
from rpbench.two_dimensional.boxlib import (
    ParametricCircles,
    ParametricMaze,
    ParametricMazeSpecial,
)
from rpbench.two_dimensional.double_integrator_trajopt import (
    TrajectoryBound,
    TrajectoryCostFunction,
    TrajectoryDifferentialConstraint,
    TrajectoryEndPointConstraint,
    TrajectoryObstacleAvoidanceConstraint,
)
from rpbench.two_dimensional.utils import Grid2d
from rpbench.utils import temp_seed

# NOTE: as double-integrator planning is quite different from other tasks
# we violate the task and other protocols here. So there is not type checking
# will be performed for this file.


@dataclass
class DoubleIntegratorPlanningProblem:
    start: np.ndarray
    goal: np.ndarray
    sdf: SDFProtocol
    tbound: TrajectoryBound
    dt: float

    def check_init_feasibility(self) -> Tuple[bool, str]:
        return True, "always_ok"


@dataclass
class DoubleIntegratorPlanningConfig:
    n_wp: int
    n_max_call: int
    timeout: Optional[float] = None
    step_size_init: float = 1.0
    step_size_step: float = 0.0
    only_closest: bool = False


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

    def get_result_type(self) -> Type[DoubleIntegratorPlanningResult]:
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

        ineq_const = TrajectoryObstacleAvoidanceConstraint(
            traj_conf, problem.sdf, only_closest=self.config.only_closest
        )

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

        osqp_conf = OsqpSqpConfig(
            n_max_eval=self.config.n_max_call,
            step_size_init=self.config.step_size_init,
            step_size_step=self.config.step_size_step,
            verbose=False,
        )
        ret = self.osqp_solver.solve(traj_guess.to_array(), osqp_conf)
        if ret.success:
            X = ret.x.reshape(-1, self.traj_conf.n_dim)
            self.problem.sdf(X)
            traj = diopt.Trajectory.from_array(ret.x, self.traj_conf)
            return DoubleIntegratorPlanningResult(traj, None, ret.nit)
        else:
            return DoubleIntegratorPlanningResult(None, None, ret.nit)


@dataclass
class DoubleIntegratorRRTConfig:
    dt: float = 1.0
    n_max_call: int = 100000
    timeout: Optional[float] = None


@dataclass
class DoubleIntegratorRRTResult:
    traj: Optional[disbmp.Trajectory]
    time_elapsed: Optional[float]
    n_call: int

    @classmethod
    def abnormal(cls) -> "DoubleIntegratorPlanningResult":
        return cls(None, None, -1)


@dataclass
class DoubleIntegratorRRTSolver(
    AbstractScratchSolver[DoubleIntegratorRRTConfig, DoubleIntegratorRRTResult]
):
    config: DoubleIntegratorRRTConfig
    problem: Optional[DoubleIntegratorPlanningProblem]
    rrt: Optional[RRT]
    _n_call: Optional[int]

    def get_result_type(self) -> Type[DoubleIntegratorPlanningResult]:
        return DoubleIntegratorPlanningResult

    @classmethod
    def init(cls, conf: DoubleIntegratorRRTConfig) -> "DoubleIntegratorRRTSolver":
        return cls(conf, None, None, None)

    def _setup(self, problem: DoubleIntegratorPlanningProblem) -> None:
        s_min = np.hstack([problem.tbound.x_min, problem.tbound.v_min])
        s_max = np.hstack([problem.tbound.x_max, problem.tbound.v_max])
        bbox = BoundingBox(s_min, s_max)
        s_start = State(np.hstack([problem.start, np.zeros(2)]))
        s_goal = State(np.hstack([problem.goal, np.zeros(2)]))

        self._n_call = 0

        def is_obstacle_free(state: State) -> bool:
            self._n_call += 1
            x = state.to_vector()[:2]
            return problem.sdf(np.expand_dims(x, axis=0))[0] > 0.0

        assert (
            self.config.dt > problem.dt
        )  # trajectory piece dt must be larger than collision check dt
        rrt = RRT(s_start, s_goal, is_obstacle_free, bbox, self.config.dt, problem.dt)
        self.rrt = rrt

    def _solve(self, guiding_traj: Optional[disbmp.Trajectory]) -> DoubleIntegratorPlanningResult:
        assert guiding_traj is None
        assert self.problem is not None
        assert self.rrt is not None
        is_solved = self.rrt.solve(self.config.n_max_call)
        assert self._n_call is not None
        n_call = self._n_call
        self._n_call = None
        if is_solved:
            traj = self.rrt.get_solution()
            return DoubleIntegratorPlanningResult(traj, None, n_call)
        else:
            return DoubleIntegratorPlanningResult.abnormal()


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
class BubblyWorldBase(SamplableWorldBase):
    obstacles: List[CircleObstacle]

    @classmethod
    def from_desc(cls: Type[BubblyWorldT], desc: np.ndarray) -> BubblyWorldT:
        n_obs = len(desc) // 3
        assert len(desc) % 3 == 0, f"Invalid description length: {len(desc)}"
        obstacles = []
        for i in range(n_obs):
            desc_this = desc[i * 3 : (i + 1) * 3]
            center = desc_this[:2]
            radius = desc_this[2]
            obstacles.append(CircleObstacle(center, radius))
        return cls(obstacles)

    @classmethod
    @abstractmethod
    def get_meta_parameter(cls) -> BubblyMetaParameter:
        ...

    @classmethod
    def get_margin(cls) -> float:
        return 0.01

    @classmethod
    def sample(cls: Type[BubblyWorldT]) -> BubblyWorldT:
        standard = False
        meta_param = cls.get_meta_parameter()
        n_obs = meta_param.n_obs
        r_min = meta_param.circle_r_min
        r_max = meta_param.circle_r_max

        if n_obs == 0:
            dummy_circle = CircleObstacle(np.ones(2) * 1.0, 0.0)
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

    @classmethod
    def get_world_dof(cls) -> int:
        return 3 * cls.get_meta_parameter().n_obs

    def export_intrinsic_description(self) -> np.ndarray:
        return np.hstack([obs.as_vector() for obs in self.obstacles])

    def grid(self) -> Grid2d:
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


class BubblyPointConnectTaskBase(TaskWithWorldCondBase[BubblyWorldT, np.ndarray, None]):
    @classmethod
    def from_task_param(cls, param: np.ndarray) -> "BubblyPointConnectTaskBase":
        assert param.ndim == 1
        world_type = cls.get_world_type()
        world_dof = world_type.get_world_dof()
        world_desc = param[:world_dof]
        world = world_type.from_desc(world_desc)
        goal = param[world_dof:]
        return cls(world, goal)

    @classmethod
    def get_robot_model(cls) -> None:
        return None

    @classmethod
    def sample_description(
        cls, world: BubblyWorldT, standard: bool = False
    ) -> Optional[np.ndarray]:
        start = np.ones(2) * 0.1
        if standard:
            goal = np.ones(2) * 0.95
            return goal
        else:
            sdf = world.get_exact_sdf()
            goal = np.random.rand(2)
            if np.linalg.norm(goal - start) > 0.4:
                val = sdf(np.expand_dims(goal, axis=0))[0]
                if val > 0.0:
                    return goal
        return None

    def export_task_expression(self, use_matrix: bool) -> TaskExpression:
        if use_matrix:
            world_vec = None
            world_mat = self.world.get_grid_map()
        else:
            world_vec = self.world.export_intrinsic_description()
            world_mat = None
        return TaskExpression(world_vec, world_mat, self.description)

    @dataclass
    class _FMTResult:
        traj: Optional[disbmp.Trajectory]
        time_elapsed: float
        n_call: int

    def solve_default(self) -> ResultProtocol:
        problem = self.export_problem()
        s_min = np.hstack([problem.tbound.x_min, problem.tbound.v_min])
        s_max = np.hstack([problem.tbound.x_max, problem.tbound.v_max])
        bbox = BoundingBox(s_min, s_max)
        x_start = np.ones(2) * 0.1
        s_start = State(np.hstack([x_start, np.zeros(2)]))
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

    def export_problem(self) -> Problem:
        sdf = self.world.get_exact_sdf()

        tbound = TrajectoryBound(
            np.ones(2) * 0.0,
            np.ones(2) * 1.0,
            np.ones(2) * -0.3,
            np.ones(2) * 0.3,
            np.ones(2) * -0.1,
            np.ones(2) * 0.1,
        )

        x_start = np.ones(2) * 0.1
        goal = self.description
        problem = DoubleIntegratorPlanningProblem(x_start, goal, sdf, tbound, 0.2)
        return problem

    @classmethod
    def get_task_dof(cls) -> int:
        return cls.get_world_type().get_world_dof() + 2

    def create_viewer(self) -> "Taskvisualizer":
        return Taskvisualizer(self)


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
        goal = task.description
        start = np.ones(2) * 0.1
        ax.plot(start[0], start[1], "mo", markersize=10, label="start")
        ax.plot(goal[0], goal[1], "m*", markersize=10, label="goal")
        ax.set_xlim([-0.1, 1.1])
        ax.set_ylim([-0.1, 1.1])
        ax.plot([0, 0, 1, 1, 0], [0, 1, 1, 0, 0], c="k")
        self.fax = (fig, ax)

    def show(self) -> None:
        plt.show()

    def visualize_trajectories(
        self, trajs: Union[List[diopt.Trajectory], diopt.Trajectory], **kwargs
    ) -> None:
        fig, ax = self.fax
        if isinstance(trajs, diopt.Trajectory):
            trajs = [trajs]
        for traj in trajs:
            ax.plot(traj.X[:, 0], traj.X[:, 1], **kwargs)
            ax.plot(traj.X[-1, 0], traj.X[-1, 1], "o", color=kwargs["color"], markersize=2)


@dataclass
class ParametricMazeTaskBase(TaskBase):
    world: ParametricMaze
    dof: ClassVar[int]
    is_special: ClassVar[bool] = False
    is_circle: ClassVar[bool] = False

    @dataclass
    class _FMTResult:
        traj: Optional[disbmp.Trajectory]
        time_elapsed: float
        n_call: int

    @classmethod
    def from_task_param(cls, param: np.ndarray) -> "ParametricMazeTaskBase":
        # assert len(param) == cls.dof
        if cls.is_special:
            assert len(param) == 1
            return cls(ParametricMazeSpecial(param[0]))
        else:
            if cls.is_circle:
                return cls(ParametricCircles(param))
            else:
                return cls(ParametricMaze(param))

    def visualize(self, trajs, plot_world: bool = True, fax=None, **kwargs):
        if fax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = fax

        if plot_world:
            self.world.visualize((fig, ax))
            # ax.text(0.5, 1.05, 'maze map!', transform=ax.transAxes, ha='center', va='bottom', fontsize=12, fontweight='bold')
            y_pos = self.world.y_length + 0.04
            if not self.is_special:
                ax.text(
                    0.5, y_pos, r"$n_p$=" + str(self.dof), ha="center", va="bottom", fontsize=16
                )

        if trajs is not None:
            for traj in trajs:
                if isinstance(traj, diopt.Trajectory):
                    ax.plot(traj.X[:, 0], traj.X[:, 1], "ro-", markersize=2)
                    ax.plot(traj.X[-1, 0], traj.X[-1, 1], "o", color=kwargs["color"], markersize=2)
                else:
                    t_duration = traj.get_duration()
                    t_resolution = 0.01
                    for t in np.arange(0, t_duration, t_resolution):
                        traj.interpolate(t)
                    X = [traj.interpolate(t) for t in np.arange(0, t_duration, t_resolution)]
                    X = np.array(X)
                    ax.plot(X[:, 0], X[:, 1], **kwargs)
                    ax.plot(X[-1, 0], X[-1, 1], "o-", color=kwargs["color"], markersize=2)

        if plot_world:
            # plot start and goal
            ax.plot(0.05, 0.05, "mo", markersize=10, label="start")
            ax.plot(0.95, self.world.y_length - 0.05, "m*", markersize=10, label="goal")

    @classmethod
    def sample(
        cls,
        predicate: Optional[Callable] = None,
        timeout: int = 180,
    ) -> "ParametricMazeTaskBase":
        t_start = time.time()
        while True:
            t_elapsed = time.time() - t_start
            if t_elapsed > timeout:
                raise TimeoutError("predicated_sample: timeout!")
            if cls.is_special:
                task = cls(ParametricMazeSpecial.sample())
            else:
                if cls.is_circle:
                    task = cls(ParametricCircles.sample(cls.dof))
                else:
                    task = cls(ParametricMaze.sample(cls.dof))
            if predicate is None or predicate(task):
                return task

    def export_task_expression(self, use_matrix: bool) -> TaskExpression:
        other_vec = np.empty(0)
        return TaskExpression(self.world.param, None, other_vec)

    def solve_default(self) -> ResultProtocol:
        problem = self.export_problem()
        s_min = np.hstack([problem.tbound.x_min, problem.tbound.v_min])
        s_max = np.hstack([problem.tbound.x_max, problem.tbound.v_max])
        bbox = BoundingBox(s_min, s_max)
        s_start = State(np.hstack([problem.start, np.zeros(2)]))
        s_goal = State(np.hstack([problem.goal, np.zeros(2)]))

        def is_obstacle_free(state: State) -> bool:
            # with margin
            x = state.to_vector()[:2]
            margin = 0.01
            sdfs = problem.sdf(np.expand_dims(x, axis=0)) - margin
            return sdfs[0] > 0.0

        N = 5000
        print(f"start solving with N={N}")
        ts = time.time()
        fmt = FastMarchingTree(s_start, s_goal, is_obstacle_free, bbox, problem.dt, 1.0, N)
        is_solved = fmt.solve(N)
        time_elapsed = time.time() - ts
        if is_solved:
            traj = fmt.get_solution()
            return self._FMTResult(traj, time_elapsed, 0)  # 0 is dummy
        else:
            return self._FMTResult(None, time_elapsed, 0)  # 0 is dummy

    def export_problem(self) -> DoubleIntegratorPlanningProblem:
        margin = 0.01
        sdf: SDFProtocol = lambda x: self.world.signed_distance_batch(x[:, 0], x[:, 1]) - margin
        start = np.array([0.05, 0.05])
        goal = np.array([0.95, self.world.y_length - 0.05])
        tbound = TrajectoryBound(
            np.array([0.0, 0.0]),
            np.array([1.0, self.world.y_length]),
            np.array([-0.5, -0.5]),
            np.array([0.5, 0.5]),
            np.array([-0.05, -0.05]),
            np.array([0.05, 0.05]),
        )
        return DoubleIntegratorPlanningProblem(start, goal, sdf, tbound, 0.2)


class ParametricMazeTask1D(ParametricMazeTaskBase):
    dof: ClassVar[int] = 1


class ParametricMazeTask2D(ParametricMazeTaskBase):
    dof: ClassVar[int] = 2


class ParametricMazeTask3D(ParametricMazeTaskBase):
    dof: ClassVar[int] = 3


class ParametricMazeTask4D(ParametricMazeTaskBase):
    dof: ClassVar[int] = 4


class ParametricMazeTask5D(ParametricMazeTaskBase):
    dof: ClassVar[int] = 5


class ParametricCirclesTask1D(ParametricMazeTaskBase):
    dof: ClassVar[int] = 1
    is_circle: ClassVar[bool] = True


class ParametricCirclesTask2D(ParametricMazeTaskBase):
    dof: ClassVar[int] = 2
    is_circle: ClassVar[bool] = True


class ParametricCirclesTask3D(ParametricMazeTaskBase):
    dof: ClassVar[int] = 3
    is_circle: ClassVar[bool] = True


class ParametricCirclesTask4D(ParametricMazeTaskBase):
    dof: ClassVar[int] = 4
    is_circle: ClassVar[bool] = True


class ParametricMazeSpecialTask(ParametricMazeTaskBase):
    is_special: ClassVar[bool] = True
    dof: ClassVar[
        int
    ] = 4  # HACK: although this task is 1D, world length has same as the case of n_p = 4, which will be used to determine the waypoint number


if __name__ == "__main__":
    special = False
    is_circle = True
    if special:
        assert False
    else:
        if is_circle:
            task = ParametricCirclesTask4D.sample()
            task = ParametricCirclesTask4D.from_task_param(task.to_task_param())
        else:
            task = ParametricMazeTask2D.sample()
    result = task.solve_default()
    assert result.traj is not None

    n_point = 100 * 6
    solver_config = DoubleIntegratorPlanningConfig(n_point, 30, only_closest=True)
    solver = DoubleIntegratorOptimizationSolver.init(solver_config)
    solver.setup(task.export_problem())
    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()
    result2 = solver.solve(result.traj)
    profiler.stop()
    if result2.traj is None:
        task.visualize([result.traj], color="r")
    else:
        print("solved!")
        task.visualize([result.traj, result2.traj], color="r")
    plt.show()
