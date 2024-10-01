from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Type

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numba
import numpy as np
from ompl import ERTConnectPlanner, Planner
from skmp.solver.interface import AbstractSolver, ConfigProtocol, ResultProtocol
from skmp.trajectory import Trajectory

from rpbench.interface import TaskBase, TaskExpression


@dataclass
class PureRRTCResult(ResultProtocol):
    traj: Optional[Trajectory]
    time_elapsed: Optional[float]
    n_call: int

    @classmethod
    def abnormal(cls) -> "PureRRTCResult":
        return PureRRTCResult(traj=None, time_elapsed=None, n_call=-1)


@dataclass
class PureRRTCConfig(ConfigProtocol):
    n_max_call: int = 300000
    timeout: Optional[float] = None


@dataclass
class PureProblem:
    start: np.ndarray
    goal: np.ndarray
    is_valid: Callable[Tuple[float, float], bool]

    def check_init_feasibility(self):
        return True, "dummy"


@dataclass
class PureRRTC(AbstractSolver[PureRRTCConfig, PureRRTCResult, Trajectory]):
    config: PureRRTCConfig
    problem: Optional[PureProblem]
    rrtc: Optional[Planner]
    ertc: Optional[ERTConnectPlanner]
    _n_call: int

    @classmethod
    def init(cls, config: PureRRTCConfig) -> "PureRRTC":
        return cls(config, None, None, None, 0)

    def _setup(self, problem: PureProblem) -> None:
        self.problem = problem

        def is_valid(p: np.ndarray) -> bool:
            self._n_call += 1
            return problem.is_valid(p[0], p[1])

        self.rrtc = Planner([0, 0], [1, 1], is_valid, self.config.n_max_call, [0.001, 0.001])
        self.ertc = ERTConnectPlanner(
            [0, 0], [1, 1], is_valid, self.config.n_max_call, [0.001, 0.001]
        )
        self.ertc.set_parameters(eps=0.1)

    def _solve(self, guiding_traj: Optional[Trajectory]) -> PureRRTCResult:
        if guiding_traj is None:
            ret = self.rrtc.solve(self.problem.start, self.problem.goal)
        else:
            self.ertc.set_heuristic(guiding_traj.numpy())
            ret = self.ertc.solve(self.problem.start, self.problem.goal, guiding_traj)
        if ret is None:
            self._n_call = 0
            return PureRRTCResult.abnormal()
        res = PureRRTCResult(Trajectory(ret), None, self._n_call)
        self._n_call = 0
        return res

    def get_result_type(self) -> Type[PureRRTCResult]:
        return PureRRTCResult


class ParametricMaze:
    wall_thickness = 0.03
    holl_width = 0.03

    def __init__(self, param: np.ndarray):
        self.n = len(param)
        self.wall_ys = np.linspace(1 / (self.n + 1), self.n / (self.n + 1), self.n)
        self.holl_xs = np.array(param)

        half_wall_thickness = self.wall_thickness * 0.5
        half_holl_width = self.holl_width * 0.5
        self.wall_y_mins = self.wall_ys - half_wall_thickness
        self.wall_y_maxs = self.wall_ys + half_wall_thickness
        self.holl_x_mins = self.holl_xs - half_holl_width
        self.holl_x_maxs = self.holl_xs + half_holl_width

    @classmethod
    def sample(cls, n_dim: int) -> "ParametricMaze":
        x_min = 0.5 * cls.holl_width
        x_max = 1 - 0.5 * cls.holl_width
        param = np.random.uniform(x_min, x_max, n_dim)
        return cls(param)

    def is_collision(self, x_point: numba.float64, y_point: numba.float64) -> numba.boolean:
        if x_point < 0 or x_point > 1 or y_point < 0 or y_point > 1:
            return True

        for i in range(self.n):
            if self.wall_y_mins[i] <= y_point <= self.wall_y_maxs[i]:
                if x_point < self.holl_x_mins[i] or x_point > self.holl_x_maxs[i]:
                    return True
        return False

    def visualize(self, traj: Optional[Trajectory] = None) -> None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for i in range(self.n):
            wall_y = self.wall_ys[i]
            wall_y_min = wall_y - self.wall_thickness / 2
            holl_x_center = self.holl_xs[i]
            holl_x_min = holl_x_center - self.holl_width / 2
            holl_x_max = holl_x_center + self.holl_width / 2
            holl_x_min = max(holl_x_min, 0)
            holl_x_max = min(holl_x_max, 1)
            if holl_x_min > 0:
                left_wall = patches.Rectangle(
                    (0, wall_y_min), holl_x_min, self.wall_thickness, color="black"
                )
                ax.add_patch(left_wall)
            if holl_x_max < 1:
                right_wall = patches.Rectangle(
                    (holl_x_max, wall_y_min), 1 - holl_x_max, self.wall_thickness, color="black"
                )
                ax.add_patch(right_wall)
        boundary = patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor="black", linewidth=2)

        if traj is not None:
            traj = traj.numpy()
            ax.plot(traj[:, 0], traj[:, 1], color="red", linewidth=2)

        ax.add_patch(boundary)
        ax.set_aspect("equal")
        plt.title("Parametric Maze Visualization")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.grid(True)
        plt.show()


@dataclass
class ParametricMazeTaskBase(TaskBase):
    world: ParametricMaze

    @classmethod
    def from_task_param(cls, param: np.ndarray) -> "ParametricMazeTask":
        return cls(ParametricMaze(param))

    @classmethod
    def sample(
        cls,
        predicate: Optional[Callable] = None,
        timeout: int = 180,
    ) -> "ParametricMazeTaskBase":
        task = cls(ParametricMaze.sample(8))
        return task

    def sample_description(self) -> None:
        return None

    def export_task_expression(self, use_matrix: bool) -> TaskExpression:
        return TaskExpression(self.world.holl_xs, None, np.empty(0))

    def solve_default(self) -> PureRRTCResult:
        conf = PureRRTCConfig(n_max_call=500000)
        solver = PureRRTC.init(conf)
        solver.setup(self.export_problem())
        return solver.solve()

    def export_problem(self) -> PureProblem:
        is_valid = lambda x, y: not self.world.is_collision(x, y)
        return PureProblem(np.array([0.02, 0.02]), np.array([0.98, 0.98]), is_valid)


if __name__ == "__main__":
    task = ParametricMazeTaskBase.sample()
    r = task.solve_default()
    task.world.visualize(r.traj)
