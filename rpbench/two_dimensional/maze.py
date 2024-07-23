from dataclasses import dataclass
from typing import Optional, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
from ompl import Algorithm, ERTConnectPlanner, Planner
from skmp.solver.interface import AbstractScratchSolver, ResultProtocol
from skmp.trajectory import Trajectory

from rpbench.interface import TaskBase, TaskExpression, TaskExpressionProtocol
from rpbench.utils import temp_seed


def random_maze(
    width: int = 80, height: int = 80, complexity: float = 0.9, density: float = 0.5
) -> np.ndarray:
    shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
    complexity = int(complexity * (5 * (shape[0] + shape[1])))
    density = int(density * ((shape[0] // 2) * (shape[1] // 2)))

    Z = np.zeros(shape, dtype=bool)
    Z[0, :] = Z[-1, :] = 1
    Z[:, 0] = Z[:, -1] = 1

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

    return Z.astype(int)


maze_size = 80
with temp_seed(0, True):
    _maze = random_maze(width=maze_size, height=maze_size)
block_size = 1.0 / (maze_size + 1)
x_start = np.array([1.5, 1.5]) * block_size


def coord_to_index_vectorized(coords):
    indices = coords * (maze_size + 1)
    indices = indices.astype(int)
    return indices


def is_collide_vectorized(X):
    indices = coord_to_index_vectorized(X)
    return _maze[indices[:, 1], indices[:, 0]] == 1


@dataclass
class MazeProblem:
    goal: np.ndarray

    def check_init_feasibility(self) -> Tuple[bool, str]:
        return True, "always_ok"


@dataclass
class MazeSolverResult:
    traj: Optional[Trajectory]
    time_elapsed: Optional[float]
    n_call: int

    @classmethod
    def abnormal(cls) -> "MazeSolverResult":
        return cls(None, None, -1)


@dataclass
class MazeSolverConfig:
    n_max_call: int
    timeout: Optional[float] = None


@dataclass
class MazeSolver(AbstractScratchSolver[MazeSolverConfig, MazeSolverResult]):
    config: MazeSolverConfig
    x_goal: Optional[np.ndarray] = None

    @classmethod
    def init(cls, config: MazeSolverConfig) -> "MazeSolver":
        return cls(config, None)

    def get_result_type(self) -> Type[MazeSolverResult]:
        return MazeSolverResult

    def _setup(self, problem: MazeProblem):
        self.x_goal = problem.goal

    def _solve(self, guiding_traj: Optional[Trajectory] = None) -> MazeSolverResult:
        n_call = 0

        def is_valid(x):
            nonlocal n_call
            n_call += 1
            x = np.array(x)
            return not is_collide_vectorized(x.reshape(1, 2))

        assert self.x_goal is not None
        if guiding_traj is None:
            planner = Planner(
                np.zeros(2),
                np.ones(2),
                is_valid,
                self.config.n_max_call,
                0.003,
                Algorithm.RRTConnect,
            )
            ret = planner.solve(x_start, self.x_goal, simplify=False)
        else:
            planner = ERTConnectPlanner(
                np.zeros(2), np.ones(2), is_valid, self.config.n_max_call, 0.003
            )
            planner.set_heuristic(guiding_traj.numpy())
            # planner.set_parameters(eps=0.1)
            planner.set_parameters(eps=5.0)
            ret = planner.solve(x_start, self.x_goal, simplify=True)
        if ret is None:
            return MazeSolverResult.abnormal()
        return MazeSolverResult(Trajectory(ret), None, n_call)


@dataclass
class MazeTask(TaskBase):
    x_goal: np.ndarray

    @classmethod
    def from_task_param(cls, param: np.ndarray) -> "MazeTask":
        return cls(param)

    @classmethod
    def sample(
        cls,
        predicate=None,
        timeout: int = 180,
    ) -> "MazeTask":
        while True:
            x_goal = np.random.rand(2)
            if not is_collide_vectorized(x_goal.reshape(1, 2)):
                break
        return cls(x_goal)

    def export_task_expression(self, use_matrix: bool) -> TaskExpressionProtocol:
        return TaskExpression(None, None, self.x_goal)

    def solve_default(self) -> ResultProtocol:
        x_start = np.array([1.5, 1.5]) * block_size

        def is_valid(x):
            x = np.array(x)
            return not is_collide_vectorized(x.reshape(1, 2))

        solver = MazeSolver.init(MazeSolverConfig(3000000))
        solver.setup(MazeProblem(self.x_goal))
        tmp = solver.solve()
        assert tmp.traj is not None
        tmp2 = solver.solve(tmp.traj)
        return tmp2

    def export_problem(self) -> MazeProblem:
        return MazeProblem(self.x_goal)


if __name__ == "__main__":
    n = 1000
    eps = 1e-6
    xlin = np.linspace(0, 1 - eps, n)
    ylin = np.linspace(0, 1 - eps, n)
    X, Y = np.meshgrid(xlin, ylin)
    pts = np.vstack([X.ravel(), Y.ravel()]).T
    collide = is_collide_vectorized(pts)
    Z = collide.reshape(n, n)
    fig, ax = plt.subplots()
    ax.imshow(~Z, extent=(0, 1, 0, 1), origin="lower", cmap="gray")
    ax.scatter(*x_start, color="red")

    np.random.seed(0)
    task = MazeTask.sample()
    re = task.solve_default()
    path1 = re.traj.numpy()
    problem = task.export_problem()
    solver = MazeSolver.init(MazeSolverConfig(3000000))
    solver.setup(problem)
    re2 = solver.solve(re.traj)
    path2 = re2.traj.numpy()

    # plot goal and path
    ax.scatter(*task.x_goal, color="green")
    ax.plot(path1[:, 0], path1[:, 1], color="blue")
    ax.plot(path2[:, 0], path2[:, 1], color="orange")
    plt.show()

    # # sample goal any free cell
    # while True:
    #     x_goal = np.random.rand(2)
    #     if not is_collide_vectorized(x_goal.reshape(1, 2)):
    #         break
    # ax.scatter(*x_goal, color='green')

    # global n_call
    # n_call = 0
    # def is_valid(x):
    #     x = np.array(x)
    #     global n_call
    #     n_call += 1
    #     return not is_collide_vectorized(x.reshape(1, 2))

    # planner = Planner(np.zeros(2), np.ones(2), is_valid, 3000000, 0.003, Algorithm.RRTConnect)
    # ret = planner.solve(x_start, x_goal, simplify=True)
    # print(n_call)
    # assert ret is not None
    # path = np.array(ret)
    # ax.plot(path[:, 0], path[:, 1], color='blue')

    # plt.show()
