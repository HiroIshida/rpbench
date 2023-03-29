from abc import abstractmethod
from dataclasses import dataclass
from typing import ClassVar, List, Tuple, Type, TypeVar, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from skmp.constraint import BoxConst, ConfigPointConst, PointCollFreeConst
from skmp.solver.interface import Problem, ResultProtocol
from skmp.solver.nlp_solver.sqp_based_solver import SQPBasedSolver, SQPBasedSolverConfig
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig
from skmp.trajectory import Trajectory
from voxbloxpy.core import Grid

from rpbench.interface import DescriptionTable, SDFProtocol, TaskBase, WorldBase

RingWorldT = TypeVar("RingWorldT", bound="RingWorldBase")


@dataclass
class RingWorldBase(WorldBase):
    obstacles: List[np.ndarray]
    r_outer: ClassVar[float] = 0.5
    r_inner: ClassVar[float] = 0.4
    r_obstacle: ClassVar[float] = 0.01

    @classmethod
    def sample(cls: Type[RingWorldT], standard: bool = False) -> RingWorldT:
        return cls([])

    def get_margin(self) -> float:
        return self.r_obstacle * 0.1

    def get_exact_sdf(self) -> SDFProtocol:
        def ring_sdf(pts: np.ndarray):
            c = np.ones(2) * 0.5
            radii = np.sqrt(np.sum((pts - c) ** 2, axis=1))
            dists_to_outer = self.r_outer - radii
            dists_to_inner = radii - self.r_inner
            dists = np.minimum(dists_to_outer, dists_to_inner)
            return dists

        def f(pts: np.ndarray):
            wall_dists = ring_sdf(pts)
            if self.get_num_obstacle() > 0:
                dist_list = [
                    np.sqrt(np.sum((pts - c) ** 2, axis=1)) - self.r_obstacle
                    for c in self.obstacles
                ]
                obs_dists = np.min(np.array(dist_list), axis=0)
                return np.minimum(obs_dists, wall_dists)
            else:
                return wall_dists

        return f

    def get_grid(self) -> Grid:
        raise NotImplementedError("girdsdf currently supports only 3d")  # TODO

    def visualize(self, fax) -> None:
        fig, ax = fax

        circle_outer = Circle((0.5, 0.5), self.r_outer, fill=False)
        circle_inner = Circle((0.5, 0.5), self.r_inner, fill=False)
        ax.add_patch(circle_outer)
        ax.add_patch(circle_inner)

        for obs_pos in self.obstacles:
            circle_obs = Circle(obs_pos, self.r_obstacle, fill=True)
            ax.add_patch(circle_obs)

        ax.set_aspect(1)

    @classmethod
    @abstractmethod
    def get_num_obstacle(cls) -> int:
        ...


class RingObstacleFreeWorld(RingWorldBase):
    @classmethod
    def get_num_obstacle(cls) -> int:
        return 0


class RingNSpherePlanningTask(TaskBase[RingWorldT, Tuple[np.ndarray, ...], None]):
    @staticmethod
    def get_robot_model() -> None:
        return None

    @classmethod
    def sample_descriptions(
        cls, world: RingWorldT, n_sample: int, standard: bool = False
    ) -> List[Tuple[np.ndarray, ...]]:

        descriptions = []

        start = np.array([0.5, 0.05])
        for _ in range(n_sample):
            if standard:
                goal = np.array([0.5, 0.95])
            else:
                sdf = world.get_exact_sdf()

                while True:
                    goal = np.random.rand(2)
                    val = sdf(np.expand_dims(goal, axis=0))[0]
                    if val > world.get_margin():
                        break
            descriptions.append((start, goal))
        return descriptions  # type: ignore

    def export_table(self) -> DescriptionTable:
        wd = {}
        for i, obs in enumerate(self.world.obstacles):
            name = "obs{}".format(i)
            wd[name] = obs

        wcd_list = []
        for desc in self.descriptions:
            wcd = {}
            wcd["start"] = desc[0]
            wcd["goal"] = desc[1]
            wcd_list.append(wcd)
        return DescriptionTable(wd, wcd_list)

    def solve_default_each(self, problem: Problem) -> ResultProtocol:
        ompl_sovler = OMPLSolver.init(OMPLSolverConfig(n_max_call=10000))
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
        sdf = self.world.get_exact_sdf()
        probs = []
        for desc in self.descriptions:
            start, goal = desc

            box = BoxConst(np.zeros(self.get_dof()), np.ones(self.get_dof()))
            goal_const = ConfigPointConst(goal)
            prob = Problem(start, box, goal_const, PointCollFreeConst(sdf), None)
            probs.append(prob)
        return probs

    @staticmethod
    def create_gridsdf(world: RingWorldBase, robot_model: None) -> None:
        return None


class RingObstacleFreePlanningTask(RingNSpherePlanningTask[RingObstacleFreeWorld]):
    @staticmethod
    def get_world_type() -> Type[RingObstacleFreeWorld]:
        return RingObstacleFreeWorld


class Taskvisualizer:
    fax: Tuple

    def __init__(self, task: RingNSpherePlanningTask):
        fig, ax = plt.subplots()
        task.world.visualize((fig, ax))
        for start, goal in task.descriptions:
            ax.scatter(start[0], start[1], c="k")
            ax.scatter(goal[0], goal[1], c="r")
        self.fax = (fig, ax)

    def visualize_trajectories(self, trajs: Union[List[Trajectory], Trajectory]) -> None:
        fig, ax = self.fax
        if isinstance(trajs, Trajectory):
            trajs = [trajs]

        for traj in trajs:
            arr = traj.numpy()
            ax.plot(arr[:, 0], arr[:, 1])
