import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, List, Tuple, Type, TypeVar, Union

import numpy as np
from skmp.constraint import BoxConst, ConfigPointConst, PointCollFreeConst
from skmp.solver.interface import Problem, ResultProtocol
from skmp.solver.nlp_solver.sqp_based_solver import SQPBasedSolver, SQPBasedSolverConfig
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig
from skmp.trajectory import Trajectory
from skrobot.model.primitives import Box, LineString, Sphere
from skrobot.viewers import TrimeshSceneViewer
from voxbloxpy.core import Grid, GridSDF

from rpbench.interface import DescriptionTable, SDFProtocol, TaskBase, WorldBase
from rpbench.utils import SceneWrapper, temp_seed

CubicWorldT = TypeVar("CubicWorldT", bound="CubicWorld")


@dataclass
class CubicWorld(WorldBase):
    obstacles: List[np.ndarray]

    @classmethod
    def sample(cls: Type[CubicWorldT], standard: bool = False) -> CubicWorldT:
        r = cls.get_radius()
        R = 2 * (r + cls.get_margin())

        def no_overwrap(new_sphere: np.ndarray, sphers: List[np.ndarray]):
            for s in sphers:
                dist = np.linalg.norm(new_sphere - s)
                if dist < R:
                    return False
            return True

        with temp_seed(0, standard):
            obstacles: List[np.ndarray] = []
            while len(obstacles) < cls.get_num_obstacle():
                if standard and len(obstacles) > 0:
                    n_cand = 100
                    obs_stacked = np.array(obstacles)
                    minmaxdist = -np.inf
                    x = None
                    for i in range(n_cand):
                        x_cand = r + np.random.rand(3) * (1 - 2 * r)
                        if np.linalg.norm(x_cand - np.ones(3) * 0.1) < r + cls.get_margin():
                            continue
                        if np.linalg.norm(x_cand - np.ones(3) * 0.9) < r + cls.get_margin():
                            continue
                        dists = [np.linalg.norm(x_cand - obs) for obs in obs_stacked]
                        mindist = np.min(dists)
                        if mindist > minmaxdist:
                            x = x_cand
                            minmaxdist = mindist
                    assert x is not None
                else:
                    x = r + np.random.rand(3) * (1 - 2 * r)
                obstacles.append(x)
        return cls(obstacles)

    def get_exact_sdf(self) -> SDFProtocol:
        def f(x: np.ndarray):
            dist_list = [
                np.sqrt(np.sum((x - c) ** 2, axis=1)) - self.get_radius() for c in self.obstacles
            ]
            return np.min(np.array(dist_list), axis=0)

        return f

    def get_grid(self) -> Grid:
        return Grid(np.zeros(3), np.ones(3), (50, 50, 50))

    @classmethod
    def get_margin(cls) -> float:
        return 0.05

    @classmethod
    def get_radius(cls) -> float:
        radius = (cls.get_packing_ratio() * 3 / cls.get_num_obstacle() / 4 / np.pi) ** (1 / 3.0)
        return radius

    @classmethod
    @abstractmethod
    def get_num_obstacle(cls) -> int:
        ...

    @classmethod
    def get_packing_ratio(cls) -> float:
        return 0.2

    def visualize(self, viewer: Union[TrimeshSceneViewer, SceneWrapper]) -> None:
        viewer.add(Box([1, 1, 1], pos=(0.5, 0.5, 0.5), face_colors=[255, 0, 0, 30]))
        for obs_pos in self.obstacles:
            obj = Sphere(self.get_radius(), pos=obs_pos)
            viewer.add(obj)


class Cubic1SphereWorld(CubicWorld):
    @classmethod
    def get_num_obstacle(cls) -> int:
        return 1


class Cubic2SphereWorld(CubicWorld):
    @classmethod
    def get_num_obstacle(cls) -> int:
        return 2


class Cubic3SphereWorld(CubicWorld):
    @classmethod
    def get_num_obstacle(cls) -> int:
        return 3


class Cubic4SphereWorld(CubicWorld):
    @classmethod
    def get_num_obstacle(cls) -> int:
        return 4


class Cubic5SphereWorld(CubicWorld):
    @classmethod
    def get_num_obstacle(cls) -> int:
        return 5


class CubicNSpherePlanningTask(TaskBase[CubicWorldT, Tuple[np.ndarray, ...], None]):
    @staticmethod
    def get_robot_model() -> None:
        return None

    @classmethod
    def sample_descriptions(
        cls, world: CubicWorldT, n_sample: int, standard: bool = False
    ) -> List[Tuple[np.ndarray, ...]]:
        descriptions = []
        for _ in range(n_sample):
            if standard:
                start = np.ones(3) * 0.1
                goal = np.ones(3) * 0.9
            else:
                start = np.ones(3) * 0.1
                sdf = world.get_exact_sdf()

                while True:
                    goal = np.random.rand(3)
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
        return 3

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


class ExactGridSDFCreator:
    @staticmethod
    def create_gridsdf(world: CubicWorld, robot_model: None) -> GridSDF:
        grid = world.get_grid()
        sdf = world.get_exact_sdf()

        X, Y, Z = grid.get_meshgrid(indexing="ij")
        pts = np.array(list(zip(X.flatten(), Y.flatten(), Z.flatten())))
        values = sdf.__call__(pts)
        gridsdf = GridSDF(grid, values, 2.0, create_itp_lazy=True)
        gridsdf = gridsdf.get_quantized()
        return gridsdf


class Cubic1SpherePlanningTask(ExactGridSDFCreator, CubicNSpherePlanningTask[Cubic1SphereWorld]):
    @staticmethod
    def get_world_type() -> Type[Cubic1SphereWorld]:
        return Cubic1SphereWorld


class Cubic2SpherePlanningTask(ExactGridSDFCreator, CubicNSpherePlanningTask[Cubic2SphereWorld]):
    @staticmethod
    def get_world_type() -> Type[Cubic2SphereWorld]:
        return Cubic2SphereWorld


class Cubic3SpherePlanningTask(ExactGridSDFCreator, CubicNSpherePlanningTask[Cubic3SphereWorld]):
    @staticmethod
    def get_world_type() -> Type[Cubic3SphereWorld]:
        return Cubic3SphereWorld


class Cubic4SpherePlanningTask(ExactGridSDFCreator, CubicNSpherePlanningTask[Cubic4SphereWorld]):
    @staticmethod
    def get_world_type() -> Type[Cubic4SphereWorld]:
        return Cubic4SphereWorld


class Cubic5SpherePlanningTask(ExactGridSDFCreator, CubicNSpherePlanningTask[Cubic5SphereWorld]):
    @staticmethod
    def get_world_type() -> Type[Cubic5SphereWorld]:
        return Cubic5SphereWorld


ViewerT = TypeVar("ViewerT", bound=Union[TrimeshSceneViewer, SceneWrapper])


class TaskVisualizerBase(Generic[ViewerT, CubicWorldT], ABC):
    # TODO: this class actually take any Task if it has config provider
    task: CubicNSpherePlanningTask[CubicWorldT]
    viewer: ViewerT
    _show_called: bool

    def __init__(self, task: CubicNSpherePlanningTask[CubicWorldT]):
        viewer = self.viewer_type()()
        task.world.visualize(viewer)

        for desc in task.descriptions:
            start, goal = desc
            start_marker = Sphere(0.02, pos=start)
            goal_marker = Sphere(0.02, pos=goal)
            viewer.add(start_marker)
            viewer.add(goal_marker)

        self.task = task
        self.viewer = viewer
        self._show_called = False

    def visualize_trajectory(self, trajectory: Trajectory, t_interval: float = 0.6) -> None:
        lines = LineString(trajectory.numpy())
        self.viewer.add(lines)

    @classmethod
    @abstractmethod
    def viewer_type(cls) -> Type[ViewerT]:
        ...


class InteractiveTaskVisualizer(TaskVisualizerBase[TrimeshSceneViewer, CubicWorldT]):
    def show(self) -> None:
        self.viewer.show()
        time.sleep(1.0)
        self._show_called = True

    @classmethod
    def viewer_type(cls) -> Type[TrimeshSceneViewer]:
        return TrimeshSceneViewer


class StaticTaskVisualizer(TaskVisualizerBase[SceneWrapper, CubicWorldT]):
    @classmethod
    def viewer_type(cls) -> Type[SceneWrapper]:
        return SceneWrapper

    def save_image(self, path: Union[Path, str]) -> None:
        if isinstance(path, str):
            path = Path(path)
        png = self.viewer.save_image(resolution=[640, 480], visible=True)
        with path.open(mode="wb") as f:
            f.write(png)
