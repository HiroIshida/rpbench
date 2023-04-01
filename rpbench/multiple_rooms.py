import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import ClassVar, List, Tuple, Type, TypeVar, Union

import matplotlib.pyplot as plt
import numpy as np
from skmp.constraint import BoxConst, ConfigPointConst, PointCollFreeConst
from skmp.solver.interface import Problem, ResultProtocol
from skmp.solver.nlp_solver.sqp_based_solver import SQPBasedSolver, SQPBasedSolverConfig
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig
from skmp.trajectory import Trajectory
from voxbloxpy.core import Grid

from rpbench.interface import DescriptionTable, SDFProtocol, TaskBase, WorldBase

MultipleRoomsWorldT = TypeVar("MultipleRoomsWorldT", bound="MultipleRoomsWorldBase")


def box_sdf(points: np.ndarray, width: np.ndarray, origin: np.ndarray):
    n_pts, _ = points.shape

    half_extent = width * 0.5
    pts_from_center = points - origin
    sd_vals_each_axis = np.abs(pts_from_center) - half_extent[None, :]

    positive_dists_each_axis = np.maximum(sd_vals_each_axis, 0.0)
    positive_dists = np.sqrt(np.sum(positive_dists_each_axis**2, axis=1))

    negative_dists_each_axis = np.max(sd_vals_each_axis, axis=1)
    negative_dists = np.minimum(negative_dists_each_axis, 0.0)

    sd_vals = positive_dists + negative_dists
    return sd_vals


@dataclass
class MultipleRoomsWorldBase(WorldBase):
    angles: List[float]
    r_main_room: ClassVar[float] = 0.25
    r_side_room: ClassVar[float] = 0.25
    w_bridge: ClassVar[float] = 0.08

    @classmethod
    @abstractmethod
    def get_n_room(cls) -> int:
        ...

    def get_margin(self) -> float:
        return self.w_bridge * 0.1

    def get_grid(self) -> Grid:
        raise NotImplementedError("girdsdf currently supports only 3d")  # TODO

    @classmethod
    def sample(cls: Type[MultipleRoomsWorldT], standard: bool = False) -> MultipleRoomsWorldT:
        n_room = cls.get_n_room()
        assert n_room < 14

        angles: List[float] = []
        room_centers: List[np.ndarray] = []
        for i in range(n_room):
            angle = 2 * math.pi / n_room * i
            room_center = np.array([math.cos(angle), math.sin(angle)]) * (1 - cls.r_side_room)
            angles.append(angle)
            room_centers.append(room_center)
        return cls(angles)

    def get_exact_sdf(self) -> SDFProtocol:

        self.get_n_room()

        def f(pts: np.ndarray):

            # side circles
            dists_list = []
            for angle in self.angles:
                room_center = np.array([math.cos(angle), math.sin(angle)]) * (1 - self.r_side_room)
                dists = np.sqrt(np.sum((pts - room_center) ** 2, axis=1)) - self.r_side_room
                dists_list.append(dists)

            # bridge square
            for angle in self.angles:
                coss = math.cos(angle)
                sinn = math.sin(angle)
                rot = np.array([[coss, -sinn], [sinn, coss]])

                pts_rotated = pts.dot(rot.T)
                center = np.array([1, 0]) * 0.5 * (self.r_main_room + (1 - self.r_side_room * 2))
                size = np.array([0.25 * 1.1, self.w_bridge])
                dists = box_sdf(pts_rotated, size, center)
                dists_list.append(dists)

            dists = np.sqrt(np.sum((pts) ** 2, axis=1)) - self.r_main_room
            dists_list.append(dists)
            min_dists = np.min(dists_list, axis=0)
            return -min_dists

        return f

    def visualize(self, fax) -> None:
        fig, ax = fax

        xlin, ylin = [np.linspace(-1.1, +1.1, 200) for i in range(2)]
        meshes = np.meshgrid(xlin, ylin)
        meshes_flatten = [mesh.flatten() for mesh in meshes]
        pts = np.array([p for p in zip(*meshes_flatten)])
        sdf = self.get_exact_sdf()
        sdf_mesh = sdf(pts).reshape(200, 200)
        if False:
            ax.contourf(xlin, ylin, sdf_mesh, cmap="summer")
        ax.contour(xlin, ylin, sdf_mesh, cmap="gray", levels=[0.0])
        return fig, ax


class EightRoomsWorld(MultipleRoomsWorldBase):
    @classmethod
    def get_n_room(cls) -> int:
        return 8


class MultipleRoomsPlanningTaskBase(TaskBase[MultipleRoomsWorldT, Tuple[np.ndarray, ...], None]):
    @staticmethod
    def get_robot_model() -> None:
        return None

    @classmethod
    def sample_descriptions(
        cls, world: MultipleRoomsWorldT, n_sample: int, standard: bool = False
    ) -> List[Tuple[np.ndarray, ...]]:

        descriptions = []

        start = np.zeros(2)
        for _ in range(n_sample):
            if standard:
                goal = np.array([0.0, 1 - world.r_side_room])
            else:
                sdf = world.get_exact_sdf()

                while True:
                    goal = np.random.rand(2) * 2 - np.ones(2)
                    val = sdf(np.expand_dims(goal, axis=0))[0]
                    if val > world.get_margin():
                        break
            descriptions.append((start, goal))

        return descriptions  # type: ignore

    def export_table(self) -> DescriptionTable:
        wd = {}
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

            box = BoxConst(-np.ones(self.get_dof()), np.ones(self.get_dof()))
            goal_const = ConfigPointConst(goal)
            prob = Problem(
                start, box, goal_const, PointCollFreeConst(sdf), None, motion_step_box_=0.03
            )
            probs.append(prob)
        return probs

    @staticmethod
    def create_gridsdf(world: EightRoomsWorld, robot_model: None) -> None:
        return None


class EightRoomsPlanningTask(MultipleRoomsPlanningTaskBase[EightRoomsWorld]):
    @staticmethod
    def get_world_type() -> Type[EightRoomsWorld]:
        return EightRoomsWorld


class Taskvisualizer:
    fax: Tuple

    def __init__(self, task: EightRoomsPlanningTask):
        fig, ax = plt.subplots()
        task.world.visualize((fig, ax))
        for start, goal in task.descriptions:
            ax.scatter(start[0], start[1], c="k")
            ax.scatter(goal[0], goal[1], c="r")
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        self.fax = (fig, ax)

    def visualize_trajectories(self, trajs: Union[List[Trajectory], Trajectory]) -> None:
        fig, ax = self.fax
        if isinstance(trajs, Trajectory):
            trajs = [trajs]

        for traj in trajs:
            arr = traj.numpy()
            ax.plot(arr[:, 0], arr[:, 1])
