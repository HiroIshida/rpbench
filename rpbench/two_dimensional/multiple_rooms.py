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


def rotation_matrix(angle: float) -> np.ndarray:
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)

    return np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])


@dataclass
class MultipleRoomsWorldBase(WorldBase):
    angles: List[float]
    r_main_room: ClassVar[float] = 0.25
    r_side_room: ClassVar[float] = 0.25
    w_bridge: ClassVar[float] = 0.05

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
        def f(pts: np.ndarray):
            def norm_batch(pts, center) -> np.ndarray:
                dists = np.sqrt(np.sum((pts - center) ** 2, axis=1))
                return dists

            dists_list = []
            theta_bridge = 1.0
            for angle in self.angles:
                # sdf of room center
                room_center = np.array([math.cos(angle), math.sin(angle)]) * (1 - self.r_side_room)
                dists = norm_batch(pts, room_center) - self.r_side_room
                dists_list.append(dists)

                # sdf of bridge
                L = np.linalg.norm(room_center)
                R = L * 0.5 / np.cos(-0.5 * np.pi + theta_bridge)
                bridge_circle_center = (
                    rotation_matrix(-0.5 * np.pi + theta_bridge).dot(room_center) * R / L
                )
                R_outer = R + 0.5 * self.w_bridge
                R_inner = R - 0.5 * self.w_bridge

                dists_outer = norm_batch(pts, bridge_circle_center) - R_outer
                dists_inner = norm_batch(pts, bridge_circle_center) - R_inner
                dists_subtract = np.maximum(dists_outer, -dists_inner)

                n1 = np.array([np.cos(angle + theta_bridge), np.sin(angle + theta_bridge)])
                dists_block1 = np.dot(pts, +n1)
                dists_subtract = np.maximum(dists_subtract, -dists_block1)

                n2 = np.array([np.cos(angle - theta_bridge), np.sin(angle - theta_bridge)])
                dists_block2 = np.dot(pts - room_center, -n2)
                dists_subtract = np.maximum(dists_subtract, -dists_block2)

                dists_list.append(dists_subtract)

            dists = np.sqrt(np.sum((pts) ** 2, axis=1)) - self.r_main_room
            dists_list.append(dists)
            min_dists = np.min(dists_list, axis=0)
            return -min_dists

        return f

    def visualize(self, fax) -> Tuple:
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
        wd = {}  # type: ignore[var-annotated]
        wcd_list = []
        for desc in self.descriptions:
            wcd = {}
            wcd["start"] = desc[0]
            wcd["goal"] = desc[1]
            wcd_list.append(wcd)
        return DescriptionTable(wd, wcd_list)

    def solve_default_each(self, problem: Problem) -> ResultProtocol:
        ompl_sovler = OMPLSolver.init(OMPLSolverConfig(n_max_call=10000, algorithm_range=0.1))
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

        def sdf_modified(x):
            val = sdf(x) - self.world.get_margin()
            return val

        probs = []
        for desc in self.descriptions:
            start, goal = desc

            box = BoxConst(-np.ones(self.get_dof()), np.ones(self.get_dof()))
            goal_const = ConfigPointConst(goal)
            prob = Problem(
                start,
                box,
                goal_const,
                PointCollFreeConst(sdf_modified),
                None,
                motion_step_box_=0.05,
            )
            probs.append(prob)
        return probs

    @staticmethod
    def create_gridsdf(world: EightRoomsWorld, robot_model: None) -> None:  # type: ignore[override]
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
            ax.plot(arr[:, 0], arr[:, 1], ".-")
