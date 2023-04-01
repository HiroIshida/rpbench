from abc import abstractmethod
from dataclasses import dataclass
from typing import ClassVar, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
from skmp.solver.interface import Problem, ResultProtocol
from skrobot.model import RobotModel
from skrobot.model.primitives import Axis, Box, Coordinates, MeshLink, Sphere
from skrobot.sdf import UnionSDF
from skrobot.viewers import TrimeshSceneViewer
from voxbloxpy.core import Grid, GridSDF

from rpbench.interface import DescriptionTable, TaskBase, WorldBase
from rpbench.pr2.common import CachedPR2ConstProvider, CachedRArmPR2ConstProvider
from rpbench.utils import SceneWrapper

KivapodWorldT = TypeVar("KivapodWorldT", bound="KivapodWorldBase")


_kivapod_mesh: Optional[MeshLink] = None

Primitive = Union[Box, Sphere]


@dataclass
class KivapodWorldBase(WorldBase):
    kivapod_mesh: MeshLink
    target_region: Box
    obstacles: List[Primitive]

    @classmethod
    def sample(cls: Type[KivapodWorldT], standard: bool = False) -> KivapodWorldT:
        global _kivapod_mesh
        if _kivapod_mesh is None:
            _kivapod_mesh = MeshLink(
                "/home/h-ishida/Downloads/kiva_pod/meshes/pod_lowres.stl",
                dim_grid=300,
                padding_grid=10,
                with_sdf=True,
            )
            _kivapod_mesh.visual_mesh.visual.face_colors = [255, 255, 255, 200]
        kivapod = _kivapod_mesh
        kivapod.rotate(np.pi * 0.5, "x")
        kivapod.rotate(-np.pi * 0.5, "z", wrt="world")
        kivapod.translate([1.0, 0.0, 0.0], wrt="world")

        target_region = Box([0.8, 0.72, 0.4])
        target_region.newcoords(kivapod.copy_worldcoords())
        target_region.translate([-0.2, 0, 1.45], wrt="world")
        target_region.visual_mesh.visual.face_colors = [255, 0, 255, 30]

        return cls(kivapod, target_region, [])

    @classmethod
    @abstractmethod
    def sample_objects(cls, kivapod_mesh: MeshLink) -> List[Primitive]:
        ...

    def get_exact_sdf(self) -> UnionSDF:
        lst = [self.kivapod_mesh.sdf]
        for obstacle in self.obstacles:
            lst.append(obstacle.sdf)
        return UnionSDF(lst)

    def get_grid(self) -> Grid:
        ...

    def visualize(self, viewer: Union[TrimeshSceneViewer, SceneWrapper]) -> None:
        # add origin
        viewer.add(Axis())
        kivapod_axis = Axis.from_coords(self.kivapod_mesh.copy_worldcoords())
        viewer.add(kivapod_axis)
        viewer.add(self.target_region)
        viewer.add(self.kivapod_mesh)


class KivapodEmptyWorld(KivapodWorldBase):
    @classmethod
    def sample_objects(cls, kivapod_mesh: MeshLink) -> List[Primitive]:
        return []


@dataclass
class KivapodReachingTaskBase(TaskBase[KivapodWorldT, Tuple[Coordinates, ...], RobotModel]):
    config_provider: ClassVar[Type[CachedPR2ConstProvider]] = CachedRArmPR2ConstProvider

    @staticmethod
    def get_robot_model() -> RobotModel:
        return CachedPR2ConstProvider.get_pr2()

    @classmethod
    def get_dof(cls) -> int:
        return cls.config_provider.get_dof()

    def export_table(self) -> DescriptionTable:
        ...

    @classmethod
    def sample_descriptions(
        cls, world: KivapodWorldT, n_sample: int, standard: bool = False
    ) -> List[Tuple[Coordinates, ...]]:
        # TODO: duplication of tabletop.py
        if standard:
            assert n_sample == 1
        pose_list: List[Tuple[Coordinates, ...]] = []
        while len(pose_list) < n_sample:
            poses = cls.sample_target_poses(world, standard)
            is_valid_poses = True
            for pose in poses:
                position = np.expand_dims(pose.worldpos(), axis=0)
                if world.get_exact_sdf()(position)[0] < 1e-3:
                    is_valid_poses = False
            if is_valid_poses:
                pose_list.append(poses)
        return pose_list

    @classmethod
    @abstractmethod
    def sample_target_poses(cls, world: KivapodWorldT, standard: bool) -> Tuple[Coordinates, ...]:
        ...


@dataclass
class KivapodEmptyReachingTask(KivapodReachingTaskBase[KivapodEmptyWorld]):
    @staticmethod
    def get_world_type() -> Type[KivapodEmptyWorld]:
        return KivapodEmptyWorld

    @classmethod
    def sample_target_poses(cls, world: KivapodEmptyWorld, standard: bool) -> Tuple[Coordinates]:
        sdf = world.get_exact_sdf()

        n_max_trial = 100
        for _ in range(n_max_trial):
            ext = np.array(world.target_region._extents)
            p_local = -0.5 * ext + np.random.rand(3) * ext
            co = world.target_region.copy_worldcoords()
            co.translate(p_local)
            points = np.expand_dims(co.worldpos(), axis=0)
            sd_val = sdf(points)[0]
            if sd_val > 0.03:
                co.rotate(-np.pi * 0.5, "x")
                co.rotate(+np.pi * 0.5, "z")
                return (co,)
        assert False

    @staticmethod
    def create_gridsdf(world: KivapodEmptyWorld, robot_model: RobotModel) -> GridSDF:
        ...

    def export_problems(self) -> List[Problem]:
        ...

    def solve_default_each(self, problem: Problem) -> ResultProtocol:
        ...
