import copy
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
from skmp.solver.interface import Problem, ResultProtocol
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig
from skrobot.coordinates import CascadedCoords, Coordinates
from skrobot.model import RobotModel
from skrobot.model.primitives import Axis, Box, Sphere
from skrobot.sdf import GridSDF as SkrobotGridSDF
from skrobot.sdf import UnionSDF
from skrobot.viewers import TrimeshSceneViewer
from voxbloxpy.core import Grid

from rpbench.interface import DescriptionTable, TaskBase, WorldBase
from rpbench.pr2.common import CachedPR2ConstProvider, CachedRArmPR2ConstProvider
from rpbench.pr2.utils import MeshLink
from rpbench.utils import SceneWrapper, create_union_sdf, skcoords_to_pose_vec

KivapodWorldT = TypeVar("KivapodWorldT", bound="KivapodWorldBase")


_kivapod_mesh: Optional[MeshLink] = None

Primitive = Union[Box, Sphere]


@dataclass
class KivapodWorldBase(WorldBase):
    kivapod_mesh: MeshLink
    target_region: Box
    obstacles: List[Primitive]

    @classmethod
    def _create_kivapod_mesh_if_necessary(cls) -> None:
        global _kivapod_mesh
        if _kivapod_mesh is None:
            current_script_path = Path(__file__).resolve().parent
            stl_file_path = str(current_script_path / "pod_lowres.stl")
            sdf = SkrobotGridSDF.from_objfile(
                stl_file_path, dim_grid=300, padding_grid=10, fill_value=2.0
            )
            _kivapod_mesh = MeshLink(stl_file_path, with_sdf=True, forced_sdf=sdf)
            _kivapod_mesh.visual_mesh.visual.face_colors = [255, 255, 255, 200]

    @classmethod
    def sample(cls: Type[KivapodWorldT], standard: bool = False) -> KivapodWorldT:
        cls._create_kivapod_mesh_if_necessary()
        global _kivapod_mesh
        assert _kivapod_mesh is not None
        sdf = copy.copy(_kivapod_mesh.sdf)
        sdf.coords = CascadedCoords()
        kivapod = MeshLink(visual_mesh=_kivapod_mesh.visual_mesh, with_sdf=True, forced_sdf=sdf)
        kivapod.rotate(np.pi * 0.5, "x")
        kivapod.rotate(-np.pi * 0.5, "z", wrt="world")
        kivapod.translate([1.0, 0.0, 0.0], wrt="world")

        target_region = Box([0.8, 0.52, 0.3])
        target_region.newcoords(kivapod.copy_worldcoords())
        target_region.translate([-0.25, 0, 1.05], wrt="world")
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
        raise NotImplementedError("girdsdf is not used")

    def visualize(self, viewer: Union[TrimeshSceneViewer, SceneWrapper]) -> None:
        # add origin
        viewer.add(Axis())
        kivapod_axis = Axis.from_coords(self.kivapod_mesh.copy_worldcoords())
        viewer.add(kivapod_axis)
        viewer.add(self.target_region)
        viewer.add(self.kivapod_mesh)

    def __getstate__(self):
        state = self.__dict__.copy()

        # remove kivapod mesh from the state because it's too heavy
        state["kivapod_mesh_co"] = state[
            "kivapod_mesh"
        ].copy_worldcoords()  # use again in __setstate__
        state["kivapod_mesh"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        self._create_kivapod_mesh_if_necessary()
        global _kivapod_mesh
        assert _kivapod_mesh is not None
        sdf = copy.copy(_kivapod_mesh.sdf)
        sdf.coords = CascadedCoords()
        self.kivapod_mesh = MeshLink(
            visual_mesh=_kivapod_mesh.visual_mesh, with_sdf=True, forced_sdf=sdf
        )
        self.kivapod_mesh.newcoords(state["kivapod_mesh_co"])


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
        world_dict = {}
        if len(self.world.obstacles) > 0:
            assert self._gridsdf is not None
            world_dict["world"] = self._gridsdf.values.reshape(self._gridsdf.grid.sizes)

        world_dict["kivapod_pose"] = skcoords_to_pose_vec(
            self.world.kivapod_mesh.copy_worldcoords()
        )

        desc_dicts = []
        for desc in self.descriptions:
            desc_dict = {}
            for idx, co in enumerate(desc):
                pose = skcoords_to_pose_vec(co)
                name = "target_pose-{}".format(idx)
                desc_dict[name] = pose
            desc_dicts.append(desc_dict)
        return DescriptionTable(world_dict, desc_dicts)

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
        ext = np.array(world.target_region._extents)
        if standard:
            co = world.target_region.copy_worldcoords()
            co.rotate(-np.pi * 0.5, "x")
            co.rotate(+np.pi * 0.5, "z")
            co.translate([-0.1, 0.0, 0.12], wrt="local")
            return (co,)

        sdf = world.get_exact_sdf()

        n_max_trial = 100
        for _ in range(n_max_trial):
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
    def create_gridsdf(world: KivapodEmptyWorld, robot_model: RobotModel) -> None:
        return None

    def export_problems(self) -> List[Problem]:
        provider = self.config_provider
        q_start = provider.get_start_config()
        box_const = provider.get_box_const()

        if len(self.world.obstacles) > 0:
            assert self._gridsdf is not None
            sdf = create_union_sdf([self._gridsdf, self.world.kivapod_mesh.sdf])
        else:
            sdf = self.world.kivapod_mesh.sdf

        ineq_const = provider.get_collfree_const(sdf)

        problems = []
        for desc in self.descriptions:
            pose_const = provider.get_pose_const(list(desc))
            problem = Problem(q_start, box_const, pose_const, ineq_const, None)
            problems.append(problem)
        return problems

    def solve_default_each(self, problem: Problem) -> ResultProtocol:
        n_planning_budget = 30
        for _ in range(n_planning_budget):
            solcon = OMPLSolverConfig(n_max_call=20000, n_max_satisfaction_trial=100, simplify=True)
            ompl_solver = OMPLSolver.init(solcon)
            ompl_solver.setup(problem)
            ompl_ret = ompl_solver.solve()
            print(ompl_ret)
            if ompl_ret.traj is not None:
                return ompl_ret
        return ompl_ret
