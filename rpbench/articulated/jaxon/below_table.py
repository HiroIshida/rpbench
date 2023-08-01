from dataclasses import dataclass
from typing import ClassVar, List, Tuple, Type, Union

import numpy as np
from skmp.constraint import CollFreeConst, IneqCompositeConst, PoseConstraint
from skmp.robot.jaxon import Jaxon
from skmp.robot.utils import get_robot_state
from skmp.satisfy import SatisfactionConfig
from skmp.solver.myrrt_solver import MyRRTConfig, MyRRTConnectSolver, MyRRTResult
from skmp.solver.nlp_solver.sqp_based_solver import (
    SQPBasedSolver,
    SQPBasedSolverConfig,
    SQPBasedSolverResult,
)
from skrobot.coordinates import Coordinates
from skrobot.model.primitives import Box
from skrobot.model.robot_model import RobotModel
from skrobot.sdf.signed_distance_function import UnionSDF
from skrobot.viewers import TrimeshSceneViewer
from tinyfk import BaseType

from rpbench.articulated.jaxon.common import CachedJaxonConstProvider
from rpbench.interface import (
    DescriptionTable,
    Problem,
    ReachingTaskBase,
    ResultProtocol,
    WorldBase,
)
from rpbench.timeout_decorator import TimeoutError, timeout
from rpbench.utils import SceneWrapper, skcoords_to_pose_vec


@dataclass
class TableWorld(WorldBase):
    target_region: Box
    table: Box
    obstacle: Box
    _intrinsic_desc: np.ndarray

    @classmethod
    def sample(cls, standard: bool = False) -> "TableWorld":
        if standard:
            table_position = np.array([0.8, 0.0, 0.8])
        else:
            table_position = np.array([0.7, 0.0, 0.6]) + np.random.rand(3) * np.array(
                [0.5, 0.0, 0.4]
            )
        table = Box([1.0, 3.0, 0.1], with_sdf=True)
        table.translate(table_position)

        table_height = table_position[2]
        target_region = Box([0.8, 0.8, table_height], with_sdf=True)
        target_region.visual_mesh.visual.face_colors = [0, 255, 100, 100]
        target_region.translate([0.6, -0.7, 0.5 * table_height])

        # determine obstacle
        if standard:
            obs = Box([0.1, 0.1, 0.5], pos=[0.6, -0.2, 0.25], with_sdf=True)
        else:
            region_width = np.array(target_region._extents[:2])
            region_center = target_region.worldpos()[:2]
            b_min = region_center - region_width * 0.5
            b_max = region_center + region_width * 0.5

            obs_width = np.random.rand(2) * np.ones(2) * 0.2 + np.ones(2) * 0.1
            obs_height = 0.3 + np.random.rand() * 0.5
            b_min = region_center - region_width * 0.5 + 0.5 * obs_width
            b_max = region_center + region_width * 0.5 - 0.5 * obs_width
            pos2d = np.random.rand(2) * (b_max - b_min) + b_min
            pos = np.hstack([pos2d, obs_height * 0.5])
            obs = Box(np.hstack([obs_width, obs_height]), pos=pos, with_sdf=True)

        return cls(target_region, table, obs, table_position)

    def get_exact_sdf(self) -> UnionSDF:
        return UnionSDF([self.table.sdf, self.obstacle.sdf])

    def export_intrinsic_description(self) -> np.ndarray:
        return self._intrinsic_desc

    def visualize(self, viewer: Union[TrimeshSceneViewer, SceneWrapper]) -> None:
        # self.target_region.visual_mesh.visual.face_colors = [255, 255, 255, 120]
        # viewer.add(self.target_region)
        viewer.add(self.table)
        self.obstacle.visual_mesh.visual.face_colors = [255, 0, 0, 150]
        viewer.add(self.obstacle)


class HumanoidTableReachingTask(ReachingTaskBase[TableWorld, Jaxon]):
    config_provider: ClassVar[Type[CachedJaxonConstProvider]] = CachedJaxonConstProvider

    @staticmethod
    def get_robot_model() -> RobotModel:
        return CachedJaxonConstProvider.get_jaxon()

    @staticmethod
    def get_world_type() -> Type[TableWorld]:
        return TableWorld

    @staticmethod
    def create_gridsdf(world: TableWorld, robot_model: RobotModel) -> None:
        return None

    @classmethod
    def get_dof(cls) -> int:
        config = CachedJaxonConstProvider.get_config()
        return len(config._get_control_joint_names()) + 6

    def export_table(self) -> DescriptionTable:
        world_dict = {}
        world_dict["world"] = np.hstack(
            [
                self.world.table.worldpos(),
                self.world.obstacle.worldpos(),
                np.array(self.world.obstacle._extents),
            ]
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
    def sample_target_poses(cls, world: TableWorld, standard: bool) -> Tuple[Coordinates]:
        if standard:
            co = Coordinates([0.55, -0.6, 0.45], rot=[0, -0.5 * np.pi, 0])
            return (co,)

        sdf = world.get_exact_sdf()

        n_max_trial = 100
        ext = np.array(world.target_region._extents)
        for _ in range(n_max_trial):
            p_local = -0.5 * ext + np.random.rand(3) * ext
            co = world.target_region.copy_worldcoords()
            co.translate(p_local)
            points = np.expand_dims(co.worldpos(), axis=0)
            sd_val = sdf(points)[0]
            if sd_val > 0.03:
                co.rotate(-0.5 * np.pi, "y")
                return (co,)
        assert False

    @classmethod
    def sample_descriptions(
        cls, world: TableWorld, n_sample: int, standard: bool = False
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

    def export_problems(self) -> List[Problem]:
        provider = self.config_provider
        jaxon_config = provider.get_config()

        jaxon = provider.get_jaxon()

        # ineq const
        com_const = provider.get_com_const(jaxon)
        colkin = jaxon_config.get_collision_kin()
        colfree_const = CollFreeConst(
            colkin, self.world.get_exact_sdf(), jaxon, only_closest_feature=True
        )

        # the order of ineq const is important here. see comment in IneqCompositeConst
        ineq_const = IneqCompositeConst([com_const, colfree_const])

        q_start = get_robot_state(jaxon, jaxon_config._get_control_joint_names(), BaseType.FLOATING)
        box_const = provider.get_box_const()

        # eq const
        leg_coords_list = [jaxon.rleg_end_coords, jaxon.lleg_end_coords]
        efkin_legs = jaxon_config.get_endeffector_kin(rarm=False, larm=False)
        global_eq_const = PoseConstraint.from_skrobot_coords(leg_coords_list, efkin_legs, jaxon)  # type: ignore

        problems = []
        for desc in self.descriptions:
            goal_eq_const = provider.get_dual_legs_pose_const(jaxon, co_rarm=desc[0])

            problem = Problem(
                q_start,
                box_const,
                goal_eq_const,
                ineq_const,
                global_eq_const,
                motion_step_box_=jaxon_config.get_motion_step_box() * 0.5,
            )
            problems.append(problem)
        return problems

    def solve_default_each(self, problem: Problem) -> ResultProtocol:
        try:
            return self._solve_default_each(problem)
        except TimeoutError:
            print("timeout!! solved default failed.")
            return MyRRTResult.abnormal()

    @timeout(180)
    def _solve_default_each(self, problem: Problem) -> ResultProtocol:
        rrt_conf = MyRRTConfig(5000, satisfaction_conf=SatisfactionConfig(n_max_eval=20))

        sqp_config = SQPBasedSolverConfig(
            n_wp=40,
            n_max_call=20,
            motion_step_satisfaction="explicit",
            verbose=False,
            ctol_eq=1e-3,
            ctol_ineq=1e-3,
            ineq_tighten_coef=0.0,
        )

        for _ in range(4):
            rrt = MyRRTConnectSolver.init(rrt_conf)
            rrt.setup(problem)
            rrt_result = rrt.solve()

            if rrt_result.traj is not None:
                sqp = SQPBasedSolver.init(sqp_config)
                sqp.setup(problem)
                sqp_result = sqp.solve(rrt_result.traj)
                if sqp_result.traj is not None:
                    return sqp_result

        return SQPBasedSolverResult.abnormal()

    def export_intrinsic_descriptions(self) -> List[np.ndarray]:
        world_vec = self.world.export_intrinsic_description()

        intrinsic_descs = []
        for desc in self.descriptions:
            pose_vecs = [skcoords_to_pose_vec(pose) for pose in desc]
            vecs = [world_vec] + pose_vecs
            intrinsic_desc = np.hstack(vecs)
            intrinsic_descs.append(intrinsic_desc)
        return intrinsic_descs
