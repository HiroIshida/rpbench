from typing import ClassVar, List, Tuple, Type

import numpy as np
from skmp.constraint import CollFreeConst, IneqCompositeConst, PoseConstraint
from skmp.robot.jaxon import Jaxon
from skmp.robot.utils import get_robot_state
from skmp.satisfy import SatisfactionConfig
from skmp.solver.myrrt_solver import MyRRTConfig, MyRRTConnectSolver
from skmp.solver.nlp_solver.sqp_based_solver import (
    SQPBasedSolver,
    SQPBasedSolverConfig,
    SQPBasedSolverResult,
)
from skrobot.coordinates import Coordinates
from skrobot.model.robot_model import RobotModel
from tinyfk import BaseType
from voxbloxpy.core import GridSDF

from rpbench.articulated.jaxon.common import CachedJaxonConstProvider
from rpbench.articulated.world.tabletop import TabletopBoxWorld
from rpbench.interface import (
    DescriptionTable,
    Problem,
    ReachingTaskBase,
    ResultProtocol,
)
from rpbench.utils import skcoords_to_pose_vec


class HumanoidTableTopRarmReachingTask(ReachingTaskBase[TabletopBoxWorld, Jaxon]):
    config_provider: ClassVar[Type[CachedJaxonConstProvider]] = CachedJaxonConstProvider

    @staticmethod
    def get_robot_model() -> RobotModel:
        return CachedJaxonConstProvider.get_jaxon()

    @staticmethod
    def get_world_type() -> Type[TabletopBoxWorld]:
        return TabletopBoxWorld

    @staticmethod
    def create_gridsdf(world: TabletopBoxWorld, robot_model: RobotModel) -> GridSDF:
        grid = world.get_grid()
        sdf = world.get_exact_sdf()

        X, Y, Z = grid.get_meshgrid(indexing="ij")
        pts = np.array(list(zip(X.flatten(), Y.flatten(), Z.flatten())))
        values = sdf.__call__(pts)
        gridsdf = GridSDF(grid, values, 2.0, create_itp_lazy=True)
        gridsdf = gridsdf.get_quantized()
        return gridsdf

    @classmethod
    def get_dof(cls) -> int:
        config = CachedJaxonConstProvider.get_config()
        return len(config._get_control_joint_names()) + 6

    def export_table(self) -> DescriptionTable:
        world_dict = {}
        world_dict["world"] = self.gridsdf.values.reshape(self.gridsdf.grid.sizes)
        world_dict["table_pose"] = skcoords_to_pose_vec(self.world.table.copy_worldcoords())

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
    def sample_target_poses(
        cls, world: TabletopBoxWorld, standard: bool
    ) -> Tuple[Coordinates, ...]:
        # NOTE: COPIED from pr2.tabletop

        if standard:
            _, _, height = world.table._extents
            co = world.table.copy_worldcoords()
            co.translate([0.0, -0.2, 0.5 * height + 0.2])
            co.rotate(-np.pi * 0.5, "y")
            return (co,)
        else:
            co = world.sample_pose()

            world.get_exact_sdf()
            tabletop_center = world.table.copy_worldcoords()

            margin_from_plane = 0.1

            depth, width, height = world.table._extents
            relative_pos = np.random.rand(3) * np.array([depth, width, 0.1]) + np.array(
                [-depth * 0.5, -width * 0.5, height * 0.5 + margin_from_plane]
            )
            co = tabletop_center.copy_worldcoords()
            co.translate(relative_pos)
            co.rotate(np.pi * 0.5, "y")
            return (co,)

    @classmethod
    def sample_descriptions(
        cls, world: TabletopBoxWorld, n_sample: int, standard: bool = False
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
        rrt_conf = MyRRTConfig(5000, satisfaction_conf=SatisfactionConfig(n_max_eval=20))

        sqp_config = SQPBasedSolverConfig(
            n_wp=40,
            n_max_call=20,
            motion_step_satisfaction="explicit",
            verbose=True,
            ctol_eq=1e-3,
            ctol_ineq=1e-3,
            ineq_tighten_coef=0.0,
        )

        for _ in range(4):
            rrt = MyRRTConnectSolver.init(rrt_conf)
            rrt.setup(problem)
            rrt_result = rrt.solve()

            if rrt_result.traj is not None:
                return rrt_result
                sqp = SQPBasedSolver.init(sqp_config)
                sqp.setup(problem)
                sqp_result = sqp.solve(rrt_result.traj)
                if sqp_result.traj is not None:
                    return sqp_result

        return SQPBasedSolverResult.abnormal(np.inf)
