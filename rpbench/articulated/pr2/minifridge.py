from typing import Any, ClassVar, List, Literal, Tuple, Type, overload

import numpy as np
from skmp.constraint import CollFreeConst
from skmp.robot.utils import get_robot_state, set_robot_state
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig
from skmp.visualization.solution_visualizer import (
    InteractiveSolutionVisualizer,
    SolutionVisualizerBase,
    StaticSolutionVisualizer,
)
from skrobot.coordinates import Coordinates
from skrobot.model.primitives import Axis
from skrobot.model.robot_model import RobotModel
from tinyfk import BaseType

from rpbench.articulated.pr2.common import (
    CachedRArmFixedPR2ConstProvider,
    CachedRArmPR2ConstProvider,
)
from rpbench.articulated.world.minifridge import MiniFridgeWorld
from rpbench.interface import Problem, ResultProtocol, TaskBase, TaskExpression
from rpbench.utils import skcoords_to_pose_vec, temp_seed


class PR2MiniFridgeTask(TaskBase[MiniFridgeWorld, Tuple[Coordinates, np.ndarray], RobotModel]):
    config_provider: ClassVar[
        Type[CachedRArmFixedPR2ConstProvider]
    ] = CachedRArmFixedPR2ConstProvider

    @staticmethod
    def get_world_type() -> Type[MiniFridgeWorld]:
        return MiniFridgeWorld

    @staticmethod
    def get_robot_model() -> RobotModel:
        return CachedRArmFixedPR2ConstProvider.get_pr2()

    @classmethod
    def sample_descriptions(
        cls, world: MiniFridgeWorld, n_sample: int, standard: bool = False
    ) -> List[Tuple[Coordinates, np.ndarray]]:

        if standard:
            assert n_sample == 1

        with temp_seed(0, use_tempseed=standard):
            pose_list: List[Coordinates] = []
            while len(pose_list) < n_sample:
                pose = world.sample_pregrasp_coords()
                if pose is not None:
                    pose_list.append(pose)

            colkin = CachedRArmFixedPR2ConstProvider.get_whole_body_colkin()
            pr2 = CachedRArmPR2ConstProvider.get_pr2()
            sdf = world.get_exact_sdf()
            collfree_const = CollFreeConst(colkin, sdf, pr2)

        if standard:
            base_pos = np.array([-0.1, 0.0, 0.0])
        else:
            while True:
                base_pos = np.random.randn(3) * np.array([0.1, 0.2, 0.3]) - np.array(
                    [-0.05, 0.0, 0.0]
                )
                set_robot_state(pr2, [], base_pos, base_type=BaseType.PLANER)
                colkin.reflect_skrobot_model(pr2)
                q = get_robot_state(pr2, colkin.control_joint_names)

                if collfree_const.is_valid(q):
                    break
        descriptions = [(pose, base_pos) for pose in pose_list]
        return descriptions

    def export_task_expression(self, use_matrix: bool) -> TaskExpression:
        if use_matrix:
            world_vec = np.array([self.world.fridge_conts.fridge.angle])
            world_mat = self.world.fridge_conts.create_heightmap()
        else:
            world_vec = self.world.to_parameter()
            world_mat = None

        other_vec_list = []
        for desc in self.descriptions:
            target_pose, init_pose = desc
            other_vec = np.hstack([skcoords_to_pose_vec(target_pose, yaw_only=True), init_pose])
            other_vec_list.append(other_vec)
        return TaskExpression(world_vec, world_mat, other_vec_list)

    @classmethod
    def from_task_params(cls, params: np.ndarray) -> "PR2MiniFridgeTask":
        world_type = cls.get_world_type()
        world_param_dof = world_type.get_world_dof()
        world = None
        desc_list = []
        for param in params:
            world_param = param[:world_param_dof]
            if world is None:
                world = world_type.from_parameter(world_param)
            other_param = param[world_param_dof:]
            pose_param = other_param[:4]
            ypr = (pose_param[3], 0, 0)
            co = Coordinates(pose_param[:3], ypr)
            base_param = other_param[4:]
            desc_list.append((co, base_param))
        assert world is not None
        return cls(world, desc_list)

    def export_problems(self) -> List[Problem]:
        provider = self.config_provider
        q_start = provider.get_start_config()
        box_const = provider.get_box_const()

        sdf = self.world.get_exact_sdf()
        ineq_const = provider.get_collfree_const(sdf)

        pr2 = provider.get_pr2()

        joint_names = provider.get_config()._get_control_joint_names()
        assert joint_names == [
            "r_shoulder_pan_joint",
            "r_shoulder_lift_joint",
            "r_upper_arm_roll_joint",
            "r_elbow_flex_joint",
            "r_forearm_roll_joint",
            "r_wrist_flex_joint",
            "r_wrist_roll_joint",
        ]
        motion_step_box = np.array(
            [
                0.05,  # r_shoulder_pan_joint
                0.05,  # r_shoulder_lift_joint
                0.1,  # r_upper_arm_rolr_joint
                0.1,  # r_elbow_flex_joint
                0.2,  # r_forearm_rolr_joint
                0.2,  # r_wrist_flex_joint
                0.5,  # r_wrist_rolr_joint
            ]
        )

        problems = []
        for target_pose, base_pose in self.descriptions:
            set_robot_state(pr2, [], base_pose, base_type=BaseType.PLANER)
            pose_const = provider.get_pose_const([target_pose])

            pose_const.reflect_skrobot_model(pr2)
            ineq_const.reflect_skrobot_model(pr2)

            problem = Problem(
                q_start, box_const, pose_const, ineq_const, None, motion_step_box_=motion_step_box
            )
            problems.append(problem)
        return problems

    def solve_default_each(self, problem: Problem) -> ResultProtocol:
        solcon = OMPLSolverConfig(n_max_call=40000, n_max_satisfaction_trial=200, simplify=True)

        ompl_solver = OMPLSolver.init(solcon)
        ompl_solver.setup(problem)
        return ompl_solver.solve()

    @classmethod
    def get_task_dof(cls) -> int:
        return cls.get_world_type().get_world_dof() + 4 + 3  # type: ignore[attr-defined]

    @overload
    def create_viewer(self, mode: Literal["static"]) -> StaticSolutionVisualizer:
        ...

    @overload
    def create_viewer(self, mode: Literal["interactive"]) -> InteractiveSolutionVisualizer:
        ...

    def create_viewer(self, mode: str) -> Any:
        assert len(self.descriptions) == 1
        target_co, base_pose = self.descriptions[0]
        geometries = [Axis.from_coords(target_co)]
        # geometries = []
        config = self.config_provider.get_config()  # type: ignore[attr-defined]
        pr2 = self.config_provider.get_pr2()  # type: ignore[attr-defined]
        set_robot_state(pr2, [], base_pose, base_type=BaseType.PLANER)

        def robot_updator(robot, q):
            set_robot_state(pr2, config._get_control_joint_names(), q, config.base_type)

        if mode == "static":
            obj: SolutionVisualizerBase = StaticSolutionVisualizer(
                pr2,
                geometry=geometries,
                visualizable=self.world,
                robot_updator=robot_updator,
                show_wireframe=True,
            )
        elif mode == "interactive":
            obj = InteractiveSolutionVisualizer(
                pr2, geometry=geometries, visualizable=self.world, robot_updator=robot_updator
            )
        else:
            assert False

        t = np.array(
            [
                [-0.74452768, 0.59385861, -0.30497620, -0.28438419],
                [-0.66678662, -0.68392597, 0.29604201, 0.80949977],
                [-0.03277405, 0.42376552, 0.90517879, 3.65387983],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        # the below for specifically for visualizing the container contents
        # t = np.array([[-0.00814724,  0.72166326, -0.69219633, -0.01127641],
        #               [-0.99957574,  0.01348003,  0.02581901,  0.06577777],
        #               [ 0.02796346,  0.69211302,  0.72124726,  1.52418492],
        #               [ 0.        ,  0.        ,  0.        ,  1.        ]])

        obj.viewer.camera_transform = t
        return obj
