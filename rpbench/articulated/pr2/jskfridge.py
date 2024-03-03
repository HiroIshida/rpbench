from abc import abstractmethod
from typing import Any, ClassVar, List, Literal, Tuple, Type, TypeVar, overload

import numpy as np
from skmp.constraint import CollFreeConst, IneqCompositeConst
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

from rpbench.articulated.pr2.common import CachedLArmFixedPR2ConstProvider
from rpbench.articulated.world.jskfridge import (
    JskFridgeWorld,
    JskFridgeWorld2,
    JskFridgeWorld3,
    JskFridgeWorldBase,
)
from rpbench.interface import Problem, ResultProtocol, TaskBase, TaskExpression
from rpbench.utils import skcoords_to_pose_vec, temp_seed

DescriptionT = TypeVar("DescriptionT")


class JskFridgeReachingTaskBase(
    TaskBase[JskFridgeWorldBase, Tuple[Coordinates, np.ndarray], RobotModel]
):

    config_provider: ClassVar[
        Type[CachedLArmFixedPR2ConstProvider]
    ] = CachedLArmFixedPR2ConstProvider

    @classmethod
    def get_robot_model(cls) -> RobotModel:
        pr2 = CachedLArmFixedPR2ConstProvider.get_pr2()
        # this configuration hide the arm from kinect so that
        # fridge recognition is easire
        # also, with this configuration, robot can get closer to the fridge
        pr2.torso_lift_joint.joint_angle(0.11444855356985413)
        pr2.r_upper_arm_roll_joint.joint_angle(-1.9933312942796328)
        pr2.r_shoulder_pan_joint.joint_angle(-1.9963322165144708)
        pr2.r_shoulder_lift_joint.joint_angle(1.1966709813458699)
        pr2.r_forearm_roll_joint.joint_angle(9.692626501089645 - 4 * np.pi)
        pr2.r_elbow_flex_joint.joint_angle(-1.8554994146413022)
        pr2.r_wrist_flex_joint.joint_angle(-1.6854605316990736)
        pr2.r_wrist_roll_joint.joint_angle(3.30539700424134 - 2 * np.pi)

        pr2.l_upper_arm_roll_joint.joint_angle(0.6)
        pr2.l_shoulder_pan_joint.joint_angle(+1.5)
        pr2.l_shoulder_lift_joint.joint_angle(-0.3)
        pr2.l_forearm_roll_joint.joint_angle(0.0)
        pr2.l_elbow_flex_joint.joint_angle(-1.8554994146413022)
        pr2.l_wrist_flex_joint.joint_angle(-1.6854605316990736)
        pr2.l_wrist_roll_joint.joint_angle(-3.30539700424134 + 2 * np.pi)

        # so that see the inside of the fridge better
        pr2.head_pan_joint.joint_angle(-0.026808257310632896)
        pr2.head_tilt_joint.joint_angle(0.82)
        return pr2

    @staticmethod
    @abstractmethod
    def sample_pose(world: JskFridgeWorldBase) -> Coordinates:
        ...

    @classmethod
    def sample_descriptions(
        cls, world: JskFridgeWorldBase, n_sample: int, standard: bool = False
    ) -> List[Tuple[Coordinates, np.ndarray]]:

        if standard:
            assert n_sample == 1

        with temp_seed(0, use_tempseed=standard):
            pose_list: List[Coordinates] = []
            while len(pose_list) < n_sample:
                pose = cls.sample_pose(world)
                if pose is not None:
                    pose_list.append(pose)

            base_pos_list: List[np.ndarray] = []
            pr2 = cls.get_robot_model()
            sdf = world.get_exact_sdf()
            colkin = CachedLArmFixedPR2ConstProvider.get_whole_body_colkin()
            collfree_const = CollFreeConst(colkin, sdf, pr2)

            while len(base_pos_list) < n_sample:
                base_pos = np.array(
                    [
                        np.random.uniform(-0.6, -0.4),
                        np.random.uniform(-0.45, -0.1),
                        np.random.uniform(-0.1 * np.pi, 0.1 * np.pi),
                    ]
                )

                set_robot_state(pr2, [], base_pos, base_type=BaseType.PLANER)
                colkin.reflect_skrobot_model(pr2)
                q = get_robot_state(pr2, colkin.control_joint_names)

                if collfree_const.is_valid(q):
                    base_pos_list.append(base_pos)

        descriptions = list(zip(pose_list, base_pos_list))
        return descriptions

    def export_task_expression(self, use_matrix: bool) -> TaskExpression:
        assert use_matrix, "under construction"
        world_vec = None
        world_mat = self.world.heightmap()
        target_pose, init_state = self.description
        other_vec = np.hstack([skcoords_to_pose_vec(target_pose, yaw_only=True), init_state])
        return TaskExpression(world_vec, world_mat, other_vec)

    def export_problem(self) -> Problem:
        pr2 = self.get_robot_model()

        provider = self.config_provider
        q_start = provider.get_start_config()
        box_const = provider.get_box_const()

        sdf = self.world.get_exact_sdf()
        collfree_const = provider.get_collfree_const(sdf)
        selfcollfree_const = provider.get_self_collision_free_const()
        ineq_const = IneqCompositeConst([collfree_const, selfcollfree_const])

        joint_names = provider.get_config()._get_control_joint_names()
        assert joint_names == [
            "l_shoulder_pan_joint",
            "l_shoulder_lift_joint",
            "l_upper_arm_roll_joint",
            "l_elbow_flex_joint",
            "l_forearm_roll_joint",
            "l_wrist_flex_joint",
            "l_wrist_roll_joint",
        ]
        motion_step_box = np.array(
            [
                0.05,  # l_shoulder_pan_joint
                0.05,  # l_shoulder_lift_joint
                0.1,  # l_upper_arm_roll_joint
                0.1,  # l_elbow_flex_joint
                0.2,  # l_forearm_roll_joint
                0.2,  # l_wrist_flex_joint
                0.5,  # l_wrist_roll_joint
            ]
        )

        target_pose, base_pose = self.description
        set_robot_state(pr2, [], base_pose, base_type=BaseType.PLANER)
        pose_const = provider.get_pose_const([target_pose])

        pose_const.reflect_skrobot_model(pr2)
        ineq_const.reflect_skrobot_model(pr2)

        problem = Problem(
            q_start, box_const, pose_const, ineq_const, None, motion_step_box_=motion_step_box
        )
        return problem

    def solve_default_each(self, problem: Problem) -> ResultProtocol:
        solcon = OMPLSolverConfig(n_max_call=40000, n_max_satisfaction_trial=100, simplify=True)

        ompl_solver = OMPLSolver.init(solcon)
        ompl_solver.setup(problem)
        return ompl_solver.solve()

    @overload
    def create_viewer(self, mode: Literal["static"]) -> StaticSolutionVisualizer:
        ...

    @overload
    def create_viewer(self, mode: Literal["interactive"]) -> InteractiveSolutionVisualizer:
        ...

    def create_viewer(self, mode: str, show_wireframe: bool = False) -> Any:
        # copied from rpbench.articulated.pr2.minifridge
        target_co, base_pose = self.description
        geometries = [Axis.from_coords(target_co)]

        config = self.config_provider.get_config()  # type: ignore[attr-defined]
        pr2 = self.get_robot_model()
        set_robot_state(pr2, [], base_pose, base_type=BaseType.PLANER)

        def robot_updator(robot, q):
            set_robot_state(pr2, config._get_control_joint_names(), q, config.base_type)

        if mode == "static":
            obj: SolutionVisualizerBase = StaticSolutionVisualizer(
                pr2,
                geometry=geometries,
                visualizable=self.world,
                robot_updator=robot_updator,
                show_wireframe=show_wireframe,
            )
        elif mode == "interactive":
            colkin = self.config_provider.get_colkin()  # type: ignore[attr-defined]
            sdf = self.world.get_exact_sdf()
            obj = InteractiveSolutionVisualizer(
                pr2,
                geometry=geometries,
                visualizable=self.world,
                robot_updator=robot_updator,
                colkin=colkin,
                sdf=sdf,
            )
        else:
            assert False

        t = np.array(
            [
                [-7.37651916e-02, 4.05048811e-01, -9.11314522e-01, -1.64011619e00],
                [-9.96711581e-01, 7.86646061e-04, 8.10271856e-02, 4.48154882e-01],
                [3.35368471e-02, 9.14294724e-01, 4.03658814e-01, 1.49644830e00],
                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        )

        t = np.array(
            [
                [-0.05759581, 0.62366497, -0.77956701, -1.6860403],
                [-0.99674649, -0.08002414, 0.00962095, 0.19343155],
                [-0.05638393, 0.77758481, 0.62624493, 2.13690459],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        obj.viewer.camera_transform = t
        return obj


class JskFridgeReachingTask(JskFridgeReachingTaskBase):
    @staticmethod
    def sample_pose(world: JskFridgeWorldBase) -> Coordinates:
        return world.sample_pose()

    @staticmethod
    def get_world_type() -> Type[JskFridgeWorldBase]:
        return JskFridgeWorld


class JskFridgeVerticalReachingTask(JskFridgeReachingTaskBase):
    @staticmethod
    def sample_pose(world: JskFridgeWorldBase) -> Coordinates:
        return world.sample_pose_vertical()

    @staticmethod
    def get_world_type() -> Type[JskFridgeWorld]:
        return JskFridgeWorld


class JskFridgeVerticalReachingTask2(JskFridgeReachingTaskBase):
    @staticmethod
    def sample_pose(world: JskFridgeWorldBase) -> Coordinates:
        return world.sample_pose_vertical()

    @staticmethod
    def get_world_type() -> Type[JskFridgeWorldBase]:
        return JskFridgeWorld2


class JskFridgeVerticalReachingTask3(JskFridgeReachingTaskBase):
    @staticmethod
    def sample_pose(world: JskFridgeWorldBase) -> Coordinates:
        return world.sample_pose_vertical()

    @staticmethod
    def get_world_type() -> Type[JskFridgeWorldBase]:
        return JskFridgeWorld3
