from abc import abstractmethod
from typing import (
    Any,
    ClassVar,
    Iterator,
    List,
    Literal,
    Tuple,
    Type,
    TypeVar,
    overload,
)

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
from rpbench.articulated.world.jskfridge import JskFridgeWorld
from rpbench.articulated.world.minifridge import TabletopClutteredFridgeWorld
from rpbench.interface import DescriptionTable, Problem, ResultProtocol, TaskBase
from rpbench.utils import skcoords_to_pose_vec, temp_seed

DescriptionT = TypeVar("DescriptionT")


class JskFridgeReachingTaskBase(
    TaskBase[JskFridgeWorld, Tuple[Coordinates, np.ndarray], RobotModel]
):

    config_provider: ClassVar[
        Type[CachedLArmFixedPR2ConstProvider]
    ] = CachedLArmFixedPR2ConstProvider

    @staticmethod
    def get_world_type() -> Type[JskFridgeWorld]:
        return JskFridgeWorld

    @staticmethod
    def get_robot_model() -> RobotModel:
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

    @classmethod
    def get_dof(cls) -> int:
        return cls.config_provider.get_dof()

    @staticmethod
    def create_cache(world: TabletopClutteredFridgeWorld, robot_model: RobotModel) -> None:
        return None  # do not crete cache

    @staticmethod
    @abstractmethod
    def sample_pose(world: TabletopClutteredFridgeWorld) -> Coordinates:
        ...

    @classmethod
    def sample_descriptions(
        cls, world: JskFridgeWorld, n_sample: int, standard: bool = False
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

    def export_table(self) -> DescriptionTable:
        world_dict = {}  # type: ignore
        world_dict["mesh"] = self.world.heightmap()

        desc_dicts = []
        for desc in self.descriptions:
            desc_dict = {}
            target_pose, init_state = desc
            desc_dict["target_pose"] = skcoords_to_pose_vec(target_pose)
            desc_dict["init_state"] = init_state
            desc_dicts.append(desc_dict)
        return DescriptionTable(world_dict, desc_dicts)

    def export_problems(self) -> Iterator[Problem]:
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

        for target_pose, base_pose in self.descriptions:
            set_robot_state(pr2, [], base_pose, base_type=BaseType.PLANER)
            pose_const = provider.get_pose_const([target_pose])

            pose_const.reflect_skrobot_model(pr2)
            ineq_const.reflect_skrobot_model(pr2)

            problem = Problem(
                q_start, box_const, pose_const, ineq_const, None, motion_step_box_=motion_step_box
            )
            yield problem

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

    def create_viewer(self, mode: str) -> Any:
        # copied from rpbench.articulated.pr2.minifridge
        assert len(self.descriptions) == 1
        target_co, base_pose = self.descriptions[0]
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
                show_wireframe=True,
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
                enable_colvis=True,
                sdf=sdf,
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

        obj.viewer.camera_transform = t
        return obj


class JskFridgeReachingTask(JskFridgeReachingTaskBase):
    @staticmethod
    def sample_pose(world: JskFridgeWorld) -> Coordinates:
        return world.sample_pose()


class JskFridgeVerticalReachingTask(JskFridgeReachingTaskBase):
    @staticmethod
    def sample_pose(world: JskFridgeWorld) -> Coordinates:
        return world.sample_pose_vertical()
