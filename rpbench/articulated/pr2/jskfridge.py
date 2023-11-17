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

from rpbench.articulated.pr2.common import CachedRArmFixedPR2ConstProvider
from rpbench.articulated.world.jskfridge import JskFridgeWorld
from rpbench.articulated.world.minifridge import TabletopClutteredFridgeWorld
from rpbench.interface import DescriptionTable, Problem, ResultProtocol, TaskBase
from rpbench.utils import skcoords_to_pose_vec, temp_seed

DescriptionT = TypeVar("DescriptionT")


class JskFridgeReachingTask(TaskBase[JskFridgeWorld, Tuple[np.ndarray, np.ndarray], RobotModel]):

    config_provider: ClassVar[
        Type[CachedRArmFixedPR2ConstProvider]
    ] = CachedRArmFixedPR2ConstProvider

    @staticmethod
    def get_world_type() -> Type[JskFridgeWorld]:
        return JskFridgeWorld

    @staticmethod
    def get_robot_model() -> RobotModel:
        pr2 = CachedRArmFixedPR2ConstProvider.get_pr2()
        # this configuration hide the arm from kinect so that
        # fridge recognition is easire
        # also, with this configuration, robot can get closer to the fridge
        pr2.r_upper_arm_roll_joint.joint_angle(-1.4842047205551403)
        pr2.r_shoulder_pan_joint.joint_angle(-0.10780237973830653)
        pr2.r_shoulder_lift_joint.joint_angle(1.1548898902227176)
        pr2.r_forearm_roll_joint.joint_angle(0.47479469282041364)
        pr2.r_elbow_flex_joint.joint_angle(-1.976093417223845)
        pr2.r_wrist_flex_joint.joint_angle(-1.254902952073706)
        pr2.r_wrist_roll_joint.joint_angle(6.250323641384378)
        pr2.l_upper_arm_roll_joint.joint_angle(2.124180832597487)
        pr2.l_shoulder_pan_joint.joint_angle(1.6304653141178242)
        pr2.l_shoulder_lift_joint.joint_angle(1.2242577271684127)
        pr2.l_forearm_roll_joint.joint_angle(2.913237145917637)
        pr2.l_elbow_flex_joint.joint_angle(-1.9041423736661933)
        pr2.l_wrist_flex_joint.joint_angle(-1.5152339138194952)
        pr2.l_wrist_roll_joint.joint_angle(-0.5311941899356818)

        # so that see the inside of the fridge better
        pr2.head_tilt_joint.joint_angle(0.9)
        return pr2

    @classmethod
    def get_dof(cls) -> int:
        return cls.config_provider.get_dof()

    @staticmethod
    def create_cache(world: TabletopClutteredFridgeWorld, robot_model: RobotModel) -> None:
        return None  # do not crete cache

    @classmethod
    def sample_descriptions(
        cls, world: JskFridgeWorld, n_sample: int, standard: bool = False
    ) -> List[Tuple[Coordinates, np.ndarray]]:

        if standard:
            assert n_sample == 1

        with temp_seed(0, use_tempseed=standard):
            pose_list: List[Coordinates] = []
            while len(pose_list) < n_sample:
                pose = world.sample_pose()
                if pose is not None:
                    pose_list.append(pose)

            base_pos_list: List[np.ndarray] = []
            pr2 = cls.get_robot_model()
            colkin = CachedRArmFixedPR2ConstProvider.get_colkin()
            sdf = world.get_exact_sdf()
            collfree_const = CollFreeConst(colkin, sdf, pr2)

            while len(base_pos_list) < n_sample:
                base_pos = np.array(
                    [
                        np.random.uniform(-0.6, -0.3),
                        np.random.uniform(-0.1, 0.2),
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
        provider = self.config_provider
        q_start = provider.get_start_config()
        box_const = provider.get_box_const()

        sdf = self.world.get_exact_sdf()
        ineq_const = provider.get_collfree_const(sdf)

        pr2 = self.get_robot_model()

        for target_pose, base_pose in self.descriptions:
            set_robot_state(pr2, [], base_pose, base_type=BaseType.PLANER)
            pose_const = provider.get_pose_const([target_pose])

            pose_const.reflect_skrobot_model(pr2)
            ineq_const.reflect_skrobot_model(pr2)

            problem = Problem(
                q_start, box_const, pose_const, ineq_const, None, motion_step_box_=0.05
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
