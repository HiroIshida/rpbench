from abc import abstractmethod
from typing import Any, Literal, Optional, Tuple, Type, TypeVar, overload

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
    CachedPR2ConstProvider,
    CachedRArmFixedPR2ConstProvider,
    CachedRArmPR2ConstProvider,
)
from rpbench.articulated.world.minifridge import MiniFridgeWorld
from rpbench.interface import (
    Problem,
    ResultProtocol,
    TaskExpression,
    TaskWithWorldCondBase,
)
from rpbench.utils import skcoords_to_pose_vec, temp_seed

PR2MiniFridgeTaskT = TypeVar("PR2MiniFridgeTaskT", bound="PR2MiniFridgeTaskBase")


class PR2MiniFridgeTaskBase(
    TaskWithWorldCondBase[MiniFridgeWorld, Tuple[Coordinates, np.ndarray], RobotModel]
):
    @classmethod
    @abstractmethod
    def get_config_provider(cls) -> Type[CachedPR2ConstProvider]:
        ...

    @staticmethod
    def get_world_type() -> Type[MiniFridgeWorld]:
        return MiniFridgeWorld

    @classmethod
    def get_robot_model(cls) -> RobotModel:
        return cls.get_config_provider().get_pr2()

    @classmethod
    def sample_description(
        cls, world: MiniFridgeWorld, standard: bool = False
    ) -> Optional[Tuple[Coordinates, np.ndarray]]:
        provider = cls.get_config_provider()
        config = provider.get_config()

        with temp_seed(0, use_tempseed=standard):
            pose = world.sample_pregrasp_coords()
            if pose is None:
                return None

        colkin = provider.get_whole_body_colkin()
        pr2 = provider.get_pr2()
        sdf = world.get_exact_sdf()
        collfree_const = CollFreeConst(colkin, sdf, pr2)

        if standard:
            base_pos = np.array([-0.1, 0.0, 0.0])
        else:
            base_pos = np.random.randn(3) * np.array([0.1, 0.2, 0.3]) - np.array([-0.05, 0.0, 0.0])
            set_robot_state(pr2, [], base_pos, base_type=BaseType.PLANER)
            colkin.reflect_skrobot_model(pr2)
            q = get_robot_state(pr2, colkin.control_joint_names, base_type=config.base_type)

            if not collfree_const.is_valid(q):
                return None
        return pose, base_pos

    def export_task_expression(self, use_matrix: bool) -> TaskExpression:
        if use_matrix:
            world_vec = np.array([self.world.fridge_conts.fridge.angle])
            world_mat = self.world.fridge_conts.create_heightmap()
        else:
            world_vec = self.world.to_parameter()
            world_mat = None

        target_pose, init_pose = self.description
        other_vec = np.hstack([skcoords_to_pose_vec(target_pose, yaw_only=True), init_pose])
        return TaskExpression(world_vec, world_mat, other_vec)

    @classmethod
    def from_task_param(cls: Type[PR2MiniFridgeTaskT], param: np.ndarray) -> PR2MiniFridgeTaskT:
        world_type = cls.get_world_type()
        world_param_dof = world_type.get_world_dof()
        world_param = param[:world_param_dof]
        world = world_type.from_parameter(world_param)

        other_param = param[world_param_dof:]
        pose_param = other_param[:4]
        ypr = (pose_param[3], 0, 0)
        co = Coordinates(pose_param[:3], ypr)
        base_param = other_param[4:]
        desc = (co, base_param)
        return cls(world, desc)

    def export_problem(self) -> Problem:
        provider = self.get_config_provider()
        config = provider.get_config()
        q_start = provider.get_start_config()

        sdf = self.world.get_exact_sdf()
        ineq_const = provider.get_collfree_const(sdf)

        pr2 = provider.get_pr2()
        motion_step_box = provider.get_config().get_default_motion_step_box()

        target_pose, base_pose = self.description
        if config.base_type == BaseType.PLANER:
            lb = base_pose - np.array([0.5, 0.5, 0.5])
            ub = base_pose + np.array([0.5, 0.5, 0.5])
            base_bound = (tuple(lb), tuple(ub))
            q_start[-3:] = base_pose  # dirty...
        else:
            base_bound = None
        box_const = provider.get_box_const(base_bound=base_bound)

        pose_const = provider.get_pose_const([target_pose])

        # set pr2 to the initial state
        # NOTE: because provider cache's pr2 state and when calling any function
        # it reset the pr2 state to the original state. So the following
        # two lines must be placed here right before reflecting the model
        set_robot_state(pr2, config.get_control_joint_names(), q_start, base_type=config.base_type)
        if config.base_type == BaseType.FIXED:
            set_robot_state(pr2, [], base_pose, base_type=BaseType.PLANER)
        pose_const.reflect_skrobot_model(pr2)
        ineq_const.reflect_skrobot_model(pr2)

        problem = Problem(
            q_start, box_const, pose_const, ineq_const, None, motion_step_box_=motion_step_box
        )
        return problem

    def solve_default(self) -> ResultProtocol:
        problem = self.export_problem()
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
        target_co, base_pose = self.description
        geometries = [Axis.from_coords(target_co)]
        provider = self.get_config_provider()
        config = provider.get_config()
        pr2 = provider.get_pr2()  # type: ignore[attr-defined]
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


class FixedPR2MiniFridgeTask(PR2MiniFridgeTaskBase):
    @classmethod
    def get_config_provider(cls) -> Type[CachedPR2ConstProvider]:
        return CachedRArmFixedPR2ConstProvider


class MovingPR2MiniFridgeTask(PR2MiniFridgeTaskBase):
    @classmethod
    def get_config_provider(cls) -> Type[CachedPR2ConstProvider]:
        return CachedRArmPR2ConstProvider
