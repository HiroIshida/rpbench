import time
from abc import abstractmethod
from typing import ClassVar, Type, TypeVar

import numpy as np
from plainmp.ompl_solver import OMPLSolver, OMPLSolverConfig
from plainmp.problem import Problem
from plainmp.robot_spec import BaseType, PR2LarmSpec
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import rpy_angle
from skrobot.model.primitives import Axis
from skrobot.model.robot_model import RobotModel
from skrobot.models.pr2 import PR2
from skrobot.viewers import PyrenderViewer

from rpbench.articulated.pr2.common import CachedLArmFixedPR2ConstProvider
from rpbench.articulated.vision import create_heightmap_z_slice
from rpbench.articulated.world.jskfridge import JskFridgeWorld, get_fridge_model
from rpbench.interface import ResultProtocol, TaskExpression, TaskWithWorldCondBase


def _prepare_angle_vector():
    pr2 = PR2()
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

    spec = PR2LarmSpec(use_fixed_uuid=True)
    q = np.array([getattr(pr2, name).joint_angle() for name in spec.control_joint_names])
    return pr2.angle_vector(), q


AV_INIT, Q_INIT = _prepare_angle_vector()

DescriptionT = TypeVar("DescriptionT")


class JskFridgeReachingTaskBase(TaskWithWorldCondBase[JskFridgeWorld, np.ndarray, RobotModel]):

    config_provider: ClassVar[
        Type[CachedLArmFixedPR2ConstProvider]
    ] = CachedLArmFixedPR2ConstProvider

    @classmethod
    def get_robot_model(cls) -> RobotModel:
        pass

    @staticmethod
    @abstractmethod
    def sample_pose(world: JskFridgeWorld) -> Coordinates:
        ...

    @classmethod
    def sample_description(cls, world: JskFridgeWorld) -> np.ndarray:
        pose = None
        while True:
            pose = cls.sample_pose(world)
            if pose is not None:
                break
        assert isinstance(pose, Coordinates)
        y, p, r = rpy_angle(pose.worldrot())[0]
        pose = np.hstack([pose.worldpos()[:3], y])

        spec = PR2LarmSpec(base_type=BaseType.PLANAR, use_fixed_uuid=True)
        pr2 = spec.get_robot_model(deepcopy=False)
        pr2.angle_vector(AV_INIT)
        spec.reflect_skrobot_model_to_kin(pr2)

        spec.get_kin()
        cst = spec.create_collision_const()
        sdf = world.get_exact_sdf()
        cst.set_sdf(sdf)

        q = np.zeros(len(spec.control_joint_names) + 3)
        q[:7] = Q_INIT

        while True:
            x = np.random.uniform(-0.65, -0.3)
            y = np.random.uniform(-0.45, -0.1)
            yaw = np.random.uniform(-0.25 * np.pi, 0.25 * np.pi)
            q[7] = x
            q[8] = y
            q[9] = yaw
            if cst.is_valid(q):
                return np.hstack([pose, x, y, yaw])

    def export_task_expression(self, use_matrix: bool) -> TaskExpression:
        other_vec = np.hstack(self.description)
        if use_matrix:
            world_vec = None
            region = get_fridge_model().regions[self.world.attention_region_index]
            obstacles = self.world.get_obstacle_list()
            world_mat = create_heightmap_z_slice(region.box, obstacles, 112)
        else:
            world_vec = self.world.obstacles_param
            world_mat = None
        return TaskExpression(world_vec, world_mat, other_vec)

    @classmethod
    def from_task_param(cls, param: np.ndarray) -> "JskFridgeReachingTaskBase":
        # the last 7 elements for other param
        world_param, other_param = param[:-7], param[-7:]
        world = JskFridgeWorld(world_param)
        description = other_param
        return cls(world, description)

    def export_problem(self) -> Problem:
        spec = PR2LarmSpec(use_fixed_uuid=True)
        pr2 = spec.get_robot_model(deepcopy=False)
        spec.reflect_skrobot_model_to_kin(pr2)
        ineq_cst = spec.create_collision_const()
        sdf = self.world.get_exact_sdf()
        ineq_cst.set_sdf(sdf)

        motion_step_box = np.ones(7) * 0.03
        target_pose, base_pose = self.description[:4], self.description[-3:]
        pos, yaw = target_pose[:3], target_pose[3]
        quat = np.array([0, 0, np.sin(yaw / 2), np.cos(yaw / 2)])
        gripper_cst = spec.create_gripper_pose_const(np.hstack([pos, quat]))

        spec.get_kin().set_base_pose(
            [
                base_pose[0],
                base_pose[1],
                0.0,
                0.0,
                0.0,
                np.sin(base_pose[2] / 2),
                np.cos(base_pose[2] / 2),
            ]
        )

        lb, ub = spec.angle_bounds()
        problem = Problem(Q_INIT, lb, ub, gripper_cst, ineq_cst, None, motion_step_box)
        return problem

    def solve_default(self) -> ResultProtocol:
        problem = self.export_problem()
        conf = OMPLSolverConfig(
            shortcut=True, bspline=True, n_max_call=1000000, timeout=3.0, n_max_ik_trial=1000
        )
        solver = OMPLSolver(conf)
        ret = solver.solve(problem)
        return ret


class JskFridgeReachingTask(JskFridgeReachingTaskBase):
    @staticmethod
    def sample_pose(world: JskFridgeWorld) -> Coordinates:
        return world.sample_pose()

    @staticmethod
    def get_world_type() -> Type[JskFridgeWorld]:
        return JskFridgeWorld


class JskFridgeVerticalReachingTask(JskFridgeReachingTaskBase):
    @staticmethod
    def sample_pose(world: JskFridgeWorld) -> Coordinates:
        return world.sample_pose_vertical()

    @staticmethod
    def get_world_type() -> Type[JskFridgeWorld]:
        return JskFridgeWorld


if __name__ == "__main__":
    # np.random.seed()
    while True:
        print("trial..")
        task = JskFridgeReachingTask.sample()
        ret = task.solve_default()
        if ret.traj is not None:
            break

    print(ret)

    param = task.to_task_param()
    task_again = JskFridgeReachingTask.from_task_param(param)

    v = PyrenderViewer()
    task.world.visualize(v)
    pr2 = PR2(use_tight_joint_limit=False)
    pr2.angle_vector(AV_INIT)
    base_pose = task.description[-3:]
    pr2.translate(np.hstack([base_pose[:2], 0.0]))
    pr2.rotate(base_pose[2], "z")
    v.add(pr2)

    axis = Axis()
    axis.translate(task.description[:3])
    axis.rotate(task.description[3], "z")
    v.add(axis)
    v.show()

    spec = PR2LarmSpec(use_fixed_uuid=True)
    for q in ret.traj.resample(30):
        spec.set_skrobot_model_state(pr2, q)
        time.sleep(0.1)
        v.redraw()

    import time

    time.sleep(1000)
