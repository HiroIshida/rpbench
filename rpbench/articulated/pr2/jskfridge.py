import time
from abc import abstractmethod
from typing import ClassVar, Optional, Tuple, Type, TypeVar

import numpy as np
from plainmp.constraint import LinkPoseCst, SphereAttachmentSpec, SphereCollisionCst
from plainmp.ik import IKResult
from plainmp.ompl_solver import OMPLSolver as _OMPLSolver
from plainmp.ompl_solver import OMPLSolverConfig
from plainmp.problem import Problem
from plainmp.psdf import CylinderSDF, Pose
from plainmp.robot_spec import BaseType
from plainmp.robot_spec import PR2LarmSpec as _PR2LarmSpec
from plainmp.trajectory import Trajectory
from pr2_ikfast import sample_ik_solution
from skrobot.coordinates import Coordinates, Transform
from skrobot.coordinates.math import quaternion2matrix, rpy_angle, xyzw2wxyz
from skrobot.model.primitives import Axis, Cylinder
from skrobot.model.robot_model import RobotModel
from skrobot.models.pr2 import PR2
from skrobot.viewers import PyrenderViewer

from rpbench.articulated.pr2.pr2_reachability_map.model import load_classifier
from rpbench.articulated.vision import create_heightmap_z_slice_cylinders
from rpbench.articulated.world.jskfridge import JskFridgeWorld, get_fridge_model
from rpbench.interface import ResultProtocol, TaskExpression, TaskWithWorldCondBase


class PR2LarmSpec(_PR2LarmSpec):

    # the original angle bounds from urdf does not consider hardware limits
    def angle_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        mins = np.array(
            [
                -0.5646017957154016,
                -0.3536002157955471,
                -0.6500007560154842,
                -2.121308079458948,
                -np.pi * 2,
                -2.000007696445342,
                -np.pi * 2,
            ]
        )
        maxs = np.array(
            [
                2.1353928865225424,
                1.2962996686874884,
                3.7499969775424966,
                -0.15000005363462504,
                np.pi * 2,
                -0.10000003575641671,
                np.pi * 2,
            ]
        )
        return mins, maxs


class OMPLSolver(_OMPLSolver):
    pass


class OMPLSolver(_OMPLSolver):
    def solve_ik(self, problem: Problem, guess: Optional[Trajectory] = None) -> IKResult:
        # IK is supposed to stop within the timeout but somehow it does not work well
        # so we set...
        assert isinstance(problem.goal_const, LinkPoseCst)
        assert isinstance(problem.global_ineq_const, SphereCollisionCst)
        lb = problem.lb if problem.goal_lb is None else problem.goal_lb
        ub = problem.ub if problem.goal_ub is None else problem.goal_ub

        pose = problem.goal_const.get_desired_poses()[0]
        assert len(pose) == 7
        trans, quat = pose[:3], pose[3:]
        rotmat = quaternion2matrix(xyzw2wxyz(quat))
        tf_desired_to_world = Transform(trans, rotmat)

        kin = PR2LarmSpec(spec_id="rpbench-pr2-jskfridge").get_kin()
        pose = kin.get_base_pose()
        trans_base, quat_base = pose[:3], pose[3:]
        rotmat_base = quaternion2matrix(xyzw2wxyz(quat_base))
        tf_base_to_world = Transform(trans_base, rotmat_base)
        tf_desired_to_base = tf_desired_to_world * tf_base_to_world.inverse_transformation()

        torso_value = 0.11444855356985413  # same as _prepare_angle_vector()

        if guess is not None:
            q_guess = guess._points[-1]
            upper_arm_joint_guess = q_guess[2]

            def sampler():
                yield upper_arm_joint_guess
                for deg_abs in np.linspace(1.0, 40.0, 20):
                    rad_abs = np.pi * deg_abs / 180.0
                    yield upper_arm_joint_guess + rad_abs
                    yield upper_arm_joint_guess - rad_abs

        else:
            sampler = None

        # NOTE: please do not care np.inf, I set it randomly for the value that will not be used later
        if guess is None:
            gen = sample_ik_solution(
                tf_desired_to_base.translation,
                tf_desired_to_base.rotation.tolist(),
                torso_value,
                False,
                sampler,
            )
            for sol in gen:
                if np.all(sol >= lb) and np.all(sol <= ub):
                    if problem.global_ineq_const.is_valid(sol):
                        return IKResult(sol, np.inf, True, np.inf)
        else:
            gen = sample_ik_solution(
                tf_desired_to_base.translation,
                tf_desired_to_base.rotation.tolist(),
                torso_value,
                False,
                sampler,
                batch=True,
            )
            for sols in gen:
                sol_closest = None
                min_dist = np.inf
                for sol in sols:
                    if np.all(sol >= lb) and np.all(sol <= ub):
                        if problem.global_ineq_const.is_valid(sol):
                            dist = np.linalg.norm(sol - q_guess)
                            if dist < min_dist:
                                min_dist = dist
                                sol_closest = sol
                if sol_closest is not None:
                    return IKResult(sol_closest, np.inf, True, np.inf)

        return IKResult(np.zeros(7), np.inf, False, np.inf)


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

    spec = PR2LarmSpec(use_fixed_spec_id=True)
    q = np.array([getattr(pr2, name).joint_angle() for name in spec.control_joint_names])
    return pr2.angle_vector(), q


AV_INIT, Q_INIT = _prepare_angle_vector()

DescriptionT = TypeVar("DescriptionT")

larm_reach_clf = load_classifier("larm")
larm_reach_clf.torso_position = 0.11444855356985413  # same as above  # FIXME: hard coded


def create_cylinder_points(height: float, radius: float, n: int) -> np.ndarray:
    xlin, ylin, zlin = (
        np.linspace(-radius, radius, n),
        np.linspace(-radius, radius, n),
        np.linspace(-0.5 * height, 0.5 * height, n),
    )
    X, Y, Z = np.meshgrid(xlin, ylin, zlin)
    pts = np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T
    cylinder_sdf = CylinderSDF(radius, height, Pose())
    eps = 0.001
    pts_inside = pts[cylinder_sdf.evaluate_batch(pts.T) < eps]
    return pts_inside


def determine_cylinder_height(height: float, offset: float) -> np.ndarray:
    region = get_fridge_model().regions[1]
    D, W, H = region.box.extents
    z_box_lower = region.box.worldpos()[2] - 0.5 * H
    z_cylinder = z_box_lower + 0.5 * height + offset
    return z_cylinder


class JskFridgeReachingTaskBase(TaskWithWorldCondBase[JskFridgeWorld, np.ndarray, RobotModel]):
    @classmethod
    def get_robot_model(cls) -> RobotModel:
        # dummy to pass the abstract method
        pass

    @classmethod
    @abstractmethod
    def is_grasping(cls) -> bool:
        ...

    @classmethod
    @abstractmethod
    def is_grasping(cls) -> bool:
        ...

    def export_task_expression(self, use_matrix: bool) -> TaskExpression:
        other_vec = np.hstack(self.description)
        if use_matrix:
            world_vec = None
            region = get_fridge_model().regions[self.world.attention_region_index]
            cylinders_param = self.world.get_cylinder_params()
            world_mat = create_heightmap_z_slice_cylinders(region.box, cylinders_param, 112)

            # obstacles = self.world.get_obstacle_list()
            # world_mat = create_heightmap_z_slice(region.box, obstacles, 112)
        else:
            n_elem = 1 + (self.world.N_MAX_OBSTACLES * 4)  # 1 for the number of obstacles
            world_vec = np.zeros(n_elem)
            world_vec[0] = len(self.world.obstacles_param) // 4
            world_vec[1 : 1 + len(self.world.obstacles_param)] = self.world.obstacles_param
            world_mat = None
        return TaskExpression(world_vec, world_mat, other_vec)

    @classmethod
    def from_task_param(cls, param: np.ndarray) -> "JskFridgeReachingTaskBase":
        n_other = 12 if cls.is_grasping() else 7
        world_param_all, other_param = param[:-n_other], param[-n_other:]
        n_obstacle = round(world_param_all[0])
        world_param = world_param_all[1 : 1 + n_obstacle * 4]
        world = JskFridgeWorld(world_param)
        description = other_param
        return cls(world, description)

    def export_problem(self) -> Problem:
        if self.is_grasping():
            gripper_width = self.description[0]
            grasp_cylinder_param = self.description[1:5]
            target_pose, base_pose = self.description[5:9], self.description[-3:]
        else:
            grasp_cylinder_param = None
            target_pose, base_pose = self.description[:4], self.description[-3:]

        spec = PR2LarmSpec(spec_id="rpbench-pr2-jskfridge")
        pr2 = spec.get_robot_model(deepcopy=False)
        pr2.angle_vector(AV_INIT)

        if self.is_grasping():
            pr2.l_gripper_l_finger_joint.joint_angle(gripper_width)

        attachments = tuple()
        if self.is_grasping():
            # cylinder
            x_relative, y_relative, h, r = grasp_cylinder_param
            z_cylinder = determine_cylinder_height(h, self.eps)
            z_relative = z_cylinder - target_pose[2]
            pts = create_cylinder_points(h, r, 8) + np.array([x_relative, y_relative, z_relative])
            radii = np.ones(len(pts)) * 0.005
            attachment = SphereAttachmentSpec("l_gripper_tool_frame", pts.T, radii, False)
            attachments = (attachment,)

        spec.reflect_skrobot_model_to_kin(pr2)
        ineq_cst = spec.create_collision_const(attachments=attachments)
        sdf = self.world.get_exact_sdf()
        ineq_cst.set_sdf(sdf)

        motion_step_box = np.ones(7) * 0.03
        pos, yaw = target_pose[:3], target_pose[3]
        quat = np.array([0, 0, np.sin(yaw / 2), np.cos(yaw / 2)])
        gripper_cst = spec.create_gripper_pose_const(np.hstack([pos, quat]))

        def create_ik_cst(x_eps, y_eps, yaw_eps) -> SphereCollisionCst:
            yaw_now = target_pose[3]
            rotmat = np.array(
                [[np.cos(yaw_now), -np.sin(yaw_now)], [np.sin(yaw_now), np.cos(yaw_now)]]
            )
            pos2d = target_pose[:2] + np.dot(rotmat, np.array([x_eps, y_eps]))
            pos = np.hstack([pos2d, target_pose[2]])
            yaw = target_pose[3] + yaw_eps
            quat = np.array([0, 0, np.sin(yaw / 2), np.cos(yaw / 2)])
            return spec.create_gripper_pose_const(np.hstack([pos, quat]))

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
        # problem.post_ik_goal_eq_consts = [
        #     create_ik_cst(0.075, 0.0, 0.0),
        #     create_ik_cst(0.15, 0.0, 0.0),
        # ]
        # ineq_cst2 = spec.create_collision_const(use_cache=False)
        # ineq_cst2.set_sdf(get_fridge_model_sdf())
        # problem.post_ik_goal_ineq_const = ineq_cst2
        return problem

    @staticmethod
    def get_world_type() -> Type[JskFridgeWorld]:
        return JskFridgeWorld

    @classmethod
    def sample_pose(
        cls,
        world,
        pr2_pose: np.ndarray,
        grasping_cylinder_param: Optional[np.ndarray],
    ) -> Coordinates:
        larm_reach_clf.set_base_pose(pr2_pose)

        pts_inside_local = None
        if grasping_cylinder_param is not None:
            pts_inside_local = create_cylinder_points(
                grasping_cylinder_param[2], grasping_cylinder_param[3], 8
            )

        region = get_fridge_model().regions[1]
        D, W, H = region.box.extents
        horizontal_margin = 0.08
        depth_margin = 0.03
        width_effective = np.array([D - 2 * depth_margin, W - 2 * horizontal_margin])
        sdf = world.get_exact_sdf()

        n_max_trial = 100
        for _ in range(n_max_trial):
            trans_lb = -0.5 * width_effective
            trans_lb[
                0
            ] -= 0.05  # because gripper position when grasping the object inside the region must be smaller than object's x position
            trans_ub = 0.5 * width_effective
            trans = np.random.uniform(trans_lb, trans_ub)

            trans = np.hstack([trans, -0.5 * H + 0.09])
            co = region.box.copy_worldcoords()
            co.translate(trans)
            if sdf.evaluate(co.worldpos()) < 0.02:
                continue
            co.rotate(np.random.uniform(-(1.0 / 4.0) * np.pi, (1.0 / 4.0) * np.pi), "z")
            if not larm_reach_clf.predict(co):
                continue
            co_dummy = co.copy_worldcoords()
            co_dummy.translate([-0.07, 0.0, 0.0])

            if sdf.evaluate(co_dummy.worldpos()) < 0.04:
                continue
            co_dummy.translate([-0.07, 0.0, 0.0])
            if sdf.evaluate(co_dummy.worldpos()) < 0.04:
                continue

            if grasping_cylinder_param is not None:
                # cylinder pose
                co_cylinder_center = co.copy_worldcoords()
                z = determine_cylinder_height(grasping_cylinder_param[2], cls.eps)
                z_now = co_cylinder_center.worldpos()[2]
                z_trans = z - z_now
                co_cylinder_center.translate(
                    [grasping_cylinder_param[0], grasping_cylinder_param[1], z_trans]
                )

                # translate the cylinder points
                pts_inside = pts_inside_local + co_cylinder_center.worldpos()
                if np.any(sdf.evaluate_batch(pts_inside.T) < 0.0):
                    continue
            return co
        return co  # invalid one but no choice

    @classmethod
    def sample_description(cls, world: JskFridgeWorld) -> np.ndarray:
        grasp_cylinder_param = None
        gripper_width = None
        if cls.is_grasping():
            gripper_width = np.random.uniform(0.0, 0.548)
            grasp_cylinder_param = cls.sample_grasp_cylinder_param()

        spec = PR2LarmSpec(base_type=BaseType.PLANAR, use_fixed_spec_id=True)
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
            x = np.random.uniform(-0.6, -0.3)
            y = np.random.uniform(-0.3, +0.2)
            yaw = np.random.uniform(-0.5 * np.pi, 0.25 * np.pi)
            q[7] = x
            q[8] = y
            q[9] = yaw
            co_reach = cls.sample_pose(world, q[7:10], grasp_cylinder_param)
            assert isinstance(co_reach, Coordinates)
            ypr = rpy_angle(co_reach.worldrot())[0]
            pose = np.hstack([co_reach.worldpos()[:3], ypr[0]])

            if cst.is_valid(q):
                if cls.is_grasping():
                    return np.hstack([gripper_width, grasp_cylinder_param, pose, x, y, yaw])
                else:
                    return np.hstack([pose, x, y, yaw])


class JskFridgeReachingTask(JskFridgeReachingTaskBase):
    @classmethod
    def is_grasping(cls) -> bool:
        return False

    def solve_default(self) -> ResultProtocol:
        problem = self.export_problem()
        conf = OMPLSolverConfig(
            shortcut=True, bspline=True, n_max_call=1000000, timeout=5.0, n_max_ik_trial=1000
        )
        solver = OMPLSolver(conf)
        ret = solver.solve(problem)
        return ret


class JskFridgeGraspingReachingTask(JskFridgeReachingTaskBase):
    eps: ClassVar[float] = 0.025  # to avoid collision between grasping object and the fridge

    def visualize(self) -> Tuple[PyrenderViewer, PR2]:
        v = PyrenderViewer()
        self.world.visualize(v)
        gripper_width, grasp_cylinder_param, target_pose, base_pose = (
            self.description[0],
            self.description[1:5],
            self.description[5:9],
            self.description[-3:],
        )
        pr2 = PR2(use_tight_joint_limit=False)
        pr2.angle_vector(AV_INIT)
        pr2.l_gripper_l_finger_joint.joint_angle(gripper_width)
        z_cylinder = determine_cylinder_height(grasp_cylinder_param[2], self.eps)
        z_cylinder_offset = z_cylinder - target_pose[2]
        pos_relative = np.array(
            [grasp_cylinder_param[0], grasp_cylinder_param[1], z_cylinder_offset]
        )

        cylinder = Cylinder(grasp_cylinder_param[3], grasp_cylinder_param[2])
        cylinder.translate(pos_relative)
        pr2.l_gripper_tool_frame.assoc(cylinder, "local")

        x, y, yaw = base_pose
        pr2.translate(np.hstack([x, y, 0.0]))
        pr2.rotate(yaw, "z")

        v.add(pr2)
        v.add(cylinder)

        axis = Axis()
        axis.translate(target_pose[:3])
        axis.rotate(target_pose[3], "z")
        v.add(axis)
        return v, pr2

    @classmethod
    def is_grasping(cls) -> bool:
        return True

    @classmethod
    def sample_grasp_cylinder_param(cls) -> np.ndarray:
        while True:
            # assuming that robot grasping a cylinder
            region = get_fridge_model().regions[1]
            D, W, H = region.box.extents
            grasping_cylinder_height = np.random.uniform(0.1, 0.15)
            grasping_cylinder_radius = np.random.uniform(0.02, 0.05)

            # xy position relative to the gripper (sample from box and reject)
            # note that z is automatically determined by the height of the cylinder
            # assuming that the bottom of the cylinder is cls.eps above the ground
            sampling_radius = grasping_cylinder_radius + 0.02
            x = np.random.uniform(-0.02, sampling_radius)
            y = np.random.uniform(-sampling_radius, sampling_radius)
            dist = np.sqrt(x**2 + y**2)
            if dist < sampling_radius:
                return np.array([x, y, grasping_cylinder_height, grasping_cylinder_radius])

    def solve_default(self) -> ResultProtocol:
        problem = self.export_problem()
        conf = OMPLSolverConfig(
            shortcut=True, bspline=True, n_max_call=10000000, timeout=10.0, n_max_ik_trial=100000
        )
        solver = OMPLSolver(conf)
        ret = solver.solve(problem)
        return ret


if __name__ == "__main__":
    import tqdm

    task = JskFridgeGraspingReachingTask.sample()

    for _ in tqdm.tqdm(range(200)):
        task = JskFridgeReachingTask.sample()
        param = task.to_task_param()
        task_again = JskFridgeReachingTask.from_task_param(param)
        param_again = task_again.to_task_param()
        assert np.allclose(param, param_again)

    while True:
        print("trial..")
        task = JskFridgeReachingTask.sample()
        ts = time.time()
        ret = task.solve_default()
        print(time.time() - ts)
        if ret.traj is not None:
            break

    expression = task.export_task_expression(True)
    import matplotlib.pyplot as plt

    plt.imshow(expression.world_mat)
    plt.show()

    param = task.to_task_param()
    task_again = JskFridgeReachingTask.from_task_param(param)
    param_again = task_again.to_task_param()
    task = task_again

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

    spec = PR2LarmSpec(use_fixed_spec_id=True)
    for q in ret.traj.resample(30):
        spec.set_skrobot_model_state(pr2, q)
        time.sleep(0.1)
        v.redraw()

    import time

    time.sleep(1000)
