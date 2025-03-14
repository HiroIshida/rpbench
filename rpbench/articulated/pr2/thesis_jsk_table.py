import time
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, List, Optional, Tuple, Union

import numpy as np
from plainmp.ompl_solver import OMPLSolver, OMPLSolverConfig, Problem
from plainmp.problem import Problem
from plainmp.psdf import BoxSDF, Pose, UnionSDF
from plainmp.robot_spec import Coordinates, PR2BaseOnlySpec, PR2LarmSpec, PR2RarmSpec
from skrobot.coordinates import CascadedCoords
from skrobot.coordinates.math import rpy_angle
from skrobot.model.primitives import Axis, Box
from skrobot.models.pr2 import PR2
from skrobot.viewers import PyrenderViewer, TrimeshSceneViewer

from rpbench.articulated.vision import create_heightmap_z_slice
from rpbench.articulated.world.utils import BoxSkeleton
from rpbench.interface import ResultProtocol, TaskBase, TaskExpression
from rpbench.planer_box_utils import Box2d, PlanerCoords, is_colliding, sample_box
from rpbench.utils import SceneWrapper


def fit_radian(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


def define_av_init() -> Tuple[np.ndarray, np.ndarray]:
    # called only once
    model = PR2()
    model.reset_manip_pose()
    model.r_shoulder_pan_joint.joint_angle(-1.94)
    model.r_shoulder_lift_joint.joint_angle(1.23)
    model.r_upper_arm_roll_joint.joint_angle(-2.0)
    model.r_elbow_flex_joint.joint_angle(-1.02)
    model.r_forearm_roll_joint.joint_angle(-1.03)
    model.r_wrist_flex_joint.joint_angle(-1.18)
    model.r_wrist_roll_joint.joint_angle(-0.78)

    model.l_shoulder_pan_joint.joint_angle(+1.94)
    model.l_shoulder_lift_joint.joint_angle(1.23)
    model.l_upper_arm_roll_joint.joint_angle(+2.0)
    model.l_elbow_flex_joint.joint_angle(-1.02)
    model.l_forearm_roll_joint.joint_angle(+1.03)
    model.l_wrist_flex_joint.joint_angle(-1.08)
    model.l_wrist_roll_joint.joint_angle(0.78)

    larm_spec = PR2LarmSpec()
    larm_init_angles = [getattr(model, jn).joint_angle() for jn in larm_spec.control_joint_names]
    rarm_spec = PR2RarmSpec()
    rarm_init_angles = [getattr(model, jn).joint_angle() for jn in rarm_spec.control_joint_names]
    return model.angle_vector(), larm_init_angles, rarm_init_angles


DARKRED_COLOR = (139, 0, 0, 200)
BROWN_COLOR = (204, 102, 0, 200)
DARKGREEN_COLOR = (0, 100, 0, 200)
AV_INIT, LARM_INIT_ANGLES, RARM_INIT_ANGLES = define_av_init()


class JskChair(CascadedCoords):
    DEPTH: ClassVar[float] = 0.56
    WIDTH: ClassVar[float] = 0.5
    SEAT_HEIGHT: ClassVar[float] = 0.42
    BACK_HEIGHT: ClassVar[float] = 0.8
    chair_primitives: List[BoxSkeleton]

    def __init__(self):
        super().__init__()
        base = BoxSkeleton(
            [self.DEPTH, self.WIDTH, self.SEAT_HEIGHT], pos=[0, 0, 0.5 * self.SEAT_HEIGHT]
        )
        back = BoxSkeleton(
            [0.1, self.WIDTH, self.BACK_HEIGHT],
            pos=[-0.5 * self.DEPTH + 0.05, 0, 0.5 * self.BACK_HEIGHT],
        )
        primitive_list = [base, back]
        for prim in primitive_list:
            self.assoc(prim)
        self.chair_primitives = primitive_list

    def visualize(self, viewer: Union[TrimeshSceneViewer, SceneWrapper]) -> List[Box]:
        handles = [p.to_visualizable(DARKRED_COLOR) for p in self.chair_primitives]
        for h in handles:
            viewer.add(h)
        return handles

    def create_sdf(self) -> UnionSDF:
        return UnionSDF([p.to_plainmp_sdf() for p in self.chair_primitives])


_chair_origin_sdf: UnionSDF = JskChair().create_sdf()


def get_chair_origin_sdf() -> UnionSDF:
    return _chair_origin_sdf.clone()


class JskTable(CascadedCoords):
    size: List[float]
    table_primitives: List[BoxSkeleton]
    TABLE_HEIGHT: ClassVar[float] = 0.7
    TABLE_DEPTH: ClassVar[float] = 0.8
    TABLE_WIDTH: ClassVar[float] = 1.24
    DIST_FROM_FRIDGESIDE_WALL: ClassVar[float] = 1.32
    DIST_FROM_DOORSIDE_WALL: ClassVar[float] = 0.37

    def __init__(self):
        super().__init__()
        plate_d = 0.05
        size = [self.TABLE_DEPTH, self.TABLE_WIDTH, plate_d]
        height = self.TABLE_HEIGHT
        plate = BoxSkeleton(size, pos=[0, 0, height - 0.5 * plate_d])
        leg_width = 0.1
        leg1 = BoxSkeleton(
            [leg_width, leg_width, height],
            pos=[0.5 * size[0] - 0.5 * leg_width, 0.5 * size[1] - 0.5 * leg_width, height * 0.5],
        )
        leg2 = BoxSkeleton(
            [leg_width, leg_width, height],
            pos=[0.5 * size[0] - 0.5 * leg_width, -0.5 * size[1] + 0.5 * leg_width, height * 0.5],
        )
        leg3 = BoxSkeleton(
            [leg_width, leg_width, height],
            pos=[-0.5 * size[0] + 0.5 * leg_width, -0.5 * size[1] + 0.5 * leg_width, height * 0.5],
        )
        leg4 = BoxSkeleton(
            [leg_width, leg_width, height],
            pos=[-0.5 * size[0] + 0.5 * leg_width, 0.5 * size[1] - 0.5 * leg_width, height * 0.5],
        )

        # define fridge-side wall
        fridge_side_wall = BoxSkeleton(
            [1.0, 3.0, 2.0],
            pos=[
                -0.5 * self.TABLE_DEPTH - self.DIST_FROM_FRIDGESIDE_WALL - 0.5,
                -0.5 * self.TABLE_WIDTH + 1.5,
                1.0,
            ],
        )

        # define door-side wall 0.37 is the distance between the wall and table
        door_side_wall = BoxSkeleton(
            [0.48, self.TABLE_WIDTH + 0.1, 1.25],
            pos=[
                0.5 * self.TABLE_DEPTH + self.DIST_FROM_DOORSIDE_WALL + 0.5 * 0.48,
                -0.5 * self.TABLE_WIDTH + 0.5 * (self.TABLE_WIDTH + 0.1),
                0.625,
            ],
        )

        # define the TV-side wall
        d_tv_side_wall = (
            1.0
            + 0.48
            + self.DIST_FROM_DOORSIDE_WALL
            + self.DIST_FROM_FRIDGESIDE_WALL
            + self.TABLE_DEPTH
        )
        tv_side_wall = BoxSkeleton(
            [d_tv_side_wall, 1.0, self.TABLE_HEIGHT + 0.1],
            pos=[-0.73, -0.5 * self.TABLE_WIDTH - 0.5, 0.5 * self.TABLE_HEIGHT + 0.05],
        )

        primitive_list = [
            plate,
            leg1,
            leg2,
            leg3,
            leg4,
            fridge_side_wall,
            door_side_wall,
            tv_side_wall,
        ]

        # the table's coordinates equals to the center of the plate
        co_surface_center = plate.copy_worldcoords()
        co_surface_center.translate([0, 0, 0.5 * plate_d])
        self.newcoords(co_surface_center)
        for prim in primitive_list:
            self.assoc(prim)

        self.size = size
        self.table_primitives = primitive_list

    def visualize(self, viewer: Union[TrimeshSceneViewer, SceneWrapper]) -> None:
        for prim in self.table_primitives[:-3]:
            viewer.add(prim.to_visualizable(BROWN_COLOR))
        for prim in self.table_primitives[-3:]:
            viewer.add(prim.to_visualizable((200, 200, 200, 255)))

    def create_sdf(self) -> UnionSDF:
        return UnionSDF([p.to_plainmp_sdf() for p in self.table_primitives])

    def translate(self, *args, **kwargs):
        raise NotImplementedError("This method is deleted")

    def rotate(self, *args, **kwargs):
        raise NotImplementedError("This method is deleted")


_jsk_table_sdf = JskTable().create_sdf()  # always the same. So, pre-compute it.


def get_jsk_table_sdf() -> UnionSDF:
    return _jsk_table_sdf.clone()


@dataclass
class JskMessyTableTaskBase(TaskBase):
    obstacles_param: np.ndarray  # (6 * n_obstacle,)
    chairs_param: np.ndarray  # (3 * n_chair,)
    pr2_coords: np.ndarray
    reaching_pose: np.ndarray
    N_MAX_OBSTACLE: ClassVar[int] = 10
    N_MAX_CHAIR: ClassVar[int] = 5
    OBSTACLE_W_MIN: ClassVar[float] = 0.05
    OBSTACLE_W_MAX: ClassVar[float] = 0.25
    OBSTACLE_H_MIN: ClassVar[float] = 0.05
    OBSTACLE_H_MAX: ClassVar[float] = 0.35

    # abstract override
    @classmethod
    def from_task_param(cls, param: np.ndarray) -> "JskMessyTableTask":
        # deserialize obstacles
        head = 0
        bbox_param_list = []
        for _ in range(cls.N_MAX_OBSTACLE):
            bbox_param = param[head : head + 6]
            head += 6
            if np.isnan(bbox_param).any():
                continue
            bbox_param_list.append(bbox_param)
        if len(bbox_param_list) > 0:
            obstacles_param = np.hstack(bbox_param_list)
        else:
            obstacles_param = np.empty(0)

        # deserialize chairs
        chair_param_list = []
        for _ in range(cls.N_MAX_CHAIR):
            chair_param = param[head : head + 3]
            head += 3
            if np.isnan(chair_param).any():
                continue
            chair_param_list.append(chair_param)
        if len(chair_param_list) > 0:
            chairs_param = np.hstack(chair_param_list)
        else:
            chairs_param = np.empty(0)

        # deserialize pr2_coords and reaching_pose
        pr2_coords = param[head : head + 3]
        head += 3
        reaching_pose = param[head : head + 4]
        return cls(obstacles_param, chairs_param, pr2_coords, reaching_pose)

    # abstract override
    def export_task_expression(self, use_matrix: bool) -> TaskExpression:
        if use_matrix:
            world_vec = None
            obstacle_env_region = BoxSkeleton(
                [
                    JskTable.TABLE_DEPTH,
                    JskTable.TABLE_WIDTH,
                    JskMessyTableTask.OBSTACLE_H_MAX + 0.05,
                ]
            )
            obstacle_env_region.translate(
                [0.0, 0.0, JskTable.TABLE_HEIGHT + obstacle_env_region.extents[2] * 0.5]
            )

            obstacles = []
            for obs_param in self.obstacles_param.reshape(-1, 6):
                x, y, yaw, d, w, h = obs_param
                box = BoxSkeleton([d, w, h], pos=[x, y, JskTable.TABLE_HEIGHT + 0.5 * h])
                box.rotate(yaw, "z")
                obstacles.append(box)
            table_mat = create_heightmap_z_slice(obstacle_env_region, obstacles, 112)
            other_vec = np.hstack([self.pr2_coords, self.reaching_pose])
            if self.consider_chair():
                region, _ = self._prepare_target_region()
                primitives = []
                for chair_param in self.chairs_param.reshape(-1, 3):
                    x, y, yaw = chair_param
                    chair = JskChair()
                    chair.translate([x, y, 0.0])
                    chair.rotate(yaw, "z")
                    primitives.extend(chair.chair_primitives)
                chair_mat = create_heightmap_z_slice(region, primitives, 112)
                world_mat = np.stack([table_mat, chair_mat], axis=0)
            else:
                world_mat = table_mat
            return TaskExpression(world_vec, world_mat, other_vec)
        else:
            # param filled with nan
            world_vec = np.full(6 * self.N_MAX_OBSTACLE + 3 * self.N_MAX_CHAIR, np.nan)
            head = 0
            world_vec[head : head + self.obstacles_param.size] = self.obstacles_param
            head += self.N_MAX_OBSTACLE * 6
            world_vec[head : head + self.chairs_param.size] = self.chairs_param
            other_vec = np.hstack([self.pr2_coords, self.reaching_pose])
            return TaskExpression(world_vec, None, other_vec)

    # abstract override
    def solve_default(self) -> ResultProtocol:
        problem = self.export_problem()
        conf = OMPLSolverConfig(
            shortcut=True, bspline=True, n_max_call=1000000, timeout=3.0, n_max_ik_trial=1000
        )
        solver = OMPLSolver(conf)
        ret = solver.solve(problem)
        return ret

    def get_total_sdf(self) -> UnionSDF:
        total_sdf = get_jsk_table_sdf()
        if len(self.chairs_param) > 0:
            for chair_param in self.chairs_param.reshape(-1, 3):
                x, y, yaw = chair_param
                chair_sdf = get_chair_origin_sdf()
                chair_sdf.translate([x, y, 0.0])
                chair_sdf.rotate_z(yaw)
                total_sdf.merge(chair_sdf)

        if len(self.obstacles_param) > 0:
            for obs_param in self.obstacles_param.reshape(-1, 6):
                x, y, yaw, d, w, h = obs_param
                box_sdf = BoxSDF([d, w, h], Pose())
                box_sdf.translate([x, y, JskTable.TABLE_HEIGHT + 0.5 * h])
                box_sdf.rotate_z(yaw)
                total_sdf.add(box_sdf)
        return total_sdf

    def is_using_rarm(self) -> bool:
        yaw_robot = self.pr2_coords[2]
        y_axis = np.array([-np.sin(yaw_robot), np.cos(yaw_robot)])
        vec_robot_to_reach = self.reaching_pose[:2] - self.pr2_coords[:2]
        right_side = np.dot(y_axis, vec_robot_to_reach) < 0
        return right_side

    # abstract override
    def export_problem(self) -> Problem:
        right_side = self.is_using_rarm()
        if right_side:
            spec = PR2RarmSpec(use_fixed_uuid=True)
            elbow_name = "r_elbow_flex_link"
            q_init = RARM_INIT_ANGLES
        else:
            spec = PR2LarmSpec(use_fixed_uuid=True)
            elbow_name = "l_elbow_flex_link"
            q_init = LARM_INIT_ANGLES

        self._spec_cache = spec  # for debugging

        pr2 = spec.get_robot_model(deepcopy=False)
        pr2.angle_vector(AV_INIT)
        x_pos, y_pos, yaw = self.pr2_coords
        pr2.newcoords(Coordinates([x_pos, y_pos, 0], [yaw, 0, 0]))
        spec.reflect_skrobot_model_to_kin(pr2)

        x, y, z, yaw = self.reaching_pose
        eq_cst = spec.create_gripper_pose_const(np.array([x, y, z, 0, 0, yaw]))

        # also we impose a constraint that the robot's elbow never be higher than
        # certain height, to assure that the robot gripper is visible from the camera
        min_elbow_height = JskTable.TABLE_HEIGHT + 0.05
        max_elbow_height = JskTable.TABLE_HEIGHT + 0.35
        elbow_cst = spec.create_position_bound_const(
            elbow_name, 2, min_elbow_height, max_elbow_height
        )

        coll_cst = spec.create_collision_const()
        coll_cst.set_sdf(self.get_total_sdf())

        lb, ub = spec.angle_bounds()

        motion_step_box = np.array([0.03] * 7)
        problem = Problem(q_init, lb, ub, eq_cst, coll_cst, None, motion_step_box)
        problem.goal_ineq_const = elbow_cst
        # narrow down the goal bounds for the later manipulatability
        problem.goal_lb = lb + 0.3
        problem.goal_ub = ub - 0.3
        return problem

    # abstract override
    @classmethod
    def sample(
        cls, predicate: Optional[Any] = None, timeout: int = 180
    ) -> Optional["JskMessyTableTask"]:
        t_start = time.time()

        table = JskTable()
        obstacles, obstacle_env_region = cls._sample_obstacles_on_table(table)
        while True:
            elapsed = time.time() - t_start
            if elapsed > timeout:
                return None
            reaching_pose = cls._sample_reaching_pose(obstacles)
            target_region, table_box2d_wrt_region = cls._prepare_target_region()
            pr2_coords = cls._sample_robot(table, target_region, reaching_pose)
            if pr2_coords is None:
                continue

            if cls.consider_chair():
                chair_list = cls._sample_chairs(
                    table, target_region, table_box2d_wrt_region, pr2_coords
                )
            else:
                chair_list = []

            # serialize obstacles
            obstacle_param_list = []
            for obstacle in obstacles:
                pos = obstacle.worldpos()
                rot = obstacle.worldrot()
                ypr = rpy_angle(rot)[0]
                obstacle_param = np.hstack([pos[0], pos[1], ypr[0], obstacle.extents])
                obstacle_param_list.append(obstacle_param)
            if len(obstacle_param_list) > 0:
                obstacles_param = np.hstack(obstacle_param_list)
            else:
                obstacles_param = np.empty(0)

            # serialize chairs
            chair_param_list = []
            for chair in chair_list:
                pos = chair.worldpos()
                rot = chair.worldrot()
                ypr = rpy_angle(rot)[0]
                chair_param = np.array([pos[0], pos[1], ypr[0]])
                chair_param_list.append(chair_param)
            if len(chair_param_list) > 0:
                chairs_param = np.hstack(chair_param_list)
            else:
                chairs_param = np.empty(0)
            task = cls(obstacles_param, chairs_param, pr2_coords, reaching_pose)
            if predicate is None:
                return task
            if predicate(task):
                return task

    @staticmethod
    def _sample_obstacles_on_table(table: JskTable) -> Tuple[List[BoxSkeleton], BoxSkeleton]:
        region_size = [table.size[0], table.size[1], JskMessyTableTask.OBSTACLE_H_MAX + 0.05]
        obstacle_env_region = BoxSkeleton(region_size)
        table.assoc(obstacle_env_region, relative_coords="local")
        obstacle_env_region.translate([0.0, 0.0, region_size[2] * 0.5])

        n_min, n_max = 5, JskMessyTableTask.N_MAX_OBSTACLE
        n_obstacle = np.random.randint(n_min, n_max)
        region2d = Box2d(np.array(table.size[:2]), PlanerCoords.standard())
        obj2d_list = []
        for _ in range(n_obstacle):
            w = np.random.uniform(
                JskMessyTableTask.OBSTACLE_W_MIN, JskMessyTableTask.OBSTACLE_W_MAX
            )
            d = np.random.uniform(
                JskMessyTableTask.OBSTACLE_W_MIN, JskMessyTableTask.OBSTACLE_W_MAX
            )
            yaw = np.random.uniform(0.0, np.pi)
            center = region2d.sample_point()
            obj2d = Box2d(np.array([w, d]), PlanerCoords(center, yaw))
            if not region2d.contains(obj2d) or any(is_colliding(obj2d, o) for o in obj2d_list):
                continue
            obj2d_list.append(obj2d)

        obj_list = []
        for obj2d in obj2d_list:
            h = np.random.uniform(
                JskMessyTableTask.OBSTACLE_H_MIN, JskMessyTableTask.OBSTACLE_H_MAX
            )
            extent = np.hstack([obj2d.extent, h])
            obj = BoxSkeleton(extent, pos=np.hstack([obj2d.coords.pos, 0.0]))
            obj.rotate(obj2d.coords.angle, "z")
            obj.translate([0.0, 0.0, 0.5 * h - region_size[2] * 0.5])
            obj_list.append(obj)
            obstacle_env_region.assoc(obj, relative_coords="local")
        return obj_list, obstacle_env_region

    @staticmethod
    def _sample_reaching_pose(obstacles: List[BoxSkeleton]) -> np.ndarray:
        co2d = PlanerCoords(
            np.array([JskTable.TABLE_DEPTH * 0.5, -JskTable.TABLE_WIDTH * 0.5]), 0.0
        )
        large_box2d = Box2d(
            np.array([JskTable.TABLE_DEPTH * 2, JskTable.TABLE_WIDTH * 2]), co2d
        )  # hypothetical large box

        obstacle_sdf = UnionSDF([o.to_plainmp_sdf() for o in obstacles])

        while True:
            x = np.random.uniform(-JskTable.TABLE_DEPTH * 0.5, JskTable.TABLE_DEPTH * 0.5)
            y = np.random.uniform(-JskTable.TABLE_WIDTH * 0.5, JskTable.TABLE_WIDTH * 0.5)
            z = np.random.uniform(0.05, 0.2) + JskTable.TABLE_HEIGHT
            eps = 1e-3
            pts = np.array([[x, y], [x + eps, y], [x, y + eps]])
            sd, sd_pdx, sd_pdy = large_box2d.sd(pts)
            if sd < -0.5:
                continue  # too far and less likely to be reachable

            grad = np.array([sd_pdx - sd, sd_pdy - sd])
            yaw_center = fit_radian(np.arctan2(grad[1], grad[0]) + np.pi)
            yaw = fit_radian(yaw_center + np.random.uniform(-np.pi / 4, np.pi / 4))

            cand = np.array([x, y, z, yaw])

            # check if the candidate is colliding with obstacles
            pos = cand[:3]
            if obstacle_sdf.evaluate(pos) < 0.03:
                continue
            pos_slided = pos - np.array([np.cos(yaw), np.sin(yaw), 0.0]) * 0.1
            if obstacle_sdf.evaluate(pos_slided) < 0.03:
                continue
            return cand

    @staticmethod
    def _prepare_target_region() -> Tuple[BoxSkeleton, Box2d]:
        target_region = BoxSkeleton(
            [
                JskTable.TABLE_DEPTH
                + JskTable.DIST_FROM_DOORSIDE_WALL
                + JskTable.DIST_FROM_FRIDGESIDE_WALL,
                JskTable.TABLE_WIDTH + 1.2,
                1.0,
            ],
            pos=[-0.46, 0.56, 0.5],  # hand-tuned to fit well
        )
        table_pos_from_region = -target_region.worldpos()[:2]
        table_box2d_wrt_region = Box2d(
            np.array([JskTable.TABLE_DEPTH, JskTable.TABLE_WIDTH]),
            PlanerCoords(table_pos_from_region, 0.0),
        )
        return target_region, table_box2d_wrt_region

    @staticmethod
    def _sample_robot(
        table: JskTable, target_region: BoxSkeleton, reaching_pose: np.ndarray
    ) -> Optional[np.ndarray]:
        pr2_base_spec = PR2BaseOnlySpec(use_fixed_uuid=True)
        table_sdf = table.create_sdf()
        table_box2d = Box2d(
            np.array([table.TABLE_DEPTH, table.TABLE_WIDTH]), PlanerCoords.standard()
        )
        pr2_base_spec.get_kin()
        skmodel = pr2_base_spec.get_robot_model(deepcopy=False)
        skmodel.angle_vector(AV_INIT)
        pr2_base_spec.reflect_skrobot_model_to_kin(skmodel)
        cst = pr2_base_spec.create_collision_const()
        cst.set_sdf(table_sdf)

        n_max_trial = 100
        count = 0
        while True:
            count += 1
            if count > n_max_trial:
                return None
            pr2_point = target_region.sample_points(1)[0][:2]
            dist = np.linalg.norm(pr2_point - reaching_pose[:2])
            if dist > 0.8:
                continue

            sd = table_box2d.sd(pr2_point.reshape(1, 2))[0]
            if sd > 0.55 or sd < 0.0:
                continue

            yaw_reaching = reaching_pose[3]
            yaw = fit_radian(yaw_reaching + np.random.uniform(-np.pi / 4, np.pi / 4))

            pr2_coords = np.hstack([pr2_point, yaw])
            if cst.is_valid(pr2_coords):
                return pr2_coords

    @staticmethod
    def _sample_chairs(
        table: JskTable,
        target_region: BoxSkeleton,
        table_box2d_wrt_region: Box2d,
        pr2_coords: np.ndarray,
    ) -> List[JskChair]:

        n_chair = np.random.randint(0, JskMessyTableTask.N_MAX_CHAIR)
        pr2_obstacle_box = Box2d(
            np.array([0.7, 0.7]),
            PlanerCoords(table_box2d_wrt_region.coords.pos + pr2_coords[:2], pr2_coords[2]),
        )
        chair_box2d_list = []
        for _ in range(n_chair):
            obstacles = [table_box2d_wrt_region, pr2_obstacle_box] + chair_box2d_list
            chair_box2d = sample_box(
                target_region.extents[:2], np.array([0.5, 0.5]), obstacles, n_budget=100
            )
            if chair_box2d is not None:
                chair_box2d_list.append(chair_box2d)

        pr2_base_spec = PR2BaseOnlySpec(use_fixed_uuid=True)
        pr2_base_spec.get_kin()
        skmodel = pr2_base_spec.get_robot_model(deepcopy=False)
        skmodel.angle_vector(AV_INIT)
        pr2_base_spec.reflect_skrobot_model_to_kin(skmodel)
        table_sdf = table.create_sdf()
        cst = pr2_base_spec.create_collision_const()
        cst.set_sdf(table_sdf)

        chair_list = []
        for chair_box2d in chair_box2d_list:
            chair = JskChair()
            chair.translate([chair_box2d.coords.pos[0], chair_box2d.coords.pos[1], 0.0])
            chair.rotate(chair_box2d.coords.angle, "z")
            target_region.assoc(chair, relative_coords="local")
            chair_sdf = chair.create_sdf()
            # Combine the table's sdf with the chair's sdf.
            combined_sdf = UnionSDF([table_sdf, chair_sdf])
            cst.set_sdf(combined_sdf)
            if cst.is_valid(pr2_coords):
                chair_list.append(chair)
        return chair_list

    def visualize(self, viewer: Union[TrimeshSceneViewer, SceneWrapper]) -> None:
        table = JskTable()
        table.visualize(viewer)

        reaching_position, reaching_yaw = self.reaching_pose[:3], self.reaching_pose[3]
        co = Coordinates(reaching_position, rot=[reaching_yaw, 0, 0])
        ax = Axis.from_coords(co)
        viewer.add(ax)

        for obs_param in self.obstacles_param.reshape(-1, 6):
            x, y, yaw, d, w, h = obs_param
            box = BoxSkeleton(np.array([d, w, h]))
            box.translate([x, y, table.TABLE_HEIGHT + 0.5 * h])
            box.rotate(yaw, "z")
            viewer.add(box.to_visualizable(DARKGREEN_COLOR))

        for chair_param in self.chairs_param.reshape(-1, 3):
            x, y, yaw = chair_param
            chair = JskChair()
            chair.translate(np.hstack([x, y, 0.0]))
            chair.rotate(yaw, "z")
            chair.visualize(viewer)

    @classmethod
    @abstractmethod
    def consider_chair(self) -> bool:
        raise NotImplementedError


class JskMessyTableTaskWithChair(JskMessyTableTaskBase):
    @classmethod
    def consider_chair(cls) -> bool:
        return True


class JskMessyTableTask(JskMessyTableTaskBase):
    @classmethod
    def consider_chair(cls) -> bool:
        return False


if __name__ == "__main__":
    np.random.seed(17)
    table = JskTable()

    pr2 = PR2(use_tight_joint_limit=False)
    pr2.reset_manip_pose()

    task = JskMessyTableTaskWithChair.sample()
    task = JskMessyTableTaskWithChair.from_task_param(task.to_task_param())

    problem = task.export_problem()
    solver = OMPLSolver(OMPLSolverConfig(shortcut=True, bspline=True))
    ret = solver.solve(problem)
    spec = task._spec_cache

    pr2.translate(np.hstack([task.pr2_coords[:2], 0.0]))
    pr2.rotate(task.pr2_coords[2], "z")
    pr2.angle_vector(AV_INIT)

    v = PyrenderViewer()
    task.visualize(v)
    v.add(pr2)
    v.show()
    for q in ret.traj.resample(80):
        spec.set_skrobot_model_state(pr2, q)
        v.redraw()
        time.sleep(0.02)

    exp = task.export_task_expression(True)
    import matplotlib.pyplot as plt

    plt.imshow(exp.world_mat[1, :, :])
    plt.show()

    time.sleep(1000)
