import time
from dataclasses import dataclass
from typing import Any, ClassVar, List, Optional, Tuple, Union

import numpy as np
from plainmp.problem import Problem
from plainmp.psdf import UnionSDF
from plainmp.robot_spec import Coordinates, PR2BaseOnlySpec, PR2LarmSpec, PR2RarmSpec
from skrobot.coordinates import CascadedCoords
from skrobot.coordinates.math import rpy_angle
from skrobot.model.primitives import Axis
from skrobot.models.pr2 import PR2
from skrobot.viewers import PyrenderViewer, TrimeshSceneViewer

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

    def visualize(self, viewer: Union[TrimeshSceneViewer, SceneWrapper]) -> None:
        for prim in self.chair_primitives:
            viewer.add(prim.to_visualizable(DARKRED_COLOR))

    def create_sdf(self) -> UnionSDF:
        return UnionSDF([p.to_plainmp_sdf() for p in self.chair_primitives])


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


@dataclass
class JskMessyTableTask(TaskBase):
    tabletop_obstacle_list: List[BoxSkeleton]
    chair_list: List[JskChair]
    pr2_coords: np.ndarray
    reaching_pose: np.ndarray
    N_MAX_OBSTACLE: ClassVar[int] = 20
    N_MAX_CHAIR: ClassVar[int] = 5
    OBSTACLE_W_MIN: ClassVar[float] = 0.05
    OBSTACLE_W_MAX: ClassVar[float] = 0.25
    OBSTACLE_H_MIN: ClassVar[float] = 0.05
    OBSTACLE_H_MAX: ClassVar[float] = 0.35

    # @classmethod
    # def from_semantic_params(
    #     cls, pr2_coords: np.ndarray, bbox_param_list: List[np.ndarray]
    #     ) -> "JskMessyTableTask":
    #     """
    #     Args:
    #         pr2_coords: [x, y, yaw]
    #         bbox_param_list: [[x, y, yaw, w, d, h], ...]
    #     """
    #     param = np.zeros(3 + 6 * cls.N_MAX_OBSTACLE)
    #     param[:3] = pr2_coords
    #     for i, bbox in enumerate(bbox_param_list):
    #         param[3 + 6 * i : 2 + 6 * (i + 1)] = bbox
    #     return cls.from_parameter(param)

    # abstract override
    @classmethod
    def from_task_param(cls, param: np.ndarray) -> "JskMessyTableTask":
        ...

    # abstract override
    @classmethod
    def to_task_param(self) -> np.ndarray:
        ...

    # abstract override
    def export_task_expression(self, use_matrix: bool) -> TaskExpression:
        ...

    # abstract override
    def solve_default(self) -> ResultProtocol:
        raise NotImplementedError

    # abstract override
    def export_problem(self) -> Problem:
        raise NotImplementedError

    # abstract override
    @classmethod
    def sample(cls, predicate: Optional[Any] = None) -> Optional["JskMessyTableTask"]:
        assert predicate is None, "predicate is not supported"
        timeout = 10
        t_start = time.time()

        table = JskTable()
        obstacles, obstacle_env_region = cls._sample_obstacles_on_table(table)
        while True:
            elapsed = time.time() - t_start
            if elapsed > timeout:
                return None
            reaching_pose = cls._sample_reaching_pose(obstacles)
            target_region, table_box2d_wrt_region = cls._prepare_target_region(table)
            pr2_coords = cls._sample_robot(table, target_region, reaching_pose)
            if pr2_coords is None:
                continue
            chair_list = cls._sample_chairs(
                table, target_region, table_box2d_wrt_region, pr2_coords
            )
            return cls(obstacles, chair_list, pr2_coords, reaching_pose)

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
    def _prepare_target_region(table: JskTable) -> Tuple[BoxSkeleton, Box2d]:
        target_region = BoxSkeleton(
            [
                table.TABLE_DEPTH + table.DIST_FROM_DOORSIDE_WALL + table.DIST_FROM_FRIDGESIDE_WALL,
                table.TABLE_WIDTH + 1.2,
                0.1,
            ],
            pos=[-0.46, 0.56, 0],  # hand-tuned to fit well
        )
        table.assoc(target_region)
        table_pos_from_region = -target_region.worldpos()[:2]
        table_box2d_wrt_region = Box2d(
            np.array([table.TABLE_DEPTH, table.TABLE_WIDTH]),
            PlanerCoords(table_pos_from_region, 0.0),
        )
        return target_region, table_box2d_wrt_region

    @staticmethod
    def _sample_robot(
        table: JskTable, target_region: BoxSkeleton, reaching_pose: np.ndarray
    ) -> Optional[np.ndarray]:
        pr2_base_spec = PR2BaseOnlySpec()
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

            y_axis = np.array([-np.sin(yaw), np.cos(yaw)])
            vec_robot_to_reach = reaching_pose[:2] - pr2_point
            right_side = np.dot(y_axis, vec_robot_to_reach) < 0
            if not right_side:
                continue

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
            else:
                print("Failed to sample chair box")

        pr2_base_spec = PR2BaseOnlySpec()
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

    def get_all_obstacles(self) -> List[BoxSkeleton]:
        return self.tabletop_obstacle_list + self.table.table_primitives

    def visualize(self, viewer: Union[TrimeshSceneViewer, SceneWrapper]) -> None:
        table = JskTable()
        table.visualize(viewer)

        reaching_position, reaching_yaw = self.reaching_pose[:3], self.reaching_pose[3]
        co = Coordinates(reaching_position, rot=[reaching_yaw, 0, 0])
        ax = Axis.from_coords(co)
        viewer.add(ax)

        for chair in self.chair_list:
            chair.visualize(viewer)

        for obs in self.tabletop_obstacle_list:
            viewer.add(obs.to_visualizable((0, 255, 0, 255)))

    def to_parameter(self) -> np.ndarray:
        head = 0
        obstacle_param = np.zeros(6 * self.N_MAX_OBSTACLE)
        for obs in self.tabletop_obstacle_list:
            pos = obs.worldpos()
            rot = obs.worldrot()
            ypr = rpy_angle(rot)[0]
            obstacle_pose = np.array([pos[0], pos[1], ypr[0]])
            obstacle_param[head : head + 6] = np.hstack([obstacle_pose, obs.extents])
            head += 6
        return np.hstack([self.pr2_coords, obstacle_param])

    @classmethod
    def from_parameter(cls, param: np.ndarray) -> "JskMessyTableTask":
        table = JskTable()

        region_size = [table.size[0], table.size[1], cls.OBSTACLE_H_MAX + 0.05]
        obstacle_env_region = BoxSkeleton(region_size)
        table.assoc(obstacle_env_region, relative_coords="local")
        obstacle_env_region.translate([0.0, 0.0, region_size[2] * 0.5])

        pr2_planar_coords = param[:3]

        region_size = [table.size[0], table.size[1], cls.OBSTACLE_H_MAX + 0.05]
        obstacle_env_region = BoxSkeleton(region_size)
        table.assoc(obstacle_env_region, relative_coords="local")
        obstacle_env_region.translate([0.0, 0.0, region_size[2] * 0.5])

        obstacle_param = param[3:]
        n_obstacle = len(obstacle_param) // 6
        obstacle_list = []
        head = 0
        for _ in range(n_obstacle):
            sub_param = obstacle_param[head : head + 6]
            if np.all(sub_param == 0.0):
                break
            pose_param, extent = sub_param[:3], sub_param[3:]
            box = BoxSkeleton(extent)
            trans = table.worldpos()[2] + 0.5 * extent[2]
            box.translate(np.hstack([pose_param[:2], trans]))
            box.rotate(pose_param[2], "z")
            obstacle_env_region.assoc(box, relative_coords="world")
            obstacle_list.append(box)
            head += 6

        return cls(table, pr2_planar_coords, [], obstacle_list, obstacle_env_region)


if __name__ == "__main__":
    pr2 = PR2()
    pr2.reset_manip_pose()
    world = JskMessyTableTask.sample()

    pr2.translate(np.hstack([world.pr2_coords[:2], 0.0]))
    pr2.rotate(world.pr2_coords[2], "z")
    pr2.angle_vector(AV_INIT)

    v = PyrenderViewer()
    world.visualize(v)
    v.add(pr2)
    v.show()
    # v.add(fetch)
    import time

    time.sleep(1000)
