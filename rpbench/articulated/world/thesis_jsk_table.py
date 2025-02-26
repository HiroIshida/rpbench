from abc import abstractmethod
from dataclasses import dataclass
from typing import ClassVar, List, Tuple, Union

import numpy as np
from plainmp.psdf import UnionSDF
from plainmp.robot_spec import PR2BaseOnlySpec
from plainmp.utils import primitive_to_plainmp_sdf
from skrobot.coordinates import CascadedCoords
from skrobot.coordinates.math import rpy_angle
from skrobot.models.pr2 import PR2
from skrobot.viewers import PyrenderViewer, TrimeshSceneViewer

from rpbench.articulated.world.utils import BoxSkeleton
from rpbench.interface import SamplableWorldBase
from rpbench.planer_box_utils import Box2d, PlanerCoords, is_colliding, sample_box
from rpbench.utils import SceneWrapper

BROWN_COLOR = (204, 102, 0, 200)


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
            viewer.add(prim.to_visualizable(BROWN_COLOR))

    def create_sdf(self) -> UnionSDF:
        psdfs = [
            primitive_to_plainmp_sdf(prim.to_skrobot_primitive()) for prim in self.chair_primitives
        ]
        return UnionSDF(psdfs)


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
        psdfs = [
            primitive_to_plainmp_sdf(prim.to_skrobot_primitive()) for prim in self.table_primitives
        ]
        return UnionSDF(psdfs)


@dataclass
class JskMessyTableWorldBase(SamplableWorldBase):
    table: JskTable
    pr2_box: BoxSkeleton
    chair_list: List[JskChair]
    tabletop_obstacle_list: List[BoxSkeleton]
    obstacle_env_region: BoxSkeleton
    N_MAX_OBSTACLE: ClassVar[int] = 40
    OBSTACLE_W_MIN: ClassVar[float] = 0.05
    OBSTACLE_W_MAX: ClassVar[float] = 0.25
    OBSTACLE_H_MIN: ClassVar[float] = 0.05
    OBSTACLE_H_MAX: ClassVar[float] = 0.35

    @classmethod
    @abstractmethod
    def get_rotation_angle(cls) -> float:
        pass

    @classmethod
    def from_semantic_params(
        cls, table_2dpos: np.ndarray, bbox_param_list: List[np.ndarray]
    ) -> "JskMessyTableWorldBase":
        """
        Args:
            table_2dpos: [x, y]
            bbox_param_list: [[x, y, yaw, w, d, h], ...]
        NOTE: all in world (robot's root) frame
        """
        param = np.zeros(2 + 6 * cls.N_MAX_OBSTACLE)
        param[:2] = table_2dpos
        for i, bbox in enumerate(bbox_param_list):
            param[2 + 6 * i : 2 + 6 * (i + 1)] = bbox
        return cls.from_parameter(param)

    def is_out_of_distribution(self) -> bool:
        if len(self.tabletop_obstacle_list) > self.N_MAX_OBSTACLE:
            return True
        co = self.table.copy_worldcoords()
        pos = co.worldpos()
        x_min, x_max, y_min, y_max = self.get_table_position_minmax()
        if not (x_min <= pos[0] <= x_max):
            return True
        if not (y_min <= pos[1] <= y_max):
            return True
        for obs in self.tabletop_obstacle_list:
            h = obs.extents[2]
            if not (self.OBSTACLE_H_MIN <= h <= self.OBSTACLE_H_MAX):
                return True
            # TODO: check positions ...
            # all obstacles are on the table and should not extend outside the table
            # but checking this is bit tidious
        return False

    @classmethod
    @abstractmethod
    def get_table_position_minmax(cls) -> Tuple[float, float, float, float]:
        # xmin, xmax, ymin, ymax
        pass

    @classmethod
    def sample(cls, standard: bool = False) -> "JskMessyTableWorldBase":
        table = JskTable()

        target_region = BoxSkeleton(
            [
                table.TABLE_DEPTH + table.DIST_FROM_DOORSIDE_WALL + table.DIST_FROM_FRIDGESIDE_WALL,
                table.TABLE_WIDTH + 1.2,
                0.1,
            ],
            pos=[-0.46, +0.56, 0],  # hand-tuned to fit well
        )
        table.assoc(target_region)
        table_pos_from_region = -target_region.worldpos()[:2]
        table_box2d_wrt_region = Box2d(
            np.array([table.TABLE_DEPTH, table.TABLE_WIDTH]),
            PlanerCoords(table_pos_from_region, 0.0),
        )  # table box2d wrt world (the table's center)

        # >> sample robot inside the target map region
        pr2_box2d = None
        pr2_base_spec = PR2BaseOnlySpec()
        table_sdf = table.create_sdf()
        table_box2d_wrt_table = Box2d(
            np.array([table.TABLE_DEPTH, table.TABLE_WIDTH]), PlanerCoords.standard()
        )
        pr2_base_spec.get_kin()
        skmodel = pr2_base_spec.get_robot_model()
        av_init = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.30000001192092896,
                0.0,
                0.0,
                0.8726646304130554,
                0.0,
                -1.3089969158172607,
                0.4000000059604645,
                -2.5999999046325684,
                0.3490658402442932,
                -1.919862151145935,
                -0.1745329201221466,
                -0.1745329201221466,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.3089969158172607,
                0.4000000059604645,
                2.5999999046325684,
                -0.3490658402442932,
                -1.919862151145935,
                -0.1745329201221466,
                -0.1745329201221466,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        skmodel.angle_vector(av_init)
        pr2_base_spec.reflect_skrobot_model_to_kin(skmodel)
        cst = pr2_base_spec.create_collision_const()
        cst.set_sdf(table_sdf)
        eps = 1e-2
        while True:
            pr2_point = target_region.sample_points(1)[0][:2]
            sd = table_box2d_wrt_table.sd(pr2_point.reshape(1, 2))[0]
            if sd > 0.6:
                continue  # too far from the table
            if sd < 0.0:
                continue  # apparently colliding with the table

            pr2_point_plus_x = pr2_point + np.array([eps, 0.0])
            pr2_point_plus_y = pr2_point + np.array([0.0, eps])
            sds = table_box2d_wrt_table.sd(np.array([pr2_point_plus_x, pr2_point_plus_y]))
            grad_sd = (sds[0] - sd, sds[1] - sd)
            yaw_center = np.arctan2(grad_sd[1], grad_sd[0]) + np.pi  # TODO: randomize
            yaw = yaw_center + np.random.uniform(-np.pi / 6, np.pi / 6)
            quaternion = np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])
            q = np.hstack([pr2_point, 0, quaternion])
            cst.evaluate(q)[0]
            if cst.is_valid(q):
                pr2_box2d = Box2d(np.array([0.5, 0.5]), PlanerCoords(pr2_point, yaw))
                break
        pr2_box = BoxSkeleton(np.hstack([pr2_box2d.extent, 0.3]))
        pr2_box.translate([pr2_box2d.coords.pos[0], pr2_box2d.coords.pos[1], 0.0])
        pr2_box.rotate(pr2_box2d.coords.angle, "z")
        # << sample robot inside the target map region

        # >> sample chairs inside the target map region
        sample_chair = True
        chair_list = []
        if sample_chair:
            n_chair = np.random.randint(0, 5)
            chair_box2d_list = []
            for _ in range(n_chair):
                obstacles = [table_box2d_wrt_region] + chair_box2d_list
                chair_box2d = sample_box(
                    target_region.extents[:2], np.array([0.5, 0.5]), obstacles, n_budget=50
                )
                if chair_box2d is not None:
                    chair_box2d_list.append(chair_box2d)
            for chair_box2d in chair_box2d_list:
                chair = JskChair()
                chair.translate([chair_box2d.coords.pos[0], chair_box2d.coords.pos[1], 0.0])
                chair.rotate(chair_box2d.coords.angle, "z")
                target_region.assoc(chair, relative_coords="local")
                chair_sdf = chair.create_sdf()
                cst.set_sdf(chair_sdf)
                if cst.is_valid(q):
                    chair_list.append(chair)
        # << sample chairs inside the target map region

        # sample obstacle on the table
        region_size = [table.size[0], table.size[1], cls.OBSTACLE_H_MAX + 0.05]
        obstacle_env_region = BoxSkeleton(region_size)
        table.assoc(obstacle_env_region, relative_coords="local")
        obstacle_env_region.translate([0.0, 0.0, region_size[2] * 0.5])

        if standard:
            assert False
        else:
            # implement obstacle distribution
            n_min_obstacle = 5
            n_max_obstacle = cls.N_MAX_OBSTACLE
            n_obstacle = np.random.randint(n_min_obstacle, n_max_obstacle)

            # consider sampling boxes inside a planer box (table)
            region2d = Box2d(np.array(table.size[:2]), PlanerCoords.standard())
            obj2d_list = []  # type: ignore
            for _ in range(n_obstacle):
                w = np.random.uniform(cls.OBSTACLE_W_MIN, cls.OBSTACLE_W_MAX)
                d = np.random.uniform(cls.OBSTACLE_W_MIN, cls.OBSTACLE_W_MAX)
                yaw = np.random.uniform(0.0, np.pi)
                center = region2d.sample_point()
                obj2d = Box2d(np.array([w, d]), PlanerCoords(center, yaw))  # type: ignore
                if not region2d.contains(obj2d):  # if stick out of the table
                    continue
                is_any_colliding = any([is_colliding(obj2d, o) for o in obj2d_list])
                if is_any_colliding:
                    continue
                obj2d_list.append(obj2d)

            obj_list = []
            for obj2d in obj2d_list:
                h = np.random.uniform(cls.OBSTACLE_H_MIN, cls.OBSTACLE_H_MAX)
                extent = np.hstack([obj2d.extent, h])
                obj = BoxSkeleton(extent, pos=np.hstack([obj2d.coords.pos, 0.0]))
                obj.rotate(obj2d.coords.angle, "z")
                obj.translate([0.0, 0.0, 0.5 * h - region_size[2] * 0.5])
                obj_list.append(obj)
                obstacle_env_region.assoc(obj, relative_coords="local")

            # return cls(table, obj_list, obstacle_env_region)
            tmp = cls(table, pr2_box, chair_list, obj_list, obstacle_env_region)
            tmp.tmp = target_region
            return tmp

    def get_all_obstacles(self) -> List[BoxSkeleton]:
        return self.tabletop_obstacle_list + self.table.table_primitives

    def visualize(self, viewer: Union[TrimeshSceneViewer, SceneWrapper]) -> None:
        self.table.visualize(viewer)

        viewer.add(self.pr2_box.to_visualizable((255, 0, 0, 255)))

        for chair in self.chair_list:
            chair.visualize(viewer)

        for obs in self.tabletop_obstacle_list:
            viewer.add(obs.to_visualizable((0, 255, 0, 255)))
        viewer.add(self.tmp.to_visualizable((0, 0, 255, 50)))

    def to_parameter(self) -> np.ndarray:
        pos = self.table.worldpos()
        rot = self.table.worldrot()
        ypr = rpy_angle(rot)[0]
        # np.testing.assert_allclose(ypr, np.zeros(3))
        table_pose = np.array([pos[0], pos[1]])

        head = 0
        obstacle_param = np.zeros(6 * self.N_MAX_OBSTACLE)
        for obs in self.tabletop_obstacle_list:
            pos = obs.worldpos()
            rot = obs.worldrot()
            ypr = rpy_angle(rot)[0]
            obstacle_pose = np.array([pos[0], pos[1], ypr[0]])
            obstacle_param[head : head + 6] = np.hstack([obstacle_pose, obs.extents])
            head += 6
        return np.hstack([table_pose, obstacle_param])

    @classmethod
    def from_parameter(cls, param: np.ndarray) -> "JskMessyTableWorldBase":
        table_pose = param[:2]
        table = JskTable()
        table.translate(np.hstack([table_pose, 0]))
        angle = cls.get_rotation_angle()
        table.rotate(angle, "z")

        region_size = [table.size[0], table.size[1], cls.OBSTACLE_H_MAX + 0.05]
        obstacle_env_region = BoxSkeleton(region_size)
        table.assoc(obstacle_env_region, relative_coords="local")
        obstacle_env_region.translate([0.0, 0.0, region_size[2] * 0.5])

        obstacle_param = param[2:]
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

        return cls(table, obstacle_list, obstacle_env_region)


class FetchJskMessyTableWorld(JskMessyTableWorldBase):
    ...


class JskMessyTableWorld(FetchJskMessyTableWorld):
    @classmethod
    def get_rotation_angle(cls) -> float:
        return 0.0

    @classmethod
    def get_table_position_minmax(cls) -> Tuple[float, float, float, float]:
        return 0.6, 0.8, JskTable.TABLE_WIDTH * -0.5, JskTable.TABLE_WIDTH * 0.5


class JskMessyTableWorld2(FetchJskMessyTableWorld):
    @classmethod
    def get_rotation_angle(cls) -> float:
        return -np.pi / 2

    @classmethod
    def get_table_position_minmax(cls) -> Tuple[float, float, float, float]:
        margin = 0.2
        return 0.9, 1.1, JskTable.TABLE_DEPTH * -0.5 - margin, JskTable.TABLE_DEPTH * 0.5 + margin


if __name__ == "__main__":
    pr2 = PR2()
    pr2.reset_manip_pose()
    world = JskMessyTableWorld.sample(standard=False)
    # [JskMessyTableWorld.sample(standard=False) for _ in tqdm.tqdm(range(1000))]
    pr2.newcoords(world.pr2_box.copy_worldcoords())

    av_init = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.30000001192092896,
            0.0,
            0.0,
            0.8726646304130554,
            0.0,
            -1.3089969158172607,
            0.4000000059604645,
            -2.5999999046325684,
            0.3490658402442932,
            -1.919862151145935,
            -0.1745329201221466,
            -0.1745329201221466,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.3089969158172607,
            0.4000000059604645,
            2.5999999046325684,
            -0.3490658402442932,
            -1.919862151145935,
            -0.1745329201221466,
            -0.1745329201221466,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    pr2.angle_vector(av_init)

    # world = JskMessyTableWorld.from_parameter(world.to_parameter())
    v = PyrenderViewer()
    world.visualize(v)
    v.add(pr2)
    v.show()
    # v.add(fetch)
    import time

    time.sleep(1000)
