from dataclasses import dataclass
from typing import ClassVar, List, Union

import numpy as np
from skrobot.coordinates import CascadedCoords
from skrobot.coordinates.math import rpy_angle
from skrobot.viewers import PyrenderViewer, TrimeshSceneViewer

from rpbench.articulated.world.utils import BoxSkeleton
from rpbench.interface import SamplableWorldBase
from rpbench.planer_box_utils import Box2d, PlanerCoords, is_colliding
from rpbench.utils import SceneWrapper

BROWN_COLOR = (204, 102, 0, 200)


class JskTable(CascadedCoords):
    size: List[float]
    table_primitives: List[BoxSkeleton]

    def __init__(self):
        super().__init__()
        plate_d = 0.05
        size = [0.8, 1.24, plate_d]
        height = 0.7
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
        primitive_list = [plate, leg1, leg2, leg3, leg4]

        # the table's coordinates equals to the center of the plate
        co_surface_center = plate.copy_worldcoords()
        co_surface_center.translate([0, 0, 0.5 * plate_d])
        self.newcoords(co_surface_center)
        for prim in primitive_list:
            self.assoc(prim)

        self.size = size
        self.table_primitives = primitive_list

    def visualize(self, viewer: Union[TrimeshSceneViewer, SceneWrapper]) -> None:
        for prim in self.table_primitives:
            viewer.add(prim.to_visualizable(BROWN_COLOR))


@dataclass
class JskMessyTableWorld(SamplableWorldBase):
    table: JskTable
    tabletop_obstacle_list: List[BoxSkeleton]
    obstacle_env_region: BoxSkeleton
    N_MAX_OBSTACLE: ClassVar[int] = 25
    OBSTACLE_H_MIN: ClassVar[float] = 0.05
    OBSTACLE_H_MAX: ClassVar[float] = 0.3

    @classmethod
    def sample(cls, standard: bool = False) -> "JskMessyTableWorld":
        table = JskTable()
        region_size = [table.size[0], table.size[1], cls.OBSTACLE_H_MAX + 0.05]
        obstacle_env_region = BoxSkeleton(region_size)
        table.assoc(obstacle_env_region, relative_coords="local")
        obstacle_env_region.translate([0.0, 0.0, region_size[2] * 0.5])

        if standard:
            # only single obstacle is placed
            co_obstacle = table.co_surface_center.copy_worldcoords()
            co_obstacle.translate([-0.2, 0.0, 0.15])
            box = BoxSkeleton([0.3, 0.3, 0.3])
            box.newcoords(co_obstacle)
            return cls(table, [box], obstacle_env_region)
        else:
            # implement obstacle distribution
            n_min_obstacle = 5
            n_max_obstacle = cls.N_MAX_OBSTACLE
            n_obstacle = np.random.randint(n_min_obstacle, n_max_obstacle)

            # consider sampling boxes inside a planer box (table)
            region2d = Box2d(np.array(table.size[:2]), PlanerCoords.standard())
            obj2d_list = []  # type: ignore
            for _ in range(n_obstacle):
                w = np.random.uniform(0.05, 0.2)
                d = np.random.uniform(0.05, 0.2)
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
            return cls(table, obj_list, obstacle_env_region)

    def get_all_obstacles(self) -> List[BoxSkeleton]:
        return self.tabletop_obstacle_list + self.table.table_primitives

    def visualize(self, viewer: Union[TrimeshSceneViewer, SceneWrapper]) -> None:
        self.table.visualize(viewer)
        for obs in self.tabletop_obstacle_list:
            viewer.add(obs.to_visualizable((100, 100, 100, 255)))
        viewer.add(self.obstacle_env_region.to_visualizable((0, 255, 0, 50)))

    def to_parameter(self) -> np.ndarray:
        pos = self.table.worldpos()
        rot = self.table.worldrot()
        ypr = rpy_angle(rot)[0]
        table_pose = np.array([pos[0], pos[1], ypr[0]])

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
    def from_parameter(cls, param: np.ndarray) -> "JskMessyTableWorld":
        table_pose = param[:3]
        table = JskTable()
        table.translate(np.hstack([table_pose[:2], 0]))
        table.rotate(table_pose[2], "z")

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
        return cls(table, obstacle_list, obstacle_env_region)


if __name__ == "__main__":
    from skrobot.models import Fetch

    fetch = Fetch()
    fetch.reset_pose()
    world = JskMessyTableWorld.sample(standard=False)
    world.table.translate([0.2, 0.0, 0.0])
    param = world.to_parameter()
    world = JskMessyTableWorld.from_parameter(param)
    param_again = world.to_parameter()
    world.table.translate([0.4, 0.0, 0.0])
    np.testing.assert_allclose(param, param_again)
    print(param)
    print(param_again)
    v = PyrenderViewer()
    world.visualize(v)
    v.show()
    v.add(fetch)
    import time

    time.sleep(1000)
