from dataclasses import dataclass
from typing import ClassVar, List, Union

import numpy as np
from skrobot.coordinates import CascadedCoords
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
    obstacle_list: List[BoxSkeleton]
    obstacle_env_region: BoxSkeleton
    N_MAX_OBSTACLE: ClassVar[int] = 25

    @classmethod
    def sample(cls, standard: bool = False) -> "JskMessyTableWorld":
        table = JskTable()
        obstacle_h_max = 0.3
        obstacle_h_min = 0.05
        region_size = [table.size[0], table.size[1], obstacle_h_max + 0.05]
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
            obj2d_list = []
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
                h = np.random.uniform(obstacle_h_min, obstacle_h_max)
                extent = np.hstack([obj2d.extent, h])
                obj = BoxSkeleton(extent, pos=np.hstack([obj2d.coords.pos, 0.0]))
                obj.rotate(obj2d.coords.angle, "z")
                obj.translate([0.0, 0.0, 0.5 * h - region_size[2] * 0.5])
                obj_list.append(obj)
                obstacle_env_region.assoc(obj, relative_coords="local")
            return cls(table, obj_list, obstacle_env_region)

    def visualize(self, viewer: Union[TrimeshSceneViewer, SceneWrapper]) -> None:
        self.table.visualize(viewer)
        for obs in self.obstacle_list:
            viewer.add(obs.to_visualizable((100, 100, 100, 255)))
        viewer.add(self.obstacle_env_region.to_visualizable((0, 255, 0, 50)))


if __name__ == "__main__":
    from skrobot.models import Fetch

    fetch = Fetch()
    fetch.reset_pose()
    world = JskMessyTableWorld.sample(standard=False)
    world.table.translate([0.6, 0.0, 0.0])
    v = PyrenderViewer()
    # v = TrimeshSceneViewer()
    world.visualize(v)
    v.show()
    v.add(fetch)
    import time

    time.sleep(1000)
