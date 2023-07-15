from dataclasses import dataclass
from typing import List, Optional, TypeVar, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm
from skrobot.coordinates import Coordinates
from skrobot.model.link import Link
from skrobot.model.primitives import Box
from skrobot.sdf import UnionSDF
from skrobot.viewers import TrimeshSceneViewer
from voxbloxpy.core import Grid

from rpbench.interface import WorldBase
from rpbench.planer_box_utils import Box2d, PlanerCoords, sample_box
from rpbench.two_dimensional.utils import Grid2d
from rpbench.utils import SceneWrapper
from rpbench.vision import Camera, RayMarchingConfig

TabletopWorldT = TypeVar("TabletopWorldT", bound="TabletopWorldBase")


@dataclass
class TabletopWorldBase(WorldBase):
    table: Box
    obstacles: List[Link]

    def visualize(self, viewer: Union[TrimeshSceneViewer, SceneWrapper]) -> None:
        viewer.add(self.table)
        for obs in self.obstacles:
            viewer.add(obs)

    def get_exact_sdf(self) -> UnionSDF:
        lst = [self.table.sdf]
        for obstacle in self.obstacles:
            lst.append(obstacle.sdf)
        return UnionSDF(lst)

    @staticmethod
    def create_standard_table() -> Box:
        # create jsk-lab 73b2 table
        table_depth = 0.5
        table_width = 0.75
        table_height = 0.7
        pos = [0.5 + table_depth * 0.5, 0.0, table_height * 0.5]
        table = Box(extents=[table_depth, table_width, table_height], pos=pos, with_sdf=True)
        return table

    def get_grid(self) -> Grid:
        grid_sizes = (56, 56, 28)
        mesh_height = 0.5
        depth, width, height = self.table._extents
        lb = np.array([-0.5 * depth, -0.5 * width, 0.5 * height - 0.1])
        ub = np.array([+0.5 * depth, +0.5 * width, 0.5 * height + mesh_height])
        lb = self.table.transform_vector(lb)
        ub = self.table.transform_vector(ub)
        return Grid(lb, ub, grid_sizes)

    def get_grid2d(self) -> Grid2d:
        grid3d = self.get_grid()
        size = (grid3d.sizes[0], grid3d.sizes[1])
        return Grid2d(grid3d.lb[:2], grid3d.ub[:2], size)

    def create_exact_heightmap(self) -> np.ndarray:
        grid2d = self.get_grid2d()
        depth, width, height = self.table._extents

        height_from_table = 1.0

        step = self.table._extents[:2] / np.array(grid2d.sizes)
        xlin = (
            np.linspace(step[0] * 0.5, step[0] * (grid2d.sizes[0] - 0.5), grid2d.sizes[0])
            - depth * 0.5
        )
        ylin = (
            np.linspace(step[1] * 0.5, step[1] * (grid2d.sizes[1] - 0.5), grid2d.sizes[1])
            - width * 0.5
        )
        X, Y = np.meshgrid(xlin, ylin)
        Z = np.zeros_like(X) - height * 0.5 + height_from_table
        points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        points = self.table.transform_vector(points)
        dirs = np.tile(np.array([0, 0, -1]), (len(points), 1))

        conf = RayMarchingConfig()
        dists = Camera.ray_marching(points, dirs, self.get_exact_sdf(), conf)
        is_valid = dists < height_from_table
        dists[~is_valid] = np.inf
        # debug point cloud
        # return points[is_valid] + dists[is_valid, None] * dirs[is_valid, :]
        return np.reshape(dists, (grid2d.sizes[0], grid2d.sizes[1]))

    def sample_pose(self) -> np.ndarray:
        self.get_exact_sdf()
        tabletop_center = self.table.copy_worldcoords()

        margin_from_plane = 0.03

        depth, width, height = self.table._extents
        np.array([depth, width])
        relative_pos = np.random.rand(3) * np.array([depth, width, 0.1]) + np.array(
            [-depth * 0.5, -width * 0.5, height * 0.5 + margin_from_plane]
        )
        co = tabletop_center.copy_worldcoords()
        co.translate(relative_pos)
        return co


@dataclass
class TabletopClutterWorld(TabletopWorldBase):
    @classmethod
    def sample(cls, standard: bool = False) -> "TabletopClutterWorld":
        intrinsic_desc = []

        table = cls.create_standard_table()
        table_depth, table_width, table_height = table._extents

        if not standard:
            x_rand, y_rand = np.random.randn(2)
            x = 0.1 + 0.05 * x_rand
            y = 0.1 * y_rand
            z = 0.0
            table.translate([x, y, z])
            intrinsic_desc.extend([x_rand, y_rand])
        else:
            intrinsic_desc.extend([0.0, 0.0])

        table_tip = table.copy_worldcoords()
        table_tip.translate([-table_depth * 0.5, -table_width * 0.5, +0.5 * table_height])

        table_extent = np.array([table_depth, table_width])

        obstacles = []  # reaching box is also an obstacle in planning context
        obstacles_2ds = []

        if standard:
            n_obs = 0
        else:
            n_obs = 1 + np.random.randint(8)

        for _ in range(n_obs):
            obs_extent = lognorm(s=0.5, scale=1.0).rvs(size=3) * np.array([0.1, 0.1, 0.15])
            obs2d = sample_box(table_extent, obs_extent[:2], [])
            if obs2d is None:
                break
            obstacles_2ds.append(obs2d)

            obs = Box(obs_extent, with_sdf=True)
            obs.newcoords(table.copy_worldcoords())
            obs.translate(
                np.hstack([obs2d.coords.pos, 0.5 * (obs_extent[-1] + table._extents[-1])])
            )
            obs.rotate(obs2d.coords.angle, "z")
            obs.visual_mesh.visual.face_colors = [0, 255, 0, 200]
            obstacles.append(obs)

        # check if all obstacle dont collide each other

        debug_matplotlib = False
        if debug_matplotlib:
            fig, ax = plt.subplots()
            for obs in obstacles_2ds:
                obs.visualize((fig, ax), "green")
            ax.set_aspect("equal", adjustable="box")

        return cls(table, obstacles)


@dataclass
class TabletopBoxWorld(TabletopWorldBase):
    box: Box

    @classmethod
    def sample(cls, standard: bool = False) -> "TabletopBoxWorld":
        intrinsic_desc = []

        table = cls.create_standard_table()
        table_depth, table_width, table_height = table._extents

        if not standard:
            x_rand, y_rand = np.random.randn(2)
            x = 0.1 + 0.05 * x_rand
            y = 0.1 * y_rand
            z = 0.0
            table.translate([x, y, z])
            intrinsic_desc.extend([x_rand, y_rand])
        else:
            intrinsic_desc.extend([0.0, 0.0])

        table_tip = table.copy_worldcoords()
        table_tip.translate([-table_depth * 0.5, -table_width * 0.5, +0.5 * table_height])

        table_extent = np.array([table_depth, table_width])
        while True:
            box2d: Optional[Box2d]
            if standard:
                box_extent = np.ones(3) * 0.15
                box2d = Box2d(box_extent[:2], PlanerCoords(np.zeros(2), 0.0 * np.pi))
            else:
                box_extent = np.ones(3) * 0.1 + np.random.rand(3) * np.array([0.1, 0.1, 0.05])
                box2d = sample_box(table_extent, box_extent[:2], [])
            if box2d is not None:
                break

        box = Box(box_extent, with_sdf=True)
        box.newcoords(table.copy_worldcoords())
        box.translate(np.hstack([box2d.coords.pos, 0.5 * (box_extent[-1] + table._extents[-1])]))
        box.rotate(box2d.coords.angle, "z")
        box.visual_mesh.visual.face_colors = [255, 0, 0, 200]

        obstacles = [box]  # reaching box is also an obstacle in planning context
        obstacles_2ds = []

        if standard:
            n_obs = 0
        else:
            n_obs = 1 + np.random.randint(8)

        for _ in range(n_obs):
            obs_extent = lognorm(s=0.5, scale=1.0).rvs(size=3) * np.array([0.04, 0.04, 0.2])
            obs2d = sample_box(table_extent, obs_extent[:2], [box2d])
            if obs2d is None:
                break
            obstacles_2ds.append(obs2d)

            obs = Box(obs_extent, with_sdf=True)
            obs.newcoords(table.copy_worldcoords())
            obs.translate(
                np.hstack([obs2d.coords.pos, 0.5 * (obs_extent[-1] + table._extents[-1])])
            )
            obs.rotate(obs2d.coords.angle, "z")
            obs.visual_mesh.visual.face_colors = [0, 255, 0, 200]
            obstacles.append(obs)

        # check if all obstacle dont collide each other

        debug_matplotlib = False
        if debug_matplotlib:
            fig, ax = plt.subplots()
            box2d.visualize((fig, ax), "red")
            for obs in obstacles_2ds:
                obs.visualize((fig, ax), "green")
            ax.set_aspect("equal", adjustable="box")

        return cls(table, obstacles, box)


@dataclass
class TabletopOvenWorld(TabletopWorldBase):
    box_center: Coordinates
    box_d: float
    box_w: float
    box_h: float
    box_t: float
    _intrinsic_desc: np.ndarray

    def export_intrinsic_description(self) -> np.ndarray:
        return self._intrinsic_desc

    @classmethod
    def sample(cls, standard: bool = False) -> "TabletopOvenWorld":
        intrinsic_desc = []

        table = cls.create_standard_table()
        table_depth, table_width, table_height = table._extents

        if not standard:
            x_rand, y_rand = np.random.randn(2)
            x = 0.1 + 0.05 * x_rand
            y = 0.1 * y_rand
            z = 0.0
            table.translate([x, y, z])

            intrinsic_desc.extend([x_rand, y_rand])
        else:
            intrinsic_desc.extend([0.0, 0.0])

        table_tip = table.copy_worldcoords()
        table_tip.translate([-table_depth * 0.5, -table_width * 0.5, +0.5 * table_height])

        obstacles = []

        color = np.array([255, 220, 0, 150])

        # box
        d = 0.2
        w = 0.3
        h = 0.2
        if standard:
            d += 0.15
            w += 0.15
            h += 0.15
            intrinsic_desc.extend([0.0, 0.0, 0.0])
        else:
            d_rand, w_rand, h_rand = np.random.randn(3)
            d += 0.15 + d_rand * 0.05
            w += 0.15 + w_rand * 0.05
            h += 0.15 + h_rand * 0.05
            intrinsic_desc.extend([d_rand, w_rand, h_rand])

        t = 0.03

        if standard:
            box_center = table.copy_worldcoords()
            box_center.translate([0, 0, 0.5 * table_height])
            intrinsic_desc.extend([0.0, 0.0])
        else:
            box_center = table.copy_worldcoords()
            box_center.translate([0, 0, 0.5 * table_height])

            margin_x = table_depth - d
            margin_y = table_width - w

            x_rand, y_rand = np.random.randn(2)

            trans = np.array([0.5 * 0.3 * margin_x * x_rand, 0.5 * 0.3 * margin_y * y_rand, 0.0])
            box_center.translate(trans)

            intrinsic_desc.extend([x_rand, y_rand])

        lower_plate = Box([d, w, t], with_sdf=True, face_colors=color)
        lower_plate.newcoords(box_center.copy_worldcoords())
        lower_plate.translate([0, 0, 0.5 * t])
        obstacles.append(lower_plate)

        upper_plate = Box([d, w, t], with_sdf=True, face_colors=color)
        upper_plate.newcoords(box_center.copy_worldcoords())
        upper_plate.translate([0, 0, h - 0.5 * t])
        obstacles.append(upper_plate)

        left_plate = Box([d, t, h], with_sdf=True, face_colors=color)
        left_plate.newcoords(box_center.copy_worldcoords())
        left_plate.translate([0, 0.5 * w - 0.5 * t, 0.5 * h])
        obstacles.append(left_plate)

        right_plate = Box([d, t, h], with_sdf=True, face_colors=color)
        right_plate.newcoords(box_center.copy_worldcoords())
        right_plate.translate([0, -0.5 * w + 0.5 * t, 0.5 * h])
        obstacles.append(right_plate)

        opposite_plate = Box([t, w, h], with_sdf=True, face_colors=color)
        opposite_plate.newcoords(box_center.copy_worldcoords())
        opposite_plate.translate([0.5 * d - 0.5 * t, 0.0, 0.5 * h])
        obstacles.append(opposite_plate)

        return cls(table, obstacles, box_center, d, w, h, t, np.array(intrinsic_desc))
