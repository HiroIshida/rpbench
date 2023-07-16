from dataclasses import dataclass
from typing import List, TypeVar, Union

import numpy as np
from scipy.stats import lognorm
from skrobot.model.primitives import Box
from skrobot.sdf import UnionSDF
from skrobot.viewers import TrimeshSceneViewer
from voxbloxpy.core import Grid

from rpbench.articulated.world.utils import BoxSkeleton
from rpbench.interface import WorldBase
from rpbench.two_dimensional.utils import Grid2d
from rpbench.utils import SceneWrapper
from rpbench.vision import Camera, RayMarchingConfig

GroundWorldT = TypeVar("GroundWorldT", bound="GroundWorldBase")


@dataclass
class GroundWorldBase(WorldBase):
    ground: BoxSkeleton
    foot_box: BoxSkeleton
    obstacles: List[BoxSkeleton]

    @classmethod
    def default_ground(cls) -> BoxSkeleton:
        ground = BoxSkeleton([3.0, 3.0, 0.1], with_sdf=True)
        ground.translate([0.0, 0.0, -0.05])
        return ground

    def visualize(self, viewer: Union[TrimeshSceneViewer, SceneWrapper]) -> None:
        ground = self.ground.to_box()
        viewer.add(ground)

        for obs_tmp in self.obstacles:
            obs = obs_tmp.to_box()
            obs.visual_mesh.visual.face_colors = [255, 0, 0, 120]
            viewer.add(obs)

    def get_exact_sdf(self) -> UnionSDF:
        lst = [self.ground.sdf]
        for obstacle in self.obstacles:
            lst.append(obstacle.sdf)
        return UnionSDF(lst)

    def get_grid(self) -> Grid:
        grid_sizes = (56, 56, 28)
        depth = 1.0
        width = 1.0
        height = 0.7
        lb = np.array([0.0, -0.5 * width, height])
        ub = lb + np.array([depth, width, height])
        return Grid(lb, ub, grid_sizes)

    def get_grid2d(self) -> Grid2d:
        grid3d = self.get_grid()
        return Grid2d(grid3d.lb[:2], grid3d.ub[:2], (112, 112))

    def create_exact_heightmap(self) -> np.ndarray:
        grid2d = self.get_grid2d()

        subplane = Box(extents=[1.0, 1.0, 0.7], pos=[0.5, 0.0, -0.35])
        depth, width, height = subplane._extents

        height_from_plane = 1.0

        step = subplane._extents[:2] / np.array(grid2d.sizes)
        xlin = (
            np.linspace(step[0] * 0.5, step[0] * (grid2d.sizes[0] - 0.5), grid2d.sizes[0])
            - depth * 0.5
        )
        ylin = (
            np.linspace(step[1] * 0.5, step[1] * (grid2d.sizes[1] - 0.5), grid2d.sizes[1])
            - width * 0.5
        )
        X, Y = np.meshgrid(xlin, ylin)
        Z = np.zeros_like(X) - height * 0.5 + height_from_plane
        points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        points = subplane.transform_vector(points)
        dirs = np.tile(np.array([0, 0, -1]), (len(points), 1))

        conf = RayMarchingConfig()
        dists = Camera.ray_marching(points, dirs, self.get_exact_sdf(), conf)
        is_valid = dists < height_from_plane
        dists[~is_valid] = np.inf
        # debug point cloud
        # return points[is_valid] + dists[is_valid, None] * dirs[is_valid, :]
        return np.reshape(dists, (grid2d.sizes[0], grid2d.sizes[1]))


@dataclass
class GroundClutteredWorld(GroundWorldBase):
    @staticmethod
    def is_aabb_collide(box1: BoxSkeleton, box2: BoxSkeleton) -> bool:
        U1 = box1.worldpos() + np.array(box1._extents) * 0.5
        L1 = box1.worldpos() - np.array(box1._extents) * 0.5
        U2 = box2.worldpos() + np.array(box2._extents) * 0.5
        L2 = box2.worldpos() - np.array(box2._extents) * 0.5

        if U1[0] < L2[0] or L1[0] > U2[0]:
            return False
        if U1[1] < L2[1] or L1[1] > U2[1]:
            return False
        if U1[2] < L2[2] or L1[2] > U2[2]:
            return False
        return True

    @classmethod
    def sample(cls, standard: bool = False) -> "GroundClutteredWorld":
        ground = cls.default_ground()
        foot_box = BoxSkeleton(extents=[0.4, 0.5, 0.7], pos=[0.0, 0, 0.25], with_sdf=True)

        n_obstacle = np.random.randint(20)
        obstacles: List[BoxSkeleton] = []
        if not standard:

            while len(obstacles) < n_obstacle:
                box_size = lognorm(s=0.5, scale=1.0).rvs(size=3) * np.array([0.2, 0.2, 0.4])
                box_size[2] = min(box_size[2], 0.7)
                pos2d = np.random.rand(2)
                pos2d[1] += -0.5
                pos3d = np.hstack([pos2d, box_size[2] * 0.5])
                box = BoxSkeleton(box_size, pos=pos3d, with_sdf=True)

                if cls.is_aabb_collide(foot_box, box):
                    continue

                obstacles.append(box)

        return cls(ground, foot_box, obstacles)
