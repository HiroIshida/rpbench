from dataclasses import dataclass
from typing import Sequence

import numpy as np
from skrobot.sdf import UnionSDF

from rpbench.articulated.world.utils import BoxSkeleton, PrimitiveSkelton
from rpbench.vision import Camera, RayMarchingConfig


@dataclass
class HeightmapConfig:
    resol_x: int = 112
    resol_y: int = 112
    inf_subst_value: float = -1.0


@dataclass
class LocatedHeightmap:
    heightmap: np.ndarray
    surrounding_box: BoxSkeleton
    config: HeightmapConfig

    @classmethod
    def by_raymarching(
        cls,
        target_region: BoxSkeleton,
        objects: Sequence[PrimitiveSkelton],
        conf: HeightmapConfig = HeightmapConfig(),
        raymarching_conf: RayMarchingConfig = RayMarchingConfig(),
    ) -> "LocatedHeightmap":
        # although height map can be reated by rule-based method. But because
        # python is slow in loops thus, we resort to raymarching which can
        # be easily be vectorized.
        extent_plane = target_region.extents[:2]
        xlin = np.linspace(-0.5 * extent_plane[0], +0.5 * extent_plane[0], conf.resol_x)
        ylin = np.linspace(-0.5 * extent_plane[1], +0.5 * extent_plane[1], conf.resol_y)
        X, Y = np.meshgrid(xlin, ylin)
        height = target_region.extents[2]
        Z = np.zeros_like(X) + height

        points_wrt_region = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        points_wrt_world = target_region.transform_vector(points_wrt_region)
        dirs = np.tile(np.array([0, 0, -1]), (len(points_wrt_world), 1))

        union_sdf = UnionSDF([o.sdf for o in objects])

        dists = Camera.ray_marching(points_wrt_world, dirs, union_sdf, raymarching_conf)
        is_valid = dists < height + 1e-3

        dists_from_ground = height - dists
        dists_from_ground[~is_valid] = conf.inf_subst_value

        heightmap = dists_from_ground.reshape((conf.resol_x, conf.resol_y))
        return cls(heightmap, target_region, conf)

    @property
    def grid_width(self) -> np.ndarray:
        return self.surrounding_box.extents[:2] / np.array(
            [self.config.resol_x, self.config.resol_y]
        )

    def is_colliding(self, point_wrt_world: np.ndarray) -> bool:
        point_wrt_local = self.surrounding_box.inverse_transform_vector(point_wrt_world)
        idx_x, idx_y = np.floor(
            (point_wrt_local[:2] - (-0.5 * self.surrounding_box.extents[:2])) / self.grid_width
        ).astype(int)
        max_height_local = self.heightmap[idx_y, idx_x]
        return point_wrt_local[2] < max_height_local
