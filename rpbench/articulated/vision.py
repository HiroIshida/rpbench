import copy
import logging
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np
import threadpoolctl
from skrobot.coordinates.geo import orient_coords_to_axis
from skrobot.coordinates.math import matrix2quaternion, rotation_matrix_from_axis
from skrobot.model.primitives import Axis, Coordinates
from skrobot.sdf import UnionSDF

try:
    from voxbloxpy.core import CameraPose, EsdfMap

    VOXBLOX_INSTALLED = True
except ImportError:

    class CameraPose:
        ...

    class EsdfMap:
        ...

    VOXBLOX_INSTALLED = False

from rpbench.articulated.world.utils import (
    BoxSkeleton,
    CylinderSkelton,
    PrimitiveSkelton,
)

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    resolx: int = 640
    resoly: int = 480
    fovx: float = np.deg2rad(57.0)
    fovy: float = np.deg2rad(43.0)


@dataclass
class RayMarchingConfig:
    surface_closeness_threashold: float = 1e-3
    max_iter: int = 30
    max_dist: float = 2.0
    terminate_active_rate_threshold: float = 0.005


class Camera(Axis):
    config: CameraConfig

    def __init__(
        self, camera_pos: Optional[np.ndarray] = None, config: Optional[CameraConfig] = None
    ):
        if config is None:
            config = CameraConfig()
        super().__init__(axis_radius=0.02, axis_length=0.3, pos=camera_pos)
        self.config = config

    def look_at(self, p: np.ndarray, horizontal: bool = True) -> None:
        if np.all(p == self.worldpos()):
            return
        diff = p - self.worldpos()
        orient_coords_to_axis(self, diff, axis="x")
        if horizontal:
            co = Coordinates(
                pos=self.worldpos(), rot=rotation_matrix_from_axis(diff, [0, 0, 1], axes="xz")
            )
            self.newcoords(co)

    def generate_point_cloud(
        self,
        sdf: Callable[[np.ndarray], np.ndarray],
        rm_config: RayMarchingConfig,
        hit_only: bool = False,
    ) -> np.ndarray:
        """Generate a point cloud wrt global coordinate"""

        with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
            # limiting numpy thread seems to make stable. but not sure why..
            co_proj_plane = self.copy_worldcoords()
            co_proj_plane.translate(np.array([1.0, 0.0, 0.0]))

            half_width = np.tan(self.config.fovx * 0.5)
            half_height = np.tan(self.config.fovy * 0.5)

            wlin = np.linspace(-half_width, half_width, self.config.resolx)
            hlin = np.linspace(-half_height, half_height, self.config.resoly)
            W, H = np.meshgrid(wlin, hlin)

            # First, create a points w.r.t projection plane
            grid_wh_local = np.array(list(zip(W.flatten(), H.flatten())))
            grid_local = np.hstack((np.zeros((len(grid_wh_local), 1)), grid_wh_local))
            grid_global = co_proj_plane.transform_vector(grid_local)

            ray_start = self.worldpos()
            ray_starts = np.tile(np.expand_dims(ray_start, axis=0), (len(grid_global), 1))
            ray_directions = grid_global - ray_start

            norms = np.linalg.norm(ray_directions, axis=1)
            ray_directions_unit = ray_directions / norms[:, None]

            dists = self.ray_marching(ray_starts, ray_directions_unit, sdf, rm_config)
            clouds = ray_starts + dists[:, None] * ray_directions_unit

            if hit_only:
                hit_indices = dists < rm_config.max_dist - 1e-3
                clouds = clouds[hit_indices]
        return clouds

    @staticmethod
    def ray_marching(
        pts_starts, direction_arr_unit, f_sdf, rm_config: RayMarchingConfig
    ) -> np.ndarray:
        pts_starts = copy.deepcopy(pts_starts)
        ray_tips = pts_starts
        n_point = len(pts_starts)
        active_flags = np.ones(n_point).astype(bool)
        frying_dists = np.zeros(n_point)
        for _ in range(rm_config.max_iter):
            active_indices = np.where(active_flags)[0]

            sd_vals_active = f_sdf(ray_tips[active_indices])
            diff = direction_arr_unit[active_flags] * sd_vals_active[:, None]
            ray_tips[active_indices] += diff
            frying_dists[active_indices] += sd_vals_active

            is_close_enough = sd_vals_active < rm_config.surface_closeness_threashold
            active_flags[active_indices[is_close_enough]] = False

            is_far_enough = frying_dists[active_indices] > rm_config.max_dist
            active_flags[active_indices[is_far_enough]] = False
            frying_dists[active_indices[is_far_enough]] = rm_config.max_dist
            ray_tips[active_indices[is_far_enough]] = np.inf

            active_ratio = np.sum(active_flags) / n_point
            if active_ratio < rm_config.terminate_active_rate_threshold:
                active_indices = np.where(active_flags)[0]
                break

        frying_dists[active_indices] = rm_config.max_dist  # type: ignore
        modified_dists = np.minimum(frying_dists, rm_config.max_dist)
        return modified_dists

    def get_voxbloxpy_camera_pose(self) -> CameraPose:
        pos = self.worldpos()
        rot = self.worldrot()
        quat = matrix2quaternion(rot)
        return CameraPose(pos, quat)


def create_synthetic_esdf(
    sdf: Callable[[np.ndarray], np.ndarray],
    camera: Camera,
    rm_config: RayMarchingConfig,
    esdf: Optional[EsdfMap] = None,
    max_dist: float = 2.5,
) -> EsdfMap:
    if esdf is None:
        esdf = EsdfMap.create(0.02)

    std_rotate = 0.03
    std_trans = 0.1
    n_camera = 2 + np.random.randint(4)
    # n_camera = 1
    for _ in range(n_camera):
        camera_random = copy.deepcopy(camera)
        camera_random.translate(np.random.randn(3) * std_trans)
        camera_random.rotate(np.random.randn() * std_rotate, axis="z")
        ts = time.time()
        cloud_global = camera_random.generate_point_cloud(sdf, rm_config=rm_config)
        logger.debug("elapsed time to generate cloud: {} sec".format(time.time() - ts))

        cloud_camera = camera_random.inverse_transform_vector(cloud_global)

        ts = time.time()
        esdf.update(camera_random.get_voxbloxpy_camera_pose(), cloud_camera)
        logger.debug("elapsed time to update esdf: {} sec".format(time.time() - ts))
    return esdf


def create_heightmap_z_slice(
    target_region: BoxSkeleton, objects: List[PrimitiveSkelton], resolution: int
) -> np.ndarray:
    extent_plane = target_region.extents[:2]
    xlin = np.linspace(-0.5 * extent_plane[0], +0.5 * extent_plane[0], resolution)
    ylin = np.linspace(-0.5 * extent_plane[1], +0.5 * extent_plane[1], resolution)
    X, Y = np.meshgrid(xlin, ylin)
    pts = np.column_stack((X.ravel(), Y.ravel(), np.zeros_like(X.ravel())))
    pts = target_region.transform_vector(pts)

    z_region_bottom = target_region.worldpos()[2] - 0.5 * target_region.extents[2]

    hmap_flatten = np.zeros_like(X.ravel())  # later we will reshape this
    for obj in objects:
        if isinstance(obj, BoxSkeleton):
            pos = obj.worldpos()
            h_max_obj = pos[2] + obj.extents[2] * 0.5
        elif isinstance(obj, CylinderSkelton):
            pos = obj.worldpos()
            h_max_obj = pos[2] + obj.height * 0.5
        else:
            raise ValueError("not supported type")
        pts[:, 2] = pos[2]
        bools_isinside = obj.sdf(pts) < 0.0
        hmap_flatten[bools_isinside] = h_max_obj - z_region_bottom
    hmap = hmap_flatten.reshape((resolution, resolution))
    return hmap


def create_heightmap_z_slice_cylinders(  # if all objects are cylinders
    target_region: BoxSkeleton, cylinders_param: np.ndarray, resolution: int
) -> np.ndarray:
    extent_plane = target_region.extents[:2]
    xlin = np.linspace(-0.5 * extent_plane[0], +0.5 * extent_plane[0], resolution)
    ylin = np.linspace(-0.5 * extent_plane[1], +0.5 * extent_plane[1], resolution)
    X, Y = np.meshgrid(xlin, ylin)
    pts = np.column_stack((X.ravel(), Y.ravel(), np.zeros_like(X.ravel())))
    pts = target_region.transform_vector(pts)

    z_region_bottom = target_region.worldpos()[2] - 0.5 * target_region.extents[2]

    hmap_flatten = np.zeros_like(X.ravel())  # later we will reshape this
    for param in cylinders_param:
        x, y, z, h, r = param
        h_max_obj = z + h * 0.5
        pts[:, 2] = z
        dists = np.linalg.norm(pts[:, :2] - (x, y), axis=1)
        hmap_flatten[dists < r] = h_max_obj - z_region_bottom
    hmap = hmap_flatten.reshape((resolution, resolution))
    return hmap


def create_voxelmap(
    target_region: BoxSkeleton,
    objects: List[PrimitiveSkelton],
    resolution: int,
    voxel_size: float,
    inf_subst_value: float = -1.0,
) -> np.ndarray:
    center = target_region.worldpos()
    extent = target_region.extents
    xlin = np.linspace(center[0] - 0.5 * extent[0], center[0] + 0.5 * extent[0], resolution)
    ylin = np.linspace(center[1] - 0.5 * extent[1], center[1] + 0.5 * extent[1], resolution)
    zlin = np.linspace(center[2] - 0.5 * extent[2], center[2] + 0.5 * extent[2], resolution)
    X, Y, Z = np.meshgrid(xlin, ylin, zlin)
    np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))


def create_heightmap_ray_projection(
    target_region: BoxSkeleton,
    objects: List[PrimitiveSkelton],
    resolution: int,
    inf_subst_value: float = -1.0,
    height_lower_cutoff: float = 0.0,
) -> np.ndarray:
    raymarching_conf = RayMarchingConfig()

    extent_plane = target_region.extents[:2]
    xlin = np.linspace(-0.5 * extent_plane[0], +0.5 * extent_plane[0], resolution)
    ylin = np.linspace(-0.5 * extent_plane[1], +0.5 * extent_plane[1], resolution)
    X, Y = np.meshgrid(xlin, ylin)
    height = target_region.extents[2]
    Z = np.zeros_like(X) + 0.5 * height

    points_wrt_region = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    points_wrt_world = target_region.transform_vector(points_wrt_region)
    dirs = np.tile(np.array([0, 0, -1]), (len(points_wrt_world), 1))

    # we must consider "ground" as obstacle
    ground = target_region.detach_clone()
    ground.translate([0.0, 0.0, -ground.extents[2]])

    union_sdf = UnionSDF([o.sdf for o in objects] + [ground.sdf])

    dists = Camera.ray_marching(points_wrt_world, dirs, union_sdf, raymarching_conf)
    is_valid = dists < height + 1e-3

    dists_from_ground = height - dists
    dists_from_ground[~is_valid] = inf_subst_value

    heightmap = dists_from_ground.reshape((resolution, resolution))
    heightmap[heightmap < height_lower_cutoff] = height_lower_cutoff
    return heightmap


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
    debug_points: Optional[np.ndarray]

    def __post_init__(self):
        print("LocatedHeightmap is currently deprecated.")

    @classmethod
    def by_raymarching(
        cls,
        target_region: BoxSkeleton,
        objects: Sequence[PrimitiveSkelton],
        conf: HeightmapConfig = HeightmapConfig(),
        raymarching_conf: RayMarchingConfig = RayMarchingConfig(),
        create_debug_points: bool = False,
        height_lower_cutoff: float = 0.0,
    ) -> "LocatedHeightmap":
        # although height map can be reated by rule-based method. But because
        # python is slow in loops thus, we resort to raymarching which can
        # be easily be vectorized.

        extent_plane = target_region.extents[:2]
        xlin = np.linspace(-0.5 * extent_plane[0], +0.5 * extent_plane[0], conf.resol_x)
        ylin = np.linspace(-0.5 * extent_plane[1], +0.5 * extent_plane[1], conf.resol_y)
        X, Y = np.meshgrid(xlin, ylin)
        height = target_region.extents[2]
        Z = np.zeros_like(X) + 0.5 * height

        points_wrt_region = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        points_wrt_world = target_region.transform_vector(points_wrt_region)
        dirs = np.tile(np.array([0, 0, -1]), (len(points_wrt_world), 1))

        # we must consider "ground" as obstacle
        ground = target_region.detach_clone()
        ground.translate([0.0, 0.0, -ground.extents[2]])

        union_sdf = UnionSDF([o.sdf for o in objects] + [ground.sdf])

        dists = Camera.ray_marching(points_wrt_world, dirs, union_sdf, raymarching_conf)
        is_valid = dists < height + 1e-3

        dists_from_ground = height - dists
        dists_from_ground[~is_valid] = conf.inf_subst_value

        if create_debug_points:
            debug_points = points_wrt_world + dists[:, None] * dirs
        else:
            debug_points = None

        heightmap = dists_from_ground.reshape((conf.resol_x, conf.resol_y))
        heightmap[heightmap < height_lower_cutoff] = height_lower_cutoff
        return cls(heightmap, target_region, conf, debug_points)

    @property
    def grid_width(self) -> np.ndarray:
        return self.surrounding_box.extents[:2] / np.array(
            [self.config.resol_x, self.config.resol_y]
        )

    def get_max_height(self, point_wrt_world: np.ndarray) -> float:
        # note that height from ground
        point_wrt_local = self.surrounding_box.inverse_transform_vector(point_wrt_world)
        idx_x, idx_y = np.floor(
            (point_wrt_local[:2] - (-0.5 * self.surrounding_box.extents[:2])) / self.grid_width
        ).astype(int)
        max_height_local = self.heightmap[idx_y, idx_x]
        return max_height_local

    def is_colliding(self, point_wrt_world: np.ndarray) -> bool:
        point_wrt_local = self.surrounding_box.inverse_transform_vector(point_wrt_world)
        return point_wrt_local[2] + 0.5 * self.surrounding_box.extents[2] < self.get_max_height(
            point_wrt_world
        )
