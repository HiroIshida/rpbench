from dataclasses import dataclass, fields
from enum import Enum
from typing import List, Optional, Tuple, Union

import numpy as np
from skrobot.coordinates import CascadedCoords, Coordinates
from skrobot.model.primitives import Axis
from skrobot.sdf import UnionSDF
from skrobot.viewers import TrimeshSceneViewer

from rpbench.articulated.world.utils import BoxSkeleton
from rpbench.interface import SDFProtocol, WorldBase
from rpbench.planer_box_utils import Box2d, PlanerCoords, sample_box
from rpbench.utils import SceneWrapper


class ShelfMock(CascadedCoords):
    shelf: BoxSkeleton
    target_region: BoxSkeleton
    percel: BoxSkeleton
    obs_list: List[BoxSkeleton]
    shelf_sub_regions: List[BoxSkeleton]

    def __init__(
        self,
        shelf: BoxSkeleton,
        target_region: BoxSkeleton,
        percel: BoxSkeleton,
        obs_list: List[BoxSkeleton],
        shelf_sub_regions: List[BoxSkeleton],
    ):
        super().__init__()
        self.assoc(shelf)
        for sub in shelf_sub_regions:
            self.assoc(sub)
        self.assoc(target_region)
        self.shelf = shelf
        self.target_region = target_region
        self.percel = percel
        self.obs_list = obs_list
        self.shelf_sub_regions = shelf_sub_regions

    @dataclass
    class Param:
        shelf_depth: Optional[float] = None
        region_height: Optional[float] = None
        region_pos2d: Optional[np.ndarray] = None
        percel_size: Optional[np.ndarray] = None
        percel_pos2d: Optional[np.ndarray] = None
        percel_yaw: Optional[float] = None

        def to_vector(self) -> np.ndarray:
            vecs = []
            for f in fields(self):
                attr = getattr(self, f.name)
                vecs.append(attr)
            return np.hstack(vecs)

    @classmethod
    def sample(cls, standard: bool = False, n_obstacle: int = 0) -> Optional["ShelfMock"]:
        param = cls.Param()  # NOTE: dict will do, but dataclass is less buggy
        if standard:
            param.shelf_depth = 0.4
            param.region_height = 0.4
            param.region_pos2d = np.array([0.0, 1.2])
            param.percel_size = np.array([0.25, 0.25, 0.2])
        else:
            param.shelf_depth = np.random.rand() * 0.3 + 0.2
            param.region_height = 0.3 + np.random.rand() * 0.3

            region_max_height = 1.4
            region_min_height = param.region_height * 0.5
            param.region_pos2d = np.random.rand(2) * np.array(
                [1.0, region_max_height - region_min_height]
            ) + np.array([-0.5, region_min_height])
            param.percel_size = np.array([0.1, 0.1, 0.15]) + np.random.rand(3) * np.array(
                [0.2, 0.2, 0.15]
            )

        assert param.shelf_depth is not None
        assert param.region_height is not None
        assert param.region_pos2d is not None
        assert param.percel_size is not None

        # define shelf
        D, W, H = param.shelf_depth, 3.0, 3.0
        shelf = BoxSkeleton([D, W, H])
        shelf.translate([0, 0, 0.5 * H])

        # define target region
        region_width = 0.8
        region_extents = np.hstack((param.shelf_depth, region_width, param.region_height))
        region_pos = np.hstack((0.0, param.region_pos2d))  # type: ignore[arg-type]
        target_region = BoxSkeleton(region_extents, pos=region_pos)

        # define percel in the target region
        if standard:
            co = PlanerCoords(np.array([0.05, 0.0]), 0.0)
            percel_box2d = Box2d(param.percel_size[:2], co)
        else:
            ret = sample_box(region_extents[:2], param.percel_size[:2], [])
            if ret is None:
                return None
            percel_box2d = ret
        assert percel_box2d is not None
        param.percel_pos2d = percel_box2d.coords.pos
        param.percel_yaw = percel_box2d.coords.angle
        percel_pos = np.hstack(
            [param.percel_pos2d, 0.5 * param.percel_size[2] - 0.5 * param.region_height]
        )
        percel = BoxSkeleton(param.percel_size, percel_pos, True)
        percel.rotate(param.percel_yaw, "z")
        target_region.assoc(percel, relative_coords="local")

        # define obstacles
        grasp_poses = list(cls._get_grasp_poses(target_region, percel))
        grasp_poses_transed = [p.copy_worldcoords() for p in grasp_poses]
        print("trasp_poses", grasp_poses)
        for gp_transed in grasp_poses_transed:
            gp_transed.translate([-0.15, -0.15, 0.0])

        def is_colliding(obs: BoxSkeleton) -> bool:
            # check if new obstacle is colliding the grasp poses
            for pose in grasp_poses:
                sd = obs.sdf(np.expand_dims(pose.worldpos(), axis=0))[0]
                print(sd)
                if sd < 0.06:
                    return True

            return False

        mult_scale = 2.0
        plane_extents = np.array([param.shelf_depth, region_width * mult_scale])
        obs_list: List[BoxSkeleton] = []
        while len(obs_list) < n_obstacle:
            obs_size_2d = np.random.rand(2) * np.ones(2) * 0.25 + 0.05
            obs2d = sample_box(plane_extents, obs_size_2d, [percel_box2d])
            if obs2d is not None:
                heigh_margin = 0.05
                max_height = target_region.extents[2] - heigh_margin
                height = min(np.random.lognormal(0.0, 0.3) * (region_extents[2] / 2.5), max_height)
                pos = np.hstack([obs2d.coords.pos, 0.5 * height - 0.5 * param.region_height])

                obs_size = np.hstack([obs_size_2d, height])
                obs = BoxSkeleton(obs_size, pos, True)
                target_region.assoc(obs, relative_coords="local")
                if not is_colliding(obs):
                    obs_list.append(obs)
                else:
                    target_region.dissoc(obs)

        # define subregions
        d, w, h = target_region.extents
        _, y, z = target_region.worldpos()
        right_box_width = y - 0.5 * w + 0.5 * W
        right_box = BoxSkeleton(
            [D, right_box_width, H], pos=[0, y - 0.5 * w - 0.5 * right_box_width, 0.5 * H]
        )

        left_box_width = 0.5 * W - (y + 0.5 * w)
        left_box = BoxSkeleton(
            [D, left_box_width, H], pos=[0, y + 0.5 * w + 0.5 * left_box_width, 0.5 * H]
        )

        bottom_box_height = z - 0.5 * h
        bottom_box = BoxSkeleton([D, w, bottom_box_height], pos=[0, y, 0.5 * bottom_box_height])

        top_box_height = H - (z + 0.5 * h)
        top_box = BoxSkeleton(
            [D, w, top_box_height], pos=[0, y, z + 0.5 * h + 0.5 * top_box_height]
        )

        sub_regions = [right_box, left_box, bottom_box, top_box]

        return cls(shelf, target_region, percel, obs_list, sub_regions)

    def get_grasp_poses(self) -> Tuple[Coordinates, Coordinates]:
        return self._get_grasp_poses(self.target_region, self.percel)

    @staticmethod
    def _get_grasp_poses(
        target_region: BoxSkeleton, percel: BoxSkeleton
    ) -> Tuple[Coordinates, Coordinates]:
        class GraspType(Enum):
            X_OBVERSE = 0
            X_REVERSE = 1
            Y_OBVERSE = 2
            Y_REVERSE = 3

        ex_percel, ey_percel, _ = percel.worldrot()
        ex_region, _, _ = target_region.worldrot()

        dots = [ex_region.dot(v) for v in [ex_percel, -ex_percel, ey_percel, -ey_percel]]
        gtype = GraspType(np.argmax(dots))
        assert gtype != GraspType.X_REVERSE

        co_right = percel.copy_worldcoords()
        co_left = percel.copy_worldcoords()
        d, w, h = percel.extents
        margin_slide = 0.06
        margin_surface = 0.04
        if gtype in [GraspType.X_OBVERSE, GraspType.X_REVERSE]:
            co_right.translate([0.0, -0.5 * w - margin_surface, 0.0])
            co_left.translate([0.0, +0.5 * w + margin_surface, 0.0])

            # slide
            co_right.translate([-0.5 * d + margin_slide, 0.0, 0.0])
            co_left.translate([-0.5 * d + margin_slide, 0.0, 0.0])
        else:
            co_right.translate([-0.5 * d - margin_surface, 0.0, 0.0])
            co_left.translate([+0.5 * d + margin_surface, 0.0, 0.0])

            # slide
            co_right.translate([0.0, -0.5 * w + margin_slide, 0.0])
            co_left.translate([0.0, -0.5 * w + margin_slide, 0.0])

            co_right.rotate(0.5 * np.pi, "z")
            co_left.rotate(0.5 * np.pi, "z")

            if gtype == GraspType.Y_OBVERSE:
                co_right.rotate(np.pi, "z")
                co_left.rotate(np.pi, "z")

            if gtype == GraspType.Y_REVERSE:
                co_right, co_left = co_left, co_right

        co_right.translate([0.0, 0.0, 0.5 * h - 0.05])
        co_right.rotate(0.5 * np.pi, "x")
        co_left.translate([0.0, 0.0, 0.5 * h - 0.05])
        co_left.rotate(-0.5 * np.pi, "x")
        return co_right, co_left

    def visualize(
        self, viewer: Union[TrimeshSceneViewer, SceneWrapper], show_grasp_pose: bool = False
    ) -> None:
        for shelf_sub in self.shelf_sub_regions:
            viewer.add(shelf_sub.to_visualizable((255, 255, 255, 100)))
        viewer.add(self.target_region.to_visualizable((255, 0, 0, 150)))
        viewer.add(self.percel.to_visualizable((0, 255, 0, 150)))
        for obs in self.obs_list:
            viewer.add(obs.to_visualizable((0, 0, 255, 150)))

        if show_grasp_pose:
            co_right, co_left = self.get_grasp_poses()
            ax_right = Axis.from_coords(co_right)
            ax_left = Axis.from_coords(co_left)
            viewer.add(ax_right)
            viewer.add(ax_left)

    def get_exact_sdf(self) -> SDFProtocol:
        # NOTE: sdf subtraction (i.e. max(d1, -d2)) is not good for grad based solved thus...
        sdf_shelf = UnionSDF([sub.sdf for sub in self.shelf_sub_regions])

        def sdf_all(x: np.ndarray) -> np.ndarray:
            sdfs = [obs.sdf for obs in self.obs_list]
            sdfs.append(self.percel.sdf)
            sdfs.append(sdf_shelf)
            vals = np.vstack([f(x) for f in sdfs])  # type: ignore[misc]
            return np.min(vals, axis=0)

        return sdf_all


@dataclass
class ShelfWorld(WorldBase):
    shelf: ShelfMock

    @classmethod
    def sample(cls, standard: bool = False) -> "ShelfWorld":
        if standard:
            n_obs = 0
        else:
            n_obs = np.random.randint(10)
        while True:
            shelf = ShelfMock.sample(standard=standard, n_obstacle=n_obs)
            if shelf is not None:
                shelf.translate([0.8 + shelf.shelf.extents[0] * 0.5, 0.0, 0.0])
                return cls(shelf)

    def get_exact_sdf(self) -> SDFProtocol:
        return self.shelf.get_exact_sdf()

    def visualize(
        self, viewer: Union[TrimeshSceneViewer, SceneWrapper], show_grasp_pose: bool = False
    ) -> None:
        self.shelf.visualize(viewer, show_grasp_pose=show_grasp_pose)
