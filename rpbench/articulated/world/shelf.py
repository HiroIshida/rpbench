from dataclasses import dataclass, fields
from enum import Enum
from typing import List, Optional, Tuple, Union

import numpy as np
from skrobot.coordinates import CascadedCoords, Coordinates
from skrobot.model.primitives import Axis
from skrobot.viewers import TrimeshSceneViewer

from rpbench.articulated.world.utils import BoxSkeleton
from rpbench.planer_box_utils import sample_box
from rpbench.utils import SceneWrapper


class ShelfMock(CascadedCoords):
    shelf: BoxSkeleton
    target_region: BoxSkeleton
    percel: BoxSkeleton
    obs_list: List[BoxSkeleton]

    def __init__(
        self,
        shelf: BoxSkeleton,
        target_region: BoxSkeleton,
        percel: BoxSkeleton,
        obs_list: List[BoxSkeleton],
    ):
        self.shelf = shelf
        self.target_region = target_region
        self.percel = percel
        self.obs_list = obs_list

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
    def sample(cls, standard: bool = False, n_obstacle: int = 0) -> "ShelfMock":
        param = cls.Param()  # NOTE: dict will do, but dataclass is less buggy
        if standard:
            param.shelf_depth = 0.4
            param.region_height = 0.4
            param.region_pos2d = np.array([0.0, 1.2])
            param.percel_size = np.array([0.25, 0.25, 0.2])
        else:
            param.shelf_depth = np.random.rand() * 0.3 + 0.2
            param.region_height = 0.3 + np.random.rand() * 0.3
            param.region_pos2d = np.array([2.0, 1.8]) + np.array([-1.0, 0.0])
            param.percel_size = np.array([0.1, 0.1, 0.15]) + np.random.rand(3) * np.array(
                [0.2, 0.2, 0.15]
            )

        # define shelf
        d, w, h = param.shelf_depth, 3.0, 3.0
        shelf = BoxSkeleton([d, w, h])
        shelf.translate([0, 0, 0.5 * h])

        # define target region
        region_width = 0.6
        region_extents = np.hstack((param.shelf_depth, region_width, param.region_height))
        region_pos = np.hstack((0.0, param.region_pos2d))
        target_region = BoxSkeleton(region_extents, pos=region_pos)

        # define percel in the target region
        percel_box2d = sample_box(region_extents[:2], param.percel_size[:2], [])
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
                obs2d.coords.angle

                obs_size = np.hstack([obs_size_2d, height])
                obs = BoxSkeleton(obs_size, pos, True)
                target_region.assoc(obs, relative_coords="local")
                obs_list.append(obs)

        return cls(shelf, target_region, percel, obs_list)

    def grasp_poses(self) -> Tuple[Coordinates, Coordinates]:
        class GraspType(Enum):
            X_OBVERSE = 0
            X_REVERSE = 1
            Y_OBVERSE = 2
            Y_REVERSE = 3

        ex_percel, ey_percel, _ = self.percel.worldrot()
        ex_region, _, _ = self.target_region.worldrot()

        dots = [ex_region.dot(v) for v in [ex_percel, -ex_percel, ey_percel, -ey_percel]]
        gtype = GraspType(np.argmax(dots))

        co_right = self.percel.copy_worldcoords()
        co_left = self.percel.copy_worldcoords()
        d, w, h = self.percel.extents
        margin = 0.06
        if gtype in [GraspType.X_OBVERSE, GraspType.X_REVERSE]:
            co_right.translate([0.0, -0.5 * w - margin, 0.0])
            co_left.translate([0.0, +0.5 * w + margin, 0.0])

            # slide
            co_right.translate([-0.5 * d + margin, 0.0, 0.0])
            co_left.translate([-0.5 * d + margin, 0.0, 0.0])
        else:
            co_right.translate([-0.5 * d - margin, 0.0, 0.0])
            co_left.translate([+0.5 * d + margin, 0.0, 0.0])

            # slide
            co_right.translate([0.0, -0.5 * w + margin, 0.0])
            co_left.translate([0.0, -0.5 * w + margin, 0.0])

            co_right.rotate(0.5 * np.pi, "z")
            co_left.rotate(0.5 * np.pi, "z")

        if gtype in [GraspType.X_REVERSE, GraspType.Y_OBVERSE]:
            co_right.rotate(np.pi, "z")
            co_left.rotate(np.pi, "z")
            co_right, co_left = co_left, co_right

        co_right.translate([0.0, 0.0, 0.5 * h - 0.05])
        co_left.translate([0.0, 0.0, 0.5 * h - 0.05])
        return co_right, co_left

    def visualize(
        self, viewer: Union[TrimeshSceneViewer, SceneWrapper], show_grasp_pose: bool = False
    ) -> None:
        viewer.add(self.shelf.to_visualizable((255, 255, 255, 100)))
        viewer.add(self.target_region.to_visualizable((255, 0, 0, 150)))
        viewer.add(self.percel.to_visualizable((0, 255, 0, 150)))
        for obs in self.obs_list:
            viewer.add(obs.to_visualizable((0, 0, 255, 150)))

        if show_grasp_pose:
            co_right, co_left = self.grasp_poses()
            ax_right = Axis.from_coords(co_right)
            ax_left = Axis.from_coords(co_left)
            viewer.add(ax_right)
            viewer.add(ax_left)
