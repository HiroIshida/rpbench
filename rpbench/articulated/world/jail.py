import time
from dataclasses import dataclass
from typing import Callable, ClassVar, List, Union

import numpy as np
from scipy.spatial import KDTree
from skrobot.model.primitives import PointCloudLink
from skrobot.sdf import UnionSDF
from skrobot.viewers import PyrenderViewer, TrimeshSceneViewer

from rpbench.articulated.world.utils import (
    BoxSkeleton,
    CylinderSkelton,
    VoxelGrid,
    VoxelGridSkelton,
)
from rpbench.interface import SamplableWorldBase

BROWN_COLOR = (204, 102, 0, 200)


@dataclass
class JailWorld(SamplableWorldBase):
    region: BoxSkeleton
    panels: List[BoxSkeleton]
    voxels: VoxelGrid
    box_width: ClassVar[float] = 0.7
    box_height: ClassVar[float] = 0.7
    box_depth: ClassVar[float] = 0.6
    panel_thickness: ClassVar[float] = 0.01

    @classmethod
    def create_region_and_panels(cls):
        # define attention region
        region = BoxSkeleton([cls.box_depth, cls.box_width, cls.box_height])
        region.translate([0, 0, cls.box_height * 0.5])

        # jail box
        bottom = BoxSkeleton([cls.box_depth, cls.box_width, cls.panel_thickness])
        bottom.translate([0, 0, cls.panel_thickness * 0.5])
        top = BoxSkeleton([cls.box_depth, cls.box_width, cls.panel_thickness])
        top.translate([0, 0, cls.box_height - cls.panel_thickness * 0.5])
        left = BoxSkeleton([cls.box_depth, cls.panel_thickness, cls.box_height])
        left.translate([0, cls.box_width * 0.5 - cls.panel_thickness * 0.5, cls.box_height * 0.5])
        right = BoxSkeleton([cls.box_depth, cls.panel_thickness, cls.box_height])
        right.translate([0, -cls.box_width * 0.5 + cls.panel_thickness * 0.5, cls.box_height * 0.5])
        panels = [bottom, top, left, right]
        for panel in panels:
            region.assoc(panel)
        # slide jail
        region.translate([0.65, 0, 0.7])
        return region, panels

    @classmethod
    def sample(cls, standard: bool = False) -> "JailWorld":
        region, panels = cls.create_region_and_panels()

        # sample jail bars
        sizes = np.array([cls.box_depth, cls.box_width, cls.box_height])
        margin = 0.1
        radius_min = 0.015
        radius_max = 0.05
        lb = -sizes * 0.5 + margin
        ub = sizes * 0.5 - margin
        n_bar = np.random.randint(1, 6)
        infinite_length = 10.0
        bars = []
        for _ in range(n_bar):
            pos = np.random.uniform(lb, ub)
            radius = np.random.uniform(radius_min, radius_max)
            bar = CylinderSkelton(radius, infinite_length)
            bar.translate(pos)
            roll = np.random.uniform(0, np.pi / 4)
            bar.rotate(roll, axis=[1, 0, 0])
            pitch = np.random.uniform(0, 2 * np.pi)
            bar.rotate(pitch, axis=[0, 0, 1], wrt="world")
            bars.append(bar)
        for bar in bars:
            region.assoc(bar, relative_coords="local")

        bar_sdf = UnionSDF([bar.sdf for bar in bars])

        # create voxel grid
        voxel_skeleton = VoxelGridSkelton.from_box(region, (56, 56, 56))
        voxel_grid = VoxelGrid.from_sdf(bar_sdf, voxel_skeleton)
        return cls(region, panels, voxel_grid)

    def visualize(self, viewer: Union[PyrenderViewer, TrimeshSceneViewer]):
        for panel in self.panels:
            viewer.add(panel.to_visualizable(BROWN_COLOR))
        cloud = self.voxels.to_points()
        plink = PointCloudLink(cloud)
        viewer.add(plink)

    def get_sdf(self) -> Callable[[np.ndarray], np.ndarray]:
        kdtree = KDTree(self.voxels.to_points())

        def sdf(X: np.ndarray) -> np.ndarray:
            dists, _ = kdtree.query(X)
            for panel in self.panels:
                dists = np.minimum(dists, panel.sdf(X))
            return dists

        return sdf

    def serialize(self) -> bytes:
        return self.voxels.serialize()

    @classmethod
    def deserialize(cls, data: bytes) -> "JailWorld":
        voxels = VoxelGrid.deserialize(data)
        region, panels = cls.create_region_and_panels()
        return cls(region, panels, voxels)


if __name__ == "__main__":
    import tqdm
    from skrobot.models.fetch import Fetch

    for _ in tqdm.tqdm(range(20)):
        world = JailWorld.sample()
        # n_bytes = len(world.serialize())
        # print(f"n_bytes: {n_bytes}")

    world_rt = JailWorld.deserialize(world.serialize())
    ts = time.time()
    sdf = world_rt.get_sdf()
    print(f"elapsed time: {time.time() - ts}")
    viewer = PyrenderViewer()
    # viewer = TrimeshSceneViewer()
    world_rt.visualize(viewer)
    fetch = Fetch()
    fetch.reset_pose()
    viewer.add(fetch)
    viewer.show()
    import time

    time.sleep(1000)
