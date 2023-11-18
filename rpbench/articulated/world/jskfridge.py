from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
from skrobot.coordinates import CascadedCoords, Coordinates
from skrobot.model.primitives import Axis, Link
from skrobot.sdf import UnionSDF
from skrobot.viewers import TrimeshSceneViewer

from rpbench.articulated.vision import HeightmapConfig, LocatedHeightmap
from rpbench.articulated.world.utils import (
    BoxSkeleton,
    CylinderSkelton,
    PrimitiveSkelton,
)
from rpbench.interface import WorldBase
from rpbench.utils import SceneWrapper


@dataclass
class FridgeParameter:
    W: float = 0.54
    D: float = 0.53
    upper_H: float = 0.65
    container_w: float = 0.48
    container_h: float = 0.62
    container_d: float = 0.49
    panel_d: float = 0.29
    panel_t: float = 0.01
    # panel_hights: Tuple[float, ...] = (0.15, 0.29, 0.46)  # default panel
    panel_hights: Tuple[float, ...] = (0.15, 0.34, 0.46)  # slide up to upper slot
    door_D = 0.05
    lower_H = 0.81
    joint_x = -0.035
    joint_y = +0.015
    t_bump = 0.02
    d_bump = 0.06


@dataclass
class Region:
    box: BoxSkeleton
    obstacles: List[PrimitiveSkelton]

    def create_heightmap(self, n_grid: int = 56) -> np.ndarray:
        config = HeightmapConfig(n_grid, n_grid)
        hmap = LocatedHeightmap.by_raymarching(self.box, self.obstacles, conf=config)
        return hmap.heightmap


class FridgeModel(CascadedCoords):
    param: FridgeParameter
    links: List[BoxSkeleton]
    regions: List[Region]
    joint: CascadedCoords
    shelf: BoxSkeleton
    lower_box: BoxSkeleton
    joint_angle: float
    _visualizable_list: Optional[List[Link]]

    def __init__(self, joint_angle: float = 1.3, param: Optional[FridgeParameter] = None):

        super().__init__()
        if param is None:
            param = FridgeParameter()

        # define upper container
        t_side = 0.5 * (param.W - param.container_w)
        upper_container_co = CascadedCoords()
        side_panel_left = BoxSkeleton(
            [param.container_d, t_side, param.upper_H],
            pos=(0.5 * param.container_d, 0.5 * param.W - 0.5 * t_side, 0.5 * param.upper_H),
        )
        side_panel_right = BoxSkeleton(
            [param.container_d, t_side, param.upper_H],
            pos=(0.5 * param.container_d, -0.5 * param.W + 0.5 * t_side, 0.5 * param.upper_H),
        )
        upper_container_co.assoc(side_panel_left, relative_coords="world")
        upper_container_co.assoc(side_panel_right, relative_coords="world")

        t_top = param.upper_H - param.container_h
        top_panel = BoxSkeleton(
            [param.container_d, param.container_w, t_top],
            pos=(0.5 * param.container_d, 0.0, param.upper_H - 0.5 * t_top),
        )
        upper_container_co.assoc(top_panel, relative_coords="world")

        t_back = param.D - param.container_d
        back_panel = BoxSkeleton(
            [t_back, param.W, param.upper_H], pos=(param.D - 0.5 * t_back, 0.0, 0.5 * param.upper_H)
        )
        upper_container_co.assoc(back_panel, relative_coords="world")

        ditch = BoxSkeleton(
            [0.035, 0.32, 0.08],
            pos=(param.container_d - 0.5 * 0.032, 0.0, param.panel_hights[0] + 0.5 * 0.08),
        )
        upper_container_co.assoc(ditch, relative_coords="world")

        body_links = [side_panel_left, side_panel_right, top_panel, back_panel, ditch]

        for panel_h in param.panel_hights:
            panel = BoxSkeleton(
                [param.panel_d, param.container_w, param.panel_t],
                pos=(param.container_d - 0.5 * param.panel_d, 0.0, panel_h),
            )
            upper_container_co.assoc(panel, relative_coords="world")
            body_links.append(panel)

        # define regions
        regions: List[Region] = []
        tmp = np.array([0.0] + list(param.panel_hights) + [param.container_h])
        lowers, uppers = tmp[:-1], tmp[1:]
        for lower, upper in zip(lowers, uppers):
            box = BoxSkeleton(
                [param.panel_d, param.container_w, upper - lower],
                pos=(param.container_d - 0.5 * param.panel_d, 0.0, lower + 0.5 * (upper - lower)),
            )
            upper_container_co.assoc(box, relative_coords="world")
            regions.append(Region(box, []))

        # define joint
        joint = CascadedCoords(pos=(param.joint_x, -0.5 * param.W + param.joint_y, 0.0))
        upper_container_co.assoc(joint, relative_coords="world")

        # define door
        door = BoxSkeleton(
            [param.door_D, param.W, param.upper_H],
            pos=(-0.5 * param.door_D, 0.0, 0.5 * param.upper_H),
        )
        bump_left = BoxSkeleton(
            [param.d_bump, param.t_bump, param.container_h],
            pos=(
                +0.5 * param.d_bump,
                0.5 * param.container_w - 0.5 * param.t_bump,
                0.5 * param.container_h,
            ),
        )
        bump_right = BoxSkeleton(
            [param.d_bump, param.t_bump, param.container_h],
            pos=(
                +0.5 * param.d_bump,
                -0.5 * param.container_w + 0.5 * param.t_bump,
                0.5 * param.container_h,
            ),
        )
        joint.assoc(door, relative_coords="world")
        door.assoc(bump_left, relative_coords="world")
        door.assoc(bump_right, relative_coords="world")
        joint.rotate(joint_angle, "z")
        door_links = [door, bump_left, bump_right]

        # define lower box
        lower_box = BoxSkeleton(
            [param.D + param.door_D, param.W, param.lower_H],
            pos=(0.5 * param.D - 0.5 * param.door_D, 0.0, -0.5 * param.lower_H),
        )
        lower_box.assoc(upper_container_co, relative_coords="world")
        body_links.append(lower_box)
        lower_box.translate([0, 0, param.lower_H])

        # define side shelf
        side_shelf = BoxSkeleton(
            [1.0, 1.0, 1.13], pos=(0.5 - param.door_D, 0.5 * param.W + 0.5 + 0.02, 0.565)
        )
        lower_box.assoc(side_shelf, relative_coords="world")
        body_links.append(side_shelf)

        # define base
        self.assoc(lower_box, relative_coords="world")

        self.param = param
        self.joint = joint
        self.links = body_links + door_links
        self.body_links = body_links
        self.door_links = door_links
        self.shelf = side_shelf
        self.lower_box = lower_box
        self.regions = regions
        self.joint_angle = joint_angle
        self._visualizable_list = None

    def reset_joint_angle(self, joint_angle: float) -> None:
        # adhoc. maybe cumurate error
        diff = joint_angle - self.joint_angle
        self.joint.rotate(diff, "z")
        self.joint_angle = joint_angle

    def add(self, v: TrimeshSceneViewer) -> None:
        assert self._visualizable_list is None, "already added"
        visualizable_list = []
        for link in self.links:
            visualizable = link.to_visualizable((240, 240, 225, 150))
            v.add(visualizable)
            visualizable_list.append(visualizable)
        for region in self.regions:
            for obstacle in region.obstacles:
                visualizable = obstacle.to_visualizable((197, 245, 187, 255))
                v.add(visualizable)
                visualizable_list.append(visualizable)
        self._visualizable_list = visualizable_list

    def delete_from(self, v: TrimeshSceneViewer) -> None:
        assert self._visualizable_list is not None, "not added"
        print(self._visualizable_list)
        for visualizable in self._visualizable_list:
            v.delete(visualizable)


def randomize_region(region: Region, n_obstacles: int = 5):
    D, W, H = region.box._extents
    obstacle_h_max = H - 0.05
    obstacle_h_min = 0.1

    # determine pos-r pairs
    pairs = []
    while len(pairs) < n_obstacles:
        r = np.random.rand() * 0.03 + 0.02
        D_effective = D - 2 * r
        W_effective = W - 2 * r
        pos_2d = (
            np.random.rand(2) * np.array([D_effective, W_effective])
            - np.array([D_effective, W_effective]) * 0.5
        )
        if not any([np.linalg.norm(pos_2d - pos_2d2) < (r + r2) for pos_2d2, r2 in pairs]):
            pairs.append((pos_2d, r))

    for pos_2d, r in pairs:
        h = np.random.rand() * (obstacle_h_max - obstacle_h_min) + obstacle_h_min
        pos = np.array([*pos_2d, -0.5 * H + 0.5 * h])
        obstacle = CylinderSkelton(r, h, pos=pos)
        region.box.assoc(obstacle, relative_coords="local")
        region.obstacles.append(obstacle)


@dataclass
class JskFridgeWorld(WorldBase):
    fridge: FridgeModel
    _heightmap: Optional[np.ndarray] = None  # lazy

    def export_intrinsic_description(self) -> np.ndarray:
        raise NotImplementedError

    def heightmap(self) -> np.ndarray:
        if self._heightmap is None:
            self._heightmap = self.fridge.regions[2].create_heightmap()
        return self._heightmap

    @classmethod
    def sample(cls, standard: bool = False) -> Optional["JskFridgeWorld"]:
        fridge = FridgeModel(joint_angle=np.pi * 0.9)
        if not standard:
            n_obstacles = np.random.randint(1, 6)
            randomize_region(fridge.regions[1], n_obstacles)
        return cls(fridge, None)

    def visualize(self, viewer: Union[TrimeshSceneViewer, SceneWrapper]) -> None:
        self.fridge.add(viewer)  # type: ignore

    def get_exact_sdf(self) -> UnionSDF:
        sdfs = []
        for link in self.fridge.links:
            sdfs.append(link.sdf)
        for region in self.fridge.regions:
            for obstacle in region.obstacles:
                sdfs.append(obstacle.sdf)
        sdf = UnionSDF(sdfs)
        return sdf

    def sample_pose(self) -> Coordinates:
        region = self.fridge.regions[1]
        D, W, H = region.box.extents
        horizontal_margin = 0.08
        depth_margin = 0.03
        width_effective = np.array([D - 2 * depth_margin, W - 2 * horizontal_margin])
        sdf = self.get_exact_sdf()
        while True:
            trans = np.random.rand(2) * width_effective - 0.5 * width_effective
            trans = np.hstack([trans, -0.5 * H + 0.09])
            co = region.box.copy_worldcoords()
            co.translate(trans)
            if sdf(np.expand_dims(co.worldpos(), axis=0)) < 0.03:
                continue
            co.rotate(np.random.uniform(-(1.0 / 6.0) * np.pi, (1.0 / 6.0) * np.pi), "z")
            co_dummy = co.copy_worldcoords()
            co_dummy.translate([-0.07, 0.0, 0.0])
            if sdf(np.expand_dims(co_dummy.worldpos(), axis=0)) < 0.04:
                continue
            co_dummy.translate([-0.07, 0.0, 0.0])
            if sdf(np.expand_dims(co_dummy.worldpos(), axis=0)) < 0.04:
                continue
            return co


if __name__ == "__main__":
    from skrobot.viewers import TrimeshSceneViewer

    world = JskFridgeWorld.sample()
    v = TrimeshSceneViewer()
    world.visualize(v)
    for _ in range(100):
        pose = world.sample_pose()
        axis = Axis.from_coords(pose, axis_radius=0.001, axis_length=0.03)
        v.add(axis)
    v.show()
    import time

    time.sleep(1000)
