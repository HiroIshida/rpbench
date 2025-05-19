from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
from plainmp.psdf import CylinderSDF, Pose, UnionSDF
from skrobot.coordinates import CascadedCoords, Coordinates
from skrobot.model.primitives import PointCloudLink
from skrobot.viewers import PyrenderViewer, TrimeshSceneViewer

from rpbench.articulated.vision import HeightmapConfig, LocatedHeightmap
from rpbench.articulated.world.utils import (
    BoxSkeleton,
    CylinderSkelton,
    PrimitiveSkelton,
)
from rpbench.interface import SamplableWorldBase
from rpbench.utils import SceneWrapper


class FridgeParameter:
    W: ClassVar[float] = 0.54
    D: ClassVar[float] = 0.53
    upper_H: ClassVar[float] = 0.65
    container_w: ClassVar[float] = 0.48
    container_h: ClassVar[float] = 0.62
    container_d: ClassVar[float] = 0.49
    panel_d: ClassVar[float] = 0.32
    panel_t: ClassVar[float] = 0.01
    # panel_hights: Tuple[float, ...] = (0.15, 0.34, 0.46)  # uncalibrated
    panel_hights: ClassVar[Tuple[float, ...]] = (0.15, 0.34, 0.48)
    door_D: ClassVar[float] = 0.05
    lower_H: ClassVar[float] = 0.81 + 0.02  # 0.02 if use calibrated
    joint_x: ClassVar[float] = -0.035
    joint_y: ClassVar[float] = +0.015
    t_bump: ClassVar[float] = 0.02
    d_bump: ClassVar[float] = 0.06


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
    _visualizable_table: Dict

    def __init__(self, joint_angle: float = 0.9 * np.pi, param: Optional[FridgeParameter] = None):

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
        self._visualizable_table = {}

    def reset_joint_angle(self, joint_angle: float) -> None:
        # adhoc. maybe cumurate error
        diff = joint_angle - self.joint_angle
        self.joint.rotate(diff, "z")
        self.joint_angle = joint_angle

    def add(self, v: TrimeshSceneViewer) -> None:
        assert len(self._visualizable_table) == 0, "already added"
        table = {}  # type: ignore
        for link in self.links:
            visualizable = link.to_visualizable((240, 240, 225, 100))
            v.add(visualizable)
            table[link] = visualizable
        for region in self.regions:
            for obstacle in region.obstacles:
                visualizable = obstacle.to_visualizable((150, 150, 150, 255))
                v.add(visualizable)
                table[obstacle] = visualizable  # type: ignore
        self._visualizable_table = table


_fridge_model = FridgeModel()
_fridge_model_sdf = UnionSDF([l.to_plainmp_sdf() for l in _fridge_model.links])


def get_fridge_model() -> FridgeModel:
    return _fridge_model  # no copy required, and not supposed to be modified


def get_fridge_model_sdf() -> UnionSDF:
    return _fridge_model_sdf.clone()  # copy required to avoid modification


def randomize_region(region: Region, n_obstacles: int = 5) -> np.ndarray:
    # randomize using only cylinder
    D, W, H = region.box._extents
    obstacle_h_max = H - 0.03
    obstacle_h_min = 0.05

    # determine pos-r pairs
    pairs = []  # type: ignore
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

    params = np.zeros(n_obstacles * 4)
    head = 0
    for pos_2d, r in pairs:
        h = np.random.rand() * (obstacle_h_max - obstacle_h_min) + obstacle_h_min
        params[head : head + 4] = np.array([*pos_2d, h, r])
        head += 4
    return params


@dataclass
class JskFridgeWorld(SamplableWorldBase):
    obstacles_param: np.ndarray
    attention_region_index: ClassVar[int] = 1
    N_MAX_OBSTACLES: ClassVar[int] = 5

    def export_intrinsic_description(self) -> np.ndarray:
        raise NotImplementedError

    def get_obstacle_list(self) -> List[CylinderSkelton]:
        region = get_fridge_model().regions[self.attention_region_index]
        H_region = region.box.extents[2]
        region_pos = region.box.worldpos()

        obstacle_list = []
        for param in self.obstacles_param.reshape(-1, 4):
            x, y, h, r = param
            pos_relative = np.hstack([x, y, -0.5 * H_region + 0.5 * h])
            pos = region_pos + pos_relative
            cylinder = CylinderSkelton(r, h, pos)
            obstacle_list.append(cylinder)
        return obstacle_list

    def visualize(self, viewer: Union[TrimeshSceneViewer, SceneWrapper]) -> None:
        fridge = FridgeModel()
        fridge.add(viewer)

        obs_list = self.get_obstacle_list()
        vis_list = []
        for obs in obs_list:
            visualizable = obs.to_visualizable((150, 150, 150, 255))
            viewer.add(visualizable)
            vis_list.append(visualizable)
        return vis_list

    def get_exact_sdf(self) -> UnionSDF:
        fridge_sdf = get_fridge_model_sdf()
        region = get_fridge_model().regions[self.attention_region_index]
        for param in self.obstacles_param.reshape(-1, 4):
            x, y, h, r = param
            H_region = region.box.extents[2]
            region_pos = region.box.worldpos()
            pos_relative = np.hstack([x, y, -0.5 * H_region + 0.5 * h])
            pos = region_pos + pos_relative
            pose = Pose(pos, np.eye(3))
            cylinder_sdf = CylinderSDF(r, h, pose)
            fridge_sdf.add(cylinder_sdf)
        return fridge_sdf

    @staticmethod
    def is_obviously_infeasible(sdf, co: Coordinates) -> bool:
        if sdf(np.expand_dims(co.worldpos(), axis=0)) < 0.02:
            return True
        co_dummy = co.copy_worldcoords()
        co_dummy.translate([-0.05, -0.05, 0.0])
        if sdf(np.expand_dims(co_dummy.worldpos(), axis=0)) < 0.02:
            return True

        co_dummy = co.copy_worldcoords()
        co_dummy.translate([-0.05, 0.05, 0.0])
        if sdf(np.expand_dims(co_dummy.worldpos(), axis=0)) < 0.02:
            return True

        co_dummy = co.copy_worldcoords()
        co_dummy.translate([-0.1, 0.0, 0.0])
        if sdf(np.expand_dims(co_dummy.worldpos(), axis=0)) < 0.03:
            return True

        return False

    @classmethod
    def sample(cls, standard: bool = False) -> Optional["JskFridgeWorld"]:
        fridge = FridgeModel(joint_angle=np.pi * 0.9)
        n_obstacles = np.random.randint(1, cls.N_MAX_OBSTACLES + 1)
        obstacles_param = randomize_region(fridge.regions[cls.attention_region_index], n_obstacles)
        return cls(obstacles_param)


if __name__ == "__main__":
    np.random.seed(0)
    world = JskFridgeWorld.sample()
    sdf = world.get_exact_sdf()
    points = np.random.randn(1000000, 3)
    points[:, 0] *= 0.7
    points[:, 1] *= 0.7
    points[:, 2] += 0.5
    sdf_values = sdf.evaluate_batch(points.T)
    points_inside = points[sdf_values < 0.0]
    link = PointCloudLink(points_inside)
    v = PyrenderViewer()
    world.visualize(v)
    v.add(link)
    v.show()
    import time

    time.sleep(1000)
