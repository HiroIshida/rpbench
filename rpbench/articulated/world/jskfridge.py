import copy
from dataclasses import dataclass
from functools import lru_cache
from typing import ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import ycb_utils
from skrobot.coordinates import CascadedCoords, Coordinates
from skrobot.sdf import UnionSDF
from skrobot.viewers import TrimeshSceneViewer
from trimesh import Trimesh

from rpbench.articulated.vision import HeightmapConfig, LocatedHeightmap
from rpbench.articulated.world.utils import (
    BoxSkeleton,
    CylinderSkelton,
    MeshSkelton,
    PrimitiveSkelton,
)
from rpbench.interface import WorldBase
from rpbench.planer_box_utils import Box2d, Circle, PlanerCoords, is_colliding
from rpbench.utils import SceneWrapper


@dataclass
class FridgeParameter:
    W: float = 0.54
    D: float = 0.53
    upper_H: float = 0.65
    container_w: float = 0.48
    container_h: float = 0.62
    container_d: float = 0.49
    panel_d: float = 0.32
    panel_t: float = 0.01
    # panel_hights: Tuple[float, ...] = (0.15, 0.34, 0.46)  # uncalibrated
    panel_hights: Tuple[float, ...] = (0.15, 0.34, 0.48)
    door_D = 0.05
    lower_H = 0.81 + 0.02  # 0.02 if use calibrated
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
    _visualizable_table: Dict

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


def randomize_region(region: Region, n_obstacles: int = 5):
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

    for pos_2d, r in pairs:
        h = np.random.rand() * (obstacle_h_max - obstacle_h_min) + obstacle_h_min
        pos = np.array([*pos_2d, -0.5 * H + 0.5 * h])
        obstacle = CylinderSkelton(r, h, pos=pos)
        region.box.assoc(obstacle, relative_coords="local")
        region.obstacles.append(obstacle)


def randomize_region2(region: Region, n_obstacles: int = 5):
    # randomize using both cylinder and box
    D, W, H = region.box._extents
    obstacle_h_max = H - 0.03
    obstacle_h_min = 0.05

    region2d = Box2d(np.array([D, W]), PlanerCoords.standard())

    obj2d_list = []  # type: ignore
    while len(obj2d_list) < n_obstacles:
        center = region2d.sample_point()
        sample_circle = np.random.rand() < 0.5
        if sample_circle:
            r = np.random.rand() * 0.03 + 0.02
            obj2d = Circle(center, r)
        else:
            w = np.random.uniform(0.05, 0.1)
            d = np.random.uniform(0.05, 0.1)
            yaw = np.random.uniform(0.0, np.pi)
            obj2d = Box2d(np.array([w, d]), PlanerCoords(center, yaw))  # type: ignore

        if not region2d.contains(obj2d):
            continue
        if any([is_colliding(obj2d, o) for o in obj2d_list]):
            continue
        obj2d_list.append(obj2d)

    for obj2d in obj2d_list:
        h = np.random.rand() * (obstacle_h_max - obstacle_h_min) + obstacle_h_min
        if isinstance(obj2d, Box2d):
            extent = np.hstack([obj2d.extent, h])
            obj = BoxSkeleton(extent, pos=np.hstack([obj2d.coords.pos, 0.0]))
            obj.rotate(obj2d.coords.angle, "z")
        elif isinstance(obj2d, Circle):
            obj = CylinderSkelton(obj2d.radius, h, pos=np.hstack([obj2d.center, 0.0]))
        else:
            assert False
        obj.translate([0.0, 0.0, -0.5 * H + 0.5 * h])
        region.box.assoc(obj, relative_coords="local")
        region.obstacles.append(obj)


@lru_cache(maxsize=None)
def ycb_utils_load_singleton(name: str, scale: float = 1.0) -> Trimesh:
    return ycb_utils.load_with_scale(name, scale=scale)


def randomize_region3(region: Region, n_obstacles: int = 5):

    mesh_list = [
        ycb_utils_load_singleton("006_mustard_bottle", scale=0.75),
        ycb_utils_load_singleton("010_potted_meat_can"),
        ycb_utils_load_singleton("013_apple"),
        ycb_utils_load_singleton("019_pitcher_base", scale=0.6),
    ]
    skelton_list = [MeshSkelton(mesh, fill_value=0.03, dim_grid=30) for mesh in mesh_list]
    box = region.box

    obj_list = []  # type: ignore
    while len(obj_list) < n_obstacles:
        skelton = copy.deepcopy(np.random.choice(skelton_list))
        assert isinstance(skelton, MeshSkelton)
        pos = box.sample_points(1)[0]
        pos[2] = box.worldpos()[2] - 0.5 * box._extents[2] + 0.01
        skelton.newcoords(Coordinates(pos))
        yaw = np.random.uniform(0.0, 2 * np.pi)
        skelton.rotate(yaw, "z")

        values = box.sdf(skelton.surface_points)
        is_containd = np.all(values < -0.005)
        if not is_containd:
            continue

        is_colliding = False
        for obj in obj_list:
            if np.any(skelton.sdf(obj.surface_points) < 0.0):
                is_colliding = True
                break
        if is_colliding:
            continue
        skelton.sdf.itp.fill_value = 1.0
        obj_list.append(skelton)
        region.box.assoc(skelton, relative_coords="world")
        region.obstacles.append(skelton)


@dataclass
class JskFridgeWorldBase(WorldBase):
    fridge: FridgeModel
    _heightmap: Optional[np.ndarray] = None  # lazy
    attention_region_index: ClassVar[int] = 1

    def export_intrinsic_description(self) -> np.ndarray:
        raise NotImplementedError

    def heightmap(self) -> np.ndarray:
        if self._heightmap is None:
            self._heightmap = self.fridge.regions[self.attention_region_index].create_heightmap()
        return self._heightmap

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
        region = self.fridge.regions[self.attention_region_index]
        D, W, H = region.box.extents
        horizontal_margin = 0.08
        depth_margin = 0.03
        width_effective = np.array([D - 2 * depth_margin, W - 2 * horizontal_margin])
        sdf = self.get_exact_sdf()

        n_max_trial = 100
        for _ in range(n_max_trial):
            trans = np.random.rand(2) * width_effective - 0.5 * width_effective
            trans = np.hstack([trans, -0.5 * H + 0.09])
            co = region.box.copy_worldcoords()
            co.translate(trans)
            if sdf(np.expand_dims(co.worldpos(), axis=0)) < 0.03:
                continue
            co.rotate(np.random.uniform(-(1.0 / 4.0) * np.pi, (1.0 / 4.0) * np.pi), "z")
            co_dummy = co.copy_worldcoords()
            co_dummy.translate([-0.07, 0.0, 0.0])
            if sdf(np.expand_dims(co_dummy.worldpos(), axis=0)) < 0.04:
                continue
            co_dummy.translate([-0.07, 0.0, 0.0])
            if sdf(np.expand_dims(co_dummy.worldpos(), axis=0)) < 0.04:
                continue
            return co
        return co  # invalid one but no choice

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

    def sample_pose_vertical(self) -> Coordinates:
        # NOTE: unlike sample pose height is also sampled
        region = self.fridge.regions[self.attention_region_index]
        b_min = -0.5 * region.box.extents
        b_max = +0.5 * region.box.extents

        horizontal_margin = 0.05
        height_margin = 0.06
        b_min[
            0
        ] -= horizontal_margin  # - is not a mistake. this makes it possible to pose be slightly outside
        b_max[0] -= horizontal_margin
        b_min[2] += height_margin
        b_max[2] -= height_margin

        sdf = self.get_exact_sdf()

        n_max_trial = 100
        for _ in range(n_max_trial):
            trans = np.random.rand(3) * (b_max - b_min) + b_min
            co = region.box.copy_worldcoords()
            co.translate(trans)
            co.rotate(np.random.uniform(-(1.0 / 4.0) * np.pi, (1.0 / 4.0) * np.pi), "z")
            if self.is_obviously_infeasible(sdf, co):
                continue
            co.rotate(0.5 * np.pi, "x")
            if self.is_obviously_infeasible(sdf, co):
                continue

            # NOTE: these conditions are effective to exclude collision poses
            # but it's reduce the "domain" size, which becomes problematic
            # to detect infeasible pose.
            # co_dummy = co.copy_worldcoords()
            # co_dummy.translate([-0.07, 0.0, 0.0])
            # if sdf(np.expand_dims(co_dummy.worldpos(), axis=0)) < 0.04:
            #     continue
            # co_dummy.translate([-0.07, 0.0, 0.0])
            # if sdf(np.expand_dims(co_dummy.worldpos(), axis=0)) < 0.04:
            #     continue
            return co
        return co  # invalid one but no choice


class JskFridgeWorld(JskFridgeWorldBase):
    @classmethod
    def sample(cls, standard: bool = False) -> Optional["JskFridgeWorld"]:
        fridge = FridgeModel(joint_angle=np.pi * 0.9)
        if not standard:
            n_obstacles = np.random.randint(1, 6)
            randomize_region(fridge.regions[cls.attention_region_index], n_obstacles)
        return cls(fridge, None)


class JskFridgeWorld2(JskFridgeWorldBase):
    @classmethod
    def sample(cls, standard: bool = False) -> Optional["JskFridgeWorld2"]:
        fridge = FridgeModel(joint_angle=np.pi * 0.9)
        if not standard:
            n_obstacles = np.random.randint(1, 6)
            randomize_region2(fridge.regions[cls.attention_region_index], n_obstacles)
        return cls(fridge, None)


class JskFridgeWorld3(JskFridgeWorldBase):
    @classmethod
    def sample(cls, standard: bool = False) -> Optional["JskFridgeWorld3"]:
        fridge = FridgeModel(joint_angle=np.pi * 0.9)
        if not standard:
            n_obstacles = np.random.randint(1, 6)
            randomize_region3(fridge.regions[cls.attention_region_index], n_obstacles)
        return cls(fridge, None)


if __name__ == "__main__":
    import time

    import matplotlib.pyplot as plt
    import numpy as np
    from skrobot.sdf import UnionSDF
    from skrobot.viewers import TrimeshSceneViewer

    np.random.seed(0)
    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()
    world = JskFridgeWorld3.sample()
    profiler.stop()
    print(profiler.output_text(unicode=True, color=True, show_all=True))

    v = TrimeshSceneViewer()
    world.visualize(v)
    v.show()
    ts = time.time()
    hmap = world.heightmap()
    print(time.time() - ts)
    fig, ax = plt.subplots()
    ax.imshow(hmap)
    plt.show()
