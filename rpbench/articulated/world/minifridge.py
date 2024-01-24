import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
from skrobot.coordinates import CascadedCoords, Coordinates
from skrobot.sdf import UnionSDF
from skrobot.viewers import TrimeshSceneViewer

from rpbench.articulated.vision import HeightmapConfig, LocatedHeightmap
from rpbench.articulated.world.utils import (
    BoxSkeleton,
    CylinderSkelton,
    PrimitiveSkelton,
)
from rpbench.interface import WorldBase
from rpbench.planer_box_utils import Box2d, Circle, PlanerCoords, is_colliding
from rpbench.utils import SceneWrapper

_HMAP_INF_SUBST = -1.0


class Fridge(CascadedCoords):
    panels: Dict[str, BoxSkeleton]
    size: np.ndarray
    thickness: float
    angle: float
    target_region: BoxSkeleton

    def __init__(self, size: np.ndarray, thickness: float, angle: float):
        CascadedCoords.__init__(self)
        d, w, h = size
        plane_xaxis = BoxSkeleton([thickness, w, h])
        plane_yaxis = BoxSkeleton([d, thickness, h])
        plane_zaxis = BoxSkeleton([d, w, thickness])

        bottom = copy.deepcopy(plane_zaxis)
        bottom.translate([0, 0, 0.5 * thickness])
        self.assoc(bottom, relative_coords="local")

        top = copy.deepcopy(plane_zaxis)
        top.translate([0, 0, h - 0.5 * thickness])
        self.assoc(top, relative_coords="local")

        right = copy.deepcopy(plane_yaxis)
        right.translate([0, -0.5 * w + 0.5 * thickness, 0.5 * h])
        self.assoc(right, relative_coords="local")

        left = copy.deepcopy(plane_yaxis)
        left.translate([0, +0.5 * w - 0.5 * thickness, 0.5 * h])
        self.assoc(left, relative_coords="local")

        back = copy.deepcopy(plane_xaxis)
        back.translate([0.5 * d - 0.5 * thickness, 0.0, 0.5 * h])
        self.assoc(back, relative_coords="local")

        door = copy.deepcopy(plane_xaxis)
        door.translate([-0.5 * d + 0.5 * thickness, 0.0, 0.5 * h])
        door.rotate(angle, [0, 0, 1.0])
        door.translate([-0.5 * w * np.sin(angle), 0.5 * w - 0.5 * w * np.cos(angle), 0.0])
        self.assoc(door, relative_coords="local")

        self.panels = {
            "bottom": bottom,
            "top": top,
            "right": right,
            "left": left,
            "back": back,
            "door": door,
        }

        target_region = BoxSkeleton(size - 2 * thickness, [0, 0, 0.5 * h])
        self.assoc(target_region, relative_coords="local")

        self.size = size
        self.thickness = thickness
        self.angle = angle
        self.target_region = target_region

    @classmethod
    def sample(cls, standard: bool = False) -> "Fridge":
        size = np.array([0.5, 0.5, 0.4])
        thickness = 0.06
        if standard:
            angle = 140 * (np.pi / 180.0)
        else:
            angle = (90 + np.random.rand() * 80) * (np.pi / 180.0)
        return cls(size, thickness, angle)

    def visualize(self, viewer: Union[TrimeshSceneViewer, SceneWrapper]) -> None:
        for panel in self.panels.values():
            viewer.add(panel.to_visualizable((255, 0, 0, 150)))

    def get_exact_sdf(self) -> UnionSDF:
        sdf = UnionSDF([p.sdf for p in self.panels.values()])
        return sdf

    def is_outside(self, pos: np.ndarray) -> bool:
        backward_margin = 0.1
        self.size + np.array([backward_margin, 0.0, 0.0])
        solid_box = BoxSkeleton(self.size)
        solid_box.newcoords(self.copy_worldcoords())
        solid_box.translate([-0.5 * backward_margin, 0.0, self.size[2] * 0.5])
        assert solid_box.sdf is not None
        val = solid_box.sdf(np.expand_dims(pos, axis=0))[0]
        return val > 0.0


class FridgeWithContents(CascadedCoords):
    fridge: Fridge
    contents: List[PrimitiveSkelton]

    def __init__(self, fridge: Fridge, contents: List[PrimitiveSkelton]):
        super().__init__()
        self.assoc(fridge, wrt="local")
        for c in contents:
            self.assoc(c, wrt="local", force=True)
        self.fridge = fridge
        self.contents = contents

    @staticmethod
    def sample_contents(target_region: BoxSkeleton, n_obstacles: int) -> List[PrimitiveSkelton]:
        D, W, H = target_region._extents
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

        contents: List[Any] = []
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
            contents.append(obj)
        return contents

    @classmethod
    def sample(cls, standard: bool = False):
        fridge = Fridge.sample(standard)

        if standard:
            cylinder = CylinderSkelton(radius=0.02, height=0.12)
            co = fridge.copy_worldcoords()
            co.translate([0.0, 0.0, 0.06 + fridge.thickness])
            cylinder.newcoords(co)
            return cls(fridge, [cylinder])
        n_obstacles = np.random.randint(1, 6)
        contents = cls.sample_contents(fridge.target_region, n_obstacles)
        for content in contents:
            fridge.target_region.assoc(content, relative_coords="local")
        return cls(fridge, contents)

    def visualize(self, viewer: Union[TrimeshSceneViewer, SceneWrapper]) -> None:
        self.fridge.visualize(viewer)
        for content in self.contents:
            viewer.add(content.to_visualizable((0, 255, 0, 150)))

    def get_exact_sdf(self) -> UnionSDF:
        fridge_sdf = self.fridge.get_exact_sdf()
        sdfs = [c.sdf for c in self.contents] + [fridge_sdf]
        sdf = UnionSDF(sdfs)
        return sdf

    @staticmethod
    def is_obviously_infeasible(sdf, co: Coordinates) -> bool:
        if sdf(np.expand_dims(co.worldpos(), axis=0)) < 0.03:
            return True
        co_dummy = co.copy_worldcoords()
        co_dummy.translate([-0.05, -0.05, 0.0])
        if sdf(np.expand_dims(co_dummy.worldpos(), axis=0)) < 0.03:
            return True

        co_dummy = co.copy_worldcoords()
        co_dummy.translate([-0.05, 0.05, 0.0])
        if sdf(np.expand_dims(co_dummy.worldpos(), axis=0)) < 0.03:
            return True

        co_dummy = co.copy_worldcoords()
        co_dummy.translate([-0.1, 0.0, 0.0])
        if sdf(np.expand_dims(co_dummy.worldpos(), axis=0)) < 0.05:
            return True

        co_dummy = co.copy_worldcoords()
        co_dummy.translate([-0.14, 0.0, 0.0])
        if sdf(np.expand_dims(co_dummy.worldpos(), axis=0)) < 0.05:
            return True

        co_dummy = co.copy_worldcoords()
        co_dummy.translate([-0.18, 0.0, 0.0])
        if sdf(np.expand_dims(co_dummy.worldpos(), axis=0)) < 0.05:
            return True

        return False

    def sample_pregrasp_coords(self) -> Optional[Coordinates]:
        region = self.fridge.target_region
        n_budget = 100
        sdf = self.get_exact_sdf()
        for _ in range(n_budget):
            pos = region.sample_points(1)[0]
            co = Coordinates(pos)
            yaw = np.random.uniform(-0.3 * np.pi, 0.3 * np.pi)
            co.rotate(yaw, "z")
            co.rotate(0.5 * np.pi, "x")

            if not self.is_obviously_infeasible(sdf, co):
                return co

        pos = self.transform_vector(np.zeros(3))
        co = Coordinates(pos)
        return co

    def create_heightmap(self, n_grid: int = 56) -> np.ndarray:
        hmap_config = HeightmapConfig(n_grid, n_grid)
        hmap = LocatedHeightmap.by_raymarching(
            self.fridge.target_region, self.contents, conf=hmap_config
        )
        return hmap.heightmap


@dataclass
class TabletopClutteredFridgeWorld(WorldBase):
    table: BoxSkeleton
    fridge_conts: FridgeWithContents
    _heightmap: Optional[np.ndarray] = None  # lazy

    def export_intrinsic_description(self) -> np.ndarray:
        return np.array([self.fridge_conts.fridge.angle])

    def heightmap(self) -> np.ndarray:
        if self._heightmap is None:
            self._heightmap = self.fridge_conts.create_heightmap()
        return self._heightmap

    @classmethod
    def sample(cls, standard: bool = False) -> Optional["TabletopClutteredFridgeWorld"]:
        table_size = np.array([0.6, 3.0, 0.8])
        table = BoxSkeleton(table_size)
        table.translate(np.array([0.0, 0.0, table_size[2] * 0.5]))

        fridge_conts = FridgeWithContents.sample(standard)
        fridge_conts.translate([0.0, 0.0, table_size[2]])

        slide = 0.6
        angle = 0.0
        table.translate([slide, 0.0, 0.0])
        table.rotate(angle, "z")
        fridge_conts.translate([slide, 0.0, 0.0])
        fridge_conts.rotate(angle, "z")
        return cls(table, fridge_conts)

    def visualize(self, viewer: Union[TrimeshSceneViewer, SceneWrapper]) -> None:
        self.fridge_conts.visualize(viewer)
        viewer.add(self.table.to_visualizable())

    def get_exact_sdf(self) -> UnionSDF:
        fridge_conts_sdf = self.fridge_conts.get_exact_sdf()
        sdf = UnionSDF([fridge_conts_sdf, self.table.sdf])
        return sdf

    def sample_pregrasp_coords(self) -> Optional[Coordinates]:
        return self.fridge_conts.sample_pregrasp_coords()
