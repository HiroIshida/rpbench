import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
from skrobot.coordinates import CascadedCoords, Coordinates
from skrobot.sdf import UnionSDF
from skrobot.viewers import TrimeshSceneViewer

from rpbench.articulated.vision import HeightmapConfig, LocatedHeightmap
from rpbench.articulated.world.utils import BoxSkeleton, CylinderSkelton
from rpbench.interface import WorldBase
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

        target_region = BoxSkeleton(size, [0, 0, 0.5 * h])
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
    contents: List[CylinderSkelton]

    def __init__(self, fridge: Fridge, contents: List[CylinderSkelton]):
        super().__init__()
        self.assoc(fridge, wrt="local")
        for c in contents:
            self.assoc(c, wrt="local")
        self.fridge = fridge
        self.contents = contents

    @classmethod
    def sample(cls, standard: bool = False):
        fridge = Fridge.sample(standard)

        if standard:
            cylinder = CylinderSkelton(radius=0.02, height=0.12)
            co = fridge.copy_worldcoords()
            co.translate([0.0, 0.0, 0.06 + fridge.thickness])
            cylinder.newcoords(co)
            return cls(fridge, [cylinder])
        else:
            n_obs = np.random.randint(5) + 1
            contents: List[CylinderSkelton] = []

            def is_colliding(pos2d, radius):
                for c in contents:
                    dist = np.linalg.norm(pos2d - c.worldpos()[:2])
                    if dist < (c.radius + radius):
                        return True
                return False

            while len(contents) < n_obs:
                R = np.random.rand() * 0.05 + 0.05
                r = 0.5 * R
                h = np.random.rand() * 0.2 + 0.1

                available_size = fridge.size[:2] - fridge.thickness * 2 - r
                pos2d_wrt_fridge = np.random.rand(2) * available_size - available_size * 0.5

                if not is_colliding(pos2d_wrt_fridge, r):
                    c_new = CylinderSkelton(radius=r, height=h)
                    co = fridge.copy_worldcoords()
                    co.translate(np.hstack([pos2d_wrt_fridge, fridge.thickness + 0.5 * h]))
                    c_new.newcoords(co)
                    contents.append(c_new)
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

    def sample_pregrasp_coords(self) -> Optional[Coordinates]:
        n_budget = 100
        sdf = self.get_exact_sdf()
        for _ in range(n_budget):
            pos = np.random.rand(3) * self.fridge.size - 0.5 * self.fridge.size
            pos_world = self.transform_vector(pos)
            sd_val = sdf(np.expand_dims(pos_world, axis=0))[0]
            if sd_val > 0.05 and not self.fridge.is_outside(pos_world):
                co = Coordinates(pos_world)
                co.rotation = self.rotation
                angle = np.random.randn() * 0.2
                co.rotate(angle, "z")

                co_back = co.copy_worldcoords()
                co_back.translate([-0.1, 0.0, 0])

                sd_val = sdf(np.expand_dims(co_back.worldpos(), axis=0))[0]
                if sd_val > 0.05:
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

    @property
    def vector_dsecription(self) -> np.ndarray:
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
