import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
from skrobot.coordinates import CascadedCoords, Coordinates
from skrobot.sdf import UnionSDF
from skrobot.viewers import TrimeshSceneViewer

from rpbench.articulated.vision import Camera, RayMarchingConfig
from rpbench.articulated.world.utils import BoxSkeleton, CylinderSkelton
from rpbench.interface import WorldBase
from rpbench.utils import SceneWrapper

_HMAP_INF_SUBST = -1.0


class Oven(CascadedCoords):
    panels: Dict[str, BoxSkeleton]
    size: np.ndarray
    thickness: float
    angle: float

    def __init__(self, size: np.ndarray, thickness: float, angle: float):
        CascadedCoords.__init__(self)
        d, w, h = size
        plane_xaxis = BoxSkeleton([thickness, w, h], with_sdf=True)
        plane_yaxis = BoxSkeleton([d, thickness, h], with_sdf=True)
        plane_zaxis = BoxSkeleton([d, w, thickness], with_sdf=True)

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
        self.size = size
        self.thickness = thickness
        self.angle = angle

    @classmethod
    def sample(cls, standard: bool = False) -> "Oven":
        if standard:
            size = np.array([0.4, 0.5, 0.4])
            thickness = 0.03
            angle = 140 * (np.pi / 180.0)
        else:
            size = np.array([0.3, 0.4, 0.3]) + np.random.rand(3) * 0.2
            thickness = 0.02 + np.random.rand() * 0.03
            angle = (40 + np.random.rand() * 120) * (np.pi / 180.0)
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
        solid_box = BoxSkeleton(self.size, with_sdf=True)
        solid_box.newcoords(self.copy_worldcoords())
        solid_box.translate([-0.5 * backward_margin, 0.0, self.size[2] * 0.5])
        assert solid_box.sdf is not None
        val = solid_box.sdf(np.expand_dims(pos, axis=0))[0]
        return val > 0.0


class OvenWithContents(CascadedCoords):
    oven: Oven
    contents: List[CylinderSkelton]

    def __init__(self, oven: Oven, contents: List[CylinderSkelton]):
        super().__init__()
        self.assoc(oven, wrt="local")
        for c in contents:
            self.assoc(c, wrt="local")
        self.oven = oven
        self.contents = contents

    @classmethod
    def sample(cls, standard: bool = False):
        oven = Oven.sample(standard)

        if standard:
            cylinder = CylinderSkelton(radius=0.02, height=0.12, with_sdf=True)
            co = oven.copy_worldcoords()
            co.translate([0.0, 0.0, 0.06 + oven.thickness])
            cylinder.newcoords(co)
            return cls(oven, [cylinder])
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

                available_size = oven.size[:2] - oven.thickness * 2 - r
                pos2d_wrt_oven = np.random.rand(2) * available_size - available_size * 0.5

                if not is_colliding(pos2d_wrt_oven, r):
                    c_new = CylinderSkelton(radius=r, height=h, with_sdf=True)
                    co = oven.copy_worldcoords()
                    co.translate(np.hstack([pos2d_wrt_oven, oven.thickness + 0.5 * h]))
                    c_new.newcoords(co)
                    contents.append(c_new)
            return cls(oven, contents)

    def visualize(self, viewer: Union[TrimeshSceneViewer, SceneWrapper]) -> None:
        self.oven.visualize(viewer)
        for content in self.contents:
            viewer.add(content.to_visualizable((0, 255, 0, 150)))

    def get_exact_sdf(self) -> UnionSDF:
        oven_sdf = self.oven.get_exact_sdf()
        sdfs = [c.sdf for c in self.contents] + [oven_sdf]
        sdf = UnionSDF(sdfs)
        return sdf

    def sample_pregrasp_coords(self) -> Optional[Coordinates]:
        idx = np.random.randint(len(self.contents))
        cylinder = self.contents[idx]
        co = cylinder.copy_worldcoords()
        rot_angle = np.random.rand() * np.pi - 0.5 * np.pi

        min_height = 0.07
        grasp_height = (
            np.random.rand() * (cylinder.height - min_height) + min_height - cylinder.height * 0.5
        )
        co.rotate(rot_angle, "z")
        co.translate([-0.1 - cylinder.radius, 0.0, grasp_height])

        sdf = self.get_exact_sdf()
        dist = sdf(np.expand_dims(co.worldpos(), axis=0))[0]
        if dist < 0.05 or self.oven.is_outside(co.worldpos()):
            return None
        return co

    def create_heightmap(self, n_grid: int = 56) -> np.ndarray:
        available_size = self.oven.size[:2] - 2 * self.oven.thickness
        b_min_wrt_oven = -available_size * 0.5
        b_max_wrt_oven = +available_size * 0.5
        xlin = np.linspace(b_min_wrt_oven[0], b_max_wrt_oven[0], n_grid)
        ylin = np.linspace(b_min_wrt_oven[1], b_max_wrt_oven[1], n_grid)
        X, Y = np.meshgrid(xlin, ylin)
        height_from_plane = 0.5
        Z = np.zeros_like(X) + height_from_plane

        points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        points = self.transform_vector(points)
        dirs = np.tile(np.array([0, 0, -1]), (len(points), 1))

        # create sdf
        sdfs = []
        sdfs.append(self.oven.panels["bottom"].sdf)
        for c in self.contents:
            sdfs.append(c.sdf)
        union_sdf = UnionSDF(sdfs)

        conf = RayMarchingConfig()
        dists = Camera.ray_marching(points, dirs, union_sdf, conf)
        is_valid = dists < height_from_plane + 1e-3

        dists_from_ground = height_from_plane - dists
        dists_from_ground[~is_valid] = _HMAP_INF_SUBST
        # points_hit = points + dists[:, None] * dirs
        # points_hit_z = points_hit[:, 2]
        # points_hit_z[~is_valid] = _HMAP_INF_SUBST

        self._heightmap = dists_from_ground.reshape((n_grid, n_grid))
        # from IPython import embed; embed()
        return self._heightmap
        # return points


@dataclass
class TabletopClutteredOvenWorld(WorldBase):
    table: BoxSkeleton
    oven_conts: OvenWithContents
    _heightmap: Optional[np.ndarray] = None  # lazy

    @classmethod
    def sample(cls, standard: bool = False) -> Optional["TabletopClutteredOvenWorld"]:
        if standard:
            table_size = np.array([0.6, 3.0, 0.7])
        else:
            table_size = np.array([0.6, 3.0, np.random.rand() * 0.2 + 0.6])
        table = BoxSkeleton(table_size, with_sdf=True)
        table.translate(np.array([0.0, 0.0, table_size[2] * 0.5]))

        oven_conts = OvenWithContents.sample(standard)
        if standard:
            oven_conts.translate([0.0, 0.0, table_size[2]])
        else:
            oven_conts.translate([np.random.rand() * 0.3 - 0.15, 0.0, table_size[2]])
            print(oven_conts.worldpos())

        # slide table and oven to some extent
        slide = 1.0 + np.random.rand() * 0.5
        angle = np.random.rand() * 0.5 * np.pi - 0.25 * np.pi
        table.translate([slide, 0.0, 0.0])
        table.rotate(angle, "z")
        oven_conts.translate([slide, 0.0, 0.0])
        oven_conts.rotate(angle, "z")
        return cls(table, oven_conts)

    def visualize(self, viewer: Union[TrimeshSceneViewer, SceneWrapper]) -> None:
        self.oven_conts.visualize(viewer)
        viewer.add(self.table.to_visualizable())

    def get_exact_sdf(self) -> UnionSDF:
        oven_conts_sdf = self.oven_conts.get_exact_sdf()
        sdf = UnionSDF([oven_conts_sdf, self.table.sdf])
        return sdf

    def sample_pregrasp_coords(self) -> Optional[Coordinates]:
        return self.oven_conts.sample_pregrasp_coords()
