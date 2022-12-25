from dataclasses import dataclass
from typing import List, Tuple, Type

import numpy as np
from skrobot.coordinates import Coordinates
from skrobot.model.link import Link
from skrobot.model.primitives import Box
from skrobot.sdf import UnionSDF
from voxbloxpy.core import Grid, GridSDF

from rpbench.interface import DescriptionTable, ProblemBase, WorldBase
from rpbench.utils import skcoords_to_pose_vec


@dataclass
class TabletopWorldBase(WorldBase):
    table: Box
    obstacles: List[Link]


@dataclass
class TabletopBoxWorld(TabletopWorldBase):
    box_center: Coordinates
    box_d: float
    box_w: float
    box_h: float
    box_t: float

    def get_exact_sdf(self) -> UnionSDF:
        lst = [self.table.sdf]
        for obstacle in self.obstacles:
            lst.append(obstacle.sdf)
        return UnionSDF(lst)

    @classmethod
    def sample(cls, standard: bool = False) -> "TabletopBoxWorld":
        table = cls.create_standard_table()
        table_depth, table_width, table_height = table._extents
        x = np.random.rand() * 0.2
        y = -0.2 + np.random.rand() * 0.4
        z = 0.0
        table.translate([x, y, z])

        table_tip = table.copy_worldcoords()
        table_tip.translate([-table_depth * 0.5, -table_width * 0.5, +0.5 * table_height])

        obstacles = []

        color = np.array([255, 220, 0, 150])

        # box
        d = 0.2 + np.random.rand() * 0.3
        w = 0.3 + np.random.rand() * 0.3
        h = 0.2 + np.random.rand() * 0.3
        t = 0.03

        if standard:
            box_center = table.copy_worldcoords()
            box_center.translate([0, 0, 0.5 * table_height])
        else:
            box_center = table_tip.copy_worldcoords()
            box_center.translate([0.5 * d, 0.5 * w, 0.0])
            pos_from_tip = np.array([table_depth - d, table_width - w, 0]) * np.random.rand(3)
            box_center.translate(pos_from_tip)

        lower_plate = Box([d, w, t], with_sdf=True, face_colors=color)
        lower_plate.newcoords(box_center.copy_worldcoords())
        lower_plate.translate([0, 0, 0.5 * t])
        obstacles.append(lower_plate)

        upper_plate = Box([d, w, t], with_sdf=True, face_colors=color)
        upper_plate.newcoords(box_center.copy_worldcoords())
        upper_plate.translate([0, 0, h - 0.5 * t])
        obstacles.append(upper_plate)

        left_plate = Box([d, t, h], with_sdf=True, face_colors=color)
        left_plate.newcoords(box_center.copy_worldcoords())
        left_plate.translate([0, 0.5 * w - 0.5 * t, 0.5 * h])
        obstacles.append(left_plate)

        right_plate = Box([d, t, h], with_sdf=True, face_colors=color)
        right_plate.newcoords(box_center.copy_worldcoords())
        right_plate.translate([0, -0.5 * w + 0.5 * t, 0.5 * h])
        obstacles.append(right_plate)

        opposite_plate = Box([t, w, h], with_sdf=True, face_colors=color)
        opposite_plate.newcoords(box_center.copy_worldcoords())
        opposite_plate.translate([0.5 * d - 0.5 * t, 0.0, 0.5 * h])
        obstacles.append(opposite_plate)

        return cls(table, obstacles, box_center, d, w, h, t)

    @staticmethod
    def create_standard_table() -> Box:
        # create jsk-lab 73b2 table
        table_depth = 0.5
        table_width = 0.75
        table_height = 0.7
        pos = [0.5 + table_depth * 0.5, 0.0, table_height * 0.5]
        table = Box(extents=[table_depth, table_width, table_height], pos=pos, with_sdf=True)
        return table

    def get_grid(self) -> Grid:
        grid_sizes = (56, 56, 28)
        mesh_height = 0.5
        depth, width, height = self.table._extents
        lb = np.array([-0.5 * depth, -0.5 * width, 0.5 * height - 0.1])
        ub = np.array([+0.5 * depth, +0.5 * width, 0.5 * height + mesh_height])
        lb = self.table.transform_vector(lb)
        ub = self.table.transform_vector(ub)
        return Grid(lb, ub, grid_sizes)


class SimpleCreateGridSdfMixin:
    @staticmethod
    def create_gridsdf(world: TabletopWorldBase) -> GridSDF:
        grid = world.get_grid()
        sdf = world.get_exact_sdf()

        X, Y, Z = grid.get_meshgrid(indexing="ij")
        pts = np.array(list(zip(X.flatten(), Y.flatten(), Z.flatten())))
        values = sdf.__call__(pts)
        gridsdf = GridSDF(grid, values, 2.0, create_itp_lazy=True)
        gridsdf = gridsdf.get_quantized()
        return gridsdf


def tabletop_box_sample_target_pose(
    world: TabletopBoxWorld, n_sample: int, standard: bool = False
) -> Coordinates:
    table = world.table
    table_depth, table_width, table_height = table._extents

    co = world.box_center.copy_worldcoords()
    if standard:
        d_trans = 0.0
        w_trans = 0.0
        h_trans = 0.5 * world.box_h
        theta = 0.0
    else:
        margin = 0.03
        box_dt = world.box_d - 2 * (world.box_t + margin)
        box_wt = world.box_w - 2 * (world.box_t + margin)
        box_ht = world.box_h - 2 * (world.box_t + margin)
        d_trans = -0.5 * box_dt + np.random.rand() * box_dt
        w_trans = -0.5 * box_wt + np.random.rand() * box_wt
        h_trans = world.box_t + margin + np.random.rand() * box_ht
        theta = -np.deg2rad(45) + np.random.rand() * np.deg2rad(90)

    co.translate([d_trans, w_trans, h_trans])
    co.rotate(theta, "z")

    points = np.expand_dims(co.worldpos(), axis=0)
    sdf = world.get_exact_sdf()

    sd_val = sdf(points)[0]
    assert sd_val > -0.0001
    return co


class TabletopBoxSingleArmSampleDescriptionsMixin:
    @staticmethod
    def sample_descriptions(
        world: TabletopBoxWorld, n_sample: int, standard: bool = False
    ) -> List[Tuple[Coordinates]]:
        # using single element Tuple looks bit cumbsersome but
        # for generality
        if standard:
            assert n_sample == 1
        pose_list: List[Tuple[Coordinates]] = []
        while len(pose_list) < n_sample:
            pose = tabletop_box_sample_target_pose(world, n_sample, standard)
            position = np.expand_dims(pose.worldpos(), axis=0)
            if world.get_exact_sdf()(position)[0] > 1e-3:
                pose_list.append((pose,))
        return pose_list


class TabletopBoxProblemBase(ProblemBase[TabletopBoxWorld, Tuple[Coordinates, ...]]):
    @staticmethod
    def get_world_type() -> Type[TabletopBoxWorld]:
        return TabletopBoxWorld

    def as_table(self) -> DescriptionTable:
        assert self._gridsdf is not None
        world_dict = {}
        world_dict["world"] = self._gridsdf.values.reshape(self._gridsdf.grid.sizes)
        world_dict["table_pose"] = skcoords_to_pose_vec(self.world.table.worldcoords())

        desc_dicts = []
        for desc in self.descriptions:
            desc_dict = {}
            for idx, co in enumerate(desc):
                pose = skcoords_to_pose_vec(co)
                name = "target_pose-{}".format(idx)
                desc_dict[name] = pose
            desc_dicts.append(desc_dict)
        return DescriptionTable(world_dict, desc_dicts)


class TabletopBoxSingleArmReaching(
    TabletopWorldBase, TabletopBoxSingleArmSampleDescriptionsMixin, SimpleCreateGridSdfMixin
):
    ...
