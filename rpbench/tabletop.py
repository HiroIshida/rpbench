import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, ClassVar, List, Tuple, Type

import numpy as np
from skmp.constraint import BoxConst, CollFreeConst, IneqCompositeConst, PoseConstraint
from skmp.kinematics import (
    ArticulatedCollisionKinematicsMap,
    ArticulatedEndEffectorKinematicsMap,
)
from skmp.robot.pr2 import PR2Config
from skmp.robot.utils import set_robot_state
from skmp.solver.interface import Problem
from skmp.trajectory import Trajectory
from skrobot.coordinates import Coordinates
from skrobot.model.link import Link
from skrobot.model.primitives import Axis, Box
from skrobot.models.pr2 import PR2
from skrobot.sdf import UnionSDF
from skrobot.viewers import TrimeshSceneViewer
from voxbloxpy.core import Grid, GridSDF

from rpbench.interface import DescriptionTable, TaskBase, WorldBase
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

    def visualize(self, viewer: TrimeshSceneViewer) -> None:
        viewer.add(self.table)
        for obs in self.obstacles:
            viewer.add(obs)


def create_exact_gridsdf(world: TabletopWorldBase) -> GridSDF:
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


class CachedPR2ConstProvider(ABC):
    """
    loading robot model is a process that takes some times.
    So, by utilizing classmethod with lru_cache, all program
    that calls this class share the same robot model and
    other stuff.
    """

    @classmethod
    @abstractmethod
    def get_config(cls) -> PR2Config:
        ...

    @classmethod
    def get_box_const(cls) -> BoxConst:
        config = cls.get_config()
        return config.get_box_const()

    @classmethod
    def get_pose_const(cls, target_pose_list: List[Coordinates]) -> PoseConstraint:
        config = cls.get_config()
        const = PoseConstraint.from_skrobot_coords(
            target_pose_list, config.get_endeffector_kin(), cls.get_pr2()
        )
        return const

    @classmethod
    def get_start_config(cls) -> np.ndarray:
        config = cls.get_config()
        pr2 = cls.get_pr2()
        angles = []
        for jn in config._get_control_joint_names():
            a = pr2.__dict__[jn].joint_angle()
            angles.append(a)
        return np.array(angles)

    @classmethod
    def get_collfree_const(cls, sdf: Callable[[np.ndarray], np.ndarray]) -> IneqCompositeConst:
        config = cls.get_config()
        colfree = CollFreeConst(cls.get_colkin(), sdf, cls.get_pr2())
        selcolfree = config.get_neural_selcol_const(cls.get_pr2())
        return IneqCompositeConst([colfree, selcolfree])

    @classmethod
    @lru_cache
    def get_pr2(cls) -> PR2:
        pr2 = PR2(use_tight_joint_limit=False)
        pr2.reset_manip_pose()
        return pr2

    @classmethod
    @lru_cache
    def get_efkin(cls) -> ArticulatedEndEffectorKinematicsMap:
        config = cls.get_config()
        return config.get_endeffector_kin()

    @classmethod
    @lru_cache
    def get_colkin(cls) -> ArticulatedCollisionKinematicsMap:
        config = cls.get_config()
        return config.get_collision_kin()


class CachedRArmPR2ConstProvider(CachedPR2ConstProvider):
    @classmethod
    def get_config(cls) -> PR2Config:
        return PR2Config(with_base=False)


@dataclass
class TabletopBoxTaskBase(TaskBase[TabletopBoxWorld, Tuple[Coordinates, ...]]):
    config_provider: ClassVar[Type[CachedPR2ConstProvider]]

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


class TabletopBoxRightArmReachingTask(TabletopBoxTaskBase):
    config_provider: ClassVar[Type[CachedPR2ConstProvider]] = CachedRArmPR2ConstProvider

    @staticmethod
    def sample_descriptions(
        world: TabletopBoxWorld, n_sample: int, standard: bool = False
    ) -> List[Tuple[Coordinates, ...]]:
        # using single element Tuple looks bit cumbsersome but
        # for generality
        if standard:
            assert n_sample == 1
        pose_list: List[Tuple[Coordinates, ...]] = []
        while len(pose_list) < n_sample:
            pose = tabletop_box_sample_target_pose(world, n_sample, standard)
            position = np.expand_dims(pose.worldpos(), axis=0)
            if world.get_exact_sdf()(position)[0] > 1e-3:
                pose_list.append((pose,))
        return pose_list

    def export_problems(self) -> List[Problem]:
        provider = self.config_provider
        q_start = provider.get_start_config()
        box_const = provider.get_box_const()

        assert self._gridsdf is not None
        ineq_const = provider.get_collfree_const(self._gridsdf)

        problems = []
        for desc in self.descriptions:
            pose_const = provider.get_pose_const(list(desc))
            problem = Problem(q_start, box_const, pose_const, ineq_const, None)
            problems.append(problem)
        return problems

    @staticmethod
    def create_gridsdf(world: TabletopBoxWorld) -> GridSDF:
        return create_exact_gridsdf(world)


class TaskVisualizer:
    # TODO: this class actually take any Task if it has config provider
    task: TabletopBoxTaskBase
    viewer: TrimeshSceneViewer
    _show_called: bool

    def __init__(self, task: TabletopBoxTaskBase):
        viewer = TrimeshSceneViewer()

        robot_config = task.config_provider()
        robot_model = robot_config.get_pr2()
        viewer.add(robot_model)

        task.world.visualize(viewer)
        for desc in task.descriptions:
            for co in desc:
                axis = Axis.from_coords(co)
                viewer.add(axis)

        self.task = task
        self.viewer = viewer
        self._show_called = False

    def show(self) -> None:
        self.viewer.show()
        time.sleep(1.0)
        self._show_called = True

    def visualize_trajectory(self, trajectory: Trajectory, t_interval: float = 0.6) -> None:
        assert self._show_called
        robot_config_provider = self.task.config_provider()
        robot_model = robot_config_provider.get_pr2()

        config = robot_config_provider.get_config()

        for q in trajectory:
            set_robot_state(robot_model, config._get_control_joint_names(), q, config.with_base)
            self.viewer.redraw()
            time.sleep(t_interval)

        print("==> Press [q] to close window")
        while not self.viewer.has_exit:
            time.sleep(0.1)
            self.viewer.redraw()
