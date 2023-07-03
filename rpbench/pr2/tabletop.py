from abc import abstractmethod
from dataclasses import dataclass
from typing import ClassVar, List, Optional, Tuple, Type, TypeVar, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm
from skmp.solver.interface import Problem, ResultProtocol
from skmp.solver.nlp_solver import SQPBasedSolver, SQPBasedSolverConfig
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig, TerminateState
from skrobot.coordinates import Coordinates
from skrobot.model import RobotModel
from skrobot.model.link import Link
from skrobot.model.primitives import Box
from skrobot.sdf import UnionSDF
from skrobot.viewers import TrimeshSceneViewer
from voxbloxpy.core import Grid, GridSDF, IntegratorType

from rpbench.interface import (
    DescriptionT,
    DescriptionTable,
    ReachingTaskBase,
    SamplableBase,
    SamplableT,
    WorldBase,
)
from rpbench.planer_box_utils import Box2d, PlanerCoords, sample_box
from rpbench.pr2.common import (
    CachedDualArmPR2ConstProvider,
    CachedPR2ConstProvider,
    CachedRArmPR2ConstProvider,
)
from rpbench.two_dimensional.utils import Grid2d
from rpbench.utils import SceneWrapper, create_union_sdf, skcoords_to_pose_vec
from rpbench.vision import Camera, EsdfMap, RayMarchingConfig, create_synthetic_esdf

TabletopWorldT = TypeVar("TabletopWorldT", bound="TabletopWorldBase")
TabletopSamplableT = TypeVar("TabletopSamplableT", bound="TabletopSamplableBase")
OtherTabletopBoxSamplableT = TypeVar("OtherTabletopBoxSamplableT", bound="TabletopSamplableBase")


@dataclass
class TabletopWorldBase(WorldBase):
    table: Box
    obstacles: List[Link]

    def get_exact_sdf(self) -> UnionSDF:
        lst = [self.table.sdf]
        for obstacle in self.obstacles:
            lst.append(obstacle.sdf)
        return UnionSDF(lst)

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

    def get_grid2d(self) -> Grid2d:
        grid3d = self.get_grid()
        size = (112, 112)
        return Grid2d(grid3d.lb[:2], grid3d.ub[:2], size)

    def create_exact_heightmap(self, invalid_subs: float = -1.0) -> np.ndarray:
        grid2d = self.get_grid2d()
        depth, width, height = self.table._extents

        height_from_table = 1.0

        step = self.table._extents[:2] / np.array(grid2d.sizes)
        xlin = (
            np.linspace(step[0] * 0.5, step[0] * (grid2d.sizes[0] - 0.5), grid2d.sizes[0])
            - depth * 0.5
        )
        ylin = (
            np.linspace(step[1] * 0.5, step[1] * (grid2d.sizes[1] - 0.5), grid2d.sizes[1])
            - width * 0.5
        )
        X, Y = np.meshgrid(xlin, ylin)
        Z = np.zeros_like(X) + height * 0.5 + height_from_table
        points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        points = self.table.transform_vector(points)
        dirs = np.tile(np.array([0, 0, -1]), (len(points), 1))

        conf = RayMarchingConfig()
        dists = Camera.ray_marching(points, dirs, self.get_exact_sdf(), conf)
        is_valid = dists < height_from_table + 1e-2
        dists_flip = height_from_table - dists
        dists_flip[~is_valid] = invalid_subs
        mesh = np.reshape(dists_flip, (grid2d.sizes[0], grid2d.sizes[1]))
        # debug point cloud
        # return points[is_valid] + dists[is_valid, None] * dirs[is_valid, :]

        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        # fig, ax = plt.subplots()
        # ax = fig.add_subplot(111, projection='3d')
        # X, Y = np.meshgrid(np.arange(112), np.arange(112))
        # ax.plot_surface(X, Y, mesh, cmap='terrain')
        # plt.show()
        return mesh


@dataclass
class TabletopBoxWorld(TabletopWorldBase):
    box: Box

    def visualize(self, viewer: Union[TrimeshSceneViewer, SceneWrapper]) -> None:
        viewer.add(self.table)
        for obs in self.obstacles:
            viewer.add(obs)
        viewer.add(self.box)

    @classmethod
    def sample(cls, standard: bool = False) -> "TabletopBoxWorld":
        intrinsic_desc = []

        table = cls.create_standard_table()
        table_depth, table_width, table_height = table._extents

        if not standard:
            x_rand, y_rand = np.random.randn(2)
            x = 0.1 + 0.05 * x_rand
            y = 0.1 * y_rand
            z = 0.0
            table.translate([x, y, z])
            intrinsic_desc.extend([x_rand, y_rand])
        else:
            intrinsic_desc.extend([0.0, 0.0])

        table_tip = table.copy_worldcoords()
        table_tip.translate([-table_depth * 0.5, -table_width * 0.5, +0.5 * table_height])

        table_extent = np.array([table_depth, table_width])
        while True:
            box2d: Optional[Box2d]
            if standard:
                box_extent = np.ones(3) * 0.15
                box2d = Box2d(box_extent[:2], PlanerCoords(np.zeros(2), 0.0 * np.pi))
            else:
                box_extent = np.ones(3) * 0.1 + np.random.rand(3) * np.array([0.1, 0.1, 0.05])
                box2d = sample_box(table_extent, box_extent[:2], [])
            if box2d is not None:
                break

        box = Box(box_extent, with_sdf=True)
        box.newcoords(table.copy_worldcoords())
        box.translate(np.hstack([box2d.coords.pos, 0.5 * (box_extent[-1] + table._extents[-1])]))
        box.rotate(box2d.coords.angle, "z")
        box.visual_mesh.visual.face_colors = [255, 0, 0, 200]

        obstacles = [box]  # reaching box is also an obstacle in planning context
        obstacles_2ds = []

        if standard:
            n_obs = 0
        else:
            n_obs = 1 + np.random.randint(8)

        for _ in range(n_obs):
            obs_extent = lognorm(s=0.5, scale=1.0).rvs(size=3) * np.array([0.04, 0.04, 0.2])
            obs2d = sample_box(table_extent, obs_extent[:2], [box2d])
            if obs2d is None:
                break
            obstacles_2ds.append(obs2d)

            obs = Box(obs_extent, with_sdf=True)
            obs.newcoords(table.copy_worldcoords())
            obs.translate(
                np.hstack([obs2d.coords.pos, 0.5 * (obs_extent[-1] + table._extents[-1])])
            )
            obs.rotate(obs2d.coords.angle, "z")
            obs.visual_mesh.visual.face_colors = [0, 255, 0, 200]
            obstacles.append(obs)

        # check if all obstacle dont collide each other

        debug_matplotlib = False
        if debug_matplotlib:
            fig, ax = plt.subplots()
            box2d.visualize((fig, ax), "red")
            for obs in obstacles_2ds:
                obs.visualize((fig, ax), "green")
            ax.set_aspect("equal", adjustable="box")

        return cls(table, obstacles, box)


@dataclass
class TabletopOvenWorld(TabletopWorldBase):
    box_center: Coordinates
    box_d: float
    box_w: float
    box_h: float
    box_t: float
    _intrinsic_desc: np.ndarray

    def visualize(self, viewer: Union[TrimeshSceneViewer, SceneWrapper]) -> None:
        viewer.add(self.table)
        for obs in self.obstacles:
            viewer.add(obs)

    def export_intrinsic_description(self) -> np.ndarray:
        return self._intrinsic_desc

    @classmethod
    def sample(cls, standard: bool = False) -> "TabletopOvenWorld":
        intrinsic_desc = []

        table = cls.create_standard_table()
        table_depth, table_width, table_height = table._extents

        if not standard:
            x_rand, y_rand = np.random.randn(2)
            x = 0.1 + 0.05 * x_rand
            y = 0.1 * y_rand
            z = 0.0
            table.translate([x, y, z])

            intrinsic_desc.extend([x_rand, y_rand])
        else:
            intrinsic_desc.extend([0.0, 0.0])

        table_tip = table.copy_worldcoords()
        table_tip.translate([-table_depth * 0.5, -table_width * 0.5, +0.5 * table_height])

        obstacles = []

        color = np.array([255, 220, 0, 150])

        # box
        d = 0.2
        w = 0.3
        h = 0.2
        if standard:
            d += 0.15
            w += 0.15
            h += 0.15
            intrinsic_desc.extend([0.0, 0.0, 0.0])
        else:
            d_rand, w_rand, h_rand = np.random.randn(3)
            d += 0.15 + d_rand * 0.05
            w += 0.15 + w_rand * 0.05
            h += 0.15 + h_rand * 0.05
            intrinsic_desc.extend([d_rand, w_rand, h_rand])

        t = 0.03

        if standard:
            box_center = table.copy_worldcoords()
            box_center.translate([0, 0, 0.5 * table_height])
            intrinsic_desc.extend([0.0, 0.0])
        else:
            box_center = table.copy_worldcoords()
            box_center.translate([0, 0, 0.5 * table_height])

            margin_x = table_depth - d
            margin_y = table_width - w

            x_rand, y_rand = np.random.randn(2)

            trans = np.array([0.5 * 0.3 * margin_x * x_rand, 0.5 * 0.3 * margin_y * y_rand, 0.0])
            box_center.translate(trans)

            intrinsic_desc.extend([x_rand, y_rand])

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

        return cls(table, obstacles, box_center, d, w, h, t, np.array(intrinsic_desc))


class ExactGridSDFCreator:
    @staticmethod
    def create_gridsdf(world: TabletopWorldBase, robot_model: RobotModel) -> GridSDF:
        grid = world.get_grid()
        sdf = world.get_exact_sdf()

        X, Y, Z = grid.get_meshgrid(indexing="ij")
        pts = np.array(list(zip(X.flatten(), Y.flatten(), Z.flatten())))
        values = sdf.__call__(pts)
        gridsdf = GridSDF(grid, values, 2.0, create_itp_lazy=True)
        gridsdf = gridsdf.get_quantized()
        return gridsdf


class VoxbloxGridSDFCreator:
    @staticmethod
    def get_pr2_kinect_camera(robot_model: RobotModel) -> Camera:
        camera = Camera()
        # actually this is pr2 specific
        robot_model.head_plate_frame.assoc(camera, relative_coords="local")
        camera.translate(np.array([-0.2, 0.0, 0.17]))
        return camera

    @staticmethod
    def create_gridsdf(world: TabletopWorldBase, robot_model: RobotModel) -> GridSDF:
        grid = world.get_grid()
        sdf = world.get_exact_sdf()

        camera = VoxbloxGridSDFCreator.get_pr2_kinect_camera(robot_model)
        rm_config = RayMarchingConfig(max_dist=2.0)

        esdf = EsdfMap.create(0.02, integrator_type=IntegratorType.MERGED)
        esdf = create_synthetic_esdf(sdf, camera, rm_config=rm_config, esdf=esdf)
        grid_sdf = esdf.get_grid_sdf(grid, fill_value=1.0, create_itp_lazy=True)
        return grid_sdf


class TabletopOvenWorldMixin:
    @staticmethod
    def get_world_type() -> Type[TabletopOvenWorld]:
        return TabletopOvenWorld


class TabletopBoxWorldMixin:
    @staticmethod
    def get_world_type() -> Type[TabletopBoxWorld]:
        return TabletopBoxWorld


@dataclass
class TabletopSamplableBase(SamplableBase[TabletopWorldT, DescriptionT, RobotModel]):
    @staticmethod
    def get_robot_model() -> RobotModel:
        return CachedPR2ConstProvider.get_pr2()

    def export_table(self) -> DescriptionTable:
        assert self._gridsdf is not None
        world_dict = {}
        # world_dict["world"] = self._gridsdf.values.reshape(self._gridsdf.grid.sizes)
        world_dict["world"] = self.world.create_exact_heightmap()
        world_dict["table_pose"] = skcoords_to_pose_vec(self.world.table.worldcoords())

        desc_dicts = []
        for desc in self.descriptions:
            if desc is None:
                # TODO: idealy speaking, such conditioning should be avoided
                # and instead, part of "export_table" should be implemented
                # class-wise
                continue

            desc_dict = {}
            for idx, co in enumerate(desc):
                pose = skcoords_to_pose_vec(co)
                name = "target_pose-{}".format(idx)
                desc_dict[name] = pose
            desc_dicts.append(desc_dict)
        return DescriptionTable(world_dict, desc_dicts)

    def export_intrinsic_descriptions(self) -> List[np.ndarray]:
        return [self.world.export_intrinsic_description()] * self.n_inner_task

    @classmethod
    def cast_from(cls: Type[TabletopSamplableT], other: SamplableT) -> TabletopSamplableT:
        assert isinstance(other, TabletopSamplableBase)
        is_compatible_meshgen = cls.create_gridsdf == other.create_gridsdf
        assert is_compatible_meshgen
        return cls(other.world, [], other._gridsdf)


class TabletopWorldWrapBase(TabletopSamplableBase[TabletopWorldT, None]):
    @classmethod
    def sample_descriptions(
        cls, world: TabletopWorldBase, n_sample: int, standard: bool = False
    ) -> List[None]:
        return [None for _ in range(n_sample)]


class TabletopOvenWorldWrapBase(TabletopOvenWorldMixin, TabletopWorldWrapBase[TabletopOvenWorld]):
    ...


class TabletopBoxWorldWrapBase(TabletopBoxWorldMixin, TabletopWorldWrapBase[TabletopBoxWorld]):
    @classmethod
    def acceptable_time_admissible(cls) -> float:
        return 100.0


class TabletopTaskBase(
    ReachingTaskBase[TabletopWorldT, RobotModel],
    TabletopSamplableBase[TabletopWorldT, Tuple[Coordinates, ...]],
):
    config_provider: ClassVar[Type[CachedPR2ConstProvider]]

    @classmethod
    def get_dof(cls) -> int:
        return cls.config_provider.get_dof()

    def export_problems(self) -> List[Problem]:
        provider = self.config_provider
        q_start = provider.get_start_config()
        box_const = provider.get_box_const()

        assert self._gridsdf is not None
        sdf = create_union_sdf([self._gridsdf, self.world.table.sdf])
        ineq_const = provider.get_collfree_const(sdf)

        problems = []
        for desc in self.descriptions:
            pose_const = provider.get_pose_const(list(desc))
            problem = Problem(q_start, box_const, pose_const, ineq_const, None)
            problems.append(problem)
        return problems

    def solve_default_each(self, problem: Problem) -> ResultProtocol:
        """use rrt-connect and then smooth path by sqp
        if rrt planning succeeded but, nlp failed return rrt path.
        """
        n_satisfaction_budget = 1
        n_planning_budget = 4
        solcon = OMPLSolverConfig(n_max_call=20000, n_max_satisfaction_trial=100, simplify=True)
        ompl_solver = OMPLSolver.init(solcon)

        satisfaction_fail_count = 0
        planning_fail_count = 0
        while (satisfaction_fail_count < n_satisfaction_budget) and (
            planning_fail_count < n_planning_budget
        ):
            ompl_solver.setup(problem)
            ompl_ret = ompl_solver.solve()
            if ompl_ret.traj is not None:
                # now, smooth out the solution

                # first solve with smaller number of waypoint
                nlp_conf = SQPBasedSolverConfig(
                    n_wp=20, n_max_call=20, motion_step_satisfaction="debug_ignore"
                )
                nlp_solver = SQPBasedSolver.init(nlp_conf)
                nlp_solver.setup(problem)
                nlp_ret = nlp_solver.solve(ompl_ret.traj)

                # Then try to find more find-grained solution
                if nlp_ret.traj is not None:
                    nlp_conf = SQPBasedSolverConfig(
                        n_wp=60, n_max_call=20, motion_step_satisfaction="post"
                    )
                    nlp_solver = SQPBasedSolver.init(nlp_conf)
                    nlp_solver.setup(problem)
                    nlp_ret = nlp_solver.solve(nlp_ret.traj)
                    if nlp_ret.traj is not None:
                        return nlp_ret
                return ompl_ret

            if ompl_ret.terminate_state == TerminateState.FAIL_SATISFACTION:
                satisfaction_fail_count += 1
            else:
                planning_fail_count += 1

        assert ompl_ret.traj is None  # because solve is supposed to be failed
        return ompl_ret

    @classmethod
    def sample_descriptions(
        cls, world: TabletopWorldT, n_sample: int, standard: bool = False
    ) -> Optional[List[Tuple[Coordinates, ...]]]:
        # using single element Tuple looks bit cumbsersome but
        # for generality

        if standard:
            assert n_sample == 1

        poses_list: List[Tuple[Coordinates, ...]] = []
        n_budget = 100 * n_sample
        for _ in range(n_budget):
            poses = cls.sample_target_poses(world, standard)
            is_valid_poses = True
            for pose in poses:
                position = np.expand_dims(pose.worldpos(), axis=0)
                if world.get_exact_sdf()(position)[0] < 1e-3:
                    is_valid_poses = False
            if is_valid_poses:
                poses_list.append(poses)
            if len(poses_list) == n_sample:
                return poses_list
        return None

    @classmethod
    @abstractmethod
    def sample_target_poses(cls, world: TabletopWorldT, standard: bool) -> Tuple[Coordinates, ...]:
        ...


class TabletopOvenRightArmReachingTaskBase(
    TabletopOvenWorldMixin, TabletopTaskBase[TabletopOvenWorld]
):
    config_provider: ClassVar[Type[CachedPR2ConstProvider]] = CachedRArmPR2ConstProvider

    @classmethod
    def sample_target_poses(
        cls, world: TabletopOvenWorld, standard: bool
    ) -> Tuple[Coordinates, ...]:
        table = world.table
        table_depth, table_width, table_height = table._extents

        n_max_trial = 100
        for _ in range(n_max_trial):
            co = world.box_center.copy_worldcoords()
            if standard:
                d_trans = -0.1
                w_trans = 0.0
                h_trans = 0.5 * world.box_h
                theta = 0.0
            else:
                margin = 0.03
                box_dt = world.box_d - 2 * (world.box_t + margin)
                box_wt = world.box_w - 2 * (world.box_t + margin)
                box_ht = world.box_h - 2 * (world.box_t + margin)
                d_trans = np.random.randn() * box_dt * 0.2
                w_trans = np.random.randn() * box_wt * 0.2
                h_trans = 0.5 * world.box_h + np.random.randn() * 0.2 * box_ht
                theta = np.random.randn() * np.deg2rad(90) * 0.2

            co.translate([d_trans, w_trans, h_trans])
            co.rotate(theta, "z")

            points = np.expand_dims(co.worldpos(), axis=0)
            sdf = world.get_exact_sdf()

            sd_val = sdf(points)[0]
            if sd_val > -0.0001:
                return (co,)
        assert False


class TabletopOvenDualArmReachingTaskBase(
    TabletopOvenWorldMixin, TabletopTaskBase[TabletopOvenWorld]
):
    config_provider: ClassVar[Type[CachedPR2ConstProvider]] = CachedDualArmPR2ConstProvider

    @classmethod
    def sample_target_poses(
        cls, world: TabletopOvenWorld, standard: bool
    ) -> Tuple[Coordinates, ...]:
        table = world.table
        table_depth, table_width, table_height = table._extents
        co = world.box_center.copy_worldcoords()

        if standard:
            d_trans = -0.1
            w_trans = 0.0
            h_trans = 0.5 * world.box_h
            hands_width = 0.15
        else:
            margin = 0.03
            box_dt = world.box_d - 2 * (world.box_t + margin)
            box_wt = world.box_w - 2 * (world.box_t + margin)
            box_ht = world.box_h - 2 * (world.box_t + margin)
            d_trans = -0.5 * box_dt + np.random.rand() * box_dt
            w_trans = -0.5 * box_wt + np.random.rand() * box_wt
            h_trans = world.box_t + margin + np.random.rand() * box_ht

            hands_width_min = 0.08
            hands_width_max = ((box_wt * 0.5) - abs(w_trans)) * 2.0
            hands_width = hands_width_min + np.random.rand() * (hands_width_max - hands_width_min)

        co.translate([d_trans, 0, h_trans])

        left_co = co.copy_worldcoords()
        right_co = co.copy_worldcoords()

        right_co.translate([0.0, -hands_width * 0.5, 0.0])
        left_co.translate([0.0, hands_width * 0.5, 0.0])
        right_co.rotate(np.deg2rad(90), "x")
        left_co.rotate(np.deg2rad(90), "x")
        return (right_co, left_co)


class TabletopBoxDualArmReachingTaskBase(TabletopBoxWorldMixin, TabletopTaskBase[TabletopBoxWorld]):
    config_provider: ClassVar[Type[CachedPR2ConstProvider]] = CachedDualArmPR2ConstProvider

    @classmethod
    def sample_target_poses(
        cls, world: TabletopBoxWorld, standard: bool
    ) -> Tuple[Coordinates, ...]:

        table_rot = world.table.copy_worldcoords().rotation
        table_ex, table_ey = table_rot[:, 0], table_rot[:, 1]

        box_rot = world.box.copy_worldcoords().rotation
        box_ex, box_ey = box_rot[:, 0], box_rot[:, 1]

        margin = 0.05
        if table_ex.dot(box_ex) < 1 / np.sqrt(2):
            x_half_width = 0.5 * world.box._extents[0]
            co_box = world.box.copy_worldcoords()
            pos_target1 = co_box.worldpos() + box_ex * (x_half_width + margin)
            pos_target2 = co_box.worldpos() - box_ex * (x_half_width + margin)
            right_co = Coordinates(pos=pos_target1, rot=world.box.copy_worldcoords().rotation)
            left_co = Coordinates(pos=pos_target2, rot=world.box.copy_worldcoords().rotation)
            if table_ey.dot(box_ex) < 0.0:
                right_co.rotate(+np.pi * 0.5, "z")
                left_co.rotate(+np.pi * 0.5, "z")
            else:
                right_co.rotate(-np.pi * 0.5, "z")
                left_co.rotate(-np.pi * 0.5, "z")
        else:
            y_half_width = 0.5 * world.box._extents[1]
            co_box = world.box.copy_worldcoords()
            pos_target1 = co_box.worldpos() + box_ey * (y_half_width + margin)
            pos_target2 = co_box.worldpos() - box_ey * (y_half_width + margin)
            right_co = Coordinates(pos=pos_target1, rot=world.box.copy_worldcoords().rotation)
            left_co = Coordinates(pos=pos_target2, rot=world.box.copy_worldcoords().rotation)

        # use the same height for all boxes
        reach_height = world.table._extents[2] + 0.1
        h_translate = reach_height - world.box.worldpos()[2]
        d_translate = 0.05

        right_co.translate([d_translate, 0, h_translate])
        left_co.translate([d_translate, 0, h_translate])

        right_co.rotate(np.pi * 0.5, "x")
        left_co.rotate(np.pi * 0.5, "x")

        if right_co.worldpos()[1] < left_co.worldpos()[1]:
            return (right_co, left_co)
        else:
            return (left_co, right_co)

    @classmethod
    def acceptable_time_admissible(cls) -> float:
        return 100.0


# fmt: off
class TabletopOvenRightArmReachingTask(ExactGridSDFCreator, TabletopOvenRightArmReachingTaskBase): ...  # noqa
class TabletopOvenVoxbloxRightArmReachingTask(VoxbloxGridSDFCreator, TabletopOvenRightArmReachingTaskBase): ...  # noqa
class TabletopOvenDualArmReachingTask(ExactGridSDFCreator, TabletopOvenDualArmReachingTaskBase): ...  # noqa
class TabletopOvenWorldWrap(ExactGridSDFCreator, TabletopOvenWorldWrapBase): ...  # noqa
class TabletopOvenVoxbloxWorldWrap(VoxbloxGridSDFCreator, TabletopOvenWorldWrapBase): ...  # noqa
class TabletopVoxbloxOvenWorldWrap(VoxbloxGridSDFCreator, TabletopOvenWorldWrapBase): ...  # noqa
class TabletopOvenVoxbloxDualArmReachingTask(VoxbloxGridSDFCreator, TabletopOvenDualArmReachingTaskBase): ...  # noqa

class TabletopBoxWorldWrap(ExactGridSDFCreator, TabletopBoxWorldWrapBase): ...  # noqa
class TabletopBoxDualArmReachingTask(ExactGridSDFCreator, TabletopBoxDualArmReachingTaskBase): ...  # noqa
# fmt: on
