import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, ClassVar, List, Tuple, Type, TypeVar

import numpy as np
from skmp.constraint import (
    AbstractIneqConst,
    BoxConst,
    CollFreeConst,
    IneqCompositeConst,
    PoseConstraint,
)
from skmp.kinematics import (
    ArticulatedCollisionKinematicsMap,
    ArticulatedEndEffectorKinematicsMap,
)
from skmp.robot.pr2 import PR2Config
from skmp.robot.utils import get_robot_state, set_robot_state
from skmp.solver.interface import Problem, ResultProtocol
from skmp.solver.nlp_solver import SQPBasedSolver, SQPBasedSolverConfig
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig, TerminateState
from skmp.trajectory import Trajectory
from skrobot.coordinates import Coordinates
from skrobot.model import RobotModel
from skrobot.model.link import Link
from skrobot.model.primitives import Axis, Box
from skrobot.models.pr2 import PR2
from skrobot.sdf import UnionSDF
from skrobot.viewers import TrimeshSceneViewer
from voxbloxpy.core import Grid, GridSDF, IntegratorType

from rpbench.interface import (
    DescriptionT,
    DescriptionTable,
    SamplableBase,
    SamplableT,
    TaskBase,
    WorldBase,
)
from rpbench.utils import skcoords_to_pose_vec
from rpbench.vision import Camera, EsdfMap, RayMarchingConfig, create_synthetic_esdf

TabletopBoxSamplableT = TypeVar("TabletopBoxSamplableT", bound="TabletopBoxSamplableBase")
OtherTabletopBoxSamplableT = TypeVar("OtherTabletopBoxSamplableT", bound="TabletopBoxSamplableBase")


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

        if not standard:
            x = np.random.rand() * 0.2
            y = -0.2 + np.random.rand() * 0.4
            z = 0.0
            table.translate([x, y, z])

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
        else:
            d += np.random.rand() * 0.3
            w += np.random.rand() * 0.3
            h += np.random.rand() * 0.3
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


class ExactGridSDFCreator:
    @staticmethod
    def create_gridsdf(world: TabletopBoxWorld, robot_model: RobotModel) -> GridSDF:
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
    def create_gridsdf(world: TabletopBoxWorld, robot_model: RobotModel) -> GridSDF:
        grid = world.get_grid()
        sdf = world.get_exact_sdf()

        camera = VoxbloxGridSDFCreator.get_pr2_kinect_camera(robot_model)
        rm_config = RayMarchingConfig(max_dist=2.0)

        esdf = EsdfMap.create(0.02, integrator_type=IntegratorType.MERGED)
        esdf = create_synthetic_esdf(sdf, camera, rm_config=rm_config, esdf=esdf)
        grid_sdf = esdf.get_grid_sdf(grid, fill_value=1.0, create_itp_lazy=True)
        return grid_sdf


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
    @lru_cache
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
        angles = get_robot_state(pr2, config._get_control_joint_names(), config.with_base)
        return angles

    @classmethod
    @abstractmethod
    def get_collfree_const(cls, sdf: Callable[[np.ndarray], np.ndarray]) -> AbstractIneqConst:
        """get collision free constraint"""
        # make this method abstract because usually self-collision must be considerd
        # when dual-arm planning, but not have to be in single arm planning.
        ...

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

    @classmethod
    @lru_cache
    def get_dof(cls) -> int:
        config = cls.get_config()
        names = config._get_control_joint_names()
        dof = len(names)
        if config.with_base:
            dof += 3
        return dof


class CachedRArmPR2ConstProvider(CachedPR2ConstProvider):
    @classmethod
    def get_config(cls) -> PR2Config:
        return PR2Config(with_base=True)

    @classmethod
    def get_collfree_const(cls, sdf: Callable[[np.ndarray], np.ndarray]) -> CollFreeConst:
        colfree = CollFreeConst(cls.get_colkin(), sdf, cls.get_pr2())
        return colfree


class CachedDualArmPR2ConstProvider(CachedPR2ConstProvider):
    @classmethod
    def get_config(cls) -> PR2Config:
        return PR2Config(with_base=True, control_arm="dual")

    @classmethod
    def get_collfree_const(cls, sdf: Callable[[np.ndarray], np.ndarray]) -> IneqCompositeConst:
        colfree = CollFreeConst(cls.get_colkin(), sdf, cls.get_pr2())
        selcolfree = cls.get_config().get_neural_selcol_const(cls.get_pr2())
        return IneqCompositeConst([colfree, selcolfree])


@dataclass
class TabletopBoxSamplableBase(SamplableBase[TabletopBoxWorld, DescriptionT]):
    @staticmethod
    def get_world_type() -> Type[TabletopBoxWorld]:
        return TabletopBoxWorld

    @staticmethod
    def get_robot_model() -> RobotModel:
        return CachedPR2ConstProvider.get_pr2()

    def export_table(self) -> DescriptionTable:
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

    @classmethod
    def cast_from(cls: Type[TabletopBoxSamplableT], other: SamplableT) -> TabletopBoxSamplableT:
        assert isinstance(other, TabletopBoxSamplableBase)
        is_compatible_meshgen = cls.create_gridsdf == other.create_gridsdf
        assert is_compatible_meshgen
        return cls(other.world, [], other._gridsdf)


class TabletopBoxWorldWrapBase(TabletopBoxSamplableBase[None]):
    @classmethod
    def sample_descriptions(
        cls, world: TabletopWorldBase, n_sample: int, standard: bool = False
    ) -> List[None]:
        return [None for _ in range(n_sample)]


class TabletopBoxTaskBase(
    TaskBase[TabletopBoxWorld, Tuple[Coordinates, ...]],
    TabletopBoxSamplableBase[Tuple[Coordinates, ...]],
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
        ineq_const = provider.get_collfree_const(self._gridsdf)

        problems = []
        for desc in self.descriptions:
            pose_const = provider.get_pose_const(list(desc))
            problem = Problem(q_start, box_const, pose_const, ineq_const, None)
            problems.append(problem)
        return problems

    def solve_default_each(self, problem: Problem) -> ResultProtocol:
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
            ret = ompl_solver.solve()
            if ret.traj is not None:
                # now, smooth out the solution

                # first solve with smaller number of waypoint
                nlp_conf = SQPBasedSolverConfig(
                    n_wp=20, n_max_call=20, motion_step_satisfaction="debug_ignore"
                )
                nlp_solver = SQPBasedSolver.init(nlp_conf)
                nlp_solver.setup(problem)
                nlp_ret = nlp_solver.solve(ret.traj)

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

            if ret.terminate_state == TerminateState.FAIL_SATISFACTION:
                satisfaction_fail_count += 1
            else:
                planning_fail_count += 1

        # force set traj = None to indicate its failed
        ret.traj = None
        return ret  # return latest one (failed)

    @classmethod
    def sample_descriptions(
        cls, world: TabletopBoxWorld, n_sample: int, standard: bool = False
    ) -> List[Tuple[Coordinates, ...]]:
        # using single element Tuple looks bit cumbsersome but
        # for generality
        if standard:
            assert n_sample == 1
        pose_list: List[Tuple[Coordinates, ...]] = []
        while len(pose_list) < n_sample:
            poses = cls.sample_target_poses(world, standard)
            is_valid_poses = True
            for pose in poses:
                position = np.expand_dims(pose.worldpos(), axis=0)
                if world.get_exact_sdf()(position)[0] < 1e-3:
                    is_valid_poses = False
            if is_valid_poses:
                pose_list.append(poses)
        return pose_list

    @classmethod
    @abstractmethod
    def sample_target_poses(
        cls, world: TabletopBoxWorld, standard: bool
    ) -> Tuple[Coordinates, ...]:
        ...


class TabletopBoxRightArmReachingTaskBase(TabletopBoxTaskBase):
    config_provider: ClassVar[Type[CachedPR2ConstProvider]] = CachedRArmPR2ConstProvider

    @classmethod
    def sample_target_poses(
        cls, world: TabletopBoxWorld, standard: bool
    ) -> Tuple[Coordinates, ...]:
        table = world.table
        table_depth, table_width, table_height = table._extents

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
        return (co,)


class TabletopBoxDualArmReachingTaskBase(TabletopBoxTaskBase):
    config_provider: ClassVar[Type[CachedPR2ConstProvider]] = CachedDualArmPR2ConstProvider

    @classmethod
    def sample_target_poses(
        cls, world: TabletopBoxWorld, standard: bool
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


# fmt: off
class TabletopBoxRightArmReachingTask(ExactGridSDFCreator, TabletopBoxRightArmReachingTaskBase): ...  # noqa
class TabletopBoxVoxbloxRightArmReachingTask(VoxbloxGridSDFCreator, TabletopBoxRightArmReachingTaskBase): ...  # noqa
class TabletopBoxDualArmReachingTask(ExactGridSDFCreator, TabletopBoxDualArmReachingTaskBase): ...  # noqa
class TabletopBoxVoxbloxDualArmReachingTask(VoxbloxGridSDFCreator, TabletopBoxDualArmReachingTaskBase): ...  # noqa
class TabletopBoxWorldWrap(ExactGridSDFCreator, TabletopBoxWorldWrapBase): ...  # noqa
class TabletopVoxbloxBoxWorldWrap(VoxbloxGridSDFCreator, TabletopBoxWorldWrapBase): ...  # noqa
# fmt: on


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
